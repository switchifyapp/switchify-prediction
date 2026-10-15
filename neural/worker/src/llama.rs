//! Llama forward pass for the pinned SmolLM2 GGUF with a selectable batch cache.
//!
//! Adapted from candle-transformers 0.11.0 `models::quantized_llama` (MIT OR
//! Apache-2.0), keeping its operation order so a single row computes the same
//! values. Only dense Llama layers with interleaved RoPE are supported. Rows of
//! the attention cache can be selected, so beam extensions sharing a length run
//! as one batched forward pass.
use anyhow::{Result, ensure};
use candle_core::{
    DType, Device, IndexOp, Module, Tensor,
    quantized::{QMatMul, gguf_file},
};
use candle_nn::Embedding;
use candle_transformers::{quantized_nn::RmsNorm, utils};

const MAX_SEQ_LEN: usize = 4096;

#[derive(Clone)]
struct Layer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    attention_norm: RmsNorm,
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
    ffn_norm: RmsNorm,
    kv_cache: Option<(Tensor, Tensor)>,
}

/// Rotary angles for every position, `(MAX_SEQ_LEN, rope_dim / 2)`.
fn angles(rope_dim: usize, freq_base: f32) -> Result<Tensor> {
    let device = &Device::Cpu;
    let theta: Vec<_> = (0..rope_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / rope_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    Ok(Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?)
}

#[derive(Clone)]
pub struct Llama {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
}

impl Llama {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
    ) -> Result<Self> {
        let device = &Device::Cpu;
        let get = |key: &str| {
            ct.metadata
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("metadata"))
        };
        ensure!(
            get("general.architecture")?.to_string()? == "llama"
                && !ct.metadata.contains_key("llama.expert_count"),
            "architecture"
        );
        let n_head = get("llama.attention.head_count")?.to_u32()? as usize;
        let n_kv_head = get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let blocks = get("llama.block_count")?.to_u32()? as usize;
        let embedding = get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = get("llama.rope.dimension_count")?.to_u32()? as usize;
        let eps = get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let freq_base = get("llama.rope.freq_base")
            .and_then(|m| Ok(m.to_f32()?))
            .unwrap_or(10000.);
        let angles = angles(rope_dim, freq_base)?;
        let embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let embeddings = embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, eps)?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => embeddings_q,
        };
        let mut layers = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let mut weight =
                |name: &str| ct.tensor(reader, &format!("blk.{i}.{name}.weight"), device);
            layers.push(Layer {
                wq: QMatMul::from_qtensor(weight("attn_q")?)?,
                wk: QMatMul::from_qtensor(weight("attn_k")?)?,
                wv: QMatMul::from_qtensor(weight("attn_v")?)?,
                wo: QMatMul::from_qtensor(weight("attn_output")?)?,
                w1: QMatMul::from_qtensor(weight("ffn_gate")?)?,
                w2: QMatMul::from_qtensor(weight("ffn_down")?)?,
                w3: QMatMul::from_qtensor(weight("ffn_up")?)?,
                attention_norm: RmsNorm::from_qtensor(weight("attn_norm")?, eps)?,
                ffn_norm: RmsNorm::from_qtensor(weight("ffn_norm")?, eps)?,
                kv_cache: None,
            });
        }
        Ok(Self {
            embeddings: Embedding::new(embeddings, embedding),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            n_head,
            n_kv_head,
            head_dim: embedding / n_head,
            cos: angles.cos()?,
            sin: angles.sin()?,
            neg_inf: Tensor::new(f32::NEG_INFINITY, device)?,
        })
    }

    /// Keep cache rows `rows`, in that order, possibly repeating rows.
    pub fn select(&mut self, rows: &[u32]) -> Result<()> {
        let rows = Tensor::new(rows, &Device::Cpu)?;
        for layer in &mut self.layers {
            if let Some((k, v)) = &layer.kv_cache {
                layer.kv_cache = Some((k.index_select(&rows, 0)?, v.index_select(&rows, 0)?));
            }
        }
        Ok(())
    }

    /// Keep the first `len` cached positions, so the next forward pass
    /// continues at position `len`.
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        for layer in &mut self.layers {
            if let Some((k, v)) = &layer.kv_cache {
                ensure!(len <= k.dim(2)?, "cache");
                layer.kv_cache = Some((k.narrow(2, 0, len)?, v.narrow(2, 0, len)?));
            }
        }
        Ok(())
    }

    /// `ids` is `(batch, seq)`. Returns the last position's logits per row.
    pub fn forward(&mut self, ids: &Tensor, index_pos: usize) -> Result<Vec<Vec<f32>>> {
        let (_, seq_len) = ids.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(utils::build_causal_mask(seq_len, index_pos, &Device::Cpu)?)
        };
        let mut x = self.embeddings.forward(ids)?;
        for i in 0..self.layers.len() {
            let residual = &x;
            let h = self.layers[i].attention_norm.forward(&x)?;
            let attn = self.attention(i, &h, mask.as_ref(), index_pos)?;
            let h = (attn + residual)?;
            let layer = &self.layers[i];
            let residual = &h;
            let m = layer.ffn_norm.forward(&h)?;
            let gate = layer.w1.forward(&m)?;
            let up = layer.w3.forward(&m)?;
            let m = layer.w2.forward(&(candle_nn::ops::silu(&gate)? * up)?)?;
            x = (m + residual)?;
        }
        let x = self.norm.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        Ok(self.output.forward(&x)?.to_vec2()?)
    }

    fn rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        Ok(candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)?)
    }

    fn attention(
        &mut self,
        i: usize,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, n_embd) = x.dims3()?;
        let (n_head, n_kv_head, head_dim) = (self.n_head, self.n_kv_head, self.head_dim);
        let layer = &self.layers[i];
        let q = layer
            .wq
            .forward(x)?
            .reshape((b, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let k = layer
            .wk
            .forward(x)?
            .reshape((b, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        let v = layer
            .wv
            .forward(x)?
            .reshape((b, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.rope(&q, index_pos)?;
        let k = self.rope(&k, index_pos)?;
        let layer = &mut self.layers[i];
        let (k, v) = match &layer.kv_cache {
            Some((k_cache, v_cache)) if index_pos != 0 => (
                Tensor::cat(&[k_cache, &k], 2)?,
                Tensor::cat(&[v_cache, &v], 2)?,
            ),
            _ => (k, v),
        };
        layer.kv_cache = Some((k.clone(), v.clone()));
        let k = utils::repeat_kv(k, n_head / n_kv_head)?;
        let v = utils::repeat_kv(v, n_head / n_kv_head)?;
        let att = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                mask.where_cond(&self.neg_inf.broadcast_as(att.shape().dims())?, &att)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b, seq_len, n_embd])?;
        Ok(layer.wo.forward(&y)?)
    }
}

#[cfg(test)]
impl Llama {
    /// A small random model with the pinned model's layout: Q8 weights, F32
    /// norms, grouped-query attention and tied embeddings.
    pub fn tiny(vocab: usize) -> Result<Self> {
        use candle_core::quantized::{GgmlDType, QTensor};
        let device = &Device::Cpu;
        let (embedding, n_head, n_kv_head, hidden) = (64, 2, 1, 96);
        let head_dim = embedding / n_head;
        let q8 = |shape: (usize, usize)| -> Result<QMatMul> {
            let weights = Tensor::randn(0f32, 0.3, shape, device)?;
            Ok(QMatMul::from_qtensor(QTensor::quantize(
                &weights,
                GgmlDType::Q8_0,
            )?)?)
        };
        let norm = || -> Result<RmsNorm> {
            let weights = Tensor::ones(embedding, DType::F32, device)?;
            Ok(RmsNorm::from_qtensor(
                QTensor::quantize(&weights, GgmlDType::F32)?,
                1e-5,
            )?)
        };
        let layers = (0..2)
            .map(|_| {
                Ok(Layer {
                    wq: q8((embedding, embedding))?,
                    wk: q8((n_kv_head * head_dim, embedding))?,
                    wv: q8((n_kv_head * head_dim, embedding))?,
                    wo: q8((embedding, embedding))?,
                    w1: q8((hidden, embedding))?,
                    w2: q8((embedding, hidden))?,
                    w3: q8((hidden, embedding))?,
                    attention_norm: norm()?,
                    ffn_norm: norm()?,
                    kv_cache: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let embeddings = Tensor::randn(0f32, 1., (vocab, embedding), device)?;
        let angles = angles(head_dim, 10000.)?;
        Ok(Self {
            output: QMatMul::from_qtensor(QTensor::quantize(&embeddings, GgmlDType::Q8_0)?)?,
            embeddings: Embedding::new(embeddings, embedding),
            layers,
            norm: norm()?,
            n_head,
            n_kv_head,
            head_dim,
            cos: angles.cos()?,
            sin: angles.sin()?,
            neg_inf: Tensor::new(f32::NEG_INFINITY, device)?,
        })
    }
}
