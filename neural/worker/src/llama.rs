//! Llama forward pass for the pinned SmolLM2 GGUF with a selectable batch cache.
//!
//! Adapted from candle-transformers 0.11.0 `models::quantized_llama` (MIT OR
//! Apache-2.0), keeping its operation order so a single row computes the same
//! values. Only dense Llama layers with interleaved RoPE are supported. Rows of
//! the attention cache can be selected, so beam extensions sharing a length run
//! as one batched forward pass.
//!
//! The experimental `metal` feature runs the same operations on an Apple GPU.
//! It is untested on hardware, and like the explicit ISA builds it is not
//! expected to match the CPU kernels bitwise.
use anyhow::{Result, ensure};
use candle_core::{
    DType, Device, IndexOp, Module, Tensor,
    quantized::{QMatMul, gguf_file},
};
use candle_nn::Embedding;
use candle_transformers::{quantized_nn::RmsNorm, utils};

const MAX_SEQ_LEN: usize = 4096;

/// The CPU, or the first Metal GPU in the experimental `metal` build.
fn device() -> Result<Device> {
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    Ok(device)
}

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
    device: Device,
}

impl Llama {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
    ) -> Result<Self> {
        let device = &device()?;
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
        let theta: Vec<_> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / freq_base.powf(i as f32 / rope_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let angles = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
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
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Keep cache rows `rows`, in that order, possibly repeating rows.
    pub fn select(&mut self, rows: &[u32]) -> Result<()> {
        let rows = Tensor::new(rows, &self.device)?;
        for layer in &mut self.layers {
            if let Some((k, v)) = &layer.kv_cache {
                layer.kv_cache = Some((k.index_select(&rows, 0)?, v.index_select(&rows, 0)?));
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
            Some(utils::build_causal_mask(seq_len, index_pos, &self.device)?)
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
