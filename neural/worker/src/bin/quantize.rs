//! Local conversion of the pinned SmolLM2 weights. No downloaded GGUF is trusted.
use anyhow::{Result, ensure};
use candle_core::{
    DType, Device, Tensor,
    quantized::{
        GgmlDType, QTensor,
        gguf_file::{self, Value},
    },
};
use candle_transformers::models::llama::LlamaConfig;
use std::{collections::BTreeMap, fs, path::PathBuf};

fn interleave_rope(tensor: Tensor, heads: usize) -> Result<Tensor> {
    let (rows, cols) = tensor.dims2()?;
    ensure!(heads > 0 && rows % (heads * 2) == 0, "Invalid rotary shape");
    Ok(tensor
        .reshape((heads, 2, rows / heads / 2, cols))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, cols))?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(args.len() == 2, "Usage: quantize MODEL_DIR OUTPUT.gguf");
    let root = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    ensure!(!output.exists(), "Output already exists");
    let cfg: LlamaConfig = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
    ensure!(
        cfg.hidden_size == 576
            && cfg.num_hidden_layers == 30
            && cfg.tie_word_embeddings == Some(true)
            && cfg.rope_scaling.is_none(),
        "Expected pinned SmolLM2 configuration"
    );
    let source = candle_core::safetensors::load(root.join("model.safetensors"), &Device::Cpu)?;
    let mut tensors = BTreeMap::new();
    let mut add = |target: String, name: &str, heads: Option<usize>| -> Result<()> {
        let mut tensor: Tensor = source
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Missing {name}"))?
            .to_dtype(DType::F32)?;
        // HF RoPE pairs the two halves; GGUF llama pairs adjacent coordinates.
        if let Some(heads) = heads {
            tensor = interleave_rope(tensor, heads)?;
        }
        let kind = if tensor.rank() == 2 {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        tensors.insert(target, QTensor::quantize(&tensor, kind)?);
        Ok(())
    };
    add(
        "token_embd.weight".into(),
        "model.embed_tokens.weight",
        None,
    )?;
    add("output_norm.weight".into(), "model.norm.weight", None)?;
    for i in 0..cfg.num_hidden_layers {
        for (target, name, heads) in [
            ("attn_q", "self_attn.q_proj", Some(cfg.num_attention_heads)),
            (
                "attn_k",
                "self_attn.k_proj",
                Some(cfg.num_key_value_heads()),
            ),
            ("attn_v", "self_attn.v_proj", None),
            ("attn_output", "self_attn.o_proj", None),
            ("ffn_gate", "mlp.gate_proj", None),
            ("ffn_down", "mlp.down_proj", None),
            ("ffn_up", "mlp.up_proj", None),
            ("attn_norm", "input_layernorm", None),
            ("ffn_norm", "post_attention_layernorm", None),
        ] {
            add(
                format!("blk.{i}.{target}.weight"),
                &format!("model.layers.{i}.{name}.weight"),
                heads,
            )?;
        }
    }
    let metadata = [
        ("general.architecture", Value::String("llama".into())),
        (
            "llama.attention.head_count",
            Value::U32(cfg.num_attention_heads as u32),
        ),
        (
            "llama.attention.head_count_kv",
            Value::U32(cfg.num_key_value_heads() as u32),
        ),
        (
            "llama.block_count",
            Value::U32(cfg.num_hidden_layers as u32),
        ),
        ("llama.embedding_length", Value::U32(cfg.hidden_size as u32)),
        (
            "llama.rope.dimension_count",
            Value::U32((cfg.hidden_size / cfg.num_attention_heads) as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F32(cfg.rms_norm_eps as f32),
        ),
        ("llama.rope.freq_base", Value::F32(cfg.rope_theta)),
    ];
    let metadata: Vec<_> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensors: Vec<_> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    gguf_file::write(&mut fs::File::create_new(output)?, &metadata, &tensors)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotary_permutation_stays_within_each_attention_head() {
        let tensor = Tensor::arange(0_f32, 16_f32, &Device::Cpu)
            .unwrap()
            .reshape((8, 2))
            .unwrap();
        let result = interleave_rope(tensor.clone(), 2)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            result.iter().map(|row| row[0]).collect::<Vec<_>>(),
            [0., 4., 2., 6., 8., 12., 10., 14.]
        );
        assert!(interleave_rope(tensor, 3).is_err());
    }
}
