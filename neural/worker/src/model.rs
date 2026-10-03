use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use switchify_prediction_neural::bundle::Bundle;
use tokenizers::Tokenizer;

struct Prepared {
    session: u64,
    ids: Vec<u32>,
    state: ModelWeights,
    logits: Vec<f32>,
}
pub struct Model {
    empty: ModelWeights,
    tokenizer: Tokenizer,
    boundaries: Vec<usize>,
    prepared: Option<Prepared>,
}
fn forward(state: &mut ModelWeights, ids: &[u32], position: usize) -> Result<Vec<f32>> {
    Ok(state
        .forward(&Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?, position)?
        .squeeze(0)?
        .to_vec1()?)
}
fn log_prob(logits: &[f32], token: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    logits[token] as f64
        - max
        - logits
            .iter()
            .map(|x| (*x as f64 - max).exp())
            .sum::<f64>()
            .ln()
}
fn log_boundary(logits: &[f32], ids: &[usize]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let all = logits.iter().map(|x| (*x as f64 - max).exp()).sum::<f64>();
    (ids.iter()
        .map(|&i| (logits[i] as f64 - max).exp())
        .sum::<f64>()
        / all)
        .ln()
}
impl Model {
    pub fn load(bundle: Bundle) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(&bundle.tokenizer).map_err(anyhow::Error::msg)?;
        ensure!(
            tokenizer.get_vocab_size(true) == 49152
                && tokenizer.token_to_id("<|endoftext|>") == Some(0),
            "tokenizer"
        );
        let mut reader = std::io::Cursor::new(bundle.weights);
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
        let empty = ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)?;
        let mut boundaries = vec![0];
        for id in 1..49152 {
            let text = tokenizer.decode(&[id], false).map_err(anyhow::Error::msg)?;
            if text.chars().next().is_some_and(|c| {
                c.is_whitespace() || c.is_numeric() || ".!?;:,()[]{}\"-/".contains(c)
            }) {
                boundaries.push(id as usize);
            }
        }
        Ok(Self {
            empty,
            tokenizer,
            boundaries,
            prepared: None,
        })
    }
    pub fn reset(&mut self) {
        self.prepared = None;
    }
    pub fn rank(
        &mut self,
        session: u64,
        before: &str,
        candidates: &[String],
        limit: usize,
    ) -> Result<(Vec<String>, bool)> {
        let encoded = self
            .tokenizer
            .encode(before, false)
            .map_err(anyhow::Error::msg)?;
        let tokens = encoded.get_ids();
        let mut ids = vec![0];
        ids.extend_from_slice(&tokens[tokens.len().saturating_sub(63)..]);
        let cache_hit = self
            .prepared
            .as_ref()
            .is_some_and(|p| p.session == session && p.ids == ids);
        if !cache_hit {
            self.reset();
            let mut state = self.empty.clone();
            let logits = forward(&mut state, &ids, 0)?;
            self.prepared = Some(Prepared {
                session,
                ids,
                state,
                logits,
            });
        }
        let prepared = self.prepared.as_ref().unwrap();
        let mut scored = Vec::new();
        for (index, word) in candidates.iter().enumerate() {
            let text = if before.is_empty() {
                word.clone()
            } else {
                format!(" {word}")
            };
            let encoded = self
                .tokenizer
                .encode(text, false)
                .map_err(anyhow::Error::msg)?;
            ensure!(!encoded.get_ids().is_empty(), "candidate");
            let mut state = prepared.state.clone();
            let mut logits = prepared.logits.clone();
            let mut score = 0.;
            for (offset, &token) in encoded.get_ids().iter().enumerate() {
                score += log_prob(&logits, token as usize);
                logits = forward(&mut state, &[token], prepared.ids.len() + offset)?;
            }
            score += log_boundary(&logits, &self.boundaries);
            ensure!(score.is_finite(), "score");
            scored.push((score, index, word.clone()));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok((
            scored.into_iter().take(limit).map(|x| x.2).collect(),
            cache_hit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundary_probability_is_part_of_whole_word_score() {
        assert!((log_prob(&[0., 0.], 0) + 2_f64.ln()).abs() < 1e-12);
        assert!((log_boundary(&[0., 0.], &[1]) + 2_f64.ln()).abs() < 1e-12);
        assert_eq!(log_boundary(&[0., 0.], &[0, 1]), 0.);
    }

    #[test]
    #[ignore = "requires explicit SWITCHIFY_SMOL_BUNDLE; never downloads model assets"]
    fn pinned_model_order_and_session_cache() {
        let path =
            std::env::var_os("SWITCHIFY_SMOL_BUNDLE").expect("explicit model bundle required");
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        let candidates: Vec<_> = [
            "receipt",
            "order",
            "tickets",
            "confirmation",
            "car",
            "details",
            "directions",
            "address",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let (first, hit) = model.rank(1, "please send the", &candidates, 5).unwrap();
        assert!(!hit);
        // Q8 kernels have ISA-dependent rounding; freeze each supported reference
        // build separately rather than asserting cross-kernel bitwise parity.
        let expected = if cfg!(feature = "accelerated") {
            ["address", "details", "order", "receipt", "directions"]
        } else {
            ["address", "details", "order", "directions", "car"]
        };
        assert_eq!(first, expected);
        let (second, hit) = model.rank(1, "please send the", &candidates, 5).unwrap();
        assert!(hit);
        assert_eq!(first, second);
        assert!(!model.rank(2, "please send the", &candidates, 5).unwrap().1);
        model.reset();
        assert!(!model.rank(2, "please send the", &candidates, 5).unwrap().1);
    }
}
