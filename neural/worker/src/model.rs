use crate::llama::Llama;
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use switchify_prediction_neural::bundle::Bundle;
use tokenizers::Tokenizer;

struct Prepared {
    session: u64,
    ids: Vec<u32>,
    state: Llama,
    logits: Vec<f32>,
}
pub struct Model {
    empty: Llama,
    tokenizer: Tokenizer,
    boundaries: Vec<usize>,
    pieces: Vec<Vec<u8>>,
    prepared: Option<Prepared>,
}
fn forward(state: &mut Llama, ids: &[u32], position: usize) -> Result<Vec<f32>> {
    let mut rows = state.forward(&Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?, position)?;
    ensure!(rows.len() == 1, "logits");
    Ok(rows.remove(0))
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
/// The bonus, in log probability, for a preferred word: one that has been
/// learned from this person's typing.
const PREFERENCE: f64 = std::f64::consts::LN_2 * 2.;

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        return high;
    }
    high + ((a - high).exp() + (b - high).exp()).ln()
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
        let empty = Llama::from_gguf(content, &mut reader)?;
        let mut boundaries = vec![0];
        for id in 1..49152 {
            let text = tokenizer.decode(&[id], false).map_err(anyhow::Error::msg)?;
            if text.chars().next().is_some_and(|c| {
                c.is_whitespace() || c.is_numeric() || ".!?;:,()[]{}\"-/".contains(c)
            }) {
                boundaries.push(id as usize);
            }
        }
        let pieces = crate::generation::pieces(&tokenizer)?;
        Ok(Self {
            pieces,
            empty,
            tokenizer,
            boundaries,
            prepared: None,
        })
    }
    pub fn reset(&mut self) {
        self.prepared = None;
    }
    fn prepare(&mut self, session: u64, before: &str) -> Result<bool> {
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
        Ok(cache_hit)
    }

    pub fn generate(
        &mut self,
        query: &switchify_prediction_neural::protocol::GenerationQuery,
    ) -> Result<(Vec<String>, bool)> {
        let started = std::time::Instant::now();
        let hit = self.prepare(query.session, &query.before)?;
        let prefix = switchify_prediction::normalize(&query.prefix);
        let preferred: Vec<String> = query
            .prefer
            .iter()
            .filter(|w| w.starts_with(&prefix) && query.accepts(std::slice::from_ref(w)))
            .cloned()
            .collect();
        // Scored first, so the search's time limit includes them.
        let scores = self.score_words(!query.before.is_empty(), &preferred)?;
        let prepared = self.prepared.as_ref().unwrap();
        // Rows of `state` are the current beams; the context cache has one row.
        let mut state = prepared.state.clone();
        let generated = crate::generation::search(
            query,
            &self.pieces,
            prepared.logits.clone(),
            usize::from(!hit),
            started,
            |parents, tokens, depth| {
                state.select(parents)?;
                let ids = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(1)?;
                state.forward(&ids, prepared.ids.len() + depth)
            },
        )?;
        if preferred.is_empty() {
            return Ok((generated.into_iter().map(|(w, _)| w).collect(), hit));
        }
        // A preferred word takes the better of its generated and scored
        // probability, plus the bonus.
        let mut ranked: Vec<(String, f64)> = generated
            .into_iter()
            .filter(|(w, _)| !preferred.contains(w))
            .collect();
        for (word, score) in preferred.iter().zip(scores) {
            ranked.push((word.clone(), score + PREFERENCE));
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(query.limit);
        Ok((ranked.into_iter().map(|(w, _)| w).collect(), hit))
    }

    /// Log probability of each word following the prepared context, with a
    /// following word boundary, over its lowercase and capitalised forms.
    /// Every form advances one token per batched forward pass.
    fn score_words(&mut self, separated: bool, words: &[String]) -> Result<Vec<f64>> {
        let mut forms: Vec<(usize, Vec<u32>)> = Vec::new();
        for (index, word) in words.iter().enumerate() {
            let mut chars = word.chars();
            let capital: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            let mut variants = vec![word.clone()];
            if capital != *word {
                variants.push(capital);
            }
            for variant in variants {
                let text = if separated {
                    format!(" {variant}")
                } else {
                    variant
                };
                let encoded = self
                    .tokenizer
                    .encode(text, false)
                    .map_err(anyhow::Error::msg)?;
                ensure!(!encoded.get_ids().is_empty(), "candidate");
                forms.push((index, encoded.get_ids().to_vec()));
            }
        }
        let mut result = vec![f64::NEG_INFINITY; words.len()];
        let Some(prepared) = self.prepared.as_ref().filter(|_| !forms.is_empty()) else {
            return Ok(result);
        };
        let mut scores = vec![0f64; forms.len()];
        let mut logits = vec![prepared.logits.clone(); forms.len()];
        let mut state = prepared.state.clone();
        let mut active: Vec<usize> = (0..forms.len()).collect();
        let mut rows: Vec<u32> = vec![0; forms.len()];
        for step in 0.. {
            for &form in &active {
                scores[form] += log_prob(&logits[form], forms[form].1[step] as usize);
            }
            state.select(&rows)?;
            let tokens: Vec<u32> = active.iter().map(|&f| forms[f].1[step]).collect();
            let ids = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(1)?;
            let out = state.forward(&ids, prepared.ids.len() + step)?;
            let (mut next, mut next_rows) = (Vec::new(), Vec::new());
            for (row, (&form, row_logits)) in active.iter().zip(out).enumerate() {
                if forms[form].1.len() == step + 1 {
                    scores[form] += log_boundary(&row_logits, &self.boundaries);
                } else {
                    next.push(form);
                    next_rows.push(row as u32);
                }
                logits[form] = row_logits;
            }
            if next.is_empty() {
                break;
            }
            (active, rows) = (next, next_rows);
        }
        for ((index, _), score) in forms.iter().zip(scores) {
            ensure!(score.is_finite(), "score");
            result[*index] = log_sum_exp(result[*index], score);
        }
        Ok(result)
    }

    pub fn rank(
        &mut self,
        session: u64,
        before: &str,
        candidates: &[String],
        limit: usize,
    ) -> Result<(Vec<String>, bool)> {
        let cache_hit = self.prepare(session, before)?;
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

    #[test]
    #[ignore = "requires explicit SWITCHIFY_SMOL_BUNDLE; never downloads model assets"]
    fn batched_rows_match_single_row_forwards() {
        let path =
            std::env::var_os("SWITCHIFY_SMOL_BUNDLE").expect("explicit model bundle required");
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        for (before, prefix) in [("please send the", ""), ("I would like a", "c"), ("", "")] {
            let query = switchify_prediction_neural::protocol::GenerationQuery {
                id: 1,
                session: 1,
                before: before.into(),
                prefix: prefix.into(),
                exclude: vec![],
                limit: 3,
                prefer: vec![],
            };
            model.prepare(1, before).unwrap();
            let prepared = model.prepared.as_ref().unwrap();
            let position = prepared.ids.len();
            let mut batched = prepared.state.clone();
            let mut single = vec![prepared.state.clone()];
            let mut compared = 0;
            crate::generation::search(
                &query,
                &model.pieces,
                prepared.logits.clone(),
                1,
                std::time::Instant::now(),
                |parents, tokens, depth| {
                    batched.select(parents)?;
                    let ids = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(1)?;
                    let rows = batched.forward(&ids, position + depth)?;
                    let mut next = Vec::new();
                    for ((&parent, &token), row) in parents.iter().zip(tokens).zip(&rows) {
                        let mut state = single[parent as usize].clone();
                        let expected = forward(&mut state, &[token], position + depth)?;
                        let difference = expected
                            .iter()
                            .zip(row)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        ensure!(difference <= 1e-3, "batched row differs by {difference}");
                        compared += 1;
                        next.push(state);
                    }
                    single = next;
                    Ok(rows)
                },
            )
            .unwrap();
            assert!(compared >= 8);
        }
    }
}
