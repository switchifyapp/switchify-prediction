use crate::llama::Llama;
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use switchify_prediction_neural::bundle::Bundle;
use tokenizers::Tokenizer;

const RANKED_WORDS: usize = 64;

struct Prepared {
    session: u64,
    ids: Vec<u32>,
    state: Llama,
    logits: Vec<f32>,
    /// Whole-word scores for this exact prepared context, FIFO-bounded.
    ranked: std::collections::VecDeque<(Vec<u32>, f64)>,
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
        // All weights are now owned by the model. Release the source file's
        // 136 MiB before constructing the tokenizer lookup tables.
        drop(reader);
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
                ranked: std::collections::VecDeque::new(),
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
        let prepared = self.prepared.as_ref().unwrap();
        // Rows of `state` are the current beams; the context cache has one row.
        let mut state = prepared.state.clone();
        let words = crate::generation::search(
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
        Ok((words, hit))
    }

    pub fn generate_phrases(
        &mut self,
        query: &switchify_prediction_neural::protocol::PhraseQuery,
    ) -> Result<(Vec<Vec<String>>, bool)> {
        let started = std::time::Instant::now();
        let hit = self.prepare(query.session, &query.before)?;
        let prepared = self.prepared.as_ref().unwrap();
        let mut state = prepared.state.clone();
        let phrases = crate::generation::search_phrases(
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
        Ok((phrases, hit))
    }

    pub fn rank(
        &mut self,
        session: u64,
        before: &str,
        candidates: &[String],
        limit: usize,
    ) -> Result<(Vec<String>, bool)> {
        let hit = self.prepare(session, before)?;
        let prepared = self.prepared.as_mut().unwrap();
        let encoded: Vec<Vec<u32>> = candidates
            .iter()
            .map(|word| {
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
                Ok(encoded.get_ids().to_vec())
            })
            .collect::<Result<_>>()?;
        let mut state = prepared.state.clone();
        let mut logits = vec![prepared.logits.clone()];
        let mut scores = vec![0.; candidates.len()];
        let mut active = Vec::new();
        for (i, ids) in encoded.iter().enumerate() {
            if let Some((_, score)) = prepared.ranked.iter().find(|(key, _)| key == ids) {
                scores[i] = *score;
            } else {
                active.push((i, 0));
            }
        }
        let mut depth = 0;
        while !active.is_empty() {
            let parents: Vec<u32> = active.iter().map(|&(_, row)| row as u32).collect();
            let tokens: Vec<u32> = active.iter().map(|&(i, _)| encoded[i][depth]).collect();
            for (&(i, row), &token) in active.iter().zip(&tokens) {
                scores[i] += log_prob(&logits[row], token as usize);
            }
            state.select(&parents)?;
            let ids = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(1)?;
            logits = state.forward(&ids, prepared.ids.len() + depth)?;
            let mut next = Vec::new();
            for (row, &(i, _)) in active.iter().enumerate() {
                if encoded[i].len() == depth + 1 {
                    scores[i] += log_boundary(&logits[row], &self.boundaries);
                    ensure!(scores[i].is_finite(), "score");
                } else {
                    next.push((i, row));
                }
            }
            active = next;
            depth += 1;
        }
        for (ids, &score) in encoded.into_iter().zip(&scores) {
            if !prepared.ranked.iter().any(|(key, _)| key == &ids) {
                if prepared.ranked.len() == RANKED_WORDS {
                    prepared.ranked.pop_front();
                }
                prepared.ranked.push_back((ids, score));
            }
        }
        let mut order: Vec<usize> = (0..candidates.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        Ok((
            order
                .into_iter()
                .take(limit)
                .map(|i| candidates[i].clone())
                .collect(),
            hit,
        ))
    }

    #[cfg(test)]
    fn rank_sequential(
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
    #[ignore = "requires explicit SWITCHIFY_SMOL_BUNDLE; bounded score cache"]
    fn ranked_scores_are_bounded_and_reset_with_context() {
        let path = std::env::var_os("SWITCHIFY_SMOL_BUNDLE").unwrap();
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        model.prepare(1, "please").unwrap();
        model.prepared.as_mut().unwrap().ranked = (0..RANKED_WORDS)
            .map(|i| (vec![u32::MAX - i as u32], 0.))
            .collect();
        model.rank(1, "please", &["order".into()], 1).unwrap();
        let ranked = &model.prepared.as_ref().unwrap().ranked;
        assert_eq!(ranked.len(), RANKED_WORDS);
        assert_eq!(ranked.front().unwrap().0, vec![u32::MAX - 1]);
        model.prepare(2, "please").unwrap();
        assert!(model.prepared.as_ref().unwrap().ranked.is_empty());
        model.rank(2, "please", &["order".into()], 1).unwrap();
        model.prepare(2, "please send").unwrap();
        assert!(model.prepared.as_ref().unwrap().ranked.is_empty());
        model.reset();
        assert!(model.prepared.is_none());
    }

    #[test]
    #[ignore = "requires explicit SWITCHIFY_SMOL_BUNDLE; reranking parity"]
    fn batched_reranking_matches_sequential_reference() {
        let path = std::env::var_os("SWITCHIFY_SMOL_BUNDLE").unwrap();
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        let sets = [
            vec![
                "receipt",
                "order",
                "tickets",
                "confirmation",
                "car",
                "details",
                "directions",
                "address",
            ],
            vec![
                "in",
                "information",
                "informational",
                "inform",
                "international",
                "interview",
                "can't",
                "café",
            ],
            vec![
                "a",
                "a",
                "physiotherapy",
                "wheelchair",
                "accessibility",
                "water",
                "watermelon",
                "waterproof",
            ],
            vec![],
        ];
        for before in [
            "",
            "please send the",
            "I would like a",
            "I have been thinking about our plans for the weekend and would like to",
        ] {
            for words in &sets {
                let candidates: Vec<String> = words.iter().map(|s| s.to_string()).collect();
                let expected = model.rank_sequential(1, before, &candidates, 5).unwrap().0;
                let actual = model.rank(1, before, &candidates, 5).unwrap().0;
                assert_eq!(
                    actual, expected,
                    "context {before:?}, candidates {candidates:?}"
                );
                let mut overlap = candidates.clone();
                overlap.reverse();
                overlap.truncate(6);
                overlap.push("independently".into());
                let expected = model.rank_sequential(1, before, &overlap, 5).unwrap().0;
                assert_eq!(model.rank(1, before, &overlap, 5).unwrap().0, expected);
            }
        }
    }

    #[test]
    #[ignore = "requires explicit SWITCHIFY_SMOL_BUNDLE; local inference profile"]
    fn profile_forward_shapes() {
        let path = std::env::var_os("SWITCHIFY_SMOL_BUNDLE").unwrap();
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        for before in [
            "",
            "Please send the",
            "I have been thinking about our plans for the weekend and would like to",
        ] {
            model.prepare(1, before).unwrap();
            let prepared = model.prepared.as_ref().unwrap();
            for batch in [1, 8] {
                let started = std::time::Instant::now();
                for _ in 0..5 {
                    let mut state = prepared.state.clone();
                    state.select(&vec![0; batch]).unwrap();
                    let ids =
                        Tensor::from_vec(vec![42u32; batch], (batch, 1), &Device::Cpu).unwrap();
                    std::hint::black_box(state.forward(&ids, prepared.ids.len()).unwrap());
                }
                println!(
                    "context_tokens={} batch={batch} forward_ms={:.2}",
                    prepared.ids.len(),
                    started.elapsed().as_secs_f64() * 200.
                );
            }
        }
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
