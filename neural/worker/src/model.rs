use crate::llama::Llama;
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use switchify_prediction_neural::{CONTEXT_TOKENS, bundle::Bundle};
use tokenizers::Tokenizer;

/// When the context outgrows the window, its start moves to a later sentence,
/// or within a long sentence by a multiple of this many tokens. Consecutive
/// requests then keep the same start, so the prepared state can be extended.
const WINDOW_STEP: usize = 32;
/// Context positions evaluated per forward pass. Quantized evaluation costs
/// nearly the same per token in groups of this size as in one pass, and
/// extending a state evaluates fewer than this many tokens again.
const CHUNK: usize = 16;

/// A single-row model state that can be extended and cut back.
trait Context: Clone {
    /// Evaluate `ids` from `position`, returning the last position's logits.
    fn forward(&mut self, ids: &[u32], position: usize) -> Result<Vec<f32>>;
    /// Keep the first `len` positions.
    fn truncate(&mut self, len: usize) -> Result<()>;
}
impl Context for Llama {
    fn forward(&mut self, ids: &[u32], position: usize) -> Result<Vec<f32>> {
        forward(self, ids, position)
    }
    fn truncate(&mut self, len: usize) -> Result<()> {
        Llama::truncate(self, len)
    }
}

struct Prepared<S = Llama> {
    session: u64,
    /// The end-of-text token, then the context window. Position `i` of
    /// `state` holds `ids[i]`.
    ids: Vec<u32>,
    state: S,
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

/// Where the context window starts in `tokens`. All of them fit after the
/// end-of-text token if they can. Otherwise the window starts at the earliest
/// sentence start that leaves at most `CONTEXT_TOKENS - 1` tokens, or, if a
/// long sentence stands in the way, a multiple of `WINDOW_STEP` tokens after the
/// start of that sentence. The start depends only on the text near it and at
/// the end, so it stays put while text is added or the oldest text is clipped.
fn window_start(len: usize, sentence_start: impl Fn(usize) -> bool) -> usize {
    let room = CONTEXT_TOKENS - 1;
    if len <= room {
        return 0;
    }
    let need = len - room;
    let base = (1..=need).rev().find(|&i| sentence_start(i)).unwrap_or(0);
    let stepped = base + (need - base).div_ceil(WINDOW_STEP) * WINDOW_STEP;
    (need..stepped)
        .find(|&i| sentence_start(i))
        .unwrap_or(stepped)
}

/// Prepare `ids` for `session`, reusing `cached` when it belongs to the same
/// session. Returns whether `ids` were already prepared.
///
/// Quantized evaluation rounds slightly differently depending on how positions
/// are grouped into forward passes. Positions are therefore always evaluated in
/// `CHUNK`-aligned groups, so an extended state holds exactly the values a
/// fresh one would. Only the chunks from the one holding the first changed
/// token, or the last token, are evaluated again.
fn advance<S: Context>(
    cached: Option<Prepared<S>>,
    empty: &S,
    session: u64,
    ids: Vec<u32>,
) -> Result<(Prepared<S>, bool)> {
    ensure!(!ids.is_empty(), "context");
    let cached = cached.filter(|p| p.session == session);
    if let Some(prepared) = cached.as_ref()
        && prepared.ids == ids
    {
        return Ok((cached.unwrap(), true));
    }
    let shared = cached.as_ref().map_or(0, |p| {
        p.ids.iter().zip(&ids).take_while(|(a, b)| a == b).count()
    });
    let keep = shared.min(ids.len() - 1) / CHUNK * CHUNK;
    let mut state = match cached {
        Some(prepared) if keep > 0 => {
            let mut state = prepared.state;
            state.truncate(keep)?;
            state
        }
        _ => empty.clone(),
    };
    let mut logits = Vec::new();
    for start in (keep..ids.len()).step_by(CHUNK) {
        let end = (start + CHUNK).min(ids.len());
        logits = state.forward(&ids[start..end], start)?;
    }
    Ok((
        Prepared {
            session,
            ids,
            state,
            logits,
        },
        false,
    ))
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
        // A token ending a sentence or line is followed by a sentence start.
        let start = window_start(tokens.len(), |i| {
            self.pieces[tokens[i - 1] as usize]
                .last()
                .is_some_and(|b| b".!?\n".contains(b))
        });
        let mut ids = vec![0];
        ids.extend_from_slice(&tokens[start..]);
        let (prepared, cache_hit) = advance(self.prepared.take(), &self.empty, session, ids)?;
        self.prepared = Some(prepared);
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
    use std::{cell::Cell, rc::Rc};

    /// Records the tokens it evaluates; logits identify the whole sequence.
    #[derive(Clone, Default)]
    struct Fake {
        sequence: Vec<u32>,
        evaluated: Rc<Cell<usize>>,
    }
    fn identify(sequence: &[u32]) -> Vec<f32> {
        let hash = sequence
            .iter()
            .fold(17u64, |h, &t| h.wrapping_mul(31).wrapping_add(u64::from(t)));
        vec![sequence.len() as f32, (hash % 1_000_003) as f32]
    }
    impl Context for Fake {
        fn forward(&mut self, ids: &[u32], position: usize) -> Result<Vec<f32>> {
            ensure!(position == self.sequence.len(), "position");
            ensure!(
                position % CHUNK == 0 && (1..=CHUNK).contains(&ids.len()),
                "chunk"
            );
            self.sequence.extend_from_slice(ids);
            self.evaluated.set(self.evaluated.get() + ids.len());
            Ok(identify(&self.sequence))
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            ensure!(len <= self.sequence.len(), "cache");
            self.sequence.truncate(len);
            Ok(())
        }
    }

    #[test]
    fn state_is_extended_from_the_longest_shared_prefix() {
        let empty = Fake::default();
        let counter = Rc::clone(&empty.evaluated);
        let mut prepared = None;
        let mut step = |session: u64, ids: &[u32]| {
            counter.set(0);
            let (next, hit) = advance(prepared.take(), &empty, session, ids.to_vec()).unwrap();
            assert_eq!(next.state.sequence, ids);
            assert_eq!(next.logits, identify(ids));
            prepared = Some(next);
            (hit, counter.get())
        };
        let text = |len: u32| (0..len).collect::<Vec<_>>();
        // Fresh, then an exact hit that evaluates nothing.
        assert_eq!(step(1, &text(40)), (false, 40));
        assert_eq!(step(1, &text(40)), (true, 0));
        // Typing on evaluates the new tokens and the rest of their chunk.
        assert_eq!(step(1, &text(43)), (false, 43 - 32));
        assert_eq!(step(1, &text(48)), (false, 48 - 32));
        assert_eq!(step(1, &text(50)), (false, 50 - 48));
        // A changed word is evaluated from its chunk.
        let mut changed = text(50);
        changed[20] = 99;
        assert_eq!(step(1, &changed), (false, 50 - 16));
        // Deleting evaluates the last remaining token's chunk for its logits.
        assert_eq!(step(1, &changed[..32]), (false, 32 - 16));
        assert_eq!(step(1, &changed[..31]), (false, 31 - 16));
        // A moved window shares only the end-of-text token.
        let moved: Vec<u32> = [0].into_iter().chain(20..60).collect();
        assert_eq!(step(1, &moved), (false, moved.len()));
        // Another session never reuses the state.
        assert_eq!(step(2, &moved), (false, moved.len()));
        assert_eq!(step(2, &[0]), (false, 1));
        assert_eq!(step(2, &[0]), (true, 0));
    }

    #[test]
    fn reused_state_matches_a_fresh_forward_pass() {
        let empty = Llama::tiny(40).unwrap();
        let ids: Vec<u32> = (0..50).map(|i| (i * 7 + 3) % 40).collect();
        let fresh = advance(None, &empty, 1, ids.clone()).unwrap().0;
        let mut diverged = ids[..36].to_vec();
        diverged.extend([1, 2, 3]);
        for earlier in [
            ids[..20].to_vec(),
            ids[..49].to_vec(),
            diverged,
            ids.clone(),
            [ids.clone(), vec![5, 6]].concat(),
        ] {
            let hit = earlier == ids;
            let cached = advance(None, &empty, 1, earlier).unwrap().0;
            let (reused, reused_hit) = advance(Some(cached), &empty, 1, ids.clone()).unwrap();
            assert_eq!(reused_hit, hit);
            assert_eq!(reused.logits, fresh.logits);
            // Search continues both states from the same position.
            let next = |prepared: &Prepared| {
                let mut state = prepared.state.clone();
                forward(&mut state, &[11], prepared.ids.len()).unwrap()
            };
            assert_eq!(next(&reused), next(&fresh));
        }
    }

    #[test]
    fn window_moves_by_sentences_or_large_steps() {
        let room = CONTEXT_TOKENS - 1;
        let none = |_: usize| false;
        assert_eq!(window_start(0, none), 0);
        assert_eq!(window_start(room, none), 0);
        // Without sentences the start moves 32 tokens at a time, keeping at
        // least 96 tokens and the same start while 32 more are added.
        for len in room + 1..=room + WINDOW_STEP {
            assert_eq!(window_start(len, none), WINDOW_STEP);
        }
        assert_eq!(window_start(room + WINDOW_STEP + 1, none), 2 * WINDOW_STEP);
        // Otherwise it starts at the earliest sentence that fits.
        let starts = [10, 40, 45, 200];
        let sentence = |i: usize| starts.contains(&i);
        assert_eq!(window_start(room + 1, sentence), 10);
        assert_eq!(window_start(room + 10, sentence), 10);
        assert_eq!(window_start(room + 11, sentence), 40);
        assert_eq!(window_start(room + 41, sentence), 45);
        // Within a long sentence, steps count from the sentence start.
        assert_eq!(window_start(room + 46, sentence), 45 + WINDOW_STEP);
        assert_eq!(
            window_start(room + 45 + WINDOW_STEP, sentence),
            45 + WINDOW_STEP
        );
        assert_eq!(
            window_start(room + 46 + WINDOW_STEP, sentence),
            45 + 2 * WINDOW_STEP
        );
        // Clipping the oldest tokens keeps a full window on the same text.
        for len in room + 1..room + 150 {
            for clipped in (0..=10).filter(|c| len - c > room) {
                let shifted = |i: usize| sentence(i + clipped);
                assert_eq!(
                    window_start(len - clipped, shifted) + clipped,
                    window_start(len, sentence),
                    "len {len} clipped {clipped}"
                );
            }
        }
        // The window never exceeds the context or falls below 96 tokens.
        for len in room + 1..room + 300 {
            let kept = len - window_start(len, sentence);
            assert!((room - WINDOW_STEP + 1..=room).contains(&kept), "{len}");
        }
    }

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
    fn pinned_model_windows_and_extends_earlier_sentences() {
        let path =
            std::env::var_os("SWITCHIFY_SMOL_BUNDLE").expect("explicit model bundle required");
        let bundle =
            switchify_prediction_neural::bundle::load(std::path::Path::new(&path)).unwrap();
        let mut model = Model::load(bundle).unwrap();
        let sentences = [
            "We packed the car before sunrise.",
            "The road to the coast was quiet and the kids slept most of the way.",
            "When we arrived the tide was out, so we walked along the sand for an hour.",
            "Later we found a small cafe that served fresh fish and chips.",
            "It started to rain in the afternoon, so we drove home early.",
        ];
        let mut text = String::new();
        let mut timings = Vec::new();
        for sentence in sentences.iter().chain(&sentences) {
            for word in sentence.split(' ') {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(word);
                model.reset();
                let started = std::time::Instant::now();
                assert!(!model.prepare(1, &text).unwrap());
                let fresh = started.elapsed();
                let expected = model.prepared.as_ref().unwrap().logits.clone();
                // Prepare the text without its last word, then extend it.
                model.reset();
                model
                    .prepare(1, text.rsplit_once(' ').map_or("", |(head, _)| head))
                    .unwrap();
                let started = std::time::Instant::now();
                assert!(!model.prepare(1, &text).unwrap());
                let extended = started.elapsed();
                let prepared = model.prepared.as_ref().unwrap();
                let tokens = prepared.ids.len();
                assert!(tokens <= CONTEXT_TOKENS);
                assert!(expected == prepared.logits, "{tokens} tokens");
                assert!(model.prepare(1, &text).unwrap());
                timings.push((tokens, fresh, extended));
            }
        }
        let longest = timings.iter().map(|t| t.0).max().unwrap();
        assert!(longest > CONTEXT_TOKENS - WINDOW_STEP, "the window filled");
        for (tokens, fresh, extended) in timings.iter().filter(|t| t.0 + 8 >= longest) {
            eprintln!("{tokens} tokens: fresh {fresh:?}, extended {extended:?}");
        }
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
