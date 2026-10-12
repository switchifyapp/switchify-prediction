//! Bounded byte-level beam search. Prefixes constrain decoded words, not token IDs.
use anyhow::{Result, ensure};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
use switchify_prediction::normalize;
use switchify_prediction_neural::protocol::{GenerationQuery, valid_word};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

const WIDTH: usize = 8;
const TOKENS: usize = 8;
const EVALUATIONS: usize = 64;

/// Reverse the tokenizer's GPT-2 byte alphabet without replacing partial UTF-8.
pub fn pieces(tokenizer: &tokenizers::Tokenizer) -> Result<Vec<Vec<u8>>> {
    let mut bytes: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet: BTreeMap<char, u8> = bytes.iter().map(|&b| (char::from(b), b)).collect();
    let mut extra = 256;
    for b in 0..=255 {
        if !bytes.contains(&b) {
            alphabet.insert(char::from_u32(extra).unwrap(), b);
            bytes.push(b);
            extra += 1;
        }
    }
    (0..tokenizer.get_vocab_size(true) as u32)
        .map(|id| {
            let token = tokenizer
                .id_to_token(id)
                .ok_or_else(|| anyhow::anyhow!("tokenizer"))?;
            if token.starts_with("<|") {
                return Ok(Vec::new());
            }
            token
                .chars()
                .map(|c| {
                    alphabet
                        .get(&c)
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("tokenizer"))
                })
                .collect()
        })
        .collect()
}

fn boundary(piece: &[u8], id: usize) -> bool {
    id == 0
        || piece.first().is_some_and(|b| {
            b.is_ascii_whitespace() || b".!?;:,()[]{}\"-/".contains(b) || b.is_ascii_digit()
        })
}

/// A trailing incomplete code point may become valid on the next token.
fn partial(bytes: &[u8], separated: bool, prefix: &str) -> bool {
    let bytes = if separated {
        if bytes.first() != Some(&b' ') {
            return false;
        }
        &bytes[1..]
    } else {
        bytes.strip_prefix(b" ").unwrap_or(bytes)
    };
    if bytes.len() > 128 {
        return false;
    }
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) if e.error_len().is_none() => {
            std::str::from_utf8(&bytes[..e.valid_up_to()]).unwrap()
        }
        Err(_) => return false,
    };
    let normalized = normalize(text);
    let mut letter = false;
    let mut apostrophe = false;
    for c in normalized.chars() {
        if c.is_alphabetic() {
            letter = true;
            apostrophe = false;
        } else if is_combining_mark(c) && letter && !apostrophe {
        } else if c == '\'' && letter && !apostrophe {
            apostrophe = true;
        } else {
            return false;
        }
    }
    let text: String = normalized.nfd().collect();
    let prefix: String = prefix.nfd().collect();
    text.starts_with(&prefix) || prefix.starts_with(&text)
}

fn completed(bytes: &[u8], query: &GenerationQuery) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    let word = normalize(text.strip_prefix(' ').unwrap_or(text));
    (valid_word(&word) && query.accepts(std::slice::from_ref(&word))).then_some(word)
}

struct Beam {
    logits: Vec<f32>,
    bytes: Vec<u8>,
    score: f64,
    /// Upper bound on the log probability of every later completion through
    /// this beam.
    open: f64,
}
struct Candidate {
    parent: usize,
    token: usize,
    bytes: Vec<u8>,
    score: f64,
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        return high;
    }
    high + ((a - high).exp() + (b - high).exp()).ln()
}

/// Fraction of the next-token distribution on word boundaries.
fn boundary_mass(logits: &[f32], boundaries: &[usize]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let all: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
    let mass: f64 = boundaries
        .iter()
        .map(|&id| (logits[id] as f64 - max).exp())
        .sum();
    mass / all
}

/// Whether further search can no longer change the ranked result. Every later
/// completion extends a live beam by a piece that does not begin with a word
/// boundary, since such a piece never continues a word, so all later
/// probability mass, for any word, is at most the beams' combined `open` bound.
/// Each returned word must lead the next ranked word, or an unseen word, by
/// more than that mass.
fn settled(finished: &BTreeMap<String, f64>, beams: &[Beam], limit: usize) -> bool {
    if finished.len() < limit {
        return false;
    }
    let remaining = beams
        .iter()
        .fold(f64::NEG_INFINITY, |total, beam| {
            log_sum_exp(total, beam.open)
        })
        .exp();
    let mut scores: Vec<f64> = finished.values().map(|s| s.exp()).collect();
    scores.sort_by(|a, b| b.total_cmp(a));
    (0..limit).all(|i| scores[i] > scores.get(i + 1).copied().unwrap_or(0.) + remaining)
}

/// The caller accounts for the context forward pass. Each selected extension
/// costs one evaluation; terminal boundary mass costs none. Scores stay private.
///
/// `forward(parents, tokens, depth)` extends, for each `i`, the beam at
/// `parents[i]` by `tokens[i]`, and returns one logit row per extension. The
/// extensions become the next beams in the same order. One call evaluates every
/// extension at a depth, which the caller may batch.
pub fn search(
    query: &GenerationQuery,
    pieces: &[Vec<u8>],
    logits: Vec<f32>,
    evaluations: usize,
    started: Instant,
    forward: impl FnMut(&[u32], &[u32], usize) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<String>> {
    run(query, pieces, logits, evaluations, started, forward, true)
}

fn run(
    query: &GenerationQuery,
    pieces: &[Vec<u8>],
    logits: Vec<f32>,
    mut evaluations: usize,
    started: Instant,
    mut forward: impl FnMut(&[u32], &[u32], usize) -> Result<Vec<Vec<f32>>>,
    stop_when_settled: bool,
) -> Result<Vec<String>> {
    let boundaries: Vec<usize> = (0..pieces.len())
        .filter(|&id| boundary(&pieces[id], id))
        .collect();
    let mut beams = vec![Beam {
        logits,
        bytes: Vec::new(),
        score: 0.,
        open: 0.,
    }];
    let mut finished: BTreeMap<String, f64> = BTreeMap::new();
    for depth in 0..TOKENS {
        let mut candidates: Vec<Candidate> = Vec::new();
        for (parent, beam) in beams.iter().enumerate() {
            ensure!(
                beam.logits.len() == pieces.len() && beam.logits.iter().all(|v| v.is_finite()),
                "logits"
            );
            let max = beam
                .logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max) as f64;
            let normalizer = max
                + beam
                    .logits
                    .iter()
                    .map(|v| (*v as f64 - max).exp())
                    .sum::<f64>()
                    .ln();
            for (token, piece) in pieces.iter().enumerate() {
                if piece.is_empty() {
                    continue;
                }
                let score = beam.score + beam.logits[token] as f64 - normalizer;
                if candidates.len() == WIDTH && score <= candidates.last().unwrap().score {
                    continue;
                }
                let mut bytes = beam.bytes.clone();
                bytes.extend_from_slice(piece);
                if partial(&bytes, !query.before.is_empty(), &query.prefix) {
                    candidates.push(Candidate {
                        parent,
                        token,
                        bytes,
                        score,
                    });
                    candidates.sort_by(|a, b| {
                        b.score
                            .total_cmp(&a.score)
                            .then(a.parent.cmp(&b.parent))
                            .then(a.token.cmp(&b.token))
                    });
                    candidates.truncate(WIDTH);
                }
            }
        }
        // Reserve time for the final evaluation and reply transport; the parent still enforces 2 s.
        if started.elapsed() >= Duration::from_millis(1600) {
            break;
        }
        candidates.truncate(EVALUATIONS.saturating_sub(evaluations));
        if candidates.is_empty() {
            break;
        }
        let parents: Vec<u32> = candidates.iter().map(|c| c.parent as u32).collect();
        let tokens: Vec<u32> = candidates.iter().map(|c| c.token as u32).collect();
        let rows = forward(&parents, &tokens, depth)?;
        ensure!(rows.len() == candidates.len(), "logits");
        evaluations += candidates.len();
        let mut next = Vec::new();
        for (candidate, logits) in candidates.into_iter().zip(rows) {
            ensure!(
                logits.len() == pieces.len() && logits.iter().all(|v| v.is_finite()),
                "logits"
            );
            let mass = boundary_mass(&logits, &boundaries);
            if let Some(word) = completed(&candidate.bytes, query) {
                // A token fragment is not a completed word merely because EOS
                // has nonzero probability. Require a probable word boundary.
                if mass >= 0.5 {
                    let score = candidate.score + mass.ln();
                    // Different tokenizations/case variants represent disjoint paths.
                    finished
                        .entry(word)
                        .and_modify(|old| *old = log_sum_exp(*old, score))
                        .or_insert(score);
                }
            }
            next.push(Beam {
                logits,
                bytes: candidate.bytes,
                score: candidate.score,
                open: candidate.score + (1. - mass).max(0.).ln(),
            });
        }
        beams = next;
        if evaluations >= EVALUATIONS
            || (stop_when_settled && settled(&finished, &beams, query.limit))
        {
            break;
        }
    }
    let mut words: Vec<_> = finished.into_iter().collect();
    words.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(words
        .into_iter()
        .take(query.limit)
        .map(|(word, _)| word)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_prefix_and_word_shapes() {
        assert!(partial(b" caf\xc3", true, "café"));
        assert!(partial(" cafe\u{301}".as_bytes(), true, "café"));
        assert!(partial(b" can't", true, "can'"));
        assert!(!partial(b" two words", true, ""));
        assert!(!partial(b" help", true, "z"));
        assert!(!partial(b" \xff", true, ""));
        assert!(!partial(b" ''", true, ""));
    }
    #[test]
    fn constrained_search_spans_prefix_and_respects_budget() {
        let query = GenerationQuery {
            id: 1,
            session: 1,
            before: "say".into(),
            prefix: "he".into(),
            exclude: vec!["hello".into()],
            limit: 3,
        };
        let pieces = vec![
            vec![],
            b" hello".to_vec(),
            b" hel".to_vec(),
            b"ium".to_vec(),
            b" world".to_vec(),
        ];
        let mut calls = 0;
        let words = search(
            &query,
            &pieces,
            vec![-10., 5., 4., -10., 0.],
            1,
            Instant::now(),
            |_, tokens, _| {
                calls += tokens.len();
                Ok(tokens
                    .iter()
                    .map(|&token| {
                        if token == 2 {
                            vec![-20., -20., -20., 10., -20.]
                        } else {
                            vec![10., -20., -20., -20., -20.]
                        }
                    })
                    .collect())
            },
        )
        .unwrap();
        assert_eq!(words[0], "helium");
        assert!(!words.contains(&"hello".into()));
        assert!(calls <= 63);
    }

    #[test]
    fn incomplete_utf8_and_decomposed_words_complete_without_replacement() {
        for (first, second) in [
            (b" caf\xc3".to_vec(), b"\xa9".to_vec()),
            (b" cafe".to_vec(), "\u{301}".as_bytes().to_vec()),
        ] {
            let query = GenerationQuery {
                id: 1,
                session: 1,
                before: "a".into(),
                prefix: "café".into(),
                exclude: vec![],
                limit: 3,
            };
            let pieces = vec![vec![], first, second];
            let result = search(
                &query,
                &pieces,
                vec![-20., 10., -20.],
                1,
                Instant::now(),
                |_, tokens, _| {
                    Ok(tokens
                        .iter()
                        .map(|&token| {
                            if token == 1 {
                                vec![-20., -20., 10.]
                            } else {
                                vec![10., -20., -20.]
                            }
                        })
                        .collect())
                },
            )
            .unwrap();
            assert_eq!(result[0], "café");
            assert!(result.iter().all(|w| !w.contains('\u{FFFD}')));
        }
    }

    #[test]
    fn budget_and_fragment_gate_do_not_fill_from_statistics() {
        let query = GenerationQuery {
            id: 1,
            session: 1,
            before: "".into(),
            prefix: "".into(),
            exclude: vec![],
            limit: 3,
        };
        let pieces = std::iter::once(vec![])
            .chain((b'a'..=b'h').map(|b| vec![b]))
            .collect::<Vec<_>>();
        let mut calls = 0;
        let result = search(
            &query,
            &pieces,
            vec![0.; 9],
            1,
            Instant::now(),
            |_, tokens, _| {
                calls += tokens.len();
                Ok(vec![vec![0.; 9]; tokens.len()])
            },
        )
        .unwrap();
        assert_eq!(calls, 63);
        assert!(
            result.is_empty(),
            "continuations are more probable than boundaries"
        );
        assert!(!partial(&[b'a'; 129], false, ""));
    }

    #[test]
    fn settled_requires_each_rank_to_lead_by_the_open_mass() {
        let finished: BTreeMap<String, f64> = [("a", 0.5), ("b", 0.3), ("c", 0.1)]
            .into_iter()
            .map(|(w, p)| (w.to_owned(), f64::ln(p)))
            .collect();
        let beam = |open: f64| Beam {
            logits: vec![],
            bytes: vec![],
            score: 0.,
            open: open.ln(),
        };
        assert!(settled(&finished, &[beam(0.04), beam(0.05)], 3));
        assert!(!settled(&finished, &[beam(0.1)], 3));
        assert!(!settled(&finished, &[beam(0.25)], 2));
        assert!(settled(&finished, &[], 2));
        assert!(!settled(&finished, &[], 4));
    }

    #[test]
    fn stopping_when_settled_returns_the_full_search_result() {
        let pieces: Vec<Vec<u8>> = [
            "", " a", " b", " c", " d", " be", "e", "f", "g", "'", ".", " ",
        ]
        .iter()
        .map(|p| p.as_bytes().to_vec())
        .collect();
        let row = |path: u64| -> Vec<f32> {
            let mut state = path;
            (0..pieces.len())
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 40) as f32 / (1u64 << 24) as f32) * 12. - 6.
                })
                .collect()
        };
        let mut stopped_early = 0;
        for seed in 0..400u64 {
            let mut results = Vec::new();
            for stop in [false, true] {
                let query = GenerationQuery {
                    id: 1,
                    session: 1,
                    before: "x".into(),
                    prefix: if seed % 3 == 0 { "b".into() } else { "".into() },
                    exclude: if seed % 5 == 0 {
                        vec!["a".into()]
                    } else {
                        vec![]
                    },
                    limit: 3,
                };
                let mut paths = vec![seed];
                let mut calls = 0;
                let words = run(
                    &query,
                    &pieces,
                    row(seed),
                    1,
                    Instant::now(),
                    |parents, tokens, _| {
                        calls += tokens.len();
                        paths = parents
                            .iter()
                            .zip(tokens)
                            .map(|(&p, &t)| {
                                paths[p as usize]
                                    .wrapping_mul(31)
                                    .wrapping_add(t as u64 + 1)
                            })
                            .collect();
                        Ok(paths.iter().map(|&p| row(p)).collect())
                    },
                    stop,
                )
                .unwrap();
                results.push((words, calls));
            }
            assert_eq!(results[0].0, results[1].0, "seed {seed}");
            assert!(results[1].1 <= results[0].1);
            stopped_early += usize::from(results[1].1 < results[0].1);
        }
        assert!(stopped_early > 0);
    }
}
