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

struct Beam<S> {
    state: S,
    logits: Vec<f32>,
    bytes: Vec<u8>,
    score: f64,
}
struct Candidate {
    parent: usize,
    token: usize,
    bytes: Vec<u8>,
    score: f64,
}

/// The caller accounts for the context forward pass. Each selected extension
/// costs one evaluation; terminal boundary mass costs none. Scores stay private.
pub fn search<S: Clone>(
    query: &GenerationQuery,
    pieces: &[Vec<u8>],
    state: S,
    logits: Vec<f32>,
    mut evaluations: usize,
    started: Instant,
    mut forward: impl FnMut(&mut S, u32, usize) -> Result<Vec<f32>>,
) -> Result<Vec<String>> {
    let mut beams = vec![Beam {
        state,
        logits,
        bytes: Vec::new(),
        score: 0.,
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
        candidates.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.parent.cmp(&b.parent))
                .then(a.token.cmp(&b.token))
        });
        let mut next = Vec::new();
        for candidate in candidates.into_iter().take(WIDTH) {
            // Reserve time for the final evaluation and reply transport; the parent still enforces 2 s.
            if evaluations >= EVALUATIONS || started.elapsed() >= Duration::from_millis(1600) {
                break;
            }
            let mut state = beams[candidate.parent].state.clone();
            let logits = forward(&mut state, candidate.token as u32, depth)?;
            evaluations += 1;
            ensure!(
                logits.len() == pieces.len() && logits.iter().all(|v| v.is_finite()),
                "logits"
            );
            if let Some(word) = completed(&candidate.bytes, query) {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let all: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
                let mass: f64 = logits
                    .iter()
                    .enumerate()
                    .filter(|(id, _)| boundary(&pieces[*id], *id))
                    .map(|(_, v)| (*v as f64 - max).exp())
                    .sum();
                let score = candidate.score + (mass / all).ln();
                // A token fragment is not a completed word merely because EOS
                // has nonzero probability. Require a probable word boundary.
                if mass / all < 0.5 {
                    next.push(Beam {
                        state,
                        logits,
                        bytes: candidate.bytes,
                        score: candidate.score,
                    });
                    continue;
                }
                // Different tokenizations/case variants represent disjoint paths.
                finished
                    .entry(word)
                    .and_modify(|old| {
                        let high = old.max(score);
                        *old = high + ((*old - high).exp() + (score - high).exp()).ln();
                    })
                    .or_insert(score);
            }
            next.push(Beam {
                state,
                logits,
                bytes: candidate.bytes,
                score: candidate.score,
            });
        }
        beams = next;
        if beams.is_empty()
            || evaluations >= EVALUATIONS
            || started.elapsed() >= Duration::from_millis(1600)
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
            (),
            vec![-10., 5., 4., -10., 0.],
            1,
            Instant::now(),
            |_, token, _| {
                calls += 1;
                Ok(if token == 2 {
                    vec![-20., -20., -20., 10., -20.]
                } else {
                    vec![10., -20., -20., -20., -20.]
                })
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
                (),
                vec![-20., 10., -20.],
                1,
                Instant::now(),
                |_, token, _| {
                    Ok(if token == 1 {
                        vec![-20., -20., 10.]
                    } else {
                        vec![10., -20., -20.]
                    })
                },
            )
            .unwrap();
            assert_eq!(result[0], "café");
            assert!(result.iter().all(|w| !w.contains('\uFFFD')));
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
            (),
            vec![0.; 9],
            1,
            Instant::now(),
            |_, _, _| {
                calls += 1;
                Ok(vec![0.; 9])
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
}
