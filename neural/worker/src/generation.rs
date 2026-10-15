//! Bounded byte-level beam search. Prefixes constrain decoded words, not token IDs.
use anyhow::{Result, ensure};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
use switchify_prediction::normalize;
use switchify_prediction_neural::protocol::{GenerationQuery, PhraseQuery, valid_word};
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

const WIDTH: usize = 8;
/// Search depth per word; a phrase shares this many per word between its words.
const TOKENS: usize = 8;
const EVALUATIONS: usize = 64;

/// Beam width and forward-evaluation budget. Production builds always use the
/// policy constants. Only the benchmark-only `experiment` feature can lower them.
#[cfg(not(feature = "experiment"))]
fn limits() -> (usize, usize) {
    (WIDTH, EVALUATIONS)
}
#[cfg(feature = "experiment")]
static LIMITS: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
#[cfg(feature = "experiment")]
fn limits() -> (usize, usize) {
    *LIMITS.get_or_init(|| (WIDTH, EVALUATIONS))
}
/// Read lowered limits once at startup. Invalid values fail rather than silently
/// measuring the policy limits.
#[cfg(feature = "experiment")]
pub fn configure() -> Result<(usize, usize)> {
    let read = |name: &str, default: usize| -> Result<usize> {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => Ok(default),
            Ok(value) => {
                let value: usize = value.parse()?;
                ensure!((1..=default).contains(&value), "experiment");
                Ok(value)
            }
            Err(_) => anyhow::bail!("experiment"),
        }
    };
    let limits = (
        read("SWITCHIFY_SMOL_EXPERIMENT_BEAM_WIDTH", WIDTH)?,
        read("SWITCHIFY_SMOL_EXPERIMENT_EVALUATIONS", EVALUATIONS)?,
    );
    ensure!(*LIMITS.get_or_init(|| limits) == limits, "experiment");
    Ok(limits)
}

/// What a search completes: `words` whole words separated by single spaces,
/// the first constrained by `prefix`, each result checked by `accepts`.
pub struct Target<'a> {
    pub before: &'a str,
    pub prefix: &'a str,
    pub words: usize,
    pub limit: usize,
    pub accepts: &'a dyn Fn(&[String]) -> bool,
}
impl<'a> Target<'a> {
    pub fn word(query: &'a GenerationQuery, accepts: &'a dyn Fn(&[String]) -> bool) -> Self {
        Self {
            before: &query.before,
            prefix: &query.prefix,
            words: 1,
            limit: query.limit,
            accepts,
        }
    }
    pub fn phrase(query: &'a PhraseQuery, accepts: &'a dyn Fn(&[String]) -> bool) -> Self {
        Self {
            before: &query.before,
            prefix: &query.prefix,
            words: switchify_prediction_neural::protocol::PHRASE_WORDS,
            limit: query.limit,
            accepts,
        }
    }
}

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

/// A trailing incomplete code point may become valid on the next token. Up to
/// `words` words separated by single spaces; every word but the last is whole,
/// and the first is constrained by the prefix.
fn partial(bytes: &[u8], separated: bool, prefix: &str, words: usize) -> bool {
    let bytes = if separated {
        if bytes.first() != Some(&b' ') {
            return false;
        }
        &bytes[1..]
    } else {
        bytes.strip_prefix(b" ").unwrap_or(bytes)
    };
    if bytes.len() > 129 * words - 1 {
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
    let parts: Vec<&str> = normalized.split(' ').collect();
    if parts.len() > words {
        return false;
    }
    let last = parts.len() - 1;
    if parts[..last].iter().any(|w| !valid_word(w)) || parts[last].len() > 128 {
        return false;
    }
    let mut letter = false;
    let mut apostrophe = false;
    for c in parts[last].chars() {
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
    let first: String = parts[0].nfd().collect();
    let prefix: String = prefix.nfd().collect();
    first.starts_with(&prefix) || (last == 0 && prefix.starts_with(&first))
}

/// Words begun in `bytes`, counting a trailing space as beginning one.
fn begun(bytes: &[u8]) -> usize {
    let bytes = bytes.strip_prefix(b" ").unwrap_or(bytes);
    1 + bytes.iter().filter(|&&b| b == b' ').count()
}

fn completed(bytes: &[u8], target: &Target) -> Option<Vec<String>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let words: Vec<String> = normalize(text.strip_prefix(' ').unwrap_or(text))
        .split(' ')
        .map(String::from)
        .collect();
    (words.len() == target.words && words.iter().all(|w| valid_word(w)) && (target.accepts)(&words))
        .then_some(words)
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

/// Whether further search can no longer change the ranked result. All later
/// probability mass, for any result, is at most the beams' combined `open`
/// bound. Each returned result must lead the next ranked one, or an unseen one,
/// by more than that mass.
fn settled<K>(finished: &BTreeMap<K, f64>, beams: &[Beam], limit: usize) -> bool {
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
    let accepts = |words: &[String]| query.accepts(words);
    let target = Target::word(query, &accepts);
    let found = run(&target, pieces, logits, evaluations, started, forward, true)?;
    Ok(found.into_iter().flatten().collect())
}

/// Like `search`, completing whole phrases within the same budget.
pub fn search_phrases(
    query: &PhraseQuery,
    pieces: &[Vec<u8>],
    logits: Vec<f32>,
    evaluations: usize,
    started: Instant,
    forward: impl FnMut(&[u32], &[u32], usize) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<Vec<String>>> {
    let accepts = |phrase: &[String]| query.accepts(&[phrase.to_vec()]);
    let target = Target::phrase(query, &accepts);
    run(&target, pieces, logits, evaluations, started, forward, true)
}

fn run(
    target: &Target,
    pieces: &[Vec<u8>],
    logits: Vec<f32>,
    mut evaluations: usize,
    started: Instant,
    mut forward: impl FnMut(&[u32], &[u32], usize) -> Result<Vec<Vec<f32>>>,
    stop_when_settled: bool,
) -> Result<Vec<Vec<String>>> {
    let boundaries: Vec<usize> = (0..pieces.len())
        .filter(|&id| boundary(&pieces[id], id))
        .collect();
    let mut beams = vec![Beam {
        logits,
        bytes: Vec::new(),
        score: 0.,
        open: 0.,
    }];
    let mut finished: BTreeMap<Vec<String>, f64> = BTreeMap::new();
    let mut batch = Duration::ZERO;
    let (width, budget) = limits();
    for depth in 0..TOKENS * target.words {
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
                if candidates.len() == width && score <= candidates.last().unwrap().score {
                    continue;
                }
                let mut bytes = beam.bytes.clone();
                bytes.extend_from_slice(piece);
                if partial(
                    &bytes,
                    !target.before.is_empty(),
                    target.prefix,
                    target.words,
                ) {
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
                    candidates.truncate(width);
                }
            }
        }
        // Reserve time for this batch, estimated by the last one, and reply
        // transport; the parent still enforces 2 s.
        if started.elapsed() + batch >= Duration::from_millis(1600) {
            break;
        }
        candidates.truncate(budget.saturating_sub(evaluations));
        if candidates.is_empty() {
            break;
        }
        let parents: Vec<u32> = candidates.iter().map(|c| c.parent as u32).collect();
        let tokens: Vec<u32> = candidates.iter().map(|c| c.token as u32).collect();
        let evaluated = Instant::now();
        let rows = forward(&parents, &tokens, depth)?;
        batch = evaluated.elapsed();
        ensure!(rows.len() == candidates.len(), "logits");
        evaluations += candidates.len();
        let mut next = Vec::new();
        for (candidate, logits) in candidates.into_iter().zip(rows) {
            ensure!(
                logits.len() == pieces.len() && logits.iter().all(|v| v.is_finite()),
                "logits"
            );
            let mass = boundary_mass(&logits, &boundaries);
            if let Some(words) = completed(&candidate.bytes, target) {
                // A token fragment is not a completed word merely because EOS
                // has nonzero probability. Require a probable word boundary.
                if mass >= 0.5 {
                    let score = candidate.score + mass.ln();
                    // Different tokenizations/case variants represent disjoint paths.
                    finished
                        .entry(words)
                        .and_modify(|old| *old = log_sum_exp(*old, score))
                        .or_insert(score);
                }
            }
            // Within the last word, every later completion continues it with a
            // piece that does not begin with a word boundary. Before the last
            // word, a boundary may begin the next word, so all mass stays open.
            let open = if begun(&candidate.bytes) < target.words {
                candidate.score
            } else {
                candidate.score + (1. - mass).max(0.).ln()
            };
            next.push(Beam {
                logits,
                bytes: candidate.bytes,
                score: candidate.score,
                open,
            });
        }
        beams = next;
        if evaluations >= budget || (stop_when_settled && settled(&finished, &beams, target.limit))
        {
            break;
        }
    }
    let mut found: Vec<_> = finished.into_iter().collect();
    found.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(found
        .into_iter()
        .take(target.limit)
        .map(|(words, _)| words)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn search_limits_match_the_frozen_policy() {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../model-bundle.json")).unwrap();
        let policy = &manifest["policy"]["generation"];
        assert_eq!(policy["beam_width"], WIDTH);
        assert_eq!(policy["max_forward_evaluations"], EVALUATIONS);
        // Without configured overrides, experiment builds search like production.
        assert_eq!(limits(), (WIDTH, EVALUATIONS));
    }

    #[test]
    fn byte_prefix_and_word_shapes() {
        assert!(partial(b" caf\xc3", true, "café", 1));
        assert!(partial(" cafe\u{301}".as_bytes(), true, "café", 1));
        assert!(partial(b" can't", true, "can'", 1));
        assert!(!partial(b" two words", true, "", 1));
        assert!(!partial(b" help", true, "z", 1));
        assert!(!partial(b" \xff", true, "", 1));
        assert!(!partial(b" ''", true, "", 1));
    }
    #[test]
    fn phrase_shapes_complete_earlier_words_and_constrain_the_first() {
        assert!(partial(b" two words", true, "", 2));
        assert!(partial(b" would like", true, "wo", 2));
        assert!(partial(b" would", true, "wo", 2));
        assert!(partial(b" would ", true, "wo", 2));
        assert!(partial(b" wo", true, "would", 2));
        assert!(partial(b"would l", false, "", 2));
        assert!(partial(b" caf\xc3", true, "café", 2));
        assert!(partial(" café au".as_bytes(), true, "caf", 2));
        // A completed first word must itself complete the prefix.
        assert!(!partial(b" wo like", true, "would", 2));
        assert!(!partial(b" would like", true, "z", 2));
        assert!(!partial(b" would  like", true, "", 2));
        assert!(!partial(b" would like you", true, "", 2));
        assert!(!partial(b" would like ", true, "", 2));
        assert!(!partial(b" can' t", true, "", 2));
        assert!(!partial(b" would, like", true, "", 2));
        assert!(!partial(b"would like", true, "", 2));
        assert!(!partial(&[b'a'; 129], false, "", 2));
        let mut long = vec![b'a'; 128];
        long.push(b' ');
        long.extend([b'b'; 128]);
        assert!(partial(&long, false, "", 2));
        long.push(b'b');
        assert!(!partial(&long, false, "", 2));
        assert_eq!(begun(b" would"), 1);
        assert_eq!(begun(b" would "), 2);
        assert_eq!(begun(b"would like"), 2);
    }
    #[test]
    fn phrase_search_completes_two_words_within_budget() {
        let query = PhraseQuery {
            id: 1,
            session: 1,
            before: "i".into(),
            prefix: "wo".into(),
            exclude: vec![vec!["would".into(), "love".into()]],
            limit: 3,
        };
        let pieces: Vec<Vec<u8>> = ["", " would", " like", " love", " wo", "rk", ".", " it"]
            .iter()
            .map(|p| p.as_bytes().to_vec())
            .collect();
        // After the context: " would" or " wo"; "rk" completes "work"; after a
        // first word: " like", " love" or " it"; after a second word: ".".
        let row = |path: &[u32]| -> Vec<f32> {
            match path {
                [] => vec![-20., 4., -20., -20., 3., -20., -20., -20.],
                [4] => vec![-20., -20., -20., -20., -20., 10., -20., -20.],
                [1] | [4, 5] => vec![-20., -20., 5., 4., -20., -20., -20., 3.],
                _ => vec![-20., -20., -20., -20., -20., -20., 10., -20.],
            }
        };
        let mut paths: Vec<Vec<u32>> = vec![vec![]];
        let mut calls = 0;
        let phrases = search_phrases(
            &query,
            &pieces,
            row(&[]),
            1,
            Instant::now(),
            |parents, tokens, _| {
                calls += tokens.len();
                paths = parents
                    .iter()
                    .zip(tokens)
                    .map(|(&p, &t)| {
                        let mut path = paths[p as usize].clone();
                        path.push(t);
                        path
                    })
                    .collect();
                Ok(paths.iter().map(|p| row(p)).collect())
            },
        )
        .unwrap();
        let phrase = |a: &str, b: &str| vec![a.to_owned(), b.to_owned()];
        assert_eq!(phrases[0], phrase("would", "like"));
        assert!(phrases.contains(&phrase("work", "like")));
        assert!(!phrases.contains(&phrase("would", "love")));
        assert!(phrases.len() <= 3);
        assert!(calls <= 63);
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
        assert!(!partial(&[b'a'; 129], false, "", 1));
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
        let mut stopped_early = [0, 0];
        let mut found = [0, 0];
        for words in [1, 2] {
            for seed in 0..400u64 {
                let mut results = Vec::new();
                for stop in [false, true] {
                    let before = if seed % 2 == 0 { "x" } else { "" };
                    let prefix = if seed % 3 == 0 { "b" } else { "" };
                    let word = GenerationQuery {
                        id: 1,
                        session: 1,
                        before: before.into(),
                        prefix: prefix.into(),
                        exclude: if seed % 5 == 0 {
                            vec!["a".into()]
                        } else {
                            vec![]
                        },
                        limit: 3,
                    };
                    let phrase = PhraseQuery {
                        id: 1,
                        session: 1,
                        before: before.into(),
                        prefix: prefix.into(),
                        exclude: if seed % 5 == 0 {
                            vec![vec!["b".into(), "a".into()]]
                        } else {
                            vec![]
                        },
                        limit: 3,
                    };
                    let accepts_word = |w: &[String]| word.accepts(w);
                    let accepts_phrase = |p: &[String]| phrase.accepts(&[p.to_vec()]);
                    let target = if words == 1 {
                        Target::word(&word, &accepts_word)
                    } else {
                        Target::phrase(&phrase, &accepts_phrase)
                    };
                    let mut paths = vec![seed];
                    let mut calls = 0;
                    let result = run(
                        &target,
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
                    assert!(result.iter().all(|r| r.len() == words));
                    results.push((result, calls));
                }
                assert_eq!(results[0].0, results[1].0, "words {words} seed {seed}");
                assert!(results[1].1 <= results[0].1);
                stopped_early[words - 1] += usize::from(results[1].1 < results[0].1);
                found[words - 1] += usize::from(!results[0].0.is_empty());
            }
        }
        assert!(stopped_early.iter().all(|&n| n > 0), "{stopped_early:?}");
        assert!(found.iter().all(|&n| n > 0), "{found:?}");
    }
}
