//! Deterministic sentence-held-out evaluation; never learns from test sentences.
use crate::{Options, Predictor, Result, build, digest, sentences};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path, time::Instant};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Serialize)]
pub struct Accuracy {
    pub prefix_chars: usize,
    pub queries: usize,
    pub top1: f64,
    pub top5: f64,
    pub vocabulary_coverage: f64,
    pub unigram_top1: f64,
    pub unigram_top5: f64,
}
#[derive(Serialize)]
pub struct Report {
    pub protocol: String,
    pub corpus_sha256: String,
    pub train_sentences: usize,
    pub test_sentences: usize,
    pub accuracy: Vec<Accuracy>,
    pub warm_p50_ms: f64,
    pub warm_p95_ms: f64,
    pub warm_query_count: usize,
    pub cold_load_ms: f64,
    pub model_payload_bytes: usize,
    pub peak_process_rss_bytes: Option<u64>,
    pub database_bytes: u64,
    pub hardware: String,
    pub warm_p95_target_ms: f64,
    pub warm_p95_target_met: bool,
}
/// Deduplicate normalized sentences, hash-sort, then hold out the first 10%.
pub fn split(text: &str) -> (Vec<String>, Vec<String>) {
    let unique: BTreeSet<String> = sentences(text).into_iter().map(|s| s.join(" ")).collect();
    let mut unique: Vec<_> = unique.into_iter().collect();
    unique.sort_by_cached_key(|s| (digest(s.as_bytes()), s.clone()));
    let test_count = unique.len().div_ceil(10);
    let train = unique.split_off(test_count);
    (train, unique)
}
fn ratio(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}
pub fn evaluate(text: &str, hardware: String) -> Result<Report> {
    let (train, test) = split(text);
    if train.is_empty() || test.is_empty() {
        return Err(crate::Error::Invalid(
            "evaluation needs at least two unique sentences".into(),
        ));
    }
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("evaluation.sqlite");
    build(
        &path,
        &train.join("\n"),
        "held-out evaluation training partition",
    )?;
    let start = Instant::now();
    let predictor = Predictor::open(&path, None)?;
    let cold_load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let vocabulary: BTreeSet<_> = train.iter().flat_map(|s| s.split_whitespace()).collect();
    let mut accuracy = Vec::new();
    let mut timings = Vec::new();
    // Prime the model before timing; no personal model is attached.
    predictor.predict(
        "",
        "",
        Options {
            min_chars: 0,
            ..Options::default()
        },
    );
    for prefix_chars in 0..=4 {
        let (mut queries, mut top1, mut top5, mut covered, mut uni1, mut uni5) = (0, 0, 0, 0, 0, 0);
        for sentence in &test {
            let words: Vec<_> = sentence.split_whitespace().collect();
            for (i, target) in words.iter().enumerate() {
                let graphemes: Vec<_> = target.graphemes(true).collect();
                // Only actual completions, not already completed words.
                if graphemes.len() <= prefix_chars {
                    continue;
                }
                let prefix = graphemes[..prefix_chars].concat();
                let before = words[i.saturating_sub(2)..i].join(" ");
                let options = Options {
                    min_chars: 0,
                    ..Options::default()
                };
                let start = Instant::now();
                let results = predictor.predict(&before, &prefix, options);
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
                let unigram = predictor.predict(
                    &before,
                    &prefix,
                    Options {
                        unigram_only: true,
                        ..options
                    },
                );
                queries += 1;
                covered += usize::from(vocabulary.contains(target));
                top1 += usize::from(results.first().is_some_and(|s| s.word == *target));
                top5 += usize::from(results.iter().any(|s| s.word == *target));
                uni1 += usize::from(unigram.first().is_some_and(|s| s.word == *target));
                uni5 += usize::from(unigram.iter().any(|s| s.word == *target));
            }
        }
        accuracy.push(Accuracy {
            prefix_chars,
            queries,
            top1: ratio(top1, queries),
            top5: ratio(top5, queries),
            vocabulary_coverage: ratio(covered, queries),
            unigram_top1: ratio(uni1, queries),
            unigram_top5: ratio(uni5, queries),
        });
    }
    timings.sort_by(f64::total_cmp);
    let percentile = |percent: usize| {
        timings[(timings.len() * percent)
            .div_ceil(100)
            .saturating_sub(1)
            .min(timings.len() - 1)]
    };
    let p95 = percentile(95);
    Ok(Report {
        protocol: "v1: normalized sentence deduplication; SHA-256 sort; first ceil(N/10) sentences held out; prefix lengths 0..4; words longer than prefix only; evaluation database excludes all held-out sentences".into(),
        corpus_sha256: digest(text.as_bytes()), train_sentences: train.len(), test_sentences: test.len(), accuracy,
        warm_p50_ms: percentile(50), warm_p95_ms: p95, warm_query_count: timings.len(), cold_load_ms,
        model_payload_bytes: predictor.model_payload_bytes(), peak_process_rss_bytes: None,
        database_bytes: std::fs::metadata(Path::new(&path))?.len(), hardware,
        warm_p95_target_ms: 20.0, warm_p95_target_met: p95 < 20.0,
    })
}
