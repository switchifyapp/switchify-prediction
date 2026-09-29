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

#[derive(Serialize)]
pub struct PrefixAccuracy {
    pub prefix_chars: usize,
    pub queries: usize,
    pub top1: f64,
    pub top5: f64,
    pub vocabulary_coverage: f64,
}
#[derive(Serialize)]
pub struct ScoreReport {
    pub evaluation_sha256: String,
    pub database_sha256: String,
    pub evaluated_sentences: usize,
    pub accuracy: Vec<PrefixAccuracy>,
    pub selection_proxy: SelectionProxy,
    pub warm_p50_ms: f64,
    pub warm_p95_ms: f64,
    pub cold_load_ms: f64,
    pub model_payload_bytes: usize,
    pub database_bytes: u64,
    pub hardware: String,
}
#[derive(Serialize)]
pub struct SelectionProxy {
    pub definition: String,
    pub characters_without_prediction: usize,
    pub simulated_selections_with_prediction: usize,
    pub savings_fraction: f64,
}
/// Score a fixed database on external sentences. The quality pipeline checks partition separation.
pub fn score_database(path: &Path, text: &str, hardware: String) -> Result<ScoreReport> {
    let test: BTreeSet<_> = sentences(text).into_iter().map(|s| s.join(" ")).collect();
    if test.is_empty() {
        return Err(crate::Error::Invalid("evaluation set is empty".into()));
    }
    let start = Instant::now();
    let predictor = Predictor::open(path, None)?;
    let cold_load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let vocabulary = predictor
        .combined
        .0
        .get(&Vec::new())
        .expect("validated baseline has unigrams");
    let mut counts = [[0usize; 4]; 5]; // queries, top1, top5, in vocabulary
    let mut timings = Vec::new();
    let mut chars_total = 0;
    let mut selection_total = 0;
    let opts = Options {
        min_chars: 0,
        ..Options::default()
    };
    predictor.predict("", "", opts);
    for sentence in &test {
        let words: Vec<_> = sentence.split_whitespace().collect();
        for (i, target) in words.iter().enumerate() {
            let graphemes: Vec<_> = target.graphemes(true).collect();
            let before = words[i.saturating_sub(2)..i].join(" ");
            chars_total += graphemes.len();
            let mut best = graphemes.len();
            for n in 0..graphemes.len() {
                if n >= 5 && n + 1 >= best {
                    break;
                }
                let prefix = graphemes[..n].concat();
                let start = Instant::now();
                let predictions = predictor.predict(&before, &prefix, opts);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let rank = predictions.iter().position(|s| s.word == *target);
                if n < 5 {
                    timings.push(elapsed);
                    counts[n][0] += 1;
                    counts[n][1] += usize::from(rank == Some(0));
                    counts[n][2] += usize::from(rank.is_some());
                    counts[n][3] += usize::from(vocabulary.contains_key(*target));
                }
                if n >= 2
                    && let Some(rank) = rank
                {
                    best = best.min(n + rank + 1);
                }
            }
            selection_total += best;
        }
    }
    timings.sort_by(f64::total_cmp);
    let percentile = |p: usize| {
        timings[(timings.len() * p)
            .div_ceil(100)
            .saturating_sub(1)
            .min(timings.len() - 1)]
    };
    Ok(ScoreReport {
        evaluation_sha256:digest(text.as_bytes()), database_sha256:digest(&std::fs::read(path)?),
        evaluated_sentences:test.len(),
        accuracy:counts.into_iter().enumerate().map(|(prefix_chars,c)| PrefixAccuracy{prefix_chars,queries:c[0],top1:ratio(c[1],c[0]),top5:ratio(c[2],c[0]),vocabulary_coverage:ratio(c[3],c[0])}).collect(),
        selection_proxy:SelectionProxy{
            definition:"Optimistic offline rank-sensitive proxy: each typed grapheme costs one selection; accept costs rank+1 among five slots; choose cheapest completion after >=2 graphemes, or type word fully. Excludes spaces, scan navigation/timing, errors and cognitive effort; not measured AAC switch savings.".into(),
            characters_without_prediction:chars_total,simulated_selections_with_prediction:selection_total,
            savings_fraction:1.0-ratio(selection_total,chars_total),
        },
        warm_p50_ms:percentile(50),warm_p95_ms:percentile(95),cold_load_ms,
        model_payload_bytes:predictor.model_payload_bytes(),database_bytes:std::fs::metadata(path)?.len(),hardware,
    })
}
