//! Fixed synthetic-neighbor benchmark; no native input or personal learning.
use std::{env, fs, path::PathBuf, time::Instant};
use switchify_prediction::{KeyHitMap, Options, Predictor, digest, production, sentences};
use unicode_segmentation::UnicodeSegmentation;

struct Case {
    id: String,
    before: String,
    clean: String,
    noisy: String,
    target: String,
}

fn cases(text: &str, length: usize, map: &KeyHitMap) -> Vec<Case> {
    let mut result = Vec::new();
    for (sentence_index, sentence) in sentences(text).iter().enumerate() {
        for (word_index, word) in sentence.iter().enumerate() {
            let parts: Vec<_> = word.graphemes(true).collect();
            if parts.len() <= length {
                continue;
            }
            let mut prefix = parts[..length].to_vec();
            let eligible: Vec<_> = (0..length)
                .filter(|&i| !map.neighbors(prefix[i]).is_empty())
                .collect();
            if eligible.is_empty() {
                continue;
            }
            let id =
                digest(format!("key-hits-v1:{sentence_index}:{word_index}:{length}").as_bytes());
            let seed = u64::from_str_radix(&id[..16], 16).unwrap();
            let position = eligible[(seed % eligible.len() as u64) as usize];
            let neighbors = map.neighbors(prefix[position]);
            let typed =
                &neighbors[((seed / eligible.len() as u64) % neighbors.len() as u64) as usize].key;
            let clean = prefix.concat();
            prefix[position] = typed;
            result.push(Case {
                id,
                before: sentence[word_index.saturating_sub(2)..word_index].join(" "),
                clean,
                noisy: prefix.concat(),
                target: word.clone(),
            });
        }
    }
    result.sort_by(|a, b| a.id.cmp(&b.id));
    result.truncate(1000);
    result
}

fn hit(words: impl Iterator<Item = impl AsRef<str>>, target: &str) -> (usize, usize) {
    let words: Vec<_> = words.map(|w| w.as_ref().to_owned()).collect();
    (
        usize::from(words.first().is_some_and(|w| w == target)),
        usize::from(words.iter().any(|w| w == target)),
    )
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * quantile).ceil() as usize]
}

fn short_prefix_probes(predictor: &Predictor, map: &KeyHitMap) -> Vec<serde_json::Value> {
    let options = Options {
        limit: 5,
        min_chars: 0,
        ..Options::default()
    };
    ["", "s"]
        .into_iter()
        .map(|prefix| {
            let mut exact_times = Vec::new();
            let mut hit_times = Vec::new();
            // These fixed probes are separate from the corpus accuracy sample.
            predictor.predict("i want ", prefix, options);
            predictor.predict_with_hit_map("i want ", prefix, options, map);
            for _ in 0..50 {
                let started = Instant::now();
                let exact = predictor.predict("i want ", prefix, options);
                exact_times.push(started.elapsed().as_secs_f64() * 1000.);
                let started = Instant::now();
                let hits = predictor.predict_with_hit_map("i want ", prefix, options, map);
                hit_times.push(started.elapsed().as_secs_f64() * 1000.);
                if prefix.is_empty() {
                    assert_eq!(
                        exact.iter().map(|s| (&s.word, s.score)).collect::<Vec<_>>(),
                        hits.iter().map(|s| (&s.word, s.score)).collect::<Vec<_>>()
                    );
                }
            }
            serde_json::json!({"prefix":prefix,"prefix_graphemes":prefix.len(),"queries":50,
            "exact_p50_ms":percentile(&mut exact_times,0.5),
            "exact_p95_ms":percentile(&mut exact_times,0.95),
            "hit_map_p50_ms":percentile(&mut hit_times,0.5),
            "hit_map_p95_ms":percentile(&mut hit_times,0.95)})
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: key_hit_benchmark BASELINE EVALUATION_TEXT".into());
    }
    let baseline = PathBuf::from(&args[0]);
    let input = PathBuf::from(&args[1]);
    let validated = production::validate_english(&baseline)?;
    let text = fs::read_to_string(&input)?;
    let predictor = Predictor::open(&baseline, None)?;
    let map = KeyHitMap::qwerty();
    let options = Options {
        limit: 5,
        min_chars: 2,
        ..Options::default()
    };
    let mut metrics = Vec::new();
    for length in [2, 3, 4] {
        let queries = cases(&text, length, &map);
        if queries.is_empty() {
            return Err("no eligible benchmark cases".into());
        }
        let ids = queries
            .iter()
            .map(|c| c.id.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let mut counts = [[0usize; 2]; 4];
        let mut exact_times = Vec::new();
        let mut soft_times = Vec::new();
        let mut clean_times = Vec::new();
        for case in &queries {
            let clean = predictor.predict(&case.before, &case.clean, options);
            let started = Instant::now();
            let soft_clean =
                predictor.predict_with_hit_map(&case.before, &case.clean, options, &map);
            clean_times.push(started.elapsed().as_secs_f64() * 1000.);
            let started = Instant::now();
            let noisy = predictor.predict(&case.before, &case.noisy, options);
            exact_times.push(started.elapsed().as_secs_f64() * 1000.);
            let started = Instant::now();
            let soft_noisy =
                predictor.predict_with_hit_map(&case.before, &case.noisy, options, &map);
            soft_times.push(started.elapsed().as_secs_f64() * 1000.);
            let outcomes = [
                hit(clean.iter().map(|s| &s.word), &case.target),
                hit(soft_clean.iter().map(|s| &s.word), &case.target),
                hit(noisy.iter().map(|s| &s.word), &case.target),
                hit(soft_noisy.iter().map(|s| &s.word), &case.target),
            ];
            for (total, (top1, top5)) in counts.iter_mut().zip(outcomes) {
                total[0] += top1;
                total[1] += top5;
            }
        }
        let accuracy: Vec<_> = counts
            .iter()
            .map(|c| {
                serde_json::json!({
                    "top1": c[0] as f64 / queries.len() as f64,
                    "top5": c[1] as f64 / queries.len() as f64,
                })
            })
            .collect();
        metrics.push(
            serde_json::json!({"prefix_graphemes":length,"queries":queries.len(),
            "case_ids_sha256":digest(ids.as_bytes()),
            "clean_exact":accuracy[0],"clean_hit_map":accuracy[1],
            "noisy_exact":accuracy[2],"noisy_hit_map":accuracy[3],
            "noisy_exact_p95_ms":percentile(&mut exact_times,0.95),
            "clean_hit_map_p50_ms":percentile(&mut clean_times,0.5),
            "clean_hit_map_p95_ms":percentile(&mut clean_times,0.95),
            "noisy_hit_map_p50_ms":percentile(&mut soft_times,0.5),
            "noisy_hit_map_p95_ms":percentile(&mut soft_times,0.95)}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "protocol":"key-hits-v2", "sampling_seed":"key-hits-v1",
            "evaluation_sha256":digest(text.as_bytes()),
            "baseline":validated,"metrics":metrics,
            "layout":"QWERTY radius 1.3 spread 0.45; one uniform retained-neighbor substitution",
            "short_prefix_probes":short_prefix_probes(&predictor,&map),
            "sampling":"hash-rank at most 1000 eligible targets per prefix; target longer than prefix",
            "limitations":"Synthetic one-substitution errors, not observed switch-user mistakes. No empirical calibration or user-benefit claim. Exact noisy target recovery is zero by construction. Known development corpus, personal learning disabled. Corpus timing covers clean/noisy 2-4 graphemes. Short-prefix probes use only fixed i-want contexts, not a representative workload. Timing describes this machine, not a latency ceiling."
        }))?
    );
    Ok(())
}
