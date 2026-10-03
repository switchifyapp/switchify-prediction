use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
use switchify_prediction::{Options, Predictor, sentences};
use tokenizers::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

const BASELINE_SHA: &str = "222253417d0a7a705823ffb7e599a3bcf5d5d3daf4a9d76161ac6b3e555aeaad";
const TRAIN_SHA: &str = "ff724de5ab3b6e609ed77b699c51287fc82f79d026fdecfa4892939abfbb2834";
const SHORTLIST: usize = 8;
const CONTEXT_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
enum Mode {
    Baseline,
    Neural,
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    fixtures: PathBuf,
    #[arg(long)]
    training: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    /// Use 64 for the frozen experiment; smaller values are calibration only.
    #[arg(long, default_value_t = 64)]
    per_cell: usize,
}

fn sha(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

#[derive(Deserialize)]
struct Source {
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    files: BTreeMap<String, Source>,
}

fn verify_model(model: &Path, manifest: &Path) -> Result<()> {
    let data: Manifest = serde_json::from_slice(&fs::read(manifest)?)?;
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        let expected = data.files.get(name).context("Missing model checksum")?;
        ensure!(
            sha(&model.join(name))? == expected.sha256,
            "Model checksum mismatch: {name}"
        );
    }
    Ok(())
}

fn boundary(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|c| c.is_whitespace() || c.is_numeric() || ".!?;:,()[]{}\"-/".contains(c))
}

fn log_prob(logits: &[f32], token: usize) -> f64 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    logits[token] as f64
        - maximum
        - logits
            .iter()
            .map(|x| (*x as f64 - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn log_boundary(logits: &[f32], ids: &[usize]) -> f64 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let all: f64 = logits.iter().map(|x| (*x as f64 - maximum).exp()).sum();
    let end: f64 = ids
        .iter()
        .map(|&i| (logits[i] as f64 - maximum).exp())
        .sum();
    (end / all).ln()
}

fn word_score<S: Clone>(
    tokens: &[u32],
    first_logits: &[f32],
    cache: &S,
    position: usize,
    boundary_ids: &[usize],
    mut forward: impl FnMut(u32, usize, &mut S) -> Result<Vec<f32>>,
) -> Result<f64> {
    ensure!(!tokens.is_empty(), "Empty candidate tokenization");
    let mut branch = cache.clone();
    let mut logits = first_logits.to_vec();
    let mut score = 0.;
    for (index, &token) in tokens.iter().enumerate() {
        score += log_prob(&logits, token as usize);
        logits = forward(token, position + index, &mut branch)?;
    }
    score += log_boundary(&logits, boundary_ids);
    ensure!(score.is_finite(), "Non-finite neural score");
    Ok(score)
}

struct Neural {
    model: Llama,
    tokenizer: Tokenizer,
    empty_cache: Cache,
    boundary_ids: Vec<usize>,
}

impl Neural {
    fn load(path: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let cfg: LlamaConfig = serde_json::from_slice(&fs::read(path.join("config.json"))?)?;
        let config = cfg.into_config(false);
        ensure!(
            config.bos_token_id == Some(0),
            "Expected pinned SmolLM2 BOS token"
        );
        let empty_cache = Cache::new(true, DType::F32, &config, &Device::Cpu)?;
        // Safe loader owns its bytes; no mmap lifetime or externally mutable mapping.
        let weights = fs::read(path.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &Device::Cpu)?;
        let model = Llama::load(vb, &config)?;
        let mut boundary_ids = vec![0]; // Pinned EOS.
        for id in 1..config.vocab_size {
            let decoded = tokenizer
                .decode(&[id as u32], false)
                .map_err(anyhow::Error::msg)?;
            if boundary(&decoded) {
                boundary_ids.push(id);
            }
        }
        Ok(Self {
            model,
            tokenizer,
            empty_cache,
            boundary_ids,
        })
    }

    fn forward(&self, ids: &[u32], position: usize, cache: &mut Cache) -> Result<Vec<f32>> {
        let input = Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?;
        Ok(self
            .model
            .forward(&input, position, cache)?
            .squeeze(0)?
            .to_vec1()?)
    }

    fn rank(&self, before: &str, candidates: &[String]) -> Result<Vec<String>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self
            .tokenizer
            .encode(before, false)
            .map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        let mut context = vec![0];
        context.extend_from_slice(&ids[ids.len().saturating_sub(CONTEXT_TOKENS - 1)..]);
        let mut cache = self.empty_cache.clone();
        let first_logits = self.forward(&context, 0, &mut cache)?;
        let mut scored = Vec::new();
        for (rank, word) in candidates.iter().enumerate() {
            let text = if before.is_empty() {
                word.clone()
            } else {
                format!(" {word}")
            };
            let encoded = self
                .tokenizer
                .encode(text, false)
                .map_err(anyhow::Error::msg)?;
            let tokens = encoded.get_ids();
            let score = word_score(
                tokens,
                &first_logits,
                &cache,
                context.len(),
                &self.boundary_ids,
                |token, position, branch| self.forward(&[token], position, branch),
            )?;
            scored.push((score, rank, word.clone()));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(scored.into_iter().take(5).map(|x| x.2).collect())
    }
}

#[derive(Clone, Debug)]
struct Query {
    domain: String,
    prefix_chars: usize,
    before: String,
    prefix: String,
    target: String,
    key: String,
}

fn queries(
    fixtures: &BTreeMap<String, Vec<String>>,
    training: &BTreeSet<String>,
    per_cell: usize,
) -> Result<Vec<Query>> {
    ensure!(per_cell > 0, "per-cell must be positive");
    let mut result = Vec::new();
    for (domain, texts) in fixtures {
        let unique: BTreeSet<String> = texts
            .iter()
            .flat_map(|t| sentences(t))
            .map(|s| s.join(" "))
            .collect();
        ensure!(
            unique.is_disjoint(training),
            "Fixture/training overlap in {domain}"
        );
        for prefix_chars in 0..=4 {
            let mut cell = Vec::new();
            let mut seen = BTreeSet::new();
            for sentence in &unique {
                let words: Vec<_> = sentence.split_whitespace().collect();
                for (index, target) in words.iter().enumerate() {
                    let graphemes: Vec<_> = target.graphemes(true).collect();
                    if graphemes.len() <= prefix_chars {
                        continue;
                    }
                    let before = words[..index].join(" ");
                    let prefix = graphemes[..prefix_chars].concat();
                    if !seen.insert((before.clone(), prefix.clone(), target.to_string())) {
                        continue;
                    }
                    let key = switchify_prediction::digest(
                        format!("{domain}\n{sentence}\n{index}\n{prefix_chars}").as_bytes(),
                    );
                    cell.push(Query {
                        domain: domain.clone(),
                        prefix_chars,
                        before,
                        prefix,
                        target: target.to_string(),
                        key,
                    });
                }
            }
            cell.sort_by(|a, b| a.key.cmp(&b.key));
            ensure!(
                cell.len() >= per_cell,
                "Not enough queries in {domain}/{prefix_chars}: {}",
                cell.len()
            );
            result.extend(cell.into_iter().take(per_cell));
        }
    }
    Ok(result)
}

#[derive(Default, Serialize)]
struct Cell {
    queries: usize,
    top1_hits: usize,
    top5_hits: usize,
    shortlist_hits: usize,
}
#[derive(Serialize)]
struct Report {
    mode: Mode,
    fixture_sha256: String,
    baseline_sha256: String,
    model_manifest_sha256: String,
    per_cell: usize,
    shortlist: usize,
    context_tokens: usize,
    cold_load_ms: f64,
    query_count: usize,
    warm_median_ms: f64,
    warm_p95_ms: f64,
    warm_max_ms: f64,
    failed_queries: usize,
    model_file_bytes: u64,
    baseline_payload_bytes: usize,
    cells: BTreeMap<String, BTreeMap<usize, Cell>>,
}

fn percentile(times: &[f64], percent: usize) -> f64 {
    times[(times.len() * percent).div_ceil(100).saturating_sub(1)]
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "Output already exists");
    ensure!(
        sha(&args.baseline)? == BASELINE_SHA,
        "Baseline checksum mismatch"
    );
    ensure!(
        sha(&args.training)? == TRAIN_SHA,
        "Training checksum mismatch"
    );
    let fixtures: BTreeMap<String, Vec<String>> =
        serde_json::from_slice(&fs::read(&args.fixtures)?)?;
    let training = fs::read_to_string(&args.training)?
        .lines()
        .map(str::to_string)
        .collect();
    let cases = queries(&fixtures, &training, args.per_cell)?;
    if cases.is_empty() {
        bail!("Empty workload");
    }
    verify_model(&args.model, &args.manifest)?;
    let start = Instant::now();
    let baseline = Predictor::open(&args.baseline, None)?;
    let neural = match args.mode {
        Mode::Baseline => None,
        Mode::Neural => Some(Neural::load(&args.model)?),
    };
    let cold_load_ms = start.elapsed().as_secs_f64() * 1000.;
    let predict = |q: &Query| -> Result<(Vec<String>, Vec<String>)> {
        let candidates: Vec<_> = baseline
            .predict(
                &q.before,
                &q.prefix,
                Options {
                    limit: SHORTLIST,
                    min_chars: 0,
                    ..Options::default()
                },
            )
            .into_iter()
            .map(|s| s.word)
            .collect();
        let words = match &neural {
            Some(n) => n.rank(&q.before, &candidates)?,
            None => candidates.iter().take(5).cloned().collect(),
        };
        ensure!(
            words.iter().all(|w| w.starts_with(&q.prefix)),
            "Invalid prefix completion"
        );
        Ok((words, candidates))
    };
    // Untimed prime. Each measured request starts with a fresh neural context cache.
    predict(&cases[0])?;
    let mut times = Vec::new();
    let mut cells: BTreeMap<String, BTreeMap<usize, Cell>> = BTreeMap::new();
    for (index, q) in cases.iter().enumerate() {
        let start = Instant::now();
        let (words, candidates) = predict(q)?;
        times.push(start.elapsed().as_secs_f64() * 1000.);
        let cell = cells
            .entry(q.domain.clone())
            .or_default()
            .entry(q.prefix_chars)
            .or_default();
        cell.queries += 1;
        cell.top1_hits += usize::from(words.first() == Some(&q.target));
        cell.top5_hits += usize::from(words.contains(&q.target));
        cell.shortlist_hits += usize::from(candidates.contains(&q.target));
        if index % 50 == 0 {
            eprintln!("Scored {}/{}", index + 1, cases.len());
        }
    }
    times.sort_by(f64::total_cmp);
    let report = Report {
        mode: args.mode,
        fixture_sha256: sha(&args.fixtures)?,
        baseline_sha256: sha(&args.baseline)?,
        model_manifest_sha256: sha(&args.manifest)?,
        per_cell: args.per_cell,
        shortlist: SHORTLIST,
        context_tokens: CONTEXT_TOKENS,
        cold_load_ms,
        query_count: times.len(),
        warm_median_ms: percentile(&times, 50),
        warm_p95_ms: percentile(&times, 95),
        warm_max_ms: *times.last().unwrap(),
        failed_queries: 0,
        model_file_bytes: fs::metadata(args.model.join("model.safetensors"))?.len(),
        baseline_payload_bytes: baseline.model_payload_bytes(),
        cells,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_word_scoring_uses_every_token_and_isolates_branches() {
        let context = vec![99];
        let forward = |token: u32, position: usize, branch: &mut Vec<u32>| {
            assert_eq!(position, branch.len());
            branch.push(token);
            Ok(vec![0., 0., 0., 0.])
        };
        let one = word_score(&[1], &[0.; 4], &context, 1, &[0, 3], forward).unwrap();
        let two = word_score(&[1, 2], &[0.; 4], &context, 1, &[0, 3], forward).unwrap();
        assert!((one.exp() - 0.125).abs() < 1e-9);
        assert!((two.exp() - 0.03125).abs() < 1e-9);
        assert_eq!(context, vec![99]);
        assert!(word_score(&[], &[0.; 4], &context, 1, &[0], forward).is_err());
        assert!(
            word_score(&[1], &[0.; 4], &context, 1, &[0], |_, _, _| bail!(
                "fake inference failure"
            ))
            .is_err()
        );
    }
    #[test]
    fn unicode_boundaries_and_sentence_context() {
        let fixtures = BTreeMap::from([(
            "test".into(),
            vec![
                "Earlier sentence. Café’s résumé includes naïveté and multilingual communication."
                    .into(),
            ],
        )]);
        let cases = queries(&fixtures, &BTreeSet::new(), 1).unwrap();
        assert_eq!(cases.len(), 5);
        assert!(cases.iter().all(|q| !q.before.contains("sentence café")));
        assert_eq!(
            queries(&fixtures, &BTreeSet::new(), 1)
                .unwrap()
                .iter()
                .map(|q| &q.key)
                .collect::<Vec<_>>(),
            cases.iter().map(|q| &q.key).collect::<Vec<_>>()
        );
        assert!(queries(&fixtures, &BTreeSet::from(["earlier sentence".into()]), 1).is_err());
        assert!(queries(&fixtures, &BTreeSet::new(), 0).is_err());
        assert!(boundary(" next"));
        assert!(boundary("."));
        assert!(!boundary("ing"));
        assert!(!boundary("'s"));
        assert!(!boundary(""));
    }
    #[test]
    fn probabilities_and_percentiles() {
        let logits = [0., 0., 0., 0.];
        assert!((log_prob(&logits, 1).exp() - 0.25).abs() < 1e-9);
        assert!((log_boundary(&logits, &[0, 2]).exp() - 0.5).abs() < 1e-9);
        assert_eq!(percentile(&[1., 2., 3., 4.], 95), 4.);
    }
}
