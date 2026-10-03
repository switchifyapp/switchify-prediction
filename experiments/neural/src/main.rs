use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use candle_transformers::models::quantized_llama::ModelWeights;
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
    /// Explicit locally converted GGUF; its digest is mandatory.
    #[arg(long, requires = "gguf_sha256")]
    gguf: Option<PathBuf>,
    #[arg(long, requires = "gguf")]
    gguf_sha256: Option<String>,
    /// Measure a second request with the same context, retaining only one KV entry.
    #[arg(long)]
    cache_probe: bool,
    /// Validate the F32 GGUF conversion against the original loader before scoring.
    #[arg(long)]
    validate_conversion: bool,
    /// Export only the frozen synthetic workload for external decoder comparisons.
    #[arg(long)]
    export_queries: bool,
    /// Evaluate candidate branches together, padding finished lanes only.
    #[arg(long)]
    batch_candidates: bool,
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

fn batch_word_scores<S: Clone>(
    tokens: &[Vec<u32>],
    first_logits: &[Vec<f32>],
    cache: &S,
    position: usize,
    boundary_ids: &[usize],
    mut forward: impl FnMut(&[u32], usize, &mut S) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<f64>> {
    ensure!(
        !tokens.is_empty() && tokens.iter().all(|t| !t.is_empty()),
        "Empty candidate tokenization"
    );
    ensure!(tokens.len() == first_logits.len(), "Invalid batch width");
    let mut branch = cache.clone();
    let mut logits = first_logits.to_vec();
    let mut scores = vec![0.; tokens.len()];
    for index in 0..tokens.iter().map(Vec::len).max().unwrap() {
        let input: Vec<_> = tokens
            .iter()
            .map(|t| t.get(index).copied().unwrap_or(0))
            .collect();
        for (lane, t) in tokens.iter().enumerate() {
            if index < t.len() {
                scores[lane] += log_prob(&logits[lane], t[index] as usize);
            }
        }
        logits = forward(&input, position + index, &mut branch)?;
        ensure!(logits.len() == tokens.len(), "Invalid output batch width");
        for (lane, t) in tokens.iter().enumerate() {
            if index + 1 == t.len() {
                scores[lane] += log_boundary(&logits[lane], boundary_ids);
            }
        }
    }
    ensure!(
        scores.iter().all(|s| s.is_finite()),
        "Non-finite batched score"
    );
    Ok(scores)
}

enum Engine {
    Float(Llama),
    Quantized,
}

#[derive(Clone)]
enum State {
    Float(Cache),
    Quantized(ModelWeights),
}

#[derive(Clone)]
struct Prepared {
    ids: Vec<u32>,
    lanes: usize,
    state: State,
    logits: Vec<Vec<f32>>,
}

fn refresh_context<T>(
    entry: &mut Option<T>,
    matches: impl FnOnce(&T) -> bool,
    build: impl FnOnce() -> Result<T>,
) -> Result<()> {
    if !entry.as_ref().is_some_and(matches) {
        *entry = None;
        *entry = Some(build()?);
    }
    Ok(())
}

struct Neural {
    model: Engine,
    tokenizer: Tokenizer,
    empty_cache: State,
    boundary_ids: Vec<usize>,
    prepared: Option<Prepared>,
    batch_candidates: bool,
}

impl Neural {
    fn load(path: &Path, gguf: Option<&Path>) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let cfg: LlamaConfig = serde_json::from_slice(&fs::read(path.join("config.json"))?)?;
        let config = cfg.into_config(false);
        ensure!(
            config.bos_token_id == Some(0),
            "Expected pinned SmolLM2 BOS token"
        );
        let (model, empty_cache) = if let Some(path) = gguf {
            let mut reader = fs::File::open(path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
            let model = ModelWeights::from_gguf(content, &mut reader, &Device::Cpu)?;
            (Engine::Quantized, State::Quantized(model))
        } else {
            let cache = Cache::new(true, DType::F32, &config, &Device::Cpu)?;
            // Safe loader owns its bytes; no mmap lifetime or externally mutable mapping.
            let weights = fs::read(path.join("model.safetensors"))?;
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &Device::Cpu)?;
            (
                Engine::Float(Llama::load(vb, &config)?),
                State::Float(cache),
            )
        };
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
            prepared: None,
            batch_candidates: false,
        })
    }

    fn forward(&self, ids: &[u32], position: usize, cache: &mut State) -> Result<Vec<f32>> {
        let input = Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?;
        Ok(self
            .forward_tensor(&input, position, cache)?
            .squeeze(0)?
            .to_vec1()?)
    }

    fn forward_tensor(&self, input: &Tensor, position: usize, cache: &mut State) -> Result<Tensor> {
        let logits = match (&self.model, cache) {
            (Engine::Float(model), State::Float(cache)) => model.forward(input, position, cache)?,
            (Engine::Quantized, State::Quantized(model)) => model.forward(input, position)?,
            _ => bail!("Mismatched inference state"),
        };
        Ok(logits)
    }

    fn rank(&mut self, before: &str, candidates: &[String]) -> Result<Vec<String>> {
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
        let lanes = if self.batch_candidates {
            candidates.len()
        } else {
            1
        };
        let mut prepared = self.prepared.take();
        refresh_context(
            &mut prepared,
            |p| p.ids == context && p.lanes == lanes,
            || {
                let mut state = self.empty_cache.clone();
                let logits = if lanes > 1 && matches!(self.model, Engine::Quantized) {
                    // Candle 0.11 GGUF's output slice is non-contiguous for B>1,
                    // T>1. Single-token prefill keeps its public API contiguous.
                    let mut last = None;
                    for (position, token) in context.iter().enumerate() {
                        let input = Tensor::new(&[*token], &Device::Cpu)?
                            .unsqueeze(0)?
                            .repeat((lanes, 1))?;
                        last = Some(self.forward_tensor(&input, position, &mut state)?);
                    }
                    last.context("Empty context")?.to_vec2()?
                } else {
                    let input = Tensor::new(context.as_slice(), &Device::Cpu)?
                        .unsqueeze(0)?
                        .repeat((lanes, 1))?;
                    self.forward_tensor(&input, 0, &mut state)?.to_vec2()?
                };
                Ok(Prepared {
                    ids: context.clone(),
                    lanes,
                    state,
                    logits,
                })
            },
        )?;
        self.prepared = prepared;
        let prepared = self.prepared.as_ref().context("Missing context state")?;
        let cache = &prepared.state;
        let first_logits = &prepared.logits;
        let mut scored = Vec::new();
        let mut all_tokens = Vec::new();
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
            if self.batch_candidates {
                all_tokens.push(tokens.to_vec());
                continue;
            }
            let score = word_score(
                tokens,
                &first_logits[0],
                cache,
                context.len(),
                &self.boundary_ids,
                |token, position, branch| self.forward(&[token], position, branch),
            )?;
            scored.push((score, rank, word.clone()));
        }
        if self.batch_candidates {
            let scores = batch_word_scores(
                &all_tokens,
                first_logits,
                cache,
                context.len(),
                &self.boundary_ids,
                |ids, position, state| {
                    let input = Tensor::new(ids, &Device::Cpu)?.unsqueeze(1)?;
                    Ok(self.forward_tensor(&input, position, state)?.to_vec2()?)
                },
            )?;
            scored.extend(
                scores
                    .into_iter()
                    .zip(candidates)
                    .enumerate()
                    .map(|(rank, (score, word))| (score, rank, word.clone())),
            );
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(scored.into_iter().take(5).map(|x| x.2).collect())
    }
}

#[derive(Clone, Debug, Serialize)]
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
    gguf_sha256: Option<String>,
    cache_hit: Option<Timing>,
    prediction_sha256: String,
    batch_candidates: bool,
}

#[derive(Serialize)]
struct Timing {
    queries: usize,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn validate_conversion(path: &Path, converted: &Neural) -> Result<()> {
    let original = Neural::load(path, None)?;
    for text in [
        "",
        "I would like to",
        "Please send the documents tomorrow",
        "Café résumé",
    ] {
        let encoded = original
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        let mut ids = vec![0];
        ids.extend_from_slice(encoded.get_ids());
        let mut left = original.empty_cache.clone();
        let mut right = converted.empty_cache.clone();
        // Check prefill and a continuation, exercising positional rotation and KV caches.
        for (tokens, position) in [(ids.as_slice(), 0), (&[42_u32][..], ids.len())] {
            let a = original.forward(tokens, position, &mut left)?;
            let b = converted.forward(tokens, position, &mut right)?;
            let error = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            ensure!(
                error < 0.002,
                "F32 conversion logit error {error} exceeds tolerance"
            );
        }
    }
    eprintln!("F32 conversion prefill and continuation parity passed");
    Ok(())
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
    if args.export_queries {
        fs::write(&args.output, serde_json::to_vec_pretty(&cases)?)?;
        return Ok(());
    }
    verify_model(&args.model, &args.manifest)?;
    if let Some(path) = &args.gguf {
        ensure!(
            Some(sha(path)?) == args.gguf_sha256,
            "GGUF checksum mismatch"
        );
        ensure!(
            matches!(args.mode, Mode::Neural),
            "GGUF requires neural mode"
        );
    }
    let start = Instant::now();
    let baseline = Predictor::open(&args.baseline, None)?;
    let mut neural = match args.mode {
        Mode::Baseline => None,
        Mode::Neural => Some(Neural::load(&args.model, args.gguf.as_deref())?),
    };
    let cold_load_ms = start.elapsed().as_secs_f64() * 1000.;
    if let Some(neural) = &mut neural {
        neural.batch_candidates = args.batch_candidates;
    }
    if args.validate_conversion {
        ensure!(args.gguf.is_some(), "Conversion validation requires GGUF");
        validate_conversion(
            &args.model,
            neural.as_ref().context("Requires neural mode")?,
        )?;
    }
    let mut predict = |q: &Query, clear: bool| -> Result<(Vec<String>, Vec<String>)> {
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
        let words = match &mut neural {
            Some(n) => {
                if clear {
                    n.prepared = None;
                }
                n.rank(&q.before, &candidates)?
            }
            None => candidates.iter().take(5).cloned().collect(),
        };
        ensure!(
            words.iter().all(|w| w.starts_with(&q.prefix)),
            "Invalid prefix completion"
        );
        Ok((words, candidates))
    };
    // Untimed prime. Each measured request starts with a fresh neural context cache.
    predict(&cases[0], true)?;
    let mut times = Vec::new();
    let mut hits = Vec::new();
    let mut predictions = Sha256::new();
    let mut cells: BTreeMap<String, BTreeMap<usize, Cell>> = BTreeMap::new();
    for (index, q) in cases.iter().enumerate() {
        let start = Instant::now();
        let (words, candidates) = predict(q, true)?;
        times.push(start.elapsed().as_secs_f64() * 1000.);
        predictions.update(serde_json::to_vec(&(&q.key, &words))?);
        if args.cache_probe {
            let start = Instant::now();
            let (cached, _) = predict(q, false)?;
            hits.push(start.elapsed().as_secs_f64() * 1000.);
            ensure!(cached == words, "Cached inference changed results");
        }
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
    hits.sort_by(f64::total_cmp);
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
        model_file_bytes: fs::metadata(
            args.gguf
                .clone()
                .unwrap_or_else(|| args.model.join("model.safetensors")),
        )?
        .len(),
        baseline_payload_bytes: baseline.model_payload_bytes(),
        cells,
        gguf_sha256: args.gguf_sha256,
        cache_hit: (!hits.is_empty()).then(|| Timing {
            queries: hits.len(),
            median_ms: percentile(&hits, 50),
            p95_ms: percentile(&hits, 95),
            max_ms: *hits.last().unwrap(),
        }),
        prediction_sha256: format!("{:x}", predictions.finalize()),
        batch_candidates: args.batch_candidates,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_scoring_ignores_padding_after_each_word_boundary() {
        let cache = vec![99];
        let scores = batch_word_scores(
            &[vec![1], vec![1, 2]],
            &[vec![0.; 4], vec![0.; 4]],
            &cache,
            1,
            &[0, 3],
            |ids, position, state| {
                assert_eq!(state.len(), position);
                if position == 1 {
                    assert_eq!(ids, [1, 1]);
                } else {
                    assert_eq!(ids, [0, 2]);
                }
                state.push(0);
                Ok(vec![vec![0.; 4]; 2])
            },
        )
        .unwrap();
        assert!((scores[0].exp() - 0.125).abs() < 1e-9);
        assert!((scores[1].exp() - 0.03125).abs() < 1e-9);
        assert_eq!(cache, [99]);
    }
    #[test]
    fn context_changes_and_failed_rebuilds_discard_stale_state() {
        let mut entry = Some((vec![0, 1], 7));
        refresh_context(&mut entry, |p| p.0 == [0, 1], || bail!("must reuse")).unwrap();
        assert_eq!(entry.as_ref().unwrap().1, 7);
        refresh_context(&mut entry, |p| p.0 == [0, 2], || Ok((vec![0, 2], 9))).unwrap();
        assert_eq!(entry.as_ref().unwrap().1, 9);
        assert!(refresh_context(&mut entry, |p| p.0 == [0, 3], || bail!("fake error")).is_err());
        assert!(entry.is_none());
        refresh_context(&mut entry, |_| false, || Ok((vec![0], 1))).unwrap();
        assert_eq!(entry.unwrap(), (vec![0], 1));
    }
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
