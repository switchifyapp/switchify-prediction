# General-purpose neural experiment

Frozen 2026-10-03 before scoring. Switchify is a general-purpose typing app.
Messages, email, documents and search-style text receive equal weight. AAC is
not a selection objective. The existing released model is a control, including
its historical training mixture; this experiment does not endorse that mixture.

## Fixed comparison

Compare the released predictor with a SmolLM2-135M base-model reranker using
Candle 0.11.0, CPU F32, four Rayon threads and no GPU, personal learning or
network access during inference. This initial implementation is unquantized.
Weights, tokenizer and configuration are pinned by SHA-256 and upstream commit.
The experiment is a separate, unpublished Rust crate. Production dependencies,
APIs and model formats stay unchanged.

Both modes request the existing engine's top eight prefix-matching words. The
control returns its first five. The neural mode scores each entire candidate's
token sequence, followed by the probability of a word boundary, and returns
the five highest scores. It uses BOS plus the last 63 tokens of preceding text
within the current sentence. It does not score only the first subword, normalize
by word length, mix frequency scales or learn from targets. Ties retain baseline
order. Each query starts with an empty KV cache; the context cache is shared only
between candidate branches within that query. Prefixes filter candidates before
neural scoring. This is a bounded reranker, not unrestricted neural generation;
report top-eight candidate coverage as its accuracy ceiling.

Boundary probability sums the next-token probabilities for EOS and tokens whose
decoded text starts with whitespace, a number, or one of `. ! ? ; : , ( ) [ ] { }
" - /`. Apostrophe continuations remain inside a word. This is an explicit
approximation, not an exact tokenizer-independent word probability.

## Workload and decision

The checked-in fixtures contain 24 newly authored synthetic examples in each
of four domains. They are diagnostic fixtures, not a representative user corpus
or independent evidence of real-world accuracy. Normalize with the existing
Unicode/sentence tokenizer. Reject exact full-sentence overlap with the pinned
production training file. Unknown overlap with neural pretraining remains.

For each domain and each prefix length 0 through 4 Unicode graphemes, hash-sort
eligible target positions and take 64 distinct queries, excluding words already
complete at that prefix. The fixed workload has 1,280 queries per mode. Duplicate
preceding-text/prefix/target triples within a cell are removed. All words before
the target are supplied identically; each predictor uses its supported context.
Lowercasing and removing punctuation limit conclusions about realistic writing.

Report top-one, top-five and candidate coverage per domain/prefix, cold load,
warm median, p95, maximum, failures, model file size and process peak memory.
Prime once without timing. Abort on inference errors instead of silently falling
back. A smaller `--per-cell` run is calibration only, not a result used to tune
the model, shortlist or ranking rule.

A promising result requires higher mean top-five accuracy over prefixes 0,1,2
and all four domains, no domain/prefix loss above one percentage point there,
and warm p95 below 20 ms. Do not tune after seeing fixture scores or promote a
model automatically. Describe tradeoffs even if the criterion fails. Windows
is the local benchmark platform; macOS build/tests do not establish macOS speed.

Investigate FUTO's official model integration separately. Do not equate generic
GGUF loading with implementing its custom keyboard prompts and tokenizer rules.
No FUTO accuracy claim without running its actual prediction path.

## Reproduction

Use Rust 1.97.1 and Python 3. Prepare the unchanged production partitions with
`python scripts/aac_experiment.py --prepare-only` if they are not cached. Extract
`english.sqlite` from the model bundle on the
[v0.1.0 release](https://github.com/switchifyapp/switchify-prediction/releases/tag/v0.1.0).
Then run from the repository root:

```sh
cargo build --release --locked --manifest-path experiments/neural/Cargo.toml --target-dir target/neural
python scripts/general_neural_experiment.py --baseline /path/to/english.sqlite --output artifacts/general-neural
```

The Python command downloads and verifies the pinned model before launching the
offline Rust executables. Use a fresh output directory for each full run.
It records aggregate results and Windows process peak working set. Other
platforms report memory as unavailable, not zero. No captured user text is used.

Run formatting, Clippy and tests with the separate crate's manifest in addition
to the repository checks. Its lockfile contains Candle's transitive `paste`
1.0.15 dependency, which has the unmaintained-package advisory RUSTSEC-2024-0436.
The production dependency audit remains unchanged. For this unpublished
experiment only, audit the separate lockfile with that single advisory exception:

```sh
cargo audit --file experiments/neural/Cargo.lock --deny warnings --ignore RUSTSEC-2024-0436
```

This exception does not assert that the dependency is maintained. Reassess the
dependency before considering production integration.
