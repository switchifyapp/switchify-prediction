# Corpus source benchmark

This benchmark compares the data used by the existing predictor. It keeps the
unigram/bigram/trigram scoring weights at 0.1/0.3/0.6 and disables personal
learning. It does not change the production model.

## Comparisons

The ten models in `corpus-benchmark-policy.json` are fixed before scoring:

- The current production blend.
- Tatoeba/WorldAlphabets, Taskmaster-1 USER turns, AAC-style crowdsourced text,
  and the selected 20,000 spoken OANC sentences, each trained separately.
- Four blends with one source removed. Other sources keep their production
  counts. Sentences shared between sources keep the remaining source's counts.
- The production blend with all eligible spoken OANC sentences instead of the
  20,000-sentence sample.

The production blend weights are Tatoeba 3, Taskmaster 1, AAC 10, OANC 1.
Standalone models use each unique sentence once. Source sizes differ, so their
scores do not isolate data quality at a fixed training budget. Removing a source
also changes the relative proportions of the remaining sources. Ablations show
the effect of removing the source at the current weights, not its best possible
weight or its value in every other mixture.

All six existing development/test sets are excluded from every training model.
All models see the same evaluation sentences within a suite. Report AAC,
general English and Taskmaster-style conversation separately, without collapsing
them into a winner. General and Taskmaster tests resemble their source domains.
These are known regression sets, not a new sample of real AAC users. The AAC
source consists of imagined communication written by crowd workers.

## Run

Use Rust 1.97.1 and Python 3.9 or later. From the repository root:

```sh
cargo build --release --locked
python scripts/aac_experiment.py --prepare-only
python scripts/corpus_benchmark.py --output artifacts/corpus-benchmark-dev
```

Preparation downloads and verifies the pinned source files. If the prepared
inputs already exist, the benchmark itself does not download or overwrite them.
It verifies the prepared source hashes, reconstructs the exact production
training counts, and checks the rebuilt production model's logical hash.
`--prepare-only` writes training inputs and checks their separation without
building models. Every run requires a fresh output directory to prevent stale
results from being mistaken for new measurements.

Development is the default. To run the same frozen matrix on the existing test
sets, choose a separate directory and pass `--partition test`. Future data or
weight selection should use development, with test confirmation recorded
separately. This benchmark does not automatically select or promote a model.

`report.json` records input, policy, script and executable hashes, source sizes,
model hashes and the complete CLI measurements. A partial report has
`complete: false`. Completed runs include source manifests, attribution and
corpus notices. Retain these when redistributing any model.

The main comparison is top-five accuracy after two typed graphemes. The report
also contains top-one/top-five accuracy after zero through four graphemes,
vocabulary coverage, modeled selection savings, model bytes, vocabulary and
n-gram counts, and warm p50/p95 latency. Timing is descriptive for this machine;
run on representative hardware before drawing deployment conclusions. Modeled
savings do not measure actual switch operation.

Google Books counts and wordfreq are not included in this first source matrix.
They require separate data import and mixture policies before comparison. Google
Books should be assessed as an added source against this same baseline and
development suites, with its publication-year filter and retained n-gram coverage
recorded explicitly.
