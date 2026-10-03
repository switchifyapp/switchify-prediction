# Switchify Prediction

Offline English word completion using a Rust library and bundled SQLite. The
read-only baseline and a separate personal database feed an in-memory n-gram
predictor. No keystroke capture, background monitoring, abbreviations or Switchify
PC integration is included.

The default model is **en-aac-oanc-v1**: 21,674 words and 539,151 n-grams. It adds
AAC-focused communication and a controlled spoken-English sample to the earlier
WorldAlphabets/Taskmaster model. Its frozen AAC-style test top-five accuracy after
two characters is **85.17%**, compared with 78.11% for the earlier model. General
English improves from 65.59% to 66.40%; Taskmaster falls from 86.56% to 86.24%.
These are offline corpus measurements, not observed AAC-user outcomes.

## Download and use

Choose a successful run for the desired commit in
[GitHub Actions](https://github.com/switchifyapp/switchify-prediction/actions).
Download `english-prediction-database` and the `cli-...` artifact for your OS.
Each contains a ZIP and archive checksum. The CLI ZIP filename identifies its
architecture. Extract the CLI and model into **separate directories**, verify
the checksums, and run each bundle's `python verify_bundle.py`.

```sh
/path/to/switchify-prediction validate --database /path/to/model/english.sqlite --production
/path/to/switchify-prediction predict --baseline /path/to/model/english.sqlite \
  --before 'I need ' --prefix 'he'
```

On Windows use `switchify-prediction.exe`. The executable includes SQLite; no
server or Rust installation is needed. CLI packages are unsigned, not desktop app
installers. Actions downloads require GitHub login and expire after 90 days.
Approved version tags run the same validation and prepare a draft release for
durable distribution; see [release and operation guidance](docs/production.md).
No network access is needed for prediction or learning.

Prediction returns JSON with `word` and `score`. Defaults are five suggestions
after two Unicode grapheme clusters. Override with `--limit` and `--min-chars`;
use `--min-chars 0` for next-word prediction. Pass preceding completed text in
`--before` and the unfinished current word separately in `--prefix`. Suggestions
are lowercase; insertion and capitalization are the application's responsibility.
Scores are interpolated probabilities before prefix filtering, not renormalized
over the returned suggestions or calibrated confidence estimates.

## Build reproducibly

Requires Rust **1.97.1**, a C compiler for bundled SQLite, Python 3.9+ and curl.
Run from this checkout:

```sh
cargo build --release --locked
python3 scripts/artifacts.py
```

This fetches checksum-pinned public data, prepares the frozen partitions, builds
the comparison models, evaluates development and test, checks the frozen quality
results, then produces the production bundle in `artifacts/model-bundle` and
unpacked files in `artifacts/production`. It never reads personal data. The OANC
download is about 655 MB. Cached verified sources allow offline reruns.

To prepare and build without repeating the quality benchmark:

```sh
python3 scripts/aac_experiment.py --prepare-only
mkdir -p artifacts
target/release/switchify-prediction build --output artifacts/english.sqlite
```

The default `build` requires the exact prepared training text and verifies the
model fingerprint before publishing. It embeds full source and policy provenance
in SQLite. `build --input training.txt --output custom.sqlite` instead builds a
custom model from explicit UTF-8 text. Existing output files are never overwritten.
Plain `validate` checks any compatible model; `validate --production` additionally
requires this binary's exact shipped model and provenance. A custom build does
not inherit the official baseline's evaluation or distribution permissions.

## Personal learning

```sh
switchify-prediction import --baseline english.sqlite \
  --personal personal.sqlite --input training.txt
printf 'I would like watermelon.' | switchify-prediction learn \
  --baseline english.sqlite --personal personal.sqlite
switchify-prediction predict --baseline english.sqlite --personal personal.sqlite \
  --before 'I would like ' --prefix 'wa'
switchify-prediction reset-personal --baseline english.sqlite --personal personal.sqlite
```

Imports deduplicate exact UTF-8 content by SHA-256. Every `learn` call represents
one completed, non-overlapping segment; never repeatedly submit a growing typing
buffer. Context does not cross calls, newlines, `.`, `!` or `?`. Internal
apostrophes are preserved, curly apostrophes normalize to ASCII, and keys use
Unicode NFC and lowercase. Other punctuation separates words. Numbers are ignored.

Missing personal databases are initialized atomically; invalid existing databases
are rejected. Writes are transactional, including validation of the new in-memory
state before commit. Multiple writers serialize through SQLite with a five-second
busy timeout. Each predictor uses a snapshot; call `refresh_personal()` to see
another instance's updates. Successful writes and duplicate imports refresh the
calling instance. Errors preserve its previous snapshot.

Personal n-grams can reveal sensitive content and are stored unencrypted. Keep
these files in an access-controlled application data directory, outside source
control and distribution bundles. Reset removes logical counts and import markers,
not forensic copies or backups. The library does not upload or log training text.

## Rust API

```rust,no_run
use std::path::Path;
use switchify_prediction::{Options, Predictor, Result};

fn example() -> Result<()> {
    let mut predictor = Predictor::open(
        Path::new("english.sqlite"), Some(Path::new("personal.sqlite")))?;
    let suggestions = predictor.predict("I would like ", "wa", Options::default());
    predictor.learn("I would like watermelon.")?;
    predictor.refresh_personal()?;
    assert!(suggestions.len() <= 5);
    Ok(())
}
```

`build`, `validate`, `production::build_english`, `production::validate_english`,
`Predictor::import` and `Predictor::reset_personal` cover the other operations.
Fallible operations return typed `Error` values. Reuse a long-lived predictor;
keep loading, imports, learning, refresh and resets on a worker thread. Prediction
uses memory only. The model uses unigram/bigram/trigram weights **0.1/0.3/0.6**,
renormalized over available contexts, and combines baseline counts with **5×**
personal counts before probability calculation. Prefix matches rank by score,
then alphabetically. Schema v1 and existing personal databases remain compatible.

## Validation and model selection

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo audit --deny warnings
```

Install the CI-pinned auditing tool with
`cargo install cargo-audit --locked --version 0.22.0`. CI runs these checks and
packaged CLI/model smoke tests on macOS, Windows and Linux (dependency audit runs
on Linux). Tests use temporary files and synthetic personal text.

[production-model.json](production-model.json) pins the accepted training and
logical count hashes, parameters, source records and partition hashes.
[production-quality.json](production-quality.json) preserves the accepted quality
measurements. Production artifacts require both AAC development/test gates to
pass and every frozen accuracy/coverage/selection result to match. Warm latency
is reported against a 20 ms p95 target, not used as a timing-sensitive CI gate.
The accepted candidate measured below 6 ms p95 on an Apple M2 Max.

The training mixture is the prior 3× WorldAlphabets / 1× Taskmaster USER model,
plus 10× each of 4,299 eligible unique AAC training sentences and 1× each of
20,000 hash-selected short OANC spoken utterances. Held-out text stays excluded.
Official AAC worker and Taskmaster conversation splits are preserved; exact
normalized sentence overlap is removed across training and evaluation.

The model bundle contains `quality-report.json`, validation, source/partition
records, checksums and attribution. The evaluated candidate's file hash differs
from the production file because production adds full provenance; their logical
counts are identical and checked. The selection-saving metric is an optimistic
rank-sensitive proxy, **not measured switch savings**. AAC is imagined by crowd
workers, OANC is older American speech and Taskmaster is simulated task dialogue.
Near-duplicates, domain bias, misspellings and inappropriate suggestions can
remain. English alone is supported.

The [AAC experiment protocol](docs/aac-experiment.md) and earlier
[conversational results](docs/quality-results.md) document model-selection history.
Switchify is a general-purpose typing app. The separate
[neural experiment](docs/general-neural-protocol.md) tests messages, email,
documents and search-style text equally, without changing the production model.
The `prepare` and `evaluate` commands retain the earlier comparison workflows;
`prepare` alone does not prepare the promoted production corpus. Use the commands
above for production. A new model-selection exercise needs a new protocol; these
test sets are now known regression sets.

New code is MIT licensed. Corpus licences are separate: retain
[ATTRIBUTION.md](ATTRIBUTION.md), source manifests and `corpus-notices` with model
distributions. [Production guidance](docs/production.md) covers the recorded
licensing evidence, OANC's differing historical/current notices, private-data
handling, backup, upgrades, rollback and releases.
