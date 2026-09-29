# Switchify Prediction

Offline English word completion using a Rust library and SQLite databases. A
read-only language baseline and a separate personal database feed an in-memory
word n-gram predictor. Abbreviations and application integration are out of scope.

## Build and try it

Requires Rust **1.97.1**, a C compiler for bundled SQLite, and Python 3 for fetching
the public corpus and producing artifacts. Run commands from this repository.

```sh
python3 scripts/fetch_corpus.py
cargo build --release --locked
mkdir -p artifacts
target/release/switchify-prediction build --output artifacts/english.sqlite
target/release/switchify-prediction predict --baseline artifacts/english.sqlite \
  --before 'I would like ' --prefix 'wa'
target/release/switchify-prediction validate --database artifacts/english.sqlite
```

On Windows use `python` and `target/release/switchify-prediction.exe`; create the
output directory with PowerShell's `New-Item -ItemType Directory -Force artifacts`.
After fetching, builds, predictions and learning require no network connection.
The default build verifies `data/en.txt` against the pinned source manifest.
`build --input training.txt --output custom.sqlite` builds from explicitly supplied
UTF-8 text instead. Existing outputs are never replaced by the `build` command.

Prediction returns JSON objects with `word` and `score`. Defaults are five results
after two grapheme clusters; use `--limit 3 --min-chars 1` to change them. Set
`--min-chars 0` and omit `--prefix` for next-word prediction. Pass **only completed
preceding text** in `--before`, with the partial current word in `--prefix`.
Suggestions are normalized lowercase; application-specific capitalization and
insertion remain the caller's responsibility. Scores are interpolated model
probabilities before prefix filtering, not probabilities renormalized over results.

## Personal writing and learning

```sh
target/release/switchify-prediction import --baseline artifacts/english.sqlite \
  --personal personal.sqlite --input training.txt
printf 'I would like watermelon.' | target/release/switchify-prediction learn \
  --baseline artifacts/english.sqlite --personal personal.sqlite
target/release/switchify-prediction predict --baseline artifacts/english.sqlite \
  --personal personal.sqlite --before 'I would like ' --prefix 'wa'
target/release/switchify-prediction reset-personal --baseline artifacts/english.sqlite \
  --personal personal.sqlite
```

Imports are deduplicated by the exact UTF-8 content's SHA-256. Every `learn` call
adds one independent, completed segment: callers must not submit overlapping text
or repeatedly submit a growing typing buffer. The library does not capture input.
Sentence context never crosses calls, newlines, `.`, `!`, or `?`. Apostrophes inside
words are retained, curly apostrophes normalize to ASCII, and lookup keys use
Unicode NFC and lowercase. Other punctuation separates words. Numbers are ignored.

A missing personal database is initialized; an existing invalid database is rejected.
Reset removes learned counts and import markers, permitting later reimport. It is
not a secure-erasure operation. Personal text is not retained verbatim as documents,
but learned words and n-grams are sensitive and stored unencrypted. The project
never searches for personal files, uploads them, or logs learned text. Keep them
outside Git; generated databases, `data/`, `personal/`, and `training.txt` are ignored.
Use one long-lived predictor as the learning owner; other open predictors have
snapshots and must reopen to see its updates.

## Rust API and storage

```rust,no_run
use std::path::Path;
use switchify_prediction::{Options, Predictor, Result};

fn example() -> Result<()> {
    let mut predictor = Predictor::open(
        Path::new("english.sqlite"), Some(Path::new("personal.sqlite")))?;
    let results = predictor.predict("I would like ", "wa", Options::default());
    predictor.learn("I would like watermelon.")?;
    assert!(results.len() <= 5);
    Ok(())
}
```

`build`, `validate`, `evaluation::evaluate`, `Predictor::import`, and
`Predictor::reset_personal` cover the remaining operations. Fallible operations
return typed `Error` values. Prediction operates on an already loaded model.
SQLite schema v1 stores metadata, a vocabulary, n-gram counts keyed by JSON context
and word, and personal import hashes. The database header's `user_version` rejects
unknown schemas; only language `en` is supported. Initial release has no migrations.

The model uses unigram/bigram/trigram weights **0.1/0.3/0.6**, dropping unavailable
contexts and renormalizing the weights. Baseline counts and **5× personal counts**
are combined before computing probabilities. Candidate prefix matches are ranked
by score, then alphabetically for ties. The baseline is immutable and replaceable
without overwriting the personal database. Personal writes use transactions; the
in-memory model reloads after successful writes. Keep disk access off an app's UI
thread when integrating this library.

## Validation and evaluation

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --release --locked
python3 scripts/artifacts.py
```

CI runs checks on Linux, macOS and Windows. Tests use synthetic text and temporary
files, never real input adapters. They cover learning persistence and rollback,
duplicate imports, deterministic builds, normalization, sentence boundaries,
backoff, thresholds, bad databases, and the CLI.

Evaluation deduplicates normalized sentences, sorts them by SHA-256, and holds out
the first `ceil(N/10)` sentences. Held-out sentences never enter training. For each
held-out word and prefix length 0–4, it reports top-1/top-5 accuracy, vocabulary
coverage, and unigram-only accuracy. Words already fully typed at that prefix
length are excluded; results therefore have different denominators. It measures
all warm contextual queries, including next-word queries, and reports p50/p95,
cold predictor load time, evaluation database size, estimated model payload bytes,
and hardware. The artifact script also records child-process peak RSS on Unix;
that includes evaluation and corpus-building overhead, not just model memory.
The target is warm p95 below **20 ms**, reported rather than enforced in shared CI.

The artifact's `english.sqlite` uses the **full corpus**. Its accompanying
`evaluation.json` describes a **separate held-out evaluation model**. The artifact
script regenerates only `artifacts/english.sqlite`; use other paths for custom data.
The corpus is a small general sentence collection, not a personalized conversation
model. Accuracy here is not an estimate for every user or evidence of superiority
over an existing neural predictor. Other languages need explicit tokenization and
evaluation work before being supported.

## GitHub downloads

Open [Actions](https://github.com/switchifyapp/switchify-prediction/actions), choose
a successful **CI** run for the desired commit, and download
**english-prediction-database** from its artifacts. GitHub login is required for
artifact downloads. Retention is 90 days; a manual workflow run can rebuild them.
The archive includes `english.sqlite`, `SHA256SUMS`, validation and evaluation JSON,
source provenance, corpus attribution and the software license. Verify the files
with `shasum -a 256 -c SHA256SUMS` (macOS) or `sha256sum -c SHA256SUMS` (Linux).

New software is MIT licensed. Corpus rights are separate: see [ATTRIBUTION.md](ATTRIBUTION.md)
and [source-manifest.json](source-manifest.json). The upstream English manifest
identifies a Tatoeba CC0 subset but marks it `verify: false`; its attribution and
license note are preserved rather than treating the corpus as independently audited.
