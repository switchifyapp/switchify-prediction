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
target/release/switchify-prediction prepare
mkdir -p artifacts
target/release/switchify-prediction build --output artifacts/english.sqlite
target/release/switchify-prediction predict --baseline artifacts/english.sqlite \
  --before 'I would like ' --prefix 'wa'
target/release/switchify-prediction validate --database artifacts/english.sqlite
```

On Windows use `python` and `target/release/switchify-prediction.exe`; create the
output directory with PowerShell's `New-Item -ItemType Directory -Force artifacts`.
After fetching, builds, predictions and learning require no network connection.
The default build verifies `data/prepared/candidate.txt` against the pinned source
manifest. `prepare` verifies both public sources and derives the frozen training,
development, and test partitions using the predictor’s own tokenizer.
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

`build`, `validate`, `corpus::prepare`, `evaluation::score_database`, `Predictor::import`, and
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

The current baseline combines **3× WorldAlphabets training sentences** with **1×
unique Taskmaster-1 USER training sentences**. Taskmaster contributes human-written
requests covering six task domains. Its assistant turns are excluded. The ranking
algorithm is unchanged; personal learning remains separate and off during evaluation.

The reproducible quality pipeline is:

```sh
python3 scripts/quality.py --development-only
python3 scripts/artifacts.py
```

`quality-policy.json` freezes the mixture and acceptance criteria. The development
comparison must gain at least five percentage points of conversational top-five
accuracy after two characters, lose no more than two points on general English,
and improve or match the conversation selection-saving proxy. Timing is reported
against the 20 ms p95 target, not used as a flaky CI gate. Only a passing development
candidate proceeds to the separate test evaluation and publication.

WorldAlphabets uses normalized sentence deduplication and SHA-256 ordering, with
80/10/10 training/development/test partitions (rounded held-out sizes). Its test
partition is the same known regression set used in v1. Taskmaster uses its **official
conversation-level train/dev/test split**. Exact normalized sentences shared with
held-out general data are removed from Taskmaster training. Conversational dev/test
excludes all general data and Taskmaster training matches; test also excludes all
Taskmaster dev matches. Each conversational evaluation set samples up to 100
sentences per domain by hash. Source and derived partition checksums are pinned.

Reports compare a WorldAlphabets-only baseline and the candidate on **identical
queries**, without personal learning. Top-1/top-5 accuracy and vocabulary coverage
are reported at prefix lengths 0–4; fully typed words are excluded. The selection
proxy counts each typed grapheme as one selection and accepting a completion as
its rank (1–5), choosing the cheapest completion after at least two characters.
It excludes scan navigation, timing, spaces and cognitive effort, and assumes
perfect choices: **it is not measured AAC switch savings**.

The downloadable database has exactly the candidate’s evaluated counts. Unlike
v1’s full-corpus artifact, **it excludes held-out development and test sentences**,
preserving these sets for regression checks. `quality-report.json` contains both
development and test comparisons; `development.json` records the acceptance
result. `partitions.json` records sizes, hashes and preparation rules. `score
--baseline PATH --input FILE` evaluates any explicit external set without learning.
The older `evaluate` command remains available for the original WorldAlphabets
sentence-split experiment; it is not the conversational quality benchmark.

See [the frozen protocol](docs/quality-protocol.md) and [the measured results](docs/quality-results.md).
Taskmaster is simulated task dialogue, not a representative AAC corpus. Exact
sentence overlap is removed, but near-duplicates and shared task templates can
remain. Domain bias, sparse everyday contexts, spelling errors, numbers, and
non-English writing remain limitations. No ranking parameters were tuned on the
conversational test results.

## GitHub downloads

Open [Actions](https://github.com/switchifyapp/switchify-prediction/actions), choose
a successful **CI** run for the desired commit, and download
**english-prediction-database** from its artifacts. GitHub login is required for
artifact downloads. Retention is 90 days; a manual workflow run can rebuild them.
The archive includes `english.sqlite`, `SHA256SUMS`, validation and quality reports,
source and partition provenance, corpus attribution and the software license. Verify the files
with `shasum -a 256 -c SHA256SUMS` (macOS) or `sha256sum -c SHA256SUMS` (Linux).

New software is MIT licensed. The database includes CC BY 4.0-derived Taskmaster counts. Corpus rights are separate: see [ATTRIBUTION.md](ATTRIBUTION.md)
and [source-manifest.json](source-manifest.json). The upstream English manifest
identifies a Tatoeba CC0 subset but marks it `verify: false`; its attribution and
license note are preserved rather than treating the corpus as independently audited.

## AAC and spoken-English experiment

An additional, separately published experiment compares the existing baseline
with AAC-focused training messages and a controlled OANC spoken sample. It does
not change the default database or prediction algorithm. See the
[protocol and source terms](docs/aac-experiment.md) for the frozen mixture,
acceptance criteria, reproducible commands and artifact download instructions.
