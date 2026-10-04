# SmolLM2 companion

An optional library and CLI for immediate statistical suggestions followed by offline SmolLM2 refinement. The root predictor, learning APIs and SQLite formats are unchanged. This package has its own workspace and lockfile. It is not integrated with Switchify PC.

This candidate is not yet production-qualified. The optimized Windows worker passed the latency target and improved overall top-five accuracy, but two development cells failed the frozen quality gate. See `docs/smol-production-results.md` in the repository for the full comparison and platform limits. `Ready` means the worker is available, not that the quality gate has passed.

The fixed policy reranks eight statistical candidates using SmolLM2-135M Q8. It scores every token of a word plus the probability of a following word boundary, uses the current sentence capped at 64 tokens and returns at most five ordered words. Scores from different models are never combined or exposed as shared probabilities. Limits above five are errors. Minimum grapheme settings are honored; unigram-only requests stay statistical. The same caller-owned predictor supplies immediate suggestions and the shortlist, including its current personal snapshot.

The controller returns immediate words synchronously. A persistent child loads the model once and refines through private bounded pipes. There is one active request and one replaceable pending request. `poll()` yields only the latest request; callers should also match its request ID before displaying it. Session changes and `reset()` invalidate old results and clear the worker's context. Loading is separate from the 2 second reply deadline for inference and reset. Startup has a 30 second bound. Failure stops the worker and leaves immediate suggestions available. Recovery requires `retry()`. `shutdown()` kills and joins the worker and clears pending text.

## Build

Use Rust 1.97.1. From the repository root:

```sh
cargo build --manifest-path neural/Cargo.toml --workspace --release --locked --target-dir target/smol-portable
cargo test --manifest-path neural/Cargo.toml --workspace --features test-support --locked
cargo clippy --manifest-path neural/Cargo.toml --workspace --all-targets --features test-support --locked -- -D warnings
cargo fmt --manifest-path neural/Cargo.toml --all -- --check
```

The companion library depends on the root crate by path. Add `switchify-prediction-neural = { path = "path/to/switchify-prediction/neural" }` to an embedding application. See `examples/refine.rs` for the complete lifecycle. No crates have been published.

On x64 only, build a separate optimized worker with `RUSTFLAGS="-C target-feature=+avx2,+fma,+f16c"` and `cargo build --manifest-path neural/Cargo.toml -p switchify-smol-worker --features accelerated --release --locked --target-dir target/smol-avx2`. PowerShell uses `$env:RUSTFLAGS` and should clear it after that command. Windows packaging adds `+crt-static` to both builds. Never use `target-cpu=native`. Pass the optimized worker as an optional path to the portable parent; do not start it directly on an unknown CPU.

## Model bundle

The runtime is offline and never downloads assets. `source-manifest.json` pins the upstream revision and source file hashes. `model-bundle.json` pins the converted Q8 bytes, tokenizer, config, model card, license and scoring policy. Missing, changed or unsupported bundles fail explicitly. The 143,041,952-byte GGUF is separate from the existing statistical database and from binary packages.

Acquire the files in `source-manifest.json` into a local source directory. Run the local converter and bundle assembler:

```sh
target/smol-portable/release/quantize SOURCE_DIR MODEL.gguf
python scripts/neural_bundle.py --source SOURCE_DIR --gguf MODEL.gguf --output MODEL_BUNDLE
target/smol-portable/release/switchify-prediction-neural validate --bundle MODEL_BUNDLE
```

Append `.exe` on Windows. The assembler verifies every source and converted hash and refuses to overwrite an existing destination. It includes source provenance, converter source, model card and Apache 2.0 license. Do not substitute a third-party GGUF with an edited manifest.

## CLI

`once` accepts one JSON request on stdin. `stream` accepts JSONL and keeps the worker loaded. The CLI waits for a validated model before accepting requests, and exits nonzero if startup fails. The library can return immediate suggestions during Loading. After successful CLI startup, an inference failure leaves immediate predictions available and allows explicit retry. Supply `--baseline`, `--bundle` and `--worker` explicitly. Optional flags are `--personal`, `--accelerated-worker` and `--threads 1..4`. Existing personal files are read through the established predictor; these commands never learn.

```sh
switchify-prediction-neural stream --baseline english.sqlite --bundle MODEL_BUNDLE --worker switchify-smol-worker
```

Example stdin line:

```json
{"command":"predict","before":"please send the","prefix":"","session":1,"limit":5,"min_chars":0}
```

The default minimum is two graphemes. Output starts with `ready` and capabilities, then `immediate` with a request ID, and later `refined` with the same ID if still current. `status` reports Loading, Ready or Unavailable. Send `{"command":"reset"}` on focus/session cleanup and `{"command":"retry"}` to recover explicitly. Invalid startup configuration or malformed/oversized input exits nonzero. EOF drains the last request; closing the process ends the worker. Text is accepted only on stdin, never command-line arguments. No application fields are read automatically.

## Validation and packaging

The frozen protocol is `evaluation-protocol.json`. Install the benchmark-only dependencies with `python -m pip install -r neural/evaluation-requirements.txt` and run:

```sh
python scripts/neural_evaluate.py --cli target/smol-portable/release/switchify-prediction-neural --worker target/smol-portable/release/switchify-smol-worker --baseline english.sqlite --bundle MODEL_BUNDLE --training data/aac-oanc/prepared/candidate.txt --output results.json
```

Add `--accelerated-worker target/smol-avx2/release/switchify-smol-worker` for the optimized comparison. Each run contains 1,760 warmed queries with personal learning disabled. Reports include quality cells, immediate and IPC-inclusive refinement timings, cache hits/misses, cold load and sampled process-tree RSS. Training overlap is checked; unknown neural pretraining overlap remains possible. These are regression comparisons and do not prove unseen-data accuracy. A failed quality or latency gate prevents a production-quality claim; it does not cause test-set tuning or automatic promotion.

`python scripts/package_neural.py --portable target/smol-portable/release --accelerated target/smol-avx2/release` packages binaries, hashes and dependency notices without model files. Omit the accelerated path for ARM. CI builds/tests Windows x64, Linux x64, macOS ARM64 and macOS x64. Build/test success is not a claim of measured model latency on those platforms. See `SECURITY.md` for the scoped dependency advisory exception and deployment boundaries.


## Asynchronous generation

The generation API returns up to three additional normalized whole words without
requiring statistical-vocabulary membership. Call
Refiner::generate(before, prefix, session, instant_words, 3), then poll using the
returned request ID. A subsequent submit, generate or reset invalidates old results.
The existing submit/reranking API is unchanged. The stream CLI accepts
{"command":"generate","before":"please send the","prefix":"","session":1} on stdin.

Worker protocol 2 is required; protocol 1 workers fail cleanly. Model weights and
tokenizer hashes are unchanged. The manifest records beam width 8, at most 8 tokens
per word, at most 64 forward evaluations including uncached context, and a 1600 ms
search cutoff within the unchanged 2000 ms parent reply deadline. Results are
ranked by whole-word probability including following boundary mass. A boundary
probability of at least 0.5 excludes likely unfinished fragments. Search can return
fewer than three words, including none. This is English-focused and not a spelling
dictionary; plausible but incorrect words remain possible.

Run scripts/generation_evaluate.py with explicit --cli, --worker, --baseline,
--bundle and --output paths. Optional --samples 1000 selects a reproducible spread
across the existing frozen corpus partitions and zero through four graphemes.
Omit --samples for the full comparison. Install psutil==7.0.0 and regex==2025.11.3.
The report includes top-three/top-six counts, fill, OOV, regressions, latency and
process-tree RSS. This is a regression comparison, not unseen-data qualification.
Existing model-quality failures remain; generation has not been promoted to a
qualified model. CI artifacts are prepared without publishing a release.
