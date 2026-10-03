# FUTO native probe

This is a local research harness, not a production Rust adapter or an Android
equivalence certification. Upstream code and weights stay in ignored artifacts.
Do not distribute either through the MIT prediction library.

Use the pinned source and model from `futo-investigation.md`. The preparation
script verifies the source archive SHA-256 and extracts its native code locally.
The model SHA-256 is checked before execution. Keep upstream licence notices
with the downloaded source. Redistribution terms remain a separate review.

The harness keeps upstream `LanguageModelState`, `PredictNextWord`,
`PredictCorrection`, tokenization, logit transforms, sampling and three-result
limit. Typed ASCII prefixes use upstream's no-coordinate fallback with one
character weight set to one; no pointer coordinates are synthesized or injected.
Non-ASCII/nonletter prefixes are reported as unsupported, not silently dropped.
Corrections may violate the typed prefix. Report native top-one/top-three and
exact-prefix-filtered top-one/top-three separately, without backfill. Do not call
these five-slot results or silently compare them as equivalent to the reranker.

Use the same 1,280 Rust-exported synthetic queries and one initial warm-up query.
Keep upstream's single-thread setting. Record load time, internal query latency,
model bytes, peak working set and failures. A 30-minute process timeout terminates
a stuck native run. No keyboard/pointer input, learning or network inference.

Mechanical local source changes remove JNI includes/entry points, replace one
variable-length C++ array with `std::vector`, and guard empty/whitespace trimming
and unsigned character classification.
Two bundled Abseil include paths are adjusted to the repository's layout.
The harness retains the JNI wrapper's exact-match preference before sorting.
All four decode calls are checked, and upstream empty/default error returns
raise a native error. Each completed query emits an explicit success marker;
the runner rejects missing/error markers and non-finite timings. These checks
change failure handling only, never successful rankings.
No weights, sampling thresholds or ranking are tuned against the fixtures.
The native state maintains its own context reuse across the fixed query order.
These timings are separate from the explicit hit/miss Candle measurements.

Also run a fixed four-thread comparison via `llama_set_n_threads`, matching the
Candle thread budget. Use `--threads 4` with a new output directory. This changes
execution parallelism only; compare ordered-output digests with the one-thread
run. Neither threading choice changes the native three-result policy.

Preparation and build:

```sh
curl -fL https://codeload.github.com/futo-org/android-keyboard/zip/70a5d390c505a6bbcc4e14966e5628e43ca3f1fc -o futo-source.zip
python scripts/prepare_futo_probe.py --archive futo-source.zip --output artifacts/futo-probe
cmake -S experiments/futo -B artifacts/futo-build -DFUTO_SOURCE=/absolute/path/artifacts/futo-probe/source -DFUTO_PREPARED=/absolute/path/artifacts/futo-probe
cmake --build artifacts/futo-build --config Release --target futo-probe
```

Export queries using the neural executable's normal pinned baseline/model/
manifest/fixtures/training arguments, plus `--mode baseline --export-queries
--output artifacts/futo-queries.json`. Then run:

```sh
python scripts/futo_experiment.py --executable /path/to/futo-probe --model /path/to/ml4_q6_k.gguf --queries artifacts/futo-queries.json --output artifacts/futo-results
```

Use an empty output directory. Raw fixture outputs remain local; commit only
aggregate measurements and provenance. A failed native build/run must be
reported as such, with no invented accuracy or latency.
