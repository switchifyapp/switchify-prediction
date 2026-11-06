# Neural worker performance

`scripts/neural_benchmark.py` measures the worker directly without a statistical
database or desktop application. See the [benchmark instructions](../neural/README.md#worker-performance-benchmark)
for repeatable before-and-after runs. Existing corpus evaluations still measure
accuracy and controller-inclusive latency.

## Quantized embedding rows

The worker previously expanded the entire 49,152 by 576 token embedding table
to F32, in addition to keeping its tied Q8 output weights. It now shares the
quantized table and decodes only requested rows through Candle's embedding API.
It also releases the original GGUF file bytes immediately after loading weights,
before constructing tokenizer lookup tables. Model bytes and search settings are
unchanged.

Measured on Windows x64 on 2026-11-06 with Rust 1.97.1, portable release workers,
four inference threads and the pinned SmolLM2-135M Q8 bundle. The baseline worker
was built from `858cef4`. Each build ran in three fresh processes, with one
warm-up pass and five measured passes per process. There are 24 cases per pass,
covering reranking, words and phrases across four contexts with cache misses and
hits, or 360 measured requests per build. Builds and tests were stopped before
timing runs. OS file caches were uncontrolled.

| Metric | Before | After |
| --- | ---: | ---: |
| Worker RSS at ready, median | 265.5 MiB | 157.8 MiB |
| Worker RSS after measured requests, median | 265.7 MiB | 157.7 MiB |
| Sampled peak worker RSS, median | 403.8 MiB | 281.0 MiB |
| Fresh-process startup to ready, median | 341.9 ms | 325.1 ms |

Warmed RSS fell 40.6%, and sampled peak RSS fell 30.4%. Request medians changed
between -3.2% and +2.3% across cases, consistent with essentially unchanged
inference speed in this small desktop run. Startup was 4.9% faster by median,
but three samples and uncontrolled file caches do not establish a reliable
startup speedup.

All returned prediction hashes matched across builds and were stable within each
build. This workload is performance evidence, not accuracy qualification.
The exact embedding-row test, both explicit local-model tests, the neural
workspace tests and Clippy passed. RSS covers the worker only, sampled every
5 ms; it excludes the controller and can miss brief peaks. These are portable
Windows measurements, not accelerated or cross-platform results.

Raw reports retain model, worker and workload hashes and all per-case timing
distributions: [before](neural-memory-before.json) and [after](neural-memory-after.json).

Reproduce with the same compiled worker variants and validated bundle:

```sh
python scripts/neural_benchmark.py --worker BEFORE_WORKER --bundle MODEL_BUNDLE --output before.json
python scripts/neural_benchmark.py --worker AFTER_WORKER --bundle MODEL_BUNDLE --output after.json --compare before.json
```

The comparison fails if settings or model differ, or if any case's prediction
hashes change. It reports timing and memory changes without imposing a speed
threshold from a single machine. Use the same benchmark before attempting
inference changes, then rerun the corpus evaluations if numerical behavior or
search policy changes.
