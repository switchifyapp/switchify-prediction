# Neural worker performance

`scripts/neural_benchmark.py` measures the worker directly without a statistical
database or desktop application. See the [benchmark instructions](../neural/README.md#worker-performance-benchmark)
for repeatable before-and-after runs. Existing corpus evaluations still measure
accuracy and controller-inclusive latency.

## Quantized embedding rows

The worker previously expanded the entire 49,152 by 576 token embedding table
to F32, in addition to keeping its tied Q8 output weights. It now shares the
quantized table and decodes only requested rows through Candle's embedding API.
Embedding rows remain F32 even when an inherited Candle setting changes matrix
multiplication precision.
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
| Worker RSS after measured requests, median | 265.7 MiB | 157.5 MiB |
| Sampled peak worker RSS, median | 403.8 MiB | 275.6 MiB |
| Fresh-process startup to ready, median | 341.9 ms | 320.6 ms |

Warmed RSS fell 40.7%, and sampled peak RSS fell 31.7%. Request medians changed
between -1.5% and +4.9% across cases, consistent with essentially unchanged
inference speed in this small desktop run. Startup was 6.2% faster by median,
but three samples and uncontrolled file caches do not establish a reliable
startup speedup.

All returned prediction hashes matched across builds and were stable within each
build. This workload is performance evidence, not accuracy qualification.
The exact embedding-row test, both explicit local-model tests, the neural
workspace tests and Clippy passed. RSS covers the worker only, sampled every
5 ms; it excludes the controller and can miss brief peaks. An explicit inference
check with `CANDLE_DEQUANTIZE_ALL_F16=1` also passed. These are portable
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

## CPU-constrained test

The Windows benchmark can apply a hard CPU-time quota to the worker before it
starts. The test on the same host used one inference thread, affinity to logical
processor 0, and a requested budget of 50% of one logical processor. On this
24-thread machine the Windows quota is 208 out of 10,000 system CPU units,
equivalent to 49.92% of one processor after rounding. It leaves the benchmark
controller and the rest of the machine unrestricted.

The first diagnostic exposed a separate problem: setting `RAYON_NUM_THREADS=1`
left Candle's barrier pool at its physical-core default. With all those threads
sharing one CPU, a reranking request did not return within the harness's
35-second timeout. Setting `CANDLE_NUM_THREADS=1` as well removed that stall.
The companion controller and benchmark now set both variables from the requested
thread count. The fake-worker lifecycle tests check their agreement.

The measurements below use explicit thread settings for both worker builds.
The benchmark implementation is `63c8a18`; worker binary hashes are in the
reports. These are portable Windows builds, not the accelerated variant.
Each configuration runs in three fresh processes, with one warm-up pass and
one measured pass per process. That gives 72 measured requests, three samples
per case; tail statistics from so few samples should be treated as diagnostic.
The unrestricted control uses the updated worker and the same single-thread
settings, with no CPU quota or affinity restriction.

| Throttled metric | Original worker | Updated worker |
| --- | ---: | ---: |
| Observed CPU use, median, percent of one processor | 50.7% | 50.8% |
| Warmed worker RSS, median | 265.0 MiB | 157.3 MiB |
| Replies exceeding 2000 ms, of 72 measured | 0 | 2 |
| Slowest measured reply | 1892.5 ms | 2160.9 ms |

The two late updated-worker replies were a fresh-context reranking request and
a fresh-context word-generation request for the longer context. Three samples
per case and ordinary desktop scheduling are insufficient to attribute this
tail difference to the embedding change or claim a reliable speed regression.
The CPU-constrained run does not pass a strict two-second deadline requirement.

The before/after comparison found different phrase outputs in two cases,
`GeneratePhrases/empty/miss` and `GeneratePhrases/empty/hit`, and exited nonzero
after saving the report. Phrase outputs also varied within repeats. The earlier
unthrottled 360-request parity result does not imply parity under timed search
cutoffs. No accuracy qualification is claimed from this workload.

Raw reports: [throttled original worker](neural-throttled-before.json) and
[throttled updated worker](neural-throttled-after.json).

The updated worker's [unrestricted single-thread control](neural-single-thread-control.json)
used a median 99.3% of one processor, with no replies over 2000 ms and a slowest
reply of 1521.4 ms. Its phrase outputs differed from the throttled updated worker
in six cases. These are descriptive comparisons across deliberately different
resource settings; `--compare` rejects such a mismatch for before/after
implementation checks.

| Updated worker, empty context, cache miss, median | No CPU cap | Half-processor budget |
| --- | ---: | ---: |
| Reranking | 750.2 ms | 1234.1 ms |
| Word generation | 906.9 ms | 1694.7 ms |
| Phrase generation | 1338.0 ms | 1416.0 ms |

The phrase timings do not represent identical work: the search can stop sooner
under the real wall-clock cutoff and return different/fewer phrases when capped.

The 2000 ms counts describe worker replies that would be late for the controller.
This direct-worker benchmark does not exercise controller termination/recovery.
Generation keeps its real 1600 ms search cutoff, so its returned results can vary
with scheduling even when numerical behavior is unchanged. Reports retain those
output hash differences and result counts. CPU scarcity does not reproduce a
low-end machine's cache, memory bandwidth or instruction-set performance.
