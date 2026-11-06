# Portable neural inference: SSE2 hill climb

Issue #52, PR #53. Baseline: merged PR #51 on main,
`f342dc92d216ac239472045455c05e66ff1d3134`. This change accelerates portable
x86-64 Q8_0 matrix products without requiring AVX2. Model, tokenization,
scoring and search policy are unchanged.

## Complete frozen reranking evaluation

Both portable builds use the same controller and four threads on Windows 11,
AMD64 Family 26 Model 36, 24 logical CPUs. Neither uses an accelerated worker.
The frozen corpus contains 1,760 queries; one empty context needs no refinement.

| Refinement including IPC | Main | SSE2 |
| --- | ---: | ---: |
| Median | 540.4 ms | 152.5 ms |
| p95 | 813.4 ms | 227.5 ms |
| Maximum | 1,454.6 ms | 365.1 ms |
| Failures | 0 | 0 |

Prediction SHA-256 and every quality cell match. Median and p95 fall about 72%.
The p95 target of 150 ms still fails. Existing quality regressions in
`development/documents/1` and `development/messages/0` remain. This is a
regression comparison with unknown pretraining overlap, not production
qualification.

Raw reports: [main](neural-sse2-frozen-portable-before.json),
[SSE2](neural-sse2-frozen-portable-after.json). These used candidate `59d4cfa`.
Final candidate `af726e8` adds the AVX2 build guard and benchmark provenance;
default portable arithmetic is unchanged. Reports distinguish worker hashes.
Cold load was 649.3 → 448.7 ms, one sample each, insufficient for a startup claim.

## Fresh work on one fixed logical CPU

Final paired runs use `af726e8`, one thread, CPU 0 affinity, and reset before
every measured request. There are no score-cache hit measurements below.
Each build runs in three fresh processes with two warmups per case/process.
Runs were sequential, with matching benchmark/helper hashes and recorded UTC
start/end times. Model, manifest, workload, outputs and result counts match
within each pair.

| CPU budget / case | Samples per build | Main median / p95 | SSE2 median / p95 |
| --- | ---: | ---: | ---: |
| 100% of one CPU / empty | 30 | 1,053.9 / 1,131.2 ms | 223.9 / 273.6 ms |
| 100% of one CPU / long | 30 | 1,644.0 / 2,164.4 ms | 356.1 / 441.4 ms |
| 50% of one CPU / long | 36 | 2,018.9 / 2,156.2 ms | 523.1 / 799.0 ms |

Half-CPU replies over 2,000 ms fall from 21/36 to 0/36; maximum falls from
2,331.0 to 893.2 ms. This direct-worker benchmark waits beyond the production
parent deadline. Median worker CPU time falls from 1,015.6 to 250.0 ms.
Windows CPU accounting is coarse (observed 15.625 ms increments).

The Job Object hard caps normalize to 100.08% and 49.92% of one logical CPU
on this host. Quotas measure scheduling as well as compute; this is one machine,
not a simulation of every low-end processor. No heavy build/test ran alongside
the final pairs.

Raw reports: full budget [main](neural-sse2-paired-main-full-cpu.json) /
[SSE2](neural-sse2-paired-sse2-full-cpu.json); half budget
[main](neural-sse2-paired-main-half-cpu.json) /
[SSE2](neural-sse2-paired-sse2-half-cpu.json).

Memory is essentially unchanged: full-budget median ready worker RSS is
157.82 → 157.82 MiB, warmed RSS 162.77 → 163.00 MiB, sampled peak
285.17 → 284.16 MiB. Sampling every 5 ms can miss brief peaks. Startup includes
temporary model loading. This retains PR #51's packed, shared weights.

## Generation spot check

The same four-thread portable builds process a reproducible spread of 100
natural-context queries after 20 warmups. Median generation including IPC falls
from 1,038.7 to 272.3 ms, p95 from 1,545.6 to 482.8 ms, maximum from
1,645.0 to 738.0 ms. Neither build has warmup or measured failures.

Current top-three hits remain 50; six-slot top-three/top-six hits remain 39/74.
Generated words rise from 282 to 287, changing two aggregate cells. Faster
inference can finish more work before the unchanged 1,600 ms search cutoff.
This does not establish identical generation outputs; 100 queries do not
replace broader generation qualification.

Raw reports: [main](neural-sse2-generation-portable-before.json),
[SSE2](neural-sse2-generation-portable-after.json).

## Implementation and validation

`linear.rs` uses baseline SSE2 for CPU Q8_0/F32 products only on x86-64 builds
without the accelerated feature or AVX2 target feature. All other configurations,
quantizations and precision overrides retain Candle's path. Packed weights are
borrowed. Each signed 32-byte block produces exact i32 sums (maximum 524,288),
then scales/accumulates in Candle's scalar floating-point order. Checked slices
bound the two unaligned 16-byte loads.

Input quantization uses Candle after materializing logical values, fixing
nonzero tensor-view offsets. The small input copies are a future optimization
opportunity. No dependency versions changed: half/rayon were already transitive.

Passed: portable release workspace tests, all five explicit portable model tests,
signed-byte extremes, bit-exact matrix parity at one/four threads, chunk and
projection boundaries, full 49,152-row vocabulary projection, offset and
non-contiguous views, precision fallbacks, accelerated model fixtures, AVX2
target build without the accelerated feature, Clippy with warnings denied,
11 Python benchmark tests, and a real-worker F16 override smoke test.

Independent Opus review found no blocking correctness or safety issue. Its AVX2
configuration finding was fixed and tested. Measurement provenance was addressed
with the final sequential pairs; CPU-resolution and view behavior comments were
also corrected.

## Reproduction

Preserve separate default portable release workers from main and this branch.
Use the same pinned bundle and final scripts. Run each command to completion
before starting the next build's benchmark.

```powershell
python scripts/neural_benchmark.py --worker MAIN_WORKER --bundle MODEL_BUNDLE --output main-full.json --modes Predict --cases empty long --fresh-only --threads 1 --cpu-percent 100 --cpu-affinity 0 --runs 3 --repeats 10 --warmup 2
python scripts/neural_benchmark.py --worker SSE2_WORKER --bundle MODEL_BUNDLE --output sse2-full.json --compare main-full.json --modes Predict --cases empty long --fresh-only --threads 1 --cpu-percent 100 --cpu-affinity 0 --runs 3 --repeats 10 --warmup 2
python scripts/neural_benchmark.py --worker MAIN_WORKER --bundle MODEL_BUNDLE --output main-half.json --modes Predict --cases long --fresh-only --threads 1 --cpu-percent 50 --cpu-affinity 0 --runs 3 --repeats 12 --warmup 2
python scripts/neural_benchmark.py --worker SSE2_WORKER --bundle MODEL_BUNDLE --output sse2-half.json --compare main-half.json --modes Predict --cases long --fresh-only --threads 1 --cpu-percent 50 --cpu-affinity 0 --runs 3 --repeats 12 --warmup 2
python scripts/neural_evaluate.py --cli CONTROLLER --worker WORKER --bundle MODEL_BUNDLE --baseline COMPACT_BASELINE --training TRAINING_TEXT --output frozen.json
python scripts/generation_evaluate.py --cli CONTROLLER --worker WORKER --bundle MODEL_BUNDLE --baseline SQLITE_BASELINE --samples 100 --natural-context --output generation.json
```

The last two commands use a four-thread controller. Frozen reports record hashes
of the compact baseline, training, fixtures, controller and workers. Generation
uses the equivalent SQLite baseline. OS file caches were not controlled.
