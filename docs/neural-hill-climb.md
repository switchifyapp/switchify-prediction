# Neural latency hill climb

The next iteration, [portable SSE2 inference](neural-portable-sse2.md), compares
merged main against baseline-SSE2 matrix products, including repeated CPU-quota
tests and the complete frozen corpus.

The objective is lower inference latency with the same model, candidate counts,
beam width, context limit and search budgets. Both fresh-context and warm
requests are measured, with output comparisons and frozen corpus checks.

Measurements use the existing Windows Ryzen AI 9 HX 370 reference host. Portable
and explicitly AVX2/FMA/F16C-compiled workers are distinct configurations. The
accelerated build already existed before this work; its advantage is not claimed
as a new implementation improvement. The controller still selects it only when
the CPU supports it and an accelerated worker path was supplied.

## Experiments

The [screening results](neural-hill-screening.json) record binary, model and
workload hashes and per-case timing distributions. Most screens use one fresh
process, one warm-up pass and three measured passes, with one inference thread.
These small runs select candidates for broader validation, not production claims.

| Candidate | Decision | Screening observation |
| --- | --- | --- |
| Batch independent reranking paths by token depth | Keep | Reranking medians improved across all eight cases; returned words matched |
| Reuse each generation beam's normalizer at the next depth | Keep | Word/phrase generation medians improved across all sixteen cases; returned words matched |
| Cache whole-word scores for an unchanged context | Keep | Identical repeated shortlists avoided inference; returned words matched |
| ASCII constraint fast path | Reject | Small mixed changes consistent with noise |
| Share token-prefix paths across candidates | Reject | Mixed results even on the completion shortlist |
| Fuse exponentials and boundary accumulation into one branchy loop | Reject | Consistently slower |

Reranking now runs independent candidates together at each token depth, preserving
token score accumulation and original-index tie breaks. A FIFO cache holds at
most 64 candidate scores for the prepared context. Keys are encoded candidate
token IDs, including any leading separator, rather than raw word strings.
Context/session changes and reset discard the cache. Changing shortlists score
only missing candidates. Candidate scores do not depend on the unfinished prefix
or on statistical/personal counts; those affect which candidates are submitted.

An identical repeat is a best-case cache hit. It must not be presented as the
cost of every typed character. Fresh-context latency and the frozen workload's
changing shortlists provide the other measurements.
New benchmark comparisons label `Predict/*/hit` cases as cache reuse, since
the old worker recomputed candidates there while the new worker can reuse scores.
Their percentage changes must not be interpreted as inference-only speedups.

Generation keeps the same scoring, evaluation limit, boundary threshold and
wall-clock cutoff. The normalizer was previously computed once for boundary mass
and again when expanding the same beam. It is now retained with the beam; the
same sums and floating-point operation order are used.

## Validation

The neural workspace tests, Clippy with warnings denied, and Python neural tests
passed. Explicit model tests passed on portable and accelerated builds, including
comparison with the original sequential scorer for mixed token lengths,
duplicates, overlapping cached/uncached shortlists, cache bounds, session/context
invalidation and the existing frozen per-build ranking fixture. Independent code
review found no actionable issue.

The older model's production qualification limits remain; performance
improvements do not establish accuracy qualification.

## Frozen reranking comparison

Both runs use the same controller, production compact statistical baseline,
SmolLM2 weights, accelerated build flags and four inference threads. The full
frozen evaluation contains 1760 queries, including 1759 legitimate neural
refinements and one empty shortlist. Predictions and every quality cell's counts
matched exactly. Neither run had a failed request. Existing quality-gate failures
were preserved rather than changing thresholds or tuning the test set.

Here, "before" is the build after the memory change but before the latency
hill climb. It is not main. Both runs use this branch's controller, including
the Candle thread limit. No full frozen-corpus run compares main with this head;
the embedding change is instead covered by exact row tests and the earlier
memory-stage parity checks.

| Refinement including controller/IPC | Before | After |
| --- | ---: | ---: |
| Median | 95.1 ms | 65.6 ms |
| p95 | 126.9 ms | 93.5 ms |
| Maximum | 214.9 ms | 153.0 ms |
| Fresh-context median | 96.9 ms | 67.9 ms |
| Context-hit median, with changing shortlists | 86.2 ms | 45.8 ms |

These compare implementation changes within the accelerated configuration.
They do not compare the earlier 750 ms single-thread scalar measurement with a
different CPU kernel or thread count. The frozen workload's mixed changing
shortlists are more representative than identical-repeat cache microbenchmarks.

Raw reports: [before](neural-hill-frozen-before.json) and
[after](neural-hill-frozen-after.json). The accelerated binary hash is recorded
alongside the controller and portable-worker hashes.

## Generation spot check

A reproducible spread of 100 existing frozen queries used natural fixture context,
the same accelerated workers and four threads. The evaluator needs SQLite for
its vocabulary/OOV check, so these runs used the SQLite copy of the same pinned
statistical model. All quality-cell counts matched, with no warm-up or measured
failures. The report contains aggregate quality counts rather than per-query
generation output hashes.

| Generation including controller/IPC | Before | After |
| --- | ---: | ---: |
| Median | 148.9 ms | 149.8 ms |
| p95 | 241.9 ms | 237.6 ms |
| Maximum | 699.2 ms | 365.1 ms |

There is no clear overall median/p95 improvement in this four-thread spot check,
despite the normalizer reuse win in the single-thread screening cases. The maximum
from 100 samples is also too noisy to establish a tail guarantee. This is a
generation regression spot check, not full generation qualification.

Raw generation reports: [before](neural-hill-generation-before.json) and
[after](neural-hill-generation-after.json).

Startup also needs separate measurement: the one cold-load sample rose from
555.3 to 654.0 ms in the frozen runs and from 3887.7 to 4634.3 ms in the
generation runs. These single samples and uncontrolled file caches do not
establish a startup regression, but they do not support a startup speedup either.

## Remaining scalar latency

In the single-thread portable screen, empty-context fresh reranking decreased
from 758.7 ms to 698.5 ms. The longer fresh-context case remained 1061.2 ms.
Exact repeats were about 0.3 to 0.4 ms because all scores were cached. Those fast
repeats do not make the scalar cold path acceptable for responsive typing.
This hill climb preserves the original memory reduction, but it does not solve
that fallback's cold inference cost.

Raw portable results, [before](neural-hill-portable-before.json) and
[after](neural-hill-portable-final.json), include unchanged prediction hashes.

## CPU quota checks

Repeated accelerated runs used three fresh processes, one inference thread,
affinity to logical CPU 0 and the same 50%-of-one-processor budget. Each build
completed 72 measured requests with no replies over 2000 ms. Prediction hashes
matched across builds. Warmed worker RSS remained about 157 MiB.

The short per-case quota timings are noisy and sometimes move in opposite
directions. A hard quota permits bursts within scheduling windows and then
stops the worker, so three observations per case do not establish an interactive
tail-latency target under CPU scarcity. Passing the 2000 ms failure boundary is
not the same as meeting a responsive-typing requirement.

Raw accelerated quota reports: [before](neural-hill-avx2-throttle-before.json)
and [after](neural-hill-avx2-throttle-after.json).

Fresh long-context reranking became slower in these quota samples:

| Worker | Before median | After median | Before maximum | After maximum |
| --- | ---: | ---: | ---: | ---: |
| Portable | 1868.6 ms | 2223.8 ms | 2153.4 ms | 2279.2 ms |
| AVX2 | 194.6 ms | 481.4 ms | 489.3 ms | 490.5 ms |

The earlier portable run's two late replies were one reranking request and one
word-generation request. Both late replies in the hill-climb rerun were long-context
reranking requests. Keeping the total at two does not mean the failure pattern
was unchanged. Quota-window alignment and batching's additional KV-row copies
are possible contributors; these runs do not identify the cause. More repeats
of `--modes Predict --cases long` under the same quota are needed to determine
whether this is a repeatable regression. It remains an observed risk, not a
resolved or dismissed result.

The [portable quota rerun](neural-hill-portable-throttle-after.json), compared
with the earlier [portable quota report](neural-throttled-after.json), still had
two late replies out of 72, with a maximum of 2279.2 ms. Phrase output hashes
differed in six cases under the timed search cutoff. Warmed worker RSS was
163.2 MiB in this run, compared with 157.3 MiB in the earlier one. Batching and
faster search can change transient allocations; the original pre-optimization
worker's warmed RSS was about 265 MiB. This is not a portable fallback deadline
or output-parity pass.
