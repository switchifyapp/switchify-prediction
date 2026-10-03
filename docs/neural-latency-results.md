# Prediction speed follow-up

FUTO's native decoder met the initial speed target on this Windows PC: **12.83 ms
p95 with four threads**, compared with 24.07 ms using its default one thread.
Both returned exactly the same ordered suggestions. It is a promising next
candidate, but its correction behavior needs adapting and evaluating before use
as an exact-prefix word-completion engine.

Optimizing SmolLM2 helped, but did not meet 20 ms. The best measured cache-miss
p95 was **101.78 ms** for host-optimized Q8, down from 226.72 ms for F32.
Batching improved cache hits but made Q8 context preparation much slower.
No model is promoted and Switchify PC is unchanged.

## Workload and limits

Measured on 2026-10-03, Windows 11, AMD Ryzen AI 9 HX 370, 24 logical CPUs.
Each completed mode used the same 1,280 frozen synthetic queries across messages,
email, documents and search. Neural modes also measured 1,280 immediate repeated
requests with a retained context cache and checked exact ordered-result equality.
These repetitions measure best-case cache hits, not a typing trace or real hit
rate. Candidate scores were always recomputed.

The fixtures are diagnostic comparisons, not representative user text or proof
of unseen-data accuracy. Training overlap beyond the checked exact sentence
matches is unknown. The statistical control retains its historical training
mixture. No personal learning, live field capture or input injection was used.
Benchmarks ran sequentially, without concurrent builds or other benchmarks.

## Neural performance

Times are milliseconds. Miss means the context cache was cleared before every
query. Hit means the immediately repeated request reused context KV/logits.
All completed neural modes had zero failed queries.

| Mode | Miss median | Miss p95 | Miss max | Hit median | Hit p95 | Hit max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Statistical baseline | 0.08 | 6.13 | 8.58 | n/a | n/a | n/a |
| Portable F32 | 165.99 | 226.72 | 398.29 | 128.82 | 187.20 | 381.02 |
| Portable F32, batched | 146.96 | 225.84 | 304.25 | 59.42 | 132.78 | 213.26 |
| Portable Q8 | 223.94 | 319.86 | 539.56 | 136.65 | 204.88 | 443.22 |
| Host-native Q8 | 74.36 | 101.78 | 176.14 | 56.68 | 79.88 | 166.62 |
| Host-native Q8, batched | 159.81 | 297.36 | 436.31 | 32.38 | 69.11 | 114.19 |

Candle's Q8 kernels have compile-time AVX2 gates. The portable build did not
enable them; the separate `-C target-cpu=native` build did. Its binary is specific
to this PC's instruction set and is not a portable release artifact. Four Rayon
threads were used throughout. Q8 reduces weights on disk from 256.60 MiB to
136.42 MiB, but smaller weights alone did not make the portable build faster.

The first portable Q8 batched attempt failed with a non-contiguous output slice
in Candle 0.11 when batch and prefill lengths both exceeded one. The subsequent
adapter uses token-wise prefill for batched Q8. This avoids modifying Candle,
but explains its expensive cache misses. That failed attempt has no accepted
timing/accuracy score and is retained in the machine-readable record.

| Mode | Process cold load, ms | Peak process working set, MiB |
| --- | ---: | ---: |
| Statistical baseline | 2595.63 | 371.16 |
| Portable F32 | 2803.57 | 1036.82 |
| Portable F32, batched | 2844.71 | not retained |
| Portable Q8 | 2716.64 | 513.93 |
| Host-native Q8 | 2710.77 | 514.79 |
| Host-native Q8, batched | 2743.26 | 530.75 |

Cold load excludes checksum verification and does not imply a cold OS file
cache. Working set is the Windows process high-water value and includes loading,
validation and scoring, not just model tensors. The baseline database is
28.42 MiB. The portable batched F32 per-mode report survived the later Q8 failure,
but its aggregate memory/executable metadata did not; neither is invented here.
The runner now saves each completed mode and failure record incrementally.

## Neural accuracy

Early accuracy averages zero, one and two typed Unicode graphemes equally across
the four domains, 768 queries. The same top-eight shortlist supplied every
reranker; each returned five words. Full domain/prefix cells are in the JSON.

| Mode | Early top-one | Early top-five |
| --- | ---: | ---: |
| Statistical baseline | 19.27% | 32.94% |
| F32, sequential or batched | 28.26% | 37.50% |
| Portable Q8 | 28.78% | 37.37% |
| Host-native Q8, sequential or batched | 28.52% | 37.50% |

F32 batching preserved every ordered result. Host-native Q8 batching also
preserved every result of its sequential counterpart. Quantized kernels changed
some rankings relative to F32 and between portable/native builds; matching
aggregate accuracy does not imply identical predictions. Every cache-hit pair
matched its cache-miss result. F32 GGUF conversion passed the fixed prefill and
continuation logit check before quantized results were accepted.

## FUTO native decoder

The probe uses the pinned upstream native decoder and matching 30,662,880-byte
model, verified locally against SHA-256
`6545c1c9ef2d76e9bfb87ad4fcf2061889513af84fcf30d907412be7fcdedb7b`.
It retains upstream tokenization, sampling, three-result limit and the JNI
wrapper's exact-match preference. Typed prefixes use its no-coordinate
correction fallback. See [the probe protocol](futo-probe-protocol.md) and
[upstream sources](futo-investigation.md).

| Threads | Median, ms | p95, ms | Max, ms | Cold load, ms | Peak working set, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.85 | 24.07 | 30.34 | 86.59 | 107.07 |
| 4 | 7.59 | 12.83 | 16.77 | 82.86 | 106.70 |

These are internal native query timings; there is no per-query process startup.
The Windows C++ build enables AVX2. Both runs completed 1,279 supported inferences
and explicitly rejected one prefix containing a nonletter. No decode failures
occurred. All four native decode calls are checked, upstream error paths fail
closed, and the runner requires explicit query-success markers. The four-thread
run's ordered-output digest exactly matches the one-thread run.

This is a three-result correction decoder, not a five-slot reranker. After
normalization, 789 suggestions across each full run did not match the typed
prefix and were filtered without backfill. Native and exact-prefix results are
therefore reported separately. Targets are longer than their tested prefixes,
so removing nonmatching corrections leaves top-three target hits unchanged.

| Typed graphemes | Native top-one | Native top-three | Filtered top-one | Filtered top-three | Empty after filtering |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 14.45% | 22.27% | 14.45% | 22.27% | 0.00% |
| 1 | 4.30% | 12.50% | 8.20% | 12.50% | 15.23% |
| 2 | 36.72% | 59.38% | 40.62% | 59.38% | 0.78% |
| 3 | 45.31% | 71.09% | 47.27% | 71.09% | 2.34% |
| 4 | 53.52% | 82.81% | 55.47% | 82.81% | 1.95% |

FUTO is fast enough in this experiment, but the one-letter results are a clear
weakness for Switchify's completion use. A constrained completion adapter and
real-writing evaluation should come before choosing it. Android output
equivalence has not been tested, macOS runtime performance is untested, and
model/code redistribution terms need a separate review. No upstream decoder
source or model weights are committed or published by this work.

## Validation and reproduction

Root formatting, Clippy with warnings denied and 22 Rust tests passed. The
separate neural crate passed formatting, Clippy and six fake/pure Rust tests;
14 Python tests passed. The native Windows build succeeded. Root dependency
audit passed; the isolated Candle experiment retains its documented
`RUSTSEC-2024-0436` unmaintained-`paste` exception. Production dependencies and
its strict audit remain unchanged. CI and independent latest-head review are
tracked on [PR #16](https://github.com/switchifyapp/switchify-prediction/pull/16).

[Machine-readable results](neural-latency-results.json) retain all completed
domain/prefix cells, hashes, memory data, the failed portable Q8 batch attempt
and measurement limits. Reproduce with [the neural protocol](neural-latency-protocol.md)
and [the FUTO protocol](futo-probe-protocol.md). Use separate empty output
directories for each run. Keep the existing predictor as the production default.
