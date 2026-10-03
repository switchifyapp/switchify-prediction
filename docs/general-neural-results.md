# General-purpose neural prediction results

The SmolLM2 reranker improved early top-five accuracy by **4.56 percentage
points** on these synthetic fixtures, but missed the 20 ms p95 target. It is
worth optimizing and evaluating on independent everyday writing before any
production decision. The shipping model and Switchify PC are unchanged.

Measured 2026-10-03 on Windows 11, AMD Ryzen AI 9 HX 370, 24 logical CPUs.
Candle 0.11.0 used CPU F32 with four Rayon threads. Each mode completed 1,280
queries, 64 per domain/prefix cell, with zero inference failures.

## Accuracy

These are 96 newly authored diagnostic fixtures, 24 each for messages, email,
documents and search-style text. They are not real-user observations. AAC was
not included in the selection objective. The existing control retains its
historical AAC-weighted training mixture; reranking does not fix its candidate
vocabulary or recover words absent from its top-eight shortlist.

Early accuracy averages prefixes 0, 1 and 2 with equal domain/cell weights.

| Domain | Current top-one | Neural top-one | Current top-five | Neural top-five | Top-eight ceiling |
|---|---:|---:|---:|---:|---:|
| documents | 9.90% | 18.75% | 17.71% | 22.40% | 22.40% |
| email | 23.96% | 34.90% | 43.23% | 47.92% | 48.96% |
| messages | 33.85% | 41.67% | 53.12% | 58.33% | 60.94% |
| search | 9.38% | 17.71% | 17.71% | 21.35% | 21.35% |

| Typed graphemes | Current top-five | Neural top-five | Change |
|---|---:|---:|---:|
| 0 | 14.06% | 20.70% | +6.64 pp |
| 1 | 32.03% | 35.16% | +3.12 pp |
| 2 | 52.73% | 56.64% | +3.91 pp |
| 3 | 66.80% | 73.44% | +6.64 pp |
| 4 | 79.69% | 86.33% | +6.64 pp |

The mean early top-five score rose from 32.94% to 37.50%. Every one of the
twelve early domain/prefix cells improved, with the smallest gain one hit out
of 64. The accuracy criterion passed. Prefix rows use independently hash-selected
target positions, so they do not measure actual keystrokes saved for a fixed
set of words. Per-domain/per-prefix raw hit counts and candidate coverage are
available in the JSON report.

## Speed and memory

| Metric | Current engine | Neural reranker |
|---|---:|---:|
| Cold load ms | 2543.128 | 2818.639 |
| Warm median ms | 0.079 | 165.775 |
| Warm p95 ms | 6.029 | 229.354 |
| Warm maximum ms | 8.280 | 396.552 |
| Peak process working set MiB | 371.07 | 1037.20 |
| Inference failures | 0 | 0 |

The baseline database is 28.42 MiB. The neural weights add 256.60 MiB, plus
about 2 MiB for the tokenizer and small configuration/notice files. F32 tensors
and runtime allocations make memory use larger than the stored weights.

Both modes load once and predict entirely in memory. The timed neural path
includes baseline shortlist generation and whole-word scoring of up to eight
candidates. Context computation is shared within a query, with no reuse across
keystrokes. Each candidate has its own cloned context cache. Cold load starts
after checksum verification and therefore does not represent an OS-cold disk.
Peak working set includes validation, model loading and scoring. It is not
comparable to a model-payload estimate.

No local builds or other benchmark processes ran during the full scoring run.
The 20-query calibration checked execution and timing before the full run; it
did not change the shortlist, model, fixtures, selection rule or scoring method.

## Decision and next work

Do not promote this implementation. Its accuracy result is encouraging, but
229 ms p95 is well above 20 ms. Quantization, batching and reuse of context across
keystrokes are optimization candidates, not measured improvements here. Any
optimized version must repeat accuracy checks and be tested on Windows and macOS.

A stronger evaluation should use independent, appropriately licensed everyday
writing with realistic capitalization and punctuation. The current fixtures
were authored for this experiment, exact training-sentence overlap was rejected,
and unknown overlap with neural pretraining cannot be excluded. Candidate
shortlisting also limits how much longer-context reasoning can help.

The [FUTO investigation](futo-investigation.md) located the official English
model and its custom prediction path. It has not been run here. It remains a
separate engineering candidate; these measurements do not rank FUTO against
SmolLM2.

## Validation and reproduction

Production formatting, Clippy with warnings denied, 22 Rust tests and 12 Python
tests passed locally. The separate Rust experiment passed formatting, Clippy
and three fake/pure tests covering full-token scores, cache isolation, sentence
boundaries, Unicode queries and inference errors. Automated tests inject no input.

The production dependency audit passed. The separate experiment documents one
unmaintained-package exception for Candle's transitive `paste` dependency,
`RUSTSEC-2024-0436`. It is not a production dependency change.

See the [frozen protocol and reproduction command](general-neural-protocol.md)
and [aggregate machine-readable results](general-neural-results.json).
No experimental weights or user text are committed or distributed.
