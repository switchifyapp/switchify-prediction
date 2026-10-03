# Legacy slot-fill comparison results

Measured on Windows x86-64 on 2026-10-03 using Rust 1.97.1, the released en-aac-oanc-v1 database and the recovered rc.14 legacy database. The mixing rule was frozen before scoring. Both source hashes remained unchanged. No personal data was loaded, desktop integration performed or model promoted.

## Accuracy after two characters

| Partition | Original top five | Newer top five | Combined top five | Gain over newer |
| --- | ---: | ---: | ---: | ---: |
| aac dev | 76.59% | 84.16% | 84.16% | +0.00 percentage points |
| general dev | 63.86% | 67.69% | 67.69% | +0.00 percentage points |
| conversation dev | 68.37% | 86.15% | 86.15% | +0.00 percentage points |
| aac test | 79.25% | 85.17% | 85.17% | +0.00 percentage points |
| general test | 63.41% | 66.40% | 66.44% | +0.04 percentage points |
| conversation test | 68.29% | 86.24% | 86.27% | +0.03 percentage points |

All six partitions passed position-preservation and top-five non-regression checks at every prefix length from zero through four. This conservative policy cannot displace an existing newer suggestion. When all five slots are occupied, legacy data cannot help, even if it knows the intended word.

## Words missing from the newer model

At two typed characters, these are query occurrences, not unique words. The last column counts queries where combined mode appended at least one legacy suggestion, whether correct or not.

| Test partition | Missing-word queries | Original hits | Combined hits | Queries receiving legacy suggestions |
| --- | ---: | ---: | ---: | ---: |
| aac | 19 | 2 | 0 | 1 |
| general | 361 | 33 | 2 | 10 |
| conversation | 60 | 3 | 1 | 11 |

## Resource costs

Each mode ran in a separate process on the same machine. Warm timings cover all scored queries after priming, with at least 1,000 per process. Values below show ranges across partitions for cold load and the worst partition for warm p95, maximum and peak working set. Memory includes evaluation bookkeeping and vocabulary sets, so it is whole-command memory, not pure resident-model size. Database hashing is streamed.

| Mode | Cold load range | Worst p95 | Maximum query | Peak process working set | Database bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 4014 to 4216 ms | 2.40 ms | 5.99 ms | 189.7 MiB | 102,690,816 |
| newer | 2465 to 2588 ms | 6.66 ms | 18.18 ms | 364.8 MiB | 29,802,496 |
| combined | 6560 to 6797 ms | 6.40 ms | 14.34 ms | 422.0 MiB | 132,493,312 |

All modes met the initial 20 ms warm p95 target. The combined predictor adds legacy loading and storage costs while retaining the newer predictor's ranking. These measurements cover this Windows machine only, not macOS/Linux performance.

The original training corpus is unknown and may overlap these sentences. These are regression comparisons, not evidence of unseen-data accuracy or measured AAC-user benefit. The legacy adapter deliberately applies modern normalization and single-word filtering, so its results are not a byte-for-byte reproduction of historical display text.

The aggregate [JSON report](legacy-comparison-results.json) includes all prefix lengths, top-one/top-five accuracy, vocabulary coverage, missing-word hits, fill counts, timing distributions, payload sizes and source hashes. The [protocol](legacy-comparison-protocol.md) gives the reproduction command. Retain this as an experiment; no automatic promotion is recommended from these results alone.
