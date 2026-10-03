# Taskmaster-2 experiment results

Recorded 2026-10-03 on Windows 11, AMD Ryzen AI 9 HX 370, 24 logical CPUs.

Two candidates add the same 21,000 distinct Taskmaster-2 USER sentences at weights
1 and 3 to the released en-aac-oanc-v1 mixture. The predictor algorithm, existing
training weights, API and shipping model remain unchanged. No personal learning.

## Development decision

| Candidate | Mean early top-five change | Worst domain/prefix change | Passes selection gates |
|---|---:|---:|---|
| tm2-1x | -0.128 pp | -0.613 pp | False |
| tm2-3x | -0.751 pp | -1.417 pp | False |

Neither candidate passed. The frozen rule selected `tm2-1x` for diagnostic test
scoring only.
The frozen rule averages top-five accuracy at prefixes 0, 1 and 2 across the
three development domains. Selection was saved before test scoring. The test
results below did not select a weight or change a model.

## Test accuracy

Top-five percentages. Each row excludes words already complete at that prefix.
Top-one, token vocabulary coverage, query counts and selection-cost proxy are
in the machine-readable results. COMM2 uses 1,415 normalized sentences after
deduplication and removing 48 training overlaps and 5 existing evaluation overlaps.

| Test set | Typed graphemes | Current | Taskmaster-2 candidate | Change |
|---|---:|---:|---:|---:|
| aac | 0 | 41.09% | 40.69% | -0.40 pp |
| aac | 1 | 73.43% | 73.34% | -0.10 pp |
| aac | 2 | 85.17% | 84.93% | -0.24 pp |
| aac | 3 | 92.79% | 92.97% | +0.18 pp |
| aac | 4 | 94.28% | 94.45% | +0.17 pp |
| general | 0 | 28.53% | 28.44% | -0.09 pp |
| general | 1 | 54.98% | 54.86% | -0.12 pp |
| general | 2 | 66.40% | 65.98% | -0.43 pp |
| general | 3 | 79.14% | 78.94% | -0.21 pp |
| general | 4 | 83.24% | 83.35% | +0.11 pp |
| conversation | 0 | 45.23% | 45.38% | +0.15 pp |
| conversation | 1 | 75.23% | 75.62% | +0.39 pp |
| conversation | 2 | 86.24% | 86.09% | -0.15 pp |
| conversation | 3 | 93.93% | 94.01% | +0.08 pp |
| conversation | 4 | 95.38% | 95.51% | +0.13 pp |
| comm2 | 0 | 37.98% | 38.23% | +0.26 pp |
| comm2 | 1 | 70.70% | 70.92% | +0.22 pp |
| comm2 | 2 | 81.89% | 81.84% | -0.06 pp |
| comm2 | 3 | 91.42% | 91.36% | -0.05 pp |
| comm2 | 4 | 93.45% | 93.85% | +0.39 pp |

## Earlier legacy backfill comparison

These are prior-run accuracy measurements on the same test hashes. The current
baseline exactly reproduces all prior newer-mode accuracy and coverage rows.
Legacy timing is not compared across runs. COMM2 was not tested with legacy.

| Test set, two graphemes | Original 2017 | Current | Legacy backfill | Taskmaster-2 |
|---|---:|---:|---:|---:|
| aac | 79.25% | 85.17% | 85.17% | 84.93% |
| general | 63.41% | 66.40% | 66.44% | 65.98% |
| conversation | 68.29% | 86.24% | 86.27% | 86.09% |

[Prior experiment source](https://github.com/switchifyapp/switchify-prediction/blob/63d7867f994d38b04fdadffd0cf529ba7f133d71/docs/legacy-comparison-results.md).

## Cost

Ranges across the available development/test scoring processes. The 3x candidate
has development measurements only. Each individual report has more than 1,000
queries at prefixes 0 through 4. Warm timings also include longer prefixes
used by the selection-cost proxy. The scorer reports median and p95, not maximum.

| Model | Database MiB | Model payload MiB | Cold load ms | Warm median ms | Worst p95 ms |
|---|---:|---:|---:|---:|---:|
| baseline | 28.42 | 16.04 | 2537-2637 | 0.193-0.262 | 6.425 |
| tm2-1x | 32.36 | 18.16 | 2925-3447 | 0.214-0.276 | 6.885 |
| tm2-3x | 32.42 | 18.16 | 2988-3017 | 0.207-0.269 | 6.758 |

Payload is an estimate of model contents, not process RSS. Cold load measures a
new predictor process with an uncontrolled OS disk cache. These desktop timings
are observations, not a latency guarantee. No other local build or benchmark was
run during scoring.

## Interpretation and limits

Keep the current model. Neither Taskmaster-2 weight improved average early
prediction on development data. The lighter mixture reduced two-grapheme top-five
accuracy on all four test sets, despite small gains in some other prefix rows.
The database grew by 13.9%, estimated payload by 13.2%, and worst observed p95
rose from 6.425 to 6.885 ms. All p95 observations remain below the 20 ms target.

There are small gains worth recording. At two graphemes, top-one accuracy rose
by 0.47 percentage points on conversation and 0.34 on COMM2, while falling by
0.12 on AAC and 0.30 on general English. Whole-target token vocabulary coverage
at zero graphemes rose by 0.09 to 0.13 percentage points across the four sets.
The modeled selection-saving fraction improved by 0.14 percentage points on
conversation and 0.13 on COMM2; it fell by 0.12 on AAC and 0.01 on general English.
These are modest tradeoffs, not a consistent improvement under the frozen rule.

The earlier legacy backfill retained or improved two-grapheme accuracy on the
three existing test sets. It remains a separate experiment. This comparison
does not establish that either source is better for unseen AAC users. Further
data work should target AAC language rather than simply add more task dialogue.

Taskmaster-2 contains simulated task dialogue rather than real AAC use. COMM2 is
crowdsourced imagined communication. The existing test sets are known regression
sets; exact-match filtering cannot remove paraphrase or topical overlap. The
2017 training corpus is unknown. None of these results proves unseen-user accuracy.

Taskmaster-2's CC BY 4.0 attribution and COMM2's differing current/historical
notices are pinned in the source manifest and retained in local downloads.
No corpus text, experimental database or default model change is included here.

See the [frozen protocol and reproduction command](taskmaster2-protocol.md),
[machine-readable results](taskmaster2-results.json), and
[source manifest](../taskmaster2-source-manifest.json).
