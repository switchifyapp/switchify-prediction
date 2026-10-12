# Generation performance

Issue #25 makes whole-word generation faster without changing its results, and
gives the model the current sentence as typed. The model, tokenizer, manifest,
search policy and worker protocol 2 are unchanged.

## Changes

- **Batched search.** Each beam extension used its own forward pass, run one
  after another: up to 64 per request. Extensions at one depth all have the same
  length, so they now run as one batched forward pass over a selectable attention
  cache. The worker carries a Llama forward pass adapted from candle-transformers
  0.11.0 that keeps candle's operation order, so a single row computes the same
  values. The pinned-model ranking test still passes on the portable build.
- **Exact early stop.** A piece beginning with a word boundary never continues a
  word, so later completions through a beam are bounded by its probability times
  its non-boundary probability. Search stops once each returned word leads the
  next ranked word, or any unseen word, by more than the beams' combined bound.
  This returns the same ranked words as the full search. A randomized test
  compares both over 400 synthetic models.
- **Natural context.** Generation previously saw the current sentence lowercased
  with punctuation removed. It now sees the sentence as typed, with whitespace
  runs collapsed and trailing opening quotes or brackets dropped. Reranking and
  statistical prediction are unchanged.

## Results

These were measured on an AMD Ryzen AI 9 HX 370 (12 cores) under Windows 11. They
use the full frozen workload of 1,760 queries, or 440 for portable, with four
threads, and give each query a fresh context as `generation_evaluate.py` does.
Latency covers the stream CLI round trip.

| Worker | Median | p95 | Maximum | Identical words |
| --- | ---: | ---: | ---: | ---: |
| AVX2 before | 581 ms | 881 ms | 1,658 ms | |
| AVX2 after | 121 ms | 220 ms | 517 ms | 1,760 / 1,760 |
| Portable before (440) | 1,257 ms | 1,637 ms | 1,671 ms | |
| Portable after (440) | 424 ms | 767 ms | 1,294 ms | 440 / 440 |

Speed by prefix length (AVX2, median): no letters 597 → 90 ms; one letter 582 → 95
ms; two letters 591 → 118 ms; three letters 568 → 149 ms; four letters 564 → 175
ms. Peak process-tree memory was unchanged at about 660 MiB.

Natural context, measured with `--natural-context` on the AVX2 worker:

| Context | Six-slot top six | Generated slots hit | Neural fill |
| --- | ---: | ---: | ---: |
| Normalized | 72.4% | 31.2% | 93.7% |
| As typed | 73.2% | 32.0% | 94.8% |

That is 28 queries gained and 15 lost. Next-word prediction with no letters typed
was unchanged at 32.1%. Latency was unchanged.

A 16-wide beam with 128 evaluations was also tried. It raised top six by a further
0.5 points (10 gained, 1 lost), but median latency rose from 121 to 200 ms, and
next-word prediction did not improve. The policy keeps width 8.

These are regression comparisons on synthetic fixtures. Overlap with the model's
pretraining is unknown, and they make no claim about unseen data. macOS ARM uses
the portable worker. It was not measured on this host, and CI records its timing.

## Reproduce

Build the workspace and the accelerated worker as CI does, then run the following,
once with and once without `--natural-context`:

```text
python scripts/generation_evaluate.py --cli <cli> --baseline <english.sqlite> --bundle <model-directory> --worker <portable-worker> [--accelerated-worker <avx2-worker>] --output <report.json> [--natural-context]
```
