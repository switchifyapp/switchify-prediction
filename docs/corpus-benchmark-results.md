# Corpus benchmark results

Ten predeclared models, unchanged predictor, development sets only. These
results compare the four current datasets. Google Books has not been imported.
No model is selected or promoted from this run.

## Useful suggestions after two typed graphemes

Each value is the fraction of target words appearing in the first five suggestions.
Higher is better. The same evaluation sentences are used for every model.

| Training data | AAC | General English | Task conversations |
| --- | ---: | ---: | ---: |
| Current blend | 84.16% | 67.69% | 86.15% |
| Tatoeba alone | 69.70% | 67.62% | 61.60% |
| Taskmaster alone | 74.26% | 50.52% | 87.64% |
| AAC alone | 83.63% | 48.45% | 65.34% |
| OANC alone | 71.88% | 57.11% | 65.72% |
| Without Tatoeba | 84.53% | 57.29% | 86.34% |
| Without Taskmaster | 83.10% | 68.08% | 69.54% |
| Without AAC | 78.23% | 67.49% | 86.56% |
| Without OANC | 83.69% | 66.37% | 85.91% |
| Full OANC blend | 83.00% | 68.08% | 85.31% |

## Removing a source

Changes from the current blend, in percentage points. Negative values mean fewer
target words appeared among the suggestions. Other source counts stay fixed.

| Removed source | AAC | General English | Task conversations |
| --- | ---: | ---: | ---: |
| tatoeba | +0.37 | -10.40 | +0.19 |
| taskmaster | -1.06 | +0.39 | -16.61 |
| aac | -5.93 | -0.20 | +0.41 |
| oanc | -0.48 | -1.32 | -0.24 |

## Size and speed

Source-only models have different training budgets. Their scores cannot establish
which corpus is best at a fixed number of words or sentences. Timings describe
this Windows machine and are not deployment guarantees. Warm p95 below is the
largest of the three suite measurements.

| Model | Unique training sentences | Vocabulary | N-grams | Model MiB | Warm p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current blend | 83,504 | 21,674 | 539,151 | 8.07 | 4.098 |
| Tatoeba alone | 6,425 | 8,578 | 77,895 | 1.29 | 1.358 |
| Taskmaster alone | 52,974 | 11,295 | 235,672 | 3.53 | 1.866 |
| AAC alone | 4,299 | 2,186 | 21,936 | 0.34 | 0.319 |
| OANC alone | 20,000 | 10,837 | 262,366 | 3.99 | 1.925 |
| Without Tatoeba | 77,130 | 17,892 | 480,155 | 7.15 | 3.236 |
| Without Taskmaster | 30,716 | 15,534 | 338,344 | 5.16 | 2.672 |
| Without AAC | 79,352 | 21,459 | 527,663 | 7.91 | 4.313 |
| Without OANC | 63,504 | 17,104 | 315,081 | 4.78 | 3.098 |
| Full OANC blend | 161,914 | 29,609 | 1,180,209 | 17.17 | 6.098 |

## Interpretation

- Removing AAC costs 5.93 percentage points on AAC development examples.
- Removing Taskmaster costs 16.61 points on task conversations.
- Removing Tatoeba costs 10.40 points on general English, while slightly helping
  AAC and task conversations.
- Removing OANC reduces all three scores at the current weights. Expanding its
  sample from 20,000 to 98,410 sentences helps general English but hurts AAC and
  task conversations.

These prepared sources have different sizes and domain filters. The OANC
component already excludes sentences in the other sources. This is a comparison
of our current components, not a ranking of the complete upstream corpora.
Removing a source changes its weight to zero; this does not establish the best
nonzero weight for it.

These are familiar corpus regression suites, not real-world user testing. AAC
text is imagined communication. Existing test sets were excluded from training
but were not scored in this run. Development findings can guide a separately
specified mixture experiment; they are not independent test confirmation.

## Reproduce and audit

See [the protocol](corpus-benchmark.md) for commands. The
[full report](corpus-benchmark-results.json) includes top-one/top-five accuracy
at prefixes zero through four, vocabulary coverage, modeled selection savings,
all timing measurements, corpus sizes and source/model/executable hashes.
The ten training corpora and six evaluation files were re-prepared with the
strengthened frozen-evaluation hash check after independent review. Every
prepared output hash matched the run. The measurement commit predates only that
input guard and these result documents; prediction and corpus recipes did not
change.

Measurement commit: `068ea9d94525df8a518782f56316815ab9fdea31`. Working tree clean at start: `True`.
