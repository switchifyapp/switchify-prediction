# Google Books pilot results

Six predeclared models, development sets only, unchanged predictor. The source
is Google Books English One Million version 20090715, publication years
1980-2008. This is a bounded sample, not a benchmark of the full Google Books
corpus or its newer editions. No model is promoted and no test scores are used.

## Useful suggestions after two typed graphemes

The target word must appear in the first five suggestions. Each suite uses the
same evaluation sentences for every model. Shares describe Google's unigram
probability mass in the combined model, not a new prediction algorithm weight.

| Model | AAC | General English | Task conversations |
| --- | ---: | ---: | ---: |
| Current blend | 84.16% | 67.69% | 86.15% |
| Google sample alone | 56.94% | 47.18% | 51.52% |
| 5% Google | 84.32% | 67.94% | 86.10% |
| 20% Google | 83.85% | 68.03% | 85.88% |
| 50% Google | 83.47% | 68.13% | 85.50% |
| 20% Google words only | 83.74% | 67.72% | 85.88% |

## Changes from production

Values are percentage points; positive means more target words appeared in the
first five suggestions.

| Model | AAC | General English | Task conversations |
| --- | ---: | ---: | ---: |
| Google sample alone | -27.22 | -20.51 | -34.63 |
| 5% Google | +0.16 | +0.25 | -0.05 |
| 20% Google | -0.32 | +0.34 | -0.27 |
| 50% Google | -0.69 | +0.45 | -0.65 |
| 20% Google words only | -0.42 | +0.04 | -0.27 |

The difference between the 20% sampled n-gram blend and its word-only control
isolates the added sampled bigram/trigram counts, holding unigram counts fixed.

| Added contextual counts vs control | AAC | General English | Task conversations |
| --- | ---: | ---: | ---: |
| 20% sampled n-grams minus 20% words only | +0.11 | +0.30 | +0.00 |

## Coverage and size

All ten unigram shards, four of 100 bigram shards and four of 200 trigram shards
were parsed completely. File sampling is not assumed random. Counts are not
upsampled to represent missing files. The source sample is 2,445,733,303
compressed bytes.

| Order | Eligible distinct n-grams | Retained n-grams | Eligible count mass | Retained count mass |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,983,570 | 40,000 | 16,098,219,919 | 15,523,719,425 |
| 2 | 1,044,906 | 200,000 | 468,351,259 | 445,137,221 |
| 3 | 1,783,635 | 200,000 | 163,012,349 | 132,574,443 |

Higher-order eligibility already requires every word to be in the retained
unigram vocabulary. Retention additionally applies count and size limits.

| Model | Vocabulary | N-grams | Compact MiB | AAC vocabulary coverage | Max warm p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current blend | 21,674 | 539,151 | 8.09 | 99.16% | 4.249 |
| Google sample alone | 40,000 | 440,000 | 6.78 | 98.05% | 6.160 |
| 5% Google | 47,086 | 953,126 | 14.14 | 99.62% | 8.444 |
| 20% Google | 47,086 | 953,126 | 14.14 | 99.62% | 8.254 |
| 50% Google | 47,086 | 953,126 | 14.14 | 99.62% | 8.318 |
| 20% Google words only | 47,086 | 564,563 | 8.65 | 99.62% | 8.084 |

## Limits and next decision

- At two typed graphemes, the 5% blend gains 0.16 percentage points on AAC and
  0.25 on general English, while losing 0.05 on task conversations. These are
  small development differences, not independent evidence of a user benefit.
- Its larger gains occur after four typed graphemes: 0.86 points on AAC and
  2.94 on general English. This is a stronger signal for later word completion
  than for next-word prediction, where changes are small and mixed.
- Higher Google shares hurt AAC at two graphemes. The retained contextual
  counts help relative to the 20% unigram control on AAC and general English;
  they do not change that task-conversation score.

The standalone sample has sparse contextual coverage. Its result cannot establish
how a complete Google Books model would perform. Publication dates, book style,
ASCII-only filtering and vocabulary/context pruning also limit the comparison.
Existing evaluation suites are known regressions and AAC examples are imagined
communication. Whole-sentence overlap cannot be removed from aggregate counts.
Timing is descriptive for this machine, not a deployment guarantee.

These development observations can guide a separately specified candidate and
test confirmation. Exact integer mixture coefficients change absolute count
units. Before any production use, normalize count units and independently check
personal learning. The shipping model remains unchanged.

See [the protocol](google-books-experiment.md) and
[full measurements](google-books-results.json) for all prefix lengths, coverage,
modeled savings, hashes, pruning, source row counts and mixture coefficients.
Source attribution and licensing are in the manifest and accompanying notices.

Measurement commit: `d284cb62a43a7e1f12ef8c8a9493bb35e030d08e`. Working tree clean at start: `True`. Source rows parsed: `608,573,094`.
