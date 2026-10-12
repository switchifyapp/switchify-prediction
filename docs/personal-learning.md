# Personal learning weight

Issue #29 makes learned words count. Learned counts used to be added to a copy
of the baseline at 5×. Learned text is tiny beside the baseline, so a name or
care term typed a few times hardly ever reached the suggestions.

## Design

Learned counts are kept apart from the baseline and combined when predicting:

- **Weight.** One learned occurrence counts as 500 baseline occurrences while
  little has been learned. The weight is `500 / (1 + learned words / 2000)`, so
  it has halved after 2,000 learned words. However much is learned, learned
  unigram mass approaches 500 × 2,000 = 1,000,000 occurrences, about the size
  of the production baseline's 1,076,617, so learned words approach half the
  unigram distribution rather than outgrowing the baseline. These constants are
  tuned to this baseline; a much larger or smaller one would change that share.
- **New and rare words.** Learned counts of a word are used once it has been
  learned twice, unless the baseline has it at least three times. A single
  misspelling is not suggested, including one the baseline itself contains
  once or twice: the production baseline has over 9,000 such words, many of
  them misspellings. Words the baseline knows well count from their first use.
  A word learned once still counts as context for the word after it.
- **Memory and speed.** The baseline is no longer copied. Peak memory of the
  predictor falls from 362 MiB to 134 MiB, with or without a personal database.
  What prediction uses of the learned counts is rebuilt when they change, so a
  prediction costs about the same as before: 5.3 ms instead of 7.5 ms for an
  empty sentence on the production baseline, and 8.9 ms after 2,000 new words
  were learned twice.
- **No learning.** With no personal database, or an empty one, predictions are
  bit-identical to before.

Scores are computed in floating point, so stored counts cannot overflow when
weighted. The schema and API are unchanged, and existing personal databases
keep working. The `personal_weight: 5` in the v1 model manifest records how that
model was described at build time; this document describes the runtime
weighting.

## Measurement

`scripts/personal_evaluate.py` uses `neural/fixtures/personal.json`. The fixture
is synthetic text imitating one person's typing; it is not user data.

- It has 44 learned sentences and 25 held-out test sentences that reuse their
  names, places, care words and interests. Four of the learned sentences each
  contain one misspelling.
- Each test word is queried with 0–2 typed letters: 445 queries.
- 78 queries are personal words: learned, and with fewer than 10 baseline
  occurrences. They cover only 18 distinct words, each learned 1–3 times.
- A query is a hit when the word is in any of the six slots: 3 statistical and
  3 generated.

Results from this branch, with v0.2.1 generation:

| Weighting | Top six | Personal words | Misspelling suggested |
| --- | ---: | ---: | ---: |
| no learning | 50.3% | 1.3% | 0% |
| 500×, new words at once | 61.8% | 53.8% | 0.2% |
| **500× bounded, new words twice** | **60.4%** | **46.2%** | **0%** |

Some personal words in the fixture are learned only once, so requiring two uses
costs part of the gain here. In real use, names that matter recur.

An earlier sweep of a plain multiplier, measured on the faster generation build
from #26 and before the misspellings were added, showed how much the weight
matters. Personal words reached the six slots in 16.7% of queries at 5×, 28.2%
at 50×, 53.8% at 500×, 60.3% at 2,000× and 65.4% at 5,000×.

General English must not suffer for someone who has learned text. 600 queries
from the frozen general workload, with natural context, were run with that
person's learning present. "Light" is the 40 original sentences learned once;
"heavy" is the same text learned ten times. On this branch, with v0.2.1
generation: no learning 70.3%, light 70.7%, heavy 70.2%. These differences are
within noise for 600 queries. The general-workload runs used a separate local
harness, not this script.

Offering learned words to the neural generator as extra candidates was also
tried. It added under half a point and nearly doubled generation latency, so it
is not included.

Measured on Windows 11 with an AMD Ryzen AI 9 HX 370. These are synthetic
regression measurements, not evidence of real users' accuracy.

## Reproduce

Build the root CLI for learning and the neural workspace, then run:

```text
python scripts/personal_evaluate.py --cli <neural-cli> --learner <switchify-prediction> --baseline <english.sqlite> --bundle <model-directory> --worker <portable-worker> [--accelerated-worker <avx2-worker>] --output <report.json>
```
