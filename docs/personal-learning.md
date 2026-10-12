# Personal learning weight

Issue #29 raises the weight of learned counts from 5 to 500 baseline
occurrences each. Learned text is tiny beside the baseline. At 5×, a name or care
term typed a few times hardly ever reaches the suggestions.

## Measurement

`scripts/personal_evaluate.py` uses `neural/fixtures/personal.json`. The fixture
is synthetic text imitating one person's typing; it is not user data. It has 40
learned sentences and 25 held-out test sentences that reuse their names, places,
care words and interests. Each test word is queried with 0–2 typed letters, 445
queries in all. 78 of them are personal words: learned and rare in the baseline.
Both six slots count.

| Personal weight | Six-slot top six | Personal words |
| ---: | ---: | ---: |
| no learning | 51.0% | 1.3% |
| 5 | 54.6% | 16.7% |
| 50 | 57.1% | 28.2% |
| **500** | **61.8%** | **53.8%** |
| 2,000 | 62.9% | 60.3% |
| 5,000 | 64.0% | 65.4% |

General English must not suffer for someone who has learned text. 600 queries
from the frozen general workload, with natural context, were run with that
person's learning present. "Light" is the 40 sentences learned once; "heavy" is
the same text learned ten times.

| Personal weight | Light | Heavy |
| ---: | ---: | ---: |
| no learning | 71.0% | |
| 5 | 71.2% | 71.2% |
| **500** | **71.5%** | **71.3%** |
| 2,000 | 71.5% | 71.0% |
| 5,000 | 71.3% | 71.0% |

Gains continue above 500 but slow, and the test sentences reuse exactly the
learned vocabulary, which favours high weights. 500 keeps most of the gain with
no measurable cost to general text. Differences of under about two points on 600
queries are within noise.

Offering learned words to the neural generator as extra candidates was also
tried. It added under half a point at 500 and nearly doubled generation
latency, so it is not included.

Measured on Windows 11 with an AMD Ryzen AI 9 HX 370. These are synthetic
regression measurements, not evidence of real users' accuracy.

## Reproduce

Build the root CLI for learning and the neural workspace, then run:

```text
python scripts/personal_evaluate.py --cli <neural-cli> --learner <switchify-prediction> --baseline <english.sqlite> --bundle <model-directory> --worker <portable-worker> [--accelerated-worker <avx2-worker>] --output <report.json>
```
