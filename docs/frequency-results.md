# Broad frequency experiment results

Issue #58. Recorded 2026-11-11 on Windows 11, AMD Ryzen AI 9 HX 370.

Keep the current production model. The broad-frequency experiment did not pass its frozen promotion gates.

## What was tested

Six candidates supplement the shipped en-aac-oanc-v1 unigram counts with wordfreq 3.1.1 frequency estimates. `missing` adds only absent vocabulary; `all` also supplements existing words. Added mass is 1%, 5% or 20% of the original unigram total. Every bigram and trigram count stays unchanged. The runtime algorithm and personal schema are unchanged. No personal database was attached.

The policy and source pins were committed before scoring. All models use the same frozen AAC, general English and Taskmaster regression partitions. Development selected one candidate before any test metric was read. No candidate, threshold, corpus or weight was changed in response to these results.

## Vocabulary audit

The production vocabulary has 21,674 words. Of 50,000 eligible normalized wordfreq entries, 33,181 are absent. Missing dictionary entries are not the same as frequently needed words or prediction errors. The allocation budget determines which missing words actually receive a positive count.

| Candidate | Added unigram mass | New vocabulary entries | Compact model MiB |
| --- | ---: | ---: | ---: |
| all-01 | 10,766 | 96 | 8.076 |
| all-05 | 53,831 | 1,968 | 8.117 |
| all-20 | 215,323 | 9,158 | 8.277 |
| missing-01 | 10,766 | 8,656 | 8.266 |
| missing-05 | 53,831 | 24,673 | 8.622 |
| missing-20 | 215,323 | 33,181 | 8.811 |

## Development decision

Changes below are percentage points. The mean equally weights top-five results at prefixes 0, 1 and 2 in each of the three domains. The frozen minimum mean gain is 0.200 points, with additional AAC, per-row and selection-proxy preservation gates.

| Candidate | Mean early top-five change | Worst early row change | AAC two-character change | Eligible |
| --- | ---: | ---: | ---: | --- |
| all-01 | +0.011 | -0.126 | +0.000 | False |
| all-05 | +0.026 | -0.126 | +0.053 | False |
| all-20 | +0.029 | -0.190 | +0.000 | False |
| missing-01 | +0.004 | -0.014 | +0.053 | False |
| missing-05 | -0.000 | -0.071 | +0.053 | False |
| missing-20 | -0.032 | -0.446 | -0.053 | False |

Selected **all-20**. Diagnostic-only selection: **True**. The decision and development report hash were saved in `selection.json` before test scoring. Test results did not select the candidate.

## Test results

| Domain | Typed graphemes | Baseline top five | Candidate top five | Change, pp |
| --- | ---: | ---: | ---: | ---: |
| aac | 0 | 41.09% | 41.09% | +0.000 |
| aac | 1 | 73.43% | 73.38% | -0.048 |
| aac | 2 | 85.17% | 85.11% | -0.060 |
| aac | 3 | 92.79% | 93.42% | +0.623 |
| aac | 4 | 94.28% | 96.01% | +1.733 |
| general | 0 | 28.53% | 28.58% | +0.044 |
| general | 1 | 54.98% | 55.59% | +0.611 |
| general | 2 | 66.40% | 66.48% | +0.074 |
| general | 3 | 79.14% | 79.50% | +0.359 |
| general | 4 | 83.24% | 86.30% | +3.061 |
| conversation | 0 | 45.23% | 45.32% | +0.085 |
| conversation | 1 | 75.23% | 74.75% | -0.481 |
| conversation | 2 | 86.24% | 86.24% | +0.000 |
| conversation | 3 | 93.93% | 93.72% | -0.201 |
| conversation | 4 | 95.38% | 95.64% | +0.268 |

| Domain | Baseline token coverage | Candidate token coverage | Selection-saving change, pp |
| --- | ---: | ---: | ---: |
| aac | 99.11% | 99.73% | +0.156 |
| general | 94.71% | 97.13% | +0.713 |
| conversation | 98.72% | 99.04% | +0.005 |

Test quality gates passed: **False**. Test mean early change: **+0.025 percentage points**.

## Cost and reproducibility

| Domain | Model | Cold load ms | Warm p50 ms | Warm p95 ms |
| --- | --- | ---: | ---: | ---: |
| aac | baseline | 22.53 | 0.107 | 4.174 |
| aac | all-20 | 28.08 | 0.153 | 6.502 |
| general | baseline | 20.73 | 0.090 | 4.074 |
| general | all-20 | 28.37 | 0.122 | 6.251 |
| conversation | baseline | 21.12 | 0.106 | 4.194 |
| conversation | all-20 | 27.11 | 0.146 | 6.339 |

These are single-process desktop observations with an uncontrolled OS cache and ordinary background activity, not guaranteed latencies. The scorer times prefixes 0 through 4 and longer prefixes used by its selection proxy. Model payload is an encoded-content estimate, not process RSS.

Independent rebuild has identical logical counts: **True**. Original baseline file unchanged: **True**. Every source token round-tripped through the Rust normalizer. Every candidate passed model validation and compact conversion without a logical hash change. All contextual counts retain the baseline hash `7b869827852b95a403d235ce20b74bf2603f397af554cabc664f65e311b5721d`.

## Interpretation and next work

The useful signal is in later completion. At four typed graphemes, the selected
candidate improves general-English top-five by 3.061 percentage points and AAC by
1.733 points. General token coverage also increases. Those gains coexist with
weaker early conversation predictions, including a 0.481-point loss after one
grapheme. This supports investigating a separate vocabulary fallback while
preserving the established early ranking. It does not validate such a fallback.

Broad vocabulary coverage and early top-five quality are different objectives. These results do not justify replacing the production model solely to increase vocabulary size. The next data experiment should target missing contextual phrases or a separately specified low-priority fallback, evaluated against both existing suggestions and genuinely new communication examples. This experiment does not establish that those alternatives will help.

The existing baseline already contains AAC-like messages, spoken OANC, Tatoeba-derived text and Taskmaster-1. An earlier unmerged Taskmaster-2 experiment also failed its development rule. These findings argue for measuring the specific failure type before adding another large source.

## Limits and data terms

The tests are known regression corpora; external word-frequency sources may overlap their material. AAC messages are imagined by crowdworkers. More coverage on these tests is not proof of benefit for unseen AAC users. The selection metric is a modeled proxy, not measured switch scan/search time. No real personal data or personal learning was tested. Changes to unigram scale need a separate personal-learning regression before promotion.

wordfreq code is Apache-2.0, while its frequency data carries CC BY-SA 4.0 and source-attribution terms. The original wheel, full dist-info notices and pinned MessagePack source archive remain beside the local models. No extracted word lists, corpus text or experimental models are committed or published. MIT licensing of our new experiment code does not relicense that data. No SmartKey code or data was copied.

## Reproduce and validate

See the [frozen protocol](frequency-protocol.md), [source manifest](../frequency-source-manifest.json), [quality policy](../frequency-quality-policy.json) and [machine-readable results](frequency-results.json). The report records exact source/member hashes, partition hashes, CLI/script hashes, implementation commit, candidate logical hashes and the pre-test selection-file hash.

Local validation: the Python script suite discovered 27 tests; 26 passed and one
existing optional psutil test was skipped. The five new isolation/selection tests
passed. Rust formatting and Clippy passed; Rust tests passed with 41 tests and
five existing opt-in tests ignored. Repository CI passed root checks on
Windows, macOS and Linux. Its audit jobs exposed an unchanged rustix 1.1.5
advisory, tracked separately in
[issue #60](https://github.com/switchifyapp/switchify-prediction/issues/60).
