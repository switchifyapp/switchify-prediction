# Key hit-map development results

The shipped production model, fixed QWERTY geometry, and personal learning
disabled. Each row contains at most 1,000 hash-selected development targets with one
synthetic nearby-key substitution. No empirical user-error data is used. The AAC four-grapheme row has 697
queries; the other eight rows have 1,000.

Clean input uses the original prefix; noisy input replaces one mapped letter
with a uniformly selected retained neighbor. Exact noisy target retrieval is
zero by construction because the target no longer starts with the typed prefix.
Recovery below is not a comparison against observed user behavior.

| Suite | Prefix graphemes | Queries | Clean exact top 5 | Clean hit-map top 5 | Noisy hit-map top 5 | Noisy hit-map p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aac | 2 | 1000 | 84.4% | 81.9% | 72.2% | 0.874 |
| aac | 3 | 1000 | 92.5% | 90.9% | 85.6% | 0.117 |
| aac | 4 | 697 | 95.4% | 94.4% | 93.5% | 0.046 |
| general | 2 | 1000 | 65.2% | 62.2% | 53.2% | 0.663 |
| general | 3 | 1000 | 82.0% | 78.0% | 70.0% | 0.101 |
| general | 4 | 1000 | 83.6% | 82.3% | 79.1% | 0.048 |
| conversation | 2 | 1000 | 86.2% | 83.9% | 76.0% | 0.675 |
| conversation | 3 | 1000 | 95.0% | 93.9% | 90.0% | 0.100 |
| conversation | 4 | 1000 | 95.8% | 95.3% | 94.6% | 0.048 |

Nearby suggestions recover many of these synthetic errors, but they also displace
correct completions on clean input. Across these rows, clean top-five retrieval
falls by 0.5 to 4.0 percentage points. The API therefore remains opt-in and does
not change exact prediction. A selected correction must replace the owned
unfinished prefix; callers must not simply append its suffix.

This is an engineering check on known development corpora, not evidence of real
switch-user benefit or a complete autocorrection system. The radius and spread
are not empirically calibrated. Timing describes this Windows machine and is
not a cross-platform guarantee. The initial run and Unicode-fix rerun used the
same cases and produced identical retrieval metrics; the table shows a fresh run of this independently split benchmark branch, with
identical retrieval metrics and updated timings.

See [the benchmark protocol](key-hit-benchmark.md), [the library API](key-hit-maps.md) and [the full report](key-hit-results.json)
for top-one retrieval, case/corpus hashes, model fingerprint and timings. No
application integration, text editing or automatic replacement is enabled.

Measurement commit: `463719622e12f5eb6826745dba1a761f0b21f5df`.

Hardware: Windows-11-10.0.26200-SP0; AMD64; 24 logical CPUs.
