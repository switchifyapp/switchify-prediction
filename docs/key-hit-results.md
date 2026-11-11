# Key hit-map development results

The shipped production model, QWERTY radius 1.3/spread 0.45, and personal
learning disabled. Each row contains at most 1,000 hash-selected development
targets with one synthetic nearby-key substitution. The AAC four-grapheme row
has 697 queries; the other eight rows have 1,000. No empirical user-error data
is used.

Clean input uses the original prefix; noisy input replaces one mapped letter
with a uniformly selected retained neighbor. Exact noisy target retrieval is
zero by construction because the target no longer starts with the typed prefix.
Recovery below is not a comparison against observed user behavior.

| Suite | Prefix graphemes | Queries | Clean exact top 5 | Clean hit-map top 5 | Noisy hit-map top 5 | Clean hit-map p95 ms | Noisy hit-map p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aac | 2 | 1000 | 84.4% | 81.9% | 72.2% | 0.915 | 0.680 |
| aac | 3 | 1000 | 92.5% | 90.9% | 85.6% | 0.113 | 0.104 |
| aac | 4 | 697 | 95.4% | 94.4% | 93.5% | 0.055 | 0.045 |
| general | 2 | 1000 | 65.2% | 62.2% | 53.2% | 0.769 | 0.565 |
| general | 3 | 1000 | 82.0% | 78.0% | 70.0% | 0.115 | 0.087 |
| general | 4 | 1000 | 83.6% | 82.3% | 79.1% | 0.052 | 0.045 |
| conversation | 2 | 1000 | 86.2% | 83.9% | 76.0% | 0.727 | 0.630 |
| conversation | 3 | 1000 | 95.0% | 93.9% | 90.0% | 0.107 | 0.088 |
| conversation | 4 | 1000 | 95.8% | 95.3% | 94.6% | 0.052 | 0.045 |

Nearby suggestions recover many synthetic errors, but they also displace correct
completions on clean input. Across these rows, clean top-five retrieval falls
by 0.5 to 4.0 percentage points. The API remains opt-in and does not change exact
prediction. A selected correction must replace the owned unfinished prefix;
callers must not simply append its suffix.

## Short-prefix timing probes

These use only the fixed context `i want `, with an empty prefix or `s`,
`min_chars: 0` and 50 timed repetitions after warmup. They are diagnostic
examples, not corpus-level short-prefix measurements or a latency ceiling.
Empty-prefix requests now use the normal predictor directly. One-letter
requests still search more candidate words and can be substantially slower
than requests with two through four letters. The main table cannot establish
the feature's worst-case latency.

| Suite run | Prefix | Repetitions | Exact p95 ms | Hit-map p95 ms |
| --- | --- | ---: | ---: | ---: |
| aac | empty | 50 | 4.514 | 4.441 |
| aac | `s` | 50 | 0.438 | 4.287 |
| general | empty | 50 | 4.213 | 4.326 |
| general | `s` | 50 | 0.405 | 4.199 |
| conversation | empty | 50 | 4.314 | 4.227 |
| conversation | `s` | 50 | 0.411 | 4.087 |

## Interpretation and reproduction

This is an engineering check on known development corpora, not evidence of real
switch-user benefit or a complete autocorrection system. The radius and spread
are not empirically calibrated. Timing describes this Windows machine and is
not a cross-platform guarantee. Protocol v2 adds clean timing and short-prefix
probes and moves radius 1.25 to 1.3 to avoid a floating-point boundary. It keeps
the original sampling seed. Every case hash and retrieval metric matched the
previous run; only timing coverage and diagnostics changed.

See [the committed runner and exact input paths](key-hit-benchmark.md),
[the library API](key-hit-maps.md) and [the full report](key-hit-results.json).
The runner assembles all three suites and records source, executable, model,
corpus and case hashes. No application integration, text editing or automatic
replacement is enabled.

Measurement commit: `b16e3070754dd3dce4eb665ef0a0c140b7b8af49`. Working tree clean at start: `True`.

Hardware: Windows-11-10.0.26200-SP0; AMD64; 24 logical CPUs.
