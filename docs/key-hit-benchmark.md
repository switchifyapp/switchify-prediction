# Key hit-map development benchmark

This benchmark exercises the [core hit-map API](key-hit-maps.md) directly and does
not depend on CLI exposure.

The fixed `examples/key_hit_benchmark.rs` protocol compares exact and hit-map
prediction on clean and synthetic-noisy inputs. For each prefix length 2, 3 and
4, it hash-selects at most 1,000 eligible targets longer than the prefix, chooses
one mapped position by a deterministic seed and uniformly selects one retained
neighbor as the typed mistake. The same cases are used for all comparisons.
It reports top-one/top-five retrieval and warm clean/noisy-query timing. Noisy exact
retrieval is zero by construction; it is not evidence of a measured user gain.
Clean-input degradation is reported alongside recovery.

The [initial development results](key-hit-results.md) include recovery, clean-input
changes and latency. They support keeping nearby prediction opt-in.

## Reproduce the complete report

Prepare inputs with `python scripts/aac_experiment.py --prepare-only` if needed.
Use Rust 1.97.1 and Python 3.9 or later, then run from the repository root:

```sh
python scripts/key_hit_benchmark.py \
  --baseline /path/to/english.ngrams \
  --output artifacts/key-hit-benchmark-new-run
```

The committed runner builds the release example, checks all three inputs against
the production manifest, checks each result's protocol/model/evaluation identity,
and writes the combined `report.json`. It records the Git head and dirty state,
source/executable/model hashes, hardware, Python and Rust versions. Timings are
stored at full precision. Display tables round milliseconds to three decimals.
Choose an empty output directory; partial reports have `complete: false`.
The executable path comes from Cargo's JSON artifact output, so both
`CARGO_TARGET_DIR` and Cargo's `build.target-dir` configuration are respected.
There is no fallback to a binary in the default `target/` directory.

The inputs are specifically the AAC/OANC-prepared development files, including
the general English file from that preparation. `data/prepared/general-dev.txt`
is a different partition and must not be substituted.

| Suite | Input | SHA-256 |
| --- | --- | --- |
| AAC | `data/aac-oanc/prepared/aac-dev.txt` | `d85649bf5f81343b7cf32d5d930c39c482a378c64e711c7b3c1c46cbccbbe77d` |
| General | `data/aac-oanc/prepared/general-dev.txt` | `99cb104dbe893608313b300cca933600c97f2c1bedc9799c6e12e148dcd01f3e` |
| Conversation | `data/aac-oanc/prepared/conversation-dev.txt` | `8991ecbdeb263e6345b2e4e0b2301cf3287930ea4289cca9d649aad89144faac` |

To run one suite manually, supply its exact input path to:

```sh
cargo run --release --example key_hit_benchmark -- \
  /path/to/english.ngrams data/aac-oanc/prepared/aac-dev.txt
```

## Timing scope and protocol revision

The corpus measurements cover clean and noisy prefixes of two through four
graphemes. Separate 50-query probes use only the fixed context `i want ` with
an empty prefix or `s`, with `min_chars: 0`. They are diagnostic examples, not a
representative short-prefix workload. One-letter requests can search many more
words than longer prefixes. None of these measurements is a latency ceiling.

Protocol v2 times clean queries, adds the short-prefix probes and uses QWERTY
radius 1.3 to move off the exact 1.25-distance lattice boundary. Spread remains
0.45. The sampling seed remains `key-hits-v1`. The neighbor set, case hashes and
retrieval metrics are checked against the earlier run on the measured machine;
this is a reproducibility/performance correction, not quality tuning.

Use the known AAC, general and task-conversation development suites separately.
Record corpus and code hashes. Synthetic errors do not model switch selection
mistakes, real touch distributions, or actual AAC-user outcomes. The radius,
spread, single-error scope and sampling cap are fixed before each recorded run.
Further tuning belongs in a separately recorded development experiment.
