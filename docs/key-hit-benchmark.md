# Key hit-map development benchmark

This benchmark exercises the [core hit-map API](key-hit-maps.md) directly and does
not depend on CLI exposure.

The fixed `examples/key_hit_benchmark.rs` protocol compares exact and hit-map
prediction on clean and synthetic-noisy inputs. For each prefix length 2, 3 and
4, it hash-selects at most 1,000 eligible targets longer than the prefix, chooses
one mapped position by a deterministic seed and uniformly selects one retained
neighbor as the typed mistake. The same cases are used for all comparisons.
It reports top-one/top-five retrieval and warm noisy-query timing. Noisy exact
retrieval is zero by construction; it is not evidence of a measured user gain.
Clean-input degradation is reported alongside recovery.

The [initial development results](key-hit-results.md) include recovery, clean-input
changes and latency. They support keeping nearby prediction opt-in.

```sh
cargo run --release --example key_hit_benchmark -- \
  /path/to/english.ngrams data/aac-oanc/prepared/aac-dev.txt
```

Use the known AAC, general and task-conversation development suites separately.
Record corpus and code hashes. Synthetic errors do not model switch selection
mistakes, real touch distributions, or actual AAC-user outcomes. The radius,
spread, single-error scope and sampling cap are fixed before this initial run.
Further tuning belongs in a separately recorded development experiment.
