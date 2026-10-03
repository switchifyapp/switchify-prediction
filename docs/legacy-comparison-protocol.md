# Legacy slot-fill protocol v1

Frozen before scoring: preserve every newer result and append normalized unique legacy results only to empty slots. Never interpolate scores, learn evaluation text, or tune on these partitions. Compare original, newer and combined at prefixes 0 through 4 with five slots and minimum prefix zero. Give all modes identical preceding text, preserving the original three-word context. Use all six existing AAC/general/conversation development/test partitions and record their hashes. Report OOV subsets and legacy fill rates alongside accuracy and resource costs. Warm timing uses at least 1,000 queries per mode, with a 20 ms p95 target. Accuracy must not regress against newer; no automatic promotion.

The legacy database is recovered locally from switchify-pc tag v1.0.0-rc.14, path src-tauri/resources/WordData2017051601.db, SHA-256 dedd65d263bde8315e7e5ed7d2c8e04f17c33598a68506c9d70661a6d1f57318. It is never committed, packaged or uploaded. Only BASE_FREQUENCY is read. Ranking follows rc.14 lookup/database_reference: longest context first, frequency descending then target ID ascending, then unigram backoff. Modern Unicode normalization and single-word validation are intentional differences from the historical adapter. Source training overlap is unknown, so results are regression comparisons, not unseen-data estimates.

## Reproduce

Use Rust 1.97.1, Python 3.9+, curl, and Git. On Windows clone with
`git -c core.autocrlf=false clone ...` so checksum-pinned files retain exact bytes.
Obtain the v0.1.0 model bundle and verify it using its `verify_bundle.py`.
Recover the legacy database from an existing Switchify PC checkout without
changing that checkout, using Python binary output:

```python
import subprocess
from pathlib import Path
Path("old.sqlite").write_bytes(subprocess.check_output([
    "git", "-C", "../switchify-pc", "show",
    "v1.0.0-rc.14:src-tauri/resources/WordData2017051601.db"]))
```

```sh
cargo build --release --locked
python scripts/fetch_corpus.py
python scripts/aac_experiment.py --prepare-only
python scripts/compare_legacy.py --exe target/release/switchify-prediction.exe --baseline /path/to/english.sqlite --legacy old.sqlite --output artifacts/legacy-comparison.json
```

Omit `.exe` on macOS/Linux. The comparison command verifies both database hashes
and all six partition hashes, runs each mode in a separate process, checks
position preservation and top-five non-regression, and rechecks database hashes
after completion. Reports contain aggregate counts, never captured typing or
personal databases. Peak process memory includes evaluation bookkeeping; payload
bytes exclude allocator overhead. Non-Windows memory polling is a lower bound.
