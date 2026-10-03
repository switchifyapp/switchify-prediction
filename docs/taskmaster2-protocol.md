# Taskmaster-2 experiment protocol

Frozen before scoring on 2026-10-03. This experiment uses the unchanged predictor
on current main and does not promote a database or change the shipping model.

Train two candidates by adding Taskmaster-2 USER turns at weights 1 and 3 to the
unchanged en-aac-oanc-v1 training mixture. Normalize with the Rust CLI, retain
sentences of 2 through 30 words, deduplicate, and exclude exact normalized
matches to current training, all six development/test partitions, and COMM2.
Process domains alphabetically; SHA-256-sort eligible sentences and take at
most 3,000 per domain. A selected sentence cannot appear in another domain.
No assistant utterances, personal data, or personal learning enter the models.

Select using the mean top-five gain across AAC, general English and conversation
development sets at 0, 1 and 2 typed Unicode graphemes. Eligible candidates must
improve that mean, lose no more than one percentage point on any of those nine
measurements, and have development warm p95 below 20 ms. Prefer higher mean gain,
then lower weight. If neither qualifies, test the candidate with the highest
mean as a diagnostic only. Persist the selection before scoring tests. Do not
tune from test results.

Evaluate the baseline and selected candidate on the three existing test sets
and COMM2. Remove COMM2 IDs, normalize, deduplicate, and exclude matches to any
current training or existing evaluation sentence. Report top-one, top-five and
token vocabulary coverage at 0 through 4 graphemes, selection-cost proxy, cold
load, warm median/p95, database size and estimated model payload. Prefix rows
exclude target words already complete at that prefix. The existing scorer also
times longer prefixes used by the selection-cost proxy. Payload is not process
RSS; cold load does not imply an OS-cold disk cache. No latency claims about
legacy should be inferred from these measurements.

Use the prior legacy-fill experiment only as a reference on matching test hashes.
Its results used a different run and the original 2017 training corpus is unknown.
All corpus results are regression comparisons, not proof of unseen-user accuracy.
Taskmaster-2 contains simulated task dialogue, not actual AAC conversations.
Exact deduplication cannot eliminate paraphrases or topical overlap.

Sources are pinned by URL and SHA-256 in `taskmaster2-source-manifest.json`.
Taskmaster-2 is CC BY 4.0, credited to Bill Byrne, Karthik Krishnamoorthi,
Saravanan Ganesh, Amit Dubey, Kyu-Young Kim and Andy Cedilnik from Google LLC.
COMM2 is by Keith Vertanen. Its current webpage says CC BY 4.0; the download's
older README says CC BY-ND 3.0. Keep both notices in the local download and do not
publish corpus text or derived models in this experiment.

## Reproduction

Build the CLI with `cargo build --release --locked`. Prepare the pinned existing
training partitions using `python scripts/aac_experiment.py --prepare-only`.
Extract `english.sqlite` from `switchify-english-en-aac-oanc-v1.zip` on the
[v0.1.0 release](https://github.com/switchifyapp/switchify-prediction/releases/tag/v0.1.0).
The database must have SHA-256
`222253417d0a7a705823ffb7e599a3bcf5d5d3daf4a9d76161ac6b3e555aeaad`.
Then run:

```sh
python scripts/taskmaster2_experiment.py --baseline /path/to/english.sqlite --output artifacts/taskmaster2
```

The output contains source downloads and notices, normalized local corpora,
prepared hashes, candidate databases, the development decision, and results.
Use a fresh output directory for each full run. Published results contain
aggregate metrics and hashes only. CI exercises selection and isolation with
synthetic data; downloading and scoring the full experiment is a local job.
