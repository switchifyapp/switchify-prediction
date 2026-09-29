# Production operation and release

The default model is `en-aac-oanc-v1` (schema 1). It promotes exactly the accepted
AAC/OANC experiment: 21,674 words, 539,151 n-grams and logical SHA-256
`e7d83562681951d4b1a1c12a598c3f9934b211bc697d210cb787672e18c7ec64`.
No new tuning or held-out training is involved. Software and model versions are
separate: model identity is embedded in the baseline's provenance metadata.

## Distribution and verification

CI builds standalone CLI ZIPs for the native architecture of the macOS, Windows
and Linux runners (the target triple is in each filename and BUILD.json). The
model bundle is platform-independent. Download both from the **same successful
commit's** CI run and extract into **separate directories**. Bundled SQLite needs
no server installation. Linux GNU builds require a glibc version compatible with
the build environment recorded in BUILD.json (currently Ubuntu 24.04 runners);
Windows builds use the static C runtime. The CLI executables are unsigned; they are not
notarized/signed desktop installers. Production applications should consume the
library/data through their own signed application release process.

Verify the archive SHA-256 against its `.sha256` file, then run the included
`python verify_bundle.py` in each unpacked bundle. This checks every listed file.
Checksums detect corruption, not an untrusted distributor: obtain packages from
the project's GitHub Actions or Releases. Then run:

```sh
/path/to/switchify-prediction validate --database /path/to/model/english.sqlite --production
/path/to/switchify-prediction predict --baseline /path/to/model/english.sqlite --before 'I need' --prefix he
```

`--production` rejects altered model counts and missing/changed embedded
provenance even when the SQLite structure is otherwise valid. A CLI version
recognizes the model pinned when it was compiled. For custom models or inspecting
older model versions, use plain `validate` with the corresponding application
schema support. There is no silent schema migration or automatic network updater.

CI downloads the actual ZIPs and exercises production validation, prediction,
learning, reset and baseline immutability from a fresh directory on all three
operating systems. Distribution is refused if either development/test quality
gate fails, frozen accuracy changes, source/partition hashes differ, or repeated
builds produce different logical counts. Timing is recorded rather than used as
a noisy shared-runner gate. Dependency audit warnings also fail CI.

## Embedding and personal data

Open one long-lived `Predictor` on a worker thread. Reusing it keeps prediction
entirely in memory; spawning a CLI process or reopening the model for each
keystroke incurs cold validation/loading. Loading the promoted model takes about
1.6 seconds on the development M2 Max, and learning validates personal state and
copies model counts. Keep open/import/learn/reset/refresh off the UI thread.

`Predictor` is an owned, mutable learning session. Serialize access in the host
application; do not share an unsynchronized connection across threads. Multiple
processes can write the same personal database: immediate transactions serialize
writes with a five-second SQLite busy timeout. Handle lock errors as retryable
only after the operation has returned an error; do not resubmit a completed
segment after success. Every successful learn/import refreshes that instance's
snapshot, including duplicate imports. Other instances call
`refresh_personal()` or reopen to see new learning/reset. Failed refreshes and
failed writes preserve the previous in-memory snapshot. Committed state is
validated before commit, so a post-commit reload failure cannot cause accidental
double learning on retry.

Use an application-owned per-user directory with appropriate access controls,
separate from bundled read-only baselines. Personal n-grams can reveal sensitive
content; SQLite is unencrypted. The library has no background capture, uploads,
telemetry, or raw-text logging. File imports are explicit. Applications must
avoid putting user text into logs or command-line arguments; `learn` accepts
stdin, while production integrations should use the Rust API. Reset removes
logical learning and import hashes, not forensic copies or backups. Crash recovery
uses SQLite transactions/journals; do not delete live journal files.

Back up personal databases while all writers are closed (or with a proper SQLite
online backup implementation). Do not copy an actively written SQLite file alone.
To update the baseline: verify the new bundle and model version, stop/close
predictors, install the new baseline at a separate path, reopen with the unchanged
personal database, and retain the previous baseline for rollback. Never replace
the personal database with a baseline. Baseline update and schema migration are
different operations; schema v1 remains compatible and has explicit tests.

## Corpus rights

Retain ATTRIBUTION.md, both source manifests, production-model.json and
corpus-notices with redistributed databases. MIT applies to new code; it does
not relicense corpus material. AAC and Taskmaster are CC BY 4.0 with credit,
licence links and modification notices. WorldAlphabets identifies its English
Tatoeba CC0 source. OANC's current publisher states unrestricted use and
redistribution including commercial use. Its older XML-release licence contains
different restrictions; we preserve both that notice and the current publisher
grant used for the GrAF source. This is documented evidence, not a claim that the
two texts are identical or an independent legal clearance.

Only public source files are fetched, pinned and used in artifacts. The production
model embeds complete source and policy provenance. Raw training and held-out
files are excluded from distributions. AAC is crowd-imagined communication and
OANC is older American speech. Names, misspellings, disfluencies and inappropriate
suggestions can occur. Scores are model probabilities, not calibrated confidence.
No claim of clinical suitability or measured real-world switch savings is made.

## Maintainer release process

1. Review the milestone-scoped PR and all required CI, artifact smoke tests,
   dependency audit and independent review. Merge only on explicit authorization.
2. On approved `main`, verify the Cargo version, then create the matching `vX.Y.Z`
   tag. No tag or release is created by the production-readiness PR itself.
3. The tag workflow confirms ancestry/version, reruns the full reusable CI and
   creates a **draft** GitHub release with verified model and CLI ZIPs/checksums.
4. Review the draft assets and source notices, then publish explicitly. Release
   assets provide durable downloads; ordinary Actions artifacts expire in 90 days.
5. Close the release milestone only after resolving/moving remaining open work.

Open a milestone-scoped issue and branch for dependency updates. Actions are
pinned to reviewed commit SHAs and Cargo.lock is required. Re-run the complete quality and
packaging gates when changing tokenizer, ranking, schema, model sources or pins.
Fresh model selection requires a new declared protocol and appropriately fresh
evaluation; the existing AAC test is now a known regression set.
