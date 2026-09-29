# AAC and spoken-English corpus experiment

This is the frozen selection protocol from issue #5. Issue #7 promotes its
accepted counts to the production default; see [production guidance](production.md).
Statements below about an unchanged default describe the original experiment.

Issue #5 compares one predeclared corpus mixture against the existing
WorldAlphabets + Taskmaster conversational candidate. No predictor algorithm
changes are made. Personal data is never read.

## Protocol fixed before scoring

- Preserve the current candidate counts as the comparison baseline.
- Add each unique eligible AAC training sentence ten times.
- Add 20,000 unique OANC spoken sentences once, restricted to 2–30 words and
  selected by SHA-256 order (text breaks ties). No content-based hand selection.
- Use the predictor's Rust tokenizer for every source; no separate Python
  approximation of Unicode normalization or sentence boundaries.
- AAC keeps the original worker-based train/development/test assignment.
  Deduplicate within each partition. Remove all AAC development/test text from
  new training sources; evaluation additionally excludes overlap with the
  existing baseline, AAC training, and existing evaluation suites. AAC test also
  excludes every original AAC development sentence. This conservatively removes
  repeated sentences rather than reassigning them between worker partitions.
- Exclude all existing general and Taskmaster held-out sentences from added
  training. Assert zero normalized exact-sentence training/evaluation overlap
  and zero overlap between evaluation partitions. This is not a claim of zero
  semantic or near-duplicate overlap.
- Extract OANC from the current GrAF archive's **spoken** files only, using its
  sentence/utterance span annotations. Collapse formatting whitespace within
  each span and reset context between spans. Preserve disfluencies and spelling;
  do not invent corrections. The OANC sample is training-only.
- Score development first, then the test sets once regardless of the development
  outcome, to report a complete fixed experiment. Do not tune from these results.
  General and Taskmaster tests are known regression sets; AAC test is fresh.
- Require both development and test to improve AAC top-5 accuracy after two
  characters by at least 5 percentage points, lose no more than 1 point on
  general English and 2 points on Taskmaster, and preserve AAC selection-proxy
  savings. Warm p95 below 20 ms is reported, not enforced in CI.
- Measure accuracy/coverage at prefix lengths 0–4, selection proxy, cold load,
  warm p50/p95, payload memory, pipeline peak RSS, and size using existing
  scoring code. The selection proxy is not measured AAC switch effort.
- Build twice and compare logical contents. Publish a clearly **experimental**
  database and report separately from the unchanged default artifact. A passing
  result is a recommendation for promotion, not an automatic default change.

The exact parameters are in `aac-quality-policy.json`. All source files and
licensing evidence are pinned by `aac-source-manifest.json`; derived partition
hashes and counts are recorded in each report. The protocol commit predates any
AAC development or test scoring.

## Reproduce

```sh
cargo build --release --locked
python3 scripts/aac_experiment.py
```

Python 3.9+ and curl are required. Source acquisition downloads the 655 MB OANC
GrAF archive plus the small AAC files and original baseline sources. After that,
verified cached files support offline reruns. No archives are unpacked onto the
filesystem: only explicitly selected spoken entries are read. Run
`--prepare-only` to inspect partition statistics without scoring. The output is
`artifacts/aac-experiment`; it does not overwrite `artifacts/english.sqlite`.

Download the `aac-oanc-experiment` artifact from the PR's successful CI run. Read
`report.json`, `README.txt`, `ATTRIBUTION.md` and `corpus-notices` before using the
experimental database. The manifest and SHA256SUMS link the model to exact
sources, partition files, parameters and notices. No raw training or test text
is published in the artifact.

## Source terms and limitations

AAC: Keith Vertanen and Per Ola Kristensson (2011), *The Imagination of Crowds:
Conversational AAC Language Modeling using Crowdsourcing and Large Data
Sources*, EMNLP, pp. 700–711. The publisher explicitly grants CC BY 4.0 for the
three `sent_*_aac.txt` files. Neither of the separately excepted legacy test
files, pretrained models, nor Turk/COMM2 corpora are used. Retain attribution,
licence link and notices of modifications.

OANC: Nancy Ide and Keith Suderman (2007), *The Open American National Corpus*.
The publisher's current website states unrestricted usage and redistribution,
including commercial use. We pin and preserve that statement from its official
website repository. The selected GrAF archive contains no separate licence
file. An older XML release contains an OANC licence that distinguishes
transparent changes from substantive changes; its notice is also retained,
without mislabelling OANC as CC0 or MIT. We retain attribution and modification
records and distribute n-gram counts, not rewritten source utterances. The
historical notice and current broad grant differ; this record is evidence of
published terms, not a representation that they are identical or legal advice.

The AAC communications were imagined by crowd workers, not contributed by AAC
users. They include mistakes and may overrepresent care-related requests. OANC
contains older American English telephone and interview speech with fillers and
false starts. Neither source is representative of every person's communication.
