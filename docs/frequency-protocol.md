# Broad frequency experiment

Issue #58. Frozen on 2026-11-11 before scoring. This experiment keeps the current
predictor and its conversational bigram/trigram counts. It does not change or
promote the shipping model, public API, or personal database schema.

SmartKey's hybrid corpus generator separates broad unigram word frequencies from
phrase frequencies. Test that idea using wordfreq 3.1.1 and the shipped
en-aac-oanc-v1 model. The source wheel, parser source and checksums are pinned in
`frequency-source-manifest.json`. No SmartKey code or data is copied.

Read wordfreq's English frequency bins, most frequent first, alphabetical within
each bin. Retain the first 50,000 distinct words that normalize to one Switchify
word of 2 through 32 Unicode code points. Use NFC, lowercase and straight
apostrophes; reject digits, separators and leading/trailing apostrophes. Verify
the selected list round-trips through the Rust normalizer before building.
Combine any raw spellings that map to the same normalized word by summing their
frequency estimates within the selected prefix of the frequency list.

Create six fixed candidates. `missing` adds only words absent from the baseline;
`all` supplements frequencies of existing words as well. Each mode uses an added
unigram mass of 1%, 5% or 20% of the baseline's unigram total, rounded to the nearest
integer with halves rounded up. Allocate that exact integer budget proportionally
to retained source frequencies, using floors followed by largest remainders with
lexical tie-breaking. Zero allocations are omitted. This changes only unigram
counts and vocabulary; it never invents a phrase. Baseline counts are never
reduced. The two modes can add different numbers of words. Their mass fractions
are measured against the full baseline, not just the missing-word subset.

Validate the baseline with `--production`; check every frozen development and
test file against production-model.json. No personal database is attached.
Score the baseline and every candidate on AAC, general and conversation
development data. The selection metric is the equally weighted mean top-five
gain at prefixes 0, 1 and 2 across those three domains. A candidate must gain at
least 0.2 percentage points on that mean, lose no more than 0.5 points in any of
the nine rows, preserve AAC two-character top-five, and lose no more than 0.1
point of the selection-saving fraction in each domain. Timing is reported
separately against the 20 ms target and never selects a candidate.

Prefer the eligible candidate with the highest mean gain, then the smaller mass,
then lexical mode name. If none is eligible, select the same best-mean candidate
for diagnostic test scoring only. Save the development decision, policy/source
hashes, model fingerprints and development report hashes before reading test
metrics. Score only that selected candidate and the baseline on frozen test
files. Apply the same quality gates to those results. Both partitions must pass
before recommending a separate promotion; this experiment never promotes a model.
Record rejected candidates as well as successes, including coverage, top-one,
top-five, selection proxy, cold load, warm p50/p95, model payload and file sizes.

The original wheel and all upstream attribution/licence notices stay alongside
local experimental models. The PR contains aggregate results, code and hashes,
not word lists, downloaded corpus text or candidate models. MIT for our new code
does not change the source data's terms. wordfreq is a historical frequency
snapshot, not a live vocabulary service. A distribution/licence decision would
be needed before any production promotion.

These known regression partitions are not evidence of unseen AAC-user benefit.
External frequency sources can overlap with their source material; we cannot
prove independence from every upstream source. The selection proxy assumes one
action to choose a suggestion and does not measure actual switch scan/search
time. Personal learning must be evaluated separately before shipping a changed
unigram scale. Model payload is not measured process RSS.
