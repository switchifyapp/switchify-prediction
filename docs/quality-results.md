# Conversational baseline quality results

One predeclared corpus mixture was selected using development results, before
scoring the frozen conversational test partition. The ranking algorithm is unchanged.
These are local measurements on an Apple M2 Max; CI repeats the experiment on Linux.

## Accuracy and selection proxy

Both models use identical held-out queries and no personal learning. The comparison
baseline is WorldAlphabets-only **80% training data**, not the old full-corpus download.
The candidate adds Taskmaster training requests with general sentences weighted 3×.

| Partition | Suite | Baseline top-5 at 2 chars | Candidate top-5 at 2 chars | Baseline selection savings | Candidate selection savings |
|---|---|---:|---:|---:|---:|
| development | general | 67.62% | 66.42% | 17.87% | 17.82% |
| development | conversation | 61.60% | 86.61% | 12.00% | 23.75% |
| test | general | 68.27% | 65.59% | 18.15% | 17.88% |
| test | conversation | 61.65% | 86.56% | 11.53% | 23.11% |

Development passed the predeclared +5 percentage-point conversational gain and
at-most-2-point general loss gates. The subsequent test result shows **+24.91 points
on task conversation and -2.68 points on general English**. The general test loss
exceeds the development guardrail: this is a tradeoff, not universal improvement.
No corpus weights or ranking parameters were changed after observing test scores.
The WorldAlphabets-only alternative can still be built from `data/prepared/baseline.txt`.

The selection metric is an optimistic offline proxy: one selection per typed
character, acceptance cost equal to suggestion rank, choosing the cheapest option
after two characters. It does not model the actual Switchify scanning UI, timing,
cognitive effort, spaces, or errors. It must not be advertised as measured AAC savings.

## Cost and reproducibility

Candidate warm p95: **4.31 ms**; cold model load: **876 ms**; evaluation database: **15.88 MiB**. Both suites stay below the 20 ms warm p95 target locally. Initial load includes schema and model validation; OS caches are not flushed.

The published database has the same logical count hash as the evaluated candidate:
`cb9eddcd254dfbc97e73dc4aff11881216ab8c038a4a805337f67820bc09938b`.
It contains 16,761 words and 302,112 n-grams. Held-out development and test sentences
are excluded from publication. Source, policy, and derived text hashes are pinned.
The quality report records file checksums for both compared models; publishing
adds full provenance metadata, so the published file hash differs while counts match.

## Actual candidate suggestions

These examples are illustrative checks after the mixture was frozen, not acceptance tests.

| Typed text | Suggestions in order |
|---|---|
| `I need he` | help, here, he, hey, hello |
| `I would like wa` | want, was, watching, way, water |
| `I would like wat` | watching, water, watch, waterfront, watched |
| `I drink wa` | want, was, way, wait, watch |
| `I want to ` | order, go, see, get, be |

Taskmaster contains human-written simulated requests in six domains, not real AAC
usage or unrestricted everyday conversation. Sparse contexts such as `I drink wa`
remain weak. Exact duplicate sentences are removed across evaluation boundaries;
shared templates and near-duplicates can remain. The general test set is a known
regression set from v1. New language support, actual AAC evaluation, and personal
writing remain separate work; personal learning can already be used explicitly.
