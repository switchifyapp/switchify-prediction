# Six-slot generation qualification

Issue #23 adds generation while preserving statistical and reranking APIs.
The first three slots belong to the unchanged statistical predictor. The next
three are generated words, excluding the normalized instant words. Empty positions
stay empty.

The worker uses protocol 2 and the unchanged pinned Q8 model/tokenizer. See
neural/model-bundle.json for the frozen search policy. Reports are produced by
scripts/generation_evaluate.py on the existing frozen fixtures. Unknown overlap
with model pretraining prevents claims of unseen-data accuracy.

Validation and platform measurements are in progress. No release is published.
