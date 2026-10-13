# Two-word phrases

Issue #33 adds suggestions of the next two words: the word in progress, or the
next word when nothing is typed yet, then the word after it. Both engines offer
phrases through new APIs. `predict`, `submit` and `generate` and their results
are unchanged.

## Statistical phrases

`Predictor::predict_phrases(before, prefix, options)` returns up to
`options.limit` phrases, each `Phrase { words, score }` with exactly two words.

1. The first words are the top eight `predict` results for the same `before`,
   `prefix` and `min_chars`.
2. The second word is any word the baseline or usable learned counts have seen
   directly after the first word. A first word never seen with a following word
   in a sentence yields no phrase, so phrases never cross sentence punctuation.
3. A phrase scores the product of the first word's score and the second word's
   interpolated probability, with the first word appended to the context. This
   is the n-gram model's joint probability of the two words.

`unigram_only` yields no phrases, since the second word needs the first as
context. Learned counts take part with the same weight and thresholds as in
`predict`. Ties sort by words.

The CLI's `predict` command takes `--phrases`.

Phrases from the n-gram model reflect only the last two words of context.
For example, after "What time is" the most probable phrase can be a television
programme from the spoken-English corpus.

## Generated phrases

Worker protocol 3 adds `Command::GeneratePhrases(PhraseQuery)` and
`Reply::GeneratedPhrases`. `Refiner::generate_phrases(before, prefix, session,
exclude, limit)` mirrors `generate`: at most three results, at most three
excluded phrases, the same input limits, the same latest-only queue and the same
2-second reply deadline. Results arrive through `poll` in `Refined::phrases`,
which is empty for word requests. `Refined::words` is empty for phrase requests.
The stream CLI accepts `generate_phrases` with the same fields as `generate`; its
immediate result contains the top three statistical phrases, which the generated
phrases exclude.

The worker runs one beam search over both words, not two searches.

- A beam may hold up to two words separated by a single space. Every word but the
  last must already be a whole, valid word. The first must complete the prefix.
- A phrase is complete when it has two whole words and the boundary probability
  after the second is at least 0.5, as for single words. Its score is the joint
  probability of the whole byte sequence, so it includes the first word's
  boundary.
- A phrase may take up to 16 tokens in total, twice the single-word depth, shared
  between its words. The budget stays at 64 forward evaluations including
  uncached context, and the search cutoff stays at 1,600 ms.
- Early stopping keeps its guarantee. Within the second word, later completions
  continue the word, as before. A beam still in its first word may complete
  through a following space, so all its probability stays open. A test compares
  stopped and full searches on 400 seeded random models for words and phrases,
  and fails if a beam in the first word uses the single-word bound.

The bundle manifest's policy records the phrase search, and `worker_protocol`
in packaged builds is 3. Protocol 2 workers fail to load cleanly, as protocol 1
workers did before. Applications must update the pinned manifest and worker
protocol when they adopt this release.

## Validation

Unit tests cover phrase shapes, prefix and exclusion rules, the search's
budget, protocol limits, and the phrase lifecycle with fake workers, including
foreign, duplicate, single-word, wrong-ID, stalled and crashed replies.

`scripts/generation_evaluate.py --phrases` measures phrase generation on the
frozen fixtures, for each query whose target has a following word in the same
sentence. A hit needs both words. Sampling follows that filter. It reports statistical, generated and combined phrase accuracy,
first-word accuracy, fill rate, failures and latency. CI runs 1,000 queries on
Windows and macOS.

On a Windows x64 development machine, eight hand-written queries with the
release CLI generated phrases in at most 563 ms with the portable worker and
180 ms with the AVX2 worker. Statistical phrases on the production baseline take
about 2 to 3 times as long as `predict`: 2 to 21 ms without learning and up to
113 ms after 40,000 learned new words.
