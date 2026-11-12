# Abbreviated words

`Predictor::predict_abbreviated` optionally offers words from some of their
letters typed in order. For example, `abt` can suggest `about`, `ppl` `people`
and `dont` `don't`. The existing `predict` method stays exact-prefix only. This
feature is not enabled in Switchify PC.

```rust,no_run
use switchify_prediction::{Options, Predictor};
# fn main() -> switchify_prediction::Result<()> {
let predictor = Predictor::open(std::path::Path::new("english.ngrams"), None)?;
let suggestions = predictor.predict_abbreviated("tell me ", "abt", Options::default());
// Each result includes word, score, language_score and abbreviated.
# Ok(()) }
```

From the CLI:

```sh
switchify-prediction predict --baseline english.ngrams --prefix abt --abbreviations
```

Without the flag, prediction stays exact-prefix only and its JSON is unchanged.
`--abbreviations` cannot be combined with `--phrases` or `--nearby-qwerty`.

## Matching

A word matches when it starts with the first typed letter and contains the
other typed letters in the same order, with any letters between them. `abt`
matches `about` and `absent`, but not `bat` or `table`. Matching compares
normalized Unicode grapheme clusters, so letter case and curly apostrophes
behave as they do for `predict`. An apostrophe in the word can be skipped like
any other letter.

Words that start with the typed text are exact completions and keep their
`predict` score with `abbreviated: false`. A word that matches only by skipping
letters has `abbreviated: true`, and its score is the language score multiplied
by `ABBREVIATION_WEIGHT` (0.5). That fixed starting value keeps an exact
completion ahead of an equally likely abbreviated match. It is not a measured
or calibrated probability that the abbreviation was meant.

Language context and personal learning still contribute, through the same
baseline and learned words that `predict` uses. `Options` work unchanged.
Typed text shorter than two graphemes, or longer than 32, uses exact prediction.

## Using the results

An abbreviated word does not start with the typed text, so accepting it means
replacing the unfinished word rather than appending a suffix. The library only
returns suggestions; it never types, edits text or captures input.

Abbreviation matching can push exact completions out of a short list, which is
why it is opt-in. On the shipped model, a call scans every word starting with
the typed first letter; on a desktop PC this added about 10 ms for `s`, the
largest group. Rare misspellings in the baseline, such as `tommorrow`, can also
appear after the intended word.
