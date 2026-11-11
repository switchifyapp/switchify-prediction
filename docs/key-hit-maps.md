# Nearby-key prediction

`Predictor::predict_with_hit_map` optionally accepts a keyboard layout to offer
words when one typed letter is near the intended key. The existing `predict`
method remains exact-prefix only. This feature is not enabled in Switchify PC.

For example, `hrl` can suggest `hello` because R is next to E on QWERTY. The
caller chooses whether to show and accept that correction. The library never
types, replaces text, captures input or saves key positions.

```rust,no_run
use switchify_prediction::{KeyHitMap, Options, Predictor};
# fn main() -> switchify_prediction::Result<()> {
let predictor = Predictor::open(std::path::Path::new("english.ngrams"), None)?;
let suggestions = predictor.predict_with_hit_map(
    "say ", "hrl", Options::default(), &KeyHitMap::qwerty(),
);
// Each result includes word, language_score, key_weight and substitution.
# Ok(()) }
```

## Layout and scoring

`KeyHitMap::new` takes `KeyPosition { key, x, y }` values, a neighbor radius
and a spread. Positions, radius and spread use the same units. Supply the
positions of the actual layout, including any rearranged letter keys. The
QWERTY preset uses staggered rows with unit key pitch, radius 1.3 and spread
0.45. These are starting values, not calibrated user error measurements.

Exact-prefix candidates have key weight one. A nearby substitution receives
`exp(-0.5 * (distance / spread)^2)`. The final score is the existing language
score multiplied by that weight. Language context and personal learning still
contribute through the existing predictor. Scores are relative ranking values,
not calibrated probabilities that a correction is right.

The constructor rejects empty/oversized layouts, nonfinite coordinates, duplicate
normalized keys, identical key centers and invalid radius/spread values. It
accepts letters and an apostrophe key, including Unicode letters. Key labels
and typed prefixes use the predictor's Unicode normalization. An unmapped
character stays exact while other mapped positions may be substituted.

Layouts have at most 128 keys and retain the nearest eight alternatives per key.
Only one substituted grapheme is considered, across at most 32 prefix graphemes.
This bounds the search to 257 prefix variants. Longer prefixes use exact
prediction. Insertions, deletions, swapped letters and multiple mistakes are
outside this implementation. Substitutions that merge or alter surrounding
grapheme boundaries are skipped. Actual pointer/touch coordinates and learning a
personal error distribution are also outside this API; observations are typed
key labels, and the map derives alternatives from their centers.

When only the exact variant is available, the API delegates directly to normal
prediction with the requested limit. Short prefixes can match many vocabulary
entries across several variants. The variant bound is not a latency guarantee;
one-letter requests can cost more than requests with two or more letters.

## Accepting suggestions

This API returns `HitSuggestion`, including `language_score`, `key_weight` and
an optional `KeySubstitution`. The substitution gives the zero-based grapheme
index, typed key and intended key in the normalized prefix. Exact completions
have no substitution.

A corrected candidate may not start with the text already typed. Accept it by
replacing the owned unfinished prefix with the selected word, rather than
blindly appending a suffix. Preserve the application's normal ownership,
capitalization and input-cleanup rules. Merely showing a suggestion does not
approve or perform a replacement. Nearby prediction can also reorder clean
input's suggestions; keep exact prediction available where appropriate.
