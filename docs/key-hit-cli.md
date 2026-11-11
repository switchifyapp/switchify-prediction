# Nearby-key CLI suggestions

Enable the QWERTY key hit map explicitly for word suggestions:

```sh
switchify-prediction predict --baseline english.ngrams --prefix hrl --nearby-qwerty
```

For example, `hrl` can offer `help` or `hello` because R is next to E. Without
the flag, prediction remains exact-prefix only and its JSON is unchanged.

With the flag, each result includes `word`, the weighted `score`,
`language_score`, `key_weight` and an optional `substitution`. The substitution
identifies the zero-based grapheme index, typed key and intended key in the
normalized prefix. An exact completion has no substitution.

A corrected word may not start with the typed prefix. Accepting it requires
replacing the owned unfinished prefix, rather than appending a suffix. The CLI
only prints suggestions; it never edits text. Nearby-key ranking can also change
suggestions for clean input, so it remains opt-in.

`--nearby-qwerty` cannot be combined with `--phrases`. Other keyboard layouts can
be supplied through the [library API](key-hit-maps.md). This flag does not enable
nearby-key prediction in Switchify PC.
