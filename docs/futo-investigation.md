# FUTO keyboard model investigation

Inspected on 2026-10-03. FUTO is a relevant general-purpose keyboard reference.
It has not been benchmarked or integrated in this experiment.

The official source at commit
`70a5d390c505a6bbcc4e14966e5628e43ca3f1fc` selects the English `ml4_q6_k` model.
The resource is available from the official `keyboard-large-resources` submodule
at commit `d87d9dbdf3966bbe18413be375dab2f6c7bbdfdd`, path
`raw/ml4_q6_k.gguf`. GitLab's file metadata reports 30,662,880 bytes and content
SHA-256 `6545c1c9ef2d76e9bfb87ad4fcf2061889513af84fcf30d907412be7fcdedb7b`.
These are publisher metadata, not a locally verified download hash. Its name
identifies the Q6_K model; no comparison with the F32 SmolLM2 timings is implied.

The model is not an arbitrary chat GGUF. The Android loader requires KeyboardLM
metadata and recognized feature flags. The native prediction path includes
special word-boundary behavior, candidate search, capitalization filters and
`<XBU>`, `<XBC>`, `<XEC>` / character tokens for supported correction modes.
Loading weights with a generic GGUF runtime alone would not reproduce FUTO.

A standalone Windows/macOS benchmark appears feasible by isolating the native
prediction implementation and its matching runtime behind a Rust adapter.
That is an engineering assessment from the source, not a tested build. A useful
next step is to reproduce its next-word and partial-word outputs on fixed
fixtures before comparing latency or accuracy. Its correction mode must not
silently violate Switchify's exact-prefix completion contract.

The repository identifies its code licence as FUTO Source First 1.1. Model terms
also need checking before any redistribution; this investigation neither copies
FUTO code into the MIT library nor distributes its model.

## Sources

- [Pinned model loader](https://github.com/futo-org/android-keyboard/blob/70a5d390c505a6bbcc4e14966e5628e43ca3f1fc/java/src/org/futo/inputmethod/latin/xlm/ModelPaths.kt)
- [Pinned native prediction path](https://github.com/futo-org/android-keyboard/blob/70a5d390c505a6bbcc4e14966e5628e43ca3f1fc/native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp)
- [Pinned resource submodule declarations](https://github.com/futo-org/android-keyboard/blob/70a5d390c505a6bbcc4e14966e5628e43ca3f1fc/.gitmodules)
- [Official model resource](https://gitlab.futo.org/keyboard/keyboard-large-resources/-/blob/d87d9dbdf3966bbe18413be375dab2f6c7bbdfdd/raw/ml4_q6_k.gguf)
- [Pinned code licence](https://github.com/futo-org/android-keyboard/blob/70a5d390c505a6bbcc4e14966e5628e43ca3f1fc/LICENSE.md)
