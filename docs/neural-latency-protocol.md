# Neural latency follow-up

Frozen before scoring on 2026-10-03. Follow-up to the general-purpose experiment,
using exactly its fixtures, hash-selected queries, normalization, top-eight
shortlist, whole-word scoring and four-thread CPU setting. No personal learning.

Compare the current statistical baseline, original F32 neural inference, and
Q8_0 neural inference. Convert the pinned SmolLM2 weights locally with Candle
0.11.0. Quantize all matrices including tied embeddings/output to Q8_0; keep
normalization vectors F32. The GGUF runtime dequantizes embeddings in memory.
Record file hashes and sizes. No weights are committed or redistributed.

First convert without quantization and check all logits on four fixed contexts
and one continuation token each against the original implementation. Maximum
absolute error must be below 0.002. This catches tensor-name and RoPE-layout
conversion mistakes before Q8 results are accepted.

Each of 1,280 frozen queries starts with no saved context. For each neural mode,
repeat that same request once with its context KV cache and first-token logits
retained. Recompute every candidate score, and require identical ordered results
for the pair. Report miss and hit latency separately, each with 1,280 samples.
The single-entry cache replaces itself when the effective context tokens differ.
No candidate results or text are persisted by the cache.

This measures best-case context reuse, not a realistic typing trace or hit rate.
Do not average hit and miss timings into an apparent real-world number. Cache
reuse across a changed prefix is valid because the completed-word context stays
fixed, but the changed shortlist must still be scored. Sentence normalization
and the 64-token context limit remain unchanged. Desktop field identity and
password checks are outside this isolated benchmark.

Report accuracy by domain/prefix, whole-output digest, cold load time after
checksum verification, warm median/p95/maximum, peak process memory and failures.
The initial p95 target remains 20 ms. Do not tune the model or shortlist from
these results, and do not promote automatically. Synthetic fixtures are
diagnostic comparisons, not proof of representative or unseen-user accuracy.

Additional optimization fixed before its runs: batch candidate branches for both
F32 and Q8. Prefill identical context in each batch lane, score all next tokens
together, and add each word's boundary probability at its own final token.
Finished lanes receive padding that never contributes to their score. The cache
key includes batch width. This trades duplicated context work and memory for
fewer model calls. Keep the same queries and report both misses and hits. Compare
ordered-output digests and cell accuracy with the sequential implementation;
floating-point kernel differences may affect close rankings. `--only-batched`
allows reproducing this follow-up separately from the initial three runs.

Inspecting Candle's pinned Q8 implementation also identified compile-time AVX2
gates. The default portable Rust build does not enable these. Run a separate
host-native build with `RUSTFLAGS="-C target-cpu=native"`, a separate target
directory, and the same Q8/Q8-batched workloads. Label these results explicitly
and record executable hashes. This is a hardware-specific experiment, not a
portable shipping binary. FUTO's Windows harness likewise uses `/arch:AVX2`.
No accuracy policy changes are allowed for this build comparison.

The initial portable Q8 batched attempt failed on Candle's non-contiguous output
slice when both batch and prefill lengths exceeded one. The adapter now prefills
Q8 batched contexts one token at a time, keeping the public API's input/output
layout valid. This preserves causal attention semantics but changes prefill cost;
report it explicitly when comparing the subsequent host-native batched run.
Completed per-mode results are saved even if a later mode fails.

Pass `--binary-dir target/neural-native/release --build-label host-native
--modes q8 q8-batched` to the runner for the separately built executable.

Reproduce after preparing the existing training data:

```sh
cargo build --release --locked --manifest-path experiments/neural/Cargo.toml --target-dir target/neural
python scripts/neural_latency_experiment.py --baseline /path/to/english.sqlite --model artifacts/general-neural/model --output artifacts/neural-latency-run
```

Use an empty output directory. The existing pinned model cache can be reused.
The experiment retains the previously documented isolated `paste` audit warning
exception. Production dependencies and its strict audit remain unchanged.
