# SmolLM2 companion qualification

The implementation provides the separate library, CLI and isolated worker. The model policy is **not production-qualified**. Keep it opt-in. The optimized Windows worker meets the speed target and improves aggregate top-five accuracy, but two development cells fail the frozen regression limit. No desktop integration, model publication or release has been performed.

The policy and fixtures were committed in `1a44602` before scoring. Both workers use SmolLM2-135M Q8, eight statistical candidates, the last sentence capped at 64 tokens, whole-word plus boundary probability and sequential scoring. Personal learning is disabled. The immediate list comes from the same predictor snapshot as the shortlist. The existing statistical APIs, model files and counts are unchanged.

## Quality

The optimized worker completed 1,760 queries, with 1,759 neural refinements and one legitimate empty shortlist. There were no failed requests. Each target in the new partitions is tested with zero through four typed Unicode graphemes, in order. Existing regression queries retain their earlier hash-selection rule. The new development and test sentences were frozen together before evaluation.

| Partition | Queries | Statistical top one | Refined top one | Statistical top five | Refined top five |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing regression | 1,280 | 27.58% | 43.12% | 49.06% | 54.45% |
| New development | 160 | 24.38% | 40.00% | 44.38% | 46.25% |
| New test | 320 | 22.19% | 36.25% | 42.19% | 46.88% |
| All | 1,760 | | | 47.39% | 52.33% |

At zero through two graphemes, existing-regression top five rises from 32.94% to 37.50%, development is unchanged at 22.92%, and the new test partition rises from 19.27% to 23.44%.

Two development cells fail the maximum one-percentage-point decline rule. Documents at one grapheme loses one hit out of eight, from 12.5% to 0%. Messages at zero graphemes loses one hit out of eight, from 25% to 12.5%. All existing-regression and new-test early cells pass. The small cells are noisy, but the agreed gate still fails. The policy was not tuned after seeing these results. A new policy or qualification protocol needs a separately frozen comparison, not a rewritten pass threshold.

The corpus is synthetic general writing covering messages, email, documents and search. Exact normalized overlap with the statistical training file is checked. These results are regression evidence only; overlap with the neural model's unknown pretraining examples cannot be excluded. The older statistical baseline's AAC/spoken-data provenance is retained as a control, not treated as the intended product use case.

## Windows performance

Reference machine: Windows 11, AMD Ryzen AI 9 HX 370, 24 logical CPUs, four inference threads. Timings include private IPC and controller polling unless marked immediate. The optimized binary uses explicit AVX2, FMA and F16C flags. It does not use `target-cpu=native`.

| Optimized measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Immediate predictor call | 0.57 ms | 7.50 ms | 9.95 ms |
| Immediate including CLI transport | 0.82 ms | 7.82 ms | 17.92 ms |
| Refinement including IPC | 91.56 ms | 123.78 ms | 210.27 ms |
| Context miss, 1,368 samples | 92.28 ms | 125.76 ms | 210.27 ms |
| Context hit, 391 samples | 67.62 ms | 115.73 ms | 152.74 ms |

Cold startup to CLI ready was 3.29 seconds, including the statistical database and model verification/load. Peak sampled parent-plus-worker RSS was 638.8 MiB. RSS is summed every 10 ms; shared pages can be counted twice, and brief peaks can be missed. This measures the whole process tree, not just the controller. Weights occupy 143,041,952 bytes; tokenizer and notices are separate bundle files.

The initial portable run recorded a 335.50 ms refinement p95 and one inference timeout after 924 successful refinements. The worker stopped and statistical results remained available. That initial diagnostic did not qualify as the required 1,000 successful neural samples. The benchmark now explicitly requests retry after a measured failure, counts the failed query and reports reload time separately. A warm-up failure aborts clearly instead of silently benchmarking statistical fallback.

## Reproduction and limits

Run the command in [the companion README](../neural/README.md) once with only `--worker`, then once with `--accelerated-worker`. Both use the same pinned model bundle and baseline file. Reports contain binary/input hashes, per-cell counts, timing distributions and process-tree memory. The benchmark source and environment are committed; model and database bytes remain local. The existing baseline logical fingerprint remains `e7d83562681951d4b1a1c12a598c3f9934b211bc697d210cb787672e18c7ec64`.

Windows x64 has actual model runtime measurements. Linux x64 and both macOS architectures have CI compilation, lifecycle tests and packaging, not measured model performance. Do not infer macOS latency from Windows results. No keyboard or pointer input was injected. Tests use synthetic text, fake processes and an explicit local-model parity test.

The remaining qualification work is the failed quality gate and actual model runtime/performance validation on the other target platforms. The portable worker also misses the reference latency target. Passing software checks does not override these limits.
