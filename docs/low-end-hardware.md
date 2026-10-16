# Low-end hardware and GPU acceleration

Issue #39 asks for measured evidence for a minimum spec and for whether a GPU path is worth having. **No low-end machine has been measured yet.** Everything below was measured on one fast laptop as a stand-in, or is analysis only. The minimum spec at the end is therefore provisional. The production worker, its default and AVX2 builds, the policy, the model bundle and the baseline are unchanged.

## What is new

- `scripts/hardware_benchmark.py` runs the frozen `generation_evaluate.py` workload and, with `--training`, the `neural_evaluate.py` refinement benchmark, unchanged. It records the CPU model, SSE4.1/SSE4.2/AVX/AVX2 support, cores, logical CPUs, RAM, OS, power state and the native machine when an x64 process is emulated. It summarizes median, p95 and maximum latency, failed requests, reply timeouts, replies at or past the 2 s deadline, cold startup, process-tree peak RSS and thread count. It refuses a model or baseline that differs from the pins and refuses to start on a busy machine.
- A benchmark-only `experiment` worker feature lowers beam width and `max_forward_evaluations` through the environment. The harness confirms the worker is such a build, so a production worker cannot be mislabelled. Production builds keep the policy constants, checked against `model-bundle.json` by a test.
- An experimental `metal` worker feature selects Candle 0.11's Metal device on macOS. It is compile-checked in CI on macOS ARM64 only. **It has not run on any hardware.**
- `--cpus` restricts the CLI and worker to chosen logical CPUs. That is a simulation of a weaker CPU, not a low-end measurement, and reports say so.

## Stand-in host

13th Gen Intel Core i9-13900H laptop: 6 performance cores with hyper-threading (logical CPUs 0–11) and 8 efficiency cores (12–19), 20 logical CPUs, 32 GB RAM, Windows 11 (10.0.26300), mains power, Balanced power plan. SSE4.1, SSE4.2, AVX and AVX2 are present. This is neither the Ryzen AI 9 HX 370 reference machine nor a low-end machine, so these numbers must not be read as a minimum-spec result. It ran a normal desktop session; each report records system CPU use just before and after the run.

All runs use the pinned bundle (model `8d75e9b9…`) and baseline (`22225341…`), a reproducible 440-query spread across every frozen partition and zero to four typed graphemes (`--samples 440`), 20 warm-up queries and four threads unless stated. Latency is the stream CLI round trip for a generation request after the desktop's context reset, so every query is a context miss. Accuracy is the six-slot top-six rate on the same 440 queries. RSS is a 10 ms sum over the parent and worker, so shared pages can count twice and brief peaks can be missed. Threads are the peak sum over both processes.

### Builds and threads

The SSE4.1 build adds `+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt` to the portable flags (x86-64-v2, which Gemini Lake and Jasper Lake support). The AVX build also adds `+avx`, which those Celerons do not have.

| Worker | Threads | Median | p95 | Maximum | Failed | Deadline misses | Top six | Cold start | Peak RSS | Peak threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Portable | 4 | 329 ms | 549 ms | 857 ms | 0 | 0 | 72.0% | 2.57 s | 544 MiB | 27 |
| Portable | 2 | 323 ms | 530 ms | 707 ms | 0 | 0 | 72.0% | 4.01 s | 548 MiB | 25 |
| Portable | 1 | 322 ms | 527 ms | 702 ms | 0 | 0 | 72.0% | 2.83 s | 544 MiB | 24 |
| SSE4.1 | 4 | 326 ms | 530 ms | 695 ms | 0 | 0 | 72.0% | 4.62 s | 469 MiB | 27 |
| AVX | 4 | 317 ms | 532 ms | 753 ms | 0 | 0 | 72.0% | 3.95 s | 533 MiB | 27 |
| AVX2 | 4 | 157 ms | 280 ms | 405 ms | 0 | 0 | 71.8% | 3.54 s | 543 MiB | 27 |

On this host, thread count made no measurable difference to generation latency, although the peak thread counts confirm the setting took effect. The SSE4.1 and AVX builds were no faster than portable and produced exactly the same generated words (`generated_sha256` `2fa0528c…` in all portable, SSE4.1 and AVX reports). That is consistent with Candle 0.11 enabling its SIMD Q8 kernels only for `target_feature = "avx2"` (`candle-core/src/quantized/k_quants.rs`). An intermediate build is therefore not a mitigation without kernel work. Only AVX2 changes speed, and its words differ, as the companion's notes already say. Whether threads help on a CPU with four slow cores is not known until such a machine is measured.

The refinement (reranking) benchmark ran once per production build on the full 1,760 queries:

| Worker | Successful | Median | p95 | Maximum | Failed | Deadline misses | Statistical top five | Refined top five |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Portable, 4 threads | 1,759 | 193 ms | 290 ms | 444 ms | 0 | 0 | 47.39% | 52.22% |
| AVX2, 4 threads | 1,759 | 98 ms | 201 ms | 324 ms | 0 | 0 | 47.39% | 52.33% |

The AVX2 `prediction_sha256`, `a1521815…`, is identical to the one in [the earlier Ryzen report](smol-avx2-results.json), so the AVX2 build ranked all 1,760 queries identically on both CPUs.

### Smaller search

The experiment worker ran with the portable flags and four threads. At the policy limits it produced the same words as the production worker.

| Beam width | Forward evaluations | Median | p95 | Maximum | Top six | Neural fill | Failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (policy) | 64 (policy) | 327 ms | 533 ms | 682 ms | 72.0% | 95.7% | 0 |
| 4 | 32 | 234 ms | 347 ms | 496 ms | 66.4% | 86.3% | 0 |
| 2 | 16 | 171 ms | 268 ms | 344 ms | 61.4% | 76.0% | 0 |

The statistical six alone score 45.9% on these queries. Halving the search cuts median latency by about 29% and costs 5.6 points of top-six accuracy. The AVX2 build at the policy limits is both faster (157 ms) and more accurate (71.8%) than either reduced search, so on any CPU with AVX2, the AVX2 build is the mitigation to use. Reduced search is only relevant to CPUs without AVX2, and its value there is unmeasured.

### Simulated weak CPU

Restricting the CLI and worker to two efficiency cores (logical CPUs 12 and 13) with two threads was meant to simulate a weaker CPU. It is a simulation, not a measurement of a Celeron: Gracemont efficiency cores are faster than Gemini Lake or Jasper Lake cores and do have AVX2, though the portable worker does not use it, and this laptop's clocks and Windows scheduling under restriction are unknown.

| Worker | Successful | Median | p95 | Maximum | Failed requests | Deadline misses | Top six |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Portable | 59 / 440 | 1,899 ms | 2,008 ms | 2,039 ms | 821 | 824 | 38.2% |
| Experiment, width 4, 32 evaluations | 56 / 440 | 1,903 ms | 1,988 ms | 2,013 ms | 823 | 825 | 38.2% |

All but one of the 880 reranking requests and 87% of generation requests timed out. The few successful generation replies returned no words, so the six slots fell below the statistical six. The benchmark retries after each failure, so most requests also followed a fresh model load.

A control run restricted to two performance cores (logical CPUs 0 and 2, two threads, 40 queries, `sim-2pcore-portable-2t-40.json`) also timed out on all 80 requests, although the same worker with one thread and no restriction answered in 322 ms. So the restriction itself, on this laptop and Windows build, causes the timeouts; it does not model a slower CPU. **These simulation results say nothing reliable about low-end hardware and should not be used for the minimum spec.** The cause was not investigated further. `--cpus` stays in the harness for Linux and other machines, but its results need the same control before they are trusted.

### Equivalence of the default build

The branch's default portable build, from the same source as the experiment and Metal features, produced the same 440 generated words as the `main` build (`2fa0528c…`, `host-branch-portable-4t.json`). The Rust tests check that the policy limits match the manifest in every build.

## Metal experiment

`cargo build --manifest-path neural/Cargo.toml -p switchify-smol-worker --features metal --release --locked` on macOS builds a worker that loads the same GGUF onto the first Metal device and runs the same forward pass there. It reports `accelerated: false`, so it is passed as `--worker`. CI compiles it on macOS ARM64. It was not run: no Mac was available, and GitHub's macOS runners are not a fair place to measure a GPU.

To measure it on an Apple Silicon Mac, run the harness once with the portable worker and once with the Metal worker, using the same arguments. Record power alongside each run with `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` in another terminal. Compare latency, RSS and power. As with the CPU builds, bitwise parity is not expected: compare `generated_sha256` to see whether the candidate ordering changed, and the accuracy to see whether it matters. Q8 matrix multiplication, RMS norm, RoPE, softmax and index selection all have Metal kernels in Candle 0.11, but whether every operation the worker uses runs on Metal, and whether Metal is faster for batches of at most eight rows, is unknown until it runs.

## CUDA and DirectML/ONNX Runtime (analysis only)

Neither was implemented.

**CUDA.** Candle 0.11 has a `cuda` feature with quantized kernels, so a CUDA worker would be a similar feature-gated build. It is a poor fit for this issue. Building needs the CUDA toolkit; running needs an NVIDIA GPU and a matching driver, and shipping cuBLAS-class libraries adds hundreds of megabytes under NVIDIA's redistribution terms. The machines the minimum spec is about, Celeron, Pentium Silver and older i5 laptops, almost never have NVIDIA GPUs, and machines that do already have fast CPUs. The workload is also small: a 135M-parameter model, one context of at most 64 tokens and batches of at most eight rows, where launch and transfer overhead can dominate. CUDA should not be pursued unless measurements show a CPU gap on NVIDIA machines specifically.

**DirectML or ONNX Runtime.** This would reach integrated Intel and AMD GPUs on Windows, which low-end machines do have, but it is a different inference stack rather than a backend switch. The SmolLM2 model would need an ONNX export and a different quantization, so the pinned Q8 GGUF, its converter and hashes, the batched attention cache and the exact early stop would all need reimplementing and a new frozen qualification. The runtime and DirectML libraries add tens of megabytes. Microsoft now directs new Windows work to Windows ML on ONNX Runtime with vendor execution providers, and DirectML itself is no longer the focus of new development, so the choice of provider should be confirmed before any work. Integrated GPUs in Celeron-class machines are weak and share memory bandwidth with the CPU, so a speedup is not assured. This is only worth considering if measurements on the machines below fail by a wide margin that CPU mitigations cannot close.

## Provisional minimum spec

This is provisional. It rests on one fast laptop, and must be confirmed or replaced by the machines listed below.

- **x86-64 with AVX2, FMA and F16C:** supported with the AVX2 worker. On the stand-in it halves portable latency, with no failures and no missed deadlines. On the Ryzen reference it already met the reranking target.
- **x86-64 without AVX2** (Celeron and Pentium Silver class): do not offer the neural companion by default; keep statistical prediction, until a real machine shows it meets the 2 s deadline. This is a cautious default, not a measured failure. Even on a fast core the portable worker is about twice as slow as AVX2 (329 against 157 ms median for generation). SSE4.1 and AVX builds give nothing, thread count made no difference here, and halving the search costs 5.6 points of accuracy for a 29% gain. The CPU-restriction simulation was not usable as evidence.
- **Memory:** sampled process-tree peak RSS was 0.43–0.55 GiB in every configuration. Whether that fits alongside other applications on a 4 GB machine is unmeasured.
- **Apple Silicon and Windows on ARM:** unknown. Do not infer them from these results.
- **Threads:** keep the default of four. Nothing here supports changing it.

**GPU path:** do not start CUDA or DirectML/ONNX Runtime work. Keep the Metal build as an off-by-default experiment only until one Apple Silicon run compares it with the portable worker. If it is not clearly faster or lower in power, remove it. For x86 CPUs without AVX2, the more promising lever appears to be CPU kernels: Candle has no SSE4.1 or SSSE3 Q8 dot product. That would need its own issue, and any production change its own frozen qualification.

## Machines still to measure

Each needs one run of the harness per mode, committed as JSON under `docs/low-end-results/`:

1. x86-64 without AVX2, Celeron or Pentium Silver class (Gemini Lake or Jasper Lake): portable worker at 4, 2 and 1 threads; the SSE4.1 build; and the experiment worker at reduced limits.
2. An older 4-core x86-64 with AVX2, such as an 8th-generation i5: portable plus AVX2 workers.
3. An Apple Silicon Mac, M1 class: portable worker, then the Metal worker with `powermetrics`.
4. An Intel Mac, and Windows on ARM running the x64 build under emulation, if available.

## Running the harness

Build as CI does, prepare the pinned inputs with `scripts/prepare_generation_qualification.py`, and install `psutil==7.0.0` and `regex==2025.11.3`. Close other work and keep the machine on mains power. Then, with `.exe` on Windows:

```text
python scripts/hardware_benchmark.py --label <machine>-portable-4t --cli <cli> --worker <portable-worker> --baseline <english.sqlite> --bundle <bundle> --samples 440 --output docs/low-end-results/<machine>-portable-4t.json
```

Add `--accelerated-worker <avx2-worker>` on AVX2 machines, `--threads 1|2` for the thread sweep, and `--training data/aac-oanc/prepared/candidate.txt` (from `scripts/aac_experiment.py --prepare-only`) to add the refinement benchmark. Omit `--samples` for the full 1,760-query workload. For the intermediate builds and mitigation experiments:

```text
RUSTFLAGS="-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt" cargo build --manifest-path neural/Cargo.toml -p switchify-smol-worker --release --locked --target-dir target/smol-sse41
cargo build --manifest-path neural/Cargo.toml -p switchify-smol-worker --features experiment --release --locked --target-dir target/smol-experiment
python scripts/hardware_benchmark.py ... --worker target/smol-experiment/release/switchify-smol-worker --beam-width 4 --max-forward-evaluations 32
```

Add `+avx` to the SSE4.1 flags for the AVX build, and `+crt-static` on Windows as packaging does. Never use `target-cpu=native`. The intermediate builds are not packaged and report `accelerated: false`, so pass them as `--worker`.

Reports contain hardware facts, aggregate counts, timing, hashes and digests, never words, prompts or the hostname. No keyboard or pointer input is injected.
