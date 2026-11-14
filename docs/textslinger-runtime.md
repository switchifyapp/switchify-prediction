# TextSlinger runtime

Switchify PC predicts words with the n-gram model from
[TextSlinger](https://github.com/kdv123/textslinger). This repository builds the
runtime it bundles, for each platform, and publishes it with each release as
`textslinger-runtime-<target>.zip` plus a `.sha256` file. Switchify PC pins the
archive by hash, the same way it pins the neural worker. Installed apps never
download it.

## What is inside

| Path | Contents |
|---|---|
| `python.exe` or `bin/python3.10` | Standalone CPython 3.10.22 from python-build-standalone 20261009, without tests, tkinter, idlelib, ensurepip, lib2to3 or turtledemo |
| site-packages | numpy, regex, textslinger and KenLM; no PyTorch |
| `model/lm_dec19_char_tiny_12gram.kenlm` | The binary tiny 12-gram character model, redistributed with the TextSlinger authors' permission |
| `manifest.json` | Target, interpreter and model paths, the model's alphabet, component versions |
| `notices/` | `NOTICE.md`, the CPython licence and KenLM's licence texts |

Targets: `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` and `x86_64-apple-darwin`.
Switchify PC also runs the Windows x64 runtime on Windows ARM64, using Windows' x64
emulation, because there is no Python 3.10 or numpy 1.26 build for ARM64. The macOS
KenLM build targets macOS 13.3, the oldest version Switchify PC supports.

textslinger requires Python 3.10, which reached end of life in October 2026.
[kdv123/textslinger#3](https://github.com/kdv123/textslinger/issues/3) asks for newer
versions to be allowed.

## Building

`textslinger/runtime.json` pins every input: the Python archive for each target,
the package wheels and KenLM source by SHA-256, the KenLM build tools, and the model.
On Windows x64 or macOS, with uv, git, Python 3.10 or newer and a C++ compiler
(Visual Studio C++ tools or Xcode command line tools):

```bash
python scripts/package_textslinger.py
```

The script builds KenLM with `MAX_ORDER=12`, which this model needs. It assembles
the runtime and applies the temporary patch below. It then checks that the runtime
loads the model and predicts without importing PyTorch, and only then writes
`artifacts/textslinger/`. Downloads are cached in `target/textslinger/`.

## Temporary patch

textslinger 0.4.0 imports PyTorch when it loads, even for the n-gram model.
`textslinger/lazy-torch-imports.patch` defers those imports. It is the change in
[kdv123/textslinger#2](https://github.com/kdv123/textslinger/pull/2). The build
fails if the patch does not apply. When a textslinger release includes the change,
update the pin and delete the patch and its step in the script.

## CI and releases

`.github/workflows/textslinger.yml` runs the packaging unit tests and builds every
target on pull requests and on `main`. The release workflow calls it, so each
draft release carries the three archives and their checksums.
