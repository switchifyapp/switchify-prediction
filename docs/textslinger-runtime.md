# TextSlinger runtime

Switchify PC is moving its word suggestions to the n-gram model from
[TextSlinger](https://github.com/kdv123/textslinger). This repository builds the
runtime it bundles, for each platform, and publishes it with each release as
`textslinger-runtime-<target>.zip` plus a `.sha256` file. Switchify PC pins the
archive by hash at build time. Installed apps never download it.

## What is inside

| Path | Contents |
|---|---|
| `python.exe` or `bin/python3.10` | Standalone CPython 3.10.22 from python-build-standalone 20261009, without tests, tkinter, idlelib, ensurepip, lib2to3 or turtledemo |
| site-packages | numpy, regex, textslinger and KenLM; no PyTorch |
| `model/lm_dec19_char_tiny_12gram.kenlm` | The binary tiny 12-gram character model, redistributed with the TextSlinger authors' permission |
| `manifest.json` | Target, interpreter and model paths, the model's alphabet, component versions |
| `notices/` | `NOTICE.md`; the licences of CPython and the libraries built into it; KenLM's licences, including its bundled double-conversion and integer_to_string code; and the exact KenLM source the library was built from |

Targets: `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` and `x86_64-apple-darwin`.
Switchify PC also runs the Windows x64 runtime on Windows ARM64, using Windows' x64
emulation, because there is no Python 3.10 or numpy 1.26 build for ARM64. Every
component's macOS deployment target is older than macOS 13.3, the oldest version
Switchify PC supports.

textslinger requires Python 3.10, which reached end of life in October 2026.
[kdv123/textslinger#3](https://github.com/kdv123/textslinger/issues/3) asks for newer
versions to be allowed.

## Building

`textslinger/runtime.json` pins every input: the Python archive for each target,
the package wheels and KenLM source by SHA-256, the KenLM build tools, and the model.
On Windows x64 or macOS, with uv, git, Python 3.10.12 or newer and a C++ compiler
(Visual Studio C++ tools or Xcode command line tools):

```bash
python scripts/package_textslinger.py
```

The script builds KenLM with `MAX_ORDER=12`, which this model needs. KenLM's
optional compression support is turned off, so the build never links a library
that happens to be installed on the build machine. On Windows the C++ runtime is
linked statically, so no Visual C++ Redistributable is needed. The script assembles
the runtime and applies the temporary patch below. Before it writes
`artifacts/textslinger/`, it checks two things:

- the runtime loads the model and predicts without importing PyTorch;
- every native file depends only on libraries inside the runtime or ones the operating system provides.

The second check reads Windows import tables, or the macOS load commands through
`otool`. It catches dependencies that the build machine happens to satisfy.
Downloads are cached in `target/textslinger/`, and network errors are retried.

The model comes from this repository's [`textslinger-model-1`](https://github.com/switchifyapp/switchify-prediction/releases/tag/textslinger-model-1)
release first, and from OSF if that fails. That release holds an unchanged copy of the
OSF file, published with the TextSlinger authors' permission, so a build does not depend
on OSF being reachable. Whichever source is used must match the pinned size and hash.
A different model gets a new `textslinger-model-<n>` release; existing ones never change.

The CPython licence texts in `textslinger/licenses/python-build-standalone/` come from
python-build-standalone's full 3.10.22 archives, which carry the same texts for every
target. Because 3.10.22 is the last 3.10 release, they are stored here rather than
downloaded.

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
