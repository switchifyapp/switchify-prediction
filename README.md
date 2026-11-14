# Switchify Prediction

Builds the TextSlinger runtime that Switchify PC bundles for word suggestions.

Switchify PC is moving its word suggestions to the n-gram model from
[TextSlinger](https://github.com/kdv123/textslinger). This repository packages a
self-contained runtime for it, for each platform Switchify PC supports, and
publishes it with each release. Switchify PC pins the archives by hash at build
time; installed apps never download anything.

Each release carries `textslinger-runtime-<target>.zip` and a `.sha256` file for:

- `x86_64-pc-windows-msvc`, also used on Windows ARM64 under x64 emulation
- `aarch64-apple-darwin`
- `x86_64-apple-darwin`

See [docs/textslinger-runtime.md](docs/textslinger-runtime.md) for what is inside,
how to build it, and the temporary upstream patch.

## Releasing

1. Set `VERSION` and add `docs/releases/v<version>.md` with the release notes.
2. Merge to `main`, then push the tag `v<version>`.
3. The release workflow builds every target and creates a draft release with the
   archives and checksums. Review it, then publish.

## History

Releases up to v0.5.3 contained Switchify's own n-gram and neural prediction
engine, which Switchify PC used before TextSlinger. That code remains in this
repository's history and in its earlier tags and releases.

## Licence

MIT, see [LICENSE](LICENSE). The runtime archives contain third-party components
under their own licences, listed in each archive's `notices/NOTICE.md`.
