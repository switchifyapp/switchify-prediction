# Dependency and deployment notes

This companion is opt-in. It does not change the root crate's dependencies or audit policy.

The neural dependency audit uses `cargo audit --file neural/Cargo.lock --deny warnings --ignore RUSTSEC-2024-0436`. This single exception covers `paste` 1.0.15, an unmaintained build-time macro dependency of Candle. The advisory reports maintenance status, not a known memory-safety exploit. Replacement requires an upstream-compatible Candle update or an independently reviewed patch. Review this exception at each dependency update. No other advisory is ignored. The root audit still denies all warnings.

Worker files are executable code supplied by the embedding application. Install them in a directory writable only by trusted users. The model manifest is not a signature; compatibility comes from hashes compiled into the library. Runtime loading hashes the exact owned bytes used by inference. The worker has no network API, file writes, clipboard access or input-injection code. Process isolation bounds hangs and crashes; it is not an operating-system security sandbox.

Context, prefixes and suggestions are kept in process memory and private pipes. Error messages deliberately omit their contents. The CLI writes requested suggestions to stdout, so callers must avoid logging or redirecting that stream when handling private text. Reset drops context buffers and invalidates pending results; it does not promise cryptographic memory erasure or protection against OS paging, crash dumps or a privileged debugger.

The default four-thread cap applies to the worker's Rayon pool. Accelerated binaries require AVX2, FMA and F16C and must be launched through the portable parent, which checks CPU support. Portable builds must not inherit `target-cpu=native` or additional ISA flags. Packaged executables are unsigned and are not desktop installers. Model redistribution is separate and must retain the Apache 2.0 license and model card.
