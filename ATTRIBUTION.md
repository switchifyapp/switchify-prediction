# Corpus attribution

The baseline derives from the English sentence corpus in
[AACTools/WorldAlphabets](https://github.com/AACTools/WorldAlphabets), pinned to
`dfec80e9d5641b2ad080f0780a72fc108333d164`.

Upstream identifies the English corpus as `tatoeba-sentences-cc0`, drawn from
[Tatoeba](https://tatoeba.org) per-language sentence exports. Its source manifest
states that Tatoeba sentences are CC-BY 2.0 FR and that the CC0 subset is used
where the language has substantive coverage. We retain that attribution and
license note verbatim in `source-manifest.json`; upstream marks `verify: false`.
Do not interpret the software's MIT license as a license for every source corpus.

The database transforms the supplied text into normalized vocabulary and n-gram
counts. The manifest records the source URLs, commit, SHA-256 hashes, source
metadata, and build settings. Both this attribution and the manifest accompany
each downloadable database.

License references: [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
and [CC-BY 2.0 France](https://creativecommons.org/licenses/by/2.0/fr/).
