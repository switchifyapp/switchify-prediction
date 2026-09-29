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

# Taskmaster-1 conversational source

The improved baseline also derives counts from USER utterances in the
human-written `TM-1-2019/self-dialogs.json` dataset, by Bill Byrne, Karthik
Krishnamoorthi, Chinnadhurai Sankar, Arvind Neelakantan, Amit Dubey, Kyu-Young Kim
and Andy Cedilnik of Google LLC. The dataset's [copyright notice](https://github.com/google-research-datasets/Taskmaster/blob/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/README.md)
makes it available under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Citation: Byrne et al. (2019), [Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset](https://aclanthology.org/D19-1459/), EMNLP-IJCNLP.

We modify the source by selecting USER turns, normalizing text, splitting
sentences, deduplicating, excluding held-out sentences, and extracting n-gram
counts. The official conversation partitions remain separate; assistant turns
are not used. The source is human-written task simulation, not real AAC user
history. We make no claim that it comprehensively represents AAC communication.
The pinned upstream notice, source hashes and partition hashes are recorded in
`source-manifest.json`. The resulting database includes material derived from this
CC BY 4.0 source; retain this attribution when redistributing it. Our MIT license
covers the code, not a relicensing of the source data.

# Production AAC/OANC model

The default `en-aac-oanc-v1` production model (and its retained experiment) additionally derives counts from:

- Keith Vertanen and Per Ola Kristensson (2011), *The Imagination of Crowds:
  Conversational AAC Language Modeling using Crowdsourcing and Large Data
  Sources*, EMNLP, pp. 700–711. [Source](https://aactext.org/imagine/),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Only the official `sent_train_aac.txt` training file contributes counts;
  development/test files are used for evaluation and overlap exclusion.
- Nancy Ide and Keith Suderman (2007), *The Open American National Corpus
  (OANC)*. [Source and current terms](https://anc.org/data/oanc/).
  The spoken GrAF subset comprises Charlotte narratives/conversations and
  Switchboard transcripts. The historical OANC notice credits Switchboard:
  Copyright (c) 1997–2002 Trustees of the University of Pennsylvania.
  Full historical notices and the current publisher grant are retained in
  `corpus-notices`; see `docs/aac-experiment.md` for the distinction.

Modified by Switchify prediction contributors, 2026-09-29: select spoken
utterance spans, normalize case and Unicode, split sentences, deduplicate,
remove held-out overlaps, deterministically sample, weight training counts and
extract word n-grams. These are statistical derivatives, not original source
transcripts. Original URLs and exact hashes are in `aac-source-manifest.json`.
No author or institution endorses Switchify. Retain this attribution, the
source manifests and accompanying notices when redistributing the production or experimental
model. The MIT code licence does not relicense these sources.
