# The n-gram baseline file

Issue #40 asked whether the statistical baseline should leave SQLite. It
should. This report profiles the SQLite open, describes the n-gram file that
replaces it, shows that both give identical predictions, and recommends
adopting the file. Measurements are in `baseline-format-results.json`.

## Where the SQLite open spent its time

The production baseline `english.sqlite` (29.8 MB, SHA-256
`222253417d0a7a705823ffb7e599a3bcf5d5d3daf4a9d76161ac6b3e555aeaad`) is
read-only and pinned by hash. Even so, every open ran `PRAGMA integrity_check`,
read all 539,151 rows into `BTreeMap<Vec<String>, BTreeMap<String, i64>>`,
parsed each JSON-encoded context, re-validated each row, and walked every
n-gram against the unigrams. SQLite was only a container.

Measured by the ignored test `profile_sqlite_baseline_open` at commit `1b5fc54`
in a release build, Windows 11 Pro 10.0.26200, AMD Ryzen AI 9 HX 370, median of
three runs:

| Phase | ms |
| --- | ---: |
| Open and begin transaction | 0.3 |
| `PRAGMA integrity_check` | 954 |
| Schema check (includes the integrity check) | 878 |
| Row load, JSON parsing and row validation | 1,099 |
| Vocabulary and unigram consistency walk | about 700 |
| **`Predictor::open` total** | **2,844** |

The CLI `predict` process peaked at 134.4 MiB working set and took 2.1–2.2 s
end to end. The maps held 8.4 MB of keys and counts but many more bytes of
tree nodes and small strings.

## The file

`src/ngrams.rs` defines a compact, immutable file, extension `.ngrams`. All
integers are little-endian; the layout is documented at the top of the module:

- a 16-byte magic, format version 1, and a JSON metadata block with the
  language, kind and the same provenance string the SQLite `metadata` table
  held;
- the vocabulary: `N` words sorted bytewise, as `N + 1` offsets into one UTF-8
  blob. A word's position is its id, so the unigram table is an array of `N`
  counts indexed by id, and prefix completion is two binary searches that give
  the id range of the words starting with the prefix;
- bigrams: `N + 1` start indices by context word id into arrays of word ids
  and counts, ids ascending within a context;
- trigrams: `T` sorted `(id, id)` contexts, `T + 1` start indices, and the same
  id and count arrays;
- a trailing SHA-256 of everything before it.

The predictor reads the whole file into memory and indexes those bytes: a
lookup is a binary search over ids, nothing is parsed into maps, and the file's
8.5 MB is the model's whole memory. Per-context totals are summed once at open,
in the same word order as before, so interpolation sees the same floats.

**Why not memory-map it.** The open hashes the whole file, so every page is
touched anyway. Reading the 8.5 MB takes 2 ms; mapping it would save at most
that, and would add an `unsafe` mapping whose bytes can change under the
program if the file does. A plain read keeps the parser entirely safe Rust.

**Why not an `fst` vocabulary.** With 21,674 words, a sorted blob with offsets
gives exact lookup and prefix ranges in about 15 string comparisons, with no
new dependency. No crate was added.

### Validation

`Ngrams::parse` checks, in order: the magic and that the trailer equals the
SHA-256 of the rest; the version; that every array fits the file and nothing
follows the last one; the metadata (language `en`, kind `baseline`); that the
vocabulary is non-empty, strictly ascending and made of valid normalized
words; that every count is positive and within the SQLite schema's limit;
that ids are in range and strictly ascending within each context; that
trigram contexts are strictly ascending and non-empty; and that no n-gram
count exceeds its word's unigram count. These are the rules the SQLite
validation enforced, minus the ones the layout makes impossible (a word
without a unigram, or a vocabulary that differs from the unigrams).

The logical fingerprint is computed from the file in the SQLite order, so
`validate` reports the same `logical_sha256` for both formats, and
`validate --production` accepts the converted production model.

`tests/ngrams.rs` rejects truncation at twelve points, a file one byte longer,
padding before the trailer, single-bit flips every 97 bytes, and resigned
mutations of the version, metadata, word order, word validity, counts, start
indices, ids and trigram order, through `validate`, `Predictor::open` and
`validate_english`.

### Compatibility

- `Predictor::open` accepts either format and tells them apart by the first
  16 bytes. A SQLite baseline is validated exactly as before, then encoded in
  memory into the same representation, so there is one prediction path.
- `build` still writes SQLite by default; `build --format ngrams` writes the
  file for custom models. `convert --input x.sqlite --output x.ngrams` converts
  a baseline and fails, leaving no file, unless the logical fingerprints match.
  Neither replaces an existing file.
- `Validation` gains a `format` field (`sqlite` or `ngrams`).
- Personal learning is unchanged and stays on SQLite, as scoped.
- The production artifact is unchanged: `scripts/artifacts.py` still builds
  and ships `english.sqlite`.

## Parity

Every existing test runs against the new representation, since SQLite
baselines are now encoded into it. `tests/ngrams.rs` also checks that a
converted file, a file built from the same text, and the SQLite original give
bit-identical scores and phrases, with and without learning, and that
converting and building produce identical bytes.

The ignored test `production_parity` compared the production `english.sqlite`
with its conversion on the frozen held-out AAC, conversation and general test
partitions (1,827 sentences): for every word, with 0–2 typed letters, the top
eight words and phrases, as word and score bits, first without learning and
then after learning half of the sentences into a shared personal database.
That is 40,559 queries per phase, 81,118 in all. Every comparison was identical.

## Results

Same machine and build as above. The n-gram file is
`c6df388c845354a059333b1e4abe68719e5a7016e8514c44581e14ed16bf964a`, 8,477,752
bytes, converted from the production model.

### Opening

| | SQLite | N-gram file |
| --- | ---: | ---: |
| File | 29.8 MB | 8.5 MB |
| Read file | | 2.2 ms |
| SHA-256 of the file | | 4.2 ms |
| Parse, hash and all checks | | 21.8 ms |
| **`Predictor::open`** | **2,844 ms** | **19.7 ms** |
| `validate` (adds the logical fingerprint) | about 2,900 ms | 89 ms |
| CLI `predict` process, peak working set | 134.4 MiB | 14.0 MiB |
| CLI `predict` process, wall time | 2,142 ms | 27 ms |

### Predicting and learning

From the ignored tests `prediction_cost_on_production_baseline` and
`learning_cost_on_production_baseline`, release build, one run each. Times are
per call. Learning cost is dominated by the personal database check and is
unchanged.

| | SQLite at `1b5fc54` | N-gram file |
| --- | ---: | ---: |
| `predict("", "")`, no learning | 4.4 ms | 3.8 ms |
| `predict_phrases("", "")`, no learning | 16.6 ms | 9.1 ms |
| `predict("i want to", "")` | 9.2 ms | 3.7 ms |
| `predict("the", "s")` | 0.9 ms | 0.3 ms |
| `predict("", "")`, 2,000 new words learned | 7.9 ms | 3.8 ms |
| `predict("", "")`, 40,000 new words learned | 27.2 ms | 22.6 ms |
| One learn, 2,000 new words | 15 ms | 11 ms |
| One learn, 40,000 new words | 173 ms | 190 ms |

Single runs on an otherwise idle desktop. Prediction is as fast or faster: a
binary search over ids replaces string-keyed tree lookups. Learning is within
noise, as expected.

Not measured: macOS and a low-end machine, which #39 covers. A 2 s open on a
fast desktop would be far longer on a Celeron-class laptop; 20 ms would not.

## Recommendation

Adopt the n-gram file as the shipped baseline. The SQLite open was almost
entirely validation of a file that is already pinned by hash, and the file
format makes that validation cheap. Suggested steps, each its own issue:

1. **Bundle.** Have `scripts/artifacts.py` convert the built model and ship
   `english.ngrams` in the model bundle, with its hash in `validation.json`,
   and have `smoke_package.py` exercise it. Keep `english.sqlite` in the bundle
   for one release so older CLIs and pins keep working.
2. **Switchify PC.** Pin the file in `scripts/prediction-model.json`, resource
   it in `tauri.conf.json`, update `check-packaged-prediction.mjs` and the
   hash in `Model::open`. Prediction becomes ready at startup instead of two
   seconds later, and the installer shrinks by 21 MB.
3. **Later.** Once nothing opens a SQLite baseline, make `build` write the
   n-gram file by default and drop SQLite from the baseline path. `rusqlite`
   stays for personal learning.
