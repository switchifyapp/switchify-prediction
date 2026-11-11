# Google Books n-gram pilot

This experiment compares a fixed Google Books sample against the shipped model.
It keeps the predictor unchanged and scores only the existing development sets.
It does not promote a model or score the test sets.

## Source and scope

The source is Google Books English One Million, version `20090715`. Publication
years 1980 through 2008 are retained. This is older book language, not current
conversation. The newer full exports are much larger; this bounded first trial
does not represent the full Google Books corpus or its newer editions.

`google-books-source-manifest.json` pins 18 archives by URL, size, SHA-256 and
publisher MD5. Downloads initially matched the publisher's MD5 headers. The
runner checks SHA-256 before parsing and ZIP CRCs while reading every member.
The compressed sample is 2,445,733,303 bytes:

- All 10 unigram shards.
- Bigram shards 0, 25, 50 and 75 of 100.
- Trigram shards 0, 50, 100 and 150 of 200.

Shards are not assumed to be random. These fractions describe files, not token
coverage. Counts are not multiplied to pretend the missing shards were present.
Google version 1 may include n-grams across sentence boundaries. Punctuation
tokens are rejected rather than joined, but original sentence boundaries cannot
be recovered or guaranteed from these aggregates.

Only lowercase ASCII words with optional internal apostrophes are retained.
Curly apostrophes become ASCII. Tags, digits, punctuation and non-ASCII words
are rejected. Year and case variants are aggregated. Retain the 40,000 most
frequent unigrams, then at most 200,000 bigrams and 200,000 trigrams, each with
at least 20 occurrences in the chosen years and all words in that vocabulary.
Ties use lexical order. These choices are fixed in `google-books-policy.json`
before any scoring.

## Comparisons

The six models are production, Google alone, 5%/20%/50% Google blends, and a
20% blend adding only Google's unigrams. The latter isolates the effect of
adding the retained contextual counts at the same vocabulary/frequency mixture.
Shares refer to unigram probability mass, not file sizes or an independent
interpolation weight. The predictor's 0.1/0.3/0.6 weights remain unchanged.

For baseline unigram mass B, Google mass G and Google percentage p, multiply
baseline counts by `(100-p)*G` and Google counts by `p*B`, reducing both integer
coefficients by their greatest common divisor. Apply the same coefficients to
every n-gram order. This preserves low-frequency contextual evidence without
rounding it away. The unigram control omits the Google bigram/trigram additions.
Unigram consistency and stored count limits are validated before conversion.

This exact integer scaling changes absolute count units. Personal learning is
disabled for comparison. A candidate must not ship with existing learning
weights without separately calibrating counts and checking personal adaptation.

The runner verifies all six frozen evaluation hashes against the production
manifest and validates the baseline's production fingerprint. Google aggregate
counts cannot identify or remove whole evaluation sentences. Existing suites
are known regressions, not independently unseen user communication. AAC examples
are imagined communication written by crowd workers. Model savings are a proxy
and do not measure actual switch use.

## Run

Use the corpus preparation from [the source benchmark](corpus-benchmark.md).
From the repository root, with Rust 1.97.1 and Python 3.9 or later:

```sh
cargo build --release --locked
python scripts/google_books_experiment.py \
  --baseline /path/to/shipped/english.sqlite \
  --output artifacts/google-books-pilot-v1
```

The runner downloads any missing archives, checks their pinned hashes, parses
with four workers by default, builds models and scores AAC, general English
and task-conversation development sets. Choose an empty output directory.
`--workers 1` reduces memory pressure; valid values are one through eight.
`--prepare-only` stops after importing and validating the Google model.
Reports remain marked `complete: false` until every comparison completes.

The completed report includes raw source row counts, retained n-gram counts and
mass, all accuracy/coverage/selection/latency metrics, policy and executable
hashes, recipe coefficients and output hashes. Scoring is sequential after
parallel parsing completes. Timing is descriptive for this machine.

## Attribution

The publisher licenses the compilation under CC BY 3.0. Retain the source
manifest, [Google Books notice](corpus-notices/google-books-notice.txt), licence
links and attribution when redistributing derivatives. Other sources keep their
existing notices. The code's MIT licence does not relicense corpus data.

Source: [Google Books bulk exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html).
