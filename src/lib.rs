//! Offline English word prediction. Learning accepts non-overlapping completed segments.
use rusqlite::{Connection, OpenFlags, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub mod corpus;
pub mod evaluation;
pub mod ngrams;
pub mod production;

/// The largest stored count, as the SQLite schema's check constraint allows.
pub(crate) const MAX_COUNT: u64 = (i64::MAX / 10) as u64;
/// A stored word: non-empty, normalized, and one word to the tokenizer.
pub(crate) fn valid_word(w: &str) -> bool {
    !w.is_empty() && normalize(w) == w && sentences(w) == vec![vec![w.to_string()]]
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("database operation failed: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("file operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid database or input: {0}")]
    Invalid(String),
    #[error("JSON operation failed: {0}")]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn normalize(text: &str) -> String {
    text.replace('’', "'").to_lowercase().nfc().collect()
}

/// Words grouped into sentences; punctuation/newlines prevent cross-sentence n-grams.
pub fn sentences(text: &str) -> Vec<Vec<String>> {
    let normalized = normalize(text);
    let chars: Vec<char> = normalized.chars().collect();
    let mut result = Vec::new();
    let mut sentence = Vec::new();
    let mut word = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_alphabetic()
            || (unicode_normalization::char::is_combining_mark(c) && !word.is_empty())
            || (c == '\''
                && !word.is_empty()
                && chars.get(i + 1).is_some_and(|x| x.is_alphabetic()))
        {
            word.push(c);
        } else {
            if !word.is_empty() {
                sentence.push(std::mem::take(&mut word));
            }
            if matches!(c, '.' | '!' | '?' | '\n' | '\r') && !sentence.is_empty() {
                result.push(std::mem::take(&mut sentence));
            }
        }
    }
    if !word.is_empty() {
        sentence.push(word);
    }
    if !sentence.is_empty() {
        result.push(sentence);
    }
    result
}

#[derive(Default, Clone)]
struct Counts(BTreeMap<Vec<String>, BTreeMap<String, i64>>);
impl Counts {
    fn add_text(&mut self, text: &str) {
        for words in sentences(text) {
            for (i, word) in words.iter().enumerate() {
                for n in 0..=i.min(2) {
                    *self
                        .0
                        .entry(words[i - n..i].to_vec())
                        .or_default()
                        .entry(word.clone())
                        .or_default() += 1;
                }
            }
        }
    }
    fn merge(&mut self, other: &Self, weight: i64) {
        for (context, words) in &other.0 {
            for (word, count) in words {
                *self
                    .0
                    .entry(context.clone())
                    .or_default()
                    .entry(word.clone())
                    .or_default() += count * weight;
            }
        }
    }
    fn load(db: &Connection) -> Result<Self> {
        let mut result = Self::default();
        let mut stmt =
            db.prepare("SELECT context, word, count FROM counts ORDER BY context, word")?;
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (context, word, count) = row?;
            let encoded = context;
            let context: Vec<String> = serde_json::from_str(&encoded)?;
            if context.len() > 2
                || count <= 0
                || count as u64 > MAX_COUNT
                || encoded != serde_json::to_string(&context)?
                || !valid_word(&word)
                || context.iter().any(|w| !valid_word(w))
            {
                return Err(Error::Invalid("invalid n-gram".into()));
            }
            result.0.entry(context).or_default().insert(word, count);
        }
        Ok(result)
    }
    fn write(&self, db: &Connection) -> Result<()> {
        let mut stmt = db.prepare("INSERT INTO counts(context,word,count) VALUES (?1,?2,?3) ON CONFLICT(context,word) DO UPDATE SET count=count+excluded.count")?;
        for (context, words) in &self.0 {
            let context = serde_json::to_string(context)?;
            for (word, count) in words {
                db.execute("INSERT OR IGNORE INTO vocabulary(word) VALUES (?1)", [word])?;
                stmt.execute(params![context, word, count])?;
            }
        }
        Ok(())
    }
}
fn initialize(db: &Connection, kind: &str, provenance: &str) -> Result<()> {
    db.execute_batch("PRAGMA user_version=1;
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE vocabulary(word TEXT PRIMARY KEY) WITHOUT ROWID;
        CREATE TABLE counts(context TEXT NOT NULL,word TEXT NOT NULL REFERENCES vocabulary(word),count INTEGER NOT NULL CHECK(typeof(count)='integer' AND count>0 AND count<=922337203685477580),PRIMARY KEY(context,word)) WITHOUT ROWID;
        CREATE INDEX counts_word ON counts(word);
        CREATE TABLE imports(hash TEXT PRIMARY KEY) WITHOUT ROWID;")?;
    for (key, value) in [
        ("language", "en"),
        ("kind", kind),
        ("provenance", provenance),
    ] {
        db.execute("INSERT INTO metadata VALUES (?1,?2)", [key, value])?;
    }
    Ok(())
}
fn readonly(path: &Path) -> Result<Connection> {
    Ok(Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?)
}
fn check(db: &Connection) -> Result<String> {
    let version: u32 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version != 1 {
        return Err(Error::Invalid("unsupported schema version".into()));
    }
    let language: String =
        db.query_row("SELECT value FROM metadata WHERE key='language'", [], |r| {
            r.get(0)
        })?;
    if language != "en" {
        return Err(Error::Invalid("unsupported language".into()));
    }
    for (table, expected) in [
        ("metadata", vec![("key", "TEXT", 1), ("value", "TEXT", 0)]),
        ("vocabulary", vec![("word", "TEXT", 1)]),
        (
            "counts",
            vec![
                ("context", "TEXT", 1),
                ("word", "TEXT", 2),
                ("count", "INTEGER", 0),
            ],
        ),
        ("imports", vec![("hash", "TEXT", 1)]),
    ] {
        let mut stmt = db.prepare(&format!("PRAGMA table_info({table})"))?;
        let actual: Vec<(String, String, i32)> = stmt
            .query_map([], |r| Ok((r.get(1)?, r.get(2)?, r.get(5)?)))?
            .collect::<std::result::Result<_, _>>()?;
        if actual
            .iter()
            .map(|(name, ty, pk)| (name.as_str(), ty.as_str(), *pk))
            .collect::<Vec<_>>()
            != expected
        {
            return Err(Error::Invalid("incompatible database tables".into()));
        }
    }
    let _: String = db.query_row(
        "SELECT value FROM metadata WHERE key='provenance'",
        [],
        |r| r.get(0),
    )?;
    let integrity: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    if integrity != "ok" {
        return Err(Error::Invalid("integrity check failed".into()));
    }
    Ok(
        db.query_row("SELECT value FROM metadata WHERE key='kind'", [], |r| {
            r.get(0)
        })?,
    )
}
/// Validate application relationships in addition to SQLite's physical integrity.
fn load_checked(db: &Connection, kind: &str) -> Result<Counts> {
    let counts = Counts::load(db)?;
    let empty = BTreeMap::new();
    let unigrams = counts.0.get(&Vec::new()).unwrap_or(&empty);
    if kind == "baseline" && unigrams.is_empty() {
        return Err(Error::Invalid("baseline has no unigrams".into()));
    }
    let mut stmt = db.prepare("SELECT word FROM vocabulary ORDER BY word")?;
    let vocabulary: BTreeSet<String> = stmt
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    if vocabulary != unigrams.keys().cloned().collect() {
        return Err(Error::Invalid("vocabulary does not match unigrams".into()));
    }
    for (context, words) in &counts.0 {
        if context.iter().any(|word| !unigrams.contains_key(word))
            || words
                .iter()
                .any(|(word, count)| unigrams.get(word).is_none_or(|total| count > total))
        {
            return Err(Error::Invalid(
                "n-gram references inconsistent with unigrams".into(),
            ));
        }
    }
    let mut stmt = db.prepare("SELECT hash FROM imports")?;
    for hash in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let hash = hash?;
        if hash.len() != 64
            || !hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Invalid("invalid import fingerprint".into()));
        }
    }
    Ok(counts)
}
#[derive(Serialize)]
pub struct Validation {
    /// The SQLite schema version, or the n-gram file format version.
    pub schema_version: u32,
    /// `sqlite` or `ngrams`.
    pub format: String,
    pub language: String,
    pub kind: String,
    pub vocabulary: usize,
    pub ngrams: usize,
    pub logical_sha256: String,
}
/// Whether the file starts like an n-gram file rather than a SQLite database.
fn is_ngrams_file(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut magic = [0; ngrams::MAGIC.len()];
    match fs::File::open(path)?.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == ngrams::MAGIC),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error.into()),
    }
}
/// Validates either format. Returns the embedded provenance as well.
fn validate_with_provenance(path: &Path) -> Result<(Validation, String)> {
    if is_ngrams_file(path)? {
        let ngrams = ngrams::Ngrams::open(path)?;
        return Ok((
            ngrams_validation(&ngrams)?,
            ngrams.metadata().provenance.clone(),
        ));
    }
    let mut connection = readonly(path)?;
    let db = connection.transaction()?;
    let validation = validate_connection(&db)?;
    let provenance: String = db.query_row(
        "SELECT value FROM metadata WHERE key='provenance'",
        [],
        |r| r.get(0),
    )?;
    Ok((validation, provenance))
}
fn ngrams_validation(ngrams: &ngrams::Ngrams) -> Result<Validation> {
    Ok(Validation {
        schema_version: 1,
        format: "ngrams".into(),
        language: ngrams.metadata().language.clone(),
        kind: ngrams.metadata().kind.clone(),
        vocabulary: ngrams.vocabulary_len(),
        ngrams: ngrams.ngram_count(),
        logical_sha256: ngrams.logical_sha256()?,
    })
}
pub fn validate(path: &Path) -> Result<Validation> {
    validate_with_provenance(path).map(|(validation, _)| validation)
}
fn validate_connection(db: &Connection) -> Result<Validation> {
    let kind = check(db)?;
    if kind != "baseline" && kind != "personal" {
        return Err(Error::Invalid("unknown database kind".into()));
    }
    let counts = load_checked(db, &kind)?;
    Ok(Validation {
        schema_version: 1,
        format: "sqlite".into(),
        language: "en".into(),
        kind,
        vocabulary: counts.0.get(&Vec::new()).map_or(0, BTreeMap::len),
        ngrams: counts.0.values().map(BTreeMap::len).sum(),
        logical_sha256: logical_sha256(&counts)?,
    })
}
/// One JSON line per n-gram in map order, hashed.
fn logical_sha256(counts: &Counts) -> Result<String> {
    let mut logical = String::new();
    for (context, words) in &counts.0 {
        for (word, count) in words {
            logical.push_str(&serde_json::to_string(&(context, word, count))?);
            logical.push('\n');
        }
    }
    Ok(digest(logical.as_bytes()))
}
/// Creates a new baseline without replacing an existing file.
pub fn build(path: &Path, text: &str, provenance: &str) -> Result<Validation> {
    build_expected(path, text, provenance, None)
}
fn build_expected(
    path: &Path,
    text: &str,
    provenance: &str,
    expected_logical_sha256: Option<&str>,
) -> Result<Validation> {
    let mut counts = Counts::default();
    counts.add_text(text);
    if counts.0.is_empty() {
        return Err(Error::Invalid("training text contains no words".into()));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    let mut db = Connection::open(temp.path())?;
    let tx = db.transaction()?;
    initialize(&tx, "baseline", provenance)?;
    counts.write(&tx)?;
    tx.commit()?;
    db.close().map_err(|(_, e)| Error::Database(e))?;
    let validation = validate(temp.path())?;
    if expected_logical_sha256.is_some_and(|expected| expected != validation.logical_sha256) {
        return Err(Error::Invalid(
            "production model count fingerprint mismatch".into(),
        ));
    }
    temp.as_file().sync_all()?;
    temp.persist_noclobber(path)
        .map_err(|e| Error::Io(e.error))?;
    Ok(validation)
}
/// Writes verified n-gram file bytes next to `path`, then moves them into
/// place without replacing an existing file.
fn persist_ngrams(
    path: &Path,
    bytes: &[u8],
    expected_logical_sha256: Option<&str>,
) -> Result<Validation> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    let validation = ngrams_validation(&ngrams::Ngrams::open(temp.path())?)?;
    if expected_logical_sha256.is_some_and(|expected| expected != validation.logical_sha256) {
        return Err(Error::Invalid(
            "converted counts differ from the source database".into(),
        ));
    }
    temp.persist_noclobber(path)
        .map_err(|e| Error::Io(e.error))?;
    Ok(validation)
}
/// Creates a new baseline as an n-gram file without replacing an existing file.
pub fn build_ngrams(path: &Path, text: &str, provenance: &str) -> Result<Validation> {
    let mut counts = Counts::default();
    counts.add_text(text);
    if counts.0.is_empty() {
        return Err(Error::Invalid("training text contains no words".into()));
    }
    let metadata = ngrams::Metadata {
        language: "en".into(),
        kind: "baseline".into(),
        provenance: provenance.into(),
    };
    persist_ngrams(path, &ngrams::Ngrams::encode(&counts, &metadata)?, None)
}
/// Converts a SQLite baseline to an n-gram file with the same counts and
/// provenance. The output's logical fingerprint must equal the input's.
pub fn convert(input: &Path, output: &Path) -> Result<Validation> {
    let (counts, metadata) = load_sqlite_baseline(input)?;
    let logical_sha256 = logical_sha256(&counts)?;
    persist_ngrams(
        output,
        &ngrams::Ngrams::encode(&counts, &metadata)?,
        Some(&logical_sha256),
    )
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Suggestion {
    pub word: String,
    pub score: f64,
}
/// Two whole words, the first completing the prefix.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Phrase {
    pub words: Vec<String>,
    pub score: f64,
}
/// First words considered for phrases before ranking by joint probability.
const PHRASE_FIRST_WORDS: usize = 8;
/// Baseline and usable learned counts after one context, their weighted total,
/// and the interpolation weight.
struct Part<'a> {
    base: Option<ngrams::Table<'a>>,
    learned: Option<&'a BTreeMap<String, i64>>,
    total: f64,
    weight: f64,
}
struct Distributions<'a> {
    parts: Vec<Part<'a>>,
    weight_sum: f64,
    personal: f64,
}
impl Distributions<'_> {
    /// `id` is the word's baseline id, or `None` for a word the baseline lacks.
    fn score(&self, word: &str, id: Option<u32>) -> f64 {
        self.parts
            .iter()
            .map(|part| {
                let own = part.learned.and_then(|l| l.get(word)).copied().unwrap_or(0);
                let base = match (part.base, id) {
                    (Some(table), Some(id)) => table.get(id).unwrap_or(0),
                    _ => 0,
                };
                (base as f64 + self.personal * own as f64) / part.total * part.weight
                    / self.weight_sum
            })
            .sum()
    }
}
#[derive(Clone, Copy)]
pub struct Options {
    pub limit: usize,
    pub min_chars: usize,
    pub unigram_only: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            limit: 5,
            min_chars: 2,
            unigram_only: false,
        }
    }
}
/// How many baseline occurrences one learned occurrence counts as while little
/// has been learned. Learned text is tiny beside the baseline, so a low weight
/// leaves personal words almost never suggested; see docs/personal-learning.md.
const PERSONAL_WEIGHT: f64 = 500.;
/// Learned words at which the weight has halved. However much is learned,
/// learned counts approach a bounded share of the baseline rather than
/// outgrowing it.
const PERSONAL_SATURATION: f64 = 2000.;
/// Times a word must be learned before its learned counts are used, unless the
/// baseline has it at least `TRUSTED_BASELINE` times. A single misspelling,
/// including one the baseline itself has once or twice, is not suggested.
const NEW_WORD_REPEATS: i64 = 2;
const TRUSTED_BASELINE: u64 = 3;
pub struct Predictor {
    baseline: ngrams::Ngrams,
    /// Learned counts, as stored.
    learned: Counts,
    /// What predicting uses of `learned`, rebuilt whenever it changes.
    usable: Usable,
    personal: Option<Connection>,
    /// The personal database's `data_version` when `learned` last matched it.
    /// It changes only when another connection commits.
    seen: Option<i64>,
}

/// The learned counts whose words may count, their total per context, and
/// the weight of one learned occurrence.
#[derive(Default)]
struct Usable {
    counts: Counts,
    totals: BTreeMap<Vec<String>, f64>,
    /// All learned unigram occurrences, which set the weight.
    learned: f64,
    weight: f64,
}

fn data_version(db: &Connection) -> Result<i64> {
    Ok(db.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

type Schema = Vec<(String, String, String, Option<String>)>;
fn schema(db: &Connection) -> Result<Schema> {
    let mut stmt =
        db.prepare("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")?;
    Ok(stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<std::result::Result<_, _>>()?)
}
/// Whether the database has exactly the schema `initialize` creates. Other
/// objects, such as triggers or replacing constraints, can give a write side
/// effects that only a full reload validates.
fn standard_schema(db: &Connection) -> Result<bool> {
    static STANDARD: std::sync::OnceLock<Option<Schema>> = std::sync::OnceLock::new();
    let standard = STANDARD.get_or_init(|| {
        let db = Connection::open_in_memory().ok()?;
        initialize(&db, "personal", "schema").ok()?;
        schema(&db).ok()
    });
    Ok(standard.as_ref() == Some(&schema(db)?))
}
/// Opens a baseline in either format. A SQLite baseline is validated as
/// before, then encoded in memory, so prediction has one code path.
fn open_baseline(path: &Path) -> Result<ngrams::Ngrams> {
    if is_ngrams_file(path)? {
        return ngrams::Ngrams::open(path);
    }
    let (counts, metadata) = load_sqlite_baseline(path)?;
    ngrams::Ngrams::parse(ngrams::Ngrams::encode(&counts, &metadata)?)
}
/// A validated SQLite baseline's counts, with the metadata an n-gram file
/// of it carries.
fn load_sqlite_baseline(path: &Path) -> Result<(Counts, ngrams::Metadata)> {
    let mut connection = readonly(path)?;
    let db = connection.transaction()?;
    if check(&db)? != "baseline" {
        return Err(Error::Invalid("expected baseline database".into()));
    }
    let counts = load_checked(&db, "baseline")?;
    let provenance: String = db.query_row(
        "SELECT value FROM metadata WHERE key='provenance'",
        [],
        |r| r.get(0),
    )?;
    let metadata = ngrams::Metadata {
        language: "en".into(),
        kind: "baseline".into(),
        provenance,
    };
    Ok((counts, metadata))
}
impl Predictor {
    pub fn open(baseline: &Path, personal: Option<&Path>) -> Result<Self> {
        let counts = open_baseline(baseline)?;
        let personal = if let Some(path) = personal {
            if path.exists() && fs::canonicalize(path)? == fs::canonicalize(baseline)? {
                return Err(Error::Invalid(
                    "personal and baseline paths must differ".into(),
                ));
            }
            if !path.exists() {
                let parent = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let temp = tempfile::NamedTempFile::new_in(parent)?;
                let mut new = Connection::open(temp.path())?;
                let tx = new.transaction()?;
                initialize(&tx, "personal", "local personal learning")?;
                tx.commit()?;
                new.close().map_err(|(_, e)| Error::Database(e))?;
                temp.as_file().sync_all()?;
                if let Err(error) = temp.persist_noclobber(path)
                    && error.error.kind() != std::io::ErrorKind::AlreadyExists
                {
                    return Err(Error::Io(error.error));
                }
            }
            let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
            conn.busy_timeout(std::time::Duration::from_secs(5))?;
            Some(conn)
        } else {
            None
        };
        let mut result = Self {
            baseline: counts,
            learned: Counts::default(),
            usable: Usable::default(),
            personal,
            seen: None,
        };
        result.refresh_personal()?;
        Ok(result)
    }
    /// Refresh the in-memory snapshot after another process changes personal learning.
    /// On failure the previous prediction snapshot is retained.
    pub fn refresh_personal(&mut self) -> Result<()> {
        let mut learned = Counts::default();
        if let Some(db) = &mut self.personal {
            let tx = db.transaction()?;
            if check(&tx)? != "personal" {
                return Err(Error::Invalid("expected personal database".into()));
            }
            learned = load_checked(&tx, "personal")?;
            let version = data_version(&tx)?;
            if let Err(error) = tx.commit() {
                // The outcome is unknown, so the next write reloads.
                self.seen = None;
                return Err(error.into());
            }
            self.seen = Some(version);
        }
        self.learned = learned;
        self.rebuild();
        Ok(())
    }
    /// Rebuilds what predicting uses after `learned` changes. Summed as
    /// floats: stored counts are bounded one by one, not in total.
    fn rebuild(&mut self) {
        let mut usable = Usable::default();
        for (context, words) in &self.learned.0 {
            let kept: BTreeMap<String, i64> = words
                .iter()
                .filter(|(w, _)| self.usable_word(w))
                .map(|(w, c)| (w.clone(), *c))
                .collect();
            if !kept.is_empty() {
                usable
                    .totals
                    .insert(context.clone(), kept.values().map(|c| *c as f64).sum());
                usable.counts.0.insert(context.clone(), kept);
            }
        }
        self.usable = usable;
        self.usable.learned = self
            .learned
            .0
            .get(&Vec::new())
            .map_or(0., |w| w.values().map(|c| *c as f64).sum());
        self.reweigh();
    }
    /// Updates what prediction uses after `delta` was added to `learned`:
    /// its usable counts are added, and a word that has just become usable
    /// gets all its learned counts. Counts are whole numbers, so while totals
    /// stay below 2^53, which learning cannot pass, adding them
    /// in place gives the same totals as rebuilding.
    fn update(&mut self, delta: &Counts) {
        let empty = BTreeMap::new();
        let added = delta.0.get(&Vec::new()).unwrap_or(&empty);
        let learned = self.learned.0.get(&Vec::new()).unwrap_or(&empty);
        let newly: BTreeSet<&String> = added
            .iter()
            .filter(|(word, count)| {
                let now = learned.get(*word).copied().unwrap_or(0);
                !self.trusted(word) && now >= NEW_WORD_REPEATS && now - *count < NEW_WORD_REPEATS
            })
            .map(|(word, _)| word)
            .collect();
        let mut changes: Vec<(Vec<String>, String, i64)> = Vec::new();
        if !newly.is_empty() {
            for (context, words) in &self.learned.0 {
                for (word, count) in words {
                    if newly.contains(word) {
                        changes.push((context.clone(), word.clone(), *count));
                    }
                }
            }
        }
        for (context, words) in &delta.0 {
            for (word, count) in words {
                if !newly.contains(word) && self.usable_word(word) {
                    changes.push((context.clone(), word.clone(), *count));
                }
            }
        }
        for (context, word, count) in changes {
            *self.usable.totals.entry(context.clone()).or_default() += count as f64;
            *self
                .usable
                .counts
                .0
                .entry(context)
                .or_default()
                .entry(word)
                .or_default() += count;
        }
        self.usable.learned += added.values().map(|c| *c as f64).sum::<f64>();
        self.reweigh();
    }
    fn trusted(&self, word: &str) -> bool {
        self.baseline
            .id(word)
            .is_some_and(|id| self.baseline.unigram(id) >= TRUSTED_BASELINE)
    }
    /// Whether the baseline vocabulary has `word`.
    pub(crate) fn baseline_has(&self, word: &str) -> bool {
        self.baseline.id(word).is_some()
    }
    /// Whether learned counts of `word` are used.
    fn usable_word(&self, word: &str) -> bool {
        self.trusted(word)
            || self
                .learned
                .0
                .get(&Vec::new())
                .and_then(|w| w.get(word))
                .is_some_and(|c| *c >= NEW_WORD_REPEATS)
    }
    fn reweigh(&mut self) {
        self.usable.weight = PERSONAL_WEIGHT / (1. + self.usable.learned / PERSONAL_SATURATION);
    }
    /// The current sentence's words before the word in progress.
    fn context(before: &str) -> Vec<String> {
        let tail = before
            .rsplit(['.', '!', '?', '\n', '\r'])
            .next()
            .unwrap_or("");
        sentences(tail).pop().unwrap_or_default()
    }
    /// Interpolated trigram/bigram/unigram distributions of the word after `context`.
    fn distributions<'a>(&'a self, context: &[String], unigram_only: bool) -> Distributions<'a> {
        let mut parts = Vec::new();
        for (n, weight) in [(0, 0.1), (1, 0.3), (2, 0.6)] {
            if n > context.len() || (unigram_only && n != 0) {
                continue;
            }
            let key = &context[context.len() - n..];
            let base = self.baseline.table_for_words(key);
            let learned = self.usable.counts.0.get(key);
            if base.is_none() && learned.is_none() {
                continue;
            }
            let mut total: f64 = base.map_or(0., |table| table.total());
            if learned.is_some() {
                total += self.usable.weight * self.usable.totals[key];
            }
            parts.push(Part {
                base,
                learned,
                total,
                weight,
            });
        }
        Distributions {
            weight_sum: parts.iter().map(|part| part.weight).sum(),
            parts,
            personal: self.usable.weight,
        }
    }
    pub fn predict(&self, before: &str, prefix: &str, options: Options) -> Vec<Suggestion> {
        let prefix = normalize(prefix);
        if prefix.graphemes(true).count() < options.min_chars || options.limit == 0 {
            return vec![];
        }
        let distributions = self.distributions(&Self::context(before), options.unigram_only);
        // Baseline words, then learned words the baseline lacks, starting with
        // the prefix.
        let mut result: Vec<Suggestion> = self
            .baseline
            .prefix_range(&prefix)
            .map(|id| Suggestion {
                word: self.baseline.word(id).to_owned(),
                score: distributions.score(self.baseline.word(id), Some(id)),
            })
            .collect();
        result.extend(
            self.usable
                .counts
                .0
                .get(&Vec::new())
                .into_iter()
                .flat_map(|v| v.range(prefix.clone()..))
                .take_while(|(w, _)| w.starts_with(&prefix))
                .filter(|(w, _)| !self.baseline_has(w))
                .map(|(word, _)| Suggestion {
                    word: word.clone(),
                    score: distributions.score(word, None),
                }),
        );
        result.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.word.cmp(&b.word))
        });
        result.truncate(options.limit);
        result
    }
    /// Two-word phrases: a first word chosen by the `predict` rules, then a word
    /// observed after it, in the baseline or usable learned counts. A phrase
    /// scores the product of both words' interpolated probabilities. First words
    /// never observed with a following word yield no phrase, and `unigram_only`
    /// yields none, since the second word needs the first as context.
    pub fn predict_phrases(&self, before: &str, prefix: &str, options: Options) -> Vec<Phrase> {
        if options.unigram_only || options.limit == 0 {
            return vec![];
        }
        let context = Self::context(before);
        let mut result = Vec::new();
        let firsts = self.predict(
            before,
            prefix,
            Options {
                limit: PHRASE_FIRST_WORDS,
                ..options
            },
        );
        for first in firsts.into_iter().filter(|s| s.score > 0.) {
            let key = std::slice::from_ref(&first.word);
            // Followers with their baseline ids, in word order.
            let mut followers: BTreeMap<&str, Option<u32>> = BTreeMap::new();
            if let Some(table) = self.baseline.table_for_words(key) {
                followers.extend(table.ids().map(|id| (self.baseline.word(id), Some(id))));
            }
            for second in self
                .usable
                .counts
                .0
                .get(key)
                .into_iter()
                .flat_map(|w| w.keys())
            {
                followers
                    .entry(second)
                    .or_insert_with(|| self.baseline.id(second));
            }
            let mut extended = context.clone();
            extended.push(first.word.clone());
            let distributions = self.distributions(&extended, false);
            for (second, id) in followers {
                result.push(Phrase {
                    words: vec![first.word.clone(), second.to_owned()],
                    score: first.score * distributions.score(second, id),
                });
            }
        }
        result.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.words.cmp(&b.words))
        });
        result.truncate(options.limit);
        result
    }
    fn learn_inner(&mut self, text: &str, hash: Option<&str>) -> Result<bool> {
        let mut delta = Counts::default();
        delta.add_text(text);
        let db = self
            .personal
            .as_mut()
            .ok_or_else(|| Error::Invalid("personal database required".into()))?;
        // Serialize writers before reading shared state; validate and prepare the
        // new memory snapshot inside the transaction. No fallible work follows commit.
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if check(&tx)? != "personal" {
            return Err(Error::Invalid("expected personal database".into()));
        }
        let imported = if let Some(hash) = hash {
            tx.execute("INSERT OR IGNORE INTO imports VALUES (?1)", [hash])? != 0
        } else {
            true
        };
        if imported {
            delta.write(&tx)?;
        }
        // While no other connection has written since the snapshot, adding the
        // new counts to it equals reloading, without copying the baseline.
        // Otherwise reload and validate everything, including other writes.
        // A connection's own commits leave its data_version unchanged.
        let version = data_version(&tx)?;
        let current = self.seen == Some(version) && standard_schema(&tx)?;
        let reloaded = if current {
            None
        } else {
            Some(load_checked(&tx, "personal")?)
        };
        if let Err(error) = tx.commit() {
            // The outcome is unknown, so the next write reloads.
            self.seen = None;
            return Err(error.into());
        }
        self.seen = Some(version);
        match reloaded {
            Some(learned) => {
                self.learned = learned;
                self.rebuild();
            }
            None if imported => {
                self.learned.merge(&delta, 1);
                self.update(&delta);
            }
            None => {}
        }
        Ok(imported)
    }
    pub fn learn(&mut self, completed_text: &str) -> Result<()> {
        self.learn_inner(completed_text, None).map(|_| ())
    }
    pub fn import(&mut self, path: &Path) -> Result<bool> {
        let text = fs::read_to_string(path)?;
        self.learn_inner(&text, Some(&digest(text.as_bytes())))
    }
    pub fn reset_personal(&mut self) -> Result<()> {
        let db = self
            .personal
            .as_mut()
            .ok_or_else(|| Error::Invalid("personal database required".into()))?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if check(&tx)? != "personal" {
            return Err(Error::Invalid("expected personal database".into()));
        }
        tx.execute_batch("DELETE FROM counts; DELETE FROM vocabulary; DELETE FROM imports;")?;
        // Fail before commit if an incompatible trigger prevents a complete reset.
        if !load_checked(&tx, "personal")?.0.is_empty() {
            return Err(Error::Invalid("personal reset did not clear counts".into()));
        }
        let imports: i64 = tx.query_row("SELECT count(*) FROM imports", [], |r| r.get(0))?;
        if imports != 0 {
            return Err(Error::Invalid(
                "personal reset did not clear imports".into(),
            ));
        }
        let version = data_version(&tx)?;
        if let Err(error) = tx.commit() {
            // The outcome is unknown, so the next write reloads.
            self.seen = None;
            return Err(error.into());
        }
        self.seen = Some(version);
        self.learned = Counts::default();
        self.rebuild();
        Ok(())
    }
    /// The baseline's bytes in memory, plus the approximate bytes of learned
    /// UTF-8 keys and counts, excluding allocator/tree overhead.
    pub fn model_payload_bytes(&self) -> usize {
        self.baseline.bytes_len()
            + self
                .learned
                .0
                .iter()
                .map(|(ctx, words)| {
                    ctx.iter().map(String::len).sum::<usize>()
                        + words.keys().map(|w| w.len() + 8).sum::<usize>()
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run with SWITCHIFY_BASELINE pointing at a SQLite baseline. Prints how
    /// long each part of opening it takes, in milliseconds.
    #[test]
    #[ignore = "requires explicit SWITCHIFY_BASELINE"]
    fn profile_sqlite_baseline_open() {
        let path = std::path::PathBuf::from(std::env::var_os("SWITCHIFY_BASELINE").unwrap());
        let ms = |started: std::time::Instant| started.elapsed().as_secs_f64() * 1000.0;
        let mut phases = Vec::new();
        let started = std::time::Instant::now();
        let mut connection = readonly(&path).unwrap();
        let db = connection.transaction().unwrap();
        phases.push(("open_and_begin", ms(started)));
        let started = std::time::Instant::now();
        let integrity: String = db
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
        phases.push(("integrity_check", ms(started)));
        let started = std::time::Instant::now();
        assert_eq!(check(&db).unwrap(), "baseline");
        phases.push(("schema_check_including_integrity", ms(started)));
        let started = std::time::Instant::now();
        let counts = Counts::load(&db).unwrap();
        phases.push(("row_load_and_validation", ms(started)));
        let started = std::time::Instant::now();
        drop(counts);
        phases.push(("drop_counts", ms(started)));
        let started = std::time::Instant::now();
        let counts = load_checked(&db, "baseline").unwrap();
        phases.push(("load_checked_total", ms(started)));
        let started = std::time::Instant::now();
        let predictor = Predictor::open(&path, None).unwrap();
        phases.push(("predictor_open_total", ms(started)));
        println!(
            "{}",
            serde_json::json!({
                "phases_ms": phases.iter().map(|(k, v)| (k.to_string(), v)).collect::<BTreeMap<_, _>>(),
                "contexts": counts.0.len(),
                "ngrams": counts.0.values().map(BTreeMap::len).sum::<usize>(),
                "payload_bytes": predictor.model_payload_bytes(),
            })
        );
    }

    /// Adds a learned count only memory has, which a reload from the
    /// database removes and adding counts in place keeps.
    fn mark(p: &mut Predictor, word: &str) {
        p.learned
            .0
            .entry(vec![])
            .or_default()
            .insert(word.into(), 1);
    }
    fn reloaded(p: &Predictor, word: &str) -> bool {
        !p.learned.0[&Vec::new()].contains_key(word)
    }

    #[test]
    fn updating_in_place_matches_rebuilding() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.sqlite");
        let personal = dir.path().join("personal.sqlite");
        // "like" and "tea" are trusted: the baseline has them three times.
        build(
            &base,
            "hello water. i like tea. we like tea. you like tea.",
            "fixture",
        )
        .unwrap();
        let imported = dir.path().join("imported.txt");
        fs::write(&imported, "zed likes tea. Aoife likes tea. Aoife said zed.").unwrap();
        let mut p = Predictor::open(&base, Some(&personal)).unwrap();
        let check = |p: &mut Predictor, label: &str| {
            let (counts, totals, weight) = (
                p.usable.counts.0.clone(),
                p.usable.totals.clone(),
                p.usable.weight,
            );
            p.rebuild();
            assert_eq!(counts, p.usable.counts.0, "{label}");
            assert_eq!(totals, p.usable.totals, "{label}");
            assert_eq!(weight.to_bits(), p.usable.weight.to_bits(), "{label}");
        };
        assert!(p.import(&imported).unwrap());
        check(&mut p, "import");
        assert!(!p.import(&imported).unwrap());
        check(&mut p, "duplicate import");
        for text in [
            "Siobhan likes zed water.",
            "zed Siobhan likes tea.",
            "hello zed. Siobhan said hello.",
            "",
            "we like Siobhan and zed",
            "you like tea.",
        ] {
            p.learn(text).unwrap();
            check(&mut p, text);
            if text == "we like Siobhan and zed" {
                // "like" is trusted, so its first learned use counts.
                assert!(p.usable.counts.0[&vec!["we".to_owned()]].contains_key("like"));
            }
        }
        assert!(p.usable.counts.0[&Vec::new()].contains_key("siobhan"));
    }

    #[test]
    fn the_learned_weight_halves_as_much_is_learned() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.sqlite");
        let personal = dir.path().join("personal.sqlite");
        build(&base, "hello water", "fixture").unwrap();
        let mut p = Predictor::open(&base, Some(&personal)).unwrap();
        assert_eq!(p.usable.weight, PERSONAL_WEIGHT);
        p.learned
            .0
            .entry(vec![])
            .or_default()
            .insert("hello".into(), PERSONAL_SATURATION as i64);
        p.rebuild();
        assert_eq!(p.usable.weight, PERSONAL_WEIGHT / 2.);
    }

    #[test]
    fn own_learning_reloads_only_after_other_writes_or_with_triggers() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.sqlite");
        let personal = dir.path().join("personal.sqlite");
        build(&base, "hello water", "fixture").unwrap();
        let mut p = Predictor::open(&base, Some(&personal)).unwrap();
        mark(&mut p, "first");
        p.learn("hello walnut").unwrap();
        p.learn("").unwrap();
        assert!(!reloaded(&p, "first"), "own learning adds counts in place");
        let mut other = Predictor::open(&base, Some(&personal)).unwrap();
        other.learn("hello walrus").unwrap();
        p.learn("hello wallaby").unwrap();
        assert!(reloaded(&p, "first"), "another writer forces a reload");
        let db = Connection::open(&personal).unwrap();
        db.execute_batch("CREATE TRIGGER t AFTER INSERT ON counts BEGIN SELECT 1; END;")
            .unwrap();
        p.learn("hello").unwrap();
        mark(&mut p, "second");
        p.learn("hello wombat").unwrap();
        assert!(reloaded(&p, "second"), "a trigger forces a reload");
        db.execute_batch("DROP TRIGGER t").unwrap();
        p.learn("hello").unwrap();
        p.reset_personal().unwrap();
        mark(&mut p, "third");
        p.learn("hello walnut").unwrap();
        assert!(!reloaded(&p, "third"), "own reset keeps the fast path");
    }
}
