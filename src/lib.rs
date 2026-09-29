//! Offline English word prediction. Learning accepts non-overlapping completed segments.
use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

pub mod evaluation;

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
            let valid_word = |w: &str| {
                !w.is_empty() && normalize(w) == w && sentences(w) == vec![vec![w.to_string()]]
            };
            if context.len() > 2
                || count <= 0
                || count > i64::MAX / 10
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
    pub schema_version: u32,
    pub language: String,
    pub kind: String,
    pub vocabulary: usize,
    pub ngrams: usize,
    pub logical_sha256: String,
}
pub fn validate(path: &Path) -> Result<Validation> {
    let db = readonly(path)?;
    let kind = check(&db)?;
    if kind != "baseline" && kind != "personal" {
        return Err(Error::Invalid("unknown database kind".into()));
    }
    let counts = load_checked(&db, &kind)?;
    let mut logical = String::new();
    for (context, words) in &counts.0 {
        for (word, count) in words {
            logical.push_str(&serde_json::to_string(&(context, word, count))?);
            logical.push('\n');
        }
    }
    Ok(Validation {
        schema_version: 1,
        language: "en".into(),
        kind,
        vocabulary: counts.0.get(&Vec::new()).map_or(0, BTreeMap::len),
        ngrams: counts.0.values().map(BTreeMap::len).sum(),
        logical_sha256: digest(logical.as_bytes()),
    })
}
/// Creates a new baseline without replacing an existing file.
pub fn build(path: &Path, text: &str, provenance: &str) -> Result<Validation> {
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
    temp.as_file().sync_all()?;
    temp.persist_noclobber(path)
        .map_err(|e| Error::Io(e.error))?;
    Ok(validation)
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Suggestion {
    pub word: String,
    pub score: f64,
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
pub struct Predictor {
    baseline: Counts,
    combined: Counts,
    personal: Option<Connection>,
}
impl Predictor {
    pub fn open(baseline: &Path, personal: Option<&Path>) -> Result<Self> {
        let db = readonly(baseline)?;
        if check(&db)? != "baseline" {
            return Err(Error::Invalid("expected baseline database".into()));
        }
        let counts = load_checked(&db, "baseline")?;
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
                temp.persist_noclobber(path)
                    .map_err(|e| Error::Io(e.error))?;
            }
            let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
            if check(&conn)? != "personal" {
                return Err(Error::Invalid("expected personal database".into()));
            }
            Some(conn)
        } else {
            None
        };
        let mut result = Self {
            baseline: counts.clone(),
            combined: counts,
            personal,
        };
        result.reload()?;
        Ok(result)
    }
    fn reload(&mut self) -> Result<()> {
        let mut combined = self.baseline.clone();
        if let Some(db) = &self.personal {
            combined.merge(&load_checked(db, "personal")?, 5);
        }
        self.combined = combined;
        Ok(())
    }
    pub fn predict(&self, before: &str, prefix: &str, options: Options) -> Vec<Suggestion> {
        let prefix = normalize(prefix);
        if prefix.graphemes(true).count() < options.min_chars || options.limit == 0 {
            return vec![];
        }
        let tail = before
            .rsplit(['.', '!', '?', '\n', '\r'])
            .next()
            .unwrap_or("");
        let context = sentences(tail).pop().unwrap_or_default();
        let mut distributions = Vec::new();
        for (n, weight) in [(0, 0.1), (1, 0.3), (2, 0.6)] {
            if n > context.len() || (options.unigram_only && n != 0) {
                continue;
            }
            if let Some(words) = self.combined.0.get(&context[context.len() - n..]) {
                let total: f64 = words.values().map(|c| *c as f64).sum();
                distributions.push((words, total, weight));
            }
        }
        let weight_sum: f64 = distributions.iter().map(|(_, _, w)| w).sum();
        let mut result = Vec::new();
        if let Some(vocabulary) = self.combined.0.get(&Vec::new()) {
            for (word, _) in vocabulary.range(prefix.clone()..) {
                if !word.starts_with(&prefix) {
                    break;
                }
                let score = distributions
                    .iter()
                    .map(|(words, total, w)| {
                        words.get(word).copied().unwrap_or(0) as f64 / total * w / weight_sum
                    })
                    .sum();
                result.push(Suggestion {
                    word: word.clone(),
                    score,
                });
            }
        }
        result.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.word.cmp(&b.word))
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
        let tx = db.transaction()?;
        if let Some(hash) = hash
            && tx.execute("INSERT OR IGNORE INTO imports VALUES (?1)", [hash])? == 0
        {
            return Ok(false);
        }
        delta.write(&tx)?;
        tx.commit()?;
        self.reload()?;
        Ok(true)
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
        let tx = db.transaction()?;
        tx.execute_batch("DELETE FROM counts; DELETE FROM vocabulary; DELETE FROM imports;")?;
        tx.commit()?;
        self.reload()
    }
    /// Approximate bytes of stored UTF-8 keys and counts, excluding allocator/tree overhead.
    pub fn model_payload_bytes(&self) -> usize {
        [&self.baseline, &self.combined]
            .iter()
            .map(|counts| {
                counts
                    .0
                    .iter()
                    .map(|(ctx, words)| {
                        ctx.iter().map(String::len).sum::<usize>()
                            + words.keys().map(|w| w.len() + 8).sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .sum()
    }
}
