//! Opt-in conservative legacy backfill. No personal data or shared score scale.
use crate::{Error, Options, Predictor, Result, normalize, sentences};
use rusqlite::{Connection, OpenFlags};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Newer,
    Original,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct RankedWord {
    pub word: String,
    pub source: Source,
}

/// Frequency/ID-ranked lists reproduce rc.14 longest-context-first backoff.
/// Normalization and valid-single-word filtering deliberately follow the new API.
pub struct LegacyPredictor {
    words: Vec<String>,
    ids: HashMap<String, i64>,
    groups: HashMap<Vec<i64>, Vec<usize>>,
    prefixes: BTreeMap<String, usize>,
}
impl LegacyPredictor {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let integrity: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
        if integrity != "ok" {
            return Err(Error::Invalid("legacy integrity check failed".into()));
        }
        let mut words = Vec::new();
        let mut ids = HashMap::new();
        let mut ranks = HashMap::new();
        let mut all_ids = HashSet::new();
        let mut prefixes = BTreeMap::new();
        let mut stmt = db.prepare(
            "SELECT ID,WORD,BASE_FREQUENCY FROM WORDS ORDER BY BASE_FREQUENCY DESC,ID ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })?;
        for row in rows {
            let (id, text, frequency) = row?;
            if id <= 0 || frequency < 0 || !all_ids.insert(id) {
                return Err(Error::Invalid("invalid legacy word record".into()));
            }
            let word = normalize(&text);
            if word.chars().count() > 48 || sentences(&word) != vec![vec![word.clone()]] {
                continue;
            }
            let rank = words.len();
            ids.entry(word.clone()).or_insert(id);
            prefixes.entry(word.clone()).or_insert(rank);
            ranks.insert(id, rank);
            words.push(word);
        }
        if words.is_empty() {
            return Err(Error::Invalid("empty legacy vocabulary".into()));
        }
        let mut groups: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
        for (n, table) in [(2, "BIGRAMS"), (3, "TRIGRAMS"), (4, "QUADGRAMS")] {
            let fields = (1..=n)
                .map(|i| format!("ID{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let mut stmt = db.prepare(&format!(
                "SELECT {fields},BASE_FREQUENCY FROM {table} ORDER BY BASE_FREQUENCY DESC,ID{n} ASC"
            ))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let mut key = Vec::new();
                for i in 0..n {
                    key.push(row.get::<_, i64>(i)?);
                }
                if row.get::<_, i64>(n)? < 0 || key.iter().any(|id| !all_ids.contains(id)) {
                    return Err(Error::Invalid("invalid legacy n-gram".into()));
                }
                let target = key.pop().unwrap();
                if let Some(&rank) = ranks.get(&target) {
                    groups.entry(key).or_default().push(rank);
                }
            }
        }
        Ok(Self {
            words,
            ids,
            groups,
            prefixes,
        })
    }
    pub fn vocabulary(&self) -> impl Iterator<Item = &str> {
        self.prefixes.keys().map(String::as_str)
    }
    /// Key/list payload only, not allocator overhead or process RSS.
    pub fn model_payload_bytes(&self) -> usize {
        self.words.iter().map(String::len).sum::<usize>()
            + self
                .groups
                .iter()
                .map(|(k, v)| k.len() * 8 + v.len() * std::mem::size_of::<usize>())
                .sum::<usize>()
            + self.ids.keys().map(|s| s.len() + 8).sum::<usize>()
            + self
                .prefixes
                .keys()
                .map(|s| s.len() + std::mem::size_of::<usize>())
                .sum::<usize>()
    }
    pub fn predict(&self, before: &str, prefix: &str, options: Options) -> Vec<RankedWord> {
        let prefix = normalize(prefix);
        if options.limit == 0 || prefix.graphemes(true).count() < options.min_chars {
            return Vec::new();
        }
        let tail = before
            .rsplit(['.', '!', '?', '\n', '\r'])
            .next()
            .unwrap_or("");
        let context = sentences(tail).pop().unwrap_or_default();
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        let mut push = |rank: usize| {
            let word = &self.words[rank];
            if word.starts_with(&prefix) && seen.insert(word.clone()) {
                result.push(RankedWord {
                    word: word.clone(),
                    source: Source::Original,
                });
            }
            result.len() >= options.limit
        };
        if !options.unigram_only {
            for n in (1..=context.len().min(3)).rev() {
                let key: Option<Vec<_>> = context[context.len() - n..]
                    .iter()
                    .map(|w| self.ids.get(w).copied())
                    .collect();
                if let Some(list) = key.and_then(|k| self.groups.get(&k)) {
                    for &rank in list {
                        if push(rank) {
                            return result;
                        }
                    }
                }
            }
        }
        let mut ranks: Vec<_> = self
            .prefixes
            .range(prefix.clone()..)
            .take_while(|(w, _)| w.starts_with(&prefix))
            .map(|(_, r)| *r)
            .collect();
        ranks.sort_unstable();
        for rank in ranks {
            if push(rank) {
                break;
            }
        }
        result
    }
}

pub struct CombinedPredictor {
    newer: Predictor,
    legacy: LegacyPredictor,
}
impl CombinedPredictor {
    pub fn open(baseline: &Path, legacy: &Path) -> Result<Self> {
        Ok(Self {
            newer: Predictor::open(baseline, None)?,
            legacy: LegacyPredictor::open(legacy)?,
        })
    }
    pub fn predict(&self, before: &str, prefix: &str, options: Options) -> Vec<RankedWord> {
        combine(&self.newer, &self.legacy, before, prefix, options)
    }
}
pub fn combine(
    newer: &Predictor,
    legacy: &LegacyPredictor,
    before: &str,
    prefix: &str,
    options: Options,
) -> Vec<RankedWord> {
    let mut result: Vec<_> = newer
        .predict(before, prefix, options)
        .into_iter()
        .map(|s| RankedWord {
            word: s.word,
            source: Source::Newer,
        })
        .collect();
    if result.len() < options.limit {
        let mut seen: HashSet<_> = result.iter().map(|r| normalize(&r.word)).collect();
        // Request the full limit so overlap cannot hide eligible fallback words.
        for item in legacy.predict(before, prefix, options) {
            if seen.insert(normalize(&item.word)) {
                result.push(item);
            }
            if result.len() == options.limit {
                break;
            }
        }
    }
    result
}

/// One process per mode makes peak process memory comparable. Baseline vocabulary
/// is read separately for OOV accounting, without loading an unused predictor.
pub fn compare(
    mode: &str,
    baseline: &Path,
    legacy: &Path,
    input: &Path,
) -> Result<serde_json::Value> {
    use std::{fs, time::Instant};
    if !matches!(mode, "original" | "newer" | "combined") {
        return Err(Error::Invalid("unknown comparison mode".into()));
    }
    let text = fs::read_to_string(input)?;
    let tests: std::collections::BTreeSet<_> = sentences(&text).into_iter().collect();
    if tests.is_empty() {
        return Err(Error::Invalid("empty evaluation partition".into()));
    }
    let modern_vocabulary: HashSet<String> = {
        let db = Connection::open_with_flags(baseline, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        db.prepare("SELECT word FROM vocabulary")?
            .query_map([], |r| r.get(0))?
            .collect::<std::result::Result<_, _>>()?
    };
    let start = Instant::now();
    let newer = if mode != "original" {
        Some(Predictor::open(baseline, None)?)
    } else {
        None
    };
    let original = if mode != "newer" {
        Some(LegacyPredictor::open(legacy)?)
    } else {
        None
    };
    let cold = start.elapsed().as_secs_f64() * 1000.0;
    let vocabulary: HashSet<_> = modern_vocabulary
        .iter()
        .filter(|_| mode != "original")
        .cloned()
        .chain(
            original
                .as_ref()
                .into_iter()
                .flat_map(|p| p.vocabulary().map(str::to_owned)),
        )
        .collect();
    let opts = Options {
        limit: 5,
        min_chars: 0,
        unigram_only: false,
    };
    let predict = |before: &str, prefix: &str| -> Vec<RankedWord> {
        match (newer.as_ref(), original.as_ref()) {
            (Some(n), Some(o)) => combine(n, o, before, prefix, opts),
            (Some(n), None) => n
                .predict(before, prefix, opts)
                .into_iter()
                .map(|s| RankedWord {
                    word: s.word,
                    source: Source::Newer,
                })
                .collect(),
            (None, Some(o)) => o.predict(before, prefix, opts),
            _ => unreachable!(),
        }
    };
    predict("I need ", "he");
    let mut counts = [[0usize; 8]; 5]; // queries, top1, top5, covered, OOV queries/top1/top5, filled queries
    let mut timings = Vec::new();
    let mut queries = Vec::new();
    let mut position_violations = 0usize;
    for words in &tests {
        for (i, target) in words.iter().enumerate() {
            let before = words[..i].join(" ");
            let graphemes: Vec<_> = target.graphemes(true).collect();
            for n in 0..=4.min(graphemes.len().saturating_sub(1)) {
                let prefix = graphemes[..n].concat();
                let start = Instant::now();
                let result = predict(&before, &prefix);
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
                let rank = result.iter().position(|r| r.word == *target);
                let c = &mut counts[n];
                c[0] += 1;
                c[1] += usize::from(rank == Some(0));
                c[2] += usize::from(rank.is_some());
                c[3] += usize::from(vocabulary.contains(target));
                if !modern_vocabulary.contains(target) {
                    c[4] += 1;
                    c[5] += usize::from(rank == Some(0));
                    c[6] += usize::from(rank.is_some());
                }
                c[7] += usize::from(result.iter().any(|r| r.source == Source::Original));
                if mode == "combined" {
                    let modern = newer.as_ref().unwrap().predict(&before, &prefix, opts);
                    if !modern
                        .iter()
                        .zip(&result)
                        .all(|(a, b)| a.word == b.word && b.source == Source::Newer)
                        || result.len() < modern.len()
                    {
                        position_violations += 1;
                    }
                }
                queries.push((before.clone(), prefix));
            }
        }
    }
    while timings.len() < 1000 {
        for (before, prefix) in &queries {
            let start = Instant::now();
            predict(before, prefix);
            timings.push(start.elapsed().as_secs_f64() * 1000.0);
            if timings.len() >= 1000 {
                break;
            }
        }
    }
    timings.sort_by(f64::total_cmp);
    let percentile = |p: usize| timings[(timings.len() * p).div_ceil(100).saturating_sub(1)];
    let accuracy:Vec<_>=counts.iter().enumerate().map(|(n,c)| serde_json::json!({"prefix_chars":n,"queries":c[0],"top1":c[1] as f64/c[0].max(1) as f64,"top5":c[2] as f64/c[0].max(1) as f64,"vocabulary_coverage":c[3] as f64/c[0].max(1) as f64,"newer_oov_queries":c[4],"newer_oov_top1_hits":c[5],"newer_oov_top5_hits":c[6],"legacy_fill_queries":c[7]})).collect();
    let payload = newer.as_ref().map_or(0, Predictor::model_payload_bytes)
        + original
            .as_ref()
            .map_or(0, LegacyPredictor::model_payload_bytes);
    Ok(
        serde_json::json!({"protocol":"legacy-slot-fill-v1","mode":mode,"input_sha256":crate::digest(text.as_bytes()),"baseline_sha256":crate::digest(&fs::read(baseline)?),"legacy_sha256":crate::digest(&fs::read(legacy)?),"sentences":tests.len(),"accuracy":accuracy,"position_violations":position_violations,"warm_query_count":timings.len(),"cold_load_ms":cold,"warm_median_ms":percentile(50),"warm_p95_ms":percentile(95),"warm_max_ms":timings.last(),"warm_p95_target_met":percentile(95)<20.0,"model_payload_bytes":payload,"database_bytes":if mode=="original" {fs::metadata(legacy)?.len()} else if mode=="newer" {fs::metadata(baseline)?.len()} else {fs::metadata(legacy)?.len()+fs::metadata(baseline)?.len()}}),
    )
}
