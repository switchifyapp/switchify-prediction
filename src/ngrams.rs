//! Compact, immutable baseline file read without SQLite: a sorted vocabulary,
//! fixed-width n-gram records and a whole-file SHA-256. Lookups index the
//! file's bytes in memory; nothing is parsed into maps.
//!
//! Layout, all integers little-endian:
//!
//! ```text
//! magic "SwitchifyNgrams\0"      16 bytes
//! format version                 u32 = 1
//! metadata length, metadata      u32, JSON {language, kind, provenance}
//! word count N                   u32
//! word offsets                   u32 × (N + 1), into the word bytes
//! word bytes                     UTF-8, words sorted bytewise
//! unigram counts                 u64 × N, by word id
//! bigram starts                  u32 × (N + 1), by context word id
//! bigram word ids, counts        u32 × B, then u64 × B, ids ascending per context
//! trigram context count T        u32
//! trigram contexts               (u32, u32) × T, ascending
//! trigram starts                 u32 × (T + 1)
//! trigram word ids, counts       u32 × R, then u64 × R, ids ascending per context
//! SHA-256 of everything above    32 bytes
//! ```
use crate::{Counts, Error, MAX_COUNT, Result, digest, valid_word};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range, path::Path};

pub const MAGIC: &[u8; 16] = b"SwitchifyNgrams\0";
const VERSION: u32 = 1;
const HASH_BYTES: usize = 32;
/// Word ids and counts after one context.
type Records = Vec<(u32, u64)>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub language: String,
    pub kind: String,
    pub provenance: String,
}

/// A parsed, verified n-gram file. Offsets are byte positions in `bytes`;
/// every index reachable through the public methods was checked when parsing.
pub struct Ngrams {
    bytes: Vec<u8>,
    metadata: Metadata,
    n_words: usize,
    word_offsets: usize,
    words: usize,
    unigram_counts: usize,
    bigram_starts: usize,
    bigram_ids: usize,
    bigram_counts: usize,
    n_trigram_contexts: usize,
    trigram_contexts: usize,
    trigram_starts: usize,
    trigram_ids: usize,
    trigram_counts: usize,
    /// Context totals as prediction sums them, in word order.
    unigram_total: f64,
    bigram_totals: Vec<f64>,
    trigram_totals: Vec<f64>,
}

/// The words observed after one context, with their counts.
#[derive(Clone, Copy)]
pub struct Table<'a> {
    ngrams: &'a Ngrams,
    /// `None` for the unigram table, where the word id is the index.
    sorted: Option<(usize, usize, usize)>,
    total: f64,
}
impl Table<'_> {
    pub fn get(&self, id: u32) -> Option<u64> {
        match self.sorted {
            None => Some(self.ngrams.u64_at(self.ngrams.unigram_counts, id as usize)),
            Some((ids, counts, len)) => {
                let (mut low, mut high) = (0, len);
                while low < high {
                    let mid = low + (high - low) / 2;
                    match self.ngrams.u32_at(ids, mid).cmp(&id) {
                        std::cmp::Ordering::Less => low = mid + 1,
                        std::cmp::Ordering::Greater => high = mid,
                        std::cmp::Ordering::Equal => {
                            return Some(self.ngrams.u64_at(counts, mid));
                        }
                    }
                }
                None
            }
        }
    }
    /// The float sum of the counts, in word order.
    pub fn total(&self) -> f64 {
        self.total
    }
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        let (ids, len) = match self.sorted {
            None => (None, self.ngrams.n_words),
            Some((ids, _, len)) => (Some(ids), len),
        };
        (0..len).map(move |i| match ids {
            None => i as u32,
            Some(ids) => self.ngrams.u32_at(ids, i),
        })
    }
}

fn invalid(what: &str) -> Error {
    Error::Invalid(format!("invalid n-gram file: {what}"))
}

/// Bounds-checked sequential reader used only while parsing.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    end: usize,
}
impl Cursor<'_> {
    fn take(&mut self, len: usize) -> Result<usize> {
        let start = self.pos;
        self.pos = start
            .checked_add(len)
            .filter(|&end| end <= self.end)
            .ok_or_else(|| invalid("truncated"))?;
        Ok(start)
    }
    fn array(&mut self, count: usize, width: usize) -> Result<usize> {
        let len = count
            .checked_mul(width)
            .ok_or_else(|| invalid("truncated"))?;
        self.take(len)
    }
    fn u32(&mut self) -> Result<u32> {
        let at = self.take(4)?;
        Ok(u32::from_le_bytes(
            self.bytes[at..at + 4].try_into().unwrap(),
        ))
    }
}

impl Ngrams {
    fn u32_at(&self, offset: usize, index: usize) -> u32 {
        let at = offset + index * 4;
        u32::from_le_bytes(self.bytes[at..at + 4].try_into().unwrap())
    }
    fn u64_at(&self, offset: usize, index: usize) -> u64 {
        let at = offset + index * 8;
        u64::from_le_bytes(self.bytes[at..at + 8].try_into().unwrap())
    }

    pub fn open(path: &Path) -> Result<Self> {
        Self::parse(std::fs::read(path)?)
    }

    /// Parses and fully validates a file's bytes. Every structural rule the
    /// SQLite baseline validation enforced is checked here, plus the file hash.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let hashed = bytes
            .len()
            .checked_sub(HASH_BYTES)
            .filter(|_| bytes.starts_with(MAGIC))
            .ok_or_else(|| invalid("not an n-gram file"))?;
        if Sha256::digest(&bytes[..hashed]).as_slice() != &bytes[hashed..] {
            return Err(invalid("file hash mismatch"));
        }
        let mut cursor = Cursor {
            bytes: &bytes,
            pos: MAGIC.len(),
            end: hashed,
        };
        if cursor.u32()? != VERSION {
            return Err(invalid("unsupported format version"));
        }
        let metadata_len = cursor.u32()? as usize;
        let metadata_at = cursor.take(metadata_len)?;
        let metadata: Metadata =
            serde_json::from_slice(&bytes[metadata_at..metadata_at + metadata_len])
                .map_err(|_| invalid("metadata"))?;
        if metadata.language != "en" {
            return Err(Error::Invalid("unsupported language".into()));
        }
        if metadata.kind != "baseline" {
            return Err(Error::Invalid("unknown database kind".into()));
        }
        let n_words = cursor.u32()? as usize;
        if n_words == 0 {
            return Err(Error::Invalid("baseline has no unigrams".into()));
        }
        let word_offsets = cursor.array(n_words + 1, 4)?;
        let offset_at = |i: usize| {
            u32::from_le_bytes(bytes[word_offsets + i * 4..][..4].try_into().unwrap()) as usize
        };
        let words = cursor.take(offset_at(n_words))?;
        let unigram_counts = cursor.array(n_words, 8)?;
        let bigram_starts = cursor.array(n_words + 1, 4)?;
        let start_at = |offset: usize, i: usize| {
            u32::from_le_bytes(bytes[offset + i * 4..][..4].try_into().unwrap()) as usize
        };
        let n_bigrams = start_at(bigram_starts, n_words);
        let bigram_ids = cursor.array(n_bigrams, 4)?;
        let bigram_counts = cursor.array(n_bigrams, 8)?;
        let n_trigram_contexts = cursor.u32()? as usize;
        let trigram_contexts = cursor.array(n_trigram_contexts, 8)?;
        let trigram_starts = cursor.array(n_trigram_contexts + 1, 4)?;
        let n_trigrams = start_at(trigram_starts, n_trigram_contexts);
        let trigram_ids = cursor.array(n_trigrams, 4)?;
        let trigram_counts = cursor.array(n_trigrams, 8)?;
        if cursor.pos != hashed {
            return Err(invalid("unexpected trailing bytes"));
        }
        let mut result = Self {
            bytes,
            metadata,
            n_words,
            word_offsets,
            words,
            unigram_counts,
            bigram_starts,
            bigram_ids,
            bigram_counts,
            n_trigram_contexts,
            trigram_contexts,
            trigram_starts,
            trigram_ids,
            trigram_counts,
            unigram_total: 0.,
            bigram_totals: Vec::new(),
            trigram_totals: Vec::new(),
        };
        let (unigram_total, bigram_totals, trigram_totals) = result.check()?;
        result.unigram_total = unigram_total;
        result.bigram_totals = bigram_totals;
        result.trigram_totals = trigram_totals;
        Ok(result)
    }

    /// Validates words, order and counts, and computes the context totals:
    /// unigrams, then each bigram context, then each trigram context.
    fn check(&self) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        if self.u32_at(self.word_offsets, 0) != 0 {
            return Err(invalid("word offsets"));
        }
        let mut previous: Option<&str> = None;
        for id in 0..self.n_words {
            let (start, end) = (
                self.u32_at(self.word_offsets, id) as usize,
                self.u32_at(self.word_offsets, id + 1) as usize,
            );
            if end < start || end > self.u32_at(self.word_offsets, self.n_words) as usize {
                return Err(invalid("word offsets"));
            }
            let word = std::str::from_utf8(&self.bytes[self.words + start..self.words + end])
                .map_err(|_| invalid("word encoding"))?;
            if !valid_word(word) || previous.is_some_and(|p| p >= word) {
                return Err(invalid("vocabulary order or word"));
            }
            previous = Some(word);
        }
        let mut unigram_total = 0.;
        for id in 0..self.n_words {
            let count = self.u64_at(self.unigram_counts, id);
            if count == 0 || count > MAX_COUNT {
                return Err(invalid("unigram count"));
            }
            unigram_total += count as f64;
        }
        let check_records = |ids: usize, counts: usize, range: Range<usize>| -> Result<f64> {
            let mut total = 0.;
            let mut previous = None;
            for i in range {
                let (id, count) = (self.u32_at(ids, i), self.u64_at(counts, i));
                if (id as usize) >= self.n_words
                    || previous.is_some_and(|p| p >= id)
                    || count == 0
                    || count > self.u64_at(self.unigram_counts, id as usize)
                {
                    return Err(invalid("n-gram references inconsistent with unigrams"));
                }
                previous = Some(id);
                total += count as f64;
            }
            Ok(total)
        };
        let starts = |offset: usize, i: usize| self.u32_at(offset, i) as usize;
        if starts(self.bigram_starts, 0) != 0 {
            return Err(invalid("bigram starts"));
        }
        let mut bigram_totals = Vec::with_capacity(self.n_words);
        for context in 0..self.n_words {
            let range =
                starts(self.bigram_starts, context)..starts(self.bigram_starts, context + 1);
            if range.end < range.start {
                return Err(invalid("bigram starts"));
            }
            bigram_totals.push(check_records(self.bigram_ids, self.bigram_counts, range)?);
        }
        if starts(self.trigram_starts, 0) != 0 {
            return Err(invalid("trigram starts"));
        }
        let mut trigram_totals = Vec::with_capacity(self.n_trigram_contexts);
        let mut previous = None;
        for context in 0..self.n_trigram_contexts {
            let pair = (
                self.u32_at(self.trigram_contexts, context * 2),
                self.u32_at(self.trigram_contexts, context * 2 + 1),
            );
            if (pair.0 as usize) >= self.n_words
                || (pair.1 as usize) >= self.n_words
                || previous.is_some_and(|p| p >= pair)
            {
                return Err(invalid("trigram context order"));
            }
            previous = Some(pair);
            let range =
                starts(self.trigram_starts, context)..starts(self.trigram_starts, context + 1);
            if range.end <= range.start {
                return Err(invalid("trigram starts"));
            }
            trigram_totals.push(check_records(self.trigram_ids, self.trigram_counts, range)?);
        }
        Ok((unigram_total, bigram_totals, trigram_totals))
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn vocabulary_len(&self) -> usize {
        self.n_words
    }
    pub fn ngram_count(&self) -> usize {
        self.n_words
            + self.u32_at(self.bigram_starts, self.n_words) as usize
            + self.u32_at(self.trigram_starts, self.n_trigram_contexts) as usize
    }
    /// The file's size, which is also its memory footprint.
    pub fn bytes_len(&self) -> usize {
        self.bytes.len()
    }
    /// SHA-256 of the whole file, as written in its trailer.
    pub fn file_sha256(&self) -> String {
        digest(&self.bytes)
    }

    pub fn word(&self, id: u32) -> &str {
        let (start, end) = (
            self.u32_at(self.word_offsets, id as usize) as usize,
            self.u32_at(self.word_offsets, id as usize + 1) as usize,
        );
        std::str::from_utf8(&self.bytes[self.words + start..self.words + end])
            .expect("validated when parsed")
    }
    /// The first id whose word is not below `predicate`'s boundary.
    fn partition_point(&self, mut predicate: impl FnMut(&str) -> bool) -> u32 {
        let (mut low, mut high) = (0, self.n_words);
        while low < high {
            let mid = low + (high - low) / 2;
            if predicate(self.word(mid as u32)) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low as u32
    }
    pub fn id(&self, word: &str) -> Option<u32> {
        let id = self.partition_point(|w| w < word);
        ((id as usize) < self.n_words && self.word(id) == word).then_some(id)
    }
    /// Ids of the words starting with `prefix`, in word order.
    pub fn prefix_range(&self, prefix: &str) -> Range<u32> {
        let start = self.partition_point(|w| w < prefix);
        let end = self.partition_point(|w| w < prefix || w.starts_with(prefix));
        start..end.max(start)
    }
    pub fn unigram(&self, id: u32) -> u64 {
        self.u64_at(self.unigram_counts, id as usize)
    }
    /// The words after `context` (up to two word ids), if any were observed.
    pub fn table(&self, context: &[u32]) -> Option<Table<'_>> {
        match *context {
            [] => Some(Table {
                ngrams: self,
                sorted: None,
                total: self.unigram_total,
            }),
            [first] => {
                let range = self.u32_at(self.bigram_starts, first as usize) as usize
                    ..self.u32_at(self.bigram_starts, first as usize + 1) as usize;
                (!range.is_empty()).then(|| Table {
                    ngrams: self,
                    sorted: Some((
                        self.bigram_ids + range.start * 4,
                        self.bigram_counts + range.start * 8,
                        range.len(),
                    )),
                    total: self.bigram_totals[first as usize],
                })
            }
            [first, second] => {
                let (mut low, mut high) = (0, self.n_trigram_contexts);
                while low < high {
                    let mid = low + (high - low) / 2;
                    let pair = (
                        self.u32_at(self.trigram_contexts, mid * 2),
                        self.u32_at(self.trigram_contexts, mid * 2 + 1),
                    );
                    match pair.cmp(&(first, second)) {
                        std::cmp::Ordering::Less => low = mid + 1,
                        std::cmp::Ordering::Greater => high = mid,
                        std::cmp::Ordering::Equal => {
                            let range = self.u32_at(self.trigram_starts, mid) as usize
                                ..self.u32_at(self.trigram_starts, mid + 1) as usize;
                            return Some(Table {
                                ngrams: self,
                                sorted: Some((
                                    self.trigram_ids + range.start * 4,
                                    self.trigram_counts + range.start * 8,
                                    range.len(),
                                )),
                                total: self.trigram_totals[mid],
                            });
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
    /// The table after a context given as words; `None` if any word is unknown.
    pub fn table_for_words(&self, context: &[String]) -> Option<Table<'_>> {
        let mut ids = [0; 2];
        if context.len() > ids.len() {
            return None;
        }
        for (slot, word) in ids.iter_mut().zip(context) {
            *slot = self.id(word)?;
        }
        self.table(&ids[..context.len()])
    }

    /// The logical fingerprint SQLite baselines use: one JSON line per n-gram,
    /// contexts in `Vec<String>` order, words in order within each context.
    pub fn logical_sha256(&self) -> Result<String> {
        let mut logical = String::new();
        let mut line = |context: &[&str], id: u32, count: u64| -> Result<()> {
            logical.push_str(&serde_json::to_string(&(
                context,
                self.word(id),
                count as i64,
            ))?);
            logical.push('\n');
            Ok(())
        };
        for id in 0..self.n_words as u32 {
            line(&[], id, self.unigram(id))?;
        }
        let mut trigram_context = 0;
        for first in 0..self.n_words as u32 {
            if let Some(table) = self.table(&[first]) {
                for id in table.ids() {
                    line(&[self.word(first)], id, table.get(id).unwrap_or(0))?;
                }
            }
            while trigram_context < self.n_trigram_contexts
                && self.u32_at(self.trigram_contexts, trigram_context * 2) == first
            {
                let second = self.u32_at(self.trigram_contexts, trigram_context * 2 + 1);
                let table = self.table(&[first, second]).expect("listed context");
                for id in table.ids() {
                    line(
                        &[self.word(first), self.word(second)],
                        id,
                        table.get(id).unwrap_or(0),
                    )?;
                }
                trigram_context += 1;
            }
        }
        Ok(digest(logical.as_bytes()))
    }

    /// Encodes counts into the file format. The counts must be consistent:
    /// every context word and every word has a unigram, and no n-gram count
    /// exceeds its word's unigram count.
    pub(crate) fn encode(counts: &Counts, metadata: &Metadata) -> Result<Vec<u8>> {
        let unigrams = counts
            .0
            .get(&Vec::new())
            .filter(|u| !u.is_empty())
            .ok_or_else(|| Error::Invalid("baseline has no unigrams".into()))?;
        let ids: BTreeMap<&str, u32> = unigrams
            .keys()
            .enumerate()
            .map(|(id, word)| (word.as_str(), id as u32))
            .collect();
        let id_of = |word: &str| -> Result<u32> {
            ids.get(word).copied().ok_or_else(|| {
                Error::Invalid("n-gram references inconsistent with unigrams".into())
            })
        };
        let count_of = |word: &str, count: i64| -> Result<u64> {
            if count <= 0 || count > MAX_COUNT as i64 || count > unigrams[word] {
                return Err(Error::Invalid(
                    "n-gram references inconsistent with unigrams".into(),
                ));
            }
            Ok(count as u64)
        };
        let mut bigrams: Vec<Vec<(u32, u64)>> = vec![Vec::new(); ids.len()];
        let mut trigrams: Vec<((u32, u32), Records)> = Vec::new();
        for (context, words) in &counts.0 {
            if context.is_empty() {
                continue;
            }
            let records = words
                .iter()
                .map(|(word, count)| Ok((id_of(word)?, count_of(word, *count)?)))
                .collect::<Result<Vec<_>>>()?;
            match context.as_slice() {
                [first] => bigrams[id_of(first)? as usize] = records,
                [first, second] => trigrams.push(((id_of(first)?, id_of(second)?), records)),
                _ => return Err(Error::Invalid("invalid n-gram".into())),
            }
        }
        trigrams.sort_unstable_by_key(|(context, _)| *context);
        let mut out = Vec::new();
        let push_u32 = |out: &mut Vec<u8>, value: usize| -> Result<()> {
            let value = u32::try_from(value).map_err(|_| invalid("too large"))?;
            out.extend_from_slice(&value.to_le_bytes());
            Ok(())
        };
        out.extend_from_slice(MAGIC);
        push_u32(&mut out, VERSION as usize)?;
        let metadata = serde_json::to_vec(metadata)?;
        push_u32(&mut out, metadata.len())?;
        out.extend_from_slice(&metadata);
        push_u32(&mut out, ids.len())?;
        let mut offset = 0;
        for word in ids.keys() {
            if !valid_word(word) {
                return Err(Error::Invalid("invalid n-gram".into()));
            }
            push_u32(&mut out, offset)?;
            offset += word.len();
        }
        push_u32(&mut out, offset)?;
        for word in ids.keys() {
            out.extend_from_slice(word.as_bytes());
        }
        for (word, count) in unigrams {
            out.extend_from_slice(&count_of(word, *count)?.to_le_bytes());
        }
        let mut start = 0;
        for records in &bigrams {
            push_u32(&mut out, start)?;
            start += records.len();
        }
        push_u32(&mut out, start)?;
        for (id, _) in bigrams.iter().flatten() {
            out.extend_from_slice(&id.to_le_bytes());
        }
        for (_, count) in bigrams.iter().flatten() {
            out.extend_from_slice(&count.to_le_bytes());
        }
        push_u32(&mut out, trigrams.len())?;
        for ((first, second), _) in &trigrams {
            out.extend_from_slice(&first.to_le_bytes());
            out.extend_from_slice(&second.to_le_bytes());
        }
        let mut start = 0;
        for (_, records) in &trigrams {
            push_u32(&mut out, start)?;
            start += records.len();
        }
        push_u32(&mut out, start)?;
        for (id, _) in trigrams.iter().flat_map(|(_, r)| r) {
            out.extend_from_slice(&id.to_le_bytes());
        }
        for (_, count) in trigrams.iter().flat_map(|(_, r)| r) {
            out.extend_from_slice(&count.to_le_bytes());
        }
        let hash = Sha256::digest(&out);
        out.extend_from_slice(&hash);
        Ok(out)
    }
}
