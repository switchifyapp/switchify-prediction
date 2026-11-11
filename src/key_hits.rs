//! Opt-in, layout-based single-key substitutions. No input capture or text edits.
use crate::{Error, Options, Predictor, Result, normalize, valid_word};
use serde::Serialize;
use std::collections::{BTreeMap, btree_map::Entry};
use unicode_segmentation::UnicodeSegmentation;

const MAX_KEYS: usize = 128;
const MAX_NEIGHBORS: usize = 8;
const MAX_PREFIX: usize = 32;

/// Character at a key's center. All coordinates, radius and spread share units.
#[derive(Clone, Copy, Debug)]
pub struct KeyPosition {
    pub key: char,
    pub x: f64,
    pub y: f64,
}

/// An intended key near the typed key, with a relative weight at most one.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyAlternative {
    pub key: String,
    pub weight: f64,
}

/// Immutable geometry-derived alternatives, normalized like prediction prefixes.
#[derive(Clone, Debug)]
pub struct KeyHitMap {
    neighbors: BTreeMap<String, Vec<KeyAlternative>>,
}

/// One substituted grapheme in the normalized typed prefix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KeySubstitution {
    pub grapheme_index: usize,
    pub typed: String,
    pub intended: String,
}

/// A candidate may replace the typed prefix, rather than simply extend it.
/// Selection and replacement belong to the caller; this API never edits text.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HitSuggestion {
    pub word: String,
    /// Language score multiplied by key_weight, not a calibrated probability.
    pub score: f64,
    pub language_score: f64,
    pub key_weight: f64,
    /// None for an exact-prefix completion.
    pub substitution: Option<KeySubstitution>,
}

impl KeyHitMap {
    /// Build from 1..=128 distinct normalized keys at distinct finite positions.
    /// Radius bounds neighbors; spread controls exp(-0.5 * (distance/spread)^2).
    /// At most eight nearest alternatives are retained per key. Weights are
    /// relative to an exact hit's weight of one, not learned error probabilities.
    pub fn new(keys: &[KeyPosition], radius: f64, spread: f64) -> Result<Self> {
        if keys.is_empty()
            || keys.len() > MAX_KEYS
            || !radius.is_finite()
            || radius <= 0.
            || !spread.is_finite()
            || spread <= 0.
        {
            return Err(Error::Invalid("invalid keyboard hit-map dimensions".into()));
        }
        let mut positions = BTreeMap::new();
        for position in keys {
            let key = normalize(&position.key.to_string());
            if key.graphemes(true).count() != 1
                || (!valid_word(&key) && key != "'")
                || !position.x.is_finite()
                || !position.y.is_finite()
                || positions.contains_key(&key)
                || positions
                    .values()
                    .any(|&(x, y)| x == position.x && y == position.y)
            {
                return Err(Error::Invalid("invalid or duplicate keyboard key".into()));
            }
            positions.insert(key, (position.x, position.y));
        }
        let mut neighbors = BTreeMap::new();
        for (typed, &(x, y)) in &positions {
            let mut alternatives: Vec<_> = positions
                .iter()
                .filter(|(key, _)| *key != typed)
                .filter_map(|(key, &(nx, ny))| {
                    let distance = (nx - x).hypot(ny - y);
                    let weight = (-0.5 * (distance / spread).powi(2)).exp();
                    (distance <= radius && weight > 0.).then(|| {
                        (
                            distance,
                            KeyAlternative {
                                key: key.clone(),
                                weight,
                            },
                        )
                    })
                })
                .collect();
            alternatives.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.key.cmp(&b.1.key)));
            neighbors.insert(
                typed.clone(),
                alternatives
                    .into_iter()
                    .take(MAX_NEIGHBORS)
                    .map(|(_, a)| a)
                    .collect(),
            );
        }
        Ok(Self { neighbors })
    }

    /// Conventional staggered QWERTY letters, in units of one key pitch.
    /// Radius 1.3 and spread 0.45 are fixed starting values, not user calibration.
    pub fn qwerty() -> Self {
        let keys: Vec<_> = [("qwertyuiop", 0.), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75)]
            .iter()
            .enumerate()
            .flat_map(|(y, &(row, offset))| {
                row.chars().enumerate().map(move |(x, key)| KeyPosition {
                    key,
                    x: x as f64 + offset,
                    y: y as f64,
                })
            })
            .collect();
        Self::new(&keys, 1.3, 0.45).expect("built-in QWERTY geometry is valid")
    }

    /// Nearest alternatives to a typed key. Unmapped keys have no alternatives.
    pub fn neighbors(&self, key: &str) -> &[KeyAlternative] {
        self.neighbors
            .get(&normalize(key))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl Predictor {
    /// Complete an exact prefix or one containing a single nearby-key mistake.
    /// Uses the existing language model, personal counts and Options unchanged.
    /// No insertion, deletion or transposition correction is attempted. Unknown
    /// keys stay exact. Prefixes over 32 graphemes use exact prediction only.
    /// Search is bounded to at most 257 prefix variants. Substitutions that
    /// merge or alter surrounding grapheme boundaries are skipped.
    pub fn predict_with_hit_map(
        &self,
        before: &str,
        prefix: &str,
        options: Options,
        hit_map: &KeyHitMap,
    ) -> Vec<HitSuggestion> {
        let prefix = normalize(prefix);
        let parts: Vec<_> = prefix.graphemes(true).collect();
        if parts.len() < options.min_chars || options.limit == 0 {
            return vec![];
        }
        let mut variants = vec![(prefix.clone(), 1., None)];
        if parts.len() <= MAX_PREFIX {
            for (index, typed) in parts.iter().enumerate() {
                for alternative in hit_map.neighbors(typed) {
                    let mut corrected = parts.clone();
                    corrected[index] = &alternative.key;
                    let candidate = normalize(&corrected.concat());
                    if !candidate.graphemes(true).eq(corrected.iter().copied()) {
                        continue;
                    }
                    variants.push((
                        candidate,
                        alternative.weight,
                        Some(KeySubstitution {
                            grapheme_index: index,
                            typed: (*typed).to_string(),
                            intended: alternative.key.clone(),
                        }),
                    ));
                }
            }
        }
        if variants.len() == 1 {
            return self
                .predict(before, &prefix, options)
                .into_iter()
                .map(|suggestion| HitSuggestion {
                    word: suggestion.word,
                    score: suggestion.score,
                    language_score: suggestion.score,
                    key_weight: 1.,
                    substitution: None,
                })
                .collect();
        }
        let mut best: BTreeMap<String, HitSuggestion> = BTreeMap::new();
        for (candidate, key_weight, substitution) in variants {
            // Rank after weighting, including ties introduced by floating-point
            // multiplication. Unicode byte-prefix ranges can overlap, so keep
            // the highest weighted result per word, preferring exact on ties.
            for suggestion in self.predict(
                before,
                &candidate,
                Options {
                    limit: usize::MAX,
                    ..options
                },
            ) {
                let hit = HitSuggestion {
                    word: suggestion.word,
                    score: suggestion.score * key_weight,
                    language_score: suggestion.score,
                    key_weight,
                    substitution: substitution.clone(),
                };
                match best.entry(hit.word.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(hit);
                    }
                    Entry::Occupied(mut entry) if hit.score > entry.get().score => {
                        entry.insert(hit);
                    }
                    Entry::Occupied(_) => {}
                }
            }
        }
        let mut result: Vec<_> = best.into_values().collect();
        result.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.word.cmp(&b.word))
        });
        result.truncate(options.limit);
        result
    }
}
