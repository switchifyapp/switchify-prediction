//! Opt-in completion from letters typed in order, such as `abt` for `about`.
//! No input capture or text edits.
use crate::{Options, Predictor, Suggestion, normalize};
use serde::Serialize;
use unicode_segmentation::UnicodeSegmentation;

/// Longer typed text uses exact-prefix prediction only.
const MAX_TYPED: usize = 32;
/// Score factor for a word that matches only by skipping letters. A fixed
/// starting value that keeps exact completions ahead of equally likely
/// abbreviation matches, not a calibrated probability.
pub const ABBREVIATION_WEIGHT: f64 = 0.5;

/// A word whose letters include the typed letters in order.
/// Selection and replacement belong to the caller; this API never edits text.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AbbreviatedSuggestion {
    pub word: String,
    /// Language score, multiplied by `ABBREVIATION_WEIGHT` when abbreviated.
    /// A relative ranking value, not a calibrated probability.
    pub score: f64,
    pub language_score: f64,
    /// False when the typed text is a prefix of the word. True when the word
    /// matches only by skipping letters, so the caller replaces the typed text.
    pub abbreviated: bool,
}

/// `None` unless `word` starts with the first typed grapheme and contains the
/// rest in order. Otherwise whether letters had to be skipped.
fn matches(word: &str, typed: &[&str]) -> Option<bool> {
    let (first, rest) = typed.split_first()?;
    let mut letters = word.graphemes(true);
    if letters.next()? != *first {
        return None;
    }
    let mut skipped = false;
    for wanted in rest {
        loop {
            if letters.next()? == *wanted {
                break;
            }
            skipped = true;
        }
    }
    Some(skipped)
}

fn exact(suggestion: Suggestion) -> AbbreviatedSuggestion {
    AbbreviatedSuggestion {
        word: suggestion.word,
        score: suggestion.score,
        language_score: suggestion.score,
        abbreviated: false,
    }
}

impl Predictor {
    /// Complete an exact prefix, or a word that starts with the first typed
    /// letter and contains the remaining typed letters in order with any
    /// letters between them. `abt` can suggest `about`, and `dont` `don't`.
    /// Uses the existing language model, personal counts and Options unchanged.
    /// Typed text over 32 graphemes uses exact prediction only.
    pub fn predict_abbreviated(
        &self,
        before: &str,
        typed: &str,
        options: Options,
    ) -> Vec<AbbreviatedSuggestion> {
        let typed = normalize(typed);
        let parts: Vec<_> = typed.graphemes(true).collect();
        if parts.len() < options.min_chars || options.limit == 0 {
            return vec![];
        }
        if parts.len() < 2 || parts.len() > MAX_TYPED {
            return self
                .predict(before, &typed, options)
                .into_iter()
                .map(exact)
                .collect();
        }
        let distributions = self.distributions(&Self::context(before), options.unigram_only);
        let suggestion = |word: &str, id: Option<u32>| {
            matches(word, &parts).map(|abbreviated| {
                let language_score = distributions.score(word, id);
                AbbreviatedSuggestion {
                    word: word.to_owned(),
                    score: if abbreviated {
                        language_score * ABBREVIATION_WEIGHT
                    } else {
                        language_score
                    },
                    language_score,
                    abbreviated,
                }
            })
        };
        // Baseline words, then learned words the baseline lacks, starting with
        // the first typed grapheme, as `predict` considers them.
        let first = parts[0];
        let mut result: Vec<_> = self
            .baseline
            .prefix_range(first)
            .filter_map(|id| suggestion(self.baseline.word(id), Some(id)))
            .collect();
        result.extend(
            self.usable
                .counts
                .0
                .get(&Vec::new())
                .into_iter()
                .flat_map(|v| v.range(first.to_owned()..))
                .take_while(|(w, _)| w.starts_with(first))
                .filter(|(w, _)| !self.baseline_has(w))
                .filter_map(|(word, _)| suggestion(word, None)),
        );
        result.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.word.cmp(&b.word))
        });
        result.truncate(options.limit);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::matches;

    fn typed(text: &str) -> Vec<&str> {
        unicode_segmentation::UnicodeSegmentation::graphemes(text, true).collect()
    }

    #[test]
    fn letters_must_start_the_word_and_stay_in_order() {
        assert_eq!(matches("about", &typed("abt")), Some(true));
        assert_eq!(matches("about", &typed("abo")), Some(false));
        assert_eq!(matches("about", &typed("about")), Some(false));
        assert_eq!(matches("don't", &typed("dont")), Some(true));
        assert_eq!(matches("about", &typed("bt")), None);
        assert_eq!(matches("about", &typed("atb")), None);
        assert_eq!(matches("about", &typed("abouts")), None);
        assert_eq!(matches("café", &typed("cé")), Some(true));
        assert_eq!(matches("cafe\u{301}", &typed("cafe")), None);
    }
}
