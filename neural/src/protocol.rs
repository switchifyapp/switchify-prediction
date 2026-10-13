//! Bounded private child-process transport. No sockets and no text diagnostics.
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Read, Write};

pub const VERSION: u32 = 3;
/// Words in a generated phrase.
pub const PHRASE_WORDS: usize = 2;
pub const MAX_FRAME: usize = 65_536;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub id: u64,
    pub session: u64,
    pub before: String,
    pub candidates: Vec<String>,
    pub limit: usize,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationQuery {
    pub id: u64,
    pub session: u64,
    pub before: String,
    pub prefix: String,
    pub exclude: Vec<String>,
    pub limit: usize,
}

impl GenerationQuery {
    pub fn valid(&self) -> bool {
        self.before.len() <= 16_384
            && self.prefix.len() <= 256
            && self.limit <= 3
            && self.exclude.len() <= 3
            && self.exclude.iter().all(|w| w.len() <= 128)
    }
    pub fn accepts(&self, words: &[String]) -> bool {
        let prefix = switchify_prediction::normalize(&self.prefix);
        words.len() <= self.limit
            && words.iter().all(|w| {
                valid_word(w)
                    && w.starts_with(&prefix)
                    && !self
                        .exclude
                        .iter()
                        .any(|e| switchify_prediction::normalize(e) == *w)
            })
            && words
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == words.len()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhraseQuery {
    pub id: u64,
    pub session: u64,
    pub before: String,
    pub prefix: String,
    pub exclude: Vec<Vec<String>>,
    pub limit: usize,
}

impl PhraseQuery {
    pub fn valid(&self) -> bool {
        self.before.len() <= 16_384
            && self.prefix.len() <= 256
            && self.limit <= 3
            && self.exclude.len() <= 3
            && self
                .exclude
                .iter()
                .all(|p| p.len() == PHRASE_WORDS && p.iter().all(|w| w.len() <= 128))
    }
    /// Distinct phrases of whole words, the first completing the prefix,
    /// none excluded.
    pub fn accepts(&self, phrases: &[Vec<String>]) -> bool {
        let prefix = switchify_prediction::normalize(&self.prefix);
        let exclude: Vec<Vec<String>> = self
            .exclude
            .iter()
            .map(|p| {
                p.iter()
                    .map(|w| switchify_prediction::normalize(w))
                    .collect()
            })
            .collect();
        phrases.len() <= self.limit
            && phrases.iter().all(|p| {
                p.len() == PHRASE_WORDS
                    && p.iter().all(|w| valid_word(w))
                    && p[0].starts_with(&prefix)
                    && !exclude.contains(p)
            })
            && phrases
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == phrases.len()
    }
}

/// A canonical, single whole word. Apostrophes are allowed only internally.
pub fn valid_word(word: &str) -> bool {
    !word.is_empty()
        && word.len() <= 128
        && switchify_prediction::normalize(word) == word
        && switchify_prediction::sentences(word) == vec![vec![word.to_owned()]]
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Command {
    Predict(Query),
    Generate(GenerationQuery),
    GeneratePhrases(PhraseQuery),
    Reset,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Reply {
    Ready {
        version: u32,
        accelerated: bool,
    },
    Ranked {
        id: u64,
        words: Vec<String>,
        cache_hit: bool,
    },
    Generated {
        id: u64,
        words: Vec<String>,
        cache_hit: bool,
    },
    GeneratedPhrases {
        id: u64,
        phrases: Vec<Vec<String>>,
        cache_hit: bool,
    },
    Reset,
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid worker frame")
}

pub fn write_frame(mut out: impl Write, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|_| invalid())?;
    if bytes.is_empty() || bytes.len() > MAX_FRAME {
        return Err(invalid());
    }
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(&bytes)?;
    out.flush()
}

pub fn read_frame<T: DeserializeOwned>(mut input: impl Read) -> io::Result<T> {
    let mut header = [0; 4];
    input.read_exact(&mut header)?;
    let length = u32::from_le_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(invalid());
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes).map_err(|_| invalid())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phrase_queries_bound_inputs_and_accept_only_whole_new_phrases() {
        let phrase = |a: &str, b: &str| vec![a.to_owned(), b.to_owned()];
        let query = PhraseQuery {
            id: 1,
            session: 1,
            before: "i".into(),
            prefix: "wo".into(),
            exclude: vec![phrase("would", "like")],
            limit: 2,
        };
        assert!(query.valid());
        assert!(query.accepts(&[phrase("would", "love"), phrase("work", "is")]));
        assert!(query.accepts(&[]));
        for phrases in [
            vec![phrase("would", "like")],
            vec![phrase("would", "love"), phrase("would", "love")],
            vec![
                phrase("would", "love"),
                phrase("work", "is"),
                phrase("won", "it"),
            ],
            vec![phrase("help", "me")],
            vec![phrase("would", "Love")],
            vec![phrase("would", "two words")],
            vec![vec!["would".into()]],
            vec![vec!["would".into(), "love".into(), "it".into()]],
        ] {
            assert!(!query.accepts(&phrases));
        }
        for invalid in [
            PhraseQuery {
                limit: 4,
                ..query.clone()
            },
            PhraseQuery {
                exclude: vec![phrase("a", "b"); 4],
                ..query.clone()
            },
            PhraseQuery {
                exclude: vec![vec!["a".into()]],
                ..query.clone()
            },
            PhraseQuery {
                exclude: vec![phrase("a", &"b".repeat(129))],
                ..query.clone()
            },
            PhraseQuery {
                prefix: "a".repeat(257),
                ..query.clone()
            },
        ] {
            assert!(!invalid.valid());
        }
    }
    #[test]
    fn rejects_bad_frames_without_echoing_input() {
        for bytes in [vec![], vec![255; 4], vec![0; 4], vec![1, 0, 0, 0, b'!']] {
            assert!(read_frame::<Reply>(&bytes[..]).is_err());
        }
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &Command::Reset).unwrap();
        assert!(matches!(
            read_frame::<Command>(&bytes[..]).unwrap(),
            Command::Reset
        ));
    }
}
