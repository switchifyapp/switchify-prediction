//! The n-gram baseline file: parity with SQLite baselines, and rejection of
//! malformed files.
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
use switchify_prediction::{
    Options, Predictor, build, build_ngrams, convert, ngrams::MAGIC, validate,
};

const TEXT: &str = "I drink water. I drink water. I drink wine. We walk home. We walk home. \
    apple apricot. café école. Don't stop! I need to call Siobhan about the physio. \
    the physio is late. the physio appointment. we drink coffee. i like tea. i like coffee.";
const QUERIES: &[(&str, &str)] = &[
    ("", ""),
    ("", "w"),
    ("", "ca"),
    ("", "zzzz"),
    ("I drink", ""),
    ("I drink", "wa"),
    ("we walk", "h"),
    ("unknown walk", ""),
    ("We walk.", ""),
    ("the physio", "a"),
    ("siobhan", ""),
    ("I need to", "c"),
    ("don't", "s"),
    ("i like", "t"),
];

fn ranked(p: &Predictor, before: &str, prefix: &str, unigram_only: bool) -> Vec<(String, u64)> {
    p.predict(
        before,
        prefix,
        Options {
            limit: 50,
            min_chars: 0,
            unigram_only,
        },
    )
    .into_iter()
    .map(|s| (s.word, s.score.to_bits()))
    .collect()
}
fn phrases(p: &Predictor, before: &str, prefix: &str) -> Vec<(Vec<String>, u64)> {
    p.predict_phrases(
        before,
        prefix,
        Options {
            limit: 50,
            min_chars: 0,
            unigram_only: false,
        },
    )
    .into_iter()
    .map(|s| (s.words, s.score.to_bits()))
    .collect()
}
fn assert_same_predictions(a: &Predictor, b: &Predictor, label: &str) {
    for (before, prefix) in QUERIES {
        for unigram_only in [false, true] {
            assert_eq!(
                ranked(a, before, prefix, unigram_only),
                ranked(b, before, prefix, unigram_only),
                "{label}: {before:?} {prefix:?} unigram_only={unigram_only}"
            );
        }
        assert_eq!(
            phrases(a, before, prefix),
            phrases(b, before, prefix),
            "{label}: phrases {before:?} {prefix:?}"
        );
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    sqlite: std::path::PathBuf,
    ngrams: std::path::PathBuf,
}
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let sqlite = dir.path().join("baseline.sqlite");
    let ngrams = dir.path().join("baseline.ngrams");
    build(&sqlite, TEXT, "synthetic").unwrap();
    convert(&sqlite, &ngrams).unwrap();
    Fixture {
        dir,
        sqlite,
        ngrams,
    }
}

#[test]
fn converted_and_built_files_match_the_sqlite_baseline_exactly() {
    let f = fixture();
    let built = f.dir.path().join("built.ngrams");
    build_ngrams(&built, TEXT, "synthetic").unwrap();
    // The encoding is deterministic: converting and building give the same bytes.
    assert_eq!(fs::read(&f.ngrams).unwrap(), fs::read(&built).unwrap());
    let sqlite = validate(&f.sqlite).unwrap();
    let ngrams = validate(&f.ngrams).unwrap();
    assert_eq!(sqlite.format, "sqlite");
    assert_eq!(ngrams.format, "ngrams");
    assert_eq!(sqlite.logical_sha256, ngrams.logical_sha256);
    assert_eq!(sqlite.vocabulary, ngrams.vocabulary);
    assert_eq!(sqlite.ngrams, ngrams.ngrams);
    assert_eq!(ngrams.kind, "baseline");
    assert!(fs::metadata(&f.ngrams).unwrap().len() < fs::metadata(&f.sqlite).unwrap().len());

    let from_sqlite = Predictor::open(&f.sqlite, None).unwrap();
    let from_ngrams = Predictor::open(&f.ngrams, None).unwrap();
    assert_same_predictions(&from_sqlite, &from_ngrams, "no learning");
    assert_eq!(
        from_ngrams.predict("I drink ", "wa", Options::default())[0].word,
        "water"
    );
}

#[test]
fn learning_on_either_format_matches_the_other() {
    let f = fixture();
    let personal = f.dir.path().join("personal.sqlite");
    let mut from_sqlite = Predictor::open(&f.sqlite, Some(&personal)).unwrap();
    let mut from_ngrams = Predictor::open(&f.ngrams, Some(&personal)).unwrap();
    for text in [
        "I drink walnut milk. I drink walnut milk.",
        "Siobhan said the physio is at Ballymena.",
        "We walk home with Siobhan. I like teh tea.",
        "",
    ] {
        from_sqlite.learn(text).unwrap();
        from_ngrams.refresh_personal().unwrap();
        assert_same_predictions(&from_sqlite, &from_ngrams, text);
        let reopened = Predictor::open(&f.ngrams, Some(&personal)).unwrap();
        assert_same_predictions(&from_sqlite, &reopened, text);
    }
    from_ngrams.learn("the physio rang Siobhan").unwrap();
    from_sqlite.refresh_personal().unwrap();
    assert_same_predictions(&from_sqlite, &from_ngrams, "learned through ngrams");
    assert!(ranked(&from_ngrams, "I drink", "wal", false)[0].0 == "walnut");
    from_ngrams.reset_personal().unwrap();
    from_sqlite.refresh_personal().unwrap();
    assert_same_predictions(&from_sqlite, &from_ngrams, "after reset");
    assert_same_predictions(
        &from_ngrams,
        &Predictor::open(&f.ngrams, None).unwrap(),
        "reset equals none",
    );
    // The baseline file is never written.
    assert_eq!(
        fs::read(&f.ngrams).unwrap(),
        fs::read(f.dir.path().join("baseline.ngrams")).unwrap()
    );
}

#[test]
fn outputs_are_never_replaced_and_only_baselines_convert() {
    let f = fixture();
    let before = fs::read(&f.ngrams).unwrap();
    assert!(convert(&f.sqlite, &f.ngrams).is_err());
    assert!(build_ngrams(&f.ngrams, "other words", "x").is_err());
    assert_eq!(before, fs::read(&f.ngrams).unwrap());
    assert!(build_ngrams(&f.dir.path().join("empty.ngrams"), "123 !!", "x").is_err());
    assert!(!f.dir.path().join("empty.ngrams").exists());
    let personal = f.dir.path().join("personal.sqlite");
    drop(Predictor::open(&f.ngrams, Some(&personal)).unwrap());
    let out = f.dir.path().join("personal.ngrams");
    assert!(convert(&personal, &out).is_err());
    assert!(!out.exists());
    assert!(convert(&f.ngrams, &out).is_err(), "only SQLite converts");
    assert!(!out.exists());
    // An n-gram file is not a personal database.
    assert!(Predictor::open(&f.sqlite, Some(&f.ngrams)).is_err());
    assert!(Predictor::open(&f.ngrams, Some(&f.ngrams)).is_err());
}

/// Byte offsets of the file's regions, from the documented layout.
struct Layout {
    version: usize,
    metadata: std::ops::Range<usize>,
    n_words: usize,
    word_offsets: usize,
    words: usize,
    unigram_counts: usize,
    bigram_starts: usize,
    bigram_ids: usize,
    bigram_counts: usize,
    trigram_contexts: usize,
    end: usize,
}
fn u32_at(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize
}
fn layout(bytes: &[u8]) -> Layout {
    let version = MAGIC.len();
    let metadata_len = u32_at(bytes, version + 4);
    let metadata = version + 8..version + 8 + metadata_len;
    let n_words_at = metadata.end;
    let n_words = u32_at(bytes, n_words_at);
    let word_offsets = n_words_at + 4;
    let words = word_offsets + (n_words + 1) * 4;
    let unigram_counts = words + u32_at(bytes, word_offsets + n_words * 4);
    let bigram_starts = unigram_counts + n_words * 8;
    let bigram_ids = bigram_starts + (n_words + 1) * 4;
    let n_bigrams = u32_at(bytes, bigram_starts + n_words * 4);
    let bigram_counts = bigram_ids + n_bigrams * 4;
    let trigram_contexts = bigram_counts + n_bigrams * 8 + 4;
    Layout {
        version,
        metadata,
        n_words,
        word_offsets,
        words,
        unigram_counts,
        bigram_starts,
        bigram_ids,
        bigram_counts,
        trigram_contexts,
        end: bytes.len() - 32,
    }
}
fn resigned(mut bytes: Vec<u8>) -> Vec<u8> {
    let end = bytes.len() - 32;
    let hash = Sha256::digest(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash);
    bytes
}
fn rejected(dir: &Path, name: &str, bytes: &[u8]) {
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    assert!(validate(&path).is_err(), "validate accepted {name}");
    assert!(
        Predictor::open(&path, None).is_err(),
        "open accepted {name}"
    );
    assert!(
        switchify_prediction::production::validate_english(&path).is_err(),
        "production validation accepted {name}"
    );
}

#[test]
fn malformed_and_truncated_files_are_rejected() {
    let f = fixture();
    let bytes = fs::read(&f.ngrams).unwrap();
    let l = layout(&bytes);
    assert!(validate(&f.ngrams).is_ok());
    assert_eq!(
        resigned(bytes.clone()),
        bytes,
        "the trailer is the file hash"
    );
    for len in [
        0,
        MAGIC.len(),
        l.version + 4,
        l.metadata.start,
        l.metadata.end,
        l.words,
        l.unigram_counts + 3,
        l.bigram_counts,
        l.trigram_contexts,
        l.end - 1,
        l.end,
        bytes.len() - 1,
    ] {
        rejected(f.dir.path(), &format!("truncated-{len}"), &bytes[..len]);
    }
    rejected(f.dir.path(), "empty", &[]);
    rejected(f.dir.path(), "not-ngrams", b"not an n-gram file at all");
    let mut longer = bytes.clone();
    longer.push(0);
    rejected(f.dir.path(), "longer", &longer);
    let mut padded = bytes[..l.end].to_vec();
    padded.extend_from_slice(&[0; 36]);
    rejected(f.dir.path(), "padded", &resigned(padded));
    // Every byte matters to the hash.
    for at in (0..l.end).step_by(97).chain([l.end - 1]) {
        let mut flipped = bytes.clone();
        flipped[at] ^= 0x01;
        rejected(f.dir.path(), &format!("flipped-{at}"), &flipped);
    }
    let mutate = |name: &str, edit: &dyn Fn(&mut Vec<u8>)| {
        let mut mutated = bytes.clone();
        edit(&mut mutated);
        rejected(f.dir.path(), name, &resigned(mutated));
    };
    mutate("version", &|b| b[l.version] = 2);
    let metadata = String::from_utf8(bytes[l.metadata.clone()].to_vec()).unwrap();
    for (name, from, to) in [
        ("kind", "\"baseline\"", "\"baseXXXX\""),
        ("language", "\"en\"", "\"fr\""),
        ("metadata-json", "\"kind\"", "\"kinX\""),
    ] {
        assert!(metadata.contains(from), "{name}");
        let replaced = metadata.replacen(from, to, 1);
        assert_eq!(replaced.len(), metadata.len());
        mutate(name, &|b| {
            b[l.metadata.clone()].copy_from_slice(replaced.as_bytes())
        });
    }
    mutate("no-words", &|b| {
        b[l.n_words..l.n_words + 4].copy_from_slice(&0u32.to_le_bytes())
    });
    mutate("first-offset", &|b| b[l.word_offsets] = 1);
    mutate("word-order", &|b| {
        // Swap the first two words' bytes where they have equal length.
        let (a, c, d) = (
            u32_at(b, l.word_offsets),
            u32_at(b, l.word_offsets + 4),
            u32_at(b, l.word_offsets + 8),
        );
        let (first, second) = (
            b[l.words + a..l.words + c].to_vec(),
            b[l.words + c..l.words + d].to_vec(),
        );
        if first.len() == second.len() {
            b[l.words + a..l.words + c].copy_from_slice(&second);
            b[l.words + c..l.words + d].copy_from_slice(&first);
        } else {
            b[l.words + a] = b'z';
        }
    });
    mutate("uppercase-word", &|b| b[l.words] = b'A');
    mutate("invalid-utf8", &|b| b[l.words] = 0xff);
    mutate("zero-unigram", &|b| {
        b[l.unigram_counts..l.unigram_counts + 8].fill(0)
    });
    mutate("huge-unigram", &|b| {
        b[l.unigram_counts..l.unigram_counts + 8].fill(0xff)
    });
    mutate("bigram-start", &|b| b[l.bigram_starts] = 1);
    mutate("bigram-id-out-of-range", &|b| {
        b[l.bigram_ids..l.bigram_ids + 4].copy_from_slice(&(l.n_words as u32).to_le_bytes())
    });
    mutate("bigram-count-above-unigram", &|b| {
        b[l.bigram_counts..l.bigram_counts + 8].copy_from_slice(&1000u64.to_le_bytes())
    });
    mutate("bigram-zero-count", &|b| {
        b[l.bigram_counts..l.bigram_counts + 8].fill(0)
    });
    mutate("bigram-order", &|b| {
        // The first two records of a context with at least two, swapped.
        let start = (0..l.n_words)
            .map(|context| u32_at(b, l.bigram_starts + context * 4))
            .zip((1..=l.n_words).map(|next| u32_at(b, l.bigram_starts + next * 4)))
            .find(|(start, end)| end - start >= 2)
            .expect("a word with two followers")
            .0;
        for (region, width) in [(l.bigram_ids, 4), (l.bigram_counts, 8)] {
            let at = region + start * width;
            let first = b[at..at + width].to_vec();
            let second = b[at + width..at + 2 * width].to_vec();
            b[at..at + width].copy_from_slice(&second);
            b[at + width..at + 2 * width].copy_from_slice(&first);
        }
    });
    mutate("trigram-context-out-of-range", &|b| {
        b[l.trigram_contexts..l.trigram_contexts + 4]
            .copy_from_slice(&(l.n_words as u32).to_le_bytes())
    });
    mutate("trigram-context-order", &|b| {
        b[l.trigram_contexts..l.trigram_contexts + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        b[l.trigram_contexts..l.trigram_contexts + 4]
            .copy_from_slice(&((l.n_words - 1) as u32).to_le_bytes());
        b[l.trigram_contexts + 4..l.trigram_contexts + 8]
            .copy_from_slice(&((l.n_words - 1) as u32).to_le_bytes());
    });
    // The untouched file is still fine.
    assert!(validate(&f.ngrams).is_ok());
}

#[test]
fn cli_builds_converts_validates_and_predicts_with_ngram_files() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    fs::write(&input, "hello world. hello water. hello water.").unwrap();
    let exe = env!("CARGO_BIN_EXE_switchify-prediction");
    let run = |args: &[&str]| {
        let output = Command::new(exe).args(args).output().unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
    };
    let (sqlite, built, converted) = (
        dir.path().join("base.sqlite"),
        dir.path().join("built.ngrams"),
        dir.path().join("converted.ngrams"),
    );
    let (input, sqlite_s, built_s, converted_s) = (
        input.to_str().unwrap(),
        sqlite.to_str().unwrap(),
        built.to_str().unwrap(),
        converted.to_str().unwrap(),
    );
    let from_text = run(&[
        "build", "--input", input, "--output", built_s, "--format", "ngrams",
    ]);
    let from_sqlite = run(&["build", "--input", input, "--output", sqlite_s]);
    let from_convert = run(&["convert", "--input", sqlite_s, "--output", converted_s]);
    assert_eq!(from_text["format"], "ngrams");
    assert_eq!(from_sqlite["format"], "sqlite");
    assert_eq!(from_text, from_convert);
    assert_eq!(from_text["logical_sha256"], from_sqlite["logical_sha256"]);
    assert_eq!(run(&["validate", "--database", built_s]), from_text);
    let predicted = run(&[
        "predict",
        "--baseline",
        converted_s,
        "--before",
        "hello",
        "--prefix",
        "w",
        "--min-chars",
        "0",
    ]);
    assert_eq!(predicted[0]["word"], "water");
    assert_eq!(
        predicted,
        run(&[
            "predict",
            "--baseline",
            sqlite_s,
            "--before",
            "hello",
            "--prefix",
            "w",
            "--min-chars",
            "0",
        ])
    );
    let personal = dir.path().join("personal.sqlite");
    let learned = run(&[
        "import",
        "--baseline",
        built_s,
        "--personal",
        personal.to_str().unwrap(),
        "--input",
        input,
    ]);
    assert_eq!(learned["imported"], true);
    assert!(
        !Command::new(exe)
            .args([
                "build",
                "--output",
                dir.path().join("x").to_str().unwrap(),
                "--format",
                "ngrams"
            ])
            .output()
            .unwrap()
            .status
            .success(),
        "the production model is built as SQLite"
    );
    assert!(
        !Command::new(exe)
            .args(["validate", "--database", built_s, "--production"])
            .output()
            .unwrap()
            .status
            .success()
    );
}

/// Run with SWITCHIFY_BASELINE (the production english.sqlite), SWITCHIFY_NGRAMS
/// (its conversion) and SWITCHIFY_PARITY_TEXT (sentences to query). Compares
/// every word and phrase prediction, with and without learning.
#[test]
#[ignore = "requires explicit SWITCHIFY_BASELINE, SWITCHIFY_NGRAMS and SWITCHIFY_PARITY_TEXT"]
fn production_parity() {
    use unicode_segmentation::UnicodeSegmentation;
    let env = |name: &str| std::path::PathBuf::from(std::env::var_os(name).unwrap());
    let (sqlite, ngrams, text) = (
        env("SWITCHIFY_BASELINE"),
        env("SWITCHIFY_NGRAMS"),
        fs::read_to_string(env("SWITCHIFY_PARITY_TEXT")).unwrap(),
    );
    let sentences = switchify_prediction::sentences(&text);
    let dir = tempfile::tempdir().unwrap();
    let personal = dir.path().join("personal.sqlite");
    let mut from_sqlite = Predictor::open(&sqlite, Some(&personal)).unwrap();
    let mut queries = 0;
    let options = Options {
        limit: 8,
        min_chars: 0,
        unigram_only: false,
    };
    for phase in ["none", "learned"] {
        if phase == "learned" {
            for sentence in sentences.iter().take(sentences.len() / 2) {
                from_sqlite.learn(&sentence.join(" ")).unwrap();
            }
        }
        let from_ngrams = Predictor::open(&ngrams, Some(&personal)).unwrap();
        for sentence in &sentences {
            for (i, target) in sentence.iter().enumerate() {
                let before = sentence[..i].join(" ");
                let graphemes: Vec<_> = target.graphemes(true).collect();
                for n in 0..=graphemes.len().min(2) {
                    let prefix = graphemes[..n].concat();
                    let a = from_sqlite.predict(&before, &prefix, options);
                    let b = from_ngrams.predict(&before, &prefix, options);
                    assert_eq!(
                        a.iter()
                            .map(|s| (&s.word, s.score.to_bits()))
                            .collect::<Vec<_>>(),
                        b.iter()
                            .map(|s| (&s.word, s.score.to_bits()))
                            .collect::<Vec<_>>(),
                        "{phase}: {before:?} {prefix:?}"
                    );
                    let a = from_sqlite.predict_phrases(&before, &prefix, options);
                    let b = from_ngrams.predict_phrases(&before, &prefix, options);
                    assert_eq!(
                        a.iter()
                            .map(|s| (&s.words, s.score.to_bits()))
                            .collect::<Vec<_>>(),
                        b.iter()
                            .map(|s| (&s.words, s.score.to_bits()))
                            .collect::<Vec<_>>(),
                        "{phase}: phrases {before:?} {prefix:?}"
                    );
                    queries += 1;
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "sentences": sentences.len(),
            "queries_per_phase": queries / 2,
            "sqlite_sha256": switchify_prediction::digest(&fs::read(&sqlite).unwrap()),
            "ngrams_sha256": switchify_prediction::digest(&fs::read(&ngrams).unwrap()),
            "text_sha256": switchify_prediction::digest(text.as_bytes()),
            "identical": true,
        })
    );
}

/// Run with SWITCHIFY_NGRAMS pointing at a converted production baseline.
/// Prints how long reading, hashing, parsing and opening take.
#[test]
#[ignore = "requires explicit SWITCHIFY_NGRAMS"]
fn profile_ngrams_open() {
    let path = std::path::PathBuf::from(std::env::var_os("SWITCHIFY_NGRAMS").unwrap());
    let ms = |started: std::time::Instant| started.elapsed().as_secs_f64() * 1000.0;
    let mut phases = Vec::new();
    let started = std::time::Instant::now();
    let bytes = fs::read(&path).unwrap();
    phases.push(("read_file", ms(started)));
    let started = std::time::Instant::now();
    let _ = Sha256::digest(&bytes);
    phases.push(("sha256_only", ms(started)));
    let started = std::time::Instant::now();
    let ngrams = switchify_prediction::ngrams::Ngrams::parse(bytes.clone()).unwrap();
    phases.push(("parse_including_hash_and_checks", ms(started)));
    let started = std::time::Instant::now();
    let logical = ngrams.logical_sha256().unwrap();
    phases.push(("logical_fingerprint_for_validate_only", ms(started)));
    let started = std::time::Instant::now();
    let predictor = Predictor::open(&path, None).unwrap();
    phases.push(("predictor_open_total", ms(started)));
    let started = std::time::Instant::now();
    let validation = validate(&path).unwrap();
    phases.push(("validate_total", ms(started)));
    println!(
        "{}",
        serde_json::json!({
            "phases_ms": phases.iter().map(|(k, v)| (k.to_string(), v)).collect::<std::collections::BTreeMap<_, _>>(),
            "file_bytes": bytes.len(),
            "payload_bytes": predictor.model_payload_bytes(),
            "vocabulary": validation.vocabulary,
            "ngrams": validation.ngrams,
            "logical_sha256": logical,
            "file_sha256": ngrams.file_sha256(),
        })
    );
}
