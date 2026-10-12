use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
use switchify_prediction::{
    Options, Predictor, build, digest, evaluation, normalize, sentences, validate,
};

fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("baseline.sqlite");
    build(
        &path,
        "I drink water. I drink water. I drink wine. We walk home. We walk home. apple apricot.",
        "synthetic",
    )
    .unwrap();
    (tmp, path)
}
fn words(p: &Predictor, context: &str, prefix: &str) -> Vec<String> {
    p.predict(
        context,
        prefix,
        Options {
            min_chars: 0,
            ..Options::default()
        },
    )
    .into_iter()
    .map(|s| s.word)
    .collect()
}
#[test]
fn normalization_sentences_and_contractions() {
    assert_eq!(normalize("CAFÉ cafe\u{301} DON’T"), "café café don't");
    assert_eq!(
        sentences("Don't stop! Next\n'word' café."),
        vec![vec!["don't", "stop"], vec!["next"], vec!["word", "café"]]
    );
}
#[test]
fn context_backoff_prefix_and_thresholds() {
    let (_tmp, path) = fixture();
    let p = Predictor::open(&path, None).unwrap();
    assert_eq!(words(&p, "I drink ", "w")[0], "water");
    assert_eq!(words(&p, "We walk ", "")[0], "home");
    assert_eq!(words(&p, "unseen walk ", "")[0], "home");
    assert_eq!(words(&p, "unknown ", ""), words(&p, "", ""));
    assert_eq!(words(&p, "We walk. ", ""), words(&p, "", ""));
    assert!(words(&p, "", "zzzz").is_empty());
    assert!(p.predict("", "w", Options::default()).is_empty());
    assert!(
        p.predict("", "wa", Options::default())
            .iter()
            .all(|s| s.word.starts_with("wa"))
    );
    assert!(
        p.predict(
            "",
            "",
            Options {
                limit: 0,
                min_chars: 0,
                unigram_only: false
            }
        )
        .is_empty()
    );
    assert_eq!(words(&p, "", "ap"), ["apple", "apricot"]);
}
#[test]
fn unicode_graphemes_and_lowercase_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("x.sqlite");
    build(&db, "café école", "test").unwrap();
    let p = Predictor::open(&db, None).unwrap();
    assert!(p.predict("", "e\u{301}", Options::default()).is_empty());
    assert_eq!(words(&p, "", "CA")[0], "café");
}
#[test]
fn learning_persistence_import_dedup_reset_and_baseline_immutability() {
    let (tmp, path) = fixture();
    let original = fs::read(&path).unwrap();
    let personal = tmp.path().join("personal.sqlite");
    let text = tmp.path().join("training.txt");
    fs::write(&text, "I drink watermelon. I like watermelon.").unwrap();
    let mut p = Predictor::open(&path, Some(&personal)).unwrap();
    assert!(p.import(&text).unwrap());
    assert!(!p.import(&text).unwrap());
    assert_eq!(words(&p, "I drink ", "wa")[0], "watermelon");
    p.learn("I drink walnut. I drink walnut.").unwrap();
    drop(p);
    let mut p = Predictor::open(&path, Some(&personal)).unwrap();
    assert!(words(&p, "I drink ", "wa").contains(&"walnut".to_string()));
    p.reset_personal().unwrap();
    assert!(!words(&p, "", "wa").contains(&"walnut".to_string()));
    assert!(p.import(&text).unwrap());
    assert_eq!(original, fs::read(&path).unwrap());
    assert!(Predictor::open(&path, Some(&path)).is_err());
}
#[test]
fn transaction_failure_rolls_back_counts_and_import_marker() {
    let (tmp, path) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&path, Some(&personal)).unwrap();
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute_batch("CREATE TRIGGER fail BEFORE INSERT ON counts WHEN NEW.word='zebra' BEGIN SELECT RAISE(ABORT,'test'); END;").unwrap();
    let input = tmp.path().join("text.txt");
    fs::write(&input, "aardvark zebra").unwrap();
    assert!(p.import(&input).is_err());
    assert!(words(&p, "", "aard").is_empty());
    assert_eq!(
        db.query_row("SELECT count(*) FROM counts", [], |r| r.get::<_, u32>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM imports", [], |r| r.get::<_, u32>(0))
            .unwrap(),
        0
    );
    db.execute_batch("DROP TRIGGER fail").unwrap();
    assert!(p.import(&input).unwrap());
}
#[test]
fn invalid_databases_and_non_destructive_build() {
    let (tmp, path) = fixture();
    let original = fs::read(&path).unwrap();
    assert!(build(&path, "replacement", "test").is_err());
    assert_eq!(original, fs::read(&path).unwrap());
    let absent = tmp.path().join("absent");
    assert!(Predictor::open(&absent, None).is_err());
    assert!(!absent.exists());
    let corrupt = tmp.path().join("corrupt");
    fs::write(&corrupt, "not sqlite").unwrap();
    assert!(validate(&corrupt).is_err());
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("PRAGMA user_version=99").unwrap();
    assert!(Predictor::open(&path, None).is_err());
    assert!(build(&tmp.path().join("empty"), "123 !!", "test").is_err());
}
#[test]
fn builds_have_deterministic_logical_contents() {
    let (tmp, path) = fixture();
    let other = tmp.path().join("other.sqlite");
    build(
        &other,
        "I drink water. I drink water. I drink wine. We walk home. We walk home. apple apricot.",
        "synthetic",
    )
    .unwrap();
    assert_eq!(
        validate(&path).unwrap().logical_sha256,
        validate(&other).unwrap().logical_sha256
    );
}
#[test]
fn evaluation_deduplicates_and_separates_sentences() {
    let text = "One two. One two. Three four. Five six. Seven eight. Nine ten. Eleven twelve. Thirteen fourteen. Fifteen sixteen. Seventeen eighteen. Nineteen twenty.";
    let (train, test) = evaluation::split(text);
    assert_eq!((train.len(), test.len()), (9, 1));
    assert!(!train.iter().any(|s| test.contains(s)));
    assert_eq!((train, test), evaluation::split(text));
    let report = evaluation::evaluate(text, "test machine".into()).unwrap();
    assert_eq!(report.accuracy.len(), 5);
    assert_eq!(report.test_sentences, 1);
    assert!(report.warm_p95_ms.is_finite());
}
#[test]
fn cli_build_predict_learn_validate() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("input.txt");
    let db = tmp.path().join("base.sqlite");
    let personal = tmp.path().join("personal.sqlite");
    fs::write(&input, "hello world. hello water.").unwrap();
    let exe = env!("CARGO_BIN_EXE_switchify-prediction");
    let result = Command::new(exe)
        .args(["build", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&db)
        .output()
        .unwrap();
    assert!(result.status.success(), "{:?}", result);
    let mut child = Command::new(exe)
        .args(["learn", "--baseline"])
        .arg(&db)
        .arg("--personal")
        .arg(&personal)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hello walnut. hello walnut")
        .unwrap();
    assert!(child.wait().unwrap().success());
    let output = Command::new(exe)
        .args(["predict", "--baseline"])
        .arg(&db)
        .arg("--personal")
        .arg(&personal)
        .args(["--before", "hello ", "--prefix", "wa"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value[0]["word"], "walnut");
    assert!(
        Command::new(exe)
            .args(["validate", "--database"])
            .arg(Path::new(&db))
            .stdout(Stdio::null())
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn interpolation_renormalizes_and_personal_counts_are_weighted_before_probability() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("baseline.sqlite");
    let personal = tmp.path().join("personal.sqlite");
    build(&base, "i like tea. i like coffee. we drink coffee.", "test").unwrap();
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    let opts = Options {
        limit: 100,
        min_chars: 0,
        unigram_only: false,
    };
    let tea = p.predict("i like ", "tea", opts).remove(0).score;
    assert!((tea - (0.6 * 0.5 + 0.3 * 0.5 + 0.1 / 9.0)).abs() < 1e-12);
    let backed_off = p.predict("unknown like ", "tea", opts).remove(0).score;
    assert!((backed_off - (0.3 * 0.5 + 0.1 / 9.0) / 0.4).abs() < 1e-12);
    // Words this rare in the baseline count once learned twice.
    p.learn("i like tea.").unwrap();
    assert_eq!(p.predict("i like ", "tea", opts).remove(0).score, tea);
    p.learn("i like tea.").unwrap();
    let learned = p.predict("i like ", "tea", opts).remove(0).score;
    // One learned occurrence counts as 500 baseline ones, less as more is
    // learned: here six words.
    let w = 500.0 / (1.0 + 6.0 / 2000.0);
    let expected =
        0.9 * (1.0 + 2.0 * w) / (2.0 + 2.0 * w) + 0.1 * (1.0 + 2.0 * w) / (9.0 + 6.0 * w);
    assert!((learned - expected).abs() < 1e-12);
    let total: f64 = p.predict("i like ", "", opts).iter().map(|s| s.score).sum();
    assert!((total - 1.0).abs() < 1e-12);
}

#[test]
fn sentence_boundaries_do_not_create_cross_sentence_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("base.sqlite");
    build(
        &path,
        "hello world. next thing! another day\nlast line",
        "test",
    )
    .unwrap();
    let db = rusqlite::Connection::open(&path).unwrap();
    for (context, word) in [
        ("[\"world\"]", "next"),
        ("[\"thing\"]", "another"),
        ("[\"day\"]", "last"),
    ] {
        let count: u32 = db
            .query_row(
                "SELECT count(*) FROM counts WHERE context=?1 AND word=?2",
                [context, word],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn wrong_personal_kind_and_invalid_utf8_are_rejected_without_changes() {
    let (tmp, base) = fixture();
    let copy = tmp.path().join("copy.sqlite");
    fs::copy(&base, &copy).unwrap();
    let before = fs::read(&copy).unwrap();
    assert!(Predictor::open(&base, Some(&copy)).is_err());
    assert_eq!(before, fs::read(&copy).unwrap());
    let personal = tmp.path().join("personal.sqlite");
    let input = tmp.path().join("bad.txt");
    fs::write(&input, [0xff, 0xfe]).unwrap();
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    let checksum = validate(&personal).unwrap().logical_sha256;
    assert!(p.import(&input).is_err());
    assert_eq!(checksum, validate(&personal).unwrap().logical_sha256);
}

#[test]
fn incomplete_schema_and_inconsistent_models_are_rejected() {
    let mutations = [
        "DROP TABLE vocabulary",
        "DROP TABLE imports",
        "DELETE FROM metadata WHERE key='provenance'",
        "DELETE FROM counts WHERE context='[]'",
        "DELETE FROM counts",
        "DELETE FROM vocabulary WHERE word='water'",
        "INSERT INTO vocabulary VALUES ('ghost')",
        "INSERT INTO counts VALUES ('[\"ghost\"]','water',1)",
        "UPDATE counts SET count=100 WHERE context='[\"drink\"]' AND word='water'",
        "UPDATE counts SET context='[ ]' WHERE context='[]'",
        "INSERT INTO imports VALUES ('not-a-sha256')",
    ];
    for mutation in mutations {
        let (_tmp, path) = fixture();
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch(&format!("PRAGMA foreign_keys=OFF; {mutation}"))
            .unwrap();
        assert!(validate(&path).is_err(), "validate accepted {mutation}");
        assert!(
            Predictor::open(&path, None).is_err(),
            "open accepted {mutation}"
        );
    }
}

#[test]
fn empty_personal_database_is_valid_but_missing_its_schema_is_not() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    drop(Predictor::open(&base, Some(&personal)).unwrap());
    assert_eq!(validate(&personal).unwrap().vocabulary, 0);
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute_batch("DROP TABLE imports").unwrap();
    assert!(validate(&personal).is_err());
    assert!(Predictor::open(&base, Some(&personal)).is_err());
}

#[test]
fn validation_failure_rolls_back_disk_and_keeps_prediction_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.sqlite");
    let personal = dir.path().join("personal.sqlite");
    build(&base, "hello water", "fixture").unwrap();
    let mut predictor = Predictor::open(&base, Some(&personal)).unwrap();
    let before = predictor.predict("hello", "wa", Options::default());
    let baseline_hash = digest(&fs::read(&base).unwrap());
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute_batch("CREATE TRIGGER poison AFTER INSERT ON counts WHEN NEW.word='walnut' BEGIN INSERT OR IGNORE INTO vocabulary VALUES('orphan'); END;").unwrap();
    assert!(predictor.learn("hello walnut").is_err());
    assert_eq!(predictor.predict("hello", "wa", Options::default()), before);
    assert_eq!(validate(&personal).unwrap().ngrams, 0);
    db.execute_batch("DROP TRIGGER poison").unwrap();
    predictor.learn("hello walnut. hello walnut").unwrap();
    assert_eq!(
        predictor.predict("hello", "wa", Options::default())[0].word,
        "walnut"
    );
    assert_eq!(digest(&fs::read(&base).unwrap()), baseline_hash);
}

#[test]
fn multiple_predictors_refresh_duplicate_imports_and_reset() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.sqlite");
    let personal = dir.path().join("personal.sqlite");
    let training = dir.path().join("fixture.txt");
    build(&base, "hello water", "fixture").unwrap();
    fs::write(&training, "hello walnut. hello walnut").unwrap();
    let mut first = Predictor::open(&base, Some(&personal)).unwrap();
    let mut second = Predictor::open(&base, Some(&personal)).unwrap();
    assert!(first.import(&training).unwrap());
    assert_eq!(
        second.predict("hello", "wa", Options::default())[0].word,
        "water"
    );
    assert!(!second.import(&training).unwrap());
    assert_eq!(
        second.predict("hello", "wa", Options::default())[0].word,
        "walnut"
    );
    first.reset_personal().unwrap();
    second.refresh_personal().unwrap();
    assert_eq!(
        second.predict("hello", "wa", Options::default())[0].word,
        "water"
    );
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute("INSERT INTO vocabulary VALUES('orphan')", [])
        .unwrap();
    assert!(second.refresh_personal().is_err());
    assert_eq!(
        second.predict("hello", "wa", Options::default())[0].word,
        "water"
    );
}

#[test]
fn simultaneous_personal_creation_and_writes_preserve_all_segments() {
    use std::sync::{Arc, Barrier};
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.sqlite");
    let personal = dir.path().join("personal.sqlite");
    build(&base, "hello water", "fixture").unwrap();
    let baseline_hash = digest(&fs::read(&base).unwrap());
    let barrier = Arc::new(Barrier::new(4));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let barrier = barrier.clone();
            let base = base.clone();
            let personal = personal.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let mut p = Predictor::open(&base, Some(&personal)).unwrap();
                p.learn("hello walnut").unwrap();
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    let db = rusqlite::Connection::open(&personal).unwrap();
    let count: i64 = db
        .query_row(
            "SELECT count FROM counts WHERE context='[]' AND word='walnut'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 4);
    validate(&personal).unwrap();
    assert_eq!(digest(&fs::read(&base).unwrap()), baseline_hash);
}

fn ranked(p: &Predictor, context: &str, prefix: &str) -> Vec<(String, u64)> {
    p.predict(
        context,
        prefix,
        Options {
            limit: 20,
            min_chars: 0,
            unigram_only: false,
        },
    )
    .into_iter()
    .map(|s| (s.word, s.score.to_bits()))
    .collect()
}

fn assert_matches_reload(p: &Predictor, base: &Path, personal: &Path) {
    let reloaded = Predictor::open(base, Some(personal)).unwrap();
    for (context, prefix) in [
        ("", ""),
        ("", "w"),
        ("I drink", ""),
        ("I drink", "wa"),
        ("we walk", "h"),
        ("Siobhan said", ""),
        ("the physio", "a"),
    ] {
        assert_eq!(
            ranked(p, context, prefix),
            ranked(&reloaded, context, prefix),
            "{context:?} {prefix:?}"
        );
    }
}

#[test]
fn incremental_learning_matches_a_full_reload() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    for text in [
        "I drink walnut milk.",
        "Siobhan said the physio is at Ballymena. I drink water!",
        "We walk home with Siobhan",
        "the physio appointment",
    ] {
        p.learn(text).unwrap();
        assert_matches_reload(&p, &base, &personal);
    }
    assert_eq!(words(&p, "Siobhan said the", "p"), ["physio"]);
}

#[test]
fn learning_after_another_writer_includes_its_counts() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut first = Predictor::open(&base, Some(&personal)).unwrap();
    let mut second = Predictor::open(&base, Some(&personal)).unwrap();
    first
        .learn("I drink walnut milk. I drink walnut milk.")
        .unwrap();
    second
        .learn("I drink walrus milk. I drink walrus milk.")
        .unwrap();
    assert!(words(&second, "I drink", "wal").starts_with(&["walnut".into(), "walrus".into()]));
    assert_matches_reload(&second, &base, &personal);
    second.learn("We walk home.").unwrap();
    assert_matches_reload(&second, &base, &personal);
    first.reset_personal().unwrap();
    second.learn("apple pie").unwrap();
    assert_eq!(words(&second, "I drink", "wal"), ["walk"]);
    assert_matches_reload(&second, &base, &personal);
}

/// Run with SWITCHIFY_BASELINE pointing at the production english.sqlite.
#[test]
#[ignore = "requires explicit SWITCHIFY_BASELINE"]
fn learning_cost_on_production_baseline() {
    let base = std::path::PathBuf::from(std::env::var_os("SWITCHIFY_BASELINE").unwrap());
    let tmp = tempfile::tempdir().unwrap();
    let personal = tmp.path().join("personal.sqlite");
    let started = std::time::Instant::now();
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    println!("open: {:?}", started.elapsed());
    let mut times = Vec::new();
    for i in 0..50 {
        let started = std::time::Instant::now();
        p.learn(&format!(
            "I need to call Siobhan about the physio appointment number {i}."
        ))
        .unwrap();
        times.push(started.elapsed());
    }
    times.sort();
    println!("learn median {:?}, max {:?}", times[25], times[49]);
    assert_eq!(words(&p, "I need to call", "si")[0], "siobhan");
}

#[test]
fn a_trigger_side_effect_is_still_validated_before_commit() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute_batch(
        "CREATE TRIGGER ghost AFTER INSERT ON counts WHEN NEW.word='zebra' AND NEW.context='[]' \
         BEGIN INSERT OR IGNORE INTO counts VALUES('[\"ghost\"]','zebra',1); END;",
    )
    .unwrap();
    p.learn("hello there").unwrap();
    assert!(p.learn("zebra").is_err());
    db.execute_batch("DROP TRIGGER ghost").unwrap();
    assert!(Predictor::open(&base, Some(&personal)).is_ok());
    assert_matches_reload(&p, &base, &personal);
}

#[test]
fn duplicate_imports_and_empty_text_match_a_full_reload() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let input = tmp.path().join("text.txt");
    fs::write(&input, "Siobhan said the physio is late.").unwrap();
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    assert!(p.import(&input).unwrap());
    assert!(!p.import(&input).unwrap());
    p.learn("").unwrap();
    p.learn("... !!").unwrap();
    assert_matches_reload(&p, &base, &personal);
}

#[test]
fn a_replacing_constraint_is_still_validated_before_commit() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute_batch(
        "PRAGMA foreign_keys=OFF; DROP INDEX counts_word; ALTER TABLE counts RENAME TO old; \
         CREATE TABLE counts(context TEXT NOT NULL,word TEXT NOT NULL REFERENCES vocabulary(word),\
         count INTEGER NOT NULL,PRIMARY KEY(context,word),UNIQUE(word) ON CONFLICT REPLACE) WITHOUT ROWID; \
         CREATE INDEX counts_word ON counts(word); DROP TABLE old;",
    )
    .unwrap();
    p.learn("hello").unwrap();
    assert!(p.learn("say hello").is_err());
    assert!(Predictor::open(&base, Some(&personal)).is_ok());
}

#[test]
fn a_misspelling_learned_once_is_not_suggested_but_a_new_word_used_twice_is() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    // A word the baseline knows well counts from its first learned use; one
    // it has rarely seen, which may itself be a misspelling, needs a second.
    assert_eq!(words(&p, "We walk ", "")[0], "home");
    p.learn("We walk wine.").unwrap();
    assert_eq!(words(&p, "We walk ", "")[0], "home");
    p.learn("We walk drink.").unwrap();
    assert_eq!(words(&p, "We walk ", "")[0], "drink");
    p.learn("I drink teh water.").unwrap();
    assert!(!words(&p, "I drink ", "t").contains(&"teh".to_string()));
    assert!(!words(&p, "", "te").contains(&"teh".to_string()));
    p.learn("I drink teh tea.").unwrap();
    assert!(words(&p, "I drink ", "t").contains(&"teh".to_string()));
}

#[test]
fn an_empty_personal_database_predicts_exactly_like_none() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let with = Predictor::open(&base, Some(&personal)).unwrap();
    let without = Predictor::open(&base, None).unwrap();
    for (context, prefix) in [("", ""), ("I drink", ""), ("I drink", "wa"), ("we", "w")] {
        assert_eq!(
            ranked(&with, context, prefix),
            ranked(&without, context, prefix)
        );
    }
}

#[test]
fn huge_stored_personal_counts_do_not_overflow() {
    let (tmp, base) = fixture();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    p.learn("I drink water. I drink tea.").unwrap();
    drop(p);
    let db = rusqlite::Connection::open(&personal).unwrap();
    db.execute("UPDATE counts SET count = 922337203685477580", [])
        .unwrap();
    validate(&personal).unwrap();
    let p = Predictor::open(&base, Some(&personal)).unwrap();
    let ranked = p.predict(
        "I drink ",
        "",
        Options {
            limit: 20,
            min_chars: 0,
            unigram_only: false,
        },
    );
    assert!(!ranked.is_empty());
    assert!(ranked.iter().all(|s| s.score.is_finite()));
}

#[test]
fn without_learning_scores_are_exactly_the_interpolated_counts() {
    let (_tmp, base) = fixture();
    let p = Predictor::open(&base, None).unwrap();
    let water = p.predict("I drink ", "wa", Options::default()).remove(0);
    assert_eq!(water.word, "water");
    // Seventeen baseline words; "i drink" and "drink" are followed by water
    // twice and wine once. Summed in the order the predictor sums them.
    let weights: f64 = 0.1 + 0.3 + 0.6;
    let expected =
        0.0 + 2.0 / 17.0 * 0.1 / weights + 2.0 / 3.0 * 0.3 / weights + 2.0 / 3.0 * 0.6 / weights;
    assert_eq!(water.score.to_bits(), expected.to_bits());
}

/// Run with SWITCHIFY_BASELINE pointing at the production english.sqlite.
#[test]
#[ignore = "requires explicit SWITCHIFY_BASELINE"]
fn prediction_cost_on_production_baseline() {
    let base = std::path::PathBuf::from(std::env::var_os("SWITCHIFY_BASELINE").unwrap());
    let tmp = tempfile::tempdir().unwrap();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&base, Some(&personal)).unwrap();
    let options = Options {
        limit: 6,
        min_chars: 0,
        unigram_only: false,
    };
    let time = |p: &Predictor, label: &str| {
        for (before, prefix) in [("", ""), ("i want to", ""), ("the", "s")] {
            let started = std::time::Instant::now();
            for _ in 0..50 {
                p.predict(before, prefix, options);
            }
            println!(
                "{label} {before:?} {prefix:?}: {:?}",
                started.elapsed() / 50
            );
        }
    };
    time(&p, "none");
    // Digits split words, so each new word is spelled in letters.
    let letters = |mut i: usize| {
        let mut word = String::from("zq");
        loop {
            word.push((b'a' + (i % 26) as u8) as char);
            i /= 26;
            if i == 0 {
                return word;
            }
        }
    };
    for (count, label) in [(2_000, "2k new words"), (40_000, "40k new words")] {
        let text: String = (0..count)
            .map(|i| format!("I need to call {} about the physio. ", letters(i)))
            .collect();
        p.learn(&text).unwrap();
        p.learn(&text).unwrap();
        let started = std::time::Instant::now();
        p.learn("I need to call the physio about my zzhoist.")
            .unwrap();
        println!("{label}: one learn {:?}", started.elapsed());
        // A second use makes the new word usable, adding its counts in one pass.
        let started = std::time::Instant::now();
        p.learn("I need to call the physio about my zzhoist.")
            .unwrap();
        println!(
            "{label}: learn making a word usable {:?}",
            started.elapsed()
        );
        time(&p, label);
    }
}
