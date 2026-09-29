use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
use switchify_prediction::{Options, Predictor, build, evaluation, normalize, sentences, validate};

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
    fs::write(&text, "I drink watermelon.").unwrap();
    let mut p = Predictor::open(&path, Some(&personal)).unwrap();
    assert!(p.import(&text).unwrap());
    assert!(!p.import(&text).unwrap());
    assert_eq!(words(&p, "I drink ", "wa")[0], "watermelon");
    p.learn("I drink walnut.").unwrap();
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
        .write_all(b"hello walnut")
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
    p.learn("i like tea.").unwrap();
    let learned = p.predict("i like ", "tea", opts).remove(0).score;
    assert!((learned - (0.9 * 6.0 / 7.0 + 0.1 * 6.0 / 24.0)).abs() < 1e-12);
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
