use std::{fs, process::Command};
use switchify_prediction::{ABBREVIATION_WEIGHT, Options, Predictor, build, convert};

fn options() -> Options {
    Options {
        limit: 10,
        min_chars: 0,
        ..Options::default()
    }
}

fn predictor(text: &str) -> (tempfile::TempDir, Predictor) {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(&db, text, "synthetic").unwrap();
    let predictor = Predictor::open(&db, None).unwrap();
    (tmp, predictor)
}

#[test]
fn letters_in_order_suggest_the_word_and_are_flagged() {
    let (_tmp, p) = predictor("Talk about it. About time. About. Absent. Bat. Abt.");
    let result = p.predict_abbreviated("", "abt", options());
    let words: Vec<_> = result.iter().map(|s| s.word.as_str()).collect();
    assert_eq!(words, ["about", "abt", "absent"]);
    assert!(result[0].abbreviated);
    assert_eq!(
        result[0].score,
        result[0].language_score * ABBREVIATION_WEIGHT
    );
    assert!(!result[1].abbreviated);
    assert_eq!(result[1].score, result[1].language_score);
}

#[test]
fn exact_completions_keep_full_scores_and_match_predict() {
    let (_tmp, p) = predictor("The cat sat. The car. The cart. Cleat. Cat.");
    for typed in ["", "c", "ca", "car"] {
        let exact = p.predict("the ", typed, options());
        let abbreviated = p.predict_abbreviated("the ", typed, options());
        let flagged: Vec<_> = abbreviated
            .iter()
            .filter(|s| !s.abbreviated)
            .map(|s| (s.word.clone(), s.score))
            .collect();
        let expected: Vec<_> = exact.into_iter().map(|s| (s.word, s.score)).collect();
        assert_eq!(flagged, expected, "typed {typed:?}");
    }
    let ct = p.predict_abbreviated("", "ct", options());
    assert!(ct.iter().any(|s| s.word == "cat" && s.abbreviated));
    assert!(ct.iter().any(|s| s.word == "cleat" && s.abbreviated));
    assert!(!ct.iter().any(|s| s.word == "car"));
}

#[test]
fn options_context_and_formats_are_respected() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    let text = "I went home. Go home. Go house. Go home. Hello.";
    build(&db, text, "synthetic").unwrap();
    let file = tmp.path().join("base.ngrams");
    convert(&db, &file).unwrap();
    let sqlite = Predictor::open(&db, None).unwrap();
    let ngrams = Predictor::open(&file, None).unwrap();
    assert_eq!(
        sqlite.predict_abbreviated("go ", "hm", options()),
        ngrams.predict_abbreviated("go ", "hm", options())
    );
    let defaults = Options::default();
    assert!(sqlite.predict_abbreviated("", "h", defaults).is_empty());
    let limited = Options {
        limit: 1,
        ..options()
    };
    assert_eq!(sqlite.predict_abbreviated("go ", "hm", limited).len(), 1);
    let none = Options {
        limit: 0,
        ..options()
    };
    assert!(sqlite.predict_abbreviated("go ", "hm", none).is_empty());
    assert_eq!(
        sqlite.predict_abbreviated("", "HM", options())[0].word,
        "home"
    );
    let long = "h".repeat(40);
    assert!(sqlite.predict_abbreviated("", &long, options()).is_empty());
}

#[test]
fn learned_words_are_included_once_usable() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(&db, "Hello there.", "synthetic").unwrap();
    let personal = tmp.path().join("personal.sqlite");
    let mut p = Predictor::open(&db, Some(&personal)).unwrap();
    p.learn("Switchify works. ").unwrap();
    assert!(p.predict_abbreviated("", "swfy", options()).is_empty());
    p.learn("Switchify again. ").unwrap();
    let result = p.predict_abbreviated("", "swfy", options());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].word, "switchify");
    assert!(result[0].abbreviated);
}

#[test]
fn cli_flag_is_opt_in() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(&db, "Talk about it.", "synthetic").unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_switchify-prediction"));
    command
        .args(["predict", "--baseline"])
        .arg(&db)
        .args(["--prefix", "abt"]);
    let exact = command.output().unwrap();
    assert!(exact.status.success());
    assert!(
        serde_json::from_slice::<Vec<serde_json::Value>>(&exact.stdout)
            .unwrap()
            .is_empty()
    );
    let before = fs::read(&db).unwrap();
    let opted = command.arg("--abbreviations").output().unwrap();
    assert!(opted.status.success());
    let result: Vec<serde_json::Value> = serde_json::from_slice(&opted.stdout).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["word"], "about");
    assert_eq!(result[0]["abbreviated"], true);
    assert_eq!(fs::read(&db).unwrap(), before);
    let mut nearby = Command::new(env!("CARGO_BIN_EXE_switchify-prediction"));
    nearby.args(["predict", "--baseline"]).arg(&db).args([
        "--prefix",
        "abt",
        "--abbreviations",
        "--nearby-qwerty",
    ]);
    assert!(!nearby.output().unwrap().status.success());
    assert!(!command.arg("--phrases").output().unwrap().status.success());
}

#[test]
fn long_typed_text_is_exact_only_and_curly_apostrophes_match() {
    let exact = "a".repeat(34);
    let spread = format!("{}b{}", "a".repeat(16), "a".repeat(17));
    let (_tmp, p) = predictor(&format!("{exact}. {spread}. Don't. Didn't."));
    let typed = "a".repeat(33);
    let result = p.predict_abbreviated("", &typed, options());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].word, exact);
    assert!(!result[0].abbreviated);
    let curly = p.predict_abbreviated("", "dn’t", options());
    let words: Vec<_> = curly.iter().map(|s| s.word.as_str()).collect();
    assert_eq!(words, ["didn't", "don't"]);
    assert!(curly.iter().all(|s| s.abbreviated));
}
