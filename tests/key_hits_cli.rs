use std::{fs, process::Command};
use switchify_prediction::build;

#[test]
fn cli_is_opt_in_and_returns_explicit_substitution_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(&db, "Hello hello.", "synthetic").unwrap();
    let before = fs::read(&db).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_switchify-prediction"));
    command
        .args(["predict", "--baseline"])
        .arg(&db)
        .args(["--prefix", "hrl"]);
    let exact = command.output().unwrap();
    assert!(exact.status.success());
    assert!(
        serde_json::from_slice::<Vec<serde_json::Value>>(&exact.stdout)
            .unwrap()
            .is_empty()
    );
    let soft = command.arg("--nearby-qwerty").output().unwrap();
    assert!(soft.status.success());
    let result: Vec<serde_json::Value> = serde_json::from_slice(&soft.stdout).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["word"], "hello");
    assert_eq!(result[0]["substitution"]["typed"], "r");
    assert_eq!(result[0]["substitution"]["grapheme_index"], 1);
    assert_eq!(result[0]["substitution"]["intended"], "e");
    let score = result[0]["score"].as_f64().unwrap();
    let language_score = result[0]["language_score"].as_f64().unwrap();
    let key_weight = result[0]["key_weight"].as_f64().unwrap();
    assert!(language_score > 0.);
    assert!(key_weight > 0. && key_weight < 1.);
    assert_eq!(score, language_score * key_weight);
    assert_eq!(fs::read(&db).unwrap(), before);
    assert!(!command.arg("--phrases").output().unwrap().status.success());
}
