use rusqlite::Connection;
use switchify_prediction::{
    Options, Predictor, build,
    experimental::{CombinedPredictor, LegacyPredictor, Source, compare},
};
fn fixture(path: &std::path::Path) {
    let c = Connection::open(path).unwrap();
    c.execute_batch("CREATE TABLE WORDS(ID INTEGER,WORD TEXT,BASE_FREQUENCY INTEGER,USER_FREQUENCY INTEGER); INSERT INTO WORDS VALUES(1,'I',9,0),(2,'would',8,0),(3,'like',7,0),(4,'water',1,999999),(5,'wine',20,0),(6,'walk',30,0),(7,'CAFÉ',4,0),(8,'café',2,0),(9,'can’t',3,0),(10,'two words',999,0),(11,'123',998,0); CREATE TABLE BIGRAMS(ID1 INTEGER,ID2 INTEGER,BASE_FREQUENCY INTEGER,USER_FREQUENCY INTEGER); INSERT INTO BIGRAMS VALUES(3,5,9,0); CREATE TABLE TRIGRAMS(ID1 INTEGER,ID2 INTEGER,ID3 INTEGER,BASE_FREQUENCY INTEGER); CREATE TABLE QUADGRAMS(ID1 INTEGER,ID2 INTEGER,ID3 INTEGER,ID4 INTEGER,BASE_FREQUENCY INTEGER); INSERT INTO QUADGRAMS VALUES(1,2,3,4,1);").unwrap();
}
fn opts() -> Options {
    Options {
        limit: 5,
        min_chars: 0,
        unigram_only: false,
    }
}
#[test]
fn historical_ranking_normalization_and_no_database_access_after_load() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("old.db");
    fixture(&p);
    let bytes = std::fs::read(&p).unwrap();
    let legacy = LegacyPredictor::open(&p).unwrap();
    assert_eq!(
        legacy
            .predict("I would like ", "w", opts())
            .iter()
            .map(|s| s.word.as_str())
            .collect::<Vec<_>>(),
        ["water", "wine", "walk", "would"]
    );
    assert_eq!(legacy.predict("I would like.", "w", opts())[0].word, "walk");
    assert_eq!(
        legacy.predict("I would like\n", "w", opts())[0].word,
        "walk"
    );
    assert_eq!(legacy.predict("", "CAFE\u{301}", opts()).len(), 1);
    assert_eq!(legacy.predict("", "can’", opts())[0].word, "can't");
    assert!(legacy.predict("", "12", opts()).is_empty());
    assert!(legacy.predict("", "two", opts()).is_empty());
    assert!(
        legacy
            .predict(
                "",
                "w",
                Options {
                    min_chars: 2,
                    ..opts()
                }
            )
            .is_empty()
    );
    assert!(
        legacy
            .predict("", "", Options { limit: 0, ..opts() })
            .is_empty()
    );
    assert_eq!(
        legacy.predict(
            "I would like ",
            "w",
            Options {
                unigram_only: true,
                ..opts()
            }
        )[0]
        .word,
        "walk"
    );
    assert_eq!(std::fs::read(&p).unwrap(), bytes);
    std::fs::remove_file(&p).unwrap();
    assert!(!legacy.predict("", "", opts()).is_empty());
}
#[test]
fn combined_keeps_positions_fills_slots_and_ignores_personal_counts() {
    let d = tempfile::tempdir().unwrap();
    let old = d.path().join("old.db");
    fixture(&old);
    let new = d.path().join("new.db");
    build(&new, "I want water. Café can’t wait.", "fixture").unwrap();
    let old_bytes = std::fs::read(&old).unwrap();
    let new_bytes = std::fs::read(&new).unwrap();
    let mixed = CombinedPredictor::open(&new, &old).unwrap();
    let modern = Predictor::open(&new, None).unwrap();
    for prefix in ["", "w", "wa", "caf", "can’", "zz"] {
        let a = modern.predict("I would like ", prefix, opts());
        let b = mixed.predict("I would like ", prefix, opts());
        assert!(b.len() <= 5);
        for (i, s) in a.iter().enumerate() {
            assert_eq!(s.word, b[i].word);
            assert_eq!(b[i].source, Source::Newer);
        }
        let unique: std::collections::HashSet<_> = b.iter().map(|r| &r.word).collect();
        assert_eq!(unique.len(), b.len());
    }
    let words = mixed.predict("I would like ", "w", opts());
    assert!(
        words
            .iter()
            .any(|r| r.word == "wine" && r.source == Source::Original)
    );
    let full = mixed.predict("", "", Options { limit: 1, ..opts() });
    assert_eq!(full.len(), 1);
    assert_eq!(full[0].source, Source::Newer);
    assert!(
        mixed
            .predict(
                "",
                "w",
                Options {
                    min_chars: 2,
                    ..opts()
                }
            )
            .is_empty()
    );
    assert_eq!(std::fs::read(&old).unwrap(), old_bytes);
    assert_eq!(std::fs::read(&new).unwrap(), new_bytes);
    let text = d.path().join("test.txt");
    std::fs::write(&text, "I would like wine.").unwrap();
    let report = compare("combined", &new, &old, &text).unwrap();
    assert_eq!(report["position_violations"], 0);
    assert!(report["warm_query_count"].as_u64().unwrap() >= 1000);
}
#[test]
fn corrupt_missing_and_dangling_data_fail_and_cli_requires_paths() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("old.db");
    assert!(LegacyPredictor::open(&p).is_err());
    fixture(&p);
    Connection::open(&p)
        .unwrap()
        .execute("INSERT INTO BIGRAMS VALUES(1,9999,1,0)", [])
        .unwrap();
    assert!(LegacyPredictor::open(&p).is_err());
    std::fs::write(&p, "bad sqlite").unwrap();
    assert!(LegacyPredictor::open(&p).is_err());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_switchify-prediction"))
        .args(["experimental-predict", "--mode", "combined"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
