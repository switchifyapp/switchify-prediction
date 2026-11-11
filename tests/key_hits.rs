use switchify_prediction::{KeyHitMap, KeyPosition, Options, Predictor, build, convert};

fn options() -> Options {
    Options {
        limit: 5,
        min_chars: 0,
        ..Options::default()
    }
}

#[test]
fn geometry_is_normalized_bounded_and_validated() {
    let map = KeyHitMap::qwerty();
    assert!(map.neighbors("R").iter().any(|a| a.key == "e"));
    assert!(
        !map.neighbors("r")
            .iter()
            .any(|a| a.key == "r" || a.key == "p")
    );
    assert!(map.neighbors("é").is_empty());
    let dense: Vec<_> = "abcdefghijk"
        .chars()
        .enumerate()
        .map(|(i, key)| KeyPosition {
            key,
            x: i as f64 / 100.,
            y: 0.,
        })
        .collect();
    assert_eq!(
        KeyHitMap::new(&dense, 1., 0.45)
            .unwrap()
            .neighbors("a")
            .len(),
        8
    );
    for key in "abcdefghijklmnopqrstuvwxyz".chars() {
        let neighbors = map.neighbors(&key.to_string());
        assert!(neighbors.len() <= 8);
        assert!(neighbors.iter().all(|a| a.weight > 0. && a.weight <= 1.));
    }
    let key = KeyPosition {
        key: 'a',
        x: 0.,
        y: 0.,
    };
    for keys in [
        vec![],
        vec![key, key],
        vec![
            key,
            KeyPosition {
                key: 'A',
                x: 1.,
                y: 0.,
            },
        ],
        vec![KeyPosition { x: f64::NAN, ..key }],
        vec![KeyPosition { key: ' ', ..key }],
    ] {
        assert!(KeyHitMap::new(&keys, 1.25, 0.45).is_err());
    }
    for bad in [0., -1., f64::INFINITY, f64::NAN] {
        assert!(KeyHitMap::new(&[key], bad, 0.45).is_err());
        assert!(KeyHitMap::new(&[key], 1.25, bad).is_err());
    }
}

#[test]
fn one_nearby_substitution_uses_context_and_keeps_exact_api_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(
        &db,
        &format!("{} We see tree. Hello there.", "We drink tea. ".repeat(20)),
        "synthetic",
    )
    .unwrap();
    let p = Predictor::open(&db, None).unwrap();
    let map = KeyHitMap::qwerty();
    let exact = p.predict("We drink ", "tr", options());
    assert_eq!(exact[0].word, "tree");
    let soft = p.predict_with_hit_map("We drink ", "tr", options(), &map);
    assert_eq!(soft[0].word, "tea");
    let correction = soft[0].substitution.as_ref().unwrap();
    assert_eq!(
        (
            correction.grapheme_index,
            correction.typed.as_str(),
            correction.intended.as_str()
        ),
        (1, "r", "e")
    );
    assert_eq!(soft[0].score, soft[0].language_score * soft[0].key_weight);
    assert_eq!(p.predict("We drink ", "tr", options()), exact);
    assert!(
        p.predict_with_hit_map("", "hrl", options(), &map)
            .iter()
            .any(|s| s.word == "hello")
    );
    assert!(
        p.predict_with_hit_map("", "gel", options(), &map)
            .iter()
            .any(|s| s.word == "hello")
    );
    assert!(
        !p.predict_with_hit_map("", "hrr", options(), &map)
            .iter()
            .any(|s| s.word == "hello")
    );
    assert!(
        !p.predict_with_hit_map("", "hpl", options(), &map)
            .iter()
            .any(|s| s.word == "hello")
    );
    assert!(
        p.predict_with_hit_map("", "h", Options::default(), &map)
            .is_empty()
    );
    assert!(
        p.predict_with_hit_map(
            "",
            "hrl",
            Options {
                limit: 0,
                ..options()
            },
            &map
        )
        .is_empty()
    );
}

#[test]
fn custom_layout_unicode_and_long_prefix_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    let long = "a".repeat(40);
    build(&db, &format!("café café can't. {}.", long), "synthetic").unwrap();
    let p = Predictor::open(&db, None).unwrap();
    let map = KeyHitMap::new(
        &[
            KeyPosition {
                key: 'é',
                x: 0.,
                y: 0.,
            },
            KeyPosition {
                key: 'r',
                x: 1.,
                y: 0.,
            },
        ],
        1.25,
        0.45,
    )
    .unwrap();
    let result = p.predict_with_hit_map("", "cafr", options(), &map);
    assert_eq!(result[0].word, "café");
    let clean = p.predict_with_hit_map("", "CAFE\u{301}", options(), &map);
    assert_eq!(clean[0].word, "café");
    assert!(clean[0].substitution.is_none());
    for prefix in ["", "can't", &long[..33]] {
        let exact = p.predict("", prefix, options());
        let hits = p.predict_with_hit_map("", prefix, options(), &map);
        assert_eq!(
            hits.iter()
                .map(|s| (&s.word, s.language_score))
                .collect::<Vec<_>>(),
            exact.iter().map(|s| (&s.word, s.score)).collect::<Vec<_>>()
        );
    }
    let too_long_typo = format!("{}s", "a".repeat(32));
    assert!(
        p.predict_with_hit_map("", &too_long_typo, options(), &KeyHitMap::qwerty())
            .is_empty()
    );
}

#[test]
fn compact_and_sqlite_and_personal_learning_agree() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    let compact = tmp.path().join("base.ngrams");
    build(&db, "Hello hello. Help me.", "synthetic").unwrap();
    convert(&db, &compact).unwrap();
    let map = KeyHitMap::qwerty();
    let mut a = Predictor::open(&db, Some(&tmp.path().join("a.sqlite"))).unwrap();
    let mut b = Predictor::open(&compact, Some(&tmp.path().join("b.sqlite"))).unwrap();
    for p in [&mut a, &mut b] {
        p.learn("Zelda zelda.").unwrap();
    }
    for prefix in ["hrl", "zrl", "hel"] {
        assert_eq!(
            a.predict_with_hit_map("", prefix, options(), &map),
            b.predict_with_hit_map("", prefix, options(), &map)
        );
    }
    assert!(
        a.predict_with_hit_map("", "zrl", options(), &map)
            .iter()
            .any(|s| s.word == "zelda")
    );
}

#[test]
fn unicode_boundaries_are_preserved_and_overlapping_prefixes_keep_best_score() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("base.sqlite");
    build(&db, "가나다 क़लम.", "synthetic").unwrap();
    let p = Predictor::open(&db, None).unwrap();
    let jamo = KeyHitMap::new(
        &[
            KeyPosition {
                key: 'x',
                x: 0.,
                y: 0.,
            },
            KeyPosition {
                key: '\u{1100}',
                x: 1.,
                y: 0.,
            },
        ],
        1.25,
        0.45,
    )
    .unwrap();
    assert!(
        p.predict_with_hit_map("", "x\u{1161}", options(), &jamo)
            .is_empty()
    );
    let indic = KeyHitMap::new(
        &[
            KeyPosition {
                key: 'x',
                x: 0.,
                y: 0.,
            },
            KeyPosition {
                key: 'क',
                x: 1.,
                y: 0.,
            },
            KeyPosition {
                key: '\u{0958}',
                x: 0.25,
                y: 0.,
            },
        ],
        1.25,
        0.45,
    )
    .unwrap();
    let results = p.predict_with_hit_map("", "x", options(), &indic);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].substitution.as_ref().unwrap().intended, "क़");
    assert_eq!(results[0].key_weight, indic.neighbors("x")[0].weight);
    let exact = p.predict_with_hit_map("", "क", options(), &indic);
    assert_eq!(exact.len(), 1);
    assert!(exact[0].substitution.is_none());
}
