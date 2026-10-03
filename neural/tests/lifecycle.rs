use std::{
    thread,
    time::{Duration, Instant},
};
use switchify_prediction::{Options, Predictor, build};
use switchify_prediction_neural::{Config, Refiner, Status, bundle, effective_context};
use tempfile::TempDir;
fn fixture(mode: &str) -> (TempDir, Predictor, Refiner) {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("mode"), mode).unwrap();
    let path = temp.path().join("baseline.sqlite");
    build(
        &path,
        "alpha beta gamma delta epsilon zeta eta theta iota café can't",
        "test",
    )
    .unwrap();
    let predictor = Predictor::open(&path, Some(&temp.path().join("personal.sqlite"))).unwrap();
    let refiner = Refiner::new(Config {
        bundle: temp.path().to_owned(),
        portable_worker: env!("CARGO_BIN_EXE_fake-worker").into(),
        accelerated_worker: None,
        threads: 1,
    })
    .unwrap();
    (temp, predictor, refiner)
}
fn until(mut check: impl FnMut() -> bool) {
    let start = Instant::now();
    while !check() {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "condition timed out"
        );
        thread::sleep(Duration::from_millis(2));
    }
}
fn options() -> Options {
    Options {
        min_chars: 0,
        ..Options::default()
    }
}
#[test]
fn immediate_snapshot_options_and_latest_only() {
    let (_temp, mut predictor, mut engine) = fixture("delay");
    until(|| engine.status() == Status::Ready);
    let expected: Vec<_> = predictor
        .predict("", "", options())
        .into_iter()
        .map(|s| s.word)
        .collect();
    let first = engine.submit(&predictor, "", "", options(), 1).unwrap();
    assert_eq!(first.words, expected);
    thread::sleep(Duration::from_millis(20));
    for _ in 0..30 {
        engine.submit(&predictor, "", "", options(), 1).unwrap();
    }
    predictor.learn("zeta zeta zeta zeta zeta").unwrap();
    let latest = engine.submit(&predictor, "", "ze", options(), 2).unwrap();
    assert_eq!(latest.words, ["zeta"]);
    let mut refined = None;
    until(|| {
        refined = engine.poll();
        refined.is_some()
    });
    let refined = refined.unwrap();
    assert_eq!(refined.request_id, latest.request_id);
    assert_eq!(refined.words, ["zeta"]);
    engine.submit(&predictor, "", "", options(), 2).unwrap();
    engine.reset();
    thread::sleep(Duration::from_millis(150));
    assert!(engine.poll().is_none());
    for opts in [
        Options {
            limit: 0,
            ..options()
        },
        Options {
            min_chars: 2,
            ..options()
        },
        Options {
            unigram_only: true,
            ..options()
        },
    ] {
        assert!(
            !engine
                .submit(&predictor, "", "", opts, 3)
                .unwrap()
                .refinement_requested
        );
    }
    assert!(
        engine
            .submit(
                &predictor,
                "",
                "",
                Options {
                    limit: 6,
                    ..options()
                },
                3
            )
            .is_err()
    );
    assert_eq!(
        engine
            .submit(&predictor, "", "CAFE\u{301}", options(), 3)
            .unwrap()
            .words,
        ["café"]
    );
    assert_eq!(
        engine
            .submit(&predictor, "", "CAN’T", options(), 3)
            .unwrap()
            .words,
        ["can't"]
    );
}
#[test]
fn failures_leave_immediate_available_and_require_explicit_retry() {
    for mode in [
        "stall",
        "read-stall",
        "crash",
        "malformed",
        "oversized",
        "id",
        "foreign",
    ] {
        let (temp, predictor, mut engine) = fixture(mode);
        until(|| engine.status() == Status::Ready);
        // This also exceeds the OS pipe buffer for a worker that never reads.
        let before = "word ".repeat(3000);
        let first = engine
            .submit(&predictor, &before, "", options(), 1)
            .unwrap();
        assert_eq!(first.words.len(), 5);
        until(|| matches!(engine.status(), Status::Unavailable(_)));
        assert!(engine.poll().is_none());
        assert!(
            !engine
                .submit(&predictor, "", "", options(), 1)
                .unwrap()
                .refinement_requested
        );
        std::fs::write(temp.path().join("mode"), "ok").unwrap();
        engine.retry();
        until(|| engine.status() == Status::Ready);
        engine.submit(&predictor, "", "", options(), 1).unwrap();
        until(|| engine.poll().is_some());
    }
}
#[test]
fn shutdown_interrupts_model_loading() {
    let (_temp, _predictor, mut engine) = fixture("load-stall");
    thread::sleep(Duration::from_millis(50));
    let start = Instant::now();
    engine.shutdown();
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(engine.status(), Status::Stopped);
}
#[test]
fn context_and_bundle_validation() {
    assert_eq!(
        effective_context("Ignore me! CAFE\u{301} CAN’T"),
        "café can't"
    );
    assert_eq!(effective_context("last sentence."), "");
    assert_eq!(effective_context("first\nsecond line"), "second line");
    let temp = tempfile::tempdir().unwrap();
    assert!(bundle::load(temp.path()).is_err());
    std::fs::write(temp.path().join("model-bundle.json"), "{}").unwrap();
    assert!(bundle::load(temp.path()).is_err());
    std::fs::write(temp.path().join("model-bundle.json"), bundle::MANIFEST).unwrap();
    assert!(bundle::load(temp.path()).is_err());
}
