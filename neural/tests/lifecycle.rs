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
fn replies_beyond_half_a_second_refine_without_blocking_the_snapshot() {
    for mode in ["slow", "slow-reset"] {
        let (_temp, predictor, mut engine) = fixture(mode);
        until(|| engine.status() == Status::Ready);
        assert_eq!(engine.capabilities().deadline_ms, 2_000);
        let immediate = engine.submit(&predictor, "", "", options(), 1).unwrap();
        assert_eq!(immediate.words.len(), 5);
        assert!(immediate.refinement_requested);
        assert!(engine.poll().is_none());
        let mut refined = None;
        until(|| {
            refined = engine.poll();
            refined.is_some()
        });
        assert_eq!(refined.unwrap().request_id, immediate.request_id);
        assert_eq!(engine.status(), Status::Ready);
    }
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
fn learning_after_submission_does_not_change_captured_shortlist() {
    let (_temp, mut predictor, mut engine) = fixture("delay");
    until(|| engine.status() == Status::Ready);
    let mut expected: Vec<_> = predictor
        .predict(
            "",
            "",
            Options {
                limit: 8,
                ..options()
            },
        )
        .into_iter()
        .map(|s| s.word)
        .collect();
    expected.reverse();
    expected.truncate(5);
    engine.submit(&predictor, "", "", options(), 1).unwrap();
    predictor.learn("newword newword newword newword").unwrap();
    let mut result = None;
    until(|| {
        result = engine.poll();
        result.is_some()
    });
    assert_eq!(result.unwrap().words, expected);
}

#[test]
fn rejected_request_also_invalidates_previous_work() {
    let (_temp, predictor, mut engine) = fixture("delay");
    until(|| engine.status() == Status::Ready);
    engine.submit(&predictor, "", "", options(), 1).unwrap();
    thread::sleep(Duration::from_millis(20));
    assert!(
        engine
            .submit(&predictor, &"x".repeat(16_385), "", options(), 2)
            .is_err()
    );
    thread::sleep(Duration::from_millis(200));
    assert!(engine.poll().is_none());
    engine.submit(&predictor, "", "", options(), 2).unwrap();
    thread::sleep(Duration::from_millis(200));
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
                2
            )
            .is_err()
    );
    assert!(engine.poll().is_none());
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

#[test]
fn cli_errors_do_not_echo_input_and_once_requires_request() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let (temp, _predictor, mut engine) = fixture("ok");
    engine.shutdown();
    for input in [
        "private-malformed-text".to_string(),
        "private".repeat(10_000),
        String::new(),
    ] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_switchify-prediction-neural"))
            .arg("once")
            .arg("--baseline")
            .arg(temp.path().join("baseline.sqlite"))
            .arg("--bundle")
            .arg(temp.path())
            .arg("--worker")
            .arg(env!("CARGO_BIN_EXE_fake-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let _ = child.stdin.take().unwrap().write_all(input.as_bytes());
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("private"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("private"));
    }
}

#[test]
fn bare_worker_filename_resolves_in_callers_directory() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let (temp, _predictor, mut engine) = fixture("ok");
    engine.shutdown();
    let filename = std::path::Path::new(env!("CARGO_BIN_EXE_fake-worker"))
        .file_name()
        .unwrap();
    std::fs::copy(
        env!("CARGO_BIN_EXE_fake-worker"),
        temp.path().join(filename),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_switchify-prediction-neural"))
        .current_dir(temp.path())
        .arg("once")
        .arg("--baseline")
        .arg("baseline.sqlite")
        .arg("--bundle")
        .arg(".")
        .arg("--worker")
        .arg(filename)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"command":"predict","before":"","prefix":"","session":1,"min_chars":0}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("refined"));
}

#[test]
fn generation_is_async_normalized_and_latest_only() {
    let (_temp, _predictor, mut engine) = fixture("delay");
    until(|| engine.status() == Status::Ready);
    let start = Instant::now();
    let first = engine
        .generate("old. say", "he", 1, &["HELLO".into()], 3)
        .unwrap()
        .unwrap();
    assert!(start.elapsed() < Duration::from_millis(50));
    let latest = engine
        .generate("say", "cafe\u{301}", 2, &[], 3)
        .unwrap()
        .unwrap();
    assert_ne!(first, latest);
    let mut result = None;
    until(|| {
        result = engine.poll();
        result.is_some()
    });
    let result = result.unwrap();
    assert_eq!(result.request_id, latest);
    assert_eq!(result.words, ["café"]);
    engine.generate("", "", 2, &[], 3).unwrap();
    engine.reset();
    thread::sleep(Duration::from_millis(150));
    assert!(engine.poll().is_none());
    assert!(engine.generate("", "", 2, &[], 4).is_err());
}

#[test]
fn generation_rejects_invalid_workers_and_timeout_requires_retry() {
    for mode in ["foreign", "duplicate", "id", "stall", "crash"] {
        let (temp, _predictor, mut engine) = fixture(mode);
        until(|| engine.status() == Status::Ready);
        engine.generate("", "", 1, &[], 3).unwrap();
        until(|| matches!(engine.status(), Status::Unavailable(_)));
        assert!(engine.poll().is_none());
        assert_eq!(engine.generate("", "", 1, &[], 3).unwrap(), None);
        std::fs::write(temp.path().join("mode"), "delay").unwrap();
        engine.retry();
        until(|| engine.status() == Status::Ready);
        let id = engine
            .generate("", "he", 2, &["HELLO".into()], 3)
            .unwrap()
            .unwrap();
        let mut result = None;
        until(|| {
            result = engine.poll();
            result.is_some()
        });
        let result = result.unwrap();
        assert_eq!(result.request_id, id);
        assert_eq!(result.words, ["help", "helium"]);
    }
}

#[test]
fn generation_rejects_old_protocol_without_retry_loop() {
    let (_temp, _predictor, mut engine) = fixture("old");
    until(|| matches!(engine.status(), Status::Unavailable(_)));
    assert_eq!(engine.generate("", "", 1, &[], 3).unwrap(), None);
}
