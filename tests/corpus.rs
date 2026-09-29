use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};
use switchify_prediction::{build, corpus, digest, evaluation};

fn fixture(root: &Path) -> (String, String) {
    fs::create_dir_all(root.join("taskmaster")).unwrap();
    let world = "world alpha. world beta. world gamma. world delta. world epsilon. world zeta. world eta. world theta. world iota. world kappa. world lambda. world mu. world nu. world xi. world omicron. world pi. world rho. world sigma. world tau. world omega.";
    fs::write(root.join("en.txt"), world).unwrap();
    fs::write(root.join("SOURCES.json"), "{}").unwrap();
    let (_, held) = evaluation::split(world);
    let dialogs = json!([
        {"conversation_id":"train", "instruction_id":"coffee-ordering-1","utterances":[
            {"speaker":"USER","text":format!("Please bring water. {}.",held[0])},
            {"speaker":"ASSISTANT","text":"assistant words must never enter the corpus"}]},
        {"conversation_id":"dev", "instruction_id":"coffee-ordering-1","utterances":[
            {"speaker":"USER","text":"Please bring water. Can you fix my bicycle?"}]},
        {"conversation_id":"test", "instruction_id":"coffee-ordering-1","utterances":[
            {"speaker":"USER","text":"Please bring water. Can you fix my bicycle? I want a table near the window."}]}
    ]);
    fs::write(
        root.join("taskmaster/self-dialogs.json"),
        dialogs.to_string(),
    )
    .unwrap();
    fs::write(root.join("taskmaster/README.md"), "test license fixture").unwrap();
    for part in ["train", "dev", "test"] {
        fs::write(
            root.join(format!("taskmaster/{part}.csv")),
            format!("{part},\n"),
        )
        .unwrap();
    }
    let mut files = serde_json::Map::new();
    for name in [
        "self-dialogs.json",
        "README.md",
        "train.csv",
        "dev.csv",
        "test.csv",
    ] {
        files.insert(
            name.to_string(),
            json!({"sha256":digest(&fs::read(root.join("taskmaster").join(name)).unwrap())}),
        );
    }
    let manifest=json!({"corpus":{"sha256":digest(world.as_bytes())},"upstream_manifest":{"sha256":digest(b"{}")},"taskmaster":{"files":files}}).to_string();
    let policy=json!({"candidate_world_weight":3,"candidate_taskmaster_weight":1,"max_taskmaster_evaluation_sentences_per_domain":100}).to_string();
    (manifest, policy)
}
fn lines(path: &Path) -> BTreeSet<String> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}
#[test]
fn official_splits_remove_exact_leakage_and_assistant_text() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let (manifest, policy) = fixture(dir.path());
    let prepared = corpus::prepare(dir.path(), &out, &manifest, &policy).unwrap();
    let training = lines(&out.join("candidate.txt"));
    assert!(training.contains("please bring water"));
    assert!(!training.iter().any(|s| s.contains("assistant")));
    let mut held = BTreeSet::new();
    for file in [
        "general-dev.txt",
        "general-test.txt",
        "conversation-dev.txt",
        "conversation-test.txt",
    ] {
        let set = lines(&out.join(file));
        assert!(set.is_disjoint(&training));
        assert!(set.is_disjoint(&held));
        held.extend(set);
    }
    assert_eq!(
        lines(&out.join("conversation-dev.txt")),
        BTreeSet::from(["can you fix my bicycle".into()])
    );
    assert_eq!(
        lines(&out.join("conversation-test.txt")),
        BTreeSet::from(["i want a table near the window".into()])
    );
    let again = corpus::prepare(dir.path(), &dir.path().join("again"), &manifest, &policy).unwrap();
    assert_eq!(
        serde_json::to_value(prepared).unwrap(),
        serde_json::to_value(again).unwrap()
    );
    let raw = fs::read_to_string(out.join("candidate.txt")).unwrap();
    let baseline = lines(&out.join("baseline.txt"));
    assert_eq!(
        raw.lines().filter(|s| baseline.contains(*s)).count(),
        baseline.len() * 3
    );
}
#[test]
fn corrupted_sources_and_overlapping_conversation_ids_fail() {
    let dir = tempfile::tempdir().unwrap();
    let (manifest, policy) = fixture(dir.path());
    fs::write(dir.path().join("taskmaster/dev.csv"), "train,\n").unwrap();
    assert!(corpus::prepare(dir.path(), &dir.path().join("out"), &manifest, &policy).is_err());
    let mut manifest: Value = serde_json::from_str(&manifest).unwrap();
    manifest["taskmaster"]["files"]["dev.csv"]["sha256"] = digest(b"train,\n").into();
    assert!(
        corpus::prepare(
            dir.path(),
            &dir.path().join("out"),
            &manifest.to_string(),
            &policy
        )
        .is_err()
    );
}
#[test]
fn external_scoring_reports_rank_sensitive_savings_and_empty_data_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("base.sqlite");
    build(&db, "watermelon watermelon watermelon", "test").unwrap();
    let report = evaluation::score_database(&db, "watermelon", "test".into()).unwrap();
    assert_eq!(report.accuracy[2].top1, 1.0);
    assert_eq!(report.selection_proxy.characters_without_prediction, 10);
    assert_eq!(
        report.selection_proxy.simulated_selections_with_prediction,
        3
    );
    assert!((report.selection_proxy.savings_fraction - 0.7).abs() < 1e-10);
    assert!(evaluation::score_database(&db, "123!", "test".into()).is_err());
}

#[test]
fn pinned_preparation_and_default_build_reject_changed_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let (manifest, policy) = fixture(dir.path());
    let mut manifest: Value = serde_json::from_str(&manifest).unwrap();
    manifest["prepared"] = json!({"policy_sha256":"wrong"});
    assert!(
        corpus::prepare(
            dir.path(),
            &dir.path().join("out"),
            &manifest.to_string(),
            &policy
        )
        .is_err()
    );
    manifest["prepared"] = json!({"policy_sha256":digest(policy.as_bytes()),"files":{}});
    assert!(
        corpus::prepare(
            dir.path(),
            &dir.path().join("out"),
            &manifest.to_string(),
            &policy
        )
        .is_err()
    );
    fs::create_dir_all(dir.path().join("data/prepared")).unwrap();
    fs::write(
        dir.path().join("data/prepared/candidate.txt"),
        "tampered training input",
    )
    .unwrap();
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_switchify-prediction"))
        .current_dir(dir.path())
        .args(["build", "--output", "bad.sqlite"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!dir.path().join("bad.sqlite").exists());
}
