//! Frozen source partitions. Text is normalized by exactly the predictor's tokenizer.
use crate::{Error, Result, digest, sentences};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Deserialize)]
struct Dialogue {
    conversation_id: String,
    instruction_id: String,
    utterances: Vec<Utterance>,
}
#[derive(Deserialize)]
struct Utterance {
    speaker: String,
    text: String,
}
#[derive(Deserialize)]
struct Policy {
    candidate_world_weight: usize,
    candidate_taskmaster_weight: usize,
    max_taskmaster_evaluation_sentences_per_domain: usize,
}
#[derive(Serialize)]
pub struct Prepared {
    pub protocol: String,
    pub policy_sha256: String,
    pub source_manifest_sha256: String,
    pub taskmaster_dialogues: BTreeMap<String, usize>,
    pub files: BTreeMap<String, FileInfo>,
}
#[derive(Serialize)]
pub struct FileInfo {
    pub sha256: String,
    pub sentences: usize,
}
fn normalized(text: &str) -> BTreeSet<String> {
    sentences(text).into_iter().map(|s| s.join(" ")).collect()
}
fn verified(path: &Path, expected: &str) -> Result<String> {
    let bytes = fs::read(path)?;
    if digest(&bytes) != expected {
        return Err(Error::Invalid("source checksum mismatch".into()));
    }
    String::from_utf8(bytes).map_err(|_| Error::Invalid("source must be UTF-8".into()))
}
fn domain(instruction: &str) -> Result<&'static str> {
    for (prefix, name) in [
        ("pizza-", "pizza"),
        ("auto-", "auto"),
        ("coffee-", "coffee"),
        ("restaurant-", "restaurant"),
        ("uber-", "ride"),
        ("movie-", "movie"),
    ] {
        if instruction.starts_with(prefix) {
            return Ok(name);
        }
    }
    Err(Error::Invalid("unknown Taskmaster domain".into()))
}
fn sampled(
    domains: &BTreeMap<String, BTreeSet<String>>,
    excluded: &BTreeSet<String>,
    limit: usize,
) -> Vec<String> {
    let mut selected = BTreeSet::new();
    for words in domains.values() {
        let mut words: Vec<_> = words.difference(excluded).cloned().collect();
        words.sort_by_cached_key(|s| (digest(s.as_bytes()), s.clone()));
        selected.extend(words.into_iter().take(limit));
    }
    selected.into_iter().collect()
}
/// Checksum verification, official conversation partitions, and exact normalized overlap removal.
pub fn prepare(
    data: &Path,
    output: &Path,
    manifest_text: &str,
    policy_text: &str,
) -> Result<Prepared> {
    let manifest: serde_json::Value = serde_json::from_str(manifest_text)?;
    let policy: Policy = serde_json::from_str(policy_text)?;
    if let Some(expected) = manifest["prepared"]["policy_sha256"].as_str()
        && digest(policy_text.as_bytes()) != expected
    {
        return Err(Error::Invalid("quality policy checksum mismatch".into()));
    }

    if policy.candidate_world_weight == 0
        || policy.candidate_taskmaster_weight == 0
        || policy.max_taskmaster_evaluation_sentences_per_domain == 0
    {
        return Err(Error::Invalid(
            "corpus weights and sample limit must be positive".into(),
        ));
    }
    let hash = |v: &serde_json::Value| -> Result<String> {
        v["sha256"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| Error::Invalid("missing source hash".into()))
    };
    let world = verified(&data.join("en.txt"), &hash(&manifest["corpus"])?)?;
    verified(
        &data.join("SOURCES.json"),
        &hash(&manifest["upstream_manifest"])?,
    )?;
    let mut tm_files = BTreeMap::new();
    for name in [
        "self-dialogs.json",
        "README.md",
        "train.csv",
        "dev.csv",
        "test.csv",
    ] {
        tm_files.insert(
            name,
            verified(
                &data.join("taskmaster").join(name),
                &hash(&manifest["taskmaster"]["files"][name])?,
            )?,
        );
    }
    let dialogues: Vec<Dialogue> = serde_json::from_str(&tm_files["self-dialogs.json"])?;
    let mut membership = BTreeMap::new();
    let mut group_counts = BTreeMap::new();
    for part in ["train", "dev", "test"] {
        let content = &tm_files[format!("{part}.csv").as_str()];
        let mut count = 0;
        for id in content
            .lines()
            .map(|l| l.split(',').next().unwrap_or("").trim())
            .filter(|s| !s.is_empty())
        {
            if membership.insert(id.to_string(), part).is_some() {
                return Err(Error::Invalid(
                    "overlapping or duplicate conversation IDs".into(),
                ));
            }
            count += 1;
        }
        group_counts.insert(part.to_string(), count);
    }
    let mut groups: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for dialogue in dialogues {
        if !seen.insert(dialogue.conversation_id.clone()) {
            return Err(Error::Invalid("duplicate source conversation".into()));
        }
        let part = membership
            .get(&dialogue.conversation_id)
            .ok_or_else(|| Error::Invalid("unpartitioned conversation".into()))?;
        let domain = domain(&dialogue.instruction_id)?;
        let values = groups
            .entry(part.to_string())
            .or_default()
            .entry(domain.to_string())
            .or_default();
        for utterance in dialogue.utterances {
            if utterance.speaker == "USER" {
                values.extend(normalized(&utterance.text));
            }
        }
    }
    if seen.len() != membership.len() {
        return Err(Error::Invalid(
            "partition references missing conversation".into(),
        ));
    }
    let mut world: Vec<_> = normalized(&world).into_iter().collect();
    world.sort_by_cached_key(|s| (digest(s.as_bytes()), s.clone()));
    let held = world.len().div_ceil(10);
    if world.len() <= held * 2 {
        return Err(Error::Invalid("not enough general sentences".into()));
    }
    let general_test = world[..held].to_vec(); // Retain v1's known regression test set.
    let general_dev = world[held..held * 2].to_vec();
    let world_train = world[held * 2..].to_vec();
    let reserved: BTreeSet<_> = general_test.iter().chain(&general_dev).cloned().collect();
    let tm_train: BTreeSet<_> = groups["train"]
        .values()
        .flat_map(|s| s.iter().cloned())
        .filter(|s| !reserved.contains(s))
        .collect();
    let mut exclude: BTreeSet<_> = world
        .iter()
        .cloned()
        .chain(tm_train.iter().cloned())
        .collect();
    let task_dev = sampled(
        &groups["dev"],
        &exclude,
        policy.max_taskmaster_evaluation_sentences_per_domain,
    );
    // All development text is excluded from test, not only the sampled subset.
    exclude.extend(groups["dev"].values().flat_map(|s| s.iter().cloned()));
    let task_test = sampled(
        &groups["test"],
        &exclude,
        policy.max_taskmaster_evaluation_sentences_per_domain,
    );
    let baseline = world_train.clone();
    let mut candidate = Vec::new();
    for _ in 0..policy.candidate_world_weight {
        candidate.extend(world_train.clone());
    }
    for _ in 0..policy.candidate_taskmaster_weight {
        candidate.extend(tm_train.iter().cloned());
    }
    let mut files = BTreeMap::new();
    fs::create_dir_all(output)?;
    for (name, lines) in [
        ("baseline.txt", baseline),
        ("candidate.txt", candidate),
        ("general-dev.txt", general_dev),
        ("general-test.txt", general_test),
        ("conversation-dev.txt", task_dev),
        ("conversation-test.txt", task_test),
    ] {
        if lines.is_empty() {
            return Err(Error::Invalid("empty prepared partition".into()));
        }
        let contents = lines.join("\n") + "\n";
        fs::write(output.join(name), &contents)?;
        files.insert(
            name.to_string(),
            FileInfo {
                sha256: digest(contents.as_bytes()),
                sentences: lines.len(),
            },
        );
    }
    if let Some(expected_files) = manifest["prepared"]["files"].as_object() {
        for (name, info) in &files {
            if expected_files.get(name).and_then(|v| v["sha256"].as_str())
                != Some(info.sha256.as_str())
            {
                return Err(Error::Invalid(
                    "prepared partition checksum mismatch".into(),
                ));
            }
        }
    }
    let result=Prepared{
        protocol:"quality-v1: WorldAlphabets hash-ranked 80/10/10 (v1 test retained); Taskmaster official conversation splits, USER only, unique normalized sentences; exclude all general dev/test sentences from task training; conversational dev/test excludes all WorldAlphabets and task training sentences; test also excludes all task dev sentences; sample by hash up to 100 sentences per domain; published candidate excludes held-out partitions".into(),
        policy_sha256:digest(policy_text.as_bytes()),source_manifest_sha256:digest(manifest_text.as_bytes()),taskmaster_dialogues:group_counts,files,
    };
    fs::write(
        output.join("partitions.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    Ok(result)
}
