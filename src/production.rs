//! Pinned, evaluated English model. Corpus rights are distinct from the MIT code licence.
use crate::{Error, Result, Validation, build_expected, digest, readonly, validate_connection};
use serde_json::Value;
use std::path::Path;

pub const MANIFEST: &str = include_str!("../production-model.json");
pub const TRAINING_PATH: &str = "data/aac-oanc/prepared/candidate.txt";

/// Build the shipped model only from the exact evaluated training text.
/// Both text and logical model fingerprints are checked before publication.
pub fn build_english(path: &Path, text: &str) -> Result<Validation> {
    let manifest: Value = serde_json::from_str(MANIFEST)?;
    if Some(digest(text.as_bytes()).as_str()) != manifest["training_sha256"].as_str() {
        return Err(Error::Invalid(
            "production training checksum mismatch; run scripts/aac_experiment.py --prepare-only"
                .into(),
        ));
    }
    let expected = manifest["logical_sha256"]
        .as_str()
        .ok_or_else(|| Error::Invalid("production manifest has no model fingerprint".into()))?;
    build_expected(path, text, MANIFEST, Some(expected))
}

/// Check the shipped model identity and embedded provenance, as well as schema/integrity.
/// Distribution file checksums should also be verified before opening a download.
pub fn validate_english(path: &Path) -> Result<Validation> {
    let mut connection = readonly(path)?;
    let db = connection.transaction()?;
    let result = validate_connection(&db)?;
    let manifest: Value = serde_json::from_str(MANIFEST)?;
    let provenance: String = db.query_row(
        "SELECT value FROM metadata WHERE key='provenance'",
        [],
        |r| r.get(0),
    )?;
    let provenance: Value = serde_json::from_str(&provenance)?;
    if result.kind != "baseline"
        || Some(result.logical_sha256.as_str()) != manifest["logical_sha256"].as_str()
        || provenance != manifest
    {
        return Err(Error::Invalid(
            "database is not the pinned production model".into(),
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_fingerprint_never_publishes_and_custom_models_fail_identity_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.sqlite");
        assert!(build_expected(&path, "hello water", MANIFEST, Some("wrong")).is_err());
        assert!(!path.exists());
        crate::build(&path, "hello water", MANIFEST).unwrap();
        assert!(validate_english(&path).is_err());
        assert!(crate::validate(&path).is_ok());
    }
}
