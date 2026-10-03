//! Compiled compatibility pins, independent of claims in a downloaded manifest.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};

pub const MODEL_ID: &str = "smollm2-135m-q8-v1";
pub const MANIFEST: &str = include_str!("../model-bundle.json");

pub struct Bundle {
    pub weights: Vec<u8>,
    pub tokenizer: Vec<u8>,
}

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path).map_err(|_| Error::Bundle)?;
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Bundle)?;
    if bytes.len() as u64 > limit {
        return Err(Error::Bundle);
    }
    Ok(bytes)
}

pub fn load(path: &Path) -> Result<Bundle> {
    let manifest = bounded_read(&path.join("model-bundle.json"), 16_384)?;
    let actual: serde_json::Value = serde_json::from_slice(&manifest).map_err(|_| Error::Bundle)?;
    let expected: serde_json::Value = serde_json::from_str(MANIFEST).map_err(|_| Error::Bundle)?;
    if actual != expected {
        return Err(Error::Bundle);
    }
    let mut weights = None;
    let mut tokenizer = None;
    for (name, pin) in expected["files"].as_object().ok_or(Error::Bundle)? {
        let file = path.join(name);
        let size = pin["bytes"].as_u64().ok_or(Error::Bundle)?;
        if fs::metadata(&file).map_err(|_| Error::Bundle)?.len() != size {
            return Err(Error::Bundle);
        }
        let bytes = bounded_read(&file, size)?;
        if bytes.len() as u64 != size
            || format!("{:x}", Sha256::digest(&bytes))
                != pin["sha256"].as_str().ok_or(Error::Bundle)?
        {
            return Err(Error::Bundle);
        }
        match name.as_str() {
            "model.gguf" => weights = Some(bytes),
            "tokenizer.json" => tokenizer = Some(bytes),
            _ => {}
        }
    }
    Ok(Bundle {
        weights: weights.ok_or(Error::Bundle)?,
        tokenizer: tokenizer.ok_or(Error::Bundle)?,
    })
}
