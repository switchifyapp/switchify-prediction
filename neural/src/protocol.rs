//! Bounded private child-process transport. No sockets and no text diagnostics.
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Read, Write};

pub const VERSION: u32 = 1;
pub const MAX_FRAME: usize = 65_536;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub id: u64,
    pub session: u64,
    pub before: String,
    pub candidates: Vec<String>,
    pub limit: usize,
}

#[derive(Serialize, Deserialize)]
pub enum Command {
    Predict(Query),
    Reset,
}

#[derive(Serialize, Deserialize)]
pub enum Reply {
    Ready {
        version: u32,
        accelerated: bool,
    },
    Ranked {
        id: u64,
        words: Vec<String>,
        cache_hit: bool,
    },
    Reset,
    Failed,
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid worker frame")
}

pub fn write_frame(mut out: impl Write, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|_| invalid())?;
    if bytes.is_empty() || bytes.len() > MAX_FRAME {
        return Err(invalid());
    }
    out.write_all(&(bytes.len() as u32).to_le_bytes())?;
    out.write_all(&bytes)?;
    out.flush()
}

pub fn read_frame<T: DeserializeOwned>(mut input: impl Read) -> io::Result<T> {
    let mut header = [0; 4];
    input.read_exact(&mut header)?;
    let length = u32::from_le_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(invalid());
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes)?;
    serde_json::from_slice(&bytes).map_err(|_| invalid())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_frames_without_echoing_input() {
        for bytes in [vec![], vec![255; 4], vec![0; 4], vec![1, 0, 0, 0, b'!']] {
            assert!(read_frame::<Reply>(&bytes[..]).is_err());
        }
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &Command::Reset).unwrap();
        assert!(matches!(
            read_frame::<Command>(&bytes[..]).unwrap(),
            Command::Reset
        ));
    }
}
