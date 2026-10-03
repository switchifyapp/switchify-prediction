//! Optional asynchronous refinement. The caller owns the statistical predictor.
//! Text is held in bounded memory and sent only through private child-process pipes.
pub mod bundle;
mod process;
pub mod protocol;

use protocol::Query;
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};
use switchify_prediction::{Options, Predictor, normalize, sentences};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid neural configuration")]
    Config,
    #[error("model bundle is missing, corrupt or incompatible")]
    Bundle,
    #[error("input exceeds neural limits")]
    Input,
}

#[derive(Clone)]
pub struct Config {
    pub bundle: PathBuf,
    pub portable_worker: PathBuf,
    pub accelerated_worker: Option<PathBuf>,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Failure {
    Load,
    Timeout,
    Worker,
    Protocol,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Status {
    Loading,
    Ready,
    Unavailable(Failure),
    Stopped,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Capabilities {
    pub accelerated: bool,
    pub threads: usize,
    pub context_tokens: usize,
    pub shortlist: usize,
    pub deadline_ms: u64,
}
#[derive(Serialize)]
pub struct Immediate {
    pub request_id: u64,
    pub words: Vec<String>,
    pub status: Status,
    pub refinement_requested: bool,
}
#[derive(Serialize)]
pub struct Refined {
    pub request_id: u64,
    pub words: Vec<String>,
    pub cache_hit: bool,
}

struct Shared {
    latest: u64,
    session: u64,
    pending: Option<Query>,
    result: Option<Refined>,
    status: Status,
    reset: bool,
    retry: bool,
    stop: bool,
}
type State = Arc<(Mutex<Shared>, Condvar)>;

/// Single controller, one in-flight query and one replaceable pending query.
/// Poll and submit on the application's controller thread. No callback races.
pub struct Refiner {
    state: State,
    actor: Option<JoinHandle<()>>,
    capabilities: Capabilities,
}

pub fn accelerated_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Normalize only the current sentence, matching the established word tokenizer.
pub fn effective_context(before: &str) -> String {
    sentences(
        before
            .rsplit(['.', '!', '?', '\n', '\r'])
            .next()
            .unwrap_or_default(),
    )
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" ")
}

impl Refiner {
    pub fn new(mut config: Config) -> Result<Self> {
        config.bundle = config.bundle.canonicalize().map_err(|_| Error::Config)?;
        config.portable_worker = config
            .portable_worker
            .canonicalize()
            .map_err(|_| Error::Config)?;
        config.accelerated_worker = config
            .accelerated_worker
            .map(|p| p.canonicalize())
            .transpose()
            .map_err(|_| Error::Config)?;
        if !(1..=4).contains(&config.threads)
            || !config.bundle.is_dir()
            || !config.portable_worker.is_file()
            || config
                .accelerated_worker
                .as_ref()
                .is_some_and(|p| !p.is_file())
        {
            return Err(Error::Config);
        }
        let accelerated = accelerated_supported() && config.accelerated_worker.is_some();
        let capabilities = Capabilities {
            accelerated,
            threads: config.threads,
            context_tokens: 64,
            shortlist: 8,
            deadline_ms: 500,
        };
        let state = Arc::new((
            Mutex::new(Shared {
                latest: 0,
                session: 0,
                pending: None,
                result: None,
                status: Status::Loading,
                reset: false,
                retry: true,
                stop: false,
            }),
            Condvar::new(),
        ));
        let cloned = Arc::clone(&state);
        let actor = thread::Builder::new()
            .name("prediction-refiner".into())
            .spawn(move || process::run(config, accelerated, cloned))
            .map_err(|_| Error::Config)?;
        Ok(Self {
            state,
            actor: Some(actor),
            capabilities,
        })
    }

    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }
    pub fn status(&self) -> Status {
        self.state.0.lock().unwrap().status
    }

    /// Capture both immediate results and the shortlist from the same caller snapshot.
    /// Limits above five are rejected. Context is capped at 16 KiB, prefix at 256 bytes.
    pub fn submit(
        &mut self,
        predictor: &Predictor,
        before: &str,
        prefix: &str,
        options: Options,
        session: u64,
    ) -> Result<Immediate> {
        if before.len() > 16_384 || prefix.len() > 256 || options.limit > 5 {
            self.reset();
            return Err(Error::Input);
        }
        let candidates: Vec<String> = predictor
            .predict(
                before,
                prefix,
                Options {
                    limit: 8,
                    ..options
                },
            )
            .into_iter()
            .map(|s| s.word)
            .collect();
        let words: Vec<String> = candidates.iter().take(options.limit).cloned().collect();
        let valid = candidates
            .iter()
            .all(|w| w.len() <= 128 && normalize(w) == *w && sentences(w) == vec![vec![w.clone()]]);
        let mut shared = self.state.0.lock().unwrap();
        shared.latest = shared.latest.checked_add(1).ok_or(Error::Input)?;
        if shared.session != session {
            shared.reset = true;
            shared.session = session;
        }
        shared.result = None;
        shared.pending = None;
        let refinement_requested = valid
            && options.limit > 0
            && !options.unigram_only
            && !candidates.is_empty()
            && matches!(shared.status, Status::Loading | Status::Ready);
        if refinement_requested {
            shared.pending = Some(Query {
                id: shared.latest,
                session,
                before: effective_context(before),
                candidates,
                limit: options.limit,
            });
        }
        let result = Immediate {
            request_id: shared.latest,
            words,
            status: shared.status,
            refinement_requested,
        };
        self.state.1.notify_one();
        Ok(result)
    }

    /// A subsequent submit/reset invalidates any previously unconsumed result.
    pub fn poll(&mut self) -> Option<Refined> {
        self.state.0.lock().unwrap().result.take()
    }

    /// Clear queued text and worker context. In-flight replies are discarded.
    pub fn reset(&mut self) {
        let mut shared = self.state.0.lock().unwrap();
        shared.pending = None;
        shared.result = None;
        shared.reset = true;
        shared.latest = shared.latest.saturating_add(1);
        self.state.1.notify_one();
    }

    /// Explicitly reload after failure. Never retries automatically in the background.
    pub fn retry(&mut self) {
        let mut shared = self.state.0.lock().unwrap();
        if matches!(shared.status, Status::Unavailable(_)) {
            shared.status = Status::Loading;
            shared.retry = true;
            self.state.1.notify_one();
        }
    }

    pub fn shutdown(&mut self) {
        {
            let mut shared = self.state.0.lock().unwrap();
            shared.stop = true;
            shared.pending = None;
            shared.result = None;
            self.state.1.notify_one();
        }
        if let Some(actor) = self.actor.take() {
            let _ = actor.join();
        }
    }
}
impl Drop for Refiner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn pause(state: &State) {
    let shared = state.0.lock().unwrap();
    let _ = state
        .1
        .wait_timeout(shared, Duration::from_millis(5))
        .unwrap();
}
