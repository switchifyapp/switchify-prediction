use crate::{
    Config, Failure, REPLY_DEADLINE_MS, Refined, State, Status, pause,
    protocol::{self, Command, Reply},
};
use std::{
    io,
    process::{Child, Command as Spawn, Stdio},
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Worker {
    child: Child,
    commands: Option<SyncSender<Command>>,
    writer: Option<JoinHandle<()>>,
    replies: Receiver<io::Result<Reply>>,
    reader: Option<JoinHandle<()>>,
}
impl Worker {
    fn start(config: &Config, accelerated: bool) -> Result<Self, Failure> {
        let path = if accelerated {
            config.accelerated_worker.as_ref().unwrap()
        } else {
            &config.portable_worker
        };
        let mut command = Spawn::new(path);
        command
            .arg(&config.bundle)
            .env("RAYON_NUM_THREADS", config.threads.to_string())
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().map_err(|_| Failure::Load)?;
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        // The reader never buffers a stream of unsolicited replies.
        let (tx, replies) = mpsc::sync_channel(1);
        let reader = thread::spawn(move || {
            loop {
                let result = protocol::read_frame(&mut output);
                let failed = result.is_err();
                if tx.try_send(result).is_err() || failed {
                    break;
                }
            }
        });
        let (commands, rx) = mpsc::sync_channel(1);
        let writer = thread::spawn(move || {
            while let Ok(command) = rx.recv() {
                if protocol::write_frame(&mut input, &command).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            commands: Some(commands),
            writer: Some(writer),
            replies,
            reader: Some(reader),
        })
    }
    fn receive(&self, state: &State, timeout: Duration) -> Result<Reply, Failure> {
        let start = Instant::now();
        loop {
            if state.0.lock().unwrap().stop {
                return Err(Failure::Worker);
            }
            match self.replies.recv_timeout(Duration::from_millis(5)) {
                // A descheduled controller may wake with a reply already queued.
                // Do not accept it after the request's deadline has elapsed.
                Ok(Ok(reply)) => {
                    return if start.elapsed() < timeout {
                        Ok(reply)
                    } else {
                        Err(Failure::Timeout)
                    };
                }
                Ok(Err(_)) => return Err(Failure::Protocol),
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(Failure::Worker),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if start.elapsed() >= timeout {
                return Err(Failure::Timeout);
            }
        }
    }
    fn send(&mut self, command: Command) -> Result<(), Failure> {
        self.commands
            .as_ref()
            .unwrap()
            .try_send(command)
            .map_err(|_| Failure::Worker)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.commands.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

pub(super) fn run(config: Config, accelerated: bool, state: State) {
    let mut worker: Option<Worker> = None;
    loop {
        let (stop, retry, reset, query) = {
            let mut s = state.0.lock().unwrap();
            (
                s.stop,
                std::mem::take(&mut s.retry),
                std::mem::take(&mut s.reset),
                s.pending.take(),
            )
        };
        if stop {
            break;
        }
        let result = (|| {
            if retry {
                worker = Some(Worker::start(&config, accelerated)?);
                match worker
                    .as_ref()
                    .unwrap()
                    .receive(&state, Duration::from_secs(30))?
                {
                    Reply::Ready {
                        version: protocol::VERSION,
                        accelerated: actual,
                    } if actual == accelerated => {
                        state.0.lock().unwrap().status = Status::Ready;
                    }
                    _ => return Err(Failure::Load),
                }
            }
            if let Some(w) = &mut worker {
                if reset {
                    w.send(Command::Reset)?;
                    if !matches!(
                        w.receive(&state, Duration::from_millis(REPLY_DEADLINE_MS))?,
                        Reply::Reset
                    ) {
                        return Err(Failure::Protocol);
                    }
                }
                if let Some(query) = query {
                    let id = match &query {
                        Command::Predict(q) => q.id,
                        Command::Generate(q) => q.id,
                        Command::Reset => return Err(Failure::Protocol),
                    };
                    if state.0.lock().unwrap().latest != id {
                        return Ok(());
                    }
                    w.send(query.clone())?;
                    let reply = w.receive(&state, Duration::from_millis(REPLY_DEADLINE_MS))?;
                    let (words, cache_hit) = match (query, reply) {
                        (
                            Command::Predict(q),
                            Reply::Ranked {
                                id: actual,
                                words,
                                cache_hit,
                            },
                        ) if actual == id
                            && words.len() == q.limit.min(q.candidates.len())
                            && words.iter().all(|w| q.candidates.contains(w))
                            && words
                                .iter()
                                .collect::<std::collections::BTreeSet<_>>()
                                .len()
                                == words.len() =>
                        {
                            (words, cache_hit)
                        }
                        (
                            Command::Generate(q),
                            Reply::Generated {
                                id: actual,
                                words,
                                cache_hit,
                            },
                        ) if actual == id && q.accepts(&words) => (words, cache_hit),
                        _ => return Err(Failure::Protocol),
                    };
                    let mut s = state.0.lock().unwrap();
                    if s.latest == id && !s.reset && !s.stop {
                        s.result = Some(Refined {
                            request_id: id,
                            words,
                            cache_hit,
                        });
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            worker = None;
            let mut s = state.0.lock().unwrap();
            s.status = Status::Unavailable(error);
            s.pending = None;
            s.result = None;
        }
        pause(&state);
    }
    drop(worker);
    state.0.lock().unwrap().status = Status::Stopped;
}
