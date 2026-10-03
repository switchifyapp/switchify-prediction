use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
use switchify_prediction::{Options, Predictor};
use switchify_prediction_neural::{Config, Refiner, Status, bundle};

#[derive(Parser)]
#[command(
    version,
    about = "Offline SmolLM2 companion. User text is accepted only on stdin."
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    Validate {
        #[arg(long)]
        bundle: PathBuf,
    },
    Once(Run),
    Stream(Run),
}
#[derive(clap::Args)]
struct Run {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    personal: Option<PathBuf>,
    #[arg(long)]
    bundle: PathBuf,
    #[arg(long)]
    worker: PathBuf,
    #[arg(long)]
    accelerated_worker: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    threads: usize,
}
#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Input {
    Predict {
        before: String,
        prefix: String,
        session: u64,
        #[serde(default = "five")]
        limit: usize,
        #[serde(default = "two")]
        min_chars: usize,
        #[serde(default)]
        unigram_only: bool,
    },
    Reset,
    Retry,
}
fn five() -> usize {
    5
}
fn two() -> usize {
    2
}
fn emit(value: serde_json::Value) -> Result<(), ()> {
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &value).map_err(|_| ())?;
    writeln!(out).and_then(|_| out.flush()).map_err(|_| ())
}
fn input() -> mpsc::Receiver<Result<Input, ()>> {
    let (tx, rx) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut stdin = io::stdin().lock();
        loop {
            let mut bytes = Vec::new();
            let read = (&mut stdin).take(65_537).read_until(b'\n', &mut bytes);
            if matches!(read, Ok(0)) {
                break;
            }
            let item = if read.is_err() || bytes.len() > 65_536 {
                Err(())
            } else {
                serde_json::from_slice(&bytes).map_err(|_| ())
            };
            let failed = item.is_err();
            if tx.send(item).is_err() || failed {
                break;
            }
        }
    });
    rx
}
fn run(args: Run, once: bool) -> Result<(), ()> {
    let predictor = Predictor::open(&args.baseline, args.personal.as_deref()).map_err(|_| ())?;
    let mut engine = Refiner::new(Config {
        bundle: args.bundle,
        portable_worker: args.worker,
        accelerated_worker: args.accelerated_worker,
        threads: args.threads,
    })
    .map_err(|_| ())?;
    while engine.status() == Status::Loading {
        thread::sleep(Duration::from_millis(5));
    }
    if engine.status() != Status::Ready {
        return Err(());
    }
    emit(json!({"type":"ready", "capabilities":engine.capabilities()}))?;
    let rx = input();
    let mut outstanding = false;
    let mut eof = false;
    let mut predicted = false;
    let mut started = Instant::now();
    let mut last_status = Status::Ready;
    loop {
        if let Some(result) = engine.poll() {
            emit(
                json!({"type":"refined", "result":result, "elapsed_ms":started.elapsed().as_secs_f64()*1000.}),
            )?;
            outstanding = false;
        }
        let status = engine.status();
        if status != last_status {
            emit(json!({"type":"status", "status":status}))?;
            last_status = status;
            if matches!(status, Status::Unavailable(_)) {
                outstanding = false;
            }
        }
        if eof && !outstanding {
            return if once && !predicted { Err(()) } else { Ok(()) };
        }
        if eof {
            thread::sleep(Duration::from_millis(1));
            continue;
        }
        match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(Ok(Input::Predict {
                before,
                prefix,
                session,
                limit,
                min_chars,
                unigram_only,
            })) => {
                started = Instant::now();
                predicted = true;
                let result = engine
                    .submit(
                        &predictor,
                        &before,
                        &prefix,
                        Options {
                            limit,
                            min_chars,
                            unigram_only,
                        },
                        session,
                    )
                    .map_err(|_| ())?;
                outstanding = result.refinement_requested;
                emit(
                    json!({"type":"immediate", "result":result, "elapsed_ms":started.elapsed().as_secs_f64()*1000.}),
                )?;
                if once {
                    eof = true;
                }
            }
            Ok(Ok(Input::Reset)) => {
                engine.reset();
                outstanding = false;
                emit(json!({"type":"reset"}))?;
            }
            Ok(Ok(Input::Retry)) => engine.retry(),
            Ok(Err(())) => return Err(()),
            Err(mpsc::RecvTimeoutError::Disconnected) => eof = true,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let result = match Args::parse().command {
        Commands::Validate { bundle: path } => bundle::load(&path)
            .map(|_| ())
            .map_err(|_| ())
            .and_then(|_| emit(json!({"valid":true,"model_id":bundle::MODEL_ID}))),
        Commands::Once(args) => run(args, true),
        Commands::Stream(args) => run(args, false),
    };
    if result.is_err() {
        eprintln!("neural command failed; check configuration, bundle and input schema");
        std::process::exit(1);
    }
}
