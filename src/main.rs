use clap::{Parser, Subcommand};
use std::{
    fs,
    io::{self, Read},
    path::PathBuf,
};
use switchify_prediction::{Options, Predictor, Result, build, digest, evaluation, validate};

#[derive(Parser)]
#[command(version, about = "Offline English n-gram word prediction")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Build a new baseline. Defaults to the verified pinned corpus.
    Build {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        input: Option<PathBuf>,
    },
    Predict {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        personal: Option<PathBuf>,
        #[arg(long, default_value = "")]
        before: String,
        #[arg(long, default_value = "")]
        prefix: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, default_value_t = 2)]
        min_chars: usize,
    },
    Import {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        personal: PathBuf,
        #[arg(long)]
        input: PathBuf,
    },
    /// Learn a non-overlapping completed segment supplied on stdin.
    Learn {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        personal: PathBuf,
    },
    ResetPersonal {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        personal: PathBuf,
    },
    Validate {
        #[arg(long)]
        database: PathBuf,
    },
    Evaluate {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        hardware: Option<String>,
    },
}
fn input(path: Option<PathBuf>) -> Result<(String, String)> {
    if let Some(path) = path {
        let text = fs::read_to_string(path)?;
        let provenance = serde_json::json!({"source":"user-supplied UTF-8 text", "sha256":digest(text.as_bytes()), "language":"en", "order":3}).to_string();
        return Ok((text, provenance));
    }
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../source-manifest.json"))?;
    let text = fs::read_to_string("data/en.txt")?;
    if Some(digest(text.as_bytes()).as_str()) != manifest["corpus"]["sha256"].as_str() {
        return Err(switchify_prediction::Error::Invalid(
            "pinned corpus checksum mismatch; run scripts/fetch_corpus.py".into(),
        ));
    }
    Ok((text, manifest.to_string()))
}
fn print(value: impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Build {
            output,
            input: path,
        } => {
            let (text, provenance) = input(path)?;
            print(build(&output, &text, &provenance)?)
        }
        Command::Predict {
            baseline,
            personal,
            before,
            prefix,
            limit,
            min_chars,
        } => {
            let predictor = Predictor::open(&baseline, personal.as_deref())?;
            print(predictor.predict(
                &before,
                &prefix,
                Options {
                    limit,
                    min_chars,
                    unigram_only: false,
                },
            ))
        }
        Command::Import {
            baseline,
            personal,
            input,
        } => {
            let imported = Predictor::open(&baseline, Some(&personal))?.import(&input)?;
            print(serde_json::json!({"imported":imported}))
        }
        Command::Learn { baseline, personal } => {
            let mut text = String::new();
            io::stdin().read_to_string(&mut text)?;
            Predictor::open(&baseline, Some(&personal))?.learn(&text)?;
            print(serde_json::json!({"learned":true}))
        }
        Command::ResetPersonal { baseline, personal } => {
            Predictor::open(&baseline, Some(&personal))?.reset_personal()?;
            print(serde_json::json!({"reset":true}))
        }
        Command::Validate { database } => print(validate(&database)?),
        Command::Evaluate {
            input: path,
            hardware,
        } => {
            let (text, _) = input(path)?;
            print(evaluation::evaluate(
                &text,
                hardware.unwrap_or_else(|| {
                    format!(
                        "{}-{}; CPU unspecified",
                        std::env::consts::OS,
                        std::env::consts::ARCH
                    )
                }),
            )?)
        }
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
