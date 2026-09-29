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
    /// Normalize a UTF-8 corpus using exactly the predictor's sentence tokenizer.
    Normalize {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Prepare checksum-verified training and frozen development/test partitions.
    Prepare {
        #[arg(long, default_value = "data")]
        data: PathBuf,
        #[arg(long, default_value = "data/prepared")]
        output: PathBuf,
    },
    /// Score a prebuilt database on an explicit held-out UTF-8 text file.
    Score {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        hardware: Option<String>,
    },
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
fn input(path: Option<PathBuf>, prepared: bool) -> Result<(String, String)> {
    if let Some(path) = path {
        let text = fs::read_to_string(path)?;
        let provenance = serde_json::json!({"source":"user-supplied UTF-8 text", "sha256":digest(text.as_bytes()), "language":"en", "order":3}).to_string();
        return Ok((text, provenance));
    }
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../source-manifest.json"))?;
    let text = fs::read_to_string(if prepared {
        "data/prepared/candidate.txt"
    } else {
        "data/en.txt"
    })?;
    let expected = if prepared {
        &manifest["prepared"]["files"]["candidate.txt"]
    } else {
        &manifest["corpus"]
    };
    if Some(digest(text.as_bytes()).as_str()) != expected["sha256"].as_str() {
        return Err(switchify_prediction::Error::Invalid(
            "pinned corpus checksum mismatch; run scripts/fetch_corpus.py and the prepare command"
                .into(),
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
        Command::Normalize { input, output } => {
            let text = fs::read_to_string(input)?;
            let normalized = switchify_prediction::sentences(&text)
                .into_iter()
                .map(|words| words.join(" "))
                .collect::<Vec<_>>()
                .join("\n");
            fs::write(output, normalized + "\n")?;
            Ok(())
        }
        Command::Prepare { data, output } => print(switchify_prediction::corpus::prepare(
            &data,
            &output,
            include_str!("../source-manifest.json"),
            include_str!("../quality-policy.json"),
        )?),
        Command::Score {
            baseline,
            input,
            hardware,
        } => print(evaluation::score_database(
            &baseline,
            &fs::read_to_string(input)?,
            hardware
                .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)),
        )?),
        Command::Build {
            output,
            input: path,
        } => {
            let (text, provenance) = input(path, true)?;
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
            let (text, _) = input(path, false)?;
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
