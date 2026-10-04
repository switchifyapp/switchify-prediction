//! Dedicated offline inference process. All errors are deliberately text-free.
mod generation;
mod model;
use std::{io, path::PathBuf};
use switchify_prediction_neural::{
    bundle,
    protocol::{self, Command, Reply},
};

#[cfg(all(
    feature = "accelerated",
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c"
    ))
))]
compile_error!("accelerated workers require x86_64 +avx2,+fma,+f16c");

fn run() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(args.len() == 1, "configuration");
    let bundle = bundle::load(&PathBuf::from(&args[0]))?;
    let mut model = model::Model::load(bundle)?;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    protocol::write_frame(
        &mut output,
        &Reply::Ready {
            version: protocol::VERSION,
            accelerated: cfg!(feature = "accelerated"),
        },
    )?;
    loop {
        let command = match protocol::read_frame(&mut input) {
            Ok(command) => command,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(_) => anyhow::bail!("protocol"),
        };
        let reply = match command {
            Command::Reset => {
                model.reset();
                Reply::Reset
            }
            Command::Generate(query) => {
                anyhow::ensure!(query.valid(), "query");
                let (words, cache_hit) = model.generate(&query)?;
                anyhow::ensure!(query.accepts(&words), "generation");
                Reply::Generated {
                    id: query.id,
                    words,
                    cache_hit,
                }
            }
            Command::Predict(query) => {
                anyhow::ensure!(
                    query.before.len() <= 16_384
                        && query.candidates.len() <= 8
                        && query.limit <= 5
                        && query
                            .candidates
                            .iter()
                            .all(|s| !s.is_empty() && s.len() <= 128),
                    "query"
                );
                let (words, cache_hit) =
                    model.rank(query.session, &query.before, &query.candidates, query.limit)?;
                Reply::Ranked {
                    id: query.id,
                    words,
                    cache_hit,
                }
            }
        };
        protocol::write_frame(&mut output, &reply)?;
    }
}
fn main() {
    // Panic messages may include third-party input values. Never send them to logs.
    std::panic::set_hook(Box::new(|_| {}));
    if run().is_err() {
        std::process::exit(1);
    }
}
