use std::{path::PathBuf, thread, time::Duration};
use switchify_prediction::{Options, Predictor};
use switchify_prediction_neural::{Config, Refiner, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        return Err("usage: refine BASELINE BUNDLE WORKER".into());
    }
    let predictor = Predictor::open(&args[0], None)?;
    let mut engine = Refiner::new(Config {
        bundle: args[1].clone(),
        portable_worker: args[2].clone(),
        accelerated_worker: None,
        threads: 4,
    })?;
    // Synthetic example only. A UI can submit while Loading and still gets immediate results.
    let immediate = engine.submit(&predictor, "please send", "th", Options::default(), 1)?;
    println!(
        "request {}, {} immediate words",
        immediate.request_id,
        immediate.words.len()
    );
    loop {
        if let Some(refined) = engine.poll() {
            println!(
                "request {}, {} refined words",
                refined.request_id,
                refined.words.len()
            );
            break;
        }
        if !immediate.refinement_requested || matches!(engine.status(), Status::Unavailable(_)) {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    engine.reset();
    engine.shutdown();
    Ok(())
}
