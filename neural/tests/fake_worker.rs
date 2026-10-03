use std::{
    io::{self, Write},
    path::PathBuf,
    thread,
    time::Duration,
};
use switchify_prediction_neural::protocol::{self, Command, Reply};
fn main() {
    let mode =
        std::fs::read_to_string(PathBuf::from(std::env::args_os().nth(1).unwrap()).join("mode"))
            .unwrap();
    if mode == "load-stall" {
        thread::sleep(Duration::from_secs(30));
        return;
    }
    let mut out = io::stdout().lock();
    protocol::write_frame(
        &mut out,
        &Reply::Ready {
            version: protocol::VERSION,
            accelerated: false,
        },
    )
    .unwrap();
    if mode == "read-stall" {
        thread::sleep(Duration::from_secs(30));
        return;
    }
    let mut input = io::stdin().lock();
    while let Ok(command) = protocol::read_frame(&mut input) {
        let reply = match command {
            Command::Reset => Reply::Reset,
            Command::Predict(mut query) => {
                match mode.as_str() {
                    "crash" => return,
                    "stall" => thread::sleep(Duration::from_secs(30)),
                    "delay" => thread::sleep(Duration::from_millis(100)),
                    "malformed" => {
                        out.write_all(&[1, 0, 0, 0, b'!']).unwrap();
                        out.flush().unwrap();
                        return;
                    }
                    "oversized" => {
                        out.write_all(&[255; 4]).unwrap();
                        out.flush().unwrap();
                        return;
                    }
                    "id" => query.id += 1,
                    "foreign" => query.candidates = vec!["not-in-shortlist".into()],
                    _ => {}
                }
                query.candidates.reverse();
                Reply::Ranked {
                    id: query.id,
                    words: query.candidates.into_iter().take(query.limit).collect(),
                    cache_hit: false,
                }
            }
        };
        if protocol::write_frame(&mut out, &reply).is_err() {
            break;
        }
    }
}
