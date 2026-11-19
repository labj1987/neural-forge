//! The command line on synthetic dumps (ported from scripts/test_cache_replay.py, test_command_line).

use std::path::PathBuf;
use std::process::Command;

use cache_replay::replay::{make_synthetic, MODES};
use serde_json::Value;

fn run(args: &[&str]) -> i32 {
    let out = Command::new(env!("CARGO_BIN_EXE_cache-replay")).args(args).output().expect("run cache-replay");
    out.status.code().unwrap_or(-1)
}

#[test]
fn command_line() {
    let root: PathBuf =
        [env!("CARGO_MANIFEST_DIR"), "..", "..", "target", "test-scratch", &format!("cache-replay-cli-{}", std::process::id())].iter().collect();
    let _ = std::fs::remove_dir_all(&root);
    let pan = root.join("pan");
    let still = root.join("still");
    make_synthetic(&pan, "pan").unwrap();
    make_synthetic(&still, "still").unwrap();
    let out = root.join("r.json");
    let cad = root.join("r.json.cad");
    let (pan_s, still_s, out_s, cad_s) = (pan.to_str().unwrap(), still.to_str().unwrap(), out.to_str().unwrap(), cad.to_str().unwrap());
    assert_eq!(run(&[pan_s, "--lags", "1", "2", "--json", out_s, "--quiet"]), 0);
    assert_eq!(run(&[still_s, "--self-check"]), 0);
    assert_eq!(run(&[pan_s, "--cadence", "2", "--json", cad_s]), 0);

    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let results = doc["results"].as_object().unwrap();
    let mut keys: Vec<&str> = results.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["1", "2"]);
    let mut modes: Vec<&str> = results["1"].as_object().unwrap().keys().map(String::as_str).collect();
    modes.sort_unstable();
    let mut want = MODES.to_vec();
    want.sort_unstable();
    assert_eq!(modes, want);
    let cad_doc: Value = serde_json::from_str(&std::fs::read_to_string(&cad).unwrap()).unwrap();
    let labels: Vec<&str> = cad_doc["runs"].as_array().unwrap().iter().map(|r| r["label"].as_str().unwrap()).collect();
    assert_eq!(labels, ["i2-base"]);
    std::fs::remove_dir_all(&root).unwrap();
}
