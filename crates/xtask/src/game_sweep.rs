//! `cargo xtask game-sweep`: the pre-release game sweep (game_sweep/help.txt has what it does and its options; `--help`
//! prints it). The work is done by crates/supervisor/examples/game_sweep.rs: this builds it (release) and becomes it,
//! with the arguments and the working directory it was given; `--self-test` runs the example's own.
//!
//! It replaced scripts/game-sweep.sh: the same output and exit codes.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use crate::sh;

pub fn main(args: &[String]) -> ExitCode {
    let root = crate::root();
    let first = args.first().map(String::as_str).unwrap_or("");
    if first == "--self-test" {
        let ok = Command::new("cargo")
            .args(["run", "-q", "-p", "neural-forge-supervisor", "--example", "game-sweep", "--", "--self-test"])
            .current_dir(&root)
            .status()
            .is_ok_and(|s| s.success());
        return ExitCode::from(u8::from(!ok));
    }
    if matches!(first, "-h" | "--help") {
        print!("{}", include_str!("game_sweep/help.txt"));
        return ExitCode::SUCCESS;
    }
    let built =
        Command::new("cargo").args(["build", "-q", "--release", "-p", "neural-forge-supervisor", "--example", "game-sweep"]).current_dir(&root).status();
    if !built.is_ok_and(|s| s.success()) {
        return ExitCode::FAILURE;
    }
    let target = sh::var("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);
    let sweep = target.join("release/examples/game-sweep");
    let error = Command::new(&sweep).args(args).exec();
    eprintln!("{}: {error}", sweep.display());
    ExitCode::from(126)
}
