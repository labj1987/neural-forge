//! Checks the dependency graph with cargo-deny (deny.toml): RustSec advisories, licences and sources. Runs
//! the pinned cargo-deny release below, fetched once (SHA-256 checked) into tools/native (or NEURAL_FORGE_NATIVE_TOOLS);
//! cargo xtask check-pins reads and bumps the pin. The advisory database is fetched fresh on every run.
//!
//!   cargo xtask cargo-deny [cargo-deny check arguments]
//!
//! It replaced scripts/cargo-deny.sh: the same output and files, and cargo-deny's own exit code.

use std::process::{Command, ExitCode};

use crate::fetch;
use crate::sh::{self, Res};

const CARGO_DENY_VERSION: &str = "0.22.0";
const SHA: &str = "98c56e6a408bfa5c3c60744b6015aaa6aa08f743fdeb88386d06fca035bb1786";

fn run(args: &[String]) -> Res<ExitCode> {
    sh::trap_signals();
    let url = format!(
        "https://github.com/EmbarkStudios/cargo-deny/releases/download/{CARGO_DENY_VERSION}/cargo-deny-{CARGO_DENY_VERSION}-x86_64-unknown-linux-musl.tar.gz"
    );
    let dest = fetch::tools_dir().join("cargo-deny");
    if !sh::is_executable(&dest.join("cargo-deny")) || !fetch::stamped(&dest, SHA) {
        fetch::fetch_one_directory("cargo-deny", "cargo-deny", &url, SHA, "cargo-deny-", &dest)?;
    }
    sh::check(Command::new(dest.join("cargo-deny")).args(["--locked", "check"]).args(args).current_dir(crate::root()))?;
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::finish(run(args))
}
