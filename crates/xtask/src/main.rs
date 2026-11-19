//! xtask: the repository's development tasks, one subcommand each (`cargo xtask <task> ...`, the alias in
//! .cargo/config.toml). Each keeps the flags, output and exit codes of the script it replaced;
//! the dispatcher itself exits 64 (EX_USAGE) for no task or an unknown one.

use std::path::PathBuf;
use std::process::ExitCode;

mod agreement;
mod bench_iq;
mod bench_report;
mod check_namespace;
mod check_pins;
mod check_proton_scripts;
mod check_shaders;
mod gen_model_shape;
mod gen_third_party_crates;
mod gta_bench;
mod hdr_encode;
mod py;
mod release_notes;
mod sync_appdata_releases;
mod wine_builtin;

/// The exit code of an unknown task or none (sysexits' EX_USAGE). Not 1 to 4: tasks use those for their own
/// outcomes (check-proton-scripts' network failure is 2, check-pins' nothing-to-bump is 3 and broken check 4, bench-iq's
/// usage error is 2),
/// and no task uses 64, so a caller can tell "no such task" from a task's own result.
const EX_USAGE: u8 = 64;

type Task = fn(&[String]) -> ExitCode;

const TASKS: &[(&str, Task, &str)] = &[
    ("agreement", agreement::main, "how far the pre-upscaler's answer agrees with the model run on the finished frame"),
    ("bench-iq", bench_iq::main, "image-quality numbers of a benchmark's captured frames"),
    ("bench-report", bench_report::main, "summarise scripts/gta-bench.sh runs"),
    ("check-namespace", check_namespace::main, "the Neural Forge namespace in the shipped files"),
    ("check-pins", check_pins::main, "pins Dependabot cannot see, against upstream (--bump rewrites them)"),
    ("check-proton-scripts", check_proton_scripts::main, "the Proton scripts the NGX wrapper patches, against upstream"),
    ("check-shaders", check_shaders::main, "the committed SPIR-V against the shaders (--update rebuilds it)"),
    ("gen-model-shape", gen_model_shape::main, "crates/supervisor/src/model_shape.rs from a weights listing"),
    ("gen-third-party-crates", gen_third_party_crates::main, "THIRD_PARTY_CRATES.md and the licence texts"),
    ("gta-bench", gta_bench::main, "scripts/gta-bench.sh's crash-dump, frame-generation and Proton checks"),
    ("hdr-encode", hdr_encode::main, "the HDR encode's reference"),
    ("release-notes", release_notes::main, "a version's CHANGELOG section, for its release page"),
    ("sync-appdata-releases", sync_appdata_releases::main, "the appdata <releases> list from CHANGELOG.md"),
    ("wine-builtin", wine_builtin::main, "stamp a PE file as a Wine builtin DLL, or check the stamp"),
];

/// The repository root (this crate is crates/xtask).
pub fn root() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.parent().and_then(|p| p.parent()).map(PathBuf::from).unwrap_or(dir)
}

fn usage() -> ExitCode {
    eprintln!("usage: cargo xtask <task> [args...]\n\ntasks:");
    for (name, _, what) in TASKS {
        eprintln!("  {name:<24}{what}");
    }
    ExitCode::from(EX_USAGE)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(task) = args.first() else { return usage() };
    match TASKS.iter().find(|(name, ..)| name == task) {
        Some((_, run, _)) => run(&args[1..]),
        None => usage(),
    }
}
