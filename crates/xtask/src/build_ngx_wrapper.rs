//! cargo xtask build-ngx-wrapper [OUT]: builds the NGX wrapper for Proton (crates/ngx-wrapper, x86_64-pc-windows-gnu) and
//! writes it to OUT (default target/ngx-wrapper/_nvngx.dll, relative to the repository) stamped as a Wine builtin: the
//! 32 bytes at file offset 0x40 read "Wine builtin DLL", what `winebuild --builtin` writes and what Wine's loader checks
//! before it serves a DLL from its builtin directories (docs/PROTON_NGX.md). With NVIDIA's driver DLL installed, also
//! checks that the wrapper exports every name of it with the same ordinal.
//!   --self-test   stamp and check a minimal PE header, no build
//! Needs the Rust target x86_64-pc-windows-gnu (rustup target add) and the mingw-w64 linker (gcc-mingw-w64-x86-64).
//!
//! It replaced scripts/build-ngx-wrapper.sh: the same output, files and exit codes. The stamp, its check and the export
//! listing are crate::wine_builtin's.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use crate::sh::{self, Fail, Res};
use crate::wine_builtin;

const OBJDUMP: &str = "x86_64-w64-mingw32-objdump";

/// Each NVSDK export of a DLL and its ordinal, one per line (`objdump -p`, read by wine_builtin::exports).
fn exports(dll: &Path) -> Res<String> {
    let mut objdump = Command::new(OBJDUMP);
    let out = objdump.arg("-p").arg(dll).stdin(Stdio::null()).stderr(Stdio::inherit()).output().map_err(|e| sh::not_started(&objdump, e))?;
    if !out.status.success() {
        return Err(Fail::silent(sh::code(out.status)));
    }
    let lines: Vec<String> =
        wine_builtin::exports(&String::from_utf8_lossy(&out.stdout)).into_iter().map(|(name, ordinal)| format!("{name} {ordinal}")).collect();
    Ok(lines.join("\n"))
}

/// The first twenty lines of `diff` between the two listings.
fn diff_head(theirs: &str, ours: &str) -> Res<String> {
    let dir = sh::mktemp()?;
    let (a, b) = (dir.path().join("nvidia"), dir.path().join("ours"));
    for (path, text) in [(&a, theirs), (&b, ours)] {
        std::fs::write(path, format!("{text}\n")).map_err(|e| Fail::new(1, format!("{}: {e}", path.display())))?;
    }
    let mut diff = Command::new("diff");
    let out = diff.arg(&a).arg(&b).output().map_err(|e| sh::not_started(&diff, e))?;
    Ok(String::from_utf8_lossy(&out.stdout).lines().take(20).map(|l| format!("{l}\n")).collect())
}

fn run(args: &[String]) -> Res<ExitCode> {
    let root = crate::root();
    std::env::set_current_dir(&root).map_err(|e| Fail::new(1, format!("{}: {e}", root.display())))?;
    if args.first().is_some_and(|a| a == "--self-test") {
        if wine_builtin::main(&["--self-test".to_string()]) != ExitCode::SUCCESS {
            println!("self-test: the Wine builtin stamp failed");
            return Ok(ExitCode::FAILURE);
        }
        println!("build-ngx-wrapper self-test: ok");
        return Ok(ExitCode::SUCCESS);
    }

    let out = args.first().filter(|a| !a.is_empty()).cloned().unwrap_or_else(|| "target/ngx-wrapper/_nvngx.dll".to_string());
    let target_dir = sh::var("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from).join("ngx-wrapper-build");
    sh::check(
        Command::new("cargo")
            .args(["build", "--release", "--locked", "--manifest-path", "crates/ngx-wrapper/Cargo.toml", "--target", "x86_64-pc-windows-gnu", "--target-dir"])
            .arg(&target_dir),
    )?;
    if let Some(dir) = Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| Fail::new(1, format!("{}: {e}", dir.display())))?;
    }
    let built = target_dir.join("x86_64-pc-windows-gnu/release/_nvngx.dll");
    let tmp = format!("{out}.tmp");
    let mut bytes = std::fs::read(&built).map_err(|e| Fail::new(1, format!("{}: {e}", built.display())))?;
    wine_builtin::stamp(&tmp, &mut bytes)?;
    std::fs::write(&tmp, &bytes).map_err(|e| Fail::new(1, format!("{tmp}: {e}")))?;
    if !wine_builtin::stamped(&std::fs::read(&tmp).map_err(|e| Fail::new(1, format!("{tmp}: {e}")))?) {
        return Err(Fail::silent(1));
    }
    std::fs::rename(&tmp, &out).map_err(|e| Fail::new(1, format!("{out}: {e}")))?;

    let nvidia = ["/usr/lib/x86_64-linux-gnu/nvidia/wine", "/usr/lib64/nvidia/wine", "/usr/lib/nvidia/wine"]
        .iter()
        .map(|dir| Path::new(dir).join("_nvngx.dll"))
        .find(|dll| dll.is_file());
    if let Some(nvidia) = nvidia.filter(|_| sh::which(OBJDUMP).is_some()) {
        let nvidia_exports = exports(&nvidia)?;
        let our_exports = exports(Path::new(&out))?;
        let count = our_exports.lines().filter(|l| !l.is_empty()).count();
        if count == 0 {
            return Err(Fail::new(1, format!("error: no exports found in {out}")));
        }
        if nvidia_exports == our_exports {
            println!("exports: all {count} of {}, same ordinals", nvidia.display());
        } else {
            println!("exports differ from {}:", nvidia.display());
            print!("{}", diff_head(&nvidia_exports, &our_exports)?);
            return Ok(ExitCode::FAILURE);
        }
    }
    let size = std::fs::metadata(&out).map_err(|e| Fail::new(1, format!("{out}: {e}")))?.len();
    println!("{out}: {size} bytes, Wine builtin");
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::finish(run(args))
}
