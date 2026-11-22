//! Builds the AppImage from the working tree and puts exactly that build where the rig runs it:
//! the launcher copy (~/AppImages/neural-forge.appimage) and the installed layers, then
//! proves the installed files are byte-identical to the ones inside the AppImage. Leaves nothing
//! behind: no backup of the previous AppImage (releases are the rollback) and no temp files.
//!
//! Usage: cargo xtask deploy-rig <host>      (or NF_DEPLOY_HOST=<host> cargo xtask deploy-rig)
//!
//! It replaced scripts/deploy-rig.sh: the same steps, output and exit codes. What runs on the rig is still a shell
//! text (deploy_rig/remote.sh, sent to `bash -s` over ssh, as before): the rig has no cargo and no checkout.

use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use crate::sh::{self, Fail, Res};

const REMOTE: &str = include_str!("deploy_rig/remote.sh");

/// The local build only exists to be copied; releases come from CI, so it is removed however this ends.
struct Built(Vec<PathBuf>);

impl Drop for Built {
    fn drop(&mut self) {
        for path in &self.0 {
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(path);
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn run(args: &[String]) -> Res<ExitCode> {
    let root = crate::root();
    std::env::set_current_dir(&root).map_err(|e| Fail::new(1, format!("{}: {e}", root.display())))?;
    let Some(host) = args.first().filter(|a| !a.is_empty()).cloned().or_else(|| sh::var("NF_DEPLOY_HOST")) else {
        return Err(Fail::new(2, "usage: cargo xtask deploy-rig <host>  (or set NF_DEPLOY_HOST)"));
    };
    let manifest = std::fs::read_to_string("Cargo.toml").map_err(|e| Fail::new(1, format!("Cargo.toml: {e}")))?;
    let version = manifest.lines().find(|l| l.starts_with("version")).and_then(|l| l.split('"').nth(1)).unwrap_or("").to_string();
    let appimage = format!("neural-forge-{version}-x86_64.AppImage");

    println!("==> building {appimage}");
    sh::trap_signals();
    let scratch = sh::mktemp()?;
    let build_log = scratch.path().join("build.log");
    let _built = Built(vec![PathBuf::from(&appimage), PathBuf::from(format!("{appimage}.zsync")), PathBuf::from("build-appimage")]);
    let log = std::fs::File::create(&build_log).map_err(|e| Fail::new(1, format!("{}: {e}", build_log.display())))?;
    let both = log.try_clone().map_err(|e| Fail::new(1, format!("{}: {e}", build_log.display())))?;
    let built = Command::new("bash").arg("build-appimage.sh").stdout(log).stderr(both).status();
    sh::interrupted()?;
    if !built.is_ok_and(|s| s.success()) {
        let text = String::from_utf8_lossy(&std::fs::read(&build_log).unwrap_or_default()).into_owned();
        let lines: Vec<&str> = text.lines().collect();
        for line in &lines[lines.len().saturating_sub(20)..] {
            println!("{line}");
        }
        return Ok(ExitCode::FAILURE);
    }

    println!("==> copying to {host}");
    sh::check(Command::new("scp").arg("-q").arg(&appimage).arg(format!("{host}:/tmp/nf-deploy.AppImage")))?;
    let mut ssh = Command::new("ssh");
    let mut remote = ssh.arg(&host).arg(format!("VERSION={version} bash -s")).stdin(Stdio::piped()).spawn().map_err(|e| sh::not_started(&ssh, e))?;
    if let Some(mut stdin) = remote.stdin.take() {
        let _ = std::io::Write::write_all(&mut stdin, REMOTE.as_bytes());
    }
    let status = remote.wait().map_err(|e| Fail::new(1, format!("ssh: {e}")))?;
    sh::interrupted()?;
    if !status.success() {
        return Err(Fail::silent(sh::code(status)));
    }
    println!("==> deployed and verified");
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::finish(run(args))
}

#[cfg(test)]
mod tests {
    /// The text sent to the rig is a shell script: it must at least parse.
    #[test]
    fn the_remote_part_parses() {
        let dir = crate::sh::mktemp().unwrap();
        let script = dir.path().join("remote.sh");
        std::fs::write(&script, super::REMOTE).unwrap();
        assert!(std::process::Command::new("bash").arg("-n").arg(&script).status().unwrap().success());
        assert!(super::REMOTE.starts_with("set -eo pipefail\n"));
    }
}
