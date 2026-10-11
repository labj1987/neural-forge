//! Exit codes and checks of single commands, run as a person would run them (scratch XDG dirs).

use std::path::Path;
use std::process::Command;

fn cli(scratch: &Path, shm: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_neural-forge-cli"))
        .args(args)
        .env("XDG_CONFIG_HOME", scratch.join("config"))
        .env("XDG_DATA_HOME", scratch.join("data"))
        .env("XDG_STATE_HOME", scratch.join("state"))
        .env("NEURAL_FORGE_SHM", shm)
        // No `nvidia-smi` or `journalctl` to find: `doctor` must not read the real GPU or kernel log here.
        .env("PATH", scratch.join("no-bin"))
        .output()
        .expect("run neural-forge-cli");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("neural-forge-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn import_binaries_fails_when_the_dll_is_not_there() {
    let dir = scratch("import");
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (ok, out) = cli(&dir, &dir.join("shm.bin"), &["import-binaries", empty.to_str().unwrap()]);
    assert!(!ok, "{out}");
    assert!(out.contains("nvngx_dlssnr.dll"), "{out}");

    std::fs::write(empty.join("nvngx_dlssnr.dll"), "dll").unwrap();
    let (ok, out) = cli(&dir, &dir.join("shm.bin"), &["import-binaries", empty.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(dir.join("data/neural-forge/binaries/nvngx_dlssnr.dll").is_file());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn doctor_opens_the_channel() {
    let dir = scratch("doctor");
    let (_, out) = cli(&dir, &dir.join("shm.bin"), &["doctor"]);
    assert!(out.lines().any(|l| l == "[ok] channel: the settings channel opens"), "{out}");
    assert!(out.contains("[info] xid: journalctl is not installed"), "the kernel log is not read: {out}");

    // A runtime directory that cannot hold the channel (a file where the directory goes).
    std::fs::write(dir.join("not-a-dir"), "").unwrap();
    let (ok, out) = cli(&dir, &dir.join("not-a-dir/shm.bin"), &["doctor"]);
    assert!(!ok);
    assert!(out.lines().any(|l| l.starts_with("[FAILURE] channel: the settings channel cannot be opened")), "{out}");
    std::fs::remove_dir_all(&dir).ok();
}
