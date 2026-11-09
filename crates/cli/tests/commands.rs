//! Exit codes and checks of single commands, run as a person would run them (scratch XDG dirs).

use std::path::Path;
use std::process::Command;

fn cli(scratch: &Path, shm: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_neural-forge-cli"))
        .args(args)
        .env("XDG_CONFIG_HOME", scratch.join("config"))
        .env("XDG_DATA_HOME", scratch.join("data"))
        .env("XDG_STATE_HOME", scratch.join("state"))
        // The home too: `doctor` finds Steam (its libraries, launch options, app cache) under it.
        .env("HOME", scratch.join("home"))
        .env("NEURAL_FORGE_SHM", shm)
        // No `nvidia-smi` or `journalctl` to find: `doctor` must not read the real GPU or kernel log here.
        .env("PATH", scratch.join("no-bin"))
        .output()
        .expect("run neural-forge-cli");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// A scratch directory, removed when the test ends, passed or failed.
struct Scratch(std::path::PathBuf);

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("neural-forge-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

#[test]
fn import_binaries_fails_when_the_dll_is_not_there() {
    let dir = scratch("import");
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (ok, out) = cli(&dir, &dir.join("shm.bin"), &["import-binaries", empty.to_str().unwrap()]);
    assert!(!ok, "{out}");
    assert!(out.contains("nvngx_dlssnr.dll"), "{out}");

    // Not a DLL: refused, and nothing is written (the import itself is tested in `model`).
    std::fs::write(empty.join("nvngx_dlssnr.dll"), "dll").unwrap();
    let (ok, out) = cli(&dir, &dir.join("shm.bin"), &["import-binaries", empty.to_str().unwrap()]);
    assert!(!ok, "{out}");
    assert!(out.contains("not a readable DLL"), "{out}");
    assert!(!dir.join("data/neural-forge/binaries/nvngx_dlssnr.dll").exists());
}

#[test]
fn doctor_reads_the_channel_without_creating_it() {
    let dir = scratch("doctor");
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    let (_, out) = cli(&dir, &dir.join("shm.bin"), &["doctor"]);
    assert!(out.lines().any(|l| l.starts_with("[info] channel: no settings channel yet")), "{out}");
    assert!(out.lines().any(|l| l == "[info] steam: no Steam installation found"), "the real home's Steam is not read: {out}");
    assert!(!dir.join("shm.bin").exists() && !dir.join("data").exists(), "the doctor only looks");
    assert!(out.contains("[info] xid: journalctl is not installed"), "the kernel log is not read: {out}");
    let (ok, out) = cli(&dir, &dir.join("shm.bin"), &["shmctl", "set", "debug_view", "0"]);
    assert!(ok, "{out}");
    let (_, out) = cli(&dir, &dir.join("shm.bin"), &["doctor"]);
    assert!(out.lines().any(|l| l == "[ok] channel: the settings channel opens"), "{out}");

    // A runtime directory that cannot hold the channel (a file where the directory goes).
    std::fs::write(dir.join("not-a-dir"), "").unwrap();
    let (ok, out) = cli(&dir, &dir.join("not-a-dir/shm.bin"), &["doctor"]);
    assert!(!ok);
    assert!(out.lines().any(|l| l.starts_with("[FAILURE] channel: the settings channel cannot be opened")), "{out}");
}

/// A game's own settings: only what differs from the settings for all games is kept, a value the game gets
/// anyway removes its line, and `clear` restores its defaults; lines this build does not know stay.
#[test]
fn game_settings_keep_only_what_differs() {
    let dir = scratch("game-settings");
    let shm = dir.join("shm.bin");
    let config = dir.join("config/neural-forge");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(config.join("config.ini"), "set_intensity=1.5\n").unwrap();
    std::fs::write(config.join("games.ini"), "[570]\nfuture_key=1\n").unwrap();
    let games = || std::fs::read_to_string(config.join("games.ini")).unwrap_or_default();

    let (ok, out) = cli(&dir, &shm, &["game-settings", "set", "570", "intensity", "0.5"]);
    assert!(ok, "{out}");
    assert!(games().contains("set_intensity=0.5") && games().contains("future_key=1"), "{}", games());
    let (ok, out) = cli(&dir, &shm, &["game-settings", "set", "exe:Game.exe", "placement", "present"]);
    assert!(ok, "{out}");
    assert!(games().contains("[exe:game.exe]\nset_placement=3"), "a placement by name, stored as config.ini stores it:\n{}", games());
    let (ok, out) = cli(&dir, &shm, &["game-settings", "list"]);
    assert!(ok && out.contains("570  intensity=0.5") && out.contains("exe:game.exe  placement=3"), "{out}");
    let (ok, out) = cli(&dir, &shm, &["game-settings", "show", "570"]);
    assert!(ok && out.contains("intensity=0.5"), "{out}");
    // All games' value: the line goes.
    let (ok, out) = cli(&dir, &shm, &["game-settings", "set", "570", "intensity", "1.5"]);
    assert!(ok && out.contains("follows the settings for all games"), "{out}");
    assert!(!games().contains("set_intensity"), "{}", games());
    let (ok, out) = cli(&dir, &shm, &["game-settings", "set", "570", "intensity", "9"]);
    assert!(!ok && out.contains("from 0 to 4"), "{out}");
    let (ok, out) = cli(&dir, &shm, &["game-settings", "set", "570", "no_such_setting", "1"]);
    assert!(!ok && out.contains("unknown setting"), "{out}");
    let (ok, out) = cli(&dir, &shm, &["game-settings", "clear", "exe:game.exe"]);
    assert!(ok && out.contains("removed 1 setting"), "{out}");
    assert_eq!(games(), "[570]\nfuture_key=1\n\n");
    let (ok, out) = cli(&dir, &shm, &["game-settings", "show", "no such game"]);
    assert!(!ok && out.contains("no installed Steam game"), "{out}");
    assert!(!shm.exists(), "no channel is created for a file change");
}
