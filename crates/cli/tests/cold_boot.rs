//! `shmctl` against a channel nothing has configured yet (a cold boot: no layer, no GUI) must act
//! on the saved settings, and what it changes must be saved for the next boot.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli(scratch: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_neural-forge-cli"))
        .args(args)
        .env("XDG_CONFIG_HOME", scratch.join("config"))
        .env("XDG_DATA_HOME", scratch.join("data"))
        .env("XDG_STATE_HOME", scratch.join("state"))
        // The home too: `doctor` finds Steam (its libraries, launch options, app cache) under it.
        .env("HOME", scratch.join("home"))
        .env("NEURAL_FORGE_SHM", scratch.join("shm.bin"))
        .output()
        .expect("run neural-forge-cli");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

#[test]
fn shmctl_on_a_cold_boot_starts_from_the_saved_settings_and_saves_its_change() {
    /// Removed when the test ends, passed or failed.
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let guard = Scratch(std::env::temp_dir().join(format!("neural-forge-cli-cold-boot-{}", std::process::id())));
    let scratch = guard.0.clone();
    let config: PathBuf = scratch.join("config/neural-forge/config.ini");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    // The channel's directory must be private, as the real runtime directory is.
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(&config, "set_local_tone=2.5\n").unwrap();

    let (ok, out) = cli(&scratch, &["shmctl", "set", "intensity", "3"]);
    assert!(ok, "{out}");
    let (ok, status) = cli(&scratch, &["shmctl", "status"]);
    assert!(ok, "{status}");
    assert!(status.lines().any(|l| l == "local_tone=2.5"), "the saved setting must be applied, not the default:\n{status}");
    assert!(status.lines().any(|l| l == "intensity=3"), "{status}");

    let saved = std::fs::read_to_string(&config).unwrap();
    assert!(saved.contains("set_intensity=3") && saved.contains("set_local_tone=2.5"), "{saved}");

    // A reboot: the channel goes, config.ini stays. `status` only looks: it says there is no
    // channel and creates none; the next command that changes something starts from config.ini.
    std::fs::remove_file(scratch.join("shm.bin")).unwrap();
    let (ok, status) = cli(&scratch, &["shmctl", "status"]);
    assert!(ok && status.starts_with("no channel at "), "{status}");
    assert!(!scratch.join("shm.bin").exists(), "status created the channel");
    let (ok, out) = cli(&scratch, &["shmctl", "set", "debug_view", "0"]);
    assert!(ok, "{out}");
    let (_, status) = cli(&scratch, &["shmctl", "status"]);
    assert!(status.lines().any(|l| l == "intensity=3") && status.lines().any(|l| l == "local_tone=2.5"), "{status}");
    drop(guard);
}
