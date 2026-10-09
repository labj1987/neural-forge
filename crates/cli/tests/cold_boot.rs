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
        .env("NEURAL_FORGE_SHM", scratch.join("shm.bin"))
        .output()
        .expect("run neural-forge-cli");
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

#[test]
fn shmctl_on_a_cold_boot_starts_from_the_saved_settings_and_saves_its_change() {
    let scratch = std::env::temp_dir().join(format!("neural-forge-cli-cold-boot-{}", std::process::id()));
    let config: PathBuf = scratch.join("config/neural-forge/config.ini");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    std::fs::write(&config, "set_local_tone=2.5\n").unwrap();

    let (ok, out) = cli(&scratch, &["shmctl", "set", "intensity", "3"]);
    assert!(ok, "{out}");
    let (ok, status) = cli(&scratch, &["shmctl", "status"]);
    assert!(ok, "{status}");
    assert!(status.lines().any(|l| l == "local_tone=2.5"), "the saved setting must be applied, not the default:\n{status}");
    assert!(status.lines().any(|l| l == "intensity=3"), "{status}");

    let saved = std::fs::read_to_string(&config).unwrap();
    assert!(saved.contains("set_intensity=3") && saved.contains("set_local_tone=2.5"), "{saved}");

    // A reboot: the channel goes, config.ini stays.
    std::fs::remove_file(scratch.join("shm.bin")).unwrap();
    let (ok, status) = cli(&scratch, &["shmctl", "status"]);
    assert!(ok, "{status}");
    assert!(status.lines().any(|l| l == "intensity=3") && status.lines().any(|l| l == "local_tone=2.5"), "{status}");

    std::fs::remove_dir_all(&scratch).ok();
}
