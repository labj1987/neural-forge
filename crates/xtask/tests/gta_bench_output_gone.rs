//! gta-bench puts the settings --set changed back when its output goes away mid-run (a closed pipe, a dropped ssh
//! connection of a --host run), as the shell script's EXIT trap did. Rust ignores SIGPIPE, so a plain `println!` there
//! panics on the failed write, which aborts in this workspace's profile, and nothing would be put back.
//!
//! A stand-in rig in a scratch directory: a CLI that keeps `model_interval` in a file, a Steam Linux Runtime entry point
//! that only leaves a mark. No game is started and none is looked for beyond the runner's own "already running" check.

use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn script(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn gta_bench_puts_settings_back_when_its_output_goes_away() {
    let dir = std::env::temp_dir().join(format!("nf-xtask-output-gone-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let d = dir.to_string_lossy().into_owned();
    let runtime = dir.join("steam/steamapps/common/SteamLinuxRuntime_4");
    std::fs::create_dir_all(&runtime).unwrap();
    std::fs::create_dir_all(dir.join("home")).unwrap();
    script(&runtime.join("_v2-entry-point"), &format!("#!/bin/sh\necho launched > '{d}/launched'\n"));
    // The CLI: `model_interval` in $dir/mi, every call logged. Once the setting is changed, `status` waits for the test
    // to have closed the runner's output ($dir/closed), so the runner's next line is written to nobody.
    script(
        &dir.join("cli"),
        &format!(
            "#!/bin/sh\necho \"$*\" >> '{d}/cli.log'\ncase \"$2\" in\nstatus)\n  if [ \"$(cat '{d}/mi')\" = 2 ]; then n=0; while [ ! -e '{d}/closed' ] && [ $n -lt 200 ]; do sleep 0.05; n=$((n+1)); done; fi\n  \
             echo \"model_interval=$(cat '{d}/mi')\";;\nset) echo \"$4\" > '{d}/mi';;\nesac\n"
        ),
    );
    std::fs::write(dir.join("mi"), "1\n").unwrap();

    let mut bench = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gta-bench", "--set", "model_interval=2", "--tries", "1", "run", ""])
        .env("HOME", dir.join("home"))
        .env("NF_BENCH_STEAM_ROOT", dir.join("steam"))
        .env("NF_BENCH_PROTON", dir.join("proton"))
        .env("NF_BENCH_CLI", dir.join("cli"))
        .env("NF_BENCH_DIR", dir.join("runs"))
        .env_remove("NF_BENCH_STEAM_LIBRARY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let log = || std::fs::read_to_string(dir.join("cli.log")).unwrap_or_default();
    let end = Instant::now() + Duration::from_secs(30);
    while !log().contains("shmctl set model_interval 2") && Instant::now() < end && bench.try_wait().unwrap().is_none() {
        std::thread::sleep(Duration::from_millis(20));
    }
    if !log().contains("shmctl set model_interval 2") {
        let out = bench.wait_with_output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        // On a machine where GTA is running the runner refuses to start, rightly: nothing to check then.
        assert!(stderr.contains("GTA is already running"), "the runner never changed the setting: {stderr}");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    // The reader goes away, then the runner goes on.
    drop(bench.stdout.take());
    std::fs::write(dir.join("closed"), "").unwrap();
    let out = bench.wait_with_output().unwrap();
    let (log, left) = (log(), std::fs::read_to_string(dir.join("mi")).unwrap());
    let launched = dir.join("launched").exists();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(left.trim(), "1", "model_interval was left changed (exit {:?}, signal {:?}); the CLI saw:\n{log}", out.status.code(), out.status.signal());
    assert!(log.lines().any(|l| l == "shmctl set model_interval 1"), "{log}");
    // As a shell ends that wrote to a closed pipe: 128 + SIGPIPE.
    assert_eq!(out.status.code(), Some(141));
    assert!(!launched, "a game was launched after the output had gone");
}
