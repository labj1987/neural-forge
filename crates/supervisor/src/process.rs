//! Process supervision: start a detached child in its own session (so it and
//! whatever it execs survive this CLI process exiting, and so its whole process
//! *group* can be signaled together), track it via a PID file, and stop it.
//!
//! This is the one place upstream's bash script (`setsid ...`, `kill -TERM -- -$pid`)
//! was doing something bash is naturally good at, so the plan explicitly calls out
//! double-checking the `nix`-based port actually replicates the process-group
//! semantics rather than just the "looks equivalent" case of killing one PID —
//! [`tests::stop_kills_the_whole_process_group_not_just_the_leader`] is that check,
//! run for real (spawns a real shell with a real child of its own, confirms `stop`
//! takes down both).

use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

/// Appends one line to the helper's log (for supervisor-side problems the helper never sees).
pub(crate) fn append_log(log: &str, line: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
        let _ = writeln!(f, "{line}");
    }
}

/// Spawns `program` with `args`/`envs`, in a new session (`setsid`) so it becomes its
/// own process group leader, redirecting stdout/stderr to `log_path` (append). Writes
/// the child's PID to `pid_file`, whose directory must be private to this user (see
/// `neural_forge_protocol::private_dir`); if that write fails the child's whole process
/// group is killed before the error is returned, so no helper runs untracked. Returns the
/// child's PID.
pub fn start_detached(
    program: &str,
    args: &[String],
    envs: &[(String, String)],
    log_path: &str,
    pid_file: &str,
) -> std::io::Result<i32> {
    if let Some(dir) = Path::new(log_path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(log_path)?;
    let log_err = log.try_clone()?;

    let mut cmd = Command::new(program);
    cmd.args(args).envs(envs.iter().cloned()).stdin(std::process::Stdio::null()).stdout(log).stderr(log_err);
    isolate_helper_layers(&mut cmd, std::env::var("VK_INSTANCE_LAYERS").ok().as_deref());
    // SAFETY: `setsid()` is async-signal-safe and the only thing this closure does;
    // it runs in the forked child before exec, exactly what `pre_exec` guarantees.
    unsafe {
        cmd.pre_exec(|| nix::unistd::setsid().map(|_| ()).map_err(std::io::Error::from));
    }
    let mut child = cmd.spawn()?;
    let pid = child.id() as i32;

    if let Err(error) = neural_forge_protocol::private_dir::write_private_file_atomic(pid_file, pid.to_string().as_bytes()) {
        // `spawn()` returns only after the exec, so `setsid()` has already made the child a
        // group leader: the negative-pid kill reaches everything it has started so far.
        let _ = signal_group(pid, Signal::SIGKILL);
        let _ = child.wait();
        return Err(error);
    }
    // The child is meant to keep running after this process exits, so it is never
    // waited on inline. But a long-lived parent (the GUI) that just dropped or
    // forgot it would leave a zombie once the helper exits, and a zombie still
    // answers `kill(pid, 0)`. A reaper thread blocks in `wait()` and collects the
    // exit status; in a short-lived CLI it simply dies with the process, leaving the
    // child running exactly as before.
    let reaper = std::thread::Builder::new().name("helper-reaper".into()).spawn(move || {
        let _ = child.wait();
    });
    if let Err(error) = reaper {
        // Out of threads: nothing sensible to do but let the child run unreaped.
        eprintln!("could not start the helper reaper thread: {error}");
    }
    Ok(pid)
}

// The helper creates its own compute device; it must not inherit game injection.
// Preserve unrelated layers (notably validation) and never change the parent/session.
fn isolate_helper_layers(command: &mut Command, layers: Option<&str>) {
    command.env_remove("VKLayer_DLSS5")
        .env_remove("DLSSNR_ENABLE")
        .env_remove("NEURAL_FORGE_ENABLE");
    if let Some(layers) = layers {
        let kept = layers.split(':').filter(|name| !matches!(*name,
            "VK_LAYER_NV_dlssnr" | "VK_LAYER_neuralforge_neural"))
            .collect::<Vec<_>>().join(":");
        if kept.is_empty() { command.env_remove("VK_INSTANCE_LAYERS"); }
        else { command.env("VK_INSTANCE_LAYERS", kept); }
    }
}

/// The PID in `pid_file` if that process is alive (not a zombie) and, when
/// `expected_cmdline` is given, has it in its command line. A pid file can outlive its
/// process -- a `/tmp` that is not a tmpfs keeps it across a reboot -- and an unrelated
/// process that later got the same PID must not read as the helper.
pub fn running_pid(pid_file: &str, expected_cmdline: Option<&str>) -> Option<i32> {
    let text = std::fs::read_to_string(pid_file).ok()?;
    let pid: i32 = text.trim().parse().ok()?;
    if pid <= 0 {
        return None;
    }
    // Signal 0 sends nothing, just checks whether the process exists and is signalable by us.
    let alive = signal::kill(Pid::from_raw(pid), None).is_ok() && !is_zombie(pid);
    let matches = expected_cmdline.is_none_or(|needle| cmdline_contains(pid, needle));
    (alive && matches).then_some(pid)
}

/// Whether `pid` has exited but not yet been reaped by its parent. A zombie still
/// passes `kill(pid, 0)`, so without this a dead helper would read as running forever.
fn is_zombie(pid: i32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { return false };
    zombie_state(&stat)
}

/// The state field of a `/proc/<pid>/stat` line. The command name is parenthesised and
/// may itself contain spaces or parentheses, so the state is the first field after the
/// *last* `)`.
fn zombie_state(stat: &str) -> bool {
    stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next()) == Some("Z")
}

/// Whether `/proc/<pid>/cmdline` mentions `needle`. Used before signaling a PID read
/// from a file: the PID may have been reused by an unrelated process since it was
/// written, and the process *group* signal below would hit that process's group.
fn cmdline_contains(pid: i32, needle: &str) -> bool {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|raw| String::from_utf8_lossy(&raw).contains(needle))
        .unwrap_or(false)
}

/// Sends `signal` to the whole process *group* `pid` leads (the negative-pid `kill(2)`
/// convention) -- not just `pid` itself, so children it spawned (a Proton/Wine tree,
/// in real use) go down too.
fn signal_group(pid: i32, sig: Signal) -> nix::Result<()> {
    signal::kill(Pid::from_raw(-pid), sig)
}

/// Graceful-then-forced stop: `SIGTERM` the process group, wait up to `timeout` for
/// it to exit, `SIGKILL` it if it hasn't.
///
/// When `expected_cmdline` is given the process at the recorded PID must
/// have it in its command line, or it is presumed to be an unrelated process that
/// inherited a reused PID: nothing is signaled and the stale pid file is dropped.
pub fn stop_matching(pid_file: &str, timeout: Duration, expected_cmdline: Option<&str>) -> std::io::Result<()> {
    let Some(pid) = running_pid(pid_file, expected_cmdline) else {
        let _ = std::fs::remove_file(pid_file);
        return Ok(());
    };
    let _ = signal_group(pid, Signal::SIGTERM);

    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if signal::kill(Pid::from_raw(pid), None).is_err() {
            let _ = std::fs::remove_file(pid_file);
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = signal_group(pid, Signal::SIGKILL);
    std::thread::sleep(Duration::from_millis(200));
    let _ = std::fs::remove_file(pid_file);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_drops_game_injection_but_keeps_validation_layers() {
        let mut command = Command::new("sh");
        command.env_clear().env("VKLayer_DLSS5", "1")
            .env("DLSSNR_ENABLE", "1").env("NEURAL_FORGE_ENABLE", "1");
        isolate_helper_layers(&mut command, Some("VK_LAYER_NV_dlssnr:VK_LAYER_KHRONOS_validation:VK_LAYER_neuralforge_neural:VK_LAYER_NV_present"));
        let result = command.args(["-c", "test -z \"$VKLayer_DLSS5$DLSSNR_ENABLE$NEURAL_FORGE_ENABLE\" && test \"$VK_INSTANCE_LAYERS\" = VK_LAYER_KHRONOS_validation:VK_LAYER_NV_present"]).status().unwrap();
        assert!(result.success());
    }

    /// A path inside a private per-process scratch directory (the pid file's directory
    /// must be private to this user), never the real runtime directory.
    fn scratch_path(name: &str) -> String {
        // One directory per test (named by the file stem), removed by the test when it ends.
        let stem = name.split('.').next().unwrap_or(name);
        let path = format!("{}/neural-forge-process-test-{}-{stem}/{name}", std::env::temp_dir().display(), std::process::id());
        assert!(neural_forge_protocol::private_dir::ensure_private_parent_dir(&path));
        path
    }

    /// Gone, or exited and waiting for its parent to reap it. `kill(pid, 0)` still
    /// succeeds on a zombie, so "is_err" alone is not what "stopped" means.
    fn gone(pid: i32) -> bool {
        signal::kill(Pid::from_raw(pid), None).is_err() || is_zombie(pid)
    }

    #[test]
    fn zombie_state_reads_the_field_after_the_last_paren() {
        assert!(zombie_state("123 (helper) Z 1 123 123 0"));
        assert!(!zombie_state("123 (helper) S 1 123 123 0"));
        // A command name containing spaces and parentheses must not confuse it.
        assert!(zombie_state("123 (we ird) Z) Z 1 1"));
        assert!(!zombie_state("123 (a) Z) S 1 1"));
        assert!(!zombie_state("garbage"));
    }

    #[test]
    fn a_helper_that_exits_is_reaped_and_reported_not_running() {
        let pid_file = scratch_path("reap.pid");
        let log = scratch_path("reap.log");
        let pid = start_detached("/bin/sh", &["-c".to_string(), "exit 0".to_string()], &[], &log, &pid_file).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while running_pid(&pid_file, None).is_some() {
            assert!(Instant::now() < deadline, "pid {pid} still reported running after it exited");
            std::thread::sleep(Duration::from_millis(50));
        }
        // Reaped, not merely hidden from `running_pid`.
        assert!(signal::kill(Pid::from_raw(pid), None).is_err(), "exited child was left as a zombie");
        let _ = std::fs::remove_file(&pid_file);
        let _ = std::fs::remove_dir_all(Path::new(&log).parent().unwrap());
    }

    #[test]
    fn stop_matching_leaves_an_unrelated_process_alone() {
        let pid_file = scratch_path("reuse.pid");
        let log = scratch_path("reuse.log");
        let pid = start_detached("/bin/sleep", &["30".to_string()], &[], &log, &pid_file).unwrap();
        stop_matching(&pid_file, Duration::from_millis(300), Some("definitely-not-this-process")).unwrap();
        assert!(signal::kill(Pid::from_raw(pid), None).is_ok(), "an unrelated process must not be signaled");
        assert!(!Path::new(&pid_file).exists(), "stale pid file should be dropped");
        let _ = signal::kill(Pid::from_raw(pid), Signal::SIGKILL);
        let _ = std::fs::remove_dir_all(Path::new(&log).parent().unwrap());
    }

    #[test]
    fn a_pid_file_naming_an_unrelated_process_is_not_running() {
        let pid_file = scratch_path("unrelated.pid");
        let log = scratch_path("unrelated.log");
        let pid = start_detached("/bin/sleep", &["30".to_string()], &[], &log, &pid_file).unwrap();
        assert_eq!(running_pid(&pid_file, None), Some(pid));
        assert_eq!(running_pid(&pid_file, Some("neural-forge-helper")), None, "a live process that is not the helper must not count");
        let _ = signal::kill(Pid::from_raw(pid), Signal::SIGKILL);
        let _ = std::fs::remove_file(&pid_file);
        let _ = std::fs::remove_dir_all(Path::new(&log).parent().unwrap());
    }

    #[test]
    fn a_pid_file_that_cannot_be_written_kills_the_child() {
        use std::os::unix::fs::PermissionsExt;
        let shared = format!("{}/neural-forge-process-shared-{}", std::env::temp_dir().display(), std::process::id());
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
        let log = scratch_path("unwritable.log");
        let token = format!("neural-forge-untracked-{}", std::process::id());
        let script = format!("sleep 30; : {token}");
        let err = start_detached("/bin/sh", &["-c".to_string(), script], &[], &log, &format!("{shared}/helper.pid"));
        assert!(err.is_err(), "a pid file in a shared directory must be refused");
        // The child was killed (and reaped) before the error came back.
        let survivors: Vec<i32> = std::fs::read_dir("/proc")
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok())
            .filter(|&pid| cmdline_contains(pid, &token) && !gone(pid))
            .collect();
        assert!(survivors.is_empty(), "untracked child left running: {survivors:?}");
        let _ = std::fs::remove_dir_all(&shared);
        let _ = std::fs::remove_dir_all(Path::new(&log).parent().unwrap());
    }

    #[test]
    fn stop_kills_the_whole_process_group_not_just_the_leader() {
        let pid_file = scratch_path("pgroup.pid");
        let log = scratch_path("pgroup.log");
        let marker = scratch_path("pgroup.marker");
        let _ = std::fs::remove_file(&marker);

        // A shell that spawns a background child of its own, then waits on it -- the
        // exact shape `nix`-based `stop()` has to handle correctly: killing only the
        // shell (the "leader") would leave `sleep`, its child, still running.
        let script = format!("sleep 30 & echo $! > {marker}; wait");
        let leader_pid = start_detached("/bin/sh", &["-c".to_string(), script], &[], &log, &pid_file)
            .expect("failed to start the test process group");

        // Wait for the child to actually report its own pid, so this test isn't
        // racing the shell's own startup.
        let child_pid: i32 = {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if let Ok(text) = std::fs::read_to_string(&marker) {
                    if let Ok(pid) = text.trim().parse() {
                        break pid;
                    }
                }
                assert!(Instant::now() < deadline, "the test's own child never started");
                std::thread::sleep(Duration::from_millis(50));
            }
        };

        assert!(signal::kill(Pid::from_raw(leader_pid), None).is_ok(), "leader should be running");
        assert!(signal::kill(Pid::from_raw(child_pid), None).is_ok(), "child should be running");

        stop_matching(&pid_file, Duration::from_secs(2), None).expect("stop should succeed");

        // The grandchild is reparented when the leader dies and may sit as a zombie until
        // its new parent reaps it; that is stopped.
        assert!(gone(leader_pid), "leader should be gone after stop");
        assert!(gone(child_pid), "child should be gone after stop -- this is the real point of this test");

        let _ = std::fs::remove_file(&pid_file);
        let _ = std::fs::remove_dir_all(Path::new(&log).parent().unwrap());
        let _ = std::fs::remove_file(&marker);
    }
}
