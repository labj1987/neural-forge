//! Shared helper-process supervision: config, XDG paths, and start/stop, used by
//! both `neural-forge-cli` and `neural-forge` so "how to launch the helper" (runner type,
//! environment variables, the Proton-vs-Wine command line) exists in exactly one
//! place instead of being duplicated and risking drift between the two front ends.
//! Linux-only: spawns child processes, reads XDG env vars.

pub mod config;
pub mod gpu;
pub mod install;
pub mod install_dir;
pub mod paths;
pub mod profiles;
pub mod provision;
mod process;
pub mod runners;

pub use config::Config;

use std::os::fd::{AsRawFd, OwnedFd};
use std::time::Duration;

/// What every front end needs to recognise the helper's process by: both runners (plain
/// Wine, Proton) carry the helper's path in their own command line.
const HELPER_CMDLINE: &str = "neural-forge-helper";

pub fn pid_file() -> String {
    format!("{}/helper.pid", neural_forge_protocol::shm_runtime_dir())
}

fn lock_file() -> String {
    format!("{}/helper.lock", neural_forge_protocol::shm_runtime_dir())
}

/// The one channel path everything this supervisor touches uses: `config.ini`'s `shm=` when
/// it is set (and inside this project's namespace), else `$NEURAL_FORGE_SHM`, else the
/// default under the runtime directory. The helper is started on it, saved settings are
/// applied to it, and the GUI and CLI open it through [`open_channel`].
pub fn channel_path(cfg: &Config) -> String {
    if !cfg.shm.is_empty() && neural_forge_protocol::isolated_path(&cfg.shm) {
        return cfg.shm.clone();
    }
    neural_forge_protocol::env::var("NEURAL_FORGE_SHM")
        .filter(|s| !s.is_empty() && neural_forge_protocol::isolated_path(s))
        .unwrap_or_else(neural_forge_protocol::shm_default_path)
}

/// Opens the mapping at [`channel_path`].
pub fn open_channel(cfg: &Config) -> Result<neural_forge_protocol::mapping::Mapping, neural_forge_protocol::mapping::OpenError> {
    neural_forge_protocol::mapping::open_path(&channel_path(cfg))
}

/// The helper's PID if the process at the PID in the pid file is alive and is actually the
/// helper. A pid file can outlive its process (a `/tmp` that is not a tmpfs survives a
/// reboot), and whatever later takes that PID is not the helper.
pub fn is_running() -> Option<i32> {
    process::running_pid(&pid_file(), Some(HELPER_CMDLINE))
}

/// `flock(LOCK_EX | LOCK_NB)` on `<runtime dir>/helper.lock`, held for the whole of one
/// `start()` or `stop()`, so a GUI auto-start racing `neural-forge-cli start` (or a double
/// click) cannot spawn a second helper between one caller's "is it running?" and its pid
/// file write. Released when dropped (the descriptor is close-on-exec, so the helper never
/// inherits it).
struct HelperLock {
    _fd: OwnedFd,
}

enum LockError {
    Busy,
    Io(std::io::Error),
}

impl HelperLock {
    fn acquire() -> Result<Self, LockError> {
        let path = lock_file();
        let fd = neural_forge_protocol::private_dir::open_private_file(&path, libc::O_RDWR | libc::O_CREAT, 0o600).ok_or_else(|| {
            LockError::Io(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("cannot open {path}: its directory must be a private directory owned by this user"),
            ))
        })?;
        // SAFETY: `fd` is an open descriptor owned by this function.
        if unsafe { libc::flock(fd.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(Self { _fd: fd });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            Err(LockError::Busy)
        } else {
            Err(LockError::Io(err))
        }
    }
}

/// Graceful-then-forced stop of the whole helper process group. Fails with
/// `ErrorKind::WouldBlock` when another start or stop holds the helper lock.
///
/// Real, confirmed bug on `lordnikon` (2026-09-11): `process::stop`'s process-group
/// kill can report success while the actual Wine-hosted `neural-forge-helper.exe` survives
/// anyway, once wineserver has taken it over -- Wine's own internal process
/// management doesn't reliably keep every process inside the group the original
/// `setsid()` created. Confirmed via a real, orphaned helper left running after a
/// `stop()`/`start()` cycle: it kept writing to the same live SHM mapping as the new
/// helper, corrupting shared state (`helper_state` flapping between two independent
/// writers) with no crash or error anywhere to point at the real cause. A real
/// `wineserver -k` against this exact prefix -- the same recovery this project's own
/// manual testing has used by hand every time this exact symptom came up -- is
/// cheap, targeted, and closes the gap: best-effort (a plain Wine install with no
/// `wineserver` on `PATH`, or nothing left to kill, are not real failures worth
/// surfacing), run after the normal group kill so a routine stop/restart no longer
/// needs a human to notice and clean this up by hand.
pub fn stop(timeout: Duration) -> std::io::Result<()> {
    let _lock = match HelperLock::acquire() {
        Ok(lock) => lock,
        Err(LockError::Busy) => return Err(std::io::Error::new(std::io::ErrorKind::WouldBlock, "a helper start or stop is already in progress")),
        Err(LockError::Io(e)) => return Err(e),
    };
    // The command-line check guards against signaling a process that reused the PID.
    process::stop_matching(&pid_file(), timeout, Some(HELPER_CMDLINE))?;
    let cfg = Config::load();
    if let Some(wineserver) = wineserver_binary(&cfg) {
        let _ = std::process::Command::new(wineserver).arg("-k").env("WINEPREFIX", real_wineprefix(&cfg, &paths::prefix_dir())).status();
    }
    Ok(())
}

/// The `wineserver` binary belonging to `cfg`'s configured runner -- for Proton, a
/// sibling of the `proton` script itself (`<runner dir>/files/bin/wineserver`,
/// confirmed present at that exact relative path on real Proton-CachyOS/GE
/// installs); for plain Wine, `wineserver` is normally already resolvable via
/// `PATH` alongside `wine` itself, so no path derivation is needed.
fn wineserver_binary(cfg: &Config) -> Option<String> {
    if cfg.runner_type == "proton" {
        let dir = std::path::Path::new(&cfg.runner_path).parent()?;
        let candidate = dir.join("files/bin/wineserver");
        candidate.is_file().then(|| candidate.display().to_string())
    } else if cfg.runner_type.is_empty() {
        None
    } else {
        Some("wineserver".to_string())
    }
}

/// The real `WINEPREFIX` a real wineserver command needs, given `prefix_dir`
/// (`paths::prefix_dir()` in real use, passed in rather than read directly so this
/// stays a pure function -- easy to test without touching the process-wide
/// `XDG_DATA_HOME` env var `prefix_dir()` itself depends on). **Not** simply
/// `prefix_dir` unchanged: real, confirmed bug caught testing the fix above on
/// `lordnikon` before it ever shipped -- for `runner_type = "proton"`, `start()`
/// hands `prefix_dir` to Proton as `STEAM_COMPAT_DATA_PATH`, but Proton's own
/// launch script internally re-derives and uses `STEAM_COMPAT_DATA_PATH/pfx` as
/// wine's *real* prefix. Confirmed directly: `wineserver -k` with
/// `WINEPREFIX=<prefix_dir>` exits `1` and kills nothing; `WINEPREFIX=
/// <prefix_dir>/pfx` exits `0` and actually works, against the exact same live
/// orphaned process. Plain Wine (`runner_type = "wine"`) has no such nesting;
/// `prefix_dir` is already the real prefix there.
fn real_wineprefix(cfg: &Config, prefix_dir: &str) -> String {
    if cfg.runner_type == "proton" {
        format!("{prefix_dir}/pfx")
    } else {
        prefix_dir.to_string()
    }
}

#[derive(Debug)]
pub enum StartError {
    AlreadyRunning(i32),
    /// Another `start()` or `stop()` (this process or another) holds the helper lock.
    Busy,
    HelperNotFound,
    NoRunnerConfigured,
    /// A Proton runner needs the Steam client's install directory, and none was found.
    NoSteamInstall,
    /// Preparing the managed prefix for the system-Wine runner failed.
    Provision(String),
    Spawn(std::io::Error),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::AlreadyRunning(pid) => write!(f, "helper already running (pid {pid})"),
            StartError::Busy => write!(f, "another helper start or stop is already in progress"),
            StartError::HelperNotFound => write!(f, "neural-forge-helper.exe not found"),
            StartError::NoRunnerConfigured => write!(f, "no runner configured (run `neural-forge-cli init` first)"),
            StartError::NoSteamInstall => write!(f, "the Proton runner needs a Steam install, and none was found"),
            StartError::Provision(e) => write!(f, "preparing the Wine prefix failed: {e}"),
            StartError::Spawn(e) => write!(f, "failed to start helper: {e}"),
        }
    }
}

impl std::error::Error for StartError {}

pub struct StartedHelper {
    pub pid: i32,
    pub runner_type: String,
    pub runner_path: String,
    pub log: String,
}

/// Applies the saved settings (`config.ini`) to the live header at [`channel_path`] if
/// nothing has applied them since it was last initialised. The header is re-initialised with
/// defaults by whichever of the layer, helper or GUI finds it missing -- a game started first,
/// or a fresh boot -- and before this only the GUI, and only when it had created the file
/// itself, put the user's settings back. `init_defaults` leaves `tuning_seq` at 0 and
/// `persist::apply` bumps it, which is what "not yet applied" is read from.
pub fn apply_saved_settings(cfg: &Config) {
    let Ok(mapping) = open_channel(cfg) else { return };
    let hdr = mapping.header();
    if hdr.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        neural_forge_protocol::persist::apply(hdr, &cfg.settings);
        hdr.control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Starts the helper under `cfg`'s configured runner (Proton or plain Wine),
/// building the same environment variables (`WINEPREFIX`, `NEURAL_FORGE_SHM`, `NEURAL_FORGE_UID`,
/// the Proton-specific NVAPI/compat-data ones) either front end needs -- this is the
/// one place that construction happens. Holds the helper lock throughout (see
/// [`HelperLock`]); a concurrent call gets [`StartError::Busy`] or, once the first has
/// finished, [`StartError::AlreadyRunning`]. Blocking (it may run `wineboot` or download
/// the Wine runtime DLLs): a GUI calls it off its main thread.
pub fn start(cfg: &Config) -> Result<StartedHelper, StartError> {
    let _lock = match HelperLock::acquire() {
        Ok(lock) => lock,
        Err(LockError::Busy) => return Err(StartError::Busy),
        Err(LockError::Io(e)) => return Err(StartError::Spawn(e)),
    };
    if let Some(pid) = is_running() {
        return Err(StartError::AlreadyRunning(pid));
    }
    let Some(helper) = install_dir::helper_exe() else {
        return Err(StartError::HelperNotFound);
    };
    apply_saved_settings(cfg);
    if cfg.runner_path.is_empty() {
        return Err(StartError::NoRunnerConfigured);
    }
    // Real, confirmed crash on `lordnikon` (2026-09-10): both Proton and plain Wine
    // are handed this directory as their own prefix root (`WINEPREFIX` below, plus
    // `STEAM_COMPAT_DATA_PATH` for Proton specifically) and assume it already exists
    // -- Proton's own `setup_prefix()` fails opening `pfx.lock` inside it with a
    // `FileNotFoundError` otherwise. Normally invisible (every run after the first
    // successful one finds it already there), until it's missing for any reason --
    // `ensure_dirs()` now creates it too, but `start()` doesn't get to assume
    // whatever called that ran recently, or that this is the app's own supervisor
    // creating it for the first time; `create_dir_all` is a no-op if it already
    // exists, so paying for it unconditionally here costs nothing on the common path.
    let _ = std::fs::create_dir_all(paths::prefix_dir());

    // Without a `binaries=` line the helper would be told to look in `Z:` itself; fall back to
    // the default binaries directory the way `doctor` does.
    let binaries = if cfg.binaries.is_empty() { paths::binaries_dir() } else { cfg.binaries.clone() };
    let mut envs = vec![
        ("WINEPREFIX".to_string(), paths::prefix_dir()),
        // POSIX: the helper maps it onto `Z:` itself (see `neural_forge_helper::shm`).
        ("NEURAL_FORGE_SHM".to_string(), channel_path(cfg)),
        // The helper opens these two with Windows file APIs, which read a bare POSIX path
        // as relative to the current drive; Wine exposes the host filesystem as `Z:`.
        ("NEURAL_FORGE_LOG".to_string(), format!("Z:{}", cfg.log)),
        ("NEURAL_FORGE_BIN_DIR".to_string(), format!("Z:{binaries}")),
        ("WINEDEBUG".to_string(), "-all".to_string()),
    ];
    // SAFETY-relevant only in the "matches a real deployment" sense, not memory
    // safety: NEURAL_FORGE_UID has to be the same value the layer computes
    // `shm_runtime_dir()` from, which reads it from the environment too -- passing it
    // explicitly here is what keeps both sides pointed at the same file.
    // SAFETY: getuid() takes no arguments and cannot fail.
    let uid = unsafe { libc::getuid() };
    envs.push(("NEURAL_FORGE_UID".to_string(), uid.to_string()));

    let (program, args): (String, Vec<String>) = if cfg.runner_type == "proton" {
        envs.push(("PROTON_ENABLE_NVAPI".to_string(), "1".to_string()));
        envs.push(("NEURAL_FORGE_SKIP_NVAPI".to_string(), "1".to_string()));
        envs.push(("STEAM_COMPAT_DATA_PATH".to_string(), paths::prefix_dir()));
        // Proton's own launch script reads this directly out of the environment
        // (`os.environ["STEAM_COMPAT_CLIENT_INSTALL_PATH"]`, no fallback) during
        // prefix setup, before it ever gets to running the helper .exe -- omitting it
        // is a real, confirmed `KeyError` crash on *every* start attempt, found
        // 2026-09-10 running this against a real game session on `lordnikon`: the
        // helper never got further than Proton's own setup_prefix() step. Without a
        // Steam install there is nothing to set it to, so the start fails here with a
        // reason instead of reporting a pid that dies immediately.
        let Some(steam_dir) = paths::steam_install_dir() else {
            return Err(StartError::NoSteamInstall);
        };
        envs.push(("STEAM_COMPAT_CLIENT_INSTALL_PATH".to_string(), steam_dir));
        (cfg.runner_path.clone(), vec!["run".to_string(), helper.display().to_string()])
    } else {
        if cfg.runner_type == "wine" {
            // Plain Wine has no DXVK-NVAPI of its own: the managed prefix is given DXVK's dxgi and
            // DXVK-NVAPI (see `provision`), and the helper loads NVAPI itself. Without them NGX
            // cannot find the GPU, so a failure here fails the start.
            if let Err(e) = provision::prepare_wine_prefix(cfg) {
                crate::process::append_log(&cfg.log, &format!("[neural-forge] system-Wine prefix setup failed: {e}"));
                return Err(StartError::Provision(e));
            }
            envs.extend(provision::wine_env());
        }
        (cfg.runner_path.clone(), vec![helper.display().to_string()])
    };

    process::start_detached(&program, &args, &envs, &cfg.log, &pid_file())
        .map(|pid| StartedHelper {
            pid,
            runner_type: cfg.runner_type.clone(),
            runner_path: cfg.runner_path.clone(),
            log: cfg.log.clone(),
        })
        .map_err(StartError::Spawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Points every path `start()` touches at a scratch directory -- XDG dirs, `$HOME`, the
    /// runtime dir (through a unique `NEURAL_FORGE_UID`), the install dir holding a dummy
    /// helper -- under the crate's env lock, restoring everything on drop. A dummy runner
    /// script stands in for Wine/Proton: it records its pid and environment and sleeps.
    struct ScratchStart {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Vec<(&'static str, Option<String>)>,
        dir: PathBuf,
        runtime: String,
    }

    impl ScratchStart {
        fn new(tag: &str) -> Self {
            let guard = crate::paths::tests::XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-start-test-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("install/helper")).unwrap();
            std::fs::write(dir.join("install/helper/neural-forge-helper.exe"), "").unwrap();
            let uid = format!("starttest-{tag}-{}", std::process::id());
            let mut prev = Vec::new();
            for (var, value) in [
                ("HOME", Some(dir.join("home").display().to_string())),
                ("XDG_DATA_HOME", Some(dir.join("data").display().to_string())),
                ("XDG_CONFIG_HOME", Some(dir.join("config").display().to_string())),
                ("XDG_STATE_HOME", Some(dir.join("state").display().to_string())),
                ("NEURAL_FORGE_UID", Some(uid.clone())),
                ("NEURAL_FORGE_INSTALL_DIR", Some(dir.join("install").display().to_string())),
                ("NEURAL_FORGE_SHM", None),
            ] {
                prev.push((var, std::env::var(var).ok()));
                match value {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
            let runtime = neural_forge_protocol::shm_runtime_dir();
            assert!(runtime.ends_with(&uid), "the runtime dir must be the scratch one: {runtime}");
            Self { _guard: guard, prev, dir, runtime }
        }

        /// A `custom` runner: records `$$` and the helper's environment, then sleeps.
        fn config(&self) -> Config {
            let runner = self.dir.join("runner.sh");
            let record = self.dir.join("record");
            std::fs::write(
                &runner,
                format!(
                    "#!/bin/sh\necho \"$$ $NEURAL_FORGE_SHM $NEURAL_FORGE_LOG $NEURAL_FORGE_BIN_DIR\" >> {}\nsleep 30\n",
                    record.display()
                ),
            )
            .unwrap();
            std::fs::set_permissions(&runner, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
            Config {
                runner_type: "custom".to_string(),
                runner_path: runner.display().to_string(),
                shm: format!("{}/channel/shm.bin", self.dir.display()),
                log: self.dir.join("helper.log").display().to_string(),
                ..Config::default()
            }
        }

        /// Lines the dummy runner wrote, waiting until `n` have appeared.
        fn records(&self, n: usize) -> Vec<String> {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                let lines: Vec<String> = std::fs::read_to_string(self.dir.join("record")).unwrap_or_default().lines().map(str::to_string).collect();
                if lines.len() >= n || std::time::Instant::now() > deadline {
                    return lines;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    impl Drop for ScratchStart {
        fn drop(&mut self) {
            // Kill whatever the test started (no `stop()`: it would run `wineserver -k`).
            let _ = process::stop_matching(&pid_file(), Duration::from_secs(2), None);
            for line in std::fs::read_to_string(self.dir.join("record")).unwrap_or_default().lines() {
                if let Some(pid) = line.split_whitespace().next().and_then(|p| p.parse::<i32>().ok()) {
                    let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(-pid), nix::sys::signal::Signal::SIGKILL);
                }
            }
            for (var, value) in self.prev.drain(..) {
                match value {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
            let _ = std::fs::remove_dir_all(&self.runtime);
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Reproduced before the lock: 16 concurrent CLI starts spawned four helpers.
    #[test]
    fn concurrent_starts_spawn_exactly_one_helper() {
        let scratch = ScratchStart::new("race");
        let cfg = scratch.config();
        let barrier = std::sync::Barrier::new(16);
        let results: Vec<Result<StartedHelper, StartError>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        start(&cfg)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let started: Vec<i32> = results.iter().filter_map(|r| r.as_ref().ok().map(|h| h.pid)).collect();
        assert_eq!(started.len(), 1, "exactly one start may succeed: {:?}", results.iter().map(|r| r.as_ref().map(|h| h.pid).map_err(ToString::to_string)).collect::<Vec<_>>());
        for r in &results {
            assert!(matches!(r, Ok(_) | Err(StartError::Busy) | Err(StartError::AlreadyRunning(_))), "unexpected: {:?}", r.as_ref().err());
        }
        // Give any second spawn time to show up before counting.
        std::thread::sleep(Duration::from_millis(300));
        let records = scratch.records(1);
        assert_eq!(records.len(), 1, "exactly one helper process may run: {records:?}");
        assert_eq!(is_running(), Some(started[0]));
    }

    #[test]
    fn the_helper_gets_the_configured_channel_and_windows_paths() {
        let scratch = ScratchStart::new("env");
        let cfg = scratch.config();
        // A different channel in the environment must not win over config.ini's.
        std::env::set_var("NEURAL_FORGE_SHM", format!("{}/env-channel/shm.bin", scratch.dir.display()));
        let mut cfg_with_setting = cfg.clone();
        cfg_with_setting.settings.insert("set_intensity".into(), "0.5".into());
        start(&cfg_with_setting).expect("start");
        let records = scratch.records(1);
        let fields: Vec<&str> = records[0].split_whitespace().collect();
        assert_eq!(fields[1], cfg.shm, "the helper runs on the configured channel");
        assert_eq!(fields[2], format!("Z:{}", cfg.log), "the log is a Wine path");
        assert_eq!(fields[3], format!("Z:{}", paths::binaries_dir()), "no binaries= line falls back to the default dir");
        // The saved settings went onto the same channel, not the environment's.
        let mapping = neural_forge_protocol::mapping::open_path(&cfg.shm).unwrap();
        assert_eq!(f32::from_bits(mapping.header().intensity_bits.load(std::sync::atomic::Ordering::Relaxed)), 0.5);
        assert!(!Path::new(&format!("{}/env-channel/shm.bin", scratch.dir.display())).exists());
    }

    #[test]
    fn channel_path_prefers_config_then_environment_then_default() {
        let scratch = ScratchStart::new("channel");
        let mut cfg = Config { shm: "/tmp/neural-forge-x/configured.bin".into(), ..Config::default() };
        std::env::set_var("NEURAL_FORGE_SHM", "/tmp/neural-forge-x/env.bin");
        assert_eq!(channel_path(&cfg), "/tmp/neural-forge-x/configured.bin");
        cfg.shm.clear();
        assert_eq!(channel_path(&cfg), "/tmp/neural-forge-x/env.bin");
        std::env::set_var("NEURAL_FORGE_SHM", "/tmp/dlssnr-1000/shm.bin");
        assert_eq!(channel_path(&cfg), neural_forge_protocol::shm_default_path(), "a path outside the namespace is ignored");
        std::env::remove_var("NEURAL_FORGE_SHM");
        assert_eq!(channel_path(&cfg), format!("{}/shm.bin", scratch.runtime));
    }

    #[test]
    fn proton_without_a_steam_install_fails_to_start() {
        let scratch = ScratchStart::new("nosteam");
        let cfg = Config { runner_type: "proton".to_string(), ..scratch.config() };
        assert!(matches!(start(&cfg), Err(StartError::NoSteamInstall)));
        assert!(scratch.records(0).is_empty() && is_running().is_none());
    }

    #[test]
    fn wineserver_binary_finds_a_real_sibling_next_to_a_proton_runner() {
        let dir = std::env::temp_dir().join(format!("neural-forge-wineserver-test-{}", std::process::id()));
        let wineserver_dir = dir.join("files/bin");
        std::fs::create_dir_all(&wineserver_dir).unwrap();
        let wineserver_path = wineserver_dir.join("wineserver");
        std::fs::write(&wineserver_path, "").unwrap();

        let cfg = Config { runner_type: "proton".to_string(), runner_path: dir.join("proton").display().to_string(), ..Config::default() };
        assert_eq!(wineserver_binary(&cfg), Some(wineserver_path.display().to_string()));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wineserver_binary_is_none_for_proton_with_no_real_sibling() {
        // A path that doesn't exist at all -- no `files/bin/wineserver` to find.
        let cfg = Config { runner_type: "proton".to_string(), runner_path: "/nonexistent/proton".to_string(), ..Config::default() };
        assert_eq!(wineserver_binary(&cfg), None);
    }

    #[test]
    fn wineserver_binary_falls_back_to_path_for_plain_wine() {
        let cfg = Config { runner_type: "wine".to_string(), runner_path: "/usr/bin/wine".to_string(), ..Config::default() };
        assert_eq!(wineserver_binary(&cfg), Some("wineserver".to_string()));
    }

    #[test]
    fn wineserver_binary_is_none_with_no_runner_configured() {
        assert_eq!(wineserver_binary(&Config::default()), None);
    }

    #[test]
    fn real_wineprefix_appends_pfx_for_proton() {
        let cfg = Config { runner_type: "proton".to_string(), ..Config::default() };
        assert_eq!(real_wineprefix(&cfg, "/home/alex/.local/share/neural-forge/prefix"), "/home/alex/.local/share/neural-forge/prefix/pfx");
    }

    #[test]
    fn real_wineprefix_is_unchanged_for_plain_wine() {
        let cfg = Config { runner_type: "wine".to_string(), ..Config::default() };
        assert_eq!(real_wineprefix(&cfg, "/home/alex/.local/share/neural-forge/prefix"), "/home/alex/.local/share/neural-forge/prefix");
    }
}
