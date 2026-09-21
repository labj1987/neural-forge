//! Shared helper-process supervision: config, XDG paths, and start/stop, used by
//! both `neural-forge-cli` and `neural-forge` so "how to launch the helper" (runner type,
//! environment variables, the Proton-vs-Wine command line) exists in exactly one
//! place instead of being duplicated and risking drift between the two front ends.
//! Linux-only: spawns child processes, reads XDG env vars.

pub mod config;
pub mod gpu;
pub mod install;
pub mod install_dir;
pub mod migrate;
pub mod paths;
pub mod profiles;
pub mod provision;
mod process;
pub mod runners;

pub use config::Config;

use std::time::Duration;

pub fn pid_file() -> String {
    format!("{}/helper.pid", neural_forge_protocol::shm_runtime_dir())
}

/// The pid file a helper started by a pre-0.1.77 install wrote (`/tmp/neuralforge-<uid>/`).
fn legacy_pid_file() -> Option<String> {
    neural_forge_protocol::compat::legacy_runtime_dir().map(|dir| format!("{}/helper.pid", dir.display()))
}

/// The helper's PID if a process is actually alive at the PID in the pid file (or in the
/// pre-0.1.77 one, so a helper started by an older install still counts as running).
pub fn is_running() -> Option<i32> {
    process::running_pid(&pid_file()).or_else(|| legacy_pid_file().and_then(|f| process::running_pid(&f)))
}

/// Graceful-then-forced stop of the whole helper process group.
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
    // "forge-helper" matches both `neural-forge-helper.exe` and the pre-0.1.76 name
    // `neuralforge-helper.exe`, so a helper started by an older install is still stopped.
    // Both runners (plain Wine, Proton) carry the helper's path in their own command
    // line, which is what guards against signaling a process that reused the PID.
    process::stop_matching(&pid_file(), timeout, Some("forge-helper"))?;
    if let Some(legacy) = legacy_pid_file() {
        process::stop_matching(&legacy, timeout, Some("forge-helper"))?;
    }
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
    HelperNotFound,
    NoRunnerConfigured,
    Spawn(std::io::Error),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartError::AlreadyRunning(pid) => write!(f, "helper already running (pid {pid})"),
            StartError::HelperNotFound => write!(f, "neural-forge-helper.exe not found"),
            StartError::NoRunnerConfigured => write!(f, "no runner configured (run `neural-forge-cli init` first)"),
            StartError::Spawn(e) => write!(f, "failed to start helper: {e}"),
        }
    }
}

pub struct StartedHelper {
    pub pid: i32,
    pub runner_type: String,
    pub runner_path: String,
    pub log: String,
}

/// Starts the helper under `cfg`'s configured runner (Proton or plain Wine),
/// building the same environment variables (`WINEPREFIX`, `NEURAL_FORGE_SHM`, `NEURAL_FORGE_UID`,
/// the Proton-specific NVAPI/compat-data ones) either front end needs -- this is the
/// one place that construction happens.
/// Applies the saved settings (`config.ini`) to the live header if nothing has applied them
/// since it was last initialised. The header is re-initialised with defaults by whichever of the
/// layer, helper or GUI finds it missing or at another version -- a game started first, or an
/// upgrade -- and before this only the GUI, and only when it had created the file itself, put
/// the user's settings back. `init_defaults` leaves `tuning_seq` at 0 and `persist::apply` bumps
/// it, which is what "not yet applied" is read from.
pub fn apply_saved_settings(cfg: &Config) {
    let Some(mapping) = neural_forge_protocol::mapping::open() else { return };
    let hdr = mapping.header();
    if hdr.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        neural_forge_protocol::persist::apply(hdr, &cfg.settings);
        hdr.control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn start(cfg: &Config) -> Result<StartedHelper, StartError> {
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

    let mut envs = vec![
        ("WINEPREFIX".to_string(), paths::prefix_dir()),
        ("NEURAL_FORGE_SHM".to_string(), cfg.shm.clone()),
        ("NEURAL_FORGE_LOG".to_string(), cfg.log.clone()),
        ("NEURAL_FORGE_BIN_DIR".to_string(), format!("Z:{}", cfg.binaries)),
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
        // helper never got further than Proton's own setup_prefix() step. Every
        // manual SSH test this project's own history has ever done set this by hand
        // without that fix ever making it back into this function -- this is that fix.
        if let Some(steam_dir) = paths::steam_install_dir() {
            envs.push(("STEAM_COMPAT_CLIENT_INSTALL_PATH".to_string(), steam_dir));
        }
        (cfg.runner_path.clone(), vec!["run".to_string(), helper.display().to_string()])
    } else {
        if cfg.runner_type == "wine" {
            // Plain Wine has no DXVK-NVAPI of its own: the managed prefix is given DXVK's dxgi and
            // DXVK-NVAPI (see `provision`), and the helper loads NVAPI itself.
            if let Err(e) = provision::prepare_wine_prefix(cfg) {
                crate::process::append_log(&cfg.log, &format!("[neural-forge] system-Wine prefix setup failed: {e}"));
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
