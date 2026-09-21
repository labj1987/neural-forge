//! XDG-aware paths, matching upstream's own reasoning (see the plan's `layer`
//! section and upstream's README for why the runtime/SHM path specifically lives
//! under `/tmp`, not `$XDG_RUNTIME_DIR` — that one's `neural_forge_protocol::shm_runtime_dir`,
//! already shared code; everything else here is config/data/state, which has no
//! Steam-container wrinkle to work around.

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
}

fn xdg(var: &str, fallback_under_home: &str) -> String {
    std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/{fallback_under_home}", home()))
}

/// The raw `XDG_DATA_HOME` itself (not the `neural-forge` subdirectory [`data_dir`]
/// returns) -- `install.rs` needs it as-is, matching `scripts/install.py`'s own
/// `data` variable: several installed files (the Vulkan manifest, `.desktop` file,
/// icon, AppStream metainfo) live under the shared per-user data hierarchy's own
/// well-known subdirectories, siblings of `neural-forge/` rather than inside it.
pub fn data_home() -> String {
    xdg("XDG_DATA_HOME", ".local/share")
}

/// The raw `XDG_CONFIG_HOME` (see [`data_home`]).
pub fn config_home() -> String {
    xdg("XDG_CONFIG_HOME", ".config")
}

/// The raw `XDG_STATE_HOME` (see [`data_home`]).
pub fn state_home() -> String {
    xdg("XDG_STATE_HOME", ".local/state")
}

pub fn config_dir() -> String {
    format!("{}/neural-forge", xdg("XDG_CONFIG_HOME", ".config"))
}

pub fn config_file() -> String {
    format!("{}/config.ini", config_dir())
}

pub fn data_dir() -> String {
    format!("{}/neural-forge", xdg("XDG_DATA_HOME", ".local/share"))
}

pub fn state_dir() -> String {
    format!("{}/neural-forge", xdg("XDG_STATE_HOME", ".local/state"))
}

pub fn log_file() -> String {
    format!("{}/helper.log", state_dir())
}

pub fn binaries_dir() -> String {
    format!("{}/binaries", data_dir())
}

/// The managed Wine/Proton prefix `neural-forge` creates and owns, distinct from any
/// prefix a game or Steam manages -- so importing NGX DLLs into it, or a bad prefix
/// state, never touches anything else.
pub fn prefix_dir() -> String {
    format!("{}/neural-forge/prefix", xdg("XDG_DATA_HOME", ".local/share"))
}

pub fn ensure_dirs() -> std::io::Result<()> {
    // `prefix_dir()` included since 2026-09-10: a real, confirmed bug on `lordnikon`
    // -- `start()` passes it as both `WINEPREFIX` and `STEAM_COMPAT_DATA_PATH`, but
    // nothing ever created the directory itself first. Normally invisible (Proton
    // creates everything *inside* it on first successful init, so it already exists
    // on every subsequent run), until the directory is missing for any reason (a
    // fresh install, or this parent being removed/reset by hand) -- then Proton's own
    // `setup_prefix()` fails with a `FileNotFoundError` opening `pfx.lock`, since it
    // assumes the directory it's locking already exists. `start()` itself also
    // creates this directly (see its own comment) so this doesn't depend on whatever
    // called `ensure_dirs()` last having actually run recently.
    for dir in [config_dir(), data_dir(), state_dir(), binaries_dir(), prefix_dir()] {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

/// The real Steam client install root (the directory containing `steamapps/`,
/// `compatibilitytools.d/`, etc.) -- what `STEAM_COMPAT_CLIENT_INSTALL_PATH` needs to
/// point at for Proton's own launch script to run at all (`start()`'s own doc comment
/// explains why this has to be set). Same candidate list `neural-forge-cli`'s own Proton
/// discovery (`runners.rs::candidate_dirs`) already scans for
/// `compatibilitytools.d` -- this just checks the *parent* of each and returns the
/// first that's a real directory, since a real Steam install is what actually creates
/// these paths, in this same order of likelihood (native package first, then the
/// sandboxed variants).
pub fn steam_install_dir() -> Option<String> {
    let xdg_data_home = xdg("XDG_DATA_HOME", ".local/share");
    let home = home();
    [
        format!("{xdg_data_home}/Steam"),
        format!("{home}/.var/app/com.valvesoftware.Steam/data/Steam"),
        format!("{home}/snap/steam/common/.local/share/Steam"),
    ]
    .into_iter()
    .find(|dir| std::path::Path::new(dir).is_dir())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // Guards every test in this crate that mutates `XDG_DATA_HOME` (a real,
    // process-wide environment variable, not something scoped per-test) -- Rust's
    // default test harness runs tests in parallel threads within the same process,
    // so two such tests running concurrently can and did race for real: one test's
    // `assert_eq!` observing the *other* test's own `XDG_DATA_HOME` value mid-flight.
    // Confirmed genuinely intermittent, not a one-off: `cargo test` (workspace-wide,
    // different thread scheduling than running this crate alone) failed roughly 1 run
    // in 3 before this lock existed, 2026-09-11. `pub(crate)` (not private to this
    // module) on purpose: `install::tests` also mutates `XDG_DATA_HOME` and had its
    // own separate lock racing against this one for the same reason, until both were
    // unified onto this single instance, 2026-09-15. Every test anywhere in this
    // crate that touches this env var must acquire this lock for its entire
    // duration, restoring the prior value (or removing it) before releasing.
    pub(crate) static XDG_DATA_HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Real filesystem, real env var override -- confirms `steam_install_dir` actually
    /// finds a directory that exists (not just "returns some string unconditionally"),
    /// and returns `None` when nothing does. Found and fixed as a real bug 2026-09-10:
    /// `start()` never set `STEAM_COMPAT_CLIENT_INSTALL_PATH` at all before this
    /// existed, causing a real `KeyError` crash in Proton's own script on every single
    /// start attempt with `runner_type = "proton"`.
    #[test]
    fn finds_a_real_steam_install_under_xdg_data_home() {
        let _guard = XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let scratch = std::env::temp_dir().join(format!("neural-forge-steam-detect-test-{}", std::process::id()));
        let steam_dir = scratch.join("Steam");
        std::fs::create_dir_all(&steam_dir).unwrap();

        let prev = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &scratch);

        let found = steam_install_dir();
        assert_eq!(found.as_deref(), steam_dir.to_str());

        match prev {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        std::fs::remove_dir_all(&scratch).ok();
    }

    #[test]
    fn returns_none_when_no_candidate_exists() {
        let _guard = XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let scratch = std::env::temp_dir().join(format!("neural-forge-steam-detect-test-none-{}", std::process::id()));
        // Deliberately do not create `scratch` itself -- every candidate under it is
        // real-but-nonexistent, the case this function must fail open on.

        let prev = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &scratch);

        assert_eq!(steam_install_dir(), None);

        match prev {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
    }
}
