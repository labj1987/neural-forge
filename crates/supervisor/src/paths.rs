//! XDG-aware paths, matching upstream's own reasoning (see the plan's `layer`
//! section and upstream's README for why the runtime/SHM path specifically lives
//! under `/tmp`, not `$XDG_RUNTIME_DIR` — that one's `neural_forge_protocol::shm_runtime_dir`,
//! already shared code; everything else here is config/data/state, which has no
//! Steam-container wrinkle to work around.

use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

pub(crate) fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".to_string())
}

fn xdg(var: &str, fallback_under_home: &str) -> String {
    std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/{fallback_under_home}", home()))
}

/// The raw `XDG_DATA_HOME` itself (not the `neural-forge` subdirectory [`data_dir`]
/// returns) -- `install.rs` needs it as-is, matching `scripts/install.py`'s own
/// `data` variable: the installed Vulkan manifests live under the shared per-user data
/// hierarchy's own well-known subdirectory, a sibling of `neural-forge/` rather than inside it.
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

pub fn binaries_dir() -> String {
    format!("{}/binaries", data_dir())
}

pub fn ensure_dirs() -> std::io::Result<()> {
    for dir in [config_dir(), data_dir(), state_dir(), binaries_dir()] {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

/// Writes `content` to `path`, replacing any existing file by renaming a same-directory
/// staged file over it -- never truncates the destination in place, so an
/// already-running process that has the old inode mapped (a GUI, a game) keeps reading the old content until it reopens the path, exactly like
/// `install.py`'s own `os.replace` step. A crash part-way leaves the old file whole, never an
/// empty one: `config.ini`, `profiles.ini` and `dxvk.conf` are written through this too.
pub fn write_atomic(path: &Path, content: &[u8], mode: u32) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| std::io::Error::other(format!("{} has no parent directory", path.display())))?;
    std::fs::create_dir_all(parent)?;
    let mut attempt = 0u32;
    let staged = loop {
        let candidate = parent.join(format!(".neural-forge-{}-{attempt}.tmp", std::process::id()));
        match std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&candidate) {
            Ok(mut file) => {
                file.write_all(content)?;
                file.sync_all()?;
                break candidate;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempt < 1000 => attempt += 1,
            Err(e) => return Err(e),
        }
    };
    std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(mode)).inspect_err(|_| {
        let _ = std::fs::remove_file(&staged);
    })?;
    std::fs::rename(&staged, path)
}

#[cfg(test)]
pub(crate) mod tests {
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
}
