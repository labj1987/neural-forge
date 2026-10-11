//! The layer's state log: a small file of state transitions and errors that outlives the game, so
//! `neural-forge-cli doctor` and the GUI can say what went wrong in the last session without a log
//! having been asked for.
//!
//! Only transitions are written (the layer engaging, a copy staying inert, the device lost, the
//! device unable to run the network, the network failing or coming back, a fence wait timing out),
//! each at most once per process or once per change, never once per frame: the present path pays
//! nothing for it in steady state (see the layer's `logging.rs` on why that matters). The layer
//! writes synchronously, one `write` per line, and only from those transition sites.
//!
//! Where it lives: `$XDG_STATE_HOME/neural-forge/layer.log`, or `$HOME/.local/state/...` without
//! the variable, the same rule the supervisor's `paths::state_dir` uses. Inside Steam's runtime
//! container (pressure-vessel) the game keeps the user's real home directory and the `XDG_*`
//! variables it was started with, so the layer and the GUI name the same file; the shared-memory
//! channel crosses the container under `/tmp` instead (`path.rs`), which does not survive a reboot,
//! and a GPU hang is exactly when a reboot follows. A Flatpak Steam points `XDG_STATE_HOME` into its
//! own sandbox; the file is then there, not where `doctor` looks.
//!
//! The file is rotated to `layer.log.1` when it reaches [`ROTATE_BYTES`].
//!
//! Grammar, one line per event, tab-separated: `<unix seconds>\t<pid>\t<process>\t<kind>\t<message>`.
//! `kind` is one of the constants below. Readers match on the exact kind, never on a substring of
//! the message.

use std::io::Write;
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "layer.log";
pub const ROTATED_NAME: &str = "layer.log.1";
/// The size at which the file is rotated: years of transitions, and small enough to read whole.
pub const ROTATE_BYTES: u64 = 256 * 1024;

/// The layer engaged in a game for the first time in that process (it was rendering steadily).
pub const ATTACH: &str = "attach";
/// Another copy of the layer is live in the process; this one stays inert.
pub const DUPLICATE: &str = "duplicate";
/// A layer-issued Vulkan call returned `VK_ERROR_DEVICE_LOST`; the layer is inert from then on.
pub const DEVICE_LOST: &str = "device-lost";
/// The device lacks something the network needs; the message is `device cannot run the network: <first missing>`.
pub const NATIVE_UNAVAILABLE: &str = "native-unavailable";
/// Loading or building the network failed (each different reason once).
pub const NATIVE_FAILED: &str = "native-failed";
/// The network was built after a failure.
pub const NATIVE_RECOVERED: &str = "native-recovered";
/// A bounded fence wait timed out (a driver stall without device loss); the message carries the site
/// and the breadcrumb trail.
pub const FENCE_TIMEOUT: &str = "fence-timeout";

pub const KINDS: [&str; 7] = [ATTACH, DUPLICATE, DEVICE_LOST, NATIVE_UNAVAILABLE, NATIVE_FAILED, NATIVE_RECOVERED, FENCE_TIMEOUT];

/// The prefix of a [`NATIVE_UNAVAILABLE`] message, and of the layer's status line for the same case.
pub const CANNOT_RUN_PREFIX: &str = "device cannot run the network: ";

/// The state log's path from the environment's values (`XDG_STATE_HOME`, `HOME`), `None` when
/// neither names a directory.
pub fn path(xdg_state_home: Option<&str>, home: Option<&str>) -> Option<PathBuf> {
    let base = match xdg_state_home.filter(|s| !s.is_empty()) {
        Some(state) => PathBuf::from(state),
        None => Path::new(home.filter(|h| !h.is_empty())?).join(".local/state"),
    };
    Some(base.join("neural-forge").join(FILE_NAME))
}

/// One parsed line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub time: u64,
    pub pid: u32,
    pub process: String,
    pub kind: String,
    pub message: String,
}

/// Formats one line (with its newline). Tabs and newlines inside the fields become spaces, so a
/// line always parses back into the same five fields.
pub fn format_line(time: u64, pid: u32, process: &str, kind: &str, message: &str) -> String {
    let clean = |s: &str| s.replace(['\t', '\n', '\r'], " ");
    format!("{time}\t{pid}\t{}\t{kind}\t{}\n", clean(process), clean(message))
}

/// Parses one line; `None` for anything that is not exactly the grammar (a torn last line, an
/// unknown kind, a line from something else).
pub fn parse_line(line: &str) -> Option<Event> {
    let mut fields = line.trim_end_matches(['\n', '\r']).splitn(5, '\t');
    let time = fields.next()?.parse().ok()?;
    let pid = fields.next()?.parse().ok()?;
    let process = fields.next()?.to_string();
    let kind = fields.next()?;
    let message = fields.next()?.to_string();
    KINDS.contains(&kind).then(|| Event { time, pid, process, kind: kind.to_string(), message })
}

/// Every parseable event in `dir`'s rotated file and then its current one, oldest first.
pub fn read_dir_events(dir: &Path) -> std::io::Result<Vec<Event>> {
    let mut text = std::fs::read_to_string(dir.join(ROTATED_NAME)).unwrap_or_default();
    match std::fs::read_to_string(dir.join(FILE_NAME)) {
        Ok(current) => text.push_str(&current),
        Err(e) if text.is_empty() => return Err(e),
        Err(_) => {}
    }
    Ok(text.lines().filter_map(parse_line).collect())
}

/// Appends `line` to the file at `path`, rotating it to [`ROTATED_NAME`] first when it has reached
/// `rotate_at` bytes. Creates the directory (0700) and the file (0600). One `write` per line with
/// `O_APPEND`, so two processes appending at once never interleave inside a line.
pub fn append(path: &Path, line: &str, rotate_at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let dir = path.parent().ok_or_else(|| std::io::Error::other("the state log has no directory"))?;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
    if std::fs::metadata(path).is_ok_and(|m| m.len() >= rotate_at) {
        std::fs::rename(path, dir.join(ROTATED_NAME))?;
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(path)?;
    file.write_all(line.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neural-forge-state-log-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn path_prefers_xdg_state_home_then_home() {
        assert_eq!(path(Some("/s"), Some("/h")), Some(PathBuf::from("/s/neural-forge/layer.log")));
        assert_eq!(path(Some(""), Some("/h")), Some(PathBuf::from("/h/.local/state/neural-forge/layer.log")));
        assert_eq!(path(None, Some("/h")), Some(PathBuf::from("/h/.local/state/neural-forge/layer.log")));
        assert_eq!(path(None, None), None);
        assert_eq!(path(None, Some("")), None);
    }

    #[test]
    fn a_line_round_trips_and_stray_tabs_cannot_add_fields() {
        let line = format_line(1_760_000_000, 4242, "Hogwarts\tLegacy.exe", DEVICE_LOST, "lost\tat\nsubmit");
        assert_eq!(line.matches('\t').count(), 4);
        assert!(line.ends_with('\n') && line.matches('\n').count() == 1);
        let e = parse_line(&line).unwrap();
        assert_eq!((e.time, e.pid, e.process.as_str(), e.kind.as_str(), e.message.as_str()), (1_760_000_000, 4242, "Hogwarts Legacy.exe", DEVICE_LOST, "lost at submit"));
    }

    #[test]
    fn only_the_exact_grammar_parses() {
        // A kind that merely contains a known one, a torn line, free text mentioning a kind.
        assert_eq!(parse_line("1\t2\tgame\tdevice-lost-ish\tx"), None);
        assert_eq!(parse_line("1\t2\tgame\tdevice-lost"), None);
        assert_eq!(parse_line("[neural-forge-layer] [layer] VK_ERROR_DEVICE_LOST; device-lost"), None);
        assert_eq!(parse_line("x\t2\tgame\tdevice-lost\tm"), None);
        assert!(parse_line("1\t2\tgame\tdevice-lost\t").is_some());
    }

    #[test]
    fn append_rotates_at_the_limit_and_reads_back_oldest_first() {
        let dir = scratch("rotate");
        let file = dir.join("neural-forge").join(FILE_NAME);
        append(&file, &format_line(1, 1, "a", ATTACH, "first"), 40).unwrap();
        append(&file, &format_line(2, 1, "a", NATIVE_FAILED, "second"), 40).unwrap();
        // The file now holds two lines, over 40 bytes: the third goes into a fresh file.
        append(&file, &format_line(3, 1, "a", NATIVE_RECOVERED, "third"), 40).unwrap();
        let state = dir.join("neural-forge");
        assert!(state.join(ROTATED_NAME).exists());
        let events = read_dir_events(&state).unwrap();
        assert_eq!(events.iter().map(|e| e.time).collect::<Vec<_>>(), [1, 2, 3]);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_log_is_an_error_but_a_missing_rotation_is_not() {
        let dir = scratch("missing");
        assert!(read_dir_events(&dir).is_err());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(FILE_NAME), format_line(5, 9, "g", ATTACH, "m")).unwrap();
        assert_eq!(read_dir_events(&dir).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
