//! Installing an extracted AppImage AppDir into persistent user storage, and
//! uninstalling it again -- the same operation, the same `installation.json`
//! provenance record format (a flat `path -> sha256` map, refusing to touch any file
//! that record doesn't say this installer itself last wrote), and the same
//! atomic-replace-never-truncate file writes as `scripts/install.py`. Reimplemented
//! in Rust so the GUI can run it on every launch from an AppImage (a no-op when the
//! installed copy is already current) without a `python3` dependency; both tools
//! read/write the exact same record on purpose, so either can pick up after the other.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::paths;
use crate::paths::write_atomic;

pub const APP_ID: &str = "io.github.labj1987.NeuralForge";
pub const LAYER: &str = "VK_LAYER_neuralforge_neural";
/// The installed manifest's file name (the loader reads every `*.json` in
/// `implicit_layer.d`, so it need not match the layer name). Named after the layer library.
pub const MANIFEST: &str = "neural_forge_layer.json";

#[derive(Debug)]
pub enum InstallError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidManifest(String),
    RefusedSymlink(PathBuf),
    /// A symlink in the AppDir's files: following it could copy a tree from anywhere (or loop).
    RefusedSourceSymlink(PathBuf),
    RefusedUnowned(PathBuf),
    CorruptRecord(CorruptRecord),
}

/// `installation.json` exists but is not a record this installer can read.
#[derive(Debug)]
pub struct CorruptRecord {
    pub path: PathBuf,
    pub reason: String,
}

impl std::fmt::Display for CorruptRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the install record {} cannot be read ({}). Move it aside (gio trash {}) and install again: \
             installed files that match the new version are taken over, and any other file in the way is named",
            self.path.display(),
            self.reason,
            self.path.display()
        )
    }
}

impl From<CorruptRecord> for std::io::Error {
    fn from(e: CorruptRecord) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    }
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::InvalidManifest(msg) => write!(f, "{msg}"),
            Self::RefusedSymlink(path) => write!(f, "refusing symlink destination: {}", path.display()),
            Self::RefusedSourceSymlink(path) => write!(f, "refusing a symlink in the AppDir: {}", path.display()),
            Self::RefusedUnowned(path) => write!(f, "refusing to overwrite unowned or changed file: {}", path.display()),
            Self::CorruptRecord(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for InstallError {}
impl From<std::io::Error> for InstallError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for InstallError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[derive(Debug)]
pub struct InstallReport {
    /// False when every installed file already matched the AppDir, so nothing was written.
    pub changed: bool,
    pub root: PathBuf,
    pub gui_path: PathBuf,
    pub cli_path: PathBuf,
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    // Lowercase hex, as Python's hexdigest() writes it (sha2 0.11's output has no LowerHex).
    let mut hex = String::with_capacity(64);
    for byte in hasher.finalize().iter() {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn digest_file(path: &Path) -> std::io::Result<String> {
    Ok(digest(&std::fs::read(path)?))
}

fn record_path() -> PathBuf {
    PathBuf::from(paths::data_dir()).join("installation.json")
}

/// Why the install record could not be used.
#[derive(Debug)]
enum RecordError {
    /// It is there but could not be read (permissions, an I/O error): it may be fine, so nothing
    /// may act as if there were none.
    Unreadable(std::io::Error),
    /// It was read and is not a record.
    Corrupt(CorruptRecord),
}

impl From<RecordError> for InstallError {
    fn from(e: RecordError) -> Self {
        match e {
            RecordError::Unreadable(e) => InstallError::Io(e),
            RecordError::Corrupt(e) => InstallError::CorruptRecord(e),
        }
    }
}

impl From<RecordError> for std::io::Error {
    fn from(e: RecordError) -> Self {
        match e {
            RecordError::Unreadable(e) => e,
            RecordError::Corrupt(e) => e.into(),
        }
    }
}

/// The install record: empty when there is none yet, an error when one is there but cannot be
/// read (silently starting from an empty record would refuse every installed file as unowned).
fn load_record() -> Result<BTreeMap<String, String>, RecordError> {
    let path = record_path();
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(RecordError::Unreadable(std::io::Error::new(e.kind(), format!("the install record {} cannot be read: {e}", path.display())))),
    };
    serde_json::from_slice(&bytes).map_err(|e| RecordError::Corrupt(CorruptRecord { path, reason: e.to_string() }))
}

/// The installed layer manifest, the one file the Vulkan loader finds on its own.
fn manifest_path() -> PathBuf {
    PathBuf::from(paths::data_home()).join(format!("vulkan/implicit_layer.d/{MANIFEST}"))
}

/// Whether a recorded path is one this installer may write or remove: under the data dir or
/// the implicit-layer directory, with no `..` to step out again. Anything else in the record
/// (a hand-edited or foreign record) is never touched.
fn is_owned_path(path: &Path) -> bool {
    use std::path::Component;
    let plain = path.is_absolute() && path.components().all(|c| matches!(c, Component::RootDir | Component::Normal(_)));
    let layer_dir = PathBuf::from(paths::data_home()).join("vulkan/implicit_layer.d");
    plain && (path.starts_with(paths::data_dir()) || path.starts_with(layer_dir))
}

/// Splits the record into the entries this installer owns and the names of any it does not.
fn owned_record(record: BTreeMap<String, String>) -> (BTreeMap<String, String>, Vec<String>) {
    let (owned, foreign): (BTreeMap<_, _>, BTreeMap<_, _>) = record.into_iter().partition(|(name, _)| is_owned_path(Path::new(name)));
    (owned, foreign.into_keys().collect())
}

fn save_record(record: &BTreeMap<String, String>) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(record).expect("a string-keyed map always serializes") + "\n";
    write_atomic(&record_path(), text.as_bytes(), 0o644)
}

/// Every regular file under `dir` (none when it is not a directory). A symlink anywhere below it is
/// refused, not followed: a link to a directory elsewhere would copy that tree into the install and
/// record it as owned, and a link loop would recurse until the stack overflows.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), InstallError> {
    match std::fs::symlink_metadata(dir) {
        Ok(m) if m.is_symlink() => return Err(InstallError::RefusedSourceSymlink(dir.to_path_buf())),
        Ok(m) if m.is_dir() => {}
        _ => return Ok(()),
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_symlink() {
            return Err(InstallError::RefusedSourceSymlink(path));
        } else if kind.is_dir() {
            collect_files(&path, out)?;
        } else if kind.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

/// Installs `appdir` (an extracted AppImage's `AppDir`) into persistent user storage.
/// Refuses (writing nothing at all) if any destination is a symlink, or already
/// exists with content this installer's own record doesn't recognize as what it last
/// wrote there -- the same "never touch a file this didn't put there" guarantee
/// `install.py` makes.
pub fn install(appdir: &Path) -> Result<InstallReport, InstallError> {
    let root = PathBuf::from(paths::data_dir());
    let (old, foreign) = owned_record(load_record()?);
    let usr = appdir.join("usr");

    let mut files: BTreeMap<PathBuf, Vec<u8>> = BTreeMap::new();

    let lib_src_root = usr.join("lib/neural-forge");
    let mut lib_files = Vec::new();
    collect_files(&lib_src_root, &mut lib_files)?;
    for src in lib_files {
        let rel = src.strip_prefix(&lib_src_root).expect("collect_files only yields paths under lib_src_root");
        files.insert(root.join("lib/neural-forge").join(rel), std::fs::read(&src)?);
    }

    for binary in ["neural-forge", "neural-forge-cli"] {
        files.insert(root.join("bin").join(binary), std::fs::read(usr.join("bin").join(binary))?);
    }

    let manifest_src = usr.join(format!("share/vulkan/implicit_layer.d/{MANIFEST}"));
    let mut manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest_src)?)?;
    let layer_name = manifest.get("layer").and_then(|l| l.get("name")).and_then(|n| n.as_str()).map(str::to_owned);
    if layer_name.as_deref() != Some(LAYER) {
        return Err(InstallError::InvalidManifest(format!("{}: expected layer.name {LAYER:?}, found {layer_name:?}", manifest_src.display())));
    }
    let library_path = root.join("lib/neural-forge/libneural_forge_layer.so");
    manifest["layer"]["library_path"] = serde_json::Value::String(library_path.to_string_lossy().into_owned());
    let manifest_out = serde_json::to_string_pretty(&manifest)? + "\n";
    let manifest_dest = manifest_path();
    files.insert(manifest_dest.clone(), manifest_out.into_bytes());

    // No `.desktop` file, icon or AppStream metainfo: every install comes from an
    // AppImage, and menu integration belongs to whatever integrates that AppImage
    // (Gear Lever, AppImageLauncher, ...). Installing a second entry here gave users two
    // "Neural Forge" launchers.

    // Validate every destination before writing any of them, exactly like
    // `install.py`: a failure partway through must leave nothing changed, not a
    // half-installed mix of old and new files. A file already holding exactly what is about to
    // be written is adopted: an install interrupted between writing a file and recording it
    // left it there, and refusing it would block every later install.
    let mut adopted = BTreeMap::new();
    for (path, content) in &files {
        // The destination and every directory between it and the XDG data home: a symlinked
        // $HOME or ~/.local/share is an ordinary setup (refusing it refused every launch), but a
        // symlinked `neural-forge`, `lib` or `vulkan` below it would send the install elsewhere.
        if let Some(link) = symlink_below_data_home(path) {
            return Err(InstallError::RefusedSymlink(link));
        }
        if path.exists() {
            let current = digest_file(path)?;
            let name = path.to_string_lossy().into_owned();
            if old.get(&name) != Some(&current) {
                if current != digest(content) {
                    return Err(InstallError::RefusedUnowned(path.clone()));
                }
                adopted.insert(name, current);
            }
        }
    }

    // Already current (the GUI runs this on every launch from an AppImage): every file is
    // in place with exactly this content, and nothing stale is left to remove.
    let up_to_date = files.iter().all(|(path, content)| path.is_file() && old.get(&path.to_string_lossy().into_owned()) == Some(&digest(content)))
        && old.keys().all(|name| files.contains_key(Path::new(name)))
        && foreign.is_empty();
    if up_to_date {
        return Ok(InstallReport { changed: false, root: root.clone(), gui_path: root.join("bin/neural-forge"), cli_path: root.join("bin/neural-forge-cli") });
    }

    // Ownership is recorded *as files land*, not once at the end: a failure partway
    // through must leave every file already written tracked (otherwise the next run
    // sees them as unowned and refuses to touch them). Each file's record entry is
    // saved before the next file is written, so an interruption can orphan at most the
    // single file in flight. Entries outside the owned directories are dropped here, never
    // acted on. The layer manifest goes last, once everything it points at is in place.
    let bin_dir = root.join("bin");
    let mut record = old.clone();
    record.extend(adopted);
    save_record(&record)?;
    for (path, content) in files.iter().filter(|(path, _)| **path != manifest_dest) {
        let mode = if path.parent() == Some(bin_dir.as_path()) { 0o755 } else { 0o644 };
        write_atomic(path, content, mode)?;
        record.insert(path.to_string_lossy().into_owned(), digest(content));
        save_record(&record)?;
    }

    // Anything an older install shipped that this one no longer does is stale: remove
    // it if it is still exactly what this installer wrote, and stop tracking it either
    // way (a file someone has since edited is theirs now, left in place).
    let current: std::collections::BTreeSet<String> = files.keys().map(|p| p.to_string_lossy().into_owned()).collect();
    let stale: Vec<(String, String)> = old.iter().filter(|(name, _)| !current.contains(*name)).map(|(n, d)| (n.clone(), d.clone())).collect();
    for (name, recorded) in stale {
        let path = PathBuf::from(&name);
        if removable(&path) && digest_file(&path).ok().as_ref() == Some(&recorded) {
            std::fs::remove_file(&path)?;
        }
        record.remove(&name);
        save_record(&record)?;
        if let Some(parent) = path.parent() {
            prune_empty_dirs(parent, &root);
        }
    }

    write_atomic(&manifest_dest, &files[&manifest_dest], 0o644)?;
    record.insert(manifest_dest.to_string_lossy().into_owned(), digest(&files[&manifest_dest]));
    save_record(&record)?;

    Ok(InstallReport { changed: true, root: root.clone(), gui_path: root.join("bin/neural-forge"), cli_path: root.join("bin/neural-forge-cli") })
}

/// The first symlink among `path` and its ancestors below the XDG data home, if any.
fn symlink_below_data_home(path: &Path) -> Option<PathBuf> {
    let data_home = PathBuf::from(paths::data_home());
    path.ancestors().take_while(|a| *a != data_home && a.starts_with(&data_home)).find(|a| a.is_symlink()).map(Path::to_path_buf)
}

/// Whether the file at `path` may be removed: a regular file, not a symlink, with no symlink
/// between it and the XDG data home ([`symlink_below_data_home`], the check every write makes): a
/// symlinked `neural-forge`, `lib` or `vulkan` directory would send the removal somewhere else.
fn removable(path: &Path) -> bool {
    path.is_file() && !path.is_symlink() && symlink_below_data_home(path).is_none()
}

/// Whether the file at `path` is a layer manifest naming this project's layer, and [`removable`].
fn is_our_manifest(path: &Path) -> bool {
    removable(path)
        && std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .is_some_and(|m| m.get("layer").and_then(|l| l.get("name")).and_then(|n| n.as_str()) == Some(LAYER))
}

/// What an install puts in place, by location alone: for an uninstall without a usable record
/// (missing, or unreadable), so a lost record never leaves a layer every game still loads.
fn remove_known_install() -> std::io::Result<()> {
    let manifest = manifest_path();
    if is_our_manifest(&manifest) {
        std::fs::remove_file(&manifest)?;
        remove_empty_parents(&manifest);
    }
    let root = PathBuf::from(paths::data_dir());
    for file in [root.join("bin/neural-forge"), root.join("bin/neural-forge-cli")] {
        if removable(&file) {
            std::fs::remove_file(&file)?;
            remove_empty_parents(&file);
        }
    }
    let lib = root.join("lib/neural-forge");
    if lib.is_dir() && symlink_below_data_home(&lib).is_none() {
        std::fs::remove_dir_all(&lib)?;
        remove_empty_parents(&lib.join("x"));
    }
    Ok(())
}

/// Removes now-empty directories from `dir` upward, stopping at (and keeping) `stop`.
fn prune_empty_dirs(dir: &Path, stop: &Path) {
    let mut current = dir;
    while current != stop && current.starts_with(stop) && std::fs::remove_dir(current).is_ok() {
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
}

/// Removes every tracked file whose on-disk content still matches this installer's
/// own record (a file the user or another program has since changed is left alone,
/// reported as preserved rather than silently deleted), then clears the record. The layer
/// manifest goes first, so no game loads a layer whose files are half gone. A record entry
/// outside the directories this installer owns is skipped: neither removed nor reported, and so is
/// one reached through a symlinked directory ([`removable`]), reported as preserved. Without a
/// record, or with one that is not a record (corrupt), what an install puts in place is removed by
/// its location ([`remove_known_install`]): the layer manifest (when it names this layer), the
/// binaries and the layer's library directory. A record that is there but cannot be read (an I/O
/// error, permissions) is an error: nothing is removed, and the record is kept.
pub fn uninstall() -> std::io::Result<Vec<PathBuf>> {
    let record = match load_record() {
        Ok(record) if !record.is_empty() => record,
        Err(e @ RecordError::Unreadable(_)) => return Err(e.into()),
        Ok(_) | Err(RecordError::Corrupt(_)) => {
            remove_known_install()?;
            let record = record_path();
            if record.exists() {
                std::fs::remove_file(record)?;
            }
            return Ok(Vec::new());
        }
    };
    let (old, _foreign) = owned_record(record);
    let manifest = manifest_path().to_string_lossy().into_owned();
    let mut preserved = Vec::new();
    let ordered = old.get_key_value(&manifest).into_iter().chain(old.iter().filter(|(name, _)| **name != manifest));
    for (name, expected) in ordered {
        let path = PathBuf::from(name);
        let matches = removable(&path) && digest_file(&path).ok().as_ref() == Some(expected);
        if matches {
            std::fs::remove_file(&path)?;
            remove_empty_parents(&path);
        } else {
            preserved.push(path);
        }
    }
    let record = record_path();
    if record.exists() {
        std::fs::remove_file(record)?;
    }
    Ok(preserved)
}

/// Removes `path`'s parent directories while they are empty, stopping at the XDG base
/// directories themselves (never removes `~/.local/share` and the like).
fn remove_empty_parents(path: &Path) {
    let stops: Vec<PathBuf> = [crate::paths::data_home(), crate::paths::config_home(), crate::paths::home()]
        .into_iter()
        .map(PathBuf::from)
        .collect();
    let mut dir = path.parent();
    while let Some(d) = dir {
        if stops.iter().any(|s| s == d) || std::fs::remove_dir(d).is_err() {
            break; // a base dir, or not empty
        }
        dir = d.parent();
    }
}

/// `uninstall`, then everything else Neural Forge ever wrote: its config, data (the imported
/// NVIDIA DLL and the extracted model included), state and `/tmp/neural-forge-$UID`. Each directory
/// must be one of Neural Forge's own (named `neural-forge` or `neural-forge-<uid>`) or it is left
/// alone. Returns what was removed.
pub fn purge() -> std::io::Result<Vec<PathBuf>> {
    let preserved = uninstall()?;
    for path in &preserved {
        if removable(path) {
            let _ = std::fs::remove_file(path);
            remove_empty_parents(path);
        }
    }
    // Whatever the record said: a manifest naming this layer would point every game at a library
    // that is about to be deleted.
    let manifest = manifest_path();
    if is_our_manifest(&manifest) {
        std::fs::remove_file(&manifest)?;
        remove_empty_parents(&manifest);
    }
    let mut removed = Vec::new();
    let runtime = neural_forge_protocol::shm_runtime_dir();
    for dir in [crate::paths::config_dir(), crate::paths::data_dir(), crate::paths::state_dir(), runtime] {
        let dir = PathBuf::from(dir);
        let ours = dir
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "neural-forge" || n.starts_with("neural-forge-"));
        if !ours || !dir.is_dir() || dir.is_symlink() {
            continue;
        }
        std::fs::remove_dir_all(&dir)?;
        removed.push(dir);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ScratchDataHome {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Option<String>,
        prev_extra: Vec<(&'static str, Option<String>)>,
        dir: PathBuf,
    }

    impl ScratchDataHome {
        fn new(tag: &str) -> Self {
            // Shares `paths::tests`' own lock, not a separate one -- both mutate the
            // same process-wide `XDG_DATA_HOME`, and two independent locks around one
            // env var don't actually exclude each other. See that lock's own doc
            // comment for the real race this guards against.
            let guard = crate::paths::tests::XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-install-test-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let prev = std::env::var("XDG_DATA_HOME").ok();
            std::env::set_var("XDG_DATA_HOME", &dir);
            // Keep the tests away from the real config/state dirs and runtime dir: a unique uid
            // names a runtime dir that cannot exist.
            let mut prev_extra = Vec::new();
            for (var, value) in [
                ("XDG_CONFIG_HOME", dir.join("config-home").display().to_string()),
                ("XDG_STATE_HOME", dir.join("state-home").display().to_string()),
                ("NEURAL_FORGE_UID", format!("installtest-{}", std::process::id())),
            ] {
                prev_extra.push((var, std::env::var(var).ok()));
                std::env::set_var(var, value);
            }
            Self { _guard: guard, prev, prev_extra, dir }
        }
    }

    impl Drop for ScratchDataHome {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var("XDG_DATA_HOME", v),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
            for (var, value) in self.prev_extra.drain(..) {
                match value {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    /// Builds the same minimal fixture `scripts/test_install.py` uses, so both
    /// implementations are exercised against the same shape of AppDir.
    fn write_fixture_appdir(appdir: &Path) {
        let files = [
            ("usr/bin/neural-forge", "gui"),
            ("usr/bin/neural-forge-cli", "cli"),
            ("usr/lib/neural-forge/libneural_forge_layer.so", "layer"),
            (&format!("usr/share/applications/{APP_ID}.desktop"), "[Desktop Entry]\nExec=neural-forge\n"),
            ("usr/share/icons/hicolor/scalable/apps/neural-forge.svg", "<svg/>"),
            (&format!("usr/share/metainfo/{APP_ID}.appdata.xml"), "<component/>"),
            (&format!("usr/share/vulkan/implicit_layer.d/{MANIFEST}"), r#"{"layer": {"name": "VK_LAYER_neuralforge_neural"}}"#),
        ];
        for (rel, content) in files {
            let path = appdir.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, content).unwrap();
        }
    }

    #[test]
    fn install_writes_every_expected_destination_and_rewrites_the_manifest_path() {
        let scratch = ScratchDataHome::new("basic");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);

        let report = install(&appdir).expect("install should succeed against a well-formed AppDir");
        let root = PathBuf::from(paths::data_dir());
        assert_eq!(report.root, root);
        assert_eq!(std::fs::read_to_string(root.join("bin/neural-forge")).unwrap(), "gui");
        assert_eq!(std::fs::read_to_string(root.join("bin/neural-forge-cli")).unwrap(), "cli");
        assert_eq!(std::fs::read_to_string(root.join("lib/neural-forge/libneural_forge_layer.so")).unwrap(), "layer");

        let manifest_path = PathBuf::from(paths::data_home()).join(format!("vulkan/implicit_layer.d/{MANIFEST}"));
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["layer"]["name"], LAYER);
        assert_eq!(manifest["layer"]["library_path"], root.join("lib/neural-forge/libneural_forge_layer.so").to_string_lossy().into_owned());

        let data_home = PathBuf::from(paths::data_home());
        for integration in [format!("applications/{APP_ID}.desktop"), "icons/hicolor/scalable/apps/neural-forge.svg".into(), format!("metainfo/{APP_ID}.appdata.xml")] {
            assert!(!data_home.join(&integration).exists(), "{integration}: menu integration belongs to the AppImage, not the install");
        }

        assert!(record_path().exists());
    }

    #[test]
    fn a_record_entry_outside_the_owned_dirs_is_dropped_and_never_touched() {
        let scratch = ScratchDataHome::new("foreign-entry");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let outside = scratch.dir.join("not-ours.txt");
        let sneaky = PathBuf::from(paths::data_dir()).join("../not-ours-either.txt");
        for path in [&outside, &sneaky] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "someone else's").unwrap();
        }
        let mut record = BTreeMap::new();
        for path in [&outside, &sneaky] {
            record.insert(path.to_string_lossy().into_owned(), digest(b"someone else's"));
        }
        save_record(&record).unwrap();

        install(&appdir).unwrap();
        let record = load_record().unwrap();
        for path in [&outside, &sneaky] {
            assert!(path.exists(), "{}: a stale entry outside the owned dirs must not be removed", path.display());
            assert!(!record.contains_key(&path.to_string_lossy().into_owned()), "{}: must be dropped from the record", path.display());
        }

        // Uninstall skips such an entry too: it is neither removed nor reported as preserved
        // (purge deletes what is reported).
        let mut record = load_record().unwrap();
        record.insert(outside.to_string_lossy().into_owned(), digest(b"someone else's"));
        save_record(&record).unwrap();
        let preserved = uninstall().unwrap();
        assert!(outside.exists());
        assert!(!preserved.contains(&outside), "{preserved:?}");
        assert!(!manifest_path().exists());
    }

    #[test]
    fn a_file_left_unrecorded_by_an_interrupted_install_is_adopted() {
        let scratch = ScratchDataHome::new("orphan");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();
        // An install interrupted after writing a file but before recording it.
        let binary = PathBuf::from(paths::data_dir()).join("bin/neural-forge");
        let mut record = load_record().unwrap();
        record.remove(&binary.to_string_lossy().into_owned());
        save_record(&record).unwrap();

        install(&appdir).expect("a file holding exactly the new content must not block the install");
        assert_eq!(load_record().unwrap().get(&binary.to_string_lossy().into_owned()), Some(&digest(b"gui")));

        // A different unrecorded file in the way is still refused.
        let mut record = load_record().unwrap();
        record.remove(&binary.to_string_lossy().into_owned());
        save_record(&record).unwrap();
        std::fs::write(&binary, "someone else's").unwrap();
        let err = install(&appdir).expect_err("an unowned file with other content must be refused");
        assert!(matches!(err, InstallError::RefusedUnowned(ref path) if path == &binary), "unexpected error: {err}");
    }

    #[test]
    fn a_corrupt_record_is_a_distinct_error_that_says_what_to_do() {
        let scratch = ScratchDataHome::new("corrupt-record");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();
        std::fs::write(record_path(), "{ not json").unwrap();

        let err = install(&appdir).expect_err("an unreadable record must not read as empty");
        assert!(matches!(err, InstallError::CorruptRecord(_)), "unexpected error: {err}");
        let message = err.to_string();
        assert!(message.contains(&record_path().display().to_string()) && message.contains("Move it aside"), "{message}");

        // Uninstall cannot use it, but still removes what an install puts in place, by location.
        uninstall().expect("uninstall works without a readable record");
        assert!(!PathBuf::from(paths::data_dir()).join("bin/neural-forge").exists());
        assert!(!manifest_path().exists());
    }

    #[test]
    fn install_removes_files_a_newer_version_no_longer_ships() {
        let scratch = ScratchDataHome::new("stale-cleanup");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let extra = appdir.join("usr/lib/neural-forge/old-only.so");
        std::fs::write(&extra, "old").unwrap();
        install(&appdir).unwrap();
        let installed = PathBuf::from(paths::data_dir()).join("lib/neural-forge/old-only.so");
        assert!(installed.exists());

        std::fs::remove_file(&extra).unwrap();
        install(&appdir).unwrap();
        assert!(!installed.exists(), "a file the new install no longer ships must be removed");
        assert!(!load_record().unwrap().contains_key(&installed.to_string_lossy().into_owned()));
    }

    #[test]
    fn stale_file_the_user_edited_is_left_alone_but_untracked() {
        let scratch = ScratchDataHome::new("stale-edited");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let extra = appdir.join("usr/lib/neural-forge/old-only.so");
        std::fs::write(&extra, "old").unwrap();
        install(&appdir).unwrap();
        let installed = PathBuf::from(paths::data_dir()).join("lib/neural-forge/old-only.so");
        std::fs::write(&installed, "edited by hand").unwrap();

        std::fs::remove_file(&extra).unwrap();
        install(&appdir).unwrap();
        assert_eq!(std::fs::read_to_string(&installed).unwrap(), "edited by hand");
        assert!(!load_record().unwrap().contains_key(&installed.to_string_lossy().into_owned()));
    }

    /// A failure partway through writing must leave the files already written tracked,
    /// so the next attempt can proceed instead of refusing them as unowned.
    #[test]
    fn a_failure_partway_leaves_written_files_tracked_and_the_retry_succeeds() {
        let scratch = ScratchDataHome::new("partial");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        // Put a regular file where a destination's parent directory must go: the
        // up-front validation passes (nothing exists at the destination itself) but
        // the write fails, after earlier files (BTreeMap order) have already landed.
        let root = PathBuf::from(paths::data_dir());
        let blocked = root.join("lib/neural-forge");
        std::fs::create_dir_all(blocked.parent().unwrap()).unwrap();
        std::fs::write(&blocked, "in the way").unwrap();
        let err = install(&appdir);
        assert!(err.is_err(), "the blocked destination must fail the install");
        let record = load_record().unwrap();
        assert!(!record.is_empty(), "files written before the failure must be recorded");
        for (name, digest_recorded) in &record {
            assert_eq!(&digest_file(Path::new(name)).unwrap(), digest_recorded, "{name}");
        }
        std::fs::remove_file(&blocked).unwrap();
        install(&appdir).expect("the retry must not refuse its own earlier files");
    }

    #[test]
    fn install_is_idempotent() {
        let scratch = ScratchDataHome::new("idempotent");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        assert!(install(&appdir).unwrap().changed);
        let layer = PathBuf::from(paths::data_dir()).join("lib/neural-forge/libneural_forge_layer.so");
        let inode = std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&layer).unwrap());
        let again = install(&appdir).expect("reinstalling the exact same AppDir should succeed");
        assert!(!again.changed, "an up-to-date install must not rewrite anything");
        assert_eq!(std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(&layer).unwrap()), inode, "the layer must not be replaced");

        std::fs::write(appdir.join("usr/lib/neural-forge/libneural_forge_layer.so"), "new layer").unwrap();
        assert!(install(&appdir).unwrap().changed, "a changed AppDir file must be installed");
        assert_eq!(std::fs::read_to_string(&layer).unwrap(), "new layer");
    }

    #[test]
    fn reinstall_replaces_by_rename_not_in_place_truncation() {
        let scratch = ScratchDataHome::new("rename-replace");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();

        let binary = PathBuf::from(paths::data_dir()).join("bin/neural-forge");
        // Stands in for an already-running process with the old inode mapped --
        // opening it for read before the reinstall, then confirming it still reads
        // the pre-update bytes after the reinstall's write lands.
        let running = std::fs::File::open(&binary).unwrap();
        std::fs::write(appdir.join("usr/bin/neural-forge"), "updated gui").unwrap();
        install(&appdir).unwrap();

        use std::io::Read;
        let mut still_mapped = String::new();
        (&running).read_to_string(&mut still_mapped).unwrap();
        assert_eq!(still_mapped, "gui");
        assert_eq!(std::fs::read_to_string(&binary).unwrap(), "updated gui");
    }

    #[test]
    fn install_refuses_to_overwrite_a_file_the_user_changed_by_hand() {
        let scratch = ScratchDataHome::new("refuse-unowned");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();

        let binary = PathBuf::from(paths::data_dir()).join("bin/neural-forge");
        std::fs::write(&binary, "user changed").unwrap();

        let err = install(&appdir).expect_err("install must refuse once a tracked file no longer matches its record");
        assert!(matches!(err, InstallError::RefusedUnowned(ref path) if path == &binary), "unexpected error: {err}");
        // Refusal must not have touched anything else either.
        assert_eq!(std::fs::read_to_string(&binary).unwrap(), "user changed");
    }

    #[test]
    fn install_never_touches_an_unrelated_upstream_file() {
        let scratch = ScratchDataHome::new("upstream-untouched");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let upstream = PathBuf::from(paths::data_home()).join("vulkan/implicit_layer.d/VkLayer_DLSS5.json");
        std::fs::create_dir_all(upstream.parent().unwrap()).unwrap();
        std::fs::write(&upstream, "upstream sentinel").unwrap();

        install(&appdir).unwrap();
        assert_eq!(std::fs::read_to_string(&upstream).unwrap(), "upstream sentinel");
    }

    #[test]
    fn uninstall_removes_only_unchanged_tracked_files() {
        let scratch = ScratchDataHome::new("uninstall");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();

        let cli = PathBuf::from(paths::data_dir()).join("bin/neural-forge-cli");
        let binary = PathBuf::from(paths::data_dir()).join("bin/neural-forge");
        std::fs::write(&binary, "user changed").unwrap();

        let preserved = uninstall().unwrap();
        assert!(!cli.exists(), "an unchanged tracked file should be removed");
        assert!(binary.exists(), "a changed tracked file should be preserved");
        assert_eq!(std::fs::read_to_string(&binary).unwrap(), "user changed");
        assert_eq!(preserved, vec![binary]);
        assert!(!record_path().exists());
    }

    #[test]
    fn a_symlinked_data_home_is_fine_but_a_symlinked_destination_directory_is_not() {
        let scratch = ScratchDataHome::new("symlinks");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        // $XDG_DATA_HOME itself behind a symlink (a symlinked $HOME, say): an ordinary setup.
        let real = scratch.dir.join("real-data");
        std::fs::create_dir_all(&real).unwrap();
        let link = scratch.dir.join("linked-data");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        std::env::set_var("XDG_DATA_HOME", &link);
        install(&appdir).expect("an ancestor symlink must not refuse the install");
        assert!(real.join("neural-forge/bin/neural-forge").is_file());

        // The directory a file lands in being a symlink is refused.
        uninstall().unwrap();
        let bin = PathBuf::from(paths::data_dir()).join("bin");
        let _ = std::fs::remove_dir_all(&bin);
        let elsewhere = scratch.dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &bin).unwrap();
        let err = install(&appdir).expect_err("a symlinked destination directory must be refused");
        assert!(matches!(err, InstallError::RefusedSymlink(_)), "unexpected error: {err}");
        assert_eq!(std::fs::read_dir(&elsewhere).unwrap().count(), 0, "nothing may be written through the symlink");
    }

    #[test]
    fn a_symlinked_neural_forge_or_lib_directory_is_refused() {
        let scratch = ScratchDataHome::new("symlinked-root");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let elsewhere = scratch.dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(&elsewhere, paths::data_dir()).unwrap();
        assert!(matches!(install(&appdir), Err(InstallError::RefusedSymlink(_))));
        assert_eq!(std::fs::read_dir(&elsewhere).unwrap().count(), 0, "nothing may be written through the symlink");
    }

    #[test]
    fn without_a_record_uninstall_still_removes_the_install_and_purge_the_manifest() {
        let scratch = ScratchDataHome::new("no-record");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();
        let manifest = manifest_path();
        let lib = PathBuf::from(paths::data_dir()).join("lib/neural-forge/libneural_forge_layer.so");
        assert!(manifest.is_file() && lib.is_file());
        // A lost record: the layer and the manifest that loads it still go.
        std::fs::remove_file(record_path()).unwrap();
        assert!(uninstall().unwrap().is_empty());
        assert!(!manifest.exists() && !lib.exists());
        assert!(!PathBuf::from(paths::data_dir()).join("bin/neural-forge-cli").exists());
        // A corrupt record: the same.
        install(&appdir).unwrap();
        std::fs::write(record_path(), "{ not json").unwrap();
        uninstall().unwrap();
        assert!(!manifest.exists() && !lib.exists() && !record_path().exists());
        // purge never leaves a manifest pointing at the deleted library, record or not.
        install(&appdir).unwrap();
        std::fs::remove_file(record_path()).unwrap();
        purge().unwrap();
        assert!(!manifest.exists() && !PathBuf::from(paths::data_dir()).exists());
        // A manifest of another layer at our name is not ours to remove.
        std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::fs::write(&manifest, r#"{"layer": {"name": "VK_LAYER_other"}}"#).unwrap();
        uninstall().unwrap();
        assert!(manifest.exists());
    }

    #[test]
    fn install_rejects_a_manifest_with_the_wrong_layer_name() {
        let scratch = ScratchDataHome::new("wrong-layer-name");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        std::fs::write(appdir.join(format!("usr/share/vulkan/implicit_layer.d/{MANIFEST}")), r#"{"layer": {"name": "VK_LAYER_something_else"}}"#).unwrap();

        let err = install(&appdir).expect_err("a manifest with the wrong layer name must be rejected");
        assert!(matches!(err, InstallError::InvalidManifest(_)), "unexpected error: {err}");
    }

    #[test]
    fn a_record_that_cannot_be_read_stops_uninstall_and_is_kept() {
        let scratch = ScratchDataHome::new("unreadable-record");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();
        let lib_dir = PathBuf::from(paths::data_dir()).join("lib/neural-forge");
        let theirs = lib_dir.join("notes.txt");
        std::fs::write(&theirs, "the user's").unwrap();
        // A record that exists but cannot be read: a directory where it goes (EISDIR, as EIO or EACCES would be).
        let record = record_path();
        std::fs::rename(&record, scratch.dir.join("record.json")).unwrap();
        std::fs::create_dir(&record).unwrap();
        let err = uninstall().expect_err("an unreadable record is not a missing one");
        assert!(err.to_string().contains("cannot be read"), "{err}");
        assert!(record.is_dir() && theirs.is_file() && manifest_path().is_file(), "nothing may be removed");
        assert!(PathBuf::from(paths::data_dir()).join("bin/neural-forge").is_file());
        assert!(purge().is_err());
        assert!(theirs.is_file() && manifest_path().is_file());
        assert!(matches!(install(&appdir), Err(InstallError::Io(_))), "install says the record cannot be read");
    }

    #[test]
    fn removal_never_goes_through_a_symlinked_directory() {
        let scratch = ScratchDataHome::new("symlinked-removal");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        install(&appdir).unwrap();
        // The user moves the data dir to another disk and links it back: the files there match the record.
        let root = PathBuf::from(paths::data_dir());
        let moved = scratch.dir.join("other-disk/neural-forge");
        std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
        std::fs::rename(&root, &moved).unwrap();
        std::os::unix::fs::symlink(&moved, &root).unwrap();
        let preserved = uninstall().unwrap();
        for file in ["bin/neural-forge", "bin/neural-forge-cli", "lib/neural-forge/libneural_forge_layer.so"] {
            assert!(moved.join(file).is_file(), "{file} was removed through the symlink");
            assert!(preserved.contains(&root.join(file)), "{file} is reported as preserved: {preserved:?}");
        }
        // Without a record, removal by location does not follow it either.
        assert!(uninstall().unwrap().is_empty());
        assert!(moved.join("bin/neural-forge").is_file() && moved.join("lib/neural-forge").is_dir());
        // A stale file of an older version under a symlinked directory stays too.
        std::fs::remove_file(&root).unwrap();
        std::fs::rename(&moved, &root).unwrap();
        install(&appdir).unwrap();
        let old_moved = scratch.dir.join("other-disk/old-dir");
        std::fs::create_dir_all(&old_moved).unwrap();
        std::fs::write(old_moved.join("old.so"), "old").unwrap();
        std::os::unix::fs::symlink(&old_moved, root.join("old-dir")).unwrap();
        let mut record = load_record().unwrap();
        record.insert(root.join("old-dir/old.so").to_string_lossy().into_owned(), digest(b"old"));
        save_record(&record).unwrap();
        install(&appdir).unwrap();
        assert!(old_moved.join("old.so").is_file(), "the stale file was removed through the symlink");
        assert!(!load_record().unwrap().contains_key(&root.join("old-dir/old.so").to_string_lossy().into_owned()), "but no longer tracked");
    }

    #[test]
    fn a_symlink_in_the_appdir_is_refused_not_followed() {
        let scratch = ScratchDataHome::new("appdir-symlink");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let lib = appdir.join("usr/lib/neural-forge");
        // A loop: following it recursed until the stack overflowed.
        std::os::unix::fs::symlink(&lib, lib.join("loop")).unwrap();
        assert!(matches!(install(&appdir), Err(InstallError::RefusedSourceSymlink(p)) if p == lib.join("loop")));
        std::fs::remove_file(lib.join("loop")).unwrap();
        // A link to a directory elsewhere: its tree would be copied in and recorded as ours.
        let elsewhere = scratch.dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("secret"), "x").unwrap();
        std::os::unix::fs::symlink(&elsewhere, lib.join("more")).unwrap();
        assert!(matches!(install(&appdir), Err(InstallError::RefusedSourceSymlink(_))));
        assert!(!PathBuf::from(paths::data_dir()).join("lib/neural-forge/more").exists(), "nothing is written");
        assert!(!manifest_path().exists());
    }
}
