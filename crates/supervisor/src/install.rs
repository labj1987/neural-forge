//! Installing an extracted AppImage AppDir into persistent user storage, and
//! uninstalling it again -- the same operation, the same `installation.json`
//! provenance record format (a flat `path -> sha256` map, refusing to touch any file
//! that record doesn't say this installer itself last wrote), and the same
//! atomic-replace-never-truncate file writes as `scripts/install.py`. Reimplemented
//! in Rust so the GUI's Setup tab can offer an "Install for Steam games" button
//! without a `python3` dependency; both tools read/write the exact same record on
//! purpose, so either can pick up after the other left off.

use std::collections::BTreeMap;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::paths;

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
    RefusedUnowned(PathBuf),
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::InvalidManifest(msg) => write!(f, "{msg}"),
            Self::RefusedSymlink(path) => write!(f, "refusing symlink destination: {}", path.display()),
            Self::RefusedUnowned(path) => write!(f, "refusing to overwrite unowned or changed file: {}", path.display()),
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
    pub root: PathBuf,
    pub gui_path: PathBuf,
    pub cli_path: PathBuf,
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn digest_file(path: &Path) -> std::io::Result<String> {
    Ok(digest(&std::fs::read(path)?))
}

fn record_path() -> PathBuf {
    PathBuf::from(paths::data_dir()).join("installation.json")
}

fn load_record() -> BTreeMap<String, String> {
    std::fs::read_to_string(record_path()).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

fn save_record(record: &BTreeMap<String, String>) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(record).expect("a string-keyed map always serializes") + "\n";
    write_atomic(&record_path(), text.as_bytes(), 0o644)
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

/// Writes `content` to `path`, replacing any existing file by renaming a same-directory
/// staged file over it -- never truncates the destination in place, so an
/// already-running process that has the old inode mapped (a GUI, a game, a Wine
/// helper) keeps reading the old content until it reopens the path, exactly like
/// `install.py`'s own `os.replace` step.
fn write_atomic(path: &Path, content: &[u8], mode: u32) -> std::io::Result<()> {
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

/// Installs `appdir` (an extracted AppImage's `AppDir`) into persistent user storage.
/// Refuses (writing nothing at all) if any destination is a symlink, or already
/// exists with content this installer's own record doesn't recognize as what it last
/// wrote there -- the same "never touch a file this didn't put there" guarantee
/// `install.py` makes.
pub fn install(appdir: &Path) -> Result<InstallReport, InstallError> {
    let data_home = PathBuf::from(paths::data_home());
    let root = PathBuf::from(paths::data_dir());
    let old = load_record();
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
    files.insert(data_home.join(format!("vulkan/implicit_layer.d/{MANIFEST}")), manifest_out.into_bytes());

    // The 32-bit layer's own manifest (its own layer name), when the AppDir carries it.
    let manifest32_src = usr.join("share/vulkan/implicit_layer.d/neural_forge_layer_i686.json");
    if manifest32_src.is_file() {
        let mut manifest32: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&manifest32_src)?)?;
        let library32 = root.join("lib/neural-forge/i686/libneural_forge_layer.so");
        manifest32["layer"]["library_path"] = serde_json::Value::String(library32.to_string_lossy().into_owned());
        let out = serde_json::to_string_pretty(&manifest32)? + "\n";
        files.insert(data_home.join("vulkan/implicit_layer.d/neural_forge_layer_i686.json"), out.into_bytes());
    }

    // No `.desktop` file, icon or AppStream metainfo: every install comes from an
    // AppImage, and menu integration belongs to whatever integrates that AppImage
    // (Gear Lever, AppImageLauncher, ...). Installing a second entry here gave users two
    // "Neural Forge" launchers. Older installs that did write them get them removed by
    // the stale-entry pass below, as long as they are still exactly what was written.

    // Validate every destination before writing any of them, exactly like
    // `install.py`: a failure partway through must leave nothing changed, not a
    // half-installed mix of old and new files.
    for path in files.keys() {
        for ancestor in path.ancestors() {
            if !ancestor.as_os_str().is_empty() && ancestor.is_symlink() {
                return Err(InstallError::RefusedSymlink(path.clone()));
            }
        }
        if path.exists() {
            let current = digest_file(path)?;
            if old.get(&path.to_string_lossy().into_owned()) != Some(&current) {
                return Err(InstallError::RefusedUnowned(path.clone()));
            }
        }
    }

    // Ownership is recorded *as files land*, not once at the end: a failure partway
    // through must leave every file already written tracked (otherwise the next run
    // sees them as unowned and refuses to touch them). Each file's record entry is
    // saved before the next file is written, so an interruption can orphan at most the
    // single file in flight.
    let bin_dir = root.join("bin");
    let mut record = old.clone();
    for (path, content) in &files {
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
        if path.is_file() && !path.is_symlink() && digest_file(&path).ok().as_ref() == Some(&recorded) {
            std::fs::remove_file(&path)?;
        }
        record.remove(&name);
        save_record(&record)?;
        if let Some(parent) = path.parent() {
            prune_empty_dirs(parent, &root);
        }
    }

    Ok(InstallReport { root: root.clone(), gui_path: root.join("bin/neural-forge"), cli_path: root.join("bin/neural-forge-cli") })
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
/// reported as preserved rather than silently deleted), then clears the record.
pub fn uninstall() -> std::io::Result<Vec<PathBuf>> {
    let old = load_record();
    let mut preserved = Vec::new();
    for (name, expected) in &old {
        let path = PathBuf::from(name);
        let matches = path.is_file() && !path.is_symlink() && digest_file(&path).ok().as_ref() == Some(expected);
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
/// NGX DLLs and the managed Wine prefix included), state and `/tmp/neural-forge-$UID`. Stops a
/// running helper first. Each directory must be one of Neural Forge's own (named
/// `neural-forge` or `neural-forge-<uid>`) or it is left alone. Returns what was removed.
pub fn purge() -> std::io::Result<Vec<PathBuf>> {
    let _ = crate::stop(std::time::Duration::from_secs(5));
    let preserved = uninstall()?;
    for path in &preserved {
        if path.is_file() {
            let _ = std::fs::remove_file(path);
            remove_empty_parents(path);
        }
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
            // Keep the tests away from the real config/state dirs and from any real
            // helper's pid file: a unique uid names a runtime dir that cannot exist.
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
            ("usr/lib/neural-forge/helper/neural-forge-helper.exe", "helper"),
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
        assert_eq!(std::fs::read_to_string(root.join("lib/neural-forge/helper/neural-forge-helper.exe")).unwrap(), "helper");

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
    fn install_removes_a_desktop_entry_an_older_install_wrote() {
        let scratch = ScratchDataHome::new("old-desktop-entry");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let desktop = PathBuf::from(paths::data_home()).join(format!("applications/{APP_ID}.desktop"));
        let old_text = "[Desktop Entry]\nExec=\"/old/bin/neural-forge\"\n";
        std::fs::create_dir_all(desktop.parent().unwrap()).unwrap();
        std::fs::write(&desktop, old_text).unwrap();
        let mut record = BTreeMap::new();
        record.insert(desktop.to_string_lossy().into_owned(), digest(old_text.as_bytes()));
        save_record(&record).unwrap();

        install(&appdir).unwrap();
        assert!(!desktop.exists(), "the entry an older install wrote must go, leaving the AppImage's own entry as the only one");
        assert!(!load_record().contains_key(&desktop.to_string_lossy().into_owned()));
    }

    #[test]
    fn install_removes_files_a_newer_version_no_longer_ships() {
        let scratch = ScratchDataHome::new("stale-cleanup");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let extra = appdir.join("usr/lib/neural-forge/helper/old-only.dll");
        std::fs::write(&extra, "old").unwrap();
        install(&appdir).unwrap();
        let installed = PathBuf::from(paths::data_dir()).join("lib/neural-forge/helper/old-only.dll");
        assert!(installed.exists());

        std::fs::remove_file(&extra).unwrap();
        install(&appdir).unwrap();
        assert!(!installed.exists(), "a file the new install no longer ships must be removed");
        assert!(!load_record().contains_key(&installed.to_string_lossy().into_owned()));
    }

    #[test]
    fn stale_file_the_user_edited_is_left_alone_but_untracked() {
        let scratch = ScratchDataHome::new("stale-edited");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        let extra = appdir.join("usr/lib/neural-forge/helper/old-only.dll");
        std::fs::write(&extra, "old").unwrap();
        install(&appdir).unwrap();
        let installed = PathBuf::from(paths::data_dir()).join("lib/neural-forge/helper/old-only.dll");
        std::fs::write(&installed, "edited by hand").unwrap();

        std::fs::remove_file(&extra).unwrap();
        install(&appdir).unwrap();
        assert_eq!(std::fs::read_to_string(&installed).unwrap(), "edited by hand");
        assert!(!load_record().contains_key(&installed.to_string_lossy().into_owned()));
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
        let blocked = root.join("lib/neural-forge/helper");
        std::fs::create_dir_all(blocked.parent().unwrap()).unwrap();
        std::fs::write(&blocked, "in the way").unwrap();
        let err = install(&appdir);
        assert!(err.is_err(), "the blocked destination must fail the install");
        let record = load_record();
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
        install(&appdir).unwrap();
        install(&appdir).expect("reinstalling the exact same AppDir should succeed");
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
    fn install_rejects_a_manifest_with_the_wrong_layer_name() {
        let scratch = ScratchDataHome::new("wrong-layer-name");
        let appdir = scratch.dir.join("AppDir");
        write_fixture_appdir(&appdir);
        std::fs::write(appdir.join(format!("usr/share/vulkan/implicit_layer.d/{MANIFEST}")), r#"{"layer": {"name": "VK_LAYER_something_else"}}"#).unwrap();

        let err = install(&appdir).expect_err("a manifest with the wrong layer name must be rejected");
        assert!(matches!(err, InstallError::InvalidManifest(_)), "unexpected error: {err}");
    }
}
