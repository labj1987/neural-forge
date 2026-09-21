//! One-time migration from the pre-0.1.77 `neuralforge` directory names to `neural-forge`.
//!
//! Runs at the start of the GUI, of every CLI command and of `install`; it is idempotent
//! and costs three `stat`s once there is nothing left to move. For each of the XDG config,
//! data and state homes it renames `<home>/neuralforge` to `<home>/neural-forge` -- an atomic
//! same-filesystem rename, never a copy, which matters for the Wine prefix inside the data
//! directory (large, and full of files a half-finished copy would corrupt). When the new
//! directory already exists (an old build ran again, or `ensure_dirs` beat us to it) the two
//! are merged entry by entry without ever overwriting anything.
//!
//! Afterwards every absolute path that was recorded under the old names is rewritten:
//! `config.ini`/`profiles.ini`, the `installation.json` provenance record (its keys), the
//! installed Vulkan manifest's `library_path` and `.desktop` `Exec=` (re-hashed in the record
//! so the installer still recognises them as its own), and inside the Wine prefix the
//! registry and other top-level text files plus any absolute symlink.
//!
//! A helper still running from the old layout is stopped first: its wineserver is bound to
//! the old prefix path, and moving that under it would leave it half-alive.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

const OLD: &str = "neuralforge";
const NEW: &str = "neural-forge";
/// Prefix files bigger than this are not scanned for recorded paths.
const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WALK_ENTRIES: usize = 500_000;

#[derive(Debug, Default)]
pub struct Report {
    /// `old -> new` for every directory or entry moved.
    pub moved: Vec<String>,
    /// Files whose recorded absolute paths were rewritten.
    pub rewritten: Vec<String>,
    /// Things left alone (conflicts, failures). Nothing here is fatal.
    pub warnings: Vec<String>,
    pub stopped_helper: bool,
}

impl Report {
    pub fn is_empty(&self) -> bool {
        self.moved.is_empty() && self.rewritten.is_empty() && self.warnings.is_empty() && !self.stopped_helper
    }

    pub fn summary(&self) -> String {
        let mut out = format!("migrated {} neuralforge -> neural-forge entr{}", self.moved.len(), if self.moved.len() == 1 { "y" } else { "ies" });
        if self.stopped_helper {
            out.push_str("; stopped the old helper (restart it)");
        }
        if !self.rewritten.is_empty() {
            out.push_str(&format!("; fixed recorded paths in {} file(s)", self.rewritten.len()));
        }
        for w in &self.warnings {
            out.push_str(&format!("\n  warning: {w}"));
        }
        out
    }
}

struct Ctx {
    config_home: PathBuf,
    data_home: PathBuf,
    state_home: PathBuf,
    /// Pid files of helpers started by the old layout; stopped before the data move.
    legacy_pid_files: Vec<String>,
}

/// Migrates the real user directories (per the XDG environment). Safe to call repeatedly
/// and concurrently (serialised by a lock file).
pub fn migrate() -> Report {
    let mut legacy_pid_files = Vec::new();
    if let Some(dir) = neural_forge_protocol::compat::legacy_runtime_dir() {
        legacy_pid_files.push(format!("{}/helper.pid", dir.display()));
    }
    run(&Ctx {
        config_home: crate::paths::config_home().into(),
        data_home: crate::paths::data_home().into(),
        state_home: crate::paths::state_home().into(),
        legacy_pid_files,
    })
}

fn is_real_dir(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok_and(|m| m.is_dir())
}

fn run(ctx: &Ctx) -> Report {
    let homes = [&ctx.config_home, &ctx.data_home, &ctx.state_home];
    if !homes.iter().any(|h| is_real_dir(&h.join(OLD))) {
        return Report::default();
    }
    let _lock = Lock::acquire(&ctx.data_home);
    // Re-check under the lock: another process may have just finished the job.
    if !homes.iter().any(|h| is_real_dir(&h.join(OLD))) {
        return Report::default();
    }
    let mut report = Report::default();
    let (old_config, new_config) = (ctx.config_home.join(OLD), ctx.config_home.join(NEW));
    let (old_data, new_data) = (ctx.data_home.join(OLD), ctx.data_home.join(NEW));
    let (old_state, new_state) = (ctx.state_home.join(OLD), ctx.state_home.join(NEW));

    move_dir(&old_config, &new_config, &mut report);

    if is_real_dir(&old_data) {
        stop_legacy_helper(ctx, &old_data, &mut report);
    }
    move_dir(&old_data, &new_data, &mut report);
    move_dir(&old_state, &new_state, &mut report);

    // Every string that may have been recorded, in the three spellings a path takes
    // (plain, backslashed, and doubled for Wine registry files).
    let mut subs: Vec<(Vec<u8>, Vec<u8>, bool)> = Vec::new();
    for (old, new) in [(&old_config, &new_config), (&old_data, &new_data), (&old_state, &new_state)] {
        for variant in path_variants(old, new) {
            subs.push((variant.0, variant.1, true));
        }
    }
    for (old, new) in [("/tmp/neuralforge-", "/tmp/neural-forge-"), ("\\tmp\\neuralforge-", "\\tmp\\neural-forge-"), ("\\\\tmp\\\\neuralforge-", "\\\\tmp\\\\neural-forge-")] {
        subs.push((old.as_bytes().to_vec(), new.as_bytes().to_vec(), false));
    }

    for name in ["config.ini", "profiles.ini"] {
        rewrite_file(&new_config.join(name), &subs, &mut report);
    }
    fix_record(&new_data, &ctx.data_home, &subs, &mut report);
    fix_prefix(&new_data.join("prefix"), &old_data, &new_data, &subs, &mut report);
    report
}

fn move_dir(old: &Path, new: &Path, report: &mut Report) {
    if !is_real_dir(old) {
        return;
    }
    if std::fs::symlink_metadata(new).is_err() {
        match std::fs::rename(old, new) {
            Ok(()) => report.moved.push(format!("{} -> {}", old.display(), new.display())),
            Err(e) => report.warnings.push(format!("could not move {} to {}: {e}", old.display(), new.display())),
        }
        return;
    }
    merge_dir(old, new, report);
    let _ = std::fs::remove_dir(old); // only succeeds when everything was moved out
}

/// Moves every entry of `old` that `new` does not have; recurses into directories both
/// have. Never overwrites: a genuine conflict stays behind in `old` with a warning.
fn merge_dir(old: &Path, new: &Path, report: &mut Report) {
    let Ok(entries) = std::fs::read_dir(old) else { return };
    for entry in entries.flatten() {
        let (src, dst) = (entry.path(), new.join(entry.file_name()));
        match std::fs::symlink_metadata(&dst) {
            Err(_) => match std::fs::rename(&src, &dst) {
                Ok(()) => report.moved.push(format!("{} -> {}", src.display(), dst.display())),
                Err(e) => report.warnings.push(format!("could not move {}: {e}", src.display())),
            },
            Ok(_) if is_real_dir(&src) && is_real_dir(&dst) => {
                // An empty placeholder (from `ensure_dirs`) loses to the real content.
                if std::fs::remove_dir(&dst).is_ok() {
                    match std::fs::rename(&src, &dst) {
                        Ok(()) => report.moved.push(format!("{} -> {}", src.display(), dst.display())),
                        Err(e) => report.warnings.push(format!("could not move {}: {e}", src.display())),
                    }
                } else {
                    merge_dir(&src, &dst, report);
                    let _ = std::fs::remove_dir(&src);
                }
            }
            Ok(_) => report.warnings.push(format!("{} exists in both old and new locations; left {} in place", dst.display(), src.display())),
        }
    }
}

fn stop_legacy_helper(ctx: &Ctx, old_data: &Path, report: &mut Report) {
    let mut any = false;
    for pid_file in &ctx.legacy_pid_files {
        if crate::process::running_pid(pid_file).is_some() {
            any = true;
            let _ = crate::process::stop_matching(pid_file, Duration::from_secs(5), Some("forge-helper"));
        }
    }
    if !any {
        return;
    }
    report.stopped_helper = true;
    // The wineserver a killed helper leaves behind is bound to the old prefix path.
    let cfg = crate::Config::load();
    if let Some(wineserver) = crate::wineserver_binary(&cfg) {
        let prefix = crate::real_wineprefix(&cfg, &old_data.join("prefix").display().to_string());
        let _ = std::process::Command::new(wineserver).arg("-k").env("WINEPREFIX", prefix).status();
    }
}

fn path_variants(old: &Path, new: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let (o, n) = (old.display().to_string(), new.display().to_string());
    let back = |s: &str, doubled: bool| s.replace('/', if doubled { "\\\\" } else { "\\" }).into_bytes();
    vec![
        (o.clone().into_bytes(), n.clone().into_bytes()),
        (back(&o, false), back(&n, false)),
        (back(&o, true), back(&n, true)),
    ]
}

/// Replaces every occurrence of each `(old, new, boundary)`; with `boundary` the match
/// must not run on into a longer name (`.../neuralforge-x` is not `.../neuralforge`).
/// `None` when nothing changed.
fn replace_all(hay: &[u8], subs: &[(Vec<u8>, Vec<u8>, bool)]) -> Option<Vec<u8>> {
    let mut cur = hay.to_vec();
    let mut changed = false;
    for (old, new, boundary) in subs {
        if old.is_empty() {
            continue;
        }
        let mut out = Vec::with_capacity(cur.len());
        let mut i = 0;
        while i < cur.len() {
            if cur[i..].starts_with(old) {
                let next = cur.get(i + old.len()).copied();
                let ok = !*boundary || !next.is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.');
                if ok {
                    out.extend_from_slice(new);
                    i += old.len();
                    changed = true;
                    continue;
                }
            }
            out.push(cur[i]);
            i += 1;
        }
        cur = out;
    }
    changed.then_some(cur)
}

/// Rewrites `path` in place (atomically, keeping its mode) if it records an old path.
fn rewrite_file(path: &Path, subs: &[(Vec<u8>, Vec<u8>, bool)], report: &mut Report) -> Option<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_SCAN_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let fixed = replace_all(&bytes, subs)?;
    match write_replace(path, &fixed, meta.permissions().mode() & 0o7777) {
        Ok(()) => {
            report.rewritten.push(path.display().to_string());
            Some(fixed)
        }
        Err(e) => {
            report.warnings.push(format!("could not update recorded paths in {}: {e}", path.display()));
            None
        }
    }
}

fn write_replace(path: &Path, content: &[u8], mode: u32) -> std::io::Result<()> {
    let tmp = path.with_file_name(format!(".{}.{}.migrate", path.file_name().and_then(|n| n.to_str()).unwrap_or("file"), std::process::id()));
    std::fs::write(&tmp, content)?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode)).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Rewrites `installation.json`'s keys, then every tracked file *outside* the data
/// directory that embeds a path into it (the Vulkan manifest, the `.desktop` file), and
/// re-records those files' digests so the installer still treats them as its own.
fn fix_record(new_data: &Path, data_home: &Path, subs: &[(Vec<u8>, Vec<u8>, bool)], report: &mut Report) {
    let record = new_data.join("installation.json");
    rewrite_file(&record, subs, report);
    let Some(text) = std::fs::read_to_string(&record).ok() else { return };
    let Ok(mut map) = serde_json::from_str::<std::collections::BTreeMap<String, String>>(&text) else { return };
    let mut changed = false;
    for (name, digest) in map.iter_mut() {
        let path = Path::new(name);
        // Only files that lie in the shared data hierarchy but not inside our own dir.
        if !path.starts_with(data_home) || path.starts_with(new_data) {
            continue;
        }
        let Ok(before) = std::fs::read(path) else { continue };
        if sha256_hex(&before) != *digest {
            continue; // not exactly what we wrote: the user's now, leave it
        }
        if let Some(after) = rewrite_file(path, subs, report) {
            *digest = sha256_hex(&after);
            changed = true;
        }
    }
    if changed {
        let out = serde_json::to_string_pretty(&map).expect("a string map serializes") + "\n";
        if let Err(e) = write_replace(&record, out.as_bytes(), 0o644) {
            report.warnings.push(format!("could not re-record digests in {}: {e}", record.display()));
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// The Wine prefix: top-level text files (`config_info`, `version`, ...) and registry
/// files of the prefix and its `pfx` subdirectory, plus absolute symlinks anywhere in it.
fn fix_prefix(prefix: &Path, old_data: &Path, new_data: &Path, subs: &[(Vec<u8>, Vec<u8>, bool)], report: &mut Report) {
    if !is_real_dir(prefix) {
        return;
    }
    for dir in [prefix.to_path_buf(), prefix.join("pfx")] {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            rewrite_file(&entry.path(), subs, report);
        }
    }
    let (old_s, new_s) = (old_data.display().to_string(), new_data.display().to_string());
    let mut budget = MAX_WALK_ENTRIES;
    fix_symlinks(prefix, &old_s, &new_s, &mut budget, report);
}

fn fix_symlinks(dir: &Path, old: &str, new: &str, budget: &mut usize, report: &mut Report) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let path = entry.path();
        let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
        if meta.is_dir() {
            fix_symlinks(&path, old, new, budget, report);
        } else if meta.file_type().is_symlink() {
            let Ok(target) = std::fs::read_link(&path) else { continue };
            let t = target.display().to_string();
            let rest = t.strip_prefix(old).filter(|r| r.is_empty() || r.starts_with('/'));
            if let Some(rest) = rest {
                let fixed = format!("{new}{rest}");
                let ok = std::fs::remove_file(&path).and_then(|()| std::os::unix::fs::symlink(&fixed, &path));
                match ok {
                    Ok(()) => report.rewritten.push(format!("{} (symlink)", path.display())),
                    Err(e) => report.warnings.push(format!("could not repoint symlink {}: {e}", path.display())),
                }
            }
        }
    }
}

struct Lock(#[allow(dead_code)] Option<std::fs::File>);

impl Lock {
    fn acquire(data_home: &Path) -> Self {
        use std::os::fd::AsRawFd;
        let _ = std::fs::create_dir_all(data_home);
        let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(data_home.join(".neural-forge-migrate.lock")).ok();
        if let Some(f) = &file {
            // SAFETY: valid fd; blocks until any concurrent migration finishes.
            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) };
        }
        Lock(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("neural-forge-migrate-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn ctx(root: &Path) -> Ctx {
        Ctx { config_home: root.join("config"), data_home: root.join("data"), state_home: root.join("state"), legacy_pid_files: vec![] }
    }

    fn write(p: &Path, s: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, s).unwrap();
    }

    #[test]
    fn nothing_to_do_is_a_noop() {
        let root = scratch("noop");
        assert!(run(&ctx(&root)).is_empty());
        assert!(!root.join("data").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn moves_dirs_fixes_recorded_paths_and_keeps_the_prefix_inode() {
        use std::os::unix::fs::MetadataExt;
        let root = scratch("full");
        let c = ctx(&root);
        let (od, oc, os) = (c.data_home.join("neuralforge"), c.config_home.join("neuralforge"), c.state_home.join("neuralforge"));
        let nd = c.data_home.join("neural-forge");
        write(&oc.join("config.ini"), &format!("binaries={}/binaries\nshm=/tmp/neuralforge-1000/shm.bin\nlog={}/helper.log\nrunner_type=proton\n", od.display(), os.display()));
        write(&os.join("helper.log"), "log");
        write(&od.join("binaries/nvngx_dlssnr.dll"), "dll");
        write(&od.join("prefix/pfx/user.reg"), &format!("[Software\\\\X] 1\n\"P\"=\"Z:{}\"\n\"Q\"=\"Z:{}\"\n", od.display().to_string().replace('/', "\\\\"), format!("{}-keep/x", od.display())));
        write(&od.join("prefix/config_info"), &format!("{}/prefix/pfx\n", od.display()));
        std::os::unix::fs::symlink(od.join("binaries"), od.join("prefix/pfx/bin-link")).unwrap();
        std::os::unix::fs::symlink("../drive_c", od.join("prefix/pfx/rel")).unwrap();
        let prefix_ino = std::fs::metadata(od.join("prefix")).unwrap().ino();

        // Recorded install: a lib file inside the data dir, a manifest and a desktop file outside.
        let manifest = c.data_home.join("vulkan/implicit_layer.d/VK_LAYER_neuralforge_neural.json");
        let lib = od.join("lib/neuralforge/libneuralforge_layer.so");
        write(&lib, "so");
        write(&manifest, &format!("{{\"library_path\": \"{}\"}}", lib.display()));
        let desktop = c.data_home.join("applications/io.github.labj1987.NeuralForge.desktop");
        write(&desktop, &format!("Exec=\"{}/bin/neuralforge\"\n", od.display()));
        let dig = |p: &Path| sha256_hex(&std::fs::read(p).unwrap());
        let record: std::collections::BTreeMap<String, String> = [&lib, &manifest, &desktop].iter().map(|p| (p.display().to_string(), dig(p))).collect();
        write(&od.join("installation.json"), &serde_json::to_string(&record).unwrap());

        let report = run(&c);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert!(!od.exists() && !oc.exists() && !os.exists());
        assert_eq!(std::fs::metadata(nd.join("prefix")).unwrap().ino(), prefix_ino, "the prefix must be renamed, not copied");
        let cfg = std::fs::read_to_string(c.config_home.join("neural-forge/config.ini")).unwrap();
        assert!(cfg.contains(&format!("binaries={}/binaries", nd.display())));
        assert!(cfg.contains("shm=/tmp/neural-forge-1000/shm.bin"));
        assert!(cfg.contains(&format!("log={}/helper.log", c.state_home.join("neural-forge").display())));
        let reg = std::fs::read_to_string(nd.join("prefix/pfx/user.reg")).unwrap();
        assert!(reg.contains(&nd.display().to_string().replace('/', "\\\\")));
        assert!(reg.contains("neuralforge-keep"), "a longer sibling name must not be rewritten");
        assert!(std::fs::read_to_string(nd.join("prefix/config_info")).unwrap().starts_with(&nd.display().to_string()));
        assert_eq!(std::fs::read_link(nd.join("prefix/pfx/bin-link")).unwrap(), nd.join("binaries"));
        assert_eq!(std::fs::read_link(nd.join("prefix/pfx/rel")).unwrap(), Path::new("../drive_c"));

        let new_lib = nd.join("lib/neuralforge/libneuralforge_layer.so");
        let text = std::fs::read_to_string(&manifest).unwrap();
        assert!(text.contains(&new_lib.display().to_string()));
        assert!(new_lib.exists());
        let rec: std::collections::BTreeMap<String, String> = serde_json::from_str(&std::fs::read_to_string(nd.join("installation.json")).unwrap()).unwrap();
        assert_eq!(rec[&new_lib.display().to_string()], sha256_hex(b"so"));
        assert_eq!(rec[&manifest.display().to_string()], dig(&manifest), "rewritten manifest must be re-hashed");
        assert_eq!(rec[&desktop.display().to_string()], dig(&desktop));
        assert!(std::fs::read_to_string(&desktop).unwrap().contains(&nd.display().to_string()));

        assert!(run(&c).is_empty(), "second run is a no-op");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn merges_into_an_existing_new_dir_without_overwriting() {
        let root = scratch("merge");
        let c = ctx(&root);
        let (od, nd) = (c.data_home.join("neuralforge"), c.data_home.join("neural-forge"));
        write(&od.join("prefix/pfx/system.reg"), "real prefix");
        write(&od.join("binaries/a.dll"), "old a");
        std::fs::create_dir_all(nd.join("prefix")).unwrap(); // empty placeholder from ensure_dirs
        write(&nd.join("binaries/a.dll"), "new a");
        write(&od.join("binaries/b.dll"), "old b");
        let report = run(&c);
        assert_eq!(std::fs::read_to_string(nd.join("prefix/pfx/system.reg")).unwrap(), "real prefix");
        assert_eq!(std::fs::read_to_string(nd.join("binaries/a.dll")).unwrap(), "new a");
        assert_eq!(std::fs::read_to_string(nd.join("binaries/b.dll")).unwrap(), "old b");
        assert_eq!(std::fs::read_to_string(od.join("binaries/a.dll")).unwrap(), "old a", "conflict stays behind");
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
