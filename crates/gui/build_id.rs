//! The build's identity for `NEURAL_FORGE_BUILD_ID` (#73), and the paths whose change gives a new one. Used by
//! `build.rs`; `tests/build_id.rs` includes it too, so its rules are tested.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    out.status.success().then_some(out.stdout)
}

fn hash_of(bytes: &[u8]) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Build output (`cargo build` run inside a crate), never a source.
fn is_build_output(path: &Path) -> bool {
    path.is_dir() && path.file_name().is_some_and(|n| n == "target")
}

/// The entries of `dir` in a stable order, build output left out.
fn entries(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| !is_build_output(p)).collect();
    paths.sort();
    paths
}

/// Whether a `target` directory is anywhere under `dir`.
fn holds_build_output(dir: &Path) -> bool {
    let Ok(read) = std::fs::read_dir(dir) else { return false };
    read.filter_map(|e| e.ok().map(|e| e.path())).any(|p| is_build_output(&p) || (p.is_dir() && holds_build_output(&p)))
}

/// What to watch for `dir`: the directory itself (Cargo scans it whole), or, when build output is somewhere inside,
/// its other entries one by one.
pub fn watch_tree(dir: &Path, out: &mut Vec<PathBuf>) {
    if !holds_build_output(dir) {
        out.push(dir.to_path_buf());
        return;
    }
    for path in entries(dir) {
        if path.is_dir() {
            watch_tree(&path, out);
        } else {
            out.push(path);
        }
    }
}

/// Every file under `path` (or the file itself), in a stable order, into `h` (path and contents); build output left out.
pub fn hash_tree(path: &Path, h: &mut std::hash::DefaultHasher) {
    if path.is_dir() {
        for entry in entries(path) {
            hash_tree(&entry, h);
        }
    } else if let Ok(bytes) = std::fs::read(path) {
        path.hash(h);
        bytes.hash(h);
    }
}

/// The sources the GUI is built from: every crate, the data files compiled in (`data/games.toml` through the supervisor),
/// and the workspace's manifest and lock file.
pub fn sources(root: &Path) -> Vec<PathBuf> {
    vec![root.join("crates"), root.join("data"), root.join("Cargo.toml"), root.join("Cargo.lock")]
}

/// The build's identity, and the existing paths whose change gives a new one. A path that does not exist is never
/// watched: Cargo would take it as changed and build the GUI again on every run.
pub fn build_id(root: &Path, version: &str) -> (String, Vec<PathBuf>) {
    let sources = sources(root);
    let mut watch = Vec::new();
    for path in sources.iter().filter(|p| p.exists()) {
        if path.is_dir() {
            watch_tree(path, &mut watch);
        } else {
            watch.push(path.clone());
        }
    }
    let id = if let Some(commit) = git(root, &["rev-parse", "HEAD"]) {
        let commit = String::from_utf8_lossy(&commit).trim().to_owned();
        // HEAD names the branch (or the commit when detached), HEAD's reflog changes with every commit and checkout,
        // and the branch's loose ref when it has one (a packed one changes packed-refs).
        let mut git_paths = vec!["HEAD".to_owned(), "logs/HEAD".to_owned(), "packed-refs".to_owned()];
        if let Some(r) = git(root, &["symbolic-ref", "-q", "HEAD"]) {
            git_paths.push(String::from_utf8_lossy(&r).trim().to_owned());
        }
        for name in git_paths {
            if let Some(p) = git(root, &["rev-parse", "--git-path", &name]) {
                let p = root.join(String::from_utf8_lossy(&p).trim());
                if p.exists() {
                    watch.push(p);
                }
            }
        }
        let dirty = git(root, &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
        match git(root, &["diff", "HEAD"]).filter(|_| dirty) {
            Some(diff) => format!("{commit}+{:016x}", hash_of(&diff)),
            None => commit,
        }
    } else {
        let mut h = std::hash::DefaultHasher::new();
        for path in &sources {
            hash_tree(path, &mut h);
        }
        format!("{version}+src{:016x}", h.finish())
    };
    (id, watch)
}
