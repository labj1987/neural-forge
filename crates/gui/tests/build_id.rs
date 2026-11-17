//! The build identity's rules (`build_id.rs`, run by `build.rs`).

#[path = "../build_id.rs"]
mod build_id;

use std::path::{Path, PathBuf};
use std::process::Command;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("nf-build-id-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.0.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A workspace like this one, with build output inside a crate.
fn workspace(s: &Scratch) {
    s.write("Cargo.toml", "[workspace]\n");
    s.write("Cargo.lock", "# lock\n");
    s.write("data/games.toml", "[games]\n");
    s.write("crates/gui/src/main.rs", "fn main() {}\n");
    s.write("crates/gui/target/debug/out", "built\n");
    s.write("crates/protocol/src/lib.rs", "\n");
}

fn git(dir: &Path, args: &[&str]) {
    let st = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"])
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .status()
        .unwrap();
    assert!(st.success(), "git {args:?}");
}

#[test]
fn build_output_is_neither_watched_nor_hashed() {
    let s = Scratch::new("target");
    workspace(&s);
    let mut watch = Vec::new();
    build_id::watch_tree(&s.0.join("crates"), &mut watch);
    assert!(watch.iter().all(|p| !p.components().any(|c| c.as_os_str() == "target")), "{watch:?}");
    // The crate without build output is watched whole, the other one by its entries.
    assert!(watch.contains(&s.0.join("crates/protocol")) && watch.contains(&s.0.join("crates/gui/src")), "{watch:?}");
    let hash = || {
        let mut h = std::hash::DefaultHasher::new();
        build_id::hash_tree(&s.0.join("crates"), &mut h);
        std::hash::Hasher::finish(&h)
    };
    let before = hash();
    s.write("crates/gui/target/debug/out", "built again\n");
    assert_eq!(hash(), before, "build output changes nothing");
    s.write("crates/gui/src/main.rs", "fn main() { }\n");
    assert_ne!(hash(), before, "a source change gives a new identity");
}

#[test]
fn without_git_every_source_counts_and_only_existing_paths_are_watched() {
    let s = Scratch::new("nogit");
    workspace(&s);
    let (id, watch) = build_id::build_id(&s.0, "9.9.9");
    assert!(id.starts_with("9.9.9+src"), "{id}");
    assert!(watch.iter().all(|p| p.exists()), "{watch:?}");
    assert!(watch.contains(&s.0.join("data")) && watch.contains(&s.0.join("Cargo.toml")) && watch.contains(&s.0.join("Cargo.lock")), "{watch:?}");
    s.write("data/games.toml", "[games]\nx = 1\n");
    assert_ne!(build_id::build_id(&s.0, "9.9.9").0, id, "data compiled into the app is a source");
    // A workspace without data/: nothing missing is watched.
    std::fs::remove_dir_all(s.0.join("data")).unwrap();
    assert!(build_id::build_id(&s.0, "9.9.9").1.iter().all(|p| p.exists()));
}

#[test]
fn in_git_the_commit_and_only_existing_git_files_are_watched() {
    let s = Scratch::new("git");
    workspace(&s);
    git(&s.0, &["init", "-q", "-b", "main"]);
    git(&s.0, &["add", "Cargo.toml", "Cargo.lock", "data", "crates/gui/src", "crates/protocol"]);
    git(&s.0, &["commit", "-q", "-m", "one"]);
    let (id, watch) = build_id::build_id(&s.0, "9.9.9");
    assert_eq!(id.len(), 40, "a clean tree is its commit: {id}");
    // No packed-refs in a fresh repository: watching it would build the GUI again on every run.
    assert!(!s.0.join(".git/packed-refs").exists());
    assert!(watch.iter().all(|p| p.exists()), "{watch:?}");
    let git_files: Vec<_> = watch.iter().filter(|p| p.starts_with(s.0.join(".git"))).collect();
    assert!(git_files.iter().any(|p| p.ends_with("HEAD") && !p.ends_with("logs/HEAD")), "{git_files:?}");
    assert!(git_files.iter().any(|p| p.ends_with("logs/HEAD")), "{git_files:?}");
    assert!(git_files.iter().any(|p| p.ends_with("refs/heads/main")), "{git_files:?}");
    // Packed refs: the loose ref is gone and no longer watched, packed-refs is.
    git(&s.0, &["pack-refs", "--all"]);
    let (_, watch) = build_id::build_id(&s.0, "9.9.9");
    assert!(watch.iter().all(|p| p.exists()), "{watch:?}");
    assert!(watch.iter().any(|p| p.ends_with("packed-refs")), "{watch:?}");
    // An uncommitted change gives another identity.
    s.write("crates/protocol/src/lib.rs", "// changed\n");
    let (dirty, _) = build_id::build_id(&s.0, "9.9.9");
    assert!(dirty.starts_with(&id) && dirty.len() > id.len(), "{dirty}");
}
