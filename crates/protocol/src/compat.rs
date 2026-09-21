//! Runtime-directory compatibility with layers older than 0.1.77.
//!
//! Until 0.1.76 the mapping lived in `/tmp/neuralforge-<uid>/`; it is now
//! `/tmp/neural-forge-<uid>/`. A game that loads an old layer (a stale manifest, a
//! copy pinned by hand) still opens the old path, and would otherwise talk to nobody:
//! the helper and GUI only watch the new one. Each side that owns the new directory
//! therefore hard-links the two files an old layer touches -- `shm.bin` and
//! `shm.bin.owner` -- into a private legacy directory, so both spellings name the *same
//! inodes* (same mapping, same ownership `flock`). A hard link, not a symlink, because
//! the old layer `lstat`s its parent directory and refuses anything that is not a real,
//! private directory it owns.
//!
//! Best effort and silent: a failure just means an old layer does not get bridged.

use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use crate::shm_runtime_dir;

/// The pre-0.1.77 directory matching the current runtime directory.
pub fn legacy_runtime_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(shm_runtime_dir());
    let name = dir.file_name()?.to_str()?;
    let legacy = name.strip_prefix("neural-forge-").map(|rest| format!("neuralforge-{rest}"))?;
    Some(dir.with_file_name(legacy))
}

/// Bridges `shm_path` (which must be the default `<runtime dir>/shm.bin`) to the legacy
/// directory. No-op for any other path (tests, explicit overrides).
pub fn link_legacy_runtime(shm_path: &str) {
    let path = Path::new(shm_path);
    let (Some(dir), Some(legacy)) = (path.parent(), legacy_runtime_dir()) else { return };
    if dir != Path::new(&shm_runtime_dir()) || path.file_name().and_then(|n| n.to_str()) != Some("shm.bin") {
        return;
    }
    let _ = link_into(dir, &legacy);
}

pub(crate) fn link_into(dir: &Path, legacy: &Path) -> std::io::Result<()> {
    // Real, private, ours -- or bail. Never follow or adopt anything else in /tmp.
    match std::fs::symlink_metadata(legacy) {
        Ok(_) => {}
        Err(_) => std::fs::DirBuilder::new().mode(0o700).create(legacy)?,
    }
    let meta = std::fs::symlink_metadata(legacy)?;
    // SAFETY: getuid() takes no arguments and cannot fail.
    if !meta.is_dir() || meta.uid() != unsafe { libc::getuid() } || meta.mode() & 0o077 != 0 {
        return Err(std::io::Error::other("legacy runtime dir is not a private directory of ours"));
    }

    // The ownership file must exist before it can be shared.
    let owner = dir.join("shm.bin.owner");
    if std::fs::symlink_metadata(&owner).is_err() {
        std::fs::OpenOptions::new().write(true).create(true).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(&owner)?;
    }

    // A live old-layer session holding the legacy channel keeps it: do not pull its
    // files out from under it (the new lease would then be a different inode).
    let legacy_owner = legacy.join("shm.bin.owner");
    if let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW).open(&legacy_owner) {
        use std::os::fd::AsRawFd;
        // SAFETY: valid fd for the call's duration.
        let free = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0;
        if !free && !same_inode(&legacy_owner, &owner) {
            return Ok(());
        }
    }

    for name in ["shm.bin", "shm.bin.owner"] {
        let (src, dst) = (dir.join(name), legacy.join(name));
        if std::fs::symlink_metadata(&src).is_err() || same_inode(&src, &dst) {
            continue;
        }
        let tmp = legacy.join(format!(".{name}.{}.link", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        std::fs::hard_link(&src, &tmp)?;
        // rename(2) atomically replaces a stale, unheld leftover from an old run.
        if let Err(e) = std::fs::rename(&tmp, &dst) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    }
    Ok(())
}

fn same_inode(a: &Path, b: &Path) -> bool {
    match (std::fs::symlink_metadata(a), std::fs::symlink_metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("neural-forge-compat-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn legacy_files_are_the_same_inodes() {
        let root = scratch("link");
        let (new, old) = (root.join("neural-forge-1"), root.join("neuralforge-1"));
        std::fs::DirBuilder::new().mode(0o700).create(&new).unwrap();
        std::fs::write(new.join("shm.bin"), b"map").unwrap();
        link_into(&new, &old).unwrap();
        assert!(same_inode(&new.join("shm.bin"), &old.join("shm.bin")));
        assert!(same_inode(&new.join("shm.bin.owner"), &old.join("shm.bin.owner")));
        // Idempotent.
        link_into(&new, &old).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_stale_unheld_legacy_file_is_replaced_and_a_held_one_is_kept() {
        let root = scratch("stale");
        let (new, old) = (root.join("neural-forge-1"), root.join("neuralforge-1"));
        std::fs::DirBuilder::new().mode(0o700).create(&new).unwrap();
        std::fs::DirBuilder::new().mode(0o700).create(&old).unwrap();
        std::fs::write(new.join("shm.bin"), b"new").unwrap();
        std::fs::write(old.join("shm.bin"), b"stale").unwrap();
        std::fs::write(old.join("shm.bin.owner"), b"").unwrap();
        link_into(&new, &old).unwrap();
        assert_eq!(std::fs::read(old.join("shm.bin")).unwrap(), b"new");

        // Now a live old session: separate inodes, legacy owner locked.
        std::fs::remove_file(old.join("shm.bin")).unwrap();
        std::fs::remove_file(old.join("shm.bin.owner")).unwrap();
        std::fs::write(old.join("shm.bin"), b"live").unwrap();
        let held = std::fs::OpenOptions::new().write(true).create(true).truncate(false).open(old.join("shm.bin.owner")).unwrap();
        use std::os::fd::AsRawFd;
        assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        link_into(&new, &old).unwrap();
        assert_eq!(std::fs::read(old.join("shm.bin")).unwrap(), b"live");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_non_private_legacy_dir_is_refused() {
        let root = scratch("perm");
        let (new, old) = (root.join("neural-forge-1"), root.join("neuralforge-1"));
        std::fs::DirBuilder::new().mode(0o700).create(&new).unwrap();
        std::fs::DirBuilder::new().mode(0o755).create(&old).unwrap();
        std::fs::write(new.join("shm.bin"), b"map").unwrap();
        assert!(link_into(&new, &old).is_err());
        assert!(!old.join("shm.bin").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
