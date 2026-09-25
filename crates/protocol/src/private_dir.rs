//! The runtime directory (`/tmp/neural-forge-<uid>/`) lives under the world-writable
//! `/tmp`, so nothing may be created or opened in it until it is known to be ours: a real
//! directory (not a symlink), owned by this uid, with no group or other permission bits.
//! The layer, the GUI/CLI mapping and the supervisor's pid and lock files all go through
//! this one predicate, and files inside the directory are opened relative to a descriptor
//! of the directory that was checked, never by re-resolving its path.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

fn uid() -> libc::uid_t {
    // SAFETY: getuid() takes no arguments and cannot fail.
    unsafe { libc::getuid() }
}

fn fstat(fd: &OwnedFd) -> Option<libc::stat> {
    // SAFETY: `st` is a plain out-parameter; zero-initializing it is always valid.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `fd` is an open descriptor for the call's duration.
    (unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } == 0).then_some(st)
}

fn is_private_dir(st: &libc::stat) -> bool {
    (st.st_mode & libc::S_IFMT) == libc::S_IFDIR && st.st_uid == uid() && (st.st_mode & (libc::S_IRWXG | libc::S_IRWXO)) == 0
}

/// `mkdir -p dir` with mode 0700 at each level it creates, ignoring every failure: the
/// ownership check that follows is what decides whether the result is usable.
fn mkdir_all(dir: &str) {
    let mut built = String::new();
    for part in dir.split('/') {
        if part.is_empty() {
            continue;
        }
        if dir.starts_with('/') || !built.is_empty() {
            built.push('/');
        }
        built.push_str(part);
        if let Ok(c) = CString::new(built.as_str()) {
            // SAFETY: `c` is a valid NUL-terminated C string for the call's duration.
            unsafe {
                libc::mkdir(c.as_ptr(), 0o700);
            }
        }
    }
}

/// Opens `dir` itself (never through a symlink in its last component) and returns the
/// descriptor if it names a private directory owned by this uid.
fn open_dir_checked(dir: &str) -> Option<OwnedFd> {
    let c_dir = CString::new(dir).ok()?;
    // SAFETY: `c_dir` is a valid NUL-terminated C string for the call's duration.
    let raw = unsafe { libc::open(c_dir.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
    if raw < 0 {
        return None;
    }
    // SAFETY: `raw` was just returned by a successful `open()` and is owned nowhere else.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    is_private_dir(&fstat(&fd)?).then_some(fd)
}

/// Splits `path` into its directory and final component. `None` when there is no
/// directory component or no file name.
fn split(path: &str) -> Option<(&str, &str)> {
    let slash = path.rfind('/')?;
    let (dir, name) = (&path[..slash], &path[slash + 1..]);
    if dir.is_empty() || name.is_empty() || name == "." || name == ".." {
        return None;
    }
    Some((dir, name))
}

/// Creates `path`'s parent directory if it is missing and reports whether it is private:
/// a directory (not a symlink), owned by this uid, with no group/other permission bits.
/// A path with no directory component has nothing to check and passes.
pub fn ensure_private_parent_dir(path: &str) -> bool {
    let Some(slash) = path.rfind('/') else { return true };
    let dir = &path[..slash];
    if dir.is_empty() {
        return true;
    }
    mkdir_all(dir);
    open_dir_checked(dir).is_some()
}

/// Like [`ensure_private_parent_dir`], but first takes a directory this uid already owns
/// back to mode 0700 (through a descriptor, so a symlink is never followed). An earlier
/// build created the runtime directory with the umask's mode, and the layer refuses such a
/// directory for good; a directory someone else owns is left alone and refused.
pub fn heal_private_parent_dir(path: &str) -> bool {
    if let Some((dir, _)) = split(path) {
        mkdir_all(dir);
        if let Ok(c_dir) = CString::new(dir) {
            // SAFETY: `c_dir` is a valid NUL-terminated C string for the call's duration.
            let raw = unsafe { libc::open(c_dir.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC) };
            if raw >= 0 {
                // SAFETY: `raw` was just returned by a successful `open()`.
                let fd = unsafe { OwnedFd::from_raw_fd(raw) };
                if let Some(st) = fstat(&fd) {
                    if (st.st_mode & libc::S_IFMT) == libc::S_IFDIR && st.st_uid == uid() && st.st_mode & 0o077 != 0 {
                        // SAFETY: `fd` is an open directory descriptor we own.
                        unsafe {
                            libc::fchmod(fd.as_raw_fd(), 0o700);
                        }
                    }
                }
            }
        }
    }
    ensure_private_parent_dir(path)
}

/// Opens the regular file `path` inside a private parent directory (see
/// [`ensure_private_parent_dir`]): the directory is opened and checked by descriptor, the
/// file is opened relative to that descriptor with `O_NOFOLLOW`, and the result must be a
/// regular file owned by this uid. `flags` are `open(2)` flags (`O_NOFOLLOW` and
/// `O_CLOEXEC` are always added); `mode` applies when `O_CREAT` creates the file.
/// `None` on any failure or when any check does not hold.
pub fn open_private_file(path: &str, flags: libc::c_int, mode: libc::mode_t) -> Option<OwnedFd> {
    let (dir, name) = split(path)?;
    mkdir_all(dir);
    let dir_fd = open_dir_checked(dir)?;
    let c_name = CString::new(name).ok()?;
    // SAFETY: `dir_fd` is an open directory descriptor and `c_name` a valid C string.
    let raw = unsafe { libc::openat(dir_fd.as_raw_fd(), c_name.as_ptr(), flags | libc::O_NOFOLLOW | libc::O_CLOEXEC, libc::c_uint::from(mode)) };
    if raw < 0 {
        return None;
    }
    // SAFETY: `raw` was just returned by a successful `openat()`.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let st = fstat(&fd)?;
    ((st.st_mode & libc::S_IFMT) == libc::S_IFREG && st.st_uid == uid()).then_some(fd)
}

/// Replaces `path` (inside a private parent directory) with `content`: written to a staged
/// file created next to it relative to the checked directory descriptor, then renamed over
/// it, so a reader never sees a truncated file.
pub fn write_private_file_atomic(path: &str, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let refused = || std::io::Error::new(std::io::ErrorKind::PermissionDenied, format!("{path}: parent directory is not private to this user"));
    let (dir, name) = split(path).ok_or_else(refused)?;
    mkdir_all(dir);
    let dir_fd = open_dir_checked(dir).ok_or_else(refused)?;
    static STAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = STAGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let staged = format!(".{name}.{}-{n}.tmp", std::process::id());
    let c_staged = CString::new(staged).map_err(std::io::Error::other)?;
    let c_name = CString::new(name).map_err(std::io::Error::other)?;
    // SAFETY: `dir_fd` is open and `c_staged` is a valid C string. Unlinking a leftover
    // staged file of this pid first keeps `O_EXCL` below from failing on a stale one.
    unsafe {
        libc::unlinkat(dir_fd.as_raw_fd(), c_staged.as_ptr(), 0);
    }
    // SAFETY: as above.
    let raw = unsafe {
        libc::openat(dir_fd.as_raw_fd(), c_staged.as_ptr(), libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC, 0o600 as libc::c_uint)
    };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `raw` was just returned by a successful `openat()`.
    let mut file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(raw) });
    let written = file.write_all(content).and_then(|()| file.sync_all());
    // SAFETY: both names are valid C strings relative to the open `dir_fd`.
    let result = written.and_then(|()| {
        if unsafe { libc::renameat(dir_fd.as_raw_fd(), c_staged.as_ptr(), dir_fd.as_raw_fd(), c_name.as_ptr()) } == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    });
    if result.is_err() {
        // SAFETY: as above.
        unsafe {
            libc::unlinkat(dir_fd.as_raw_fd(), c_staged.as_ptr(), 0);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn scratch_dir() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}/neural-forge-private-dir-test-{}-{n}", std::env::temp_dir().display(), std::process::id())
    }

    #[test]
    fn creates_a_private_directory_and_accepts_it() {
        let dir = scratch_dir();
        assert!(ensure_private_parent_dir(&format!("{dir}/rt/shm.bin")));
        assert_eq!(std::fs::metadata(format!("{dir}/rt")).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_a_shared_directory_and_heal_fixes_our_own() {
        let dir = scratch_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = format!("{dir}/shm.bin");
        assert!(!ensure_private_parent_dir(&path));
        assert!(open_private_file(&path, libc::O_RDWR | libc::O_CREAT, 0o600).is_none());
        assert!(heal_private_parent_dir(&path));
        assert!(open_private_file(&path, libc::O_RDWR | libc::O_CREAT, 0o600).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_a_symlinked_directory_and_a_symlinked_file() {
        let dir = scratch_dir();
        let real = format!("{dir}/real");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&real, format!("{dir}/link")).unwrap();
        assert!(!ensure_private_parent_dir(&format!("{dir}/link/shm.bin")));
        assert!(!heal_private_parent_dir(&format!("{dir}/link/shm.bin")));

        std::fs::write(format!("{real}/target"), "x").unwrap();
        std::os::unix::fs::symlink(format!("{real}/target"), format!("{real}/shm.bin")).unwrap();
        assert!(open_private_file(&format!("{real}/shm.bin"), libc::O_RDWR, 0).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn refuses_a_non_regular_file() {
        let dir = scratch_dir();
        std::fs::create_dir_all(format!("{dir}/sub")).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(open_private_file(&format!("{dir}/sub"), libc::O_RDONLY, 0).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn atomic_write_replaces_by_rename() {
        let dir = scratch_dir();
        let path = format!("{dir}/helper.pid");
        write_private_file_atomic(&path, b"1").unwrap();
        let before = std::fs::File::open(&path).unwrap();
        write_private_file_atomic(&path, b"22").unwrap();
        use std::io::Read;
        let mut old = String::new();
        (&before).read_to_string(&mut old).unwrap();
        assert_eq!(old, "1", "the old inode must be left intact");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "22");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no staged file left behind");
        std::fs::remove_dir_all(&dir).ok();
    }
}
