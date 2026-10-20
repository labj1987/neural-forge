//! Opens (or creates) the mapping on Linux and hands back a live `&ShmHeader` — the
//! "just attach and read/write settings" case the GUI and CLI both need, as opposed
//! to `neural_forge_layer`'s own copy of this same open/create/mmap dance (kept separate
//! there because it's entangled with that crate's request/response round-trip state
//! machine, which the GUI/CLI have no reason to duplicate or depend on).
//!
//! Linux only (the crate is; see its docs).

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::sync::atomic::Ordering;

use crate::{shm_default_path, shm_total_bytes, ShmHeader, HEADER_BYTES, SHM_MAGIC, SHM_VERSION};

pub struct Mapping {
    _fd: OwnedFd,
    header: *mut ShmHeader,
    /// Whether this call is what created the mapping (no header with our magic to
    /// reattach to) rather than reattaching to one an already-running instance owns.
    /// A caller that persists settings to `config.ini` (the GUI) uses this to know
    /// when it's safe to apply persisted overrides -- doing that on a warm reattach
    /// would fight whatever the already-running instance currently has live.
    pub freshly_created: bool,
}

// SAFETY: same reasoning as `ShmHeader` itself being `Sync` -- every access through
// `header` goes through its own atomics/seqlock-guarded accessors.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn header(&self) -> &ShmHeader {
        // SAFETY: `header` was mmap'd for at least `HEADER_BYTES` in `open_path()` and is
        // never unmapped before `self` is dropped (there is no `Drop` impl that
        // unmaps it -- deliberately: the mapping is meant to outlive this handle's
        // owner for as long as the process runs, matching a settings GUI/CLI's
        // lifetime expectations, so leaking the mapping until process exit is fine).
        unsafe { &*self.header }
    }
}

/// Why [`open`]/[`open_path`] could not hand back a mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenError {
    /// The path is outside this project's namespace, its directory is not private to
    /// this user, or an open/size/map call failed.
    Unavailable,
    /// The file exists and holds something else ([`ChannelFile::Foreign`]). Nothing was
    /// written and its length is unchanged: a channel path pointed at an unrelated file.
    Foreign,
    /// The file is one of ours (the magic matches) but laid out by another build. Nothing
    /// was written: another process in the chain is out of date, and reinitializing the
    /// header would pull it out from under that process.
    WrongVersion { found: u32 },
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::Unavailable => write!(f, "the shared memory file could not be opened"),
            OpenError::Foreign => write!(f, "the shared memory path names a file that is not a Neural Forge mapping; it was left untouched (point shm= / NEURAL_FORGE_SHM at another path, or remove the file)"),
            OpenError::WrongVersion { found } => write!(
                f,
                "shared memory is version {found}, this build speaks {SHM_VERSION} -- the next game started with this version re-creates it (close a game still running an older one)"
            ),
        }
    }
}

impl std::error::Error for OpenError {}

/// What an opened channel file holds, decided from its length and header bytes before anything
/// is written to it or its length changed ([`channel_file`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelFile {
    /// Empty: just created by this open, or by a peer that has not sized it yet.
    New,
    /// It starts with [`SHM_MAGIC`]: a Neural Forge mapping (of this build's layout or another's).
    Ours,
    /// Exactly a mapping's length with an all-zero header: a peer has sized it and not yet
    /// written the header.
    Blank,
    /// Anything else. Not ours to resize or initialise.
    Foreign,
}

/// Classifies the open file `fd` without changing it. `None` when it cannot be examined.
///
/// Opening a channel sizes the file to [`shm_total_bytes`] and writes a header over its start,
/// so the path must not be allowed to name an unrelated file: only [`ChannelFile::New`],
/// [`ChannelFile::Ours`] and [`ChannelFile::Blank`] may be sized or initialised.
pub fn channel_file(fd: BorrowedFd<'_>) -> Option<ChannelFile> {
    // SAFETY: `st` is a plain out-parameter; zero-initializing it is always valid.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `fd` is an open descriptor for the call's duration.
    if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
        return None;
    }
    let size = usize::try_from(st.st_size).ok()?;
    if size == 0 {
        return Some(ChannelFile::New);
    }
    let mut head = vec![0u8; size.min(HEADER_BYTES)];
    let mut read = 0;
    while read < head.len() {
        // SAFETY: the destination is `head.len() - read` writable bytes; `pread` moves no
        // file offset.
        let n = unsafe { libc::pread(fd.as_raw_fd(), head[read..].as_mut_ptr().cast(), head.len() - read, read as libc::off_t) };
        if n <= 0 {
            return None;
        }
        read += n as usize;
    }
    if head.len() >= 4 && head[..4] == SHM_MAGIC.to_ne_bytes() {
        Some(ChannelFile::Ours)
    } else if size == shm_total_bytes() && head.iter().all(|&b| b == 0) {
        Some(ChannelFile::Blank)
    } else {
        Some(ChannelFile::Foreign)
    }
}

/// Runs `f` holding an exclusive `flock` on the channel file `fd`: classifying, sizing and
/// initialising a channel happen under it in every process that opens one (this module and the
/// layer), so two processes finding the same new file cannot both initialise it (the header has a
/// single writer per field), nor can one read a header another is still writing. The lock is the
/// file's own, separate from the ownership lease (`<channel>.owner`), and held only for those steps.
/// A read-only look ([`open_existing`]) takes it shared.
pub fn with_init_lock<R>(fd: BorrowedFd<'_>, f: impl FnOnce() -> R) -> R {
    with_lock(fd, libc::LOCK_EX, f)
}

fn with_lock<R>(fd: BorrowedFd<'_>, operation: i32, f: impl FnOnce() -> R) -> R {
    struct Unlock(i32);
    impl Drop for Unlock {
        fn drop(&mut self) {
            // SAFETY: plain call on a descriptor that is open for the guard's lifetime.
            unsafe { libc::flock(self.0, libc::LOCK_UN) };
        }
    }
    // SAFETY: plain call on an open descriptor. Interrupted waits are retried; any other failure
    // (no flock support) proceeds unlocked, as before.
    while unsafe { libc::flock(fd.as_raw_fd(), operation) } != 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {}
    let _unlock = Unlock(fd.as_raw_fd());
    f()
}

/// The `version` word of the header at the start of the file `fd`, read without mapping it; `None`
/// when the file is too short to hold it or cannot be read.
fn header_version(fd: BorrowedFd<'_>) -> Option<u32> {
    let at = std::mem::offset_of!(ShmHeader, version);
    let mut word = [0u8; 4];
    // SAFETY: the destination is 4 writable bytes; `pread` moves no file offset.
    let n = unsafe { libc::pread(fd.as_raw_fd(), word.as_mut_ptr().cast(), word.len(), at as libc::off_t) };
    (n == word.len() as isize).then(|| u32::from_ne_bytes(word))
}

/// A read-only look at an existing channel, for diagnostics (`doctor`, the report, `shmctl
/// status`): nothing is created, sized, initialised or healed, and nothing written through it
/// reaches the file (the mapping is private). `Ok(None)` when there is no channel file yet (no
/// game and no front end has opened one this boot).
pub fn open_existing(path: &str) -> Result<Option<Mapping>, OpenError> {
    if !crate::isolated_path(path) {
        return Err(OpenError::Unavailable);
    }
    if std::fs::symlink_metadata(path).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound) {
        return Ok(None);
    }
    let fd = crate::private_dir::open_private_file(path, libc::O_RDONLY, 0).ok_or(OpenError::Unavailable)?;
    // Shared under the init lock: a process initialising the file (which withdraws the magic while it
    // writes the header) finishes first, so a channel being created is not read as a foreign file.
    let header = with_lock(fd.as_fd(), libc::LOCK_SH, || look(fd.as_fd()))?;
    Ok(header.map(|header| Mapping { _fd: fd, header, freshly_created: false }))
}

/// [`open_existing`]'s steps under the shared init lock: classify, map privately, check the version.
fn look(fd: BorrowedFd<'_>) -> Result<Option<*mut ShmHeader>, OpenError> {
    match channel_file(fd) {
        Some(ChannelFile::Ours) => {}
        Some(ChannelFile::New | ChannelFile::Blank) => return Ok(None),
        Some(ChannelFile::Foreign) => return Err(OpenError::Foreign),
        None => return Err(OpenError::Unavailable),
    }
    // SAFETY: plain out-parameter.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `fd` is open.
    if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 || (st.st_size as usize) < HEADER_BYTES {
        return Err(OpenError::Unavailable);
    }
    // SAFETY: the file is at least `HEADER_BYTES` long. A private writable mapping of a read-only
    // descriptor: loads see the file, any store stays in this process. Never unmapped (see
    // `Mapping::header`).
    let map = unsafe { libc::mmap(std::ptr::null_mut(), HEADER_BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE, fd.as_raw_fd(), 0) };
    if map == libc::MAP_FAILED {
        return Err(OpenError::Unavailable);
    }
    let header = map.cast::<ShmHeader>();
    // SAFETY: just mapped, large enough for `ShmHeader`.
    let found = unsafe { &*header }.version.load(Ordering::Acquire);
    if found != SHM_VERSION {
        // SAFETY: mapped above with exactly this length; nothing derived from it escapes.
        unsafe { libc::munmap(map, HEADER_BYTES) };
        return Err(OpenError::WrongVersion { found });
    }
    Ok(Some(header))
}

/// Opens the mapping at `$NEURAL_FORGE_SHM` (or the default runtime path), creating it if
/// necessary. A caller that has a `config.ini` resolves the path once through the
/// supervisor (`neural_forge_supervisor::channel_path`) and uses [`open_path`] instead,
/// so every process it starts or reports on names the same file.
pub fn open() -> Result<Mapping, OpenError> {
    let path = crate::env::var("NEURAL_FORGE_SHM").filter(|s| !s.is_empty()).unwrap_or_else(shm_default_path);
    open_path(&path)
}

/// Opens (or creates) the mapping at `path`.
///
/// The parent directory must be private to this user (see [`crate::private_dir`]). This
/// project's own runtime directory, when this user already owns it with a looser mode, is first
/// tightened to 0700: an earlier build created it with the umask's mode, and the layer refuses
/// such a directory for the rest of its life. Any other directory is used only if it is private
/// already, and never has its mode changed. The file itself is opened relative to the checked
/// directory and must be a regular file this user owns.
///
/// A new file (empty, or sized by a peer and still all zero) is sized and initialized with
/// defaults. A header with our magic and another version is left untouched and reported as
/// [`OpenError::WrongVersion`]. A file holding anything else is not a mapping: it is neither
/// resized nor written, and reported as [`OpenError::Foreign`].
pub fn open_path(path: &str) -> Result<Mapping, OpenError> {
    if !crate::isolated_path(path) || !crate::private_dir::heal_private_parent_dir(path) {
        return Err(OpenError::Unavailable);
    }
    let fd = crate::private_dir::open_private_file(path, libc::O_RDWR | libc::O_CREAT, 0o600).ok_or(OpenError::Unavailable)?;
    let opened = with_init_lock(fd.as_fd(), || open_locked(fd.as_fd()))?;
    Ok(Mapping { _fd: fd, header: opened.0, freshly_created: opened.1 })
}

/// [`open_path`]'s steps under the file's init lock: classify, size, map, initialise or check.
fn open_locked(fd: BorrowedFd<'_>) -> Result<(*mut ShmHeader, bool), OpenError> {
    // Before the file is sized or written: is it one of ours (at this build's version), or new?
    match channel_file(fd) {
        Some(ChannelFile::New | ChannelFile::Blank) => {}
        Some(ChannelFile::Ours) => match header_version(fd) {
            Some(SHM_VERSION) => {}
            // Another build's layout (a shorter one, for an older build) is left at its length.
            Some(found) => return Err(OpenError::WrongVersion { found }),
            None => return Err(OpenError::Unavailable),
        },
        Some(ChannelFile::Foreign) => return Err(OpenError::Foreign),
        None => return Err(OpenError::Unavailable),
    }

    let total = shm_total_bytes();
    // SAFETY: `st` is a plain out-parameter; zero-initializing it is always valid.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let needs_truncate = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 || (st.st_size as usize) < total;
    if needs_truncate && unsafe { libc::ftruncate(fd.as_raw_fd(), total as libc::off_t) } != 0 {
        return Err(OpenError::Unavailable);
    }

    // SAFETY: `fd` is open and sized to at least `HEADER_BYTES`; mapping only that
    // many bytes is always in-bounds. The mapping is intentionally never unmapped
    // (see `Mapping::header`'s doc comment), so the returned pointer stays valid for
    // as long as anything derived from it is used.
    let map = unsafe {
        libc::mmap(std::ptr::null_mut(), HEADER_BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0)
    };
    if map == libc::MAP_FAILED {
        return Err(OpenError::Unavailable);
    }

    let header = map.cast::<ShmHeader>();
    // SAFETY: just mapped above, `HEADER_BYTES` is large enough for `ShmHeader`.
    let hdr = unsafe { &*header };
    let freshly_created = hdr.magic.load(Ordering::Relaxed) != SHM_MAGIC;
    if freshly_created {
        hdr.init_defaults();
    } else if hdr.version.load(Ordering::Relaxed) != SHM_VERSION {
        let found = hdr.version.load(Ordering::Relaxed);
        // SAFETY: mapped above with exactly this length; nothing derived from it escapes.
        unsafe {
            libc::munmap(map, HEADER_BYTES);
        }
        return Err(OpenError::WrongVersion { found });
    }
    Ok((header, freshly_created))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn scratch_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}/mapping-{}-{n}/shm.bin", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/test-scratch").display(), std::process::id())
    }

    fn cleanup(path: &str) {
        std::fs::remove_dir_all(std::path::Path::new(path).parent().unwrap()).ok();
    }

    #[test]
    fn open_path_creates_a_valid_mapping_with_real_defaults() {
        let path = scratch_path();
        let mapping = open_path(&path).expect("open_path should succeed");
        assert!(mapping.freshly_created);
        assert!(mapping.header().is_valid());
        assert!(mapping.header().neural_enabled());
        assert_eq!(f32::from_bits(mapping.header().intensity_bits.load(Ordering::Relaxed)), 1.0);
        cleanup(&path);
    }

    #[test]
    fn open_existing_creates_nothing_and_writes_nothing() {
        let path = scratch_path();
        assert!(open_existing(&path).unwrap().is_none(), "no file: no channel");
        assert!(!std::path::Path::new(&path).exists(), "and none was created");
        let live = open_path(&path).unwrap();
        live.header().intensity_bits.store(0.25f32.to_bits(), Ordering::Relaxed);
        let look = open_existing(&path).unwrap().unwrap();
        assert_eq!(f32::from_bits(look.header().intensity_bits.load(Ordering::Relaxed)), 0.25);
        look.header().intensity_bits.store(3.0f32.to_bits(), Ordering::Relaxed);
        assert_eq!(f32::from_bits(live.header().intensity_bits.load(Ordering::Relaxed)), 0.25, "a store through the look stays private");
        live.header().version.store(SHM_VERSION + 1, Ordering::Relaxed);
        assert_eq!(open_existing(&path).err(), Some(OpenError::WrongVersion { found: SHM_VERSION + 1 }));
        cleanup(&path);
    }

    /// Many openers racing on one new file: exactly one initialises it, and none sees a header
    /// that is not either absent or complete.
    #[test]
    fn concurrent_openers_initialise_a_new_file_once() {
        let path = scratch_path();
        let fresh: usize = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8).map(|_| s.spawn(|| open_path(&path).map(|m| usize::from(m.freshly_created)))).collect();
            handles.into_iter().map(|h| h.join().unwrap().expect("every opener succeeds")).sum()
        });
        assert_eq!(fresh, 1);
        cleanup(&path);
    }

    #[test]
    fn open_path_reattaches_to_an_existing_mapping_without_resetting_it() {
        let path = scratch_path();
        {
            let mapping = open_path(&path).unwrap();
            mapping.header().intensity_bits.store(0.42f32.to_bits(), Ordering::Relaxed);
        }
        let reattached = open_path(&path).expect("re-open should succeed");
        assert!(!reattached.freshly_created);
        assert_eq!(f32::from_bits(reattached.header().intensity_bits.load(Ordering::Relaxed)), 0.42);
        cleanup(&path);
    }

    #[test]
    fn a_header_at_another_version_is_reported_and_left_untouched() {
        let path = scratch_path();
        {
            let mapping = open_path(&path).unwrap();
            let h = mapping.header();
            h.version.store(SHM_VERSION + 1, Ordering::Relaxed);
            h.seq_req.store(77, Ordering::Relaxed);
            h.intensity_bits.store(0.25f32.to_bits(), Ordering::Relaxed);
        }
        assert_eq!(open_path(&path).err(), Some(OpenError::WrongVersion { found: SHM_VERSION + 1 }));
        // Read the raw bytes back: nothing may have been reinitialized.
        let bytes = std::fs::read(&path).unwrap();
        let word = |offset: usize| u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(word(std::mem::offset_of!(ShmHeader, version)), SHM_VERSION + 1);
        assert_eq!(word(std::mem::offset_of!(ShmHeader, seq_req)), 77);
        assert_eq!(word(std::mem::offset_of!(ShmHeader, intensity_bits)), 0.25f32.to_bits());
        cleanup(&path);
    }

    /// A look never maps a file too short to hold a header (a load past its end would be SIGBUS),
    /// even one that starts with the magic.
    #[test]
    fn a_look_refuses_a_file_too_short_for_a_header() {
        let path = scratch_path();
        assert!(crate::private_dir::ensure_private_parent_dir(&path));
        std::fs::write(&path, SHM_MAGIC.to_ne_bytes()).unwrap();
        assert_eq!(open_existing(&path).err(), Some(OpenError::Unavailable));
        std::fs::write(&path, b"").unwrap();
        assert!(open_existing(&path).unwrap().is_none(), "an empty file is a channel not made yet");
        cleanup(&path);
    }

    /// A header of ours at another version, in a file shorter than this build's mapping (an older
    /// build's layout): refused before anything is sized, so the file keeps its length.
    #[test]
    fn a_shorter_file_at_another_version_is_not_grown() {
        let path = scratch_path();
        assert!(crate::private_dir::ensure_private_parent_dir(&path));
        let mut bytes = vec![0u8; HEADER_BYTES];
        bytes[..4].copy_from_slice(&SHM_MAGIC.to_ne_bytes());
        let at = std::mem::offset_of!(ShmHeader, version);
        bytes[at..at + 4].copy_from_slice(&(SHM_VERSION - 1).to_ne_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(open_path(&path).err(), Some(OpenError::WrongVersion { found: SHM_VERSION - 1 }));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), HEADER_BYTES as u64, "the file's length is unchanged");
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        cleanup(&path);
    }

    /// A look at the channel while another process is initialising it (the magic withdrawn, the header
    /// half written) waits for it, and sees the finished channel, not a foreign file.
    #[test]
    fn a_look_during_initialisation_waits_for_it() {
        use std::os::fd::AsFd;
        let path = scratch_path();
        assert!(crate::private_dir::ensure_private_parent_dir(&path));
        let file = std::fs::File::options().read(true).write(true).create(true).truncate(false).open(&path).unwrap();
        file.set_len(shm_total_bytes() as u64).unwrap();
        let initialised = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let look = with_init_lock(file.as_fd(), || {
            // Mid-initialisation: the file has its length, some of the header is written, no magic.
            std::os::unix::fs::FileExt::write_all_at(&file, &1u32.to_ne_bytes(), 100).unwrap();
            assert_eq!(channel_file(file.as_fd()), Some(ChannelFile::Foreign), "what a look without the lock would see");
            let (path, done) = (path.clone(), initialised.clone());
            let look = std::thread::spawn(move || {
                let found = open_existing(&path).map(|m| m.map(|m| m.header().is_valid()));
                (found, done.load(Ordering::Acquire))
            });
            std::thread::sleep(std::time::Duration::from_millis(20));
            // SAFETY: the file is `shm_total_bytes()` long; this test's own mapping, unmapped below.
            let map = unsafe { libc::mmap(std::ptr::null_mut(), HEADER_BYTES, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0) };
            assert_ne!(map, libc::MAP_FAILED);
            // SAFETY: just mapped, large enough for the header; nothing derived from it outlives it.
            unsafe {
                (*map.cast::<ShmHeader>()).init_defaults();
                libc::munmap(map, HEADER_BYTES);
            }
            initialised.store(true, Ordering::Release);
            look
        });
        let (found, after) = look.join().unwrap();
        assert_eq!(found, Ok(Some(true)), "a valid channel, not a foreign file");
        assert!(after, "the look waited for the initialisation");
        cleanup(&path);
    }

    /// A channel path that names an existing, unrelated file: opening it would size the file to
    /// the mapping's length and write a header over its start. It is refused, with its contents
    /// and length as they were, and its directory's permissions as they were.
    #[test]
    fn an_unrelated_existing_file_is_refused_and_left_untouched() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |dir: &std::path::Path| std::fs::metadata(dir).unwrap().permissions().mode() & 0o777;
        for content in [vec![0xAB; 64], b"notes\n".to_vec(), vec![0u8; 4096], vec![7u8; HEADER_BYTES + 1]] {
            let path = scratch_path();
            let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
            assert!(crate::private_dir::ensure_private_parent_dir(&path));
            std::fs::write(&path, &content).unwrap();
            assert_eq!(open_path(&path).err(), Some(OpenError::Foreign));
            assert_eq!(std::fs::read(&path).unwrap(), content, "contents and length unchanged");
            assert_eq!(mode(&dir), 0o700);
            // The same file in a directory other users can read: refused for the directory, and
            // the directory's mode is not this project's to change.
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
            assert_eq!(open_path(&path).err(), Some(OpenError::Unavailable));
            assert_eq!(mode(&dir), 0o755, "the parent directory's permissions are unchanged");
            assert_eq!(std::fs::read(&path).unwrap(), content);
            cleanup(&path);
        }
    }

    /// What may be sized and initialised: an empty file, one of ours, and a file a peer has
    /// just sized to the mapping's length and not written yet.
    #[test]
    fn new_blank_and_our_own_files_are_told_from_foreign_ones() {
        use std::os::fd::AsFd;
        let path = scratch_path();
        assert!(crate::private_dir::ensure_private_parent_dir(&path));
        let kind = |path: &str| channel_file(std::fs::File::open(path).unwrap().as_fd());
        std::fs::write(&path, b"").unwrap();
        assert_eq!(kind(&path), Some(ChannelFile::New));
        // Sized by a peer, header not yet written: sparse and all zero.
        std::fs::File::options().write(true).open(&path).unwrap().set_len(shm_total_bytes() as u64).unwrap();
        assert_eq!(kind(&path), Some(ChannelFile::Blank));
        let mapping = open_path(&path).expect("a blank file of the mapping's length is initialised");
        assert!(mapping.freshly_created && mapping.header().is_valid());
        assert_eq!(kind(&path), Some(ChannelFile::Ours));
        // A zero-filled file of any other length is someone's data, not a mapping in the making.
        std::fs::File::options().write(true).open(&path).unwrap().set_len(0).unwrap();
        std::fs::File::options().write(true).open(&path).unwrap().set_len(1 << 20).unwrap();
        assert_eq!(kind(&path), Some(ChannelFile::Foreign));
        assert_eq!(open_path(&path).err(), Some(OpenError::Foreign));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 1 << 20);
        cleanup(&path);
    }

    #[test]
    fn a_directory_shared_with_other_users_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        // Named like a runtime directory, but not the one `shm_runtime_dir` names: refused as it
        // is, its mode untouched (only the real runtime directory is healed, `private_dir`).
        let scratch = scratch_path();
        let path = format!("{}/neural-forge-heal/shm.bin", scratch.trim_end_matches("/shm.bin"));
        let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(open_path(&path).err(), Some(OpenError::Unavailable));
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o755);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(open_path(&path).is_ok());
        // A symlink in place of the directory is refused.
        let link = format!("{}-link", dir.display());
        std::os::unix::fs::symlink(&dir, &link).unwrap();
        assert_eq!(open_path(&format!("{link}/shm.bin")).err(), Some(OpenError::Unavailable));
        std::fs::remove_file(&link).ok();
        cleanup(&scratch);
    }
}
