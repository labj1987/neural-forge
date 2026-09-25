//! Opens (or creates) the mapping on Linux and hands back a live `&ShmHeader` — the
//! "just attach and read/write settings" case the GUI and CLI both need, as opposed
//! to `neural_forge_layer`'s own copy of this same open/create/mmap dance (kept separate
//! there because it's entangled with that crate's request/response round-trip state
//! machine, which the GUI/CLI have no reason to duplicate or depend on).
//!
//! Linux-only (`cfg(unix)`, though in practice only ever built for Linux in this
//! workspace) — the Windows-side equivalent is `neural_forge_helper::shm`, a different
//! enough set of Win32 APIs that sharing this module across the OS boundary would
//! cost more in `cfg` noise than it would save in shared logic.

use std::os::fd::{AsRawFd, OwnedFd};
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
    /// The file is one of ours (the magic matches) but laid out by another build. Nothing
    /// was written: another process in the chain is out of date, and reinitializing the
    /// header would pull it out from under that process.
    WrongVersion { found: u32 },
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::Unavailable => write!(f, "the shared memory file could not be opened"),
            OpenError::WrongVersion { found } => write!(
                f,
                "shared memory is version {found}, this build speaks {SHM_VERSION} -- restart the helper and the game on the same Neural Forge version"
            ),
        }
    }
}

impl std::error::Error for OpenError {}

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
/// The parent directory must be private to this user (see [`crate::private_dir`]). A
/// directory this user already owns with a looser mode is first tightened to 0700: an
/// earlier build created it with the umask's mode, and the layer refuses such a
/// directory for the rest of its life. The file itself is opened relative to the checked
/// directory and must be a regular file this user owns.
///
/// A header with the wrong magic (a new, zero-filled file, or something that is not ours)
/// is initialized with defaults. A header with our magic and another version is left
/// untouched and reported as [`OpenError::WrongVersion`].
pub fn open_path(path: &str) -> Result<Mapping, OpenError> {
    if !crate::isolated_path(path) || !crate::private_dir::heal_private_parent_dir(path) {
        return Err(OpenError::Unavailable);
    }
    let fd = crate::private_dir::open_private_file(path, libc::O_RDWR | libc::O_CREAT, 0o600).ok_or(OpenError::Unavailable)?;

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

    Ok(Mapping { _fd: fd, header, freshly_created })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn scratch_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}/neural-forge-mapping-test-{}-{n}/shm.bin", std::env::temp_dir().display(), std::process::id())
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

    #[test]
    fn a_file_with_foreign_magic_is_initialized() {
        let path = scratch_path();
        assert!(crate::private_dir::ensure_private_parent_dir(&path));
        std::fs::write(&path, vec![0xAB; 64]).unwrap();
        let mapping = open_path(&path).expect("a foreign file is reinitialized");
        assert!(mapping.freshly_created && mapping.header().is_valid());
        cleanup(&path);
    }

    #[test]
    fn a_directory_shared_with_other_users_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let path = scratch_path();
        let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        std::fs::create_dir_all(&dir).unwrap();
        // Our own directory with a loose mode is tightened, not refused.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(open_path(&path).is_ok());
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        // A symlink in place of the directory is refused.
        let link = format!("{}-link", dir.display());
        std::os::unix::fs::symlink(&dir, &link).unwrap();
        assert_eq!(open_path(&format!("{link}/shm.bin")).err(), Some(OpenError::Unavailable));
        std::fs::remove_file(&link).ok();
        cleanup(&path);
    }
}
