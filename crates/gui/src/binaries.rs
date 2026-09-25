//! Importing NVIDIA's NGX DLLs into `neural_forge_supervisor::paths::binaries_dir()` --
//! the same destination `neural-forge-cli import-binaries` uses. The path itself now comes
//! from the shared `neural-forge-supervisor` crate; only the actual file-copy loop is kept
//! here, since it's a handful of lines with nothing else in `supervisor` needing it.

const NGX_FILES: [&str; 3] = ["nvngx_dlssnr.dll", "nvngx.dll", "nvapi64.dll"];

pub fn dir() -> std::path::PathBuf {
    std::path::PathBuf::from(neural_forge_supervisor::paths::binaries_dir())
}

/// Every known NGX file, and whether it's currently present in [`dir`] -- the
/// per-file status the Setup tab's binaries group shows, the same list `import_from`
/// itself copies from.
pub fn status() -> Vec<(&'static str, bool)> {
    let dest = dir();
    NGX_FILES.iter().map(|&name| (name, dest.join(name).is_file())).collect()
}

/// Smallest file accepted as one of the DLLs. The real ones are megabytes; this only
/// rejects empty or truncated files and stray text files with the right name.
const MIN_DLL_SIZE: u64 = 4096;

/// Whether `path` looks like a Windows DLL: at least [`MIN_DLL_SIZE`] bytes and starting
/// with the `MZ` DOS header every PE file has.
fn looks_like_dll(path: &std::path::Path) -> std::io::Result<bool> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    if file.metadata()?.len() < MIN_DLL_SIZE {
        return Ok(false);
    }
    let mut magic = [0u8; 2];
    file.read_exact(&mut magic)?;
    Ok(&magic == b"MZ")
}

/// Copies whichever of the known NGX DLLs are present in `src` into [`dir`]. Returns
/// how many were copied. Every file found is checked first, and nothing is copied if
/// any of them is not a DLL.
pub fn import_from(src: &std::path::Path) -> std::io::Result<usize> {
    let found: Vec<&str> = NGX_FILES.into_iter().filter(|name| src.join(name).is_file()).collect();
    for name in &found {
        if !looks_like_dll(&src.join(name))? {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{name} is not a Windows DLL (no MZ header, or too small)")));
        }
    }
    let dest = dir();
    std::fs::create_dir_all(&dest)?;
    for name in &found {
        std::fs::copy(src.join(name), dest.join(name))?;
    }
    Ok(found.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both tests point `XDG_DATA_HOME` at their own scratch dir; the variable is process-wide.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fake_dll() -> Vec<u8> {
        let mut bytes = vec![0u8; MIN_DLL_SIZE as usize];
        bytes[..2].copy_from_slice(b"MZ");
        bytes
    }

    #[test]
    fn import_from_rejects_files_that_are_not_dlls_and_copies_nothing() {
        let _env = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let src = std::env::temp_dir().join(format!("neural-forge-binaries-bad-src-{}", std::process::id()));
        let dest_home = std::env::temp_dir().join(format!("neural-forge-binaries-bad-home-{}", std::process::id()));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("nvapi64.dll"), fake_dll()).unwrap();
        std::fs::write(src.join("nvngx_dlssnr.dll"), b"not a dll").unwrap();
        let mut wrong_magic = fake_dll();
        wrong_magic[..2].copy_from_slice(b"PK");
        std::fs::write(src.join("nvngx.dll"), wrong_magic).unwrap();

        let prev_xdg = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &dest_home);

        assert_eq!(import_from(&src).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
        assert!(!dir().join("nvapi64.dll").exists());

        std::fs::write(src.join("nvngx_dlssnr.dll"), fake_dll()).unwrap();
        assert_eq!(import_from(&src).unwrap_err().kind(), std::io::ErrorKind::InvalidData);

        match prev_xdg {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&dest_home).ok();
    }

    #[test]
    fn import_from_copies_known_files_and_skips_unknown_ones() {
        let _env = ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let src = std::env::temp_dir().join(format!("neural-forge-binaries-test-src-{}", std::process::id()));
        let dest_home = std::env::temp_dir().join(format!("neural-forge-binaries-test-home-{}", std::process::id()));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("nvngx_dlssnr.dll"), fake_dll()).unwrap();
        std::fs::write(src.join("nvapi64.dll"), fake_dll()).unwrap();
        std::fs::write(src.join("unrelated.txt"), b"ignore me").unwrap();

        let prev_xdg = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", &dest_home);

        let copied = import_from(&src).unwrap();
        assert_eq!(copied, 2);
        assert!(dir().join("nvngx_dlssnr.dll").is_file());
        assert!(dir().join("nvapi64.dll").is_file());
        assert!(!dir().join("unrelated.txt").exists());
        assert!(!dir().join("nvngx.dll").exists());

        match prev_xdg {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&dest_home).ok();
    }
}
