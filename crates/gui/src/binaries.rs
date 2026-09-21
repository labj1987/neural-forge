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

/// Copies whichever of the known NGX DLLs are present in `src` into [`dir`]. Returns
/// how many were copied.
pub fn import_from(src: &std::path::Path) -> std::io::Result<usize> {
    let dest = dir();
    std::fs::create_dir_all(&dest)?;
    let mut copied = 0;
    for name in NGX_FILES {
        let from = src.join(name);
        if from.is_file() {
            std::fs::copy(&from, dest.join(name))?;
            copied += 1;
        }
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_from_copies_known_files_and_skips_unknown_ones() {
        let src = std::env::temp_dir().join(format!("neuralforge-binaries-test-src-{}", std::process::id()));
        let dest_home = std::env::temp_dir().join(format!("neuralforge-binaries-test-home-{}", std::process::id()));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("nvngx_dlssnr.dll"), b"model").unwrap();
        std::fs::write(src.join("nvapi64.dll"), b"nvapi").unwrap();
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
