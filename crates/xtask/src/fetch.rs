//! The pinned archives the build's tools come from: downloaded once into tools/native (or NEURAL_FORGE_NATIVE_TOOLS),
//! checked against the SHA-256 pinned in the task that names them before anything is unpacked, and stamped with that
//! hash, so a second run finds them up to date and a changed pin fetches again.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::sh::{self, Fail, Res, TempDir};

/// Where the tools are unpacked: NEURAL_FORGE_NATIVE_TOOLS, else tools/native in the repository.
pub fn tools_dir() -> PathBuf {
    sh::var("NEURAL_FORGE_NATIVE_TOOLS").map_or_else(|| crate::root().join("tools/native"), PathBuf::from)
}

/// Whether `dest` is the unpacked archive with this SHA-256.
pub fn stamped(dest: &Path, sha: &str) -> bool {
    sh::cat(&dest.join(".fetched-sha256")) == sha
}

pub fn create_dir_all(dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir).map_err(|e| Fail::new(1, format!("{}: {e}", dir.display())))
}

/// A staging directory `.fetch-<label>.XXXXXX` in the tools directory (the same filesystem the tool ends up on).
pub fn stage(tools: &Path, label: &str) -> Res<TempDir> {
    sh::mktemp_d(&tools.join(format!(".fetch-{label}."))).map_err(|e| Fail::new(1, format!("cannot create a staging directory in {}: {e}", tools.display())))
}

/// Downloads `url` to `<stage>/archive.tar.gz` and checks its SHA-256; `name` starts the lines it prints.
pub fn download(name: &str, url: &str, sha: &str, stage: &Path) -> Res<PathBuf> {
    println!("{name}: downloading {url}");
    let archive = stage.join("archive.tar.gz");
    sh::check(Command::new("curl").args(["-fsSL", "--retry", "3", "-o"]).arg(&archive).arg(url))?;
    let bytes = std::fs::read(&archive).map_err(|e| Fail::new(1, format!("{}: {e}", archive.display())))?;
    let got: String = Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect();
    if got != sha {
        return Err(Fail::new(1, format!("{name}: SHA-256 mismatch: expected {sha}, got {got}")));
    }
    Ok(archive)
}

pub fn untar(archive: &Path, into: &Path) -> Res<()> {
    sh::check(Command::new("tar").arg("-xzf").arg(archive).arg("-C").arg(into))
}

/// The entries of `dir` a shell's `*` gives: not the hidden ones, in name order.
pub fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| !n.to_string_lossy().starts_with('.'))).collect())
        .unwrap_or_default();
    found.sort();
    found
}

/// Stamps the unpacked `top` with the archive's hash and puts it in `dest`'s place.
pub fn install(top: &Path, sha: &str, dest: &Path) -> Res<()> {
    let stamp = top.join(".fetched-sha256");
    std::fs::write(&stamp, format!("{sha}\n")).map_err(|e| Fail::new(1, format!("{}: {e}", stamp.display())))?;
    if dest.is_dir() && !dest.is_symlink() {
        std::fs::remove_dir_all(dest).map_err(|e| Fail::new(1, format!("{}: {e}", dest.display())))?;
    } else if dest.exists() || dest.is_symlink() {
        std::fs::remove_file(dest).map_err(|e| Fail::new(1, format!("{}: {e}", dest.display())))?;
    }
    std::fs::rename(top, dest).map_err(|e| Fail::new(1, format!("cannot move {} to {}: {e}", top.display(), dest.display())))
}

/// Fetches an archive that unpacks to one directory whose name starts with `top_prefix`, into `dest`.
pub fn fetch_one_directory(name: &str, label: &str, url: &str, sha: &str, top_prefix: &str, dest: &Path) -> Res<()> {
    let tools = dest.parent().unwrap_or(Path::new("."));
    create_dir_all(tools)?;
    let stage = stage(tools, label)?;
    let archive = download(name, url, sha, stage.path())?;
    untar(&archive, stage.path())?;
    let top = entries(stage.path())
        .into_iter()
        .find(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(top_prefix)))
        .ok_or_else(|| Fail::new(1, format!("{name}: no {top_prefix}* directory in the archive")))?;
    install(&top, sha, dest)
}
