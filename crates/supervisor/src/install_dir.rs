//! Where the helper executable and its vendored DXVK DLL live, relative to this
//! binary's own location — matches the AppImage layout the plan's "Build & packaging"
//! section describes (`usr/lib/neural-forge/helper/neural-forge-helper.exe`,
//! `usr/lib/neural-forge/dxvk/...`), not upstream's RPM tree.

use std::path::PathBuf;

fn candidate_install_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(explicit) = neural_forge_protocol::env::var("NEURAL_FORGE_INSTALL_DIR") {
        dirs.push(PathBuf::from(explicit));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(bin_dir) = exe.parent() {
            dirs.push(bin_dir.join("../lib/neural-forge"));
            dirs.push(bin_dir.join("../lib64/neural-forge"));
            dirs.push(bin_dir.to_path_buf());
        }
    }
    dirs.push(PathBuf::from("/usr/lib/neural-forge"));
    dirs.push(PathBuf::from("/usr/lib64/neural-forge"));
    dirs
}

pub fn helper_exe() -> Option<PathBuf> {
    for dir in candidate_install_dirs() {
        let candidate = dir.join("helper/neural-forge-helper.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

