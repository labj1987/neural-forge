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
            // Pre-0.1.77 install layout, so a not-yet-upgraded install tree keeps working.
            dirs.push(bin_dir.join("../lib/neuralforge"));
            dirs.push(bin_dir.to_path_buf());
        }
    }
    dirs.push(PathBuf::from("/usr/lib/neural-forge"));
    dirs.push(PathBuf::from("/usr/lib64/neural-forge"));
    dirs.push(PathBuf::from("/usr/lib/neuralforge"));
    dirs.push(PathBuf::from("/usr/lib64/neuralforge"));
    dirs
}

pub fn helper_exe() -> Option<PathBuf> {
    for dir in candidate_install_dirs() {
        // The pre-0.1.76 name is still accepted so an old, not-yet-upgraded install
        // tree (or NEURAL_FORGE_INSTALL_DIR pointing at one) keeps working.
        for name in ["helper/neural-forge-helper.exe", "helper/neuralforge-helper.exe"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

