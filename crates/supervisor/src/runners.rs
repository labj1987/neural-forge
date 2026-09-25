//! Compatibility-tool (Proton build) discovery and scoring, plus the system Wine
//! fallback. Reimplemented directly from upstream's own heuristic (not GPL-tainted,
//! not particularly novel either — a scoring rule over directory names) rather than
//! ported from its shell: user directories are scanned first so a user-installed
//! tool wins ties against a system copy, and a handful of known community-Proton
//! naming patterns are preferred over an unrecognized one.

use std::path::{Path, PathBuf};

pub struct Runner {
    pub name: String,
    pub path: PathBuf,
    score: u64,
}

fn candidate_dirs() -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
    let xdg_data_home = std::env::var("XDG_DATA_HOME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{home}/.local/share"));

    let mut dirs = vec![
        PathBuf::from(format!("{xdg_data_home}/Steam/compatibilitytools.d")),
        PathBuf::from(format!("{home}/.var/app/com.valvesoftware.Steam/data/Steam/compatibilitytools.d")),
        PathBuf::from(format!("{home}/snap/steam/common/.local/share/Steam/compatibilitytools.d")),
    ];

    // Steam Runtime rewrites XDG_DATA_DIRS inside its container, so the XDG defaults
    // are scanned unconditionally too, not just as a fallback when the variable is
    // unset -- matching upstream's own reasoning for doing the same in bash.
    if let Ok(data_dirs) = std::env::var("XDG_DATA_DIRS") {
        for entry in data_dirs.split(':').filter(|s| !s.is_empty()) {
            dirs.push(PathBuf::from(format!("{entry}/steam/compatibilitytools.d")));
        }
    }
    dirs.push(PathBuf::from("/usr/local/share/steam/compatibilitytools.d"));
    dirs.push(PathBuf::from("/usr/share/steam/compatibilitytools.d"));
    dirs
}

fn is_known_pattern(name: &str) -> bool {
    ["GE", "Cachy", "Sugar", "Tkg", "Wine-GE"].iter().any(|pat| name.contains(pat))
}

/// Base score by naming pattern, then nudged by whatever version number the name
/// itself contains (a higher point release should outrank a lower one with the same
/// pattern) -- same shape as upstream's `proton_score`, reimplemented.
fn score(name: &str) -> u64 {
    let base: u64 = if name.contains("Cachy") {
        1_000_000
    } else if name.contains("Wine-GE") {
        850_000
    } else if name.contains("GE") {
        900_000
    } else {
        700_000
    };

    // Pull out the first run of digits, and (if a '.' immediately follows it) the
    // run of digits after that -- e.g. "Proton-CachyOS-9.0" -> major=9, minor=0.
    let bytes = name.as_bytes();
    let digit_start = bytes.iter().position(|b| b.is_ascii_digit());
    let (major_num, minor_num) = match digit_start {
        Some(start) => {
            let major_end = bytes[start..].iter().take_while(|b| b.is_ascii_digit()).count() + start;
            let major_num: u64 = name[start..major_end].parse().unwrap_or(0);
            let minor_num = if bytes.get(major_end) == Some(&b'.') {
                let minor_start = major_end + 1;
                let minor_end = bytes[minor_start..].iter().take_while(|b| b.is_ascii_digit()).count() + minor_start;
                name[minor_start..minor_end].parse().unwrap_or(0)
            } else {
                0
            };
            (major_num, minor_num)
        }
        None => (0, 0),
    };

    base + major_num * 1000 + minor_num
}

/// Every discovered custom compatibility tool, best (highest-scored) first.
pub fn discover_proton() -> Vec<Runner> {
    let mut found = Vec::new();
    for base in candidate_dirs() {
        let Ok(entries) = std::fs::read_dir(&base) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let proton = path.join("proton");
            if !is_executable(&proton) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if !is_known_pattern(&name) {
                continue;
            }
            found.push(Runner { name: name.clone(), path: proton, score: score(&name) });
        }
    }
    found.sort_by(|a, b| b.score.cmp(&a.score));
    found
}

pub fn best_proton() -> Option<Runner> {
    discover_proton().into_iter().next()
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}

pub fn find_wine() -> Option<PathBuf> {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            let candidate = PathBuf::from(dir).join("wine");
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    for fallback in ["/usr/bin/wine", "/usr/local/bin/wine", "/opt/wine-devel/bin/wine"] {
        let path = PathBuf::from(fallback);
        if is_executable(&path) {
            return Some(path);
        }
    }
    None
}

/// The runner a first run selects when none is configured: the best discovered Proton build,
/// else system Wine, as `(runner_type, runner_path)`. `neural-forge-cli init` and the GUI's
/// Setup tab both call this, so they always pick the same one.
pub fn default_runner() -> Option<(&'static str, PathBuf)> {
    if let Some(proton) = best_proton() {
        return Some(("proton", proton.path));
    }
    find_wine().map(|wine| ("wine", wine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cachy_outranks_ge_outranks_generic() {
        assert!(score("Proton-CachyOS-9.0") > score("GE-Proton9-1"));
        assert!(score("GE-Proton9-1") > score("Proton-Unknown"));
    }

    #[test]
    fn higher_version_outranks_lower_within_same_pattern() {
        assert!(score("Proton-CachyOS-9.5") > score("Proton-CachyOS-9.1"));
    }

    #[test]
    fn known_patterns_are_recognized() {
        for name in ["GE-Proton9-1", "Proton-CachyOS-9.0", "Proton-Sugar", "Proton-Tkg", "Wine-GE-Proton9"] {
            assert!(is_known_pattern(name), "{name} should be recognized");
        }
        assert!(!is_known_pattern("Proton Experimental"));
    }
}
