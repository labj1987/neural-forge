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
    score: (u64, Vec<u64>),
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

/// Matched case-insensitively: CachyOS packages its Proton as `proton-cachyos-slr`, and
/// the community release tarballs extract to lowercase names too.
fn is_known_pattern(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["ge", "cachy", "sugar", "tkg", "wine-ge"].iter().any(|pat| name.contains(pat))
}

/// Base score by naming pattern, then every run of digits in the name as a version tuple
/// (a higher release outranks a lower one with the same pattern: `GE-Proton9-12` over
/// `GE-Proton9-2`, which a first-number-only score tied and left to directory order) --
/// same shape as upstream's `proton_score`, reimplemented.
fn score(name: &str) -> (u64, Vec<u64>) {
    let lower = name.to_ascii_lowercase();
    let base: u64 = if lower.contains("cachy") {
        1_000_000
    } else if lower.contains("wine-ge") {
        850_000
    } else if lower.contains("ge") {
        900_000
    } else {
        700_000
    };
    let version = lower
        .split(|c: char| !c.is_ascii_digit())
        .filter(|run| !run.is_empty())
        .map(|run| run.parse::<u64>().unwrap_or(u64::MAX))
        .collect();
    (base, version)
}

/// Best first: by score, then by name, both descending, so the order never depends on
/// the order `read_dir` happened to list the directories in.
fn sort_best_first(runners: &mut [Runner]) {
    runners.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| b.name.cmp(&a.name)));
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
    sort_best_first(&mut found);
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
    fn every_version_number_counts_not_just_the_first() {
        assert!(score("GE-Proton9-12") > score("GE-Proton9-2"));
        assert!(score("GE-Proton10-1") > score("GE-Proton9-27"));
        assert!(score("GE-Proton11-6-x86_64") > score("GE-Proton11-5-x86_64"));
        assert!(score("proton-cachyos-10.0-20250725-slr-x86_64_v3") > score("proton-cachyos-10.0-20250601-slr-x86_64_v3"));
    }

    #[test]
    fn lowercase_names_score_like_their_capitalised_forms() {
        assert_eq!(score("proton-cachyos-slr").0, score("Proton-CachyOS-SLR").0);
        assert!(score("proton-cachyos-slr") > score("GE-Proton11-6-x86_64"));
        assert_eq!(score("ge-proton9-1"), score("GE-Proton9-1"));
    }

    #[test]
    fn ties_are_broken_by_name_not_directory_order() {
        let runner = |name: &str| Runner { name: name.to_string(), path: PathBuf::new(), score: score(name) };
        let mut a = vec![runner("Proton-GE Latest"), runner("Proton-CachyOS Latest"), runner("GE-Proton11-6-x86_64"), runner("Proton-GE Beta")];
        let mut b = vec![runner("Proton-GE Beta"), runner("GE-Proton11-6-x86_64"), runner("Proton-CachyOS Latest"), runner("Proton-GE Latest")];
        sort_best_first(&mut a);
        sort_best_first(&mut b);
        let names = |v: &[Runner]| v.iter().map(|r| r.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&a), names(&b));
        assert_eq!(a[0].name, "Proton-CachyOS Latest");
    }

    #[test]
    fn known_patterns_are_recognized() {
        for name in [
            "GE-Proton9-1",
            "Proton-CachyOS-9.0",
            "Proton-Sugar",
            "Proton-Tkg",
            "Wine-GE-Proton9",
            "proton-cachyos-slr",
            "proton-cachyos-10.0-20250725-slr-x86_64_v3",
            "Proton-CachyOS Latest",
            "GE-Proton11-6-x86_64",
            "Proton-GE Latest",
        ] {
            assert!(is_known_pattern(name), "{name} should be recognized");
        }
        assert!(!is_known_pattern("Proton Experimental"));
    }
}
