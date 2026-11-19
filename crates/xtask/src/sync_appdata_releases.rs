//! Regenerates the <releases> list in the AppStream metadata from CHANGELOG.md.
//!
//! The changelog has one `## <version> — <date>` heading per release, so it is the
//! in-tree source of truth (git tags are not reliably present in a shallow CI checkout).
//! Also verifies that the newest release is the version in Cargo.toml.
//!
//!   cargo xtask sync-appdata-releases           rewrite the metadata in place
//!   cargo xtask sync-appdata-releases --check   exit 1 if it is out of date (for CI)

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use regex::{NoExpand, Regex};

fn appdata(root: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> =
        std::fs::read_dir(root.join("data")).ok()?.flatten().map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with(".appdata.xml")).collect();
    found.sort();
    found.into_iter().next()
}

fn cargo_version(root: &Path) -> String {
    let text = std::fs::read_to_string(root.join("Cargo.toml")).unwrap_or_default();
    Regex::new(r#"(?m)^version\s*=\s*"([^"]+)""#).unwrap().captures(&text).map(|c| c[1].to_string()).unwrap_or_default()
}

/// Every release heading, newest first by numeric version.
pub fn releases(changelog: &str) -> Vec<(String, String)> {
    let heading = Regex::new(r"(?m)^## (\d+\.\d+\.\d+) — (\d{4}-\d{2}-\d{2})\s*$").unwrap();
    let mut out: Vec<(String, String)> = Vec::new();
    for c in heading.captures_iter(changelog) {
        if !out.iter().any(|(v, _)| v == &c[1]) {
            out.push((c[1].to_string(), c[2].to_string()));
        }
    }
    let key = |v: &str| v.split('.').map(|p| p.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    // Stable, like Python's sort with reverse=True: equal keys keep their order.
    out.sort_by_key(|a| std::cmp::Reverse(key(&a.0)));
    out
}

pub fn render(text: &str, releases: &[(String, String)]) -> String {
    let rows: String = releases.iter().map(|(v, d)| format!("    <release version=\"{v}\" date=\"{d}\"/>\n")).collect();
    let block = format!("  <releases>\n{rows}  </releases>");
    Regex::new(r"(?s)  <releases>.*?</releases>").unwrap().replace_all(text, NoExpand(&block)).into_owned()
}

pub fn main(args: &[String]) -> ExitCode {
    let root = crate::root();
    let changelog = crate::py::universal_newlines(&std::fs::read_to_string(root.join("CHANGELOG.md")).unwrap_or_default());
    let rel = releases(&changelog);
    let version = cargo_version(&root);
    if rel.first().is_none_or(|(v, _)| v != &version) {
        let newest = rel.first().map_or("None", |(v, _)| v.as_str());
        eprintln!("newest CHANGELOG.md release ({newest}) is not the Cargo.toml version {version}: add its `## {version} — YYYY-MM-DD` heading");
        return ExitCode::FAILURE;
    }
    let Some(path) = appdata(&root) else {
        eprintln!("no data/*.appdata.xml");
        return ExitCode::FAILURE;
    };
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    let updated = render(&current, &rel);
    if args.iter().any(|a| a == "--check") {
        if updated != current {
            eprintln!("appdata.xml <releases> is out of date; run cargo xtask sync-appdata-releases");
            return ExitCode::FAILURE;
        }
        println!("AppStream releases: OK");
        return ExitCode::SUCCESS;
    }
    if let Err(e) = std::fs::write(&path, updated) {
        eprintln!("{}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_are_newest_first_once_each() {
        let text = "## 1.10.0 — 2026-02-01\n\n## 1.9.0 — 2026-01-01\n## 1.10.0 — 2026-03-01\n## Unreleased\n## 2.0.0 — 2026-04-01  \n";
        let rel = releases(text);
        let versions: Vec<&str> = rel.iter().map(|(v, _)| v.as_str()).collect();
        assert_eq!(versions, ["2.0.0", "1.10.0", "1.9.0"]);
        assert_eq!(rel[1].1, "2026-02-01");
    }

    #[test]
    fn render_replaces_the_block() {
        let xml = "<c>\n  <releases>\n    <release version=\"0\" date=\"x\"/>\n  </releases>\n</c>\n";
        let rel = vec![("1.0.0".to_string(), "2026-01-01".to_string())];
        assert_eq!(render(xml, &rel), "<c>\n  <releases>\n    <release version=\"1.0.0\" date=\"2026-01-01\"/>\n  </releases>\n</c>\n");
    }
}
