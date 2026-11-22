//! Print one version's CHANGELOG.md section: the release page's text.
//!
//! usage: cargo xtask release-notes <version> [CHANGELOG.md]
//!
//! The section is everything under `## <version> — <date>` up to the next `## ` heading, as written
//! (the heading itself left out: the release is titled with the version). Exits 1 when the version has
//! no section or it is empty, so a release never goes out without its notes.

use std::process::ExitCode;

use regex::Regex;

use crate::py::splitlines;

const USAGE: &str = "Print one version's CHANGELOG.md section: the release page's text.

usage: cargo xtask release-notes <version> [CHANGELOG.md]

The section is everything under `## <version> — <date>` up to the next `## ` heading, as written
(the heading itself left out: the release is titled with the version). Exits 1 when the version has
no section or it is empty, so a release never goes out without its notes.
";

pub fn notes(changelog: &str, version: &str) -> String {
    let head = Regex::new(&format!(r"^## {} — \d{{4}}-\d{{2}}-\d{{2}}\s*$", regex::escape(version))).unwrap();
    let lines = splitlines(changelog);
    let Some(start) = lines.iter().position(|line| head.is_match(line)) else { return String::new() };
    let body: Vec<&str> = lines[start + 1..].iter().take_while(|line| !line.starts_with("## ")).copied().collect();
    unwrap(body.join("\n").trim())
}

/// Joins the changelog's hard-wrapped lines: GitHub shows every newline of a release's text as a
/// line break. A list item's continuation lines (indented), a paragraph's lines and a quote's lines
/// become one line; blank lines, headings (nothing is joined onto one), list items, tables and code
/// fences stay as they are.
pub fn unwrap(text: &str) -> String {
    let numbered = Regex::new(r"^\d+\. ").unwrap();
    let mut out: Vec<String> = Vec::new();
    let mut fenced = false;
    for line in splitlines(text) {
        let stripped = line.trim();
        if stripped.starts_with("```") {
            fenced = !fenced;
            out.push(line.to_string());
            continue;
        }
        let previous = out.last().map(|l| l.trim().to_string()).unwrap_or_default();
        let joinable = !previous.is_empty() && !fenced && !["```", "#", "|"].iter().any(|p| previous.starts_with(p));
        if joinable && previous.starts_with('>') && stripped.starts_with('>') {
            // The next line of the same quote.
            let last = out.last_mut().unwrap();
            *last = format!("{} {}", last.trim_end(), stripped[1..].trim());
            continue;
        }
        let starts_block =
            stripped.is_empty() || ["- ", "* ", "+ ", "> ", "#", "|"].iter().any(|p| stripped.starts_with(p)) || stripped == ">" || numbered.is_match(stripped);
        if joinable && !stripped.is_empty() && !starts_block {
            let last = out.last_mut().unwrap();
            *last = format!("{} {}", last.trim_end(), stripped);
        } else {
            out.push(line.to_string());
        }
    }
    out.join("\n")
}

fn self_test() {
    let text = "# T\n\n## Unreleased\n\n- x\n\n## 1.2.3 — 2026-10-05\n\nIntro.\n\n- **A.** one\n  two\n\n## 1.2.2 — 2026-10-01\n\n- old\n";
    assert_eq!(notes(text, "1.2.3"), "Intro.\n\n- **A.** one two");
    assert_eq!(unwrap("a\nb\n\n- x\n  y\n- z\n```\nk\nl\n```"), "a b\n\n- x y\n- z\n```\nk\nl\n```");
    // Nothing is joined onto a heading or a table row; quotes and `+` items are blocks of their own.
    assert_eq!(unwrap("### Fixed\nThe layer\nno longer"), "### Fixed\nThe layer no longer");
    assert_eq!(unwrap("| a |\nb"), "| a |\nb");
    assert_eq!(unwrap("text\n> quoted\n> more\nlazy"), "text\n> quoted more lazy");
    assert_eq!(unwrap("text\n+ item\n  on\n+ next"), "text\n+ item on\n+ next");
    assert_eq!(notes(text, "1.2.2"), "- old");
    assert_eq!(notes(text, "1.2"), "");
    println!("release_notes self-test ok");
}

pub fn main(args: &[String]) -> ExitCode {
    if args == ["--self-test"] {
        self_test();
        return ExitCode::SUCCESS;
    }
    if !(1..=2).contains(&args.len()) {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let path = args.get(1).map_or("CHANGELOG.md", String::as_str);
    let changelog = match std::fs::read_to_string(path) {
        Ok(text) => crate::py::universal_newlines(&text),
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let text = notes(&changelog, args[0].trim_start_matches('v'));
    if text.is_empty() {
        eprintln!("no CHANGELOG.md section for {}", args[0]);
        return ExitCode::FAILURE;
    }
    println!("{text}");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_test() {
        super::self_test();
    }
}
