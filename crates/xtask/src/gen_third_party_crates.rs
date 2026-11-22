//! Lists, and gathers the licence texts of, everything third-party linked into Neural Forge's binaries.
//!
//! The crates: every crate the workspace members (layer, native backend, supervisor, GUI, CLI, protocol) link,
//! walking normal (non-dev, non-build) dependencies only and leaving out proc-macro crates (they run in the
//! compiler and are not linked), nor anything reached only through one, nor what only this development tool
//! (xtask, never shipped) links. The C and C++ code compiled into the layer: the network from OpenDLSS-NR
//! (third_party/opendlss-nr), and volk and the Vulkan-Headers from tools/native (cargo xtask fetch-native-tools). A
//! crate that ships no licence text of its own gets the standard text of each licence its SPDX expression names,
//! from scripts/spdx/ (for MIT, the copyright line of another crate from the same repository when one has it); one
//! naming a licence scripts/spdx/ lacks fails: add the text there.
//!
//! build-appimage.sh ships the `--licences` output as usr/share/doc/neural-forge/THIRD_PARTY_LICENSES.txt.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use regex::Regex;
use serde_json::Value;

use crate::py::{splitlines, universal_newlines};

const USAGE: &str = "  cargo xtask gen-third-party-crates                  rewrite THIRD_PARTY_CRATES.md (the crate listing)
  cargo xtask gen-third-party-crates --check          exit 1, with a diff, if THIRD_PARTY_CRATES.md is out of date
  cargo xtask gen-third-party-crates --licences OUT   write OUT: every licence and notice file of every crate (from
                                                      its source in the cargo registry or the repository, offline)
                                                      and of the native code; fails if one has none
  cargo xtask gen-third-party-crates --self-test      check the dependency walk on a synthetic graph";

const TARGET: &str = "x86_64-unknown-linux-gnu";

/// Workspace members that ship in no binary: their own dependencies are not listed.
const NOT_SHIPPED: &[&str] = &["xtask"];

/// A non-crate work: (name, licence, base, files, what it is part of), the files under the repository ("repo") or
/// tools/native ("tools").
type Native = (&'static str, &'static str, &'static str, &'static [&'static str], Option<&'static str>);

/// The native code compiled into the layer, and other non-crate works the binaries embed.
const NATIVE: &[Native] = &[
    ("OpenDLSS-NR", "MIT", "repo", &["third_party/opendlss-nr/LICENSE", "third_party/opendlss-nr/NOTICE"], None),
    ("volk", "MIT", "tools", &["volk/LICENSE.md"], None),
    (
        "Vulkan-Headers",
        "Apache-2.0 OR MIT",
        "tools",
        &["Vulkan-Headers/LICENSE.md", "Vulkan-Headers/LICENSES/Apache-2.0.txt", "Vulkan-Headers/LICENSES/MIT.txt"],
        None,
    ),
];

/// Licence files a crate must ship besides its own: works it embeds under another licence. The GUI toolkit's assets
/// crate compiles the Lucide icons (ISC) into the GUI, and their licence is a file of its own.
const REQUIRED_FILES: &[(&str, &[&str])] = &[("gpui-kit-assets", &["LICENSE-LUCIDE"])];

/// A crate's licence and notice files, by name.
fn licence_file(name: &str) -> bool {
    Regex::new(r"(?i)^(licen[cs]e|copying|notice|copyright|unlicense)([-_.].*)?$").unwrap().is_match(name)
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The crate's repository or homepage.
fn source_url(pkg: &Value) -> &str {
    [s(pkg, "repository"), s(pkg, "homepage")].into_iter().find(|u| !u.is_empty()).unwrap_or("")
}

fn metadata(extra: &[&str]) -> Result<Value, String> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--locked", "--filter-platform", TARGET])
        .args(extra)
        .current_dir(crate::root())
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!("cargo metadata failed:\n{}", String::from_utf8_lossy(&out.stderr)));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata: {e}"))
}

fn proc_macro(pkg: &Value) -> bool {
    pkg["targets"].as_array().into_iter().flatten().any(|t| t["kind"].as_array().into_iter().flatten().any(|k| k == "proc-macro"))
}

/// The packages linked into the workspace's binaries: the shipped members and their normal dependencies,
/// proc-macro crates (and what only they reach) left out.
fn linked(meta: &Value) -> Vec<&Value> {
    let pkgs: HashMap<&str, &Value> = meta["packages"].as_array().into_iter().flatten().map(|p| (s(p, "id"), p)).collect();
    let nodes: HashMap<&str, &Value> = meta["resolve"]["nodes"].as_array().into_iter().flatten().map(|n| (s(n, "id"), n)).collect();
    let members: BTreeSet<&str> = meta["workspace_members"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let mut seen = BTreeSet::new();
    let mut stack: Vec<&str> = members.iter().copied().filter(|m| !pkgs.get(m).is_some_and(|p| NOT_SHIPPED.contains(&s(p, "name")))).collect();
    while let Some(i) = stack.pop() {
        if !seen.insert(i) {
            continue;
        }
        for dep in nodes[i]["deps"].as_array().into_iter().flatten() {
            let normal = dep["dep_kinds"].as_array().into_iter().flatten().any(|k| k["kind"].is_null());
            let id = s(dep, "pkg");
            if normal && !proc_macro(pkgs[id]) {
                stack.push(id);
            }
        }
    }
    seen.into_iter().filter(|i| !members.contains(i)).map(|i| pkgs[i]).collect()
}

fn licence_of(pkg: &Value) -> &str {
    Some(s(pkg, "license")).filter(|l| !l.is_empty()).unwrap_or("see crate")
}

fn listing(meta: &Value) -> (String, usize) {
    let rows: BTreeMap<(&str, &str), (&str, &str)> =
        linked(meta).into_iter().map(|p| ((s(p, "name"), s(p, "version")), (licence_of(p), source_url(p)))).collect();
    let mut lines: Vec<String> = [
        "# Third-party Rust crates",
        "",
        "Statically linked into the Neural Forge binaries (layer, GUI and CLI); proc-macro crates, which run",
        "only in the compiler, are not. Generated by `cargo xtask gen-third-party-crates` from `Cargo.lock`; do",
        "not edit. Every crate's licence and notice files ship in the AppImage, collected from the crates'",
        "sources at build time: `usr/share/doc/neural-forge/THIRD_PARTY_LICENSES.txt`.",
        "",
        "| Crate | Version | License | Source |",
        "|---|---|---|---|",
    ]
    .iter()
    .map(|l| l.to_string())
    .collect();
    for ((name, version), (lic, repo)) in &rows {
        lines.push(format!("| {name} | {version} | {lic} | {repo} |"));
    }
    (lines.join("\n") + "\n", rows.len())
}

/// The crate's licence and notice files: its `license-file` and those in its own directory.
fn licence_files(pkg: &Value) -> Vec<PathBuf> {
    let dir = Path::new(s(pkg, "manifest_path")).parent().unwrap_or(Path::new("")).to_path_buf();
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut out: Vec<PathBuf> = Vec::new();
    let extra = s(pkg, "license_file");
    if !extra.is_empty() && dir.join(extra).is_file() {
        out.extend(std::fs::canonicalize(dir.join(extra)));
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|e| e.flatten().map(|e| e.path()).filter(|p| p.is_file() && licence_file(&p.file_name().unwrap_or_default().to_string_lossy())).collect())
        .unwrap_or_default();
    found.sort();
    for f in found {
        if !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

/// The licence ids an SPDX expression names, in order, each once ("MIT/Apache-2.0" too); not the exception after
/// WITH.
fn spdx_ids(expression: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    let mut skip = false;
    for word in Regex::new(r"[\s()/]+").unwrap().split(expression) {
        if skip {
            skip = false;
        } else if word.to_uppercase() == "WITH" {
            skip = true;
        } else if !word.is_empty() && !["OR", "AND"].contains(&word.to_uppercase().as_str()) && !ids.iter().any(|i| i == word) {
            ids.push(word.to_string());
        }
    }
    ids
}

/// The copyright lines of the MIT texts among `texts`, in order, each once.
fn copyright_lines(texts: &[String]) -> Vec<String> {
    let copyright = Regex::new(r"(?i)^copyright\b.*\S").unwrap();
    let mut lines: Vec<String> = Vec::new();
    for text in texts.iter().filter(|t| t.contains("Permission is hereby granted, free of charge")) {
        for line in splitlines(text) {
            let line = line.trim();
            if copyright.is_match(line) && !line.to_lowercase().contains("permission notice") && !lines.iter().any(|l| l == line) {
                lines.push(line.to_string());
            }
        }
    }
    lines
}

/// `Path.read_text(errors="replace")`.
fn read_text(path: &Path) -> String {
    universal_newlines(&String::from_utf8_lossy(&std::fs::read(path).unwrap_or_default()))
}

fn spdx_dir() -> PathBuf {
    crate::root().join("scripts/spdx")
}

/// For a crate with no licence file of its own: (file name, text) of the standard text of each licence its SPDX
/// expression names, from scripts/spdx/, and the ids scripts/spdx/ lacks. The MIT text gets the copyright lines of
/// the other crates of its repository (`same_repository`), or a pointer to the repository.
fn standard_texts(pkg: &Value, same_repository: &[&Value]) -> (Vec<(String, String)>, Vec<String>) {
    let (mut texts, mut missing) = (Vec::new(), Vec::new());
    for id in spdx_ids(s(pkg, "license")) {
        let path = spdx_dir().join(format!("{id}.txt"));
        if !path.is_file() {
            missing.push(id);
            continue;
        }
        let mut text = read_text(&path).trim().to_string();
        if id == "MIT" {
            let mut found: Vec<String> = Vec::new();
            for other in same_repository {
                let texts: Vec<String> = licence_files(other).iter().map(|f| read_text(f)).collect();
                for line in copyright_lines(&texts) {
                    if !found.contains(&line) {
                        found.push(line);
                    }
                }
            }
            let place = if s(pkg, "repository").is_empty() { format!("https://crates.io/crates/{}", s(pkg, "name")) } else { s(pkg, "repository").to_string() };
            let notice = if found.is_empty() { format!("Copyright: see {place}") } else { found.join("\n") };
            text = text.replacen("MIT License\n", &format!("MIT License\n\n{notice}\n"), 1);
        }
        texts.push((format!("{id} (standard text; the crate ships none)"), text));
    }
    (texts, missing)
}

/// A component's licence section.
fn section(title: &str, files: &[(String, String)]) -> String {
    let rule = "=".repeat(78);
    let mut out = format!("{rule}\n{title}\n{rule}\n");
    for (name, text) in files {
        out.push_str(&format!("--- {name} ---\n{}\n\n", text.trim_end()));
    }
    out
}

fn named(files: &[PathBuf]) -> Vec<(String, String)> {
    files.iter().map(|f| (f.file_name().unwrap_or_default().to_string_lossy().into_owned(), read_text(f))).collect()
}

fn licences(meta: &Value, out_path: &str) -> Result<(), String> {
    let root = crate::root();
    let tools = std::env::var("NEURAL_FORGE_NATIVE_TOOLS").unwrap_or_default();
    let tools = if tools.trim().is_empty() { root.join("tools/native") } else { PathBuf::from(tools.trim()) };
    let mut parts = vec!["Third-party licences and notices for Neural Forge\n\n\
         Neural Forge itself is AGPL-3.0-or-later (LICENSE). Below: every Rust crate statically linked into its\n\
         binaries (THIRD_PARTY_CRATES.md lists them; the GUI toolkit's icons, Lucide, come with its gpui-kit-assets\n\
         crate) and the C/C++ code compiled into its Vulkan layer, each with its licence and notice files as its\n\
         source ships them (the standard text of its licence for a crate that ships none). Generated by\n\
         `cargo xtask gen-third-party-crates` at build time; ATTRIBUTION.md says what each project contributed.\n\n"
        .to_string()];
    let mut missing = Vec::new();
    for (name, lic, base, rel, into) in NATIVE {
        let files: Vec<PathBuf> = rel.iter().map(|r| if *base == "repo" { root.join(r) } else { tools.join(r) }).collect();
        let absent: Vec<String> = files.iter().filter(|f| !f.is_file()).map(|f| f.display().to_string()).collect();
        if !absent.is_empty() {
            let hint = if *base == "repo" { "" } else { " (run cargo xtask fetch-native-tools)" };
            missing.push(format!("{name}: {}{hint}", absent.join(", ")));
            continue;
        }
        parts.push(section(&format!("{name} ({lic}), {}", into.unwrap_or("compiled into the Vulkan layer")), &named(&files)));
    }
    let mut packages: Vec<&Value> = meta["packages"].as_array().into_iter().flatten().collect();
    packages.sort_by_key(|p| (s(p, "name"), s(p, "version")));
    let mut by_repository: HashMap<&str, Vec<&Value>> = HashMap::new();
    for p in &packages {
        if !s(p, "repository").is_empty() {
            by_repository.entry(s(p, "repository").trim_end_matches('/')).or_default().push(p);
        }
    }
    let mut shipped = linked(meta);
    shipped.sort_by_key(|p| (s(p, "name"), s(p, "version")));
    for pkg in shipped {
        let (name, version) = (s(pkg, "name"), s(pkg, "version"));
        let dir = Path::new(s(pkg, "manifest_path")).parent().unwrap_or(Path::new("")).display().to_string();
        let files = licence_files(pkg);
        let have: BTreeSet<String> = files.iter().map(|f| f.file_name().unwrap_or_default().to_string_lossy().into_owned()).collect();
        let required = REQUIRED_FILES.iter().find(|(n, _)| *n == name).map_or(&[][..], |(_, f)| f);
        let absent: Vec<&str> = required.iter().copied().filter(|n| !have.contains(*n)).collect();
        if !absent.is_empty() {
            missing.push(format!("{name} {version}: {} missing in {dir}", absent.join(", ")));
            continue;
        }
        let texts = if files.is_empty() {
            let siblings: Vec<&Value> =
                by_repository.get(s(pkg, "repository").trim_end_matches('/')).into_iter().flatten().copied().filter(|p| s(p, "id") != s(pkg, "id")).collect();
            let (texts, unknown) = standard_texts(pkg, &siblings);
            if !unknown.is_empty() || texts.is_empty() {
                let why = if unknown.is_empty() { "no licence declared".to_string() } else { format!("no text in scripts/spdx/ for {}", unknown.join(", ")) };
                missing.push(format!("{name} {version}: no licence file in {dir}, {why}"));
                continue;
            }
            texts
        } else {
            named(&files)
        };
        parts.push(section(format!("{name} {version} ({}) {}", licence_of(pkg), source_url(pkg)).trim_end(), &texts));
    }
    if !missing.is_empty() {
        return Err(format!("licence texts missing:\n  {}", missing.join("\n  ")));
    }
    std::fs::write(out_path, parts.concat()).map_err(|e| format!("{out_path}: {e}"))?;
    println!("wrote {out_path}: {} components", parts.len() - 1);
    Ok(())
}

/// `difflib.unified_diff(a, b, from, to)` written out.
fn unified_diff(old: &str, new: &str, from: &str, to: &str) -> String {
    similar::TextDiff::from_lines(old, new).unified_diff().context_radius(3).header(from, to).to_string()
}

fn self_test() {
    let pkg = |i: &str, kinds: &[&str]| serde_json::json!({"id": i, "name": i, "version": "1.0.0", "targets": [{"kind": kinds}], "source": "registry"});
    let normal = serde_json::json!([{"kind": null}]);
    let meta = serde_json::json!({
        "packages": [pkg("app", &["bin"]), pkg("lib", &["lib"]), pkg("macro", &["proc-macro"]), pkg("syn", &["lib"]), pkg("build", &["lib"]), pkg("dev", &["lib"]), pkg("shared", &["lib"]),
                     pkg("xtask", &["bin"]), pkg("devonly", &["lib"])],
        "workspace_members": ["app", "xtask"],
        "resolve": {"nodes": [
            {"id": "app", "deps": [
                {"pkg": "lib", "dep_kinds": normal},
                {"pkg": "macro", "dep_kinds": normal},
                {"pkg": "build", "dep_kinds": [{"kind": "build"}]},
                {"pkg": "dev", "dep_kinds": [{"kind": "dev"}]},
            ]},
            {"id": "lib", "deps": [{"pkg": "shared", "dep_kinds": normal}]},
            {"id": "macro", "deps": [{"pkg": "syn", "dep_kinds": normal}, {"pkg": "shared", "dep_kinds": normal}]},
            {"id": "xtask", "deps": [{"pkg": "devonly", "dep_kinds": normal}, {"pkg": "shared", "dep_kinds": normal}]},
            {"id": "syn", "deps": []}, {"id": "build", "deps": []}, {"id": "dev", "deps": []}, {"id": "shared", "deps": []}, {"id": "devonly", "deps": []},
        ]},
    });
    let names: Vec<&str> = linked(&meta).iter().map(|p| s(p, "name")).collect();
    assert_eq!(names, ["lib", "shared"]);
    for name in ["LICENSE", "LICENSE-MIT", "license.txt", "COPYING", "NOTICE", "UNLICENSE", "LICENCE-APACHE"] {
        assert!(licence_file(name), "{name}");
    }
    for name in ["README.md", "Cargo.toml", "licensing.rs", "AUTHORS", "AUTHORS.md"] {
        assert!(!licence_file(name), "{name}");
    }
    assert!(licence_file("LICENSE-LUCIDE"));
    assert_eq!(spdx_ids("MIT OR Apache-2.0"), ["MIT", "Apache-2.0"]);
    assert_eq!(spdx_ids("MIT/Apache-2.0"), ["MIT", "Apache-2.0"]);
    assert_eq!(spdx_ids("(Apache-2.0 WITH LLVM-exception OR MIT) AND Zlib"), ["Apache-2.0", "MIT", "Zlib"]);
    let mit = "MIT License\n\nCopyright (c) 2020 A. Author\n\nPermission is hereby granted, free of charge, to".to_string();
    assert_eq!(copyright_lines(&[mit, "Apache\nCopyright 2019 Nobody".into()]), ["Copyright (c) 2020 A. Author"]);
    let bare = serde_json::json!({"id": "bare", "name": "bare", "version": "1", "license": "MIT OR Apache-2.0", "repository": "https://example.org/r",
                                  "manifest_path": "/nonexistent/Cargo.toml"});
    let (texts, unknown) = standard_texts(&bare, &[]);
    assert!(unknown.is_empty() && texts.len() == 2, "{texts:?} {unknown:?}");
    assert!(texts[0].1.contains("Copyright: see https://example.org/r") && texts[0].1.contains("Permission is hereby"));
    assert!(texts[1].1.contains("Apache License"));
    let mut made_up = bare.clone();
    made_up["license"] = "MIT OR Made-Up-1.0".into();
    assert_eq!(standard_texts(&made_up, &[]).1, ["Made-Up-1.0"]);
    assert!(section("bare", &texts).contains("--- MIT (standard text"));
    assert_eq!(
        unified_diff("a\nb\nc\n", "a\nB\nc\n", "THIRD_PARTY_CRATES.md", "generated"),
        "--- THIRD_PARTY_CRATES.md\n+++ generated\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    );
    println!("gen_third_party_crates self-test ok");
}

fn run(args: &[String]) -> Result<(), String> {
    if args == ["--self-test"] {
        self_test();
        return Ok(());
    }
    if args.len() == 2 && args[0] == "--licences" {
        return licences(&metadata(&["--offline"])?, &args[1]);
    }
    let (text, count) = listing(&metadata(&[])?);
    let out = crate::root().join("THIRD_PARTY_CRATES.md");
    if args == ["--check"] {
        let current = std::fs::read_to_string(&out).map(|t| universal_newlines(&t)).unwrap_or_default();
        if current != text {
            print!("{}", unified_diff(&current, &text, "THIRD_PARTY_CRATES.md", "generated"));
            return Err("THIRD_PARTY_CRATES.md is out of date: run cargo xtask gen-third-party-crates".into());
        }
        return Ok(());
    }
    if !args.is_empty() {
        return Err(USAGE.into());
    }
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("wrote THIRD_PARTY_CRATES.md: {count} crates");
    Ok(())
}

pub fn main(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_test() {
        super::self_test();
    }
}
