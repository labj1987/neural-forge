//! Reports, and updates, the pinned versions Dependabot cannot see.
//!
//! The pins: the native toolchain in crates/xtask/src/fetch_native_tools.rs, cargo-deny in crates/xtask/src/cargo_deny.rs,
//! appimagetool and the AppImage runtime in build-appimage.sh, and the Rust toolchain in
//! rust-toolchain.toml. (GitHub Actions are pinned by SHA and updated by Dependabot; Cargo crates by
//! Dependabot.) Each is compared with upstream's newest release or tag through the GitHub API
//! ($GITHUB_TOKEN raises its rate limit). Vulkan-Headers and volk follow the Vulkan SDK releases
//! (vulkan-sdk-* tags), not Khronos's weekly spec-update tags: both are pinned to the same SDK, the newest
//! one both repositories have tagged, so the headers and the loader always come from one SDK.
//!
//! ```text
//! cargo xtask check-pins               print a Markdown report; exit 0 when every pin is current, 10 when one
//!                                      is behind (or Vulkan-Headers and volk name different SDKs), 1 when
//!                                      upstream could not be asked or named no version to compare with (a pin
//!                                      that cannot be checked is never current)
//! cargo xtask check-pins --bump        the same check, then move every pin that is behind to upstream's version:
//!                                      the new archive is downloaded, its SHA-256 computed (and compared with
//!                                      upstream's published .sha256 where there is one), and every place the pin
//!                                      is named is rewritten. One line per bump on stdout. Exit 0 when something
//!                                      was bumped, 3 when nothing was behind, 1 on any failure (nothing written)
//! cargo xtask check-pins --self-test   check the parsing, comparison and rewriting offline
//! ```
//!
//! .github/workflows/pin-check.yml runs it weekly: it bumps what is behind, runs the full build and check
//! on the bump, and lands it on main when that passes; one issue carries the report.

pub(crate) mod http;

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

const BEHIND: u8 = 10;
const NOTHING_TO_BUMP: u8 = 3;
/// The check itself failed (curl or a local fault, not the network): pin-check.yml fails the run on it instead of
/// retrying as for an unreachable upstream (exit 1).
const BROKEN: u8 = 4;

/// Every file a pin is named in, by the short name the rewrites use.
const FILES: [(&str, &str); 5] = [
    ("fetch", "crates/xtask/src/fetch_native_tools.rs"),
    ("build", "build-appimage.sh"),
    ("toolchain", "rust-toolchain.toml"),
    ("deny", "crates/xtask/src/cargo_deny.rs"),
    ("attribution", "ATTRIBUTION.md"),
];
const SDK: &str = "Vulkan SDK (Vulkan-Headers = volk)";
const USER_AGENT: &str = "neural-forge-pin-check";

type Res<T> = Result<T, String>;

/// Why asking upstream failed: the network or its answer (reported, the pin is left unchecked), or curl itself (the
/// check is broken: it stops the run).
#[derive(Debug)]
enum Up {
    Net(String),
    Tool(String),
}

impl From<String> for Up {
    fn from(m: String) -> Self {
        Up::Net(m)
    }
}

impl From<&str> for Up {
    fn from(m: &str) -> Self {
        Up::Net(m.to_string())
    }
}

type UpRes<T> = Result<T, Up>;
type Texts = BTreeMap<&'static str, String>;
type Sha<'a> = &'a dyn Fn(&str) -> Res<String>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Release,
    Sdk,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Release => "release",
            Kind::Sdk => "sdk",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Behind,
    Current,
    Unknown,
    Mismatch,
}

#[derive(Clone, Debug)]
struct Pin {
    name: String,
    pinned: String,
    kind: Kind,
    repo: String,
}

#[derive(Clone, Debug)]
struct Row {
    name: String,
    pinned: String,
    upstream: Option<String>,
    state: State,
}

fn row(name: &str, pinned: &str, upstream: Option<&str>, state: State) -> Row {
    Row { name: name.into(), pinned: pinned.into(), upstream: upstream.map(String::from), state }
}

fn rel_path(key: &str) -> &'static str {
    FILES.iter().find(|(k, _)| *k == key).map(|(_, rel)| *rel).unwrap_or("")
}

/// Python's re.escape(): the characters that can mean something in a pattern get a backslash.
fn py_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A file's text as Python's `Path.read_text()` gives it: UTF-8 with universal newlines.
fn read(rel: &str) -> Res<String> {
    let text = std::fs::read_to_string(crate::root().join(rel)).map_err(|e| format!("{rel}: {e}"))?;
    Ok(crate::py::universal_newlines(&text))
}

fn read_all() -> Res<Texts> {
    FILES.iter().map(|(key, rel)| Ok((*key, read(rel)?))).collect()
}

/// The first group of the first match of `pattern` in `text`.
fn one(pattern: &str, text: &str, what: &str, multiline: bool) -> Res<String> {
    let full = if multiline { format!("(?m){pattern}") } else { pattern.to_string() };
    let re = regex::Regex::new(&full).map_err(|e| format!("{what}: bad pattern {pattern}: {e}"))?;
    match re.captures(text) {
        Some(c) => Ok(c.get(1).map(|m| m.as_str()).unwrap_or("").to_string()),
        None => Err(format!("{what}: pattern {} not found", crate::py::repr_str(pattern))),
    }
}

fn pins(texts: &Texts) -> Res<Vec<Pin>> {
    let (fetch, build, toolchain, deny) = (&texts["fetch"], &texts["build"], &texts["toolchain"], &texts["deny"]);
    let sdk_repos = "KhronosGroup/Vulkan-Headers zeux/volk";
    let mk = |name: &str, pinned: String, kind: Kind, repo: &str| Pin { name: name.into(), pinned, kind, repo: repo.into() };
    Ok(vec![
        mk("glslang", one(r"glslang/releases/download/([^/]+)/", fetch, "glslang", false)?, Kind::Release, "KhronosGroup/glslang"),
        mk("Vulkan-Headers", one(r"Vulkan-Headers/archive/refs/tags/(vulkan-sdk-[0-9.]+)\.tar", fetch, "Vulkan-Headers", false)?, Kind::Sdk, sdk_repos),
        mk("volk", one(r"volk/archive/refs/tags/(vulkan-sdk-[0-9.]+)\.tar", fetch, "volk", false)?, Kind::Sdk, sdk_repos),
        mk("cargo-deny", one(r#"^const CARGO_DENY_VERSION: &str = "([^"]+)""#, deny, "cargo-deny", true)?, Kind::Release, "EmbarkStudios/cargo-deny"),
        mk("appimagetool", one(r#"^APPIMAGETOOL_VERSION="([^"]+)""#, build, "appimagetool", true)?, Kind::Release, "AppImage/appimagetool"),
        mk("AppImage type2-runtime", one(r#"^RUNTIME_VERSION="([^"]+)""#, build, "type2-runtime", true)?, Kind::Release, "AppImage/type2-runtime"),
        mk("Rust toolchain", one(r#"^channel = "([^"]+)""#, toolchain, "Rust toolchain", true)?, Kind::Release, "rust-lang/rust"),
    ])
}

/// A version as numbers for comparing (`v1.4.363`, `vulkan-sdk-1.4.357.0`, `16.6.0`, `20251108`).
fn version_key(v: &str) -> Vec<u128> {
    let re = regex::Regex::new(r"\d+").expect("static pattern");
    re.find_iter(v).map(|m| m.as_str().parse().unwrap_or(u128::MAX)).collect()
}

/// The highest of `versions` that is a plain release (no -rc, beta, continuous); the first of equals, like Python's max().
fn newest(versions: &[String]) -> Option<String> {
    let plain = regex::Regex::new(r"\A(?:[A-Za-z-]*v?[0-9][0-9.]*)\z").expect("static pattern");
    let mut best: Option<(&String, Vec<u128>)> = None;
    for v in versions.iter().filter(|v| plain.is_match(v)) {
        let key = version_key(v);
        if best.as_ref().is_none_or(|(_, k)| key > *k) {
            best = Some((v, key));
        }
    }
    best.map(|(v, _)| v.clone())
}

fn github_headers() -> Vec<(String, String)> {
    let mut headers = vec![("Accept".to_string(), "application/vnd.github+json".to_string()), ("User-Agent".to_string(), USER_AGENT.to_string())];
    let token = std::env::var("GITHUB_TOKEN").unwrap_or_default();
    let token = token.trim();
    if !token.is_empty() {
        headers.push(("Authorization".to_string(), format!("Bearer {token}")));
    }
    headers
}

fn api(path: &str) -> UpRes<serde_json::Value> {
    let headers = github_headers();
    let refs: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let body = http::get(&format!("https://api.github.com/{path}"), &refs, 30).map_err(up)?;
    serde_json::from_slice(&body).map_err(|e| Up::Net(e.to_string()))
}

/// A fetch error as an `Up`: only curl's own failure (not the network's) is a tool error.
fn up(e: http::FetchError) -> Up {
    match e {
        http::FetchError::Tool(m) => Up::Tool(m),
        e => Up::Net(e.to_string()),
    }
}

fn json_list(v: serde_json::Value, what: &str) -> Res<Vec<serde_json::Value>> {
    match v {
        serde_json::Value::Array(a) => Ok(a),
        other => Err(format!("{what}: upstream answered with a shape this check does not know: {}", other.to_string().chars().take(200).collect::<String>())),
    }
}

fn field<'a>(v: &'a serde_json::Value, key: &str) -> Res<&'a serde_json::Value> {
    v.get(key).ok_or_else(|| format!("'{key}'"))
}

/// Python truthiness of a JSON value.
fn truthy(v: &serde_json::Value) -> bool {
    use serde_json::Value::*;
    match v {
        Null => false,
        Bool(b) => *b,
        Number(n) => n.as_f64() != Some(0.0),
        String(s) => !s.is_empty(),
        Array(a) => !a.is_empty(),
        Object(o) => !o.is_empty(),
    }
}

/// Every tag of `repo` starting with `prefix` (matching-refs lists them all, unlike /tags' first page).
fn tags(repo: &str, prefix: &str) -> UpRes<Vec<String>> {
    let refs = json_list(api(&format!("repos/{repo}/git/matching-refs/tags/{prefix}"))?, repo)?;
    refs.iter()
        .map(|r| {
            let name = field(r, "ref")?.as_str().ok_or("tag ref is not a string")?;
            Ok(name.strip_prefix("refs/tags/").unwrap_or(name).to_string())
        })
        .collect()
}

/// The newest vulkan-sdk-* tag every repository has, so headers and loader come from one SDK.
fn newest_common_sdk(per_repo: &[Vec<String>]) -> Option<String> {
    let mut sets = per_repo.iter().map(|t| t.iter().cloned().collect::<BTreeSet<String>>());
    let first = sets.next()?;
    let common: Vec<String> = sets.fold(first, |acc, s| acc.intersection(&s).cloned().collect()).into_iter().collect();
    newest(&common)
}

fn latest(kind: Kind, repo: &str, cache: &mut Option<Option<String>>) -> UpRes<Option<String>> {
    match kind {
        Kind::Release => {
            let releases = json_list(api(&format!("repos/{repo}/releases?per_page=100"))?, repo)?;
            let mut names = Vec::new();
            for r in &releases {
                if !truthy(field(r, "prerelease")?) && !truthy(field(r, "draft")?) {
                    names.push(field(r, "tag_name")?.as_str().ok_or("tag_name is not a string")?.to_string());
                }
            }
            Ok(newest(&names))
        }
        Kind::Sdk => {
            if cache.is_none() {
                let per_repo = repo.split_whitespace().map(|r| tags(r, "vulkan-sdk-")).collect::<UpRes<Vec<_>>>()?;
                *cache = Some(newest_common_sdk(&per_repo));
            }
            Ok(cache.clone().flatten())
        }
    }
}

/// behind, current, or unknown when upstream gave nothing to compare with.
fn state(pinned: &str, upstream: Option<&str>) -> State {
    match upstream {
        None => State::Unknown,
        Some(u) if version_key(u) > version_key(pinned) => State::Behind,
        Some(_) => State::Current,
    }
}

/// The row that keeps Vulkan-Headers and volk on one SDK: current when both name the same tag.
fn sdk_row(rows: &[Row]) -> Row {
    let pinned = |name: &str| rows.iter().find(|r| r.name == name).map(|r| r.pinned.clone()).unwrap_or_default();
    let (headers, volk) = (pinned("Vulkan-Headers"), pinned("volk"));
    let st = if headers == volk { State::Current } else { State::Mismatch };
    Row { name: SDK.into(), pinned: format!("{headers} = {volk}"), upstream: None, state: st }
}

fn report(rows: &[Row]) -> String {
    let mut lines: Vec<String> = [
        "Pinned versions that Dependabot cannot see, compared with upstream by the weekly pin check",
        "(`cargo xtask check-pins`), which also bumps the ones that are behind and lands the bump on main",
        "once the full build and check passes on it.",
        "",
        "| Pin | Pinned | Upstream | |",
        "|---|---|---|---|",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for r in rows {
        let shown = match r.state {
            State::Behind => "**behind**",
            State::Current => "current",
            State::Unknown => "**unknown**",
            State::Mismatch => "**mismatch**",
        };
        let upstream = if r.name == SDK { "" } else { r.upstream.as_deref().unwrap_or("?") };
        lines.push(format!("| {} | {} | {} | {} |", r.name, r.pinned, upstream, shown));
    }
    lines.join("\n") + "\n"
}

/// 1 when any pin could not be checked (never reported as current), else BEHIND or 0.
fn exit_status(rows: &[Row], failed: &[String]) -> u8 {
    if !failed.is_empty() || rows.iter().any(|r| r.state == State::Unknown) {
        return 1;
    }
    if rows.iter().any(|r| matches!(r.state, State::Behind | State::Mismatch)) {
        BEHIND
    } else {
        0
    }
}

// ---- Bumping -----------------------------------------------------------------------------------

fn sha256_url(url: &str) -> Res<String> {
    let mut hash = Sha256::new();
    http::get_stream(url, &[("User-Agent", USER_AGENT)], 300, &mut |chunk| hash.update(chunk)).map_err(|e| e.to_string())?;
    Ok(hex(&hash.finalize()))
}

fn text_url(url: &str) -> Res<String> {
    let body = http::get(url, &[("User-Agent", USER_AGENT)], 60).map_err(|e| e.to_string())?;
    String::from_utf8(body).map_err(|e| e.to_string())
}

/// Replace `old` by `new` in texts[key], which must hold it exactly `count` times.
fn replace(texts: &mut Texts, key: &'static str, old: &str, new: &str, count: usize) -> Res<()> {
    let found = texts[key].matches(old).count();
    if found != count {
        return Err(format!("{}: expected {count} of {}, found {found}", rel_path(key), crate::py::repr_str(old)));
    }
    let replaced = texts[key].replace(old, new);
    texts.insert(key, replaced);
    Ok(())
}

fn glslang_url(v: &str) -> String {
    format!("https://github.com/KhronosGroup/glslang/releases/download/{v}/glslang-{v}-linux-x86_64-release.tar.gz")
}

fn tag_url(repo: &str, v: &str) -> String {
    format!("https://github.com/{repo}/archive/refs/tags/{v}.tar.gz")
}

fn deny_url(v: &str) -> String {
    format!("https://github.com/EmbarkStudios/cargo-deny/releases/download/{v}/cargo-deny-{v}-x86_64-unknown-linux-musl.tar.gz")
}

/// The `name|url|sha256|check` entry in fetch_native_tools.rs's PINS.
fn fetch_line(texts: &Texts, name: &str) -> Res<Vec<String>> {
    let pattern = format!(r#"^\s*"({}\|[^"]+)""#, py_escape(name));
    let parts: Vec<String> = one(&pattern, &texts["fetch"], name, true)?.split('|').map(String::from).collect();
    match parts.len() {
        4 => Ok(parts),
        n if n < 4 => Err(format!("not enough values to unpack (expected 4, got {n})")),
        _ => Err("too many values to unpack (expected 4)".to_string()),
    }
}

fn bump_fetch_pin(texts: &mut Texts, name: &str, new_url: &str, sha: &str) -> Res<()> {
    let line = fetch_line(texts, name)?;
    replace(texts, "fetch", &format!("{}|{}|", line[1], line[2]), &format!("{new_url}|{sha}|"), 1)
}

/// Rewrite every place pin `name` is named from `old` to `new` in `texts` (in place).
fn bump(texts: &mut Texts, name: &str, old: &str, new: &str, sha_of: Sha, sha_file: Sha) -> Res<()> {
    match name {
        "glslang" => {
            let sha = sha_of(&glslang_url(new))?;
            bump_fetch_pin(texts, name, &glslang_url(new), &sha)?;
            replace(texts, "fetch", &format!("glslang {old} ("), &format!("glslang {new} ("), 1)
        }
        "Vulkan-Headers" | "volk" => {
            let repo = if name == "volk" { "zeux/volk" } else { "KhronosGroup/Vulkan-Headers" };
            let sha = sha_of(&tag_url(repo, new))?;
            bump_fetch_pin(texts, name, &tag_url(repo, new), &sha)?;
            replace(texts, "fetch", &format!("{name} {old}"), &format!("{name} {new}"), 1)?;
            replace(texts, "attribution", &format!("/{repo}) | {old} |"), &format!("/{repo}) | {new} |"), 1)
        }
        "cargo-deny" => {
            let sha = sha_of(&deny_url(new))?;
            let published_file = sha_file(&format!("{}.sha256", deny_url(new)))?;
            let published = published_file.split_whitespace().next().ok_or("list index out of range")?;
            if published != sha {
                return Err(format!("cargo-deny {new}: downloaded SHA-256 {sha} is not the published {published}"));
            }
            let old_sha = one(r#"^const SHA: &str = "([0-9a-f]{64})""#, &texts["deny"], "cargo-deny SHA", true)?;
            replace(texts, "deny", &format!("const CARGO_DENY_VERSION: &str = \"{old}\""), &format!("const CARGO_DENY_VERSION: &str = \"{new}\""), 1)?;
            replace(texts, "deny", &format!("const SHA: &str = \"{old_sha}\""), &format!("const SHA: &str = \"{sha}\""), 1)
        }
        "appimagetool" | "AppImage type2-runtime" => {
            let (var, repo, asset) = if name == "appimagetool" {
                ("APPIMAGETOOL", "AppImage/appimagetool", "appimagetool-x86_64.AppImage")
            } else {
                ("RUNTIME", "AppImage/type2-runtime", "runtime-x86_64")
            };
            let sha = sha_of(&format!("https://github.com/{repo}/releases/download/{new}/{asset}"))?;
            let old_sha = one(&format!(r#"^{var}_SHA256="([0-9a-f]{{64}})""#), &texts["build"], &format!("{name} SHA"), true)?;
            replace(texts, "build", &format!("{var}_VERSION=\"{old}\""), &format!("{var}_VERSION=\"{new}\""), 1)?;
            replace(texts, "build", &format!("{var}_SHA256=\"{old_sha}\""), &format!("{var}_SHA256=\"{sha}\""), 1)
        }
        "Rust toolchain" => replace(texts, "toolchain", &format!("channel = \"{old}\""), &format!("channel = \"{new}\""), 1),
        _ => Err(format!("no bump rule for {name}")),
    }
}

/// Bump every behind (or SDK-mismatched) pin in `texts`; returns one summary line per bump.
/// The SDK pair always moves together, to the newest SDK both repositories have.
fn bump_all(texts: &mut Texts, rows: &[Row], sha_of: Sha, sha_file: Sha) -> Res<Vec<String>> {
    let mut lines = Vec::new();
    let mismatch = rows.iter().any(|r| r.name == SDK && r.state == State::Mismatch);
    let is_sdk = |name: &str| matches!(name, "Vulkan-Headers" | "volk");
    for r in rows.iter().filter(|r| r.name != SDK) {
        let moves_with_sdk = is_sdk(&r.name) && mismatch && r.upstream.as_ref().is_some_and(|u| *u != r.pinned);
        if r.state == State::Behind || moves_with_sdk {
            let upstream = r.upstream.clone().unwrap_or_else(|| "None".into());
            bump(texts, &r.name, &r.pinned, &upstream, sha_of, sha_file)?;
            lines.push(format!("{}: {} -> {}", r.name, r.pinned, upstream));
        }
    }
    let after: BTreeMap<String, String> = pins(texts)?.into_iter().map(|p| (p.name, p.pinned)).collect();
    for r in rows.iter().filter(|r| r.name != SDK) {
        if (r.state == State::Behind || (is_sdk(&r.name) && mismatch)) && r.upstream.as_deref() != after.get(&r.name).map(String::as_str) {
            let now = after.get(&r.name).cloned().unwrap_or_default();
            return Err(format!("{}: still reads {now} after the bump to {}", r.name, r.upstream.as_deref().unwrap_or("None")));
        }
    }
    if after["Vulkan-Headers"] != after["volk"] {
        return Err(format!("Vulkan-Headers {} and volk {} would name different SDKs", after["Vulkan-Headers"], after["volk"]));
    }
    Ok(lines)
}

/// The rows with the SDK row appended, and the failures.
fn check() -> Res<(Vec<Row>, Vec<String>)> {
    let (mut rows, mut failed, mut cache) = (Vec::new(), Vec::new(), None);
    for p in pins(&read_all()?)? {
        // Network or rate limit trouble is reported, not guessed.
        let upstream = match latest(p.kind, &p.repo, &mut cache) {
            Ok(None) => {
                failed.push(format!("{}: upstream {} names no plain {} to compare with", p.name, p.repo, p.kind.name()));
                None
            }
            Ok(u) => u,
            Err(Up::Net(e)) => {
                failed.push(format!("{}: {e}", p.name));
                None
            }
            // curl itself failed (not the network): the check is broken, so the run fails instead of reporting "could not check".
            Err(Up::Tool(e)) => return Err(format!("{}: {e}", p.name)),
        };
        let st = state(&p.pinned, upstream.as_deref());
        rows.push(Row { name: p.name, pinned: p.pinned, upstream, state: st });
    }
    let sdk = sdk_row(&rows);
    rows.push(sdk);
    Ok((rows, failed))
}

pub fn main(args: &[String]) -> ExitCode {
    if args == ["--self-test"] {
        return match self_test() {
            Ok(()) => {
                println!("check_pins self-test ok");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("check_pins self-test failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let bumping = args == ["--bump"];
    if !args.is_empty() && !bumping {
        eprintln!("usage: cargo xtask check-pins [--bump | --self-test]");
        return ExitCode::from(BROKEN);
    }
    let (rows, failed) = match check() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(BROKEN);
        }
    };
    let status = exit_status(&rows, &failed);
    if !bumping {
        print!("{}", report(&rows));
        if !failed.is_empty() {
            println!("\nCould not check upstream: {}", failed.join("; "));
        }
        return ExitCode::from(status);
    }
    if status == 1 {
        let why = if failed.is_empty() { "a pin has no upstream version".to_string() } else { failed.join("; ") };
        eprintln!("cannot bump: upstream could not be checked: {why}");
        return ExitCode::FAILURE;
    }
    if status == 0 {
        println!("every pin is current; nothing to bump");
        return ExitCode::from(NOTHING_TO_BUMP);
    }
    match bump_and_write(&rows) {
        Ok(lines) => {
            println!("{}", lines.join("\n"));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bump failed, nothing written: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Nothing is written unless every bump succeeded.
fn bump_and_write(rows: &[Row]) -> Res<Vec<String>> {
    let mut texts = read_all()?;
    let lines = bump_all(&mut texts, rows, &sha256_url, &text_url)?;
    for (key, rel) in FILES {
        if texts[key] != read(rel)? {
            std::fs::write(crate::root().join(rel), &texts[key]).map_err(|e| format!("{rel}: {e}"))?;
        }
    }
    Ok(lines)
}

// ---- Self-test ---------------------------------------------------------------------------------

macro_rules! ensure {
    ($cond:expr) => {
        if !$cond {
            return Err(format!("assertion failed: {}", stringify!($cond)));
        }
    };
}

fn fixture() -> Texts {
    let hex = |c: &str| c.repeat(64);
    let mut t = Texts::new();
    t.insert(
        "fetch",
        format!(
            "//!   glslang 16.5.0 (Linux x86_64 release binary), Vulkan-Headers vulkan-sdk-1.4.357.0 and volk vulkan-sdk-1.4.357.0.\nconst PINS: &[&str] = &[\n  \
             \"glslang|https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz|{}|bin/glslang\",\n  \
             \"Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz|{}|include/vulkan/vulkan_core.h\",\n  \
             \"volk|https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz|{}|volk.c\",\n];\n",
            hex("a"),
            hex("b"),
            hex("c")
        ),
    );
    t.insert(
        "build",
        format!("APPIMAGETOOL_VERSION=\"1.9.0\"\nAPPIMAGETOOL_SHA256=\"{}\"\nRUNTIME_VERSION=\"20251108\"\nRUNTIME_SHA256=\"{}\"\n", hex("d"), hex("e")),
    );
    t.insert("toolchain", "[toolchain]\nchannel = \"1.98.0\"\n".to_string());
    t.insert("deny", format!("const CARGO_DENY_VERSION: &str = \"0.19.9\";\nconst SHA: &str = \"{}\";\n", hex("f")));
    t.insert(
        "attribution",
        "| [volk](https://github.com/zeux/volk) | vulkan-sdk-1.4.357.0 | MIT | x |\n| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | vulkan-sdk-1.4.357.0 | Apache-2.0 / MIT | x |\n"
            .to_string(),
    );
    t
}

fn fake_sha(url: &str) -> String {
    hex(&Sha256::digest(url.as_bytes()))
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn self_test() -> Res<()> {
    let found = pins(&read_all()?)?;
    ensure!(
        found.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
            == ["glslang", "Vulkan-Headers", "volk", "cargo-deny", "appimagetool", "AppImage type2-runtime", "Rust toolchain"]
    );
    ensure!(found.iter().all(|p| !p.pinned.is_empty()));
    if found[1].pinned != found[2].pinned {
        return Err("the repository's Vulkan-Headers and volk name different SDKs".into());
    }
    ensure!(version_key("vulkan-sdk-1.4.357.0") < version_key("vulkan-sdk-1.4.360.0"));
    ensure!(newest(&strings(&["v1.4.363", "v1.4.370", "v1.4.99", "v1.5.0-rc1", "continuous"])).as_deref() == Some("v1.4.370"));
    ensure!(newest(&strings(&["continuous"])).is_none());
    let sdk = |a: &[&str]| strings(a);
    ensure!(
        newest_common_sdk(&[
            sdk(&["vulkan-sdk-1.4.357.0", "vulkan-sdk-1.4.363.0", "vulkan-sdk-1.4.370.0"]),
            sdk(&["vulkan-sdk-1.4.357.0", "vulkan-sdk-1.4.363.0"])
        ])
        .as_deref()
            == Some("vulkan-sdk-1.4.363.0")
    );
    ensure!(newest_common_sdk(&[sdk(&["vulkan-sdk-1.4.370.0"]), sdk(&["vulkan-sdk-1.4.363.0"])]).is_none());
    // 1.100 is newer than 1.99, not older.
    ensure!(state("1.99.1", Some("1.100.0")) == State::Behind && state("1.100.0", Some("1.99.1")) == State::Current);
    ensure!(state("1.96.0", Some("1.97.0")) == State::Behind && state("1.96.0", Some("1.96.0")) == State::Current);
    // Upstream that names no version (only `continuous` or -rc tags) is not a current pin.
    ensure!(state("vulkan-sdk-1.4.363.0", None) == State::Unknown && state("1.96.0", None) == State::Unknown);
    let text = report(&[
        row("x", "1.0", Some("1.1"), State::Behind),
        row("y", "3.0", Some("3.0"), State::Current),
        row("z", "2.0", None, State::Unknown),
        row(SDK, "a = b", None, State::Mismatch),
    ]);
    ensure!(text.contains("| x | 1.0 | 1.1 | **behind** |") && text.contains("| y | 3.0 | 3.0 | current |"));
    ensure!(text.contains("| z | 2.0 | ? | **unknown** |") && text.contains(&format!("| {SDK} | a = b |  | **mismatch** |")));
    ensure!(exit_status(&[row("x", "1.0", Some("1.1"), State::Behind)], &[]) == BEHIND);
    ensure!(exit_status(&[row("y", "1.0", Some("1.0"), State::Current)], &[]) == 0);
    ensure!(exit_status(&[row("y", "1.0", Some("1.0"), State::Current), row(SDK, "a = b", None, State::Mismatch)], &[]) == BEHIND);
    ensure!(exit_status(&[row("x", "1.0", Some("1.1"), State::Behind), row("z", "2.0", None, State::Unknown)], &strings(&["z: no version"])) == 1);
    let sdk_rows = [row("Vulkan-Headers", "vulkan-sdk-1.4.363.0", None, State::Current), row("volk", "vulkan-sdk-1.4.357.0", None, State::Current)];
    ensure!(sdk_row(&sdk_rows).state == State::Mismatch);
    ensure!(sdk_row(&[sdk_rows[0].clone(), row("volk", "vulkan-sdk-1.4.363.0", None, State::Current)]).state == State::Current);

    // Bumping, on fixture text: every pin one step back.
    let mut texts = fixture();
    let up = [
        ("glslang", "16.6.0"),
        ("Vulkan-Headers", "vulkan-sdk-1.4.363.0"),
        ("volk", "vulkan-sdk-1.4.363.0"),
        ("cargo-deny", "0.20.2"),
        ("appimagetool", "1.9.1"),
        ("AppImage type2-runtime", "20251108"),
        ("Rust toolchain", "1.99.1"),
    ];
    let upstream_of = |name: &str| up.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap_or("");
    let mut rows: Vec<Row> =
        pins(&texts)?.iter().map(|p| row(&p.name, &p.pinned, Some(upstream_of(&p.name)), state(&p.pinned, Some(upstream_of(&p.name))))).collect();
    let sdk = sdk_row(&rows);
    rows.push(sdk);
    let sha_of = |url: &str| Ok(fake_sha(url));
    let sha_file = |url: &str| Ok(format!("{}  file\n", fake_sha(url.strip_suffix(".sha256").unwrap_or(url))));
    let no_file = |_: &str| -> Res<String> { Err("no published checksum expected".into()) };
    let lines = bump_all(&mut texts, &rows, &sha_of, &sha_file)?;
    ensure!(
        lines
            == strings(&[
                "glslang: 16.5.0 -> 16.6.0",
                "Vulkan-Headers: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0",
                "volk: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0",
                "cargo-deny: 0.19.9 -> 0.20.2",
                "appimagetool: 1.9.0 -> 1.9.1",
                "Rust toolchain: 1.98.0 -> 1.99.1"
            ])
    );
    ensure!(pins(&texts)?.iter().all(|p| p.pinned == upstream_of(&p.name)));
    ensure!(texts["fetch"].contains(&format!("download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz|{}|bin/glslang", fake_sha(&glslang_url("16.6.0")))));
    ensure!(texts["fetch"].contains("glslang 16.6.0 (") && texts["fetch"].contains("Vulkan-Headers vulkan-sdk-1.4.363.0 and volk vulkan-sdk-1.4.363.0."));
    ensure!(texts["deny"].contains(&format!("const SHA: &str = \"{}\";", fake_sha(&deny_url("0.20.2")))));
    ensure!(texts["build"].contains(&format!(
        "APPIMAGETOOL_SHA256=\"{}\"",
        fake_sha("https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage")
    )));
    ensure!(texts["build"].contains(&format!("RUNTIME_SHA256=\"{}", "e".repeat(64)))); // current: untouched
    ensure!(texts["attribution"].matches("vulkan-sdk-1.4.363.0").count() == 2 && !texts["attribution"].contains("1.4.357"));
    ensure!(!texts["fetch"].contains(&"a".repeat(64)) && !texts["fetch"].contains(&"b".repeat(64)) && !texts["fetch"].contains(&"c".repeat(64)));
    // Nothing behind: nothing bumped.
    let mut current: Vec<Row> = up.iter().map(|(n, v)| row(n, v, Some(v), State::Current)).collect();
    current.push(row(SDK, "x = x", None, State::Current));
    ensure!(bump_all(&mut texts.clone(), &current, &sha_of, &no_file)?.is_empty());
    // A mismatched pair is brought to the common SDK even when neither is behind it.
    let mut texts2 = fixture();
    let fetch = texts2["fetch"].replace("volk/archive/refs/tags/vulkan-sdk-1.4.357.0", "volk/archive/refs/tags/vulkan-sdk-1.4.363.0");
    texts2.insert("fetch", fetch.replace("volk vulkan-sdk-1.4.357.0", "volk vulkan-sdk-1.4.363.0"));
    let attribution = texts2["attribution"].replace("volk) | vulkan-sdk-1.4.357.0", "volk) | vulkan-sdk-1.4.363.0");
    texts2.insert("attribution", attribution);
    let mut rows2 = vec![
        row("Vulkan-Headers", "vulkan-sdk-1.4.357.0", Some("vulkan-sdk-1.4.363.0"), State::Behind),
        row("volk", "vulkan-sdk-1.4.363.0", Some("vulkan-sdk-1.4.363.0"), State::Current),
    ];
    let sdk2 = sdk_row(&rows2);
    rows2.push(sdk2);
    ensure!(bump_all(&mut texts2, &rows2, &sha_of, &no_file)? == strings(&["Vulkan-Headers: vulkan-sdk-1.4.357.0 -> vulkan-sdk-1.4.363.0"]));
    // A published checksum that disagrees with the download stops the bump.
    let deny_rows: Vec<Row> = rows.iter().filter(|r| r.name == "cargo-deny").cloned().chain([row(SDK, "x = x", None, State::Current)]).collect();
    let zeros = |_: &str| Ok("0".repeat(64));
    match bump_all(&mut fixture(), &deny_rows, &sha_of, &zeros) {
        Err(e) => ensure!(e.contains("not the published")),
        Ok(_) => return Err("a wrong published cargo-deny checksum was accepted".into()),
    }
    // A pin whose text has drifted from what the rewrite expects fails loudly instead of half-bumping.
    let mut drifted = fixture();
    drifted.insert("attribution", String::new());
    let headers_rows: Vec<Row> = rows.iter().filter(|r| r.name == "Vulkan-Headers").cloned().chain([row(SDK, "x = x", None, State::Current)]).collect();
    match bump_all(&mut drifted, &headers_rows, &sha_of, &no_file) {
        Err(e) => ensure!(e.contains("ATTRIBUTION.md")),
        Ok(_) => return Err("a bump with a missing site was accepted".into()),
    }
    Ok(())
}

/// Lower-case hex of a digest.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_pins_tool_errors_are_not_the_network() {
        assert!(matches!(up(http::FetchError::Tool("curl failed (exit status 3)".into())), Up::Tool(_)));
        assert!(matches!(up(http::FetchError::Io("timed out".into())), Up::Net(m) if m == "timed out"));
        assert!(
            matches!(up(http::FetchError::Http { code: 404, reason: "Not Found".into(), headers: vec![] }), Up::Net(m) if m == "HTTP Error 404: Not Found")
        );
    }

    #[test]
    fn check_pins_self_test() {
        if let Err(e) = self_test() {
            panic!("{e}");
        }
    }

    #[test]
    fn check_pins_python_text_helpers() {
        assert_eq!(crate::py::repr_str(r#"^X="([^"]+)""#), r#"'^X="([^"]+)"'"#);
        assert_eq!(crate::py::repr_str("a'b"), "\"a'b\"");
        assert_eq!(crate::py::repr_str("a\\.b\n"), "'a\\\\.b\\n'");
        assert_eq!(py_escape("Vulkan-Headers"), "Vulkan\\-Headers");
        assert!(regex::Regex::new(&py_escape("a b-c.d")).unwrap().is_match("a b-c.d"));
    }

    #[test]
    fn check_pins_equal_versions_keep_the_first() {
        // Python's max() returns the first of equal keys.
        assert_eq!(newest(&strings(&["v1.0", "1.0"])).as_deref(), Some("v1.0"));
        assert_eq!(newest(&strings(&["1.0", "v1.0"])).as_deref(), Some("1.0"));
    }

    #[test]
    fn check_pins_missing_pattern_message() {
        let e = one(r#"^X="([^"]+)""#, "nothing", "x", true).unwrap_err();
        assert_eq!(e, r#"x: pattern '^X="([^"]+)"' not found"#);
    }
}
