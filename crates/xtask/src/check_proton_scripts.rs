//! Does Neural Forge's patch of Proton's `proton` script still apply to the newest upstream scripts?
//!
//! A Neural Forge Proton copy gets two DLL overrides after the `NVIDIA_WINE_DLL_DIR` block of its source's `proton` script
//! (crates/supervisor/src/proton_tool.rs, `patch_proton_script`; docs/PROTON_NGX.md). The patch is refused unless that block
//! is there exactly once. Valve's script still carries a TODO to replace the block with an `NGXCore\NGXPath` registry key:
//! when that lands (or CachyOS's fork changes it) the anchor breaks and no copy can be made from that Proton. This fetches
//! the scripts of
//!
//! ```text
//! Valve's Proton Experimental   the newest experimental_<major>.<minor> branch, and bleeding-edge
//! proton-cachyos                the newest cachyos_<major>.<minor>_<date>/main release branch, and bleeding-edge
//! ```
//!
//! through the GitHub API ($GITHUB_TOKEN raises its rate limit) and runs the patch on each
//! (`cargo run -p neural-forge-supervisor --example check_proton_script`).
//!
//! ```text
//! cargo xtask check-proton-scripts              print a Markdown report; exit 0 when the patch applies to every script, 10 when it
//!                                               does not to one (or a repo has no release branch the check knows), 2 when upstream
//!                                               could not be reached (a network failure: retry later, nothing is known), 1 when the
//!                                               check itself failed (the patch example did not build or run, the API answered in a
//!                                               shape it does not know, any other error): that must fail the run
//! cargo xtask check-proton-scripts --decide STATE ISSUE_STATE OLD_BODY NEW_BODY
//!                                               print what to do with the standing issue (create, edit, reopen, close; nothing
//!                                               for none): STATE is current or broken, ISSUE_STATE none, OPEN or CLOSED, the
//!                                               bodies are files. An issue closed by hand is not reopened while the report is
//!                                               the one it was closed with (the report's hash is kept in the body)
//! cargo xtask check-proton-scripts --self-test  check the branch choice, the report, the exit codes and the issue decisions offline
//! ```
//!
//! .github/workflows/proton-script-check.yml runs it weekly; one issue carries the report.

use crate::check_pins::http::{self, FetchError};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

const BROKEN: u8 = 10;
const NETWORK: u8 = 2;
const TOOL: u8 = 1;
const USER_AGENT: &str = "neural-forge-proton-script-check";
const REPOS: [(&str, &str); 2] = [("ValveSoftware/Proton", "Valve's Proton Experimental"), ("CachyOS/proton-cachyos", "proton-cachyos")];

/// Why a step failed: the network (retry later), the check itself (must fail the run), or anything else (also fails it).
#[derive(Debug)]
enum Failure {
    Fetch(FetchError),
    Tool(String),
    Other(String),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Fetch(e @ FetchError::Http { .. }) => write!(f, "HTTP error: {e}"),
            Failure::Fetch(e) => write!(f, "{e}"),
            Failure::Tool(m) | Failure::Other(m) => f.write_str(m),
        }
    }
}

type ResultLine = (String, String, String);
type Api<'a> = &'a dyn Fn(&str, bool) -> Result<Vec<u8>, Failure>;
type Patch<'a> = &'a dyn Fn(&[PathBuf]) -> Result<Vec<String>, Failure>;

fn release_branch(repo: &str) -> &'static str {
    match repo {
        "ValveSoftware/Proton" => r"experimental_(\d+)\.(\d+)",
        _ => r"cachyos_(\d+)\.(\d+)_(\d{8})/main",
    }
}

fn label(repo: &str) -> &'static str {
    REPOS.iter().find(|(r, _)| *r == repo).map(|(_, l)| *l).unwrap_or("")
}

/// The branch among `names` matching `pattern` (its groups are ints, compared as a tuple) with the highest groups.
fn newest(names: &[String], pattern: &str) -> Option<String> {
    let re = regex::RegexBuilder::new(&format!(r"\A(?:{pattern})\z")).unicode(false).build().expect("static pattern");
    names
        .iter()
        .filter_map(|name| {
            let caps = re.captures(name)?;
            let groups: Vec<u128> = caps.iter().skip(1).flatten().map(|g| g.as_str().parse().unwrap_or(u128::MAX)).collect();
            Some((groups, name.clone()))
        })
        .max()
        .map(|(_, name)| name)
}

/// The branches of `repo` whose `proton` script is checked. The release branch comes first and is None when no branch
/// matches its pattern (upstream renamed them: the check no longer knows what to look at, which is a failure).
fn refs_for(repo: &str, branches: &[String]) -> Vec<Option<String>> {
    let mut refs = vec![newest(branches, release_branch(repo))];
    if branches.iter().any(|b| b == "bleeding-edge") {
        refs.push(Some("bleeding-edge".to_string()));
    }
    refs
}

fn percent_encode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"_.-~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// The first non-blank token among the environment variables, trimmed. One with a line break inside is an error (it
/// would end the Authorization header and start another): it is never sent.
fn token(vars: [Result<String, std::env::VarError>; 2]) -> Result<Option<String>, String> {
    let Some(token) = vars.into_iter().filter_map(Result::ok).map(|t| t.trim().to_string()).find(|t| !t.is_empty()) else { return Ok(None) };
    if token.contains(['\r', '\n']) {
        return Err("GITHUB_TOKEN/GH_TOKEN contains a line break; refusing to send it".into());
    }
    Ok(Some(token))
}

fn api(path: &str, raw: bool) -> Result<Vec<u8>, Failure> {
    let accept = if raw { "application/vnd.github.raw" } else { "application/vnd.github+json" };
    let mut headers = vec![("Accept".to_string(), accept.to_string()), ("User-Agent".to_string(), USER_AGENT.to_string())];
    if let Some(token) = token([std::env::var("GITHUB_TOKEN"), std::env::var("GH_TOKEN")]).map_err(Failure::Tool)? {
        headers.push(("Authorization".to_string(), format!("Bearer {token}")));
    }
    let refs: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    http::get(&format!("https://api.github.com/{path}"), &refs, 60).map_err(|e| match e {
        FetchError::Tool(m) => Failure::Tool(m),
        e => Failure::Fetch(e),
    })
}

fn branches_of(repo: &str, api: Api) -> Result<Vec<String>, Failure> {
    let (mut names, mut page) = (Vec::new(), 1);
    loop {
        let body = api(&format!("repos/{repo}/branches?per_page=100&page={page}"), false)?;
        let chunk: serde_json::Value = serde_json::from_slice(&body).map_err(|e| Failure::Other(format!("the branch list is not JSON: {e}")))?;
        let known: Option<Vec<&str>> = chunk.as_array().and_then(|a| a.iter().map(|b| b.get("name").and_then(|n| n.as_str())).collect());
        let Some(known) = known else {
            let shown: String = crate::py::repr(Some(&chunk)).chars().take(200).collect();
            return Err(Failure::Tool(format!("the branch list of {repo} has a shape this check does not know: {shown}")));
        };
        names.extend(known.iter().map(|n| n.to_string()));
        if known.len() < 100 {
            return Ok(names);
        }
        page += 1;
    }
}

/// `results`: (repo, ref, outcome line) with 'ok ...' or 'fail ...: why'. The Markdown report, and whether all are ok.
fn report(results: &[ResultLine]) -> (String, bool) {
    let bad: Vec<&ResultLine> = results.iter().filter(|r| !r.2.starts_with("ok")).collect();
    let mut lines: Vec<String> = Vec::new();
    if !bad.is_empty() {
        lines.push("Neural Forge's patch of Proton's `proton` script no longer applies to the newest upstream script(s), so no copy can be made from that Proton once it ships:".into());
        lines.push(String::new());
    } else {
        lines.push("Neural Forge's patch of Proton's `proton` script applies to the newest upstream scripts:".into());
        lines.push(String::new());
    }
    for (repo, git_ref, outcome) in results {
        let mark = if outcome.starts_with("ok") { "ok" } else { outcome.split(' ').next().unwrap_or("") };
        let why = if mark == "ok" { String::new() } else { format!(" {}", outcome.split_once(": ").map_or(outcome.as_str(), |(_, w)| w)) };
        lines.push(format!("- {} (`{repo}` @ `{git_ref}`): {mark}{why}", label(repo)));
    }
    if !bad.is_empty() {
        lines.push(String::new());
        lines.push(
            "Valve's script has a TODO to replace the `NVIDIA_WINE_DLL_DIR` block with an `NGXCore\\NGXPath` registry key; when it lands the anchor in \
             `proton_tool::patch_proton_script` has to move to it (docs/PROTON_NGX.md)."
                .into(),
        );
    }
    let text = lines.join("\n") + "\n";
    // The hash covers the repos and their failures only, not the branch names: a new dated cachyos_* branch with the
    // same failure is the same report and must not reopen an issue closed by hand.
    let mut failures: Vec<String> = bad.iter().map(|(repo, _, outcome)| format!("{repo}: {outcome}")).collect();
    failures.sort();
    let hash = crate::check_pins::hex(&Sha256::digest(failures.join("\n").as_bytes()));
    (format!("{text}\n<!-- report-sha256: {hash} -->\n"), bad.is_empty())
}

/// The report hash kept in an issue body, or None.
fn marker(body: &str) -> Option<String> {
    let re = regex::Regex::new(r"<!-- report-sha256: ([0-9a-f]{64}) -->").expect("static pattern");
    re.captures(body).map(|c| c[1].to_string())
}

/// What to do with the standing issue: a list of create, edit, reopen, close. At most one notification per change
/// of the report: an issue closed by hand stays closed while the report is the one it was closed with.
fn decide(state: &str, issue_state: &str, old_body: &str, new_body: &str) -> Vec<&'static str> {
    let same = marker(old_body).is_some() && marker(old_body) == marker(new_body);
    match (state, issue_state) {
        ("broken", "none") => vec!["create"],
        ("broken", _) if same => vec![],
        ("broken", "OPEN") => vec!["edit"],
        ("broken", _) => vec!["edit", "reopen"],
        ("current", "OPEN") => vec!["edit", "close"],
        // Silent edit, no reopen: the body then holds the good report's hash, so the same failure coming back later
        // is a change of the report and reopens the issue.
        ("current", "CLOSED") if !same => vec!["edit"],
        _ => vec![],
    }
}

/// The example's output lines for `files` (paths in). `command` replaces the cargo run (NF_CHECK_PROTON_SCRIPT).
fn run_patch(files: &[PathBuf], command: Option<&str>) -> Result<Vec<String>, Failure> {
    let mut cmd = match command {
        Some(c) if !c.is_empty() => Command::new(c),
        _ => {
            let mut c = Command::new("cargo");
            c.args(["run", "-q", "-p", "neural-forge-supervisor", "--example", "check_proton_script", "--"]);
            c
        }
    };
    // A missing or unrunnable command is the check's own failure, not upstream's: it must fail the run.
    let out = cmd.args(files).current_dir(crate::root()).output().map_err(|e| Failure::Tool(format!("the patch check could not start: {e}")))?;
    let code = out.status.code();
    if code != Some(0) && code != Some(i32::from(BROKEN)) {
        let shown = match code {
            Some(c) => c.to_string(),
            None => {
                use std::os::unix::process::ExitStatusExt;
                format!("-{}", out.status.signal().unwrap_or(0))
            }
        };
        let (stderr, stdout) = (String::from_utf8_lossy(&out.stderr).trim().to_string(), String::from_utf8_lossy(&out.stdout).trim().to_string());
        return Err(Failure::Tool(format!("the patch check could not run (exit {shown}): {}", if stderr.is_empty() { stdout } else { stderr })));
    }
    Ok(crate::py::splitlines(&String::from_utf8_lossy(&out.stdout)).into_iter().map(String::from).collect())
}

/// A scratch directory removed on drop, also when a step fails.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self, Failure> {
        let dir = crate::check_pins::http::private_dir("check-proton-scripts")
            .map_err(|e| Failure::Other(format!("could not create a private temporary directory: {e}")))?;
        Ok(TempDir(dir))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn check(api: Api, patch: Patch) -> Result<(String, u8), Failure> {
    let mut results: Vec<ResultLine> = Vec::new();
    let mut files: Vec<(&str, String, PathBuf)> = Vec::new();
    let tmp = TempDir::new()?;
    for (repo, _) in REPOS {
        for git_ref in refs_for(repo, &branches_of(repo, api)?) {
            let Some(git_ref) = git_ref else {
                results.push((repo.into(), "-".into(), format!("fail proton: no release branch matches {} any more", release_branch(repo))));
                continue;
            };
            let path = tmp.0.join(format!("{}.proton", files.len()));
            let script = api(&format!("repos/{repo}/contents/proton?ref={}", percent_encode(&git_ref)), true)?;
            std::fs::write(&path, script).map_err(|e| Failure::Tool(format!("could not write {repo} @ {git_ref}'s script: {e}")))?;
            files.push((repo, git_ref, path));
        }
    }
    let paths: Vec<PathBuf> = files.iter().map(|f| f.2.clone()).collect();
    let lines = patch(&paths)?;
    let mut by_path: HashMap<String, &String> = HashMap::new();
    for l in &lines {
        let key = l.split(' ').nth(1).ok_or_else(|| Failure::Other("a patch-check line has no second field".into()))?;
        by_path.insert(key.trim_end_matches(':').to_string(), l);
    }
    for (repo, git_ref, path) in &files {
        let shown = path.to_string_lossy().to_string();
        let line = by_path.get(&shown).ok_or_else(|| Failure::Tool(format!("no answer for {repo} @ {git_ref}")))?;
        results.push((repo.to_string(), git_ref.clone(), line.replace(&shown, "proton")));
    }
    let (text, ok) = report(&results);
    Ok((text, if ok { 0 } else { BROKEN }))
}

/// Whether `f` is a failure to reach upstream (retry later) rather than one of the check itself.
fn is_network(f: &Failure) -> bool {
    match f {
        Failure::Fetch(e @ FetchError::Http { code, .. }) => {
            let rate_limited = *code == 403 && (e.header("X-RateLimit-Remaining") == Some("0") || e.header("Retry-After").is_some());
            *code >= 500 || *code == 429 || rate_limited
        }
        // A refused connection, a timeout, a TLS failure, a dropped transfer.
        Failure::Fetch(FetchError::Io(_)) => true,
        _ => false,
    }
}

struct Outcome {
    code: u8,
    stdout: String,
    stderr: String,
}

/// check() with its failures turned into exit codes.
fn run_with(api: Api, patch: Patch) -> Outcome {
    match check(api, patch) {
        Ok((text, code)) => Outcome { code, stdout: text, stderr: String::new() },
        Err(e) if is_network(&e) => Outcome { code: NETWORK, stdout: String::new(), stderr: format!("could not reach upstream: {e}\n") },
        Err(e) => Outcome { code: TOOL, stdout: String::new(), stderr: format!("the check failed: {e}\n") },
    }
}

/// A file as Python's `Path.read_text()` gives it: UTF-8 with universal newlines.
fn read_text(path: &str) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(crate::py::universal_newlines(&text))
}

pub fn main(args: &[String]) -> ExitCode {
    if args == ["--self-test"] {
        return match self_test() {
            Ok(()) => {
                println!("check-proton-scripts: self-test ok");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("check-proton-scripts self-test failed: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() == 5 && args[0] == "--decide" {
        return match (read_text(&args[3]), read_text(&args[4])) {
            (Ok(old), Ok(new)) => {
                println!("{}", decide(&args[1], &args[2], &old, &new).join(" "));
                ExitCode::SUCCESS
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    if !args.is_empty() {
        eprintln!("usage: cargo xtask check-proton-scripts [--self-test | --decide STATE ISSUE_STATE OLD_BODY NEW_BODY]");
        return ExitCode::FAILURE;
    }
    let command = std::env::var("NF_CHECK_PROTON_SCRIPT").ok();
    let out = run_with(&api, &|files| run_patch(files, command.as_deref()));
    print!("{}", out.stdout);
    eprint!("{}", out.stderr);
    ExitCode::from(out.code)
}

// ---- Self-test ---------------------------------------------------------------------------------

macro_rules! ensure {
    ($cond:expr) => {
        if !$cond {
            return Err(format!("assertion failed: {}", stringify!($cond)));
        }
    };
    ($cond:expr, $why:expr) => {
        if !$cond {
            return Err(format!("assertion failed: {}", $why));
        }
    };
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn res(repo: &str, git_ref: &str, outcome: &str) -> ResultLine {
    (repo.into(), git_ref.into(), outcome.into())
}

fn http_error(code: u16, reason: &str, headers: &[(&str, &str)]) -> Failure {
    Failure::Fetch(FetchError::Http { code, reason: reason.into(), headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() })
}

/// run()'s exit codes with upstream and the patch example stubbed.
fn exit_code_tests() -> Result<(), String> {
    let cases: Vec<(Failure, u8)> = vec![
        (Failure::Fetch(FetchError::Io("no route".into())), NETWORK),
        (Failure::Fetch(FetchError::Io("timed out".into())), NETWORK),
        (Failure::Fetch(FetchError::Io("reset".into())), NETWORK),
        (http_error(502, "bad gateway", &[]), NETWORK),
        (http_error(403, "rate limited", &[("X-RateLimit-Remaining", "0")]), NETWORK),
        (http_error(403, "secondary limit", &[("Retry-After", "60")]), NETWORK),
        (Failure::Fetch(FetchError::Io("incomplete read".into())), NETWORK),
        (Failure::Fetch(FetchError::Io("remote end closed".into())), NETWORK),
        (Failure::Fetch(FetchError::Io("handshake".into())), NETWORK),
        (Failure::Fetch(FetchError::Io("network unreachable".into())), NETWORK),
        (http_error(403, "forbidden", &[]), TOOL),
        (http_error(404, "not found", &[]), TOOL),
        (http_error(401, "bad credentials", &[]), TOOL),
        (Failure::Other("a branch has no name".into()), TOOL),
        // curl itself failed (not the network): the check fails, it is not "retry later".
        (Failure::Tool("curl failed (exit status 3): URL rejected".into()), TOOL),
    ];
    let no_patch = |_: &[PathBuf]| -> Result<Vec<String>, Failure> { Err(Failure::Other("the patch must not run".into())) };
    for (failure, want) in cases {
        let shown = failure.to_string();
        let slot = RefCell::new(Some(failure));
        let failing = |_: &str, _: bool| -> Result<Vec<u8>, Failure> { Err(slot.borrow_mut().take().expect("one call")) };
        let got = run_with(&failing, &no_patch).code;
        ensure!(got == want, format!("{shown}: got {got}, want {want}"));
    }
    let ok_patch = |files: &[PathBuf]| -> Result<Vec<String>, Failure> { Ok(files.iter().map(|f| format!("ok {}", f.display())).collect()) };
    let moved = |_: &str, _: bool| -> Result<Vec<u8>, Failure> { Ok(br#"{"message": "moved"}"#.to_vec()) };
    ensure!(run_with(&moved, &ok_patch).code == TOOL, "an API answer of the wrong shape fails the run");
    let empty = |_: &str, raw: bool| -> Result<Vec<u8>, Failure> { Ok(if raw { b"script".to_vec() } else { b"[]".to_vec() }) };
    ensure!(run_with(&empty, &ok_patch).code == BROKEN, "no release branch is a failure, not 'applies'");
    let both = |_: &str, raw: bool| -> Result<Vec<u8>, Failure> {
        Ok(if raw { b"script".to_vec() } else { br#"[{"name": "experimental_11.0"}, {"name": "cachyos_11.0_20261014/main"}]"#.to_vec() })
    };
    ensure!(run_with(&both, &ok_patch).code == 0);
    let false_cmd = |files: &[PathBuf]| run_patch(files, Some("false"));
    ensure!(run_with(&both, &false_cmd).code == TOOL, "the patch example failing to run fails the run");
    let missing = |files: &[PathBuf]| run_patch(files, Some("/nonexistent/neural-forge-check-proton-script"));
    ensure!(run_with(&both, &missing).code == TOOL, "a patch command that cannot start fails the run, it is not a network failure");
    Ok(())
}

fn self_test() -> Result<(), String> {
    let v =
        strings(&["experimental_10.0", "experimental_9.0", "experimental_11.0", "experimental_5.13-dxvk-d3d11-funny-refcount", "bleeding-edge", "proton_11.0"]);
    ensure!(refs_for("ValveSoftware/Proton", &v) == [Some("experimental_11.0".to_string()), Some("bleeding-edge".to_string())]);
    let c = strings(&[
        "cachyos_10.0_20251120/main",
        "cachyos_11.0_20261014/main",
        "cachyos_11.0_20261014/main_native",
        "cachyos_11.0_20261001/main",
        "cachyos_9.0_20250117/main-runtime",
        "cachyos_main",
    ]);
    ensure!(refs_for("CachyOS/proton-cachyos", &c) == [Some("cachyos_11.0_20261014/main".to_string())]);
    ensure!(
        refs_for("ValveSoftware/Proton", &strings(&["proton_11.0", "bleeding-edge"])) == [None, Some("bleeding-edge".to_string())],
        "a missing release branch is kept as None"
    );
    let mut with_edge = c.clone();
    with_edge.push("bleeding-edge".into());
    ensure!(refs_for("CachyOS/proton-cachyos", &with_edge).last() == Some(&Some("bleeding-edge".to_string())));
    ensure!(
        newest(&strings(&["experimental_9.0", "experimental_10.0"]), r"experimental_(\d+)\.(\d+)").as_deref() == Some("experimental_10.0"),
        "10 is newer than 9"
    );
    ensure!(newest(&[], r"x(\d+)").is_none());
    let good = vec![res("ValveSoftware/Proton", "experimental_11.0", "ok proton"), res("CachyOS/proton-cachyos", "bleeding-edge", "ok proton")];
    let (text, ok) = report(&good);
    ensure!(ok && text.contains("applies") && !text.contains("no longer"), text);
    let fail_new = "fail proton: the proton script's NVIDIA_WINE_DLL_DIR block was found 0 times, not once";
    let mut bad = good.clone();
    bad.push(res("CachyOS/proton-cachyos", "cachyos_11.0_20261014/main", fail_new));
    let (text, ok) = report(&bad);
    ensure!(!ok && text.contains("no longer applies") && text.contains("found 0 times") && text.contains("NGXPath"), text);
    ensure!(marker(&text).is_some() && marker(&text) != marker(&report(&good).0));
    // Issue decisions: one notification per change of the report, never a reopen of a hand-closed issue with the same one.
    let other = report(&[bad[0].clone(), res("CachyOS/proton-cachyos", "-", "fail proton: no release branch matches x any more")]).0;
    let good_report = report(&good).0;
    ensure!(decide("broken", "none", "", &text) == ["create"]);
    ensure!(decide("broken", "OPEN", &text, &text).is_empty());
    ensure!(decide("broken", "OPEN", &other, &text) == ["edit"]);
    ensure!(decide("broken", "CLOSED", &text, &text).is_empty(), "closed by hand with the same report: left closed");
    ensure!(decide("broken", "CLOSED", &other, &text) == ["edit", "reopen"]);
    ensure!(decide("broken", "CLOSED", "a body without a hash", &text) == ["edit", "reopen"]);
    ensure!(decide("current", "OPEN", &text, &good_report) == ["edit", "close"]);
    ensure!(decide("current", "CLOSED", &text, &good_report) == ["edit"], "closed + current + a different report: silent edit, no reopen");
    ensure!(decide("current", "CLOSED", &good_report, &good_report).is_empty() && decide("current", "none", "", &good_report).is_empty());
    // The hash ignores branch names: a new dated branch with the same failure is the same report.
    let r1 = report(&[good[0].clone(), res("CachyOS/proton-cachyos", "cachyos_11.0_20261014/main", fail_new)]).0;
    let r2 = report(&[good[0].clone(), res("CachyOS/proton-cachyos", "cachyos_11.0_20261101/main", fail_new)]).0;
    ensure!(r1 != r2 && marker(&r1) == marker(&r2) && decide("broken", "CLOSED", &r1, &r2).is_empty());
    ensure!(marker(&r1) != marker(&good_report));
    // Exit codes: a network failure is 2 (retry, quiet), a failure of the check itself 1, a missing release branch 10.
    exit_code_tests()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_proton_scripts_self_test() {
        if let Err(e) = self_test() {
            panic!("{e}");
        }
    }

    #[test]
    fn check_proton_scripts_helpers() {
        assert_eq!(percent_encode("cachyos_11.0_20261014/main"), "cachyos_11.0_20261014%2Fmain");
        let ok = |t: &str| Ok::<String, std::env::VarError>(t.to_string());
        assert_eq!(token([ok(" abc\n"), ok("other")]), Ok(Some("abc".into())));
        assert_eq!(token([ok("  \n"), ok("other")]), Ok(Some("other".into())));
        assert_eq!(token([Err(std::env::VarError::NotPresent), ok("")]), Ok(None));
        assert!(token([ok("abc\r\nX-Evil: 1"), ok("")]).is_err());
        assert!(token([ok("abc\ndef"), ok("")]).is_err());
    }

    #[test]
    fn check_proton_scripts_answers_by_path() {
        let api = |_: &str, raw: bool| -> Result<Vec<u8>, Failure> {
            Ok(if raw { b"s".to_vec() } else { br#"[{"name": "experimental_1.0"}, {"name": "cachyos_1.0_20260101/main"}]"#.to_vec() })
        };
        let patch = |files: &[PathBuf]| -> Result<Vec<String>, Failure> {
            Ok(files.iter().enumerate().map(|(i, f)| if i == 0 { format!("fail {}: broken", f.display()) } else { format!("ok {}", f.display()) }).collect())
        };
        let out = run_with(&api, &patch);
        assert_eq!(out.code, BROKEN);
        assert!(out.stdout.contains("- Valve's Proton Experimental (`ValveSoftware/Proton` @ `experimental_1.0`): fail broken"), "{}", out.stdout);
        assert!(out.stdout.contains("- proton-cachyos (`CachyOS/proton-cachyos` @ `cachyos_1.0_20260101/main`): ok\n"), "{}", out.stdout);
    }
}
