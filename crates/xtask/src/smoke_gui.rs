//! cargo xtask smoke-gui APPDIR: starts the extracted AppImage's GUI (APPDIR/AppRun) on a virtual X display (Xvfb, its
//! own D-Bus session) and checks that it reaches its main loop: its window ("Neural Forge", app id
//! io.github.labj1987.NeuralForge) is mapped (xdotool), it is still running, and its stderr holds no loader error, no
//! panic and no line the GUI logged at WARN or ERROR level (it logs "neural-forge: LEVEL [target] message"). Xvfb has no
//! GPU: the window is drawn on lavapipe (VK_DRIVER_FILES set to Mesa's lvp_icd*.json when there is one, as CI does for
//! the layer's tests), never on the machine's own GPU. Everything it writes goes to a scratch HOME, XDG directories and
//! a private XDG_RUNTIME_DIR, never the user's. Needs xvfb-run (xvfb), dbus-run-session (dbus-daemon) and xdotool.
//!   --self-test   check the stderr verdicts, no GUI
//!
//! It replaced scripts/smoke-gui.sh: the same checks, output and exit codes. What the script ran inside the virtual
//! session as a `bash -c` text is this program again, started there as `xtask smoke-gui --in-session APPDIR DIR WAIT`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use regex::Regex;

use crate::sh::{self, Fail, Res};

const USAGE: &str = "usage: cargo xtask smoke-gui APPDIR | --self-test";
const IN_SESSION: &str = "--in-session";

/// Why the run's stderr fails (the offending lines, then a summary), or nothing.
fn verdict(stderr: &str) -> String {
    let bad = Regex::new(r"error while loading shared libraries|undefined symbol|panicked|can't start|^neural-forge: (WARN|ERROR) ").unwrap();
    // What the bare session lacks is not the GUI's fault: Xvfb has no XInput pointer device (the toolkit reports that at
    // error level and works without).
    let mut lines: Vec<&str> = stderr.lines().filter(|l| !l.contains("Found no xinput mouse pointers") && bad.is_match(l)).collect();
    if lines.is_empty() {
        return String::new();
    }
    lines.push("errors on stderr");
    lines.join("\n")
}

fn verdict_of(file: &Path) -> String {
    verdict(&String::from_utf8_lossy(&std::fs::read(file).unwrap_or_default()))
}

/// What a start wrote of itself, on one line ("mapped running" for a window that showed and a GUI still running 2 s
/// later; "running" for no window; nothing for a GUI that ended).
fn read_state(file: &Path) -> String {
    std::fs::read_to_string(file).unwrap_or_default().lines().collect::<Vec<_>>().join(" ")
}

/// pass (window mapped, GUI running, stderr clean), retry (no window, but the GUI was running and its stderr clean: the
/// one case that gets a second start) or fail.
fn outcome(state: &str, why: &str) -> &'static str {
    match (state, why.is_empty()) {
        ("mapped running", true) => "pass",
        ("running", true) => "retry",
        _ => "fail",
    }
}

/// The self-test's own failure: said on stdout, exit 1.
fn told(message: String) -> Fail {
    println!("self-test: {message}");
    Fail::silent(1)
}

fn write(path: &Path, text: &str) -> Res<()> {
    std::fs::write(path, text).map_err(|e| Fail::new(1, format!("{}: {e}", path.display())))
}

fn write_script(path: &Path, text: &str) -> Res<()> {
    write(path, text)?;
    std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).map_err(|e| Fail::new(1, format!("{}: {e}", path.display())))
}

/// This program again, with its output caught.
fn nested(app: &Path, wait: Option<&str>) -> Res<(bool, String)> {
    let exe = std::env::current_exe().map_err(|e| Fail::new(1, format!("cannot find this program: {e}")))?;
    let mut again = Command::new(exe);
    again.arg("smoke-gui").arg(app).env("SMOKE_GUI_NESTED", "1").stdin(Stdio::null());
    if let Some(wait) = wait {
        again.env("SMOKE_GUI_WAIT", wait);
    }
    let out = again.output().map_err(|e| sh::not_started(&again, e))?;
    Ok((out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))))
}

fn self_test() -> Res<ExitCode> {
    let t = sh::mktemp()?;
    let t = t.path();
    // A start as the run records it: the window mapped, then the GUI still running.
    write(&t.join("state"), "mapped\nrunning\n")?;
    let state = read_state(&t.join("state"));
    if state != "mapped running" {
        return Err(told(format!("a recorded start read as '{state}'")));
    }
    if outcome(&state, "") != "pass" {
        return Err(told("a working GUI did not pass".into()));
    }
    if outcome(&read_state(&t.join("missing")), "") != "fail" {
        return Err(told("a GUI that ended passed".into()));
    }
    if outcome("running", "") != "retry" {
        return Err(told("no window with a running GUI is not retried".into()));
    }
    if outcome("running", "neural-forge: ERROR x") != "fail" {
        return Err(told("an error on stderr is retried".into()));
    }
    if outcome("mapped running", "neural-forge: ERROR x") != "fail" {
        return Err(told("an error on stderr passed".into()));
    }
    // The whole run against a stand-in GUI (a window with the GUI's title), where the tools for one are installed.
    if sh::var("SMOKE_GUI_NESTED").is_none() && ["xmessage", "xvfb-run", "xdotool", "dbus-run-session"].iter().all(|tool| sh::which(tool).is_some()) {
        let app = t.join("app");
        std::fs::create_dir_all(&app).map_err(|e| Fail::new(1, format!("{}: {e}", app.display())))?;
        write_script(&app.join("AppRun"), "#!/bin/sh\nexec xmessage -title \"Neural Forge\" stand-in\n")?;
        let (ok, output) = nested(&app, None)?;
        if !ok {
            println!("self-test: a working stand-in GUI failed:");
            print!("{output}");
            return Err(Fail::silent(1));
        }
        write_script(&app.join("AppRun"), "#!/bin/sh\nexec sleep 100\n")?;
        let (ok, output) = nested(&app, Some("4"))?;
        if ok {
            return Err(told("a GUI without a window passed".into()));
        }
        if !output.contains("starting it once more") {
            println!("self-test: no second start for a GUI without a window:");
            print!("{output}");
            return Err(Fail::silent(1));
        }
    }
    if !verdict("neural-forge: INFO [neural_forge::app] window opened\nneural-forge: DEBUG [x] WARN in a message is fine\n").is_empty() {
        return Err(told("a clean run failed".into()));
    }
    if !verdict("neural-forge: ERROR [gpui_linux::linux::x11::client] Found no xinput mouse pointers.\n").is_empty() {
        return Err(told("what the bare session lacks failed a run".into()));
    }
    for bad in [
        "neural-forge: WARN [neural_forge::status] no channel",
        "neural-forge: ERROR [gpui] boom",
        "thread 'main' panicked at src/main.rs:1:1:",
        "./neural-forge: error while loading shared libraries: libxcb.so.1: cannot open shared object file",
        "./neural-forge: symbol lookup error: ./neural-forge: undefined symbol: xkb_x",
        "Neural Forge: Neural Forge can't start because these system libraries are missing: libEGL.so.1.",
    ] {
        if verdict(&format!("neural-forge: INFO [x] started\n{bad}\n")).is_empty() {
            return Err(told(format!("passed: {bad}")));
        }
    }
    println!("smoke-gui self-test: ok");
    Ok(ExitCode::SUCCESS)
}

/// Inside the virtual session: the GUI, up to WAIT half-seconds for its window, a check that it still runs 2 s later,
/// then its end. Writes `out`, `err` and `state` into DIR.
fn in_session(appdir: &str, dir: &str, wait: &str) -> Res<ExitCode> {
    let dir = Path::new(dir);
    let file = |name: &str| std::fs::File::create(dir.join(name)).map_err(|e| Fail::new(1, format!("{}: {e}", dir.join(name).display())));
    let mut gui = Command::new(Path::new(appdir).join("AppRun"));
    let mut gui = gui.stdout(file("out")?).stderr(file("err")?).spawn().map_err(|e| sh::not_started(&gui, e))?;
    let mut state = String::new();
    for _ in 0..wait.parse::<u64>().unwrap_or(0) {
        let found = Command::new("xdotool").args(["search", "--onlyvisible", "--name", "^Neural Forge$"]).stdout(Stdio::null()).stderr(Stdio::null()).status();
        if found.is_ok_and(|s| s.success()) {
            state.push_str("mapped\n");
            break;
        }
        if !matches!(gui.try_wait(), Ok(None)) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    std::thread::sleep(std::time::Duration::from_secs(2));
    if matches!(gui.try_wait(), Ok(None)) {
        state.push_str("running\n");
        sh::kill(gui.id() as i32, libc::SIGTERM);
    }
    if !state.is_empty() {
        write(&dir.join("state"), &state)?;
    }
    let _ = gui.wait();
    Ok(ExitCode::SUCCESS)
}

/// One start in a fresh virtual session; the state it recorded and why its stderr fails.
fn start_once(exe: &Path, appdir: &Path, s: &Path, icds: &str, wait: &str) -> (String, String) {
    for name in ["state", "out", "err"] {
        let _ = std::fs::remove_file(s.join(name));
    }
    let home = s.join("home");
    let mut session = Command::new("xvfb-run");
    session.env_clear().env("PATH", "/usr/local/bin:/usr/bin:/bin").env("HOME", &home).env("XDG_RUNTIME_DIR", s.join("run"));
    session.env("XDG_CONFIG_HOME", home.join(".config")).env("XDG_DATA_HOME", home.join(".local/share"));
    session.env("XDG_STATE_HOME", home.join(".local/state")).env("XDG_CACHE_HOME", home.join(".cache")).env("LANG", "C.UTF-8");
    if !icds.is_empty() {
        session.env("VK_DRIVER_FILES", icds);
    }
    session.args(["-a", "-s", "-screen 0 1920x1080x24", "dbus-run-session", "--"]).arg(exe).args(["smoke-gui", IN_SESSION]).arg(appdir).arg(s).arg(wait);
    if let Ok(log) = std::fs::File::create(s.join("session")) {
        session.stderr(log);
    }
    let _ = session.status();
    (read_state(&s.join("state")), verdict_of(&s.join("err")))
}

fn indented(file: &Path) {
    for line in String::from_utf8_lossy(&std::fs::read(file).unwrap_or_default()).lines() {
        eprintln!("    {line}");
    }
}

fn run(args: &[String]) -> Res<ExitCode> {
    match args {
        [flag, ..] if flag == "--self-test" => return self_test(),
        [flag, appdir, dir, wait] if flag == IN_SESSION => return in_session(appdir, dir, wait),
        [_] => {}
        _ => return Err(Fail::new(2, USAGE)),
    }
    let appdir = std::path::absolute(&args[0]).ok().filter(|d| d.is_dir()).ok_or_else(|| Fail::new(1, format!("smoke-gui: {}: not a directory", args[0])))?;
    for tool in ["xvfb-run", "dbus-run-session", "xdotool"] {
        if sh::which(tool).is_none() {
            return Err(Fail::new(1, format!("smoke-gui: {tool} is missing")));
        }
    }
    let mut icds: Vec<PathBuf> = std::fs::read_dir("/usr/share/vulkan/icd.d").map(|entries| entries.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    icds.retain(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("lvp_icd") && n.to_string_lossy().ends_with(".json")));
    icds.sort();
    let icds = icds.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>().join(":");
    if icds.is_empty() {
        eprintln!("smoke-gui: no lavapipe ICD in /usr/share/vulkan/icd.d (mesa-vulkan-drivers); the GUI picks its own renderer");
    }
    let exe = std::env::current_exe().map_err(|e| Fail::new(1, format!("cannot find this program: {e}")))?;
    let scratch = sh::mktemp()?;
    let s = scratch.path();
    for dir in ["home", "run"] {
        std::fs::create_dir_all(s.join(dir)).map_err(|e| Fail::new(1, format!("{}: {e}", s.join(dir).display())))?;
    }
    let _ = std::fs::set_permissions(s.join("run"), std::os::unix::fs::PermissionsExt::from_mode(0o700));
    let wait = sh::var("SMOKE_GUI_WAIT").unwrap_or_else(|| "60".to_string());

    let (mut state, mut why) = start_once(&exe, &appdir, s, &icds, &wait);
    let mut tries = "first start";
    // The toolkit's X11 code has been seen to hang at start on Xvfb before its window shows (the process alive, nothing
    // on stderr): that one case gets a second start, and the output says so.
    if outcome(&state, &why) == "retry" {
        println!("smoke-gui: no window on the first start (the process was alive, stderr clean): starting it once more");
        (state, why) = start_once(&exe, &appdir, s, &icds, &wait);
        tries = "second start, the first showed no window";
    }
    if outcome(&state, &why) != "pass" {
        let state = if state.is_empty() { "no window" } else { &state };
        let why = if why.is_empty() { "stderr clean" } else { &why };
        eprintln!("smoke-gui: FAILED on the {tries} ({state}; {why})");
        eprintln!("  the GUI's stderr:");
        indented(&s.join("err"));
        eprintln!("  the session's (Xvfb, D-Bus):");
        indented(&s.join("session"));
        return Ok(ExitCode::FAILURE);
    }
    println!("smoke-gui: window mapped, main loop running, stderr clean ({tries})");
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::finish(run(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts_and_outcomes() {
        assert_eq!(verdict("neural-forge: INFO [x] ok\n"), "");
        assert_eq!(verdict("a\nneural-forge: WARN [x] w\n"), "neural-forge: WARN [x] w\nerrors on stderr");
        assert_eq!(outcome("mapped running", ""), "pass");
        assert_eq!(outcome("running", ""), "retry");
        assert_eq!(outcome("mapped", ""), "fail");
        assert_eq!(outcome("", ""), "fail");
    }
}
