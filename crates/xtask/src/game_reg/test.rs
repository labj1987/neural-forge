//! `cargo xtask test-game-reg`: tests game-reg with stand-ins for the CLI, steam and journalctl, under a scratch HOME
//! and XDG_RUNTIME_DIR, with waits that take no time (it never starts a game):
//! - a game already running (a process matching the pattern) is refused and left running, and `enabled` is not set;
//! - a run puts the layer's `enabled` back as it was, and quits only the game it started;
//! - Steam gets the session's XAUTHORITY, else GNOME's Xwayland's, else ~/.Xauthority, else none.
//!
//! It replaced scripts/test_game_reg.sh, case for case. One thing is stricter: the steam stand-in is a bash text, so the
//! "game" it starts under the pattern's name really runs (`exec -a` is bash's; under /bin/sh, where that is dash, the
//! stand-in started nothing and the check that the run quits its game had nothing to find).

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use regex::Regex;

use super::Env;
use crate::proc;
use crate::sh::{self, Fail, Res};

fn fail(message: String) -> Fail {
    Fail::new(1, format!("test_game_reg: {message}"))
}

fn text(path: &Path) -> String {
    sh::cat(path)
}

fn stub(path: &Path, interpreter: &str, body: &str) -> Res<()> {
    std::fs::write(path, format!("#!{interpreter}\n{body}\n")).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).map_err(|e| fail(format!("{}: {e}", path.display())))
}

/// Ends the stand-in games however the test ends.
struct Games(Regex);

impl Drop for Games {
    fn drop(&mut self) {
        for pid in proc::matching(&self.0) {
            sh::kill(pid, libc::SIGTERM);
        }
    }
}

pub fn run() -> Res<ExitCode> {
    let dir = sh::mktemp()?;
    let tmp = dir.path().to_string_lossy().into_owned();
    let at = |name: &str| PathBuf::from(format!("{tmp}/{name}"));
    for sub in ["bin", "home", "run"] {
        std::fs::create_dir_all(at(sub)).map_err(|e| fail(format!("{}: {e}", at(sub).display())))?;
    }
    // The CLI: `enabled` kept in $tmp/enabled, every call logged.
    stub(
        &at("bin/cli"),
        "/bin/sh",
        &format!(
            "echo \"$*\" >> '{tmp}/cli.log'\ncase \"$2\" in\nstatus) echo \"enabled=$(cat '{tmp}/enabled')\";;\nset) echo \"$4\" > '{tmp}/enabled';;\ncapture) exit 1;;\nesac"
        ),
    )?;
    // steam starts the "game" (a process whose command line holds the pattern) and says which X authority it got.
    // The pattern is this test's own: two tests at once (the task and `cargo test`) must not see each other's games.
    let pattern = format!("nf-game-reg-test-{}-{}", std::process::id(), dir.path().file_name().unwrap_or_default().to_string_lossy().replace('.', "-"));
    stub(&at("bin/steam"), "/bin/bash", &format!("echo \"${{XAUTHORITY-unset}}\" > '{tmp}/xauth'; (exec -a {pattern}-run /bin/sleep 300) >/dev/null 2>&1 &"))?;
    stub(&at("bin/journalctl"), "/bin/sh", "exit 0")?;
    let _games = Games(Regex::new(&pattern).map_err(|e| fail(e.to_string()))?);
    let started_game = Regex::new(&format!("{pattern}-run")).map_err(|e| fail(e.to_string()))?;
    let path = format!("{tmp}/bin:{}", std::env::var("PATH").unwrap_or_default());
    let run = |xauthority: Option<&str>| {
        let env = Env {
            home: format!("{tmp}/home"),
            cli: at("bin/cli"),
            runtime_dir: at("run"),
            xauthority: xauthority.map(String::from),
            display: ":0".to_string(),
            vars: vec![("HOME".into(), format!("{tmp}/home")), ("XDG_RUNTIME_DIR".into(), format!("{tmp}/run")), ("PATH".into(), path.clone())],
            sleep: |_| Ok(()),
        };
        let args: Vec<String> = ["t1", "123", &pattern, "0"].iter().map(|s| s.to_string()).collect();
        super::run(&args, &env, &mut Vec::new(), &mut Vec::new())
    };

    // A run that must pass; the "game" it started is given a moment to be gone before the next one looks for it.
    let passes = |case: &str, xauthority: Option<&str>| -> Res<()> {
        let _ = run(xauthority).map_err(|e| fail(format!("the run failed{case}: {}", e.message.unwrap_or_default())))?;
        for _ in 0..10 {
            if proc::matching(&started_game).is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        if proc::matching(&started_game).is_empty() {
            Ok(())
        } else {
            Err(fail("the game the run started was not quit".into()))
        }
    };

    // A game already running: refused, still running, nothing set. (Started through a shell that ends at once, so it is
    // not this process's own child: a run skips what it started itself.)
    let mut shell = Command::new("/bin/bash");
    shell.args(["-c", "(exec -a \"$0\" /bin/sleep 300) >/dev/null 2>&1 &"]).arg(format!("{pattern}-mine")).stdin(Stdio::null());
    shell.status().map_err(|e| sh::not_started(&shell, e))?;
    let mine = Regex::new(&format!("{pattern}-mine")).map_err(|e| fail(e.to_string()))?;
    let running = || proc::matching(&mine);
    for _ in 0..50 {
        if !running().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let game = *running().first().ok_or_else(|| fail("the stand-in game did not start".into()))?;
    std::fs::write(at("enabled"), "0\n").map_err(|e| fail(e.to_string()))?;
    match run(None) {
        Ok(_) => return Err(fail("a run started while the game was running".into())),
        Err(refusal) if !refusal.message.as_deref().unwrap_or("").contains("already running") => {
            return Err(fail(format!("no reason given: {}", refusal.message.unwrap_or_default())));
        }
        Err(_) => {}
    }
    if !running().contains(&game) {
        return Err(fail("the game that was already running was closed".into()));
    }
    if at("cli.log").exists() {
        return Err(fail(format!("the CLI was called: {}", text(&at("cli.log")))));
    }
    sh::kill(game, libc::SIGTERM);
    for _ in 0..50 {
        if running().is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // A run: enabled back to 0, the game it started quit, XAUTHORITY passed on.
    let session_auth = format!("{tmp}/session-auth");
    passes("", Some(&session_auth))?;
    if text(&at("enabled")) != "0" {
        return Err(fail(format!("enabled left at {}", text(&at("enabled")))));
    }
    if !text(&at("cli.log")).lines().any(|l| l == "shmctl set enabled 1") {
        return Err(fail("enabled was not set for the run".into()));
    }
    if text(&at("xauth")) != session_auth {
        return Err(fail(format!("steam got {}, not XAUTHORITY", text(&at("xauth")))));
    }

    // No XAUTHORITY: GNOME's Xwayland's newest, else ~/.Xauthority, else none (and no failure).
    let touch = |name: &str| std::fs::write(at(name), "").map_err(|e| fail(format!("{}: {e}", at(name).display())));
    touch("run/.mutter-Xwaylandauth.A")?;
    std::thread::sleep(std::time::Duration::from_millis(10));
    touch("run/.mutter-Xwaylandauth.B")?;
    touch("home/.Xauthority")?;
    passes(" with GNOME's X authority", None)?;
    if text(&at("xauth")) != format!("{tmp}/run/.mutter-Xwaylandauth.B") {
        return Err(fail(format!("steam got {}, not the newest mutter file", text(&at("xauth")))));
    }
    for name in ["run/.mutter-Xwaylandauth.A", "run/.mutter-Xwaylandauth.B"] {
        let _ = std::fs::remove_file(at(name));
    }
    passes(" with ~/.Xauthority", None)?;
    if text(&at("xauth")) != format!("{tmp}/home/.Xauthority") {
        return Err(fail(format!("steam got {}, not ~/.Xauthority", text(&at("xauth")))));
    }
    let _ = std::fs::remove_file(at("home/.Xauthority"));
    passes(" without any X authority", None)?;
    if text(&at("xauth")) != "unset" {
        return Err(fail(format!("steam got {} with no X authority anywhere", text(&at("xauth")))));
    }
    println!("test_game_reg: ok");
    Ok(ExitCode::SUCCESS)
}

pub fn main(_args: &[String]) -> ExitCode {
    sh::finish(run())
}
