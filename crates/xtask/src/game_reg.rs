//! cargo xtask game-reg <label> <appid> <exe-pattern> <boot-s> <steps...>: one run of a Steam game with the layer
//! (its own launch options): launch, wait <boot-s>, send controller steps (a step "W<n>" waits n s),
//! measure 60 s in play, capture the presented frame, quit. Prints holds, presents, timeouts, Xid.
//! Refuses to start while a process matching <exe-pattern> already runs (a game someone is playing is never closed),
//! and puts the layer's `enabled` back as it was. NF_REG_CLI names the CLI (cargo xtask test-game-reg uses a stand-in).
//!
//! It replaced scripts/game-reg.sh: the same files in ~/nf-spike/runs/<label>, output and exit codes.

pub mod test;

use std::io::Write;
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use regex::Regex;

use crate::proc;
use crate::sh::{self, Fail, Res};

/// Controller steps go to the pad helper (~/nf-spike/pad/nf-pad.py) through its FIFO.
const PAD: &str = "/tmp/nf-pad.fifo";

/// What a run takes from its environment (the test gives it a scratch one, and waits that take no time).
pub struct Env {
    pub home: String,
    /// NF_REG_CLI, else the installed neural-forge-cli.
    pub cli: PathBuf,
    /// XDG_RUNTIME_DIR, else /run/user/<uid>.
    pub runtime_dir: PathBuf,
    /// XAUTHORITY as it is set (None: not set).
    pub xauthority: Option<String>,
    /// DISPLAY, else :0.
    pub display: String,
    /// Variables every program the run starts gets on top of this process's.
    pub vars: Vec<(String, String)>,
    pub sleep: fn(f64) -> Res<()>,
}

impl Env {
    fn real() -> Env {
        let home = std::env::var("HOME").unwrap_or_default();
        // SAFETY: getuid has no failure.
        let uid = unsafe { libc::getuid() };
        Env {
            cli: sh::var("NF_REG_CLI").map_or_else(|| PathBuf::from(format!("{home}/.local/share/neural-forge/bin/neural-forge-cli")), PathBuf::from),
            runtime_dir: sh::var("XDG_RUNTIME_DIR").map_or_else(|| PathBuf::from(format!("/run/user/{uid}")), PathBuf::from),
            xauthority: std::env::var("XAUTHORITY").ok(),
            display: sh::var("DISPLAY").unwrap_or_else(|| ":0".to_string()),
            vars: Vec::new(),
            sleep: sh::sleep,
            home,
        }
    }

    fn command(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut cmd = Command::new(program);
        cmd.envs(self.vars.iter().map(|(k, v)| (k, v)));
        cmd
    }

    fn cli(&self) -> Command {
        self.command(&self.cli)
    }
}

/// The processes matching the pattern that are not this run's: this process's own command line holds the pattern, as
/// do those of the processes above it (cargo), so this process, everything above it and anything it started itself are
/// skipped.
fn game_pids(pattern: &str) -> Vec<i32> {
    let regex = match Regex::new(pattern) {
        Ok(regex) => regex,
        Err(e) => {
            eprintln!("{pattern}: not a pattern: {e}");
            return Vec::new();
        }
    };
    let me = std::process::id() as i32;
    let mut mine = Vec::new();
    let mut p = me;
    while p > 1 {
        mine.push(p);
        match proc::ppid(p) {
            Some(parent) => p = parent,
            None => break,
        }
    }
    proc::matching(&regex).into_iter().filter(|p| !mine.contains(p) && proc::ppid(*p).is_some_and(|parent| parent != me)).collect()
}

fn quit_game(pattern: &str) {
    for pid in game_pids(pattern) {
        sh::kill(pid, libc::SIGTERM);
    }
}

/// Ends the game this run started however the run ends (a failed step included), and puts `enabled` back.
struct OnExit<'a> {
    env: &'a Env,
    pattern: &'a str,
    was_enabled: String,
}

impl Drop for OnExit<'_> {
    fn drop(&mut self) {
        quit_game(self.pattern);
        if !self.was_enabled.is_empty() && self.was_enabled != "1" {
            let _ = self.env.cli().args(["shmctl", "set", "enabled", &self.was_enabled]).stdout(Stdio::null()).status();
        }
    }
}

fn is_fifo(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.file_type().is_fifo())
}

/// Writes the step to the pad helper's FIFO. Never written unless it is a FIFO: a write to a missing one would create a
/// plain file, every step would "succeed" at once with no input reaching the game, and the run would measure a menu.
/// False when it cannot be delivered (no FIFO, or no reader within 5 s).
fn deliver(step: &str, fifo: &Path) -> bool {
    if !is_fifo(fifo) {
        return false;
    }
    let end = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match std::fs::OpenOptions::new().write(true).custom_flags(libc::O_NONBLOCK).open(fifo) {
            Ok(mut pad) => return pad.write_all(format!("{step}\n").as_bytes()).is_ok(),
            // No reader has the FIFO open yet.
            Err(e) if e.raw_os_error() == Some(libc::ENXIO) && std::time::Instant::now() < end => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => return false,
        }
    }
}

/// The X authority Steam is reached with: the session's own (XAUTHORITY), else GNOME's Xwayland's newest, else
/// ~/.Xauthority; none on a display that needs none.
fn x_authority(env: &Env) -> Option<String> {
    if let Some(own) = env.xauthority.clone().filter(|x| !x.is_empty()) {
        return Some(own);
    }
    let mut mutter: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&env.runtime_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with(".mutter-Xwaylandauth."))
                .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
                .collect()
        })
        .unwrap_or_default();
    mutter.sort();
    if let Some((_, newest)) = mutter.pop() {
        return Some(newest.to_string_lossy().into_owned());
    }
    let own = format!("{}/.Xauthority", env.home);
    Path::new(&own).is_file().then_some(own)
}

/// The files under `dir` named `*-original.png` that were written after `marker`, in name order.
fn captures_since(dir: &Path, marker: std::time::SystemTime, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        if meta.is_dir() {
            captures_since(&path, marker, found);
        } else if entry.file_name().to_string_lossy().ends_with("-original.png") && meta.modified().is_ok_and(|m| m > marker) {
            found.push(path);
        }
    }
    found.sort();
}

fn journal(env: &Env, args: &[&str]) -> Vec<u8> {
    env.command("journalctl").args(args).stdin(Stdio::null()).stderr(Stdio::inherit()).output().map(|o| o.stdout).unwrap_or_default()
}

/// The run's result lines: the layer's last hold counters and the presented rate over the measured window (the layer's
/// lines in Steam's journal), and the fence-wait timeouts of the whole launch.
fn summary(label: &str, window: &str, launch_log: &str, xid: usize) -> String {
    let mut out = format!("== {label}\n");
    if let Some(line) = window.lines().rfind(|l| l.contains("mode=model")) {
        let holds = Regex::new(r"^.*(holds=[0-9]+).*(misses=[0-9]+ \(total [0-9]+\)).*(holds_per_s=[0-9.]+).*$").unwrap();
        match holds.captures(line) {
            Some(c) => out.push_str(&format!("{} {} {}\n", &c[1], &c[2], &c[3])),
            None => out.push_str(&format!("{line}\n")),
        }
    }
    let present = Regex::new(r"^.*\] ([0-9.]+) fps \(([0-9.]+)/s.*$").unwrap();
    let (mut fps, mut composited, mut n) = (0.0, 0.0, 0u32);
    for line in window.lines().filter(|l| l.contains("[present]")) {
        let (first, second) = match present.captures(line) {
            Some(c) => (c[1].to_string(), c[2].to_string()),
            None => {
                let mut fields = line.split_whitespace();
                (fields.next().unwrap_or("").to_string(), fields.next().unwrap_or("").to_string())
            }
        };
        fps += sh::leading_number(&first);
        composited += sh::leading_number(&second);
        n += 1;
    }
    if n > 0 {
        out.push_str(&format!("presents {:.1}/s composited {:.1}/s over {n} windows\n", fps / f64::from(n), composited / f64::from(n)));
    }
    let timeouts = launch_log.lines().filter(|l| l.contains("fence wait timed out")).count();
    out.push_str(&format!("timeouts {timeouts} Xid {xid}\n"));
    out
}

/// One run. What it prints goes to `out`, its warnings to `err`; an error is returned.
pub fn run(args: &[String], env: &Env, out: &mut dyn Write, err: &mut dyn Write) -> Res<ExitCode> {
    let [label, appid, pattern, boot, steps @ ..] = args else {
        return Err(Fail::new(2, "usage: cargo xtask game-reg <label> <appid> <exe-pattern> <boot-s> <steps...>"));
    };
    if label.is_empty() || label == "." || label == ".." || label.contains('/') {
        return Err(Fail::new(2, "error: label must be a plain name (no '/', not empty)"));
    }
    let dir = PathBuf::from(format!("{}/nf-spike/runs/{label}", env.home));
    std::fs::create_dir_all(&dir).map_err(|e| Fail::new(1, format!("{}: {e}", dir.display())))?;
    let pad = Path::new(PAD);
    // A step that cannot be delivered ends the run; without the FIFO it does not start.
    if steps.iter().any(|s| !s.starts_with('W')) && !is_fifo(pad) {
        return Err(Fail::new(
            1,
            if pad.exists() {
                format!("error: {PAD} is not a FIFO (a stray file?); remove it and start the pad helper")
            } else {
                format!("error: {PAD} does not exist: start the pad helper (~/nf-spike/pad/nf-pad.py) first")
            },
        ));
    }
    let running = game_pids(pattern);
    if !running.is_empty() {
        let pids: String = running.iter().map(|p| format!("{p} ")).collect();
        return Err(Fail::new(1, format!("error: {pattern} is already running (pid {pids}); not launching it again or closing it")));
    }
    let start = sh::now();
    let write = |name: &str, bytes: &[u8]| std::fs::write(dir.join(name), bytes).map_err(|e| Fail::new(1, format!("{}: {e}", dir.join(name).display())));
    write("start", format!("{start}\n").as_bytes())?;

    // The layer's `enabled` as it was, put back however this run ends.
    let status = env.cli().args(["shmctl", "status"]).stdin(Stdio::null()).stderr(Stdio::null()).output().map(|o| o.stdout).unwrap_or_default();
    let enabled = Regex::new(r"^enabled=([0-9]*)").unwrap();
    let was_enabled = String::from_utf8_lossy(&status).lines().find_map(|l| enabled.captures(l).map(|c| c[1].to_string())).unwrap_or_default();
    let _on_exit = OnExit { env, pattern, was_enabled };
    sh::check(env.cli().args(["shmctl", "set", "enabled", "1"]).stdout(Stdio::null()))?;

    let mut steam = env.command("steam");
    steam.args(["-applaunch", appid]).env("DISPLAY", &env.display).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    match (x_authority(env), &env.xauthority) {
        (Some(authority), _) => steam.env("XAUTHORITY", authority),
        (None, Some(as_set)) => steam.env("XAUTHORITY", as_set),
        (None, None) => steam.env_remove("XAUTHORITY"),
    };
    let _ = sh::timeout(10, &mut steam);
    let seconds = |text: &str| text.parse::<f64>().map_err(|_| Fail::new(1, format!("sleep: invalid time interval '{text}'")));
    (env.sleep)(seconds(boot)?)?;
    for step in steps {
        match step.strip_prefix('W') {
            Some(wait) => (env.sleep)(seconds(wait)?)?,
            None => {
                if !deliver(step, pad) {
                    return Err(Fail::new(1, format!("error: controller step {step} was not delivered (is the pad helper running?)")));
                }
                (env.sleep)(1.0)?;
            }
        }
    }
    let measured_from = sh::now();
    (env.sleep)(60.0)?;
    // This run's capture only: a file older than the marker is an earlier run's, never copied as this one's.
    let marker = dir.join(".capture-marker");
    write(".capture-marker", b"")?;
    let marked = std::fs::metadata(&marker).and_then(|m| m.modified()).map_err(|e| Fail::new(1, format!("{}: {e}", marker.display())))?;
    let capture_log = std::fs::File::create(dir.join("capture.txt")).map_err(|e| Fail::new(1, format!("{}: {e}", dir.join("capture.txt").display())))?;
    let both = capture_log.try_clone().map_err(|e| Fail::new(1, format!("capture.txt: {e}")))?;
    if sh::timeout(40, env.cli().args(["shmctl", "capture"]).stdin(Stdio::null()).stdout(capture_log).stderr(both)).is_ok_and(|code| code == 0) {
        let mut shots = Vec::new();
        captures_since(Path::new(&format!("{}/.local/share/neural-forge/captures", env.home)), marked, &mut shots);
        match shots.first() {
            Some(shot) => {
                std::fs::copy(shot, dir.join("end.png")).map_err(|e| Fail::new(1, format!("{}: {e}", shot.display())))?;
            }
            None => {
                let _ = writeln!(err, "warning: the capture wrote no original frame");
            }
        }
    } else {
        let text = std::fs::read(dir.join("capture.txt")).unwrap_or_default();
        let _ = writeln!(err, "warning: capture failed: {}", String::from_utf8_lossy(&text).lines().next_back().unwrap_or(""));
    }
    let _ = std::fs::remove_file(&marker);
    let launch_log = journal(env, &["--user", "-u", "steam-relaunch", "--since", &start, "-o", "cat"]);
    write("launch.log", &launch_log)?;

    quit_game(pattern);
    (env.sleep)(8.0)?;
    let window = journal(env, &["--user", "-u", "steam-relaunch", "--since", &measured_from, "--until", &sh::now(), "-o", "cat"]);
    let kernel = journal(env, &["-k", "--since", &start, "--no-pager"]);
    let xid = String::from_utf8_lossy(&kernel).lines().filter(|l| l.contains("Xid")).count();
    let _ = out.write_all(summary(label, &String::from_utf8_lossy(&window), &String::from_utf8_lossy(&launch_log), xid).as_bytes());
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::trap_signals();
    sh::finish(run(args, &Env::real(), &mut std::io::stdout(), &mut std::io::stderr()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_reads_the_layer_s_lines() {
        let window = "x [sync] mode=model holds=10 a misses=1 (total 2) b holds_per_s=3.5 c\n\
                      y [sync] mode=model holds=20 a misses=2 (total 4) b holds_per_s=58.9 c\n\
                      [neural-forge-layer] [present] 118.0 fps (59.0/s composited)\n\
                      [neural-forge-layer] [present] 120.0 fps (60.0/s composited)\n";
        assert_eq!(
            summary("t1", window, "a\nfence wait timed out\n", 3),
            "== t1\nholds=20 misses=2 (total 4) holds_per_s=58.9\npresents 119.0/s composited 59.5/s over 2 windows\ntimeouts 1 Xid 3\n"
        );
        assert_eq!(summary("t2", "", "", 0), "== t2\ntimeouts 0 Xid 0\n");
    }

    #[test]
    fn a_label_must_be_a_plain_name() {
        let env = Env::real();
        for bad in ["", ".", "..", "a/b"] {
            let args: Vec<String> = [bad, "1", "x", "0"].iter().map(|s| s.to_string()).collect();
            let fail = run(&args, &env, &mut Vec::new(), &mut Vec::new()).unwrap_err();
            assert_eq!(fail.code, 2, "{bad}");
        }
        assert_eq!(run(&["only".to_string()], &env, &mut Vec::new(), &mut Vec::new()).unwrap_err().code, 2);
    }

    /// The whole test, so `cargo test` runs it too.
    #[test]
    fn game_reg() {
        if let Err(fail) = test::run() {
            panic!("{}", fail.message.unwrap_or_default());
        }
    }
}
