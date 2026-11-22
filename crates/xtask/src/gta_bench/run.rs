//! The benchmark runner itself (gta_bench/usage.txt has what it does and its options): it replaced scripts/gta-bench.sh,
//! step for step, with the same files in the run's directory, the same lines on stdout and the same exit codes (0 for a
//! result, 1 for none or a setting that could not be put back, 2 for a usage error).
//!
//! What the script asked other programs for and this reads itself: the processes (/proc, not pgrep and ps), the
//! remote-desktop connection (/proc/net/tcp, not ss) and the crash dumps, MangoHud's log and Steam's Proton choice (the
//! functions of the parent module, which the script ran as `xtask gta-bench dumps|fg|proton`).

use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};

use regex::Regex;

use crate::proc;
use crate::sh::{self, Fail, Res};

/// The runner's lines: written so that an output that has gone away cannot end the run before the settings are put
/// back (sh::say).
macro_rules! say {
    ($($arg:tt)*) => { sh::say(&format!($($arg)*)) };
}
macro_rules! warn {
    ($($arg:tt)*) => { sh::warn(&format!($($arg)*)) };
}

const APP: &str = "3240220";
const GAME: &str = "GTA5_Enhanced.e";
const NEURAL_FORGE: &str = "VK_LAYER_neuralforge_neural";

#[derive(Debug)]
pub struct Args {
    sets: Vec<String>,
    tries: u64,
    label: String,
    /// The Vulkan layers above MangoHud.
    mid: String,
    /// The game's extra environment, in the order given.
    env: Vec<(String, String)>,
}

fn usage() -> Fail {
    Fail::new(2, include_str!("usage.txt").trim_end_matches('\n'))
}

pub fn parse(args: &[String]) -> Res<Args> {
    let (mut sets, mut tries, mut rest) = (Vec::new(), "3".to_string(), args);
    loop {
        match rest {
            [flag, value, more @ ..] if flag == "--set" => {
                sets.push(value.clone());
                rest = more;
            }
            [flag, value, more @ ..] if flag == "--tries" => {
                tries = value.clone();
                rest = more;
            }
            _ => break,
        }
    }
    let [label, mid, env @ ..] = rest else { return Err(usage()) };
    // The label names a directory that is deleted and recreated below: never empty, never a path.
    if label.is_empty() || label == "." || label == ".." || label.contains('/') {
        return Err(Fail::new(2, "error: label must be a plain name (not empty, no '/')"));
    }
    if tries.is_empty() || tries == "0" || !tries.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Fail::new(2, "error: --tries takes a count of 1 or more"));
    }
    let tries = tries.parse().unwrap_or(u64::MAX);
    let env = env
        .iter()
        .map(|pair| match pair.split_once('=') {
            Some((name, value)) if !name.is_empty() => Ok((name.to_string(), value.to_string())),
            _ => Err(Fail::new(2, format!("error: {pair} is not NAME=VALUE (the game's environment)"))),
        })
        .collect::<Res<Vec<_>>>()?;
    Ok(Args { sets, tries, label: label.clone(), mid: mid.clone(), env })
}

/// The rig's paths and what one run needs of them.
struct Rig {
    /// The Steam client.
    client: String,
    /// The library holding GTA.
    library: String,
    proton: String,
    game: String,
    runtime: String,
    prefix: String,
    /// Where GTA writes its Benchmark and FrameTimes files.
    docs: String,
    crash_logs: String,
    cli: String,
    stuck_s: i64,
    uid: u32,
    /// The run's directory.
    out: String,
    mid: String,
    /// Frame generation is on in GTA's settings.xml, and the refresh rate there.
    fg_on: bool,
    refresh: i64,
    env: Vec<(String, String)>,
}

impl Rig {
    /// `neural-forge-cli shmctl ...` on the session's channel, given 30 s.
    fn shmctl(&self, args: &[&str]) -> Command {
        let mut cli = Command::new(&self.cli);
        cli.arg("shmctl").args(args).stdin(Stdio::null());
        cli.env("NEURAL_FORGE_SHM", format!("/tmp/neural-forge-{}/shm.bin", self.uid)).env("NEURAL_FORGE_UID", self.uid.to_string());
        cli
    }

    /// The value `shmctl status` shows for a setting (empty when it shows none).
    fn status_of(&self, key: &str) -> String {
        let (_, status) = sh::timeout_output(30, &mut self.shmctl(&["status"])).unwrap_or_default();
        let prefix = format!("{key}=");
        status.lines().filter_map(|l| l.strip_prefix(&prefix)).collect::<Vec<_>>().join("\n")
    }

    fn file(&self, name: &str) -> PathBuf {
        Path::new(&self.out).join(name)
    }

    fn write(&self, name: &str, text: &str) -> Res<()> {
        std::fs::write(self.file(name), text).map_err(|e| Fail::new(1, format!("{}: {e}", self.file(name).display())))
    }

    /// A file of the run's evidence: one that cannot be written is reported and the run goes on (the script went on
    /// past a failed redirect too), so a launch is always ended.
    fn put(&self, name: &str, text: &str) {
        if let Err(fail) = self.write(name, text) {
            warn!("{}", fail.message.unwrap_or_default());
        }
    }

    fn append(&self, name: &str, text: &str) {
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(self.file(name)) {
            let _ = file.write_all(text.as_bytes());
        }
    }

    /// Prints the line and adds it to a file of the run (`| tee -a`).
    fn tee(&self, name: &str, line: &str) {
        say!("{line}");
        self.append(name, &format!("{line}\n"));
    }

    /// GTA's newest Benchmark file (by name: the names carry the time), empty when there is none.
    fn newest_benchmark(&self) -> String {
        let mut found: Vec<String> = names(&self.docs).into_iter().filter(|n| n.starts_with("Benchmark-")).collect();
        found.sort();
        found.pop().map_or_else(String::new, |n| format!("{}/{n}", self.docs))
    }

    /// MangoHud's newest per-frame log of this launch.
    fn newest_csv(&self) -> Option<PathBuf> {
        let log = Regex::new(r"^GTA5_Enhanced_.*[0-9]\.csv$").unwrap();
        let mut found: Vec<(std::cmp::Reverse<(i64, i64)>, String)> = names(&self.out)
            .into_iter()
            .filter(|n| log.is_match(n))
            .filter_map(|n| std::fs::metadata(self.file(&n)).ok().map(|m| (std::cmp::Reverse((m.mtime(), m.mtime_nsec())), n)))
            .collect();
        found.sort();
        found.first().map(|(_, n)| self.file(n))
    }
}

fn names(dir: &str) -> Vec<String> {
    std::fs::read_dir(dir).map(|entries| entries.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default()
}

fn pid_list(pids: &[i32]) -> String {
    pids.iter().map(|p| format!("{p} ")).collect()
}

/// An established TCP connection to this local port (`ss -tn state established '( sport = :PORT )'`).
fn connected_to(port: u16) -> bool {
    let suffix = format!(":{port:04X}");
    ["/proc/net/tcp", "/proc/net/tcp6"].iter().any(|table| {
        std::fs::read_to_string(table).is_ok_and(|text| {
            text.lines().skip(1).any(|line| {
                let fields: Vec<&str> = line.split_whitespace().collect();
                fields.get(1).is_some_and(|local| local.ends_with(&suffix)) && fields.get(3) == Some(&"01")
            })
        })
    })
}

/// When the process started (Unix time), None when it is gone.
fn started_at(pid: i32) -> Option<i64> {
    let ticks: i64 = proc::stat(&format!("/proc/{pid}/stat"))?.get(19)?.parse().ok()?;
    let boot: i64 = std::fs::read_to_string("/proc/stat").ok()?.lines().find_map(|l| l.strip_prefix("btime "))?.trim().parse().ok()?;
    // SAFETY: sysconf has no failure that matters here (a non-positive answer is replaced).
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1);
    Some(boot + ticks / hz)
}

fn uid_of(pid: i32) -> Option<u32> {
    std::fs::metadata(format!("/proc/{pid}")).ok().map(|m| m.uid())
}

/// Every thread of the game: name, state, kernel wait channel and CPU ticks, for a game that stopped
/// presenting (readable without privileges).
fn thread_states(pid: i32) -> String {
    let mut tasks = names(&format!("/proc/{pid}/task"));
    tasks.sort();
    let mut out = String::new();
    for tid in tasks {
        let dir = format!("/proc/{pid}/task/{tid}");
        let read = |name: &str| std::fs::read_to_string(format!("{dir}/{name}")).unwrap_or_default().trim_end_matches('\n').to_string();
        let stat = proc::stat(&format!("{dir}/stat")).unwrap_or_default();
        let field = |i: usize| stat.get(i).cloned().unwrap_or_default();
        out.push_str(&format!("{tid} {} state={} wchan={} utime={} stime={}\n", read("comm"), field(0), read("wchan"), field(11), field(12)));
    }
    out
}

/// The median of the displayed rate over a log's last 300 frames, as the script's sort and awk gave it.
fn end_fps(csv: &Path) -> String {
    let text = String::from_utf8_lossy(&std::fs::read(csv).unwrap_or_default()).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    let mut first: Vec<&str> = lines[lines.len().saturating_sub(300)..].iter().map(|l| l.split(',').next().unwrap_or("")).collect();
    first.sort_by(|a, b| sh::leading_number(a).total_cmp(&sh::leading_number(b)).then_with(|| a.cmp(b)));
    first.get(first.len() / 2).and_then(|l| l.split_whitespace().next()).unwrap_or("").to_string()
}

/// What one launch left running: the launch itself and nvidia-smi's sampling.
struct Launch {
    child: Child,
    /// The launch's process group (it is its own session).
    group: i32,
    smi: Option<Child>,
    start: i64,
}

impl Drop for Launch {
    /// nvidia-smi's sampling ends with the launch's record, however the launch is left.
    fn drop(&mut self) {
        if let Some(mut smi) = self.smi.take() {
            sh::kill(smi.id() as i32, libc::SIGTERM);
            let _ = smi.wait();
        }
    }
}

impl Rig {
    /// Kills what one launch left: the game, the launch's process group, and the Wine processes of
    /// GTA's prefix started since the launch (winedevice.exe spins a core each when left behind).
    fn end_launch(&self, launch: &mut Launch) -> Res<()> {
        if let Some(mut smi) = launch.smi.take() {
            sh::kill(smi.id() as i32, libc::SIGTERM);
            let _ = smi.wait();
        }
        for pid in proc::named(GAME) {
            sh::kill(pid, libc::SIGTERM);
        }
        sh::sleep(15.0)?;
        sh::kill(-launch.group, libc::SIGTERM);
        sh::sleep(5.0)?;
        sh::kill(-launch.group, libc::SIGKILL);
        let _ = launch.child.try_wait();
        let wine = Regex::new(r"\.exe|wineserver|wine64|wine-preloader").unwrap();
        let marker = format!("STEAM_COMPAT_DATA_PATH={}", self.prefix);
        for pid in proc::matching(&wine) {
            if uid_of(pid) != Some(self.uid) {
                continue;
            }
            let environ = std::fs::read(format!("/proc/{pid}/environ")).unwrap_or_default();
            if !environ.windows(marker.len()).any(|w| w == marker.as_bytes()) {
                continue;
            }
            if started_at(pid).is_some_and(|at| at >= launch.start) {
                sh::kill(pid, libc::SIGKILL);
            }
        }
        Ok(())
    }

    fn spawn(&self) -> Res<Launch> {
        let uid = self.uid;
        let run_dir = format!("/run/user/{uid}");
        // GNOME's Xwayland's newest authority file (empty when there is none).
        let mut authorities: Vec<((i64, i64), String)> = names(&run_dir)
            .into_iter()
            .filter(|n| n.starts_with(".mutter-Xwaylandauth."))
            .filter_map(|n| std::fs::metadata(format!("{run_dir}/{n}")).ok().map(|m| ((m.mtime(), m.mtime_nsec()), n)))
            .collect();
        authorities.sort();
        let authority = authorities.pop().map_or_else(String::new, |(_, n)| format!("{run_dir}/{n}"));
        if authority.is_empty() {
            warn!("no {run_dir}/.mutter-Xwaylandauth.* (GNOME's Xwayland authority)");
        }
        let layers =
            if self.mid.is_empty() { "VK_LAYER_MANGOHUD_overlay_x86_64".to_string() } else { format!("{}:VK_LAYER_MANGOHUD_overlay_x86_64", self.mid) };
        let (client, library) = (&self.client, &self.library);
        let log = std::fs::File::create(self.file("launch.log")).map_err(|e| Fail::new(1, format!("{}: {e}", self.file("launch.log").display())))?;
        let both = log.try_clone().map_err(|e| Fail::new(1, format!("launch.log: {e}")))?;
        let mut game = Command::new(format!("{}/_v2-entry-point", self.runtime));
        game.env_clear();
        game.env("HOME", std::env::var_os("HOME").unwrap_or_default()).env("USER", std::env::var_os("USER").unwrap_or_default());
        game.env("PATH", "/usr/bin:/bin").env("LANG", sh::var("LANG").unwrap_or_else(|| "C.UTF-8".to_string()));
        game.env("DISPLAY", ":0").env("WAYLAND_DISPLAY", "wayland-0").env("XDG_RUNTIME_DIR", &run_dir);
        game.env("DBUS_SESSION_BUS_ADDRESS", format!("unix:path={run_dir}/bus")).env("XAUTHORITY", authority);
        game.env("SteamAppId", APP).env("SteamGameId", APP).env("SteamOverlayGameId", APP).env("STEAM_COMPAT_APP_ID", APP);
        game.env("STEAM_COMPAT_CLIENT_INSTALL_PATH", client).env("STEAM_COMPAT_DATA_PATH", &self.prefix);
        game.env("STEAM_COMPAT_INSTALL_PATH", &self.game).env("STEAM_COMPAT_LIBRARY_PATHS", format!("{library}/steamapps:{client}/steamapps"));
        game.env("STEAM_COMPAT_TOOL_PATHS", format!("{}:{}", self.proton, self.runtime));
        game.env("STEAM_COMPAT_SHADER_PATH", format!("{library}/steamapps/shadercache/{APP}"));
        game.env("VK_INSTANCE_LAYERS", layers).env("VK_LOADER_DEBUG", "layer");
        game.env("MANGOHUD_CONFIG", format!("output_folder={},autostart_log=1,log_duration=3600,log_interval=0", self.out));
        game.envs(self.env.iter().map(|(k, v)| (k, v)));
        game.args(["--verb=waitforexitandrun", "--"]).arg(format!("{}/proton", self.proton)).arg("waitforexitandrun");
        game.arg(format!("{}/PlayGTAV.exe", self.game)).args(["-benchmark", "-benchmarkIterations", "1", "-benchmarkFrameTimes"]);
        game.stdin(Stdio::null()).stdout(log).stderr(both);
        // Its own session and process group (`setsid`), so the whole launch can be ended as one.
        // SAFETY: setsid is async-signal-safe and touches nothing of this process.
        unsafe {
            game.pre_exec(|| {
                libc::setsid();
                Ok(())
            })
        };
        let start = sh::epoch();
        self.put("start", &format!("{}\n", sh::local_time(start)));
        let child = game.spawn().map_err(|e| sh::not_started(&game, e))?;
        let smi = std::fs::File::create(self.file("gpu.csv")).ok().and_then(|samples| {
            Command::new("nvidia-smi")
                .args(["--query-gpu=timestamp,utilization.gpu,power.draw", "--format=csv,noheader,nounits", "-l", "1"])
                .stdin(Stdio::null())
                .stdout(samples)
                .stderr(Stdio::null())
                .spawn()
                .ok()
        });
        let group = child.id() as i32;
        say!("launched pgid={group}");
        Ok(Launch { child, group, smi, start })
    }

    /// One launch: "result", or what went wrong.
    fn launch_once(&self) -> Res<String> {
        // A run that was told to end (or whose output is gone) starts no game.
        sh::interrupted()?;
        let before = self.newest_benchmark();
        let mut launch = self.spawn()?;
        let start = launch.start;
        let (mut why, mut result) = (String::new(), String::new());
        let (mut stalled_at, mut seen, mut note_fg): (Option<i64>, Option<i64>, bool) = (None, None, false);
        let stalled = |at: Option<i64>| at.map_or_else(String::new, |s| s.to_string());
        for i in 1..=self.stuck_s / 5 {
            sh::sleep(5.0)?;
            let now = sh::epoch();
            let new = self.newest_benchmark();
            if !new.is_empty() && new != before {
                say!("result {new} after {}s", now - start);
                why = "result".to_string();
                result = new;
                sh::sleep(5.0)?;
                break;
            }
            let game = proc::named(GAME).first().copied();
            if !matches!(launch.child.try_wait(), Ok(None)) {
                why = match stalled_at {
                    Some(at) => format!("the game stopped presenting at {at}s and exited after {}s", now - start),
                    None => format!("launch exited early after {}s", now - start),
                };
                break;
            }
            let Some(game) = game else {
                match seen {
                    // The game is gone but the launch is not (the launcher or its crash reporter lingers).
                    Some(seen) if now - seen >= 30 => {
                        why = format!("the game exited after {}s", seen - start);
                        break;
                    }
                    None if now - start >= 120 => {
                        why = format!("the game did not start within {}s (the launcher is still running)", now - start);
                        break;
                    }
                    _ => continue,
                }
            };
            seen = Some(now);
            let Some(csv) = self.newest_csv() else { continue };
            // MangoHud writes a line per presented frame: a log that stops growing is a game that stopped presenting.
            let age = now - std::fs::metadata(&csv).map_or(now, |m| m.mtime());
            if age >= 15 && stalled_at.is_none() {
                stalled_at = Some(now - start - age);
                say!("  the game stopped presenting at {}s; GTA's own watchdog ends it 60 s later", stalled(stalled_at));
                self.put("stalled-threads.txt", &thread_states(game));
            } else if age >= 90 {
                why = format!("the game stopped presenting at {}s and was still stalled after {}s", stalled(stalled_at), now - start);
                break;
            } else if age < 15 && stalled_at.is_some() {
                say!("  presenting again after {}s", now - start);
                stalled_at = None;
            }
            if self.fg_on && !note_fg && i % 2 == 0 {
                let line = super::fg_line(&csv.to_string_lossy(), &(2 * self.refresh).to_string()).unwrap_or_default();
                if line.starts_with("stopped") {
                    let mut words = line.split_whitespace().skip(1);
                    let (at, shown) = (words.next().unwrap_or(""), words.next().unwrap_or(""));
                    let best = words.collect::<Vec<_>>().join(" ");
                    let note = format!("  frame generation stopped at about {at}s (displayed {shown} fps, {best} before); GTA's own, not retried");
                    say!("{note}");
                    self.put("fg", &format!("{note}\n"));
                    note_fg = true;
                }
            }
            if i % 12 == 0 {
                say!("  waiting {}s: game={}", now - start, proc::named(GAME).len());
            }
        }
        if why.is_empty() {
            why = format!("stuck: no benchmark result after {}s with the game running", sh::epoch() - start);
            // What the game shows: the layer's next presented frame, when Neural Forge is in the launch.
            if self.mid.contains(NEURAL_FORGE) {
                if let Ok(capture) = std::fs::File::create(self.file("stuck-capture.txt")) {
                    if let Ok(both) = capture.try_clone() {
                        let _ = sh::timeout(30, self.shmctl(&["capture"]).stdout(capture).stderr(both));
                    }
                }
            }
            if let Some(csv) = self.newest_csv() {
                self.append("stuck-capture.txt", &format!("displayed fps at the end: {}\n", end_fps(&csv)));
            }
        }
        sh::sleep(3.0)?;
        self.put("end", &format!("{}\n", sh::now()));
        self.end_launch(&mut launch)?;
        if why != "result" {
            let new = self.newest_benchmark();
            if !new.is_empty() && new != before {
                why = "result".to_string();
                result = new;
            }
        }
        if why == "result" {
            let new = result;
            let name = new.rsplit('/').next().unwrap_or("");
            let stamp = name.strip_prefix("Benchmark-").unwrap_or(name);
            let stamp = stamp.strip_suffix(".txt").unwrap_or(stamp);
            if let Err(e) = std::fs::copy(&new, self.file("benchmark.txt")) {
                warn!("{new}: {e}");
            }
            let (prefix, suffix) = ("FrameTimes-Pass", format!("-{stamp}.txt"));
            let mut passes: Vec<String> = names(&self.docs).into_iter().filter(|n| n.starts_with(prefix) && n.ends_with(&suffix)).collect();
            passes.sort();
            if passes.is_empty() {
                warn!("no {}/{prefix}*{suffix}", self.docs);
            }
            for pass in passes {
                let to = self.file(pass.strip_prefix("FrameTimes-").unwrap_or(&pass));
                if let Err(e) = std::fs::copy(format!("{}/{pass}", self.docs), &to) {
                    warn!("{}/{pass}: {e}", self.docs);
                }
            }
            return Ok(why);
        }
        // What the system and the game recorded about the failed launch.
        let journal =
            Command::new("journalctl").args(["-k", "--since", &format!("@{start}"), "--no-pager"]).stdin(Stdio::null()).stderr(Stdio::null()).output();
        let journal = journal.map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let faults = Regex::new("Xid|NVRM|segfault|Out of memory|oom-kill").unwrap();
        let kernel: String = journal.lines().filter(|l| faults.is_match(l)).map(|l| format!("{l}\n")).collect();
        self.put("kernel.txt", &kernel);
        let crash = match super::dump_lines(&self.crash_logs, start as f64) {
            Ok(lines) => lines.iter().map(|l| format!("{l}\n")).collect::<String>(),
            Err(e) => format!("{e}\n"),
        };
        self.put("crash.txt", &crash);
        if !crash.is_empty() {
            why = format!("{why}; {}", crash.lines().collect::<Vec<_>>().join(";"));
        }
        if let Some(xid) = kernel.lines().find_map(|l| l.find("Xid").map(|at| &l[at..])) {
            why = format!("{why}; GPU hang: {xid}");
        }
        Ok(why)
    }
}

/// Puts every --set back and checks it took: a value left changed (the CLI can no longer open the
/// channel, say) is an error and fails the run, whatever its result. True when all are back.
fn put_back(rig: &Rig, restore: &[(String, String)]) -> bool {
    let mut ok = true;
    for (key, want) in restore {
        // Nothing interrupts this: a signal, or the "restored" line of the setting before written to an output that is
        // gone, would otherwise end this setting's commands as they start (sh::timeout), and it would stay changed.
        sh::take_interruption();
        let _ = sh::timeout(30, rig.shmctl(&["set", key, want]).stdout(Stdio::null()).stderr(Stdio::null()));
        sh::take_interruption();
        let got = rig.status_of(key);
        if got == *want {
            say!("restored {key}={got}");
        } else {
            warn!("error: {key} is {} after the run, not the previous {want}: set it back by hand", if got.is_empty() { "unreadable" } else { &got });
            ok = false;
        }
    }
    ok
}

/// The run on the rig, once its directory exists; `restore` collects what --set changed.
fn bench(rig: &Rig, args: &Args, restore: &mut Vec<(String, String)>) -> Res<ExitCode> {
    for set in &args.sets {
        let (key, value) = set.split_once('=').unwrap_or((set, set));
        let old = rig.status_of(key);
        // Without the value it has now it could not be put back.
        if old.is_empty() {
            return Err(Fail::new(1, format!("error: shmctl status shows no {key}; not changing it")));
        }
        restore.push((key.to_string(), old));
        if !sh::timeout(30, rig.shmctl(&["set", key, value]).stdout(Stdio::null())).is_ok_and(|code| code == 0) {
            return Err(Fail::new(1, format!("shmctl set {set} failed")));
        }
        rig.tee("settings", &format!("set {key}={}", rig.status_of(key)));
    }
    if connected_to(3390) {
        rig.tee("settings", "warning: a remote desktop session is connected (port 3390); it costs about 8 fps");
    }

    let mut attempt = 0;
    let why = loop {
        attempt += 1;
        let why = rig.launch_once()?;
        rig.append("attempts", &format!("{attempt} {} {why}\n", sh::now()));
        if why == "result" {
            break why;
        }
        say!("attempt {attempt}: {why}");
        let kept = rig.file(&format!("attempt-{attempt}"));
        std::fs::create_dir_all(&kept).map_err(|e| Fail::new(1, format!("{}: {e}", kept.display())))?;
        for name in names(&rig.out) {
            if name != "attempts" && name != "settings" && rig.file(&name).is_file() {
                let _ = std::fs::rename(rig.file(&name), kept.join(&name));
            }
        }
        if attempt >= args.tries {
            say!("no result after {attempt} launches");
            break why;
        }
        sh::sleep(15.0)?;
    };
    // A frame series the layer captured during the run goes into the run's directory.
    let series = Regex::new(r"^.*\[capture\] capturing the next [0-9]* presents into (.*)$").unwrap();
    let log = String::from_utf8_lossy(&std::fs::read(rig.file("launch.log")).unwrap_or_default()).into_owned();
    for dir in log.lines().filter_map(|l| series.captures(l).map(|c| c[1].trim().to_string())) {
        if Path::new(&dir).is_dir() {
            let to = rig.file(Path::new(&dir).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default().as_str());
            if std::fs::rename(&dir, &to).is_err() {
                // Another filesystem: mv copies.
                let _ = Command::new("mv").arg(&dir).arg(format!("{}/", rig.out)).status();
            }
        }
    }
    let _ = Command::new("ls").arg(&rig.out).status();
    Ok(ExitCode::from(u8::from(why != "result")))
}

fn local(args: &[String]) -> Res<ExitCode> {
    let args = parse(args)?;
    sh::trap_signals();
    let home = std::env::var("HOME").unwrap_or_default();
    let client = sh::var("NF_BENCH_STEAM_ROOT").unwrap_or_else(|| format!("{home}/.local/share/Steam"));
    let library = sh::var("NF_BENCH_STEAM_LIBRARY").unwrap_or_else(|| client.clone());
    // The Proton Steam runs GTA with (config.vdf's CompatToolMapping), so a benchmark runs on what the game runs on
    // (a Neural Forge Proton copy gives the layer DLSS's own records); NF_BENCH_PROTON overrides it.
    let proton = match sh::var("NF_BENCH_PROTON") {
        Some(proton) => proton,
        None => super::mapped_proton(&client, &[library.clone(), client.clone()], APP).map_err(|e| Fail::new(1, e))?,
    };
    say!("Proton: {proton}");
    let prefix = format!("{library}/steamapps/compatdata/{APP}");
    let user = format!("{prefix}/pfx/drive_c/users/steamuser");
    let settings = format!("{user}/Documents/Rockstar Games/GTAV Enhanced/settings.xml");
    let stuck = sh::var("NF_BENCH_STUCK_S").unwrap_or_else(|| "480".to_string());
    let settings_text = String::from_utf8_lossy(&std::fs::read(&settings).unwrap_or_default()).into_owned();
    let refresh = Regex::new(r#"<RefreshRate value="([0-9]*)""#).unwrap();
    let mut rig = Rig {
        game: format!("{library}/steamapps/common/Grand Theft Auto V Enhanced"),
        runtime: format!("{library}/steamapps/common/SteamLinuxRuntime_4"),
        docs: format!("{user}/Documents/Rockstar Games/GTAV Enhanced/Benchmarks"),
        crash_logs: format!("{user}/AppData/Local/Rockstar Games/GTAV Enhanced/CrashLogs"),
        cli: sh::var("NF_BENCH_CLI").unwrap_or_else(|| format!("{home}/.local/share/neural-forge/bin/neural-forge-cli")),
        stuck_s: stuck.parse().map_err(|_| Fail::new(1, format!("error: NF_BENCH_STUCK_S is not a number of seconds: {stuck}")))?,
        // SAFETY: getuid has no failure.
        uid: unsafe { libc::getuid() },
        out: String::new(),
        mid: args.mid.clone(),
        fg_on: settings_text.contains("<FrameGenType value=\"1\""),
        refresh: settings_text.lines().find_map(|l| refresh.captures(l)).and_then(|c| c[1].parse().ok()).unwrap_or(288),
        env: args.env.clone(),
        client,
        library,
        proton,
        prefix,
    };
    for exe in [GAME, "PlayGTAV.exe"] {
        let running = proc::named(exe);
        if !running.is_empty() {
            return Err(Fail::new(1, format!("error: GTA is already running ({exe}, pid {}); not launching a second one", pid_list(&running))));
        }
    }
    if !sh::is_executable(Path::new(&format!("{}/_v2-entry-point", rig.runtime))) {
        return Err(Fail::new(1, format!("error: no Steam Linux Runtime 4 at {} (set NF_BENCH_STEAM_LIBRARY)", rig.runtime)));
    }
    rig.out = format!("{}/{}", sh::var("NF_BENCH_DIR").unwrap_or_else(|| format!("{home}/nf-spike/gta")), args.label);
    let _ = std::fs::remove_dir_all(&rig.out);
    std::fs::create_dir_all(&rig.out).map_err(|e| Fail::new(1, format!("{}: {e}", rig.out)))?;

    // Whatever happens from here on, the settings --set changed are put back.
    let mut restore = Vec::new();
    let result = bench(&rig, &args, &mut restore);
    // The signal that ended the run (a lost output counts as one) is taken first: putting back runs its commands to
    // their end.
    let signal = sh::take_interruption();
    if !put_back(&rig, &restore) {
        if let Err(Fail { message: Some(message), .. }) = &result {
            warn!("{message}");
        }
        return Ok(ExitCode::FAILURE);
    }
    match signal {
        Some(signal) => Err(Fail::silent((128 + signal) as u8)),
        None => result,
    }
}

/// `--host HOST ...`: this program (or NF_BENCH_XTASK) copied to the rig, run there with the rest of the arguments, and
/// removed again.
fn on_host(host: &str, args: &[String]) -> Res<ExitCode> {
    let binary = match sh::var("NF_BENCH_XTASK") {
        Some(named) => PathBuf::from(named),
        None => std::env::current_exe().map_err(|e| Fail::new(1, format!("error: cannot find this program: {e}")))?,
    };
    let copy = std::fs::File::open(&binary).map_err(|e| Fail::new(1, format!("error: {}: {e}", binary.display())))?;
    let cannot_copy = || Fail::new(1, format!("error: cannot copy xtask to {host}"));
    let sent = Command::new("ssh")
        .arg(host)
        .arg("f=$(mktemp) && cat > \"$f\" && chmod +x \"$f\" && echo \"$f\"")
        .stdin(copy)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| cannot_copy())?;
    let remote = String::from_utf8_lossy(&sent.stdout).trim_end_matches('\n').to_string();
    if !sent.status.success() || remote.is_empty() {
        return Err(cannot_copy());
    }
    let remote = sh::quote(&remote);
    sh::trap_signals();
    let words: String = args.iter().map(|a| format!(" {}", sh::quote(a))).collect();
    // The remote shell removes the binary itself when it ends (also when the connection drops: sshd hangs it up); the
    // removal below is the backstop for a run that never got that far.
    let run =
        Command::new("ssh").arg(host).arg(format!("f={remote}; trap 'rm -f \"$f\"' EXIT HUP INT TERM; \"$f\" gta-bench{words}")).stdin(Stdio::null()).status();
    let _ = Command::new("ssh").arg(host).arg(format!("rm -f {remote}")).stdin(Stdio::null()).status();
    if sh::interrupted().is_err() {
        return Ok(ExitCode::from(130));
    }
    Ok(ExitCode::from(run.map_or(255, sh::code)))
}

/// The argument checks of the self-test: a label that is no plain name is refused before anything is touched.
pub fn self_test() -> Result<(), String> {
    for bad in ["", ".", "..", "a/b"] {
        match parse(&[bad.to_string(), String::new()]) {
            Err(Fail { code: 2, message: Some(message) }) if message.contains("label must be") => {}
            _ => return Err(format!("label '{bad}' accepted")),
        }
    }
    say!("gta-bench label checks ok");
    Ok(())
}

pub fn main(args: &[String]) -> Res<ExitCode> {
    match args {
        [flag, host, rest @ ..] if flag == "--host" => on_host(host, rest),
        _ => local(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn arguments_as_the_script_read_them() {
        let args = parse(&strings(&["--set", "a=1", "--tries", "5", "--set", "b=2", "run-1", "", "X=1", "W=a=b;c"])).unwrap();
        assert_eq!((args.sets, args.tries, args.label.as_str(), args.mid.as_str()), (strings(&["a=1", "b=2"]), 5, "run-1", ""));
        assert_eq!(args.env, [("X".to_string(), "1".to_string()), ("W".to_string(), "a=b;c".to_string())]);
        assert_eq!(parse(&strings(&["only-a-label"])).unwrap_err().code, 2);
        for tries in ["0", "", "x", "-1", "1.5"] {
            let fail = parse(&strings(&["--tries", tries, "l", ""])).unwrap_err();
            assert_eq!((fail.code, fail.message.as_deref()), (2, Some("error: --tries takes a count of 1 or more")), "{tries}");
        }
        assert_eq!(parse(&strings(&["l", "", "not-an-assignment"])).unwrap_err().code, 2);
        self_test().unwrap();
    }

    #[test]
    fn the_median_of_the_last_frames() {
        let dir = sh::mktemp().unwrap();
        let csv = dir.path().join("log.csv");
        std::fs::write(&csv, "fps,frametime\n30.5,1\n10.0,2\n20.25,3\n").unwrap();
        // The header line reads as 0 and sorts first, as it did for `sort -n`.
        assert_eq!(end_fps(&csv), "20.25");
    }
}
