//! The pre-release game sweep: runs the tested games installed on the machine, each through its built-in benchmark,
//! and says from the layer's status, its log and the kernel log whether each still works as the tested games say
//! (`data/games.toml`, `neural_forge_supervisor::compat`): the placement reached, frames held (the model ran),
//! the placement kept, frame generation on where expected, no missed frames, no device lost or `[panic]`, no Xid, the
//! HDR verdict expected, and a clean recording. Prints a table.
//! `scripts/game-sweep.sh` builds and runs it; docs/RUNNING_AND_MEASURING.md, "The pre-release game sweep".
//!
//! Every run (`scripts/game-sweep.toml`, one `[[run]]` per run) is started by Steam
//! (`steam steam://rungameid/<appid>`), on the Proton Steam has for the game, with Neural Forge's variables, the run's
//! log file, Proton's HDR switches (with `hdr = true`) and the run's arguments in its launch options. Launch options
//! are only written while Steam is shut down, so for each run the sweep shuts Steam down, writes them, starts Steam
//! (through `systemd-run --user`, with the session's environment), runs the game, and puts them back the same way.
//!
//! Each run is made from start to finish under the rig's GPU lock (`--lock`, default `$NF_GPU_LOCK` or
//! `~/scratch/gpu.lock`; flock), unless a parent holds it (`NF_GPU_LOCK_HELD`), and everything it changed is put
//! back before the lock is let go, also after an error or Ctrl-C (not after a panic: the build aborts on one, and
//! puts nothing back; the files as found are then in `<out>/<label>/`): the launch options (Steam shut down, every
//! account's `localconfig.vdf` put back byte for byte, Steam started again), the game settings the run edits
//! (`edit_file`, byte for byte, hash-checked), the installed layer when `--layer` swaps in a build under test
//! (hash-checked) and the channel directory that build would re-create. Between runs the lock is free, so other
//! runs on the machine go in between. What it runs gets `NF_GPU_LOCK_HELD=1`.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use neural_forge_protocol::enums::preupscale_state;
use neural_forge_supervisor::compat::{self, Placement, Value};
use neural_forge_supervisor::{launch_options, steam, vdf};

static CANCELLED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    CANCELLED.store(true, Ordering::Relaxed);
}

fn cancelled() -> bool {
    CANCELLED.load(Ordering::Relaxed)
}

/// Sleeps `d`, waking early on a cancel.
fn nap(d: Duration) {
    let until = Instant::now() + d;
    while Instant::now() < until && !cancelled() {
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditKind {
    /// Cyberpunk 2077's `UserSettings.json`: the option objects named `NAME` get `value` (bools from true/false).
    JsonOption,
    /// `<NAME value="..." />` elements (GTA V's `settings.xml`).
    XmlValue,
}

/// What a run expects of the game's HDR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HdrExpect {
    Hdr10,
    Sdr,
}

/// One `[[run]]` of the sweep's configuration.
#[derive(Debug, Clone)]
struct Run {
    label: String,
    appid: u32,
    /// Arguments after `%command%` (the game's own: a benchmark switch).
    args: String,
    /// How long to watch once the expected placement is reached (or until the game exits by itself).
    seconds: u64,
    /// How long to wait for the game to start and reach its placement.
    timeout: u64,
    /// When the recording starts, in seconds after the placement was reached (default [`RECORD_AFTER`]): usually within
    /// one scene of a benchmark, not across its own loading screens (the time to the placement varies).
    record_after: u64,
    /// Record with the CPU encoder (`recorder = "cpu"`): only where the GPU encoder can't run beside the game (see
    /// [`Recorder`]); the GPU encoder keeps every frame and is the one vw's thresholds were set with.
    cpu_recorder: bool,
    /// The game runs with frame generation (its settings as found, or the run's edits): the run fails when the
    /// layer's presents per DLSS evaluate say it is off ([`FG_MIN_RATIO`]).
    frame_generation: bool,
    hdr: bool,
    ray_reconstruction: bool,
    edit_kind: Option<EditKind>,
    /// Relative to the game's Proton user folder (`compatdata/<appid>/pfx/drive_c/users/steamuser`).
    edit_file: Option<String>,
    edits: Vec<(String, String)>,
    /// More variables for the game's launch options (`env = ["NAME=VALUE"]`): a run of a setting that only a
    /// variable switches.
    env: Vec<String>,
}

impl Run {
    fn from_table(t: &BTreeMap<String, Value>) -> Result<Run, String> {
        const KEYS: [&str; 14] = [
            "label",
            "appid",
            "args",
            "seconds",
            "timeout",
            "record_after",
            "recorder",
            "frame_generation",
            "hdr",
            "ray_reconstruction",
            "edit_kind",
            "edit_file",
            "edits",
            "env",
        ];
        if let Some(k) = t.keys().find(|k| !KEYS.contains(&k.as_str())) {
            return Err(format!("unknown key {k}"));
        }
        let text = |k: &str| t.get(k).and_then(Value::as_str).map(str::to_string);
        let int = |k: &str, default: i64| match t.get(k) {
            None => Ok(default),
            Some(Value::Int(n)) if *n >= 0 => Ok(*n),
            Some(_) => Err(format!("{k} must be a number of seconds")),
        };
        let flag = |k: &str| match t.get(k) {
            None => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(format!("{k} must be true or false")),
        };
        let appid = match t.get("appid") {
            Some(Value::Int(id)) if *id > 0 && *id < i64::from(i32::MAX) => *id as u32,
            _ => return Err("appid must be a Steam app id".into()),
        };
        let label = text("label").ok_or("label is missing")?;
        if label.is_empty() || !label.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
            return Err(format!("label {label:?} must be letters, digits, - and _"));
        }
        let edit_kind = match text("edit_kind").as_deref() {
            None => None,
            Some("json-option") => Some(EditKind::JsonOption),
            Some("xml-value") => Some(EditKind::XmlValue),
            Some(other) => return Err(format!("edit_kind {other:?} is not json-option or xml-value")),
        };
        let edits = match t.get("edits") {
            None => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| {
                    v.as_str().and_then(|s| s.split_once('=')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).ok_or("edits are NAME=VALUE strings")
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err("edits must be an array".into()),
        };
        let env = match t.get("env") {
            None => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| s.split_once('=').is_some_and(|(k, _)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')))
                        .map(str::to_string)
                        .ok_or("env entries are NAME=VALUE strings")
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err("env must be an array".into()),
        };
        let run = Run {
            label,
            appid,
            args: text("args").unwrap_or_default(),
            seconds: int("seconds", 90)? as u64,
            timeout: int("timeout", 600)? as u64,
            record_after: int("record_after", RECORD_AFTER as i64)? as u64,
            cpu_recorder: match text("recorder").as_deref() {
                None | Some("gpu") => false,
                Some("cpu") => true,
                Some(other) => return Err(format!("recorder {other:?} is not gpu or cpu")),
            },
            frame_generation: flag("frame_generation")?,
            hdr: flag("hdr")?,
            ray_reconstruction: flag("ray_reconstruction")?,
            edit_kind,
            edit_file: text("edit_file"),
            edits,
            env,
        };
        if run.record_after + RECORD_SECONDS > run.seconds {
            return Err(format!("record_after + {RECORD_SECONDS} s of recording must fit in seconds ({})", run.seconds));
        }
        if run.edits.is_empty() != run.edit_file.is_none() || run.edit_file.is_some() != run.edit_kind.is_some() {
            return Err("edit_file, edit_kind and edits go together".into());
        }
        Ok(run)
    }

    fn game(&self) -> Option<&'static compat::Game> {
        compat::find(Some(self.appid), "")
    }

    /// The placement this run expects: the game's own, or with Ray Reconstruction on.
    fn placement(&self) -> Option<Placement> {
        let g = self.game()?;
        Some(if self.ray_reconstruction { g.placement_rr.unwrap_or(g.placement) } else { g.placement })
    }

    fn hdr_expect(&self) -> HdrExpect {
        if self.hdr {
            HdrExpect::Hdr10
        } else {
            HdrExpect::Sdr
        }
    }
}

fn parse_config(text: &str) -> Result<Vec<Run>, String> {
    let runs = compat::tables(text, "run")?
        .iter()
        .enumerate()
        .map(|(i, t)| Run::from_table(t).map_err(|e| format!("run {}: {e}", i + 1)))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, r) in runs.iter().enumerate() {
        if runs[..i].iter().any(|o| o.label == r.label) {
            return Err(format!("label {} is used twice", r.label));
        }
        let Some(game) = r.game() else { return Err(format!("{}: app {} is not in data/games.toml", r.label, r.appid)) };
        if r.hdr && game.hdr != compat::Hdr::Hdr10 {
            return Err(format!("{}: {} has no HDR10 in data/games.toml", r.label, game.name));
        }
        if r.ray_reconstruction && game.placement_rr.is_none() {
            return Err(format!("{}: {} has no Ray Reconstruction placement in data/games.toml", r.label, game.name));
        }
    }
    Ok(runs)
}

/// One `shmctl status` read.
#[derive(Debug, Clone, Default, PartialEq)]
struct Sample {
    secs: u64,
    layer_frames: u64,
    state: u32,
    reason: String,
    misses: u64,
    holds: u64,
    device_lost: u64,
    hdr: String,
    /// DLSS evaluates the layer matched to their launches so far (`ngx_matched`): one per rendered frame.
    ngx_matched: u64,
}

fn parse_status(text: &str, secs: u64) -> Option<Sample> {
    let mut s = Sample { secs, ..Sample::default() };
    let mut seen = false;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        let num = || v.split_whitespace().next().and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
        match k {
            "layer_frames" => {
                s.layer_frames = num();
                seen = true;
            }
            "preupscale_state" => s.state = num() as u32,
            "preupscale_reason" => s.reason = v.to_string(),
            "preupscale_misses" => s.misses = num(),
            "device_lost_at" => s.device_lost = num(),
            "ngx_matched" => s.ngx_matched = num(),
            "hdr_verdict" => s.hdr = v.to_string(),
            "edit_cache: refresh_holds" => {
                // edit_cache: refresh_holds=N cached_holds=M ...
                let refresh = num();
                let cached = line.split("cached_holds=").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
                s.holds = refresh + cached;
            }
            _ => {}
        }
    }
    seen.then_some(s)
}

fn state_name(state: u32) -> &'static str {
    neural_forge_supervisor::shm_status::preupscale_state_name(state)
}

/// A run's verdict: each column of the table, and the failures.
#[derive(Debug, Default)]
struct Outcome {
    reached: String,
    holds: String,
    misses: String,
    /// Status samples after the placement was reached in which it was lost (not held, not a game pause).
    lost: String,
    /// Frame generation: presents per rendered frame while holding (`x3.0`), `-` with no measure.
    fg: String,
    hdr: String,
    faults: String,
    /// The recording's verdict ([`picture_verdict`]): "ok", the flags, or why there is none.
    picture: String,
    failures: Vec<String>,
}

/// More samples than this (5 s each) after the placement was reached without it, outside the game's own pauses,
/// fail the run: the model has to keep running, not only get there (the VRAM guard pausing it counts).
const MAX_LOST_SAMPLES: usize = 1;
/// Presents per rendered frame (DLSS evaluate) under which frame generation counts as off.
const FG_MIN_RATIO: f64 = 1.5;
/// Above this the measure is not believed (DLSS frame generation makes at most 4x): "not measured".
const FG_MAX_RATIO: f64 = 4.5;
/// DLSS evaluates an interval between two samples (5 s) needs to count: a few rendered frames say nothing.
const FG_MIN_EVALUATES: u64 = 25;

/// Whether `state` keeps the run's placement: the state it expects (`want`), or the game's own pause (a menu, a
/// loading screen, and also a DLSS game before DLSS's first frame, with nothing held yet), after which the model
/// resumes where it was. Another placement (after DLSS falling back to before
/// the upscaler, or both) is not the one the run tests.
fn kept(state: u32, want: u32) -> bool {
    state == want || state == preupscale_state::PAUSED
}

/// Presents per DLSS evaluate over the intervals between consecutive samples that both hold `want`: about 1 without
/// frame generation, its multiplier with it. Only intervals with at least [`FG_MIN_EVALUATES`] count; `None` when none
/// does, or the ratio is above [`FG_MAX_RATIO`] (not a measure to trust).
fn frame_generation_ratio(samples: &[Sample], want: u32) -> Option<f64> {
    let (mut frames, mut evaluates) = (0u64, 0u64);
    for w in samples.windows(2) {
        let n = w[1].ngx_matched.saturating_sub(w[0].ngx_matched);
        let f = w[1].layer_frames.saturating_sub(w[0].layer_frames);
        // An interval above the cap is left out (a loading screen's presents counted against a few evaluates).
        if w[0].state == want && w[1].state == want && n >= FG_MIN_EVALUATES && (f as f64) <= FG_MAX_RATIO * n as f64 {
            frames += f;
            evaluates += n;
        }
    }
    let ratio = (evaluates > 0).then(|| frames as f64 / evaluates as f64)?;
    (ratio <= FG_MAX_RATIO).then_some(ratio)
}

/// Judges a run from its status samples, its layer log and the kernel's Xid lines since it started.
/// `window` is the recording's span in seconds since the launch: the samples over it must hold the run's placement,
/// with holds growing (the picture is judged with the model running).
fn evaluate(expect: Placement, hdr: HdrExpect, fg_expected: bool, window: Option<(u64, u64)>, samples: &[Sample], log: &str, xid: &str) -> Outcome {
    let mut o = Outcome::default();
    let want = expect.preupscale_state();
    let Some(last) = samples.last() else {
        o.failures.push("no status from the channel: the layer never attached (or the CLI speaks another channel version)".into());
        o.reached = "-".into();
        return o;
    };
    if !samples.iter().any(|s| s.layer_frames > 0) {
        o.failures.push("the layer presented no frame".into());
    }
    let first_reached = samples.iter().position(|s| s.state == want);
    o.reached = match first_reached {
        Some(i) => format!("{} after {} s", short_state(want), samples[i].secs),
        None => format!("no: {}", short_state(last.state)),
    };
    match first_reached {
        None => o.failures.push(format!(
            "expected {} ({}), saw {}{}",
            expect.describe(),
            state_name(want),
            state_name(last.state),
            if last.reason.is_empty() { String::new() } else { format!(": {}", last.reason) }
        )),
        Some(i) => {
            let grew = last.holds.saturating_sub(samples[i].holds);
            o.holds = format!("{} (+{grew})", last.holds);
            if expect != Placement::FinishedFrame && grew == 0 && last.holds == 0 {
                o.failures.push("no frame was held: the model did not run in the game's frame".into());
            }
            let lost: Vec<&Sample> = samples[i..].iter().filter(|s| !kept(s.state, want)).collect();
            o.lost = lost.len().to_string();
            if lost.len() > MAX_LOST_SAMPLES {
                let first = lost[0];
                o.failures.push(format!(
                    "lost the placement in {} samples after reaching it (from {} s: {}{})",
                    lost.len(),
                    first.secs,
                    state_name(first.state),
                    if first.reason.is_empty() { String::new() } else { format!(": {}", first.reason) }
                ));
            }
        }
    }
    if o.lost.is_empty() {
        o.lost = "-".into();
    }
    if let Some((from, to)) = window {
        // The samples taken over the recording: from the one at its start to the one after its end, which covers its
        // last seconds (so the placement may be demanded up to 5 s past the end).
        let over: Vec<&Sample> = samples.iter().filter(|s| s.secs >= from && s.secs <= to + 5).collect();
        let before = samples.iter().rev().find(|s| s.secs < from);
        if over.is_empty() {
            o.failures.push("no status sample over the recording".into());
        } else if let Some(s) = over.iter().find(|s| s.state != want) {
            o.failures.push(format!(
                "the recording was not made with the model at its placement ({} s: {}{})",
                s.secs,
                state_name(s.state),
                if s.reason.is_empty() { String::new() } else { format!(": {}", s.reason) }
            ));
        } else if expect != Placement::FinishedFrame && over.last().map(|s| s.holds) <= before.map(|s| s.holds) {
            o.failures.push("the recording was made with no frame held (holds did not grow over it)".into());
        }
    }
    let ratio = frame_generation_ratio(samples, want);
    o.fg = ratio.map_or("-".into(), |r| format!("x{r:.1}"));
    if fg_expected && ratio.is_none_or(|r| r < FG_MIN_RATIO) {
        o.failures.push(format!(
            "frame generation is off (presents per rendered frame while holding: {}); the run expects it on",
            ratio.map_or("not measured, or not believed (above 4.5)".into(), |r| format!("{r:.2}"))
        ));
    }
    if o.holds.is_empty() {
        o.holds = last.holds.to_string();
    }
    o.misses = last.misses.to_string();
    if last.misses > 0 {
        o.failures.push(format!("{} missed frames", last.misses));
    }
    let verdict = samples.iter().rev().map(|s| s.hdr.as_str()).find(|h| h.starts_with("game ")).unwrap_or("");
    o.hdr = match verdict {
        v if v.starts_with("game outputs HDR10") => "HDR10".into(),
        v if v.starts_with("game outputs scRGB") => "scRGB".into(),
        v if v.starts_with("game outputs SDR") => "SDR".into(),
        v if v.starts_with("game asked for SDR") => "SDR, metadata sent".into(),
        _ => "unknown".into(),
    };
    let hdr_ok = match hdr {
        HdrExpect::Hdr10 => o.hdr == "HDR10",
        HdrExpect::Sdr => o.hdr == "SDR",
    };
    if !hdr_ok {
        o.failures.push(format!(
            "HDR: expected {}, the verdict says {}",
            if hdr == HdrExpect::Hdr10 { "HDR10" } else { "SDR" },
            if verdict.is_empty() { "nothing" } else { verdict }
        ));
    }
    let mut faults = Vec::new();
    if samples.iter().any(|s| s.device_lost != 0) {
        faults.push("device lost");
        o.failures.push("the layer recorded a lost device".into());
    }
    if log.contains("[panic]") {
        faults.push("panic");
        o.failures.push(format!("the layer panicked: {}", log.lines().find(|l| l.contains("[panic]")).unwrap_or("").trim()));
    }
    let xids: Vec<&str> = xid.lines().filter(|l| l.contains("Xid")).collect();
    if !xids.is_empty() {
        faults.push("Xid");
        o.failures.push(format!("{} Xid line(s): {}", xids.len(), xids[0].trim()));
    }
    o.faults = if faults.is_empty() { "none".into() } else { faults.join(", ") };
    o
}

fn short_state(state: u32) -> &'static str {
    match state {
        preupscale_state::HOLDING => "before",
        preupscale_state::HOLDING_AFTER => "after DLSS",
        preupscale_state::HOLDING_BOTH => "both",
        preupscale_state::IDENTIFIED => "refused",
        preupscale_state::WAITING => "waiting",
        preupscale_state::PAUSED => "paused",
        _ => "finished frame",
    }
}

/// The launch options a run starts the game with: the game's own, without Neural Forge's log variables (which the
/// sweep sets), with Neural Forge on for the game's executables, the run's log file, Proton's HDR switches and the
/// run's arguments after `%command%`.
fn sweep_options(existing: &str, exe: &[String], log: &Path, hdr: bool, env: &[String], args: &str) -> String {
    let base = launch_options::merge(existing, &exe.join(","));
    let tokens = launch_options::split_tokens(&base);
    let cmd = tokens.iter().position(|t| t == "%command%").unwrap_or(tokens.len());
    let drop = |t: &String| {
        t.starts_with("NEURAL_FORGE_LOG=")
            || t.starts_with("NEURAL_FORGE_LOG_TIME=")
            || t.starts_with("PROTON_ENABLE_WAYLAND=")
            || t.starts_with("PROTON_ENABLE_HDR=")
    };
    let mut out: Vec<String> = tokens[..cmd].iter().filter(|t| !drop(t)).cloned().collect();
    out.push(format!("NEURAL_FORGE_LOG={}", launch_options::shell_quote(&log.display().to_string())));
    if hdr {
        out.push("PROTON_ENABLE_WAYLAND=1".into());
        out.push("PROTON_ENABLE_HDR=1".into());
    }
    // The run's own variables last: they win over the game's own of the same name.
    for kv in env {
        let name = kv.split_once('=').map_or(kv.as_str(), |(k, _)| k);
        out.retain(|t| t.split_once('=').is_none_or(|(k, _)| k != name));
        out.push(kv.clone());
    }
    out.extend(tokens[cmd..].iter().cloned());
    if tokens.len() == cmd {
        out.push("%command%".into());
    }
    // The run's arguments, each switch with the values after it (`-benchmarkIterations 1`, `-offset -5`) as one unit.
    // The same switch in the game's options after %command% gets the run's values instead of its own; any other unit
    // is added at the end.
    let at = out.iter().position(|t| t == "%command%").map_or(out.len(), |i| i + 1);
    for unit in arg_units(&launch_options::split_tokens(args)) {
        let game = arg_units(&out[at..]);
        let mut pos = at;
        let mut found = None;
        for g in &game {
            if is_switch(&unit[0]) && g[0] == unit[0] {
                found = Some((pos, g.len()));
                break;
            }
            if !is_switch(&unit[0]) && *g == unit {
                found = Some((pos, g.len()));
                break;
            }
            pos += g.len();
        }
        match found {
            Some((pos, len)) => {
                out.splice(pos..pos + len, unit);
            }
            None => out.extend(unit),
        }
    }
    out.join(" ")
}

/// A switch on a command line: `-x`, `--x`, not a negative number (`-5`, `-.5`).
fn is_switch(t: &str) -> bool {
    t.strip_prefix('-').and_then(|r| r.chars().next()).is_some_and(|c| !c.is_ascii_digit() && c != '.')
}

/// Command-line tokens as units: each switch with the values after it, a leading value on its own.
fn arg_units(tokens: &[String]) -> Vec<Vec<String>> {
    let mut units: Vec<Vec<String>> = Vec::new();
    for t in tokens {
        match units.last_mut() {
            Some(unit) if !is_switch(t) && is_switch(&unit[0]) => unit.push(t.clone()),
            _ => units.push(vec![t.clone()]),
        }
    }
    units
}

/// `text` with each edit applied; `Err` names an edit that matched nothing.
fn apply_edits(kind: EditKind, text: &str, edits: &[(String, String)]) -> Result<String, String> {
    match kind {
        EditKind::XmlValue => {
            let mut out = text.to_string();
            for (name, value) in edits {
                let open = format!("<{name} value=\"");
                let at = out.find(&open).ok_or_else(|| format!("no <{name} value=...> element"))? + open.len();
                let end = out[at..].find('"').ok_or_else(|| format!("<{name}>'s value is not closed"))? + at;
                out.replace_range(at..end, value);
            }
            Ok(out)
        }
        EditKind::JsonOption => {
            let mut json: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
            for (name, value) in edits {
                let mut found = false;
                for group in json.get_mut("data").and_then(|d| d.as_array_mut()).into_iter().flatten() {
                    for option in group.get_mut("options").and_then(|o| o.as_array_mut()).into_iter().flatten() {
                        if option.get("name").and_then(|n| n.as_str()) != Some(name) {
                            continue;
                        }
                        let typed = match option.get("type").and_then(|t| t.as_str()) {
                            Some("bool") => serde_json::Value::Bool(matches!(value.as_str(), "true" | "1")),
                            Some("int") => value.parse::<i64>().map(serde_json::Value::from).map_err(|_| format!("{name} takes a whole number"))?,
                            Some("float") => value.parse::<f64>().map(serde_json::Value::from).map_err(|_| format!("{name} takes a number"))?,
                            _ => serde_json::Value::String(value.clone()),
                        };
                        option["value"] = typed;
                        found = true;
                    }
                }
                if !found {
                    return Err(format!("no option named {name}"));
                }
            }
            serde_json::to_string_pretty(&json).map_err(|e| e.to_string())
        }
    }
}

fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// The user session's environment (`systemctl --user show-environment`): what Steam, started from the desktop, sees
/// (the display, the session bus, `environment.d`'s variables). Programs the sweep starts get it.
fn session_env() -> Vec<(String, String)> {
    let out = Command::new("systemctl").args(["--user", "show-environment"]).output().ok();
    out.map(|o| String::from_utf8_lossy(&o.stdout).lines().filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect())
        .unwrap_or_default()
}

fn session_command(program: &str, env: &[(String, String)]) -> Command {
    let mut c = Command::new(program);
    c.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str()))).stdin(Stdio::null());
    c
}

/// Whether `kv` (one `NAME=value` of an environment) holds `needle` as a whole path part: followed by `/` or its end
/// (`compatdata/1091500` is not `compatdata/10915000`).
fn names_path(kv: &[u8], needle: &[u8]) -> bool {
    kv.windows(needle.len()).enumerate().any(|(i, w)| w == needle && kv.get(i + needle.len()).is_none_or(|&b| b == b'/'))
}

/// Every process (not this one) whose environment names `needle` (`compatdata/<appid>`) as a whole path part.
fn pids_with_env(needle: &str) -> Vec<i32> {
    let me = std::process::id() as i32;
    let Ok(dir) = std::fs::read_dir("/proc") else { return Vec::new() };
    dir.flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok())
        .filter(|&pid| pid != me)
        .filter(|pid| std::fs::read(format!("/proc/{pid}/environ")).is_ok_and(|env| env.split(|&b| b == 0).any(|kv| names_path(kv, needle.as_bytes()))))
        .collect()
}

/// Whether a process runs one of `exes` (by its command line, any case).
fn game_running(exes: &[String]) -> bool {
    let Ok(dir) = std::fs::read_dir("/proc") else { return false };
    let exes: Vec<String> = exes.iter().map(|e| e.to_lowercase()).collect();
    dir.flatten().filter(|e| e.file_name().to_str().is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))).any(|e| {
        std::fs::read(e.path().join("cmdline")).is_ok_and(|c| {
            let c = String::from_utf8_lossy(&c).to_lowercase();
            exes.iter().any(|x| c.contains(x.as_str()))
        })
    })
}

/// Ends every process of the game's prefix: TERM, then KILL after 10 s.
fn end_game(appid: u32) {
    let needle = format!("compatdata/{appid}");
    for signal in [libc::SIGTERM, libc::SIGKILL] {
        let pids = pids_with_env(&needle);
        if pids.is_empty() {
            return;
        }
        for pid in pids {
            // SAFETY: plain kill(2) on a pid read from /proc.
            unsafe { libc::kill(pid, signal) };
        }
        std::thread::sleep(Duration::from_secs(10));
    }
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/".into())
}

fn steam_running() -> bool {
    steam::steam_running(Path::new("/proc"))
}

fn stop_steam(env: &[(String, String)]) -> Result<(), String> {
    if !steam_running() {
        return Ok(());
    }
    let _ = session_command("steam", env).arg("-shutdown").stdout(Stdio::null()).stderr(Stdio::null()).status();
    for _ in 0..120 {
        if !steam_running() {
            std::thread::sleep(Duration::from_secs(3));
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Err("Steam did not shut down within 2 minutes".into())
}

fn start_steam(env: &[(String, String)]) -> Result<(), String> {
    if steam_running() {
        return Ok(());
    }
    // In its own unit, with the user manager's environment (the session's, environment.d included), so it outlives
    // the sweep the way the desktop's Steam does.
    let status = Command::new("systemd-run")
        .args(["--user", "--collect", "--quiet", &format!("--unit=neural-forge-sweep-steam-{}", std::process::id()), "steam", "-silent"])
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("systemd-run: {e}"))?;
    if !status.success() {
        return Err(format!("systemd-run steam exited with {status}"));
    }
    let _ = env;
    for _ in 0..120 {
        if steam_running() && Path::new(&home_dir()).join(".steam/steam.pipe").exists() {
            // Signed in and its library loaded: Steam takes a while after its window process appears.
            nap(Duration::from_secs(30));
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Err("Steam did not start within 2 minutes".into())
}

/// Everything the sweep changed, put back in [`Drop`] after an error or a Ctrl-C (not after a panic: the build has
/// `panic = "abort"`; the localconfig.vdf files, game settings and installed layer as found are then in the run's
/// folder, `<out>/<label>/`).
struct Restore {
    out: PathBuf,
    /// Each account's `localconfig.vdf` as found.
    localconfigs: Vec<(PathBuf, Vec<u8>)>,
    /// Game settings files as found.
    files: Vec<(PathBuf, Vec<u8>)>,
    /// The installed layer as found, and where it is.
    layer: Option<(PathBuf, Vec<u8>)>,
    /// The channel directory moved aside, and where it was.
    channel: Option<(PathBuf, PathBuf)>,
    steam_was_running: bool,
    steam_stopped: bool,
    env: Vec<(String, String)>,
}

impl Restore {
    fn put_back(&mut self) -> Vec<String> {
        let mut report = Vec::new();
        if !self.localconfigs.is_empty() {
            if let Err(e) = stop_steam(&self.env) {
                report.push(format!("ERROR: {e}: the launch options are NOT restored; put {} back by hand", self.out.join("localconfig").display()));
                return report;
            }
            for (path, bytes) in self.localconfigs.drain(..) {
                report.push(restore_file(&path, &bytes, "launch options"));
            }
            self.steam_stopped = true;
        }
        for (path, bytes) in self.files.drain(..) {
            report.push(restore_file(&path, &bytes, "game settings"));
        }
        if let Some((path, bytes)) = self.layer.take() {
            report.push(restore_file(&path, &bytes, "installed layer"));
        }
        if let Some((aside, path)) = self.channel.take() {
            let _ = std::fs::remove_dir_all(&path);
            match std::fs::rename(&aside, &path) {
                Ok(()) => report.push(format!("channel directory {} restored", path.display())),
                Err(e) => report.push(format!("ERROR: moving {} back to {} failed: {e}", aside.display(), path.display())),
            }
        }
        if self.steam_stopped && self.steam_was_running {
            match start_steam(&self.env) {
                Ok(()) => report.push("Steam started again".into()),
                Err(e) => report.push(format!("ERROR: {e}")),
            }
            self.steam_stopped = false;
        }
        report
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        for line in self.put_back() {
            println!("restore: {line}");
        }
    }
}

/// Writes `bytes` back to `path` (through a temporary file, keeping its permissions) and checks its hash.
fn restore_file(path: &Path, bytes: &[u8], what: &str) -> String {
    let tmp = path.with_extension("sweep-restore");
    let perms = std::fs::metadata(path).map(|m| m.permissions()).ok();
    let written = std::fs::write(&tmp, bytes).and_then(|()| {
        if let Some(p) = perms {
            std::fs::set_permissions(&tmp, p)?;
        }
        std::fs::rename(&tmp, path)
    });
    match (written, std::fs::read(path)) {
        (Ok(()), Ok(now)) if sha256(&now) == sha256(bytes) => format!("{what} restored: {} ({})", path.display(), &sha256(bytes)[..12]),
        (Err(e), _) => format!("ERROR: {what}: writing {} failed: {e}", path.display()),
        _ => format!("ERROR: {what}: {} does not match what was found", path.display()),
    }
}

/// Takes the GPU lock (flock), waiting for it; the returned file holds it. The wait is a blocking `flock` (on a
/// thread, so a cancel still ends the sweep): the kernel hands the lock to one of its waiters when it is let go, and a
/// sweep polling for it would lose to every waiter that blocks.
fn take_lock(path: &Path) -> Result<std::fs::File, String> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: flock(2) on a file this process holds open.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(file);
    }
    println!("waiting for the GPU lock {} (another game run holds it)", path.display());
    let fd = file.as_raw_fd();
    let (tx, rx) = std::sync::mpsc::channel();
    // SAFETY: as above; `file` outlives the wait (it is returned, or the process exits on a cancel).
    std::thread::spawn(move || tx.send(unsafe { libc::flock(fd, libc::LOCK_EX) }));
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(0) => return Ok(file),
            Ok(_) => return Err(format!("flock {}: {}", path.display(), std::io::Error::last_os_error())),
            Err(_) if cancelled() => return Err("cancelled while waiting for the GPU lock".into()),
            Err(_) => {}
        }
    }
}

struct Options {
    config: PathBuf,
    only: Vec<String>,
    layer: Option<PathBuf>,
    cli: PathBuf,
    out: PathBuf,
    lock: PathBuf,
    dry_run: bool,
    /// The screen recorder and analyser (`vw`: `vw replay start|save OUT|stop`, `vw analyze VIDEO --out DIR`), whose verdict
    /// every run needs to pass ([`picture_verdict`]); `None` with `--no-record`, and every run then fails.
    vw: Option<PathBuf>,
}

/// How long each run is recorded once its placement was reached, and how long after reaching it.
const RECORD_SECONDS: u64 = 10;
const RECORD_AFTER: u64 = 10;

const USAGE: &str =
    "usage: game-sweep [--config FILE] [--only LABEL,...] [--layer SO] [--cli PATH] [--out DIR] [--lock FILE] [--vw PATH|--no-record] [--dry-run]\n\
                     \x20      game-sweep --self-test";

fn parse_args() -> Result<Options, String> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let home = PathBuf::from(home_dir());
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut o = Options {
        config: repo.join("scripts/game-sweep.toml"),
        only: Vec::new(),
        layer: None,
        cli: PathBuf::from(neural_forge_supervisor::paths::data_home()).join("neural-forge/bin/neural-forge-cli"),
        out: home.join(format!("nf-sweep/{stamp}")),
        lock: std::env::var_os("NF_GPU_LOCK").map_or_else(|| home.join("scratch/gpu.lock"), PathBuf::from),
        dry_run: false,
        vw: Some(home.join("scratch/vw/bin/vw")),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or(format!("{a} takes a value"));
        match a.as_str() {
            "--config" => o.config = PathBuf::from(value()?),
            "--only" => o.only = value()?.split(',').map(str::to_string).collect(),
            "--layer" => o.layer = Some(PathBuf::from(value()?)),
            "--cli" => o.cli = PathBuf::from(value()?),
            "--out" => o.out = PathBuf::from(value()?),
            "--lock" => o.lock = PathBuf::from(value()?),
            "--dry-run" => o.dry_run = true,
            "--vw" => o.vw = Some(PathBuf::from(value()?)),
            "--no-record" => o.vw = None,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(o)
}

fn main() -> std::process::ExitCode {
    if std::env::args().nth(1).as_deref() == Some("--self-test") {
        return self_test();
    }
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("{USAGE}");
            return std::process::ExitCode::from(2);
        }
    };
    match sweep(&o) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// A run, prepared: the installed game it runs.
struct Prepared {
    run: Run,
    game: steam::Game,
    facts: &'static compat::Game,
    log: PathBuf,
}

fn sweep(o: &Options) -> Result<bool, String> {
    let text = std::fs::read_to_string(&o.config).map_err(|e| format!("{}: {e}", o.config.display()))?;
    let runs: Vec<Run> = parse_config(&text)?.into_iter().filter(|r| o.only.is_empty() || o.only.contains(&r.label)).collect();
    if runs.is_empty() {
        return Err("no run selected".into());
    }
    let roots = steam::roots();
    let installed = steam::games(&roots);
    let mut prepared = Vec::new();
    for run in runs {
        let Some(game) = installed.iter().find(|g| g.appid == run.appid.to_string()).cloned() else {
            println!("skip {}: app {} is not installed", run.label, run.appid);
            continue;
        };
        let facts = run.game().expect("parse_config checked it");
        let log = o.out.join(format!("{}.log", run.label));
        prepared.push(Prepared { run, game, facts, log });
    }
    let user_dir = |p: &Prepared| p.game.library.join("steamapps/compatdata").join(p.run.appid.to_string()).join("pfx/drive_c/users/steamuser");
    if o.dry_run {
        for p in &prepared {
            let current =
                steam::launch_options(&p.game.root.path, &p.run.appid.to_string()).into_iter().find(|a| a.has_app).and_then(|a| a.options).unwrap_or_default();
            println!("{} ({}):", p.run.label, p.facts.name);
            println!("  launch options: {}", sweep_options(&current, &p.facts.exe, &p.log, p.run.hdr, &p.run.env, &p.run.args));
            if let (Some(file), Some(kind)) = (&p.run.edit_file, p.run.edit_kind) {
                let path = user_dir(p).join(file);
                let check = std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| apply_edits(kind, &t, &p.run.edits).map(|_| "applies"));
                println!("  edits {}: {:?} ({})", path.display(), p.run.edits, check.unwrap_or_else(|e| Box::leak(e.into_boxed_str())));
            }
            println!("  expects {} and {:?}", p.run.placement().map_or("?", Placement::describe), p.run.hdr_expect());
        }
        return Ok(true);
    }
    if let Some(vw) = o.vw.as_ref().filter(|v| !v.is_file()) {
        return Err(format!(
            "no screen recorder at {} (--vw PATH; every run is recorded and passes only with a clean picture; --no-record fails every run)",
            vw.display()
        ));
    }
    if !o.cli.is_file() {
        return Err(format!("no CLI at {} (--cli: the build's neural-forge-cli, which reads the channel of the layer under test)", o.cli.display()));
    }
    // SAFETY: installs a handler that only stores to an atomic.
    unsafe {
        for s in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            libc::signal(s, on_signal as *const () as libc::sighandler_t);
        }
    }
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    if Command::new("pgrep").args(["-x", "neural-forge"]).stdout(Stdio::null()).status().is_ok_and(|s| s.success()) {
        return Err("the Neural Forge window is open: close it first (it keeps the channel of the installed version)".into());
    }
    if o.vw.is_some() {
        Recorder::none_running()?;
    }
    let env = session_env();
    let held = std::env::var_os("NF_GPU_LOCK_HELD").is_some() || std::env::var_os("NF_GPU_LOCKED").is_some();
    let mut rows = Vec::new();
    let mut report = Vec::new();
    let mut all_ok = true;
    // One run at a time, each from start to finish under the GPU lock, and everything it changed put back before the
    // lock is let go: other runs on the machine use the same installed layer, channel, Steam and game settings.
    for p in &prepared {
        if cancelled() {
            break;
        }
        let lock = if held { None } else { Some(take_lock(&o.lock)?) };
        if !pids_with_env(&format!("compatdata/{}", p.run.appid)).is_empty() {
            println!("skip {}: {} is running (another run's?)", p.run.label, p.facts.name);
            drop(lock);
            continue;
        }
        let mut restore = Restore {
            out: o.out.join(&p.run.label),
            localconfigs: Vec::new(),
            files: Vec::new(),
            layer: None,
            channel: None,
            steam_was_running: steam_running(),
            steam_stopped: false,
            env: env.clone(),
        };
        let outcome = (|| -> Result<Outcome, String> {
            std::fs::create_dir_all(&restore.out).map_err(|e| format!("{}: {e}", restore.out.display()))?;
            // The build under test in place of the installed layer, and its channel kept apart from the installed one's.
            let installed = PathBuf::from(neural_forge_supervisor::paths::data_home()).join("neural-forge/lib/neural-forge/libneural_forge_layer.so");
            if let Some(layer) = &o.layer {
                let found = std::fs::read(&installed).map_err(|e| format!("{}: {e}", installed.display()))?;
                let new = std::fs::read(layer).map_err(|e| format!("{}: {e}", layer.display()))?;
                println!("installed layer {}; the layer under test {} in place", &sha256(&found)[..12], &sha256(&new)[..12]);
                std::fs::write(restore.out.join("libneural_forge_layer.so.as-found"), &found).map_err(|e| e.to_string())?;
                restore.layer = Some((installed.clone(), found));
                let tmp = installed.with_extension("sweep-tmp");
                std::fs::write(&tmp, &new).and_then(|()| std::fs::rename(&tmp, &installed)).map_err(|e| format!("swapping in {}: {e}", layer.display()))?;
                // SAFETY: getuid(2) cannot fail.
                let channel = PathBuf::from(format!("/tmp/neural-forge-{}", unsafe { libc::getuid() }));
                if channel.exists() {
                    let aside = PathBuf::from(format!("{}-sweep-aside", channel.display()));
                    std::fs::rename(&channel, &aside).map_err(|e| format!("moving {} aside: {e}", channel.display()))?;
                    restore.channel = Some((aside, channel));
                }
            }
            // Launch options: written only while Steam is shut down, put back (and Steam started again) before the
            // lock is let go, so no other run starts the game with them.
            {
                stop_steam(&env)?;
                restore.steam_stopped = true;
                let backup = restore.out.join("localconfig");
                std::fs::create_dir_all(&backup).map_err(|e| e.to_string())?;
                for root in roots.iter().filter(|r| !r.flatpak) {
                    for file in steam::localconfigs(&root.path) {
                        let bytes = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
                        let account =
                            file.parent().and_then(Path::parent).and_then(Path::file_name).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        std::fs::write(backup.join(format!("{account}-localconfig.vdf")), &bytes).map_err(|e| e.to_string())?;
                        restore.localconfigs.push((file, bytes));
                    }
                }
                write_launch_options(&restore.localconfigs, p)?;
            }
            // The run's game settings.
            if let (Some(file), Some(kind)) = (&p.run.edit_file, p.run.edit_kind) {
                let path = user_dir(p).join(file);
                let found = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                // A copy on disk too: a panic aborts and puts nothing back.
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                std::fs::write(restore.out.join(format!("{name}.as-found")), &found).map_err(|e| e.to_string())?;
                restore.files.push((path.clone(), found.clone()));
                let edited = apply_edits(kind, &String::from_utf8_lossy(&found), &p.run.edits).map_err(|e| format!("{}: {e}", path.display()))?;
                std::fs::write(&path, edited).map_err(|e| format!("{}: {e}", path.display()))?;
            }
            start_steam(&env)?;
            restore.steam_stopped = false;
            println!("== {} ({})", p.run.label, p.facts.name);
            let started = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            // No recording, no verdict: the game is not started without the recorder.
            let recorder = Recorder::start(o, p.run.cpu_recorder)?;
            let watched = run_steam(o, p, &env);
            let samples = &watched.samples;
            drop(recorder);
            std::fs::write(o.out.join(format!("{}.samples", p.run.label)), samples.iter().map(|s| format!("{s:?}\n")).collect::<String>()).ok();
            let log = std::fs::read_to_string(&p.log).unwrap_or_default();
            let xid = Command::new("journalctl")
                .args(["-k", "--no-pager", "--since", &format!("@{started}")])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).lines().filter(|l| l.contains("NVRM: Xid")).collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            let mut outcome =
                evaluate(p.run.placement().unwrap_or(p.facts.placement), p.run.hdr_expect(), p.run.frame_generation, watched.window, samples, &log, &xid);
            match if watched.exited_while_recording { Err("no verdict: the game exited during the recording".into()) } else { picture_verdict(o, p) } {
                Ok(v) => outcome.picture = v,
                Err(e) => {
                    outcome.picture = "FAIL".into();
                    outcome.failures.push(format!("the picture: {e}"));
                }
            }
            Ok(outcome)
        })();
        let put_back = restore.put_back();
        drop(lock);
        for line in &put_back {
            println!("restore: {line}");
        }
        report.extend(put_back.into_iter().map(|l| format!("{}: {l}", p.run.label)));
        let outcome =
            outcome.unwrap_or_else(|e| Outcome { reached: "-".into(), failures: vec![format!("the run could not be made: {e}")], ..Outcome::default() });
        for f in &outcome.failures {
            println!("  FAIL: {f}");
        }
        all_ok &= outcome.failures.is_empty();
        rows.push((p, p.run.placement().unwrap_or(p.facts.placement), outcome));
    }
    let mut table = String::new();
    table.push_str(&format!(
        "{:<14} {:<32} {:<18} {:<20} {:>14} {:>6} {:>4} {:>5} {:<18} {:<8} {:<8} result\n",
        "run", "game", "expected", "reached", "holds", "misses", "lost", "FG", "HDR", "faults", "picture"
    ));
    for (p, expect, out) in &rows {
        table.push_str(&format!(
            "{:<14} {:<32} {:<18} {:<20} {:>14} {:>6} {:>4} {:>5} {:<18} {:<8} {:<8} {}\n",
            p.run.label,
            p.facts.name,
            format!("{}{}", short_state(expect.preupscale_state()), if p.run.hdr { ", HDR10" } else { ", SDR" }),
            out.reached,
            out.holds,
            out.misses,
            out.lost,
            out.fg,
            out.hdr,
            out.faults,
            out.picture,
            if out.failures.is_empty() { "pass" } else { "FAIL" }
        ));
    }
    if cancelled() {
        table.push_str("cancelled: the runs after the last row were not made\n");
    }
    println!("\n{table}");
    for line in &report {
        println!("restore: {line}");
    }
    let mut file = std::fs::File::create(o.out.join("results.txt")).map_err(|e| e.to_string())?;
    let _ = writeln!(file, "{table}");
    for (p, _, out) in &rows {
        for f in &out.failures {
            let _ = writeln!(file, "{}: {f}", p.run.label);
        }
    }
    for line in &report {
        let _ = writeln!(file, "restore: {line}");
    }
    let restored = report.iter().all(|l| !l.starts_with("ERROR"));
    Ok(all_ok && restored && !cancelled())
}

/// Sets the run's launch options in every account that has the game, starting from each file as found (so no other
/// run's options stay behind). Steam must be shut down.
fn write_launch_options(localconfigs: &[(PathBuf, Vec<u8>)], p: &Prepared) -> Result<(), String> {
    let appid = p.run.appid.to_string();
    let path = ["UserLocalConfigStore", "Software", "Valve", "Steam", "apps", &appid, "LaunchOptions"];
    for (file, found) in localconfigs {
        let Ok(tree) = vdf::parse(found) else { return Err(format!("{} does not parse", file.display())) };
        let text = if tree.path(&path[..6]).is_none() {
            found.clone()
        } else {
            let current = tree.string_at(&path).unwrap_or_default().to_string();
            let options = sweep_options(&current, &p.facts.exe, &p.log, p.run.hdr, &p.run.env, &p.run.args);
            let (new, _) = vdf::set_string(found, &path, &options).map_err(|e| format!("{}: {e}", file.display()))?;
            if !vdf::verify_edit(found, &new, &path, &options) {
                return Err(format!("{}: the edit did not verify", file.display()));
            }
            println!("{}: launch options {options}", p.run.label);
            new
        };
        std::fs::write(file, &text).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    Ok(())
}

fn status(cli: &Path, secs: u64) -> Option<Sample> {
    let out = Command::new("timeout").arg("20").arg(cli).args(["shmctl", "status"]).stdin(Stdio::null()).output().ok()?;
    parse_status(&String::from_utf8_lossy(&out.stdout), secs)
}

/// What a run's watch saw: the status samples, the recording's span (seconds since the launch) and whether the game
/// was gone while it was being made.
#[derive(Debug, Default)]
struct Watched {
    samples: Vec<Sample>,
    window: Option<(u64, u64)>,
    exited_while_recording: bool,
}

/// Watches a run: status every 5 s until it has held its placement for the run's seconds, the game exits, or the
/// timeout passes without the placement.
fn watch(o: &Options, p: &Prepared, start: Instant, mut done: impl FnMut() -> bool) -> Watched {
    let mut samples = Vec::new();
    let mut window = None;
    let mut exited_while_recording = false;
    let mut recording: Option<Option<std::thread::JoinHandle<()>>> = None;
    let want = p.run.placement().unwrap_or(p.facts.placement).preupscale_state();
    let mut reached: Option<Instant> = None;
    let mut seen = false;
    let mut gone_since: Option<Instant> = None;
    loop {
        nap(Duration::from_secs(5));
        if cancelled() {
            break;
        }
        let secs = start.elapsed().as_secs();
        if let Some(s) = status(&o.cli, secs) {
            if reached.is_none() && s.state == want {
                reached = Some(Instant::now());
                println!("  {}: {} after {secs} s", p.run.label, state_name(want));
            }
            if secs.is_multiple_of(30) || samples.is_empty() {
                println!("  t={secs}s frames={} state={} holds={} misses={}", s.layer_frames, state_name(s.state), s.holds, s.misses);
            }
            samples.push(s);
        }
        if done() {
            println!("  the run ended by itself after {secs} s");
            break;
        }
        let running = game_running(&p.facts.exe);
        seen |= running;
        if !running && matches!(&recording, Some(Some(h)) if !h.is_finished()) {
            exited_while_recording = true;
        }
        if seen && !running {
            let since = *gone_since.get_or_insert_with(Instant::now);
            if since.elapsed() > Duration::from_secs(15) {
                println!("  the game exited after {secs} s");
                break;
            }
        } else {
            gone_since = None;
        }
        if recording.is_none() && reached.is_some_and(|r| r.elapsed() >= Duration::from_secs(p.run.record_after)) {
            recording = Some(record(o, p));
            window = Some((secs, secs + RECORD_SECONDS));
        }
        if reached.is_some_and(|r| r.elapsed() >= Duration::from_secs(p.run.seconds)) {
            break;
        }
        if reached.is_none() && secs >= p.run.timeout {
            println!("  no {} within {} s", state_name(want), p.run.timeout);
            break;
        }
    }
    // The recording finishes while the game still runs (it is ended after the watch).
    if let Some(Some(h)) = recording {
        while !h.is_finished() {
            std::thread::sleep(Duration::from_millis(500));
            if !game_running(&p.facts.exe) {
                exited_while_recording = true;
            }
        }
        let _ = h.join();
    }
    Watched { samples, window, exited_while_recording }
}

/// The run's recording ([`RECORD_SECONDS`] of the screen, while the game is at its placement).
fn video(o: &Options, p: &Prepared) -> PathBuf {
    o.out.join(format!("{}.mp4", p.run.label))
}

/// The screen recorder, started before the game (`vw replay start`: GPU Screen Recorder keeping the last seconds in
/// memory, so it has its video memory before the game takes it all) and stopped after it (`vw replay stop`, also
/// after an error or Ctrl-C), with its files in `<out>/recorder` (`VW_DIR`). The GPU encoder takes about 310 MiB. At
/// 4K with Ray Reconstruction and the model after DLSS that was more than there was (no CUDA context once the game
/// was up; started before it, the VRAM guard paused the model); runs are at 1440p since. `recorder = "cpu"` encodes
/// on the CPU (`VW_ENCODER=cpu`, about 20 MiB), which keeps fewer frames (about 115 of 150 a second at 1920x1080)
/// than vw's thresholds were set with, so only where the GPU encoder can't be. Neither keeps the model running by
/// itself: the game's own memory grows during a benchmark (the run's `lost` column).
struct Recorder {
    vw: PathBuf,
    dir: PathBuf,
}

impl Recorder {
    /// Before the first run: a GPU Screen Recorder already running is an error (a second one fails to capture). The
    /// sweep does not stop it: it is someone else's (a recording another test is making).
    fn none_running() -> Result<(), String> {
        let found =
            Command::new("pgrep").args(["-a", "-x", "gpu-screen-reco"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let Some(line) = found.lines().next() else { return Ok(()) };
        let pid = line.split_whitespace().next().unwrap_or("?");
        let out = line.split(" -o ").nth(1).and_then(|r| r.split_whitespace().next()).unwrap_or("?");
        Err(format!("a GPU Screen Recorder is running (pid {pid}, writing {out}); a second one fails to capture: wait for it or stop it first"))
    }

    /// `None` with `--no-record`; an error (the run is not made) when the recorder does not start.
    fn start(o: &Options, cpu: bool) -> Result<Option<Recorder>, String> {
        let Some(vw) = o.vw.clone() else { return Ok(None) };
        let r = Recorder { vw, dir: recorder_dir(o) };
        let secs = (RECORD_SECONDS + 2).to_string();
        match r.vw(&["replay", "start", &secs]).env("VW_ENCODER", if cpu { "cpu" } else { "gpu" }).status() {
            Ok(s) if s.success() => Ok(Some(r)),
            Ok(s) => Err(format!("the recorder did not start ({s}; see {})", r.dir.join("replay.log").display())),
            Err(e) => Err(format!("vw replay start: {e}")),
        }
    }

    fn vw(&self, args: &[&str]) -> Command {
        let mut c = Command::new(&self.vw);
        c.args(args).env("VW_DIR", &self.dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        c
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.vw(&["replay", "stop"]).status();
    }
}

/// Where the run's recorder keeps its buffer, pid and log (`VW_DIR`).
fn recorder_dir(o: &Options) -> PathBuf {
    o.out.join("recorder")
}

/// Records the screen for the run ([`video`]) on a thread, so the watch goes on sampling: the next
/// [`RECORD_SECONDS`], saved from the [`Recorder`]'s memory (`vw replay save`), with the recorder's log so far next
/// to it.
fn record(o: &Options, p: &Prepared) -> Option<std::thread::JoinHandle<()>> {
    let vw = o.vw.clone()?;
    let out = video(o, p);
    let dir = recorder_dir(o);
    let label = p.run.label.clone();
    Some(std::thread::spawn(move || {
        let _ = std::fs::remove_file(&out);
        nap(Duration::from_secs(RECORD_SECONDS));
        // The recorder keeps more than it was asked for (up to twice): the last RECORD_SECONDS of what it saved, cut
        // at a keyframe without re-encoding, is the recording.
        let full = out.with_extension("full.mp4");
        let _ = std::fs::remove_file(&full);
        let quiet = |c: &mut Command| {
            c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        };
        let mut save = Command::new(&vw);
        save.args(["replay", "save"]).arg(&full).env("VW_DIR", &dir);
        quiet(&mut save);
        let saved = save.status();
        let log = out.with_extension("mp4.log");
        let _ = std::fs::copy(dir.join("replay.log"), &log);
        let saved = saved.and_then(|s| {
            if !s.success() {
                return Ok(s);
            }
            let mut cut = Command::new("ffmpeg");
            cut.args(["-nostdin", "-v", "error", "-y", "-sseof"]).arg(format!("-{RECORD_SECONDS}")).arg("-i").arg(&full).args(["-c", "copy"]).arg(&out);
            quiet(&mut cut);
            let cut = cut.status();
            // The whole buffer stays when the cut fails, to be looked at.
            if cut.as_ref().is_ok_and(|c| c.success()) {
                let _ = std::fs::remove_file(&full);
            }
            cut
        });
        match saved {
            Ok(s) if s.success() && out.is_file() => println!("  {label}: recorded {RECORD_SECONDS} s ({})", out.display()),
            Ok(s) => println!("  {label}: the recording failed ({s}; see {} and {})", log.display(), full.display()),
            Err(e) => println!("  {label}: the recording failed: {e}"),
        }
    }))
}

/// The run's picture: `vw analyze` over its recording, and the flags it raised (a colour flash, frames alternating
/// between two states, black frames, freezes). "ok" only with none; the contact sheet (`<label>-vw/sheet.jpg`) is
/// there to be looked at as well, against the model off.
fn picture_verdict(o: &Options, p: &Prepared) -> Result<String, String> {
    let Some(vw) = &o.vw else { return Err("not recorded (--no-record)".into()) };
    let video = video(o, p);
    if !video.is_file() {
        return Err("not recorded (the placement was never reached, or the recorder failed: <label>.mp4.log)".into());
    }
    let dir = o.out.join(format!("{}-vw", p.run.label));
    let analysed = Command::new(vw).arg("analyze").arg(&video).arg("--out").arg(&dir).stdin(Stdio::null()).output().map_err(|e| format!("vw analyze: {e}"))?;
    if !analysed.status.success() {
        return Err(format!("vw analyze failed: {}", String::from_utf8_lossy(&analysed.stderr).trim()));
    }
    let summary = std::fs::read_to_string(dir.join("summary.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let flags = vw_flags(&summary).ok_or("vw's summary has no flags")?;
    if flags.is_empty() {
        Ok("ok".into())
    } else {
        Err(flags.join("; "))
    }
}

/// The `flags` list of a `vw analyze` summary.json (strings), if it has one.
fn vw_flags(summary: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(summary).ok()?;
    Some(v.get("flags")?.as_array()?.iter().filter_map(|f| f.as_str().map(str::to_string)).collect())
}

fn run_steam(o: &Options, p: &Prepared, env: &[(String, String)]) -> Watched {
    let _ = std::fs::remove_file(&p.log);
    let start = Instant::now();
    let launched = session_command("steam", env).arg(format!("steam://rungameid/{}", p.run.appid)).stdout(Stdio::null()).stderr(Stdio::null()).status();
    if let Err(e) = launched {
        println!("  steam: {e}");
        return Watched::default();
    }
    let watched = watch(o, p, start, || false);
    end_game(p.run.appid);
    nap(Duration::from_secs(10));
    watched
}

fn self_test() -> std::process::ExitCode {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = std::fs::read_to_string(repo.join("scripts/game-sweep.toml")).expect("scripts/game-sweep.toml");
    match parse_config(&config) {
        Ok(runs) => println!("game-sweep self-test: {} runs in scripts/game-sweep.toml", runs.len()),
        Err(e) => {
            eprintln!("scripts/game-sweep.toml: {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(secs: u64, state: u32, holds: u64, misses: u64, hdr: &str) -> Sample {
        Sample { secs, layer_frames: secs * 60, state, reason: String::new(), misses, holds, device_lost: 0, hdr: hdr.into(), ngx_matched: secs * 20 }
    }

    #[test]
    fn the_committed_config_parses() {
        let config = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/game-sweep.toml")).unwrap();
        let runs = parse_config(&config).unwrap();
        assert!(runs.iter().any(|r| r.appid == 1091500 && r.hdr), "Cyberpunk with HDR");
        let gta = runs.iter().find(|r| r.appid == 3240220).expect("GTA's benchmark");
        assert!(gta.args.contains("-benchmark") && gta.frame_generation, "GTA's benchmark with frame generation");
        assert!(gta.edits.iter().any(|(k, v)| k == "FrameLimit" && v == "120"), "frame generation needs GTA's frame limit (docs/LESSONS.md)");
    }

    #[test]
    fn config_mistakes_are_refused() {
        let ok = "[[run]]\nlabel = \"cp\"\nappid = 1091500\n";
        assert_eq!(parse_config(ok).unwrap()[0].seconds, 90);
        assert!(parse_config(&format!("{ok}hdr = 1\n")).unwrap_err().contains("hdr"));
        assert!(parse_config(&ok.replace("1091500", "42")).unwrap_err().contains("not in data/games.toml"));
        assert!(parse_config(&format!("{ok}{ok}")).unwrap_err().contains("used twice"));
        assert!(parse_config(&format!("{ok}route = \"gta-bench\"\n")).unwrap_err().contains("unknown key route"), "every game is started by Steam");
        assert!(parse_config(&format!("{ok}frame_generation = true\n")).unwrap()[0].frame_generation);
        assert!(parse_config(&format!("{ok}edits = [\"A=1\"]\n")).unwrap_err().contains("go together"));
        assert_eq!(parse_config(&format!("{ok}record_after = 80\n")).unwrap()[0].record_after, 80);
        assert!(parse_config(&format!("{ok}recorder = \"cpu\"\n")).unwrap()[0].cpu_recorder);
        assert!(!parse_config(ok).unwrap()[0].cpu_recorder);
        assert!(parse_config(&format!("{ok}recorder = \"nvenc\"\n")).unwrap_err().contains("recorder"));
        assert!(parse_config(&format!("{ok}record_after = 85\n")).unwrap_err().contains("must fit"), "the recording ends after the run");
    }

    #[test]
    fn status_text_is_read() {
        let text = "# live status\nlayer_frames=4535\npreupscale_state=2 (holding before the upscaler)\npreupscale_reason=\npreupscale_misses=0\n\
                    hdr_verdict=game outputs HDR10 (swapchain A2B10G10R10_UNORM_PACK32 HDR10_ST2084_EXT); desktop: GNOME: HDMI-1 HDR on\n\
                    device_lost_at=0\nedit_cache: refresh_holds=30 cached_holds=29 rejected=1.12% gpu_ms=0.000\n";
        let s = parse_status(text, 5).unwrap();
        assert_eq!((s.layer_frames, s.state, s.misses, s.holds), (4535, 2, 0, 59));
        assert!(s.hdr.starts_with("game outputs HDR10"));
        assert!(parse_status("no channel at /tmp/x\n", 0).is_none());
    }

    /// vw's verdict: its flags, none for a clean picture, and nothing from a summary without them.
    #[test]
    fn the_recordings_flags_are_read() {
        assert_eq!(vw_flags(r#"{"frames": 480, "flags": []}"#), Some(vec![]));
        let flagged = vw_flags(r#"{"flags": ["COLOUR_FLASH: 31 frames tinted against their neighbours", "BLACK: 2 black frames"]}"#).unwrap();
        assert_eq!(flagged.len(), 2);
        assert!(flagged[0].starts_with("COLOUR_FLASH"));
        assert_eq!(vw_flags(r#"{"frames": 480}"#), None);
        assert_eq!(vw_flags("not json"), None);
    }

    #[test]
    fn a_run_passes_only_when_every_check_does() {
        use preupscale_state::*;
        let hdr10 = "game outputs HDR10 (swapchain ...)";
        let good = [sample(5, WAITING, 0, 0, ""), sample(10, HOLDING, 5, 0, hdr10), sample(60, HOLDING, 900, 0, hdr10)];
        let o = evaluate(Placement::BeforeInside, HdrExpect::Hdr10, false, None, &good, "fine\n", "");
        assert!(o.failures.is_empty(), "{:?}", o.failures);
        assert_eq!((o.lost.as_str(), o.fg.as_str()), ("0", "x3.0"));
        assert_eq!((o.reached.as_str(), o.holds.as_str(), o.hdr.as_str(), o.faults.as_str()), ("before after 10 s", "900 (+895)", "HDR10", "none"));
        // SDR where HDR10 was expected; a miss; a panic; an Xid.
        let o = evaluate(
            Placement::BeforeInside,
            HdrExpect::Hdr10,
            false,
            None,
            &good.map(|mut s| {
                s.hdr = "game asked for SDR although HDR metadata was sent: ...".into();
                s.misses = 2;
                s
            }),
            "[panic] the layer panicked\n",
            "NVRM: Xid (PCI:0000:01:00): 109",
        );
        assert_eq!(o.failures.len(), 4, "{:?}", o.failures);
        assert_eq!(o.faults, "panic, Xid");
        // Never reached, or refused at the end.
        let refused = [sample(10, IDENTIFIED, 0, 0, "game outputs SDR (x)")];
        let o = evaluate(Placement::Before, HdrExpect::Sdr, false, None, &refused, "", "");
        assert!(o.failures[0].starts_with("expected before the upscaler"), "{:?}", o.failures);
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, false, None, &[], "", "").failures[0].starts_with("no status"));
    }

    /// Reached, then lost (the VRAM guard pausing the model, as Cyberpunk 2077 with Ray Reconstruction at 4K did on a 12 GB card):
    /// the run fails even though it was reached and nothing missed; the game's own pauses and one stray sample don't.
    #[test]
    fn a_placement_lost_after_reaching_it_fails() {
        use preupscale_state::*;
        let sdr = "game outputs SDR (x)";
        let mut paused_by_guard = vec![sample(40, HOLDING_AFTER, 58, 0, sdr), sample(66, HOLDING_AFTER, 507, 0, sdr)];
        for t in [71, 76, 81, 86] {
            let mut s = sample(t, if t == 76 { WAITING } else { IDENTIFIED }, 510, 0, sdr);
            s.reason = "the network is paused: too little video memory for it (VRAM guard); it resumes once there is room".into();
            paused_by_guard.push(s);
        }
        let o = evaluate(Placement::AfterDlss, HdrExpect::Sdr, false, None, &paused_by_guard, "", "");
        assert_eq!(o.lost, "4");
        assert!(
            o.failures.iter().any(|f| f.contains("lost the placement in 4 samples") && f.contains("from 71 s") && f.contains("VRAM guard")),
            "{:?}",
            o.failures
        );
        // GTA's loading screens between benchmark passes (the game's pause) and one stray sample pass.
        let gta = [
            sample(100, HOLDING, 32, 0, sdr),
            sample(105, PAUSED, 67, 0, sdr),
            sample(110, HOLDING, 184, 0, sdr),
            sample(115, IDENTIFIED, 184, 0, sdr),
            sample(120, HOLDING, 300, 0, sdr),
        ];
        let o = evaluate(Placement::Before, HdrExpect::Sdr, false, None, &gta, "", "");
        assert!(o.failures.is_empty(), "{:?}", o.failures);
        assert_eq!(o.lost, "1");
    }

    /// The kept set is the run's own placement or the game's pause: the finished frame (state off) is kept, after DLSS
    /// falling back to before the upscaler is not.
    #[test]
    fn only_the_runs_own_placement_is_kept() {
        use preupscale_state::*;
        let sdr = "game outputs SDR (x)";
        let finished = [sample(10, OFF, 0, 0, sdr), sample(15, PAUSED, 0, 0, sdr), sample(20, OFF, 0, 0, sdr), sample(25, OFF, 0, 0, sdr)];
        let o = evaluate(Placement::FinishedFrame, HdrExpect::Sdr, false, Some((20, 30)), &finished, "", "");
        assert!(o.failures.is_empty(), "{:?}", o.failures);
        assert_eq!(o.lost, "0");
        let fell_back = [sample(10, HOLDING_AFTER, 5, 0, sdr), sample(15, HOLDING, 50, 0, sdr), sample(20, HOLDING, 90, 0, sdr)];
        let o = evaluate(Placement::AfterDlss, HdrExpect::Sdr, false, None, &fell_back, "", "");
        assert_eq!(o.lost, "2");
        assert!(o.failures.iter().any(|f| f.starts_with("lost the placement")), "{:?}", o.failures);
    }

    /// The picture is judged only with the model running at its placement over the recording, holds growing.
    #[test]
    fn the_recording_needs_the_model_running() {
        use preupscale_state::*;
        let sdr = "game outputs SDR (x)";
        let run = |states: [u32; 5], holds: [u64; 5]| -> Vec<Sample> { (0..5).map(|k| sample(40 + 5 * k as u64, states[k], holds[k], 0, sdr)).collect() };
        let fine = run([HOLDING; 5], [10, 60, 110, 160, 210]);
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, false, Some((45, 55)), &fine, "", "").failures.is_empty());
        // Paused (a loading screen) during the recording: allowed elsewhere, not over it.
        let paused = run([HOLDING, HOLDING, PAUSED, HOLDING, HOLDING], [10, 60, 60, 110, 160]);
        let o = evaluate(Placement::Before, HdrExpect::Sdr, false, Some((45, 55)), &paused, "", "");
        assert!(o.failures.iter().any(|f| f.starts_with("the recording was not made with the model at its placement (50 s")), "{:?}", o.failures);
        // At the placement but nothing held.
        let idle = run([HOLDING; 5], [60, 60, 60, 60, 60]);
        let o = evaluate(Placement::Before, HdrExpect::Sdr, false, Some((45, 55)), &idle, "", "");
        assert!(o.failures.iter().any(|f| f.contains("holds did not grow")), "{:?}", o.failures);
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, false, Some((90, 100)), &fine, "", "").failures.iter().any(|f| f.starts_with("no status sample")));
    }

    /// Frame generation from the layer's presents per DLSS evaluate while holding.
    #[test]
    fn frame_generation_is_measured_and_checked() {
        use preupscale_state::*;
        let sdr = "game outputs SDR (x)";
        let at =
            |secs: u64, state: u32, frames: u64, evaluates: u64| Sample { layer_frames: frames, ngx_matched: evaluates, ..sample(secs, state, 10, 0, sdr) };
        // GTA with frame generation off: 50 presents and 50 evaluates a second.
        let off = [at(100, HOLDING, 0, 0), at(105, HOLDING, 250, 250), at(110, HOLDING, 500, 500)];
        let o = evaluate(Placement::Before, HdrExpect::Sdr, true, None, &off, "", "");
        assert_eq!(o.fg, "x1.0");
        assert!(o.failures.iter().any(|f| f.starts_with("frame generation is off")), "{:?}", o.failures);
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, false, None, &off, "", "").failures.is_empty(), "not expected, not checked");
        // On (x4); a loading screen between (paused, presents with no evaluate) does not count.
        let on = [at(100, HOLDING, 0, 0), at(105, HOLDING, 800, 200), at(110, PAUSED, 2000, 200), at(115, HOLDING, 2800, 400), at(120, HOLDING, 3600, 600)];
        let o = evaluate(Placement::Before, HdrExpect::Sdr, true, None, &on, "", "");
        assert!(o.failures.is_empty(), "{:?}", o.failures);
        assert_eq!(o.fg, "x4.0");
        // Not believed above 4.5x, and intervals with a handful of evaluates don't count.
        let wild = [at(100, HOLDING, 0, 0), at(105, HOLDING, 3000, 500)];
        assert_eq!(evaluate(Placement::Before, HdrExpect::Sdr, false, None, &wild, "", "").fg, "-");
        let few = [at(100, HOLDING, 0, 0), at(105, HOLDING, 800, 10)];
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, true, None, &few, "", "").failures.iter().any(|f| f.contains("not measured")));
        // Never measured (no evaluate while holding): expected, so it fails.
        assert!(evaluate(Placement::Before, HdrExpect::Sdr, true, None, &[at(5, HOLDING, 10, 0)], "", "").failures.iter().any(|f| f.contains("not measured")));
    }

    #[test]
    fn the_launch_options_carry_the_run() {
        let log = Path::new("/home/a/nf-sweep/1/cp.log");
        let exe = vec!["Cyberpunk2077.exe".to_string()];
        let found =
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG=/x/cp.log NEURAL_FORGE_LOG_TIME=1 %command% --launcher-skip";
        assert_eq!(
            sweep_options(found, &exe, log, true, &[], "--launcher-skip -skipStartScreen -benchmark"),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1 %command% --launcher-skip -skipStartScreen -benchmark"
        );
        assert_eq!(
            sweep_options("", &exe, log, false, &[], ""),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' %command%"
        );
        // The run's own variables go in front of %command%, in place of the game's own of the same name.
        assert_eq!(
            sweep_options("NEURAL_FORGE_OFFLOAD_GPU=1 %command%", &exe, log, false, &["NEURAL_FORGE_OFFLOAD_GPU=same".to_string()], ""),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' NEURAL_FORGE_OFFLOAD_GPU=same %command%"
        );
        // Three or more of the sweep's own variables in the game's options (dropped and set again): no slicing past
        // the end, and the run's arguments go after %command%.
        let many = "NEURAL_FORGE_LOG=/a NEURAL_FORGE_LOG_TIME=1 PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1 %command%";
        assert_eq!(
            sweep_options(many, &exe, log, false, &[], "-benchmark"),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' %command% -benchmark"
        );
        // The game's own value for the same switch gives way to the run's; a negative number is a value.
        assert_eq!(
            sweep_options("%command% -benchmarkIterations 3 -x", &exe, log, false, &[], "-benchmarkIterations 1 -offset -5"),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=Cyberpunk2077.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' %command% -benchmarkIterations 1 -x -offset -5"
        );
        assert!(is_switch("-x") && is_switch("--launcher-skip") && !is_switch("-5") && !is_switch("-.5") && !is_switch("-") && !is_switch("5"));
        // A switch and its value are one unit: `1` is kept although another `1` is there already.
        let gta = vec!["PlayGTAV.exe".to_string()];
        assert_eq!(
            sweep_options("%command% -foo 1", &gta, log, false, &[], "-benchmark -benchmarkIterations 1 -benchmarkFrameTimes"),
            "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=PlayGTAV.exe NEURAL_FORGE_LOG='/home/a/nf-sweep/1/cp.log' %command% -foo 1 -benchmark -benchmarkIterations 1 -benchmarkFrameTimes"
        );
    }

    #[test]
    fn a_prefix_is_named_only_as_a_whole_path_part() {
        assert!(names_path(b"STEAM_COMPAT_DATA_PATH=/s/steamapps/compatdata/1091500", b"compatdata/1091500"));
        assert!(names_path(b"WINEPREFIX=/s/steamapps/compatdata/1091500/pfx", b"compatdata/1091500"));
        assert!(!names_path(b"STEAM_COMPAT_DATA_PATH=/s/steamapps/compatdata/10915000", b"compatdata/1091500"));
    }

    #[test]
    fn game_settings_are_edited_in_place() {
        let xml = "<Settings>\n  <ScreenWidth value=\"2560\" />\n  <ScreenHeight value=\"1440\" />\n</Settings>\n";
        let edited = apply_edits(EditKind::XmlValue, xml, &[("ScreenWidth".into(), "3840".into()), ("ScreenHeight".into(), "2160".into())]).unwrap();
        assert_eq!(edited, xml.replace("2560", "3840").replace("1440", "2160"));
        assert!(apply_edits(EditKind::XmlValue, xml, &[("Missing".into(), "1".into())]).is_err());
        let json = r#"{"version": 2, "data": [{"group_name": "/video/display", "options": [{"name": "Resolution", "type": "string_list", "value": "2560x1440"}, {"name": "DLSS_D", "type": "bool", "value": true}]}]}"#;
        let edited = apply_edits(EditKind::JsonOption, json, &[("Resolution".into(), "3840x2160".into()), ("DLSS_D".into(), "false".into())]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&edited).unwrap();
        assert_eq!(v["data"][0]["options"][0]["value"], "3840x2160");
        assert_eq!(v["data"][0]["options"][1]["value"], false);
        assert!(apply_edits(EditKind::JsonOption, json, &[("HDRModes".into(), "HDR10_PQ".into())]).is_err());
    }
}
