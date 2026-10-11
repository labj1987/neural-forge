//! `neural-forge-cli doctor` and the GUI's Diagnose button: every check that can name why neural
//! rendering is not running, shared by both front ends.
//!
//! Each check reads one thing (a file, a header field, a command's output), and every finding says
//! what it read (`evidence`) and one concrete fix. Everything a check reads is a field of [`Roots`]:
//! [`Roots::system`] names the real paths and runs the real commands, and the tests build the same
//! tree in a temporary directory and hand in the commands' text, so no test reads the real XDG
//! dirs, `/proc`, the kernel log or the GPU.
//!
//! Logs are read by their own grammar: the layer's state log through
//! `neural_forge_protocol::state_log` (exact kinds, the newest matching line is the verdict), the
//! kernel log by the driver's exact `NVRM: Xid` prefix.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use neural_forge_protocol::state_log;

/// How bad a finding is. Ordered, so the worst sorts first ([`sort_worst_first`]); only `Failure`
/// makes `doctor` exit non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Ok,
    Info,
    Warning,
    Failure,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Ok => "ok",
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Failure => "FAILURE",
        }
    }
}

/// One check's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// A short id for the check (`manifest`, `xid`, ...).
    pub check: &'static str,
    pub severity: Severity,
    pub summary: String,
    /// What the check read: a path, a line, a field.
    pub evidence: String,
    /// One concrete thing to do about it.
    pub fix: Option<String>,
}

impl Finding {
    fn new(check: &'static str, severity: Severity, summary: impl Into<String>, evidence: impl Into<String>, fix: Option<String>) -> Self {
        Self { check, severity, summary: summary.into(), evidence: evidence.into(), fix }
    }
}

/// What running an external command gave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandOutput {
    /// It ran and succeeded: its standard output.
    Ran(String),
    /// It ran and failed, or could not be run for another reason than being absent: why.
    Failed(String),
    /// It is not installed.
    Missing,
}

/// Runs an external command with the given arguments.
pub type Command = Box<dyn Fn(&[&str]) -> CommandOutput>;

/// Whether the game's Steam launch options carry `NEURAL_FORGE_ENABLE=1 %command%`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaunchOptionState {
    Present,
    /// Missing for the named game.
    Missing(String),
}

/// Every root path and every command the checks read, so tests can drive them from fixtures.
pub struct Roots {
    /// `/proc`: `driver/nvidia/gpus/*/information`, and `self/status` for the groups.
    pub proc_root: PathBuf,
    /// `/sys`: `module/nvidia/version`.
    pub sys_root: PathBuf,
    /// `/etc/group`, for the `input` group's id.
    pub group_file: PathBuf,
    pub config_file: PathBuf,
    /// The raw `XDG_DATA_HOME`: the layer manifest lives in its `vulkan/implicit_layer.d`.
    pub data_home: PathBuf,
    pub data_dir: PathBuf,
    /// `$XDG_STATE_HOME/neural-forge`: the layer's state log.
    pub state_dir: PathBuf,
    pub config_dir: PathBuf,
    pub binaries_dir: PathBuf,
    pub model_dir: PathBuf,
    /// The channel (`channel_path`).
    pub shm_path: PathBuf,
    /// `nvidia-smi`.
    pub nvidia_smi: Command,
    /// `journalctl`.
    pub journalctl: Command,
    /// Whether the launch option is set, when something knows; `None` is unknown.
    pub launch_option_state: Option<LaunchOptionState>,
}

/// Runs `program` with `args`, its standard output on success.
fn run_command(program: &str, args: &[&str]) -> CommandOutput {
    match std::process::Command::new(program).args(args).stdin(std::process::Stdio::null()).output() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CommandOutput::Missing,
        Err(e) => CommandOutput::Failed(e.to_string()),
        Ok(out) if out.status.success() => CommandOutput::Ran(String::from_utf8_lossy(&out.stdout).into_owned()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let first = stderr.lines().find(|l| !l.trim().is_empty()).unwrap_or("no message");
            CommandOutput::Failed(format!("{} ({first})", out.status))
        }
    }
}

impl Roots {
    /// The real system: this user's XDG dirs, `/proc`, `/sys`, the configured channel, and the real
    /// `nvidia-smi` and `journalctl`.
    pub fn system() -> Self {
        let cfg = crate::Config::load();
        let binaries = if cfg.binaries.is_empty() { crate::paths::binaries_dir() } else { cfg.binaries.clone() };
        Self {
            proc_root: "/proc".into(),
            sys_root: "/sys".into(),
            group_file: "/etc/group".into(),
            config_file: crate::paths::config_file().into(),
            data_home: crate::paths::data_home().into(),
            data_dir: crate::paths::data_dir().into(),
            state_dir: crate::paths::state_dir().into(),
            config_dir: crate::paths::config_dir().into(),
            binaries_dir: binaries.into(),
            model_dir: crate::model::model_dir().into(),
            shm_path: crate::channel_path(&cfg).into(),
            nvidia_smi: Box::new(|args| run_command("nvidia-smi", args)),
            journalctl: Box::new(|args| run_command("journalctl", args)),
            launch_option_state: None,
        }
    }

    fn manifest_path(&self) -> PathBuf {
        self.data_home.join("vulkan/implicit_layer.d").join(crate::install::MANIFEST)
    }
}

/// Every finding, worst first.
pub fn run(roots: &Roots) -> Vec<Finding> {
    let mut out = vec![check_config(roots), check_ngx_dll(roots), check_model(roots), check_user_dirs(roots)];
    let manifest = check_manifest(roots);
    let manifest_ok = manifest.severity == Severity::Ok;
    out.push(manifest);
    out.extend(check_gpu(roots));
    let gpus = query_gpus(roots);
    out.extend(check_compute_capability(&gpus));
    let events = state_log::read_dir_events(&roots.state_dir);
    let session = events.as_ref().ok().and_then(|e| last_session(e));
    match neural_forge_protocol::mapping::open_path(&roots.shm_path.to_string_lossy()) {
        Ok(mapping) => {
            let hdr = mapping.header();
            out.push(Finding::new("channel", Severity::Ok, "the settings channel opens", roots.shm_path.display().to_string(), None));
            out.extend(check_vram(&gpus, Some(hdr)));
            out.push(check_attached(roots, hdr, manifest_ok));
            out.extend(check_device_lost(Some(hdr), session.as_deref()));
            out.extend(check_cannot_run(Some(hdr), session.as_deref()));
            out.extend(check_native_status(hdr));
        }
        Err(e) => {
            out.push(Finding::new(
                "channel",
                Severity::Failure,
                format!("the settings channel cannot be opened: {e}"),
                roots.shm_path.display().to_string(),
                Some("Close any game still running an older Neural Forge, then run the check again.".into()),
            ));
            out.extend(check_vram(&gpus, None));
            out.extend(check_device_lost(None, session.as_deref()));
            out.extend(check_cannot_run(None, session.as_deref()));
        }
    }
    out.extend(check_xid(roots));
    out.extend(check_input_group(roots));
    out.extend(check_session_log(roots, events.as_deref().map_err(|e| e.kind()), session.as_deref()));
    sort_worst_first(&mut out);
    out
}

/// Worst first; checks of one severity keep the order they ran in.
pub fn sort_worst_first(findings: &mut [Finding]) {
    findings.sort_by_key(|f| std::cmp::Reverse(f.severity));
}

/// Whether any finding is a real failure (`doctor`'s exit status).
pub fn any_failure(findings: &[Finding]) -> bool {
    findings.iter().any(|f| f.severity == Severity::Failure)
}

/// The findings as text, in the order given, each with its evidence and fix, and a count at the end.
pub fn render_text(findings: &[Finding]) -> String {
    let mut s = String::new();
    for f in findings {
        s.push_str(&format!("[{}] {}: {}\n", f.severity.label(), f.check, f.summary));
        for line in f.evidence.lines() {
            s.push_str(&format!("    {line}\n"));
        }
        if let Some(fix) = &f.fix {
            s.push_str(&format!("    fix: {fix}\n"));
        }
    }
    let count = |sev| findings.iter().filter(|f| f.severity == sev).count();
    s.push_str(&format!(
        "\n{} failure(s), {} warning(s), {} note(s), {} ok\n",
        count(Severity::Failure),
        count(Severity::Warning),
        count(Severity::Info),
        count(Severity::Ok)
    ));
    s
}

// ---- the five original checks ------------------------------------------------------------------

fn check_config(roots: &Roots) -> Finding {
    let path = roots.config_file.display().to_string();
    if roots.config_file.exists() {
        Finding::new("config", Severity::Ok, "the config file exists", path, None)
    } else {
        Finding::new("config", Severity::Failure, "the config file is missing", path, Some("Run `neural-forge-cli init`.".into()))
    }
}

fn check_ngx_dll(roots: &Roots) -> Finding {
    let dll = roots.binaries_dir.join("nvngx_dlssnr.dll");
    if dll.exists() {
        Finding::new("nvngx-dll", Severity::Ok, "NVIDIA's nvngx_dlssnr.dll is imported", dll.display().to_string(), None)
    } else {
        Finding::new(
            "nvngx-dll",
            Severity::Info,
            "nvngx_dlssnr.dll is not imported (only needed to extract the model)",
            dll.display().to_string(),
            Some("Only if the model is missing too: `neural-forge-cli import-binaries DIR`.".into()),
        )
    }
}

fn check_model(roots: &Roots) -> Finding {
    let dir = roots.model_dir.display().to_string();
    match crate::model::installed(&roots.model_dir) {
        Some(build) => {
            let note = if crate::model::installed_verified(&roots.model_dir) == Some(false) { ", not verified against NVIDIA's runtime" } else { "" };
            Finding::new("model", Severity::Ok, format!("the model is extracted (build {build}{note})"), dir, None)
        }
        None => Finding::new(
            "model",
            Severity::Failure,
            "the model is not extracted: the network has nothing to load",
            format!("no readable manifest.json in {dir}"),
            Some("Extract it in the Setup tab, or run `neural-forge-cli extract-model DIR` with the folder that holds nvngx_dlssnr.dll.".into()),
        ),
    }
}

fn check_user_dirs(roots: &Roots) -> Finding {
    let dirs = [&roots.config_dir, &roots.data_dir, &roots.state_dir, &roots.binaries_dir];
    let evidence = dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ");
    match dirs.iter().try_for_each(std::fs::create_dir_all) {
        Ok(()) => Finding::new("user-dirs", Severity::Ok, "the user dirs are writable", evidence, None),
        Err(e) => Finding::new(
            "user-dirs",
            Severity::Failure,
            format!("the user dirs are not writable: {e}"),
            evidence,
            Some("Make them owned by and writable for your user (`ls -ld` each to see which).".into()),
        ),
    }
}

// ---- the layer manifest --------------------------------------------------------------------------

fn check_manifest(roots: &Roots) -> Finding {
    const CHECK: &str = "manifest";
    let path = roots.manifest_path();
    let shown = path.display().to_string();
    let install = || Some("Start the Neural Forge AppImage once: it installs the layer for your user.".to_string());
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Finding::new(CHECK, Severity::Failure, "the Vulkan layer is not installed: no game can load it", format!("{shown} does not exist"), install());
        }
        Err(e) => return Finding::new(CHECK, Severity::Failure, format!("the layer manifest cannot be read: {e}"), shown, install()),
    };
    let json: serde_json::Value = match serde_json::from_str(&text) {
        Ok(json) => json,
        Err(e) => return Finding::new(CHECK, Severity::Failure, "the layer manifest is not valid JSON: the Vulkan loader ignores it", format!("{shown}: {e}"), install()),
    };
    let Some(library) = json["layer"]["library_path"].as_str() else {
        return Finding::new(CHECK, Severity::Failure, "the layer manifest names no library", format!("{shown}: no layer.library_path"), install());
    };
    if !Path::new(library).is_file() {
        return Finding::new(
            CHECK,
            Severity::Failure,
            "the layer manifest points at a library that does not exist",
            format!("{shown}: library_path {library}"),
            install(),
        );
    }
    Finding::new(CHECK, Severity::Ok, "the Vulkan layer is installed", format!("{shown} -> {library}"), None)
}

// ---- the GPU and driver --------------------------------------------------------------------------

/// The "Model:" line of every `driver/nvidia/gpus/*/information` under `proc_root`, sorted by bus.
fn gpu_models(proc_root: &Path) -> Vec<(PathBuf, String)> {
    let mut found = Vec::new();
    let Ok(dir) = std::fs::read_dir(proc_root.join("driver/nvidia/gpus")) else { return found };
    for entry in dir.flatten() {
        let info = entry.path().join("information");
        let Ok(text) = std::fs::read_to_string(&info) else { continue };
        if let Some(model) = text.lines().find_map(|l| l.strip_prefix("Model:")) {
            found.push((info, model.trim().to_string()));
        }
    }
    found.sort();
    found
}

fn check_gpu(roots: &Roots) -> Vec<Finding> {
    let version_file = roots.sys_root.join("module/nvidia/version");
    let Ok(version) = std::fs::read_to_string(&version_file) else {
        return vec![Finding::new(
            "nvidia-module",
            Severity::Failure,
            "the NVIDIA kernel module is not loaded: the layer only works on NVIDIA's driver",
            format!("{} does not exist", version_file.display()),
            Some("Install NVIDIA's driver (or boot the kernel it was built for), then reboot.".into()),
        )];
    };
    let version = version.trim().to_string();
    let models = gpu_models(&roots.proc_root);
    if models.is_empty() {
        return vec![Finding::new(
            "gpu",
            Severity::Failure,
            format!("driver {version} is loaded but reports no GPU"),
            format!("no Model: line under {}", roots.proc_root.join("driver/nvidia/gpus").display()),
            Some("Check that the card is seated and powered, and that `nvidia-smi` lists it.".into()),
        )];
    }
    let names = models.iter().map(|(_, m)| m.as_str()).collect::<Vec<_>>().join(", ");
    let evidence = models.iter().map(|(p, m)| format!("{}: Model: {m}", p.display())).chain([format!("{}: {version}", version_file.display())]).collect::<Vec<_>>().join("\n");
    vec![Finding::new("gpu", Severity::Ok, format!("{names}, driver {version}"), evidence, None)]
}

/// One GPU as `nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free` reports it.
#[derive(Clone, Debug, PartialEq)]
struct Gpu {
    name: String,
    /// `major * 10 + minor`: 89 for 8.9.
    compute: Option<u32>,
    total_mib: Option<u64>,
    free_mib: Option<u64>,
    line: String,
}

/// The query's arguments: one call answers the compute capability and the memory.
const SMI_QUERY: [&str; 2] = ["--query-gpu=name,compute_cap,memory.total,memory.free", "--format=csv,noheader,nounits"];

enum Gpus {
    Listed(Vec<Gpu>),
    Failed(String),
    Missing,
}

fn query_gpus(roots: &Roots) -> Gpus {
    match (roots.nvidia_smi)(&SMI_QUERY) {
        CommandOutput::Ran(text) => Gpus::Listed(text.lines().filter(|l| !l.trim().is_empty()).map(parse_gpu_line).collect()),
        CommandOutput::Failed(why) => Gpus::Failed(why),
        CommandOutput::Missing => Gpus::Missing,
    }
}

/// `"8.9"` as 89; anything else `None`.
fn parse_compute_cap(s: &str) -> Option<u32> {
    let (major, minor) = s.trim().split_once('.')?;
    let (major, minor): (u32, u32) = (major.parse().ok()?, minor.parse().ok()?);
    (minor < 10).then_some(major * 10 + minor)
}

fn parse_gpu_line(line: &str) -> Gpu {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let number = |i: usize| fields.get(i).and_then(|f| f.parse().ok());
    Gpu {
        name: fields.first().copied().unwrap_or_default().to_string(),
        compute: fields.get(1).and_then(|f| parse_compute_cap(f)),
        total_mib: number(2),
        free_mib: number(3),
        line: line.trim().to_string(),
    }
}

/// The lowest compute capability the network runs on: its kernels are PTX for `sm_89`
/// (`third_party/opendlss-nr/scripts/ptx/ptxgen.py`, `target="sm_89"`) using FP8 (e4m3) tensor-core
/// instructions (`mma.sync...e4m3`, e.g. `mlp_e4m3.py`), and the device needs `VK_EXT_shader_float8`
/// (`crates/native/cpp/nf_native.cpp`). PTX for sm_89 is only compiled by the driver for 8.9 and newer,
/// and FP8 tensor cores start with Ada (RTX 40): an RTX 20 (7.5) or 30 (8.6) cannot run it at all.
pub const MIN_COMPUTE_CAPABILITY: u32 = 89;

fn check_compute_capability(gpus: &Gpus) -> Vec<Finding> {
    const CHECK: &str = "compute-capability";
    let list = match gpus {
        // No nvidia-smi: nothing to say (the driver check above covers the driver itself).
        Gpus::Missing => return Vec::new(),
        Gpus::Failed(why) => {
            return vec![Finding::new(
                CHECK,
                Severity::Info,
                "nvidia-smi failed, so the GPU's compute capability is unknown",
                format!("nvidia-smi {}: {why}", SMI_QUERY.join(" ")),
                Some(format!("Run `nvidia-smi {}` to see why.", SMI_QUERY.join(" "))),
            )]
        }
        Gpus::Listed(list) => list,
    };
    let evidence = list.iter().map(|g| format!("nvidia-smi: {}", g.line)).collect::<Vec<_>>().join("\n");
    let shown = |c: u32| format!("{}.{}", c / 10, c % 10);
    let able: Vec<&Gpu> = list.iter().filter(|g| g.compute.is_some_and(|c| c >= MIN_COMPUTE_CAPABILITY)).collect();
    if let Some(gpu) = able.first() {
        return vec![Finding::new(CHECK, Severity::Ok, format!("{} has compute capability {}", gpu.name, shown(gpu.compute.unwrap_or_default())), evidence, None)];
    }
    match list.iter().find(|g| g.compute.is_some()) {
        Some(gpu) => vec![Finding::new(
            CHECK,
            Severity::Failure,
            format!(
                "{} (compute capability {}) cannot run the network: it needs an RTX 40-series or newer GPU (compute capability {} or higher, for its FP8 tensor-core kernels)",
                gpu.name,
                shown(gpu.compute.unwrap_or_default()),
                shown(MIN_COMPUTE_CAPABILITY)
            ),
            evidence,
            Some("There is no workaround on this card: the network needs an RTX 40-series or newer GPU.".into()),
        )],
        None => vec![Finding::new(CHECK, Severity::Info, "nvidia-smi reported no readable compute capability", evidence, None)],
    }
}

// ---- VRAM ----------------------------------------------------------------------------------------

/// What the network needs at a size: `(width, height, MiB)`, from `dlss5vk bench`'s peak VRAM over the
/// idle desktop in docs/NATIVE_BACKEND.md, "0.5 Network time on the test machine" (weights,
/// activations, staging). The README's "about 1.5 GB" at 4K DLSS Balanced is the 2228x1253 row.
const NETWORK_VRAM_MIB: [(u32, u32, u64); 4] = [(1485, 836, 945), (2228, 1253, 1511), (2560, 1440, 1810), (3840, 2160, 3474)];

/// The network's need at `width` x `height`: the smallest measured size that covers it, or the largest.
fn network_need_mib(width: u32, height: u32) -> (u32, u32, u64) {
    let pixels = u64::from(width) * u64::from(height);
    NETWORK_VRAM_MIB.iter().copied().find(|&(w, h, _)| u64::from(w) * u64::from(h) >= pixels).unwrap_or(NETWORK_VRAM_MIB[NETWORK_VRAM_MIB.len() - 1])
}

fn check_vram(gpus: &Gpus, hdr: Option<&neural_forge_protocol::ShmHeader>) -> Vec<Finding> {
    const CHECK: &str = "vram";
    let Gpus::Listed(list) = gpus else { return Vec::new() };
    let Some(gpu) = list.iter().find(|g| g.compute.is_some_and(|c| c >= MIN_COMPUTE_CAPABILITY)).or(list.first()) else { return Vec::new() };
    let (Some(total), Some(free)) = (gpu.total_mib, gpu.free_mib) else { return Vec::new() };
    // The size the network runs at: DLSS's input before the upscaler, else the captured frame.
    let size = hdr.and_then(|h| {
        let pre = (h.preupscale_width.load(Ordering::Relaxed), h.preupscale_height.load(Ordering::Relaxed));
        let post = (h.layer_width.load(Ordering::Relaxed), h.layer_height.load(Ordering::Relaxed));
        [pre, post].into_iter().find(|&(w, h)| w != 0 && h != 0)
    });
    let running = hdr.is_some_and(|h| h.native_running.load(Ordering::Relaxed) != 0);
    let evidence = format!("nvidia-smi: {}: {free} MiB free of {total} MiB", gpu.name);
    let Some((w, h)) = size else {
        return vec![Finding::new(
            CHECK,
            Severity::Info,
            format!("{free} MiB of {total} MiB VRAM free; the network needs about {}-{} MiB depending on the game's resolution", NETWORK_VRAM_MIB[0].2, NETWORK_VRAM_MIB[3].2),
            evidence,
            None,
        )];
    };
    let (mw, mh, need) = network_need_mib(w, h);
    let evidence = format!("{evidence}\nthe layer's last frame: {w}x{h}; the network measured {need} MiB at {mw}x{mh}");
    if running {
        // Already resident: what is free is what is left beside it.
        return vec![Finding::new(CHECK, Severity::Ok, format!("the network is running at {w}x{h}; {free} MiB VRAM still free"), evidence, None)];
    }
    if free < need {
        return vec![Finding::new(
            CHECK,
            Severity::Warning,
            format!("only {free} MiB VRAM free, below the {need} MiB the network needs at {w}x{h}: building it can fail"),
            evidence,
            Some("Lower the game's resolution or DLSS quality mode, or its texture quality, to leave the network room.".into()),
        )];
    }
    vec![Finding::new(CHECK, Severity::Ok, format!("{free} MiB VRAM free, enough for the network at {w}x{h} ({need} MiB)"), evidence, None)]
}

// ---- the channel: attachment, device loss, the native backend's status ----------------------------

fn check_attached(roots: &Roots, hdr: &neural_forge_protocol::ShmHeader, manifest_ok: bool) -> Finding {
    const CHECK: &str = "layer-attached";
    let beats = hdr.layer_heartbeat.load(Ordering::Relaxed);
    let attached = hdr.layer_attached.load(Ordering::Relaxed) != 0;
    let frames = neural_forge_protocol::load64(&hdr.layer_frames_lo, &hdr.layer_frames_hi);
    let evidence = format!("{}: layer_attached={}, layer_heartbeat={beats}, layer_frames={frames}", roots.shm_path.display(), u32::from(attached));
    if attached || beats != 0 {
        let game = hdr.game_name();
        let game = if game.is_empty() { "a game".to_string() } else { game };
        return Finding::new(CHECK, Severity::Ok, format!("{game}'s layer attached to the channel since it was created ({frames} frames)"), evidence, None);
    }
    // The likely causes, most likely first.
    let mut causes = Vec::new();
    match &roots.launch_option_state {
        Some(LaunchOptionState::Present) => {}
        Some(LaunchOptionState::Missing(game)) => causes.push(format!("{game}'s Steam launch options lack `NEURAL_FORGE_ENABLE=1 %command%`")),
        None => causes.push("the game's Steam launch options may lack `NEURAL_FORGE_ENABLE=1 %command%`".to_string()),
    }
    causes.push("the game was already running when Neural Forge was installed or updated: restart it".to_string());
    if !manifest_ok {
        causes.push("the layer manifest is missing or broken (see the manifest finding)".to_string());
    }
    let numbered = causes.iter().enumerate().map(|(i, c)| format!("{}. {c}", i + 1)).collect::<Vec<_>>().join("\n");
    let fix = match &roots.launch_option_state {
        Some(LaunchOptionState::Missing(game)) => format!("Set {game}'s launch options to `NEURAL_FORGE_ENABLE=1 %command%` (Steam: the game's Properties > General), then start it."),
        _ => "Check the game's launch options for `NEURAL_FORGE_ENABLE=1 %command%`, then start the game again (a running game keeps the layer it started with).".to_string(),
    };
    Finding::new(CHECK, Severity::Warning, "no game's layer has attached since the channel was created (this boot)", format!("{evidence}\nlikely causes:\n{numbered}"), Some(fix))
}

/// The newest session in the state log: the events of the process that last engaged (or, with no
/// engagement at all, the process that wrote last), oldest first.
fn last_session(events: &[state_log::Event]) -> Option<Vec<state_log::Event>> {
    let pid = events.iter().rev().find(|e| e.kind == state_log::ATTACH).or(events.last())?.pid;
    // A pid is reused across boots; the session is this pid's run of lines since its last attach.
    let mut session: Vec<state_log::Event> = Vec::new();
    for e in events {
        if e.pid != pid {
            continue;
        }
        if e.kind == state_log::ATTACH && session.iter().any(|s| s.kind == state_log::ATTACH) {
            session.clear();
        }
        session.push(e.clone());
    }
    Some(session)
}

/// Unix seconds as `YYYY-MM-DD HH:MM:SS UTC`.
fn utc(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn event_line(e: &state_log::Event) -> String {
    format!("{} {} (pid {}) {}: {}", utc(e.time), e.process, e.pid, e.kind, e.message)
}

fn check_device_lost(hdr: Option<&neural_forge_protocol::ShmHeader>, session: Option<&[state_log::Event]>) -> Vec<Finding> {
    let at = hdr.map_or(0, |h| h.device_lost_at.load(Ordering::Relaxed));
    let logged = session.and_then(|s| s.iter().rev().find(|e| e.kind == state_log::DEVICE_LOST));
    let mut evidence = Vec::new();
    if at != 0 {
        evidence.push(format!("channel: device_lost_at={at} ({})", utc(u64::from(at))));
    }
    if let Some(e) = logged {
        evidence.push(format!("state log: {}", event_line(e)));
    }
    if evidence.is_empty() {
        // Only what was actually read is evidence: with neither the channel nor a session, nothing is known.
        let read: Vec<&str> = [hdr.map(|_| "channel: device_lost_at=0"), session.map(|_| "state log: no device-lost line in the last session")].into_iter().flatten().collect();
        if read.is_empty() {
            return Vec::new();
        }
        return vec![Finding::new("device-lost", Severity::Ok, "the GPU device was not lost in the last game session", read.join("\n"), None)];
    }
    vec![Finding::new(
        "device-lost",
        Severity::Failure,
        "the GPU device was lost during the last game session: the layer went inert, and the game likely froze or crashed",
        evidence.join("\n"),
        Some("Look at the kernel log finding for the Xid at that time, then restart the game (a lost device does not come back).".into()),
    )]
}

fn check_cannot_run(hdr: Option<&neural_forge_protocol::ShmHeader>, session: Option<&[state_log::Event]>) -> Vec<Finding> {
    let reason = hdr.map(neural_forge_protocol::ShmHeader::layer_reason).filter(|r| r.starts_with(state_log::CANNOT_RUN_PREFIX));
    let logged = session.and_then(|s| s.iter().rev().find(|e| e.kind == state_log::NATIVE_UNAVAILABLE));
    let line = reason.clone().or_else(|| logged.map(|e| e.message.clone()));
    let Some(line) = line else { return Vec::new() };
    let first = line.strip_prefix(state_log::CANNOT_RUN_PREFIX).unwrap_or(&line).to_string();
    let mut evidence = Vec::new();
    if let Some(r) = &reason {
        evidence.push(format!("channel: layer_reason \"{r}\""));
    }
    if let Some(e) = logged {
        evidence.push(format!("state log: {}", event_line(e)));
    }
    vec![Finding::new(
        "device-requirements",
        Severity::Failure,
        format!("the GPU cannot run the network: it lacks {first}"),
        evidence.join("\n"),
        Some(format!("Update the NVIDIA driver (the network needs {first}); on an RTX 20 or 30-series card no driver adds it.")),
    )]
}

/// The native backend's own status line when it says the network is not running.
fn check_native_status(hdr: &neural_forge_protocol::ShmHeader) -> Vec<Finding> {
    let reason = hdr.layer_reason();
    let Some(why) = reason.strip_prefix("native network not running: ") else { return Vec::new() };
    vec![Finding::new(
        "native-status",
        Severity::Warning,
        format!("the native network was not running in the last game: {why}"),
        format!("channel: layer_reason \"{reason}\""),
        Some("Fix what the line names; the layer retries on its own, and a game restart starts it fresh.".into()),
    )]
}

// ---- the kernel log ------------------------------------------------------------------------------

/// The arguments: this boot's kernel messages, with ISO times.
const JOURNAL_ARGS: [&str; 5] = ["-k", "-b", "--no-pager", "-o", "short-iso"];

/// One `NVRM: Xid` line.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Xid {
    time: String,
    number: u32,
    process: Option<String>,
    line: String,
}

/// Parses `<time> <host> kernel: NVRM: Xid (PCI:0000:01:00): 109, pid=..., name=X, ...`. Only the
/// driver's exact prefix counts: a line that merely mentions an Xid is not one.
fn parse_xid(line: &str) -> Option<Xid> {
    let at = line.find("NVRM: Xid (")?;
    let rest = &line[at..];
    let after = &rest[rest.find("): ")? + 3..];
    let number = after.split(',').next()?.trim().parse().ok()?;
    let process = after.split(", ").find_map(|f| f.strip_prefix("name=")).map(|n| n.trim_end_matches(',').to_string());
    let time = line.split_whitespace().next().unwrap_or_default().to_string();
    Some(Xid { time, number, process, line: line.trim().to_string() })
}

/// What the README's Known issues say about an Xid number.
fn xid_meaning(number: u32) -> Option<&'static str> {
    match number {
        109 => Some("CTX_SWITCH_TIMEOUT, a GPU hang: a widely reported NVIDIA Linux driver bug under Proton, also seen in games with no DLSS or neural rendering at all"),
        119 => Some("a GPU hang: a widely reported NVIDIA Linux driver bug under Proton, also seen in games with no DLSS or neural rendering at all"),
        13 | 32 => Some("a GPU fault; seen intermittently in Remnant II and the Black Myth: Wukong Benchmark Tool with frame generation on (README, Known issues)"),
        _ => None,
    }
}

/// How many of the newest Xid lines are reported.
const XID_SHOWN: usize = 3;

fn check_xid(roots: &Roots) -> Vec<Finding> {
    const CHECK: &str = "xid";
    let command = format!("journalctl {}", JOURNAL_ARGS.join(" "));
    let text = match (roots.journalctl)(&JOURNAL_ARGS) {
        CommandOutput::Ran(text) => text,
        CommandOutput::Failed(why) => {
            return vec![Finding::new(
                CHECK,
                Severity::Info,
                "the kernel log could not be read, so GPU faults (Xid) are unknown",
                format!("{command}: {why}"),
                Some(format!("Run `sudo {command} | grep 'NVRM: Xid'`, or add yourself to the systemd-journal group.")),
            )]
        }
        CommandOutput::Missing => {
            return vec![Finding::new(
                CHECK,
                Severity::Info,
                "journalctl is not installed, so GPU faults (Xid) are unknown",
                command,
                Some("Run `sudo dmesg | grep 'NVRM: Xid'`.".into()),
            )]
        }
    };
    let xids: Vec<Xid> = text.lines().filter_map(parse_xid).collect();
    if xids.is_empty() {
        return vec![Finding::new(CHECK, Severity::Ok, "no NVIDIA Xid (GPU fault) in this boot's kernel log", command, None)];
    }
    let newest: Vec<&Xid> = xids.iter().rev().take(XID_SHOWN).collect();
    let mut numbers: Vec<u32> = newest.iter().map(|x| x.number).collect();
    numbers.dedup();
    let latest = newest[0];
    let mut evidence = format!("{command}: {} Xid line(s) this boot, newest first:", xids.len());
    for x in &newest {
        evidence.push_str(&format!("\n{}", x.line));
    }
    for n in &numbers {
        if let Some(meaning) = xid_meaning(*n) {
            evidence.push_str(&format!("\nXid {n}: {meaning}"));
        }
    }
    let who = latest.process.as_deref().map_or(String::new(), |p| format!(" in {p}"));
    vec![Finding::new(
        CHECK,
        Severity::Warning,
        format!("the GPU faulted this boot: newest Xid {} at {}{who}", latest.number, latest.time),
        evidence,
        Some(match latest.number {
            109 | 119 => "Update the NVIDIA driver; README, Known issues (Xid 109/119) lists the workarounds other users report.".to_string(),
            _ => "Note the game and settings, try the game again without frame generation, and report it with this output.".to_string(),
        }),
    )]
}

// ---- the hotkey ----------------------------------------------------------------------------------

/// The toggle key reads `/dev/input` through evdev, which needs the `input` group (keyboards are
/// `root:input` with no uaccess ACL); without it the layer falls back to XInput2, which works on an
/// X11 or XWayland desktop but not inside gamescope (`crates/layer/src/hotkey.rs`). Only a note.
fn check_input_group(roots: &Roots) -> Vec<Finding> {
    const CHECK: &str = "input-group";
    let status_file = roots.proc_root.join("self/status");
    let groups: Option<Vec<u32>> = std::fs::read_to_string(&status_file)
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("Groups:")).map(|g| g.split_whitespace().filter_map(|n| n.parse().ok()).collect()));
    let input_gid: Option<u32> = std::fs::read_to_string(&roots.group_file).ok().and_then(|s| {
        s.lines().find_map(|l| {
            let mut f = l.split(':');
            (f.next() == Some("input")).then(|| f.nth(1).and_then(|g| g.parse().ok())).flatten()
        })
    });
    let (Some(groups), Some(gid)) = (groups, input_gid) else { return Vec::new() };
    let evidence = format!("{}: Groups: {}; {}: input is gid {gid}", status_file.display(), groups.iter().map(u32::to_string).collect::<Vec<_>>().join(" "), roots.group_file.display());
    if groups.contains(&gid) {
        return vec![Finding::new(CHECK, Severity::Ok, "you are in the input group: the toggle key works everywhere", evidence, None)];
    }
    vec![Finding::new(
        CHECK,
        Severity::Info,
        "you are not in the input group: the in-game toggle key works on an X11 or XWayland desktop, but not inside gamescope",
        evidence,
        Some("Only for gamescope: `sudo usermod -aG input $USER`, then log out and back in.".into()),
    )]
}

// ---- the state log -------------------------------------------------------------------------------

fn check_session_log(roots: &Roots, events: Result<&[state_log::Event], std::io::ErrorKind>, session: Option<&[state_log::Event]>) -> Vec<Finding> {
    const CHECK: &str = "last-session";
    let file = roots.state_dir.join(state_log::FILE_NAME);
    let shown = file.display().to_string();
    match events {
        Err(std::io::ErrorKind::NotFound) => {
            return vec![Finding::new(CHECK, Severity::Info, "no game has run with this version's layer yet (no state log)", format!("{shown} does not exist"), None)]
        }
        Err(kind) => return vec![Finding::new(CHECK, Severity::Info, format!("the layer's state log cannot be read: {kind}"), shown, None)],
        Ok(_) => {}
    }
    let Some(session) = session.filter(|s| !s.is_empty()) else {
        return vec![Finding::new(CHECK, Severity::Info, "the layer's state log holds no events", shown, None)];
    };
    let first = &session[0];
    let header = format!("{shown}: the last session, {} (pid {}) from {}", first.process, first.pid, utc(first.time));
    let mut out = Vec::new();
    // The newest of the network's failure and recovery lines is the verdict.
    if let Some(e) = session.iter().rev().find(|e| e.kind == state_log::NATIVE_FAILED || e.kind == state_log::NATIVE_RECOVERED) {
        if e.kind == state_log::NATIVE_FAILED {
            out.push(Finding::new(
                CHECK,
                Severity::Warning,
                format!("the network failed in the last session and did not recover: {}", e.message),
                format!("{header}\n{}", event_line(e)),
                Some("Fix what the line names (a missing model: extract it in Setup; out of memory: lower the resolution), then restart the game.".into()),
            ));
        }
    }
    if let Some(e) = session.iter().rev().find(|e| e.kind == state_log::FENCE_TIMEOUT) {
        out.push(Finding::new(
            CHECK,
            Severity::Warning,
            "a GPU wait timed out in the last session (a driver stall without device loss): the game stuttered or froze",
            format!("{header}\n{}", event_line(e)),
            Some("Check the kernel log finding for an Xid at that time; if there is none, report this output.".into()),
        ));
    }
    if let Some(e) = session.iter().rev().find(|e| e.kind == state_log::DUPLICATE) {
        out.push(Finding::new(
            CHECK,
            Severity::Warning,
            "two copies of the layer were loaded in the last session; the second stayed inert",
            format!("{header}\n{}", event_line(e)),
            Some("Remove the extra implicit-layer manifest the line names (only the installed one should remain).".into()),
        ));
    }
    if out.is_empty() {
        let errors = [state_log::DEVICE_LOST, state_log::NATIVE_UNAVAILABLE];
        // Device loss and a device that cannot run the network have their own findings.
        let engaged = session.iter().any(|e| e.kind == state_log::ATTACH);
        let clean = !session.iter().any(|e| errors.contains(&e.kind.as_str()));
        let summary = match (engaged, clean) {
            (true, true) => "the last session engaged with no errors".to_string(),
            (true, false) => "the last session engaged; its errors are reported above".to_string(),
            (false, _) => "the last session wrote events but never engaged (the game never rendered steadily)".to_string(),
        };
        let lines = session.iter().map(event_line).collect::<Vec<_>>().join("\n");
        out.push(Finding::new(CHECK, if engaged { Severity::Ok } else { Severity::Info }, summary, format!("{header}\n{lines}"), None));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    /// A fixture tree: every root under one temporary directory, the commands answering `smi`/`journal`.
    struct Fixture {
        dir: PathBuf,
        roots: Roots,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn fixture(smi: CommandOutput, journal: CommandOutput) -> Fixture {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("neural-forge-doctor-{}-{}", std::process::id(), COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        // The channel's directory must be private, as the real runtime dir is.
        std::fs::create_dir_all(dir.join("run")).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.join("run"), std::fs::Permissions::from_mode(0o700)).unwrap();
        let roots = Roots {
            proc_root: dir.join("proc"),
            sys_root: dir.join("sys"),
            group_file: dir.join("etc/group"),
            config_file: dir.join("config/neural-forge/config.ini"),
            data_home: dir.join("data"),
            data_dir: dir.join("data/neural-forge"),
            state_dir: dir.join("state/neural-forge"),
            config_dir: dir.join("config/neural-forge"),
            binaries_dir: dir.join("data/neural-forge/binaries"),
            model_dir: dir.join("data/neural-forge/model"),
            shm_path: dir.join("run/shm.bin"),
            nvidia_smi: Box::new(move |_| smi.clone()),
            journalctl: Box::new(move |_| journal.clone()),
            launch_option_state: None,
        };
        Fixture { dir, roots }
    }

    fn quiet() -> Fixture {
        fixture(CommandOutput::Missing, CommandOutput::Ran(String::new()))
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// The channel at the fixture's path, as a game's layer would leave it.
    fn channel(f: &Fixture) -> neural_forge_protocol::mapping::Mapping {
        neural_forge_protocol::mapping::open_path(&f.roots.shm_path.to_string_lossy()).unwrap()
    }

    fn find<'a>(findings: &'a [Finding], check: &str) -> Vec<&'a Finding> {
        findings.iter().filter(|f| f.check == check).collect()
    }

    fn one<'a>(findings: &'a [Finding], check: &str) -> &'a Finding {
        let found = find(findings, check);
        assert_eq!(found.len(), 1, "{check}: {findings:#?}");
        found[0]
    }

    fn manifest(f: &Fixture, library: &Path) {
        write(&f.roots.data_home.join("vulkan/implicit_layer.d/neural_forge_layer.json"), &format!(r#"{{"layer": {{"name": "VK_LAYER_neuralforge_neural", "library_path": "{}"}}}}"#, library.display()));
    }

    #[test]
    fn severities_sort_worst_first_and_only_failure_fails() {
        let mut v = vec![
            Finding::new("a", Severity::Ok, "", "", None),
            Finding::new("b", Severity::Failure, "", "", None),
            Finding::new("c", Severity::Info, "", "", None),
            Finding::new("d", Severity::Warning, "", "", None),
            Finding::new("e", Severity::Failure, "", "", None),
        ];
        sort_worst_first(&mut v);
        assert_eq!(v.iter().map(|f| f.check).collect::<String>(), "bedca");
        assert!(any_failure(&v));
        assert!(!any_failure(&v[2..]));
    }

    #[test]
    fn the_text_has_each_finding_with_its_evidence_and_fix_and_a_count() {
        let v = vec![
            Finding::new("model", Severity::Failure, "the model is missing", "line one\nline two", Some("extract it".into())),
            Finding::new("gpu", Severity::Ok, "RTX 5070", "/proc/x", None),
        ];
        let text = render_text(&v);
        assert!(text.starts_with("[FAILURE] model: the model is missing\n    line one\n    line two\n    fix: extract it\n[ok] gpu: RTX 5070\n    /proc/x\n"), "{text}");
        assert!(text.ends_with("1 failure(s), 0 warning(s), 0 note(s), 1 ok\n"), "{text}");
    }

    #[test]
    fn the_original_checks_report_a_missing_config_and_model() {
        let f = quiet();
        let v = run(&f.roots);
        assert_eq!(one(&v, "config").severity, Severity::Failure);
        assert_eq!(one(&v, "model").severity, Severity::Failure);
        assert_eq!(one(&v, "nvngx-dll").severity, Severity::Info, "the DLL is only needed to extract");
        assert_eq!(one(&v, "user-dirs").severity, Severity::Ok);
        assert_eq!(one(&v, "channel").severity, Severity::Ok);
        assert!(f.roots.state_dir.is_dir(), "the user dirs are created");
        // Worst first.
        assert!(v.windows(2).all(|w| w[0].severity >= w[1].severity));
    }

    #[test]
    fn a_present_config_and_model_are_ok() {
        let f = quiet();
        write(&f.roots.config_file, "");
        write(&f.roots.model_dir.join("manifest.json"), r#"{"source": {"build": "310.8.0", "verified": true}}"#);
        write(&f.roots.binaries_dir.join("nvngx_dlssnr.dll"), "");
        let v = run(&f.roots);
        assert_eq!(one(&v, "config").severity, Severity::Ok);
        assert_eq!(one(&v, "model").summary, "the model is extracted (build 310.8.0)");
        assert_eq!(one(&v, "nvngx-dll").severity, Severity::Ok);
    }

    #[test]
    fn the_manifest_must_exist_parse_and_point_at_the_library() {
        let f = quiet();
        let check = || check_manifest(&f.roots);
        assert!(check().summary.contains("not installed"));
        write(&f.roots.data_home.join("vulkan/implicit_layer.d/neural_forge_layer.json"), "{ not json");
        assert!(check().summary.contains("not valid JSON"));
        write(&f.roots.data_home.join("vulkan/implicit_layer.d/neural_forge_layer.json"), r#"{"layer": {}}"#);
        assert!(check().summary.contains("names no library"));
        let library = f.dir.join("lib/neural-forge/libneural_forge_layer.so");
        manifest(&f, &library);
        let missing = check();
        assert_eq!(missing.severity, Severity::Failure);
        assert!(missing.summary.contains("does not exist") && missing.evidence.contains("libneural_forge_layer.so"));
        write(&library, "");
        assert_eq!(check().severity, Severity::Ok);
    }

    #[test]
    fn the_gpu_model_and_driver_version_are_read() {
        let f = quiet();
        write(&f.roots.sys_root.join("module/nvidia/version"), "615.78.08\n");
        write(&f.roots.proc_root.join("driver/nvidia/gpus/0000:01:00.0/information"), "Model: \t\t NVIDIA GeForce RTX 5070\nIRQ:   16\n");
        let v = check_gpu(&f.roots);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].severity, Severity::Ok);
        assert_eq!(v[0].summary, "NVIDIA GeForce RTX 5070, driver 615.78.08");
        assert!(v[0].evidence.contains("information: Model: NVIDIA GeForce RTX 5070"));
    }

    #[test]
    fn a_missing_kernel_module_is_its_own_failure() {
        let f = quiet();
        write(&f.roots.proc_root.join("driver/nvidia/gpus/0000:01:00.0/information"), "Model: RTX\n");
        let v = check_gpu(&f.roots);
        assert_eq!((v[0].check, v[0].severity), ("nvidia-module", Severity::Failure));
        write(&f.roots.sys_root.join("module/nvidia/version"), "615.78.08\n");
        std::fs::remove_dir_all(f.roots.proc_root.join("driver")).unwrap();
        let v = check_gpu(&f.roots);
        assert_eq!((v[0].check, v[0].severity), ("gpu", Severity::Failure));
    }

    #[test]
    fn compute_capability_parses_as_major_times_ten_plus_minor() {
        assert_eq!(parse_compute_cap("8.9"), Some(89));
        assert_eq!(parse_compute_cap(" 12.0 "), Some(120));
        assert_eq!(parse_compute_cap("8.6"), Some(86));
        assert_eq!(parse_compute_cap("[N/A]"), None);
        assert_eq!(parse_compute_cap("8.10"), None);
        let gpu = parse_gpu_line("NVIDIA GeForce RTX 5070, 12.0, 12227, 10718");
        assert_eq!((gpu.compute, gpu.total_mib, gpu.free_mib), (Some(120), Some(12227), Some(10718)));
    }

    #[test]
    fn a_card_below_8_9_is_unsupported_and_no_nvidia_smi_says_nothing() {
        let smi = |text: &str| query_gpus(&fixture(CommandOutput::Ran(text.into()), CommandOutput::Missing).roots);
        let v = check_compute_capability(&smi("NVIDIA GeForce RTX 3080, 8.6, 10240, 9000\n"));
        assert_eq!(v[0].severity, Severity::Failure);
        assert!(v[0].summary.contains("RTX 3080 (compute capability 8.6) cannot run the network"), "{}", v[0].summary);
        assert!(v[0].evidence.contains("nvidia-smi: NVIDIA GeForce RTX 3080, 8.6"));
        let v = check_compute_capability(&smi("NVIDIA GeForce RTX 4060, 8.9, 8188, 7000\n"));
        assert_eq!(v[0].severity, Severity::Ok);
        let v = check_compute_capability(&smi("NVIDIA GeForce RTX 5070, 12.0, 12227, 10718\n"));
        assert_eq!(v[0].summary, "NVIDIA GeForce RTX 5070 has compute capability 12.0");
        // A 30-series beside a 40-series: the able card counts.
        let v = check_compute_capability(&smi("NVIDIA GeForce RTX 3060, 8.6, 12288, 11000\nNVIDIA GeForce RTX 4090, 8.9, 24564, 23000\n"));
        assert_eq!(v[0].severity, Severity::Ok);
        assert!(check_compute_capability(&Gpus::Missing).is_empty());
        assert_eq!(check_compute_capability(&Gpus::Failed("exit status: 9".into()))[0].severity, Severity::Info);
    }

    #[test]
    fn vram_below_the_networks_need_at_the_games_size_warns() {
        assert_eq!(network_need_mib(1485, 836).2, 945);
        assert_eq!(network_need_mib(1920, 1080).2, 1511);
        assert_eq!(network_need_mib(2560, 1440).2, 1810);
        assert_eq!(network_need_mib(7680, 4320).2, 3474);
        let gpus = |free: u64| Gpus::Listed(vec![parse_gpu_line(&format!("NVIDIA GeForce RTX 5070, 12.0, 12227, {free}"))]);
        let hdr = neural_forge_protocol::ShmHeader::default();
        hdr.init_defaults();
        // No game size yet: a note.
        assert_eq!(check_vram(&gpus(500), Some(&hdr))[0].severity, Severity::Info);
        hdr.preupscale_width.store(2228, Ordering::Relaxed);
        hdr.preupscale_height.store(1253, Ordering::Relaxed);
        let low = check_vram(&gpus(1200), Some(&hdr));
        assert_eq!(low[0].severity, Severity::Warning);
        assert!(low[0].summary.contains("below the 1511 MiB the network needs at 2228x1253"), "{}", low[0].summary);
        assert_eq!(check_vram(&gpus(4000), Some(&hdr))[0].severity, Severity::Ok);
        // Running: the network is already in what is used.
        hdr.native_running.store(1, Ordering::Relaxed);
        assert_eq!(check_vram(&gpus(300), Some(&hdr))[0].severity, Severity::Ok);
    }

    #[test]
    fn a_layer_that_never_attached_lists_the_causes_in_order() {
        let mut f = quiet();
        let mapping = channel(&f);
        let v = check_attached(&f.roots, mapping.header(), false);
        assert_eq!(v.severity, Severity::Warning);
        let (launch, restart, manifest) = (v.evidence.find("1. the game's Steam launch options may lack").unwrap(), v.evidence.find("2. the game was already running").unwrap(), v.evidence.find("3. the layer manifest").unwrap());
        assert!(launch < restart && restart < manifest);
        // Known missing for a game: named, first, and the fix says where.
        f.roots.launch_option_state = Some(LaunchOptionState::Missing("Cyberpunk 2077".into()));
        let v = check_attached(&f.roots, mapping.header(), true);
        assert!(v.evidence.contains("1. Cyberpunk 2077's Steam launch options lack"), "{}", v.evidence);
        assert!(!v.evidence.contains("3."), "an installed manifest is not a cause");
        assert!(v.fix.unwrap().starts_with("Set Cyberpunk 2077's launch options"));
        f.roots.launch_option_state = Some(LaunchOptionState::Present);
        assert!(check_attached(&f.roots, mapping.header(), true).evidence.contains("1. the game was already running"));
        // A heartbeat: attached.
        mapping.header().layer_heartbeat.store(3, Ordering::Relaxed);
        mapping.header().set_game_name("GTA5_Enhanced.exe");
        let v = check_attached(&f.roots, mapping.header(), true);
        assert_eq!(v.severity, Severity::Ok);
        assert!(v.summary.starts_with("GTA5_Enhanced.exe's layer attached"));
    }

    fn event(time: u64, pid: u32, kind: &str, message: &str) -> String {
        state_log::format_line(time, pid, "Game.exe", kind, message)
    }

    #[test]
    fn device_loss_is_read_from_the_channel_and_from_the_state_log() {
        let f = quiet();
        let mapping = channel(&f);
        let hdr = mapping.header();
        assert_eq!(check_device_lost(Some(hdr), None)[0].severity, Severity::Ok);
        assert_eq!(check_device_lost(Some(hdr), None)[0].evidence, "channel: device_lost_at=0");
        assert!(check_device_lost(None, None).is_empty(), "nothing read, nothing claimed");
        hdr.device_lost_at.store(1_760_119_200, Ordering::Relaxed);
        let v = check_device_lost(Some(hdr), None);
        assert_eq!(v[0].severity, Severity::Failure);
        assert!(v[0].summary.starts_with("the GPU device was lost during the last game session"));
        assert!(v[0].evidence.contains("device_lost_at=1760119200 (2025-10-10 18:00:00 UTC)"), "{}", v[0].evidence);
        // From the log alone (the channel is gone after a reboot).
        let session: Vec<_> = [event(1, 7, state_log::ATTACH, "engaged"), event(2, 7, state_log::DEVICE_LOST, "VK_ERROR_DEVICE_LOST")].iter().filter_map(|l| state_log::parse_line(l)).collect();
        let v = check_device_lost(None, Some(&session));
        assert_eq!(v[0].severity, Severity::Failure);
        assert!(v[0].evidence.contains("state log: 1970-01-01 00:00:02 UTC Game.exe (pid 7) device-lost"));
    }

    #[test]
    fn a_device_that_cannot_run_the_network_is_reported_from_layer_reason_or_the_log() {
        let f = quiet();
        let mapping = channel(&f);
        let hdr = mapping.header();
        assert!(check_cannot_run(Some(hdr), None).is_empty());
        hdr.set_layer_reason("device cannot run the network: VK_EXT_shader_float8");
        let v = check_cannot_run(Some(hdr), None);
        assert_eq!(v[0].severity, Severity::Failure);
        assert_eq!(v[0].summary, "the GPU cannot run the network: it lacks VK_EXT_shader_float8");
        // Another status line that merely mentions the words is not this.
        hdr.set_layer_reason("native network: the device cannot run the network: later");
        assert!(check_cannot_run(Some(hdr), None).is_empty());
        let session: Vec<_> = [event(1, 7, state_log::NATIVE_UNAVAILABLE, "device cannot run the network: shaderInt64")].iter().filter_map(|l| state_log::parse_line(l)).collect();
        assert_eq!(check_cannot_run(None, Some(&session))[0].summary, "the GPU cannot run the network: it lacks shaderInt64");
    }

    #[test]
    fn the_native_status_line_warns_only_when_the_network_is_not_running() {
        let hdr = neural_forge_protocol::ShmHeader::default();
        hdr.set_layer_reason("native network running");
        assert!(check_native_status(&hdr).is_empty());
        hdr.set_layer_reason("native network not running: no model at /m: extract it in the Setup tab (neural-forge-cli extract-model)");
        let v = check_native_status(&hdr);
        assert_eq!(v[0].severity, Severity::Warning);
        assert!(v[0].summary.ends_with("no model at /m: extract it in the Setup tab (neural-forge-cli extract-model)"));
    }

    const XID_LOG: &str = "\
2026-10-10T10:25:59-04:00 LordNikon kernel: NVRM: Xid (PCI:0000:01:00): 13, pid=1, name=Wukong.exe, Graphics Exception
2026-10-10T10:26:00-04:00 LordNikon kernel: usb 1-1: a line that says Xid (PCI:0000:01:00): 99, but is not the driver's
2026-10-10T14:03:24-04:00 LordNikon kernel: NVRM: Xid (PCI:0000:01:00): 32, pid=2, name=Remnant2.exe, Channel ID 0x15
2026-10-10T14:15:00-04:00 LordNikon kernel: NVRM: Xid (PCI:0000:01:00): 109, pid=860236, name=HogwartsLegacy., channel 0x00000015, errorString CTX SWITCH TIMEOUT, Info 0x1c018
";

    #[test]
    fn xid_lines_are_parsed_by_the_drivers_exact_prefix() {
        let x = parse_xid(XID_LOG.lines().nth(3).unwrap()).unwrap();
        assert_eq!((x.time.as_str(), x.number, x.process.as_deref()), ("2026-10-10T14:15:00-04:00", 109, Some("HogwartsLegacy.")));
        assert_eq!(parse_xid(XID_LOG.lines().nth(1).unwrap()), None, "the substring trap");
        assert_eq!(parse_xid("NVRM: Xid (PCI:0000:01:00): notanumber, pid=1"), None);
    }

    #[test]
    fn the_newest_xids_are_reported_with_the_readmes_meaning() {
        let f = fixture(CommandOutput::Missing, CommandOutput::Ran(XID_LOG.into()));
        let v = check_xid(&f.roots);
        assert_eq!(v[0].severity, Severity::Warning);
        assert_eq!(v[0].summary, "the GPU faulted this boot: newest Xid 109 at 2026-10-10T14:15:00-04:00 in HogwartsLegacy.");
        assert!(v[0].evidence.contains("3 Xid line(s) this boot"), "{}", v[0].evidence);
        assert!(v[0].evidence.contains("Xid 109: CTX_SWITCH_TIMEOUT, a GPU hang: a widely reported NVIDIA Linux driver bug under Proton"));
        assert!(v[0].evidence.contains("Xid 32: a GPU fault"));
        assert!(!v[0].evidence.contains(": 99,"), "the substring trap is not reported");
        let clean = fixture(CommandOutput::Missing, CommandOutput::Ran("2026-10-10T10:00:00-04:00 host kernel: Linux version 7.2\n".into()));
        assert_eq!(check_xid(&clean.roots)[0].severity, Severity::Ok);
    }

    #[test]
    fn an_unreadable_kernel_log_says_so_and_gives_the_command() {
        let f = fixture(CommandOutput::Missing, CommandOutput::Failed("exit status: 1 (No journal files were opened due to insufficient permissions.)".into()));
        let v = check_xid(&f.roots);
        assert_eq!(v[0].severity, Severity::Info);
        assert!(v[0].evidence.contains("insufficient permissions"));
        assert!(v[0].fix.as_deref().unwrap().contains("journalctl -k -b --no-pager -o short-iso | grep 'NVRM: Xid'"));
        let f = fixture(CommandOutput::Missing, CommandOutput::Missing);
        assert!(check_xid(&f.roots)[0].fix.as_deref().unwrap().contains("dmesg"));
    }

    #[test]
    fn the_input_group_is_only_a_note() {
        let f = quiet();
        assert!(check_input_group(&f.roots).is_empty(), "nothing to read, nothing said");
        write(&f.roots.group_file, "inputx:x:5:\ninput:x:994:\nplugdev:x:46:\n");
        write(&f.roots.proc_root.join("self/status"), "Name:\tx\nGroups:\t4 27 46 \n");
        let v = check_input_group(&f.roots);
        assert_eq!(v[0].severity, Severity::Info);
        assert!(v[0].evidence.contains("input is gid 994"), "not the `inputx` group: {}", v[0].evidence);
        write(&f.roots.proc_root.join("self/status"), "Groups:\t4 994\n");
        assert_eq!(check_input_group(&f.roots)[0].severity, Severity::Ok);
    }

    #[test]
    fn the_last_session_is_the_last_attached_process_and_its_last_native_line_is_the_verdict() {
        let f = quiet();
        let log = [
            event(10, 100, state_log::ATTACH, "engaged"),
            event(11, 100, state_log::NATIVE_FAILED, "an older session's failure"),
            event(20, 200, state_log::NATIVE_FAILED, "no model at /m"),
            event(21, 200, state_log::ATTACH, "engaged"),
            event(22, 200, state_log::NATIVE_RECOVERED, "the network is built again"),
            // A launcher that never engaged writes after the game: not the session.
            event(30, 300, state_log::NATIVE_UNAVAILABLE, "device cannot run the network: x"),
        ]
        .concat();
        write(&f.roots.state_dir.join(state_log::FILE_NAME), &log);
        let events = state_log::read_dir_events(&f.roots.state_dir).unwrap();
        let session = last_session(&events).unwrap();
        assert_eq!(session.iter().map(|e| e.time).collect::<Vec<_>>(), [20, 21, 22]);
        let v = check_session_log(&f.roots, Ok(events.as_slice()), Some(&session));
        assert_eq!((v.len(), v[0].severity), (1, Severity::Ok), "recovered: the newest line is the verdict: {v:#?}");
        // Failed after recovering: the failure is the verdict.
        let mut failed = session.clone();
        failed.extend(state_log::parse_line(&event(23, 200, state_log::NATIVE_FAILED, "building the network for 3840x2160 failed: out of device memory")));
        let v = check_session_log(&f.roots, Ok(events.as_slice()), Some(&failed));
        assert_eq!(v[0].severity, Severity::Warning);
        assert!(v[0].summary.ends_with("building the network for 3840x2160 failed: out of device memory"));
    }

    #[test]
    fn a_fence_timeout_and_a_duplicate_copy_in_the_last_session_warn() {
        let f = quiet();
        let session: Vec<_> = [
            event(1, 5, state_log::ATTACH, "engaged"),
            event(2, 5, state_log::FENCE_TIMEOUT, "fence wait timed out after 5s at capture; stages, newest first: a < b"),
            event(3, 5, state_log::DUPLICATE, "another copy is already loaded from /opt/x.so"),
        ]
        .iter()
        .filter_map(|l| state_log::parse_line(l))
        .collect();
        let v = check_session_log(&f.roots, Ok(session.as_slice()), Some(&session));
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|f| f.severity == Severity::Warning));
        assert!(v[0].evidence.contains("stages, newest first: a < b"));
        assert!(v[1].evidence.contains("/opt/x.so"));
    }

    #[test]
    fn no_state_log_is_a_note() {
        let f = quiet();
        let v = run(&f.roots);
        let s = one(&v, "last-session");
        assert_eq!(s.severity, Severity::Info);
        assert!(s.summary.contains("no state log"));
    }

    #[test]
    fn run_reads_the_session_through_to_the_device_loss_and_fails() {
        let f = quiet();
        write(&f.roots.state_dir.join(state_log::FILE_NAME), &[event(1, 9, state_log::ATTACH, "engaged"), event(2, 9, state_log::DEVICE_LOST, "VK_ERROR_DEVICE_LOST")].concat());
        let v = run(&f.roots);
        assert_eq!(one(&v, "device-lost").severity, Severity::Failure);
        assert_eq!(one(&v, "last-session").summary, "the last session engaged; its errors are reported above");
        assert_eq!(v[0].severity, Severity::Failure);
    }

    #[test]
    fn utc_formats_unix_seconds() {
        assert_eq!(utc(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00:00 UTC");
        assert_eq!(utc(1_791_656_100), "2026-10-10 18:15:00 UTC");
    }
}
