//! The diagnostic report: one zip on the desktop with what a problem report needs, so nobody has to
//! be asked for each file in turn. `neural-forge-cli report` and the GUI's "Save report" write it;
//! nothing is opened in a browser or uploaded, the user attaches the file themselves.
//!
//! Members: `summary.txt` (version, GPU, driver, compute capability, kernel, session type, the
//! game's Proton tool, the doctor's findings), `config.ini`, `model/manifest.json` (never the
//! weights), the layer's state-dir logs (`layer.log`, `layer.log.1`), the kernel's Xid lines, the
//! live channel's status (`neural-forge-cli shmctl status`'s text) and the newest capture pair.
//! Logs keep their last [`LOG_CAP`] bytes.
//!
//! Every text member, and every member's name, is redacted before it is written ([`Redactor`]): the
//! home directory becomes `~` and the user name `<user>`. PNGs are stored as they are: the layer's
//! encoder (`neural_forge_layer::dump::write_png`) writes no text chunks, so they carry no paths.
//!
//! Everything this module reads comes in through [`Sources`] (paths) and [`Inputs`] (text the
//! caller already has: the findings, the Xid lines, the channel status), so tests drive it from
//! temporary directories; [`Sources::from_env`] and [`probe_system`] are the only parts that read
//! the real system.
//!
//! The zip is written by the small writer below (deflate from `miniz_oxide` for text, store for
//! the already-compressed PNGs) rather than a zip crate. Where the zip goes, and keeping a log's
//! end, follow DLSS5oneclick-forlinux's `report.rs` (MIT, see ATTRIBUTION.md).

use std::path::{Path, PathBuf};

/// The most of each log the report keeps: its end, where the failure is.
pub const LOG_CAP: u64 = 4 << 20;

/// Where the report reads its files from.
#[derive(Debug, Clone)]
pub struct Sources {
    pub home: PathBuf,
    /// `$XDG_CONFIG_HOME` itself (holds `user-dirs.dirs` and `neural-forge/config.ini`).
    pub config_home: PathBuf,
    /// The model directory (`manifest.json` is read from it, nothing else).
    pub model_dir: PathBuf,
    /// Neural Forge's state directory (`layer.log`, `layer.log.1`).
    pub state_dir: PathBuf,
    /// The layer's capture directory (`neural_forge_layer::dump::captures_dir`).
    pub captures_dir: PathBuf,
    /// `/proc` (kernel release, NVIDIA driver and GPU names).
    pub proc_root: PathBuf,
    /// Steam installations to look in for the game's Proton tool, in order.
    pub steam_roots: Vec<PathBuf>,
}

impl Sources {
    /// The real locations, from the environment the way the rest of this crate reads it.
    pub fn from_env() -> Self {
        let home = PathBuf::from(crate::paths::home());
        Sources {
            config_home: PathBuf::from(crate::paths::config_home()),
            model_dir: PathBuf::from(crate::model::model_dir()),
            state_dir: PathBuf::from(crate::paths::state_dir()),
            captures_dir: PathBuf::from(crate::paths::captures_dir()),
            proc_root: PathBuf::from("/proc"),
            steam_roots: vec![
                PathBuf::from(crate::paths::data_home()).join("Steam"),
                home.join(".steam/steam"),
                home.join(".var/app/com.valvesoftware.Steam/.local/share/Steam"),
            ],
            home,
        }
    }
}

/// What the caller supplies as text.
#[derive(Debug, Clone, Default)]
pub struct Inputs<'a> {
    /// The doctor's findings, already rendered; `None` when none were gathered.
    pub findings: Option<&'a str>,
    /// The kernel log's NVIDIA Xid lines; `None` when they were not read. `xid.txt` is left out then.
    pub xid_lines: Option<&'a str>,
    /// `neural-forge-cli shmctl status`'s text ([`crate::shm_status::status_text`]), or why the
    /// channel could not be opened.
    pub channel_status: &'a str,
    /// The game the layer last named (the channel's game name); empty when unknown.
    pub game: &'a str,
    /// The game's Steam app id, when known: its Proton tool is looked up with it.
    pub steam_appid: Option<&'a str>,
    /// `$XDG_SESSION_TYPE`.
    pub session_type: &'a str,
    /// GPU, driver and compute capability: [`probe_system`] or the caller's own.
    pub system: SystemInfo,
}

/// The machine's GPU facts. Empty strings print as "unknown".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemInfo {
    pub gpu: String,
    pub driver: String,
    pub compute_capability: String,
}

/// GPU names and the driver version from `<proc_root>/driver/nvidia`, and the compute capability
/// from `nvidia_smi`, the output of `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
/// (`None` when it could not be run).
pub fn probe_system(proc_root: &Path, nvidia_smi: Option<&str>) -> SystemInfo {
    let nvidia = proc_root.join("driver/nvidia");
    let mut gpus: Vec<String> = std::fs::read_dir(nvidia.join("gpus"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|gpu| std::fs::read_to_string(gpu.path().join("information")).ok())
        .filter_map(|info| info.lines().find_map(|l| l.strip_prefix("Model:").map(|m| m.trim().to_string())))
        .collect();
    gpus.sort();
    let driver = std::fs::read_to_string(nvidia.join("version")).ok().and_then(|v| driver_version(&v)).unwrap_or_default();
    let compute = nvidia_smi
        .map(|out| out.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    SystemInfo { gpu: gpus.join(", "), driver, compute_capability: compute }
}

/// The version in `/proc/driver/nvidia/version`'s first line ("NVRM version: NVIDIA UNIX Open Kernel
/// Module for x86_64  615.78.08  Release Build ...").
fn driver_version(text: &str) -> Option<String> {
    let first = text.lines().next()?;
    first
        .split_whitespace()
        .find(|t| t.contains('.') && t.split('.').all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())))
        .map(str::to_string)
}

/// Runs `nvidia-smi` for [`probe_system`]'s compute capability.
pub fn run_nvidia_smi() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Replaces the home directory with `~` and the user name with `<user>` (see [`Redactor::redact`]).
#[derive(Debug, Clone)]
pub struct Redactor {
    /// Home directory spellings, longest first, without a trailing `/`.
    homes: Vec<String>,
    user: Option<String>,
}

/// User names shorter than this are not replaced on their own: a one- or two-letter name matches
/// ordinary words everywhere. The home directory, which contains it, is still replaced.
const MIN_USER_LEN: usize = 3;

impl Redactor {
    /// `homes`: every spelling of the home directory (the `$HOME` value and its resolved path, which
    /// differ where `/home` is a symlink). A home of `/` is ignored. `user`: the login name.
    pub fn new(homes: &[&str], user: &str) -> Self {
        let mut homes: Vec<String> = homes.iter().map(|h| h.trim_end_matches('/').to_string()).filter(|h| !h.is_empty()).collect();
        homes.sort_by_key(|h| std::cmp::Reverse(h.len()));
        homes.dedup();
        let user = Some(user.to_string()).filter(|u| u.chars().count() >= MIN_USER_LEN);
        Redactor { homes, user }
    }

    /// From `$HOME` (and its resolved path) and `$USER` (else `$LOGNAME`, else the home's last part).
    pub fn from_env(home: &Path) -> Self {
        let resolved = std::fs::canonicalize(home).ok();
        let mut homes = vec![home.to_string_lossy().into_owned()];
        if let Some(resolved) = resolved {
            homes.push(resolved.to_string_lossy().into_owned());
        }
        let user = ["USER", "LOGNAME"]
            .iter()
            .find_map(|v| std::env::var(v).ok().filter(|u| !u.is_empty()))
            .or_else(|| home.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let refs: Vec<&str> = homes.iter().map(String::as_str).collect();
        Redactor::new(&refs, &user)
    }

    /// The home directory first (the longest spelling first), only where the path ends or goes on
    /// with `/`: `/home/alexa` is another directory, not `~a`. Then the user name, only as a whole
    /// token: the characters on either side must not be letters, digits or `_`. So `alex` in
    /// `alex-fps.png`, `user=alex` or `/tmp/alex/` is replaced, and `alexa`, `alex2` or `xalex` are
    /// not; a name inside a longer word is far more often a word than the name.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for home in &self.homes {
            out = replace_bounded(&out, home, "~", |c| c == '/' || !is_path_char(c), |_| true);
        }
        if let Some(user) = &self.user {
            out = replace_bounded(&out, user, "<user>", |c| !is_word_char(c), |c| !is_word_char(c));
        }
        out
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Characters that continue a path component (so `/home/alex` followed by one is a longer name).
fn is_path_char(c: char) -> bool {
    is_word_char(c) || matches!(c, '-' | '.' | '+' | '@' | '~')
}

/// Replaces each `needle` in `text` whose following character passes `after_ok` and whose preceding
/// character passes `before_ok` (the text's ends always pass).
fn replace_bounded(text: &str, needle: &str, with: &str, after_ok: impl Fn(char) -> bool, before_ok: impl Fn(char) -> bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut prev: Option<char> = None;
    while let Some(at) = rest.find(needle) {
        let before = rest[..at].chars().next_back().or(if at == 0 { prev } else { None });
        let after = rest[at + needle.len()..].chars().next();
        out.push_str(&rest[..at]);
        if before.is_none_or(&before_ok) && after.is_none_or(&after_ok) {
            out.push_str(with);
        } else {
            out.push_str(needle);
        }
        prev = needle.chars().next_back();
        rest = &rest[at + needle.len()..];
    }
    out.push_str(rest);
    out
}

/// `XDG_DESKTOP_DIR` from `<config_home>/user-dirs.dirs` as xdg-user-dirs writes it
/// (`XDG_DESKTOP_DIR="$HOME/Desktop"` or an absolute path in quotes), else `~/Desktop`, else the home
/// directory -- the first that is a directory.
pub fn desktop_dir(config_home: &Path, home: &Path) -> PathBuf {
    let configured = std::fs::read_to_string(config_home.join("user-dirs.dirs")).ok().and_then(|text| user_dir(&text, "XDG_DESKTOP_DIR", home));
    configured.into_iter().chain([home.join("Desktop")]).find(|d| d.is_dir()).unwrap_or_else(|| home.to_path_buf())
}

/// One `NAME="value"` line of a `user-dirs.dirs` file: the value is `$HOME/...`, `$HOME` or an
/// absolute path; anything else is not a valid entry (xdg-user-dirs' own format).
fn user_dir(text: &str, name: &str, home: &Path) -> Option<PathBuf> {
    let value = text.lines().rev().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        line.strip_prefix(name)?.trim_start().strip_prefix('=')?.trim().strip_prefix('"')?.strip_suffix('"').map(str::to_string)
    })?;
    let value = value.replace("\\\"", "\"").replace("\\\\", "\\");
    if value == "$HOME" {
        Some(home.to_path_buf())
    } else if let Some(rest) = value.strip_prefix("$HOME/") {
        Some(home.join(rest))
    } else if value.starts_with('/') {
        Some(PathBuf::from(value))
    } else {
        None
    }
}

/// The last [`LOG_CAP`] bytes of the file at `path` (all of it when shorter), starting at a line,
/// with a first line saying how much was left out; `None` when it cannot be read.
pub fn log_tail(path: &Path, cap: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len <= cap {
        let mut all = Vec::new();
        file.read_to_end(&mut all).ok()?;
        return Some(all);
    }
    // One byte more than the cap: the byte before the kept part says whether it starts a line.
    file.seek(SeekFrom::Start(len - cap - 1)).ok()?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail).ok()?;
    // Start at the first whole line, unless the tail is one line (then it is cut, not dropped).
    let start = match tail.iter().position(|&b| b == b'\n') {
        Some(nl) if nl + 1 < tail.len() => nl + 1,
        _ => 1,
    };
    let tail = &tail[start..];
    let omitted = len - tail.len() as u64;
    let mut out = format!("[... first {omitted} bytes left out ...]\n").into_bytes();
    out.extend_from_slice(tail);
    Some(out)
}

/// The game's Proton tool from Steam's `config/config.vdf` (`CompatToolMapping`): the app's own
/// entry, else the global default (`"0"`), with which one it was. The first Steam root with a
/// readable `config.vdf` decides.
pub fn proton_tool(steam_roots: &[PathBuf], appid: &str) -> Option<String> {
    let text = steam_roots.iter().find_map(|root| std::fs::read(root.join("config/config.vdf")).ok())?;
    let tree = crate::vdf::parse(&text).ok()?;
    let mapping = tree.block_at(&["InstallConfigStore", "Software", "Valve", "Steam", "CompatToolMapping"])?;
    for (key, what) in [(appid, "this game's setting"), ("0", "Steam's default for every game")] {
        if let Some(name) = mapping.string_at(&[key, "name"]).filter(|n| !n.is_empty()) {
            return Some(format!("{name} ({what})"));
        }
    }
    None
}

/// The newest capture pair under `captures_dir`: `<stamp>-original.png` with its `-composited.png`,
/// at the top level or the last pair of a `series-<ms>/` directory, by the original's modification
/// time. Returns (name in the zip, path) for both halves.
pub fn newest_capture_pair(captures_dir: &Path) -> Option<[(String, PathBuf); 2]> {
    let mut candidates: Vec<(std::time::SystemTime, String, PathBuf)> = Vec::new();
    let mut scan = |dir: &Path, prefix: &str| {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(stem) = name.strip_suffix("-original.png") else { continue };
            if !dir.join(format!("{stem}-composited.png")).is_file() {
                continue;
            }
            if let Ok(modified) = entry.metadata().and_then(|m| m.modified()) {
                candidates.push((modified, format!("{prefix}{stem}"), dir.join(stem)));
            }
        }
    };
    scan(captures_dir, "");
    for entry in std::fs::read_dir(captures_dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("series-") && entry.path().is_dir() {
            scan(&entry.path(), &format!("{name}-"));
        }
    }
    let (_, name, base) = candidates.into_iter().max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))?;
    let half = |which: &str| (format!("captures/{name}-{which}.png"), PathBuf::from(format!("{}-{which}.png", base.display())));
    Some([half("original"), half("composited")])
}

/// One file in the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub data: Vec<u8>,
    /// Text members are redacted and deflated; binary ones (the PNGs) are stored as they are.
    pub text: bool,
}

fn or_unknown(s: &str) -> &str {
    if s.trim().is_empty() { "unknown" } else { s.trim() }
}

/// `summary.txt`, before redaction.
pub fn summary(sources: &Sources, inputs: &Inputs, created: u64) -> String {
    let kernel = std::fs::read_to_string(sources.proc_root.join("sys/kernel/osrelease")).unwrap_or_default();
    let proton = match inputs.steam_appid {
        Some(appid) => proton_tool(&sources.steam_roots, appid).unwrap_or_else(|| format!("unknown (no Steam mapping found for app {appid})")),
        None => "unknown (the game's Steam app id is not known)".to_string(),
    };
    let mut s = format!(
        "Neural Forge diagnostic report\n\
         created: {created} (unix time)\n\
         version: {}\n\
         GPU: {}\n\
         driver: {}\n\
         compute capability: {}\n\
         kernel: {}\n\
         session type: {}\n\
         game: {}\n\
         Proton: {proton}\n\n\
         Findings:\n",
        env!("CARGO_PKG_VERSION"),
        or_unknown(&inputs.system.gpu),
        or_unknown(&inputs.system.driver),
        or_unknown(&inputs.system.compute_capability),
        or_unknown(&kernel),
        or_unknown(inputs.session_type),
        if inputs.game.trim().is_empty() { "none named" } else { inputs.game.trim() },
    );
    match inputs.findings {
        Some(findings) if !findings.trim().is_empty() => {
            s.push_str(findings.trim_end());
            s.push('\n');
        }
        Some(_) => s.push_str("none\n"),
        None => s.push_str("not gathered\n"),
    }
    s
}

/// Every member of the report, redacted, in the order they are written.
pub fn collect(sources: &Sources, inputs: &Inputs, redactor: &Redactor, created: u64) -> Vec<Member> {
    let mut members = vec![Member { name: "summary.txt".into(), data: summary(sources, inputs, created).into_bytes(), text: true }];
    for (name, path) in [
        ("config.ini", sources.config_home.join("neural-forge/config.ini")),
        ("model/manifest.json", sources.model_dir.join("manifest.json")),
        ("layer.log", sources.state_dir.join("layer.log")),
        ("layer.log.1", sources.state_dir.join("layer.log.1")),
    ] {
        if let Some(data) = log_tail(&path, LOG_CAP) {
            members.push(Member { name: name.into(), data, text: true });
        }
    }
    if let Some(xid) = inputs.xid_lines {
        let data = if xid.trim().is_empty() { "no Xid lines\n".to_string() } else { xid.to_string() };
        members.push(Member { name: "xid.txt".into(), data: data.into_bytes(), text: true });
    }
    members.push(Member { name: "channel-status.txt".into(), data: inputs.channel_status.as_bytes().to_vec(), text: true });
    if let Some(pair) = newest_capture_pair(&sources.captures_dir) {
        for (name, path) in pair {
            if let Ok(data) = std::fs::read(&path) {
                members.push(Member { name, data, text: false });
            }
        }
    }
    for member in &mut members {
        member.name = redactor.redact(&member.name);
        if member.text {
            member.data = redactor.redact(&String::from_utf8_lossy(&member.data)).into_bytes();
        }
    }
    members
}

/// Writes the report into `out_dir` as `neural-forge-report-<created>.zip` and returns its path.
pub fn write_report(out_dir: &Path, sources: &Sources, inputs: &Inputs, redactor: &Redactor, created: u64) -> std::io::Result<PathBuf> {
    let members = collect(sources, inputs, redactor, created);
    let bytes = zip::write(&members, created)?;
    let path = out_dir.join(format!("neural-forge-report-{created}.zip"));
    crate::paths::write_atomic(&path, &bytes, 0o600)?;
    Ok(path)
}

/// The front ends' one call: the report from the real system ([`Sources::from_env`],
/// [`probe_system`], `$XDG_SESSION_TYPE`), with `doctor`'s findings and this boot's Xid lines,
/// redacted for this user, written to [`desktop_dir`]. `game` is the channel's game (its
/// executable); without `steam_appid`, the app id for its Proton is looked up from it.
pub fn save(channel_status: &str, game: &str, steam_appid: Option<&str>) -> std::io::Result<PathBuf> {
    let sources = Sources::from_env();
    let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
    let doctor_roots = crate::doctor::Roots::system();
    let findings = crate::doctor::render_text(&crate::doctor::run(&doctor_roots));
    let xid_lines = crate::doctor::xid_lines(&doctor_roots);
    let found_appid = match steam_appid {
        Some(_) => None,
        None => crate::steam::game_for_exe(&crate::steam::games(&doctor_roots.steam_roots), game).map(|g| g.appid.clone()),
    };
    let inputs = Inputs {
        findings: Some(&findings),
        xid_lines: Some(&xid_lines),
        channel_status,
        game,
        steam_appid: steam_appid.or(found_appid.as_deref()),
        session_type: &session_type,
        system: probe_system(&sources.proc_root, run_nvidia_smi().as_deref()),
    };
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let out_dir = desktop_dir(&sources.config_home, &sources.home);
    write_report(&out_dir, &sources, &inputs, &Redactor::from_env(&sources.home), created)
}

/// A minimal zip writer: local headers, the central directory and the end record; deflate or store.
pub mod zip {
    use super::Member;

    const CRC_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };

    /// CRC-32 (IEEE 802.3, as zip uses it).
    pub fn crc32(data: &[u8]) -> u32 {
        !data.iter().fold(!0u32, |c, &b| CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8))
    }

    /// MS-DOS date and time (UTC) for `unix` seconds; 1980-01-01 for anything earlier.
    pub fn dos_time(unix: u64) -> (u16, u16) {
        let days = unix / 86_400;
        let secs = unix % 86_400;
        // Days to civil date (Howard Hinnant's algorithm).
        let z = days as i64 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        if year < 1980 {
            return (0, (1 << 5) | 1);
        }
        let time = ((secs / 3600) << 11) | (((secs % 3600) / 60) << 5) | ((secs % 60) / 2);
        let date = (((year - 1980).min(127) as u64) << 9) | ((month as u64) << 5) | day as u64;
        (time as u16, date as u16)
    }

    fn too_big() -> std::io::Error {
        std::io::Error::other("a report member is too large for a zip without ZIP64")
    }

    /// The zip holding `members`, stamped with `modified` (unix seconds).
    pub fn write(members: &[Member], modified: u64) -> std::io::Result<Vec<u8>> {
        let (time, date) = dos_time(modified);
        let mut out = Vec::new();
        let mut central = Vec::new();
        for member in members {
            let name = member.name.as_bytes();
            let crc = crc32(&member.data);
            let (method, body) = if member.text { (8u16, miniz_oxide::deflate::compress_to_vec(&member.data, 6)) } else { (0u16, member.data.clone()) };
            let offset = u32::try_from(out.len()).map_err(|_| too_big())?;
            let size = u32::try_from(member.data.len()).map_err(|_| too_big())?;
            let packed = u32::try_from(body.len()).map_err(|_| too_big())?;
            let name_len = u16::try_from(name.len()).map_err(|_| too_big())?;
            // Bit 11: the name is UTF-8.
            let flags: u16 = 1 << 11;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            for v in [20u16, flags, method, time, date] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in [crc, packed, size] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&body);

            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            // Made by: Unix (3), spec 2.0; needed to extract: 2.0.
            for v in [(3u16 << 8) | 20, 20, flags, method, time, date] {
                central.extend_from_slice(&v.to_le_bytes());
            }
            for v in [crc, packed, size] {
                central.extend_from_slice(&v.to_le_bytes());
            }
            // Name length, extra, comment, disk number, internal attributes.
            for v in [name_len, 0, 0, 0, 0] {
                central.extend_from_slice(&v.to_le_bytes());
            }
            // External attributes: a regular file, rw-------.
            central.extend_from_slice(&(0o100_600u32 << 16).to_le_bytes());
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name);
        }
        let entries = u16::try_from(members.len()).map_err(|_| too_big())?;
        let central_at = u32::try_from(out.len()).map_err(|_| too_big())?;
        let central_len = u32::try_from(central.len()).map_err(|_| too_big())?;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        for v in [0u16, 0, entries, entries] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&central_len.to_le_bytes());
        out.extend_from_slice(&central_at.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neural-forge-report-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sources(root: &Path) -> Sources {
        let home = root.join("home/alex");
        Sources {
            config_home: home.join(".config"),
            model_dir: home.join(".local/share/neural-forge/model"),
            state_dir: home.join(".local/state/neural-forge"),
            captures_dir: home.join(".local/share/neural-forge/captures"),
            proc_root: root.join("proc"),
            steam_roots: vec![home.join(".local/share/Steam"), home.join(".steam/steam")],
            home,
        }
    }

    fn write(path: &Path, data: impl AsRef<[u8]>) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, data).unwrap();
    }

    /// Reads a zip back: (name, method, data) per member, checking every header against the central
    /// directory and every CRC.
    fn read_zip(bytes: &[u8]) -> Vec<(String, u16, Vec<u8>)> {
        let u16_at = |at: usize| u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let end = bytes.len() - 22;
        assert_eq!(u32_at(end), 0x0605_4b50, "end record");
        let count = usize::from(u16_at(end + 10));
        let mut at = u32_at(end + 16) as usize;
        assert_eq!(at + u32_at(end + 12) as usize, end, "the central directory ends where the end record starts");
        let mut members = Vec::new();
        for _ in 0..count {
            assert_eq!(u32_at(at), 0x0201_4b50);
            let (method, crc, packed, size) = (u16_at(at + 10), u32_at(at + 16), u32_at(at + 20) as usize, u32_at(at + 24) as usize);
            let name_len = usize::from(u16_at(at + 28));
            let local = u32_at(at + 42) as usize;
            let name = String::from_utf8(bytes[at + 46..at + 46 + name_len].to_vec()).unwrap();
            assert_eq!(u32_at(local), 0x0403_4b50);
            assert_eq!((u16_at(local + 8), u32_at(local + 14), u32_at(local + 18) as usize), (method, crc, packed), "{name}: local header");
            let data_at = local + 30 + usize::from(u16_at(local + 26)) + usize::from(u16_at(local + 28));
            let body = &bytes[data_at..data_at + packed];
            let data = match method {
                0 => body.to_vec(),
                8 => miniz_oxide::inflate::decompress_to_vec(body).unwrap(),
                other => panic!("{name}: method {other}"),
            };
            assert_eq!(data.len(), size, "{name}: size");
            assert_eq!(zip::crc32(&data), crc, "{name}: CRC");
            members.push((name, method, data));
            at += 46 + name_len + usize::from(u16_at(at + 30)) + usize::from(u16_at(at + 32));
        }
        members
    }

    fn fixture(root: &Path) -> Sources {
        let s = sources(root);
        let home = s.home.display().to_string();
        write(&s.config_home.join("neural-forge/config.ini"), format!("binaries={home}/.local/share/neural-forge/binaries\nset_intensity=1\n"));
        write(&s.model_dir.join("manifest.json"), format!("{{\"source\": {{\"dll\": \"{home}/Games/nvngx_dlssnr.dll\"}}}}\n"));
        write(&s.model_dir.join("model/vit.e4m3"), b"weights");
        write(&s.state_dir.join("layer.log"), format!("[neural-forge-layer] loaded from {home}/.local/lib\nuser alex started, alexa did not\n"));
        write(&s.proc_root.join("sys/kernel/osrelease"), "6.17.0-5-generic\n");
        s
    }

    fn inputs<'a>(status: &'a str, findings: Option<&'a str>, xid: Option<&'a str>) -> Inputs<'a> {
        Inputs {
            findings,
            xid_lines: xid,
            channel_status: status,
            game: "GTA5_Enhanced.exe",
            steam_appid: None,
            session_type: "wayland",
            system: SystemInfo { gpu: "NVIDIA GeForce RTX 5070".into(), driver: "615.78.08".into(), compute_capability: "12.0".into() },
        }
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(zip::crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(zip::crc32(b""), 0);
    }

    #[test]
    fn dos_time_converts_unix_seconds() {
        // 2026-10-10 12:34:56 UTC.
        let (time, date) = zip::dos_time(1_791_635_696);
        assert_eq!((date >> 9, (date >> 5) & 15, date & 31), (46, 10, 10));
        assert_eq!((time >> 11, (time >> 5) & 63, (time & 31) * 2), (12, 34, 56));
        assert_eq!(zip::dos_time(0), (0, (1 << 5) | 1), "before 1980 clamps to 1980-01-01");
    }

    #[test]
    fn the_report_is_a_valid_zip_with_every_part_redacted() {
        let root = scratch("full");
        let s = fixture(&root);
        write(&s.state_dir.join("layer.log.1"), "older\n");
        write(&s.captures_dir.join("1000-original.png"), b"\x89PNG orig /home/alex");
        write(&s.captures_dir.join("1000-composited.png"), b"\x89PNG comp");
        let home = s.home.display().to_string();
        let status = format!("# live status\nlayer_reason=model at {home}/x\n");
        let findings = format!("[warn] the model in {home}/.local/share is old");
        let redactor = Redactor::new(&[home.as_str()], "alex");
        let path = write_report(&root, &s, &inputs(&status, Some(findings.as_str()), Some("NVRM: Xid 79 for alex")), &redactor, 1_791_635_696).unwrap();
        assert_eq!(path, root.join("neural-forge-report-1791635696.zip"));
        let members = read_zip(&std::fs::read(&path).unwrap());
        let names: Vec<&str> = members.iter().map(|(n, ..)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "summary.txt",
                "config.ini",
                "model/manifest.json",
                "layer.log",
                "layer.log.1",
                "xid.txt",
                "channel-status.txt",
                "captures/1000-original.png",
                "captures/1000-composited.png"
            ]
        );
        let get = |name: &str| String::from_utf8_lossy(&members.iter().find(|(n, ..)| n == name).unwrap().2).into_owned();
        for (name, method, data) in &members {
            let text = String::from_utf8_lossy(data);
            if name.ends_with(".png") {
                assert_eq!(*method, 0, "{name} is stored");
                continue;
            }
            assert_eq!(*method, 8, "{name} is deflated");
            assert!(!text.contains(&home) && !text.contains("/home/"), "{name} still names the home directory:\n{text}");
            assert!(!text.contains("alex ") && !text.contains("alex\n"), "{name} still names the user:\n{text}");
        }
        assert!(get("summary.txt").contains("GPU: NVIDIA GeForce RTX 5070\n"));
        assert!(get("summary.txt").contains("kernel: 6.17.0-5-generic\n"));
        assert!(get("summary.txt").contains("session type: wayland\n"));
        assert!(get("summary.txt").contains("[warn] the model in ~/.local/share is old\n"));
        assert!(get("config.ini").starts_with("binaries=~/.local/share/neural-forge/binaries\n"));
        assert!(get("model/manifest.json").contains("\"~/Games/nvngx_dlssnr.dll\""));
        assert_eq!(get("layer.log"), "[neural-forge-layer] loaded from ~/.local/lib\nuser <user> started, alexa did not\n");
        assert_eq!(get("xid.txt"), "NVRM: Xid 79 for <user>");
        assert_eq!(get("channel-status.txt"), "# live status\nlayer_reason=model at ~/x\n");
        let png = &members.iter().find(|(n, ..)| n == "captures/1000-original.png").unwrap().2;
        assert_eq!(png.as_slice(), b"\x89PNG orig /home/alex", "a PNG is never rewritten");
        assert!(!names.iter().any(|n| n.contains("e4m3")), "never the weights");

        // python3's zipfile agrees, when there is one (CI has it).
        let check = "import sys, zipfile\nz = zipfile.ZipFile(sys.argv[1])\nassert z.testzip() is None\nprint(len(z.namelist()))";
        match std::process::Command::new("python3").args(["-I", "-c", check]).arg(&path).output() {
            Ok(out) => {
                assert!(out.status.success(), "python3 zipfile: {}", String::from_utf8_lossy(&out.stderr));
                assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "9");
            }
            Err(_) => eprintln!("no python3: skipping the zipfile check"),
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn missing_parts_are_left_out_and_unknowns_say_so() {
        let root = scratch("sparse");
        let s = sources(&root);
        let mut i = inputs("channel not open: unavailable\n", None, None);
        i.system = SystemInfo::default();
        i.game = "";
        i.session_type = "";
        let members = collect(&s, &i, &Redactor::new(&[], ""), 1);
        let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["summary.txt", "channel-status.txt"]);
        let summary = String::from_utf8(members[0].data.clone()).unwrap();
        for line in ["GPU: unknown", "driver: unknown", "compute capability: unknown", "kernel: unknown", "session type: unknown", "game: none named"] {
            assert!(summary.contains(line), "{line} in\n{summary}");
        }
        assert!(summary.contains("Proton: unknown (the game's Steam app id is not known)"));
        assert!(summary.ends_with("Findings:\nnot gathered\n"), "{summary}");
        i.findings = Some("  \n");
        i.xid_lines = Some("");
        let members = collect(&s, &i, &Redactor::new(&[], ""), 1);
        assert!(String::from_utf8_lossy(&members[0].data).ends_with("Findings:\nnone\n"));
        assert_eq!(members.iter().find(|m| m.name == "xid.txt").map(|m| m.data.as_slice()), Some(&b"no Xid lines\n"[..]));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn redaction_replaces_the_home_first_and_the_user_only_as_a_token() {
        let r = Redactor::new(&["/home/alex/", "/var/home/alex"], "alex");
        assert_eq!(r.redact("/home/alex"), "~");
        assert_eq!(r.redact("/home/alex/x and /var/home/alex/y"), "~/x and ~/y", "every spelling, the longest first");
        assert_eq!(r.redact("cwd=/home/alex, done"), "cwd=~, done");
        assert_eq!(r.redact("/home/alexa/x"), "/home/alexa/x", "another user's home is not this one");
        assert_eq!(r.redact("/home/alex.old/x"), "/home/<user>.old/x", "a longer directory name keeps its path; the name is still a token");
        assert_eq!(r.redact("alex"), "<user>");
        assert_eq!(r.redact("user=alex; /tmp/alex/x alex-fps.png (alex)"), "user=<user>; /tmp/<user>/x <user>-fps.png (<user>)");
        assert_eq!(r.redact("alexa alex2 xalex alex_b Alex"), "alexa alex2 xalex alex_b Alex", "inside a word, or another case, is not the name");
        assert_eq!(r.redact("alexalex alex alex"), "alexalex <user> <user>");
        assert_eq!(r.redact("élan alexé"), "élan alexé", "letters beyond ASCII bound a token too");
    }

    #[test]
    fn a_root_home_or_a_short_user_name_is_not_replaced() {
        let r = Redactor::new(&["/"], "al");
        assert_eq!(r.redact("/usr/lib al /al/"), "/usr/lib al /al/");
        let r = Redactor::new(&["/home/al"], "al");
        assert_eq!(r.redact("/home/al/x al"), "~/x al", "the home still goes");
    }

    #[test]
    fn member_names_are_redacted_too() {
        let root = scratch("names");
        let s = sources(&root);
        write(&s.captures_dir.join("series-5/alex-original.png"), b"o");
        write(&s.captures_dir.join("series-5/alex-composited.png"), b"c");
        let members = collect(&s, &inputs("", None, None), &Redactor::new(&[], "alex"), 1);
        let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"captures/series-5-<user>-original.png"), "{names:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_newest_capture_pair_wins_and_half_pairs_are_skipped() {
        let root = scratch("captures");
        let dir = root.join("captures");
        assert!(newest_capture_pair(&dir).is_none(), "no directory");
        let set_age = |path: &Path, secs_ago: u64| {
            let t = std::time::SystemTime::now() - std::time::Duration::from_secs(secs_ago);
            std::fs::File::options().write(true).open(path).unwrap().set_modified(t).unwrap();
        };
        write(&dir.join("100-original.png"), b"o");
        write(&dir.join("100-composited.png"), b"c");
        set_age(&dir.join("100-original.png"), 300);
        write(&dir.join("200-original.png"), b"o"); // no composited half
        write(&dir.join("series-150/000001-original.png"), b"o");
        write(&dir.join("series-150/000001-composited.png"), b"c");
        set_age(&dir.join("series-150/000001-original.png"), 200);
        write(&dir.join("preupscale-170/colour-preview.png"), b"p");
        let [original, composited] = newest_capture_pair(&dir).unwrap();
        assert_eq!(original, ("captures/series-150-000001-original.png".to_string(), dir.join("series-150/000001-original.png")));
        assert_eq!(composited.1, dir.join("series-150/000001-composited.png"));
        set_age(&dir.join("100-original.png"), 10);
        assert_eq!(newest_capture_pair(&dir).unwrap()[0].0, "captures/100-original.png");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_long_log_keeps_its_end_from_a_whole_line() {
        let root = scratch("tail");
        let path = root.join("layer.log");
        let mut body = Vec::new();
        for i in 0..1000 {
            body.extend_from_slice(format!("line {i:04}\n").as_bytes());
        }
        write(&path, &body);
        assert_eq!(log_tail(&path, 100_000).unwrap(), body, "a short log is whole");
        let tail = String::from_utf8(log_tail(&path, 25).unwrap()).unwrap();
        // The last 25 bytes start inside "line 0997"; the first whole line is 0998.
        assert_eq!(tail, format!("[... first {} bytes left out ...]\nline 0998\nline 0999\n", body.len() - 20));
        // A cap that ends the omitted part on a line break keeps the whole first line.
        assert_eq!(String::from_utf8(log_tail(&path, 20).unwrap()).unwrap(), format!("[... first {} bytes left out ...]\nline 0998\nline 0999\n", body.len() - 20));
        write(&path, b"no newline at all, longer than the cap");
        assert_eq!(String::from_utf8(log_tail(&path, 8).unwrap()).unwrap(), "[... first 30 bytes left out ...]\n the cap", "one line is cut, not dropped");
        assert!(log_tail(&root.join("absent.log"), 8).is_none());
        // The real cap.
        let big = vec![b'x'; LOG_CAP as usize + 10];
        write(&path, &big);
        assert!(log_tail(&path, LOG_CAP).unwrap().len() <= LOG_CAP as usize + 64);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_desktop_is_the_xdg_setting_then_desktop_then_home() {
        let root = scratch("desktop");
        let home = root.join("home");
        let config = home.join(".config");
        std::fs::create_dir_all(&config).unwrap();
        assert_eq!(desktop_dir(&config, &home), home, "nothing configured, no ~/Desktop");
        std::fs::create_dir_all(home.join("Desktop")).unwrap();
        assert_eq!(desktop_dir(&config, &home), home.join("Desktop"));
        write(&config.join("user-dirs.dirs"), "# written by xdg-user-dirs-update\nXDG_DESKTOP_DIR=\"$HOME/Schreibtisch\"\nXDG_DOWNLOAD_DIR=\"$HOME/Downloads\"\n");
        assert_eq!(desktop_dir(&config, &home), home.join("Desktop"), "a configured directory that does not exist is skipped");
        std::fs::create_dir_all(home.join("Schreibtisch")).unwrap();
        assert_eq!(desktop_dir(&config, &home), home.join("Schreibtisch"));
        let elsewhere = root.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        write(&config.join("user-dirs.dirs"), format!("XDG_DESKTOP_DIR=\"{}\"\n", elsewhere.display()));
        assert_eq!(desktop_dir(&config, &home), elsewhere, "an absolute path");
        write(&config.join("user-dirs.dirs"), "XDG_DESKTOP_DIR=\"$HOME\"\n");
        assert_eq!(desktop_dir(&config, &home), home, "xdg-user-dirs' 'disabled' value");
        write(&config.join("user-dirs.dirs"), "XDG_DESKTOP_DIR=\"relative/dir\"\n#XDG_DESKTOP_DIR=\"$HOME/Schreibtisch\"\n");
        assert_eq!(desktop_dir(&config, &home), home.join("Desktop"), "a relative value and a comment are not entries");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_proton_tool_is_the_games_mapping_else_the_default() {
        let root = scratch("proton");
        let steam = root.join("Steam");
        let roots = vec![root.join("absent"), steam.clone()];
        assert!(proton_tool(&roots, "489830").is_none(), "no config.vdf");
        write(
            &steam.join("config/config.vdf"),
            "\"InstallConfigStore\"\n{\n\t\"Software\"\n\t{\n\t\t\"Valve\"\n\t\t{\n\t\t\t\"Steam\"\n\t\t\t{\n\t\t\t\t// comment\n\t\t\t\t\"CompatToolMapping\"\n\t\t\t\t{\n\t\t\t\t\t\"0\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"proton_experimental\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t}\n\t\t\t\t\t\"3240220\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"GE-Proton10-\\\"25\\\"\"\n\t\t\t\t\t}\n\t\t\t\t\t\"271590\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"\"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n",
        );
        assert_eq!(proton_tool(&roots, "3240220").as_deref(), Some("GE-Proton10-\"25\" (this game's setting)"));
        assert_eq!(proton_tool(&roots, "271590").as_deref(), Some("proton_experimental (Steam's default for every game)"), "an empty name is no mapping");
        assert_eq!(proton_tool(&roots, "1").as_deref(), Some("proton_experimental (Steam's default for every game)"));
        write(&steam.join("config/config.vdf"), "\"InstallConfigStore\" { \"software\" { valve { steam { compattoolmapping { 1 { name proton_9 } } } } } }");
        assert_eq!(proton_tool(&roots, "1").as_deref(), Some("proton_9 (this game's setting)"), "bare tokens and any key case");
        assert!(proton_tool(&roots, "2").is_none());
        write(&steam.join("config/config.vdf"), "\"InstallConfigStore\" { \"Software\" {");
        assert!(proton_tool(&roots, "1").is_none(), "a truncated file finds nothing, and does not panic");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn the_system_probe_reads_proc_and_nvidia_smi() {
        let root = scratch("probe");
        assert_eq!(probe_system(&root, None), SystemInfo::default(), "no NVIDIA driver");
        write(
            &root.join("driver/nvidia/version"),
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.78.08  Release Build  (dvs-builder@U22)  Tue Oct  6 2026\nGCC version:  gcc version 15.2.0\n",
        );
        write(&root.join("driver/nvidia/gpus/0000:01:00.0/information"), "Model: \t\t NVIDIA GeForce RTX 5070\nIRQ:   \t\t 16\n");
        let info = probe_system(&root, Some("12.0\n"));
        assert_eq!(info, SystemInfo { gpu: "NVIDIA GeForce RTX 5070".into(), driver: "615.78.08".into(), compute_capability: "12.0".into() });
        write(&root.join("driver/nvidia/gpus/0000:02:00.0/information"), "Model: \t\t NVIDIA GeForce RTX 3060\n");
        let info = probe_system(&root, Some("12.0\n8.6\n"));
        assert_eq!(info.gpu, "NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 5070");
        assert_eq!(info.compute_capability, "12.0, 8.6");
        assert_eq!(driver_version("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.120  Fri Sep 13 2024"), Some("550.120".into()));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
