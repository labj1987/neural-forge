//! Proton with DLSS's own parameters (docs/PROTON_NGX.md): a copy of an installed Proton (GE-Proton,
//! Proton-CachyOS, Valve's Proton or Proton Experimental) that loads Neural Forge's NGX wrapper in place of
//! NVIDIA's `_nvngx.dll`, so the layer reads what the game hands DLSS (its inputs and output, jitter, motion
//! vector units, reset, exposure) instead of working them out from DLSS's kernels.
//!
//! The copy is made beside the source in Steam's `compatibilitytools.d`, with hard links where the source is on
//! the same filesystem (no extra space), and differs from it in three files, each written new (never through a
//! link): `proton` (two DLL overrides after the `NVIDIA_WINE_DLL_DIR` block, [`patch_proton_script`]),
//! `compatibilitytool.vdf` (its name in Steam) and `files/lib/wine/x86_64-windows/_nvngx.dll` (the wrapper,
//! from Neural Forge's install). Its `version` file stays the source's, so a game's prefix is not upgraded when
//! it switches between the two. A marker file ([`MARKER`]) says it is Neural Forge's: only such copies are
//! listed, updated or removed. Steam lists a new copy after it restarts; a game uses it once chosen in its
//! Properties > Compatibility. Only a Proton built on Wine [`MIN_WINE_MAJOR`] or newer is copied: the wrapper
//! names its ring by the game's Linux pid, which it asks Wine for (`ProcessWineUnixPid`), and older Wine has no
//! answer (Proton 8 and 9).

use std::path::{Path, PathBuf};

/// The copy's name: the source's version name and this.
pub const SUFFIX: &str = "-NeuralForge";
/// In a copy: what it was made from (`source=`, `source_version=`, `wrapper_sha256=`, `wine=`).
pub use neural_forge_protocol::proton_copy::MARKER;
/// The wrapper in Neural Forge's install (under the data dir).
pub const WRAPPER: &str = "lib/neural-forge/proton/_nvngx.dll";
/// Where Wine's 64-bit builtin DLLs live in a Proton tree.
const BUILTIN_DIR: &str = "files/lib/wine/x86_64-windows";
/// Turns the wrapper off for one game (its launch options): `PROTON_NO_NEURAL_FORGE_NGX=1 %command%`.
pub const OFF_ENV: &str = "PROTON_NO_NEURAL_FORGE_NGX";
/// The oldest Wine whose `NtQueryInformationProcess` answers `ProcessWineUnixPid` (1101), which the wrapper needs
/// to name its ring (`crates/ngx-wrapper/src/win.rs`, `unix_pid`): Proton 10 and newer. Proton 9 (Wine 9.0) does
/// not handle the class (its `ntdll.so` never compares with 1101; Proton Experimental's and GE-Proton11's do).
pub const MIN_WINE_MAJOR: u32 = 10;
/// Where the wrapper's rings are (`/dev/shm/neural-forge-ngx-<pid>.ring`, the game's Linux pid).
pub const RING_DIR: &str = "/dev/shm";
/// The ring format this Neural Forge's layer reads (`crates/ngx-wrapper/src/ring.rs` `VERSION`, the layer's
/// `ngx_ring::VERSION`): a ring of another version comes from a copy with an older wrapper.
pub const RING_VERSION: u32 = 2;
/// The first 8 bytes of a ring.
const RING_MAGIC: &[u8; 8] = b"NFNGXRG1";

/// What the user is told before a copy is made (Alex 2026-11-10: a clear notice, no blocking).
pub const NOTICE: &str = "Neural Forge makes a copy of this Proton next to it, named with \u{201c}-NeuralForge\u{201d}. On the same drive the copy \
takes no extra space (hard links). It differs in one thing: games run with it load Neural Forge's small NGX wrapper in place of NVIDIA's \
_nvngx.dll, which passes every DLSS call on unchanged and lets Neural Forge read what the game gives DLSS (its images, camera jitter, \
motion vectors and exposure). Your Proton is not changed. Switching a game on makes the copy and sets it for the game automatically \
(Steam must be closed): there is nothing to choose by hand. \
Online games with anti-cheat may refuse a changed module: use the copy for single-player games. Remove it here at any time.";

/// An installed Proton a copy can be made from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// The directory holding `proton`.
    pub dir: PathBuf,
    /// Its version name (`version`'s second word: `GE-Proton11-12`, `cachyos-11.0-20261015-slr`,
    /// `experimental-11.0-20261027b-x86_64`), else its directory's name.
    pub version: String,
    /// The name Steam shows for it (`display_name`, else its directory's name).
    pub display: String,
}

impl Source {
    /// The copy's name ([`SUFFIX`]).
    pub fn tool_name(&self) -> String {
        format!("{}{SUFFIX}", sanitize(&self.version))
    }
}

/// A copy made by Neural Forge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub dir: PathBuf,
    /// The source's directory and version when the copy was made ([`MARKER`]).
    pub source: PathBuf,
    pub source_version: String,
    /// The SHA-256 of the wrapper in it ([`MARKER`]).
    pub wrapper_sha256: String,
}

fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    s.trim_matches('-').to_string()
}

fn display_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("compatibilitytool.vdf")).ok()?;
    let at = text.find("\"display_name\"")? + "\"display_name\"".len();
    let rest = &text[at..];
    let start = rest.find('"')? + 1;
    let end = start + rest[start..].find('"')?;
    Some(rest[start..end].to_string())
}

/// The names Steam knows a tool in `compatibilitytools.d` by: the keys of its `compatibilitytool.vdf`'s
/// `compat_tools` (what `CompatToolMapping` names), not its folder's name.
fn internal_names(dir: &Path) -> Option<Vec<String>> {
    let text = std::fs::read(dir.join("compatibilitytool.vdf")).ok()?;
    let tree = crate::vdf::parse(&text).ok()?;
    Some(tree.block_at(&["compatibilitytools", "compat_tools"])?.0.iter().map(|e| e.key.clone()).collect())
}

fn source_at(dir: &Path) -> Option<Source> {
    let version = neural_forge_protocol::proton_copy::source_version(dir)?;
    let base = dir.file_name()?.to_string_lossy().into_owned();
    Some(Source { dir: dir.to_path_buf(), version, display: display_of(dir).unwrap_or(base) })
}

/// `root`'s `compatibilitytools.d`.
pub fn tools_dir(root: &Path) -> PathBuf {
    root.join("compatibilitytools.d")
}

/// The Protons a copy can be made from: every tool in `root`'s `compatibilitytools.d` that is not a copy, and
/// Valve's Protons (`steamapps/common/Proton*`) in each library.
pub fn sources(root: &Path, libraries: &[PathBuf]) -> Vec<Source> {
    let mut out = Vec::new();
    let mut add_from = |dir: &Path, filter: &dyn Fn(&str) -> bool| {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| filter(&n.to_string_lossy()))).collect();
        dirs.sort();
        for d in dirs {
            if let Some(s) = source_at(&d) {
                if !out.iter().any(|o: &Source| o.dir == s.dir) {
                    out.push(s);
                }
            }
        }
    };
    add_from(&tools_dir(root), &|n: &str| !n.starts_with('.'));
    for lib in libraries {
        add_from(&lib.join("steamapps/common"), &|n: &str| n.starts_with("Proton"));
    }
    out
}

/// Valve's Protons: the name `CompatToolMapping` gives each, and its Steam app.
const VALVE_TOOLS: &[(&str, &str)] = &[
    ("proton_experimental", "1493710"),
    ("proton_hotfix", "2180100"),
    ("proton_11", "4628710"),
    ("proton_10", "3658110"),
    ("proton_9", "2805730"),
    ("proton_8", "2348590"),
    ("proton_7", "1887720"),
    ("proton_63", "1580130"),
    ("proton_513", "1420170"),
];

/// The install folder of Steam app `appid` in any of `libraries`, from its `appmanifest_<appid>.acf` (Valve's
/// Protons install where their manifest says: `Proton 9.0 (Beta)` holds Proton 9.0).
fn app_dir(libraries: &[PathBuf], appid: &str) -> Option<PathBuf> {
    libraries.iter().find_map(|lib| {
        let text = std::fs::read(lib.join(format!("steamapps/appmanifest_{appid}.acf"))).ok()?;
        let tree = crate::vdf::parse(&text).ok()?;
        let dir = tree.string_at(&["AppState", "installdir"])?;
        let path = lib.join("steamapps/common").join(dir);
        (!dir.is_empty() && !dir.contains('/') && dir != ".." && path.is_dir()).then_some(path)
    })
}

/// The installed Proton Steam calls `name` (a `CompatToolMapping` name): Valve's by its app's manifest, any other
/// by the name in its `compatibilitytool.vdf` (its folder's name only when it has none). `None` when it is not
/// installed, or is one of Neural Forge's copies.
pub fn resolve(root: &Path, libraries: &[PathBuf], name: &str) -> Option<Source> {
    if let Some((_, appid)) = VALVE_TOOLS.iter().find(|(n, _)| *n == name) {
        return app_dir(libraries, appid).and_then(|d| source_at(&d));
    }
    let entries = std::fs::read_dir(tools_dir(root)).ok()?;
    let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs.iter().find_map(|d| {
        let named = match internal_names(d) {
            Some(names) => names.iter().any(|n| n == name),
            None => d.file_name().is_some_and(|f| f.to_string_lossy() == name),
        };
        named.then(|| source_at(d)).flatten()
    })
}

/// The copies in `root`'s `compatibilitytools.d`.
pub fn tools(root: &Path) -> Vec<Tool> {
    let Ok(entries) = std::fs::read_dir(tools_dir(root)) else { return Vec::new() };
    let mut out: Vec<Tool> = entries.flatten().filter(|e| !e.file_name().to_string_lossy().starts_with('.')).filter_map(|e| tool_at(&e.path())).collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn tool_at(dir: &Path) -> Option<Tool> {
    if dir.is_symlink() {
        return None;
    }
    let marker = neural_forge_protocol::proton_copy::Marker::read(dir)?;
    Some(Tool {
        name: dir.file_name()?.to_string_lossy().into_owned(),
        dir: dir.to_path_buf(),
        source: marker.source,
        source_version: marker.source_version,
        wrapper_sha256: marker.wrapper_sha256,
    })
}

/// What [`create`] left behind when it stopped part way: hidden `.<name>-NeuralForge.tmp` / `.old` folders.
pub fn leftovers(root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(tools_dir(root)) else { return Vec::new() };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            n.starts_with('.') && (n.ends_with(&format!("{SUFFIX}.tmp")) || n.ends_with(&format!("{SUFFIX}.old")))
        })
        .filter(|p| p.is_dir() && !p.is_symlink())
        .collect();
    out.sort();
    out
}

/// Whether `tool`'s source changed since it was made (a new GE-Proton in a "Latest" folder, a Proton Experimental
/// update): the copy is made again from it. A source that is gone (ProtonUp replaced GE-Proton11-12 with 11-13 in
/// its own folder) is no change: the copy holds its own files (hard links) and keeps working ([`source_gone`]).
pub fn source_changed(tool: &Tool) -> bool {
    newer_source_version(tool).is_some()
}

/// The version `tool`'s source has now when it is not the one the copy was made from: the one check behind
/// [`source_changed`], `doctor`, the Status and Setup tabs and the notice the layer logs at a game's launch
/// (`neural_forge_protocol::proton_copy`).
pub fn newer_source_version(tool: &Tool) -> Option<String> {
    neural_forge_protocol::proton_copy::newer_source(&tool.source, &tool.source_version)
}

/// Whether the Proton `tool` was made from is no longer installed. The copy keeps working as it is; it is only
/// never made again from a newer version.
pub fn source_gone(tool: &Tool) -> bool {
    !tool.source.join("proton").is_file()
}

/// Whether `tool` holds another wrapper than `wrapper` (Neural Forge was updated since it was made): its
/// `_nvngx.dll` is put in again ([`refresh_wrapper`]). False when `wrapper` cannot be read.
pub fn wrapper_changed(tool: &Tool, wrapper: &Path) -> bool {
    std::fs::read(wrapper).is_ok_and(|b| crate::install::digest(&b) != tool.wrapper_sha256)
}

/// Whether `tool` should be updated (`proton-tool update`, Make again): its source changed, or it holds another
/// wrapper than the one installed.
pub fn outdated(tool: &Tool) -> bool {
    source_changed(tool) || wrapper_changed(tool, &installed_wrapper())
}

/// The Wine version a Proton tree is built from: the `wine-<major>.<minor>` its `ntdll.so` carries
/// (`files/lib/wine/x86_64-unix`, or `files/lib64/...` in Proton 9's layout).
pub fn wine_version(dir: &Path) -> Option<(u32, u32)> {
    ["files/lib/wine/x86_64-unix/ntdll.so", "files/lib64/wine/x86_64-unix/ntdll.so"]
        .iter()
        .find_map(|rel| std::fs::read(dir.join(rel)).ok().and_then(|b| wine_version_in(&b)))
}

/// The first `wine-<major>.<minor>` that starts a string in `b`.
fn wine_version_in(b: &[u8]) -> Option<(u32, u32)> {
    let number = |at: usize| -> Option<(u32, usize)> {
        let len = b.get(at..)?.iter().take_while(|c| c.is_ascii_digit()).count();
        (len > 0 && len < 6).then(|| std::str::from_utf8(&b[at..at + len]).ok()?.parse().ok().map(|n| (n, at + len))).flatten()
    };
    let mut from = 0;
    while let Some(k) = b.get(from..)?.windows(5).position(|w| w == b"wine-") {
        let at = from + k;
        from = at + 5;
        // Not inside a longer word (`libwine-`): ntdll.so's string can follow table bytes rather than a NUL (GE-Proton11's
        // `wine-11.0 (Staging)` follows 0x80).
        if at > 0 && (b[at - 1].is_ascii_alphanumeric() || matches!(b[at - 1], b'-' | b'_' | b'.' | b'/')) {
            continue;
        }
        let Some((major, dot)) = number(at + 5) else { continue };
        if b.get(dot) != Some(&b'.') {
            continue;
        }
        if let Some((minor, _)) = number(dot + 1) {
            return Some((major, minor));
        }
    }
    None
}

/// Whether the wrapper can find the game's Linux pid under the Proton at `dir` (Wine [`MIN_WINE_MAJOR`] or
/// newer): its Wine version, or why not.
pub fn unix_pid_support(dir: &Path) -> Result<String, String> {
    match wine_version(dir) {
        Some((major, minor)) if major >= MIN_WINE_MAJOR => Ok(format!("{major}.{minor}")),
        Some((major, minor)) => Err(format!(
            "it is built on Wine {major}.{minor}, and Neural Forge's NGX wrapper needs Wine {MIN_WINE_MAJOR} or newer (Proton 10 or newer, \
             GE-Proton10 or newer) to find the game's process"
        )),
        None => Err(format!(
            "its Wine version could not be read (files/lib/wine/x86_64-unix/ntdll.so), and Neural Forge's NGX wrapper needs Wine \
             {MIN_WINE_MAJOR} or newer (Proton 10 or newer)"
        )),
    }
}

/// The lines added after the `NVIDIA_WINE_DLL_DIR` block, at its indentation.
fn added_lines(indent: &str) -> String {
    let i = indent;
    format!(
        "{i}# Neural Forge (docs/PROTON_NGX.md): NGX loads NVIDIA's signed system32 _nvngx.dll after checking its signature;\n\
         {i}# \"_nvngx=b\" hands it Neural Forge's builtin wrapper ({BUILTIN_DIR}/_nvngx.dll), which loads NVIDIA's core\n\
         {i}# from NVIDIA_WINE_DLL_DIR (that path forced native) and forwards every export. {OFF_ENV}=1 turns it off.\n\
         {i}if self.nvidia_wine_dll_dir and self.env.get(\"{OFF_ENV}\", \"0\") in (\"\", \"0\") and \\\n\
         {i}        os.path.exists(g_proton.lib_dir + \"wine/x86_64-windows/_nvngx.dll\"):\n\
         {i}    self.dlloverrides[\"_nvngx\"] = \"b\"\n\
         {i}    self.dlloverrides[\"Z:\" + self.nvidia_wine_dll_dir.replace(\"/\", \"\\\\\") + \"\\\\_nvngx\"] = \"n\"\n"
    )
}

/// `proton`'s text with the DLL overrides added after the block that sets `NVIDIA_WINE_DLL_DIR` (Valve's code,
/// in every Proton since 9); refused unless that block is there exactly once.
pub fn patch_proton_script(text: &str) -> Result<String, String> {
    const IF: &str = "if self.nvidia_wine_dll_dir:";
    const SET: &str = "self.env[\"NVIDIA_WINE_DLL_DIR\"] = self.nvidia_wine_dll_dir";
    if text.contains(OFF_ENV) {
        return Err("this proton script already loads Neural Forge's wrapper".into());
    }
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut found = Vec::new();
    for (k, pair) in lines.windows(2).enumerate() {
        if pair[0].trim() == IF && pair[1].trim() == SET {
            let indent = &pair[0][..pair[0].len() - pair[0].trim_start().len()];
            let inner = &pair[1][..pair[1].len() - pair[1].trim_start().len()];
            if inner.len() > indent.len() && inner.starts_with(indent) {
                found.push((k + 2, indent.to_string()));
            }
        }
    }
    let [(after, indent)] = found.as_slice() else {
        return Err(format!("the proton script's NVIDIA_WINE_DLL_DIR block was found {} times, not once", found.len()));
    };
    let mut out: String = lines[..*after].concat();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&added_lines(indent));
    out.push_str(&lines[*after..].concat());
    Ok(out)
}

/// `compatibilitytool.vdf` for a copy named `name` (its directory is the tool's).
pub fn manifest(name: &str) -> String {
    format!(
        "\"compatibilitytools\"\n{{\n  \"compat_tools\"\n  {{\n    \"{name}\" // Internal name of this tool\n    {{\n      \"install_path\" \".\"\n      \"display_name\" \"{name}\"\n      \"from_oslist\"  \"windows\"\n      \"to_oslist\"    \"linux\"\n    }}\n  }}\n}}\n"
    )
}

/// Files of the source left out of a copy: Python's cache and ProtonPlus's record of what it installed.
fn skipped(name: &str) -> bool {
    matches!(name, "__pycache__" | ".protonplus")
}

/// `from`'s tree into `to` (which does not exist yet): hard links where possible, else copies; symlinks as
/// symlinks.
fn link_tree(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::create_dir(to)?;
    let mut copied = 0u64;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let name = entry.file_name();
        if skipped(&name.to_string_lossy()) {
            continue;
        }
        let (src, dst) = (entry.path(), to.join(&name));
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            std::os::unix::fs::symlink(std::fs::read_link(&src)?, &dst)?;
        } else if kind.is_dir() {
            copied += link_tree(&src, &dst)?;
        } else if std::fs::hard_link(&src, &dst).is_err() {
            copied += std::fs::copy(&src, &dst)?;
        }
    }
    Ok(copied)
}

/// Writes `content` as a new file at `path`: written to `<name>.tmp` beside it, then renamed over it in one step, so
/// a reader sees the old file or the new one, never a part, and a hard link to the source at `path` is replaced,
/// never written through.
fn write_new(path: &Path, content: &[u8], mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut name = path.file_name().ok_or_else(|| std::io::Error::other("no file name"))?.to_os_string();
    name.push(".tmp");
    let tmp = path.with_file_name(name);
    match std::fs::remove_file(&tmp) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let written = std::fs::write(&tmp, content)
        .and_then(|()| std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode)))
        .and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// The processes under `proc_root` running from `dir` (their executable, or an argument naming a path in it: the
/// `proton` script's Python, Wine's preloader and server): a game runs with that Proton.
pub fn in_use(dir: &Path, proc_root: &Path) -> Vec<u32> {
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let Ok(entries) = std::fs::read_dir(proc_root) else { return Vec::new() };
    let mut out: Vec<u32> = entries
        .flatten()
        .filter_map(|e| {
            let pid: u32 = e.file_name().to_str()?.parse().ok()?;
            let exe = std::fs::read_link(e.path().join("exe")).is_ok_and(|p| p.starts_with(&dir));
            let args = std::fs::read(e.path().join("cmdline")).unwrap_or_default();
            let arg = args.split(|&c| c == 0).any(|a| {
                let a = String::from_utf8_lossy(a);
                a.starts_with('/') && Path::new(a.as_ref()).starts_with(&dir)
            });
            (exe || arg).then_some(pid)
        })
        .collect();
    out.sort_unstable();
    out
}

fn refuse_in_use(dir: &Path, proc_root: &Path, what: &str) -> Result<(), String> {
    match in_use(dir, proc_root).as_slice() {
        [] => Ok(()),
        pids => Err(format!(
            "{} is in use by a running game (process {}): close the game, then {what} again",
            dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            pids.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Swaps the directories `a` and `b` in one step (`renameat2(RENAME_EXCHANGE)`).
fn exchange(a: &Path, b: &Path) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let (a, b) = (std::ffi::CString::new(a.as_os_str().as_bytes())?, std::ffi::CString::new(b.as_os_str().as_bytes())?);
    // SAFETY: two NUL-terminated paths that live across the call.
    let r = unsafe { libc::renameat2(libc::AT_FDCWD, a.as_ptr(), libc::AT_FDCWD, b.as_ptr(), libc::RENAME_EXCHANGE) };
    if r == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// What [`create`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Created {
    pub tool: Tool,
    /// Bytes copied rather than linked (the source on another filesystem).
    pub copied: u64,
    /// A copy of the same name was replaced (an update).
    pub replaced: bool,
}

/// Makes (or makes again) the copy of `source` in `root`'s `compatibilitytools.d` with `wrapper` as its
/// `_nvngx.dll`. Built under a hidden name (`.<name>.tmp`) and swapped into place in one step, so Steam never sees
/// half a copy; a copy a running game uses is not replaced. Refused for a Proton on Wine older than
/// [`MIN_WINE_MAJOR`].
pub fn create(root: &Path, source: &Source, wrapper: &Path) -> Result<Created, String> {
    create_in(root, source, wrapper, Path::new("/proc"))
}

/// [`create`], with the processes read from `proc_root`.
pub fn create_in(root: &Path, source: &Source, wrapper: &Path, proc_root: &Path) -> Result<Created, String> {
    let dll = std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()))?;
    if dll.get(0x40..0x50) != Some(b"Wine builtin DLL".as_slice()) {
        return Err(format!("{} is not a Wine builtin DLL", wrapper.display()));
    }
    let wine = unix_pid_support(&source.dir).map_err(|e| format!("{} cannot be used: {e}", source.display))?;
    if !source.dir.join(BUILTIN_DIR).is_dir() {
        return Err(format!("{} has no {BUILTIN_DIR}: not a Proton layout the wrapper can be put in", source.display));
    }
    let script = std::fs::read_to_string(source.dir.join("proton")).map_err(|e| format!("{}: {e}", source.dir.join("proton").display()))?;
    let patched = patch_proton_script(&script)?;
    let name = source.tool_name();
    let dir = tools_dir(root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (final_dir, tmp, old) = (dir.join(&name), dir.join(format!(".{name}.tmp")), dir.join(format!(".{name}.old")));
    let replaced = final_dir.exists() || final_dir.is_symlink();
    if replaced {
        if tool_at(&final_dir).is_none() {
            return Err(format!("{} exists and is not Neural Forge's", final_dir.display()));
        }
        refuse_in_use(&final_dir, proc_root, "make it")?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_dir_all(&old);
    let fail = |e: std::io::Error| {
        let _ = std::fs::remove_dir_all(&tmp);
        format!("copying {} to {}: {e}", source.dir.display(), tmp.display())
    };
    let copied = link_tree(&source.dir, &tmp).map_err(fail)?;
    let digest = crate::install::digest(&dll);
    let marker = format!(
        "# Made by Neural Forge (neural-forge-cli proton-tool): {} with Neural Forge's NGX wrapper.\nsource={}\nsource_version={}\nwrapper_sha256={digest}\nwine={wine}\n",
        source.display,
        source.dir.display(),
        source.version
    );
    let write = || -> std::io::Result<()> {
        write_new(&tmp.join("proton"), patched.as_bytes(), 0o755)?;
        write_new(&tmp.join("compatibilitytool.vdf"), manifest(&name).as_bytes(), 0o644)?;
        write_new(&tmp.join(BUILTIN_DIR).join("_nvngx.dll"), &dll, 0o644)?;
        write_new(&tmp.join(MARKER), marker.as_bytes(), 0o644)
    };
    write().map_err(fail)?;
    if replaced {
        // One step where the filesystem has it; else the old copy aside, the new one in, the old one gone.
        if exchange(&tmp, &final_dir).is_ok() {
            let _ = std::fs::remove_dir_all(&tmp);
        } else {
            std::fs::rename(&final_dir, &old).map_err(fail)?;
            if let Err(e) = std::fs::rename(&tmp, &final_dir) {
                let _ = std::fs::rename(&old, &final_dir);
                return Err(fail(e));
            }
            let _ = std::fs::remove_dir_all(&old);
        }
    } else {
        std::fs::rename(&tmp, &final_dir).map_err(fail)?;
    }
    let tool = tool_at(&final_dir).ok_or("the copy has no marker")?;
    Ok(Created { tool, copied, replaced })
}

/// Puts `dll` (the installed wrapper) into `tool` when its marker names another one: its `_nvngx.dll` written new
/// (a running game keeps the file it loaded), then the marker. Whether it changed anything.
pub fn refresh_wrapper(tool: &Tool, dll: &[u8]) -> Result<bool, String> {
    let digest = crate::install::digest(dll);
    if tool.wrapper_sha256 == digest {
        return Ok(false);
    }
    let marker_path = tool.dir.join(MARKER);
    let text = std::fs::read_to_string(&marker_path).map_err(|e| format!("{}: {e}", marker_path.display()))?;
    let mut lines: Vec<String> = text.lines().filter(|l| !l.starts_with("wrapper_sha256=")).map(str::to_string).collect();
    lines.push(format!("wrapper_sha256={digest}"));
    let marker = lines.join("\n") + "\n";
    let target = tool.dir.join(BUILTIN_DIR).join("_nvngx.dll");
    write_new(&target, dll, 0o644).map_err(|e| format!("{}: {e}", target.display()))?;
    write_new(&marker_path, marker.as_bytes(), 0o644).map_err(|e| format!("{}: {e}", marker_path.display()))?;
    Ok(true)
}

/// [`refresh_wrapper`] for every copy in `root` with the wrapper at `wrapper` (after Neural Forge is installed or
/// updated): each copy's name and what happened. Nothing when the wrapper cannot be read.
pub fn refresh_wrappers(root: &Path, wrapper: &Path) -> Vec<(String, Result<bool, String>)> {
    let Ok(dll) = std::fs::read(wrapper) else { return Vec::new() };
    tools(root).into_iter().map(|t| (t.name.clone(), refresh_wrapper(&t, &dll))).collect()
}

/// Removes the copy named `name` (only one of Neural Forge's, and not while a game runs with it). Games set to it
/// need another Proton chosen in Steam; their prefixes are set up for that Proton on their next launch.
pub fn remove(root: &Path, name: &str) -> Result<Tool, String> {
    remove_in(root, name, Path::new("/proc"))
}

/// [`remove`], with the processes read from `proc_root`.
pub fn remove_in(root: &Path, name: &str, proc_root: &Path) -> Result<Tool, String> {
    let tool = tools(root).into_iter().find(|t| t.name == name).ok_or_else(|| format!("no Neural Forge Proton named {name}"))?;
    refuse_in_use(&tool.dir, proc_root, "remove it")?;
    std::fs::remove_dir_all(&tool.dir).map_err(|e| format!("{}: {e}", tool.dir.display()))?;
    Ok(tool)
}

/// The pid a ring file is named by (`neural-forge-ngx-<pid>.ring`).
fn ring_pid(name: &str) -> Option<u32> {
    name.strip_prefix("neural-forge-ngx-")?.strip_suffix(".ring")?.parse().ok()
}

/// Whether process `pid` exists (`kill(pid, 0)`: also when it is another user's).
pub fn pid_alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else { return false };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only checks that the process exists.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// This user's rings in `dir`, each with its pid: `neural-forge-ngx-<pid>.ring` files.
fn rings(dir: &Path) -> Vec<(PathBuf, u32)> {
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<(PathBuf, u32)> = entries
        .flatten()
        .filter_map(|e| {
            let pid = ring_pid(&e.file_name().to_string_lossy())?;
            let meta = e.path().symlink_metadata().ok()?;
            (meta.is_file() && std::os::unix::fs::MetadataExt::uid(&meta) == uid).then(|| (e.path(), pid))
        })
        .collect();
    out.sort();
    out
}

/// Removes this user's rings in `dir` whose game is gone (a ring the layer never opened stays after its game
/// exits): never one of a running process. What was removed. Run when Neural Forge starts and by
/// `uninstall --purge`.
pub fn sweep_dead_rings(dir: &Path) -> Vec<PathBuf> {
    sweep_rings_with(dir, pid_alive)
}

fn sweep_rings_with(dir: &Path, alive: impl Fn(u32) -> bool) -> Vec<PathBuf> {
    rings(dir).into_iter().filter(|(_, pid)| !alive(*pid)).filter_map(|(path, _)| std::fs::remove_file(&path).ok().map(|()| path)).collect()
}

/// A running game's ring the layer has not opened (it removes the file once it maps it): its path, pid and format
/// version (`None` when it is not a ring).
pub fn open_rings(dir: &Path) -> Vec<(PathBuf, u32, Option<u32>)> {
    rings(dir)
        .into_iter()
        .filter(|(_, pid)| pid_alive(*pid))
        .map(|(path, pid)| {
            let mut head = [0u8; 12];
            let read = std::fs::File::open(&path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head)).is_ok();
            let version = (read && &head[..8] == RING_MAGIC).then(|| u32::from_le_bytes(head[8..12].try_into().expect("4 bytes")));
            (path, pid, version)
        })
        .collect()
}

/// The names a PE DLL exports (PE32+ only), from its export directory; `None` when the file is not one.
pub fn pe_export_names(b: &[u8]) -> Option<Vec<String>> {
    let u16_at = |at: usize| -> Option<u16> { Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?)) };
    let u32_at = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?)) };
    if b.get(..2)? != b"MZ" {
        return None;
    }
    let pe = u32_at(0x3c)? as usize;
    if b.get(pe..pe + 4)? != b"PE\0\0" {
        return None;
    }
    let sections = u16_at(pe + 6)? as usize;
    let opt_size = u16_at(pe + 20)? as usize;
    let opt = pe + 24;
    if u16_at(opt)? != 0x20b {
        return None;
    }
    let export_rva = u32_at(opt + 112)?;
    let table = opt + opt_size;
    // A relative address as a file offset, through the section that holds it.
    let offset = |rva: u32| -> Option<usize> {
        (0..sections).find_map(|k| {
            let sh = table + k * 40;
            let (vsize, va, raw) = (u32_at(sh + 8)?, u32_at(sh + 12)?, u32_at(sh + 20)?);
            (rva >= va && rva < va + vsize.max(1)).then(|| (rva - va + raw) as usize)
        })
    };
    if export_rva == 0 {
        return Some(Vec::new());
    }
    let dir = offset(export_rva)?;
    let (count, names) = (u32_at(dir + 24)? as usize, offset(u32_at(dir + 32)?)?);
    let mut out = Vec::with_capacity(count);
    for k in 0..count.min(65536) {
        let at = offset(u32_at(names + 4 * k)?)?;
        let end = at + b.get(at..)?.iter().position(|&c| c == 0)?;
        out.push(String::from_utf8_lossy(&b[at..end]).into_owned());
    }
    Some(out)
}

/// NVIDIA's NGX core for Proton in the driver's `nvidia/wine` folder, where the driver puts it.
pub fn driver_core(sys_lib_dirs: &[&Path]) -> Option<PathBuf> {
    sys_lib_dirs.iter().map(|d| d.join("nvidia/wine/_nvngx.dll")).find(|p| p.is_file())
}

/// The library folders the driver's `nvidia/wine` may be in (Debian/Ubuntu, Fedora/openSUSE, Arch).
pub const DRIVER_LIB_DIRS: [&str; 3] = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"];

/// The exports the NGX wrapper implements itself (`crates/ngx-wrapper/src/hooks.rs`; `exports.rs`'s `HOOKED`): it
/// refuses to load when NVIDIA's core lacks one of them. Every other export it forwards to the core, and answers
/// with `0xBAD00000` (NGX's generic failure) where the core lacks it. Keep in step with hooks.rs.
pub const HOOKED: [&str; 10] = [
    "NVSDK_NGX_D3D11_CreateFeature",
    "NVSDK_NGX_D3D11_EvaluateFeature",
    "NVSDK_NGX_D3D11_ReleaseFeature",
    "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_EvaluateFeature",
    "NVSDK_NGX_D3D12_ReleaseFeature",
    "NVSDK_NGX_VULKAN_CreateFeature",
    "NVSDK_NGX_VULKAN_CreateFeature1",
    "NVSDK_NGX_VULKAN_EvaluateFeature",
    "NVSDK_NGX_VULKAN_ReleaseFeature",
];

/// Which Proton Steam runs `appid` with (`config.vdf`'s `CompatToolMapping`), if one is chosen for it.
pub fn mapped_tool(root: &Path, appid: &str) -> Option<String> {
    let text = std::fs::read(root.join("config/config.vdf")).ok()?;
    let tree = crate::vdf::parse(&text).ok()?;
    tree.string_at(&["InstallConfigStore", "Software", "Valve", "Steam", "CompatToolMapping", appid, "name"]).map(str::to_string)
}

/// The installed wrapper.
pub fn installed_wrapper() -> PathBuf {
    Path::new(&crate::paths::data_dir()).join(WRAPPER)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALVE: &str = "        self.nvidia_wine_dll_dir = find_nvidia_wine_dll_dir()\n        if self.nvidia_wine_dll_dir:\n            self.env[\"NVIDIA_WINE_DLL_DIR\"] = self.nvidia_wine_dll_dir\n\n        if \"PROTON_LOG\" in self.env:\n";

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("neural-forge-proton-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_overrides_go_after_the_nvidia_block_at_its_indentation() {
        let out = patch_proton_script(VALVE).unwrap();
        let want = "            self.env[\"NVIDIA_WINE_DLL_DIR\"] = self.nvidia_wine_dll_dir\n        # Neural Forge";
        assert!(out.contains(want), "{out}");
        assert!(out.contains("\n            self.dlloverrides[\"_nvngx\"] = \"b\"\n"));
        assert!(out.contains(r#"self.dlloverrides["Z:" + self.nvidia_wine_dll_dir.replace("/", "\\") + "\\_nvngx"] = "n""#), "{out}");
        assert!(out.ends_with("\n        if \"PROTON_LOG\" in self.env:\n"), "the rest follows unchanged");
        assert!(patch_proton_script(&out).unwrap_err().contains("already"), "never twice");
    }

    #[test]
    fn a_script_without_the_block_once_is_refused() {
        assert!(patch_proton_script("print('hi')\n").unwrap_err().contains("0 times"));
        assert!(patch_proton_script(&format!("{VALVE}{VALVE}")).unwrap_err().contains("2 times"));
    }

    fn fake_proton(dir: &Path, version: &str, display: &str) {
        std::fs::create_dir_all(dir.join("files/lib/wine/x86_64-windows")).unwrap();
        std::fs::create_dir_all(dir.join("__pycache__")).unwrap();
        std::fs::write(dir.join("proton"), VALVE).unwrap();
        std::fs::write(dir.join("version"), format!("1793687165 {version}\n")).unwrap();
        std::fs::write(
            dir.join("compatibilitytool.vdf"),
            format!("\"compatibilitytools\" {{ \"compat_tools\" {{ \"x\" {{ \"display_name\" \"{display}\" }} }} }}\n"),
        )
        .unwrap();
        std::fs::write(dir.join("files/lib/wine/x86_64-windows/ntdll.dll"), "ntdll").unwrap();
        std::os::unix::fs::symlink("ntdll.dll", dir.join("files/lib/wine/x86_64-windows/link.dll")).unwrap();
        fake_ntdll(dir, "lib", "11.0");
    }

    fn fake_ntdll(dir: &Path, lib: &str, wine: &str) {
        let unix = dir.join(format!("files/{lib}/wine/x86_64-unix"));
        std::fs::create_dir_all(&unix).unwrap();
        std::fs::write(unix.join("ntdll.so"), format!("\x7fELF\0wine-staging\0libwine-1.0\0wine-{wine} (Staging)\0")).unwrap();
    }

    fn fake_wrapper(path: &Path) {
        let mut b = vec![0u8; 0x100];
        b[..2].copy_from_slice(b"MZ");
        b[0x40..0x50].copy_from_slice(b"Wine builtin DLL");
        std::fs::write(path, b).unwrap();
    }

    #[test]
    fn a_copy_links_the_source_and_differs_in_three_files() {
        use std::os::unix::fs::MetadataExt;
        let root = scratch("create");
        let src = root.join("compatibilitytools.d/Proton-GE Latest");
        fake_proton(&src, "GE-Proton11-12", "Proton-GE Latest");
        let wrapper = root.join("_nvngx.dll");
        fake_wrapper(&wrapper);
        let found = sources(&root, &[]);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].version.as_str(), found[0].display.as_str()), ("GE-Proton11-12", "Proton-GE Latest"));
        let made = create(&root, &found[0], &wrapper).unwrap();
        let dir = root.join("compatibilitytools.d/GE-Proton11-12-NeuralForge");
        assert_eq!((made.tool.dir.clone(), made.copied, made.replaced), (dir.clone(), 0, false));
        // Linked, not copied; the three files are new and the source's are untouched.
        let ino = |p: &Path| std::fs::metadata(p).unwrap().ino();
        assert_eq!(ino(&dir.join("files/lib/wine/x86_64-windows/ntdll.dll")), ino(&src.join("files/lib/wine/x86_64-windows/ntdll.dll")));
        assert_eq!(std::fs::read_link(dir.join("files/lib/wine/x86_64-windows/link.dll")).unwrap(), Path::new("ntdll.dll"));
        assert_ne!(ino(&dir.join("proton")), ino(&src.join("proton")));
        assert_eq!(std::fs::read_to_string(src.join("proton")).unwrap(), VALVE);
        assert!(std::fs::read_to_string(dir.join("proton")).unwrap().contains(OFF_ENV));
        assert!(std::fs::read_to_string(dir.join("compatibilitytool.vdf")).unwrap().contains("\"display_name\" \"GE-Proton11-12-NeuralForge\""));
        assert_eq!(std::fs::read(dir.join("files/lib/wine/x86_64-windows/_nvngx.dll")).unwrap(), std::fs::read(&wrapper).unwrap());
        assert_eq!(std::fs::read_to_string(dir.join("version")).unwrap(), "1793687165 GE-Proton11-12\n", "the source's version: no prefix upgrade");
        assert!(!dir.join("__pycache__").exists());
        // Listed as a copy, not as a source; up to date until the source changes.
        assert_eq!(sources(&root, &[]).len(), 1);
        let t = &tools(&root)[0];
        assert_eq!((t.name.as_str(), t.source_version.as_str()), ("GE-Proton11-12-NeuralForge", "GE-Proton11-12"));
        assert!(!source_changed(t) && !wrapper_changed(t, &wrapper));
        assert!(std::fs::read_to_string(dir.join(MARKER)).unwrap().contains("\nwine=11.0\n"));
        std::fs::write(src.join("version"), "1793999999 GE-Proton11-13\n").unwrap();
        assert!(source_changed(t));
        // Made again under the new name; the old copy removed by name.
        let again = create(&root, &sources(&root, &[])[0], &wrapper).unwrap();
        assert_eq!(again.tool.name, "GE-Proton11-13-NeuralForge");
        assert_eq!(remove(&root, "GE-Proton11-12-NeuralForge").unwrap().name, "GE-Proton11-12-NeuralForge");
        assert!(remove(&root, "Proton-GE Latest").is_err(), "never a tool that is not ours");
        assert!(src.join("proton").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn export_names_are_read_from_a_pe_file() {
        // A minimal PE32+ with one section and an export directory naming two functions.
        let mut b = vec![0u8; 0x400];
        b[..2].copy_from_slice(b"MZ");
        b[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        b[0x80..0x84].copy_from_slice(b"PE\0\0");
        b[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        b[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        let opt = 0x98;
        b[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        b[opt + 112..opt + 116].copy_from_slice(&0x1000u32.to_le_bytes());
        let sh = opt + 240;
        b[sh + 8..sh + 12].copy_from_slice(&0x200u32.to_le_bytes());
        b[sh + 12..sh + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        b[sh + 20..sh + 24].copy_from_slice(&0x200u32.to_le_bytes());
        let dir = 0x200;
        b[dir + 24..dir + 28].copy_from_slice(&2u32.to_le_bytes());
        b[dir + 32..dir + 36].copy_from_slice(&0x1040u32.to_le_bytes());
        b[0x240..0x244].copy_from_slice(&0x1060u32.to_le_bytes());
        b[0x244..0x248].copy_from_slice(&0x1070u32.to_le_bytes());
        b[0x260..0x263].copy_from_slice(b"Abc");
        b[0x270..0x273].copy_from_slice(b"Xyz");
        assert_eq!(pe_export_names(&b), Some(vec!["Abc".to_string(), "Xyz".to_string()]));
        assert_eq!(pe_export_names(b"not a dll"), None);
    }

    #[test]
    fn the_chosen_proton_is_read_from_config_vdf() {
        let root = scratch("mapping");
        std::fs::create_dir_all(root.join("config")).unwrap();
        std::fs::write(
            root.join("config/config.vdf"),
            "\"InstallConfigStore\"\n{\n\"Software\"\n{\n\"Valve\"\n{\n\"Steam\"\n{\n\"CompatToolMapping\"\n{\n\"1091500\"\n{\n\"name\"\t\t\"GE-Proton11-12-NeuralForge\"\n}\n}\n}\n}\n}\n}\n",
        )
        .unwrap();
        assert_eq!(mapped_tool(&root, "1091500").as_deref(), Some("GE-Proton11-12-NeuralForge"));
        assert_eq!(mapped_tool(&root, "1"), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn valve_protons_in_a_library_are_sources_and_a_wrapper_must_be_a_builtin() {
        let root = scratch("library");
        let lib = root.join("lib");
        fake_proton(&lib.join("steamapps/common/Proton - Experimental"), "experimental-11.0-20261027b-x86_64", "Proton - Experimental");
        std::fs::create_dir_all(lib.join("steamapps/common/Cyberpunk 2077")).unwrap();
        let found = sources(&root, std::slice::from_ref(&lib));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].tool_name(), "experimental-11.0-20261027b-x86_64-NeuralForge");
        let plain = root.join("plain.dll");
        std::fs::write(&plain, b"MZ").unwrap();
        assert!(create(&root, &found[0], &plain).unwrap_err().contains("not a Wine builtin"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_proton_on_wine_older_than_10_or_of_unknown_wine_is_refused() {
        let root = scratch("wine");
        let wrapper = root.join("_nvngx.dll");
        fake_wrapper(&wrapper);
        let src = root.join("compatibilitytools.d/GE-Proton9-27");
        fake_proton(&src, "GE-Proton9-27", "GE-Proton9-27");
        fake_ntdll(&src, "lib", "9.0");
        assert_eq!(wine_version(&src), Some((9, 0)));
        let s = sources(&root, &[]).remove(0);
        let e = create(&root, &s, &wrapper).unwrap_err();
        assert!(e.contains("Wine 9.0") && e.contains("Wine 10 or newer"), "{e}");
        // Proton 9's layout keeps ntdll.so under lib64: still read, still refused.
        std::fs::remove_dir_all(src.join("files/lib/wine/x86_64-unix")).unwrap();
        assert_eq!(wine_version(&src), None);
        assert!(create(&root, &s, &wrapper).unwrap_err().contains("could not be read"));
        fake_ntdll(&src, "lib64", "9.0");
        assert_eq!(wine_version(&src), Some((9, 0)));
        assert!(tools(&root).is_empty() && leftovers(&root).is_empty(), "nothing is left of a refused copy");
        assert_eq!(wine_version_in(b"\0wine-10.12\0"), Some((10, 12)));
        assert_eq!(wine_version_in(b"libwine-1.0\0wine-x\0"), None, "only a string that starts with it");
        assert_eq!(wine_version_in(b"\x08\x80\x80wine-11.0 (Staging)\0"), Some((11, 0)), "GE-Proton11-12: after table bytes, not a NUL");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Valve's Protons are found through their app's manifest (Proton 9.0 is installed in `Proton 9.0 (Beta)`),
    /// other tools by the name their `compatibilitytool.vdf` gives Steam; never one of the copies.
    #[test]
    fn a_mapping_name_resolves_by_manifest_not_by_folder() {
        let root = scratch("resolve");
        let lib = root.join("lib");
        fake_proton(&lib.join("steamapps/common/Proton 11.0 (Beta)"), "proton-11.0-2e", "Proton 11.0");
        std::fs::write(
            lib.join("steamapps/appmanifest_4628710.acf"),
            "\"AppState\"\n{\n\t\"appid\"\t\t\"4628710\"\n\t\"installdir\"\t\t\"Proton 11.0 (Beta)\"\n}\n",
        )
        .unwrap();
        fake_proton(&root.join("compatibilitytools.d/Proton-GE Latest"), "GE-Proton11-12", "Proton-GE Latest");
        std::fs::write(
            root.join("compatibilitytools.d/Proton-GE Latest/compatibilitytool.vdf"),
            "\"compatibilitytools\"\n{\n  \"compat_tools\"\n  {\n    \"GE-Proton-Latest\"\n    {\n      \"display_name\" \"Proton-GE Latest\"\n    }\n  }\n}\n",
        )
        .unwrap();
        let libs = std::slice::from_ref(&lib);
        assert_eq!(resolve(&root, libs, "proton_11").unwrap().version, "proton-11.0-2e");
        assert_eq!(resolve(&root, libs, "GE-Proton-Latest").unwrap().version, "GE-Proton11-12");
        assert_eq!(resolve(&root, libs, "Proton-GE Latest"), None, "the folder's name is not Steam's name for it");
        assert_eq!(resolve(&root, libs, "proton_experimental"), None, "not installed");
        let wrapper = root.join("_nvngx.dll");
        fake_wrapper(&wrapper);
        let made = create(&root, &resolve(&root, libs, "GE-Proton-Latest").unwrap(), &wrapper).unwrap();
        assert_eq!(resolve(&root, libs, &made.tool.name), None, "a copy is never a source");
        std::fs::remove_dir_all(&root).unwrap();
    }

    fn fake_process(proc_root: &Path, pid: u32, exe: &Path, args: &[&str]) {
        let d = proc_root.join(pid.to_string());
        std::fs::create_dir_all(&d).unwrap();
        std::os::unix::fs::symlink(exe, d.join("exe")).unwrap();
        std::fs::write(d.join("cmdline"), args.iter().map(|a| format!("{a}\0")).collect::<String>()).unwrap();
    }

    /// A copy a game runs with is neither replaced nor removed; once the game is gone it is swapped in place, and a
    /// new wrapper is put into every copy that holds another.
    #[test]
    fn a_copy_in_use_is_kept_and_a_new_wrapper_is_put_in() {
        let root = scratch("inuse");
        let src = root.join("compatibilitytools.d/GE-Proton11-12");
        fake_proton(&src, "GE-Proton11-12", "GE-Proton11-12");
        let wrapper = root.join("_nvngx.dll");
        fake_wrapper(&wrapper);
        let s = sources(&root, &[]).remove(0);
        let made = create(&root, &s, &wrapper).unwrap();
        let procs = root.join("proc");
        fake_process(&procs, 4242, Path::new("/usr/bin/python3"), &["python3", &made.tool.dir.join("proton").to_string_lossy(), "waitforexitandrun"]);
        fake_process(&procs, 4243, &made.tool.dir.join("files/bin/wine64-preloader"), &["C:\\game.exe"]);
        fake_process(&procs, 7, Path::new("/usr/bin/bash"), &["bash", &src.join("proton").to_string_lossy()]);
        assert_eq!(in_use(&made.tool.dir, &procs), [4242, 4243], "the source's own users do not count");
        let e = create_in(&root, &s, &wrapper, &procs).unwrap_err();
        assert!(e.contains("in use by a running game (process 4242, 4243)"), "{e}");
        assert!(remove_in(&root, &made.tool.name, &procs).unwrap_err().contains("in use"));
        assert!(made.tool.dir.join(MARKER).is_file(), "untouched");
        std::fs::remove_dir_all(&procs).unwrap();
        let again = create_in(&root, &s, &wrapper, &procs).unwrap();
        assert!(again.replaced && leftovers(&root).is_empty(), "swapped in, nothing left aside");
        // A new wrapper: every copy that holds another gets it, the marker last.
        let mut newer = std::fs::read(&wrapper).unwrap();
        newer.push(1);
        std::fs::write(&wrapper, &newer).unwrap();
        let t = tools(&root).remove(0);
        assert!(wrapper_changed(&t, &wrapper) && !source_changed(&t));
        assert_eq!(refresh_wrappers(&root, &wrapper), [(t.name.clone(), Ok(true))]);
        assert_eq!(std::fs::read(t.dir.join("files/lib/wine/x86_64-windows/_nvngx.dll")).unwrap(), newer);
        let t = tools(&root).remove(0);
        assert!(!wrapper_changed(&t, &wrapper));
        assert_eq!(refresh_wrappers(&root, &wrapper), [(t.name.clone(), Ok(false))]);
        assert!(std::fs::read_to_string(t.dir.join(MARKER)).unwrap().contains("source_version=GE-Proton11-12\n"), "the rest of the marker stays");
        // What a stopped create left is found for purge.
        std::fs::create_dir_all(root.join("compatibilitytools.d/.GE-Proton11-13-NeuralForge.tmp")).unwrap();
        std::fs::create_dir_all(root.join("compatibilitytools.d/.hidden")).unwrap();
        assert_eq!(leftovers(&root), [root.join("compatibilitytools.d/.GE-Proton11-13-NeuralForge.tmp")]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Rings of dead games go; a running game's ring, and anything else, stays.
    #[test]
    fn only_rings_of_dead_processes_are_swept() {
        let dir = scratch("rings");
        let me = std::process::id();
        for name in [format!("neural-forge-ngx-{me}.ring"), "neural-forge-ngx-999999999.ring".into(), "neural-forge-ngx-x.ring".into(), "other.ring".into()] {
            std::fs::write(dir.join(name), b"NFNGXRG1\x01\0\0\0").unwrap();
        }
        assert!(pid_alive(me) && !pid_alive(0) && !pid_alive(u32::MAX));
        assert_eq!(open_rings(&dir), [(dir.join(format!("neural-forge-ngx-{me}.ring")), me, Some(1))]);
        assert_eq!(sweep_rings_with(&dir, |pid| pid == me), [dir.join("neural-forge-ngx-999999999.ring")]);
        assert!(sweep_dead_rings(&dir).is_empty(), "the live one stays");
        let mut left: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        left.sort();
        assert_eq!(left, [format!("neural-forge-ngx-{me}.ring"), "neural-forge-ngx-x.ring".into(), "other.ring".into()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A file is replaced in one step through `<name>.tmp`: a hard link to the source at its path is replaced, never
    /// written through, and no `.tmp` is left.
    #[test]
    fn write_new_replaces_a_hard_link_atomically() {
        let d = std::env::temp_dir().join(format!("neural-forge-write-new-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let (src, dst) = (d.join("source.dll"), d.join("_nvngx.dll"));
        std::fs::write(&src, "nvidia").unwrap();
        std::fs::hard_link(&src, &dst).unwrap();
        std::fs::write(d.join("_nvngx.dll.tmp"), "stale").unwrap();
        write_new(&dst, b"wrapper", 0o644).unwrap();
        assert_eq!(std::fs::read_to_string(&src).unwrap(), "nvidia", "the source is untouched");
        assert_eq!(std::fs::read_to_string(&dst).unwrap(), "wrapper");
        assert!(!d.join("_nvngx.dll.tmp").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
