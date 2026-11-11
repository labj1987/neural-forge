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
//! Properties > Compatibility.

use std::path::{Path, PathBuf};

/// The copy's name: the source's version name and this.
pub const SUFFIX: &str = "-NeuralForge";
/// In a copy: what it was made from (`source=`, `source_version=`, `wrapper_sha256=`).
pub const MARKER: &str = "neural-forge-proton.txt";
/// The wrapper in Neural Forge's install (under the data dir).
pub const WRAPPER: &str = "lib/neural-forge/proton/_nvngx.dll";
/// Where Wine's 64-bit builtin DLLs live in a Proton tree.
const BUILTIN_DIR: &str = "files/lib/wine/x86_64-windows";
/// Turns the wrapper off for one game (its launch options): `PROTON_NO_NEURAL_FORGE_NGX=1 %command%`.
pub const OFF_ENV: &str = "PROTON_NO_NEURAL_FORGE_NGX";

/// What the user is told before a copy is made (Alex 2026-11-10: a clear notice, no blocking).
pub const NOTICE: &str = "Neural Forge makes a copy of this Proton next to it, named with \u{201c}-NeuralForge\u{201d}. On the same drive the copy \
takes no extra space (hard links). It differs in one thing: games run with it load Neural Forge's small NGX wrapper in place of NVIDIA's \
_nvngx.dll, which passes every DLSS call on unchanged and lets Neural Forge read what the game gives DLSS (its images, camera jitter, \
motion vectors and exposure). Your Proton is not changed. Restart Steam, then choose the copy for a game in its Properties > Compatibility. \
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
}

fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    s.trim_matches('-').to_string()
}

/// `dir/version`'s second word.
fn version_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("version")).ok()?;
    text.split_whitespace().nth(1).map(str::to_string)
}

fn display_of(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("compatibilitytool.vdf")).ok()?;
    let at = text.find("\"display_name\"")? + "\"display_name\"".len();
    let rest = &text[at..];
    let start = rest.find('"')? + 1;
    let end = start + rest[start..].find('"')?;
    Some(rest[start..end].to_string())
}

fn source_at(dir: &Path) -> Option<Source> {
    if !dir.join("proton").is_file() || dir.join(MARKER).exists() {
        return None;
    }
    let base = dir.file_name()?.to_string_lossy().into_owned();
    Some(Source { dir: dir.to_path_buf(), version: version_of(dir).unwrap_or_else(|| base.clone()), display: display_of(dir).unwrap_or(base) })
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

/// The copies in `root`'s `compatibilitytools.d`.
pub fn tools(root: &Path) -> Vec<Tool> {
    let Ok(entries) = std::fs::read_dir(tools_dir(root)) else { return Vec::new() };
    let mut out: Vec<Tool> = entries.flatten().filter_map(|e| tool_at(&e.path())).collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn tool_at(dir: &Path) -> Option<Tool> {
    let text = std::fs::read_to_string(dir.join(MARKER)).ok()?;
    let field = |k: &str| text.lines().find_map(|l| l.strip_prefix(k).map(str::to_string)).unwrap_or_default();
    Some(Tool {
        name: dir.file_name()?.to_string_lossy().into_owned(),
        dir: dir.to_path_buf(),
        source: field("source=").into(),
        source_version: field("source_version="),
    })
}

/// Whether `tool`'s source changed since it was made (a new GE-Proton in a "Latest" folder, a Proton
/// Experimental update): the copy should be made again.
pub fn outdated(tool: &Tool) -> bool {
    source_at(&tool.source).is_some_and(|s| s.version != tool.source_version) || !tool.source.join("proton").is_file()
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

/// Writes `content` as a new file at `path` (removing what is there first: it may be a hard link to the source).
fn write_new(path: &Path, content: &[u8], mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    std::fs::write(path, content)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
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
/// `_nvngx.dll`. Built under a hidden name and renamed into place, so Steam never sees half a copy.
pub fn create(root: &Path, source: &Source, wrapper: &Path) -> Result<Created, String> {
    let dll = std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()))?;
    if dll.get(0x40..0x50) != Some(b"Wine builtin DLL".as_slice()) {
        return Err(format!("{} is not a Wine builtin DLL", wrapper.display()));
    }
    let script = std::fs::read_to_string(source.dir.join("proton")).map_err(|e| format!("{}: {e}", source.dir.join("proton").display()))?;
    let patched = patch_proton_script(&script)?;
    let name = source.tool_name();
    let dir = tools_dir(root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (final_dir, tmp) = (dir.join(&name), dir.join(format!(".{name}.tmp")));
    if final_dir.exists() && tool_at(&final_dir).is_none() {
        return Err(format!("{} exists and is not Neural Forge's", final_dir.display()));
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let fail = |e: std::io::Error| {
        let _ = std::fs::remove_dir_all(&tmp);
        format!("copying {} to {}: {e}", source.dir.display(), tmp.display())
    };
    let copied = link_tree(&source.dir, &tmp).map_err(fail)?;
    let digest = crate::install::digest(&dll);
    let marker = format!(
        "# Made by Neural Forge (neural-forge-cli proton-tool): {} with Neural Forge's NGX wrapper.\nsource={}\nsource_version={}\nwrapper_sha256={digest}\n",
        source.display,
        source.dir.display(),
        source.version
    );
    let write = || -> std::io::Result<()> {
        write_new(&tmp.join("proton"), patched.as_bytes(), 0o755)?;
        write_new(&tmp.join("compatibilitytool.vdf"), manifest(&name).as_bytes(), 0o644)?;
        std::fs::create_dir_all(tmp.join(BUILTIN_DIR))?;
        write_new(&tmp.join(BUILTIN_DIR).join("_nvngx.dll"), &dll, 0o644)?;
        write_new(&tmp.join(MARKER), marker.as_bytes(), 0o644)
    };
    write().map_err(fail)?;
    let replaced = final_dir.exists();
    if replaced {
        std::fs::remove_dir_all(&final_dir).map_err(|e| format!("{}: {e}", final_dir.display()))?;
    }
    std::fs::rename(&tmp, &final_dir).map_err(fail)?;
    let tool = tool_at(&final_dir).ok_or("the copy has no marker")?;
    Ok(Created { tool, copied, replaced })
}

/// Removes the copy named `name` (only one of Neural Forge's). Games set to it need another Proton chosen in
/// Steam; their prefixes are set up for that Proton on their next launch.
pub fn remove(root: &Path, name: &str) -> Result<Tool, String> {
    let tool = tools(root).into_iter().find(|t| t.name == name).ok_or_else(|| format!("no Neural Forge Proton named {name}"))?;
    std::fs::remove_dir_all(&tool.dir).map_err(|e| format!("{}: {e}", tool.dir.display()))?;
    Ok(tool)
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
        assert!(!outdated(t));
        std::fs::write(src.join("version"), "1793999999 GE-Proton11-13\n").unwrap();
        assert!(outdated(t));
        // Made again under the new name; the old copy removed by name.
        let again = create(&root, &sources(&root, &[])[0], &wrapper).unwrap();
        assert_eq!(again.tool.name, "GE-Proton11-13-NeuralForge");
        assert_eq!(remove(&root, "GE-Proton11-12-NeuralForge").unwrap().name, "GE-Proton11-12-NeuralForge");
        assert!(remove(&root, "Proton-GE Latest").is_err(), "never a tool that is not ours");
        assert!(src.join("proton").exists());
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
}
