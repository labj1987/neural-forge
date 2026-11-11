//! Steam's compatibility tool for a game (`config/config.vdf`'s `CompatToolMapping`), set to a Neural Forge Proton
//! when Neural Forge is switched on for the game and put back when it is switched off or uninstalled, the way
//! ProtonPlus and ProtonUp-Qt choose a game's Proton: an entry `"<appid>" { "name" "<tool>" "config" "" "priority"
//! "250" }`, removed again where the game had none. Edited only while Steam is closed (it writes its own copy back
//! when it exits), under a lock, verified before the write, with `config.vdf.neural-forge.orig` / `.bak` beside it.
//! What a game had before is kept in `config.vdf.neural-forge.compat` (one `appid<TAB>tool` line each, an empty tool
//! for none) until it is put back.

use std::path::{Path, PathBuf};

use crate::proton_tool::{self, Source, Tool};
use crate::vdf;

const MAPPING: [&str; 5] = ["InstallConfigStore", "Software", "Valve", "Steam", "CompatToolMapping"];

fn config_vdf(root: &Path) -> PathBuf {
    root.join("config/config.vdf")
}

fn record_path(root: &Path) -> PathBuf {
    root.join("config/config.vdf.neural-forge.compat")
}

fn entry_path(appid: &str) -> Vec<&str> {
    let mut p = MAPPING.to_vec();
    p.push(appid);
    p
}

/// The tool Steam runs `appid` with (`name` of its entry), if one is chosen for it.
pub fn current(root: &Path, appid: &str) -> Option<String> {
    proton_tool::mapped_tool(root, appid).filter(|n| !n.is_empty())
}

fn read_record(root: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(record_path(root))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(a, t)| (a.to_string(), t.to_string())))
        .collect()
}

fn write_record(root: &Path, entries: &[(String, String)]) -> std::io::Result<()> {
    let path = record_path(root);
    if entries.is_empty() {
        return match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    let text: String = entries.iter().map(|(a, t)| format!("{a}\t{t}\n")).collect();
    crate::paths::write_atomic(&path, text.as_bytes(), 0o644)
}

/// The edit of `config.vdf`, prepared and verified, then written with backups under the lock.
fn edit(root: &Path, change: impl FnOnce(&[u8]) -> Result<Option<Vec<u8>>, String>) -> Result<bool, String> {
    let file = config_vdf(root);
    let _held = crate::steam::lock_file(&file);
    let old = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    let Some(new) = change(&old)? else { return Ok(false) };
    crate::steam::write_with_backups(&file, &old, &new).map_err(|e| format!("writing {}: {e}", file.display()))?;
    Ok(true)
}

fn set_name(old: &[u8], appid: &str, tool: &str) -> Result<Option<Vec<u8>>, String> {
    // A config.vdf without `CompatToolMapping` (no game ever had a Proton chosen) gets an empty one first.
    let tree = vdf::parse(old).map_err(|e| e.0)?;
    let base = if tree.block_at(&MAPPING).is_none() {
        let with = vdf::insert_block(old, &MAPPING).map_err(|e| e.0)?;
        if !vdf::verify_insert_block(old, &with, &MAPPING) {
            return Err("config.vdf did not re-read as before with only an empty CompatToolMapping added".into());
        }
        with
    } else {
        old.to_vec()
    };
    let mut path = entry_path(appid);
    let had = vdf::parse(&base).ok().and_then(|t| t.block_at(&path).cloned()).is_some();
    path.push("name");
    let (mut new, _) = vdf::set_string(&base, &path, tool).map_err(|e| e.0)?;
    if !vdf::verify_edit(&base, &new, &path, tool) {
        return Err("config.vdf did not re-read as before with only the game's Proton changed".into());
    }
    if !had {
        for (key, value) in [("config", ""), ("priority", "250")] {
            let mut p = entry_path(appid);
            p.push(key);
            let (next, _) = vdf::set_string(&new, &p, value).map_err(|e| e.0)?;
            if !vdf::verify_edit(&new, &next, &p, value) {
                return Err("config.vdf did not re-read as expected after adding the game's entry".into());
            }
            new = next;
        }
    }
    Ok((new != old).then_some(new))
}

/// What [`switch_on`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchedOn {
    pub tool: String,
    /// The copy was made now (the notice applies).
    pub made: bool,
    pub previous: Option<String>,
}

/// The installed Proton a Neural Forge copy for `appid` is made from: the one Steam runs the game with now (or
/// the default for all games), found by its Steam name ([`proton_tool::resolve`]), else Proton Experimental where
/// nothing is chosen. Refused, never another Proton, when the chosen one is not installed.
pub fn source_for(root: &Path, libraries: &[PathBuf], appid: &str) -> Result<Source, String> {
    let chosen = current(root, appid).map(|n| (n, "")).or_else(|| current(root, "0").map(|n| (n, " (Steam's default for all games)")));
    match chosen {
        Some((name, what)) => proton_tool::resolve(root, libraries, &name).ok_or_else(|| {
            format!(
                "the game's Proton {name}{what} was not found among the installed Protons: install it, or choose another Proton for the \
                 game in Steam (Properties > Compatibility), then switch it on again"
            )
        }),
        None => proton_tool::resolve(root, libraries, "proton_experimental").ok_or_else(|| {
            "no Proton is chosen for the game and Proton Experimental is not installed: choose a Proton for it in Steam (Properties > \
             Compatibility), then switch it on again"
                .to_string()
        }),
    }
}

/// Sets `appid`'s Proton to a Neural Forge copy: the game's own Proton's copy (made now when there is none, made
/// again when its source changed, its wrapper renewed when Neural Forge's changed), or the one already chosen (kept
/// as it is, its wrapper renewed, when its source is gone: it holds its own files).
/// `Ok(None)` for a native Linux game with no Proton chosen: it runs without Proton, where the wrapper does not
/// apply. Refused while Steam runs, and for a game whose installed build cannot be told. The game's earlier choice
/// is recorded (before `config.vdf` is edited) for [`switch_off`].
pub fn switch_on(root: &Path, libraries: &[PathBuf], appid: &str, wrapper: &Path, steam_running: bool) -> Result<Option<SwitchedOn>, String> {
    if steam_running {
        return Err("Steam is running: close it to let Neural Forge choose the game's Proton".into());
    }
    let tools = proton_tool::tools(root);
    let now = current(root, appid);
    if now.is_none() {
        // Nothing chosen for the game: Steam runs a native Linux build without Proton.
        let game = crate::steam::games(&[crate::steam::Root { path: root.to_path_buf(), flatpak: false }]).into_iter().find(|g| g.appid == appid);
        match game.as_ref().map(crate::steam::installed_build) {
            Some(crate::steam::Build::Windows) => {}
            Some(crate::steam::Build::Linux) => return Ok(None),
            _ => {
                return Err("Neural Forge could not tell whether the game's installed build is for Windows: choose a Proton for it in Steam \
                            (Properties > Compatibility), then switch it on again"
                    .into())
            }
        }
    }
    let dll = || std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()));
    let from_tool: Option<&Tool> = now.as_ref().and_then(|n| tools.iter().find(|t| &t.name == n));
    if let Some(t) = from_tool.filter(|t| !proton_tool::source_changed(t)) {
        if proton_tool::wrapper_changed(t, wrapper) {
            proton_tool::refresh_wrapper(t, &dll()?)?;
        }
        return Ok(Some(SwitchedOn { tool: t.name.clone(), made: false, previous: now.clone() }));
    }
    // The source: what the copy the game is set to was made from, else the game's own Proton.
    let source = match from_tool {
        Some(t) => proton_tool::sources(root, libraries)
            .into_iter()
            .find(|s| s.dir == t.source)
            .ok_or_else(|| format!("{} was made from {}, which is gone: choose another Proton for the game in Steam first", t.name, t.source.display()))?,
        None => source_for(root, libraries, appid)?,
    };
    let name = source.tool_name();
    let made = match tools.iter().find(|t| t.name == name) {
        Some(t) => {
            if proton_tool::wrapper_changed(t, wrapper) {
                proton_tool::refresh_wrapper(t, &dll()?)?;
            }
            false
        }
        None => {
            proton_tool::create(root, &source, wrapper)?;
            true
        }
    };
    // The record first: an edit it does not record could never be put back.
    let before = read_record(root);
    let mut record = before.clone();
    if from_tool.is_none() && !record.iter().any(|(a, _)| a == appid) {
        record.push((appid.to_string(), now.clone().unwrap_or_default()));
    }
    if record != before {
        write_record(root, &record).map_err(|e| e.to_string())?;
    }
    if let Err(e) = edit(root, |old| set_name(old, appid, &name)) {
        if record != before {
            let _ = write_record(root, &before);
        }
        return Err(e);
    }
    Ok(Some(SwitchedOn { tool: name, made, previous: now }))
}

/// Puts back the Proton `appid` had before [`switch_on`] (removing the entry where it had none). Nothing to do
/// for a game Neural Forge did not set. Refused while Steam runs.
pub fn switch_off(root: &Path, appid: &str, steam_running: bool) -> Result<Option<String>, String> {
    let mut record = read_record(root);
    let Some(at) = record.iter().position(|(a, _)| a == appid) else { return Ok(None) };
    if steam_running {
        return Err("Steam is running: close it to put the game's Proton back".into());
    }
    let previous = record[at].1.clone();
    let path = entry_path(appid);
    // Only while the game is still on a Neural Forge copy: a Proton the user chose since stays.
    let ours = current(root, appid).is_some_and(|n| proton_tool::tools(root).iter().any(|t| t.name == n) || n.ends_with(proton_tool::SUFFIX));
    if ours {
        if previous.is_empty() {
            edit(root, |old| remove_entry(old, &path))?;
        } else {
            edit(root, |old| set_name(old, appid, &previous))?;
        }
    }
    record.remove(at);
    write_record(root, &record).map_err(|e| e.to_string())?;
    Ok(Some(previous))
}

fn remove_entry(old: &[u8], path: &[&str]) -> Result<Option<Vec<u8>>, String> {
    match vdf::remove_block(old, path).map_err(|e| e.0)? {
        Some(new) if vdf::verify_removal(old, &new, path) => Ok(Some(new)),
        Some(_) => Err("config.vdf did not re-read as before without the game's entry".into()),
        None => Ok(None),
    }
}

/// [`switch_off`] for every game Neural Forge set (uninstall --purge).
pub fn switch_all_off(root: &Path, steam_running: bool) -> Result<Vec<String>, String> {
    let apps: Vec<String> = read_record(root).into_iter().map(|(a, _)| a).collect();
    for a in &apps {
        switch_off(root, a, steam_running)?;
    }
    Ok(apps)
}

/// The games `CompatToolMapping` sets to the Proton named `tool`.
fn games_on(root: &Path, tool: impl Fn(&str) -> bool) -> Vec<String> {
    let Some(tree) = std::fs::read(config_vdf(root)).ok().and_then(|t| vdf::parse(&t).ok()) else { return Vec::new() };
    let Some(mapping) = tree.block_at(&MAPPING) else { return Vec::new() };
    mapping.0.iter().filter(|e| matches!(&e.value, vdf::Value::Block(b) if b.string_at(&["name"]).is_some_and(&tool))).map(|e| e.key.clone()).collect()
}

/// Takes every game still set to one of Neural Forge's copies (one the user chose by hand, or whose record is
/// lost) off it before the copies go (uninstall --purge): its entry removed, so Steam's default applies. The
/// games. Refused while Steam runs and a game is on a copy.
///
/// A copy is one of [`proton_tool::tools`] (marked as Neural Forge's), or a `-NeuralForge` name whose folder is no
/// longer there (removed already); never another tool that exists under such a name.
pub fn forget_copies(root: &Path, steam_running: bool) -> Result<Vec<String>, String> {
    let names: Vec<String> = proton_tool::tools(root).into_iter().map(|t| t.name).collect();
    let dir = proton_tool::tools_dir(root);
    let apps = games_on(root, |n| names.iter().any(|m| m == n) || (n.ends_with(proton_tool::SUFFIX) && !dir.join(n).exists()));
    if apps.is_empty() {
        return Ok(apps);
    }
    if steam_running {
        return Err(format!("Steam is running: close it to take games {} off Neural Forge's Proton", apps.join(", ")));
    }
    for a in &apps {
        edit(root, |old| remove_entry(old, &entry_path(a)))?;
    }
    Ok(apps)
}

/// What [`update`] did with one copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Updated {
    /// The copy updated.
    pub from: String,
    /// The copy its games run with now: `from` when it was made again under its name (or only got the new wrapper).
    pub to: String,
    /// The games moved from `from` to `to`.
    pub moved: Vec<String>,
    /// Why `from` is still there when it was to go (Steam is running, a game runs with it).
    pub kept: Option<String>,
}

/// Updates `tool` (`proton-tool update`, Make again): the new wrapper put in when only that changed, or when its
/// source is gone (the copy keeps working as it is); else the copy
/// made again from its source's current version, and when that is a new name, every game set to the old copy
/// moved to the new one and the old copy removed (both only while Steam is closed, and never a copy a game runs
/// with).
pub fn update(root: &Path, libraries: &[PathBuf], tool: &Tool, wrapper: &Path, steam_running: bool) -> Result<Updated, String> {
    update_in(root, libraries, tool, wrapper, steam_running, Path::new("/proc"))
}

fn update_in(root: &Path, libraries: &[PathBuf], tool: &Tool, wrapper: &Path, steam_running: bool, proc_root: &Path) -> Result<Updated, String> {
    let same = |moved| Updated { from: tool.name.clone(), to: tool.name.clone(), moved, kept: None };
    if !proton_tool::source_changed(tool) {
        let dll = std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()))?;
        proton_tool::refresh_wrapper(tool, &dll)?;
        return Ok(same(Vec::new()));
    }
    let source = proton_tool::sources(root, libraries)
        .into_iter()
        .find(|s| s.dir == tool.source)
        .ok_or_else(|| format!("its source {} is gone: remove it, and choose another Proton for its games", tool.source.display()))?;
    let made = proton_tool::create_in(root, &source, wrapper, proc_root)?;
    if made.tool.name == tool.name {
        return Ok(same(Vec::new()));
    }
    let mut out = Updated { from: tool.name.clone(), to: made.tool.name.clone(), moved: Vec::new(), kept: None };
    if steam_running {
        out.kept = Some("Steam is running: close it and update again to move its games to the new copy and remove the old one".into());
        return Ok(out);
    }
    for appid in games_on(root, |n| n == tool.name) {
        edit(root, |old| set_name(old, &appid, &made.tool.name))?;
        out.moved.push(appid);
    }
    if let Err(e) = proton_tool::remove_in(root, &tool.name, proc_root) {
        out.kept = Some(e);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = "\"InstallConfigStore\"\n{\n\t\"Software\"\n\t{\n\t\t\"Valve\"\n\t\t{\n\t\t\t\"Steam\"\n\t\t\t{\n\t\t\t\t\"CompatToolMapping\"\n\t\t\t\t{\n\t\t\t\t\t\"0\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"proton_experimental\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t\t\"priority\"\t\t\"75\"\n\t\t\t\t\t}\n\t\t\t\t\t\"1091500\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"Proton-GE Latest\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t\t\"priority\"\t\t\"250\"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n";

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("neural-forge-compat-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("config")).unwrap();
        std::fs::write(d.join("config/config.vdf"), CONFIG).unwrap();
        d
    }

    fn fake_proton(dir: &Path, version: &str) {
        std::fs::create_dir_all(dir.join("files/lib/wine/x86_64-windows")).unwrap();
        std::fs::create_dir_all(dir.join("files/lib/wine/x86_64-unix")).unwrap();
        std::fs::write(dir.join("files/lib/wine/x86_64-unix/ntdll.so"), "\0wine-11.0\0").unwrap();
        std::fs::write(dir.join("proton"), "        if self.nvidia_wine_dll_dir:\n            self.env[\"NVIDIA_WINE_DLL_DIR\"] = self.nvidia_wine_dll_dir\n")
            .unwrap();
        std::fs::write(dir.join("version"), format!("1 {version}\n")).unwrap();
    }

    /// An installed game in `library` (`steamapps/appmanifest_<appid>.acf`) with `file` in its folder.
    fn fake_app(library: &Path, appid: &str, dir: &str, file: Option<(&str, &str)>) {
        std::fs::create_dir_all(library.join("steamapps/common").join(dir)).unwrap();
        std::fs::write(
            library.join(format!("steamapps/appmanifest_{appid}.acf")),
            format!("\"AppState\"\n{{\n\t\"appid\"\t\t\"{appid}\"\n\t\"name\"\t\t\"{dir}\"\n\t\"installdir\"\t\t\"{dir}\"\n}}\n"),
        )
        .unwrap();
        if let Some((name, content)) = file {
            let path = library.join("steamapps/common").join(dir).join(name);
            std::fs::write(&path, content).unwrap();
            std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        }
    }

    /// The test's Steam: Proton-GE Latest in compatibilitytools.d, Proton Experimental (by its manifest) in `lib`.
    fn steam(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = scratch(tag);
        let lib = root.join("lib");
        fake_proton(&root.join("compatibilitytools.d/Proton-GE Latest"), "GE-Proton11-12");
        fake_proton(&lib.join("steamapps/common/Proton - Experimental"), "experimental-11.0");
        fake_app(&lib, "1493710", "Proton - Experimental", None);
        let w = wrapper(&root);
        (root, lib, w)
    }

    fn wrapper(root: &Path) -> PathBuf {
        let mut b = vec![0u8; 0x100];
        b[..2].copy_from_slice(b"MZ");
        b[0x40..0x50].copy_from_slice(b"Wine builtin DLL");
        let p = root.join("_nvngx.dll");
        std::fs::write(&p, b).unwrap();
        p
    }

    /// Switched on: the game's own Proton's copy is made and chosen; switched off: the earlier Proton is back, and
    /// a game that had none loses the entry again.
    #[test]
    fn switching_on_chooses_the_copy_of_the_game_s_proton_and_off_puts_it_back() {
        let (root, lib, w) = steam("onoff");
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        assert!(switch_on(&root, std::slice::from_ref(&lib), "1091500", &w, true).is_err(), "never while Steam runs");
        let on = switch_on(&root, std::slice::from_ref(&lib), "1091500", &w, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made, on.previous.as_deref()), ("GE-Proton11-12-NeuralForge", true, Some("Proton-GE Latest")));
        assert_eq!(current(&root, "1091500").as_deref(), Some("GE-Proton11-12-NeuralForge"));
        // A game with no entry gets the default Proton's copy and a whole entry.
        let other = switch_on(&root, std::slice::from_ref(&lib), "1174180", &w, false).unwrap().unwrap();
        assert_eq!(other.tool, "experimental-11.0-NeuralForge");
        let text = std::fs::read_to_string(root.join("config/config.vdf")).unwrap();
        assert!(text.contains("\"1174180\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"experimental-11.0-NeuralForge\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t\t\"priority\"\t\t\"250\""), "{text}");
        assert_eq!(switch_off(&root, "1091500", false).unwrap().as_deref(), Some("Proton-GE Latest"));
        assert_eq!(current(&root, "1091500").as_deref(), Some("Proton-GE Latest"));
        assert_eq!(switch_off(&root, "1174180", false).unwrap().as_deref(), Some(""));
        assert_eq!(std::fs::read_to_string(root.join("config/config.vdf")).unwrap(), CONFIG, "back byte for byte");
        assert!(!record_path(&root).exists());
        assert_eq!(switch_off(&root, "1174180", false).unwrap(), None, "nothing left to put back");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A native Linux game with no Proton chosen keeps running without one; a game whose build cannot be told, or
    /// whose chosen Proton is not installed, is refused, never set to another Proton.
    #[test]
    fn native_games_are_left_and_an_unknown_proton_is_refused() {
        let (root, lib, w) = steam("native");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "220", "Native", Some(("game.x86_64", "\x7fELF....")));
        assert_eq!(switch_on(&root, libs, "220", &w, false).unwrap(), None);
        assert_eq!(current(&root, "220"), None);
        fake_app(&root, "221", "Empty", None);
        assert!(switch_on(&root, libs, "221", &w, false).unwrap_err().contains("could not tell"));
        assert!(switch_on(&root, libs, "222", &w, false).unwrap_err().contains("could not tell"), "not installed");
        // A game set to a Proton that is not installed: refused, nothing written.
        let before = std::fs::read(root.join("config/config.vdf")).unwrap();
        std::fs::remove_dir_all(root.join("compatibilitytools.d/Proton-GE Latest")).unwrap();
        let e = switch_on(&root, libs, "1091500", &w, false).unwrap_err();
        assert!(e.contains("Proton-GE Latest was not found"), "{e}");
        assert_eq!(std::fs::read(root.join("config/config.vdf")).unwrap(), before);
        assert!(!record_path(&root).exists() && proton_tool::tools(&root).is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A config.vdf that never had a Proton chosen gets its CompatToolMapping block; the record is written before
    /// the edit and taken back when the edit fails.
    #[test]
    fn a_missing_mapping_block_is_created_and_a_failed_edit_leaves_no_record() {
        let (root, lib, w) = steam("nomapping");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        let bare = "\"InstallConfigStore\"\n{\n\t\"Software\"\n\t{\n\t\t\"Valve\"\n\t\t{\n\t\t\t\"Steam\"\n\t\t\t{\n\t\t\t\t\"SurveyDate\"\t\t\"2026-11-01\"\n\t\t\t}\n\t\t}\n\t}\n}\n";
        std::fs::write(root.join("config/config.vdf"), bare).unwrap();
        // No default either: Proton Experimental's copy.
        let on = switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        assert_eq!(on.tool, "experimental-11.0-NeuralForge");
        let text = std::fs::read_to_string(root.join("config/config.vdf")).unwrap();
        assert!(text.contains("\t\t\t\t\"SurveyDate\"\t\t\"2026-11-01\"\n\t\t\t\t\"CompatToolMapping\"\n\t\t\t\t{\n\t\t\t\t\t\"1174180\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"experimental-11.0-NeuralForge\""), "{text}");
        assert_eq!(read_record(&root), [("1174180".to_string(), String::new())]);
        assert_eq!(switch_off(&root, "1174180", false).unwrap().as_deref(), Some(""));
        // A config.vdf that cannot be edited: the record goes back to what it was.
        std::fs::write(root.join("config/config.vdf"), "\"InstallConfigStore\"\n{\n}\n").unwrap();
        assert!(switch_on(&root, libs, "1174180", &w, false).is_err());
        assert!(!record_path(&root).exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Purge takes games off the copies even without a record of them.
    #[test]
    fn games_on_a_copy_without_a_record_are_taken_off_it() {
        let (root, lib, w) = steam("forget");
        let on = switch_on(&root, std::slice::from_ref(&lib), "1091500", &w, false).unwrap().unwrap();
        std::fs::remove_file(record_path(&root)).unwrap();
        assert_eq!(switch_all_off(&root, false).unwrap(), Vec::<String>::new());
        assert_eq!(current(&root, "1091500").as_deref(), Some(on.tool.as_str()));
        assert!(forget_copies(&root, true).unwrap_err().contains("1091500"), "never while Steam runs");
        assert_eq!(forget_copies(&root, false).unwrap(), ["1091500"]);
        assert_eq!(current(&root, "1091500"), None);
        assert_eq!(current(&root, "0").as_deref(), Some("proton_experimental"), "the rest stays");
        assert!(forget_copies(&root, false).unwrap().is_empty());
        // A foreign tool with the suffix that exists stays; a suffix name whose folder is gone is ours, removed.
        std::fs::create_dir_all(root.join("compatibilitytools.d/Other-NeuralForge")).unwrap();
        edit(&root, |old| set_name(old, "10", "Other-NeuralForge")).unwrap();
        edit(&root, |old| set_name(old, "20", "Gone-NeuralForge")).unwrap();
        assert_eq!(forget_copies(&root, false).unwrap(), ["20"]);
        assert_eq!(current(&root, "10").as_deref(), Some("Other-NeuralForge"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A copy whose source folder was deleted (ProtonUp replaced GE-Proton11-12 with 11-13) keeps working: switching
    /// a game on and updating keep it and renew its wrapper, never refuse.
    #[test]
    fn a_copy_whose_source_is_gone_is_kept() {
        let (root, lib, w) = steam("gone");
        let libs = std::slice::from_ref(&lib);
        switch_on(&root, libs, "1091500", &w, false).unwrap().unwrap();
        let t = proton_tool::tools(&root).remove(0);
        std::fs::remove_dir_all(root.join("compatibilitytools.d/Proton-GE Latest")).unwrap();
        assert!(proton_tool::source_gone(&t) && !proton_tool::source_changed(&t));
        std::fs::write(&w, [std::fs::read(&w).unwrap(), vec![1]].concat()).unwrap();
        let on = switch_on(&root, libs, "1091500", &w, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made), ("GE-Proton11-12-NeuralForge", false));
        assert!(!proton_tool::wrapper_changed(&proton_tool::tools(&root)[0], &w), "the wrapper renewed");
        std::fs::write(&w, [std::fs::read(&w).unwrap(), vec![2]].concat()).unwrap();
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let t = proton_tool::tools(&root).remove(0);
        let up = update_in(&root, libs, &t, &w, false, &procs).unwrap();
        assert_eq!((up.from.as_str(), up.to.as_str(), up.kept), (t.name.as_str(), t.name.as_str(), None));
        assert!(t.dir.is_dir() && !proton_tool::wrapper_changed(&proton_tool::tools(&root)[0], &w));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// An update with Steam closed moves the old copy's games to the new copy and removes the old one; with Steam
    /// running, the old copy stays for the games set to it.
    #[test]
    fn update_moves_games_to_the_new_copy_and_removes_the_old() {
        let (root, lib, w) = steam("update");
        let libs = std::slice::from_ref(&lib);
        switch_on(&root, libs, "1091500", &w, false).unwrap().unwrap();
        let old = proton_tool::tools(&root).remove(0);
        fake_proton(&root.join("compatibilitytools.d/Proton-GE Latest"), "GE-Proton11-13");
        assert!(proton_tool::source_changed(&old));
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let running = update_in(&root, libs, &old, &w, true, &procs).unwrap();
        assert_eq!((running.to.as_str(), running.moved.len()), ("GE-Proton11-13-NeuralForge", 0));
        assert!(running.kept.unwrap().contains("Steam is running") && old.dir.is_dir());
        let done = update_in(&root, libs, &old, &w, false, &procs).unwrap();
        assert_eq!((done.moved, done.kept), (vec!["1091500".to_string()], None));
        assert_eq!(current(&root, "1091500").as_deref(), Some("GE-Proton11-13-NeuralForge"));
        assert!(!old.dir.exists());
        // Switched off, the game gets the Proton it had before Neural Forge.
        assert_eq!(switch_off(&root, "1091500", false).unwrap().as_deref(), Some("Proton-GE Latest"));
        // Only the wrapper changed: put in, same copy.
        let t = proton_tool::tools(&root).remove(0);
        std::fs::write(&w, [std::fs::read(&w).unwrap(), vec![1]].concat()).unwrap();
        let fresh = update_in(&root, libs, &t, &w, false, &procs).unwrap();
        assert_eq!(fresh.to, t.name);
        assert!(!proton_tool::wrapper_changed(&proton_tool::tools(&root)[0], &w));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
