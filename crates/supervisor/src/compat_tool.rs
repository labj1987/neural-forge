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
    let mut path = entry_path(appid);
    let had = vdf::parse(old).ok().and_then(|t| t.block_at(&path).cloned()).is_some();
    path.push("name");
    let (mut new, _) = vdf::set_string(old, &path, tool).map_err(|e| e.0)?;
    if !vdf::verify_edit(old, &new, &path, tool) {
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
/// the default for all games), else Proton Experimental, else the newest source found.
pub fn source_for(root: &Path, libraries: &[PathBuf], appid: &str) -> Option<Source> {
    let sources = proton_tool::sources(root, libraries);
    let by_name = |name: &str| -> Option<Source> {
        let valve = match name {
            "proton_experimental" => Some("Proton - Experimental".to_string()),
            "proton_hotfix" => Some("Proton Hotfix".to_string()),
            n if n.starts_with("proton_") => Some(format!("Proton {}.0", &n["proton_".len()..])),
            _ => None,
        };
        sources.iter().find(|s| s.dir.file_name().is_some_and(|d| d.to_string_lossy() == name || Some(d.to_string_lossy().into_owned()) == valve)).cloned()
    };
    current(root, appid)
        .or_else(|| current(root, "0"))
        .and_then(|n| by_name(&n))
        .or_else(|| by_name("proton_experimental"))
        .or_else(|| sources.first().cloned())
}

/// Sets `appid`'s Proton to a Neural Forge copy: the game's own Proton's copy (made now when there is none, or
/// when it is outdated), or the one already chosen. Refused while Steam runs. The game's earlier choice is kept
/// for [`switch_off`].
pub fn switch_on(root: &Path, libraries: &[PathBuf], appid: &str, wrapper: &Path, steam_running: bool) -> Result<SwitchedOn, String> {
    if steam_running {
        return Err("Steam is running: close it to let Neural Forge choose the game's Proton".into());
    }
    let tools = proton_tool::tools(root);
    let now = current(root, appid);
    if let Some(t) = now.as_ref().and_then(|n| tools.iter().find(|t| &t.name == n)) {
        if !proton_tool::outdated(t) {
            return Ok(SwitchedOn { tool: t.name.clone(), made: false, previous: now.clone() });
        }
    }
    // The source: what the copy the game is set to was made from, else the game's own Proton.
    let from_tool: Option<&Tool> = now.as_ref().and_then(|n| tools.iter().find(|t| &t.name == n));
    let source = match from_tool {
        Some(t) => proton_tool::sources(root, libraries).into_iter().find(|s| s.dir == t.source),
        None => source_for(root, libraries, appid),
    }
    .ok_or("no installed Proton to make a Neural Forge copy from (install GE-Proton or Proton Experimental)")?;
    let name = source.tool_name();
    let made = if tools.iter().any(|t| t.name == name) {
        false
    } else {
        proton_tool::create(root, &source, wrapper)?;
        true
    };
    let mut record = read_record(root);
    if from_tool.is_none() && !record.iter().any(|(a, _)| a == appid) {
        record.push((appid.to_string(), now.clone().unwrap_or_default()));
    }
    edit(root, |old| set_name(old, appid, &name))?;
    write_record(root, &record).map_err(|e| e.to_string())?;
    Ok(SwitchedOn { tool: name, made, previous: now })
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
            edit(root, |old| match vdf::remove_block(old, &path).map_err(|e| e.0)? {
                Some(new) if vdf::verify_removal(old, &new, &path) => Ok(Some(new)),
                Some(_) => Err("config.vdf did not re-read as before without the game's entry".into()),
                None => Ok(None),
            })?;
        } else {
            edit(root, |old| set_name(old, appid, &previous))?;
        }
    }
    record.remove(at);
    write_record(root, &record).map_err(|e| e.to_string())?;
    Ok(Some(previous))
}

/// [`switch_off`] for every game Neural Forge set (uninstall --purge).
pub fn switch_all_off(root: &Path, steam_running: bool) -> Result<Vec<String>, String> {
    let apps: Vec<String> = read_record(root).into_iter().map(|(a, _)| a).collect();
    for a in &apps {
        switch_off(root, a, steam_running)?;
    }
    Ok(apps)
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
        std::fs::write(dir.join("proton"), "        if self.nvidia_wine_dll_dir:\n            self.env[\"NVIDIA_WINE_DLL_DIR\"] = self.nvidia_wine_dll_dir\n")
            .unwrap();
        std::fs::write(dir.join("version"), format!("1 {version}\n")).unwrap();
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
        let root = scratch("onoff");
        let lib = root.join("lib");
        fake_proton(&root.join("compatibilitytools.d/Proton-GE Latest"), "GE-Proton11-12");
        fake_proton(&lib.join("steamapps/common/Proton - Experimental"), "experimental-11.0");
        let w = wrapper(&root);
        assert!(switch_on(&root, std::slice::from_ref(&lib), "1091500", &w, true).is_err(), "never while Steam runs");
        let on = switch_on(&root, std::slice::from_ref(&lib), "1091500", &w, false).unwrap();
        assert_eq!((on.tool.as_str(), on.made, on.previous.as_deref()), ("GE-Proton11-12-NeuralForge", true, Some("Proton-GE Latest")));
        assert_eq!(current(&root, "1091500").as_deref(), Some("GE-Proton11-12-NeuralForge"));
        // A game with no entry gets the default Proton's copy and a whole entry.
        let other = switch_on(&root, std::slice::from_ref(&lib), "1174180", &w, false).unwrap();
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
}
