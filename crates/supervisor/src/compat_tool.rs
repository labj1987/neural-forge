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
    /// Something to tell the user that is not an error: a rename that is still pending, games moved from a copy whose
    /// folder had been renamed already.
    pub note: Option<String>,
}

/// `e` with what was repaired before it failed, so the user still sees that games were moved.
fn carry_note(repaired: &Option<String>, e: String) -> String {
    match repaired {
        Some(r) => format!("{e} (before that: {r})"),
        None => e,
    }
}

/// Whether a failed rename of `t` was a refusal that waits for the user (a game runs with the copy, or another tool
/// or an incomplete copy has the new name), not a failure of its own.
fn rename_refused(t: &Tool, proc_root: &Path) -> bool {
    matches!(proton_tool::rename_plan(t), proton_tool::RenamePlan::Clash(_)) || !proton_tool::in_use(&t.dir, proc_root).is_empty()
}

/// What a failed rename of `t` (error `e`) means: the note that it is pending (the folder is still there and the rename
/// was refused: the game stays on `t`), else the real error; when the folder is gone (the rename and its undo both
/// failed) the error says the games are stranded and how they are repaired.
fn rename_failure(t: &Tool, e: String, refused: bool) -> Result<String, String> {
    let to = proton_tool::remade_name(t);
    if !t.dir.exists() {
        return Err(format!(
            "{e}; {} is gone and its games are stranded on that name: the next Make again, or `neural-forge-cli proton-tool update`, moves them to {to}",
            t.name
        ));
    }
    if !refused {
        return Err(e);
    }
    Ok(format!("{e}; the rename to {to} is still pending: the game stays on {} until Make again has run", t.name))
}

/// Two optional notes as one.
fn join_notes(a: Option<String>, b: Option<String>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a}; {b}")),
        (a, b) => a.or(b),
    }
}

/// How a `CompatToolMapping` entry reads to the user: "0" is Steam's default for every game.
pub fn app_label(appid: &str) -> String {
    if appid == "0" {
        "the default for every game".to_string()
    } else {
        format!("app {appid}")
    }
}

/// A game Steam maps to a Neural Forge copy that is not there under that name, while a copy of the same version is there
/// under the name [`proton_tool::remade_name`] gives (the folder was renamed, then the process stopped before the games
/// moved: [`rename`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orphaned {
    /// The name the games are set to (no such folder).
    pub name: String,
    /// The copy of the same version that is there.
    pub to: String,
    /// The app ids set to `name` ("0" is the default for all games).
    pub games: Vec<String>,
}

/// The games set to a missing `X-NeuralForge` tool for which the folder [`neural_forge_protocol::proton_copy::copy_name`]
/// gives (`Proton-X-NeuralForge`) exists as a Neural Forge copy.
pub fn orphaned_mappings(root: &Path) -> Vec<Orphaned> {
    let dir = proton_tool::tools_dir(root);
    let mut out = Vec::new();
    for t in proton_tool::tools(root) {
        let Some(old) = t.name.strip_prefix("Proton-") else { continue };
        if !old.ends_with(proton_tool::SUFFIX) || neural_forge_protocol::proton_copy::cloud_name(old) || dir.join(old).exists() || dir.join(old).is_symlink() {
            continue;
        }
        let games = games_on(root, |n| n == old);
        if !games.is_empty() {
            out.push(Orphaned { name: old.to_string(), to: t.name.clone(), games });
        }
    }
    out
}

/// Moves the games of every [`orphaned_mappings`] entry to the copy that is there. Refused while Steam runs.
pub fn repair_orphans(root: &Path, steam_running: bool) -> Result<Vec<Orphaned>, String> {
    let found = orphaned_mappings(root);
    if found.is_empty() {
        return Ok(found);
    }
    if steam_running {
        return Err("Steam is running: close it to move the games set to a Proton that is not installed to its renamed copy".into());
    }
    for o in &found {
        move_games(root, &o.name, &o.to)?;
    }
    Ok(found)
}

/// [`repair_orphans`] as a note for the user: what moved, or why it could not.
fn repair_note(root: &Path, steam_running: bool) -> Option<String> {
    if steam_running {
        return None;
    }
    match repair_orphans(root, false) {
        Ok(found) if found.is_empty() => None,
        Ok(found) => Some(
            found
                .iter()
                .map(|o| format!("moved {} from {} (not installed) to {}", o.games.iter().map(|g| app_label(g)).collect::<Vec<_>>().join(", "), o.name, o.to))
                .collect::<Vec<_>>()
                .join("; "),
        ),
        Err(e) => Some(format!("games set to a Proton that is not installed could not be moved to its renamed copy: {e}")),
    }
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
/// again when its source changed or its name lacks "Proton" ([`proton_tool::remake`]), its wrapper renewed when Neural
/// Forge's changed), or the one already chosen (kept as it is, its wrapper renewed, when its source is gone: it holds its
/// own files; renamed in place with all its games when its name lacks "Proton", as Make again does).
/// `Ok(None)` for a native Linux game with no Proton chosen: it runs without Proton, where the wrapper does not
/// apply. Refused while Steam runs, and for a game whose installed build cannot be told. The game's earlier choice
/// is recorded (before `config.vdf` is edited) for [`switch_off`].
pub fn switch_on(root: &Path, libraries: &[PathBuf], appid: &str, wrapper: &Path, steam_running: bool) -> Result<Option<SwitchedOn>, String> {
    if steam_running {
        return Err("Steam is running: close it to let Neural Forge choose the game's Proton".into());
    }
    if current(root, appid).is_none() {
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
    // A game left on the old name by a rename that stopped half way moves to the renamed copy first (after the native
    // check: a native game returns nothing to carry the note in); a later failure carries the note.
    let repaired = repair_note(root, steam_running);
    switch_on_core(root, libraries, appid, wrapper, repaired.clone()).map_err(|e| carry_note(&repaired, e))
}

/// [`switch_on`] once the stranded games are repaired (`repaired`, the note to give the user).
fn switch_on_core(root: &Path, libraries: &[PathBuf], appid: &str, wrapper: &Path, repaired: Option<String>) -> Result<Option<SwitchedOn>, String> {
    let tools = proton_tool::tools(root);
    let now = current(root, appid);
    let dll = || std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()));
    let from_tool: Option<&Tool> = now.as_ref().and_then(|n| tools.iter().find(|t| &t.name == n));
    if let Some(t) = from_tool.filter(|t| !proton_tool::needs_remake(t)) {
        if proton_tool::wrapper_changed(t, wrapper) {
            proton_tool::refresh_wrapper(t, &dll()?)?;
        }
        return Ok(Some(SwitchedOn { tool: t.name.clone(), made: false, previous: now.clone(), note: repaired }));
    }
    // The source: what the copy the game is set to was made from, else the game's own Proton.
    let source = match from_tool {
        Some(t) => match proton_tool::source_of(t) {
            Some(s) => s,
            // Gone, so the copy is made again only for its name: renamed in place, with every game set to it.
            None => {
                return match rename(root, t, wrapper, false, Path::new("/proc")) {
                    Ok(done) => {
                        let note = join_notes(repaired, join_notes(done.note, done.kept));
                        Ok(Some(SwitchedOn { tool: done.to, made: false, previous: now, note }))
                    }
                    // Refused (the copy is in use, another tool has the new name, its games could not move): the game
                    // stays on its copy, with the wrapper renewed, and Make again does the rename later.
                    Err(e) => {
                        let refused = rename_refused(t, Path::new("/proc"));
                        let pending = rename_failure(t, e, refused)?;
                        if proton_tool::wrapper_changed(t, wrapper) {
                            proton_tool::refresh_wrapper(t, &dll()?)?;
                        }
                        Ok(Some(SwitchedOn { tool: t.name.clone(), made: false, previous: now, note: join_notes(repaired, Some(pending)) }))
                    }
                };
            }
        },
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
    Ok(Some(SwitchedOn { tool: name, made, previous: now, note: repaired }))
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

/// A Neural Forge copy that is made again ([`proton_tool::remake`], the check `doctor` makes too: its source Proton
/// changed since the copy was made, or its name lacks "Proton"), with what the Status and Setup tabs say about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleCopy {
    pub tool: Tool,
    /// Why.
    pub why: proton_tool::Remake,
    /// The games Steam runs with the copy: their names, else their app ids.
    pub games: Vec<String>,
}

impl StaleCopy {
    /// One line for the user: the copy, its source's two versions or the Steam Cloud problem (both when both apply), and
    /// the games that run with it.
    pub fn describe(&self) -> String {
        use proton_tool::RenamePlan;
        let games = if self.games.is_empty() { "no game is set to it".to_string() } else { format!("used by {}", self.games.join(", ")) };
        let cloud = "Steam Cloud does not sync the saves games keep in its Wine prefix";
        match &self.why {
            proton_tool::Remake::Source(now) if proton_tool::steam_cloud_blind(&self.tool) => format!(
                "made from {}, but {} has {now} now, and its name lacks \u{201c}Proton\u{201d}, so {cloud}; {games}",
                self.tool.source_version,
                self.tool.source.display()
            ),
            proton_tool::Remake::Source(now) => format!("made from {}, but {} has {now} now; {games}", self.tool.source_version, self.tool.source.display()),
            proton_tool::Remake::Name if proton_tool::source_gone(&self.tool) => match proton_tool::rename_plan(&self.tool) {
                RenamePlan::Merge(there) => format!(
                    "its name lacks \u{201c}Proton\u{201d}, so {cloud}; its source is gone and {} is a copy of the same version, so Make again moves its games to it and removes this one; {games}",
                    there.name
                ),
                RenamePlan::Clash(path) => format!(
                    "its name lacks \u{201c}Proton\u{201d}, so {cloud}; its source is gone, but {} is another tool, so Make again cannot rename it: remove or rename that tool first; {games}",
                    path.display()
                ),
                _ => format!(
                    "its name lacks \u{201c}Proton\u{201d}, so {cloud}; its source is gone, so Make again renames it to {} and moves its games; {games}",
                    proton_tool::remade_name(&self.tool)
                ),
            },
            proton_tool::Remake::Name => format!("its name lacks \u{201c}Proton\u{201d}, so {cloud}; {games}"),
        }
    }
}

impl StaleCopy {
    /// Make again renames the copy in place or merges it into one of the same version (its source is gone): its games
    /// move with it, where for any other copy they move to the new copy.
    pub fn in_place(&self) -> bool {
        self.why == proton_tool::Remake::Name && proton_tool::source_gone(&self.tool)
    }
}

/// The copies in `root` that are made again ([`proton_tool::remake`]), each with the games set to it.
pub fn stale_copies(root: &Path) -> Vec<StaleCopy> {
    let stale: Vec<(Tool, proton_tool::Remake)> = proton_tool::tools(root).into_iter().filter_map(|t| proton_tool::remake(&t).map(|why| (t, why))).collect();
    if stale.is_empty() {
        return Vec::new();
    }
    let known = crate::steam::games(&[crate::steam::Root { path: root.to_path_buf(), flatpak: false }]);
    stale
        .into_iter()
        .map(|(tool, why)| {
            let games = games_on(root, |n| n == tool.name)
                .into_iter()
                .map(|appid| match known.iter().find(|g| g.appid == appid) {
                    Some(g) if appid != "0" => g.name.clone(),
                    _ => app_label(&appid),
                })
                .collect();
            StaleCopy { tool, why, games }
        })
        .collect()
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
    /// `from` was renamed to `to` in place (its source is gone: [`proton_tool::rename_in`]), not made again.
    pub renamed: bool,
    /// `from` was merged into `to`, a copy of the same version that was there already ([`proton_tool::Renamed::Existing`]):
    /// its games moved and it goes, nothing was made.
    pub merged: bool,
    /// Something to tell the user that is not an error: the wrapper could not be renewed after the rename (Make again
    /// puts it in), games moved from a copy whose folder had been renamed already.
    pub note: Option<String>,
}

/// Updates `tool` (`proton-tool update`, Make again): the new wrapper put in when only that changed, or when its
/// source is gone and its name says "Proton" (the copy keeps working as it is); else ([`proton_tool::remake`]) the
/// copy made again from its source's current version, and when that is a new name (a new version, or a name that now
/// says "Proton"), every game set to the old copy (the default for all games, "0", too) moved to the new one and the
/// old copy removed (both only while Steam is closed, and never a copy a game runs with). A copy named without
/// "Proton" whose source is gone is renamed in place instead ([`proton_tool::rename_in`]: it holds its own files),
/// its games moved, its wrapper renewed: refused while Steam runs.
pub fn update(root: &Path, tool: &Tool, wrapper: &Path, steam_running: bool) -> Result<Updated, String> {
    update_in(root, tool, wrapper, steam_running, Path::new("/proc"))
}

fn update_in(root: &Path, tool: &Tool, wrapper: &Path, steam_running: bool, proc_root: &Path) -> Result<Updated, String> {
    // A game left on the old name by a rename that stopped half way moves to the renamed copy first.
    let repaired = repair_note(root, steam_running);
    let mut done = update_core(root, tool, wrapper, steam_running, proc_root).map_err(|e| carry_note(&repaired, e))?;
    done.note = join_notes(repaired, done.note);
    Ok(done)
}

fn update_core(root: &Path, tool: &Tool, wrapper: &Path, steam_running: bool, proc_root: &Path) -> Result<Updated, String> {
    let same = |moved| Updated { from: tool.name.clone(), to: tool.name.clone(), moved, kept: None, renamed: false, merged: false, note: None };
    if !proton_tool::needs_remake(tool) {
        let dll = std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()))?;
        proton_tool::refresh_wrapper(tool, &dll)?;
        return Ok(same(Vec::new()));
    }
    // The source as `remake` read it (its folder, whichever library holds it); gone, the copy is made again only for
    // its name.
    let Some(source) = proton_tool::source_of(tool) else { return rename(root, tool, wrapper, steam_running, proc_root) };
    let made = proton_tool::create_in(root, &source, wrapper, proc_root)?;
    if made.tool.name == tool.name {
        return Ok(same(Vec::new()));
    }
    let mut out = Updated { from: tool.name.clone(), to: made.tool.name.clone(), moved: Vec::new(), kept: None, renamed: false, merged: false, note: None };
    if steam_running {
        out.kept = Some("Steam is running: close it and update again to move its games to the new copy and remove the old one".into());
        return Ok(out);
    }
    out.moved = move_games(root, &tool.name, &made.tool.name)?;
    if let Err(e) = proton_tool::remove_in(root, &tool.name, proc_root) {
        out.kept = Some(e);
    }
    Ok(out)
}

/// Every game set to the tool `from` (the default for all games, "0", too) set to `to`, in one edit of `config.vdf`:
/// the games moved.
fn move_games(root: &Path, from: &str, to: &str) -> Result<Vec<String>, String> {
    let games = games_on(root, |n| n == from);
    if !games.is_empty() {
        edit(root, |old| {
            let mut now = old.to_vec();
            for appid in &games {
                if let Some(next) = set_name(&now, appid, to)? {
                    now = next;
                }
            }
            Ok((now != old).then_some(now))
        })?;
    }
    Ok(games)
}

/// [`update`] for a copy named without "Proton" whose source is gone: renamed in place ([`proton_tool::rename_in`]),
/// its games moved to the new name (the folder and its manifest put back as they were when they cannot be) and its
/// wrapper renewed; where a Neural Forge copy of the same version already has the new name (made from the same source or
/// another path to it), the games move to it and the old copy goes. Refused while Steam runs (it would not see the
/// rename, and writes its own `config.vdf` back), while a game runs with the copy, and when another tool has the new
/// name. Once the games moved the rename is done: a wrapper that cannot be renewed after it is a note
/// ([`Updated::note`]), not an error (Make again offers it again).
fn rename(root: &Path, tool: &Tool, wrapper: &Path, steam_running: bool, proc_root: &Path) -> Result<Updated, String> {
    if steam_running {
        return Err(format!(
            "Steam is running: close it to rename {} to {} (its source is gone, so it is renamed in place) and move its games",
            tool.name,
            proton_tool::remade_name(tool)
        ));
    }
    let dll = std::fs::read(wrapper).map_err(|e| format!("Neural Forge's NGX wrapper {}: {e}", wrapper.display()))?;
    let original_manifest = std::fs::read(tool.dir.join("compatibilitytool.vdf")).ok();
    let wrapper_note =
        |t: &Tool| proton_tool::refresh_wrapper(t, &dll).err().map(|e| format!("its NGX wrapper could not be renewed ({e}): make it again to put it in"));
    match proton_tool::rename_in(tool, proc_root)? {
        proton_tool::Renamed::To(new) => {
            let moved = match move_games(root, &tool.name, &new.name) {
                Ok(moved) => moved,
                Err(e) => {
                    proton_tool::undo_rename(&new, tool, original_manifest.as_deref());
                    return Err(e);
                }
            };
            let note = wrapper_note(&new);
            Ok(Updated { from: tool.name.clone(), to: new.name, moved, kept: None, renamed: true, merged: false, note })
        }
        proton_tool::Renamed::Existing(there) => {
            let moved = move_games(root, &tool.name, &there.name)?;
            let note = wrapper_note(&there);
            let kept = proton_tool::remove_in(root, &tool.name, proc_root).err();
            Ok(Updated { from: tool.name.clone(), to: there.name, moved, kept, renamed: false, merged: true, note })
        }
    }
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
        assert_eq!(other.tool, "Proton-experimental-11.0-NeuralForge");
        let text = std::fs::read_to_string(root.join("config/config.vdf")).unwrap();
        assert!(text.contains("\"1174180\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"Proton-experimental-11.0-NeuralForge\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t\t\"priority\"\t\t\"250\""), "{text}");
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
        assert_eq!(on.tool, "Proton-experimental-11.0-NeuralForge");
        let text = std::fs::read_to_string(root.join("config/config.vdf")).unwrap();
        assert!(text.contains("\t\t\t\t\"SurveyDate\"\t\t\"2026-11-01\"\n\t\t\t\t\"CompatToolMapping\"\n\t\t\t\t{\n\t\t\t\t\t\"1174180\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"name\"\t\t\"Proton-experimental-11.0-NeuralForge\""), "{text}");
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
        let up = update_in(&root, &t, &w, false, &procs).unwrap();
        assert_eq!((up.from.as_str(), up.to.as_str(), up.kept), (t.name.as_str(), t.name.as_str(), None));
        assert!(t.dir.is_dir() && !proton_tool::wrapper_changed(&proton_tool::tools(&root)[0], &w));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A copy 4.2.0 named without "Proton" (`experimental-11.0-NeuralForge`, #71) is listed as stale while its source
    /// is there; Make again moves its games to the copy named `Proton-...` and removes it, as for a new version, and
    /// switching a game on that is set to it does the same for that game.
    #[test]
    fn a_copy_named_without_proton_is_stale_and_its_games_move_to_the_new_name() {
        let (root, lib, w) = steam("cloud");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        fake_app(&root, "990080", "Hogwarts Legacy", Some(("HogwartsLegacy.exe", "MZ")));
        switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        let (old_name, new_name) = ("experimental-11.0-NeuralForge", "Proton-experimental-11.0-NeuralForge");
        let old_dir = as_named_by_4_2_0(&root, new_name, old_name, &["1174180", "990080"]);
        let stale = stale_copies(&root);
        assert_eq!(stale.len(), 1);
        assert_eq!((stale[0].tool.name.as_str(), &stale[0].why), (old_name, &proton_tool::Remake::Name));
        let line = stale[0].describe();
        assert!(line.contains("Steam Cloud") && line.contains("RDR2") && line.contains("Hogwarts Legacy"), "{line}");
        // Switching one game on again moves that game: its source is read from the folder the copy names, also when
        // that library is not among those listed.
        let on = switch_on(&root, &[], "990080", &w, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made), (new_name, true));
        assert_eq!(current(&root, "990080").as_deref(), Some(new_name));
        // Make again moves the rest and removes the old copy.
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let done = update_in(&root, &stale[0].tool, &w, false, &procs).unwrap();
        assert_eq!((done.to.as_str(), done.moved, done.kept), (new_name, vec!["1174180".to_string()], None));
        assert_eq!(current(&root, "1174180").as_deref(), Some(new_name));
        assert!(!old_dir.exists() && stale_copies(&root).is_empty());
        // Switched off, the game gets the Proton it had before Neural Forge (none).
        assert_eq!(switch_off(&root, "1174180", false).unwrap().as_deref(), Some(""));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Puts the copy named `new_name` back under the name Neural Forge 4.2.0 gave it (`old_name`: its folder and its
    /// `compatibilitytool.vdf`) and sets `games` to it: the old copy's folder.
    fn as_named_by_4_2_0(root: &Path, new_name: &str, old_name: &str, games: &[&str]) -> PathBuf {
        let made = proton_tool::tools(root).into_iter().find(|t| t.name == new_name).expect("the copy");
        let old_dir = proton_tool::tools_dir(root).join(old_name);
        std::fs::rename(&made.dir, &old_dir).unwrap();
        std::fs::remove_file(old_dir.join("compatibilitytool.vdf")).unwrap();
        std::fs::write(old_dir.join("compatibilitytool.vdf"), proton_tool::manifest(old_name)).unwrap();
        for appid in games {
            edit(root, |old| set_name(old, appid, old_name)).unwrap();
        }
        old_dir
    }

    /// Make again on a copy named without "Proton" while Steam runs makes the new copy and keeps the old one, moving
    /// nothing; with Steam closed it finishes: every game set to it moves, the default for all games ("0") too, and a
    /// game switched off afterwards gets the Proton it had before Neural Forge.
    #[test]
    fn a_name_remake_while_steam_runs_waits_for_steam_to_close_and_moves_the_default_too() {
        let (root, lib, w) = steam("cloudrun");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        // The game had Proton Experimental chosen by name before Neural Forge.
        edit(&root, |old| set_name(old, "1174180", "proton_experimental")).unwrap();
        let on = switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        assert_eq!(on.previous.as_deref(), Some("proton_experimental"));
        let (old_name, new_name) = ("experimental-11.0-NeuralForge", "Proton-experimental-11.0-NeuralForge");
        let old_dir = as_named_by_4_2_0(&root, new_name, old_name, &["0", "1174180"]);
        let old = proton_tool::tools(&root).remove(0);
        assert_eq!(proton_tool::remake(&old), Some(proton_tool::Remake::Name));
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let running = update_in(&root, &old, &w, true, &procs).unwrap();
        assert_eq!((running.to.as_str(), running.moved.len(), running.renamed), (new_name, 0, false));
        assert!(running.kept.unwrap().contains("Steam is running"));
        assert!(old_dir.is_dir() && proton_tool::tools_dir(&root).join(new_name).is_dir());
        assert_eq!((current(&root, "0").as_deref(), current(&root, "1174180").as_deref()), (Some(old_name), Some(old_name)), "nothing moved");
        // Steam closed: the copy is there already; the games move and the old copy goes.
        let done = update_in(&root, &old, &w, false, &procs).unwrap();
        assert_eq!((done.to.as_str(), done.moved, done.kept), (new_name, vec!["0".to_string(), "1174180".to_string()], None));
        assert_eq!((current(&root, "0").as_deref(), current(&root, "1174180").as_deref()), (Some(new_name), Some(new_name)));
        assert!(!old_dir.exists() && stale_copies(&root).is_empty());
        assert_eq!(switch_off(&root, "1174180", false).unwrap().as_deref(), Some("proton_experimental"));
        assert_eq!(current(&root, "1174180").as_deref(), Some("proton_experimental"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A copy named without "Proton" whose source is gone is renamed in place (it holds its own files): its folder and
    /// `compatibilitytool.vdf` get the new name, every game set to it moves (the default "0" too) and its wrapper is
    /// renewed; refused while Steam runs, while a game runs with it and when another tool has the new name. Switching a
    /// game on does the same; a copy of the same version under the new name takes its games instead. Switched off, a
    /// game gets its earlier Proton back.
    #[test]
    fn a_copy_named_without_proton_whose_source_is_gone_is_renamed_in_place() {
        let (root, lib, w) = steam("rename");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        fake_app(&root, "990080", "Hogwarts Legacy", Some(("HogwartsLegacy.exe", "MZ")));
        edit(&root, |old| set_name(old, "990080", "proton_experimental")).unwrap();
        switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        switch_on(&root, libs, "990080", &w, false).unwrap().unwrap();
        let (old_name, new_name) = ("experimental-11.0-NeuralForge", "Proton-experimental-11.0-NeuralForge");
        let games = ["0", "1174180", "990080"];
        let old_dir = as_named_by_4_2_0(&root, new_name, old_name, &games);
        let new_dir = proton_tool::tools_dir(&root).join(new_name);
        std::fs::remove_dir_all(lib.join("steamapps/common/Proton - Experimental")).unwrap();
        let stale = stale_copies(&root);
        assert_eq!((stale.len(), &stale[0].why), (1, &proton_tool::Remake::Name));
        assert!(stale[0].describe().contains(&format!("renames it to {new_name}")), "{}", stale[0].describe());
        let old = stale[0].tool.clone();
        let on_old = |root: &Path| games.iter().all(|a| current(root, a).as_deref() == Some(old_name));
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        // Refused while Steam runs.
        let e = update_in(&root, &old, &w, true, &procs).unwrap_err();
        assert!(e.contains("Steam is running") && e.contains(new_name), "{e}");
        assert!(old_dir.is_dir() && !new_dir.exists() && on_old(&root));
        // Refused while a game runs with it.
        let pid = procs.join("4242");
        std::fs::create_dir_all(&pid).unwrap();
        std::os::unix::fs::symlink(old_dir.join("files/lib/wine/x86_64-unix/wine64"), pid.join("exe")).unwrap();
        let e = update_in(&root, &old, &w, false, &procs).unwrap_err();
        assert!(e.contains("in use by a running game") && e.contains("4242"), "{e}");
        assert!(old_dir.is_dir() && !new_dir.exists() && on_old(&root));
        std::fs::remove_dir_all(&pid).unwrap();
        // Refused when another tool has the new name.
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::write(new_dir.join("compatibilitytool.vdf"), proton_tool::manifest(new_name)).unwrap();
        let e = update_in(&root, &old, &w, false, &procs).unwrap_err();
        assert!(e.contains("already exists and is another tool"), "{e}");
        assert!(old_dir.is_dir() && on_old(&root));
        std::fs::remove_dir_all(&new_dir).unwrap();
        // Renamed, with a new wrapper to put in.
        std::fs::write(&w, [std::fs::read(&w).unwrap(), vec![1]].concat()).unwrap();
        let done = update_in(&root, &old, &w, false, &procs).unwrap();
        assert_eq!((done.from.as_str(), done.to.as_str(), done.renamed, done.kept), (old_name, new_name, true, None));
        let mut moved = done.moved.clone();
        moved.sort();
        assert_eq!(moved, ["0", "1174180", "990080"].map(String::from), "every game, in config.vdf's order");
        assert!(!old_dir.exists() && new_dir.is_dir());
        let vdf = std::fs::read_to_string(new_dir.join("compatibilitytool.vdf")).unwrap();
        assert_eq!(vdf, proton_tool::manifest(new_name));
        let t = proton_tool::tools(&root).remove(0);
        assert_eq!((t.name.as_str(), t.source_version.as_str()), (new_name, "experimental-11.0"));
        assert!(!proton_tool::wrapper_changed(&t, &w) && stale_copies(&root).is_empty());
        assert!(games.iter().all(|a| current(&root, a).as_deref() == Some(new_name)));
        // Switching a game on renames it the same way, with all its games.
        as_named_by_4_2_0(&root, new_name, old_name, &games);
        let on = switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made, on.previous.as_deref()), (new_name, false, Some(old_name)));
        assert!(!old_dir.exists() && games.iter().all(|a| current(&root, a).as_deref() == Some(new_name)));
        // A copy of the same version already under the new name: the games move to it and the old one goes.
        as_named_by_4_2_0(&root, new_name, old_name, &games);
        std::fs::create_dir_all(&new_dir).unwrap();
        std::fs::copy(old_dir.join(proton_tool::MARKER), new_dir.join(proton_tool::MARKER)).unwrap();
        std::fs::write(new_dir.join("compatibilitytool.vdf"), proton_tool::manifest(new_name)).unwrap();
        std::fs::create_dir_all(new_dir.join("files")).unwrap();
        std::fs::write(new_dir.join("proton"), "#!/usr/bin/env python3\n").unwrap();
        let old = proton_tool::tools(&root).into_iter().find(|t| t.name == old_name).unwrap();
        let done = update_in(&root, &old, &w, false, &procs).unwrap();
        assert_eq!((done.to.as_str(), done.renamed, done.kept), (new_name, false, None));
        assert!(!old_dir.exists() && games.iter().all(|a| current(&root, a).as_deref() == Some(new_name)));
        // Switched off: the earlier Proton back, or none.
        assert_eq!(switch_off(&root, "990080", false).unwrap().as_deref(), Some("proton_experimental"));
        assert_eq!(current(&root, "990080").as_deref(), Some("proton_experimental"));
        assert_eq!(switch_off(&root, "1174180", false).unwrap().as_deref(), Some(""));
        assert_eq!(current(&root, "1174180"), None);
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
        let running = update_in(&root, &old, &w, true, &procs).unwrap();
        assert_eq!((running.to.as_str(), running.moved.len()), ("GE-Proton11-13-NeuralForge", 0));
        assert!(running.kept.unwrap().contains("Steam is running") && old.dir.is_dir());
        let done = update_in(&root, &old, &w, false, &procs).unwrap();
        assert_eq!((done.moved, done.kept), (vec!["1091500".to_string()], None));
        assert_eq!(current(&root, "1091500").as_deref(), Some("GE-Proton11-13-NeuralForge"));
        assert!(!old.dir.exists());
        // Switched off, the game gets the Proton it had before Neural Forge.
        assert_eq!(switch_off(&root, "1091500", false).unwrap().as_deref(), Some("Proton-GE Latest"));
        // Only the wrapper changed: put in, same copy.
        let t = proton_tool::tools(&root).remove(0);
        std::fs::write(&w, [std::fs::read(&w).unwrap(), vec![1]].concat()).unwrap();
        let fresh = update_in(&root, &t, &w, false, &procs).unwrap();
        assert_eq!(fresh.to, t.name);
        assert!(!proton_tool::wrapper_changed(&proton_tool::tools(&root)[0], &w));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A copy is stale exactly when its source's version moved on, and names the games set to it; a copy of a source
    /// that is gone, or one that is current, is not listed. Making it again keeps the games on a copy.
    #[test]
    fn stale_copies_name_their_games_and_a_remake_clears_them() {
        let (root, lib, w) = steam("stale");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1091500", "GTA V", None);
        switch_on(&root, libs, "1091500", &w, false).unwrap().unwrap();
        assert!(stale_copies(&root).is_empty(), "current");
        let source = root.join("compatibilitytools.d/Proton-GE Latest");
        fake_proton(&source, "GE-Proton11-13");
        let stale = stale_copies(&root);
        assert_eq!(stale.len(), 1);
        assert_eq!(
            (stale[0].tool.name.as_str(), &stale[0].why, stale[0].games.clone()),
            ("GE-Proton11-12-NeuralForge", &proton_tool::Remake::Source("GE-Proton11-13".into()), vec!["GTA V".to_string()])
        );
        let line = stale[0].describe();
        assert!(line.contains("GE-Proton11-12") && line.contains("GE-Proton11-13") && line.contains("GTA V") && line.contains("Proton-GE Latest"), "{line}");
        // The same check `doctor` and Make again use.
        assert!(proton_tool::source_changed(&stale[0].tool));
        // The Proton the layer is told about at launch is this copy, and it says the same.
        assert!(neural_forge_protocol::proton_copy::stale_notice(&stale[0].tool.dir).is_some());
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let done = update_in(&root, &stale[0].tool, &w, false, &procs).unwrap();
        assert_eq!((done.to.as_str(), done.moved.clone()), ("GE-Proton11-13-NeuralForge", vec!["1091500".to_string()]));
        assert!(stale_copies(&root).is_empty());
        // A source that is gone is no stale copy: it keeps working.
        std::fs::remove_dir_all(&source).unwrap();
        assert!(stale_copies(&root).is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    const OLD: &str = "experimental-11.0-NeuralForge";
    const NEW: &str = "Proton-experimental-11.0-NeuralForge";

    /// A copy named by 4.2.0 whose source is gone, with the default ("0") and a game set to it.
    struct Gone {
        root: PathBuf,
        lib: PathBuf,
        wrapper: PathBuf,
        old_dir: PathBuf,
        new_dir: PathBuf,
        procs: PathBuf,
    }

    impl Gone {
        fn new(tag: &str) -> Self {
            let (root, lib, wrapper) = steam(tag);
            fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
            switch_on(&root, std::slice::from_ref(&lib), "1174180", &wrapper, false).unwrap().unwrap();
            let old_dir = as_named_by_4_2_0(&root, NEW, OLD, &["0", "1174180"]);
            std::fs::remove_dir_all(lib.join("steamapps/common/Proton - Experimental")).unwrap();
            let procs = root.join("proc");
            std::fs::create_dir_all(&procs).unwrap();
            let new_dir = proton_tool::tools_dir(&root).join(NEW);
            Gone { root, lib, wrapper, old_dir, new_dir, procs }
        }

        fn old(&self) -> Tool {
            proton_tool::tools(&self.root).into_iter().find(|t| t.name == OLD).expect("the old copy")
        }

        fn games_on(&self, name: &str) -> bool {
            ["0", "1174180"].iter().all(|a| current(&self.root, a).as_deref() == Some(name))
        }

        /// The wrapper changed, so the copy's would be renewed.
        fn new_wrapper(&self) {
            std::fs::write(&self.wrapper, [std::fs::read(&self.wrapper).unwrap(), vec![1]].concat()).unwrap();
        }

        fn config(&self) -> Vec<u8> {
            std::fs::read(config_vdf(&self.root)).unwrap()
        }

        /// A copy of the same version under the new name, made from `source`.
        fn merge_target(&self, source: &Path) {
            std::fs::create_dir_all(&self.new_dir).unwrap();
            let marker = std::fs::read_to_string(self.old_dir.join(proton_tool::MARKER)).unwrap();
            let marker: String =
                marker.lines().map(|l| if l.starts_with("source=") { format!("source={}", source.display()) } else { l.to_string() } + "\n").collect();
            std::fs::write(self.new_dir.join(proton_tool::MARKER), marker).unwrap();
            std::fs::write(self.new_dir.join("compatibilitytool.vdf"), proton_tool::manifest(NEW)).unwrap();
            std::fs::create_dir_all(self.new_dir.join("files")).unwrap();
            std::fs::write(self.new_dir.join("proton"), "#!/usr/bin/env python3\n").unwrap();
        }
    }

    fn chmod(path: &Path, mode: u32) {
        std::fs::set_permissions(path, std::os::unix::fs::PermissionsExt::from_mode(mode)).unwrap();
    }

    /// Read-only folders do not stop root.
    fn is_root() -> bool {
        // SAFETY: geteuid has no preconditions.
        unsafe { libc::geteuid() == 0 }
    }

    /// A failed `move_games` after the rename puts the folder and its manifest back exactly, and `config.vdf` is as it
    /// was.
    #[test]
    fn a_failed_move_of_the_games_restores_the_folder_and_its_manifest() {
        if is_root() {
            return;
        }
        let g = Gone::new("rbmove");
        let custom = format!("{}// edited by hand\n", proton_tool::manifest(OLD));
        std::fs::write(g.old_dir.join("compatibilitytool.vdf"), &custom).unwrap();
        let (old, before) = (g.old(), g.config());
        let config_dir = config_vdf(&g.root).parent().unwrap().to_path_buf();
        chmod(&config_dir, 0o555);
        let e = update_in(&g.root, &old, &g.wrapper, false, &g.procs);
        chmod(&config_dir, 0o755);
        assert!(e.is_err(), "config.vdf could not be written");
        assert!(g.old_dir.is_dir() && !g.new_dir.exists(), "the folder is back under its name");
        assert_eq!(std::fs::read_to_string(g.old_dir.join("compatibilitytool.vdf")).unwrap(), custom, "the manifest as it was");
        assert_eq!(proton_tool::tools(&g.root).into_iter().map(|t| t.name).collect::<Vec<_>>(), [OLD]);
        assert_eq!(g.config(), before);
        assert!(g.games_on(OLD));
        // Nothing half done: the next try works.
        let done = update_in(&g.root, &g.old(), &g.wrapper, false, &g.procs).unwrap();
        assert_eq!((done.to.as_str(), done.renamed), (NEW, true));
        assert!(g.games_on(NEW));
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// A manifest that cannot be written stops the rename before anything changed: no folder to put back.
    #[test]
    fn a_failed_manifest_write_leaves_everything_untouched() {
        if is_root() {
            return;
        }
        let g = Gone::new("rbmanifest");
        let (old, before) = (g.old(), g.config());
        let manifest = std::fs::read(g.old_dir.join("compatibilitytool.vdf")).unwrap();
        chmod(&g.old_dir, 0o555);
        let e = update_in(&g.root, &old, &g.wrapper, false, &g.procs);
        chmod(&g.old_dir, 0o755);
        let e = e.unwrap_err();
        assert!(e.contains("compatibilitytool.vdf"), "{e}");
        assert!(g.old_dir.is_dir() && !g.new_dir.exists());
        assert_eq!(std::fs::read(g.old_dir.join("compatibilitytool.vdf")).unwrap(), manifest);
        assert!(std::fs::read_dir(&g.old_dir).unwrap().flatten().all(|e| !e.file_name().to_string_lossy().ends_with(".tmp")));
        assert_eq!(g.config(), before);
        assert!(g.games_on(OLD));
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// A folder renamed (manifest first) with the process stopped before the games moved: the games are on a name with no
    /// folder. `doctor` sees it (`orphaned_mappings`); switching a game on and Make again move them; nothing moves while
    /// Steam runs.
    #[test]
    fn games_left_on_a_renamed_copy_s_old_name_are_moved_to_it() {
        let (root, lib, w) = steam("orphan");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        fake_app(&root, "990080", "Hogwarts Legacy", Some(("HogwartsLegacy.exe", "MZ")));
        switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        let strand = |apps: &[&str]| {
            for a in apps {
                edit(&root, |old| set_name(old, a, OLD)).unwrap();
            }
        };
        assert!(orphaned_mappings(&root).is_empty());
        strand(&["0", "1174180"]);
        let found = orphaned_mappings(&root);
        assert_eq!(found, [Orphaned { name: OLD.into(), to: NEW.into(), games: vec!["0".into(), "1174180".into()] }]);
        assert!(repair_orphans(&root, true).unwrap_err().contains("Steam is running"));
        assert_eq!(current(&root, "1174180").as_deref(), Some(OLD));
        // A copy of the old name that is there is no orphan.
        let there = proton_tool::tools_dir(&root).join(OLD);
        std::fs::create_dir_all(&there).unwrap();
        assert!(orphaned_mappings(&root).is_empty());
        std::fs::remove_dir_all(&there).unwrap();
        // Switching another game on repairs the stranded ones first, and says so.
        edit(&root, |old| set_name(old, "990080", "proton_experimental")).unwrap();
        let on = switch_on(&root, libs, "990080", &w, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made), (NEW, false));
        assert!(on.note.as_deref().unwrap().contains(OLD), "{:?}", on.note);
        assert!(orphaned_mappings(&root).is_empty());
        assert_eq!((current(&root, "0").as_deref(), current(&root, "1174180").as_deref()), (Some(NEW), Some(NEW)));
        // Make again (any copy) repairs them too; the games move only while Steam is closed.
        strand(&["1174180"]);
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let copy = proton_tool::tools(&root).remove(0);
        let running = update_in(&root, &copy, &w, true, &procs).unwrap();
        assert!(running.note.is_none() && orphaned_mappings(&root).len() == 1);
        let done = update_in(&root, &copy, &w, false, &procs).unwrap();
        assert!(done.note.unwrap().contains("app 1174180"));
        assert!(orphaned_mappings(&root).is_empty() && current(&root, "1174180").as_deref() == Some(NEW));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A Neural Forge copy of the same version under the new name, made from another source path, takes the games of
    /// the old copy and the old copy goes, like one from the same path; the list says so and the default for all games
    /// reads as such.
    #[test]
    fn a_copy_of_the_same_version_from_another_path_takes_the_games() {
        let g = Gone::new("merge");
        g.merge_target(Path::new("/elsewhere/Proton - Experimental"));
        let old = g.old();
        assert!(matches!(proton_tool::rename_plan(&old), proton_tool::RenamePlan::Merge(t) if t.name == NEW));
        let stale = stale_copies(&g.root);
        let line = stale.iter().find(|s| s.tool.name == OLD).unwrap().describe();
        assert!(line.contains("moves its games to it and removes this one") && line.contains(NEW), "{line}");
        assert!(line.contains("the default for every game") && !line.contains("app 0"), "{line}");
        let done = update_in(&g.root, &old, &g.wrapper, false, &g.procs).unwrap();
        assert_eq!((done.to.as_str(), done.renamed, done.merged, done.kept), (NEW, false, true, None));
        assert_eq!(done.moved, ["0", "1174180"]);
        assert!(!g.old_dir.exists() && g.games_on(NEW));
        std::fs::remove_dir_all(&g.root).unwrap();
        // Another tool (no marker) or another version under the name is still refused.
        let g = Gone::new("clash");
        std::fs::create_dir_all(&g.new_dir).unwrap();
        assert!(matches!(proton_tool::rename_plan(&g.old()), proton_tool::RenamePlan::Clash(_)));
        assert!(stale_copies(&g.root).iter().find(|s| s.tool.name == OLD).unwrap().describe().contains("is another tool"));
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// Switching a game on whose copy cannot be renamed now (another tool has the new name) keeps it on its copy with
    /// the wrapper renewed and says the rename is pending; it does not fail.
    #[test]
    fn switching_on_keeps_the_game_on_its_copy_when_the_rename_is_refused() {
        let g = Gone::new("pending");
        std::fs::create_dir_all(&g.new_dir).unwrap();
        g.new_wrapper();
        let before = g.config();
        let on = switch_on(&g.root, std::slice::from_ref(&g.lib), "1174180", &g.wrapper, false).unwrap().unwrap();
        assert_eq!((on.tool.as_str(), on.made, on.previous.as_deref()), (OLD, false, Some(OLD)));
        let note = on.note.unwrap();
        assert!(note.contains("another tool") && note.contains("still pending") && note.contains(NEW), "{note}");
        assert!(!proton_tool::wrapper_changed(&g.old(), &g.wrapper), "the wrapper is renewed on the copy it stays on");
        assert_eq!(g.config(), before);
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// A wrapper that cannot be renewed once the rename and the move are done is a note: the rename is reported as
    /// done, and Make again offers the wrapper again.
    #[test]
    fn a_wrapper_that_cannot_be_renewed_after_the_rename_is_a_note() {
        if is_root() {
            return;
        }
        let g = Gone::new("wrappernote");
        g.new_wrapper();
        let builtin = g.old_dir.join("files/lib/wine/x86_64-windows");
        chmod(&builtin, 0o555);
        let done = update_in(&g.root, &g.old(), &g.wrapper, false, &g.procs);
        chmod(&g.new_dir.join("files/lib/wine/x86_64-windows"), 0o755);
        let done = done.unwrap();
        assert_eq!((done.to.as_str(), done.renamed), (NEW, true));
        assert!(done.note.as_deref().unwrap().contains("wrapper"), "{:?}", done.note);
        assert!(g.new_dir.is_dir() && !g.old_dir.exists() && g.games_on(NEW));
        let t = proton_tool::tools(&g.root).remove(0);
        assert!(proton_tool::outdated(&t) || proton_tool::wrapper_changed(&t, &g.wrapper), "the wrapper is still to be renewed");
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// A copy of the same version under the new name that lacks its `proton` script, its `files/` or its manifest entry
    /// is not merged into: the plan is a clash, so the games and the working copy stay as they are.
    #[test]
    fn a_damaged_copy_under_the_new_name_never_takes_the_games() {
        for damage in ["proton", "files", "manifest"] {
            let g = Gone::new(&format!("damaged-{damage}"));
            g.merge_target(Path::new("/elsewhere/Proton - Experimental"));
            assert!(matches!(proton_tool::rename_plan(&g.old()), proton_tool::RenamePlan::Merge(_)));
            match damage {
                "proton" => std::fs::remove_file(g.new_dir.join("proton")).unwrap(),
                "files" => std::fs::remove_dir_all(g.new_dir.join("files")).unwrap(),
                _ => std::fs::write(g.new_dir.join("compatibilitytool.vdf"), proton_tool::manifest("something-else")).unwrap(),
            }
            assert!(matches!(proton_tool::rename_plan(&g.old()), proton_tool::RenamePlan::Clash(_)), "{damage}");
            let before = g.config();
            let e = update_in(&g.root, &g.old(), &g.wrapper, false, &g.procs).unwrap_err();
            assert!(e.contains("already exists"), "{e}");
            assert!(g.old_dir.is_dir() && g.games_on(OLD) && g.config() == before, "{damage}: nothing moved or deleted");
            std::fs::remove_dir_all(&g.root).unwrap();
        }
    }

    /// Only a refusal with the folder still there is "pending"; any other failure is the real error, and a vanished
    /// folder says the games are stranded and how they are repaired.
    #[test]
    fn a_failed_rename_is_pending_only_when_refused() {
        let g = Gone::new("failure");
        let old = g.old();
        let pending = rename_failure(&old, "refused".into(), true).unwrap();
        assert!(pending.contains("still pending") && pending.contains(NEW), "{pending}");
        assert_eq!(rename_failure(&old, "disk error".into(), false).unwrap_err(), "disk error");
        // The refusal check: another tool with the new name.
        assert!(!rename_refused(&old, &g.procs));
        std::fs::create_dir_all(&g.new_dir).unwrap();
        assert!(rename_refused(&old, &g.procs));
        std::fs::remove_dir_all(&g.new_dir).unwrap();
        // The folder gone: both the rename and its undo failed.
        std::fs::remove_dir_all(&g.old_dir).unwrap();
        let e = rename_failure(&old, "disk error".into(), true).unwrap_err();
        assert!(e.contains("disk error") && e.contains("stranded") && e.contains("Make again") && e.contains("proton-tool update") && e.contains(NEW), "{e}");
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// A rename that fails for another reason than a refusal is an error, not a pending note, and the games stay put.
    #[test]
    fn switching_on_surfaces_a_rename_that_fails_for_another_reason() {
        if is_root() {
            return;
        }
        let g = Gone::new("realerror");
        let before = g.config();
        let config_dir = config_vdf(&g.root).parent().unwrap().to_path_buf();
        chmod(&config_dir, 0o555);
        let e = switch_on(&g.root, std::slice::from_ref(&g.lib), "1174180", &g.wrapper, false);
        chmod(&config_dir, 0o755);
        let e = e.unwrap_err();
        assert!(!e.contains("still pending"), "{e}");
        assert!(g.old_dir.is_dir() && g.games_on(OLD) && g.config() == before);
        std::fs::remove_dir_all(&g.root).unwrap();
    }

    /// The repair of stranded games is not lost when the switch-on or the update fails afterwards: the error says it.
    #[test]
    fn a_failure_after_the_repair_still_says_games_were_moved() {
        let (root, lib, w) = steam("repairfail");
        let libs = std::slice::from_ref(&lib);
        fake_app(&root, "1174180", "RDR2", Some(("RDR2.exe", "MZ")));
        fake_app(&root, "777", "Other", Some(("Other.exe", "MZ")));
        switch_on(&root, libs, "1174180", &w, false).unwrap().unwrap();
        for a in ["0", "1174180"] {
            edit(&root, |old| set_name(old, a, OLD)).unwrap();
        }
        edit(&root, |old| set_name(old, "777", "Missing Proton")).unwrap();
        let e = switch_on(&root, libs, "777", &w, false).unwrap_err();
        assert!(e.contains("not found among the installed") && e.contains("before that: moved") && e.contains(OLD), "{e}");
        assert!(orphaned_mappings(&root).is_empty(), "the repair itself stands");
        // The same for Make again: a wrapper that cannot be read fails it after the repair.
        for a in ["0", "1174180"] {
            edit(&root, |old| set_name(old, a, OLD)).unwrap();
        }
        let copy = proton_tool::tools(&root).remove(0);
        let procs = root.join("proc");
        std::fs::create_dir_all(&procs).unwrap();
        let e = update_in(&root, &copy, &root.join("missing.dll"), false, &procs).unwrap_err();
        assert!(e.contains("before that: moved") && e.contains(OLD), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
