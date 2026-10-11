//! Steam on Linux: where it is installed, which games it has, and turning Neural Forge on or off in
//! a game's launch options.
//!
//! All of it is plain files under a Steam root: `steamapps/libraryfolders.vdf` lists the libraries,
//! each library's `steamapps/appmanifest_<appid>.acf` describes one installed app, and
//! `userdata/<account>/config/localconfig.vdf` holds each account's per-game `LaunchOptions`. Every
//! function takes the root (or the home directory, or `/proc`) as a parameter, so tests drive them
//! with fixture trees and never touch a real Steam install.
//!
//! Writing `localconfig.vdf` is guarded ([`apply`]): never while Steam runs (it keeps the file in
//! memory and writes it back on exit, discarding the edit), only one value spliced into the original
//! bytes, the result re-parsed and required to be the original tree plus that one change, backups
//! next to the file, and an atomic replace that keeps the file's mode. Anything that cannot be done
//! that way returns an [`Outcome`] carrying the exact string to paste into Steam by hand instead.
//! The guarded-write sequence follows DLSS5oneclick-forlinux's `apply_with` in
//! `src/platform/steam.rs` (MIT; see ATTRIBUTION.md).

use crate::launch_options;
use crate::vdf;
use std::path::{Path, PathBuf};

/// A Steam installation directory (it has `steamapps/`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Root {
    /// Canonical: symlinks resolved, so `~/.steam/steam` and `~/.local/share/Steam` are one root.
    pub path: PathBuf,
    /// The Flatpak build's root, under `~/.var/app/com.valvesoftware.Steam`. Its games run inside
    /// the Flatpak sandbox, which does not see a Vulkan layer installed on the host.
    pub flatpak: bool,
}

const FLATPAK_ROOT: &str = ".var/app/com.valvesoftware.Steam/.local/share/Steam";

/// The Steam roots under `home`: native (`~/.local/share/Steam`, `~/.steam/steam`), Flatpak and
/// Snap, each only if it has `steamapps/`, deduplicated after resolving symlinks.
pub fn roots_from(home: &Path) -> Vec<Root> {
    let candidates = [".local/share/Steam", ".steam/steam", FLATPAK_ROOT, "snap/steam/common/.local/share/Steam"];
    let mut out: Vec<Root> = Vec::new();
    for candidate in candidates {
        let path = home.join(candidate);
        if !path.join("steamapps").is_dir() {
            continue;
        }
        let path = path.canonicalize().unwrap_or(path);
        if !out.iter().any(|r| r.path == path) {
            out.push(Root { path, flatpak: candidate == FLATPAK_ROOT });
        }
    }
    out
}

/// [`roots_from`] the user's `$HOME`.
pub fn roots() -> Vec<Root> {
    roots_from(Path::new(&crate::paths::home()))
}

/// True when Steam is installed only as the Flatpak build: then no game it runs can load the layer.
pub fn only_flatpak(roots: &[Root]) -> bool {
    !roots.is_empty() && roots.iter().all(|r| r.flatpak)
}

/// Every library of `root` (its own first), canonical and deduplicated, each with `steamapps/`.
pub fn libraries(root: &Path) -> Vec<PathBuf> {
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let mut out = vec![canon(root)];
    let Ok(text) = std::fs::read(root.join("steamapps/libraryfolders.vdf")) else { return out };
    let Ok(tree) = vdf::parse(&text) else { return out };
    let Some(folders) = tree.block_at(&["libraryfolders"]) else { return out };
    for entry in &folders.0 {
        let vdf::Value::Block(folder) = &entry.value else { continue };
        let Some(path) = folder.string_at(&["path"]) else { continue };
        let lib = canon(Path::new(path));
        if lib.join("steamapps").is_dir() && !out.contains(&lib) {
            out.push(lib);
        }
    }
    out
}

/// One installed Steam game.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Game {
    pub appid: String,
    pub name: String,
    /// Its folder name under `steamapps/common`.
    pub installdir: String,
    /// The library holding it.
    pub library: PathBuf,
    /// The Steam root it was found through (whose `userdata/` holds its launch options).
    pub root: Root,
}

/// Apps in `steamapps` that are Steam's own tools rather than games: Proton builds, the Steam Linux
/// Runtimes, Steamworks Common Redistributables and SteamVR. Known appids first; the name rules
/// catch builds released after this list was written.
const TOOL_APPIDS: &[&str] = &[
    "228980",  // Steamworks Common Redistributables
    "250820",  // SteamVR
    "1070560", // Steam Linux Runtime 1.0 (scout)
    "1391110", // Steam Linux Runtime 2.0 (soldier)
    "1628350", // Steam Linux Runtime 3.0 (sniper)
    "4183110", // Steam Linux Runtime 4.0
    "1420170", // Proton 5.13
    "1580130", // Proton 6.3
    "1887720", // Proton 7.0
    "2348590", // Proton 8.0
    "2805730", // Proton 9.0
    "4628710", // Proton 11.0
    "1493710", // Proton Experimental
    "2180100", // Proton Hotfix
    "1161040", // Proton BattlEye Runtime
    "1826330", // Proton EasyAntiCheat Runtime
];

pub fn is_tool(appid: &str, name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    TOOL_APPIDS.contains(&appid)
        || name.starts_with("proton ")
        || name.starts_with("steam linux runtime")
        || name.starts_with("steamworks common redistributables")
        || name.starts_with("steamvr")
}

/// The games in every library of every root (tools filtered out), sorted by name. A library two
/// roots share is read once.
pub fn games(roots: &[Root]) -> Vec<Game> {
    let mut out: Vec<Game> = Vec::new();
    for root in roots {
        for library in libraries(&root.path) {
            let Ok(dir) = std::fs::read_dir(library.join("steamapps")) else { continue };
            let mut manifests: Vec<PathBuf> = dir
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("appmanifest_") && n.ends_with(".acf")))
                .collect();
            manifests.sort();
            for manifest in manifests {
                let Some(tree) = std::fs::read(&manifest).ok().and_then(|t| vdf::parse(&t).ok()) else { continue };
                let field = |key: &str| tree.string_at(&["AppState", key]).map(str::to_string);
                let (Some(appid), Some(name), Some(installdir)) = (field("appid"), field("name"), field("installdir")) else { continue };
                if is_tool(&appid, &name) || out.iter().any(|g| g.appid == appid && g.library == library) {
                    continue;
                }
                out.push(Game { appid, name, installdir, library: library.clone(), root: root.clone() });
            }
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then_with(|| a.appid.cmp(&b.appid)));
    out
}

/// Why a name or appid picked no single game.
#[derive(Debug, PartialEq, Eq)]
pub enum FindError {
    NotFound,
    /// More than one game matches; their names and appids.
    Ambiguous(Vec<(String, String)>),
}

/// The game `query` names: an exact appid, else an exact name (ignoring case), else the one game
/// whose name contains it (ignoring case).
pub fn find<'a>(games: &'a [Game], query: &str) -> Result<&'a Game, FindError> {
    let query = query.trim();
    let lower = query.to_lowercase();
    let pick = |matches: Vec<&'a Game>| match matches.as_slice() {
        [] => None,
        [one] => Some(Ok(*one)),
        many => Some(Err(FindError::Ambiguous(many.iter().map(|g| (g.name.clone(), g.appid.clone())).collect()))),
    };
    pick(games.iter().filter(|g| g.appid == query).collect())
        .or_else(|| pick(games.iter().filter(|g| g.name.to_lowercase() == lower).collect()))
        .or_else(|| pick(games.iter().filter(|g| !lower.is_empty() && g.name.to_lowercase().contains(&lower)).collect()))
        .unwrap_or(Err(FindError::NotFound))
}

/// The game whose install folder holds an executable named `exe` (ignoring case), searched a few
/// folders deep (`Game/Binaries/Win64/Game-Win64-Shipping.exe` is four). The channel names a running
/// game by its executable; this is how the report finds its app id, and from it its Proton.
pub fn game_for_exe<'a>(games: &'a [Game], exe: &str) -> Option<&'a Game> {
    const DEPTH: usize = 4;
    fn holds(dir: &Path, exe: &str, depth: usize) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else { return false };
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_file() && e.file_name().to_str().is_some_and(|n| n.eq_ignore_ascii_case(exe)) {
                return true;
            }
            if kind.is_dir() && depth > 1 {
                subdirs.push(e.path());
            }
        }
        subdirs.iter().any(|d| holds(d, exe, depth - 1))
    }
    let exe = exe.trim();
    if exe.is_empty() || exe.contains('/') {
        return None;
    }
    games.iter().find(|g| holds(&g.library.join("steamapps/common").join(&g.installdir), exe, DEPTH))
}

/// Whether the Steam client is running: a process under `proc_root` (normally `/proc`) whose
/// `comm` is `steam` or `steamwebhelper` (`comm` is cut at 15 characters; both names are shorter).
pub fn steam_running(proc_root: &Path) -> bool {
    let Ok(dir) = std::fs::read_dir(proc_root) else { return false };
    dir.flatten().filter(|e| e.file_name().to_str().is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))).any(|e| {
        std::fs::read_to_string(e.path().join("comm")).is_ok_and(|comm| matches!(comm.trim_end_matches('\n'), "steam" | "steamwebhelper"))
    })
}

/// Every account's `localconfig.vdf` under `root`, sorted.
pub fn localconfigs(root: &Path) -> Vec<PathBuf> {
    let Ok(dir) = std::fs::read_dir(root.join("userdata")) else { return Vec::new() };
    let mut out: Vec<PathBuf> = dir.flatten().map(|e| e.path().join("config/localconfig.vdf")).filter(|p| p.is_file()).collect();
    out.sort();
    out
}

const APPS: [&str; 5] = ["UserLocalConfigStore", "Software", "Valve", "Steam", "apps"];

fn app_path(appid: &str) -> Vec<&str> {
    let mut path = APPS.to_vec();
    path.push(appid);
    path
}

fn options_path(appid: &str) -> Vec<&str> {
    let mut path = app_path(appid);
    path.push("LaunchOptions");
    path
}

/// One account's view of a game: whether its `localconfig.vdf` has the game's block, and the
/// launch options in it (`None` when the key is absent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountOptions {
    pub file: PathBuf,
    pub has_app: bool,
    pub options: Option<String>,
}

/// The game's launch options in every account under `root` (accounts whose file does not parse are
/// left out).
pub fn launch_options(root: &Path, appid: &str) -> Vec<AccountOptions> {
    localconfigs(root)
        .into_iter()
        .filter_map(|file| {
            let tree = vdf::parse(&std::fs::read(&file).ok()?).ok()?;
            let has_app = tree.path(&app_path(appid)).is_some();
            let options = tree.string_at(&options_path(appid)).map(str::to_string);
            Some(AccountOptions { file, has_app, options })
        })
        .collect()
}

/// Whether any account under `root` launches the game with Neural Forge on.
pub fn enabled(root: &Path, appid: &str) -> bool {
    launch_options(root, appid).iter().any(|a| a.options.as_deref().is_some_and(launch_options::is_enabled))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Turn Neural Forge on; the target executable may be empty.
    Enable { target_exe: String },
    Disable,
}

impl Action {
    fn apply_to(&self, options: &str) -> String {
        match self {
            Action::Enable { target_exe } => launch_options::merge(options, target_exe),
            Action::Disable => launch_options::strip(options),
        }
    }
}

/// One file [`apply`] rewrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEdit {
    pub file: PathBuf,
    /// The copy taken just before this edit (`localconfig.vdf.neural-forge.bak`).
    pub backup: PathBuf,
    pub before: String,
    pub after: String,
}

/// What [`apply`] did. Only `Changed` and `Unchanged` mean Steam's config now says what was asked;
/// the others carry `manual`, the complete launch-options string to paste into the game's
/// Properties in Steam instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Changed(Vec<FileEdit>),
    /// Every account that has the game already had it this way.
    Unchanged,
    /// Steam is running: it would overwrite the edit when it exits.
    SteamRunning { manual: String },
    /// Something prevented a safe edit; nothing was written (or, for a write error, see `reason`).
    Failed { reason: String, manual: String },
}

/// [`apply`] against the real `/proc`.
pub fn apply_now(root: &Path, appid: &str, action: &Action) -> Outcome {
    apply(root, appid, action, steam_running(Path::new("/proc")))
}

/// Sets the game's launch options in `root`'s `localconfig.vdf` files.
///
/// Which accounts: every account whose file already has the game's block (the accounts that have
/// run it); when none has, enabling creates the block only in the most recently modified file (the
/// account last signed in), and disabling has nothing to do. Every edit is prepared and verified
/// before any file is written, so a refusal leaves all of them untouched. For each file written:
/// `localconfig.vdf.neural-forge.orig` is a copy of it as first found (taken once, never
/// replaced), `localconfig.vdf.neural-forge.bak` a copy from just before this edit, and the new
/// file replaces the old one atomically with the old one's permissions.
pub fn apply(root: &Path, appid: &str, action: &Action, steam_running: bool) -> Outcome {
    let accounts = launch_options(root, appid);
    let current = accounts.iter().find(|a| a.has_app).and_then(|a| a.options.clone()).unwrap_or_default();
    let manual = action.apply_to(&current);
    if steam_running {
        return Outcome::SteamRunning { manual };
    }
    let files = localconfigs(root);
    if files.is_empty() {
        return Outcome::Failed { reason: format!("no Steam account found under {}", root.join("userdata").display()), manual };
    }
    if accounts.len() != files.len() {
        let unreadable: Vec<String> = files.iter().filter(|f| !accounts.iter().any(|a| &a.file == *f)).map(|f| f.display().to_string()).collect();
        return Outcome::Failed { reason: format!("cannot read {}", unreadable.join(", ")), manual };
    }
    let mut targets: Vec<&AccountOptions> = accounts.iter().filter(|a| a.has_app).collect();
    if targets.is_empty() {
        if *action == Action::Disable {
            return Outcome::Unchanged;
        }
        let newest = accounts.iter().max_by_key(|a| std::fs::metadata(&a.file).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH));
        targets.extend(newest);
    }
    let path = options_path(appid);
    let mut planned = Vec::new();
    for account in targets {
        let before = account.options.clone().unwrap_or_default();
        let after = action.apply_to(&before);
        if after == before {
            continue;
        }
        let refuse = |why: String| Outcome::Failed { reason: format!("{}: {why}; the file was left as it was", account.file.display()), manual: manual.clone() };
        let old = match std::fs::read(&account.file) {
            Ok(old) => old,
            Err(e) => return refuse(e.to_string()),
        };
        let new = match vdf::set_string(&old, &path, &after) {
            Ok((new, _)) => new,
            Err(e) => return refuse(e.to_string()),
        };
        if !vdf::verify_edit(&old, &new, &path, &after) {
            return refuse("the edited file did not re-read as the original with only the launch options changed".into());
        }
        planned.push((account.file.clone(), old, new, before, after));
    }
    if planned.is_empty() {
        return Outcome::Unchanged;
    }
    let mut edits = Vec::new();
    for (file, old, new, before, after) in planned {
        match write_with_backups(&file, &old, &new) {
            Ok(backup) => edits.push(FileEdit { file, backup, before, after }),
            Err(e) => {
                let done = if edits.is_empty() { String::new() } else { format!(" ({} other account(s) were already updated)", edits.len()) };
                return Outcome::Failed { reason: format!("writing {} failed: {e}{done}", file.display()), manual };
            }
        }
    }
    Outcome::Changed(edits)
}

fn sibling(file: &Path, suffix: &str) -> PathBuf {
    let mut name = file.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    file.with_file_name(name)
}

/// The backups and the atomic replace for one file, whose bytes read as `old` a moment ago.
fn write_with_backups(file: &Path, old: &[u8], new: &[u8]) -> std::io::Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(file)?.permissions().mode() & 0o7777;
    let orig = sibling(file, ".neural-forge.orig");
    if !orig.exists() {
        crate::paths::write_atomic(&orig, old, mode)?;
    }
    let backup = sibling(file, ".neural-forge.bak");
    crate::paths::write_atomic(&backup, old, mode)?;
    if std::fs::read(file)? != old {
        return Err(std::io::Error::other("the file changed while it was being edited"));
    }
    crate::paths::write_atomic(file, new, mode)?;
    Ok(backup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A scratch directory under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("neural-forge-steam-{name}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn a_running_exe_names_its_game() {
        let s = Scratch::new("exe");
        let game = |appid: &str, name: &str| Game { appid: appid.into(), name: name.into(), installdir: name.into(), library: s.0.clone(), root: Root { path: s.0.clone(), flatpak: false } };
        let games = vec![game("1", "Alpha"), game("2", "Remnant2"), game("3", "Deep")];
        write(&s.0.join("steamapps/common/Alpha/alpha.exe"), "");
        write(&s.0.join("steamapps/common/Remnant2/Remnant2/Binaries/Win64/Remnant2-Win64-Shipping.exe"), "");
        write(&s.0.join("steamapps/common/Deep/a/b/c/d/e/deep.exe"), "");
        assert_eq!(game_for_exe(&games, "ALPHA.EXE").map(|g| g.appid.as_str()), Some("1"), "case is ignored");
        assert_eq!(game_for_exe(&games, "Remnant2-Win64-Shipping.exe").map(|g| g.appid.as_str()), Some("2"));
        assert!(game_for_exe(&games, "deep.exe").is_none(), "deeper than four folders is not searched");
        assert!(game_for_exe(&games, "missing.exe").is_none());
        assert!(game_for_exe(&games, "").is_none());
        assert!(game_for_exe(&games, "Alpha/alpha.exe").is_none(), "a name, not a path");
    }

    fn manifest(appid: &str, name: &str) -> String {
        format!("\"AppState\"\n{{\n\t\"appid\"\t\t\"{appid}\"\n\t\"name\"\t\t\"{name}\"\n\t\"installdir\"\t\t\"{name}\"\n}}\n")
    }

    /// A `localconfig.vdf` with the given `(appid, launch options)` app blocks (`None`: the block
    /// without a LaunchOptions key), Steam's layout and tabs.
    fn localconfig(apps: &[(&str, Option<&str>)]) -> String {
        let mut blocks = String::new();
        for (appid, options) in apps {
            blocks += &format!("\t\t\t\t\t\"{appid}\"\n\t\t\t\t\t{{\n\t\t\t\t\t\t\"LastPlayed\"\t\t\"1791677747\"\n");
            if let Some(options) = options {
                blocks += &format!("\t\t\t\t\t\t\"LaunchOptions\"\t\t{}\n", vdf::quote(options));
            }
            blocks += "\t\t\t\t\t\t\"Playtime\"\t\t\"6916\"\n\t\t\t\t\t}\n";
        }
        format!("\"UserLocalConfigStore\"\n{{\n\t\"Broadcast\"\n\t{{\n\t\t\"Permissions\"\t\t\"1\"\n\t}}\n\t\"Software\"\n\t{{\n\t\t\"Valve\"\n\t\t{{\n\t\t\t\"Steam\"\n\t\t\t{{\n\t\t\t\t\"apps\"\n\t\t\t\t{{\n{blocks}\t\t\t\t}}\n\t\t\t}}\n\t\t}}\n\t}}\n\t\"apps\"\n\t{{\n\t\t\"3240220\"\n\t\t{{\n\t\t\t\"OverlayAppEnable\"\t\t\"1\"\n\t\t}}\n\t}}\n}}\n")
    }

    fn options_in(file: &Path, appid: &str) -> Option<String> {
        vdf::parse(&std::fs::read(file).unwrap()).unwrap().string_at(&options_path(appid)).map(str::to_string)
    }

    fn enable() -> Action {
        Action::Enable { target_exe: String::new() }
    }

    #[test]
    fn roots_are_found_deduplicated_and_flatpak_is_marked() {
        let home = Scratch::new("roots");
        let h = &home.0;
        assert!(roots_from(h).is_empty());
        std::fs::create_dir_all(h.join(".local/share/Steam/steamapps")).unwrap();
        std::fs::create_dir_all(h.join(".steam")).unwrap();
        std::os::unix::fs::symlink(h.join(".local/share/Steam"), h.join(".steam/steam")).unwrap();
        std::fs::create_dir_all(h.join("snap/steam/common/.local/share/Steam")).unwrap(); // no steamapps: not a root
        let roots = roots_from(h);
        assert_eq!(roots, [Root { path: h.join(".local/share/Steam").canonicalize().unwrap(), flatpak: false }]);
        assert!(!only_flatpak(&roots));
        std::fs::create_dir_all(h.join(FLATPAK_ROOT).join("steamapps")).unwrap();
        let roots = roots_from(h);
        assert_eq!(roots.len(), 2);
        assert!(roots[1].flatpak && !only_flatpak(&roots));
        assert!(only_flatpak(&roots[1..]));
        assert!(!only_flatpak(&[]));
    }

    #[test]
    fn games_come_from_every_library_without_steams_own_tools() {
        let home = Scratch::new("games");
        let root = home.0.join("Steam");
        let other = home.0.join("Storage/Steam");
        std::fs::create_dir_all(other.join("steamapps")).unwrap();
        write(
            &root.join("steamapps/libraryfolders.vdf"),
            &format!("\"libraryfolders\"\n{{\n\t\"0\"\n\t{{\n\t\t\"path\"\t\t\"{}\"\n\t}}\n\t\"1\"\n\t{{\n\t\t\"path\"\t\t\"{}\"\n\t}}\n\t\"2\"\n\t{{\n\t\t\"path\"\t\t\"/nonexistent\"\n\t}}\n}}\n", root.display(), other.display()),
        );
        write(&root.join("steamapps/appmanifest_228980.acf"), &manifest("228980", "Steamworks Common Redistributables"));
        write(&root.join("steamapps/appmanifest_990080.acf"), &manifest("990080", "Hogwarts Legacy"));
        write(&other.join("steamapps/appmanifest_3240220.acf"), &manifest("3240220", "Grand Theft Auto V Enhanced"));
        write(&other.join("steamapps/appmanifest_1817070.acf"), &manifest("1817070", "Marvel’s Spider-Man Remastered"));
        write(&other.join("steamapps/appmanifest_1493710.acf"), &manifest("1493710", "Proton Experimental"));
        write(&other.join("steamapps/appmanifest_1628350.acf"), &manifest("1628350", "Steam Linux Runtime 3.0 (sniper)"));
        write(&other.join("steamapps/appmanifest_9999999.acf"), &manifest("9999999", "Proton 12.0")); // unknown appid, name rule
        write(&other.join("steamapps/appmanifest_1.acf"), "\"AppState\" { \"appid\" \"1\" }"); // incomplete: skipped
        write(&other.join("steamapps/appmanifest_2.acf"), "not { vdf"); // broken: skipped
        let roots = [Root { path: root.clone(), flatpak: false }];
        assert_eq!(libraries(&root), [root.canonicalize().unwrap(), other.canonicalize().unwrap()]);
        let games = games(&roots);
        let names: Vec<(&str, &str)> = games.iter().map(|g| (g.appid.as_str(), g.name.as_str())).collect();
        assert_eq!(names, [("3240220", "Grand Theft Auto V Enhanced"), ("990080", "Hogwarts Legacy"), ("1817070", "Marvel’s Spider-Man Remastered")]);
        assert_eq!(games[0].library, other.canonicalize().unwrap());
        assert_eq!(games[0].installdir, "Grand Theft Auto V Enhanced");
        // The same root listed twice still gives each game once.
        assert_eq!(super::games(&[roots[0].clone(), roots[0].clone()]).len(), 3);
    }

    #[test]
    fn tools_are_recognised_by_appid_and_by_name() {
        assert!(is_tool("228980", "anything"));
        assert!(is_tool("4628710", "Proton 11.0"));
        assert!(is_tool("123", "Proton 12.0-beta"));
        assert!(is_tool("123", "Steam Linux Runtime 5.0 (future)"));
        assert!(is_tool("123", "SteamVR Beta"));
        assert!(is_tool("123", "Steamworks Common Redistributables"));
        assert!(!is_tool("3240220", "Grand Theft Auto V Enhanced"));
        assert!(!is_tool("123", "Protonball"), "a game whose name starts with the letters is not a tool");
    }

    fn game(appid: &str, name: &str) -> Game {
        Game { appid: appid.into(), name: name.into(), installdir: name.into(), library: PathBuf::new(), root: Root { path: PathBuf::new(), flatpak: false } }
    }

    #[test]
    fn games_are_found_by_appid_exact_name_or_unique_substring() {
        let games = [game("3240220", "Grand Theft Auto V Enhanced"), game("1547000", "Grand Theft Auto: San Andreas - The Definitive Edition"), game("990080", "Hogwarts Legacy"), game("1", "Remnant"), game("2", "Remnant II")];
        assert_eq!(find(&games, "3240220").unwrap().appid, "3240220");
        assert_eq!(find(&games, "hogwarts").unwrap().appid, "990080");
        assert_eq!(find(&games, "GRAND THEFT AUTO V ENHANCED").unwrap().appid, "3240220");
        assert_eq!(find(&games, "remnant").unwrap().appid, "1", "an exact name wins over substrings");
        match find(&games, "grand theft") {
            Err(FindError::Ambiguous(c)) => assert_eq!(c.len(), 2),
            other => panic!("{other:?}"),
        }
        assert_eq!(find(&games, "doom"), Err(FindError::NotFound));
        assert_eq!(find(&games, ""), Err(FindError::NotFound));
    }

    #[test]
    fn steam_running_reads_comm_names() {
        let proc = Scratch::new("proc");
        assert!(!steam_running(&proc.0));
        write(&proc.0.join("12/comm"), "bash\n");
        write(&proc.0.join("self/comm"), "steam\n"); // not a pid directory
        write(&proc.0.join("13/comm"), "steam-runtime-l\n");
        assert!(!steam_running(&proc.0));
        write(&proc.0.join("14/comm"), "steamwebhelper\n");
        assert!(steam_running(&proc.0));
        std::fs::remove_file(proc.0.join("14/comm")).unwrap();
        write(&proc.0.join("15/comm"), "steam\n");
        assert!(steam_running(&proc.0));
    }

    #[test]
    fn enable_splices_one_value_and_disable_restores_the_file() {
        let home = Scratch::new("roundtrip");
        let lc = home.0.join("userdata/128944717/config/localconfig.vdf");
        let original = localconfig(&[("3240220", Some("MANGOHUD=1 PROTON_ENABLE_WAYLAND=1 %command% -dx12 -skipintro")), ("990080", None)]);
        write(&lc, &original);
        let Outcome::Changed(edits) = apply(&home.0, "3240220", &enable(), false) else { panic!() };
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].after, "NEURAL_FORGE_ENABLE=1 MANGOHUD=1 PROTON_ENABLE_WAYLAND=1 %command% -dx12 -skipintro");
        let after = std::fs::read_to_string(&lc).unwrap();
        assert_eq!(after, original.replace("\"MANGOHUD=1", "\"NEURAL_FORGE_ENABLE=1 MANGOHUD=1"), "byte-identical except the one value");
        assert!(enabled(&home.0, "3240220") && !enabled(&home.0, "990080"));
        assert_eq!(apply(&home.0, "3240220", &enable(), false), Outcome::Unchanged);
        assert!(matches!(apply(&home.0, "3240220", &Action::Disable, false), Outcome::Changed(_)));
        assert_eq!(std::fs::read_to_string(&lc).unwrap(), original);
        assert_eq!(apply(&home.0, "3240220", &Action::Disable, false), Outcome::Unchanged);
    }

    #[test]
    fn backups_keep_the_first_original_and_the_last_state() {
        let home = Scratch::new("backups");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        let original = localconfig(&[("3240220", Some("gamemoderun %command%"))]);
        write(&lc, &original);
        std::fs::set_permissions(&lc, std::fs::Permissions::from_mode(0o775)).unwrap();
        let Outcome::Changed(edits) = apply(&home.0, "3240220", &enable(), false) else { panic!() };
        let orig = home.0.join("userdata/1/config/localconfig.vdf.neural-forge.orig");
        let bak = home.0.join("userdata/1/config/localconfig.vdf.neural-forge.bak");
        assert_eq!(edits[0].backup, bak);
        assert_eq!(std::fs::read_to_string(&orig).unwrap(), original);
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), original);
        let enabled_text = std::fs::read_to_string(&lc).unwrap();
        assert_eq!(std::fs::metadata(&lc).unwrap().permissions().mode() & 0o7777, 0o775, "the file keeps its mode");
        apply(&home.0, "3240220", &Action::Disable, false);
        assert_eq!(std::fs::read_to_string(&orig).unwrap(), original, ".orig is taken once");
        assert_eq!(std::fs::read_to_string(&bak).unwrap(), enabled_text, ".bak is the state before the last edit");
        let leftovers: Vec<_> = std::fs::read_dir(lc.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).filter(|n| n.to_string_lossy().ends_with(".tmp")).collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn steam_running_refuses_and_offers_the_string() {
        let home = Scratch::new("running");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        let original = localconfig(&[("3240220", Some("gamemoderun %command% -x"))]);
        write(&lc, &original);
        assert_eq!(apply(&home.0, "3240220", &enable(), true), Outcome::SteamRunning { manual: "NEURAL_FORGE_ENABLE=1 gamemoderun %command% -x".into() });
        assert_eq!(std::fs::read_to_string(&lc).unwrap(), original);
        assert!(!home.0.join("userdata/1/config/localconfig.vdf.neural-forge.bak").exists());
    }

    #[test]
    fn a_missing_launch_options_key_is_inserted_and_removed_cleanly() {
        let home = Scratch::new("insertkey");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        let original = localconfig(&[("990080", None)]);
        write(&lc, &original);
        assert!(matches!(apply(&home.0, "990080", &Action::Enable { target_exe: "HogwartsLegacy.exe".into() }, false), Outcome::Changed(_)));
        assert_eq!(options_in(&lc, "990080").as_deref(), Some("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=HogwartsLegacy.exe %command%"));
        let text = std::fs::read_to_string(&lc).unwrap();
        assert!(text.contains("\t\t\t\t\t\t\"Playtime\"\t\t\"6916\"\n\t\t\t\t\t\t\"LaunchOptions\"\t\t\"NEURAL_FORGE_ENABLE=1"), "{text}");
        assert_eq!(text.len(), original.len() + "\t\t\t\t\t\t\"LaunchOptions\"\t\t\"NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=HogwartsLegacy.exe %command%\"\n".len());
        // Disabling leaves the inserted key, empty: Steam's own "no launch options".
        apply(&home.0, "990080", &Action::Disable, false);
        assert_eq!(options_in(&lc, "990080").as_deref(), Some(""));
    }

    #[test]
    fn two_accounts_the_ones_that_know_the_game_are_edited() {
        let home = Scratch::new("accounts");
        let older = home.0.join("userdata/111/config/localconfig.vdf");
        let newer = home.0.join("userdata/222/config/localconfig.vdf");
        write(&older, &localconfig(&[("3240220", Some("%command%"))]));
        write(&newer, &localconfig(&[("990080", None)]));
        let set_mtime = |p: &Path, secs: u64| std::fs::File::options().write(true).open(p).unwrap().set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs)).unwrap();
        set_mtime(&older, 1_000);
        set_mtime(&newer, 2_000);
        // Only the older account has GTA: only its file changes.
        let Outcome::Changed(edits) = apply(&home.0, "3240220", &enable(), false) else { panic!() };
        assert_eq!(edits.iter().map(|e| &e.file).collect::<Vec<_>>(), [&older]);
        assert_eq!(options_in(&newer, "3240220"), None);
        // Neither has this game: the block is created in the most recently modified file only
        // (the edit above made the older one newest; put the times back).
        set_mtime(&older, 1_000);
        let Outcome::Changed(edits) = apply(&home.0, "1593500", &enable(), false) else { panic!() };
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].file, newer);
        assert_eq!(options_in(&newer, "1593500").as_deref(), Some("NEURAL_FORGE_ENABLE=1 %command%"));
        assert_eq!(options_in(&older, "1593500"), None);
        // Both have it now: both are edited, and disabling a game no account has is a no-op.
        write(&older, &localconfig(&[("1593500", Some("MANGOHUD=1 %command%"))]));
        let Outcome::Changed(edits) = apply(&home.0, "1593500", &Action::Disable, false) else { panic!() };
        assert_eq!(edits.len(), 1, "only the account that had it enabled changes");
        assert_eq!(apply(&home.0, "4242", &Action::Disable, false), Outcome::Unchanged);
    }

    #[test]
    fn crlf_and_tabs_survive_an_edit() {
        let home = Scratch::new("crlf");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        let original = localconfig(&[("3240220", Some("PROTON_LOG=1 %command%")), ("990080", None)]).replace('\n', "\r\n");
        write(&lc, &original);
        apply(&home.0, "3240220", &enable(), false);
        assert_eq!(std::fs::read_to_string(&lc).unwrap(), original.replace("\"PROTON_LOG=1", "\"NEURAL_FORGE_ENABLE=1 PROTON_LOG=1"));
        apply(&home.0, "990080", &enable(), false);
        apply(&home.0, "1593500", &enable(), false);
        let text = std::fs::read_to_string(&lc).unwrap();
        assert!(!text.replace("\r\n", "").contains('\n'), "only CRLF line endings: {text:?}");
        assert_eq!(options_in(&lc, "1593500").as_deref(), Some("NEURAL_FORGE_ENABLE=1 %command%"));
    }

    #[test]
    fn quoted_options_are_escaped_in_the_file_and_read_back() {
        let home = Scratch::new("quotes");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        let existing = "WINEDLLOVERRIDES=\"dxgi=n,b\" DXVK_CONFIG_FILE='/home/a/My Games/dxvk.conf' %command%";
        write(&lc, &localconfig(&[("3240220", Some(existing))]));
        apply(&home.0, "3240220", &Action::Enable { target_exe: "My Game.exe".into() }, false);
        let expected = format!("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='My Game.exe' {existing}");
        assert_eq!(options_in(&lc, "3240220").as_deref(), Some(expected.as_str()));
        assert!(std::fs::read_to_string(&lc).unwrap().contains("WINEDLLOVERRIDES=\\\"dxgi=n,b\\\""));
        apply(&home.0, "3240220", &Action::Disable, false);
        assert_eq!(options_in(&lc, "3240220").as_deref(), Some(existing));
    }

    #[test]
    fn an_unsafe_edit_is_refused_and_nothing_is_written() {
        let home = Scratch::new("refuse");
        let lc = home.0.join("userdata/1/config/localconfig.vdf");
        // No Software > Valve > Steam > apps: nothing to insert into.
        let text = "\"UserLocalConfigStore\"\n{\n\t\"friends\"\n\t{\n\t}\n}\n";
        write(&lc, text);
        match apply(&home.0, "3240220", &enable(), false) {
            Outcome::Failed { manual, .. } => assert_eq!(manual, "NEURAL_FORGE_ENABLE=1 %command%"),
            other => panic!("{other:?}"),
        }
        assert_eq!(std::fs::read_to_string(&lc).unwrap(), text);
        assert!(!home.0.join("userdata/1/config/localconfig.vdf.neural-forge.orig").exists());
        // A second account whose file does not parse stops the edit of the first too.
        write(&home.0.join("userdata/2/config/localconfig.vdf"), &localconfig(&[("3240220", Some("%command%"))]));
        write(&lc, "\"broken\" {");
        assert!(matches!(apply(&home.0, "3240220", &enable(), false), Outcome::Failed { .. }));
        assert_eq!(options_in(&home.0.join("userdata/2/config/localconfig.vdf"), "3240220").as_deref(), Some("%command%"));
        // No account at all.
        assert!(matches!(apply(&home.0.join("nowhere"), "3240220", &enable(), false), Outcome::Failed { .. }));
    }
}
