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
    /// The installed beta branch (`MountedConfig/BetaKey` in its app manifest); empty for the
    /// default branch. Launch entries can be per branch.
    pub branch: String,
}

impl Game {
    /// Its install folder.
    pub fn dir(&self) -> PathBuf {
        self.library.join("steamapps/common").join(&self.installdir)
    }

    /// Why Neural Forge cannot be switched on for it, if it cannot: a game of the Flatpak Steam
    /// runs in a sandbox that does not see the layer installed on the host.
    pub fn unsupported(&self) -> Option<&'static str> {
        self.root.flatpak.then_some("it belongs to the Flatpak Steam, whose games cannot load Neural Forge's layer")
    }
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

/// The games in every library of every root (tools filtered out), sorted by name, each appid once:
/// a library two roots share is read once, and of an appid with app manifests in two places (a
/// stale copy in another library, or both a native and a Flatpak Steam) the copy kept is the native
/// root's, else the one whose folder exists, else the one Steam marks fully installed, else the
/// manifest written last.
pub fn games(roots: &[Root]) -> Vec<Game> {
    // Each copy with what makes it the right one: a native root, its folder there, Steam's
    // "fully installed" state flag, the manifest written last.
    let mut copies: Vec<(Game, (bool, bool, bool, std::time::SystemTime))> = Vec::new();
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
                if is_tool(&appid, &name) || copies.iter().any(|(g, _)| g.appid == appid && g.library == library) {
                    continue;
                }
                let branch = tree.string_at(&["AppState", "MountedConfig", "BetaKey"]).unwrap_or_default().trim().to_string();
                let installed = tree.string_at(&["AppState", "StateFlags"]).and_then(|f| f.trim().parse::<u32>().ok()).is_some_and(|f| f & 4 != 0);
                let written = std::fs::metadata(&manifest).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                let game = Game { appid, name, installdir, library: library.clone(), root: root.clone(), branch };
                let rank = (!root.flatpak, game.dir().is_dir(), installed, written);
                copies.push((game, rank));
            }
        }
    }
    let mut out: Vec<Game> = Vec::new();
    for (game, rank) in &copies {
        let best = copies.iter().filter(|(g, _)| g.appid == game.appid).map(|(_, r)| *r).max();
        if best == Some(*rank) && !out.iter().any(|g| g.appid == game.appid) {
            out.push(game.clone());
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

use neural_forge_protocol::processes::{game_helper as helper_exe, launcher_like};

fn helper_dir(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["redist", "_commonredist", "redistributables", "installers", "__installer", "directx", "vcredist", "battleye", "easyanticheat", "engine", "support", "crashreporter", "crashpad"].contains(&n.as_str())
}

/// The `.exe` files directly in `dir` (case kept), sorted.
fn exes_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.to_ascii_lowercase().ends_with(".exe"))
        .collect();
    out.sort();
    out
}

/// Unreal Engine's game binaries: `<Project>/Binaries/<Win64|WinGDK|Win32>/*.exe` under `dir`.
/// Steam usually starts a small stub in the install folder that starts this one.
fn unreal_binaries(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut projects: Vec<PathBuf> = entries.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).map(|e| e.path()).collect();
    projects.sort();
    let mut out = Vec::new();
    for project in projects {
        if project.file_name().and_then(|n| n.to_str()).is_some_and(helper_dir) {
            continue;
        }
        for platform in ["Win64", "WinGDK", "Win32"] {
            out.extend(exes_in(&project.join("Binaries").join(platform)));
        }
    }
    out
}

/// Whether `path` is an ELF executable (a native Linux game's binary).
fn elf_executable(path: &Path) -> bool {
    use std::io::Read;
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::metadata(path) else { return false };
    let mut magic = [0u8; 4];
    meta.is_file() && meta.permissions().mode() & 0o111 != 0 && std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && magic == *b"\x7fELF"
}

/// How deep [`TargetFinder`] searches a game's folder for executables.
const SEARCH_DEPTH: usize = 6;

/// One executable in a game's folder that its process could be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub name: String,
    /// Its path in the install folder, `/`-separated.
    pub path: String,
    pub bytes: u64,
    /// Part of the target found for the game ([`TargetFinder::target_exes`]).
    pub chosen: bool,
}

/// Finds the executables the game's process can be, for `NEURAL_FORGE_TARGET_EXE` (the layer lets
/// only these claim the channel). Steam's app cache is read once for all the games it is made for.
pub struct TargetFinder {
    launches: std::collections::HashMap<(PathBuf, String), Vec<crate::appinfo::Launch>>,
}

impl TargetFinder {
    /// Reads each root's `appcache/appinfo.vdf` once, for every game of `games` in that root.
    pub fn new(games: &[Game]) -> TargetFinder {
        let mut launches = std::collections::HashMap::new();
        let mut roots: Vec<&Path> = games.iter().map(|g| g.root.path.as_path()).collect();
        roots.sort();
        roots.dedup();
        for root in roots {
            let appids: Vec<&str> = games.iter().filter(|g| g.root.path == root).map(|g| g.appid.as_str()).collect();
            for (appid, l) in crate::appinfo::launches_for(root, &appids) {
                launches.insert((root.to_path_buf(), appid), l);
            }
        }
        TargetFinder { launches }
    }

    /// Every executable in the game's folder (up to [`SEARCH_DEPTH`] folders deep, leaving out
    /// redistributable installers, anti-cheat services and crash reporters and the folders that
    /// hold them), best first: the found target in its order ([`Self::target_exes`]), then the
    /// rest by size, launchers last. What the Setup tab offers to choose from.
    pub fn candidates(&self, game: &Game) -> Vec<Candidate> {
        let found = all_exes(&game.dir());
        let chosen = self.target_exes(game);
        let mut out: Vec<Candidate> = chosen
            .iter()
            .map(|name| {
                let (path, bytes) = found.iter().find(|(p, _)| file_name(p).eq_ignore_ascii_case(name)).cloned().unwrap_or((name.clone(), 0));
                Candidate { name: name.clone(), path, bytes, chosen: true }
            })
            .collect();
        let mut rest: Vec<&(String, u64)> = found.iter().filter(|(p, _)| !chosen.iter().any(|c| c.eq_ignore_ascii_case(file_name(p)))).collect();
        rest.sort_by_key(|(p, bytes)| (launcher_like(file_name(p)), std::cmp::Reverse(*bytes), p.to_ascii_lowercase()));
        for (path, bytes) in rest {
            if !out.iter().any(|c| c.name.eq_ignore_ascii_case(file_name(path))) {
                out.push(Candidate { name: file_name(path).to_string(), path: path.clone(), bytes: *bytes, chosen: false });
            }
        }
        out
    }

    /// The executables of `game` that can be the process the player sees. Names only, case kept,
    /// deduplicated ignoring case.
    ///
    /// From Steam's own launch entries for the game (`appinfo.vdf`, those for Windows or Linux and the
    /// installed branch): each entry's executable, unless it is a launcher (`launcher_like`) and
    /// anything else qualifies; the other executables beside it (a store bootstrapper such as
    /// `PlayGTAV.exe` starts the game from its own folder; anti-cheat's `start_protected_game.exe`
    /// its neighbour); and Unreal Engine's real binary under `<Project>/Binaries/Win64`. Without launch
    /// entries, the executables at the top of the install folder and Unreal's. Then, from the whole
    /// folder, every other executable at least a quarter the size of the largest one: a launcher in
    /// a folder of its own starts the game from another (ESO's `Launcher/Bethesda.net_Launcher.exe`
    /// starts `game/client/eso64.exe`, FINAL FANTASY XIV's `boot/ffxivboot.exe` `game/ffxiv_dx11.exe`).
    /// Redistributable installers, anti-cheat services, crash reporters and launchers are never
    /// added from the folder. A Linux launch script stands for the ELF executables beside it.
    pub fn target_exes(&self, game: &Game) -> Vec<String> {
        let dir = game.dir();
        let mut out: Vec<String> = Vec::new();
        let add = |name: &str, out: &mut Vec<String>| {
            if !name.is_empty() && !name.contains(',') && !out.iter().any(|n| n.eq_ignore_ascii_case(name)) {
                out.push(name.to_string());
            }
        };
        let launches = self.launches.get(&(game.root.path.clone(), game.appid.clone())).map(Vec::as_slice).unwrap_or_default();
        let mut launched = false;
        let mut launchers: Vec<String> = Vec::new();
        for launch in launches.iter().filter(|l| (l.for_os("windows") || l.for_os("linux")) && l.for_branch(&game.branch)) {
            let rel = launch.executable.replace('\\', "/");
            let rel = rel.trim_start_matches("./");
            let path = dir.join(rel);
            let name = rel.rsplit('/').next().unwrap_or(rel);
            let beside = path.parent().map(Path::to_path_buf).unwrap_or_else(|| dir.clone());
            launched = true;
            if name.to_ascii_lowercase().ends_with(".sh") {
                let Ok(entries) = std::fs::read_dir(&beside) else { continue };
                let mut elves: Vec<String> = entries.flatten().filter(|e| elf_executable(&e.path())).filter_map(|e| e.file_name().to_str().map(str::to_string)).collect();
                elves.sort();
                for elf in elves {
                    add(&elf, &mut out);
                }
                continue;
            }
            if launcher_like(name) {
                launchers.push(name.to_string());
            } else {
                add(name, &mut out);
            }
            if name.to_ascii_lowercase().ends_with(".exe") {
                for exe in exes_in(&beside).iter().filter(|e| !helper_exe(e) && !launcher_like(e)) {
                    add(exe, &mut out);
                }
            }
        }
        if !launched {
            for exe in exes_in(&dir).iter().filter(|e| !helper_exe(e)) {
                add(exe, &mut out);
            }
        }
        for exe in unreal_binaries(&dir).iter().filter(|e| !helper_exe(e)) {
            add(exe, &mut out);
        }
        // The rest of the folder: the large executables, largest first.
        let mut found: Vec<(String, u64)> = all_exes(&dir).into_iter().filter(|(p, _)| !launcher_like(file_name(p))).collect();
        found.sort_by_key(|(p, bytes)| (std::cmp::Reverse(*bytes), p.to_ascii_lowercase()));
        if let Some(&(_, largest)) = found.first().filter(|(_, b)| *b > 0) {
            for (path, bytes) in &found {
                if bytes.saturating_mul(4) >= largest {
                    add(file_name(path), &mut out);
                }
            }
        }
        // A launcher Steam starts directly stays only when nothing else qualifies (a game whose
        // executable is literally called a launcher).
        if out.is_empty() {
            for name in &launchers {
                add(name, &mut out);
            }
        }
        out
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Every `.exe` under `dir` (relative path, size), up to [`SEARCH_DEPTH`] folders deep, without the
/// helper folders and helper executables ([`helper_dir`], [`helper_exe`]); symlinks not followed.
fn all_exes(dir: &Path) -> Vec<(String, u64)> {
    fn walk(dir: &Path, rel: &str, depth: usize, out: &mut Vec<(String, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let Ok(kind) = e.file_type() else { continue };
            let Some(name) = e.file_name().to_str().map(str::to_string) else { continue };
            let path = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
            if kind.is_dir() && depth > 1 && !helper_dir(&name) {
                walk(&e.path(), &path, depth - 1, out);
            } else if kind.is_file() && name.to_ascii_lowercase().ends_with(".exe") && !helper_exe(&name) {
                out.push((path, e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, "", SEARCH_DEPTH, &mut out);
    out.sort();
    out
}

/// [`TargetFinder::target_exes`] for one game.
pub fn target_exes(game: &Game) -> Vec<String> {
    TargetFinder::new(std::slice::from_ref(game)).target_exes(game)
}

/// The games whose install folder holds an executable named `exe` (ignoring case), best first: a
/// game whose [`target_exes`] name it, else the one holding it least deep, searched up to four
/// folders deep (`Game/Binaries/Win64/Game-Win64-Shipping.exe` is four). Several games tied for
/// the best place are all returned, so a caller can say the name is ambiguous. The channel names a
/// running game by its executable; this is how the report finds its app id, and from it its Proton.
pub fn games_for_exe<'a>(games: &'a [Game], exe: &str) -> Vec<&'a Game> {
    const DEPTH: usize = 4;
    /// The depth (1: the folder itself) of the shallowest file named `exe` under `dir`.
    fn depth_of(dir: &Path, exe: &str, depth: usize) -> Option<usize> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs = Vec::new();
        for e in entries.flatten() {
            let Ok(kind) = e.file_type() else { continue };
            if kind.is_file() && e.file_name().to_str().is_some_and(|n| n.eq_ignore_ascii_case(exe)) {
                return Some(DEPTH - depth + 1);
            }
            if kind.is_dir() && depth > 1 {
                subdirs.push(e.path());
            }
        }
        subdirs.iter().filter_map(|d| depth_of(d, exe, depth - 1)).min()
    }
    let exe = exe.trim();
    if exe.is_empty() || exe.contains('/') {
        return Vec::new();
    }
    // Rank: 0 when the game's own targets name it, else 1 + the depth it was found at.
    let holding: Vec<(usize, &Game)> = games.iter().filter_map(|g| Some((depth_of(&g.dir(), exe, DEPTH)?, g))).collect();
    let finder = TargetFinder::new(&holding.iter().map(|(_, g)| (*g).clone()).collect::<Vec<_>>());
    let ranked: Vec<(usize, &Game)> =
        holding.into_iter().map(|(depth, g)| (if finder.target_exes(g).iter().any(|t| t.eq_ignore_ascii_case(exe)) { 0 } else { depth }, g)).collect();
    let Some(best) = ranked.iter().map(|(r, _)| *r).min() else { return Vec::new() };
    ranked.into_iter().filter(|(r, _)| *r == best).map(|(_, g)| g).collect()
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

/// The `config.ini` key of the target executable chosen by hand for `appid` (`neural-forge-cli
/// enable GAME --target-exe EXE`). It wins over [`target_exes`] whenever the game is switched on,
/// from the command line or the GUI, so it survives switching the game off and on again.
pub fn chosen_target_key(appid: &str) -> String {
    format!("target_exe_{appid}")
}

/// The `NEURAL_FORGE_TARGET_EXE` value for `game`: the executable chosen by hand for it in `cfg`,
/// else every name [`target_exes`] finds, comma-separated. Empty only when neither has one.
pub fn target_for(cfg: &crate::Config, game: &Game) -> String {
    target_for_in(cfg, game, &TargetFinder::new(std::slice::from_ref(game)))
}

/// [`target_for`] with a [`TargetFinder`] made once for many games.
pub fn target_for_in(cfg: &crate::Config, game: &Game, finder: &TargetFinder) -> String {
    chosen_target(cfg, &game.appid).unwrap_or_else(|| finder.target_exes(game).join(","))
}

/// The target chosen by hand for `appid`, if any.
pub fn chosen_target(cfg: &crate::Config, appid: &str) -> Option<String> {
    cfg.settings.get(&chosen_target_key(appid)).map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Turn Neural Forge on for the target executable(s) (comma-separated; never empty: [`apply`]
    /// refuses an empty one, since without it a store launcher could take the channel).
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
/// run it), and when enabling also the most recently modified file (the account last signed in),
/// which gets the block if it lacks it; disabling with no account having the block has nothing to
/// do. Every edit is prepared and verified
/// before any file is written, so a refusal leaves all of them untouched. For each file written:
/// `localconfig.vdf.neural-forge.orig` is a copy of it as first found (taken once, never
/// replaced), `localconfig.vdf.neural-forge.bak` a copy from just before this edit, and the new
/// file replaces the old one atomically with the old one's permissions.
pub fn apply(root: &Path, appid: &str, action: &Action, steam_running: bool) -> Outcome {
    let accounts = launch_options(root, appid);
    let current = accounts.iter().find(|a| a.has_app).and_then(|a| a.options.clone()).unwrap_or_default();
    let manual = action.apply_to(&current);
    if matches!(action, Action::Enable { target_exe } if target_exe.trim().is_empty()) {
        return Outcome::Failed { reason: "the game's executable is not known: name it (the .exe the game runs) as the target".into(), manual };
    }
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
    if *action != Action::Disable {
        // The account last signed in (its file written last) gets the game's block too: it is the
        // one that will start the game next, whether or not it has run it before.
        let newest = accounts.iter().max_by_key(|a| std::fs::metadata(&a.file).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH));
        if let Some(newest) = newest.filter(|n| !n.has_app) {
            targets.push(newest);
        }
    } else if targets.is_empty() {
        return Outcome::Unchanged;
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
        let game = |appid: &str, name: &str| Game { appid: appid.into(), name: name.into(), installdir: name.into(), library: s.0.clone(), root: Root { path: s.0.clone(), flatpak: false }, branch: String::new() };
        let games = vec![game("1", "Alpha"), game("2", "Remnant2"), game("3", "Deep"), game("4", "Bundle"), game("5", "Twin")];
        write(&s.0.join("steamapps/common/Alpha/alpha.exe"), "");
        write(&s.0.join("steamapps/common/Remnant2/Remnant2/Binaries/Win64/Remnant2-Win64-Shipping.exe"), "");
        write(&s.0.join("steamapps/common/Deep/a/b/c/d/e/deep.exe"), "");
        // A bundle carrying another game's executable deep inside: the game whose own targets name it wins.
        write(&s.0.join("steamapps/common/Bundle/extras/alpha/alpha.exe"), "");
        write(&s.0.join("steamapps/common/Twin/twin.exe"), "");
        write(&s.0.join("steamapps/common/Alpha/twin.exe"), "");
        let appids = |exe: &str| games_for_exe(&games, exe).iter().map(|g| g.appid.as_str()).collect::<Vec<_>>();
        assert_eq!(appids("ALPHA.EXE"), ["1"], "case is ignored; the game whose target it is wins");
        assert_eq!(appids("Remnant2-Win64-Shipping.exe"), ["2"]);
        assert_eq!(appids("twin.exe"), ["1", "5"], "both games' own targets: ambiguous");
        assert!(appids("deep.exe").is_empty(), "deeper than four folders is not searched");
        assert!(appids("missing.exe").is_empty());
        assert!(appids("").is_empty());
        assert!(appids("Alpha/alpha.exe").is_empty(), "a name, not a path");
    }

    /// The layouts of real games (their files and Steam's launch entries as on a real install): the
    /// target is the executable the game's process really is.
    #[test]
    fn the_target_is_the_executable_the_game_really_runs() {
        let s = Scratch::new("targets");
        let root = Root { path: s.0.clone(), flatpak: false };
        let game = |appid: &str, dir: &str, branch: &str| Game { appid: appid.into(), name: dir.into(), installdir: dir.into(), library: s.0.clone(), root: root.clone(), branch: branch.into() };
        let files = |dir: &str, names: &[&str]| {
            for name in names {
                write(&s.0.join("steamapps/common").join(dir).join(name), "");
            }
        };
        // A store bootstrapper starting the game beside it; the store's own launcher lives elsewhere.
        files("Grand Theft Auto V Enhanced", &["PlayGTAV.exe", "GTA5_Enhanced.exe", "GTA5_Enhanced_BE.exe", "BattlEye/BEService_x64.exe", "Redistributables/Rockstar-Games-Launcher.exe", "Redistributables/Social-Club-Setup.exe"]);
        // Unreal: a stub in the folder starts <Project>/Binaries/Win64/<Name>-Win64-Shipping.exe.
        files("Remnant2", &["Remnant2.exe", "Remnant2/Binaries/Win64/Remnant2-Win64-Shipping.exe", "Engine/Binaries/Win64/CrashReportClient.exe"]);
        // Launch entries per beta branch.
        files("Hogwarts Legacy", &["HogwartsLegacy.exe", "Phoenix/Binaries/Win64/HogwartsLegacy.exe", "Engine/Binaries/Win64/CrashReportClient.exe"]);
        // Anti-cheat's bootstrapper starts its neighbour.
        files("ELDEN RING", &["Game/start_protected_game.exe", "Game/eldenring.exe", "Game/EasyAntiCheat/EasyAntiCheat_EOS_Setup.exe"]);
        // A subfolder launch with a crash handler beside it.
        files("Crimson Desert", &["bin64/CrimsonDesert.exe", "bin64/crashpad_handler.exe"]);
        // A game whose executable is literally Launcher.exe.
        files("Odd", &["Launcher.exe", "UnityCrashHandler64.exe"]);
        // Not in Steam's cache: the folder's own executables.
        files("Uncached", &["Game.exe", "vc_redist.x64.exe", "unins000.exe"]);
        // Native Linux behind a launch script.
        files("Native", &["start.sh", "data.pak"]);
        let elf = s.0.join("steamapps/common/Native/game.x86_64");
        std::fs::write(&elf, b"\x7fELF....").unwrap();
        std::fs::set_permissions(&elf, std::fs::Permissions::from_mode(0o755)).unwrap();
        let data = crate::appinfo::tests::image(&[
            (3240220, &[("PlayGTAV.exe", "windows", "")]),
            (1282100, &[("Remnant2.exe", "windows", "")]),
            (990080, &[("Phoenix-Win64-Test.exe", "windows", "development"), ("HogwartsLegacy.exe", "windows", "")]),
            (1245620, &[("Game\\start_protected_game.exe", "windows", "")]),
            (3321460, &[("bin64/CrimsonDesert.exe", "windows", ""), ("CrimsonDesert_Steam.app", "macos", "")]),
            (77, &[("Launcher.exe", "", "")]),
            (88, &[("start.sh", "linux", "")]),
        ]);
        write(&s.0.join("appcache/appinfo.vdf"), "");
        std::fs::write(s.0.join("appcache/appinfo.vdf"), data).unwrap();
        assert_eq!(target_exes(&game("3240220", "Grand Theft Auto V Enhanced", "")), ["PlayGTAV.exe", "GTA5_Enhanced.exe", "GTA5_Enhanced_BE.exe"]);
        assert_eq!(target_exes(&game("1282100", "Remnant2", "")), ["Remnant2.exe", "Remnant2-Win64-Shipping.exe"]);
        assert_eq!(target_exes(&game("990080", "Hogwarts Legacy", "")), ["HogwartsLegacy.exe"]);
        assert_eq!(target_exes(&game("990080", "Hogwarts Legacy", "development")), ["Phoenix-Win64-Test.exe", "HogwartsLegacy.exe"]);
        assert_eq!(target_exes(&game("1245620", "ELDEN RING", "")), ["start_protected_game.exe", "eldenring.exe"]);
        assert_eq!(target_exes(&game("3321460", "Crimson Desert", "")), ["CrimsonDesert.exe"]);
        assert_eq!(target_exes(&game("77", "Odd", "")), ["Launcher.exe"]);
        assert_eq!(target_exes(&game("99", "Uncached", "")), ["Game.exe"]);
        assert_eq!(target_exes(&game("88", "Native", "")), ["game.x86_64"]);
        assert!(target_exes(&game("98", "Missing", "")).is_empty());
        // A launcher in a folder of its own starting the game from another: the whole folder is
        // searched, and the large executables there are the target, the launcher never.
        let sized = |dir: &str, name: &str, mb: u64| {
            let path = s.0.join("steamapps/common").join(dir).join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(mb << 20).unwrap();
        };
        sized("Zenimax Online", "Launcher/Bethesda.net_Launcher.exe", 60);
        sized("Zenimax Online", "Launcher/BethesdaNetUpdater.exe", 2);
        sized("Zenimax Online", "Launcher/CrashReporter.exe", 3);
        sized("Zenimax Online", "The Elder Scrolls Online/game/client/eso64.exe", 90);
        sized("Zenimax Online", "The Elder Scrolls Online/game/client/eso.exe", 70);
        sized("Zenimax Online", "The Elder Scrolls Online/game/client/tools.exe", 1);
        sized("FFXIV", "boot/ffxivboot.exe", 1);
        sized("FFXIV", "boot/ffxivboot64.exe", 1);
        sized("FFXIV", "boot/ffxivlauncher64.exe", 20);
        sized("FFXIV", "game/ffxiv_dx11.exe", 50);
        sized("FFXIV", "_CommonRedist/vcredist_x64.exe", 100);
        let more = crate::appinfo::tests::image(&[
            (3240220, &[("PlayGTAV.exe", "windows", "")]),
            (306130, &[("Launcher\\Bethesda.net_Launcher.exe", "windows", "")]),
            (39210, &[("boot/ffxivboot.exe", "windows", "")]),
        ]);
        std::fs::write(s.0.join("appcache/appinfo.vdf"), more).unwrap();
        let eso = game("306130", "Zenimax Online", "");
        assert_eq!(target_exes(&eso), ["eso64.exe", "eso.exe"]);
        assert_eq!(target_exes(&game("39210", "FFXIV", "")), ["ffxiv_dx11.exe"]);
        let finder = TargetFinder::new(std::slice::from_ref(&eso));
        let names: Vec<(String, bool)> = finder.candidates(&eso).into_iter().map(|c| (c.name, c.chosen)).collect();
        assert_eq!(
            names,
            [("eso64.exe".into(), true), ("eso.exe".into(), true), ("tools.exe".into(), false), ("Bethesda.net_Launcher.exe".into(), false), ("BethesdaNetUpdater.exe".into(), false)],
            "the found target first, the rest by size, launchers last"
        );
        // A target chosen by hand wins, and survives switching off and on (it lives in config.ini).
        let gta = game("3240220", "Grand Theft Auto V Enhanced", "");
        let mut cfg = crate::Config::default();
        assert_eq!(target_for(&cfg, &gta), "PlayGTAV.exe,GTA5_Enhanced.exe,GTA5_Enhanced_BE.exe");
        cfg.settings.insert(chosen_target_key("3240220"), " GTA5_Enhanced.exe ".into());
        assert_eq!(target_for(&cfg, &gta), "GTA5_Enhanced.exe");
        assert_eq!(launch_options::launch_option(&target_for(&cfg, &gta)), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%");
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
        Action::Enable { target_exe: "G.exe".into() }
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
    fn of_two_manifests_for_one_appid_the_installed_copy_is_kept() {
        let home = Scratch::new("stale");
        let root = home.0.join("Steam");
        let other = home.0.join("Other");
        std::fs::create_dir_all(other.join("steamapps")).unwrap();
        write(
            &root.join("steamapps/libraryfolders.vdf"),
            &format!("\"libraryfolders\"\n{{\n\t\"1\"\n\t{{\n\t\t\"path\"\t\t\"{}\"\n\t}}\n}}\n", other.display()),
        );
        // A stale manifest in the root library (no folder), the real one in the other library.
        write(&root.join("steamapps/appmanifest_990080.acf"), &manifest("990080", "Hogwarts Legacy"));
        write(&other.join("steamapps/appmanifest_990080.acf"), &manifest("990080", "Hogwarts Legacy"));
        std::fs::create_dir_all(other.join("steamapps/common/Hogwarts Legacy")).unwrap();
        let games = games(&[Root { path: root.clone(), flatpak: false }]);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].library, other.canonicalize().unwrap());
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
        Game { appid: appid.into(), name: name.into(), installdir: name.into(), library: PathBuf::new(), root: Root { path: PathBuf::new(), flatpak: false }, branch: String::new() }
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
        assert_eq!(edits[0].after, "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe MANGOHUD=1 PROTON_ENABLE_WAYLAND=1 %command% -dx12 -skipintro");
        let after = std::fs::read_to_string(&lc).unwrap();
        assert_eq!(after, original.replace("\"MANGOHUD=1", "\"NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe MANGOHUD=1"), "byte-identical except the one value");
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
        assert_eq!(apply(&home.0, "3240220", &enable(), true), Outcome::SteamRunning { manual: "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe gamemoderun %command% -x".into() });
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
        // Only the older account has GTA: its file changes, and the account signed in last (the
        // newer file) gets the game's block too, since it is the one that starts the game next.
        let Outcome::Changed(edits) = apply(&home.0, "3240220", &enable(), false) else { panic!() };
        assert_eq!(edits.iter().map(|e| &e.file).collect::<Vec<_>>(), [&older, &newer]);
        assert_eq!(options_in(&newer, "3240220").as_deref(), Some("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe %command%"));
        assert!(matches!(apply(&home.0, "3240220", &Action::Enable { target_exe: " ".into() }, false), Outcome::Failed { .. }), "no target, no edit");
        // Neither has this game: the block is created in the most recently modified file only
        // (the edit above made both newer; put the times back).
        set_mtime(&older, 1_000);
        set_mtime(&newer, 2_000);
        let Outcome::Changed(edits) = apply(&home.0, "1593500", &enable(), false) else { panic!() };
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].file, newer);
        assert_eq!(options_in(&newer, "1593500").as_deref(), Some("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe %command%"));
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
        assert_eq!(std::fs::read_to_string(&lc).unwrap(), original.replace("\"PROTON_LOG=1", "\"NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe PROTON_LOG=1"));
        apply(&home.0, "990080", &enable(), false);
        apply(&home.0, "1593500", &enable(), false);
        let text = std::fs::read_to_string(&lc).unwrap();
        assert!(!text.replace("\r\n", "").contains('\n'), "only CRLF line endings: {text:?}");
        assert_eq!(options_in(&lc, "1593500").as_deref(), Some("NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe %command%"));
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
            Outcome::Failed { manual, .. } => assert_eq!(manual, "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=G.exe %command%"),
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
