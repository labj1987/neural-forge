//! The NGX file versions in `doctor` and the report: NVIDIA's core, the DLSS snippets a game loads (`nvngx_dlss.dll`
//! Super Resolution, `nvngx_dlssg.dll` Frame Generation, `nvngx_dlssd.dll` Ray Reconstruction, `nvngx_dlssnr.dll`
//! Neural Rendering, ...), the wrapper. They come from two places, so over-the-air updates, a game's own update and
//! a DLL the user replaced all show in a bug report:
//!
//! - what the game loaded in its last run: the NGX wrapper (`crates/ngx-wrapper/src/modules.rs`, which has the file
//!   format) writes `$XDG_STATE_HOME/neural-forge/ngx-modules/<exe>.txt` after a feature is created, with every loaded
//!   NGX module's path and file version, wherever it was loaded from; [`parse_record`] reads it;
//! - the files present: every `nvngx*.dll` in each switched-on game's folder and in its prefix's NGX folder
//!   (`ProgramData/NVIDIA/NGX`, where over-the-air updates land) and `system32`, and the driver's core, each with its
//!   version read from the PE resources ([`crate::model::dll_file_version`]; no code of the DLL runs).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The most directories levels below a game's folder looked in (Cyberpunk keeps its DLLs in `bin/x64`).
const GAME_DEPTH: usize = 4;
/// The same below the prefix's NGX folder (`models/<feature>/versions/<n>/files`).
const NGX_DEPTH: usize = 10;
/// The most entries looked at per folder tree: a game folder can hold hundreds of thousands of files.
const ENTRY_CAP: usize = 50_000;
/// The most DLLs listed per game and place; the rest are counted.
const LIST_CAP: usize = 40;

/// Whether `file_name` is one of NGX's: `_nvngx.dll`, or `nvngx*.dll` (the wrapper's own `is_ngx_module`).
pub fn is_ngx_dll(file_name: &str) -> bool {
    let name = file_name.to_ascii_lowercase();
    name == "_nvngx.dll" || (name.starts_with("nvngx") && name.ends_with(".dll"))
}

/// The NGX DLLs found in a tree.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Found {
    /// Sorted.
    pub dlls: Vec<PathBuf>,
    /// The entry cap ([`ENTRY_CAP`]) was reached before the whole tree was looked at: more may exist deeper down.
    pub capped: bool,
}

/// The NGX DLLs in the tree under `dir`, at most `depth` levels down, sorted. The tree is walked level by level, so
/// when the entry cap stops it the shallow DLLs (the usual place) are already found. Symbolic links to DLL files
/// count (a game folder may link them); symbolic links to folders are not followed.
pub fn find_dlls(dir: &Path, depth: usize) -> Found {
    find_dlls_capped(dir, depth, ENTRY_CAP)
}

fn find_dlls_capped(dir: &Path, depth: usize, cap: usize) -> Found {
    let mut found = Found::default();
    let mut seen = 0usize;
    let mut queue = std::collections::VecDeque::from([(dir.to_path_buf(), 0usize)]);
    'walk: while let Some((d, level)) = queue.pop_front() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            seen += 1;
            if seen > cap {
                found.capped = true;
                break 'walk;
            }
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() && level < depth {
                queue.push_back((entry.path(), level + 1));
            } else if is_ngx_dll(&entry.file_name().to_string_lossy()) && (kind.is_file() || (kind.is_symlink() && entry.path().is_file())) {
                found.dlls.push(entry.path());
            }
        }
    }
    found.dlls.sort();
    found
}

/// One module of the wrapper's record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded {
    /// `wrapper`, `core` or `snippet`.
    pub role: String,
    pub file: String,
    /// `None` when the wrapper could not read one.
    pub version: Option<String>,
    pub path: String,
}

/// What the wrapper wrote for one game's last run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub wrapper: String,
    pub exe: String,
    pub modules: Vec<Recorded>,
}

/// The wrapper's file; `None` for text of another format.
pub fn parse_record(text: &str) -> Option<Record> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "neural-forge-ngx-modules 1" {
        return None;
    }
    let (mut wrapper, mut exe, mut modules) = (String::new(), String::new(), Vec::new());
    for line in lines {
        if let Some(v) = line.strip_prefix("wrapper=") {
            wrapper = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("exe=") {
            exe = v.trim().to_string();
        } else {
            let f: Vec<&str> = line.splitn(4, '\t').collect();
            if let [role, file, version, path] = f[..] {
                modules.push(Recorded {
                    role: role.into(),
                    file: file.into(),
                    version: (version != "-").then(|| version.to_string()),
                    path: path.trim_end().into(),
                });
            }
        }
    }
    Some(Record { wrapper, exe, modules })
}

/// The records in `<state_dir>/ngx-modules`, by file name, with how long ago each was written.
pub fn read_records(state_dir: &Path, now: SystemTime) -> Vec<(String, Option<Duration>, Record)> {
    let dir = state_dir.join("ngx-modules");
    let mut names: Vec<PathBuf> =
        std::fs::read_dir(&dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    names.sort();
    names
        .into_iter()
        .filter_map(|p| {
            let record = parse_record(&std::fs::read_to_string(&p).ok()?)?;
            let age = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|t| now.duration_since(t).ok());
            Some((p.file_name()?.to_string_lossy().into_owned(), age, record))
        })
        .collect()
}

fn ago(age: Option<Duration>) -> String {
    match age.map(|a| a.as_secs()) {
        None => "at an unknown time".into(),
        Some(s) if s < 120 => "just now".into(),
        Some(s) if s < 2 * 3600 => format!("{} min ago", s / 60),
        Some(s) if s < 2 * 86400 => format!("{} h ago", s / 3600),
        Some(s) => format!("{} days ago", s / 86400),
    }
}

/// One game's files for the report.
pub struct GameFiles<'a> {
    pub name: &'a str,
    pub appid: &'a str,
    pub game_dir: &'a Path,
    /// `<library>/steamapps/compatdata/<appid>/pfx/drive_c`.
    pub drive_c: &'a Path,
}

fn version_of(path: &Path) -> String {
    match crate::model::dll_file_version(path) {
        Ok(v) => format!("{}.{}.{}.{}", v[0], v[1], v[2], v[3]),
        Err(e) => format!("version unreadable ({e})"),
    }
}

fn list(out: &mut String, label: &str, base: &Path, found: &Found) {
    let dlls = &found.dlls;
    if dlls.is_empty() {
        out.push_str(&format!("  {label}: no NGX DLLs\n"));
    }
    for dll in dlls.iter().take(LIST_CAP) {
        out.push_str(&format!("  {label}: {}  {}\n", dll.strip_prefix(base).unwrap_or(dll).display(), version_of(dll)));
    }
    if dlls.len() > LIST_CAP {
        out.push_str(&format!("  {label}: and {} more\n", dlls.len() - LIST_CAP));
    }
    if found.capped {
        out.push_str(&format!("  {label}: stopped after looking at {ENTRY_CAP} entries; there may be more NGX DLLs further down\n"));
    }
}

/// The section for `doctor` and `summary.txt`: the driver's core, the wrapper's record of each game's last run, and
/// the NGX files present for each of `games`.
pub fn render(driver_core: Option<&Path>, records: &[(String, Option<Duration>, Record)], games: &[GameFiles]) -> String {
    let mut out = String::new();
    match driver_core {
        Some(core) => out.push_str(&format!("driver core: {}  {}\n", core.display(), version_of(core))),
        None => out.push_str("driver core: not found\n"),
    }
    if records.is_empty() {
        out.push_str("loaded in a game: nothing recorded yet (the wrapper writes it after a game creates a DLSS feature with Neural Forge on)\n");
    }
    for (file, age, record) in records {
        out.push_str(&format!("loaded by {} ({}), {}, wrapper {}\n", record.exe, file, ago(*age), record.wrapper));
        for m in &record.modules {
            out.push_str(&format!("  {:<8} {:<18} {}  {}\n", m.role, m.file, m.version.as_deref().unwrap_or("no version"), m.path));
        }
    }
    for game in games {
        out.push_str(&format!("files of {} ({}):\n", game.name, game.appid));
        list(&mut out, "game folder", game.game_dir, &find_dlls(game.game_dir, GAME_DEPTH));
        let ngx = game.drive_c.join("ProgramData/NVIDIA/NGX");
        list(&mut out, "prefix NGX", &ngx, &find_dlls(&ngx, NGX_DEPTH));
        let system32 = game.drive_c.join("windows/system32");
        list(&mut out, "prefix system32", &system32, &find_dlls(&system32, 0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORD: &str = "neural-forge-ngx-modules 1\nwrapper=4.0.0\nexe=C:\\Game\\g.exe\n\
        wrapper\t_nvngx.dll\t4.0.0\tC:\\windows\\system32\\_nvngx.dll\n\
        core\t_nvngx.dll\t595.58.3.0\tZ:\\usr\\lib\\nvidia\\wine\\_nvngx.dll\n\
        snippet\tnvngx_dlssg.dll\t310.1.0.0\tC:\\Game\\nvngx_dlssg.dll\n\
        snippet\tnvngx_dlss.dll\t-\tC:\\Game\\nvngx_dlss.dll\n";

    #[test]
    fn matches_the_wrappers_module_names() {
        for name in ["_nvngx.dll", "nvngx.dll", "NVNGX_DLSSG.dll", "nvngx_dlssnr.dll"] {
            assert!(is_ngx_dll(name), "{name}");
        }
        for name in ["nvngx_dlss.dll.bak", "sl.dlss.dll", "nvngx", "x_nvngx.dll"] {
            assert!(!is_ngx_dll(name), "{name}");
        }
    }

    #[test]
    fn parses_the_wrappers_record() {
        let r = parse_record(RECORD).unwrap();
        assert_eq!((r.wrapper.as_str(), r.exe.as_str(), r.modules.len()), ("4.0.0", "C:\\Game\\g.exe", 4));
        assert_eq!(
            r.modules[2],
            Recorded { role: "snippet".into(), file: "nvngx_dlssg.dll".into(), version: Some("310.1.0.0".into()), path: "C:\\Game\\nvngx_dlssg.dll".into() }
        );
        assert_eq!(r.modules[3].version, None);
        assert_eq!(parse_record("something else\n"), None);
        assert_eq!(parse_record(""), None);
        // A damaged line is skipped, not fatal.
        assert_eq!(parse_record("neural-forge-ngx-modules 1\nwrapper=1\nbroken line\n").unwrap().modules.len(), 0);
    }

    #[test]
    fn ages_read_plainly() {
        assert_eq!(ago(None), "at an unknown time");
        assert_eq!(ago(Some(Duration::from_secs(30))), "just now");
        assert_eq!(ago(Some(Duration::from_secs(600))), "10 min ago");
        assert_eq!(ago(Some(Duration::from_secs(5 * 3600))), "5 h ago");
        assert_eq!(ago(Some(Duration::from_secs(5 * 86400))), "5 days ago");
    }

    fn scratch(tag: &str) -> crate::paths::tests::ScratchDir {
        crate::paths::tests::ScratchDir::new(&format!("ngx-versions-{tag}"))
    }

    /// The smallest PE32+ with a version resource is built by model.rs's tests; here the files are not PEs, so the
    /// listing shows the unreadable-version text, which is part of what is tested.
    #[test]
    fn finds_dlls_by_name_within_the_depth_and_renders_the_section() {
        let dir = scratch("render");
        let game = dir.join("game");
        for f in ["bin/x64/nvngx_dlss.dll", "nvngx_dlssg.dll", "readme.txt", "bin/x64/other.dll", "a/b/c/d/e/nvngx_dlssd.dll"] {
            let p = game.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"MZ").unwrap();
        }
        let found = find_dlls(&game, GAME_DEPTH).dlls;
        let names: Vec<String> = found.iter().map(|p| p.strip_prefix(&game).unwrap().display().to_string()).collect();
        assert_eq!(names, ["bin/x64/nvngx_dlss.dll", "nvngx_dlssg.dll"], "the one five levels down is out of reach");

        let drive_c = dir.join("pfx/drive_c");
        std::fs::create_dir_all(drive_c.join("windows/system32")).unwrap();
        std::fs::write(drive_c.join("windows/system32/_nvngx.dll"), b"MZ").unwrap();
        let games = [GameFiles { name: "Some Game", appid: "42", game_dir: &game, drive_c: &drive_c }];
        let records = vec![("g.txt".to_string(), Some(Duration::from_secs(3 * 3600)), parse_record(RECORD).unwrap())];
        let text = render(None, &records, &games);
        assert_eq!(
            text,
            "driver core: not found\n\
             loaded by C:\\Game\\g.exe (g.txt), 3 h ago, wrapper 4.0.0\n\
             \x20 wrapper  _nvngx.dll         4.0.0  C:\\windows\\system32\\_nvngx.dll\n\
             \x20 core     _nvngx.dll         595.58.3.0  Z:\\usr\\lib\\nvidia\\wine\\_nvngx.dll\n\
             \x20 snippet  nvngx_dlssg.dll    310.1.0.0  C:\\Game\\nvngx_dlssg.dll\n\
             \x20 snippet  nvngx_dlss.dll     no version  C:\\Game\\nvngx_dlss.dll\n\
             files of Some Game (42):\n\
             \x20 game folder: bin/x64/nvngx_dlss.dll  version unreadable (not a readable DLL: truncated DOS header)\n\
             \x20 game folder: nvngx_dlssg.dll  version unreadable (not a readable DLL: truncated DOS header)\n\
             \x20 prefix NGX: no NGX DLLs\n\
             \x20 prefix system32: _nvngx.dll  version unreadable (not a readable DLL: truncated DOS header)\n"
        );
        assert!(render(None, &[], &[]).contains("nothing recorded yet"));
    }

    #[test]
    fn walks_level_by_level_and_says_when_the_cap_stopped_it() {
        let dir = scratch("cap");
        let game = dir.join("game");
        // A deep, wide tree listed first alphabetically, and the DLL at the top: a depth-first walk would spend the cap in `a`.
        for i in 0..30 {
            let p = game.join(format!("a/d{i}/f.txt"));
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
        std::fs::write(game.join("nvngx_dlss.dll"), b"MZ").unwrap();
        let found = find_dlls_capped(&game, GAME_DEPTH, 10);
        assert!(found.capped);
        assert_eq!(found.dlls, [game.join("nvngx_dlss.dll")], "the shallow DLL is found first");
        assert!(!find_dlls_capped(&game, GAME_DEPTH, 1000).capped);
        let mut text = String::new();
        list(&mut text, "game folder", &game, &Found { dlls: Vec::new(), capped: true });
        assert!(text.contains("stopped after looking at 50000 entries"), "{text}");
    }

    #[test]
    fn follows_symlinked_dll_files_but_not_symlinked_folders() {
        let dir = scratch("links");
        let (game, elsewhere) = (dir.join("game"), dir.join("elsewhere"));
        std::fs::create_dir_all(&game).unwrap();
        std::fs::create_dir_all(elsewhere.join("sub")).unwrap();
        std::fs::write(elsewhere.join("nvngx_dlssg.dll"), b"MZ").unwrap();
        std::fs::write(elsewhere.join("sub/nvngx_dlssd.dll"), b"MZ").unwrap();
        std::os::unix::fs::symlink(elsewhere.join("nvngx_dlssg.dll"), game.join("nvngx_dlssg.dll")).unwrap();
        std::os::unix::fs::symlink(elsewhere.join("missing.dll"), game.join("nvngx_dlss.dll")).unwrap();
        std::os::unix::fs::symlink(elsewhere.join("sub"), game.join("linked")).unwrap();
        let names: Vec<String> = find_dlls(&game, GAME_DEPTH).dlls.iter().map(|p| p.strip_prefix(&game).unwrap().display().to_string()).collect();
        assert_eq!(names, ["nvngx_dlssg.dll"], "a dangling link and a linked folder are left out");
    }

    #[test]
    fn reads_records_from_the_state_dir() {
        let dir = scratch("records");
        let d = dir.join("ngx-modules");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("g.txt"), RECORD).unwrap();
        std::fs::write(d.join("bad.txt"), "nope").unwrap();
        std::fs::write(d.join("note.md"), RECORD).unwrap();
        let records = read_records(&dir, SystemTime::now());
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "g.txt");
        assert!(records[0].1.is_some());
    }
}
