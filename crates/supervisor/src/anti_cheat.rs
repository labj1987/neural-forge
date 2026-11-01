//! Anti-cheat a game ships in its install folder, by the files and folders it leaves there.
//!
//! Neural Forge does nothing to hide its Vulkan layer. An anti-cheat can refuse to start the game
//! with the layer loaded, or count it as tampering and ban the account, so switching such a game on
//! asks first (the CLI's `enable --anti-cheat-ok`, the GUI's confirmation) and the doctor warns about
//! one that is on. Anti-cheat that lives outside the game's folder (Riot Vanguard, FACEIT's client,
//! Valve's VAC) cannot be seen this way. The name rules follow DLSS-5-Master's
//! `Scanner.AntiCheatOfFile`/`AntiCheatOfFolder` in `src/DLSS5Master/Core/Scanner.cs` (MIT; see
//! ATTRIBUTION.md).

use std::path::Path;

/// How deep [`in_dir`] looks below the game's folder (BattlEye sits in `<Project>/Binaries/Win64`).
const MAX_DEPTH: usize = 6;
/// How many files and folders [`in_dir`] reads at most: a game folder can hold hundreds of
/// thousands, and anti-cheat sits near the top, which a breadth-first walk reaches first.
const MAX_ENTRIES: usize = 50_000;

/// The anti-cheat a file in a game's folder belongs to, by its name (case ignored).
pub fn of_file(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    let starts = |p: &str| n.starts_with(p);
    let ends = |s: &str| n.ends_with(s);
    let is = |s: &str| n == s;
    if starts("easyanticheat") || is("start_protected_game.exe") {
        Some("Easy Anti-Cheat")
    } else if starts("beservice") || starts("beclient") || ends("_be.exe") {
        Some("BattlEye")
    } else if starts("eaanticheat.") {
        Some("EA Javelin")
    } else if is("randgrid.sys") {
        Some("RICOCHET")
    } else if ends(".xem") {
        Some("XIGNCODE3")
    } else if starts("neacsafe") || is("neacclient.exe") || is("neacinterface.dll") {
        Some("NetEase")
    } else if is("hoyokprotect.sys") || starts("mhyprot") {
        Some("HoYoverse")
    } else if starts("denuvo-anti-cheat") || starts("denuvo anti-cheat") {
        Some("Denuvo Anti-Cheat")
    } else if starts("faceit_start_protected_game") {
        Some("FACEIT")
    } else if is("equ8_conf.json") {
        Some("EQU8")
    } else if is("gameguard.des") || (starts("gamemon") && ends(".des")) || (starts("npggnt") && ends(".des")) {
        Some("nProtect GameGuard")
    } else {
        None
    }
}

/// The anti-cheat a folder in a game's folder belongs to, by its name (case ignored).
pub fn of_folder(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "easyanticheat" | "easyanticheat_eos" | "easyanticheateos" => Some("Easy Anti-Cheat"),
        "battleye" => Some("BattlEye"),
        "eaanticheat" => Some("EA Javelin"),
        "anticheatexpert" | "aceantibotclient" => Some("Anti-Cheat Expert (ACE)"),
        "gameguard" => Some("nProtect GameGuard"),
        "xigncode" => Some("XIGNCODE3"),
        "blackcipher" => Some("Nexon Game Security"),
        _ => None,
    }
}

/// Every anti-cheat found in `dir` and below, in the order found, each once. Symbolic links are
/// not followed; an anti-cheat's own folder is not searched further.
pub fn in_dir(dir: &Path) -> Vec<&'static str> {
    let mut found: Vec<&'static str> = Vec::new();
    let mut add = |name: &'static str| {
        if !found.contains(&name) {
            found.push(name);
        }
    };
    let mut pending = std::collections::VecDeque::from([(dir.to_path_buf(), 0usize)]);
    let mut budget = MAX_ENTRIES;
    while let Some((dir, depth)) = pending.pop_front() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if budget == 0 {
                return found;
            }
            budget -= 1;
            let Some(name) = entry.file_name().to_str().map(str::to_string) else { continue };
            // `file_type` does not follow a symbolic link: a link is neither a file nor a folder here.
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if let Some(ac) = of_folder(&name) {
                    add(ac);
                } else if depth < MAX_DEPTH {
                    pending.push_back((entry.path(), depth + 1));
                }
            } else if kind.is_file() {
                if let Some(ac) = of_file(&name) {
                    add(ac);
                }
            }
        }
    }
    found
}

/// `names` as one phrase: "BattlEye", "Easy Anti-Cheat and BattlEye", "A, B and C".
pub fn list(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => (*one).to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// What Neural Forge in a game with anti-cheat risks.
pub const RISK: &str = "Neural Forge does not hide its Vulkan layer from anti-cheat: the game may refuse to start or go \
                        online with the layer loaded, and the anti-cheat can count it as tampering and ban the account, \
                        most of all in online play. Offline, or where the game lets you start it without the anti-cheat, \
                        the risk is lower.";

/// [`RISK`] for a game that ships `names`, for the CLI and the GUI.
pub fn warning(names: &[&str]) -> String {
    format!("This game ships anti-cheat ({}). {RISK}", list(names))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_told_by_name() {
        for (file, expected) in [
            ("EasyAntiCheat_EOS_Setup.exe", Some("Easy Anti-Cheat")),
            ("start_protected_game.exe", Some("Easy Anti-Cheat")),
            ("BEService_x64.exe", Some("BattlEye")),
            ("BEClient_x64.dll", Some("BattlEye")),
            ("ArkAscended_BE.exe", Some("BattlEye")),
            ("EAAntiCheat.GameServiceLauncher.exe", Some("EA Javelin")),
            ("Randgrid.sys", Some("RICOCHET")),
            ("x3_x64.xem", Some("XIGNCODE3")),
            ("NeacSafe64.sys", Some("NetEase")),
            ("HoYoKProtect.sys", Some("HoYoverse")),
            ("mhyprot2.sys", Some("HoYoverse")),
            ("denuvo-anti-cheat-runtime.dll", Some("Denuvo Anti-Cheat")),
            ("Denuvo Anti-Cheat Installer.exe", Some("Denuvo Anti-Cheat")),
            ("FACEIT_Start_Protected_Game.exe", Some("FACEIT")),
            ("equ8_conf.json", Some("EQU8")),
            ("GameMon64.des", Some("nProtect GameGuard")),
            ("npggNT64.des", Some("nProtect GameGuard")),
            ("GTA5_Enhanced.exe", None),
            ("bootstrapper.exe", None),
            ("anticheat_readme.txt", None),
            ("BE_Logo.png", None),
            ("GameMon.log", None),
        ] {
            assert_eq!(of_file(file), expected, "{file}");
        }
    }

    #[test]
    fn folders_are_told_by_name() {
        for (folder, expected) in [
            ("EasyAntiCheat", Some("Easy Anti-Cheat")),
            ("EasyAntiCheatEOS", Some("Easy Anti-Cheat")),
            ("battleye", Some("BattlEye")),
            ("AntiCheatExpert", Some("Anti-Cheat Expert (ACE)")),
            ("GameGuard", Some("nProtect GameGuard")),
            ("XIGNCODE", Some("XIGNCODE3")),
            ("BlackCipher", Some("Nexon Game Security")),
            ("EAAntiCheat", Some("EA Javelin")),
            ("Binaries", None),
            ("AntiCheat", None),
        ] {
            assert_eq!(of_folder(folder), expected, "{folder}");
        }
    }

    struct Tree(std::path::PathBuf);
    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tree(name: &str, files: &[&str], dirs: &[&str]) -> Tree {
        let root = std::env::temp_dir().join(format!("nf-anticheat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for d in dirs {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        for f in files {
            let path = root.join(f);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
        Tree(root)
    }

    #[test]
    fn a_game_folder_is_searched_deep_and_each_anti_cheat_named_once() {
        // ARK-like: BattlEye beside the game in Binaries/Win64, with its _BE launcher.
        let ark = tree(
            "ark",
            &["ShooterGame/Binaries/Win64/ArkAscended.exe", "ShooterGame/Binaries/Win64/ArkAscended_BE.exe"],
            &["ShooterGame/Binaries/Win64/BattlEye"],
        );
        assert_eq!(in_dir(&ark.0), vec!["BattlEye"]);
        // Call of Duty-like: RICOCHET's driver at the top and Easy Anti-Cheat beside it.
        let cod = tree("cod", &["cod.exe", "randgrid.sys", "EasyAntiCheat/EasyAntiCheat_EOS_Setup.exe"], &[]);
        assert_eq!(in_dir(&cod.0), vec!["Easy Anti-Cheat", "RICOCHET"]);
        let clean = tree("clean", &["GTA5_Enhanced.exe", "x64/data.rpf", "AntiCheat/readme.txt"], &[]);
        assert!(in_dir(&clean.0).is_empty());
        assert!(in_dir(Path::new("/nonexistent/nf-anticheat")).is_empty());
    }

    #[test]
    fn deeper_than_the_limit_is_not_searched_and_links_are_not_followed() {
        let deep = tree("deep", &["a/b/c/d/e/f/BEService_x64.exe", "a/b/c/d/e/f/g/randgrid.sys"], &[]);
        assert_eq!(in_dir(&deep.0), vec!["BattlEye"], "depth 6 is searched, 7 is not");
        let elsewhere = tree("elsewhere", &["BEService_x64.exe"], &["BattlEye"]);
        let linked = tree("linked", &["Game.exe"], &[]);
        std::os::unix::fs::symlink(&elsewhere.0, linked.0.join("other")).unwrap();
        std::os::unix::fs::symlink(elsewhere.0.join("BEService_x64.exe"), linked.0.join("BEService_x64.exe")).unwrap();
        assert!(in_dir(&linked.0).is_empty());
    }

    #[test]
    fn names_join_as_a_phrase() {
        assert_eq!(list(&[]), "");
        assert_eq!(list(&["BattlEye"]), "BattlEye");
        assert_eq!(list(&["Easy Anti-Cheat", "BattlEye"]), "Easy Anti-Cheat and BattlEye");
        assert_eq!(list(&["A", "B", "C"]), "A, B and C");
        assert!(warning(&["BattlEye"]).contains("(BattlEye)"));
    }
}
