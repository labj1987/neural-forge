//! Processes that are never the game a player means to run with Neural Forge. Shared by the layer
//! (which keeps them from claiming the channel when no target executable is set) and the supervisor
//! (which leaves them out of the target executables it derives for a game).

/// Processes that are never the game, by their executable name with spaces, `-` and `_` removed:
/// Wine's own services and helpers, store launchers and their web helpers (each presents a window of
/// its own, often game-sized, for as long as it is open), crash reporters and gamescope. Without a
/// `NEURAL_FORGE_TARGET_EXE` the first process to present steadily claims the channel, so one of these
/// could otherwise hold it while the game itself runs. A target that names one of them wins.
const NAMES: &[&str] = &[
    // Wine and Proton.
    "explorer.exe",
    "services.exe",
    "winedevice.exe",
    "plugplay.exe",
    "rpcss.exe",
    "svchost.exe",
    "conhost.exe",
    "start.exe",
    "wineboot.exe",
    "winemenubuilder.exe",
    "tabtip.exe",
    "xalia.exe",
    // Steam.
    "steam.exe",
    "steamwebhelper.exe",
    "steamservice.exe",
    "steamerrorreporter.exe",
    "steamerrorreporter64.exe",
    "gameoverlayui.exe",
    // Rockstar Games Launcher and Social Club (also matched by prefix below).
    "launcher.exe",
    "launcherpatcher.exe",
    "rockstarservice.exe",
    "rockstarlauncher.exe",
    "socialclubhelper.exe",
    "socialclub.exe",
    // Ubisoft Connect.
    "upc.exe",
    "uplay.exe",
    "uplaywebcore.exe",
    "uplayservice.exe",
    "ubisoftconnect.exe",
    "ubisoftgamelauncher.exe",
    "ubisoftgamelauncher64.exe",
    // EA app and Origin.
    "eadesktop.exe",
    "ealauncher.exe",
    "eabackgroundservice.exe",
    "ealocalhostsvc.exe",
    "eacefsubprocess.exe",
    "origin.exe",
    "originwebhelperservice.exe",
    "originthinsetupinternal.exe",
    "qtwebengineprocess.exe",
    // Battle.net.
    "battle.net.exe",
    "battle.netlauncher.exe",
    "agent.exe",
    "blizzarderror.exe",
    "blizzardbrowser.exe",
    // Epic Games Launcher and Epic Online Services.
    "epicgameslauncher.exe",
    "epicwebhelper.exe",
    "epiconlineserviceshost.exe",
    "epiconlineservicesuserhelper.exe",
    "eosoverlayrenderer.exe",
    "eosoverlayrendererwin64shipping.exe",
    "eosoverlayrendererwin32shipping.exe",
    // GOG Galaxy.
    "galaxyclient.exe",
    "galaxyclienthelper.exe",
    "galaxycommunication.exe",
    "goggalaxynotificationsrenderer.exe",
    // Amazon Games, Riot, Wargaming, Paradox.
    "amazongamesui.exe",
    "riotclientservices.exe",
    "riotclientux.exe",
    "riotclientuxrender.exe",
    "wgc.exe",
    "paradoxlauncher.exe",
    // Crash reporters.
    "crashreportclient.exe",
    "unitycrashhandler64.exe",
    "unitycrashhandler32.exe",
    "gamescope",
];

/// Name prefixes (spaces, `-` and `_` removed) of whole launcher families, whatever the helper.
const PREFIXES: &[&str] =
    &["rockstar", "socialclub", "xalia", "ubisoftconnect", "uplay", "battle.net", "epicgameslauncher", "epiconlineservices", "galaxyclient", "riotclient"];

/// Whether the executable `name` (a basename such as `Launcher.exe`, any case) is one of the
/// processes above.
pub fn never_the_game(name: &str) -> bool {
    let compact = name.to_ascii_lowercase().replace([' ', '-', '_'], "");
    !compact.is_empty() && (NAMES.contains(&compact.as_str()) || PREFIXES.iter().any(|prefix| compact.starts_with(prefix)))
}

/// Executables a game folder ships that are not the game: redistributable installers, anti-cheat
/// services and crash reporters (matched on the lowercase name), and every process of [`never_the_game`].
pub fn game_helper(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    never_the_game(&n)
        || [
            "setup",
            "redist",
            "prereq",
            "unins",
            "installer",
            "dotnet",
            "crashreport",
            "crashhandler",
            "crashpad",
            "crs-",
            "beservice",
            "easyanticheat",
            "start_protected_game",
            "bugsplat",
            "vcredist",
            "cefsubprocess",
            "webhelper",
        ]
        .iter()
        .any(|part| n.contains(part))
}

/// Names of programs that start the game rather than being it: store and publisher launchers,
/// bootstrappers, updaters and patchers (`Bethesda.net_Launcher.exe`, `ffxivboot64.exe`).
pub fn launcher_like(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    never_the_game(&n) || ["launcher", "updater", "patcher"].iter().any(|p| n.contains(p)) || n.ends_with("boot.exe") || n.ends_with("boot64.exe")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launchers_and_helpers_are_told_from_games() {
        for name in ["Bethesda.net_Launcher.exe", "REDprelauncher.exe", "ffxivlauncher64.exe", "ffxivboot64.exe", "BethesdaNetUpdater.exe"] {
            assert!(launcher_like(name), "{name}");
        }
        for name in ["CrashReportClient.exe", "vc_redist.x64.exe", "UnrealCEFSubProcess.exe", "start_protected_game.exe"] {
            assert!(game_helper(name), "{name}");
        }
        for name in ["eso64.exe", "ffxiv_dx11.exe", "GTA5_Enhanced.exe", "Cyberpunk2077.exe"] {
            assert!(!launcher_like(name) && !game_helper(name), "{name}");
        }
    }

    #[test]
    fn names_are_matched_without_case_spaces_dashes_or_underscores() {
        for name in ["Launcher.exe", "Battle.net Launcher.exe", "EOSOverlayRenderer-Win64-Shipping.exe", "Rockstar Games Launcher.exe", "gamescope", "UPC.EXE"]
        {
            assert!(never_the_game(name), "{name}");
        }
        for name in ["GTA5_Enhanced.exe", "Cyberpunk2077.exe", "HogwartsLegacy.exe", ""] {
            assert!(!never_the_game(name), "{name}");
        }
    }
}
