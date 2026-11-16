//! What the window says and decides, apart from how it is drawn: the texts of rows and dialogs, the values a spin row
//! keeps, the "Settings for" menu's entries. Plain functions, so they are tested without a window.

use std::sync::atomic::Ordering;

/// Whether a game's layer is alive: its heartbeat (`layer_heartbeat`) changed within the last two
/// seconds. A closed game leaves its last values in the channel; only a moving heartbeat says they
/// are current. Each timer that asks keeps its own.
#[derive(Default)]
pub(crate) struct Liveness(std::cell::Cell<(Option<u32>, Option<std::time::Instant>)>);

impl Liveness {
    pub(crate) fn alive(&self, hdr: &neural_forge_protocol::ShmHeader) -> bool {
        let now = hdr.layer_heartbeat.load(Ordering::Relaxed);
        let (seen, changed) = self.0.get();
        let changed = match seen {
            Some(seen) if seen != now => Some(std::time::Instant::now()),
            _ => changed,
        };
        self.0.set((Some(now), changed));
        changed.is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(2))
    }
}

/// The decimals a spin row shows: as many as its step has, and more (up to 3) while the value has
/// them, so a value set off the step by shmctl, a profile or config.ini (16.5 on a step-1 row, 0.004
/// on a step-0.01 one) is shown as it is.
pub(crate) fn spin_digits(step: f64, value: f64) -> u32 {
    decimals(step).max(decimals(value))
}

/// The fewest decimals (0 to 3) that write `x` exactly, 3 when it has more.
pub(crate) fn decimals(x: f64) -> u32 {
    (0..3)
        .find(|&d| {
            let shifted = x * 10f64.powi(d as i32);
            (shifted - shifted.round()).abs() <= 1e-6 * shifted.abs().max(1.0)
        })
        .unwrap_or(3)
}

/// The value a spin row keeps when its field's `text` is read back: the current `value` while the text
/// still shows it at `digits` decimals (either decimal mark), so precision the row does not show
/// survives a focus change; `None` for edited text, which the row then parses as a new value.
pub(crate) fn kept_spin_value(text: &str, value: f64, digits: u32) -> Option<f64> {
    let typed: f64 = text.trim().replace(',', ".").parse().ok()?;
    let d = digits as usize;
    (format!("{typed:.d$}") == format!("{value:.d$}")).then_some(value)
}

pub(crate) fn format_key_label(code: u32, name: Option<&str>) -> String {
    match name {
        Some(name) if name.chars().count() == 1 => name.to_uppercase(),
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => format!("Key {code}"),
    }
}

/// Whether the row of setting `name` shows the "This game" mark: a game is picked and it has its own value of
/// that setting. Rows following All games (and every row while All games is picked) stay unmarked.
pub(crate) fn override_marked(
    name: &str,
    game: Option<&neural_forge_supervisor::game_settings::GameKey>,
    own: &neural_forge_supervisor::game_settings::Section,
) -> bool {
    game.is_some() && own.contains_key(&format!("set_{name}"))
}

/// One entry of the "Settings for" menu: all games (`None`) or one game, and its label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScopeEntry {
    pub(crate) game: Option<neural_forge_supervisor::game_settings::GameKey>,
    pub(crate) label: String,
}

/// The "Settings for" menu's entries: All games, the installed Steam games by name (`steam`: app id and
/// name), then every other game with settings of its own, and `extra` (the game on the channel, the game
/// whose settings it holds), each once.
pub(crate) fn scope_entries(
    steam: &[(String, String)],
    with_settings: &[neural_forge_supervisor::game_settings::GameKey],
    extra: &[neural_forge_supervisor::game_settings::GameKey],
) -> Vec<ScopeEntry> {
    use neural_forge_supervisor::game_settings::GameKey;
    let mut entries = vec![ScopeEntry { game: None, label: "All games".into() }];
    let mut games: Vec<(String, GameKey)> = steam.iter().filter_map(|(appid, name)| GameKey::parse(appid).map(|key| (name.clone(), key))).collect();
    games.sort_by_key(|a| a.0.to_lowercase());
    for (label, game) in games {
        if !entries.iter().any(|e| e.game.as_ref() == Some(&game)) {
            entries.push(ScopeEntry { game: Some(game), label });
        }
    }
    for game in with_settings.iter().chain(extra) {
        if !entries.iter().any(|e| e.game.as_ref() == Some(game)) {
            let label = match game {
                GameKey::Steam(id) => format!("Steam app {id}"),
                GameKey::Exe(name) => name.clone(),
            };
            entries.push(ScopeEntry { game: Some(game.clone()), label });
        }
    }
    entries
}

/// The line beside the menu: what the chosen settings are, how many the game has of its own and whether it
/// runs now.
pub(crate) fn scope_summary(game: Option<&neural_forge_supervisor::game_settings::GameKey>, own: usize, shipped: usize, running: bool) -> String {
    let what = match game {
        None => "Every game without settings of its own uses these".to_string(),
        Some(_) if own == 0 && shipped == 0 => "Same as All games until you change something here".to_string(),
        Some(_) => {
            let mut parts = Vec::new();
            if own > 0 {
                parts.push(format!("{own} setting{} of its own", if own == 1 { "" } else { "s" }));
            }
            if shipped > 0 {
                parts.push(format!("{shipped} built in"));
            }
            format!("{}; the rest follow All games", parts.join(", "))
        }
    };
    if running {
        format!("Running now. {what}")
    } else {
        what
    }
}

/// The placements' names in the menus (`enums::placement`, by value) and what each does.
pub(crate) const PLACEMENTS: [(&str, &str); 4] = [
    ("Auto", "Before the upscaler; after DLSS while Ray Reconstruction runs; on the finished frame where neither works"),
    ("Before the upscaler", "On DLSS's input at the render size, fastest (with Ray Reconstruction: on the finished frame)"),
    ("After DLSS", "On DLSS's output at the output size, before the game's HUD; where the game allows it, else as Auto"),
    ("Finished frame", "On the presented frame with the HUD; the HUD settings below keep the edit off it"),
];

/// [`PLACEMENTS`]' names alone (the per-pass dialog's menu).
pub(crate) const PLACEMENT_NAMES: [&str; 4] = [PLACEMENTS[0].0, PLACEMENTS[1].0, PLACEMENTS[2].0, PLACEMENTS[3].0];

/// The HUD region rows' left and top, and width and height, in percent: no value they take leaves a region
/// empty once the header clips it to the frame (`HudRegion::clamped`, which drops an empty one), so an edit
/// can clip a region but never remove it.
pub(crate) const HUD_POSITION_RANGE: (f64, f64) = (0.0, 99.5);

pub(crate) const HUD_SIZE_RANGE: (f64, f64) = (0.5, 100.0);

/// Applies `change` to HUD region `k` of the header's list and writes the list (`write`). Returns what the
/// header holds then, and whether that differs from what was written: the header clips a region to the frame
/// and drops an empty one, and then the rows (their values, and with a region dropped their indices) no longer
/// match it and must be rebuilt from it.
pub(crate) fn edit_hud_region(
    header: &neural_forge_protocol::ShmHeader,
    k: usize,
    change: &dyn Fn(&mut neural_forge_protocol::HudRegion),
    write: &dyn Fn(&[neural_forge_protocol::HudRegion]),
) -> (Vec<neural_forge_protocol::HudRegion>, bool) {
    let mut meant = header.hud_settings().regions;
    if let Some(r) = meant.get_mut(k) {
        change(r);
    }
    write(&meant);
    let held = header.hud_settings().regions;
    let stale = held != meant;
    (held, stale)
}

/// The Network row's choices, in [`Network::ALL`]'s order (the Network setting's values).
pub(crate) const NETWORK_CHOICES: [&str; 2] = ["Full network", "Small network"];

/// The games group's description: what the switches do, or why they can't be used right now.
pub(crate) fn games_note(roots: &[neural_forge_supervisor::steam::Root], games: usize, steam_running: bool) -> String {
    if roots.is_empty() {
        return "Steam was not found. Set the launch option below by hand".to_string();
    }
    let mut note = String::new();
    if neural_forge_supervisor::steam::only_flatpak(roots) {
        note.push_str("Steam is the Flatpak build: the games it runs cannot see Neural Forge's Vulkan layer. ");
    }
    note.push_str(if games == 0 {
        "No installed games found"
    } else if steam_running {
        "Close Steam to switch games here: Steam rewrites its settings when it exits, so a change made while it runs is lost. Or copy the launch option below into the game's Properties"
    } else {
        "Turns Neural Forge on in the game's Steam launch options, keeping the options already there"
    });
    note
}

/// The Tested game row: the game's line from the tested games and its setup notes (the tooltip), or that it is not
/// among them, or that no game has attached.
pub(crate) fn tested_label(game: &str, facts: Option<&neural_forge_supervisor::compat::Game>) -> (String, Option<String>) {
    match facts {
        Some(f) => (format!("{}: {}", f.name, f.status_line()), (!f.notes.is_empty()).then(|| f.notes.join("\n"))),
        None if game.is_empty() => ("no game attached yet".to_string(), None),
        None => (format!("{game} is not among the tested games: where the model runs in it is not known yet"), None),
    }
}

/// A game row's subtitle: its appid, library and target (or why it has none), its anti-cheat, and where the model ran
/// in it in testing.
pub(crate) fn game_subtitle(game: &neural_forge_supervisor::steam::Game, target: &str, chosen: bool, anti_cheat: &[&str]) -> String {
    let detail = match (game.unsupported(), target.is_empty()) {
        (Some(why), _) => format!("cannot be switched on: {why}"),
        (None, true) => "its executable was not found: choose it with the edit button".to_string(),
        (None, false) => format!("runs {}{}", target.replace(',', ", "), if chosen { " (chosen)" } else { "" }),
    };
    let anti_cheat = match anti_cheat {
        [] => String::new(),
        found => format!(" · anti-cheat: {}", neural_forge_supervisor::anti_cheat::list(found)),
    };
    let tested = match neural_forge_supervisor::compat::find(game.appid.parse().ok(), "") {
        Some(f) => format!(" · tested: {}", f.placement.describe()),
        None => String::new(),
    };
    format!("{} · {} · {detail}{anti_cheat}{tested}", game.appid, game.library.display())
}

/// Whether a game's switch can be used: not while Steam runs; off always, on only for a game that
/// can load the layer and has a target.
pub(crate) fn switch_usable(game: &neural_forge_supervisor::steam::Game, target: &str, on: bool, steam_running: bool) -> bool {
    !steam_running && (on || (game.unsupported().is_none() && !target.is_empty()))
}

/// The confirmation's text for Uninstall: everything it removes, what it cannot while Steam runs, and that deleting
/// the AppImage alone leaves Neural Forge installed.
pub(crate) fn uninstall_body(steam_running: bool, appimage: Option<&std::path::Path>) -> String {
    let mut s = String::from(
        "This removes everything Neural Forge put on this computer:\n\
         \u{2022} the Vulkan layer every game loads, the app and the command-line tool\n\
         \u{2022} your settings, profiles and per-game settings\n\
         \u{2022} the extracted model and the imported NVIDIA DLL, captures and logs\n\
         \u{2022} Neural Forge's own Proton builds (the *-NeuralForge tools in Steam)\n\
         \u{2022} Neural Forge's part of every game's Steam launch options (your own options stay)\n",
    );
    if let Some(file) = appimage {
        s.push_str(&format!("\u{2022} this AppImage file, {}\n", file.display()));
    }
    if steam_running {
        s.push_str(
            "\nSteam is running, so the launch options cannot be changed now (Steam writes its own copy back when it \
             exits). Close Steam first to have them cleaned; otherwise the games are listed afterwards, with the options \
             to set by hand. Neural Forge's variables do nothing once the layer is gone.\n",
        );
    }
    s.push_str(
        "\nDeleting only the AppImage file does not uninstall Neural Forge: the layer it installed stays, and the games \
         switched on keep loading it. Games using a Neural Forge Proton build need another Proton chosen in their \
         Properties afterwards.",
    );
    s
}

/// What Uninstall did, for the closing dialog.
pub(crate) fn uninstall_result(purged: &neural_forge_supervisor::install::Purged, appimage_kept: Option<&std::path::Path>) -> String {
    let mut s = format!("Removed {} files and folders.", purged.removed.len());
    if !purged.switched_off.is_empty() {
        s.push_str(&format!("\n\nSwitched off in the launch options of: {}.", purged.switched_off.join(", ")));
    }
    if !purged.left.is_empty() {
        s.push_str("\n\nThese games still have Neural Forge's variables in their launch options (harmless without the layer):\n");
        s.push_str(&purged.left.join("\n"));
    }
    if let Some(file) = appimage_kept {
        s.push_str(&format!("\n\nDelete {} too: starting it again installs Neural Forge again.", file.display()));
    } else {
        s.push_str("\n\nIf you start Neural Forge from another AppImage file, delete it too: starting it installs Neural Forge again.");
    }
    s
}

/// One sample of the graph's two series: how long a frame took and how much of it the model's network took on
/// the GPU, sampled from the header on a faster timer than the once-a-second status labels need. The last few
/// seconds are kept for the drawing area.
#[derive(Clone, Copy)]
pub(crate) struct TelemetrySample {
    /// Milliseconds per presented frame over the sample's interval (`None`: no frames, or no game).
    pub(crate) frame_ms: Option<f32>,
    /// The network's GPU milliseconds on its last hold (`preupscale_network_gpu_ms`); `None` while it does not run.
    pub(crate) model_ms: Option<f32>,
}

pub(crate) const TELEMETRY_INTERVAL_MS: u32 = 200;

pub(crate) const TELEMETRY_WINDOW_SECS: u32 = 10;

pub(crate) const TELEMETRY_SAMPLES: usize = (TELEMETRY_WINDOW_SECS * 1000 / TELEMETRY_INTERVAL_MS) as usize;

/// The graph's top in milliseconds: the smallest of 5, 10, 20, 40 and 80 above every sample, so a line's height
/// means the same while the frame rate holds and the scale only moves for a real change.
pub(crate) fn telemetry_ceiling(history: &std::collections::VecDeque<TelemetrySample>) -> f32 {
    let top = history.iter().flat_map(|s| [s.frame_ms, s.model_ms]).flatten().fold(0.0f32, f32::max);
    [5.0, 10.0, 20.0, 40.0].into_iter().find(|&c| top <= c * 0.95).unwrap_or(80.0)
}

/// How long a capture request may wait for a game to take it before the GUI withdraws it, and how
/// long a taken one-shot may take to be written.
pub(crate) const CAPTURE_TAKE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

pub(crate) const CAPTURE_RESULT_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// What the capture button's row says, from the channel: whether the model edits the game's own frame now
/// ([`edits_in_game_frame`]), where a capture has no frame without the model's edit (see the button's tooltip).
pub(crate) fn capture_subtitle(in_game_frame: bool) -> &'static str {
    if in_game_frame {
        "The model runs in the game's frame now (before the upscaler or after DLSS): both images will be the presented frame, which already has its edit"
    } else {
        "Saves the next frame as two PNGs: the game's frame and Neural Forge's result"
    }
}

/// The Debug page's captures: the same one-frame request as `neural-forge-cli shmctl capture`, and the
/// folder the layer writes them to.
/// Whether `preupscale_state` says the model edits the game's own frame: holding before the upscaler, after DLSS,
/// or both. The layer then keeps the after-the-upscaler compose off (`Session::suppresses_post`: the plan has a
/// stage in the game's frame, holds engaged it and DLSS ran lately) and reads the presented frame back as both
/// images of a capture. Waiting for DLSS input, DLSS identified but not held, and off run the model at present.
pub(crate) fn edits_in_game_frame(state: u32) -> bool {
    use neural_forge_protocol::enums::preupscale_state::{HOLDING, HOLDING_AFTER, HOLDING_BOTH, PAUSED};
    // Paused with DLSS (#49) the post path is still kept off: the layer captures the presented frame as both images.
    matches!(state, HOLDING | HOLDING_AFTER | HOLDING_BOTH | PAUSED)
}

/// The Status page's one line on the pre-upscaler path, from the header's `preupscale_*` fields
/// (`neural_forge_protocol::enums::preupscale_state`). `layer_active`: the layer's heartbeat is
/// fresh; otherwise the fields are a closed game's leftovers. `reason`: why no hold runs while DLSS's input is
/// identified (`preupscale_reason`, with `IDENTIFIED`). `placement`: the Model placement setting; "At present"
/// runs the model on the finished frame by choice, so the line says that rather than "waiting" for DLSS (#49).
pub(crate) fn preupscale_label(layer_active: bool, placement: u32, state: u32, width: u32, height: u32, misses: u32, reason: &str) -> String {
    use neural_forge_protocol::enums::preupscale_state::{HOLDING, HOLDING_AFTER, HOLDING_BOTH, IDENTIFIED, NO_WRAPPER, PAUSED, WAITING};
    let missed = if misses == 0 { String::new() } else { format!(", {misses} frame{} missed", if misses == 1 { "" } else { "s" }) };
    match state {
        _ if !layer_active => "no game running".to_string(),
        _ if placement == neural_forge_protocol::enums::placement::PRESENT && !matches!(state, HOLDING | HOLDING_AFTER | HOLDING_BOTH) => {
            "at present, on the finished frame (the Model placement setting)".to_string()
        }
        HOLDING => format!("before the upscaler: holding DLSS's {width}x{height} input{missed}"),
        HOLDING_AFTER => format!("after DLSS, before the HUD: holding DLSS's {width}x{height} output{missed}"),
        HOLDING_BOTH => format!("before the upscaler and after DLSS, before the HUD{missed}"),
        WAITING => format!("after the upscaler: waiting for DLSS Super Resolution{missed}"),
        PAUSED => format!("paused with the game's DLSS (a menu or loading screen): the model resumes with its next frame{missed}"),
        NO_WRAPPER if reason.is_empty() => {
            "off: this game uses DLSS but isn't on the Neural Forge Proton; switch it on in the Setup tab to set it (Steam closed)".to_string()
        }
        NO_WRAPPER => format!("off: {reason}"),
        IDENTIFIED if reason.is_empty() => format!("after the upscaler: DLSS input identified, not holding{missed}"),
        IDENTIFIED => format!("after the upscaler: DLSS input identified, not holding{missed} -- {reason}"),
        _ => "after the upscaler".to_string(),
    }
}

/// The narrowest and lowest the window goes (its minimum size): 360 wide, and tall enough for the header, a banner and
/// a row.
pub(crate) const MIN_WIDTH: i32 = 360;

pub(crate) const MIN_HEIGHT: i32 = 400;

/// The window's first size: wide enough to show the tabs in the header (past the 1400sp breakpoint in `build_ui`)
/// and 900 high, but no more than 90% of the screen (`screen`, logical pixels; unknown: as if large enough), never below
/// the minimum. A screen narrower than ~1560 gets the bottom tab bar, which is what fits there.
pub(crate) fn default_window_size(screen: Option<(i32, i32)>) -> (i32, i32) {
    const WIDE: i32 = 1440;
    const HIGH: i32 = 900;
    let (w, h) = screen.map(|(w, h)| (w * 9 / 10, h * 9 / 10)).unwrap_or((WIDE, HIGH));
    (w.clamp(MIN_WIDTH, WIDE), h.clamp(MIN_HEIGHT, HIGH))
}

/// One program that can show a message when the app has no window: its name and the arguments before and after the
/// message.
pub(crate) type MessageTool = (&'static str, &'static [&'static str], &'static [&'static str]);

/// The arguments that ask the desktop's notification service to show a message, before and after its text.
const NOTIFY_BEFORE: &[&str] = &[
    "call",
    "--session",
    "--dest",
    "org.freedesktop.Notifications",
    "--object-path",
    "/org/freedesktop/Notifications",
    "--method",
    "org.freedesktop.Notifications.Notify",
    "--",
    // Each argument is GVariant text, typed: app name, replaces id, icon, summary; the body (`message_for`) follows.
    "'Neural Forge'",
    "uint32 0",
    "''",
    "'Neural Forge'",
];

/// The programs tried, in order, to show an error when the app has no window: the same order as AppRun's (zenity,
/// kdialog, notify-send, the notification service over D-Bus, xmessage).
pub(crate) const ERROR_DIALOGS: &[MessageTool] = &[
    ("zenity", &["--error", "--no-markup", "--title=Neural Forge", "--width=480", "--text"], &[]),
    ("kdialog", &["--title", "Neural Forge", "--error"], &[]),
    // (No expiry: some desktops drop even a critical notification after a few seconds otherwise.)
    ("notify-send", &["--urgency=critical", "--expire-time=0", "--app-name=Neural Forge", "Neural Forge"], &[]),
    ("gdbus", NOTIFY_BEFORE, &["@as []", "{'urgency': <byte 2>}", "int32 0"]),
    ("xmessage", &["-center"], &[]),
];

/// The same programs showing something that is no error (a first copy that is still starting): an information
/// dialog, a notification of normal urgency.
pub(crate) const INFO_DIALOGS: &[MessageTool] = &[
    ("zenity", &["--info", "--no-markup", "--title=Neural Forge", "--width=480", "--text"], &[]),
    ("kdialog", &["--title", "Neural Forge", "--msgbox"], &[]),
    ("notify-send", &["--urgency=normal", "--app-name=Neural Forge", "Neural Forge"], &[]),
    // (The desktop's own expiry: information need not stay up.)
    ("gdbus", NOTIFY_BEFORE, &["@as []", "{'urgency': <byte 1>}", "int32 -1"]),
    ("xmessage", &["-center"], &[]),
];

/// Whether `program`, having ended with exit `code`, did show the message, so the next one must not show it again.
/// The dialogs end with 0 when the user pressed their button and 1 when the user closed them another way; both mean it
/// was seen. Anything else (no display, a crash, a signal) does not. notify-send shows nothing itself: only 0 says the
/// desktop took the message (the same for the notification service asked over D-Bus). xmessage ends with 1 when it
/// could not open the display, and so does zenity, which then says so on its `stderr`; with `stderr` unknown (`None`),
/// a dialog's 1 is not taken as seen.
pub(crate) fn helper_showed(program: &str, code: Option<i32>, stderr: Option<&str>) -> bool {
    match (program, stderr) {
        ("notify-send" | "gdbus" | "xmessage", _) => code == Some(0),
        (_, None) => code == Some(0),
        (_, Some(stderr)) if no_display(stderr) => false,
        _ => matches!(code, Some(0 | 1)),
    }
}

/// Whether a dialog program's `stderr` says it could not open the display.
fn no_display(stderr: &str) -> bool {
    ["Failed to open display", "cannot open display", "Cannot open display"].iter().any(|text| stderr.contains(text))
}

/// The message as `program` is given it: zenity (`--no-markup`) and xmessage show it as it is; kdialog (Qt rich text)
/// and the notification service (body markup) get it with nothing they would take as markup (no &, < or >), and gdbus
/// as a GVariant text-format string, which is how it reads its arguments.
pub(crate) fn message_for(program: &str, message: &str) -> String {
    let plain = || message.replace('&', "and").replace('<', "‹").replace('>', "›");
    match program {
        "zenity" | "xmessage" => message.to_string(),
        "gdbus" => gvariant_string(&plain()),
        _ => plain(),
    }
}

/// `text` as a GVariant text-format string: single-quoted, with \, ', newline, tab and carriage return escaped.
fn gvariant_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('\'');
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// What is said when the window cannot be opened: no graphics driver at all, or no display.
pub(crate) fn no_window_message(error: &str, has_display: bool) -> String {
    if has_display {
        format!(
            "Neural Forge could not open its window: no working graphics driver was found (Vulkan or OpenGL ES). Install or \
             repair your GPU driver, or install Mesa's software renderer (mesa-vulkan-drivers on Debian, Ubuntu and Fedora; \
             vulkan-swrast on Arch). Games are not affected: the layer and neural-forge-cli work without the window.\n\nDetails: {error}"
        )
    } else {
        format!(
            "Neural Forge could not open its window: there is no display to open it on (neither WAYLAND_DISPLAY nor DISPLAY \
             is set). Start it from your desktop session.\n\nDetails: {error}"
        )
    }
}

#[cfg(test)]
mod no_window_tests {
    use super::{helper_showed, no_window_message};

    #[test]
    fn the_no_window_message_names_the_real_cause() {
        assert!(no_window_message("x", true).contains("graphics driver"));
        assert!(no_window_message("x", false).contains("no display"));
        assert!(no_window_message("the detail", true).ends_with("Details: the detail"));
    }

    #[test]
    fn an_error_dialog_that_was_seen_is_not_shown_twice() {
        let shown = |program, code| helper_showed(program, code, Some(""));
        assert!(shown("zenity", Some(0)) && shown("kdialog", Some(1)));
        assert!(!shown("zenity", Some(127)) && !shown("zenity", None));
        assert!(shown("notify-send", Some(0)) && !shown("notify-send", Some(1)));
        assert!(shown("gdbus", Some(0)) && !shown("gdbus", Some(1)));
        assert!(shown("xmessage", Some(0)) && !shown("xmessage", Some(1)), "xmessage ends with 1 when it can't open the display");
        assert!(!helper_showed("zenity", Some(1), Some("(zenity:42): Gtk-WARNING **: cannot open display: :0\n")), "zenity with no display ends with 1 too");
        assert!(!helper_showed("zenity", Some(1), Some("Failed to open display\n")));
        assert!(!helper_showed("kdialog", Some(1), Some("qt.qpa.xcb: could not connect to display\nCannot open display\n")));
        for tools in [super::ERROR_DIALOGS, super::INFO_DIALOGS] {
            let order: Vec<&str> = tools.iter().map(|&(program, ..)| program).collect();
            assert_eq!(order, ["zenity", "kdialog", "notify-send", "gdbus", "xmessage"], "the order AppRun's libcheck_show tries them in");
        }
        let said =
            |tools: &[super::MessageTool], word: &str| tools.iter().any(|(_, before, after)| before.iter().chain(after.iter()).any(|arg| arg.contains(word)));
        assert!(said(super::ERROR_DIALOGS, "--error") && said(super::ERROR_DIALOGS, "critical"));
        assert!(said(super::ERROR_DIALOGS, "--expire-time=0"), "an error's notification stays until it is closed");
        assert!(!said(super::INFO_DIALOGS, "--expire-time"));
        assert!(!said(super::INFO_DIALOGS, "--error") && !said(super::INFO_DIALOGS, "critical"), "a start that takes long is no error");
        assert!(said(super::INFO_DIALOGS, "--info") && said(super::INFO_DIALOGS, "--msgbox") && said(super::INFO_DIALOGS, "--urgency=normal"));
        let expiry = |tools: &[super::MessageTool]| tools.iter().find(|(program, ..)| *program == "gdbus").map(|(_, _, after)| after[after.len() - 1]);
        assert_eq!(expiry(super::ERROR_DIALOGS), Some("int32 0"), "an error's notification stays until it is closed");
        assert_eq!(expiry(super::INFO_DIALOGS), Some("int32 -1"), "information gets the desktop's own expiry");
        assert_eq!(super::message_for("zenity", "a <b> & c"), "a <b> & c");
        assert_eq!(super::message_for("xmessage", "a <b> & c"), "a <b> & c");
        for program in ["kdialog", "notify-send"] {
            assert_eq!(super::message_for(program, "a <b> & c"), "a ‹b› and c");
        }
        assert_eq!(super::message_for("gdbus", "a <b> & c"), "'a ‹b› and c'");
    }

    /// The Notify call as gdbus gets it: every argument GVariant text of the method's types, the body included,
    /// whatever it holds (a Windows path's backslashes, a quote, a trailing backslash, line breaks).
    #[test]
    fn gdbus_gets_every_notify_argument_as_typed_gvariant_text() {
        let body = "it's at C:\\new\\x\ttab\r\nline 2 \\";
        let quoted = super::message_for("gdbus", body);
        assert_eq!(quoted, r"'it\'s at C:\\new\\x\ttab\r\nline 2 \\'");
        for tools in [super::ERROR_DIALOGS, super::INFO_DIALOGS] {
            let (_, before, after) = tools.iter().find(|(program, ..)| *program == "gdbus").unwrap();
            let notify: Vec<&str> = before[before.len() - 4..].iter().copied().chain([quoted.as_str()]).chain(after.iter().copied()).collect();
            // Notify(app_name s, replaces_id u, app_icon s, summary s, body s, actions as, hints a{sv}, expire_timeout i)
            let types = ["s", "u", "s", "s", "s", "as", "a{sv}", "i"];
            assert_eq!(notify.len(), types.len(), "{notify:?}");
            // GLib's own parser where python3-gi is there; the exact text otherwise.
            let script = "import sys\nfrom gi.repository import GLib\nfor t, a in zip(sys.argv[1::2], sys.argv[2::2]):\n    print(GLib.Variant.parse(GLib.VariantType(t), a).print_(True))\n";
            let mut command = std::process::Command::new("python3");
            command.args(["-I", "-c", script]);
            for (t, a) in types.iter().zip(&notify) {
                command.args([t, a]);
            }
            match command.stderr(std::process::Stdio::null()).output() {
                Ok(out) if out.status.success() => {
                    let printed = String::from_utf8(out.stdout).unwrap();
                    let lines: Vec<&str> = printed.lines().collect();
                    assert_eq!(lines.len(), 8, "{printed}");
                    assert_eq!(lines[1], "uint32 0");
                    let parsed = std::process::Command::new("python3")
                        .args([
                            "-I",
                            "-c",
                            "import sys\nfrom gi.repository import GLib\nsys.stdout.write(GLib.Variant.parse(GLib.VariantType('s'), sys.argv[1]).get_string())",
                            &quoted,
                        ])
                        .output()
                        .unwrap();
                    assert_eq!(String::from_utf8(parsed.stdout).unwrap(), "it's at C:\\new\\x\ttab\r\nline 2 \\", "the body round-trips");
                }
                _ => {
                    assert_eq!(&notify[..4], ["'Neural Forge'", "uint32 0", "''", "'Neural Forge'"]);
                    assert_eq!(notify[5], "@as []");
                }
            }
        }
    }
}

/// What is said when the desktop's file dialog cannot be opened (no FileChooser portal on this desktop): the error,
/// what provides one, and `without`, the way to do the same without a dialog.
pub(crate) fn no_file_dialog(error: &str, without: &str) -> String {
    format!(
        "Could not open the file dialog: {error}. The desktop offers none: install a file-chooser portal (xdg-desktop-portal-gtk, or your desktop's own), or {without}"
    )
}

#[cfg(test)]
mod file_dialog_tests {
    #[test]
    fn a_missing_file_dialog_names_the_way_forward() {
        let said = super::no_file_dialog("no portal", "run `neural-forge-cli extract-model DIR` instead");
        assert!(said.starts_with("Could not open the file dialog: no portal."));
        assert!(said.contains("xdg-desktop-portal-gtk") && said.ends_with("neural-forge-cli extract-model DIR` instead"));
    }
}

/// The screen's size in the window's own units, from the size the toolkit reports for it and the window's scale. On
/// Wayland the toolkit divides the screen's mode by the output's whole-number scale, which for a fractional scale is
/// the next whole number up (150% arrives as 2): the reported size is then too small by that factor over the real
/// one. A whole-number scale needs no correction, and on X11 the toolkit divides by the true scale itself.
pub(crate) fn logical_screen(reported: (i32, i32), window_scale: f32, wayland: bool) -> (i32, i32) {
    if !wayland || !window_scale.is_finite() || window_scale <= 0.0 || window_scale.fract().abs() < 0.01 {
        return reported;
    }
    let factor = window_scale.ceil() / window_scale;
    ((reported.0 as f32 * factor).round() as i32, (reported.1 as f32 * factor).round() as i32)
}

/// How long after the window is up the fit to the real screen may still be made: the compositor says the window's scale
/// within its first moments. Later scale changes (the user changing the screen's scale, the window moving to another
/// screen) are not what the first fit got wrong.
pub(crate) const REFIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(3);

/// Whether the fit to the real screen may still be made: only for a window that still has its first size (`untouched`),
/// within [`REFIT_WINDOW`] of its start, on a desktop that had one screen when it started (with several, the reported
/// size is not known to be the one the window opened on).
pub(crate) fn refit_armed(untouched: bool, since_start: std::time::Duration, screens_at_start: usize) -> bool {
    untouched && since_start < REFIT_WINDOW && screens_at_start <= 1
}

/// The size to give the window once its scale is known, if its first size (fitted to the screen as reported) was
/// fitted to a screen reported too small: only while the window still has that first size (`untouched`), so a size
/// the user chose is never overridden. `None`: leave it.
pub(crate) fn refit(reported: (i32, i32), window_scale: f32, wayland: bool, untouched: bool) -> Option<(i32, i32)> {
    let (first, fitted) = (default_window_size(Some(reported)), default_window_size(Some(logical_screen(reported, window_scale, wayland))));
    (untouched && fitted != first).then_some(fitted)
}

#[cfg(test)]
mod window_size_tests {
    use super::{default_window_size, logical_screen, refit, MIN_HEIGHT, MIN_WIDTH};

    #[test]
    fn the_window_opens_wide_enough_for_the_header_tabs_where_the_screen_allows() {
        // 1440 is past the 1400sp breakpoint at the default text scale: the tabs sit in the header.
        assert_eq!(default_window_size(Some((2560, 1440))), (1440, 900));
        assert_eq!(default_window_size(Some((1920, 1080))), (1440, 900));
        assert_eq!(default_window_size(None), (1440, 900));
        // A smaller screen: 90% of it, and the bottom tab bar.
        assert_eq!(default_window_size(Some((1366, 768))), (1229, 691));
        // Never below the minimum.
        assert_eq!(default_window_size(Some((300, 200))), (MIN_WIDTH, MIN_HEIGHT));
        // Wayland, a 2560x1440 screen at 150%: reported as 1280x720 (divided by 2), really 1707x960 in the window's units.
        assert_eq!(logical_screen((1280, 720), 1.5, true), (1707, 960));
        assert_eq!(logical_screen((1280, 720), 1.25, true), (2048, 1152));
        assert_eq!(logical_screen((1280, 720), 2.0, true), (1280, 720), "a whole-number scale is reported right");
        assert_eq!(logical_screen((2560, 1440), 1.0, true), (2560, 1440));
        assert_eq!(refit((1280, 720), 1.5, true, true), Some((1440, 864)), "fitted again to the real screen");
        assert_eq!(refit((1280, 720), 1.5, true, false), None, "never over a size the user chose");
        assert_eq!(refit((1280, 720), 1.0, true, true), None, "the scale not known yet, or 100%: the first fit stands");
        assert_eq!(refit((2560, 1440), 1.5, true, true), None, "a screen large enough either way");
        // The fit is only made early, on an untouched window, with one screen.
        let (soon, late) = (std::time::Duration::from_millis(400), super::REFIT_WINDOW);
        assert!(super::refit_armed(true, soon, 1));
        assert!(!super::refit_armed(false, soon, 1), "the window has another size: the user's, or the compositor's");
        assert!(!super::refit_armed(true, late, 1), "a scale that changes later is not the first fit's mistake");
        assert!(!super::refit_armed(true, soon, 2), "two screens: which one was reported is not known");
        // X11: the toolkit reports the screen divided by the true scale (1920x1080 at 125% as 1536x864): no correction,
        // or the window would be made larger than the screen.
        assert_eq!(logical_screen((1536, 864), 1.25, false), (1536, 864));
        assert_eq!(logical_screen((1280, 720), 1.5, false), (1280, 720));
        assert_eq!(refit((1536, 864), 1.25, false, true), None);
        assert_eq!(refit((1280, 720), 1.5, false, true), None);
    }
}

#[cfg(test)]
mod hud_region_tests {
    use super::{edit_hud_region, HUD_POSITION_RANGE, HUD_SIZE_RANGE};
    use neural_forge_protocol::{enums::hud_mode, HudRegion, ShmHeader};

    fn region(x: f32, w: f32) -> HudRegion {
        HudRegion { x, y: 0.5, w, h: 0.1, mode: hud_mode::KEEP, strength: HudRegion::DEFAULT_STRENGTH }
    }

    /// An edit the header keeps as it was asks for nothing; one it clips or drops asks for the rows to be rebuilt
    /// (a dropped region moves the others' indices, so the old rows would edit the wrong one).
    #[test]
    fn an_edit_the_header_clips_or_drops_rebuilds_the_rows() {
        let header = ShmHeader::default();
        header.init_defaults();
        let write = |regions: &[HudRegion]| header.set_hud_regions(regions);
        header.set_hud_regions(&[region(0.0, 0.3), region(0.5, 0.2)]);
        let (held, stale) = edit_hud_region(&header, 1, &|r| r.w = 0.25, &write);
        assert!(!stale);
        assert_eq!(held, [region(0.0, 0.3), region(0.5, 0.25)]);
        // Left at 90 % with a width of 30 %: clipped to 10 %, which the row must show.
        let (held, stale) = edit_hud_region(&header, 0, &|r| r.x = 0.9, &write);
        assert!(stale);
        assert_eq!(held[0].w, 1.0 - 0.9f32, "{held:?}");
        // A width of 0: dropped, and the old Region 2 is index 0 now.
        let (held, stale) = edit_hud_region(&header, 0, &|r| r.w = 0.0, &write);
        assert!(stale);
        assert_eq!(held, [region(0.5, 0.25)]);
    }

    /// No value the region rows' spin buttons take can make the header drop the region.
    #[test]
    fn no_spin_value_drops_a_region() {
        for x in [HUD_POSITION_RANGE.0, 50.0, HUD_POSITION_RANGE.1] {
            for w in [HUD_SIZE_RANGE.0, 1.0, HUD_SIZE_RANGE.1] {
                let r = HudRegion {
                    x: (x / 100.0) as f32,
                    y: (x / 100.0) as f32,
                    w: (w / 100.0) as f32,
                    h: (w / 100.0) as f32,
                    mode: hud_mode::SOFTEN,
                    strength: 0.0,
                };
                assert!(r.clamped().is_some(), "{r:?}");
            }
        }
    }
}

#[cfg(test)]
mod spin_precision_tests {
    use super::{decimals, kept_spin_value, override_marked, scope_entries, scope_summary, spin_digits};

    #[test]
    fn the_settings_menu_lists_all_games_then_each_game_once() {
        use neural_forge_supervisor::game_settings::GameKey;
        let steam = [("3240220".to_string(), "Grand Theft Auto V Enhanced".to_string()), ("990080".to_string(), "Hogwarts Legacy".to_string())];
        let with = [GameKey::Steam(990080), GameKey::Exe("other.exe".into()), GameKey::Steam(570)];
        let entries = scope_entries(&steam, &with, &[GameKey::Exe("other.exe".into()), GameKey::Exe("running.exe".into())]);
        let labels: Vec<&str> = entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["All games", "Grand Theft Auto V Enhanced", "Hogwarts Legacy", "other.exe", "Steam app 570", "running.exe"]);
        assert_eq!(entries[0].game, None);
        assert_eq!(entries[2].game, Some(GameKey::Steam(990080)));
    }

    #[test]
    fn only_the_rows_a_picked_game_overrides_are_marked() {
        use neural_forge_supervisor::game_settings::GameKey;
        let own: neural_forge_supervisor::game_settings::Section =
            [("set_intensity", "0.75"), ("set_placement", "1"), ("set_pass2_intensity", "0.5")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let game = GameKey::Steam(3240220);
        let rows = ["intensity", "local_tone", "placement", "pass2_intensity", "pass2_local_tone", "enabled"];
        let marked: Vec<&str> = rows.iter().copied().filter(|name| override_marked(name, Some(&game), &own)).collect();
        assert_eq!(marked, ["intensity", "placement", "pass2_intensity"]);
        assert!(rows.iter().all(|name| !override_marked(name, None, &own)), "All games: no marks");
        assert!(rows.iter().all(|name| !override_marked(name, Some(&game), &Default::default())), "a game without its own: no marks");
    }

    #[test]
    fn the_settings_line_says_what_the_entry_holds() {
        use neural_forge_supervisor::game_settings::GameKey;
        let game = GameKey::Steam(570);
        assert_eq!(scope_summary(None, 0, 0, false), "Every game without settings of its own uses these");
        assert_eq!(scope_summary(Some(&game), 0, 0, false), "Same as All games until you change something here");
        assert_eq!(scope_summary(Some(&game), 1, 0, true), "Running now. 1 setting of its own; the rest follow All games");
        assert_eq!(scope_summary(Some(&game), 3, 2, false), "3 settings of its own, 2 built in; the rest follow All games");
    }

    #[test]
    fn a_row_shows_the_decimals_its_step_and_value_have() {
        assert_eq!(spin_digits(1.0, 16.0), 0);
        assert_eq!(spin_digits(5.0, f64::from(0.66f32) * 100.0), 0);
        assert_eq!(spin_digits(1.0, 16.5), 1);
        assert_eq!(spin_digits(0.1, 1.25), 2);
        assert_eq!(spin_digits(0.05, 0.8), 2);
        assert_eq!(spin_digits(0.01, f64::from(0.004f32)), 3);
        assert_eq!(spin_digits(0.01, f64::from(0.001f32)), 3);
        assert_eq!(decimals(f64::from(0.1f32)), 1);
        assert_eq!(decimals(1.0 / 3.0), 3, "capped");
    }

    #[test]
    fn re_reading_the_shown_text_keeps_the_value() {
        // Text that shows the value: the value, with the decimals the row does not show.
        assert_eq!(kept_spin_value("17", 16.6, 0), Some(16.6));
        assert_eq!(kept_spin_value("1.3", 1.26, 1), Some(1.26));
        assert_eq!(kept_spin_value(" 0.00 ", 0.004, 2), Some(0.004));
        assert_eq!(kept_spin_value("1,3", 1.26, 1), Some(1.26), "a comma decimal mark");
        // Edited text is parsed as a new value.
        assert_eq!(kept_spin_value("18", 16.6, 0), None);
        assert_eq!(kept_spin_value("1.2", 1.26, 1), None);
        assert_eq!(kept_spin_value("", 1.0, 1), None);
        assert_eq!(kept_spin_value("abc", 1.0, 1), None);
    }
}

#[cfg(test)]
mod preupscale_label_tests {
    use super::preupscale_label;

    #[test]
    fn says_where_the_model_runs_in_one_line() {
        assert_eq!(preupscale_label(false, 0, 2, 1485, 836, 3, ""), "no game running");
        assert_eq!(preupscale_label(true, 0, 0, 0, 0, 0, ""), "after the upscaler");
        assert_eq!(preupscale_label(true, 0, 1, 0, 0, 0, ""), "after the upscaler: waiting for DLSS Super Resolution");
        assert_eq!(preupscale_label(true, 0, 2, 1485, 836, 0, ""), "before the upscaler: holding DLSS's 1485x836 input");
        assert_eq!(preupscale_label(true, 0, 2, 1485, 836, 1, ""), "before the upscaler: holding DLSS's 1485x836 input, 1 frame missed");
        assert_eq!(preupscale_label(true, 0, 1, 1485, 836, 85, ""), "after the upscaler: waiting for DLSS Super Resolution, 85 frames missed");
        assert_eq!(preupscale_label(true, 0, 3, 2560, 1440, 0, ""), "after DLSS, before the HUD: holding DLSS's 2560x1440 output");
        assert_eq!(preupscale_label(true, 0, 4, 2560, 1440, 2, ""), "before the upscaler and after DLSS, before the HUD, 2 frames missed");
        let no_wrapper = neural_forge_protocol::enums::preupscale_state::NO_WRAPPER;
        assert!(preupscale_label(true, 0, no_wrapper, 0, 0, 0, "").contains("isn't on the Neural Forge Proton"));
        assert_eq!(
            preupscale_label(true, 0, no_wrapper, 0, 0, 0, "the layer's own reason"),
            "off: the layer's own reason",
            "the layer's reason when it gives one"
        );
    }

    /// Placement "At present" (#49): the model runs on the finished frame by choice, not while waiting for DLSS.
    #[test]
    fn says_at_present_when_the_setting_puts_it_there() {
        let present = neural_forge_protocol::enums::placement::PRESENT;
        for state in [0, 1, 5] {
            assert_eq!(preupscale_label(true, present, state, 0, 0, 0, "why"), "at present, on the finished frame (the Model placement setting)");
        }
        assert_eq!(preupscale_label(false, present, 1, 0, 0, 0, ""), "no game running");
        assert_eq!(preupscale_label(true, 2, 1, 0, 0, 0, ""), "after the upscaler: waiting for DLSS Super Resolution");
    }

    /// DLSS runs with its input identified but every frame is refused (#51): the line says so, and why.
    #[test]
    fn says_why_an_identified_dlss_input_is_not_held() {
        let why = "no hold inside DLSS's buffer: the colour input is 1920x1080, the identified one 1280x720";
        assert_eq!(preupscale_label(true, 0, 5, 1280, 720, 0, why), format!("after the upscaler: DLSS input identified, not holding -- {why}"));
        assert_eq!(
            preupscale_label(true, 0, 5, 1280, 720, 2, why),
            format!("after the upscaler: DLSS input identified, not holding, 2 frames missed -- {why}")
        );
        assert_eq!(preupscale_label(true, 0, 5, 1280, 720, 0, ""), "after the upscaler: DLSS input identified, not holding");
        assert_eq!(preupscale_label(false, 0, 5, 1280, 720, 0, why), "no game running");
        assert_eq!(preupscale_label(true, 0, 1, 0, 0, 0, why), "after the upscaler: waiting for DLSS Super Resolution", "the reason belongs to state 5 only");
    }

    /// The game paused DLSS after holds (a pause menu, #49): the line says the model resumes with it, not that it moved
    /// after the upscaler.
    #[test]
    fn says_paused_while_the_game_pauses_dlss() {
        let paused = neural_forge_protocol::enums::preupscale_state::PAUSED;
        assert_eq!(
            preupscale_label(true, 0, paused, 1188, 891, 0, ""),
            "paused with the game's DLSS (a menu or loading screen): the model resumes with its next frame"
        );
        assert_eq!(preupscale_label(false, 0, paused, 1188, 891, 0, ""), "no game running");
    }
}

#[cfg(test)]
mod capture_tests {
    use super::{capture_subtitle, edits_in_game_frame};

    #[test]
    fn the_capture_row_says_when_there_is_no_before_image() {
        assert!(capture_subtitle(true).contains("before the upscaler or after DLSS"));
        assert!(capture_subtitle(false).contains("two PNGs"));
    }

    /// Holding before the upscaler, after DLSS or both, the presented frame already has the model's edit and the
    /// layer captures it as both images; waiting, DLSS identified but not held, and off capture before and after.
    #[test]
    fn a_capture_has_no_before_image_exactly_while_the_model_edits_the_game_frame() {
        use neural_forge_protocol::enums::preupscale_state::*;
        for state in [HOLDING, HOLDING_AFTER, HOLDING_BOTH, PAUSED] {
            assert!(edits_in_game_frame(state), "state {state}");
        }
        for state in [OFF, WAITING, IDENTIFIED, 99] {
            assert!(!edits_in_game_frame(state), "state {state}");
        }
    }
}

#[cfg(test)]
mod hotkey_tests {
    use super::*;
    #[test]
    fn key_labels_use_the_key_name_and_fall_back_to_the_code() {
        assert_eq!(format_key_label(87, Some("F11")), "F11");
        assert_eq!(format_key_label(30, Some("a")), "A");
        assert_eq!(format_key_label(70, Some("Scroll_Lock")), "Scroll_Lock");
        assert_eq!(format_key_label(87, None), "Key 87");
        assert_eq!(format_key_label(30, Some("")), "Key 30");
    }
}

#[cfg(test)]
mod games_note_tests {
    use super::*;
    use neural_forge_supervisor::steam::Root;

    #[test]
    fn says_why_the_switches_are_off() {
        let native = Root { path: "/s".into(), flatpak: false };
        let flatpak = Root { path: "/f".into(), flatpak: true };
        assert!(games_note(&[], 0, false).starts_with("Steam was not found"));
        assert!(games_note(std::slice::from_ref(&native), 3, true).starts_with("Close Steam"));
        assert!(games_note(std::slice::from_ref(&native), 3, false).starts_with("Turns Neural Forge on"));
        assert!(games_note(std::slice::from_ref(&flatpak), 1, false).starts_with("Steam is the Flatpak build"));
        assert!(!games_note(&[native, flatpak], 1, false).contains("Flatpak"), "a native Steam is there too");
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::{telemetry_ceiling, TelemetrySample};

    #[test]
    fn the_graph_s_top_is_the_smallest_step_above_every_sample_and_ignores_gaps() {
        let s = |frame_ms: Option<f32>, model_ms: Option<f32>| TelemetrySample { frame_ms, model_ms };
        assert_eq!(telemetry_ceiling(&[].into()), 5.0, "empty");
        assert_eq!(telemetry_ceiling(&[s(Some(3.5), Some(1.2)), s(None, None)].into()), 5.0, "a 288 Hz frame");
        assert_eq!(telemetry_ceiling(&[s(Some(5.5), Some(3.1))].into()), 10.0, "180 fps");
        assert_eq!(telemetry_ceiling(&[s(Some(16.7), None)].into()), 20.0, "60 fps");
        assert_eq!(telemetry_ceiling(&[s(Some(19.5), None)].into()), 40.0, "close to a step goes to the next");
        assert_eq!(telemetry_ceiling(&[s(Some(250.0), None)].into()), 80.0, "a stall is capped");
    }
}

#[cfg(test)]
mod uninstall_tests {
    use super::{uninstall_body, uninstall_result};

    #[test]
    fn the_confirmation_says_what_goes_and_what_an_appimage_delete_leaves() {
        let file = std::path::Path::new("/home/u/AppImages/neural-forge.AppImage");
        let body = uninstall_body(false, Some(file));
        assert!(body.contains("*-NeuralForge") && body.contains("this AppImage file, /home/u/AppImages/neural-forge.AppImage"), "{body}");
        assert!(body.contains("Deleting only the AppImage file does not uninstall Neural Forge"), "{body}");
        assert!(!body.contains("Steam is running"));
        assert!(uninstall_body(true, None).contains("Steam is running, so the launch options cannot be changed now"));
    }

    #[test]
    fn the_result_lists_what_steam_kept() {
        let purged = neural_forge_supervisor::install::Purged {
            switched_off: vec!["Some Game (1)".into()],
            left: vec!["app 2: Steam is running, ...; its launch options without Neural Forge: `%command%`".into()],
            removed: vec!["/a".into(), "/b".into()],
            ..Default::default()
        };
        let text = uninstall_result(&purged, None);
        assert!(text.starts_with("Removed 2 files and folders."), "{text}");
        assert!(text.contains("Switched off in the launch options of: Some Game (1).") && text.contains("app 2: Steam is running"), "{text}");
        assert!(uninstall_result(&purged, Some(std::path::Path::new("/x.AppImage"))).contains("Delete /x.AppImage too"));
    }
}
