//! The window's pages: one row per `ShmHeader` setting, grouped and tabbed (Model, Composition, Debug, Status, Setup),
//! the bar that says whose settings they are, and what their buttons do.

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::link::Link;
use gpui_kit::component::switch::Switch;
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Disableable as _, Icon, IconName, Sizable as _, WindowExt as _};
use gpui_kit::Focusable as _;
use gpui_kit::{
    canvas, div, fill, point, px, rgb, size, AnyElement, App, AppContext as _, Bounds, ClipboardItem, Context, Entity, FocusHandle, FontWeight, Hsla,
    InteractiveElement as _, IntoElement, KeyDownEvent, ParentElement as _, PathBuilder, PathPromptOptions, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, WeakEntity, Window,
};
use neural_forge_protocol::mapping::Mapping;
use neural_forge_protocol::{HudRegion, ShmHeader};
use neural_forge_supervisor::model::Network;
use neural_forge_supervisor::{compat_tool, doctor, game_settings, proton_tool, steam};

use crate::text::*;
use crate::ui::{
    ask, dialog_buttons, dialog_room, dialog_title, group, inform, label, menu_button, named_group, row, toast, toast_with_action, with_tooltip, Get,
    InitialFocus, MainView, Question, Spin, Tab, ToastAction, ISSUES, WEBSITE,
};

/// One of the graph's lines: what it plots of a sample.
type Series = fn(&TelemetrySample) -> Option<f32>;

/// The graph's two lines: the frame time and the model's GPU time.
const FRAME_COLOUR: u32 = 0x987af3;
const MODEL_COLOUR: u32 = 0x34c7ad;

/// One installed game as the Setup tab shows it.
pub(crate) struct ScannedGame {
    game: steam::Game,
    enabled: bool,
    /// What switching it on writes as `NEURAL_FORGE_TARGET_EXE` (`steam::target_for`); empty when its executable was
    /// not found.
    target: String,
    /// Whether `target` was chosen by hand (`config.ini`), not found.
    chosen: bool,
    /// The executables in its folder, best first (`TargetFinder::candidates`), for the target editor.
    candidates: Vec<steam::Candidate>,
    /// The anti-cheat its folder ships (`Game::anti_cheat`): switching it on asks first.
    anti_cheat: Vec<&'static str>,
}

/// Steam as the Setup tab shows it, read on a worker thread ([`scan_steam`]): every account's `localconfig.vdf`,
/// Steam's app cache and the games' folders are too slow to read on the UI thread.
pub(crate) struct SteamScan {
    roots: Vec<steam::Root>,
    running: bool,
    games: Vec<ScannedGame>,
}

impl SteamScan {
    /// Whether Steam runs, read again since the scan (the 30 s worker): the switches are offered by it. Whether it changed.
    fn set_running(&mut self, running: bool) -> bool {
        std::mem::replace(&mut self.running, running) != running
    }
}

fn scan_steam() -> SteamScan {
    let cfg = neural_forge_supervisor::Config::load();
    let roots = steam::roots();
    let games = steam::games(&roots);
    let finder = steam::TargetFinder::new(&games);
    let games = games
        .into_iter()
        .map(|game| {
            let enabled = steam::enabled(&game.root.path, &game.appid);
            let usable = game.unsupported().is_none();
            let target = if usable { steam::target_for_in(&cfg, &game, &finder) } else { String::new() };
            let candidates = if usable { finder.candidates(&game) } else { Vec::new() };
            let chosen = steam::chosen_target(&cfg, &game.appid).is_some();
            let anti_cheat = if usable { game.anti_cheat() } else { Vec::new() };
            ScannedGame { game, enabled, target, chosen, candidates, anti_cheat }
        })
        .collect();
    SteamScan { running: steam::steam_running(std::path::Path::new("/proc")), roots, games }
}

/// What the Proton group lists: copies can be made from `sources`; `tools` are the copies, each with whether its
/// source changed since.
pub(crate) struct ProtonScan {
    root: Option<std::path::PathBuf>,
    /// Each source with why the wrapper cannot run under it (`proton_tool::unix_pid_support`, which reads its
    /// `ntdll.so`: read here, on the worker, never while drawing).
    sources: Vec<(proton_tool::Source, Option<String>)>,
    tools: Vec<(proton_tool::Tool, bool)>,
    /// The copies made again (`compat_tool::stale_copies`, the check `doctor` makes): older than their source, or
    /// named without "Proton".
    stale: Vec<compat_tool::StaleCopy>,
}

fn scan_protons() -> ProtonScan {
    let Some(root) = steam::roots().into_iter().find(|r| !r.flatpak).map(|r| r.path) else {
        return ProtonScan { root: None, sources: Vec::new(), tools: Vec::new(), stale: Vec::new() };
    };
    scan_protons_in(root)
}

fn scan_protons_in(root: std::path::PathBuf) -> ProtonScan {
    let libraries = steam::libraries(&root);
    let tools: Vec<(proton_tool::Tool, bool)> = proton_tool::tools(&root).into_iter().map(|t| (t.clone(), proton_tool::outdated(&t))).collect();
    let stale = compat_tool::stale_copies(&root);
    // A source whose current copy exists is not offered again, nor one with a copy Make again makes again (an older
    // version, or a name without "Proton"): that row's Make again is the one way to it.
    let sources = proton_tool::sources(&root, &libraries)
        .into_iter()
        .filter(|s| !tools.iter().any(|(t, _)| t.name == s.tool_name()) && !stale.iter().any(|c| c.tool.source == s.dir))
        .map(|s| {
            let unsupported = proton_tool::unix_pid_support(&s.dir).err();
            (s, unsupported)
        })
        .collect();
    ProtonScan { root: Some(root), sources, tools, stale }
}

/// What the pages keep between draws, apart from the header.
pub(crate) struct PageState {
    /// The model's row text and whether it is installed; the Network row's subtitle for each network.
    model_text: String,
    pub(crate) model_present: bool,
    network_subtitles: [String; 2],
    extracting: bool,
    /// The look-up table `config.ini` names.
    lut: Option<String>,
    mask_rules_open: bool,
    /// The toggle key row waits for a key.
    capturing_key: bool,
    /// A line to the display server's keyboard, for the window's life: the pressed key's hardware code while the row
    /// waits, and the session's keymap.
    key_tap: Option<crate::keytap::KeyTap>,
    /// The session's keyboard layout: a stored code is shown by the key's name on it.
    keys: crate::keymap::SessionKeys,
    /// The layout a key press showed to be in use, for a compositor that does not say.
    pressed_layout: Option<u32>,
    hotkey_focus: FocusHandle,
    steam: Option<SteamScan>,
    scanning: bool,
    /// The games whose launch options are being changed (their app ids).
    busy_games: HashSet<String>,
    /// The launch option group's game (0: "Another game") and executable.
    launch_pick: usize,
    exe_input: Entity<InputState>,
    protons: Option<ProtonScan>,
    proton_busy: bool,
    uninstalling: bool,
    /// The Proton copies out of date: name and what is wrong.
    pub(crate) stale: Vec<(String, String)>,
    profile_input: Entity<InputState>,
    profiles: Vec<String>,
    profile_pick: usize,
    findings: Vec<doctor::Finding>,
    diagnosing: bool,
    saving_report: bool,
    /// The capture request this window is waiting for: its id and when it was made.
    capture_wait: Option<(u32, Instant)>,
    _subs: Vec<Subscription>,
}

impl PageState {
    /// [`SteamScan::set_running`] for the scan shown, if any: whether that changed.
    pub(crate) fn set_steam_running(&mut self, running: bool) -> bool {
        self.steam.as_mut().is_some_and(|scan| scan.set_running(running))
    }

    pub(crate) fn new(window: &mut Window, cx: &mut Context<MainView>) -> Self {
        let exe_input = cx.new(|cx| InputState::new(window, cx).placeholder("Game.exe"));
        let profile_input = cx.new(|cx| InputState::new(window, cx).placeholder("Profile name"));
        let hotkey_focus = cx.focus_handle();
        let subs = vec![
            // The launch option follows the executable typed.
            cx.subscribe_in(&exe_input, window, |_, _, _: &InputEvent, _, cx| cx.notify()),
            // Leaving the toggle key's button ends the wait for a key.
            cx.on_blur(&hotkey_focus, window, |this, _, cx| {
                if std::mem::take(&mut this.pages.capturing_key) {
                    this.stop_listening_for_key();
                    cx.notify();
                }
            }),
        ];
        Self {
            model_text: String::new(),
            model_present: false,
            network_subtitles: [String::new(), String::new()],
            extracting: false,
            lut: None,
            mask_rules_open: false,
            capturing_key: false,
            key_tap: None,
            keys: crate::keymap::SessionKeys::default_layout(),
            pressed_layout: None,
            hotkey_focus,
            steam: None,
            scanning: false,
            busy_games: HashSet::new(),
            launch_pick: 0,
            exe_input,
            protons: None,
            proton_busy: false,
            uninstalling: false,
            stale: Vec::new(),
            profile_input,
            profiles: profile_names(),
            profile_pick: 0,
            findings: Vec::new(),
            diagnosing: false,
            saving_report: false,
            capture_wait: None,
            _subs: subs,
        }
    }
}

fn profile_names() -> Vec<String> {
    let mut names: Vec<String> = neural_forge_supervisor::profiles::load_all().into_keys().collect();
    names.sort();
    names
}

/// How to extract the model where the desktop has no file dialog.
const EXTRACT_WITHOUT_DIALOG: &str = "run `neural-forge-cli extract-model DIR` instead, with the folder that holds nvngx_dlssnr.dll";

/// The model's row text: present with each extracted network and the build it came from (and whether that is
/// verified), or missing.
fn model_status() -> (String, bool) {
    let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    match neural_forge_supervisor::model::installed_text(&root) {
        Some(networks) => (format!("present: {networks}"), true),
        None => ("missing: extract it from nvngx_dlssnr.dll below".to_string(), false),
    }
}

/// The Network row's subtitle for `network`: what it gives, and what runs instead when it is not extracted (the layer
/// runs the other network then).
fn network_subtitle(network: Network) -> String {
    let what = match network {
        Network::Full => "The default: stronger shading and depth",
        Network::Small => "Faster, with a subtler effect",
    };
    let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    let installed = neural_forge_supervisor::model::installed_all(&root);
    match installed.iter().find(|i| i.network != network) {
        Some(other) if !installed.iter().any(|i| i.network == network) => format!("{what}. Not extracted from your DLL, so the {} runs", other.network),
        _ => format!("{what}. A running game switches at once"),
    }
}

/// The float settings with a spin row: name, field, step, and what the row multiplies the value by (100 shows a
/// fraction as a percent). The bounds are `neural_forge_protocol::SETTING_BOUNDS`'.
const FLOAT_SPINS: &[(&str, Get, f64, f64)] = &[
    ("intensity", |h| &h.intensity_bits, 0.05, 1.0),
    ("local_tone", |h| &h.local_tone_bits, 0.05, 1.0),
    ("local_structure", |h| &h.local_structure_bits, 0.05, 1.0),
    ("skin_structure", |h| &h.skin_structure_bits, 0.05, 1.0),
    ("pass_feedback", |h| &h.pass_feedback_bits, 0.05, 1.0),
    ("preupscale_brightness", |h| &h.preupscale_brightness_bits, 0.05, 1.0),
    ("mask_dark_threshold", |h| &h.mask_dark_threshold_bits, 0.01, 1.0),
    ("mask_dark_intensity", |h| &h.mask_dark_intensity_bits, 0.05, 1.0),
    ("mask_dark_structure", |h| &h.mask_dark_structure_bits, 0.05, 1.0),
    ("mask_motion_threshold", |h| &h.mask_motion_threshold_bits, 1.0, 1.0),
    ("mask_motion_intensity", |h| &h.mask_motion_intensity_bits, 0.05, 1.0),
    ("mask_motion_structure", |h| &h.mask_motion_structure_bits, 0.05, 1.0),
    ("transfer_strength", |h| &h.transfer_strength_bits, 0.05, 1.0),
    ("colour_strength", |h| &h.colour_strength_bits, 0.05, 1.0),
    ("max_ratio", |h| &h.max_ratio_bits, 0.1, 1.0),
    ("working_scale", |h| &h.working_scale_bits, 5.0, 100.0),
    ("white_point", |h| &h.white_point_bits, 0.1, 1.0),
    ("white_point_scale", |h| &h.white_point_scale_bits, 0.05, 1.0),
    ("white_point_trim", |h| &h.white_point_trim_bits, 0.05, 1.0),
    ("colour_trust", |h| &h.colour_trust_bits, 0.1, 1.0),
    ("ratio_smooth", |h| &h.ratio_smooth_bits, 5.0, 100.0),
    ("ghost_guard", |h| &h.ghost_guard_bits, 0.05, 1.0),
    ("compare_split", |h| &h.compare_split_bits, 0.05, 1.0),
    ("compare_zoom", |h| &h.compare_zoom_bits, 0.1, 1.0),
    ("debug_scale", |h| &h.debug_scale_bits, 0.1, 1.0),
    ("edit_saturation", |h| &h.edit_saturation_bits, 0.05, 1.0),
    ("edit_contrast", |h| &h.edit_contrast_bits, 0.05, 1.0),
    ("edit_detail", |h| &h.edit_detail_bits, 0.05, 1.0),
    ("edit_hue_protection", |h| &h.edit_hue_protection_bits, 0.05, 1.0),
    ("edit_neutral_protection", |h| &h.edit_neutral_protection_bits, 0.05, 1.0),
    ("edit_shadow_protection", |h| &h.edit_shadow_protection_bits, 0.05, 1.0),
    ("edit_highlight_protection", |h| &h.edit_highlight_protection_bits, 0.05, 1.0),
    ("edit_lut_strength", |h| &h.edit_lut_strength_bits, 0.05, 1.0),
    ("edit_lut_adapt", |h| &h.edit_lut_adapt_bits, 0.01, 1.0),
    ("sharpen_cas", |h| &h.sharpen_cas_bits, 0.05, 1.0),
    ("clarity_strength", |h| &h.clarity_strength_bits, 0.05, 1.0),
    ("clarity_radius", |h| &h.clarity_radius_bits, 1.0, 1.0),
    ("clarity_midtones", |h| &h.clarity_midtones_bits, 0.05, 1.0),
    ("clarity_edges", |h| &h.clarity_edges_bits, 0.05, 1.0),
];

/// The whole-number settings with a spin row.
const INT_SPINS: &[(&str, Get)] = &[("passes", |h| &h.passes), ("model_interval", |h| &h.model_interval)];

/// A HUD region's spin fields: the key's letter, what the field is, its range in percent and the region's value.
type HudField = (&'static str, &'static str, (f64, f64), fn(&HudRegion) -> f32, fn(&mut HudRegion, f32));
const HUD_FIELDS: [HudField; 5] = [
    ("x", "Left, percent of the width", HUD_POSITION_RANGE, |r| r.x, |r, v| r.x = v),
    ("y", "Top, percent of the height", HUD_POSITION_RANGE, |r| r.y, |r, v| r.y = v),
    ("w", "Width, percent", HUD_SIZE_RANGE, |r| r.w, |r, v| r.w = v),
    ("h", "Height, percent", HUD_SIZE_RANGE, |r| r.h, |r, v| r.h = v),
    ("s", "Soften: how much of the edit stays, percent", (0.0, 100.0), |r| r.strength, |r, v| r.strength = v),
];

fn hud_key(region: usize, field: &str) -> String {
    format!("hud{region}{field}")
}

/// Writes the HUD regions to the header as one unit and saves them (`set_hud_regions` in config.ini).
fn write_hud_regions(shm: &Arc<Mapping>, regions: &[HudRegion]) {
    shm.header().set_hud_regions(regions);
    crate::shm::persist_settings_later(shm, &[neural_forge_protocol::persist::HUD_REGIONS]);
}

impl MainView {
    /// The spin fields of the settings rows, once.
    pub(crate) fn add_setting_spins(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for &(name, get, step, scale) in FLOAT_SPINS {
            let (lower, upper) = neural_forge_protocol::setting_bounds(name).unwrap_or_else(|| panic!("{name} has no bounds"));
            let (read_shm, write_shm) = (Arc::clone(&self.shm), Arc::clone(&self.shm));
            self.add_spin(
                name.to_string(),
                Rc::new(move || f64::from(f32::from_bits(get(read_shm.header()).load(Ordering::Relaxed))) * scale),
                Rc::new(move |shown| crate::shm::write_bits(&write_shm, name, get(write_shm.header()), ((shown / scale) as f32).to_bits())),
                (f64::from(lower) * scale, f64::from(upper) * scale),
                step,
                window,
                cx,
            );
        }
        for &(name, get) in INT_SPINS {
            let (lower, upper) = neural_forge_protocol::setting_bounds(name).unwrap_or_else(|| panic!("{name} has no bounds"));
            let (read_shm, write_shm) = (Arc::clone(&self.shm), Arc::clone(&self.shm));
            self.add_spin(
                name.to_string(),
                Rc::new(move || f64::from(get(read_shm.header()).load(Ordering::Relaxed))),
                Rc::new(move |shown| crate::shm::write_bits(&write_shm, name, get(write_shm.header()), shown.round() as u32)),
                (f64::from(lower), f64::from(upper)),
                1.0,
                window,
                cx,
            );
        }
    }

    /// One set of spin fields per HUD region the header holds.
    pub(crate) fn sync_hud_spins(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let count = self.live.regions.len();
        for k in 0..neural_forge_protocol::MAX_HUD_REGIONS {
            for (letter, _, range, read, write) in HUD_FIELDS {
                let key = hud_key(k, letter);
                if k >= count {
                    self.remove_spin(&key);
                } else if !self.spins.contains_key(&key) {
                    let (read_shm, write_shm) = (Arc::clone(&self.shm), Arc::clone(&self.shm));
                    self.add_spin(
                        key,
                        Rc::new(move || read_shm.header().hud_settings().regions.get(k).map_or(f64::NAN, |r| f64::from(read(r)) * 100.0)),
                        // The header clips a region to the frame and drops an empty one: the rows follow it (`changed`).
                        Rc::new(move |shown| {
                            let value = (shown / 100.0) as f32;
                            edit_hud_region(write_shm.header(), k, &move |r: &mut HudRegion| write(r, value), &|regions| {
                                write_hud_regions(&write_shm, regions)
                            });
                        }),
                        range,
                        0.5,
                        window,
                        cx,
                    );
                }
            }
        }
    }

    /// Reads the model's state again (at start and after an extraction).
    pub(crate) fn refresh_model(&mut self) {
        (self.pages.model_text, self.pages.model_present) = model_status();
        self.pages.network_subtitles = [network_subtitle(Network::Full), network_subtitle(Network::Small)];
        self.pages.lut = neural_forge_supervisor::Config::load().settings.get(neural_forge_protocol::cube::CONFIG_KEY).cloned().filter(|p| !p.is_empty());
    }

    /// Runs `work` on a worker thread and `done` with its result here.
    fn run<T: Send + 'static>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        work: impl FnOnce() -> T + Send + 'static,
        done: impl FnOnce(&mut Self, T, &mut Window, &mut Context<Self>) + 'static,
    ) {
        cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_executor().spawn(async move { work() }).await;
            let _ = this.update_in(cx, |this, window, cx| done(this, result, window, cx));
        })
        .detach();
    }

    // ── Row pieces ───────────────────────────────────────────────────────────

    /// The per-game mark of setting `name`'s row: a "This game" tag and a button that drops this one setting of the
    /// game's own, shown while the game picked in "Settings for" has its own value of it.
    fn mark(&self, name: &'static str, cx: &mut Context<Self>) -> Option<AnyElement> {
        let game = self.scope_game();
        if !override_marked(name, game.as_ref(), &self.live.own) {
            return None;
        }
        Some(
            h_flex()
                .gap_1()
                .flex_shrink_0()
                .child(with_tooltip(
                    "this-game",
                    "This game's own value: the All games value does not apply to it",
                    div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(tag_text(cx.theme())).child(label("tag", "This game")),
                ))
                .child(
                    Button::new("follow-all")
                        .ghost()
                        .small()
                        .icon(Icon::new(IconName::Undo2))
                        .tooltip("Use the All games value again")
                        .accessibility_label(format!("Use the All games value of {name} again"))
                        .on_click(cx.listener(move |this, _, window, cx| this.drop_game_override(name, window, cx))),
                )
                .into_any_element(),
        )
    }

    /// A mark's button: drops the picked game's own value of `name` (`game-settings clear GAME NAME`), so the game
    /// follows All games for it again.
    fn drop_game_override(&mut self, name: &'static str, window: &mut Window, cx: &mut Context<Self>) {
        let Some(game) = self.scope_game() else { return };
        // A change still waiting is saved first, so it cannot write the value back.
        if let Err(e) = crate::shm::flush_settings() {
            toast(window, cx, &format!("Saving the settings failed: {e}"));
        }
        match game_settings::clear(&game, Some(&format!("set_{name}"))) {
            Ok(_) => {
                if self.shm.header().settings_game() == game.scope() {
                    neural_forge_protocol::persist::apply_scope(self.shm.header(), Some(&game));
                }
                toast(window, cx, &format!("{name} follows All games again"));
            }
            Err(e) => toast(window, cx, &format!("Not changed: {e}")),
        }
        self.changed(window, cx);
    }

    fn spin_row(&self, name: &'static str, title: &'static str, subtitle: &'static str, cx: &mut Context<Self>) -> AnyElement {
        let field = self.spins.get(name).map(|spin| spin.field(title));
        row(name, title, subtitle, cx).children(self.mark(name, cx)).children(field).into_any_element()
    }

    fn switch_row(&self, name: &'static str, title: &'static str, subtitle: &'static str, get: Get, cx: &mut Context<Self>) -> AnyElement {
        let on = get(self.shm.header()).load(Ordering::Relaxed) != 0;
        row(name, title, subtitle, cx)
            .children(self.mark(name, cx))
            .child(
                Switch::new("switch")
                    .checked(on)
                    .accessibility_label(title)
                    .on_click(cx.listener(move |this, on: &bool, window, cx| this.set_bits(name, get, u32::from(*on), window, cx))),
            )
            .into_any_element()
    }

    /// A menu row for an enum-valued setting. `enabled` false greys the menu out (the value still round-trips).
    #[allow(clippy::too_many_arguments)]
    fn combo_row(
        &self,
        name: &'static str,
        title: &'static str,
        subtitle: &str,
        options: &[&str],
        get: Get,
        enabled: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let last = options.len() - 1;
        let selected = (get(self.shm.header()).load(Ordering::Relaxed) as usize).min(last);
        let view = cx.weak_entity();
        let menu = menu_button(
            "menu",
            title,
            options.iter().map(|o| SharedString::from(o.to_string())).collect(),
            selected,
            enabled,
            Rc::new(move |picked, window, cx| {
                let _ = view.update(cx, |this, cx| this.set_bits(name, get, picked as u32, window, cx));
            }),
        );
        row(name, title, subtitle.to_string(), cx).children(self.mark(name, cx)).child(menu).into_any_element()
    }

    /// A model resolution menu (`enums::model_scale::PRESETS`, then "Custom" for a value set elsewhere, e.g. with
    /// `shmctl`): choosing a preset writes its scale.
    fn scale_row(&self, name: &'static str, title: &'static str, subtitle: &'static str, get: Get, cx: &mut Context<Self>) -> AnyElement {
        use neural_forge_protocol::enums::model_scale::{preset_of, PRESETS};
        let mut names: Vec<SharedString> = PRESETS.iter().map(|&(n, _)| n.into()).collect();
        names.push("Custom".into());
        let selected = preset_of(f32::from_bits(get(self.shm.header()).load(Ordering::Relaxed))).unwrap_or(PRESETS.len());
        let view = cx.weak_entity();
        let menu = menu_button(
            "menu",
            title,
            names,
            selected,
            true,
            Rc::new(move |picked, window, cx| {
                if let Some(&(_, scale)) = PRESETS.get(picked) {
                    let _ = view.update(cx, |this, cx| this.set_bits(name, get, scale.to_bits(), window, cx));
                }
            }),
        );
        row(name, title, subtitle, cx).children(self.mark(name, cx)).child(menu).into_any_element()
    }

    /// The strength preset menu: choosing a preset writes its four values to the header as one unit (`Knobs::write`,
    /// so the layer never sees half of a preset) and saves them. It shows the preset the header matches, else
    /// "Custom".
    fn preset_row(&self, cx: &mut Context<Self>) -> AnyElement {
        use neural_forge_supervisor::presets::{matching, Knobs, KNOBS, PRESETS};
        let mut names: Vec<SharedString> = PRESETS.iter().map(|p| p.name.into()).collect();
        names.push("Custom".into());
        let selected = matching(&Knobs::from_header(self.shm.header())).unwrap_or(PRESETS.len());
        let view = cx.weak_entity();
        let menu = menu_button(
            "menu",
            "Strength preset",
            names,
            selected,
            true,
            Rc::new(move |picked, window, cx| {
                // "Custom" sets nothing.
                let Some(preset) = PRESETS.get(picked) else { return };
                let _ = view.update(cx, |this, cx| {
                    preset.knobs.write(this.shm.header());
                    crate::shm::persist_settings_later(&this.shm, &KNOBS);
                    this.changed(window, cx);
                });
            }),
        );
        row("preset", "Strength preset", "Sets the four values below. Reference is Neural Forge's defaults", cx).child(menu).into_any_element()
    }

    /// Opens the line to the display server's keyboard, once the window is up, and reads the session's keymap.
    pub(crate) fn open_key_tap(&mut self, window: &Window) {
        self.pages.key_tap = crate::keytap::KeyTap::open(window);
        self.read_session_keys();
    }

    /// Asks the display server for the next key press itself (`keytap`).
    fn listen_for_key(&mut self) {
        if let Some(tap) = &mut self.pages.key_tap {
            tap.listen();
        }
        self.read_session_keys();
    }

    fn stop_listening_for_key(&mut self) {
        if let Some(tap) = &mut self.pages.key_tap {
            tap.stop();
        }
    }

    /// The session's keymap and the layout in use now, for the toggle key's name and for finding a key by the
    /// toolkit's name for it: read at start, whenever the window gets the focus (the layout may have been switched
    /// meanwhile, and a Wayland compositor says which is in use only to the focused window) and when a key is waited
    /// for.
    pub(crate) fn read_session_keys(&mut self) {
        let Some(tap) = self.pages.key_tap.as_mut() else { return };
        if let Some(keys) = tap.keys() {
            self.pages.keys = match self.pages.pressed_layout.filter(|_| !tap.layout_known()) {
                Some(layout) => keys.in_layout(layout),
                None => keys,
            };
        }
    }

    /// The toggle key's button: the next key pressed is taken (Esc cancels).
    pub(crate) fn wait_for_toggle_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.pages.capturing_key = true;
        self.listen_for_key();
        // After the click has put the focus on the button itself.
        let focus = self.pages.hotkey_focus.clone();
        window.defer(cx, move |window, cx| focus.focus(window, cx));
        cx.notify();
    }

    /// The toggle key: the button waits for the next key press (Esc cancels), Clear unbinds.
    fn hotkey_row(&self, cx: &mut Context<Self>) -> AnyElement {
        let name = "toggle_key";
        let get: Get = |h| &h.toggle_key;
        let code = get(self.shm.header()).load(Ordering::Relaxed);
        let shown = if self.pages.capturing_key { "Press a key (Esc cancels)".to_string() } else { self.pages.keys.label(code) };
        row(
            name,
            "Toggle key",
            "Toggles Neural Forge inside a running game. Needs read access to /dev/input (the 'input' group) or an X11/XWayland session.",
            cx,
        )
        .children(self.mark(name, cx))
        .child(
            div()
                .id("capture")
                .track_focus(&self.pages.hotkey_focus)
                .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                    if !this.pages.capturing_key {
                        return;
                    }
                    cx.stop_propagation();
                    this.pages.capturing_key = false;
                    let key = event.keystroke.key.as_str();
                    // Which key it was: the display server's own report of the press, else the key that carries the
                    // toolkit's name for it on the session's layout.
                    let pressed = this.pages.key_tap.as_mut().and_then(crate::keytap::KeyTap::last_press);
                    this.stop_listening_for_key();
                    // A press also shows which layout is in use, where the compositor has not said.
                    if let Some(layout) = pressed.and_then(|code| this.pages.keys.layout_of_press(code, key)) {
                        this.pages.pressed_layout = Some(layout);
                        this.read_session_keys();
                    }
                    match pressed {
                        Some(code) => log::info!("toggle key: the display server reported key {code} for \"{key}\""),
                        None => log::info!("toggle key: no direct report of the press; \"{key}\" is looked up on the keymap"),
                    }
                    match pressed.or_else(|| this.pages.keys.code_for(key)) {
                        _ if key == "escape" => cx.notify(),
                        Some(code) => this.set_bits(name, get, code, window, cx),
                        None => {
                            toast(window, cx, &format!("The {key} key cannot be the toggle key: the keyboard layout has no key by that name"));
                            cx.notify();
                        }
                    }
                }))
                .child(
                    Button::new("key")
                        .small()
                        .label(shown.clone())
                        .accessibility_label(format!("Toggle key: {shown}"))
                        .on_click(cx.listener(move |this, _, window, cx| this.wait_for_toggle_key(window, cx))),
                ),
        )
        .child(Button::new("clear").small().label("Clear").on_click(cx.listener(move |this, _, window, cx| {
            this.pages.capturing_key = false;
            this.stop_listening_for_key();
            this.set_bits(name, get, 0, window, cx);
        })))
        .into_any_element()
    }

    // ── The bar under the header ─────────────────────────────────────────────

    /// Whose settings the tabs show and edit ("Settings for": All games or one game) and "Restore defaults" for a
    /// game. The tabs show the header's settings; choosing an entry puts that game's merge on the channel
    /// (`persist::apply_scope`), so the game running shows them while you edit, and every change is saved to the entry
    /// chosen. A game's session starting puts its own settings on the channel, and the menu follows the header.
    pub(crate) fn scope_bar(&self, cx: &mut Context<Self>) -> AnyElement {
        let view = cx.weak_entity();
        let labels: Vec<SharedString> = self.live.entries.iter().map(|e| e.label.clone().into()).collect();
        let menu = menu_button(
            "scope",
            "Settings for",
            labels,
            self.live.scope_at,
            true,
            Rc::new(move |picked, window, cx| {
                let _ = view.update(cx, |this, cx| this.pick_scope(picked, window, cx));
            }),
        );
        v_flex()
            .id("scope-bar")
            .w_full()
            .px_4()
            .py_2()
            .gap_1()
            .border_b_1()
            .border_color(cx.theme().border)
            .child(
                h_flex()
                    .gap_3()
                    .child(div().font_weight(FontWeight::SEMIBOLD).child(label("title", "Settings for")))
                    .child(with_tooltip("scope-tip", "Whose settings the tabs show and change: all games', or one game's own", menu))
                    .child(div().flex_1())
                    .child(
                        Button::new("restore")
                            .small()
                            .label("Restore defaults")
                            .tooltip("Remove this game's own settings: it uses the settings for all games again")
                            .disabled(self.live.own.is_empty())
                            .on_click(cx.listener(|this, _, window, cx| this.restore_defaults_clicked(window, cx))),
                    ),
            )
            .child(div().text_xs().text_color(cx.theme().muted_foreground).child(label("summary", self.live.summary.clone())))
            .into_any_element()
    }

    fn pick_scope(&mut self, picked: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.live.entries.get(picked).cloned() else { return };
        // A change still waiting is saved to the settings it was made in, before the header holds another's.
        if let Err(e) = crate::shm::flush_settings() {
            toast(window, cx, &format!("Saving the settings failed: {e}"));
        }
        neural_forge_protocol::persist::apply_scope(self.shm.header(), entry.game.as_ref());
        self.changed(window, cx);
    }

    fn restore_defaults_clicked(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(entry) = self.live.entries.get(self.live.scope_at).cloned() else { return };
        let Some(game) = entry.game else { return };
        let name = entry.label;
        let view = cx.weak_entity();
        ask(
            Question {
                id: "restore-defaults",
                title: format!("Restore the defaults for {name}?").into(),
                body: "Its own settings are removed: it uses the settings for all games again (with any Neural Forge has built in for it).".into(),
                ok_text: "Restore",
                danger: true,
            },
            Box::new(move |window, cx| {
                let _ = view.update(cx, |this, cx| {
                    if let Err(e) = crate::shm::flush_settings() {
                        toast(window, cx, &format!("Saving the settings failed: {e}"));
                    }
                    match game_settings::clear(&game, None) {
                        Ok(_) => {
                            // Shown (and running, if it is the game on the channel) as it now is.
                            if this.shm.header().settings_game() == game.scope() {
                                neural_forge_protocol::persist::apply_scope(this.shm.header(), Some(&game));
                            }
                            toast(window, cx, &format!("Restored the defaults for {name}"));
                        }
                        Err(e) => toast(window, cx, &format!("Not restored: {e}")),
                    }
                    this.changed(window, cx);
                });
            }),
            None,
            window,
            cx,
        );
    }

    // ── Pages ────────────────────────────────────────────────────────────────

    pub(crate) fn page(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let (id, groups) = match self.tab {
            Tab::Model => ("page-model", vec![self.model_group(cx), self.colour_group(cx), self.hud_group(cx), self.sharpen_group(cx)]),
            Tab::Composition => ("page-composition", vec![self.composition_group(cx), self.white_point_group(cx)]),
            Tab::Debug => ("page-debug", vec![self.debug_group(cx), self.capture_group(cx)]),
            Tab::Status => {
                let mut groups = Vec::new();
                groups.extend(self.stale_proton_group(cx));
                groups.extend([self.telemetry_group(cx), self.status_group(cx), self.diagnosis_group(cx)]);
                ("page-status", groups)
            }
            Tab::Setup => (
                "page-setup",
                vec![self.setup_model_group(cx), self.games_group(cx), self.launch_option_group(cx), self.proton_group(cx), self.uninstall_group(cx)],
            ),
        };
        div()
            .id(id)
            .flex_1()
            .min_h_0()
            .w_full()
            .track_scroll(&self.scroll[self.tab as usize])
            .overflow_y_scroll()
            .child(v_flex().w_full().max_w(px(1100.)).mx_auto().px_6().py_5().gap_6().children(groups))
            .into_any_element()
    }

    fn model_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut rows = vec![
            self.switch_row("enabled", "Neural rendering", "Off keeps the layer running but presents the original frame", |h| &h.enabled, cx),
            self.combo_row("style", "Style", "", &["Default", "Natural", "Cinematic"], |h| &h.style, true, cx),
            self.preset_row(cx),
            self.spin_row("intensity", "Intensity", "How strongly the model's answer replaces the frame", cx),
            self.spin_row("local_tone", "Local tone", "", cx),
            self.spin_row("local_structure", "Local structure", "", cx),
            self.spin_row("skin_structure", "Skin structure", "-1 follows local structure", cx),
            self.switch_row("auto_mask", "Auto mask", "Automatic skin/detail masking", |h| &h.auto_mask, cx),
            self.spin_row(
                "passes",
                "Passes",
                "How many times the model runs over each frame. Every pass after the first works on the previous pass's picture and costs the model's time again",
                cx,
            ),
            self.spin_row(
                "pass_feedback",
                "Pass feedback",
                "How much of a pass's picture the next pass works on: 1 is that picture, lower keeps more of the pass's own input",
                cx,
            ),
        ];
        // The Placement menu: where the model runs in the frame. Its subtitle says what the choice does.
        let placement = self.shm.header().placement.load(Ordering::Relaxed) as usize;
        rows.push(self.combo_row("placement", "Placement", PLACEMENTS.get(placement).map_or("", |&(_, d)| d), &PLACEMENT_NAMES, |h| &h.placement, true, cx));
        rows.push(self.scale_row(
            "preupscale_scale",
            "Model resolution",
            "The size the model runs at before the upscaler, of DLSS's input size. Full gives the most detail; the others are faster, and the model's edit is brought back to full size",
            |h| &h.preupscale_scale_bits,
            cx,
        ));
        // What the model sees before the upscaler: the picture's brightness for it alone (the game's is kept).
        rows.push(self.combo_row(
            "preupscale_exposure",
            "Model brightness",
            "How bright the picture the model works on is (the game's own brightness is kept). Game: the game's exposure. Auto: measured from the frame (within half to 16 times the game's), for games whose picture makes the model darken it",
            &["Game", "Auto"],
            |h| &h.preupscale_exposure,
            true,
            cx,
        ));
        rows.push(self.spin_row(
            "preupscale_brightness",
            "Model brightness scale",
            "On top of the above: above 1 the model sees a brighter picture, below 1 a darker one",
            cx,
        ));
        rows.push(
            row("per-pass", "Per-pass settings", "Give individual passes their own values", cx)
                .child(Button::new("edit").small().label("Edit…").on_click(cx.listener(|this, _, window, cx| open_pass_dialog(&this.shm, window, cx))))
                .into_any_element(),
        );
        // The control mask (before the upscaler): the sky rule, and the optional dark and motion rules.
        rows.push(self.switch_row(
            "mask_sky",
            "Keep the game's sky",
            "No edit where the game draws its sky (from the depth, before the upscaler)",
            |h| &h.mask_sky,
            cx,
        ));
        let open = self.pages.mask_rules_open;
        rows.push(
            row("mask-more", "More mask rules", "Edit less in dark or fast-moving parts of the picture (before the upscaler)", cx)
                .child(
                    Button::new("expand")
                        .ghost()
                        .small()
                        .icon(Icon::new(if open { IconName::ChevronDown } else { IconName::ChevronRight }))
                        .accessibility_label(if open { "Hide the mask rules" } else { "Show the mask rules" })
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.pages.mask_rules_open = !this.pages.mask_rules_open;
                            cx.notify();
                        })),
                )
                .into_any_element(),
        );
        if open {
            let inner = [
                self.switch_row("mask_dark", "Dark areas", "Below the brightness, fade the edit toward the values below", |h| &h.mask_dark, cx),
                self.spin_row("mask_dark_threshold", "Dark brightness", "0 is black, 1 white", cx),
                self.spin_row("mask_dark_intensity", "Dark intensity", "Of the intensity, at black", cx),
                self.spin_row("mask_dark_structure", "Dark structure", "Of the structure, at black", cx),
                self.switch_row("mask_motion", "Fast motion", "Above the motion, fade the edit toward the values below", |h| &h.mask_motion, cx),
                self.spin_row("mask_motion_threshold", "Motion", "Pixels per frame where the fade starts (full at twice)", cx),
                self.spin_row("mask_motion_intensity", "Motion intensity", "Of the intensity, at twice the motion", cx),
                self.spin_row("mask_motion_structure", "Motion structure", "Of the structure, at twice the motion", cx),
            ];
            rows.extend(inner.into_iter().map(|inner| div().pl_6().child(inner).into_any_element()));
        }
        // Model-every-Nth-frame applies after the upscaler only: shown while a game runs and the model is not running
        // before the upscaler (as is the Composition tab's model resolution).
        if self.live.after_upscaler {
            rows.push(self.spin_row(
                "model_interval",
                "Model every Nth frame",
                "1 = every frame. With frame generation on, 2 lets generated frames reuse the last answer instead of waiting for their own",
                cx,
            ));
        }
        rows.push(self.hotkey_row(cx));
        group("model", "Model", None, None, rows, cx)
    }

    /// The Model tab's Colour group: the colour of the network's edit (its change to the frame) for every pass (the
    /// per-pass dialog overrides the three scales), the gamut guard, the dither after the upscaler, the protections of
    /// each pass's result against its input (hue, near grey, shadows, highlights) and the look-up table, whose `.cube`
    /// file is named by `edit_lut` in `config.ini` (the layer reads it itself).
    fn colour_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let lut = self.pages.lut.clone().unwrap_or_else(|| "None: choose a .cube file (17 to 65 points)".to_string());
        let rows = vec![
            self.spin_row(
                "edit_saturation",
                "Saturation of the change",
                "Scales how much the model changes colour: 0 keeps the frame's colours, above 1 adds to the model's",
                cx,
            ),
            self.spin_row("edit_contrast", "Contrast of the change", "Scales how much the model brightens or darkens: 0 keeps the frame's brightness", cx),
            self.spin_row(
                "edit_detail",
                "Detail of the change",
                "Scales the fine part of the change against its broad part: 0 keeps only the broad change",
                cx,
            ),
            self.switch_row(
                "edit_gamut_guard",
                "Gamut guard",
                "Where a change would push a colour past what the frame can hold, shorten the change instead of clipping a channel (keeps the hue)",
                |h| &h.edit_gamut_guard,
                cx,
            ),
            self.switch_row(
                "edit_dither",
                "Dither",
                "Against banding in smooth gradients after the upscaler: a fixed pattern of half a step before the 8-bit answer is rounded",
                |h| &h.edit_dither,
                cx,
            ),
            self.spin_row(
                "edit_hue_protection",
                "Hue protection",
                "Keeps the frame's hues: the model may still change how colourful and how bright a colour is (0 off, 1 full)",
                cx,
            ),
            self.spin_row("edit_neutral_protection", "Grey protection", "Keeps greys and near-greys free of a tint from the model (0 off, 1 full)", cx),
            self.spin_row(
                "edit_shadow_protection",
                "Shadow protection",
                "Holds back the model darkening what was already dark, so shadow detail stays visible (0 off, 1 full)",
                cx,
            ),
            self.spin_row(
                "edit_highlight_protection",
                "Highlight protection",
                "Holds back the model brightening what was already bright, so highlights do not wash out (0 off, 1 full)",
                cx,
            ),
            row("lut", "Look-up table", lut, cx)
                .child(Button::new("choose").small().label("Choose…").on_click(cx.listener(|this, _, window, cx| this.choose_lut(window, cx))))
                .child(
                    Button::new("clear").small().icon(Icon::new(IconName::Close)).tooltip("No look-up table").accessibility_label("No look-up table").on_click(
                        cx.listener(|this, _, window, cx| {
                            this.set_lut(None, window, cx);
                        }),
                    ),
                )
                .into_any_element(),
            self.spin_row("edit_lut_strength", "Look-up table strength", "0 is off. Applied where the model changed the picture, on its last pass", cx),
            self.spin_row(
                "edit_lut_adapt",
                "Look-up table full at",
                "How large a change gets the table's full strength (0-1 of the picture's range); smaller changes get less, an unchanged pixel none",
                cx,
            ),
        ];
        group(
            "colour",
            "Colour",
            Some("Shapes the model's change to the frame, not the frame: 1 leaves the change as the model made it".into()),
            None,
            rows,
            cx,
        )
    }

    /// Names the look-up table's file in config.ini and tells the layer to read it again. False when saving failed.
    fn set_lut(&mut self, path: Option<String>, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let mut cfg = neural_forge_supervisor::Config::load();
        match &path {
            Some(p) => cfg.settings.insert(neural_forge_protocol::cube::CONFIG_KEY.to_string(), p.clone()),
            None => cfg.settings.remove(neural_forge_protocol::cube::CONFIG_KEY),
        };
        if let Err(e) = cfg.save() {
            toast(window, cx, &format!("Saving config.ini failed: {e}"));
            return false;
        }
        self.shm.header().edit_lut_seq.fetch_add(1, Ordering::Relaxed);
        self.shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
        self.pages.lut = path;
        cx.notify();
        true
    }

    /// Asks for one file with the desktop's file dialog and hands it to `picked`. The dialog has no file-type filter
    /// in this toolkit: what was picked is checked afterwards. Where the desktop has no file dialog to offer (no
    /// FileChooser portal), the message ends with `without`, the way to do the same without one.
    fn pick_file(
        &mut self,
        prompt: &'static str,
        without: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
        picked: fn(&mut Self, std::path::PathBuf, &mut Window, &mut Context<Self>),
    ) {
        let paths = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some(prompt.into()) });
        cx.spawn_in(window, async move |this, cx| {
            let result = paths.await;
            let _ = this.update_in(cx, |this, window, cx| match result {
                Ok(Ok(Some(paths))) => {
                    if let Some(path) = paths.into_iter().next() {
                        picked(this, path, window, cx);
                    }
                }
                Ok(Ok(None)) | Err(_) => {}
                Ok(Err(e)) => toast(window, cx, &no_file_dialog(&format!("{e:#}"), without)),
            });
        })
        .detach();
    }

    fn choose_lut(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let without = "name the file in config.ini instead (edit_lut=/path/to/table.cube)";
        self.pick_file("Select a .cube look-up table", without, window, cx, |this, path, window, cx| {
            let path = path.to_string_lossy().into_owned();
            match neural_forge_protocol::cube::load(&path) {
                Ok(cube) => {
                    if this.set_lut(Some(path), window, cx) {
                        toast(window, cx, &format!("Look-up table set: {} points a side. Raise its strength to use it", cube.size));
                    }
                }
                Err(e) => toast(window, cx, &format!("Not a usable look-up table: {e}")),
            }
        });
    }

    /// The HUD on the finished frame (the Placement "Finished frame", or where the model runs there anyway): the
    /// detection and the user's regions, a list of rectangles in percent of the frame.
    fn hud_group(&self, cx: &mut Context<Self>) -> AnyElement {
        use neural_forge_protocol::enums::hud_mode;
        let mut rows = vec![
            self.switch_row(
                "hud_detect",
                "Find the HUD",
                "Keep the edit off what stays still and sharp while the picture moves (text, icons, the minimap)",
                |h| &h.hud_detect,
                cx,
            ),
            row(
                "regions",
                "HUD regions",
                "Rectangles in percent of the frame (left, top, width, height): Keep has no edit inside, Soften scales it by the strength",
                cx,
            )
            .child(Button::new("add").small().label("Add region").disabled(self.live.regions.len() >= neural_forge_protocol::MAX_HUD_REGIONS).on_click(
                cx.listener(|this, _, window, cx| {
                    let mut regions = this.shm.header().hud_settings().regions;
                    // A strip along the bottom edge, where many games put their HUD; edited from there.
                    regions.push(HudRegion { x: 0.0, y: 0.88, w: 0.3, h: 0.12, mode: hud_mode::KEEP, strength: HudRegion::DEFAULT_STRENGTH });
                    write_hud_regions(&this.shm, &regions);
                    this.changed(window, cx);
                }),
            ))
            .into_any_element(),
        ];
        for (k, region) in self.live.regions.iter().enumerate() {
            let mut line = h_flex()
                .id(("region", k))
                .w_full()
                .px_3()
                .py_2()
                .gap_2()
                .flex_wrap()
                .child(div().w(px(80.)).child(label("title", format!("Region {}", k + 1))));
            let spin = |letter: &str| self.spins.get(&hud_key(k, letter));
            for (letter, tip, ..) in &HUD_FIELDS[..4] {
                if let Some(spin) = spin(letter) {
                    line = line.child(with_tooltip(SharedString::from(format!("tip-{letter}")), *tip, spin.field(format!("Region {}: {tip}", k + 1))));
                }
            }
            let view = cx.weak_entity();
            line = line.child(menu_button(
                "mode",
                "Mode",
                vec!["Keep".into(), "Soften".into()],
                usize::from(region.mode == hud_mode::SOFTEN),
                true,
                Rc::new(move |picked, window, cx| {
                    let _ = view.update(cx, |this, cx| {
                        let mode = if picked == 1 { hud_mode::SOFTEN } else { hud_mode::KEEP };
                        edit_hud_region(this.shm.header(), k, &move |r: &mut HudRegion| r.mode = mode, &|regions| write_hud_regions(&this.shm, regions));
                        this.changed(window, cx);
                    });
                }),
            ));
            if let Some(spin) = spin("s") {
                line = line.child(with_tooltip("tip-s", HUD_FIELDS[4].1, spin.field(format!("Region {}: {}", k + 1, HUD_FIELDS[4].1))));
            }
            line = line.child(
                Button::new("remove")
                    .small()
                    .icon(Icon::new(IconName::Minus))
                    .tooltip("Remove this region")
                    .accessibility_label(format!("Remove region {}", k + 1))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        let mut regions = this.shm.header().hud_settings().regions;
                        if k < regions.len() {
                            regions.remove(k);
                        }
                        write_hud_regions(&this.shm, &regions);
                        this.changed(window, cx);
                    })),
            );
            rows.push(line.into_any_element());
        }
        group(
            "hud",
            "HUD on the finished frame",
            Some(
                "Only where the model runs on the presented frame, HUD included (before the upscaler and after DLSS the HUD is not drawn yet). The regions also keep the sharpening off the HUD"
                    .into(),
            ),
            None,
            rows,
            cx,
        )
    }

    /// The sharpening of the finished frame (`crate::sharpen` in the layer): FidelityFX CAS and Clarity, both off at
    /// 0, applied to every presented frame after everything else, the HUD regions kept out of it.
    fn sharpen_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let rows = vec![
            self.spin_row(
                "sharpen_cas",
                "Adaptive sharpening (CAS)",
                "AMD FidelityFX CAS: sharpens fine detail, less where it is already sharp (0 off, about 0.7 CAS's own default, 1 strongest)",
                cx,
            ),
            self.spin_row(
                "clarity_strength",
                "Clarity",
                "Local contrast of brightness: lifts mid-size detail and texture without changing colours (0 off)",
                cx,
            ),
            self.spin_row("clarity_radius", "Clarity radius", "The size of the detail Clarity lifts, in pixels at 1440p (scaled with the resolution)", cx),
            self.spin_row("clarity_midtones", "Clarity midtones", "Keeps Clarity off the darkest and brightest parts (0 evenly, 1 midtones only)", cx),
            self.spin_row(
                "clarity_edges",
                "Clarity edge protection",
                "Stops Clarity at strong edges, against bright or dark halos around them (0 off, 1 most)",
                cx,
            ),
        ];
        group(
            "sharpening",
            "Sharpening",
            Some("On the finished frame, after the model, in every game; the HUD regions above stay untouched. 0 is off".into()),
            None,
            rows,
            cx,
        )
    }

    fn composition_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut rows = vec![
            self.spin_row("transfer_strength", "Detail strength", "How much of the model's edit reaches the frame; above 1 amplifies it", cx),
            self.spin_row("colour_strength", "Colour strength", "How much of the transfer is allowed to be colour, not just luminance", cx),
            self.spin_row("max_ratio", "Highlight guard", "The most the pass may brighten or darken a pixel by (x)", cx),
        ];
        if self.live.after_upscaler {
            rows.push(self.spin_row(
                "working_scale",
                "Model resolution after the upscaler",
                "Percent of the frame the model works at; lower is faster, softer and flickers more",
                cx,
            ));
        }
        rows.extend([
            self.combo_row(
                "reversible_mode",
                "Reversible mode",
                "",
                &["Knee", "Neutwo", "Neutwo replace", "Hybrid", "Hybrid replace"],
                |h| &h.reversible_mode,
                true,
                cx,
            ),
            // The layer does not read `hdr_mode` or `colour_mode` yet. Kept so the saved value round-trips.
            self.combo_row("hdr_mode", "HDR input", "Unavailable: not used by the layer yet", &["Auto", "Off", "Force float16"], |h| &h.hdr_mode, false, cx),
            self.combo_row(
                "colour_mode",
                "Colour mode",
                "Unavailable: not used by the layer yet",
                &["Auto", "Force display-referred", "Force linear HDR"],
                |h| &h.colour_mode,
                false,
                cx,
            ),
            self.combo_row(
                "transfer",
                "Transfer mode",
                "How the answer comes back when model resolution is below 100% (identical at 100%). Native + edit keeps text and edges sharpest",
                &["Classic", "Matched residual", "Native + edit"],
                |h| &h.transfer,
                true,
                cx,
            ),
            self.spin_row(
                "colour_trust",
                "Colour trust",
                "How far the model may change a pixel's colour; larger changes (edge fringing) are shortened. 0 = no model colour",
                cx,
            ),
            self.spin_row(
                "ratio_smooth",
                "Ratio smoothing",
                "Percent of the relighting taken from the neighbourhood rather than each pixel; removes speckle",
                cx,
            ),
            self.spin_row(
                "ghost_guard",
                "Ghost guard",
                "Only matters in pipelined mode (NEURAL_FORGE_PIPELINED=1): hides a late answer's detail where the frame has moved",
                cx,
            ),
            self.switch_row(
                "apply_model",
                "Apply model edit",
                "Off presents the clean frame — capture/transport/round-trip still run, for an honest A/B",
                |h| &h.apply_model,
                cx,
            ),
            self.switch_row(
                "hold_frame",
                "Hold frame",
                "Freeze the frame the pass works on, to re-run composition over the same picture",
                |h| &h.hold_frame,
                cx,
            ),
        ]);
        group("composition", "Composition", None, None, rows, cx)
    }

    fn white_point_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let rows = vec![
            self.combo_row("white_point_source", "White point source", "", &["Manual", "Measured"], |h| &h.white_point_source, true, cx),
            self.spin_row("white_point", "Manual white point", "Linear-light reference", cx),
            self.spin_row("white_point_scale", "White point scale", "Multiplier", cx),
            self.spin_row("white_point_trim", "White point trim", "Calibration multiplier", cx),
        ];
        group(
            "white-point",
            "White point",
            Some(
                "What the model is shown as white. Manual uses the slider; Measured reads the frame's own white level (so dark scenes reach the model well exposed) times the trim. Both are times the scale."
                    .into(),
            ),
            None,
            rows,
            cx,
        )
    }

    fn debug_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let rows = vec![
            self.combo_row("compare_mode", "Compare mode", "", &["Off", "Side by side", "Wipe"], |h| &h.compare_mode, true, cx),
            self.spin_row("compare_split", "Compare split", "Wipe position, 0=left edge, 1=right edge", cx),
            self.spin_row("compare_zoom", "Compare zoom", "", cx),
            self.switch_row("compare_swap", "Swap compare sides", "", |h| &h.compare_swap, cx),
            self.combo_row(
                "debug_view",
                "Debug view",
                "Colour trust's two only differ from the composited picture where colour trust is actually engaged. The edit cache's view replaces the frames that reuse the network's last result before the upscaler: red where it was rejected, green by how much of its detail was kept, blue where the motion left the frame",
                &[
                    "Off",
                    "Original / proxy",
                    "Model's raw answer",
                    "Amplified diff",
                    "Colour trust (green passes, red held back)",
                    "Pre-colour-trust composite",
                    "Edit cache: rejected (before the upscaler)",
                ],
                |h| &h.debug_view,
                true,
                cx,
            ),
            // The edit cache's counters (before the upscaler: the network on every frame while the camera is still,
            // every second frame while it moves).
            row("edit-cache", "Edit cache", self.live.cache.clone(), cx).into_any_element(),
            // Scene cuts before the upscaler: frames where the picture changed as a whole, so the model started afresh
            // instead of blending in the previous scene.
            row("scene-cuts", "Scene cuts", self.live.cuts.clone(), cx).into_any_element(),
            self.spin_row(
                "debug_scale",
                "Debug view 5 amplification",
                "Multiplies the pre-colour-trust colour view 5 shows, to make a subtle difference easier to see",
                cx,
            ),
        ];
        group("debug", "Compare and debug", None, None, rows, cx)
    }

    /// The Debug page's captures: the same one-frame request as `neural-forge-cli shmctl capture`, and the folder the
    /// layer writes them to.
    fn capture_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let captures_dir = neural_forge_supervisor::paths::captures_dir();
        // The game's layer serves the request (`neural_forge_layer::series`) and publishes what became of it
        // (`neural_forge_protocol::capture`), which the message reports.
        let capture = Button::new("capture")
            .small()
            .label(if self.pages.capture_wait.is_some() { "Cancel" } else { "Capture" })
            .tooltip(
                "Saves the next frame the game presents as <time>-original.png (the game's frame) and <time>-composited.png (what Neural Forge presented). While the model runs in the game's frame (before the upscaler or after DLSS, in a DLSS game), the layer has no frame without the model's edit when the game presents, so both images are the presented frame. HDR and 10-bit games cannot be captured.",
            )
            .on_click(cx.listener(|this, _, window, cx| this.capture_clicked(window, cx)));
        let dir = captures_dir.to_string();
        let rows = vec![
            row("capture", "Capture before/after", capture_subtitle(self.live.capture_in_game_frame), cx).child(capture).into_any_element(),
            row("folder", "Captures folder", captures_dir.to_string(), cx)
                .child(Button::new("open").small().label("Open").on_click(move |_, window, cx| open_folder(std::path::Path::new(&dir), window, cx)))
                .into_any_element(),
        ];
        group("captures", "Captures", None, None, rows, cx)
    }

    fn capture_clicked(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        use neural_forge_protocol::capture;
        let header = self.shm.header();
        // A second click while waiting cancels.
        if let Some((id, _)) = self.pages.capture_wait.take() {
            if capture::cancel(header, id) == capture::Cancelled::Withdrawn {
                toast(window, cx, "Capture cancelled");
            }
            cx.notify();
            return;
        }
        let id = match capture::request(header, 1) {
            Ok(id) => id,
            Err(e) => {
                toast(window, cx, &format!("Not captured: {e}"));
                return;
            }
        };
        self.pages.capture_wait = Some((id, Instant::now()));
        cx.notify();
        cx.spawn_in(window, async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_millis(250)).await;
            let waiting = this.update_in(cx, |this, window, cx| this.capture_poll(id, window, cx));
            if !matches!(waiting, Ok(true)) {
                break;
            }
        })
        .detach();
    }

    /// One look at capture `id`'s state; false once it is over (written, refused, withdrawn or cancelled here).
    fn capture_poll(&mut self, id: u32, window: &mut Window, cx: &mut Context<Self>) -> bool {
        use neural_forge_protocol::capture;
        let Some((_, since)) = self.pages.capture_wait.filter(|&(waiting, _)| waiting == id) else { return false };
        let (message, folder) = {
            let header = self.shm.header();
            let elapsed = since.elapsed();
            if let Some(outcome) = capture::result(header).filter(|o| o.id == id && o.finished()) {
                (outcome.message(), outcome.state == capture::state::WRITTEN)
            } else if capture::pending(header).is_some_and(|r| r.id == id)
                && elapsed >= CAPTURE_TAKE_WAIT
                && capture::cancel(header, id) == capture::Cancelled::Withdrawn
            {
                // Nobody took it: withdrawn, so it can never fire on a game started later.
                ("Not captured: no game took the capture. Start a game with Neural Forge on and wait until it is past its loading screen".to_string(), false)
            } else if elapsed >= CAPTURE_RESULT_WAIT {
                ("Not captured: the game did not finish the capture (its log may say why)".to_string(), false)
            } else {
                return true;
            }
        };
        self.pages.capture_wait = None;
        if folder {
            let dir = neural_forge_supervisor::paths::captures_dir().to_string();
            let open: ToastAction = Rc::new(move |window, cx| open_folder(std::path::Path::new(&dir), window, cx));
            toast_with_action(window, cx, &message, "Open folder", open);
        } else {
            toast(window, cx, &message);
        }
        cx.notify();
        false
    }

    // ── Status ───────────────────────────────────────────────────────────────

    /// The Status tab's notice for Neural Forge Proton copies older than the Proton they were made from, or named
    /// without "Proton" so Steam Cloud skips their games' saves in the prefix: one row per copy. Not there while every
    /// copy is current.
    fn stale_proton_group(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        if self.pages.stale.is_empty() {
            return None;
        }
        let warning = cx.theme().warning;
        let rows = self
            .pages
            .stale
            .iter()
            .enumerate()
            .map(|(i, (name, what))| {
                h_flex()
                    .child(div().pl_3().text_color(warning).child(Icon::new(IconName::TriangleAlert)))
                    .child(row(("stale", i), name.clone(), what.clone(), cx))
                    .into_any_element()
            })
            .collect();
        Some(group(
            "stale-proton",
            "Proton copies out of date",
            Some(
                "A game running with one of these keeps an older Proton's files, or cannot sync its Steam Cloud saves. Close Steam and use Make again in the Setup tab."
                    .into(),
            ),
            None,
            rows,
            cx,
        ))
    }

    fn telemetry_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let t = &self.telemetry;
        let muted = cx.theme().muted_foreground;
        let ceiling = telemetry_ceiling(&t.history);
        let history: Vec<TelemetrySample> = t.history.iter().copied().collect();
        let grid: Hsla = muted.opacity(0.3);
        // The newest sample at the right edge; a gap where a series has no value.
        let graph = canvas(
            |_, _, _| (),
            move |bounds: Bounds<gpui_kit::Pixels>, _, window, _| {
                let (width, height) = (bounds.size.width, bounds.size.height);
                for frac in [0.0, 0.5, 1.0] {
                    let y = bounds.origin.y + (height - px(1.)) * frac;
                    window.paint_quad(fill(Bounds::new(point(bounds.origin.x, y), size(width, px(1.))), grid));
                }
                let n = TELEMETRY_SAMPLES.max(2);
                let start = n - history.len().min(n);
                let series: [(Series, u32); 2] = [(|s| s.frame_ms, FRAME_COLOUR), (|s| s.model_ms, MODEL_COLOUR)];
                for (pick, colour) in series {
                    let mut path = PathBuilder::stroke(px(1.6));
                    let (mut drawing, mut segments) = (false, 0);
                    for (i, sample) in history.iter().enumerate() {
                        let x = bounds.origin.x + width * ((start + i) as f32 / (n - 1) as f32);
                        match pick(sample) {
                            Some(v) => {
                                let y = bounds.origin.y + (height - px(1.)) * (1.0 - v.min(ceiling) / ceiling);
                                if drawing {
                                    path.line_to(point(x, y));
                                    segments += 1;
                                } else {
                                    path.move_to(point(x, y));
                                }
                                drawing = true;
                            }
                            None => drawing = false,
                        }
                    }
                    if segments > 0 {
                        if let Ok(path) = path.build() {
                            window.paint_path(path, rgb(colour));
                        }
                    }
                }
            },
        )
        .flex_1()
        .h(px(80.));
        let key = |id: &'static str, text: String, colour: u32| h_flex().gap_2().child(div().size(px(10.)).bg(rgb(colour))).child(label(id, text));
        let scale = v_flex().h(px(80.)).w(px(44.)).justify_between().text_xs().text_color(muted).children([
            label("top", format!("{ceiling} ms")),
            label("middle", format!("{} ms", ceiling / 2.0)),
            label("bottom", "0 ms"),
        ]);
        let card = v_flex()
            .id("graph")
            .p_3()
            .gap_2()
            .child(h_flex().gap_4().child(key("frame", t.frame_label.clone(), FRAME_COLOUR)).child(key("model", t.model_label.clone(), MODEL_COLOUR)))
            .child(h_flex().gap_2().child(scale).child(graph));
        let rows = vec![
            row("model", "Model", t.model, cx).into_any_element(),
            row("game", "Game", t.game.clone(), cx).into_any_element(),
            row("fps", "Frame rate", t.fps.clone(), cx).into_any_element(),
            card.into_any_element(),
        ];
        group("telemetry", "Telemetry", Some("Live from the layer -- all zero until a targeted game attaches".into()), None, rows, cx)
    }

    /// The Status group: the layer's liveness and where the model runs, plus the model, reset and profile actions.
    fn status_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let (tested, notes) = self.live.tested.clone();
        // The running game's entry in the tested games: its setup notes are the tooltip.
        let tested_row = row("tested", "Tested game", tested, cx);
        let tested_row = match notes {
            Some(notes) => with_tooltip("tested-notes", notes, tested_row),
            None => tested_row.into_any_element(),
        };
        let profiles = &self.pages.profiles;
        let profile_names: Vec<SharedString> =
            if profiles.is_empty() { vec!["No saved profiles".into()] } else { profiles.iter().map(|p| p.clone().into()).collect() };
        let view = cx.weak_entity();
        let rows = vec![
            row("layer", "Layer", self.live.layer, cx).into_any_element(),
            // Where the model runs: before DLSS's upscaler or after it. Read-only; the Placement setting asks.
            row("placement", "Model placement", self.live.placement.clone(), cx).into_any_element(),
            tested_row,
            // The HDR verdict: what the game asked for and why (the layer's), and the desktop's HDR state.
            row("hdr", "HDR", self.live.hdr.clone(), cx).into_any_element(),
            row("model", "Model", self.pages.model_text.clone(), cx).into_any_element(),
            row("settings", "Settings", "Reset every tuning value to its default", cx)
                .child(Button::new("reset").small().danger().label("Reset…").on_click(cx.listener(|this, _, window, cx| this.reset_clicked(window, cx))))
                .into_any_element(),
            row("save-profile", "Save current as", "", cx)
                .child(div().w(px(260.)).child(Input::new(&self.pages.profile_input).small()))
                .child(Button::new("save").small().primary().label("Save").on_click(cx.listener(|this, _, window, cx| this.save_profile(window, cx))))
                .into_any_element(),
            row("load-profile", "Load profile", "", cx)
                .child(menu_button(
                    "profiles",
                    "Load profile",
                    profile_names,
                    self.pages.profile_pick.min(profiles.len().saturating_sub(1)),
                    !profiles.is_empty(),
                    Rc::new(move |picked, _, cx| {
                        let _ = view.update(cx, |this, cx| {
                            this.pages.profile_pick = picked;
                            cx.notify();
                        });
                    }),
                ))
                .child(Button::new("load").small().label("Load").on_click(cx.listener(|this, _, window, cx| this.load_profile(window, cx))))
                .child(Button::new("delete").small().danger().label("Delete").on_click(cx.listener(|this, _, window, cx| this.delete_profile(window, cx))))
                .into_any_element(),
        ];
        group("status", "Status", None, None, rows, cx)
    }

    fn reset_clicked(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let view = cx.weak_entity();
        ask(
            Question {
                id: "reset",
                title: "Reset all settings?".into(),
                body: "Every tuning value of the settings shown (Settings for, at the top) returns to its default. The running layer session (frame counters, transport state) is not affected."
                    .into(),
                ok_text: "Reset",
                danger: true,
            },
            Box::new(move |window, cx| {
                let _ = view.update(cx, |this, cx| {
                    // Reset the live mapping the running layer is attached to, then overwrite the saved settings with
                    // the same defaults, so a restart doesn't reload the values this just cleared.
                    this.shm.header().reset_persisted_settings();
                    match neural_forge_supervisor::save_settings(this.shm.header()) {
                        Ok(()) => toast(window, cx, "Settings reset"),
                        Err(e) => toast(window, cx, &format!("Reset the live session, but saving the settings failed: {e}")),
                    }
                    this.changed(window, cx);
                });
            }),
            None,
            window,
            cx,
        );
    }

    fn selected_profile(&self) -> Option<String> {
        self.pages.profiles.get(self.pages.profile_pick.min(self.pages.profiles.len().saturating_sub(1))).cloned()
    }

    fn refresh_profiles(&mut self, cx: &mut Context<Self>) {
        self.pages.profiles = profile_names();
        self.pages.profile_pick = self.pages.profile_pick.min(self.pages.profiles.len().saturating_sub(1));
        cx.notify();
    }

    fn save_profile(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let name = self.pages.profile_input.read(cx).value().trim().to_string();
        if let Some(why) = neural_forge_supervisor::profiles::invalid_name(&name) {
            toast(window, cx, &format!("Not saved: {why}"));
            return;
        }
        let settings = neural_forge_protocol::persist::snapshot(self.shm.header());
        match neural_forge_supervisor::profiles::save_profile(&name, settings) {
            Ok(()) => {
                toast(window, cx, &format!("Saved profile \"{name}\""));
                self.pages.profile_input.update(cx, |input, cx| input.set_value("", window, cx));
                self.refresh_profiles(cx);
                self.pages.profile_pick = self.pages.profiles.iter().position(|p| *p == name).unwrap_or(self.pages.profile_pick);
            }
            Err(e) => toast(window, cx, &format!("Save failed: {e}")),
        }
    }

    fn load_profile(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(name) = self.selected_profile() else {
            toast(window, cx, "No profile selected");
            return;
        };
        let profiles = neural_forge_supervisor::profiles::load_all();
        let Some(settings) = profiles.get(&name) else {
            toast(window, cx, &format!("Profile \"{name}\" is gone"));
            self.refresh_profiles(cx);
            return;
        };
        neural_forge_protocol::persist::apply(self.shm.header(), settings);
        // Applying to the live header only affects the running session, so also save the result (to the settings the
        // header holds: all games' or the game's) so it survives a reboot too.
        let message = match neural_forge_supervisor::save_settings(self.shm.header()) {
            Ok(()) => format!("Loaded profile \"{name}\""),
            Err(e) => format!("Applied to the running session, but saving the settings failed: {e}"),
        };
        toast(window, cx, &message);
        self.changed(window, cx);
    }

    fn delete_profile(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(name) = self.selected_profile() else {
            toast(window, cx, "No profile selected");
            return;
        };
        match neural_forge_supervisor::profiles::delete_profile(&name) {
            Ok(true) => toast(window, cx, &format!("Deleted profile \"{name}\"")),
            Ok(false) => toast(window, cx, "Profile already gone"),
            Err(e) => toast(window, cx, &format!("Delete failed: {e}")),
        }
        self.refresh_profiles(cx);
    }

    /// The Diagnosis group: the Diagnose button runs `neural-forge-cli doctor`'s checks
    /// (`neural_forge_supervisor::doctor`) and lists the findings, worst first. They run on a worker thread:
    /// `nvidia-smi` and `journalctl` can take seconds.
    fn diagnosis_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let actions = h_flex()
            .gap_2()
            .child(
                Button::new("diagnose")
                    .small()
                    .primary()
                    .label(if self.pages.diagnosing { "Diagnosing…" } else { "Diagnose" })
                    .disabled(self.pages.diagnosing)
                    .on_click(cx.listener(|this, _, window, cx| this.diagnose(window, cx))),
            )
            .child(
                Button::new("save-report")
                    .small()
                    .label("Save report")
                    .disabled(self.pages.saving_report)
                    .tooltip(
                        "Saves a zip to the desktop to attach to a problem report: these findings (which list your installed Steam games, their library folders and whether Neural Forge is on in them; of a game's launch options only Neural Forge's part), system details (GPU, driver, kernel, session), settings, the extracted model's manifests, the layer's logs, the kernel's GPU errors and the newest capture. Your home folder, user name and computer name are replaced; nothing is uploaded",
                    )
                    .on_click(cx.listener(|this, _, window, cx| this.save_report(window, cx))),
            );
        let rows = self.pages.findings.iter().enumerate().map(|(i, finding)| finding_row(i, finding, cx)).collect();
        group(
            "diagnosis",
            "Diagnosis",
            Some("Why neural rendering is or is not running: the install, the GPU and driver, the kernel log and the last game session".into()),
            Some(actions.into_any_element()),
            rows,
            cx,
        )
    }

    fn diagnose(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.pages.diagnosing = true;
        cx.notify();
        self.run(
            window,
            cx,
            || doctor::run(&doctor::Roots::system()),
            |this, findings, _, cx| {
                this.pages.findings = findings;
                this.pages.diagnosing = false;
                cx.notify();
            },
        );
    }

    /// "Save report": the diagnostic report (`neural_forge_supervisor::report`) on the desktop.
    fn save_report(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // The channel is read here, on the UI thread; the files and probes on a worker.
        let status = neural_forge_supervisor::shm_status::status_text(self.shm.header());
        let game = self.shm.header().game_name();
        self.pages.saving_report = true;
        cx.notify();
        self.run(
            window,
            cx,
            move || neural_forge_supervisor::report::save(&status, &game, None, None).map_err(|e| e.to_string()),
            |this, saved, window, cx| {
                this.pages.saving_report = false;
                cx.notify();
                match saved {
                    Ok(path) => {
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let dir = path.parent().map(|d| d.display().to_string()).unwrap_or_default();
                        let show: ToastAction = Rc::new(move |_, cx| cx.reveal_path(&path));
                        toast_with_action(window, cx, &format!("Saved {name} to {dir}"), "Show", show);
                    }
                    Err(e) => toast(window, cx, &format!("Saving the report failed: {e}")),
                }
            },
        );
    }

    // ── Setup ────────────────────────────────────────────────────────────────

    /// The native backend's one setup step: point at NVIDIA's DLL once and extract the model from it; and which of
    /// its networks runs (the Network setting, live).
    fn setup_model_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let present = self.pages.model_present;
        let (icon, colour) = if present { (IconName::CircleCheck, cx.theme().success) } else { (IconName::TriangleAlert, cx.theme().warning) };
        let network = Network::from_setting(self.shm.header().network.load(Ordering::Relaxed));
        let subtitle = &self.pages.network_subtitles[usize::from(network == Network::Small)];
        let rows = vec![
            h_flex()
                .child(div().pl_3().text_color(colour).child(Icon::new(icon)))
                .child(row("model", "Model", self.pages.model_text.clone(), cx))
                .into_any_element(),
            // Which network runs: a persisted setting, so a running game switches over at its next frame.
            self.combo_row("network", "Network", subtitle, &NETWORK_CHOICES, |h| &h.network, true, cx),
            row("extract", "Extract from DLL", "Choose nvngx_dlssnr.dll; every network it has is written to Neural Forge's data folder", cx)
                .child(Button::new("extract").small().label("Extract…").disabled(self.pages.extracting).on_click(
                    cx.listener(|this, _, window, cx| this.pick_file("Select nvngx_dlssnr.dll", EXTRACT_WITHOUT_DIALOG, window, cx, Self::extract_model)),
                ))
                .into_any_element(),
        ];
        group(
            "setup-model",
            "Neural rendering model",
            Some("Taken from your own nvngx_dlssnr.dll -- this project doesn't and can't ship it".into()),
            None,
            rows,
            cx,
        )
    }

    /// Extracts every network the DLL at `path` carries into the model directory
    /// ([`neural_forge_supervisor::model::extract_all`]), says how it went and brings the status rows up to date.
    fn extract_model(&mut self, path: std::path::PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        self.pages.extracting = true;
        cx.notify();
        let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
        self.run(
            window,
            cx,
            move || neural_forge_supervisor::model::extract_all(&path, &root),
            |this, result, window, cx| {
                match result {
                    Ok(all) => {
                        let networks: Vec<String> = all
                            .networks
                            .iter()
                            .map(|done| {
                                format!(
                                    "the {} of build {}{}",
                                    done.network,
                                    done.build,
                                    if done.verified { "" } else { " (not verified against NVIDIA's runtime)" }
                                )
                            })
                            .collect();
                        for (network, why) in &all.refused {
                            eprintln!("neural-forge: the DLL's {network} was not extracted: {why}");
                        }
                        let refused = if all.refused.is_empty() { "" } else { " -- a network of the DLL was refused, see the log" };
                        toast(window, cx, &format!("Extracted {}{refused} -- restart the game to use it", networks.join(" and ")));
                    }
                    // The report is too long for a message: its summary line there, all of it in the log.
                    Err(neural_forge_supervisor::model::ModelError::Shape(report)) => {
                        eprintln!("neural-forge: extraction refused:\n{report}");
                        let summary = report.lines().next().unwrap_or_default();
                        toast(window, cx, &format!("Extraction refused, a different network: {summary}"));
                    }
                    Err(e) => toast(window, cx, &format!("Extraction failed: {e}")),
                }
                this.pages.extracting = false;
                this.refresh_model();
                cx.notify();
            },
        );
    }

    /// Reads Steam again, on a worker; the games and launch option groups are filled from it.
    pub(crate) fn scan_steam(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.pages.scanning = true;
        cx.notify();
        self.run(window, cx, scan_steam, |this, scan, window, cx| {
            // The launch option group: a game already switched on first, else the first game, so the string is
            // complete by default.
            let listed: Vec<&ScannedGame> = scan.games.iter().filter(|g| g.game.unsupported().is_none() && !g.target.is_empty()).collect();
            let pick = listed.iter().position(|g| g.enabled).or((!listed.is_empty()).then_some(0));
            this.pages.launch_pick = pick.map_or(0, |i| i + 1);
            if let Some(i) = pick {
                let target = listed[i].target.clone();
                this.pages.exe_input.update(cx, |input, cx| input.set_value(target, window, cx));
            }
            this.pages.steam = Some(scan);
            this.pages.scanning = false;
            this.pages.busy_games.clear();
            cx.notify();
        });
    }

    /// The installed Steam games, each with a switch that turns Neural Forge on or off in its launch options
    /// (`neural_forge_supervisor::steam::apply`) for the game's own executable. While Steam runs the switches are
    /// off limits: it would overwrite the edit on exit. A game of the Flatpak Steam cannot be switched on (its
    /// sandbox does not see the layer) and says so.
    fn games_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let refresh = Button::new("refresh")
            .ghost()
            .small()
            .icon(Icon::new(IconName::RefreshCw))
            .tooltip("Look for games again")
            .accessibility_label("Look for games again")
            .disabled(self.pages.scanning)
            .on_click(cx.listener(|this, _, window, cx| this.scan_steam(window, cx)));
        let Some(scan) = &self.pages.steam else {
            return group("games", "Steam games", Some("Looking for games…".into()), Some(refresh.into_any_element()), Vec::new(), cx);
        };
        let mut rows = Vec::new();
        for (i, scanned) in scan.games.iter().enumerate() {
            let game = &scanned.game;
            let appid = game.appid.clone();
            let busy = self.pages.busy_games.contains(&appid);
            // Only the switch is ever off limits, so the target editor opens for a game whose executable was not found,
            // the one it is there for.
            let usable = !busy && switch_usable(game, &scanned.target, scanned.enabled, scan.running);
            let mut line = row(("game", i), game.name.clone(), game_subtitle(game, &scanned.target, scanned.chosen, &scanned.anti_cheat), cx);
            if game.unsupported().is_none() {
                let appid = appid.clone();
                line = line.child(
                    Button::new("target")
                        .ghost()
                        .small()
                        .icon(Icon::new(gpui_kit::assets::IconName::Pencil))
                        .tooltip("Choose the executable Neural Forge targets for this game")
                        .accessibility_label(format!("Choose the executable Neural Forge targets for {}", game.name))
                        .on_click(cx.listener(move |this, _, window, cx| this.edit_target(&appid, window, cx))),
                );
            }
            line = line.child(
                Switch::new("on")
                    .checked(scanned.enabled)
                    .disabled(!usable)
                    .accessibility_label(format!("Neural Forge in {}", game.name))
                    .on_click(cx.listener(move |this, on: &bool, window, cx| this.game_switched(&appid, *on, window, cx))),
            );
            rows.push(line.into_any_element());
        }
        group("games", "Steam games", Some(games_note(&scan.roots, scan.games.len(), scan.running).into()), Some(refresh.into_any_element()), rows, cx)
    }

    fn scanned(&mut self, appid: &str) -> Option<&mut ScannedGame> {
        self.pages.steam.as_mut()?.games.iter_mut().find(|g| g.game.appid == appid)
    }

    /// A game's switch: switching on a game with anti-cheat asks first; switching off never does.
    fn game_switched(&mut self, appid: &str, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(scanned) = self.scanned(appid) else { return };
        if on && !scanned.anti_cheat.is_empty() {
            let question = Question {
                id: "anti-cheat",
                title: format!("{} ships anti-cheat", scanned.game.name).into(),
                body: neural_forge_supervisor::anti_cheat::warning(&scanned.anti_cheat).into(),
                ok_text: "Switch On Anyway",
                danger: true,
            };
            let (view, appid) = (cx.weak_entity(), appid.to_string());
            ask(
                question,
                Box::new(move |window, cx| {
                    let _ = view.update(cx, |this, cx| this.apply_game(&appid, true, window, cx));
                }),
                None,
                window,
                cx,
            );
        } else {
            self.apply_game(appid, on, window, cx);
        }
    }

    /// Changes the launch options, then the game's Proton (a Neural Forge copy when switched on, made now if needed;
    /// the earlier one back when off): DLSS's placements need it. On a worker thread.
    fn apply_game(&mut self, appid: &str, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(scanned) = self.scanned(appid) else { return };
        scanned.enabled = on;
        let (game, target) = (scanned.game.clone(), scanned.target.clone());
        self.pages.busy_games.insert(appid.to_string());
        cx.notify();
        let action = if on { steam::Action::Enable { target_exe: target } } else { steam::Action::Disable };
        let (root, id) = (game.root.path.clone(), game.appid.clone());
        self.run(
            window,
            cx,
            move || {
                let outcome = steam::apply_now(&root, &id, &action);
                let proton = match (&outcome, on) {
                    (steam::Outcome::Changed(_) | steam::Outcome::Unchanged, true) => {
                        let running = steam::steam_running(std::path::Path::new("/proc"));
                        let wrapper = proton_tool::installed_wrapper();
                        Some(
                            compat_tool::switch_on(&root, &steam::libraries(&root), &id, &wrapper, running)
                                .map(|d| match d {
                                    None => "; a native Linux game: it runs without Proton".to_string(),
                                    Some(d) => format!(
                                        "; it runs with {} now{}{}",
                                        d.tool,
                                        if d.made { " (a copy made for Neural Forge: restart Steam to see it)" } else { "" },
                                        d.note.map(|n| format!(". Note: {n}")).unwrap_or_default()
                                    ),
                                })
                                .map_err(|e| e.to_string()),
                        )
                    }
                    (steam::Outcome::Changed(_) | steam::Outcome::Unchanged, false) => {
                        let running = steam::steam_running(std::path::Path::new("/proc"));
                        Some(
                            compat_tool::switch_off(&root, &id, running)
                                .map(|p| match p {
                                    Some(p) if p.is_empty() => "; it runs with Steam's default Proton again".to_string(),
                                    Some(p) => format!("; it runs with {p} again"),
                                    None => String::new(),
                                })
                                .map_err(|e| e.to_string()),
                        )
                    }
                    _ => None,
                };
                (outcome, proton)
            },
            move |this, (outcome, proton), window, cx| {
                let appid = game.appid.clone();
                this.pages.busy_games.remove(&appid);
                let mut revert = false;
                match outcome {
                    steam::Outcome::Changed(_) | steam::Outcome::Unchanged => {
                        let state = if on { "on" } else { "off" };
                        let note = match proton {
                            Some(Ok(n)) => n,
                            Some(Err(e)) => {
                                format!(". Its Proton was not changed ({e}): in a DLSS game the model stays off until it runs with a Neural Forge Proton")
                            }
                            None => String::new(),
                        };
                        toast(window, cx, &format!("Neural Forge is {state} for {}{note}", game.name));
                    }
                    steam::Outcome::SteamRunning { manual } => {
                        revert = true;
                        // A refused edit finds Steam running for all of them.
                        if let Some(scan) = &mut this.pages.steam {
                            scan.running = true;
                        }
                        manual_toast("Steam is running: close it first, or set the launch options by hand", manual, window, cx);
                    }
                    steam::Outcome::Failed { reason, manual } => {
                        revert = true;
                        manual_toast(&format!("Could not change the launch options safely: {reason}"), manual, window, cx);
                    }
                }
                if revert {
                    if let Some(scanned) = this.scanned(&appid) {
                        scanned.enabled = !on;
                    }
                }
                // A copy made or dropped shows in the Proton group.
                this.scan_protons(window, cx);
                cx.notify();
            },
        );
    }

    /// The per-game target editor: pick one of the executables found in the game's folder (best first) or type names
    /// (comma-separated). Saving remembers it for the game (`config.ini`, as `neural-forge-cli enable --target-exe`
    /// does); "Use the found ones" forgets it. A game that is on gets its launch options rewritten with the new target
    /// at once (Steam closed).
    fn edit_target(&mut self, appid: &str, window: &mut Window, cx: &mut Context<Self>) {
        let Some(scanned) = self.scanned(appid) else { return };
        let (name, target, candidates) = (scanned.game.name.clone(), scanned.target.clone(), scanned.candidates.clone());
        let (view, appid) = (cx.weak_entity(), appid.to_string());
        let editor = cx.new(|cx| TargetEditor {
            title: format!("Target for {name}").into(),
            input: cx.new(|cx| InputState::new(window, cx).placeholder("Game.exe").default_value(target)),
            candidates,
            view,
            appid,
            save_focus: cx.focus_handle(),
            cancel_focus: cx.focus_handle(),
            auto_focus: cx.focus_handle(),
            first: true,
        });
        window.open_dialog(cx, move |dialog, _, cx| {
            let (title, enter) = (editor.read(cx).title.clone(), editor.clone());
            dialog
                .title(dialog_title(title))
                .overlay(false)
                .w(px(560.))
                // Enter: what the focused button says; in the field, Save.
                .on_ok(move |_, window, cx| enter.update(cx, |editor, cx| editor.enter(window, cx)))
                .child(editor.clone())
        });
    }

    /// The target editor's Save (`typed` names) or "Use the found ones" (`None`).
    fn target_chosen(&mut self, appid: &str, typed: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(scanned) = self.scanned(appid) else { return };
        let (game, on) = (scanned.game.clone(), scanned.enabled);
        let mut cfg = neural_forge_supervisor::Config::load();
        let key = steam::chosen_target_key(&game.appid);
        match typed {
            Some(typed) => cfg.settings.insert(key, typed),
            None => cfg.settings.remove(&key),
        };
        if let Err(e) = cfg.save() {
            toast(window, cx, &format!("Saving config.ini failed: {e}"));
            return;
        }
        let worker_game = game.clone();
        self.run(
            window,
            cx,
            move || {
                let cfg = neural_forge_supervisor::Config::load();
                let new = steam::target_for(&cfg, &worker_game);
                let chosen = steam::chosen_target(&cfg, &worker_game.appid).is_some();
                // A game that is on gets the new target in its launch options at once.
                let applied = (on && !new.is_empty())
                    .then(|| steam::apply_now(&worker_game.root.path, &worker_game.appid, &steam::Action::Enable { target_exe: new.clone() }));
                (new, chosen, applied)
            },
            move |this, (new, chosen, applied), window, cx| {
                // A game whose executable was just named can be switched on now, without a refresh.
                if let Some(scanned) = this.scanned(&game.appid) {
                    scanned.target = new.clone();
                    scanned.chosen = chosen;
                }
                cx.notify();
                match applied {
                    None => toast(
                        window,
                        cx,
                        &format!("{}'s target is {}; it is used when the game is switched on", game.name, if new.is_empty() { "not found" } else { &new }),
                    ),
                    Some(steam::Outcome::Changed(_) | steam::Outcome::Unchanged) => toast(window, cx, &format!("{} now targets {new}", game.name)),
                    Some(steam::Outcome::SteamRunning { manual }) => manual_toast(
                        "Saved, but Steam is running: close it and switch the game off and on, or set the launch options by hand",
                        manual,
                        window,
                        cx,
                    ),
                    Some(steam::Outcome::Failed { reason, manual }) => {
                        manual_toast(&format!("Saved, but the launch options could not be changed safely: {reason}"), manual, window, cx)
                    }
                }
            },
        );
    }

    /// The Setup tab's hand-set launch option: pick a game and its executable is filled in (the one the games list
    /// would write), or type one for a game Steam does not list.
    fn launch_option_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let listed: Vec<&ScannedGame> =
            self.pages.steam.iter().flat_map(|scan| scan.games.iter()).filter(|g| g.game.unsupported().is_none() && !g.target.is_empty()).collect();
        let mut names: Vec<SharedString> = vec!["Another game".into()];
        names.extend(listed.iter().map(|g| SharedString::from(g.game.name.clone())));
        let targets: Vec<String> = listed.iter().map(|g| g.target.clone()).collect();
        let view = cx.weak_entity();
        let option = neural_forge_supervisor::launch_options::launch_option(&self.pages.exe_input.read(cx).value());
        let copied = option.clone();
        let rows = vec![
            row("game", "Game", "", cx)
                .child(menu_button(
                    "games",
                    "Game",
                    names,
                    self.pages.launch_pick.min(listed.len()),
                    true,
                    Rc::new(move |picked, window, cx| {
                        let target = picked.checked_sub(1).and_then(|i| targets.get(i).cloned());
                        let _ = view.update(cx, |this, cx| {
                            this.pages.launch_pick = picked;
                            if let Some(target) = target {
                                this.pages.exe_input.update(cx, |input, cx| input.set_value(target, window, cx));
                            }
                            cx.notify();
                        });
                    }),
                ))
                .into_any_element(),
            row("exe", "Target executable", "The game's .exe; several separated by commas", cx)
                .child(div().w(px(360.)).child(Input::new(&self.pages.exe_input).small()))
                .into_any_element(),
            row("option", "Launch option", option, cx)
                .child(Button::new("copy").small().label("Copy").on_click(move |_, _, cx| cx.write_to_clipboard(ClipboardItem::new_string(copied.clone()))))
                .into_any_element(),
        ];
        group(
            "launch-option",
            "Steam launch option",
            Some("To set it by hand: paste this into the game's Properties -> Launch Options in Steam".into()),
            None,
            rows,
            cx,
        )
    }

    pub(crate) fn scan_protons(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.run(window, cx, scan_protons, |this, scan, _, cx| {
            this.pages.protons = Some(scan);
            this.pages.proton_busy = false;
            cx.notify();
        });
    }

    /// Runs one of the Proton group's actions on a worker, says how it went and lists the copies again.
    fn proton_action(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
        failed: &'static str,
        work: impl FnOnce() -> Result<String, String> + Send + 'static,
    ) {
        self.pages.proton_busy = true;
        cx.notify();
        self.run(window, cx, work, move |this, done, window, cx| {
            toast(window, cx, &done.unwrap_or_else(|e| format!("{failed}: {e}")));
            this.scan_protons(window, cx);
        });
    }

    /// "Proton with DLSS's parameters": copies of installed Protons that load the NGX wrapper
    /// (`neural_forge_supervisor::proton_tool`), made and removed here, with the notice.
    fn proton_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let busy = self.pages.proton_busy;
        let mut rows = Vec::new();
        match &self.pages.protons {
            None => {}
            Some(ProtonScan { root: None, .. }) => rows.push(row("none", "No native Steam installation found", "", cx).into_any_element()),
            Some(ProtonScan { root: Some(root), sources, tools, stale }) => {
                for (i, (tool, outdated)) in tools.iter().enumerate() {
                    let subtitle = if let Some(stale) = stale.iter().find(|s| s.tool.name == tool.name) {
                        // (A copy renamed or merged in place says what happens to its games itself.)
                        if stale.in_place() {
                            format!("Out of date: {}. Close Steam and make it again", stale.describe())
                        } else {
                            format!("Out of date: {}. Close Steam and make it again: its games move to the new copy", stale.describe())
                        }
                    } else if *outdated {
                        format!("from {}; Neural Forge's NGX wrapper has changed since: close Steam and make it again", tool.source.display())
                    } else {
                        format!("from {}", tool.source.display())
                    };
                    let mut line = row(("tool", i), tool.name.clone(), subtitle, cx);
                    if *outdated {
                        let (root, tool) = (root.clone(), tool.clone());
                        line = line.child(Button::new("again").small().label("Make again").disabled(busy).on_click(cx.listener(move |this, _, window, cx| {
                            let (root, tool) = (root.clone(), tool.clone());
                            this.proton_action(window, cx, "Could not make the copy", move || {
                                let running = steam::steam_running(std::path::Path::new("/proc"));
                                compat_tool::update(&root, &tool, &proton_tool::installed_wrapper(), running)
                                    .map(|u| {
                                        let note = u.note.as_ref().map(|n| format!(". Note: {n}")).unwrap_or_default();
                                        if u.to == u.from {
                                            format!("Made {} again{note}", u.to)
                                        } else if u.renamed {
                                            format!("Renamed {} to {}, its games moved with it{note}. Restart Steam to see it", u.from, u.to)
                                        } else {
                                            let kept = u.kept.map(|k| format!("; {} is kept: {k}", u.from)).unwrap_or_else(|| format!("; removed {}", u.from));
                                            if u.merged {
                                                format!("Moved its games to {}{kept}{note}. Restart Steam to see it", u.to)
                                            } else {
                                                format!("Made {}, its games moved to it{kept}{note}. Restart Steam to see it", u.to)
                                            }
                                        }
                                    })
                                    .map_err(|e| e.to_string())
                            });
                        })));
                    }
                    let (root, name) = (root.clone(), tool.name.clone());
                    line =
                        line.child(Button::new("remove").small().danger().label("Remove").disabled(busy).on_click(cx.listener(move |this, _, window, cx| {
                            let (root, name) = (root.clone(), name.clone());
                            this.proton_action(window, cx, "Could not remove it", move || {
                                proton_tool::remove(&root, &name)
                                    .map(|t| format!("Removed {}: games set to it need another Proton in Steam", t.name))
                                    .map_err(|e| e.to_string())
                            });
                        })));
                    rows.push(line.into_any_element());
                }
                for (i, (source, unsupported)) in sources.iter().enumerate() {
                    // A Proton the wrapper cannot run under (Wine older than 10) says why instead of offering a copy.
                    let subtitle = match unsupported {
                        Some(why) => format!("{} in {}: cannot be copied, {why}", source.version, source.dir.display()),
                        None => format!("{} in {}", source.version, source.dir.display()),
                    };
                    let (root, source_for_click) = (root.clone(), source.clone());
                    rows.push(
                        row(("source", i), source.display.clone(), subtitle, cx)
                            .child(
                                Button::new("make")
                                    .small()
                                    .label("Make copy")
                                    .tooltip(format!("Make {} (see the notice above)", source.tool_name()))
                                    .disabled(busy || unsupported.is_some())
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        let (root, source) = (root.clone(), source_for_click.clone());
                                        this.proton_action(window, cx, "Could not make the copy", move || {
                                            proton_tool::create(&root, &source, &proton_tool::installed_wrapper())
                                                .map(|m| {
                                                    format!("Made {}: restart Steam, then choose it for a game in its Properties > Compatibility", m.tool.name)
                                                })
                                                .map_err(|e| e.to_string())
                                        });
                                    })),
                            )
                            .into_any_element(),
                    );
                }
            }
        }
        group("proton", "Proton with DLSS parameters", Some(proton_tool::NOTICE.into()), None, rows, cx)
    }

    /// The Uninstall group (Setup tab): one destructive button, a confirmation saying what goes, then the result; the
    /// app quits afterwards (its files are gone).
    fn uninstall_group(&self, cx: &mut Context<Self>) -> AnyElement {
        let rows = vec![row(
            "uninstall",
            "Uninstall Neural Forge",
            "Remove the layer, settings, model, Proton builds and launch options it added, and this AppImage",
            cx,
        )
        .child(
            Button::new("uninstall")
                .small()
                .danger()
                .label("Uninstall…")
                .disabled(self.pages.uninstalling)
                .on_click(cx.listener(|this, _, window, cx| this.uninstall_clicked(window, cx))),
        )
        .into_any_element()];
        group("uninstall", "Uninstall", None, None, rows, cx)
    }

    pub(crate) fn uninstall_clicked(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let appimage = std::env::var_os("APPIMAGE").map(std::path::PathBuf::from).filter(|p| p.is_file());
        let steam_running = steam::steam_running(std::path::Path::new("/proc"));
        let view = cx.weak_entity();
        ask(
            Question {
                id: "uninstall",
                title: "Uninstall Neural Forge?".into(),
                body: uninstall_body(steam_running, appimage.as_deref()).into(),
                ok_text: "Uninstall",
                danger: true,
            },
            Box::new(move |window, cx| {
                let _ = view.update(cx, |this, cx| this.uninstall(appimage, window, cx));
            }),
            None,
            window,
            cx,
        );
    }

    fn uninstall(&mut self, appimage: Option<std::path::PathBuf>, window: &mut Window, cx: &mut Context<Self>) {
        self.pages.uninstalling = true;
        cx.notify();
        let target = appimage.clone();
        self.run(
            window,
            cx,
            move || {
                neural_forge_supervisor::install::purge_with(neural_forge_supervisor::install::Leftovers::Report, target.as_deref()).map_err(|e| e.to_string())
            },
            move |this, done, window, cx| match done {
                Ok(purged) => {
                    let kept = appimage.filter(|f| f.exists());
                    let body = uninstall_result(&purged, kept.as_deref());
                    inform("uninstalled", "Neural Forge is uninstalled".into(), body.into(), "Quit", Some(Box::new(|_, cx| cx.quit())), window, cx);
                }
                Err(e) => {
                    this.pages.uninstalling = false;
                    cx.notify();
                    let body = format!("{e}\n\nWhat was removed before this stays removed; press Uninstall again once it is fixed.");
                    inform("uninstall-stopped", "Uninstall stopped".into(), body.into(), "Close", None, window, cx);
                }
            },
        );
    }
}

/// A message with a Copy button for `manual`, the launch options to set by hand.
fn manual_toast(message: &str, manual: String, window: &mut Window, cx: &mut App) {
    let copy: ToastAction = Rc::new(move |_, cx| cx.write_to_clipboard(ClipboardItem::new_string(manual.clone())));
    toast_with_action(window, cx, message, "Copy launch options", copy);
}

/// Opens `dir` in the file manager (`xdg-open`, off the UI thread), creating it first if it is missing; says so when
/// that fails.
fn open_folder(dir: &std::path::Path, window: &mut Window, cx: &mut App) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        toast(window, cx, &format!("Couldn't create {}: {e}", dir.display()));
        return;
    }
    let dir = dir.to_path_buf();
    window
        .spawn(cx, async move |cx| {
            let opened = cx.background_executor().spawn(async move { open_with_file_manager(&dir, "xdg-open") }).await;
            if let Err(why) = opened {
                let _ = cx.update(|window, cx| toast(window, cx, &format!("Couldn't open the folder: {why}")));
            }
        })
        .detach();
}

/// Runs `opener DIR` and waits for it: an error when it cannot be started or says it failed.
fn open_with_file_manager(dir: &std::path::Path, opener: &str) -> Result<(), String> {
    let status = std::process::Command::new(opener)
        .arg(dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("{opener} could not be started ({e})"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{opener} failed ({status})"))
    }
}

/// One finding as a row: the severity's icon, the summary as the title, the evidence and the fix below; the check's
/// name is the tooltip.
fn finding_row(index: usize, finding: &doctor::Finding, cx: &App) -> AnyElement {
    use doctor::Severity;
    let subtitle = match &finding.fix {
        Some(fix) => format!("{}\nFix: {fix}", finding.evidence),
        None => finding.evidence.clone(),
    };
    let (icon, colour, severity) = match finding.severity {
        Severity::Failure => (IconName::CircleX, cx.theme().danger, "Failure"),
        Severity::Warning => (IconName::TriangleAlert, cx.theme().warning, "Warning"),
        Severity::Info => (IconName::Info, cx.theme().muted_foreground, "Note"),
        Severity::Ok => (IconName::CircleCheck, cx.theme().success, "Passed"),
    };
    with_tooltip(
        ("finding-tip", index),
        finding.check,
        h_flex().child(div().pl_3().text_color(colour).child(Icon::new(icon))).child(row(
            ("finding", index),
            format!("{severity}: {}", finding.summary),
            subtitle,
            cx,
        )),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
//  The target editor's dialog
// ─────────────────────────────────────────────────────────────────────────────

struct TargetEditor {
    title: SharedString,
    input: Entity<InputState>,
    candidates: Vec<steam::Candidate>,
    view: WeakEntity<MainView>,
    appid: String,
    save_focus: FocusHandle,
    cancel_focus: FocusHandle,
    auto_focus: FocusHandle,
    first: bool,
}

impl TargetEditor {
    /// The names typed, tidied: comma-separated, no blanks.
    fn typed(&self, cx: &App) -> String {
        self.input.read(cx).value().split(',').map(str::trim).filter(|n| !n.is_empty()).collect::<Vec<_>>().join(",")
    }

    fn save(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let typed = self.typed(cx);
        if typed.is_empty() {
            toast(window, cx, "Not saved: name at least one executable");
            return;
        }
        self.finish(Some(typed), window, cx);
    }

    fn finish(&mut self, typed: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        window.close_dialog(cx);
        let (view, appid) = (self.view.clone(), self.appid.clone());
        window.defer(cx, move |window, cx| {
            let _ = view.update(cx, |this, cx| this.target_chosen(&appid, typed, window, cx));
        });
    }

    /// Enter: Cancel and "Use the found ones" when they have the focus, Save on its button or in the field; false
    /// (the dialog stays) anywhere else.
    fn enter(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.cancel_focus.is_focused(window) {
            return true;
        }
        if self.auto_focus.is_focused(window) {
            self.finish(None, window, cx);
        } else if self.save_focus.is_focused(window) || self.input.read(cx).focus_handle(cx).is_focused(window) {
            self.save(window, cx);
        }
        false
    }
}

impl Render for TargetEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if std::mem::take(&mut self.first) {
            self.save_focus.focus(window, cx);
        }
        let mut body = v_flex()
            .id("body")
            .max_h(dialog_room(window))
            .overflow_y_scroll()
            .gap_2()
            .child(label("text", "Only these executables can use Neural Forge in this game (several separated by commas)"));
        if !self.candidates.is_empty() {
            body = body.child(label("found", "Found in the game's folder, best first:"));
        }
        for (i, c) in self.candidates.iter().take(10).enumerate() {
            let name = c.name.clone();
            body = body.child(
                Button::new(("pick", i))
                    .ghost()
                    .small()
                    .label(format!("{}  ({}, {} MB){}", c.name, c.path, c.bytes >> 20, if c.chosen { "  found" } else { "" }))
                    .on_click(cx.listener(move |this, _, window, cx| this.input.update(cx, |input, cx| input.set_value(name.clone(), window, cx)))),
            );
        }
        body = body.child(Input::new(&self.input));
        named_group("target-editor", self.title.clone()).child(body).child(
            dialog_buttons()
                .child(Button::new("cancel").label("Cancel").track_focus(&self.cancel_focus).on_click(|_, window, cx| window.close_dialog(cx)))
                .child(
                    Button::new("auto")
                        .label("Use the found ones")
                        .track_focus(&self.auto_focus)
                        .on_click(cx.listener(|this, _, window, cx| this.finish(None, window, cx))),
                )
                .child(
                    Button::new("save")
                        .primary()
                        .label("Save")
                        .track_focus(&self.save_focus)
                        .on_click(cx.listener(|this, _, window, cx| this.save(window, cx))),
                ),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Per-pass settings
// ─────────────────────────────────────────────────────────────────────────────

/// One field of the per-pass dialog.
struct PassField {
    title: &'static str,
    /// The setting's name after `pass<N>_`.
    field: &'static str,
    bit: u32,
    kind: PassFieldKind,
    /// The pass's field (pass 0 is pass 1).
    get: fn(&ShmHeader, usize) -> &std::sync::atomic::AtomicU32,
}

enum PassFieldKind {
    Float {
        step: f64,
    },
    Choice(&'static [&'static str]),
    Toggle,
    /// A model resolution preset (`enums::model_scale::PRESETS`), stored as its scale.
    Scale,
}

use neural_forge_protocol::enums::pass_override as po;

const PASS_FIELDS: &[PassField] = &[
    PassField { title: "Intensity", field: "intensity", bit: po::INTENSITY, kind: PassFieldKind::Float { step: 0.05 }, get: |h, k| &h.pass[k].intensity_bits },
    PassField {
        title: "Local tone",
        field: "local_tone",
        bit: po::LOCAL_TONE,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass[k].local_tone_bits,
    },
    PassField {
        title: "Local structure",
        field: "local_structure",
        bit: po::LOCAL_STRUCTURE,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass[k].local_structure_bits,
    },
    PassField {
        title: "Skin structure",
        field: "skin_structure",
        bit: po::SKIN_STRUCTURE,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass[k].skin_structure_bits,
    },
    PassField {
        title: "Style",
        field: "style",
        bit: po::STYLE,
        kind: PassFieldKind::Choice(&["Default", "Natural", "Cinematic"]),
        get: |h, k| &h.pass[k].style,
    },
    PassField { title: "Auto mask", field: "auto_mask", bit: po::AUTO_MASK, kind: PassFieldKind::Toggle, get: |h, k| &h.pass[k].auto_mask },
    PassField { title: "Model resolution", field: "scale", bit: po::SCALE, kind: PassFieldKind::Scale, get: |h, k| &h.pass_scale_bits[k] },
    PassField {
        title: "Edit saturation",
        field: "edit_saturation",
        bit: po::EDIT_SATURATION,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass_colour[k].saturation_bits,
    },
    PassField {
        title: "Edit contrast",
        field: "edit_contrast",
        bit: po::EDIT_CONTRAST,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass_colour[k].contrast_bits,
    },
    PassField {
        title: "Edit detail",
        field: "edit_detail",
        bit: po::EDIT_DETAIL,
        kind: PassFieldKind::Float { step: 0.05 },
        get: |h, k| &h.pass_colour[k].detail_bits,
    },
    PassField { title: "Placement", field: "placement", bit: po::PLACEMENT, kind: PassFieldKind::Choice(&PLACEMENT_NAMES), get: |h, k| &h.pass_placement[k] },
];

/// Writes settings of a pass, as one unit (so the layer never reads half of a change), and saves them.
fn write_pass_settings(shm: &Arc<Mapping>, settings: &[(&'static str, u32)]) {
    if shm.header().write_settings(settings) {
        let names: Vec<&'static str> = settings.iter().map(|&(n, _)| n).collect();
        crate::shm::persist_settings_later(shm, &names);
    }
}

/// Per-pass overrides: each pass can replace any of the Model tab's values with its own; a pass that overrides
/// nothing follows the Model tab. Saved as `set_pass<N>_*` like any other setting.
struct PassDialog {
    shm: Arc<Mapping>,
    /// The pass shown (0 is pass 1).
    pass: Rc<Cell<usize>>,
    pass_spin: Spin,
    /// The value field of each float field of `PASS_FIELDS`.
    spins: Vec<Option<Spin>>,
    close_focus: Rc<InitialFocus>,
    _subs: Vec<Subscription>,
}

impl PassDialog {
    fn new(shm: Arc<Mapping>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let max = neural_forge_protocol::MAX_PASSES;
        let pass = Rc::new(Cell::new(0usize));
        let mut subs = Vec::new();
        let (read_pass, write_pass) = (pass.clone(), pass.clone());
        let (pass_spin, pass_subs) = Spin::new(
            window,
            cx,
            Rc::new(move || (read_pass.get() + 1) as f64),
            Rc::new(move |shown| write_pass.set((shown.round() as usize).saturating_sub(1).min(max - 1))),
            (1.0, max as f64),
            1.0,
            Self::edited,
        );
        subs.extend(pass_subs);
        let mut spins = Vec::new();
        for field in PASS_FIELDS {
            let PassFieldKind::Float { step } = field.kind else {
                spins.push(None);
                continue;
            };
            let (lower, upper) = neural_forge_protocol::setting_bounds(field.field).unwrap_or_else(|| panic!("{} has no bounds", field.field));
            let (read_shm, write_shm, read_pass, write_pass) = (Arc::clone(&shm), Arc::clone(&shm), pass.clone(), pass.clone());
            let (get, name) = (field.get, field.field);
            let (spin, spin_subs) = Spin::new(
                window,
                cx,
                Rc::new(move || f64::from(f32::from_bits(get(read_shm.header(), read_pass.get()).load(Ordering::Relaxed)))),
                Rc::new(move |shown| {
                    if let Some(setting) = neural_forge_protocol::pass_setting(write_pass.get(), name) {
                        write_pass_settings(&write_shm, &[(setting, (shown as f32).to_bits())]);
                    }
                }),
                (f64::from(lower), f64::from(upper)),
                step,
                Self::edited,
            );
            subs.extend(spin_subs);
            spins.push(Some(spin));
        }
        Self { shm, pass, pass_spin, spins, close_focus: InitialFocus::new(cx), _subs: subs }
    }

    /// After an edit (another pass picked, a value changed): every field shows the pass's values.
    fn edited(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for spin in self.spins.iter().flatten() {
            spin.sync(false, window, cx);
        }
        cx.notify();
    }

    fn write(&mut self, field: &'static str, bits: u32, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(setting) = neural_forge_protocol::pass_setting(self.pass.get(), field) {
            write_pass_settings(&self.shm, &[(setting, bits)]);
        }
        self.edited(window, cx);
    }
}

impl Render for PassDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.close_focus.apply(window, cx);
        let pass = self.pass.get();
        let header = self.shm.header();
        let mask = header.pass[pass].overrides.load(Ordering::Relaxed);
        let mut rows = Vec::new();
        for (i, field) in PASS_FIELDS.iter().enumerate() {
            let on = mask & field.bit != 0;
            let bit = field.bit;
            rows.push(
                row(("field", i), field.title, "Override for this pass", cx)
                    .child(Switch::new("override").checked(on).accessibility_label(format!("Override {} for this pass", field.title)).on_click(cx.listener(
                        move |this, on: &bool, window, cx| {
                            let mask = this.shm.header().pass[this.pass.get()].overrides.load(Ordering::Relaxed);
                            this.write("overrides", if *on { mask | bit } else { mask & !bit }, window, cx);
                        },
                    )))
                    .into_any_element(),
            );
            if !on {
                continue;
            }
            let bits = (field.get)(header, pass).load(Ordering::Relaxed);
            let name = field.field;
            let view = cx.weak_entity();
            let value = match &field.kind {
                PassFieldKind::Float { .. } => {
                    row(("value", i), "Value", "", cx).children(self.spins[i].as_ref().map(|spin| spin.field(format!("{} of this pass", field.title))))
                }
                PassFieldKind::Choice(options) => row(("value", i), "Value", "", cx).child(menu_button(
                    "menu",
                    field.title,
                    options.iter().map(|o| SharedString::from(*o)).collect(),
                    (bits as usize).min(options.len() - 1),
                    true,
                    Rc::new(move |picked, window, cx| {
                        let _ = view.update(cx, |this, cx| this.write(name, picked as u32, window, cx));
                    }),
                )),
                PassFieldKind::Toggle => row(("value", i), "On", "", cx).child(
                    Switch::new("value")
                        .checked(bits != 0)
                        .accessibility_label(field.title)
                        .on_click(cx.listener(move |this, on: &bool, window, cx| this.write(name, u32::from(*on), window, cx))),
                ),
                PassFieldKind::Scale => {
                    use neural_forge_protocol::enums::model_scale::{preset_of, PRESETS};
                    let mut names: Vec<SharedString> = PRESETS.iter().map(|&(n, _)| n.into()).collect();
                    names.push("Custom".into());
                    row(("value", i), "Value", "Of DLSS's input size; this pass works on the previous pass's full-size picture", cx).child(menu_button(
                        "menu",
                        field.title,
                        names,
                        preset_of(f32::from_bits(bits)).unwrap_or(PRESETS.len()),
                        true,
                        Rc::new(move |picked, window, cx| {
                            if let Some(&(_, scale)) = PRESETS.get(picked) {
                                let _ = view.update(cx, |this, cx| this.write(name, scale.to_bits(), window, cx));
                            }
                        }),
                    ))
                }
            };
            rows.push(div().pl_6().child(value).into_any_element());
        }
        let pick = group(
            "pick",
            "",
            Some(
                "Each pass of the model can override any Model-tab value with its own. A pass that overrides nothing follows the Model tab. Only the passes the Passes setting runs are used. Passes run in order: a pass placed earlier in the frame than the pass before it runs where that one does."
                    .into(),
            ),
            None,
            vec![row("pass", "Pass", "", cx).child(self.pass_spin.field("Pass")).into_any_element()],
            cx,
        );
        named_group("per-pass", PASS_TITLE)
            .child(v_flex().id("body").max_h(dialog_room(window)).overflow_y_scroll().gap_4().child(pick).child(group("fields", "", None, None, rows, cx)))
            .child(
                dialog_buttons()
                    .child(Button::new("close").label("Close").track_focus(&self.close_focus.handle).on_click(|_, window, cx| window.close_dialog(cx))),
            )
    }
}

const PASS_TITLE: &str = "Per-pass settings";

pub(crate) fn open_pass_dialog(shm: &Arc<Mapping>, window: &mut Window, cx: &mut App) {
    let shm = Arc::clone(shm);
    let dialog_view = cx.new(|cx| PassDialog::new(shm, window, cx));
    window.open_dialog(cx, move |dialog, _, _| {
        // Enter belongs to the fields here (it takes a typed value): it never closes the dialog.
        dialog.title(dialog_title(PASS_TITLE)).overlay(false).w(px(640.)).on_ok(|_, _, _| false).child(dialog_view.clone())
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  About
// ─────────────────────────────────────────────────────────────────────────────

const ABOUT_TITLE: &str = "About Neural Forge";

pub(crate) fn show_about_dialog(window: &mut Window, cx: &mut App) {
    let focus = InitialFocus::new(cx);
    window.open_dialog(cx, move |dialog, window, cx| {
        focus.apply(window, cx);
        let muted = cx.theme().muted_foreground;
        let credits = [
            "clshortfuse (RenoDX) — the colour composition design",
            "hhkbble — the matched-residual transfer mode",
            "xenmods (DLSSNR-Cost-Scaler) — the native + edit technique",
            "Dagherbou and cdozdil (OptiScaler) — the DLSS-NR shader lineage",
        ];
        dialog.title(dialog_title(ABOUT_TITLE)).overlay(false).w(px(460.)).on_ok(|_, _, _| true).child(
            named_group("about", ABOUT_TITLE)
                .child(
                    v_flex()
                        .id("body")
                        .max_h(dialog_room(window))
                        .overflow_y_scroll()
                        .gap_2()
                        .items_center()
                        .child(div().text_xl().font_weight(FontWeight::BOLD).child(label("name", "Neural Forge")))
                        .child(div().text_color(muted).child(label("version", env!("CARGO_PKG_VERSION"))))
                        .child(div().text_center().child(label(
                            "summary",
                            "Vulkan layer and settings GUI for running NVIDIA DLSS 5 Neural Rendering natively in Linux/Proton games.",
                        )))
                        .child(
                            h_flex()
                                .gap_4()
                                .flex_wrap()
                                .justify_center()
                                .child(Link::new("about-website").href(WEBSITE).child("Website"))
                                .child(Link::new("about-issues").href(ISSUES).child("Report an Issue"))
                                .child(Link::new("about-upstream").href("https://github.com/bmitch87/DLSS5VKLayer").child("Based on DLSS5VKLayer by bmitch87")),
                        )
                        .child(
                            v_flex()
                                .gap_1()
                                .items_center()
                                .text_sm()
                                .text_color(muted)
                                .text_center()
                                .child(label("developer", "Developer: Linnard Alex Brown Jr."))
                                .child(div().pt_2().font_weight(FontWeight::SEMIBOLD).child(label("credits-title", "Design and technique credits")))
                                .children(credits.into_iter().enumerate().map(|(i, credit)| label(("credit", i), credit)))
                                .child(div().pt_2().child(label("built-with", "Built with: Claude Code (Anthropic), Codex (OpenAI)")))
                                .child(label("license", "License: GNU Affero General Public License v3.0 or later")),
                        ),
                )
                .child(
                    dialog_buttons()
                        .child(Button::new("about-close").track_focus(&focus.handle).label("Close").on_click(|_, window, cx| window.close_dialog(cx))),
                ),
        )
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  The window shown instead when the channel cannot be opened
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct ErrorView {
    title: &'static str,
    description: String,
}

impl ErrorView {
    pub(crate) fn new(error: &neural_forge_protocol::mapping::OpenError, _: &mut Context<Self>) -> Self {
        use neural_forge_protocol::mapping::OpenError;
        let (title, description) = match error {
            OpenError::WrongVersion { found } => (
                "Another Neural Forge version is running",
                format!(
                    "The shared memory was set up by a build that speaks version {found}; this app speaks {}. Close the game if one is running: the next game started with this version re-creates it, and Neural Forge then opens normally.",
                    neural_forge_protocol::SHM_VERSION
                ),
            ),
            OpenError::Unavailable => ("Couldn't open the shared-memory mapping", "neural-forge-cli doctor may help.".to_string()),
            OpenError::Foreign => (
                "The shared-memory path names another file",
                "The configured channel path (shm= in config.ini, or NEURAL_FORGE_SHM) points at an existing file that is not a Neural Forge mapping. It was left untouched. Point the setting at another path, or remove it to use the default."
                    .to_string(),
            ),
        };
        Self { title, description }
    }
}

impl Render for ErrorView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("error")
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(gpui_kit::component::TitleBar::new().child(div().pl_2().text_sm().font_weight(FontWeight::MEDIUM).child("Neural Forge")))
            .child(
                v_flex()
                    .flex_1()
                    .items_center()
                    .justify_center()
                    .gap_3()
                    .p_6()
                    .child(div().text_color(cx.theme().danger).child(Icon::new(IconName::CircleX).large()))
                    .child(div().text_xl().font_weight(FontWeight::BOLD).text_center().child(label("title", self.title)))
                    .child(div().text_center().text_color(cx.theme().muted_foreground).child(label("description", self.description.clone()))),
            )
    }
}

/// Text in the theme's primary colour where it reads on the window's background (at least [`TEXT_CONTRAST`](crate::accent::TEXT_CONTRAST)), else in its link
/// colour, which a desktop accent always keeps readable: a light accent (yellow, light blue) as primary would not read on
/// white.
fn tag_text(theme: &gpui_kit::component::Theme) -> Hsla {
    use crate::accent::{contrast, Rgb, TEXT_CONTRAST};
    if contrast(Rgb::from_hsla(theme.primary), Rgb::from_hsla(theme.background)) >= TEXT_CONTRAST {
        theme.primary
    } else {
        theme.link
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steam started or closed since the scan: the list says so once the 30 s worker reads it again.
    #[test]
    fn steam_running_is_updated_after_the_scan() {
        let mut scan = SteamScan { roots: Vec::new(), running: false, games: Vec::new() };
        assert!(scan.set_running(true) && scan.running);
        assert!(!scan.set_running(true), "no change, no redraw");
        assert!(scan.set_running(false) && !scan.running);
    }

    /// Whether a Proton can take the wrapper is read in the scan (a worker), not on every draw of the Setup tab.
    #[test]
    fn the_proton_scan_says_which_sources_cannot_take_the_wrapper() {
        let root = std::env::temp_dir().join(format!("neural-forge-gui-proton-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (name, wine) in [("GE-Proton9-27", "wine-9.0"), ("GE-Proton11-12", "wine-11.0")] {
            let dir = root.join("compatibilitytools.d").join(name);
            std::fs::create_dir_all(dir.join("files/lib/wine/x86_64-unix")).unwrap();
            std::fs::write(dir.join("files/lib/wine/x86_64-unix/ntdll.so"), format!("\0{wine}\0")).unwrap();
            std::fs::write(dir.join("proton"), "").unwrap();
            std::fs::write(dir.join("version"), format!("1 {name}\n")).unwrap();
        }
        let scan = scan_protons_in(root.clone());
        let found: Vec<(String, bool)> = scan.sources.iter().map(|(s, why)| (s.version.clone(), why.is_some())).collect();
        assert_eq!(found, [("GE-Proton11-12".to_string(), false), ("GE-Proton9-27".to_string(), true)]);
        assert!(scan.sources[1].1.as_deref().unwrap().contains("Wine 9.0"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn tag_text_uses_primary_where_it_reads_and_the_link_colour_where_it_does_not() {
        use crate::accent::{paint, Rgb};
        use gpui_kit::component::{Theme, ThemeColor};
        let hex = |rgb: u32| Rgb { r: ((rgb >> 16) & 0xff) as f32 / 255., g: ((rgb >> 8) & 0xff) as f32 / 255., b: (rgb & 0xff) as f32 / 255. };
        let painted = |accent: u32| {
            let base = *ThemeColor::light();
            let mut theme = Theme::default();
            theme.colors = base;
            paint(&mut theme.colors, Some(hex(accent)), &base);
            theme
        };
        let yellow = painted(0xf6d32d);
        assert_eq!(tag_text(&yellow), yellow.link, "a yellow primary does not read on white");
        assert_ne!(yellow.link, yellow.primary);
        let purple = painted(0x9141ac);
        assert_eq!(tag_text(&purple), purple.primary, "a purple primary reads on white");
    }

    #[test]
    fn a_folder_that_cannot_be_opened_says_why() {
        let dir = std::env::temp_dir();
        assert_eq!(open_with_file_manager(&dir, "true"), Ok(()));
        let failed = open_with_file_manager(&dir, "false").unwrap_err();
        assert!(failed.starts_with("false failed ("), "{failed}");
        let missing = open_with_file_manager(&dir, "/nonexistent/neural-forge-opener").unwrap_err();
        assert!(missing.contains("could not be started"), "{missing}");
    }
}
