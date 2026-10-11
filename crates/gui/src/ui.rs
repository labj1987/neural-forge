//! The settings UI: one row per `ShmHeader` setting, grouped and tabbed the way
//! upstream's Qt GUI does (Model, Motion, Composition, Status) — used as a checklist
//! of what has to exist, not as layout code to port; a per-field `QCheckBox`/
//! `QSpinBox` binder has no logic worth transliterating either way.

use std::sync::atomic::Ordering;

use neural_forge_protocol::enums::{colour_mode, reversible_mode};
use gtk4::prelude::*;
use libadwaita as adw;
use libadwaita::prelude::*;

use crate::shm::{bind_bool, bind_float, bind_u32, Shm};

thread_local! {
    /// Set while a refresh is writing header values into widgets, so the widgets' own change
    /// handlers do not write the same values straight back (and re-persist them).
    static REFRESHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// One closure per bound row that brings the row up to date with the header.
    static REFRESHERS: std::cell::RefCell<Vec<Box<dyn Fn()>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn refreshing() -> bool {
    REFRESHING.with(std::cell::Cell::get)
}

fn register_refresher(f: impl Fn() + 'static) {
    REFRESHERS.with(|r| r.borrow_mut().push(Box::new(f)));
}

thread_local! {
    /// Everything that shows whether the model is installed, brought up to date after an extract.
    static MODEL_WATCHERS: std::cell::RefCell<Vec<Box<dyn Fn()>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn on_model_changed(f: impl Fn() + 'static) {
    MODEL_WATCHERS.with(|w| w.borrow_mut().push(Box::new(f)));
}

fn model_changed() {
    MODEL_WATCHERS.with(|w| w.borrow().iter().for_each(|f| f()));
}

/// Brings every bound row up to date with the header -- run once a second, so a change made
/// with `shmctl`, a loaded profile or Reset shows up without restarting the app.
fn refresh_all_rows() {
    REFRESHING.with(|f| f.set(true));
    REFRESHERS.with(|r| r.borrow().iter().for_each(|f| f()));
    REFRESHING.with(|f| f.set(false));
}

/// A spin row for the persisted setting `setting`, bounded by its range in
/// `neural_forge_protocol::SETTING_BOUNDS`.
fn spin_row(title: &str, subtitle: &str, value: f32, setting: &str, step: f64, setter: impl Fn(f32) + 'static) -> adw::SpinRow {
    spin_row_scaled(title, subtitle, value, setting, step, 1.0, setter)
}

/// A spin row that shows the field multiplied by `scale` (100.0 shows a fraction as a percent).
/// `step` is in displayed units; the setter still receives the stored value.
fn spin_row_scaled(title: &str, subtitle: &str, value: f32, setting: &str, step: f64, scale: f64, setter: impl Fn(f32) + 'static) -> adw::SpinRow {
    let (lower, upper) = neural_forge_protocol::setting_bounds(setting).unwrap_or_else(|| panic!("{setting} has no bounds"));
    let (lower, upper) = (f64::from(lower) * scale, f64::from(upper) * scale);
    let adjustment = gtk4::Adjustment::new(f64::from(value) * scale, lower, upper, step, step * 10.0, 0.0);
    let digits = if step >= 1.0 { 0 } else if step >= 0.1 { 1 } else { 2 };
    let row = adw::SpinRow::new(Some(&adjustment), step, digits);
    row.set_title(title);
    row.set_subtitle(subtitle);
    adjustment.connect_value_changed(move |adj| {
        if !refreshing() {
            setter((adj.value() / scale) as f32);
        }
    });
    if let Some((kind, read)) = crate::shm::take_pending_reader() {
        let adjustment = adjustment.clone();
        register_refresher(move || {
            let bits = read();
            let v = scale
                * match kind {
                    crate::shm::FieldKind::Float => f64::from(f32::from_bits(bits)),
                    crate::shm::FieldKind::Int => f64::from(bits),
                };
            if v.is_finite() && (adjustment.value() - v).abs() > 1e-6 {
                adjustment.set_value(v);
            }
        });
    }
    row
}

fn switch_row(title: &str, subtitle: &str, active: bool, setter: impl Fn(bool) + 'static) -> adw::SwitchRow {
    let row = adw::SwitchRow::new();
    row.set_title(title);
    row.set_subtitle(subtitle);
    row.set_active(active);
    row.connect_active_notify(move |row| {
        if !refreshing() {
            setter(row.is_active());
        }
    });
    if let Some((_, read)) = crate::shm::take_pending_reader() {
        let row = row.downgrade();
        register_refresher(move || {
            if let Some(row) = row.upgrade() {
                let on = read() != 0;
                if row.is_active() != on {
                    row.set_active(on);
                }
            }
        });
    }
    row
}

fn combo_row(title: &str, options: &[&str], selected: u32, setter: impl Fn(u32) + 'static) -> adw::ComboRow {
    let model = gtk4::StringList::new(options);
    let row = adw::ComboRow::new();
    row.set_title(title);
    row.set_model(Some(&model));
    let last = options.len() as u32 - 1;
    row.set_selected(selected.min(last));
    row.connect_selected_notify(move |row| {
        if !refreshing() {
            setter(row.selected());
        }
    });
    if let Some((_, read)) = crate::shm::take_pending_reader() {
        let row = row.downgrade();
        register_refresher(move || {
            if let Some(row) = row.upgrade() {
                let v = read().min(last);
                if row.selected() != v {
                    row.set_selected(v);
                }
            }
        });
    }
    row
}

/// The strength preset menu: choosing a preset sets the four rows (`rows`, in
/// `neural_forge_supervisor::presets::KNOBS`' order) through their own handlers, so the values are
/// written and saved exactly as if typed in. It shows the preset the rows match, else "Custom".
fn preset_row(rows: &[adw::SpinRow; 4]) -> adw::ComboRow {
    use neural_forge_supervisor::presets::{matching, PRESETS};
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.push("Custom");
    let custom = PRESETS.len() as u32;
    let row = adw::ComboRow::new();
    row.set_title("Strength preset");
    row.set_subtitle("Sets the four values below. Reference is Neural Forge's defaults");
    row.set_model(Some(&gtk4::StringList::new(&names)));
    let rows = std::rc::Rc::new(rows.clone());
    // Set while this code changes the menu or the rows, so neither reacts to the other.
    let syncing = std::rc::Rc::new(std::cell::Cell::new(false));
    let sync = {
        let row = row.downgrade();
        let rows = std::rc::Rc::clone(&rows);
        let syncing = std::rc::Rc::clone(&syncing);
        std::rc::Rc::new(move || {
            let Some(row) = row.upgrade() else { return };
            let values = [0, 1, 2, 3].map(|i| rows[i].value() as f32);
            let shown = matching(values).map_or(custom, |i| i as u32);
            if row.selected() != shown {
                syncing.set(true);
                row.set_selected(shown);
                syncing.set(false);
            }
        })
    };
    sync();
    {
        let rows = std::rc::Rc::clone(&rows);
        let syncing = std::rc::Rc::clone(&syncing);
        let sync = std::rc::Rc::clone(&sync);
        row.connect_selected_notify(move |row| {
            if syncing.get() || refreshing() {
                return;
            }
            // "Custom" sets nothing; the menu goes back to what the rows match.
            if let Some(preset) = PRESETS.get(row.selected() as usize) {
                syncing.set(true);
                for (spin, value) in rows.iter().zip(preset.values) {
                    spin.set_value(f64::from(value));
                }
                syncing.set(false);
            }
            sync();
        });
    }
    // A value changed by hand, by a profile, Reset or `shmctl` (the rows follow the header).
    for spin in rows.iter() {
        let syncing = std::rc::Rc::clone(&syncing);
        let sync = std::rc::Rc::clone(&sync);
        spin.connect_value_notify(move |_| {
            if !syncing.get() {
                sync();
            }
        });
    }
    row
}

/// GDK on Linux X11/Wayland uses XKB hardware codes (evdev + 8).
fn evdev_keycode(hardware: u32) -> Option<u32> {
    hardware.checked_sub(8).filter(|&code| code > 0 && code <= 767)
}

/// The key's name on the current keyboard layout ("F11", "Scroll_Lock", "A") for an evdev
/// code, falling back to the bare code when there is no display or no mapping.
fn key_label(code: u32) -> String {
    if code == 0 {
        return "Unbound".to_owned();
    }
    let name = gtk4::gdk::Display::default()
        .and_then(|display| display.map_keycode(code + 8))
        // Prefer group 0, level 0 (the unshifted key); otherwise whatever came first.
        .and_then(|entries| entries.iter().find(|(k, _)| k.group() == 0 && k.level() == 0).or(entries.first()).map(|(_, key)| *key))
        .and_then(|key| key.name())
        .map(|name| name.to_string());
    format_key_label(code, name.as_deref())
}

fn format_key_label(code: u32, name: Option<&str>) -> String {
    match name {
        Some(name) if name.chars().count() == 1 => name.to_uppercase(),
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => format!("Key {code}"),
    }
}

fn hotkey_row(initial: u32, setter: impl Fn(u32) + 'static) -> adw::ActionRow {
    let reader = crate::shm::take_pending_reader();
    let row = adw::ActionRow::builder().title("Toggle key")
        .subtitle("Toggles Neural Forge inside a running game. Needs read access to /dev/input (the 'input' group) or an X11/XWayland session.").build();
    let label = key_label;
    let button = gtk4::Button::with_label(&label(initial));
    button.set_valign(gtk4::Align::Center);
    let clear = gtk4::Button::with_label("Clear");
    clear.set_valign(gtk4::Align::Center);
    let value = std::rc::Rc::new(std::cell::Cell::new(initial));
    let capturing = std::rc::Rc::new(std::cell::Cell::new(false));
    let setter = std::rc::Rc::new(setter);
    let controller = gtk4::EventControllerKey::new();
    controller.set_propagation_phase(gtk4::PropagationPhase::Capture);
    {
        let capturing = capturing.clone();
        button.connect_clicked(move |button| {
            capturing.set(true);
            button.set_label("Press a key (Esc cancels)");
            button.grab_focus();
        });
    }
    {
        let button = button.downgrade();
        let capturing = capturing.clone();
        let value = value.clone();
        let setter = setter.clone();
        controller.connect_key_pressed(move |_, key, hardware, _| {
            if !capturing.get() { return glib::Propagation::Proceed; }
            if let Some(button) = button.upgrade() {
                if key != gtk4::gdk::Key::Escape {
                    if let Some(code) = evdev_keycode(hardware) {
                        setter(code);
                        value.set(code);
                    }
                }
                capturing.set(false);
                button.set_label(&label(value.get()));
            }
            glib::Propagation::Stop
        });
    }
    {
        let capturing = capturing.clone();
        let value = value.clone();
        button.connect_has_focus_notify(move |button| {
            if !button.has_focus() && capturing.replace(false) {
                button.set_label(&label(value.get()));
            }
        });
    }
    {
        let button = button.clone();
        let capturing = capturing.clone();
        let value = value.clone();
        clear.connect_clicked(move |_| {
            capturing.set(false);
            value.set(0);
            setter(0);
            button.set_label("Unbound");
        });
    }
    button.add_controller(controller);
    // Follow the header, so Reset, a loaded profile or `shmctl` shows up here too.
    if let Some((_, read)) = reader {
        let button = button.downgrade();
        register_refresher(move || {
            let Some(button) = button.upgrade() else { return };
            let code = read();
            if !capturing.get() && value.get() != code {
                value.set(code);
                button.set_label(&label(code));
            }
        });
    }
    row.add_suffix(&button);
    row.add_suffix(&clear);
    row
}

pub fn build_ui(app: &adw::Application, install_error: Option<String>) {
    let shm = match Shm::open() {
        Ok(shm) => shm,
        Err(e) => {
            build_error_window(app, &e);
            return;
        }
    };
    let shm = shm.0;

    // --- Model -------------------------------------------------------------------
    let model_group = adw::PreferencesGroup::new();
    model_group.set_title("Model");

    let (enabled, set_enabled) = bind_bool(&shm, Some("enabled"), |h| &h.enabled);
    model_group.add(&switch_row("Neural rendering", "Off keeps the layer running but presents the original frame", enabled, set_enabled));

    let (style, set_style) = bind_u32(&shm, Some("style"), |h| &h.style);
    model_group.add(&combo_row("Style", &["Default", "Natural", "Cinematic"], style, set_style));

    let (intensity, set_intensity) = bind_float(&shm, Some("intensity"), |h| &h.intensity_bits);
    let intensity_row = spin_row("Intensity", "How strongly the model's answer replaces the frame", intensity, "intensity", 0.05, set_intensity);

    let (local_tone, set_local_tone) = bind_float(&shm, Some("local_tone"), |h| &h.local_tone_bits);
    let local_tone_row = spin_row("Local tone", "", local_tone, "local_tone", 0.05, set_local_tone);

    let (local_structure, set_local_structure) = bind_float(&shm, Some("local_structure"), |h| &h.local_structure_bits);
    let local_structure_row = spin_row("Local structure", "", local_structure, "local_structure", 0.05, set_local_structure);

    let (skin_structure, set_skin_structure) = bind_float(&shm, Some("skin_structure"), |h| &h.skin_structure_bits);
    let skin_structure_row = spin_row("Skin structure", "-1 follows local structure", skin_structure, "skin_structure", 0.05, set_skin_structure);

    // In `neural_forge_supervisor::presets::KNOBS`' order.
    let strength_rows = [intensity_row, local_tone_row, local_structure_row, skin_structure_row];
    model_group.add(&preset_row(&strength_rows));
    for row in &strength_rows {
        model_group.add(row);
    }

    let (auto_mask, set_auto_mask) = bind_bool(&shm, Some("auto_mask"), |h| &h.auto_mask);
    model_group.add(&switch_row("Auto mask", "Automatic skin/detail masking", auto_mask, set_auto_mask));

    let (interval, set_interval) = bind_u32(&shm, Some("model_interval"), |h| &h.model_interval);
    let interval_row = spin_row(
        "Model every Nth frame",
        "1 = every frame. With frame generation on, 2 lets generated frames reuse the last answer instead of waiting for their own",
        interval as f32,
        "model_interval",
        1.0,
        move |v| set_interval(v as u32),
    );
    model_group.add(&interval_row);

    let (toggle_key, set_toggle_key) = bind_u32(&shm, Some("toggle_key"), |h| &h.toggle_key);
    model_group.add(&hotkey_row(toggle_key, set_toggle_key));

    // --- Composition -----------------------------------------------------------------
    let comp_group = adw::PreferencesGroup::new();
    comp_group.set_title("Composition");

    let (transfer_strength, set_transfer_strength) = bind_float(&shm, Some("transfer_strength"), |h| &h.transfer_strength_bits);
    comp_group.add(&spin_row("Detail strength", "How much of the model's edit reaches the frame; above 1 amplifies it", transfer_strength, "transfer_strength", 0.05, set_transfer_strength));

    let (colour_strength, set_colour_strength) = bind_float(&shm, Some("colour_strength"), |h| &h.colour_strength_bits);
    comp_group.add(&spin_row("Colour strength", "How much of the transfer is allowed to be colour, not just luminance", colour_strength, "colour_strength", 0.05, set_colour_strength));

    let (max_ratio, set_max_ratio) = bind_float(&shm, Some("max_ratio"), |h| &h.max_ratio_bits);
    comp_group.add(&spin_row("Highlight guard", "The most the pass may brighten or darken a pixel by (x)", max_ratio, "max_ratio", 0.1, set_max_ratio));

    let (working_scale, set_working_scale) = bind_float(&shm, Some("working_scale"), |h| &h.working_scale_bits);
    let scale_row = spin_row_scaled("Model resolution", "Percent of the frame the model works at; lower is faster and softer", working_scale, "working_scale", 5.0, 100.0, set_working_scale);
    comp_group.add(&scale_row);


    let (reversible, set_reversible) = bind_u32(&shm, Some("reversible_mode"), |h| &h.reversible_mode);
    comp_group.add(&combo_row(
        "Reversible mode",
        &["Knee", "Neutwo", "Neutwo replace", "Hybrid", "Hybrid replace"],
        reversible,
        set_reversible,
    ));
    debug_assert_eq!(reversible_mode::KNEE, 0);

    let (hdr_mode, set_hdr_mode) = bind_u32(&shm, Some("hdr_mode"), |h| &h.hdr_mode);
    let hdr_row = combo_row("HDR input", &["Auto", "Off", "Force float16"], hdr_mode, set_hdr_mode);
    // The layer does not read `hdr_mode` or `colour_mode` yet. Kept, like the
    // supersampling filter, so the saved value round-trips.
    hdr_row.set_subtitle("Unavailable: not used by the layer yet");
    hdr_row.set_sensitive(false);
    comp_group.add(&hdr_row);

    let (colour_mode, set_colour_mode) = bind_u32(&shm, Some("colour_mode"), |h| &h.colour_mode);
    let colour_mode_row = combo_row("Colour mode", &["Auto", "Force display-referred", "Force linear HDR"], colour_mode, set_colour_mode);
    colour_mode_row.set_subtitle("Unavailable: not used by the layer yet");
    colour_mode_row.set_sensitive(false);
    comp_group.add(&colour_mode_row);
    debug_assert_eq!(colour_mode::AUTO, 0);

    let hdr_group = adw::PreferencesGroup::new();
    hdr_group.set_title("White point");
    hdr_group.set_description(Some(
        "What the model is shown as white. Manual uses the slider; Measured reads the frame's own white \
         level (so dark scenes reach the model well exposed) times the trim. Both are times the scale.",
    ));
    let (source, set_source) = bind_u32(&shm, Some("white_point_source"), |h| &h.white_point_source);
    hdr_group.add(&combo_row("White point source", &["Manual", "Measured"], source, set_source));
    let (white, set_white) = bind_float(&shm, Some("white_point"), |h| &h.white_point_bits);
    hdr_group.add(&spin_row("Manual white point", "Linear-light reference", white, "white_point", 0.1, set_white));
    let (scale, set_scale) = bind_float(&shm, Some("white_point_scale"), |h| &h.white_point_scale_bits);
    hdr_group.add(&spin_row("White point scale", "Multiplier", scale, "white_point_scale", 0.05, set_scale));
    let (trim, set_trim) = bind_float(&shm, Some("white_point_trim"), |h| &h.white_point_trim_bits);
    hdr_group.add(&spin_row("White point trim", "Calibration multiplier", trim, "white_point_trim", 0.05, set_trim));

    let (transfer, set_transfer) = bind_u32(&shm, Some("transfer"), |h| &h.transfer);
    let transfer_row = combo_row("Transfer mode", &["Classic", "Matched residual", "Native + edit"], transfer, set_transfer);
    transfer_row.set_subtitle("How the answer comes back when model resolution is below 100% (identical at 100%). Native + edit keeps text and edges sharpest");
    comp_group.add(&transfer_row);

    let (colour_trust, set_colour_trust) = bind_float(&shm, Some("colour_trust"), |h| &h.colour_trust_bits);
    comp_group.add(&spin_row(
        "Colour trust",
        "How far the model may change a pixel's colour; larger changes (edge fringing) are shortened. 0 = no model colour",
        colour_trust,
        "colour_trust",
        0.1,
        set_colour_trust,
    ));

    let (ratio_smooth, set_ratio_smooth) = bind_float(&shm, Some("ratio_smooth"), |h| &h.ratio_smooth_bits);
    comp_group.add(&spin_row_scaled(
        "Ratio smoothing",
        "Percent of the relighting taken from the neighbourhood rather than each pixel; removes speckle",
        ratio_smooth,
        "ratio_smooth",
        5.0,
        100.0,
        set_ratio_smooth,
    ));

    let (ghost_guard, set_ghost_guard) = bind_float(&shm, Some("ghost_guard"), |h| &h.ghost_guard_bits);
    comp_group.add(&spin_row(
        "Ghost guard",
        "Only matters in pipelined mode (NEURAL_FORGE_PIPELINED=1): hides a late answer's detail where the frame has moved",
        ghost_guard,
        "ghost_guard",
        0.05,
        set_ghost_guard,
    ));

    let (apply_model, set_apply_model) = bind_bool(&shm, Some("apply_model"), |h| &h.apply_model);
    comp_group.add(&switch_row("Apply model edit", "Off presents the clean frame — capture/transport/round-trip still run, for an honest A/B", apply_model, set_apply_model));

    let (hold_frame, set_hold_frame) = bind_bool(&shm, Some("hold_frame"), |h| &h.hold_frame);
    comp_group.add(&switch_row("Hold frame", "Freeze the frame the pass works on, to re-run composition over the same picture", hold_frame, set_hold_frame));

    // --- Compare and debug ----------------------------------------------------------
    let debug_group = adw::PreferencesGroup::new();
    // Not "Compare & debug" -- `AdwPreferencesGroup::title` is parsed as Pango markup,
    // and a bare `&` breaks it (confirmed via a real run: "Failed to set text ...
    // Entity did not end with a semicolon").
    debug_group.set_title("Compare and debug");

    let (compare_mode, set_compare_mode) = bind_u32(&shm, Some("compare_mode"), |h| &h.compare_mode);
    debug_group.add(&combo_row("Compare mode", &["Off", "Side by side", "Wipe"], compare_mode, set_compare_mode));

    let (compare_split, set_compare_split) = bind_float(&shm, Some("compare_split"), |h| &h.compare_split_bits);
    debug_group.add(&spin_row("Compare split", "Wipe position, 0=left edge, 1=right edge", compare_split, "compare_split", 0.05, set_compare_split));

    let (compare_zoom, set_compare_zoom) = bind_float(&shm, Some("compare_zoom"), |h| &h.compare_zoom_bits);
    debug_group.add(&spin_row("Compare zoom", "", compare_zoom, "compare_zoom", 0.1, set_compare_zoom));

    let (compare_swap, set_compare_swap) = bind_bool(&shm, Some("compare_swap"), |h| &h.compare_swap);
    debug_group.add(&switch_row("Swap compare sides", "", compare_swap, set_compare_swap));

    let (debug_view, set_debug_view) = bind_u32(&shm, Some("debug_view"), |h| &h.debug_view);
    let debug_view_row = combo_row(
        "Debug view",
        &[
            "Off",
            "Original / proxy",
            "Model's raw answer",
            "Amplified diff",
            "Colour trust (green passes, red held back)",
            "Pre-colour-trust composite",
        ],
        debug_view,
        set_debug_view,
    );
    debug_view_row.set_subtitle("The last two only differ from the composited picture where colour trust is actually engaged");
    debug_group.add(&debug_view_row);

    let (debug_scale, set_debug_scale) = bind_float(&shm, Some("debug_scale"), |h| &h.debug_scale_bits);
    debug_group.add(&spin_row(
        "Debug view 5 amplification",
        "Multiplies the pre-colour-trust colour view 5 shows, to make a subtle difference easier to see",
        debug_scale,
        "debug_scale",
        0.1,
        set_debug_scale,
    ));

    let toasts = adw::ToastOverlay::new();
    if let Some(message) = install_error {
        toasts.add_toast(adw::Toast::builder().title(message).timeout(0).build());
    }

    // Tabbed like upstream's Qt GUI, rather than one long scrolling page -- each tab
    // is still an AdwPreferencesPage, which scrolls internally on its own if its
    // content overflows the window.
    let view_stack = adw::ViewStack::new();
    view_stack.set_vexpand(true);

    let model_page = adw::PreferencesPage::new();
    model_page.add(&model_group);
    view_stack.add_titled_with_icon(&model_page, Some("model"), "Model", "applications-graphics-symbolic");

    // Model resolution and model-every-Nth-frame apply after the upscaler only: shown while a game runs
    // and the model is not running before the upscaler.
    {
        let shm = std::sync::Arc::clone(&shm);
        let beat = std::cell::Cell::new((0u32, std::time::Instant::now()));
        let update = move || {
            let after_upscaler = {
                let hdr = shm.header();
                let now = hdr.layer_heartbeat.load(Ordering::Relaxed);
                let (seen, at) = beat.get();
                let active = if now != seen {
                    beat.set((now, std::time::Instant::now()));
                    seen != 0
                } else {
                    at.elapsed() < std::time::Duration::from_secs(2)
                };
                active && hdr.native_running.load(Ordering::Relaxed) == 0
            };
            for row in [interval_row.upcast_ref::<gtk4::Widget>(), scale_row.upcast_ref()] {
                row.set_visible(after_upscaler);
            }
        };
        update();
        glib::timeout_add_seconds_local(1, move || {
            update();
            glib::ControlFlow::Continue
        });
    }

    let composition_page = adw::PreferencesPage::new();
    composition_page.add(&comp_group);
    composition_page.add(&hdr_group);
    view_stack.add_titled_with_icon(&composition_page, Some("composition"), "Composition", "view-paged-symbolic");

    let debug_page = adw::PreferencesPage::new();
    debug_page.add(&debug_group);
    debug_page.add(&build_capture_group(&shm, &toasts));
    // Short tab label -- the group's own title inside the page ("Compare and debug")
    // carries the full wording; ViewSwitcher button labels are cramped for five tabs
    // and (like AdwPreferencesGroup::title) are Pango markup, so no bare "&" either.
    view_stack.add_titled_with_icon(&debug_page, Some("debug"), "Debug", "edit-find-symbolic");

    let setup_page = build_setup_page(&toasts);

    let status_page = adw::PreferencesPage::new();
    status_page.add(&build_telemetry_group(&shm));
    let status_group = build_status_group(&shm, &toasts);
    status_page.add(&status_group);
    status_page.add(&build_report_group(&shm, &toasts));
    view_stack.add_titled_with_icon(&status_page, Some("status"), "Status", "network-transmit-receive-symbolic");

    view_stack.add_titled_with_icon(&setup_page, Some("setup"), "Setup", "preferences-system-symbolic");

    // First-run flow: no model means neural rendering can't work at all yet (fail-open just
    // presents untouched frames, which a first-time user would never notice) -- land on Setup
    // instead of Model, with a banner explaining why, rather than a silently-inert app.
    let missing_model = !model_status().1;
    if missing_model {
        view_stack.set_visible_child_name("setup");
    }
    let banner = adw::Banner::new("The neural rendering model is missing -- extract it in Setup");
    banner.set_button_label(Some("Open Setup"));
    banner.set_revealed(missing_model);
    {
        let banner = banner.clone();
        on_model_changed(move || banner.set_revealed(!model_status().1));
    }
    {
        let view_stack = view_stack.clone();
        let banner_for_closure = banner.clone();
        banner.connect_button_clicked(move |_| {
            view_stack.set_visible_child_name("setup");
            banner_for_closure.set_revealed(false);
        });
    }

    // `AdwViewSwitcherTitle`/`Bar` are deprecated since libadwaita 1.4 in favor of a
    // plain `AdwViewSwitcher` plus `AdwBreakpoint` -- previously left as the
    // version-compatible choice because whether CI's own libadwaita was actually new
    // enough was never confirmed. Confirmed 2026-09-15 directly against a real CI run's own `apt-get
    // install` log, not assumed: GitHub Actions' `ubuntu-latest` (Ubuntu 24.04
    // "noble") installs libadwaita 1.5.0, comfortably past the v1_4 this needs, and
    // that same run's build output was already emitting deprecation warnings for the
    // old widgets -- both reasons to migrate now rather than keep suppressing the
    // warning. The header shows the switcher permanently rather than reproducing
    // `ViewSwitcherTitle`'s separate "plain title text" state (the OS-level window
    // title still says "NeuralForge"; the common pattern in real Adwaita apps like
    // GNOME Text Editor omits an in-header title entirely once a view switcher is
    // present) -- a real, deliberate simplification, not an oversight.
    let switcher = adw::ViewSwitcher::new();
    switcher.set_stack(Some(&view_stack));
    switcher.set_policy(adw::ViewSwitcherPolicy::Wide);

    let header = adw::HeaderBar::new();
    header.set_title_widget(Some(&switcher));
    let about_btn = gtk4::Button::builder().icon_name("help-about-symbolic").tooltip_text("About").build();
    header.pack_end(&about_btn);

    // Reveals this bottom bar and hides the header's own switcher when the window
    // narrows past the breakpoint below -- the explicit, `AdwBreakpoint`-driven
    // replacement for what `ViewSwitcherTitle`'s `title-visible` binding did
    // implicitly, so the window can still be narrowed without the tab bar becoming
    // unusable.
    let switcher_bar = adw::ViewSwitcherBar::new();
    switcher_bar.set_stack(Some(&view_stack));

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.add_top_bar(&banner);
    toolbar_view.set_content(Some(&view_stack));
    toolbar_view.add_bottom_bar(&switcher_bar);
    toasts.set_child(Some(&toolbar_view));

    // Developer aid: `NEURAL_FORGE_GUI_OPEN=<tab id>` (model/composition/debug/status/setup) selects
    // that tab at startup, so it can be screenshotted where synthetic input cannot reliably reach a
    // page navigated to after the window opens.
    if let Some(tab @ ("model" | "composition" | "debug" | "status" | "setup")) = neural_forge_protocol::env::var("NEURAL_FORGE_GUI_OPEN").as_deref() {
        view_stack.set_visible_child_name(tab);
    }

    // Follow the header: a change made with `shmctl`, a loaded profile, Reset or another
    // instance appears here within a second instead of needing an app restart.
    glib::timeout_add_seconds_local(1, || {
        refresh_all_rows();
        glib::ControlFlow::Continue
    });

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Neural Forge")
        .default_width(620)
        .default_height(700)
        .content(&toasts)
        .build();

    {
        let toasts = toasts.clone();
        crate::shm::on_save_error(move |message| toasts.add_toast(adw::Toast::new(&message)));
    }
    // A change made in the last moment before closing is still saved.
    window.connect_close_request(|_| {
        if let Err(e) = crate::shm::flush_settings() {
            eprintln!("neural-forge: saving config.ini failed: {e}");
        }
        glib::Propagation::Proceed
    });

    // Below 1400sp wide: hide the header's inline switcher, reveal the bottom bar
    // instead. 1400, not the far narrower value an adaptive/mobile-first app would
    // use, because this window's own `PreferencesPage` content already wants ~1340px
    // of natural width (six settings groups with subtitles, spin rows, etc.) -- real
    // screenshot evidence (2026-09-16, this sandbox's X11 screenshot workaround)
    // caught a `Wide`-policy header switcher rendering with every tab label truncated
    // to a single character at that natural size: plenty of *window* width, but not
    // enough left over for six full icon+label tabs once the header's own symmetric
    // title-centering and the About button are accounted for. A plain width guess
    // from docs would have shipped that broken. Confirmed via screenshot both ways
    // after raising the threshold: fully legible tabs in the header only once the
    // window is genuinely wide (tested maximized), the bottom bar otherwise.
    // `add_setters` requires every tuple in one call to share the same concrete
    // widget type, hence the upcast to the common `gtk4::Widget` -- `switcher` and
    // `switcher_bar` are otherwise different libadwaita types.
    let narrow = adw::BreakpointCondition::new_length(adw::BreakpointConditionLengthType::MaxWidth, 1400.0, adw::LengthUnit::Sp);
    let breakpoint = adw::Breakpoint::new(narrow);
    breakpoint.add_setters(&[
        (switcher.upcast_ref::<gtk4::Widget>(), "visible", false),
        (switcher_bar.upcast_ref::<gtk4::Widget>(), "reveal", true),
    ]);
    window.add_breakpoint(breakpoint);

    {
        let window = window.clone();
        about_btn.connect_clicked(move |_| {
            let dialog = adw::AboutDialog::builder()
                .application_name("Neural Forge")
                .version(env!("CARGO_PKG_VERSION"))
                .developers(vec!["Linnard Alex Brown Jr."])
                .comments("Vulkan layer and settings GUI for running NVIDIA DLSS 5 Neural Rendering natively in Linux/Proton games.")
                .website("https://github.com/labj1987/neural-forge")
                .issue_url("https://github.com/labj1987/neural-forge/issues")
                // Matches Cargo.toml's `AGPL-3.0-or-later`; the upstream project's own
                // license is AGPL-3.0, which is what requires it for the adapted code.
                .license_type(gtk4::License::Agpl30)
                .build();
            dialog.add_link("Based on DLSS5VKLayer by bmitch87", "https://github.com/bmitch87/DLSS5VKLayer");
            dialog.add_acknowledgement_section(
                Some("Design and technique credits"),
                &[
                    "clshortfuse (RenoDX) — the colour composition design",
                    "hhkbble — the matched-residual transfer mode",
                    "xenmods (DLSSNR-Cost-Scaler) — the native + edit technique",
                    "Dagherbou and cdozdil (OptiScaler) — the DLSS-NR shader lineage",
                ],
            );
            dialog.add_credit_section(
                Some("Built with"),
                &["Claude Code (Anthropic)", "Codex (OpenAI)"],
            );
            dialog.present(Some(&window));
        });
    }

    window.present();
}

fn profile_names() -> Vec<String> {
    let mut names: Vec<String> = neural_forge_supervisor::profiles::load_all().into_keys().collect();
    names.sort();
    names
}

/// The profile name the combo shows as selected, read from the combo's own model (the list on disk
/// may have changed since it was built); `None` when nothing real is selected.
fn selected_profile(combo: &adw::ComboRow) -> Option<String> {
    if !combo.is_sensitive() {
        return None; // the "No saved profiles" placeholder
    }
    combo.selected_item().and_downcast::<gtk4::StringObject>().map(|item| item.string().to_string())
}

/// Rebuilds the combo's model from disk -- called on init and after every
/// save/delete, since the set of saved profiles can only change through this same
/// window (single-user, single-process local GUI).
fn refresh_profile_combo(combo: &adw::ComboRow) {
    let names = profile_names();
    if names.is_empty() {
        combo.set_model(Some(&gtk4::StringList::new(&["No saved profiles"])));
        combo.set_sensitive(false);
    } else {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        combo.set_model(Some(&gtk4::StringList::new(&refs)));
        combo.set_sensitive(true);
    }
}

/// The exact Steam launch-option string for these settings -- pulled out of the
/// closure below so it's a plain, unit-testable function instead of only ever being
/// exercised live through GTK signal handlers.
///
/// There is no DMA-BUF switch: nothing reads `NEURAL_FORGE_DMABUF` (the transport is not wired),
/// and a switch that does nothing was worse than none.
fn launch_option(target_exe: &str) -> String {
    let mut parts = vec!["NEURAL_FORGE_ENABLE=1".to_string()];
    let target_exe = target_exe.trim();
    if !target_exe.is_empty() {
        parts.push(format!("NEURAL_FORGE_TARGET_EXE={}", shell_quote(target_exe)));
    }
    parts.push("%command%".to_string());
    parts.join(" ")
}

/// Steam runs the launch option through a shell, so a value with a space or any other
/// shell character is single-quoted (a `'` inside becomes `'\''`).
fn shell_quote(value: &str) -> String {
    if value.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn build_launch_option_group() -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Steam launch option");
    group.set_description(Some("Paste this into the game's Properties -> Launch Options in Steam"));

    let exe_row = adw::EntryRow::new();
    exe_row.set_title("Target executable (optional, for a multi-process game)");
    group.add(&exe_row);

    let preview_row = adw::ActionRow::new();
    preview_row.set_title("Launch option");
    preview_row.add_css_class("property");
    let copy_button = gtk4::Button::with_label("Copy");
    copy_button.set_valign(gtk4::Align::Center);
    preview_row.add_suffix(&copy_button);
    group.add(&preview_row);

    let exe_row_for_build = exe_row.clone();
    let build_option = std::rc::Rc::new(move || launch_option(&exe_row_for_build.text()));

    preview_row.set_subtitle(&build_option());

    {
        let preview_row = preview_row.clone();
        let build_option = std::rc::Rc::clone(&build_option);
        exe_row.connect_changed(move |_| preview_row.set_subtitle(&build_option()));
    }
    copy_button.connect_clicked(move |button| {
        button.display().clipboard().set_text(&build_option());
    });

    group
}

/// The model's row text: present with the build it came from (and whether that build is verified),
/// or missing.
fn model_status() -> (String, bool) {
    let dir = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    match neural_forge_supervisor::model::installed(&dir) {
        Some(build) if neural_forge_supervisor::model::installed_verified(&dir) == Some(false) => {
            (format!("present, from build {build}, not verified against NVIDIA's runtime"), true)
        }
        Some(build) => (format!("present, from build {build}"), true),
        None => ("missing: extract it from nvngx_dlssnr.dll below".to_string(), false),
    }
}

/// The native backend's one setup step: point at NVIDIA's DLL once and extract the model from it.
fn build_model_group(toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Neural rendering model");
    group.set_description(Some(
        "Taken from your own nvngx_dlssnr.dll -- this project doesn't and can't ship it",
    ));
    let (text, present) = model_status();
    let status_row = adw::ActionRow::new();
    status_row.set_title("Model");
    status_row.set_subtitle(&text);
    let icon = gtk4::Image::from_icon_name(if present { "emblem-ok-symbolic" } else { "dialog-warning-symbolic" });
    status_row.add_prefix(&icon);
    group.add(&status_row);

    let extract_row = adw::ActionRow::new();
    extract_row.set_title("Extract from DLL");
    extract_row.set_subtitle("Choose nvngx_dlssnr.dll; the model is written to Neural Forge's data folder");
    let button = gtk4::Button::with_label("Extract…");
    button.set_valign(gtk4::Align::Center);
    extract_row.add_suffix(&button);
    extract_row.set_activatable_widget(Some(&button));
    group.add(&extract_row);

    let toasts = toasts.clone();
    button.connect_clicked(move |button| {
        let toasts = toasts.clone();
        let (status_row, icon) = (status_row.clone(), icon.clone());
        let parent = button.root().and_downcast::<gtk4::Window>();
        let dialog = gtk4::FileDialog::builder().title("Select nvngx_dlssnr.dll").build();
        dialog.open(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
            let Ok(file) = result else { return };
            let Some(path) = file.path() else {
                toasts.add_toast(adw::Toast::new("That file has no local path -- choose a file on this computer"));
                return;
            };
            glib::spawn_future_local(async move {
                let out = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
                match gio::spawn_blocking(move || neural_forge_supervisor::model::extract(&path, &out)).await {
                    Ok(Ok(done)) => {
                        let unverified = if done.verified { "" } else { " (not verified against NVIDIA's runtime)" };
                        toasts.add_toast(adw::Toast::new(&format!(
                            "Model extracted from build {}{unverified} -- restart the game to use it",
                            done.build
                        )))
                    }
                    // The report is too long for a toast: its summary line there, all of it in the log.
                    Ok(Err(neural_forge_supervisor::model::ModelError::Shape(report))) => {
                        eprintln!("neural-forge: extraction refused:\n{report}");
                        let summary = report.lines().next().unwrap_or_default();
                        toasts.add_toast(adw::Toast::new(&format!("Extraction refused, a different network: {summary}")));
                    }
                    Ok(Err(e)) => toasts.add_toast(adw::Toast::new(&format!("Extraction failed: {e}"))),
                    Err(_) => toasts.add_toast(adw::Toast::new("Extraction failed: the worker panicked")),
                }
                let (text, present) = model_status();
                status_row.set_subtitle(&text);
                icon.set_icon_name(Some(if present { "emblem-ok-symbolic" } else { "dialog-warning-symbolic" }));
                model_changed();
            });
        });
    });
    group
}

fn build_setup_page(toasts: &adw::ToastOverlay) -> adw::PreferencesPage {
    let page = adw::PreferencesPage::new();
    page.add(&build_model_group(toasts));
    page.add(&build_launch_option_group());
    page
}

/// One sample of the two timing series the sparkline plots, both already published by the layer for
/// `neural-forge-cli shmctl status` -- this just samples them on a faster timer than the
/// once-a-second status labels need, and keeps the last few seconds for the drawing area to plot.
#[derive(Clone, Copy)]
struct TelemetrySample {
    /// After the upscaler: capture and compose.
    layer_ms: f32,
    /// Before the upscaler: how long the last hold took.
    hold_ms: f32,
}

const TELEMETRY_INTERVAL_MS: u32 = 200;
const TELEMETRY_WINDOW_SECS: u32 = 5;
const TELEMETRY_SAMPLES: usize = (TELEMETRY_WINDOW_SECS * 1000 / TELEMETRY_INTERVAL_MS) as usize;

fn draw_telemetry_sparkline(cr: &gtk4::cairo::Context, width: i32, height: i32, history: &std::collections::VecDeque<TelemetrySample>) {
    let (width, height) = (f64::from(width), f64::from(height));
    let _ = cr.save();
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.0);
    let _ = cr.paint();

    if history.len() < 2 {
        let _ = cr.restore();
        return;
    }

    // A fixed ceiling (not the window's own max) so the line's height means the same
    // thing frame to frame instead of visually flattening every series out whenever
    // one of them briefly spikes -- 20ms is comfortably above a healthy per-stage
    // budget at the frame rates this project targets, without being so tall that
    // ordinary sub-millisecond noise disappears into the bottom pixel row.
    const CEILING_MS: f32 = 20.0;
    let plot = |cr: &gtk4::cairo::Context, pick: fn(&TelemetrySample) -> f32| {
        for (i, sample) in history.iter().enumerate() {
            let x = width * (i as f64) / ((history.len() - 1) as f64);
            let y = height * (1.0 - f64::from(pick(sample).min(CEILING_MS) / CEILING_MS));
            if i == 0 {
                cr.move_to(x, y);
            } else {
                cr.line_to(x, y);
            }
        }
        let _ = cr.stroke();
    };

    cr.set_line_width(1.6);
    cr.set_source_rgb(0.204, 0.780, 0.678); // hold before the upscaler -- teal
    plot(cr, |s| s.hold_ms);
    cr.set_source_rgb(0.596, 0.478, 0.953); // layer (capture+compose) -- violet, matches the app's own accent
    plot(cr, |s| s.layer_ms);

    let _ = cr.restore();
}

fn legend_label(text: &str, rgb: (f64, f64, f64)) -> gtk4::Box {
    let row = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
    let swatch = gtk4::DrawingArea::new();
    swatch.set_content_width(10);
    swatch.set_content_height(10);
    swatch.set_valign(gtk4::Align::Center);
    swatch.set_draw_func(move |_, cr, w, h| {
        cr.set_source_rgb(rgb.0, rgb.1, rgb.2);
        cr.rectangle(0.0, 0.0, f64::from(w), f64::from(h));
        let _ = cr.fill();
    });
    row.append(&swatch);
    row.append(&gtk4::Label::new(Some(text)));
    row
}

fn build_telemetry_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Telemetry");
    group.set_description(Some("Live from the layer -- all zero until a targeted game attaches"));

    let model_row = adw::ActionRow::new();
    model_row.set_title("Model");
    let game_row = adw::ActionRow::new();
    game_row.set_title("Game");
    let fps_row = adw::ActionRow::new();
    fps_row.set_title("Frame rate");
    group.add(&model_row);
    group.add(&game_row);
    group.add(&fps_row);

    let legend = gtk4::Box::new(gtk4::Orientation::Horizontal, 16);
    legend.set_margin_top(10);
    legend.set_margin_start(10);
    legend.set_margin_end(10);
    legend.append(&legend_label("Hold (before the upscaler)", (0.204, 0.780, 0.678)));
    legend.append(&legend_label("Layer (after the upscaler)", (0.596, 0.478, 0.953)));

    let sparkline = gtk4::DrawingArea::new();
    // A minimum, not a ceiling: `AdwPreferencesGroup` still stretches this row taller
    // than 80px on a short page with room to spare (confirmed visually; `vexpand`
    // false the whole way down this box's ancestor chain didn't stop it either).
    // Harmless -- `draw_telemetry_sparkline` normalizes against whatever height it's
    // actually given each draw, so the plot still reads correctly at any size.
    sparkline.set_size_request(-1, 80);
    sparkline.set_hexpand(true);
    sparkline.set_vexpand(false);
    sparkline.set_margin_start(10);
    sparkline.set_margin_end(10);
    sparkline.set_margin_bottom(10);

    // One card-styled box for both, matching the boxed-row look every other group in
    // this app already has -- adding `legend`/`sparkline` straight to `group` instead
    // makes each its own bare, unstyled top-level row with layout that doesn't match
    // (confirmed visually: `AdwPreferencesGroup` gives each direct child list-row
    // spacing meant for `AdwActionRow`-shaped content, not a fixed-height drawing
    // area, so `set_content_height` alone doesn't produce the intended fixed size).
    let card = gtk4::Box::new(gtk4::Orientation::Vertical, 8);
    card.add_css_class("card");
    card.set_vexpand(false);
    card.set_valign(gtk4::Align::Start);
    card.set_margin_top(6);
    card.set_margin_bottom(6);
    card.set_margin_start(12);
    card.set_margin_end(12);
    card.append(&legend);
    card.append(&sparkline);
    group.add(&card);

    let history = std::rc::Rc::new(std::cell::RefCell::new(std::collections::VecDeque::<TelemetrySample>::with_capacity(TELEMETRY_SAMPLES)));
    {
        let history = std::rc::Rc::clone(&history);
        sparkline.set_draw_func(move |_, cr, width, height| draw_telemetry_sparkline(cr, width, height, &history.borrow()));
    }

    let shm_for_timer = std::sync::Arc::clone(shm);
    let sparkline_for_timer = sparkline.clone();
    let mut last_layer_frames: Option<u64> = None;
    let mut layer_beat_seen = (0u32, std::time::Instant::now());
    glib::timeout_add_local(std::time::Duration::from_millis(u64::from(TELEMETRY_INTERVAL_MS)), move || {
        let hdr = shm_for_timer.header();

        model_row.set_subtitle(if hdr.native_running.load(Ordering::Relaxed) != 0 { "running before the upscaler" } else { "not running before the upscaler" });
        // The name is only as current as the frames: a closed game leaves its name behind.
        let beat = hdr.layer_heartbeat.load(Ordering::Relaxed);
        if beat != layer_beat_seen.0 {
            layer_beat_seen = (beat, std::time::Instant::now());
        }
        let game = hdr.game_name();
        let active = layer_beat_seen.1.elapsed() < std::time::Duration::from_secs(2) && layer_beat_seen.0 != 0;
        game_row.set_subtitle(&match (game.is_empty(), active) {
            (_, false) => "none attached".to_string(),
            (true, true) => "attached".to_string(),
            (false, true) => game,
        });

        let layer_frames = neural_forge_protocol::load64(&hdr.layer_frames_lo, &hdr.layer_frames_hi);
        let per_second = 1000.0 / f64::from(TELEMETRY_INTERVAL_MS);
        let layer_fps = last_layer_frames.map(|prev| (layer_frames.saturating_sub(prev)) as f64 * per_second);
        last_layer_frames = Some(layer_frames);
        match layer_fps {
            Some(l) => fps_row.set_subtitle(&format!("{l:.1} presents/s")),
            None => fps_row.set_subtitle("—"),
        }

        let sample = TelemetrySample {
            layer_ms: f32::from_bits(hdr.layer_ms_bits.load(Ordering::Relaxed)),
            hold_ms: f32::from_bits(hdr.preupscale_hold_ms_bits.load(Ordering::Relaxed)),
        };
        {
            let mut history = history.borrow_mut();
            if history.len() == TELEMETRY_SAMPLES {
                history.pop_front();
            }
            history.push_back(sample);
        }
        sparkline_for_timer.queue_draw();

        glib::ControlFlow::Continue
    });

    group
}

/// The Status group -- the layer's liveness and where the model runs, refreshed on a timer, plus the
/// model, reset and profile actions.
fn build_status_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Status");

    let layer_row = adw::ActionRow::new();
    layer_row.set_title("Layer");
    // Where the model runs: before DLSS's upscaler (when the game uses DLSS Super Resolution) or
    // after it (everything else). Read-only; there is no setting for it.
    let preupscale_row = adw::ActionRow::new();
    preupscale_row.set_title("Model placement");
    group.add(&layer_row);
    group.add(&preupscale_row);

    let shm_for_timer = std::sync::Arc::clone(shm);
    // Liveness is read from the heartbeat, not `layer_attached`: a closed game never clears it.
    let layer_beat = std::cell::Cell::new((0u32, std::time::Instant::now()));
    let fresh = |cell: &std::cell::Cell<(u32, std::time::Instant)>, now_value: u32| {
        let (seen, at) = cell.get();
        if now_value != seen {
            cell.set((now_value, std::time::Instant::now()));
            true
        } else {
            at.elapsed() < std::time::Duration::from_secs(2)
        }
    };
    glib::timeout_add_seconds_local(1, move || {
        let hdr = shm_for_timer.header();
        // A header laid out by another build is refused at open (see `Shm::open`), so the
        // version needs no check here.
        let layer_active = fresh(&layer_beat, hdr.layer_heartbeat.load(Ordering::Relaxed));
        let attached = hdr.layer_attached.load(Ordering::Relaxed) != 0;
        layer_row.set_subtitle(if layer_active {
            "active: processing the game's frames"
        } else if attached {
            "idle: a game attached, but no frames are being processed (loading screen, paused or closed)"
        } else {
            "no game attached yet"
        });
        let label = preupscale_label(
            layer_active,
            hdr.preupscale_state.load(Ordering::Relaxed),
            hdr.preupscale_width.load(Ordering::Relaxed),
            hdr.preupscale_height.load(Ordering::Relaxed),
            hdr.preupscale_misses.load(Ordering::Relaxed),
        );
        // The layer's own status line: the model running, or why it is not.
        let layer_reason = if layer_active { hdr.layer_reason() } else { String::new() };
        preupscale_row.set_subtitle(&if layer_reason.is_empty() { label } else { format!("{label} -- {layer_reason}") });
        glib::ControlFlow::Continue
    });

    let model_row = adw::ActionRow::new();
    model_row.set_title("Model");
    model_row.set_subtitle(&model_status().0);
    group.add(&model_row);
    {
        let model_row = model_row.clone();
        on_model_changed(move || model_row.set_subtitle(&model_status().0));
    }

    let settings_row = adw::ActionRow::new();
    settings_row.set_title("Settings");
    settings_row.set_subtitle("Reset every tuning value to its default");
    let reset_button = gtk4::Button::with_label("Reset…");
    reset_button.add_css_class("destructive-action");
    reset_button.set_valign(gtk4::Align::Center);
    settings_row.add_suffix(&reset_button);
    group.add(&settings_row);

    {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        reset_button.connect_clicked(move |button| {
            let shm = std::sync::Arc::clone(&shm);
            let toasts = toasts.clone();
            let parent = button.root().and_downcast::<gtk4::Window>();
            let dialog = adw::AlertDialog::builder()
                .heading("Reset all settings?")
                .body("Every tuning value returns to its default. The running layer \
                       session (frame counters, transport state) is not affected.")
                .default_response("cancel")
                .close_response("cancel")
                .build();
            dialog.add_response("cancel", "Cancel");
            dialog.add_response("reset", "Reset");
            dialog.set_response_appearance("reset", adw::ResponseAppearance::Destructive);
            dialog.choose(parent.as_ref(), None::<&gio::Cancellable>, move |response| {
                if response != "reset" { return; }
                // Reset the live mapping the running layer is already
                // attached to, then overwrite config.ini with the same defaults so a
                // restart doesn't just reload the values this just cleared -- the
                // same two places `persist_one` already keeps in sync for a single
                // setting, done here for all of them at once.
                shm.header().reset_persisted_settings();
                let mut cfg = neural_forge_supervisor::Config::load();
                for (name, value) in neural_forge_protocol::persist::snapshot(shm.header()) {
                    cfg.settings.insert(name, value);
                }
                match cfg.save() {
                    Ok(()) => toasts.add_toast(adw::Toast::new("Settings reset")),
                    Err(e) => toasts.add_toast(adw::Toast::new(&format!("Reset the live session, but saving config.ini failed: {e}"))),
                }
            });
        });
    }

    let save_profile_row = adw::EntryRow::new();
    save_profile_row.set_title("Save current as");
    let save_profile_button = gtk4::Button::with_label("Save");
    save_profile_button.set_valign(gtk4::Align::Center);
    save_profile_button.add_css_class("suggested-action");
    save_profile_row.add_suffix(&save_profile_button);
    group.add(&save_profile_row);

    let profile_combo = adw::ComboRow::new();
    profile_combo.set_title("Load profile");
    refresh_profile_combo(&profile_combo);
    let load_profile_button = gtk4::Button::with_label("Load");
    load_profile_button.set_valign(gtk4::Align::Center);
    profile_combo.add_suffix(&load_profile_button);
    let delete_profile_button = gtk4::Button::with_label("Delete");
    delete_profile_button.add_css_class("destructive-action");
    delete_profile_button.set_valign(gtk4::Align::Center);
    profile_combo.add_suffix(&delete_profile_button);
    group.add(&profile_combo);

    {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        let profile_combo = profile_combo.clone();
        let save_profile_row = save_profile_row.clone();
        save_profile_button.connect_clicked(move |_| {
            let name = save_profile_row.text().trim().to_string();
            if name.is_empty() {
                toasts.add_toast(adw::Toast::new("Enter a name before saving"));
                return;
            }
            let settings = neural_forge_protocol::persist::snapshot(shm.header());
            match neural_forge_supervisor::profiles::save_profile(&name, settings) {
                Ok(()) => {
                    toasts.add_toast(adw::Toast::new(&format!("Saved profile \"{name}\"")));
                    save_profile_row.set_text("");
                    refresh_profile_combo(&profile_combo);
                }
                Err(e) => toasts.add_toast(adw::Toast::new(&format!("Save failed: {e}"))),
            }
        });
    }

    {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        let profile_combo = profile_combo.clone();
        load_profile_button.connect_clicked(move |_| {
            let Some(name) = selected_profile(&profile_combo) else {
                toasts.add_toast(adw::Toast::new("No profile selected"));
                return;
            };
            let profiles = neural_forge_supervisor::profiles::load_all();
            let Some(settings) = profiles.get(&name) else {
                toasts.add_toast(adw::Toast::new(&format!("Profile \"{name}\" is gone")));
                refresh_profile_combo(&profile_combo);
                return;
            };
            neural_forge_protocol::persist::apply(shm.header(), settings);
            // Same reasoning as the reset button above: applying to the live header
            // only affects the running session, so also fold the result into
            // config.ini via a fresh snapshot so it survives a reboot too.
            let mut cfg = neural_forge_supervisor::Config::load();
            cfg.replace_tuning(neural_forge_protocol::persist::snapshot(shm.header()));
            let message = match cfg.save() {
                Ok(()) => format!("Loaded profile \"{name}\""),
                Err(e) => format!("Applied to the running session, but saving config.ini failed: {e}"),
            };
            toasts.add_toast(adw::Toast::new(&message));
        });
    }

    {
        let toasts = toasts.clone();
        let profile_combo = profile_combo.clone();
        delete_profile_button.connect_clicked(move |_| {
            let Some(name) = selected_profile(&profile_combo) else {
                toasts.add_toast(adw::Toast::new("No profile selected"));
                return;
            };
            match neural_forge_supervisor::profiles::delete_profile(&name) {
                Ok(true) => {
                    toasts.add_toast(adw::Toast::new(&format!("Deleted profile \"{name}\"")));
                    refresh_profile_combo(&profile_combo);
                }
                Ok(false) => toasts.add_toast(adw::Toast::new("Profile already gone")),
                Err(e) => toasts.add_toast(adw::Toast::new(&format!("Delete failed: {e}"))),
            }
        });
    }

    group
}

/// Opens `dir` in the file manager, creating it first if it is missing.
fn open_folder(dir: &std::path::Path, parent: Option<gtk4::Window>, toasts: &adw::ToastOverlay) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        toasts.add_toast(adw::Toast::new(&glib::markup_escape_text(&format!("Couldn't create {}: {e}", dir.display()))));
        return;
    }
    let toasts = toasts.clone();
    gtk4::FileLauncher::new(Some(&gio::File::for_path(dir))).launch(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
        if let Err(e) = result {
            toasts.add_toast(adw::Toast::new(&glib::markup_escape_text(&format!("Couldn't open the folder: {e}"))));
        }
    });
}

/// What the capture button's row says, from the channel: whether frames are held before the upscaler
/// now, where a capture has no frame without the model's edit (see the button's tooltip).
fn capture_subtitle(holding: bool) -> &'static str {
    if holding {
        "The model runs before the upscaler now: both images will be the presented frame, which already has its edit"
    } else {
        "Saves the next frame as two PNGs: the game's frame and Neural Forge's result"
    }
}

/// The Debug page's captures: the same one-frame request as `neural-forge-cli shmctl capture`, and the
/// folder the layer writes them to.
fn build_capture_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Captures");

    let capture_row = adw::ActionRow::new();
    capture_row.set_title("Capture before/after");
    capture_row.set_subtitle(capture_subtitle(false));
    let capture_button = gtk4::Button::with_label("Capture");
    capture_button.set_valign(gtk4::Align::Center);
    // `neural_forge_layer::capture::run` serves the request after the upscaler; while frames are held
    // before it, `neural_forge_layer::series::take_request` serves it as a one-frame series instead.
    capture_button.set_tooltip_text(Some(
        "After the upscaler, the next frame is saved as <time>-original.png (the game's frame) and \
         <time>-composited.png (Neural Forge's result). While the model runs before the upscaler (a DLSS game), \
         the layer has no frame without the model's edit when the game presents, so both images are the \
         presented frame, in a series-<time> folder. A game that draws into its swapchain with a copy cannot \
         be captured after the upscaler; the layer's log says so.",
    ));
    capture_row.add_suffix(&capture_button);
    group.add(&capture_row);

    let folder_row = adw::ActionRow::new();
    folder_row.set_title("Captures folder");
    let captures_dir = std::path::PathBuf::from(neural_forge_supervisor::paths::captures_dir());
    folder_row.set_subtitle(&glib::markup_escape_text(&captures_dir.display().to_string()));
    let folder_button = gtk4::Button::with_label("Open");
    folder_button.set_valign(gtk4::Align::Center);
    folder_row.add_suffix(&folder_button);
    group.add(&folder_row);

    {
        let toasts = toasts.clone();
        let captures_dir = captures_dir.clone();
        folder_button.connect_clicked(move |button| open_folder(&captures_dir, button.root().and_downcast::<gtk4::Window>(), &toasts));
    }

    {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        capture_button.connect_clicked(move |button| {
            let header = shm.header();
            if header.capture_request.load(Ordering::Relaxed) != 0 {
                toasts.add_toast(adw::Toast::new("A capture is already waiting for the game's next frame"));
                return;
            }
            header.capture_request.store(1, Ordering::Relaxed);
            // The layer clears the request when it has taken the frame.
            let shm = std::sync::Arc::clone(&shm);
            let toasts = toasts.clone();
            let captures_dir = captures_dir.clone();
            let parent = button.root().and_downcast::<gtk4::Window>();
            glib::timeout_add_seconds_local_once(2, move || {
                if shm.header().capture_request.load(Ordering::Relaxed) != 0 {
                    toasts.add_toast(adw::Toast::new("The capture is waiting for a game to present a frame"));
                    return;
                }
                let toast = adw::Toast::builder().title("Captured").button_label("Open folder").build();
                let toasts_for_button = toasts.clone();
                toast.connect_button_clicked(move |_| open_folder(&captures_dir, parent.clone(), &toasts_for_button));
                toasts.add_toast(toast);
            });
        });
    }

    // Holding before the upscaler: from the channel, while the layer's heartbeat is fresh.
    let shm = std::sync::Arc::clone(shm);
    let beat = std::cell::Cell::new((0u32, std::time::Instant::now()));
    glib::timeout_add_seconds_local(1, move || {
        let hdr = shm.header();
        let now = hdr.layer_heartbeat.load(Ordering::Relaxed);
        let (seen, at) = beat.get();
        let active = if now != seen {
            beat.set((now, std::time::Instant::now()));
            seen != 0
        } else {
            at.elapsed() < std::time::Duration::from_secs(2)
        };
        capture_row.set_subtitle(capture_subtitle(active && hdr.preupscale_state.load(Ordering::Relaxed) == 2));
        glib::ControlFlow::Continue
    });
    group
}

/// "Save report": the diagnostic report (`neural_forge_supervisor::report`) on the desktop.
fn build_report_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Report");
    let row = adw::ActionRow::new();
    row.set_title("Diagnostic report");
    row.set_subtitle(
        "Saves a zip to the desktop to attach to a problem report: system details, settings, the layer's logs and \
         the newest capture. Your home folder and user name are replaced; nothing is uploaded",
    );
    let button = gtk4::Button::with_label("Save report");
    button.set_valign(gtk4::Align::Center);
    row.add_suffix(&button);
    group.add(&row);

    let shm = std::sync::Arc::clone(shm);
    let toasts = toasts.clone();
    button.connect_clicked(move |button| {
        // The channel is read here, on the main thread; the files and probes on a worker.
        let status = neural_forge_supervisor::shm_status::status_text(shm.header());
        let game = shm.header().game_name();
        let toasts = toasts.clone();
        let button = button.clone();
        button.set_sensitive(false);
        glib::spawn_future_local(async move {
            let saved = gio::spawn_blocking(move || neural_forge_supervisor::report::save(&status, &game, None, None, None)).await;
            button.set_sensitive(true);
            match saved {
                Ok(Ok(path)) => {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let dir = path.parent().map(|d| d.display().to_string()).unwrap_or_default();
                    let toast = adw::Toast::builder()
                        .title(glib::markup_escape_text(&format!("Saved {name} to {dir}")))
                        .button_label("Show")
                        .timeout(10)
                        .build();
                    let parent = button.root().and_downcast::<gtk4::Window>();
                    toast.connect_button_clicked(move |_| {
                        gtk4::FileLauncher::new(Some(&gio::File::for_path(&path))).open_containing_folder(parent.as_ref(), None::<&gio::Cancellable>, |_| {});
                    });
                    toasts.add_toast(toast);
                }
                Ok(Err(e)) => toasts.add_toast(adw::Toast::new(&glib::markup_escape_text(&format!("Saving the report failed: {e}")))),
                Err(_) => toasts.add_toast(adw::Toast::new("Saving the report failed: the worker panicked")),
            }
        });
    });
    group
}

/// The Status page's one line on the pre-upscaler path, from the header's `preupscale_*` fields
/// (0 off, 1 waiting for DLSS's input, 2 holding). `layer_active`: the layer's heartbeat is fresh;
/// otherwise the fields are a closed game's leftovers.
fn preupscale_label(layer_active: bool, state: u32, width: u32, height: u32, misses: u32) -> String {
    let missed = if misses == 0 { String::new() } else { format!(", {misses} frame{} missed", if misses == 1 { "" } else { "s" }) };
    match state {
        _ if !layer_active => "no game running".to_string(),
        2 => format!("before the upscaler: holding DLSS's {width}x{height} input every frame{missed}"),
        1 => format!("after the upscaler: waiting for DLSS Super Resolution{missed}"),
        _ => "after the upscaler".to_string(),
    }
}

fn build_error_window(app: &adw::Application, error: &neural_forge_protocol::mapping::OpenError) {
    use neural_forge_protocol::mapping::OpenError;
    let (title, description) = match error {
        OpenError::WrongVersion { found } => (
            "Another Neural Forge version is running",
            format!(
                "The shared memory was set up by a build that speaks version {found}; this app speaks {}. \
                 Close the game if one is running: the next game started with this version re-creates it, and Neural Forge then opens normally.",
                neural_forge_protocol::SHM_VERSION
            ),
        ),
        OpenError::Unavailable => (
            "Couldn't open the shared-memory mapping",
            "neural-forge-cli doctor may help.".to_string(),
        ),
        OpenError::Foreign => (
            "The shared-memory path names another file",
            "The configured channel path (shm= in config.ini, or NEURAL_FORGE_SHM) points at an existing file that is not a \
             Neural Forge mapping. It was left untouched. Point the setting at another path, or remove it to use the default."
                .to_string(),
        ),
    };
    let status = adw::StatusPage::builder().icon_name("dialog-error-symbolic").title(title).description(description).build();
    let window = adw::ApplicationWindow::builder().application(app).title("Neural Forge").content(&status).build();
    window.present();
}

#[cfg(test)]
mod preupscale_label_tests {
    use super::preupscale_label;

    #[test]
    fn says_where_the_model_runs_in_one_line() {
        assert_eq!(preupscale_label(false, 2, 1485, 836, 3), "no game running");
        assert_eq!(preupscale_label(true, 0, 0, 0, 0), "after the upscaler");
        assert_eq!(preupscale_label(true, 1, 0, 0, 0), "after the upscaler: waiting for DLSS Super Resolution");
        assert_eq!(preupscale_label(true, 2, 1485, 836, 0), "before the upscaler: holding DLSS's 1485x836 input every frame");
        assert_eq!(preupscale_label(true, 2, 1485, 836, 1), "before the upscaler: holding DLSS's 1485x836 input every frame, 1 frame missed");
        assert_eq!(preupscale_label(true, 1, 1485, 836, 85), "after the upscaler: waiting for DLSS Super Resolution, 85 frames missed");
    }
}

#[cfg(test)]
mod capture_tests {
    use super::capture_subtitle;

    #[test]
    fn the_capture_row_says_when_there_is_no_before_image() {
        assert!(capture_subtitle(true).contains("before the upscaler"));
        assert!(capture_subtitle(false).contains("two PNGs"));
    }
}

#[cfg(test)] mod hotkey_tests {
    use super::*;
    #[test] fn hardware_codes_are_converted_without_underflow() {
        assert_eq!(evdev_keycode(95),Some(87)); // F11: XKB -> Linux evdev.
        assert_eq!(evdev_keycode(38),Some(30)); // physical A key on evdev.
        assert_eq!(evdev_keycode(0),None);
        assert_eq!(evdev_keycode(8),None);
        assert_eq!(evdev_keycode(u32::MAX),None);
    }
    #[test] fn key_labels_use_the_key_name_and_fall_back_to_the_code() {
        assert_eq!(format_key_label(87, Some("F11")), "F11");
        assert_eq!(format_key_label(30, Some("a")), "A");
        assert_eq!(format_key_label(70, Some("Scroll_Lock")), "Scroll_Lock");
        assert_eq!(format_key_label(87, None), "Key 87");
        assert_eq!(key_label(0), "Unbound");
    }
}

#[cfg(test)]
mod launch_option_tests {
    use super::*;

    #[test]
    fn matches_the_documented_baseline_with_no_target_exe() {
        assert_eq!(launch_option(""), "NEURAL_FORGE_ENABLE=1 %command%");
    }

    #[test]
    fn includes_target_exe_when_given() {
        assert_eq!(launch_option("GTA5_Enhanced.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%");
    }

    #[test]
    fn trims_whitespace_around_target_exe() {
        assert_eq!(launch_option("  GTA5_Enhanced.exe  "), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%");
    }

    #[test]
    fn whitespace_only_target_exe_is_treated_as_empty() {
        assert_eq!(launch_option("   "), "NEURAL_FORGE_ENABLE=1 %command%");
    }

    #[test]
    fn quotes_a_target_exe_with_a_space_or_a_quote() {
        assert_eq!(launch_option("My Game.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='My Game.exe' %command%");
        assert_eq!(launch_option("it's.exe"), "NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE='it'\\''s.exe' %command%");
    }
}
