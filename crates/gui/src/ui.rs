//! The settings UI: one row per `ShmHeader` setting, grouped and tabbed the way
//! upstream's Qt GUI does (Model, Motion, Composition, Status) — used as a checklist
//! of what has to exist, not as layout code to port; a per-field `QCheckBox`/
//! `QSpinBox` binder has no logic worth transliterating either way.

use std::sync::atomic::Ordering;

use gtk4::prelude::*;
use libadwaita as adw;
use libadwaita::prelude::*;
use neural_forge_protocol::enums::{colour_mode, reversible_mode};
use neural_forge_supervisor::model::Network;

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

/// A toast showing `text` as written. libadwaita reads a toast's title as Pango markup, so a game,
/// profile or file name holding `&` or `<` would otherwise be shown wrong or not at all: every
/// toast goes through here.
fn toast(text: &str) -> adw::Toast {
    adw::Toast::new(&glib::markup_escape_text(text))
}

/// Whether a game's layer is alive: its heartbeat (`layer_heartbeat`) changed within the last two
/// seconds. A closed game leaves its last values in the channel; only a moving heartbeat says they
/// are current. Each timer that asks keeps its own.
#[derive(Default)]
struct Liveness(std::cell::Cell<(Option<u32>, Option<std::time::Instant>)>);

impl Liveness {
    fn alive(&self, hdr: &neural_forge_protocol::ShmHeader) -> bool {
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

thread_local! {
    /// Set when the window closes: every periodic update ([`every`]) stops then.
    static STOPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `update` now and then every `period` for as long as `owner` exists and the window is open.
/// The timer holds `owner` weakly, so it neither keeps a closed page's widgets alive nor runs on
/// after them.
fn every<W: IsA<glib::Object>>(owner: &W, period: std::time::Duration, update: impl Fn(&W) + 'static) {
    update(owner);
    let owner = owner.downgrade();
    glib::timeout_add_local(period, move || match owner.upgrade() {
        Some(owner) if !STOPPED.with(std::cell::Cell::get) => {
            update(&owner);
            glib::ControlFlow::Continue
        }
        _ => glib::ControlFlow::Break,
    });
}

const SECOND: std::time::Duration = std::time::Duration::from_secs(1);

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
    let row = adw::SpinRow::new(Some(&adjustment), step, 0);
    keep_spin_precision(&row, &adjustment, step);
    row.set_title(title);
    row.set_subtitle(subtitle);
    adjustment.connect_value_changed(move |adj| {
        if !refreshing() {
            setter((adj.value() / scale) as f32);
        }
    });
    if let Some((kind, read)) = crate::shm::take_pending_reader() {
        mark_game_override(row.upcast_ref());
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

/// GtkSpinButton parses its text again on Enter and when it loses the focus. A value with more
/// decimals than the row shows would come back rounded and be written to the header and config.ini
/// though nobody edited it. So the row shows the value's own decimals (`spin_digits`), and text that
/// still shows the current value keeps it exactly (`kept_spin_value`).
fn keep_spin_precision(row: &adw::SpinRow, adjustment: &gtk4::Adjustment, step: f64) {
    row.set_digits(spin_digits(step, adjustment.value()));
    {
        let adjustment = adjustment.clone();
        row.connect_input(move |row| kept_spin_value(&row.text(), adjustment.value(), row.digits()).map(Ok));
    }
    let row = row.downgrade();
    adjustment.connect_value_changed(move |adj| {
        if let Some(row) = row.upgrade() {
            row.set_digits(spin_digits(step, adj.value()));
        }
    });
}

/// The decimals a spin row shows: as many as its step has, and more (up to 3) while the value has
/// them, so a value set off the step by shmctl, a profile or config.ini (16.5 on a step-1 row, 0.004
/// on a step-0.01 one) is shown as it is.
fn spin_digits(step: f64, value: f64) -> u32 {
    decimals(step).max(decimals(value))
}

/// The fewest decimals (0 to 3) that write `x` exactly, 3 when it has more.
fn decimals(x: f64) -> u32 {
    (0..3)
        .find(|&d| {
            let shifted = x * 10f64.powi(d as i32);
            (shifted - shifted.round()).abs() <= 1e-6 * shifted.abs().max(1.0)
        })
        .unwrap_or(3)
}

/// The value a spin row keeps when GTK re-reads its `text`: the current `value` while the text still
/// shows it at `digits` decimals (either decimal mark), so precision the row does not show survives
/// a focus change; `None` lets GTK parse an edited text as usual.
fn kept_spin_value(text: &str, value: f64, digits: u32) -> Option<f64> {
    let typed: f64 = text.trim().replace(',', ".").parse().ok()?;
    let d = digits as usize;
    (format!("{typed:.d$}") == format!("{value:.d$}")).then_some(value)
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
        mark_game_override(row.upcast_ref());
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
        mark_game_override(row.upcast_ref());
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

/// A model resolution menu (`enums::model_scale::PRESETS`, then "Custom" for a value set elsewhere, e.g. with
/// `shmctl`): choosing a preset writes its scale. Follows the header like the other rows.
fn scale_row(title: &str, subtitle: &str, value: f32, setter: impl Fn(f32) + 'static) -> adw::ComboRow {
    use neural_forge_protocol::enums::model_scale::{preset_of, PRESETS};
    let mut names: Vec<&str> = PRESETS.iter().map(|&(n, _)| n).collect();
    names.push("Custom");
    let custom = PRESETS.len() as u32;
    let index = move |v: f32| preset_of(v).map_or(custom, |i| i as u32);
    let row = adw::ComboRow::new();
    row.set_title(title);
    row.set_subtitle(subtitle);
    row.set_model(Some(&gtk4::StringList::new(&names)));
    row.set_selected(index(value));
    row.connect_selected_notify(move |row| {
        if let Some(&(_, scale)) = PRESETS.get(row.selected() as usize) {
            if !refreshing() {
                setter(scale);
            }
        }
    });
    if let Some((_, read)) = crate::shm::take_pending_reader() {
        mark_game_override(row.upcast_ref());
        let row = row.downgrade();
        register_refresher(move || {
            if let Some(row) = row.upgrade() {
                let v = index(f32::from_bits(read()));
                if row.selected() != v {
                    row.set_selected(v);
                }
            }
        });
    }
    row
}

/// The strength preset menu: choosing a preset writes its four values to the header as one unit
/// (`Knobs::write`, so the layer never sees half of a preset) and saves them; the four rows follow
/// the header. It shows the preset the header matches, else "Custom".
fn preset_row(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, rows: &[adw::SpinRow]) -> adw::ComboRow {
    use neural_forge_supervisor::presets::{matching, Knobs, KNOBS, PRESETS};
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.push("Custom");
    let custom = PRESETS.len() as u32;
    let row = adw::ComboRow::new();
    row.set_title("Strength preset");
    row.set_subtitle("Sets the four values below. Reference is Neural Forge's defaults");
    row.set_model(Some(&gtk4::StringList::new(&names)));
    // Set while this code changes the menu, so it does not react to itself.
    let syncing = std::rc::Rc::new(std::cell::Cell::new(false));
    let sync = {
        let row = row.downgrade();
        let shm = std::sync::Arc::clone(shm);
        let syncing = std::rc::Rc::clone(&syncing);
        std::rc::Rc::new(move || {
            let Some(row) = row.upgrade() else { return };
            let shown = matching(&Knobs::from_header(shm.header())).map_or(custom, |i| i as u32);
            if row.selected() != shown {
                syncing.set(true);
                row.set_selected(shown);
                syncing.set(false);
            }
        })
    };
    sync();
    {
        let shm = std::sync::Arc::clone(shm);
        let syncing = std::rc::Rc::clone(&syncing);
        let sync = std::rc::Rc::clone(&sync);
        row.connect_selected_notify(move |row| {
            if syncing.get() || refreshing() {
                return;
            }
            // "Custom" sets nothing; the menu goes back to what the header matches.
            if let Some(preset) = PRESETS.get(row.selected() as usize) {
                preset.knobs.write(shm.header());
                crate::shm::persist_settings_later(&shm, &KNOBS);
                refresh_all_rows();
            }
            sync();
        });
    }
    // A value changed by hand, by a profile, Reset or `shmctl` (the rows follow the header).
    for spin in rows {
        let sync = std::rc::Rc::clone(&sync);
        spin.connect_value_notify(move |_| sync());
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
    let row = adw::ActionRow::builder()
        .title("Toggle key")
        .subtitle("Toggles Neural Forge inside a running game. Needs read access to /dev/input (the 'input' group) or an X11/XWayland session.")
        .build();
    if reader.is_some() {
        mark_game_override(&row);
    }
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
            if !capturing.get() {
                return glib::Propagation::Proceed;
            }
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

    let strength_rows = [intensity_row, local_tone_row, local_structure_row, skin_structure_row];
    model_group.add(&preset_row(&shm, &strength_rows));
    for row in &strength_rows {
        model_group.add(row);
    }

    let (auto_mask, set_auto_mask) = bind_bool(&shm, Some("auto_mask"), |h| &h.auto_mask);
    model_group.add(&switch_row("Auto mask", "Automatic skin/detail masking", auto_mask, set_auto_mask));

    let (passes, set_passes) = bind_u32(&shm, Some("passes"), |h| &h.passes);
    model_group.add(&spin_row(
        "Passes",
        "How many times the model runs over each frame. Every pass after the first works on the previous pass's picture and costs the model's time again",
        passes as f32,
        "passes",
        1.0,
        move |v| set_passes(v as u32),
    ));
    let (feedback, set_feedback) = bind_float(&shm, Some("pass_feedback"), |h| &h.pass_feedback_bits);
    model_group.add(&spin_row(
        "Pass feedback",
        "How much of a pass's picture the next pass works on: 1 is that picture, lower keeps more of the pass's own input",
        feedback,
        "pass_feedback",
        0.05,
        set_feedback,
    ));
    let (placement, set_placement) = bind_u32(&shm, Some("placement"), |h| &h.placement);
    model_group.add(&placement_row(placement, set_placement));
    let (scale, set_scale) = bind_float(&shm, Some("preupscale_scale"), |h| &h.preupscale_scale_bits);
    model_group.add(&scale_row(
        "Model resolution",
        "The size the model runs at before the upscaler, of DLSS's input size. Full gives the most detail; the others are faster, and the model's edit is brought back to full size",
        scale,
        set_scale,
    ));
    // What the model sees before the upscaler: the picture's brightness for it alone (the game's is kept).
    let (exposure, set_exposure) = bind_u32(&shm, Some("preupscale_exposure"), |h| &h.preupscale_exposure);
    let exposure_row = combo_row("Model brightness", &["Game", "Auto"], exposure, set_exposure);
    exposure_row.set_subtitle(
        "How bright the picture the model works on is (the game's own brightness is kept). Game: the game's exposure. Auto: measured from the frame (within half to 16 times the game's), for games whose picture makes the model darken it",
    );
    model_group.add(&exposure_row);
    let (brightness, set_brightness) = bind_float(&shm, Some("preupscale_brightness"), |h| &h.preupscale_brightness_bits);
    model_group.add(&spin_row(
        "Model brightness scale",
        "On top of the above: above 1 the model sees a brighter picture, below 1 a darker one",
        brightness,
        "preupscale_brightness",
        0.05,
        set_brightness,
    ));
    let pass_row = adw::ActionRow::new();
    pass_row.set_title("Per-pass settings");
    pass_row.set_subtitle("Give individual passes their own values");
    let pass_button = gtk4::Button::with_label("Edit…");
    pass_button.set_valign(gtk4::Align::Center);
    pass_row.add_suffix(&pass_button);
    pass_row.set_activatable_widget(Some(&pass_button));
    {
        let shm = std::sync::Arc::clone(&shm);
        pass_button.connect_clicked(move |b| open_pass_dialog(&shm, b.upcast_ref()));
    }
    model_group.add(&pass_row);

    // The control mask (before the upscaler): the sky rule, and the optional dark and motion rules.
    let (mask_sky, set_mask_sky) = bind_bool(&shm, Some("mask_sky"), |h| &h.mask_sky);
    model_group.add(&switch_row("Keep the game's sky", "No edit where the game draws its sky (from the depth, before the upscaler)", mask_sky, set_mask_sky));
    let mask_more = adw::ExpanderRow::new();
    mask_more.set_title("More mask rules");
    mask_more.set_subtitle("Edit less in dark or fast-moving parts of the picture (before the upscaler)");
    let (mask_dark, set_mask_dark) = bind_bool(&shm, Some("mask_dark"), |h| &h.mask_dark);
    mask_more.add_row(&switch_row("Dark areas", "Below the brightness, fade the edit toward the values below", mask_dark, set_mask_dark));
    let (v, set) = bind_float(&shm, Some("mask_dark_threshold"), |h| &h.mask_dark_threshold_bits);
    mask_more.add_row(&spin_row("Dark brightness", "0 is black, 1 white", v, "mask_dark_threshold", 0.01, set));
    let (v, set) = bind_float(&shm, Some("mask_dark_intensity"), |h| &h.mask_dark_intensity_bits);
    mask_more.add_row(&spin_row("Dark intensity", "Of the intensity, at black", v, "mask_dark_intensity", 0.05, set));
    let (v, set) = bind_float(&shm, Some("mask_dark_structure"), |h| &h.mask_dark_structure_bits);
    mask_more.add_row(&spin_row("Dark structure", "Of the structure, at black", v, "mask_dark_structure", 0.05, set));
    let (mask_motion, set_mask_motion) = bind_bool(&shm, Some("mask_motion"), |h| &h.mask_motion);
    mask_more.add_row(&switch_row("Fast motion", "Above the motion, fade the edit toward the values below", mask_motion, set_mask_motion));
    let (v, set) = bind_float(&shm, Some("mask_motion_threshold"), |h| &h.mask_motion_threshold_bits);
    mask_more.add_row(&spin_row("Motion", "Pixels per frame where the fade starts (full at twice)", v, "mask_motion_threshold", 1.0, set));
    let (v, set) = bind_float(&shm, Some("mask_motion_intensity"), |h| &h.mask_motion_intensity_bits);
    mask_more.add_row(&spin_row("Motion intensity", "Of the intensity, at twice the motion", v, "mask_motion_intensity", 0.05, set));
    let (v, set) = bind_float(&shm, Some("mask_motion_structure"), |h| &h.mask_motion_structure_bits);
    mask_more.add_row(&spin_row("Motion structure", "Of the structure, at twice the motion", v, "mask_motion_structure", 0.05, set));
    model_group.add(&mask_more);

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
    comp_group.add(&spin_row(
        "Detail strength",
        "How much of the model's edit reaches the frame; above 1 amplifies it",
        transfer_strength,
        "transfer_strength",
        0.05,
        set_transfer_strength,
    ));

    let (colour_strength, set_colour_strength) = bind_float(&shm, Some("colour_strength"), |h| &h.colour_strength_bits);
    comp_group.add(&spin_row(
        "Colour strength",
        "How much of the transfer is allowed to be colour, not just luminance",
        colour_strength,
        "colour_strength",
        0.05,
        set_colour_strength,
    ));

    let (max_ratio, set_max_ratio) = bind_float(&shm, Some("max_ratio"), |h| &h.max_ratio_bits);
    comp_group.add(&spin_row("Highlight guard", "The most the pass may brighten or darken a pixel by (x)", max_ratio, "max_ratio", 0.1, set_max_ratio));

    let (working_scale, set_working_scale) = bind_float(&shm, Some("working_scale"), |h| &h.working_scale_bits);
    let working_scale_row = spin_row_scaled(
        "Model resolution after the upscaler",
        "Percent of the frame the model works at; lower is faster, softer and flickers more",
        working_scale,
        "working_scale",
        5.0,
        100.0,
        set_working_scale,
    );
    comp_group.add(&working_scale_row);

    let (reversible, set_reversible) = bind_u32(&shm, Some("reversible_mode"), |h| &h.reversible_mode);
    comp_group.add(&combo_row("Reversible mode", &["Knee", "Neutwo", "Neutwo replace", "Hybrid", "Hybrid replace"], reversible, set_reversible));
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
    comp_group.add(&switch_row(
        "Apply model edit",
        "Off presents the clean frame — capture/transport/round-trip still run, for an honest A/B",
        apply_model,
        set_apply_model,
    ));

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
            "Edit cache: rejected (before the upscaler)",
        ],
        debug_view,
        set_debug_view,
    );
    debug_view_row.set_subtitle(
        "Colour trust's two only differ from the composited picture where colour trust is actually engaged. The edit cache's view replaces the frames that reuse the network's last result before the upscaler: red where it was rejected, green by how much of its detail was kept, blue where the motion left the frame",
    );
    debug_group.add(&debug_view_row);

    // The edit cache's counters (before the upscaler: the network on every frame while the camera is
    // still, every second frame while it moves).
    let cache_row = adw::ActionRow::builder().title("Edit cache").subtitle("Not running").build();
    debug_group.add(&cache_row);
    {
        let shm = std::sync::Arc::clone(&shm);
        let live = Liveness::default();
        every(&cache_row, SECOND, move |cache_row| {
            let hdr = shm.header();
            let refresh = hdr.cache_refresh_holds.load(Ordering::Relaxed);
            let cached = hdr.cache_cached_holds.load(Ordering::Relaxed);
            // A closed game's counters stay in the channel: only a live layer's are current.
            let text = if !live.alive(hdr) || (refresh == 0 && cached == 0) {
                "Not running".to_string()
            } else {
                format!(
                    "{refresh} network frames, {cached} reused; {:.1}% rejected on the last reused frame; cache passes {:.3} ms",
                    f64::from(hdr.cache_rejected.load(Ordering::Relaxed)) / 100.0,
                    f32::from_bits(hdr.cache_gpu_ms_bits.load(Ordering::Relaxed))
                )
            };
            cache_row.set_subtitle(&text);
        });
    }

    // Scene cuts before the upscaler: frames where the picture changed as a whole, so the model started
    // afresh instead of blending in the previous scene.
    let cuts_row = adw::ActionRow::builder().title("Scene cuts").subtitle("Not running").build();
    debug_group.add(&cuts_row);
    {
        let shm = std::sync::Arc::clone(&shm);
        let live = Liveness::default();
        every(&cuts_row, SECOND, move |cuts_row| {
            let hdr = shm.header();
            // Counted only while the model runs before the upscaler (the edit cache counts its frames then).
            let running = hdr.cache_refresh_holds.load(Ordering::Relaxed) != 0;
            let text = if !live.alive(hdr) || !running {
                "Not running".to_string()
            } else {
                match hdr.scene_cuts.load(Ordering::Relaxed) {
                    1 => "1 frame where the picture changed completely: the model started afresh there".to_string(),
                    n => format!("{n} frames where the picture changed completely: the model started afresh there"),
                }
            };
            cuts_row.set_subtitle(&text);
        });
    }

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
        toasts.add_toast(adw::Toast::builder().title(glib::markup_escape_text(&message)).timeout(0).build());
    }

    // Tabbed like upstream's Qt GUI, rather than one long scrolling page -- each tab
    // is still an AdwPreferencesPage, which scrolls internally on its own if its
    // content overflows the window.
    let view_stack = adw::ViewStack::new();
    view_stack.set_vexpand(true);

    let model_page = adw::PreferencesPage::new();
    model_page.add(&model_group);
    model_page.add(&build_colour_group(&shm, &toasts));
    model_page.add(&build_hud_group(&shm));
    model_page.add(&build_sharpen_group(&shm));
    view_stack.add_titled_with_icon(&model_page, Some("model"), "Model", "applications-graphics-symbolic");

    // Model resolution and model-every-Nth-frame apply after the upscaler only: shown while a game runs
    // and the model is not running before the upscaler.
    {
        let shm = std::sync::Arc::clone(&shm);
        let live = Liveness::default();
        let working_scale_row = working_scale_row.clone();
        every(&interval_row, SECOND, move |interval_row| {
            let hdr = shm.header();
            let after_upscaler = live.alive(hdr) && hdr.native_running.load(Ordering::Relaxed) == 0;
            interval_row.set_visible(after_upscaler);
            working_scale_row.set_visible(after_upscaler);
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

    let setup_page = build_setup_page(&shm, &toasts);

    let status_page = adw::PreferencesPage::new();
    status_page.add(&build_telemetry_group(&shm));
    let status_group = build_status_group(&shm, &toasts);
    status_page.add(&status_group);
    status_page.add(&build_diagnosis_group(&shm, &toasts));
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

    let scope_bar = build_scope_bar(&shm, &toasts);
    // The Setup tab edits no setting.
    {
        let scope_bar = scope_bar.clone();
        view_stack.connect_visible_child_name_notify(move |stack| scope_bar.set_visible(stack.visible_child_name().as_deref() != Some("setup")));
    }
    scope_bar.set_visible(view_stack.visible_child_name().as_deref() != Some("setup"));

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.add_top_bar(&banner);
    toolbar_view.add_top_bar(&scope_bar);
    toolbar_view.set_content(Some(&view_stack));
    toolbar_view.add_bottom_bar(&switcher_bar);
    toasts.set_child(Some(&toolbar_view));

    // Developer aid: `NEURAL_FORGE_GUI_OPEN=<tab id>` (model/composition/debug/status/setup) selects
    // that tab at startup, and `NEURAL_FORGE_GUI_OPEN=passes` opens the per-pass dialog, so either can be
    // screenshotted where synthetic input cannot reliably reach a page navigated to after the window opens.
    match neural_forge_protocol::env::var("NEURAL_FORGE_GUI_OPEN").as_deref() {
        Some("passes") => {
            let shm = std::sync::Arc::clone(&shm);
            let view = view_stack.clone();
            glib::timeout_add_local_once(std::time::Duration::from_millis(500), move || open_pass_dialog(&shm, view.upcast_ref()));
        }
        Some(tab @ ("model" | "composition" | "debug" | "status" | "setup")) => view_stack.set_visible_child_name(tab),
        _ => {}
    }

    // Follow the header: a change made with `shmctl`, a loaded profile, Reset or another
    // instance appears here within a second instead of needing an app restart.
    every(&toasts, SECOND, |_| refresh_all_rows());

    let window = adw::ApplicationWindow::builder().application(app).title("Neural Forge").default_width(620).default_height(700).content(&toasts).build();

    {
        let toasts = toasts.clone();
        crate::shm::on_save_error(move |message| toasts.add_toast(toast(&message)));
    }
    // A change made in the last moment before closing is still saved; the periodic updates stop.
    window.connect_close_request(|_| {
        STOPPED.with(|s| s.set(true));
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
    breakpoint.add_setters(&[(switcher.upcast_ref::<gtk4::Widget>(), "visible", false), (switcher_bar.upcast_ref::<gtk4::Widget>(), "reveal", true)]);
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
            dialog.add_credit_section(Some("Built with"), &["Claude Code (Anthropic)", "Codex (OpenAI)"]);
            dialog.present(Some(&window));
        });
    }

    window.present();
}

/// The state the rows' per-game marks follow: the game whose settings are shown (`None`: all games'), its
/// own settings, and what a mark's button needs to drop one.
#[derive(Default)]
struct OverrideMarks {
    /// Each marked row's setting and its mark (held weakly: the per-pass dialog's rows come and go).
    marks: Vec<(&'static str, glib::WeakRef<gtk4::Box>)>,
    game: Option<neural_forge_supervisor::game_settings::GameKey>,
    own: neural_forge_supervisor::game_settings::Section,
    shm: Option<std::sync::Arc<neural_forge_protocol::mapping::Mapping>>,
    toasts: Option<adw::ToastOverlay>,
}

thread_local! {
    static OVERRIDE_MARKS: std::cell::RefCell<OverrideMarks> = std::cell::RefCell::new(OverrideMarks::default());
}

/// Whether the row of setting `name` shows the "This game" mark: a game is picked and it has its own value of
/// that setting. Rows following All games (and every row while All games is picked) stay unmarked.
fn override_marked(name: &str, game: Option<&neural_forge_supervisor::game_settings::GameKey>, own: &neural_forge_supervisor::game_settings::Section) -> bool {
    game.is_some() && own.contains_key(&format!("set_{name}"))
}

/// Gives `row` (built for the setting its binding named) the per-game mark: a "This game" tag and a button
/// that drops this one setting of the game's own, shown while the game picked in "Settings for" has its own
/// value of it.
fn mark_game_override(row: &adw::ActionRow) {
    let Some(name) = crate::shm::take_pending_name() else { return };
    let mark = gtk4::Box::new(gtk4::Orientation::Horizontal, 4);
    mark.set_valign(gtk4::Align::Center);
    let tag = gtk4::Label::new(Some("This game"));
    tag.add_css_class("caption-heading");
    tag.add_css_class("accent");
    tag.set_tooltip_text(Some("This game's own value: the All games value does not apply to it"));
    let drop = gtk4::Button::from_icon_name("edit-undo-symbolic");
    drop.add_css_class("flat");
    drop.add_css_class("circular");
    drop.set_valign(gtk4::Align::Center);
    drop.set_tooltip_text(Some("Use the All games value again"));
    drop.connect_clicked(move |_| drop_game_override(name));
    mark.append(&tag);
    mark.append(&drop);
    // Beside the row's own control, so the titles of marked and unmarked rows stay in line.
    row.add_suffix(&mark);
    OVERRIDE_MARKS.with(|m| {
        let mut m = m.borrow_mut();
        mark.set_visible(override_marked(name, m.game.as_ref(), &m.own));
        m.marks.push((name, mark.downgrade()));
    });
}

/// Shows each row's mark for `game`'s own settings `own` (`None`: all games', no marks).
fn update_override_marks(
    shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>,
    toasts: &adw::ToastOverlay,
    game: Option<&neural_forge_supervisor::game_settings::GameKey>,
    own: neural_forge_supervisor::game_settings::Section,
) {
    OVERRIDE_MARKS.with(|m| {
        let mut m = m.borrow_mut();
        m.shm.get_or_insert_with(|| std::sync::Arc::clone(shm));
        m.toasts.get_or_insert_with(|| toasts.clone());
        m.game = game.cloned();
        m.own = own;
        let OverrideMarks { marks, game, own, .. } = &mut *m;
        marks.retain(|(name, mark)| match mark.upgrade() {
            Some(mark) => {
                let shown = override_marked(name, game.as_ref(), own);
                if mark.is_visible() != shown {
                    mark.set_visible(shown);
                }
                true
            }
            None => false,
        });
    });
}

/// A mark's button: drops the picked game's own value of `name` (`game-settings clear GAME NAME`), so the
/// game follows All games for it again, and shows (and runs, if the channel holds the game's settings) the
/// result.
fn drop_game_override(name: &'static str) {
    use neural_forge_supervisor::game_settings;
    let (game, shm, toasts) = OVERRIDE_MARKS.with(|m| {
        let m = m.borrow();
        (m.game.clone(), m.shm.clone(), m.toasts.clone())
    });
    let (Some(game), Some(shm)) = (game, shm) else { return };
    let say = |text: String| match &toasts {
        Some(toasts) => toasts.add_toast(toast(&text)),
        None => eprintln!("neural-forge: {text}"),
    };
    // A change still waiting is saved first, so it cannot write the value back.
    if let Err(e) = crate::shm::flush_settings() {
        say(format!("Saving the settings failed: {e}"));
    }
    match game_settings::clear(&game, Some(&format!("set_{name}"))) {
        Ok(_) => {
            if shm.header().settings_game() == game.scope() {
                neural_forge_protocol::persist::apply_scope(shm.header(), Some(&game));
                refresh_all_rows();
            }
            let own = game_settings::overrides(&game_settings::load_all(), &game);
            let toasts = OVERRIDE_MARKS.with(|m| m.borrow().toasts.clone());
            if let Some(toasts) = &toasts {
                update_override_marks(&shm, toasts, Some(&game), own);
            }
            say(format!("{name} follows All games again"));
        }
        Err(e) => say(format!("Not changed: {e}")),
    }
}

/// One entry of the "Settings for" menu: all games (`None`) or one game, and its label.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopeEntry {
    game: Option<neural_forge_supervisor::game_settings::GameKey>,
    label: String,
}

/// The "Settings for" menu's entries: All games, the installed Steam games by name (`steam`: app id and
/// name), then every other game with settings of its own, and `extra` (the game on the channel, the game
/// whose settings it holds), each once.
fn scope_entries(
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
fn scope_summary(game: Option<&neural_forge_supervisor::game_settings::GameKey>, own: usize, shipped: usize, running: bool) -> String {
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

/// The bar under the header: whose settings the tabs show and edit ("Settings for": All games or one game)
/// and "Restore defaults" for a game. The tabs show the header's settings; choosing an entry puts that
/// game's merge on the channel (`persist::apply_scope`), so the game running shows them while you edit, and
/// every change is saved to the entry chosen (`game_settings::save_live`). A game's session starting puts
/// its own settings on the channel, and the menu follows the header.
fn build_scope_bar(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> gtk4::Box {
    use neural_forge_supervisor::game_settings::{self, GameKey};
    // The menu and the button on one line, what the entry holds on the line below.
    let bar = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    bar.add_css_class("toolbar");
    bar.set_margin_start(6);
    bar.set_margin_end(6);
    let line = gtk4::Box::new(gtk4::Orientation::Horizontal, 12);
    let title = gtk4::Label::new(Some("Settings for"));
    title.add_css_class("heading");
    let model = gtk4::StringList::new(&[]);
    let menu =
        gtk4::DropDown::new(Some(model.clone()), Some(gtk4::PropertyExpression::new(gtk4::StringObject::static_type(), None::<&gtk4::Expression>, "string")));
    menu.set_enable_search(true);
    menu.set_tooltip_text(Some("Whose settings the tabs show and change: all games', or one game's own"));
    menu.set_hexpand(true);
    let summary = gtk4::Label::new(None);
    summary.add_css_class("dim-label");
    summary.add_css_class("caption");
    summary.set_xalign(0.0);
    summary.set_wrap(true);
    summary.set_margin_start(6);
    summary.set_margin_bottom(6);
    let restore = gtk4::Button::with_label("Restore defaults");
    restore.set_tooltip_text(Some("Remove this game's own settings: it uses the settings for all games again"));
    line.append(&title);
    line.append(&menu);
    line.append(&restore);
    bar.append(&line);
    bar.append(&summary);

    let entries: std::rc::Rc<std::cell::RefCell<Vec<ScopeEntry>>> = std::rc::Rc::default();
    let steam: std::rc::Rc<std::cell::RefCell<Vec<(String, String)>>> = std::rc::Rc::default();
    // Set while this code changes the menu, so it does not react to itself.
    let syncing = std::rc::Rc::new(std::cell::Cell::new(false));
    let live = std::rc::Rc::new(Liveness::default());

    let selected_game = {
        let (menu, entries) = (menu.clone(), entries.clone());
        move || entries.borrow().get(menu.selected() as usize).and_then(|e| e.game.clone())
    };
    // The menu brought in line with the header: its entries (a game the header names that the menu lacks is
    // added), the entry whose settings it holds, the summary and the button.
    let sync = {
        let (menu, model, entries, steam, syncing, live, summary, restore, shm, toasts) = (
            menu.clone(),
            model.clone(),
            entries.clone(),
            steam.clone(),
            syncing.clone(),
            live.clone(),
            summary.clone(),
            restore.clone(),
            std::sync::Arc::clone(shm),
            toasts.clone(),
        );
        std::rc::Rc::new(move || {
            let hdr = shm.header();
            let sections = game_settings::load_all();
            let held = game_settings::held_game(hdr);
            // The game on the channel: by app id, or by the executable the layer names.
            let on_channel: Vec<GameKey> = GameKey::exe(&hdr.game_name()).into_iter().collect();
            let running = neural_forge_protocol::games::key_for_scope(hdr.running_game.load(Ordering::Relaxed), &sections, &on_channel);
            let extra: Vec<GameKey> = held.iter().chain(running.iter()).cloned().collect();
            let wanted = scope_entries(&steam.borrow(), &game_settings::games_with_settings(&sections), &extra);
            if *entries.borrow() != wanted {
                syncing.set(true);
                let labels: Vec<&str> = wanted.iter().map(|e| e.label.as_str()).collect();
                model.splice(0, model.n_items(), &labels);
                *entries.borrow_mut() = wanted;
                syncing.set(false);
            }
            let scope = hdr.settings_game();
            let at = entries.borrow().iter().position(|e| e.game.as_ref().map_or(0, GameKey::scope) == scope);
            if let Some(at) = at.map(|at| at as u32) {
                if menu.selected() != at {
                    syncing.set(true);
                    menu.set_selected(at);
                    syncing.set(false);
                }
            }
            let game = entries.borrow().get(menu.selected() as usize).and_then(|e| e.game.clone());
            let own_settings = game.as_ref().map(|g| game_settings::overrides(&sections, g)).unwrap_or_default();
            let own = own_settings.len();
            update_override_marks(&shm, &toasts, game.as_ref(), own_settings);
            let shipped = game.as_ref().and_then(neural_forge_protocol::games::shipped).map_or(0, <[_]>::len);
            let running_now = live.alive(hdr) && game.as_ref().map_or(0, GameKey::scope) == hdr.running_game.load(Ordering::Relaxed) && game.is_some();
            summary.set_text(&scope_summary(game.as_ref(), own, shipped, running_now));
            restore.set_sensitive(own > 0);
        })
    };
    sync();
    {
        let sync = std::rc::Rc::clone(&sync);
        every(&bar, SECOND, move |_| sync());
    }
    // The installed Steam games' names, read on a worker thread like the Setup tab's list.
    {
        let (steam, sync) = (steam.clone(), std::rc::Rc::clone(&sync));
        glib::spawn_future_local(async move {
            let found = gio::spawn_blocking(|| {
                use neural_forge_supervisor::steam;
                steam::games(&steam::roots()).into_iter().map(|g| (g.appid, g.name)).collect::<Vec<_>>()
            })
            .await;
            if let Ok(found) = found {
                *steam.borrow_mut() = found;
                sync();
            }
        });
    }
    {
        let (shm, toasts, syncing, sync, entries) = (std::sync::Arc::clone(shm), toasts.clone(), syncing.clone(), std::rc::Rc::clone(&sync), entries.clone());
        menu.connect_selected_notify(move |menu| {
            if syncing.get() {
                return;
            }
            let Some(entry) = entries.borrow().get(menu.selected() as usize).cloned() else { return };
            // A change still waiting is saved to the settings it was made in, before the header holds another's.
            if let Err(e) = crate::shm::flush_settings() {
                toasts.add_toast(toast(&format!("Saving the settings failed: {e}")));
            }
            neural_forge_protocol::persist::apply_scope(shm.header(), entry.game.as_ref());
            refresh_all_rows();
            sync();
        });
    }
    {
        let (shm, toasts, sync) = (std::sync::Arc::clone(shm), toasts.clone(), std::rc::Rc::clone(&sync));
        let label_of = {
            let (menu, entries) = (menu.clone(), entries.clone());
            move || entries.borrow().get(menu.selected() as usize).map(|e| e.label.clone()).unwrap_or_default()
        };
        restore.connect_clicked(move |button| {
            let Some(game) = selected_game() else { return };
            let name = label_of();
            let dialog = adw::AlertDialog::builder()
                .heading(format!("Restore the defaults for {name}?"))
                .body("Its own settings are removed: it uses the settings for all games again (with any Neural Forge has built in for it).")
                .default_response("cancel")
                .close_response("cancel")
                .build();
            dialog.add_response("cancel", "Cancel");
            dialog.add_response("restore", "Restore");
            dialog.set_response_appearance("restore", adw::ResponseAppearance::Destructive);
            let (shm, toasts, sync) = (std::sync::Arc::clone(&shm), toasts.clone(), std::rc::Rc::clone(&sync));
            dialog.choose(button.root().and_downcast::<gtk4::Window>().as_ref(), None::<&gio::Cancellable>, move |response| {
                if response != "restore" {
                    return;
                }
                if let Err(e) = crate::shm::flush_settings() {
                    toasts.add_toast(toast(&format!("Saving the settings failed: {e}")));
                }
                match game_settings::clear(&game, None) {
                    Ok(_) => {
                        // Shown (and running, if it is the game on the channel) as it now is.
                        if shm.header().settings_game() == game.scope() {
                            neural_forge_protocol::persist::apply_scope(shm.header(), Some(&game));
                            refresh_all_rows();
                        }
                        toasts.add_toast(toast(&format!("Restored the defaults for {name}")));
                    }
                    Err(e) => toasts.add_toast(toast(&format!("Not restored: {e}"))),
                }
                sync();
            });
        });
    }
    bar
}

/// The Model tab's Colour group: the colour of the network's edit (its change to the frame) for every pass
/// (the per-pass dialog overrides the three scales), the gamut guard, the dither after the upscaler, the
/// protections of each pass's result against its input (hue, near grey, shadows, highlights) and the
/// look-up table, whose `.cube` file is named by `edit_lut` in `config.ini` (the layer reads it itself).
fn build_colour_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Colour");
    group.set_description(Some("Shapes the model's change to the frame, not the frame: 1 leaves the change as the model made it"));
    let (v, set) = bind_float(shm, Some("edit_saturation"), |h| &h.edit_saturation_bits);
    group.add(&spin_row(
        "Saturation of the change",
        "Scales how much the model changes colour: 0 keeps the frame's colours, above 1 adds to the model's",
        v,
        "edit_saturation",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_contrast"), |h| &h.edit_contrast_bits);
    group.add(&spin_row(
        "Contrast of the change",
        "Scales how much the model brightens or darkens: 0 keeps the frame's brightness",
        v,
        "edit_contrast",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_detail"), |h| &h.edit_detail_bits);
    group.add(&spin_row(
        "Detail of the change",
        "Scales the fine part of the change against its broad part: 0 keeps only the broad change",
        v,
        "edit_detail",
        0.05,
        set,
    ));
    let (v, set) = bind_bool(shm, Some("edit_gamut_guard"), |h| &h.edit_gamut_guard);
    group.add(&switch_row(
        "Gamut guard",
        "Where a change would push a colour past what the frame can hold, shorten the change instead of clipping a channel (keeps the hue)",
        v,
        set,
    ));
    let (v, set) = bind_bool(shm, Some("edit_dither"), |h| &h.edit_dither);
    group.add(&switch_row(
        "Dither",
        "Against banding in smooth gradients after the upscaler: a fixed pattern of half a step before the 8-bit answer is rounded",
        v,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_hue_protection"), |h| &h.edit_hue_protection_bits);
    group.add(&spin_row(
        "Hue protection",
        "Keeps the frame's hues: the model may still change how colourful and how bright a colour is (0 off, 1 full)",
        v,
        "edit_hue_protection",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_neutral_protection"), |h| &h.edit_neutral_protection_bits);
    group.add(&spin_row(
        "Grey protection",
        "Keeps greys and near-greys free of a tint from the model (0 off, 1 full)",
        v,
        "edit_neutral_protection",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_shadow_protection"), |h| &h.edit_shadow_protection_bits);
    group.add(&spin_row(
        "Shadow protection",
        "Holds back the model darkening what was already dark, so shadow detail stays visible (0 off, 1 full)",
        v,
        "edit_shadow_protection",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_highlight_protection"), |h| &h.edit_highlight_protection_bits);
    group.add(&spin_row(
        "Highlight protection",
        "Holds back the model brightening what was already bright, so highlights do not wash out (0 off, 1 full)",
        v,
        "edit_highlight_protection",
        0.05,
        set,
    ));

    let lut_row = adw::ActionRow::new();
    lut_row.set_title("Look-up table");
    let current = || neural_forge_supervisor::Config::load().settings.get(neural_forge_protocol::cube::CONFIG_KEY).cloned().filter(|p| !p.is_empty());
    let describe = |path: Option<String>| path.unwrap_or_else(|| "None: choose a .cube file (17 to 65 points)".to_string());
    lut_row.set_subtitle(&glib::markup_escape_text(&describe(current())));
    let choose = gtk4::Button::with_label("Choose…");
    choose.set_valign(gtk4::Align::Center);
    let clear = gtk4::Button::from_icon_name("edit-clear-symbolic");
    clear.set_valign(gtk4::Align::Center);
    clear.set_tooltip_text(Some("No look-up table"));
    lut_row.add_suffix(&choose);
    lut_row.add_suffix(&clear);
    group.add(&lut_row);
    // Names the file in config.ini and tells the layer to read it again.
    let set_lut = {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        let lut_row = lut_row.clone();
        std::rc::Rc::new(move |path: Option<String>| {
            let mut cfg = neural_forge_supervisor::Config::load();
            match &path {
                Some(p) => cfg.settings.insert(neural_forge_protocol::cube::CONFIG_KEY.to_string(), p.clone()),
                None => cfg.settings.remove(neural_forge_protocol::cube::CONFIG_KEY),
            };
            if let Err(e) = cfg.save() {
                toasts.add_toast(toast(&format!("Saving config.ini failed: {e}")));
                return;
            }
            shm.header().edit_lut_seq.fetch_add(1, Ordering::Relaxed);
            shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
            lut_row.set_subtitle(&glib::markup_escape_text(&describe(path)));
        })
    };
    {
        let set_lut = std::rc::Rc::clone(&set_lut);
        let toasts = toasts.clone();
        choose.connect_clicked(move |button| {
            let parent = button.root().and_downcast::<gtk4::Window>();
            let filter = gtk4::FileFilter::new();
            filter.set_name(Some("3D look-up tables (.cube)"));
            filter.add_suffix("cube");
            let filters = gio::ListStore::new::<gtk4::FileFilter>();
            filters.append(&filter);
            let dialog = gtk4::FileDialog::builder().title("Select a .cube look-up table").filters(&filters).build();
            let set_lut = std::rc::Rc::clone(&set_lut);
            let toasts = toasts.clone();
            dialog.open(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
                let Ok(file) = result else { return };
                let Some(path) = file.path() else {
                    toasts.add_toast(toast("That file has no local path -- choose a file on this computer"));
                    return;
                };
                let path = path.to_string_lossy().into_owned();
                match neural_forge_protocol::cube::load(&path) {
                    Ok(cube) => {
                        set_lut(Some(path));
                        toasts.add_toast(toast(&format!("Look-up table set: {} points a side. Raise its strength to use it", cube.size)));
                    }
                    Err(e) => toasts.add_toast(toast(&format!("Not a usable look-up table: {e}"))),
                }
            });
        });
    }
    clear.connect_clicked(move |_| set_lut(None));
    let (v, set) = bind_float(shm, Some("edit_lut_strength"), |h| &h.edit_lut_strength_bits);
    group.add(&spin_row(
        "Look-up table strength",
        "0 is off. Applied where the model changed the picture, on its last pass",
        v,
        "edit_lut_strength",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("edit_lut_adapt"), |h| &h.edit_lut_adapt_bits);
    group.add(&spin_row(
        "Look-up table full at",
        "How large a change gets the table's full strength (0-1 of the picture's range); smaller changes get less, an unchanged pixel none",
        v,
        "edit_lut_adapt",
        0.01,
        set,
    ));
    group
}

/// The placements' names in the menus (`enums::placement`, by value) and what each does.
const PLACEMENTS: [(&str, &str); 4] = [
    ("Auto", "Before the upscaler; after DLSS while Ray Reconstruction runs; on the finished frame where neither works"),
    ("Before the upscaler", "On DLSS's input at the render size, fastest (with Ray Reconstruction: on the finished frame)"),
    ("After DLSS", "On DLSS's output at the output size, before the game's HUD; where the game allows it, else as Auto"),
    ("Finished frame", "On the presented frame with the HUD; the HUD settings below keep the edit off it"),
];

/// [`PLACEMENTS`]' names alone (the per-pass dialog's menu).
const PLACEMENT_NAMES: [&str; 4] = [PLACEMENTS[0].0, PLACEMENTS[1].0, PLACEMENTS[2].0, PLACEMENTS[3].0];

/// The Placement menu: where the model runs in the frame. Its subtitle says what the choice does.
fn placement_row(selected: u32, setter: impl Fn(u32) + 'static) -> adw::ComboRow {
    let names: Vec<&str> = PLACEMENTS.iter().map(|&(n, _)| n).collect();
    let row = combo_row("Placement", &names, selected, setter);
    let describe = |row: &adw::ComboRow| row.set_subtitle(PLACEMENTS.get(row.selected() as usize).map_or("", |&(_, d)| d));
    describe(&row);
    row.connect_selected_notify(describe);
    row
}

/// The sharpening of the finished frame (`crate::sharpen` in the layer): FidelityFX CAS and Clarity, both off at 0,
/// applied to every presented frame after everything else, the HUD regions kept out of it.
fn build_sharpen_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Sharpening");
    group.set_description(Some("On the finished frame, after the model, in every game; the HUD regions above stay untouched. 0 is off"));
    let (v, set) = bind_float(shm, Some("sharpen_cas"), |h| &h.sharpen_cas_bits);
    group.add(&spin_row(
        "Adaptive sharpening (CAS)",
        "AMD FidelityFX CAS: sharpens fine detail, less where it is already sharp (0 off, about 0.7 CAS's own default, 1 strongest)",
        v,
        "sharpen_cas",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("clarity_strength"), |h| &h.clarity_strength_bits);
    group.add(&spin_row(
        "Clarity",
        "Local contrast of brightness: lifts mid-size detail and texture without changing colours (0 off)",
        v,
        "clarity_strength",
        0.05,
        set,
    ));
    let (v, set) = bind_float(shm, Some("clarity_radius"), |h| &h.clarity_radius_bits);
    group.add(&spin_row(
        "Clarity radius",
        "The size of the detail Clarity lifts, in pixels at 1440p (scaled with the resolution)",
        v,
        "clarity_radius",
        1.0,
        set,
    ));
    let (v, set) = bind_float(shm, Some("clarity_midtones"), |h| &h.clarity_midtones_bits);
    group.add(&spin_row("Clarity midtones", "Keeps Clarity off the darkest and brightest parts (0 evenly, 1 midtones only)", v, "clarity_midtones", 0.05, set));
    let (v, set) = bind_float(shm, Some("clarity_edges"), |h| &h.clarity_edges_bits);
    group.add(&spin_row(
        "Clarity edge protection",
        "Stops Clarity at strong edges, against bright or dark halos around them (0 off, 1 most)",
        v,
        "clarity_edges",
        0.05,
        set,
    ));
    group
}

/// The HUD region rows' left and top, and width and height, in percent: no value they take leaves a region
/// empty once the header clips it to the frame (`HudRegion::clamped`, which drops an empty one), so an edit
/// can clip a region but never remove it.
const HUD_POSITION_RANGE: (f64, f64) = (0.0, 99.5);
const HUD_SIZE_RANGE: (f64, f64) = (0.5, 100.0);

/// Applies `change` to HUD region `k` of the header's list and writes the list (`write`). Returns what the
/// header holds then, and whether that differs from what was written: the header clips a region to the frame
/// and drops an empty one, and then the rows (their values, and with a region dropped their indices) no longer
/// match it and must be rebuilt from it.
fn edit_hud_region(
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

/// The HUD on the finished frame (the Placement "Finished frame", or where the model runs there anyway): the
/// detection and the user's regions, a list of rectangles in percent of the frame.
fn build_hud_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>) -> adw::PreferencesGroup {
    use neural_forge_protocol::{enums::hud_mode, HudRegion, MAX_HUD_REGIONS};
    use std::rc::Rc;
    let group = adw::PreferencesGroup::new();
    group.set_title("HUD on the finished frame");
    group.set_description(Some(
        "Only where the model runs on the presented frame, HUD included (before the upscaler and after DLSS the HUD is not drawn yet). The regions also keep the sharpening off the HUD",
    ));
    let (v, set) = bind_bool(shm, Some("hud_detect"), |h| &h.hud_detect);
    group.add(&switch_row("Find the HUD", "Keep the edit off what stays still and sharp while the picture moves (text, icons, the minimap)", v, set));
    let list = gtk4::ListBox::new();
    list.set_selection_mode(gtk4::SelectionMode::None);
    list.add_css_class("boxed-list");
    let add = gtk4::Button::with_label("Add region");
    add.set_halign(gtk4::Align::Start);
    add.set_margin_top(6);
    let holder = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
    holder.append(&list);
    holder.append(&add);
    let regions_row = adw::ActionRow::new();
    regions_row.set_title("HUD regions");
    regions_row.set_subtitle("Rectangles in percent of the frame (left, top, width, height): Keep has no edit inside, Soften scales it by the strength");
    group.add(&regions_row);
    group.add(&holder);
    // Writes the regions to the header as one unit and saves them (`set_hud_regions` in config.ini).
    let write = {
        let shm = std::sync::Arc::clone(shm);
        Rc::new(move |regions: &[HudRegion]| {
            shm.header().set_hud_regions(regions);
            crate::shm::persist_settings_later(&shm, &[neural_forge_protocol::persist::HUD_REGIONS]);
        })
    };
    // The list's rows, rebuilt from the header (after an add or a removal, and when something else changed it).
    type Rebuild = Rc<dyn Fn()>;
    let rebuild: Rc<std::cell::RefCell<Option<Rebuild>>> = Rc::new(std::cell::RefCell::new(None));
    let shown: Rc<std::cell::RefCell<Vec<HudRegion>>> = Rc::new(std::cell::RefCell::new(Vec::new()));
    {
        let (shm, list, write, rebuild_ref, shown, add) =
            (std::sync::Arc::clone(shm), list.clone(), Rc::clone(&write), Rc::clone(&rebuild), Rc::clone(&shown), add.clone());
        let f: Rebuild = Rc::new(move || {
            while let Some(child) = list.first_child() {
                list.remove(&child);
            }
            let regions = shm.header().hud_settings().regions;
            *shown.borrow_mut() = regions.clone();
            add.set_sensitive(regions.len() < MAX_HUD_REGIONS);
            for (k, r) in regions.iter().enumerate() {
                let row = adw::ActionRow::new();
                row.set_title(&format!("Region {}", k + 1));
                let edit = {
                    let (shm, write, shown, rebuild) = (std::sync::Arc::clone(&shm), Rc::clone(&write), Rc::clone(&shown), Rc::clone(&rebuild_ref));
                    Rc::new(move |change: &dyn Fn(&mut HudRegion)| {
                        if refreshing() {
                            return;
                        }
                        let (held, stale) = edit_hud_region(shm.header(), k, change, &*write);
                        *shown.borrow_mut() = held;
                        if stale {
                            // From an idle callback: the widget being edited is inside the list.
                            let rebuild = Rc::clone(&rebuild);
                            glib::idle_add_local_once(move || {
                                if let Some(f) = rebuild.borrow().clone() {
                                    f();
                                }
                            });
                        }
                    })
                };
                let spin = |value: f32, tip: &str, (min, max): (f64, f64), set: fn(&mut HudRegion, f32)| {
                    let b = gtk4::SpinButton::with_range(min, max, 0.5);
                    b.set_value(f64::from(value) * 100.0);
                    // As `keep_spin_precision` does for the spin rows.
                    b.set_digits(spin_digits(0.5, b.value()));
                    b.connect_input(|b| kept_spin_value(&b.text(), b.value(), b.digits()).map(Ok));
                    b.set_valign(gtk4::Align::Center);
                    b.set_tooltip_text(Some(tip));
                    let edit = Rc::clone(&edit);
                    b.connect_value_changed(move |b| {
                        b.set_digits(spin_digits(0.5, b.value()));
                        let v = (b.value() / 100.0) as f32;
                        edit(&move |r: &mut HudRegion| set(r, v));
                    });
                    row.add_suffix(&b);
                };
                spin(r.x, "Left, percent of the width", HUD_POSITION_RANGE, |r, v| r.x = v);
                spin(r.y, "Top, percent of the height", HUD_POSITION_RANGE, |r, v| r.y = v);
                spin(r.w, "Width, percent", HUD_SIZE_RANGE, |r, v| r.w = v);
                spin(r.h, "Height, percent", HUD_SIZE_RANGE, |r, v| r.h = v);
                let mode = gtk4::DropDown::from_strings(&["Keep", "Soften"]);
                mode.set_selected(if r.mode == hud_mode::SOFTEN { 1 } else { 0 });
                mode.set_valign(gtk4::Align::Center);
                {
                    let edit = Rc::clone(&edit);
                    mode.connect_selected_notify(move |m| {
                        let soften = m.selected() == 1;
                        edit(&move |r: &mut HudRegion| r.mode = if soften { hud_mode::SOFTEN } else { hud_mode::KEEP });
                    });
                }
                row.add_suffix(&mode);
                spin(r.strength, "Soften: how much of the edit stays, percent", (0.0, 100.0), |r, v| r.strength = v);
                let remove = gtk4::Button::from_icon_name("list-remove-symbolic");
                remove.set_valign(gtk4::Align::Center);
                remove.set_tooltip_text(Some("Remove this region"));
                {
                    let (shm, write, rebuild) = (std::sync::Arc::clone(&shm), Rc::clone(&write), Rc::clone(&rebuild_ref));
                    remove.connect_clicked(move |_| {
                        let mut regions = shm.header().hud_settings().regions;
                        if k < regions.len() {
                            regions.remove(k);
                        }
                        write(&regions);
                        if let Some(f) = rebuild.borrow().clone() {
                            f();
                        }
                    });
                }
                row.add_suffix(&remove);
                list.append(&row);
            }
        });
        *rebuild.borrow_mut() = Some(f);
    }
    {
        let (shm, write, rebuild) = (std::sync::Arc::clone(shm), Rc::clone(&write), Rc::clone(&rebuild));
        add.connect_clicked(move |_| {
            let mut regions = shm.header().hud_settings().regions;
            // A strip along the bottom edge, where many games put their HUD; edited from there.
            regions.push(HudRegion { x: 0.0, y: 0.88, w: 0.3, h: 0.12, mode: hud_mode::KEEP, strength: HudRegion::DEFAULT_STRENGTH });
            write(&regions);
            if let Some(f) = rebuild.borrow().clone() {
                f();
            }
        });
    }
    if let Some(f) = rebuild.borrow().clone() {
        f();
    }
    // Follows changes made elsewhere (shmctl, a profile, Reset); not its own edits (`shown` has those).
    let shm = std::sync::Arc::clone(shm);
    register_refresher(move || {
        if shm.header().hud_settings().regions != *shown.borrow() {
            if let Some(f) = rebuild.borrow().clone() {
                f();
            }
        }
    });
    group
}

/// One field of the per-pass dialog.
struct PassField {
    title: &'static str,
    /// The setting's name after `pass<N>_`.
    field: &'static str,
    bit: u32,
    kind: PassFieldKind,
    /// The pass's field (pass 0 is pass 1).
    get: fn(&neural_forge_protocol::ShmHeader, usize) -> &std::sync::atomic::AtomicU32,
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

/// Per-pass overrides: each pass can replace any of the Model tab's values with its own; a pass that
/// overrides nothing follows the Model tab. Saved as `set_pass<N>_*` like any other setting, and
/// written as one unit (`write_settings`) so the layer never reads half of a change.
fn open_pass_dialog(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, parent: &gtk4::Widget) {
    use neural_forge_protocol::enums::pass_override as po;
    use std::cell::Cell;
    use std::rc::Rc;
    const FIELDS: &[PassField] = &[
        PassField {
            title: "Intensity",
            field: "intensity",
            bit: po::INTENSITY,
            kind: PassFieldKind::Float { step: 0.05 },
            get: |h, k| &h.pass[k].intensity_bits,
        },
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
        PassField {
            title: "Placement",
            field: "placement",
            bit: po::PLACEMENT,
            kind: PassFieldKind::Choice(&PLACEMENT_NAMES),
            get: |h, k| &h.pass_placement[k],
        },
    ];
    let max = neural_forge_protocol::MAX_PASSES;

    let dialog = adw::PreferencesDialog::new();
    dialog.set_title("Per-pass settings");
    let page = adw::PreferencesPage::new();
    let pick = adw::PreferencesGroup::new();
    pick.set_description(Some(
        "Each pass of the model can override any Model-tab value with its own. A pass that overrides nothing \
         follows the Model tab. Only the passes the Passes setting runs are used. Passes run in order: a pass \
         placed earlier in the frame than the pass before it runs where that one does.",
    ));
    let pass_adj = gtk4::Adjustment::new(1.0, 1.0, max as f64, 1.0, 1.0, 0.0);
    let pass_row = adw::SpinRow::new(Some(&pass_adj), 1.0, 0);
    pass_row.set_title("Pass");
    pick.add(&pass_row);
    page.add(&pick);
    let fields_group = adw::PreferencesGroup::new();
    page.add(&fields_group);

    let loading = Rc::new(Cell::new(false));
    let current_pass = move |adj: &gtk4::Adjustment| (adj.value() as usize).saturating_sub(1).min(max - 1);
    // Writes settings of the pass shown, as one unit, and saves them.
    let write = {
        let shm = std::sync::Arc::clone(shm);
        move |settings: &[(&'static str, u32)]| {
            if shm.header().write_settings(settings) {
                let names: Vec<&'static str> = settings.iter().map(|&(n, _)| n).collect();
                crate::shm::persist_settings_later(&shm, &names);
            }
        }
    };
    let write = Rc::new(write);
    let mut reloads: Vec<Box<dyn Fn(usize)>> = Vec::new();

    for field in FIELDS {
        let expander = adw::ExpanderRow::new();
        expander.set_title(field.title);
        expander.set_subtitle("Override for this pass");
        expander.set_show_enable_switch(true);
        let set_value = {
            let pass_adj = pass_adj.clone();
            let loading = Rc::clone(&loading);
            let write = Rc::clone(&write);
            let name = field.field;
            Rc::new(move |bits: u32| {
                if loading.get() {
                    return;
                }
                if let Some(setting) = neural_forge_protocol::pass_setting(current_pass(&pass_adj), name) {
                    write(&[(setting, bits)]);
                }
            })
        };
        let set_widget: Box<dyn Fn(u32)> = match &field.kind {
            PassFieldKind::Float { step } => {
                let (lower, upper) = neural_forge_protocol::setting_bounds(field.field).unwrap_or_else(|| panic!("{} has no bounds", field.field));
                let adj = gtk4::Adjustment::new(0.0, f64::from(lower), f64::from(upper), *step, step * 10.0, 0.0);
                let row = adw::SpinRow::new(Some(&adj), *step, 0);
                keep_spin_precision(&row, &adj, *step);
                row.set_title("Value");
                let w = Rc::clone(&set_value);
                adj.connect_value_changed(move |a| w((a.value() as f32).to_bits()));
                expander.add_row(&row);
                Box::new(move |bits| adj.set_value(f64::from(f32::from_bits(bits))))
            }
            PassFieldKind::Choice(options) => {
                let row = adw::ComboRow::new();
                row.set_title("Value");
                row.set_model(Some(&gtk4::StringList::new(options)));
                let w = Rc::clone(&set_value);
                row.connect_selected_notify(move |r| w(r.selected()));
                expander.add_row(&row);
                let last = options.len() as u32 - 1;
                Box::new(move |bits| row.set_selected(bits.min(last)))
            }
            PassFieldKind::Toggle => {
                let row = adw::SwitchRow::new();
                row.set_title("On");
                let w = Rc::clone(&set_value);
                row.connect_active_notify(move |r| w(u32::from(r.is_active())));
                expander.add_row(&row);
                Box::new(move |bits| row.set_active(bits != 0))
            }
            PassFieldKind::Scale => {
                use neural_forge_protocol::enums::model_scale::{preset_of, PRESETS};
                let mut names: Vec<&str> = PRESETS.iter().map(|&(n, _)| n).collect();
                names.push("Custom");
                let custom = PRESETS.len() as u32;
                let row = adw::ComboRow::new();
                row.set_title("Value");
                row.set_subtitle("Of DLSS's input size; this pass works on the previous pass's full-size picture");
                row.set_model(Some(&gtk4::StringList::new(&names)));
                let w = Rc::clone(&set_value);
                row.connect_selected_notify(move |r| {
                    if let Some(&(_, scale)) = PRESETS.get(r.selected() as usize) {
                        w(scale.to_bits());
                    }
                });
                expander.add_row(&row);
                Box::new(move |bits| row.set_selected(preset_of(f32::from_bits(bits)).map_or(custom, |i| i as u32)))
            }
        };
        {
            let shm = std::sync::Arc::clone(shm);
            let pass_adj = pass_adj.clone();
            let loading = Rc::clone(&loading);
            let write = Rc::clone(&write);
            let bit = field.bit;
            expander.connect_enable_expansion_notify(move |e| {
                if loading.get() {
                    return;
                }
                let pass = current_pass(&pass_adj);
                let mask = shm.header().pass[pass].overrides.load(Ordering::Relaxed);
                let mask = if e.enables_expansion() { mask | bit } else { mask & !bit };
                if let Some(setting) = neural_forge_protocol::pass_setting(pass, "overrides") {
                    write(&[(setting, mask)]);
                }
            });
        }
        fields_group.add(&expander);
        let shm = std::sync::Arc::clone(shm);
        let get = field.get;
        let bit = field.bit;
        reloads.push(Box::new(move |pass| {
            let h = shm.header();
            expander.set_enable_expansion(h.pass[pass].overrides.load(Ordering::Relaxed) & bit != 0);
            set_widget(get(h, pass).load(Ordering::Relaxed));
        }));
    }

    let reload = {
        let loading = Rc::clone(&loading);
        let reloads = Rc::new(reloads);
        move |pass: usize| {
            loading.set(true);
            reloads.iter().for_each(|r| r(pass));
            loading.set(false);
        }
    };
    reload(0);
    pass_adj.connect_value_changed(move |a| reload(current_pass(a)));

    dialog.add(&page);
    dialog.present(Some(parent));
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

/// The Setup tab's hand-set launch option: pick a game and its executable is filled in (the one
/// the games list would write), or type one for a game Steam does not list. Filled from the Steam
/// scan by the returned function.
fn build_launch_option_group() -> (adw::PreferencesGroup, impl Fn(&SteamScan)) {
    let group = adw::PreferencesGroup::new();
    group.set_title("Steam launch option");
    group.set_description(Some("To set it by hand: paste this into the game's Properties -> Launch Options in Steam"));

    let game_row = adw::ComboRow::new();
    game_row.set_title("Game");
    game_row.set_model(Some(&gtk4::StringList::new(&["Another game"])));
    group.add(&game_row);

    let exe_row = adw::EntryRow::new();
    exe_row.set_title("Target executable (the game's .exe; several separated by commas)");
    group.add(&exe_row);

    let preview_row = adw::ActionRow::new();
    preview_row.set_title("Launch option");
    preview_row.add_css_class("property");
    preview_row.set_use_markup(false);
    let copy_button = gtk4::Button::with_label("Copy");
    copy_button.set_valign(gtk4::Align::Center);
    preview_row.add_suffix(&copy_button);
    group.add(&preview_row);

    let exe_row_for_build = exe_row.clone();
    let build_option = std::rc::Rc::new(move || neural_forge_supervisor::launch_options::launch_option(&exe_row_for_build.text()));
    preview_row.set_subtitle(&build_option());
    {
        let preview_row = preview_row.clone();
        let build_option = std::rc::Rc::clone(&build_option);
        exe_row.connect_changed(move |_| preview_row.set_subtitle(&build_option()));
    }
    copy_button.connect_clicked(move |button| {
        button.display().clipboard().set_text(&build_option());
    });

    // The games the combo lists (after "Another game"), with their targets.
    let targets: std::rc::Rc<std::cell::RefCell<Vec<String>>> = std::rc::Rc::default();
    {
        let (targets, exe_row) = (targets.clone(), exe_row.clone());
        game_row.connect_selected_notify(move |row| {
            if let Some(target) = (row.selected() as usize).checked_sub(1).and_then(|i| targets.borrow().get(i).cloned()) {
                exe_row.set_text(&target);
            }
        });
    }
    let fill = move |scan: &SteamScan| {
        let listed: Vec<&ScannedGame> = scan.games.iter().filter(|g| g.game.unsupported().is_none() && !g.target.is_empty()).collect();
        *targets.borrow_mut() = listed.iter().map(|g| g.target.clone()).collect();
        let mut names = vec!["Another game"];
        names.extend(listed.iter().map(|g| g.game.name.as_str()));
        game_row.set_model(Some(&gtk4::StringList::new(&names)));
        // A game already switched on first, else the first game: the string is complete by default.
        let pick = listed.iter().position(|g| g.enabled).or((!listed.is_empty()).then_some(0));
        game_row.set_selected(pick.map_or(0, |i| i as u32 + 1));
        if let Some(i) = pick {
            exe_row.set_text(&listed[i].target);
        }
    };
    (group, fill)
}

/// The model's row text: present with each extracted network and the build it came from (and
/// whether that is verified), or missing.
fn model_status() -> (String, bool) {
    let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    match neural_forge_supervisor::model::installed_text(&root) {
        Some(networks) => (format!("present: {networks}"), true),
        None => ("missing: extract it from nvngx_dlssnr.dll below".to_string(), false),
    }
}

/// The Network row's choices, in [`Network::ALL`]'s order (the Network setting's values).
const NETWORK_CHOICES: [&str; 2] = ["Full network", "Small network"];

/// The Network row's subtitle for `network`: what it gives, and what runs instead when it is not
/// extracted (the layer runs the other network then).
fn network_subtitle(network: Network) -> String {
    let what = match network {
        Network::Full => "The default: stronger shading and depth",
        Network::Small => "Faster, with a subtler effect",
    };
    let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    let installed = neural_forge_supervisor::model::installed_all(&root);
    match installed.iter().find(|i| i.network != network) {
        Some(other) if !installed.iter().any(|i| i.network == network) => {
            format!("{what}. Not extracted from your DLL, so the {} runs", other.network)
        }
        _ => format!("{what}. A running game switches at once"),
    }
}

/// The model group's rows an extraction reads and updates.
#[derive(Clone)]
struct ModelRows {
    toasts: adw::ToastOverlay,
    status_row: adw::ActionRow,
    icon: gtk4::Image,
    button: gtk4::Button,
}

/// Extracts every network the DLL at `path` carries into the model directory
/// ([`neural_forge_supervisor::model::extract_all`]), says how it went in a toast and brings the
/// status rows up to date. The button is insensitive meanwhile.
async fn extract_model(path: std::path::PathBuf, rows: ModelRows) {
    rows.button.set_sensitive(false);
    let root = std::path::PathBuf::from(neural_forge_supervisor::model::model_dir());
    let result = gio::spawn_blocking(move || neural_forge_supervisor::model::extract_all(&path, &root)).await;
    let toasts = &rows.toasts;
    match result {
        Ok(Ok(all)) => {
            let networks: Vec<String> = all
                .networks
                .iter()
                .map(|done| {
                    format!("the {} of build {}{}", done.network, done.build, if done.verified { "" } else { " (not verified against NVIDIA's runtime)" })
                })
                .collect();
            for (network, why) in &all.refused {
                eprintln!("neural-forge: the DLL's {network} was not extracted: {why}");
            }
            let refused = if all.refused.is_empty() { "" } else { " -- a network of the DLL was refused, see the log" };
            toasts.add_toast(toast(&format!("Extracted {}{refused} -- restart the game to use it", networks.join(" and "))))
        }
        // The report is too long for a toast: its summary line there, all of it in the log.
        Ok(Err(neural_forge_supervisor::model::ModelError::Shape(report))) => {
            eprintln!("neural-forge: extraction refused:\n{report}");
            let summary = report.lines().next().unwrap_or_default();
            toasts.add_toast(toast(&format!("Extraction refused, a different network: {summary}")));
        }
        Ok(Err(e)) => toasts.add_toast(toast(&format!("Extraction failed: {e}"))),
        Err(_) => toasts.add_toast(toast("Extraction failed: the worker panicked")),
    }
    let (text, present) = model_status();
    rows.status_row.set_subtitle(&text);
    rows.icon.set_icon_name(Some(if present { "object-select-symbolic" } else { "dialog-warning-symbolic" }));
    rows.button.set_sensitive(true);
    model_changed();
}

/// The native backend's one setup step: point at NVIDIA's DLL once and extract the model from it;
/// and which of its networks runs (the Network setting, live).
fn build_model_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Neural rendering model");
    group.set_description(Some("Taken from your own nvngx_dlssnr.dll -- this project doesn't and can't ship it"));
    let (text, present) = model_status();
    let status_row = adw::ActionRow::new();
    status_row.set_title("Model");
    status_row.set_subtitle(&text);
    let icon = gtk4::Image::from_icon_name(if present { "object-select-symbolic" } else { "dialog-warning-symbolic" });
    status_row.add_prefix(&icon);
    group.add(&status_row);

    // Which network runs: a persisted setting, so a running game switches over at its next frame.
    let (network, set_network) = bind_u32(shm, Some("network"), |h| &h.network);
    let network_row = combo_row("Network", &NETWORK_CHOICES, network, set_network);
    let subtitle = |row: &adw::ComboRow| row.set_subtitle(&network_subtitle(Network::from_setting(row.selected())));
    subtitle(&network_row);
    network_row.connect_selected_notify(subtitle);
    {
        let network_row = network_row.clone();
        on_model_changed(move || subtitle(&network_row));
    }
    group.add(&network_row);

    let extract_row = adw::ActionRow::new();
    extract_row.set_title("Extract from DLL");
    extract_row.set_subtitle("Choose nvngx_dlssnr.dll; every network it has is written to Neural Forge's data folder");
    let button = gtk4::Button::with_label("Extract…");
    button.set_valign(gtk4::Align::Center);
    extract_row.add_suffix(&button);
    extract_row.set_activatable_widget(Some(&button));
    group.add(&extract_row);

    let rows = ModelRows { toasts: toasts.clone(), status_row, icon, button: button.clone() };
    button.connect_clicked(move |button| {
        let rows = rows.clone();
        let parent = button.root().and_downcast::<gtk4::Window>();
        let dialog = gtk4::FileDialog::builder().title("Select nvngx_dlssnr.dll").build();
        dialog.open(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
            let Ok(file) = result else { return };
            let Some(path) = file.path() else {
                rows.toasts.add_toast(toast("That file has no local path -- choose a file on this computer"));
                return;
            };
            glib::spawn_future_local(extract_model(path, rows));
        });
    });
    group
}

/// The games group's description: what the switches do, or why they can't be used right now.
fn games_note(roots: &[neural_forge_supervisor::steam::Root], games: usize, steam_running: bool) -> String {
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

/// A toast with a Copy button for `manual`, the launch options to set by hand.
fn manual_toast(message: &str, manual: String) -> adw::Toast {
    let toast = toast(message);
    toast.set_timeout(0);
    toast.set_button_label(Some("Copy launch options"));
    toast.connect_button_clicked(move |_| {
        if let Some(display) = gtk4::gdk::Display::default() {
            display.clipboard().set_text(&manual);
        }
    });
    toast
}

/// One installed game as the Setup tab shows it.
struct ScannedGame {
    game: neural_forge_supervisor::steam::Game,
    enabled: bool,
    /// What switching it on writes as `NEURAL_FORGE_TARGET_EXE` (`steam::target_for`); empty when
    /// its executable was not found.
    target: String,
    /// Whether `target` was chosen by hand (`config.ini`), not found.
    chosen: bool,
    /// The executables in its folder, best first (`TargetFinder::candidates`), for the target editor.
    candidates: Vec<neural_forge_supervisor::steam::Candidate>,
    /// The anti-cheat its folder ships (`Game::anti_cheat`): switching it on asks first.
    anti_cheat: Vec<&'static str>,
}

/// A game row's subtitle: its appid, library and target (or why it has none), and its anti-cheat.
fn game_subtitle(game: &neural_forge_supervisor::steam::Game, target: &str, chosen: bool, anti_cheat: &[&str]) -> String {
    let detail = match (game.unsupported(), target.is_empty()) {
        (Some(why), _) => format!("cannot be switched on: {why}"),
        (None, true) => "its executable was not found: choose it with the edit button".to_string(),
        (None, false) => format!("runs {}{}", target.replace(',', ", "), if chosen { " (chosen)" } else { "" }),
    };
    let anti_cheat = match anti_cheat {
        [] => String::new(),
        found => format!(" · anti-cheat: {}", neural_forge_supervisor::anti_cheat::list(found)),
    };
    format!("{} · {} · {detail}{anti_cheat}", game.appid, game.library.display())
}

/// Asks before switching on a game whose folder ships `anti_cheat`: "Switch On Anyway" runs
/// `apply`, anything else turns the switch back off.
fn confirm_anti_cheat(switch: &gtk4::Switch, game: &str, anti_cheat: &[&str], reverting: &std::rc::Rc<std::cell::Cell<bool>>, apply: impl FnOnce() + 'static) {
    let dialog = adw::AlertDialog::builder()
        .heading(format!("{game} ships anti-cheat"))
        .body(neural_forge_supervisor::anti_cheat::warning(anti_cheat))
        .default_response("cancel")
        .close_response("cancel")
        .build();
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("on", "Switch On Anyway");
    dialog.set_response_appearance("on", adw::ResponseAppearance::Destructive);
    let (switch_weak, reverting) = (switch.downgrade(), reverting.clone());
    dialog.choose(switch.root().as_ref(), None::<&gio::Cancellable>, move |response| {
        if response == "on" {
            apply();
        } else if let Some(switch) = switch_weak.upgrade() {
            reverting.set(true);
            switch.set_active(false);
            reverting.set(false);
        }
    });
}

/// The per-game target editor: pick one of the executables found in the game's folder (best
/// first) or type names (comma-separated). Saving remembers it for the game (`config.ini`, as
/// `neural-forge-cli enable --target-exe` does); "Use the found ones" forgets it. A game that is on
/// gets its launch options rewritten with the new target at once (Steam closed).
#[allow(clippy::too_many_arguments)]
fn target_editor(
    button: &gtk4::Button,
    row: &adw::ActionRow,
    switch: &gtk4::Switch,
    game: &neural_forge_supervisor::steam::Game,
    target: &std::rc::Rc<std::cell::RefCell<String>>,
    candidates: &[neural_forge_supervisor::steam::Candidate],
    anti_cheat: &[&'static str],
    toasts: &adw::ToastOverlay,
    running: &std::rc::Rc<std::cell::Cell<bool>>,
) {
    use neural_forge_supervisor::steam;
    let body = gtk4::Box::new(gtk4::Orientation::Vertical, 6);
    let entry = gtk4::Entry::new();
    entry.set_text(&target.borrow());
    entry.set_placeholder_text(Some("Game.exe"));
    if !candidates.is_empty() {
        let label = gtk4::Label::new(Some("Found in the game's folder, best first:"));
        label.set_xalign(0.0);
        body.append(&label);
    }
    for c in candidates.iter().take(10) {
        let pick = gtk4::Button::with_label(&format!("{}  ({}, {} MB){}", c.name, c.path, c.bytes >> 20, if c.chosen { "  found" } else { "" }));
        pick.add_css_class("flat");
        if let Some(label) = pick.child().and_downcast::<gtk4::Label>() {
            label.set_xalign(0.0);
            label.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
        }
        let (entry, name) = (entry.clone(), c.name.clone());
        pick.connect_clicked(move |_| entry.set_text(&name));
        body.append(&pick);
    }
    body.append(&entry);
    let dialog = adw::AlertDialog::builder()
        .heading(format!("Target for {}", game.name))
        .body("Only these executables can use Neural Forge in this game (several separated by commas)")
        .default_response("save")
        .close_response("cancel")
        .build();
    dialog.set_extra_child(Some(&body));
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("auto", "Use the found ones");
    dialog.add_response("save", "Save");
    dialog.set_response_appearance("save", adw::ResponseAppearance::Suggested);
    let (row, switch, game, target, toasts, running) = (row.clone(), switch.clone(), game.clone(), target.clone(), toasts.clone(), running.clone());
    let anti_cheat = anti_cheat.to_vec();
    dialog.choose(button.root().as_ref(), None::<&gio::Cancellable>, move |response| {
        let typed = entry.text().split(',').map(str::trim).filter(|n| !n.is_empty()).collect::<Vec<_>>().join(",");
        let mut cfg = neural_forge_supervisor::Config::load();
        let key = steam::chosen_target_key(&game.appid);
        match response.as_str() {
            "save" if !typed.is_empty() => {
                cfg.settings.insert(key, typed);
            }
            "save" => {
                toasts.add_toast(toast("Not saved: name at least one executable"));
                return;
            }
            "auto" => {
                cfg.settings.remove(&key);
            }
            _ => return,
        }
        if let Err(e) = cfg.save() {
            toasts.add_toast(toast(&format!("Saving config.ini failed: {e}")));
            return;
        }
        let (row, switch, game, target, toasts, running) = (row.clone(), switch.clone(), game.clone(), target.clone(), toasts.clone(), running.clone());
        let anti_cheat = anti_cheat.clone();
        glib::spawn_future_local(async move {
            let worker_game = game.clone();
            let found = gio::spawn_blocking(move || {
                let cfg = neural_forge_supervisor::Config::load();
                let new = steam::target_for(&cfg, &worker_game);
                let chosen = steam::chosen_target(&cfg, &worker_game.appid).is_some();
                (new, chosen)
            })
            .await;
            let Ok((new, chosen)) = found else { return };
            *target.borrow_mut() = new.clone();
            row.set_subtitle(&game_subtitle(&game, &new, chosen, &anti_cheat));
            // A game whose executable was just named can be switched on now, without a refresh.
            switch.set_sensitive(switch_usable(&game, &new, switch.is_active(), running.get()));
            if !switch.is_active() || new.is_empty() {
                toasts.add_toast(toast(&format!(
                    "{}'s target is {}; it is used when the game is switched on",
                    game.name,
                    if new.is_empty() { "not found" } else { &new }
                )));
                return;
            }
            let (root, appid, action) = (game.root.path.clone(), game.appid.clone(), steam::Action::Enable { target_exe: new.clone() });
            match gio::spawn_blocking(move || steam::apply_now(&root, &appid, &action)).await {
                Ok(steam::Outcome::Changed(_) | steam::Outcome::Unchanged) => toasts.add_toast(toast(&format!("{} now targets {new}", game.name))),
                Ok(steam::Outcome::SteamRunning { manual }) => toasts
                    .add_toast(manual_toast("Saved, but Steam is running: close it and switch the game off and on, or set the launch options by hand", manual)),
                Ok(steam::Outcome::Failed { reason, manual }) => {
                    toasts.add_toast(manual_toast(&format!("Saved, but the launch options could not be changed safely: {reason}"), manual))
                }
                Err(_) => toasts.add_toast(toast("Saved, but changing the launch options stopped")),
            }
        });
    });
}

/// Steam as the Setup tab shows it, read on a worker thread ([`scan_steam`]): every account's
/// `localconfig.vdf`, Steam's app cache and the games' folders are too slow to read on the main loop.
struct SteamScan {
    roots: Vec<neural_forge_supervisor::steam::Root>,
    running: bool,
    games: Vec<ScannedGame>,
}

fn scan_steam() -> SteamScan {
    use neural_forge_supervisor::steam;
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

/// The installed Steam games, each with a switch that turns Neural Forge on or off in its launch
/// options (`neural_forge_supervisor::steam::apply`) for the game's own executable. While Steam runs
/// the switches are insensitive: it would overwrite the edit on exit. A game of the Flatpak Steam
/// cannot be switched on (its sandbox does not see the layer) and says so. Filled from the Steam
/// scan by the returned function; the edit itself also runs on a worker thread.
fn build_games_group(toasts: &adw::ToastOverlay, refresh: &gtk4::Button) -> (adw::PreferencesGroup, impl Fn(&SteamScan)) {
    use neural_forge_supervisor::steam;
    let group = adw::PreferencesGroup::new();
    group.set_title("Steam games");
    group.set_description(Some("Looking for games…"));
    group.set_header_suffix(Some(refresh));

    // Each game: its row and the switch in it. Only the switch is ever insensitive, so the target
    // editor opens for a game whose executable was not found, the one it is there for.
    let rows: std::rc::Rc<std::cell::RefCell<Vec<(adw::ActionRow, gtk4::Switch)>>> = std::rc::Rc::default();
    let fill = {
        let (group, rows, toasts) = (group.clone(), rows.clone(), toasts.clone());
        move |scan: &SteamScan| {
            for (row, _) in rows.borrow_mut().drain(..) {
                group.remove(&row);
            }
            group.set_description(Some(&games_note(&scan.roots, scan.games.len(), scan.running)));
            // Shared by the rows: a refused edit finds Steam running for all of them.
            let running = std::rc::Rc::new(std::cell::Cell::new(scan.running));
            for scanned in &scan.games {
                let game = scanned.game.clone();
                // Shared with the target editor, which changes it.
                let target = std::rc::Rc::new(std::cell::RefCell::new(scanned.target.clone()));
                let row = adw::ActionRow::new();
                row.set_use_markup(false);
                row.set_title(&game.name);
                row.set_subtitle(&game_subtitle(&game, &target.borrow(), scanned.chosen, &scanned.anti_cheat));
                let switch = gtk4::Switch::new();
                switch.set_valign(gtk4::Align::Center);
                switch.set_active(scanned.enabled);
                row.add_suffix(&switch);
                row.set_activatable_widget(Some(&switch));
                if game.unsupported().is_none() {
                    let edit = gtk4::Button::from_icon_name("document-edit-symbolic");
                    edit.set_tooltip_text(Some("Choose the executable Neural Forge targets for this game"));
                    edit.add_css_class("flat");
                    edit.set_valign(gtk4::Align::Center);
                    row.add_suffix(&edit);
                    let (row_weak, switch_weak, game, target, toasts, candidates, running) =
                        (row.downgrade(), switch.downgrade(), game.clone(), target.clone(), toasts.clone(), scanned.candidates.clone(), running.clone());
                    let anti_cheat = scanned.anti_cheat.clone();
                    edit.connect_clicked(move |button| {
                        let (Some(row), Some(switch)) = (row_weak.upgrade(), switch_weak.upgrade()) else { return };
                        target_editor(button, &row, &switch, &game, &target, &candidates, &anti_cheat, &toasts, &running);
                    });
                }
                // Switching off always works; switching on needs a usable game with a known executable.
                switch.set_sensitive(switch_usable(&game, &target.borrow(), scanned.enabled, scan.running));
                let reverting = std::rc::Rc::new(std::cell::Cell::new(false));
                let anti_cheat = scanned.anti_cheat.clone();
                // Weak: the list holds the rows, and each row's handler would otherwise hold the list.
                let (group_for_row, rows_for_row, toasts, running) = (group.downgrade(), std::rc::Rc::downgrade(&rows), toasts.clone(), running.clone());
                switch.connect_active_notify(move |row| {
                    if reverting.get() {
                        return;
                    }
                    let on = row.is_active();
                    let action = if on { steam::Action::Enable { target_exe: target.borrow().clone() } } else { steam::Action::Disable };
                    let (row_weak, game_for_apply, reverting_for_apply, toasts) = (row.downgrade(), game.clone(), reverting.clone(), toasts.clone());
                    let (group_for_row, rows_for_row, running, target) = (group_for_row.clone(), rows_for_row.clone(), running.clone(), target.clone());
                    let apply = move || {
                        let (row, game, reverting) = (row_weak, game_for_apply, reverting_for_apply);
                        if let Some(switch) = row.upgrade() {
                            switch.set_sensitive(false);
                        }
                        let proton_note: std::rc::Rc<std::cell::RefCell<Option<Result<String, String>>>> = std::rc::Rc::default();
                        glib::spawn_future_local(async move {
                            let (root, appid) = (game.root.path.clone(), game.appid.clone());
                            // The launch options, then the game's Proton (a Neural Forge copy when switched on, made now if
                            // needed; the earlier one back when off): DLSS's placements need it.
                            let outcome = gio::spawn_blocking(move || {
                                let outcome = steam::apply_now(&root, &appid, &action);
                                let proton = match (&outcome, on) {
                                    (steam::Outcome::Changed(_) | steam::Outcome::Unchanged, true) => {
                                        let running = steam::steam_running(std::path::Path::new("/proc"));
                                        let w = neural_forge_supervisor::proton_tool::installed_wrapper();
                                        Some(neural_forge_supervisor::compat::switch_on(&root, &steam::libraries(&root), &appid, &w, running).map(|d| {
                                            format!(
                                                "; it runs with {} now{}",
                                                d.tool,
                                                if d.made { " (a copy made for Neural Forge: restart Steam to see it)" } else { "" }
                                            )
                                        }))
                                    }
                                    (steam::Outcome::Changed(_) | steam::Outcome::Unchanged, false) => {
                                        let running = steam::steam_running(std::path::Path::new("/proc"));
                                        Some(neural_forge_supervisor::compat::switch_off(&root, &appid, running).map(|p| match p {
                                            Some(p) if p.is_empty() => "; it runs with Steam's default Proton again".to_string(),
                                            Some(p) => format!("; it runs with {p} again"),
                                            None => String::new(),
                                        }))
                                    }
                                    _ => None,
                                };
                                (outcome, proton)
                            })
                            .await
                            .map(|(o, p)| {
                                proton_note.replace(p);
                                o
                            });
                            let Some(row) = row.upgrade() else { return };
                            let revert = || {
                                reverting.set(true);
                                row.set_active(!on);
                                reverting.set(false);
                            };
                            match outcome {
                                Ok(steam::Outcome::Changed(_) | steam::Outcome::Unchanged) => {
                                    let state = if on { "on" } else { "off" };
                                    let note = match proton_note.take() {
                                        Some(Ok(n)) => n,
                                        Some(Err(e)) => format!(
                                            ". Its Proton was not changed ({e}): in a DLSS game the model stays off until it runs with a Neural Forge Proton"
                                        ),
                                        None => String::new(),
                                    };
                                    toasts.add_toast(toast(&format!("Neural Forge is {state} for {}{note}", game.name)));
                                }
                                Ok(steam::Outcome::SteamRunning { manual }) => {
                                    revert();
                                    running.set(true);
                                    if let (Some(rows), Some(group)) = (rows_for_row.upgrade(), group_for_row.upgrade()) {
                                        for (_, other) in rows.borrow().iter() {
                                            other.set_sensitive(false);
                                        }
                                        group.set_description(Some(&games_note(&steam::roots(), rows.borrow().len(), true)));
                                    }
                                    toasts.add_toast(manual_toast("Steam is running: close it first, or set the launch options by hand", manual));
                                }
                                Ok(steam::Outcome::Failed { reason, manual }) => {
                                    revert();
                                    toasts.add_toast(manual_toast(&format!("Could not change the launch options safely: {reason}"), manual));
                                }
                                Err(_) => {
                                    revert();
                                    toasts.add_toast(toast("Could not change the launch options: the worker stopped"));
                                }
                            }
                            // As the scan sets it: a game switched off that cannot be switched on (the Flatpak
                            // Steam's, or one without a target) keeps its switch insensitive.
                            row.set_sensitive(switch_usable(&game, &target.borrow(), row.is_active(), running.get()));
                        });
                    };
                    // Switching on a game with anti-cheat asks first; switching off never does.
                    if on && !anti_cheat.is_empty() {
                        confirm_anti_cheat(row, &game.name, &anti_cheat, &reverting, apply);
                    } else {
                        apply();
                    }
                });
                group.add(&row);
                rows.borrow_mut().push((row, switch));
            }
        }
    };
    (group, fill)
}

/// Whether a game's switch can be used: not while Steam runs; off always, on only for a game that
/// can load the layer and has a target.
fn switch_usable(game: &neural_forge_supervisor::steam::Game, target: &str, on: bool, steam_running: bool) -> bool {
    !steam_running && (on || (game.unsupported().is_none() && !target.is_empty()))
}

fn build_setup_page(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesPage {
    let page = adw::PreferencesPage::new();
    page.add(&build_model_group(shm, toasts));
    let refresh = gtk4::Button::from_icon_name("view-refresh-symbolic");
    refresh.set_tooltip_text(Some("Look for games again"));
    refresh.add_css_class("flat");
    refresh.set_valign(gtk4::Align::Center);
    let (games_group, fill_games) = build_games_group(toasts, &refresh);
    let (launch_group, fill_launch) = build_launch_option_group();
    page.add(&games_group);
    page.add(&launch_group);
    page.add(&build_proton_group(toasts));
    let fill = std::rc::Rc::new(move |scan: &SteamScan| {
        fill_games(scan);
        fill_launch(scan);
    });
    let scan = move |button: Option<gtk4::Button>| {
        let fill = std::rc::Rc::clone(&fill);
        if let Some(button) = &button {
            button.set_sensitive(false);
        }
        glib::spawn_future_local(async move {
            if let Ok(scan) = gio::spawn_blocking(scan_steam).await {
                fill(&scan);
            }
            if let Some(button) = button {
                button.set_sensitive(true);
            }
        });
    };
    scan(None);
    refresh.connect_clicked(move |button| scan(Some(button.clone())));
    page
}

/// What the Proton group lists: copies can be made from `sources`; `tools` are the copies, each with whether its
/// source changed since.
struct ProtonScan {
    root: Option<std::path::PathBuf>,
    sources: Vec<neural_forge_supervisor::proton_tool::Source>,
    tools: Vec<(neural_forge_supervisor::proton_tool::Tool, bool)>,
}

fn scan_protons() -> ProtonScan {
    use neural_forge_supervisor::{proton_tool, steam};
    let Some(root) = steam::roots().into_iter().find(|r| !r.flatpak).map(|r| r.path) else {
        return ProtonScan { root: None, sources: Vec::new(), tools: Vec::new() };
    };
    let libraries = steam::libraries(&root);
    let tools: Vec<(proton_tool::Tool, bool)> = proton_tool::tools(&root).into_iter().map(|t| (t.clone(), proton_tool::outdated(&t))).collect();
    // A source whose current copy exists is not offered again.
    let sources = proton_tool::sources(&root, &libraries).into_iter().filter(|s| !tools.iter().any(|(t, _)| t.name == s.tool_name())).collect();
    ProtonScan { root: Some(root), sources, tools }
}

/// "Proton with DLSS's parameters": copies of installed Protons that load the NGX wrapper
/// (`neural_forge_supervisor::proton_tool`), made and removed here, with the notice (Alex 2026-11-10: a clear
/// notice, nothing blocked).
fn build_proton_group(toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    use neural_forge_supervisor::proton_tool;
    let group = adw::PreferencesGroup::new();
    group.set_title("Proton with DLSS parameters");
    group.set_description(Some(proton_tool::NOTICE));
    let rows: std::rc::Rc<std::cell::RefCell<Vec<adw::ActionRow>>> = std::rc::Rc::default();
    type Refill = std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<dyn Fn()>>>>;
    let refill: Refill = std::rc::Rc::default();
    let fill = {
        let (group, rows, toasts, refill) = (group.clone(), rows.clone(), toasts.clone(), refill.clone());
        move |scan: ProtonScan| {
            for row in rows.borrow_mut().drain(..) {
                group.remove(&row);
            }
            let Some(root) = scan.root else {
                let row = adw::ActionRow::new();
                row.set_title("No native Steam installation found");
                group.add(&row);
                rows.borrow_mut().push(row);
                return;
            };
            for (tool, outdated) in &scan.tools {
                let row = adw::ActionRow::new();
                row.set_use_markup(false);
                row.set_title(&tool.name);
                row.set_subtitle(&if *outdated {
                    format!("from {}, which has changed since: make it again, choose the new copy for your games, then remove this one", tool.source.display())
                } else {
                    format!("from {}", tool.source.display())
                });
                if *outdated {
                    let again = gtk4::Button::with_label("Make again");
                    again.set_valign(gtk4::Align::Center);
                    let (root, source, toasts, refill) = (root.clone(), tool.source.clone(), toasts.clone(), refill.clone());
                    again.connect_clicked(move |b| {
                        b.set_sensitive(false);
                        let (root, source, toasts, refill) = (root.clone(), source.clone(), toasts.clone(), refill.clone());
                        glib::spawn_future_local(async move {
                            let done = gio::spawn_blocking(move || {
                                let libs = neural_forge_supervisor::steam::libraries(&root);
                                let s = proton_tool::sources(&root, &libs).into_iter().find(|s| s.dir == source).ok_or("its source is gone".to_string())?;
                                proton_tool::create(&root, &s, &proton_tool::installed_wrapper())
                                    .map(|m| format!("Made {}: restart Steam to see it", m.tool.name))
                            })
                            .await;
                            toasts.add_toast(toast(
                                &done.unwrap_or_else(|_| Err("the worker stopped".into())).unwrap_or_else(|e| format!("Could not make the copy: {e}")),
                            ));
                            if let Some(f) = refill.borrow().clone() {
                                f();
                            }
                        });
                    });
                    row.add_suffix(&again);
                }
                let remove = gtk4::Button::with_label("Remove");
                remove.set_valign(gtk4::Align::Center);
                remove.add_css_class("destructive-action");
                let (root_r, name, toasts_r, refill_r) = (root.clone(), tool.name.clone(), toasts.clone(), refill.clone());
                remove.connect_clicked(move |b| {
                    b.set_sensitive(false);
                    let (root, name, toasts, refill) = (root_r.clone(), name.clone(), toasts_r.clone(), refill_r.clone());
                    glib::spawn_future_local(async move {
                        let done = gio::spawn_blocking(move || {
                            proton_tool::remove(&root, &name).map(|t| format!("Removed {}: games set to it need another Proton in Steam", t.name))
                        })
                        .await;
                        toasts.add_toast(toast(
                            &done.unwrap_or_else(|_| Err("the worker stopped".into())).unwrap_or_else(|e| format!("Could not remove it: {e}")),
                        ));
                        if let Some(f) = refill.borrow().clone() {
                            f();
                        }
                    });
                });
                row.add_suffix(&remove);
                group.add(&row);
                rows.borrow_mut().push(row);
            }
            for source in &scan.sources {
                let row = adw::ActionRow::new();
                row.set_use_markup(false);
                row.set_title(&source.display);
                row.set_subtitle(&format!("{} in {}", source.version, source.dir.display()));
                let make = gtk4::Button::with_label("Make copy");
                make.set_valign(gtk4::Align::Center);
                make.set_tooltip_text(Some(&format!("Make {} (see the notice above)", source.tool_name())));
                let (root_m, source, toasts_m, refill_m) = (root.clone(), source.clone(), toasts.clone(), refill.clone());
                make.connect_clicked(move |b| {
                    b.set_sensitive(false);
                    let (root, source, toasts, refill) = (root_m.clone(), source.clone(), toasts_m.clone(), refill_m.clone());
                    glib::spawn_future_local(async move {
                        let done = gio::spawn_blocking(move || {
                            proton_tool::create(&root, &source, &proton_tool::installed_wrapper())
                                .map(|m| format!("Made {}: restart Steam, then choose it for a game in its Properties > Compatibility", m.tool.name))
                        })
                        .await;
                        toasts.add_toast(toast(
                            &done.unwrap_or_else(|_| Err("the worker stopped".into())).unwrap_or_else(|e| format!("Could not make the copy: {e}")),
                        ));
                        if let Some(f) = refill.borrow().clone() {
                            f();
                        }
                    });
                });
                row.add_suffix(&make);
                group.add(&row);
                rows.borrow_mut().push(row);
            }
        }
    };
    let fill = std::rc::Rc::new(fill);
    let rescan: std::rc::Rc<dyn Fn()> = std::rc::Rc::new(move || {
        let fill = fill.clone();
        glib::spawn_future_local(async move {
            if let Ok(scan) = gio::spawn_blocking(scan_protons).await {
                fill(scan);
            }
        });
    });
    *refill.borrow_mut() = Some(rescan.clone());
    rescan();
    group
}

/// One sample of the graph's two series: how long a frame took and how much of it the model's network took on
/// the GPU, sampled from the header on a faster timer than the once-a-second status labels need. The last few
/// seconds are kept for the drawing area.
#[derive(Clone, Copy)]
struct TelemetrySample {
    /// Milliseconds per presented frame over the sample's interval (`None`: no frames, or no game).
    frame_ms: Option<f32>,
    /// The network's GPU milliseconds on its last hold (`preupscale_network_gpu_ms`); `None` while it does not run.
    model_ms: Option<f32>,
}

const TELEMETRY_INTERVAL_MS: u32 = 200;
const TELEMETRY_WINDOW_SECS: u32 = 10;
const TELEMETRY_SAMPLES: usize = (TELEMETRY_WINDOW_SECS * 1000 / TELEMETRY_INTERVAL_MS) as usize;
const FRAME_RGB: (f64, f64, f64) = (0.596, 0.478, 0.953);
const MODEL_RGB: (f64, f64, f64) = (0.204, 0.780, 0.678);

/// The graph's top in milliseconds: the smallest of 5, 10, 20, 40 and 80 above every sample, so a line's height
/// means the same while the frame rate holds and the scale only moves for a real change.
fn telemetry_ceiling(history: &std::collections::VecDeque<TelemetrySample>) -> f32 {
    let top = history.iter().flat_map(|s| [s.frame_ms, s.model_ms]).flatten().fold(0.0f32, f32::max);
    [5.0, 10.0, 20.0, 40.0].into_iter().find(|&c| top <= c * 0.95).unwrap_or(80.0)
}

fn draw_telemetry_graph(cr: &gtk4::cairo::Context, width: i32, height: i32, history: &std::collections::VecDeque<TelemetrySample>) {
    let (width, height) = (f64::from(width), f64::from(height));
    let _ = cr.save();
    let ceiling = telemetry_ceiling(history);
    // The scale: a line at the top, the middle and the bottom, each labelled, in the text colour at low alpha.
    let fg = (0.5, 0.5, 0.5);
    cr.set_font_size(10.0);
    let label_w = 34.0;
    for (frac, value) in [(0.0, ceiling), (0.5, ceiling / 2.0), (1.0, 0.0)] {
        let y = (height - 1.0) * frac + 0.5;
        cr.set_source_rgba(fg.0, fg.1, fg.2, 0.25);
        cr.set_line_width(1.0);
        cr.move_to(label_w, y);
        cr.line_to(width, y);
        let _ = cr.stroke();
        cr.set_source_rgba(fg.0, fg.1, fg.2, 0.9);
        let text = format!("{value} ms");
        let baseline = if frac == 0.0 {
            y + 9.0
        } else if frac == 1.0 {
            y - 2.0
        } else {
            y + 3.5
        };
        cr.move_to(0.0, baseline);
        let _ = cr.show_text(&text);
    }
    let plot_w = width - label_w;
    let n = TELEMETRY_SAMPLES.max(2);
    // The newest sample at the right edge; a gap where a series has no value.
    let plot = |pick: fn(&TelemetrySample) -> Option<f32>| {
        let mut drawing = false;
        let start = n - history.len();
        for (i, sample) in history.iter().enumerate() {
            let x = label_w + plot_w * ((start + i) as f64) / ((n - 1) as f64);
            match pick(sample) {
                Some(v) => {
                    let y = (height - 1.0) * (1.0 - f64::from(v.min(ceiling) / ceiling)) + 0.5;
                    if drawing {
                        cr.line_to(x, y);
                    } else {
                        cr.move_to(x, y);
                    }
                    drawing = true;
                }
                None => drawing = false,
            }
        }
        let _ = cr.stroke();
    };
    cr.set_line_width(1.6);
    cr.set_source_rgb(FRAME_RGB.0, FRAME_RGB.1, FRAME_RGB.2);
    plot(|s| s.frame_ms);
    cr.set_source_rgb(MODEL_RGB.0, MODEL_RGB.1, MODEL_RGB.2);
    plot(|s| s.model_ms);
    let _ = cr.restore();
}

fn legend_label(text: &str, rgb: (f64, f64, f64)) -> (gtk4::Box, gtk4::Label) {
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
    let label = gtk4::Label::new(Some(text));
    row.append(&label);
    (row, label)
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
    let (frame_key, frame_label) = legend_label("Frame time", FRAME_RGB);
    let (model_key, model_label) = legend_label("Model (GPU)", MODEL_RGB);
    legend.append(&frame_key);
    legend.append(&model_key);

    let sparkline = gtk4::DrawingArea::new();
    // A minimum, not a ceiling: `AdwPreferencesGroup` still stretches this row taller
    // than 80px on a short page with room to spare (confirmed visually; `vexpand`
    // false the whole way down this box's ancestor chain didn't stop it either).
    // Harmless -- `draw_telemetry_graph` normalizes against whatever height it's
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
        sparkline.set_draw_func(move |_, cr, width, height| draw_telemetry_graph(cr, width, height, &history.borrow()));
    }

    let shm_for_timer = std::sync::Arc::clone(shm);
    // The last frame count and when it was read: the rate is over the time that really passed (a
    // busy main loop runs the timer late).
    let last_layer_frames: std::cell::Cell<Option<(u64, std::time::Instant)>> = std::cell::Cell::new(None);
    let live = Liveness::default();
    every(&sparkline, std::time::Duration::from_millis(u64::from(TELEMETRY_INTERVAL_MS)), move |sparkline| {
        let hdr = shm_for_timer.header();

        model_row.set_subtitle(if hdr.native_running.load(Ordering::Relaxed) != 0 { "running before the upscaler" } else { "not running before the upscaler" });
        // The name is only as current as the frames: a closed game leaves its name behind.
        let game = hdr.game_name();
        let active = live.alive(hdr);
        game_row.set_subtitle(&match (game.is_empty(), active) {
            (_, false) => "none attached".to_string(),
            (true, true) => "attached".to_string(),
            (false, true) => game,
        });

        let layer_frames = neural_forge_protocol::load64(&hdr.layer_frames_lo, &hdr.layer_frames_hi);
        let now = std::time::Instant::now();
        let layer_fps = last_layer_frames.get().and_then(|(prev, at)| {
            let secs = now.duration_since(at).as_secs_f64();
            (secs > 0.0).then(|| layer_frames.saturating_sub(prev) as f64 / secs)
        });
        last_layer_frames.set(Some((layer_frames, now)));
        match layer_fps {
            Some(l) if active => fps_row.set_subtitle(&format!("{l:.1} presents/s")),
            _ => fps_row.set_subtitle("—"),
        }

        let model = f32::from_bits(hdr.preupscale_network_gpu_ms_bits.load(Ordering::Relaxed));
        let sample = TelemetrySample {
            frame_ms: layer_fps.filter(|&f| active && f > 0.5).map(|f| (1000.0 / f) as f32),
            model_ms: (active && hdr.native_running.load(Ordering::Relaxed) != 0 && model.is_finite() && model > 0.0).then_some(model),
        };
        let ms = |v: Option<f32>| v.map_or_else(|| "—".to_string(), |v| format!("{v:.1} ms"));
        frame_label.set_text(&format!("Frame time {}", ms(sample.frame_ms)));
        model_label.set_text(&format!("Model (GPU) {}", ms(sample.model_ms)));
        {
            let mut history = history.borrow_mut();
            if history.len() == TELEMETRY_SAMPLES {
                history.pop_front();
            }
            history.push_back(sample);
        }
        sparkline.queue_draw();
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
    let live = Liveness::default();
    let preupscale_for_timer = preupscale_row.clone();
    every(&layer_row, SECOND, move |layer_row| {
        let preupscale_row = &preupscale_for_timer;
        let hdr = shm_for_timer.header();
        // A header laid out by another build is refused at open (see `Shm::open`), so the
        // version needs no check here.
        let layer_active = live.alive(hdr);
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
            hdr.placement.load(Ordering::Relaxed),
            hdr.preupscale_state.load(Ordering::Relaxed),
            hdr.preupscale_width.load(Ordering::Relaxed),
            hdr.preupscale_height.load(Ordering::Relaxed),
            hdr.preupscale_misses.load(Ordering::Relaxed),
            &hdr.preupscale_reason(),
        );
        // The layer's own status line: the model running, or why it is not.
        let layer_reason = if layer_active { hdr.layer_reason() } else { String::new() };
        preupscale_row.set_subtitle(&if layer_reason.is_empty() { label } else { format!("{label} -- {layer_reason}") });
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
                .body(
                    "Every tuning value of the settings shown (Settings for, at the top) returns to its \
                       default. The running layer session (frame counters, transport state) is not affected.",
                )
                .default_response("cancel")
                .close_response("cancel")
                .build();
            dialog.add_response("cancel", "Cancel");
            dialog.add_response("reset", "Reset");
            dialog.set_response_appearance("reset", adw::ResponseAppearance::Destructive);
            dialog.choose(parent.as_ref(), None::<&gio::Cancellable>, move |response| {
                if response != "reset" {
                    return;
                }
                // Reset the live mapping the running layer is already
                // attached to, then overwrite config.ini with the same defaults so a
                // restart doesn't just reload the values this just cleared -- the
                // same two places `persist_one` already keeps in sync for a single
                // setting, done here for all of them at once.
                shm.header().reset_persisted_settings();
                match neural_forge_supervisor::save_settings(shm.header()) {
                    Ok(()) => toasts.add_toast(toast("Settings reset")),
                    Err(e) => toasts.add_toast(toast(&format!("Reset the live session, but saving the settings failed: {e}"))),
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
            if let Some(why) = neural_forge_supervisor::profiles::invalid_name(&name) {
                toasts.add_toast(toast(&format!("Not saved: {why}")));
                return;
            }
            let settings = neural_forge_protocol::persist::snapshot(shm.header());
            match neural_forge_supervisor::profiles::save_profile(&name, settings) {
                Ok(()) => {
                    toasts.add_toast(toast(&format!("Saved profile \"{name}\"")));
                    save_profile_row.set_text("");
                    refresh_profile_combo(&profile_combo);
                }
                Err(e) => toasts.add_toast(toast(&format!("Save failed: {e}"))),
            }
        });
    }

    {
        let shm = std::sync::Arc::clone(shm);
        let toasts = toasts.clone();
        let profile_combo = profile_combo.clone();
        load_profile_button.connect_clicked(move |_| {
            let Some(name) = selected_profile(&profile_combo) else {
                toasts.add_toast(toast("No profile selected"));
                return;
            };
            let profiles = neural_forge_supervisor::profiles::load_all();
            let Some(settings) = profiles.get(&name) else {
                toasts.add_toast(toast(&format!("Profile \"{name}\" is gone")));
                refresh_profile_combo(&profile_combo);
                return;
            };
            neural_forge_protocol::persist::apply(shm.header(), settings);
            // Same reasoning as the reset button above: applying to the live header
            // only affects the running session, so also save the result (to the settings
            // the header holds: all games' or the game's) so it survives a reboot too.
            let message = match neural_forge_supervisor::save_settings(shm.header()) {
                Ok(()) => format!("Loaded profile \"{name}\""),
                Err(e) => format!("Applied to the running session, but saving the settings failed: {e}"),
            };
            toasts.add_toast(toast(&message));
        });
    }

    {
        let toasts = toasts.clone();
        let profile_combo = profile_combo.clone();
        delete_profile_button.connect_clicked(move |_| {
            let Some(name) = selected_profile(&profile_combo) else {
                toasts.add_toast(toast("No profile selected"));
                return;
            };
            match neural_forge_supervisor::profiles::delete_profile(&name) {
                Ok(true) => {
                    toasts.add_toast(toast(&format!("Deleted profile \"{name}\"")));
                    refresh_profile_combo(&profile_combo);
                }
                Ok(false) => toasts.add_toast(toast("Profile already gone")),
                Err(e) => toasts.add_toast(toast(&format!("Delete failed: {e}"))),
            }
        });
    }

    group
}

/// The Diagnosis group: the Diagnose button runs `neural-forge-cli doctor`'s checks
/// (`neural_forge_supervisor::doctor`) and lists the findings, worst first. They run on a worker thread:
/// `nvidia-smi` and `journalctl` can take seconds.
fn build_diagnosis_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Diagnosis");
    group.set_description(Some("Why neural rendering is or is not running: the install, the GPU and driver, the kernel log and the last game session"));
    // The group's actions, side by side in its header.
    let actions = gtk4::Box::new(gtk4::Orientation::Horizontal, 6);
    let diagnose = gtk4::Button::with_label("Diagnose");
    diagnose.add_css_class("suggested-action");
    diagnose.set_valign(gtk4::Align::Center);
    actions.append(&diagnose);
    let save = gtk4::Button::with_label("Save report");
    save.set_valign(gtk4::Align::Center);
    save.set_tooltip_text(Some(
        "Saves a zip to the desktop to attach to a problem report: these findings (which list your installed \
         Steam games, their library folders and whether Neural Forge is on in them; of a game's launch options only \
         Neural Forge's part), system details (GPU, driver, kernel, session), settings, the extracted model's \
         manifests, the layer's logs, the kernel's GPU errors and the newest capture. Your home folder, user name \
         and computer name are replaced; nothing is uploaded",
    ));
    connect_save_report(&save, shm, toasts);
    actions.append(&save);
    group.set_header_suffix(Some(&actions));

    let rows: std::rc::Rc<std::cell::RefCell<Vec<adw::ActionRow>>> = std::rc::Rc::default();
    let weak_group = group.downgrade();
    diagnose.connect_clicked(move |button| {
        let Some(group) = weak_group.upgrade() else { return };
        button.set_sensitive(false);
        button.set_label("Diagnosing…");
        let (button, rows) = (button.clone(), rows.clone());
        glib::spawn_future_local(async move {
            let findings = gio::spawn_blocking(|| neural_forge_supervisor::doctor::run(&neural_forge_supervisor::doctor::Roots::system())).await;
            for row in rows.borrow_mut().drain(..) {
                group.remove(&row);
            }
            let findings = findings.unwrap_or_else(|_| {
                vec![neural_forge_supervisor::doctor::Finding {
                    check: "diagnose",
                    severity: neural_forge_supervisor::doctor::Severity::Failure,
                    summary: "the checks stopped with an internal error".into(),
                    evidence: "run `neural-forge-cli doctor` in a terminal to see it".into(),
                    fix: None,
                    report_evidence: None,
                }]
            });
            for finding in &findings {
                let row = finding_row(finding);
                group.add(&row);
                rows.borrow_mut().push(row);
            }
            button.set_label("Diagnose");
            button.set_sensitive(true);
        });
    });
    group
}

/// Opens `dir` in the file manager, creating it first if it is missing.
fn open_folder(dir: &std::path::Path, parent: Option<gtk4::Window>, toasts: &adw::ToastOverlay) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        toasts.add_toast(toast(&format!("Couldn't create {}: {e}", dir.display())));
        return;
    }
    let toasts = toasts.clone();
    gtk4::FileLauncher::new(Some(&gio::File::for_path(dir))).launch(parent.as_ref(), None::<&gio::Cancellable>, move |result| {
        if let Err(e) = result {
            toasts.add_toast(toast(&format!("Couldn't open the folder: {e}")));
        }
    });
}

/// How long a capture request may wait for a game to take it before the GUI withdraws it, and how
/// long a taken one-shot may take to be written.
const CAPTURE_TAKE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
const CAPTURE_RESULT_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// What the capture button's row says, from the channel: whether the model edits the game's own frame now
/// ([`edits_in_game_frame`]), where a capture has no frame without the model's edit (see the button's tooltip).
fn capture_subtitle(in_game_frame: bool) -> &'static str {
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
fn edits_in_game_frame(state: u32) -> bool {
    use neural_forge_protocol::enums::preupscale_state::{HOLDING, HOLDING_AFTER, HOLDING_BOTH, PAUSED};
    // Paused with DLSS (#49) the post path is still kept off: the layer captures the presented frame as both images.
    matches!(state, HOLDING | HOLDING_AFTER | HOLDING_BOTH | PAUSED)
}

fn build_capture_group(shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) -> adw::PreferencesGroup {
    let group = adw::PreferencesGroup::new();
    group.set_title("Captures");

    let capture_row = adw::ActionRow::new();
    capture_row.set_title("Capture before/after");
    capture_row.set_subtitle(capture_subtitle(false));
    let capture_button = gtk4::Button::with_label("Capture");
    capture_button.set_valign(gtk4::Align::Center);
    // The game's layer serves the request (`neural_forge_layer::series`) and publishes what became of
    // it (`neural_forge_protocol::capture`), which the toast reports.
    capture_button.set_tooltip_text(Some(
        "Saves the next frame the game presents as <time>-original.png (the game's frame) and \
         <time>-composited.png (what Neural Forge presented). While the model runs in the game's frame \
         (before the upscaler or after DLSS, in a DLSS game), the layer has no frame without the model's \
         edit when the game presents, so both images are the presented frame. HDR and 10-bit games \
         cannot be captured.",
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
        // The request this window is waiting for: its id and when it was made.
        let waiting: std::rc::Rc<std::cell::Cell<Option<(u32, std::time::Instant)>>> = std::rc::Rc::default();
        capture_button.connect_clicked(move |button| {
            use neural_forge_protocol::capture;
            let header = shm.header();
            // A second click while waiting cancels.
            if let Some((id, _)) = waiting.take() {
                button.set_label("Capture");
                if capture::cancel(header, id) == capture::Cancelled::Withdrawn {
                    toasts.add_toast(toast("Capture cancelled"));
                }
                return;
            }
            let id = match capture::request(header, 1) {
                Ok(id) => id,
                Err(e) => {
                    toasts.add_toast(toast(&format!("Not captured: {e}")));
                    return;
                }
            };
            waiting.set(Some((id, std::time::Instant::now())));
            button.set_label("Cancel");
            let shm = std::sync::Arc::clone(&shm);
            let toasts = toasts.clone();
            let captures_dir = captures_dir.clone();
            let waiting = waiting.clone();
            let button = button.downgrade();
            glib::timeout_add_local(std::time::Duration::from_millis(250), move || {
                let Some((id, since)) = waiting.get() else { return glib::ControlFlow::Break };
                let Some(button) = button.upgrade() else { return glib::ControlFlow::Break };
                let header = shm.header();
                let done = |message: String, folder: bool| {
                    waiting.set(None);
                    button.set_label("Capture");
                    let toast = adw::Toast::builder().title(glib::markup_escape_text(&message)).timeout(if folder { 6 } else { 8 }).build();
                    if folder {
                        toast.set_button_label(Some("Open folder"));
                        let parent = button.root().and_downcast::<gtk4::Window>();
                        let toasts_for_button = toasts.clone();
                        let dir = captures_dir.clone();
                        toast.connect_button_clicked(move |_| open_folder(&dir, parent.clone(), &toasts_for_button));
                    }
                    toasts.add_toast(toast);
                    glib::ControlFlow::Break
                };
                if let Some(outcome) = capture::result(header).filter(|o| o.id == id && o.finished()) {
                    let written = outcome.state == capture::state::WRITTEN;
                    return done(outcome.message(), written);
                }
                let taken = capture::pending(header).is_none_or(|r| r.id != id);
                let elapsed = since.elapsed();
                if !taken && elapsed >= CAPTURE_TAKE_WAIT {
                    // Nobody took it: withdrawn, so it can never fire on a game started later.
                    if capture::cancel(header, id) == capture::Cancelled::Withdrawn {
                        return done(
                            "Not captured: no game took the capture. Start a game with Neural Forge on and wait until it is past its loading screen".into(),
                            false,
                        );
                    }
                }
                if elapsed >= CAPTURE_RESULT_WAIT {
                    return done("Not captured: the game did not finish the capture (its log may say why)".into(), false);
                }
                glib::ControlFlow::Continue
            });
        });
    }

    // The model in the game's frame: from the channel, while the layer's heartbeat is fresh.
    let shm = std::sync::Arc::clone(shm);
    let live = Liveness::default();
    every(&capture_row, SECOND, move |capture_row| {
        let hdr = shm.header();
        let in_game_frame = edits_in_game_frame(hdr.preupscale_state.load(Ordering::Relaxed));
        capture_row.set_subtitle(capture_subtitle(live.alive(hdr) && in_game_frame));
    });
    group
}

/// "Save report": the diagnostic report (`neural_forge_supervisor::report`) on the desktop.
fn connect_save_report(button: &gtk4::Button, shm: &std::sync::Arc<neural_forge_protocol::mapping::Mapping>, toasts: &adw::ToastOverlay) {
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
            let saved = gio::spawn_blocking(move || neural_forge_supervisor::report::save(&status, &game, None)).await;
            button.set_sensitive(true);
            match saved {
                Ok(Ok(path)) => {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let dir = path.parent().map(|d| d.display().to_string()).unwrap_or_default();
                    let toast =
                        adw::Toast::builder().title(glib::markup_escape_text(&format!("Saved {name} to {dir}"))).button_label("Show").timeout(10).build();
                    let parent = button.root().and_downcast::<gtk4::Window>();
                    toast.connect_button_clicked(move |_| {
                        gtk4::FileLauncher::new(Some(&gio::File::for_path(&path))).open_containing_folder(parent.as_ref(), None::<&gio::Cancellable>, |_| {});
                    });
                    toasts.add_toast(toast);
                }
                Ok(Err(e)) => toasts.add_toast(toast(&format!("Saving the report failed: {e}"))),
                Err(_) => toasts.add_toast(toast("Saving the report failed: the worker panicked")),
            }
        });
    });
}

/// One finding as a row: the severity's icon, the summary as the title, the evidence and the fix below.
fn finding_row(finding: &neural_forge_supervisor::doctor::Finding) -> adw::ActionRow {
    use neural_forge_supervisor::doctor::Severity;
    let row = adw::ActionRow::new();
    // Plain text: evidence quotes log lines and paths, which may hold `<` or `&`.
    row.set_use_markup(false);
    row.set_title(&finding.summary);
    let subtitle = match &finding.fix {
        Some(fix) => format!("{}\nFix: {fix}", finding.evidence),
        None => finding.evidence.clone(),
    };
    row.set_subtitle(&subtitle);
    row.set_subtitle_selectable(true);
    row.set_tooltip_text(Some(finding.check));
    let (icon, class) = match finding.severity {
        Severity::Failure => ("dialog-error-symbolic", "error"),
        Severity::Warning => ("dialog-warning-symbolic", "warning"),
        Severity::Info => ("dialog-information-symbolic", "dim-label"),
        Severity::Ok => ("object-select-symbolic", "success"),
    };
    let image = gtk4::Image::from_icon_name(icon);
    image.add_css_class(class);
    row.add_prefix(&image);
    row
}

/// The Status page's one line on the pre-upscaler path, from the header's `preupscale_*` fields
/// (`neural_forge_protocol::enums::preupscale_state`). `layer_active`: the layer's heartbeat is
/// fresh; otherwise the fields are a closed game's leftovers. `reason`: why no hold runs while DLSS's input is
/// identified (`preupscale_reason`, with `IDENTIFIED`). `placement`: the Model placement setting; "At present"
/// runs the model on the finished frame by choice, so the line says that rather than "waiting" for DLSS (#49).
fn preupscale_label(layer_active: bool, placement: u32, state: u32, width: u32, height: u32, misses: u32, reason: &str) -> String {
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
        NO_WRAPPER => "off: this game uses DLSS but isn't on the Neural Forge Proton; switch it on in the Setup tab to set it (Steam closed)".to_string(),
        IDENTIFIED if reason.is_empty() => format!("after the upscaler: DLSS input identified, not holding{missed}"),
        IDENTIFIED => format!("after the upscaler: DLSS input identified, not holding{missed} -- {reason}"),
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
        OpenError::Unavailable => ("Couldn't open the shared-memory mapping", "neural-forge-cli doctor may help.".to_string()),
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
        // Edited text is GTK's to parse.
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
    fn hardware_codes_are_converted_without_underflow() {
        assert_eq!(evdev_keycode(95), Some(87)); // F11: XKB -> Linux evdev.
        assert_eq!(evdev_keycode(38), Some(30)); // physical A key on evdev.
        assert_eq!(evdev_keycode(0), None);
        assert_eq!(evdev_keycode(8), None);
        assert_eq!(evdev_keycode(u32::MAX), None);
    }
    #[test]
    fn key_labels_use_the_key_name_and_fall_back_to_the_code() {
        assert_eq!(format_key_label(87, Some("F11")), "F11");
        assert_eq!(format_key_label(30, Some("a")), "A");
        assert_eq!(format_key_label(70, Some("Scroll_Lock")), "Scroll_Lock");
        assert_eq!(format_key_label(87, None), "Key 87");
        assert_eq!(key_label(0), "Unbound");
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
