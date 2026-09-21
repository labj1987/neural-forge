//! Binds GTK widgets to `neural_forge_protocol::ShmHeader` fields — this crate's equivalent
//! of upstream's `shm_binder.h/.cpp`, written fresh against the protocol crate's Rust
//! types (there's no logic in a per-field binder worth porting either way, just a
//! mechanical widget<->field mapping).
//!
//! Also persists every [`neural_forge_protocol::ShmHeader::persisted_settings`] value to
//! `config.ini` on change, and applies whatever was last persisted when this call is
//! the one that created the mapping fresh (see `Shm::open`) -- the SHM mapping itself
//! lives under `/tmp` and does not survive a reboot, so without this, every tuning
//! change would quietly reset on the next login. Confirmed missing by comparing
//! against upstream DLSS5VKLayer's real, installed `config.ini` on 2026-09-10, which
//! persists the equivalent of every one of these fields.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use neural_forge_protocol::mapping::Mapping;

/// Wraps the open mapping so the UI module can pass one `Arc` around to every
/// callback instead of re-opening or re-threading raw pointers everywhere.
pub struct Shm(pub Arc<Mapping>);

impl Shm {
    pub fn open() -> Option<Self> {
        let mapping = neural_forge_protocol::mapping::open()?;
        if mapping.freshly_created {
            let cfg = neural_forge_supervisor::Config::load();
            neural_forge_protocol::persist::apply(mapping.header(), &cfg.settings);
        }
        Some(Shm(Arc::new(mapping)))
    }
}

/// After every write, also persists this one setting to `config.ini` -- a full
/// load-modify-save of the file rather than threading a shared `Config` through every
/// binder closure, since this is a small local file and correctness (never writing a
/// stale copy of some other field) matters more than avoiding a few extra syscalls on
/// a settings change a human just triggered by hand.
fn persist_one(name: &str, is_float: bool, bits: u32) {
    let mut cfg = neural_forge_supervisor::Config::load();
    let value = if is_float { f32::from_bits(bits).to_string() } else { bits.to_string() };
    cfg.settings.insert(format!("set_{name}"), value);
    let _ = cfg.save();
}

/// Binds a GTK `Scale`/`SpinButton`-shaped float control to one `f32`-bits field:
/// reads the current value to initialize the widget, and writes back (bumping
/// `control_seq` so the layer/helper notice) whenever the widget changes. `name` must
/// match one of `ShmHeader::persisted_settings`'s names for the value to survive a
/// reboot; pass `None` for a field that isn't meant to (there are none of those among
/// the GUI's rows today, but the option exists for e.g. a future debug-only control).
pub fn bind_float(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (f32, impl Fn(f32) + 'static) {
    let initial = f32::from_bits(get(shm.header()).load(Ordering::Relaxed));
    let shm = Arc::clone(shm);
    let setter = move |value: f32| {
        get(shm.header()).store(value.to_bits(), Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
        if let Some(name) = name {
            persist_one(name, true, value.to_bits());
        }
    };
    (initial, setter)
}

/// Same shape as [`bind_float`], for a plain `u32`-valued field (enable flags,
/// enum-valued settings, etc).
pub fn bind_u32(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (u32, impl Fn(u32) + 'static) {
    let initial = get(shm.header()).load(Ordering::Relaxed);
    let shm = Arc::clone(shm);
    let setter = move |value: u32| {
        get(shm.header()).store(value, Ordering::Relaxed);
        shm.header().control_seq.fetch_add(1, Ordering::Relaxed);
        if let Some(name) = name {
            persist_one(name, false, value);
        }
    };
    (initial, setter)
}

/// Same shape again, for a boolean flag stored as `0`/`1` in a `u32` field.
pub fn bind_bool(
    shm: &Arc<Mapping>,
    name: Option<&'static str>,
    get: impl Fn(&neural_forge_protocol::ShmHeader) -> &std::sync::atomic::AtomicU32 + 'static,
) -> (bool, impl Fn(bool) + 'static) {
    let (initial, setter) = bind_u32(shm, name, get);
    (initial != 0, move |value: bool| setter(if value { 1 } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the real production path end to end: change a value through
    /// `bind_float`'s setter (as a GUI slider would), confirm it landed in
    /// `config.ini`, then open a brand-new mapping (simulating a reboot -- a fresh
    /// `/tmp`) and confirm `Shm::open` applied the persisted value automatically
    /// instead of leaving the hardcoded default.
    #[test]
    fn a_setting_changed_through_bind_float_survives_a_simulated_reboot() {
        let scratch = std::env::temp_dir().join(format!("neural-forge-shm-persist-test-{}", std::process::id()));
        let config_home = scratch.join("config");
        let shm_path = scratch.join("shm.bin");
        std::fs::create_dir_all(&config_home).unwrap();

        let prev_xdg_config = std::env::var("XDG_CONFIG_HOME").ok();
        let prev_shm = std::env::var("NEURAL_FORGE_SHM").ok();
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var("NEURAL_FORGE_SHM", &shm_path);

        {
            let shm = Shm::open().expect("first open should succeed and create a fresh mapping");
            let (initial, set_intensity) = bind_float(&shm.0, Some("intensity"), |h| &h.intensity_bits);
            assert_eq!(initial, 1.0, "default intensity should be the hardcoded default on a fresh mapping");
            set_intensity(1.75);
        }

        let saved = std::fs::read_to_string(config_home.join("neural-forge/config.ini")).expect("config.ini should exist");
        assert!(saved.contains("set_intensity=1.75"), "config.ini should have the new value:\n{saved}");

        // Simulate a reboot: the SHM file is what actually goes away (/tmp), not the
        // config file, so a fresh mapping at the same path is the right reproduction.
        std::fs::remove_file(&shm_path).ok();
        let shm = Shm::open().expect("second open should succeed and create another fresh mapping");
        let intensity_after = f32::from_bits(shm.0.header().intensity_bits.load(Ordering::Relaxed));
        assert_eq!(intensity_after, 1.75, "persisted value should have been applied on the fresh mapping");

        match prev_xdg_config {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        match prev_shm {
            Some(v) => std::env::set_var("NEURAL_FORGE_SHM", v),
            None => std::env::remove_var("NEURAL_FORGE_SHM"),
        }
        std::fs::remove_dir_all(&scratch).ok();
    }
}
