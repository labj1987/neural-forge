//! Round-trips [`crate::ShmHeader::persisted_settings`] through the plain
//! `set_<name>=<value>` string pairs `config.ini` actually stores, so a GUI/CLI
//! doesn't need to know `f32::to_bits()` exists -- it just hands this module a
//! `BTreeMap` it already has (from parsing/about to write the file) and a header.

use std::collections::BTreeMap;

use crate::ShmHeader;

/// Every persisted setting's current value, as `set_<name>` -> string, ready to
/// merge into whatever else `config.ini` is about to write out.
pub fn snapshot(header: &ShmHeader) -> BTreeMap<String, String> {
    let fmt = |is_float: bool, bits: u32| if is_float { f32::from_bits(bits).to_string() } else { bits.to_string() };
    header.persisted_settings().into_iter().map(|(name, is_float, bits)| (format!("set_{name}"), fmt(is_float, bits))).collect()
}

/// `config.ini`: `$XDG_CONFIG_HOME/neural-forge/config.ini`, or `~/.config/neural-forge/config.ini`.
pub fn config_file() -> String {
    let base = crate::env::var("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("{}/.config", std::env::var("HOME").unwrap_or_default()));
    format!("{base}/neural-forge/config.ini")
}

/// Every `key=value` line of the config file at `path` (blank lines and `#` comments skipped); empty
/// when it cannot be read.
pub fn read_config(path: &str) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else { return BTreeMap::new() };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect()
}

/// Applies the saved settings in [`config_file`] to `header` if nothing has since it was last
/// initialised (`tuning_seq` still 0): the layer does this when it finds a fresh mapping, since no
/// other process is running to do it. Returns whether it applied them.
pub fn apply_saved(header: &ShmHeader) -> bool {
    if header.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        return false;
    }
    apply(header, &read_config(&config_file()));
    header.control_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Applies whichever `set_<name>` keys `raw` has onto `header` -- silently skips a
/// key it doesn't recognize (an older/newer build's config.ini) or a value that
/// doesn't parse, rather than failing the whole load over one bad line.
pub fn apply(header: &ShmHeader, raw: &BTreeMap<String, String>) {
    for (name, is_float, _) in header.persisted_settings() {
        let Some(value) = raw.get(&format!("set_{name}")) else { continue };
        let bits = if is_float { value.parse::<f32>().ok().map(f32::to_bits) } else { value.parse::<u32>().ok() };
        if let Some(bits) = bits {
            header.apply_persisted_setting(name, bits);
        }
    }
    header.tuning_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::mapping;

    #[test]
    fn snapshot_then_apply_round_trips_every_setting() {
        let path = format!("{}/neural-forge-persist-test-{}/shm.bin", std::env::temp_dir().display(), std::process::id());
        let m = mapping::open_path(&path).expect("failed to create test mapping");
        let h = m.header();

        // Every persisted field gets its own value, distinct from its default and from
        // every other field's, so a name that `apply_persisted_setting` silently drops (or
        // maps onto the wrong field) shows up as a mismatch below.
        // The value is inside the setting's range, so it is stored unclamped.
        let defaults = h.persisted_settings();
        let value = |i: usize, is_float: bool, default_bits: u32, name: &str| {
            let (min, max) = crate::setting_bounds(name).expect(name);
            if is_float {
                let v = min + (max - min) * (0.1 + 0.8 * i as f32 / defaults.len() as f32);
                if v.to_bits() == default_bits { ((v + max) / 2.0).to_bits() } else { v.to_bits() }
            } else if default_bits == max as u32 {
                min as u32
            } else {
                default_bits + 1
            }
        };
        for (i, (name, is_float, default_bits)) in defaults.iter().enumerate() {
            let bits = value(i, *is_float, *default_bits, name);
            assert_ne!(bits, *default_bits, "{name}");
            assert!(h.apply_persisted_setting(name, bits), "{name}");
        }
        for (i, ((name, is_float, bits), (_, _, default_bits))) in h.persisted_settings().iter().zip(defaults.iter()).enumerate() {
            assert_eq!(*bits, value(i, *is_float, *default_bits, name), "{name} was not stored by apply_persisted_setting");
        }
        let snap = snapshot(h);

        h.init_defaults();
        assert_eq!(h.persisted_settings(), defaults);

        apply(h, &snap);
        assert_eq!(snapshot(h), snap, "every setting must survive reinitialization");
        for ((name, _, restored), (_, _, default_bits)) in h.persisted_settings().iter().zip(defaults.iter()) {
            assert_ne!(restored, default_bits, "{name} came back at its default");
        }

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(std::path::Path::new(&path).parent().unwrap()).ok();
    }

    #[test]
    fn every_persisted_setting_has_bounds_holding_its_default() {
        let h = crate::ShmHeader::default();
        h.init_defaults();
        assert_eq!(crate::SETTING_BOUNDS.len(), h.persisted_settings().len());
        for (name, is_float, bits) in h.persisted_settings() {
            let (min, max) = crate::setting_bounds(name).unwrap_or_else(|| panic!("{name} has no bounds"));
            let v = if is_float { f32::from_bits(bits) } else { bits as f32 };
            assert!(min <= v && v <= max, "{name}'s default {v} is outside {min}..={max}");
        }
    }

    #[test]
    fn apply_rejects_non_finite_and_clamps_out_of_range_values() {
        let h = crate::ShmHeader::default();
        h.init_defaults();
        let intensity = || f32::from_bits(h.intensity_bits.load(std::sync::atomic::Ordering::Relaxed));
        let raw = |k: &str, v: &str| BTreeMap::from([(k.to_string(), v.to_string())]);

        for bad in ["nan", "NaN", "inf", "-inf", "infinity"] {
            apply(&h, &raw("set_intensity", bad));
            assert_eq!(intensity(), 1.0, "{bad} must be rejected");
        }
        apply(&h, &raw("set_intensity", "-2.5"));
        assert_eq!(intensity(), 0.0, "a negative intensity clamps to 0");
        apply(&h, &raw("set_intensity", "1e30"));
        assert_eq!(intensity(), 4.0, "an out-of-range intensity clamps to the maximum");
        apply(&h, &raw("set_working_scale", "0.01"));
        assert_eq!(f32::from_bits(h.working_scale_bits.load(std::sync::atomic::Ordering::Relaxed)), 0.25);
        apply(&h, &raw("set_model_interval", "0"));
        assert_eq!(h.model_interval.load(std::sync::atomic::Ordering::Relaxed), 1);
        apply(&h, &raw("set_reversible_mode", "4000000000"));
        assert_eq!(h.reversible_mode.load(std::sync::atomic::Ordering::Relaxed), 4);
        // A negative value for a whole-number setting does not parse and leaves it alone.
        apply(&h, &raw("set_style", "-1"));
        assert_eq!(h.style.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(!h.apply_persisted_setting("intensity", f32::NAN.to_bits()));
        assert!(!h.apply_persisted_setting("not_a_setting", 0));
    }

    #[test]
    fn apply_ignores_unknown_and_unparsable_keys() {
        let path = format!("{}/neural-forge-persist-test2-{}/shm.bin", std::env::temp_dir().display(), std::process::id());
        let m = mapping::open_path(&path).expect("failed to create test mapping");
        let h = m.header();

        let mut raw = BTreeMap::new();
        raw.insert("set_nonexistent_field".to_string(), "1".to_string());
        raw.insert("set_intensity".to_string(), "not a float".to_string());
        // Should not panic, and should leave intensity at its default.
        apply(h, &raw);
        assert_eq!(f32::from_bits(h.intensity_bits.load(std::sync::atomic::Ordering::Relaxed)), 1.0);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(std::path::Path::new(&path).parent().unwrap()).ok();
    }
}
