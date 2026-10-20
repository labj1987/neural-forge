//! Round-trips [`crate::ShmHeader::persisted_settings`] through the plain
//! `set_<name>=<value>` string pairs `config.ini` actually stores, so a GUI/CLI
//! doesn't need to know `f32::to_bits()` exists -- it just hands this module a
//! `BTreeMap` it already has (from parsing/about to write the file) and a header.

use std::collections::BTreeMap;

use crate::ShmHeader;

/// The one setting that is text, not a number: the HUD regions at present (`set_hud_regions=x,y,w,h;...`,
/// [`crate::HudRegion::parse_list`]).
pub const HUD_REGIONS: &str = "hud_regions";

/// Every persisted setting's current value, as `set_<name>` -> string, ready to
/// merge into whatever else `config.ini` is about to write out (the HUD regions as their text).
pub fn snapshot(header: &ShmHeader) -> BTreeMap<String, String> {
    let fmt = |is_float: bool, bits: u32| if is_float { f32::from_bits(bits).to_string() } else { bits.to_string() };
    let mut all: BTreeMap<String, String> = header.persisted_settings().into_iter().map(|(name, is_float, bits)| (format!("set_{name}"), fmt(is_float, bits))).collect();
    all.insert(format!("set_{HUD_REGIONS}"), crate::HudRegion::list_text(&header.hud_settings().regions));
    all
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
    // Read as bytes: one line that is not UTF-8 (a path in another encoding) costs that line's text
    // (read with U+FFFD), not every saved setting.
    let Ok(bytes) = std::fs::read(path) else { return BTreeMap::new() };
    let text = String::from_utf8_lossy(&bytes);
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
    // Not configured yet is decided again as the write is entered ([`apply_if_unconfigured`]); this
    // only spares reading the file when it is plainly too late.
    if header.tuning_seq.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        return false;
    }
    apply_if_unconfigured(header, &read_config(&config_file()))
}

/// [`apply`], only if nothing has written settings to `header` since it was initialised: decided as
/// the write section is entered, so a write that got in first (the GUI's, while a game's layer reads
/// `config.ini`) is never overwritten by the saved values. Returns whether they were applied.
pub fn apply_if_unconfigured(header: &ShmHeader, raw: &BTreeMap<String, String>) -> bool {
    let (settings, regions) = parse(header, raw);
    header.write_settings_if_unconfigured(&settings, regions.as_deref())
}

/// Applies whichever `set_<name>` keys `raw` has onto `header` -- silently skips a
/// key it doesn't recognize (an older/newer build's config.ini) or a value that
/// doesn't parse, rather than failing the whole load over one bad line.
pub fn apply(header: &ShmHeader, raw: &BTreeMap<String, String>) {
    let (settings, regions) = parse(header, raw);
    // One unit: a profile is never seen half applied, and `tuning_seq` marks the header configured.
    header.write_settings_and_hud(&settings, regions.as_deref());
}

/// The settings `raw` names, as `(name, bits)`, and its HUD region list if it has one that parses.
fn parse(header: &ShmHeader, raw: &BTreeMap<String, String>) -> (Vec<(&'static str, u32)>, Option<Vec<crate::HudRegion>>) {
    let mut settings = Vec::new();
    for (name, is_float, _) in header.persisted_settings() {
        let Some(value) = raw.get(&format!("set_{name}")) else { continue };
        let bits = if is_float { value.trim().parse::<f32>().ok().map(f32::to_bits) } else { value.trim().parse::<u32>().ok() };
        if let Some(bits) = bits {
            settings.push((name, bits));
        }
    }
    // A HUD region list that does not parse is left as it is, like any value that does not.
    let regions = raw.get(&format!("set_{HUD_REGIONS}")).and_then(|t| crate::HudRegion::parse_list(t).ok());
    (settings, regions)
}

#[cfg(test)]
mod tests {
    /// A line that is not UTF-8 costs only its own text: the other saved settings are still read.
    #[test]
    fn a_line_that_is_not_utf8_keeps_the_other_settings() {
        let dir = std::env::temp_dir().join(format!("neural-forge-persist-lossy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("config.ini");
        std::fs::write(&file, b"binaries=/home/x/caf\xe9\nset_intensity=0.5\n").unwrap();
        let read = read_config(&file.to_string_lossy());
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(read.get("set_intensity").map(String::as_str), Some("0.5"));
        assert_eq!(read.get("binaries").map(String::as_str), Some("/home/x/caf\u{fffd}"));
    }

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
    fn the_hud_regions_round_trip_as_text_and_reset_to_none() {
        let h = crate::ShmHeader::default();
        h.init_defaults();
        assert_eq!(snapshot(&h).get("set_hud_regions").map(String::as_str), Some(""), "no region by default");
        let raw = BTreeMap::from([("set_hud_regions".to_string(), "0.02,0.85,0.25,0.12; 0.7,0.02,0.28,0.1,soften,0.3".to_string())]);
        apply(&h, &raw);
        let regions = h.hud_settings().regions;
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[1], crate::HudRegion { x: 0.7, y: 0.02, w: 0.28, h: 0.1, mode: crate::enums::hud_mode::SOFTEN, strength: 0.3 });
        let snap = snapshot(&h);
        assert_eq!(snap["set_hud_regions"], "0.02,0.85,0.25,0.12,keep;0.7,0.02,0.28,0.1,soften,0.3");
        h.init_defaults();
        apply(&h, &snap);
        assert_eq!(h.hud_settings().regions, regions, "the text reads back the same regions");
        // A list that does not parse leaves the regions as they are.
        apply(&h, &BTreeMap::from([("set_hud_regions".to_string(), "0.1,0.1,nope".to_string())]));
        assert_eq!(h.hud_settings().regions, regions);
        h.reset_persisted_settings();
        assert!(h.hud_settings().regions.is_empty(), "a reset clears them");
        assert!(h.hud_settings().detect, "detection is on by default");
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

    /// The saved settings and a live write racing on a fresh header (a game's layer applying
    /// `config.ini` while the GUI writes the user's change): whichever enters first, the live write is
    /// what stays.
    #[test]
    fn the_saved_settings_never_overwrite_a_write_that_got_in_first() {
        let saved = BTreeMap::from([("set_intensity".to_string(), "2.0".to_string()), ("set_local_tone".to_string(), "2.0".to_string())]);
        let h = crate::ShmHeader::default();
        let intensity = || f32::from_bits(h.intensity_bits.load(std::sync::atomic::Ordering::Relaxed));
        let (mut saved_first, mut live_first) = (0, 0);
        for _ in 0..2000 {
            h.init_defaults();
            let go = std::sync::Barrier::new(2);
            let applied = std::thread::scope(|s| {
                s.spawn(|| {
                    go.wait();
                    h.write_settings(&[("intensity", 0.5f32.to_bits())]);
                });
                go.wait();
                apply_if_unconfigured(&h, &saved)
            });
            assert_eq!(intensity(), 0.5, "the live write was overwritten by the saved settings");
            if applied { saved_first += 1 } else { live_first += 1 }
            assert_eq!(f32::from_bits(h.local_tone_bits.load(std::sync::atomic::Ordering::Relaxed)), if applied { 2.0 } else { 1.0 }, "applied whole or not at all");
        }
        assert!(saved_first + live_first == 2000);
        assert!(!apply_if_unconfigured(&h, &saved), "configured: not applied again");
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
