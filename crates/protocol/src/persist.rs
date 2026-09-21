//! Round-trips [`crate::ShmHeader::persisted_settings`] through the plain
//! `set_<name>=<value>` string pairs `config.ini` actually stores, so a GUI/CLI
//! doesn't need to know `f32::to_bits()` exists -- it just hands this module a
//! `BTreeMap` it already has (from parsing/about to write the file) and a header.

use std::collections::BTreeMap;

use crate::ShmHeader;

/// One pass's persisted fields: (key suffix, is_float, field).
fn pass_fields(p: &crate::PassControl) -> [(&'static str, bool, &std::sync::atomic::AtomicU32); 8] {
    [
        ("intensity", true, &p.intensity_bits),
        ("local_tone", true, &p.local_tone_bits),
        ("local_structure", true, &p.local_structure_bits),
        ("skin_structure", true, &p.skin_structure_bits),
        ("sharpness", true, &p.sharpness_bits),
        ("style", false, &p.style),
        ("preset", false, &p.preset),
        ("auto_mask", false, &p.auto_mask),
    ]
}

/// Every persisted setting's current value, as `set_<name>` -> string, ready to
/// merge into whatever else `config.ini` is about to write out. Per-pass overrides are
/// `set_pass_<n>_mask` (always written, so clearing an override is saved too) and, for a pass
/// with any override, `set_pass_<n>_<field>`.
pub fn snapshot(header: &ShmHeader) -> BTreeMap<String, String> {
    let fmt = |is_float: bool, bits: u32| if is_float { f32::from_bits(bits).to_string() } else { bits.to_string() };
    let mut out: BTreeMap<String, String> = header
        .persisted_settings()
        .into_iter()
        .map(|(name, is_float, bits)| (format!("set_{name}"), fmt(is_float, bits)))
        .collect();
    for (n, p) in header.pass.iter().enumerate() {
        let mask = p.override_mask.load(std::sync::atomic::Ordering::Relaxed);
        out.insert(format!("set_pass_{n}_mask"), mask.to_string());
        if mask != 0 {
            for (field, is_float, a) in pass_fields(p) {
                out.insert(format!("set_pass_{n}_{field}"), fmt(is_float, a.load(std::sync::atomic::Ordering::Relaxed)));
            }
        }
    }
    out
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
    for (n, p) in header.pass.iter().enumerate() {
        for (field, is_float, a) in pass_fields(p) {
            let Some(value) = raw.get(&format!("set_pass_{n}_{field}")) else { continue };
            let bits = if is_float { value.parse::<f32>().ok().map(f32::to_bits) } else { value.parse::<u32>().ok() };
            if let Some(bits) = bits {
                a.store(bits, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // The mask last, so a pass never names an override whose value has not been loaded.
        if let Some(mask) = raw.get(&format!("set_pass_{n}_mask")).and_then(|v| v.parse::<u32>().ok()) {
            p.override_mask.store(mask, std::sync::atomic::Ordering::Relaxed);
        }
    }
    header.tuning_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::mapping;

    #[test]
    fn per_pass_overrides_round_trip_and_a_cleared_override_stays_cleared() {
        let path = format!("{}/neural-forge-persist-pass-test-{}/shm.bin", std::env::temp_dir().display(), std::process::id());
        let m = mapping::open_at(&path).expect("mapping");
        let h = m.header();
        use std::sync::atomic::Ordering::Relaxed;
        h.pass[1].intensity_bits.store(2.5f32.to_bits(), Relaxed);
        h.pass[1].style.store(2, Relaxed);
        h.pass[1].override_mask.store(crate::enums::pass_override::INTENSITY | crate::enums::pass_override::STYLE, Relaxed);
        let snap = snapshot(h);
        let mask = (crate::enums::pass_override::INTENSITY | crate::enums::pass_override::STYLE).to_string();
        assert_eq!(snap.get("set_pass_1_mask"), Some(&mask));
        assert_eq!(snap.get("set_pass_1_intensity").map(String::as_str), Some("2.5"));
        assert_eq!(snap.get("set_pass_0_mask").map(String::as_str), Some("0"));
        assert!(!snap.contains_key("set_pass_0_intensity"));

        h.init_defaults();
        assert_eq!(h.pass[1].override_mask.load(Relaxed), 0);
        apply(h, &snap);
        let t = h.resolve_pass(1);
        assert_eq!((t.intensity, t.style), (2.5, 2));
        assert_eq!(h.resolve_pass(0).intensity, 1.0);

        // Clearing the override and saving again must not bring it back.
        h.pass[1].override_mask.store(0, Relaxed);
        let mut merged = snap.clone();
        merged.extend(snapshot(h));
        h.init_defaults();
        h.pass[1].override_mask.store(0, Relaxed);
        apply(h, &merged);
        assert_eq!(h.resolve_pass(1).intensity, 1.0);
        std::fs::remove_dir_all(std::path::Path::new(&path).parent().unwrap()).ok();
    }

    #[test]
    fn snapshot_then_apply_round_trips_every_setting() {
        let path = format!("{}/neural-forge-persist-test-{}/shm.bin", std::env::temp_dir().display(), std::process::id());
        let m = mapping::open_at(&path).expect("failed to create test mapping");
        let h = m.header();

        h.intensity_bits.store(1.75f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        h.style.store(2, std::sync::atomic::Ordering::Relaxed);
        h.auto_mask.store(0, std::sync::atomic::Ordering::Relaxed);

        h.white_point_source.store(1, std::sync::atomic::Ordering::Relaxed);
        h.white_point_bits.store(2.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        h.white_point_scale_bits.store(1.5f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        h.white_point_trim_bits.store(0.75f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
        h.toggle_key.store(87, std::sync::atomic::Ordering::Relaxed);
        let snap = snapshot(h);
        assert_eq!(snap.get("set_intensity").map(String::as_str), Some("1.75"));
        assert_eq!(snap.get("set_style").map(String::as_str), Some("2"));
        assert_eq!(snap.get("set_auto_mask").map(String::as_str), Some("0"));

        h.init_defaults();
        assert_ne!(f32::from_bits(h.intensity_bits.load(std::sync::atomic::Ordering::Relaxed)), 1.75);

        apply(h, &snap);
        assert_eq!(snapshot(h),snap,"every setting, including HDR and hotkey, must survive reinitialization");
        assert_eq!(f32::from_bits(h.intensity_bits.load(std::sync::atomic::Ordering::Relaxed)), 1.75);
        assert_eq!(h.style.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(h.auto_mask.load(std::sync::atomic::Ordering::Relaxed), 0);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(std::path::Path::new(&path).parent().unwrap()).ok();
    }

    #[test]
    fn apply_ignores_unknown_and_unparsable_keys() {
        let path = format!("{}/neural-forge-persist-test2-{}/shm.bin", std::env::temp_dir().display(), std::process::id());
        let m = mapping::open_at(&path).expect("failed to create test mapping");
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
