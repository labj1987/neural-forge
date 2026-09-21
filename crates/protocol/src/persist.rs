//! Round-trips [`crate::ShmHeader::persisted_settings`] through the plain
//! `set_<name>=<value>` string pairs `config.ini` actually stores, so a GUI/CLI
//! doesn't need to know `f32::to_bits()` exists -- it just hands this module a
//! `BTreeMap` it already has (from parsing/about to write the file) and a header.

use std::collections::BTreeMap;

use crate::ShmHeader;

/// Every persisted setting's current value, as `set_<name>` -> string, ready to
/// merge into whatever else `config.ini` is about to write out.
pub fn snapshot(header: &ShmHeader) -> BTreeMap<String, String> {
    header
        .persisted_settings()
        .into_iter()
        .map(|(name, is_float, bits)| {
            let value = if is_float { f32::from_bits(bits).to_string() } else { bits.to_string() };
            (format!("set_{name}"), value)
        })
        .collect()
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
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::mapping;

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
