//! Named strength presets: values for the model's four knobs (intensity, local tone, local
//! structure, skin structure). Not a second settings mechanism: the GUI's preset menu writes these
//! four values through the same setters as their own rows, so they are saved like any other change
//! (profiles stay the way to keep a whole set of settings).

/// One preset: its name and (intensity, local_tone, local_structure, skin_structure).
pub struct Preset {
    pub name: &'static str,
    pub values: [f32; 4],
}

/// The settings the presets set, in [`Preset::values`]' order (`ShmHeader::persisted_settings` names).
pub const KNOBS: [&str; 4] = ["intensity", "local_tone", "local_structure", "skin_structure"];

/// Reference is the defaults (`ShmHeader::init_defaults`): a skin structure of -1 follows local structure.
pub const PRESETS: [Preset; 4] = [
    Preset { name: "Light", values: [0.45, 0.55, 0.25, 0.15] },
    Preset { name: "Moderate", values: [0.70, 0.80, 0.55, 0.40] },
    Preset { name: "Reference", values: [1.00, 1.00, 1.00, -1.00] },
    Preset { name: "Overdrive", values: [1.30, 1.25, 1.45, 1.20] },
];

/// The preset `values` are, if any (to within the spin rows' 0.01 resolution).
pub fn matching(values: [f32; 4]) -> Option<usize> {
    PRESETS.iter().position(|p| p.values.iter().zip(values).all(|(a, b)| (a - b).abs() < 0.005))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn knobs(header: &neural_forge_protocol::ShmHeader) -> [f32; 4] {
        KNOBS.map(|name| {
            let (_, is_float, bits) = header.persisted_settings().into_iter().find(|(n, ..)| *n == name).unwrap_or_else(|| panic!("{name} is a setting"));
            assert!(is_float, "{name}");
            f32::from_bits(bits)
        })
    }

    #[test]
    fn reference_is_the_defaults() {
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        assert_eq!(knobs(&header), PRESETS[2].values);
        assert_eq!(matching(knobs(&header)), Some(2));
    }

    #[test]
    fn every_preset_is_inside_the_settings_bounds_and_matches_itself() {
        for (i, preset) in PRESETS.iter().enumerate() {
            for (name, v) in KNOBS.iter().zip(preset.values) {
                let (min, max) = neural_forge_protocol::setting_bounds(name).unwrap();
                assert!((min..=max).contains(&v), "{} {name}={v}", preset.name);
            }
            assert_eq!(matching(preset.values), Some(i));
        }
    }

    #[test]
    fn values_off_every_preset_match_none() {
        assert_eq!(matching([0.45, 0.55, 0.25, 0.16]), None);
        assert_eq!(matching([1.0, 1.0, 1.0, 1.0]), None, "a skin structure of 1 is not 'follow local structure'");
        // The f32 a spin row stores for 0.70 still matches.
        assert_eq!(matching([0.7f64 as f32, 0.8, 0.55, 0.4]), Some(1));
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        header.intensity_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        assert_eq!(matching(knobs(&header)), None);
    }
}
