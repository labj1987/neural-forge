//! Named strength presets: values for the model's four knobs (intensity, local tone, local
//! structure, skin structure). Not a second settings mechanism: a preset is written to the header as
//! one unit (`ShmHeader::write_settings`, so the layer never sees half of one) and saved like any
//! other change (profiles stay the way to keep a whole set of settings).

/// The four knobs, by name. [`Knobs::settings`] pairs each with its setting name ([`KNOBS`]), so no
/// caller depends on an order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knobs {
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// -1 follows local structure.
    pub skin_structure: f32,
}

/// The settings the presets set (`ShmHeader::persisted_settings` names), as [`Knobs::settings`]
/// names them.
pub const KNOBS: [&str; 4] = ["intensity", "local_tone", "local_structure", "skin_structure"];

impl Knobs {
    /// Each knob with its setting name.
    pub fn settings(&self) -> [(&'static str, f32); 4] {
        [(KNOBS[0], self.intensity), (KNOBS[1], self.local_tone), (KNOBS[2], self.local_structure), (KNOBS[3], self.skin_structure)]
    }

    /// The knobs as the header holds them now (read whole, `ShmHeader::global_tuning`).
    pub fn from_header(header: &neural_forge_protocol::ShmHeader) -> Knobs {
        let t = header.global_tuning();
        Knobs { intensity: t.intensity, local_tone: t.local_tone, local_structure: t.local_structure, skin_structure: t.skin_structure }
    }

    /// Writes the four values to `header` as one unit; whether all were stored.
    pub fn write(&self, header: &neural_forge_protocol::ShmHeader) -> bool {
        let settings = self.settings().map(|(name, v)| (name, v.to_bits()));
        header.write_settings(&settings)
    }

    /// Whether `self` and `other` agree to within the spin rows' 0.01 resolution.
    fn close_to(&self, other: &Knobs) -> bool {
        self.settings().iter().zip(other.settings()).all(|((_, a), (_, b))| (a - b).abs() < 0.005)
    }
}

/// One preset: its name and its knobs.
pub struct Preset {
    pub name: &'static str,
    pub knobs: Knobs,
}

const fn knobs(intensity: f32, local_tone: f32, local_structure: f32, skin_structure: f32) -> Knobs {
    Knobs { intensity, local_tone, local_structure, skin_structure }
}

/// Reference is the defaults (`ShmHeader::init_defaults`): a skin structure of -1 follows local structure.
pub const PRESETS: [Preset; 4] = [
    Preset { name: "Light", knobs: knobs(0.45, 0.55, 0.25, 0.15) },
    Preset { name: "Moderate", knobs: knobs(0.70, 0.80, 0.55, 0.40) },
    Preset { name: "Reference", knobs: knobs(1.00, 1.00, 1.00, -1.00) },
    Preset { name: "Overdrive", knobs: knobs(1.30, 1.25, 1.45, 1.20) },
];

/// The preset `values` are, if any (to within the spin rows' 0.01 resolution).
pub fn matching(values: &Knobs) -> Option<usize> {
    PRESETS.iter().position(|p| p.knobs.close_to(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn the_knobs_are_the_settings_they_name() {
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        let k = knobs(0.25, 0.5, 0.75, 1.5);
        assert!(k.write(&header));
        for (name, v) in k.settings() {
            let (_, is_float, bits) = header.persisted_settings().into_iter().find(|(n, ..)| *n == name).unwrap_or_else(|| panic!("{name} is a setting"));
            assert!(is_float, "{name}");
            assert_eq!(f32::from_bits(bits), v, "{name}");
        }
        assert_eq!(Knobs::from_header(&header), k);
        assert_eq!(k.settings().map(|(n, _)| n), KNOBS);
    }

    #[test]
    fn reference_is_the_defaults() {
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        assert_eq!(Knobs::from_header(&header), PRESETS[2].knobs);
        assert_eq!(matching(&Knobs::from_header(&header)), Some(2));
    }

    #[test]
    fn every_preset_is_inside_the_settings_bounds_and_matches_itself() {
        for (i, preset) in PRESETS.iter().enumerate() {
            for (name, v) in preset.knobs.settings() {
                let (min, max) = neural_forge_protocol::setting_bounds(name).unwrap();
                assert!((min..=max).contains(&v), "{} {name}={v}", preset.name);
            }
            assert_eq!(matching(&preset.knobs), Some(i));
        }
    }

    #[test]
    fn values_off_every_preset_match_none() {
        assert_eq!(matching(&knobs(0.45, 0.55, 0.25, 0.16)), None);
        assert_eq!(matching(&knobs(1.0, 1.0, 1.0, 1.0)), None, "a skin structure of 1 is not 'follow local structure'");
        // The f32 a spin row stores for 0.70 still matches.
        assert_eq!(matching(&knobs(0.7f64 as f32, 0.8, 0.55, 0.4)), Some(1));
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        header.intensity_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        assert_eq!(matching(&Knobs::from_header(&header)), None);
    }
}
