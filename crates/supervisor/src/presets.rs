//! Named strength presets: values for the model's four knobs (intensity, local tone, local
//! structure, skin structure), for every pass: a preset also ends each pass's own values for the four
//! ([`STRENGTH_OVERRIDES`]), so no pass stays on other strengths. Reference is Neural Forge's reference look, so it
//! also puts the style and the colour of the edit back to their defaults ([`LOOK`], [`LOOK_OVERRIDES`]): a grade
//! left on under it is not the reference picture (#63). Not a second settings mechanism: a preset is written to
//! the header as one unit (`ShmHeader::write_settings`, so the layer never sees half of one) and saved like any
//! other change (profiles stay the way to keep a whole set of settings).

use neural_forge_protocol::enums::pass_override;
use std::sync::atomic::Ordering;

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

/// The look a preset with [`Preset::resets_look`] puts back, each setting with its default
/// (`ShmHeader::init_defaults`): the style, the colour of the edit (saturation, contrast, detail, its protections,
/// the gamut guard) and the LUT's strength.
pub const LOOK: [(&str, Value); 10] = [
    ("style", Value::Whole(0)),
    ("edit_saturation", Value::Float(1.0)),
    ("edit_contrast", Value::Float(1.0)),
    ("edit_detail", Value::Float(1.0)),
    ("edit_gamut_guard", Value::Whole(1)),
    ("edit_lut_strength", Value::Float(0.0)),
    ("edit_hue_protection", Value::Float(0.0)),
    ("edit_neutral_protection", Value::Float(0.0)),
    ("edit_shadow_protection", Value::Float(0.0)),
    ("edit_highlight_protection", Value::Float(0.0)),
];

/// A setting's value, as the header stores it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Whole(u32),
    Float(f32),
}

impl Value {
    fn bits(self) -> u32 {
        match self {
            Value::Whole(v) => v,
            Value::Float(v) => v.to_bits(),
        }
    }

    /// Whether a stored `bits` is this value (a float to within the spin rows' 0.01 resolution).
    fn is(self, bits: u32) -> bool {
        match self {
            Value::Whole(v) => v == bits,
            Value::Float(v) => (f32::from_bits(bits) - v).abs() < 0.005,
        }
    }
}

/// A pass's own values every preset ends (`pass<N>_overrides` bits): the four knobs.
pub const STRENGTH_OVERRIDES: u32 = pass_override::INTENSITY | pass_override::LOCAL_TONE | pass_override::LOCAL_STRUCTURE | pass_override::SKIN_STRUCTURE;
/// A pass's own values a preset that resets the look ends as well: its style and the colour of its edit.
pub const LOOK_OVERRIDES: u32 = pass_override::STYLE | pass_override::EDIT_SATURATION | pass_override::EDIT_CONTRAST | pass_override::EDIT_DETAIL;

/// One preset: its name, its knobs, and whether it also puts the look back ([`LOOK`]).
pub struct Preset {
    pub name: &'static str,
    pub knobs: Knobs,
    pub resets_look: bool,
}

impl Preset {
    /// The bits of a pass's `overrides` this preset ends.
    pub fn ended_overrides(&self) -> u32 {
        if self.resets_look {
            STRENGTH_OVERRIDES | LOOK_OVERRIDES
        } else {
            STRENGTH_OVERRIDES
        }
    }

    /// Everything the preset writes to `header`, by setting name with its bits: the four knobs, each pass's
    /// `overrides` without the ones it ends (only the passes that have one), and the look where it resets it.
    pub fn settings(&self, header: &neural_forge_protocol::ShmHeader) -> Vec<(&'static str, u32)> {
        let mut out: Vec<(&'static str, u32)> = self.knobs.settings().iter().map(|&(name, v)| (name, v.to_bits())).collect();
        if self.resets_look {
            out.extend(LOOK.iter().map(|&(name, v)| (name, v.bits())));
        }
        let ended = self.ended_overrides();
        for (k, pass) in header.pass.iter().enumerate() {
            let mask = pass.overrides.load(Ordering::Relaxed);
            if mask & ended != 0 {
                let name = neural_forge_protocol::pass_setting(k, "overrides").expect("every pass has its overrides setting");
                out.push((name, mask & !ended));
            }
        }
        out
    }

    /// Writes the preset to `header` as one unit; the names of the settings it wrote (for saving them), or `None`
    /// when one could not be stored.
    pub fn write(&self, header: &neural_forge_protocol::ShmHeader) -> Option<Vec<&'static str>> {
        let settings = self.settings(header);
        header.write_settings(&settings).then(|| settings.iter().map(|&(name, _)| name).collect())
    }

    /// Whether `header` holds this preset: its knobs, no pass on its own values for what the preset sets, and the
    /// look at its defaults where the preset resets it.
    pub fn matches(&self, header: &neural_forge_protocol::ShmHeader) -> bool {
        let ended = self.ended_overrides();
        self.knobs.close_to(&Knobs::from_header(header))
            && header.pass.iter().all(|p| p.overrides.load(Ordering::Relaxed) & ended == 0)
            && (!self.resets_look || {
                let stored = header.persisted_settings();
                LOOK.iter().all(|&(name, v)| stored.iter().any(|&(n, _, bits)| n == name && v.is(bits)))
            })
    }
}

const fn knobs(intensity: f32, local_tone: f32, local_structure: f32, skin_structure: f32) -> Knobs {
    Knobs { intensity, local_tone, local_structure, skin_structure }
}

/// Reference is the defaults (`ShmHeader::init_defaults`): a skin structure of -1 follows local structure, and the
/// look goes back to its defaults with it. Remix is RTX Remix's defaults for NVIDIA's neural rendering
/// (`rtx_dlss_neural_rendering.h`: intensity 1, structure 0.7, tone 0.3, character structure 0.5 with its auto mask
/// on). The others set strengths only: a style or a grade chosen with them stays.
pub const PRESETS: [Preset; 5] = [
    Preset { name: "Light", knobs: knobs(0.45, 0.55, 0.25, 0.15), resets_look: false },
    Preset { name: "Moderate", knobs: knobs(0.70, 0.80, 0.55, 0.40), resets_look: false },
    Preset { name: "Reference", knobs: knobs(1.00, 1.00, 1.00, -1.00), resets_look: true },
    Preset { name: "Overdrive", knobs: knobs(1.30, 1.25, 1.45, 1.20), resets_look: false },
    Preset { name: "Remix", knobs: knobs(1.00, 0.30, 0.70, 0.50), resets_look: false },
];

/// The preset `header` holds, if any ([`Preset::matches`]; the knobs to within the spin rows' 0.01 resolution).
/// Where the knobs are Reference's and the look is not, none is: choosing Reference then puts the look back.
pub fn matching(header: &neural_forge_protocol::ShmHeader) -> Option<usize> {
    PRESETS.iter().position(|p| p.matches(header))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(matching(&header), Some(2));
        // The look it puts back is the defaults' too, every setting of it.
        let stored = header.persisted_settings();
        for (name, v) in LOOK {
            let (_, _, bits) = stored.iter().find(|(n, ..)| *n == name).unwrap_or_else(|| panic!("{name} is a setting"));
            assert!(v.is(*bits), "{name} defaults to {:?}", f32::from_bits(*bits));
        }
    }

    #[test]
    fn remix_is_rtx_remixs_defaults() {
        let remix = PRESETS.iter().find(|p| p.name == "Remix").expect("a Remix preset");
        assert_eq!(remix.knobs, knobs(1.0, 0.3, 0.7, 0.5));
        assert_eq!(PRESETS.iter().position(|p| p.name == "Reference"), Some(2), "the default preset stays Reference");
    }

    #[test]
    fn every_preset_is_inside_the_settings_bounds_and_matches_itself() {
        for (i, preset) in PRESETS.iter().enumerate() {
            for (name, v) in preset.knobs.settings() {
                let (min, max) = neural_forge_protocol::setting_bounds(name).unwrap();
                assert!((min..=max).contains(&v), "{} {name}={v}", preset.name);
            }
            let header = neural_forge_protocol::ShmHeader::default();
            header.init_defaults();
            assert!(preset.write(&header).is_some());
            assert_eq!(matching(&header), Some(i));
        }
    }

    /// A header with a grade on (#63's settings): saturation 2.05, style 1, pass 2 on its own strengths, style and
    /// colour, with its own model resolution and placement.
    fn graded() -> neural_forge_protocol::ShmHeader {
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        let own = STRENGTH_OVERRIDES | LOOK_OVERRIDES | pass_override::SCALE | pass_override::PLACEMENT | pass_override::AUTO_MASK;
        assert!(header.write_settings(&[
            ("edit_saturation", 2.05f32.to_bits()),
            ("edit_contrast", 0.8f32.to_bits()),
            ("edit_detail", 1.4f32.to_bits()),
            ("edit_hue_protection", 0.5f32.to_bits()),
            ("edit_lut_strength", 1.0f32.to_bits()),
            ("style", 1),
            ("local_structure", 0.25f32.to_bits()),
            ("passes", 2),
            ("pass2_overrides", own),
            ("pass2_style", 2),
            ("pass2_edit_saturation", 0.85f32.to_bits()),
            ("pass2_local_tone", 1.25f32.to_bits()),
            ("pass3_overrides", pass_override::EDIT_CONTRAST | pass_override::LOCAL_STRUCTURE),
        ]));
        header
    }

    #[test]
    fn reference_puts_the_look_back_in_every_pass() {
        let header = graded();
        // The four knobs back at Reference's by hand: still not Reference while the grade is on.
        assert!(PRESETS[2].knobs.write(&header));
        assert_eq!(matching(&header), None, "a grade under Reference's strengths is not Reference");
        let wrote = PRESETS[2].write(&header).expect("stored");
        assert_eq!(matching(&header), Some(2));
        for name in ["style", "edit_saturation", "edit_hue_protection", "pass2_overrides", "pass3_overrides", "intensity"] {
            assert!(wrote.contains(&name), "{name} is saved with the preset");
        }
        assert!(!wrote.contains(&"pass1_overrides"), "a pass with nothing of its own is not written");
        let fresh = neural_forge_protocol::ShmHeader::default();
        fresh.init_defaults();
        let t = header.pass_tunings();
        assert_eq!(t.tunings[..2], [fresh.global_tuning(); 2], "both passes run the defaults");
        let stored = header.persisted_settings();
        let bits = |name: &str| stored.iter().find(|(n, ..)| *n == name).unwrap().2;
        assert_eq!(f32::from_bits(bits("edit_saturation")), 1.0);
        assert_eq!(bits("style"), 0);
        // What is not the look stays the pass's own: its model resolution, placement and auto mask.
        assert_eq!(bits("pass2_overrides"), pass_override::SCALE | pass_override::PLACEMENT | pass_override::AUTO_MASK);
        assert_eq!(bits("pass3_overrides"), 0);
        assert_eq!(bits("passes"), 2);
    }

    #[test]
    fn a_strength_preset_reaches_every_pass_and_keeps_the_look() {
        let header = graded();
        assert!(PRESETS[0].write(&header).is_some());
        assert_eq!(matching(&header), Some(0));
        let stored = header.persisted_settings();
        let bits = |name: &str| stored.iter().find(|(n, ..)| *n == name).unwrap().2;
        assert_eq!(f32::from_bits(bits("edit_saturation")), 2.05, "the grade stays");
        assert_eq!(bits("style"), 1);
        assert_eq!(bits("pass2_overrides") & STRENGTH_OVERRIDES, 0, "no pass keeps its own strengths");
        assert_eq!(bits("pass2_overrides") & LOOK_OVERRIDES, LOOK_OVERRIDES, "its own style and colour stay");
        let t = header.pass_tunings();
        assert_eq!(t.tunings[1].local_tone, PRESETS[0].knobs.local_tone);
    }

    #[test]
    fn values_off_every_preset_match_none() {
        let holds = |k: Knobs| {
            let header = neural_forge_protocol::ShmHeader::default();
            header.init_defaults();
            assert!(k.write(&header));
            matching(&header)
        };
        assert_eq!(holds(knobs(0.45, 0.55, 0.25, 0.16)), None);
        assert_eq!(holds(knobs(1.0, 1.0, 1.0, 1.0)), None, "a skin structure of 1 is not 'follow local structure'");
        // The f32 a spin row stores for 0.70 still matches.
        assert_eq!(holds(knobs(0.7f64 as f32, 0.8, 0.55, 0.4)), Some(1));
        assert_eq!(holds(knobs(0.5, 1.0, 1.0, -1.0)), None);
    }
}
