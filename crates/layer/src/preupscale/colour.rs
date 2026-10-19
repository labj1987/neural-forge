//! The colour of the network's edit (docs/NATIVE_BACKEND.md, "The colour of the edit"): what the two
//! paths' shaders are told (`native_colour.comp` before the upscaler, `native_post_composite.comp` after
//! it), and the look-up table both read, loaded from the `.cube` file `edit_lut` in `config.ini` names.
//!
//! The table cannot travel through the shared memory sensibly (65^3 entries are 4.4 MB, and the header is a
//! fixed 64 KiB of settings), so the layer reads the file itself, on a thread of its own, whenever
//! `edit_lut_seq` moves on while the table's strength is above 0; until it is read the previous table (or
//! none) stays in use.

use std::sync::{Arc, Mutex};

use neural_forge_protocol::cube::Cube;
use neural_forge_protocol::{ColourSettings, EditColour};

/// The shaders' step bits (`p.colour_flags.x`).
pub(crate) const STEP: u32 = 1;
pub(crate) const GUARD: u32 = 2;
pub(crate) const LUT: u32 = 4;
pub(crate) const DETAIL: u32 = 8;
pub(crate) const DITHER: u32 = 16;

/// A loaded table, numbered so a path's GPU copy knows when to upload it again.
pub(crate) struct Lut {
    pub generation: u64,
    pub cube: Cube,
}

impl Lut {
    /// The entries as the shaders read them (vec4 per entry, red fastest).
    pub(crate) fn entries(&self) -> Vec<[f32; 4]> {
        self.cube.table.iter().map(|e| [e[0], e[1], e[2], 0.0]).collect()
    }

    /// Per channel, the scale and offset that take a value to the table's grid: (size - 1) / (max - min)
    /// and -min times that.
    pub(crate) fn grid(&self) -> ([f32; 3], [f32; 3]) {
        let n = (self.cube.size - 1) as f32;
        let scale: [f32; 3] = std::array::from_fn(|c| n / (self.cube.domain_max[c] - self.cube.domain_min[c]));
        let offset: [f32; 3] = std::array::from_fn(|c| -self.cube.domain_min[c] * scale[c]);
        (scale, offset)
    }
}

/// One pass's shader parameters: `colour` (saturation, contrast, detail, 0), `flags` (steps, table size, 0,
/// 0), `lut_scale` (grid scale, strength) and `lut_offset` (grid offset, adapt).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PassColour {
    pub colour: [f32; 4],
    pub flags: [u32; 4],
    pub lut_scale: [f32; 4],
    pub lut_offset: [f32; 4],
}

impl PassColour {
    /// Pass `index`'s parameters when `last` is the last pass the frame runs; `lut` the table the path has
    /// on the GPU; `dither` only after the upscaler (the answer before it is half floats, not 8 bits).
    pub(crate) fn of(s: &ColourSettings, index: usize, last: bool, lut: Option<&Lut>, dither: bool) -> Self {
        let c = s.each[index.min(s.each.len() - 1)];
        let lut = lut.filter(|_| last && s.lut_strength > 0.0);
        let mut flags = 0;
        if c.saturation != 1.0 || c.contrast != 1.0 {
            flags |= STEP;
        }
        if c.detail != 1.0 {
            flags |= STEP | DETAIL;
        }
        if lut.is_some() {
            flags |= STEP | LUT;
        }
        if flags & STEP != 0 && s.gamut_guard {
            flags |= GUARD;
        }
        if dither && s.dither {
            flags |= DITHER;
        }
        let (scale, offset) = lut.map_or(([0.0; 3], [0.0; 3]), Lut::grid);
        Self {
            colour: [c.saturation, c.contrast, c.detail, 0.0],
            flags: [flags, lut.map_or(0, |l| l.cube.size as u32), 0, 0],
            lut_scale: [scale[0], scale[1], scale[2], if lut.is_some() { s.lut_strength } else { 0.0 }],
            lut_offset: [offset[0], offset[1], offset[2], s.lut_adapt.max(1e-3)],
        }
    }

    /// Whether the pass's shader has anything to do (a neutral pass is not touched at all).
    #[cfg(test)]
    pub(crate) fn any(&self) -> bool {
        self.flags[0] & (STEP | DITHER) != 0
    }
}

/// Whether any of the frame's first `count` passes reshapes its edit (or the table applies on the last).
pub(crate) fn wanted(s: &ColourSettings, count: usize, lut: bool) -> bool {
    s.each.iter().take(count.max(1)).any(|c| !EditColour::is_neutral(c)) || (lut && s.lut_strength > 0.0)
}

#[derive(Default)]
struct Store {
    /// The `edit_lut_seq` last read for (`None`: never).
    seen: Option<u32>,
    loading: bool,
    lut: Option<Arc<Lut>>,
    generation: u64,
}

static STORE: Mutex<Store> = Mutex::new(Store { seen: None, loading: false, lut: None, generation: 0 });

/// The table to use: `None` while the strength is 0 (nothing is read) or no file is named or it could not
/// be read. A new `seq` starts reading the file again in the background.
pub(crate) fn current(s: &ColourSettings) -> Option<Arc<Lut>> {
    if s.lut_strength <= 0.0 {
        return None;
    }
    let mut store = STORE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if store.seen != Some(s.lut_seq) && !store.loading {
        store.seen = Some(s.lut_seq);
        store.loading = true;
        let started = std::thread::Builder::new().name("nf-colour-lut".into()).spawn(|| {
            let config = neural_forge_protocol::persist::read_config(&neural_forge_protocol::persist::config_file());
            let path = config.get(neural_forge_protocol::cube::CONFIG_KEY).cloned().unwrap_or_default();
            let cube = if path.is_empty() {
                crate::log!("[colour] no look-up table named (edit_lut in config.ini)");
                None
            } else {
                match neural_forge_protocol::cube::load(&path) {
                    Ok(c) => {
                        crate::log!("[colour] look-up table {path}: {} points a side", c.size);
                        Some(c)
                    }
                    Err(e) => {
                        crate::log!("[colour] the look-up table could not be read: {e}");
                        None
                    }
                }
            };
            let mut store = STORE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            store.generation += 1;
            let generation = store.generation;
            store.lut = cube.map(|cube| Arc::new(Lut { generation, cube }));
            store.loading = false;
        });
        if started.is_err() {
            store.loading = false;
        }
    }
    store.lut.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_neutral_pass_has_nothing_to_do_and_each_value_turns_on_its_step() {
        let s = ColourSettings::DEFAULT;
        let lut = Lut { generation: 1, cube: Cube::identity(3) };
        assert!(!PassColour::of(&s, 0, true, Some(&lut), true).any(), "the defaults leave the edit alone");
        assert!(!wanted(&s, 10, true));
        let mut t = s;
        t.each[1].saturation = 0.5;
        assert!(!PassColour::of(&t, 0, false, None, false).any(), "pass 1 is still neutral");
        assert_eq!(PassColour::of(&t, 1, false, None, false).flags[0], STEP | GUARD);
        assert!(wanted(&t, 2, false) && !wanted(&t, 1, false), "only the passes the frame runs count");
        t.each[1].detail = 0.0;
        t.gamut_guard = false;
        assert_eq!(PassColour::of(&t, 1, false, None, false).flags[0], STEP | DETAIL);
        let mut l = s;
        l.lut_strength = 0.5;
        assert_eq!(PassColour::of(&l, 0, false, Some(&lut), false).flags[0], 0, "the table is the last pass's");
        let last = PassColour::of(&l, 0, true, Some(&lut), false);
        assert_eq!((last.flags[0], last.flags[1], last.lut_scale[3]), (STEP | LUT | GUARD, 3, 0.5));
        assert_eq!((last.lut_scale[0], last.lut_offset[0]), (2.0, 0.0), "a 0-1 domain on a 3-point grid");
        let mut d = s;
        d.dither = true;
        assert_eq!(PassColour::of(&d, 0, true, None, false).flags[0], 0, "no dither before the upscaler");
        assert_eq!(PassColour::of(&d, 0, true, None, true).flags[0], DITHER);
    }
}
