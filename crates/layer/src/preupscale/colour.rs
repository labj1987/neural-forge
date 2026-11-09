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
/// The Oklab protections of the pass's result against its input (`protect`).
pub(crate) const PROTECT: u32 = 32;

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
    /// and -min times that. `None` when one is not finite: a DOMAIN span so small (a subnormal one) that the
    /// scale overflows, and its offset is NaN. The shaders would index the table with that (out of bounds
    /// without robust buffer access), so such a table is never used ([`grid_of`]).
    pub(crate) fn grid(&self) -> Option<([f32; 3], [f32; 3])> {
        grid_of(&self.cube)
    }
}

/// [`Lut::grid`] of `cube`.
fn grid_of(cube: &Cube) -> Option<([f32; 3], [f32; 3])> {
    let n = (cube.size - 1) as f32;
    let scale: [f32; 3] = std::array::from_fn(|c| n / (cube.domain_max[c] - cube.domain_min[c]));
    let offset: [f32; 3] = std::array::from_fn(|c| -cube.domain_min[c] * scale[c]);
    scale.iter().chain(&offset).all(|v| v.is_finite()).then_some((scale, offset))
}

/// One pass's shader parameters: `colour` (saturation, contrast, detail, 0), `flags` (steps, table size, 0,
/// 0), `lut_scale` (grid scale, strength), `lut_offset` (grid offset, adapt) and `protect` (the hue, near-grey
/// chroma, shadow and highlight protections, 0-1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PassColour {
    pub colour: [f32; 4],
    pub flags: [u32; 4],
    pub lut_scale: [f32; 4],
    pub lut_offset: [f32; 4],
    pub protect: [f32; 4],
}

impl PassColour {
    /// Pass `index`'s parameters when `last` is the last pass the frame runs; `lut` the table the path has
    /// on the GPU; `dither` only after the upscaler (the answer before it is half floats, not 8 bits).
    pub(crate) fn of(s: &ColourSettings, index: usize, last: bool, lut: Option<&Lut>, dither: bool) -> Self {
        let c = s.each[index.min(s.each.len() - 1)];
        // A table without a finite grid is never loaded (`load`); the check here keeps one from the shaders anyway.
        let grid = lut.filter(|_| last && s.lut_strength > 0.0).and_then(|l| Some((l, l.grid()?)));
        let lut = grid.map(|(l, _)| l);
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
        let protect = s.protect.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
        if protect.iter().any(|&v| v > 0.0) {
            flags |= STEP | PROTECT;
        }
        if flags & STEP != 0 && s.gamut_guard {
            flags |= GUARD;
        }
        if dither && s.dither {
            flags |= DITHER;
        }
        let (scale, offset) = grid.map_or(([0.0; 3], [0.0; 3]), |(_, g)| g);
        Self {
            colour: [c.saturation, c.contrast, c.detail, 0.0],
            flags: [flags, lut.map_or(0, |l| l.cube.size as u32), 0, 0],
            lut_scale: [scale[0], scale[1], scale[2], if lut.is_some() { s.lut_strength } else { 0.0 }],
            lut_offset: [offset[0], offset[1], offset[2], s.lut_adapt.max(1e-3)],
            protect,
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
    s.each.iter().take(count.max(1)).any(|c| !EditColour::is_neutral(c)) || (lut && s.lut_strength > 0.0) || s.protect.iter().any(|&v| v.is_finite() && v > 0.0)
}

#[derive(Default)]
struct Store {
    /// The `edit_lut_seq` last read for (`None`: never).
    seen: Option<u32>,
    loading: bool,
    lut: Option<Arc<Lut>>,
    generation: u64,
    /// The `edit_lut_seq` whose loader thread could not be started, logged.
    failed: Option<u32>,
}

static STORE: Mutex<Store> = Mutex::new(Store { seen: None, loading: false, lut: None, generation: 0, failed: None });

/// The table to use: `None` while the strength is 0 (nothing is read) or no file is named or it could not
/// be read. A new `seq` starts reading the file again in the background.
pub(crate) fn current(s: &ColourSettings) -> Option<Arc<Lut>> {
    if s.lut_strength <= 0.0 {
        return None;
    }
    let mut store = STORE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    request(&mut store, s.lut_seq, || std::thread::Builder::new().name("nf-colour-lut".into()).spawn(load).map(drop));
    store.lut.clone()
}

/// Starts reading the table for `seq` with `spawn` unless it was read for it or is being read. `seq` counts
/// as read only once the reading started: a thread that could not be started is tried again at the next call
/// (logged once for the seq), where taking it as read kept the previous table until `seq` moved on.
fn request(store: &mut Store, seq: u32, spawn: impl FnOnce() -> std::io::Result<()>) {
    if store.seen == Some(seq) || store.loading {
        return;
    }
    store.loading = true;
    match spawn() {
        Ok(()) => store.seen = Some(seq),
        Err(e) => {
            store.loading = false;
            if store.failed.replace(seq) != Some(seq) {
                crate::log!("[colour] the look-up table's loader thread could not be started ({e}); tried again at the next frame");
            }
        }
    }
}

/// The loader thread: reads the file `edit_lut` names into [`STORE`].
fn load() {
    let config = neural_forge_protocol::persist::read_config(&neural_forge_protocol::persist::config_file());
    let path = config.get(neural_forge_protocol::cube::CONFIG_KEY).cloned().unwrap_or_default();
    let cube = if path.is_empty() {
        crate::log!("[colour] no look-up table named (edit_lut in config.ini)");
        None
    } else {
        table_of(&path, neural_forge_protocol::cube::load(&path))
    };
    let mut store = STORE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    store.generation += 1;
    let generation = store.generation;
    store.lut = cube.map(|cube| Arc::new(Lut { generation, cube }));
    store.loading = false;
}

/// The table `load` keeps of what reading `path` gave: none when it could not be read or parsed, or when its
/// grid is not finite ([`Lut::grid`]), each logged.
fn table_of(path: &str, loaded: Result<Cube, String>) -> Option<Cube> {
    match loaded {
        Ok(c) if grid_of(&c).is_none() => {
            crate::log!(
                "[colour] the look-up table {path} cannot be used: its DOMAIN_MIN to DOMAIN_MAX span ({:?} to {:?}) is too small for its {} points a side",
                c.domain_min,
                c.domain_max,
                c.size
            );
            None
        }
        Ok(c) => {
            crate::log!("[colour] look-up table {path}: {} points a side", c.size);
            Some(c)
        }
        Err(e) => {
            crate::log!("[colour] the look-up table could not be read: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loader thread that could not be started leaves the seq unread: the next call starts one, where the
    /// table chosen was otherwise never loaded until the seq moved on. One reading at a time, once per seq.
    #[test]
    fn a_seq_counts_as_read_only_once_its_loader_started() {
        let mut store = Store::default();
        request(&mut store, 4, || Err(std::io::Error::other("no more threads")));
        assert_eq!((store.seen, store.loading), (None, false), "not started: still to read");
        let mut started = 0;
        request(&mut store, 4, || {
            started += 1;
            Ok(())
        });
        assert_eq!((started, store.seen, store.loading), (1, Some(4), true), "started at the next call");
        request(&mut store, 5, || unreachable!("one reading at a time"));
        store.loading = false;
        request(&mut store, 4, || unreachable!("read for this seq already"));
        request(&mut store, 5, || Ok(()));
        assert_eq!(store.seen, Some(5), "a new seq reads again");
    }

    /// A `.cube` whose DOMAIN span is subnormal (the parser takes any max above min) has an infinite grid scale
    /// and a NaN offset, which the shaders would turn into an index outside the table: it is not loaded, and a
    /// table with such a grid gets no LUT step even if it were.
    #[test]
    fn a_table_whose_grid_is_not_finite_is_never_used() {
        let mut cube = Cube::identity(3);
        cube.domain_max[1] = 1e-40;
        cube.domain_min[1] = 0.0;
        let lut = Lut { generation: 1, cube: cube.clone() };
        assert_eq!(lut.grid(), None);
        let s = ColourSettings { lut_strength: 1.0, ..ColourSettings::DEFAULT };
        let pass = PassColour::of(&s, 0, true, Some(&lut), false);
        assert_eq!(pass.flags[0] & LUT, 0, "no LUT step");
        assert!(pass.lut_scale.iter().chain(&pass.lut_offset).all(|v| v.is_finite()), "{pass:?}");
        assert!(table_of("/x.cube", Ok(cube)).is_none(), "not loaded");
        // A small but normal span is fine: a finite grid.
        let mut fine = Cube::identity(3);
        fine.domain_max = [1e-3; 3];
        assert!(Lut { generation: 2, cube: fine.clone() }.grid().is_some());
        assert!(table_of("/y.cube", Ok(fine)).is_some());
    }

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

    /// The protections apply to every pass the frame runs (each against its own input), turn on the guard with
    /// their step, and are clamped to 0-1 (a value that is not a number is off).
    #[test]
    fn the_protections_turn_on_their_step_on_every_pass() {
        let mut s = ColourSettings::DEFAULT;
        s.protect = [0.5, 0.0, 0.0, 0.0];
        assert!(wanted(&s, 1, false));
        for k in 0..3 {
            let c = PassColour::of(&s, k, k == 2, None, false);
            assert_eq!(c.flags[0], STEP | PROTECT | GUARD, "pass {k}");
            assert_eq!(c.protect, [0.5, 0.0, 0.0, 0.0]);
        }
        s.protect = [f32::NAN, 2.0, -1.0, 0.0];
        let c = PassColour::of(&s, 0, true, None, false);
        assert_eq!(c.protect, [0.0, 1.0, 0.0, 0.0]);
        s.protect = [f32::NAN, 0.0, -1.0, 0.0];
        assert!(!PassColour::of(&s, 0, true, None, false).any() && !wanted(&s, 1, false), "nothing left to do");
    }
}
