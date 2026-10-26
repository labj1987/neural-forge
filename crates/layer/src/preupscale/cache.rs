//! The pre-upscaler edit cache (docs/ARCHITECTURE.md 4.10): the network runs on every hold while the
//! camera is still and on one hold in [`MOVING_INTERVAL`] while it moves (a refresh, [`Regime`]),
//! and the holds between (cached holds) reuse its correction, carried along the game's motion and
//! multiplied onto their own fresh frame. `model_interval` is the after-the-upscaler path's setting
//! and is not read here; `NEURAL_FORGE_EDIT_CACHE=off` ([`interval`]) makes every hold a refresh.
//!
//! The GPU side is seven compute passes (`shaders/preupscale_cache_*.comp`), the step-for-step mirror
//! of `scripts/cache_common.py` (the numpy reference the offline replay measures with):
//!
//! - in the capture (`C`) of every hold: `prep` (this hold's pseudo-linear depth, the depth's
//!   reversed-Z counts) and `motion` (the accumulated motion
//!   back to the previous network frame, which the network's history reads too, and the
//!   [`Regime`]'s counts). At interval 1 only
//!   `motion` runs, copying DLSS's vectors as they are, so the network's input is bit for bit what it
//!   was;
//! - in a refresh hold's write-back (`W`), right before the decode: `capture` (the edit the decode is
//!   about to write, against the frame, with this hold's depth and fingerprint);
//! - as a cached hold's whole write-back, in place of the network and the decode: `history` (the
//!   refresh's edit read at the chained position, depth-admitted, checked against this hold),
//!   `reduce` three times (the coverage-weighted pyramid), `apply` (band split), `guided` three
//!   times at half size (a guided filter of that edit on this hold's log luma), `apply` again (the
//!   filter blended in by how far each pixel moved, written back with the decode's rules).
//!
//! One [`CachePass`] per [`super::Resources`] (one extent), built on the first native hold. Above
//! interval 1 its buffers live in the network's input features ([`ScratchLayout`]), which only the
//! network's frame uses: the refresh's edit is written after that frame and read before the next
//! one's. It has none of its own beyond the counters.

use ash::vk;
use ash::vk::Handle;

use super::hdr;
use super::native::OwnBuffer;
use super::HostBuffer;

const PREP_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_prep.spv");
const MOTION_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_motion.spv");
const CAPTURE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_capture.spv");
const HISTORY_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_history.spv");
const REDUCE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_reduce.spv");
const APPLY_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_apply.spv");
const GUIDED_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_guided.spv");

const PREP: usize = 0;
const MOTION: usize = 1;
const CAPTURE: usize = 2;
const HISTORY: usize = 3;
const REDUCE: usize = 4;
const APPLY: usize = 5;
const GUIDED: usize = 6;
const SPV: [&[u8]; 7] = [PREP_SPV, MOTION_SPV, CAPTURE_SPV, HISTORY_SPV, REDUCE_SPV, APPLY_SPV, GUIDED_SPV];

/// The shaders' `F_*` flags.
pub(crate) const F_REVERSE_Z: u32 = 1;
pub(crate) const F_NEAREST: u32 = 2;
pub(crate) const F_FRESH: u32 = 4;
pub(crate) const F_CHAINED: u32 = 8;
pub(crate) const F_DEBUG_MASK: u32 = 32;
pub(crate) const F_KEEP_CLAMPED: u32 = 64;
pub(crate) const F_COUNT_DEPTH: u32 = 256;
/// The hold has no motion vectors (none copied: the motion buffer holds an older hold's): the motion
/// pass writes a field of NaN (no history) and counts nothing for the cadence.
pub(crate) const F_NO_MVEC: u32 = 512;
/// The capture decodes with the ACES transfer (`hdr::ACES`), its highlight recovery included.
pub(crate) const F_ACES: u32 = 1024;
/// Tests: the capture's answer image holds the decoded answer (the replay tool's dumps store it so).
#[cfg(test)]
const F_SCENE_ANSWER: u32 = 128;

/// The shaders' `STAT_*` words of the host-visible counters.
const STAT_Z0: usize = 0;
const STAT_Z1: usize = 1;
const STAT_DEPTH_N: usize = 2;
const STAT_TOP_SUM: usize = 3;
const STAT_TOP_N: usize = 4;
const STAT_BOTTOM_SUM: usize = 5;
const STAT_BOTTOM_N: usize = 6;
const STAT_REJECTED: usize = 7;
const STAT_EDIT_OK: usize = 8;
const STAT_APPLY_BAD: usize = 9;
const STAT_STILL: usize = 10;
const STAT_MOTION_N: usize = 11;
/// The high 32 bits of `STAT_TOP_SUM` and `STAT_BOTTOM_SUM` (a quarter of the rows of a 5120x2160 or
/// larger frame sums past 2^32).
const STAT_TOP_SUM_HI: usize = 12;
const STAT_BOTTOM_SUM_HI: usize = 13;
/// 1 while the last refresh's edit is stored and the holds since followed it without a gap (the shaders'
/// `STAT_EDIT_STORED`): written by the refresh's capture, cleared by the hold guard inside DLSS's buffer
/// (`preupscale_hold_guard.comp`), never with the counters ([`CLEARED_BYTES`]).
pub(crate) const STAT_EDIT_STORED: usize = 14;
/// The counters' bytes a hold clears: every word before [`STAT_EDIT_STORED`].
pub(crate) const CLEARED_BYTES: u64 = STAT_EDIT_STORED as u64 * 4;
pub(crate) const STAT_WORDS: usize = 16;
const STAT_BYTES: u64 = STAT_WORDS as u64 * 4;

/// The interval while the camera moves: the network on every second hold. 3 trails further (p99.5
/// at 3 over 2: pan 0.131 -> 0.185 stop, drive 0.339 -> 0.437; docs/LESSONS.md, "3.3 edit cache: what
/// was tried").
pub(crate) const MOVING_INTERVAL: u32 = 2;

/// The variable that switches the cache off: `off` (or `0`) runs the network on every hold, as 3.2.0
/// did (the A/B and rollback switch). Read once, like `NEURAL_FORGE_PREUPSCALE`.
pub(crate) const ENV: &str = "NEURAL_FORGE_EDIT_CACHE";

/// Whether `raw` (the [`ENV`] value) leaves the cache on: unset, empty, `on` or `1`; `off` or `0`
/// switch it off; anything else is an error (and on, the default).
pub(crate) fn parse_env(raw: Option<&str>) -> Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("" | "on" | "1") => Ok(true),
        Some("off" | "0") => Ok(false),
        Some(other) => Err(other.to_string()),
    }
}

/// The cache's interval while the camera moves for this process: [`MOVING_INTERVAL`], or 1 (every
/// hold a refresh, bit for bit 3.2.0's hold) with `NEURAL_FORGE_EDIT_CACHE=off`. Cached; logged once.
pub(crate) fn interval() -> u32 {
    static INTERVAL: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let on = match parse_env(neural_forge_protocol::env::var(ENV).as_deref()) {
            Ok(on) => on,
            Err(other) => {
                crate::log!("[preupscale] {ENV}={other:?} is not one of on, off; the edit cache stays on");
                true
            }
        };
        if !on {
            crate::log!("[preupscale] edit cache off ({ENV}=off): the network runs on every hold");
        }
        if on {
            MOVING_INTERVAL
        } else {
            1
        }
    });
    *INTERVAL
}

/// The push constants of every `preupscale_cache_*.comp`: their `Params` blocks match this field for
/// field (append only, in all nine places).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CachePush {
    /// `uvec2 size`: the render extent.
    pub size: [u32; 2],
    /// `uint flags`: `F_*`.
    pub flags: u32,
    /// `uint depth_kind`: 0 none, 1 f32 bits, 2 D24 UNORM, 3 D16 UNORM ([`depth_kind`]).
    pub depth_kind: u32,
    /// `vec2 mvec_scale`: motion-vector units to render pixels.
    pub mvec_scale: [f32; 2],
    /// `vec2 step_jitter`: jitter[t-1] - jitter[t].
    pub step_jitter: [f32; 2],
    /// `vec2 refresh_jitter`: jitter[refresh] - jitter[t].
    pub refresh_jitter: [f32; 2],
    /// `float paper_white`.
    pub paper_white: f32,
    /// `uint level`: the level the reduce pass builds.
    pub level: u32,
    /// The carry's parameters (`cache_common.Params`, [`Tuning`]).
    pub eps_v: f32,
    pub max_luma_edit: f32,
    pub chroma_stops: f32,
    pub depth_tol: f32,
    pub colour_tol: f32,
    pub colour_floor: f32,
    pub print_tol: f32,
    pub sigma_depth: f32,
    pub sigma_luma: f32,
    /// `float guided_eps`: the guided filter's regularisation ([`Tuning`]).
    pub guided_eps: f32,
}

/// The carry's parameters: `scripts/cache_common.py`'s `Params` defaults (the replay's numbers are
/// for these).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tuning {
    pub eps_v: f32,
    pub max_luma_edit: f32,
    pub chroma_stops: f32,
    pub depth_tol: f32,
    pub colour_tol: f32,
    pub colour_floor: f32,
    pub print_tol: f32,
    pub sigma_depth: f32,
    pub sigma_luma: f32,
    pub guided_eps: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            eps_v: 1.0 / 512.0,
            max_luma_edit: 1.5,
            chroma_stops: 1.0,
            depth_tol: 0.10,
            colour_tol: 0.50,
            colour_floor: 0.6,
            print_tol: 0.25,
            sigma_depth: 0.15,
            sigma_luma: 1.0,
            guided_eps: 0.005,
        }
    }
}

/// How the shaders read the layer's depth copy (`native::DepthCopy`, `cache_dump::depth_to_f32`'s
/// rule): 1 f32 bits (D32, depth handed over as R32), 2 D24 UNORM in the low 24 bits, 3 D16 UNORM
/// (two texels a word).
pub(crate) fn depth_kind(format: vk::Format) -> u32 {
    match format {
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => 3,
        vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D24_UNORM_S8_UINT => 2,
        _ => 1,
    }
}

/// `level_size`: a quarter each side, rounded up, `level + 1` times.
pub(crate) fn level_size(width: u32, height: u32, level: u32) -> (u32, u32) {
    let (mut w, mut h) = (width, height);
    for _ in 0..=level {
        (w, h) = (w.div_ceil(4), h.div_ceil(4));
    }
    (w, h)
}

/// The bytes a [`ScratchLayout`] region starts on (`minStorageBufferOffsetAlignment` is at most 256).
const SCRATCH_ALIGN: u64 = 256;

/// A buffer of the cache inside the network's input features: its offset and size in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Region {
    pub offset: u64,
    pub bytes: u64,
}

/// Where the cache's buffers live in the network's input features (`nn::Frame::features`, 64 bytes a
/// texel of the network's padded field, so at least 64 a render pixel), which the network's frame
/// alone writes and reads: `native_preprocess.comp` rewrites every texel of it before the graph reads
/// any. So it is free outside that frame, and the cache uses it for everything that does not cross
/// one: the refresh's edit (written in the refresh's write-back, after its network frame; read by the
/// cached holds until the next refresh's frame), the staleness (from one hold to the next; a
/// refresh's is never read again), and each hold's own buffers (its depth, the previous field's copy
/// in the capture; the history read, the pyramid and the shown edit in a cached write-back, the
/// guided filter's grids over the first two once the apply has read them). Bytes a render pixel:
/// edits 8 (three 16-bit stops), depths and the rest 4, the refresh's fingerprint 12; 56 in all with
/// the grids, about 156 MB at a 4K output's Balanced render size (2228x1253).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScratchLayout {
    pub src_edit: Region,
    pub src_lin: Region,
    pub src_print: Region,
    pub shown: Region,
    pub stale: Region,
    pub lin: Region,
    /// In the capture, over the write-back's buffers below.
    pub acc_prev: Region,
    pub stale_prev: Region,
    /// In a cached write-back; the guided filter's grids over the three once the apply's first
    /// dispatch has read them.
    pub hist: Region,
    pub conf: Region,
    pub levels: Region,
    pub guide_a: Region,
    pub guide_b: Region,
    pub total: u64,
}

impl ScratchLayout {
    pub(crate) fn new(width: u32, height: u32) -> Self {
        let n = u64::from(width) * u64::from(height);
        let half = u64::from(width.div_ceil(2)) * u64::from(height.div_ceil(2));
        let levels: u64 = (0..3).map(|l| level_size(width, height, l)).map(|(w, h)| u64::from(w) * u64::from(h)).sum();
        fn take(at: &mut u64, bytes: u64) -> Region {
            let r = Region { offset: *at, bytes };
            *at = (*at + bytes).next_multiple_of(SCRATCH_ALIGN);
            r
        }
        let mut at = 0u64;
        let mut l = Self::default();
        for (r, bytes) in
            [(&mut l.src_edit, n * 8), (&mut l.src_lin, n * 4), (&mut l.src_print, n * 12), (&mut l.shown, n * 8), (&mut l.stale, n * 4), (&mut l.lin, n * 4)]
        {
            *r = take(&mut at, bytes);
        }
        let shared = at;
        let mut ends = [shared; 3];
        for (end, set) in ends.iter_mut().zip([
            vec![(&mut l.acc_prev, n * 8), (&mut l.stale_prev, n * 4)],
            vec![(&mut l.hist, n * 8), (&mut l.conf, n * 4), (&mut l.levels, levels * 32)],
            vec![(&mut l.guide_a, half * 32), (&mut l.guide_b, half * 32)],
        ]) {
            for (r, bytes) in set {
                *r = take(end, bytes);
            }
        }
        l.total = ends.into_iter().max().unwrap_or(shared);
        l
    }

    #[cfg(test)]
    fn regions(&self) -> [Region; 13] {
        [
            self.src_edit,
            self.src_lin,
            self.src_print,
            self.shown,
            self.stale,
            self.lin,
            self.acc_prev,
            self.stale_prev,
            self.hist,
            self.conf,
            self.levels,
            self.guide_a,
            self.guide_b,
        ]
    }
}

/// What a hold binds: the frame's images and buffers, which change from hold to hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bindings {
    /// DLSS's colour input (the HDR pass's view of it, GENERAL).
    pub colour: vk::ImageView,
    pub encoded: vk::ImageView,
    pub answer: vk::ImageView,
    pub exposure: vk::Buffer,
    /// The depth copy, when this hold has one.
    pub depth: Option<vk::Buffer>,
    /// DLSS's motion vectors as the capture copied them (packed half2).
    pub mvec: vk::Buffer,
    /// The accumulated field the network reads (`native::NativePass::acc`).
    pub acc: vk::Buffer,
    /// The network's input features and their size in bytes ([`ScratchLayout`]).
    pub scratch: (vk::Buffer, u64),
    /// Which objects the handles above name: the HDR pass's view of the colour input
    /// (`hdr::HdrPass::colour_binding`), the network frame's pass (`native::NativePass::id`: `mvec`, `acc`) and
    /// the network's build (`native::Built::generation`: the input features). A destroyed object's handle comes
    /// back for the next one made, so equal handles alone do not say the set still points at the right objects.
    pub made: [u64; 3],
}

/// A hold of the cache: what the network does on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The network runs; with the interval above 1 its edit is captured for the holds after it.
    Refresh,
    /// No network: the last refresh's edit is carried to this hold.
    Cached,
}

/// What the counters of the last submitted hold said, once its fence was waited on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Outcome {
    /// A cached hold: the share of texels whose history was rejected (0..1).
    pub rejected: Option<f32>,
}

/// The adaptive cadence (`scripts/cache_replay.py`'s `Regime`, which this mirrors): while the camera
/// is still the network runs on every hold, so the view a player looks at closest keeps all of the
/// model's detail (a cached hold cannot carry the part of the answer that follows each hold's
/// jitter); while it moves, on one hold in [`MOVING_INTERVAL`]. A hold reads as still when its median
/// texel moves less than a quarter of the pace, max(0.1% of the render width, 0.5 px) / 3 a hold;
/// the motion pass counts the texels (`STAT_STILL`, `STAT_MOTION_N`). A change waits for consecutive
/// holds that want it, 3 into the still regime and 20 out of it, so the cadence never hops
/// (docs/LESSONS.md, "3.3 edit cache: what was tried").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Regime {
    still: bool,
    /// Consecutive holds that wanted the other regime.
    against: u32,
}

impl Regime {
    /// Holds in a row that move it into the still regime (the shorter interval), and out of it.
    const INTO_STILL: u32 = 3;
    const OUT_OF_STILL: u32 = 20;

    /// The interval for the next hold, `max` being the moving one ([`interval`]).
    pub(crate) fn interval(self, max: u32) -> u32 {
        if self.still {
            1
        } else {
            max
        }
    }

    pub(crate) fn still(self) -> bool {
        self.still
    }

    /// One hold's counts: texels under a quarter of the pace, and all of them.
    pub(crate) fn observe(&mut self, still: u32, n: u32) {
        if n == 0 {
            return;
        }
        let wants = 2 * u64::from(still) > u64::from(n);
        if wants == self.still {
            self.against = 0;
            return;
        }
        self.against += 1;
        if self.against >= if wants { Self::INTO_STILL } else { Self::OUT_OF_STILL } {
            *self = Self { still: wants, against: 0 };
        }
    }
}

/// The cache's bookkeeping on the CPU. A hold takes the stored edit out at its start
/// ([`Self::plan`]) and only a hold that went through ([`Self::went_through`]) puts it back, so a hold
/// that misses for any reason leaves the next one to refresh; so does every reset of the network's
/// history ([`Hold::history_ok`]), a DLSS frame that was not held ([`Hold::hold`] not the next one),
/// and anything else [`Self::drop_edit`] is called for.
#[derive(Debug, Default)]
pub(crate) struct State {
    /// A refresh's edit is stored and may be carried.
    valid: bool,
    /// Cached holds since that refresh; the refresh's jitter and whether it had depth.
    since_refresh: u32,
    refresh_jitter: [f32; 2],
    refresh_depth: bool,
    /// The previous hold's jitter, and whether the accumulated field continues from it (the previous
    /// DLSS frame was a cached hold of this pass: the network did not run since).
    prev_jitter: [f32; 2],
    chain_live: bool,
    /// `Session::seen` of the last hold that went through (consecutive holds: the next is this + 1),
    /// and the network's build it used.
    last_seen: Option<u64>,
    generation: u64,
    /// The depth's convention, decided once [`REVERSE_Z_HOLDS`] holds in a row with conclusive counts
    /// agreed (`infer_reverse_z`); the streak so far.
    reverse_z: Option<bool>,
    reverse_z_streak: (Option<bool>, u32),
    /// The kind of the last submitted hold whose counters are still to be read.
    pending: Option<Kind>,
    /// The adaptive cadence, from the counts of the holds before.
    pub(crate) regime: Regime,
}

/// What a hold brings to [`State::plan`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hold {
    /// The interval while the camera moves ([`Regime`]): [`interval`] (the tests take others).
    pub interval: u32,
    /// `Session::seen`: the DLSS submit this hold is.
    pub hold: u64,
    pub jitter: Option<[f32; 2]>,
    /// The capture copies motion vectors and depth this hold.
    pub mvec: bool,
    pub depth: bool,
    /// The network's history would be kept on this hold (`HistoryGap::pending` is `None`).
    pub history_ok: bool,
    /// A cache dump is armed for this hold (it needs the network's answer).
    pub armed: bool,
    /// `debug_view` asks for the rejection mask.
    pub debug_mask: bool,
    /// `native::Built::generation`: a rebuilt network has new input features, which held the
    /// refresh's edit and the shown one.
    pub generation: u64,
}

/// What [`State::plan`] decided for a hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Plan {
    pub kind: Kind,
    /// The push constants' flags and jitters for this hold's passes ([`CachePass::push`]).
    pub flags: u32,
    pub step_jitter: [f32; 2],
    pub refresh_jitter: [f32; 2],
}

impl State {
    /// Drops the stored edit: the next hold refreshes.
    pub(crate) fn drop_edit(&mut self) {
        self.valid = false;
        self.chain_live = false;
    }

    /// Whether this hold is cached or refreshes, and its passes' flags. Takes the stored edit out.
    /// The depth's convention once decided ([`infer_reverse_z`]): `Some(true)` for reversed-Z.
    pub(crate) fn reverse_z(&self) -> Option<bool> {
        self.reverse_z
    }

    pub(crate) fn plan(&mut self, h: &Hold) -> Plan {
        let consecutive = self.last_seen.is_some_and(|s| s.wrapping_add(1) == h.hold) && self.generation == h.generation;
        let had = std::mem::take(&mut self.valid);
        let chain_live = std::mem::take(&mut self.chain_live) && consecutive;
        let jitter = h.jitter.unwrap_or_default();
        let chained = h.interval > 1;
        // The hold before went through on this chain, and this one can read the refresh's edit.
        let continues = chained && had && consecutive && h.jitter.is_some() && h.mvec && h.depth == self.refresh_depth && h.history_ok && !h.armed;
        let cached = continues && self.since_refresh + 1 < self.regime.interval(h.interval);
        let mut flags = if super::hdr::highlight_flags() != 0 { F_KEEP_CLAMPED } else { 0 }
            | if super::hdr::transfer_flags() & super::hdr::ACES != 0 { F_ACES } else { 0 };
        if chained {
            flags |= F_CHAINED;
            if !chain_live {
                flags |= F_FRESH;
            }
            if h.depth {
                flags |= F_NEAREST;
            }
            match self.reverse_z {
                Some(false) => {}
                Some(true) => flags |= F_REVERSE_Z,
                // Assumed until the prep pass's counts tell.
                None => flags |= F_REVERSE_Z | F_COUNT_DEPTH,
            }
            if cached && h.debug_mask {
                flags |= F_DEBUG_MASK;
            }
        }
        let delta = |a: [f32; 2]| [a[0] - jitter[0], a[1] - jitter[1]];
        Plan {
            kind: if cached { Kind::Cached } else { Kind::Refresh },
            flags,
            step_jitter: if consecutive { delta(self.prev_jitter) } else { [0.0; 2] },
            refresh_jitter: delta(self.refresh_jitter),
        }
    }

    /// The hold `h` planned as `plan` went through (everything submitted).
    pub(crate) fn went_through(&mut self, h: &Hold, plan: &Plan) {
        self.last_seen = Some(h.hold);
        self.generation = h.generation;
        self.prev_jitter = h.jitter.unwrap_or_default();
        let chained = h.interval > 1;
        self.pending = chained.then_some(plan.kind);
        match plan.kind {
            Kind::Refresh => {
                self.valid = chained && h.jitter.is_some();
                self.since_refresh = 0;
                self.refresh_jitter = self.prev_jitter;
                self.refresh_depth = h.depth;
                self.chain_live = false;
            }
            Kind::Cached => {
                self.valid = true;
                self.since_refresh += 1;
                self.chain_live = true;
            }
        }
    }
}

/// Holds in a row whose depth counts must agree before the convention is decided: one frame that
/// says little (a loading screen, a fade) does not fix it for the extent.
pub(crate) const REVERSE_Z_HOLDS: u32 = 3;

/// The share of depth texels that must be neither 0 nor 1 (drawn geometry) for a hold's counts to
/// count (`cache_common.DEPTH_GEOMETRY_MIN`): a frame of cleared depth alone, or depth never written,
/// tells nothing about which way it runs.
pub(crate) const DEPTH_GEOMETRY_MIN: f64 = 0.05;

/// `infer_reverse_z` from one hold's prep counts: the cleared value first (a real share of texels
/// at exactly 0, few at 1, is reversed-Z), else the ground (depth growing toward the bottom is
/// reversed-Z; means of the top and bottom quarters, where the replay tool takes medians). `None`
/// when the counts cannot tell, and for a frame with too little drawn geometry
/// ([`DEPTH_GEOMETRY_MIN`]).
pub(crate) fn infer_reverse_z(counts: &[u32; STAT_WORDS]) -> Option<bool> {
    let n = counts[STAT_DEPTH_N];
    if n == 0 {
        return None;
    }
    let (z0, z1) = (f64::from(counts[STAT_Z0]) / f64::from(n), f64::from(counts[STAT_Z1]) / f64::from(n));
    if 1.0 - z0 - z1 < DEPTH_GEOMETRY_MIN {
        return None;
    }
    if z0.max(z1) > 0.005 && (z0 > 2.0 * z1 || z1 > 2.0 * z0) {
        return Some(z0 > z1);
    }
    let (tn, bn) = (counts[STAT_TOP_N], counts[STAT_BOTTOM_N]);
    if tn == 0 || bn == 0 {
        return None;
    }
    let sum = |lo: usize, hi: usize| (u64::from(counts[hi]) << 32 | u64::from(counts[lo])) as f64;
    let top = sum(STAT_TOP_SUM, STAT_TOP_SUM_HI) / f64::from(tn);
    let bottom = sum(STAT_BOTTOM_SUM, STAT_BOTTOM_SUM_HI) / f64::from(bn);
    (top != bottom).then_some(bottom > top)
}

impl State {
    /// One hold's verdict on the depth's convention ([`infer_reverse_z`]); decides it once
    /// [`REVERSE_Z_HOLDS`] conclusive holds in a row agree. Inconclusive holds neither count nor break
    /// the streak. `true` when this hold decided it.
    pub(crate) fn vote_reverse_z(&mut self, verdict: Option<bool>) -> bool {
        let Some(v) = verdict.filter(|_| self.reverse_z.is_none()) else { return false };
        self.reverse_z_streak = match self.reverse_z_streak {
            (Some(prev), n) if prev == v => (Some(v), n + 1),
            _ => (Some(v), 1),
        };
        if self.reverse_z_streak.1 < REVERSE_Z_HOLDS {
            return false;
        }
        self.reverse_z = Some(v);
        true
    }
}

/// The pass for one extent: seven pipelines over one descriptor set, the counters, the timers.
pub(crate) struct CachePass {
    width: u32,
    height: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: [vk::Pipeline; 7],
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    stats: HostBuffer,
    /// Bound where the scratch or the depth is not there (the shaders do not read it then).
    dummy: OwnBuffer,
    /// The interval is above 1 and the network's input features hold the [`ScratchLayout`].
    chained: bool,
    /// Where the scratch buffers are ([`ScratchLayout::new`] for this extent), and the buffer they are
    /// in for this hold ([`Self::bind`]).
    layout: ScratchLayout,
    scratch: vk::Buffer,
    /// The network's input features were too small for the layout (logged once).
    said_small: bool,
    /// GPU timestamps around the passes in the capture and in the write-back (before, or in place
    /// of, the decode). None inside DLSS's buffer (the hold times its slot).
    timers: [Option<crate::gpu_timer::GpuTimer>; 2],
    /// What the set points at ([`Self::bind`], with [`Bindings::made`]) and whether it was chained: written again
    /// only when that changes.
    bound: Option<(Bindings, bool)>,
    pub(crate) state: State,
    pub(crate) tuning: Tuning,
}

// SAFETY: plain handles and the layer's own mapping, only used behind the device's `State` mutex.
unsafe impl Send for CachePass {}

/// Descriptor types of bindings 0..=20 (the shaders' layout).
fn binding_type(n: u32) -> vk::DescriptorType {
    match n {
        0 | 1 | 14 => vk::DescriptorType::STORAGE_IMAGE,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}
const BINDINGS: u32 = 21;

impl CachePass {
    /// # Safety
    /// `device` is live; `family` is the queue family the holds are submitted on.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        family: u32,
        width: u32,
        height: u32,
        timed: bool,
    ) -> Option<Self> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS)
            .map(|n| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(n)
                    .descriptor_type(binding_type(n))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let mut p = Partial {
            device,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            pool: vk::DescriptorPool::null(),
            stats: None,
            dummy: None,
        };
        // SAFETY: valid create infos; everything made is owned by `p` until it moves into the result.
        unsafe {
            p.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<CachePush>() as u32);
            p.pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&p.set_layout)).push_constant_ranges(std::slice::from_ref(&push)),
                    None,
                )
                .ok()?;
            for spv in SPV {
                let pipeline = hdr::compute_pipeline(device, p.pipeline_layout, spv)?;
                p.pipelines.push(pipeline);
            }
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 3 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: BINDINGS - 3 },
            ];
            p.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).ok()?;
            let set = device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(p.pool).set_layouts(std::slice::from_ref(&p.set_layout)))
                .ok()?[0];
            // TRANSFER: inside DLSS's buffer the hold clears them and copies them out for the host itself.
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
            p.stats = Some(super::own_host_buffer_with(device, instance, physical_device, STAT_BYTES, usage)?);
            p.dummy = Some(OwnBuffer::new(device, instance, physical_device, 256, vk::BufferUsageFlags::STORAGE_BUFFER)?);
            let stats = p.stats.take()?;
            let dummy = p.dummy.take()?;
            std::ptr::write_bytes(stats.ptr, 0, STAT_BYTES as usize);
            let pipelines: [vk::Pipeline; 7] = std::mem::take(&mut p.pipelines).try_into().ok()?;
            Some(Self {
                width,
                height,
                set_layout: std::mem::take(&mut p.set_layout),
                pipeline_layout: std::mem::take(&mut p.pipeline_layout),
                pipelines,
                pool: std::mem::take(&mut p.pool),
                set,
                stats,
                dummy,
                chained: false,
                layout: ScratchLayout::new(width, height),
                scratch: vk::Buffer::null(),
                said_small: false,
                // Only made when kept: a timer has no destructor, and one dropped leaks its query pool.
                timers: [(); 2].map(|()| if timed { crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family) } else { None }),
                bound: None,
                state: State::default(),
                tuning: Tuning::default(),
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    /// The bytes its buffers take above interval 1 ([`ScratchLayout::total`]).
    pub(crate) fn scratch_bytes(&self) -> u64 {
        self.layout.total
    }

    /// Chains holds above interval 1: `false` when the network's input features (`scratch_bytes`)
    /// cannot hold the [`ScratchLayout`] (the hold then runs at interval 1).
    pub(crate) fn chain(&mut self, scratch_bytes: u64) -> bool {
        if scratch_bytes < self.layout.total {
            if !std::mem::replace(&mut self.said_small, true) {
                crate::log!(
                    "[preupscale] edit cache: the network's input features ({scratch_bytes} bytes) cannot hold its buffers ({} bytes); interval 1",
                    self.layout.total
                );
            }
            return false;
        }
        if !std::mem::replace(&mut self.chained, true) {
            crate::log!(
                "[preupscale] edit cache: chaining {}x{}, {:.1} MiB of the network's input features",
                self.width,
                self.height,
                self.layout.total as f64 / 1048576.0
            );
        }
        true
    }

    /// Back at interval 1: drops the edit.
    pub(crate) fn unchain(&mut self) {
        if std::mem::replace(&mut self.chained, false) {
            self.state.drop_edit();
            self.state.regime = Regime::default();
        }
    }

    /// Points the set at this hold's resources.
    ///
    /// # Safety
    /// Nothing of the pass is pending; every handle is live.
    pub(crate) unsafe fn bind(&mut self, device: &ash::Device, b: &Bindings) {
        if self.bound == Some((*b, self.chained)) {
            return;
        }
        // Not while a hold recorded into DLSS's buffer with the set may run (the hold then misses).
        if !super::may_rewrite("the edit cache's descriptor set") {
            return;
        }
        self.bound = Some((*b, self.chained));
        let dummy = self.dummy.buffer;
        // The scratch only above interval 1 and with room for the layout.
        let scratch = (self.chained && b.scratch.1 >= self.layout.total).then_some(b.scratch.0);
        self.scratch = scratch.unwrap_or(vk::Buffer::null());
        let l = self.layout;
        let region = |r: Region| scratch.map_or((dummy, 0, vk::WHOLE_SIZE), |buffer| (buffer, r.offset, r.bytes));
        let whole = |buffer: vk::Buffer| (buffer, 0, vk::WHOLE_SIZE);
        let buffers: [(u32, (vk::Buffer, u64, u64)); 18] = [
            (2, whole(b.exposure)),
            (3, whole(b.depth.unwrap_or(dummy))),
            (4, whole(b.mvec)),
            (5, whole(b.acc)),
            (6, region(l.acc_prev)),
            (7, region(l.lin)),
            (8, region(l.src_edit)),
            (9, region(l.src_print)),
            (10, region(l.hist)),
            (11, region(l.conf)),
            (12, region(l.levels)),
            (13, whole(self.stats.buffer)),
            (15, region(l.shown)),
            (16, region(l.src_lin)),
            (17, region(l.guide_a)),
            (18, region(l.guide_b)),
            (19, region(l.stale)),
            (20, region(l.stale_prev)),
        ];
        let images = [(0u32, b.colour), (1, b.encoded), (14, b.answer)];
        let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|&(_, (buffer, offset, range))| [vk::DescriptorBufferInfo { buffer, offset, range }]).collect();
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = images
            .iter()
            .map(|&(_, view)| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }])
            .collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(BINDINGS as usize);
        for ((n, _), info) in buffers.iter().zip(&buffer_infos) {
            writes.push(
                vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(*n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info),
            );
        }
        for ((n, _), info) in images.iter().zip(&image_infos) {
            writes
                .push(vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(*n).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(info));
        }
        // SAFETY: the set is not in use (contract); every handle is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// The push constants of this hold with `flags`, before the per-pass fields.
    pub(crate) fn push(&self, flags: u32, depth_kind: u32, mvec_scale: [f32; 2], paper_white: f32) -> CachePush {
        let t = self.tuning;
        CachePush {
            size: [self.width, self.height],
            flags,
            depth_kind,
            mvec_scale,
            step_jitter: [0.0; 2],
            refresh_jitter: [0.0; 2],
            paper_white,
            level: 0,
            eps_v: t.eps_v,
            max_luma_edit: t.max_luma_edit,
            chroma_stops: t.chroma_stops,
            depth_tol: t.depth_tol,
            colour_tol: t.colour_tol,
            colour_floor: t.colour_floor,
            print_tol: t.print_tol,
            sigma_depth: t.sigma_depth,
            sigma_luma: t.sigma_luma,
            guided_eps: t.guided_eps,
        }
    }

    /// # Safety
    /// `cmd` is recording.
    unsafe fn dispatch(&self, device: &ash::Device, cmd: vk::CommandBuffer, pipeline: usize, push: &CachePush, width: u32, height: u32) {
        // SAFETY: `CachePush` is `repr(C)` plain data, read as bytes for its own size.
        let bytes = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(push).cast::<u8>(), std::mem::size_of::<CachePush>()) };
        // SAFETY: forwarded; the set, layout and pipelines are this pass's own.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[pipeline]);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.set), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
        }
    }

    /// # Safety
    /// `cmd` is recording.
    unsafe fn compute_barrier(device: &ash::Device, cmd: vk::CommandBuffer, src_stage: vk::PipelineStageFlags, src: vk::AccessFlags) {
        let b = vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: forwarded.
        unsafe { device.cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[b], &[], &[]) };
    }

    /// # Safety
    /// `cmd` is recording.
    unsafe fn to_host(device: &ash::Device, cmd: vk::CommandBuffer) {
        let b = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
        // SAFETY: forwarded.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            )
        };
    }

    /// Records the capture's part, after its copies and the encode: at interval 1 (`push.flags`
    /// without [`F_CHAINED`]) the vectors into the accumulated field as they are; else the prep pass,
    /// the previous field kept, and the motion pass.
    ///
    /// # Safety
    /// `cmd` is recording; [`Self::bind`] was called for this hold with `acc`; with [`F_CHAINED`] the
    /// scratch is bound.
    pub(crate) unsafe fn record_motion(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush, acc: vk::Buffer) {
        let chained = push.flags & F_CHAINED != 0;
        // SAFETY: forwarded.
        unsafe {
            Self::compute_barrier(
                device,
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
            );
            if let Some(t) = &self.timers[0] {
                t.record_start(device, cmd);
            }
            if chained {
                let l = self.layout;
                self.dispatch(device, cmd, PREP, push, self.width, self.height);
                // The previous field and staleness kept for the motion pass to read (the network's binding is
                // the field itself).
                let kept = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[kept],
                    &[],
                    &[],
                );
                device.cmd_copy_buffer(cmd, acc, self.scratch, &[vk::BufferCopy { src_offset: 0, dst_offset: l.acc_prev.offset, size: l.acc_prev.bytes }]);
                device.cmd_copy_buffer(
                    cmd,
                    self.scratch,
                    self.scratch,
                    &[vk::BufferCopy { src_offset: l.stale.offset, dst_offset: l.stale_prev.offset, size: l.stale.bytes }],
                );
                Self::compute_barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
                );
            }
            self.dispatch(device, cmd, MOTION, push, self.width, self.height);
            if let Some(t) = &self.timers[0] {
                t.record_end(device, cmd);
            }
            Self::to_host(device, cmd);
        }
    }

    /// Records the refresh hold's capture of the edit, in the write-back after the answer was copied
    /// into its image and before the decode runs.
    ///
    /// # Safety
    /// `cmd` is recording; [`Self::bind`] was called for this hold, chained.
    pub(crate) unsafe fn record_capture(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush) {
        // SAFETY: forwarded.
        unsafe {
            if let Some(t) = &self.timers[1] {
                t.record_start(device, cmd);
            }
            self.dispatch(device, cmd, CAPTURE, push, self.width, self.height);
            if let Some(t) = &self.timers[1] {
                t.record_end(device, cmd);
            }
            Self::to_host(device, cmd);
            // The decode writes the colour input this pass read.
            Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
        }
    }

    /// Records a cached hold's write-back: the history read, the pyramid, the band split, the guided
    /// filter at half size, and its blend written over the colour input.
    ///
    /// # Safety
    /// `cmd` is recording, after the write-back's opening barrier; [`Self::bind`] was called for this
    /// hold, chained.
    pub(crate) unsafe fn record_apply(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush) {
        // SAFETY: forwarded.
        unsafe {
            if let Some(t) = &self.timers[1] {
                t.record_start(device, cmd);
            }
            self.dispatch(device, cmd, HISTORY, push, self.width, self.height);
            for level in 0..3 {
                Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
                let (lw, lh) = level_size(self.width, self.height, level);
                self.dispatch(device, cmd, REDUCE, &CachePush { level, ..*push }, lw, lh);
            }
            Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
            self.dispatch(device, cmd, APPLY, &CachePush { level: 0, ..*push }, self.width, self.height);
            let (hw, hh) = (self.width.div_ceil(2), self.height.div_ceil(2));
            for level in 0..3 {
                Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
                self.dispatch(device, cmd, GUIDED, &CachePush { level, ..*push }, hw, hh);
            }
            Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
            self.dispatch(device, cmd, APPLY, &CachePush { level: 1, ..*push }, self.width, self.height);
            if let Some(t) = &self.timers[1] {
                t.record_end(device, cmd);
            }
            Self::to_host(device, cmd);
        }
    }

    /// Records only a cached hold's history read (tests: the guided filter's grids overwrite it later
    /// in [`Self::record_apply`]).
    ///
    /// # Safety
    /// As [`Self::record_apply`].
    #[cfg(test)]
    unsafe fn record_history(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush) {
        // SAFETY: forwarded.
        unsafe { self.dispatch(device, cmd, HISTORY, push, self.width, self.height) };
    }

    /// Marks the timers whose passes were submitted (`capture`: the capture's part; `writeback`: the
    /// write-back's).
    pub(crate) fn submitted(&mut self, capture: bool, writeback: bool) {
        for (t, on) in self.timers.iter_mut().zip([capture, writeback]) {
            if let Some(t) = t.as_mut().filter(|_| on) {
                t.mark_submitted();
            }
        }
    }

    /// The last hold's cache GPU time (both parts that ran), once its fence was waited on.
    pub(crate) fn take_gpu_ms(&mut self, device: &ash::Device) -> Option<f32> {
        let mut total = None;
        for t in self.timers.iter_mut().flatten() {
            t.read_if_pending(device);
            if let Some(ms) = t.take_reading() {
                total = Some(total.unwrap_or(0.0) + ms);
            }
        }
        total
    }

    fn counters(&self) -> [u32; STAT_WORDS] {
        let mut out = [0u32; STAT_WORDS];
        // SAFETY: host-coherent, mapped for STAT_BYTES; the hold that wrote it was waited on and its
        // passes closed with a barrier to the host.
        unsafe { std::ptr::copy_nonoverlapping(self.stats.ptr.cast::<u32>(), out.as_mut_ptr(), STAT_WORDS) };
        out
    }

    /// Clears the counters before a hold's passes are submitted ([`CLEARED_BYTES`]).
    pub(crate) fn clear_counters(&self) {
        // SAFETY: as `counters`; nothing pending writes them (after the hold's wait).
        unsafe { std::ptr::write_bytes(self.stats.ptr, 0, CLEARED_BYTES as usize) };
    }

    /// The counters' buffer: the hold inside DLSS's buffer clears it there ([`CLEARED_BYTES`]) and copies it
    /// out for the host, and the hold guard clears [`STAT_EDIT_STORED`] in it.
    pub(crate) fn stats_buffer(&self) -> vk::Buffer {
        self.stats.buffer
    }

    /// Reads what the last submitted hold's passes counted (its fence was waited on): the stored
    /// edit is dropped when the refresh's exposure was unusable or a cached hold could not apply,
    /// the motion's counts go to the [`Regime`], and the depth's convention is decided once.
    pub(crate) fn after_fence(&mut self) -> Outcome {
        let Some(kind) = self.state.pending.take() else { return Outcome::default() };
        let c = self.counters();
        self.after_counters(kind, &c)
    }

    /// [`Self::after_fence`] for counters `c` of a hold planned as `kind` (inside DLSS's buffer, read back
    /// once its buffer ran).
    pub(crate) fn after_counters(&mut self, kind: Kind, c: &[u32; STAT_WORDS]) -> Outcome {
        let decided = self.state.reverse_z.is_some();
        let mut out = Outcome::default();
        self.state.regime.observe(c[STAT_STILL], c[STAT_MOTION_N]);
        if !decided && self.state.vote_reverse_z(infer_reverse_z(c)) {
            let rev = self.state.reverse_z == Some(true);
            crate::log!(
                "[preupscale] edit cache: depth is {} ({REVERSE_Z_HOLDS} holds agreeing; the last: {} of {} texels at 0, {} at 1)",
                if rev { "reversed-Z" } else { "forward" },
                c[STAT_Z0],
                c[STAT_DEPTH_N],
                c[STAT_Z1]
            );
            // The stored edit's depth was read with the assumed convention.
            self.state.drop_edit();
        }
        match kind {
            Kind::Refresh => {
                if c[STAT_EDIT_OK] == 0 {
                    self.state.drop_edit();
                }
            }
            Kind::Cached => {
                if c[STAT_APPLY_BAD] != 0 {
                    self.state.drop_edit();
                }
                let n = u64::from(self.width) * u64::from(self.height);
                out.rejected = Some((f64::from(c[STAT_REJECTED]) / n.max(1) as f64) as f32);
            }
        }
        out
    }

    /// # Safety
    /// Nothing of the pass may still be pending.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; destroying the pool frees the set.
        unsafe {
            for t in self.timers.iter().flatten() {
                t.destroy(device);
            }
            self.stats.destroy(device);
            self.dummy.destroy(device);
            device.destroy_descriptor_pool(self.pool, None);
            for &p in &self.pipelines {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// [`CachePass::build`]'s objects so far; destroys them unless moved out.
struct Partial<'a> {
    device: &'a ash::Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    pool: vk::DescriptorPool,
    stats: Option<HostBuffer>,
    dummy: Option<OwnBuffer>,
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        let d = self.device;
        // SAFETY: nothing here was ever submitted; null handles are ignored by the destroy calls.
        unsafe {
            if let Some(b) = self.stats.take() {
                b.destroy(d);
            }
            if let Some(b) = self.dummy.take() {
                b.destroy(d);
            }
            d.destroy_descriptor_pool(self.pool, None);
            for &p in &self.pipelines {
                if p.as_raw() != 0 {
                    d.destroy_pipeline(p, None);
                }
            }
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hdr::OwnImage;
    use super::*;

    #[test]
    fn the_push_constants_match_the_shaders_params_block() {
        // uvec2 size, uint flags, uint depth_kind, vec2 x3, float, uint, ten floats: 88 bytes, std430.
        assert_eq!(std::mem::size_of::<CachePush>(), 88);
        assert_eq!(std::mem::offset_of!(CachePush, flags), 8);
        assert_eq!(std::mem::offset_of!(CachePush, depth_kind), 12);
        assert_eq!(std::mem::offset_of!(CachePush, mvec_scale), 16);
        assert_eq!(std::mem::offset_of!(CachePush, step_jitter), 24);
        assert_eq!(std::mem::offset_of!(CachePush, refresh_jitter), 32);
        assert_eq!(std::mem::offset_of!(CachePush, paper_white), 40);
        assert_eq!(std::mem::offset_of!(CachePush, level), 44);
        assert_eq!(std::mem::offset_of!(CachePush, eps_v), 48);
        assert_eq!(std::mem::offset_of!(CachePush, print_tol), 72);
        assert_eq!(std::mem::offset_of!(CachePush, sigma_luma), 80);
        assert_eq!(std::mem::offset_of!(CachePush, guided_eps), 84);
        for spv in SPV {
            assert!(spv.len() > 20 && spv.len().is_multiple_of(4), "embedded SPIR-V");
        }
        assert_eq!(depth_kind(vk::Format::D32_SFLOAT_S8_UINT), 1);
        assert_eq!(depth_kind(vk::Format::D24_UNORM_S8_UINT), 2);
        assert_eq!(depth_kind(vk::Format::D16_UNORM), 3);
        assert_eq!(level_size(1485, 836, 0), (372, 209));
        assert_eq!(level_size(1485, 836, 2), (24, 14));
    }

    /// The cache's buffers fit in the network's input features at every size: 64 bytes a texel of
    /// its field, at least the render extent (the hold checks the real size too: `chain`).
    #[test]
    fn the_scratch_layout_fits_in_the_networks_input_features() {
        // From the replay tool's synthetic dumps up (smaller frames are not held: the region
        // alignment would not fit, and `chain` says so).
        for (w, h) in [(96, 64), (97, 65), (641, 359), (1485, 836), (2228, 1253), (2560, 1440), (3840, 2160)] {
            let l = ScratchLayout::new(w, h);
            let features = u64::from(w) * u64::from(h) * 64;
            assert!(l.total <= features, "{w}x{h}: {} bytes in {features}", l.total);
            for r in l.regions() {
                assert!(r.offset.is_multiple_of(SCRATCH_ALIGN) && r.offset + r.bytes <= l.total, "{w}x{h}: {r:?}");
            }
            // What is live at once does not overlap: the long-lived buffers, then the capture's or the
            // write-back's (the grids replace the history read and the pyramid).
            let long = [l.src_edit, l.src_lin, l.src_print, l.shown, l.stale, l.lin];
            for set in [&[l.acc_prev, l.stale_prev][..], &[l.hist, l.conf, l.levels], &[l.guide_a, l.guide_b]] {
                let live: Vec<Region> = long.iter().chain(set).copied().collect();
                for (i, a) in live.iter().enumerate() {
                    for b in &live[i + 1..] {
                        assert!(a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset, "{w}x{h}: {a:?} overlaps {b:?}");
                    }
                }
            }
        }
        // At a 4K output's Balanced render size: 56 bytes a pixel.
        let l = ScratchLayout::new(2228, 1253);
        assert!(l.total < 57 * 2228 * 1253, "{}", l.total);
    }

    fn hold(h: u64) -> Hold {
        Hold { interval: 2, hold: h, jitter: Some([0.25, -0.25]), mvec: true, depth: true, history_ok: true, armed: false, debug_mask: false, generation: 1 }
    }

    /// Runs `holds` through the state, each going through; returns the kinds.
    fn kinds(state: &mut State, holds: &[Hold]) -> Vec<Kind> {
        holds
            .iter()
            .map(|h| {
                let plan = state.plan(h);
                state.went_through(h, &plan);
                plan.kind
            })
            .collect()
    }

    /// The interval's rhythm, and every reset forcing a refresh: a DLSS frame that was not held, the
    /// network's history reset, a dropped edit, a hold that missed, no jitter or vectors, depth coming
    /// or going, an armed dump; and interval 1 never caches.
    #[test]
    fn a_hold_is_cached_only_right_after_a_good_one_within_the_interval() {
        use Kind::{Cached as C, Refresh as R};
        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[hold(1), hold(2), hold(3), hold(4), hold(5)]), [R, C, R, C, R]);
        let mut s = State::default();
        let four = |h| Hold { interval: 4, ..hold(h) };
        assert_eq!(kinds(&mut s, &[four(1), four(2), four(3), four(4), four(5), four(6)]), [R, C, C, C, R, C]);
        let mut s = State::default();
        let one = |h| Hold { interval: 1, ..hold(h) };
        assert_eq!(kinds(&mut s, &[one(1), one(2), one(3)]), [R, R, R]);
        // A DLSS frame between two holds went to DLSS untouched (hold 3 missing).
        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[hold(1), hold(4), hold(5)]), [R, R, C]);
        // The network's history was reset (500 ms, a skip, another extent).
        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[hold(1), Hold { history_ok: false, ..hold(2) }, hold(3)]), [R, R, C]);
        // A dropped edit (the refresh's exposure was unusable, ...).
        let mut s = State::default();
        kinds(&mut s, &[hold(1)]);
        s.drop_edit();
        assert_eq!(kinds(&mut s, &[hold(2), hold(3)]), [R, C]);
        // A hold that planned but did not go through takes the edit with it.
        let mut s = State::default();
        kinds(&mut s, &[hold(1)]);
        let missed = s.plan(&hold(2));
        assert_eq!(missed.kind, C);
        assert_eq!(kinds(&mut s, &[hold(3), hold(4)]), [R, C]);
        // No jitter, no vectors, depth gone, an armed dump, a rebuilt network (its input features
        // held the edit).
        for odd in [
            Hold { jitter: None, ..hold(2) },
            Hold { mvec: false, ..hold(2) },
            Hold { depth: false, ..hold(2) },
            Hold { armed: true, ..hold(2) },
            Hold { generation: 2, ..hold(2) },
        ] {
            let mut s = State::default();
            assert_eq!(kinds(&mut s, &[hold(1), odd]), [R, R], "{odd:?}");
        }
        // A refresh without jitter stores nothing to carry.
        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[Hold { jitter: None, ..hold(1) }, hold(2)]), [R, R]);
    }

    /// `NEURAL_FORGE_EDIT_CACHE`: the cache is on unless the variable says `off` (or `0`); a value it
    /// does not know is an error, which [`interval`] logs and keeps the cache on for.
    #[test]
    fn the_edit_cache_is_on_unless_switched_off() {
        for on in [None, Some(""), Some("on"), Some("1"), Some(" on ")] {
            assert_eq!(parse_env(on), Ok(true), "{on:?}");
        }
        for off in [Some("off"), Some("0"), Some("off\n")] {
            assert_eq!(parse_env(off), Ok(false), "{off:?}");
        }
        assert_eq!(parse_env(Some("2")), Err("2".into()));
        assert_eq!(MOVING_INTERVAL, 2);
    }

    /// The adaptive cadence: 3 still holds in a row (the median texel under a quarter of the pace)
    /// make every hold a refresh, 20 moving ones give the moving interval back; a reading against the
    /// regime restarts the count.
    #[test]
    fn a_still_camera_runs_the_network_on_every_hold() {
        use Kind::{Cached as C, Refresh as R};
        let (still, moving) = ((60, 100), (10, 100));
        let mut r = Regime::default();
        for c in [still, still, moving, still, still] {
            r.observe(c.0, c.1);
        }
        assert!(!r.still(), "the moving reading restarted the count");
        r.observe(still.0, still.1);
        assert!(r.still());
        assert_eq!(r.interval(3), 1);
        for _ in 0..30 {
            r.observe(moving.0, moving.1);
            r.observe(still.0, still.1);
        }
        r.observe(0, 0);
        assert!(r.still(), "a still reading (and nothing counted) keeps the regime");
        for _ in 0..19 {
            r.observe(moving.0, moving.1);
        }
        assert!(r.still());
        r.observe(moving.0, moving.1);
        assert!(!r.still());
        assert_eq!(r.interval(3), 3);

        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[hold(1), hold(2)]), [R, C]);
        s.regime = Regime { still: true, against: 0 };
        assert_eq!(kinds(&mut s, &[hold(3), hold(4), hold(5)]), [R, R, R]);
        s.regime = Regime::default();
        assert_eq!(kinds(&mut s, &[hold(6), hold(7), hold(8)]), [C, R, C], "moving again: the hold after a refresh is cached");
    }

    /// The flags and jitters: the hold after a refresh starts the accumulated field afresh, the
    /// next continues it; jitters are differences to this hold's.
    #[test]
    fn the_plan_chains_the_motion_from_the_refresh() {
        let mut s = State::default();
        let j = |h: u64, x: f32| Hold { jitter: Some([x, 0.0]), interval: 4, ..hold(h) };
        let p1 = s.plan(&j(1, 0.5));
        assert_eq!(p1.kind, Kind::Refresh);
        assert_ne!(p1.flags & F_FRESH, 0, "the first hold has no field to continue");
        s.went_through(&j(1, 0.5), &p1);
        let p2 = s.plan(&j(2, -0.25));
        assert_eq!(p2.kind, Kind::Cached);
        assert_ne!(p2.flags & F_FRESH, 0, "the network ran on the previous hold");
        assert_eq!(p2.step_jitter, [0.75, 0.0]);
        assert_eq!(p2.refresh_jitter, [0.75, 0.0]);
        s.went_through(&j(2, -0.25), &p2);
        let p3 = s.plan(&j(3, 0.125));
        assert_eq!(p3.kind, Kind::Cached);
        assert_eq!(p3.flags & F_FRESH, 0, "continues the field of the cached hold before");
        assert_eq!(p3.step_jitter, [-0.375, 0.0]);
        assert_eq!(p3.refresh_jitter, [0.375, 0.0]);
        assert_ne!(p3.flags & (F_CHAINED | F_NEAREST | F_REVERSE_Z | F_KEEP_CLAMPED), 0);
        let one = s.plan(&Hold { interval: 1, ..j(4, 0.0) });
        assert_eq!(one.flags, F_KEEP_CLAMPED, "interval 1: the vectors as they are");
    }

    #[test]
    fn reversed_z_is_told_by_the_cleared_value_then_by_the_ground() {
        let mut c = [0u32; STAT_WORDS];
        assert_eq!(infer_reverse_z(&c), None);
        c[STAT_DEPTH_N] = 1000;
        c[STAT_Z0] = 300;
        assert_eq!(infer_reverse_z(&c), Some(true));
        c[STAT_Z0] = 0;
        c[STAT_Z1] = 300;
        assert_eq!(infer_reverse_z(&c), Some(false));
        c[STAT_Z1] = 0;
        (c[STAT_TOP_N], c[STAT_BOTTOM_N], c[STAT_TOP_SUM], c[STAT_BOTTOM_SUM]) = (10, 10, 100, 300);
        assert_eq!(infer_reverse_z(&c), Some(true), "depth grows toward the bottom");
        c[STAT_BOTTOM_SUM] = 100;
        assert_eq!(infer_reverse_z(&c), None);
    }

    /// A frame with (almost) no geometry tells nothing: an all-zero depth (a loading screen, depth
    /// never written) once fixed reversed-Z for the extent.
    #[test]
    fn a_frame_of_cleared_depth_alone_is_not_evidence() {
        let mut c = [0u32; STAT_WORDS];
        c[STAT_DEPTH_N] = 1000;
        c[STAT_Z0] = 1000;
        assert_eq!(infer_reverse_z(&c), None, "all zero");
        c[STAT_Z0] = 960;
        assert_eq!(infer_reverse_z(&c), None, "96% cleared, 4% drawn");
        c[STAT_Z0] = 940;
        assert_eq!(infer_reverse_z(&c), Some(true), "6% drawn");
        (c[STAT_Z0], c[STAT_Z1]) = (0, 1000);
        assert_eq!(infer_reverse_z(&c), None, "all one");
    }

    /// The ground's sums carry into their high words: a quarter of the rows of a 5120x2160 frame at
    /// depth 1 sums to 2048 x 2,764,800, past 2^32.
    #[test]
    fn the_ground_reads_sums_past_32_bits() {
        let mut c = [0u32; STAT_WORDS];
        let n = 5120u64 * 540;
        c[STAT_DEPTH_N] = (5120 * 2160) as u32;
        (c[STAT_TOP_N], c[STAT_BOTTOM_N]) = (n as u32, n as u32);
        let (top, bottom) = (n * 1000, n * 2000);
        assert!(bottom > u64::from(u32::MAX));
        (c[STAT_TOP_SUM], c[STAT_TOP_SUM_HI]) = (top as u32, (top >> 32) as u32);
        (c[STAT_BOTTOM_SUM], c[STAT_BOTTOM_SUM_HI]) = (bottom as u32, (bottom >> 32) as u32);
        assert_eq!(infer_reverse_z(&c), Some(true), "the bottom is nearer");
        // Without the high word the bottom's low 32 bits are below the top's sum.
        assert!(u64::from(bottom as u32) < top);
    }

    #[test]
    fn reversed_z_is_decided_once_enough_holds_agree() {
        let mut s = State::default();
        assert!(!s.vote_reverse_z(Some(true)));
        assert!(!s.vote_reverse_z(None), "inconclusive: no vote");
        assert!(!s.vote_reverse_z(Some(false)), "a disagreeing hold starts the streak again");
        assert!(!s.vote_reverse_z(Some(false)));
        assert_eq!(s.reverse_z, None);
        assert!(s.vote_reverse_z(Some(false)), "the third in a row decides");
        assert_eq!(s.reverse_z, Some(false));
        assert!(!s.vote_reverse_z(Some(true)), "decided: no more votes");
        assert_eq!(s.reverse_z, Some(false));
    }

    /// On the GPU: the prep pass's ground sums carry into their high words (the counters start just
    /// below 2^32, as a quarter of an 8K frame's rows reaches), and a hold without vectors
    /// (`F_NO_MVEC`) makes the motion pass write no history and count nothing, whatever the motion
    /// buffer holds.
    #[test]
    fn the_prep_sums_carry_and_a_hold_without_vectors_has_no_motion() {
        let Some(rig) = Rig::new(16, 8) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (w, h) = (rig.w, rig.h);
        let n = (w * h) as usize;
        let depth: Vec<u8> = (0..n).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        Rig::write(&rig.depth, &depth);
        let near_top = u32::MAX - 100;
        // SAFETY: host-coherent counters, idle.
        unsafe {
            let stats = rig.pass.stats.ptr.cast::<u32>();
            std::ptr::write_bytes(rig.pass.stats.ptr, 0, STAT_BYTES as usize);
            *stats.add(STAT_TOP_SUM) = near_top;
        }
        let push = rig.pass.push(F_CHAINED | F_FRESH | F_REVERSE_Z | F_COUNT_DEPTH, 1, [1.0, 1.0], 3.0);
        // SAFETY: recording into the rig's buffers; the set was bound.
        rig.one_shot(|cmd| unsafe { rig.pass.record_motion(&rig.gpu.3, cmd, &push, rig.acc.buffer) });
        let c = rig.pass.counters();
        // The top quarter (2 rows of 16) at depth 0.5: 32 x 1024 added to just below 2^32.
        let top = u64::from(c[STAT_TOP_SUM_HI]) << 32 | u64::from(c[STAT_TOP_SUM]);
        assert_eq!(top, u64::from(near_top) + 32 * 1024, "the sum carried into its high word");
        assert_eq!((c[STAT_TOP_N], c[STAT_BOTTOM_SUM_HI]), (32, 0));
        assert_eq!(c[STAT_MOTION_N], n as u32, "a hold with vectors counts every texel");
        // Without vectors: NaN everywhere, nothing counted, though the buffer holds vectors.
        let vectors: Vec<u8> = (0..n).flat_map(|_| 0x3c00_3c00u32.to_le_bytes()).collect();
        Rig::write(&rig.mvec, &vectors);
        rig.pass.clear_counters();
        let push = rig.pass.push(F_CHAINED | F_FRESH | F_NO_MVEC, 1, [1.0, 1.0], 3.0);
        // SAFETY: as above.
        rig.one_shot(|cmd| unsafe { rig.pass.record_motion(&rig.gpu.3, cmd, &push, rig.acc.buffer) });
        assert!(rig.read_f32(rig.acc.buffer, 2 * n).iter().all(|v| v.is_nan()), "no history from a hold without vectors");
        let c = rig.pass.counters();
        assert_eq!((c[STAT_STILL], c[STAT_MOTION_N]), (0, 0), "nothing counted for the cadence");
    }

    /// A handle the last bind named comes back for another object once that one is destroyed (the split hold
    /// makes a view of the colour input every hold, and a destroyed view's handle is the next one's): the set
    /// is written again when what the handles name changed ([`Bindings::made`]), and only then. Bound to a
    /// destroyed view, the cache's passes wrote wherever its descriptor pointed (GTA V Enhanced, 2026-10-20:
    /// Xid 109 after a loading screen, 4 runs of 4).
    #[test]
    fn the_set_is_written_again_for_other_objects_under_the_same_handles() {
        let Some(mut rig) = Rig::new(16, 8) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let device = rig.gpu.3.clone();
        let first = rig.pass.bound.expect("bound by the rig").0;
        // SAFETY: the rig's own pass, idle; every handle is live.
        unsafe { rig.pass.bind(&device, &first) };
        assert_eq!(rig.pass.bound.map(|(b, _)| b.made), Some([0; 3]), "the same objects: nothing to write");
        for (k, made) in [[1, 0, 0], [1, 1, 0], [1, 1, 1]].into_iter().enumerate() {
            let again = Bindings { made, ..first };
            // SAFETY: as above.
            unsafe { rig.pass.bind(&device, &again) };
            assert_eq!(rig.pass.bound.map(|(b, _)| b.made), Some(made), "another object behind the same handles ({k}): written again");
        }
        // HdrPass::bind_colour counts every view it makes, whatever handle the driver gives it.
        let (instance, physical) = (rig.gpu.1.clone(), rig.gpu.2);
        // SAFETY: a fresh pass on the rig's device; the colour image is a live RGBA16F image with storage use.
        unsafe {
            let mut hdr = super::super::hdr::HdrPass::build(&device, &instance, physical, rig.w, rig.h).expect("the HDR pass builds");
            let mut seen = Vec::new();
            for _ in 0..3 {
                assert!(hdr.bind_colour(&device, rig.colour.image));
                seen.push(hdr.colour_binding());
            }
            assert_eq!(seen, [1, 2, 3], "one more for every view made");
            // Inside DLSS's buffer the view of the staging image is made once, and again for other staging
            // images (built for another extent) even under the same handle: the destroyed image's view
            // otherwise stayed bound.
            let once = |hdr: &mut super::super::hdr::HdrPass, made: u64| {
                assert!(hdr.bind_colour_once(&device, rig.colour.image, made));
                hdr.colour_binding()
            };
            assert_eq!(once(&mut hdr, 7), 4, "after a split hold's view: one of its own");
            assert_eq!(once(&mut hdr, 7), 4, "the same staging images: the view is kept");
            assert_eq!(once(&mut hdr, 8), 5, "other staging images under the same handle: a view of them");
            hdr.destroy(&device);
        }
    }

    /// The test's GPU objects around a [`CachePass`].
    pub(crate) struct Rig {
        pub gpu: (ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32),
        pool: vk::CommandPool,
        pub w: u32,
        pub h: u32,
        pub colour: OwnImage,
        pub encoded: OwnImage,
        pub answer: OwnImage,
        pub exposure: HostBuffer,
        pub depth: HostBuffer,
        pub mvec: HostBuffer,
        pub acc: OwnBuffer,
        /// Stands for the network's input features.
        pub scratch: OwnBuffer,
        pub staging: HostBuffer,
        pub pass: CachePass,
    }

    impl Rig {
        pub fn new(w: u32, h: u32) -> Option<Self> {
            let gpu = crate::composition::gpu::test_device()?;
            let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
            let host = |bytes| {
                super::super::own_host_buffer_with(
                    device,
                    instance,
                    physical,
                    bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .unwrap()
            };
            let n = u64::from(w * h);
            // SAFETY: a fresh device; everything below is used on its queue only and waited for.
            unsafe {
                let pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .unwrap();
                let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
                let mut pass = CachePass::build(device, instance, physical, family, w, h, true).expect("the cache pass builds");
                let scratch_bytes = ScratchLayout::new(w, h).total;
                assert!(pass.chain(scratch_bytes));
                let mut rig = Self {
                    colour: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    encoded: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    answer: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    exposure: host(16),
                    depth: host(n * 4),
                    mvec: host(n * 4),
                    acc: OwnBuffer::new(
                        device,
                        instance,
                        physical,
                        n * 8,
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .unwrap(),
                    scratch: OwnBuffer::new(
                        device,
                        instance,
                        physical,
                        scratch_bytes,
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .unwrap(),
                    staging: host(n * 16),
                    pass,
                    pool,
                    w,
                    h,
                    gpu: gpu.clone(),
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                rig.one_shot(|cmd| {
                    let general = |image| {
                        vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image)
                            .subresource_range(range)
                    };
                    let all = [general(rig.colour.image), general(rig.encoded.image), general(rig.answer.image)];
                    rig.gpu.3.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &all,
                    );
                    for image in [rig.encoded.image, rig.answer.image] {
                        rig.gpu.3.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
                    }
                });
                rig.pass.bind(
                    device,
                    &Bindings {
                        colour: rig.colour.view,
                        encoded: rig.encoded.view,
                        answer: rig.answer.view,
                        exposure: rig.exposure.buffer,
                        depth: Some(rig.depth.buffer),
                        mvec: rig.mvec.buffer,
                        acc: rig.acc.buffer,
                        scratch: (rig.scratch.buffer, scratch_bytes),
                        made: [0; 3],
                    },
                );
                Some(rig)
            }
        }

        pub fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.gpu.3;
            // SAFETY: the test's own pool, queue and device; waited for.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let all = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[all],
                    &[],
                    &[],
                );
                record(cmd);
                let back = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[back],
                    &[],
                    &[],
                );
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::default().command_buffers(&[cmd])], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.gpu.4).unwrap();
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        /// Writes `bytes` (tightly packed RGBA16F at the extent) into `image`.
        pub fn upload_image(&self, image: vk::Image, bytes: &[u8]) {
            assert_eq!(bytes.len() as u64, u64::from(self.w * self.h) * 8);
            // SAFETY: the staging buffer is mapped for w * h * 16 bytes and idle.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.staging.ptr, bytes.len()) };
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: self.w, height: self.h, depth: 1 },
                ..Default::default()
            };
            // SAFETY: recording into the test's buffer; the image is in GENERAL.
            self.one_shot(|cmd| unsafe { self.gpu.3.cmd_copy_buffer_to_image(cmd, self.staging.buffer, image, vk::ImageLayout::GENERAL, &[region]) });
        }

        pub fn write(buffer: &HostBuffer, bytes: &[u8]) {
            // SAFETY: the test's host buffers are mapped for at least what is written; idle.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.ptr, bytes.len()) };
        }

        /// `count` words of a device buffer from byte `offset`.
        pub fn read_words(&self, buffer: vk::Buffer, offset: u64, count: usize) -> Vec<u32> {
            let size = count as u64 * 4;
            // SAFETY: recording into the test's buffer.
            self.one_shot(|cmd| unsafe {
                self.gpu.3.cmd_copy_buffer(cmd, buffer, self.staging.buffer, &[vk::BufferCopy { src_offset: offset, dst_offset: 0, size }])
            });
            // SAFETY: mapped and idle (waited for).
            unsafe { std::slice::from_raw_parts(self.staging.ptr.cast::<u32>(), count).to_vec() }
        }

        /// The first `count` f32 of a device buffer.
        pub fn read_f32(&self, buffer: vk::Buffer, count: usize) -> Vec<f32> {
            self.read_words(buffer, 0, count).into_iter().map(f32::from_bits).collect()
        }

        /// A scratch region of EDIT16 texels as (edit rgb, the spare half) per texel.
        pub fn read_edits(&self, region: fn(&ScratchLayout) -> Region) -> (Vec<f32>, Vec<u32>) {
            let r = region(&self.pass.layout);
            let n = (self.w * self.h) as usize;
            let words = self.read_words(self.scratch.buffer, r.offset, n * 2);
            let t = self.pass.tuning;
            let range = t.max_luma_edit + t.chroma_stops;
            let un = |q: u32| q as f32 / 65535.0 * (2.0 * range) - range;
            let edits = words.as_chunks::<2>().0.iter().flat_map(|w| [un(w[0] & 0xffff), un(w[0] >> 16), un(w[1] & 0xffff)]).collect();
            (edits, words.as_chunks::<2>().0.iter().map(|w| w[1] >> 16).collect())
        }

        /// The colour image as f32 RGBA.
        pub fn read_colour(&self) -> Vec<f32> {
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: self.w, height: self.h, depth: 1 },
                ..Default::default()
            };
            // SAFETY: recording into the test's buffer; the image is in GENERAL.
            self.one_shot(|cmd| unsafe {
                self.gpu.3.cmd_copy_image_to_buffer(cmd, self.colour.image, vk::ImageLayout::GENERAL, self.staging.buffer, &[region])
            });
            // SAFETY: mapped and idle.
            let halves = unsafe { std::slice::from_raw_parts(self.staging.ptr.cast::<u16>(), (self.w * self.h * 4) as usize) };
            halves.iter().map(|&h| super::super::f16_to_f32(h)).collect()
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let d = &self.gpu.3;
            // SAFETY: every submission was waited for.
            unsafe {
                self.pass.destroy(d);
                for i in [&self.colour, &self.encoded, &self.answer] {
                    i.destroy(d);
                }
                for b in [&self.exposure, &self.depth, &self.mvec, &self.staging] {
                    b.destroy(d);
                }
                self.acc.destroy(d);
                self.scratch.destroy(d);
                d.destroy_command_pool(self.pool, None);
            }
        }
    }

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// A fresh directory for one call: numbered, because tests run in parallel in one process and two
    /// of them can ask for the same name (both carry tests replay "depth-edge" without the depth
    /// image); the second would otherwise remove the first one's files while it reads them.
    fn scratch(name: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join("target/test-scratch")
            .join(format!("edit-cache-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Runs `scripts/cache_replay.py` with `args`; `false` (the test skips) without python3 and numpy.
    fn replay_tool(args: &[&str]) -> bool {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("scripts/cache_replay.py");
        match std::process::Command::new("python3").arg(&script).args(args).output() {
            Ok(out) if out.status.success() => true,
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(err.contains("No module named 'numpy'"), "cache_replay.py {args:?} failed: {err}");
                false
            }
            Err(_) => false,
        }
    }

    /// One synthetic kind through the shaders: hold 0 refreshes (its edit captured from the dump's
    /// answer), every later hold is cached; returns per cached hold (t, the shown edit RGB, the
    /// weight, the accumulated field) from the GPU and the tool's own.
    #[allow(clippy::type_complexity)]
    fn carry_on_gpu(
        kind: &str,
        depth_image: bool,
    ) -> Option<(Rig, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>, Vec<f32>)> {
        let dir = scratch(&format!("{kind}-{depth_image}"));
        let dump = dir.join("dump");
        let out = dir.join("carry");
        if !replay_tool(&["--make-synthetic", dump.to_str()?, "--kind", kind])
            || !replay_tool(&[dump.to_str()?, "--export-carry", out.to_str()?, "--reverse-z", "yes"])
        {
            eprintln!("python3 with numpy is not available: the shader-against-tool test is skipped");
            return None;
        }
        let holds: Vec<[f32; 3]> = std::fs::read_to_string(out.join("holds.txt"))
            .unwrap()
            .lines()
            .map(|l| {
                let v: Vec<f32> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
                [v[0], v[1], v[2]]
            })
            .collect();
        let read = |name: String| std::fs::read(dump.join(name)).unwrap();
        // The synthetic dumps are 96 x 64.
        let (w, h) = (96u32, 64u32);
        let n = (w * h) as usize;
        let Some(rig) = Rig::new(w, h) else {
            eprintln!("no Vulkan device: skipped");
            return None;
        };
        assert_eq!(holds[0][2], 1.5, "the synthetic exposure");
        Rig::write(&rig.exposure, &0x3e00u32.to_le_bytes());
        let d = &rig.gpu.3;
        let mut results = Vec::new();
        let mut src_edit = Vec::new();
        // The split hold copies the depth aspect of DLSS's depth image into the layer's buffer; the
        // hold inside DLSS's buffer copies it into a slot buffer there, which the capture copies over.
        let depth = depth_image.then(|| DepthImage::new(&rig));
        for (t, hj) in holds.iter().enumerate() {
            rig.upload_image(rig.colour.image, &read(format!("hold-{t:02}/colour.rgba16f")));
            Rig::write(&rig.mvec, &read(format!("hold-{t:02}/mvec.rg16f")));
            let depth_bytes = read(format!("hold-{t:02}/depth.r32f"));
            match &depth {
                Some(image) => image.route_through(&rig, &depth_bytes),
                None => Rig::write(&rig.depth, &depth_bytes),
            }
            let mut flags = F_CHAINED | F_NEAREST | F_REVERSE_Z | F_KEEP_CLAMPED;
            if t <= 1 {
                flags |= F_FRESH;
            }
            let mut push = rig.pass.push(flags, 1, [1.0, 1.0], 3.0);
            if t > 0 {
                push.step_jitter = [holds[t - 1][0] - hj[0], holds[t - 1][1] - hj[1]];
                push.refresh_jitter = [holds[0][0] - hj[0], holds[0][1] - hj[1]];
            }
            // SAFETY (both): recording into the test's buffer; bound in `Rig::new`.
            rig.one_shot(|cmd| unsafe { rig.pass.record_motion(d, cmd, &push, rig.acc.buffer) });
            if t == 0 {
                rig.upload_image(rig.answer.image, &read("hold-00/answer.rgba16f".to_string()));
                let capture = CachePush { flags: push.flags | F_SCENE_ANSWER, ..push };
                rig.one_shot(|cmd| unsafe { rig.pass.record_capture(d, cmd, &capture) });
                src_edit = rig.read_edits(|l| l.src_edit).0;
                continue;
            }
            // SAFETY: as above. The history read first on its own: the guided filter's grids take its
            // place in the whole write-back (which reads it again, the same).
            rig.one_shot(|cmd| unsafe { rig.pass.record_history(d, cmd, &push) });
            let (history, weight) = rig.read_edits(|l| l.hist);
            let weight: Vec<f32> = weight.into_iter().map(|q| q as f32 / 65535.0).collect();
            // SAFETY: as above.
            rig.one_shot(|cmd| unsafe { rig.pass.record_apply(d, cmd, &push) });
            let edit = rig.read_edits(|l| l.shown).0;
            let acc = rig.read_f32(rig.acc.buffer, n * 2);
            let tool = |name: &str| f32s(&std::fs::read(out.join(format!("{name}-{t:02}.f32"))).unwrap());
            results.push((t, edit, weight, acc, tool("cache"), tool("weight"), tool("acc"), history));
            // The picture written back, in the edit's place for the comparison between the paths.
            let colour = rig.read_colour();
            results.last_mut().unwrap().4.extend(colour);
        }
        if let Some(image) = depth {
            image.destroy(&rig);
        }
        let _ = std::fs::remove_dir_all(&dir);
        Some((rig, results, src_edit))
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| if x.is_nan() && y.is_nan() { 0.0 } else { (x - y).abs() }).fold(0.0, f32::max)
    }

    /// The shaders give the replay tool's numbers (cache_common.py, `--chain recurrence`) on its
    /// synthetic dumps, within 1e-3 stop: the shown edit, the accumulated motion and the weight. And
    /// on each kind what it is for: a still scene reproduces the refresh's edit; a pan keeps it
    /// aligned, its field the composed motion; a moving box's uncovered background rejects the
    /// carried high band; a still view under sub-pixel jitter reads each texel's edit whole from the
    /// refresh's nearest texel (no bilinear blend of a near-pixel-scale edit).
    #[test]
    fn the_shaders_carry_the_edit_as_the_replay_tool_does() {
        for kind in ["still", "pan", "depth-edge", "jitter"] {
            let Some((rig, results, src_edit)) = carry_on_gpu(kind, false) else { return };
            let (w, h) = (rig.w as usize, rig.h as usize);
            let n = w * h;
            for (t, edit, weight, acc, t_edit, t_weight, t_acc, _) in &results {
                let t_edit = &t_edit[..n * 3];
                let de = max_diff(edit, t_edit);
                let dw = max_diff(weight, t_weight);
                let da = max_diff(acc, t_acc);
                eprintln!("{kind} hold {t}: edit max diff {de:.2e} stop, weight {dw:.2e}, motion {da:.2e} px");
                assert!(de <= 1e-3, "{kind} hold {t}: the shown edit differs from the tool's by {de} stop");
                assert!(da <= 1e-4, "{kind} hold {t}: the accumulated motion differs by {da} px");
                assert!(dw <= 1e-3, "{kind} hold {t}: the weight differs by {dw}");
            }
            let src = &src_edit;
            match kind {
                "still" => {
                    for (t, edit, ..) in &results {
                        let d = max_diff(edit, src);
                        assert!(d < 1e-4, "still hold {t}: the cached edit is the refresh's within {d}");
                    }
                }
                "pan" => {
                    // 2.5 x 1 px a hold: after three cached holds the field is the composed motion.
                    let (t, _, _, acc, ..) = &results[2];
                    assert_eq!(*t, 3);
                    for y in 4..h - 4 {
                        for x in 4..w - 8 {
                            let a = &acc[(y * w + x) * 2..][..2];
                            assert!((a[0] - 7.5).abs() < 1e-5 && (a[1] - 3.0).abs() < 1e-5, "pan field at {x},{y}: {a:?}");
                        }
                    }
                    // Aligned: hold 1 moved 2.5 px right and 1 down, with a jitter change of 0.5: each
                    // interior texel reads the refresh's edit from 3 px right and 1 down, unchanged
                    // (the history read, before the band split and the guided filter).
                    let (_, _, weight, .., edit) = &results[0];
                    let mut worst = 0.0f32;
                    for y in 4..h - 4 {
                        for x in 4..w - 8 {
                            let i = y * w + x;
                            assert!(weight[i] > 0.9, "pan weight at {x},{y}: {}", weight[i]);
                            let s = ((y + 1) * w + x + 3) * 3;
                            worst = worst.max(max_diff(&edit[i * 3..i * 3 + 3], &src[s..s + 3]));
                        }
                    }
                    assert!(worst < 1e-3, "pan: the carried edit is off its source by up to {worst} stop");
                }
                "jitter" => {
                    // Nothing moves: the refresh's texel nearest to where this hold's pixel samples
                    // (the jitters of cache_replay.py's synthetic_holds).
                    const JITTER: [[f32; 2]; 6] = [[0.0, 0.0], [0.3, -0.2], [-0.35, 0.15], [0.1, 0.4], [-0.2, -0.3], [0.45, 0.05]];
                    let refresh = JITTER[0];
                    for (t, .., history) in &results {
                        let j = JITTER[*t];
                        let mut worst = 0.0f32;
                        for y in 2..h - 2 {
                            for x in 2..w - 2 {
                                let sx = (x as f32 + 0.5 + refresh[0] - j[0]).floor() as usize;
                                let sy = (y as f32 + 0.5 + refresh[1] - j[1]).floor() as usize;
                                let i = y * w + x;
                                let s = (sy * w + sx) * 3;
                                worst = worst.max(max_diff(&history[i * 3..i * 3 + 3], &src[s..s + 3]));
                            }
                        }
                        assert!(worst < 1e-4, "jitter hold {t}: the history read is off the nearest texel's edit by {worst} stop");
                    }
                }
                "depth-edge" => {
                    // Hold 1: the box (rows 20..44) moved from x 10 to 15; columns 10..15 show the
                    // background it covered on the refresh hold.
                    let (t, _, weight, ..) = &results[0];
                    assert_eq!(*t, 1);
                    for y in 21..43 {
                        for x in 11..14 {
                            assert!(weight[y * w + x] < 0.1, "uncovered background at {x},{y} keeps weight {}", weight[y * w + x]);
                        }
                        for x in 20..30 {
                            assert!(weight[y * w + x] > 0.9, "the box at {x},{y} lost its edit: {}", weight[y * w + x]);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    /// The shaders run the replay tool's cadence (`--export-cadence`: the model one hold in 3, the
    /// guided filter by staleness): every hold's edit (a refresh's own, a cached hold's shown one)
    /// within 1e-3 stop of the tool's. `regrade`'s model brightens the picture 0.15 stop a hold.
    #[test]
    fn the_shaders_run_the_cadence_as_the_replay_tool_does() {
        const INTERVAL: usize = 3;
        for kind in ["still", "pan", "depth-edge", "regrade", "jitter"] {
            let dir = scratch(&format!("cadence-{kind}"));
            let dump = dir.join("dump");
            let out = dir.join("cadence");
            if !replay_tool(&["--make-synthetic", dump.to_str().unwrap(), "--kind", kind])
                || !replay_tool(&[dump.to_str().unwrap(), "--cadence", "3", "--export-cadence", out.to_str().unwrap(), "--reverse-z", "yes"])
            {
                eprintln!("python3 with numpy is not available: the shader-against-tool test is skipped");
                return;
            }
            let lines: Vec<Vec<f64>> =
                std::fs::read_to_string(out.join("holds.txt")).unwrap().lines().map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect()).collect();
            let holds: Vec<[f32; 2]> = lines.iter().map(|v| [v[0] as f32, v[1] as f32]).collect();
            let read = |name: String| std::fs::read(dump.join(name)).unwrap();
            let (w, h) = (96u32, 64u32);
            let Some(rig) = Rig::new(w, h) else {
                eprintln!("no Vulkan device: skipped");
                return;
            };
            Rig::write(&rig.exposure, &0x3e00u32.to_le_bytes());
            let d = &rig.gpu.3;
            for (t, hj) in holds.iter().enumerate() {
                rig.upload_image(rig.colour.image, &read(format!("hold-{t:02}/colour.rgba16f")));
                Rig::write(&rig.mvec, &read(format!("hold-{t:02}/mvec.rg16f")));
                Rig::write(&rig.depth, &read(format!("hold-{t:02}/depth.r32f")));
                let since = t % INTERVAL;
                let refresh = since == 0;
                let mut flags = F_CHAINED | F_NEAREST | F_REVERSE_Z | F_KEEP_CLAMPED;
                if t == 0 || since == 1 {
                    flags |= F_FRESH;
                }
                let mut push = rig.pass.push(flags, 1, [1.0, 1.0], 3.0);
                if t > 0 {
                    let r = holds[t - since];
                    push.step_jitter = [holds[t - 1][0] - hj[0], holds[t - 1][1] - hj[1]];
                    push.refresh_jitter = [r[0] - hj[0], r[1] - hj[1]];
                }
                rig.pass.clear_counters();
                // SAFETY (all): recording into the test's buffer; bound in `Rig::new`.
                rig.one_shot(|cmd| unsafe { rig.pass.record_motion(d, cmd, &push, rig.acc.buffer) });
                // The adaptive cadence's counts (still_counts), within a texel or two of the tool's.
                let c = rig.pass.counters();
                let counts = [c[STAT_STILL], c[STAT_MOTION_N]];
                for (k, (&gpu, &tool)) in counts.iter().zip(&lines[t][2..4]).enumerate() {
                    assert!((f64::from(gpu) - tool).abs() <= 2.0, "{kind} hold {t}: still count {k} is {gpu}, the tool's {tool}");
                }
                if refresh {
                    rig.upload_image(rig.answer.image, &read(format!("hold-{t:02}/answer.rgba16f")));
                    let capture = CachePush { flags: push.flags | F_SCENE_ANSWER, ..push };
                    rig.one_shot(|cmd| unsafe { rig.pass.record_capture(d, cmd, &capture) });
                } else {
                    rig.one_shot(|cmd| unsafe { rig.pass.record_apply(d, cmd, &push) });
                }
                let edit = rig.read_edits(if refresh { |l: &ScratchLayout| l.src_edit } else { |l: &ScratchLayout| l.shown }).0;
                let tool = f32s(&std::fs::read(out.join(format!("shown-{t:02}.f32"))).unwrap());
                let de = max_diff(&edit, &tool);
                eprintln!("{kind} hold {t}: shown edit max diff {de:.2e} stop");
                assert!(de <= 1e-3, "{kind} hold {t} ({}): the shown edit differs from the tool's by {de} stop", if refresh { "refresh" } else { "cached" });
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A D32_SFLOAT_S8_UINT image standing for DLSS's depth, and a slot buffer as the hold inside
    /// DLSS's buffer has.
    struct DepthImage {
        image: vk::Image,
        memory: vk::DeviceMemory,
        slot: HostBuffer,
    }

    impl DepthImage {
        fn new(rig: &Rig) -> Self {
            let (instance, physical, d) = (&rig.gpu.1, rig.gpu.2, &rig.gpu.3);
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .extent(vk::Extent3D { width: rig.w, height: rig.h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: the test's device; nothing submitted yet.
            unsafe {
                let image = d.create_image(&info, None).unwrap();
                let reqs = d.get_image_memory_requirements(image);
                let props = instance.get_physical_device_memory_properties(physical);
                let index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
                let memory = d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None).unwrap();
                d.bind_image_memory(image, memory, 0).unwrap();
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);
                rig.one_shot(|cmd| {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[general],
                    )
                });
                let slot = super::super::own_host_buffer_with(
                    d,
                    instance,
                    physical,
                    u64::from(rig.w * rig.h) * 4,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .unwrap();
                Self { image, memory, slot }
            }
        }

        /// `bytes` (f32 device depth) into the image's depth aspect, then into the layer's depth
        /// buffer the way the hold inside DLSS's buffer does (image -> slot buffer -> layer's buffer).
        fn route_through(&self, rig: &Rig, bytes: &[u8]) {
            let d = &rig.gpu.3;
            Rig::write(&rig.staging, bytes);
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: rig.w, height: rig.h, depth: 1 },
                ..Default::default()
            };
            let size = bytes.len() as u64;
            // SAFETY: recording into the test's buffers; the image is in GENERAL.
            rig.one_shot(|cmd| unsafe { d.cmd_copy_buffer_to_image(cmd, rig.staging.buffer, self.image, vk::ImageLayout::GENERAL, &[region]) });
            rig.one_shot(|cmd| unsafe { d.cmd_copy_image_to_buffer(cmd, self.image, vk::ImageLayout::GENERAL, self.slot.buffer, &[region]) });
            rig.one_shot(|cmd| unsafe { d.cmd_copy_buffer(cmd, self.slot.buffer, rig.depth.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]) });
        }

        fn destroy(&self, rig: &Rig) {
            let d = &rig.gpu.3;
            // SAFETY: every submission was waited for.
            unsafe {
                self.slot.destroy(d);
                d.destroy_image(self.image, None);
                d.free_memory(self.memory, None);
            }
        }
    }

    /// The hold inside DLSS's buffer gets its depth through the game's image and a slot buffer, the
    /// split hold as given: the same inputs give the same picture, bit for bit, through the same passes.
    #[test]
    fn the_hold_inside_dlss_buffer_writes_the_same_picture_as_the_split_hold() {
        let Some((_, split, _)) = carry_on_gpu("depth-edge", false) else { return };
        let Some((_, inline, _)) = carry_on_gpu("depth-edge", true) else { return };
        assert_eq!(split.len(), inline.len());
        for (a, b) in split.iter().zip(&inline) {
            assert!(a.1 == b.1 && a.2 == b.2, "hold {}: the shown edit or its weight differ", a.0);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(bits(&a.4), bits(&b.4), "hold {}: the picture written back differs", a.0);
        }
    }

    /// The edit cache's write-backs stay within the finite half range at a low exposure. The refresh's
    /// capture takes the decode's rule: at 5e-5 a fully clipped answer (1.0) decodes past 65504, and as
    /// +Inf it would read as no light (`to_v`) and store a darkening edit for the cached holds; it stores
    /// the brightening one (+1.5 stops, the luma limit). A cached hold of the same picture at half the
    /// exposure (twice the scene value, the same in v) carries that edit onto 40000, past the half range:
    /// it stores 65504, not +Inf. (Lavapipe's float-to-half conversion rounds toward zero, so it lands on
    /// 65504 there anyway: this pins what every driver must give, NVIDIA's round-to-nearest one included.)
    #[test]
    fn the_cache_writes_back_finite_halves_at_a_low_exposure() {
        let (w, h) = (8u32, 8u32);
        let Some(rig) = Rig::new(w, h) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let n = (w * h) as usize;
        let half = crate::preupscale::tests::f32_to_f16;
        let grey = |v: f32| -> Vec<u8> { (0..n).flat_map(|_| [v, v, v, 1.0]).flat_map(|c| half(c).to_le_bytes()).collect() };
        let exposure = |e: f32| Rig::write(&rig.exposure, &u32::from(half(e)).to_le_bytes());
        Rig::write(&rig.depth, &vec![0u8; n * 4]);
        Rig::write(&rig.mvec, &vec![0u8; n * 4]);
        let d = &rig.gpu.3;
        let push = rig.pass.push(F_CHAINED | F_FRESH | F_KEEP_CLAMPED, 0, [1.0, 1.0], 3.0);
        // The refresh hold: the answer clipped at 1.0 over a grey of 20000 (v = 1/3).
        exposure(5e-5);
        rig.upload_image(rig.colour.image, &grey(20000.0));
        rig.upload_image(rig.answer.image, &grey(1.0));
        // SAFETY (all): recording into the test's buffer; bound in `Rig::new`.
        rig.one_shot(|cmd| unsafe { rig.pass.record_motion(d, cmd, &push, rig.acc.buffer) });
        rig.one_shot(|cmd| unsafe { rig.pass.record_capture(d, cmd, &push) });
        let (edit, _) = rig.read_edits(|l| l.src_edit);
        let limit = rig.pass.tuning.max_luma_edit;
        assert!(edit.iter().all(|e| (e - limit).abs() < 1e-3), "the refresh stores the brightening edit (+{limit} stops), got {:?}", &edit[..3]);
        // The cached hold.
        exposure(2.5e-5);
        rig.upload_image(rig.colour.image, &grey(40000.0));
        rig.one_shot(|cmd| unsafe { rig.pass.record_motion(d, cmd, &push, rig.acc.buffer) });
        rig.one_shot(|cmd| unsafe { rig.pass.record_apply(d, cmd, &push) });
        let colour = rig.read_colour();
        for (i, px) in colour.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(&px[..3], &[65504.0; 3], "the cached hold at texel {i} stores the largest finite half");
        }
    }
}
