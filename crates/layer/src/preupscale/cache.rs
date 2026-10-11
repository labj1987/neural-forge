//! The pre-upscaler edit cache (docs/ARCHITECTURE.md 4.10): with `model_interval` N above 1 the
//! network runs on one hold in N (a refresh) and the holds between (cached holds) reuse its
//! correction, carried along the game's motion and multiplied onto their own fresh frame.
//!
//! The GPU side is eight compute passes (`shaders/preupscale_cache_*.comp`), the step-for-step mirror
//! of `scripts/cache_common.py` (the numpy reference the offline replay measures with):
//!
//! - in the capture (`C`) of every hold: `prep` (this hold's fingerprint, pseudo-linear depth, the
//!   depth's reversed-Z counts) and `motion` (the accumulated motion
//!   back to the previous network frame, which the network's history reads too). At interval 1 only
//!   `motion` runs, copying DLSS's vectors as they are, so the network's input is bit for bit what it
//!   was;
//! - in a refresh hold's write-back (`W`), right before the decode: `capture` (the edit the decode is
//!   about to write, against the frame, with this hold's depth and fingerprint, and where the shown
//!   edit walks from: what was on screen on the hold before); right after it: `refresh` (this hold's
//!   step of the walk over the decode's picture, and the edit shown kept for the next refresh);
//! - as a cached hold's whole write-back, in place of the network and the decode: `history` (the
//!   refresh's edit read at the chained position, depth-admitted, checked against this hold),
//!   `reduce` three times (the coverage-weighted pyramid), `apply` (band split), `guided` three
//!   times at half size (a guided filter of that edit on this hold's log luma), `apply` again (the
//!   filter blended in by how far each pixel moved, written back with the decode's rules).
//!
//! One [`CachePass`] per [`super::Resources`] (one extent), built on the first native hold. Its
//! buffers beyond the accumulated field ([`Store`]) exist only while the interval is above 1.

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
const REFRESH_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_refresh.spv");
const GUIDED_SPV: &[u8] = include_bytes!("../../shaders/preupscale_cache_guided.spv");

const PREP: usize = 0;
const MOTION: usize = 1;
const CAPTURE: usize = 2;
const HISTORY: usize = 3;
const REDUCE: usize = 4;
const APPLY: usize = 5;
const REFRESH: usize = 6;
const GUIDED: usize = 7;
const SPV: [&[u8]; 8] = [PREP_SPV, MOTION_SPV, CAPTURE_SPV, HISTORY_SPV, REDUCE_SPV, APPLY_SPV, REFRESH_SPV, GUIDED_SPV];

/// The shaders' `F_*` flags.
pub(crate) const F_REVERSE_Z: u32 = 1;
pub(crate) const F_NEAREST: u32 = 2;
pub(crate) const F_FRESH: u32 = 4;
pub(crate) const F_CHAINED: u32 = 8;
/// A refresh walks from the last shown edit (the hold before showed one).
pub(crate) const F_WALK: u32 = 16;
pub(crate) const F_DEBUG_MASK: u32 = 32;
pub(crate) const F_KEEP_CLAMPED: u32 = 64;
pub(crate) const F_COUNT_DEPTH: u32 = 256;
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
const STAT_WORDS: usize = 16;
const STAT_BYTES: u64 = STAT_WORDS as u64 * 4;

/// The highest interval the cache serves (`ShmHeader::model_interval`'s bound).
pub(crate) const MAX_INTERVAL: u32 = 4;

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
    /// `float walk`: how far the shown edit has walked from the refresh's start to its answer
    /// ([`Plan::walk`]).
    pub walk: f32,
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

/// The cache's buffers beyond the accumulated field, at the render extent: built when the interval
/// first goes above 1, freed when it is back at 1.
pub(crate) struct Store {
    acc_prev: OwnBuffer,
    lin: OwnBuffer,
    print: OwnBuffer,
    src_edit: OwnBuffer,
    src_print: OwnBuffer,
    pub(crate) hist: OwnBuffer,
    conf: OwnBuffer,
    levels: OwnBuffer,
    src_from: OwnBuffer,
    pub(crate) shown: OwnBuffer,
    step: OwnBuffer,
    guide_a: OwnBuffer,
    guide_b: OwnBuffer,
    stale: OwnBuffer,
    stale_prev: OwnBuffer,
}

impl Store {
    /// # Safety
    /// `device` is live.
    unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32) -> Option<Self> {
        let n = u64::from(width) * u64::from(height);
        let half = u64::from(width.div_ceil(2)) * u64::from(height.div_ceil(2));
        let levels: u64 = (0..3).map(|l| level_size(width, height, l)).map(|(w, h)| u64::from(w) * u64::from(h)).sum();
        let sizes = [n * 8, n * 4, n * 16, n * 16, n * 16, n * 16, n * 4, levels * 32, n * 16, n * 16, n * 8, half * 32, half * 32, n * 4, n * 4];
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let mut made: Vec<OwnBuffer> = Vec::with_capacity(sizes.len());
        for bytes in sizes {
            // SAFETY: forwarded.
            match unsafe { OwnBuffer::new(device, instance, physical_device, bytes, usage) } {
                Some(b) => made.push(b),
                None => {
                    for b in &made {
                        // SAFETY: nothing was submitted with them.
                        unsafe { b.destroy(device) };
                    }
                    return None;
                }
            }
        }
        let mut it = made.into_iter();
        let mut next = || it.next().expect("one buffer per size");
        Some(Self { acc_prev: next(), lin: next(), print: next(), src_edit: next(), src_print: next(), hist: next(), conf: next(), levels: next(), src_from: next(), shown: next(), step: next(), guide_a: next(), guide_b: next(), stale: next(), stale_prev: next() })
    }

    fn all(&self) -> [&OwnBuffer; 15] {
        [
            &self.acc_prev,
            &self.lin,
            &self.print,
            &self.src_edit,
            &self.src_print,
            &self.hist,
            &self.conf,
            &self.levels,
            &self.src_from,
            &self.shown,
            &self.step,
            &self.guide_a,
            &self.guide_b,
            &self.stale,
            &self.stale_prev,
        ]
    }

    /// # Safety
    /// Nothing submitted may still use them.
    unsafe fn destroy(&self, device: &ash::Device) {
        for b in self.all() {
            // SAFETY: forwarded.
            unsafe { b.destroy(device) };
        }
    }
}

/// What a hold binds: the frame's images and buffers, which change from hold to hold.
#[derive(Clone, Copy, Debug)]
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
    /// `Session::seen` of the last hold that went through (consecutive holds: the next is this + 1).
    last_seen: Option<u64>,
    /// The depth's convention, decided from the first counts (`infer_reverse_z`).
    reverse_z: Option<bool>,
    /// The kind of the last submitted hold whose counters are still to be read.
    pending: Option<Kind>,
}

/// What a hold brings to [`State::plan`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hold {
    /// `model_interval`, 1..=[`MAX_INTERVAL`].
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
}

/// What [`State::plan`] decided for a hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Plan {
    pub kind: Kind,
    /// The push constants' flags and jitters for this hold's passes ([`CachePass::push`]).
    pub flags: u32,
    pub step_jitter: [f32; 2],
    pub refresh_jitter: [f32; 2],
    /// How far the shown edit has walked from the refresh's start (what was on screen) to its
    /// answer: 1/interval on the refresh, (n + 1)/interval on its n-th cached hold, so it arrives on
    /// the interval's last hold (`run_cadence`'s crossfade over the whole interval).
    pub walk: f32,
}

impl State {
    /// Drops the stored edit: the next hold refreshes.
    pub(crate) fn drop_edit(&mut self) {
        self.valid = false;
        self.chain_live = false;
    }

    /// Whether this hold is cached or refreshes, and its passes' flags. Takes the stored edit out.
    pub(crate) fn plan(&mut self, h: &Hold) -> Plan {
        let consecutive = self.last_seen.is_some_and(|s| s.wrapping_add(1) == h.hold);
        let had = std::mem::take(&mut self.valid);
        let chain_live = std::mem::take(&mut self.chain_live) && consecutive;
        let jitter = h.jitter.unwrap_or_default();
        let chained = h.interval > 1;
        // The hold before showed an edit of this chain, and this one can read it (or the refresh's).
        let continues = chained && had && consecutive && h.jitter.is_some() && h.mvec && h.depth == self.refresh_depth && h.history_ok && !h.armed;
        let cached = continues && self.since_refresh + 1 < h.interval;
        let mut flags = F_KEEP_CLAMPED;
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
            if continues && !cached {
                flags |= F_WALK;
            }
        }
        let n = h.interval.max(1) as f32;
        let walk = if cached { ((self.since_refresh + 2) as f32 / n).min(1.0) } else { 1.0 / n };
        let delta = |a: [f32; 2]| [a[0] - jitter[0], a[1] - jitter[1]];
        Plan {
            kind: if cached { Kind::Cached } else { Kind::Refresh },
            flags,
            step_jitter: if consecutive { delta(self.prev_jitter) } else { [0.0; 2] },
            refresh_jitter: delta(self.refresh_jitter),
            walk,
        }
    }

    /// The hold `h` planned as `plan` went through (everything submitted).
    pub(crate) fn went_through(&mut self, h: &Hold, plan: &Plan) {
        self.last_seen = Some(h.hold);
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

/// `infer_reverse_z` from the prep pass's counts: the cleared value first (a real share of texels
/// at exactly 0, few at 1, is reversed-Z), else the ground (depth growing toward the bottom is
/// reversed-Z; means of the top and bottom quarters, where the replay tool takes medians). `None`
/// when the counts cannot tell.
pub(crate) fn infer_reverse_z(counts: &[u32; STAT_WORDS]) -> Option<bool> {
    let n = counts[STAT_DEPTH_N];
    if n == 0 {
        return None;
    }
    let (z0, z1) = (f64::from(counts[STAT_Z0]) / f64::from(n), f64::from(counts[STAT_Z1]) / f64::from(n));
    if z0.max(z1) > 0.005 && (z0 > 2.0 * z1 || z1 > 2.0 * z0) {
        return Some(z0 > z1);
    }
    let (tn, bn) = (counts[STAT_TOP_N], counts[STAT_BOTTOM_N]);
    if tn == 0 || bn == 0 {
        return None;
    }
    let top = f64::from(counts[STAT_TOP_SUM]) / f64::from(tn);
    let bottom = f64::from(counts[STAT_BOTTOM_SUM]) / f64::from(bn);
    (top != bottom).then_some(bottom > top)
}

/// The pass for one extent: eight pipelines over one descriptor set, the counters, the timers.
pub(crate) struct CachePass {
    width: u32,
    height: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: [vk::Pipeline; 8],
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    stats: HostBuffer,
    /// Bound where a buffer of [`Store`] or the depth is not there (the shaders do not read it then).
    dummy: OwnBuffer,
    pub(crate) store: Option<Store>,
    /// GPU timestamps around the passes in the capture, in the write-back before (or in place of)
    /// the decode, and after the decode.
    timers: [Option<crate::gpu_timer::GpuTimer>; 3],
    pub(crate) state: State,
    pub(crate) tuning: Tuning,
}

// SAFETY: plain handles and the layer's own mapping, only used behind the device's `State` mutex.
unsafe impl Send for CachePass {}

/// Descriptor types of bindings 0..=22 (the shaders' layout).
fn binding_type(n: u32) -> vk::DescriptorType {
    match n {
        0 | 1 | 15 => vk::DescriptorType::STORAGE_IMAGE,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}
const BINDINGS: u32 = 23;

impl CachePass {
    /// # Safety
    /// `device` is live; `family` is the queue family the holds are submitted on.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, width: u32, height: u32) -> Option<Self> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS)
            .map(|n| vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(binding_type(n)).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build())
            .collect();
        let mut p = Partial { device, set_layout: vk::DescriptorSetLayout::null(), pipeline_layout: vk::PipelineLayout::null(), pipelines: Vec::new(), pool: vk::DescriptorPool::null(), stats: None, dummy: None };
        // SAFETY: valid create infos; everything made is owned by `p` until it moves into the result.
        unsafe {
            p.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<CachePush>() as u32).build();
            p.pipeline_layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&p.set_layout)).push_constant_ranges(std::slice::from_ref(&push)), None)
                .ok()?;
            for spv in SPV {
                let pipeline = hdr::compute_pipeline(device, p.pipeline_layout, spv)?;
                p.pipelines.push(pipeline);
            }
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 3 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: BINDINGS - 3 },
            ];
            p.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).ok()?;
            let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(p.pool).set_layouts(std::slice::from_ref(&p.set_layout))).ok()?[0];
            p.stats = Some(super::own_host_buffer_with(device, instance, physical_device, STAT_BYTES, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?);
            p.dummy = Some(OwnBuffer::new(device, instance, physical_device, 256, vk::BufferUsageFlags::STORAGE_BUFFER)?);
            let stats = p.stats.take()?;
            let dummy = p.dummy.take()?;
            std::ptr::write_bytes(stats.ptr, 0, STAT_BYTES as usize);
            let pipelines: [vk::Pipeline; 8] = std::mem::take(&mut p.pipelines).try_into().ok()?;
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
                store: None,
                timers: [
                    crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family),
                    crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family),
                    crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family),
                ],
                state: State::default(),
                tuning: Tuning::default(),
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    /// Builds the [`Store`] if it is not built. `false` when it cannot be.
    ///
    /// # Safety
    /// `device` is live; nothing of the pass is pending.
    pub(crate) unsafe fn ensure_store(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
        if self.store.is_none() {
            // SAFETY: forwarded.
            self.store = unsafe { Store::build(device, instance, physical_device, self.width, self.height) };
        }
        self.store.is_some()
    }

    /// Frees the [`Store`] (the interval is back at 1) and drops the edit.
    ///
    /// # Safety
    /// Nothing of the pass is pending.
    pub(crate) unsafe fn free_store(&mut self, device: &ash::Device) {
        if let Some(store) = self.store.take() {
            // SAFETY: forwarded.
            unsafe { store.destroy(device) };
            self.state.drop_edit();
        }
    }

    /// Points the set at this hold's resources.
    ///
    /// # Safety
    /// Nothing of the pass is pending; every handle is live.
    pub(crate) unsafe fn bind(&self, device: &ash::Device, b: &Bindings) {
        let dummy = self.dummy.buffer;
        let s = self.store.as_ref();
        let pick = |f: fn(&Store) -> vk::Buffer| s.map_or(dummy, f);
        let buffers: [(u32, vk::Buffer); 20] = [
            (2, b.exposure),
            (3, b.depth.unwrap_or(dummy)),
            (4, b.mvec),
            (5, b.acc),
            (6, pick(|s| s.acc_prev.buffer)),
            (7, pick(|s| s.lin.buffer)),
            (8, pick(|s| s.print.buffer)),
            (9, pick(|s| s.src_edit.buffer)),
            (10, pick(|s| s.src_print.buffer)),
            (11, pick(|s| s.hist.buffer)),
            (12, pick(|s| s.conf.buffer)),
            (13, pick(|s| s.levels.buffer)),
            (14, self.stats.buffer),
            (16, pick(|s| s.src_from.buffer)),
            (17, pick(|s| s.shown.buffer)),
            (18, pick(|s| s.step.buffer)),
            (19, pick(|s| s.guide_a.buffer)),
            (20, pick(|s| s.guide_b.buffer)),
            (21, pick(|s| s.stale.buffer)),
            (22, pick(|s| s.stale_prev.buffer)),
        ];
        let images = [(0u32, b.colour), (1, b.encoded), (15, b.answer)];
        let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers.iter().map(|&(_, buffer)| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> =
            images.iter().map(|&(_, view)| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }]).collect();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(BINDINGS as usize);
        for ((n, _), info) in buffers.iter().zip(&buffer_infos) {
            writes.push(vk::WriteDescriptorSet::builder().dst_set(self.set).dst_binding(*n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info).build());
        }
        for ((n, _), info) in images.iter().zip(&image_infos) {
            writes.push(vk::WriteDescriptorSet::builder().dst_set(self.set).dst_binding(*n).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(info).build());
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
            walk: 1.0,
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
        let b = vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).build();
        // SAFETY: forwarded.
        unsafe { device.cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[b], &[], &[]) };
    }

    /// # Safety
    /// `cmd` is recording.
    unsafe fn to_host(device: &ash::Device, cmd: vk::CommandBuffer) {
        let b = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ).build();
        // SAFETY: forwarded.
        unsafe { device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[b], &[], &[]) };
    }

    /// Records the capture's part, after its copies and the encode: at interval 1 (`push.flags`
    /// without [`F_CHAINED`]) the vectors into the accumulated field as they are; else the prep pass,
    /// the previous field kept, and the motion pass.
    ///
    /// # Safety
    /// `cmd` is recording; [`Self::bind`] was called for this hold with `acc`; with [`F_CHAINED`] the
    /// store is built.
    pub(crate) unsafe fn record_motion(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush, acc: vk::Buffer) {
        let chained = push.flags & F_CHAINED != 0;
        // SAFETY: forwarded.
        unsafe {
            Self::compute_barrier(device, cmd, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE);
            if let Some(t) = &self.timers[0] {
                t.record_start(device, cmd);
            }
            if chained {
                let store = self.store.as_ref().expect("the store is built for a chained hold");
                let bytes = u64::from(self.width) * u64::from(self.height) * 8;
                self.dispatch(device, cmd, PREP, push, self.width, self.height);
                // The previous field and staleness kept for the motion pass to read (the network's binding is
                // the field itself).
                let kept = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[kept], &[], &[]);
                device.cmd_copy_buffer(cmd, acc, store.acc_prev.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
                device.cmd_copy_buffer(cmd, store.stale.buffer, store.stale_prev.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes / 2 }]);
                Self::compute_barrier(device, cmd, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE);
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
    /// `cmd` is recording; [`Self::bind`] was called for this hold; the store is built.
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

    /// Records the refresh hold's step of the walk, in the write-back right after the decode.
    ///
    /// # Safety
    /// `cmd` is recording, after the decode; [`Self::record_capture`] was recorded for this hold.
    pub(crate) unsafe fn record_refresh(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &CachePush) {
        // SAFETY: forwarded.
        unsafe {
            // The decode wrote the colour input this pass reads and writes.
            Self::compute_barrier(device, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
            if let Some(t) = &self.timers[2] {
                t.record_start(device, cmd);
            }
            self.dispatch(device, cmd, REFRESH, push, self.width, self.height);
            if let Some(t) = &self.timers[2] {
                t.record_end(device, cmd);
            }
        }
    }

    /// Records a cached hold's write-back: the history read, the pyramid, the band split, the guided
    /// filter at half size, and its blend written over the colour input.
    ///
    /// # Safety
    /// `cmd` is recording, after the write-back's opening barrier; [`Self::bind`] was called for this
    /// hold; the store is built.
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

    /// Marks the timers whose passes were submitted (`capture`: the capture's part; `writeback`: the
    /// write-back's; `refresh`: the refresh pass after the decode).
    pub(crate) fn submitted(&mut self, capture: bool, writeback: bool, refresh: bool) {
        for (t, on) in self.timers.iter_mut().zip([capture, writeback, refresh]) {
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

    /// Clears the counters before a hold's passes are submitted.
    pub(crate) fn clear_counters(&self) {
        // SAFETY: as `counters`; nothing pending writes them (after the hold's wait).
        unsafe { std::ptr::write_bytes(self.stats.ptr, 0, STAT_BYTES as usize) };
    }

    /// Reads what the last submitted hold's passes counted (its fence was waited on): the stored
    /// edit is dropped when the refresh's exposure was unusable or a cached hold could not apply,
    /// and the depth's convention is decided once.
    pub(crate) fn after_fence(&mut self) -> Outcome {
        let Some(kind) = self.state.pending.take() else { return Outcome::default() };
        let decided = self.state.reverse_z.is_some();
        let c = self.counters();
        let mut out = Outcome::default();
        if !decided {
            if let Some(rev) = infer_reverse_z(&c) {
                crate::log!("[preupscale] edit cache: depth is {} ({} of {} texels at 0, {} at 1)", if rev { "reversed-Z" } else { "forward" }, c[STAT_Z0], c[STAT_DEPTH_N], c[STAT_Z1]);
                self.state.reverse_z = Some(rev);
                // The stored edit's depth was read with the assumed convention.
                self.state.drop_edit();
            }
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
            if let Some(store) = &self.store {
                store.destroy(device);
            }
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
        // uvec2 size, uint flags, uint depth_kind, vec2 x3, float, uint, eleven floats: 92 bytes, std430.
        assert_eq!(std::mem::size_of::<CachePush>(), 92);
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
        assert_eq!(std::mem::offset_of!(CachePush, walk), 84);
        assert_eq!(std::mem::offset_of!(CachePush, guided_eps), 88);
        for spv in SPV {
            assert!(spv.len() > 20 && spv.len().is_multiple_of(4), "embedded SPIR-V");
        }
        assert_eq!(depth_kind(vk::Format::D32_SFLOAT_S8_UINT), 1);
        assert_eq!(depth_kind(vk::Format::D24_UNORM_S8_UINT), 2);
        assert_eq!(depth_kind(vk::Format::D16_UNORM), 3);
        assert_eq!(level_size(1485, 836, 0), (372, 209));
        assert_eq!(level_size(1485, 836, 2), (24, 14));
    }

    fn hold(h: u64) -> Hold {
        Hold { interval: 2, hold: h, jitter: Some([0.25, -0.25]), mvec: true, depth: true, history_ok: true, armed: false, debug_mask: false }
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
        // No jitter, no vectors, depth gone, an armed dump.
        for odd in [Hold { jitter: None, ..hold(2) }, Hold { mvec: false, ..hold(2) }, Hold { depth: false, ..hold(2) }, Hold { armed: true, ..hold(2) }] {
            let mut s = State::default();
            assert_eq!(kinds(&mut s, &[hold(1), odd]), [R, R], "{odd:?}");
        }
        // A refresh without jitter stores nothing to carry.
        let mut s = State::default();
        assert_eq!(kinds(&mut s, &[Hold { jitter: None, ..hold(1) }, hold(2)]), [R, R]);
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

    /// The walk: a refresh right after a hold of the chain walks from it (1/interval of the way), its
    /// cached holds go on to the answer on the interval's last hold; the first refresh, one after a
    /// reset, and interval 1 do not walk.
    #[test]
    fn a_refresh_walks_from_the_hold_before_across_the_interval() {
        let four = |h| Hold { interval: 4, ..hold(h) };
        let mut s = State::default();
        let mut seen = Vec::new();
        for h in 1..=6 {
            let plan = s.plan(&four(h));
            s.went_through(&four(h), &plan);
            seen.push((plan.kind, plan.flags & F_WALK != 0, plan.walk));
        }
        use Kind::{Cached as C, Refresh as R};
        assert_eq!(seen, [(R, false, 0.25), (C, false, 0.5), (C, false, 0.75), (C, false, 1.0), (R, true, 0.25), (C, false, 0.5)]);
        // A reset: the refresh after it takes the answer at once.
        let p = s.plan(&Hold { history_ok: false, ..four(7) });
        assert_eq!((p.kind, p.flags & F_WALK), (R, 0));
        // At interval 2 the cached hold arrives.
        let two = |h| Hold { interval: 2, ..hold(h) };
        let mut s = State::default();
        let walks: Vec<(bool, f32)> = (1..=4)
            .map(|h| {
                let plan = s.plan(&two(h));
                s.went_through(&two(h), &plan);
                (plan.flags & F_WALK != 0, plan.walk)
            })
            .collect();
        assert_eq!(walks, [(false, 0.5), (false, 1.0), (true, 0.5), (false, 1.0)]);
        let one = State::default().plan(&Hold { interval: 1, ..hold(1) });
        assert_eq!((one.flags & F_WALK, one.walk), (0, 1.0));
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
        pub staging: HostBuffer,
        pub pass: CachePass,
    }

    impl Rig {
        pub fn new(w: u32, h: u32) -> Option<Self> {
            let gpu = crate::composition::gpu::test_device()?;
            let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
            let host = |bytes| {
                super::super::own_host_buffer_with(device, instance, physical, bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap()
            };
            let n = u64::from(w * h);
            // SAFETY: a fresh device; everything below is used on its queue only and waited for.
            unsafe {
                let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
                let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
                let mut pass = CachePass::build(device, instance, physical, family, w, h).expect("the cache pass builds");
                assert!(pass.ensure_store(device, instance, physical));
                let rig = Self {
                    colour: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    encoded: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    answer: OwnImage::new(device, instance, physical, w, h, usage).unwrap(),
                    exposure: host(16),
                    depth: host(n * 4),
                    mvec: host(n * 4),
                    acc: OwnBuffer::new(device, instance, physical, n * 8, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap(),
                    staging: host(n * 16),
                    pass,
                    pool,
                    w,
                    h,
                    gpu: gpu.clone(),
                };
                let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
                rig.one_shot(|cmd| {
                    let general = |image| {
                        vk::ImageMemoryBarrier::builder()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(image)
                            .subresource_range(range)
                            .build()
                    };
                    let all = [general(rig.colour.image), general(rig.encoded.image), general(rig.answer.image)];
                    rig.gpu.3.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &all);
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
                    },
                );
                Some(rig)
            }
        }

        pub fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.gpu.3;
            // SAFETY: the test's own pool, queue and device; waited for.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let all = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[all], &[], &[]);
                record(cmd);
                let back = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ).build();
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[back], &[], &[]);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
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

        /// The first `count` f32 of a device buffer.
        pub fn read_f32(&self, buffer: vk::Buffer, count: usize) -> Vec<f32> {
            let size = count as u64 * 4;
            // SAFETY: recording into the test's buffer.
            self.one_shot(|cmd| unsafe { self.gpu.3.cmd_copy_buffer(cmd, buffer, self.staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]) });
            // SAFETY: mapped and idle (waited for).
            unsafe { std::slice::from_raw_parts(self.staging.ptr.cast::<f32>(), count).to_vec() }
        }

        /// The colour image as f32 RGBA.
        pub fn read_colour(&self) -> Vec<f32> {
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: self.w, height: self.h, depth: 1 },
                ..Default::default()
            };
            // SAFETY: recording into the test's buffer; the image is in GENERAL.
            self.one_shot(|cmd| unsafe { self.gpu.3.cmd_copy_image_to_buffer(cmd, self.colour.image, vk::ImageLayout::GENERAL, self.staging.buffer, &[region]) });
            // SAFETY: mapped and idle.
            let halves = unsafe { std::slice::from_raw_parts(self.staging.ptr.cast::<u16>(), (self.w * self.h * 4) as usize) };
            halves.iter().map(|&h| super::super::f16_to_f32(h)).collect()
        }

        pub fn store_buffer(&self, pick: fn(&Store) -> vk::Buffer) -> vk::Buffer {
            pick(self.pass.store.as_ref().expect("built"))
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
                d.destroy_command_pool(self.pool, None);
            }
        }
    }

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/test-scratch").join(format!("edit-cache-{name}-{}", std::process::id()));
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
    fn carry_on_gpu(kind: &str, depth_image: bool) -> Option<(Rig, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>, Vec<f32>)> {
        let dir = scratch(&format!("{kind}-{depth_image}"));
        let dump = dir.join("dump");
        let out = dir.join("carry");
        if !replay_tool(&["--make-synthetic", dump.to_str()?, "--kind", kind]) || !replay_tool(&[dump.to_str()?, "--export-carry", out.to_str()?, "--reverse-z", "yes"]) {
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
                src_edit = rig.read_f32(rig.store_buffer(|s| s.src_edit.buffer), n * 4);
                continue;
            }
            // SAFETY: as above.
            rig.one_shot(|cmd| unsafe { rig.pass.record_apply(d, cmd, &push) });
            let hist = rig.read_f32(rig.store_buffer(|s| s.hist.buffer), n * 4);
            let shown = rig.read_f32(rig.store_buffer(|s| s.shown.buffer), n * 4);
            let acc = rig.read_f32(rig.acc.buffer, n * 2);
            let edit: Vec<f32> = shown.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
            let weight: Vec<f32> = hist.chunks_exact(4).map(|c| c[3]).collect();
            let tool = |name: &str| f32s(&std::fs::read(out.join(format!("{name}-{t:02}.f32"))).unwrap());
            let history: Vec<f32> = hist.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
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
    /// carried high band.
    #[test]
    fn the_shaders_carry_the_edit_as_the_replay_tool_does() {
        for kind in ["still", "pan", "depth-edge"] {
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
            let src: Vec<f32> = src_edit.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
            match kind {
                "still" => {
                    for (t, edit, ..) in &results {
                        let d = max_diff(edit, &src);
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
    /// shown edit walking from what was on screen to each refresh's answer, soft refresh): every
    /// hold's shown edit, refresh holds included, within 1e-3 stop of the tool's. `regrade` is where
    /// the walk shows: its model brightens the picture 0.15 stop a hold.
    #[test]
    fn the_shaders_walk_as_the_replay_tool_does() {
        const INTERVAL: usize = 3;
        for kind in ["still", "pan", "depth-edge", "regrade"] {
            let dir = scratch(&format!("cadence-{kind}"));
            let dump = dir.join("dump");
            let out = dir.join("cadence");
            if !replay_tool(&["--make-synthetic", dump.to_str().unwrap(), "--kind", kind])
                || !replay_tool(&[dump.to_str().unwrap(), "--cadence", "3", "--export-cadence", out.to_str().unwrap(), "--reverse-z", "yes"])
            {
                eprintln!("python3 with numpy is not available: the shader-against-tool test is skipped");
                return;
            }
            let holds: Vec<[f32; 2]> = std::fs::read_to_string(out.join("holds.txt"))
                .unwrap()
                .lines()
                .map(|l| {
                    let v: Vec<f32> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
                    [v[0], v[1]]
                })
                .collect();
            let read = |name: String| std::fs::read(dump.join(name)).unwrap();
            let (w, h) = (96u32, 64u32);
            let n = (w * h) as usize;
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
                if refresh && t > 0 {
                    flags |= F_WALK;
                }
                let mut push = rig.pass.push(flags, 1, [1.0, 1.0], 3.0);
                push.walk = if refresh { 1.0 / INTERVAL as f32 } else { ((since + 1) as f32 / INTERVAL as f32).min(1.0) };
                if t > 0 {
                    let r = holds[t - since];
                    push.step_jitter = [holds[t - 1][0] - hj[0], holds[t - 1][1] - hj[1]];
                    push.refresh_jitter = [r[0] - hj[0], r[1] - hj[1]];
                }
                // SAFETY (all): recording into the test's buffer; bound in `Rig::new`.
                rig.one_shot(|cmd| unsafe { rig.pass.record_motion(d, cmd, &push, rig.acc.buffer) });
                if refresh {
                    rig.upload_image(rig.answer.image, &read(format!("hold-{t:02}/answer.rgba16f")));
                    let capture = CachePush { flags: push.flags | F_SCENE_ANSWER, ..push };
                    rig.one_shot(|cmd| unsafe {
                        rig.pass.record_capture(d, cmd, &capture);
                        rig.pass.record_refresh(d, cmd, &capture);
                    });
                } else {
                    rig.one_shot(|cmd| unsafe { rig.pass.record_apply(d, cmd, &push) });
                }
                let shown = rig.read_f32(rig.store_buffer(|s| s.shown.buffer), n * 4);
                let edit: Vec<f32> = shown.chunks_exact(4).flat_map(|c| [c[0], c[1], c[2]]).collect();
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
            let info = vk::ImageCreateInfo::builder()
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
                let memory = d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None).unwrap();
                d.bind_image_memory(image, memory, 0).unwrap();
                let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
                let general = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range)
                    .build();
                rig.one_shot(|cmd| d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[general]));
                let slot = super::super::own_host_buffer_with(d, instance, physical, u64::from(rig.w * rig.h) * 4, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap();
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
}
