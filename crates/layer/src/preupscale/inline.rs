//! The hold inside DLSS's own command buffer (2.0.4; recorded into it since 3.8.1).
//!
//! The split hold (`super::submit_around`) runs the model in front of DLSS's launch buffer. That is
//! only correct when nothing in the buffer before DLSS's first colour launch produces or
//! synchronizes the colour input. Crimson Desert and Cyberpunk 2077 render the frame in the same
//! buffer (draws, dispatches and write barriers before DLSS's first launch), so the split is
//! refused there ([`super::Hazard`]). This module holds at the launch itself instead, by recording the
//! whole hold into the application's buffer at record time ([`Inline::record_outcome`]), right before the
//! buffer's first launch that names the colour input (or right after DLSS's output kernel, the passes placed
//! after DLSS, [`super::placement`]):
//!
//! 1. the colour input copied (or, from R11G11B10, blitted) into the stage's RGBA16F staging image, with
//!    DLSS's exposure, motion vectors and depth beside it ([`Staging`], [`record_copy_in`]);
//! 2. the hold itself, recorded by the caller's closure into the same buffer: the native backend's capture
//!    and encode, the network (`super::native`), the edit cache's passes, the decode into the staging image;
//! 3. the staging image copied back over the colour input ([`record_copy_back`]);
//! 4. the application's compute state bound again ([`super::restore`]): the hold's dispatches bound the
//!    layer's pipelines, descriptor sets and push constants, and vkd3d-proton does not bind its own again.
//!
//! Nothing waits on the host and nothing runs on another queue: the hold is part of DLSS's buffer and runs
//! when the GPU reaches it, in the buffer's own order. Several of DLSS's buffers can be recorded before the
//! first of them runs; each holds a slot ([`MAX_SLOTS`]) until it can no longer run: re-recorded, reset,
//! freed, or its pool reset or destroyed (each reaches the layer, and Vulkan forbids each while the buffer is
//! pending). A slot carries what is read back once its buffer has run ([`Harvest`]): GPU timestamps and the
//! counters the CPU's bookkeeping reads (`super::Session::harvest`). What the GPU state of one hold assumes
//! about the hold before it is checked on the GPU (`preupscale_hold_guard.comp`), so a buffer recorded and
//! never submitted, or buffers run out of the order they were recorded in, only cost the history.
//!
//! A buffer the split can take (GTA V's) gets nothing recorded: [`super::Tracker::inline_point`] only
//! answers where the split would be refused. Nor does a secondary command buffer, one from a pool of another
//! queue family than DLSS's graphics one, one recorded for simultaneous use, one whose compute state cannot
//! be restored, or any while the model is switched off (F11).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use ash::vk;

use super::placement::Stage;
use super::{lock, Aux, InlinePoint};

/// Slots per device: DLSS's buffers recorded with a hold whose holds may still run. A slot is free again only
/// once its buffer cannot be submitted again unchanged: re-recorded, reset, freed, or its pool reset or
/// destroyed; never by time.
pub(crate) const MAX_SLOTS: usize = 8;

/// The timestamps a hold writes into its slot's pool: after the copy in, after the capture, after the
/// network, after the write-back.
pub(crate) const TIMESTAMPS: u32 = 4;

/// What a hold copies into its slot's readback buffer at its end, and where.
pub(crate) const READBACK_BYTES: u64 = 256;
/// The edit cache's counters (`cache::STAT_WORDS` words).
pub(crate) const READ_STATS: u64 = 0;
/// The scene-cut detector's result (4 words).
pub(crate) const READ_SCENE: u64 = 64;
/// The exposure the encode used (the HDR pass's exposure buffer's first 4 bytes).
pub(crate) const READ_EXPOSURE: u64 = 80;
/// The auto-exposure's state (`hdr::AutoState`'s 5 words).
pub(crate) const READ_AUTO: u64 = 96;

/// `NEURAL_FORGE_INLINE=off` turns the hold inside DLSS's buffer off; the split hold is unaffected.
pub(crate) fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_INLINE").is_none_or(|v| v != "off" && v != "0"));
    *ON
}

/// The images and buffer one stage's holds work in, at the colour input's extent: the colour input
/// (RGBA16F, what the hold reads and writes), DLSS's exposure (1x1 R16F), its motion vectors (RG16F) and
/// depth (4 bytes a texel at most). Shared by every hold of the stage: their buffers run on DLSS's queue,
/// one after another.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Staging {
    pub extent: (u32, u32),
    pub colour: vk::Image,
    colour_memory: vk::DeviceMemory,
    pub exposure: vk::Image,
    exposure_memory: vk::DeviceMemory,
    pub mvec: vk::Image,
    mvec_memory: vk::DeviceMemory,
    pub depth: vk::Buffer,
    depth_memory: vk::DeviceMemory,
}

/// One slot: the buffer it belongs to and what its hold left to read back.
struct Slot {
    owner: Option<vk::CommandBuffer>,
    stage: Stage,
    /// Reserved while a hold is being recorded into `owner` (not yet a hold).
    reserved: bool,
    recorded: Option<Instant>,
    /// Submissions of the owner since it was recorded (each ran the hold once).
    submits: u32,
    queries: vk::QueryPool,
    readback: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_ptr: *const u8,
    /// A cache dump's readbacks for an armed hold (`super::cache_dump`), built when one is armed.
    cache_dump: Option<super::cache_dump::Buffers>,
    meta: Option<Meta>,
}

// SAFETY: plain handles and the layer's own mapping, only used behind `Inline::slots`.
unsafe impl Send for Slot {}

/// What a hold recorded into a buffer says about itself, for what is read back once the buffer ran.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Meta {
    pub stage: Stage,
    /// The resources it was recorded with (`super::Resources::id`): readings of other ones are dropped.
    pub resources: u64,
    pub identification: u64,
    /// What the readback buffer holds.
    pub stats: bool,
    pub scene: Option<super::scene_cut::Recorded>,
    pub auto: bool,
    /// The edit cache's plan for the hold.
    pub kind: Option<super::cache::Kind>,
    /// The native pass (its id) the hold needs initialised, and the command buffer that does (submitted once,
    /// in front of the first buffer that runs a hold of the pass: [`Inline::prologue`]).
    pub init: Option<(u64, vk::CommandBuffer)>,
    /// The later passes' histories the hold took out of `UNDEFINED` (the pass's id, one bit per later pass).
    pub fresh: (u64, u32),
    pub exposure_source: Option<super::ExposureSource>,
    /// A cache dump's armed hold: what it recorded.
    pub cache_dump: Option<super::cache_dump::Pending>,
    /// The hold wrote timestamps.
    pub timed: bool,
}

/// A slot's hold, read back after its buffer ran.
pub(crate) struct Harvest {
    pub meta: Meta,
    /// How many times the buffer ran the hold.
    pub runs: u32,
    /// GPU milliseconds: the capture, the network (with the copies between its passes), the write-back.
    pub gpu_ms: [Option<f32>; 3],
    pub readback: [u8; READBACK_BYTES as usize],
    pub cache_frame: Option<super::cache_dump::Frame>,
}

#[derive(Default)]
struct Slots {
    slots: Vec<Slot>,
    /// The slots each buffer holds, in the order its holds were recorded (the GPU reaches them in that order):
    /// one before the upscaler and one after DLSS's output at most ([`super::placement`]).
    by_cb: HashMap<vk::CommandBuffer, Vec<usize>>,
    simultaneous: HashSet<vk::CommandBuffer>,
    /// The pool each of the device's command buffers was allocated from: resetting or destroying a
    /// pool ends every one of its buffers' pending state, so their slots are free again.
    /// vkd3d-proton resets a DX12 command allocator's pool and records into other buffers in the
    /// meantime, so a slot freed only at the same buffer's next `vkBeginCommandBuffer` stayed taken:
    /// all eight were in use within a few frames (Black Myth: Wukong's benchmark, 2026-10-05).
    pool_of: HashMap<vk::CommandBuffer, vk::CommandPool>,
    /// The queue family of each of the device's command pools (`vkCreateCommandPool`).
    pool_family: HashMap<vk::CommandPool, u32>,
    /// The device's secondary command buffers (`vkAllocateCommandBuffers`' level).
    secondary: HashSet<vk::CommandBuffer>,
    /// Each stage's staging images.
    staging: [Option<Staging>; 2],
    /// The queue the last buffer carrying a hold was submitted on.
    queue: Option<vk::Queue>,
    /// The native passes whose initialisation went to the queue ([`Inline::prologue`]).
    initialised: HashSet<u64>,
}

fn stage_index(stage: Stage) -> usize {
    usize::from(stage == Stage::Upscaled)
}

impl Slots {
    /// Holds of `stage` that may still run, besides slot `except`.
    fn outstanding(&self, stage: Stage, except: Option<usize>) -> usize {
        self.slots.iter().enumerate().filter(|(i, s)| Some(*i) != except && s.owner.is_some() && s.stage == stage).count()
    }
}

/// What [`Inline::record_outcome`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Recorded {
    /// The hold is recorded.
    Held,
    /// Not recorded this time, for a reason that passes.
    Skipped,
    /// Not recorded, for a reason that stays with this buffer or image.
    Refused,
}

/// What the caller's closure gets to record a hold into the application's buffer with.
pub(crate) struct Ctx<'a> {
    pub cb: vk::CommandBuffer,
    /// The hold point, its auxiliary inputs replaced by the staging copies the hold reads (in `GENERAL`).
    pub point: InlinePoint,
    /// The colour input's own point (what [`Self::open`] copies from).
    source: InlinePoint,
    pub staging: Staging,
    /// The depth format copied into the staging buffer, if one is.
    pub depth: Option<vk::Format>,
    /// The slot's index: the network's descriptor pool for this buffer (`nn::HOLD_SLOTS`).
    pub slot: u32,
    pub queries: vk::QueryPool,
    pub readback: vk::Buffer,
    /// A cache dump's readbacks, for a hold armed for one (built by [`Self::cache_dump_buffers`]).
    pub cache_dump: &'a mut Option<super::cache_dump::Buffers>,
    /// Holds of this stage recorded into other buffers that may still run: while there are any, nothing
    /// they use may change (`super::Frozen`).
    pub outstanding: usize,
    convert: bool,
    exposure: Option<(vk::Image, vk::ImageLayout)>,
    mvec: Option<(vk::Image, vk::ImageLayout, (u32, u32))>,
    depth_source: Option<(vk::Image, vk::ImageLayout, vk::Format)>,
    device: &'a ash::Device,
    opened: bool,
    closed: bool,
}

impl Ctx<'_> {
    /// Records the copies in (step 1): the first command of the hold. Everything that could still fail
    /// must be decided before.
    ///
    /// # Safety
    /// [`Self::cb`] is being recorded on this thread (the launch hook).
    pub(crate) unsafe fn open(&mut self) {
        if std::mem::replace(&mut self.opened, true) {
            return;
        }
        let s = self.staging;
        // SAFETY: the contract; every handle is live.
        unsafe {
            record_copy_in(
                self.device,
                self.cb,
                (self.source.colour, self.source.layout, self.convert),
                s.colour,
                s.extent,
                self.exposure.map(|(image, layout)| (image, layout, s.exposure)),
                self.mvec.map(|(image, layout, extent)| (image, layout, s.mvec, extent)),
                self.depth_source.map(|(image, layout, format)| (image, layout, format, s.depth)),
            );
            if self.queries != vk::QueryPool::null() {
                self.device.cmd_reset_query_pool(self.cb, self.queries, 0, TIMESTAMPS);
                self.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::TRANSFER, self.queries, 0);
            }
        }
    }

    /// Writes timestamp `k` (1: after the capture, 2: after the network) once everything before finished.
    ///
    /// # Safety
    /// As [`Self::open`], which was called.
    pub(crate) unsafe fn stamp(&self, k: u32) {
        if self.queries != vk::QueryPool::null() && self.opened {
            // SAFETY: the contract.
            unsafe { self.device.cmd_write_timestamp(self.cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.queries, k) };
        }
    }

    /// Records the end of the hold (step 3): the last timestamp and the staging image copied back over the
    /// colour input.
    ///
    /// # Safety
    /// As [`Self::open`], which was called.
    pub(crate) unsafe fn close(&mut self) {
        if !self.opened || std::mem::replace(&mut self.closed, true) {
            return;
        }
        // SAFETY: the contract.
        unsafe {
            self.stamp(3);
            record_copy_back(self.device, self.cb, (self.source.colour, self.source.layout, self.convert), self.staging.colour, self.staging.extent);
        }
    }
}

pub(crate) struct Inline {
    device: Arc<ash::Device>,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// The game's graphics family (where DLSS's buffers are submitted).
    app_family: u32,
    /// Whether `B10G11R11_UFLOAT_PACK32` (optimal tiling) can be blitted from and to: the conversion
    /// to and from the RGBA16F staging image for an R11G11B10 colour input.
    blit_r11g11b10: bool,
    /// The timestamps' tick in nanoseconds and valid bits (`None`: the family writes none).
    ticks: Option<(f64, u64)>,
    slots: Mutex<Slots>,
    harvest: Mutex<Vec<Harvest>>,
    /// Holds recorded and not yet released, on every stage: the network's loader rebuilds nothing they use
    /// while there are any (`native::Loader`).
    outstanding: Arc<AtomicUsize>,
    /// The hold's buffers were submitted on more than one queue: no more holds are recorded (their work
    /// shares the stage's resources and must run one after another).
    stopped: AtomicBool,
    said: Mutex<HashSet<String>>,
    /// The shared memory's header once the device's state opened it ([`watch_switch`]): the model's
    /// switch (F11, the GUI), read at every launch so nothing is recorded while it is off.
    switch: AtomicPtr<neural_forge_protocol::ShmHeader>,
    /// The application's compute state per command buffer ([`super::restore`]).
    pub(crate) restore: Arc<super::restore::Tracker>,
}

static INLINES: LazyLock<Mutex<HashMap<vk::Device, Arc<Inline>>>> = LazyLock::new(Default::default);

impl Inline {
    /// The hold inside DLSS's buffers on `device`, whose buffers run on `app_family`; `outstanding` counts the
    /// holds recorded and not yet released (shared with the network's loader).
    pub(crate) fn new(
        device: Arc<ash::Device>, instance: &ash::Instance, physical_device: vk::PhysicalDevice, app_family: u32, outstanding: Arc<AtomicUsize>, restore: super::restore::Fns,
    ) -> Arc<Self> {
        // SAFETY: `physical_device` is the device's own.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let blit = vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST;
        // SAFETY: as above.
        let blit_r11g11b10 = unsafe { instance.get_physical_device_format_properties(physical_device, vk::Format::B10G11R11_UFLOAT_PACK32) }
            .optimal_tiling_features
            .contains(blit)
            // SAFETY: as above.
            && unsafe { instance.get_physical_device_format_properties(physical_device, vk::Format::R16G16B16A16_SFLOAT) }.optimal_tiling_features.contains(blit);
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        // SAFETY: as above.
        let period = unsafe { instance.get_physical_device_properties(physical_device) }.limits.timestamp_period;
        let ticks = families
            .get(app_family as usize)
            .map(|f| f.timestamp_valid_bits)
            .filter(|&bits| bits > 0 && period.is_finite() && period > 0.0)
            .map(|bits| (f64::from(period), if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }));
        let inline = Arc::new(Self {
            device: device.clone(),
            memory,
            app_family,
            blit_r11g11b10,
            ticks,
            slots: Mutex::new(Slots::default()),
            harvest: Mutex::new(Vec::new()),
            outstanding,
            stopped: AtomicBool::new(false),
            said: Mutex::new(HashSet::new()),
            switch: AtomicPtr::new(std::ptr::null_mut()),
            restore: Arc::new(super::restore::Tracker::new(restore)),
        });
        lock(&INLINES).insert(device.handle(), inline.clone());
        crate::log!("[preupscale] hold inside DLSS's command buffer available: recorded into it, on family {app_family} (NEURAL_FORGE_INLINE=off turns it off)");
        crate::logging::flush();
        inline
    }

    /// The game's graphics family, where DLSS's buffers (and so the holds) run.
    pub(crate) fn app_family(&self) -> u32 {
        self.app_family
    }

    fn say_once(&self, what: String) {
        if lock(&self.said).insert(what.clone()) {
            crate::log!("[preupscale] {what}");
            crate::logging::flush();
        }
    }

    /// `vkBeginCommandBuffer`: the buffer is not pending, so a slot it held is free again (and its hold read
    /// back), and nothing is bound in it.
    pub(crate) fn begin(&self, cb: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) {
        self.restore.clear(cb);
        let mut s = lock(&self.slots);
        self.release(&mut s, cb);
        if flags.contains(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE) {
            s.simultaneous.insert(cb);
        } else {
            s.simultaneous.remove(&cb);
        }
    }

    /// `vkFreeCommandBuffers`.
    pub(crate) fn free(&self, cbs: &[vk::CommandBuffer]) {
        let mut s = lock(&self.slots);
        for cb in cbs {
            self.restore.forget(*cb);
            self.release(&mut s, *cb);
            s.simultaneous.remove(cb);
            s.pool_of.remove(cb);
            s.secondary.remove(cb);
        }
    }

    /// `vkCreateCommandPool` made `pool` for queue family `family`.
    pub(crate) fn pool_created(&self, pool: vk::CommandPool, family: u32) {
        lock(&self.slots).pool_family.insert(pool, family);
    }

    /// `vkAllocateCommandBuffers` returned `cbs` of `level` from `pool`.
    pub(crate) fn allocated(&self, pool: vk::CommandPool, level: vk::CommandBufferLevel, cbs: &[vk::CommandBuffer]) {
        let mut s = lock(&self.slots);
        for cb in cbs {
            s.pool_of.insert(*cb, pool);
            if level == vk::CommandBufferLevel::SECONDARY {
                s.secondary.insert(*cb);
            } else {
                s.secondary.remove(cb);
            }
        }
    }

    /// `vkResetCommandBuffer`: the buffer is not pending (Vulkan requires it), so its slot is free.
    pub(crate) fn reset(&self, cb: vk::CommandBuffer) {
        self.restore.clear(cb);
        let mut s = lock(&self.slots);
        self.release(&mut s, cb);
    }

    /// `vkResetCommandPool` (`destroyed`: `vkDestroyCommandPool`): none of the pool's buffers is
    /// pending (Vulkan requires it), so their slots are free.
    pub(crate) fn reset_pool(&self, pool: vk::CommandPool, destroyed: bool) {
        let mut s = lock(&self.slots);
        let members: Vec<vk::CommandBuffer> = s.pool_of.iter().filter(|(_, p)| **p == pool).map(|(cb, _)| *cb).collect();
        for cb in &members {
            if destroyed {
                self.restore.forget(*cb);
            } else {
                self.restore.clear(*cb);
            }
            self.release(&mut s, *cb);
        }
        if destroyed {
            for cb in members {
                s.pool_of.remove(&cb);
                s.simultaneous.remove(&cb);
                s.secondary.remove(&cb);
            }
            s.pool_family.remove(&pool);
        }
    }

    /// `vkCmdExecuteCommands` into `primary`: every state of the primary is undefined after it (the
    /// application binds again).
    pub(crate) fn executed(&self, primary: vk::CommandBuffer) {
        self.restore.clear(primary);
    }

    /// Whether `cb` carries a hold.
    #[cfg(test)]
    pub(crate) fn owns(&self, cb: vk::CommandBuffer) -> bool {
        lock(&self.slots).by_cb.contains_key(&cb)
    }

    /// Whether `cb` carries a hold before the upscaler (the split path then leaves it alone; one carrying only
    /// the hold after DLSS's output is still split in front of).
    pub(crate) fn owns_before(&self, cb: vk::CommandBuffer) -> bool {
        let s = lock(&self.slots);
        s.by_cb.get(&cb).is_some_and(|slots| slots.iter().any(|&i| s.slots[i].stage == Stage::Before && s.slots[i].meta.is_some()))
    }

    /// Whether the model is switched off (F11, the GUI, `shmctl set enabled 0`), as far as known: before
    /// the device's state opened the shared memory nothing is known, and the hold is recorded.
    pub(crate) fn model_switched_off(&self) -> bool {
        let header = self.switch.load(Ordering::Acquire);
        // SAFETY: a header [`watch_switch`] stored: mapped for the life of the process.
        !header.is_null() && unsafe { !((*header).neural_enabled() && (*header).apply_model.load(Ordering::Relaxed) != 0) }
    }

    /// The holds read back since the last call, in the order their buffers were released.
    pub(crate) fn take_harvest(&self) -> Vec<Harvest> {
        std::mem::take(&mut *lock(&self.harvest))
    }

    /// What has to run in front of a submission of `cbs` about to be made: the initialisation of each native
    /// pass a hold in them was recorded with, the first time (the GPU then runs it before those holds, and
    /// before every later hold of the pass, which are submitted after it). The caller submits them on the
    /// same queue first, and calls [`Self::prologue_failed`] with them if that fails.
    pub(crate) fn prologue(&self, cbs: impl IntoIterator<Item = vk::CommandBuffer>) -> Vec<(u64, vk::CommandBuffer)> {
        let mut s = lock(&self.slots);
        let mut out: Vec<(u64, vk::CommandBuffer)> = Vec::new();
        for cb in cbs {
            let Some(held) = s.by_cb.get(&cb).cloned() else { continue };
            for i in held {
                if let Some((pass, cmd)) = s.slots[i].meta.and_then(|m| m.init) {
                    if s.initialised.insert(pass) {
                        out.push((pass, cmd));
                    }
                }
            }
        }
        out
    }

    /// The submission of [`Self::prologue`]'s buffers failed: they are due again.
    pub(crate) fn prologue_failed(&self, inits: &[(u64, vk::CommandBuffer)]) {
        let mut s = lock(&self.slots);
        for (pass, _) in inits {
            s.initialised.remove(pass);
        }
    }

    /// A submission the next layer accepted, on `queue`, of `cbs`: each hold in them runs once more. Holds
    /// submitted on another queue than the last ones stop the holds (their work shares the stage's resources).
    pub(crate) fn submitted(&self, queue: vk::Queue, cbs: impl IntoIterator<Item = vk::CommandBuffer>) {
        let mut s = lock(&self.slots);
        let mut any = false;
        for cb in cbs {
            let Some(held) = s.by_cb.get(&cb).cloned() else { continue };
            for i in held {
                if s.slots[i].meta.is_some() {
                    s.slots[i].submits += 1;
                    any = true;
                }
            }
        }
        if !any {
            return;
        }
        match s.queue {
            Some(q) if q != queue => {
                if !self.stopped.swap(true, Ordering::Relaxed) {
                    drop(s);
                    crate::log!("[preupscale] DLSS's buffers with a hold inside are submitted on more than one queue; no more holds are recorded into them (their work would overlap)");
                    crate::logging::flush();
                }
            }
            _ => s.queue = Some(queue),
        }
    }

    /// Records a hold into `cb` right before the launch being recorded (or right after DLSS's output kernel):
    /// the copies in, what `hold` records, the copy back and the application's compute state. `hold` decides
    /// (on what it can know: the device's state, the network) whether to hold; it records nothing until it
    /// calls [`Ctx::open`], and returns the hold's [`Meta`] once recorded, or why it did not
    /// ([`Recorded::Skipped`] for what passes, [`Recorded::Refused`] for what stays).
    pub(crate) fn record_outcome(&self, cb: vk::CommandBuffer, point: InlinePoint, hold: impl FnOnce(&mut Ctx) -> Result<Meta, Recorded>) -> Recorded {
        // Switched off: nothing to hold for.
        if self.model_switched_off() {
            return Recorded::Skipped;
        }
        if self.stopped.load(Ordering::Relaxed) {
            return Recorded::Refused;
        }
        let stage = if point.hold.stage == Stage::Upscaled { Stage::Upscaled } else { Stage::Before };
        let extent = (point.desc.width, point.desc.height);
        let convert = point.desc.format != vk::Format::R16G16B16A16_SFLOAT;
        let (index, staging, queries, readback, outstanding, mut cache_dump) = {
            let mut s = lock(&self.slots);
            if s.simultaneous.contains(&cb) {
                self.say_once("a DLSS buffer recorded with SIMULTANEOUS_USE is not held inside".into());
                return Recorded::Refused;
            }
            if s.secondary.contains(&cb) {
                self.say_once("DLSS is recorded in a secondary command buffer; it is not held inside".into());
                return Recorded::Refused;
            }
            // The hold's dispatches, copies and blits run on DLSS's graphics queue: a buffer of another
            // family's pool is not held (nor one of an unknown pool).
            match s.pool_of.get(&cb).and_then(|p| s.pool_family.get(p)).copied() {
                Some(family) if family == self.app_family => {}
                family => {
                    self.say_once(match family {
                        Some(f) => format!("DLSS is recorded for queue family {f}, not the graphics family {}; it is not held inside", self.app_family),
                        None => "DLSS's command buffer comes from a pool the layer did not see created; it is not held inside".to_string(),
                    });
                    return Recorded::Refused;
                }
            }
            // One hold per stage in a buffer: a second launch at the same point adds none.
            if s.by_cb.get(&cb).is_some_and(|slots| slots.iter().any(|&i| s.slots[i].stage == stage)) {
                return Recorded::Skipped;
            }
            if convert && !self.blit_r11g11b10 {
                self.say_once(format!("the colour input is {:?} and the device cannot blit it to RGBA16F; not holding", point.desc.format));
                return Recorded::Refused;
            }
            // The application's compute state must be put back after the hold: noted from the first hold
            // point on, for buffers begun after it.
            if !self.restore.activate() {
                self.say_once("noting the application's compute state from here on, for the hold inside DLSS's buffer".into());
                return Recorded::Skipped;
            }
            if !self.restore.with(cb, |s| s.complete()) {
                return Recorded::Skipped;
            }
            let unrestorable = self.restore.with(cb, |s| s.unrestorable().or(s.needs_missing(&self.restore.fns).then_some("a binding command the layer cannot repeat")));
            if let Some(why) = unrestorable {
                self.say_once(format!("a DLSS buffer is not held inside: its compute state cannot be put back after the hold ({why})"));
                return Recorded::Skipped;
            }
            let index = match s.slots.iter().position(|sl| sl.owner.is_none()) {
                Some(i) => i,
                None if s.slots.len() < MAX_SLOTS => {
                    // SAFETY: the device is live.
                    let Some(slot) = (unsafe { self.build_slot() }) else {
                        self.say_once("a slot for the hold inside DLSS's buffer could not be built; not holding there".into());
                        return Recorded::Refused;
                    };
                    s.slots.push(slot);
                    s.slots.len() - 1
                }
                None => {
                    self.say_busy(&s);
                    return Recorded::Skipped;
                }
            };
            // The stage's staging images at this extent: built again only once no hold of the stage may run.
            let k = stage_index(stage);
            let others = s.outstanding(stage, None);
            if s.staging[k].is_none_or(|st| st.extent != extent) {
                if others > 0 {
                    return Recorded::Skipped;
                }
                if let Some(old) = s.staging[k].take() {
                    // SAFETY: no hold of the stage may still run.
                    unsafe { self.free_staging(&old) };
                }
                // SAFETY: the device is live.
                s.staging[k] = unsafe { self.build_staging(extent) };
                if s.staging[k].is_none() {
                    self.say_once(format!("staging images of {}x{} could not be built; that buffer is not held inside", extent.0, extent.1));
                    return Recorded::Refused;
                }
            }
            let staging = s.staging[k].expect("built above");
            let slot = &mut s.slots[index];
            slot.owner = Some(cb);
            slot.stage = stage;
            slot.reserved = true;
            slot.meta = None;
            slot.submits = 0;
            (index, staging, slot.queries, slot.readback, others, slot.cache_dump.take())
        };
        // What is copied beside the colour input, from where it can be read.
        let exposure = point
            .exposure_input
            .filter(|a| a.readable && a.format == vk::Format::R16_SFLOAT && matches!(a.layout, Some(vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL)))
            .map(|a| (a.image, a.layout.unwrap_or(vk::ImageLayout::GENERAL)));
        let mvec = point
            .mvec_input
            .filter(|a| a.readable && point.mvec_extent.0 > 0 && point.mvec_extent.1 > 0)
            .and_then(|a| a.layout.map(|layout| (a.image, layout, point.mvec_extent)));
        let depth_source = point.depth_input.filter(|a| a.readable && super::edit_cache_on()).and_then(|a| a.layout.map(|layout| (a.image, layout, a.format)));
        let held = InlinePoint {
            colour: staging.colour,
            layout: vk::ImageLayout::GENERAL,
            exposure_input: exposure.map(|_| Aux { image: staging.exposure, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }),
            // The copy is at the colour input's extent; the source's extent stays (the vectors' unit,
            // `InlinePoint::mvec_scale`).
            mvec_input: mvec.map(|_| Aux { image: staging.mvec, format: vk::Format::R16G16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }),
            depth_input: None,
            ..point
        };
        let mut ctx = Ctx {
            cb,
            point: held,
            source: point,
            staging,
            depth: depth_source.map(|(_, _, f)| f),
            slot: index as u32,
            queries,
            readback,
            cache_dump: &mut cache_dump,
            outstanding,
            convert,
            exposure,
            mvec,
            depth_source,
            device: &self.device,
            opened: false,
            closed: false,
        };
        let outcome = hold(&mut ctx);
        let opened = ctx.opened;
        if opened {
            // SAFETY: recording into the application's buffer on this thread (the launch hook).
            unsafe {
                ctx.close();
                let fns = self.restore.fns;
                self.restore.with(cb, |s| s.replay(&self.device, &fns, cb));
            }
        }
        let mut s = lock(&self.slots);
        let slot = &mut s.slots[index];
        slot.reserved = false;
        slot.cache_dump = cache_dump;
        match outcome {
            // Something was recorded: the slot is the buffer's until it can no longer run, whatever the hold
            // decided after opening (its readings are then dropped).
            Ok(meta) if opened => {
                slot.meta = Some(meta);
                slot.recorded = Some(Instant::now());
                s.by_cb.entry(cb).or_default().push(index);
                self.outstanding.fetch_add(1, Ordering::SeqCst);
                drop(s);
                self.say_once(format!(
                    "holding inside DLSS's command buffer, {} ({:?} in {:?}{}; why not the split: {:?}; motion vectors {})",
                    match stage {
                        Stage::Upscaled => format!("right after its output kernel, on its {}x{} output (passes {}-{})", extent.0, extent.1, point.hold.upscaled.0 + 1, point.hold.upscaled.1),
                        _ => "right before its first launch that reads the colour input".to_string(),
                    },
                    point.desc.format,
                    point.layout,
                    if convert { ", blitted to and from RGBA16F" } else { "" },
                    point.hazard,
                    match (mvec, point.mvec_input) {
                        (Some((_, layout, e)), _) if e == extent => format!("copied from {layout:?}"),
                        (Some((_, layout, e)), _) => format!("resampled from {}x{} in {layout:?}", e.0, e.1),
                        (None, Some(a)) => format!("not copied ({:?} in {:?}, readable {})", a.format, a.layout, a.readable),
                        (None, None) => "not known".to_string(),
                    }
                ));
                Recorded::Held
            }
            other => {
                if opened {
                    // Recorded, then refused (cannot happen: the closure decides before opening); kept as a
                    // hold with nothing to read back.
                    let stage = slot.stage;
                    slot.meta = Some(Meta::empty(stage));
                    s.by_cb.entry(cb).or_default().push(index);
                    self.outstanding.fetch_add(1, Ordering::SeqCst);
                    return Recorded::Held;
                }
                slot.owner = None;
                match other {
                    Err(r) => r,
                    Ok(_) => Recorded::Skipped,
                }
            }
        }
    }

    /// Logs once, when every slot is taken, what their owners are doing.
    fn say_busy(&self, s: &Slots) {
        if !lock(&self.said).insert("all slots in use".into()) {
            return;
        }
        let now = Instant::now();
        let owners: Vec<String> = s
            .slots
            .iter()
            .map(|sl| {
                format!(
                    "{:?} recorded {:.1}s ago, submitted {}x, pool {}",
                    sl.owner,
                    sl.recorded.map_or(f32::NAN, |r| now.duration_since(r).as_secs_f32()),
                    sl.submits,
                    sl.owner.and_then(|c| s.pool_of.get(&c)).map_or("unknown".to_string(), |p| format!("{p:?}"))
                )
            })
            .collect();
        crate::log!(
            "[preupscale] all {MAX_SLOTS} slots of the hold inside DLSS's buffer are in use; that buffer is not held (owners: {}; {} buffers with a known pool)",
            owners.join("; "),
            s.pool_of.len()
        );
        crate::logging::flush();
    }

    /// Frees `cb`'s slots, reading back each hold that ran.
    fn release(&self, s: &mut Slots, cb: vk::CommandBuffer) {
        for i in s.by_cb.remove(&cb).unwrap_or_default() {
            let slot = &mut s.slots[i];
            if slot.reserved {
                continue;
            }
            if let Some(meta) = slot.meta.take() {
                self.outstanding.fetch_sub(1, Ordering::SeqCst);
                if slot.submits > 0 {
                    // SAFETY: the buffer cannot be pending (the application's call that released it requires
                    // that), and it ran: everything its hold wrote is complete, and the hold's last barrier
                    // made the readback available to the host.
                    let h = unsafe { self.read_back(slot, meta) };
                    lock(&self.harvest).push(h);
                }
            }
            slot.owner = None;
            slot.submits = 0;
        }
    }

    /// The readings of a slot whose buffer ran.
    ///
    /// # Safety
    /// As in [`Self::release`].
    unsafe fn read_back(&self, slot: &mut Slot, meta: Meta) -> Harvest {
        let mut readback = [0u8; READBACK_BYTES as usize];
        // SAFETY: host-coherent, mapped for READBACK_BYTES (the contract: complete and available).
        unsafe { std::ptr::copy_nonoverlapping(slot.readback_ptr, readback.as_mut_ptr(), READBACK_BYTES as usize) };
        let mut gpu_ms = [None; 3];
        if let (true, Some((ns, mask))) = (meta.timed && slot.queries != vk::QueryPool::null(), self.ticks) {
            let mut ticks = [0u64; TIMESTAMPS as usize];
            // SAFETY: the pool holds TIMESTAMPS queries, written by the hold that ran; no wait.
            let read = unsafe { self.device.get_query_pool_results(slot.queries, 0, TIMESTAMPS, &mut ticks, vk::QueryResultFlags::TYPE_64) };
            if read.is_ok() {
                for k in 0..3 {
                    let d = (ticks[k + 1] & mask).wrapping_sub(ticks[k] & mask) & mask;
                    let ms = (d as f64 * ns / 1.0e6) as f32;
                    gpu_ms[k] = ms.is_finite().then_some(ms);
                }
            }
        }
        // R16_SFLOAT in the low 16 bits.
        let at = READ_EXPOSURE as usize;
        let exposure = super::f16_to_f32(u16::from_le_bytes([readback[at], readback[at + 1]]));
        let cache_frame = meta.cache_dump.and_then(|pending| {
            let b = slot.cache_dump.as_mut()?;
            b.pending = Some(pending);
            b.take(exposure)
        });
        Harvest { meta, runs: slot.submits, gpu_ms, readback, cache_frame }
    }

    /// A new slot: its timestamp pool (none where the family writes no timestamps) and its readback buffer.
    ///
    /// # Safety
    /// The device is live.
    unsafe fn build_slot(&self) -> Option<Slot> {
        let d = &self.device;
        // SAFETY: valid create infos on a live device; each is destroyed on a later failure.
        unsafe {
            let queries = match self.ticks {
                Some(_) => d.create_query_pool(&vk::QueryPoolCreateInfo::builder().query_type(vk::QueryType::TIMESTAMP).query_count(TIMESTAMPS), None).ok()?,
                None => vk::QueryPool::null(),
            };
            let info = vk::BufferCreateInfo::builder().size(READBACK_BYTES).usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let made = d.create_buffer(&info, None).ok().and_then(|buffer| {
                let reqs = d.get_buffer_memory_requirements(buffer);
                let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let index = (0..self.memory.memory_type_count)
                    .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(host));
                let memory = index.and_then(|i| d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(i), None).ok());
                let ptr = memory.and_then(|m| d.bind_buffer_memory(buffer, m, 0).and_then(|()| d.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())).ok());
                match (memory, ptr) {
                    (Some(m), Some(p)) => {
                        std::ptr::write_bytes(p.cast::<u8>(), 0, READBACK_BYTES as usize);
                        Some((buffer, m, p.cast::<u8>().cast_const()))
                    }
                    (m, _) => {
                        d.destroy_buffer(buffer, None);
                        if let Some(m) = m {
                            d.free_memory(m, None);
                        }
                        None
                    }
                }
            });
            let Some((readback, readback_memory, readback_ptr)) = made else {
                d.destroy_query_pool(queries, None);
                return None;
            };
            Some(Slot {
                owner: None,
                stage: Stage::Before,
                reserved: false,
                recorded: None,
                submits: 0,
                queries,
                readback,
                readback_memory,
                readback_ptr,
                cache_dump: None,
                meta: None,
            })
        }
    }

    /// One stage's staging images at `extent` (`None` when any of them cannot be built).
    ///
    /// # Safety
    /// The device is live.
    unsafe fn build_staging(&self, extent: (u32, u32)) -> Option<Staging> {
        // SAFETY: forwarded; each part is freed when another fails.
        unsafe {
            let colour = self.image(vk::Format::R16G16B16A16_SFLOAT, extent);
            let exposure = self.image(vk::Format::R16_SFLOAT, (1, 1));
            let mvec = self.image(vk::Format::R16G16_SFLOAT, extent);
            let depth = self.buffer(u64::from(extent.0) * u64::from(extent.1) * 4);
            match (colour, exposure, mvec, depth) {
                (Some(c), Some(e), Some(m), Some(dp)) => Some(Staging {
                    extent,
                    colour: c.0,
                    colour_memory: c.1,
                    exposure: e.0,
                    exposure_memory: e.1,
                    mvec: m.0,
                    mvec_memory: m.1,
                    depth: dp.0,
                    depth_memory: dp.1,
                }),
                (c, e, m, dp) => {
                    for (image, memory) in [c, e, m].into_iter().flatten() {
                        self.device.destroy_image(image, None);
                        self.device.free_memory(memory, None);
                    }
                    if let Some((buffer, memory)) = dp {
                        self.device.destroy_buffer(buffer, None);
                        self.device.free_memory(memory, None);
                    }
                    None
                }
            }
        }
    }

    /// # Safety
    /// Nothing that may still run uses them.
    unsafe fn free_staging(&self, s: &Staging) {
        let d = &self.device;
        // SAFETY: the contract.
        unsafe {
            for (image, memory) in [(s.colour, s.colour_memory), (s.exposure, s.exposure_memory), (s.mvec, s.mvec_memory)] {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            }
            d.destroy_buffer(s.depth, None);
            d.free_memory(s.depth_memory, None);
        }
    }

    /// A device-local `TRANSFER_SRC | TRANSFER_DST` buffer of the layer's.
    ///
    /// # Safety
    /// The device is live.
    unsafe fn buffer(&self, bytes: u64) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        #[cfg(test)]
        if tests::FAIL_IMAGES.get() {
            return None;
        }
        let d = &self.device;
        let info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info on a live device.
        let buffer = unsafe { d.create_buffer(&info, None) }.ok()?;
        // SAFETY: just created.
        let reqs = unsafe { d.get_buffer_memory_requirements(buffer) };
        let index = (0..self.memory.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        // SAFETY: valid allocate info.
        let memory = index.and_then(|index| unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.ok());
        // SAFETY: a fresh buffer and an allocation of its required size and type.
        match memory.filter(|&m| unsafe { d.bind_buffer_memory(buffer, m, 0) }.is_ok()) {
            Some(memory) => Some((buffer, memory)),
            None => {
                // SAFETY: never used.
                unsafe {
                    d.destroy_buffer(buffer, None);
                    if let Some(m) = memory {
                        d.free_memory(m, None);
                    }
                }
                None
            }
        }
    }

    /// A device-local image of the layer's: `TRANSFER_SRC | TRANSFER_DST | STORAGE` (the encode and
    /// decode use the staging image as a storage image).
    ///
    /// # Safety
    /// The device is live.
    unsafe fn image(&self, format: vk::Format, extent: (u32, u32)) -> Option<(vk::Image, vk::DeviceMemory)> {
        #[cfg(test)]
        if tests::FAIL_IMAGES.get() {
            return None;
        }
        let d = &self.device;
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.0, height: extent.1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info on a live device.
        let image = match unsafe { d.create_image(&info, None) } {
            Ok(image) => image,
            Err(e) => {
                crate::log!("[preupscale] staging image {}x{} {format:?}: vkCreateImage failed ({e:?})", extent.0, extent.1);
                crate::logging::flush();
                return None;
            }
        };
        // SAFETY: `image` was just created.
        let reqs = unsafe { d.get_image_memory_requirements(image) };
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let index = (0..self.memory.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(local))
            .or_else(|| (0..self.memory.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0));
        let alloc = index.map(|index| vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index).build());
        // SAFETY: valid allocate info.
        let memory = alloc.and_then(|a| unsafe { d.allocate_memory(&a, None) }.inspect_err(|e| crate::log!("[preupscale] staging image: vkAllocateMemory of {} bytes failed ({e:?})", reqs.size)).ok());
        // SAFETY: a fresh image and an allocation of its required size and type.
        match memory.filter(|&m| unsafe { d.bind_image_memory(image, m, 0) }.is_ok()) {
            Some(memory) => Some((image, memory)),
            None => {
                // SAFETY: never used.
                unsafe {
                    d.destroy_image(image, None);
                    if let Some(m) = memory {
                        d.free_memory(m, None);
                    }
                }
                crate::logging::flush();
                None
            }
        }
    }
}

impl Meta {
    /// A hold with nothing to read back.
    pub(crate) fn empty(stage: Stage) -> Self {
        Self {
            stage,
            resources: 0,
            identification: 0,
            stats: false,
            scene: None,
            auto: false,
            kind: None,
            init: None,
            fresh: (0, 0),
            exposure_source: None,
            cache_dump: None,
            timed: false,
        }
    }
}

/// The barrier of `range` of `image` from `old` to `new`.
fn image_barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags, range: vk::ImageSubresourceRange) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build()
}

const COLOUR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
const COLOUR_LAYERS: vk::ImageSubresourceLayers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };

/// The copies recorded into the application's buffer at the start of a hold: the colour input (`colour`: the
/// image, its layout there, and whether it is converted (blitted) rather than copied: an R11G11B10 colour
/// input) into the RGBA16F staging image, and beside it DLSS's exposure (`exposure`: source, its layout, the
/// staging image), motion vectors (`mvec`: source, its layout, the staging image, the source's extent: larger
/// ones are blitted down) and depth aspect (`depth`: source, its layout, its format, the staging buffer). A
/// colour input, motion vectors or depth outside a layout a transfer reads are moved to the transfer layout
/// and back. Ends with everything copied visible to the hold's commands.
///
/// # Safety
/// `cb` is being recorded on this thread; every image is in the given layout at this point of the buffer
/// and has the usage its transfers need; `staging` is RGBA16F of the colour input's extent.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_copy_in(
    d: &ash::Device, cb: vk::CommandBuffer, (colour, layout, convert): (vk::Image, vk::ImageLayout, bool), staging: vk::Image, extent: (u32, u32),
    exposure: Option<(vk::Image, vk::ImageLayout, vk::Image)>, mvec: Option<(vk::Image, vk::ImageLayout, vk::Image, (u32, u32))>,
    depth: Option<(vk::Image, vk::ImageLayout, vk::Format, vk::Buffer)>,
) {
    let region = vk::ImageCopy {
        src_subresource: COLOUR_LAYERS,
        src_offset: vk::Offset3D::default(),
        dst_subresource: COLOUR_LAYERS,
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D { width: extent.0, height: extent.1, depth: 1 },
    };
    let far = vk::Offset3D { x: extent.0 as i32, y: extent.1 as i32, z: 1 };
    let blit = vk::ImageBlit { src_subresource: COLOUR_LAYERS, src_offsets: [vk::Offset3D::default(), far], dst_subresource: COLOUR_LAYERS, dst_offsets: [vk::Offset3D::default(), far] };
    let general = layout == vk::ImageLayout::GENERAL;
    let read_layout = if general { vk::ImageLayout::GENERAL } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
    // Everything the buffer wrote before this point (the frame, rendered in this same buffer) is visible to
    // the copies; the staging images' old contents are not needed.
    let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).build();
    let fresh = |image| image_barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE, COLOUR_RANGE);
    let mut to_read = vec![fresh(staging)];
    to_read.extend(exposure.map(|(_, _, own)| fresh(own)));
    if !general {
        to_read.push(image_barrier(colour, layout, read_layout, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::TRANSFER_READ, COLOUR_RANGE));
    }
    // DLSS's motion vectors, read in place when readable there, else moved to TRANSFER_SRC_OPTIMAL and back.
    // High-resolution ones (larger than the colour input) are blitted down to it, NEAREST: each render pixel
    // takes one of the vectors it covers, never an average across an edge.
    let mvec_moved = mvec.filter(|(_, l, _, _)| !matches!(*l, vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL));
    to_read.extend(mvec.map(|(_, _, own, _)| fresh(own)));
    to_read.extend(mvec_moved.map(|(source, l, _, _)| image_barrier(source, l, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::TRANSFER_READ, COLOUR_RANGE)));
    let mut back: Vec<vk::ImageMemoryBarrier> = mvec_moved
        .map(|(source, l, _, _)| image_barrier(source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, l, vk::AccessFlags::empty(), vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, COLOUR_RANGE))
        .into_iter()
        .collect();
    // DLSS's depth aspect, read in place in GENERAL or TRANSFER_SRC_OPTIMAL, else moved there and back.
    let depth_moved = depth.filter(|(_, l, _, _)| !matches!(*l, vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL));
    to_read.extend(depth_moved.map(|(source, l, f, _)| image_barrier(source, l, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::TRANSFER_READ, super::aux_range(f))));
    back.extend(depth_moved.map(|(source, l, f, _)| {
        image_barrier(source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, l, vk::AccessFlags::empty(), vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, super::aux_range(f))
    }));
    let exposure_region = vk::ImageCopy { extent: vk::Extent3D { width: 1, height: 1, depth: 1 }, ..region };
    // The copies visible to the hold's commands (transfers and shaders), the sources back where they were.
    let copied = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
    // SAFETY: the contract; every command is legal outside a render pass instance, where a CUDA launch is.
    unsafe {
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[open], &[], &to_read);
        if convert {
            d.cmd_blit_image(cb, colour, read_layout, staging, vk::ImageLayout::GENERAL, &[blit], vk::Filter::NEAREST);
        } else {
            d.cmd_copy_image(cb, colour, read_layout, staging, vk::ImageLayout::GENERAL, &[region]);
        }
        if let Some((source, source_layout, own)) = exposure {
            d.cmd_copy_image(cb, source, source_layout, own, vk::ImageLayout::GENERAL, &[exposure_region]);
        }
        if let Some((source, source_layout, own, (sw, sh))) = mvec {
            let from = if mvec_moved.is_some() { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { source_layout };
            if (sw, sh) == extent {
                d.cmd_copy_image(cb, source, from, own, vk::ImageLayout::GENERAL, &[region]);
            } else {
                let down = vk::ImageBlit {
                    src_subresource: COLOUR_LAYERS,
                    src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw as i32, y: sh as i32, z: 1 }],
                    dst_subresource: COLOUR_LAYERS,
                    dst_offsets: [vk::Offset3D::default(), far],
                };
                d.cmd_blit_image(cb, source, from, own, vk::ImageLayout::GENERAL, &[down], vk::Filter::NEAREST);
            }
        }
        if let Some((source, source_layout, format, own)) = depth {
            let from = if depth_moved.is_some() { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { source_layout };
            let aspect = if super::DEPTH_FORMATS.contains(&format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
            let copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: aspect, ..COLOUR_LAYERS },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: extent.0, height: extent.1, depth: 1 },
            };
            d.cmd_copy_image_to_buffer(cb, source, from, own, &[copy]);
        }
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[copied], &[], &back);
    }
}

/// The copy back recorded at the end of a hold: the staging image (in `GENERAL`, holding the hold's answer,
/// or the frame as it was copied in when the hold wrote nothing) over the colour input, in its layout (moved
/// to `TRANSFER_DST_OPTIMAL` and back outside `GENERAL`; blitted for an R11G11B10 one), then visible to
/// everything after it in the buffer: DLSS's launch first.
///
/// # Safety
/// As [`record_copy_in`], recorded before it in the same buffer.
pub(crate) unsafe fn record_copy_back(d: &ash::Device, cb: vk::CommandBuffer, (colour, layout, convert): (vk::Image, vk::ImageLayout, bool), staging: vk::Image, extent: (u32, u32)) {
    let region = vk::ImageCopy {
        src_subresource: COLOUR_LAYERS,
        src_offset: vk::Offset3D::default(),
        dst_subresource: COLOUR_LAYERS,
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D { width: extent.0, height: extent.1, depth: 1 },
    };
    let far = vk::Offset3D { x: extent.0 as i32, y: extent.1 as i32, z: 1 };
    let blit = vk::ImageBlit { src_subresource: COLOUR_LAYERS, src_offsets: [vk::Offset3D::default(), far], dst_subresource: COLOUR_LAYERS, dst_offsets: [vk::Offset3D::default(), far] };
    let general = layout == vk::ImageLayout::GENERAL;
    let write_layout = if general { vk::ImageLayout::GENERAL } else { vk::ImageLayout::TRANSFER_DST_OPTIMAL };
    // The hold's writes of the staging image (shaders or transfers) visible to the copy; outside GENERAL the
    // colour input went to TRANSFER_SRC_OPTIMAL for the copy in, and now to the write layout.
    let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).build();
    let to_write: Vec<vk::ImageMemoryBarrier> = (!general)
        .then(|| image_barrier(colour, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, write_layout, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE, COLOUR_RANGE))
        .into_iter()
        .collect();
    let close = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
    let back: Vec<vk::ImageMemoryBarrier> = (!general)
        .then(|| image_barrier(colour, write_layout, layout, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, COLOUR_RANGE))
        .into_iter()
        .collect();
    // SAFETY: the contract.
    unsafe {
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[open], &[], &to_write);
        if convert {
            d.cmd_blit_image(cb, staging, vk::ImageLayout::GENERAL, colour, write_layout, &[blit], vk::Filter::NEAREST);
        } else {
            d.cmd_copy_image(cb, staging, vk::ImageLayout::GENERAL, colour, write_layout, &[region]);
        }
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[close], &[], &back);
    }
}

/// The shared memory's header for `device`'s hold inside DLSS's buffer: from here on nothing is
/// recorded there while the model is switched off ([`Inline::record_outcome`]). Set once; the mapping is
/// never unmapped.
pub(crate) fn watch_switch(device: vk::Device, header: *const neural_forge_protocol::ShmHeader) {
    if let Some(inline) = lock(&INLINES).get(&device) {
        inline.switch.store(header.cast_mut(), Ordering::Release);
    }
}

/// Frees the slots and staging images: called from `vkDestroyDevice` before anything else of the device's
/// is freed, once the device is idle (no buffer of the application's still runs a hold).
///
/// # Safety
/// The caller meets `vkDestroyDevice`'s external synchronization requirements.
pub(crate) unsafe fn destroy(device: vk::Device) {
    let Some(inline) = lock(&INLINES).remove(&device) else { return };
    let d = &inline.device;
    // SAFETY: the application destroys the device and owns all its queues now.
    match unsafe { d.device_wait_idle() } {
        Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {
            let mut s = lock(&inline.slots);
            for slot in s.slots.drain(..) {
                // SAFETY: idle (or lost): nothing uses them.
                unsafe {
                    d.destroy_query_pool(slot.queries, None);
                    d.destroy_buffer(slot.readback, None);
                    d.free_memory(slot.readback_memory, None);
                    if let Some(b) = &slot.cache_dump {
                        b.destroy(d);
                    }
                }
            }
            for staging in s.staging.iter_mut().filter_map(Option::take) {
                // SAFETY: as above.
                unsafe { inline.free_staging(&staging) };
            }
        }
        Err(e) => crate::log!("[preupscale] inline hold teardown wait failed: {e:?}; its slots are not freed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preupscale::native::{self, Built, CacheAsk, Loader, Passes, Setup};
    use crate::preupscale::{f16_to_f32, Hazard, ImageDesc, Resources, Session, Sink, Target};

    thread_local! {
        /// Makes every staging image this thread asks for fail to build.
        pub(crate) static FAIL_IMAGES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    const W: u32 = 20;
    const H: u32 = 12;
    const FW: u32 = 32;
    const FH: u32 = 16;
    const RANGE: vk::ImageSubresourceRange = COLOUR_RANGE;

    fn half(v: f32) -> u16 {
        crate::preupscale::tests::f32_to_f16(v)
    }

    /// Frame `f`'s colour input: a picture that changes a little from frame to frame (never a scene cut).
    fn colour_at(x: u32, y: u32, c: u32, f: u32) -> f32 {
        0.125 + ((x * 7 + y * 3 + c * 5) % 32) as f32 / 32.0 + f as f32 / 256.0
    }

    /// Frame `f`'s motion vectors: a pan of a few pixels a frame (the edit cache's moving regime: the
    /// network on one hold in two, the others cached).
    fn vector_of(f: u32) -> [f32; 2] {
        [3.0 + (f % 3) as f32 * 0.5, -2.0]
    }

    /// What the test's "network" answers (the head the composite reads).
    fn head_at(x: u32, y: u32, c: u32) -> f32 {
        (((x + y + c) % 9) as f32 - 4.0) * 0.25
    }

    /// A mapped host-visible buffer of the test's.
    pub(crate) struct Host {
        pub buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        pub ptr: *mut u8,
        pub bytes: u64,
    }

    /// The test device, and what a game and NVIDIA's runtime give a hold: DLSS's colour input, motion vectors
    /// and exposure input, and a "network": its features and head buffers, an empty recorded graph for the
    /// split hold (whose head the test writes from the host), and for the hold inside DLSS's buffer a recorder
    /// that writes the same head there.
    pub(crate) struct Rig {
        _entry: ash::Entry,
        pub instance: ash::Instance,
        pub physical: vk::PhysicalDevice,
        pub device: Arc<ash::Device>,
        pub queue: vk::Queue,
        pub family: u32,
        props: vk::PhysicalDeviceMemoryProperties,
        pub pool: vk::CommandPool,
        graph_pool: vk::CommandPool,
        images: Vec<(vk::Image, vk::DeviceMemory)>,
        hosts: Vec<Host>,
        pub colour: vk::Image,
        mvec: vk::Image,
        exposure: vk::Image,
        head: usize,
        head_bytes: Vec<u8>,
        built: Built,
        loader: Option<Arc<Loader>>,
        pub inline: Arc<Inline>,
        session: Session,
        split: Option<Resources>,
    }

    impl Rig {
        /// `None` without a Vulkan device; `descriptor_buffer`: on a device with VK_EXT_descriptor_buffer
        /// (`None` without one).
        pub(crate) fn open(descriptor_buffer: bool) -> Option<Self> {
            let (entry, instance, physical, device, queue, family) =
                if descriptor_buffer { crate::composition::gpu::test_device_with_descriptor_buffer()? } else { crate::composition::gpu::test_device()? };
            let device = Arc::new(device);
            // SAFETY: test-only objects on a live device; everything is destroyed in `drop`.
            unsafe {
                let props = instance.get_physical_device_memory_properties(physical);
                let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
                let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
                let graph = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1)).unwrap()[0];
                let inheritance = vk::CommandBufferInheritanceInfo::default();
                device.begin_command_buffer(graph, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance)).unwrap();
                device.end_command_buffer(graph).unwrap();
                let holds = Arc::new(AtomicUsize::new(0));
                // The replay's functions, as `device.rs` resolves them below the layer.
                let gdpa = |name: &std::ffi::CStr| (instance.fp_v1_0().get_device_proc_addr)(device.handle(), name.as_ptr());
                let fns = super::super::restore::Fns {
                    push_descriptor_set: gdpa(c"vkCmdPushDescriptorSetKHR").map(|f| std::mem::transmute(f)),
                    push_descriptor_set_with_template: gdpa(c"vkCmdPushDescriptorSetWithTemplateKHR").map(|f| std::mem::transmute(f)),
                    set_descriptor_buffer_offsets: gdpa(c"vkCmdSetDescriptorBufferOffsetsEXT").map(|f| std::mem::transmute(f)),
                    bind_descriptor_buffer_embedded_samplers: gdpa(c"vkCmdBindDescriptorBufferEmbeddedSamplersEXT").map(|f| std::mem::transmute(f)),
                    bind_descriptor_buffers: gdpa(c"vkCmdBindDescriptorBuffersEXT").map(|f| std::mem::transmute(f)),
                    bind_shaders: gdpa(c"vkCmdBindShadersEXT").map(|f| std::mem::transmute(f)),
                };
                let inline = Inline::new(device.clone(), &instance, physical, family, holds.clone(), fns);
                inline.pool_created(pool, family);
                // As after the device's first hold point: every buffer the tests begin is noted.
                inline.restore.activate();
                let mut rig = Self {
                    _entry: entry,
                    instance,
                    physical,
                    device: device.clone(),
                    queue,
                    family,
                    props,
                    pool,
                    graph_pool,
                    images: Vec::new(),
                    hosts: Vec::new(),
                    colour: vk::Image::null(),
                    mvec: vk::Image::null(),
                    exposure: vk::Image::null(),
                    head: 0,
                    head_bytes: Vec::new(),
                    built: Built { width: W, height: H, frame: neural_forge_native::Frame::default(), graph, graph_own: graph, generation: 1, network: neural_forge_protocol::enums::network::FULL },
                    loader: None,
                    inline,
                    session: Session::default(),
                    split: None,
                };
                let transfer = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
                rig.colour = rig.image(vk::Format::R16G16B16A16_SFLOAT, (W, H), transfer | vk::ImageUsageFlags::STORAGE);
                rig.mvec = rig.image(vk::Format::R16G16_SFLOAT, (W, H), transfer | vk::ImageUsageFlags::SAMPLED);
                rig.exposure = rig.image(vk::Format::R16_SFLOAT, (1, 1), transfer | vk::ImageUsageFlags::STORAGE);
                let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
                let features = rig.host(u64::from(FW * FH) * 64, storage);
                rig.head = rig.host(u64::from(FW * FH) * 16, storage);
                let h: Vec<f32> = (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [head_at(x, y, 0), head_at(x, y, 1), head_at(x, y, 2), 1.5])).collect();
                rig.head_bytes = h.iter().flat_map(|v| v.to_le_bytes()).collect();
                rig.built.frame = neural_forge_native::Frame {
                    field_width: FW,
                    field_height: FH,
                    features: rig.hosts[features].buffer,
                    features_bytes: rig.hosts[features].bytes,
                    head: rig.hosts[rig.head].buffer,
                    head_bytes: rig.hosts[rig.head].bytes,
                    blend_scale: 0.5,
                    chained: 0,
                };
                // The network inside DLSS's buffer: the head written there, in the buffer's order.
                let (d, head, bytes) = (device.clone(), rig.hosts[rig.head].buffer, rig.head_bytes.clone());
                rig.loader = Some(Arc::new(Loader::for_test(
                    Setup::for_test(&rig._entry, &rig.instance, physical, family),
                    vec![rig.built],
                    holds,
                    Box::new(move |cmd, _| d.cmd_update_buffer(cmd, head, 0, &bytes)),
                )));
                // The exposure input: 1.0.
                rig.upload_image(rig.exposure, (1, 1), &0x3c00u16.to_le_bytes());
                Some(rig)
            }
        }

        pub(crate) fn image(&mut self, format: vk::Format, (w, h): (u32, u32), usage: vk::ImageUsageFlags) -> vk::Image {
            let d = &self.device;
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: test-only, on a live device; destroyed in `drop`.
            unsafe {
                let image = d.create_image(&info, None).unwrap();
                let reqs = d.get_image_memory_requirements(image);
                let index = (0..self.props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
                let memory = d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None).unwrap();
                d.bind_image_memory(image, memory, 0).unwrap();
                self.images.push((image, memory));
                // Into GENERAL, the layout the game keeps DLSS's inputs in.
                self.one_shot(|cmd| {
                    let b = image_barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, RANGE);
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
                });
                image
            }
        }

        /// A mapped host-visible buffer of `bytes` (its index in `hosts`).
        pub(crate) fn host(&mut self, bytes: u64, usage: vk::BufferUsageFlags) -> usize {
            let d = &self.device;
            // SAFETY: test-only, on a live device; destroyed in `drop`.
            unsafe {
                let buffer = d.create_buffer(&vk::BufferCreateInfo::builder().size(bytes).usage(usage), None).unwrap();
                let reqs = d.get_buffer_memory_requirements(buffer);
                let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let index = (0..self.props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.props.memory_types[i as usize].property_flags.contains(host)).unwrap();
                let mut flags = vk::MemoryAllocateFlagsInfo::builder().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                let mut info = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index);
                if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                    info = info.push_next(&mut flags);
                }
                let memory = d.allocate_memory(&info, None).unwrap();
                d.bind_buffer_memory(buffer, memory, 0).unwrap();
                let ptr = d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast::<u8>();
                std::ptr::write_bytes(ptr, 0, bytes as usize);
                self.hosts.push(Host { buffer, memory, ptr, bytes });
                self.hosts.len() - 1
            }
        }

        pub(crate) fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.device;
            // SAFETY: the test's own pool, queue and device.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.queue).unwrap();
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        fn upload_image(&mut self, image: vk::Image, (w, h): (u32, u32), bytes: &[u8]) {
            let k = self.host(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
            // SAFETY: mapped for its size.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.hosts[k].ptr, bytes.len()) };
            let buffer = self.hosts[k].buffer;
            self.one_shot(|cmd| unsafe { self.device.cmd_copy_buffer_to_image(cmd, buffer, image, vk::ImageLayout::GENERAL, &[whole(w, h)]) });
        }

        /// Frame `f`'s colour input and motion vectors, as bytes for `upload`.
        fn frame_bytes(f: u32) -> (Vec<u8>, Vec<u8>) {
            let colour: Vec<u8> = (0..H).flat_map(|y| (0..W).flat_map(move |x| [half(colour_at(x, y, 0, f)), half(colour_at(x, y, 1, f)), half(colour_at(x, y, 2, f)), half(1.0)])).flat_map(u16::to_le_bytes).collect();
            let v = vector_of(f);
            let mvec: Vec<u8> = (0..W * H).flat_map(|_| [half(v[0]), half(v[1])]).flat_map(u16::to_le_bytes).collect();
            (colour, mvec)
        }

        /// A host buffer holding frame `f`'s colour input, then its motion vectors (for [`Self::record_frame`]).
        fn upload_of(&mut self, f: u32) -> usize {
            let (colour, mvec) = Self::frame_bytes(f);
            let k = self.host((colour.len() + mvec.len()) as u64, vk::BufferUsageFlags::TRANSFER_SRC);
            // SAFETY: mapped for its size.
            unsafe {
                std::ptr::copy_nonoverlapping(colour.as_ptr(), self.hosts[k].ptr, colour.len());
                std::ptr::copy_nonoverlapping(mvec.as_ptr(), self.hosts[k].ptr.add(colour.len()), mvec.len());
            }
            k
        }

        /// Records the game "rendering" frame `f` (from upload buffer `k`) into `cmd`.
        fn record_frame(&self, cmd: vk::CommandBuffer, k: usize) {
            let d = &self.device;
            let colour_bytes = u64::from(W * H) * 8;
            let all = vk::PipelineStageFlags::ALL_COMMANDS;
            let written = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
            // SAFETY: recording the test's buffer.
            unsafe {
                d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[written], &[], &[]);
                d.cmd_copy_buffer_to_image(cmd, self.hosts[k].buffer, self.colour, vk::ImageLayout::GENERAL, &[whole(W, H)]);
                d.cmd_copy_buffer_to_image(cmd, self.hosts[k].buffer, self.mvec, vk::ImageLayout::GENERAL, &[vk::BufferImageCopy { buffer_offset: colour_bytes, ..whole(W, H) }]);
                d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[written], &[], &[]);
            }
        }

        /// Records a copy of the colour input into readback buffer `k`.
        fn record_read(&self, cmd: vk::CommandBuffer, k: usize) {
            let d = &self.device;
            let all = vk::PipelineStageFlags::ALL_COMMANDS;
            let written = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ).build();
            let read = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ).build();
            // SAFETY: recording the test's buffer.
            unsafe {
                d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[written], &[], &[]);
                d.cmd_copy_image_to_buffer(cmd, self.colour, vk::ImageLayout::GENERAL, self.hosts[k].buffer, &[whole(W, H)]);
                d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[read], &[], &[]);
            }
        }

        fn read(&self, k: usize) -> Vec<u16> {
            // SAFETY: mapped; the copy into it finished (the caller waited).
            unsafe { std::slice::from_raw_parts(self.hosts[k].ptr.cast::<u16>(), (W * H * 4) as usize).to_vec() }
        }

        fn ask(hold: u64) -> CacheAsk {
            CacheAsk { hold, interval: 2, debug_mask: false }
        }

        /// The split hold (the reference): frame `f` uploaded, held in front of the game's buffer as hold
        /// number `hold` (the history reset first with `reset`), and the colour input read back.
        pub(crate) fn split_frame(&mut self, f: u32, hold: u64, reset: bool) -> Vec<u16> {
            let k = self.upload_of(f);
            self.one_shot(|cmd| self.record_frame(cmd, k));
            // The split hold's head comes from the host (its recorded graph is empty).
            // SAFETY: mapped for its size; nothing pending uses it (the queue is idle).
            unsafe { std::ptr::copy_nonoverlapping(self.head_bytes.as_ptr(), self.hosts[self.head].ptr, self.head_bytes.len()) };
            let (device, instance, physical, family, queue) = (self.device.clone(), self.instance.clone(), self.physical, self.family, self.queue);
            if self.split.is_none() {
                // SAFETY: the device is live.
                self.split = unsafe { Resources::build_inline(&device, &instance, physical, family, W, H) };
            }
            let res = self.split.as_mut().expect("resources");
            if reset {
                if let Some(n) = res.native.as_mut() {
                    n.gap.skipped();
                }
                if let Some(e) = res.edit.as_mut() {
                    e.state.drop_edit();
                }
            }
            let aux = |image, format| Some(Aux { image, format, layout: Some(vk::ImageLayout::GENERAL), readable: true });
            let target = Target {
                colour: self.colour,
                width: W,
                height: H,
                depth: None,
                depth_buffer: None,
                mvec: aux(self.mvec, vk::Format::R16G16_SFLOAT),
                mvec_scale: [1.0, 1.0],
                exposure: [None; crate::preupscale::MAX_EXPOSURE],
                exposure_input: aux(self.exposure, vk::Format::R16_SFLOAT),
                paper_white: 2.5,
                identification: 1,
            };
            let mut submit = |_: crate::preupscale::Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe { device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence) };
            let jitter = Some([0.25 * (f % 4) as f32 - 0.375, 0.125]);
            // SAFETY: the resources are this device's, at this extent; the colour input is in GENERAL.
            let result = unsafe {
                native::run_native_hold(&device, &instance, physical, res, &target, self.loader.as_ref().expect("loader"), jitter, true, Passes::default(), &[(W, H)], None, Self::ask(hold), neural_forge_protocol::enums::network::FULL, None, Sink::Queue(&mut submit))
            };
            assert!(result.wrote_back, "the split hold of frame {f}: {result:?}");
            // SAFETY: the test's queue.
            unsafe { device.queue_wait_idle(queue) }.unwrap();
            let k = self.host(u64::from(W * H) * 8, vk::BufferUsageFlags::TRANSFER_DST);
            self.one_shot(|cmd| self.record_read(cmd, k));
            self.read(k)
        }

        /// A primary command buffer of the test's pool, known to the hold.
        pub(crate) fn buffer(&self) -> vk::CommandBuffer {
            // SAFETY: the test's pool.
            let cb = unsafe { self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).command_buffer_count(1)) }.unwrap()[0];
            self.inline.allocated(self.pool, vk::CommandBufferLevel::PRIMARY, &[cb]);
            cb
        }

        /// The point the tracking gives the hold for this rig's DLSS buffer.
        pub(crate) fn point(&self) -> InlinePoint {
            let aux = |image, format| Some(Aux { image, format, layout: Some(vk::ImageLayout::GENERAL), readable: true });
            InlinePoint {
                colour: self.colour,
                desc: ImageDesc { width: W, height: H, format: vk::Format::R16G16B16A16_SFLOAT, usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE, plain: true },
                layout: vk::ImageLayout::GENERAL,
                exposure_input: aux(self.exposure, vk::Format::R16_SFLOAT),
                mvec_input: aux(self.mvec, vk::Format::R16G16_SFLOAT),
                mvec_extent: (W, H),
                depth_input: None,
                identification: 1,
                hazard: Hazard::MemoryBarrier,
                hold: crate::preupscale::placement::Hold::ALL_BEFORE,
            }
        }

        /// The hold recorded into `cb` (being recorded) as `device.rs` records it: what earlier buffers left
        /// applied, the inline resources ensured, and the native hold recorded through the buffer sink.
        pub(crate) fn hold_into(&mut self, cb: vk::CommandBuffer, f: u32, hold: u64) -> Recorded {
            let point = self.point();
            let Self { inline, session, loader, device, instance, physical, family, .. } = self;
            let loader = loader.as_ref().expect("loader");
            let (device, instance, physical, family) = (device.clone(), instance.clone(), *physical, *family);
            for h in inline.take_harvest() {
                session.harvest(h);
            }
            let jitter = Some([0.25 * (f % 4) as f32 - 0.375, 0.125]);
            inline.record_outcome(cb, point, |ctx| {
                // SAFETY: the device is live.
                unsafe { session.ensure_inline(&device, &instance, physical, family, W, H, crate::preupscale::placement::Stage::Before, ctx.outstanding) }.map_err(|_| Recorded::Skipped)?;
                let target = Target {
                    colour: ctx.point.colour,
                    width: W,
                    height: H,
                    depth: None,
                    depth_buffer: None,
                    mvec: ctx.point.mvec_input,
                    mvec_scale: ctx.point.mvec_scale(),
                    exposure: [None; crate::preupscale::MAX_EXPOSURE],
                    exposure_input: ctx.point.exposure_input,
                    paper_white: 2.5,
                    identification: 1,
                };
                let res = session.res_inline(crate::preupscale::placement::Stage::Before).as_mut().expect("ensured");
                // SAFETY: `cb` is being recorded on this thread; the resources are this device's.
                let mut result = unsafe {
                    native::run_native_hold(&device, &instance, physical, res, &target, loader, jitter, false, Passes::default(), &[(W, H)], None, Self::ask(hold), neural_forge_protocol::enums::network::FULL, None, Sink::Buffer(ctx))
                };
                result.meta.take().ok_or(Recorded::Skipped)
            })
        }
    }

    impl Rig {
        /// Submits `cbs` (recorded, in order) as `device.rs` does: the initialisations the holds in them need
        /// first, then the buffers, then the hold told; waits for them.
        pub(crate) fn submit(&self, cbs: &[vk::CommandBuffer]) {
            let d = &self.device;
            // SAFETY: the test's recorded buffers and the layer's initialisation buffers, on the test's queue.
            unsafe {
                let inits: Vec<vk::CommandBuffer> = self.inline.prologue(cbs.iter().copied()).into_iter().map(|(_, c)| c).collect();
                if !inits.is_empty() {
                    d.queue_submit(self.queue, &[vk::SubmitInfo::builder().command_buffers(&inits).build()], vk::Fence::null()).unwrap();
                }
                d.queue_submit(self.queue, &[vk::SubmitInfo::builder().command_buffers(cbs).build()], vk::Fence::null()).unwrap();
                self.inline.submitted(self.queue, cbs.iter().copied());
                d.queue_wait_idle(self.queue).unwrap();
            }
        }

        /// Begins `cb` again (the hold's slot in it, if any, is free and read back).
        pub(crate) fn begin(&self, cb: vk::CommandBuffer) {
            let begin = vk::CommandBufferBeginInfo::default();
            // SAFETY: the test's buffer, not pending.
            unsafe { self.device.begin_command_buffer(cb, &begin) }.unwrap();
            self.inline.begin(cb, begin.flags);
        }

        pub(crate) fn end(&self, cb: vk::CommandBuffer) {
            // SAFETY: the test's buffer, recording.
            unsafe { self.device.end_command_buffer(cb) }.unwrap();
        }

        /// Records frame `f` with its hold (number `hold`) into `cb`, the colour input read back into a new
        /// readback buffer afterwards (its index).
        pub(crate) fn record_held(&mut self, cb: vk::CommandBuffer, f: u32, hold: u64) -> usize {
            let k = self.upload_of(f);
            let readback = self.host(u64::from(W * H) * 8, vk::BufferUsageFlags::TRANSFER_DST);
            self.begin(cb);
            self.record_frame(cb, k);
            assert_eq!(self.hold_into(cb, f, hold), Recorded::Held, "frame {f}'s hold");
            self.record_read(cb, readback);
            self.end(cb);
            readback
        }

        /// The colour input of frame `f` as the game rendered it (RGBA16F bits).
        pub(crate) fn input(f: u32) -> Vec<u16> {
            let (colour, _) = Self::frame_bytes(f);
            colour.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect()
        }

        /// Clears the head the host gave the split hold: from here on only the recorder inside DLSS's buffer
        /// writes it.
        pub(crate) fn clear_head(&self) {
            // SAFETY: mapped for its size; the queue is idle.
            unsafe { std::ptr::write_bytes(self.hosts[self.head].ptr, 0, self.head_bytes.len()) };
        }
    }

    /// The first texel where two RGBA16F readbacks differ, for the failure message.
    fn first_difference(a: &[u16], b: &[u16]) -> Option<(usize, f32, f32)> {
        a.iter().zip(b).position(|(x, y)| x != y).map(|i| (i, f16_to_f32(a[i]), f16_to_f32(b[i])))
    }

    /// A hold recorded into DLSS's buffer runs the model's work there: frame after frame (the network on one
    /// hold in two, the edit cache carrying its edit to the others), each buffer leaves the colour input bit
    /// for bit as the split hold does, whose work is submitted in front of the buffer. Nothing waits on the
    /// host (the buffer is submitted, then waited for like any), and the network's answer reaches the
    /// composite only from inside the buffer: the host's copy of the head is cleared.
    #[test]
    fn a_hold_recorded_into_the_buffer_runs_the_model_there_as_the_split_hold_does() {
        let Some(mut rig) = Rig::open(false) else { return };
        const FRAMES: u32 = 6;
        let expected: Vec<Vec<u16>> = (0..FRAMES).map(|f| rig.split_frame(f, u64::from(f) + 1, false)).collect();
        assert_ne!(expected[0], Rig::input(0), "the model changes the picture");
        assert_ne!(expected[1], Rig::input(1), "the cached hold carries the edit");
        rig.clear_head();
        let cb = rig.buffer();
        for f in 0..FRAMES {
            let readback = rig.record_held(cb, f, u64::from(f) + 1);
            rig.submit(&[cb]);
            let got = rig.read(readback);
            assert!(got == expected[f as usize], "frame {f}: the hold inside the buffer differs from the split hold (first difference {:?})", first_difference(&got, &expected[f as usize]));
        }
    }

    /// Several of DLSS's buffers recorded before any runs: each hold's parameters (its jitter, its frame's
    /// parity, the edit cache's plan) and the history it reads are its own, and three buffers recorded ahead
    /// and run in order leave the frames as the split hold does. A buffer recorded and recorded again before
    /// it ever ran: the hold after it finds on the GPU that the hold the CPU planned it after never ran, and
    /// starts afresh: a network frame without history (as the split hold after a reset), a cached hold without
    /// the edit of a refresh that never ran (the frame as it was); the holds after it carry on.
    #[test]
    fn buffers_recorded_before_any_runs_keep_each_hold_s_parameters_and_history() {
        let Some(mut rig) = Rig::open(false) else { return };
        let split: Vec<Vec<u16>> = [(0, 1, false), (1, 2, false), (2, 3, false), (4, 5, true), (5, 6, false)].into_iter().map(|(f, h, reset)| rig.split_frame(f, h, reset)).collect();
        rig.clear_head();
        let cbs: Vec<vk::CommandBuffer> = (0..3).map(|_| rig.buffer()).collect();
        let reads: Vec<usize> = (0..3).map(|i| rig.record_held(cbs[i], i as u32, i as u64 + 1)).collect();
        rig.submit(&cbs);
        for i in 0..3 {
            let got = rig.read(reads[i]);
            assert!(got == split[i], "frame {i}, recorded ahead: differs from the split hold (first difference {:?})", first_difference(&got, &split[i]));
        }
        // Frame 3's buffer (hold 4, cached) is recorded and recorded again, as frame 4 (hold 5, a network frame),
        // before it ever ran.
        let b = rig.buffer();
        let _ = rig.record_held(b, 3, 4);
        let read = rig.record_held(b, 4, 5);
        rig.submit(&[b]);
        let got = rig.read(read);
        assert!(got == split[3], "the network frame after a hold that never ran differs from one after a reset (first difference {:?})", first_difference(&got, &split[3]));
        let read = rig.record_held(b, 5, 6);
        rig.submit(&[b]);
        let got = rig.read(read);
        assert!(got == split[4], "the cached hold after it differs from the split hold's (first difference {:?})", first_difference(&got, &split[4]));
        // Frame 6's (hold 7, a refresh) recorded and recorded again as frame 7 (hold 8, planned as cached on it).
        let _ = rig.record_held(b, 6, 7);
        let read = rig.record_held(b, 7, 8);
        rig.submit(&[b]);
        let got = rig.read(read);
        assert!(got == Rig::input(7), "a cached hold applied the edit of a refresh that never ran (first difference {:?})", first_difference(&got, &Rig::input(7)));
    }

    const PROBE_SPV: &[u8] = include_bytes!("../../shaders/test_restore_probe.spv");

    /// The application's compute state is put back after the hold: the game binds its compute pipeline, its
    /// descriptor set (with a dynamic offset) and pushes its constants, renders, the hold is recorded (whose
    /// dispatches bind the layer's own), and the game dispatches again without binding anything: its shader
    /// sees its own push constants and the buffer its set points at.
    #[test]
    fn the_application_s_compute_state_is_put_back_after_the_hold() {
        restore_case(false);
    }

    /// The same with VK_EXT_descriptor_buffer (vkd3d-proton's binding model on NVIDIA): the hold's descriptor
    /// sets invalidate the game's descriptor-buffer offsets, which are set again (the descriptor buffers it
    /// bound stay bound). Skipped without the extension.
    #[test]
    fn descriptor_buffer_offsets_are_put_back_after_the_hold() {
        restore_case(true);
    }

    fn restore_case(descriptor_buffer: bool) {
        let Some(mut rig) = Rig::open(descriptor_buffer) else {
            eprintln!("restore test (descriptor buffer {descriptor_buffer}): no such device, skipping");
            return;
        };
        let address = if descriptor_buffer { vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS } else { vk::BufferUsageFlags::empty() };
        let input = rig.host(256, vk::BufferUsageFlags::STORAGE_BUFFER | address);
        let output = rig.host(16, vk::BufferUsageFlags::STORAGE_BUFFER | address);
        // SAFETY: mapped for its size; nothing uses it yet.
        unsafe { std::ptr::copy_nonoverlapping(0x1234_5678u32.to_le_bytes().as_ptr(), rig.hosts[input].ptr.add(64), 4) };
        let d = rig.device.clone();
        let db = descriptor_buffer.then(|| ash::extensions::ext::DescriptorBuffer::new(&rig.instance, &d));
        // SAFETY: test-only objects on the rig's device, destroyed at the end.
        unsafe {
            let binding = |n: u32, ty| vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build();
            // Classic sets: the input through a dynamic offset (64 bytes in).
            let in_type = if descriptor_buffer { vk::DescriptorType::STORAGE_BUFFER } else { vk::DescriptorType::STORAGE_BUFFER_DYNAMIC };
            let bindings = [binding(0, in_type), binding(1, vk::DescriptorType::STORAGE_BUFFER)];
            let flags = if descriptor_buffer { vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT } else { vk::DescriptorSetLayoutCreateFlags::empty() };
            let set_layout = d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().flags(flags).bindings(&bindings), None).unwrap();
            let push_range = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).size(8).build();
            let layout = d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(&[set_layout]).push_constant_ranges(&[push_range]), None).unwrap();
            let code: Vec<u32> = PROBE_SPV.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code), None).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").build();
            let pflags = if descriptor_buffer { vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT } else { vk::PipelineCreateFlags::empty() };
            let pipeline = d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::builder().flags(pflags).stage(stage).layout(layout).build()], None).unwrap()[0];
            d.destroy_shader_module(module, None);
            let mut pool = vk::DescriptorPool::null();
            let mut set = vk::DescriptorSet::null();
            let mut descriptors = None;
            match &db {
                None => {
                    let sizes = [
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, descriptor_count: 1 },
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
                    ];
                    pool = d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).unwrap();
                    set = d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
                    let i = [vk::DescriptorBufferInfo { buffer: rig.hosts[input].buffer, offset: 0, range: 16 }];
                    let o = [vk::DescriptorBufferInfo { buffer: rig.hosts[output].buffer, offset: 0, range: vk::WHOLE_SIZE }];
                    d.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC).buffer_info(&i).build(),
                            vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&o).build(),
                        ],
                        &[],
                    );
                }
                Some(db) => {
                    let mut props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
                    let mut p2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut props).build();
                    rig.instance.get_physical_device_properties2(rig.physical, &mut p2);
                    let size = db.get_descriptor_set_layout_size(set_layout);
                    let k = rig.host(size.max(256), vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
                    let addr = |b: vk::Buffer| d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(b));
                    for (n, (buffer, at, range)) in [(rig.hosts[input].buffer, 64u64, 16u64), (rig.hosts[output].buffer, 0, 16)].into_iter().enumerate() {
                        let info = vk::DescriptorAddressInfoEXT::builder().address(addr(buffer) + at).range(range).build();
                        let get = vk::DescriptorGetInfoEXT::builder().ty(vk::DescriptorType::STORAGE_BUFFER).data(vk::DescriptorDataEXT { p_storage_buffer: &info }).build();
                        let offset = db.get_descriptor_set_layout_binding_offset(set_layout, n as u32) as usize;
                        let mut bytes = vec![0u8; props.storage_buffer_descriptor_size];
                        db.get_descriptor(&get, &mut bytes);
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), rig.hosts[k].ptr.add(offset), bytes.len());
                    }
                    descriptors = Some(addr(rig.hosts[k].buffer));
                }
            }
            let cb = rig.buffer();
            let frame = rig.upload_of(0);
            rig.begin(cb);
            let restore = rig.inline.restore.clone();
            // The game's compute state, noted as the layer's hooks note it.
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            restore.with(cb, |s| s.bind_compute(super::super::restore::Compute::Pipeline(pipeline)));
            match (&db, descriptors) {
                (Some(db), Some(address)) => {
                    let info = vk::DescriptorBufferBindingInfoEXT::builder().address(address).usage(vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT).build();
                    db.cmd_bind_descriptor_buffers(cb, &[info]);
                    restore.with(cb, |s| s.bind_buffers(&[info]));
                    db.cmd_set_descriptor_buffer_offsets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[0], &[0]);
                    restore.with(cb, |s| s.bind(super::super::restore::Binding::Offsets { layout, first: 0, indices: vec![0], offsets: vec![0] }));
                }
                _ => {
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[64]);
                    restore.with(cb, |s| s.bind(super::super::restore::Binding::Sets { layout, first: 0, sets: vec![set], dynamic: vec![64] }));
                }
            }
            let push: Vec<u8> = [0xa11ce_u32, 0xb0b_u32].iter().flat_map(|v| v.to_le_bytes()).collect();
            d.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            restore.with(cb, |s| s.push(layout, vk::ShaderStageFlags::COMPUTE, 0, &push));
            rig.record_frame(cb, frame);
            assert_eq!(rig.hold_into(cb, 0, 1), Recorded::Held);
            // The game's next dispatch, binding nothing.
            d.cmd_dispatch(cb, 1, 1, 1);
            let done = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ).build();
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[done], &[], &[]);
            rig.end(cb);
            rig.submit(&[cb]);
            let out = std::slice::from_raw_parts(rig.hosts[output].ptr.cast::<u32>(), 4).to_vec();
            assert_eq!(out, vec![0xa11ce, 0x1234_5678, 0xb0b, 0x600d_beef], "the game's dispatch after the hold did not see its own pipeline, descriptors and push constants");
            d.device_wait_idle().unwrap();
            d.destroy_pipeline(pipeline, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            if pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(pool, None);
            }
        }
    }

    /// The hold the gating tests record: the copies in and back only, nothing read back.
    fn copies_only(ctx: &mut Ctx) -> Result<Meta, Recorded> {
        // SAFETY: the test's buffer, recording on this thread.
        unsafe { ctx.open() };
        Ok(Meta::empty(Stage::Before))
    }

    /// Only a primary of a pool of DLSS's graphics family is held inside, and nothing while the model is
    /// switched off: a secondary's hold would run wherever it is executed, another family's queue cannot run
    /// the hold's work, and a pool the layer never saw created has no known family. Nothing is recorded into
    /// any of them, and no slot is built for them.
    #[test]
    fn only_a_graphics_familys_primary_is_held_inside_and_only_with_the_model_on() {
        let Some(rig) = Rig::open(false) else { return };
        let inline = &rig.inline;
        let cb = |raw: u64| <vk::CommandBuffer as vk::Handle>::from_raw(raw);
        let pool_of = |raw: u64| <vk::CommandPool as vk::Handle>::from_raw(raw);
        let (pool, other_pool, unseen) = (pool_of(1), pool_of(2), pool_of(3));
        inline.pool_created(pool, rig.family);
        inline.pool_created(other_pool, rig.family + 1);
        inline.allocated(pool, vk::CommandBufferLevel::SECONDARY, &[cb(10)]);
        inline.allocated(other_pool, vk::CommandBufferLevel::PRIMARY, &[cb(11)]);
        inline.allocated(unseen, vk::CommandBufferLevel::PRIMARY, &[cb(12)]);
        for (c, why) in [(cb(10), "a secondary"), (cb(11), "another family's pool"), (cb(12), "a pool never seen created"), (cb(13), "a buffer never seen allocated")] {
            assert_eq!(inline.record_outcome(c, rig.point(), |_| unreachable!("{why}: nothing is recorded")), Recorded::Refused, "{why} is not held inside");
            assert!(!inline.owns(c));
        }
        assert!(lock(&inline.slots).slots.is_empty(), "nothing was even built for them");
        // The model switched off: nothing is recorded; on again, the hold is.
        let mut header: Box<neural_forge_protocol::ShmHeader> = Box::default();
        header.enabled.store(1, Ordering::Relaxed);
        header.apply_model.store(0, Ordering::Relaxed);
        inline.switch.store(&mut *header, Ordering::Release);
        let real = rig.buffer();
        rig.begin(real);
        assert_eq!(inline.record_outcome(real, rig.point(), copies_only), Recorded::Skipped, "switched off");
        header.apply_model.store(1, Ordering::Relaxed);
        assert_eq!(inline.record_outcome(real, rig.point(), copies_only), Recorded::Held, "switched on again");
        // A second launch at the same stage in that buffer adds no hold, and that passes.
        assert_eq!(inline.record_outcome(real, rig.point(), copies_only), Recorded::Skipped, "this stage is held in it already");
        assert_eq!(inline.outstanding.load(Ordering::SeqCst), 1, "one hold may run");
        rig.end(real);
        // Its pool reset: the slot is free for another buffer, and nothing may run any more.
        inline.reset_pool(rig.pool, false);
        assert!(!inline.owns(real));
        assert_eq!(inline.outstanding.load(Ordering::SeqCst), 0);
        inline.switch.store(std::ptr::null_mut(), Ordering::Release);
        // A buffer whose compute state cannot be put back is not held.
        rig.begin(real);
        inline.restore.with(real, |s| s.refuse("a test"));
        assert_eq!(inline.record_outcome(real, rig.point(), copies_only), Recorded::Skipped);
        rig.end(real);
    }

    /// Staging images that could not be built are never recorded: the hold is refused while building fails,
    /// and built at the next hold once it works again; a stage's images change size only once no hold of the
    /// stage may still run.
    #[test]
    fn staging_images_are_built_again_only_when_no_hold_may_run() {
        let Some(mut rig) = Rig::open(false) else { return };
        let small = rig.point();
        let big_image = rig.image(vk::Format::R16G16B16A16_SFLOAT, (2 * W, 2 * H), vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE);
        let big = InlinePoint { colour: big_image, desc: ImageDesc { width: 2 * W, height: 2 * H, ..small.desc }, mvec_input: None, exposure_input: None, ..small };
        let (a, b) = (rig.buffer(), rig.buffer());
        let staging = |rig: &Rig| lock(&rig.inline.slots).staging[0].map(|s| s.extent);
        FAIL_IMAGES.set(true);
        rig.begin(a);
        assert_eq!(rig.inline.record_outcome(a, small, copies_only), Recorded::Refused, "no staging image");
        assert_eq!(staging(&rig), None);
        FAIL_IMAGES.set(false);
        assert_eq!(rig.inline.record_outcome(a, small, copies_only), Recorded::Held);
        rig.end(a);
        assert_eq!(staging(&rig), Some((W, H)));
        // Another extent while `a`'s hold may still run: not built, not held.
        rig.begin(b);
        assert_eq!(rig.inline.record_outcome(b, big, copies_only), Recorded::Skipped);
        assert_eq!(staging(&rig), Some((W, H)));
        rig.end(b);
        rig.submit(&[a]);
        // `a` recorded again: nothing of the stage may run any more, and the images are built at the new size.
        rig.begin(a);
        rig.begin(b);
        assert_eq!(rig.inline.record_outcome(b, big, copies_only), Recorded::Held);
        assert_eq!(staging(&rig), Some((2 * W, 2 * H)));
        rig.end(b);
        rig.end(a);
        rig.submit(&[b]);
    }

    /// The copies into the staging images, with the hold writing nothing: DLSS's motion vectors (in
    /// `SHADER_READ_ONLY_OPTIMAL`, at the colour input's extent or twice it, resampled to it) land in the
    /// stage's motion-vector image, DLSS's images are left as they were and in their layouts, and an
    /// R11G11B10 colour input (Unreal Engine 5's, sampled) comes back byte for byte through the RGBA16F staging
    /// image.
    #[test]
    fn the_copies_in_and_back_leave_dlss_s_inputs_as_they_were() {
        let Some(mut rig) = Rig::open(false) else { return };
        for scale in [1u32, 2] {
            let (mw, mh) = (W * scale, H * scale);
            let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
            let mvec = rig.image(vk::Format::R16G16_SFLOAT, (mw, mh), usage);
            // Each texel holds the position of the colour input's pixel it covers (constant over each block of
            // `scale` x `scale` texels: which of them NEAREST takes at a block's centre is the device's choice).
            let bytes: Vec<u8> = (0..mh).flat_map(|y| (0..mw).flat_map(move |x| [half((x / scale) as f32), half((y / scale) as f32)])).flat_map(u16::to_le_bytes).collect();
            rig.upload_image(mvec, (mw, mh), &bytes);
            let k = rig.host(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_DST);
            let all = vk::PipelineStageFlags::ALL_COMMANDS;
            let to = |old, new| image_barrier(mvec, old, new, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::MEMORY_READ, RANGE);
            let point = InlinePoint {
                mvec_input: Some(Aux { image: mvec, format: vk::Format::R16G16_SFLOAT, layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL), readable: true }),
                mvec_extent: (mw, mh),
                ..rig.point()
            };
            let cb = rig.buffer();
            rig.begin(cb);
            // SAFETY: recording the test's buffer.
            unsafe {
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
                assert_eq!(rig.inline.record_outcome(cb, point, copies_only), Recorded::Held);
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL)]);
                rig.device.cmd_copy_image_to_buffer(cb, mvec, vk::ImageLayout::GENERAL, rig.hosts[k].buffer, &[whole(mw, mh)]);
            }
            rig.end(cb);
            rig.submit(&[cb]);
            // SAFETY: mapped; the copy finished.
            let source: Vec<u16> = unsafe { std::slice::from_raw_parts(rig.hosts[k].ptr.cast::<u16>(), (mw * mh * 2) as usize) }.to_vec();
            for (i, t) in source.as_chunks::<2>().0.iter().enumerate() {
                let (x, y) = ((i as u32 % mw / scale) as f32, (i as u32 / mw / scale) as f32);
                assert_eq!([f16_to_f32(t[0]), f16_to_f32(t[1])], [x, y], "DLSS's motion vectors after the hold");
            }
            let staged = lock(&rig.inline.slots).staging[0].expect("built").mvec;
            let k2 = rig.host(u64::from(W * H) * 4, vk::BufferUsageFlags::TRANSFER_DST);
            rig.one_shot(|cmd| unsafe { rig.device.cmd_copy_image_to_buffer(cmd, staged, vk::ImageLayout::GENERAL, rig.hosts[k2].buffer, &[whole(W, H)]) });
            // SAFETY: as above.
            let copy: Vec<u16> = unsafe { std::slice::from_raw_parts(rig.hosts[k2].ptr.cast::<u16>(), (W * H * 2) as usize) }.to_vec();
            for (i, t) in copy.as_chunks::<2>().0.iter().enumerate() {
                let (x, y) = ((i as u32 % W) as f32, (i as u32 / W) as f32);
                assert_eq!([f16_to_f32(t[0]), f16_to_f32(t[1])], [x, y], "the staging copy at ({x}, {y}), scale {scale}");
            }
        }
        // An R11G11B10 colour input in SHADER_READ_ONLY_OPTIMAL.
        let format = vk::Format::B10G11R11_UFLOAT_PACK32;
        let blit = vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST;
        // SAFETY: a plain query.
        if !unsafe { rig.instance.get_physical_device_format_properties(rig.physical, format) }.optimal_tiling_features.contains(blit) {
            eprintln!("no R11G11B10 blit on this device, skipping that part");
            return;
        }
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let colour = rig.image(format, (W, H), usage);
        let k = rig.host(u64::from(W * H) * 4, vk::BufferUsageFlags::TRANSFER_DST);
        let run = |rig: &mut Rig, with_hold: bool| -> Vec<u8> {
            let cb = rig.buffer();
            let all = vk::PipelineStageFlags::ALL_COMMANDS;
            let to = |old, new| image_barrier(colour, old, new, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE, RANGE);
            rig.begin(cb);
            // SAFETY: recording the test's buffer.
            unsafe {
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)]);
                rig.device.cmd_clear_color_image(cb, colour, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [1.75, 0.3125, 6.5, 1.0] }, &[RANGE]);
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
                if with_hold {
                    let point = InlinePoint {
                        colour,
                        desc: ImageDesc { width: W, height: H, format, usage, plain: true },
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        exposure_input: None,
                        mvec_input: None,
                        hazard: Hazard::NotSplittable,
                        ..rig.point()
                    };
                    assert_eq!(rig.inline.record_outcome(cb, point, copies_only), Recorded::Held);
                }
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)]);
                rig.device.cmd_copy_image_to_buffer(cb, colour, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rig.hosts[k].buffer, &[whole(W, H)]);
                rig.device.cmd_pipeline_barrier(cb, all, all, vk::DependencyFlags::empty(), &[], &[], &[to(vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::GENERAL)]);
            }
            rig.end(cb);
            rig.submit(&[cb]);
            // SAFETY: mapped; the copy finished.
            unsafe { std::slice::from_raw_parts(rig.hosts[k].ptr, (W * H * 4) as usize) }.to_vec()
        };
        let reference = run(&mut rig, false);
        let held = run(&mut rig, true);
        assert!(reference.iter().any(|&b| b != 0), "the clear wrote something");
        assert_eq!(held, reference, "an R11G11B10 frame changed across a hold that wrote nothing");
    }

    fn whole(w: u32, h: u32) -> vk::BufferImageCopy {
        vk::BufferImageCopy { image_subresource: COLOUR_LAYERS, image_extent: vk::Extent3D { width: w, height: h, depth: 1 }, ..Default::default() }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let d = self.device.clone();
            // SAFETY: test teardown: the device is waited idle first, then everything of the test's destroyed.
            unsafe {
                d.device_wait_idle().unwrap();
                // Every buffer is free again (its slots read back), then the hold's own.
                d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()).unwrap();
                self.inline.reset_pool(self.pool, true);
                destroy(d.handle());
                self.session.destroy(&d);
                if let Some(r) = self.split.take() {
                    r.destroy(&d);
                }
                d.destroy_command_pool(self.pool, None);
                d.destroy_command_pool(self.graph_pool, None);
                for (image, memory) in self.images.drain(..) {
                    d.destroy_image(image, None);
                    d.free_memory(memory, None);
                }
                for h in self.hosts.drain(..) {
                    d.destroy_buffer(h.buffer, None);
                    d.free_memory(h.memory, None);
                }
            }
        }
    }
}
