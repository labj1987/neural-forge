//! The before-DLSS placement on a second GPU (docs/NATIVE_BACKEND.md, "Before DLSS on a second GPU"): the exchange
//! between the hold on the game's GPU and the network on the other one, and the worker that serves it.
//!
//! The hold before DLSS is part of DLSS's own command buffer, and DLSS needs the edited colour input before it can
//! run: waiting there for the other GPU's answer to the same frame leaves the game's GPU idle for the whole round
//! trip. So the *edit* is delayed, not the frame: hold N sends its frame out and carries the edit of frame N-1,
//! which the other GPU finished while the game's GPU rendered frame N, onto its own fresh frame along DLSS's motion
//! vectors, as a cached hold of the edit cache does (`super::cache`). DLSS gets frame N with frame N's depth,
//! vectors and jitter: nothing of its inputs or of NGX's parameters is delayed, and no latency is added.
//!
//! Two regions of system memory carry the frames ([`Layout`]): OUT (the game's GPU writes the frame's encode, its
//! accumulated motion, its depth and the small state the network's frame reads; the other GPU reads) and IN (the
//! other GPU writes the answer; the game's GPU reads). Both are this module's own page-aligned allocations, so each
//! device imports them as host memory where it can (`super::region_buffer`) and stages them where it cannot.
//!
//! Nothing on the game's GPU waits for the other one: a command buffer may not wait on an event the host sets
//! while it is pending (VUID-vkSetEvent-event-09543), so the wait is the CPU's, at the submit of the hold's buffer,
//! where everything the game submitted before (the frame's own rendering) is already running:
//!
//! - [`Exchange::record_send`], after the hold's capture: the copies into OUT and the event `sent` set, in the
//!   order the GPU runs the holds (DLSS's buffers can be recorded ahead, recorded again or never submitted);
//! - the [`Worker`] polls `sent`, resets it, runs the frame through its [`Runner`] (the network on the other GPU)
//!   and notes it finished ([`Progress`]);
//! - [`Exchange::at_submit`], before a hold's buffer goes to the driver: it waits ([`ANSWER_BUDGET`] at most) for
//!   the worker to be done with the frame sent last, when that frame's buffer was handed to the driver in an
//!   earlier submit ([`Exchange::handed`]), and writes IN's header: whose answer the region holds and whether it
//!   is complete. The game's GPU goes idle there only when the other GPU is slower than the game's frame, which
//!   bounds the frame rate by the other GPU, as the finished-frame offload's present does. A frame not answered
//!   within the budget is abandoned: its answer is never taken;
//! - [`Exchange::record_receive`], first in the hold: the answer copied where the hold wants it, and
//!   [`Exchange::record_gate`] after the edit cache captured it: where the header does not name the expected
//!   frame's complete answer, the edit is left not stored and the cached write-back passes the frame through.
//!
//! The worker writes IN only between a send and that frame's finish, and a send is only recorded in a hold whose
//! submit settled the send before it, so a hold never reads an answer that is being written (but one abandoned,
//! which its gate refuses), and the worker never reads OUT while the game's GPU writes it (but for an abandoned
//! frame, whose answer is dropped and after which the runner starts its history afresh).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ash::vk;

use super::HostBuffer;

const GATE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_offload_gate.spv");
const PACK_SPV: &[u8] = include_bytes!("../../shaders/preupscale_offload_pack.spv");
/// `preupscale_offload_pack.comp`'s local size and the texels a row of its dispatch covers.
const PACK_LOCAL: u64 = 64;
const PACK_ROW: u64 = 65536;
/// The half a texel of the packed motion holds in both components where the field is not a number (the chain left
/// the frame, or the hold had no vectors): the one pattern the pack ever writes that is not a number.
#[cfg(test)]
const HALF_NAN: u32 = 0x7e00_7e00;

/// `NEURAL_FORGE_OFFLOAD_MOTION`: `float` sends the accumulated motion as it is (2 x f32 a texel, as 4.3.0 did);
/// anything else, or unset, packs it to two halves a texel (`preupscale_offload_pack.comp`), half the bytes.
pub(crate) fn motion_packed() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_MOTION").is_none_or(|v| !v.trim().eq_ignore_ascii_case("float")));
    *ON
}

/// Each region's header, and the alignment of every section after it (a storage buffer's offset alignment is at
/// most this on every driver the layer runs on).
pub(crate) const HEADER_BYTES: u64 = 256;
/// OUT's header words: the frame's sequence number (never 0), and the sections' sizes as the send wrote them.
pub(crate) const OUT_SEQ: usize = 0;
/// IN's header words (`preupscale_offload_gate.comp`), written by [`Exchange::at_submit`]: 1 when the region holds
/// a complete answer, and whose it is.
pub(crate) const IN_VALID: usize = 0;
pub(crate) const IN_SEQ: usize = 1;

/// How long the submit of a hold's buffer waits for the answer to the frame sent before it, at most; past it the
/// hold goes on without the answer. The finished-frame offload's floor for a late frame.
pub(crate) const ANSWER_BUDGET: Duration = super::offload::LATE_FLOOR;

/// Which hold the other GPU serves: the one before DLSS (at DLSS's render size), or the one after DLSS's output (at
/// the output size: the placement with Ray Reconstruction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Where {
    Before,
    After,
}

impl Where {
    /// Of a hold at `stage`.
    pub(crate) fn of(stage: super::placement::Stage) -> Self {
        if stage == super::placement::Stage::Upscaled {
            Self::After
        } else {
            Self::Before
        }
    }

    /// Its bit in [`SERVING_HOLD`], and its place in [`SERVER_TIME`].
    fn bit(self) -> u8 {
        match self {
            Self::Before => 1,
            Self::After => 2,
        }
    }

    /// As the log and the Status tab say it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Before => "before DLSS",
            Self::After => "after DLSS",
        }
    }
}

/// The sizes of what crosses, in bytes (0 for a section the hold does not have).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Sizes {
    /// The frame as the capture encoded it.
    pub frame: u64,
    /// The accumulated motion the network's history reads: 2 x f32 a texel, or packed to two halves.
    pub motion: u64,
    /// The texels of the motion when it crosses packed (`preupscale_offload_pack.comp`); 0 when it crosses as floats.
    pub packed_texels: u64,
    /// DLSS's depth as the capture copied it (the control mask's sky rule).
    pub depth: u64,
    /// The small buffers the frame shaders read (the exposure, the scene cut's word), one after another.
    pub side: u64,
    /// The network's answer.
    pub answer: u64,
}

impl Sizes {
    /// What a hold of `width` x `height` at `at` exchanges: the encode and the answer (RGBA16F at the padded
    /// extent), the accumulated motion (two halves a texel with `packed`, else 2 x f32), the small state, and DLSS's
    /// depth before DLSS only (the hold after DLSS's output has none at its size).
    pub(crate) fn of(at: Where, width: u32, height: u32, packed: bool) -> Self {
        let (pw, ph) = super::padded(width, height);
        let (texels, padded) = (u64::from(width) * u64::from(height), u64::from(pw) * u64::from(ph));
        Self {
            frame: padded * 8,
            motion: texels * if packed { 4 } else { 8 },
            packed_texels: if packed { texels } else { 0 },
            depth: if at == Where::Before { texels * 4 } else { 0 },
            side: super::hdr::HdrPass::EXPOSURE_BYTES + 16,
            answer: padded * 8,
        }
    }
}

/// Where each section is in its region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub sizes: Sizes,
    pub frame_at: u64,
    pub motion_at: u64,
    pub depth_at: u64,
    pub side_at: u64,
    pub out_bytes: u64,
    pub answer_at: u64,
    pub in_bytes: u64,
}

impl Layout {
    pub(crate) fn of(sizes: Sizes) -> Self {
        let up = |n: u64| n.div_ceil(HEADER_BYTES) * HEADER_BYTES;
        let frame_at = HEADER_BYTES;
        let motion_at = frame_at + up(sizes.frame);
        let depth_at = motion_at + up(sizes.motion);
        let side_at = depth_at + up(sizes.depth);
        Self {
            sizes,
            frame_at,
            motion_at,
            depth_at,
            side_at,
            out_bytes: side_at + up(sizes.side),
            answer_at: HEADER_BYTES,
            in_bytes: HEADER_BYTES + up(sizes.answer),
        }
    }
}

/// What a hold does with the exchange, decided as it is recorded ([`Cadence::hold`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HoldPlan {
    /// The frame sent last, which the worker must be done with before this hold's buffer is submitted
    /// ([`Exchange::at_submit`]).
    pub settle: Option<u32>,
    /// Take that frame's answer: the hold recorded right before this one sent it.
    pub take: bool,
    /// Send this hold's frame under this sequence number.
    pub send: Option<u32>,
}

/// The exchange's bookkeeping on the CPU, in recording order. What the GPU then finds is checked there
/// (`preupscale_offload_gate.comp`, the hold guard).
#[derive(Debug, Default)]
pub(crate) struct Cadence {
    next: u32,
    /// The sequence number sent last, and the hold that sent it (`Session::seen`) until a hold has taken it or
    /// could not.
    sent: Option<u32>,
    takeable: Option<u64>,
}

impl Cadence {
    /// Hold number `hold` (`Session::seen`: consecutive holds differ by 1). `send`: this hold's frame can be sent
    /// (it has the motion vectors and the jitter the carried edit needs at the next hold, and the offload serves).
    pub(crate) fn hold(&mut self, hold: u64, send: bool) -> HoldPlan {
        let take = self.takeable.take().is_some_and(|at| at.wrapping_add(1) == hold);
        let settle = self.sent;
        let send = send.then(|| {
            self.next = if self.next == u32::MAX { 1 } else { self.next + 1 };
            self.sent = Some(self.next);
            self.takeable = Some(hold);
            self.next
        });
        HoldPlan { settle, take: take && settle.is_some(), send }
    }

    /// The hold planned last recorded nothing after all: what it would have sent is not there to take. The
    /// sequence number stays the one to settle (a send that never runs settles at once, [`Progress::settle`]).
    pub(crate) fn dropped(&mut self) {
        self.takeable = None;
    }
}

/// How a frame ended on the other GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ran {
    /// The answer is complete in IN.
    Answered,
    /// No answer for this frame, and nothing wrong (the network is still loading there).
    Skipped,
    /// The frame cannot run there or did not finish (why): the offload is given up.
    Failed(String),
}

/// What runs a frame on the other GPU.
pub(crate) trait Runner: Send {
    /// Runs the frame in OUT (`out`, laid out as `layout`) and writes its answer to IN (`inn`). A runner that
    /// imported the regions on its device reads and writes them there and ignores the pointers. It returns once
    /// the frame is done with both regions.
    ///
    /// # Safety
    /// `out` and `inn` are valid for the layout's `out_bytes` and `in_bytes`.
    unsafe fn run(&mut self, seq: u32, out: *const u8, inn: *mut u8, layout: &Layout) -> Ran;

    /// The frame just run was abandoned by the hold that waited for it: the game's GPU may have written OUT while
    /// it ran. Its answer is dropped; whatever the runner keeps from frame to frame (the network's history) starts
    /// afresh.
    fn abandoned(&mut self) {}

    /// The offload was given up: the runner lets go of what the game's GPU needs back (the network).
    fn gave_up(&mut self) {}

    /// A frame never finished on the other GPU: nothing it uses can be freed or taken back.
    fn stuck(&self) -> bool {
        false
    }
}

/// Whether sequence number `a` was given out after `b` (serial-number arithmetic: the numbers wrap, and two that
/// are compared are never half the range apart).
fn after(a: u32, b: u32) -> bool {
    a != b && a.wrapping_sub(b) < u32::MAX / 2
}

/// Which frames went to the driver and which the worker is done with: where [`Exchange::at_submit`] waits.
#[derive(Debug, Default)]
pub(crate) struct Progress {
    state: Mutex<Prog>,
    changed: Condvar,
    /// Holds submitted that took the answer to the frame before them, and those planned to that found none.
    taken: AtomicU64,
    untaken: AtomicU64,
    /// Held while the host resets `sent` (the worker) and while a hold records its set: `vkResetEvent` needs the
    /// event externally synchronized, and the hold's thread records it while the worker runs.
    event: Mutex<()>,
    /// Tests: how many times a settle blocked (waited for the worker at all), whatever it then found.
    #[cfg(test)]
    blocked: AtomicU64,
}

#[derive(Debug, Default)]
struct Prog {
    /// The send in a buffer of the submit being made, and the one handed to the driver last.
    queued: Option<u32>,
    handed: Option<u32>,
    /// The frame the worker finished last, and whether its answer is complete in IN.
    finished: Option<(u32, bool)>,
    /// The newest frame a submit gave up waiting for: its answer, and that of every frame sent before it, is never
    /// taken. Sequence numbers are compared as serial numbers ([`after`]), so their wrap changes nothing.
    abandoned: Option<u32>,
    /// The worker runs no frames any more.
    stopped: bool,
}

impl Progress {
    fn lock(&self) -> std::sync::MutexGuard<'_, Prog> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Waits, `budget` at most, for the worker to be done with frame `seq`; `true` when its answer is complete in
    /// IN. No wait for a frame whose buffer was not handed to the driver in an earlier submit (never submitted, or
    /// in the submit being made: it cannot have run), nor once the worker stopped. A frame not done within the
    /// budget is abandoned.
    pub(crate) fn settle(&self, seq: u32, budget: Duration) -> bool {
        let deadline = Instant::now() + budget;
        let mut p = self.lock();
        loop {
            if let Some((done, ok)) = p.finished.filter(|(done, _)| *done == seq) {
                let _ = done;
                return ok;
            }
            if p.handed != Some(seq) || p.stopped {
                return false;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                p.abandoned = Some(p.abandoned.map_or(seq, |a| if after(seq, a) { seq } else { a }));
                return false;
            }
            #[cfg(test)]
            self.blocked.fetch_add(1, Ordering::Relaxed);
            p = self.changed.wait_timeout(p, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }

    /// Whether frame `seq` went to the driver in an earlier submit (handed last, or already finished): only then can a
    /// hold have its answer, and only then does [`settle_and_mark`] write IN's header for it.
    fn went_out(&self, seq: u32) -> bool {
        let p = self.lock();
        p.handed == Some(seq) || p.finished.is_some_and(|(done, _)| done == seq)
    }

    /// A buffer that sends frame `seq` is in the submit being made.
    fn queue(&self, seq: u32) {
        self.lock().queued = Some(seq);
    }

    /// The submit being made went to the driver.
    fn handed(&self) {
        let mut p = self.lock();
        if let Some(seq) = p.queued.take() {
            p.handed = Some(seq);
        }
    }

    /// The worker is done with frame `seq` (`ok`: its answer is complete in IN). `false` when a submit had
    /// abandoned it: the answer is not taken.
    fn finish(&self, seq: u32, ok: bool) -> bool {
        let mut p = self.lock();
        let kept = p.abandoned.is_none_or(|a| after(seq, a));
        p.finished = Some((seq, ok && kept));
        self.changed.notify_all();
        kept
    }

    fn stop(&self) {
        self.lock().stopped = true;
        self.changed.notify_all();
    }
}

/// What the worker counts, for the `[offload]` summary and the Status tab.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub sent: AtomicU64,
    pub answered: AtomicU64,
    pub late: AtomicU64,
    /// Microseconds from a frame's send to its answer, summed over the answered frames.
    pub answer_us: AtomicU64,
}

impl Counters {
    /// The summary line's text since the last call (the counters start over).
    pub(crate) fn summary(&self, at: Where) -> String {
        let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
        let (sent, answered, late, us) = (take(&self.sent), take(&self.answered), take(&self.late), take(&self.answer_us));
        let mean = if answered == 0 { 0.0 } else { us as f64 / answered as f64 / 1000.0 };
        format!(
            "{}: {sent} frames sent to the other GPU, {answered} answered in time (mean {mean:.1} ms from send to answer), {late} without an answer",
            at.name()
        )
    }
}

const SERVING: u8 = 0;
const PASSING: u8 = 1;

/// The exchange on the game's device: the two regions, the two events and the gate.
pub(crate) struct Exchange {
    pub layout: Layout,
    out: HostBuffer,
    inn: HostBuffer,
    /// The regions' own memory (page aligned, for the host-memory import), freed with the exchange (and the last
    /// submit that writes IN's header, [`Submits`]).
    regions: [Arc<Region>; 2],
    /// Both regions are imported on the game's device: another device may import the same memory.
    shared: bool,
    sent: vk::Event,
    progress: Arc<Progress>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The counters the gate's set points at.
    gated: vk::Buffer,
    /// The motion's pack (`Sizes::packed_texels`).
    pack: Option<Pack>,
}

/// The pack of the accumulated motion to halves before it is sent (`preupscale_offload_pack.comp`): its pipeline (over
/// the gate's layout: two storage buffers and one pushed word), its set and the packed field.
struct Pack {
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    packed: super::native::OwnBuffer,
    texels: u64,
    /// The field the set points at.
    bound: vk::Buffer,
}

impl Pack {
    /// # Safety
    /// `device` is live; `layout` and `set_layout` are the gate's.
    unsafe fn build(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        layout: vk::PipelineLayout,
        set_layout: vk::DescriptorSetLayout,
        texels: u64,
    ) -> Option<Self> {
        // SAFETY: valid create infos; everything made is destroyed on a later failure.
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
            let packed = super::native::OwnBuffer::new(device, instance, physical, texels * 4, usage)?;
            let Some(pipeline) = super::hdr::compute_pipeline(device, layout, PACK_SPV) else {
                packed.destroy(device);
                return None;
            };
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).ok();
            let set = pool.and_then(|pool| {
                device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).ok()
            });
            match (pool, set) {
                (Some(pool), Some(sets)) => Some(Self { pipeline, pool, set: sets[0], packed, texels, bound: vk::Buffer::null() }),
                (pool, _) => {
                    if let Some(pool) = pool {
                        device.destroy_descriptor_pool(pool, None);
                    }
                    device.destroy_pipeline(pipeline, None);
                    packed.destroy(device);
                    None
                }
            }
        }
    }

    /// # Safety
    /// Nothing submitted may still use it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the contract.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            self.packed.destroy(device);
        }
    }
}

/// Page-aligned zeroed memory.
struct Region {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl Region {
    const PAGE: usize = 4096;

    fn new(bytes: u64, alignment: u64) -> Option<Self> {
        let align = (alignment as usize).max(Self::PAGE).next_power_of_two();
        let size = (bytes as usize).div_ceil(align) * align;
        let layout = std::alloc::Layout::from_size_align(size, align).ok()?;
        // SAFETY: a non-zero size (every region has its header).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        (!ptr.is_null()).then_some(Self { ptr, layout })
    }
}

// SAFETY: plain memory; who reads and writes it when is the exchange's protocol.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// What a send copies from: the hold's own buffers and image, as its capture left them.
pub(crate) struct Sources<'a> {
    /// The encoded frame: an image in `GENERAL` with its extent, or a buffer.
    pub frame: FrameSource,
    pub motion: Option<vk::Buffer>,
    pub depth: Option<vk::Buffer>,
    /// The small buffers, each from its start for that many bytes (their sum is at most the layout's `side`).
    pub side: &'a [(vk::Buffer, u64)],
}

#[derive(Clone, Copy)]
pub(crate) enum FrameSource {
    Image {
        image: vk::Image,
        width: u32,
        height: u32,
    },
    /// (the tests' frames)
    #[cfg(test)]
    Buffer(vk::Buffer),
}

/// The worker's end of the exchange: the game's device's events and the regions' memory.
#[derive(Clone)]
pub(crate) struct Link {
    device: ash::Device,
    sent: vk::Event,
    progress: Arc<Progress>,
    out: *mut u8,
    inn: *mut u8,
    /// The regions' whole sizes (rounded up for the import's alignment).
    pub capacity: (usize, usize),
    layout: Layout,
    /// The regions can be imported on another device as host memory (they are this module's own allocations).
    pub shared: bool,
}

// SAFETY: the pointers are the exchange's regions, which outlive the worker ([`Worker`] is stopped before the
// exchange is destroyed); the event is only read and reset by the worker on the host.
unsafe impl Send for Link {}

impl Link {
    #[cfg(test)]
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    /// The regions' memory, for a runner that imports them on its own device.
    #[cfg(test)]
    pub(crate) fn regions(&self) -> (*mut u8, *mut u8) {
        (self.out, self.inn)
    }
}

impl Exchange {
    /// # Safety
    /// `device` is live; `import` says it has `VK_EXT_external_memory_host` enabled.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, sizes: Sizes) -> Option<Self> {
        let layout = Layout::of(sizes);
        let alignment = if import { crate::capture::min_imported_host_pointer_alignment(instance, physical).unwrap_or(0) } else { 0 };
        let regions = [Arc::new(Region::new(layout.out_bytes, alignment)?), Arc::new(Region::new(layout.in_bytes, alignment)?)];
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
        // SAFETY: the regions are valid for their sizes and live as long as the buffers (dropped after them).
        let out = unsafe { super::region_buffer(device, instance, physical, import, regions[0].ptr, regions[0].layout.size(), layout.out_bytes, usage) }?;
        // SAFETY: as above; on a failure what was made is destroyed.
        let Some(inn) = (unsafe { super::region_buffer(device, instance, physical, import, regions[1].ptr, regions[1].layout.size(), layout.in_bytes, usage) })
        else {
            unsafe { out.destroy(device) };
            return None;
        };
        let shared = !out.staged && !inn.staged;
        let b = |n: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(n)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [b(0), b(1)];
        // SAFETY: valid create infos; everything made is destroyed on a later failure.
        unsafe {
            let made = (|| {
                let sent = device.create_event(&vk::EventCreateInfo::default(), None).ok()?;
                let undo_events = || device.destroy_event(sent, None);
                let Ok(set_layout) = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) else {
                    undo_events();
                    return None;
                };
                let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(4);
                let Ok(pipeline_layout) =
                    device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&[push]), None)
                else {
                    device.destroy_descriptor_set_layout(set_layout, None);
                    undo_events();
                    return None;
                };
                let undo = |pipeline: Option<vk::Pipeline>, pool: Option<vk::DescriptorPool>| {
                    if let Some(p) = pool {
                        device.destroy_descriptor_pool(p, None);
                    }
                    if let Some(p) = pipeline {
                        device.destroy_pipeline(p, None);
                    }
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                    undo_events();
                    None
                };
                let Some(pipeline) = super::hdr::compute_pipeline(device, pipeline_layout, GATE_SPV) else { return undo(None, None) };
                let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
                let Ok(pool) = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) else {
                    return undo(Some(pipeline), None);
                };
                let Ok(sets) = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]))
                else {
                    return undo(Some(pipeline), Some(pool));
                };
                Some((sent, set_layout, pipeline_layout, pipeline, pool, sets[0]))
            })();
            let Some((sent, set_layout, pipeline_layout, pipeline, pool, set)) = made else {
                out.destroy(device);
                inn.destroy(device);
                return None;
            };
            let pack = if sizes.packed_texels > 0 { Pack::build(device, instance, physical, pipeline_layout, set_layout, sizes.packed_texels) } else { None };
            if sizes.packed_texels > 0 && pack.is_none() {
                device.destroy_descriptor_pool(pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
                device.destroy_event(sent, None);
                out.destroy(device);
                inn.destroy(device);
                return None;
            }
            Some(Self {
                layout,
                out,
                inn,
                regions,
                shared,
                sent,
                progress: Arc::default(),
                set_layout,
                pipeline_layout,
                pipeline,
                pool,
                set,
                gated: vk::Buffer::null(),
                pack,
            })
        }
    }

    /// The worker's end.
    pub(crate) fn link(&self, device: &ash::Device) -> Link {
        Link {
            device: device.clone(),
            sent: self.sent,
            progress: self.progress.clone(),
            out: self.out.ptr,
            inn: self.inn.ptr,
            capacity: (self.regions[0].layout.size(), self.regions[1].layout.size()),
            layout: self.layout,
            shared: self.shared,
        }
    }

    /// Points the gate at the edit cache's counters `stats`, when they changed.
    ///
    /// # Safety
    /// `stats` is live and at least 64 bytes; nothing that may run uses the set unless it is unchanged.
    pub(crate) unsafe fn bind_gate(&mut self, device: &ash::Device, stats: vk::Buffer) {
        if self.gated == stats {
            return;
        }
        let header = [vk::DescriptorBufferInfo { buffer: self.inn.buffer, offset: 0, range: HEADER_BYTES }];
        let counters = [vk::DescriptorBufferInfo { buffer: stats, offset: 0, range: vk::WHOLE_SIZE }];
        let w =
            |n: u32, i| vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i);
        // SAFETY: the contract.
        unsafe { device.update_descriptor_sets(&[w(0, &header), w(1, &counters)], &[]) };
        self.gated = stats;
    }

    /// Points the motion's pack at the accumulated field `field` (2 x f32 a texel), when it changed and the motion
    /// crosses packed.
    ///
    /// # Safety
    /// `field` is live and holds the layout's packed texels; nothing that may run uses the set unless it is unchanged.
    pub(crate) unsafe fn bind_pack(&mut self, device: &ash::Device, field: vk::Buffer) {
        let Some(pack) = self.pack.as_mut().filter(|p| p.bound != field) else { return };
        let from = [vk::DescriptorBufferInfo { buffer: field, offset: 0, range: pack.texels * 8 }];
        let to = [vk::DescriptorBufferInfo { buffer: pack.packed.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let w =
            |n: u32, i| vk::WriteDescriptorSet::default().dst_set(pack.set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i);
        // SAFETY: the contract.
        unsafe { device.update_descriptor_sets(&[w(0, &from), w(1, &to)], &[]) };
        pack.bound = field;
    }

    /// Whether the motion crosses packed to halves.
    pub(crate) fn motion_half(&self) -> bool {
        self.pack.is_some()
    }

    /// Before the buffer of a hold planned as `plan` goes to the driver (in the order of the submit's buffers):
    /// waits for the worker to be done with the frame sent last ([`Progress::settle`]), writes IN's header for a
    /// hold that takes its answer, and notes this hold's send as part of the submit ([`Self::handed`] after it).
    /// `true` when the hold takes an answer that is there.
    pub(crate) fn at_submit(&self, plan: &HoldPlan) -> bool {
        settle_and_mark(&self.progress, self.inn.ptr, plan)
    }

    /// The submit [`Self::at_submit`] was called for went to the driver: its send can run from here on.
    pub(crate) fn handed(&self) {
        self.progress.handed();
    }

    /// Records the receive of a hold that takes the answer: the answer region ordered after the host's and the
    /// other GPU's writes (with `images`' transitions), `copy` (the hold's copy out of the region: its buffer and
    /// the answer's offset in it), and the copy ordered before whatever reads it next. Nothing waits here
    /// ([`Self::at_submit`] did).
    ///
    /// # Safety
    /// `cmd` is recording on the game's device, outside a render pass; `copy` records a transfer out of the buffer
    /// it is given, at most the layout's answer size from that offset.
    pub(crate) unsafe fn record_receive(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        images: &[vk::ImageMemoryBarrier],
        copy: &dyn Fn(vk::Buffer, u64),
    ) {
        // SAFETY: the contract.
        unsafe {
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                images,
            );
            copy(self.inn.buffer, self.layout.answer_at);
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
    }

    /// Records the gate for a hold that took the answer of frame `seq` ([`Self::bind_gate`]'s counters): it
    /// binds a compute pipeline and set and pushes constants, as the hold's other passes do.
    ///
    /// # Safety
    /// `cmd` is recording, after the receive and whatever stored the edit; the gate is bound.
    pub(crate) unsafe fn record_gate(&self, device: &ash::Device, cmd: vk::CommandBuffer, seq: u32) {
        let order = |src: vk::AccessFlags, dst: vk::AccessFlags| vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst);
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        // SAFETY: the contract.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[order(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE, rw)],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.set), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &seq.to_ne_bytes());
            device.cmd_dispatch(cmd, 1, 1, 1);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[order(vk::AccessFlags::SHADER_WRITE, rw | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)],
                &[],
                &[],
            );
        }
    }

    /// Records the hold's send of its frame as number `seq` (never 0): the motion's pack where it crosses packed (a
    /// compute pipeline, set and pushed word, as the gate's), the copies into OUT, then `sent` set. The hold's submit
    /// settles the send before it first ([`Self::at_submit`]).
    ///
    /// # Safety
    /// `cmd` is recording on the game's device, outside a render pass, after the hold's capture; every source is
    /// live, at least its section's size, and the image is in `GENERAL` with `TRANSFER_SRC` usage; with the motion
    /// packed, the pack is bound to the motion's buffer ([`Self::bind_pack`]).
    pub(crate) unsafe fn record_send(&self, device: &ash::Device, cmd: vk::CommandBuffer, seq: u32, from: &Sources) {
        let l = &self.layout;
        let mut header = [0u32; 8];
        header[OUT_SEQ] = seq;
        // SAFETY: plain words read as bytes.
        let header_bytes = unsafe { std::slice::from_raw_parts(header.as_ptr().cast::<u8>(), std::mem::size_of_val(&header)) };
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
        let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        let copy = |src: vk::Buffer, at: u64, size: u64| {
            if size > 0 {
                // SAFETY: the contract.
                unsafe { device.cmd_copy_buffer(cmd, src, self.out.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: at, size }]) };
            }
        };
        // The motion packed to halves first: what is copied below is the packed field.
        let motion = match (from.motion, self.pack.as_ref()) {
            (Some(field), Some(pack)) => {
                debug_assert_eq!(pack.bound, field, "the pack is bound to the hold's field");
                let field_ready = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                let rows = pack.texels.div_ceil(PACK_ROW) as u32;
                let groups = if rows > 1 { PACK_ROW / PACK_LOCAL } else { pack.texels.div_ceil(PACK_LOCAL) } as u32;
                // SAFETY: the contract.
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[field_ready],
                        &[],
                        &[],
                    );
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pack.pipeline);
                    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&pack.set), &[]);
                    device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, &(pack.texels as u32).to_ne_bytes());
                    device.cmd_dispatch(cmd, groups, rows, 1);
                }
                Some(pack.packed.buffer)
            }
            (motion, _) => motion,
        };
        // SAFETY: the contract.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_update_buffer(cmd, self.out.buffer, 0, header_bytes);
            match from.frame {
                #[cfg(test)]
                FrameSource::Buffer(b) => copy(b, l.frame_at, l.sizes.frame),
                FrameSource::Image { image, width, height } => {
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(l.frame_at)
                        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                        .image_extent(vk::Extent3D { width, height, depth: 1 });
                    device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, self.out.buffer, &[region]);
                }
            }
            if let Some(m) = motion {
                copy(m, l.motion_at, l.sizes.motion);
            }
            if let Some(d) = from.depth {
                copy(d, l.depth_at, l.sizes.depth);
            }
            let mut at = l.side_at;
            for &(buffer, bytes) in from.side {
                debug_assert!(at + bytes <= l.side_at + l.sizes.side);
                copy(buffer, at, bytes);
                at += bytes;
            }
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[to_host],
                &[],
                &[],
            );
            let _event = self.progress.event.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            device.cmd_set_event(cmd, self.sent, vk::PipelineStageFlags::TRANSFER);
        }
    }

    /// # Safety
    /// The worker is stopped, and nothing submitted may still use the exchange (drain before free).
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: the contract.
        unsafe {
            if let Some(pack) = &self.pack {
                pack.destroy(device);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_event(self.sent, None);
            self.out.destroy(device);
            self.inn.destroy(device);
        }
        drop(self.regions);
    }
}

/// The thread that serves the exchange: it runs each sent frame through its [`Runner`] and notes each one finished.
pub(crate) struct Worker {
    quit: Arc<AtomicBool>,
    mode: Arc<AtomicU8>,
    pub counters: Arc<Counters>,
    /// Why the hold's side wants the offload given up ([`Self::give_up`]), until the worker took it.
    asked: Arc<Mutex<Option<String>>>,
    thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    /// When its side last planned a hold ([`now_ms`]).
    touched: Arc<AtomicU64>,
    /// The game's device and the hold it serves.
    game: vk::Device,
    at: Where,
}

/// A running worker, by the game's device and the hold it serves: so the device's destruction can end them (and with
/// them free the other device's side) before the offload device goes ([`stop_for`]), and a side can find who else
/// serves its device ([`serving_at`], [`release_idle`]).
struct Running {
    game: vk::Device,
    at: Where,
    quit: Arc<AtomicBool>,
    thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    touched: Arc<AtomicU64>,
}

/// A hold's side that has had no hold for this long gives the offload device up to another set of hold resources
/// of the same game that waits for it ([`release_idle`]).
pub(crate) const IDLE_SIDE: Duration = Duration::from_secs(3);

fn now_ms() -> u64 {
    static START: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);
    START.elapsed().as_millis() as u64 + 1
}

/// Ends the workers of the game's device `game` that serve the hold `at` and whose side had no hold for `idle`
/// (their runners free their part of the other device's side, and the last one there gives the device back): another
/// set of hold resources at that hold can then take its place. The sides they belonged to find their worker ended
/// ([`Worker::ended`]) and are built again when they next hold. Whether any ended.
pub(crate) fn release_idle(game: vk::Device, at: Where, idle: Duration) -> bool {
    let now = now_ms();
    release(|r| r.game == game && r.at == at && now.saturating_sub(r.touched.load(Ordering::Relaxed)) >= idle.as_millis() as u64)
}

/// Ends the workers `pick` names. A released worker's hold is no longer served: its place on the Status line goes
/// (the side that takes its place, if one does, sets it again when it is built).
fn release(pick: impl Fn(&Running) -> bool) -> bool {
    let ours: Vec<Running> = {
        let mut all = WORKERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !all.iter().any(&pick) {
            return false;
        }
        let (ours, rest) = std::mem::take(&mut *all).into_iter().partition(&pick);
        *all = rest;
        ours
    };
    for r in &ours {
        join(&r.quit, &r.thread);
        set_serving(r.at, false);
    }
    !ours.is_empty()
}
static WORKERS: Mutex<Vec<Running>> = Mutex::new(Vec::new());

fn join(quit: &AtomicBool, thread: &Mutex<Option<std::thread::JoinHandle<()>>>) {
    quit.store(true, Ordering::Release);
    let handle = thread.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
    if let Some(t) = handle {
        let _ = t.join();
    }
}

/// Ends every worker serving the game's device `game`: called inside its `vkDestroyDevice`, before the offload
/// device made with it is destroyed (`offload::destroy_for`). Nothing of the game's device runs any more.
pub(crate) fn stop_for(game: vk::Device) {
    let ours: Vec<Running> = {
        let mut all = WORKERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (ours, rest) = std::mem::take(&mut *all).into_iter().partition(|r| r.game == game);
        *all = rest;
        ours
    };
    for r in ours {
        join(&r.quit, &r.thread);
        // Their sides are destroyed after this and find their workers ended, as released ones do: the status goes here.
        #[cfg(target_arch = "x86_64")]
        side_gone(Some(r.at));
    }
}

/// What the worker tells its owner when it stops serving (it keeps finishing the frames sent, without an answer):
/// why the offload is given up, and whether a frame on the other GPU never finished ([`Runner::stuck`]: the network
/// cannot be taken back, so the model is off from here on).
pub(crate) type GiveUp = Box<dyn Fn(String, bool) + Send>;

impl Worker {
    /// Starts the worker on `link` with `runner`. `late` judges the answers against the game's frame
    /// (`offload::LateRule`, `None` in the `same` test, where the other device shares the game's GPU); `give_up`
    /// is told once when the worker stops serving.
    pub(crate) fn start(
        link: Link,
        at: Where,
        runner: Box<dyn Runner>,
        late: Option<super::offload::LateRule>,
        dims: (u32, u32),
        give_up: GiveUp,
    ) -> Option<Self> {
        let quit = Arc::new(AtomicBool::new(false));
        let mode = Arc::new(AtomicU8::new(SERVING));
        let counters = Arc::new(Counters::default());
        let asked = Arc::new(Mutex::new(None));
        let (q, m, c, a) = (quit.clone(), mode.clone(), counters.clone(), asked.clone());
        let game = link.device.handle();
        let thread =
            std::thread::Builder::new().name("nf-offload-pre".into()).spawn(move || serve(&link, runner, late, dims, &q, &m, &a, &c, &give_up)).ok()?;
        let thread = Arc::new(Mutex::new(Some(thread)));
        let touched = Arc::new(AtomicU64::new(now_ms()));
        WORKERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(Running {
            game,
            at,
            quit: quit.clone(),
            thread: thread.clone(),
            touched: touched.clone(),
        });
        Some(Self { quit, mode, counters, asked, thread, touched, game, at })
    }

    /// Whether frames are run on the other GPU (else every send is finished at once, without an answer).
    pub(crate) fn serving(&self) -> bool {
        self.mode.load(Ordering::Acquire) == SERVING
    }

    /// Its side plans a hold now. A worker of the same game device that serves the other hold, and whose side had
    /// planned no hold for [`IDLE_SIDE`] when this side planned its last one, is ended ([`release`]): the frame no
    /// longer has passes at that hold, and what its runner keeps on the other GPU (a pass at its size) is freed. Measured
    /// against this side's last hold, not against now: after a pause of both holds (a menu, a loading screen)
    /// neither is idle against the other, and both go on.
    pub(crate) fn touch(&self) {
        self.touch_at(now_ms());
    }

    /// [`Self::touch`] at `now` ([`now_ms`]).
    fn touch_at(&self, now: u64) {
        let last = self.touched.swap(now, Ordering::Relaxed);
        let (game, at) = (self.game, self.at);
        release(|r| r.game == game && r.at != at && idle_against(last, r.touched.load(Ordering::Relaxed)));
    }

    /// The worker's thread has ended (stopped, or released for another set of hold resources, [`release_idle`]).
    pub(crate) fn ended(&self) -> bool {
        self.thread.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none()
    }

    /// Gives the offload up from the hold's side (`why`): the worker stops running frames and tells its owner, as
    /// for a failure of its own.
    pub(crate) fn give_up(&self, why: String) {
        *self.asked.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(why);
    }

    /// Stops the worker: no submit waits for it any more ([`Progress::settle`]), and the holds still recorded with
    /// the exchange take no answer.
    pub(crate) fn stop(&mut self) {
        join(&self.quit, &self.thread);
        WORKERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).retain(|r| !Arc::ptr_eq(&r.quit, &self.quit));
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Whether a side whose last hold was at `other` was idle ([`IDLE_SIDE`]) when a side of the same device planned a
/// hold at `last` (both [`now_ms`]).
fn idle_against(last: u64, other: u64) -> bool {
    last.saturating_sub(other) >= IDLE_SIDE.as_millis() as u64
}

/// How long the worker sleeps between two looks at `sent`: short while frames come, longer once none has for a while.
fn poll_pause(idle_polls: u32) -> Duration {
    if idle_polls < 20_000 {
        Duration::from_micros(50)
    } else {
        Duration::from_millis(1)
    }
}

#[allow(clippy::too_many_arguments)]
fn serve(
    link: &Link,
    mut runner: Box<dyn Runner>,
    mut late: Option<super::offload::LateRule>,
    dims: (u32, u32),
    quit: &AtomicBool,
    mode: &AtomicU8,
    asked: &Mutex<Option<String>>,
    counters: &Counters,
    give_up: &GiveUp,
) {
    let device = &link.device;
    let stop_serving = |why: String, stuck: bool| {
        if mode.swap(PASSING, Ordering::AcqRel) == SERVING {
            give_up(why, stuck);
        }
    };
    let mut idle = 0u32;
    let mut last_sent: Option<Instant> = None;
    let mut retired = false;
    while !quit.load(Ordering::Acquire) {
        // Asked from the hold's side to give up ([`Worker::give_up`]).
        if let Some(why) = asked.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            stop_serving(why, false);
        }
        // Given up: the runner lets go of what the game's GPU needs back, once.
        if !retired && mode.load(Ordering::Acquire) != SERVING {
            runner.gave_up();
            retired = true;
        }
        // SAFETY: the game's device and its event, live while the worker runs.
        match unsafe { device.get_event_status(link.sent) } {
            Ok(true) => {}
            Ok(false) => {
                idle = idle.saturating_add(1);
                std::thread::sleep(poll_pause(idle));
                continue;
            }
            Err(_) => {
                // The game's device is lost: none of its holds run any more.
                stop_serving("the game's device was lost".into(), false);
                break;
            }
        }
        idle = 0;
        let now = Instant::now();
        let interval = last_sent.replace(now).map(|t| now - t);
        // SAFETY: as above; the GPU set it for this frame, and no command buffer waits on it.
        let _ = unsafe {
            let _event = link.progress.event.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            device.reset_event(link.sent)
        };
        // SAFETY: OUT's header, written before `sent` was set.
        let seq = unsafe { link.out.cast::<u32>().add(OUT_SEQ).read_volatile() };
        counters.sent.fetch_add(1, Ordering::Relaxed);
        let mut answer = None;
        let mut ok = false;
        if mode.load(Ordering::Acquire) == SERVING {
            // SAFETY: the regions are the exchange's, valid for the layout's sizes.
            match unsafe { runner.run(seq, link.out, link.inn, &link.layout) } {
                Ran::Answered => ok = true,
                Ran::Skipped => {}
                Ran::Failed(why) => stop_serving(why, runner.stuck()),
            }
        }
        // Another send ran while this frame did (a hold whose submit did not settle this one: two holds in one
        // submit, or after an abandoned frame): the game's GPU wrote OUT under the runner, the frame is not this one's.
        // SAFETY: as above.
        let torn =
            ok && (unsafe { link.out.cast::<u32>().add(OUT_SEQ).read_volatile() } != seq || matches!(unsafe { device.get_event_status(link.sent) }, Ok(true)));
        let kept = link.progress.finish(seq, ok && !torn);
        if ok && kept && !torn {
            let took = now.elapsed();
            counters.answered.fetch_add(1, Ordering::Relaxed);
            counters.answer_us.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
            answer = Some(took);
        } else {
            counters.late.fetch_add(1, Ordering::Relaxed);
            if ok {
                // Abandoned by the hold that waited for it, or written over while it ran.
                runner.abandoned();
            }
        }
        if let (Some(rule), Some(interval)) = (late.as_mut(), interval) {
            // A frame the network was not ready for is no late answer.
            if mode.load(Ordering::Acquire) == SERVING && ok {
                if let Some(why) = rule.note(answer, interval, dims) {
                    stop_serving(why, false);
                }
            }
        }
    }
    link.progress.stop();
}

/// What [`Exchange::at_submit`] needs, for the submit of a hold recorded into DLSS's buffer (`device.rs`, where
/// the hold's resources are not at hand): kept by the resources' id while their exchange lives.
#[derive(Clone)]
struct Submits {
    progress: Arc<Progress>,
    /// Keeps IN's memory while a submit writes its header.
    inn: Arc<Region>,
}

static SUBMITS: Mutex<Vec<(u64, Submits)>> = Mutex::new(Vec::new());

/// Before a submit whose buffers hold these holds, in order (`plans`: each with the id of the resources it was
/// recorded with, `inline::Meta`): [`Exchange::at_submit`] for each, then the sends count as handed to the driver
/// (the next submit on the queue comes after this one).
pub(crate) fn at_submit(plans: &[(u64, HoldPlan)]) {
    if plans.is_empty() {
        return;
    }
    let all: Vec<(u64, Submits)> = SUBMITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let mut touched: Vec<&Submits> = Vec::new();
    for (id, plan) in plans {
        let Some((_, s)) = all.iter().find(|(i, _)| i == id) else { continue };
        settle_and_mark(&s.progress, s.inn.ptr, plan);
        touched.push(s);
    }
    for s in touched {
        s.progress.handed();
    }
}

/// [`Exchange::at_submit`]'s body.
fn settle_and_mark(progress: &Progress, inn: *mut u8, plan: &HoldPlan) -> bool {
    let mut taken = false;
    if let Some(seq) = plan.settle {
        let went_out = progress.went_out(seq);
        let ok = progress.settle(seq, ANSWER_BUDGET);
        // IN has one header for every hold of a submit. A hold whose frame did not go out before this submit (sent by
        // an earlier hold of the same submit, or never submitted) cannot have its answer, and its gate already refuses
        // the header's frame: the header is left to the hold before it in this submit, which may have a good answer.
        if plan.take && !went_out {
            progress.untaken.fetch_add(1, Ordering::Relaxed);
        } else if plan.take {
            // SAFETY: IN's header, the exchange's own memory; the GPU reads it only in holds submitted after this.
            unsafe {
                inn.cast::<u32>().add(IN_SEQ).write_volatile(seq);
                inn.cast::<u32>().add(IN_VALID).write_volatile(u32::from(ok));
            }
            std::sync::atomic::fence(Ordering::SeqCst);
            taken = ok;
            if ok { &progress.taken } else { &progress.untaken }.fetch_add(1, Ordering::Relaxed);
        }
    }
    if let Some(seq) = plan.send {
        progress.queue(seq);
    }
    taken
}

/// The devices created with `VK_EXT_external_memory_host` enabled (`device.rs`, at their creation).
static IMPORTING: Mutex<Vec<vk::Device>> = Mutex::new(Vec::new());

/// Notes that `device` can import host memory (`on`), or forgets it (at its destruction).
pub(crate) fn note_import(device: vk::Device, on: bool) {
    let mut all = IMPORTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|d| *d != device);
    if on {
        all.push(device);
    }
}

fn imports(device: vk::Device) -> bool {
    IMPORTING.lock().unwrap_or_else(std::sync::PoisonError::into_inner).contains(&device)
}

/// Whether a worker serves the hold `at` of the game's device `game` already (another set of hold resources at that
/// hold has its place on the offload device).
pub(crate) fn serving_at(game: vk::Device, at: Where) -> bool {
    WORKERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().any(|r| r.game == game && r.at == at)
}

/// Whether the held placements (before DLSS, after its output) are asked onto another GPU and not given up
/// (`NEURAL_FORGE_OFFLOAD_GPU`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn wanted() -> bool {
    super::offload::Choice::current() != super::offload::Choice::Off && !super::offload::given_up() && !super::offload::standin()
}

/// Which holds' models run on the other GPU now (the Status tab's line, [`status_line`]): [`Where::bit`] of the hold
/// before DLSS and of the hold after DLSS's output (both in a frame with passes at both).
static SERVING_HOLD: AtomicU8 = AtomicU8::new(0);

/// The hold `at` sends its frames to the other GPU from here on (`on`), or no longer.
fn set_serving(at: Where, on: bool) {
    if on {
        SERVING_HOLD.fetch_or(at.bit(), Ordering::AcqRel);
    } else {
        SERVING_HOLD.fetch_and(!at.bit(), Ordering::AcqRel);
    }
}

/// No hold sends its frames to the other GPU any more (the offload was given up, or a frame there never finished).
fn clear_serving() {
    SERVING_HOLD.store(0, Ordering::Release);
}

/// The holds whose frames the other GPU runs now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Serving {
    pub before: bool,
    pub after: bool,
}

pub(crate) fn serving() -> Serving {
    let bits = SERVING_HOLD.load(Ordering::Acquire);
    Serving { before: bits & Where::Before.bit() != 0, after: bits & Where::After.bit() != 0 }
}

/// A hold (before DLSS or after its output) sends its frames to the other GPU now.
pub(crate) fn serving_hold() -> bool {
    SERVING_HOLD.load(Ordering::Acquire) != 0
}

/// The network's frames on the other GPU per hold ([`Where::bit`] - 1): how many, and the sum of their times (µs,
/// copies included), for each side's summary line.
static SERVER_TIME: [[AtomicU64; 2]; 2] = [[AtomicU64::new(0), AtomicU64::new(0)], [AtomicU64::new(0), AtomicU64::new(0)]];

/// A frame of the hold `at` took `took` on the other GPU.
pub(crate) fn note_server_frame(at: Where, took: Duration) {
    let [frames, us] = &SERVER_TIME[usize::from(at.bit() - 1)];
    frames.fetch_add(1, Ordering::Relaxed);
    us.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
}

/// What the runners of one game device share on the other GPU (`T`: `offload_server`'s device, loader and queue),
/// by the game's device, with how many runners have a seat.
///
/// The process has one network and the game's device one offload device, and a frame with passes before DLSS and
/// after its output has two holds, each with a side of its own (its exchange, its worker, its runner). Both runners
/// take a seat on the one `T`: the first makes it, the second joins, each locks it for the whole of a frame of its
/// own (so nothing of `T` is used by two frames at once, and every frame is waited for before the lock goes), and the
/// last to leave drops it, still under the registry's lock: a runner that asks for a seat meanwhile waits, and then
/// makes a new `T` from what the old one's drop gave back.
pub(crate) struct Seats<T> {
    all: Mutex<Vec<Seated<T>>>,
}

/// A game's device, what its runners share, and how many of them have a seat.
type Seated<T> = (vk::Device, Arc<Mutex<T>>, usize);

/// One runner's seat ([`Seats::join`]).
pub(crate) struct Seat<T: 'static> {
    seats: &'static Seats<T>,
    game: vk::Device,
    shared: Option<Arc<Mutex<T>>>,
}

impl<T> Seats<T> {
    pub(crate) const fn new() -> Self {
        Self { all: Mutex::new(Vec::new()) }
    }

    /// A seat on `game`'s `T`: the one that is there when `admit` takes it (`Err` with why not: nothing is made in
    /// its place), else a new one from `make`.
    pub(crate) fn join(
        &'static self,
        game: vk::Device,
        admit: impl FnOnce(&T) -> Result<(), String>,
        make: impl FnOnce() -> Result<T, String>,
    ) -> Result<Seat<T>, String> {
        let mut all = self.all.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((_, shared, seats)) = all.iter_mut().find(|(g, _, _)| *g == game) {
            admit(&shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner))?;
            *seats += 1;
            return Ok(Seat { seats: self, game, shared: Some(shared.clone()) });
        }
        let shared = Arc::new(Mutex::new(make()?));
        all.push((game, shared.clone(), 1));
        Ok(Seat { seats: self, game, shared: Some(shared) })
    }

    /// How many runners have a seat on `game`'s `T`.
    #[cfg(test)]
    pub(crate) fn seated(&self, game: vk::Device) -> usize {
        self.all.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(g, _, _)| *g == game).map_or(0, |(_, _, n)| *n)
    }
}

impl<T> Seat<T> {
    /// The shared `T`, for one frame (or one change of it) of this runner's.
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.shared.as_ref().expect("held until the seat is dropped").lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<T> Drop for Seat<T> {
    fn drop(&mut self) {
        let mut all = self.seats.all.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // This seat's own reference first: the registry's is then the last one when the count reaches 0, and `T` is
        // dropped right here, before the registry's lock is released.
        drop(self.shared.take());
        if let Some(k) = all.iter().position(|(g, _, _)| *g == self.game) {
            all[k].2 -= 1;
            if all[k].2 == 0 {
                drop(all.swap_remove(k));
            }
        }
    }
}

/// The hold's side of the offload, for one set of hold resources (one render extent): the exchange, the worker
/// with the network on the other device, the sent frame's kept colour, and the cadence.
#[cfg(target_arch = "x86_64")]
pub(crate) struct HoldSide {
    pub exchange: Exchange,
    cadence: Cadence,
    worker: Worker,
    /// The hold it serves.
    at: Where,
    metas: super::offload_server::Metas,
    /// The colour input of the frame sent last, as the game left it: what the late capture measures the edit
    /// against (the hold's write-back has written the colour input since).
    snapshot: super::hdr::OwnImage,
    /// The hold that sent last and its cache push, for the capture at the next hold.
    sent: Option<(super::cache::Hold, super::cache::CachePush)>,
    resources: u64,
    holds: u64,
    /// The hold planned last has not gone through yet: if it never does, what it planned is dropped at the next plan.
    pending: bool,
    /// Holds in a row whose frame could not be sent (no motion vectors, no jitter, no chained cache), and when the
    /// row began.
    unsendable: Unsendable,
}

/// This many holds in a row that cannot send their frame, over at least [`UNSENDABLE_FOR`], give the offload up: the
/// game's frames do not have what the next frame needs to carry the edit, and the model would not run at all. A
/// few such holds (a cut, a menu's first frames) do not.
pub(crate) const UNSENDABLE_HOLDS: u32 = 30;
pub(crate) const UNSENDABLE_FOR: Duration = Duration::from_secs(1);

/// A pause between two holds longer than this starts the row afresh: the second of [`UNSENDABLE_FOR`] is a second
/// of holds that keep coming, not a pause (a loading screen) between two short bursts.
pub(crate) const UNSENDABLE_GAP: Duration = Duration::from_millis(250);

/// The row of holds that could not send: how many, when it began and when its last hold was.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Unsendable {
    holds: u32,
    began: Option<Instant>,
    last: Option<Instant>,
    /// Given up for this row already.
    said: bool,
}

/// Whether the row of holds that could not send (`streak`) gives the offload up now, after one more hold at `now`
/// that `can` or cannot. Once per row.
pub(crate) fn unsendable_gives_up(streak: &mut Unsendable, can: bool, now: Instant) -> bool {
    if can {
        *streak = Unsendable::default();
        return false;
    }
    let paused = streak.last.is_some_and(|l| now.saturating_duration_since(l) > UNSENDABLE_GAP);
    if streak.began.is_none() || paused {
        *streak = Unsendable { holds: 1, began: Some(now), last: Some(now), said: false };
        return false;
    }
    streak.holds = streak.holds.saturating_add(1);
    streak.last = Some(now);
    let long = streak.began.is_some_and(|b| now.saturating_duration_since(b) >= UNSENDABLE_FOR);
    if !streak.said && streak.holds >= UNSENDABLE_HOLDS && long {
        streak.said = true;
        return true;
    }
    false
}

/// Why the offload gave up on a row of holds that could not send (`why`, the last one's reason): the Status tab's
/// words (short enough for the channel's line).
pub(crate) fn unsendable_line(at: Where, why: &str) -> String {
    format!("{}: its frames cannot be sent: {why}", at.name())
}

#[cfg(target_arch = "x86_64")]
impl HoldSide {
    /// # Safety
    /// `device` is the game's live device; the offload device was made with it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        resources: u64,
        at: Where,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let import = imports(device.handle());
        let sizes = Sizes::of(at, width, height, motion_packed());
        // SAFETY: the contract.
        let exchange =
            unsafe { Exchange::build(device, instance, physical, import, sizes) }.ok_or("the frame's regions could not be made on the game's GPU")?;
        // SAFETY: a live device; destroyed below on a later failure.
        let Some(snapshot) = (unsafe { super::hdr::OwnImage::new(device, instance, physical, width, height, vk::ImageUsageFlags::TRANSFER_DST) }) else {
            unsafe { exchange.destroy(device) };
            return Err("the sent frame's copy could not be made on the game's GPU".into());
        };
        let link = exchange.link(device);
        let metas = super::offload_server::Metas::default();
        let (runner, name) = match super::offload_server::PreRunner::start(instance.clone(), device.handle(), link.shared, link.capacity, metas.clone(), at) {
            Ok(r) => r,
            Err(why) => {
                // SAFETY: never used.
                unsafe {
                    snapshot.destroy(device);
                    exchange.destroy(device);
                }
                return Err(why);
            }
        };
        // Not judged late in the `same` test, where the other device shares the game's GPU (as the finished frame).
        let late = (super::offload::Choice::current() != super::offload::Choice::Same).then(super::offload::LateRule::default);
        let stalled = name.clone();
        let give_up: GiveUp = Box::new(move |why, stuck| {
            // For every hold: both holds' runners share the other device and its network, and a give-up is the process's.
            clear_serving();
            if stuck {
                // The network is still in use by that frame: it cannot go back to the game's GPU. (Said once: the
                // other hold's worker finds the same frame stuck.)
                if !matches!(super::offload::state(), super::offload::State::Stalled(_)) {
                    crate::log!("[offload] {why}: the network cannot be taken back while that frame may run; the model is off for the rest of the session");
                    crate::logging::flush();
                }
                super::offload::note_state(super::offload::State::Stalled(stalled.clone()));
                return;
            }
            if super::offload::given_up() {
                // The other hold's worker gave up first and said why; this one's runner found the network closed.
                return;
            }
            crate::log!("[offload] back to the game's GPU: {why}");
            crate::logging::flush();
            super::offload::note_state(super::offload::State::FellBack(why));
        });
        let Some(worker) = Worker::start(link, at, Box::new(runner), late, (width, height), give_up) else {
            // SAFETY: never used (the runner was dropped with the failed start and gave its device back).
            unsafe {
                snapshot.destroy(device);
                exchange.destroy(device);
            }
            return Err("the offload's worker could not be started".into());
        };
        SUBMITS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((resources, Submits { progress: exchange.progress.clone(), inn: exchange.regions[1].clone() }));
        super::offload::note_state(super::offload::State::Active(name.clone()));
        set_serving(at, true);
        crate::log!(
            "[offload] {}: the network's frames of {width}x{height} run on {name}; the motion crosses as {}; on the game's GPU the frame's regions are {}",
            at.name(),
            if exchange.motion_half() { "two halves a texel" } else { "two floats a texel (NEURAL_FORGE_OFFLOAD_MOTION=float)" },
            if exchange.shared {
                "imported host memory (no CPU copy)"
            } else if import {
                "its own buffers (the import was refused): one more copy on the CPU each way"
            } else {
                "its own buffers (no host-memory import on this device): one more copy on the CPU each way"
            }
        );
        crate::logging::flush();
        Ok(Self {
            exchange,
            cadence: Cadence::default(),
            worker,
            at,
            metas,
            snapshot,
            sent: None,
            resources,
            holds: 0,
            pending: false,
            unsendable: Unsendable::default(),
        })
    }

    /// Whether the other GPU runs the frames (else the offload is given up, and the holds run the network here).
    pub(crate) fn serving(&self) -> bool {
        self.worker.serving() && !self.worker.ended()
    }

    /// Its worker was ended for another set of hold resources ([`release_idle`]): the side is to be built again.
    pub(crate) fn released(&self) -> bool {
        self.worker.ended()
    }

    pub(crate) fn snapshot_view(&self) -> vk::ImageView {
        self.snapshot.view
    }

    /// The cache push of the hold that sent last (the late capture's).
    pub(crate) fn sent_push(&self) -> Option<super::cache::CachePush> {
        self.sent.map(|(_, p)| p)
    }

    /// Plans hold `h`: what it does with the exchange, and, when it takes the answer to the frame sent before it,
    /// that frame's refresh noted in the cache's `state` (before the hold's own plan is made there). `unsendable`:
    /// why this hold's frame cannot be sent (`native::unsendable`), `None` when it can.
    pub(crate) fn plan(&mut self, state: &mut super::cache::State, h: &super::cache::Hold, unsendable: Option<&'static str>) -> HoldPlan {
        let send = unsendable.is_none();
        if std::mem::replace(&mut self.pending, true) {
            // The hold planned before this one never went through (a miss after its plan): it sent nothing.
            self.dropped();
            self.pending = true;
        }
        self.worker.touch();
        if self.serving() && unsendable_gives_up(&mut self.unsendable, send, Instant::now()) {
            let why = unsendable.unwrap_or("");
            crate::log!(
                "[offload] {}: {UNSENDABLE_HOLDS} frames in a row and more than a second could not be sent ({why}): nothing can carry the other GPU's edit to the next frame",
                self.at.name()
            );
            self.worker.give_up(unsendable_line(self.at, why));
        }
        let mut plan = self.cadence.hold(h.hold, send && self.serving());
        // At a hold the game reset, the answer to the frame before is the old scene's: not taken (it still sends).
        if plan.take {
            if let Some((sent, _)) = &self.sent {
                plan.take = state.late_refresh(sent);
            }
        }
        plan
    }

    /// The parameters of the frame a hold is about to send, for the worker: before the send can run (the split
    /// hold's capture submit), once per frame.
    pub(crate) fn queue_meta(&mut self, meta: super::offload_server::FrameMeta) {
        let mut q = self.metas.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if q.back().is_some_and(|m| m.seq == meta.seq) {
            return;
        }
        // Frames whose buffers were never submitted leave theirs behind.
        while q.len() > 64 {
            q.pop_front();
        }
        q.push_back(meta);
    }

    /// The hold planned as `plan` went through: its frame's parameters go to the worker (unless queued already),
    /// and it is the one the next hold's capture measures against.
    pub(crate) fn went_through(
        &mut self,
        h: &super::cache::Hold,
        push: &super::cache::CachePush,
        plan: &HoldPlan,
        meta: Option<super::offload_server::FrameMeta>,
    ) {
        self.pending = false;
        self.sent = plan.send.map(|_| (*h, *push));
        if let Some(meta) = meta.filter(|_| plan.send.is_some()) {
            self.queue_meta(meta);
        }
        self.holds += 1;
        if self.holds.is_multiple_of(600) {
            let p = &self.exchange.progress;
            let (taken, untaken) = (p.taken.swap(0, Ordering::Relaxed), p.untaken.swap(0, Ordering::Relaxed));
            let [frames, us] = &SERVER_TIME[usize::from(self.at.bit() - 1)];
            let (frames, us) = (frames.swap(0, Ordering::Relaxed), us.swap(0, Ordering::Relaxed));
            crate::log!(
                "[offload] {}; {taken} holds carried the edit of the frame before them, {untaken} found no answer; the network's frame on the other GPU {:.2} ms (copies included)",
                self.worker.counters.summary(self.at),
                if frames == 0 { 0.0 } else { us as f64 / frames as f64 / 1000.0 }
            );
        }
    }

    /// The hold planned last recorded nothing after all.
    pub(crate) fn dropped(&mut self) {
        self.pending = false;
        // Its parameters, when they were queued before the send (the split hold's): the frame never ran.
        if let Some(seq) = self.cadence.sent {
            let mut q = self.metas.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.cadence.takeable.is_some() && q.back().is_some_and(|m| m.seq == seq) {
                q.pop_back();
            }
        }
        self.cadence.dropped();
        self.sent = None;
    }

    /// Records the copy of the colour input (`colour`, `GENERAL`, readable by transfers) into the kept image,
    /// left in `GENERAL` for the next hold's capture.
    ///
    /// # Safety
    /// `cmd` is recording on the game's device, outside a render pass, before the hold's write-back.
    pub(crate) unsafe fn record_snapshot(&self, device: &ash::Device, cmd: vk::CommandBuffer, colour: vk::Image, width: u32, height: u32) {
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
        let image = |old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.snapshot.image)
                .subresource_range(range)
        };
        let before = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let copy = vk::ImageCopy::default().src_subresource(layers).dst_subresource(layers).extent(vk::Extent3D { width, height, depth: 1 });
        // SAFETY: the contract.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[image(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)],
            );
            device.cmd_copy_image(cmd, colour, vk::ImageLayout::GENERAL, self.snapshot.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ)],
            );
        }
    }

    /// # Safety
    /// Nothing recorded with the exchange or the kept image may still run (drain before free).
    pub(crate) unsafe fn destroy(mut self, device: &ash::Device) {
        SUBMITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner).retain(|(id, _)| *id != self.resources);
        // The worker first: its runner frees the other device's side and gives the device back.
        let owned = !self.worker.ended();
        self.worker.stop();
        side_gone(owned.then_some(self.at));
        // SAFETY: the contract.
        unsafe {
            self.snapshot.destroy(device);
            self.exchange.destroy(device);
        }
    }
}

/// A hold's side is destroyed. `owned`: the hold it served while its worker still ran, so that hold's place on the
/// offload was its own; a side whose worker was ended ([`release`]: for another set of hold resources at its hold,
/// or because its hold had no passes any more) leaves the status to the sides that serve now. The offload is off
/// once no hold is served.
#[cfg(target_arch = "x86_64")]
fn side_gone(owned: Option<Where>) {
    let Some(at) = owned else { return };
    set_serving(at, false);
    if !serving_hold() && matches!(super::offload::state(), super::offload::State::Active(_)) {
        super::offload::note_state(super::offload::State::Off);
    }
}

/// The Status tab's words while the holds `serving` send their frames to `gpu` (the hold before DLSS when neither
/// is said).
pub(crate) fn status_line(gpu: &str, serving: Serving) -> String {
    match (serving.before, serving.after) {
        (true, true) => format!("second GPU: the model runs on {gpu} before and after DLSS, its edits applied one frame later"),
        (false, true) => format!("second GPU: the model runs on {gpu} {}, its edit applied one frame later", Where::After.name()),
        _ => format!("second GPU: the model runs on {gpu} {}, its edit applied one frame later", Where::Before.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After DLSS's output the hold has no depth at its size: nothing of it crosses, and the rest is the output size's.
    #[test]
    fn the_hold_after_dlss_exchanges_no_depth() {
        let before = Sizes::of(Where::Before, 1485, 835, false);
        let (pw, ph) = crate::preupscale::padded(1485, 835);
        assert_eq!(before.depth, 1485 * 835 * 4);
        assert_eq!((before.frame, before.answer, before.motion), (u64::from(pw * ph) * 8, u64::from(pw * ph) * 8, 1485 * 835 * 8));
        let after = Sizes::of(Where::After, 2560, 1440, false);
        assert_eq!(after.depth, 0);
        let (pw, ph) = crate::preupscale::padded(2560, 1440);
        assert_eq!((after.frame, after.answer, after.motion), (u64::from(pw * ph) * 8, u64::from(pw * ph) * 8, 2560 * 1440 * 8));
        // The side section follows the motion directly, on its alignment, and the regions hold every section.
        let l = Layout::of(after);
        assert_eq!(l.side_at, l.depth_at);
        assert_eq!(l.side_at % HEADER_BYTES, 0);
        assert!(l.out_bytes >= l.side_at + after.side && l.in_bytes >= l.answer_at + after.answer);
        // At 2560x1440: about 59 MB out and 29.5 MB back a frame.
        assert!(l.out_bytes < 60 << 20 && l.in_bytes < 30 << 20, "{l:?}");
    }

    #[test]
    fn the_status_line_names_the_hold_the_other_gpu_serves() {
        let (before, after) = (Serving { before: true, after: false }, Serving { before: false, after: true });
        assert_eq!(status_line("X", before), "second GPU: the model runs on X before DLSS, its edit applied one frame later");
        assert_eq!(status_line("X", after), "second GPU: the model runs on X after DLSS, its edit applied one frame later");
        let both = status_line("X", Serving { before: true, after: true });
        assert_eq!(both, "second GPU: the model runs on X before and after DLSS, its edits applied one frame later");
        // With a long GPU name the line still fits the channel's field whole.
        let long = status_line("NVIDIA GeForce RTX 5080 Laptop GPU", Serving { before: true, after: true });
        assert!(long.len() < neural_forge_protocol::REASON_BYTES, "{} bytes", long.len());
        assert_eq!(Where::of(crate::preupscale::placement::Stage::Upscaled), Where::After);
        assert_eq!(Where::of(crate::preupscale::placement::Stage::Before), Where::Before);
    }

    /// Given up on frames that cannot be sent, the Status tab says which hold and why, within the channel's line.
    #[test]
    fn the_unsendable_line_names_the_hold_and_the_reason() {
        let line = unsendable_line(Where::After, "the motion vectors' layout is unknown");
        assert_eq!(line, "after DLSS: its frames cannot be sent: the motion vectors' layout is unknown");
        let status = format!("second GPU: given up ({line}); the model runs on the game's GPU");
        assert!(status.len() < neural_forge_protocol::REASON_BYTES, "{status}");
    }

    #[test]
    fn the_layout_keeps_every_section_apart_and_aligned() {
        let l = Layout::of(Sizes { frame: 1000, motion: 300, packed_texels: 0, depth: 0, side: 32, answer: 777 });
        assert_eq!(l.frame_at, HEADER_BYTES);
        assert_eq!(l.motion_at, HEADER_BYTES + 1024);
        assert_eq!(l.depth_at, l.motion_at + 512);
        assert_eq!(l.side_at, l.depth_at);
        assert_eq!(l.out_bytes, l.side_at + 256);
        assert_eq!((l.answer_at, l.in_bytes), (HEADER_BYTES, HEADER_BYTES + 1024));
        for at in [l.frame_at, l.motion_at, l.depth_at, l.side_at, l.answer_at] {
            assert_eq!(at % HEADER_BYTES, 0);
        }
    }

    #[test]
    fn a_hold_takes_only_the_answer_of_the_hold_right_before_it() {
        let mut c = Cadence::default();
        let plan = |settle, take, send| HoldPlan { settle, take, send };
        // The first hold has nothing to take or settle.
        assert_eq!(c.hold(10, true), plan(None, false, Some(1)));
        assert_eq!(c.hold(11, true), plan(Some(1), true, Some(2)));
        // A hold that cannot send still takes; the one after it has nothing to take, and settles what was sent last.
        assert_eq!(c.hold(12, false), plan(Some(2), true, None));
        assert_eq!(c.hold(13, true), plan(Some(2), false, Some(3)));
        // A DLSS frame that was not held in between: its answer is another frame's edit, not taken.
        assert_eq!(c.hold(15, true), plan(Some(3), false, Some(4)));
        // A hold that recorded nothing after all sent nothing to take.
        c.dropped();
        assert_eq!(c.hold(16, true), plan(Some(4), false, Some(5)));
    }

    #[test]
    fn the_sequence_number_is_never_0() {
        let mut c = Cadence { next: u32::MAX - 1, ..Default::default() };
        assert_eq!(c.hold(1, true).send, Some(u32::MAX));
        assert_eq!(c.hold(2, true).send, Some(1));
    }

    #[test]
    fn a_submit_waits_only_for_a_frame_that_went_to_the_driver() {
        let p = Progress::default();
        let long = Duration::from_secs(30);
        // Never submitted, or in the submit being made: no wait, no answer.
        let t = Instant::now();
        assert!(!p.settle(1, long));
        p.queue(1);
        assert!(!p.settle(1, long));
        assert!(t.elapsed() < Duration::from_secs(5));
        // Handed and finished: its answer, at once; a failed frame has none.
        p.handed();
        assert!(p.finish(1, true));
        assert!(p.settle(1, long));
        p.queue(2);
        p.handed();
        assert!(p.finish(2, false));
        assert!(!p.settle(2, long));
        // Handed and not finished within the budget: abandoned, and its late answer is not taken.
        p.queue(3);
        p.handed();
        assert!(!p.settle(3, Duration::from_millis(5)));
        assert!(!p.finish(3, true));
        assert!(!p.settle(3, long));
        // The worker stopped: nothing waits.
        p.queue(4);
        p.handed();
        p.stop();
        let t = Instant::now();
        assert!(!p.settle(4, long));
        assert!(t.elapsed() < Duration::from_secs(5));
    }

    /// Two holds in one submit, the first taking the answer to a frame of an earlier submit: the second (whose frame is
    /// the first's, in the same submit) leaves IN's header to the first, so its good answer is kept.
    #[test]
    fn the_second_hold_of_a_submit_leaves_the_header_to_the_first() {
        let p = Progress::default();
        let mut inn = [0u32; (HEADER_BYTES / 4) as usize];
        let at = inn.as_mut_ptr().cast::<u8>();
        p.queue(1);
        p.handed();
        assert!(p.finish(1, true));
        let first = HoldPlan { settle: Some(1), take: true, send: Some(2) };
        let second = HoldPlan { settle: Some(2), take: true, send: Some(3) };
        assert!(settle_and_mark(&p, at, &first));
        assert!(!settle_and_mark(&p, at, &second));
        assert_eq!((inn[IN_VALID], inn[IN_SEQ]), (1, 1), "the header still names frame 1's complete answer");
        // Alone in a later submit, a hold whose frame went out writes it (here: frame 3 failed, no answer).
        p.handed();
        assert!(p.finish(3, false));
        assert!(!settle_and_mark(&p, at, &HoldPlan { settle: Some(3), take: true, send: None }));
        assert_eq!((inn[IN_VALID], inn[IN_SEQ]), (0, 3));
    }

    /// Two frames abandoned before the first finished: neither late answer is taken (one slot kept only the last).
    #[test]
    fn every_abandoned_frame_stays_abandoned() {
        let p = Progress::default();
        for seq in [1, 2] {
            p.queue(seq);
            p.handed();
            assert!(!p.settle(seq, Duration::from_millis(2)));
        }
        assert!(!p.finish(1, true), "frame 1 was abandoned before frame 2 was");
        assert!(!p.finish(2, true));
        assert!(!p.settle(1, Duration::from_secs(30)) && !p.settle(2, Duration::from_secs(30)));
        // The frames after them are answered again.
        p.queue(3);
        p.handed();
        assert!(p.finish(3, true));
        assert!(p.settle(3, Duration::from_secs(30)));
    }

    #[test]
    fn sequence_numbers_compare_across_their_wrap() {
        assert!(after(2, 1) && !after(1, 2) && !after(5, 5));
        assert!(after(1, u32::MAX) && !after(u32::MAX, 1));
        // An abandoned frame from before the wrap does not keep the answers after it out.
        let p = Progress::default();
        p.queue(u32::MAX);
        p.handed();
        assert!(!p.settle(u32::MAX, Duration::from_millis(2)));
        assert!(!p.finish(u32::MAX, true));
        p.queue(1);
        p.handed();
        assert!(p.finish(1, true));
        assert!(p.settle(1, Duration::from_secs(30)));
    }

    #[test]
    fn frames_that_cannot_be_sent_give_up_only_in_a_long_row() {
        let mut streak = Unsendable::default();
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        // One hold without vectors now and then does not.
        for n in 0..200 {
            assert!(!unsendable_gives_up(&mut streak, n % 10 != 0, at(n * 100)));
        }
        // Nor does a quick burst of them: 60 holds within half a second.
        for n in 0..60 {
            assert!(!unsendable_gives_up(&mut streak, false, at(30_000 + n * 8)));
        }
        assert!(!unsendable_gives_up(&mut streak, true, at(31_000)));
        // A row of them over a second does, once.
        let gave_up: Vec<bool> = (0..4 * u64::from(UNSENDABLE_HOLDS)).map(|n| unsendable_gives_up(&mut streak, false, at(40_000 + n * 20))).collect();
        assert_eq!(gave_up.iter().filter(|g| **g).count(), 1);
        assert_eq!(gave_up.iter().position(|g| *g), Some(50), "30 holds are there after 0.6 s, the second is full at hold 51");
        // Two short bursts with a pause between them (a loading screen) are not a second of holds without vectors.
        let mut streak = Unsendable::default();
        for n in 0..29 {
            assert!(!unsendable_gives_up(&mut streak, false, at(60_000 + n * 8)));
        }
        for n in 0..29 {
            assert!(!unsendable_gives_up(&mut streak, false, at(63_000 + n * 8)), "the row began again after the pause");
        }
    }

    #[test]
    fn a_submit_gets_the_answer_that_comes_while_it_waits() {
        let p = Arc::new(Progress::default());
        p.queue(7);
        p.handed();
        let q = p.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            q.finish(7, true)
        });
        assert!(p.settle(7, Duration::from_secs(30)));
        assert!(worker.join().unwrap());
    }

    #[test]
    fn the_summary_counts_and_starts_over() {
        let c = Counters::default();
        c.sent.store(10, Ordering::Relaxed);
        c.answered.store(8, Ordering::Relaxed);
        c.late.store(2, Ordering::Relaxed);
        c.answer_us.store(80_000, Ordering::Relaxed);
        assert_eq!(
            c.summary(Where::Before),
            "before DLSS: 10 frames sent to the other GPU, 8 answered in time (mean 10.0 ms from send to answer), 2 without an answer"
        );
        assert!(c.summary(Where::After).starts_with("after DLSS: 0 frames sent"));
    }

    #[test]
    fn the_worker_looks_less_often_once_no_frame_comes() {
        assert!(poll_pause(0) < Duration::from_millis(1));
        assert_eq!(poll_pause(20_000), Duration::from_millis(1));
    }

    /// One device of the test driver (lavapipe), with host-memory import where it has it.
    struct Dev {
        /// Keeps the Vulkan loader loaded while the device lives.
        _entry: ash::Entry,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        device: ash::Device,
        queue: vk::Queue,
        family: u32,
        import: bool,
    }

    impl Dev {
        fn open() -> Option<Self> {
            if let Some((entry, instance, physical, device, queue, family)) = crate::composition::gpu::test_device_with_external_memory_host() {
                return Some(Self { _entry: entry, instance, physical, device, queue, family, import: true });
            }
            let (entry, instance, physical, device, queue, family) = crate::composition::gpu::test_device()?;
            Some(Self { _entry: entry, instance, physical, device, queue, family, import: false })
        }

        fn close(self) {
            // SAFETY: everything made on the device was destroyed by the test.
            unsafe {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    const TEXELS: u64 = 4096;
    const SIZES: Sizes = Sizes { frame: TEXELS * 4, motion: 1024, packed_texels: 0, depth: 0, side: 16, answer: TEXELS * 4 };

    /// The stand-in for the network on the second device: the answer is the frame, copied on that device's queue
    /// from OUT to IN (imported there where the device can, else staged through its own buffers).
    struct CopyRunner {
        dev: Option<Dev>,
        out: HostBuffer,
        inn: HostBuffer,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
        /// These sequence numbers take longer than a submit waits.
        late: Vec<u32>,
        fail: Option<u32>,
        abandoned: Arc<std::sync::Mutex<Vec<u32>>>,
        last: u32,
        ran: Arc<std::sync::Mutex<Vec<u32>>>,
    }

    // SAFETY: the buffers' pointers are the exchange's regions (or the device's own mappings), used on the worker only.
    unsafe impl Send for CopyRunner {}

    impl CopyRunner {
        fn new(link: &Link, late: Vec<u32>, fail: Option<u32>, ran: Arc<std::sync::Mutex<Vec<u32>>>) -> Option<Self> {
            let dev = Dev::open()?;
            let l = link.layout();
            let (out_ptr, in_ptr) = link.regions();
            let import = dev.import && link.shared;
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            // SAFETY: the regions outlive the runner (the worker is stopped before the exchange goes).
            let out =
                unsafe { crate::preupscale::region_buffer(&dev.device, &dev.instance, dev.physical, import, out_ptr, link.capacity.0, l.out_bytes, usage) }?;
            let inn =
                unsafe { crate::preupscale::region_buffer(&dev.device, &dev.instance, dev.physical, import, in_ptr, link.capacity.1, l.in_bytes, usage) }?;
            // SAFETY: a live device.
            unsafe {
                let pool = dev
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(dev.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .ok()?;
                let cmd = dev.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0];
                let fence = dev.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
                Some(Self { dev: Some(dev), out, inn, pool, cmd, fence, late, fail, abandoned: Arc::default(), last: 0, ran })
            }
        }
    }

    impl Runner for CopyRunner {
        unsafe fn run(&mut self, seq: u32, out: *const u8, inn: *mut u8, layout: &Layout) -> Ran {
            self.last = seq;
            if self.fail == Some(seq) {
                return Ran::Failed("the test's frame failed".into());
            }
            if self.late.contains(&seq) {
                std::thread::sleep(ANSWER_BUDGET + Duration::from_millis(150));
            }
            let d = &self.dev.as_ref().expect("open").device;
            // SAFETY: the test's own objects; the regions as the contract.
            unsafe {
                if self.out.staged {
                    std::ptr::copy_nonoverlapping(out, self.out.ptr, layout.out_bytes as usize);
                }
                d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let copy = vk::BufferCopy { src_offset: layout.frame_at, dst_offset: layout.answer_at, size: layout.sizes.answer.min(layout.sizes.frame) };
                d.cmd_copy_buffer(self.cmd, self.out.buffer, self.inn.buffer, &[copy]);
                d.end_command_buffer(self.cmd).unwrap();
                let cmds = [self.cmd];
                d.queue_submit(self.dev.as_ref().expect("open").queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence).unwrap();
                d.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
                d.reset_fences(&[self.fence]).unwrap();
                d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                if self.inn.staged {
                    std::ptr::copy_nonoverlapping(
                        self.inn.ptr.add(layout.answer_at as usize),
                        inn.add(layout.answer_at as usize),
                        layout.sizes.answer as usize,
                    );
                }
            }
            self.ran.lock().unwrap().push(seq);
            Ran::Answered
        }

        fn abandoned(&mut self) {
            self.abandoned.lock().unwrap().push(self.last);
        }
    }

    impl CopyRunner {
        fn abandoned_list(&self) -> Arc<std::sync::Mutex<Vec<u32>>> {
            self.abandoned.clone()
        }
    }

    impl Drop for CopyRunner {
        fn drop(&mut self) {
            let dev = self.dev.take().expect("open");
            // SAFETY: idle (every frame was waited for).
            unsafe {
                dev.device.destroy_fence(self.fence, None);
                dev.device.destroy_command_pool(self.pool, None);
                self.out.destroy(&dev.device);
                self.inn.destroy(&dev.device);
            }
            dev.close();
        }
    }

    /// The game's side of the test: the exchange, a frame source the test fills, the buffer the answers land in
    /// and the counters the gate writes.
    struct Game {
        dev: Dev,
        exchange: Exchange,
        frame: HostBuffer,
        side: HostBuffer,
        answer: HostBuffer,
        stats: HostBuffer,
        pool: vk::CommandPool,
        fence: vk::Fence,
    }

    impl Game {
        fn new() -> Option<Self> {
            let dev = Dev::open()?;
            let own = |bytes: u64| {
                crate::preupscale::own_host_buffer_with(
                    &dev.device,
                    &dev.instance,
                    dev.physical,
                    bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                )
            };
            // SAFETY: a live device.
            unsafe {
                let mut exchange = Exchange::build(&dev.device, &dev.instance, dev.physical, dev.import, SIZES)?;
                let (frame, side, answer, stats) = (own(SIZES.frame)?, own(SIZES.side)?, own(SIZES.answer)?, own(64)?);
                exchange.bind_gate(&dev.device, stats.buffer);
                let pool = dev.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.family), None).ok()?;
                let fence = dev.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
                Some(Self { dev, exchange, frame, side, answer, stats, pool, fence })
            }
        }

        /// Records a hold as the layer would: the receive and the gate (after the "edit stored" word is set as the
        /// cache's capture sets it), then this hold's frame (every word `value`) and its send.
        fn record(&self, plan: HoldPlan, value: u32) -> (vk::CommandBuffer, HoldPlan) {
            let d = &self.dev.device;
            // SAFETY: the test's own objects.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                d.cmd_fill_buffer(cmd, self.stats.buffer, 14 * 4, 4, u32::from(plan.take));
                if let Some(seq) = plan.settle.filter(|_| plan.take) {
                    let (answer, size) = (self.answer.buffer, SIZES.answer);
                    self.exchange.record_receive(d, cmd, &[], &|region, at| {
                        d.cmd_copy_buffer(cmd, region, answer, &[vk::BufferCopy { src_offset: at, dst_offset: 0, size }])
                    });
                    self.exchange.record_gate(d, cmd, seq);
                }
                if let Some(seq) = plan.send {
                    d.cmd_fill_buffer(cmd, self.frame.buffer, 0, SIZES.frame, value);
                    let side = [(self.side.buffer, SIZES.side)];
                    self.exchange.record_send(d, cmd, seq, &Sources { frame: FrameSource::Buffer(self.frame.buffer), motion: None, depth: None, side: &side });
                }
                d.end_command_buffer(cmd).unwrap();
                (cmd, plan)
            }
        }

        /// Submits the holds in this order in one submit, as the layer's submit hook would, and waits for them.
        fn run(&self, holds: &[(vk::CommandBuffer, HoldPlan)]) {
            let d = &self.dev.device;
            let cmds: Vec<vk::CommandBuffer> = holds.iter().map(|(c, _)| *c).collect();
            for (_, plan) in holds {
                self.exchange.at_submit(plan);
            }
            // SAFETY: the test's own objects.
            unsafe {
                d.queue_submit(self.dev.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence).unwrap();
                self.exchange.handed();
                d.wait_for_fences(&[self.fence], true, 30_000_000_000).expect("the holds finished");
                d.reset_fences(&[self.fence]).unwrap();
            }
        }

        /// The answer buffer's words when they are all the same, and whether the gate left the edit stored.
        fn landed(&self) -> (Option<u32>, bool) {
            // SAFETY: host-visible buffers of the sizes read; the holds were waited for.
            unsafe {
                let words = std::slice::from_raw_parts(self.answer.ptr.cast::<u32>(), (SIZES.answer / 4) as usize);
                let same = words.iter().all(|w| *w == words[0]).then_some(words[0]);
                (same, self.stats.ptr.cast::<u32>().add(14).read_volatile() == 1)
            }
        }

        fn close(self, mut worker: Worker) {
            worker.stop();
            // SAFETY: every hold was waited for and the worker stopped.
            unsafe {
                let d = &self.dev.device;
                d.destroy_fence(self.fence, None);
                d.destroy_command_pool(self.pool, None);
                for b in [&self.frame, &self.side, &self.answer, &self.stats] {
                    b.destroy(d);
                }
                self.exchange.destroy(d);
            }
            self.dev.close();
        }
    }

    type Ranlist = Arc<std::sync::Mutex<Vec<u32>>>;

    /// Waits (20 s at most) until the worker is done with frame `seq`.
    fn wait_finished(game: &Game, seq: u32) {
        let start = Instant::now();
        while game.exchange.progress.lock().finished.is_none_or(|(done, _)| done != seq) {
            assert!(start.elapsed() < Duration::from_secs(20), "the worker finished frame {seq}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[allow(clippy::type_complexity)]
    fn rig(late: Vec<u32>, fail: Option<u32>) -> Option<(Game, Worker, Ranlist, Arc<std::sync::Mutex<Vec<String>>>, Ranlist)> {
        let Some(game) = Game::new() else {
            assert_ne!(neural_forge_protocol::env::var("NEURAL_FORGE_REQUIRE_VULKAN").as_deref(), Some("1"), "no Vulkan device for the exchange's tests");
            return None;
        };
        let link = game.exchange.link(&game.dev.device);
        let ran = Ranlist::default();
        let runner = CopyRunner::new(&link, late, fail, ran.clone())?;
        let abandoned = runner.abandoned_list();
        let gave_up = Arc::new(std::sync::Mutex::new(Vec::new()));
        let g = gave_up.clone();
        let worker = Worker::start(link, Where::Before, Box::new(runner), None, (64, 64), Box::new(move |why, _| g.lock().unwrap().push(why)))?;
        Some((game, worker, ran, gave_up, abandoned))
    }

    /// Two devices: every hold gets exactly the frame the hold before it sent, through the second device.
    #[test]
    fn each_hold_carries_the_answer_to_the_frame_before_it() {
        let Some((game, worker, ran, gave_up, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        for hold in 1..=6u64 {
            let plan = cadence.hold(hold, true);
            // The answer is there before the hold that takes it (the budget is not what this test is about).
            if hold > 1 {
                wait_finished(&game, hold as u32 - 1);
            }
            game.run(&[game.record(plan, 100 + hold as u32)]);
            if hold == 1 {
                assert!(!plan.take);
            } else {
                assert_eq!(game.landed(), (Some(100 + hold as u32 - 1), true), "hold {hold} has the answer to the frame of hold {}", hold - 1);
            }
        }
        // The last frame is answered too.
        wait_finished(&game, 6);
        game.run(&[game.record(cadence.hold(7, false), 0)]);
        assert_eq!(game.landed(), (Some(106), true));
        assert_eq!(*ran.lock().unwrap(), vec![1, 2, 3, 4, 5, 6]);
        assert!(gave_up.lock().unwrap().is_empty());
        assert!(worker.serving());
        assert_eq!(worker.counters.answered.load(Ordering::Relaxed), 6);
        game.close(worker);
    }

    /// DLSS's buffers recorded ahead and submitted one by one: what counts is the order they run in.
    #[test]
    fn holds_recorded_ahead_carry_their_answers_when_submitted_in_turn() {
        let Some((game, worker, ran, _, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        let holds: Vec<_> = (1..=4u64).map(|hold| game.record(cadence.hold(hold, true), 200 + hold as u32)).collect();
        for (i, hold) in holds.iter().enumerate() {
            if i > 0 {
                wait_finished(&game, i as u32);
            }
            game.run(std::slice::from_ref(hold));
            if i > 0 {
                assert_eq!(game.landed(), (Some(200 + i as u32), true));
            }
        }
        // The last hold's frame runs too, after it (whenever the worker gets to it).
        wait_finished(&game, 4);
        assert_eq!(*ran.lock().unwrap(), vec![1, 2, 3, 4]);
        game.close(worker);
    }

    /// The worker resets `sent` on the host while the game's thread records the next hold's set of it: the two are
    /// kept apart (`vkResetEvent` needs the event externally synchronized; the validation layer reports a threading
    /// error otherwise). Many holds, each recorded right after the one before was submitted, as the worker picks it up.
    #[test]
    fn the_worker_s_reset_and_a_hold_s_recording_of_the_event_do_not_overlap() {
        let Some((game, worker, _, gave_up, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        for hold in 1..=200u64 {
            game.run(&[game.record(cadence.hold(hold, true), hold as u32)]);
        }
        wait_finished(&game, 200);
        assert!(gave_up.lock().unwrap().is_empty());
        game.close(worker);
    }

    /// Two holds in one submit: the second cannot have the first's answer (the first has not run when the submit is
    /// made), so it neither waits for it nor takes whatever is in the region.
    #[test]
    fn holds_in_one_submit_do_not_wait_for_each_other() {
        let Some((game, worker, _, _, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        let holds: Vec<_> = (1..=3u64).map(|hold| game.record(cadence.hold(hold, true), 250 + hold as u32)).collect();
        // Counted, not timed: the submit's wall time also holds the device's work and the worker's, which a slow
        // machine (lavapipe under validation) can stretch past the budget without any settle waiting.
        game.run(&holds);
        assert_eq!(game.exchange.progress.blocked.load(Ordering::Relaxed), 0, "no submit waited for another's frame");
        assert_eq!(game.exchange.progress.lock().abandoned, None, "no submit waited out its budget");
        assert!(!game.landed().1, "no edit: the answer was not there when the submit was made");
        // The stream goes on from the last frame sent, once the worker is done with it (however long the machine takes;
        // the hold's budget is not what this test is about).
        wait_finished(&game, 3);
        game.run(&[game.record(cadence.hold(4, true), 254)]);
        assert_eq!(game.landed(), (Some(253), true));
        game.close(worker);
    }

    /// Two holds in one submit while the other GPU is still on the first's frame: the second's send writes OUT under
    /// the runner, so the first frame's answer is not kept and the runner is told (its history starts afresh).
    #[test]
    fn a_frame_written_over_while_it_runs_is_not_answered() {
        let Some((game, worker, ran, gave_up, abandoned)) = rig(vec![1], None) else { return };
        let mut cadence = Cadence::default();
        let holds: Vec<_> = (1..=2u64).map(|hold| game.record(cadence.hold(hold, true), 700 + hold as u32)).collect();
        game.run(&holds);
        // Frame 2 is answered once the worker is past frame 1; hold 3 carries it.
        wait_finished(&game, 2);
        game.run(&[game.record(cadence.hold(3, false), 0)]);
        assert_eq!(game.landed(), (Some(702), true));
        // Whether the worker woke between the two sends is up to the scheduler: if it ran frame 1, frame 2's send
        // wrote OUT under it and frame 1 is not answered (the runner is told); if it only ever saw frame 2, frame 1 was
        // never run and there is nothing to tell. Either way only frame 2 is answered.
        if ran.lock().unwrap().first() == Some(&1) {
            assert_eq!(*abandoned.lock().unwrap(), vec![1], "frame 1 was written over by frame 2's send while it ran");
        } else {
            assert_eq!(*ran.lock().unwrap(), vec![2], "the worker woke after both sends and ran only frame 2");
            assert!(abandoned.lock().unwrap().is_empty(), "frame 1 never ran: nothing to abandon");
        }
        assert_eq!(worker.counters.answered.load(Ordering::Relaxed), 1);
        assert!(gave_up.lock().unwrap().is_empty());
        game.close(worker);
    }

    /// A hold that was recorded but never submitted: the hold after it neither waits nor takes another frame's
    /// answer as its own.
    #[test]
    fn a_hold_never_submitted_leaves_the_next_without_an_edit() {
        let Some((game, worker, _, _, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        game.run(&[game.record(cadence.hold(1, true), 301)]);
        let _never = game.record(cadence.hold(2, true), 302);
        let blocked = game.exchange.progress.blocked.load(Ordering::Relaxed);
        game.run(&[game.record(cadence.hold(3, true), 303)]);
        assert_eq!(game.exchange.progress.blocked.load(Ordering::Relaxed), blocked, "hold 3 did not wait for frame 2");
        assert!(!game.landed().1, "frame 2 never ran: no edit at hold 3");
        wait_finished(&game, 3);
        game.run(&[game.record(cadence.hold(4, true), 304)]);
        assert_eq!(game.landed(), (Some(303), true));
        game.close(worker);
    }

    /// A frame the other GPU does not answer within the budget: the submit goes on without it, the late answer is
    /// never taken, the runner is told, and the frames after it are answered again.
    #[test]
    fn a_late_answer_lets_the_hold_through_without_an_edit() {
        let Some((game, worker, _, gave_up, abandoned)) = rig(vec![2], None) else { return };
        let mut cadence = Cadence::default();
        for hold in 1..=3u64 {
            game.run(&[game.record(cadence.hold(hold, true), 400 + hold as u32)]);
            match hold {
                2 => assert_eq!(game.landed(), (Some(401), true)),
                3 => assert!(!game.landed().1, "frame 2 was not answered in time: no edit at hold 3"),
                _ => {}
            }
        }
        // Frame 3 waits behind the late frame on the worker; give it its time, then the stream is whole again.
        std::thread::sleep(Duration::from_millis(300));
        game.run(&[game.record(cadence.hold(4, true), 404)]);
        game.run(&[game.record(cadence.hold(5, false), 0)]);
        assert_eq!(game.landed(), (Some(404), true));
        assert_eq!(*abandoned.lock().unwrap(), vec![2]);
        assert!(worker.counters.late.load(Ordering::Relaxed) >= 1);
        assert!(gave_up.lock().unwrap().is_empty());
        game.close(worker);
    }

    /// A frame that cannot run on the other GPU gives the offload up, once; the holds still recorded with the
    /// exchange go through without an edit and without waiting.
    #[test]
    fn a_failed_frame_gives_up_and_the_holds_still_go_through() {
        let Some((game, worker, ran, gave_up, _)) = rig(Vec::new(), Some(2)) else { return };
        let mut cadence = Cadence::default();
        for hold in 1..=5u64 {
            game.run(&[game.record(cadence.hold(hold, true), 500 + hold as u32)]);
            if hold >= 3 {
                assert!(!game.landed().1, "no edit once the offload is given up (hold {hold})");
            }
        }
        assert_eq!(*ran.lock().unwrap(), vec![1]);
        assert_eq!(*gave_up.lock().unwrap(), vec!["the test's frame failed".to_string()]);
        assert!(!worker.serving());
        game.close(worker);
    }

    /// A side that had no hold for a while gives its worker up to another set of hold resources of the same device.
    #[test]
    fn an_idle_side_is_released_for_another() {
        let _state = crate::preupscale::offload::TEST_STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some((game, worker, _, _, _)) = rig(Vec::new(), None) else { return };
        let device = game.dev.device.handle();
        assert!(serving_at(device, Where::Before) && !serving_at(device, Where::After));
        worker.touch();
        assert!(!release_idle(device, Where::Before, Duration::from_secs(3600)), "it just held");
        assert!(!release_idle(device, Where::After, Duration::ZERO), "the other hold's place is not this side's");
        assert!(!worker.ended());
        clear_serving();
        set_serving(Where::Before, true);
        assert!(release_idle(device, Where::Before, Duration::ZERO));
        assert!(worker.ended() && !serving_at(device, Where::Before));
        assert!(!serving_hold(), "its hold is not served until the side that takes its place is built");
        // The released side's end leaves the status to the side that serves now; an owning side's end clears it.
        set_serving(Where::Before, true);
        side_gone((!worker.ended()).then_some(Where::Before));
        assert!(serving_hold(), "another set's side serves: the Status tab keeps the second GPU");
        side_gone(Some(Where::Before));
        assert!(!serving_hold());
        // Its side's submits do not wait for it.
        let mut cadence = Cadence::default();
        game.run(&[game.record(cadence.hold(1, true), 801)]);
        let blocked = game.exchange.progress.blocked.load(Ordering::Relaxed);
        game.run(&[game.record(cadence.hold(2, false), 0)]);
        assert_eq!(game.exchange.progress.blocked.load(Ordering::Relaxed), blocked, "hold 2 did not wait for the released worker");
        game.close(worker);
    }

    /// The worker stopped: the submit of a hold recorded with a receive does not wait.
    #[test]
    fn a_stopped_worker_leaves_the_submits_free() {
        let Some((game, mut worker, _, _, _)) = rig(Vec::new(), None) else { return };
        let mut cadence = Cadence::default();
        game.run(&[game.record(cadence.hold(1, true), 601)]);
        let start = Instant::now();
        while worker.counters.answered.load(Ordering::Relaxed) == 0 && start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(1));
        }
        game.run(&[game.record(cadence.hold(2, true), 602)]);
        assert_eq!(game.landed(), (Some(601), true));
        worker.stop();
        let blocked = game.exchange.progress.blocked.load(Ordering::Relaxed);
        game.run(&[game.record(cadence.hold(3, false), 0)]);
        assert_eq!(game.exchange.progress.blocked.load(Ordering::Relaxed), blocked, "hold 3 did not wait for the stopped worker");
        // Frame 2's answer when the worker finished it before it stopped, else none: never a wait, never another frame's.
        let landed = game.landed();
        assert!(landed == (Some(602), true) || !landed.1, "{landed:?}");
        game.close(worker);
    }

    /// The Status line names both holds while both send their frames, and each alone otherwise; a hold's end leaves
    /// the other's place.
    #[test]
    fn the_serving_holds_are_kept_apart() {
        let _state = crate::preupscale::offload::TEST_STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        clear_serving();
        set_serving(Where::Before, true);
        set_serving(Where::After, true);
        assert_eq!(serving(), Serving { before: true, after: true });
        side_gone(None);
        assert_eq!(serving(), Serving { before: true, after: true }, "a released side's end changes nothing");
        side_gone(Some(Where::Before));
        assert_eq!(serving(), Serving { before: false, after: true });
        assert!(serving_hold());
        side_gone(Some(Where::After));
        assert!(!serving_hold());
    }

    /// A side is idle against another only when it had no hold for the idle time by that side's last hold.
    #[test]
    fn a_side_is_idle_only_against_holds_that_kept_coming() {
        let idle = IDLE_SIDE.as_millis() as u64;
        assert!(!idle_against(10_000, 10_000) && !idle_against(10_000, 9_990), "both hold");
        assert!(!idle_against(10_000, 10_000 - idle + 1));
        assert!(idle_against(10_000, 10_000 - idle));
        assert!(!idle_against(5_000, 20_000), "the other held since");
    }

    /// What two runners share in the seats' tests: how many were made and dropped, and how many are alive.
    #[derive(Default)]
    struct Tally {
        made: AtomicU64,
        dropped: AtomicU64,
        alive: AtomicU64,
        most: AtomicU64,
    }

    struct Shared(Arc<Tally>);

    impl Shared {
        fn make(tally: &Arc<Tally>) -> Self {
            tally.made.fetch_add(1, Ordering::SeqCst);
            let alive = tally.alive.fetch_add(1, Ordering::SeqCst) + 1;
            tally.most.fetch_max(alive, Ordering::SeqCst);
            Self(tally.clone())
        }
    }

    impl Drop for Shared {
        fn drop(&mut self) {
            self.0.dropped.fetch_add(1, Ordering::SeqCst);
            self.0.alive.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn key(n: u64) -> vk::Device {
        use ash::vk::Handle;
        vk::Device::from_raw(n)
    }

    /// The second runner of a game device joins what the first made; it is dropped once, when the last leaves, and
    /// the next runner makes a new one. Another game device has its own.
    #[test]
    fn two_runners_of_a_device_share_one_seat_and_the_last_to_leave_drops_it() {
        static SEATS: Seats<Shared> = Seats::new();
        let tally = Arc::new(Tally::default());
        let (game, other) = (key(0x5ea7_0001), key(0x5ea7_0002));
        let a = SEATS.join(game, |_| Ok(()), || Ok(Shared::make(&tally))).unwrap();
        let b = SEATS.join(game, |_| Ok(()), || Err("the second runner makes nothing".into())).unwrap();
        // One at a time: both seats lock the same thing.
        let first = a.lock().0.clone();
        assert!(Arc::ptr_eq(&first, &b.lock().0));
        assert_eq!((SEATS.seated(game), tally.made.load(Ordering::SeqCst)), (2, 1));
        let c = SEATS.join(other, |_| Ok(()), || Ok(Shared::make(&tally))).unwrap();
        assert_eq!((SEATS.seated(other), tally.made.load(Ordering::SeqCst)), (1, 2));
        drop(a);
        assert_eq!((SEATS.seated(game), tally.dropped.load(Ordering::SeqCst)), (1, 0), "the other runner still has it");
        // One that is there and not admitted: no seat, and nothing made in its place.
        assert_eq!(SEATS.join(game, |_| Err("stuck".into()), || Ok(Shared::make(&tally))).err().as_deref(), Some("stuck"));
        assert_eq!((SEATS.seated(game), tally.made.load(Ordering::SeqCst)), (1, 2));
        drop(b);
        assert_eq!((SEATS.seated(game), tally.dropped.load(Ordering::SeqCst)), (0, 1));
        let d = SEATS.join(game, |_| Ok(()), || Ok(Shared::make(&tally))).unwrap();
        assert_eq!(tally.made.load(Ordering::SeqCst), 3, "a new one after the last left");
        // A failed make leaves no seat behind.
        let third = key(0x5ea7_0003);
        assert!(SEATS.join(third, |_| Ok(()), || Err::<Shared, _>("no device".into())).is_err());
        assert_eq!(SEATS.seated(third), 0);
        drop((c, d));
        assert_eq!((tally.dropped.load(Ordering::SeqCst), tally.alive.load(Ordering::SeqCst)), (3, 0));
    }

    /// Runners that come and go on several threads: a game device never has two of what they share alive at once
    /// (the offload device is taken once and given back before it is taken again), and every one made is dropped.
    #[test]
    fn runners_coming_and_going_never_make_a_second_one_beside_the_first() {
        static SEATS: Seats<Shared> = Seats::new();
        let tally = Arc::new(Tally::default());
        let game = key(0x5ea7_0010);
        let threads: Vec<_> = (0..6)
            .map(|t| {
                let tally = tally.clone();
                std::thread::spawn(move || {
                    for n in 0..400u32 {
                        let seat = SEATS.join(game, |_| Ok(()), || Ok(Shared::make(&tally))).unwrap();
                        {
                            let shared = seat.lock();
                            assert_eq!(shared.0.alive.load(Ordering::SeqCst), 1, "one alive while a runner has a seat");
                        }
                        if (n + t) % 3 == 0 {
                            std::thread::yield_now();
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(tally.most.load(Ordering::SeqCst), 1);
        assert_eq!(tally.made.load(Ordering::SeqCst), tally.dropped.load(Ordering::SeqCst));
        assert_eq!(SEATS.seated(game), 0);
    }

    /// A core no runner may run a frame on: a frame never finished there (which is a stall), or the offload was given
    /// up and the network closed.
    #[test]
    fn a_stuck_or_closed_core_takes_no_frame() {
        use crate::preupscale::offload_server::refused;
        assert_eq!(refused(false, false), None);
        assert!(refused(true, false).is_some_and(|(_, stuck)| stuck));
        assert!(refused(true, true).is_some_and(|(_, stuck)| stuck), "a stall first: nothing may be taken back");
        assert!(refused(false, true).is_some_and(|(_, stuck)| !stuck));
    }

    type Log = Arc<std::sync::Mutex<Vec<&'static str>>>;

    /// The second device as both holds' runners share it (the tests' `offload_server::Core`).
    struct Other {
        dev: Option<Dev>,
        fence: vk::Fence,
        /// Frames submitted to it.
        submits: u32,
        /// A frame of one runner is on the device now.
        running: bool,
        stuck: bool,
        closed: bool,
        log: Log,
    }

    // SAFETY: the test's own device, used under the seats' lock.
    unsafe impl Send for Other {}

    impl Drop for Other {
        fn drop(&mut self) {
            self.log.lock().unwrap().push("dropped");
            let dev = self.dev.take().expect("open");
            // SAFETY: idle; every runner freed what it made before it left its seat.
            unsafe { dev.device.destroy_fence(self.fence, None) };
            dev.close();
        }
    }

    static OTHERS: Seats<Other> = Seats::new();

    /// A hold's runner on the shared second device: its own imports of its exchange's regions and its own command
    /// buffer; the device, its queue and its fence are the seat's, locked for the whole of a frame.
    struct SeatedRunner {
        seat: Seat<Other>,
        device: ash::Device,
        queue: vk::Queue,
        out: HostBuffer,
        inn: HostBuffer,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        /// This frame fails (a give-up); this one never finishes (a stall).
        fail: Option<u32>,
        stick: Option<u32>,
        stuck: bool,
        ran: Ranlist,
    }

    // SAFETY: as `CopyRunner`.
    unsafe impl Send for SeatedRunner {}

    impl SeatedRunner {
        fn new(game: vk::Device, link: &Link, log: &Log, fail: Option<u32>, stick: Option<u32>, ran: Ranlist) -> Option<Self> {
            use crate::preupscale::offload_server::refused;
            let seat = OTHERS
                .join(
                    game,
                    |o| refused(o.stuck, o.closed).map_or(Ok(()), |(why, _)| Err(why.to_string())),
                    || {
                        let dev = Dev::open().ok_or("no device")?;
                        // SAFETY: a live device.
                        let fence = unsafe { dev.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(|e| e.to_string())?;
                        log.lock().unwrap().push("made");
                        Ok(Other { dev: Some(dev), fence, submits: 0, running: false, stuck: false, closed: false, log: log.clone() })
                    },
                )
                .ok()?;
            let (device, instance, physical, queue, family, import) = {
                let o = seat.lock();
                let d = o.dev.as_ref().expect("open");
                (d.device.clone(), d.instance.clone(), d.physical, d.queue, d.family, d.import && link.shared)
            };
            let l = link.layout();
            let (out_ptr, in_ptr) = link.regions();
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            // SAFETY: the regions outlive the runner; a live device, used here under no frame (none was sent yet).
            unsafe {
                let out = crate::preupscale::region_buffer(&device, &instance, physical, import, out_ptr, link.capacity.0, l.out_bytes, usage)?;
                let inn = crate::preupscale::region_buffer(&device, &instance, physical, import, in_ptr, link.capacity.1, l.in_bytes, usage)?;
                let pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .ok()?;
                let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0];
                Some(Self { seat, device, queue, out, inn, pool, cmd, fail, stick, stuck: false, ran })
            }
        }
    }

    impl Runner for SeatedRunner {
        unsafe fn run(&mut self, seq: u32, out: *const u8, inn: *mut u8, layout: &Layout) -> Ran {
            let mut other = self.seat.lock();
            if let Some((why, stuck)) = crate::preupscale::offload_server::refused(other.stuck, other.closed) {
                self.stuck |= stuck;
                return Ran::Failed(why.into());
            }
            if self.stick == Some(seq) {
                other.stuck = true;
                self.stuck = true;
                return Ran::Failed("the test's frame never finished".into());
            }
            if self.fail == Some(seq) {
                return Ran::Failed("the test's frame failed".into());
            }
            assert!(!std::mem::replace(&mut other.running, true), "one frame at a time on the shared device");
            other.submits += 1;
            let d = &self.device;
            // SAFETY: the test's own objects; the regions as the contract; the device's queue and fence under the seat's lock.
            unsafe {
                if self.out.staged {
                    std::ptr::copy_nonoverlapping(out, self.out.ptr, layout.out_bytes as usize);
                }
                d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let copy = vk::BufferCopy { src_offset: layout.frame_at, dst_offset: layout.answer_at, size: layout.sizes.answer.min(layout.sizes.frame) };
                d.cmd_copy_buffer(self.cmd, self.out.buffer, self.inn.buffer, &[copy]);
                d.end_command_buffer(self.cmd).unwrap();
                let cmds = [self.cmd];
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], other.fence).unwrap();
                d.wait_for_fences(&[other.fence], true, u64::MAX).unwrap();
                d.reset_fences(&[other.fence]).unwrap();
                d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                if self.inn.staged {
                    std::ptr::copy_nonoverlapping(
                        self.inn.ptr.add(layout.answer_at as usize),
                        inn.add(layout.answer_at as usize),
                        layout.sizes.answer as usize,
                    );
                }
            }
            other.running = false;
            self.ran.lock().unwrap().push(seq);
            Ran::Answered
        }

        fn gave_up(&mut self) {
            let mut other = self.seat.lock();
            if !other.stuck {
                other.closed = true;
            }
        }

        fn stuck(&self) -> bool {
            self.stuck
        }
    }

    impl Drop for SeatedRunner {
        fn drop(&mut self) {
            let _other = self.seat.lock();
            // SAFETY: idle (every frame was waited for under the seat's lock).
            unsafe {
                self.device.destroy_command_pool(self.pool, None);
                self.out.destroy(&self.device);
                self.inn.destroy(&self.device);
            }
        }
    }

    /// One hold of the frame on the game's device: its exchange at its own size, the frame it sends, where its
    /// answers land and the counters its gate writes.
    struct HoldRig {
        exchange: Exchange,
        sizes: Sizes,
        frame: HostBuffer,
        side: HostBuffer,
        answer: HostBuffer,
        stats: HostBuffer,
    }

    /// The game's device with both holds of a frame: before DLSS (the smaller frame) and after its output.
    struct Pair {
        dev: Dev,
        holds: [HoldRig; 2],
        pool: vk::CommandPool,
        fence: vk::Fence,
    }

    const SIZES_AFTER: Sizes = Sizes { frame: TEXELS * 12, motion: 2048, packed_texels: 0, depth: 0, side: 16, answer: TEXELS * 12 };

    impl Pair {
        fn new() -> Option<Self> {
            let dev = Dev::open()?;
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
            let hold = |sizes: Sizes| {
                let own = |bytes: u64| crate::preupscale::own_host_buffer_with(&dev.device, &dev.instance, dev.physical, bytes, usage);
                // SAFETY: a live device.
                unsafe {
                    let mut exchange = Exchange::build(&dev.device, &dev.instance, dev.physical, dev.import, sizes)?;
                    let (frame, side, answer, stats) = (own(sizes.frame)?, own(sizes.side)?, own(sizes.answer)?, own(64)?);
                    exchange.bind_gate(&dev.device, stats.buffer);
                    Some(HoldRig { exchange, sizes, frame, side, answer, stats })
                }
            };
            let holds = [hold(SIZES)?, hold(SIZES_AFTER)?];
            // SAFETY: a live device.
            unsafe {
                let pool = dev.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.family), None).ok()?;
                let fence = dev.device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
                Some(Self { dev, holds, pool, fence })
            }
        }

        /// Records hold `k` (0 before DLSS, 1 after) as the layer would ([`Game::record`]).
        fn record(&self, k: usize, plan: HoldPlan, value: u32) -> (usize, vk::CommandBuffer, HoldPlan) {
            let (d, h) = (&self.dev.device, &self.holds[k]);
            // SAFETY: the test's own objects.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                d.cmd_fill_buffer(cmd, h.stats.buffer, 14 * 4, 4, u32::from(plan.take));
                if let Some(seq) = plan.settle.filter(|_| plan.take) {
                    let (answer, size) = (h.answer.buffer, h.sizes.answer);
                    h.exchange.record_receive(d, cmd, &[], &|region, at| {
                        d.cmd_copy_buffer(cmd, region, answer, &[vk::BufferCopy { src_offset: at, dst_offset: 0, size }])
                    });
                    h.exchange.record_gate(d, cmd, seq);
                }
                if let Some(seq) = plan.send {
                    d.cmd_fill_buffer(cmd, h.frame.buffer, 0, h.sizes.frame, value);
                    let side = [(h.side.buffer, h.sizes.side)];
                    h.exchange.record_send(d, cmd, seq, &Sources { frame: FrameSource::Buffer(h.frame.buffer), motion: None, depth: None, side: &side });
                }
                d.end_command_buffer(cmd).unwrap();
                (k, cmd, plan)
            }
        }

        /// Submits the holds in this order in one submit, as the layer's submit hook does for a frame whose two holds
        /// are in one submit ([`at_submit`]: each hold's exchange settles its own frame), and waits for them.
        fn run(&self, holds: &[(usize, vk::CommandBuffer, HoldPlan)]) {
            let d = &self.dev.device;
            let cmds: Vec<vk::CommandBuffer> = holds.iter().map(|(_, c, _)| *c).collect();
            for (k, _, plan) in holds {
                self.holds[*k].exchange.at_submit(plan);
            }
            // SAFETY: the test's own objects.
            unsafe {
                d.queue_submit(self.dev.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence).unwrap();
                for (k, _, _) in holds {
                    self.holds[*k].exchange.handed();
                }
                d.wait_for_fences(&[self.fence], true, 30_000_000_000).expect("the holds finished");
                d.reset_fences(&[self.fence]).unwrap();
            }
        }

        fn landed(&self, k: usize) -> (Option<u32>, bool) {
            let h = &self.holds[k];
            // SAFETY: host-visible buffers of the sizes read; the holds were waited for.
            unsafe {
                let words = std::slice::from_raw_parts(h.answer.ptr.cast::<u32>(), (h.sizes.answer / 4) as usize);
                let same = words.iter().all(|w| *w == words[0]).then_some(words[0]);
                (same, h.stats.ptr.cast::<u32>().add(14).read_volatile() == 1)
            }
        }

        fn wait_finished(&self, k: usize, seq: u32) {
            let start = Instant::now();
            while self.holds[k].exchange.progress.lock().finished.is_none_or(|(done, _)| done != seq) {
                assert!(start.elapsed() < Duration::from_secs(20), "the worker finished frame {seq} of hold {k}");
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        fn close(self, workers: Vec<Worker>) {
            for mut w in workers {
                w.stop();
            }
            // SAFETY: every hold was waited for and the workers stopped.
            unsafe {
                let d = &self.dev.device;
                d.destroy_fence(self.fence, None);
                d.destroy_command_pool(self.pool, None);
                for h in self.holds {
                    for b in [&h.frame, &h.side, &h.answer, &h.stats] {
                        b.destroy(d);
                    }
                    h.exchange.destroy(d);
                }
            }
            self.dev.close();
        }
    }

    type GaveUp = Arc<std::sync::Mutex<Vec<(usize, String, bool)>>>;

    /// Both holds of a frame on the game's device, each with its worker; their runners share the second device.
    /// `fail` and `stick`: (hold, frame) whose frame fails there, or never finishes.
    #[allow(clippy::type_complexity)]
    fn pair(fail: Option<(usize, u32)>, stick: Option<(usize, u32)>) -> Option<(Pair, Vec<Worker>, [Ranlist; 2], GaveUp, Log)> {
        let Some(pair) = Pair::new() else {
            assert_ne!(neural_forge_protocol::env::var("NEURAL_FORGE_REQUIRE_VULKAN").as_deref(), Some("1"), "no Vulkan device for the exchange's tests");
            return None;
        };
        let game = pair.dev.device.handle();
        let (log, gave_up) = (Log::default(), GaveUp::default());
        let ran = [Ranlist::default(), Ranlist::default()];
        let mut workers = Vec::new();
        for (k, at) in [Where::Before, Where::After].into_iter().enumerate() {
            let link = pair.holds[k].exchange.link(&pair.dev.device);
            let only = |x: Option<(usize, u32)>| x.filter(|(hold, _)| *hold == k).map(|(_, seq)| seq);
            let runner = SeatedRunner::new(game, &link, &log, only(fail), only(stick), ran[k].clone())?;
            let g = gave_up.clone();
            workers.push(Worker::start(link, at, Box::new(runner), None, (64, 64), Box::new(move |why, stuck| g.lock().unwrap().push((k, why, stuck))))?);
        }
        assert_eq!(OTHERS.seated(game), 2);
        assert_eq!(*log.lock().unwrap(), vec!["made"], "the second hold's runner joined the first's device");
        Some((pair, workers, ran, gave_up, log))
    }

    /// A frame with passes before DLSS and after its output: both holds, recorded into one submit, each carry the
    /// answer to their own frame before, through the one second device, at their own sizes.
    #[test]
    fn both_holds_of_a_frame_carry_their_answers_through_one_device() {
        let Some((pair, workers, ran, gave_up, log)) = pair(None, None) else { return };
        let game = pair.dev.device.handle();
        let mut cadence = [Cadence::default(), Cadence::default()];
        for frame in 1..=8u64 {
            let n = frame as u32;
            if frame > 1 {
                // The answers are there before the holds that take them (the budget is not what this test is about).
                pair.wait_finished(0, n - 1);
                pair.wait_finished(1, n - 1);
            }
            let before = pair.record(0, cadence[0].hold(frame, true), 100 + n);
            let after = pair.record(1, cadence[1].hold(frame, true), 200 + n);
            pair.run(&[before, after]);
            if frame > 1 {
                assert_eq!(pair.landed(0), (Some(100 + n - 1), true), "frame {frame}: the hold before DLSS has its own answer");
                assert_eq!(pair.landed(1), (Some(200 + n - 1), true), "frame {frame}: the hold after DLSS has its own answer");
            }
        }
        pair.wait_finished(0, 8);
        pair.wait_finished(1, 8);
        for k in 0..2 {
            assert_eq!(*ran[k].lock().unwrap(), (1..=8).collect::<Vec<u32>>());
            assert_eq!(workers[k].counters.answered.load(Ordering::Relaxed), 8);
            assert!(workers[k].serving());
        }
        assert!(gave_up.lock().unwrap().is_empty());
        assert!(serving_at(game, Where::Before) && serving_at(game, Where::After));
        pair.close(workers);
        assert_eq!(OTHERS.seated(game), 0);
        assert_eq!(*log.lock().unwrap(), vec!["made", "dropped"], "one device for both holds, given back once both left");
    }

    /// The submits of both holds wait for the one second device without waiting for each other's frames: with no
    /// pause between frames every hold still gets its answer or goes without, and nothing is abandoned or torn.
    #[test]
    fn both_holds_keep_their_answers_apart_at_full_speed() {
        let Some((pair, workers, ran, gave_up, _)) = pair(None, None) else { return };
        let mut cadence = [Cadence::default(), Cadence::default()];
        for frame in 1..=40u64 {
            let n = frame as u32;
            let before = pair.record(0, cadence[0].hold(frame, true), 1000 + n);
            let after = pair.record(1, cadence[1].hold(frame, true), 2000 + n);
            // Every third frame as two submits (the split hold before DLSS, then DLSS's buffer with the hold after it).
            if frame % 3 == 0 {
                pair.run(&[before]);
                pair.run(&[after]);
            } else {
                pair.run(&[before, after]);
            }
            if frame > 1 {
                for (k, base) in [(0, 1000), (1, 2000)] {
                    let (value, stored) = pair.landed(k);
                    // The answer of the frame before (the submit waited for it), never another frame's or the other hold's.
                    assert!(!stored || value == Some(base + n - 1), "frame {frame}, hold {k}: {value:?}");
                }
            }
        }
        pair.wait_finished(0, 40);
        pair.wait_finished(1, 40);
        assert!(gave_up.lock().unwrap().is_empty());
        for (k, list) in ran.iter().enumerate() {
            let list = list.lock().unwrap();
            assert!(list.windows(2).all(|w| w[0] < w[1]), "hold {k}'s frames ran in order: {list:?}");
        }
        pair.close(workers);
    }

    /// One hold's frame fails on the second device: its worker gives the offload up, the network is closed for
    /// both, and the other hold's next frame is refused without being submitted. Both holds go on without an edit.
    #[test]
    fn a_give_up_by_one_hold_ends_the_other_s_frames_too() {
        let Some((pair, workers, ran, gave_up, log)) = pair(Some((1, 2)), None) else { return };
        let game = pair.dev.device.handle();
        let mut cadence = [Cadence::default(), Cadence::default()];
        for frame in 1..=6u64 {
            let n = frame as u32;
            pair.run(&[pair.record(0, cadence[0].hold(frame, true), 300 + n), pair.record(1, cadence[1].hold(frame, true), 400 + n)]);
            pair.wait_finished(0, n);
            pair.wait_finished(1, n);
        }
        assert!(!workers[0].serving() && !workers[1].serving());
        assert!(!pair.landed(0).1 && !pair.landed(1).1, "no edit once the offload is given up");
        assert_eq!(*ran[1].lock().unwrap(), vec![1]);
        let said = gave_up.lock().unwrap().clone();
        assert_eq!(said.len(), 2, "each worker tells its owner once: {said:?}");
        assert!(said.contains(&(1, "the test's frame failed".to_string(), false)));
        assert!(said.iter().any(|(k, why, stuck)| *k == 0 && why == "the second GPU was given up" && !stuck), "{said:?}");
        // Nothing went to the device for the hold before DLSS after the network was closed.
        let before_ran = ran[0].lock().unwrap().len() as u32;
        assert_eq!(OTHERS.all.lock().unwrap().iter().find(|(g, _, _)| *g == game).map(|(_, o, _)| o.lock().unwrap().submits), Some(before_ran + 1));
        assert!(before_ran < 6);
        pair.close(workers);
        assert_eq!(*log.lock().unwrap(), vec!["made", "dropped"]);
    }

    /// One hold's frame never finishes on the second device: the other hold's worker finds it stuck too (a stall,
    /// not a fall-back), submits nothing more there, and no new runner is given a seat on that device.
    #[test]
    fn a_frame_that_never_finishes_stops_both_holds_and_the_device_is_not_joined_again() {
        let Some((pair, workers, ran, gave_up, log)) = pair(None, Some((0, 3))) else { return };
        let game = pair.dev.device.handle();
        let mut cadence = [Cadence::default(), Cadence::default()];
        for frame in 1..=6u64 {
            let n = frame as u32;
            pair.run(&[pair.record(0, cadence[0].hold(frame, true), 500 + n), pair.record(1, cadence[1].hold(frame, true), 600 + n)]);
            pair.wait_finished(0, n);
            pair.wait_finished(1, n);
        }
        let said = gave_up.lock().unwrap().clone();
        assert_eq!(said.len(), 2, "{said:?}");
        assert!(said.iter().all(|(_, _, stuck)| *stuck), "both are a stall: nothing is taken back: {said:?}");
        assert_eq!(*ran[0].lock().unwrap(), vec![1, 2]);
        assert!(ran[1].lock().unwrap().len() <= 3, "nothing submitted for the other hold after the stall");
        let submits = OTHERS.all.lock().unwrap().iter().find(|(g, _, _)| *g == game).map(|(_, o, _)| o.lock().unwrap().submits);
        assert_eq!(submits, Some(2 + ran[1].lock().unwrap().len() as u32));
        // Another set of hold resources asks for the device: refused, and nothing is made beside the stuck one.
        let link = pair.holds[0].exchange.link(&pair.dev.device);
        assert!(SeatedRunner::new(game, &link, &log, None, None, Ranlist::default()).is_none());
        assert_eq!((OTHERS.seated(game), log.lock().unwrap().len()), (2, 1));
        pair.close(workers);
    }

    /// The frame no longer has passes at one hold: once that hold's side has been idle against the holds that keep
    /// coming at the other, its worker is ended and its runner leaves the second device, which stays with the hold that
    /// still serves. A pause of both holds releases neither.
    #[test]
    fn the_hold_that_stopped_coming_leaves_the_device_to_the_other() {
        let _state = crate::preupscale::offload::TEST_STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some((pair, workers, _, _, log)) = pair(None, None) else { return };
        let game = pair.dev.device.handle();
        clear_serving();
        set_serving(Where::Before, true);
        set_serving(Where::After, true);
        let idle = IDLE_SIDE.as_millis() as u64;
        let t0 = now_ms();
        // Both hold, then both pause for longer than the idle time, then both hold again.
        for t in [t0, t0 + 16, t0 + 4 * idle, t0 + 4 * idle + 16] {
            workers[0].touch_at(t);
            workers[1].touch_at(t + 1);
        }
        assert!(!workers[0].ended() && !workers[1].ended(), "a pause of both holds releases neither");
        // Only the hold before DLSS keeps coming.
        let t1 = t0 + 4 * idle + 32;
        workers[0].touch_at(t1);
        workers[0].touch_at(t1 + idle - 100);
        assert!(!workers[1].ended(), "not idle for long enough yet");
        workers[0].touch_at(t1 + idle + 16);
        assert!(!workers[1].ended(), "measured against this side's hold before: that one was too early");
        workers[0].touch_at(t1 + idle + 32);
        assert!(workers[1].ended() && !workers[0].ended());
        assert_eq!(serving(), Serving { before: true, after: false });
        assert!(serving_at(game, Where::Before) && !serving_at(game, Where::After));
        assert_eq!(OTHERS.seated(game), 1);
        assert_eq!(*log.lock().unwrap(), vec!["made"], "the device stays with the hold that serves");
        // That hold goes on getting its answers.
        let mut cadence = Cadence::default();
        pair.run(&[pair.record(0, cadence.hold(1, true), 901)]);
        pair.wait_finished(0, 1);
        pair.run(&[pair.record(0, cadence.hold(2, true), 902)]);
        assert_eq!(pair.landed(0), (Some(901), true));
        clear_serving();
        pair.close(workers);
        assert_eq!(*log.lock().unwrap(), vec!["made", "dropped"]);
    }

    /// The motion packed to halves is half the bytes: at 2560x1440 after DLSS the way out drops from 56 MB to 42 MB.
    #[test]
    fn the_packed_motion_is_half_the_floats() {
        let (floats, packed) = (Sizes::of(Where::After, 2560, 1440, false), Sizes::of(Where::After, 2560, 1440, true));
        assert_eq!((floats.motion, floats.packed_texels), (2560 * 1440 * 8, 0));
        assert_eq!((packed.motion, packed.packed_texels), (2560 * 1440 * 4, 2560 * 1440));
        assert_eq!(Sizes { motion: 0, packed_texels: 0, ..packed }, Sizes { motion: 0, packed_texels: 0, ..floats }, "nothing else changes");
        let (a, b) = (Layout::of(floats), Layout::of(packed));
        assert_eq!(a.out_bytes - b.out_bytes, 2560 * 1440 * 4);
        assert!(b.out_bytes < 43 << 20 && a.in_bytes == b.in_bytes, "{b:?}");
        let before = Sizes::of(Where::Before, 1485, 835, true);
        assert_eq!((before.motion, before.depth), (1485 * 835 * 4, 1485 * 835 * 4));
    }

    fn half_to_f32(h: u16) -> f32 {
        let (sign, exponent, fraction) = (if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }, i32::from((h >> 10) & 31), f32::from(h & 0x3ff));
        match exponent {
            0 => sign * fraction * 2f32.powi(-24),
            31 => {
                if fraction == 0.0 {
                    sign * f32::INFINITY
                } else {
                    f32::NAN
                }
            }
            e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
        }
    }

    /// The pack on the GPU, over every finite half: a field that is DLSS's vectors (halves) times the vectors' scale
    /// and divided by it again, as the motion pass leaves it with a fresh chain, packs back to exactly those halves,
    /// whatever the scale. A texel with a NaN in either component is the reserved pattern in both; nothing else ever
    /// is a NaN or an infinity (values past the half's range are its largest finite value).
    #[test]
    fn the_pack_gives_dlss_s_halves_back_exactly_and_keeps_the_nan_marker() {
        let Some(dev) = Dev::open() else {
            assert_ne!(neural_forge_protocol::env::var("NEURAL_FORGE_REQUIRE_VULKAN").as_deref(), Some("1"), "no Vulkan device for the pack's test");
            return;
        };
        const SCALES: [f32; 4] = [1.0, 1485.0, 835.0, 0.000_673_4];
        // Every finite half as x (y another half), once per scale, then the special texels: more than three rows
        // of the dispatch.
        let finite: Vec<u16> = (0..=u16::MAX).filter(|h| (h >> 10) & 31 != 31).collect();
        let mut field: Vec<[f32; 2]> = Vec::new();
        let mut expected: Vec<u32> = Vec::new();
        for scale in SCALES {
            for (k, &h) in finite.iter().enumerate() {
                let other = finite[(k * 7919 + 13) % finite.len()];
                let through = |h: u16| (half_to_f32(h) * scale) / scale;
                field.push([through(h), through(other)]);
                expected.push(u32::from(h) | (u32::from(other) << 16));
            }
        }
        let specials: [([f32; 2], u32); 8] = [
            ([f32::NAN, f32::NAN], HALF_NAN),
            ([f32::NAN, 1.5], HALF_NAN),
            ([-2.25, f32::NAN], HALF_NAN),
            ([f32::INFINITY, f32::NEG_INFINITY], 0x7bff | (0xfbff << 16)),
            ([1.0e9, -1.0e9], 0x7bff | (0xfbff << 16)),
            ([65504.0, -65504.0], 0x7bff | (0xfbff << 16)),
            ([0.0, -0.0], 0x8000 << 16),
            ([0.5, -3.25], 0x3800 | (0xc280 << 16)),
        ];
        for (v, bits) in specials {
            field.push(v);
            expected.push(bits);
        }
        let texels = field.len() as u64;
        assert!(texels > 3 * PACK_ROW, "more than one row of the dispatch");
        let sizes = Sizes { frame: 256, motion: texels * 4, packed_texels: texels, depth: 0, side: 16, answer: 256 };
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
        let own = |bytes: u64| crate::preupscale::own_host_buffer_with(&dev.device, &dev.instance, dev.physical, bytes, usage).unwrap();
        // SAFETY: a live device; the test's own objects, destroyed below.
        unsafe {
            let mut exchange = Exchange::build(&dev.device, &dev.instance, dev.physical, dev.import, sizes).expect("the exchange with a pack");
            assert!(exchange.motion_half());
            let (source, frame, side) = (own(texels * 8), own(sizes.frame), own(sizes.side));
            std::ptr::copy_nonoverlapping(field.as_ptr().cast::<u8>(), source.ptr, field.len() * 8);
            exchange.bind_pack(&dev.device, source.buffer);
            let d = &dev.device;
            let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(dev.family), None).unwrap();
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            let sides = [(side.buffer, sizes.side)];
            exchange.record_send(d, cmd, 9, &Sources { frame: FrameSource::Buffer(frame.buffer), motion: Some(source.buffer), depth: None, side: &sides });
            d.end_command_buffer(cmd).unwrap();
            d.queue_submit(dev.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            d.wait_for_fences(&[fence], true, 60_000_000_000).expect("the send finished");
            let sent = std::slice::from_raw_parts(exchange.out.ptr.add(exchange.layout.motion_at as usize).cast::<u32>(), texels as usize);
            let wrong: Vec<usize> = (0..sent.len()).filter(|&k| sent[k] != expected[k]).take(5).collect();
            assert!(
                wrong.is_empty(),
                "{} texels differ; the first: {:?}",
                (0..sent.len()).filter(|&k| sent[k] != expected[k]).count(),
                wrong.iter().map(|&k| (k, field[k], format!("{:08x} for {:08x}", sent[k], expected[k]))).collect::<Vec<_>>()
            );
            // The marker is the only pattern that is not a number, and no texel is an infinity.
            let not_finite = |h: u32| (h >> 10) & 31 == 31;
            assert_eq!(sent.iter().filter(|&&w| not_finite(w & 0xffff) || not_finite(w >> 16)).count(), 3);
            assert!(sent.iter().all(|&w| !(not_finite(w & 0xffff) || not_finite(w >> 16)) || w == HALF_NAN));
            assert_eq!(exchange.out.ptr.cast::<u32>().add(OUT_SEQ).read_volatile(), 9, "the frame's header went with it");
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
            for b in [&source, &frame, &side] {
                b.destroy(d);
            }
            exchange.destroy(d);
        }
        dev.close();
    }
}
