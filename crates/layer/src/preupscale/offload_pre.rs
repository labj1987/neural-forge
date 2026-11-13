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
                gated: None,
                pack,
            })
