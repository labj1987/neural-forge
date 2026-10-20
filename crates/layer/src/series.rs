//! Captures (`neural_forge_protocol::capture`): the one-shot pair the GUI's capture button and
//! `shmctl capture` ask for, and frame series (`shmctl capture --frames N`,
//! `NEURAL_FORGE_CAPTURE_AT`) of the next N presented frames as matched original/composited PNG
//! pairs, for measuring how well the model's edit lands frame by frame (`scripts/agreement.py`,
//! `crates/layer/examples/pan.rs`).
//!
//! Both read the swapchain image back just before `capture::run` (the game's frame) and again just
//! after it (exactly what is presented, whichever path composed it), and leave the present path itself
//! alone: a capture shows the live pipeline's own result (its working scale, model-pixel cap, even
//! model raster, mode-2 composition and white point), and the frame on screen is the one presented
//! anyway. A one-shot is a series of one, written as `<ms>-original.png`/`<ms>-composited.png` in the
//! captures folder; a series goes to its own `series-<ms>/` folder with an `index.tsv`.
//!
//! What became of a request is published on the channel as it goes ([`Report`]): running, then
//! written (with the files' paths and how many pairs the effect composed), failed or cancelled, each
//! with a reason. The PNGs are encoded and written on writer threads, never on the present thread; the
//! last writer to finish publishes the final outcome.
//!
//! Cost while a capture runs: two blocking image readbacks per present. The writers are bounded
//! ([`BACKLOG_BYTES`]): a long series whose PNGs cannot keep up paces the presents down to the
//! writers' speed rather than growing memory without limit.

use ash::vk;
use neural_forge_protocol::capture::{reason, state, Outcome, Request};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// What the writers hold at most before the present thread waits on them: the pairs queued for
/// them, the pair each is writing and the copy its encoder makes of one image ([`writer_budget`]).
const BACKLOG_BYTES: usize = 512 << 20;
const WRITERS: usize = 4;

/// How many writers a capture of `len`-byte images starts and how many pairs may queue for them,
/// within [`BACKLOG_BYTES`]: a writer holds its pair (`2 * len`) and the encoder's opaque copy of
/// one image (`len`), a queued pair `2 * len`. At least one writer, and a queue of 0 (each pair
/// handed straight to a writer) when the writers alone fill the budget.
fn writer_budget(len: usize, frames: u32) -> (usize, usize) {
    let len = len.max(1);
    let writers = WRITERS.min(frames.max(1) as usize).min(BACKLOG_BYTES / (3 * len)).max(1);
    (writers, BACKLOG_BYTES.saturating_sub(writers * 3 * len) / (2 * len))
}

/// What the real present waits on after [`Series::after`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentWait {
    /// What it would have: nothing of the present's was consumed.
    Unchanged,
    /// The readback waited on the composition's semaphore and was seen complete: the present must
    /// not wait on that semaphore again (a binary semaphore is consumed by one wait).
    Consumed,
    /// A readback was not seen complete (its bounded wait failed): the present waits on this, signalled
    /// once everything the layer submitted for it has run, in place of the composition's semaphore.
    Instead(vk::Semaphore),
}

struct Job {
    seq: u32,
    original: Vec<u8>,
    composited: Vec<u8>,
    width: u32,
    height: u32,
    bgr_order: bool,
}

/// Host-visible copy target for one readback.
struct Readback {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *const u8,
    coherent: bool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

struct Resources {
    pool: vk::CommandPool,
    queue_family: u32,
    bytes: u64,
    original: Readback,
    composited: Readback,
}

/// Where a capture's files go.
#[derive(Clone)]
enum Target {
    /// The one-shot pair: `<dir>/<stem>-original.png` and `<dir>/<stem>-composited.png`.
    Pair { dir: PathBuf, stem: String },
    /// A series: `<dir>/<seq>-original.png` ... and `<dir>/index.tsv`.
    Series { dir: PathBuf },
}

impl Target {
    fn names(&self, seq: u32) -> (PathBuf, PathBuf) {
        match self {
            Target::Pair { dir, stem } => (dir.join(format!("{stem}-original.png")), dir.join(format!("{stem}-composited.png"))),
            Target::Series { dir } => {
                let (o, c) = pair_names(seq);
                (dir.join(o), dir.join(c))
            }
        }
    }

    /// The outcome's paths: a one-shot's two files, a series' folder.
    fn paths(&self) -> String {
        match self {
            Target::Pair { .. } => {
                let (o, c) = self.names(0);
                format!("{}\n{}", o.display(), c.display())
            }
            Target::Series { dir } => dir.display().to_string(),
        }
    }
}

/// How a capture ended on the present thread (its files may still be being written).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Close {
    Complete,
    Cancelled,
    Failed(u32),
}

/// The published outcome of the running capture, shared with its writers: whoever changes it
/// publishes it under the lock, so the header's result block has one writer at a time.
pub(crate) struct Report {
    channel: Option<crate::shm::Channel>,
    progress: Mutex<Progress>,
}

struct Progress {
    outcome: Outcome,
    /// Pairs handed to the writers, and how many of them were written and failed.
    queued: u32,
    written: u32,
    failed: u32,
    /// Set when the present thread is done with the capture.
    closed: Option<Close>,
    /// Frames composed by the effect, and the first reason a frame was not.
    composed: u32,
    untouched: u32,
    held: bool,
}

/// The request id of the last capture this process started (`u64::MAX`: none): a channel this process
/// takes back must not have its own running capture reported as ended ([`own_capture`]).
static OWN_CAPTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Whether the capture with request `id` is one this process started.
pub(crate) fn own_capture(id: u32) -> bool {
    OWN_CAPTURE.load(std::sync::atomic::Ordering::Acquire) == u64::from(id)
}

#[cfg(test)]
pub(crate) fn set_own_capture(id: u32) {
    OWN_CAPTURE.store(u64::from(id), std::sync::atomic::Ordering::Release);
}

impl Report {
    fn new(channel: Option<crate::shm::Channel>, request: Request, target: &Target, held: bool) -> Arc<Self> {
        OWN_CAPTURE.store(u64::from(request.id), std::sync::atomic::Ordering::Release);
        let outcome = Outcome { id: request.id, state: state::RUNNING, total: request.frames, paths: target.paths(), ..Outcome::default() };
        let report = Arc::new(Self {
            channel,
            progress: Mutex::new(Progress { outcome, queued: 0, written: 0, failed: 0, closed: None, composed: 0, untouched: reason::NONE, held }),
        });
        report.update(|_| {});
        report
    }

    /// Publishes a request that was refused before it started.
    fn refuse(channel: Option<crate::shm::Channel>, request: Request, why: u32) {
        let outcome = Outcome { id: request.id, state: state::FAILED, reason: why, total: request.frames, ..Outcome::default() };
        publish(channel.as_ref(), &outcome);
    }

    /// Applies `change`, settles the outcome, publishes it. Never panics: a poisoned lock is taken over.
    fn update(&self, change: impl FnOnce(&mut Progress)) {
        let mut p = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut p);
        p.settle();
        publish(self.channel.as_ref(), &p.outcome);
    }

    fn outcome(&self) -> Outcome {
        self.progress.lock().unwrap_or_else(|e| e.into_inner()).outcome.clone()
    }
}

/// Publishes `outcome` on the channel. The result block is a sequence-locked unit with one writer at
/// a time; this lock is that one writer, whichever capture or thread publishes.
pub(crate) fn publish(channel: Option<&crate::shm::Channel>, outcome: &Outcome) {
    static PUBLISHING: Mutex<()> = Mutex::new(());
    if let Some(hdr) = channel.and_then(crate::shm::Channel::header) {
        let _one = PUBLISHING.lock().unwrap_or_else(|e| e.into_inner());
        neural_forge_protocol::capture::publish(hdr, outcome);
    }
}

impl Progress {
    /// The outcome from the counts: still running until the present thread closed the capture and
    /// every queued pair was written (or failed).
    fn settle(&mut self) {
        let o = &mut self.outcome;
        o.done = self.written;
        o.composed = self.composed;
        let Some(close) = self.closed.filter(|_| self.written + self.failed == self.queued) else {
            o.state = state::RUNNING;
            return;
        };
        (o.state, o.reason) = match close {
            Close::Cancelled => (state::CANCELLED, reason::NONE),
            Close::Failed(why) => (state::FAILED, why),
            Close::Complete if self.failed > 0 || self.written == 0 => (state::FAILED, reason::WRITE_FAILED),
            Close::Complete if self.held => (state::WRITTEN, reason::BEFORE_UPSCALER),
            Close::Complete if self.composed == self.written => (state::WRITTEN, reason::NONE),
            Close::Complete if self.composed == 0 => {
                (state::WRITTEN, if self.untouched == reason::NONE { reason::NOT_COMPOSED } else { self.untouched })
            }
            Close::Complete => (state::WRITTEN, reason::PARTLY_COMPOSED),
        };
    }
}

/// A running capture: where it writes, how many frames are left, and its writers.
struct Active {
    request: Request,
    target: Target,
    remaining: u32,
    seq: u32,
    started: std::time::Instant,
    index: Option<std::io::BufWriter<std::fs::File>>,
    sender: Option<mpsc::SyncSender<Job>>,
    report: Arc<Report>,
    /// Set by [`Series::before`] when the original readback is on the queue.
    original_pending: bool,
}

#[derive(Default)]
pub struct Series {
    active: Option<Active>,
    resources: Option<Resources>,
    /// Readback resources whose fence was not seen signalled, and the semaphores presents waited on
    /// in their place ([`PresentWait::Instead`]): never reused, freed only at device teardown once
    /// the device is idle.
    stranded: Vec<Resources>,
    stranded_semaphores: Vec<vk::Semaphore>,
    /// Writers of finished captures still writing: waited for (bounded) only at device teardown.
    draining: Vec<std::thread::JoinHandle<()>>,
}

// SAFETY: the raw mapped pointers are only dereferenced on the thread holding the device
// state's `Mutex`, like every other Vulkan resource in that state.
unsafe impl Send for Series {}

/// Whether this present takes `request`. While frames are `held` before the upscaler
/// (`crate::preupscale`) and with `NEURAL_FORGE_PREUPSCALE=dump`, a one-shot belongs to the
/// pre-upscaler dump (`dump_owns_one_shot`), which writes DLSS's input instead; every other request
/// is the present's.
pub fn serves(request: &Request, dump_owns_one_shot: bool) -> bool {
    request.frames != 1 || !dump_owns_one_shot
}

/// The capture request this present takes ([`serves`]).
pub fn take_request(shm: &crate::shm::ShmClient, dump_owns_one_shot: bool) -> Option<Request> {
    shm.take_capture(|r| serves(r, dump_owns_one_shot))
}

/// Publishes `request` as refused with `why` (and logs it).
pub fn refuse(shm: &crate::shm::ShmClient, request: Request, why: u32) {
    crate::log!("[capture] request {} for {} frames refused: {}", request.id, request.frames, reason::describe(why).unwrap_or("?"));
    crate::logging::flush();
    Report::refuse(shm.channel(), request, why);
}

/// Takes the waiting request and publishes it as failed with `why` (the present cannot capture this
/// game's frames at all). Nothing to do without one.
pub fn refuse_waiting(shm: &crate::shm::ShmClient, dump_owns_one_shot: bool, why: u32) {
    if let Some(request) = take_request(shm, dump_owns_one_shot) {
        refuse(shm, request, why);
    }
}

impl Series {
    /// Starts `request`'s capture under `base` (the captures folder): a one-shot pair, or a series
    /// in its own `series-<ms>/` folder. `held`: frames are held before the upscaler, so both halves
    /// of each pair are the presented frame (the outcome says so).
    pub fn start(&mut self, request: Request, base: &Path, channel: Option<crate::shm::Channel>, held: bool) {
        self.draining.retain(|w| !w.is_finished());
        if !request.valid() {
            crate::log!("[capture] request {} for {} frames refused: 1 to {} frames", request.id, request.frames, neural_forge_protocol::capture::MAX_FRAMES);
            Report::refuse(channel, request, reason::BAD_COUNT);
            return;
        }
        if let Some(active) = &self.active {
            crate::log!("[capture] capture {} is still running; request {} for {} frames refused", active.request.id, request.id, request.frames);
            Report::refuse(channel, request, reason::BUSY);
            // The running capture's own outcome again, so a reader of the channel sees it, not the refusal.
            active.report.update(|_| {});
            return;
        }
        let target = match create_target(base, request.frames) {
            Ok(target) => target,
            Err(e) => {
                crate::log!("[capture] could not create the capture's files under {}: {e}", base.display());
                crate::logging::flush();
                Report::refuse(channel, request, reason::WRITE_FAILED);
                return;
            }
        };
        let index = match &target {
            Target::Series { dir } => std::fs::File::create(dir.join("index.tsv")).ok().map(std::io::BufWriter::new),
            Target::Pair { .. } => None,
        };
        match &target {
            Target::Pair { dir, stem } => crate::log!("[capture] capturing the next present as {stem}-{{original,composited}}.png in {}", dir.display()),
            Target::Series { dir } => crate::log!("[capture] capturing the next {} presents into {}", request.frames, dir.display()),
        }
        crate::logging::flush();
        let report = Report::new(channel, request, &target, held);
        self.active = Some(Active {
            request,
            target,
            remaining: request.frames,
            seq: 0,
            started: std::time::Instant::now(),
            index,
            sender: None,
            report,
            original_pending: false,
        });
    }

    pub fn running(&self) -> bool {
        self.active.is_some()
    }

    /// The running capture's published outcome (tests, and the log).
    #[cfg(test)]
    fn outcome(&self) -> Option<Outcome> {
        self.active.as_ref().map(|a| a.report.outcome())
    }

    /// Ends the running capture because this present cannot read the frame (`why`), keeping the pairs
    /// already taken.
    pub fn abandon(&mut self, why: u32) {
        if self.active.is_some() {
            crate::log!("[capture] capture abandoned: {}", reason::describe(why).unwrap_or("?"));
            self.finish(Close::Failed(why));
        }
    }

    /// Queues the readback of `image` as this present's original. Call before anything of the
    /// layer's own is submitted for this present; `image` is in `PRESENT_SRC_KHR` and every
    /// earlier submission on `queue` (the application's, via the relay) is ordered before it.
    /// A capture the requester cancelled ends here instead.
    ///
    /// # Safety
    /// `queue` is externally synchronized for this call and the following [`Self::after`], and
    /// belongs to `queue_family`; `image` is a live `width`x`height` 4-byte-per-pixel image
    /// created with `TRANSFER_SRC` usage.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn before(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family: u32,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let Some(active) = self.active.as_mut() else { return };
        active.original_pending = false;
        let cancelled = active.report.channel.as_ref().and_then(crate::shm::Channel::header).is_some_and(|hdr| neural_forge_protocol::capture::cancel_requested(hdr, active.request.id));
        if cancelled {
            crate::log!("[capture] capture {} cancelled after {} frames", active.request.id, active.seq);
            self.finish(Close::Cancelled);
            return;
        }
        let bytes = u64::from(width) * u64::from(height) * 4;
        if self.resources.as_ref().is_some_and(|r| r.bytes != bytes || r.queue_family != queue_family) {
            // SAFETY: every submission on these resources was waited on in `after`.
            unsafe { destroy_resources(self.resources.take(), device) };
        }
        if self.resources.is_none() {
            self.resources = unsafe { create_resources(device, instance, physical_device, queue_family, bytes) };
        }
        let Some(r) = self.resources.as_ref() else {
            crate::log!("[capture] could not create readback resources; capture abandoned");
            self.finish(Close::Failed(reason::READBACK_FAILED));
            return;
        };
        // SAFETY: per this function's contract.
        active.original_pending = unsafe { submit_copy(device, queue, &r.original, image, width, height, None) };
    }

    /// Reads `image` back as this present's composited result, waiting on `compose_done` (the
    /// semaphore `capture::run` returned, if any) first, and hands the pair to the writers.
    /// `untouched`: when `compose_done` is `None`, why the effect did not compose this frame (a
    /// [`reason`] code; [`reason::NONE`] when unknown). Returns what the real present waits on
    /// ([`PresentWait`]).
    ///
    /// # Safety
    /// Same as [`Self::before`], called once after it for the same present.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn after(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        image: vk::Image,
        width: u32,
        height: u32,
        bgr_order: bool,
        compose_done: Option<vk::Semaphore>,
        untouched: u32,
    ) -> PresentWait {
        let Some(active) = self.active.as_mut() else { return PresentWait::Unchanged };
        let Some(r) = self.resources.as_ref() else { return PresentWait::Unchanged };
        if !active.original_pending {
            return PresentWait::Unchanged;
        }
        active.original_pending = false;
        // SAFETY: per this function's contract.
        let submitted = unsafe { submit_copy(device, queue, &r.composited, image, width, height, compose_done) };
        let consumed = if submitted { PresentWait::Consumed } else { PresentWait::Unchanged };
        let original_done = unsafe { wait_and_reset(device, &r.original) };
        let composited_done = submitted && unsafe { wait_and_reset(device, &r.composited) };
        if !(original_done && composited_done) {
            crate::log!("[capture] readback failed on frame {}; capture abandoned", active.seq);
            self.finish(Close::Failed(reason::READBACK_FAILED));
            // A fence that never signalled leaves its resources in an unknown state: they are kept
            // aside rather than freed under the GPU or reused with a pending fence.
            self.stranded.extend(self.resources.take());
            // A copy still in flight reads the image and puts it back in `PRESENT_SRC_KHR` only when it
            // runs: the present must not go before it. With `compose_done` still unconsumed (the second
            // copy was never submitted) the present's wait on it already orders it after the first.
            let in_flight = !original_done || (submitted && !composited_done);
            if !in_flight || (!submitted && compose_done.is_some()) {
                return consumed;
            }
            // SAFETY: per this function's contract.
            return match unsafe { signal_after_submitted(device, queue) } {
                Some(semaphore) => {
                    self.stranded_semaphores.push(semaphore);
                    PresentWait::Instead(semaphore)
                }
                None => {
                    crate::log!("[capture] could not order the present after the stalled readback");
                    consumed
                }
            };
        }
        let len = (u64::from(width) * u64::from(height) * 4) as usize;
        // SAFETY: both fences are signalled, the buffers are `len` bytes and stay mapped.
        let (original, composited) = unsafe {
            invalidate(device, &r.original);
            invalidate(device, &r.composited);
            (std::slice::from_raw_parts(r.original.ptr, len).to_vec(), std::slice::from_raw_parts(r.composited.ptr, len).to_vec())
        };
        let seq = active.seq;
        let composed = compose_done.is_some();
        if let Some(index) = active.index.as_mut() {
            if seq == 0 {
                // `gpu_composed`: the GPU composition ran for this present (it returned a
                // semaphore). 0 is a frame that went out untouched.
                let _ = writeln!(index, "seq\tms\tgpu_composed\twidth\theight");
            }
            let _ = writeln!(index, "{seq}\t{:.3}\t{}\t{width}\t{height}", active.started.elapsed().as_secs_f64() * 1000.0, u8::from(composed));
        }
        if active.sender.is_none() {
            match spawn_writers(&active.target, &active.report, len, active.request.frames) {
                Some((sender, writers)) => {
                    active.sender = Some(sender);
                    self.draining.extend(writers);
                }
                None => {
                    crate::log!("[capture] could not start a writer thread; capture abandoned");
                    self.finish(Close::Failed(reason::WRITE_FAILED));
                    return consumed;
                }
            }
        }
        let Some(active) = self.active.as_mut() else { return consumed };
        active.report.update(|p| {
            p.queued += 1;
            if composed {
                p.composed += 1;
            } else if p.untouched == reason::NONE {
                p.untouched = untouched;
            }
        });
        let sent = active.sender.as_ref().is_some_and(|s| s.send(Job { seq, original, composited, width, height, bgr_order }).is_ok());
        if !sent {
            // Every writer is gone (it panicked): the pair counts as failed.
            active.report.update(|p| p.failed += 1);
        }
        active.seq += 1;
        active.remaining = active.remaining.saturating_sub(1);
        if active.remaining == 0 {
            self.finish(Close::Complete);
        }
        consumed
    }

    /// Ends the running capture on the present thread. Never waits for its files: the writers drain
    /// what is queued and the last of them publishes the final outcome.
    fn finish(&mut self, close: Close) {
        let Some(mut active) = self.active.take() else { return };
        if let Some(mut index) = active.index.take() {
            let _ = index.flush();
        }
        drop(active.sender.take());
        let frames = active.seq;
        active.report.update(|p| p.closed = Some(close));
        crate::log!("[capture] {} frames captured ({close:?}); their files are written in the background", frames);
        crate::logging::flush();
    }

    /// Whether a stalled readback's resources wait for the device's idle wait at teardown.
    pub fn has_stranded(&self) -> bool {
        !self.stranded.is_empty() || !self.stranded_semaphores.is_empty()
    }

    /// `idle`: the device was waited idle (or is lost), so the stranded readbacks and the semaphores
    /// presents waited on in their place may go too; otherwise they are leaked.
    ///
    /// # Safety
    /// Nothing submitted through this series may still be pending apart from the stranded
    /// readbacks (nothing else is after [`Self::after`] returns), and with `idle` not those either,
    /// nor any present waiting on a stranded semaphore.
    pub unsafe fn destroy(&mut self, device: &ash::Device, idle: bool) {
        if self.active.is_some() {
            self.finish(Close::Failed(reason::GAME_ENDED));
        }
        // Device teardown holds the layer's state lock: a game quitting mid-capture must not
        // wait out a long PNG backlog. Past the limit the writers finish on their own threads (the
        // process may exit first, losing the pairs still queued).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !self.draining.iter().all(|w| w.is_finished()) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if self.draining.iter().any(|w| !w.is_finished()) {
            crate::log!("[capture] writers still busy at teardown; the last pairs may be missing");
            crate::logging::flush();
        }
        self.draining.clear();
        unsafe { destroy_resources(self.resources.take(), device) };
        if idle {
            for stranded in self.stranded.drain(..) {
                // SAFETY: the device is idle: nothing of these is pending.
                unsafe { destroy_resources(Some(stranded), device) };
            }
            for semaphore in self.stranded_semaphores.drain(..) {
                // SAFETY: as above, and the presents that waited on it are done with the swapchains.
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
        } else if self.has_stranded() {
            crate::log!("[capture] the device was not idle at teardown: the stalled readback's resources are leaked");
            self.stranded.clear();
            self.stranded_semaphores.clear();
        }
    }
}

/// The one-shot's free `<ms>` stem in `base`, or a new `series-<ms>` folder under it. A name that is
/// taken (two captures in one millisecond) gets a `-1`, `-2`, ... suffix instead of overwriting.
fn create_target(base: &Path, frames: u32) -> std::io::Result<Target> {
    std::fs::create_dir_all(base)?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    for n in 0..1000u32 {
        let name = if n == 0 { stamp.to_string() } else { format!("{stamp}-{n}") };
        if frames == 1 {
            let target = Target::Pair { dir: base.to_path_buf(), stem: name };
            let (o, c) = target.names(0);
            if !o.exists() && !c.exists() {
                return Ok(target);
            }
        } else {
            let dir = base.join(format!("series-{name}"));
            // `create_dir`, not `create_dir_all`: an existing folder is another series'.
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(Target::Series { dir }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
    Err(std::io::Error::other("no free capture name"))
}

/// Starts the capture's writers (at most one per frame): `None` when not one thread could be started.
fn spawn_writers(target: &Target, report: &Arc<Report>, len: usize, frames: u32) -> Option<(mpsc::SyncSender<Job>, Vec<std::thread::JoinHandle<()>>)> {
    let (writers_wanted, bound) = writer_budget(len, frames);
    let (sender, receiver) = mpsc::sync_channel::<Job>(bound);
    let receiver = Arc::new(Mutex::new(receiver));
    let fast = matches!(target, Target::Series { .. });
    let mut writers = Vec::new();
    for i in 0..writers_wanted {
        let receiver = receiver.clone();
        let target = target.clone();
        let report = report.clone();
        let spawned = std::thread::Builder::new().name(format!("nf-capture-{i}")).spawn(move || loop {
            let job = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
            let Ok(job) = job else { return };
            let ok = write_job(&target, &job, fast);
            report.update(|p| if ok { p.written += 1 } else { p.failed += 1 });
            let outcome = report.outcome();
            if outcome.finished() {
                crate::log!("[capture] {}: {} pairs written to {}", outcome.message(), outcome.done, outcome.paths.replace('\n', ", "));
                crate::logging::flush();
            }
        });
        match spawned {
            Ok(handle) => writers.push(handle),
            Err(e) => crate::log!("[capture] could not start writer {i}: {e}"),
        }
    }
    (!writers.is_empty()).then_some((sender, writers))
}

/// The file names one series frame is written under.
pub fn pair_names(seq: u32) -> (String, String) {
    (format!("{seq:06}-original.png"), format!("{seq:06}-composited.png"))
}

fn write_job(target: &Target, job: &Job, fast: bool) -> bool {
    let (original, composited) = target.names(job.seq);
    let a = crate::dump::write_png(&original, &job.original, job.width, job.height, job.bgr_order, fast);
    let b = crate::dump::write_png(&composited, &job.composited, job.width, job.height, job.bgr_order, fast);
    for (path, result) in [(&original, &a), (&composited, &b)] {
        if let Err(e) = result {
            crate::log!("[capture] writing {} failed: {e}", path.display());
        }
    }
    a.is_ok() && b.is_ok()
}

unsafe fn create_readback(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
    bytes: u64,
) -> Option<Readback> {
    let info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let find = |wanted: vk::MemoryPropertyFlags| {
        (0..mem_props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted))
    };
    // Cached host memory first: reading uncached memory back on the CPU is slow.
    let Some(type_index) = find(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| find(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)) else {
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };
    let coherent = mem_props.memory_types[type_index as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
    let memory = match unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) } {
        Ok(m) => m,
        Err(_) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return None;
        }
    };
    let cleanup = || unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
    };
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        cleanup();
        return None;
    }
    let Ok(ptr) = (unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }) else {
        cleanup();
        return None;
    };
    let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
    // The buffer is submitted through the loader's trampolines: it needs the device's loader data,
    // like every buffer the layer allocates.
    let Some(cmd) = unsafe { crate::loader_data::allocate_commands(device, &alloc) }.ok().and_then(|v| v.first().copied()) else {
        cleanup();
        return None;
    };
    let Ok(fence) = (unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }) else {
        cleanup();
        return None;
    };
    Some(Readback { buffer, memory, ptr: ptr.cast(), coherent, cmd, fence })
}

unsafe fn create_resources(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: u64,
) -> Option<Resources> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = unsafe { device.create_command_pool(&pool_info, None) }.ok()?;
    let original = unsafe { create_readback(device, &mem_props, pool, bytes) };
    let composited = unsafe { create_readback(device, &mem_props, pool, bytes) };
    match (original, composited) {
        (Some(original), Some(composited)) => Some(Resources { pool, queue_family, bytes, original, composited }),
        (a, b) => {
            for r in [a, b].into_iter().flatten() {
                unsafe { destroy_readback(device, r) };
            }
            unsafe { device.destroy_command_pool(pool, None) };
            None
        }
    }
}

unsafe fn destroy_readback(device: &ash::Device, r: Readback) {
    unsafe {
        device.destroy_fence(r.fence, None);
        device.destroy_buffer(r.buffer, None);
        device.free_memory(r.memory, None);
    }
}

unsafe fn destroy_resources(resources: Option<Resources>, device: &ash::Device) {
    let Some(r) = resources else { return };
    unsafe {
        destroy_readback(device, r.original);
        destroy_readback(device, r.composited);
        // Frees the command buffers with it.
        device.destroy_command_pool(r.pool, None);
    }
}

/// Records and submits "`image` (`PRESENT_SRC_KHR`) -> `r.buffer`, back to `PRESENT_SRC_KHR`",
/// optionally waiting on `wait` at the transfer stage, signalling `r.fence`.
unsafe fn submit_copy(device: &ash::Device, queue: vk::Queue, r: &Readback, image: vk::Image, width: u32, height: u32, wait: Option<vk::Semaphore>) -> bool {
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let to_src = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build();
    let to_present = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::empty())
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build();
    let to_host = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(r.buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        .build();
    let region = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width, height, depth: 1 })
        .build();
    let begin = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        if device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()).is_err() || device.begin_command_buffer(r.cmd, &begin).is_err() {
            return false;
        }
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
        device.cmd_copy_image_to_buffer(r.cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, r.buffer, &[region]);
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[], &[to_host], &[to_present]);
        if device.end_command_buffer(r.cmd).is_err() {
            return false;
        }
        let stage = [vk::PipelineStageFlags::TRANSFER];
        let waits: Vec<vk::Semaphore> = wait.into_iter().collect();
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stage[..waits.len()])
            .command_buffers(std::slice::from_ref(&r.cmd))
            .build();
        device.queue_submit(queue, &[submit], r.fence).is_ok()
    }
}

/// Waits (bounded, [`crate::FENCE_WAIT_TIMEOUT`]) for `r`'s submission, then resets its fence for the
/// next one. Through [`crate::note_fence_wait`], like every bounded wait of the layer: a timeout
/// leaves its trail, and a lost device latches the layer inert on it.
unsafe fn wait_and_reset(device: &ash::Device, r: &Readback) -> bool {
    #[cfg(test)]
    if TEST_READBACK_STALLS.with(std::cell::Cell::get) {
        return false;
    }
    unsafe {
        let wait = device.wait_for_fences(&[r.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64);
        crate::note_fence_wait(device.handle(), wait, "series readback").is_ok() && crate::note_vk(device.handle(), device.reset_fences(&[r.fence])).is_ok()
    }
}

#[cfg(test)]
thread_local! {
    /// Makes [`wait_and_reset`] report a stalled readback on this thread.
    static TEST_READBACK_STALLS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// An empty batch on `queue` signalling a new semaphore: its signal's first synchronization scope
/// is everything submitted to `queue` before it, so a wait on it orders the present after the
/// layer's stalled readback. `None` when it cannot be made.
///
/// # Safety
/// `queue` is externally synchronized for the call.
unsafe fn signal_after_submitted(device: &ash::Device, queue: vk::Queue) -> Option<vk::Semaphore> {
    // SAFETY: plain creation and a submit that references only the new semaphore.
    unsafe {
        let semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()?;
        let submit = vk::SubmitInfo::builder().signal_semaphores(std::slice::from_ref(&semaphore)).build();
        if crate::note_vk(device.handle(), device.queue_submit(queue, &[submit], vk::Fence::null())).is_err() {
            device.destroy_semaphore(semaphore, None);
            return None;
        }
        Some(semaphore)
    }
}

unsafe fn invalidate(device: &ash::Device, r: &Readback) {
    if !r.coherent {
        let range = vk::MappedMemoryRange::builder().memory(r.memory).offset(0).size(vk::WHOLE_SIZE).build();
        unsafe {
            let _ = device.invalidate_mapped_memory_ranges(&[range]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_forge_protocol::capture::{self as protocol, MAX_FRAMES};

    fn clear(device: &ash::Device, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image, from: vk::ImageLayout, colour: [f32; 4]) {
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let barrier = |old, new, src, dst| {
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
        };
        unsafe {
            let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier(from, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::MEMORY_READ, vk::AccessFlags::TRANSFER_WRITE)]);
            device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: colour }, &[range]);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[barrier(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::MEMORY_READ)]);
            device.end_command_buffer(cmd).unwrap();
            // No fence: queue order alone has to keep the series' readbacks around this.
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], vk::Fence::null()).unwrap();
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!("target/test-scratch/capture-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn decode(path: &Path) -> Vec<u8> {
        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap())).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().expect("PNG frame size fits in memory")];
        reader.next_frame(&mut buf).unwrap();
        buf
    }

    /// The published outcome once the writers are done with it (bounded).
    fn finished(hdr: &neural_forge_protocol::ShmHeader, id: u32) -> Outcome {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if let Some(o) = protocol::result(hdr).filter(|o| o.id == id && o.finished()) {
                return o;
            }
            assert!(std::time::Instant::now() < deadline, "capture {id} never finished: {:?}", protocol::result(hdr));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn request(id: u32, frames: u32) -> Request {
        Request { id, frames }
    }

    /// A one-shot belongs to the pre-upscaler dump only with `NEURAL_FORGE_PREUPSCALE=dump`; a series
    /// is always the present's, and a request is taken once.
    #[test]
    fn requests_are_served_by_one_owner_and_taken_once() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = crate::shm::ShmClient::test_over_header(&header);
        assert_eq!(take_request(&shm, false), None);
        let id = protocol::request(&header, 1).unwrap();
        assert_eq!(take_request(&shm, true), None, "with the dump on, the one-shot is the dump's");
        assert_eq!(shm.take_capture_request(), Some(request(id, 1)), "the dump takes it");
        let id = protocol::request(&header, 1).unwrap();
        assert_eq!(take_request(&shm, false), Some(request(id, 1)));
        assert_eq!(take_request(&shm, false), None, "served once");
        let id = protocol::request(&header, 30).unwrap();
        assert_eq!(shm.take_capture_request(), None, "a series is never the dump's");
        assert_eq!(take_request(&shm, true), Some(request(id, 30)));
    }

    /// Requests that cannot start are answered at once, with the reason, for their own id.
    #[test]
    fn bad_counts_unwritable_folders_and_a_busy_capture_are_refused_with_a_reason() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = crate::shm::ShmClient::test_over_header(&header);
        let base = scratch("refused");
        let mut series = Series::default();
        for frames in [0, MAX_FRAMES + 1] {
            series.start(request(5, frames), &base, shm.channel(), false);
            assert!(!series.running());
            let o = protocol::result(&header).unwrap();
            assert_eq!((o.id, o.state, o.reason), (5, state::FAILED, reason::BAD_COUNT), "{frames} frames");
        }
        // A file where the folder should be.
        std::fs::create_dir_all(base.parent().unwrap()).unwrap();
        std::fs::write(&base, b"not a folder").unwrap();
        series.start(request(6, 1), &base, shm.channel(), false);
        assert!(!series.running());
        assert_eq!(protocol::result(&header).map(|o| (o.id, o.state, o.reason)), Some((6, state::FAILED, reason::WRITE_FAILED)));
        std::fs::remove_file(&base).unwrap();
        // Busy: the second request is refused, and the running capture's own outcome is published again.
        series.start(request(7, 3), &base, shm.channel(), false);
        assert!(series.running());
        series.start(request(8, 1), &base, shm.channel(), false);
        let o = protocol::result(&header).unwrap();
        assert_eq!((o.id, o.state, o.total), (7, state::RUNNING, 3), "the running capture is what the channel shows");
        series.abandon(reason::GAME_ENDED);
        assert_eq!(finished(&header, 7).reason, reason::GAME_ENDED);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The outcome from the counts: what a written capture says about its composited images.
    #[test]
    fn the_outcome_says_whether_the_effect_composed_the_frames() {
        let settle = |queued, written, failed, composed, untouched, held, close| {
            let mut p = Progress { outcome: Outcome::default(), queued, written, failed, closed: close, composed, untouched, held };
            p.settle();
            (p.outcome.state, p.outcome.reason, p.outcome.done)
        };
        assert_eq!(settle(2, 1, 0, 2, 0, false, Some(Close::Complete)).0, state::RUNNING, "a pair still being written");
        assert_eq!(settle(2, 2, 0, 2, 0, false, None).0, state::RUNNING, "frames still to capture");
        assert_eq!(settle(2, 2, 0, 2, 0, false, Some(Close::Complete)), (state::WRITTEN, reason::NONE, 2));
        assert_eq!(settle(2, 2, 0, 0, reason::EFFECT_OFF, false, Some(Close::Complete)), (state::WRITTEN, reason::EFFECT_OFF, 2));
        assert_eq!(settle(1, 1, 0, 0, reason::NONE, false, Some(Close::Complete)), (state::WRITTEN, reason::NOT_COMPOSED, 1));
        assert_eq!(settle(3, 3, 0, 1, reason::NO_ANSWER, false, Some(Close::Complete)), (state::WRITTEN, reason::PARTLY_COMPOSED, 3));
        assert_eq!(settle(1, 1, 0, 0, reason::BEFORE_UPSCALER, true, Some(Close::Complete)), (state::WRITTEN, reason::BEFORE_UPSCALER, 1));
        assert_eq!(settle(2, 1, 1, 2, 0, false, Some(Close::Complete)), (state::FAILED, reason::WRITE_FAILED, 1));
        assert_eq!(settle(1, 1, 0, 1, 0, false, Some(Close::Cancelled)), (state::CANCELLED, reason::NONE, 1));
        assert_eq!(settle(0, 0, 0, 0, 0, false, Some(Close::Failed(reason::READBACK_FAILED))), (state::FAILED, reason::READBACK_FAILED, 0));
    }

    /// Two captures in the same millisecond never overwrite each other's files or share a folder.
    #[test]
    fn capture_names_never_collide() {
        let base = scratch("names");
        let a = create_target(&base, 1).unwrap();
        let (o, c) = a.names(0);
        std::fs::write(&o, b"x").unwrap();
        std::fs::write(&c, b"x").unwrap();
        let b = create_target(&base, 1).unwrap();
        assert_ne!(b.names(0).0, o, "a taken stem is skipped");
        let s1 = create_target(&base, 5).unwrap();
        let s2 = create_target(&base, 5).unwrap();
        assert_ne!(s1.paths(), s2.paths(), "each series gets a folder of its own");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The writers' budget covers what each writer holds as well as the queue, so a 4K series stays
    /// within `BACKLOG_BYTES`; a frame too large for even four writers gets fewer, and a queue of 0.
    #[test]
    fn the_writers_and_their_queue_stay_within_the_backlog() {
        for (width, height) in [(1280usize, 720usize), (2560, 1440), (3840, 2160), (7680, 4320)] {
            let len = width * height * 4;
            let (writers, bound) = writer_budget(len, 600);
            assert!(writers >= 1);
            if 3 * len <= BACKLOG_BYTES {
                assert!(writers * 3 * len + bound * 2 * len <= BACKLOG_BYTES, "{width}x{height}: {writers} writers, {bound} queued");
            } else {
                assert_eq!((writers, bound), (1, 0), "{width}x{height}: one writer, nothing queued");
            }
        }
        assert_eq!(writer_budget(3840 * 2160 * 4, 600), (4, 2));
        assert_eq!(writer_budget(7680 * 4320 * 4, 600), (1, 0));
        assert_eq!(writer_budget(64, 1).0, 1, "a one-shot needs one writer");
    }

    /// A readback whose bounded wait fails ends the capture, and the present it belonged to waits on
    /// a semaphore signalled after the stalled copy (which consumed the composition's semaphore)
    /// instead of going out ahead of it. The stranded resources go at teardown.
    #[test]
    fn a_stalled_readback_orders_the_present_after_it() {
        let Some((_entry, instance, physical_device, device, queue, family)) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let (width, height) = (16u32, 8u32);
        let base = scratch("stall");
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = crate::shm::ShmClient::test_over_header(&header);
        unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = device.create_image(&info, None).unwrap();
            let reqs = device.get_image_memory_requirements(image);
            let props = instance.get_physical_device_memory_properties(physical_device);
            let type_index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None).unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            clear(&device, queue, pool, image, vk::ImageLayout::UNDEFINED, [0.5, 0.0, 0.0, 1.0]);

            let mut series = Series::default();
            series.start(request(21, 3), &base, shm.channel(), false);
            series.before(&device, &instance, physical_device, queue, family, image, width, height);
            // The composition's semaphore, as `capture::run` returns it.
            let compose_done = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
            let signal = vk::SubmitInfo::builder().signal_semaphores(std::slice::from_ref(&compose_done)).build();
            device.queue_submit(queue, &[signal], vk::Fence::null()).unwrap();
            TEST_READBACK_STALLS.with(|s| s.set(true));
            let wait = series.after(&device, queue, image, width, height, true, Some(compose_done), reason::NONE);
            TEST_READBACK_STALLS.with(|s| s.set(false));
            let PresentWait::Instead(present_wait) = wait else { panic!("the present must wait on the stalled readback: {wait:?}") };
            assert!(!series.running(), "the capture ends");
            assert_eq!(finished(&header, 21).reason, reason::READBACK_FAILED);
            assert!(series.has_stranded());
            // The present's wait, standing in for `vkQueuePresentKHR`'s.
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
            let present = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&present_wait)).wait_dst_stage_mask(&stage).build();
            device.queue_submit(queue, &[present], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);

            device.queue_wait_idle(queue).unwrap();
            series.destroy(&device, true);
            assert!(!series.has_stranded(), "freed once the device is idle");
            device.destroy_semaphore(compose_done, None);
            device.destroy_command_pool(pool, None);
            device.destroy_image(image, None);
            device.free_memory(memory, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A series of three presents where something (standing in for the composition) rewrites
    /// the image between `before` and `after`: each pair must hold the frame as it was before
    /// and after that write, the files must be named by sequence, and the series must end by
    /// itself after the requested count, with an index row per frame and an outcome that says the
    /// effect did not compose them (no semaphore) and why. Then, on the same device: a one-shot
    /// while frames are held before the upscaler (both halves the presented frame, said so), and a
    /// series cancelled by its requester after two frames.
    #[test]
    fn captures_read_back_each_present_before_and_after_the_composition() {
        let Some((_entry, instance, physical_device, device, queue, family)) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let (width, height) = (16u32, 8u32);
        let base = scratch("series");
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = crate::shm::ShmClient::test_over_header(&header);
        let series_dir;
        unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = device.create_image(&info, None).unwrap();
            let reqs = device.get_image_memory_requirements(image);
            let props = instance.get_physical_device_memory_properties(physical_device);
            let type_index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None).unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();

            let mut series = Series::default();
            series.start(request(11, 3), &base, shm.channel(), false);
            assert!(series.running());
            assert_eq!(series.outcome().map(|o| (o.state, o.total)), Some((state::RUNNING, 3)));
            let mut layout = vk::ImageLayout::UNDEFINED;
            for frame in 0..4u32 {
                let game = frame as f32 / 8.0;
                clear(&device, queue, pool, image, layout, [game, 0.0, 0.0, 1.0]);
                layout = vk::ImageLayout::PRESENT_SRC_KHR;
                let was_running = series.running();
                series.before(&device, &instance, physical_device, queue, family, image, width, height);
                clear(&device, queue, pool, image, layout, [game, 1.0, 0.0, 1.0]);
                series.after(&device, queue, image, width, height, true, None, reason::NO_ANSWER);
                assert_eq!(was_running, frame < 3, "the series must end after exactly 3 presents");
            }
            assert!(!series.running());
            let o = finished(&header, 11);
            assert_eq!((o.state, o.reason, o.done, o.total, o.composed), (state::WRITTEN, reason::NO_ANSWER, 3, 3, 0), "{o:?}");
            series_dir = PathBuf::from(o.paths.clone());

            // Held before the upscaler: a one-shot reads the presented frame twice.
            series.start(request(12, 1), &base, shm.channel(), true);
            clear(&device, queue, pool, image, layout, [0.25, 0.5, 0.0, 1.0]);
            series.before(&device, &instance, physical_device, queue, family, image, width, height);
            series.after(&device, queue, image, width, height, true, None, reason::BEFORE_UPSCALER);
            assert!(!series.running(), "a one-shot is one frame");
            let o = finished(&header, 12);
            assert_eq!((o.state, o.reason, o.done), (state::WRITTEN, reason::BEFORE_UPSCALER, 1));
            let paths: Vec<PathBuf> = o.paths().map(PathBuf::from).collect();
            assert_eq!(paths.len(), 2);
            assert!(paths[0].to_string_lossy().ends_with("-original.png") && paths[1].to_string_lossy().ends_with("-composited.png"));
            assert_eq!(paths[0].parent(), Some(base.as_path()), "a one-shot's pair sits in the captures folder itself");
            let (a, b) = (decode(&paths[0]), decode(&paths[1]));
            assert_eq!(a, b, "held: both halves are the presented frame");
            assert!(a[0].abs_diff(64) <= 1 && a[1].abs_diff(128) <= 1 && a[2] == 0 && a[3] == 255, "{:?}", &a[..4]);

            // Cancelled by the requester after two frames: it stops at the next present, keeping both.
            series.start(request(13, 10), &base, shm.channel(), false);
            for _ in 0..2 {
                series.before(&device, &instance, physical_device, queue, family, image, width, height);
                series.after(&device, queue, image, width, height, true, None, reason::NONE);
            }
            assert_eq!(protocol::cancel(&header, 13), protocol::Cancelled::Signalled);
            series.before(&device, &instance, physical_device, queue, family, image, width, height);
            assert!(!series.running());
            let o = finished(&header, 13);
            assert_eq!((o.state, o.done), (state::CANCELLED, 2));

            series.destroy(&device, true);
            device.queue_wait_idle(queue).unwrap();
            device.destroy_command_pool(pool, None);
            device.destroy_image(image, None);
            device.free_memory(memory, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
        let dir = series_dir;
        assert!(dir.file_name().unwrap().to_string_lossy().starts_with("series-"));
        for seq in 0..3u32 {
            let (original, composited) = pair_names(seq);
            let red = (seq as f32 / 8.0 * 255.0).round() as u8;
            // Written as true RGBA (the image is BGRA): red carries the frame, green the "edit".
            let o = decode(&dir.join(original));
            let c = decode(&dir.join(composited));
            assert_eq!(&o[..4], &[red, 0, 0, 255], "frame {seq} original");
            assert_eq!(&c[..4], &[red, 255, 0, 255], "frame {seq} composited");
            assert!(o.chunks(4).all(|p| p == &o[..4]) && c.chunks(4).all(|p| p == &c[..4]));
        }
        assert!(!dir.join(pair_names(3).0).exists(), "a fourth present must not be captured");
        let index = std::fs::read_to_string(dir.join("index.tsv")).unwrap();
        assert_eq!(index.lines().count(), 4, "header plus one row per frame:\n{index}");
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".part")), "no temporary file is left");
        let _ = std::fs::remove_dir_all(&base);
    }
}
