//! The delayed present's relay (the model on a second GPU at the finished frame; docs/NATIVE_BACKEND.md,
//! "Finished-frame pacing").
//!
//! The delayed present shows each frame one present late, composed with the other GPU's answer to it. On its own
//! the game's thread waits three times a present: for the capture of this frame to finish on the GPU (so its request
//! can be posted), for the compose's read of the answer region (the server writes it next), and, at the next
//! present, for the answer. The first two are waits for work queued behind the game's whole frame: the game's
//! thread cannot record the next frame while its GPU renders this one.
//!
//! With the relay the present only submits: it keeps the frame, composes the frame before, and copies this frame's
//! proxy from the kept pair into the proxy region behind that compose (`GpuCompose::send_delayed`). The relay's
//! thread waits for that copy's fence and then lets the other GPU run the frame. The one wait left on the game's
//! thread is for the answer, at the next present, and it is over at once when the other GPU is faster than a frame.
//!
//! What this takes from MAKO's present path (github.com/eugeniosegala/MAKO at 81f1604bc7a2, GPL-3.0-or-later;
//! nothing of its code is used) is the split: the present's thread never waits for GPU work to finish, a host-side
//! thread does, with every wait bounded so that a lost server cannot hang the presents. What it cannot take is
//! MAKO's GPU-side wait for the answer (a submission of the present waiting on a timeline semaphore the host signals
//! when the answer is in): the present's own binary semaphore would then sit behind a wait whose signal is not
//! submitted yet, which the validation layer refuses (VUID-vkQueueSubmit-pWaitSemaphores-03238,
//! VUID-vkQueuePresentKHR-pWaitSemaphores-03268), also when the semaphore is signalled by an ungated submission
//! behind the gated one. MAKO's host is another process, whose signal the game's device never sees pending. What
//! this module has instead, opt-in with `NEURAL_FORGE_OFFLOAD_PACING=gpu` ([`gpu_wait_wanted`]), is a wait on the
//! other device's own submission: the server exports a sync fd after it has submitted the frame ([`publish`]), and
//! the compose and the next send import it and wait for it on the GPU. That signal is always submitted already, so
//! the validation layer has nothing to refuse; but nothing can release such a wait, so a hung other GPU would stall
//! the game's queue, and the default stays the relay with the answer waited for on the game's thread.
//!
//! The releaser is the relay itself: a frame whose copy does not finish or whose answer does not come within its
//! budget is resolved without one ([`Pacer::lost`] from then on: the present goes back to its own waits, which have
//! the channel's rules for a server that does not answer), and the relay's end resolves everything posted.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ash::vk;

/// The longest the relay waits for the game's GPU to be done with a frame's copies.
const GPU_WAIT: Duration = crate::FENCE_WAIT_TIMEOUT;

/// This many delayed presents in a row composed with their answer, on the present's own waits, before the relay
/// takes the frames: the network is built and answers every frame by then, and everything a server that does not
/// answer needs (the warm-up, the give-up, the retry) stays where it is, on the present's own path.
pub(crate) const PACE_AFTER: u32 = 30;

/// The devices set up for the wait on the GPU: created with `VK_KHR_external_semaphore_fd` enabled while it is chosen
/// ([`gpu_wait_wanted`]; `lib.rs`, at their creation; the offload device in `offload::Offload::create`): a sync fd
/// can be exported from, and imported on, those.
static SEMAPHORE_FD: Mutex<Vec<vk::Device>> = Mutex::new(Vec::new());

/// Notes that `device` has `VK_KHR_external_semaphore_fd` (`on`), or forgets it (at its destruction).
pub(crate) fn note_semaphore_fd(device: vk::Device, on: bool) {
    let mut all = SEMAPHORE_FD.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|d| *d != device);
    if on {
        all.push(device);
    }
}

/// Whether `device` was created with `VK_KHR_external_semaphore_fd` ([`note_semaphore_fd`]).
pub(crate) fn has_semaphore_fd(device: vk::Device) -> bool {
    SEMAPHORE_FD.lock().unwrap_or_else(std::sync::PoisonError::into_inner).contains(&device)
}

/// What the other device's submission of a request's frame gave ([`publish`]): a sync fd that signals once that
/// submission's work is done (the answer is in the answer region, which that device writes itself), and what the
/// answer will be, known when the frame was recorded.
pub(crate) struct Submitted {
    pub fd: std::os::fd::OwnedFd,
    /// The size the answer is for (the proxy's, echoed), and whether the frame runs through the network
    /// (`capture::answer_fits`).
    pub dims: (u32, u32),
    pub evaluated: bool,
}

/// The newest submission published per channel (its header's address) and slot, until the relay takes it.
static SUBMITS: Mutex<Vec<(usize, usize, u32, Submitted)>> = Mutex::new(Vec::new());

/// The server on the other device submitted request `req` of `slot` on the channel whose header is at `channel`.
/// Replaces what was published for that slot before (its fd is closed).
pub(crate) fn publish(channel: usize, slot: usize, req: u32, submitted: Submitted) {
    let mut all = SUBMITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|(c, s, _, _)| (*c, *s) != (channel, slot));
    all.push((channel, slot, req, submitted));
}

/// What was published for request `req` ([`publish`]), once.
pub(crate) fn take_published(channel: usize, slot: usize, req: u32) -> Option<Submitted> {
    let mut all = SUBMITS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let k = all.iter().position(|(c, s, r, _)| (*c, *s, *r) == (channel, slot, req))?;
    Some(all.swap_remove(k).3)
}

/// `NEURAL_FORGE_OFFLOAD_PACING`: `cpu` keeps every wait of the delayed present on the game's thread (as before
/// 4.4.0); unset (or anything else) hands the frames to the relay and waits for each answer on the game's thread;
/// `gpu` (experimental) also waits for the answer on the GPU where both devices can pass sync-fd semaphores.
///
/// The GPU wait is opt-in: once the game's queue waits on the other GPU's sync fd, nothing can release it, so a hung
/// or slow other GPU stalls the game's queue (a frozen game), where the waits on the game's thread give up after
/// their budget. Only with it are sync-fd semaphores asked for on either device (`lib.rs`, `offload.rs`) and the
/// server's frames exported (`native_post`).
pub(crate) fn gpu_wait_wanted() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| gpu_wait_in(neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_PACING").as_deref()));
    *ON
}

/// [`gpu_wait_wanted`] for the variable's value `v`.
fn gpu_wait_in(v: Option<&str>) -> bool {
    v.is_some_and(|v| v.trim().eq_ignore_ascii_case("gpu"))
}

/// `NEURAL_FORGE_OFFLOAD_PACING` is not `cpu`: the delayed present hands its frames to the relay.
pub(crate) fn wanted() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_PACING").is_none_or(|v| !v.trim().eq_ignore_ascii_case("cpu")));
    *ON
}

/// The request side of the channel as the relay drives it (the header's atomics; a test's stand-in).
pub(crate) trait Port: Send + Sync {
    /// Posts the next request for the frame in the proxy region; its number, or `None` when the channel is gone.
    fn begin(&self) -> Option<u32>;
    /// Whether request `req` is answered.
    fn answered(&self, req: u32) -> bool;
    /// Whether the answer to `req` is the one to a frame sent as a proxy of `proxy` texels (and, for an HDR10 frame,
    /// ran through the network): `capture::answer_fits`.
    fn fits(&self, req: u32, proxy: (u32, u32), pq10: bool) -> bool;
    /// What the other device's submission of request `req` gave, once it was made and published ([`publish`]); taken
    /// once. `None` from a server that publishes nothing (its answers are waited for on the game's thread).
    fn submitted(&self, _req: u32) -> Option<Submitted> {
        None
    }
    /// The channel is shutting down.
    fn quit(&self) -> bool;
    /// Folds a white-meter reading into the published value.
    fn publish_white(&self, reading: f32);
}

/// The white meter's read of a frame in the proxy region, once the game's GPU has written it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Meter {
    pub frame: *const u8,
    pub bytes: usize,
    pub width: u32,
    pub height: u32,
    pub bgr: bool,
}

/// One frame the present sent: what the relay needs to resolve it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Token {
    /// Its number (from 1, one more each frame sent).
    pub number: u64,
    /// The fence of its copy into the proxy region (`GpuCompose::send_delayed`): signalled once the game's GPU is
    /// done with the regions for this frame (the compose before the copy has read the answer region, too).
    pub copied: vk::Fence,
    /// The size it was sent at, and whether it is an HDR10 frame ([`Port::fits`]).
    pub proxy: (u32, u32),
    pub pq10: bool,
    /// The frame's size (the late rule's).
    pub dims: (u32, u32),
    /// How long its answer may take from its request.
    pub budget: Duration,
    /// When the present sent it.
    pub posted: Instant,
    /// Its answers are judged against the game's frame (`offload::LateRule`; not in the `same` test).
    pub judged: bool,
    pub meter: Option<Meter>,
}

// SAFETY: the meter's pointer is the channel's proxy region, mapped for the life of the process.
unsafe impl Send for Token {}

#[derive(Default)]
struct Queue {
    tokens: VecDeque<Token>,
    /// The relay is working on a token.
    busy: bool,
    quit: bool,
    /// The relay's thread has ended: nothing posted from here on is resolved.
    ended: bool,
    /// The newest token posted.
    posted: u64,
    /// The last tokens resolved, and whether each got a good answer.
    resolved: VecDeque<(u64, bool)>,
    /// A request the relay posted and gave up on, still out at the server: its number and when it was posted.
    outstanding: Option<(u32, Instant)>,
    /// The frame in the relay's hand was submitted on the other device: its number, the sync fd of that submission and
    /// whether the answer will fit, until the present takes it ([`Pacer::wait_ready`]) or the frame is resolved.
    submitted: Option<(u64, std::os::fd::OwnedFd, bool)>,
}

struct Shared {
    device: ash::Device,
    port: Box<dyn Port>,
    queue: Mutex<Queue>,
    changed: Condvar,
    lost: AtomicBool,
}

// SAFETY: the device is only used to wait for a token's fence, which the present's thread resets only once the
// token is resolved (`GpuCompose::send_delayed` reuses a pair's fence two frames later, after the present waited for
// the frame between).
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// The relay between the present's submissions and the other GPU's answers, for one swapchain's delayed present.
pub(crate) struct Pacer {
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn quitting(&self) -> bool {
        self.lock().quit
    }

    /// Token `t` from the game's GPU's copies to the answer. Whether it has a good answer.
    fn run(&self, t: &Token, last_posted: Option<Instant>) -> bool {
        use crate::preupscale::offload::STATS;
        // 1. The game's GPU is done with the regions for this frame.
        let started = Instant::now();
        loop {
            // SAFETY: a fence of the live device, submitted with the frame's copy and not reset before the token is resolved.
            match unsafe { self.device.wait_for_fences(&[t.copied], true, 20_000_000) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) if started.elapsed() < GPU_WAIT && !self.quitting() => {}
                Err(_) => {
                    self.lose("the game's GPU did not finish a frame's copies");
                    return false;
                }
            }
        }
        if let Some(m) = t.meter {
            // SAFETY: the proxy region holds this frame (its copy just finished) and nothing writes it before it is resolved.
            let frame = unsafe { std::slice::from_raw_parts(m.frame, m.bytes) };
            if let Some(white) = crate::capture::meter_white(frame, m.width, m.height, m.bgr) {
                self.port.publish_white(white);
            }
        }
        // 2. The other GPU's turn.
        let Some(req) = self.port.begin() else {
            self.lose("the channel to the model is gone");
            return false;
        };
        let sent = Instant::now();
        let mut polls = 0u32;
        let mut published = false;
        while !self.port.answered(req) {
            // The other device's submission of this frame, for a present that waits for the answer on the GPU.
            if !published {
                if let Some(s) = self.port.submitted(req) {
                    published = true;
                    let fits = crate::capture::answer_fits((s.dims != (0, 0)).then_some(s.dims), t.proxy, t.pq10, s.evaluated);
                    self.lock().submitted = Some((t.number, s.fd, fits));
                    self.changed.notify_all();
                }
            }
            if sent.elapsed() >= t.budget || self.port.quit() || self.quitting() {
                self.lock().outstanding = Some((req, sent));
                self.lose("a frame's answer did not come within its budget");
                self.judge(t, None, last_posted);
                return false;
            }
            polls += 1;
            // As the CPU's wait polls; a yield first, since most answers are a few milliseconds away.
            if polls < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        std::sync::atomic::fence(Ordering::Acquire);
        let took = sent.elapsed();
        let fits = self.port.fits(req, t.proxy, t.pq10);
        if fits {
            STATS.round_trip_us.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
            STATS.round_trips.fetch_add(1, Ordering::Relaxed);
        }
        self.judge(t, Some(took), last_posted);
        fits
    }

    /// The late rule, as the CPU's wait feeds it: the answer's time against the game's interval between two frames.
    fn judge(&self, t: &Token, answer: Option<Duration>, last_posted: Option<Instant>) {
        let (true, Some(last)) = (t.judged, last_posted) else { return };
        let interval = t.posted.saturating_duration_since(last);
        let verdict =
            crate::preupscale::offload::LATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_or_insert_default().note(answer, interval, t.dims);
        if let Some(why) = verdict {
            crate::preupscale::offload::give_up(why);
        }
    }

    fn lose(&self, why: &str) {
        if !self.lost.swap(true, Ordering::AcqRel) {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 5 {
                crate::log!("[offload] the presents take their waits back ({why}): the frame waiting goes out as it is, and the relay has the frames again once the answers come every frame");
                crate::logging::flush();
            }
        }
    }
}

/// Resolves everything posted, without an answer, when the relay's thread ends, however it ends: no present waits
/// for a frame the relay will never resolve.
struct ReleaseOnExit<'a>(&'a Shared);

impl Drop for ReleaseOnExit<'_> {
    fn drop(&mut self) {
        let s = self.0;
        s.lost.store(true, Ordering::Release);
        let mut q = s.lock();
        q.ended = true;
        q.busy = false;
        q.tokens.clear();
        drop(q);
        s.changed.notify_all();
    }
}

fn relay(s: &Shared) {
    let _release = ReleaseOnExit(s);
    let mut last_posted: Option<Instant> = None;
    loop {
        let token = {
            let mut q = s.lock();
            loop {
                if q.quit {
                    return;
                }
                if let Some(t) = q.tokens.pop_front() {
                    q.busy = true;
                    break t;
                }
                q = s.changed.wait(q).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        // Once lost, a frame already posted is resolved without being sent (its request would be written over the
        // one that is still out).
        let valid = !s.lost.load(Ordering::Acquire) && s.run(&token, last_posted);
        last_posted = Some(token.posted);
        let mut q = s.lock();
        q.busy = false;
        // Not taken by a present: the answer is in, nothing waits for it on the GPU.
        q.submitted = None;
        q.resolved.push_back((token.number, valid));
        while q.resolved.len() > 8 {
            q.resolved.pop_front();
        }
        drop(q);
        s.changed.notify_all();
    }
}

impl Pacer {
    /// Starts the relay for frames of `device` over `port`.
    ///
    /// # Safety
    /// `device` is live until the pacer is stopped, and so is the fence of every token posted until it is resolved.
    pub(crate) unsafe fn start(device: ash::Device, port: Box<dyn Port>) -> Option<Self> {
        let shared = Arc::new(Shared { device, port, queue: Mutex::default(), changed: Condvar::new(), lost: AtomicBool::new(false) });
        let s = shared.clone();
        let thread = std::thread::Builder::new().name("nf-offload-pace".into()).spawn(move || relay(&s)).ok()?;
        Some(Self { shared, thread: Mutex::new(Some(thread)) })
    }

    /// A frame the present just sent (its copy into the proxy region is submitted with the token's fence). `false`
    /// when the relay has ended: the frame is not sent.
    pub(crate) fn post(&self, token: Token) -> bool {
        let mut q = self.shared.lock();
        if q.ended || q.quit {
            return false;
        }
        q.posted = q.posted.max(token.number);
        q.tokens.push_back(token);
        drop(q);
        self.shared.changed.notify_all();
        true
    }

    /// A frame got no answer in time, or the relay ended: the present paces on the CPU until [`Self::recover`].
    pub(crate) fn lost(&self) -> bool {
        self.shared.lost.load(Ordering::Acquire)
    }

    /// The relay's thread has ended: the pacer cannot be used again.
    pub(crate) fn ended(&self) -> bool {
        self.shared.lock().ended
    }

    /// Waits until frame `number` is resolved or `deadline` passes: the present's one wait, for the answer to the
    /// frame before. Whether it has a good answer; `None` when it is not resolved in time.
    pub(crate) fn wait_resolved(&self, number: u64, deadline: Instant) -> Option<bool> {
        let mut q = self.shared.lock();
        loop {
            if let Some((_, valid)) = q.resolved.iter().find(|(n, _)| *n == number) {
                return Some(*valid);
            }
            if q.ended {
                return Some(false);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            q = self.shared.changed.wait_timeout(q, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }

    /// Waits until frame `number` is resolved, or submitted on the other device ([`Port::submitted`]), or `deadline`
    /// passes: the present that can wait for the answer on the GPU. A frame already resolved is [`Ready::Resolved`]
    /// (nothing is left to wait for); `None` when neither came in time.
    pub(crate) fn wait_ready(&self, number: u64, deadline: Instant) -> Option<Ready> {
        let mut q = self.shared.lock();
        loop {
            if let Some((_, valid)) = q.resolved.iter().find(|(n, _)| *n == number) {
                return Some(Ready::Resolved(*valid));
            }
            if q.ended {
                return Some(Ready::Resolved(false));
            }
            if q.submitted.as_ref().is_some_and(|(n, _, _)| *n == number) {
                let (_, fd, fits) = q.submitted.take().expect("checked");
                return Some(Ready::Submitted { fd, fits });
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            q = self.shared.changed.wait_timeout(q, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }

    /// Waits until the relay has nothing posted and nothing in hand, `deadline` at most. Whether it is so.
    pub(crate) fn drain(&self, deadline: Instant) -> bool {
        let mut q = self.shared.lock();
        loop {
            if q.ended || (q.tokens.is_empty() && !q.busy) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            q = self.shared.changed.wait_timeout(q, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }

    /// The request the relay gave up on, if it is still out at the server (taken once): the channel's client takes
    /// it over, so that no new request is written over it.
    pub(crate) fn take_outstanding(&self) -> Option<(u32, Instant)> {
        self.shared.lock().outstanding.take()
    }

    /// The relay is idle and every frame resolved ([`Self::drain`]): it may be given frames again.
    pub(crate) fn recover(&self) -> bool {
        let q = self.shared.lock();
        if q.ended || q.busy || !q.tokens.is_empty() || q.outstanding.is_some() {
            return false;
        }
        self.shared.lost.store(false, Ordering::Release);
        true
    }

    /// Ends the relay: the frame in its hand is resolved without an answer. Before the fences it waits on are
    /// destroyed.
    pub(crate) fn stop(&self) {
        self.shared.lock().quit = true;
        self.shared.changed.notify_all();
        let thread = self.thread.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(t) = thread {
            let _ = t.join();
        }
    }
}

impl Drop for Pacer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// What a present finds of the frame before it ([`Pacer::wait_ready`]).
pub(crate) enum Ready {
    /// Resolved: with a good answer, or without.
    Resolved(bool),
    /// Submitted on the other device and not finished: the sync fd of that submission, and whether its answer will
    /// fit the frame.
    Submitted { fd: std::os::fd::OwnedFd, fits: bool },
}

/// How many delayed presents handed their frame to the relay and how many made every wait themselves, and how long
/// the game's thread spent in them (µs), for the `[offload]` summary.
pub(crate) struct Waits {
    /// Of the presents with the relay: those whose compose waited for the answer on the GPU.
    pub gated: AtomicU64,
    pub gpu_presents: AtomicU64,
    pub gpu_us: AtomicU64,
    pub cpu_presents: AtomicU64,
    pub cpu_us: AtomicU64,
}

pub(crate) static WAITS: Waits =
    Waits { gated: AtomicU64::new(0), gpu_presents: AtomicU64::new(0), gpu_us: AtomicU64::new(0), cpu_presents: AtomicU64::new(0), cpu_us: AtomicU64::new(0) };

/// Tests: every compose that waited for its answer on the GPU, never reset (the summary resets [`Waits::gated`]).
#[cfg(test)]
pub(crate) static GATED_TOTAL: AtomicU64 = AtomicU64::new(0);

impl Waits {
    /// A delayed present took the game's thread `took`, with the relay (`gpu`) or on its own waits.
    pub(crate) fn note(&self, gpu: bool, took: Duration) {
        let (n, us) = if gpu { (&self.gpu_presents, &self.gpu_us) } else { (&self.cpu_presents, &self.cpu_us) };
        n.fetch_add(1, Ordering::Relaxed);
        us.fetch_add(took.as_micros() as u64, Ordering::Relaxed);
    }

    /// The summary's words since the last call (the counters start over).
    pub(crate) fn summary(&self) -> String {
        let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
        let mean = |us: u64, n: u64| if n == 0 { 0.0 } else { us as f64 / n as f64 / 1000.0 };
        let (gn, gu, cn, cu, gated) = (take(&self.gpu_presents), take(&self.gpu_us), take(&self.cpu_presents), take(&self.cpu_us), take(&self.gated));
        format!(
            "the game's thread spent {:.2} ms in each of {gn} presents that left their GPU waits to the relay ({gated} of them waited for the answer on the GPU) and {:.2} ms in each of {cn} that waited themselves",
            mean(gu, gn),
            mean(cu, cn)
        )
    }
}

/// The tests' other device: a second device of the test driver whose every "frame" is an empty submission that
/// signals an exportable semaphore, exported as a sync fd once the submission is made, as the server on the other GPU
/// does for its frames.
#[cfg(test)]
pub(crate) mod test_support {
    use ash::vk;
    use std::os::fd::{FromRawFd, OwnedFd};

    pub(crate) struct OtherDevice {
        _entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        queue: vk::Queue,
        api: ash::khr::external_semaphore_fd::Device,
        done: vk::Semaphore,
    }

    // SAFETY: the test's own device, used by one thread at a time.
    unsafe impl Send for OtherDevice {}

    impl OtherDevice {
        /// `None` where the test driver has no sync-fd semaphores.
        pub(crate) fn open() -> Option<Self> {
            // SAFETY: as `composition::gpu::test_device`; plain queries and valid requests.
            unsafe {
                let entry = ash::Entry::load().ok()?;
                let (instance, _) = crate::composition::gpu::test_instance(&entry)?;
                let made = (|| {
                    let physical = *instance.enumerate_physical_devices().ok()?.first()?;
                    let name = ash::khr::external_semaphore_fd::NAME;
                    let offered = instance
                        .enumerate_device_extension_properties(physical)
                        .is_ok_and(|e| e.iter().any(|x| std::ffi::CStr::from_ptr(x.extension_name.as_ptr()) == name));
                    let kind = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                    let mut props = vk::ExternalSemaphoreProperties::default();
                    instance.get_physical_device_external_semaphore_properties(physical, &kind, &mut props);
                    if !offered || !props.external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE) {
                        return None;
                    }
                    let names = [name.as_ptr()];
                    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&[1.0])];
                    instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&names), None).ok()
                })();
                let Some(device) = made else {
                    instance.destroy_instance(None);
                    return None;
                };
                let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                let done = device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None).unwrap();
                let api = ash::khr::external_semaphore_fd::Device::new(&instance, &device);
                Some(Self { queue: device.get_device_queue(0, 0), _entry: entry, instance, device, api, done })
            }
        }

        /// One frame's submission on this device, and its sync fd.
        pub(crate) fn submit(&self) -> OwnedFd {
            let signals = [self.done];
            // SAFETY: the test's own objects; the export follows the submission that signals the semaphore, and takes its
            // payload, so the semaphore is ready for the next frame.
            unsafe {
                self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().signal_semaphores(&signals)], vk::Fence::null()).unwrap();
                let info = vk::SemaphoreGetFdInfoKHR::default().semaphore(self.done).handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                OwnedFd::from_raw_fd(self.api.get_semaphore_fd(&info).unwrap())
            }
        }

        pub(crate) fn close(self) {
            // SAFETY: nothing of it is pending once a submission behind every frame's has finished.
            unsafe {
                let last = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
                self.device.queue_submit(self.queue, &[], last).unwrap();
                self.device.wait_for_fences(&[last], true, 30_000_000_000).unwrap();
                self.device.destroy_fence(last, None);
                self.device.destroy_semaphore(self.done, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wait on the GPU cannot be released, so it is only ever chosen by name: unset and every other value keep the
    /// answer's wait on the game's thread.
    #[test]
    fn the_gpu_wait_is_opt_in() {
        for off in [None, Some("relay"), Some("cpu"), Some(""), Some("1"), Some("on")] {
            assert!(!gpu_wait_in(off), "{off:?} does not choose the GPU wait");
        }
        for on in ["gpu", "GPU", " gpu "] {
            assert!(gpu_wait_in(Some(on)), "{on:?} chooses it");
        }
    }

    /// The channel's request side for the relay's tests: requests are numbered from 1, answered when the test says
    /// so, and fit when it says so; submitted on "the other device" when the test hands a fd over.
    #[derive(Default)]
    struct FakePort {
        begun: AtomicU64,
        answers: AtomicBool,
        fits: AtomicBool,
        submitted: Mutex<Option<Submitted>>,
    }

    impl Port for Arc<FakePort> {
        fn begin(&self) -> Option<u32> {
            Some(self.begun.fetch_add(1, Ordering::SeqCst) as u32 + 1)
        }
        fn answered(&self, _req: u32) -> bool {
            self.answers.load(Ordering::SeqCst)
        }
        fn fits(&self, _req: u32, _proxy: (u32, u32), _pq10: bool) -> bool {
            self.fits.load(Ordering::SeqCst)
        }
        fn submitted(&self, _req: u32) -> Option<Submitted> {
            self.submitted.lock().unwrap().take()
        }
        fn quit(&self) -> bool {
            false
        }
        fn publish_white(&self, _reading: f32) {}
    }

    /// A present that can wait for the answer on the GPU is given the other device's submission as soon as it is
    /// made, before the answer is in, with whether the answer will fit; once the answer is in, there is nothing left
    /// to hand over.
    #[test]
    fn a_present_gets_the_other_devices_submission_before_the_answer() {
        let Some(mut rig) = Rig::open() else { return };
        let any_fd = || std::os::fd::OwnedFd::from(std::fs::File::open("/dev/null").unwrap());
        let one = rig.token(1, Duration::from_secs(20));
        assert!(rig.pacer.post(one));
        rig.gpu_done(1);
        assert!(rig.pacer.wait_ready(1, Instant::now() + Duration::from_millis(60)).is_none(), "posted, not submitted there yet");
        *rig.port.submitted.lock().unwrap() = Some(Submitted { fd: any_fd(), dims: (16, 24), evaluated: false });
        assert!(matches!(rig.pacer.wait_ready(1, soon()), Some(Ready::Submitted { fits: true, .. })));
        // Taken once; the frame is resolved when its answer is in, as always.
        rig.port.fits.store(true, Ordering::SeqCst);
        rig.port.answers.store(true, Ordering::SeqCst);
        assert!(matches!(rig.pacer.wait_ready(1, soon()), Some(Ready::Resolved(true))));
        // Another size than the frame was sent at: said with the submission.
        rig.port.answers.store(false, Ordering::SeqCst);
        let two = rig.token(2, Duration::from_secs(20));
        assert!(rig.pacer.post(two));
        rig.gpu_done(2);
        *rig.port.submitted.lock().unwrap() = Some(Submitted { fd: any_fd(), dims: (8, 8), evaluated: false });
        assert!(matches!(rig.pacer.wait_ready(2, soon()), Some(Ready::Submitted { fits: false, .. })));
        // A submission no present took is dropped when the answer comes.
        let three = rig.token(3, Duration::from_secs(20));
        assert!(rig.pacer.post(three));
        *rig.port.submitted.lock().unwrap() = Some(Submitted { fd: any_fd(), dims: (16, 24), evaluated: false });
        rig.port.answers.store(true, Ordering::SeqCst);
        rig.gpu_done(3);
        assert_eq!(rig.pacer.wait_resolved(3, soon()), Some(true));
        assert!(matches!(rig.pacer.wait_ready(3, soon()), Some(Ready::Resolved(true))));
        // The published submissions are kept per channel and slot, newest only, and taken once.
        publish(0x51ab, 0, 7, Submitted { fd: any_fd(), dims: (1, 1), evaluated: true });
        publish(0x51ab, 0, 8, Submitted { fd: any_fd(), dims: (2, 2), evaluated: true });
        assert!(take_published(0x51ab, 0, 7).is_none() && take_published(0x51ab, 1, 8).is_none());
        assert!(take_published(0x51ab, 0, 8).is_some_and(|s| s.dims == (2, 2)) && take_published(0x51ab, 0, 8).is_none());
        rig.close();
    }

    /// A device, fences standing in for the frames' copies, and the relay over a [`FakePort`]. Each frame's "copy" is
    /// an empty submission that waits for the rig's timeline to reach the frame's number, so the test decides when
    /// the game's GPU is done with it.
    struct Rig {
        _entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        queue: vk::Queue,
        gpu: vk::Semaphore,
        fences: Vec<vk::Fence>,
        port: Arc<FakePort>,
        pacer: Pacer,
    }

    impl Rig {
        fn open() -> Option<Self> {
            // SAFETY: as `composition::gpu::test_device`, with timeline semaphores for the rig's own gate.
            let entry = unsafe { ash::Entry::load() }.ok()?;
            let (instance, surface) = crate::composition::gpu::test_instance(&entry)?;
            // SAFETY: the instance just made; plain queries and a valid request.
            let made = unsafe {
                let physical = *instance.enumerate_physical_devices().ok()?.first()?;
                let mut has = vk::PhysicalDeviceVulkan12Features::default();
                instance.get_physical_device_features2(physical, &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut has));
                let names = crate::composition::gpu::test_device_extensions(&instance, physical, surface, &[]);
                let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&[1.0])];
                let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
                let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&names).push_next(&mut v12);
                if has.timeline_semaphore == vk::TRUE {
                    instance.create_device(physical, &info, None).ok()
                } else {
                    None
                }
            };
            let Some(device) = made else {
                assert_ne!(neural_forge_protocol::env::var("NEURAL_FORGE_REQUIRE_VULKAN").as_deref(), Some("1"), "no Vulkan device for the relay's tests");
                // SAFETY: nothing was made from it.
                unsafe { instance.destroy_instance(None) };
                return None;
            };
            // SAFETY: the device just made.
            let (queue, gpu) = unsafe {
                let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
                (device.get_device_queue(0, 0), device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None).unwrap())
            };
            let port = Arc::new(FakePort::default());
            // SAFETY: the device and the fences live until `close`, which stops the relay first.
            let pacer = unsafe { Pacer::start(device.clone(), Box::new(port.clone())) }?;
            Some(Self { _entry: entry, instance, device, queue, gpu, fences: Vec::new(), port, pacer })
        }

        /// Frame `number`, its copy submitted and not done until [`Self::gpu_done`].
        fn token(&mut self, number: u64, budget: Duration) -> Token {
            // SAFETY: a live device; the fence is used by this submit, then only waited on.
            let copied = unsafe {
                let copied = self.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
                let (waits, values, stages) = ([self.gpu], [number], [vk::PipelineStageFlags::ALL_COMMANDS]);
                let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&values);
                let submit = vk::SubmitInfo::default().wait_semaphores(&waits).wait_dst_stage_mask(&stages).push_next(&mut timeline);
                self.device.queue_submit(self.queue, &[submit], copied).unwrap();
                copied
            };
            self.fences.push(copied);
            Token { number, copied, proxy: (16, 24), pq10: false, dims: (16, 24), budget, posted: Instant::now(), judged: false, meter: None }
        }

        /// The game's GPU is done with frame `number`'s copy.
        fn gpu_done(&self, number: u64) {
            // SAFETY: the rig's own timeline, signalled with rising values by the test alone.
            unsafe { self.device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(self.gpu).value(number)) }.unwrap();
        }

        fn close(self) {
            self.pacer.stop();
            // SAFETY: the relay ended; nothing else uses them.
            unsafe {
                self.device.device_wait_idle().unwrap();
                for f in self.fences {
                    self.device.destroy_fence(f, None);
                }
                self.device.destroy_semaphore(self.gpu, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    fn soon() -> Instant {
        Instant::now() + Duration::from_secs(20)
    }

    /// A frame is sent to the other GPU only once the game's GPU is done with its copy, and resolved when its answer
    /// is in: good, or not fitting.
    #[test]
    fn frames_are_resolved_in_order_once_the_gpu_is_done_and_the_answer_is_in() {
        let Some(mut rig) = Rig::open() else { return };
        rig.port.answers.store(true, Ordering::SeqCst);
        rig.port.fits.store(true, Ordering::SeqCst);
        let (one, two) = (rig.token(1, Duration::from_secs(20)), rig.token(2, Duration::from_secs(20)));
        assert!(rig.pacer.post(one) && rig.pacer.post(two));
        assert_eq!(rig.pacer.wait_resolved(1, Instant::now() + Duration::from_millis(60)), None, "not before the game's GPU is done with it");
        assert_eq!(rig.port.begun.load(Ordering::SeqCst), 0, "no request before the frame is in the region");
        assert!(!rig.pacer.drain(Instant::now()), "the relay has the frame");
        rig.gpu_done(1);
        assert_eq!(rig.pacer.wait_resolved(1, soon()), Some(true));
        assert_eq!(rig.pacer.wait_resolved(2, Instant::now() + Duration::from_millis(60)), None, "frame 2 waits for its own copy");
        rig.port.fits.store(false, Ordering::SeqCst);
        rig.gpu_done(2);
        assert_eq!(rig.pacer.wait_resolved(2, soon()), Some(false), "an answer that does not fit is no answer");
        assert_eq!(rig.port.begun.load(Ordering::SeqCst), 2);
        assert!(!rig.pacer.lost() && rig.pacer.drain(soon()));
        rig.close();
    }

    /// The releaser: a frame whose answer does not come within its budget is resolved without it, so the present
    /// waits no longer; a frame posted behind it is resolved without being sent (its request would be written over
    /// the one still out), and the request that is out is handed over, once.
    #[test]
    fn a_frame_without_an_answer_in_its_budget_is_given_up_and_the_request_handed_over() {
        let Some(mut rig) = Rig::open() else { return };
        rig.port.fits.store(true, Ordering::SeqCst);
        let (one, two) = (rig.token(1, Duration::from_millis(40)), rig.token(2, Duration::from_secs(20)));
        assert!(rig.pacer.post(one) && rig.pacer.post(two));
        rig.gpu_done(1);
        assert_eq!(rig.pacer.wait_resolved(1, soon()), Some(false));
        assert!(rig.pacer.lost());
        assert_eq!(rig.pacer.wait_resolved(2, soon()), Some(false), "resolved without waiting for its copy or an answer");
        assert_eq!(rig.port.begun.load(Ordering::SeqCst), 1, "frame 2 was never sent");
        assert!(rig.pacer.drain(soon()));
        assert!(!rig.pacer.recover(), "not while its request is still out");
        assert_eq!(rig.pacer.take_outstanding().map(|(req, _)| req), Some(1));
        assert_eq!(rig.pacer.take_outstanding(), None);
        assert!(rig.pacer.recover() && !rig.pacer.lost());
        // Given frames again: the next one goes through.
        rig.port.answers.store(true, Ordering::SeqCst);
        let three = rig.token(3, Duration::from_secs(20));
        assert!(rig.pacer.post(three));
        rig.gpu_done(3);
        assert_eq!(rig.pacer.wait_resolved(3, soon()), Some(true));
        rig.close();
    }

    /// The relay's end, however it comes, resolves everything posted: no present waits for a frame it will never
    /// resolve, and it takes nothing more.
    #[test]
    fn the_relays_end_resolves_everything_that_waits() {
        let Some(mut rig) = Rig::open() else { return };
        let (one, two) = (rig.token(1, Duration::from_secs(20)), rig.token(2, Duration::from_secs(20)));
        assert!(rig.pacer.post(one) && rig.pacer.post(two));
        // The game's GPU never finishes frame 1's copy; the relay is stopped while it waits for it.
        let started = Instant::now();
        rig.pacer.stop();
        assert!(started.elapsed() < Duration::from_secs(4), "stopping does not wait out the GPU's bound");
        assert!(rig.pacer.ended() && rig.pacer.lost());
        assert_eq!(rig.pacer.wait_resolved(1, Instant::now()), Some(false));
        assert_eq!(rig.pacer.wait_resolved(2, Instant::now()), Some(false));
        let three = rig.token(3, Duration::from_secs(1));
        assert!(!rig.pacer.post(three), "nothing is taken once it ended");
        assert!(rig.pacer.drain(Instant::now()) && !rig.pacer.recover());
        rig.gpu_done(3);
        rig.close();
    }

    #[test]
    fn the_summary_says_what_the_games_thread_spent_with_and_without_the_relay() {
        let w = Waits {
            gated: AtomicU64::new(0),
            gpu_presents: AtomicU64::new(0),
            gpu_us: AtomicU64::new(0),
            cpu_presents: AtomicU64::new(0),
            cpu_us: AtomicU64::new(0),
        };
        w.note(true, Duration::from_micros(300));
        w.note(true, Duration::from_micros(500));
        w.note(false, Duration::from_millis(6));
        let line = w.summary();
        assert!(line.contains("0.40 ms in each of 2 presents that left") && line.contains("6.00 ms in each of 1 that waited themselves"), "{line}");
        assert!(line.contains("(0 of them waited for the answer on the GPU)"), "{line}");
        assert!(w.summary().contains("each of 0 presents"), "the counters start over");
    }
}
