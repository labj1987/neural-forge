//! The native backend (docs/NATIVE_BACKEND.md, "Phase 1"): the network runs on the game's own device,
//! inside the game's own DLSS submit, instead of in the model server.
//!
//! A hold in native mode submits three command buffers in front of DLSS's launch buffer, with no CPU
//! wait between them and none for them:
//!
//! - `C`, the capture (`super::record_capture`): the exposure texel, the HDR encode into the encoded
//!   image, and a copy of DLSS's motion vectors into [`NativePass`]'s motion buffer;
//! - `N[parity]`, recorded once per build ([`NativePass::record`]): `native_preprocess.comp` (the
//!   network's 16 input lanes from the encoded proxy and the reprojected history), the network's
//!   recorded graph (a secondary from `neural-forge-native`), and `native_composite.comp` (the
//!   residual, the blend toward the history, the next history and the answer buffer);
//! - `W`, the write-back (`super::record_writeback`): the answer decoded over DLSS's colour input.
//!
//! With the edit cache on (the default) the holds between two network frames are its cached
//! holds (`super::cache`, docs/ARCHITECTURE.md 4.10): `C` as always, no `N`, and a `W` that carries the
//! last network frame's correction to this frame instead of decoding an answer. The capture's motion
//! pass writes [`NativePass::acc`], the motion back to the last network frame that the frame shaders
//! read (at interval 1, DLSS's vectors as they are).
//!
//! The ordering is submission order on the game's queue plus each buffer's opening barrier
//! (`ALL_COMMANDS/MEMORY_WRITE` to what it reads), the same chain as the model server hold's
//! (`super`, "The dependency chain of a hold"), minus the host round trip.
//!
//! The network itself (model, kernels, the graph for one size) is loaded and built on a thread of its
//! own, on a queue the layer added to the device for it ([`Loader`]); until it is ready, frames go to
//! DLSS untouched.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use ash::vk;
use ash::vk::Handle;
use neural_forge_native as nn;
use neural_forge_protocol::history::{HistoryGap, Stale};
use neural_forge_protocol::rebuild::{BuildRetry, FailInject, Step};

use super::hdr::{self, OwnImage};
use super::{own_host_buffer_with, HostBuffer, Target};

const PREPROCESS_SPV: &[u8] = include_bytes!("../../shaders/native_preprocess.spv");
const COMPOSITE_SPV: &[u8] = include_bytes!("../../shaders/native_composite.spv");

/// `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`: the frames record everything but the network's launches.
fn skip_graph() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_SKIP_GRAPH").as_deref() == Some("1"));
    *ON
}

/// `NEURAL_FORGE_NATIVE_CHAIN=0`: barriers between the network's launches instead of counter chaining.
fn chain_wanted() -> bool {
    neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_CHAIN").is_none_or(|v| v != "0")
}

/// `extract-model`'s output: `$XDG_DATA_HOME/neural-forge/model`.
pub(crate) fn model_dir() -> PathBuf {
    crate::dump::captures_dir().with_file_name("model")
}

/// Whether the model directory is there at all (the hash check is the loader's).
pub(crate) fn model_present() -> bool {
    model_dir().join("manifest.json").is_file()
}

/// "build 310.8.0.0, verified": the manifest's `source.build` and `source.verified`, for the log.
/// Both keys appear once in `extract-model`'s manifest, so a string scan is enough. A manifest
/// without `verified` is unverified.
fn model_source(text: &str) -> String {
    let field = |key: &str| {
        let (_, rest) = text.split_once(&format!("\"{key}\": "))?;
        Some(rest.split([',', '\n', '}']).next()?.trim().trim_matches('"').to_string())
    };
    let build = field("build").unwrap_or_else(|| "unknown".to_string());
    let verified = field("verified").is_some_and(|v| v == "true");
    format!("build {build}, {}", if verified { "verified" } else { "not verified against NVIDIA's runtime" })
}

/// Where the network runs on a device: what `create_device` added for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Setup {
    pub gipa: vk::PFN_vkGetInstanceProcAddr,
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    /// The game's graphics family, where DLSS runs and the frame work goes.
    pub frame_family: u32,
    /// The layer's own queue for the network (a compute family without graphics): the loader's
    /// uploads and warm-up only.
    pub family: u32,
    pub index: u32,
    /// A second queue of that family: the after-the-upscaler path's network frames (`native_post`).
    pub post_index: u32,
    /// Which memory-properties query the instance may make (the VRAM readings, `crate::vram`).
    pub properties2: crate::vram::Properties2,
}

/// The network for one frame size, as the hold uses it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Built {
    pub width: u32,
    pub height: u32,
    pub frame: nn::Frame,
    pub graph: vk::CommandBuffer,
    /// The same graph recorded for the layer's own compute family (`native_post`).
    pub graph_own: vk::CommandBuffer,
    /// Changes with every build: [`NativePass`] re-records its buffers when it does.
    pub generation: u64,
}

#[derive(Default)]
struct LoaderState {
    /// `None` while the worker has it out (opening, building), before it is opened, and after a
    /// re-initialisation closed it.
    network: Option<nn::Network>,
    /// Why the network is not usable right now (the Status tab shows it); cleared by a success.
    failed: Option<String>,
    built: Option<Built>,
    /// The size the holds want; the worker builds it.
    wanted: Option<(u32, u32)>,
    fall_back: bool,
    quit: bool,
    generation: u64,
    /// Opened at least once (a later `None` network means closed for a re-initialisation).
    opened: bool,
    /// A wait on the network's queue timed out ([`nn::Network::stalled`]): the network is dead on this
    /// device and never opened or built again.
    dead: bool,
}

/// The network on one device, opened and built off the game's threads. Nothing is loaded until the
/// first hold asks for it ([`Self::ready`]): only the device DLSS runs on pays for the model. A
/// failed open or build is tried again on the model server's schedule (`neural_forge_protocol::rebuild`):
/// 0.5, 1 and 2 s, then a re-initialisation (the network closed and opened again), then every 30 s.
pub(crate) struct Loader {
    shared: Arc<(Mutex<LoaderState>, Condvar)>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    pending: Mutex<Option<(vk::Device, Setup)>>,
    setup: Setup,
    /// Who uses the network's frame (activations, counters), which the two paths share: the last
    /// pre-upscaler hold and the last after-the-upscaler frame (ms since [`epoch_ms`]'s start), and
    /// whether the latter is running now ([`Self::claim_post`], [`Self::claim_pre`]).
    last_pre: std::sync::atomic::AtomicU64,
    last_post: std::sync::atomic::AtomicU64,
    post_busy: std::sync::atomic::AtomicBool,
    /// The fence of the pre-upscaler path's last submitted hold while it has not been waited on
    /// ([`Self::pre_submitted`]): the after-the-upscaler path does not take the network (whose build
    /// for another size would free what that hold runs) until it has signalled.
    pre_inflight: Mutex<Option<InFlight>>,
    /// The after-the-upscaler frame that did not finish in time, still holding the claim
    /// ([`Self::post_stalled`]): released once its fence signals.
    post_inflight: Mutex<Option<InFlight>>,
}

/// A submission of one path that may still use the network, by its fence.
#[derive(Clone, Copy)]
struct InFlight {
    device: vk::Device,
    status: vk::PFN_vkGetFenceStatus,
    fence: vk::Fence,
}

impl InFlight {
    fn of(device: &ash::Device, fence: vk::Fence) -> Self {
        Self { device: device.handle(), status: device.fp_v1_0().get_fence_status, fence }
    }

    /// Whether its work may still run: not signalled (a lost device runs nothing).
    fn running(&self) -> bool {
        // SAFETY: a fence of the live device, kept alive while recorded (`Loader::pre_idle`,
        // `post_stalled`); vkGetFenceStatus needs no external synchronization.
        let status = unsafe { (self.status)(self.device, self.fence) };
        status == vk::Result::NOT_READY
    }
}

/// Milliseconds since the first call in this process (a monotonic stamp for [`Loader`]'s claims).
fn epoch_ms() -> u64 {
    static START: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);
    START.elapsed().as_millis() as u64 + 1
}

/// How long after one path last used the network the other may.
const PATH_SWITCH_MS: u64 = 1000;

unsafe extern "C" fn init_dispatchable(_user: *mut std::ffi::c_void, device: vk::Device, object: *mut std::ffi::c_void) {
    // SAFETY: a dispatchable object of `device` the network just got from below the loader.
    let _ = unsafe { crate::loader_data::initialize_object(device, object) };
}

impl Loader {
    pub(crate) fn start(device: vk::Device, setup: Setup) -> Self {
        Self {
            shared: Arc::default(),
            worker: Mutex::new(None),
            pending: Mutex::new(Some((device, setup))),
            setup,
            last_pre: Default::default(),
            last_post: Default::default(),
            post_busy: Default::default(),
            pre_inflight: Mutex::new(None),
            post_inflight: Mutex::new(None),
        }
    }

    pub(crate) fn setup(&self) -> Setup {
        self.setup
    }

    /// The after-the-upscaler path may run a frame now (no pre-upscaler hold for a second). Pair with
    /// [`Self::release_post`] once its work completed.
    pub(crate) fn claim_post(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        if epoch_ms().saturating_sub(self.last_pre.load(SeqCst)) < PATH_SWITCH_MS || self.pre_running() {
            return false;
        }
        self.post_busy.store(true, SeqCst);
        if epoch_ms().saturating_sub(self.last_pre.load(SeqCst)) < PATH_SWITCH_MS {
            self.post_busy.store(false, SeqCst);
            return false;
        }
        true
    }

    pub(crate) fn release_post(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        self.last_post.store(epoch_ms(), SeqCst);
        self.post_busy.store(false, SeqCst);
    }

    /// The after-the-upscaler frame holding the claim did not finish in time: its work (on `fence`)
    /// may still use the network's buffers, so the claim is kept until that fence signals (checked by
    /// [`Self::claim_pre`]). The fence stays alive: a stalled server destroys nothing.
    pub(crate) fn post_stalled(&self, device: &ash::Device, fence: vk::Fence) {
        *self.post_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(InFlight::of(device, fence));
    }

    /// Releases a stalled after-the-upscaler claim whose frame has finished since.
    fn post_finished(&self) {
        let mut stalled = self.post_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if stalled.is_some_and(|f| !f.running()) {
            *stalled = None;
            drop(stalled);
            self.release_post();
        }
    }

    /// The pre-upscaler path submitted a hold whose work `fence` covers.
    pub(crate) fn pre_submitted(&self, device: &ash::Device, fence: vk::Fence) {
        *self.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(InFlight::of(device, fence));
    }

    /// The pre-upscaler path's work has completed (its fence was waited on), and the fence may be
    /// reset or destroyed from here on.
    pub(crate) fn pre_idle(&self) {
        *self.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// Whether a pre-upscaler hold's work may still use the network.
    fn pre_running(&self) -> bool {
        let mut inflight = self.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if inflight.is_some_and(|f| !f.running()) {
            *inflight = None;
        }
        inflight.is_some()
    }

    /// The pre-upscaler path may submit a frame now (the after-the-upscaler path is not running one and
    /// has not for a second). Notes the hold either way, which keeps the other path off.
    pub(crate) fn claim_pre(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        self.last_pre.store(epoch_ms(), SeqCst);
        self.post_finished();
        !self.post_busy.load(SeqCst) && epoch_ms().saturating_sub(self.last_post.load(SeqCst)) >= PATH_SWITCH_MS
    }

    /// Keeps the after-the-upscaler path off the network without claiming it: the hold inside DLSS's
    /// buffer says it wants the network before it waits for the device's state, which the
    /// after-the-upscaler path may be holding while its own frames run.
    pub(crate) fn note_pre(&self) {
        self.last_pre.store(epoch_ms(), std::sync::atomic::Ordering::SeqCst);
    }

    /// Starts the loading thread the first time.
    fn spawn(&self) {
        let Some((device, setup)) = self.pending.lock().unwrap().take() else { return };
        let shared = self.shared.clone();
        let worker = std::thread::Builder::new().name("nf-native-load".into()).spawn(move || Self::work(&shared, device, setup)).ok();
        if worker.is_none() {
            crate::log!("[native] could not start the loader thread; frames go to DLSS untouched");
            note_failure("the loader thread could not be started");
            self.shared.0.lock().unwrap().failed = Some("the loader thread could not be started".into());
        }
        *self.worker.lock().unwrap() = worker;
    }

    fn open(device: vk::Device, setup: Setup) -> Result<nn::Network, nn::OpenError> {
        let dir = model_dir();
        if !model_present() {
            return Err(nn::OpenError { message: format!("no model at {}: extract it in the Setup tab (neural-forge-cli extract-model)", dir.display()), stalled: false });
        }
        let open = nn::OpenInfo {
            gipa: setup.gipa,
            instance: setup.instance,
            physical: setup.physical,
            device,
            queue_family: setup.family,
            queue_index: setup.index,
            frame_family: setup.frame_family,
            model_dir: &dir,
            chain: chain_wanted(),
            fence_timeout_ms: crate::FENCE_WAIT_TIMEOUT.as_millis() as u32,
            init_dispatchable: Some(init_dispatchable),
        };
        let t = Instant::now();
        // SAFETY: the device was created with the network's additions, and the queue is the loader's
        // alone (`crate::take_native`).
        let network = unsafe { nn::Network::open(&open) }
            .map_err(|why| nn::OpenError { message: format!("the model at {} could not be loaded: {}", dir.display(), why.message), stalled: why.stalled })?;
        crate::log!("[native] model loaded and verified from {} in {} ms (chaining {})", dir.display(), t.elapsed().as_millis(), if chain_wanted() { "on" } else { "off" });
        crate::log!("[native] model source: {}", model_source(&std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_default()));
        Ok(network)
    }

    fn work(shared: &(Mutex<LoaderState>, Condvar), device: vk::Device, setup: Setup) {
        let (lock, wake) = shared;
        // This device's VRAM readings for the build lines (it outlives this thread: `Drop` joins it).
        let vram = crate::vram::probe(setup.gipa, setup.instance, setup.physical, setup.properties2);
        let mut retry = BuildRetry::default();
        let mut inject = neural_forge_protocol::env::var(FailInject::ENV).and_then(|v| FailInject::parse(&v));
        if let Some(f) = inject {
            crate::log!("[native] {}: the next {} network builds will fail on purpose", FailInject::ENV, f.remaining());
        }
        let mut state = lock.lock().unwrap();
        loop {
            // Work is wanted when the network is not open yet, a size is wanted that is not built, or
            // the graph is to fall back to barriers; it is due when the retry schedule says so.
            let wanted = |s: &LoaderState| !s.dead && (!s.opened || s.fall_back || (s.wanted.is_some() && s.wanted != s.built.map(|b| (b.width, b.height))));
            if state.quit {
                return;
            }
            if !wanted(&state) {
                state = wake.wait(state).unwrap();
                continue;
            }
            let step = retry.step(Instant::now());
            if let Step::Wait(d) = step {
                state = wake.wait_timeout(state, d).unwrap().0;
                continue;
            }
            let mut network = state.network.take();
            let fall_back = std::mem::take(&mut state.fall_back);
            let size = state.wanted.or(state.built.map(|b| (b.width, b.height)));
            state.built = None;
            drop(state);
            if step == Step::ReinitThenBuild && network.take().is_some() {
                crate::log!("[native] {} failed attempts in a row: closing the network and loading it again", retry.streak());
            }
            let t = Instant::now();
            let vram_before = vram.as_ref().map(crate::vram::reading);
            let mut open_stalled = false;
            let mut switched = false;
            let result = match network.take() {
                Some(n) => Ok(n),
                None => Self::open(device, setup).map_err(|e| {
                    open_stalled = e.stalled;
                    e.message
                }),
            }
            .and_then(|mut n| {
                // The switch to barriers drops the chained graph; the build below records the new one
                // once, for the size wanted now (never a build at the old size first).
                if fall_back {
                    // SAFETY: nothing of the previous build is pending (`fall_back` is called after the
                    // frames that used it were drained).
                    unsafe { n.fall_back_to_barriers() };
                    switched = true;
                }
                match size {
                    None => Ok((n, None)),
                    Some(_) if inject.as_mut().is_some_and(FailInject::should_fail) => {
                        network = Some(n);
                        Err(format!("{} asked this build to fail", FailInject::ENV))
                    }
                    Some((w, h)) => {
                        // SAFETY: nothing of the previous build is pending: a wanted size or a fallback is
                        // only asked for once the holds stopped using it (`ready` returns another size's
                        // build to no hold; `fall_back` is called after the frames that used it were
                        // drained; neither path takes the network while the other's work may run).
                        match unsafe { n.build(w, h) } {
                            Ok(frame) => Ok((n, Some((w, h, frame)))),
                            Err(why) => {
                                network = Some(n);
                                Err(format!("building the network for {w}x{h} failed: {why}"))
                            }
                        }
                    }
                }
            });
            let vram_after = vram.as_ref().filter(|_| matches!(result, Ok((_, Some(_))))).map(crate::vram::reading);
            let stalled = result.is_err() && (open_stalled || network.as_ref().is_some_and(nn::Network::stalled));
            // The state log, written before the loader's lock is taken again: holds take it on the game's thread.
            match &result {
                Ok((_, Some(_))) => note_recovered(),
                Ok(_) => {}
                Err(why) if stalled => note_failure(&format!("{why} (the network's queue stopped responding; off until the game restarts)")),
                Err(why) => note_failure(why),
            }
            // The log lines too: written once the lock is let go again.
            let mut lines = Vec::new();
            state = lock.lock().unwrap();
            state.opened = true;
            match result {
                Ok((n, built)) => {
                    if let Some((w, h, frame)) = built {
                        state.generation += 1;
                        state.built = Some(Built { width: w, height: h, frame, graph: n.graph_commands(), graph_own: n.graph_commands_own(), generation: state.generation });
                        lines.push(format!(
                            "[native] network {}built for {w}x{h} (field {}x{}, chained {}) in {} ms{}",
                            if fall_back { "re" } else { "" },
                            frame.field_width,
                            frame.field_height,
                            frame.chained != 0,
                            t.elapsed().as_millis(),
                            match (vram_before, vram_after) {
                                (Some((before, _)), Some((after, budget))) => format!(" (vram {before} -> {after} of {budget} MiB)"),
                                _ => String::new(),
                            }
                        ));
                        retry.succeeded();
                        if let Some(streak) = retry.take_recovered() {
                            lines.push(format!("[native] the network is up again after {streak} failed attempt(s)"));
                        }
                        state.failed = None;
                    }
                    state.network = Some(n);
                }
                Err(why) if stalled => {
                    // Its work may still run on the GPU: nothing of it is destroyed (dropping it leaks what
                    // it made) and nothing is tried again on this device.
                    lines.push(format!("[native] {why}; a wait on the network's queue timed out, so the network is off on this device until the game restarts; frames go to DLSS untouched"));
                    state.failed = Some(format!("{why} (the network's queue stopped responding; off until the game restarts)"));
                    state.dead = true;
                    drop(network);
                }
                Err(why) => {
                    let wait = retry.failed(Instant::now());
                    lines.push(format!("[native] {why}; frames go to DLSS untouched, trying again in {} ms", wait.as_millis()));
                    state.failed = Some(why);
                    state.network = network;
                    // A failed build leaves no graph; the switch to barriers, once made, stays, and one the
                    // failed open never got to is asked for again.
                    state.fall_back |= fall_back && !switched;
                }
            }
            drop(state);
            for line in &lines {
                crate::log!("{line}");
            }
            crate::logging::flush();
            state = lock.lock().unwrap();
        }
    }

    /// The network built for `width` x `height`, or `None` (and the build is asked for) while it is not.
    pub(crate) fn ready(&self, width: u32, height: u32) -> Option<Built> {
        self.spawn();
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if let Some(b) = state.built.filter(|b| (b.width, b.height) == (width, height)) {
            return Some(b);
        }
        if state.wanted != Some((width, height)) {
            state.wanted = Some((width, height));
            wake.notify_all();
        }
        None
    }

    /// Why the network is not usable right now, if a load or build failed.
    pub(crate) fn failed(&self) -> Option<String> {
        self.shared.0.lock().unwrap().failed.clone()
    }

    /// Counter waits that gave up since the last reset (read once the frames that ran the graph
    /// completed), and where; `None` when there were none (no allocation on that, every hold's, path).
    pub(crate) fn chain_timeouts(&self) -> Option<(u32, String)> {
        let state = self.shared.0.lock().unwrap();
        let network = state.network.as_ref()?;
        (network.chain_timeout_count() > 0).then(|| network.chain_timeouts())
    }

    /// Rebuilds the graph with barriers. The caller has drained every frame that used the current one.
    pub(crate) fn fall_back(&self) {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        state.fall_back = true;
        state.built = None;
        wake.notify_all();
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        {
            let (lock, wake) = &*self.shared;
            lock.lock().unwrap().quit = true;
            wake.notify_all();
        }
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
        // The network (closed here) must not be pending: the device's teardown drained the layer's work.
        self.shared.0.lock().unwrap().network.take();
    }
}

/// The frame's parameters, `native_preprocess.comp`/`native_composite.comp`'s `Params` (std140, field
/// for field, append only).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Params {
    dims: [u32; 4],
    frame: [u32; 4],
    motion: [f32; 4],
    control: [f32; 4],
    more: [f32; 4],
}

/// The network's conditioning (the Model tab's settings that reach its input lanes).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conditioning {
    pub style: u32,
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// Below 0: follows `local_structure`.
    pub skin_structure: f32,
    pub auto_mask: bool,
}

impl From<neural_forge_protocol::Tuning> for Conditioning {
    /// With the model server's clamps (`ngx.rs::NgxTuning`): the same values reach the network.
    fn from(t: neural_forge_protocol::Tuning) -> Self {
        Self {
            style: t.style,
            intensity: t.intensity.clamp(0.0, 4.0),
            local_tone: t.local_tone.clamp(0.0, 4.0),
            local_structure: t.local_structure.clamp(0.0, 4.0),
            skin_structure: t.skin_structure.clamp(-1.0, 4.0),
            auto_mask: t.auto_mask != 0,
        }
    }
}

impl Default for Conditioning {
    fn default() -> Self {
        Self { style: 0, intensity: 1.0, local_tone: 1.0, local_structure: 1.0, skin_structure: -1.0, auto_mask: true }
    }
}

/// A device-local buffer of the layer's own.
pub(crate) struct OwnBuffer {
    pub(super) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl OwnBuffer {
    /// # Safety
    /// `device` is live.
    pub(super) unsafe fn new(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, bytes: u64, usage: vk::BufferUsageFlags) -> Option<Self> {
        let info = vk::BufferCreateInfo::builder().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
        // SAFETY: just created.
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        // SAFETY: plain query.
        let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let index = (0..props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        let memory = index.and_then(|index| {
            // SAFETY: valid allocation info.
            unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.ok()
        });
        // SAFETY: sized for each other; on failure nothing was submitted.
        match memory.filter(|&m| unsafe { device.bind_buffer_memory(buffer, m, 0) }.is_ok()) {
            Some(memory) => Some(Self { buffer, memory }),
            None => {
                // SAFETY: nothing submitted uses either.
                unsafe {
                    if let Some(m) = memory {
                        device.free_memory(m, None);
                    }
                    device.destroy_buffer(buffer, None);
                }
                None
            }
        }
    }

    /// # Safety
    /// Nothing submitted may still use it.
    pub(super) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

const PARAMS_BYTES: u64 = std::mem::size_of::<Params>() as u64;

/// DLSS's depth, copied by every native hold's capture (`C`) into a device-local buffer of the
/// layer's at the render extent, wherever it can be read: on the split path from the identified
/// depth image (its depth aspect), inside DLSS's buffer from the slot buffer the hold recorded there
/// (`inline`). Kept raw, tagged with the game's format: 4 bytes a texel (D32, D24 with its top byte,
/// depth handed over as R32), or 2 for D16 (the buffer's first half). No conversion pass: what reads
/// it converts (the cache dump writes f32, `cache_dump::depth_to_f32`).
pub(crate) struct DepthCopy {
    buffer: OwnBuffer,
    /// What the last hold copied: the game's depth format, or `None` when that hold copied none (not
    /// readable, a barrier on it in front of the launch, no depth inside DLSS's buffer).
    pub(crate) valid: Option<vk::Format>,
    /// GPU timestamps around the copy alone (the first command of `C`).
    pub(crate) timer: Option<crate::gpu_timer::GpuTimer>,
}

impl DepthCopy {
    /// # Safety
    /// `device` is live; `family` is the queue family the holds are submitted on.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, width: u32, height: u32) -> Option<Self> {
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
        // SAFETY: forwarded.
        let buffer = unsafe { OwnBuffer::new(device, instance, physical_device, u64::from(width) * u64::from(height) * 4, usage) }?;
        Some(Self { buffer, valid: None, timer: crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family) })
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    /// The copy's GPU time on the last hold that copied, once its fence was waited on.
    pub(crate) fn take_gpu_ms(&mut self, device: &ash::Device) -> Option<f32> {
        let timer = self.timer.as_mut()?;
        timer.read_if_pending(device);
        timer.take_reading()
    }

    /// # Safety
    /// Nothing submitted may still use it.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            self.buffer.destroy(device);
            if let Some(t) = &self.timer {
                t.destroy(device);
            }
        }
    }
}

/// Why a hold copies no depth ([`depth_source`] is `None`), for a cache dump's manifest.
pub(crate) fn depth_missing(target: &Target) -> &'static str {
    match target.depth {
        None => "no depth for this hold (a barrier on it before the launch, or none copied inside DLSS's buffer)",
        Some(a) if !a.readable => "the depth image cannot be copied (no TRANSFER_SRC, or another extent)",
        Some(a) if a.layout.is_none() => "the depth image's layout is not known",
        Some(_) => "the depth image is in a layout a copy cannot read (UNDEFINED)",
    }
}

/// What the capture copies into [`DepthCopy`] this hold: the slot buffer's copy inside DLSS's
/// buffer, else the depth image where it is readable in a layout a copy can read from (the device
/// leaves it out when a barrier on it comes before the launch).
pub(crate) fn depth_source(target: &Target) -> Option<vk::Format> {
    if let Some((_, format)) = target.depth_buffer {
        return Some(format);
    }
    target.depth.filter(|a| a.readable && super::aux_copy_layout(a.layout).is_some()).map(|a| a.format)
}

/// The layer's side of the native frame for one extent: the two frame shaders, the history images,
/// DLSS's motion vectors' copy, the answer buffer, and the two recorded `N` buffers.
pub(crate) struct NativePass {
    width: u32,
    height: u32,
    padded_width: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    preprocess: vk::Pipeline,
    composite: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: [vk::DescriptorSet; 2],
    /// `history[p]` is read by parity `p`'s frame and `history[1 - p]` written.
    history: [OwnImage; 2],
    /// DLSS's motion vectors as the capture copies them (packed half2).
    pub(crate) mvec: OwnBuffer,
    /// The motion back to the previous network frame the frame shaders read (binding 2), f32 x and y
    /// per texel: `preupscale_cache_motion.comp`'s accumulated field (`super::cache`).
    pub(crate) acc: OwnBuffer,
    /// The edit cache is off: binding 2 is `mvec` and the frame shaders read it as packed halves
    /// (`MOTION_HALF`), as they did before the accumulated field (3.2.0).
    motion_half: bool,
    answer: OwnBuffer,
    state: OwnBuffer,
    params: [HostBuffer; 2],
    command_pool: vk::CommandPool,
    /// `N[parity]`, then the one-time initialisation (history layouts, state cleared).
    commands: [vk::CommandBuffer; 3],
    /// GPU timestamps around `N[parity]` (`None` where timestamps are not supported).
    timers: [Option<crate::gpu_timer::GpuTimer>; 2],
    /// The parity of the last frame submitted, whose timer is read at the next hold.
    last_parity: Option<usize>,
    /// The [`Built::generation`] `commands` were recorded for, and whether the initialisation ran.
    recorded: Option<u64>,
    initialised: bool,
    /// The frame counter of the holds that ran the network, its seed since the last history reset,
    /// and the last frame's jitter.
    frames: u64,
    seed: u32,
    last_jitter: Option<[f32; 2]>,
    pub(crate) gap: HistoryGap<(u32, u32)>,
}

// SAFETY: plain handles and the layer's own mappings, only used behind the device's `State` mutex.
unsafe impl Send for NativePass {}

impl NativePass {
    /// With `motion_half` the frame shaders read DLSS's packed vectors (`MOTION_HALF`), as 3.2.0's
    /// did: the edit cache off (`NEURAL_FORGE_EDIT_CACHE=off`). Without, the accumulated field.
    ///
    /// # Safety
    /// `device` is live; `family` is the queue family the frames are submitted on.
    pub(crate) unsafe fn build(
        device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, width: u32, height: u32, motion_half: bool,
    ) -> Option<Self> {
        // SAFETY: forwarded.
        unsafe { Self::build_failing_at(device, instance, physical_device, family, width, height, None, motion_half) }
    }

    /// [`Self::build`], failing on purpose at creation step `fail_at` (`0..BUILD_STEPS`) when given.
    /// Built piece by piece; [`PassPartial`] destroys whatever exists on an early return.
    ///
    /// # Safety
    /// As [`Self::build`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn build_failing_at(
        device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, width: u32, height: u32, fail_at: Option<usize>,
        motion_half: bool,
    ) -> Option<Self> {
        let (padded_width, padded_height) = super::padded(width, height);
        let binding = |n: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build()
        };
        let bindings = [
            binding(0, vk::DescriptorType::STORAGE_IMAGE),
            binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
            binding(3, vk::DescriptorType::STORAGE_BUFFER),
            binding(4, vk::DescriptorType::UNIFORM_BUFFER),
            binding(5, vk::DescriptorType::STORAGE_BUFFER),
            binding(6, vk::DescriptorType::STORAGE_BUFFER),
            binding(7, vk::DescriptorType::STORAGE_IMAGE),
            binding(8, vk::DescriptorType::STORAGE_BUFFER),
            binding(9, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let mut steps = 0usize;
        // `Some` unless this creation step is the one to fail.
        let mut step = || {
            let n = steps;
            steps += 1;
            (fail_at != Some(n)).then_some(())
        };
        let mut p = PassPartial::new(device);
        // SAFETY: valid create infos throughout; every object created here is owned by `p` until the
        // end, where it moves into the result.
        unsafe {
            step()?;
            p.set_layout = made(device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?);
            step()?;
            p.pipeline_layout = made(device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&p.set_layout)), None).ok()?);
            step()?;
            // MOTION_HALF (constant_id 0), a VkBool32.
            let half = u32::from(motion_half).to_le_bytes();
            let entry = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
            let spec = vk::SpecializationInfo::builder().map_entries(&entry).data(&half).build();
            p.preprocess = made(hdr::compute_pipeline_with(device, p.pipeline_layout, PREPROCESS_SPV, Some(&spec))?);
            step()?;
            p.composite = made(hdr::compute_pipeline_with(device, p.pipeline_layout, COMPOSITE_SPV, Some(&spec))?);
            step()?;
            p.sampler = made(
                device
                    .create_sampler(
                        &vk::SamplerCreateInfo::builder()
                            .mag_filter(vk::Filter::LINEAR)
                            .min_filter(vk::Filter::LINEAR)
                            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                        None,
                    )
                    .ok()?,
            );
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 4 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 12 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 },
            ];
            step()?;
            p.pool = made(device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&sizes), None).ok()?);
            let layouts = [p.set_layout, p.set_layout];
            step()?;
            // Freed with the pool.
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(p.pool).set_layouts(&layouts)).ok()?;
            for _ in 0..2 {
                step()?;
                p.history.push(made(OwnImage::new(device, instance, physical_device, width, height, vk::ImageUsageFlags::SAMPLED)?));
            }
            let texels = u64::from(width) * u64::from(height);
            step()?;
            // TRANSFER_SRC: a cache dump reads it back (`cache_dump`).
            p.mvec = Some(made(OwnBuffer::new(
                device,
                instance,
                physical_device,
                texels * 4,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?));
            step()?;
            p.acc = Some(made(OwnBuffer::new(
                device,
                instance,
                physical_device,
                texels * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?));
            let answer_bytes = u64::from(padded_width) * u64::from(padded_height) * 8;
            step()?;
            p.answer = Some(made(OwnBuffer::new(device, instance, physical_device, answer_bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER)?));
            step()?;
            p.state = Some(made(OwnBuffer::new(device, instance, physical_device, 16, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?));
            for _ in 0..2 {
                step()?;
                p.params.push(made(own_host_buffer_with(device, instance, physical_device, PARAMS_BYTES, vk::BufferUsageFlags::UNIFORM_BUFFER)?));
            }
            step()?;
            p.command_pool = made(
                device
                    .create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                    .ok()?,
            );
            let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(p.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(3);
            step()?;
            // Freed with the command pool.
            let cmds = crate::loader_data::allocate_commands(device, &alloc).ok()?;
            debug_assert_eq!(steps, BUILD_STEPS);
            // Optional (`None` where timestamps are not supported): nothing fails from here on.
            let timers = [
                crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family).map(made),
                crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family).map(made),
            ];
            let take_buffer = |b: &mut Option<OwnBuffer>| b.take().expect("built above");
            let (mvec, acc, answer, state) = (take_buffer(&mut p.mvec), take_buffer(&mut p.acc), take_buffer(&mut p.answer), take_buffer(&mut p.state));
            let mut history = std::mem::take(&mut p.history).into_iter();
            let history = [history.next().expect("built above"), history.next().expect("built above")];
            let mut params = std::mem::take(&mut p.params).into_iter();
            let params = [params.next().expect("built above"), params.next().expect("built above")];
            Some(Self {
                width,
                height,
                padded_width,
                set_layout: std::mem::take(&mut p.set_layout),
                pipeline_layout: std::mem::take(&mut p.pipeline_layout),
                preprocess: std::mem::take(&mut p.preprocess),
                composite: std::mem::take(&mut p.composite),
                sampler: std::mem::take(&mut p.sampler),
                pool: std::mem::take(&mut p.pool),
                sets: [sets[0], sets[1]],
                history,
                mvec,
                acc,
                motion_half,
                answer,
                state,
                params,
                command_pool: std::mem::take(&mut p.command_pool),
                commands: [cmds[0], cmds[1], cmds[2]],
                timers,
                last_parity: None,
                recorded: None,
                initialised: false,
                frames: 0,
                seed: 0,
                last_jitter: None,
                gap: HistoryGap::default(),
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    pub(crate) fn answer(&self) -> vk::Buffer {
        self.answer.buffer
    }

    /// Points both sets at this frame's resources and records `N[0]`, `N[1]` and the initialisation
    /// for `built`, unless they already are. Nothing of them may be pending.
    ///
    /// # Safety
    /// `device` is live; `encoded` and `exposure` are the HDR pass's (live for as long as this pass).
    /// `graph` is `built`'s graph recorded for this pass's queue family.
    pub(crate) unsafe fn record(&mut self, device: &ash::Device, built: &Built, graph: vk::CommandBuffer, encoded: vk::ImageView, exposure: vk::Buffer) -> bool {
        if self.recorded == Some(built.generation) {
            return true;
        }
        let f = built.frame;
        for p in 0..2 {
            let set = self.sets[p];
            let general = |view| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
            let enc = general(encoded);
            let next = general(self.history[1 - p].view);
            let prev = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: self.history[p].view, image_layout: vk::ImageLayout::GENERAL }];
            let buf = |buffer, range| [vk::DescriptorBufferInfo { buffer, offset: 0, range }];
            let (mv, feat, head, params, state, expo, ans) = (
                buf(if self.motion_half { self.mvec.buffer } else { self.acc.buffer }, vk::WHOLE_SIZE),
                buf(f.features, vk::WHOLE_SIZE),
                buf(f.head, vk::WHOLE_SIZE),
                buf(self.params[p].buffer, PARAMS_BYTES),
                buf(self.state.buffer, vk::WHOLE_SIZE),
                buf(exposure, 4),
                buf(self.answer.buffer, vk::WHOLE_SIZE),
            );
            let w = |binding, ty| vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(binding).descriptor_type(ty);
            let writes = [
                w(0, vk::DescriptorType::STORAGE_IMAGE).image_info(&enc).build(),
                w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&prev).build(),
                w(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&mv).build(),
                w(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&feat).build(),
                w(4, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&params).build(),
                w(5, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&state).build(),
                w(6, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&expo).build(),
                w(7, vk::DescriptorType::STORAGE_IMAGE).image_info(&next).build(),
                w(8, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ans).build(),
                w(9, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&head).build(),
            ];
            // SAFETY: the sets are not in use (nothing of this pass is pending); every handle is live.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let ok = (0..2).all(|p| {
            // SAFETY: forwarded from this function's contract.
            unsafe { self.record_frame(device, p, built, graph) }.is_ok()
        });
        // SAFETY: as above.
        let ok = ok && unsafe { self.record_init(device) }.is_ok();
        self.recorded = ok.then_some(built.generation);
        ok
    }

    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_frame(&self, device: &ash::Device, p: usize, built: &Built, graph: vk::CommandBuffer) -> ash::prelude::VkResult<()> {
        let cmd = self.commands[p];
        let f = built.frame;
        let compute = |src: vk::AccessFlags, dst: vk::AccessFlags| vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(dst).build();
        // SAFETY: the layer's own buffer, not pending (forwarded); every handle is live.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))?;
            if let Some(timer) = &self.timers[p] {
                timer.record_start(device, cmd);
            }
            // `C`'s encode and motion-vector copy, and the previous frame's history write, before
            // anything here reads them.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::UNIFORM_READ)],
                &[],
                &[],
            );
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[p]], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.preprocess);
            device.cmd_dispatch(cmd, f.field_width.div_ceil(8), f.field_height.div_ceil(8), 1);
            // The features to the network (the graph opens with compute barriers of its own).
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            // `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`: everything but the network's launches (diagnostic: the
            // answer is then the head left from before).
            if !skip_graph() {
                device.cmd_execute_commands(cmd, &[graph]);
            }
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            // The graph's secondary may have bound its own pipeline layout and sets: bind again.
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[p]], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.composite);
            device.cmd_dispatch(cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);
            // The answer to `W`'s copy, the history to the next frame's samplers.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
                &[],
                &[],
            );
            if let Some(timer) = &self.timers[p] {
                timer.record_end(device, cmd);
            }
            device.end_command_buffer(cmd)
        }
    }

    /// The previous frame's network GPU time, once its work has completed (the caller waited for it).
    pub(crate) fn take_gpu_ms(&mut self, device: &ash::Device) -> Option<f32> {
        let timer = self.timers[self.last_parity.take()?].as_mut()?;
        timer.read_if_pending(device);
        timer.take_reading()
    }

    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_init(&self, device: &ash::Device) -> ash::prelude::VkResult<()> {
        let cmd = self.commands[2];
        let to_general = |image| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                .build()
        };
        // SAFETY: as `record_frame`.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;
            device.cmd_fill_buffer(cmd, self.state.buffer, 0, vk::WHOLE_SIZE, 0);
            let filled = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).build();
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[filled],
                &[],
                &[to_general(self.history[0].image), to_general(self.history[1].image)],
            );
            device.end_command_buffer(cmd)
        }
    }

    /// The command buffers for this frame, in submission order: the initialisation until one was
    /// submitted, then `N[parity]`. Writes the frame's parameters. Each buffer submitted is reported
    /// to [`Self::submitted`] (only then is the initialisation done and the timer pending).
    /// `mvec_scale` turns the copied motion vectors into render pixels ([`super::Target::mvec_scale`]).
    pub(crate) fn frame(&mut self, built: &Built, history: bool, jitter: Option<[f32; 2]>, mvec_scale: [f32; 2], c: Conditioning) -> [Option<vk::CommandBuffer>; 2] {
        let p = (self.frames & 1) as usize;
        self.seed = if history { self.seed.wrapping_add(1) } else { 0 };
        let delta = match (self.last_jitter, jitter) {
            (Some(prev), Some(cur)) if history => [prev[0] - cur[0], prev[1] - cur[1]],
            _ => [0.0, 0.0],
        };
        self.last_jitter = jitter;
        let f = built.frame;
        let params = Params {
            dims: [f.field_width, f.field_height, self.width, self.height],
            frame: [self.padded_width, self.seed, u32::from(history), p as u32],
            motion: [delta[0], delta[1], f.blend_scale, c.intensity],
            control: [c.style as f32, c.local_tone, c.local_structure, c.skin_structure],
            more: [if c.auto_mask { 1.0 } else { 0.0 }, mvec_scale[0], mvec_scale[1], 0.0],
        };
        // SAFETY: host-coherent and mapped for PARAMS_BYTES; parity `p`'s previous use (two frames
        // ago) completed: the hold waits for the previous hold's work before it records anything.
        unsafe { std::ptr::copy_nonoverlapping(std::ptr::from_ref(&params).cast::<u8>(), self.params[p].ptr, PARAMS_BYTES as usize) };
        self.frames += 1;
        [(!self.initialised).then_some(self.commands[2]), Some(self.commands[p])]
    }

    /// `cmd`, one of [`Self::frame`]'s, was submitted: the initialisation is done, or `N[parity]`'s
    /// timer is pending (read at the next hold).
    pub(crate) fn submitted(&mut self, cmd: vk::CommandBuffer) {
        if cmd == self.commands[2] {
            self.initialised = true;
        } else if let Some(p) = (0..2).find(|&p| self.commands[p] == cmd) {
            self.last_parity = Some(p);
            if let Some(timer) = self.timers[p].as_mut() {
                timer.mark_submitted();
            }
        }
    }

    /// # Safety
    /// Nothing of this pass may still be pending.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; freeing the pools frees their sets and buffers.
        unsafe {
            for t in self.timers.iter().flatten() {
                t.destroy(device);
                destroyed(1);
            }
            destroyed(PASS_OBJECTS);
            device.destroy_command_pool(self.command_pool, None);
            for h in &self.history {
                h.destroy(device);
            }
            self.mvec.destroy(device);
            self.acc.destroy(device);
            self.answer.destroy(device);
            self.state.destroy(device);
            for b in &self.params {
                b.destroy(device);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.preprocess, None);
            device.destroy_pipeline(self.composite, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The creation steps of [`NativePass::build`] that can fail, in order.
const BUILD_STEPS: usize = 17;
/// The objects [`NativePass::destroy`] destroys besides its timers.
const PASS_OBJECTS: isize = 15;

#[cfg(test)]
thread_local! {
    /// Objects of the native pass created minus destroyed on this thread (the tests' leak check).
    static LIVE: std::cell::Cell<isize> = const { std::cell::Cell::new(0) };
}

/// Notes one object created (counted in tests only).
fn made<T>(object: T) -> T {
    #[cfg(test)]
    LIVE.with(|l| l.set(l.get() + 1));
    object
}

/// Notes `n` objects destroyed (counted in tests only).
fn destroyed(n: isize) {
    #[cfg(test)]
    LIVE.with(|l| l.set(l.get() - n));
    #[cfg(not(test))]
    let _ = n;
}

/// [`NativePass::build`]'s objects so far; destroys them unless moved out.
struct PassPartial<'a> {
    device: &'a ash::Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    preprocess: vk::Pipeline,
    composite: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    history: Vec<OwnImage>,
    mvec: Option<OwnBuffer>,
    acc: Option<OwnBuffer>,
    answer: Option<OwnBuffer>,
    state: Option<OwnBuffer>,
    params: Vec<HostBuffer>,
    command_pool: vk::CommandPool,
}

impl<'a> PassPartial<'a> {
    fn new(device: &'a ash::Device) -> Self {
        Self {
            device,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            preprocess: vk::Pipeline::null(),
            composite: vk::Pipeline::null(),
            sampler: vk::Sampler::null(),
            pool: vk::DescriptorPool::null(),
            history: Vec::new(),
            mvec: None,
            acc: None,
            answer: None,
            state: None,
            params: Vec::new(),
            command_pool: vk::CommandPool::null(),
        }
    }
}

impl Drop for PassPartial<'_> {
    fn drop(&mut self) {
        let d = self.device;
        let handles = [
            self.command_pool.as_raw(),
            self.pool.as_raw(),
            self.sampler.as_raw(),
            self.preprocess.as_raw(),
            self.composite.as_raw(),
            self.pipeline_layout.as_raw(),
            self.set_layout.as_raw(),
        ];
        let objects = handles.iter().filter(|&&h| h != 0).count() + self.history.len() + self.params.len();
        let buffers = [self.mvec.take(), self.acc.take(), self.answer.take(), self.state.take()];
        destroyed((objects + buffers.iter().flatten().count()) as isize);
        // SAFETY: nothing here was ever submitted; null handles are ignored by the destroy calls;
        // freeing the pools frees the sets and command buffers allocated from them.
        unsafe {
            d.destroy_command_pool(self.command_pool, None);
            for h in &self.history {
                h.destroy(d);
            }
            for b in buffers.iter().flatten() {
                b.destroy(d);
            }
            for b in &self.params {
                b.destroy(d);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_sampler(self.sampler, None);
            d.destroy_pipeline(self.preprocess, None);
            d.destroy_pipeline(self.composite, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// When a hold builds a pass again after building it failed: at once for another extent, else on the
/// network loader's schedule ([`BuildRetry`]: 0.5, 1 and 2 s, then every 30 s), never on every hold.
#[derive(Debug, Default)]
pub(crate) struct PassRetry {
    failed: Option<(u32, u32)>,
    retry: BuildRetry,
}

impl PassRetry {
    /// The result of `build` for `extent`, or `None` without calling it while a failed build of the
    /// same extent is not due again.
    pub(crate) fn build<T>(&mut self, extent: (u32, u32), now: Instant, build: impl FnOnce() -> Option<T>) -> Option<T> {
        if self.failed == Some(extent) && matches!(self.retry.step(now), Step::Wait(_)) {
            return None;
        }
        let built = build();
        if built.is_some() {
            self.failed = None;
            self.retry.succeeded();
        } else {
            self.failed = Some(extent);
            self.retry.failed(now);
        }
        built
    }
}

/// The retry state of a device's native hold: the HDR pass, its auto-exposure, the native pass.
#[derive(Debug, Default)]
pub(crate) struct Retries {
    pub(crate) hdr: PassRetry,
    pub(crate) auto: PassRetry,
    pub(crate) pass: PassRetry,
    pub(crate) depth: PassRetry,
    pub(crate) edit: PassRetry,
}

/// `NEURAL_FORGE_NATIVE_FAIL=chain`: reports one counter-chain timeout at the 300th network frame, to
/// exercise the fallback to barriers on the rig. Unset (the default) does nothing.
fn fake_chain_timeout(frames: u64) -> Option<(u32, String)> {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_FAIL").as_deref() == Some("chain"));
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    (*ON && frames >= 300 && !DONE.swap(true, std::sync::atomic::Ordering::Relaxed)).then(|| (1, "NEURAL_FORGE_NATIVE_FAIL=chain".to_string()))
}

/// What the Status tab says about the native backend after a hold: running, or why not.
pub(crate) fn status(result: &super::HoldResult, loader: &Loader) -> String {
    if result.evaluated {
        return "native network running".to_string();
    }
    match (loader.failed(), result.miss) {
        (Some(why), _) => format!("native network not running: {why}"),
        (None, Some(miss)) => format!("native network: {miss}"),
        (None, None) => String::new(),
    }
}

/// The last network failure written to the state log, so a retry schedule that keeps failing the same
/// way writes it once; `None` once the network was built again.
static LAST_FAILURE: Mutex<Option<String>> = Mutex::new(None);

/// A load or build failure for the state log: each different reason once.
fn note_failure(why: &str) {
    if failure_is_new(&mut LAST_FAILURE.lock().unwrap(), why) {
        crate::event!(neural_forge_protocol::state_log::NATIVE_FAILED, "{why}");
    }
}

/// A successful build: written to the state log only when it ends a failure.
fn note_recovered() {
    if LAST_FAILURE.lock().unwrap().take().is_some() {
        crate::event!(neural_forge_protocol::state_log::NATIVE_RECOVERED, "the network is built again");
    }
}

fn failure_is_new(last: &mut Option<String>, why: &str) -> bool {
    if last.as_deref() == Some(why) {
        return false;
    }
    *last = Some(why.to_string());
    true
}

/// Logs a history reset's reason (the model server logs the same reasons for NGX's).
pub(crate) fn note_reset(stale: Option<Stale>) {
    match stale {
        Some(Stale::Skipped) => crate::log!("[native] resetting the history (frames went to DLSS untouched since the last one)"),
        Some(Stale::Idle(gap)) => crate::log!("[native] resetting the history ({} ms since the last frame)", gap.as_millis()),
        Some(Stale::FormatChanged) => crate::log!("[native] resetting the history (another extent or identification)"),
        None => {}
    }
}

/// Every hold's view of the target, for [`HistoryGap`]: the input's size. Not its identification: DLSS's
/// input alternates between two colour images every frame in Black Myth: Wukong with frame generation on,
/// and each switch reset the history.
pub(crate) fn format_of(target: &Target) -> (u32, u32) {
    (target.width, target.height)
}


/// The motion vectors the capture copies into the native pass this frame, by the capture's own rule
/// (`super::capture_reads`): readable, in a layout a copy can read from. The history is only enabled
/// with them; one in `UNDEFINED` is not copied, and the history would reproject with stale vectors.
fn copied_mvec(mvec: Option<super::Aux>) -> Option<super::Aux> {
    mvec.filter(|a| a.readable && super::aux_copy_layout(a.layout).is_some())
}

/// What the edit cache needs from the hold's caller (`super::run_native`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheAsk {
    /// `Session::seen`: which DLSS submit this hold is (consecutive holds differ by 1).
    pub hold: u64,
    /// The cache's interval while the camera moves (`super::cache::interval`: 2, or 1 with
    /// `NEURAL_FORGE_EDIT_CACHE=off`).
    pub interval: u32,
    /// `debug_view` asks for the cache's rejection mask.
    pub debug_mask: bool,
}

/// One native hold (module docs): the capture with the motion-vector copy, the network, the write-back,
/// submitted on the game's queue through `submit` with no wait in between. The previous hold's work
/// is waited for (bounded) first, as every hold does. Frames the network is not ready for go to DLSS
/// untouched, with `miss` saying why.
///
/// # Safety
/// As `super::run_hold`; `loader` belongs to `device`, and `res` was built for the queue family the
/// loader's network was opened for.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_native_hold(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, res: &mut super::Resources, target: &Target,
    loader: &Loader, jitter: Option<[f32; 2]>, chain_ok: bool, conditioning: Conditioning, arm: Option<super::cache_dump::Arm>, ask: CacheAsk,
    submit: &mut dyn FnMut(super::Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>,
) -> super::HoldResult {
    let mut result = super::HoldResult { native: true, ..Default::default() };
    let t_hold = Instant::now();
    let mut writeback_gpu = None;
    if !res.wait_idle(device, &mut writeback_gpu) {
        result.miss = Some("the previous hold's work is still pending");
        return result;
    }
    result.writeback_gpu_ms = writeback_gpu;
    result.network_gpu_ms = res.native.as_mut().and_then(|n| n.take_gpu_ms(device));
    result.depth_gpu_ms = res.depth.as_mut().and_then(|d| d.take_gpu_ms(device));
    // The previous hold's edit-cache counters (finished: waited above).
    if let Some(c) = res.edit.as_mut() {
        result.cache_gpu_ms = c.take_gpu_ms(device);
        result.cache_rejected = c.after_fence().rejected;
    }
    // An armed hold's readbacks (finished: waited above), before anything clears the exposure.
    result.cache_frame = res.take_cache_frame();
    if arm.is_none() {
        // SAFETY: idle (waited above).
        unsafe { res.free_cache(device) };
    }
    // The game's exposure the previous hold's capture read (finished: waited above), for the
    // caller's check (`Session::check_exposure`). Only for this identification.
    let previous_read = res.game_exposure_read.take() == Some(target.identification);
    if !loader.claim_pre() {
        result.miss = Some("the network is serving the after-the-upscaler path");
        return result;
    }
    // Every frame that ran the graph has completed (this path's: the hold waited above; the other
    // path's: `claim_pre` says none ran for a second): the chain watchdog's count is final for them.
    let fake = fake_chain_timeout(res.native.as_ref().map_or(0, |n| n.frames));
    if let Some((waits, at)) = loader.chain_timeouts().filter(|(n, _)| *n > 0).or(fake) {
        crate::log!("[native] {waits} counter-chain wait(s) of the network gave up ({at}); rebuilding it with barriers between its launches");
        crate::logging::flush();
        loader.fall_back();
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some("the network's counter chain timed out; it is being rebuilt with barriers");
        return result;
    }
    let Some(built) = loader.ready(target.width, target.height) else {
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some(if loader.failed().is_some() { "the native network could not be loaded or built" } else { "the native network is still loading" });
        return result;
    };
    // Counter chaining faulted the GPU from the hold inside DLSS's buffer (Black Myth: Wukong, 3 runs
    // of 4: Xid 13 then 32 within seconds; none of 2 with barriers): that hold rebuilds the network
    // with barriers between its launches, for the rest of the process (`fall_back_to_barriers`).
    if !chain_ok && built.frame.chained != 0 {
        crate::log!("[native] the hold inside DLSS's buffer runs the network with barriers between its launches, not counter chaining; rebuilding it");
        crate::logging::flush();
        loader.fall_back();
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some("the network is being rebuilt without counter chaining");
        return result;
    }
    // The HDR encode and decode, exactly as the model server hold uses them.
    let source = super::ExposureSource::of(target);
    result.exposure_source = Some(source);
    // A failed build is tried again on the retry schedule or for another extent, not on every hold.
    let now = Instant::now();
    let extent = (res.width, res.height);
    if res.hdr.is_none() {
        // SAFETY: `device` is live.
        res.hdr = res.native_retry.hdr.build(extent, now, || unsafe { hdr::HdrPass::build(device, instance, physical_device, extent.0, extent.1) });
    }
    let Some(pass) = res.hdr.as_mut() else {
        result.miss = Some("the HDR encode/decode pipelines could not be built");
        return result;
    };
    if source == super::ExposureSource::Auto {
        // SAFETY: `device` is live; nothing of the pass is pending.
        if res.native_retry.auto.build(extent, now, || unsafe { pass.ensure_auto(device, instance, physical_device) }.then_some(())).is_none() {
            result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
            return result;
        }
        pass.auto_for(target.identification);
    }
    if previous_read && source == super::ExposureSource::Game {
        result.previous_exposure = Some(pass.exposure_value());
    }
    pass.clear_exposure();
    // SAFETY: nothing of the pass is pending; the colour input is a live RGBA16F storage image.
    if !unsafe { pass.bind_colour(device, target.colour) } {
        result.miss = Some("a storage view of the colour input could not be created");
        return result;
    }
    let (encoded, exposure) = (pass.encoded_view(), pass.exposure_buffer());
    // The graph for the queue the frames go to: the game's graphics family (the split hold) or the
    // network's own compute family (the hold inside DLSS's buffer, on the layer's side queue).
    let setup = loader.setup();
    let graph = if res.queue_family == setup.frame_family {
        built.graph
    } else if res.queue_family == setup.family {
        built.graph_own
    } else {
        result.miss = Some("the hold's queue family is not one the network was recorded for");
        return result;
    };
    // With the edit cache off (`NEURAL_FORGE_EDIT_CACHE=off`) the hold is 3.2.0's: no depth copy, none of
    // the cache's passes, and the frame shaders read DLSS's packed vectors as they did then
    // (`MOTION_HALF`; bit for bit the accumulated field at interval 1, the test
    // `at_interval_1_the_network_sees_exactly_what_it_saw_before_the_accumulated_field`).
    let cache_on = ask.interval > 1;
    // The resources are built for one extent (`Session::ensure` rebuilds them for another), so
    // everything in them is at the colour input's.
    debug_assert_eq!((res.width, res.height), (target.width, target.height));
    debug_assert!(res.native.as_ref().is_none_or(|n| n.matches(target.width, target.height)));
    debug_assert!(res.edit.as_ref().is_none_or(|c| c.matches(target.width, target.height)));
    let mut native = match res.native.take() {
        Some(n) => n,
        None => {
            let family = res.queue_family;
            // SAFETY: `device` is live; `family` is the frames' family.
            let built = res.native_retry.pass.build((target.width, target.height), now, || unsafe {
                NativePass::build(device, instance, physical_device, family, target.width, target.height, !cache_on)
            });
            match built {
                Some(n) => n,
                None => {
                    result.miss = Some("the native frame resources could not be built");
                    return result;
                }
            }
        }
    };
    // SAFETY: nothing of the pass is pending (`wait_idle` above); the HDR pass outlives it in `res`.
    if !unsafe { native.record(device, &built, graph, encoded, exposure) } {
        res.native = Some(native);
        result.miss = Some("recording the native frame failed");
        return result;
    }
    // The history needs DLSS's motion vectors copied this frame.
    let mvec = copied_mvec(target.mvec);
    // DLSS's depth, wherever it can be read, into the layer's own buffer: for the edit cache only.
    let depth = depth_source(target).filter(|_| cache_on);
    if depth.is_some() && res.depth.is_none() {
        let family = res.queue_family;
        // SAFETY: `device` is live; `family` is the holds' family.
        res.depth = res.native_retry.depth.build((target.width, target.height), now, || unsafe {
            DepthCopy::build(device, instance, physical_device, family, target.width, target.height)
        });
    }
    let depth = depth.filter(|_| res.depth.is_some());
    // The edit cache (`super::cache`): its passes for this extent, and its store while the interval
    // is above 1 (at 1 only the motion pass runs, copying the vectors for the network as they are).
    if cache_on && res.edit.is_none() {
        let family = res.queue_family;
        // SAFETY: `device` is live; `family` is the holds' family.
        res.edit = res.native_retry.edit.build((target.width, target.height), now, || unsafe {
            super::cache::CachePass::build(device, instance, physical_device, family, target.width, target.height)
        });
    }
    let Some(colour_view) = res.hdr.as_ref().and_then(hdr::HdrPass::colour_view) else {
        res.native = Some(native);
        result.miss = Some("a storage view of the colour input could not be created");
        return result;
    };
    if cache_on && res.edit.is_none() {
        res.native = Some(native);
        result.miss = Some("the edit cache's passes could not be built");
        return result;
    }
    native.gap.note_format(format_of(target));
    let history_ok = native.frames > 0 && native.gap.pending(Instant::now()).is_none();
    // The cache's plan for this hold (a refresh at interval 1 with the cache off).
    let mut cache_hold = None;
    let mut plan = None;
    let mut push = None;
    let mut chained = false;
    if let Some(edit) = res.edit.as_mut().filter(|_| cache_on) {
        // Above interval 1 the cache's buffers are in the network's input features (`cache::ScratchLayout`).
        chained = edit.chain(built.frame.features_bytes);
        if !chained {
            edit.unchain();
        }
        let h = super::cache::Hold {
            interval: if chained { ask.interval } else { 1 },
            hold: ask.hold,
            jitter,
            mvec: mvec.is_some(),
            depth: depth.is_some(),
            history_ok,
            armed: arm.is_some(),
            debug_mask: ask.debug_mask,
            generation: built.generation,
        };
        let p = edit.state.plan(&h);
        let hdr_ref = res.hdr.as_ref().expect("built above");
        let bindings = super::cache::Bindings {
            colour: colour_view,
            encoded: hdr_ref.encoded_view(),
            answer: hdr_ref.answer_view(),
            exposure: hdr_ref.exposure_buffer(),
            depth: depth.and(res.depth.as_ref()).map(DepthCopy::buffer),
            mvec: native.mvec.buffer,
            acc: native.acc.buffer,
            scratch: (built.frame.features, built.frame.features_bytes),
        };
        // SAFETY: nothing of the pass is pending; every handle is live.
        unsafe { edit.bind(device, &bindings) };
        edit.clear_counters();
        // No vectors copied this hold: the motion pass writes no history and counts nothing.
        let flags = p.flags | if mvec.is_none() { super::cache::F_NO_MVEC } else { 0 };
        let mut q = edit.push(flags, depth.map_or(0, super::cache::depth_kind), target.mvec_scale, target.paper_white);
        q.step_jitter = p.step_jitter;
        q.refresh_jitter = p.refresh_jitter;
        cache_hold = Some(h);
        plan = Some(p);
        push = Some(q);
    }
    let cached = plan.is_some_and(|p| p.kind == super::cache::Kind::Cached);
    // The motion pass while the cache runs (its prep and the accumulated field the network reads), at
    // interval 1 whenever there are vectors to convert.
    let run_motion = push.is_some() && (chained || mvec.is_some());
    // A cache dump's readbacks for an armed hold, built on its first one.
    if arm.is_some() && res.cache.is_none() {
        // SAFETY: `device` is live.
        res.cache = unsafe { super::cache_dump::Buffers::build(device, instance, physical_device, target.width, target.height) };
    }
    let cache = arm.and(res.cache.as_ref()).map(|c| super::CacheCopies {
        colour: c.colour.buffer,
        answer: c.answer.buffer,
        mvec: mvec.map(|_| (native.mvec.buffer, c.mvec.buffer)),
        depth: depth.and(res.depth.as_ref()).map(|d| (d.buffer(), c.depth.buffer)),
    });
    let hdr_pass = res.hdr.as_ref();
    let depth_copy = depth.and(res.depth.as_ref()).map(|d| (d.buffer(), d.timer.as_ref()));
    let edit = res.edit.as_ref().filter(|_| cache_on);
    let acc = native.acc.buffer;
    let no_push = super::cache::CachePush::default();
    let push_ref = push.as_ref().unwrap_or(&no_push);
    // SAFETY (the three closures, only called while the cache runs): recording into the hold's own
    // buffers; the set was bound above.
    let motion = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_motion(device, cmd, push_ref, acc) };
    let capture_edit = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_capture(device, cmd, push_ref) };
    let apply = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_apply(device, cmd, push_ref) };
    // SAFETY: the capture buffer is idle; the colour view was bound above. The encode stays in the HDR
    // pass's image: the network reads it there (no copy into slot 0's proxy, the after-the-upscaler
    // path's).
    let recorded = unsafe {
        super::record_capture(
            device,
            res,
            target,
            None,
            hdr_pass,
            None,
            mvec.map(|_| native.mvec.buffer),
            depth_copy,
            cache.map(|c| c.colour),
            run_motion.then_some(&motion as &dyn Fn(vk::CommandBuffer)),
        )
    };
    if recorded.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the capture failed");
        return result;
    }
    // SAFETY: as above: on a cached hold the cache's passes write the colour input; else the decode,
    // with the native pass's answer as its source (and the edit captured first while the cache runs).
    let recorded = unsafe {
        if cached {
            super::record_cached_writeback(device, res, &apply)
        } else {
            super::record_writeback(device, res, target, native.answer(), hdr_pass, cache.as_ref(), chained.then_some(&capture_edit as &dyn Fn(vk::CommandBuffer)))
        }
    };
    if recorded.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the write-back failed");
        return result;
    }
    // A cached hold keeps the network's history as it is (no frame of the network): its parity,
    // seed and jitter stay the last refresh's.
    let (history, frame_cmds) = if cached {
        (false, [None, None])
    } else {
        let stale = native.gap.begin(Instant::now());
        if native.frames > 0 {
            note_reset(stale);
        }
        let history = native.frames > 0 && stale.is_none() && mvec.is_some();
        (history, native.frame(&built, history, jitter, target.mvec_scale, conditioning))
    };
    res.native = Some(native);
    result.timing.prep = Some(t_hold.elapsed());
    if let Some(d) = res.depth.as_mut() {
        d.valid = None;
    }
    if crate::note_vk(submit(super::Which::Capture, res.capture_cmd, vk::Fence::null())).is_err() {
        result.miss = Some("the capture submit failed");
        return result;
    }
    if let Some(d) = res.depth.as_mut() {
        d.valid = depth;
        if depth.is_some() {
            if let Some(t) = d.timer.as_mut() {
                t.mark_submitted();
            }
        }
    }
    result.waits_consumed = true;
    res.game_exposure_read = (source == super::ExposureSource::Game).then_some(target.identification);
    // From here on something of this hold is on the queue: a failure fences what went (so the next
    // hold waits for it before it records any of those buffers again), and the network's history and
    // the cache's chain start afresh (the capture's motion pass moved the field on already).
    let failed = |res: &mut super::Resources, submit: &mut dyn FnMut(super::Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>| {
        // SAFETY: the write-back fence is idle (waited above; a failed submit leaves it as it was).
        unsafe { res.fence_submitted(device, submit) };
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        if let Some(e) = res.edit.as_mut() {
            e.state.drop_edit();
        }
        if res.writeback_pending {
            loader.pre_submitted(device, res.writeback_fence);
        }
    };
    for cmd in frame_cmds.into_iter().flatten() {
        if crate::note_vk(submit(super::Which::Native, cmd, vk::Fence::null())).is_err() {
            // The capture ran; nothing reads its result. The decode is not submitted, so DLSS gets
            // the frame as it was.
            failed(res, submit);
            result.miss = Some("the network's submit failed");
            return result;
        }
        if let Some(n) = res.native.as_mut() {
            n.submitted(cmd);
        }
    }
    // SAFETY: the layer's own fence, idle (`wait_idle` above, or never submitted).
    if unsafe { device.reset_fences(&[res.writeback_fence]) }.is_err() {
        failed(res, submit);
        result.miss = Some("resetting the write-back fence failed");
        return result;
    }
    if crate::note_vk(submit(super::Which::WriteBack, res.writeback_cmd, res.writeback_fence)).is_err() {
        failed(res, submit);
        result.miss = Some("the write-back submit failed");
        return result;
    }
    // The write-back's fence also covers the capture and the network, submitted before it on the
    // same queue.
    res.writeback_pending = true;
    loader.pre_submitted(device, res.writeback_fence);
    if let Some(timer) = res.writeback_timer.as_mut() {
        timer.mark_submitted();
    }
    if cached {
        // The 500 ms rule counts from this hold, not the last network frame.
        if let Some(n) = res.native.as_mut() {
            n.gap.touch(Instant::now());
        }
    }
    if let (Some(edit), Some(h), Some(p)) = (res.edit.as_mut(), cache_hold.as_ref(), plan.as_ref()) {
        edit.state.went_through(h, p);
        edit.submitted(run_motion, cached || chained);
    }
    if let (Some(arm), Some(c)) = (arm.filter(|_| cache.is_some()), res.cache.as_mut()) {
        c.pending = Some(super::cache_dump::Pending {
            arm,
            time_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            jitter,
            history,
            mvec: mvec.is_some(),
            depth,
            depth_missing: depth.is_none().then(|| if cache_on { depth_missing(target) } else { "the edit cache is off (NEURAL_FORGE_EDIT_CACHE=off): no depth is copied" }),
            depth_layout: target.depth.and_then(|a| a.layout),
            mvec_scale: target.mvec_scale,
            paper_white: target.paper_white,
        });
    }
    result.evaluated = true;
    result.wrote_back = true;
    result.cached = cached;
    result.cache_still = cache_hold.is_some_and(|h| h.interval > 1) && res.edit.as_ref().is_some_and(|e| e.state.regime.still());
    result.timing.writeback = Some(t_hold.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_is_new_once_per_reason_until_it_clears() {
        let mut last = None;
        assert!(failure_is_new(&mut last, "no model at /x"));
        assert!(!failure_is_new(&mut last, "no model at /x"), "a retry failing the same way is not written again");
        assert!(failure_is_new(&mut last, "building the network for 1280x720 failed: out of memory"));
        last = None;
        assert!(failure_is_new(&mut last, "no model at /x"), "after a recovery the same failure is new again");
    }

    #[test]
    fn model_source_reads_build_and_verified() {
        let source = |extra: &str| format!("{{\n  \"source\": {{\n    \"build\": \"310.9.1.0\",\n    \"resource\": \"WEIGHTS_HT\"{extra}\n  }}\n}}");
        assert_eq!(model_source(&source(",\n    \"verified\": false")), "build 310.9.1.0, not verified against NVIDIA's runtime");
        assert_eq!(model_source(&source(",\n    \"verified\": true")), "build 310.9.1.0, verified");
        assert_eq!(model_source(&source("")), "build 310.9.1.0, not verified against NVIDIA's runtime");
        assert_eq!(model_source(""), "build unknown, not verified against NVIDIA's runtime");
    }

    #[test]
    fn the_history_needs_motion_vectors_the_capture_copies() {
        let aux = |layout, readable| Some(super::super::Aux { image: vk::Image::null(), format: vk::Format::R16G16_SFLOAT, layout, readable });
        assert!(copied_mvec(aux(Some(vk::ImageLayout::UNDEFINED), true)).is_none(), "UNDEFINED is not copied");
        assert!(copied_mvec(aux(Some(vk::ImageLayout::PREINITIALIZED), true)).is_none(), "PREINITIALIZED is not copied");
        assert!(copied_mvec(aux(None, true)).is_none(), "no barrier seen");
        assert!(copied_mvec(aux(Some(vk::ImageLayout::GENERAL), false)).is_none(), "not readable");
        assert!(copied_mvec(None).is_none());
        assert!(copied_mvec(aux(Some(vk::ImageLayout::GENERAL), true)).is_some());
        assert!(copied_mvec(aux(Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL), true)).is_some(), "transitioned for the copy");
    }

    /// A failure at every creation step of the native pass destroys everything created before it, and a
    /// failed build is not tried again by the next hold inside the retry's backoff.
    #[test]
    fn a_failed_native_pass_build_destroys_what_it_made_and_waits_before_trying_again() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let live = || LIVE.with(std::cell::Cell::get);
        for at in 0..BUILD_STEPS {
            // SAFETY: a fresh device; nothing is submitted.
            let pass = unsafe { NativePass::build_failing_at(device, instance, physical, family, W, H, Some(at), false) };
            assert!(pass.is_none(), "step {at} fails");
            assert_eq!(live(), 0, "every object made before step {at} is destroyed");
        }
        // SAFETY: as above.
        let pass = unsafe { NativePass::build_failing_at(device, instance, physical, family, W, H, Some(BUILD_STEPS), false) }.expect("no step fails");
        assert!(live() > 0);
        // SAFETY: nothing of it was submitted.
        unsafe { pass.destroy(device) };
        assert_eq!(live(), 0, "destroy destroys everything build made");

        // The hold's retry: a failure, then a second hold inside the backoff does not build.
        let mut retry = PassRetry::default();
        let calls = std::cell::Cell::new(0);
        let t0 = Instant::now();
        let hold = |retry: &mut PassRetry, extent: (u32, u32), at: Instant, fail: bool| {
            retry.build(extent, at, || {
                calls.set(calls.get() + 1);
                // SAFETY: as above.
                unsafe { NativePass::build_failing_at(device, instance, physical, family, extent.0, extent.1, fail.then_some(3), false) }
            })
        };
        assert!(hold(&mut retry, (W, H), t0, true).is_none());
        assert!(hold(&mut retry, (W, H), t0 + std::time::Duration::from_millis(100), true).is_none());
        assert_eq!(calls.get(), 1, "the second hold inside the backoff does not build");
        // Another extent is tried at once; the first one again once its backoff passed.
        assert!(hold(&mut retry, (W + 2, H), t0 + std::time::Duration::from_millis(200), true).is_none());
        assert_eq!(calls.get(), 2, "another extent builds at once");
        let later = t0 + BuildRetry::SHORT[0] + BuildRetry::SHORT[1] + std::time::Duration::from_millis(300);
        let pass = hold(&mut retry, (W, H), later, false).expect("built once the backoff passed");
        assert_eq!(calls.get(), 3);
        assert_eq!(retry.failed, None);
        // SAFETY: nothing of it was submitted.
        unsafe { pass.destroy(device) };
        assert_eq!(live(), 0);
    }

    /// At interval 1 the frame shaders read the accumulated field the cache's motion pass made from
    /// DLSS's vectors, where they read the packed vectors before (`MOTION_HALF`, the old path): the
    /// network's 16 input lanes and its answer are bit for bit the same over frames with history,
    /// with fractional vectors and a high-resolution scale.
    #[test]
    fn at_interval_1_the_network_sees_exactly_what_it_saw_before_the_accumulated_field() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(device, instance, physical, u64::from(pw * ph) * 8, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            let graph = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1)).unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device.begin_command_buffer(graph, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance)).unwrap();
            device.end_command_buffer(graph).unwrap();
            let frame = nn::Frame { field_width: FW, field_height: FH, features: features.buffer, features_bytes: u64::from(FW * FH) * 64, head: head.buffer, head_bytes: u64::from(FW * FH) * 16, blend_scale: 0.5, chained: 0 };
            let built = Built { width: W, height: H, frame, graph, graph_own: graph, generation: 1 };
            let texels: Vec<u16> = (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::builder().old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).image(encoded.image).subresource_range(range).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
                let region = vk::BufferImageCopy { image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 }, ..Default::default() };
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            // The old path (packed halves) and the new (the field), side by side.
            let mut old = NativePass::build_failing_at(device, instance, physical, family, W, H, None, true).unwrap();
            let mut new = NativePass::build(device, instance, physical, family, W, H, false).unwrap();
            assert!(old.record(device, &built, graph, encoded.view, exposure.buffer));
            assert!(new.record(device, &built, graph, encoded.view, exposure.buffer));
            let mut cache = super::super::cache::CachePass::build(device, instance, physical, family, W, H).unwrap();
            cache.bind(
                device,
                &super::super::cache::Bindings { colour: encoded.view, encoded: encoded.view, answer: encoded.view, exposure: exposure.buffer, depth: None, mvec: new.mvec.buffer, acc: new.acc.buffer, scratch: (vk::Buffer::null(), 0) },
            );
            let answer_of = |pass: &NativePass| -> Vec<u16> {
                fx.one_shot(|cmd| device.cmd_copy_buffer(cmd, pass.answer(), staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }]));
                std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).to_vec()
            };
            let lanes = || -> Vec<u32> { std::slice::from_raw_parts(features.ptr.cast::<u32>(), (FW * FH * 16) as usize).to_vec() };
            // Fractional vectors of both signs, some pointing out of the frame, a NaN and an infinity.
            let vectors = |frame: u32| -> Vec<u32> {
                (0..W * H)
                    .map(|i| {
                        let (x, y) = (i % W, i / W);
                        let vx = ((x * 7 + y * 3 + frame * 5) % 23) as f32 * 0.37 - 4.1;
                        let vy = ((x * 5 + y * 11 + frame) % 19) as f32 * -0.29 + 2.3;
                        let bits = |v: f32| u32::from(super::super::tests::f32_to_f16(v));
                        match i {
                            17 => u32::from(0x7e00u16) | bits(vy) << 16,
                            41 => u32::from(0x7c00u16) | bits(vy) << 16,
                            _ => bits(vx) | bits(vy) << 16,
                        }
                    })
                    .collect()
            };
            for (frame, (history, scale, jitter)) in [(false, [1.0, 1.0], Some([0.1, -0.2])), (true, [1.0, 1.0], Some([-0.3, 0.25])), (true, [0.6667, 0.6667], Some([0.4, 0.1])), (true, [0.5, 0.25], None)].into_iter().enumerate() {
                let frame = frame as u32;
                let v = vectors(frame);
                let fill = |pass: &NativePass| {
                    std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), staging.ptr, v.len() * 4);
                    fx.one_shot(|cmd| device.cmd_copy_buffer(cmd, staging.buffer, pass.mvec.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(W * H) * 4 }]));
                };
                fill(&old);
                fill(&new);
                // The capture's motion pass at interval 1.
                let push = cache.push(super::super::cache::F_KEEP_CLAMPED, 0, scale, 3.0);
                fx.one_shot(|cmd| cache.record_motion(device, cmd, &push, new.acc.buffer));
                let h: Vec<f32> = (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, frame), residual(x, y, 1, frame), residual(x, y, 2, frame), 1.5])).collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
                let cmds = old.frame(&built, history, jitter, scale, Conditioning::default());
                fx.submit(&mut old, cmds);
                let (old_lanes, old_answer) = (lanes(), answer_of(&old));
                let cmds = new.frame(&built, history, jitter, scale, Conditioning::default());
                fx.submit(&mut new, cmds);
                let (new_lanes, new_answer) = (lanes(), answer_of(&new));
                assert!(old_lanes == new_lanes, "frame {frame}: the network's input lanes differ");
                assert!(old_answer == new_answer, "frame {frame}: the answer differs");
            }
            for pass in [&old, &new] {
                pass.destroy(device);
            }
            cache.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    const W: u32 = 20;
    const H: u32 = 12;
    const FW: u32 = 32;
    const FH: u32 = 16;

    fn half(v: f32) -> u16 {
        // Truncates toward zero, which is also the composite's rounding; the test's values are exact halves.
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = b & 0x7f_ffff;
        if v == 0.0 {
            return sign;
        }
        assert!(exp > 0 && exp < 31, "test values stay normal halves");
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }

    fn f16(h: u16) -> f32 {
        super::super::f16_to_f32(h)
    }

    /// The composite's truncation to the half grid (toward zero), on the CPU.
    fn trunc_half(v: f32) -> f32 {
        let h = half(v);
        let back = f16(h);
        if back.abs() > v.abs() { f16(h - 1) } else { back }
    }

    fn centre(v: f32) -> f32 {
        // f16(f16(f16(v) - 0.5) * 0.125): exact for the test's values.
        (v - 0.5) * 0.125
    }

    fn proxy(x: u32, y: u32, c: u32) -> f32 {
        ((x * 7 + y * 3 + c * 5) % 64) as f32 / 64.0
    }

    fn residual(x: u32, y: u32, c: u32, frame: u32) -> f32 {
        (((x + y + c + frame * 3) % 9) as f32 - 4.0) * 0.25
    }

    fn neural(x: u32, y: u32, c: u32, frame: u32) -> f32 {
        let inner = residual(x, y, c, frame) * 0.03125 + (proxy(x, y, c) * 0.125 - 0.0625);
        trunc_half((inner * 8.0 + 0.5).clamp(0.0, 1.0))
    }

    struct Fixture {
        gpu: (ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32),
        pool: vk::CommandPool,
    }

    impl Fixture {
        fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.gpu.3;
            // SAFETY: the test's own pool, queue and device.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.gpu.4).unwrap();
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        /// Submits a frame's buffers ([`NativePass::frame`]) and tells the pass, as the hold does.
        fn submit(&self, pass: &mut NativePass, cmds: [Option<vk::CommandBuffer>; 2]) {
            let d = &self.gpu.3;
            let cmds: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().collect();
            // SAFETY: recorded, idle command buffers of this device.
            unsafe {
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::builder().command_buffers(&cmds).build()], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.gpu.4).unwrap();
            }
            for cmd in cmds {
                pass.submitted(cmd);
            }
        }
    }

    /// The native frame around the network (preprocess, the recorded graph, composite), with an empty
    /// secondary in the graph's place and a head written by the test: the input lanes, a first frame's
    /// answer, the blend toward the previous answer on a frame with history, and the intensity blend, each
    /// against the CPU (bit for bit where the arithmetic is exact), and the history through scaled motion vectors.
    #[test]
    fn the_native_frame_builds_the_lanes_and_composites_the_head_with_its_history() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            // The encoded proxy, the exposure (1.0), the network's features and head buffers, an empty "graph".
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(device, instance, physical, u64::from(pw * ph) * 8, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            let graph = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1)).unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device.begin_command_buffer(graph, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance)).unwrap();
            device.end_command_buffer(graph).unwrap();
            let blend = 0.5f32;
            let built = Built {
                width: W,
                height: H,
                frame: nn::Frame {
                    field_width: FW,
                    field_height: FH,
                    features: features.buffer,
                    features_bytes: u64::from(FW * FH) * 64,
                    head: head.buffer,
                    head_bytes: u64::from(FW * FH) * 16,
                    blend_scale: blend,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation: 1,
            };
            // The proxy into the encoded image (padded; the composite reads the valid part).
            let texels: Vec<u16> = (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::builder().old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).image(encoded.image).subresource_range(range).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
                let region = vk::BufferImageCopy { image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 }, ..Default::default() };
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            assert!(pass.record(device, &built, graph, encoded.view, exposure.buffer));
            // The frame shaders read the accumulated field (the cache's motion pass makes it from
            // DLSS's vectors): no motion.
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            let write_head = |frame: u32, logit: f32| {
                let h: Vec<f32> = (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, frame), residual(x, y, 1, frame), residual(x, y, 2, frame), logit])).collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
            };
            let answer = pass.answer();
            let read_answer = || -> Vec<f32> {
                fx.one_shot(|cmd| device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }]));
                std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).iter().map(|&h| f16(h)).collect()
            };
            let lanes = |x: u32, y: u32| -> Vec<f32> { std::slice::from_raw_parts(features.ptr.cast::<f32>().add(((y * FW + x) * 16) as usize), 16).to_vec() };

            // Frame 1: no history. Lanes 4-6 and 7-9 are the centred proxy; the answer is the residual on it.
            write_head(0, 10.0);
            let cmds = pass.frame(&built, false, None, [1.0, 1.0], Conditioning::default());
            assert!(cmds.iter().all(Option::is_some), "the initialisation, then the frame");
            fx.submit(&mut pass, cmds);
            let l = lanes(5, 3);
            for c in 0..3 {
                assert_eq!(l[4 + c as usize], centre(proxy(5, 3, c)), "proxy lane {c}");
                assert_eq!(l[7 + c as usize], centre(proxy(5, 3, c)), "history lane {c} without history");
            }
            assert_eq!((l[3], l[10], l[11], l[12], l[13], l[14], l[15]), (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0), "constant and conditioning lanes");
            // Padding mirrors the image without repeating the edge: field column W maps to W - 2.
            assert_eq!(lanes(W, 2)[4], centre(proxy(W - 2, 2, 0)), "the padding mirrors");
            let a1 = read_answer();
            for y in 0..H {
                for x in 0..W {
                    for c in 0..3 {
                        assert_eq!(a1[((y * pw + x) * 4 + c) as usize], neural(x, y, c, 0), "frame 1 at {x},{y} channel {c}");
                    }
                }
            }

            // Frame 2: history (no motion, no jitter): lanes 7-9 are the previous answer, and the answer blends
            // toward it by sigmoid(logit) * blendScale.
            write_head(1, 10.0);
            let cmds = pass.frame(&built, true, None, [1.0, 1.0], Conditioning::default());
            fx.submit(&mut pass, cmds);
            let l = lanes(5, 3);
            assert_eq!(l[7], centre(neural(5, 3, 0, 0)), "history lane from the previous answer");
            let a2 = read_answer();
            let weight = blend / (1.0 + (10.0f32 * -std::f32::consts::LOG2_E).exp2());
            for y in 0..H {
                for x in 0..W {
                    for c in 0..3 {
                        let n = neural(x, y, c, 1);
                        let want = trunc_half(weight.mul_add(neural(x, y, c, 0) - n, n));
                        let got = a2[((y * pw + x) * 4 + c) as usize];
                        assert!((got - want).abs() <= 2.0 * 2f32.powi(-11), "frame 2 at {x},{y} channel {c}: {got} vs {want}");
                    }
                }
            }

            // Frame 3: intensity 0.5, no history: the answer halfway from the proxy to the network's.
            write_head(2, 10.0);
            let cmds = pass.frame(&built, false, None, [1.0, 1.0], Conditioning { intensity: 0.5, ..Conditioning::default() });
            fx.submit(&mut pass, cmds);
            let a3 = read_answer();
            for c in 0..3 {
                let (p, n) = (proxy(7, 4, c), neural(7, 4, c, 2));
                assert_eq!(a3[((4 * pw + 7) * 4 + c) as usize], trunc_half(0.5f32.mul_add(n - p, p).clamp(0.0, 1.0)), "intensity channel {c}");
            }

            // Frame 4: history through motion vectors of (2, 4) (in output pixels) at a scale of
            // (0.5, 0.25) (high-resolution motion vectors): pixel (5, 3) was at (6, 4) in the previous frame.
            let field: Vec<f32> = (0..W * H).flat_map(|_| [2.0f32, 4.0]).collect();
            std::ptr::copy_nonoverlapping(field.as_ptr().cast::<u8>(), staging.ptr, field.len() * 4);
            fx.one_shot(|cmd| device.cmd_copy_buffer(cmd, staging.buffer, pass.acc.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(W * H) * 8 }]));
            write_head(3, 10.0);
            let cmds = pass.frame(&built, true, None, [0.5, 0.25], Conditioning::default());
            fx.submit(&mut pass, cmds);
            // The history is frame 3's network answer (before its intensity blend).
            assert_ne!(neural(6, 4, 0, 2), neural(7, 7, 0, 2), "the test tells the scaled position from the unscaled one");
            assert_eq!(lanes(5, 3)[7], centre(neural(6, 4, 0, 2)), "history lane from the previous answer where the scaled vector points");

            pass.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }
}
