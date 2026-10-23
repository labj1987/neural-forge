//! The native backend (docs/NATIVE_BACKEND.md, "Phase 1"): the network runs on the game's own device,
//! inside the game's own DLSS submit, instead of in the model server.
//!
//! A hold in native mode submits three command buffers in front of DLSS's launch buffer, with no CPU
//! wait between them and none for them:
//!
//! - `C`, the capture (`super::record_capture`): the exposure texel, the HDR encode into the encoded
//!   image, a copy of DLSS's motion vectors into [`NativePass`]'s motion buffer, and the scene-cut
//!   detector (`super::scene_cut`), whose decision the frame shaders read from the state buffer;
//! - `N[parity]`, recorded once per build ([`NativePass::record`]): `native_preprocess.comp` (the
//!   network's 16 input lanes from the encoded proxy and the reprojected history), the network's
//!   recorded graph (a secondary from `neural-forge-native`), and `native_composite.comp` (the
//!   residual, the blend toward the history, the next history and the answer buffer);
//! - `W`, the write-back (`super::record_writeback`): the answer decoded over DLSS's colour input.
//!
//! With more than one pass (the Passes setting) a network frame runs `N` once per pass between `C` and
//! `W` ([`MorePasses`]): before every pass after the first, the previous pass's answer is copied over
//! the encoded image the network reads (the encoded capture saved first and put back after the last
//! pass, for the write-back's clamped-highlight rule), so `W`, the edit cache's capture and a cache dump
//! see only the last pass's answer against the original frame. Each pass has its own parameters (the
//! Model tab's values with that pass's overrides) and the same history, motion vectors and jitter.
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
use neural_forge_protocol::enums::network;
use neural_forge_protocol::history::{HistoryGap, Stale};
use neural_forge_protocol::rebuild::{BuildRetry, FailInject, Step};

use super::hdr::{self, OwnImage};
use super::{own_host_buffer_with, HostBuffer, Target};

const PREPROCESS_SPV: &[u8] = include_bytes!("../../shaders/native_preprocess.spv");
const COMPOSITE_SPV: &[u8] = include_bytes!("../../shaders/native_composite.spv");
const MASK_SPV: &[u8] = include_bytes!("../../shaders/native_mask.spv");
const FEEDBACK_SPV: &[u8] = include_bytes!("../../shaders/native_feedback.spv");
const REDUCE_SPV: &[u8] = include_bytes!("../../shaders/native_reduce.spv");
const ENLARGE_SPV: &[u8] = include_bytes!("../../shaders/native_enlarge.spv");
const COLOUR_SPV: &[u8] = include_bytes!("../../shaders/native_colour.spv");

/// `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`: the frames record everything but the network's launches.
fn skip_graph() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_SKIP_GRAPH").as_deref() == Some("1"));
    *ON
}

/// `NEURAL_FORGE_NATIVE_CHAIN=0`: barriers between the network's launches instead of counter chaining.
fn chain_wanted() -> bool {
    neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_CHAIN").is_none_or(|v| v != "0")
}

/// `extract-model`'s output: `$XDG_DATA_HOME/neural-forge/model`, one directory per network in it.
pub(crate) fn model_dir() -> PathBuf {
    crate::dump::captures_dir().with_file_name("model")
}

/// The directory of `network` (`neural_forge_protocol::enums::network`) under the model directory.
fn network_dir(network: u32) -> PathBuf {
    model_dir().join(network::name(network))
}

/// Whether a network's directory is there at all (the hash check is the loader's).
fn network_present(network: u32) -> bool {
    network_dir(network).join("manifest.json").is_file()
}

/// The network to load for the Network setting `chosen`: the chosen one when it is extracted, else
/// the other one when that is (a DLL without the small network extracts only the full one), else
/// `None`. `present` says whether a network is extracted.
fn network_to_load(chosen: u32, present: impl Fn(u32) -> bool) -> Option<u32> {
    let other = if chosen == network::SMALL { network::FULL } else { network::SMALL };
    [chosen, other].into_iter().find(|&n| present(n))
}

/// The history holds the answers of the network that ran last: a frame of another one (the Network
/// setting switched) starts it afresh, as a frame after a skipped one does. `history_of` is the network
/// the history came from, updated to `network`. Returns whether the network changed.
fn note_network(gap: &mut HistoryGap<(u32, u32)>, history_of: &mut Option<u32>, network: u32) -> bool {
    let changed = history_of.replace(network).is_some_and(|n| n != network);
    if changed {
        gap.skipped();
    }
    changed
}

/// "full network, build 310.8.0.0, verified": the manifest's `network`, `source.build` and
/// `source.verified`, for the log. Each key appears once in `extract-model`'s manifest, so a string
/// scan is enough. A manifest without `verified` is unverified; one without `network` is the full one.
fn model_source(text: &str) -> String {
    let field = |key: &str| {
        let (_, rest) = text.split_once(&format!("\"{key}\": "))?;
        Some(rest.split([',', '\n', '}']).next()?.trim().trim_matches('"').to_string())
    };
    let build = field("build").unwrap_or_else(|| "unknown".to_string());
    let verified = field("verified").is_some_and(|v| v == "true");
    let network = field("network").unwrap_or_else(|| "full".to_string());
    format!("{network} network, build {build}, {}", if verified { "verified" } else { "not verified against NVIDIA's runtime" })
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
    /// The optical-flow queue (`NEURAL_FORGE_POST_MOTION=flow` on a device that has one; `native_post`).
    pub flow: Option<super::post_flow::FlowQueue>,
}

#[cfg(test)]
impl Setup {
    /// A setup for tests: the network's queue is the frames' (nothing loads: `Loader::for_test`).
    pub(crate) fn for_test(entry: &ash::Entry, instance: &ash::Instance, physical: vk::PhysicalDevice, family: u32) -> Self {
        Self {
            gipa: entry.static_fn().get_instance_proc_addr,
            instance: instance.handle(),
            physical,
            frame_family: family,
            family,
            index: 0,
            post_index: 1,
            properties2: crate::vram::Properties2::default(),
            flow: None,
        }
    }
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
    /// The network it runs (`neural_forge_protocol::enums::network`): the one the Network setting
    /// chose, or the other when that one is not extracted.
    pub network: u32,
}

#[derive(Default)]
struct LoaderState {
    /// `None` while the worker has it out (opening, building), before it is opened, and after a
    /// re-initialisation closed it.
    network: Option<nn::Network>,
    /// Why the network is not usable right now (the Status tab shows it); cleared by a success.
    failed: Option<String>,
    /// The graphs built, one per size (empty: none); one build makes them all (`nn::Network::build_sizes`).
    built: Vec<Built>,
    /// The sizes the holds want, in their order (the first pass's first: a model resolution per pass asks
    /// for several); the worker builds them.
    wanted: Option<Vec<(u32, u32)>>,
    fall_back: bool,
    quit: bool,
    generation: u64,
    /// Opened at least once (a later `None` network means closed for a re-initialisation).
    opened: bool,
    /// A wait on the network's queue timed out ([`nn::Network::stalled`]): the network is dead on this
    /// device and never opened or built again.
    dead: bool,
    /// The Network setting the holds last asked for ([`Loader::ready`]); 0, the full network, until then.
    choice: u32,
    /// The setting the open network was opened for, and the network it runs (the chosen one, or the
    /// other when that is not extracted).
    opened_for: Option<u32>,
    running: u32,
    /// The graph was switched to barriers ([`Loader::fall_back`]): a network opened after a switch of
    /// the Network setting is switched too.
    barriers: bool,
}

impl LoaderState {
    /// Whether the loader thread has work: the network is not open yet, the graph is to fall back to
    /// barriers, the network open is not the one the setting chooses, or a size is wanted that is not
    /// built.
    fn wants_work(&self) -> bool {
        !self.dead && (!self.opened || self.fall_back || self.switching() || self.wanted.as_ref().is_some_and(|w| *w != self.built_sizes()))
    }

    /// The sizes of what is built, in their order.
    fn built_sizes(&self) -> Vec<(u32, u32)> {
        self.built.iter().map(|b| (b.width, b.height)).collect()
    }

    /// The network was last opened for another Network setting than the holds' now: it is closed (when
    /// open) and the chosen one opened.
    fn switching(&self) -> bool {
        self.opened_for.is_some_and(|f| f != self.choice)
    }

    /// The build the holds may use for `size` and the Network setting `choice`, after noting what they
    /// want (the caller wakes the loader thread when this returns `true` in its second field).
    #[cfg(test)]
    fn want(&mut self, size: (u32, u32), choice: u32) -> (Option<Built>, bool) {
        let (built, wake) = self.want_sizes(&[size], choice);
        (built.map(|b| b[0]), wake)
    }

    /// [`Self::want`] for several sizes at once (one graph each, built together): the builds in `sizes`'
    /// order once exactly those are built.
    fn want_sizes(&mut self, sizes: &[(u32, u32)], choice: u32) -> (Option<Vec<Built>>, bool) {
        let mut wake = false;
        if self.choice != choice {
            self.choice = choice;
            wake = true;
        }
        if self.opened_for == Some(choice) && !self.built.is_empty() && self.built_sizes() == sizes {
            return (Some(self.built.clone()), wake);
        }
        if self.wanted.as_deref() != Some(sizes) {
            self.wanted = Some(sizes.to_vec());
            wake = true;
        }
        (None, wake)
    }
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
    /// Until when (ms since [`epoch_ms`]'s start) a pre-upscaler hold's work that no fence covers may
    /// still run ([`Self::pre_unfenced`]): the after-the-upscaler path keeps off the network till then.
    pre_unfenced_until: std::sync::atomic::AtomicU64,
    /// The after-the-upscaler frame that did not finish in time, still holding the claim
    /// ([`Self::post_stalled`]): released once its fence signals.
    post_inflight: Mutex<Option<InFlight>>,
    /// Holds recorded into DLSS's command buffers that may still run (`super::inline`): they read the
    /// current build, so nothing rebuilds it ([`Self::ready_sizes`], [`Self::fall_back`]) and the
    /// after-the-upscaler path does not take the network ([`Self::pre_running`]) while there are any.
    holds: Arc<std::sync::atomic::AtomicUsize>,
    /// Tests: what records "the network" into a hold's buffer, in place of the graph (no network on lavapipe).
    #[cfg(test)]
    test_graph: Option<TestGraph>,
}

/// [`Loader::test_graph`].
#[cfg(test)]
pub(crate) type TestGraph = Box<dyn Fn(vk::CommandBuffer, &Built) + Send + Sync>;

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

/// The holds' clock for the history's gap rule (`HistoryGap::MAX_GAP`): the time now, or in a test the
/// rig's stopped clock ([`stop_clock`]), so that how long the test machine takes between two holds does not
/// decide whether the second one has the first one's history.
fn hold_now() -> Instant {
    #[cfg(test)]
    if let Some(t) = STOPPED_CLOCK.get() {
        return t;
    }
    Instant::now()
}

#[cfg(test)]
thread_local! {
    static STOPPED_CLOCK: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

/// Tests: stops [`hold_now`] on this thread, for the rest of the test: its holds follow one another with no
/// gap, however long the machine takes between them (a CI runner took over `HistoryGap::MAX_GAP` between two
/// holds now and then, and the cached hold became a refresh).
#[cfg(test)]
pub(crate) fn stop_clock() {
    STOPPED_CLOCK.set(Some(Instant::now()));
}

/// Milliseconds since the first call in this process (a monotonic stamp for [`Loader`]'s claims).
fn epoch_ms() -> u64 {
    static START: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);
    START.elapsed().as_millis() as u64 + 1
}

/// How long after one path last used the network the other may.
const PATH_SWITCH_MS: u64 = 1000;

/// The next [`Built::generation`] (process-wide).
static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// `1485x836`, or `1485x836 and 743x418` for a build of several sizes.
fn sizes_text(sizes: &[(u32, u32)]) -> String {
    sizes.iter().map(|(w, h)| format!("{w}x{h}")).collect::<Vec<_>>().join(" and ")
}

unsafe extern "C" fn init_dispatchable(_user: *mut std::ffi::c_void, device: vk::Device, object: *mut std::ffi::c_void) {
    // SAFETY: a dispatchable object of `device` the network just got from below the loader.
    let _ = unsafe { crate::loader_data::initialize_object(device, object) };
}

impl Loader {
    /// The network's loader for `device`; `holds` counts the holds recorded into DLSS's buffers that may
    /// still run (shared with `super::inline::Inline`).
    pub(crate) fn start(device: vk::Device, setup: Setup, holds: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        Self {
            shared: Arc::default(),
            worker: Mutex::new(None),
            pending: Mutex::new(Some((device, setup))),
            setup,
            last_pre: Default::default(),
            last_post: Default::default(),
            post_busy: Default::default(),
            pre_inflight: Mutex::new(None),
            pre_unfenced_until: Default::default(),
            post_inflight: Mutex::new(None),
            holds,
            #[cfg(test)]
            test_graph: None,
        }
    }

    /// Tests: a loader whose network is `built` (open, no worker), with `graph` recording into a hold's
    /// buffer in the graph's place.
    #[cfg(test)]
    pub(crate) fn for_test(setup: Setup, built: Vec<Built>, holds: Arc<std::sync::atomic::AtomicUsize>, graph: TestGraph) -> Self {
        let mut loader = Self::start(vk::Device::null(), setup, holds);
        *loader.pending.get_mut().unwrap() = None;
        loader.test_graph = Some(graph);
        {
            let mut st = loader.shared.0.lock().unwrap();
            st.opened = true;
            st.opened_for = Some(network::FULL);
            st.choice = network::FULL;
            st.running = network::FULL;
            st.generation = built.first().map_or(0, |b| b.generation);
            st.built = built;
        }
        loader
    }

    /// Holds recorded into DLSS's buffers that may still run.
    fn holds_out(&self) -> bool {
        self.holds.load(std::sync::atomic::Ordering::SeqCst) > 0
    }

    /// The network is to be rebuilt, and waits for the holds recorded into DLSS's buffers with the current
    /// build to be done with ([`Self::ready_sizes`]).
    pub(crate) fn waiting_for_holds(&self) -> bool {
        self.holds_out()
    }

    /// Records the network built as `built` (one of [`Self::ready_sizes`]' builds) into `cmd`, a hold's
    /// command buffer of the game's graphics family (`super::inline`), with hold slot `hold`'s descriptor
    /// sets (`nn::Network::record_graph_at`).
    ///
    /// # Safety
    /// As `nn::Network::record_graph_at`: `cmd` is recording on this thread, and nothing recorded with slot
    /// `hold` may still run.
    pub(crate) unsafe fn record_graph(&self, built: &Built, hold: u32, cmd: vk::CommandBuffer) -> Result<(), String> {
        #[cfg(test)]
        if let Some(graph) = &self.test_graph {
            graph(cmd, built);
            return Ok(());
        }
        let mut st = self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let slot = st
            .built
            .iter()
            .position(|b| b.generation == built.generation && (b.width, b.height) == (built.width, built.height))
            .ok_or("the network was rebuilt since this hold was planned")?;
        let network = st.network.as_mut().ok_or("the network is being built")?;
        // SAFETY: forwarded; the build stays while holds recorded with it may run (`holds`).
        unsafe { network.record_graph_at(slot as u32, hold, cmd) }
    }

    pub(crate) fn setup(&self) -> Setup {
        self.setup
    }

    /// The after-the-upscaler path may run a frame now (no pre-upscaler hold for a second). Pair with
    /// [`Self::release_post`] once its work completed.
    pub(crate) fn claim_post(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        if epoch_ms().saturating_sub(self.last_pre.load(SeqCst)) < PATH_SWITCH_MS || self.pre_running() || epoch_ms() < self.pre_unfenced_until.load(SeqCst) {
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

    /// A pre-upscaler hold's work went to the queue and no fence could be put after it: the network is
    /// taken as in use for as long as any wait of the layer's is bounded ([`crate::FENCE_WAIT_TIMEOUT`]).
    pub(crate) fn pre_unfenced(&self) {
        let until = epoch_ms() + crate::FENCE_WAIT_TIMEOUT.as_millis() as u64;
        self.pre_unfenced_until.fetch_max(until, std::sync::atomic::Ordering::SeqCst);
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

    /// Whether a pre-upscaler hold's work may still use the network: a split hold's (its fence), or one
    /// recorded into DLSS's buffers.
    fn pre_running(&self) -> bool {
        if self.holds_out() {
            return true;
        }
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
        // A path that never ran (0) has nothing to switch away from.
        let last_post = self.last_post.load(SeqCst);
        !self.post_busy.load(SeqCst) && (last_post == 0 || epoch_ms().saturating_sub(last_post) >= PATH_SWITCH_MS)
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

    /// Opens the network the Network setting `choice` names ([`network_to_load`]); returns it with
    /// which network it is.
    fn open(device: vk::Device, setup: Setup, choice: u32) -> Result<(nn::Network, u32), nn::OpenError> {
        let Some(running) = network_to_load(choice, network_present) else {
            let message = format!("no model at {}: extract it in the Setup tab (neural-forge-cli extract-model)", network_dir(choice).display());
            return Err(nn::OpenError { message, stalled: false });
        };
        if running != choice {
            crate::log!(
                "[native] the {} network is chosen but not extracted (the DLL has none); running the {} network",
                network::name(choice),
                network::name(running)
            );
        }
        let dir = network_dir(running);
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
        crate::log!(
            "[native] model loaded and verified from {} in {} ms (chaining {})",
            dir.display(),
            t.elapsed().as_millis(),
            if chain_wanted() { "on" } else { "off" }
        );
        crate::log!("[native] model source: {}", model_source(&std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_default()));
        Ok((network, running))
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
            // Work is wanted when the network is not open yet, another network is chosen, a size is
            // wanted that is not built, or the graph is to fall back to barriers
            // ([`LoaderState::wants_work`]); it is due when the retry schedule says so.
            if state.quit {
                return;
            }
            if !state.wants_work() {
                state = wake.wait(state).unwrap();
                continue;
            }
            let step = retry.step(Instant::now());
            if let Step::Wait(d) = step {
                state = wake.wait_timeout(state, d).unwrap().0;
                continue;
            }
            let switching = state.switching();
            let mut network = state.network.take();
            let fall_back = std::mem::take(&mut state.fall_back) || (switching && state.barriers);
            let sizes = state.wanted.clone().or_else(|| (!state.built.is_empty()).then(|| state.built_sizes()));
            let choice = state.choice;
            state.built = Vec::new();
            drop(state);
            if step == Step::ReinitThenBuild && network.take().is_some() {
                crate::log!("[native] {} failed attempts in a row: closing the network and loading it again", retry.streak());
            }
            // Another network chosen: this one is closed and the chosen one opened (and built for the
            // size wanted). SAFETY of the close: nothing of it is pending, as for a rebuild below (a hold
            // asks for another network only in `ready`, after the frames that used this one were drained,
            // and neither path takes the network while the other's work may run).
            if switching && network.take().is_some() {
                crate::log!("[native] the {} network is chosen: closing the open one and loading it", network::name(choice));
            }
            let t = Instant::now();
            let vram_before = vram.as_ref().map(crate::vram::reading);
            let mut open_stalled = false;
            let mut opened = None;
            let mut switched = false;
            let mut loaded = None;
            let result = match network.take() {
                Some(n) => Ok(n),
                None => Self::open(device, setup, choice)
                    .map(|(n, running)| {
                        opened = Some((t.elapsed(), vram.as_ref().map(crate::vram::reading)));
                        loaded = Some(running);
                        n
                    })
                    .map_err(|e| {
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
                match sizes {
                    None => Ok((n, None)),
                    Some(_) if inject.as_mut().is_some_and(FailInject::should_fail) => {
                        network = Some(n);
                        Err(format!("{} asked this build to fail", FailInject::ENV))
                    }
                    Some(sizes) => {
                        // SAFETY: nothing of the previous build is pending: a wanted size or a fallback is
                        // only asked for once the holds stopped using it (`ready` returns another size's
                        // build to no hold; `fall_back` is called after the frames that used it were
                        // drained; neither path takes the network while the other's work may run).
                        // One size is the single build it always was.
                        let made = match sizes.as_slice() {
                            &[(w, h)] => unsafe { n.build(w, h) }.map(|f| vec![f]),
                            _ => unsafe { n.build_sizes(&sizes) },
                        };
                        match made {
                            Ok(frames) => Ok((n, Some(sizes.iter().zip(frames).map(|(&(w, h), f)| (w, h, f)).collect::<Vec<_>>()))),
                            Err(why) => {
                                network = Some(n);
                                Err(format!("building the network for {} failed: {why}", sizes_text(&sizes)))
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
            if let Some(running) = loaded {
                // The network now open (kept below whether its build succeeds) is the one for `choice`.
                state.opened_for = Some(choice);
                state.running = running;
            }
            if let (Some((took, after)), Ok((n, _))) = (opened, &result) {
                let weights = n.weight_bytes();
                lines.push(format!(
                    "[native] model loaded in {} ms{}: {} MiB of weight records on the GPU",
                    took.as_millis(),
                    match (vram_before, after) {
                        (Some((before, _)), Some((after, budget))) => format!(" (vram {before} -> {after} of {budget} MiB)"),
                        _ => String::new(),
                    },
                    weights.raw_loaded >> 20
                ));
            }
            match result {
                Ok((n, built)) => {
                    state.barriers |= switched;
                    if let Some(built) = built {
                        // Unique in the process: a hold's buffers recorded for one build are never taken
                        // for another's.
                        state.generation = NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let network = state.running;
                        let generation = state.generation;
                        state.built = built
                            .iter()
                            .enumerate()
                            .map(|(i, &(w, h, frame))| Built {
                                width: w,
                                height: h,
                                frame,
                                graph: n.graph_commands_at(i as u32),
                                graph_own: n.graph_commands_own_at(i as u32),
                                generation,
                                network,
                            })
                            .collect();
                        let fields: Vec<String> = built.iter().map(|(_, _, f)| format!("{}x{}", f.field_width, f.field_height)).collect();
                        let sizes: Vec<(u32, u32)> = built.iter().map(|&(w, h, _)| (w, h)).collect();
                        lines.push(format!(
                            "[native] network {}built for {} (field {}, chained {}) in {} ms{}",
                            if fall_back { "re" } else { "" },
                            sizes_text(&sizes),
                            fields.join(" and "),
                            built.iter().any(|(_, _, f)| f.chained != 0),
                            t.elapsed().as_millis(),
                            match (vram_before, vram_after) {
                                (Some((before, _)), Some((after, budget))) => format!(" (vram {before} -> {after} of {budget} MiB)"),
                                _ => String::new(),
                            }
                        ));
                        let weights = n.weight_bytes();
                        lines.push(format!(
                            "[native] weights on the GPU: {} MiB of matrices, {} KiB of scale vectors (the records were {} MiB as loaded)",
                            weights.matrices >> 20,
                            weights.raw >> 10,
                            weights.raw_loaded >> 20
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

    /// The network the Network setting `choice` names (`neural_forge_protocol::enums::network`), built
    /// for `width` x `height`, or `None` (and the build, or the other network, is asked for) while it
    /// is not. Called only once the caller's frames that used the network have completed: another
    /// size or network asked for here frees the current build.
    pub(crate) fn ready(&self, width: u32, height: u32, choice: u32) -> Option<Built> {
        self.ready_sizes(&[(width, height)], choice).map(|b| b[0])
    }

    /// [`Self::ready`] for several sizes (a model resolution per pass): one graph each, built together, in
    /// `sizes`' order; `None` until exactly those are built. Asking for another set frees every graph of the
    /// current one (the same rule as [`Self::ready`]'s).
    ///
    /// While holds recorded into DLSS's buffers with the current build may still run, another set of sizes or
    /// another network is not asked for (the build they read stays): `None` until they are done with
    /// ([`Self::waiting_for_holds`]).
    pub(crate) fn ready_sizes(&self, sizes: &[(u32, u32)], choice: u32) -> Option<Vec<Built>> {
        self.spawn();
        let (lock, wake) = &*self.shared;
        let mut st = lock.lock().unwrap();
        if self.holds_out() && !(st.opened_for == Some(choice) && !st.built.is_empty() && st.built_sizes() == sizes) {
            return None;
        }
        let (built, woken) = st.want_sizes(sizes, choice);
        drop(st);
        if woken {
            wake.notify_all();
        }
        built
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

    /// Rebuilds the graph with barriers. The caller has drained every frame of its own that used the current
    /// one; `false` (nothing asked) while holds recorded into DLSS's buffers with it may still run.
    pub(crate) fn fall_back(&self) -> bool {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if self.holds_out() {
            return false;
        }
        state.fall_back = true;
        state.built = Vec::new();
        wake.notify_all();
        true
    }
}

impl Loader {
    /// Stops the loading thread (joined: it submits to the network's queue) and keeps the network: the
    /// device's teardown waits for the device to go idle before the network is closed (by dropping the
    /// loader).
    pub(crate) fn stop_worker(&self) {
        {
            let (lock, wake) = &*self.shared;
            lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner).quit = true;
            wake.notify_all();
        }
        if let Some(worker) = self.worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            let _ = worker.join();
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.stop_worker();
        // The network (closed here) must not be pending: the device's teardown waited for its work.
        self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).network.take();
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
    /// The control mask ([`Mask`]): its rules, the depth's kind, reversed depth.
    mask: [u32; 4],
    sky: [f32; 4],
    dark: [f32; 4],
    moving: [f32; 4],
    /// A pass below the render extent (`native_reduce.comp`, `native_enlarge.comp`): the render extent, its
    /// padded width, and the kind of the depth the capture copied (`cache::depth_kind`).
    full: [u32; 4],
    /// The pass's extent over the render extent (x, y), the enlarge's guide sigma, unused.
    ratio: [f32; 4],
    /// The colour of the edit (`native_colour.comp`; [`super::colour::PassColour`]).
    colour: [f32; 4],
    colour_flags: [u32; 4],
    lut_scale: [f32; 4],
    lut_offset: [f32; 4],
}

/// The model resolution's guide sigma (`native_enlarge.comp`), in the encoded picture's units: how far a
/// low-resolution texel's colour may be from the full-resolution pixel's before its edit stops counting
/// there (upstream's joint bilateral sigma, 0.08, on all three channels).
pub(crate) const GUIDE_SIGMA: f32 = 0.08;

/// The network's extent for a render extent at the model resolution `scale` (`preupscale_scale`,
/// `pass<N>_scale`): each side rounded, at least 8 texels (or the side itself), at most the side. A scale of
/// 1 or more, or one that is not a number, is the render extent itself.
pub(crate) fn model_extent(width: u32, height: u32, scale: f32) -> (u32, u32) {
    if scale.is_nan() || scale >= 1.0 {
        return (width, height);
    }
    let s = scale.max(neural_forge_protocol::enums::model_scale::MIN);
    let side = |n: u32| ((n as f32 * s).round() as u32).clamp(n.min(8), n);
    (side(width), side(height))
}

/// Every pass slot's extent for a render extent and the passes' scales ([`model_extent`]).
pub(crate) fn pass_extents(width: u32, height: u32, scales: &[f32; neural_forge_protocol::MAX_PASSES]) -> [(u32, u32); neural_forge_protocol::MAX_PASSES] {
    scales.map(|s| model_extent(width, height, s))
}

/// The distinct extents the first `count` passes run at, pass 1's first: the sizes the network is built for.
pub(crate) fn graph_sizes(extents: &[(u32, u32)], count: usize) -> Vec<(u32, u32)> {
    let mut sizes: Vec<(u32, u32)> = Vec::new();
    for &e in extents.iter().take(count.max(1)) {
        if !sizes.contains(&e) {
            sizes.push(e);
        }
    }
    sizes
}

/// One extent's network as a frame runs it: its build and its recording for the hold's queue family.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Graph {
    pub built: Built,
    pub graph: vk::CommandBuffer,
}

impl Graph {
    fn of(graphs: &[Graph], extent: (u32, u32)) -> Option<&Graph> {
        graphs.iter().find(|g| (g.built.width, g.built.height) == extent)
    }
}

/// One network frame as [`NativePass::frame_plan`] planned it.
#[derive(Clone, Debug)]
pub(crate) struct FramePlan {
    /// The frame's parity: which history each pass reads.
    pub parity: usize,
    /// Each pass's parameters and the buffer they go to.
    pub params: Vec<(vk::Buffer, Params)>,
    /// The passes, in order: pass 1 first.
    pub passes: Vec<FramePass>,
    /// The later passes whose histories are still `UNDEFINED`, one bit per later pass.
    pub fresh: u32,
}

/// One pass of a [`FramePlan`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FramePass {
    /// 0 for pass 1.
    pub index: usize,
    pub set: vk::DescriptorSet,
    pub extent: (u32, u32),
    /// The pass builds the control mask (the first at its extent).
    pub mask: bool,
}

/// The control mask's rules for one frame (`native_mask.comp`): bit 1 keeps the sky, 2 the dark rule, 4
/// the motion rule; 0 is no mask (nothing reads it, the frame is as without one).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Mask {
    pub rules: u32,
    /// `cache::depth_kind` of the depth copied this frame (0: none, and the sky rule is dropped).
    pub depth_kind: u32,
    pub reversed: bool,
    pub sky_margin: f32,
    /// Threshold, intensity, structure.
    pub dark: [f32; 3],
    pub motion: [f32; 3],
}

pub(crate) const MASK_SKY: u32 = 1;
pub(crate) const MASK_DARK: u32 = 2;
pub(crate) const MASK_MOTION: u32 = 4;

impl From<neural_forge_protocol::MaskSettings> for Mask {
    fn from(m: neural_forge_protocol::MaskSettings) -> Self {
        Self {
            rules: if m.sky { MASK_SKY } else { 0 } | if m.dark { MASK_DARK } else { 0 } | if m.motion { MASK_MOTION } else { 0 },
            depth_kind: 0,
            reversed: false,
            sky_margin: m.sky_margin,
            dark: [m.dark_threshold, m.dark_intensity, m.dark_structure],
            motion: [m.motion_threshold, m.motion_intensity, m.motion_structure],
        }
    }
}

impl Mask {
    /// This frame's rules: the sky rule needs the depth and its convention (the edit cache decides it
    /// from the first holds), so without either it is dropped.
    pub(crate) fn for_frame(self, depth_kind: u32, reversed: Option<bool>) -> Self {
        let sky = depth_kind != 0 && reversed.is_some();
        Self { rules: if sky { self.rules } else { self.rules & !MASK_SKY }, depth_kind, reversed: reversed == Some(true), ..self }
    }
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

/// The network's passes for one frame: how many, and each one's conditioning (entries past `count` are
/// unused).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Passes {
    pub count: usize,
    pub each: [Conditioning; neural_forge_protocol::MAX_PASSES],
    /// The control mask, for every pass (built once, before the first).
    pub mask: Mask,
    /// How much of a pass's answer becomes the next pass's input (1: the answer itself).
    pub feedback: f32,
    /// Each pass's model resolution, a scale of the render extent (1: the network at it).
    pub scales: [f32; neural_forge_protocol::MAX_PASSES],
    /// The colour of each pass's edit, and the stack's colour controls.
    pub colour: neural_forge_protocol::ColourSettings,
}

impl Passes {
    pub(crate) fn one(c: Conditioning) -> Self {
        Self {
            count: 1,
            each: [c; neural_forge_protocol::MAX_PASSES],
            mask: Mask::default(),
            feedback: 1.0,
            scales: [1.0; neural_forge_protocol::MAX_PASSES],
            colour: neural_forge_protocol::ColourSettings::DEFAULT,
        }
    }
}

impl Default for Passes {
    fn default() -> Self {
        Self::one(Conditioning::default())
    }
}

impl Passes {
    /// Passes `first..end` as a stage runs them ([`super::placement`]): its pass 1 is pass `first + 1`, with
    /// that pass's settings, model resolution and colour. The look-up table stays the stack's last pass's: a
    /// stage that does not run it has none.
    pub(crate) fn stage(&self, first: usize, end: usize) -> Self {
        let max = neural_forge_protocol::MAX_PASSES;
        let mut out = *self;
        out.count = end.min(self.count).saturating_sub(first).max(1);
        for k in 0..max {
            let from = (first + k).min(max - 1);
            out.each[k] = self.each[from];
            out.scales[k] = self.scales[from];
            out.colour.each[k] = self.colour.each[from];
        }
        if end < self.count {
            out.colour.lut_strength = 0.0;
        }
        out
    }
}

/// The passes the hold `at` runs ([`Passes::stage`]); `None` when none of the frame's passes is at its stage.
pub(crate) fn stage_passes(all: &Passes, at: super::placement::Hold) -> Option<Passes> {
    let (first, end) = at.passes();
    let end = end.min(all.count);
    (first < end).then(|| all.stage(first, end))
}

/// The sizes the network is built for in a frame whose hold `at` runs at `own` (its colour image's size): the
/// passes before the upscaler at the render size, then those after DLSS at the output size, each size once
/// (both holds of a frame ask for the same set: another set would build the network again). With only the
/// holds before the upscaler this is [`graph_sizes`] of the render size, as it always was.
pub(crate) fn frame_sizes(all: &Passes, at: super::placement::Hold, own: (u32, u32)) -> Vec<(u32, u32)> {
    use super::placement::Stage;
    let (render, output) = match at.stage {
        Stage::Upscaled => (at.render, own),
        _ => (own, at.output),
    };
    let mut sizes = Vec::new();
    for (stage, extent) in [(Stage::Before, render), (Stage::Upscaled, output)] {
        let Some(p) = stage_passes(all, super::placement::Hold { stage, ..at }) else { continue };
        if extent.0 == 0 || extent.1 == 0 {
            continue;
        }
        for size in graph_sizes(&pass_extents(extent.0, extent.1, &p.scales), p.count) {
            if !sizes.contains(&size) {
                sizes.push(size);
            }
        }
    }
    if sizes.is_empty() {
        sizes = graph_sizes(&pass_extents(own.0, own.1, &all.scales), all.count);
    }
    sizes
}

impl From<neural_forge_protocol::PassTunings> for Passes {
    fn from(t: neural_forge_protocol::PassTunings) -> Self {
        let mut each = [Conditioning::default(); neural_forge_protocol::MAX_PASSES];
        for (c, &tuning) in each.iter_mut().zip(t.tunings.iter()) {
            *c = Conditioning::from(tuning);
        }
        Self {
            count: t.count.clamp(1, neural_forge_protocol::MAX_PASSES),
            each,
            mask: Mask::default(),
            feedback: t.feedback,
            scales: t.scales,
            colour: t.colour,
        }
    }
}

/// Passes 2 to [`neural_forge_protocol::MAX_PASSES`] (the Passes setting), built on the first frame that
/// asks for more than one: per pass its own parameters, descriptor sets, `N[parity]` buffers (the same
/// shaders and graph as pass 1's) and history (a pass blends toward its own previous answer, never another
/// pass's: pass 1 toward the multi-pass picture would compound frame after frame), and the copies between
/// passes: `X` saves the encoded capture and copies pass 1's answer over it, `Y` copies a later pass's
/// answer over it, `R` puts the capture back. One timer runs from `X`'s start to `R`'s end.
pub(crate) struct MorePasses {
    pool: vk::DescriptorPool,
    params: Vec<[HostBuffer; 2]>,
    sets: Vec<[vk::DescriptorSet; 2]>,
    commands: Vec<[vk::CommandBuffer; 2]>,
    /// Each later pass's history images (as [`NativePass`]'s), built when the pass count first reaches it.
    history: Vec<Option<[OwnImage; 2]>>,
    /// Its history images are still `UNDEFINED`: the next `X` takes them to `GENERAL`.
    fresh: Vec<bool>,
    /// The frame ([`NativePass`]'s count) the pass last ran on: its history is the previous frame's only
    /// when that was the frame before.
    last_run: Vec<Option<u64>>,
    copies: [vk::CommandBuffer; 3],
    /// Pass feedback below 1 (`native_feedback.comp`), built the first time it is.
    feedback: Option<Feedback>,
    saved: OwnBuffer,
    timer: Option<crate::gpu_timer::GpuTimer>,
    /// The [`Built::generation`] `commands` were recorded for.
    recorded: Option<u64>,
    /// Per later pass, the generation its `commands` were recorded for (a pass whose extent has no graph in
    /// that build is not recorded and does not run).
    recorded_pass: Vec<Option<u64>>,
}

/// [`MorePasses::feedback`]: the blend's pipeline and its one set (the encoded image, the answer).
struct Feedback {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The encoded view and answer buffer the set points at.
    bound: Option<(vk::ImageView, vk::Buffer)>,
}

/// `native_feedback.comp`'s push constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct FeedbackPush {
    width: u32,
    height: u32,
    feedback: f32,
}

impl Feedback {
    /// # Safety
    /// `device` is live.
    unsafe fn build(device: &ash::Device) -> Option<Self> {
        let b = |n: u32, ty| {
            vk::DescriptorSetLayoutBinding::default().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [b(0, vk::DescriptorType::STORAGE_IMAGE), b(1, vk::DescriptorType::STORAGE_BUFFER)];
        // SAFETY: valid create infos; each object is destroyed below on a later failure.
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<FeedbackPush>() as u32);
            let Ok(layout) =
                device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&[push]), None)
            else {
                device.destroy_descriptor_set_layout(set_layout, None);
                return None;
            };
            let undo = |pipeline: Option<vk::Pipeline>, pool: Option<vk::DescriptorPool>| {
                if let Some(p) = pool {
                    device.destroy_descriptor_pool(p, None);
                }
                if let Some(p) = pipeline {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
                None
            };
            let Some(pipeline) = hdr::compute_pipeline(device, layout, FEEDBACK_SPV) else { return undo(None, None) };
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
            ];
            let Ok(pool) = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) else {
                return undo(Some(pipeline), None);
            };
            let Ok(sets) = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])) else {
                return undo(Some(pipeline), Some(pool));
            };
            Some(Self { set_layout, layout, pipeline, pool, set: sets[0], bound: None })
        }
    }

    /// # Safety
    /// Nothing submitted uses it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; the set goes with the pool.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The passes [`MorePasses`] holds.
const MORE: usize = neural_forge_protocol::MAX_PASSES - 1;
/// The storage images and storage buffers one frame set binds.
const IMAGES_PER_SET: u32 = 3;
const BUFFERS_PER_SET: u32 = 13;

/// A device-local buffer of the layer's own.
pub(crate) struct OwnBuffer {
    pub(super) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl OwnBuffer {
    /// # Safety
    /// `device` is live.
    pub(super) unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Option<Self> {
        let info = vk::BufferCreateInfo::default().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
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
            unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None) }.ok()
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
    /// `timed`: with its own GPU timer (the split hold's).
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
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
        // SAFETY: forwarded.
        let buffer = unsafe { OwnBuffer::new(device, instance, physical_device, u64::from(width) * u64::from(height) * 4, usage) }?;
        // Only made when kept: a timer has no destructor, and one dropped leaks its query pool.
        let timer = if timed { crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family) } else { None };
        Some(Self { buffer, valid: None, timer })
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
///
/// With a model resolution below 1 (`preupscale_scale`, `pass<N>_scale`) a pass runs at a smaller extent
/// ([`Self::extents`]): its `N` buffer opens with `native_reduce.comp` (the encoded picture, the motion and
/// the depth at that extent, into its [`Lane`]), runs the network there with its own history, and closes
/// with `native_enlarge.comp`, which brings its edit back to the render extent in the answer buffer. Every
/// pass's answer, and everything after the passes, is at the render extent.
pub(crate) struct NativePass {
    /// Unique in the process: what a hold recorded into DLSS's buffer with this pass names (`inline::Meta`).
    id: u64,
    /// The render extent.
    width: u32,
    height: u32,
    padded_width: u32,
    /// Each pass slot's extent (the render extent at a model resolution of 1), fixed for the pass's life: a
    /// change builds another pass (and so starts every history afresh).
    extents: [(u32, u32); neural_forge_protocol::MAX_PASSES],
    /// The resources of each extent below the render extent a pass ran at, built on first use.
    lanes: Vec<Lane>,
    /// `native_reduce.comp` and `native_enlarge.comp`, built with the first lane.
    resize: Option<[vk::Pipeline; 2]>,
    /// The edit cache's buffers where pass 1's network input features cannot hold them (pass 1 below the
    /// render extent), and their size.
    cache_scratch: Option<(OwnBuffer, u64)>,
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
    /// Built with its own GPU timers (the split hold's; the hold inside DLSS's buffer times its slot).
    timed: bool,
    /// The initialisation's buffer (`commands[2]`) is recorded.
    init_recorded: bool,
    /// The [`Built::generation`] `commands` were recorded for, and whether the initialisation ran.
    recorded: Option<u64>,
    initialised: bool,
    /// The frame counter of the holds that ran the network, its seed since the last history reset,
    /// and the last frame's jitter.
    frames: u64,
    seed: u32,
    last_jitter: Option<[f32; 2]>,
    pub(crate) gap: HistoryGap<(u32, u32)>,
    /// The network whose answers the history holds ([`Built::network`]), once one has run.
    history_network: Option<u32>,
    /// Passes 2 and on, built on the first frame with more than one.
    more: Option<MorePasses>,
    /// The control mask's pipeline, its buffer (RGBA8 per pixel) and a stand-in for the depth while none is
    /// copied, built with the first recording.
    mask: Option<MaskPass>,
    /// The depth buffer the sets point binding 11 at (the stand-in's while none is copied).
    mask_depth: vk::Buffer,
    /// The colour of the edit (`native_colour.comp`): its two pipelines and the edit's low-pass, built the
    /// first time a pass reshapes its edit; `colour_wanted` says whether this frame records it
    /// ([`Self::prepare_colour`]) and `colour_recorded` whether the buffers were recorded with it.
    colour: Option<ColourPass>,
    colour_wanted: bool,
    colour_recorded: bool,
    /// The look-up table on the GPU (binding 13), the last pass's.
    lut: Option<LutBuffer>,
}

/// [`NativePass::colour`].
struct ColourPass {
    /// Phase 0 (the edit's low-pass) and 1 (the steps).
    pipelines: [vk::Pipeline; 2],
    low: OwnBuffer,
}

/// [`NativePass::lut`]: the table's entries (vec4 each) in a host-visible buffer, and the table it holds.
struct LutBuffer {
    buffer: HostBuffer,
    entries: usize,
    generation: u64,
    lut: Arc<super::colour::Lut>,
}

/// One extent below the render extent ([`NativePass::extents`]): what its passes read and write there.
struct Lane {
    width: u32,
    height: u32,
    /// The network's input at this extent (padded), written by `native_reduce.comp` from the render
    /// extent's encoded image every pass that runs here.
    encoded: OwnImage,
    /// The motion (as the render extent's: f32 x and y, or DLSS's packed halves with `MOTION_HALF`), the
    /// depth (f32 device depth) and the control mask at this extent, and the network's answer (padded).
    motion: OwnBuffer,
    depth: OwnBuffer,
    mask: OwnBuffer,
    answer: OwnBuffer,
}

impl Lane {
    /// # Safety
    /// `device` is live.
    unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32) -> Option<Self> {
        let (pw, ph) = super::padded(width, height);
        let texels = u64::from(width) * u64::from(height);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: forwarded; each is destroyed below if another fails.
        unsafe {
            let encoded = OwnImage::new(device, instance, physical_device, pw, ph, vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);
            let motion = OwnBuffer::new(device, instance, physical_device, texels * 8, usage);
            let depth = OwnBuffer::new(device, instance, physical_device, texels * 4, usage);
            let mask = OwnBuffer::new(device, instance, physical_device, texels * 4, usage);
            let answer = OwnBuffer::new(device, instance, physical_device, u64::from(pw) * u64::from(ph) * 8, usage);
            match (encoded, motion, depth, mask, answer) {
                (Some(encoded), Some(motion), Some(depth), Some(mask), Some(answer)) => {
                    Some(made(Self { width, height, encoded, motion, depth, mask, answer }))
                }
                (e, m, d, k, a) => {
                    if let Some(e) = e {
                        e.destroy(device);
                    }
                    for b in [m, d, k, a].into_iter().flatten() {
                        b.destroy(device);
                    }
                    None
                }
            }
        }
    }

    /// # Safety
    /// Nothing submitted may still use it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            self.encoded.destroy(device);
            for b in [&self.motion, &self.depth, &self.mask, &self.answer] {
                b.destroy(device);
            }
        }
        destroyed(1);
    }
}

/// [`NativePass::mask`].
struct MaskPass {
    pipeline: vk::Pipeline,
    buffer: OwnBuffer,
    no_depth: OwnBuffer,
}

// SAFETY: plain handles and the layer's own mappings, only used behind the device's `State` mutex.
unsafe impl Send for NativePass {}

impl NativePass {
    /// With `motion_half` the frame shaders read DLSS's packed vectors (`MOTION_HALF`), as 3.2.0's
    /// did: the edit cache off (`NEURAL_FORGE_EDIT_CACHE=off`). Without, the accumulated field.
    ///
    /// # Safety
    /// `device` is live; `family` is the queue family the frames are submitted on.
    #[cfg(test)]
    pub(crate) unsafe fn build(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        family: u32,
        width: u32,
        height: u32,
        motion_half: bool,
    ) -> Option<Self> {
        // SAFETY: forwarded.
        unsafe { Self::build_failing_at(device, instance, physical_device, family, width, height, None, motion_half) }
    }

    /// [`Self::build`] with each pass slot at its own extent (`extents`, [`pass_extents`]): pass 1's history
    /// at its extent; the rest is built as each pass first runs.
    ///
    /// # Safety
    /// As [`Self::build`].
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build_at(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        family: u32,
        width: u32,
        height: u32,
        extents: [(u32, u32); neural_forge_protocol::MAX_PASSES],
        motion_half: bool,
        timed: bool,
    ) -> Option<Self> {
        // SAFETY: forwarded.
        let pass = unsafe { Self::build_inner(device, instance, physical_device, family, width, height, extents, None, motion_half, timed) };
        if let Some(p) = pass.as_ref().filter(|p| p.extents[0] != (width, height)) {
            crate::log!(
                "[native] model resolution: pass 1 runs the network at {}x{} of the render extent {width}x{height} ({:.0}%)",
                p.extents[0].0,
                p.extents[0].1,
                100.0 * f64::from(p.extents[0].0) / f64::from(width)
            );
        }
        pass
    }

    /// [`Self::build`], failing on purpose at creation step `fail_at` (`0..BUILD_STEPS`) when given.
    /// Built piece by piece; [`PassPartial`] destroys whatever exists on an early return.
    ///
    /// # Safety
    /// As [`Self::build`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn build_failing_at(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        family: u32,
        width: u32,
        height: u32,
        fail_at: Option<usize>,
        motion_half: bool,
    ) -> Option<Self> {
        let extents = [(width, height); neural_forge_protocol::MAX_PASSES];
        // SAFETY: forwarded.
        unsafe { Self::build_inner(device, instance, physical_device, family, width, height, extents, fail_at, motion_half, true) }
    }

    /// [`Self::build_failing_at`] with each pass slot's extent.
    ///
    /// # Safety
    /// As [`Self::build`].
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn build_inner(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        family: u32,
        width: u32,
        height: u32,
        extents: [(u32, u32); neural_forge_protocol::MAX_PASSES],
        fail_at: Option<usize>,
        motion_half: bool,
        timed: bool,
    ) -> Option<Self> {
        let (padded_width, padded_height) = super::padded(width, height);
        let binding = |n: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::default().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
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
            // The control mask, and the depth copy it reads (`native_mask.comp`).
            binding(10, vk::DescriptorType::STORAGE_BUFFER),
            binding(11, vk::DescriptorType::STORAGE_BUFFER),
            // At the render extent, for a pass below it (`native_reduce.comp`, `native_enlarge.comp`): the
            // encoded image, the motion, the depth copy and the answer.
            binding(12, vk::DescriptorType::STORAGE_IMAGE),
            binding(13, vk::DescriptorType::STORAGE_BUFFER),
            binding(14, vk::DescriptorType::STORAGE_BUFFER),
            binding(15, vk::DescriptorType::STORAGE_BUFFER),
            // The colour of the edit, at the render extent: the edit's low-pass and the look-up table
            // (`native_colour.comp`).
            binding(16, vk::DescriptorType::STORAGE_BUFFER),
            binding(17, vk::DescriptorType::STORAGE_BUFFER),
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
            p.set_layout = made(device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?);
            step()?;
            p.pipeline_layout =
                made(device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&p.set_layout)), None).ok()?);
            step()?;
            // MOTION_HALF (constant_id 0), a VkBool32.
            let half = u32::from(motion_half).to_le_bytes();
            let entry = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
            let spec = vk::SpecializationInfo::default().map_entries(&entry).data(&half);
            p.preprocess = made(hdr::compute_pipeline_with(device, p.pipeline_layout, PREPROCESS_SPV, Some(&spec))?);
            step()?;
            p.composite = made(hdr::compute_pipeline_with(device, p.pipeline_layout, COMPOSITE_SPV, Some(&spec))?);
            step()?;
            p.sampler = made(
                device
                    .create_sampler(
                        &vk::SamplerCreateInfo::default()
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
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 * IMAGES_PER_SET },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 * BUFFERS_PER_SET },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 },
            ];
            step()?;
            p.pool = made(device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes), None).ok()?);
            let layouts = [p.set_layout, p.set_layout];
            step()?;
            // Freed with the pool.
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(p.pool).set_layouts(&layouts)).ok()?;
            for _ in 0..2 {
                step()?;
                // Pass 1's history, at its extent.
                p.history.push(made(OwnImage::new(device, instance, physical_device, extents[0].0, extents[0].1, vk::ImageUsageFlags::SAMPLED)?));
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
            p.answer = Some(made(OwnBuffer::new(
                device,
                instance,
                physical_device,
                answer_bytes,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?));
            step()?;
            p.state =
                Some(made(OwnBuffer::new(device, instance, physical_device, 32, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?));
            for _ in 0..2 {
                step()?;
                // TRANSFER_DST: inside DLSS's buffer the hold writes them there (`vkCmdUpdateBuffer`).
                p.params.push(made(own_host_buffer_with(
                    device,
                    instance,
                    physical_device,
                    PARAMS_BYTES,
                    vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                )?));
            }
            step()?;
            p.command_pool = made(
                device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .ok()?,
            );
            let alloc = vk::CommandBufferAllocateInfo::default().command_pool(p.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(3);
            step()?;
            // Freed with the command pool.
            let cmds = crate::loader_data::allocate_commands(device, &alloc).ok()?;
            debug_assert_eq!(steps, BUILD_STEPS);
            // Optional (`None` where timestamps are not supported): nothing fails from here on.
            // Only made when kept (inside DLSS's buffer the hold times its slot): a timer has no destructor.
            let timer = || if timed { crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family).map(made) } else { None };
            let timers = [timer(), timer()];
            let take_buffer = |b: &mut Option<OwnBuffer>| b.take().expect("built above");
            let (mvec, acc, answer, state) = (take_buffer(&mut p.mvec), take_buffer(&mut p.acc), take_buffer(&mut p.answer), take_buffer(&mut p.state));
            let mut history = std::mem::take(&mut p.history).into_iter();
            let history = [history.next().expect("built above"), history.next().expect("built above")];
            let mut params = std::mem::take(&mut p.params).into_iter();
            let params = [params.next().expect("built above"), params.next().expect("built above")];
            Some(Self {
                id: NEXT_PASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                width,
                height,
                padded_width,
                extents,
                lanes: Vec::new(),
                resize: None,
                cache_scratch: None,
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
                timed,
                init_recorded: false,
                recorded: None,
                initialised: false,
                frames: 0,
                seed: 0,
                last_jitter: None,
                more: None,
                mask: None,
                mask_depth: vk::Buffer::null(),
                colour: None,
                colour_wanted: false,
                colour_recorded: false,
                lut: None,
                gap: HistoryGap::default(),
                history_network: None,
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    /// Built for the render extent `width` x `height` with every pass slot at `extents`.
    pub(crate) fn matches_at(&self, width: u32, height: u32, extents: &[(u32, u32); neural_forge_protocol::MAX_PASSES]) -> bool {
        self.matches(width, height) && self.extents == *extents
    }

    /// The edit cache's buffers: pass 1's network input features when they can hold `bytes` (as at a model
    /// resolution of 1), else a buffer of the pass's own, built on first use. `None` when it cannot be.
    ///
    /// # Safety
    /// `device` is live; nothing of the pass is pending.
    pub(crate) unsafe fn cache_scratch(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        features: (vk::Buffer, u64),
        bytes: u64,
    ) -> Option<(vk::Buffer, u64)> {
        if features.1 >= bytes {
            return Some(features);
        }
        if self.cache_scratch.as_ref().is_none_or(|(_, n)| *n < bytes) {
            if self.cache_scratch.is_some() && !super::may_rewrite("the edit cache's buffers") {
                return None;
            }
            if let Some((old, _)) = self.cache_scratch.take() {
                // SAFETY: idle (the caller's contract).
                unsafe { old.destroy(device) };
                destroyed(1);
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            // SAFETY: forwarded.
            self.cache_scratch = unsafe { OwnBuffer::new(device, instance, physical_device, bytes, usage) }.map(|b| (made(b), bytes));
        }
        self.cache_scratch.as_ref().map(|(b, n)| (b.buffer, *n))
    }

    /// Builds the [`Lane`] of every extent below the render extent the first `count` passes run at, and the
    /// two pipelines they use. `false` when something could not be built (the frame then misses). Nothing to
    /// do at a model resolution of 1.
    ///
    /// # Safety
    /// `device` is live; nothing of this pass is pending.
    pub(crate) unsafe fn ensure_lanes(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, count: usize) -> bool {
        let render = (self.width, self.height);
        let wanted: Vec<(u32, u32)> = graph_sizes(&self.extents, count).into_iter().filter(|&e| e != render).collect();
        if wanted.is_empty() {
            return true;
        }
        if self.resize.is_none() {
            let half = u32::from(self.motion_half).to_le_bytes();
            let entry = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
            let spec = vk::SpecializationInfo::default().map_entries(&entry).data(&half);
            // SAFETY: `device` is live; the layout is this pass's.
            let reduce = unsafe { hdr::compute_pipeline_with(device, self.pipeline_layout, REDUCE_SPV, Some(&spec)) };
            // SAFETY: as above.
            let enlarge = unsafe { hdr::compute_pipeline_with(device, self.pipeline_layout, ENLARGE_SPV, Some(&spec)) };
            match (reduce, enlarge) {
                (Some(r), Some(e)) => self.resize = Some([made(r), made(e)]),
                (r, e) => {
                    // SAFETY: never used.
                    unsafe {
                        for p in [r, e].into_iter().flatten() {
                            device.destroy_pipeline(p, None);
                        }
                    }
                    return false;
                }
            }
        }
        for e in wanted {
            if self.lane(e).is_some() {
                continue;
            }
            // SAFETY: forwarded.
            let Some(lane) = (unsafe { Lane::build(device, instance, physical_device, e.0, e.1) }) else { return false };
            self.lanes.push(lane);
            // Every set is written again with the lane's resources.
            self.recorded = None;
            if let Some(m) = self.more.as_mut() {
                m.recorded = None;
            }
        }
        true
    }

    fn lane(&self, extent: (u32, u32)) -> Option<&Lane> {
        self.lanes.iter().find(|l| (l.width, l.height) == extent)
    }

    /// The frame shaders' state buffer (binding 5: `ok[2]`, then `cut`, which the scene-cut detector writes
    /// in the capture; then the inline hold guard's `last` and the detector's `changing`).
    pub(crate) fn state_buffer(&self) -> vk::Buffer {
        self.state.buffer
    }

    /// The last network frame was a scene cut, which its shaders ran without history (seed 0): the next
    /// frame's seed counts on from there, as after a reset on the CPU.
    pub(crate) fn restart_seed(&mut self) {
        self.seed = 0;
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
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        built: &Built,
        graph: vk::CommandBuffer,
        encoded: vk::ImageView,
        exposure: vk::Buffer,
        depth: Option<vk::Buffer>,
    ) -> bool {
        // SAFETY: forwarded.
        unsafe { self.record_graphs(device, instance, physical_device, &[Graph { built: *built, graph }], encoded, exposure, depth) }
    }

    /// [`Self::record`] with the network built for each extent the passes run at (`graphs`, one build: one
    /// generation). Pass 1's extent must be among them; a later pass whose extent is not is not recorded (and
    /// does not run).
    ///
    /// # Safety
    /// As [`Self::record`]; every graph is recorded for this pass's queue family.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record_graphs(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        graphs: &[Graph],
        encoded: vk::ImageView,
        exposure: vk::Buffer,
        depth: Option<vk::Buffer>,
    ) -> bool {
        let Some(first) = Graph::of(graphs, self.extents[0]) else { return false };
        let generation = first.built.generation;
        // SAFETY: forwarded; nothing of this pass is pending.
        if !unsafe { self.ensure_mask(device, instance, physical_device) } {
            return false;
        }
        let depth = depth.unwrap_or_else(|| self.mask.as_ref().map_or(vk::Buffer::null(), |m| m.no_depth.buffer));
        // SAFETY: forwarded; nothing of this pass is pending. Without its pipelines the edit is left as it is.
        if self.colour_wanted && !unsafe { self.ensure_colour(device, instance, physical_device) } {
            crate::log!("[colour] the colour-of-the-edit pipelines could not be built; the edit is left as the network made it");
            self.colour_wanted = false;
        }
        let more_done = self.more.as_ref().is_none_or(|m| m.recorded == Some(generation));
        if self.recorded == Some(generation) && more_done && self.mask_depth == depth && self.colour_recorded == self.colour_wanted {
            return true;
        }
        // Every set is written again below: not while a hold recorded into DLSS's buffer with them may run.
        if !super::may_rewrite("the network frame's descriptor sets") {
            return false;
        }
        if self.colour_recorded != self.colour_wanted {
            self.colour_recorded = self.colour_wanted;
            if let Some(m) = self.more.as_mut() {
                m.recorded = None;
            }
        }
        if self.mask_depth != depth {
            // Every set points at the depth: all are written again.
            self.mask_depth = depth;
            if let Some(m) = self.more.as_mut() {
                m.recorded = None;
            }
        }
        // Pass 1 below the render extent needs its lane (`ensure_lanes`).
        let render = (self.width, self.height);
        if self.extents[0] != render && (self.lane(self.extents[0]).is_none() || self.resize.is_none()) {
            return false;
        }
        for p in 0..2 {
            // SAFETY: forwarded; the sets are not in use.
            unsafe { self.write_set(device, self.sets[p], p, self.params[p].buffer, &self.history, &first.built, self.extents[0], encoded, exposure) };
        }
        let ok = (0..2).all(|p| {
            // SAFETY: forwarded from this function's contract.
            unsafe { self.record_frame(device, self.commands[p], self.sets[p], self.timers[p].as_ref(), first, self.extents[0], true) }.is_ok()
        });
        // SAFETY: as above.
        // The initialisation's buffer is the same whatever the sets point at: recorded once (it may have been
        // submitted in front of a hold recorded into DLSS's buffer since).
        let ok = ok && (self.init_recorded || unsafe { self.record_init(device) }.is_ok());
        self.init_recorded |= ok;
        let mut recorded_pass = vec![None; MORE];
        let more_ok = match &self.more {
            None => true,
            Some(m) => (0..MORE).all(|k| {
                let Some(history) = &m.history[k] else { return true };
                let extent = self.extents[k + 1];
                // A pass whose extent has no graph in this build, or no lane yet, is not recorded.
                let Some(g) = Graph::of(graphs, extent) else { return true };
                if extent != render && self.lane(extent).is_none() {
                    return true;
                }
                // The control mask is built by the first pass at each extent.
                let mask = !self.extents[..=k].contains(&extent);
                let ok = (0..2).all(|p| {
                    // SAFETY: as above: nothing of this pass is pending.
                    unsafe {
                        self.write_set(device, m.sets[k][p], p, m.params[k][p].buffer, history, &g.built, extent, encoded, exposure);
                        self.record_frame(device, m.commands[k][p], m.sets[k][p], None, g, extent, mask)
                    }
                    .is_ok()
                });
                if ok {
                    recorded_pass[k] = Some(generation);
                }
                ok
            }),
        };
        self.recorded = ok.then_some(generation);
        if let Some(m) = self.more.as_mut() {
            m.recorded = more_ok.then_some(generation);
            m.recorded_pass = recorded_pass;
        }
        ok && more_ok
    }

    /// The colour of the edit for this frame: whether any of its passes reshapes its edit (then the frame
    /// buffers record `native_colour.comp` after each composite), and the look-up table, uploaded when it
    /// changed. A table whose buffer cannot be made is left out (logged).
    ///
    /// # Safety
    /// `device` is live; nothing of this pass is pending.
    pub(crate) unsafe fn prepare_colour(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        passes: &Passes,
        lut: Option<&Arc<super::colour::Lut>>,
    ) {
        let lut = lut.filter(|_| passes.colour.lut_strength > 0.0);
        match lut {
            None => {}
            Some(l) if self.lut.as_ref().is_some_and(|b| b.generation == l.generation) => {}
            // The table's buffer is read by every hold recorded with it: not changed while one may run.
            Some(_) if !super::may_rewrite("the look-up table") => {}
            Some(l) => {
                let entries = l.entries();
                if self.lut.as_ref().is_none_or(|b| b.entries != entries.len()) {
                    if let Some(old) = self.lut.take() {
                        // SAFETY: not pending (contract).
                        unsafe { old.buffer.destroy(device) };
                        destroyed(1);
                        // The sets point at it: written and recorded again.
                        self.recorded = None;
                    }
                    let bytes = (entries.len() * 16) as u64;
                    match own_host_buffer_with(device, instance, physical_device, bytes, vk::BufferUsageFlags::STORAGE_BUFFER) {
                        Some(buffer) => {
                            made(());
                            self.lut = Some(LutBuffer { buffer, entries: entries.len(), generation: 0, lut: Arc::clone(l) });
                            self.recorded = None;
                        }
                        None => crate::log!("[colour] no buffer for the look-up table; it is left out"),
                    }
                }
                if let Some(b) = self.lut.as_mut() {
                    // SAFETY: host-coherent, mapped and sized for `entries`; not in use (contract).
                    unsafe { std::ptr::copy_nonoverlapping(entries.as_ptr().cast::<u8>(), b.buffer.ptr, entries.len() * 16) };
                    b.generation = l.generation;
                    b.lut = Arc::clone(l);
                }
            }
        }
        let loaded = lut.is_some() && self.lut.is_some();
        self.colour_wanted = super::colour::wanted(&passes.colour, passes.count, loaded);
    }

    /// Builds [`ColourPass`] once. `false` when it could not be built.
    ///
    /// # Safety
    /// `device` is live; nothing of this pass is pending.
    unsafe fn ensure_colour(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
        if self.colour.is_some() {
            return true;
        }
        let pipeline = |phase: u32| {
            let data = phase.to_le_bytes();
            let entry = [vk::SpecializationMapEntry { constant_id: 1, offset: 0, size: 4 }];
            let spec = vk::SpecializationInfo::default().map_entries(&entry).data(&data);
            // SAFETY: `device` is live; the layout is this pass's.
            unsafe { hdr::compute_pipeline_with(device, self.pipeline_layout, COLOUR_SPV, Some(&spec)) }
        };
        let (low_pass, steps) = (pipeline(0), pipeline(1));
        // SAFETY: as above.
        let low = unsafe {
            OwnBuffer::new(device, instance, physical_device, u64::from(self.width) * u64::from(self.height) * 16, vk::BufferUsageFlags::STORAGE_BUFFER)
        };
        match (low_pass, steps, low) {
            (Some(a), Some(b), Some(low)) => {
                made(());
                made(());
                made(());
                self.colour = Some(ColourPass { pipelines: [a, b], low });
                // The sets point binding 12 at the low-pass: written and recorded again.
                self.recorded = None;
                if let Some(m) = self.more.as_mut() {
                    m.recorded = None;
                }
                true
            }
            (a, b, low) => {
                // SAFETY: never used.
                unsafe {
                    for p in [a, b].into_iter().flatten() {
                        device.destroy_pipeline(p, None);
                    }
                    if let Some(l) = low {
                        l.destroy(device);
                    }
                }
                false
            }
        }
    }

    /// Builds [`MaskPass`] once. `false` when it could not be built (the frame then misses).
    ///
    /// # Safety
    /// `device` is live; nothing of this pass is pending.
    unsafe fn ensure_mask(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
        if self.mask.is_some() {
            return true;
        }
        let half = u32::from(self.motion_half).to_le_bytes();
        let entry = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
        let spec = vk::SpecializationInfo::default().map_entries(&entry).data(&half);
        // SAFETY: `device` is live; the layout is this pass's.
        let Some(pipeline) = (unsafe { hdr::compute_pipeline_with(device, self.pipeline_layout, MASK_SPV, Some(&spec)) }) else { return false };
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        // SAFETY: as above.
        let buffer = unsafe { OwnBuffer::new(device, instance, physical_device, u64::from(self.width) * u64::from(self.height) * 4, usage) };
        // SAFETY: as above.
        let no_depth = unsafe { OwnBuffer::new(device, instance, physical_device, 16, usage) };
        match (buffer, no_depth) {
            (Some(buffer), Some(no_depth)) => {
                made(());
                made(());
                made(());
                self.mask = Some(MaskPass { pipeline, buffer, no_depth });
                true
            }
            (a, b) => {
                // SAFETY: never used.
                unsafe {
                    for x in [a, b].into_iter().flatten() {
                        x.destroy(device);
                    }
                    device.destroy_pipeline(pipeline, None);
                }
                false
            }
        }
    }

    /// Points `set` at this frame's resources for parity `p`, with `params` as its parameters and `history`
    /// as its history images.
    ///
    /// # Safety
    /// As [`Self::record`]; `set` is not in use.
    /// `extent` is the pass's: below the render extent it reads and writes its [`Lane`]'s resources, and the
    /// render extent's are bound beside them (bindings 12 to 15) for the reduce and the enlarge.
    #[allow(clippy::too_many_arguments)]
    unsafe fn write_set(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        p: usize,
        params: vk::Buffer,
        history: &[OwnImage; 2],
        built: &Built,
        extent: (u32, u32),
        encoded: vk::ImageView,
        exposure: vk::Buffer,
    ) {
        let f = built.frame;
        let general = |view| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
        let lane = self.lane(extent).filter(|_| extent != (self.width, self.height));
        let full_motion = if self.motion_half { self.mvec.buffer } else { self.acc.buffer };
        let enc = general(lane.map_or(encoded, |l| l.encoded.view));
        let full_enc = general(encoded);
        let next = general(history[1 - p].view);
        let prev = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: history[p].view, image_layout: vk::ImageLayout::GENERAL }];
        let buf = |buffer, range| [vk::DescriptorBufferInfo { buffer, offset: 0, range }];
        let mask = lane.map_or_else(|| self.mask.as_ref().map_or(vk::Buffer::null(), |m| m.buffer.buffer), |l| l.mask.buffer);
        let (mask_buf, depth_buf) = (buf(mask, vk::WHOLE_SIZE), buf(lane.map_or(self.mask_depth, |l| l.depth.buffer), vk::WHOLE_SIZE));
        let (full_motion_buf, full_depth_buf, full_answer_buf) =
            (buf(full_motion, vk::WHOLE_SIZE), buf(self.mask_depth, vk::WHOLE_SIZE), buf(self.answer.buffer, vk::WHOLE_SIZE));
        // The colour of the edit's buffers, or the mask's 16-byte stand-in until they exist (nothing reads them).
        let stand_in = self.mask.as_ref().map_or(vk::Buffer::null(), |m| m.no_depth.buffer);
        let low_buf = buf(self.colour.as_ref().map_or(stand_in, |c| c.low.buffer), vk::WHOLE_SIZE);
        let lut_buf = buf(self.lut.as_ref().map_or(stand_in, |l| l.buffer.buffer), vk::WHOLE_SIZE);
        let (mv, feat, head, params, state, expo, ans) = (
            buf(lane.map_or(full_motion, |l| l.motion.buffer), vk::WHOLE_SIZE),
            buf(f.features, vk::WHOLE_SIZE),
            buf(f.head, vk::WHOLE_SIZE),
            buf(params, PARAMS_BYTES),
            buf(self.state.buffer, vk::WHOLE_SIZE),
            buf(exposure, 4),
            buf(lane.map_or(self.answer.buffer, |l| l.answer.buffer), vk::WHOLE_SIZE),
        );
        let w = |binding, ty| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
        let writes = [
            w(0, vk::DescriptorType::STORAGE_IMAGE).image_info(&enc),
            w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&prev),
            w(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&mv),
            w(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&feat),
            w(4, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&params),
            w(5, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&state),
            w(6, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&expo),
            w(7, vk::DescriptorType::STORAGE_IMAGE).image_info(&next),
            w(8, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ans),
            w(9, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&head),
            w(10, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&mask_buf),
            w(11, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&depth_buf),
            w(12, vk::DescriptorType::STORAGE_IMAGE).image_info(&full_enc),
            w(13, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&full_motion_buf),
            w(14, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&full_depth_buf),
            w(15, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&full_answer_buf),
            w(16, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&low_buf),
            w(17, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&lut_buf),
        ];
        // SAFETY: the set is not in use (contract); every handle is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Builds [`MorePasses`] once (its command buffers from this pass's pool) and the history of every pass
    /// up to `count`. `false` when that could not be built: the frame then runs one pass.
    ///
    /// # Safety
    /// `device` is live; `family` is this pass's family; nothing of this pass is pending.
    unsafe fn ensure_more(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, count: usize) -> bool {
        // SAFETY: forwarded.
        if !unsafe { self.ensure_more_base(device, instance, physical_device, family) } {
            return false;
        }
        let extents = self.extents;
        let Some(m) = self.more.as_mut() else { return false };
        for k in 0..count.saturating_sub(1).min(MORE) {
            if m.history[k].is_some() {
                continue;
            }
            // Pass k + 2's history, at its extent.
            let (width, height) = extents[k + 1];
            // SAFETY: `device` is live.
            let a = unsafe { OwnImage::new(device, instance, physical_device, width, height, vk::ImageUsageFlags::SAMPLED) };
            // SAFETY: as above.
            let b = unsafe { OwnImage::new(device, instance, physical_device, width, height, vk::ImageUsageFlags::SAMPLED) };
            match (a, b) {
                (Some(a), Some(b)) => {
                    made(());
                    made(());
                    m.history[k] = Some([a, b]);
                    m.fresh[k] = true;
                    m.last_run[k] = None;
                    // Its sets are written and its buffers recorded with the next `record`.
                    m.recorded = None;
                }
                (a, b) => {
                    // SAFETY: never used.
                    unsafe {
                        for i in [a, b].into_iter().flatten() {
                            i.destroy(device);
                        }
                    }
                    return false;
                }
            }
        }
        true
    }

    /// [`Self::ensure_more`]'s part built once, for every pass.
    ///
    /// # Safety
    /// As [`Self::ensure_more`].
    unsafe fn ensure_more_base(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32) -> bool {
        if self.more.is_some() {
            return true;
        }
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 * IMAGES_PER_SET * MORE as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 * MORE as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 * BUFFERS_PER_SET * MORE as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 * MORE as u32 },
        ];
        // SAFETY: valid create info.
        let Ok(pool) = (unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2 * MORE as u32).pool_sizes(&sizes), None) })
        else {
            return false;
        };
        let mut params: Vec<[HostBuffer; 2]> = Vec::with_capacity(MORE);
        // Whatever exists so far, destroyed on a failure.
        let fail = |params: Vec<[HostBuffer; 2]>, saved: Option<OwnBuffer>| {
            // SAFETY: none of it was used.
            unsafe {
                for b in params.iter().flatten() {
                    b.destroy(device);
                }
                if let Some(s) = saved {
                    s.destroy(device);
                }
                device.destroy_descriptor_pool(pool, None);
            }
            false
        };
        for _ in 0..MORE {
            let one = || {
                own_host_buffer_with(device, instance, physical_device, PARAMS_BYTES, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            };
            match (one(), one()) {
                (Some(a), Some(b)) => params.push([a, b]),
                (a, b) => {
                    // SAFETY: never used.
                    unsafe {
                        for x in [a, b].into_iter().flatten() {
                            x.destroy(device);
                        }
                    }
                    return fail(params, None);
                }
            }
        }
        let bytes = u64::from(self.padded_width) * u64::from(super::padded(self.width, self.height).1) * 8;
        // SAFETY: `device` is live.
        let Some(saved) =
            (unsafe { OwnBuffer::new(device, instance, physical_device, bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST) })
        else {
            return fail(params, None);
        };
        let layouts = vec![self.set_layout; 2 * MORE];
        // SAFETY: the pool has room for them (freed with it).
        let Ok(sets) = (unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) })
        else {
            return fail(params, Some(saved));
        };
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2 * MORE as u32 + 3);
        // SAFETY: the pass's own pool (freed with it).
        let Ok(cmds) = (unsafe { crate::loader_data::allocate_commands(device, &alloc) }) else {
            return fail(params, Some(saved));
        };
        // Only made when kept: a timer has no destructor.
        let timer = if self.timed { crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family) } else { None };
        made(());
        made(());
        for _ in 0..2 * MORE + usize::from(timer.is_some()) {
            made(());
        }
        self.more = Some(MorePasses {
            pool,
            params,
            sets: sets.chunks(2).map(|s| [s[0], s[1]]).collect(),
            commands: cmds[..2 * MORE].chunks(2).map(|c| [c[0], c[1]]).collect(),
            history: (0..MORE).map(|_| None).collect(),
            feedback: None,
            fresh: vec![false; MORE],
            last_run: vec![None; MORE],
            copies: [cmds[2 * MORE], cmds[2 * MORE + 1], cmds[2 * MORE + 2]],
            saved,
            timer,
            recorded: None,
            recorded_pass: vec![None; MORE],
        });
        true
    }

    /// Records [`MorePasses`]' copies for this hold into its own buffers: `X`, `Y` and `R` over `encoded`
    /// (the HDR pass's image, in `GENERAL`), with the timer's start in `X` and its end in `R`.
    ///
    /// # Safety
    /// [`MorePasses`] is built and none of its copies is pending.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_copies(
        &mut self,
        device: &ash::Device,
        encoded: vk::Image,
        encoded_view: vk::ImageView,
        region: vk::BufferImageCopy,
        count: usize,
        feedback: f32,
    ) -> ash::prelude::VkResult<()> {
        // SAFETY: forwarded.
        if !unsafe { self.prepare_copies(device, encoded_view, feedback) } {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let Some(m) = &self.more else { return Ok(()) };
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let fresh = m.fresh.clone();
        // SAFETY: forwarded; every handle is live.
        unsafe {
            for (i, &cmd) in m.copies.iter().enumerate() {
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
                device.begin_command_buffer(cmd, &begin)?;
                self.record_copy(device, cmd, i, encoded, region, count, feedback, &fresh, i == 0 && m.timer.is_some(), i == 2 && m.timer.is_some());
                device.end_command_buffer(cmd)?;
            }
        }
        Ok(())
    }

    /// Builds the feedback blend the copies need (pass feedback below 1) and points its set at `encoded_view`
    /// and the answer (when they changed). `false` when it could not be built, or its set could not be
    /// written now (`super::may_rewrite`).
    ///
    /// # Safety
    /// [`MorePasses`] is built; nothing pending uses the feedback's set unless it is unchanged.
    pub(crate) unsafe fn prepare_copies(&mut self, device: &ash::Device, encoded_view: vk::ImageView, feedback: f32) -> bool {
        // Feedback below 1 blends the answer into the input (`native_feedback.comp`) where 1 copies it.
        if feedback >= 1.0 {
            return true;
        }
        if self.more.as_ref().is_some_and(|m| m.feedback.is_none()) {
            // SAFETY: `device` is live.
            let Some(f) = (unsafe { Feedback::build(device) }) else { return false };
            if let Some(m) = self.more.as_mut() {
                m.feedback = Some(f);
            }
        }
        let answer_buffer = self.answer.buffer;
        let Some(f) = self.more.as_mut().and_then(|m| m.feedback.as_mut()) else { return true };
        if f.bound == Some((encoded_view, answer_buffer)) {
            return true;
        }
        if !super::may_rewrite("the feedback blend's descriptor set") {
            return false;
        }
        let image = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: encoded_view, image_layout: vk::ImageLayout::GENERAL }];
        let answer = [vk::DescriptorBufferInfo { buffer: answer_buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&image),
            vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&answer),
        ];
        // SAFETY: the set is not in use (contract: it changes only here, when nothing that bound it may run).
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        f.bound = Some((encoded_view, answer_buffer));
        true
    }

    /// Copy `i` of [`MorePasses`] (`X` = 0, `Y` = 1, `R` = 2) into `cmd`, opened and closed by its barriers:
    /// `X` takes the histories of the passes still `fresh` out of `UNDEFINED`, saves the encoded capture and puts
    /// pass 1's answer over it; `Y` a later pass's answer; `R` the capture back. `start`/`end`: the timer's.
    ///
    /// # Safety
    /// [`MorePasses`] is built ([`Self::prepare_copies`] called for this hold); `cmd` is recording, outside a
    /// render pass instance; `encoded` is in `GENERAL`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record_copy(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        i: usize,
        encoded: vk::Image,
        region: vk::BufferImageCopy,
        count: usize,
        feedback: f32,
        fresh: &[bool],
        start: bool,
        end: bool,
    ) {
        let Some(m) = &self.more else { return };
        let fb = m.feedback.as_ref().filter(|_| feedback < 1.0);
        let push = FeedbackPush { width: region.image_extent.width, height: region.image_extent.height, feedback };
        let tc = vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER;
        // The blend in place of a copy of the answer over the input: a barrier from what came before in the
        // buffer (the save's read of the input), the dispatch.
        let mix = |cmd: vk::CommandBuffer, f: &Feedback| {
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            // SAFETY: recording; the set is bound to this hold's image and answer.
            unsafe {
                device.cmd_pipeline_barrier(cmd, tc, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[before], &[], &[]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, f.pipeline);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, f.layout, 0, &[f.set], &[]);
                device.cmd_push_constants(
                    cmd,
                    f.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<FeedbackPush>()),
                );
                device.cmd_dispatch(cmd, push.width.div_ceil(8), push.height.div_ceil(8), 1);
            }
        };
        // The history of a pass about to run for the first time, out of `UNDEFINED` (its frame does not read
        // it: `last_run` is unset).
        let to_general: Vec<vk::ImageMemoryBarrier> = (0..count.saturating_sub(1).min(MORE))
            .filter(|&k| fresh.get(k).copied().unwrap_or(false))
            .filter_map(|k| m.history[k].as_ref())
            .flatten()
            .map(|h| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(h.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            })
            .collect();
        let t = vk::PipelineStageFlags::TRANSFER;
        let open = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        let mid = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let close = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: the contract; every handle is live.
        unsafe {
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, tc, vk::DependencyFlags::empty(), &[open], &[], &[]);
            match i {
                0 => {
                    if !to_general.is_empty() {
                        device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &to_general,
                        );
                    }
                    if let Some(timer) = m.timer.as_ref().filter(|_| start) {
                        timer.record_start(device, cmd);
                    }
                    device.cmd_copy_image_to_buffer(cmd, encoded, vk::ImageLayout::GENERAL, m.saved.buffer, &[region]);
                    match fb {
                        Some(f) => mix(cmd, f),
                        None => {
                            device.cmd_pipeline_barrier(cmd, t, t, vk::DependencyFlags::empty(), &[mid], &[], &[]);
                            device.cmd_copy_buffer_to_image(cmd, self.answer.buffer, encoded, vk::ImageLayout::GENERAL, &[region]);
                        }
                    }
                }
                1 => match fb {
                    Some(f) => mix(cmd, f),
                    None => device.cmd_copy_buffer_to_image(cmd, self.answer.buffer, encoded, vk::ImageLayout::GENERAL, &[region]),
                },
                _ => device.cmd_copy_buffer_to_image(cmd, m.saved.buffer, encoded, vk::ImageLayout::GENERAL, &[region]),
            }
            device.cmd_pipeline_barrier(cmd, tc, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[close], &[], &[]);
            if let Some(timer) = m.timer.as_ref().filter(|_| end) {
                timer.record_end(device, cmd);
            }
        }
    }

    /// # Safety
    /// As [`Self::record`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_frame(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        timer: Option<&crate::gpu_timer::GpuTimer>,
        g: &Graph,
        extent: (u32, u32),
        mask: bool,
    ) -> ash::prelude::VkResult<()> {
        let graph = g.graph;
        // SAFETY: the layer's own buffer, not pending (forwarded); every handle is live.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))?;
            if let Some(timer) = timer {
                timer.record_start(device, cmd);
            }
            // `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`: everything but the network's launches (diagnostic: the
            // answer is then the head left from before).
            // The hold inside DLSS's buffer records the graph there, not from these buffers (never submitted:
            // no secondary to execute).
            let mut network = |cmd: vk::CommandBuffer| {
                if !skip_graph() && graph != vk::CommandBuffer::null() {
                    device.cmd_execute_commands(cmd, &[graph]);
                }
                Ok(())
            };
            // Executing the graph's secondary cannot fail.
            let _ = self.record_frame_body(device, cmd, set, g, extent, mask, &mut network);
            if let Some(timer) = timer {
                timer.record_end(device, cmd);
            }
            device.end_command_buffer(cmd)
        }
    }

    /// The commands of one pass of a network frame (`N`), into `cmd`: the opening barrier, the reduce
    /// below the render extent, the control mask, the preprocess, the network (`network`, which records it:
    /// the graph's secondary executed, or inside DLSS's buffer the graph itself), the composite, the enlarge,
    /// the colour of the edit, and the closing barrier. Binds this pass's pipelines and `set`. On an error of
    /// `network`, what came before it is recorded and nothing after.
    ///
    /// # Safety
    /// As [`Self::record`]; `cmd` is recording, outside a render pass instance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record_frame_body(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        g: &Graph,
        extent: (u32, u32),
        mask: bool,
        network: &mut dyn FnMut(vk::CommandBuffer) -> Result<(), String>,
    ) -> Result<(), String> {
        let f = g.built.frame;
        let (width, height) = extent;
        // Below the render extent: the reduce first, the enlarge last (`ensure_lanes` built them).
        let resize = self.lane(extent).filter(|_| extent != (self.width, self.height)).zip(self.resize);
        let compute = |src: vk::AccessFlags, dst: vk::AccessFlags| vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst);
        // SAFETY: the contract; every handle is live.
        unsafe {
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[set], &[]);
            if let Some((lane, [reduce, _])) = resize {
                // The network's input, motion and depth at this extent, from the render extent's (the lane's
                // encoded image is rewritten whole: from UNDEFINED).
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(lane.encoded.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce);
                let (pw, ph) = super::padded(width, height);
                device.cmd_dispatch(cmd, pw.div_ceil(8), ph.div_ceil(8), 1);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                    &[],
                    &[],
                );
            }
            // The first pass at each extent builds the control mask the passes there read (it writes nothing
            // while no rule is on).
            if let Some(m) = self.mask.as_ref().filter(|_| mask) {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, m.pipeline);
                device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ)],
                    &[],
                    &[],
                );
            }
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
            network(cmd)?;
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
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[set], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.composite);
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
            if let Some((_, [_, enlarge])) = resize {
                // The edit back to the render extent, into the answer buffer `W` and the next pass read.
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                    &[],
                    &[],
                );
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, enlarge);
                let (pw, ph) = super::padded(self.width, self.height);
                device.cmd_dispatch(cmd, pw.div_ceil(8), ph.div_ceil(8), 1);
            }
            // The colour of the edit (`native_colour.comp`), at the render extent on the pass's full-size answer
            // (after the enlarge below it): the edit's low-pass, then the steps in place. Each returns at once
            // for a pass whose parameters leave its edit alone. The history was written above, unreshaped.
            if let Some(c) = self.colour.as_ref().filter(|_| self.colour_recorded) {
                for pipeline in c.pipelines {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[compute(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                        &[],
                        &[],
                    );
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    device.cmd_dispatch(cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);
                }
            }
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
        }
        Ok(())
    }

    /// The previous frame's network GPU time, once its work has completed (the caller waited for it):
    /// pass 1's, plus the later passes' with their copies when it ran more than one.
    pub(crate) fn take_gpu_ms(&mut self, device: &ash::Device) -> Option<f32> {
        let more = self.more.as_mut().and_then(|m| m.timer.as_mut()).and_then(|t| {
            t.read_if_pending(device);
            t.take_reading()
        });
        let timer = self.timers[self.last_parity.take()?].as_mut()?;
        timer.read_if_pending(device);
        Some(timer.take_reading()? + more.unwrap_or(0.0))
    }

    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_init(&self, device: &ash::Device) -> ash::prelude::VkResult<()> {
        let cmd = self.commands[2];
        // SAFETY: as `record_frame`.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            self.record_init_body(device, cmd);
            device.end_command_buffer(cmd)
        }
    }

    /// The initialisation's commands, into `cmd`: the state buffer cleared (no history, no cut, no hold
    /// before), the history images from `UNDEFINED` to `GENERAL`.
    ///
    /// # Safety
    /// `cmd` is recording, outside a render pass instance; nothing recorded earlier in it or on its queue
    /// reads the history images after this.
    unsafe fn record_init_body(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let to_general = |image| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        };
        // Whatever came before on the queue (an earlier hold reading the state) is done before the clear.
        let before = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let filled = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
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
            device.cmd_fill_buffer(cmd, self.state.buffer, 0, vk::WHOLE_SIZE, 0);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[filled],
                &[],
                &[to_general(self.history[0].image), to_general(self.history[1].image)],
            );
        }
    }

    /// The command buffers for this frame, in submission order: the initialisation until one was
    /// submitted, then `N[parity]`. Writes the frame's parameters. Each buffer submitted is reported
    /// to [`Self::submitted`] (only then is the initialisation done and the timer pending).
    /// `mvec_scale` turns the copied motion vectors into render pixels ([`super::Target::mvec_scale`]).
    #[cfg(test)]
    pub(crate) fn frame(
        &mut self,
        built: &Built,
        history: bool,
        jitter: Option<[f32; 2]>,
        mvec_scale: [f32; 2],
        c: Conditioning,
    ) -> [Option<vk::CommandBuffer>; 2] {
        self.frame_passes(built, history, jitter, mvec_scale, &Passes::one(c)).0
    }

    /// [`Self::frame`] for `passes`: also writes each later pass's parameters (when [`MorePasses`] is
    /// built) and returns its `N[parity]` buffers, pass 2 first, to submit after pass 1's.
    #[cfg(test)]
    pub(crate) fn frame_passes(
        &mut self,
        built: &Built,
        history: bool,
        jitter: Option<[f32; 2]>,
        mvec_scale: [f32; 2],
        passes: &Passes,
    ) -> ([Option<vk::CommandBuffer>; 2], Vec<vk::CommandBuffer>) {
        let graph = [Graph { built: *built, graph: vk::CommandBuffer::null() }];
        self.frame_graphs(&graph, history, jitter, mvec_scale, passes)
    }

    /// [`Self::frame_passes`] with the network built for each extent the passes run at (`graphs`, as given
    /// to [`Self::record_graphs`]). A pass below the render extent gets its motion vectors' scale, the
    /// jitter's step and the control mask's motion threshold in its own pixels.
    pub(crate) fn frame_graphs(
        &mut self,
        graphs: &[Graph],
        history: bool,
        jitter: Option<[f32; 2]>,
        mvec_scale: [f32; 2],
        passes: &Passes,
    ) -> ([Option<vk::CommandBuffer>; 2], Vec<vk::CommandBuffer>) {
        let plan = self.frame_plan(graphs, history, jitter, mvec_scale, passes);
        for (buffer, params) in &plan.params {
            let target =
                self.params.iter().chain(self.more.iter().flat_map(|m| m.params.iter().flatten())).find(|b| b.buffer == *buffer).expect("one of the pass's");
            // SAFETY: host-coherent and mapped for PARAMS_BYTES; parity `p`'s previous use (two frames
            // ago) completed: the hold waits for the previous hold's work before it records anything.
            unsafe { std::ptr::copy_nonoverlapping(std::ptr::from_ref(params).cast::<u8>(), target.ptr, PARAMS_BYTES as usize) };
        }
        let p = plan.parity;
        let later = plan.passes.iter().skip(1).map(|r| self.more.as_ref().expect("a later pass").commands[r.index - 1][p]).collect();
        ([(!self.initialised).then_some(self.commands[2]), Some(self.commands[p])], later)
    }

    /// What one network frame runs, and the parameters each of its passes reads (written by the caller: on
    /// the host for the split hold, [`Self::frame_graphs`]; into DLSS's buffer for the hold inside it). Moves
    /// the frame count, the seed and the jitter on, and notes which passes ran.
    pub(crate) fn frame_plan(&mut self, graphs: &[Graph], history: bool, jitter: Option<[f32; 2]>, mvec_scale: [f32; 2], passes: &Passes) -> FramePlan {
        let p = (self.frames & 1) as usize;
        self.seed = if history { self.seed.wrapping_add(1) } else { 0 };
        let delta = match (self.last_jitter, jitter) {
            (Some(prev), Some(cur)) if history => [prev[0] - cur[0], prev[1] - cur[1]],
            _ => [0.0, 0.0],
        };
        self.last_jitter = jitter;
        let render = (self.width, self.height);
        let generation = Graph::of(graphs, self.extents[0]).map(|g| g.built.generation);
        // The passes this frame runs: pass 1, then each later one recorded for this build (the loop below). The
        // last of them gets the look-up table.
        let runs = 1 + self.more.as_ref().filter(|m| generation.is_some() && m.recorded == generation).map_or(0, |m| {
            (1..passes.count.min(neural_forge_protocol::MAX_PASSES)).take_while(|&k| m.history[k - 1].is_some() && m.recorded_pass[k - 1] == generation).count()
        });
        let lut = self.lut.as_ref().map(|l| &*l.lut);
        let recorded = self.colour_recorded;
        let colour = |k: usize| {
            if recorded {
                super::colour::PassColour::of(&passes.colour, k, k + 1 == runs, lut, false)
            } else {
                super::colour::PassColour::default()
            }
        };
        // A pass without history (pass 2 on whose previous frame it did not run) starts afresh: seed 0, no
        // jitter step, no blend toward its history.
        let params = |c: &Conditioning, history: bool, extent: (u32, u32), index: usize| {
            let f = Graph::of(graphs, extent).map_or_else(nn::Frame::default, |g| g.built.frame);
            let (width, height) = extent;
            let scaled = extent != render;
            // Model pixels per render pixel.
            let k = [width as f32 / render.0 as f32, height as f32 / render.1 as f32];
            let (delta, mvec_scale) =
                if scaled { ([delta[0] * k[0], delta[1] * k[1]], [mvec_scale[0] * k[0], mvec_scale[1] * k[1]]) } else { (delta, mvec_scale) };
            let threshold = if scaled { passes.mask.motion[0] * 0.5 * (k[0] + k[1]) } else { passes.mask.motion[0] };
            // A lane's depth is f32 device depth (`native_reduce.comp` converts the copy).
            let depth_kind = if scaled && passes.mask.depth_kind != 0 { 1 } else { passes.mask.depth_kind };
            Params {
                dims: [f.field_width, f.field_height, width, height],
                frame: [super::padded(width, height).0, if history { self.seed } else { 0 }, u32::from(history), p as u32],
                motion: [if history { delta[0] } else { 0.0 }, if history { delta[1] } else { 0.0 }, f.blend_scale, c.intensity],
                control: [c.style as f32, c.local_tone, c.local_structure, c.skin_structure],
                more: [if c.auto_mask { 1.0 } else { 0.0 }, mvec_scale[0], mvec_scale[1], 0.0],
                mask: [passes.mask.rules, depth_kind, u32::from(passes.mask.reversed), 0],
                sky: [passes.mask.sky_margin, 0.0, 0.0, 0.0],
                dark: [passes.mask.dark[0], passes.mask.dark[1], passes.mask.dark[2], 0.0],
                moving: [threshold, passes.mask.motion[1], passes.mask.motion[2], 0.0],
                full: [render.0, render.1, self.padded_width, passes.mask.depth_kind],
                ratio: [k[0], k[1], GUIDE_SIGMA, 0.0],
                colour: colour(index).colour,
                colour_flags: colour(index).flags,
                lut_scale: colour(index).lut_scale,
                lut_offset: colour(index).lut_offset,
            }
        };
        let mut plan = FramePlan {
            parity: p,
            params: vec![(self.params[p].buffer, params(&passes.each[0], history, self.extents[0], 0))],
            passes: vec![FramePass { index: 0, set: self.sets[p], extent: self.extents[0], mask: true }],
            fresh: 0,
        };
        let frame = self.frames;
        let extents = self.extents;
        if let Some(m) = self.more.as_mut().filter(|m| generation.is_some() && m.recorded == generation) {
            for (k, &extent) in extents.iter().enumerate().take(passes.count.min(neural_forge_protocol::MAX_PASSES)).skip(1) {
                if m.history[k - 1].is_none() || m.recorded_pass[k - 1] != generation {
                    break;
                }
                // Its own history only from the frame before, and only once its images are out of `UNDEFINED`
                // (inside DLSS's buffer that is known once a buffer that took them there ran).
                let own = history && frame > 0 && m.last_run[k - 1] == Some(frame - 1) && !m.fresh[k - 1];
                plan.params.push((m.params[k - 1][p].buffer, params(&passes.each[k], own, extent, k)));
                // The control mask is built by the first pass at each extent.
                plan.passes.push(FramePass { index: k, set: m.sets[k - 1][p], extent, mask: !extents[..k].contains(&extent) });
                if m.fresh[k - 1] {
                    plan.fresh |= 1 << (k - 1);
                }
                m.last_run[k - 1] = Some(frame);
            }
        }
        self.frames += 1;
        plan
    }

    /// A hold recorded into DLSS's buffer with this pass ran (read back at its buffer's release): the later
    /// passes' histories it took out of `UNDEFINED` (`fresh`, one bit per later pass) are in `GENERAL`.
    pub(crate) fn inline_ran(&mut self, fresh: u32) {
        if let Some(m) = self.more.as_mut() {
            for (k, f) in m.fresh.iter_mut().enumerate() {
                if fresh & (1 << k) != 0 {
                    *f = false;
                }
            }
        }
    }

    /// [`Self::id`].
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The initialisation's command buffer ([`Self::record_graphs`] records it), submitted once in front of
    /// the first buffer that runs a hold recorded with this pass (`inline::Inline::prologue`).
    pub(crate) fn init_command(&self) -> vk::CommandBuffer {
        self.commands[2]
    }

    /// The graph for each pass of `plan` and the barriers' and pipelines' state they need, recorded into
    /// `cmd` (DLSS's buffer): pass 1's frame, and with more passes the copies between them. `network`
    /// records one extent's network. Stops at the first `network` error (returned).
    ///
    /// # Safety
    /// As [`Self::record_frame_body`] and [`Self::record_copy`] (for more than one pass, [`Self::prepare_copies`]
    /// was called for this hold); `cmd` is DLSS's buffer, recording on this thread.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record_frame_inline(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        plan: &FramePlan,
        graphs: &[Graph],
        encoded: vk::Image,
        region: vk::BufferImageCopy,
        count: usize,
        feedback: f32,
        network: &mut dyn FnMut(vk::CommandBuffer, &Built) -> Result<(), String>,
    ) -> Result<(), String> {
        let fresh: Vec<bool> = (0..MORE).map(|k| plan.fresh & (1 << k) != 0).collect();
        let later = plan.passes.len() > 1;
        for (n, pass) in plan.passes.iter().enumerate() {
            let g = Graph::of(graphs, pass.extent).ok_or("no network for a pass's extent")?;
            if n > 0 {
                // `X` before pass 2, `Y` before the others.
                // SAFETY: forwarded.
                unsafe { self.record_copy(device, cmd, usize::from(n > 1), encoded, region, count, feedback, &fresh, false, false) };
            }
            let built = g.built;
            // SAFETY: forwarded.
            unsafe { self.record_frame_body(device, cmd, pass.set, g, pass.extent, pass.mask, &mut |cmd| network(cmd, &built)) }?;
        }
        if later {
            // `R`: the capture back for the write-back.
            // SAFETY: forwarded.
            unsafe { self.record_copy(device, cmd, 2, encoded, region, count, feedback, &fresh, false, false) };
        }
        Ok(())
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
            if let Some(m) = &self.mask {
                device.destroy_pipeline(m.pipeline, None);
                m.buffer.destroy(device);
                m.no_depth.destroy(device);
                destroyed(3);
            }
            if let Some(c) = &self.colour {
                for p in c.pipelines {
                    device.destroy_pipeline(p, None);
                }
                c.low.destroy(device);
                destroyed(3);
            }
            if let Some(l) = &self.lut {
                l.buffer.destroy(device);
                destroyed(1);
            }
            for lane in &self.lanes {
                lane.destroy(device);
            }
            for p in self.resize.iter().flatten() {
                device.destroy_pipeline(*p, None);
                destroyed(1);
            }
            if let Some((b, _)) = &self.cache_scratch {
                b.destroy(device);
                destroyed(1);
            }
            if let Some(m) = &self.more {
                // The command buffers go with the pass's pool.
                if let Some(t) = &m.timer {
                    t.destroy(device);
                }
                for b in m.params.iter().flatten() {
                    b.destroy(device);
                }
                for h in m.history.iter().flatten().flatten() {
                    h.destroy(device);
                    destroyed(1);
                }
                m.saved.destroy(device);
                if let Some(f) = &m.feedback {
                    f.destroy(device);
                }
                device.destroy_descriptor_pool(m.pool, None);
                destroyed(2 + 2 * MORE as isize + isize::from(m.timer.is_some()));
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

/// The hold guard inside DLSS's buffer (`preupscale_hold_guard.comp`): the first dispatch of every hold
/// recorded there, it checks on the GPU that the hold recorded before this one is the one that ran before it,
/// and otherwise clears what the hold would carry from the holds before (the network's history, the edit
/// cache's stored edit). Its one set points at the native pass's state buffer and the edit cache's counters
/// (or a stand-in).
pub(crate) struct HoldGuard {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Stands in for the counters while the edit cache is off.
    no_stats: OwnBuffer,
    /// What the set points at, with the native pass whose state buffer it is ([`NativePass::id`]): a rebuilt
    /// pass's buffer may come back under the old one's handle.
    bound: Option<(vk::Buffer, vk::Buffer, u64)>,
}

const GUARD_SPV: &[u8] = include_bytes!("../../shaders/preupscale_hold_guard.spv");

impl HoldGuard {
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<Self> {
        let b = |n: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(n)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [b(0), b(1)];
        // SAFETY: valid create infos; each object is destroyed below on a later failure.
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(4);
            let Ok(layout) =
                device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&[push]), None)
            else {
                device.destroy_descriptor_set_layout(set_layout, None);
                return None;
            };
            let undo = |pipeline: Option<vk::Pipeline>, pool: Option<vk::DescriptorPool>| {
                if let Some(p) = pool {
                    device.destroy_descriptor_pool(p, None);
                }
                if let Some(p) = pipeline {
                    device.destroy_pipeline(p, None);
                }
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
                None
            };
            let Some(pipeline) = hdr::compute_pipeline(device, layout, GUARD_SPV) else { return undo(None, None) };
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
            let Ok(pool) = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) else {
                return undo(Some(pipeline), None);
            };
            let Ok(sets) = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])) else {
                return undo(Some(pipeline), Some(pool));
            };
            let Some(no_stats) = OwnBuffer::new(device, instance, physical_device, 64, vk::BufferUsageFlags::STORAGE_BUFFER) else {
                return undo(Some(pipeline), Some(pool));
            };
            Some(Self { set_layout, layout, pipeline, pool, set: sets[0], no_stats, bound: None })
        }
    }

    /// Points the set at the `state` buffer of native pass `pass` ([`NativePass::id`]) and the edit cache's
    /// counters `stats` (when they changed). `false` when it cannot be written now ([`super::may_rewrite`]).
    ///
    /// # Safety
    /// Both buffers are live; nothing that may run uses the set unless it is unchanged.
    pub(crate) unsafe fn bind(&mut self, device: &ash::Device, state: vk::Buffer, pass: u64, stats: Option<vk::Buffer>) -> bool {
        let stats = stats.unwrap_or(self.no_stats.buffer);
        if self.bound == Some((state, stats, pass)) {
            return true;
        }
        if !super::may_rewrite("the hold guard's descriptor set") {
            return false;
        }
        let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (a, b) = (info(state), info(stats));
        fn w(set: vk::DescriptorSet, n: u32, i: &[vk::DescriptorBufferInfo; 1]) -> vk::WriteDescriptorSet<'_> {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i)
        }
        // SAFETY: the contract.
        unsafe { device.update_descriptor_sets(&[w(self.set, 0, &a), w(self.set, 1, &b)], &[]) };
        self.bound = Some((state, stats, pass));
        true
    }

    /// Records the guard for the hold numbered `seq` (1, 2, ... in recording order), between barriers that
    /// order it after everything before on the queue and before everything after.
    ///
    /// # Safety
    /// `cmd` is recording, outside a render pass instance; [`Self::bind`] returned `true` for this hold.
    pub(crate) unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, seq: u64) {
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let seq = (seq as u32).to_le_bytes();
        // SAFETY: the contract.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &seq);
            device.cmd_dispatch(cmd, 1, 1, 1);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
    }

    /// # Safety
    /// Nothing that may still run uses it.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; the set goes with the pool.
        unsafe {
            self.no_stats.destroy(device);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// The next [`NativePass::id`].
static NEXT_PASS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

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

/// The retry state of a device's native hold: the HDR pass, its auto-exposure, the native pass, the depth
/// copy, the edit cache and the scene-cut detector.
#[derive(Debug, Default)]
pub(crate) struct Retries {
    pub(crate) hdr: PassRetry,
    pub(crate) auto: PassRetry,
    pub(crate) pass: PassRetry,
    pub(crate) depth: PassRetry,
    pub(crate) edit: PassRetry,
    pub(crate) scene: PassRetry,
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
        Some(Stale::SceneCut(mean)) => crate::log!(
            "[native] resetting the history (scene cut: the picture's luma changed by {mean} of 255 on average, threshold {})",
            super::scene_cut::threshold()
        ),
        None => {}
    }
}

/// A network frame's history reset `stale` ([`HistoryGap::begin`]) after the pass's first frame (`frames`
/// before it): logged, and a scene cut it consumes counted in `scene_cut`, or-ed with a cut the GPU already
/// reset on the last network frame ([`apply_scene_cut`], read earlier in the same hold), which a network
/// frame without a latched cut must not take back.
fn note_network_reset(frames: u64, stale: Option<Stale>, scene_cut: &mut bool) {
    if frames > 0 {
        note_reset(stale);
        *scene_cut |= matches!(stale, Some(Stale::SceneCut(_)));
    }
}

/// The CPU's side of a scene-cut reading ([`super::scene_cut::effect`]), at the hold after the one it was
/// taken on. `true` when it counts a reset (the GPU's, logged here); a latched cut is counted when the next
/// network frame consumes it (`HistoryGap::begin`).
pub(crate) fn apply_scene_cut(effect: super::scene_cut::Effect, native: Option<&mut NativePass>, edit: Option<&mut super::cache::State>) -> bool {
    match effect {
        super::scene_cut::Effect::ResetOnGpu(mean) => {
            note_reset(Some(Stale::SceneCut(mean)));
            if let Some(n) = native {
                n.restart_seed();
            }
            true
        }
        super::scene_cut::Effect::Latch(mean) => {
            if let Some(n) = native {
                n.gap.scene_cut(mean);
            }
            if let Some(e) = edit {
                e.drop_edit();
            }
            false
        }
        super::scene_cut::Effect::None => false,
    }
}

/// Every hold's view of the target, for [`HistoryGap`]: the input's size. Not its identification: DLSS's
/// input alternates between two colour images every frame in Black Myth: Wukong with frame generation on,
/// and each switch reset the history.
pub(crate) fn format_of(target: &Target) -> (u32, u32) {
    (target.width, target.height)
}

/// The native pass when it was built for other pass extents than `extents` (another model resolution),
/// taken out of `native` for the caller to destroy; `None` (and `native` as it was) when it matches or there
/// is none.
pub(crate) fn take_other_extents(
    native: &mut Option<NativePass>,
    width: u32,
    height: u32,
    extents: &[(u32, u32); neural_forge_protocol::MAX_PASSES],
) -> Option<NativePass> {
    if native.as_ref().is_some_and(|n| !n.matches_at(width, height, extents)) {
        native.take()
    } else {
        None
    }
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

/// Why a hold inside DLSS's buffer records nothing while changes wait for the holds recorded before it.
pub(crate) const WAITING: &str = "waiting for the holds recorded in DLSS's buffers to be done before changing what they use";

/// One native hold (module docs): the capture with the motion-vector copy, the network, the write-back.
/// Through `sink`: on the split hold, submitted on the game's queue with no wait in between (the previous
/// hold's work is waited for, bounded, first, and its readings taken); inside DLSS's buffer, recorded into
/// it (`super::inline`: nothing waited for; what the holds before left is in `res.readings`, read back once
/// their buffers ran). Frames the network is not ready for go to DLSS untouched, with `miss` saying why;
/// inside DLSS's buffer nothing is recorded for them.
///
/// # Safety
/// As `super::run_hold`; `loader` belongs to `device`, and `res` was built for the queue family the
/// loader's network was opened for. Inside DLSS's buffer, the buffer is being recorded on this thread.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_native_hold(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    res: &mut super::Resources,
    target: &Target,
    loader: &Loader,
    jitter: Option<[f32; 2]>,
    chain_ok: bool,
    passes: Passes,
    sizes: &[(u32, u32)],
    arm: Option<super::cache_dump::Arm>,
    ask: CacheAsk,
    choice: u32,
    lut: Option<&Arc<super::colour::Lut>>,
    sink: super::Sink,
) -> super::HoldResult {
    let mut result = super::HoldResult { native: true, ..Default::default() };
    let t_hold = Instant::now();
    let (submit, mut ctx) = match sink {
        super::Sink::Queue(s) => (Some(s), None),
        super::Sink::Buffer(c) => (None, Some(c)),
    };
    let inline = ctx.is_some();
    // Inside DLSS's buffer, while holds recorded with these resources may still run, nothing they use changes
    // (`super::may_rewrite`): a hold that would need a change records nothing until they are done.
    let frozen = super::Frozen::new(ctx.as_ref().is_some_and(|c| c.outstanding > 0));
    let mut previous_exposure = None;
    if inline {
        // What the holds whose buffers ran since the last hold left: their GPU times, the edit cache's
        // counters and the scene cut (applied when they were read back, `Session::harvest`), the exposure.
        let r = std::mem::take(&mut res.readings);
        result.capture_gpu_ms = r.capture_gpu_ms;
        result.network_gpu_ms = r.network_gpu_ms;
        result.writeback_gpu_ms = r.writeback_gpu_ms;
        result.cache_rejected = r.cache_rejected;
        result.scene_cut = r.scene_cut;
        result.auto_exposure = r.auto;
        previous_exposure = r.previous_exposure.filter(|(id, _)| *id == target.identification).map(|(_, e)| e);
    } else {
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
        // The previous hold's scene-cut reading (finished: waited above): a network frame the GPU started
        // afresh is logged and counted, and the CPU's seed restarts with it; a cut on a cached hold resets
        // the next network frame, and the edit cache refreshes rather than carry an edit across it.
        if let Some((reading, recorded)) = res.scene.as_mut().and_then(super::scene_cut::ScenePass::take) {
            let effect = super::scene_cut::effect(reading, recorded);
            result.scene_cut = apply_scene_cut(effect, res.native.as_mut(), res.edit.as_mut().map(|e| &mut e.state));
        }
        // An armed hold's readbacks (finished: waited above), before anything clears the exposure.
        result.cache_frame = res.take_cache_frame();
        if arm.is_none() {
            // SAFETY: idle (waited above).
            unsafe { res.free_cache(device) };
        }
        // The game's exposure the previous hold's capture read (finished: waited above), for the
        // caller's check (`Session::check_exposure`). Only for this identification.
        if res.game_exposure_read.take() == Some(target.identification) {
            previous_exposure = res.hdr.as_ref().map(hdr::HdrPass::exposure_value);
        }
    }
    if !loader.claim_pre() {
        result.miss = Some("the network is serving the after-the-upscaler path");
        return result;
    }
    // Every frame that ran the graph has completed (this path's: the hold waited above; the other
    // path's: `claim_pre` says none ran for a second): the chain watchdog's count is final for them.
    let fake = fake_chain_timeout(res.native.as_ref().map_or(0, |n| n.frames));
    if let Some((waits, at)) = loader.chain_timeouts().filter(|(n, _)| *n > 0).or(fake) {
        if !loader.fall_back() {
            result.miss = Some(WAITING);
            return result;
        }
        crate::log!("[native] {waits} counter-chain wait(s) of the network gave up ({at}); rebuilding it with barriers between its launches");
        crate::logging::flush();
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some("the network's counter chain timed out; it is being rebuilt with barriers");
        return result;
    }
    // The model resolution (`preupscale_scale`, `pass<N>_scale`): each pass slot's extent. The sizes the
    // network is built for are the frame's (`frame_sizes`: one at a model resolution of 1 with every pass
    // before the upscaler, the render extent, as always), this hold's among them.
    let extents = pass_extents(target.width, target.height, &passes.scales);
    let Some(own) = sizes.iter().position(|s| *s == extents[0]) else {
        result.miss = Some("the network's sizes do not include this hold's");
        return result;
    };
    // The Network setting (`choice`): a switch frees the current build here (this hold's previous work
    // was waited for above, the other path's is kept off by `claim_pre`, and the holds recorded into DLSS's
    // buffers keep it until they are done: `Loader::ready_sizes`), and the frames go to DLSS untouched until
    // the chosen network is built. So does another set of sizes.
    let Some(builts) = loader.ready_sizes(sizes, choice) else {
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some(if loader.failed().is_some() {
            "the native network could not be loaded or built"
        } else if loader.waiting_for_holds() {
            WAITING
        } else {
            "the native network is still loading"
        });
        return result;
    };
    // Pass 1's: its network, its generation and its input features (where the edit cache's buffers live).
    let built = builts[own];
    // Counter chaining faulted the GPU from the hold inside DLSS's buffer (Black Myth: Wukong, 3 runs
    // of 4: Xid 13 then 32 within seconds; none of 2 with barriers): that hold rebuilds the network
    // with barriers between its launches, for the rest of the process (`fall_back_to_barriers`).
    if !chain_ok && builts.iter().any(|b| b.frame.chained != 0) {
        if !loader.fall_back() {
            result.miss = Some(WAITING);
            return result;
        }
        crate::log!("[native] the hold inside DLSS's buffer runs the network with barriers between its launches, not counter chaining; rebuilding it");
        crate::logging::flush();
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
    // Inside DLSS's buffer the auto-exposure's state is reset there, in the buffer's order.
    let mut auto_reset = false;
    if source == super::ExposureSource::Auto {
        // SAFETY: `device` is live; nothing of the pass is pending.
        if res.native_retry.auto.build(extent, now, || unsafe { pass.ensure_auto(device, instance, physical_device) }.then_some(())).is_none() {
            result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
            return result;
        }
        if inline {
            auto_reset = pass.auto_reset_due(target.identification);
        } else {
            pass.auto_for(target.identification);
        }
    }
    if source == super::ExposureSource::Game {
        result.previous_exposure = previous_exposure;
    }
    if inline {
        // The staging image is the layer's own and stays: bound once.
        // SAFETY: nothing that may run uses the set unless it is unchanged; the colour input is a live
        // RGBA16F storage image.
        let staging = ctx.as_ref().map_or(0, |c| c.staging.id);
        if !unsafe { pass.bind_colour_once(device, target.colour, staging) } {
            result.miss = Some(frozen.blocked().map_or("a storage view of the colour input could not be created", |_| WAITING));
            return result;
        }
    } else {
        pass.clear_exposure();
        // SAFETY: nothing of the pass is pending; the colour input is a live RGBA16F storage image.
        if !unsafe { pass.bind_colour(device, target.colour) } {
            result.miss = Some("a storage view of the colour input could not be created");
            return result;
        }
    }
    let (encoded, exposure) = (pass.encoded_view(), pass.exposure_buffer());
    // The graph for the queue the frames go to: the game's graphics family (the split hold executes the
    // graph's secondary of that family, the hold inside DLSS's buffer records the graph into it), or the
    // network's own compute family.
    let setup = loader.setup();
    let graphs: Vec<Graph> = if inline {
        if res.queue_family != setup.frame_family {
            result.miss = Some("the hold's queue family is not the one the network runs on");
            return result;
        }
        builts.iter().map(|&b| Graph { built: b, graph: vk::CommandBuffer::null() }).collect()
    } else if res.queue_family == setup.frame_family {
        builts.iter().map(|&b| Graph { built: b, graph: b.graph }).collect()
    } else if res.queue_family == setup.family {
        builts.iter().map(|&b| Graph { built: b, graph: b.graph_own }).collect()
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
    debug_assert!(res.scene.as_ref().is_none_or(|s| s.matches(target.width, target.height)));
    // Another model resolution: the pass is built again (every history starts afresh, at its new extent),
    // and the edit cache refreshes rather than carry an edit through the old pass's motion.
    if res.native.as_ref().is_some_and(|n| !n.matches_at(target.width, target.height, &extents)) && !super::may_rewrite("the network's frame resources") {
        result.miss = Some(WAITING);
        return result;
    }
    if let Some(old) = take_other_extents(&mut res.native, target.width, target.height, &extents) {
        crate::log!("[native] resetting the history (the model resolution changed)");
        // SAFETY: idle (`wait_idle` above, or no hold recorded with it may still run).
        unsafe { old.destroy(device) };
        if let Some(e) = res.edit.as_mut() {
            e.state.drop_edit();
        }
    }
    let mut native = match res.native.take() {
        Some(n) => n,
        None => {
            let family = res.queue_family;
            // SAFETY: `device` is live; `family` is the frames' family.
            let built = res.native_retry.pass.build((target.width, target.height), now, || unsafe {
                NativePass::build_at(device, instance, physical_device, family, target.width, target.height, extents, !cache_on, !inline)
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
    // More than one pass: their buffers, built once (a failure runs one pass).
    let more = passes.count > 1 && {
        let family = res.queue_family;
        // SAFETY: `device` is live; `family` is the pass's; nothing of it is pending.
        unsafe { native.ensure_more(device, instance, physical_device, family, passes.count) }
    };
    // Below a model resolution of 1: the passes' resources at their extents.
    // SAFETY: as above.
    if !unsafe { native.ensure_lanes(device, instance, physical_device, passes.count) } {
        res.native = Some(native);
        result.miss = Some("the model resolution's buffers could not be built");
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
            DepthCopy::build(device, instance, physical_device, family, target.width, target.height, !inline)
        });
    }
    let depth = depth.filter(|_| res.depth.is_some());
    // The control mask's rules for this frame: the sky rule reads the depth copy, in the convention the edit
    // cache decided.
    let mut passes = passes;
    passes.mask = passes.mask.for_frame(depth.map_or(0, super::cache::depth_kind), res.edit.as_ref().and_then(|e| e.state.reverse_z()));
    let depth_buffer = depth.and(res.depth.as_ref()).map(DepthCopy::buffer);
    // The colour of the edit: whether this frame records it, and the look-up table.
    // SAFETY: nothing of the pass is pending (`wait_idle` above), or its table is not written now.
    unsafe { native.prepare_colour(device, instance, physical_device, &passes, lut) };
    // SAFETY: nothing of the pass is pending (`wait_idle` above), or its sets are not written now; the HDR
    // pass outlives it in `res`.
    if !unsafe { native.record_graphs(device, instance, physical_device, &graphs, encoded, exposure, depth_buffer) } {
        res.native = Some(native);
        result.miss = Some(frozen.blocked().map_or("recording the native frame failed", |_| WAITING));
        return result;
    }
    // The edit cache (`super::cache`): its passes for this extent, and its store while the interval
    // is above 1 (at 1 only the motion pass runs, copying the vectors for the network as they are).
    if cache_on && res.edit.is_none() {
        let family = res.queue_family;
        // SAFETY: `device` is live; `family` is the holds' family.
        res.edit = res.native_retry.edit.build((target.width, target.height), now, || unsafe {
            super::cache::CachePass::build(device, instance, physical_device, family, target.width, target.height, !inline)
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
    // The scene-cut detector, unless switched off (`NEURAL_FORGE_SCENE_CUT=off`). Without it (a failed
    // build, tried again on the retry schedule) the history resets on the other rules only.
    let threshold = super::scene_cut::threshold();
    if threshold > 0 && res.scene.is_none() {
        res.scene = res.native_retry.scene.build((target.width, target.height), now, || {
            // SAFETY: `device` is live.
            let built = unsafe { super::scene_cut::ScenePass::build(device, instance, physical_device, target.width, target.height) };
            if built.is_none() {
                crate::log!("[native] the scene-cut detector could not be built; the history resets on the other rules only");
            }
            built
        });
    }
    if let (Some(s), Some(hdr_ref)) = (res.scene.as_mut().filter(|_| threshold > 0), res.hdr.as_ref()) {
        // SAFETY: nothing of the detector is pending (`wait_idle` above), or its set is not written now; the
        // encoded view and the native pass's state buffer are live.
        unsafe { s.bind(device, hdr_ref.encoded_view(), native.state_buffer(), [hdr_ref.id(), native.id()]) };
    }
    native.gap.note_format(format_of(target));
    // A frame of another network than the history's (the Network setting switched) resets the history
    // (`frame_passes` below then runs with no history: seed 0, no blend toward it, in every pass).
    if note_network(&mut native.gap, &mut native.history_network, built.network) {
        crate::log!("[native] resetting the history (the {} network runs from here)", network::name(built.network));
    }
    let history_ok = native.frames > 0 && native.gap.pending(hold_now()).is_none();
    // The cache's plan for this hold (a refresh at interval 1 with the cache off).
    let mut cache_hold = None;
    let mut plan = None;
    let mut push = None;
    let mut chained = false;
    // Above interval 1 the cache's buffers are in the network's input features (`cache::ScratchLayout`), or
    // with pass 1 below the render extent (whose features are too small) in a buffer of the pass's own.
    let scratch = match res.edit.as_ref().filter(|_| cache_on) {
        Some(edit) => {
            let bytes = edit.scratch_bytes();
            // SAFETY: `device` is live; nothing of the pass is pending (`wait_idle` above), or its buffers are
            // not replaced now.
            unsafe { native.cache_scratch(device, instance, physical_device, (built.frame.features, built.frame.features_bytes), bytes) }
                .unwrap_or((built.frame.features, built.frame.features_bytes))
        }
        None => (built.frame.features, built.frame.features_bytes),
    };
    if let Some(edit) = res.edit.as_mut().filter(|_| cache_on) {
        chained = edit.chain(scratch.1);
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
        let hdr_ref = res.hdr.as_ref().expect("built above");
        let bindings = super::cache::Bindings {
            colour: colour_view,
            encoded: hdr_ref.encoded_view(),
            answer: hdr_ref.answer_view(),
            exposure: hdr_ref.exposure_buffer(),
            depth: depth.and(res.depth.as_ref()).map(DepthCopy::buffer),
            mvec: native.mvec.buffer,
            acc: native.acc.buffer,
            scratch,
            // The split hold makes a view of the colour input every hold (`HdrPass::bind_colour`): the set is
            // written every hold, as the view's handle often comes back for the next one.
            made: [hdr_ref.colour_binding(), native.id(), built.generation],
        };
        // SAFETY: nothing of the pass is pending, or its set is not written now; every handle is live.
        unsafe { edit.bind(device, &bindings) };
        // A hold that records nothing (below) must not have taken the stored edit out of the plan.
        if frozen.blocked().is_none() {
            let p = edit.state.plan(&h);
            if !inline {
                edit.clear_counters();
            }
            // No vectors copied this hold: the motion pass writes no history and counts nothing.
            let flags = p.flags | if mvec.is_none() { super::cache::F_NO_MVEC } else { 0 };
            let mut q = edit.push(flags, depth.map_or(0, super::cache::depth_kind), target.mvec_scale, target.paper_white);
            q.step_jitter = p.step_jitter;
            q.refresh_jitter = p.refresh_jitter;
            cache_hold = Some(h);
            plan = Some(p);
            push = Some(q);
        }
    }
    let cached = plan.is_some_and(|p| p.kind == super::cache::Kind::Cached);
    // The motion pass while the cache runs (its prep and the accumulated field the network reads), at
    // interval 1 whenever there are vectors to convert.
    let run_motion = push.is_some() && (chained || mvec.is_some());
    // A cache dump's readbacks for an armed hold, built on its first one (inside DLSS's buffer, the slot's).
    let dump_buffers = match ctx.as_deref_mut() {
        Some(c) => {
            if arm.is_some() && c.cache_dump.is_none() {
                // SAFETY: `device` is live.
                *c.cache_dump = unsafe { super::cache_dump::Buffers::build(device, instance, physical_device, target.width, target.height) };
            }
            arm.and(c.cache_dump.as_ref())
        }
        None => {
            if arm.is_some() && res.cache.is_none() {
                // SAFETY: `device` is live.
                res.cache = unsafe { super::cache_dump::Buffers::build(device, instance, physical_device, target.width, target.height) };
            }
            arm.and(res.cache.as_ref())
        }
    };
    let cache = dump_buffers.map(|c| super::CacheCopies {
        colour: c.colour.buffer,
        answer: c.answer.buffer,
        mvec: mvec.map(|_| (native.mvec.buffer, c.mvec.buffer)),
        depth: depth.and(res.depth.as_ref()).map(|d| (d.buffer(), c.depth.buffer)),
    });
    // Inside DLSS's buffer: the guard that checks this hold follows the one recorded before it, and the copies
    // between the network's passes, made ready now (nothing of them is written once recording starts).
    if inline {
        if res.guard.is_none() {
            // SAFETY: `device` is live.
            res.guard = unsafe { HoldGuard::build(device, instance, physical_device) };
        }
        let stats = res.edit.as_ref().filter(|_| cache_on).map(super::cache::CachePass::stats_buffer);
        // SAFETY: both buffers are live; the set is not written while a hold that bound it may run.
        let bound = res.guard.as_mut().is_some_and(|g| unsafe { g.bind(device, native.state_buffer(), native.id(), stats) });
        if !bound && frozen.blocked().is_none() {
            res.native = Some(native);
            result.miss = Some("the hold guard could not be built");
            return result;
        }
        if more && !cached {
            let encoded_view = res.hdr.as_ref().expect("built above").encoded_view();
            // SAFETY: the feedback blend's set is not written while a hold that bound it may run.
            if !unsafe { native.prepare_copies(device, encoded_view, passes.feedback) } && frozen.blocked().is_none() {
                res.native = Some(native);
                result.miss = Some("the feedback blend between the network's passes could not be built");
                return result;
            }
        }
    }
    if let Some(why) = frozen.blocked() {
        crate::log!("[native] the hold waits: {why} would change while holds recorded with it may still run");
        res.native = Some(native);
        result.miss = Some(WAITING);
        return result;
    }
    let hdr_pass = res.hdr.as_ref();
    let depth_copy = depth.and(res.depth.as_ref()).map(|d| (d.buffer(), d.timer.as_ref()));
    let edit = res.edit.as_ref().filter(|_| cache_on);
    let acc = native.acc.buffer;
    let no_push = super::cache::CachePush::default();
    let push_ref = push.as_ref().unwrap_or(&no_push);
    // SAFETY (the three closures, only called while the cache runs): recording into the hold's own
    // buffers or DLSS's; the set was bound above.
    let motion = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_motion(device, cmd, push_ref, acc) };
    let capture_edit = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_capture(device, cmd, push_ref) };
    let apply = |cmd: vk::CommandBuffer| unsafe { edit.expect("the cache runs").record_apply(device, cmd, push_ref) };
    // After the encode in the capture: the scene-cut detector, then the cache's motion pass.
    let scene = res.scene.as_ref().filter(|_| threshold > 0);
    let after_encode = |cmd: vk::CommandBuffer| {
        if let Some(s) = scene {
            // SAFETY: recording the capture, after the encode; bound above.
            unsafe { s.record(device, cmd, threshold) };
        }
        if run_motion {
            motion(cmd);
        }
    };
    let more = more && !cached;
    if let Some(ctx) = ctx {
        // ---- Inside DLSS's buffer: everything recorded into it, in the split hold's order. ----
        let cmd = ctx.cb;
        // The frame's plan (its parameters, its history, its parity), as the split hold makes it.
        let (history, frame_plan) = if cached {
            (false, None)
        } else {
            let stale = native.gap.begin(hold_now());
            note_network_reset(native.frames, stale, &mut result.scene_cut);
            let history = native.frames > 0 && stale.is_none() && mvec.is_some();
            (history, Some(native.frame_plan(&graphs, history, jitter, target.mvec_scale, &passes)))
        };
        let seq = res.seq + 1;
        let stats = edit.map(super::cache::CachePass::stats_buffer);
        let auto_state = res.hdr.as_ref().and_then(hdr::HdrPass::auto_state_buffer).filter(|_| auto_reset);
        let auto_read = res.hdr.as_ref().and_then(hdr::HdrPass::auto_state_buffer).filter(|_| source == super::ExposureSource::Auto);
        let scene_result = scene.map(super::scene_cut::ScenePass::result_buffer);
        // SAFETY: DLSS's buffer is being recorded on this thread (the contract); every handle is live and
        // stays so while the buffer may run (nothing is destroyed or rewritten while a hold may run).
        unsafe {
            ctx.open();
            res.guard.as_ref().expect("bound above").record(device, cmd, seq);
            // What the split hold writes from the host before it submits: here, in the buffer's order.
            device.cmd_fill_buffer(cmd, exposure, 0, hdr::HdrPass::EXPOSURE_BYTES, 0);
            if let Some(stats) = stats {
                device.cmd_fill_buffer(cmd, stats, 0, super::cache::CLEARED_BYTES, 0);
            }
            if let Some((state, bytes)) = auto_state {
                device.cmd_fill_buffer(cmd, state, 0, bytes, 0);
            }
            for (buffer, params) in frame_plan.iter().flat_map(|p| p.params.iter()) {
                device.cmd_update_buffer(cmd, *buffer, 0, std::slice::from_raw_parts(std::ptr::from_ref(params).cast::<u8>(), PARAMS_BYTES as usize));
            }
            let written = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
            let _ = super::record_capture(
                device,
                res,
                target,
                None,
                hdr_pass,
                None,
                mvec.map(|_| native.mvec.buffer),
                depth_copy,
                cache.map(|c| c.colour),
                (scene.is_some() || run_motion).then_some(&after_encode as &dyn Fn(vk::CommandBuffer)),
                Some(cmd),
            );
            ctx.stamp(1);
        }
        let mut network_ok = true;
        if let Some(fp) = frame_plan.as_ref() {
            let hdr_ref = res.hdr.as_ref().expect("built above");
            let slot = ctx.slot;
            let mut network = |cmd: vk::CommandBuffer, built: &Built| {
                // SAFETY: `cmd` is DLSS's buffer, recording; no buffer recorded with this slot may still run.
                unsafe { loader.record_graph(built, slot, cmd) }
            };
            // SAFETY: as above; the copies were made ready above.
            let recorded = unsafe {
                native.record_frame_inline(
                    device,
                    cmd,
                    fp,
                    &graphs,
                    hdr_ref.encoded_image(),
                    hdr_ref.padded_region(),
                    passes.count,
                    passes.feedback,
                    &mut network,
                )
            };
            if let Err(why) = recorded {
                crate::log!("[native] recording the network into DLSS's buffer failed: {why}; this frame goes to DLSS as it was");
                crate::logging::flush();
                network_ok = false;
            }
        }
        // SAFETY: as above.
        unsafe {
            ctx.stamp(2);
            if network_ok {
                let _ = if cached {
                    super::record_cached_writeback(device, res, &apply, Some(cmd))
                } else {
                    super::record_writeback(
                        device,
                        res,
                        target,
                        native.answer(),
                        hdr_pass,
                        cache.as_ref(),
                        chained.then_some(&capture_edit as &dyn Fn(vk::CommandBuffer)),
                        Some(cmd),
                    )
                };
            }
            // What the host reads once the buffer ran (`inline::Harvest`): the counters, the scene cut, the
            // exposure and the auto-exposure's state, copied out of the buffers the next holds write again.
            let before = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            let copy =
                |src: vk::Buffer, at: u64, size: u64| device.cmd_copy_buffer(cmd, src, ctx.readback, &[vk::BufferCopy { src_offset: 0, dst_offset: at, size }]);
            if let Some(stats) = stats {
                copy(stats, super::inline::READ_STATS, super::cache::STAT_WORDS as u64 * 4);
            }
            if let Some(result) = scene_result {
                copy(result, super::inline::READ_SCENE, 16);
            }
            copy(exposure, super::inline::READ_EXPOSURE, 4);
            if let Some((state, bytes)) = auto_read {
                copy(state, super::inline::READ_AUTO, bytes.min(super::inline::READBACK_BYTES - super::inline::READ_AUTO));
            }
            let read = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[read], &[], &[]);
        }
        res.seq = seq;
        result.timing.prep = Some(t_hold.elapsed());
        result.waits_consumed = true;
        if let Some(d) = res.depth.as_mut() {
            d.valid = depth;
        }
        if let Some(s) = res.scene.as_mut().filter(|_| threshold > 0) {
            s.recorded_inline();
        }
        res.game_exposure_read = (source == super::ExposureSource::Game).then_some(target.identification);
        let (pass_id, init_command) = (native.id(), native.init_command());
        res.native = Some(native);
        let meta = super::inline::Meta {
            stage: if ctx.point.hold.stage == super::placement::Stage::Upscaled { super::placement::Stage::Upscaled } else { super::placement::Stage::Before },
            resources: res.id,
            identification: target.identification,
            stats: stats.is_some(),
            scene: scene_result.map(|_| super::scene_cut::Recorded { network: !cached, history }),
            auto: auto_read.is_some(),
            kind: plan.map(|p| p.kind),
            init: Some((pass_id, init_command)),
            fresh: (pass_id, frame_plan.as_ref().map_or(0, |p| p.fresh)),
            exposure_source: Some(source),
            cache_dump: None,
            timed: ctx.queries != vk::QueryPool::null(),
        };
        if !network_ok {
            if let Some(n) = res.native.as_mut() {
                n.gap.skipped();
            }
            if let Some(e) = res.edit.as_mut() {
                e.state.drop_edit();
            }
            result.miss = Some("recording the network into DLSS's buffer failed");
            result.meta = Some(meta);
            return result;
        }
        if cached {
            // The 500 ms rule counts from this hold, not the last network frame.
            if let Some(n) = res.native.as_mut() {
                n.gap.touch(hold_now());
            }
        }
        if let (Some(edit), Some(h), Some(p)) = (res.edit.as_mut(), cache_hold.as_ref(), plan.as_ref()) {
            edit.state.went_through(h, p);
        }
        let pending = arm.filter(|_| cache.is_some()).map(|arm| super::cache_dump::Pending {
            arm,
            time_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64),
            jitter,
            history,
            mvec: mvec.is_some(),
            depth,
            depth_missing: depth.is_none().then(|| {
                if cache_on {
                    depth_missing(target)
                } else {
                    "the edit cache is off (NEURAL_FORGE_EDIT_CACHE=off): no depth is copied"
                }
            }),
            depth_layout: target.depth.and_then(|a| a.layout),
            mvec_scale: target.mvec_scale,
            paper_white: target.paper_white,
        });
        result.meta = Some(super::inline::Meta { cache_dump: pending, ..meta });
        result.evaluated = true;
        result.wrote_back = true;
        result.cached = cached;
        result.cache_still = cache_hold.is_some_and(|h| h.interval > 1) && res.edit.as_ref().is_some_and(|e| e.state.regime.still());
        result.timing.writeback = Some(t_hold.elapsed());
        return result;
    }
    let submit = submit.expect("the split hold submits");
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
            (scene.is_some() || run_motion).then_some(&after_encode as &dyn Fn(vk::CommandBuffer)),
            None,
        )
    };
    if recorded.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the capture failed");
        return result;
    }
    // The copies between the passes, on a network frame with more than one.
    if more {
        let pass = res.hdr.as_ref().expect("built above");
        // SAFETY: built above and not pending (the hold waited for the previous one's fence, which
        // comes after them); the encoded image is in GENERAL.
        if unsafe { native.record_copies(device, pass.encoded_image(), pass.encoded_view(), pass.padded_region(), passes.count, passes.feedback) }.is_err() {
            res.native = Some(native);
            result.miss = Some("recording the copies between the network's passes failed");
            return result;
        }
    }
    // SAFETY: as above: on a cached hold the cache's passes write the colour input; else the decode,
    // with the native pass's answer as its source (and the edit captured first while the cache runs).
    let recorded = unsafe {
        if cached {
            super::record_cached_writeback(device, res, &apply, None)
        } else {
            super::record_writeback(
                device,
                res,
                target,
                native.answer(),
                hdr_pass,
                cache.as_ref(),
                chained.then_some(&capture_edit as &dyn Fn(vk::CommandBuffer)),
                None,
            )
        }
    };
    if recorded.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the write-back failed");
        return result;
    }
    // A cached hold keeps the network's history as it is (no frame of the network): its parity,
    // seed and jitter stay the last refresh's.
    let (history, frame_cmds, later) = if cached {
        (false, [None, None], Vec::new())
    } else {
        let stale = native.gap.begin(hold_now());
        note_network_reset(native.frames, stale, &mut result.scene_cut);
        let history = native.frames > 0 && stale.is_none() && mvec.is_some();
        let (cmds, later) = native.frame_graphs(&graphs, history, jitter, target.mvec_scale, &passes);
        (history, cmds, if more { later } else { Vec::new() })
    };
    res.native = Some(native);
    result.timing.prep = Some(t_hold.elapsed());
    if let Some(d) = res.depth.as_mut() {
        d.valid = None;
    }
    if crate::note_vk(device.handle(), submit(super::Which::Capture, res.capture_cmd, vk::Fence::null())).is_err() {
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
    if let Some(s) = res.scene.as_mut().filter(|_| threshold > 0) {
        s.submitted(super::scene_cut::Recorded { network: !cached, history });
    }
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
            // The empty buffer's fence comes after the network's frame on the same queue: it covers it.
            loader.pre_submitted(device, res.writeback_fence);
        } else {
            // No fence could be put after it: the network may be in use for as long as a frame may run.
            loader.pre_unfenced();
        }
    };
    for cmd in frame_cmds.into_iter().flatten() {
        if crate::note_vk(device.handle(), submit(super::Which::Native, cmd, vk::Fence::null())).is_err() {
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
    if let (false, Some(copies)) = (later.is_empty(), res.native.as_ref().and_then(|n| n.more.as_ref()).map(|m| m.copies)) {
        // Pass 2 on: `X` (or `Y`) puts the previous answer where the network reads, then the pass;
        // `R` puts the capture back for the write-back.
        let mut ok = true;
        for (k, &cmd) in later.iter().enumerate() {
            let copy = if k == 0 { copies[0] } else { copies[1] };
            ok = ok
                && crate::note_vk(device.handle(), submit(super::Which::Native, copy, vk::Fence::null())).is_ok()
                && crate::note_vk(device.handle(), submit(super::Which::Native, cmd, vk::Fence::null())).is_ok();
        }
        ok = ok && crate::note_vk(device.handle(), submit(super::Which::Native, copies[2], vk::Fence::null())).is_ok();
        if !ok {
            failed(res, submit);
            result.miss = Some("a later pass of the network could not be submitted");
            return result;
        }
        if let Some(m) = res.native.as_mut().and_then(|n| n.more.as_mut()) {
            // `X` took the passes' new history to GENERAL.
            for f in m.fresh.iter_mut().take(later.len()) {
                *f = false;
            }
            if let Some(t) = m.timer.as_mut() {
                t.mark_submitted();
            }
        }
    }
    // SAFETY: the layer's own fence, idle (`wait_idle` above, or never submitted).
    if unsafe { device.reset_fences(&[res.writeback_fence]) }.is_err() {
        failed(res, submit);
        result.miss = Some("resetting the write-back fence failed");
        return result;
    }
    if crate::note_vk(device.handle(), submit(super::Which::WriteBack, res.writeback_cmd, res.writeback_fence)).is_err() {
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
            n.gap.touch(hold_now());
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
            depth_missing: depth.is_none().then(|| {
                if cache_on {
                    depth_missing(target)
                } else {
                    "the edit cache is off (NEURAL_FORGE_EDIT_CACHE=off): no depth is copied"
                }
            }),
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

    /// Resources built untimed (inside DLSS's buffer the hold times its slot) make no GPU timer at all: a timer
    /// has no destructor, so one made and dropped left its query pool behind, past `vkDestroyDevice` (which the
    /// validation layer reports here, at the end of the test).
    #[test]
    fn untimed_resources_make_no_query_pool() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        // SAFETY: a fresh device; nothing is submitted; everything is destroyed before the device.
        unsafe {
            let extents = pass_extents(W, H, &[1.0; neural_forge_protocol::MAX_PASSES]);
            let mut pass = NativePass::build_at(device, instance, physical, family, W, H, extents, false, false).expect("the pass builds");
            assert!(pass.timers.iter().all(Option::is_none), "the frames untimed");
            assert!(pass.ensure_more(device, instance, physical, family, 2), "the later passes build");
            assert!(pass.more.as_ref().is_some_and(|m| m.timer.is_none()), "the later passes untimed");
            let depth = DepthCopy::build(device, instance, physical, family, W, H, false).expect("the depth copy builds");
            assert!(depth.timer.is_none(), "the depth copy untimed");
            let cache = super::super::cache::CachePass::build(device, instance, physical, family, W, H, false).expect("the edit cache builds");
            cache.destroy(device);
            depth.destroy(device);
            pass.destroy(device);
            device.destroy_device(None);
        }
    }

    /// A failed hold whose work no fence covers keeps the after-the-upscaler path off the network for
    /// as long as a frame may run; nothing else does.
    #[test]
    fn unfenced_pre_upscaler_work_keeps_the_post_path_off_the_network() {
        unsafe extern "system" fn gipa(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
            None
        }
        let setup = Setup {
            gipa,
            instance: vk::Instance::null(),
            physical: vk::PhysicalDevice::null(),
            frame_family: 0,
            family: 0,
            index: 0,
            post_index: 0,
            properties2: crate::vram::Properties2::None,
            flow: None,
        };
        let loader = Loader::start(vk::Device::null(), setup, Arc::default());
        // No hold for longer than the path switch: the post path may claim.
        loader.last_pre.store(0, std::sync::atomic::Ordering::SeqCst);
        let _ = epoch_ms();
        std::thread::sleep(std::time::Duration::from_millis(PATH_SWITCH_MS + 5));
        assert!(loader.claim_post());
        loader.release_post();
        loader.last_post.store(0, std::sync::atomic::Ordering::SeqCst);
        loader.pre_unfenced();
        assert!(!loader.claim_post(), "unfenced work may still use the network");
        loader.pre_unfenced_until.store(epoch_ms().saturating_sub(1), std::sync::atomic::Ordering::SeqCst);
        assert!(loader.claim_post(), "once a frame can no longer be running");
        loader.release_post();
    }

    fn built(size: (u32, u32), generation: u64, network: u32) -> Built {
        let graph = vk::CommandBuffer::null();
        Built { width: size.0, height: size.1, frame: nn::Frame::default(), graph, graph_own: graph, generation, network }
    }

    /// The loader's side of a switch of the Network setting: the build of the network open is not
    /// handed out for another choice, the loader thread is woken and has work (closing it and opening
    /// the chosen one), and once that is built it is served.
    #[test]
    fn another_network_chosen_is_not_served_the_open_one_and_wakes_the_loader() {
        let mut s = LoaderState { opened: true, opened_for: Some(network::FULL), running: network::FULL, ..Default::default() };
        s.built = vec![built((64, 32), 1, network::FULL)];
        s.wanted = Some(vec![(64, 32)]);
        assert!(!s.wants_work(), "the full network is built for the size wanted");
        let (b, wake) = s.want((64, 32), network::FULL);
        assert_eq!((b.map(|b| b.generation), wake), (Some(1), false), "the default choice gets the open network");
        let (b, wake) = s.want((64, 32), network::SMALL);
        assert!(b.is_none() && wake, "the small network is chosen: the full one's build is not used, the loader is woken");
        assert!(s.switching() && s.wants_work(), "the loader closes the full network and opens the small one");
        // Asked again before the loader got to it: still nothing, and no second wake.
        assert!(!s.want((64, 32), network::SMALL).1);
        // The loader opened and built the small network (what `work` stores).
        s.opened_for = Some(network::SMALL);
        s.running = network::SMALL;
        s.built = vec![built((64, 32), 2, network::SMALL)];
        assert!(!s.switching() && !s.wants_work());
        let (b, wake) = s.want((64, 32), network::SMALL);
        assert_eq!((b.map(|b| (b.generation, b.network)), wake), (Some((2, network::SMALL)), false));
        // The small network chosen, not extracted, the full one running in its place: no switch.
        s.running = network::FULL;
        assert!(!s.switching());
        // Back to the full network: the open one is not served until it is reopened.
        assert!(s.want((64, 32), network::FULL).0.is_none());
        assert!(s.switching() && s.wants_work());
        // A dead network is never switched.
        s.dead = true;
        assert!(!s.wants_work());
    }

    /// The chosen network loads when it is extracted; when it is not, the other one does, and with
    /// neither nothing does.
    #[test]
    fn the_chosen_network_loads_and_the_other_only_when_it_is_missing() {
        let both = |_| true;
        assert_eq!(network_to_load(network::FULL, both), Some(network::FULL));
        assert_eq!(network_to_load(network::SMALL, both), Some(network::SMALL));
        let only_full = |n| n == network::FULL;
        assert_eq!(network_to_load(network::SMALL, only_full), Some(network::FULL), "a DLL without the small network");
        assert_eq!(network_to_load(network::FULL, only_full), Some(network::FULL));
        let only_small = |n| n == network::SMALL;
        assert_eq!(network_to_load(network::FULL, only_small), Some(network::SMALL));
        assert_eq!(network_to_load(network::FULL, |_| false), None);
        // The directories are the setting's names for the networks under the model directory.
        assert_eq!(network_dir(network::FULL).file_name().unwrap(), "full");
        assert_eq!(network_dir(network::SMALL).file_name().unwrap(), "small");
        assert_eq!(network_dir(network::SMALL).parent(), Some(model_dir().as_path()));
    }

    /// A frame of another network than the one whose answers the history holds starts the history
    /// afresh; frames of the same network keep it.
    #[test]
    fn switching_the_network_resets_the_history() {
        let t0 = Instant::now();
        let mut gap: HistoryGap<(u32, u32)> = HistoryGap::default();
        let mut of = None;
        let frame = |gap: &mut HistoryGap<(u32, u32)>, of: &mut Option<u32>, network: u32, at: u64| {
            gap.note_format((64, 32));
            let changed = note_network(gap, of, network);
            (changed, gap.begin(t0 + std::time::Duration::from_millis(at)))
        };
        assert_eq!(frame(&mut gap, &mut of, network::FULL, 0), (false, None), "the first frame");
        assert_eq!(frame(&mut gap, &mut of, network::FULL, 16), (false, None), "the same network keeps its history");
        assert_eq!(frame(&mut gap, &mut of, network::SMALL, 33), (true, Some(Stale::Skipped)), "a switch resets it");
        assert_eq!(frame(&mut gap, &mut of, network::SMALL, 50), (false, None));
        assert_eq!(frame(&mut gap, &mut of, network::FULL, 66), (true, Some(Stale::Skipped)), "and so does the way back");
        assert_eq!(of, Some(network::FULL));
    }

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
    fn model_source_reads_network_build_and_verified() {
        let source = |extra: &str| format!("{{\n  \"source\": {{\n    \"build\": \"310.9.1.0\",\n    \"resource\": \"WEIGHTS_HT\"{extra}\n  }}\n}}");
        assert_eq!(model_source(&source(",\n    \"verified\": false")), "full network, build 310.9.1.0, not verified against NVIDIA's runtime");
        assert_eq!(model_source(&source(",\n    \"verified\": true")), "full network, build 310.9.1.0, verified");
        assert_eq!(model_source(&source("")), "full network, build 310.9.1.0, not verified against NVIDIA's runtime");
        assert_eq!(model_source(""), "full network, build unknown, not verified against NVIDIA's runtime");
        let small = "{\n  \"source\": {\n    \"build\": \"310.9.16.0\",\n    \"verified\": false\n  },\n  \"network\": \"small\",\n  \"stages\": []\n}";
        assert_eq!(model_source(small), "small network, build 310.9.16.0, not verified against NVIDIA's runtime");
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
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
            let frame = nn::Frame {
                field_width: FW,
                field_height: FH,
                features: features.buffer,
                features_bytes: u64::from(FW * FH) * 64,
                head: head.buffer,
                head_bytes: u64::from(FW * FH) * 16,
                blend_scale: 0.5,
                chained: 0,
            };
            let built = Built { width: W, height: H, frame, graph, graph_own: graph, generation: 1, network: network::FULL };
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(encoded.image)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                let region = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                    ..Default::default()
                };
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            // The old path (packed halves) and the new (the field), side by side.
            let mut old = NativePass::build_failing_at(device, instance, physical, family, W, H, None, true).unwrap();
            let mut new = NativePass::build(device, instance, physical, family, W, H, false).unwrap();
            assert!(old.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
            assert!(new.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
            let mut cache = super::super::cache::CachePass::build(device, instance, physical, family, W, H, true).unwrap();
            cache.bind(
                device,
                &super::super::cache::Bindings {
                    colour: encoded.view,
                    encoded: encoded.view,
                    answer: encoded.view,
                    exposure: exposure.buffer,
                    depth: None,
                    mvec: new.mvec.buffer,
                    acc: new.acc.buffer,
                    scratch: (vk::Buffer::null(), 0),
                    made: [0; 3],
                },
            );
            let answer_of = |pass: &NativePass| -> Vec<u16> {
                fx.one_shot(|cmd| {
                    device.cmd_copy_buffer(cmd, pass.answer(), staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
                });
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
            for (frame, (history, scale, jitter)) in [
                (false, [1.0, 1.0], Some([0.1, -0.2])),
                (true, [1.0, 1.0], Some([-0.3, 0.25])),
                (true, [0.6667, 0.6667], Some([0.4, 0.1])),
                (true, [0.5, 0.25], None),
            ]
            .into_iter()
            .enumerate()
            {
                let frame = frame as u32;
                let v = vectors(frame);
                let fill = |pass: &NativePass| {
                    std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), staging.ptr, v.len() * 4);
                    fx.one_shot(|cmd| {
                        device.cmd_copy_buffer(
                            cmd,
                            staging.buffer,
                            pass.mvec.buffer,
                            &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(W * H) * 4 }],
                        )
                    });
                };
                fill(&old);
                fill(&new);
                // The capture's motion pass at interval 1.
                let push = cache.push(super::super::cache::F_KEEP_CLAMPED, 0, scale, 3.0);
                fx.one_shot(|cmd| cache.record_motion(device, cmd, &push, new.acc.buffer));
                let h: Vec<f32> = (0..FH)
                    .flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, frame), residual(x, y, 1, frame), residual(x, y, 2, frame), 1.5]))
                    .collect();
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
        if back.abs() > v.abs() {
            f16(h - 1)
        } else {
            back
        }
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
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::default().command_buffers(&[cmd])], vk::Fence::null()).unwrap();
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
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::default().command_buffers(&cmds)], vk::Fence::null()).unwrap();
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
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            // The encoded proxy, the exposure (1.0), the network's features and head buffers, an empty "graph".
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
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
                network: network::FULL,
            };
            // The proxy into the encoded image (padded; the composite reads the valid part).
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(encoded.image)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                let region = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                    ..Default::default()
                };
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            assert!(pass.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
            // The frame shaders read the accumulated field (the cache's motion pass makes it from
            // DLSS's vectors): no motion.
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            let write_head = |frame: u32, logit: f32| {
                let h: Vec<f32> = (0..FH)
                    .flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, frame), residual(x, y, 1, frame), residual(x, y, 2, frame), logit]))
                    .collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
            };
            let answer = pass.answer();
            let read_answer = || -> Vec<f32> {
                fx.one_shot(|cmd| {
                    device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
                });
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
            fx.one_shot(|cmd| {
                device.cmd_copy_buffer(cmd, staging.buffer, pass.acc.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(W * H) * 8 }])
            });
            write_head(3, 10.0);
            let cmds = pass.frame(&built, true, None, [0.5, 0.25], Conditioning::default());
            fx.submit(&mut pass, cmds);
            // The history is frame 3's network answer (before its intensity blend).
            assert_ne!(neural(6, 4, 0, 2), neural(7, 7, 0, 2), "the test tells the scaled position from the unscaled one");
            assert_eq!(lanes(5, 3)[7], centre(neural(6, 4, 0, 2)), "history lane from the previous answer where the scaled vector points");

            // Frame 5: a NaN head at (5, 3), residual and logit, on a frame with history (no motion again): clamp()
            // of a NaN is undefined, so the composite reads it as 0. The answer there is 0 and the history the
            // next frame reads is 0, not NaN (which would reach the network's input lanes, frame after frame).
            // Lavapipe's clamp() happens to return 0 for a NaN too; the shader no longer relies on it.
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            write_head(4, 10.0);
            let nan_at = ((3 * FW + 5) * 4) as usize;
            for k in 0..4 {
                *head.ptr.cast::<f32>().add(nan_at + k) = f32::NAN;
            }
            let cmds = pass.frame(&built, true, None, [1.0, 1.0], Conditioning::default());
            fx.submit(&mut pass, cmds);
            let a5 = read_answer();
            for c in 0..3 {
                assert_eq!(a5[((3 * pw + 5) * 4 + c) as usize].to_bits(), 0, "a NaN head answers 0 in channel {c}");
            }
            write_head(5, 10.0);
            let cmds = pass.frame(&built, true, None, [1.0, 1.0], Conditioning::default());
            fx.submit(&mut pass, cmds);
            let l = lanes(5, 3);
            for c in 0..3 {
                assert_eq!(l[7 + c].to_bits(), centre(0.0).to_bits(), "the history lane {c} after a NaN head is 0, centred");
            }

            pass.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }
    /// The colour of the edit before the upscaler (`native_colour.comp`), on edits the head makes exactly
    /// (neural = proxy + residual / 4 on mid-range proxies): neutral values leave the answer and the history
    /// bit for bit as the network made them, even with the stage recorded; saturation 0 leaves a grey change of
    /// the same luma; contrast 0 a change of no luma; detail 0 removes a checkerboard and keeps the broad part;
    /// the gamut guard keeps the red channel from clipping and the change's direction (hue), where the clamp
    /// without it turns it; the look-up table at strength 0 changes nothing and at full strength is the table
    /// where the edit is large, and nothing where the network left the pixel alone.
    #[test]
    fn the_colour_of_the_edit_reshapes_the_answer_and_leaves_the_history() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
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
                    blend_scale: 0.5,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation: 1,
                network: network::FULL,
            };
            // Mid-range proxies (0.25 to 0.49, exact halves), so a change of up to 1/8 never clips by itself.
            let mid = |x: u32, y: u32, c: u32| 0.25 + ((x * 7 + y * 3 + c * 5) % 32) as f32 / 128.0;
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                ..Default::default()
            };
            let mut first = true;
            let mut fill = |proxy: &dyn Fn(u32, u32, u32) -> f32| {
                let texels: Vec<u16> =
                    (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
                std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
                let was_first = first;
                first = false;
                fx.one_shot(|cmd| {
                    if was_first {
                        let to_general = vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(encoded.image)
                            .subresource_range(range);
                        device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[to_general],
                        );
                    }
                    device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
                });
            };
            fill(&mid);
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            // The head for an edit of `edit(x, y, c)` (neural = proxy + residual / 4, exact on these values).
            let write_head = |edit: &dyn Fn(u32, u32, u32) -> f32| {
                let h: Vec<f32> =
                    (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [4.0 * edit(x, y, 0), 4.0 * edit(x, y, 1), 4.0 * edit(x, y, 2), 10.0])).collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
            };
            let answer = pass.answer();
            let read_answer = || -> Vec<f32> {
                fx.one_shot(|cmd| {
                    device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
                });
                std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).iter().map(|&h| f16(h)).collect()
            };
            let at = |a: &[f32], x: u32, y: u32| -> [f32; 3] { std::array::from_fn(|c| a[((y * pw + x) * 4 + c as u32) as usize]) };
            let lanes = |x: u32, y: u32| -> Vec<f32> { std::slice::from_raw_parts(features.ptr.cast::<f32>().add(((y * FW + x) * 16) as usize), 16).to_vec() };
            let luma = |e: [f32; 3]| e[0] * LUMA[0] + e[1] * LUMA[1] + e[2] * LUMA[2];
            // One frame with `colour` (no history unless `history`); `force` records the stage even when
            // nothing asks for it.
            let mut run =
                |colour: neural_forge_protocol::ColourSettings, lut: Option<&Arc<super::super::colour::Lut>>, history: bool, force: bool| -> (Vec<f32>, bool) {
                    let passes = Passes { colour, ..Passes::default() };
                    pass.prepare_colour(device, instance, physical, &passes, lut);
                    if force {
                        pass.colour_wanted = true;
                    }
                    assert!(pass.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
                    let (cmds, _) = pass.frame_passes(&built, history, None, [1.0, 1.0], &passes);
                    fx.submit(&mut pass, cmds);
                    (read_answer(), pass.colour_recorded)
                };
            let neutral = neural_forge_protocol::ColourSettings::DEFAULT;
            // A varied edit of up to 1/16 a channel.
            let edit = |x: u32, y: u32, c: u32| (((x * 5 + y * 11 + c * 3) % 9) as f32 - 4.0) / 64.0;
            let want = |x: u32, y: u32, c: u32| trunc_half(mid(x, y, c) + edit(x, y, c));
            write_head(&edit);

            // Neutral: as the network made it, without the stage and with it recorded.
            let (plain, recorded) = run(neutral, None, false, false);
            assert!(!recorded, "nothing asks for the stage by default");
            for y in 0..H {
                for x in 0..W {
                    for c in 0..3 {
                        assert_eq!(at(&plain, x, y)[c as usize], want(x, y, c), "the network's answer at {x},{y} channel {c}");
                    }
                }
            }
            let (forced, recorded) = run(neutral, None, false, true);
            assert!(recorded);
            assert_eq!(
                forced.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "neutral values leave the answer bit for bit"
            );

            // Saturation 0: a grey change of the edit's luma. The history the next frame reads is the network's
            // answer, not the reshaped one.
            let (grey, recorded) = run(
                neural_forge_protocol::ColourSettings {
                    each: [neural_forge_protocol::EditColour { saturation: 0.0, ..neural_forge_protocol::EditColour::NEUTRAL };
                        neural_forge_protocol::MAX_PASSES],
                    ..neutral
                },
                None,
                false,
                false,
            );
            assert!(recorded);
            for (x, y) in [(5, 3), (0, 0), (W - 1, H - 1), (11, 7)] {
                let e: [f32; 3] = std::array::from_fn(|c| want(x, y, c as u32) - mid(x, y, c as u32));
                let g = at(&grey, x, y);
                let d: [f32; 3] = std::array::from_fn(|c| g[c] - mid(x, y, c as u32));
                assert!((d[0] - d[1]).abs() <= 2f32.powi(-10) && (d[1] - d[2]).abs() <= 2f32.powi(-10), "saturation 0 leaves a grey change at {x},{y}: {d:?}");
                assert!((d[1] - luma(e)).abs() <= 2f32.powi(-10), "of the edit's luma at {x},{y}: {d:?} against {}", luma(e));
                assert!(e.iter().zip(&e[1..]).any(|(a, b)| a != b), "the test's edit has chroma at {x},{y}");
            }
            let mut history_colour = neutral;
            history_colour.each[0].saturation = 0.0;
            run(history_colour, None, true, false);
            assert_eq!(lanes(5, 3)[7], centre(want(5, 3, 0)), "the history lane is the network's answer, not the reshaped one");

            // Contrast 0: the change has no luma; its chroma is the edit's.
            let (flat, _) = run(
                neural_forge_protocol::ColourSettings {
                    each: [neural_forge_protocol::EditColour { contrast: 0.0, ..neural_forge_protocol::EditColour::NEUTRAL };
                        neural_forge_protocol::MAX_PASSES],
                    ..neutral
                },
                None,
                false,
                false,
            );
            for (x, y) in [(5, 3), (0, 0), (W - 1, H - 1), (11, 7)] {
                let e: [f32; 3] = std::array::from_fn(|c| want(x, y, c as u32) - mid(x, y, c as u32));
                let f = at(&flat, x, y);
                let d: [f32; 3] = std::array::from_fn(|c| f[c] - mid(x, y, c as u32));
                assert!(luma(d).abs() <= 2f32.powi(-10), "contrast 0 leaves no luma change at {x},{y}: {d:?}");
                for c in 0..3 {
                    assert!((d[c] - (e[c] - luma(e))).abs() <= 2f32.powi(-10), "the chroma stays at {x},{y} channel {c}");
                }
                assert!(luma(e).abs() > 2f32.powi(-8) || (x, y) != (5, 3), "the test's edit has luma");
            }

            // Detail 0: a checkerboard on a broad change of 1/64 leaves the broad change (away from the edges,
            // where the blur clamps).
            let checker = |x: u32, y: u32, c: u32| {
                let sign = if (x + y).is_multiple_of(2) { 1.0 } else { -1.0 };
                1.0 / 64.0 + sign * (c + 1) as f32 / 64.0
            };
            write_head(&checker);
            let (broad, _) = run(
                neural_forge_protocol::ColourSettings {
                    each: [neural_forge_protocol::EditColour { detail: 0.0, ..neural_forge_protocol::EditColour::NEUTRAL }; neural_forge_protocol::MAX_PASSES],
                    ..neutral
                },
                None,
                false,
                false,
            );
            for y in 1..H - 1 {
                for x in 1..W - 1 {
                    for c in 0..3 {
                        let d = at(&broad, x, y)[c as usize] - mid(x, y, c);
                        assert!((d - 1.0 / 64.0).abs() <= 2f32.powi(-10), "detail 0 keeps the broad change only at {x},{y} channel {c}: {d}");
                    }
                }
            }

            // The gamut guard: proxy (0.875, 0.5, 0.5), edit (1/8, -1/32, -1/32); saturation 2 would take red
            // past 1. With the guard the change is shortened along itself; without, red clips and the hue turns.
            fill(&|_, _, c| if c == 0 { 0.875 } else { 0.5 });
            write_head(&|x, _, c| match (x < W / 2, c) {
                (true, 0) => 0.125,
                (true, _) => -0.03125,
                (false, 0) => 0.03125,
                (false, _) => -0.0078125,
            });
            let sat2 = |guard: bool| neural_forge_protocol::ColourSettings {
                each: [neural_forge_protocol::EditColour { saturation: 2.0, ..neural_forge_protocol::EditColour::NEUTRAL }; neural_forge_protocol::MAX_PASSES],
                gamut_guard: guard,
                ..neutral
            };
            let (guarded, _) = run(sat2(true), None, false, false);
            let (clipped, _) = run(sat2(false), None, false, false);
            let shaped = |e: [f32; 3]| -> [f32; 3] { std::array::from_fn(|c| luma(e) + 2.0 * (e[c] - luma(e))) };
            let e = shaped([0.125, -0.03125, -0.03125]);
            assert!(0.875 + e[0] > 1.0, "the test's shaped change leaves the range");
            let g = at(&guarded, 2, 2);
            assert!(g[0] <= 1.0, "the guard keeps red in range");
            let ratio = (g[1] - 0.5) / (g[0] - 0.875);
            assert!((ratio - e[1] / e[0]).abs() < 0.01, "the guarded change keeps its direction: {ratio} against {}", e[1] / e[0]);
            let k = at(&clipped, 2, 2);
            assert_eq!(k[0], 1.0, "without the guard red clips");
            let clipped_ratio = (k[1] - 0.5) / (k[0] - 0.875);
            assert!((clipped_ratio - e[1] / e[0]).abs() > 0.05, "and the clipped change turns: {clipped_ratio}");
            // A pixel whose shaped change stays in range is the same either way, bit for bit.
            let inside = at(&guarded, W - 2, 2);
            assert!(inside[0] < 1.0);
            assert_eq!(inside.map(f32::to_bits), at(&clipped, W - 2, 2).map(f32::to_bits), "in range, the guard changes nothing");

            // The look-up table (a channel rotation, exact under trilinear): strength 0 changes nothing; at full
            // strength the result is the table's where the edit is large, and untouched where there is none.
            fill(&mid);
            write_head(&|x, _, _| if x < W / 2 { 0.0 } else { 0.0625 });
            let mut cube = neural_forge_protocol::cube::Cube::identity(2);
            for e in &mut cube.table {
                *e = [e[2], e[0], e[1]];
            }
            let lut = Arc::new(super::super::colour::Lut { generation: 7, cube });
            let (none, _) = run(neutral, Some(&lut), false, true);
            let (full, _) = run(neural_forge_protocol::ColourSettings { lut_strength: 1.0, lut_adapt: 0.01, ..neutral }, Some(&lut), false, false);
            for (x, y) in [(1, 1), (W - 1, H - 1), (W / 2, 3)] {
                let n = at(&none, x, y);
                let want_n: [f32; 3] = std::array::from_fn(|c| trunc_half(mid(x, y, c as u32) + if x < W / 2 { 0.0 } else { 0.0625 }));
                assert_eq!(n, want_n, "strength 0 is the network's answer at {x},{y}");
                let f = at(&full, x, y);
                if x < W / 2 {
                    assert_eq!(f, n, "no edit, no table at {x},{y}");
                } else {
                    let rotated = [n[2], n[0], n[1]];
                    for c in 0..3 {
                        assert!(
                            (f[c] - rotated[c]).abs() <= 2f32.powi(-10),
                            "the table where the edit is large at {x},{y} channel {c}: {f:?} against {rotated:?}"
                        );
                    }
                }
            }

            pass.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// Two passes, the second with its own intensity: pass 2 reads pass 1's answer as its colour input (its
    /// proxy lanes), its answer is the network's on that picture blended by its own intensity, and the
    /// encoded capture is put back after it for the write-back.
    #[test]
    fn a_later_pass_reads_the_previous_answer_with_its_own_settings_and_the_capture_is_put_back() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
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
                    blend_scale: 0.5,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation: 1,
                network: network::FULL,
            };
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                buffer_row_length: pw,
                image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                ..Default::default()
            };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(encoded.image)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            assert!(pass.ensure_more(device, instance, physical, family, 2), "the later passes build");
            assert!(pass.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
            pass.record_copies(device, encoded.image, encoded.view, region, 2, 1.0).unwrap();
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            let h: Vec<f32> =
                (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, 0), residual(x, y, 1, 0), residual(x, y, 2, 0), 10.0])).collect();
            std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);

            let mut passes = Passes::one(Conditioning::default());
            passes.count = 2;
            passes.each[1].intensity = 0.5;
            let (cmds, later) = pass.frame_passes(&built, false, None, [1.0, 1.0], &passes);
            assert_eq!(later.len(), 1, "one later pass");
            let copies = pass.more.as_ref().unwrap().copies;
            let order: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().chain([copies[0], later[0], copies[2]]).collect();
            device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&order)], vk::Fence::null()).unwrap();
            device.queue_wait_idle(gpu.4).unwrap();
            for cmd in order {
                pass.submitted(cmd);
            }

            // Pass 1's answer is what the one-pass test computes; pass 2's proxy lanes are it, centred.
            let a1 = |x: u32, y: u32, c: u32| neural(x, y, c, 0);
            let lane = |x: u32, y: u32, i: usize| *features.ptr.cast::<f32>().add(((y * FW + x) * 16) as usize + i);
            for c in 0..3 {
                assert_eq!(lane(5, 3, 4 + c as usize), centre(a1(5, 3, c)), "pass 2 reads pass 1's answer, channel {c}");
            }
            let answer = pass.answer();
            fx.one_shot(|cmd| {
                device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
            });
            let a2: Vec<f32> = std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).iter().map(|&h| f16(h)).collect();
            for y in 0..H {
                for x in 0..W {
                    for c in 0..3 {
                        let p = a1(x, y, c);
                        let n = trunc_half(((residual(x, y, c, 0) * 0.03125 + (p * 0.125 - 0.0625)) * 8.0 + 0.5).clamp(0.0, 1.0));
                        let want = trunc_half(0.5f32.mul_add(n - p, p).clamp(0.0, 1.0));
                        let got = a2[((y * pw + x) * 4 + c) as usize];
                        assert!((got - want).abs() <= 2.0 * 2f32.powi(-11), "pass 2 at {x},{y} channel {c}: {got} vs {want}");
                    }
                }
            }
            // The capture is back in the encoded image.
            fx.one_shot(|cmd| device.cmd_copy_image_to_buffer(cmd, encoded.image, vk::ImageLayout::GENERAL, staging.buffer, &[region]));
            let back = std::slice::from_raw_parts(staging.ptr.cast::<u16>(), texels.len());
            assert_eq!(back, &texels[..], "the encoded capture is put back after the last pass");

            // Frame 2, one pass with history: pass 1 blends toward its own previous answer (frame 1's pass 1),
            // not toward the two-pass picture pass 2 made.
            let cmds = pass.frame(&built, true, None, [1.0, 1.0], Conditioning::default());
            fx.submit(&mut pass, cmds);
            assert_ne!(a1(5, 3, 0), a2[((3 * pw + 5) * 4) as usize], "the test tells the two answers apart");
            assert_eq!(lane(5, 3, 7), centre(a1(5, 3, 0)), "pass 1's history is its own answer");

            // Frame 3, two passes without history at a pass feedback of 0.5: pass 2's colour input is halfway
            // from the frame to pass 1's answer (on the encoded image's half grid).
            pass.record_copies(device, encoded.image, encoded.view, region, 2, 0.5).unwrap();
            let (cmds, later) = pass.frame_passes(&built, false, None, [1.0, 1.0], &passes);
            let copies = pass.more.as_ref().unwrap().copies;
            let order: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().chain([copies[0], later[0], copies[2]]).collect();
            device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&order)], vk::Fence::null()).unwrap();
            device.queue_wait_idle(gpu.4).unwrap();
            for cmd in order {
                pass.submitted(cmd);
            }
            for c in 0..3u32 {
                let (x, y) = (5u32, 3u32);
                let input = f16(half(f16(half(proxy(x, y, c))) + 0.5 * (a1(x, y, c) - f16(half(proxy(x, y, c))))));
                assert!(
                    (lane(x, y, 4 + c as usize) - centre(input)).abs() <= 2f32.powi(-12),
                    "pass 2's input at feedback 0.5, channel {c}: {} against {}",
                    lane(x, y, 4 + c as usize),
                    centre(input)
                );
            }

            pass.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }
    /// The control mask: an all-ones mask (the dark rule with every value at 1) gives exactly the answer
    /// and the network inputs of no mask; the sky rule leaves pixels at the far plane as they were and
    /// the rest as without it; the dark rule's structure lowers the auto mask's structure lanes.
    #[test]
    fn the_control_mask_keeps_the_sky_lowers_structure_and_changes_nothing_when_all_ones() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            // Reversed depth: the sky (0) in columns 0-3, the scene (0.5) elsewhere.
            let depth = own_host_buffer_with(device, instance, physical, u64::from(W * H) * 4, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let d: Vec<f32> = (0..H).flat_map(|_| (0..W).map(|x| if x < 4 { 0.0 } else { 0.5 })).collect();
            std::ptr::copy_nonoverlapping(d.as_ptr().cast::<u8>(), depth.ptr, d.len() * 4);
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
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
                    blend_scale: 0.5,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation: 1,
                network: network::FULL,
            };
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(encoded.image)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                let region = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                    ..Default::default()
                };
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            assert!(pass.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, Some(depth.buffer)));
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            let h: Vec<f32> =
                (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, 0), residual(x, y, 1, 0), residual(x, y, 2, 0), 10.0])).collect();
            std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
            let answer = pass.answer();
            // One frame without history with `mask`: the answer and the features.
            let mut run = |mask: Mask| -> (Vec<u16>, Vec<f32>) {
                let mut passes = Passes::one(Conditioning::default());
                passes.mask = mask;
                let (cmds, _) = pass.frame_passes(&built, false, None, [1.0, 1.0], &passes);
                fx.submit(&mut pass, cmds);
                fx.one_shot(|cmd| {
                    device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
                });
                let a = std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).to_vec();
                let f = std::slice::from_raw_parts(features.ptr.cast::<f32>(), (FW * FH * 16) as usize).to_vec();
                (a, f)
            };
            let (a0, f0) = run(Mask::default());
            let ones = Mask { rules: MASK_DARK, dark: [1.0, 1.0, 1.0], ..Mask::default() };
            let (a1, f1) = run(ones);
            assert!(a0 == a1, "an all-ones mask gives the answer of no mask, bit for bit");
            assert!(f0.iter().zip(&f1).all(|(x, y)| x.to_bits() == y.to_bits()), "and the same network inputs");

            // The sky rule: columns 0-1 see only sky in their 5x5 neighbourhood (no edit: the proxy, on the
            // half grid), columns 6 on see none (the network's answer, as without a mask).
            let sky = Mask { rules: MASK_SKY, depth_kind: 1, reversed: true, sky_margin: 0.00001, ..Mask::default() };
            let (a2, _) = run(sky);
            for y in 0..H {
                for c in 0..3u32 {
                    let at = |x: u32| ((y * pw + x) * 4 + c) as usize;
                    assert_eq!(f16(a2[at(0)]), trunc_half(f16(half(proxy(0, y, c)))), "sky at 0,{y} channel {c}: the frame as it was");
                    assert_eq!(f16(a2[at(1)]), trunc_half(f16(half(proxy(1, y, c)))), "sky at 1,{y} channel {c}");
                    assert_eq!(a2[at(7)], a0[at(7)], "scene at 7,{y} channel {c}: as without the mask");
                }
            }
            assert_ne!(a2[(2 * pw * 4) as usize], a0[(2 * pw * 4) as usize], "the test's sky pixel is edited without the mask");
            // Ordinary depth: the same picture's 0 is the near plane, so nothing is sky.
            let (a3, _) = run(Mask { reversed: false, ..sky });
            assert!(a3 == a0, "with ordinary depth no pixel of this picture is sky");

            // The dark rule's structure (intensity left at 1): b fades from 1 at the threshold (1.0, so every
            // pixel) toward 0.5 at black; with auto mask on, the skin and structure lanes take min(lane, b).
            let dark = Mask { rules: MASK_DARK, dark: [1.0, 1.0, 0.5], ..Mask::default() };
            let (_, f4) = run(dark);
            for (x, y) in [(5u32, 3u32), (9, 6)] {
                let luma = 0.2126 * f16(half(proxy(x, y, 0))) + 0.7152 * f16(half(proxy(x, y, 1))) + 0.0722 * f16(half(proxy(x, y, 2)));
                let b = 1.0 - 0.5 * (1.0 - luma).clamp(0.0, 1.0);
                let q = (b * 255.0).round() / 255.0;
                let lane = |i: usize| f4[((y * FW + x) * 16) as usize + i];
                assert_eq!(lane(12), 1.0, "auto mask on: lane 12 stays 1");
                for i in [13, 14] {
                    assert!((lane(i) - q).abs() < 2e-3, "lane {i} at {x},{y}: {} against {q}", lane(i));
                }
                assert_eq!(lane(11), f0[((y * FW + x) * 16) as usize + 11], "local tone untouched (g stays 1)");
            }

            pass.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging, &depth] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// A cut the GPU reset on the last network frame (read at the start of the hold) stays counted through a
    /// network frame that consumes no latched cut, as inside DLSS's buffer: the split hold overwrote it, so
    /// the session's count missed every such cut with the edit cache off or the camera still.
    #[test]
    fn a_cut_the_gpu_reset_stays_counted_through_the_next_network_frame() {
        for (frames, stale, before, after) in [
            (3, None, true, true),
            (3, Some(Stale::Idle(std::time::Duration::from_secs(1))), true, true),
            (3, Some(Stale::SceneCut(60)), false, true),
            (3, None, false, false),
            (0, Some(Stale::SceneCut(60)), false, false),
        ] {
            let mut cut = before;
            note_network_reset(frames, stale, &mut cut);
            assert_eq!(cut, after, "frames {frames}, {stale:?}, counted before: {before}");
        }
    }

    /// A frame whose capture the scene-cut detector marked (the state buffer's `cut` word) runs every pass
    /// exactly as a frame without history does (the same lanes, seed 0, no blend, the same answer), and with
    /// the seed restarted after it (`restart_seed`, what the next hold does on the reading) the next frame is
    /// bit for bit the one after a reset. Without the mark the frame blends toward its history.
    #[test]
    fn a_scene_cut_frame_runs_every_pass_as_a_reset_and_the_next_frame_follows_it() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (_, instance, physical, device, _, family) = (&gpu.0, &gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(W, H);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let features = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, u64::from(FW * FH) * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
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
                    blend_scale: 0.5,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation: 1,
                network: network::FULL,
            };
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                buffer_row_length: pw,
                image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                ..Default::default()
            };
            fx.one_shot(|cmd| {
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(encoded.image)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
            });
            let mut passes = Passes::one(Conditioning::default());
            passes.count = 2;
            passes.each[1].intensity = 0.5;
            let answer_of = |pass: &NativePass| -> Vec<u16> {
                let answer = pass.answer();
                fx.one_shot(|cmd| {
                    device.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }])
                });
                std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).to_vec()
            };
            let lanes = || -> Vec<u32> { std::slice::from_raw_parts(features.ptr.cast::<u32>(), (FW * FH * 16) as usize).to_vec() };
            // One network frame of both passes: `history` the CPU's flag, `cut` the detector's word.
            let frame = |pass: &mut NativePass, n: u32, history: bool, cut: bool| -> (Vec<u16>, Vec<u32>, u32) {
                let h: Vec<f32> =
                    (0..FH).flat_map(|y| (0..FW).flat_map(move |x| [residual(x, y, 0, n), residual(x, y, 1, n), residual(x, y, 2, n), 10.0])).collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
                pass.record_copies(device, encoded.image, encoded.view, region, 2, 1.0).unwrap();
                let (cmds, later) = pass.frame_passes(&built, history, None, [1.0, 1.0], &passes);
                assert_eq!(later.len(), 1, "one later pass");
                let seed = (*pass.params[((pass.frames - 1) & 1) as usize].ptr.cast::<Params>()).frame[1];
                let copies = pass.more.as_ref().unwrap().copies;
                let state = pass.state_buffer();
                let order: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().chain([copies[0], later[0], copies[2]]).collect();
                // The initialisation clears the state; the capture writes the cut word after it, before the frame.
                let init_cmd = pass.commands[2];
                let (init, rest): (Vec<vk::CommandBuffer>, Vec<vk::CommandBuffer>) = order.iter().copied().partition(|&c| c == init_cmd);
                if !init.is_empty() {
                    device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&init)], vk::Fence::null()).unwrap();
                    device.queue_wait_idle(gpu.4).unwrap();
                }
                fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, state, 8, 4, u32::from(cut)));
                device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&rest)], vk::Fence::null()).unwrap();
                device.queue_wait_idle(gpu.4).unwrap();
                for cmd in order {
                    pass.submitted(cmd);
                }
                if let Some(m) = pass.more.as_mut() {
                    m.fresh[0] = false;
                }
                (answer_of(pass), lanes(), seed)
            };
            let build = || {
                let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
                assert!(pass.ensure_more(device, instance, physical, family, 2), "the later passes build");
                assert!(pass.record(device, instance, physical, &built, graph, encoded.view, exposure.buffer, None));
                fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
                pass
            };

            // The reference: frame 2 without history (a reset on the CPU), frame 3 with it.
            let mut reset = build();
            frame(&mut reset, 0, false, false);
            let (ra2, rl2, rs2) = frame(&mut reset, 1, false, false);
            let (ra3, rl3, rs3) = frame(&mut reset, 2, true, false);
            assert_eq!((rs2, rs3), (0, 1), "a reset's seeds");
            // Frame 2 cut: the CPU kept the history (seed 1), the detector marked the frame.
            let mut cut = build();
            frame(&mut cut, 0, false, false);
            let (ca2, cl2, cs2) = frame(&mut cut, 1, true, true);
            assert_eq!(cs2, 1, "the CPU's parameters kept the history");
            assert_eq!(cl2, rl2, "every lane of the last pass (noise seed 0, no history) as on a reset");
            assert_eq!(ca2, ra2, "the answer of both passes as on a reset");
            cut.restart_seed();
            let (ca3, cl3, cs3) = frame(&mut cut, 2, true, false);
            assert_eq!(cs3, 1, "the seed restarts after the cut");
            assert_eq!(cl3, rl3, "the frame after the cut is the frame after a reset");
            assert_eq!(ca3, ra3);
            // Without the mark the same frame 2 keeps its history: the test tells the two apart.
            let mut kept = build();
            frame(&mut kept, 0, false, false);
            let (ka2, _, _) = frame(&mut kept, 1, true, false);
            assert_ne!(ka2, ra2, "with history the answer blends toward it");

            for p in [&reset, &cut, &kept] {
                p.destroy(device);
            }
            encoded.destroy(device);
            for b in [&exposure, &features, &head, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// The next hold's side of a reading: a cut the GPU reset restarts the seed and is counted; a cut on a
    /// cached hold resets the next network frame (its parameters' history off, counted then) and makes the
    /// edit cache refresh instead of carrying its edit on; no cut changes nothing.
    #[test]
    fn a_reading_restarts_the_seed_or_resets_the_next_network_frame_and_refreshes_the_cache() {
        use super::super::cache::{Hold, Kind, State};
        use super::super::scene_cut::Effect;
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let hold = |n: u64| Hold {
            interval: 2,
            hold: n,
            jitter: Some([0.0; 2]),
            mvec: true,
            depth: true,
            history_ok: true,
            armed: false,
            debug_mask: false,
            generation: 1,
        };
        // A cache that just refreshed: its next hold is cached.
        let refreshed = || {
            let mut s = State::default();
            let h = hold(1);
            let p = s.plan(&h);
            assert_eq!(p.kind, Kind::Refresh);
            s.went_through(&h, &p);
            s
        };
        let built = Built {
            width: W,
            height: H,
            frame: nn::Frame {
                field_width: FW,
                field_height: FH,
                features: vk::Buffer::null(),
                features_bytes: 0,
                head: vk::Buffer::null(),
                head_bytes: 0,
                blend_scale: 0.5,
                chained: 0,
            },
            graph: vk::CommandBuffer::null(),
            graph_own: vk::CommandBuffer::null(),
            generation: 1,
            network: network::FULL,
        };
        let seed = |pass: &NativePass| unsafe { (*pass.params[((pass.frames - 1) & 1) as usize].ptr.cast::<Params>()).frame[1] };
        // SAFETY: a fresh device; the pass is never submitted.
        unsafe {
            let mut pass = NativePass::build(device, instance, physical, family, W, H, false).expect("the native pass builds");
            let t0 = Instant::now();
            for _ in 0..3 {
                pass.gap.begin(t0);
                pass.frame_passes(&built, true, None, [1.0, 1.0], &Passes::default());
            }
            assert_eq!(seed(&pass), 3);
            let mut cache = refreshed();
            // No cut: nothing changes.
            assert!(!apply_scene_cut(Effect::None, Some(&mut pass), Some(&mut cache)));
            assert_eq!(pass.gap.pending(t0), None);
            assert_eq!(cache.plan(&hold(2)).kind, Kind::Cached, "the cache carries its edit");
            // The GPU reset the last network frame: counted, and the next frame's seed is a reset's next.
            assert!(apply_scene_cut(Effect::ResetOnGpu(80), Some(&mut pass), None));
            assert_eq!(pass.gap.pending(t0), None, "the reset already happened");
            pass.frame_passes(&built, true, None, [1.0, 1.0], &Passes::default());
            assert_eq!(seed(&pass), 1, "the seed counts on from the cut frame's 0");
            // A cut on a cached hold: the next network frame resets, the cache refreshes.
            let mut cache = refreshed();
            assert!(!apply_scene_cut(Effect::Latch(90), Some(&mut pass), Some(&mut cache)), "counted when it is consumed");
            assert_eq!(cache.plan(&hold(2)).kind, Kind::Refresh, "the cache refreshes across the cut");
            let stale = pass.gap.begin(t0);
            assert_eq!(stale, Some(Stale::SceneCut(90)));
            // run_native_hold's rule for the frame's history flag.
            assert!(!(pass.frames > 0 && stale.is_none()), "the next network frame has no history");
            pass.frame_passes(&built, false, None, [1.0, 1.0], &Passes::default());
            assert_eq!((*pass.params[((pass.frames - 1) & 1) as usize].ptr.cast::<Params>()).frame[2], 0, "its parameters' history flag is off");
            assert_eq!(pass.gap.begin(t0), None, "once");
            pass.destroy(device);
        }
    }

    /// The render extent of the model resolution tests (half of it is above the 8-texel floor).
    const SW: u32 = 20;
    const SH: u32 = 16;

    /// A network stand-in for one extent: host-visible features and head, the given field.
    struct Stand {
        built: Built,
        features: HostBuffer,
        head: HostBuffer,
    }

    impl Stand {
        /// # Safety
        /// `device` is live; `graph` is an empty secondary of it.
        unsafe fn new(
            device: &ash::Device,
            instance: &ash::Instance,
            physical: vk::PhysicalDevice,
            extent: (u32, u32),
            field: (u32, u32),
            graph: vk::CommandBuffer,
            generation: u64,
        ) -> Self {
            let texels = u64::from(field.0 * field.1);
            let features = own_host_buffer_with(device, instance, physical, texels * 64, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let head = own_host_buffer_with(device, instance, physical, texels * 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let built = Built {
                width: extent.0,
                height: extent.1,
                frame: nn::Frame {
                    field_width: field.0,
                    field_height: field.1,
                    features: features.buffer,
                    features_bytes: texels * 64,
                    head: head.buffer,
                    head_bytes: texels * 16,
                    blend_scale: 0.5,
                    chained: 0,
                },
                graph,
                graph_own: graph,
                generation,
                network: network::FULL,
            };
            Self { built, features, head }
        }

        /// The head: `residual(x, y, c)` in every field texel, logit 10.
        fn write_head(&self, residual: impl Fn(u32, u32, u32) -> f32) {
            let (fw, fh) = (self.built.frame.field_width, self.built.frame.field_height);
            let h: Vec<f32> =
                (0..fh).flat_map(|y| (0..fw).flat_map(|x| [residual(x, y, 0), residual(x, y, 1), residual(x, y, 2), 10.0]).collect::<Vec<_>>()).collect();
            // SAFETY: mapped for the whole field.
            unsafe { std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), self.head.ptr, h.len() * 4) };
        }

        /// Input lane `i` of field texel (x, y).
        fn lane(&self, x: u32, y: u32, i: usize) -> f32 {
            // SAFETY: mapped for the whole field; the frame that wrote it completed.
            unsafe { *self.features.ptr.cast::<f32>().add(((y * self.built.frame.field_width + x) * 16) as usize + i) }
        }

        fn graph(&self) -> Graph {
            Graph { built: self.built, graph: self.built.graph }
        }

        /// # Safety
        /// Nothing pending uses them.
        unsafe fn destroy(&self, device: &ash::Device) {
            // SAFETY: forwarded.
            unsafe {
                self.features.destroy(device);
                self.head.destroy(device);
            }
        }
    }

    /// An empty secondary (the graph's place) and its pool.
    ///
    /// # Safety
    /// `device` is live.
    unsafe fn empty_graph(device: &ash::Device, family: u32) -> (vk::CommandPool, vk::CommandBuffer) {
        // SAFETY: forwarded; valid create infos.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
            let graph = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1),
                )
                .unwrap()[0];
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            device
                .begin_command_buffer(
                    graph,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance),
                )
                .unwrap();
            device.end_command_buffer(graph).unwrap();
            (pool, graph)
        }
    }

    /// `texels` (RGBA16F rows of the padded width) into `image`, left in GENERAL.
    fn upload_encoded(fx: &Fixture, staging: &HostBuffer, image: &OwnImage, pw: u32, ph: u32, texels: &[u16]) {
        let d = &fx.gpu.3;
        // SAFETY: the staging buffer is mapped for the texels; the image is the test's.
        unsafe {
            std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
            fx.one_shot(|cmd| {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let to_general = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.image)
                    .subresource_range(range);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
                let region = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                    ..Default::default()
                };
                d.cmd_copy_buffer_to_image(cmd, staging.buffer, image.image, vk::ImageLayout::GENERAL, &[region]);
            });
        }
    }

    /// The answer buffer (RGBA16F rows of the padded width), as f32.
    fn read_answer(fx: &Fixture, staging: &HostBuffer, answer: vk::Buffer, pw: u32, ph: u32) -> Vec<f32> {
        let d = &fx.gpu.3;
        // SAFETY: as `upload_encoded`.
        unsafe {
            fx.one_shot(|cmd| d.cmd_copy_buffer(cmd, answer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: u64::from(pw * ph) * 8 }]));
            std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (pw * ph * 4) as usize).iter().map(|&h| f16(h)).collect()
        }
    }

    #[test]
    fn the_model_extent_follows_the_scale_and_full_size_is_the_render_extent() {
        for s in [1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(model_extent(1485, 836, s), (1485, 836), "scale {s}");
        }
        assert_eq!(model_extent(1485, 836, 0.5), (743, 418));
        assert_eq!(model_extent(1485, 836, 0.667), (990, 558));
        assert_eq!(model_extent(1485, 836, 0.333), (495, 278));
        assert_eq!(model_extent(1485, 836, 0.1), (371, 209), "at least the setting's lowest scale");
        assert_eq!(model_extent(12, 6, 0.25), (8, 6), "at least 8 texels, never more than the side");
        // At 1 every pass is at the render extent and the network is built for that one size, as before.
        let full = pass_extents(SW, SH, &[1.0; neural_forge_protocol::MAX_PASSES]);
        assert!(full.iter().all(|&e| e == (SW, SH)));
        assert_eq!(graph_sizes(&full, 3), vec![(SW, SH)]);
        let mut scales = [1.0; neural_forge_protocol::MAX_PASSES];
        scales[1] = 0.5;
        scales[2] = 0.5;
        scales[4] = 0.333;
        let mixed = pass_extents(SW, SH, &scales);
        assert_eq!(graph_sizes(&mixed, 3), vec![(SW, SH), (10, 8)], "pass 1's first, each size once");
        assert_eq!(graph_sizes(&mixed, 5), vec![(SW, SH), (10, 8), (8, 8)], "never below 8 texels");
        assert_eq!(graph_sizes(&mixed, 1), vec![(SW, SH)], "only the passes that run");
    }

    /// A model resolution of 1 is the native frame as it was, bit for bit: a pass built for every slot at the
    /// render extent (what the hold builds at the default) gives the same input lanes and answers as one built
    /// the old way, over two passes and frames with history.
    #[test]
    fn at_full_model_resolution_the_frame_is_bit_for_bit_as_before() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(SW, SH);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let (graph_pool, graph) = empty_graph(device, family);
            let net = Stand::new(device, instance, physical, (SW, SH), (FW, FH), graph, 1);
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                buffer_row_length: pw,
                image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                ..Default::default()
            };
            let extents = pass_extents(SW, SH, &[1.0; neural_forge_protocol::MAX_PASSES]);
            let mut old = NativePass::build(device, instance, physical, family, SW, SH, false).unwrap();
            let mut new = NativePass::build_at(device, instance, physical, family, SW, SH, extents, false, true).unwrap();
            let mut passes = Passes::one(Conditioning::default());
            passes.count = 2;
            passes.each[1].intensity = 0.5;
            let mut results = Vec::new();
            for (k, pass) in [&mut old, &mut new].into_iter().enumerate() {
                assert!(pass.ensure_more(device, instance, physical, family, 2));
                assert!(pass.ensure_lanes(device, instance, physical, 2), "nothing to build at full size");
                assert!(pass.lanes.is_empty() && pass.resize.is_none(), "no lane or resize pipeline at full size");
                let graphs = [net.graph()];
                assert!(if k == 0 {
                    pass.record(device, instance, physical, &net.built, graph, encoded.view, exposure.buffer, None)
                } else {
                    pass.record_graphs(device, instance, physical, &graphs, encoded.view, exposure.buffer, None)
                });
                fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
                let mut frames = Vec::new();
                for frame in 0..3u32 {
                    upload_encoded(&fx, &staging, &encoded, pw, ph, &texels);
                    net.write_head(|x, y, c| residual(x, y, c, frame));
                    // Recorded for each frame, as each hold does: they are one-time-submit buffers.
                    pass.record_copies(device, encoded.image, encoded.view, region, 2, 1.0).unwrap();
                    let (cmds, later) = if k == 0 {
                        pass.frame_passes(&net.built, frame > 0, Some([0.25, -0.25]), [1.0, 1.0], &passes)
                    } else {
                        pass.frame_graphs(&graphs, frame > 0, Some([0.25, -0.25]), [1.0, 1.0], &passes)
                    };
                    let copies = pass.more.as_ref().unwrap().copies;
                    let order: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().chain([copies[0], later[0], copies[2]]).collect();
                    device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&order)], vk::Fence::null()).unwrap();
                    device.queue_wait_idle(gpu.4).unwrap();
                    for cmd in order {
                        pass.submitted(cmd);
                    }
                    // The submit took the later pass's new history to GENERAL (the hold clears it the same way).
                    if let Some(m) = pass.more.as_mut() {
                        m.fresh[0] = false;
                    }
                    let lanes: Vec<u32> = (0..FW * FH * 16).map(|i| (*net.features.ptr.cast::<f32>().add(i as usize)).to_bits()).collect();
                    frames.push((lanes, read_answer(&fx, &staging, pass.answer(), pw, ph).iter().map(|v| v.to_bits()).collect::<Vec<u32>>()));
                }
                results.push(frames);
            }
            for (frame, (a, b)) in results[0].iter().zip(&results[1]).enumerate() {
                assert!(a.0 == b.0, "frame {frame}: the network's input lanes differ");
                assert!(a.1 == b.1, "frame {frame}: the answer differs");
            }
            for pass in [&old, &new] {
                pass.destroy(device);
            }
            net.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// At half the model resolution: the network's input is the area average of the encoded frame at half
    /// its size, and its edit comes back at full size: a constant residual (a constant edit) is added to every
    /// full-resolution pixel with its own magnitude; an edit of opposite sign on either side of a hard edge in
    /// the frame stays on its side up to the edge, where a plain bilinear enlargement would mix the two over
    /// the pixels next to it.
    #[test]
    fn at_half_model_resolution_the_edit_comes_back_at_full_size_without_crossing_an_edge() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let mut scales = [1.0; neural_forge_protocol::MAX_PASSES];
        scales[0] = 0.5;
        let extents = pass_extents(SW, SH, &scales);
        let (mw, mh) = extents[0];
        assert_eq!((mw, mh), (10, 8));
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(SW, SH);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let (graph_pool, graph) = empty_graph(device, family);
            let net = Stand::new(device, instance, physical, (mw, mh), (16, 8), graph, 1);
            // The frame: dark on the left half, bright on the right (a hard edge between columns 9 and 10, on a
            // texel boundary of the half-size picture), a fine texture on each side.
            let frame = |x: u32, y: u32, c: u32| -> f32 { (if x < SW / 2 { 0.25 } else { 0.75 }) + ((x + 2 * y + c) % 4) as f32 / 256.0 };
            let texels: Vec<u16> = (0..ph)
                .flat_map(|y| {
                    (0..pw).flat_map(move |x| {
                        [
                            half(frame(x.min(SW - 1), y.min(SH - 1), 0)),
                            half(frame(x.min(SW - 1), y.min(SH - 1), 1)),
                            half(frame(x.min(SW - 1), y.min(SH - 1), 2)),
                            half(1.0),
                        ]
                    })
                })
                .collect();
            upload_encoded(&fx, &staging, &encoded, pw, ph, &texels);
            let mut pass = NativePass::build_at(device, instance, physical, family, SW, SH, extents, false, true).unwrap();
            assert!(!pass.record_graphs(device, instance, physical, &[net.graph()], encoded.view, exposure.buffer, None), "pass 1's lane is needed first");
            assert!(pass.ensure_lanes(device, instance, physical, 1));
            let graphs = [net.graph()];
            assert!(pass.record_graphs(device, instance, physical, &graphs, encoded.view, exposure.buffer, None));
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            // The CPU's picture of the network's input: each texel the mean of its 2x2 render pixels.
            let small = |x: u32, y: u32, c: u32| (0..4).map(|k| f16(half(frame(2 * x + k % 2, 2 * y + k / 2, c)))).sum::<f32>() / 4.0;
            // The full-size answer as an edit of the frame.
            let run = |pass: &mut NativePass, edit: &dyn Fn(u32) -> f32| -> Vec<f32> {
                // A residual of 4e is an edit of e (the composite's residual / 4).
                net.write_head(|x, _, _| 4.0 * edit(x));
                let (cmds, _) = pass.frame_graphs(&graphs, false, None, [1.0, 1.0], &Passes::default());
                fx.submit(pass, cmds);
                let a = read_answer(&fx, &staging, pass.answer(), pw, ph);
                (0..SH)
                    .flat_map(|y| (0..SW).flat_map(move |x| (0..3).map(move |c| (x, y, c))))
                    .map(|(x, y, c)| a[((y * pw + x) * 4 + c) as usize] - f16(half(frame(x, y, c))))
                    .collect()
            };
            // A constant edit of 0.1.
            let e = run(&mut pass, &|_| 0.1);
            for x in 0..mw {
                for c in 0..3 {
                    let got = net.lane(x, 2, 4 + c as usize);
                    let want = (small(x, 2, c) - 0.5) * 0.125;
                    assert!((got - want).abs() < 2e-4, "the network's input at ({x}, 2) channel {c}: {got} against the area average's {want}");
                }
            }
            for (i, v) in e.iter().enumerate() {
                assert!((v - 0.1).abs() < 2e-3, "pixel {} channel {}: an edit of {v}, not 0.1", i / 3 % SW as usize, i % 3);
            }
            // +0.1 on the left half of the network's texels, -0.1 on the right: every pixel takes its own side's.
            let e = run(&mut pass, &|x| if x < mw / 2 { 0.1 } else { -0.1 });
            for y in 0..SH {
                for x in 0..SW {
                    let want = if x < SW / 2 { 0.1 } else { -0.1 };
                    for c in 0..3 {
                        let got = e[((y * SW + x) * 3 + c) as usize];
                        assert!((got - want).abs() < 4e-3, "pixel ({x}, {y}) channel {c}: an edit of {got}, its side's is {want}");
                    }
                }
            }
            pass.destroy(device);
            net.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
            assert_eq!(LIVE.with(std::cell::Cell::get), 0, "everything the pass made is destroyed");
        }
    }

    /// Another model resolution builds the pass again: the hold's rule takes the old pass out (to destroy), and
    /// the new one's first frame runs without history (its lanes 7-9 are the frame, not the previous answer),
    /// while the same resolution keeps the pass and its history.
    #[test]
    fn another_model_resolution_starts_the_history_afresh() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(SW, SH);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let (graph_pool, graph) = empty_graph(device, family);
            let full = Stand::new(device, instance, physical, (SW, SH), (FW, FH), graph, 1);
            let half_net = Stand::new(device, instance, physical, (10, 8), (16, 8), graph, 2);
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            upload_encoded(&fx, &staging, &encoded, pw, ph, &texels);
            let at = |s: f32| {
                let mut scales = [1.0; neural_forge_protocol::MAX_PASSES];
                scales[0] = s;
                pass_extents(SW, SH, &scales)
            };
            // The hold's frame: the pass for these extents (the old one taken out when it is for others), its
            // history flag by the hold's rule (a pass that ran a frame before).
            let mut slot: Option<NativePass> = None;
            let hold = |slot: &mut Option<NativePass>, scale: f32, net: &Stand, frame: u32| {
                let extents = at(scale);
                if let Some(old) = take_other_extents(slot, SW, SH, &extents) {
                    old.destroy(device);
                }
                let mut pass = slot.take().unwrap_or_else(|| NativePass::build_at(device, instance, physical, family, SW, SH, extents, false, true).unwrap());
                assert!(pass.ensure_lanes(device, instance, physical, 1));
                assert!(pass.record_graphs(device, instance, physical, &[net.graph()], encoded.view, exposure.buffer, None));
                if pass.frames == 0 {
                    fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
                }
                net.write_head(|x, y, c| residual(x, y, c, frame));
                let history = pass.frames > 0;
                let (cmds, _) = pass.frame_graphs(&[net.graph()], history, None, [1.0, 1.0], &Passes::default());
                fx.submit(&mut pass, cmds);
                *slot = Some(pass);
                history
            };
            assert!(!hold(&mut slot, 1.0, &full, 0), "the first frame has no history");
            assert!(hold(&mut slot, 1.0, &full, 1), "the same model resolution keeps the pass and its history");
            assert_eq!(full.lane(5, 3, 7), centre(neural(5, 3, 0, 0)), "lanes 7-9 hold the previous answer");
            assert!(!hold(&mut slot, 0.5, &half_net, 2), "another model resolution: a new pass, no history");
            assert_eq!(slot.as_ref().unwrap().extents[0], (10, 8));
            let input = half_net.lane(2, 1, 4);
            assert_eq!(half_net.lane(2, 1, 7), input, "lanes 7-9 are the frame itself: the history started afresh");
            assert!(hold(&mut slot, 0.5, &half_net, 3), "and is kept at that resolution");
            assert!(!hold(&mut slot, 1.0, &full, 4), "back to full size: afresh again");
            slot.take().unwrap().destroy(device);
            full.destroy(device);
            half_net.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// Two passes at their own model resolutions (pass 1 at full size, pass 2 at half): each runs at its own
    /// extent with its own network build: pass 1's input lanes are the frame at full size, pass 2's the area
    /// average of pass 1's full-size answer at half size, and pass 2's history is at half size. With pass 2's
    /// edit zero the answer is pass 1's.
    #[test]
    fn each_pass_runs_at_its_own_model_resolution() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let mut scales = [1.0; neural_forge_protocol::MAX_PASSES];
        scales[1] = 0.5;
        let extents = pass_extents(SW, SH, &scales);
        assert_eq!(graph_sizes(&extents, 2), vec![(SW, SH), (10, 8)]);
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let fx = Fixture { gpu: gpu.clone(), pool };
            let (pw, ph) = super::super::padded(SW, SH);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC).unwrap();
            let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let staging = own_host_buffer_with(
                device,
                instance,
                physical,
                u64::from(pw * ph) * 8,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .unwrap();
            let (graph_pool, graph) = empty_graph(device, family);
            // One build of two sizes: one generation.
            let first = Stand::new(device, instance, physical, (SW, SH), (FW, FH), graph, 7);
            let second = Stand::new(device, instance, physical, (10, 8), (16, 8), graph, 7);
            let texels: Vec<u16> =
                (0..ph).flat_map(|y| (0..pw).flat_map(move |x| [half(proxy(x, y, 0)), half(proxy(x, y, 1)), half(proxy(x, y, 2)), half(1.0)])).collect();
            upload_encoded(&fx, &staging, &encoded, pw, ph, &texels);
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                buffer_row_length: pw,
                image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 },
                ..Default::default()
            };
            let mut pass = NativePass::build_at(device, instance, physical, family, SW, SH, extents, false, true).unwrap();
            assert!(pass.ensure_more(device, instance, physical, family, 2));
            assert!(pass.ensure_lanes(device, instance, physical, 2));
            assert_eq!(pass.lanes.len(), 1, "one lane, pass 2's");
            let graphs = [first.graph(), second.graph()];
            assert!(pass.record_graphs(device, instance, physical, &graphs, encoded.view, exposure.buffer, None));
            pass.record_copies(device, encoded.image, encoded.view, region, 2, 1.0).unwrap();
            fx.one_shot(|cmd| device.cmd_fill_buffer(cmd, pass.acc.buffer, 0, vk::WHOLE_SIZE, 0));
            first.write_head(|x, y, c| residual(x, y, c, 0));
            second.write_head(|_, _, _| 0.0);
            let mut passes = Passes::one(Conditioning::default());
            passes.count = 2;
            passes.scales = scales;
            let (cmds, later) = pass.frame_graphs(&graphs, false, None, [1.0, 1.0], &passes);
            assert_eq!(later.len(), 1, "pass 2 runs");
            let copies = pass.more.as_ref().unwrap().copies;
            let order: Vec<vk::CommandBuffer> = cmds.into_iter().flatten().chain([copies[0], later[0], copies[2]]).collect();
            device.queue_submit(gpu.4, &[vk::SubmitInfo::default().command_buffers(&order)], vk::Fence::null()).unwrap();
            device.queue_wait_idle(gpu.4).unwrap();
            for cmd in order {
                pass.submitted(cmd);
            }
            // Pass 1 at full size: its lanes are the frame.
            for c in 0..3 {
                assert_eq!(first.lane(5, 3, 4 + c as usize), centre(proxy(5, 3, c)), "pass 1's input, channel {c}");
            }
            // Pass 2 at half size: its lanes are the 2x2 means of pass 1's answer.
            let a1 = |x: u32, y: u32, c: u32| neural(x, y, c, 0);
            for (x, y) in [(0, 0), (2, 1), (9, 5)] {
                for c in 0..3 {
                    let mean = (0..4).map(|k| a1(2 * x + k % 2, 2 * y + k / 2, c)).sum::<f32>() / 4.0;
                    let got = second.lane(x, y, 4 + c as usize);
                    assert!((got - (mean - 0.5) * 0.125).abs() < 2e-4, "pass 2's input at ({x}, {y}) channel {c}: {got} against {}", (mean - 0.5) * 0.125);
                }
            }
            let m = pass.more.as_ref().unwrap();
            assert_eq!(m.recorded_pass[0], Some(7));
            assert!(m.history[0].is_some() && pass.extents[1] == (10, 8), "pass 2's history is at its extent");
            // Pass 2's params: its extent and field, the vectors' scale at its size.
            let p2 = &*m.params[0][0].ptr.cast::<Params>();
            assert_eq!(p2.dims, [16, 8, 10, 8]);
            assert_eq!(p2.full[..3], [SW, SH, pw]);
            // Its edit is zero: the answer is pass 1's, at full size.
            let a = read_answer(&fx, &staging, pass.answer(), pw, ph);
            for y in 0..SH {
                for x in 0..SW {
                    for c in 0..3 {
                        let got = a[((y * pw + x) * 4 + c) as usize];
                        assert!((got - a1(x, y, c)).abs() < 2e-3, "({x}, {y}) channel {c}: {got} against pass 1's {}", a1(x, y, c));
                    }
                }
            }
            pass.destroy(device);
            first.destroy(device);
            second.destroy(device);
            encoded.destroy(device);
            for b in [&exposure, &staging] {
                b.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
            device.destroy_command_pool(pool, None);
        }
    }

    /// The per-pass split: two passes, one placed before the upscaler and one after DLSS. Each stage's hold
    /// runs its own pass only, with that pass's own settings (pass 1's intensity before the upscaler, pass 2's
    /// after DLSS) and no later pass recorded, and both holds ask the network for the same two sizes (render and
    /// output), so neither rebuilds it for the other. The look-up table stays with the stack's last pass.
    #[test]
    fn each_stage_records_only_its_own_passes() {
        use super::super::placement::{Frame, Hold, Plan, Stage};
        use neural_forge_protocol::enums::placement::{BEFORE, UPSCALED};
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let (render, output) = ((SW, SH), (2 * SW, 2 * SH));
        let mut all = Passes::one(Conditioning::default());
        all.count = 2;
        all.each[0].intensity = 0.25;
        all.each[1].intensity = 0.75;
        all.colour.lut_strength = 0.5;
        let mut settings = [BEFORE; neural_forge_protocol::MAX_PASSES];
        settings[1] = UPSCALED;
        let plan = Plan::of(2, &settings, Frame { rr: false, output: true, hook: true });
        let before = Hold::of(&plan, Stage::Before, render, output);
        let after = Hold::of(&plan, Stage::Upscaled, render, output);
        assert_eq!(frame_sizes(&all, before, render), vec![render, output], "both holds' sizes, the render size first");
        assert_eq!(frame_sizes(&all, after, output), frame_sizes(&all, before, render), "the same set from either hold");
        assert_eq!(frame_sizes(&all, Hold::ALL_BEFORE, render), vec![render], "every pass before the upscaler: one size, as always");
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let (graph_pool, graph) = empty_graph(device, family);
            let stands = [
                Stand::new(device, instance, physical, render, (FW, FH), graph, 3),
                Stand::new(device, instance, physical, output, (2 * FW, 2 * FH), graph, 3),
            ];
            let graphs = [stands[0].graph(), stands[1].graph()];
            for (hold, extent, intensity, lut) in [(before, render, 0.25, 0.0), (after, output, 0.75, 0.5)] {
                let passes = stage_passes(&all, hold).expect("the stage has a pass");
                assert_eq!((passes.count, passes.each[0].intensity, passes.colour.lut_strength), (1, intensity, lut), "{:?}", hold.stage);
                let (pw, ph) = super::super::padded(extent.0, extent.1);
                let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC).unwrap();
                let exposure = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
                let extents = pass_extents(extent.0, extent.1, &passes.scales);
                let mut pass = NativePass::build_at(device, instance, physical, family, extent.0, extent.1, extents, false, true).unwrap();
                assert!(pass.ensure_lanes(device, instance, physical, passes.count));
                assert!(pass.record_graphs(device, instance, physical, &graphs, encoded.view, exposure.buffer, None));
                let (cmds, later) = pass.frame_graphs(&graphs, false, None, [1.0, 1.0], &passes);
                assert!(cmds[1].is_some() && later.is_empty(), "{:?}: one pass, no later one", hold.stage);
                assert!(pass.more.is_none(), "{:?}: no later pass's buffers", hold.stage);
                let p = &*pass.params[0].ptr.cast::<Params>();
                assert_eq!(p.motion[3], intensity, "{:?}: its own pass's intensity", hold.stage);
                assert_eq!(p.dims[2..], [extent.0, extent.1], "{:?}: at its own size", hold.stage);
                pass.destroy(device);
                encoded.destroy(device);
                exposure.destroy(device);
            }
            for st in &stands {
                st.destroy(device);
            }
            device.destroy_command_pool(graph_pool, None);
        }
        // A stage with none of the frame's passes runs nothing.
        let mut settings = [BEFORE; neural_forge_protocol::MAX_PASSES];
        settings[0] = UPSCALED;
        let plan = Plan::of(2, &settings, Frame { rr: false, output: true, hook: true });
        assert!(stage_passes(&all, Hold::of(&plan, Stage::Before, render, output)).is_none());
    }
}
