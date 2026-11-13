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
use std::time::{Duration, Instant};

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

/// The bound on the network's waits on a queue shared with the application ([`Setup::shared`]): the application's work
/// queued ahead of the network's there is waited for too.
pub(crate) const SHARED_FENCE_WAIT: Duration = Duration::from_secs(60);

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
    /// `family`/`index` (and `post_index`, the same queue then) is the application's, shared: every submit to it
    /// holds this lock ([`crate::shared_queue`]).
    pub shared: Option<&'static crate::shared_queue::SharedQueue>,
    /// The application enables `VK_EXT_buffer_device_address` (Red Dead Redemption 2): the network uses its feature.
    pub bda_ext: bool,
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
            shared: None,
            bda_ext: false,
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
    /// Another device's loader asked for the network, and this one's paths are not using it ([`Loader::lend`]):
    /// the worker closes it, unless a path took it since ([`Self::uses`]). It is opened again when a path here
    /// asks for a build. While it is set, no path takes the network ([`Loader::ready_sizes`]).
    lend: bool,
    /// How many times a path took the network ([`Loader::ready_sizes`] answering with a build), and the count
    /// [`Loader::lend`] decided on: the worker closes the network only while they agree.
    uses: u64,
    lend_uses: u64,
    /// The network is opened without counter chaining ([`Loader::start`]'s `chain` false): the holds run inside
    /// DLSS's buffer, which needs barriers between its launches, so a chained build would be built again at once.
    barriers_from_start: bool,
    /// Steps the network down when it no longer fits in VRAM, and back up ([`super::vram_guard`]).
    guard: super::vram_guard::Guard,
    /// The network frames the guard was told of ([`Loader::note_gpu_ms`]), for [`fake_vram_spill`].
    guard_frames: u64,
    /// When a path last took the network ([`Loader::ready_sizes`] answering with a build): the loader thread applies
    /// a step down of the VRAM guard itself once none has for [`IDLE_STEP`] ([`LoaderState::apply_idle_step`]).
    last_use: Option<Instant>,
}

/// How long no path may have taken the network before the loader thread applies a step down of the VRAM guard itself:
/// longer than any wait of the layer's. A frame whose wait timed out may still run after it, which [`Claims::unused`]
/// tells.
const IDLE_STEP: std::time::Duration = std::time::Duration::from_secs(crate::FENCE_WAIT_TIMEOUT.as_secs() + 1);

impl LoaderState {
    /// A step down the VRAM guard decided is applied where the paths ask for the network ([`Self::want_sizes`]); while
    /// the spill stalls the game, no path may ask again: below 15 fps the layer stops working on its frames, and the
    /// network kept the video memory the game needed (Cyberpunk 2077 at 4K with Ray Reconstruction and path tracing
    /// stayed at 9 fps for minutes with the step decided). With no path having taken it for [`IDLE_STEP`] and nothing that
    /// used it in flight ([`Claims::unused`]), the loader thread applies it (and closes the network for a pause).
    fn apply_idle_step(&mut self, claims: &Claims, now: Instant) -> Option<super::vram_guard::Level> {
        let idle = self.last_use.is_none_or(|t| now.saturating_duration_since(t) >= IDLE_STEP);
        // Neither path uses the network (`Loader::unused`, as for a lend): besides the holds recorded into DLSS's buffers,
        // a split hold's work (its fence pending, or none covering it), an after-the-upscaler frame running, or one whose
        // wait timed out and whose fence has not signalled. A frame taken more than IDLE_STEP ago can still run then, and
        // closing the network under it was a copy-engine read of freed memory (Xid 31, #59).
        let unused = || claims.unused();
        (idle && self.guard.step_down_pending() && unused()).then(|| self.guard.apply()).flatten()
    }

    /// Whether the loader thread may close the network or free its builds now: nothing is open or built, or no frame that
    /// uses them may still run ([`Claims::in_flight`]).
    fn frees_now(&self, claims: &Claims) -> bool {
        (self.network.is_none() && self.built.is_empty()) || !claims.in_flight()
    }

    /// Whether the loader thread has work: the network is not open yet, the graph is to fall back to
    /// barriers, the network open is not the one the setting chooses, or a size is wanted that is not
    /// built.
    fn wants_work(&self) -> bool {
        !self.dead
            && (self.pausing()
                || self.target().is_some()
                    && (!self.opened || self.fall_back || self.switching() || self.wanted.as_ref().is_some_and(|w| *w != self.built_sizes())))
    }

    /// The network to run: the one the Network setting chose, the small one in its place, or none while the
    /// VRAM guard has it paused ([`super::vram_guard`]).
    fn target(&self) -> Option<u32> {
        self.guard.target(self.choice)
    }

    /// Paused by the VRAM guard with the network still open or built: the loader thread closes it.
    fn pausing(&self) -> bool {
        self.target().is_none() && (self.network.is_some() || !self.built.is_empty())
    }

    /// The sizes of what is built, in their order.
    fn built_sizes(&self) -> Vec<(u32, u32)> {
        self.built.iter().map(|b| (b.width, b.height)).collect()
    }

    /// The network was last opened for another Network setting than the holds' now: it is closed (when
    /// open) and the chosen one opened.
    fn switching(&self) -> bool {
        self.opened_for.is_some_and(|f| self.target().is_some_and(|t| t != f))
    }

    /// Whether the current build is what [`Self::want_sizes`] would serve for `sizes` and `choice` without
    /// applying a step of the VRAM guard.
    fn serves(&self, sizes: &[(u32, u32)], choice: u32) -> bool {
        self.choice == choice && self.target().is_some_and(|t| self.opened_for == Some(t)) && !self.built.is_empty() && self.built_sizes() == sizes
    }

    /// The build the holds may use for `size` and the Network setting `choice`, after noting what they
    /// want (the caller wakes the loader thread when this returns `true` in its second field).
    #[cfg(test)]
    fn want(&mut self, size: (u32, u32), choice: u32) -> (Option<Built>, bool) {
        let (built, wake) = self.want_sizes(&[size], choice, true);
        (built.map(|b| b[0]), wake)
    }

    /// [`Self::want`] for several sizes at once (one graph each, built together): the builds in `sizes`'
    /// order once exactly those are built. `may_step`: a step the VRAM guard decided is applied here (where
    /// a switch of the setting is, after the frames that used the network were drained), unless holds
    /// recorded with the current build may still run.
    fn want_sizes(&mut self, sizes: &[(u32, u32)], choice: u32, may_step: bool) -> (Option<Vec<Built>>, bool) {
        let mut wake = false;
        if self.choice != choice {
            self.choice = choice;
            // The user's choice wins: the guard starts afresh with it.
            self.guard.reset();
            wake = true;
        }
        if may_step && self.guard.apply().is_some() {
            wake = true;
        }
        let Some(target) = self.target() else {
            // Paused: nothing is served; the loader thread closes what is open.
            return (None, wake);
        };
        if self.opened_for == Some(target) && !self.built.is_empty() && self.built_sizes() == sizes {
            // A path takes the current build while the other path's request for another set of sizes waits for the
            // loader thread: that request is dropped, or the thread would free this build under the frame recorded
            // with it now (Cyberpunk 2077 resuming from its pause menu: the after-the-upscaler path asked for the
            // output size, a hold recorded into DLSS's buffer took the render-size build an instant later, the
            // thread rebuilt, and the GPU faulted reading the freed build: Xid 13, device lost; #59). The other
            // path asks again when it next runs, which `claim_post` keeps off while holds run.
            if self.wanted.as_deref().is_some_and(|w| w != sizes) {
                self.wanted = None;
            }
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
    /// The device's handle: its entry in [`LOADERS`].
    device: u64,
    /// Who may be using the network now, which both paths and the loader thread read ([`Claims`]).
    claims: Arc<Claims>,
    /// Tests: what records "the network" into a hold's buffer, in place of the graph (no network on lavapipe).
    #[cfg(test)]
    test_graph: Option<TestGraph>,
}

/// The device's counts of the holds recorded into DLSS's buffers that may still run, shared by its loader and its
/// `super::inline::Inline`: all of them ([`Claims::holds`]), and those that may record the network ([`Claims::local`]).
#[derive(Clone, Default)]
pub(crate) struct HoldCounts {
    pub all: Arc<std::sync::atomic::AtomicUsize>,
    pub local: Arc<std::sync::atomic::AtomicUsize>,
}

/// Who may be using the network's frame (activations, counters) and builds, which the two paths share and the loader
/// thread checks before it frees a build or closes the network ([`Claims::in_flight`], [`Claims::unused`]).
#[derive(Default)]
struct Claims {
    /// The last pre-upscaler hold and the last after-the-upscaler frame (ms since [`epoch_ms`]'s start), and
    /// whether the latter is running now ([`Loader::claim_post`], [`Loader::claim_pre`]).
    last_pre: std::sync::atomic::AtomicU64,
    last_post: std::sync::atomic::AtomicU64,
    post_busy: std::sync::atomic::AtomicBool,
    /// The fence of the pre-upscaler path's last submitted hold while it has not been waited on
    /// ([`Loader::pre_submitted`]): the after-the-upscaler path does not take the network (whose build
    /// for another size would free what that hold runs) until it has signalled.
    pre_inflight: Mutex<Option<InFlight>>,
    /// Until when (ms since [`epoch_ms`]'s start) a pre-upscaler hold's work that no fence covers may
    /// still run ([`Loader::pre_unfenced`]): the after-the-upscaler path keeps off the network till then.
    pre_unfenced_until: std::sync::atomic::AtomicU64,
    /// The after-the-upscaler frame that did not finish in time, still holding the claim
    /// ([`Loader::post_stalled`]): released once its fence signals.
    post_inflight: Mutex<Option<InFlight>>,
    /// Holds recorded into DLSS's command buffers that may still run (`super::inline`): they read the
    /// current build, so nothing rebuilds it ([`Loader::ready_sizes`], [`Loader::fall_back`]) and the
    /// after-the-upscaler path does not take the network ([`Claims::pre_running`]) while there are any.
    holds: Arc<std::sync::atomic::AtomicUsize>,
    /// Those of [`Self::holds`] that may record this network: every hold from before it asks for it until it can no
    /// longer run, except the holds before DLSS whose model runs on a second GPU (`super::offload_pre`), which
    /// `super::inline::Inline` lets go of once recorded. Those keep no build or close from happening
    /// ([`Self::holds_beyond`] counts them) but are no use of the network: a network the after-the-upscaler path
    /// opened here is lent to the second GPU's loader beside them ([`Claims::unused`]), which otherwise was refused it
    /// for good (#63: the model gone after a change of DLSS's render size, back on the game's GPU 30 s later).
    local: Arc<std::sync::atomic::AtomicUsize>,
    /// The last time a path took a build of this network ([`Loader::ready_sizes`]; ms since [`epoch_ms`]'s start). Holds
    /// whose model runs on a second GPU stamp [`Self::last_pre`] (it keeps the after-the-upscaler path off) but never
    /// this.
    last_local: std::sync::atomic::AtomicU64,
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

impl Claims {
    /// More than `reserved` holds recorded (or being recorded) into DLSS's buffers may still run ([`Loader::holds_beyond`]).
    fn holds_beyond(&self, reserved: usize) -> bool {
        self.holds.load(std::sync::atomic::Ordering::SeqCst) > reserved
    }

    /// The holds recorded (or being recorded) into DLSS's buffers that may still run and may have recorded this network
    /// ([`Self::local`]): one load, never below the true count (a hold counts from before it asks for the network).
    fn network_holds(&self) -> usize {
        self.local.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn release_post(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        self.last_post.store(epoch_ms(), SeqCst);
        self.post_busy.store(false, SeqCst);
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

    /// Whether a pre-upscaler hold's work may still use the network: a split hold's (its fence), or one
    /// recorded into DLSS's buffers.
    fn pre_running(&self) -> bool {
        if self.network_holds() > 0 {
            return true;
        }
        let mut inflight = self.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if inflight.is_some_and(|f| !f.running()) {
            *inflight = None;
        }
        inflight.is_some()
    }

    /// Whether work that uses the network may still run on the GPU, or a frame is using it now: a hold recorded into
    /// DLSS's buffers, a split hold whose fence has not signalled or whose work no fence covers (until
    /// [`crate::FENCE_WAIT_TIMEOUT`] after its submit), an after-the-upscaler frame running, or one that did not finish
    /// in time and whose fence has not signalled since ([`Loader::post_stalled`]). Nothing frees a build or closes the
    /// network while it is so: the GPU would read freed memory (Xid 31, #59).
    fn in_flight(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        self.post_finished();
        let stalled = self.post_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_some();
        self.pre_running() || self.post_busy.load(SeqCst) || stalled || epoch_ms() < self.pre_unfenced_until.load(SeqCst)
    }

    /// Neither path uses the network: nothing [`Self::in_flight`], and neither path took it for [`PATH_SWITCH_MS`]
    /// ([`Self::last_local`], [`Self::last_post`]). Holds whose model runs on a second GPU do not count: they record
    /// nothing of it.
    fn unused(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        let quiet = |at: u64| epoch_ms().saturating_sub(at) >= PATH_SWITCH_MS;
        !self.in_flight() && quiet(self.last_local.load(SeqCst)) && quiet(self.last_post.load(SeqCst))
    }
}

/// [`Loader::holds_took_over`] for the last hold's and the last after-the-upscaler frame's stamps (ms since [`epoch_ms`]'s
/// start; 0 never) and whether that path is running a frame now.
fn took_over(last_pre: u64, last_post: u64, post_busy: bool) -> bool {
    !post_busy && last_post != 0 && last_pre.saturating_sub(last_post) >= PATH_SWITCH_MS
}

/// The holds' clock for the history's gap rule (`HistoryGap::MAX_GAP`): the time now, or in a test the
/// rig's stopped clock ([`stop_clock`]), so that how long the test machine takes between two holds does not
/// decide whether the second one has the first one's history.
pub(crate) fn hold_now() -> Instant {
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

/// `NEURAL_FORGE_EXPOSURE_JUMP=off|<stops>`: how far the exposure the encode uses may change between two network
/// frames, in stops, before the second starts without history (the frame shaders' `exposureJumped`).
pub(crate) const EXPOSURE_JUMP_ENV: &str = "NEURAL_FORGE_EXPOSURE_JUMP";

/// [`EXPOSURE_JUMP_ENV`]'s default: one stop, a factor of 2. The game's own exposure jumps by far more at a
/// pause, a menu or a map (GTA V: 0.15-0.34 in play); the auto exposure adapts 5% of the way per hold, under a
/// stop between two network frames unless its target is about 10 stops away.
pub(crate) const EXPOSURE_JUMP_STOPS: f32 = 1.0;

/// [`EXPOSURE_JUMP_ENV`]'s value in stops, 0 for off: unset, empty or `on` the default; `off` or `0` off; else a
/// number of stops above 0 and up to 16.
fn parse_exposure_jump(raw: Option<&str>) -> Result<f32, String> {
    match raw.map(str::trim) {
        None | Some("" | "on") => Ok(EXPOSURE_JUMP_STOPS),
        Some("off" | "0") => Ok(0.0),
        Some(v) => v.parse::<f32>().ok().filter(|s| s.is_finite() && *s > 0.0 && *s <= 16.0).ok_or_else(|| v.to_string()),
    }
}

/// The frame shaders' exposure jump factor (`Params::more[3]`): 2 to the power of [`EXPOSURE_JUMP_ENV`]'s stops, 0
/// when off. Read once, logged when not the default.
pub(crate) fn exposure_jump_factor() -> f32 {
    static FACTOR: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        let stops = match parse_exposure_jump(neural_forge_protocol::env::var(EXPOSURE_JUMP_ENV).as_deref()) {
            Ok(s) => s,
            Err(other) => {
                crate::log!(
                    "[native] {EXPOSURE_JUMP_ENV}={other:?} is not off or a number of stops above 0 and up to 16; the default {EXPOSURE_JUMP_STOPS} stays"
                );
                EXPOSURE_JUMP_STOPS
            }
        };
        if stops == 0.0 {
            crate::log!("[native] {EXPOSURE_JUMP_ENV}=off: the history is kept when the exposure jumps");
        } else if stops != EXPOSURE_JUMP_STOPS {
            crate::log!("[native] the history resets when the exposure changes by more than {stops} stops ({EXPOSURE_JUMP_ENV})");
        }
        if stops > 0.0 {
            stops.exp2()
        } else {
            0.0
        }
    });
    *FACTOR
}

/// The composite's exposure-jump word (`Exposure::jump`: the previous network frame's exposure in the low half,
/// the jump frame's in the high, as halves), read back after its frame ran: the two exposures when it reports a
/// jump not seen before. `last` is the last word read: holds the network did not run on (the edit cache's) may
/// read the same word again where nothing clears it (inside DLSS's buffer).
fn take_exposure_jump(last: &mut u32, word: u32) -> Option<(f32, f32)> {
    let new = word != *last;
    *last = word;
    (new && word != 0).then(|| (super::f16_to_f32(word as u16), super::f16_to_f32((word >> 16) as u16)))
}

/// The CPU's side of an exposure jump the frame shaders reset on (`exposureJumped`), at a hold after the one it
/// was on: logged once per jump, and the seed restarts as after a scene cut the GPU reset ([`NativePass::restart_seed`]).
/// The edit cache needs nothing: the jump frame refreshed it at the new exposure.
pub(crate) fn apply_exposure_jump(word: u32, native: Option<&mut NativePass>) {
    let Some(n) = native else { return };
    if let Some((before, now)) = take_exposure_jump(&mut n.last_jump, word) {
        crate::log!("[native] resetting the history (the exposure changed from {before} to {now}, by {:.1} stops)", (now / before).log2().abs());
        n.restart_seed();
    }
}

/// `NEURAL_FORGE_NATIVE_NO_HISTORY=1` (diagnostics): every network frame runs without history, as after a reset,
/// to tell what the history adds to the picture from what one frame does.
fn no_history() -> bool {
    static OFF: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let off = neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_NO_HISTORY").as_deref() == Some("1");
        if off {
            crate::log!("[native] NEURAL_FORGE_NATIVE_NO_HISTORY=1: every frame without history");
        }
        off
    });
    *OFF
}

/// The device whose loader opened the network last ([`Loader::work`]). One network can be open in a process (its
/// Vulkan function table is process-wide), and a game may create several devices: Metro Last Light Redux's
/// after-the-upscaler path opened it on the device that presents the intro, and DLSS then ran on another, whose
/// holds were refused it for good (2026-10-23).
static NETWORK_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The longest a build waits for the game to stop loading ([`Loader::work`]).
const LOADING_WAIT_MAX: std::time::Duration = std::time::Duration::from_secs(60);

/// How often the loader thread looks again whether a frame still uses the network it is to close or rebuild
/// ([`LoaderState::frees_now`]).
const IN_FLIGHT_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// How often the loader thread checks the VRAM budget for the guard ([`super::vram_guard`]) while it has no other work.
const GUARD_TICK: std::time::Duration = std::time::Duration::from_secs(1);

/// The words of `nf_native_open`'s refusal while another network is open in the process (crates/native/cpp/nf_native.cpp).
const OPEN_ELSEWHERE: &str = "another network is open in this process";

/// This process's loaders by device ([`Loader::register`]), for [`Loader::lend`].
static LOADERS: Mutex<Vec<(u64, std::sync::Weak<Loader>)>> = Mutex::new(Vec::new());

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
    /// The loader of `device`'s network, opened with counter chaining when `chain` (and `NEURAL_FORGE_NATIVE_CHAIN`
    /// allows it): `false` where the holds run inside DLSS's buffer, which needs barriers.
    pub(crate) fn start(device: vk::Device, setup: Setup, holds: HoldCounts, chain: bool) -> Self {
        let shared: Arc<(Mutex<LoaderState>, Condvar)> = Arc::default();
        shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).barriers_from_start = !chain;
        Self {
            shared,
            worker: Mutex::new(None),
            pending: Mutex::new(Some((device, setup))),
            setup,
            device: device.as_raw(),
            claims: Arc::new(Claims { holds: holds.all, local: holds.local, ..Default::default() }),
            #[cfg(test)]
            test_graph: None,
        }
    }

    /// Tests: a loader whose network is `built` (open, no worker), with `graph` recording into a hold's
    /// buffer in the graph's place.
    #[cfg(test)]
    pub(crate) fn for_test(setup: Setup, built: Vec<Built>, holds: HoldCounts, graph: TestGraph) -> Self {
        let mut loader = Self::start(vk::Device::null(), setup, holds, true);
        *loader.pending.get_mut().unwrap() = None;
        loader.test_graph = Some(graph);
        {
            let mut st = loader.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            st.opened = true;
            st.opened_for = Some(network::FULL);
            st.choice = network::FULL;
            st.running = network::FULL;
            st.generation = built.first().map_or(0, |b| b.generation);
            st.built = built;
        }
        loader
    }

    /// More than `reserved` holds recorded (or being recorded) into DLSS's buffers may still run: a hold being
    /// recorded counts itself from before it asks for the network (`inline::Inline::record_outcome`), and passes
    /// 1 here, so only the others keep a rebuild off.
    fn holds_beyond(&self, reserved: usize) -> bool {
        self.claims.holds_beyond(reserved)
    }

    /// The network is to be rebuilt, and waits for the holds recorded into DLSS's buffers with the current
    /// build (besides the caller's `reserved`) to be done with ([`Self::ready_sizes`]).
    pub(crate) fn waiting_for_holds(&self, reserved: usize) -> bool {
        self.holds_beyond(reserved)
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
        if epoch_ms().saturating_sub(self.claims.last_pre.load(SeqCst)) < PATH_SWITCH_MS
            || self.claims.pre_running()
            || epoch_ms() < self.claims.pre_unfenced_until.load(SeqCst)
        {
            return false;
        }
        self.claims.post_busy.store(true, SeqCst);
        if epoch_ms().saturating_sub(self.claims.last_pre.load(SeqCst)) < PATH_SWITCH_MS {
            self.claims.post_busy.store(false, SeqCst);
            return false;
        }
        true
    }

    pub(crate) fn release_post(&self) {
        self.claims.release_post();
    }

    /// Tests: a loader on no device (its worker idles: nothing is asked of it).
    #[cfg(test)]
    pub(crate) fn idle_for_test() -> Self {
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
            shared: None,
            bda_ext: false,
        };
        Self::start(vk::Device::null(), setup, HoldCounts::default(), true)
    }

    /// Tests: the claims' stamps (the last hold's, the after-the-upscaler path's last frame's) and whether that path runs one.
    #[cfg(test)]
    pub(crate) fn stamp_for_test(&self, last_pre: u64, last_post: u64, post_busy: bool) {
        use std::sync::atomic::Ordering::SeqCst;
        self.claims.last_pre.store(last_pre, SeqCst);
        self.claims.last_post.store(last_post, SeqCst);
        self.claims.post_busy.store(post_busy, SeqCst);
    }

    /// The holds took the network over from the after-the-upscaler path: a hold asked for it [`PATH_SWITCH_MS`] or more
    /// after that path's last frame, which is not running. That path's buffers at its own size are not needed then
    /// (`native_post`), and kept they sat in VRAM beside the holds' (#59).
    pub(crate) fn holds_took_over(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        took_over(self.claims.last_pre.load(SeqCst), self.claims.last_post.load(SeqCst), self.claims.post_busy.load(SeqCst))
    }

    /// The after-the-upscaler frame holding the claim did not finish in time: its work (on `fence`)
    /// may still use the network's buffers, so the claim is kept until that fence signals (checked by
    /// [`Self::claim_pre`]). The fence stays alive: a stalled server destroys nothing.
    pub(crate) fn post_stalled(&self, device: &ash::Device, fence: vk::Fence) {
        *self.claims.post_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(InFlight::of(device, fence));
    }

    /// A pre-upscaler hold's work went to the queue and no fence could be put after it: the network is
    /// taken as in use for as long as any wait of the layer's is bounded ([`crate::FENCE_WAIT_TIMEOUT`]).
    pub(crate) fn pre_unfenced(&self) {
        let until = epoch_ms() + crate::FENCE_WAIT_TIMEOUT.as_millis() as u64;
        self.claims.pre_unfenced_until.fetch_max(until, std::sync::atomic::Ordering::SeqCst);
    }

    /// The pre-upscaler path submitted a hold whose work `fence` covers.
    pub(crate) fn pre_submitted(&self, device: &ash::Device, fence: vk::Fence) {
        *self.claims.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(InFlight::of(device, fence));
    }

    /// The pre-upscaler path's work on `fence` has completed (the fence was waited on, or nothing on it is
    /// pending), and the fence may be reset or destroyed from here on: forgotten if it is the one noted. Only
    /// that one: the hold's other resources (`Session::res_inline`, `res_after`) note fences of their own.
    pub(crate) fn pre_idle(&self, fence: vk::Fence) {
        let mut inflight = self.claims.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if inflight.is_some_and(|f| f.fence == fence) {
            *inflight = None;
        }
    }

    /// The fence noted by [`Self::pre_submitted`] and not forgotten yet.
    #[cfg(test)]
    pub(crate) fn pre_fence(&self) -> Option<vk::Fence> {
        self.claims.pre_inflight.lock().unwrap_or_else(std::sync::PoisonError::into_inner).map(|f| f.fence)
    }

    /// The pre-upscaler path may submit a frame now (the after-the-upscaler path is not running one and
    /// has not for a second). Notes the hold either way, which keeps the other path off.
    pub(crate) fn claim_pre(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        self.claims.last_pre.store(epoch_ms(), SeqCst);
        self.claims.post_finished();
        // A path that never ran (0) has nothing to switch away from.
        let last_post = self.claims.last_post.load(SeqCst);
        !self.claims.post_busy.load(SeqCst) && (last_post == 0 || epoch_ms().saturating_sub(last_post) >= PATH_SWITCH_MS)
    }

    /// Notes this loader as `device`'s, so a loader of another device of the process can ask it to lend the
    /// network ([`Self::lend`]).
    pub(crate) fn register(self: &Arc<Self>, device: vk::Device) {
        let mut loaders = LOADERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loaders.retain(|(d, l)| l.strong_count() > 0 && *d != device.as_raw());
        loaders.push((device.as_raw(), Arc::downgrade(self)));
    }

    /// Another device's loader could not open the network because this one has it: when neither path here uses
    /// it (no hold recorded into DLSS's buffers may run, no hold's or after-the-upscaler frame's work is in
    /// flight, neither path ran for [`PATH_SWITCH_MS`]), the worker closes it, the after-the-upscaler path kept
    /// off it meanwhile. `false` when it is in use: the other loader tries again on its schedule.
    ///
    /// Ordered against both paths: the uses are counted first (a path that takes the network after that is seen
    /// by the count, [`LoaderState::uses`]); the hold's stamp goes in before the after-the-upscaler path's busy
    /// flag is read (that path sets its flag before it reads the stamp, [`Self::claim_post`], so one of the two
    /// sees the other); and once lent, neither path takes the network until the worker closed it.
    fn lend(&self) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        let (lock, wake) = &*self.shared;
        let uses = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner).uses;
        // The after-the-upscaler path checks the last hold before it takes the network: it waits now.
        let stamp = epoch_ms();
        let before = self.claims.last_pre.swap(stamp, SeqCst);
        let refuse = || {
            // Not lent: the stamp goes back (unless a hold stamped since), so the other path is not kept off.
            let _ = self.claims.last_pre.compare_exchange(stamp, before, SeqCst, SeqCst);
            false
        };
        if !self.unused() {
            return refuse();
        }
        let mut st = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if st.network.is_none() || self.claims.network_holds() > 0 || st.uses != uses {
            drop(st);
            return refuse();
        }
        st.lend = true;
        st.lend_uses = uses;
        drop(st);
        wake.notify_all();
        true
    }

    /// Neither path uses the network ([`Self::lend`]'s condition).
    fn unused(&self) -> bool {
        self.claims.unused()
    }

    /// The device is being destroyed: no other device's loader reaches this one from here on ([`Self::ask_owner`]
    /// works under the registry's lock, which this waits for), so its last reference is dropped by the device's own
    /// teardown, never on another device's loading thread after the device is gone.
    pub(crate) fn retire(&self) {
        let mut loaders = LOADERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loaders.retain(|(d, l)| l.strong_count() > 0 && *d != self.device);
    }

    /// Asks the loader of the device that has the network, if it is another than `device`'s, to lend it
    /// ([`Self::lend`]). Whether it will.
    fn ask_owner(device: vk::Device) -> Option<bool> {
        let owner = NETWORK_OWNER.load(std::sync::atomic::Ordering::SeqCst);
        if owner == 0 || owner == device.as_raw() {
            return None;
        }
        // Under the registry's lock throughout: the owner's teardown retires it under that lock before it drops its
        // own references ([`Self::retire`]), so the reference taken here is never the last.
        let loaders = LOADERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let loader = loaders.iter().find(|(d, _)| *d == owner).and_then(|(_, l)| l.upgrade())?;
        let lent = loader.lend();
        drop(loader);
        drop(loaders);
        Some(lent)
    }

    /// Starts the loading thread the first time.
    fn spawn(&self) {
        let Some((device, setup)) = self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() else { return };
        let shared = self.shared.clone();
        let claims = self.claims.clone();
        let worker = std::thread::Builder::new().name("nf-native-load".into()).spawn(move || Self::work(&shared, device, setup, &claims)).ok();
        if worker.is_none() {
            crate::log!("[native] could not start the loader thread; frames go to DLSS untouched");
            note_failure("the loader thread could not be started");
            self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).failed = Some("the loader thread could not be started".into());
        }
        *self.worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = worker;
    }

    /// Opens the network the Network setting `choice` names ([`network_to_load`]); returns it with
    /// which network it is.
    /// `said_substitute`: the choice whose stand-in network was said last, so a refused open asked again every second
    /// does not say it each time.
    fn open(device: vk::Device, setup: Setup, choice: u32, chain: bool, said_substitute: &mut Option<u32>) -> Result<(nn::Network, u32), nn::OpenError> {
        let Some(running) = network_to_load(choice, network_present) else {
            let message = format!("no model at {}: extract it in the Setup tab (neural-forge-cli extract-model)", network_dir(choice).display());
            return Err(nn::OpenError { message, stalled: false });
        };
        if running != choice && said_substitute.replace(choice) != Some(choice) {
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
            chain: chain_wanted() && chain,
            // On a shared queue the network's work runs behind the application's own there, which may wait on the
            // application (a pause, a loading screen): a longer bound before the network is given up for good.
            fence_timeout_ms: if setup.shared.is_some() { SHARED_FENCE_WAIT.as_millis() as u32 } else { crate::FENCE_WAIT_TIMEOUT.as_millis() as u32 },
            init_dispatchable: Some(init_dispatchable),
            // The application's queue, shared: every submit and queue wait on that side under its lock.
            queue_lock: setup.shared.map(|_| crate::shared_queue::c_lock as nn::QueueLock),
            queue_user: setup.shared.map_or(std::ptr::null_mut(), crate::shared_queue::SharedQueue::as_user),
            buffer_device_address_ext: setup.bda_ext,
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
            if open.chain { "on" } else { "off" }
        );
        crate::log!("[native] model source: {}", model_source(&std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_default()));
        Ok((network, running))
    }

    fn work(shared: &(Mutex<LoaderState>, Condvar), device: vk::Device, setup: Setup, claims: &Claims) {
        let (lock, wake) = shared;
        // This device's VRAM readings for the build lines (it outlives this thread: `Drop` joins it).
        let vram = crate::vram::probe(setup.gipa, setup.instance, setup.physical, setup.properties2);
        let mut retry = BuildRetry::default();
        // The Network setting whose stand-in network was said last ([`Self::open`]).
        let mut said_substitute: Option<u32> = None;
        let mut inject = neural_forge_protocol::env::var(FailInject::ENV).and_then(|v| FailInject::parse(&v));
        if let Some(f) = inject {
            crate::log!("[native] {}: the next {} network builds will fail on purpose", FailInject::ENV, f.remaining());
        }
        let (mut loading_since, mut said_loading, mut said_waited): (Option<Instant>, bool, bool) = (None, false, false);
        // Since when a close or rebuild has waited for frames still using the network ([`LoaderState::frees_now`]), and
        // whether that was said.
        let (mut in_flight_since, mut said_in_flight): (Option<Instant>, bool) = (None, false);
        let mut last_tick = Instant::now();
        let mut state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            // Lent to another device's loader ([`Loader::lend`]): closed, its builds gone, and nothing asked
            // for until a path here asks again. SAFETY of the close: `lend` asked only with nothing of it in use.
            if std::mem::take(&mut state.lend) {
                // A path took it after the lend was decided, or a frame that used it may still run: kept.
                if state.uses != state.lend_uses || claims.in_flight() {
                    // The log line with the lock let go, as everywhere here: holds take it on the game's thread.
                    drop(state);
                    crate::log!("[native] the network stays on this device: a path here took it as it was being lent");
                    crate::logging::flush();
                    state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    continue;
                }
                if let Some(network) = state.network.take() {
                    state.built = Vec::new();
                    state.wanted = None;
                    state.failed = Some("lent to another device of the game, which runs the model".into());
                    drop(state);
                    drop(network);
                    crate::log!("[native] the network is closed on this device: another device of the game asked for it (one network per process)");
                    crate::logging::flush();
                    state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                continue;
            }
            // Work is wanted when the network is not open yet, another network is chosen, a size is
            // wanted that is not built, or the graph is to fall back to barriers
            // ([`LoaderState::wants_work`]); it is due when the retry schedule says so.
            if state.quit {
                return;
            }
            if !state.wants_work() {
                state = wake.wait_timeout(state, GUARD_TICK).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                if state.quit || last_tick.elapsed() < GUARD_TICK {
                    continue;
                }
                last_tick = Instant::now();
                // The VRAM guard's check: the reading taken with the lock let go (holds take it on the game's thread).
                drop(state);
                let reading = vram.as_ref().map(crate::vram::reading);
                let small = network_present(network::SMALL);
                state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let running = (!state.built.is_empty()).then_some(state.running);
                let choice = state.choice;
                let size = state.wanted.as_ref().and_then(|w| w.first().copied()).or_else(|| state.built.first().map(|b| (b.width, b.height)));
                // `NEURAL_FORGE_PROBE_VRAM=1`: what the guard saw at each tick (the reading, the network's times).
                if *PROBE_VRAM {
                    let probe = format!("[native] VRAM guard probe: reading {reading:?}, running {running:?}, {}", state.guard.probe(Instant::now()));
                    crate::log!("{probe}");
                }
                if let Some(line) = state.guard.tick(reading, running, size, choice, small, Instant::now()) {
                    drop(state);
                    crate::log!("{line}");
                    crate::logging::flush();
                    state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                if let Some(level) = state.apply_idle_step(claims, Instant::now()) {
                    drop(state);
                    crate::log!("[native] VRAM guard: no frame has used the network for {} s; the step down ({level:?}) is applied now", IDLE_STEP.as_secs());
                    crate::logging::flush();
                    state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                continue;
            }
            // Nothing below closes the network or frees a build while a frame that used it may still run on the GPU
            // (the copy engine read freed memory: Xid 31, #59): checked again here, under the lock, whatever asked for
            // the work. A frame that never finishes keeps the network as it is.
            if !state.frees_now(claims) {
                let since = *in_flight_since.get_or_insert_with(Instant::now);
                if !said_in_flight && since.elapsed() >= crate::FENCE_WAIT_TIMEOUT {
                    said_in_flight = true;
                    crate::log!("[native] the network's rebuild or close waits for a frame that still uses it on the GPU");
                    crate::logging::flush();
                }
                state = wake.wait_timeout(state, IN_FLIGHT_POLL).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                continue;
            }
            in_flight_since = None;
            // Paused by the VRAM guard: the network is closed, its memory given back to the game. SAFETY of the
            // close: the pause is applied only where a switch of the setting is ([`LoaderState::want_sizes`]), or here
            // once nothing may still use the build and no path has taken it for longer than any wait of the layer's
            // ([`LoaderState::apply_idle_step`]); and nothing is in flight now ([`LoaderState::frees_now`]).
            if state.pausing() {
                let n = state.network.take();
                state.built = Vec::new();
                state.opened_for = None;
                let why = state.guard.why().unwrap_or("paused: too little VRAM for the network").to_string();
                note_failure(&why);
                state.failed = Some(why);
                drop(state);
                drop(n);
                crate::log!("[native] VRAM guard: the network is closed; frames go to DLSS untouched until there is room again");
                crate::logging::flush();
                state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            // While the game loads, no build: each runs the network once on this queue, which a game's loading work
            // could not make room for (`crate::swapchain::loading_now`). At most a minute, for a game that renders
            // below 15 fps all along.
            if crate::swapchain::loading_now() {
                let since = *loading_since.get_or_insert_with(Instant::now);
                if since.elapsed() < LOADING_WAIT_MAX {
                    if !said_loading {
                        said_loading = true;
                        crate::log!("[native] the game is loading (frames under 15 fps or none): the network is built once it renders steadily");
                        crate::logging::flush();
                    }
                    state = wake.wait_timeout(state, std::time::Duration::from_millis(250)).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                    continue;
                }
                if !said_waited {
                    said_waited = true;
                    crate::log!("[native] the game has not rendered steadily for a minute: building the network anyway");
                    crate::logging::flush();
                }
            } else {
                loading_since = None;
            }
            let step = retry.step(Instant::now());
            if let Step::Wait(d) = step {
                state = wake.wait_timeout(state, d).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                continue;
            }
            let switching = state.switching();
            let mut network = state.network.take();
            let fall_back = std::mem::take(&mut state.fall_back) || (switching && state.barriers);
            let sizes = state.wanted.clone().or_else(|| (!state.built.is_empty()).then(|| state.built_sizes()));
            // `wants_work` is false while paused, so there is a network to run.
            let choice = state.target().unwrap_or(state.choice);
            let barriers_from_start = state.barriers_from_start;
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
                None => Self::open(device, setup, choice, !barriers_from_start, &mut said_substitute)
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
            // This loader holds an open network after the attempt (built with it, or kept for the next one after its
            // build failed): it is the owner another device's loader asks to lend it. Not only after this attempt
            // opened it: one that opened it and failed its first build keeps it open (the retry reuses it), and so
            // does a borrower whose first build after a lend failed; another device's open is refused until this one
            // gives it up.
            if result.is_ok() || network.is_some() {
                NETWORK_OWNER.store(device.as_raw(), std::sync::atomic::Ordering::SeqCst);
            }
            // Refused because another device's loader has the network: that one is asked to lend it.
            let refused = matches!(&result, Err(why) if why.contains(OPEN_ELSEWHERE));
            let asked = (refused && opened.is_none() && !open_stalled && network.is_none()).then(|| Loader::ask_owner(device)).flatten();
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
            state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            state.opened = true;
            if let Some(running) = loaded {
                // The network now open (kept below whether its build succeeds) is the one for `choice` (the
                // network to run: the setting's, or the VRAM guard's in its place).
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
                    // Opened or built: a later refusal by another device is a new run, said again.
                    retry.owner_done();
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
                        // What a fresh open and build took, and whether it already ended at the budget.
                        let fresh_need = match (opened, vram_before, vram_after) {
                            (Some(_), Some((before, _)), Some((after, _))) => Some(after.saturating_sub(before)),
                            _ => None,
                        };
                        let (running, user_choice) = (state.running, state.choice);
                        let size = (built[0].0, built[0].1);
                        if let Some(line) =
                            state.guard.built(running, fresh_need, size, vram_after, user_choice, network_present(network::SMALL), Instant::now())
                        {
                            lines.push(line);
                        }
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
                    // Another device has the network and was asked to lend it: asked again on the owner's cadence, not a
                    // failure of this loader's (its streak would reach the 30 s wait), and said once per run of the same
                    // answer ([`BuildRetry::wait_for_owner`]).
                    let (wait, say) = match asked.filter(|_| refused) {
                        Some(lending) => retry.wait_for_owner(Instant::now(), lending),
                        None => (retry.failed(Instant::now()), true),
                    };
                    if say {
                        match asked {
                            Some(true) => {
                                lines.push("[native] the network is open on another device of the game, which is not using it: it is asked to close it".into())
                            }
                            Some(false) => lines.push("[native] the network is open on another device of the game, which is using it".into()),
                            None => {}
                        }
                        lines.push(format!("[native] {why}; frames go to DLSS untouched, trying again in {} ms", wait.as_millis()));
                    }
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
            if !lines.is_empty() {
                crate::logging::flush();
            }
            state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The network the Network setting `choice` names (`neural_forge_protocol::enums::network`), built
    /// for `width` x `height`, or `None` (and the build, or the other network, is asked for) while it
    /// is not. Called only once the caller's frames that used the network have completed: another
    /// size or network asked for here frees the current build.
    pub(crate) fn ready(&self, width: u32, height: u32, choice: u32) -> Option<Built> {
        self.ready_sizes(&[(width, height)], choice, 0).map(|b| b[0])
    }

    /// [`Self::ready`] for several sizes (a model resolution per pass): one graph each, built together, in
    /// `sizes`' order; `None` until exactly those are built. Asking for another set frees every graph of the
    /// current one (the same rule as [`Self::ready`]'s).
    ///
    /// While holds recorded into DLSS's buffers with the current build may still run, another set of sizes or
    /// another network is not asked for (the build they read stays): `None` until they are done with
    /// ([`Self::waiting_for_holds`]). `reserved`: the caller's own hold being recorded into DLSS's buffer (1),
    /// which already counts among them.
    pub(crate) fn ready_sizes(&self, sizes: &[(u32, u32)], choice: u32, reserved: usize) -> Option<Vec<Built>> {
        self.spawn();
        let (lock, wake) = &*self.shared;
        let mut st = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if st.lend {
            // Being closed for another device ([`Self::lend`]): neither path takes it.
            return None;
        }
        let holds_out = self.holds_beyond(reserved);
        // A step down the VRAM guard decided waits for the holds recorded with the build: no new hold takes it meanwhile,
        // or holds that come on and on keep it from ever being applied (Cyberpunk 2077 at 4K with Ray Reconstruction and
        // path tracing: a hold after DLSS every 0.4 s at 7 fps for minutes, the step decided and never applied). Once the
        // outstanding holds are done with, the next path to ask applies it, or the loader thread does
        // ([`LoaderState::apply_idle_step`]).
        if holds_out && (!st.serves(sizes, choice) || st.guard.step_down_pending()) {
            return None;
        }
        let (built, woken) = st.want_sizes(sizes, choice, !holds_out);
        if built.is_some() {
            st.uses += 1;
            st.last_use = Some(Instant::now());
            self.claims.last_local.store(epoch_ms(), std::sync::atomic::Ordering::SeqCst);
        }
        drop(st);
        if woken {
            wake.notify_all();
        }
        built
    }

    /// One hold's network GPU time, for the VRAM guard.
    pub(crate) fn note_gpu_ms(&self, ms: f32) {
        let now = Instant::now();
        let mut st = self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        st.guard.gpu_ms(ms, now);
        st.guard_frames += 1;
        let line = if fake_vram_spill(st.guard_frames) {
            let choice = st.choice;
            st.guard.fake_spill(choice, network_present(network::SMALL), now)
        } else {
            None
        };
        drop(st);
        if let Some(line) = line {
            crate::log!("{line}");
            crate::logging::flush();
        }
    }

    /// Whether the VRAM guard has the network paused (no network to run).
    pub(crate) fn paused(&self) -> bool {
        self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).target().is_none()
    }

    /// Why the VRAM guard has the network stepped down, while it has.
    pub(crate) fn guard_note(&self) -> Option<String> {
        self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).guard.why().map(str::to_string)
    }

    /// Why the network is not usable right now, if a load or build failed.
    /// A wait on the network's own queue timed out: its work may still run, so nothing of it may be freed.
    pub(crate) fn dead(&self) -> bool {
        self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).dead
    }

    pub(crate) fn failed(&self) -> Option<String> {
        self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).failed.clone()
    }

    /// Counter waits that gave up since the last reset (read once the frames that ran the graph
    /// completed), and where; `None` when there were none (no allocation on that, every hold's, path).
    pub(crate) fn chain_timeouts(&self) -> Option<(u32, String)> {
        let state = self.shared.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let network = state.network.as_ref()?;
        (network.chain_timeout_count() > 0).then(|| network.chain_timeouts())
    }

    /// Rebuilds the graph with barriers. The caller has drained every frame of its own that used the current
    /// one; `false` (nothing asked) while holds recorded into DLSS's buffers with it (besides the caller's
    /// `reserved`, [`Self::ready_sizes`]) may still run.
    pub(crate) fn fall_back(&self, reserved: usize) -> bool {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.holds_beyond(reserved) {
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
    /// The Oklab protections of the pass's result (`native_colour.comp`'s `protect`: hue, near-grey chroma,
    /// shadow, highlight, 0-1). Appended: the other frame shaders' blocks end before it.
    protect: [f32; 4],
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

/// The passes a network frame runs: all of `passes` when the later passes' buffers are there (`more`), else pass
/// 1 alone. Decided once per hold, so nothing plans, notes as run ([`MorePasses::last_run`]) or records a later
/// pass that does not run: one noted as run would take the history it never wrote for the next frame's.
pub(crate) fn runnable(passes: Passes, more: bool) -> Passes {
    Passes { count: if more { passes.count } else { passes.count.min(1) }, ..passes }
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
    /// The composite's last exposure-jump word read back ([`apply_exposure_jump`]).
    last_jump: u32,
    pub(crate) gap: HistoryGap<(u32, u32)>,
    /// The reset reason the last network frame planned was given ([`HistoryGap::begin`]), for the tests.
    #[cfg(test)]
    last_stale: Option<Stale>,
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
    /// The look-up table on the GPU (binding 17), the last pass's.
    lut: Option<LutBuffer>,
    /// Whether this frame applies [`Self::lut`] ([`Self::prepare_colour`]): not once `colour::current` gives no
    /// table (the strength at 0, none named, or one that could not be read), though the buffer stays.
    lut_applies: bool,
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
                last_jump: 0,
                more: None,
                mask: None,
                mask_depth: vk::Buffer::null(),
                colour: None,
                colour_wanted: false,
                colour_recorded: false,
                lut: None,
                lut_applies: false,
                gap: HistoryGap::default(),
                #[cfg(test)]
                last_stale: None,
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

    /// The noise seed of the last network frame planned (0: it ran without history).
    #[cfg(test)]
    pub(crate) fn seed(&self) -> u32 {
        self.seed
    }

    /// The reset reason the last network frame planned was given (`None`: it kept the history, or it was the first).
    #[cfg(test)]
    pub(crate) fn last_stale(&self) -> Option<Stale> {
        self.last_stale
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
        self.lut_applies = loaded;
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
                // The sets point binding 16 at the low-pass: written and recorded again.
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
            // The texel the encode used, then the composite's exposure-jump word.
            buf(exposure, 8),
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
            // The answer to `W`'s copy, the history to the next frame's samplers, the composite's exposure-jump
            // word to the host.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ)],
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

    /// A network frame on another device (`super::offload_server`) begins: the history's rule as the hold applies
    /// it. `asked`: the hold had motion vectors and nothing on its side resets the history. Returns whether the
    /// frame runs with its history.
    pub(crate) fn begin_frame(&mut self, network: u32, asked: bool) -> bool {
        if note_network(&mut self.gap, &mut self.history_network, network) {
            crate::log!("[native] resetting the history (the {} network runs from here)", network::name(network));
        }
        let stale = self.gap.begin(hold_now());
        self.frames > 0 && stale.is_none() && asked && !no_history()
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
        let lut = self.lut.as_ref().filter(|_| self.lut_applies).map(|l| &*l.lut);
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
                more: [if c.auto_mask { 1.0 } else { 0.0 }, mvec_scale[0], mvec_scale[1], exposure_jump_factor()],
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
                protect: colour(index).protect,
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
/// cache's stored edit). It does the same, and leaves the next hold to start afresh, where the host found an
/// input layout the hold guessed wrong when its buffer was submitted (`inline::Verdicts`). Its one set points at
/// the native pass's state buffer, the edit cache's counters (or a stand-in) and the verdicts (or a stand-in).
pub(crate) struct HoldGuard {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Stands in for the counters while the edit cache is off, and for the verdicts where there are none (the
    /// guard then reads none: [`NO_VERDICT`]).
    no_stats: OwnBuffer,
    /// What the set points at (state, counters, verdicts), with the native pass whose state buffer it is
    /// ([`NativePass::id`]): a rebuilt pass's buffer may come back under the old one's handle.
    bound: Option<(vk::Buffer, vk::Buffer, vk::Buffer, u64)>,
}

const GUARD_SPV: &[u8] = include_bytes!("../../shaders/preupscale_hold_guard.spv");

/// The guard's slot for a hold without a verdict (no verdicts on the device): it reads none.
pub(crate) const NO_VERDICT: u32 = u32::MAX;

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
        let bindings = [b(0), b(1), b(2)];
        // SAFETY: valid create infos; each object is destroyed below on a later failure.
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(8);
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
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
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

    /// Points the set at the `state` buffer of native pass `pass` ([`NativePass::id`]), the edit cache's
    /// counters `stats` and the inline hold's verdicts `verdicts` (`inline::Verdicts`; null for none), when they
    /// changed. `false` when it cannot be written now ([`super::may_rewrite`]).
    ///
    /// # Safety
    /// The buffers are live; nothing that may run uses the set unless it is unchanged.
    pub(crate) unsafe fn bind(&mut self, device: &ash::Device, state: vk::Buffer, pass: u64, stats: Option<vk::Buffer>, verdicts: vk::Buffer) -> bool {
        let stats = stats.unwrap_or(self.no_stats.buffer);
        let verdicts = if verdicts == vk::Buffer::null() { self.no_stats.buffer } else { verdicts };
        if self.bound == Some((state, stats, verdicts, pass)) {
            return true;
        }
        if !super::may_rewrite("the hold guard's descriptor set") {
            return false;
        }
        let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (a, b, c) = (info(state), info(stats), info(verdicts));
        fn w(set: vk::DescriptorSet, n: u32, i: &[vk::DescriptorBufferInfo; 1]) -> vk::WriteDescriptorSet<'_> {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i)
        }
        // SAFETY: the contract.
        unsafe { device.update_descriptor_sets(&[w(self.set, 0, &a), w(self.set, 1, &b), w(self.set, 2, &c)], &[]) };
        self.bound = Some((state, stats, verdicts, pass));
        true
    }

    /// Records the guard for the hold numbered `seq` (1, 2, ... in recording order), whose verdict is word `slot`
    /// of the verdicts ([`NO_VERDICT`] for none), between barriers that order it after everything before on the
    /// queue and before everything after.
    ///
    /// # Safety
    /// `cmd` is recording, outside a render pass instance; [`Self::bind`] returned `true` for this hold.
    pub(crate) unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, seq: u64, slot: u32) {
        let before = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&(seq as u32).to_le_bytes());
        push[4..].copy_from_slice(&slot.to_le_bytes());
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
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
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

/// `NEURAL_FORGE_PROBE_VRAM=1`: the VRAM guard's reading and samples logged at each of its ticks.
static PROBE_VRAM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::flag("NEURAL_FORGE_PROBE_VRAM"));

/// `NEURAL_FORGE_NATIVE_FAIL=vram`: the VRAM guard steps down as if the network had spilled at the 600th network frame
/// and again at the 900th (with NVIDIA's networks both are pauses: the first ends after the backoff and the room check,
/// the network's frames count on, and the second comes after the network is back; with a small network that frees
/// enough, the first steps to it), to exercise the steps, the pause (with the hold's resources freed) and the way back
/// on the rig. Unset (the default) does nothing.
fn fake_vram_spill(frames: u64) -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_FAIL").as_deref() == Some("vram"));
    *ON && (frames == 600 || frames == 900)
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
        return running_line(loader.guard_note().as_deref());
    }
    match (loader.failed(), result.miss) {
        (Some(why), _) => format!("native network not running: {why}"),
        (None, Some(miss)) => format!("native network: {miss}"),
        (None, None) => String::new(),
    }
}

/// The Status line of a hold whose network ran: `guard`, the VRAM guard's note.
pub(crate) fn running_line(guard: Option<&str>) -> String {
    let guard = guard.map(|why| format!(" ({why})")).unwrap_or_default();
    format!("native network running{guard}")
}

/// The last network failure written to the state log, so a retry schedule that keeps failing the same
/// way writes it once; `None` once the network was built again.
static LAST_FAILURE: Mutex<Option<String>> = Mutex::new(None);

/// A load or build failure for the state log: each different reason once.
fn note_failure(why: &str) {
    if failure_is_new(&mut LAST_FAILURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner), why) {
        crate::event!(neural_forge_protocol::state_log::NATIVE_FAILED, "{why}");
    }
}

/// A successful build: written to the state log only when it ends a failure.
fn note_recovered() {
    if LAST_FAILURE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take().is_some() {
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
        Some(Stale::GameReset) => crate::log!("[native] resetting the history (the game reset DLSS's history)"),
        Some(Stale::Unsound) => crate::log!(
            "[native] resetting the history (the hold guard reset it on the GPU: a hold inside DLSS's buffer was recorded with a guessed input layout that was wrong when the buffer ran)"
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

/// The game's reset of DLSS's history for the frame being held ([`super::DlssFrame::reset`]): the next network
/// frame, this hold's or a later one's, starts afresh ([`HistoryGap::game_reset`]; logged and not counted as a
/// scene cut when consumed), and the edit cache refreshes rather than carry an edit made for the old scene.
fn apply_game_reset<F: Copy + PartialEq>(gap: &mut HistoryGap<F>, edit: Option<&mut super::cache::State>) {
    gap.game_reset();
    if let Some(e) = edit {
        e.reset_edit();
    }
}

/// Every hold's view of the target, for [`HistoryGap`]: the input's size, or with a subrect the rectangle's
/// ([`Target::region`]: the history is in its pixels, which stand for other points of the screen at another
/// render size). Not its identification: DLSS's input alternates between two colour images every frame in
/// Black Myth: Wukong with frame generation on, and each switch reset the history.
pub(crate) fn format_of(target: &Target) -> (u32, u32) {
    target.region.unwrap_or((target.width, target.height))
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
    /// The hold's stage and the frame's groups of passes (whether the model on a second GPU takes it,
    /// [`super::placement::Hold::offload_takes`]).
    pub at: super::placement::Hold,
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
    dlss: super::DlssFrame,
    chain_ok: bool,
    passes: Passes,
    sizes: &[(u32, u32)],
    arm: Option<super::cache_dump::Arm>,
    ask: CacheAsk,
    choice: u32,
    lut: Option<&Arc<super::colour::Lut>>,
    sink: super::Sink,
) -> super::HoldResult {
    // SAFETY: forwarded.
    let result = unsafe { native_hold(device, instance, physical_device, res, target, loader, dlss, chain_ok, passes, sizes, arm, ask, choice, lut, sink) };
    if !result.evaluated {
        // Every hold that missed once the pass exists, whatever the reason: this DLSS frame went to DLSS
        // without the network, so the history is a frame older than the next frame's motion vectors, and the
        // next network frame starts afresh rather than reproject it (ghosting).
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
    }
    result
}

/// [`run_native_hold`] before a miss is marked in the history.
#[allow(clippy::too_many_arguments)]
unsafe fn native_hold(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    res: &mut super::Resources,
    target: &Target,
    loader: &Loader,
    dlss: super::DlssFrame,
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
    let super::DlssFrame { jitter, reset: game_reset } = dlss;
    let (submit, mut ctx) = match sink {
        super::Sink::Queue(s) => (Some(s), None),
        super::Sink::Buffer(c) => (None, Some(c)),
    };
    let inline = ctx.is_some();
    // Inside DLSS's buffer this hold already counts among the loader's holds (`inline::Inline::record_outcome`).
    let reserved = usize::from(inline);
    // Inside DLSS's buffer, while holds recorded with these resources may still run, nothing they use changes
    // (`super::may_rewrite`): a hold that would need a change records nothing until they are done.
    let others_outstanding = ctx.as_ref().is_some_and(|c| c.outstanding > 0);
    let frozen = super::Frozen::new(others_outstanding);
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
        // The previous network frame's exposure jump (finished: waited above), before the capture clears the buffer.
        if let Some(word) = res.hdr.as_ref().map(hdr::HdrPass::exposure_jump_word) {
            apply_exposure_jump(word, res.native.as_mut());
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
            previous_exposure = res.hdr.as_ref().map(|h| h.exposure_values().1);
        }
        // What the previous hold's auto-exposure measured (finished: waited above), with that exposure.
        if res.auto_read.take() == Some(target.identification) {
            result.auto_exposure = res.hdr.as_ref().and_then(hdr::HdrPass::auto_state);
        }
    }
    if !loader.claim_pre() {
        result.miss = Some("the network is serving the after-the-upscaler path");
        return result;
    }
    // The network on a second GPU (`super::offload_pre`): this hold sends its frame there and carries the edit of
    // the frame before it, as a cached hold does: the hold before DLSS and the hold after DLSS's output (Ray
    // Reconstruction, or passes placed there), each with a side of its own; in a frame with passes at both, both sides'
    // runners share the other device and its one network (`offload_server::Core`). With the edit cache on (it is what
    // carries the edit), and not for a cache dump (which needs the network's answer to its own frame).
    let offload_at = super::offload_pre::Where::of(ask.at.stage);
    let offload_takes = ask.at.offload_takes();
    let offloaded = super::offload_pre::wanted()
        && offload_takes
        && arm.is_none()
        && {
            let cache = ask.interval > 1;
            if !cache {
                static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::log!("[offload] NEURAL_FORGE_EDIT_CACHE=off: the edit cache is what carries the other GPU's edit to the next frame; the model runs on the game's GPU");
                }
            }
            cache
        };
    // Every frame that ran the graph has completed (this path's: the hold waited above; the other
    // path's: `claim_pre` says none ran for a second): the chain watchdog's count is final for them.
    let fake = fake_chain_timeout(res.native.as_ref().map_or(0, |n| n.frames));
    if let Some((waits, at)) = loader.chain_timeouts().filter(|(n, _)| *n > 0).or(fake) {
        if !loader.fall_back(reserved) {
            result.miss = Some(WAITING);
            return result;
        }
        crate::log!("[native] {waits} counter-chain wait(s) of the network gave up ({at}); rebuilding it with barriers between its launches");
        crate::logging::flush();
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
    // Offloaded: no network on this device (the process's one network is on the other).
    let ready = if offloaded { Some(Vec::new()) } else { loader.ready_sizes(sizes, choice, reserved) };
    let Some(builts) = ready else {
        // Paused by the VRAM guard: the hold's own frame resources go too. Idle: this path's last hold was waited
        // for above, or no hold recorded into DLSS's buffers with them may still run; this hold records nothing.
        // SAFETY: as just said.
        if loader.paused() && !others_outstanding && unsafe { res.release_native(device) } {
            crate::log!("[native] VRAM guard: the hold's frame resources are freed too until the network resumes");
            crate::logging::flush();
        }
        result.miss = Some(if loader.paused() {
            super::VRAM_PAUSED
        } else if loader.failed().is_some() {
            "the native network could not be loaded or built"
        } else if loader.waiting_for_holds(reserved) {
            WAITING
        } else {
            "the native network is still loading"
        });
        return result;
    };
    // Pass 1's: its network, its generation and its input features (where the edit cache's buffers live).
    let built = builts.get(own).copied();
    let features = built.map_or((vk::Buffer::null(), 0), |b| (b.frame.features, b.frame.features_bytes));
    let generation = built.map_or(0, |b| b.generation);
    // Counter chaining faulted the GPU from the hold inside DLSS's buffer (Black Myth: Wukong, 3 runs
    // of 4: Xid 13 then 32 within seconds; none of 2 with barriers): that hold rebuilds the network
    // with barriers between its launches, for the rest of the process (`fall_back_to_barriers`).
    if !chain_ok && builts.iter().any(|b| b.frame.chained != 0) {
        if !loader.fall_back(reserved) {
            result.miss = Some(WAITING);
            return result;
        }
        crate::log!("[native] the hold inside DLSS's buffer runs the network with barriers between its launches, not counter chaining; rebuilding it");
        crate::logging::flush();
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
    // Inside DLSS's buffer the auto-exposure's state is reset there, in the buffer's order. It runs in every hold:
    // where the game's exposure is used as it is it only measures (`ExposureSource::auto_use`), and a failed
    // build then only leaves that out.
    let mut auto_reset = false;
    // SAFETY: `device` is live; nothing of the pass is pending.
    let auto_built = pass.auto_built()
        || res.native_retry.auto.build(extent, now, || unsafe { pass.ensure_auto(device, instance, physical_device) }.then_some(())).is_some();
    if !auto_built && super::ExposureSource::measures(target) {
        result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
        return result;
    }
    if auto_built {
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
        if !offloaded {
            // Offloaded, the capture clears it itself, after the late capture read the previous hold's.
            pass.clear_exposure();
        }
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
    let passes = runnable(passes, more);
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
    if !cache_on && passes.mask.rules & MASK_SKY != 0 {
        // The sky rule reads the depth copy in the convention the edit cache decides: with the cache off neither is
        // there, and the rule is dropped (said once).
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[native] the control mask's sky rule needs the edit cache (off: NEURAL_FORGE_EDIT_CACHE): it is not applied");
        }
    }
    passes.mask = passes.mask.for_frame(depth.map_or(0, super::cache::depth_kind), res.edit.as_ref().and_then(|e| e.state.reverse_z()));
    let depth_buffer = depth.and(res.depth.as_ref()).map(DepthCopy::buffer);
    // The colour of the edit: whether this frame records it, and the look-up table.
    // SAFETY: nothing of the pass is pending (`wait_idle` above), or its table is not written now.
    unsafe { native.prepare_colour(device, instance, physical_device, &passes, lut) };
    // SAFETY: nothing of the pass is pending (`wait_idle` above), or its sets are not written now; the HDR
    // pass outlives it in `res`.
    if !offloaded && !unsafe { native.record_graphs(device, instance, physical_device, &graphs, encoded, exposure, depth_buffer) } {
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
    // The offload's side of the hold: the exchange with the other device and its worker, made on the first hold.
    if res.offpre.as_ref().is_some_and(super::offload_pre::HoldSide::released) && !others_outstanding {
        // Its worker was ended for another set of hold resources while these had no hold: built again below.
        if let Some(side) = res.offpre.take() {
            // The late capture's set points at this side's kept copy: written again for the next side's.
            if let Some(edit) = res.edit.as_mut() {
                edit.forget_late();
            }
            // SAFETY: nothing recorded with these resources may still run.
            unsafe { side.destroy(device) };
        }
    }
    if offloaded && res.offpre.is_none() {
        if matches!(super::offload::state(), super::offload::State::Stalled(_)) || super::offload::is_busy(device.handle()) {
            // A frame on the other GPU never finished: its device is not used again and the network cannot be taken
            // back, so no side is built; the frames go to DLSS as they are (said when it stalled).
            res.native = Some(native);
            result.miss = Some("a frame on the second GPU never finished: the model is off for this session");
            return result;
        }
        if others_outstanding {
            // Not while holds recorded with these resources may still run (the cache's sets are written for it).
            res.native = Some(native);
            result.miss = Some(WAITING);
            return result;
        }
        if super::offload_pre::serving_at(device.handle(), offload_at)
            && !super::offload_pre::release_idle(device.handle(), offload_at, super::offload_pre::IDLE_SIDE)
        {
            // Another set of hold resources of this device serves this hold on the offload device (another render size:
            // the other GPU keeps one pass per hold) and holds with it: these wait until it has had no hold for a few
            // seconds, and say so once. (The frame's other hold has its own place there.)
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::log!(
                    "[offload] {}: the other GPU serves another size of this game; frames of this one go on as they are until it is free",
                    offload_at.name()
                );
            }
            res.native = Some(native);
            result.miss = Some("the second GPU is serving another size of this game");
            return result;
        }
        // SAFETY: the game's live device; the offload device was made with it.
        match unsafe { super::offload_pre::HoldSide::build(device, instance, physical_device, res.id, offload_at, target.width, target.height) } {
            Ok(side) => res.offpre = Some(side),
            Err(why) => {
                crate::log!("[offload] the model stays on the game's GPU: {why}");
                crate::logging::flush();
                super::offload::note_state(super::offload::State::FellBack(why));
                res.native = Some(native);
                result.miss = Some("the second GPU could not take the network; it runs on the game's GPU from the next frame");
                return result;
            }
        }
    }
    // Given up meanwhile (the worker's rules): this hold still goes as an offloaded one, without an answer; the
    // next runs the network here.
    let offloaded = offloaded && res.offpre.is_some();
    if !offloaded && res.offpre.is_some() && !super::offload_pre::wanted() && !others_outstanding {
        // Given up for the process, this hold is not an offloaded one, and nothing recorded with these resources may
        // still run (inside DLSS's buffer: none outstanding; the split hold's work was waited for above): the
        // offload's side goes (its worker ends and the other device's side is freed).
        if let Some(side) = res.offpre.take() {
            // The late capture's set points at this side's kept copy: written again for the next side's.
            if let Some(edit) = res.edit.as_mut() {
                edit.forget_late();
            }
            // SAFETY: idle, as just said.
            unsafe { side.destroy(device) };
        }
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
        // The thumbnail is DLSS's rectangle's alone (a new size has nothing to compare with).
        s.set_region(format_of(target));
    }
    native.gap.note_format(format_of(target));
    // A frame of another network than the history's (the Network setting switched) resets the history
    // (`frame_passes` below then runs with no history: seed 0, no blend toward it, in every pass).
    if let Some(b) = built {
        if note_network(&mut native.gap, &mut native.history_network, b.network) {
            crate::log!("[native] resetting the history (the {} network runs from here)", network::name(b.network));
        }
    }
    // The game reset DLSS's history for this frame: the network's history starts afresh too, at this hold's
    // network frame or the next one, and the edit cache refreshes rather than carry the old scene's edit here.
    if game_reset && super::game_reset_honoured() {
        apply_game_reset(&mut native.gap, res.edit.as_mut().map(|e| &mut e.state));
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
            unsafe { native.cache_scratch(device, instance, physical_device, features, bytes) }.unwrap_or(features)
        }
        None => features,
    };
    let mut off_plan: Option<super::offload_pre::HoldPlan> = None;
    let mut late_bound = false;
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
            // Offloaded, the network's history is the other device's: the stored edit is one hold old at most.
            history_ok: history_ok || offloaded,
            armed: arm.is_some(),
            debug_mask: ask.debug_mask,
            generation,
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
            made: [hdr_ref.colour_binding(), native.id(), generation],
        };
        // SAFETY: nothing of the pass is pending, or its set is not written now; every handle is live.
        unsafe { edit.bind(device, &bindings) };
        if let Some(side) = res.offpre.as_mut().filter(|_| offloaded) {
            // SAFETY: as above; the kept image and the counters are live.
            unsafe {
                side.exchange.bind_pack(device, native.acc.buffer);
