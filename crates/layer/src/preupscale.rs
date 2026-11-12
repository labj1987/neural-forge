//! `NEURAL_FORGE_PREUPSCALE`: run the model on the game's own DLSS input, before its upscaler
//! (docs/PRE_UPSCALER_DESIGN.md, "Implementation (layer)").
//!
//! Under vkd3d-proton + DXVK-NVAPI, DLSS Super Resolution reaches Vulkan as CUDA kernels
//! (`vkCmdCuLaunchKernelNVX`) on image views registered through `VK_NVX_image_view_handle`.
//! The probe (`crate::probe_ngx`, docs/PRE_UPSCALER_PROBE.md) showed that in GTA V Enhanced the
//! colour input is final when the launch-bearing submit starts and sits in `GENERAL`. So this
//! module:
//!
//! 1. takes everything about DLSS's evaluates from NGX's records ([`crate::ngx_ring`], written by Neural Forge's NGX
//!    wrapper in a Neural Forge Proton): which images are its colour input, depth, motion vectors, output and exposure
//!    image, the jitter, the motion vectors' units, the reset and whether it is Ray Reconstruction. Nothing is
//!    inferred from DLSS's kernels: without records nothing is held;
//! 2. tracks the registered views and marks launch-bearing command buffers ([`Tracking`]); a buffer belongs to an
//!    evaluate when one of its launches names the evaluate's colour input between the evaluate's record and its end
//!    ([`LaunchRefs::ngx`]). Only Super Resolution's are held, so DLSS Frame Generation's and Ray Reconstruction's
//!    launch-bearing submits go through untouched;
//! 3. at a `vkQueueSubmit`/`vkQueueSubmit2` carrying a buffer with Super Resolution's evaluate, splits the call
//!    around that buffer ([`plan`]) and, between the two halves, runs its own capture submit,
//!    the model round trip and a write-back submit on the same queue ([`run_hold`]).
//!
//! Modes: `model` (the default with the variable unset; it only ever holds on a device that has the
//! NVX extensions DLSS needs and a submit carrying DLSS's input, every other device and frame keeps
//! the post-upscaler path), `off` (`NEURAL_FORGE_PREUPSCALE=off`: nothing here is reachable, no extra
//! hooks, no waits; the hooked-command list is exactly 1.1.0's), `dump`
//! (capture colour, depth, motion vectors and the 1x1 exposure images once and write them to
//! disk), `identity` (capture and write the same bytes back: the hold's own cost,
//! picture-neutral), `model` (the network runs on the colour input on the GPU: the native backend,
//! [`native`]), `roundtrip` (the model's encode and decode, [`hdr`], with the proxy itself as the
//! answer: the transform's neutrality). Every failure forwards the game's submit untouched.
//!
//! # The dependency chain of a hold
//!
//! The game's call is submitted as: `head` (batches before the launch batch, plus the launch
//! batch's command buffers before the launch buffer, with the launch batch's wait semaphores),
//! then the layer's capture batch `C`, then (after the CPU waited for `C`'s fence) the layer's
//! write-back batch `W`, then `tail` (the launch buffer onward,
//! with the launch batch's signal semaphores, then the call's later batches, with the fence).
//!
//! - Game work before the launch buffer -> `C`: every such command is earlier in submission order
//!   on the same queue, and `C` opens with a pipeline barrier `ALL_COMMANDS/MEMORY_WRITE ->
//!   TRANSFER/TRANSFER_READ|WRITE`, whose first scope is every command earlier in submission order.
//!   So the game's last write of the colour input happens-before and is visible to `C`'s copy.
//! - Work on other queues the launch batch waited for -> `C`: when the launch buffer is first in
//!   its batch, the batch's wait semaphores are moved onto `C` (timeline values kept, stage masks
//!   widened to `ALL_COMMANDS` so `C`'s transfer is inside their scope). A semaphore wait's second
//!   scope also includes every command later in submission order, so `W` and `tail` stay ordered
//!   after those signals exactly as the game asked. When the launch buffer is not first, the waits
//!   stay on the prefix in `head`, which is earlier in submission order than `C`: same argument.
//! - `C` -> host: `C` ends with `TRANSFER/TRANSFER_WRITE -> HOST/HOST_READ`; the CPU waits on `C`'s
//!   fence before anyone reads the bytes.
//! - `C` -> `W`: `W` opens with `ALL_COMMANDS|HOST -> TRANSFER` (write-after-read on the colour
//!   input after `C`'s read; `C`'s fence was also waited on).
//! - `W` -> DLSS: `W` ends with `TRANSFER/TRANSFER_WRITE -> ALL_COMMANDS/MEMORY_READ|MEMORY_WRITE`.
//!   `tail` (the launch buffer, whose kernels read the colour input) is later in submission order,
//!   so it is in that barrier's second scope: the write-back is visible to DLSS.
//! - The HDR modes (model, roundtrip; [`hdr`]) add compute work inside `C` and `W`, and the same
//!   chain holds with the stages widened: `C` opens `ALL_COMMANDS/MEMORY_WRITE ->
//!   TRANSFER|COMPUTE_SHADER` (the encode reads the colour input from a storage view), copies the
//!   exposure texel, then `TRANSFER -> COMPUTE_SHADER` (exposure visible to the encode), the encode
//!   dispatch into the layer's padded image, `COMPUTE_SHADER -> TRANSFER` and the copy into the
//!   proxy region, closed as above. `W` opens `ALL_COMMANDS|HOST -> TRANSFER|COMPUTE_SHADER` (also
//!   the write-after-read on the colour input after the encode's read), copies the answer into the
//!   layer's answer image, `TRANSFER -> COMPUTE_SHADER`, then the decode dispatch, which reads and
//!   writes the colour input in place (each invocation its own pixel only), and ends with
//!   `COMPUTE_SHADER/SHADER_WRITE -> ALL_COMMANDS/MEMORY_READ|MEMORY_WRITE`. The encoded image (in
//!   `C`) and the answer image (in `W`) go `UNDEFINED -> GENERAL` in the opening barrier (rewritten every hold;
//!   the encoded image is read by `W` of the same hold, after `C`'s fence).
//! - The game's own signals and fence: a semaphore signal's and a fence's first scope include
//!   every command earlier in submission order, so the signals on `tail[0]` and the fence on the
//!   last call still cover the prefix, `C` and `W` -- everything the game's batch covered before.
//!
//! No command is recorded into a game command buffer and no layout is changed on the colour input
//! (`GENERAL` throughout).
//!
//! # When the call is not split
//!
//! The chain above assumes the colour input is final, visible and owned by the queue's family when
//! the launch buffer starts. The layer only holds when what it saw recorded is consistent with
//! that; anything else forwards the application's call untouched ([`Hazard`], [`parse2`],
//! docs/PRE_UPSCALER_DESIGN.md, "When a submit is not split"). Tracked state advances only from
//! command buffers of submissions the next layer accepted ([`Tracker::commit_submitted`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle;
use vulkan_layer::LayerVulkanCommand as VulkanCommand;

use crate::shm::ShmClient;
use neural_forge_protocol::Slot;

#[cfg(target_arch = "x86_64")]
pub(crate) mod cache;
pub(crate) mod cache_dump;
#[cfg(target_arch = "x86_64")]
pub(crate) mod colour;
pub(crate) mod dlss_game;
pub(crate) mod hdr;
pub(crate) mod inline;
#[cfg(target_arch = "x86_64")]
pub(crate) mod native;
#[cfg(target_arch = "x86_64")]
pub(crate) mod native_post;
#[cfg(target_arch = "x86_64")]
pub(crate) mod offload;
pub(crate) mod offload_pre;
pub(crate) mod placement;
#[cfg(target_arch = "x86_64")]
pub(crate) mod post_flow;
#[cfg(target_arch = "x86_64")]
pub(crate) mod post_history;
pub(crate) mod probe_image;
pub(crate) mod restore;
#[cfg(target_arch = "x86_64")]
pub(crate) mod scene_cut;
#[cfg(target_arch = "x86_64")]
pub(crate) mod vram_guard;

/// The native backend's per-device network ([`native::Loader`]); the native crate builds for x86_64 only.
#[cfg(target_arch = "x86_64")]
pub(crate) type NativeLoader = Arc<native::Loader>;
#[cfg(not(target_arch = "x86_64"))]
#[derive(Clone)]
pub(crate) enum NativeLoader {}

/// The usage an image the application creates gets added (empty: none), for 2D, single-sample, optimally
/// tiled images only:
/// - `TRANSFER_SRC` for an RG16F image (DLSS's motion vectors; the hold copies only RG16F, for the native
///   backend's history), or, while the edit cache is on, a depth image (the hold copies DLSS's depth every
///   frame for the cache, [`native::DepthCopy`]; `NEURAL_FORGE_EDIT_CACHE=off` copies none, as 3.2.0);
/// - `TRANSFER_SRC | TRANSFER_DST` for a colour candidate: an RGBA16F or B10G11R11 image the shaders read
///   or write (`SAMPLED` or `STORAGE`) that is also rendered or written into (`COLOR_ATTACHMENT` or `STORAGE`),
///   with one mip level and one layer, as DLSS's colour input is (not a texture: those are made with mip levels or
///   only sampled, and each image the rule takes is created twice to compare its memory needs, `added_usage.rs`). The hold inside DLSS's buffer copies the
///   colour input in and back with transfers ([`inline`]); a native Vulkan game creates its images with
///   exactly the usage it needs (Indiana Jones and the Great Circle's: `SAMPLED | STORAGE |
///   COLOR_ATTACHMENT`, #51), and without them nothing could ever be held there.
///
/// Only what the image does not have already. Only asked while the pre-upscaler path is on
/// (`NEURAL_FORGE_PREUPSCALE=off` creates every image as the application asked).
pub(crate) fn added_usage(info: &vk::ImageCreateInfo) -> vk::ImageUsageFlags {
    let plain = info.image_type == vk::ImageType::TYPE_2D && info.samples == vk::SampleCountFlags::TYPE_1 && info.tiling == vk::ImageTiling::OPTIMAL;
    if !cfg!(target_arch = "x86_64") || !plain {
        return vk::ImageUsageFlags::empty();
    }
    let copied = info.format == vk::Format::R16G16_SFLOAT || (DEPTH_FORMATS.contains(&info.format) && edit_cache_on());
    let target = info.usage.intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE) && info.mip_levels == 1 && info.array_layers == 1;
    let colour = OUTPUT_FORMATS.contains(&info.format) && info.usage.intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE) && target;
    let wanted = if colour {
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
    } else if copied {
        vk::ImageUsageFlags::TRANSFER_SRC
    } else {
        vk::ImageUsageFlags::empty()
    };
    wanted & !info.usage
}

/// Whether the edit cache runs (`cache::interval` above 1): only then are DLSS's depth images made
/// copyable and copied.
pub(crate) fn edit_cache_on() -> bool {
    #[cfg(target_arch = "x86_64")]
    return cache::interval() > 1;
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Whether model-mode holds on a device with `loader` go to the native backend.
pub(crate) fn native_on(loader: Option<&NativeLoader>) -> bool {
    loader.is_some()
}

/// A native hold ([`native::run_native_hold`]).
///
/// # Safety
/// As [`run_hold`].
// The `return` is needless only where the other targets' lines below are compiled out.
#[allow(clippy::too_many_arguments, clippy::needless_return)]
pub(crate) unsafe fn run_native(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    res: &mut Resources,
    target: &Target,
    loader: &NativeLoader,
    dlss: DlssFrame,
    chain_ok: bool,
    shm: &ShmClient,
    arm: Option<cache_dump::Arm>,
    hold: u64,
    at: placement::Hold,
    sink: Sink,
) -> HoldResult {
    #[cfg(target_arch = "x86_64")]
    {
        // The Passes setting and each pass's settings (the Model tab's with that pass's overrides).
        let all = shm.pass_tunings().map_or_else(native::Passes::default, native::Passes::from);
        // This stage's passes ([`placement`]), and the sizes the network is built for: both holds' of the frame,
        // so that the holds before the upscaler and after DLSS ask for the same build.
        let Some(mut passes) = native::stage_passes(&all, at) else {
            return HoldResult { native: true, miss: Some("no pass runs at this hold's stage"), ..Default::default() };
        };
        let sizes = native::frame_sizes(&all, at, (target.width, target.height));
        // The control mask's rules (the hold adds what it knows of the depth).
        passes.mask = shm.mask_settings().map_or_else(native::Mask::default, native::Mask::from);
        // The edit cache's interval (its own, not `model_interval`: that is the after-the-upscaler
        // path's), and the debug view of its rejection mask.
        let ask = native::CacheAsk { hold, interval: cache::interval(), debug_mask: shm.debug_view() == CACHE_DEBUG_VIEW };
        // Told when this hold's work is waited on (`Resources::wait_idle`): until then the
        // after-the-upscaler path keeps off the network.
        res.pre_loader = Some(loader.clone());
        // SAFETY: forwarded.
        // The Network setting: which extracted network runs (a switch takes effect at this hold).
        let choice = shm.network_choice().unwrap_or(neural_forge_protocol::enums::network::FULL);
        // The colour of the edit's look-up table (read from config.ini's `edit_lut` in the background).
        let lut = colour::current(&passes.colour);
        return unsafe {
            native::run_native_hold(
                device,
                instance,
                physical_device,
                res,
                target,
                loader,
                dlss,
                chain_ok,
                passes,
                &sizes,
                arm,
                ask,
                choice,
                lut.as_ref(),
                sink,
            )
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (device, instance, physical_device, res, target, dlss, chain_ok, shm, arm, hold, at, sink);
    #[cfg(not(target_arch = "x86_64"))]
    match *loader {}
}

/// `debug_view`'s value for the edit cache's rejection mask (`ShmHeader::debug_view`).
pub(crate) const CACHE_DEBUG_VIEW: u32 = 6;

/// The variable that selects the mode. Read through `neural_forge_protocol::env`.
pub(crate) const ENV: &str = "NEURAL_FORGE_PREUPSCALE";

/// How long after the last hold the device counts as holding (`preupscale_state` 2).
pub(crate) const RECENT: Duration = Duration::from_millis(500);

/// In model mode, on a device that has held, how long DLSS's evaluates may go on being submitted with no
/// frame held before the post-upscaler path takes over (frames are presented as DLSS made them until then): DLAA, inputs
/// no longer identified, every frame refused. With no Super Resolution launch at all since the last held frame the game
/// paused DLSS (a pause menu, a map, a loading screen): the model stays where it is for [`PAUSE_HAND_BACK`]. Handing a
/// loading screen to the post path rebuilt the model at the output size and back at every loading screen (twice per
/// screen, at 4K near full VRAM); loading screens do not need the model.
pub(crate) const HAND_BACK: Duration = Duration::from_secs(30);

/// [`HAND_BACK`] while no DLSS evaluate was submitted at all since the last held frame: the game paused it (a
/// pause menu over the last frame, which already carries the model's edit, #49). Handing those frames to the post
/// path after 30 s ran the network a second time over the menu (at the output size: 49 ms a frame at 4K), rebuilt it
/// at the output size and back at the render size on resume (the model off for over a second, and on a 12 GB card at
/// 4K the VRAM guard paused it for good after the first long pause). A game whose DLSS was switched off for another
/// upscaler gets the post path after this.
pub(crate) const PAUSE_HAND_BACK: Duration = Duration::from_secs(300);

/// The submits of DLSS evaluates a device of the process saw, held or not ([`note_sr_launch`]):
/// when the current run of them began (launches less than [`RECENT`] apart) and the last one.
static SR_LAUNCHES: Mutex<Option<SrLaunches>> = Mutex::new(None);

/// A run of DLSS Super Resolution's launches ([`SR_LAUNCHES`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SrLaunches {
    pub since: Instant,
    pub last: Instant,
}

impl SrLaunches {
    /// The run after a launch at `now`.
    fn after(run: Option<Self>, now: Instant) -> Self {
        match run {
            Some(r) if now.saturating_duration_since(r.last) < RECENT => Self { last: now, ..r },
            _ => Self { since: now, last: now },
        }
    }
}

/// A launch-bearing submit carried a DLSS evaluate's launches, held or not ([`Tracker::observe`]).
fn note_sr_launch() {
    let mut run = lock(&SR_LAUNCHES);
    *run = Some(SrLaunches::after(*run, Instant::now()));
}

fn sr_launches() -> Option<SrLaunches> {
    *lock(&SR_LAUNCHES)
}

/// When the post path gets the frames back after the last held frame at `last`: [`HAND_BACK`] after Super Resolution
/// went on being evaluated without a frame held (inputs not identified, every frame refused: a run of evaluates
/// that goes on more than [`RECENT`] past `last`, counted from `last` or from the run's start if it began later), else
/// [`PAUSE_HAND_BACK`] after `last` (DLSS paused: a pause menu, a loading screen). A run that begins as the game
/// resumes is DLSS coming back, and its input is held again within a few frames: it hands nothing back for
/// [`HAND_BACK`] (handing back at the resume rebuilt the network for the post path under the first holds, #59).
pub(crate) fn hand_back_at(last: Instant, launches: Option<SrLaunches>) -> Instant {
    match launches.filter(|r| r.last.saturating_duration_since(last) > RECENT) {
        Some(r) => r.since.max(last) + HAND_BACK,
        None => last + PAUSE_HAND_BACK,
    }
}

/// The largest value of the game's exposure image the HDR encode trusts. Measured exposures:
/// GTA V 0.15-0.34, Crimson Desert 0.02-0.06, Cyberpunk 2077 2.7-3.3; Wukong's DLSS-internal 1x1
/// read up to 59,456 ([`Session::check_exposure`]).
pub(crate) const EXPOSURE_TRUST_MAX: f32 = 1000.0;

/// The largest factor the game's exposure image may change by between two holds of one
/// identification without counting as a jump ([`Session::check_exposure`]). A game's eye adaptation
/// moves a few stops a second at most; Hogwarts Legacy's 1x1 read 0.0027-0.0034 with the frames
/// untouched, and jumped between about 0.0023 and 0.000013 (180x) from hold to hold once the
/// model's frames reached the game's own adaptation (2026-10-09): the encode then darkened the frame
/// to nothing and the network answered noise.
pub(crate) const EXPOSURE_JUMP_MAX: f32 = 32.0;

/// Jumps within [`EXPOSURE_JUMP_WINDOW`] holds that make the game's exposure untrusted. One jump, or
/// one there and back, is a cut that resets the game's adaptation (Crimson Desert: 0.0069 to 0.91
/// once, at its menus, 2026-10-09); Hogwarts Legacy's runaway jumps on most holds.
pub(crate) const EXPOSURE_JUMPS: u32 = 3;

/// Holds in a row whose game exposure read 0, negative or not a number after which it is not trusted
/// ([`Session::check_exposure`]): Metro Last Light Redux names a 1x1 R16F for DLSS that holds 0 on every frame
/// (2026-10-23), and the encode then had no usable exposure, so the decode wrote nothing back and the model
/// before the upscaler never showed. A few, so a frame whose copy came too early does not decide it.
pub(crate) const EXPOSURE_UNUSABLE_HOLDS: u32 = 8;
/// See [`EXPOSURE_JUMPS`]: the count starts over after this many holds without a jump.
pub(crate) const EXPOSURE_JUMP_WINDOW: u32 = 120;

/// A colour input the game already exposed ([`Session::check_exposure`]): Clair Obscur: Expedition 33 (Unreal
/// Engine 5 through vkd3d-proton, #51) hands DLSS its frame pre-exposed (the scene near display brightness) and
/// an eye-adaptation texel of 0.0078-0.019 beside it, which DLSS divides back out with the game's pre-exposure,
/// a value a Vulkan layer does not see. Multiplied in, that texel put the frame 7-9 stops below the brightness
/// the network is tuned for (GTA V's, [`hdr::AUTO_KEY`]), the decode multiplied the network's change back up by
/// as much, and the picture got a white fog. The game's exposure is taken for that when it puts the frame more
/// than this many stops below the auto-exposure's key ...
pub(crate) const PREEXPOSED_STOPS: f32 = 5.0;
/// ... while the frame as given needs at most this many stops of lift itself (its log-average luma at least an
/// eighth of the key: already exposed). A scene-referred frame whose exposure is far below the key (Crimson
/// Desert's dark scenes, 2026-10-22) is darker than that, and keeps the game's exposure, lifted at most
/// [`hdr::LIFT_HI`] by the brightness setting's Auto. Games whose exposure is their eye adaptation keep the frame
/// near the key: GTA V's put it at 0.47-0.76, about a stop around it.
pub(crate) const PREEXPOSED_FRAME_STOPS: f32 = 3.0;
/// ... on every hold for [`PREEXPOSED_WINDOW`] (and at least [`PREEXPOSED_HOLDS`] holds), while the game's exposure
/// stayed within [`PREEXPOSED_STEADY_STOPS`]; a hold under [`PREEXPOSED_CLEAR_STOPS`] (or a frame darker than
/// [`PREEXPOSED_FRAME_STOPS`]) starts the run over, one between the two leaves it. A scene-referred game walking
/// from daylight into a dark interior puts the frame as far below the key while its eye adaptation catches up
/// (GTA V: e 0.15 on a daylight frame of luma 6, then a tunnel of 0.1), but its exposure moves meanwhile; a
/// pre-exposed one's holds nearly still (Expedition 33: 0.0078-0.019 from scene to scene).
pub(crate) const PREEXPOSED_WINDOW: Duration = Duration::from_millis(2500);
/// See [`PREEXPOSED_WINDOW`].
pub(crate) const PREEXPOSED_HOLDS: u32 = 30;
/// See [`PREEXPOSED_WINDOW`]: how far the game's exposure may move (log2, the lowest to the highest of the run) and
/// still count as held steady. Half a stop: an eye adaptation catching up moves more within the window.
pub(crate) const PREEXPOSED_STEADY_STOPS: f32 = 0.5;
/// See [`PREEXPOSED_WINDOW`]. Once the frame is measured, the same run of holds under this (or of a frame darker
/// than [`PREEXPOSED_FRAME_STOPS`]) hands it back to the game's exposure; a hold past [`PREEXPOSED_STOPS`] starts
/// that run over.
pub(crate) const PREEXPOSED_CLEAR_STOPS: f32 = 4.0;
/// A gap between two holds longer than this starts a run of [`PREEXPOSED_WINDOW`] over: nothing was seen between.
pub(crate) const PREEXPOSED_GAP: Duration = Duration::from_millis(500);
/// Each direction of the switch is logged at most once in this long; the changes left out are counted in the next line.
pub(crate) const PREEXPOSED_SAY_EVERY: Duration = Duration::from_secs(60);

/// The fence waits of this path, by class. Both are the layer-wide bound today
/// ([`crate::FENCE_WAIT_TIMEOUT`], 5 s, there against a driver that stalls without losing the
/// device); they are named apart so that the one on the game's submit thread can be given its own
/// bound without touching the recovery wait. Not shortened without measurements from the target
/// machine: the summary line's `capture_wait max` (per window and for the session) is what to read
/// across loading screens, resolution changes, alt-tab and shutdown before choosing a value.
/// (Device teardown waits with `vkDeviceWaitIdle`, which has no bound.)
///
/// The capture fence wait inside a hold: the game's DLSS submit is blocked for its duration.
pub(crate) const FRAME_CAPTURE_WAIT: Duration = crate::FENCE_WAIT_TIMEOUT;
/// Draining the path's own in-flight work before its resources are rebuilt or destroyed
/// ([`Resources::wait_idle`]): off the steady frame path, and a timeout keeps the resources alive.
pub(crate) const CLEANUP_WAIT: Duration = crate::FENCE_WAIT_TIMEOUT;

/// A `[preupscale]` summary line every this many holds.
const SUMMARY_EVERY: u64 = 300;

/// Bytes per RGBA16F texel.
const TEXEL: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Dump,
    Identity,
    Model,
    /// The HDR encode and its inverse with the answer := the proxy itself (no model): checks
    /// the transform's neutrality on the rig.
    Roundtrip,
}

impl Mode {
    /// `None` (unset) and empty are the default, [`Mode::Model`]; `off` (or `0`) is off; anything
    /// unrecognised is an error (and off).
    pub(crate) fn parse(value: Option<&str>) -> Result<Mode, String> {
        match value.map(str::trim) {
            None | Some("") => Ok(DEFAULT),
            Some("off") | Some("0") => Ok(Mode::Off),
            Some("dump") => Ok(Mode::Dump),
            Some("identity") => Ok(Mode::Identity),
            Some("model") => Ok(Mode::Model),
            Some("roundtrip") => Ok(Mode::Roundtrip),
            Some(other) => Err(other.to_string()),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Dump => "dump",
            Mode::Identity => "identity",
            Mode::Model => "model",
            Mode::Roundtrip => "roundtrip",
        }
    }

    /// Whether the hold encodes the colour input for the model and decodes on write-back
    /// ([`hdr`]). `identity` stays a raw copy-through (the hold's own cost).
    pub(crate) fn hdr(self) -> bool {
        matches!(self, Mode::Model | Mode::Roundtrip)
    }

    /// Whether the mode holds inside DLSS's command buffer where the split is refused ([`inline`]): model
    /// mode only. The hold is recorded into the buffer and nothing of it waits on the host, so the
    /// diagnostic modes, which read the capture back on the host or hand it to the model server, hold only
    /// at the split.
    pub(crate) fn holds_inline(self) -> bool {
        matches!(self, Mode::Model)
    }
}

/// The mode with `NEURAL_FORGE_PREUPSCALE` unset: the model runs before the upscaler wherever DLSS
/// Super Resolution's input is identified (measured 66.1 fps against 61.4-61.9 for the post path,
/// GTA V Enhanced at DLSS Balanced 1440p; docs/PRE_UPSCALER_DESIGN.md, "Hand-off latency").
pub(crate) const DEFAULT: Mode = Mode::Model;

/// The mode for `raw` (the variable's value) in a process where the layer is switched on
/// (`switched_on`: `NEURAL_FORGE_ENABLE` set and `NEURAL_FORGE_DISABLE` not). Off when the layer is
/// switched off or the value is not a mode.
pub(crate) fn resolve(raw: Option<&str>, switched_on: bool) -> Result<Mode, String> {
    match Mode::parse(raw) {
        Ok(mode) if switched_on => Ok(mode),
        Ok(_) => Ok(Mode::Off),
        Err(other) => Err(other),
    }
}

/// The mode for this process. Cached; it cannot change for the life of the process.
///
/// Only the environment is read here, never `crate::layer_enabled()`: this is first asked while the
/// instance is created (for the hooked-command list), and evaluating the duplicate-copy check there
/// would let an inner copy of the layer decide before the outer one, which it never did before.
/// The hold itself checks `layer_enabled()` (`NeuralForgeDeviceInfo::preupscale_submit`).
pub(crate) fn mode() -> Mode {
    static MODE: LazyLock<Mode> = LazyLock::new(|| {
        let raw = neural_forge_protocol::env::var(ENV);
        let switched_on = crate::env_flag("NEURAL_FORGE_ENABLE") && !crate::env_flag("NEURAL_FORGE_DISABLE");
        let mode = match resolve(raw.as_deref(), switched_on) {
            Ok(mode) => mode,
            Err(other) => {
                crate::log!("[preupscale] {ENV}={other:?} is not one of off, model, dump, identity, roundtrip; staying off");
                Mode::Off
            }
        };
        if mode != Mode::Off {
            crate::log!(
                "[preupscale] mode {}{} in pid {}: NVX view-registration and launch tracking on devices that have VK_NVX_image_view_handle",
                mode.name(),
                if raw.as_deref().is_none_or(|v| v.trim().is_empty()) { " (default)" } else { "" },
                std::process::id()
            );
            if mode == Mode::Model {
                // `NEURAL_FORGE_EDIT_CACHE`, read (and logged when off) with the mode, where the
                // edit cache is built at all (`edit_cache_on`).
                let _ = edit_cache_on();
            }
            crate::logging::flush();
        }
        mode
    });
    *MODE
}

/// Whether any non-off mode is selected.
pub(crate) fn active() -> bool {
    mode() != Mode::Off
}

/// Whether a device gets the tracking (and so can ever hold): a mode is on and the device has
/// `VK_NVX_image_view_handle` (`nvx`), through which DLSS registers its inputs. A device without it
/// stays exactly on the post-upscaler path.
pub(crate) fn wanted_on_device(mode_on: bool, nvx: bool) -> bool {
    mode_on && nvx
}

/// The device commands this module adds to the framework's hooked set when a mode is on. The
/// rest it needs (`vkCreateImage`, the barriers, `vkBeginCommandBuffer`, `vkQueueSubmit*`, ...) are
/// in the default set.
pub(crate) const COMMANDS: &[VulkanCommand] = &[
    VulkanCommand::CreateImageView,
    VulkanCommand::DestroyImageView,
    VulkanCommand::GetImageViewHandleNvx,
    VulkanCommand::GetImageViewAddressNvx,
    VulkanCommand::CmdCuLaunchKernelNvx,
    // The kernels' names, for the probes' lines only (NGX's records say which launches are DLSS's).
    VulkanCommand::CreateCuFunctionNvx,
    VulkanCommand::DestroyCuFunctionNvx,
    // Synchronization a split must not come before or between ([`Hazard`]): event waits, and
    // dynamic rendering suspended in one command buffer and resumed in the next.
    VulkanCommand::CmdWaitEvents,
    VulkanCommand::CmdWaitEvents2,
    VulkanCommand::CmdBeginRendering,
    // Copies into DLSS's inputs inside its launch buffer ([`Tracker::copy_source`]); vkd3d-proton copies with
    // `vkCmdCopyImage2` (`vkCmdCopyImage` is hooked for the render tap).
    VulkanCommand::CmdCopyImage2,
    // Each pool's queue family: the hold inside DLSS's buffer is only recorded into a buffer of
    // DLSS's graphics family ([`inline::Inline::record_outcome`]).
    VulkanCommand::CreateCommandPool,
    // An image `vkCreateImage` gets `TRANSFER_SRC` added to ([`wants_copyable`]) is queried with it.
    VulkanCommand::GetDeviceImageMemoryRequirements,
    VulkanCommand::GetDeviceImageSparseMemoryRequirements,
    // What the application binds for compute and pushes, put back after a hold recorded into its buffer
    // ([`restore`]); conditional rendering, which the hold's dispatches would be subject to.
    VulkanCommand::CmdBindPipeline,
    VulkanCommand::CmdBindDescriptorSets,
    VulkanCommand::CmdPushConstants,
    VulkanCommand::CmdPushDescriptorSetKhr,
    VulkanCommand::CmdPushDescriptorSetWithTemplateKhr,
    VulkanCommand::CmdBindDescriptorBuffersExt,
    VulkanCommand::CmdSetDescriptorBufferOffsetsExt,
    VulkanCommand::CmdBindDescriptorBufferEmbeddedSamplersExt,
    VulkanCommand::CreateDescriptorUpdateTemplate,
    VulkanCommand::DestroyDescriptorUpdateTemplate,
    VulkanCommand::CmdBeginConditionalRenderingExt,
    VulkanCommand::CmdEndConditionalRenderingExt,
];

/// What the diagnostic modes' split hold (`dump`, `identity`, `roundtrip`) hooks besides [`COMMANDS`]: the
/// semaphores' creation, so the hold knows which are timeline semaphores (whose waits it checks before it
/// waits on the host: [`waits_reached`]).
pub(crate) const DIAGNOSTIC_COMMANDS: &[VulkanCommand] = &[VulkanCommand::CreateSemaphore, VulkanCommand::DestroySemaphore];

/// Whether the mode is one of the diagnostics the split hold runs with a wait on the host ([`run_hold`]).
pub(crate) fn diagnostic(mode: Mode) -> bool {
    matches!(mode, Mode::Dump | Mode::Identity | Mode::Roundtrip)
}

/// [`COMMANDS`] (and in a diagnostic mode [`DIAGNOSTIC_COMMANDS`]) when `on`, nothing otherwise.
pub(crate) fn commands(on: bool) -> impl Iterator<Item = &'static VulkanCommand> {
    let extra = if diagnostic(mode()) { DIAGNOSTIC_COMMANDS } else { &[] };
    COMMANDS.iter().chain(extra).filter(move |_| on)
}

/// Whether every wait in `waits` on a timeline semaphore (`timeline` tells which are) is already reached:
/// the split hold of a diagnostic mode waits on the host for its capture, which runs after them, inside the
/// application's submit. A value only the host signals later (after the submit returns) would hold that wait
/// to its timeout; one the GPU has not reached yet would too, for as long as the GPU takes.
///
/// # Safety
/// Every semaphore `timeline` accepts is a live timeline semaphore of `device`.
pub(crate) unsafe fn waits_reached(
    device: &ash::Device,
    waits: impl IntoIterator<Item = (vk::Semaphore, u64)>,
    timeline: impl Fn(vk::Semaphore) -> bool,
) -> bool {
    waits.into_iter().filter(|(s, _)| timeline(*s)).all(|(semaphore, value)| {
        // SAFETY: per this function's contract.
        unsafe { device.get_semaphore_counter_value(semaphore) }.is_ok_and(|reached| reached >= value)
    })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn hex<T: Handle>(handle: T) -> String {
    format!("{:#x}", handle.as_raw())
}

// ---- Registered-view tracking and colour-input identification. ----

/// What `vkCreateImage` said about an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    /// 2D, single-sample: the only kind the copies here handle.
    pub plain: bool,
}

impl ImageDesc {
    pub(crate) fn from_info(info: &vk::ImageCreateInfo) -> Self {
        Self {
            width: info.extent.width,
            height: info.extent.height,
            format: info.format,
            usage: info.usage,
            plain: info.image_type == vk::ImageType::TYPE_2D && info.samples == vk::SampleCountFlags::TYPE_1 && info.extent.depth == 1,
        }
    }
}

/// Depth as a single-channel 32-bit colour image: what DLSS reads when Streamline hands it a copy of
/// the depth. Cyberpunk 2077 with DLSS Frame Generation copies its depth into an `R32_SFLOAT` image in
/// DLSS's buffer right before the launches (NGX probe `cp/cp-fgprobe-1`, 2026-10-05), and no
/// depth-format image is registered at all then: NGX's record names it as the depth, and the hold copies it from
/// [`depth_copy_layout`]'s layouts.
const DEPTH_AS_COLOUR: [vk::Format; 2] = [vk::Format::R32_SFLOAT, vk::Format::R32_UINT];

pub(crate) const DEPTH_FORMATS: [vk::Format; 6] = [
    vk::Format::D16_UNORM,
    vk::Format::X8_D24_UNORM_PACK32,
    vk::Format::D32_SFLOAT,
    vk::Format::D16_UNORM_S8_UINT,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D32_SFLOAT_S8_UINT,
];

pub(crate) fn has_stencil(format: vk::Format) -> bool {
    matches!(format, vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT)
}

/// Bytes per texel of the depth aspect copied out of a depth format.
pub(crate) fn depth_texel_bytes(format: vk::Format) -> u64 {
    match format {
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => 2,
        _ => 4,
    }
}

/// Bytes per texel of a float colour format a 1x1 exposure image can have; `None` for any other.
pub(crate) fn exposure_texel_bytes(format: vk::Format) -> Option<u64> {
    match format {
        vk::Format::R16_SFLOAT => Some(2),
        vk::Format::R32_SFLOAT | vk::Format::R16G16_SFLOAT => Some(4),
        vk::Format::R32G32_SFLOAT | vk::Format::R16G16B16A16_SFLOAT => Some(8),
        vk::Format::R32G32B32A32_SFLOAT => Some(16),
        _ => None,
    }
}

/// The 1x1 exposure images a dump can read ([`Target::exposure`]); the split hold fills the first, with DLSS's exposure
/// input as NGX's record names it.
pub(crate) const MAX_EXPOSURE: usize = 4;

/// Bytes reserved per exposure image in the dump readback (the largest texel).
const EXPOSURE_STRIDE: u64 = 16;

/// The shape of the identified colour input ([`Tracker::colour_watch`]): every 2D single-sample image of its extent
/// and format is watched, as every depth image at the depth's extent is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ColourWatch {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

impl ColourWatch {
    fn of(d: &ImageDesc) -> Self {
        Self { width: d.width, height: d.height, format: d.format }
    }

    fn covers(&self, d: &ImageDesc) -> bool {
        d.plain && d.format == self.format && (d.width, d.height) == (self.width, self.height)
    }
}

/// At most one line per this long with the counts of the ordinary hold-inside refusals ([`InlineRefusal::quiet`]).
const QUIET_REFUSALS_EVERY: Duration = Duration::from_secs(5);

/// DLSS's inputs as NGX's record of an evaluate names them ([`crate::ngx_ring`]), each with what `vkCreateImage` said of
/// it: the colour input, depth, motion vectors and exposure image of the identification ([`Tracker::refresh`]), or of
/// one command buffer's evaluate ([`Scan::inputs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inputs {
    pub colour: (vk::Image, ImageDesc),
    pub depth: (vk::Image, ImageDesc),
    pub mvec: (vk::Image, ImageDesc),
    /// DLSS's exposure input (`ExposureTexture`), which the HDR encode multiplies in; `None` when the game gives DLSS
    /// none (the exposure is measured from the frame then).
    pub exposure_input: Option<(vk::Image, ImageDesc)>,
}

/// The motion-vector formats the hold inside DLSS's buffer copies into its RG16F staging image: RG16F as it is,
/// the others blitted (which converts; Vulkan requires blits from RGBA16F and RG32F and into RG16F with optimal
/// tiling). RGBA16F is Metro Last Light Redux's.
pub(crate) const MVEC_HOLD_FORMATS: [vk::Format; 3] = [vk::Format::R16G16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT, vk::Format::R32G32_SFLOAT];

/// The formats of an image DLSS Super Resolution writes (its output, NGX's output-size scratch).
const OUTPUT_FORMATS: [vk::Format; 2] = [vk::Format::R16G16B16A16_SFLOAT, vk::Format::B10G11R11_UFLOAT_PACK32];

/// The colour input's layouts the hold inside DLSS's buffer moves from and back to (any other, or an
/// unknown one, gets no hold there).
const INLINE_LAYOUTS: [vk::ImageLayout; 5] = [
    vk::ImageLayout::GENERAL,
    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
];

/// One image barrier on a watched image, as the application recorded it (stages and accesses as
/// their sync2 values; the sync1 bits are the same low bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageSync {
    pub image: vk::Image,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_stage: u64,
    pub dst_stage: u64,
    pub src_access: u64,
    pub dst_access: u64,
    pub src_queue_family: u32,
    pub dst_queue_family: u32,
    /// The barrier's aspect mask (`VkImageAspectFlags`); 0 where not known (the whole image).
    pub aspect: u32,
    /// A barrier on one aspect of a depth/stencil image that keeps that aspect's layout
    /// (`separateDepthStencilLayouts`): it orders memory and moves neither aspect, so the image is
    /// in whatever layout it was in before it ([`Self::layout_after`]). Set by [`Tracker::barrier`].
    pub keeps_layout: bool,
}

/// The layout tracked for a depth/stencil image after a barrier on one of its two aspects only
/// (`separateDepthStencilLayouts`): its depth and stencil may now be in different layouts, so no
/// copy or transition of the image is recorded from it ([`aux_copy_layout`], [`depth_copy_layout`])
/// until a barrier on both aspects says where they are.
pub(crate) const ASPECTS_APART: vk::ImageLayout = vk::ImageLayout::from_raw(0x7fff_0001);

impl ImageSync {
    pub(crate) fn from_barrier(b: &vk::ImageMemoryBarrier, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags) -> Self {
        Self {
            image: b.image,
            old_layout: b.old_layout,
            new_layout: b.new_layout,
            src_stage: u64::from(src_stage.as_raw()),
            dst_stage: u64::from(dst_stage.as_raw()),
            src_access: u64::from(b.src_access_mask.as_raw()),
            dst_access: u64::from(b.dst_access_mask.as_raw()),
            src_queue_family: b.src_queue_family_index,
            dst_queue_family: b.dst_queue_family_index,
            aspect: b.subresource_range.aspect_mask.as_raw(),
            keeps_layout: false,
        }
    }

    pub(crate) fn from_barrier2(b: &vk::ImageMemoryBarrier2) -> Self {
        Self {
            image: b.image,
            old_layout: b.old_layout,
            new_layout: b.new_layout,
            src_stage: b.src_stage_mask.as_raw(),
            dst_stage: b.dst_stage_mask.as_raw(),
            src_access: b.src_access_mask.as_raw(),
            dst_access: b.dst_access_mask.as_raw(),
            src_queue_family: b.src_queue_family_index,
            dst_queue_family: b.dst_queue_family_index,
            aspect: b.subresource_range.aspect_mask.as_raw(),
            keeps_layout: false,
        }
    }

    /// A barrier that leaves `image` in `layout`, with no ownership transfer.
    #[cfg(test)]
    pub(crate) fn to(image: vk::Image, layout: vk::ImageLayout) -> Self {
        Self {
            image,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: layout,
            src_stage: 0,
            dst_stage: 0,
            src_access: 0,
            dst_access: 0,
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
            aspect: 0,
            keeps_layout: false,
        }
    }

    /// Whether this barrier covers only one of a depth/stencil image's two aspects.
    fn one_aspect_of_two(&self) -> bool {
        let both = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        let a = vk::ImageAspectFlags::from_raw(self.aspect);
        a.intersects(both) && !a.contains(both)
    }

    /// The image's layout after this barrier, as the tracking records it; `None` for one that keeps
    /// whatever layout the image was in ([`Self::keeps_layout`]).
    pub(crate) fn layout_after(&self) -> Option<vk::ImageLayout> {
        (!self.keeps_layout).then_some(self.new_layout)
    }

    /// A queue-family ownership transfer (its release or its acquire: both name the same pair).
    pub(crate) fn transfers_ownership(&self) -> bool {
        self.src_queue_family != self.dst_queue_family
    }

    /// Whether what came before this barrier may have written the image: a layout transition, an
    /// ownership transfer, or a write in its source access. A same-layout barrier after reads only
    /// (GTA V Enhanced's on its depth, around the copy into DLSS's own) orders reads.
    pub(crate) fn may_follow_a_write(&self) -> bool {
        self.old_layout != self.new_layout || self.transfers_ownership() || self.src_access & WRITE_ACCESS != 0
    }

    /// Why a hold must not run before this barrier.
    fn hazard(&self) -> Hazard {
        if self.transfers_ownership() {
            Hazard::QueueFamilyTransfer
        } else if self.old_layout == self.new_layout {
            Hazard::SameLayoutBarrier
        } else {
            Hazard::LayoutTransition
        }
    }
}

/// Every write bit of `VkAccessFlags2` (the legacy ones keep their values): shader, colour and depth
/// attachment, transfer, host and memory writes, shader storage writes, video decode and encode,
/// transform feedback and its counter, command preprocess, acceleration structure, optical flow and
/// micromap writes. Read bits (`TRANSFORM_FEEDBACK_COUNTER_READ` among them) are not writes.
const WRITE_ACCESS: u64 = {
    use vk::AccessFlags2 as A;
    A::SHADER_WRITE.as_raw()
        | A::COLOR_ATTACHMENT_WRITE.as_raw()
        | A::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
        | A::TRANSFER_WRITE.as_raw()
        | A::HOST_WRITE.as_raw()
        | A::MEMORY_WRITE.as_raw()
        | A::SHADER_STORAGE_WRITE.as_raw()
        | A::VIDEO_DECODE_WRITE_KHR.as_raw()
        | A::VIDEO_ENCODE_WRITE_KHR.as_raw()
        | A::TRANSFORM_FEEDBACK_WRITE_EXT.as_raw()
        | A::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT.as_raw()
        | A::COMMAND_PREPROCESS_WRITE_NV.as_raw()
        | A::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw()
        | A::OPTICAL_FLOW_WRITE_NV.as_raw()
        | A::MICROMAP_WRITE_EXT.as_raw()
};

/// Why a launch-bearing submit cannot be split in front of its launch buffer: the hold would read
/// (and write) DLSS's inputs before synchronization the application recorded for them, or between
/// two command buffers that must stay adjacent. Such a submit is forwarded untouched.
///
/// The layer only reasons about what it saw recorded. In GTA V Enhanced under vkd3d-proton nothing
/// of this is in the launch buffer before its first launch (docs/PRE_UPSCALER_PROBE.md: the last
/// barrier on the colour input is in an earlier buffer), which is the pattern that is held; that is
/// an observation about that game, not something Vulkan promises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hazard {
    /// A queue-family ownership transfer of an input, in the launch buffer before the launch.
    QueueFamilyTransfer,
    /// A barrier on an input that keeps its layout (`GENERAL -> GENERAL`): memory visibility or an
    /// execution dependency the launch relies on.
    SameLayoutBarrier,
    /// A layout transition of an input in the launch buffer before the launch.
    LayoutTransition,
    /// A global memory barrier with a write in its source access before the launch: it can be what
    /// makes the producer's writes of an input visible.
    MemoryBarrier,
    /// `vkCmdWaitEvents*` before the launch.
    EventWait,
    /// The launch buffer resumes dynamic rendering suspended in the buffer before it
    /// (`VK_RENDERING_RESUMING_BIT`): nothing may be submitted between the two.
    SuspendedRendering,
    /// Not a synchronization hazard: the colour input is not an RGBA16F image in `GENERAL`, which the
    /// split needs (it reads and writes it as a storage image); only the hold inside the buffer,
    /// which converts, takes it ([`inline`]).
    NotSplittable,
    /// Not a hazard: the split could take this buffer, but this identification's holds already run
    /// inside DLSS's buffer ([`Tracker::inline_recorded`]). Hogwarts Legacy refuses the split for some
    /// of its DLSS buffers and not others; holding those at the split as well alternated the two
    /// holds frame by frame, and each switch rebuilt the hold's resources (the split's on DLSS's
    /// queue family, the inside hold's on the side queue's) and with them the native network's
    /// pipelines and history (2026-10-09).
    HeldInside,
    /// The launch buffer, or a buffer before it in the submit, was recorded (at least in part) before
    /// the current identification ([`Tracker::recorded_watched`]): barriers it holds on the inputs
    /// were never seen. A D3D12 command list executed again without being re-recorded, or one recorded
    /// ahead of the submit that identified the inputs, may carry a transition or a same-layout barrier
    /// before DLSS's launch.
    RecordedBefore,
    /// Not a synchronization hazard: DLSS is given part of its inputs (NGX's subrect,
    /// [`crate::ngx_ring::Record::crop`]). The split works on the whole colour input, so it would run the network
    /// on pixels DLSS never reads and write over them; only the hold inside DLSS's buffer, which copies the
    /// rectangle alone, takes such a frame.
    Subrect,
}

impl Hazard {
    pub(crate) fn why(self) -> &'static str {
        match self {
            Self::QueueFamilyTransfer => "the DLSS launch buffer transfers queue-family ownership of a DLSS input before its launch; not holding (the layer cannot run ahead of that transfer), frames go to DLSS untouched",
            Self::SameLayoutBarrier => "the DLSS launch buffer has a same-layout barrier on a DLSS input before its launch; not holding (the layer cannot run ahead of that synchronization), frames go to DLSS untouched",
            Self::LayoutTransition => "the DLSS launch buffer transitions a DLSS input's layout before its launch; not holding, frames go to DLSS untouched",
            Self::MemoryBarrier => "the DLSS launch buffer has a global memory barrier with a write in its source access before its launch; not holding (it may be what makes the input visible), frames go to DLSS untouched",
            Self::EventWait => "the DLSS launch buffer waits on an event before its launch; not holding, frames go to DLSS untouched",
            Self::SuspendedRendering => "the DLSS launch buffer resumes dynamic rendering suspended in the buffer before it; not holding (nothing may be submitted between the two), frames go to DLSS untouched",
            Self::NotSplittable => "the colour input is not an RGBA16F image in GENERAL; only the hold inside DLSS's buffer takes it",
            Self::HeldInside => "this identification's holds run inside DLSS's buffer; every buffer is held there, so the hold keeps one set of resources and the network its history",
            Self::RecordedBefore => "the DLSS launch buffer, or a buffer before it in the submit, was recorded before DLSS's inputs were identified, so its barriers on them were not seen; not holding it (buffers recorded since are held)",
            Self::Subrect => "DLSS is given part of its colour input (a subrect); the split works on the whole image, so only the hold inside DLSS's buffer takes such frames, on the rectangle alone",
        }
    }

    /// [`Self::why`] where the hold inside DLSS's buffer is on: the split's refusal is not what becomes of the frame,
    /// that hold takes such buffers and its own lines say whether it runs. Stellar Blade's log (#51) said "not
    /// holding, frames go to DLSS untouched" once, and was read as no hold, while every frame after it was held
    /// inside DLSS's buffer.
    pub(crate) fn why_inside(self) -> &'static str {
        macro_rules! inside {
            ($what:literal) => {
                concat!(
                    "the DLSS launch buffer ",
                    $what,
                    ", so the submit is not split in front of it; the hold inside DLSS's buffer takes such buffers (\"holding inside DLSS's command buffer\" and \"inside DLSS's buffer, last 5.0 s\" say whether it holds)"
                )
            };
        }
        match self {
            Self::QueueFamilyTransfer => inside!("transfers queue-family ownership of a DLSS input before its launch"),
            Self::SameLayoutBarrier => inside!("has a same-layout barrier on a DLSS input before its launch"),
            Self::LayoutTransition => inside!("transitions a DLSS input's layout before its launch"),
            Self::MemoryBarrier => inside!("has a global memory barrier with a write in its source access before its launch"),
            Self::EventWait => inside!("waits on an event before its launch"),
            Self::SuspendedRendering => inside!("resumes dynamic rendering suspended in the buffer before it"),
            Self::NotSplittable | Self::HeldInside | Self::Subrect => self.why(),
            // Not "takes such buffers": the hold inside refuses a DLSS buffer begun before the identification
            // ([`InlineRefusal::BegunBefore`]); only one begun since, behind an earlier buffer begun before, is taken.
            Self::RecordedBefore => "the DLSS launch buffer, or a buffer before it in the submit, was recorded before DLSS's inputs were identified, so its barriers on them were not seen and the submit is not split in front of it; the hold inside DLSS's buffer takes it only where DLSS's own buffer was begun since (\"holding inside DLSS's command buffer\" says when it holds), and buffers recorded since are held",
        }
    }
}

/// A command buffer's dynamic-rendering suspend/resume state ([`Tracker::rendering`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rendering {
    /// A `VK_RENDERING_RESUMING_BIT` render pass instance whose suspended half is not in this
    /// buffer: it resumes one from the buffer before it in submission order.
    open_resume: bool,
    /// The last flagged render pass instance recorded has `VK_RENDERING_SUSPENDING_BIT`.
    last_suspending: bool,
}

/// DLSS's output as its evaluates write it ([`Tracker::dlss_output`]): the output image and the inputs of the same
/// evaluate, from NGX's record, and how many of the evaluate's launches name the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DlssOutput {
    pub image: vk::Image,
    pub input: vk::Image,
    pub depth: vk::Image,
    pub mvec: vk::Image,
    /// The evaluate is Ray Reconstruction's.
    pub rr: bool,
    /// How many of the evaluate's launches (those between its record and its end, [`crate::ngx_ring::ended`]) name
    /// the output: DLSS writes it in that many parts (Cyberpunk 2077's Ray Reconstruction: 12), so the hold goes after
    /// the last ([`Tracker::launch_kernel`]).
    pub writes: u32,
}

/// Whether two evaluates' outputs are the same output: an image of the same extent and format from an input of the
/// same extent with motion vectors of the same extent and format, in as many launches, Ray Reconstruction's or not
/// (the handles may alternate frame by frame: Hogwarts Legacy's Ray Reconstruction).
fn same_output(a: &DlssOutput, b: &DlssOutput, images: &HashMap<vk::Image, ImageDesc>) -> bool {
    let kind = |i: &vk::Image| images.get(i).map(|d| (d.width, d.height, d.format));
    let extent = |i: &vk::Image| images.get(i).map(|d| (d.width, d.height));
    a.rr == b.rr
        && a.writes == b.writes
        && kind(&a.image).is_some()
        && kind(&a.image) == kind(&b.image)
        && extent(&a.input) == extent(&b.input)
        && kind(&a.mvec) == kind(&b.mvec)
}

/// Holds after DLSS's output that could not be had in a row before the output counts as not learnable
/// ([`Tracker::after_missed`]).
const AFTER_MISSES: u32 = 8;

/// How long an output that counted as not learnable stays so before the hold after it is tried again: a
/// layout first seen a few frames late, or a game still loading, does not keep the passes away for good.
const AFTER_RETRY: Duration = Duration::from_secs(5);

/// The identification ([`Tracker::refresh`]): the shapes of DLSS's colour input, motion vectors and exposure image,
/// which [`Tracker::watched_by`] watches every image of. A new frame of the same evaluate on another image of the same
/// shapes is no new identification (a game alternating DLSS's input between images: Remnant II, Expedition 33; GTA V
/// Enhanced's rotating motion vectors and exposure; 007 First Light handing DLSS brand-new colour and motion-vector
/// images every 0.4 s after a load, whose holds were refused for a frame or two at each, #49); each buffer is held
/// with its own evaluate's images ([`Tracker::scan`]).
type Key = (ColourWatch, ColourWatch, Option<ColourWatch>);

/// The most identifications [`Tracker::seen_keys`] keeps.
const SEEN_KEYS: usize = 64;

/// Per-device tracking state. Every hook that feeds it only does map operations under the lock.
#[derive(Default)]
pub(crate) struct Tracker {
    /// DLSS's output as its last evaluate wrote it ([`DlssOutput`]).
    dlss_output: Option<DlssOutput>,
    /// The launch just recorded is the last of its evaluate to write DLSS's output, and passes run after DLSS
    /// ([`placement`]): the hold goes right after it, for those passes (taken by [`Tracking::launch_with_after`]).
    after_at: Option<(vk::CommandBuffer, DlssOutput, placement::Plan)>,
    /// Holds after DLSS's output missed in a row, and for which output ([`Self::after_missed`]).
    after_misses: (Option<DlssOutput>, u32),
    /// The output that counted as not learnable, and until when ([`AFTER_RETRY`]).
    after_refused: Option<(DlssOutput, Instant)>,
    /// Outputs whose refusal was logged (once each).
    after_refused_said: Vec<DlssOutput>,
    /// Output images whose rectangle (`DLSS.Output.Subrect.Base` at the feature's output size) did not fit them, logged
    /// once each ([`Self::after_point_at`]).
    after_crop_said: Vec<vk::Image>,
    images: HashMap<vk::Image, ImageDesc>,
    views: HashMap<vk::ImageView, vk::Image>,
    /// Views registered through `vkGetImageViewHandle*NVX`/`vkGetImageViewAddressNVX`, and their image.
    registered: HashMap<vk::ImageView, vk::Image>,
    /// The identification: DLSS's inputs as the newest Super Resolution evaluate on this device names them.
    inputs: Option<Inputs>,
    key: Option<Key>,
    /// The identifications made since their images were created ([`Tracker::refresh`]): a game that rotates DLSS's
    /// motion vectors, depth or exposure image frame by frame (GTA V Enhanced: four pairs and as many exposure
    /// images, 2026-11-11) comes back to one of these, which is no new identification; at most [`SEEN_KEYS`].
    seen_keys: Vec<Key>,
    /// What the last identification log line said, so a change is logged once.
    announced: Option<String>,
    /// Command buffers with a `vkCmdCuLaunchKernelNVX` recorded since their last begin (directly or
    /// through executed secondaries), with what their launches' parameters referenced.
    launch: HashMap<vk::CommandBuffer, LaunchRefs>,
    /// The values a view registration returned (`vkGetImageViewHandle*NVX` handles,
    /// `vkGetImageViewAddressNVX` addresses), for every registered view: what a kernel parameter
    /// buffer that reads one of them carries.
    keys: Vec<(u64, vk::ImageView, vk::Image)>,
    /// Launch-bearing submits held at a buffer with DLSS's Super Resolution evaluate, and those forwarded (no
    /// buffer of theirs carries one: DLSS Frame Generation's, Ray Reconstruction's).
    pub(crate) evaluate_submits: u64,
    pub(crate) foreign_submits: u64,
    /// Which first-of-a-kind classification lines were logged.
    said_kinds: u8,
    /// Barriers recorded into each command buffer for the watched images, in recording order,
    /// awaiting submission.
    pending: HashMap<vk::CommandBuffer, Vec<ImageSync>>,
    /// Image copies recorded into each command buffer whose destination is a watched image, in recording order:
    /// (source, destination). A hold in front of DLSS's buffer reads the source of a copy that buffer makes into
    /// an input before its launch ([`Self::copy_source`]).
    copies: HashMap<vk::CommandBuffer, Vec<(vk::Image, vk::Image)>>,
    /// The watched images' layouts as of the last successful submission (submission order, not
    /// recording order): only [`Self::commit_submitted`] writes it, for command buffers the next
    /// layer accepted.
    committed: HashMap<vk::Image, vk::ImageLayout>,
    /// The queue family the last successfully submitted ownership transfer of a watched image
    /// named as its destination.
    owners: HashMap<vk::Image, u32>,
    /// The first global synchronization recorded into each command buffer while an input was
    /// watched ([`Hazard::MemoryBarrier`], [`Hazard::EventWait`]).
    global_sync: HashMap<vk::CommandBuffer, Hazard>,
    /// The command buffers that wait on an event (`vkCmdWaitEvents*`) while an input was watched, whatever
    /// synchronization came first in them: an event the host sets after the submission returns.
    event_waits: HashSet<vk::CommandBuffer>,
    /// The launch just recorded was the first of its command buffer to name the colour input of the Super
    /// Resolution evaluate it belongs to: the buffer and its record ([`Self::inline_point`] decides on a hold inside
    /// the buffer there).
    inline_at: Option<(vk::CommandBuffer, crate::ngx_ring::Record)>,
    /// Command buffers with a suspending or resuming dynamic render pass instance.
    rendering: HashMap<vk::CommandBuffer, Rendering>,
    /// Launch-bearing submits held at a buffer with an evaluate.
    evaluations: u64,
    /// The inputs changed in [`Self::refresh`] and no launch-bearing submit has been scanned since
    /// ([`Scan::identified_now`]).
    identified: bool,
    /// Counts identifications: bumped whenever [`Self::refresh`] changes the inputs
    /// ([`Scan::identification`]).
    generation: u64,
    /// The CUDA kernels' names (`vkCreateCuFunctionNVX`), for the probes.
    functions: HashMap<vk::CuFunctionNVX, String>,
    /// Launch-bearing submits seen by [`Self::observe`].
    launch_submits: u64,
    /// The newest upscaling evaluate on this device is Ray Reconstruction's ([`Self::rr_running`]), as of the last
    /// [`Self::refresh`].
    rr: bool,
    /// Registered images whose last registered view was destroyed. They stay registered until the image itself is
    /// destroyed: Black Myth: Wukong's benchmark with DLSS Frame Generation on destroys and re-creates the views it
    /// hands DLSS every frame (about 65 a second, the images never), and its launches still name the images.
    kept: HashSet<vk::Image>,
    /// Watched images attached in `vkCmdBeginRendering`, per command buffer, with their layout
    /// ([`Self::attachments`]).
    attached: HashMap<vk::CommandBuffer, Vec<(vk::Image, vk::ImageLayout)>>,
    /// Command buffers whose whole recording so far happened under the current identification: begun
    /// while these inputs were watched ([`Self::begin`]), with every secondary executed into them so far
    /// the same. Only for these are [`Self::pending`], [`Self::global_sync`] and [`Self::attached`]
    /// everything recorded on the watched images: the device hooks record barriers only while inputs
    /// are watched, and a new identification drops what was recorded ([`Self::refresh`], which empties
    /// this set too). A buffer recorded before (a re-submitted one, or one recorded ahead on another
    /// thread) may carry barriers on the inputs that were never seen ([`Hazard::RecordedBefore`]).
    recorded_watched: HashSet<vk::CommandBuffer>,
    /// Diagnostics: barriers seen on each colour image of the watched shape ([`Self::colour_watch`]) and on the
    /// identified colour input since the current identification, with the last one's new layout.
    colour_barriers: HashMap<vk::Image, (u64, vk::ImageLayout)>,
    /// The identified colour input's shape (extent and format), kept while nothing is identified: barriers on every
    /// image of that shape are recorded ([`Self::watched`]), so each image a game hands DLSS in turn has a known
    /// layout, and what was recorded on one carries across a new identification that watches it too
    /// ([`Self::keep_watched_across`]). Expedition 33 (Unreal Engine 5, #51) cycles DLSS's input through several
    /// 1488x836 B10G11R11 images.
    colour_watch: Option<ColourWatch>,
    /// Hold-inside refusals ordinary at a new identification ([`InlineRefusal::quiet`]): counted, logged with the
    /// counts at most every [`QUIET_REFUSALS_EVERY`] ([`Self::quiet_refusal_line`]), when last logged.
    quiet_refusals: (Option<Instant>, [u64; 2]),
    /// Command buffers in which a hold inside was decided on ([`Tracking::launch`] ran [`Self::inline_point`] for
    /// them), until they are begun again: a launch buffer the split refuses without one says why
    /// ([`Self::untried_line`]).
    inline_tried: HashSet<vk::CommandBuffer>,
    said_untried: u8,
    /// The identification ([`Self::generation`]) and the kinds of [`InlineRefusal`] logged for it.
    said_refusals: (u64, u16),
    /// Each colour image's layout at DLSS's launch buffer in the last submission held with it ([`Self::scan`]): what
    /// the split's check reads there. Kept across identifications as the committed layouts are
    /// ([`Self::keep_watched_across`]).
    launch_layouts: HashMap<vk::Image, vk::ImageLayout>,
    /// The exposure image the evaluates on each colour input name, and whether they name more than one ([`Self::scan`]).
    exposure_names: HashMap<vk::Image, (vk::Image, bool)>,
    /// The last launch's kernel name and the registered images it named, kept only for
    /// `NEURAL_FORGE_PROBE_IMAGE` ([`probe_image`]).
    last_launch: Option<(String, Vec<vk::Image>)>,
    /// The identification ([`Self::generation`]) a hold inside DLSS's buffer was recorded for: from
    /// then on [`Self::inline_point`] answers for the buffers the split could take too
    /// ([`Hazard::HeldInside`]).
    inline_in_use: Option<u64>,
    /// The probe's parameter-layout lines already written, by kernel name and buffer size
    /// ([`Self::probe_layout`]).
    layouts_said: Vec<(String, usize)>,
}

/// Where a submit's launch-bearing command buffer with DLSS's Super Resolution evaluate is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scan {
    pub batch: usize,
    pub index: usize,
    /// The colour input's committed layout just before the launch buffer, if any barrier on it was
    /// ever seen.
    pub colour_layout: Option<vk::ImageLayout>,
    /// The buffer's evaluate's inputs, as NGX's record names them; `None` when one of its colour input, depth and
    /// motion vectors is an image the layer did not see created on this device.
    pub inputs: Option<Inputs>,
    /// The depth and motion-vector images' committed layouts at the same point.
    pub depth_layout: Option<vk::ImageLayout>,
    pub mvec_layout: Option<vk::ImageLayout>,
    /// The exposure input's committed layout.
    pub exposure_input_layout: Option<vk::ImageLayout>,
    pub evaluation: u64,
    /// The inputs were (re)identified since the last launch-bearing submit: no layout is known yet
    /// (barriers are only recorded on watched images, and a new identification drops what was
    /// committed), so a `None` layout here does not mean "no barrier ever".
    pub identified_now: bool,
    /// Which identification the inputs are ([`Tracker`]'s count of changes): a hold's auto-exposure
    /// starts its adaptation afresh when it changes ([`Target::identification`]).
    pub identification: u64,
    /// Why the submit must not be split in front of its launch buffer, if anything recorded says so.
    pub hazard: Option<Hazard>,
    /// Why a dump (which also reads depth, the motion vectors and the exposure image) must not be taken in front of
    /// the launch buffer: a barrier on one of those images only.
    pub dump_hazard: Option<Hazard>,
    /// A barrier on the depth image itself in front of the launch: the model-mode hold then does not
    /// copy the depth (it would read it before the launch buffer finished it), and marks it invalid
    /// for that hold. GTA V Enhanced with DLSS Frame Generation on writes depth there.
    pub depth_hazard: Option<Hazard>,
    /// The queue family that owns the colour input, when a submitted ownership transfer said so.
    pub colour_owner: Option<u32>,
    /// What NGX's record of the evaluate says of its frame: the camera jitter and the game's reset.
    pub frame: DlssFrame,
}

impl Scan {
    /// Whether the colour input can be taken to be in `GENERAL` at this submit: its last committed
    /// barrier left it there, or no barrier on it was seen since it was watched. Never on the
    /// submit that (re)identified the inputs ([`Self::identified_now`]): that one goes to DLSS
    /// untouched. A later submit whose launch buffer, or a buffer before it, was recorded before the
    /// identification is refused by its [`Self::hazard`] ([`Hazard::RecordedBefore`]).
    pub(crate) fn colour_in_general(&self) -> bool {
        !self.identified_now && self.colour_layout.is_none_or(|l| l == vk::ImageLayout::GENERAL)
    }
}

/// What one command buffer's `vkCmdCuLaunchKernelNVX` launches referenced.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LaunchRefs {
    /// A launch whose parameter buffer could not be read.
    pub opaque: bool,
    /// Registered images whose handle (or address) is in a launch's parameters.
    pub images: Vec<vk::Image>,
    /// The DLSS evaluate the buffer's launches belong to: NGX's record ([`crate::ngx_ring`]) of the evaluate whose
    /// colour input the buffer's first launch naming it named before the evaluate ended. Every later launch of the
    /// buffer until the evaluate's end ([`Self::ended`]) is the evaluate's.
    pub ngx: Option<crate::ngx_ring::Record>,
    /// The evaluate's end record was read: the buffer's later launches are not its.
    pub ended: bool,
    /// What the buffer held when its first launch was recorded.
    pub before_first: Option<Before>,
    /// The same for the first launch naming each image of [`Self::images`].
    pub before_named: Vec<(vk::Image, Before)>,
    /// How many of the evaluate's launches named its output ([`DlssOutput::writes`]).
    pub output_writes: u32,
    /// A launch of DLSS Super Resolution's or Ray Reconstruction's kernels ([`dlss_game::dlss_kernel_name`]): with no
    /// NGX record, a sign that DLSS runs without the wrapper ([`crate::ngx_ring::note_launches_without_record`]).
    pub dlss_kernel: bool,
}

impl LaunchRefs {
    /// The buffer's Super Resolution evaluate, if it carries one: the hold point before the upscaler.
    pub(crate) fn sr(&self) -> Option<crate::ngx_ring::Record> {
        self.ngx.filter(|r| !r.is_rr())
    }
}

/// What NGX's record of an evaluate says of the frame a hold belongs to: the camera jitter (render pixels, x and y)
/// and whether the game reset DLSS's history.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DlssFrame {
    pub jitter: Option<[f32; 2]>,
    pub reset: bool,
}

impl DlssFrame {
    /// `record`'s jitter (where finite) and reset.
    pub(crate) fn of(record: &crate::ngx_ring::Record) -> Self {
        Self { jitter: record.jitter.iter().all(|v| v.is_finite()).then_some(record.jitter), reset: record.reset }
    }
}

/// The variable that switches honouring the game's history reset off (`off`), for testing.
pub(crate) const GAME_RESET_ENV: &str = "NEURAL_FORGE_GAME_RESET";

/// Whether the holds honour the game's history reset ([`GAME_RESET_ENV`]). Read once, logged when off.
pub(crate) fn game_reset_honoured() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        let raw = neural_forge_protocol::env::var(GAME_RESET_ENV);
        let on = parse_game_reset(raw.as_deref());
        if !on {
            crate::log!("[preupscale] {GAME_RESET_ENV}=off: the game's own reset of DLSS's history is not followed");
        } else if raw.as_deref().is_some_and(|v| !v.trim().is_empty() && v.trim() != "on") {
            crate::log!("[preupscale] {GAME_RESET_ENV}={raw:?} is not on or off; it stays on");
        }
        on
    });
    *ON
}

/// [`GAME_RESET_ENV`]'s value: on unless `off`.
fn parse_game_reset(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("off")
}

/// The layouts the hold inside DLSS's buffer copies the motion vectors from (moved to
/// `TRANSFER_SRC_OPTIMAL` and back when not already readable there).
/// Whether the hold inside DLSS's buffer copies the depth aspect of a `format` image found in `layout`
/// (moved to TRANSFER_SRC_OPTIMAL and back unless it is there or in GENERAL). The separate depth
/// layouts only for formats without stencil: with one, the stencil aspect's layout is not known.
pub(crate) fn depth_copy_layout(format: vk::Format, layout: vk::ImageLayout) -> bool {
    use vk::ImageLayout as L;
    match layout {
        L::GENERAL | L::TRANSFER_SRC_OPTIMAL | L::SHADER_READ_ONLY_OPTIMAL => true,
        L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL | L::DEPTH_STENCIL_READ_ONLY_OPTIMAL => DEPTH_FORMATS.contains(&format),
        L::DEPTH_ATTACHMENT_OPTIMAL | L::DEPTH_READ_ONLY_OPTIMAL => DEPTH_FORMATS.contains(&format) && !has_stencil(format),
        L::COLOR_ATTACHMENT_OPTIMAL => DEPTH_AS_COLOUR.contains(&format),
        _ => false,
    }
}

pub(crate) const MVEC_COPY_LAYOUTS: [vk::ImageLayout; 4] =
    [vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL];

/// What a command buffer held at the point a launch was recorded into it: how many of its
/// barriers on watched images come before the launch ([`Tracker`]'s `pending` list is in recording
/// order), the global synchronization recorded by then, and how many of its copies into watched
/// images ([`Tracker`]'s `copies`, in recording order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Before {
    pub barriers: usize,
    pub global: Option<Hazard>,
    pub copies: usize,
}

/// Why [`Tracker::inline_point`] records no hold inside DLSS's buffer at a launch naming the colour input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlineRefusal {
    /// Nothing identified.
    NotIdentified,
    /// The buffer was begun before this identification (or it changed with this submit): what its earlier
    /// commands did to the inputs is not known.
    BegunBefore,
    /// The colour image is not one the layer saw created.
    UnknownImage,
    /// Not RGBA16F or B10G11R11.
    Format(vk::Format),
    /// Not a 2D single-sample image.
    NotPlain,
    /// Without both transfer usages: the copies in and back need them (the layer adds them at creation;
    /// an image created before the layer was loaded, or one whose memory is shared, keeps its own).
    Usage(vk::ImageUsageFlags),
    /// Not the identified colour input's extent.
    Extent((u32, u32), (u32, u32)),
    /// A dynamic render pass instance is suspended across the launch, or resumed in the buffer.
    SuspendedRendering,
    /// The colour input's layout at the launch is not known.
    LayoutUnknown,
    /// A layout the copies in and back do not take.
    Layout(vk::ImageLayout),
    /// The buffer itself left the colour input in `GENERAL` and nothing in it keeps the split from it: the
    /// split hold takes the frame at the submit.
    LeftToSplit,
    /// DLSS's render rectangle (its subrect) does not fit the colour input: DLSS would read outside it.
    SubrectOutside(crate::ngx_ring::Rect, (u32, u32)),
}

impl InlineRefusal {
    /// Refused for what happens at every new identification: counted and logged with the counts
    /// ([`Tracker::quiet_refusal_line`]), not once per identification (a game that is identified again and again
    /// would log it every time). Before, these said nothing at all, and a game whose every buffer was refused so
    /// (#51) showed no reason.
    fn quiet(&self) -> bool {
        matches!(self, Self::NotIdentified | Self::BegunBefore)
    }

    /// What the log and the status say.
    fn said(&self) -> String {
        match self {
            Self::NotIdentified => "nothing is identified".into(),
            Self::BegunBefore => {
                "DLSS's buffer was begun before DLSS's inputs were identified (or identified again), so what it did to them before is not known".into()
            }
            Self::UnknownImage => "the colour input is an image the layer did not see created".into(),
            Self::Format(f) => format!("the colour input is {f:?} (RGBA16F or B10G11R11 only)"),
            Self::NotPlain => "the colour input is not a 2D single-sample image".into(),
            Self::Usage(u) => format!(
                "the colour input's usage {u:?} lacks TRANSFER_SRC and TRANSFER_DST, which the copies in and back need (the layer adds them to the images it sees created, unless their memory is shared)"
            ),
            Self::Extent(have, want) => format!("the colour input is {}x{}, the identified one {}x{}", have.0, have.1, want.0, want.1),
            Self::SuspendedRendering => "a render pass is suspended across DLSS's launch in its buffer".into(),
            Self::LayoutUnknown => "the colour input's layout at DLSS's launch is unknown".into(),
            Self::Layout(l) => format!("the colour input is in {l:?} at DLSS's launch (the copies take GENERAL, SHADER_READ_ONLY, COLOR_ATTACHMENT or TRANSFER layouts)"),
            Self::LeftToSplit => "the buffer leaves the colour input in GENERAL with nothing in the way: the hold splits the submit instead".into(),
            Self::SubrectOutside(r, (w, h)) => {
                format!("DLSS's render rectangle {}x{} at ({}, {}) does not fit its {w}x{h} colour input", r.size.0, r.size.1, r.at.0, r.at.1)
            }
        }
    }

    /// The kind, for saying each once per identification.
    fn kind(&self) -> u8 {
        match self {
            Self::NotIdentified => 0,
            Self::BegunBefore => 1,
            Self::UnknownImage => 2,
            Self::Format(_) => 3,
            Self::NotPlain => 4,
            Self::Usage(_) => 5,
            Self::Extent(..) => 6,
            Self::SuspendedRendering => 7,
            Self::LayoutUnknown => 8,
            Self::Layout(_) => 9,
            Self::LeftToSplit => 10,
            Self::SubrectOutside(..) => 11,
        }
    }
}

/// Why the last frame DLSS ran with was not held, for the status ([`Session::publish_status`]): a reason the split
/// hold or the hold inside DLSS's buffer gave.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Refusal {
    Text(std::borrow::Cow<'static, str>),
    Inside(InlineRefusal),
}

impl Refusal {
    fn text(&self) -> String {
        match self {
            Self::Text(t) => t.to_string(),
            Self::Inside(r) => format!("no hold inside DLSS's buffer: {}", r.said()),
        }
    }
}

/// The process's last [`Refusal`] (any device's; the status describes the game, as [`LAST_HELD`] does).
#[cfg(not(test))]
static LAST_REFUSAL: Mutex<Option<Refusal>> = Mutex::new(None);

#[cfg(test)]
thread_local! {
    /// In tests, each thread keeps its own last refusal: the tests running in parallel do not see each other's.
    static LAST_REFUSAL: std::cell::RefCell<Option<Refusal>> = const { std::cell::RefCell::new(None) };
}

/// `f` on the last [`Refusal`] ([`LAST_REFUSAL`]).
fn with_last_refusal<R>(f: impl FnOnce(&mut Option<Refusal>) -> R) -> R {
    #[cfg(not(test))]
    {
        f(&mut lock(&LAST_REFUSAL))
    }
    #[cfg(test)]
    {
        LAST_REFUSAL.with_borrow_mut(f)
    }
}

/// Notes why a frame was not held, for the status: it replaces any older reason.
pub(crate) fn note_refusal(refusal: Refusal) {
    with_last_refusal(|last| {
        if last.as_ref() != Some(&refusal) {
            *last = Some(refusal);
        }
    });
}

/// Why no hold runs although nothing refused one: the model is switched off (F11, the app, `shmctl`).
pub(crate) const MODEL_OFF: &str = "the model is switched off (F11 or the app)";
/// Why no hold runs before the upscaler although nothing refused one: the Placement setting puts every pass after
/// the upscaler (at present, or after DLSS once its output is learnt; the state is then 5 only while no hold after
/// DLSS runs either).
pub(crate) const NO_PASS_BEFORE: &str = "the Placement setting runs no pass before the upscaler";
/// Why no hold runs: the VRAM guard paused the network (the device at its VRAM budget, the network spilled to system
/// memory); it resumes once there is room again (`preupscale::vram_guard`).
pub(crate) const VRAM_PAUSED: &str = "the network is paused: too little video memory for it (VRAM guard); it resumes once there is room";
/// Why no hold runs: the device cannot run the network (both holds give the same reason).
pub(crate) const NATIVE_UNAVAILABLE: &str = "the native backend is not available on this device; frames go to DLSS untouched";

/// Notes one of the reasons above (or another fixed one) as why no hold runs, replacing an older refusal so the
/// status never names one that no longer applies.
pub(crate) fn note_not_held(why: &'static str) {
    note_refusal(Refusal::Text(why.into()));
}

/// The last reason noted ([`note_refusal`]) on this test's thread, as the status says it.
#[cfg(test)]
pub(crate) fn last_refusal_text() -> Option<String> {
    with_last_refusal(|last| last.as_ref().map(Refusal::text))
}

/// Whether `plan` runs a pass before the upscaler; if not, notes [`NO_PASS_BEFORE`] for the status. Both holds
/// before the upscaler (the split one and the one inside DLSS's buffer) ask it.
fn plans_before(plan: &placement::Plan) -> bool {
    let before = plan.has(placement::Stage::Before);
    if !before {
        note_not_held(NO_PASS_BEFORE);
    }
    before
}

/// Holds that ran, for the tally line ([`Tracking::scan`]): split holds that wrote back, and holds recorded
/// inside DLSS's buffers that ran (each run of a buffer counts).
static HOLDS_SPLIT: AtomicU64 = AtomicU64::new(0);
static HOLDS_INSIDE: AtomicU64 = AtomicU64::new(0);

/// A split hold wrote its frame back.
pub(crate) fn note_held_split() {
    HOLDS_SPLIT.fetch_add(1, Ordering::Relaxed);
}

/// Buffers with a hold inside them ran `runs` times.
pub(crate) fn note_held_inside(runs: u32) {
    HOLDS_INSIDE.fetch_add(u64::from(runs), Ordering::Relaxed);
}

/// Where a hold runs inside DLSS's own command buffer ([`Tracker::inline_point`],
/// [`inline`]): the colour input the buffer's first colour launch names, DLSS's exposure input as
/// the encode reads it, which identification they belong to, and what refused the split.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InlinePoint {
    pub colour: vk::Image,
    /// The colour input's description, its extent the rectangle the hold works on: the whole image, or the part DLSS
    /// is given ([`Self::offsets`]).
    pub desc: ImageDesc,
    /// The colour input's whole extent (the output's, after DLSS): the hold's staging images and network resources
    /// are this size, so a rectangle ([`Self::desc`]) that changes size inside the same image builds nothing again.
    pub image: (u32, u32),
    /// Where the hold's rectangles start in the colour input (its copy in and back), the motion vectors and the depth:
    /// DLSS's subrect ([`crate::ngx_ring::Crop`]); zero for the whole images.
    pub offsets: Offsets,
    /// The colour input's layout at the hold (restored after it).
    pub layout: vk::ImageLayout,
    pub exposure_input: Option<Aux>,
    /// DLSS's motion vectors at the hold (the native backend's history), where they can be copied.
    pub mvec_input: Option<Aux>,
    /// Their extent: the colour input's, or the output's where DLSS is given high-resolution motion
    /// vectors (the `mvhi` input kernels; Hogwarts Legacy), which the copy resamples to the colour
    /// input's ([`Self::mvec_scale`]).
    pub mvec_extent: (u32, u32),
    /// What turns the motion vectors into the colour input's pixels, from NGX's `MV.Scale`
    /// ([`crate::ngx_ring::Record::mv_units`]); [`Self::mvec_scale`] falls back to the extents without one.
    pub mv_units: Option<[f32; 2]>,
    /// DLSS's depth at the hold, where its depth aspect can be copied (the native backend's depth
    /// copy): at the colour input's extent, in a layout [`depth_copy_layout`] takes.
    pub depth_input: Option<Aux>,
    pub identification: u64,
    pub hazard: Hazard,
    /// The stage this hold runs at and which passes ([`placement::Hold`]).
    pub hold: placement::Hold,
    /// The inputs whose layout at the hold is a guess, not something this buffer set: checked when the
    /// buffer is submitted ([`Tracker::layout_when_run`], `inline::Inline::check`).
    pub guessed: Guessed,
}

/// Where a hold's rectangles start in the images it copies ([`InlinePoint::offsets`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Offsets {
    pub colour: (u32, u32),
    pub mvec: (u32, u32),
    pub depth: (u32, u32),
}

/// The layouts [`Tracker::inline_point`] took for the hold's inputs without this buffer setting them (no barrier
/// on the image and no attachment use in it before the hold): from the layouts the buffers submitted before it
/// left (as known when this one is recorded), or `GENERAL` for a storage image never seen in a barrier.
/// vkd3d-proton records several command lists in parallel and submits them later, so a buffer submitted in
/// between can leave another layout. Each `Some` is the layout guessed for that input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Guessed {
    pub colour: Option<vk::ImageLayout>,
    pub exposure: Option<vk::ImageLayout>,
    pub mvec: Option<vk::ImageLayout>,
    pub depth: Option<vk::ImageLayout>,
}

impl InlinePoint {
    /// What turns DLSS's motion vectors into the colour input's pixels, per axis: they are in the
    /// pixels of their own image (measured on Hogwarts Legacy's 2560x1440 ones: consecutive frames
    /// line up at 0.667 of their value, 1708 over 2560), so 1 at the colour input's extent and the
    /// colour input's over theirs for high-resolution ones.
    pub(crate) fn mvec_scale(&self) -> [f32; 2] {
        if let Some(scale) = self.mv_units {
            return scale;
        }
        let (w, h) = self.mvec_extent;
        if w == 0 || h == 0 {
            return [1.0, 1.0];
        }
        [self.desc.width as f32 / w as f32, self.desc.height as f32 / h as f32]
    }
}

/// The kernel parameter buffer of a `vkCmdCuLaunchKernelNVX` given in CUDA's "extra" form, the form
/// vkd3d-proton uses for DXVK-NVAPI's `NvAPI_D3D12_LaunchCubinShader` (DLSS under Proton; the probe
/// saw `params=0 extras=1` on every launch): `pExtras` is the END-terminated list
/// `{BUFFER_POINTER (1), buffer, BUFFER_SIZE (2), &size, END (0)}`. vkd3d-proton passes
/// `extraCount = 1`, CUDA's convention being the terminator, so the list is walked to its END (at
/// most 4 pairs; a larger `extraCount` also bounds the walk). The size is read as 32 bits (vkd3d-proton's is a 32-bit value; on little-endian
/// that is also the low half of a `size_t`). `None` for anything else: no extras, an unknown key,
/// no buffer, or a size of 0 or over 4 KiB. The buffer is only read, never changed, and only during
/// the application's call.
///
/// # Safety
/// `extras`, when non-null, must point at a CUDA launch-parameter list as described, valid for
/// the call (the application's own `VkCuLaunchInfoNVX::pExtras`).
pub(crate) unsafe fn launch_params<'a>(extras: *const *const c_void, extra_count: usize) -> Option<&'a [u8]> {
    if extras.is_null() || extra_count == 0 {
        return None;
    }
    let (mut buffer, mut size) = (std::ptr::null::<u8>(), None);
    let mut i = 0;
    loop {
        if i >= 8 {
            return None;
        }
        // An explicit count larger than CUDA's terminator convention ends the list there.
        if extra_count > 1 && i >= extra_count {
            break;
        }
        // SAFETY: the list is END-terminated (caller); every read is at or before the terminator
        // or the value that follows a key.
        let key = unsafe { *extras.add(i) } as usize;
        // A key's value is the next entry: past an explicit count it is not part of the list.
        if key != 0 && extra_count > 1 && i + 1 >= extra_count {
            return None;
        }
        match key {
            0 => break,
            // SAFETY: as above.
            1 => buffer = unsafe { *extras.add(i + 1) }.cast::<u8>(),
            2 => {
                // SAFETY: as above.
                let p = unsafe { *extras.add(i + 1) }.cast::<u32>();
                if p.is_null() {
                    return None;
                }
                // SAFETY: BUFFER_SIZE's value points at the size.
                size = Some(unsafe { p.read_unaligned() } as usize);
            }
            _ => return None,
        }
        i += 2;
    }
    let size = size?;
    if buffer.is_null() || size == 0 || size > 4096 {
        return None;
    }
    // SAFETY: BUFFER_POINTER points at `size` bytes of parameters, valid for the call.
    Some(unsafe { std::slice::from_raw_parts(buffer, size) })
}

impl Tracker {
    pub(crate) fn record_image(&mut self, image: vk::Image, info: &vk::ImageCreateInfo) {
        self.images.insert(image, ImageDesc::from_info(info));
    }

    pub(crate) fn forget_image(&mut self, image: vk::Image) {
        self.images.remove(&image);
        self.keys.retain(|k| k.2 != image);
        self.registered.retain(|_, i| *i != image);
        self.kept.remove(&image);
        self.committed.remove(&image);
        self.owners.remove(&image);
        self.launch_layouts.remove(&image);
        self.colour_barriers.remove(&image);
        self.exposure_names.retain(|c, (e, _)| *c != image && *e != image);
    }

    pub(crate) fn record_view(&mut self, view: vk::ImageView, image: vk::Image) {
        self.views.insert(view, image);
    }

    pub(crate) fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view);
        self.keys.retain(|k| k.1 != view);
        // The image stays registered ([`Self::kept`]): its handles stay what launches name.
        if let Some(image) = self.registered.remove(&view) {
            if !self.registered.values().any(|i| *i == image) {
                self.kept.insert(image);
            }
        }
    }

    /// A view registered through `vkGetImageViewHandle*NVX` / `vkGetImageViewAddressNVX`; `key` is
    /// the handle or address it returned (what a kernel that reads the view is given).
    pub(crate) fn register(&mut self, view: vk::ImageView, key: Option<u64>) {
        if let Some(&image) = self.views.get(&view) {
            self.kept.remove(&image);
            if let Some(old) = self.registered.insert(view, image).filter(|&old| old != image) {
                // A view handle reused for another image: the old image keeps its registration.
                if !self.registered.values().any(|i| *i == old) {
                    self.kept.insert(old);
                }
            }
            if let Some(key) = key.filter(|&k| k != 0) {
                if !self.keys.iter().any(|k| k.0 == key) {
                    self.keys.push((key, view, image));
                }
            }
        }
    }

    /// `vkCreateCuFunctionNVX` returned `function` for the kernel `name`.
    pub(crate) fn record_function(&mut self, function: vk::CuFunctionNVX, name: &str) {
        self.functions.insert(function, name.to_string());
    }

    pub(crate) fn forget_function(&mut self, function: vk::CuFunctionNVX) {
        self.functions.remove(&function);
    }

    /// Whether DLSS Ray Reconstruction runs: the newest upscaling evaluate on this device is Ray Reconstruction's (NGX's
    /// feature 13). Then nothing is identified, so nothing is held: the model must not run on Ray Reconstruction's
    /// noisy input, and runs after the upscaler instead ([`ray_reconstruction_running`]).
    fn rr_running(&self) -> bool {
        self.rr
    }

    /// [`Self::launch_kernel`] without the kernel (its name unknown).
    #[cfg(test)]
    pub(crate) fn launch(&mut self, command_buffer: vk::CommandBuffer, params: Option<&[u8]>) {
        self.launch_kernel(command_buffer, vk::CuFunctionNVX::null(), params);
    }

    /// `vkCmdCuLaunchKernelNVX` of `function` recorded into `command_buffer`; `params` is the launch's kernel parameter
    /// buffer when it could be read ([`launch_params`]), `None` otherwise. Every 8-byte word of it is compared with the
    /// registered views' handles and addresses. The buffer's first launch naming the colour input of a DLSS evaluate
    /// that has not ended ([`crate::ngx_ring::evaluate_naming`]) ties the buffer to that evaluate; for Super Resolution
    /// it is the hold point before the upscaler ([`Self::inline_at`]). The evaluate's later launches in the buffer, up
    /// to its end, count the writes of its output: the last of as many as its last evaluate made
    /// ([`Self::dlss_output`]) is the hold point after DLSS ([`Self::after_at`]).
    pub(crate) fn launch_kernel(&mut self, command_buffer: vk::CommandBuffer, function: vk::CuFunctionNVX, params: Option<&[u8]>) {
        let before = Before {
            barriers: self.pending.get(&command_buffer).map_or(0, Vec::len),
            global: self.global_sync.get(&command_buffer).copied(),
            copies: self.copies.get(&command_buffer).map_or(0, Vec::len),
        };
        let dlss_kernel = self.functions.get(&function).is_some_and(|n| dlss_game::dlss_kernel_name(n));
        let entry = self.launch.entry(command_buffer).or_default();
        entry.before_first.get_or_insert(before);
        entry.dlss_kernel |= dlss_kernel;
        let Some(bytes) = params else {
            self.launch.entry(command_buffer).or_default().opaque = true;
            return;
        };
        if let Some(name) = self.functions.get(&function) {
            probe_kernel(name, bytes);
        }
        let mut named: Vec<vk::Image> = Vec::new();
        for word in bytes.as_chunks::<8>().0.iter() {
            let value = u64::from_le_bytes(*word);
            if value == 0 {
                continue;
            }
            // The whole word, then its two 32-bit halves in memory order: Unreal Engine 5's DLSS
            // (Black Myth: Wukong's benchmark) packs two 32-bit view handles per word
            // (`0x3201c23_01401c00`), where GTA V's and Crimson Desert's give each its own.
            let halves = [value & 0xffff_ffff, value >> 32];
            let packed = value >> 32 != 0 && halves.iter().all(|&h| h != 0);
            for candidate in std::iter::once(value).chain(halves.into_iter().filter(|_| packed)) {
                for &(key, _, image) in &self.keys {
                    if key == candidate && !named.contains(&image) {
                        named.push(image);
                    }
                }
            }
        }
        if probe_image::enabled() {
            self.last_launch = self.functions.get(&function).map(|n| (n.clone(), named.clone()));
        }
        let refs = self.launch.entry(command_buffer).or_default();
        // The evaluate's end read since its last launch: the buffer's launches from here on are not its.
        if let Some(r) = refs.ngx.filter(|_| !refs.ended) {
            refs.ended = crate::ngx_ring::ended(r.seq);
        }
        let mut joined = None;
        if refs.ngx.is_none() && !named.is_empty() {
            if let Some(record) = crate::ngx_ring::evaluate_naming(&named) {
                refs.ngx = Some(record);
                joined = Some(record);
            }
        }
        let open = refs.ngx.filter(|_| !refs.ended);
        let write = open.filter(|r| r.output != vk::Image::null() && named.contains(&r.output)).map(|r| {
            refs.output_writes += 1;
            DlssOutput { image: r.output, input: r.colour, depth: r.depth, mvec: r.mvec, rr: r.is_rr(), writes: refs.output_writes }
        });
        for image in named {
            if !refs.images.contains(&image) {
                refs.images.push(image);
                refs.before_named.push((image, before));
            }
        }
        if let Some(record) = joined {
            crate::ngx_ring::note_matched(&record);
        }
        // The last write of DLSS's output as many as the evaluates before wrote it in.
        if let Some(o) = write.filter(|o| self.dlss_output.is_some_and(|l| same_output(&l, o, &self.images))) {
            let frame = self.frame();
            let plan = placement::current(placement::Frame { rr: o.rr || frame.rr, ..frame });
            if plan.has(placement::Stage::Upscaled) {
                self.after_at = Some((command_buffer, o, plan));
            }
        }
        self.inline_at = joined.filter(|r| !r.is_rr()).map(|r| (command_buffer, r));
        if crate::probe_ngx::enabled() {
            self.probe_layout(function, bytes, open.map(|r| r.colour));
        }
    }

    /// With `NEURAL_FORGE_PROBE_NGX=1`: logs once per kernel and parameter-buffer size what each
    /// 8-byte word of the launch's parameters holds (docs/DLSS_KERNEL_CATALOGUE.md is built from
    /// these lines). `colour` is the colour input of the buffer's evaluate, if any.
    fn probe_layout(&mut self, function: vk::CuFunctionNVX, bytes: &[u8], colour: Option<vk::Image>) {
        const MAX_LAYOUTS: usize = 96;
        let name = self.functions.get(&function).cloned().unwrap_or_else(|| "unnamed".to_string());
        let id = (name, bytes.len());
        if self.layouts_said.len() >= MAX_LAYOUTS || self.layouts_said.contains(&id) {
            return;
        }
        let words: Vec<String> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w))
            .enumerate()
            .map(|(i, value)| format!("{i}={}", self.describe_word(value, colour)))
            .collect();
        crate::log!("[probe-ngx] layout {} bytes={}: {}", id.0, id.1, words.join(" "));
        crate::logging::flush();
        self.layouts_said.push(id);
    }

    /// One parameter word for [`Self::probe_layout`]: a registered view (whole word, or two packed
    /// 32-bit halves) with its role, else the value read as two 32-bit halves.
    fn describe_word(&self, value: u64, colour: Option<vk::Image>) -> String {
        if value == 0 {
            return "0".into();
        }
        let view = |key: u64| -> Option<String> {
            let &(_, _, image) = self.keys.iter().find(|k| k.0 == key)?;
            let Some(d) = self.images.get(&image) else { return Some("view(unregistered)".into()) };
            let role = if Some(image) == colour {
                "colour-in"
            } else if (d.width, d.height) == (1, 1) {
                "exposure"
            } else if DEPTH_FORMATS.contains(&d.format) {
                "depth"
            } else if [vk::Format::R16G16_SFLOAT, vk::Format::R32G32_SFLOAT].contains(&d.format) {
                "mvec"
            } else {
                "image"
            };
            Some(format!("view:{role}({}x{},{:?})", d.width, d.height, d.format))
        };
        if let Some(v) = view(value) {
            return v;
        }
        let (lo, hi) = (value & 0xffff_ffff, value >> 32);
        if let (Some(a), Some(b)) = (view(lo), view(hi)) {
            return format!("[{a}|{b}]");
        }
        // Not a view: the two halves as integers when both are small, else as floats when both
        // look like ordinary floats, else raw.
        let float = |h: u64| {
            let f = f32::from_bits(h as u32);
            (h == 0 || (f.is_normal() && (1e-6..1e7).contains(&f.abs()))).then_some(f)
        };
        if lo < 1 << 20 && hi < 1 << 20 {
            format!("i({lo},{hi})")
        } else if let (Some(a), Some(b)) = (float(lo), float(hi)) {
            format!("f({a},{b})")
        } else {
            format!("{value:#x}")
        }
    }

    /// Whether a hold should run inside `cb`, right before the launch just recorded, which is the first of the buffer
    /// to name the colour input of `record`'s evaluate: the inputs are identified (not on the submit that identified
    /// them), the colour input is an RGBA16F or R11G11B10 image the layer can transfer
    /// (`TRANSFER_SRC | TRANSFER_DST`) at the identified extent, its layout at that
    /// point is known (the buffer's own barriers, else the committed state; an RGBA16F storage image
    /// no barrier was seen on is taken as `GENERAL`, as the split does), no render pass instance is
    /// suspended there, and the split in front of the buffer cannot take it: the split would be
    /// refused ([`Self::launch_hazard`]), or the colour input is not an RGBA16F image in `GENERAL`
    /// (the split reads and writes it as a storage image). A buffer the split can take (GTA V's)
    /// gets nothing.
    pub(crate) fn inline_point(&self, cb: vk::CommandBuffer, record: &crate::ngx_ring::Record) -> Result<InlinePoint, InlineRefusal> {
        let identified = self.inputs.ok_or(InlineRefusal::NotIdentified)?;
        let colour = record.colour;
        // Begun before this identification: the layouts its earlier commands left are not known.
        if self.identified || !self.recorded_watched.contains(&cb) {
            return Err(InlineRefusal::BegunBefore);
        }
        let desc = *self.images.get(&colour).ok_or(InlineRefusal::UnknownImage)?;
        let copyable = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !OUTPUT_FORMATS.contains(&desc.format) {
            return Err(InlineRefusal::Format(desc.format));
        }
        if !desc.plain {
            return Err(InlineRefusal::NotPlain);
        }
        if !desc.usage.contains(copyable) {
            return Err(InlineRefusal::Usage(desc.usage));
        }
        if (desc.width, desc.height) != (identified.colour.1.width, identified.colour.1.height) {
            return Err(InlineRefusal::Extent((desc.width, desc.height), (identified.colour.1.width, identified.colour.1.height)));
        }
        // The evaluate's own depth, motion vectors and exposure image; the identification's where its record names an
        // image the layer did not see created.
        let inputs = self.inputs_of(record).unwrap_or(identified);
        if self.rendering.get(&cb).is_some_and(|r| r.last_suspending || r.open_resume) {
            return Err(InlineRefusal::SuspendedRendering);
        }
        // DLSS given part of its inputs (NGX's subrect): the hold works on that rectangle alone, each input read at its
        // own base, and writes back only there.
        let live_mvec = self.images.get(&inputs.mvec.0).copied();
        let (named, _) = self.hold_depth(cb, inputs.depth, None, true);
        let live_depth = self.images.get(&named).copied();
        let extent = |d: ImageDesc| (d.width, d.height);
        let crop = record.crop(extent(desc), live_mvec.map(extent), live_depth.map(extent)).map_err(|_| {
            InlineRefusal::SubrectOutside(crate::ngx_ring::Rect { at: record.colour_base, size: record.render_size(extent(desc)) }, extent(desc))
        })?;
        let rgba16f = desc.format == vk::Format::R16G16B16A16_SFLOAT;
        let mut guessed = Guessed::default();
        // A layout the buffer did not set itself is a guess. The best one is what the colour input was in at DLSS's
        // launch buffer in the last submission (`launch_colour_layout`, which the split's check reads in its own
        // submission): what the buffers submitted so far left (`committed`) is the end of that submission, and
        // a game that moves the colour input out of GENERAL in a buffer in front of DLSS's and back after it
        // left both paths refusing every frame: this one leaving the frame to the split on GENERAL, the split
        // refusing it on the layout in front of the launch. Per image: a game alternating its input between
        // images (Expedition 33) has one for each.
        let learned = self.launch_layouts.get(&colour).copied();
        let (layout, set) = match self.layout_in(cb, colour, true, rgba16f && desc.usage.contains(vk::ImageUsageFlags::STORAGE)) {
            Some((l, true)) => (l, true),
            guess => learned.map(|l| (l, false)).or(guess).ok_or(InlineRefusal::LayoutUnknown)?,
        };
        if !set {
            guessed.colour = Some(layout);
        }
        if !INLINE_LAYOUTS.contains(&layout) {
            return Err(InlineRefusal::Layout(layout));
        }
        let at = Inputs { colour: (colour, desc), ..inputs };
        let hazard = match self.launch_hazard(cb, Some(&at)).0 {
            Some(h) => h,
            // The split would work on the whole image (it refuses the frame for that, [`Self::scan`]).
            None if crop.is_some() => Hazard::Subrect,
            None if rgba16f && layout == vk::ImageLayout::GENERAL && self.inline_in_use == Some(self.generation) => Hazard::HeldInside,
            None if rgba16f && layout == vk::ImageLayout::GENERAL => return Err(InlineRefusal::LeftToSplit),
            None => Hazard::NotSplittable,
        };
        // An exposure input or motion vectors destroyed since the record was written are not copied from.
        let exposure_input = inputs.exposure_input.and_then(|(image, _)| self.images.get(&image).map(|d| (image, *d))).map(|(image, d)| {
            let known = self.layout_in(cb, image, false, d.usage.contains(vk::ImageUsageFlags::STORAGE));
            guessed.exposure = known.filter(|(_, set)| !set).map(|(l, _)| l);
            Aux {
                image,
                format: d.format,
                layout: known.map(|(l, _)| l),
                readable: d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) && (d.width, d.height) == (1, 1),
            }
        });
        // DLSS's motion vectors, copied beside the colour input for the native backend's history: RG16F
        // at the colour input's extent or larger (high-resolution motion vectors, resampled to it), in
        // a layout known here (or GENERAL for a storage image). With a subrect, their own rectangle where it
        // fits their image.
        let mvec_input = live_mvec.map(|d| {
            let image = inputs.mvec.0;
            let known = self.layout_in(cb, image, true, d.usage.contains(vk::ImageUsageFlags::STORAGE));
            guessed.mvec = known.filter(|(_, set)| !set).map(|(l, _)| l);
            let layout = known.map(|(l, _)| l);
            let readable = d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
                && MVEC_HOLD_FORMATS.contains(&d.format)
                && crop.map_or(d.width >= desc.width && d.height >= desc.height, |c| {
                    c.mvec.is_some_and(|r| r.size.0 >= c.colour.size.0 && r.size.1 >= c.colour.size.1)
                })
                && layout.is_some_and(|l| MVEC_COPY_LAYOUTS.contains(&l));
            Aux { image, format: d.format, layout, readable }
        });
        let mvec_extent = match crop {
            Some(c) => c.mvec.map_or((0, 0), |r| r.size),
            None => live_mvec.map_or((0, 0), extent),
        };
        // DLSS's depth, copied beside it into a slot buffer for the native backend: at the colour
        // input's extent (with a subrect, its rectangle at its base), in a layout known here. Not when it was
        // destroyed since (Black Myth: Wukong hands DLSS a new one every frame with frame generation on).
        let depth_input = live_depth.map(|d| {
            let image = named;
            let known = self.layout_in(cb, image, true, false);
            guessed.depth = known.filter(|(_, set)| !set).map(|(l, _)| l);
            let layout = known.map(|(l, _)| l);
            let readable = d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
                && crop.map_or((d.width, d.height) == (desc.width, desc.height), |c| c.depth.is_some())
                && layout.is_some_and(|l| depth_copy_layout(d.format, l));
            Aux { image, format: d.format, layout, readable }
        });
        let whole = desc;
        let (desc, offsets) = match crop {
            Some(c) => (
                ImageDesc { width: c.colour.size.0, height: c.colour.size.1, ..desc },
                Offsets { colour: c.colour.at, mvec: c.mvec.map_or((0, 0), |r| r.at), depth: c.depth.map_or((0, 0), |r| r.at) },
            ),
            None => (desc, Offsets::default()),
        };
        Ok(InlinePoint {
            colour,
            image: extent(whole),
            desc,
            offsets,
            layout,
            exposure_input,
            mvec_input,
            mvec_extent,
            mv_units: record.mv_units(),
            depth_input,
            identification: self.generation,
            hazard,
            hold: placement::Hold::ALL_BEFORE,
            guessed,
        })
    }

    /// DLSS's inputs as `record` names them, where its colour input, depth and motion vectors are images the layer saw
    /// created on this device; its exposure image where it names one the layer saw created.
    fn inputs_of(&self, record: &crate::ngx_ring::Record) -> Option<Inputs> {
        let get = |i: vk::Image| (i != vk::Image::null()).then(|| self.images.get(&i).map(|d| (i, *d))).flatten();
        Some(Inputs { colour: get(record.colour)?, depth: get(record.depth)?, mvec: get(record.mvec)?, exposure_input: get(record.exposure) })
    }

    /// `image`'s layout at this point of `cb` and whether `cb` set it: the buffer's own last barrier on it,
    /// else (`attachments`) its last use as an attachment in `cb`; else what the buffers submitted so far left
    /// ([`Self::commit_submitted`]), else `GENERAL` when `storage_general` (a storage image no barrier was seen
    /// on), both guesses (`false`). `None` when nothing is known.
    fn layout_in(&self, cb: vk::CommandBuffer, image: vk::Image, attachments: bool, storage_general: bool) -> Option<(vk::ImageLayout, bool)> {
        let set = self
            .pending
            .get(&cb)
            .and_then(|p| p.iter().rev().filter(|s| s.image == image).find_map(ImageSync::layout_after))
            .or_else(|| attachments.then(|| self.attached.get(&cb).and_then(|a| a.iter().rev().find(|(i, _)| *i == image).map(|(_, l)| *l))).flatten());
        set.map(|l| (l, true)).or_else(|| self.committed.get(&image).map(|l| (*l, false))).or(storage_general.then_some((vk::ImageLayout::GENERAL, false)))
    }

    /// `image`'s layout when a buffer submitted right after `earlier` (the buffers in front of it in the same
    /// submission, in order) starts: what the last of them to set it left, else what the buffers submitted
    /// before left. `None` when nothing is known (an image not watched, or never seen in a barrier).
    pub(crate) fn layout_when_run(&self, earlier: &[vk::CommandBuffer], image: vk::Image) -> Option<vk::ImageLayout> {
        earlier
            .iter()
            .rev()
            .find_map(|cb| {
                // A buffer's barriers follow its attachments ([`Self::commit_submitted`]).
                self.pending
                    .get(cb)
                    .and_then(|p| p.iter().rev().filter(|s| s.image == image).find_map(ImageSync::layout_after))
                    .or_else(|| self.attached.get(cb).and_then(|a| a.iter().rev().find(|(i, _)| *i == image).map(|(_, l)| *l)))
            })
            .or_else(|| self.committed.get(&image).copied())
    }

    /// [`Self::inline_point`] refused a hold inside DLSS's buffer at a launch naming the colour input `colour`:
    /// said once per identification and kind of refusal, with the image and how it was made. The line is logged
    /// by the caller once the tracker's lock is let go ([`Said`]).
    fn note_inline_refusal(&mut self, colour: vk::Image, refusal: InlineRefusal) -> Option<Said> {
        note_refusal(Refusal::Inside(refusal));
        if refusal.quiet() {
            // Periodic: not flushed.
            return self.quiet_refusal_line(refusal, Instant::now()).map(|line| Said { line, flush: false });
        }
        let why = refusal.said();
        if self.said_refusals.0 != self.generation {
            self.said_refusals = (self.generation, 0);
        }
        let bit = 1u16 << refusal.kind();
        if self.said_refusals.1 & bit != 0 {
            return None;
        }
        self.said_refusals.1 |= bit;
        let image = self.images.get(&colour).map_or_else(|| "unknown".to_string(), |d| format!("{}x{} {:?} {:?}", d.width, d.height, d.format, d.usage));
        let layout = if refusal == InlineRefusal::LayoutUnknown { self.colour_barrier_text(colour) } else { String::new() };
        let line = format!("no hold inside DLSS's buffer (identification {}): {why}{layout}; colour input {} {image}", self.generation, hex(colour));
        Some(Said { line, flush: true })
    }

    /// What was seen of `colour`'s barriers, for a refusal on its unknown layout: how many since this identification
    /// and the last one's layout, or that it is not watched (not of the identified colour input's shape).
    fn colour_barrier_text(&self, colour: vk::Image) -> String {
        match self.colour_barriers.get(&colour) {
            Some((n, last)) => format!(" ({n} barriers on it seen since this identification, last to {last:?})"),
            None if self.watched(colour) => " (no barrier on it seen since this identification)".to_string(),
            None => " (its barriers are not watched: it is not of the identified colour input's extent and format)".to_string(),
        }
    }

    /// Counts a [`InlineRefusal::quiet`] refusal at `now`; the line with the counts since the last one, at the first
    /// and then at most every [`QUIET_REFUSALS_EVERY`].
    fn quiet_refusal_line(&mut self, refusal: InlineRefusal, now: Instant) -> Option<String> {
        let (since, counts) = &mut self.quiet_refusals;
        counts[usize::from(refusal == InlineRefusal::BegunBefore)] += 1;
        if since.is_some_and(|s| now.duration_since(s) < QUIET_REFUSALS_EVERY) {
            return None;
        }
        let line = format!(
            "no hold inside DLSS's buffer{}: {} buffer(s) begun before the current identification ({}; what they did to DLSS's inputs before it is not known), {} with nothing identified",
            since.map_or_else(String::new, |s| format!(" in the last {:.1} s", now.duration_since(s).as_secs_f32())),
            counts[1],
            self.generation,
            counts[0]
        );
        *since = Some(now);
        *counts = [0; 2];
        Some(line)
    }

    /// Why no hold inside DLSS's buffer was even decided on in launch buffer `cb`, which the split refused: once
    /// per reason. `None` when one was ([`Self::inline_tried`]: its refusal, or the tally of what the hold did,
    /// says the rest) or the reason was said.
    fn untried_line(&mut self, cb: vk::CommandBuffer) -> Option<String> {
        if self.inline_tried.contains(&cb) {
            return None;
        }
        let refs = self.launch.get(&cb);
        let (bit, why) = if refs.is_some_and(|r| r.opaque && r.ngx.is_none()) {
            (1, "its CUDA launches' parameters are not readable (not in CUDA's buffer form), so none of them is matched to NGX's record of the evaluate")
        } else if refs.is_some_and(|r| r.ngx.is_none()) {
            (2, "none of its launches names the colour input of an NGX evaluate before that evaluate's end")
        } else {
            (4, "its first launch naming the evaluate's colour input was not taken as the hold point")
        };
        if self.said_untried & bit != 0 {
            return None;
        }
        self.said_untried |= bit;
        Some(format!("the hold inside DLSS's buffer was not tried in a launch buffer the split refused either: {why}"))
    }

    /// A hold inside DLSS's buffer was recorded at a point of identification `identification`.
    pub(crate) fn inline_recorded(&mut self, identification: u64) {
        self.inline_in_use = Some(identification);
    }

    /// What NGX's record of `cb`'s evaluate says of the frame, for a hold recorded into `cb` ([`DlssFrame::of`]);
    /// nothing without one.
    fn frame_of(&self, cb: vk::CommandBuffer) -> DlssFrame {
        self.launch.get(&cb).and_then(|r| r.ngx).map_or_else(DlssFrame::default, |r| DlssFrame::of(&r))
    }

    /// Takes the mark [`Self::launch_kernel`] left for the launch just recorded.
    pub(crate) fn take_inline_at(&mut self) -> Option<(vk::CommandBuffer, crate::ngx_ring::Record)> {
        self.inline_at.take()
    }

    /// What this device's frames offer the placements ([`placement::Frame`]): Ray Reconstruction running, and
    /// DLSS's output learnt. The hook before the upscaler is taken as there: where DLSS's input is not
    /// identified nothing is held anyway.
    pub(crate) fn frame(&self) -> placement::Frame {
        self.frame_at(Instant::now())
    }

    fn frame_at(&self, now: Instant) -> placement::Frame {
        let refused = self.after_refused.is_some_and(|(o, until)| Some(o) == self.dlss_output && now < until);
        placement::Frame { rr: self.rr_running(), output: self.dlss_output.is_some() && !refused, hook: true }
    }

    /// DLSS's output size, once its output is learnt.
    fn output_extent(&self) -> (u32, u32) {
        self.dlss_output.and_then(|o| self.images.get(&o.image)).map_or((0, 0), |d| (d.width, d.height))
    }

    /// The hold right after the launch just recorded into `cb`, the last of its evaluate to write DLSS's output
    /// ([`placement`]): on DLSS's output image (RGBA16F or R11G11B10, transferable, in a layout known here), with
    /// the evaluate's motion vectors at the render size (resampled up to the output's; [`InlinePoint::mvec_scale`]
    /// is then output over render pixels) and no depth copy (the depth is at the render size), the exposure
    /// measured from the frame.
    pub(crate) fn after_point(&mut self) -> Option<InlinePoint> {
        let (cb, o, plan) = self.after_at.take()?;
        let point = self.after_point_at(cb, o, &plan);
        if point.is_none() {
            self.after_missed();
        }
        point
    }

    /// A hold after DLSS's output was refused (its point, or a buffer it cannot be recorded in; not a
    /// buffer that already holds one or busy staging slots, [`inline::Recorded`]): after [`AFTER_MISSES`] in
    /// a row its output counts as not learnable ([`Self::frame`]) for [`AFTER_RETRY`], so its passes go where
    /// `auto` would meanwhile; then it is tried again (logged once per output).
    pub(crate) fn after_missed(&mut self) {
        self.after_missed_at(Instant::now());
    }

    fn after_missed_at(&mut self, now: Instant) {
        let Some(o) = self.dlss_output else { return };
        match &mut self.after_misses {
            (seen, n) if *seen == Some(o) => *n += 1,
            m => *m = (Some(o), 1),
        }
        if self.after_misses.1 >= AFTER_MISSES {
            self.after_misses = (None, 0);
            self.after_refused = Some((o, now + AFTER_RETRY));
            if !self.after_refused_said.contains(&o) {
                self.after_refused_said.push(o);
                crate::log!(
                    "[placement] no hold after DLSS's output {} could be had in {AFTER_MISSES} tries in a row (its format, usage or layout, or the buffer it is recorded in); the passes placed there run where auto runs them, and it is tried again every {} s",
                    hex(o.image),
                    AFTER_RETRY.as_secs()
                );
                crate::logging::flush();
            }
        }
    }

    /// A hold after DLSS's output was recorded: the count of misses starts over, and a refusal ends.
    pub(crate) fn after_recorded(&mut self) {
        self.after_misses = (None, 0);
        self.after_refused = None;
    }

    fn after_point_at(&mut self, cb: vk::CommandBuffer, o: DlssOutput, plan: &placement::Plan) -> Option<InlinePoint> {
        let image = *self.images.get(&o.image)?;
        let record = self.launch.get(&cb).and_then(|r| r.ngx);
        // The colour input's extent: the image's where the layer tracks it, else the one the record gives; (0, 0) when
        // neither is known.
        let input = self.images.get(&o.input).map(|d| (d.width, d.height)).or(record.map(|r| r.colour_extent).filter(|e| e.0 > 0 && e.1 > 0)).unwrap_or((0, 0));
        // DLSS's render size (its subrect's where given) and the rectangle of the output it writes: the hold works on
        // that alone, and the render-size motion vectors at their base are resampled up to it.
        let render = record.map_or(input, |r| r.render_size(input));
        let out = match record.map(|r| r.output_crop((image.width, image.height))) {
            Some(Err(why)) => {
                if !self.after_crop_said.contains(&o.image) {
                    self.after_crop_said.push(o.image);
                    crate::log!("[placement] no hold after DLSS's output {}: {why}", hex(o.image));
                    crate::logging::flush();
                }
                return None;
            }
            Some(Ok(Some(rect))) => rect,
            _ => crate::ngx_ring::Rect::whole((image.width, image.height)),
        };
        // Without the colour input's extent nothing says what DLSS reads of it: the whole images.
        let subrect = input != (0, 0) && record.is_some_and(|r| r.has_subrect(input));
        let desc = ImageDesc { width: out.size.0, height: out.size.1, ..image };
        let copyable = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !OUTPUT_FORMATS.contains(&desc.format) || !desc.plain || !desc.usage.contains(copyable) {
            return None;
        }
        if self.rendering.get(&cb).is_some_and(|r| r.last_suspending || r.open_resume) {
            return None;
        }
        let layout_of = |image: vk::Image, d: &ImageDesc| self.layout_in(cb, image, true, d.usage.contains(vk::ImageUsageFlags::STORAGE));
        let mut guessed = Guessed::default();
        let (layout, set) = layout_of(o.image, &desc)?;
        if !set {
            guessed.colour = Some(layout);
        }
        if !INLINE_LAYOUTS.contains(&layout) {
            return None;
        }
        let mvec = self.images.get(&o.mvec).copied();
        // The motion vectors' rectangle: with a subrect, the one the hold before the upscaler reads (MVLowRes first,
        // [`crate::ngx_ring::Record::mvec_rect`]), where it fits their image. With the colour input's extent unknown but
        // DLSS's render size known and not their image's, the same rule on the render size (none where it does not fit:
        // the vectors are dropped rather than the whole image resampled over a rectangle DLSS does not read).
        let mvec_rect = mvec.and_then(|d| {
            let extent = (d.width, d.height);
            match record {
                Some(r) if subrect => r.mvec_rect(input, extent),
                Some(r) if input == (0, 0) && r.render.0 > 0 && r.render.1 > 0 && r.render != extent => r.mvec_rect(r.render, extent),
                _ => Some(crate::ngx_ring::Rect::whole(extent)),
            }
        });
        let mvec_input = mvec.map(|d| {
            let known = layout_of(o.mvec, &d);
            guessed.mvec = known.filter(|(_, set)| !set).map(|(l, _)| l);
            let layout = known.map(|(l, _)| l);
            let readable = d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
                && MVEC_HOLD_FORMATS.contains(&d.format)
                && mvec_rect.is_some()
                && layout.is_some_and(|l| MVEC_COPY_LAYOUTS.contains(&l));
            Aux { image: o.mvec, format: d.format, layout, readable }
        });
        Some(InlinePoint {
            colour: o.image,
            image: (image.width, image.height),
            desc,
            offsets: Offsets { colour: out.at, mvec: mvec_rect.map_or((0, 0), |r| r.at), depth: (0, 0) },
            layout,
            exposure_input: None,
            mvec_input,
            mvec_extent: mvec_rect.map_or((0, 0), |r| r.size),
            // NGX's `MV.Scale` gives the colour input's pixels; the hold works on the output's.
            mv_units: record
                .and_then(|r| r.mv_units())
                .filter(|_| render.0 > 0 && render.1 > 0)
                .map(|units| [units[0] * desc.width as f32 / render.0 as f32, units[1] * desc.height as f32 / render.1 as f32]),
            depth_input: None,
            // Its own identification: the output image's handle with the top bit set (never an input's).
            identification: o.image.as_raw() | (1 << 63),
            hazard: Hazard::NotSplittable,
            hold: placement::Hold::of(plan, placement::Stage::Upscaled, render, (desc.width, desc.height)),
            guessed,
        })
    }

    /// Whether any command buffer carries a launch, recorded barriers or other synchronization.
    fn armed(&self) -> bool {
        !self.launch.is_empty()
            || !self.pending.is_empty()
            || !self.copies.is_empty()
            || !self.global_sync.is_empty()
            || !self.rendering.is_empty()
            || !self.attached.is_empty()
            || !self.recorded_watched.is_empty()
    }

    /// `vkBeginCommandBuffer` (or a reset): the earlier recording is gone, and the new one is seen
    /// whole from here on when inputs are watched ([`Self::recorded_watched`]).
    pub(crate) fn begin(&mut self, command_buffer: vk::CommandBuffer) {
        self.free(&[command_buffer]);
        if self.inputs.is_some() {
            self.recorded_watched.insert(command_buffer);
        }
    }

    pub(crate) fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.launch.remove(cb);
            self.pending.remove(cb);
            self.copies.remove(cb);
            self.global_sync.remove(cb);
            self.event_waits.remove(cb);
            self.rendering.remove(cb);
            self.attached.remove(cb);
            self.recorded_watched.remove(cb);
            self.inline_tried.remove(cb);
        }
    }

    /// `vkCmdBeginRendering`'s attachments (`views` with their layouts) recorded into
    /// `command_buffer`: a watched image attached there is in that layout (Vulkan requires it), so
    /// its layout is known from there on in the buffer without a barrier. GTA San Andreas - The
    /// Definitive Edition never transitions DLSS's colour input with a barrier once identified
    /// (2026-10-06), so the hold inside DLSS's buffer had no layout for it. Kept apart from the
    /// barriers: it is no synchronization, and the hazard rules count barriers.
    pub(crate) fn attachments(&mut self, command_buffer: vk::CommandBuffer, views: &[(vk::ImageView, vk::ImageLayout)]) {
        for &(view, layout) in views {
            let Some(&image) = self.views.get(&view) else { continue };
            if self.watched(image) {
                self.attached.entry(command_buffer).or_default().push((image, layout));
            }
        }
    }

    /// Global synchronization recorded into `command_buffer` (a memory barrier with a write in its
    /// source access, an event wait); the first one is kept.
    pub(crate) fn global(&mut self, command_buffer: vk::CommandBuffer, hazard: Hazard) {
        self.global_sync.entry(command_buffer).or_insert(hazard);
        if hazard == Hazard::EventWait {
            self.event_waits.insert(command_buffer);
        }
    }

    /// `vkCmdCopyImage` or `vkCmdCopyImage2` from `src` into `dst` recorded into `command_buffer`: kept where `dst` is
    /// watched ([`Self::copy_source`]).
    pub(crate) fn copy(&mut self, command_buffer: vk::CommandBuffer, src: vk::Image, dst: vk::Image) {
        if self.watched(dst) {
            self.copies.entry(command_buffer).or_default().push((src, dst));
        }
    }

    /// Whether any of `cbs` waits on an event ([`Self::event_waits`]).
    pub(crate) fn waits_on_event(&self, cbs: &[vk::CommandBuffer]) -> bool {
        cbs.iter().any(|cb| self.event_waits.contains(cb))
    }

    /// `vkCmdBeginRendering` with `VK_RENDERING_SUSPENDING_BIT` or `VK_RENDERING_RESUMING_BIT`. A
    /// suspended render pass instance is resumed by the next one, so a resuming one that does not
    /// follow a suspending one in its own buffer resumes the previous buffer's.
    pub(crate) fn begin_rendering(&mut self, command_buffer: vk::CommandBuffer, flags: vk::RenderingFlags) {
        let r = self.rendering.entry(command_buffer).or_default();
        if flags.contains(vk::RenderingFlags::RESUMING) && !r.last_suspending {
            r.open_resume = true;
        }
        r.last_suspending = flags.contains(vk::RenderingFlags::SUSPENDING);
    }

    /// `vkCmdExecuteCommands`: each secondary's recording continues the primary's, in order.
    pub(crate) fn execute(&mut self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        for secondary in secondaries {
            // A secondary recorded (in part) before the identification makes the primary so too.
            if !self.recorded_watched.contains(secondary) {
                self.recorded_watched.remove(&primary);
            }
            // Where the secondary's recording starts in the primary's.
            let base = Before {
                barriers: self.pending.get(&primary).map_or(0, Vec::len),
                global: self.global_sync.get(&primary).copied(),
                copies: self.copies.get(&primary).map_or(0, Vec::len),
            };
            let shift = |b: Before| Before { barriers: base.barriers + b.barriers, global: base.global.or(b.global), copies: base.copies + b.copies };
            if let Some(c) = self.launch.get(secondary).cloned() {
                let refs = self.launch.entry(primary).or_default();
                refs.opaque |= c.opaque;
                if refs.before_first.is_none() {
                    refs.before_first = c.before_first.map(shift);
                }
                // The secondary's evaluate, or more of the primary's: its launches follow the primary's so far.
                match (refs.ngx, c.ngx) {
                    (None, Some(_)) => {
                        refs.ngx = c.ngx;
                        refs.ended = c.ended;
                        refs.output_writes = c.output_writes;
                    }
                    (Some(a), Some(b)) if a.seq == b.seq => {
                        refs.ended |= c.ended;
                        refs.output_writes += c.output_writes;
                    }
                    _ => {}
                }
                for image in c.images {
                    if !refs.images.contains(&image) {
                        refs.images.push(image);
                        let before = c.before_named.iter().find(|(i, _)| *i == image).map(|(_, b)| *b).or(c.before_first).unwrap_or_default();
                        refs.before_named.push((image, shift(before)));
                    }
                }
            }
            if let Some(inherited) = self.pending.get(secondary).filter(|p| !p.is_empty()).cloned() {
                self.pending.entry(primary).or_default().extend(inherited);
            }
            if let Some(inherited) = self.copies.get(secondary).filter(|c| !c.is_empty()).cloned() {
                self.copies.entry(primary).or_default().extend(inherited);
            }
            if let Some(hazard) = self.global_sync.get(secondary).copied() {
                self.global(primary, hazard);
            }
            if self.event_waits.contains(secondary) {
                self.event_waits.insert(primary);
            }
            if let Some(inherited) = self.attached.get(secondary).filter(|a| !a.is_empty()).cloned() {
                self.attached.entry(primary).or_default().extend(inherited);
            }
            if let Some(c) = self.rendering.get(secondary).copied() {
                let r = self.rendering.entry(primary).or_default();
                if c.open_resume && !r.last_suspending {
                    r.open_resume = true;
                }
                r.last_suspending = c.last_suspending;
            }
        }
    }

    /// Whether barriers on `image` are recorded now.
    fn watched(&self, image: vk::Image) -> bool {
        self.watched_by(self.inputs, self.colour_watch, image)
    }

    /// Whether barriers on `image` are recorded with `inputs` identified and `watch` the colour shape watched.
    fn watched_by(&self, inputs: Option<Inputs>, watch: Option<ColourWatch>, image: vk::Image) -> bool {
        let desc = self.images.get(&image);
        // Every image of the colour input's shape: an evaluate may name another one than the identified (Remnant II and
        // Expedition 33 alternate between several).
        watch.is_some_and(|w| desc.is_some_and(|d| w.covers(d)))
            || inputs.is_some_and(|i| {
                i.colour.0 == image
                    || i.depth.0 == image
                    || i.mvec.0 == image
                    || i.exposure_input.is_some_and(|e| e.0 == image)
                    // Every depth image at the identified depth's extent: an evaluate may name another one than the
                    // identified (`Self::hold_depth`; GTA V Enhanced alternates two).
                    || desc.is_some_and(|d| DEPTH_FORMATS.contains(&d.format) && (d.width, d.height) == (i.depth.1.width, i.depth.1.height))
                    // Every image of the motion vectors' and the exposure image's shape: GTA V Enhanced hands DLSS four
                    // motion-vector images and as many exposure images in turn (2026-11-11), and a hold needs the
                    // layout of the one its evaluate names, recorded before that evaluate's record was read.
                    || desc.is_some_and(|d| (d.width, d.height, d.format) == (i.mvec.1.width, i.mvec.1.height, i.mvec.1.format))
                    || i.exposure_input.is_some_and(|(_, e)| desc.is_some_and(|d| (d.width, d.height, d.format) == (e.width, e.height, e.format)))
            })
    }

    /// At a new identification (`inputs`, watching `watch`): what was recorded on an image watched both before and
    /// after (its committed layout and owner, its layout at DLSS's launch, its barriers in buffers not yet
    /// submitted) was recorded throughout, and is kept; everything else is dropped, as everything was before
    /// (#51: Expedition 33's identification changes as Unreal Engine 5 hands DLSS other images, and each change
    /// left the hold inside DLSS's buffer without a layout for any of them). Buffers begun before stay
    /// [`Hazard::RecordedBefore`] and [`InlineRefusal::BegunBefore`]: what they recorded on images newly watched
    /// was not seen.
    fn keep_watched_across(&mut self, inputs: Option<Inputs>, watch: Option<ColourWatch>) {
        let (old, old_watch) = (self.inputs, self.colour_watch);
        let seen: HashSet<vk::Image> = self
            .committed
            .keys()
            .chain(self.owners.keys())
            .chain(self.launch_layouts.keys())
            .copied()
            .chain(self.pending.values().flatten().map(|s| s.image))
            .chain(self.attached.values().flatten().map(|a| a.0))
            .collect();
        let keep: HashSet<vk::Image> = seen.into_iter().filter(|&i| self.watched_by(old, old_watch, i) && self.watched_by(inputs, watch, i)).collect();
        self.committed.retain(|i, _| keep.contains(i));
        self.owners.retain(|i, _| keep.contains(i));
        self.launch_layouts.retain(|i, _| keep.contains(i));
        for recorded in self.pending.values_mut() {
            recorded.retain(|s| keep.contains(&s.image));
        }
        self.pending.retain(|_, recorded| !recorded.is_empty());
        for attached in self.attached.values_mut() {
            attached.retain(|a| keep.contains(&a.0));
        }
        self.attached.retain(|_, attached| !attached.is_empty());
        self.colour_watch = watch;
    }

    pub(crate) fn barrier(&mut self, command_buffer: vk::CommandBuffer, sync: ImageSync) {
        // One aspect of a depth/stencil image: a transition leaves the two possibly apart from here on; a barrier that
        // keeps that aspect's layout moves neither. Stellar Blade (#51, vkd3d-proton) has `GENERAL -> GENERAL` barriers
        // on its depth's depth aspect alone in front of DLSS's buffer: taken as apart, the hold inside DLSS's buffer
        // (which had guessed the depth GENERAL) was found wrong by its guard at every submit, and every hold dropped
        // the network's history.
        // Only where the layout tracked at that point is the one the barrier names: one tracked otherwise (a render
        // pass's final layout the tracking did not follow, a buffer recorded before the identification) says only that
        // this aspect is in the barrier's layout, so the two may be apart. Where none is tracked, none is set: the
        // other aspect's layout is still not known, and an image without a known layout is not read.
        let one_aspect = sync.one_aspect_of_two() && self.images.get(&sync.image).is_some_and(|d| has_stencil(d.format));
        let keeps = |t: &Self| t.layout_in(command_buffer, sync.image, true, false).is_none_or(|(l, _)| l == sync.old_layout);
        let sync = match (one_aspect, sync.old_layout == sync.new_layout) {
            (true, true) if keeps(self) => ImageSync { keeps_layout: true, ..sync },
            (true, _) => ImageSync { new_layout: ASPECTS_APART, ..sync },
            (false, _) => sync,
        };
        let colour_shape = self.colour_watch.is_some_and(|w| self.images.get(&sync.image).is_some_and(|d| w.covers(d)));
        if colour_shape || self.inputs.is_some_and(|i| i.colour.0 == sync.image) {
            let seen = self.colour_barriers.entry(sync.image).or_insert((0, sync.new_layout));
            *seen = (seen.0 + 1, sync.new_layout);
        }
        if self.watched(sync.image) {
            self.pending.entry(command_buffer).or_default().push(sync);
        }
    }

    /// The command buffers of a submission the next layer accepted (`VK_SUCCESS`), in submission
    /// order: only now do their recorded barriers become the watched images' state. A failed
    /// submission executed nothing, so it is never passed here; of a split call, each part is
    /// passed once it was accepted ([`submit_around`]).
    pub(crate) fn commit_submitted(&mut self, command_buffers: impl IntoIterator<Item = vk::CommandBuffer>) {
        for cb in command_buffers {
            // Attachment layouts first, then the buffer's barriers (which follow its rendering).
            if let Some(attached) = self.attached.get(&cb) {
                for &(image, layout) in attached {
                    self.committed.insert(image, layout);
                }
            }
            let Some(recorded) = self.pending.get(&cb) else { continue };
            for sync in recorded {
                if let Some(layout) = sync.layout_after() {
                    self.committed.insert(sync.image, layout);
                }
                if sync.transfers_ownership() {
                    self.owners.insert(sync.image, sync.dst_queue_family);
                }
            }
        }
    }

    /// [`Self::scan`] followed by the commit of a submission the next layer accepted whole.
    #[cfg(test)]
    pub(crate) fn submit_ok(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        let scan = self.scan(batches);
        self.commit_submitted(batches.iter().flatten().copied());
        scan
    }

    /// Why a hold must not run in front of launch buffer `cb` reading `inputs`, from what was
    /// recorded into it before the launch that first names the colour input (before its first
    /// launch when none names it): a barrier on an image the hold reads, or global synchronization.
    ///
    /// Returns `(hazard, dump_hazard, depth_hazard)`, the last one a barrier on the depth image
    /// alone ([`Scan::depth_hazard`]). Every hold reads and writes the colour input and reads the
    /// exposure input; only a dump also reads depth and the motion vectors, so a barrier on one of
    /// those alone stops only a dump. With DLSS Frame Generation
    /// on, GTA V Enhanced copies depth and motion vectors for it inside the launch buffer, between
    /// `GENERAL -> GENERAL` barriers, right before SR's input launch (docs/PRE_UPSCALER_DESIGN.md,
    /// "When a submit is not split").
    fn launch_hazard(&self, cb: vk::CommandBuffer, inputs: Option<&Inputs>) -> (Option<Hazard>, Option<Hazard>, Option<Hazard>) {
        if self.rendering.get(&cb).is_some_and(|r| r.open_resume) {
            return (Some(Hazard::SuspendedRendering), None, None);
        }
        let Some(refs) = self.launch.get(&cb) else { return (None, None, None) };
        let colour = inputs.map(|i| i.colour.0);
        let Some(before) = colour.and_then(|c| refs.before_named.iter().find(|(i, _)| *i == c)).map(|(_, b)| *b).or(refs.before_first) else {
            return (None, None, None);
        };
        if before.global.is_some() {
            return (before.global, None, before.global);
        }
        // Unknown inputs: every watched image counts as read by every hold.
        let held = |image: vk::Image| inputs.is_none_or(|i| i.colour.0 == image || i.exposure_input.is_some_and(|e| e.0 == image));
        let dumped = |image: vk::Image| inputs.is_some_and(|i| i.depth.0 == image || i.mvec.0 == image);
        let Some(recorded) = self.pending.get(&cb) else { return (None, None, None) };
        let before = &recorded[..before.barriers.min(recorded.len())];
        (
            before.iter().find(|s| held(s.image)).map(ImageSync::hazard),
            before.iter().find(|s| dumped(s.image)).map(ImageSync::hazard),
            before.iter().find(|s| inputs.is_some_and(|i| i.depth.0 == s.image) && s.may_follow_a_write()).map(ImageSync::hazard),
        )
    }

    /// Identifies DLSS's inputs from NGX's records: the newest upscaling evaluate whose colour input the layer saw created
    /// on this device ([`crate::ngx_ring::newest`]), where its depth and motion vectors are such images too. Ray
    /// Reconstruction's identifies nothing: its input is not held. A new identification is a new shape of the colour
    /// input, motion vectors or exposure image ([`Key`]); other images of the same shapes, or another depth image, are
    /// the same one. Called at every launch-bearing submit; returns a log line when the identification changed.
    pub(crate) fn refresh(&mut self) -> Option<String> {
        let record = crate::ngx_ring::newest(|i| self.images.contains_key(&i));
        let rr = record.is_some_and(|r| r.is_rr());
        if rr {
            note_ray_reconstruction();
        }
        let inputs = record.filter(|r| !r.is_rr()).and_then(|r| self.inputs_of(&r));
        let key = inputs.map(|i| (ColourWatch::of(&i.colour.1), ColourWatch::of(&i.mvec.1), i.exposure_input.map(|e| ColourWatch::of(&e.1))));
        // Back to images identified before (a rotation): the same identification, its tracking kept.
        let rotation = key.is_some() && self.key.is_some() && self.seen_keys.contains(&key.expect("checked"));
        if let Some(k) = key.filter(|k| !self.seen_keys.contains(k)) {
            if self.seen_keys.len() == SEEN_KEYS {
                self.seen_keys.remove(0);
            }
            self.seen_keys.push(k);
        }
        let changed = (key != self.key && !rotation) || rr != self.rr;
        if key != self.key && !rotation {
            // The colour shape watched: the new colour input's, else (nothing identified) the last one's.
            self.keep_watched_across(inputs, inputs.map(|i| ColourWatch::of(&i.colour.1)).or(self.colour_watch));
            self.recorded_watched.clear();
            self.colour_barriers.clear();
            self.identified = true;
            self.generation += 1;
            self.key = key;
        }
        self.inputs = inputs;
        self.rr = rr;
        if !changed {
            return None;
        }
        let line = match (inputs, record) {
            (Some(i), _) => format!(
                "colour input: image {} ({}x{} {:?} {:?}), depth {} {:?}, motion vectors {} {:?}, exposure input {}; from NGX's record of DLSS's evaluate",
                hex(i.colour.0),
                i.colour.1.width,
                i.colour.1.height,
                i.colour.1.format,
                i.colour.1.usage,
                hex(i.depth.0),
                i.depth.1.format,
                hex(i.mvec.0),
                i.mvec.1.format,
                i.exposure_input.map_or_else(|| "none (the game gives DLSS none; measured from the frame)".to_string(), |(image, _)| hex(image)),
            ),
            (None, Some(_)) if rr => {
                "no DLSS input: DLSS Ray Reconstruction runs (NGX's record): its input is the noisy ray-traced frame with its guide buffers, so the model can't run before it; nothing is held, the model runs after the upscaler".to_string()
            }
            (None, Some(r)) => format!(
                "no DLSS input: NGX's record names images this device did not see created (colour {}, depth {}, motion vectors {})",
                hex(r.colour),
                hex(r.depth),
                hex(r.mvec)
            ),
            (None, None) => "no DLSS input: no NGX record of a DLSS evaluate on this device".to_string(),
        };
        if self.announced.as_deref() == Some(line.as_str()) {
            return None;
        }
        self.announced = Some(line.clone());
        Some(line)
    }

    /// What the submit's launch-bearing buffers carry: DLSS's evaluates as NGX recorded them. An evaluate that ended
    /// says how DLSS writes its output ([`DlssOutput`]), for the hold after it. Called before [`Self::refresh`] at every
    /// launch-bearing submit. Returns the lines to log.
    pub(crate) fn observe(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Vec<String> {
        let mut lines = Vec::new();
        let mut any = false;
        let mut evaluate = false;
        let mut dlss_kernel = false;
        for cb in batches.iter().flatten() {
            let Some(refs) = self.launch.get_mut(cb) else { continue };
            any = true;
            dlss_kernel |= refs.dlss_kernel;
            let Some(r) = refs.ngx else { continue };
            evaluate = true;
            if !refs.ended {
                refs.ended = crate::ngx_ring::ended(r.seq);
            }
            // A D3D11 evaluate's launches are all in the buffer once it is submitted (DXVK's worker records them after
            // NVIDIA's call returned, and its end record says nothing of them): its writes are counted at the submit.
            let done = refs.ended || r.api == crate::ngx_ring::API_D3D11;
            if !done || refs.output_writes == 0 {
                continue;
            }
            let seen = DlssOutput { image: r.output, input: r.colour, depth: r.depth, mvec: r.mvec, rr: r.is_rr(), writes: refs.output_writes };
            if self.dlss_output.is_some_and(|l| same_output(&l, &seen, &self.images)) {
                continue;
            }
            if let Some(l) = self.dlss_output.filter(|l| seen.writes > l.writes && same_output(l, &DlssOutput { writes: l.writes, ..seen }, &self.images)) {
                lines.push(format!("DLSS's output is written in {} launches, not {}: a hold in this frame went after part of it", seen.writes, l.writes));
            }
            self.dlss_output = Some(seen);
            let desc = |i: vk::Image| self.images.get(&i).map_or("unregistered".to_string(), |d| format!("{}x{} {:?}", d.width, d.height, d.format));
            lines.push(format!(
                "DLSS's output: image {} ({}), written by {} launch(es) of its evaluate on {} ({}; depth {}, motion vectors {}){}",
                hex(seen.image),
                desc(seen.image),
                seen.writes,
                hex(seen.input),
                desc(seen.input),
                hex(seen.depth),
                hex(seen.mvec),
                if seen.rr { "; Ray Reconstruction" } else { "" }
            ));
        }
        if !any {
            return lines;
        }
        self.launch_submits += 1;
        // DLSS ran in this submit (`hand_back_at`), or CUDA launches ran that no record names.
        if evaluate {
            note_sr_launch();
        } else {
            crate::ngx_ring::note_launches_without_record(dlss_kernel);
        }
        lines
    }

    /// Finds the first command buffer in a submit (`batches`: each batch's command buffers, in order) whose launches
    /// belong to DLSS's Super Resolution evaluate ([`LaunchRefs::sr`]) and reads the watched images' layouts just
    /// before it: the committed ones with the barriers of the buffers ahead of it in this submit applied on top, in
    /// submission order. Its inputs are its own evaluate's, as NGX's record names them. Nothing is committed here: the
    /// submit has not happened yet, and may fail ([`Self::commit_submitted`]).
    pub(crate) fn scan(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        if !self.armed() {
            return None;
        }
        // What this submit's buffers so far would leave behind.
        let mut proposed: HashMap<vk::Image, (vk::ImageLayout, Option<u32>)> = HashMap::new();
        let mut found: Option<Scan> = None;
        let mut any_launch = false;
        // A buffer up to and including the launch buffer was recorded (in part) before the current
        // identification ([`Self::recorded_watched`]): what it recorded on the inputs is not known.
        let mut recorded_before = false;
        for (bi, cbs) in batches.iter().enumerate() {
            for (ci, &cb) in cbs.iter().enumerate() {
                if found.is_none() && !self.recorded_watched.contains(&cb) {
                    recorded_before = true;
                }
                let refs = self.launch.get(&cb);
                any_launch |= refs.is_some();
                if let Some(record) = refs.and_then(LaunchRefs::sr).filter(|_| found.is_none()) {
                    let rotating = self.note_exposure_name(&record);
                    let layout = |image: Option<vk::Image>| image.and_then(|i| proposed.get(&i).map(|p| p.0).or(self.committed.get(&i).copied()));
                    // The inputs this frame's hold in front of the buffer reads: where the buffer copies into one DLSS
                    // reads before the launch, the copy's source ([`Self::copy_source`], [`Self::hold_depth`]). GTA V
                    // Enhanced copies its depth and motion vectors into the ones DLSS reads inside the launch buffer.
                    let inputs = self.inputs_of(&record).map(|i| {
                        let before = self.before_launch(cb, i.colour.0);
                        let same = |named: (vk::Image, ImageDesc)| {
                            self.copy_source(cb, named, before, |d| (d.width, d.height, d.format) == (named.1.width, named.1.height, named.1.format))
                        };
                        // An exposure image the buffer copies into before the launch has no exposure in front of it,
                        // and neither has the copy's source: GTA V Enhanced's ring read 751, 13096, NaN and negative
                        // values there and its source about -2 (not an exposure), the history reset at almost every hold
                        // and the frame flashed green (2026-11-11). The hold measures the exposure from the frame instead.
                        // Nor has one of several the evaluates on one colour input name in turn (GTA V Enhanced's
                        // four): those are DLSS's own copies, made inside its buffer, and hold no exposure in front of
                        // it (751, 13096, NaN, -2.2 read: the frame flashed green and white, 2026-11-11).
                        let exposure_input = i.exposure_input.filter(|&e| !rotating && same(e).is_none());
                        let held = Inputs { depth: self.hold_depth(cb, i.depth, before, false), mvec: same(i.mvec).unwrap_or(i.mvec), exposure_input, ..i };
                        say_hold_inputs(&i, &held);
                        held
                    });
                    let colour_owner = inputs.and_then(|i| proposed.get(&i.colour.0).and_then(|p| p.1).or(self.owners.get(&i.colour.0).copied()));
                    let (hazard, dump_hazard, depth_hazard) = self.launch_hazard(cb, inputs.as_ref());
                    // DLSS given part of its colour input: the split works on the whole image.
                    let subrect = inputs.is_some_and(|i| record.has_subrect((i.colour.1.width, i.colour.1.height)));
                    let hazard = hazard.or(subrect.then_some(Hazard::Subrect));
                    self.probe_depth(cb, inputs.as_ref(), depth_hazard);
                    found = Some(Scan {
                        batch: bi,
                        index: ci,
                        colour_layout: layout(inputs.map(|i| i.colour.0)),
                        inputs,
                        depth_layout: layout(inputs.map(|i| i.depth.0)),
                        mvec_layout: layout(inputs.map(|i| i.mvec.0)),
                        exposure_input_layout: layout(inputs.and_then(|i| i.exposure_input).map(|e| e.0)),
                        evaluation: self.evaluations,
                        identified_now: self.identified,
                        identification: self.generation,
                        // Before the barrier rules: those read only what was recorded while watched.
                        hazard: if recorded_before { Some(Hazard::RecordedBefore) } else { hazard },
                        dump_hazard,
                        depth_hazard,
                        colour_owner,
                        frame: DlssFrame::of(&record),
                    });
                    // The colour image's layout at DLSS's launch buffer as this submission has it (what the
                    // split's check reads): the guess of the next hold recorded inside a buffer that does not
                    // set it itself ([`Self::inline_point`]).
                    if let (Some(i), Some(l), false) = (inputs, found.as_ref().and_then(|s| s.colour_layout), self.identified) {
                        self.launch_layouts.insert(i.colour.0, l);
                    }
                }
                if found.is_none() {
                    for sync in self.pending.get(&cb).into_iter().flatten() {
                        let owner = sync.transfers_ownership().then_some(sync.dst_queue_family).or(proposed.get(&sync.image).and_then(|p| p.1));
                        // A barrier that keeps the layout leaves the one before it (none known: none proposed).
                        let layout = sync.layout_after().or(proposed.get(&sync.image).map(|p| p.0)).or(self.committed.get(&sync.image).copied());
                        if let Some(layout) = layout {
                            proposed.insert(sync.image, (layout, owner));
                        }
                    }
                }
            }
        }
        if found.is_some() {
            self.evaluations += 1;
            self.identified = false;
            self.evaluate_submits += 1;
        } else if any_launch {
            self.foreign_submits += 1;
        }
        found
    }

    /// `NEURAL_FORGE_PROBE_DEPTH=1`: for the first 120 launch-bearing submits held, logs the
    /// identified and the named depth, their layouts, and the launch buffer's barriers on depth
    /// images (before the launch or not). Diagnostic only.
    fn probe_depth(&self, cb: vk::CommandBuffer, inputs: Option<&Inputs>, hazard: Option<Hazard>) {
        static ON: LazyLock<bool> = LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_PROBE_DEPTH").as_deref() == Some("1"));
        static COUNT: AtomicU64 = AtomicU64::new(0);
        if !*ON || COUNT.fetch_add(1, Ordering::Relaxed) >= 120 {
            return;
        }
        let Some(i) = inputs else { return };
        let identified = self.inputs.map(|x| x.depth.0);
        let before = self.launch.get(&cb).and_then(|r| r.before_named.iter().find(|(im, _)| *im == i.colour.0).map(|(_, b)| *b).or(r.before_first));
        let barriers: Vec<String> = self
            .pending
            .get(&cb)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(_, s)| self.images.get(&s.image).is_some_and(|d| DEPTH_FORMATS.contains(&d.format)))
            .map(|(k, s)| {
                format!(
                    "#{k}{} {} {:?}->{:?} access {:#x}->{:#x}",
                    if before.is_some_and(|b| k < b.barriers) { "(before)" } else { "" },
                    hex(s.image),
                    s.old_layout,
                    s.new_layout,
                    s.src_access,
                    s.dst_access
                )
            })
            .collect();
        crate::log!(
            "[probe-depth] identified {} ({:?}) named {} ({:?}) hazard {hazard:?}; depth barriers in the launch buffer: [{}]",
            identified.map_or_else(|| "-".into(), hex),
            identified.and_then(|d| self.committed.get(&d)),
            hex(i.depth.0),
            self.committed.get(&i.depth.0),
            barriers.join("; ")
        );
    }

    /// How many of `cb`'s recorded barriers come before the launch that first names `colour` (before
    /// its first launch when none names it), as [`Self::launch_hazard`] counts them.
    fn before_launch(&self, cb: vk::CommandBuffer, colour: vk::Image) -> Option<Before> {
        let refs = self.launch.get(&cb)?;
        refs.before_named.iter().find(|(i, _)| *i == colour).map(|(_, b)| *b).or(refs.before_first)
    }

    /// The depth a hold reads for DLSS this frame, at `identified`'s extent:
    /// 1. in front of the buffer (`inside` false), the source of the buffer's copy before the launch into the depth
    ///    image NGX's record names ([`Self::copy_source`]): the frame's depth. GTA V Enhanced copies its depth, one of
    ///    two it alternates between, into a ring of three DLSS reads, inside the launch buffer, with no barrier on the
    ///    ring's image there: the ring's image (its layout known from frames before) is written after the hold and
    ///    held the depth of three frames back, the source is final before it;
    /// 2. the named one, where its layout is known and the buffer does not write it before the launch (no barrier on
    ///    it there that may follow a write): DLSS's own input, final at the hold. Inside DLSS's buffer (`inside`: the
    ///    copy is recorded at the launch itself, after anything the buffer wrote) that is enough;
    /// 3. else a depth image the buffer synchronizes reads of before the launch (a same-layout barrier
    ///    after reads only, within the barriers `before` counts, or all of them): the frame's depth
    ///    that the buffer copies into the one DLSS names, as far as the barriers tell;
    /// 4. else the named one where its layout is known (its write before the launch then leaves the
    ///    hold without depth, `Scan::depth_hazard`);
    /// 5. else `identified`.
    fn hold_depth(&self, cb: vk::CommandBuffer, identified: (vk::Image, ImageDesc), before: Option<Before>, inside: bool) -> (vk::Image, ImageDesc) {
        let extent = (identified.1.width, identified.1.height);
        let is_depth = |d: &ImageDesc| DEPTH_FORMATS.contains(&d.format) && (d.width, d.height) == extent;
        let depth = |image: vk::Image| self.images.get(&image).filter(|d| is_depth(d)).map(|d| (image, *d));
        let named = self.launch.get(&cb).and_then(|r| r.ngx).and_then(|r| depth(r.depth));
        if let Some(source) = named.filter(|_| !inside).and_then(|n| self.copy_source(cb, n, before, is_depth)) {
            return source;
        }
        let recorded = self.pending.get(&cb).map_or(&[][..], Vec::as_slice);
        let recorded = &recorded[..before.map_or(recorded.len(), |b| b.barriers).min(recorded.len())];
        let known = |image: vk::Image| self.committed.contains_key(&image) || recorded.iter().any(|s| s.image == image);
        let named = named.filter(|(d, _)| known(*d));
        let written = |image: vk::Image| recorded.iter().any(|s| s.image == image && s.may_follow_a_write());
        if let Some(n) = named.filter(|(d, _)| inside || !written(*d)) {
            return n;
        }
        let read = recorded.iter().rev().filter(|s| s.old_layout == s.new_layout && !s.may_follow_a_write()).find_map(|s| depth(s.image));
        read.or(named).unwrap_or(identified)
    }

    /// Notes the exposure image `record`'s evaluate names on its colour input ([`Self::exposure_names`]); whether
    /// evaluates on that colour input have named more than one. Said once per colour input when they start to.
    fn note_exposure_name(&mut self, record: &crate::ngx_ring::Record) -> bool {
        if record.exposure == vk::Image::null() {
            return false;
        }
        if self.exposure_names.len() > 64 && !self.exposure_names.contains_key(&record.colour) {
            self.exposure_names.clear();
        }
        let (first, rotating) = self.exposure_names.entry(record.colour).or_insert((record.exposure, false));
        if !*rotating && *first != record.exposure {
            *rotating = true;
            crate::log!(
                "[preupscale] DLSS's evaluates on colour input {} name different exposure images ({} and {}): copies made inside its buffer; the exposure is measured from the frame",
                hex(record.colour),
                hex(*first),
                hex(record.exposure)
            );
            crate::logging::flush();
        }
        *rotating
    }

    /// The image buffer `cb` copies into `named` before its launch (within the copies `before` counts, or all of
    /// them), the last such, where `shape` takes its description: the frame's input the buffer copies into the one
    /// DLSS reads, final before the buffer, where `named` is written after a hold in front of it. GTA V Enhanced copies
    /// its frame's depth and motion vectors (one image) into rings of three DLSS reads inside the launch buffer; a hold
    /// that read the ring's image had the inputs of three frames back, and the edit cache rejected all of a cached
    /// hold's history (2026-11-11).
    fn copy_source(
        &self,
        cb: vk::CommandBuffer,
        named: (vk::Image, ImageDesc),
        before: Option<Before>,
        shape: impl Fn(&ImageDesc) -> bool,
    ) -> Option<(vk::Image, ImageDesc)> {
        let copies = self.copies.get(&cb).map_or(&[][..], Vec::as_slice);
        let copies = &copies[..before.map_or(copies.len(), |b| b.copies).min(copies.len())];
        let (src, _) = copies.iter().rev().find(|(src, dst)| *dst == named.0 && *src != named.0)?;
        self.images.get(src).filter(|d| shape(d)).map(|d| (*src, *d))
    }

    /// A first-of-its-kind classification line, once per kind: a launch-bearing submit held because a buffer of it
    /// carries DLSS Super Resolution's evaluate, and one forwarded because none does.
    fn classify_line(&mut self, scan: Option<&Scan>) -> Option<String> {
        let (bit, line) = if scan.is_some() && self.said_kinds & 1 == 0 {
            (1, "a launch-bearing submit carries DLSS Super Resolution's evaluate (a launch names the colour input of NGX's record): the hold point")
        } else if scan.is_none() && self.foreign_submits > 0 && self.said_kinds & 2 == 0 {
            (2, "a launch-bearing submit without DLSS Super Resolution's evaluate (DLSS Frame Generation's, Ray Reconstruction's, or launches no record names) is forwarded untouched")
        } else {
            return None;
        };
        self.said_kinds |= bit;
        Some(line.to_string())
    }
}

/// A device's [`Tracker`] plus lock-free flags, so that on a device with NVX but no DLSS (a
/// vkd3d-proton game without DLSS) the per-command-buffer and per-submit hooks cost one relaxed
/// load: `watching` (an input is identified, so barriers on it are worth recording), `armed` (some
/// command buffer carries a launch or recorded layouts, so begin/free/execute/submit have something
/// to do) and the identified extent for the present hook.
#[derive(Default)]
pub(crate) struct Tracking {
    tracker: Mutex<Tracker>,
    watching: AtomicBool,
    armed: AtomicBool,
    /// The identified colour input's extent, `width << 32 | height`; 0 when none.
    extent: AtomicU64,
    /// The device's handle, named on the identification lines (DLSS and the swapchain may be on
    /// different devices).
    device: u64,
    /// The pool each of the device's command buffers came from: a pool reset or destroyed ends all of
    /// its buffers ([`Self::reset_pool`]). Apart from the tracker's lock (allocations are not
    /// launches).
    pools: Mutex<HashMap<vk::CommandBuffer, vk::CommandPool>>,
}

static TRACKING: LazyLock<Mutex<HashMap<vk::Device, Arc<Tracking>>>> = LazyLock::new(Default::default);

impl Tracking {
    /// [`Tracker::waits_on_event`].
    pub(crate) fn waits_on_event(&self, cbs: &[vk::CommandBuffer]) -> bool {
        self.lock().waits_on_event(cbs)
    }

    /// A new tracker for `device`, also reachable through [`tracking_for`] (the
    /// `vkGetImageViewHandle64NVX` wrapper has only the device handle).
    pub(crate) fn new_for(device: vk::Device) -> Arc<Self> {
        let tracking = Arc::new(Self { device: device.as_raw(), ..Self::default() });
        lock(&TRACKING).insert(device, tracking.clone());
        // A Proton copy older than its source says so in the log, once.
        crate::proton_notice::note_once();
        // Model mode: whether the game uses DLSS, known before DLSS's first frame ([`dlss_game`]).
        if mode() == Mode::Model {
            dlss_game::watch();
            // NGX's records, from the moment the wrapper makes its ring ([`crate::ngx_ring::watch`]).
            crate::ngx_ring::watch();
        }
        tracking
    }

    /// A hold after DLSS's output was recorded ([`Tracker::after_recorded`]).
    pub(crate) fn after_recorded(&self) {
        self.lock().after_recorded();
    }

    /// A hold after DLSS's output could not be recorded ([`Tracker::after_missed`]).
    pub(crate) fn after_missed(&self) {
        self.lock().after_missed();
    }

    /// The split hold's plan ([`placement`]): `None` when no pass runs before the upscaler (the frame goes
    /// to DLSS untouched), else the hold for the passes that do, with DLSS's render size `render`.
    pub(crate) fn split_hold(&self, render: (u32, u32)) -> Option<placement::Hold> {
        let t = self.lock();
        let plan = placement::current(t.frame());
        plans_before(&plan).then(|| placement::Hold::of(&plan, placement::Stage::Before, render, t.output_extent()))
    }

    /// What a hold recorded into `cb` now, before the upscaler, is given of DLSS's frame: the camera jitter and the
    /// game's reset, from NGX's record of the buffer's evaluate (written before any of its launches).
    pub(crate) fn dlss_frame_of(&self, cb: vk::CommandBuffer) -> DlssFrame {
        self.lock().frame_of(cb)
    }

    /// The same for a hold after DLSS's output in `cb`: DLSS's output is not jittered (its jitter is zero, known,
    /// so the edit cache keeps its interval there), and the reset is the evaluate's.
    pub(crate) fn dlss_frame_after(&self, cb: vk::CommandBuffer) -> DlssFrame {
        DlssFrame { jitter: Some([0.0, 0.0]), reset: self.lock().frame_of(cb).reset }
    }

    /// Whether `cb` has CUDA launches recorded (diagnostics: `NEURAL_FORGE_LOG_LAUNCH_QUEUES`).
    pub(crate) fn has_launches(&self, cb: vk::CommandBuffer) -> bool {
        self.lock().launch.contains_key(&cb)
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Tracker> {
        lock(&self.tracker)
    }

    /// Whether any input is identified (so barriers on it are worth recording).
    pub(crate) fn watching(&self) -> bool {
        self.watching.load(Ordering::Relaxed)
    }

    /// Whether any command buffer carries a launch or recorded layouts. Stored under the lock
    /// after every change; read without it. A command buffer's own recording happens-before its
    /// reset, free or submit (the application synchronizes those), so a `false` read for a buffer
    /// that was marked is impossible.
    pub(crate) fn armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    fn rearm(&self, t: &Tracker) {
        self.armed.store(t.armed(), Ordering::Relaxed);
    }

    /// `vkCmdCuLaunchKernelNVX` recorded into `command_buffer`, with its parameter buffer if
    /// readable ([`launch_params`]).
    /// A launch recorded into `command_buffer`. Returns where a hold should run inside the buffer,
    /// right before this launch, when it is the buffer's first to name a colour candidate and the
    /// split in front of the buffer would be refused ([`Tracker::inline_point`]).
    #[cfg(test)]
    pub(crate) fn launch(&self, command_buffer: vk::CommandBuffer, function: vk::CuFunctionNVX, params: Option<&[u8]>) -> Option<InlinePoint> {
        self.launch_with_after(command_buffer, function, params).0
    }

    /// [`Self::launch`], and the hold right after this launch when it is DLSS's output kernel and the
    /// model runs after DLSS ([`Tracker::after_point`]).
    /// The last launch's kernel name and named images with their descriptions ([`probe_image`]).
    pub(crate) fn last_launch(&self) -> Option<(String, Vec<(vk::Image, ImageDesc)>)> {
        let t = self.lock();
        let (name, named) = t.last_launch.clone()?;
        Some((name, named.iter().filter_map(|i| t.images.get(i).map(|d| (*i, *d))).collect()))
    }

    pub(crate) fn launch_with_after(
        &self,
        command_buffer: vk::CommandBuffer,
        function: vk::CuFunctionNVX,
        params: Option<&[u8]>,
    ) -> (Option<InlinePoint>, Option<InlinePoint>) {
        let mut t = self.lock();
        t.launch_kernel(command_buffer, function, params);
        self.rearm(&t);
        let after = t.after_point();
        let (before, said) = Self::before_point(&mut t);
        // Logged (and flushed, up to `logging::FLUSH_WAIT`) with the lock let go: other threads' launches wait on it.
        drop(t);
        if let Some(said) = said {
            said.say();
        }
        (before, after)
    }

    /// The hold point inside DLSS's buffer, if any, and a refusal's line to log once the lock is let go.
    fn before_point(t: &mut Tracker) -> (Option<InlinePoint>, Option<Said>) {
        let Some((cb, record)) = t.take_inline_at() else { return (None, None) };
        // No pass runs before the upscaler (they run after DLSS or at present): no hold here.
        let plan = placement::current(t.frame());
        if !plans_before(&plan) {
            return (None, None);
        }
        t.inline_tried.insert(cb);
        let output = t.output_extent();
        let point = t
            .inline_point(cb, &record)
            .map(|p| InlinePoint { hold: placement::Hold::of(&plan, placement::Stage::Before, (p.desc.width, p.desc.height), output), ..p });
        match point {
            Ok(point) => (Some(point), None),
            Err(refusal) => (None, t.note_inline_refusal(record.colour, refusal)),
        }
    }

    /// The split refused launch buffer `cb` for a hazard: logs why no hold inside it was decided on either, once per
    /// reason ([`Tracker::untried_line`]), with the lock let go.
    pub(crate) fn say_inside_untried(&self, cb: vk::CommandBuffer) {
        let line = self.lock().untried_line(cb);
        if let Some(line) = line {
            Said { line, flush: true }.say();
        }
    }

    /// `vkCreateCuFunctionNVX` created `function` for the kernel `name`.
    pub(crate) fn record_function(&self, function: vk::CuFunctionNVX, name: &str) {
        if dlss_game::dlss_kernel_name(name) {
            dlss_game::note(&format!("its CUDA kernel {name} was created"));
        }
        self.lock().record_function(function, name);
    }

    /// [`Tracker::inline_recorded`].
    pub(crate) fn inline_recorded(&self, identification: u64) {
        self.lock().inline_recorded(identification);
    }

    pub(crate) fn forget_function(&self, function: vk::CuFunctionNVX) {
        self.lock().forget_function(function);
    }

    /// `vkBeginCommandBuffer`: forgets the buffer's earlier recording, and while inputs are watched
    /// notes that the new one is seen whole ([`Tracker::recorded_watched`]). Nothing to do (no lock)
    /// while nothing is armed or watched: a buffer begun then is not in that set, and none of it is.
    pub(crate) fn begin(&self, command_buffer: vk::CommandBuffer) {
        if self.armed() || self.watching() {
            let mut t = self.lock();
            t.begin(command_buffer);
            self.rearm(&t);
        }
    }

    /// `vkFreeCommandBuffers`, as [`Self::begin`].
    pub(crate) fn free(&self, command_buffers: &[vk::CommandBuffer]) {
        {
            let mut pools = lock(&self.pools);
            for cb in command_buffers {
                pools.remove(cb);
            }
        }
        if self.armed() {
            let mut t = self.lock();
            t.free(command_buffers);
            self.rearm(&t);
        }
    }

    /// `vkAllocateCommandBuffers` returned `command_buffers` from `pool`.
    pub(crate) fn allocated(&self, pool: vk::CommandPool, command_buffers: &[vk::CommandBuffer]) {
        lock(&self.pools).extend(command_buffers.iter().map(|&cb| (cb, pool)));
    }

    /// `vkResetCommandBuffer`: what was recorded is gone, as at [`Self::begin`].
    pub(crate) fn reset(&self, command_buffer: vk::CommandBuffer) {
        self.begin(command_buffer);
    }

    /// `vkResetCommandPool` (`destroyed`: `vkDestroyCommandPool`): every buffer of the pool is reset
    /// (or gone), so what was tracked for them is dropped and the tracking disarms when nothing is
    /// left; before, their launches and barriers stayed (and the tracking armed) until each buffer
    /// was begun or freed again.
    pub(crate) fn reset_pool(&self, pool: vk::CommandPool, destroyed: bool) {
        let buffers: Vec<vk::CommandBuffer> = {
            let mut pools = lock(&self.pools);
            let of_pool = pools.iter().filter(|(_, p)| **p == pool).map(|(cb, _)| *cb).collect();
            if destroyed {
                pools.retain(|_, p| *p != pool);
            }
            of_pool
        };
        if self.armed() && !buffers.is_empty() {
            let mut t = self.lock();
            t.free(&buffers);
            self.rearm(&t);
        }
    }

    /// `vkCmdExecuteCommands`: a secondary's launch or layouts carry over to the primary. Nothing
    /// to carry while nothing is armed.
    pub(crate) fn execute(&self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        if self.armed() {
            let mut t = self.lock();
            t.execute(primary, secondaries);
            self.rearm(&t);
        }
    }

    /// Records the image barriers on watched images, and `global` synchronization recorded with
    /// them (a memory barrier with a write in its source access, an event wait).
    pub(crate) fn barriers(&self, command_buffer: vk::CommandBuffer, images: impl Iterator<Item = ImageSync>, global: Option<Hazard>) {
        let mut t = self.lock();
        for sync in images {
            t.barrier(command_buffer, sync);
        }
        if let Some(hazard) = global {
            t.global(command_buffer, hazard);
        }
        self.rearm(&t);
    }

    /// An image copy ([`Tracker::copy`]), while inputs are watched.
    pub(crate) fn copy(&self, command_buffer: vk::CommandBuffer, src: vk::Image, dst: vk::Image) {
        if !self.watching.load(Ordering::Relaxed) {
            return;
        }
        let mut t = self.lock();
        t.copy(command_buffer, src, dst);
        self.rearm(&t);
    }

    /// `vkCmdBeginRendering`'s attachments ([`Tracker::attachments`]), while inputs are watched.
    pub(crate) fn attachments(&self, command_buffer: vk::CommandBuffer, views: &[(vk::ImageView, vk::ImageLayout)]) {
        if !self.watching.load(Ordering::Relaxed) || views.is_empty() {
            return;
        }
        let mut t = self.lock();
        t.attachments(command_buffer, views);
        self.rearm(&t);
    }

    /// `vkCmdBeginRendering` with a suspending or resuming flag ([`Tracker::begin_rendering`]).
    pub(crate) fn begin_rendering(&self, command_buffer: vk::CommandBuffer, flags: vk::RenderingFlags) {
        if flags.intersects(vk::RenderingFlags::SUSPENDING | vk::RenderingFlags::RESUMING) {
            let mut t = self.lock();
            t.begin_rendering(command_buffer, flags);
            self.rearm(&t);
        }
    }

    /// See [`Tracker::commit_submitted`]. No lock while nothing is armed.
    pub(crate) fn commit_submitted(&self, command_buffers: impl IntoIterator<Item = vk::CommandBuffer>) {
        if self.armed() {
            self.lock().commit_submitted(command_buffers);
        }
    }

    /// See [`Tracker::scan`]; re-derives the inputs first and logs a change. `None` without taking
    /// the lock while nothing is armed.
    pub(crate) fn scan(&self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        if !self.armed() {
            return None;
        }
        let (scan, identified, line) = {
            let mut t = self.lock();
            if !t.armed() {
                return None;
            }
            let device = self.device;
            let observed = t.observe(batches);
            let line = t.refresh().map(|l| format!("{l} [device {device:#x}]"));
            let identified = line.is_some().then(|| t.inputs.map(|i| i.colour.0));
            self.watching.store(t.inputs.is_some() || t.colour_watch.is_some(), Ordering::Relaxed);
            self.extent.store(t.inputs.map_or(0, |i| u64::from(i.colour.1.width) << 32 | u64::from(i.colour.1.height)), Ordering::Relaxed);
            let foreign_before = t.foreign_submits;
            let scan = t.scan(batches);
            let kind_line = t.classify_line(scan.as_ref());
            // Every 3000th forwarded one: the running counts.
            let tally = (t.foreign_submits != foreign_before && t.foreign_submits.is_multiple_of(3000)).then(|| {
                format!(
                    "launch-bearing submits: {} with DLSS Super Resolution's evaluate, {} forwarded untouched; holds: {} split, {} inside DLSS's buffer",
                    t.evaluate_submits,
                    t.foreign_submits,
                    HOLDS_SPLIT.load(Ordering::Relaxed),
                    HOLDS_INSIDE.load(Ordering::Relaxed),
                )
            });
            self.rearm(&t);
            (scan, identified, observed.into_iter().map(Some).chain([line, kind_line, tally]))
        };
        for line in line.flatten() {
            crate::log!("[preupscale] {line}");
            crate::logging::flush();
        }
        if let Some(colour) = identified {
            crate::probe_ngx::note_identified_colour(colour);
        }
        scan
    }

    /// [`Self::scan`] followed by the commit of a submission the next layer accepted whole.
    #[cfg(test)]
    pub(crate) fn submit_ok(&self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        let scan = self.scan(batches);
        self.commit_submitted(batches.iter().flatten().copied());
        scan
    }

    /// The identified colour input's extent, if any (as of the last launch-bearing submit; no lock).
    pub(crate) fn extent(&self) -> Option<(u32, u32)> {
        let packed = self.extent.load(Ordering::Relaxed);
        (packed != 0).then_some(((packed >> 32) as u32, packed as u32))
    }
}

pub(crate) fn tracking_for(device: vk::Device) -> Option<Arc<Tracking>> {
    lock(&TRACKING).get(&device).cloned()
}

pub(crate) fn forget_device(device: vk::Device) {
    lock(&TRACKING).remove(&device);
}

// ---- Batch splitting. ----

/// One batch of a `vkQueueSubmit` call, owned: wait semaphores with their stage and (timeline)
/// value, command buffers, signal semaphores with their value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Wait1 {
    pub semaphore: vk::Semaphore,
    pub stage: vk::PipelineStageFlags,
    pub value: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Signal1 {
    pub semaphore: vk::Semaphore,
    pub value: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Batch1 {
    pub waits: Vec<Wait1>,
    pub cbs: Vec<vk::CommandBuffer>,
    pub signals: Vec<Signal1>,
}

/// One batch of a `vkQueueSubmit2` call, its arrays copied (`'a`: the call's submit infos, whose
/// `pNext` chains the copies keep). `p_next` is the application's chain, kept on every part (only
/// structures safe to repeat are accepted, see [`parse2`]).
#[derive(Clone, Debug)]
pub(crate) struct Batch2<'a> {
    pub flags: vk::SubmitFlags,
    pub p_next: *const c_void,
    pub waits: Vec<vk::SemaphoreSubmitInfo<'a>>,
    pub cbs: Vec<vk::CommandBufferSubmitInfo<'a>>,
    pub signals: Vec<vk::SemaphoreSubmitInfo<'a>>,
}

/// What [`plan`] needs from a batch type.
pub(crate) trait SplitBatch: Sized {
    type Wait: Clone;
    /// `(prefix, suffix)`: the prefix keeps the waits and the command buffers before `k`; the
    /// suffix has the command buffers from `k` on and the signals.
    fn split_at(self, k: usize) -> (Self, Self);
    fn take_waits(&mut self) -> Vec<Self::Wait>;
    fn put_waits(&mut self, waits: Vec<Self::Wait>);
    /// The layer's own batch: `waits` (moved from the game's launch batch, stages widened to
    /// `ALL_COMMANDS`) and one command buffer.
    fn own(waits: Vec<Self::Wait>, command_buffer: vk::CommandBuffer) -> Self;
    /// The batch's command buffers, in order.
    fn command_buffers(&self) -> Vec<vk::CommandBuffer>;
    /// A wait's semaphore and the value it waits for (a timeline semaphore's; ignored for a binary one).
    fn wait_value(wait: &Self::Wait) -> (vk::Semaphore, u64);
    /// The batch's own waits.
    fn waits(&self) -> &[Self::Wait];
}

impl SplitBatch for Batch1 {
    type Wait = Wait1;
    fn split_at(mut self, k: usize) -> (Self, Self) {
        let rest = self.cbs.split_off(k);
        let suffix = Batch1 { waits: Vec::new(), cbs: rest, signals: std::mem::take(&mut self.signals) };
        (self, suffix)
    }
    fn take_waits(&mut self) -> Vec<Wait1> {
        std::mem::take(&mut self.waits)
    }
    fn put_waits(&mut self, waits: Vec<Wait1>) {
        self.waits.splice(0..0, waits);
    }
    fn own(waits: Vec<Wait1>, command_buffer: vk::CommandBuffer) -> Self {
        let waits = waits.into_iter().map(|w| Wait1 { stage: vk::PipelineStageFlags::ALL_COMMANDS, ..w }).collect();
        Batch1 { waits, cbs: vec![command_buffer], signals: Vec::new() }
    }
    fn command_buffers(&self) -> Vec<vk::CommandBuffer> {
        self.cbs.clone()
    }
    fn wait_value(wait: &Wait1) -> (vk::Semaphore, u64) {
        (wait.semaphore, wait.value.unwrap_or(0))
    }
    fn waits(&self) -> &[Wait1] {
        &self.waits
    }
}

impl<'a> SplitBatch for Batch2<'a> {
    type Wait = vk::SemaphoreSubmitInfo<'a>;
    fn split_at(mut self, k: usize) -> (Self, Self) {
        let rest = self.cbs.split_off(k);
        let suffix = Batch2 { flags: self.flags, p_next: self.p_next, waits: Vec::new(), cbs: rest, signals: std::mem::take(&mut self.signals) };
        (self, suffix)
    }
    fn take_waits(&mut self) -> Vec<vk::SemaphoreSubmitInfo<'a>> {
        std::mem::take(&mut self.waits)
    }
    fn put_waits(&mut self, waits: Vec<vk::SemaphoreSubmitInfo<'a>>) {
        self.waits.splice(0..0, waits);
    }
    fn own(waits: Vec<vk::SemaphoreSubmitInfo<'a>>, command_buffer: vk::CommandBuffer) -> Self {
        let waits = waits.into_iter().map(|w| vk::SemaphoreSubmitInfo { stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS, ..w }).collect();
        let cb = vk::CommandBufferSubmitInfo { command_buffer, ..Default::default() };
        Batch2 { flags: vk::SubmitFlags::empty(), p_next: std::ptr::null(), waits, cbs: vec![cb], signals: Vec::new() }
    }
    fn command_buffers(&self) -> Vec<vk::CommandBuffer> {
        self.cbs.iter().map(|c| c.command_buffer).collect()
    }
    fn wait_value(wait: &vk::SemaphoreSubmitInfo<'a>) -> (vk::Semaphore, u64) {
        (wait.semaphore, wait.value)
    }
    fn waits(&self) -> &[vk::SemaphoreSubmitInfo<'a>] {
        &self.waits
    }
}

/// A submit split around its launch buffer. See the module doc comment for the order.
#[derive(Debug)]
pub(crate) struct Plan<B: SplitBatch> {
    /// Submitted before the hold, without the fence (may be empty).
    pub head: Vec<B>,
    /// The launch batch's wait semaphores when the launch buffer is first in its batch: they move
    /// onto the layer's capture batch.
    pub capture_waits: Vec<B::Wait>,
    /// Submitted after the hold, with the fence. `tail[0]` starts with the launch buffer.
    pub tail: Vec<B>,
}

impl<B: SplitBatch> Plan<B> {
    /// Puts the moved waits back on the launch batch: the capture batch was never submitted.
    pub(crate) fn restore_waits(&mut self) {
        let waits = std::mem::take(&mut self.capture_waits);
        if !waits.is_empty() {
            self.tail[0].put_waits(waits);
        }
    }
}

/// Splits `batches` at command buffer `index` of batch `batch` (the launch buffer). Batches before
/// it go to `head` unchanged; if the launch buffer is not first in its batch, the buffers before it
/// go to `head` as their own batch carrying the original waits, and the rest (with the signals)
/// starts `tail`; if it is first, the waits move to `capture_waits`. Batches after it follow in
/// `tail` unchanged.
pub(crate) fn plan<B: SplitBatch>(mut batches: Vec<B>, batch: usize, index: usize) -> Plan<B> {
    let mut tail = batches.split_off(batch);
    let mut head = batches;
    let mut launch = tail.remove(0);
    let capture_waits = if index > 0 {
        let (prefix, suffix) = launch.split_at(index);
        head.push(prefix);
        launch = suffix;
        Vec::new()
    } else {
        launch.take_waits()
    };
    tail.insert(0, launch);
    Plan { head, capture_waits, tail }
}

/// Submits a split call around a hold: `head` without a fence (skipped when empty), then `hold`,
/// which makes the layer's own submissions and is handed the moved wait semaphores (it returns
/// whether its capture batch consumed them), then `tail` with the application's `fence` -- the
/// fence always goes on the last submission. Moved waits that were not consumed go back onto the
/// launch batch. A failed `head` submission is returned at once (nothing else was submitted).
///
/// `accepted` is called with the application's command buffers of each part the next layer
/// accepted (`VK_SUCCESS`), in submission order: `head`'s once it was submitted, `tail`'s once it
/// was. Tracked state advances from those only ([`Tracker::commit_submitted`]): after a failed
/// `head` nothing is reported, after a failed `tail` only `head`'s buffers are.
///
/// A failed `tail` comes after work of the call went to the queue: `head`, and the hold's own
/// submissions, which may have consumed the launch batch's waits and written the colour input. The
/// application's `vkQueueSubmit` is all or nothing: on `VK_ERROR_OUT_OF_HOST_MEMORY` or
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY` it takes nothing to have executed and no semaphore to have been
/// waited on, and may submit the same buffers again. The specification's answer for an
/// implementation that cannot keep that promise is `VK_ERROR_DEVICE_LOST` ("If vkQueueSubmit fails
/// in such a way that the implementation is unable to make that guarantee, the implementation must
/// return VK_ERROR_DEVICE_LOST"). A failed submission changes nothing ("the state and contents of
/// any resources or synchronization primitives ... is unaffected"), so the tail is first submitted
/// once more (an out-of-memory error can be transient); only when that fails too is the call
/// reported as `VK_ERROR_DEVICE_LOST`. A tail that failed with `VK_ERROR_DEVICE_LOST` is returned
/// as it is.
pub(crate) fn submit_around<B: SplitBatch>(
    mut plan: Plan<B>,
    fence: vk::Fence,
    submit: &dyn Fn(&[B], vk::Fence) -> vk::Result,
    hold: impl FnOnce(&[B::Wait]) -> bool,
    accepted: &mut dyn FnMut(Vec<vk::CommandBuffer>),
) -> vk::Result {
    let buffers = |part: &[B]| part.iter().flat_map(B::command_buffers).collect::<Vec<_>>();
    if !plan.head.is_empty() {
        let result = submit(&plan.head, vk::Fence::null());
        if result != vk::Result::SUCCESS {
            return result;
        }
        accepted(buffers(&plan.head));
    }
    if !hold(&plan.capture_waits) {
        plan.restore_waits();
    }
    let mut result = submit(&plan.tail, fence);
    if result != vk::Result::SUCCESS && result != vk::Result::ERROR_DEVICE_LOST {
        let first = result;
        result = submit(&plan.tail, fence);
        static SAID: AtomicU64 = AtomicU64::new(0);
        if SAID.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::log!(
                "[preupscale] the held submit's last part (DLSS's buffer onward) failed with {first:?} after its earlier parts went to the queue; submitted again: {}",
                if result == vk::Result::SUCCESS {
                    "accepted".to_string()
                } else {
                    format!("{result:?}, reported to the application as VK_ERROR_DEVICE_LOST (the call can no longer be undone)")
                }
            );
            crate::logging::flush();
        }
        if result != vk::Result::SUCCESS {
            result = vk::Result::ERROR_DEVICE_LOST;
        }
    }
    if result == vk::Result::SUCCESS {
        accepted(buffers(&plan.tail));
    }
    result
}

/// # Safety
/// `len` elements at `ptr` must be valid when `len > 0` and `ptr` non-null.
unsafe fn slice<'a, T>(ptr: *const T, len: u32) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: forwarded from this function's contract.
        unsafe { std::slice::from_raw_parts(ptr, len as usize) }
    }
}

/// The application's `VkSubmitInfo`s as owned batches. Only `VkTimelineSemaphoreSubmitInfo` is
/// accepted in a pNext chain (it is rebuilt per part); anything else is refused, and the caller
/// forwards the call untouched.
///
/// # Safety
/// `submits` must be the application's own, valid for the duration of its call.
pub(crate) unsafe fn parse1(submits: &[vk::SubmitInfo]) -> Result<Vec<Batch1>, &'static str> {
    let mut out = Vec::with_capacity(submits.len());
    for s in submits {
        let mut timeline: Option<&vk::TimelineSemaphoreSubmitInfo> = None;
        let mut next = s.p_next.cast::<vk::BaseInStructure>();
        // SAFETY: a valid pNext chain of the application's own structure.
        while let Some(base) = unsafe { next.as_ref() } {
            if base.s_type == vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO && timeline.is_none() {
                // SAFETY: the sType says what it is.
                timeline = Some(unsafe { &*next.cast::<vk::TimelineSemaphoreSubmitInfo>() });
            } else {
                return Err("a VkSubmitInfo pNext structure other than VkTimelineSemaphoreSubmitInfo");
            }
            next = base.p_next;
        }
        // SAFETY: counts and pointers of the application's own structure.
        let (waits, stages, cbs, signals) = unsafe {
            (
                slice(s.p_wait_semaphores, s.wait_semaphore_count),
                slice(s.p_wait_dst_stage_mask, s.wait_semaphore_count),
                slice(s.p_command_buffers, s.command_buffer_count),
                slice(s.p_signal_semaphores, s.signal_semaphore_count),
            )
        };
        let values = |count: u32, ptr: *const u64, n: u32| -> Result<Option<&[u64]>, &'static str> {
            if count == 0 {
                Ok(None)
            } else if count == n && !ptr.is_null() {
                // SAFETY: `count` values at `ptr`, the application's own.
                Ok(Some(unsafe { slice(ptr, count) }))
            } else {
                Err("a timeline value count that does not match its semaphore count")
            }
        };
        let (wait_values, signal_values) = match timeline {
            Some(t) => (
                values(t.wait_semaphore_value_count, t.p_wait_semaphore_values, s.wait_semaphore_count)?,
                values(t.signal_semaphore_value_count, t.p_signal_semaphore_values, s.signal_semaphore_count)?,
            ),
            None => (None, None),
        };
        if stages.len() != waits.len() {
            return Err("wait semaphores without stage masks");
        }
        out.push(Batch1 {
            waits: waits.iter().zip(stages).enumerate().map(|(i, (&semaphore, &stage))| Wait1 { semaphore, stage, value: wait_values.map(|v| v[i]) }).collect(),
            cbs: cbs.to_vec(),
            signals: signals.iter().enumerate().map(|(i, &semaphore)| Signal1 { semaphore, value: signal_values.map(|v| v[i]) }).collect(),
        });
    }
    Ok(out)
}

/// `VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV` (`VK_NV_low_latency2`, newer than the pinned
/// ash): a present id tag, safe on every part of a split batch.
const LATENCY_SUBMISSION_PRESENT_ID_NV: vk::StructureType = vk::StructureType::from_raw(1_000_505_005);
/// `VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT` (`VK_EXT_frame_boundary`): marks the submit that ends a
/// frame (`VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT`). Not something to repeat on both halves of a split
/// batch, and the layer's own submissions would land on the wrong side of it: refused.
const FRAME_BOUNDARY_EXT: vk::StructureType = vk::StructureType::from_raw(1_000_375_001);

/// The application's `VkSubmitInfo2`s as owned batches. The pNext chain is kept as is on every part
/// of a split batch, so only structures that mean the same thing repeated are accepted (a latency
/// present id, a performance-query pass index); anything else is refused, `VkFrameBoundaryEXT`
/// among them. A call with a protected batch (`VK_SUBMIT_PROTECTED_BIT`) is refused as well: the
/// layer's own batches and command pool are unprotected, and protected content is not its to read.
/// The caller forwards a refused call untouched.
///
/// # Safety
/// `submits` must be the application's own, valid for the duration of its call.
pub(crate) unsafe fn parse2<'a>(submits: &[vk::SubmitInfo2<'a>]) -> Result<Vec<Batch2<'a>>, &'static str> {
    let mut out = Vec::with_capacity(submits.len());
    for s in submits {
        if s.flags.contains(vk::SubmitFlags::PROTECTED) {
            return Err("a protected VkSubmitInfo2 (VK_SUBMIT_PROTECTED_BIT): protected submits are never held");
        }
        let mut next = s.p_next.cast::<vk::BaseInStructure>();
        // SAFETY: a valid pNext chain of the application's own structure.
        while let Some(base) = unsafe { next.as_ref() } {
            if base.s_type == FRAME_BOUNDARY_EXT {
                return Err("a VkSubmitInfo2 with VkFrameBoundaryEXT: a frame boundary is never split or pushed behind the layer's own submits");
            }
            if !matches!(base.s_type, LATENCY_SUBMISSION_PRESENT_ID_NV | vk::StructureType::PERFORMANCE_QUERY_SUBMIT_INFO_KHR) {
                return Err("a VkSubmitInfo2 pNext structure that cannot be repeated on a split batch");
            }
            next = base.p_next;
        }
        // SAFETY: counts and pointers of the application's own structure.
        unsafe {
            out.push(Batch2 {
                flags: s.flags,
                p_next: s.p_next,
                waits: slice(s.p_wait_semaphore_infos, s.wait_semaphore_info_count).to_vec(),
                cbs: slice(s.p_command_buffer_infos, s.command_buffer_info_count).to_vec(),
                signals: slice(s.p_signal_semaphore_infos, s.signal_semaphore_info_count).to_vec(),
            });
        }
    }
    Ok(out)
}

/// `VkSubmitInfo`s (and the timeline structures and arrays they point at) built from owned batches.
/// The arrays live on the heap, so moving this value keeps every pointer valid.
pub(crate) struct Submits1 {
    pub infos: Vec<vk::SubmitInfo<'static>>,
    _timelines: Vec<vk::TimelineSemaphoreSubmitInfo<'static>>,
    _storage: Vec<Storage1>,
}

struct Storage1 {
    waits: Vec<vk::Semaphore>,
    stages: Vec<vk::PipelineStageFlags>,
    wait_values: Vec<u64>,
    cbs: Vec<vk::CommandBuffer>,
    signals: Vec<vk::Semaphore>,
    signal_values: Vec<u64>,
    timeline: bool,
}

pub(crate) fn build1(batches: &[Batch1]) -> Submits1 {
    let storage: Vec<Storage1> = batches
        .iter()
        .map(|b| {
            let wait_values: Vec<u64> = b.waits.iter().filter_map(|w| w.value).collect();
            let signal_values: Vec<u64> = b.signals.iter().filter_map(|s| s.value).collect();
            Storage1 {
                waits: b.waits.iter().map(|w| w.semaphore).collect(),
                stages: b.waits.iter().map(|w| w.stage).collect(),
                timeline: !wait_values.is_empty() || !signal_values.is_empty(),
                wait_values,
                cbs: b.cbs.clone(),
                signals: b.signals.iter().map(|s| s.semaphore).collect(),
                signal_values,
            }
        })
        .collect();
    let timelines: Vec<vk::TimelineSemaphoreSubmitInfo> = storage
        .iter()
        .map(|s| vk::TimelineSemaphoreSubmitInfo {
            wait_semaphore_value_count: s.wait_values.len() as u32,
            p_wait_semaphore_values: s.wait_values.as_ptr(),
            signal_semaphore_value_count: s.signal_values.len() as u32,
            p_signal_semaphore_values: s.signal_values.as_ptr(),
            ..Default::default()
        })
        .collect();
    let infos = storage
        .iter()
        .zip(&timelines)
        .map(|(s, t)| vk::SubmitInfo {
            p_next: if s.timeline { std::ptr::from_ref(t).cast() } else { std::ptr::null() },
            wait_semaphore_count: s.waits.len() as u32,
            p_wait_semaphores: s.waits.as_ptr(),
            p_wait_dst_stage_mask: s.stages.as_ptr(),
            command_buffer_count: s.cbs.len() as u32,
            p_command_buffers: s.cbs.as_ptr(),
            signal_semaphore_count: s.signals.len() as u32,
            p_signal_semaphores: s.signals.as_ptr(),
            ..Default::default()
        })
        .collect();
    Submits1 { infos, _timelines: timelines, _storage: storage }
}

/// `VkSubmitInfo2`s pointing into `batches`, which must outlive the result's use.
pub(crate) fn build2<'b>(batches: &'b [Batch2<'_>]) -> Vec<vk::SubmitInfo2<'b>> {
    batches
        .iter()
        .map(|b| vk::SubmitInfo2 {
            p_next: b.p_next,
            flags: b.flags,
            wait_semaphore_info_count: b.waits.len() as u32,
            p_wait_semaphore_infos: b.waits.as_ptr(),
            command_buffer_info_count: b.cbs.len() as u32,
            p_command_buffer_infos: b.cbs.as_ptr(),
            signal_semaphore_info_count: b.signals.len() as u32,
            p_signal_semaphore_infos: b.signals.as_ptr(),
            ..Default::default()
        })
        .collect()
}

// ---- The hold itself: capture, round trip, write-back. ----

/// `(width, height)` rounded up to even: the model needs even sizes. The padding column
/// and row duplicate the edge.
pub(crate) fn padded(width: u32, height: u32) -> (u32, u32) {
    (width + (width & 1), height + (height & 1))
}

fn colour_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }
}

/// The copy regions that capture a `width` x `height` RGBA16F image into a buffer laid out at the
/// padded size, tightly packed: the image itself, then the last column into the padding column,
/// the last row into the padding row, and the corner.
pub(crate) fn capture_regions(width: u32, height: u32) -> Vec<vk::BufferImageCopy> {
    let (pw, ph) = padded(width, height);
    let region = |x: u32, y: u32, w: u32, h: u32, dx: u32, dy: u32| vk::BufferImageCopy {
        buffer_offset: (u64::from(dy) * u64::from(pw) + u64::from(dx)) * TEXEL,
        buffer_row_length: pw,
        buffer_image_height: 0,
        image_subresource: colour_layers(),
        image_offset: vk::Offset3D { x: x as i32, y: y as i32, z: 0 },
        image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
    };
    let mut regions = vec![region(0, 0, width, height, 0, 0)];
    if pw > width {
        regions.push(region(width - 1, 0, 1, height, pw - 1, 0));
    }
    if ph > height {
        regions.push(region(0, height - 1, width, 1, 0, ph - 1));
    }
    if pw > width && ph > height {
        regions.push(region(width - 1, height - 1, 1, 1, pw - 1, ph - 1));
    }
    regions
}

/// The write-back region: the padded buffer's top-left `width` x `height`, cropping the padding.
/// The colour input at its real extent, tightly packed (a cache dump's readbacks).
fn colour_region(width: u32, height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D { width, height, depth: 1 },
    }
}

pub(crate) fn writeback_region(width: u32, height: u32) -> vk::BufferImageCopy {
    let (pw, _) = padded(width, height);
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: pw,
        buffer_image_height: 0,
        image_subresource: colour_layers(),
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D { width, height, depth: 1 },
    }
}

/// A buffer the GPU copies into or out of, readable and writable by the CPU at `ptr`: either the
/// SHM region itself, imported (`VK_EXT_external_memory_host`, zero-copy), or a mapped host-visible
/// allocation of the layer's own that the CPU copies to or from the region (`staged`).
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    staged: bool,
}

impl HostBuffer {
    /// # Safety
    /// Nothing submitted may still use the buffer.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; freeing imported memory never unmaps the host pointer.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Allocates a mapped, host-visible, host-coherent `TRANSFER_SRC|TRANSFER_DST` buffer.
fn own_host_buffer(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, bytes: u64) -> Option<HostBuffer> {
    own_host_buffer_with(device, instance, physical_device, bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
}

/// [`own_host_buffer`] with the given usage.
fn own_host_buffer_with(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    bytes: u64,
    usage: vk::BufferUsageFlags,
) -> Option<HostBuffer> {
    let info = vk::BufferCreateInfo::default().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `info` is valid.
    let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
    // SAFETY: `buffer` was just created.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: plain property query.
    let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let pick = |extra: vk::MemoryPropertyFlags| {
        (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(wanted | extra))
    };
    let Some(type_index) = pick(vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| pick(vk::MemoryPropertyFlags::empty())) else {
        // SAFETY: nothing bound yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };
    let alloc = vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: valid allocation info.
    let Ok(memory) = (unsafe { device.allocate_memory(&alloc, None) }) else {
        // SAFETY: nothing bound yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };
    // SAFETY: freshly created, sized for each other; mapping a host-visible allocation.
    let mapped =
        unsafe { device.bind_buffer_memory(buffer, memory, 0).and_then(|()| device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())) };
    match mapped {
        Ok(ptr) => Some(HostBuffer { buffer, memory, ptr: ptr.cast(), staged: true }),
        Err(_) => {
            // SAFETY: nothing submitted uses either.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            None
        }
    }
}

/// Imports `bytes` (rounded up to the driver's alignment) of the SHM region at `region` when it can,
/// else allocates a staged buffer of `bytes`; either way usable for copies and for `usage` (the
/// after-the-upscaler path binds its regions as storage buffers).
///
/// # Safety
/// `region` must be valid for `capacity` bytes for the life of the result.
#[allow(clippy::too_many_arguments)]
unsafe fn region_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    import: bool,
    region: *mut u8,
    capacity: usize,
    bytes: u64,
    usage: vk::BufferUsageFlags,
) -> Option<HostBuffer> {
    if import {
        if let Some(alignment) = crate::capture::min_imported_host_pointer_alignment(instance, physical_device) {
            let rounded = bytes.div_ceil(alignment) * alignment;
            if (region as u64).is_multiple_of(alignment) && rounded <= capacity as u64 {
                // SAFETY: `region` is valid for `capacity >= rounded` bytes and aligned.
                if let Some((buffer, memory)) = unsafe { crate::capture::import_host_buffer(device, instance, physical_device, region, rounded, usage) } {
                    return Some(HostBuffer { buffer, memory, ptr: region, staged: false });
                }
            }
        }
    }
    own_host_buffer_with(device, instance, physical_device, bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | usage)
}

/// The next [`Resources::id`].
static NEXT_RESOURCES: AtomicU64 = AtomicU64::new(1);

/// What the holds recorded into DLSS's buffers left, once their buffers ran (`Session::harvest`), for the
/// next hold to report: as the split hold reports the previous hold's readings, read after its fence.
#[derive(Default)]
pub(crate) struct Readings {
    pub capture_gpu_ms: Option<f32>,
    pub network_gpu_ms: Option<f32>,
    pub writeback_gpu_ms: Option<f32>,
    pub cache_rejected: Option<f32>,
    /// A scene cut the GPU acted on, to count once.
    pub scene_cut: bool,
    /// The game's exposure a hold of identification `.0` read.
    pub previous_exposure: Option<(u64, f32)>,
    pub auto: Option<hdr::AutoState>,
}

thread_local! {
    /// While a hold is recorded into DLSS's buffer and other holds recorded with the same resources may still
    /// run, nothing those holds use may change: no descriptor set they bound written again, no buffer they read
    /// destroyed or written from the host ([`may_rewrite`]).
    static FROZEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The first change [`may_rewrite`] refused while frozen.
    static BLOCKED: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// Whether `what` (a descriptor set, a buffer the holds read) may be changed now. `false` while a hold is
/// recorded into DLSS's buffer with other holds of the same resources outstanding ([`Frozen`]): the caller
/// leaves it as it is, and the hold then records nothing ([`Frozen::blocked`]); it is changed at a hold
/// recorded once those have run.
pub(crate) fn may_rewrite(what: &'static str) -> bool {
    if FROZEN.get() {
        if BLOCKED.get().is_none() {
            BLOCKED.set(Some(what));
        }
        return false;
    }
    true
}

/// [`may_rewrite`]'s scope, for one hold on this thread.
pub(crate) struct Frozen;

impl Frozen {
    pub(crate) fn new(frozen: bool) -> Self {
        FROZEN.set(frozen);
        BLOCKED.set(None);
        Self
    }

    /// The first change refused, if any.
    pub(crate) fn blocked(&self) -> Option<&'static str> {
        BLOCKED.get()
    }
}

impl Drop for Frozen {
    fn drop(&mut self) {
        FROZEN.set(false);
        BLOCKED.set(None);
    }
}

/// Where a native hold's work goes ([`native::run_native_hold`]).
pub(crate) enum Sink<'a, 'b> {
    /// The split hold: the layer's own command buffers, submitted through this on the game's queue in front of
    /// DLSS's buffer.
    Queue(&'a mut dyn FnMut(Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>),
    /// The hold inside DLSS's buffer: recorded into it ([`inline`]).
    Buffer(&'a mut inline::Ctx<'b>),
}

/// Everything one device's holds use. Built for one extent and queue family; rebuilt (after
/// draining) when either changes.
pub(crate) struct Resources {
    /// Unique in the process: what a hold recorded into DLSS's buffer with these names (`inline::Meta`).
    pub(crate) id: u64,
    /// Built for the hold inside DLSS's buffer: the passes keep no timers of their own (the slot is timed),
    /// and the hold guard is built on its first hold.
    pub(crate) inline: bool,
    /// The hold inside DLSS's buffer: its guard (`preupscale_hold_guard.comp`), the sequence number of its
    /// last hold, and what the holds whose buffers ran left for the next hold to report.
    #[cfg(target_arch = "x86_64")]
    pub(crate) guard: Option<native::HoldGuard>,
    pub(crate) seq: u64,
    pub(crate) readings: Readings,
    queue_family: u32,
    width: u32,
    height: u32,
    pool: vk::CommandPool,
    capture_cmd: vk::CommandBuffer,
    writeback_cmd: vk::CommandBuffer,
    /// An empty buffer, recorded once: submitted with the write-back fence after a hold's later submit
    /// failed, so that fence covers what of the hold did go ([`Self::fence_submitted`]).
    fence_cmd: vk::CommandBuffer,
    capture_fence: vk::Fence,
    writeback_fence: vk::Fence,
    capture_timer: Option<crate::gpu_timer::GpuTimer>,
    writeback_timer: Option<crate::gpu_timer::GpuTimer>,
    /// Slot 0's proxy region (the capture's destination, identity's write-back source).
    proxy: HostBuffer,
    proxy_region: *mut u8,
    /// Dump-mode depth, motion-vector and exposure readbacks, built on first use.
    dump: Option<DumpBuffers>,
    /// The HDR encode and decode (model and roundtrip modes), built on first use.
    hdr: Option<hdr::HdrPass>,
    /// The native backend's frame resources (`native`), built on its first hold.
    #[cfg(target_arch = "x86_64")]
    native: Option<native::NativePass>,
    /// When the native hold builds its passes again after a failed build.
    #[cfg(target_arch = "x86_64")]
    native_retry: native::Retries,
    /// The identification of the last native hold whose capture read the game's exposure image into
    /// the HDR pass's exposure buffer ([`HoldResult::previous_exposure`]).
    #[cfg(target_arch = "x86_64")]
    pub(crate) game_exposure_read: Option<u64>,
    /// The identification of the last split native hold that recorded the auto-exposure, read back at the next
    /// hold ([`HoldResult::auto_exposure`]).
    #[cfg(target_arch = "x86_64")]
    pub(crate) auto_read: Option<u64>,
    /// DLSS's depth as the native hold copied it, built on the first hold with a readable one.
    #[cfg(target_arch = "x86_64")]
    pub(crate) depth: Option<native::DepthCopy>,
    /// The edit cache's passes (`cache`), built on the first native hold.
    #[cfg(target_arch = "x86_64")]
    pub(crate) edit: Option<cache::CachePass>,
    /// The scene-cut detector (`scene_cut`), built on the first native hold unless it is switched off.
    #[cfg(target_arch = "x86_64")]
    pub(crate) scene: Option<scene_cut::ScenePass>,
    /// A running cache dump's readbacks (`cache_dump`), built on its first armed hold and freed
    /// after it.
    pub(crate) cache: Option<cache_dump::Buffers>,
    /// The network whose pre-upscaler work these resources' fences cover, told when it was waited on
    /// (`native::Loader::pre_idle`): each time these resources go idle, and when they are destroyed, so the
    /// loader never keeps (and later asks the status of) a fence destroyed with them.
    #[cfg(target_arch = "x86_64")]
    pub(crate) pre_loader: Option<NativeLoader>,
    capture_pending: bool,
    pub(crate) writeback_pending: bool,
    /// A hold's work went to the queue and no fence could be put after it: nothing here is reused.
    unfenced: bool,
    /// The queue the holds' work goes to ([`Self::on_queue`]), and how many waits on their fences saw them
    /// signalled: such a fence's signal covers everything submitted on its queue before it, so resources left
    /// behind there ([`Session::recover_unfenced`]) are free once a later one was seen ([`Session::bury`]).
    queue: Option<vk::Queue>,
    fences_seen: u64,
}

// SAFETY: the raw pointers are the SHM mapping (process-lifetime) or the layer's own mapped
// allocations, only ever used behind the device's `State` mutex.
unsafe impl Send for Resources {}

impl Resources {
    /// # Safety
    /// `shm` must stay open (its regions mapped) for the life of the result, which it does: the
    /// mapping is never unmapped.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn build(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        shm: &ShmClient,
        import: bool,
    ) -> Option<Self> {
        let (pw, ph) = padded(width, height);
        let bytes = u64::from(pw) * u64::from(ph) * TEXEL;
        let (proxy_region, proxy_capacity) = shm.proxy_region(Slot::Primary)?;
        if bytes > proxy_capacity as u64 {
            return None;
        }
        // SAFETY: the region is mapped for the life of the process.
        let proxy = unsafe { region_buffer(device, instance, physical_device, import, proxy_region, proxy_capacity, bytes, vk::BufferUsageFlags::empty()) }?;
        // SAFETY: forwarded.
        unsafe { Self::build_with(device, instance, physical_device, queue_family, width, height, proxy, proxy_region, false) }
    }

    /// The resources of the hold inside DLSS's buffer: no shared memory (the proxy of the diagnostic modes is
    /// a buffer of the layer's, which only the GPU reads and writes there), no timers of the passes' own.
    ///
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn build_inline(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let (pw, ph) = padded(width, height);
        let proxy = own_host_buffer(device, instance, physical_device, u64::from(pw) * u64::from(ph) * TEXEL)?;
        let region = proxy.ptr;
        // SAFETY: forwarded.
        unsafe { Self::build_with(device, instance, physical_device, queue_family, width, height, proxy, region, true) }
    }

    /// # Safety
    /// `device` is live; `proxy_region` stays valid for the result's life.
    #[allow(clippy::too_many_arguments)]
    unsafe fn build_with(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        proxy: HostBuffer,
        proxy_region: *mut u8,
        inline: bool,
    ) -> Option<Self> {
        let cleanup = |pool: Option<vk::CommandPool>, fences: &[vk::Fence]| {
            // SAFETY: none of these was ever submitted.
            unsafe {
                proxy.destroy(device);
                for &f in fences {
                    device.destroy_fence(f, None);
                }
                if let Some(pool) = pool {
                    device.destroy_command_pool(pool, None);
                }
            }
        };
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid create info on a live device.
        let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else {
            cleanup(None, &[]);
            return None;
        };
        let alloc = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(3);
        // SAFETY: `pool` was just created; the loader's dispatch pointer is set on each buffer.
        let Ok(cmds) = (unsafe { crate::loader_data::allocate_commands(device, &alloc) }) else {
            cleanup(Some(pool), &[]);
            return None;
        };
        // SAFETY: a fresh buffer of the layer's own; recorded once, empty.
        let empty = unsafe { device.begin_command_buffer(cmds[2], &vk::CommandBufferBeginInfo::default()).and_then(|()| device.end_command_buffer(cmds[2])) };
        if empty.is_err() {
            cleanup(Some(pool), &[]);
            return None;
        }
        let mut fences = Vec::new();
        for _ in 0..2 {
            // SAFETY: valid create info.
            match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
                Ok(f) => fences.push(f),
                Err(_) => {
                    cleanup(Some(pool), &fences);
                    return None;
                }
            }
        }
        Some(Self {
            id: NEXT_RESOURCES.fetch_add(1, Ordering::Relaxed),
            inline,
            #[cfg(target_arch = "x86_64")]
            guard: None,
            seq: 0,
            readings: Readings::default(),
            queue_family,
            width,
            height,
            pool,
            capture_cmd: cmds[0],
            writeback_cmd: cmds[1],
            fence_cmd: cmds[2],
            capture_fence: fences[0],
            writeback_fence: fences[1],
            // Only the split's resources are timed: an inline set creates no query pool at all (a
            // `GpuTimer` has no `Drop`, so one made and dropped would leak its pool).
            capture_timer: (!inline).then(|| crate::gpu_timer::GpuTimer::new(device, instance, physical_device, queue_family)).flatten(),
            writeback_timer: (!inline).then(|| crate::gpu_timer::GpuTimer::new(device, instance, physical_device, queue_family)).flatten(),
            proxy,
            proxy_region,
            dump: None,
            hdr: None,
            #[cfg(target_arch = "x86_64")]
            native: None,
            #[cfg(target_arch = "x86_64")]
            native_retry: native::Retries::default(),
            #[cfg(target_arch = "x86_64")]
            game_exposure_read: None,
            #[cfg(target_arch = "x86_64")]
            auto_read: None,
            #[cfg(target_arch = "x86_64")]
            depth: None,
            #[cfg(target_arch = "x86_64")]
            edit: None,
            #[cfg(target_arch = "x86_64")]
            scene: None,
            cache: None,
            #[cfg(target_arch = "x86_64")]
            pre_loader: None,
            capture_pending: false,
            writeback_pending: false,
            unfenced: false,
            queue: None,
            fences_seen: 0,
        })
    }

    /// The next hold's work goes to `queue` (the application's DLSS submit's).
    pub(crate) fn on_queue(&mut self, queue: vk::Queue) {
        self.queue = Some(queue);
    }

    pub(crate) fn matches(&self, queue_family: u32, width: u32, height: u32) -> bool {
        (self.queue_family, self.width, self.height) == (queue_family, width, height)
    }

    pub(crate) fn imported(&self) -> bool {
        !self.proxy.staged
    }

    pub(crate) fn pending(&self) -> bool {
        self.capture_pending || self.writeback_pending || self.unfenced
    }

    /// After a hold's submit failed with some of its work already on the queue (the capture, maybe
    /// the network): the empty buffer with the write-back fence, which covers everything earlier on the
    /// queue, so the next hold waits for that work before it records any of those buffers again. When
    /// even that fails nothing of these resources is used again ([`Self::wait_idle`] says pending).
    ///
    /// # Safety
    /// The write-back fence is idle (not pending, waited on); `submit` submits on the hold's queue.
    pub(crate) unsafe fn fence_submitted(
        &mut self,
        device: &ash::Device,
        submit: &mut dyn FnMut(Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>,
    ) {
        // SAFETY: the fence is idle (contract); the empty buffer is idle with it.
        let reset = unsafe { device.reset_fences(&[self.writeback_fence]) };
        if reset.is_ok() && crate::note_vk(device.handle(), submit(Which::WriteBack, self.fence_cmd, self.writeback_fence)).is_ok() {
            self.writeback_pending = true;
        } else {
            crate::log!("[preupscale] a hold's work went to the queue and no fence could be put after it: these hold resources are left behind and built again (unless the device is lost)");
            self.unfenced = true;
        }
    }

    fn padded_bytes(&self) -> usize {
        let (pw, ph) = padded(self.width, self.height);
        pw as usize * ph as usize * TEXEL as usize
    }

    /// Waits (bounded) for every submission of the layer's own that is still pending. `true` when
    /// nothing is pending afterwards. Reads the write-back's timestamps into `writeback_gpu_ms`.
    pub(crate) fn wait_idle(&mut self, device: &ash::Device, writeback_gpu_ms: &mut Option<f32>) -> bool {
        if self.unfenced {
            return false;
        }
        let both = [self.capture_fence, self.writeback_fence];
        let fences: &[vk::Fence] = match (self.capture_pending, self.writeback_pending) {
            (true, true) => &both,
            (true, false) => &both[..1],
            (false, true) => &both[1..],
            (false, false) => {
                self.idle();
                return true;
            }
        };
        // SAFETY: the layer's own fences, each submitted (only `pending` after a successful submit).
        let wait = unsafe { device.wait_for_fences(fences, true, CLEANUP_WAIT.as_nanos() as u64) };
        if crate::note_fence_wait(device.handle(), wait, "preupscale::wait_idle").is_err() {
            return false;
        }
        self.fences_seen += 1;
        self.idle();
        if std::mem::take(&mut self.writeback_pending) {
            if let Some(timer) = self.writeback_timer.as_mut() {
                timer.read_if_pending(device);
                *writeback_gpu_ms = timer.take_reading().or(*writeback_gpu_ms);
            }
        }
        self.capture_pending = false;
        true
    }

    /// Nothing of these resources' is pending: the network may be taken by the other path, and the
    /// fences reset or destroyed.
    fn idle(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if let Some(loader) = &self.pre_loader {
            loader.pre_idle(self.writeback_fence);
        }
    }

    /// # Safety
    /// Nothing of this value's may still be pending on the GPU ([`Self::wait_idle`] returned true,
    /// or the device is idle).
    pub(crate) unsafe fn destroy(mut self, device: &ash::Device) {
        self.idle();
        // SAFETY: forwarded; freeing the pool frees the command buffers.
        unsafe {
            if let Some(t) = &self.capture_timer {
                t.destroy(device);
            }
            if let Some(t) = &self.writeback_timer {
                t.destroy(device);
            }
            device.destroy_fence(self.capture_fence, None);
            device.destroy_fence(self.writeback_fence, None);
            device.destroy_command_pool(self.pool, None);
            self.proxy.destroy(device);
            if let Some(dump) = &self.dump {
                dump.depth.destroy(device);
                dump.mvec.destroy(device);
                dump.exposure.destroy(device);
            }
            if let Some(hdr) = &self.hdr {
                hdr.destroy(device);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(native) = &self.native {
                native.destroy(device);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(depth) = &self.depth {
                depth.destroy(device);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(edit) = &self.edit {
                edit.destroy(device);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(scene) = &self.scene {
                scene.destroy(device);
            }
            if let Some(cache) = &self.cache {
                cache.destroy(device);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(guard) = &self.guard {
                guard.destroy(device);
            }
        }
    }

    /// An armed hold's readbacks, once its work finished ([`Self::wait_idle`] returned true since),
    /// with the exposure its encode and decode used. Call before the next hold clears that.
    pub(crate) fn take_cache_frame(&mut self) -> Option<cache_dump::Frame> {
        let exposure = self.hdr.as_ref().map_or(0.0, hdr::HdrPass::exposure_value);
        self.cache.as_mut()?.take(exposure)
    }

    /// Whether an armed hold's readbacks wait to be read.
    pub(crate) fn cache_pending(&self) -> bool {
        self.cache.as_ref().is_some_and(|c| c.pending.is_some())
    }

    /// Frees a finished cache dump's readbacks.
    ///
    /// # Safety
    /// Nothing submitted may still use them ([`Self::wait_idle`] returned true since the last hold).
    /// Frees the native hold's frame resources (the native pass, the edit cache, the depth copy, the scene-cut
    /// detector, the HDR pass) while the VRAM guard has the network paused, so the game gets all of Neural
    /// Forge's memory back; each is built again on the first hold that runs the network. Returns whether there
    /// was anything to free.
    ///
    /// # Safety
    /// Nothing recorded with them may still run.
    #[cfg(target_arch = "x86_64")]
    pub(crate) unsafe fn release_native(&mut self, device: &ash::Device) -> bool {
        let mut any = false;
        // SAFETY (each destroy): nothing recorded with them may still run (this function's contract).
        if let Some(n) = self.native.take() {
            unsafe { n.destroy(device) };
            any = true;
        }
        if let Some(e) = self.edit.take() {
            unsafe { e.destroy(device) };
            any = true;
        }
        if let Some(d) = self.depth.take() {
            unsafe { d.destroy(device) };
            any = true;
        }
        if let Some(s) = self.scene.take() {
            unsafe { s.destroy(device) };
            any = true;
        }
        if let Some(h) = self.hdr.take() {
            unsafe { h.destroy(device) };
            any = true;
        }
        self.game_exposure_read = None;
        self.auto_read = None;
        any
    }

    pub(crate) unsafe fn free_cache(&mut self, device: &ash::Device) {
        if let Some(cache) = self.cache.take() {
            // SAFETY: forwarded.
            unsafe { cache.destroy(device) };
        }
    }
}

/// The extra copies of a hold armed for a cache dump ([`cache_dump`]): the colour input before the
/// model (in the capture), and after the write-back the colour input again (the answer written
/// back), the native pass's motion vectors and the depth copy, each `(source, readback)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheCopies {
    pub colour: vk::Buffer,
    pub answer: vk::Buffer,
    pub mvec: Option<(vk::Buffer, vk::Buffer)>,
    pub depth: Option<(vk::Buffer, vk::Buffer)>,
}

/// Dump mode's readbacks: depth and motion vectors (render extent, 4 bytes per texel), and the
/// 1x1 exposure candidates at [`EXPOSURE_STRIDE`] bytes each.
struct DumpBuffers {
    depth: HostBuffer,
    mvec: HostBuffer,
    exposure: HostBuffer,
}

/// A depth, motion-vector or 1x1 exposure image to read in dump mode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Aux {
    pub image: vk::Image,
    pub format: vk::Format,
    /// The committed layout, `None` when no barrier on it was seen (then it is not read).
    pub layout: Option<vk::ImageLayout>,
    pub readable: bool,
}

/// What a hold works on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    pub colour: vk::Image,
    pub width: u32,
    pub height: u32,
    /// Inside DLSS's buffer with a subrect: the rectangle's size at the image's corner (the rest of the
    /// `width` x `height` the hold works on is its edge, replicated). A new size resets the history: its pixels
    /// stand for other points of the screen. `None`: the whole extent.
    pub region: Option<(u32, u32)>,
    pub depth: Option<Aux>,
    /// Inside DLSS's buffer: the slot buffer the hold copied DLSS's depth aspect into there, and the
    /// depth's format (`inline`). Copied from instead of [`Self::depth`].
    pub depth_buffer: Option<(vk::Buffer, vk::Format)>,
    pub mvec: Option<Aux>,
    /// What turns `mvec`'s values into this colour input's pixels, per axis ([`InlinePoint::mvec_scale`]):
    /// the native backend's reprojection multiplies by it, a dump records it.
    pub mvec_scale: [f32; 2],
    /// The 1x1 exposure candidates, read in dump mode, each with whether its layout is assumed
    /// (no barrier on it was seen: a storage image is then read as `GENERAL`, the layout
    /// vkd3d-proton keeps those in).
    pub exposure: [Option<(Aux, bool)>; MAX_EXPOSURE],
    /// DLSS's exposure input (the registered 1x1 R16_SFLOAT image), read by the capture in model
    /// and roundtrip modes for the HDR encode; `None` when there is none (the exposure is then
    /// measured from the frame, [`ExposureSource::Auto`]). Its layout is the committed one, or
    /// `GENERAL` assumed for a storage image.
    pub exposure_input: Option<Aux>,
    /// The HDR encode's paper white ([`hdr::paper_white`]).
    pub paper_white: f32,
    /// The Model brightness setting's Auto: the exposure is measured from the frame as for
    /// [`ExposureSource::Auto`], and where the game gives one it is kept within [`hdr::LIFT_LO`]..[`hdr::LIFT_HI`]
    /// of the game's ([`ExposureSource::measures`]).
    pub lift_exposure: bool,
    /// The colour input looks already exposed by the game ([`Session::pre_exposed_measured`]): the exposure is
    /// measured from the frame, unbounded, while the game's is still copied in beside it ([`hdr::AutoUse::Beside`]).
    pub pre_exposed: bool,
    /// Which identification of DLSS's inputs this hold works on ([`Scan::identification`]): the
    /// auto-exposure starts its adaptation afresh when it changes.
    pub identification: u64,
}

impl Target {
    /// With NGX's pre-exposure and exposure scale for this colour input ([`crate::ngx_ring::paper_white_factor`]) in
    /// the paper white, where the encode uses the game's exposure as it is ([`hdr::AutoUse::Observe`]): DLSS divides
    /// the colour input by the pre-exposure and multiplies the exposure image's value by the scale. The encode and
    /// the decode read the same paper white, so the write-back stays the encode's exact inverse.
    pub(crate) fn with_ngx_exposure(mut self) -> Self {
        if ExposureSource::auto_use(&self) == hdr::AutoUse::Observe {
            self.paper_white *= crate::ngx_ring::paper_white_factor(self.colour);
        }
        self
    }
}

/// Where the HDR encode's exposure value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExposureSource {
    /// DLSS's exposure input, the game's 1x1 R16_SFLOAT image, copied in by the capture.
    Game,
    /// Measured from the colour input on the GPU (`preupscale_exposure.comp`): the game gives DLSS
    /// no exposure image, or it cannot be read.
    Auto,
}

impl ExposureSource {
    /// The source for `target`: the game's image when it can be copied from, else the frame.
    pub(crate) fn of(target: &Target) -> Self {
        if target.exposure_input.is_some_and(|a| a.readable && aux_copy_layout(a.layout).is_some()) {
            Self::Game
        } else {
            Self::Auto
        }
    }

    /// Whether the capture measures the exposure from the frame for `target`: the source is the frame, or the
    /// game's exposure is lifted toward the measured one ([`Target::lift_exposure`]).
    pub(crate) fn measures(target: &Target) -> bool {
        target.lift_exposure || target.pre_exposed || Self::of(target) == Self::Auto
    }

    /// What the auto-exposure does for `target` ([`hdr::AutoUse`]). It runs in every hold that encodes: where the
    /// game's exposure is used as it is it only measures, for [`Session::check_exposure`]'s test of a colour
    /// input the game already exposed.
    pub(crate) fn auto_use(target: &Target) -> hdr::AutoUse {
        match (Self::of(target), target.lift_exposure) {
            (Self::Auto, _) => hdr::AutoUse::Measured,
            (Self::Game, _) if target.pre_exposed => hdr::AutoUse::Beside,
            (Self::Game, true) => hdr::AutoUse::Lift,
            (Self::Game, false) => hdr::AutoUse::Observe,
        }
    }

    /// The log line's words for it.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Self::Game => "the game's 1x1 R16F",
            Self::Auto => "measured from the frame (auto)",
        }
    }
}

/// Which of the layer's two submissions [`run_hold`] asks the caller to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Which {
    /// The capture: carries the moved wait semaphores.
    Capture,
    /// The native backend's network (`native`), between the capture and the write-back.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Native,
    WriteBack,
}

/// What a hold did.
#[derive(Debug, Default)]
pub(crate) struct HoldResult {
    /// The capture batch was submitted, so the moved waits were consumed by it.
    pub waits_consumed: bool,
    /// The write-back was submitted.
    pub wrote_back: bool,
    /// Why the hold did not do its full job (a model answer over budget, a failed submit, ...).
    pub miss: Option<&'static str>,
    /// Model mode: the network ran and its answer was written back. Only such a hold engages the
    /// device ([`Session::note`]).
    pub evaluated: bool,
    pub capture_gpu_ms: Option<f32>,
    /// The previous hold's write-back GPU time, read when its fence was found signalled here.
    pub writeback_gpu_ms: Option<f32>,
    /// Dump mode: the bytes to write, captured.
    pub dump: Option<DumpFrame>,
    /// Model and roundtrip modes: the exposure value the capture read (the half the encode and
    /// the decode both use).
    pub exposure: Option<f32>,
    /// Model and roundtrip modes: where it came from.
    pub exposure_source: Option<ExposureSource>,
    /// Split holds with [`ExposureSource::Game`]: the game's own exposure, which [`Self::exposure`] is a lift of
    /// with the Model brightness setting's Auto ([`hdr::exposure_word`]), else the same value.
    pub game_exposure: Option<f32>,
    /// Native holds: the game's exposure the previous hold of the same identification on these
    /// resources read. The native hold does not wait for its own capture, so its value is read at
    /// the next hold, once that work has finished.
    pub previous_exposure: Option<f32>,
    /// What the auto-exposure measured (with the game's exposure too, where it only observes: [`hdr::AutoUse`]);
    /// native holds: the previous hold's, read with [`Self::previous_exposure`].
    pub auto_exposure: Option<hdr::AutoState>,
    /// Where the hold's CPU time went (wall time on this thread).
    pub timing: HoldTiming,
    /// The native backend ran this hold (`evaluated` means the network ran).
    pub native: bool,
    /// The native backend: the GPU time of the previous hold's network frame (`N`: preprocess, the
    /// network, composite), read when its fence was found signalled here.
    pub network_gpu_ms: Option<f32>,
    /// The native backend: the GPU time of the last depth copy (`native::DepthCopy`), read likewise.
    pub depth_gpu_ms: Option<f32>,
    /// A cache dump's armed hold read back here (the previous one on the split path).
    pub cache_frame: Option<cache_dump::Frame>,
    /// The native backend: this hold reused the last refresh's correction (the edit cache) instead of
    /// running the network. `evaluated` is set either way.
    pub cached: bool,
    /// The native backend: the edit cache's adaptive cadence had the camera still on this hold
    /// (`cache::Regime`: the network on every hold).
    pub cache_still: bool,
    /// The native backend: the GPU time of the previous hold's cache passes, read likewise.
    pub cache_gpu_ms: Option<f32>,
    /// The previous cached hold's share of rejected pixels (0..1), read likewise.
    pub cache_rejected: Option<f32>,
    /// The native backend: the network's history was reset for a scene cut (`scene_cut`), counted here
    /// once per cut: on the GPU for the previous hold's network frame (read here), or on this hold's
    /// network frame for a cut on the cached hold before it.
    pub scene_cut: bool,
    /// The hold inside DLSS's buffer: what it recorded, for what its buffer reads back once it ran.
    pub meta: Option<inline::Meta>,
}

impl HoldResult {
    /// A model-mode hold that failed before the network ran, for `why`.
    pub(crate) fn local(why: &'static str) -> Self {
        Self { miss: Some(why), ..Default::default() }
    }
}

/// Wall-clock phases of one hold, for the periodic `[preupscale]` line. `None` for a phase the
/// hold did not reach.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HoldTiming {
    /// From the start of the hold to the capture submit returning: the previous write-back's
    /// fence check, recording, the submit itself.
    pub prep: Option<Duration>,
    /// The capture fence wait (it also drains the game's work queued before the capture).
    pub capture_wait: Option<Duration>,
    /// From the capture to the write-back submit returning.
    pub writeback: Option<Duration>,
}

/// One dump's bytes, written off the submit thread by [`DumpFrame::write`].
#[derive(Debug)]
pub(crate) struct DumpFrame {
    pub width: u32,
    pub height: u32,
    pub colour: Vec<u8>,
    pub depth: Option<(Vec<u8>, vk::Format)>,
    pub mvec: Option<Vec<u8>>,
    /// `mvec`'s values times this are the colour input's pixels ([`Target::mvec_scale`]).
    pub mvec_scale: [f32; 2],
    pub frame: u64,
    /// The 1x1 exposure candidates, read or not.
    pub exposure: Vec<ExposureValue>,
}

/// One 1x1 exposure candidate in a dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExposureValue {
    pub image: u64,
    pub format: vk::Format,
    /// The layout it was read in (`None`: not read), and whether that layout was assumed.
    pub layout: Option<vk::ImageLayout>,
    pub assumed: bool,
    /// The texel (little-endian, the format's size), `None` when it was not read.
    pub bytes: Option<Vec<u8>>,
}

fn aux_copy_layout(layout: Option<vk::ImageLayout>) -> Option<(vk::ImageLayout, bool)> {
    match layout? {
        vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED | ASPECTS_APART => None,
        l @ (vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => Some((l, false)),
        _ => Some((vk::ImageLayout::TRANSFER_SRC_OPTIMAL, true)),
    }
}

pub(crate) fn aux_range(format: vk::Format) -> vk::ImageSubresourceRange {
    let aspect = if DEPTH_FORMATS.contains(&format) {
        if has_stencil(format) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    }
}

/// What a dump reads besides colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DumpPart {
    Depth,
    Mvec,
    /// The exposure candidate at this index of [`Target::exposure`].
    Exposure(usize),
    /// [`Target::exposure_input`], into the HDR pass's exposure buffer.
    ExposureInput,
    /// [`Target::mvec`], into the native backend's motion buffer.
    NativeMvec,
    /// [`Target::depth`]'s depth aspect, into the native backend's depth copy.
    NativeDepth,
}

/// One image a dump copies into a readback buffer.
#[derive(Clone, Copy, Debug)]
struct DumpRead {
    part: DumpPart,
    aux: Aux,
    buffer: vk::Buffer,
    offset: u64,
    extent: (u32, u32),
    read: vk::ImageLayout,
    transition: bool,
}

/// The capture's reads besides colour: for a dump, the depth aspect and the motion vectors at the
/// render extent and each 1x1 exposure candidate at its [`EXPOSURE_STRIDE`] slot; for the HDR
/// encode (`exposure_buffer`), the exposure input. Each readable one in a layout it can be copied
/// from (see [`aux_copy_layout`]).
fn capture_reads(
    target: &Target,
    bufs: Option<&DumpBuffers>,
    exposure_buffer: Option<vk::Buffer>,
    native_mvec: Option<vk::Buffer>,
    native_depth: Option<vk::Buffer>,
) -> Vec<DumpRead> {
    let full = (target.width, target.height);
    let mut parts = Vec::new();
    // First: the capture times it alone.
    if let Some(buffer) = native_depth.filter(|_| target.depth_buffer.is_none()) {
        parts.push((DumpPart::NativeDepth, target.depth, buffer, 0, full));
    }
    if let Some(bufs) = bufs {
        parts.push((DumpPart::Depth, target.depth, bufs.depth.buffer, 0, full));
        parts.push((DumpPart::Mvec, target.mvec, bufs.mvec.buffer, 0, full));
        for (k, e) in target.exposure.iter().enumerate() {
            parts.push((DumpPart::Exposure(k), e.map(|(a, _)| a), bufs.exposure.buffer, k as u64 * EXPOSURE_STRIDE, (1, 1)));
        }
    }
    if let Some(buffer) = exposure_buffer {
        parts.push((DumpPart::ExposureInput, target.exposure_input, buffer, 0, (1, 1)));
    }
    if let Some(buffer) = native_mvec {
        parts.push((DumpPart::NativeMvec, target.mvec, buffer, 0, full));
    }
    parts
        .into_iter()
        .filter_map(|(part, aux, buffer, offset, extent)| {
            let aux = aux.filter(|a| a.readable)?;
            let (read, transition) = aux_copy_layout(aux.layout)?;
            Some(DumpRead { part, aux, buffer, offset, extent, read, transition })
        })
        .collect()
}

/// Records the capture into `proxy` (slot 0's proxy buffer; `None` for the native backend, which reads
/// the HDR pass's encoded image and leaves slot 0 to the after-the-upscaler path) at the padded size:
/// the colour input (GENERAL) copied raw, or with `hdr`, the exposure input read into the HDR pass's
/// buffer (or, with no readable one,
/// [`ExposureSource::Auto`], measured from the colour input into the same buffer,
/// [`hdr::HdrPass::record_auto_exposure`]) and the colour input encoded for the model
/// ([`hdr::HdrPass::record_encode`]); in dump mode also the depth aspect, the
/// motion vectors and the 1x1 exposure candidates. Each auxiliary image is read in its committed
/// layout (transitioned to TRANSFER_SRC_OPTIMAL and back when that layout cannot be copied from).
/// The native backend's hold also copies DLSS's depth into its buffer (`native_depth`, with the
/// timer around that copy alone; from [`Target::depth_buffer`] inside DLSS's buffer, else the depth
/// aspect of [`Target::depth`]), for a cache dump the colour input as it is (`colour_before`,
/// tightly packed at the real extent), and last `extra` (the edit cache's passes, `cache`).
///
/// # Safety
/// `res`'s capture command buffer must not be pending; with `hdr`, its colour view is bound to
/// `target.colour` and, for [`ExposureSource::Auto`], its auto-exposure is built.
#[allow(clippy::too_many_arguments)]
unsafe fn record_capture(
    device: &ash::Device,
    res: &Resources,
    target: &Target,
    dump: Option<&DumpBuffers>,
    hdr: Option<&hdr::HdrPass>,
    proxy: Option<vk::Buffer>,
    native_mvec: Option<vk::Buffer>,
    native_depth: Option<(vk::Buffer, Option<&crate::gpu_timer::GpuTimer>)>,
    colour_before: Option<vk::Buffer>,
    extra: Option<&dyn Fn(vk::CommandBuffer)>,
    into: Option<vk::CommandBuffer>,
) -> ash::prelude::VkResult<()> {
    let relative = ExposureSource::of(target) == ExposureSource::Game;
    // The auto-exposure runs wherever it is built (only measuring beside the game's exposure).
    let auto = hdr.is_some_and(hdr::HdrPass::auto_built);
    // Into DLSS's buffer (`into`, being recorded), or the layer's own capture buffer.
    let cmd = into.unwrap_or(res.capture_cmd);
    let capture_timer = res.capture_timer.as_ref().filter(|_| into.is_none());
    if into.is_none() {
        // SAFETY: the pool allows resetting buffers individually; the buffer is not pending.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }
    }
    // The game's texel is copied in whenever it is the source, also when the auto-exposure lifts it after.
    let aux: Vec<DumpRead> =
        capture_reads(target, dump, hdr.filter(|_| relative).map(hdr::HdrPass::exposure_buffer), native_mvec, native_depth.map(|(b, _)| b));
    let depth_timer = native_depth.and_then(|(_, t)| t);
    let mut to_read: Vec<vk::ImageMemoryBarrier> = aux
        .iter()
        .filter(|r| r.transition)
        .map(|&DumpRead { aux: a, read, .. }| {
            vk::ImageMemoryBarrier::default()
                .old_layout(a.layout.unwrap_or_default())
                .new_layout(read)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(a.image)
                .subresource_range(aux_range(a.format))
        })
        .collect();
    // The encode's own image: its old contents are not needed (fully rewritten).
    to_read.extend(hdr.map(hdr::HdrPass::encoded_to_general));
    let (dst_stage, dst_access) = if hdr.is_some() {
        (
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        )
    } else {
        (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
    };
    let open = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(dst_access);
    // SAFETY: recording into the layer's own buffer; every handle is live.
    unsafe {
        device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, dst_stage, vk::DependencyFlags::empty(), &[open], &[], &to_read);
        if let Some(timer) = capture_timer {
            timer.record_start(device, cmd);
        }
        // Inside DLSS's buffer the hold already copied the depth into its slot buffer.
        if let (Some((dst, _)), Some((src, format))) = (native_depth, target.depth_buffer) {
            if let Some(t) = depth_timer {
                t.record_start(device, cmd);
            }
            let size = u64::from(target.width) * u64::from(target.height) * depth_texel_bytes(format);
            device.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            if let Some(t) = depth_timer {
                t.record_end(device, cmd);
            }
        }
        for r in &aux {
            let timed = r.part == DumpPart::NativeDepth;
            if timed {
                if let Some(t) = depth_timer {
                    t.record_start(device, cmd);
                }
            }
            let aspect = if DEPTH_FORMATS.contains(&r.aux.format) { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
            let region = vk::BufferImageCopy {
                buffer_offset: r.offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: r.extent.0, height: r.extent.1, depth: 1 },
            };
            device.cmd_copy_image_to_buffer(cmd, r.aux.image, r.read, r.buffer, &[region]);
            if timed {
                if let Some(t) = depth_timer {
                    t.record_end(device, cmd);
                }
            }
        }
        if let Some(buffer) = colour_before {
            device.cmd_copy_image_to_buffer(cmd, target.colour, vk::ImageLayout::GENERAL, buffer, &[colour_region(target.width, target.height)]);
        }
        match hdr {
            Some(pass) => {
                // Unbuilt where it measures (the caller builds it first), the exposure buffer would keep the 0
                // it was cleared to, and nothing would be written back. Where it only observes, unbuilt
                // records nothing.
                pass.record_auto_exposure(device, cmd, native::format_of(target), ExposureSource::auto_use(target));
                pass.record_encode(device, cmd, proxy, target.paper_white);
            }
            None => {
                if let Some(proxy) = proxy {
                    device.cmd_copy_image_to_buffer(cmd, target.colour, vk::ImageLayout::GENERAL, proxy, &capture_regions(target.width, target.height));
                }
            }
        }
        // The edit cache's passes (the native hold's), after every copy and the encode.
        if let Some(extra) = extra {
            extra(cmd);
        }
    }
    let back: Vec<vk::ImageMemoryBarrier> = aux
        .iter()
        .filter(|r| r.transition)
        .map(|&DumpRead { aux: a, read, .. }| {
            vk::ImageMemoryBarrier::default()
                .old_layout(read)
                .new_layout(a.layout.unwrap_or_default())
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(a.image)
                .subresource_range(aux_range(a.format))
        })
        .collect();
    // The auto-exposure's resolve wrote its state and the exposure value from a shader: those are
    // read by the host too.
    // Without the proxy copy the encode's own writes are the last of it.
    let (close_stage, close_access) = if auto || (hdr.is_some() && proxy.is_none()) {
        (vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
    } else {
        (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
    };
    let close = vk::MemoryBarrier::default().src_access_mask(close_access).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
    // SAFETY: as above.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            close_stage,
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[close],
            &[],
            &back,
        );
        if let Some(timer) = capture_timer {
            timer.record_end(device, cmd);
        }
        if into.is_none() {
            device.end_command_buffer(cmd)?;
        }
        Ok(())
    }
}

/// Records the write-back into the colour input (GENERAL), cropping the padding: `source` (padded
/// RGBA16F) copied raw, or with `hdr`, copied into the HDR pass's answer image and decoded over the
/// colour input in place ([`hdr::HdrPass::record_decode`]). Closed by a barrier that makes it
/// visible to everything later on the queue. For a cache dump (`cache`), the colour input with the
/// answer written back, the motion vectors and the depth are then copied into its readbacks.
/// `edit_capture` is the edit cache's capture, recorded right before the decode.
///
/// # Safety
/// `res`'s write-back command buffer must not be pending; with `hdr`, the pass holds this hold's
/// capture and its colour view is bound to `target.colour`.
#[allow(clippy::too_many_arguments)]
unsafe fn record_writeback(
    device: &ash::Device,
    res: &Resources,
    target: &Target,
    source: vk::Buffer,
    hdr: Option<&hdr::HdrPass>,
    cache: Option<&CacheCopies>,
    edit_capture: Option<&dyn Fn(vk::CommandBuffer)>,
    into: Option<vk::CommandBuffer>,
) -> ash::prelude::VkResult<()> {
    // Into DLSS's buffer (`into`, being recorded), or the layer's own write-back buffer.
    let cmd = into.unwrap_or(res.writeback_cmd);
    let writeback_timer = res.writeback_timer.as_ref().filter(|_| into.is_none());
    let (dst_stage, dst_access, mut src_stage, mut src_access) = if hdr.is_some() {
        (
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
        )
    } else {
        (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        )
    };
    let open = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE).dst_access_mask(dst_access);
    // The dump's copies read what the decode wrote, and the host reads them after the fence.
    let (mut close_stage, mut close_access) = (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    let decoded = vk::MemoryBarrier::default().src_access_mask(src_access).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let decoded_stage = src_stage;
    if cache.is_some() {
        src_stage |= vk::PipelineStageFlags::TRANSFER;
        src_access |= vk::AccessFlags::TRANSFER_WRITE;
        close_stage |= vk::PipelineStageFlags::HOST;
        close_access |= vk::AccessFlags::HOST_READ;
    }
    let close = vk::MemoryBarrier::default().src_access_mask(src_access).dst_access_mask(close_access);
    let answer_image: Vec<vk::ImageMemoryBarrier> = hdr.map(hdr::HdrPass::answer_to_general).into_iter().collect();
    // SAFETY: the layer's own buffer, not pending, or DLSS's being recorded; every handle is live.
    unsafe {
        if into.is_none() {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[open],
            &[],
            &answer_image,
        );
        if let Some(timer) = writeback_timer {
            timer.record_start(device, cmd);
        }
        match hdr {
            Some(pass) => pass.record_decode(device, cmd, source, target.paper_white, edit_capture),
            None => device.cmd_copy_buffer_to_image(cmd, source, target.colour, vk::ImageLayout::GENERAL, &[writeback_region(target.width, target.height)]),
        }
        if let Some(c) = cache {
            device.cmd_pipeline_barrier(cmd, decoded_stage, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[decoded], &[], &[]);
            device.cmd_copy_image_to_buffer(cmd, target.colour, vk::ImageLayout::GENERAL, c.answer, &[colour_region(target.width, target.height)]);
            let texels = u64::from(target.width) * u64::from(target.height);
            for (src, dst) in [c.mvec, c.depth].into_iter().flatten() {
                device.cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: texels * 4 }]);
            }
        }
        device.cmd_pipeline_barrier(cmd, src_stage, close_stage, vk::DependencyFlags::empty(), &[close], &[], &[]);
        if let Some(timer) = writeback_timer {
            timer.record_end(device, cmd);
        }
        if into.is_none() {
            device.end_command_buffer(cmd)?;
        }
        Ok(())
    }
}

/// Records a cached hold's write-back (the edit cache, `cache`): the write-back's opening barrier,
/// `apply` (the cache's passes, which write the colour input in place), and its closing barrier, with
/// the write-back's timer around them.
///
/// # Safety
/// `res`'s write-back command buffer must not be pending.
#[cfg(target_arch = "x86_64")]
unsafe fn record_cached_writeback(
    device: &ash::Device,
    res: &Resources,
    apply: &dyn Fn(vk::CommandBuffer),
    into: Option<vk::CommandBuffer>,
) -> ash::prelude::VkResult<()> {
    let cmd = into.unwrap_or(res.writeback_cmd);
    let writeback_timer = res.writeback_timer.as_ref().filter(|_| into.is_none());
    let open = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    let close = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // SAFETY: the layer's own buffer, not pending, or DLSS's being recorded; every handle is live.
    unsafe {
        if into.is_none() {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        }
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[open],
            &[],
            &[],
        );
        if let Some(timer) = writeback_timer {
            timer.record_start(device, cmd);
        }
        apply(cmd);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[close],
            &[],
            &[],
        );
        if let Some(timer) = writeback_timer {
            timer.record_end(device, cmd);
        }
        if into.is_none() {
            device.end_command_buffer(cmd)?;
        }
        Ok(())
    }
}

/// One diagnostic hold (model mode runs on the native backend, [`run_native`]): capture the colour
/// input into slot 0's proxy region, then per `mode` write the same bytes back (identity), encode and
/// decode them (roundtrip), or keep them for a dump. In roundtrip mode the capture encodes the frame
/// as for the model and the write-back decodes it ([`hdr`]); with no
/// readable exposure image the exposure is measured from the frame ([`ExposureSource::Auto`]), and
/// with an unusable exposure value nothing is written back. Both the encode and the decode read the
/// exposure from the same buffer, so they use the same value. `submit` makes the layer's two submissions on the game's queue (the capture one with
/// the moved wait semaphores). Never blocks longer than the capture's bounded fence wait. The
/// write-back is not waited on: its fence is checked at the next hold
/// ([`Resources::wait_idle`]).
///
/// # Safety
/// `res` must belong to `device` and match `target`'s extent; `target`'s images must be live on
/// `device`, the colour input in GENERAL; `submit` must submit on the queue the game's submit is
/// for, which the caller holds.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_hold(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    res: &mut Resources,
    target: &Target,
    mode: Mode,
    dump: bool,
    frame: u64,
    submit: &mut dyn FnMut(Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>,
) -> HoldResult {
    let mut result = HoldResult::default();
    let t_hold = Instant::now();
    let mut writeback_gpu = None;
    if !res.wait_idle(device, &mut writeback_gpu) {
        result.miss = Some("the previous hold's work is still pending");
        return result;
    }
    result.writeback_gpu_ms = writeback_gpu;
    if dump && res.dump.is_none() {
        let bytes = u64::from(target.width) * u64::from(target.height) * 4;
        let depth = own_host_buffer(device, instance, physical_device, bytes);
        let mvec = own_host_buffer(device, instance, physical_device, bytes);
        let exposure = own_host_buffer(device, instance, physical_device, MAX_EXPOSURE as u64 * EXPOSURE_STRIDE);
        match (depth, mvec, exposure) {
            (Some(depth), Some(mvec), Some(exposure)) => res.dump = Some(DumpBuffers { depth, mvec, exposure }),
            (d, m, e) => {
                // SAFETY: never submitted.
                unsafe {
                    d.iter().chain(m.iter()).chain(e.iter()).for_each(|b| b.destroy(device));
                }
            }
        }
    }
    if mode.hdr() {
        // The encode needs an exposure value: the game's, or without a readable exposure image one
        // measured from the frame.
        let source = ExposureSource::of(target);
        result.exposure_source = Some(source);
        if res.hdr.is_none() {
            // SAFETY: `device` is live.
            res.hdr = unsafe { hdr::HdrPass::build(device, instance, physical_device, res.width, res.height) };
        }
        let Some(pass) = res.hdr.as_mut() else {
            result.miss = Some("the HDR encode/decode pipelines could not be built");
            return result;
        };
        // SAFETY: `device` is live; nothing of the pass is pending (`wait_idle` above).
        let auto = unsafe { pass.ensure_auto(device, instance, physical_device) };
        if !auto && ExposureSource::measures(target) {
            result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
            return result;
        }
        pass.auto_for(target.identification);
        pass.clear_exposure();
        // SAFETY: nothing of the pass is pending (`wait_idle` above); the colour input is a live
        // plain RGBA16F storage image (`identify`).
        if !unsafe { pass.bind_colour(device, target.colour) } {
            result.miss = Some("a storage view of the colour input could not be created");
            return result;
        }
    }
    let hdr_pass = if mode.hdr() { res.hdr.as_ref() } else { None };
    let dump_bufs = if dump { res.dump.as_ref() } else { None };
    // SAFETY: the capture buffer is idle (`wait_idle` above); the colour view was bound above.
    if unsafe { record_capture(device, res, target, dump_bufs, hdr_pass, Some(res.proxy.buffer), None, None, None, None, None) }.is_err() {
        result.miss = Some("recording the capture failed");
        return result;
    }
    // SAFETY: the fence is idle (`wait_idle` above, or never submitted).
    if unsafe { device.reset_fences(&[res.capture_fence]) }.is_err() {
        result.miss = Some("resetting the capture fence failed");
        return result;
    }
    if crate::note_vk(device.handle(), submit(Which::Capture, res.capture_cmd, res.capture_fence)).is_err() {
        result.miss = Some("the capture submit failed");
        return result;
    }
    result.waits_consumed = true;
    res.capture_pending = true;
    if let Some(timer) = res.capture_timer.as_mut() {
        timer.mark_submitted();
    }
    result.timing.prep = Some(t_hold.elapsed());
    let t_capture = Instant::now();
    // SAFETY: the layer's own fence, just submitted.
    let wait = unsafe { device.wait_for_fences(&[res.capture_fence], true, FRAME_CAPTURE_WAIT.as_nanos() as u64) };
    result.timing.capture_wait = Some(t_capture.elapsed());
    if crate::note_fence_wait(device.handle(), wait, "preupscale::capture").is_err() {
        result.miss = Some("the capture did not finish");
        return result;
    }
    res.capture_pending = false;
    if let Some(timer) = res.capture_timer.as_mut() {
        timer.read_if_pending(device);
        result.capture_gpu_ms = timer.take_reading();
    }
    let bytes = res.padded_bytes();
    if res.proxy.staged {
        // SAFETY: both valid for `bytes` (the staged buffer was allocated at that size, the region
        // is larger); the capture's fence was waited on and its memory is host-coherent.
        unsafe { std::ptr::copy_nonoverlapping(res.proxy.ptr, res.proxy_region, bytes) };
    }
    if let Some(pass) = res.hdr.as_ref().filter(|_| mode.hdr()) {
        // The decode divides by it: a zero, negative or non-finite exposure means no write-back.
        let (e, game) = pass.exposure_values();
        result.exposure = Some(e);
        result.game_exposure = Some(game);
        result.auto_exposure = pass.auto_state();
        if !hdr::exposure_ok(e) {
            result.miss = Some("the exposure value is not usable (zero, negative or not finite)");
            return result;
        }
    }
    let source = match mode {
        Mode::Dump => {
            if dump {
                let n = target.width as usize * target.height as usize * 4;
                // SAFETY: the capture finished; each buffer holds at least `at + len` bytes (`n`,
                // `bytes`, `MAX_EXPOSURE * EXPOSURE_STRIDE`).
                let read = |buf: &HostBuffer, at: usize, len: usize| unsafe { std::slice::from_raw_parts(buf.ptr.add(at), len) }.to_vec();
                let reads = res.dump.as_ref().map(|bufs| capture_reads(target, Some(bufs), None, None, None)).unwrap_or_default();
                let was_read = |part: DumpPart| reads.iter().any(|r| r.part == part);
                result.dump = Some(DumpFrame {
                    width: target.width,
                    height: target.height,
                    colour: read(&res.proxy, 0, bytes),
                    depth: res.dump.as_ref().filter(|_| was_read(DumpPart::Depth)).map(|b| {
                        let format = target.depth.map_or(vk::Format::UNDEFINED, |a| a.format);
                        (read(&b.depth, 0, target.width as usize * target.height as usize * depth_texel_bytes(format) as usize), format)
                    }),
                    mvec: res.dump.as_ref().filter(|_| was_read(DumpPart::Mvec)).map(|b| read(&b.mvec, 0, n)),
                    mvec_scale: target.mvec_scale,
                    frame,
                    exposure: target
                        .exposure
                        .iter()
                        .enumerate()
                        .filter_map(|(k, e)| e.map(|(a, assumed)| (k, a, assumed)))
                        .map(|(k, a, assumed)| {
                            let read_now = was_read(DumpPart::Exposure(k));
                            ExposureValue {
                                image: a.image.as_raw(),
                                format: a.format,
                                layout: if read_now { a.layout } else { None },
                                assumed,
                                bytes: res.dump.as_ref().filter(|_| read_now).map(|b| {
                                    read(&b.exposure, k * EXPOSURE_STRIDE as usize, exposure_texel_bytes(a.format).unwrap_or(EXPOSURE_STRIDE) as usize)
                                }),
                            }
                        })
                        .collect(),
                });
            }
            return result;
        }
        // The encoded picture itself is the answer: the decode must give the frame back.
        Mode::Identity | Mode::Roundtrip => res.proxy.buffer,
        // Model mode runs on the native backend (`run_native`), and off never holds: neither gets here.
        Mode::Model | Mode::Off => {
            result.miss = Some("model mode runs on the native backend");
            return result;
        }
    };
    let t_writeback = Instant::now();
    let hdr_pass = if mode.hdr() { res.hdr.as_ref() } else { None };
    // SAFETY: the write-back buffer is idle (`wait_idle` above); in the HDR modes the pass holds this
    // hold's capture and its colour view.
    if unsafe { record_writeback(device, res, target, source, hdr_pass, None, None, None) }.is_err() {
        result.miss = Some("recording the write-back failed");
        return result;
    }
    // SAFETY: as for the capture fence.
    if unsafe { device.reset_fences(&[res.writeback_fence]) }.is_err() {
        result.miss = Some("resetting the write-back fence failed");
        return result;
    }
    if crate::note_vk(device.handle(), submit(Which::WriteBack, res.writeback_cmd, res.writeback_fence)).is_err() {
        result.miss = Some("the write-back submit failed");
        return result;
    }
    res.writeback_pending = true;
    if let Some(timer) = res.writeback_timer.as_mut() {
        timer.mark_submitted();
    }
    result.wrote_back = true;
    result.timing.writeback = Some(t_writeback.elapsed());
    result
}

// ---- Dump files. ----

/// IEEE half to single precision.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = f32::from(h & 0x3ff);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        0x1f => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + mant / 1024.0) * 2f32.powi(i32::from(e) - 15),
    }
}

/// A human-viewable preview of scene-linear RGBA16F: `x / (1 + x)` per channel, then sRGB-encoded.
pub(crate) fn preview_rgba8(colour: &[u8], row_texels: u32, width: u32, height: u32) -> Vec<u8> {
    let encode = |x: f32| -> u8 {
        let x = if x.is_finite() { x.max(0.0) } else { 0.0 };
        let t = x / (1.0 + x);
        let s = if t <= 0.003_130_8 { 12.92 * t } else { 1.055 * t.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round().clamp(0.0, 255.0) as u8
    };
    let mut out = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let at = (y * row_texels as usize + x) * TEXEL as usize;
            let c = |i: usize| f16_to_f32(u16::from_le_bytes([colour[at + i * 2], colour[at + i * 2 + 1]]));
            out.extend_from_slice(&[encode(c(0)), encode(c(1)), encode(c(2)), 255]);
        }
    }
    out
}

impl DumpFrame {
    /// Writes `colour.rgba16f` (padded), `depth.r32f` (or `depth.raw` for a non-float depth),
    /// `mvec.rg16f`, `meta.json`, `colour-preview.png` and, with any 1x1 candidates,
    /// `exposure.json` under `dir`. Returns the files written.
    pub(crate) fn write(&self, dir: &std::path::Path) -> std::io::Result<Vec<String>> {
        std::fs::create_dir_all(dir)?;
        let (pw, ph) = padded(self.width, self.height);
        let mut files = vec!["colour.rgba16f".to_string()];
        std::fs::write(dir.join("colour.rgba16f"), &self.colour)?;
        let depth_name = self.depth.as_ref().map(|(bytes, format)| {
            let name = if matches!(*format, vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT) { "depth.r32f" } else { "depth.raw" };
            (name, bytes, *format)
        });
        if let Some((name, bytes, _)) = &depth_name {
            std::fs::write(dir.join(name), bytes)?;
            files.push((*name).to_string());
        }
        if let Some(bytes) = &self.mvec {
            std::fs::write(dir.join("mvec.rg16f"), bytes)?;
            files.push("mvec.rg16f".to_string());
        }
        let preview = preview_rgba8(&self.colour, pw, self.width, self.height);
        if crate::dump::write_png(&dir.join("colour-preview.png"), &preview, self.width, self.height, false, false).is_ok() {
            files.push("colour-preview.png".to_string());
        }
        let meta = format!(
            "{{\n  \"width\": {},\n  \"height\": {},\n  \"format\": \"R16G16B16A16_SFLOAT\",\n  \"padded_width\": {pw},\n  \"padded_height\": {ph},\n  \
             \"colour\": \"colour.rgba16f (padded_width x padded_height, little-endian half floats, scene-linear)\",\n  \
             \"depth\": {},\n  \"depth_format\": \"{}\",\n  \"mvec\": {},\n  \"mvec_format\": \"R16G16_SFLOAT\",\n  \"mvec_scale\": [{}, {}],\n  \"frame\": {}\n}}\n",
            self.width,
            self.height,
            depth_name.as_ref().map_or_else(|| "null".to_string(), |(n, ..)| format!("\"{n}\"")),
            depth_name.as_ref().map_or_else(|| "none".to_string(), |(.., f)| format!("{f:?}")),
            if self.mvec.is_some() { "\"mvec.rg16f\"" } else { "null" },
            self.mvec_scale[0],
            self.mvec_scale[1],
            self.frame
        );
        std::fs::write(dir.join("meta.json"), meta)?;
        files.push("meta.json".to_string());
        if !self.exposure.is_empty() {
            std::fs::write(dir.join("exposure.json"), exposure_json(&self.exposure))?;
            files.push("exposure.json".to_string());
        }
        Ok(files)
    }
}

/// The channels of a 1x1 exposure texel as floats (halves or singles, little-endian).
pub(crate) fn exposure_channels(format: vk::Format, bytes: &[u8]) -> Vec<f32> {
    let half = matches!(format, vk::Format::R16_SFLOAT | vk::Format::R16G16_SFLOAT | vk::Format::R16G16B16A16_SFLOAT);
    if half {
        bytes.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect()
    } else {
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }
}

/// `exposure.json`: every registered 1x1 float image at the hold (DLSS's exposure input among
/// them), with its raw bytes (hex) and their values. Non-finite values are written as strings.
fn exposure_json(values: &[ExposureValue]) -> String {
    let number = |v: f32| if v.is_finite() { format!("{v:e}") } else { format!("\"{v}\"") };
    let entries: Vec<String> = values
        .iter()
        .map(|e| {
            let (raw, decoded) = match &e.bytes {
                Some(b) => (
                    format!("\"{}\"", b.iter().map(|x| format!("{x:02x}")).collect::<String>()),
                    format!("[{}]", exposure_channels(e.format, b).into_iter().map(number).collect::<Vec<_>>().join(", ")),
                ),
                None => ("null".to_string(), "null".to_string()),
            };
            format!(
                "    {{\"image\": \"{:#x}\", \"format\": \"{:?}\", \"layout\": {}, \"layout_assumed\": {}, \"raw_le_hex\": {raw}, \"values\": {decoded}}}",
                e.image,
                e.format,
                e.layout.map_or_else(|| "null".to_string(), |l| format!("\"{l:?}\"")),
                e.assumed
            )
        })
        .collect();
    format!(
        "{{\n  \"note\": \"registered 1x1 float images at the DLSS submit, read before DLSS ran (the game's exposure input among them)\",\n  \"images\": [\n{}\n  ]\n}}\n",
        entries.join(",\n")
    )
}

// ---- Per-device session: resources, statistics, status. ----

/// Hold statistics for the `[preupscale]` line and the header.
#[derive(Default)]
struct Stats {
    holds: u64,
    hold_ms: Vec<f32>,
    capture_gpu_ms: Vec<f32>,
    writeback_gpu_ms: Vec<f32>,
    network_gpu_ms: Vec<f32>,
    /// The native hold's depth copy (`native::DepthCopy`).
    depth_gpu_ms: Vec<f32>,
    /// The edit cache (`cache`): its passes' GPU time, the window's refresh and cached holds (and
    /// those in the still regime), the session's, the last cached hold's rejected share, and the
    /// median published at the last summary.
    cache_gpu_ms: Vec<f32>,
    window_refresh: u32,
    window_cached: u32,
    window_still: u32,
    refresh_holds: u32,
    cached_holds: u32,
    last_rejected: Option<f32>,
    /// History resets for a scene cut (`HoldResult::scene_cut`) since the path started.
    scene_cuts: u32,
    cache_gpu_published: Option<f32>,
    /// [`HoldTiming`]'s phases in milliseconds, per hold that reached them.
    prep_ms: Vec<f32>,
    capture_wait_ms: Vec<f32>,
    /// The longest capture fence wait since the path started (the window's is the maximum of
    /// `capture_wait_ms`): the worst case [`FRAME_CAPTURE_WAIT`] has to leave room for.
    capture_wait_session_max_ms: f32,
    writeback_cpu_ms: Vec<f32>,
    misses: u32,
    window_misses: u32,
    last_miss_log: Option<Instant>,
    unlogged_misses: u32,
    last_hold_ms: f32,
    /// When the current summary window started: the previous summary, or the first hold.
    window_start: Option<Instant>,
    /// The HDR modes' exposure values in the window, and where the last one came from.
    exposure: Vec<f32>,
    exposure_source: Option<ExposureSource>,
}

/// Holds per second over a summary window of `holds` holds that took `elapsed`.
fn per_second(holds: u64, elapsed: Duration) -> f32 {
    let secs = elapsed.as_secs_f32();
    if secs > 0.0 {
        holds as f32 / secs
    } else {
        0.0
    }
}

impl Stats {
    fn book(&mut self, t: &HoldTiming) {
        let ms = |d: Duration| d.as_secs_f32() * 1000.0;
        let pairs = [(&mut self.prep_ms, t.prep), (&mut self.capture_wait_ms, t.capture_wait), (&mut self.writeback_cpu_ms, t.writeback)];
        for (v, d) in pairs {
            if let Some(d) = d {
                v.push(ms(d));
            }
        }
        if let Some(wait) = t.capture_wait {
            self.capture_wait_session_max_ms = self.capture_wait_session_max_ms.max(ms(wait));
        }
    }

    /// The phase medians, for the periodic line.
    fn phases(&self) -> String {
        format!(
            "phases ms (median): prep={:.2} capture_wait={:.2} writeback={:.2}; capture_wait max={:.2} (session max {:.2}, bound {} ms)",
            median(&self.prep_ms),
            median(&self.capture_wait_ms),
            median(&self.writeback_cpu_ms),
            self.capture_wait_ms.iter().copied().fold(0.0, f32::max),
            self.capture_wait_session_max_ms,
            FRAME_CAPTURE_WAIT.as_millis()
        )
    }

    fn clear_phases(&mut self) {
        for v in [&mut self.prep_ms, &mut self.capture_wait_ms, &mut self.writeback_cpu_ms] {
            v.clear();
        }
    }
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// A line the tracker decided on under its lock, logged by the caller once the lock is let go: [`crate::logging::flush`]
/// waits up to 100 ms, and other threads' recording waits on the lock.
#[must_use]
pub(crate) struct Said {
    line: String,
    /// A one-off line (flushed); a periodic one is not.
    flush: bool,
}

impl Said {
    fn say(self) {
        crate::log!("[preupscale] {}", self.line);
        if self.flush {
            crate::logging::flush();
        }
    }
}

/// [`Session::check_exposure`]'s test for a colour input the game already exposed, for one identification.
#[derive(Clone, Copy, Debug, Default)]
struct PreExposed {
    identification: u64,
    /// The frame is measured, not exposed with the game's ([`hdr::AutoUse::Beside`]).
    measuring: bool,
    /// The run of holds toward the other state ([`PREEXPOSED_WINDOW`]).
    run: Option<Run>,
}

/// See [`PreExposed::run`].
#[derive(Clone, Copy, Debug)]
struct Run {
    /// The run's first hold, and its last, on the holds' clock.
    since: Instant,
    last: Instant,
    /// Its holds toward the other state (not those between).
    holds: u32,
    /// The lowest and highest game exposure in it, log2.
    lo: f32,
    hi: f32,
}

/// The holds' clock ([`native::hold_now`]: a test can stop it).
fn hold_clock() -> Instant {
    #[cfg(target_arch = "x86_64")]
    return native::hold_now();
    #[cfg(not(target_arch = "x86_64"))]
    Instant::now()
}

/// One device's pre-upscaler state, kept in its `State` (so holds and presents are serialized by
/// the same lock).
#[derive(Default)]
pub(crate) struct Session {
    /// The resources of the split holds before the upscaler (at DLSS's input size).
    pub(crate) res: Option<Resources>,
    /// The resources of the holds before the upscaler recorded into DLSS's buffer ([`inline`]): their own
    /// (no fences, no timers of the passes', the hold guard), beside the split hold's.
    pub(crate) res_inline: Option<Resources>,
    /// The resources of the hold after DLSS's output (at the output size, [`placement::Stage::Upscaled`]): its
    /// own network pass, histories, scene-cut detector and control mask, beside the holds before the upscaler
    /// in the same frame.
    pub(crate) res_after: Option<Resources>,
    stats: Stats,
    last_hold: Option<Instant>,
    /// The last hold ran the native backend (for `native_running`).
    last_hold_native: bool,
    dumped: bool,
    /// Launch-bearing submits a due dump waited for the depth and motion-vector layouts.
    dump_waits: u32,
    said: HashSet<&'static str>,
    /// The first split hold that wrote back, and the first hold recorded into DLSS's buffer that was read back
    /// (it ran), were said ([`Self::note_first_split`], [`Self::harvest`]).
    first_split_said: bool,
    first_inside_ran_said: bool,
    /// Until when the after-the-upscaler path keeps off slot 0: hold resources were left behind
    /// without a fence ([`Self::recover_unfenced`]), and their write-back may still read it.
    post_held_off_until: Option<Instant>,
    /// Split-hold resources left without a fence for their last work, until a later fence on their queue
    /// signals ([`Self::bury`]) or the device is torn down.
    left_behind: Vec<Resources>,
    /// The last launch-bearing submit with DLSS's inputs identified (held or not).
    last_dlss: Option<Instant>,
    /// A model-mode hold on this device ran the network: the pre-upscaler path works here, so the
    /// post path stays off while DLSS runs ([`post_off`]).
    engaged: bool,
    /// The identification and exposure source last logged ([`Self::note_exposure_source`]).
    exposure_said: Option<(u64, ExposureSource)>,
    /// The brightness and exposure setting last logged ([`Self::view`]).
    view_said: Option<(f32, bool)>,
    /// The extent of the last hold that wrote the frame back: DLSS's input before the upscaler, its output after
    /// it ([`Self::publish_status`]).
    last_hold_extent: Option<(u32, u32)>,
    /// The identification whose game exposure read implausibly high or jumped ([`Self::check_exposure`]):
    /// its holds measure the exposure from the frame instead.
    exposure_untrusted: Option<u64>,
    /// The game's exposure the last checked hold read, and its identification.
    last_game_exposure: Option<(u64, f32)>,
    /// Jumps of the game's exposure in the identification of [`Self::last_game_exposure`], and the
    /// holds since the last one ([`EXPOSURE_JUMPS`]).
    exposure_jumps: (u32, u32),
    /// The identification and the holds in a row its game exposure read unusable ([`EXPOSURE_UNUSABLE_HOLDS`]).
    exposure_unusable: (u64, u32),
    /// The test for a colour input the game already exposed ([`PREEXPOSED_WINDOW`]).
    exposure_preexposed: PreExposed,
    /// When [`Session::say_pre_exposed`] last logged each direction (into the game's exposure, into the measured
    /// one), and the changes it left out since.
    exposure_preexposed_said: [(Option<Instant>, u32); 2],
    /// The native backend's network on this device, when it was created for it.
    pub(crate) native: Option<NativeLoader>,
    /// The native backend's after-the-upscaler server (`native_post`), once the mapping is open.
    #[cfg(target_arch = "x86_64")]
    pub(crate) post_server: Option<native_post::PostServer>,
    /// When to start the server again after it ended for a reason that may pass (its pipelines could not be
    /// built, another server held the channel): the native loader's schedule.
    #[cfg(target_arch = "x86_64")]
    post_retry: neural_forge_protocol::rebuild::BuildRetry,
    /// The server stalled on a frame: nothing starts on this device again.
    #[cfg(target_arch = "x86_64")]
    post_stalled: bool,
    /// The server running is the one on another GPU.
    #[cfg(target_arch = "x86_64")]
    post_offload: bool,
    /// `NEURAL_FORGE_CAPTURE_AT`: how many times DLSS frames resumed after a gap, when the last resume was,
    /// and which of its entries were taken, one bit each ([`Self::capture_at_due`]).
    resumes: u32,
    resumed_at: Option<Instant>,
    captured_at: u64,
    /// The native backend's status line last published (`layer_reason`), so it is written on change only.
    #[cfg(target_arch = "x86_64")]
    native_status: String,
    /// Consecutive frames still to dump after the one just taken ([`dump_frames`]).
    burst: u32,
    /// Launch-bearing submits with DLSS's inputs identified so far ([`Self::saw_dlss`]): a cache
    /// dump's holds must be consecutive ones.
    seen: u64,
    /// The K-hold cache dump ([`cache_dump`]).
    cache_dump: cache_dump::CacheDump,
    /// This device's entry in [`POST_OFF_UNTIL`].
    post_off: PostOffEntry,
}

/// One session's entry in [`POST_OFF_UNTIL`], removed with the session (the device).
pub(crate) struct PostOffEntry(u64);

impl Default for PostOffEntry {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

impl PostOffEntry {
    fn set(&self, until: Option<Instant>) {
        let mut all = lock(&POST_OFF_UNTIL);
        match until {
            Some(until) => all.insert(self.0, until),
            None => all.remove(&self.0),
        };
    }
}

impl Drop for PostOffEntry {
    fn drop(&mut self) {
        lock(&POST_OFF_UNTIL).remove(&self.0);
    }
}

/// `NEURAL_FORGE_PROBE_KERNEL=<name prefix>`: logs the parameters of the first 3000 launches of kernels
/// whose name starts with the prefix, as 4-byte words (hex and f32), for finding a field such as the
/// camera jitter in a kernel the layer does not know yet. Diagnostic only.
fn probe_kernel(name: &str, bytes: &[u8]) {
    static PREFIX: LazyLock<Option<String>> = LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_PROBE_KERNEL").filter(|p| !p.is_empty()));
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let Some(prefix) = PREFIX.as_deref() else { return };
    if !name.starts_with(prefix) || COUNT.fetch_add(1, Ordering::Relaxed) >= 3000 {
        return;
    }
    let words: Vec<String> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| {
            let v = u32::from_le_bytes(*w);
            format!("{v:08x}:{}", f32::from_bits(v))
        })
        .collect();
    crate::log!("[probe-kernel] {name} {}", words.join(" "));
}

/// `NEURAL_FORGE_PREUPSCALE_DUMP_HAZARD=ignore`: dump even when DLSS's launch buffer synchronizes
/// the depth or motion vectors before its launch ([`Scan::dump_hazard`]). Diagnostic only: those
/// images may then be read before the game finished them, which consecutive dumps show.
pub(crate) fn dump_ignores_hazard() -> bool {
    static IGNORE: LazyLock<bool> = LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_PREUPSCALE_DUMP_HAZARD").is_some_and(|v| v == "ignore"));
    *IGNORE
}

/// One `NEURAL_FORGE_CAPTURE_AT` entry, `<resume>:<ms>[:<frames>]`, parsed ([`Session::capture_at_due`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct CaptureAt {
    resume: u32,
    after: Duration,
    frames: u32,
}

impl CaptureAt {
    /// `None` for anything malformed, and for a series of 0 frames.
    fn parse(v: &str) -> Option<Self> {
        let mut parts = v.split(':').map(str::trim);
        let resume = parts.next()?.parse().ok()?;
        let after = Duration::from_millis(parts.next()?.parse().ok()?);
        let frames = parts.next().map_or(Some(1), |f| f.parse().ok().filter(|&n| n >= 1))?;
        parts.next().is_none().then_some(Self { resume, after, frames })
    }

    /// Entries separated by commas (at most 64); `None` when any is malformed.
    fn parse_list(v: &str) -> Option<Vec<Self>> {
        v.split(',').map(Self::parse).collect::<Option<Vec<_>>>().filter(|l| !l.is_empty() && l.len() <= 64)
    }
}

/// A pause in DLSS frames that counts as a resume for `NEURAL_FORGE_CAPTURE_AT` ([`Session::capture_at_due`]).
const CAPTURE_GAP: Duration = Duration::from_secs(3);

/// `NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES`: how many consecutive DLSS frames each dump takes (default
/// 1). Two or more give frame pairs for checking the motion vectors' convention.
pub(crate) fn dump_frames() -> u32 {
    static FRAMES: LazyLock<u32> =
        LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES").and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(1));
    *FRAMES
}

impl Session {
    /// In model mode, whether the post-upscaler compose must stay off for this device's present
    /// ([`post_off`]).
    pub(crate) fn suppresses_post(&self) -> bool {
        let now = Instant::now();
        self.post_kept_off(mode() == Mode::Model, post_serves(now), sr_launches(), before_dlss(), now)
    }

    /// [`Self::suppresses_post`] at `now` in model mode (`model`), with the after-the-upscaler path serving the passes or
    /// not (`serves`, [`post_serves`]) and Super Resolution's `launches`: off until the hand-back after the last held
    /// frame ([`post_off`]), and in a DLSS game before DLSS's first frame (`before_dlss`, [`before_dlss`]) until the
    /// hand-back from the sign ([`pre_dlss_post_off`]).
    fn post_kept_off(&self, model: bool, serves: bool, launches: Option<SrLaunches>, before_dlss: Option<Instant>, now: Instant) -> bool {
        model && !serves && (post_off(self.engaged, self.last_hold, launches, now) || pre_dlss_post_off(before_dlss, launches, now))
    }

    /// Whether this device's network is paused by the VRAM guard ([`native::Loader::paused`]): the post path then has
    /// nothing to run and presents the frames as they are, with no capture (it captured and echoed every frame at the
    /// output size, Cyberpunk 2077 at 4K with Ray Reconstruction at half its frame rate after the pause, #59).
    pub(crate) fn network_paused(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        return self.native.as_ref().is_some_and(|l| l.paused());
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// How many launch-bearing submits with DLSS's inputs identified reached the hold point
    /// ([`Self::saw_dlss`]): the number of the one being held (the edit cache tells consecutive holds by it).
    pub(crate) fn seen(&self) -> u64 {
        self.seen
    }

    /// A launch-bearing submit with DLSS's inputs identified reached the hold point.
    pub(crate) fn saw_dlss(&mut self) {
        // A resume after a gap: a loading screen, the start of a benchmark pass.
        if self.last_dlss.is_none_or(|t| t.elapsed() >= CAPTURE_GAP) {
            self.resumes += 1;
            self.resumed_at = Some(Instant::now());
            // Always: a pause menu or loading screen and how long it kept DLSS paused (#49).
            if let Some(gap) = self.last_dlss.map(|t| t.elapsed()) {
                crate::log!("[preupscale] DLSS frames resumed (#{}) after {} ms", self.resumes, gap.as_millis());
            }
        }
        self.last_dlss = Some(Instant::now());
        self.seen += 1;
        self.share_post_off();
    }

    /// [`Self::saw_dlss`] for a hold being recorded into DLSS's buffer `cb`: once per buffer. With passes both before
    /// and after DLSS a buffer carries two holds, and the first one recorded counted its frame already: counted again,
    /// the edit cache saw a gap between consecutive holds and a cache dump's arm check never fired.
    pub(crate) fn saw_dlss_inline(&mut self, inline: &inline::Inline, cb: vk::CommandBuffer) {
        if !inline.owns_at(cb, None) {
            self.saw_dlss();
        }
    }

    /// Starts a requested cache dump ([`cache_dump::CacheDump::poll`]), then whether this model-mode
    /// hold (the DLSS submit [`Self::saw_dlss`] just counted) is armed for one. `copyable`: the colour
    /// input can be copied from (TRANSFER_SRC); a dump cannot run on one that cannot.
    pub(crate) fn cache_dump_arm(&mut self, shm: &ShmClient, inline: bool, copyable: bool) -> Option<cache_dump::Arm> {
        self.cache_dump.poll(shm, self.resumes, self.resumed_at);
        if !copyable && self.cache_dump.running() {
            self.cache_dump.end("the colour input lacks TRANSFER_SRC");
            return None;
        }
        self.cache_dump.arm(self.seen, inline)
    }

    /// Books a model-mode hold's part in a cache dump: the readback it brought, and whether the hold
    /// it was armed for (`arm`) wrote back.
    pub(crate) fn cache_dump_after(&mut self, arm: Option<cache_dump::Arm>, result: &mut HoldResult) {
        self.cache_dump.after_hold(arm, result.wrote_back, result.cache_frame.take());
    }

    /// Publishes, process-wide, until when this device keeps the post path off
    /// ([`post_off_by_any_device`]): a game whose DLSS runs on a device other than the one it
    /// presents on (no swapchain on the DLSS device, identified against an output-like image) must
    /// not get the model twice, before DLSS on one device and after it on the other.
    fn share_post_off(&self) {
        self.post_off.set(post_off_since(mode() == Mode::Model, self.engaged, self.last_hold));
    }

    /// Whether a hold happened within [`RECENT`].
    pub(crate) fn holding(&self) -> bool {
        self.last_hold.is_some_and(|t| t.elapsed() < RECENT)
    }

    /// Whether a dump is due: the first hold after the input is identified, then whenever a
    /// one-shot `capture_request` is pending (the caller consumes it once the dump is taken).
    pub(crate) fn dump_due(&self, shm: &ShmClient) -> bool {
        !self.dumped || self.burst > 0 || shm.capture_request_pending()
    }

    /// Whether a due dump should be taken now: once the depth and motion-vector images' layouts are
    /// known (`layouts_known`; they are learned from the game's barriers after the inputs are
    /// identified, normally within a frame), or after 120 launch-bearing submits without them (the
    /// dump then has colour only).
    pub(crate) fn dump_ready(&mut self, layouts_known: bool) -> bool {
        if layouts_known || self.dump_waits >= 120 {
            self.dump_waits = 0;
            true
        } else {
            self.dump_waits += 1;
            false
        }
    }

    /// After a split hold that wrote back: the first one on this device is said, and flushed at once (a crash
    /// right after it still shows it).
    pub(crate) fn note_first_split(&mut self, extent: (u32, u32), native: bool) {
        if !self.first_split_said {
            self.first_split_said = true;
            crate::log!(
                "[preupscale] first split hold written back on this device: {}x{} ({}); the game's DLSS submit follows it",
                extent.0,
                extent.1,
                if native { "the network ran" } else { "a diagnostic mode" }
            );
            crate::logging::flush();
        }
    }

    /// Logs `what` once per process-session for this device.
    pub(crate) fn say_once(&mut self, what: &'static str) {
        note_refusal(Refusal::Text(what.into()));
        if self.said.insert(what) {
            crate::log!("[preupscale] {what}");
            crate::logging::flush();
        }
    }

    /// Hold resources that a failed submit left without a fence ([`Resources::fence_submitted`]) are
    /// let go of on a device that is not lost: set aside, not destroyed (what went to the queue may still
    /// use them), so the next hold builds new ones and the after-the-upscaler path is not kept waiting
    /// on them. They are destroyed once a fence submitted after them on their queue was seen signalled
    /// ([`Self::bury`]), or at device teardown. On a lost device they stay, and nothing holds there again.
    /// Only the split hold's [`Self::res`] has fences: the holds recorded into DLSS's buffers
    /// ([`Self::res_inline`], [`Self::res_after`]) have no work of their own.
    pub(crate) fn recover_unfenced(&mut self, device: &ash::Device) {
        self.bury(device);
        if !self.res.as_ref().is_some_and(|r| r.unfenced) || crate::device_lost(device.handle()) {
            return;
        }
        crate::log!(
            "[preupscale] hold resources without a fence for their last work are set aside until a later fence on their queue signals, and built again"
        );
        if let Some(old) = self.res.take() {
            self.left_behind.push(old);
        }
        // What they went to the queue with may still read slot 0, which the after-the-upscaler
        // path writes: it keeps off as long as any wait of the layer's is bounded (on the native
        // path the network's loader holds it off the same way, `Loader::pre_unfenced`).
        self.post_held_off_until = Some(Instant::now() + crate::FENCE_WAIT_TIMEOUT);
    }

    /// Destroys the resources left behind ([`Self::recover_unfenced`]) on the queue the split hold's current
    /// ones work on, once one of their fences was seen signalled: those resources were built after the left
    /// ones were set aside, so that fence was submitted after the left ones' last work, and a fence's signal
    /// covers everything submitted on its queue before it (`vkQueueSubmit`'s fence signal operation).
    pub(crate) fn bury(&mut self, device: &ash::Device) {
        let Some(after) = self.res.as_ref().filter(|r| r.fences_seen > 0 && !r.unfenced).and_then(|r| r.queue) else { return };
        let mut k = 0;
        while k < self.left_behind.len() {
            if self.left_behind[k].queue == Some(after) {
                let old = self.left_behind.swap_remove(k);
                // SAFETY: everything submitted with it on its queue completed (above).
                unsafe { old.destroy(device) };
            } else {
                k += 1;
            }
        }
    }

    /// Whether the after-the-upscaler path must leave slot 0 alone this present (see
    /// [`Self::recover_unfenced`]).
    pub(crate) fn post_held_off(&self) -> bool {
        self.post_held_off_until.is_some_and(|until| Instant::now() < until)
    }

    /// Waits for the layer's own pending work (bounded). `true` when none is left. The holds recorded into
    /// DLSS's buffers have none of the layer's own: they are part of the application's.
    pub(crate) fn drain(&mut self, device: &ash::Device) -> bool {
        let mut gpu = None;
        self.res.as_mut().is_none_or(|r| r.wait_idle(device, &mut gpu))
    }

    /// The resources of the holds recorded into DLSS's buffer at `stage`: [`Self::res_inline`] before the
    /// upscaler, [`Self::res_after`] after DLSS's output (only ever held there).
    pub(crate) fn res_inline(&mut self, stage: placement::Stage) -> &mut Option<Resources> {
        match stage {
            placement::Stage::Upscaled => &mut self.res_after,
            _ => &mut self.res_inline,
        }
    }

    /// Makes sure the resources of the holds recorded into DLSS's buffer at `stage` are built for this extent
    /// and queue family. Old ones are destroyed only once no hold recorded with them may still run
    /// (`outstanding` is 0): until then the hold records nothing (`Err`, why).
    ///
    /// # Safety
    /// `device` is live.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ensure_inline(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        stage: placement::Stage,
        outstanding: usize,
    ) -> Result<(), &'static str> {
        let slot = self.res_inline(stage);
        if slot.as_ref().is_some_and(|r| r.inline && r.matches(queue_family, width, height)) {
            return Ok(());
        }
        if slot.is_some() && outstanding > 0 {
            return Err("waiting for the holds recorded in DLSS's buffers before building the hold's resources for another extent");
        }
        if let Some(old) = slot.take() {
            // SAFETY: no hold recorded with them may still run, and they have no work of their own.
            unsafe { old.destroy(device) };
        }
        // SAFETY: forwarded.
        *slot = unsafe { Resources::build_inline(device, instance, physical_device, queue_family, width, height) };
        if slot.is_none() {
            self.say_once("could not build the resources of the hold inside DLSS's buffer; not holding there");
            return Err("the hold's resources could not be built");
        }
        crate::log!("[preupscale] resources of the hold inside DLSS's buffer for {width}x{height} built ({})", stage.name());
        crate::logging::flush();
        Ok(())
    }

    /// What a hold recorded into DLSS's buffer left, read back once its buffer ran ([`inline::Harvest`]): its
    /// GPU times, the edit cache's counters, the scene-cut reading, the exposure it used, a cache dump's frame.
    /// Applied to the resources it was recorded with (dropped for resources built again since), and reported
    /// with the next hold ([`Resources::readings`]).
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn harvest(&mut self, h: inline::Harvest) {
        if let Some(frame) = h.cache_frame {
            self.cache_dump.after_hold(None, true, Some(frame));
        }
        if !self.first_inside_ran_said {
            self.first_inside_ran_said = true;
            crate::log!("[preupscale] the first hold recorded into DLSS's command buffer ran (its buffer finished on the GPU and was read back)");
            crate::logging::flush();
        }
        let Some(res) = self.res_inline(h.meta.stage).as_mut().filter(|r| r.id == h.meta.resources) else { return };
        let word = |at: u64, k: usize| {
            let i = at as usize + 4 * k;
            u32::from_le_bytes([h.readback[i], h.readback[i + 1], h.readback[i + 2], h.readback[i + 3]])
        };
        let r = &mut res.readings;
        if h.meta.timed {
            let [capture, network, writeback] = h.gpu_ms;
            r.capture_gpu_ms = capture.or(r.capture_gpu_ms);
            if h.meta.kind != Some(cache::Kind::Cached) {
                r.network_gpu_ms = network.or(r.network_gpu_ms);
            }
            r.writeback_gpu_ms = writeback.or(r.writeback_gpu_ms);
        }
        if let Some(n) = res.native.as_mut().filter(|n| h.meta.fresh.0 == n.id()) {
            n.inline_ran(h.meta.fresh.1);
        }
        if !h.sound {
            // A guessed input layout was wrong when the buffer ran (`inline::Inline::check`): the guard reset
            // what it carries (the network's history, the edit cache's stored edit) for this hold and the one
            // after it; what it measured from those inputs is not used either. The CPU's plan follows the GPU:
            // the next hold planned refreshes (a cached one would find no edit to apply and give DLSS its frame
            // unedited, every other frame while the camera moves), and the next network frame starts afresh
            // (no history, seed 0) as the GPU's did.
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(e) = res.edit.as_mut() {
                    e.state.drop_edit();
                }
                if let Some(n) = res.native.as_mut() {
                    n.gap.unsound();
                }
            }
            return;
        }
        if let (true, Some(kind), Some(edit)) = (h.meta.stats, h.meta.kind, res.edit.as_mut()) {
            let counters: [u32; cache::STAT_WORDS] = std::array::from_fn(|k| word(inline::READ_STATS, k));
            if let Some(rejected) = edit.after_counters(kind, h.meta.refresh, &counters).rejected {
                res.readings.cache_rejected = Some(rejected);
            }
        }
        if let Some(recorded) = h.meta.scene {
            let reading = scene_cut::ScenePass::reading_of(std::array::from_fn(|k| word(inline::READ_SCENE, k)));
            let effect = scene_cut::effect(reading, recorded);
            res.readings.scene_cut |= native::apply_scene_cut(effect, res.native.as_mut(), res.edit.as_mut().map(|e| &mut e.state));
        }
        native::apply_exposure_jump(word(inline::READ_EXPOSURE, 1), res.native.as_mut());
        if h.meta.exposure_source == Some(ExposureSource::Game) {
            let at = inline::READ_EXPOSURE as usize;
            let (_, game) = hdr::exposure_word([h.readback[at], h.readback[at + 1], h.readback[at + 2], h.readback[at + 3]]);
            res.readings.previous_exposure = Some((h.meta.identification, game));
        }
        if h.meta.auto {
            res.readings.auto = Some(hdr::AutoState::from_words(std::array::from_fn(|k| word(inline::READ_AUTO, k))));
        }
    }

    /// Whether any of the holds' resources exist.
    pub(crate) fn has_res(&self) -> bool {
        self.res.is_some() || self.res_inline.is_some() || self.res_after.is_some()
    }

    /// # Safety
    /// The device must be idle (teardown).
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        let left = std::mem::take(&mut self.left_behind);
        for res in [self.res.take(), self.res_inline.take(), self.res_after.take()].into_iter().flatten().chain(left) {
            // SAFETY: forwarded.
            unsafe { res.destroy(device) };
        }
    }

    /// Makes sure the split hold's resources ([`Self::res`]) are built for this extent and queue family,
    /// draining and rebuilding the old ones only when nothing of them is pending. `false` when there is
    /// nothing usable, and for any `stage` but [`placement::Stage::Before`]: the split holds run only before
    /// the upscaler. After DLSS's output the hold is recorded into DLSS's buffer, with
    /// [`Self::res_after`]'s fence-less resources, which [`Self::ensure_inline`] destroys without a wait and
    /// [`Self::drain`] never waits for; fenced resources there could be destroyed while still pending.
    ///
    /// # Safety
    /// As [`Resources::build`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn ensure(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        shm: &ShmClient,
        import: bool,
        stage: placement::Stage,
    ) -> bool {
        if stage != placement::Stage::Before {
            return false;
        }
        self.recover_unfenced(device);
        // Taken out while it is looked at (the cache dump's state is the session's).
        let mut slot = self.res.take();
        // SAFETY: forwarded.
        let ok = unsafe { self.ensure_in(&mut slot, device, instance, physical_device, queue_family, width, height, shm, import) };
        self.res = slot;
        ok
    }

    /// [`Self::ensure`] on the split hold's resources, `res`.
    ///
    /// # Safety
    /// As [`Resources::build`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn ensure_in(
        &mut self,
        res: &mut Option<Resources>,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        shm: &ShmClient,
        import: bool,
    ) -> bool {
        // A staged set serves either way; an imported one only where imports are wanted.
        if res.as_ref().is_some_and(|r| r.matches(queue_family, width, height) && (import || !r.imported())) {
            return true;
        }
        if let Some(mut old) = res.take() {
            let mut gpu = None;
            if !old.wait_idle(device, &mut gpu) {
                *res = Some(old);
                return false;
            }
            if old.cache_pending() {
                self.cache_dump.lost();
            }
            // SAFETY: drained just above.
            unsafe { old.destroy(device) };
        }
        // SAFETY: forwarded.
        *res = unsafe { Resources::build(device, instance, physical_device, queue_family, width, height, shm, import) };
        match res {
            Some(r) => {
                let (pw, ph) = padded(width, height);
                crate::log!(
                    "[preupscale] resources for {width}x{height} (padded {pw}x{ph}) built: {}",
                    if r.imported() { "zero-copy (SHM regions imported)" } else { "staged (host copies into the SHM regions)" }
                );
                crate::logging::flush();
                true
            }
            None => {
                self.say_once("could not build the hold's resources; not holding");
                false
            }
        }
    }

    /// Logs, once per identification of DLSS's inputs (and again if it changes), where the HDR
    /// encode's exposure comes from: the game's 1x1 R16F, or measured from the frame.
    /// Whether the game's exposure image may be used for `identification` (not after it read
    /// implausibly high, [`Self::check_exposure`]).
    pub(crate) fn game_exposure_trusted(&self, identification: u64) -> bool {
        self.exposure_untrusted != Some(identification)
    }

    /// After a hold that read the game's exposure: a value over [`EXPOSURE_TRUST_MAX`] is not an
    /// exposure (Black Myth: Wukong's benchmark uses DLSS's own auto-exposure; the 1x1 R16F its DLSS
    /// buffer names read 0.22 in some scenes and up to 59,456 in others, 2026-10-05), and one that
    /// jumped by more than [`EXPOSURE_JUMP_MAX`] between holds [`EXPOSURE_JUMPS`] times within
    /// [`EXPOSURE_JUMP_WINDOW`] holds is not the game's eye adaptation (Hogwarts Legacy, whose
    /// adaptation reacts to the frames the model changed), and one that puts a frame the game already exposed far
    /// below the key ([`PREEXPOSED_STOPS`]) is not one to multiply in (Clair Obscur: Expedition 33). From then on
    /// that identification's holds measure the exposure from the frame. Logged once.
    pub(crate) fn check_exposure(&mut self, identification: u64, result: &HoldResult) {
        self.check_exposure_at(identification, result, hold_clock());
    }

    /// [`Self::check_exposure`] at `now` on the holds' clock.
    fn check_exposure_at(&mut self, identification: u64, result: &HoldResult, now: Instant) {
        if result.exposure_source != Some(ExposureSource::Game) || self.exposure_untrusted == Some(identification) {
            return;
        }
        let Some(read) = result.game_exposure.or(result.exposure).or(result.previous_exposure) else { return };
        // Unusable as the shaders judge it (`hdr::exposure_ok`'s low end): a positive value at or below
        // `EXPOSURE_MIN` (a half float's subnormals) edits nothing either, and must lead to the measured exposure
        // as 0 does. Values the shaders refuse at the top are caught below (`EXPOSURE_TRUST_MAX`).
        if !(read.is_finite() && read > hdr::EXPOSURE_MIN) {
            let (id, holds) = &mut self.exposure_unusable;
            *holds = if *id == identification { *holds + 1 } else { 1 };
            *id = identification;
            if *holds >= EXPOSURE_UNUSABLE_HOLDS {
                self.exposure_untrusted = Some(identification);
                crate::log!(
                    "[preupscale] the game's exposure image read {read} on {EXPOSURE_UNUSABLE_HOLDS} holds in a row: not an exposure; measuring it from the frame for this identification"
                );
                crate::logging::flush();
            }
            return;
        }
        self.exposure_unusable = (identification, 0);
        let e = read;
        if let Some((measuring, why)) = self.pre_exposed(identification, e, result.auto_exposure, now) {
            self.say_pre_exposed(measuring, &why, now);
        }
        let previous = self.last_game_exposure.filter(|(id, _)| *id == identification).map(|(_, p)| p);
        if previous.is_none() {
            self.exposure_jumps = (0, 0);
        }
        self.last_game_exposure = Some((identification, e));
        let jump = previous.filter(|p| (e / p).max(p / e) > EXPOSURE_JUMP_MAX);
        let (jumps, quiet) = &mut self.exposure_jumps;
        if jump.is_some() {
            *jumps += 1;
            *quiet = 0;
        } else {
            *quiet += 1;
            if *quiet > EXPOSURE_JUMP_WINDOW {
                *jumps = 0;
            }
        }
        let why = if e > EXPOSURE_TRUST_MAX {
            format!("read {e:.1} (over {EXPOSURE_TRUST_MAX}): not an exposure")
        } else if let Some(p) = jump.filter(|_| *jumps >= EXPOSURE_JUMPS) {
            format!("jumped {} times by over {EXPOSURE_JUMP_MAX}x between holds (last from {p} to {e}): not the game's eye adaptation", *jumps)
        } else {
            return;
        };
        self.exposure_untrusted = Some(identification);
        crate::log!("[preupscale] the game's exposure image {why}; measuring it from the frame for this identification");
        crate::logging::flush();
    }

    /// [`Self::check_exposure`]'s test for a colour input the game already exposed ([`PREEXPOSED_STOPS`]): the
    /// game's exposure `e` against what the auto-exposure measured from the same hold's frame (`auto`: it runs
    /// beside the game's exposure, [`hdr::AutoUse::Observe`], or is used beside it, [`hdr::AutoUse::Beside`]).
    /// `Some` (whether the frame is now measured, and why) when a run of holds switched it ([`PREEXPOSED_WINDOW`]).
    /// A hold without a measurement changes nothing.
    fn pre_exposed(&mut self, identification: u64, e: f32, auto: Option<hdr::AutoState>, now: Instant) -> Option<(bool, String)> {
        let p = &mut self.exposure_preexposed;
        if p.identification != identification {
            *p = PreExposed { identification, ..PreExposed::default() };
        }
        let a = auto.filter(|a| a.valid && a.samples > 0 && a.target_log2_e.is_finite())?;
        if p.run.is_some_and(|r| now.saturating_duration_since(r.last) > PREEXPOSED_GAP) {
            p.run = None;
        }
        // The measured target is log2(key) - the frame's mean log2 luma: the lift the frame as given needs.
        let below = a.target_log2_e - e.log2();
        let exposed = a.target_log2_e <= PREEXPOSED_FRAME_STOPS;
        let far = exposed && below > PREEXPOSED_STOPS;
        let clear = !exposed || below < PREEXPOSED_CLEAR_STOPS;
        // Toward the other state, or against it (which ends the run); a hold between the two leaves the run.
        let (toward, against) = if p.measuring { (clear, far) } else { (far, clear) };
        if against {
            p.run = None;
            return None;
        }
        let l = e.log2();
        let fresh = Run { since: now, last: now, holds: 0, lo: l, hi: l };
        if p.run.is_none() {
            if !toward {
                return None;
            }
            p.run = Some(fresh);
        }
        let run = p.run.as_mut()?;
        // Into the measured frame only while the game's exposure holds steady: an eye adaptation catching up
        // moves it, and starts the run over from this hold.
        if !p.measuring && l.max(run.hi) - l.min(run.lo) > PREEXPOSED_STEADY_STOPS {
            *run = fresh;
        }
        run.lo = run.lo.min(l);
        run.hi = run.hi.max(l);
        run.last = now;
        run.holds += u32::from(toward);
        let held = now.saturating_duration_since(run.since);
        if run.holds < PREEXPOSED_HOLDS || held < PREEXPOSED_WINDOW {
            return None;
        }
        let (holds, lo, hi) = (run.holds, run.lo, run.hi);
        p.run = None;
        p.measuring.toggle();
        let seconds = held.as_secs_f32();
        Some(if p.measuring {
            (
                true,
                format!(
                    "the game's exposure {e} puts the frame {below:.1} stops below the brightness the network is tuned for over {seconds:.1} s ({holds} holds), held within {:.2} stops, while the frame as given (log-average luma {:.3}) is within {:.1} stops of it",
                    hi - lo,
                    a.mean_log2.exp2(),
                    a.target_log2_e.max(0.0)
                ),
            )
        } else {
            (
                false,
                format!(
                    "the game's exposure {e} puts the frame {below:.1} stops from the brightness the network is tuned for (log-average luma {:.3} as given) over {seconds:.1} s ({holds} holds)",
                    a.mean_log2.exp2()
                ),
            )
        })
    }

    /// Logs a switch of [`Self::pre_exposed`]: each direction at most once per [`PREEXPOSED_SAY_EVERY`], with the
    /// changes left out since counted.
    fn say_pre_exposed(&mut self, measuring: bool, why: &str, now: Instant) {
        let (said, unsaid) = &mut self.exposure_preexposed_said[usize::from(measuring)];
        if said.is_some_and(|t| now.saturating_duration_since(t) < PREEXPOSED_SAY_EVERY) {
            *unsaid += 1;
            return;
        }
        let more = match *unsaid {
            0 => String::new(),
            n => format!(" ({n} more such change(s) since the last line went unlogged)"),
        };
        *said = Some(now);
        *unsaid = 0;
        if measuring {
            crate::log!(
                "[preupscale] the colour input looks pre-exposed: {why}; measuring the exposure from the frame for this identification, not bounded by the game's{more}"
            );
        } else {
            crate::log!("[preupscale] the colour input no longer looks pre-exposed: {why}; the game's exposure again for this identification{more}");
        }
        crate::logging::flush();
    }

    /// Whether `identification`'s colour input looks already exposed by the game ([`Self::check_exposure`]): its
    /// holds measure the exposure from the frame, with the game's read beside it ([`Target::pre_exposed`]).
    pub(crate) fn pre_exposed_measured(&self, identification: u64) -> bool {
        let p = &self.exposure_preexposed;
        p.measuring && p.identification == identification
    }

    /// What the network sees before the upscaler for the next hold ([`ShmClient::preupscale_view`]): the
    /// brightness (the paper white is divided by it) and whether the exposure is measured from the frame even
    /// where the game gives one. Logged when it changes (from the defaults on the first hold).
    pub(crate) fn view(&mut self, shm: &ShmClient) -> (f32, bool) {
        let view = shm.preupscale_view();
        if self.view_said != Some(view) {
            if self.view_said.is_some() || view != (1.0, false) {
                let (brightness, measure) = view;
                crate::log!(
                    "[preupscale] what the network sees: brightness {brightness} (paper white {}), exposure {}",
                    hdr::paper_white() / brightness,
                    if measure {
                        format!("measured from the frame (within {}x-{}x of the game's where it gives one)", hdr::LIFT_LO, hdr::LIFT_HI)
                    } else {
                        "the game's where it gives one".to_string()
                    }
                );
            }
            self.view_said = Some(view);
        }
        view
    }

    pub(crate) fn note_exposure_source(&mut self, identification: u64, inputs: &Inputs, result: &HoldResult) {
        let Some(source) = result.exposure_source else { return };
        if self.exposure_said == Some((identification, source)) {
            return;
        }
        self.exposure_said = Some((identification, source));
        let detail = match source {
            ExposureSource::Game => format!(" (image {})", inputs.exposure_input.map_or_else(|| "?".to_string(), |(i, _)| hex(i))),
            ExposureSource::Auto => format!(
                " for colour input {}: trimmed log-average luma, key {}, {}% per frame towards the target{}",
                hex(inputs.colour.0),
                hdr::AUTO_KEY,
                hdr::AUTO_RATE * 100.0,
                match (result.auto_exposure, result.exposure) {
                    (Some(a), Some(e)) => format!("; first frame: mean log2 luma {:.2} over {} samples, e {e:.4}", a.mean_log2, a.samples),
                    _ => String::new(),
                }
            ),
        };
        crate::log!("[preupscale] exposure: {}{detail}", source.describe());
        crate::logging::flush();
    }

    /// Books one hold's result; publishes the header fields and logs misses (sampled) and the
    /// periodic summary.
    pub(crate) fn note(&mut self, shm: &ShmClient, result: &HoldResult, cpu: Duration, extent: (u32, u32)) {
        let now = Instant::now();
        if result.wrote_back || result.dump.is_some() {
            // A frame held: refusals from before it are no longer why nothing is.
            with_last_refusal(|last| *last = None);
            self.last_hold = Some(now);
            self.last_hold_native = result.native;
            self.last_hold_extent = Some(extent);
            self.last_dlss = Some(now);
        } else if let Some(why) = result.miss {
            // A hold that ran and held nothing: its miss is why (the VRAM guard's pause, the network still loading, ...).
            // Without it the status read "identified, not holding" with no reason while the guard had the network
            // paused (Cyberpunk 2077 at 4K with Ray Reconstruction, #59).
            note_refusal(Refusal::Text(why.into()));
        }
        if result.native && result.evaluated {
            self.engaged = true;
        }
        self.share_post_off();
        let holding = self.holding();
        let s = &mut self.stats;
        // A hold is one that submitted its capture; a skipped one only counts as a miss.
        let held = result.waits_consumed;
        if held {
            s.holds += 1;
            s.window_start.get_or_insert_with(Instant::now);
            s.last_hold_ms = cpu.as_secs_f32() * 1000.0;
            s.hold_ms.push(s.last_hold_ms);
        }
        if let Some(g) = result.capture_gpu_ms {
            s.capture_gpu_ms.push(g);
        }
        if let Some(g) = result.writeback_gpu_ms {
            s.writeback_gpu_ms.push(g);
        }
        if let Some(g) = result.network_gpu_ms {
            s.network_gpu_ms.push(g);
        }
        if let Some(g) = result.depth_gpu_ms {
            s.depth_gpu_ms.push(g);
        }
        if let Some(g) = result.cache_gpu_ms {
            s.cache_gpu_ms.push(g);
        }
        if result.cache_rejected.is_some() {
            s.last_rejected = result.cache_rejected;
        }
        if result.scene_cut {
            s.scene_cuts = s.scene_cuts.wrapping_add(1);
        }
        if result.native && result.evaluated {
            if result.cache_still {
                s.window_still += 1;
            }
            if result.cached {
                s.window_cached += 1;
                s.cached_holds = s.cached_holds.wrapping_add(1);
            } else {
                s.window_refresh += 1;
                s.refresh_holds = s.refresh_holds.wrapping_add(1);
            }
        }
        s.book(&result.timing);
        if let Some(e) = result.exposure.filter(|&e| hdr::exposure_ok(e)) {
            s.exposure.push(e);
            s.exposure_source = result.exposure_source;
        }
        // A held frame that went to DLSS untouched (the network not ready, a failed submit, ...).
        if result.waits_consumed && !result.wrote_back && result.dump.is_none() {
            s.misses += 1;
            s.window_misses += 1;
        }
        if let Some(why) = result.miss {
            s.unlogged_misses += 1;
            if s.last_miss_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(5)) {
                crate::log!(
                    "[preupscale] frame went to DLSS untouched: {why} ({} such since the last line, {} answers over budget or echoed in total)",
                    s.unlogged_misses,
                    s.misses
                );
                crate::logging::flush();
                s.last_miss_log = Some(Instant::now());
                s.unlogged_misses = 0;
            }
        }
        shm.publish_preupscale_hold(s.last_hold_ms, s.misses);
        shm.publish_ngx();
        if result.native {
            let gpu = s.cache_gpu_published.or_else(|| s.cache_gpu_ms.last().copied());
            shm.publish_cache(s.refresh_holds, s.cached_holds, s.last_rejected, gpu);
            shm.publish_scene_cuts(s.scene_cuts);
            if let Some(g) = result.network_gpu_ms {
                shm.publish_network_gpu_ms(g);
            }
        }
        shm.publish_preupscale_state(hold_state(holding), self.last_hold_native, extent.0, extent.1);
        if holding {
            *lock(&LAST_HELD) = Some((now, hold_state(true), self.last_hold_native, self.last_hold_extent.unwrap_or(extent)));
        }
        if held && s.holds.is_multiple_of(SUMMARY_EVERY) {
            let (pw, ph) = padded(extent.0, extent.1);
            // The window's rate: frames the model ran on before the upscaler (the post path's
            // "composited" count does not include them). Loading screens inside a window lower it.
            let now = Instant::now();
            let rate = per_second(SUMMARY_EVERY, s.window_start.map_or(Duration::ZERO, |t| now.duration_since(t)));
            s.window_start = Some(now);
            crate::log!(
                "[preupscale] mode={} extent={}x{} (padded {pw}x{ph}) holds={} hold_ms median={:.2} capture_gpu_ms median={:.2} writeback_gpu_ms median={:.2}{}{}{} misses={} (total {}) holds_per_s={:.1}{}",
                mode().name(),
                extent.0,
                extent.1,
                s.holds,
                median(&s.hold_ms),
                median(&s.capture_gpu_ms),
                median(&s.writeback_gpu_ms),
                if s.network_gpu_ms.is_empty() { String::new() } else { format!(" network_gpu_ms median={:.2}", median(&s.network_gpu_ms)) },
                if s.depth_gpu_ms.is_empty() {
                    String::new()
                } else {
                    format!(" depth_copy_gpu_ms median={:.4} max={:.4}", median(&s.depth_gpu_ms), s.depth_gpu_ms.iter().copied().fold(0.0, f32::max))
                },
                if s.window_cached + s.window_refresh == 0 {
                    String::new()
                } else {
                    format!(
                        " cached={} refresh={} still={} cache_gpu_ms median={:.3} rejected={}",
                        s.window_cached,
                        s.window_refresh,
                        s.window_still,
                        median(&s.cache_gpu_ms),
                        s.last_rejected.map_or_else(|| "-".to_string(), |r| format!("{:.1}%", r * 100.0))
                    )
                },
                s.window_misses,
                s.misses,
                rate,
                match s.exposure_source.filter(|_| !s.exposure.is_empty()) {
                    Some(source) => format!(
                        " exposure median={:.4} ({})",
                        median(&s.exposure),
                        if source == ExposureSource::Auto { "auto" } else { "game" }
                    ),
                    None => String::new(),
                }
            );
            crate::log!("[preupscale] {}", s.phases());
            crate::logging::flush();
            s.hold_ms.clear();
            s.exposure.clear();
            s.capture_gpu_ms.clear();
            s.writeback_gpu_ms.clear();
            s.network_gpu_ms.clear();
            s.depth_gpu_ms.clear();
            if !s.cache_gpu_ms.is_empty() {
                s.cache_gpu_published = Some(median(&s.cache_gpu_ms));
            }
            s.cache_gpu_ms.clear();
            s.window_cached = 0;
            s.window_refresh = 0;
            s.window_still = 0;
            s.clear_phases();
            s.window_misses = 0;
        }
    }

    /// Books a write-back GPU reading taken at a later drain.
    pub(crate) fn note_writeback_gpu(&mut self, ms: Option<f32>) {
        if let Some(ms) = ms {
            self.stats.writeback_gpu_ms.push(ms);
        }
    }

    /// Starts the native backend's after-the-upscaler server once the shared memory is open (once per
    /// device), on a device with the native network.
    pub(crate) fn ensure_post_server(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        import: bool,
        shm: &ShmClient,
    ) {
        // The model on another GPU's state, on the Status tab's line and for `doctor` (`layer_reason`).
        #[cfg(target_arch = "x86_64")]
        // Only while the offload is what runs (serving at the finished-frame placement) or not running (starting, stalled,
        // given up): when the placement left it, the native backend's line has the channel.
        let at_present = shm.view().is_some_and(|v| {
            let (count, placements) = v.header().placements();
            placements[..count.max(1)].iter().all(|&p| p == neural_forge_protocol::enums::placement::PRESENT)
        });
        let offload_line = match offload::state() {
            offload::State::Active(_) => offload::active() && at_present,
            _ => true,
        };
        if let Some(line) = offload::status_line().filter(|_| offload_line) {
            // Against what the channel holds (the native backend's line shares it): written when it says something else,
            // except that once the offload was given up the native backend's own line (the network running here, or why
            // not) is the one to keep.
            let now = shm.layer_reason();
            if now != line && !(offload::given_up() && now.starts_with("native network")) {
                shm.set_layer_reason(&line);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if !self.post_stalled && native_on(self.native.as_ref()) {
            use native_post::Status;
            let now = Instant::now();
            match self.post_server.as_ref().map(native_post::PostServer::status) {
                Some(Status::Starting) => return,
                Some(Status::Serving) => {
                    if self.post_retry.failing() {
                        self.post_retry.succeeded();
                    }
                    return;
                }
                // The server on another GPU waits for a frame that did not finish (`native_post::work`): it ends on its
                // own, with the network back here (Unavailable) once that frame is done.
                Some(Status::Stalled) if self.post_offload => return,
                Some(Status::Stalled) => {
                    self.post_stalled = true;
                    drop(self.post_server.take());
                    return;
                }
                // The second GPU could not start, stalled, was lost or was too slow (`offload`, logged there): the
                // network runs on this device's GPU from the next present on.
                Some(Status::Unavailable) => {
                    self.post_offload = false;
                    drop(self.post_server.take());
                    return;
                }
                Some(Status::Retry) => {
                    // The worker has ended: joining it does not wait.
                    self.post_offload = false;
                    drop(self.post_server.take());
                    self.post_retry.failed(now);
                    return;
                }
                None => {}
            }
            if matches!(self.post_retry.step(now), neural_forge_protocol::rebuild::Step::Wait(_)) {
                return;
            }
            let (Some(loader), Some(view)) = (self.native.clone(), shm.view()) else { return };
            // The model on a second GPU (`NEURAL_FORGE_OFFLOAD_GPU`, `offload`): only at the finished-frame placement, and
            // until it is given up for the session.
            // Given up for the process (`offload::given_up`): every device runs the network on its own GPU from then on.
            if offload::Choice::current() != offload::Choice::Off && !offload::given_up() {
                let (count, placements) = view.header().placements();
                if placements[..count.max(1)].iter().all(|&p| p == neural_forge_protocol::enums::placement::PRESENT) {
                    self.post_offload = true;
                    self.post_server = native_post::PostServer::start_offload(instance.clone(), device.handle(), view);
                    return;
                }
                static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::log!("[offload] NEURAL_FORGE_OFFLOAD_GPU is set, but the model moves to another GPU only at the Placement setting `present` (every pass): it runs on the game's GPU");
                }
            }

            // The network's queue is the application's, shared (`crate::shared_queue`): no after-the-upscaler server.
            // Its frames are waited for at the present, and on that queue they run behind the application's own work,
            // which can wait on what the application does only after that present: Red Dead Redemption 2 froze at its
            // loading screen. The holds before the upscaler record into DLSS's buffer and never wait there.
            if loader.setup().shared.is_some() {
                static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::log!("[native] no model on the finished frame on this device: its queue is shared with the application, and a frame waited for there could wait on the application itself; the model runs before the upscaler only");
                }
                return;
            }
            self.post_server = native_post::PostServer::start(device.clone(), instance.clone(), physical_device, loader.setup(), import, loader, view);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (device, instance, physical_device, import, shm);
    }

    /// Stops the after-the-upscaler server and drops the network (the device is being destroyed).
    pub(crate) fn stop_native(&mut self) {
        #[cfg(target_arch = "x86_64")]
        drop(self.post_server.take());
        #[cfg(target_arch = "x86_64")]
        if let Some(loader) = &self.native {
            // First: no other device's loader can reach it once this returns.
            loader.retire();
            loader.stop_worker();
        }
    }

    /// Closes the network (the device's teardown, once every thread using it was joined and the device
    /// went idle). What still holds the loader (the hold's resources) is dropped with it.
    pub(crate) fn close_native(&mut self) {
        #[cfg(target_arch = "x86_64")]
        for res in [self.res.as_mut(), self.res_inline.as_mut(), self.res_after.as_mut()].into_iter().flatten() {
            res.pre_loader = None;
        }
        drop(self.native.take());
    }

    /// `NEURAL_FORGE_CAPTURE_AT=<resume>:<ms>[:<frames>][,...]`: how many consecutive presented frames to
    /// capture as a series now (`crate::series`; 1 without `<frames>`), once per entry, `<ms>` after the
    /// `<resume>`-th time DLSS frames resumed after a gap of [`CAPTURE_GAP`] or more (1 is the first
    /// frames). A scripted benchmark restarts its camera with every pass, so the same values capture the
    /// same moment in every run, whatever the loading took (the A/B captures of docs/NATIVE_BACKEND.md,
    /// Phase 3; the flicker and added-detail series of `scripts/bench-iq.py`). Several entries cover a
    /// pass whose resume number moves: GTA sometimes pauses once more while it starts.
    pub(crate) fn capture_at_due(&mut self) -> Option<u32> {
        static AT: LazyLock<Vec<CaptureAt>> =
            LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_CAPTURE_AT").and_then(|v| CaptureAt::parse_list(&v)).unwrap_or_default());
        self.capture_at_due_for(&AT)
    }

    fn capture_at_due_for(&mut self, entries: &[CaptureAt]) -> Option<u32> {
        let (i, at) = entries
            .iter()
            .enumerate()
            .find(|&(i, at)| self.captured_at & (1 << i) == 0 && self.resumes == at.resume && self.resumed_at.is_some_and(|t| t.elapsed() >= at.after))?;
        self.captured_at |= 1 << i;
        crate::log!("[preupscale] NEURAL_FORGE_CAPTURE_AT: capturing {} frames {} ms after resume {}", at.frames, at.after.as_millis(), at.resume);
        Some(at.frames)
    }

    /// Publishes the native backend's status line after a native hold, when it changed.
    pub(crate) fn note_native(&mut self, shm: &ShmClient, result: &HoldResult) {
        #[cfg(target_arch = "x86_64")]
        if let Some(loader) = self.native.as_ref() {
            if let Some(ms) = result.network_gpu_ms {
                loader.note_gpu_ms(ms);
            }
            let status = native::status(result, loader);
            // Against what the channel holds too: the offload's line shares it.
            if status != self.native_status || (!status.is_empty() && shm.layer_reason() != status) {
                shm.set_layer_reason(&status);
                self.native_status = status;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (shm, result);
    }

    /// Marks the dump as done (a pending one-shot request is consumed by the caller).
    pub(crate) fn dumped(&mut self) {
        self.burst = if self.burst > 0 { self.burst - 1 } else { dump_frames() - 1 };
        self.dumped = true;
    }

    /// The header state for a present: holding when a hold happened within [`RECENT`], with the extent of the
    /// last hold (after DLSS: its output; with Ray Reconstruction there are no Super Resolution inputs, and the
    /// inputs' `extent` is none), else `extent`.
    pub(crate) fn publish_status(&self, shm: &ShmClient, extent: Option<(u32, u32)>) {
        if !self.holding() {
            if let Some((_, state, native, (w, h))) = (*lock(&LAST_HELD)).filter(|(at, ..)| at.elapsed() < RECENT) {
                shm.publish_preupscale_state(state, native, w, h);
                publish_reason(shm, "");
                return;
            }
        }
        let (w, h) = self.last_hold_extent.filter(|_| self.holding()).or(extent).unwrap_or((0, 0));
        let now = Instant::now();
        let launches = sr_launches();
        let kept_off = self.suppresses_post();
        let (state, reason) = with_last_refusal(|last| self.status_at(kept_off, launches, now, last.as_ref()));
        shm.publish_preupscale_state(state, self.last_hold_native, w, h);
        publish_reason(shm, &reason);
    }

    /// The header state and reason at `now` ([`status_of`]), with the post path kept off or not (`kept_off`,
    /// [`Self::suppresses_post`]) and Super Resolution's `launches`. Paused only while DLSS sends nothing at all (no Super
    /// Resolution launch within [`RECENT`]: a pause menu, a loading screen) with the post path kept off: nothing runs the
    /// model until DLSS's next frame, and that is not "waiting for DLSS input", which reads as the model having moved
    /// after the upscaler (#49). That includes a DLSS game before DLSS's first frame, kept off by
    /// [`pre_dlss_post_off`] with no frame held yet. While Super Resolution launches with no frame held (DLAA, inputs not identified, every
    /// frame refused) the game plays: never paused, and the post path takes over after [`HAND_BACK`].
    fn status_at(&self, kept_off: bool, launches: Option<SrLaunches>, now: Instant, refusal: Option<&Refusal>) -> (u32, String) {
        let recent = |t: Option<Instant>| t.is_some_and(|t| now.saturating_duration_since(t) < RECENT);
        let paused = kept_off && !recent(launches.map(|r| r.last));
        if !recent(self.last_hold) && crate::ngx_ring::no_wrapper() {
            return (neural_forge_protocol::enums::preupscale_state::NO_WRAPPER, crate::ngx_ring::no_wrapper_reason());
        }
        status_of(recent(self.last_hold), recent(self.last_dlss), paused, refusal)
    }
}

/// The pre-upscaler status of a device that is `holding` (a hold within [`RECENT`]) or not, with DLSS seen running
/// with its inputs identified within [`RECENT`] (`dlss`, [`Session::saw_dlss`]), and `refusal` the last reason a frame
/// was not held: the state and the reason the header gets. DLSS identified with no hold is every frame refused, which
/// is not the same as no DLSS at all ("waiting for DLSS input", #51): `IDENTIFIED`, with the reason. With neither, a
/// device whose post path is kept off while DLSS sends nothing at all (`paused`, [`Session::status_at`]: a pause menu
/// or a loading screen, for up to [`PAUSE_HAND_BACK`] after the last held frame, or a DLSS game before DLSS's first
/// frame, [`pre_dlss_post_off`], with no hold yet) is `PAUSED`: the model resumes with DLSS (#49).
fn status_of(holding: bool, dlss: bool, paused: bool, refusal: Option<&Refusal>) -> (u32, String) {
    if !holding && dlss {
        (neural_forge_protocol::enums::preupscale_state::IDENTIFIED, refusal.map(Refusal::text).unwrap_or_default())
    } else if !holding && paused {
        (neural_forge_protocol::enums::preupscale_state::PAUSED, String::new())
    } else {
        (hold_state(holding), String::new())
    }
}

/// Publishes why no hold ran ([`neural_forge_protocol::enums::preupscale_state::IDENTIFIED`]; empty otherwise) when it
/// changed: the header's string is written under its own sequence, not on every present.
fn publish_reason(shm: &ShmClient, reason: &str) {
    if !shm.preupscale_reason_is(reason) {
        shm.publish_preupscale_reason(reason);
    }
}

/// Whether the post-upscaler path stays off at `now` in model mode: a hold on the device has run
/// the model (`engaged`) and the hand-back after the last held frame (`last_hold`, [`hand_back_at`] with Super
/// Resolution's `launches`) has not come.
pub(crate) fn post_off(engaged: bool, last_hold: Option<Instant>, launches: Option<SrLaunches>, now: Instant) -> bool {
    engaged && last_hold.is_some_and(|t| now < hand_back_at(t, launches))
}

/// The first holds of a process say which images they read for DLSS's depth and motion vectors: the evaluate's own
/// (`named`), or the source of the launch buffer's copy into them (`held`, [`Tracker::copy_source`]). Bounded: the
/// first [`SAID_HOLD_INPUTS`] holds.
fn say_hold_inputs(named: &Inputs, held: &Inputs) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) >= SAID_HOLD_INPUTS {
        return;
    }
    let what = |n: (vk::Image, ImageDesc), h: (vk::Image, ImageDesc)| {
        if n.0 == h.0 {
            format!("{} (the evaluate's)", hex(n.0))
        } else {
            format!("{} ({}x{} {:?}, the source of the copy into the evaluate's {})", hex(h.0), h.1.width, h.1.height, h.1.format, hex(n.0))
        }
    };
    crate::log!(
        "[preupscale] hold reads depth {}, motion vectors {}, exposure {}",
        what(named.depth, held.depth),
        what(named.mvec, held.mvec),
        match (named.exposure_input, held.exposure_input) {
            (_, Some(e)) => hex(e.0),
            (Some(n), None) => format!("measured from the frame (the buffer copies into the evaluate's {} before the launch)", hex(n.0)),
            (None, None) => "none".to_string(),
        }
    );
}

const SAID_HOLD_INPUTS: u64 = 12;

/// Whether the post path stays off at `now` in a DLSS game that has not held a frame yet: `sign` is when the process
/// was first seen to be one ([`dlss_game`]), `None` once a frame was held or with no sign. Its loading screens and
/// menus before DLSS's first frame go untouched: the post path built the network at the output size on them and ran it
/// there (GTA V Enhanced's three hangs at 4K, its watchdog, all with that build before NGX's devices; Cyberpunk 2077's
/// spill with Ray Reconstruction, #59). It takes over by the hand-back rules counted from the sign ([`hand_back_at`]):
/// Super Resolution launching for [`HAND_BACK`] with nothing held, or no launch at all for [`PAUSE_HAND_BACK`] (a game
/// that ships DLSS and runs another upscaler).
pub(crate) fn pre_dlss_post_off(sign: Option<Instant>, launches: Option<SrLaunches>, now: Instant) -> bool {
    sign.is_some_and(|s| now < hand_back_at(s, launches))
}

/// When the process was first seen to be a DLSS game, while no device of it has held a frame ([`pre_dlss_post_off`]).
fn before_dlss() -> Option<Instant> {
    if lock(&LAST_HELD).is_some() {
        return None;
    }
    dlss_game::since()
}

/// The last held frame of each device that holds DLSS's input ([`post_off`] from it), by session ([`PostOffEntry`]),
/// read by every device of the process; a device that does not has no entry. Per device: one device's state never
/// overwrites another's.
static POST_OFF_UNTIL: Mutex<BTreeMap<u64, Instant>> = Mutex::new(BTreeMap::new());

/// [`POST_OFF_UNTIL`]'s value for a device in model mode (`model`): its last held frame once it has held.
fn post_off_since(model: bool, engaged: bool, last_hold: Option<Instant>) -> Option<Instant> {
    last_hold.filter(|_| model && engaged)
}

/// Keeps the post path off for the hand-back from now ([`hand_back_at`]), for the native backend's hold inside DLSS's
/// buffer. That hold needs the device's state, which the post path holds while its frame waits on
/// the GPU, and the GPU waits at the hold: with the post path on, the hold never gets the state, so
/// it never engages the device the way [`post_off_since`] wants. The network is the same one either
/// path would use, so nothing is lost by not waiting for that.
///
/// Its own slot: [`POST_OFF_UNTIL`] is rewritten from the session's state at every hold and present,
/// and the session never learns of these holds while the post path keeps its state (Black Myth: Wukong,
/// 2026-10-07: the post path started, then the GPU faulted when the holds took over).
#[cfg(target_arch = "x86_64")]
pub(crate) fn keep_post_off() {
    *lock(&NATIVE_POST_OFF_UNTIL) = Some(Instant::now());
}

/// When the native backend's hold inside DLSS's buffer last kept the post path off ([`keep_post_off`]).
static NATIVE_POST_OFF_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

/// Whether some device of the process keeps the post path off at `now` (its [`Session`]
/// suppresses it, [`Session::suppresses_post`]). On the device that holds this is the same as its
/// own [`Session::suppresses_post`]; it matters when DLSS runs on another device than the swapchain
/// (only ever set in model mode).
pub(crate) fn post_off_by_any_device(now: Instant) -> bool {
    let launch = sr_launches();
    let off = |since: Instant| now < hand_back_at(since, launch);
    // A DLSS game without NGX's records (not on the Neural Forge Proton) keeps the model off too while DLSS evaluates,
    // unless the finished frame is chosen: there it would see the HUD and cost the most (the status and the doctor say
    // why). A DLSS game that has not held yet keeps it off by the timed rules from the sign ([`pre_dlss_post_off`],
    // through each device's [`Session::suppresses_post`]), never for good.
    (lock(&POST_OFF_UNTIL).values().any(|&since| off(since)) || lock(&NATIVE_POST_OFF_UNTIL).is_some_and(off) || crate::ngx_ring::no_wrapper())
        && !post_serves(now)
}

/// The process's latest hold, from whichever device made it ([`Session::note`]): when, its header state, whether
/// the native backend ran it, and its extent. A device that does not hold publishes it while it is recent
/// ([`Session::publish_status`]): Metro Last Light Redux presents on another device than DLSS runs on, and its
/// status read "waiting for DLSS input" while the model ran (2026-10-23).
type Held = (Instant, u32, bool, (u32, u32));
static LAST_HELD: Mutex<Option<Held>> = Mutex::new(None);

/// The current run of DLSS Ray Reconstruction's launch-bearing submits a device of the process saw ([`Tracker::rr_running`]):
/// when it began (submits less than [`RAY_RECONSTRUCTION_HAND_BACK`] apart) and the last one.
static RAY_RECONSTRUCTION_RUN: Mutex<Option<SrLaunches>> = Mutex::new(None);

/// When a hold after DLSS's output last got as far as asking for the network ([`note_after_dlss_tried`]).
static AFTER_DLSS_TRIED_AT: Mutex<Option<Instant>> = Mutex::new(None);

/// When a hold was last recorded right after DLSS's output kernel ([`placement`]).
static AFTER_DLSS_AT: Mutex<Option<Instant>> = Mutex::new(None);

/// A hold was recorded after DLSS's output kernel: passes run there, so the after-the-upscaler path does not
/// take them ([`post_serves`]).
pub(crate) fn note_after_dlss() {
    *lock(&AFTER_DLSS_AT) = Some(Instant::now());
}

/// A hold after DLSS's output got its resources and asked for the network: the hold point is there, and a frame it
/// missed missed for the network's sake (the after-the-upscaler path had it, the VRAM guard paused it, a build). The
/// post path does not take those frames ([`post_serves`]): it held the network the holds then waited for (Cyberpunk
/// 2077 with Ray Reconstruction, its loading screens, #59), and with the network paused it ran its capture on every
/// frame for nothing.
pub(crate) fn note_after_dlss_tried() {
    *lock(&AFTER_DLSS_TRIED_AT) = Some(Instant::now());
}

/// Whether holds after DLSS's output were recorded or tried within [`RAY_RECONSTRUCTION_HAND_BACK`] of `now`.
fn after_dlss_lately(now: Instant) -> bool {
    let recent = |at: Option<Instant>| at.is_some_and(|at| now.saturating_duration_since(at) < RAY_RECONSTRUCTION_HAND_BACK);
    recent(*lock(&AFTER_DLSS_AT)) || recent(*lock(&AFTER_DLSS_TRIED_AT))
}

/// The header's `preupscale_state` for a device that is `holding` or not: where its holds are, from the
/// passes' placements and whether holds after DLSS are being recorded (within
/// [`RAY_RECONSTRUCTION_HAND_BACK`]).
fn hold_state(holding: bool) -> u32 {
    if !holding {
        return neural_forge_protocol::enums::preupscale_state::WAITING;
    }
    let now = Instant::now();
    let after = lock(&AFTER_DLSS_AT).is_some_and(|at| now.saturating_duration_since(at) < RAY_RECONSTRUCTION_HAND_BACK);
    let (count, settings) = placement::settings();
    let plan = placement::Plan::of(count, &settings, placement::Frame { rr: ray_reconstruction_running(now), output: after, hook: true });
    hold_state_of(plan.has(placement::Stage::Before), after && plan.has(placement::Stage::Upscaled))
}

/// [`hold_state`] for a device holding with passes `before` the upscaler and/or `after` DLSS.
fn hold_state_of(before: bool, after: bool) -> u32 {
    use neural_forge_protocol::enums::preupscale_state::{HOLDING, HOLDING_AFTER, HOLDING_BOTH};
    match (before, after) {
        (true, true) => HOLDING_BOTH,
        (false, true) => HOLDING_AFTER,
        _ => HOLDING,
    }
}

/// Whether the after-the-upscaler path serves the frames while a device holds them ([`placement`]): every
/// pass of the plan runs at present. The plan here takes the hold after DLSS as available only while such
/// holds are being recorded (within [`RAY_RECONSTRUCTION_HAND_BACK`]), so a game whose hold after DLSS
/// cannot be recorded keeps the post path, as Ray Reconstruction did before the placements.
///
/// Ray Reconstruction counts only once it has run for [`RAY_RECONSTRUCTION_SETTLE`]: its first frames come before the
/// first hold after DLSS can be recorded (DLSS's output is learned from them), and the post path that took them built
/// the network at the output size and held it while the holds waited (#59).
pub(crate) fn post_serves(now: Instant) -> bool {
    post_serves_for(placement::settings(), ray_reconstruction_settled(*lock(&RAY_RECONSTRUCTION_RUN), now), after_dlss_lately(now))
}

/// How long Ray Reconstruction runs before the post path may take its frames ([`post_serves`]).
const RAY_RECONSTRUCTION_SETTLE: Duration = Duration::from_secs(2);

/// Whether the Ray Reconstruction `run` is running at `now` and began [`RAY_RECONSTRUCTION_SETTLE`] or more before it.
fn ray_reconstruction_settled(run: Option<SrLaunches>, now: Instant) -> bool {
    run.is_some_and(|r| {
        now.saturating_duration_since(r.last) < RAY_RECONSTRUCTION_HAND_BACK && now.saturating_duration_since(r.since) >= RAY_RECONSTRUCTION_SETTLE
    })
}

/// [`post_serves`] for the passes' `settings` (count, placements), Ray Reconstruction running (`rr`) and holds
/// after DLSS recorded lately (`after`).
fn post_serves_for((count, settings): (usize, [u32; neural_forge_protocol::MAX_PASSES]), rr: bool, after: bool) -> bool {
    // Not logged here: the recording side ([`Tracker::frame`]) logs the plan the holds follow.
    placement::Plan::of(count, &settings, placement::Frame { rr, output: after, hook: true }).all_present()
}

/// How long after Ray Reconstruction's last launch-bearing submit the post path keeps running in its
/// place. Its submits come several times per real frame; a second covers a hitch.
const RAY_RECONSTRUCTION_HAND_BACK: Duration = Duration::from_secs(1);

fn note_ray_reconstruction() {
    let now = Instant::now();
    let mut run = lock(&RAY_RECONSTRUCTION_RUN);
    *run = Some(match *run {
        Some(r) if now.saturating_duration_since(r.last) < RAY_RECONSTRUCTION_HAND_BACK => SrLaunches { last: now, ..r },
        _ => SrLaunches { since: now, last: now },
    });
}

/// DLSS Ray Reconstruction runs on a device of the process: nothing is held before it, so the post path
/// runs at once (not [`HAND_BACK`] after the last hold, which left a game that switched Ray
/// Reconstruction on in its settings without the model for 30 s).
pub(crate) fn ray_reconstruction_running(now: Instant) -> bool {
    lock(&RAY_RECONSTRUCTION_RUN).is_some_and(|r| now.saturating_duration_since(r.last) < RAY_RECONSTRUCTION_HAND_BACK)
}

/// Whether a launch-bearing submit is held in `mode`, given the live switches: `None` forwards it
/// untouched; `Some(dump)` holds it, `dump` saying whether this hold writes a dump. In model mode the
/// live toggle (F11, the GUI, `shmctl set enabled 0`) or `apply_model` off mean no hold at all (the
/// post path then sees the toggle too and presents the game's frame as it is; the status says so, [`MODEL_OFF`]).
/// `layouts_known`: the depth and motion-vector layouts are known (a dump waits for them, [`Session::dump_ready`]).
pub(crate) fn gate(mode: Mode, session: &mut Session, shm: &mut ShmClient, layouts_known: bool) -> Option<bool> {
    match mode {
        Mode::Off => None,
        Mode::Dump => (session.dump_due(shm) && session.dump_ready(layouts_known)).then_some(true),
        Mode::Model if !shm.model_enabled() => {
            note_not_held(MODEL_OFF);
            None
        }
        Mode::Model => Some(false),
        Mode::Identity | Mode::Roundtrip => Some(false),
    }
}

/// Writes a dump off the submit thread and logs where it went. `capture`: the one-shot capture
/// request this dump serves (`NEURAL_FORGE_PREUPSCALE=dump`), whose outcome names the dump's folder.
pub(crate) fn write_dump_async(frame: DumpFrame, capture: Option<(neural_forge_protocol::capture::Request, Option<crate::shm::Channel>)>) {
    use neural_forge_protocol::capture::{reason, state, Outcome};
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let dir = crate::dump::captures_dir().join(format!("preupscale-{stamp}"));
    let report = move |capture: &Option<(neural_forge_protocol::capture::Request, Option<crate::shm::Channel>)>, written: bool, dir: &std::path::Path| {
        if let Some((request, channel)) = capture {
            let outcome = Outcome {
                id: request.id,
                state: if written { state::WRITTEN } else { state::FAILED },
                reason: if written { reason::DLSS_INPUT_DUMP } else { reason::WRITE_FAILED },
                done: u32::from(written),
                total: 1,
                composed: 0,
                paths: dir.display().to_string(),
            };
            crate::series::publish(channel.as_ref(), &outcome);
        }
    };
    let on_failure = (capture.clone(), dir.clone());
    let spawned = std::thread::Builder::new().name("nf-preupscale-dump".into()).spawn(move || {
        let written = match frame.write(&dir) {
            Ok(files) => {
                crate::log!(
                    "[preupscale] dump of the DLSS input ({}x{}, frame {}) written to {}: {}",
                    frame.width,
                    frame.height,
                    frame.frame,
                    dir.display(),
                    files.join(", ")
                );
                true
            }
            Err(e) => {
                crate::log!("[preupscale] dump to {} failed: {e}", dir.display());
                false
            }
        };
        crate::logging::flush();
        report(&capture, written, &dir);
    });
    if let Err(e) = spawned {
        crate::log!("[preupscale] could not start the dump writer thread: {e}");
        report(&on_failure.0, false, &on_failure.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sem(raw: u64) -> vk::Semaphore {
        vk::Semaphore::from_raw(raw)
    }
    fn cb(raw: u64) -> vk::CommandBuffer {
        vk::CommandBuffer::from_raw(raw)
    }

    /// The summary line carries the capture fence wait's worst case, for the window and for the
    /// session: the measurement the frame-path bound ([`FRAME_CAPTURE_WAIT`]) is to be chosen from.
    #[test]
    fn the_phase_line_reports_the_worst_capture_wait_of_the_window_and_the_session() {
        let mut s = Stats::default();
        for ms in [2, 40, 3] {
            s.book(&HoldTiming { capture_wait: Some(Duration::from_millis(ms)), ..Default::default() });
        }
        assert!(s.phases().ends_with("capture_wait max=40.00 (session max 40.00, bound 5000 ms)"), "{}", s.phases());
        assert!(s.phases().contains(" capture_wait=3.00 "), "the median is still there: {}", s.phases());
        s.clear_phases();
        s.book(&HoldTiming { capture_wait: Some(Duration::from_millis(5)), ..Default::default() });
        assert!(s.phases().ends_with("capture_wait max=5.00 (session max 40.00, bound 5000 ms)"), "{}", s.phases());
        // The classes are the layer-wide bound until the target machine's numbers say otherwise.
        assert_eq!((FRAME_CAPTURE_WAIT, CLEANUP_WAIT), (crate::FENCE_WAIT_TIMEOUT, crate::FENCE_WAIT_TIMEOUT));
    }

    #[test]
    fn capture_at_takes_an_optional_series_length() {
        let at = |resume, ms, frames| Some(CaptureAt { resume, after: Duration::from_millis(ms), frames });
        assert_eq!(CaptureAt::parse("5:70000"), at(5, 70000, 1), "without a length: the one frame as before");
        assert_eq!(CaptureAt::parse(" 5 : 70000 : 6 "), at(5, 70000, 6));
        for bad in ["", "5", "5:", "x:1", "5:70000:0", "5:70000:x", "5:70000:6:1", "-1:10"] {
            assert_eq!(CaptureAt::parse(bad), None, "{bad:?}");
        }
        let list = CaptureAt::parse_list("5:70100:100,6:70100:100").unwrap();
        assert_eq!(list.iter().map(|a| a.resume).collect::<Vec<_>>(), [5, 6]);
        assert_eq!(CaptureAt::parse_list("5:1,x"), None, "one bad entry refuses the list");
        assert_eq!(CaptureAt::parse_list(""), None);
    }

    /// Each entry is due once, at its resume and not before its delay, with the series length asked for.
    #[test]
    fn capture_at_is_due_once_per_entry_with_its_series_length() {
        let mut session = Session::default();
        let entries = CaptureAt::parse_list("2:0:6,3:0:4,2:60000:9").unwrap();
        assert_eq!(session.capture_at_due_for(&[]), None, "unset: never");
        session.saw_dlss();
        assert_eq!(session.capture_at_due_for(&entries), None, "resume 1 is not one asked for");
        session.last_dlss = Some(Instant::now() - CAPTURE_GAP);
        session.saw_dlss();
        assert_eq!(session.capture_at_due_for(&entries), Some(6));
        assert_eq!(session.capture_at_due_for(&entries), None, "once; the third entry is too early");
        session.last_dlss = Some(Instant::now() - CAPTURE_GAP);
        session.saw_dlss();
        assert_eq!(session.capture_at_due_for(&entries), Some(4), "the next resume's entry");
        assert_eq!(session.capture_at_due_for(&entries), None);
    }

    #[test]
    fn the_summary_rate_is_holds_over_the_window() {
        assert_eq!(per_second(300, Duration::from_secs(5)), 60.0);
        assert_eq!(per_second(300, Duration::ZERO), 0.0);
    }

    #[test]
    fn modes_parse_and_unknown_values_are_refused() {
        assert_eq!(Mode::parse(None), Ok(Mode::Model), "unset is the default: the model before the upscaler");
        assert_eq!(Mode::parse(Some("")), Ok(Mode::Model));
        assert_eq!(Mode::parse(Some("off")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some(" 0 ")), Ok(Mode::Off));
        assert_eq!(Mode::parse(Some("dump")), Ok(Mode::Dump));
        assert_eq!(Mode::parse(Some("identity")), Ok(Mode::Identity));
        assert_eq!(Mode::parse(Some(" model ")), Ok(Mode::Model));
        assert_eq!(Mode::parse(Some("roundtrip")), Ok(Mode::Roundtrip));
        assert!(Mode::parse(Some("on")).is_err());
        assert!(Mode::Model.hdr() && Mode::Roundtrip.hdr() && !Mode::Identity.hdr() && !Mode::Dump.hdr(), "identity stays the raw copy-through");
        assert_eq!(commands(false).count(), 0);
        assert!(commands(true).any(|c| *c == VulkanCommand::CmdCuLaunchKernelNvx));
        assert!(diagnostic(Mode::Dump) && diagnostic(Mode::Identity) && diagnostic(Mode::Roundtrip) && !diagnostic(Mode::Model) && !diagnostic(Mode::Off));
        assert!(!commands(true).any(|c| DIAGNOSTIC_COMMANDS.contains(c)), "outside the diagnostic modes the semaphores are not hooked");
        // The test process does not set NEURAL_FORGE_ENABLE: off, and nothing extra is hooked.
        assert_eq!(mode(), Mode::Off);
    }

    #[test]
    fn the_default_resolves_to_model_only_with_the_layer_switched_on_and_off_is_the_way_back() {
        assert_eq!(DEFAULT, Mode::Model);
        // Unset or empty, layer on: the model before the upscaler.
        assert_eq!(resolve(None, true), Ok(Mode::Model));
        assert_eq!(resolve(Some("  "), true), Ok(Mode::Model));
        // The A/B and rollback switch.
        assert_eq!(resolve(Some("off"), true), Ok(Mode::Off));
        // The diagnostics stay selectable.
        for (value, mode) in [("dump", Mode::Dump), ("identity", Mode::Identity), ("roundtrip", Mode::Roundtrip), ("model", Mode::Model)] {
            assert_eq!(resolve(Some(value), true), Ok(mode));
        }
        // The layer switched off (or disabled): nothing, whatever the variable says.
        for value in [None, Some("model"), Some("dump"), Some("off")] {
            assert_eq!(resolve(value, false), Ok(Mode::Off), "{value:?}");
        }
        // A typo is refused (the caller logs it and stays off), not taken as the default.
        assert_eq!(resolve(Some("modle"), true), Err("modle".to_string()));
    }

    fn batch1(waits: &[(u64, Option<u64>)], cbs: &[u64], signals: &[(u64, Option<u64>)]) -> Batch1 {
        Batch1 {
            waits: waits.iter().map(|&(s, value)| Wait1 { semaphore: sem(s), stage: vk::PipelineStageFlags::COMPUTE_SHADER, value }).collect(),
            cbs: cbs.iter().map(|&c| cb(c)).collect(),
            signals: signals.iter().map(|&(s, value)| Signal1 { semaphore: sem(s), value }).collect(),
        }
    }

    #[test]
    fn a_launch_buffer_first_in_its_batch_moves_the_waits_to_the_capture() {
        let batches = vec![batch1(&[(1, None)], &[10], &[(2, None)]), batch1(&[(3, Some(7)), (4, None)], &[20, 21], &[(5, Some(8))]), batch1(&[], &[30], &[])];
        let p = plan(batches.clone(), 1, 0);
        assert_eq!(p.head, vec![batches[0].clone()], "earlier batches go first, unchanged");
        assert_eq!(p.capture_waits, batches[1].waits, "the launch batch's waits, values and all");
        assert_eq!(p.tail.len(), 2);
        assert!(p.tail[0].waits.is_empty());
        assert_eq!(p.tail[0].cbs, vec![cb(20), cb(21)]);
        assert_eq!(p.tail[0].signals, batches[1].signals);
        assert_eq!(p.tail[1], batches[2], "later batches follow, unchanged");
        // The layer's own capture batch widens the moved waits to ALL_COMMANDS, keeping values.
        let own = Batch1::own(p.capture_waits.clone(), cb(99));
        assert!(own.waits.iter().all(|w| w.stage == vk::PipelineStageFlags::ALL_COMMANDS));
        assert_eq!(own.waits.iter().map(|w| (w.semaphore, w.value)).collect::<Vec<_>>(), vec![(sem(3), Some(7)), (sem(4), None)]);
        assert_eq!(own.cbs, vec![cb(99)]);
        // Not submitted: the waits go back where they were.
        let mut p = p;
        p.restore_waits();
        assert_eq!(p.tail[0], batches[1]);
    }

    #[test]
    fn a_launch_buffer_inside_its_batch_splits_it() {
        let batches = vec![batch1(&[(1, Some(3))], &[10, 11, 12, 13], &[(2, Some(4)), (6, None)])];
        let p = plan(batches, 0, 2);
        assert!(p.capture_waits.is_empty(), "the waits stay on the prefix, which runs before the capture");
        assert_eq!(p.head, vec![batch1(&[(1, Some(3))], &[10, 11], &[])]);
        assert_eq!(p.tail, vec![batch1(&[], &[12, 13], &[(2, Some(4)), (6, None)])]);
        let mut p = p;
        p.restore_waits();
        assert!(p.tail[0].waits.is_empty(), "nothing was moved, nothing comes back");
    }

    #[test]
    fn submit1_parts_rebuild_their_timeline_values() {
        let batches = vec![batch1(&[(1, Some(3)), (2, Some(9))], &[10, 11], &[(5, Some(4))]), batch1(&[(7, None)], &[12], &[])];
        let p = plan(batches, 0, 1);
        let head = build1(&p.head);
        assert_eq!(head.infos.len(), 1);
        let h = &head.infos[0];
        assert_eq!((h.wait_semaphore_count, h.command_buffer_count, h.signal_semaphore_count), (2, 1, 0));
        let t = unsafe { &*h.p_next.cast::<vk::TimelineSemaphoreSubmitInfo>() };
        assert_eq!(t.wait_semaphore_value_count, 2);
        assert_eq!(unsafe { std::slice::from_raw_parts(t.p_wait_semaphore_values, 2) }, &[3, 9]);
        assert_eq!(t.signal_semaphore_value_count, 0);
        let tail = build1(&p.tail);
        assert_eq!(tail.infos.len(), 2);
        let t0 = unsafe { &*tail.infos[0].p_next.cast::<vk::TimelineSemaphoreSubmitInfo>() };
        assert_eq!((t0.wait_semaphore_value_count, t0.signal_semaphore_value_count), (0, 1));
        assert_eq!(unsafe { *t0.p_signal_semaphore_values }, 4);
        assert_eq!(unsafe { *tail.infos[0].p_command_buffers }, cb(11));
        assert!(tail.infos[1].p_next.is_null(), "a binary-only batch needs no timeline structure");
        // The struct survives being moved: the pointers are into heap storage.
        let moved = tail;
        assert_eq!(unsafe { *moved.infos[1].p_wait_semaphores }, sem(7));
    }

    #[test]
    fn submit1_is_parsed_with_its_timeline_values_and_other_chains_refused() {
        let waits = [sem(1), sem(2)];
        let stages = [vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER];
        let cbs = [cb(10), cb(11)];
        let signals = [sem(3)];
        let wait_values = [5u64, 6];
        let signal_values = [7u64];
        let timeline = vk::TimelineSemaphoreSubmitInfo {
            wait_semaphore_value_count: 2,
            p_wait_semaphore_values: wait_values.as_ptr(),
            signal_semaphore_value_count: 1,
            p_signal_semaphore_values: signal_values.as_ptr(),
            ..Default::default()
        };
        let info = vk::SubmitInfo {
            p_next: std::ptr::from_ref(&timeline).cast(),
            wait_semaphore_count: 2,
            p_wait_semaphores: waits.as_ptr(),
            p_wait_dst_stage_mask: stages.as_ptr(),
            command_buffer_count: 2,
            p_command_buffers: cbs.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signals.as_ptr(),
            ..Default::default()
        };
        let parsed = unsafe { parse1(&[info]) }.unwrap();
        assert_eq!(parsed[0].waits[1], Wait1 { semaphore: sem(2), stage: vk::PipelineStageFlags::COMPUTE_SHADER, value: Some(6) });
        assert_eq!(parsed[0].signals, vec![Signal1 { semaphore: sem(3), value: Some(7) }]);
        assert_eq!(parsed[0].cbs, cbs.to_vec());
        // Round trip through build1: the same submit.
        let rebuilt = build1(&parsed);
        let r = &rebuilt.infos[0];
        assert_eq!(unsafe { std::slice::from_raw_parts(r.p_wait_dst_stage_mask, 2) }, &stages);
        let group = vk::DeviceGroupSubmitInfo::default();
        let refused = vk::SubmitInfo { p_next: std::ptr::from_ref(&group).cast(), ..info };
        assert!(unsafe { parse1(&[refused]) }.is_err(), "a device-group submit is forwarded untouched");
        let mismatch = vk::TimelineSemaphoreSubmitInfo { wait_semaphore_value_count: 1, ..timeline };
        assert!(unsafe { parse1(&[vk::SubmitInfo { p_next: std::ptr::from_ref(&mismatch).cast(), ..info }]) }.is_err());
        assert!(unsafe { parse1(&[vk::SubmitInfo { p_next: std::ptr::null(), ..info }]) }.unwrap()[0].waits.iter().all(|w| w.value.is_none()));
    }

    /// The submission sequence of a held call: head without a fence, the capture batch with the
    /// moved waits, the write-back, then the tail with the application's fence -- for a call with
    /// batches before and after the launch batch.
    #[test]
    fn the_fence_goes_on_the_last_submission_and_the_waits_on_the_capture() {
        let calls: std::cell::RefCell<Vec<(Vec<Batch1>, vk::Fence)>> = Default::default();
        let submit = |batches: &[Batch1], fence: vk::Fence| {
            calls.borrow_mut().push((batches.to_vec(), fence));
            vk::Result::SUCCESS
        };
        let game_fence = vk::Fence::from_raw(0xf);
        let own_fence = vk::Fence::from_raw(0xe);
        let batches = vec![batch1(&[(1, None)], &[10], &[(2, None)]), batch1(&[(3, Some(5))], &[20], &[(4, Some(6))]), batch1(&[], &[30], &[(8, None)])];
        let p = plan(batches.clone(), 1, 0);
        let mut accepted: Vec<Vec<vk::CommandBuffer>> = Vec::new();
        let result = submit_around(
            p,
            game_fence,
            &submit,
            |waits| {
                let _ = submit(&[Batch1::own(waits.to_vec(), cb(100))], own_fence);
                let _ = submit(&[Batch1::own(Vec::new(), cb(101))], own_fence);
                true
            },
            &mut |part| accepted.push(part),
        );
        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(accepted, vec![vec![cb(10)], vec![cb(20), cb(30)]], "the application's buffers of each accepted part, never the layer's own");
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (vec![batches[0].clone()], vk::Fence::null()), "head: the earlier batch, no fence");
        assert_eq!(
            calls[1].0[0].waits,
            vec![Wait1 { semaphore: sem(3), stage: vk::PipelineStageFlags::ALL_COMMANDS, value: Some(5) }],
            "the capture waits for what the launch batch waited for"
        );
        assert_eq!(calls[2].0[0].cbs, vec![cb(101)]);
        assert!(calls[2].0[0].waits.is_empty());
        assert_eq!(calls[3].1, game_fence, "the application's fence is on the last call");
        assert_eq!(calls[3].0.len(), 2);
        assert!(calls[3].0[0].waits.is_empty(), "consumed by the capture");
        assert_eq!(calls[3].0[0].signals, batches[1].signals);
        assert_eq!(calls[3].0[1], batches[2]);

        // A hold that submitted nothing: the waits go back on the launch batch, and a launch batch
        // that is first in the call has no head submission at all.
        let calls: std::cell::RefCell<Vec<(Vec<Batch1>, vk::Fence)>> = Default::default();
        let submit = |batches: &[Batch1], fence: vk::Fence| {
            calls.borrow_mut().push((batches.to_vec(), fence));
            vk::Result::SUCCESS
        };
        assert_eq!(submit_around(plan(batches[1..].to_vec(), 0, 0), game_fence, &submit, |_| false, &mut |_| {}), vk::Result::SUCCESS);
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], (batches[1..].to_vec(), game_fence), "forwarded as it was");

        // A failing head is returned at once.
        let failing = |_: &[Batch1], _: vk::Fence| vk::Result::ERROR_DEVICE_LOST;
        let (mut held, mut reported) = (false, false);
        assert_eq!(
            submit_around(
                plan(batches.clone(), 1, 0),
                game_fence,
                &failing,
                |_| {
                    held = true;
                    true
                },
                &mut |_| reported = true
            ),
            vk::Result::ERROR_DEVICE_LOST
        );
        assert!(!held);
        assert!(!reported, "a refused head reports nothing as accepted");
    }

    /// A tail refused after the head and the hold went to the queue is never reported as an
    /// out-of-memory error, which tells the application that nothing of its call ran: it is
    /// submitted once more, and reported as `VK_ERROR_DEVICE_LOST` (what the specification asks of
    /// a submit that cannot be undone) only when that fails too. `VK_ERROR_DEVICE_LOST` itself is
    /// not retried.
    #[test]
    fn a_tail_refused_after_the_head_ran_is_retried_once_then_reported_as_device_lost() {
        let batches = vec![batch1(&[(1, None)], &[10], &[(2, None)]), batch1(&[(3, Some(5))], &[20], &[(4, Some(6))])];
        let game_fence = vk::Fence::from_raw(0xf);
        // The tail's submissions fail with `tail_results` in turn; the head always succeeds.
        let run = |tail_results: &[vk::Result]| {
            let tail_calls = std::cell::Cell::new(0usize);
            let submit = |b: &[Batch1], _: vk::Fence| {
                if b.iter().any(|b| b.cbs.contains(&cb(20))) {
                    let k = tail_calls.get();
                    tail_calls.set(k + 1);
                    tail_results.get(k).copied().unwrap_or(vk::Result::SUCCESS)
                } else {
                    vk::Result::SUCCESS
                }
            };
            let mut accepted: Vec<Vec<vk::CommandBuffer>> = Vec::new();
            let result = submit_around(plan(batches.clone(), 1, 0), game_fence, &submit, |_| true, &mut |part| accepted.push(part));
            (result, tail_calls.get(), accepted)
        };
        let oom = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        assert_eq!(run(&[oom]), (vk::Result::SUCCESS, 2, vec![vec![cb(10)], vec![cb(20)]]), "a transient failure: submitted again and accepted");
        assert_eq!(
            run(&[oom, vk::Result::ERROR_OUT_OF_HOST_MEMORY]),
            (vk::Result::ERROR_DEVICE_LOST, 2, vec![vec![cb(10)]]),
            "refused twice: the head ran, so the call is lost, not out of memory"
        );
        assert_eq!(run(&[vk::Result::ERROR_DEVICE_LOST]), (vk::Result::ERROR_DEVICE_LOST, 1, vec![vec![cb(10)]]), "a lost device is not retried");
    }

    fn sem2(raw: u64, value: u64) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo { semaphore: sem(raw), value, stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER, ..Default::default() }
    }
    fn cb2(raw: u64) -> vk::CommandBufferSubmitInfo<'static> {
        vk::CommandBufferSubmitInfo { command_buffer: cb(raw), ..Default::default() }
    }

    #[test]
    fn submit2_split_keeps_values_flags_and_chain_and_moves_waits() {
        let latency = vk::BaseInStructure { s_type: LATENCY_SUBMISSION_PRESENT_ID_NV, p_next: std::ptr::null(), _marker: std::marker::PhantomData };
        let waits = [sem2(1, 41), sem2(2, 42)];
        let cbs = [cb2(10), cb2(11), cb2(12)];
        let signals = [sem2(3, 43)];
        let info = vk::SubmitInfo2 {
            p_next: std::ptr::from_ref(&latency).cast(),
            flags: vk::SubmitFlags::empty(),
            wait_semaphore_info_count: 2,
            p_wait_semaphore_infos: waits.as_ptr(),
            command_buffer_info_count: 3,
            p_command_buffer_infos: cbs.as_ptr(),
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: signals.as_ptr(),
            ..Default::default()
        };
        let other = vk::SubmitInfo2 {
            p_next: std::ptr::null(),
            flags: vk::SubmitFlags::empty(),
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cbs[0],
            wait_semaphore_info_count: 0,
            signal_semaphore_info_count: 0,
            ..info
        };
        let parsed = unsafe { parse2(&[other, info, other]) }.unwrap();
        // Launch buffer is the second of the second batch: split.
        let p = plan(parsed.clone(), 1, 1);
        assert_eq!(p.head.len(), 2);
        assert!(p.capture_waits.is_empty());
        let prefix = &p.head[1];
        assert_eq!(prefix.waits.iter().map(|w| (w.semaphore, w.value)).collect::<Vec<_>>(), vec![(sem(1), 41), (sem(2), 42)]);
        assert_eq!(prefix.command_buffers(), vec![cb(10)]);
        assert!(prefix.signals.is_empty());
        assert_eq!((prefix.flags, prefix.p_next), (vk::SubmitFlags::empty(), info.p_next), "both parts keep the flags and the repeatable chain");
        let suffix = &p.tail[0];
        assert_eq!(suffix.command_buffers(), vec![cb(11), cb(12)]);
        assert_eq!(suffix.signals[0].value, 43);
        assert_eq!(suffix.p_next, info.p_next);
        assert_eq!(p.tail.len(), 2, "the third batch follows");
        let built = build2(&p.tail);
        assert_eq!((built[0].command_buffer_info_count, built[0].signal_semaphore_info_count, built[0].wait_semaphore_info_count), (2, 1, 0));
        // Launch buffer first: the waits move, widened to ALL_COMMANDS, values kept.
        let p = plan(parsed, 1, 0);
        let own = Batch2::own(p.capture_waits.clone(), cb(99));
        assert_eq!(
            own.waits.iter().map(|w| (w.semaphore, w.value, w.stage_mask)).collect::<Vec<_>>(),
            vec![(sem(1), 41, vk::PipelineStageFlags2::ALL_COMMANDS), (sem(2), 42, vk::PipelineStageFlags2::ALL_COMMANDS)]
        );
        assert!(own.p_next.is_null() && own.flags.is_empty(), "the layer's own batch carries nothing of the game's");
        assert!(p.tail[0].waits.is_empty());
        // An unknown chain is refused.
        let keyed = vk::BaseInStructure {
            s_type: vk::StructureType::WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        assert!(unsafe { parse2(&[vk::SubmitInfo2 { p_next: std::ptr::from_ref(&keyed).cast(), ..info }]) }.is_err());
    }

    fn desc(w: u32, h: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> ImageDesc {
        ImageDesc { width: w, height: h, format, usage, plain: true }
    }

    fn gta_registered() -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        [
            (0x500, desc(640, 384, rgba, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
            (0x410, desc(2560, 1440, rgba, storage)),
            (0x300, desc(1707, 960, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT)),
            (0x200, desc(1707, 960, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)),
            (0x100, desc(1707, 960, rgba, storage | vk::ImageUsageFlags::COLOR_ATTACHMENT)),
            (0x600, desc(1, 1, vk::Format::R32G32B32A32_SFLOAT, storage)),
            (0x610, desc(1, 1, vk::Format::R16_SFLOAT, storage)),
            (0x620, desc(1, 1, vk::Format::R8G8B8A8_UNORM, storage)),
            (0x700, desc(5120, 2880, vk::Format::R16_SFLOAT, storage)),
        ]
        .into_iter()
        .map(|(raw, d)| (raw, (vk::Image::from_raw(raw), d)))
        .collect()
    }

    /// A tracker with `set` registered (view `raw + 1`, kernel handle `raw + 0x1000`).
    fn tracker_with(set: &BTreeMap<u64, (vk::Image, ImageDesc)>) -> Tracker {
        let mut t = Tracker::default();
        for (&raw, &(image, d)) in set {
            let info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D { width: d.width, height: d.height, depth: 1 },
                format: d.format,
                usage: d.usage,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            t.record_image(image, &info);
            t.record_view(vk::ImageView::from_raw(raw + 1), image);
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        t
    }

    /// NGX's record of a Super Resolution evaluate (`seq`) on `[colour, mvec, depth, output, exposure]` (raw handles,
    /// 0 for none), with a jitter of (0.25, -0.5), `MV.Scale` 1 and no reset.
    fn sr_record(seq: u64, images: [u64; 5]) -> crate::ngx_ring::Record {
        use crate::ngx_ring::{parse_record, tests::record_bytes, FEATURE_SR};
        parse_record(&record_bytes(seq, FEATURE_SR, 1, images, [0.25, -0.5], [1.0, 1.0], 1.0, 1.0, 0))
    }

    /// GTA V's evaluate in [`gta_registered`]: colour 0x100, motion vectors 0x300, depth 0x200, output 0x400, exposure
    /// 0x610.
    fn gta_record(seq: u64) -> crate::ngx_ring::Record {
        sr_record(seq, [0x100, 0x300, 0x200, 0x400, 0x610])
    }

    /// The newest Super Resolution evaluate identifies DLSS's inputs (its motion vectors, not another RG16F image),
    /// and the buffer whose launch names its colour input before the evaluate's end is held with them; a launch
    /// naming it after the end (DLSS Frame Generation reading the frame) is no evaluate's, and its buffer is
    /// forwarded. DLSS runs while evaluates are submitted (`hand_back_at`).
    #[test]
    fn the_newest_record_identifies_and_only_its_evaluate_s_buffer_is_held() {
        crate::ngx_ring::inject(vec![]);
        let mut t = tracker_with(&gta_registered());
        assert_eq!(t.refresh(), None, "no record: nothing identified, nothing said");
        assert!(t.inputs.is_none());
        let rec = sr_record(1, [0x100, 0x410, 0x200, 0x400, 0x610]);
        crate::ngx_ring::inject(vec![rec]);
        let line = t.refresh().expect("identified");
        assert!(line.starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT") && line.contains("motion vectors 0x410"), "{line}");
        assert!(line.contains("exposure input 0x610"), "{line}");
        assert_eq!(t.refresh(), None, "nothing changed");
        let names = param_block(&[0x1100, 0x1200, 0x1410, 0x1610]);
        t.launch(cb(1), Some(&param_block(&[0x1500])));
        t.launch(cb(1), Some(&names));
        assert_eq!(t.launch[&cb(1)].ngx, Some(rec));
        let (scan, lines) = submit(&mut t, &[vec![cb(1)]]);
        let scan = scan.expect("the evaluate's buffer is the hold point");
        let i = scan.inputs.unwrap();
        assert_eq!((i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw(), i.exposure_input.map(|e| e.0.as_raw())), (0x100, 0x200, 0x410, Some(0x610)));
        assert_eq!(scan.frame, DlssFrame { jitter: Some([0.25, -0.5]), reset: false });
        assert!(lines.iter().any(|l| l.contains("carries DLSS Super Resolution's evaluate")), "{lines:#?}");
        assert!(sr_launches().is_some(), "DLSS runs");
        // The evaluate ended: a launch naming its colour input after that is no evaluate's.
        crate::ngx_ring::inject(vec![rec, crate::ngx_ring::end_of(&rec, 2)]);
        t.launch(cb(2), Some(&names));
        assert_eq!(t.launch[&cb(2)].ngx, None);
        let (scan, _) = submit(&mut t, &[vec![cb(2)]]);
        assert!(scan.is_none());
        assert_eq!((t.evaluate_submits, t.foreign_submits), (1, 1));
        // A launch with parameters the layer cannot read is matched to nothing either.
        t.launch(cb(3), None);
        assert!(submit(&mut t, &[vec![cb(3)]]).0.is_none());
    }

    /// DLSS Ray Reconstruction by NGX's feature: the newest evaluate being Ray Reconstruction's identifies nothing
    /// and its buffer is never held before the upscaler; once Super Resolution's is the newest, its inputs are
    /// identified again.
    #[test]
    fn ray_reconstruction_is_never_identified_or_held() {
        let mut rr = gta_record(1);
        rr.feature = crate::ngx_ring::FEATURE_RR;
        crate::ngx_ring::inject(vec![rr]);
        let mut t = tracker_with(&gta_registered());
        t.launch(cb(1), Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        assert!(t.take_inline_at().is_none(), "no hold inside Ray Reconstruction's buffer");
        let (scan, lines) = submit(&mut t, &[vec![cb(1)]]);
        assert!(scan.is_none() && t.inputs.is_none() && t.frame().rr);
        assert!(lines.iter().any(|l| l.starts_with("no DLSS input: DLSS Ray Reconstruction runs")), "{lines:#?}");
        assert!(ray_reconstruction_running(Instant::now()));
        crate::ngx_ring::inject(vec![rr, crate::ngx_ring::end_of(&rr, 2), gta_record(3)]);
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        let (scan, _) = submit(&mut t, &[vec![cb(2)]]);
        assert!(scan.is_some_and(|s| s.identified_now) && !t.frame().rr);
    }

    /// A game handing DLSS another colour image of the same shape every frame (Remnant II, Expedition 33): every
    /// buffer is held with its own evaluate's colour input, and the identification stays the first one.
    #[test]
    fn alternating_colour_inputs_are_each_held_with_their_own_image() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let mut set = gta_registered();
        set.insert(0x110, (img(0x110), desc(1707, 960, rgba, storage)));
        let mut t = tracker_with(&set);
        let mut records = Vec::new();
        for frame in 0..8u64 {
            let colour = [0x100, 0x110][frame as usize % 2];
            let rec = sr_record(2 * frame + 1, [colour, 0x300, 0x200, 0x400, 0x610]);
            records.push(rec);
            crate::ngx_ring::inject(records.clone());
            t.begin(cb(1));
            t.launch(cb(1), Some(&param_block(&[colour + 0x1000, 0x1200, 0x1300])));
            records.push(crate::ngx_ring::end_of(&rec, 2 * frame + 2));
            crate::ngx_ring::inject(records.clone());
            let s = submit(&mut t, &[vec![cb(1)]]).0.expect("every evaluate's buffer is held");
            assert_eq!(s.inputs.map(|i| i.colour.0.as_raw()), Some(colour), "frame {frame}");
            assert_eq!(s.identified_now, frame == 0, "frame {frame}: identified once");
        }
        assert_eq!(t.generation, 1);
    }

    /// The hold after DLSS goes after the last of the launches that write DLSS's output, as many as the evaluates
    /// between their record and their end wrote it in (Cyberpunk 2077's Ray Reconstruction: 12): learnt from an
    /// evaluate that ended, then taken at that write. A later evaluate writing it in more parts says so and is
    /// learnt again. The motion vectors' units there are NGX's `MV.Scale` carried to the output's pixels.
    #[test]
    fn the_hold_after_dlss_goes_after_the_evaluate_s_last_write_of_its_output() {
        use crate::ngx_ring::{end_of, inject, parse_record, tests::record_bytes, FEATURE_RR};
        let rr = |seq| parse_record(&record_bytes(seq, FEATURE_RR, 1, [0x100, 0x300, 0x200, 0x400, 0], [0.0; 2], [1707.0, 960.0], 1.0, 1.0, 1));
        let mut t = tracker_with(&gta_registered());
        let run = |t: &mut Tracker, records: Vec<crate::ngx_ring::Record>, c: vk::CommandBuffer, writes: usize| -> Vec<bool> {
            inject(records);
            t.begin(c);
            t.launch(c, Some(&param_block(&[0x1100, 0x1200, 0x1300])));
            (0..writes)
                .map(|_| {
                    t.launch(c, Some(&param_block(&[0x1400])));
                    t.after_at.take().is_some()
                })
                .collect()
        };
        let (a, b) = (rr(1), rr(3));
        assert_eq!(run(&mut t, vec![a], cb(1), 3), [false; 3], "nothing learnt yet");
        // Not ended at the submit: nothing learnt from it.
        let (_, lines) = submit(&mut t, &[vec![cb(1)]]);
        assert!(t.dlss_output.is_none(), "{lines:#?}");
        inject(vec![a, end_of(&a, 2)]);
        let (_, lines) = submit(&mut t, &[vec![cb(1)]]);
        assert_eq!(t.dlss_output.map(|o| (o.image.as_raw(), o.writes, o.rr)), Some((0x400, 3, true)), "{lines:#?}");
        assert!(lines.iter().any(|l| l.starts_with("DLSS's output: image 0x400") && l.contains("3 launch(es)")), "{lines:#?}");
        // A launch naming the output after the evaluate's end is not one of its writes.
        let writes = run(&mut t, vec![a, end_of(&a, 2), b], cb(2), 3);
        assert_eq!(writes, [false, false, true], "after the third write");
        let plan = placement::current(t.frame());
        let point = t.after_point_at(cb(2), DlssOutput { writes: 3, ..t.dlss_output.unwrap() }, &plan).expect("a point on the output");
        assert_eq!(point.mv_units, Some([2560.0, 1440.0]), "MV.Scale (render pixels per unit) in the output's pixels");
        assert_eq!((point.offsets, point.desc.width, point.desc.height, point.mvec_extent), (Offsets::default(), 2560, 1440, (1707, 960)), "the whole images");
        // DLSS writing part of a larger output (`DLSS.Output.Subrect.Base`, at the feature's output size) from part of
        // its inputs: the hold works on that rectangle, the render-size motion vectors taken at their base.
        let whole = t.launch[&cb(2)].ngx.expect("matched");
        let sub = crate::ngx_ring::Record { render: (1280, 720), mvec_base: (2, 1), output_base: (16, 8), output_size: (1920, 1080), ..whole };
        let at = |t: &mut Tracker, r| {
            t.launch.get_mut(&cb(2)).expect("launched").ngx = Some(r);
            t.after_point_at(cb(2), DlssOutput { writes: 3, ..t.dlss_output.unwrap() }, &plan)
        };
        let point = at(&mut t, sub).expect("a point on the output's rectangle");
        assert_eq!((point.desc.width, point.desc.height), (1920, 1080));
        assert_eq!(point.offsets, Offsets { colour: (16, 8), mvec: (2, 1), depth: (0, 0) });
        assert_eq!(point.mvec_extent, (1280, 720), "the render size at their base");
        assert_eq!(point.mv_units, Some([1707.0 * 1920.0 / 1280.0, 1440.0]), "MV.Scale over the render size, in the rectangle's pixels");
        assert_eq!(point.hold, placement::Hold::of(&plan, placement::Stage::Upscaled, (1280, 720), (1920, 1080)));
        assert!(at(&mut t, crate::ngx_ring::Record { output_base: (700, 0), ..sub }).is_none(), "a rectangle outside the output: no hold there");
        // ... said once for that output, however often it comes.
        assert!(at(&mut t, crate::ngx_ring::Record { output_base: (700, 0), ..sub }).is_none());
        assert_eq!(t.after_crop_said, [img(0x400)], "the refusal said once");
        // High-resolution motion vectors (no MVLowRes, the feature's output size fits their image): the same rectangle
        // the hold before the upscaler reads, not the render size.
        let hi = crate::ngx_ring::Record {
            output_base: (0, 0),
            output_size: (1600, 900),
            create_flags: whole.create_flags & !crate::ngx_ring::FLAG_MV_LOW_RES,
            ..sub
        };
        let point = at(&mut t, hi).expect("a point");
        assert_eq!((point.offsets.mvec, point.mvec_extent), ((2, 1), (1600, 900)), "the output size at their base");
        let point = at(&mut t, crate::ngx_ring::Record { create_flags: hi.create_flags | crate::ngx_ring::FLAG_MV_LOW_RES, ..hi }).expect("a point");
        assert_eq!(point.mvec_extent, (1280, 720), "MVLowRes: the render size");
        // A colour input the layer does not track: the record's extent says what DLSS reads; with neither known, the
        // output whole, and the motion vectors' rectangle from the render size where that is known and not their image's
        // (none where it does not fit).
        let untracked = DlssOutput { input: img(0x9900), writes: 3, ..t.dlss_output.unwrap() };
        let at_untracked = |t: &mut Tracker, r| {
            t.launch.get_mut(&cb(2)).expect("launched").ngx = Some(r);
            t.after_point_at(cb(2), untracked, &plan)
        };
        let full = crate::ngx_ring::Record { render: (1707, 960), colour_extent: (1707, 960), mvec_base: (0, 0), ..whole };
        let point = at_untracked(&mut t, full).expect("a point");
        assert_eq!((point.offsets, point.mvec_extent), (Offsets::default(), (1707, 960)), "the record's extent: the whole images");
        let low = full.create_flags | crate::ngx_ring::FLAG_MV_LOW_RES;
        let point = at_untracked(&mut t, crate::ngx_ring::Record { render: (1600, 900), colour_extent: (0, 0), mvec_base: (2, 1), create_flags: low, ..full })
            .expect("a point");
        assert_eq!(
            (point.offsets, point.mvec_extent),
            (Offsets { mvec: (2, 1), ..Offsets::default() }, (1600, 900)),
            "no extent known, MVLowRes: the render size of the motion vectors at their base"
        );
        let point = at_untracked(&mut t, crate::ngx_ring::Record { render: (1600, 900), colour_extent: (0, 0), mvec_base: (2, 1), ..full }).expect("a point");
        assert_eq!(
            (point.offsets.mvec, point.mvec_extent),
            ((2, 1), (1705, 959)),
            "no extent known, high-resolution vectors and no output size: what their image holds from their base"
        );
        let point = at_untracked(&mut t, crate::ngx_ring::Record { render: (1706, 960), colour_extent: (0, 0), mvec_base: (2, 1), create_flags: low, ..full })
            .expect("a point");
        assert_eq!(point.mvec_extent, (0, 0), "the render size does not fit at their base: no motion vectors");
        assert!(point.mvec_input.is_some_and(|a| !a.readable), "and not read");
        let point = at_untracked(&mut t, crate::ngx_ring::Record { render: (0, 0), colour_extent: (0, 0), mvec_base: (0, 0), ..full }).expect("a point");
        assert_eq!((point.offsets, point.mvec_extent), (Offsets::default(), (1707, 960)), "nothing known: the whole images");
        t.launch.get_mut(&cb(2)).expect("launched").ngx = Some(whole);
        assert!(t.frame_of(cb(2)).reset, "the evaluate's reset");
        inject(vec![a, end_of(&a, 2), b, end_of(&b, 4)]);
        t.launch(cb(2), Some(&param_block(&[0x1400])));
        assert!(t.after_at.is_none() && t.launch[&cb(2)].output_writes == 3, "after its end");
        // Written in four parts this time: said, and learnt.
        let c = rr(5);
        run(&mut t, vec![c], cb(3), 4);
        inject(vec![c, end_of(&c, 6)]);
        let (_, lines) = submit(&mut t, &[vec![cb(3)]]);
        assert!(lines.iter().any(|l| l.starts_with("DLSS's output is written in 4 launches, not 3")), "{lines:#?}");
        assert_eq!(t.dlss_output.map(|o| o.writes), Some(4));
    }

    /// A D3D11 evaluate (DXVK records its launches after NVIDIA's call returned, so its end record says nothing of them):
    /// its writes of the output are counted while its buffer is recorded, and learnt at the buffer's submit.
    #[test]
    fn a_d3d11_evaluate_s_output_writes_are_counted_and_learnt_at_the_submit() {
        use crate::ngx_ring::{inject, API_D3D11};
        let rec = crate::ngx_ring::Record { api: API_D3D11, ..gta_record(1) };
        let mut t = tracker_with(&gta_registered());
        inject(vec![rec, crate::ngx_ring::end_of(&rec, 2)]);
        t.begin(cb(1));
        t.launch(cb(1), Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        t.launch(cb(1), Some(&param_block(&[0x1400])));
        t.launch(cb(1), Some(&param_block(&[0x1400])));
        assert_eq!(t.launch[&cb(1)].output_writes, 2, "its end record does not end it");
        let (_, lines) = submit(&mut t, &[vec![cb(1)]]);
        assert_eq!(t.dlss_output.map(|o| (o.image.as_raw(), o.writes)), Some((0x400, 2)), "{lines:#?}");
    }

    /// A game outside a Neural Forge Proton: its CUDA launches carry no record, nothing is held, and the status says
    /// the wrapper is missing once DLSS Super Resolution's kernels launch. Other CUDA launches (Frame Generation's) are
    /// no sign: such a game gets the post-upscaler path by the timed rules.
    #[test]
    fn without_records_nothing_is_held_and_the_missing_wrapper_is_said() {
        let mut t = tracker_with(&gta_registered());
        let (fg, sr) = (vk::CuFunctionNVX::from_raw(0xf1), vk::CuFunctionNVX::from_raw(0xf2));
        t.record_function(fg, "dlssg_flow_kernel");
        t.record_function(sr, "hiluma_engine_input_kernel");
        t.launch_kernel(cb(1), fg, Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        let (scan, _) = submit(&mut t, &[vec![cb(1)]]);
        assert!(scan.is_none() && t.inputs.is_none());
        assert!(!crate::ngx_ring::no_wrapper(), "Frame Generation's launches are no sign");
        t.launch_kernel(cb(2), sr, Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        let (scan, _) = submit(&mut t, &[vec![cb(2)]]);
        assert!(scan.is_none() && t.inputs.is_none());
        assert!(crate::ngx_ring::no_wrapper());
        let (state, reason) = Session::default().status_at(false, None, Instant::now(), None);
        assert_eq!(state, neural_forge_protocol::enums::preupscale_state::NO_WRAPPER);
        assert!(reason.contains("neural-forge-ngx-"), "names the ring's path: {reason}");
    }

    /// The process-wide post-path switch (for DLSS on another device than the swapchain, the case
    /// the output-like-image identification opens) agrees with the holding device's own at every
    /// moment, and is never set outside model mode.
    #[test]
    fn the_shared_post_off_deadline_matches_the_holding_devices_own() {
        let t0 = Instant::now();
        for engaged in [false, true] {
            for last in [None, Some(t0)] {
                let since = post_off_since(true, engaged, last);
                for launch in [None, Some(SrLaunches { since: t0, last: t0 + Duration::from_secs(5) })] {
                    for ms in [0, 1, 29_999, 30_000, 30_001, 60_000, 299_999, 300_001] {
                        let now = t0 + Duration::from_millis(ms);
                        let shared = since.is_some_and(|s| now < hand_back_at(s, launch));
                        assert_eq!(shared, post_off(engaged, last, launch, now), "engaged {engaged}, last {last:?}, launch {launch:?}, +{ms} ms");
                    }
                }
                assert_eq!(post_off_since(false, engaged, last), None, "only model mode turns the post path off");
            }
        }
    }

    /// Each device keeps its own deadline: one that does not hold (or holds no more) does not clear the
    /// one that does, and a device gone takes its entry with it.
    #[test]
    fn each_device_keeps_its_own_post_off_deadline() {
        let now = Instant::now();
        let (holding, other) = (PostOffEntry::default(), PostOffEntry::default());
        let ours = |e: &PostOffEntry| lock(&POST_OFF_UNTIL).get(&e.0).copied();
        holding.set(Some(now + HAND_BACK));
        other.set(None);
        assert_eq!(ours(&holding), Some(now + HAND_BACK), "the other device's update leaves it");
        assert_eq!(ours(&other), None);
        other.set(Some(now + Duration::from_secs(1)));
        holding.set(None);
        assert_eq!(ours(&other), Some(now + Duration::from_secs(1)));
        let key = other.0;
        drop(other);
        assert!(!lock(&POST_OFF_UNTIL).contains_key(&key), "removed with its session");
        drop(holding);
    }

    #[test]
    fn the_tracker_follows_registrations_launches_and_submitted_layouts() {
        let mut t = Tracker::default();
        let info = |w, h, format, usage| vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: w, height: h, depth: 1 },
            format,
            usage,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let storage = vk::ImageUsageFlags::STORAGE;
        for (raw, w, h, format) in [
            (0x100, 1707, 960, vk::Format::R16G16B16A16_SFLOAT),
            (0x200, 1707, 960, vk::Format::D32_SFLOAT_S8_UINT),
            (0x300, 1707, 960, vk::Format::R16G16_SFLOAT),
        ] {
            t.record_image(vk::Image::from_raw(raw), &info(w, h, format, storage));
            t.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        crate::ngx_ring::inject(vec![sr_record(1, [0x100, 0x300, 0x200, 0, 0])]);
        let line = t.refresh().expect("the identification is logged");
        assert!(line.starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT"), "{line}");
        assert!(t.refresh().is_none(), "nothing changed, nothing logged");
        let colour = vk::Image::from_raw(0x100);
        // Recorded in one order, submitted in the other; a launch in a secondary.
        t.barrier(cb(1), ImageSync::to(colour, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        t.barrier(cb(2), ImageSync::to(colour, vk::ImageLayout::GENERAL));
        t.barrier(cb(2), ImageSync::to(vk::Image::from_raw(0x999), vk::ImageLayout::GENERAL));
        t.launch(cb(5), Some(&param_block(&[0x1100])));
        t.execute(cb(3), &[cb(5)]);
        let scan = t.submit_ok(&[vec![cb(1)], vec![cb(2), cb(3)]]).expect("launch-bearing");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!(scan.colour_layout, Some(vk::ImageLayout::GENERAL), "the layout as of submission order just before the launch");
        assert!(!t.committed.contains_key(&vk::Image::from_raw(0x999)), "only the inputs are watched");
        assert!(scan.identified_now && !scan.colour_in_general(), "the first launch-bearing submit since identification is not held");
        assert!(t.submit_ok(&[vec![cb(7)]]).is_none());
        t.begin(cb(3));
        assert!(t.submit_ok(&[vec![cb(3)]]).is_none(), "re-recorded: no longer launch-bearing");
        // A destroyed input image: the record names an image no longer there, so nothing is identified.
        t.forget_image(vk::Image::from_raw(0x200));
        assert!(t.refresh().unwrap().starts_with("no DLSS input: NGX's record names images this device did not see created"));
        assert!(t.inputs.is_none());
    }

    /// The submit that (re)identifies the inputs is not held: barriers recorded before the images
    /// were watched were never seen (the device hooks only record them while watching), and a new
    /// identification drops what was committed, so "no layout" there is not "GENERAL". Nor is a later
    /// submit whose launch buffer, or a buffer before it, was recorded before the identification
    /// (submitted again without being recorded again, recorded ahead on another thread, or carrying
    /// such a secondary): its barriers on the inputs were not seen either. A launch buffer recorded
    /// since, with no barrier seen while watched, is held.
    #[test]
    fn the_submit_that_identifies_the_inputs_is_not_held() {
        let t = Tracking::default();
        let info = |w, h, format| vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: w, height: h, depth: 1 },
            format,
            usage: vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let register = |base: u64| {
            let mut tr = t.lock();
            for (raw, format) in
                [(base, vk::Format::R16G16B16A16_SFLOAT), (base + 0x100, vk::Format::D32_SFLOAT_S8_UINT), (base + 0x200, vk::Format::R16G16_SFLOAT)]
            {
                tr.record_image(vk::Image::from_raw(raw), &info(1707, 960, format));
                tr.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
                tr.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x10_0000));
            }
            crate::ngx_ring::inject(vec![sr_record(base, [base, base + 0x200, base + 0x100, 0, 0])]);
        };
        // A launch naming the colour input of either identification's record.
        let names = param_block(&[0x10_1000, 0x10_5000]);
        let record = |c: vk::CommandBuffer| {
            t.begin(c);
            t.launch(c, vk::CuFunctionNVX::null(), Some(&names));
        };
        register(0x1000);
        record(cb(1));
        let first = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(first.inputs.is_some() && first.identified_now && first.colour_layout.is_none());
        assert!(!first.colour_in_general(), "identified on this submit: nothing known, not held");
        // The same buffer again, not recorded again: whatever it holds on the inputs was not seen.
        let resubmitted = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(!resubmitted.identified_now && resubmitted.colour_in_general());
        assert_eq!(resubmitted.hazard, Some(Hazard::RecordedBefore), "recorded before the identification: not held");
        // The hold inside DLSS's buffer refuses that buffer too (begun before): the line does not say it takes it.
        assert_eq!(t.lock().inline_point(cb(1), &record_on(img(0x1000))).err(), Some(InlineRefusal::BegunBefore));
        let why = Hazard::RecordedBefore.why_inside();
        assert!(!why.contains("takes such buffers") && why.contains("only where DLSS's own buffer was begun since"), "{why}");
        record(cb(1));
        let second = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(!second.identified_now && second.colour_layout.is_none());
        assert!(second.colour_in_general() && second.hazard.is_none(), "recorded while watched, no barrier seen: GENERAL, held");
        // A buffer before the launch buffer in the submit, recorded before the identification: its
        // barriers are what the launch buffer starts from.
        assert_eq!(t.submit_ok(&[vec![cb(3)], vec![cb(1)]]).expect("launch-bearing").hazard, Some(Hazard::RecordedBefore));
        t.begin(cb(3));
        assert_eq!(t.submit_ok(&[vec![cb(3)], vec![cb(1)]]).expect("launch-bearing").hazard, None);
        assert_eq!(t.submit_ok(&[vec![cb(1), cb(7)]]).expect("launch-bearing").hazard, None, "one after the launch buffer does not matter");
        // A secondary recorded before the identification makes its primary so.
        t.begin(cb(4));
        t.execute(cb(4), &[cb(5)]);
        t.launch(cb(4), vk::CuFunctionNVX::null(), Some(&names));
        assert_eq!(t.submit_ok(&[vec![cb(4)]]).expect("launch-bearing").hazard, Some(Hazard::RecordedBefore));
        t.begin(cb(5));
        record(cb(4));
        t.execute(cb(4), &[cb(5)]);
        assert_eq!(t.submit_ok(&[vec![cb(4)]]).expect("launch-bearing").hazard, None);
        // DLSS's inputs re-created with the same shapes (007 First Light after a load, #49): the same identification,
        // every image of those shapes being watched throughout, so a buffer recorded before is not refused for it.
        t.lock().forget_image(vk::Image::from_raw(0x1000));
        register(0x5000);
        record(cb(1));
        let again = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert_eq!(again.inputs.map(|i| i.colour.0), Some(vk::Image::from_raw(0x5000)));
        assert!(!again.identified_now, "the same shapes: no new identification");
        let later = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert_ne!(later.hazard, Some(Hazard::RecordedBefore), "recorded while its images' shapes were watched");
        record(cb(1));
        let held = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(held.colour_in_general() && held.hazard.is_none());
        // A barrier out of GENERAL still refuses the hold.
        t.begin(cb(2));
        t.barriers(cb(2), [ImageSync::to(vk::Image::from_raw(0x5000), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)].into_iter(), None);
        assert!(!t.submit_ok(&[vec![cb(2), cb(1)]]).expect("launch-bearing").colour_in_general());
        // Nothing is left watched-recorded once nothing is identified, so nothing stays armed for it.
        t.lock().forget_image(vk::Image::from_raw(0x5000));
        t.lock().forget_image(vk::Image::from_raw(0x5100));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        assert!(t.lock().inputs.is_none() && t.lock().recorded_watched.is_empty());
    }

    /// On a device with NVX but no DLSS nothing is ever armed, so the per-command-buffer and
    /// per-submit hooks stay a relaxed load; a launch arms it until its buffer is re-recorded.
    #[test]
    fn tracking_stays_disarmed_without_launches_and_disarms_after_the_launch_buffer_is_reset() {
        let t = Tracking::default();
        *t.lock() = tracker_with(&gta_registered());
        crate::ngx_ring::inject(vec![gta_record(1)]);
        assert!(!t.armed());
        // A game without DLSS: buffers come and go, submits carry no launch.
        for k in 0..100 {
            t.begin(cb(k));
            t.execute(cb(k), &[cb(k + 1000)]);
        }
        t.free(&[cb(1), cb(2)]);
        assert!(!t.armed());
        assert!(t.submit_ok(&[vec![cb(1), cb(2)]]).is_none());
        assert_eq!(t.extent(), None);
        // DLSS records a launch: armed, and the submit carrying it is found.
        t.launch(cb(7), vk::CuFunctionNVX::null(), Some(&param_block(&[0x1100])));
        assert!(t.armed());
        assert!(t.submit_ok(&[vec![cb(6), cb(7)]]).is_some_and(|s| (s.batch, s.index) == (0, 1)));
        // A secondary's launch carries over to its primary.
        t.execute(cb(8), &[cb(7)]);
        t.free(&[cb(7)]);
        assert!(t.armed(), "the primary still carries the launch");
        t.free(&[cb(8)]);
        assert!(!t.armed(), "every launch buffer was reset or freed");
        assert!(t.submit_ok(&[vec![cb(7), cb(8)]]).is_none());
    }

    /// A pool reset (vkd3d-proton's command allocators) or destroyed ends its buffers' launches: the
    /// tracking disarms without waiting for each buffer to be begun or freed again, and forgets a
    /// destroyed pool's buffers. A reset buffer is as a begun one.
    #[test]
    fn a_reset_or_destroyed_pool_drops_its_buffers_launches() {
        let t = Tracking::default();
        let (pool, other) = (vk::CommandPool::from_raw(1), vk::CommandPool::from_raw(2));
        t.allocated(pool, &[cb(1), cb(2)]);
        t.allocated(other, &[cb(3)]);
        t.launch(cb(1), vk::CuFunctionNVX::null(), None);
        t.launch(cb(3), vk::CuFunctionNVX::null(), None);
        t.reset_pool(pool, false);
        assert!(t.armed(), "the other pool's launch buffer is still recorded");
        assert!(t.submit_ok(&[vec![cb(1)]]).is_none(), "the reset pool's buffer carries no launch");
        t.reset_pool(other, true);
        assert!(!t.armed());
        assert!(!lock(&t.pools).contains_key(&cb(3)), "a destroyed pool's buffers are forgotten");
        assert!(lock(&t.pools).contains_key(&cb(1)), "a reset pool keeps its buffers");
        t.launch(cb(2), vk::CuFunctionNVX::null(), None);
        t.reset(cb(2));
        assert!(!t.armed(), "a reset buffer carries nothing");
    }

    /// A protected `vkQueueSubmit2` is never parsed into a plan, wherever its protected batch is:
    /// the caller forwards the application's call as it was, with none of the layer's own batches.
    #[test]
    fn a_protected_submit_is_refused_and_forwarded_untouched() {
        let cbs = [cb2(10), cb2(11)];
        let plain = vk::SubmitInfo2 { command_buffer_info_count: 2, p_command_buffer_infos: cbs.as_ptr(), ..Default::default() };
        let protected = vk::SubmitInfo2 { flags: vk::SubmitFlags::PROTECTED, ..plain };
        assert!(unsafe { parse2(&[plain]) }.is_ok());
        for call in [vec![protected], vec![plain, protected], vec![protected, plain]] {
            let why = unsafe { parse2(&call) }.expect_err("no batches to split: the call is forwarded");
            assert!(why.contains("protected"), "{why}");
        }
        // `vkQueueSubmit`: a protected batch carries `VkProtectedSubmitInfo`, refused with every
        // chain but the timeline one.
        let chain = vk::ProtectedSubmitInfo { protected_submit: vk::TRUE, ..Default::default() };
        let raw = [cb(10)];
        let info = vk::SubmitInfo { p_next: std::ptr::from_ref(&chain).cast(), command_buffer_count: 1, p_command_buffers: raw.as_ptr(), ..Default::default() };
        assert!(unsafe { parse1(&[info]) }.is_err());
    }

    /// A call carrying `VkFrameBoundaryEXT` is forwarded untouched: split, the marker would be on
    /// both halves, with the layer's own submits between them.
    #[test]
    fn a_frame_boundary_submit_is_refused_and_forwarded_untouched() {
        let boundary = vk::BaseInStructure { s_type: FRAME_BOUNDARY_EXT, p_next: std::ptr::null(), _marker: std::marker::PhantomData };
        let latency =
            vk::BaseInStructure { s_type: LATENCY_SUBMISSION_PRESENT_ID_NV, p_next: std::ptr::from_ref(&boundary), _marker: std::marker::PhantomData };
        let cbs = [cb2(10), cb2(11)];
        let plain = vk::SubmitInfo2 { command_buffer_info_count: 2, p_command_buffer_infos: cbs.as_ptr(), ..Default::default() };
        for chain in [std::ptr::from_ref(&boundary), std::ptr::from_ref(&latency)] {
            let marked = vk::SubmitInfo2 { p_next: chain.cast(), ..plain };
            for call in [vec![marked], vec![plain, marked]] {
                let why = unsafe { parse2(&call) }.expect_err("forwarded untouched");
                assert!(why.contains("VkFrameBoundaryEXT"), "{why}");
            }
        }
    }

    /// A tracker with GTA's three inputs identified (colour 0x100, depth 0x200, motion vectors
    /// 0x300; registration keys `image + 0x1000`) and one launch-bearing submit behind it, so the
    /// next is not the identifying one. Buffers `cb(1)` to `cb(HELD_BEGUN)` were begun since: their
    /// whole recording is seen ([`Tracker::recorded_watched`]).
    fn held_tracker() -> (Tracker, Vec<u8>) {
        held_tracker_with(vk::ImageUsageFlags::STORAGE)
    }

    fn held_tracker_with(usage: vk::ImageUsageFlags) -> (Tracker, Vec<u8>) {
        let mut t = Tracker::default();
        for (raw, format) in [(0x100, vk::Format::R16G16B16A16_SFLOAT), (0x200, vk::Format::D32_SFLOAT_S8_UINT), (0x300, vk::Format::R16G16_SFLOAT)] {
            let info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
                format,
                usage,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            t.record_image(vk::Image::from_raw(raw), &info);
            t.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        crate::ngx_ring::inject(vec![sr_record(1, [0x100, 0x300, 0x200, 0, 0])]);
        t.refresh().expect("identified");
        let params = param_block(&[0x1100, 0x1200, 0x1300]);
        t.launch(cb(900), Some(&params));
        assert!(t.submit_ok(&[vec![cb(900)]]).is_some_and(|s| s.identified_now));
        t.free(&[cb(900)]);
        for k in 1..=HELD_BEGUN {
            t.begin(cb(k));
        }
        (t, params)
    }

    /// The buffers [`held_tracker`] begins after its identification.
    const HELD_BEGUN: u64 = 16;

    fn colour() -> vk::Image {
        vk::Image::from_raw(0x100)
    }

    /// The newest injected evaluate's record on `colour` (for a hold inside a buffer [`held_tracker`] launched in).
    fn record_on(colour: vk::Image) -> crate::ngx_ring::Record {
        crate::ngx_ring::Record { colour, ..crate::ngx_ring::newest(|_| true).unwrap_or_default() }
    }
    const GENERAL: vk::ImageLayout = vk::ImageLayout::GENERAL;

    fn general_to_general(image: vk::Image) -> ImageSync {
        ImageSync { old_layout: GENERAL, ..ImageSync::to(image, GENERAL) }
    }

    /// GTA V's pattern is still held: the last barrier on the colour input is in an earlier buffer,
    /// the launch buffer has none before its first launch, and vkd3d-proton's barriers after the
    /// launches (same-layout ones on the inputs, global ones) do not count.
    #[test]
    fn a_launch_buffer_with_no_synchronization_before_its_launch_is_held() {
        let (mut t, params) = held_tracker();
        t.barrier(cb(1), general_to_general(colour()));
        t.launch(cb(2), Some(&params));
        t.global(cb(2), Hazard::MemoryBarrier);
        t.barrier(cb(2), general_to_general(colour()));
        t.barrier(cb(2), general_to_general(vk::Image::from_raw(0x200)));
        let scan = t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.index, scan.hazard, scan.colour_owner), (1, None, None));
        assert!(scan.colour_in_general());
        // Resubmitted unchanged: the same.
        assert_eq!(t.submit_ok(&[vec![cb(1), cb(2)]]).and_then(|s| s.hazard), None);
    }

    /// DLSS running with its inputs identified while no hold runs is not "waiting for DLSS input" (#51): the status
    /// is `IDENTIFIED` with the last reason a frame was refused, from the split hold or the hold inside DLSS's
    /// buffer; holding, or no DLSS lately, publishes no reason. The header carries it to the CLI and GUI.
    #[test]
    fn dlss_identified_with_every_frame_refused_is_not_waiting_for_dlss_input() {
        use neural_forge_protocol::enums::preupscale_state::{IDENTIFIED, WAITING};
        let split = Refusal::Text("the colour input was last transitioned out of GENERAL; not holding".into());
        assert_eq!(status_of(false, false, false, Some(&split)), (WAITING, String::new()), "no DLSS lately: waiting");
        assert_eq!(status_of(false, true, false, Some(&split)), (IDENTIFIED, "the colour input was last transitioned out of GENERAL; not holding".into()));
        assert_eq!(status_of(false, true, false, None), (IDENTIFIED, String::new()), "identified, no reason noted yet");
        assert_ne!(status_of(true, true, false, Some(&split)).0, IDENTIFIED);
        assert!(status_of(true, true, false, Some(&split)).1.is_empty(), "holding: no reason");
        // DLSS stopped after holds (a pause menu, #49): paused, not waiting, and no reason; DLSS or a hold wins.
        use neural_forge_protocol::enums::preupscale_state::PAUSED;
        assert_eq!(status_of(false, false, true, Some(&split)), (PAUSED, String::new()));
        assert_eq!(status_of(false, true, true, None).0, IDENTIFIED, "DLSS runs again: not paused");
        assert_ne!(status_of(true, false, true, None).0, PAUSED, "holding: not paused");
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let (_, inside) = status_of(false, true, false, Some(&Refusal::Inside(InlineRefusal::Usage(usage))));
        assert!(inside.starts_with("no hold inside DLSS's buffer: the colour input's usage") && inside.contains("TRANSFER_DST"), "{inside}");
        // Through the header.
        let header = neural_forge_protocol::ShmHeader::default();
        let shm = ShmClient::test_over_header(&header);
        publish_reason(&shm, &inside);
        assert_eq!(header.preupscale_reason(), inside[..inside.len().min(neural_forge_protocol::REASON_BYTES - 1)]);
        assert!(shm.preupscale_reason_is(&header.preupscale_reason()));
        publish_reason(&shm, "");
        assert_eq!(header.preupscale_reason(), "");
    }

    /// A hold that runs and holds nothing gives the status its reason (#59): with the VRAM guard's pause in Cyberpunk 2077 at
    /// 4K with Ray Reconstruction the status was "identified, not holding" with an empty reason. The next held frame
    /// clears it.
    #[test]
    fn a_hold_that_missed_is_why_nothing_is_held() {
        use neural_forge_protocol::enums::preupscale_state::IDENTIFIED;
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        let shm = ShmClient::test_over_header(&header);
        let mut session = Session::default();
        session.note(
            &shm,
            &HoldResult { wrote_back: true, waits_consumed: true, native: true, evaluated: true, ..Default::default() },
            Duration::ZERO,
            (16, 8),
        );
        assert_eq!(last_refusal_text(), None);
        // The guard paused the network: the holds after DLSS still run, and miss.
        session.note(&shm, &HoldResult { native: true, miss: Some(VRAM_PAUSED), ..Default::default() }, Duration::ZERO, (16, 8));
        assert_eq!(last_refusal_text().as_deref(), Some(VRAM_PAUSED));
        // No frame held lately, DLSS running: identified, with the guard's reason.
        let now = Instant::now();
        session.last_hold = now.checked_sub(RECENT * 4);
        session.last_dlss = Some(now);
        let (state, reason) = with_last_refusal(|last| session.status_at(false, None, now, last.as_ref()));
        assert_eq!((state, reason.as_str()), (IDENTIFIED, VRAM_PAUSED));
        publish_reason(&shm, &reason);
        assert!(header.preupscale_reason().contains("too little video memory"), "{}", header.preupscale_reason());
        // A frame held again: no reason.
        session.note(&shm, &HoldResult { wrote_back: true, ..Default::default() }, Duration::ZERO, (16, 8));
        assert_eq!(last_refusal_text(), None);
    }

    /// The status's reason is why no hold runs now (#51): the model switched off (the split hold's gate) or a
    /// Placement with no pass before the upscaler (both holds before it ask [`plans_before`]) replace an older
    /// refusal, as a held frame clears it, so the Status line never names a refusal that no longer applies.
    #[test]
    fn the_status_reason_is_why_no_hold_runs_now() {
        use neural_forge_protocol::enums::placement::{BEFORE, PRESENT, UPSCALED};
        use neural_forge_protocol::enums::preupscale_state::IDENTIFIED;
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE;
        let refusal = Refusal::Inside(InlineRefusal::Usage(usage));
        note_refusal(refusal.clone());
        assert_eq!(last_refusal_text(), Some(refusal.text()));
        // The model switched off (F11, the app): the split hold's gate says so, over the older refusal.
        let header = neural_forge_protocol::ShmHeader::default();
        header.init_defaults();
        let mut shm = ShmClient::test_over_header(&header);
        let mut session = Session::default();
        header.apply_model.store(0, Ordering::Relaxed);
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), None);
        assert_eq!(last_refusal_text().as_deref(), Some(MODEL_OFF));
        assert_eq!(with_last_refusal(|last| status_of(false, true, false, last.as_ref())), (IDENTIFIED, MODEL_OFF.to_string()));
        header.apply_model.store(1, Ordering::Relaxed);
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), Some(false), "on again: held");
        // No pass before the upscaler: every pass at present, or after DLSS with its output learnt (without it, After
        // DLSS falls back to before the upscaler).
        let frame = placement::Frame { rr: false, output: true, hook: true };
        note_refusal(refusal.clone());
        assert!(!plans_before(&placement::Plan::of(1, &[PRESENT; neural_forge_protocol::MAX_PASSES], frame)));
        assert_eq!(last_refusal_text().as_deref(), Some(NO_PASS_BEFORE));
        note_refusal(refusal.clone());
        assert!(!plans_before(&placement::Plan::of(1, &[UPSCALED; neural_forge_protocol::MAX_PASSES], frame)));
        assert_eq!(last_refusal_text().as_deref(), Some(NO_PASS_BEFORE));
        // A plan with a pass before the upscaler notes nothing: the reason stays until a hold or another reason.
        note_refusal(refusal.clone());
        let mut before_then_after = [UPSCALED; neural_forge_protocol::MAX_PASSES];
        before_then_after[0] = BEFORE;
        assert!(plans_before(&placement::Plan::of(2, &before_then_after, frame)));
        assert_eq!(last_refusal_text(), Some(refusal.text()));
        // A held frame clears it: holding, the header's reason is empty.
        session.note(&shm, &HoldResult { wrote_back: true, ..Default::default() }, Duration::ZERO, (16, 8));
        assert_eq!(last_refusal_text(), None);
        assert!(session.holding());
        session.publish_status(&shm, Some((16, 8)));
        assert_ne!(header.preupscale_state.load(Ordering::Relaxed), IDENTIFIED);
        assert_eq!(header.preupscale_reason(), "");
    }

    /// A game that moves the colour input out of GENERAL in a buffer in front of DLSS's launch buffer and back
    /// after it, every frame: the split's check sees the layout in front of the launch and refuses; the record-time
    /// choice must not hand the next frame to the split on the GENERAL the end of the submission left, but hold
    /// inside the buffer with the layout the launch had (a guess, checked at the submit).
    #[test]
    fn a_layout_set_in_front_of_dlss_s_buffer_is_held_inside_not_left_to_the_split() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let frame = |t: &mut Tracker, front: u64, launch: u64, back: u64| {
            t.barrier(cb(front), ImageSync { old_layout: GENERAL, ..ImageSync::to(colour(), read_only) });
            t.launch(cb(launch), Some(&params));
            t.barrier(cb(back), ImageSync { old_layout: read_only, ..ImageSync::to(colour(), GENERAL) });
        };
        frame(&mut t, 1, 2, 3);
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::LeftToSplit), "nothing known of the launch's layout yet");
        let scan = t.submit_ok(&[vec![cb(1), cb(2), cb(3)]]).expect("launch-bearing");
        assert_eq!(scan.colour_layout, Some(read_only));
        assert!(!scan.colour_in_general(), "the split refuses it");
        // The next frame, recorded in fresh buffers: held inside, with the launch's layout as the guess.
        frame(&mut t, 4, 5, 6);
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        assert_eq!((point.layout, point.guessed.colour, point.hazard), (read_only, Some(read_only), Hazard::NotSplittable));
        // GTA V's pattern (GENERAL at the launch): still the split's.
        let (mut t, params) = held_tracker_with(copyable);
        for (launch, back) in [(2, 3), (5, 6)] {
            t.launch(cb(launch), Some(&params));
            t.barrier(cb(back), general_to_general(colour()));
            let (b, c) = t.take_inline_at().expect("first colour launch");
            assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::LeftToSplit));
            assert!(t.submit_ok(&[vec![cb(launch), cb(back)]]).expect("launch-bearing").colour_in_general());
        }
    }

    /// DLSS given part of its colour input (NGX's subrect; Unreal Engine's 1485x835 inside 1488x836, #51): the hold
    /// inside its buffer works on that rectangle, each input at its own base, also where the split could have taken
    /// the buffer, and the split refuses the frame (it works on the whole image). A rectangle outside the colour input
    /// refuses the hold.
    #[test]
    fn a_subrect_is_held_inside_on_its_rectangle_and_never_split() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let sub = crate::ngx_ring::Record {
            render: (1700, 956),
            colour_base: (4, 2),
            mvec_base: (1, 0),
            depth_base: (0, 3),
            ..sr_record(2, [0x100, 0x300, 0x200, 0, 0])
        };
        crate::ngx_ring::inject(vec![sub]);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(c, sub);
        let point = t.inline_point(b, &c).expect("held inside: GENERAL, nothing in the way, but a subrect");
        assert_eq!((point.desc.width, point.desc.height, point.hazard), (1700, 956, Hazard::Subrect));
        assert_eq!(point.offsets, Offsets { colour: (4, 2), mvec: (1, 0), depth: (0, 3) });
        assert_eq!(point.mvec_extent, (1700, 956));
        assert!(point.mvec_input.is_some_and(|m| m.readable), "{:?}", point.mvec_input);
        let outside = crate::ngx_ring::Record { colour_base: (8, 0), ..sub };
        assert_eq!(
            t.inline_point(b, &outside).err(),
            Some(InlineRefusal::SubrectOutside(crate::ngx_ring::Rect { at: (8, 0), size: (1700, 956) }, (1707, 960)))
        );
        // The motion vectors' rectangle outside their image: the hold goes without them.
        let point = t.inline_point(b, &crate::ngx_ring::Record { mvec_base: (8, 0), ..sub }).expect("held inside");
        assert!(point.mvec_input.is_some_and(|m| !m.readable) && point.mvec_extent == (0, 0), "{point:?}");
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.hazard, Some(Hazard::Subrect), "the split refuses it");
        // The whole image again: the split's, as before.
        let (mut t, params) = held_tracker_with(copyable);
        t.launch(cb(3), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::LeftToSplit));
        assert_eq!(t.submit_ok(&[vec![cb(3)]]).expect("launch-bearing").hazard, None);
    }

    /// Synchronization on a DLSS input inside the launch buffer, before the launch, is something
    /// the hold would run ahead of: each kind makes the submit one to forward untouched, also when
    /// the layout stays `GENERAL`.
    #[test]
    #[allow(clippy::type_complexity)]
    fn synchronization_before_the_launch_in_its_buffer_forbids_the_hold() {
        let transfer = ImageSync { src_queue_family: 0, dst_queue_family: 1, ..general_to_general(colour()) };
        let cases: [(&str, &dyn Fn(&mut Tracker), Hazard); 5] = [
            ("GENERAL -> GENERAL on the colour input", &|t| t.barrier(cb(2), general_to_general(colour())), Hazard::SameLayoutBarrier),
            ("a queue-family ownership transfer", &|t| t.barrier(cb(2), transfer), Hazard::QueueFamilyTransfer),
            (
                "a layout transition",
                &|t| t.barrier(cb(2), ImageSync { old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, ..ImageSync::to(colour(), GENERAL) }),
                Hazard::LayoutTransition,
            ),
            ("a global memory barrier", &|t| t.global(cb(2), Hazard::MemoryBarrier), Hazard::MemoryBarrier),
            ("an event wait", &|t| t.global(cb(2), Hazard::EventWait), Hazard::EventWait),
        ];
        for (what, record, hazard) in cases {
            let (mut t, params) = held_tracker();
            record(&mut t);
            t.launch(cb(2), Some(&params));
            let scan = t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing");
            assert_eq!(scan.hazard, Some(hazard), "{what}");
            assert!(!hazard.why().is_empty());
            // With the hold inside DLSS's buffer on, the line says that hold takes the buffer, never "untouched".
            assert!(hazard.why_inside().contains("the hold inside DLSS's buffer takes such buffers") && !hazard.why_inside().contains("untouched"));
            // The layout alone would have allowed it.
            assert!(scan.colour_in_general(), "{what}");
            // Re-recorded without it: held again.
            t.begin(cb(2));
            t.launch(cb(2), Some(&params));
            assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, None, "{what}");
        }
        // A launch that names other images first, then the barrier, then the launch naming the
        // colour input: what counts is what precedes the launch that reads it.
        let (mut t, params) = held_tracker();
        t.launch(cb(2), Some(&param_block(&[0x1200])));
        t.barrier(cb(2), general_to_general(colour()));
        t.launch(cb(2), Some(&params));
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, Some(Hazard::SameLayoutBarrier));
    }

    /// No hold goes inside a buffer begun before the identification: the layout its earlier commands
    /// left the colour input in, and the synchronization before the launch, were not seen. Begun again
    /// since, the same recording gets its hold.
    #[test]
    fn no_hold_goes_inside_a_buffer_begun_before_the_identification() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let early = cb(HELD_BEGUN + 1);
        t.global(early, Hazard::MemoryBarrier);
        t.launch(early, Some(&params));
        assert_eq!(t.take_inline_at().map(|(b, r)| (b, r.colour)), Some((early, colour())));
        assert_eq!(t.inline_point(early, &record_on(colour())).err(), Some(InlineRefusal::BegunBefore), "recorded (in part) before the identification");
        t.begin(early);
        t.global(early, Hazard::MemoryBarrier);
        t.launch(early, Some(&params));
        assert_eq!(t.take_inline_at().map(|(b, r)| (b, r.colour)), Some((early, colour())));
        assert_eq!(t.inline_point(early, &record_on(colour())).ok().map(|p| p.hazard), Some(Hazard::MemoryBarrier), "begun since: held inside");
    }

    /// The hold goes inside DLSS's buffer exactly where the split is refused: Crimson Desert's and
    /// Cyberpunk 2077's buffers render the frame (draws, dispatches, write barriers) before DLSS's
    /// first colour launch. Not for GTA V's pattern (the split takes it), not for a colour input the
    /// layer cannot copy or that is not in GENERAL there, and only at the buffer's first colour
    /// launch.
    #[test]
    fn the_hold_goes_inside_the_buffer_only_where_the_split_is_refused() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        // The frame rendered in the buffer: a write barrier before the colour launch.
        let (mut t, params) = held_tracker_with(copyable);
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), Some(&params));
        assert_eq!(t.take_inline_at().map(|(b, r)| (b, r.colour)), Some((cb(2), colour())));
        let point = t.inline_point(cb(2), &record_on(colour())).expect("held inside the buffer");
        assert_eq!((point.colour, point.hazard), (colour(), Hazard::MemoryBarrier));
        assert_eq!((point.desc.width, point.desc.height), (1707, 960));
        // A later launch of the same buffer naming it again: no second point.
        t.launch(cb(2), Some(&params));
        assert!(t.take_inline_at().is_none());
        // A buffer whose launch names the colour input after the evaluate's end (DLSS Frame Generation's, Black Myth:
        // Wukong with frame generation on): no evaluate's, no hold inside it.
        let rec = record_on(colour());
        crate::ngx_ring::inject(vec![rec, crate::ngx_ring::end_of(&rec, 2)]);
        t.global(cb(3), Hazard::MemoryBarrier);
        t.launch(cb(3), Some(&param_block(&[0x1100])));
        assert!(t.take_inline_at().is_none());
        t.launch(cb(3), Some(&params));
        assert!(t.take_inline_at().is_none());
        // The next evaluate's first launch naming the colour input, whatever else it names (Cyberpunk 2077's
        // `cuda_luma_convert_kernel`, before the launch naming depth and motion vectors): the hold goes there.
        crate::ngx_ring::inject(vec![rec, crate::ngx_ring::end_of(&rec, 2), sr_record(3, [0x100, 0x300, 0x200, 0, 0])]);
        t.global(cb(4), Hazard::MemoryBarrier);
        t.launch(cb(4), Some(&param_block(&[0x1100])));
        assert_eq!(t.take_inline_at().map(|(b, r)| (b, r.colour, r.seq)), Some((cb(4), colour(), 3)));
        // GTA V's pattern: nothing before the launch, the split holds it.
        let (mut t, params) = held_tracker_with(copyable);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::LeftToSplit));
        // Barriers on depth only (GTA V with frame generation): the split holds it too.
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), general_to_general(vk::Image::from_raw(0x200)));
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::LeftToSplit));
        // Not copyable (STORAGE only, an image created before the layer could add the transfer usages): nothing.
        let (mut t, params) = held_tracker();
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::Usage(vk::ImageUsageFlags::STORAGE)));
        // Left in SHADER_READ_ONLY_OPTIMAL by the buffer's own barrier: held inside, moved to the
        // transfer layouts and back to that one; in a layout outside those the hold handles: nothing.
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), ImageSync { old_layout: GENERAL, ..ImageSync::to(colour(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) });
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).ok().map(|p| p.layout), Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), ImageSync { old_layout: GENERAL, ..ImageSync::to(colour(), vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL) });
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).err(), Some(InlineRefusal::Layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)));
        // A launch naming no evaluate's colour input leaves no mark.
        let (mut t, _) = held_tracker_with(copyable);
        t.launch(cb(3), Some(&param_block(&[0x1200])));
        assert!(t.take_inline_at().is_none());
    }

    /// The diagnostic modes' split hold waits on the host behind what is submitted in front of its capture:
    /// only behind timeline values already reached (a binary semaphore's wait is never checked), and never
    /// behind a buffer that waits on an event, whatever synchronization came first in it.
    #[test]
    fn diagnostic_holds_wait_only_behind_what_needs_no_later_host_signal() {
        let (mut t, _) = held_tracker();
        t.global(cb(5), Hazard::MemoryBarrier);
        t.global(cb(5), Hazard::EventWait);
        assert_eq!(t.global_sync.get(&cb(5)), Some(&Hazard::MemoryBarrier), "the first synchronization stays the hazard");
        assert!(t.waits_on_event(&[cb(4), cb(5)]));
        assert!(!t.waits_on_event(&[cb(4)]));
        t.begin(cb(5));
        assert!(!t.waits_on_event(&[cb(5)]), "begun again");
        // SAFETY: a test device of its own, everything destroyed below.
        unsafe {
            let Ok(entry) = ash::Entry::load() else { return };
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let Ok(instance) = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) else { return };
            let Some(&physical) = instance.enumerate_physical_devices().ok().as_ref().and_then(|d| d.first()) else {
                instance.destroy_instance(None);
                return;
            };
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&[1.0])];
            let device = instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues).push_next(&mut v12), None).unwrap();
            let mut ty = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(3);
            let timeline = device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None).unwrap();
            let binary = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
            let is_timeline = |s| s == timeline;
            assert!(waits_reached(&device, [(timeline, 3), (binary, 99)], is_timeline), "reached, and a binary semaphore's value ignored");
            assert!(!waits_reached(&device, [(timeline, 2), (timeline, 4)], is_timeline), "a value the host signals only later");
            device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(timeline).value(4)).unwrap();
            assert!(waits_reached(&device, [(timeline, 4)], is_timeline));
            device.destroy_semaphore(timeline, None);
            device.destroy_semaphore(binary, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A layout the buffer set before the hold point (its own barrier) is no guess; one taken from what the
    /// buffers submitted before left, or `GENERAL` for a storage image never seen in a barrier, is. When a
    /// buffer runs, an image's layout is what the last buffer in front of it in the submission set, else what
    /// the submitted ones left.
    #[test]
    fn layouts_the_buffer_did_not_set_are_guesses_checked_against_the_buffers_before() {
        use vk::ImageLayout as L;
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        assert_eq!(point.guessed.colour, Some(L::GENERAL), "a storage image no barrier was seen on");
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), ImageSync { old_layout: L::GENERAL, ..ImageSync::to(colour(), L::SHADER_READ_ONLY_OPTIMAL) });
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        assert_eq!((point.layout, point.guessed.colour), (L::SHADER_READ_ONLY_OPTIMAL, None), "set by the buffer's own barrier");
        // When a buffer runs.
        t.barrier(cb(7), ImageSync { old_layout: L::SHADER_READ_ONLY_OPTIMAL, ..ImageSync::to(colour(), L::GENERAL) });
        assert_eq!(t.layout_when_run(&[cb(7)], colour()), Some(L::GENERAL), "the buffer in front set it");
        t.commit_submitted([cb(2)]);
        assert_eq!(t.layout_when_run(&[], colour()), Some(L::SHADER_READ_ONLY_OPTIMAL), "what the submitted ones left");
        assert_eq!(t.layout_when_run(&[cb(2), cb(7)], colour()), Some(L::GENERAL), "the last in front wins");
        assert_eq!(t.layout_when_run(&[cb(8)], colour()), Some(L::SHADER_READ_ONLY_OPTIMAL), "a buffer that does not set it");
    }

    /// Once a hold inside DLSS's buffer was recorded for an identification, a buffer the split
    /// could take is held inside too (Hogwarts Legacy refuses the split for some of its DLSS
    /// buffers only): the two holds alternating rebuilt the hold's resources, and the network's
    /// pipelines and history with them, every frame. Another identification starts over.
    #[test]
    fn once_held_inside_every_buffer_of_that_identification_is() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        // GTA V's pattern before any hold inside was recorded: the split's.
        t.launch(cb(3), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, &c).is_err());
        // Recorded: the same pattern is held inside now.
        t.inline_recorded(point.identification);
        t.launch(cb(4), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, &c).ok().map(|p| p.hazard), Some(Hazard::HeldInside));
        // A hold inside recorded for another identification: the split's again.
        t.inline_recorded(point.identification + 1);
        t.launch(cb(5), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, &c).is_err());
    }

    /// Motion vectors are in their own image's pixels: at the colour input's extent they are its pixels,
    /// high-resolution ones (Hogwarts Legacy's 2560x1440 for a 1708x964 colour input) scale down by the
    /// ratio of the extents, per axis; none known is a scale of 1.
    #[test]
    fn motion_vectors_scale_by_the_colour_input_over_their_extent() {
        let point = |mvec_extent| InlinePoint {
            colour: vk::Image::null(),
            image: (1708, 964),
            desc: ImageDesc { width: 1708, height: 964, format: vk::Format::R16G16B16A16_SFLOAT, usage: vk::ImageUsageFlags::STORAGE, plain: true },
            offsets: Offsets::default(),
            layout: vk::ImageLayout::GENERAL,
            exposure_input: None,
            mvec_input: None,
            mvec_extent,
            mv_units: None,
            depth_input: None,
            identification: 1,
            hazard: Hazard::MemoryBarrier,
            hold: placement::Hold::ALL_BEFORE,
            guessed: Guessed::default(),
        };
        assert_eq!(point((1708, 964)).mvec_scale(), [1.0, 1.0]);
        assert_eq!(point((2560, 1440)).mvec_scale(), [1708.0 / 2560.0, 964.0 / 1440.0]);
        assert_eq!(point((0, 0)).mvec_scale(), [1.0, 1.0]);
    }

    /// An implausible game exposure (Wukong's DLSS-internal 1x1 read 59,456) stops that
    /// identification from using the game's exposure image; plausible ones (GTA V's 0.22, Cyberpunk's
    /// 3.3) never do, and a measured exposure is not judged.
    #[test]
    fn an_implausible_game_exposure_switches_that_identification_to_the_measured_one() {
        let mut session = Session::default();
        let result = |source, e| HoldResult { exposure_source: Some(source), exposure: Some(e), ..HoldResult::default() };
        session.check_exposure(5, &result(ExposureSource::Game, 0.2195));
        session.check_exposure(6, &result(ExposureSource::Game, 3.32));
        assert!(session.game_exposure_trusted(5) && session.game_exposure_trusted(6));
        session.check_exposure(7, &result(ExposureSource::Game, 3.32));
        session.check_exposure(7, &result(ExposureSource::Auto, 59456.0));
        assert!(session.game_exposure_trusted(7));
        session.check_exposure(7, &result(ExposureSource::Game, 59456.0));
        assert!(!session.game_exposure_trusted(7));
        assert!(session.game_exposure_trusted(8), "a new identification starts trusted");
    }

    /// A game exposure that jumps by more than [`EXPOSURE_JUMP_MAX`] between holds, again and again,
    /// is not eye adaptation (Hogwarts Legacy with the model on: 0.0023, 0.000013, 0.0021, ...): that
    /// identification switches to the measured exposure. Adaptation over frames (Hogwarts Legacy
    /// untouched: 0.0027-0.0034; a fast scene change, 16x), a cut there and back (Crimson Desert's
    /// menus: 0.0069 to 0.91), cuts far apart and a jump across identifications do not; the native
    /// hold's value, read a hold late, counts the same.
    /// Metro Last Light Redux's exposure image holds 0: after [`EXPOSURE_UNUSABLE_HOLDS`] such holds in a row the
    /// exposure is measured from the frame. Fewer, or a usable value between them, or another identification's,
    /// do not; a NaN counts as unusable too; the native hold's value, read a hold late, counts the same.
    #[test]
    fn a_game_exposure_that_keeps_reading_zero_switches_to_the_measured_one() {
        let game = |e| HoldResult { exposure_source: Some(ExposureSource::Game), exposure: Some(e), ..HoldResult::default() };
        let native = |e| HoldResult { exposure_source: Some(ExposureSource::Game), previous_exposure: Some(e), ..HoldResult::default() };
        let mut session = Session::default();
        for _ in 1..EXPOSURE_UNUSABLE_HOLDS {
            session.check_exposure(3, &game(0.0));
        }
        session.check_exposure(3, &game(0.5));
        session.check_exposure(3, &game(0.0));
        session.check_exposure(4, &game(0.0));
        assert!(session.game_exposure_trusted(3) && session.game_exposure_trusted(4), "interrupted, and another identification");
        for _ in 0..EXPOSURE_UNUSABLE_HOLDS - 1 {
            session.check_exposure(5, &native(0.0));
        }
        assert!(session.game_exposure_trusted(5));
        session.check_exposure(5, &native(f32::NAN));
        assert!(!session.game_exposure_trusted(5), "{EXPOSURE_UNUSABLE_HOLDS} unusable in a row");
        assert!(session.game_exposure_trusted(3));
        // A positive exposure the shaders refuse (a half float's subnormal: 5.96e-8 to 9.5e-7, at or below 1e-6)
        // is as unusable as 0, and a value just above it is usable.
        for _ in 0..EXPOSURE_UNUSABLE_HOLDS {
            assert!(!hdr::exposure_ok(6.0e-8));
            session.check_exposure(6, &game(6.0e-8));
        }
        assert!(!session.game_exposure_trusted(6), "a subnormal exposure leads to the measured one");
        for _ in 0..EXPOSURE_UNUSABLE_HOLDS {
            session.check_exposure(7, &game(2.0e-6));
        }
        assert!(session.game_exposure_trusted(7) && hdr::exposure_ok(2.0e-6));
    }

    #[test]
    fn a_game_exposure_that_keeps_jumping_switches_to_the_measured_one() {
        let mut session = Session::default();
        let game = |e| HoldResult { exposure_source: Some(ExposureSource::Game), exposure: Some(e), ..HoldResult::default() };
        let native = |e| HoldResult { exposure_source: Some(ExposureSource::Game), previous_exposure: Some(e), ..HoldResult::default() };
        for e in [0.0032, 0.0027, 0.0034, 0.0031, 0.0496] {
            session.check_exposure(3, &game(e));
        }
        assert!(session.game_exposure_trusted(3), "adaptation, and a 16x step, are not jumps");
        for e in [0.0069, 0.91, 0.91, 0.0069] {
            session.check_exposure(3, &game(e));
        }
        assert!(session.game_exposure_trusted(3), "a cut there and back");
        for _ in 0..=EXPOSURE_JUMP_WINDOW {
            session.check_exposure(3, &game(0.0069));
        }
        session.check_exposure(3, &game(0.91));
        assert!(session.game_exposure_trusted(3), "cuts far apart");
        session.check_exposure(4, &game(0.000013));
        assert!(session.game_exposure_trusted(4), "a new identification is compared with nothing");
        session.check_exposure(9, &native(0.0023));
        session.check_exposure(9, &HoldResult { exposure_source: Some(ExposureSource::Game), ..HoldResult::default() });
        assert!(session.game_exposure_trusted(9), "a hold without a value read is not judged");
        session.check_exposure(9, &native(0.000013));
        session.check_exposure(9, &native(0.0021));
        assert!(session.game_exposure_trusted(9), "two jumps");
        session.check_exposure(9, &native(0.000014));
        assert!(!session.game_exposure_trusted(9));
        assert!(session.game_exposure_trusted(3));
    }

    /// Unreal Engine 5's DLSS packs two 32-bit view handles into one 8-byte parameter word (Black
    /// Myth: Wukong's benchmark, 2026-10-05: `0x3201c2301401c00` holds `0x3201c23` and `0x1401c00`):
    /// both are read, in memory order (low half first). A word holding one handle is read as before.
    #[test]
    fn two_32_bit_handles_packed_in_one_parameter_word_are_both_read() {
        let (mut t, _) = held_tracker();
        // held_tracker's keys: 0x1100 (colour), 0x1200 (depth), 0x1300 (motion vectors).
        let packed = |lo: u64, hi: u64| lo | hi << 32;
        t.launch(cb(5), Some(&param_block(&[packed(0x1200, 0x1100), 0x1300])));
        let refs = &t.launch[&cb(5)];
        assert_eq!(refs.images, vec![vk::Image::from_raw(0x200), vk::Image::from_raw(0x100), vk::Image::from_raw(0x300)]);
        assert!(refs.ngx.is_some_and(|r| r.colour == vk::Image::from_raw(0x100)), "the packed colour handle matches the evaluate");
        // A word that is one 64-bit value is not split: a key equal to its low half is not named.
        let (mut t, _) = held_tracker();
        t.launch(cb(6), Some(&param_block(&[0x1100])));
        assert_eq!(t.launch[&cb(6)].images, vec![vk::Image::from_raw(0x100)]);
    }

    /// Unreal Engine 5 (Black Myth: Wukong's benchmark): DLSS's colour input is a sampled, not storage,
    /// `B10G11R11_UFLOAT` image, as NGX's record names it. The split cannot hold an
    /// R11G11B10 input, so the hold goes inside the buffer even with nothing before the launch,
    /// in the layout the buffer's barrier left it in.
    #[test]
    fn a_sampled_r11g11b10_colour_input_is_identified_and_held_inside_the_buffer() {
        let mut t = Tracker::default();
        let usage =
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        for (raw, format, usage) in [
            (0x100, vk::Format::B10G11R11_UFLOAT_PACK32, usage),
            (0x200, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            (0x300, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::STORAGE),
        ] {
            let info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D { width: 1488, height: 836, depth: 1 },
                format,
                usage,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            t.record_image(vk::Image::from_raw(raw), &info);
            t.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        let params = param_block(&[0x1100, 0x1200, 0x1300]);
        crate::ngx_ring::inject(vec![sr_record(1, [0x100, 0x300, 0x200, 0, 0])]);
        t.launch(cb(900), Some(&params));
        let _ = submit(&mut t, &[vec![cb(900)]]);
        t.free(&[cb(900)]);
        let inputs = t.inputs.expect("identified");
        assert_eq!((inputs.colour.0, inputs.colour.1.format), (vk::Image::from_raw(0x100), vk::Format::B10G11R11_UFLOAT_PACK32));
        t.barrier(
            cb(2),
            ImageSync {
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..ImageSync::to(vk::Image::from_raw(0x100), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            },
        );
        let _ = submit(&mut t, &[vec![cb(2)]]);
        t.begin(cb(2));
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        assert_eq!((point.layout, point.hazard), (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, Hazard::NotSplittable));
        assert_eq!(point.mvec_input.map(|a| a.image), Some(vk::Image::from_raw(0x300)));
        // The motion vectors destroyed since the record was written: the hold copies nothing from them.
        t.forget_image(vk::Image::from_raw(0x300));
        let point = t.inline_point(b, &c).expect("still held inside the buffer");
        assert!(point.mvec_input.is_none(), "a destroyed motion-vector image is copied from");
    }

    /// Expedition 33 (Unreal Engine 5 under vkd3d-proton, #51): DLSS's input cycles through several 1488x836
    /// B10G11R11 images, each moved into its layout in a buffer in front of DLSS's. Every image of the identified
    /// colour input's shape is watched, so the hold inside DLSS's buffer knows the layout of each, not only of the
    /// identified one (before: "the colour input's layout at DLSS's launch is unknown" for the others, frames
    /// flickering through unprocessed). When the identified image is destroyed and another is identified, what was
    /// known of the images watched throughout is kept: the next hold on one needs no new barrier on it.
    /// GTA V Enhanced hands DLSS four depth/motion-vector pairs and exposure images in turn (2026-11-11): once each
    /// was seen, going round them is no new identification, so the holds are not skipped frame after frame.
    #[test]
    fn rotating_motion_vectors_and_exposure_images_are_one_identification() {
        let rg = vk::Format::R16G16_SFLOAT;
        let mut set = vec![(
            0x100,
            desc(
                1488,
                836,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            ),
        )];
        for k in 0..4u64 {
            set.push((0x200 + k, desc(1488, 836, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)));
            set.push((0x300 + k, desc(1488, 836, rg, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)));
            set.push((0x600 + k, desc(1, 1, vk::Format::R16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)));
        }
        let mut t = tracker_with(&set_of(&set));
        let mut generations = Vec::new();
        for frame in 0..12u64 {
            let k = frame % 4;
            crate::ngx_ring::inject(vec![sr_record(frame + 1, [0x100, 0x300 + k, 0x200 + k, 0, 0x600 + k])]);
            t.begin(cb(11));
            t.launch(cb(11), Some(&param_block(&[h(0x100), h(0x200 + k), h(0x300 + k), h(0x600 + k)])));
            let _ = submit(&mut t, &[vec![cb(11)]]);
            generations.push(t.generation);
            assert_eq!(t.inputs.map(|i| i.mvec.0), Some(img(0x300 + k)), "each frame's own motion vectors");
        }
        assert!(generations[4..].iter().all(|&g| g == generations[3]), "no new identification once round: {generations:?}");
    }

    /// 007 First Light after a load (#49): DLSS gets a brand-new colour input and motion-vector image every 0.4 s,
    /// never one seen before. Images of the same shapes are the same identification, so no hold is refused as begun
    /// before it (each new image was a new identification, and a frame or two showed the game's own picture).
    #[test]
    fn brand_new_images_of_the_same_shapes_are_the_same_identification() {
        let rg = vk::Format::R16G16_SFLOAT;
        let colour_usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let mut set = vec![(0x200, desc(1488, 836, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))];
        for k in 0..6u64 {
            set.push((0x100 + 0x10 * k, desc(1488, 836, vk::Format::R16G16B16A16_SFLOAT, colour_usage)));
            set.push((0x300 + k, desc(1488, 836, rg, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)));
        }
        let mut t = tracker_with(&set_of(&set));
        let mut generations = Vec::new();
        for k in 0..6u64 {
            let (colour, mvec) = (0x100 + 0x10 * k, 0x300 + k);
            crate::ngx_ring::inject(vec![sr_record(k + 1, [colour, mvec, 0x200, 0, 0])]);
            t.begin(cb(11));
            t.launch(cb(11), Some(&param_block(&[h(colour), h(0x200), h(mvec)])));
            let (scan, _) = submit(&mut t, &[vec![cb(11)]]);
            generations.push(t.generation);
            let scan = scan.expect("launch-bearing");
            assert_eq!(scan.inputs.map(|i| (i.colour.0, i.mvec.0)), Some((img(colour), img(mvec))), "each frame's own images");
            if k > 0 {
                assert_ne!(scan.hazard, Some(Hazard::RecordedBefore), "frame {k}: not refused as begun before a new identification");
            }
        }
        assert!(generations.iter().all(|&g| g == generations[0]), "one identification: {generations:?}");
    }

    #[test]
    fn every_image_of_the_colour_input_s_shape_has_its_layout_known_across_identifications() {
        let r11 = vk::Format::B10G11R11_UFLOAT_PACK32;
        let colour_usage =
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let set = set_of(&[
            (0x100, desc(1488, 836, r11, colour_usage)),
            (0x110, desc(1488, 836, r11, colour_usage)),
            (0x120, desc(1488, 836, r11, colour_usage)),
            (0x200, desc(1488, 836, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)),
            (0x300, desc(1488, 836, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)),
            (0x600, desc(1, 1, vk::Format::R16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)),
        ]);
        let mut t = tracker_with(&set);
        let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let rendered = |image: u64| ImageSync { old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, ..ImageSync::to(img(image), read_only) };
        // One frame: the game's buffer moves the image into SHADER_READ_ONLY (when `front`), DLSS's buffer, begun
        // since, names it. The hold inside DLSS's buffer is decided on at the launch, then the two are submitted.
        let mut seq = 0;
        let mut frame = |t: &mut Tracker, colour: u64, front: bool| -> Option<Result<InlinePoint, InlineRefusal>> {
            seq += 1;
            crate::ngx_ring::inject(vec![sr_record(seq, [colour, 0x300, 0x200, 0, 0x600])]);
            t.begin(cb(10));
            t.begin(cb(11));
            if front {
                t.barrier(cb(10), rendered(colour));
            }
            t.launch(cb(11), Some(&param_block(&[h(colour), h(0x200), h(0x300), h(0x600)])));
            let point = t.take_inline_at().map(|(b, c)| t.inline_point(b, &c));
            let _ = submit(t, &[vec![cb(10), cb(11)]]);
            point
        };
        let cycle = [0x100, 0x110, 0x120];
        frame(&mut t, cycle[0], true);
        assert!(t.inputs.is_some_and(|i| i.colour.0 == img(0x100)), "identified by the record");
        // A full cycle since: each image's barrier was committed while watched.
        for colour in cycle {
            frame(&mut t, colour, true);
        }
        for k in 0..6 {
            let colour = cycle[k % 3];
            let point = frame(&mut t, colour, true).expect("a hold point").unwrap_or_else(|r| panic!("{colour:#x}: refused, {r:?}"));
            assert_eq!((point.colour, point.layout, point.hazard), (img(colour), read_only, Hazard::NotSplittable), "{colour:#x}");
            assert_eq!(t.launch_layouts.get(&img(colour)), Some(&read_only), "{colour:#x}: its layout at DLSS's launch");
        }
        assert_eq!(t.colour_barriers.get(&img(0x110)).map(|b| b.1), Some(read_only), "barriers counted per image");
        assert_eq!(t.generation, 1, "another image of the same shape is no new identification");
        // Other motion vectors of the same shape: still the same identification, with what was committed on the images
        // watched throughout. No barrier is recorded from here on.
        t.forget_image(img(0x100));
        let mvec2 = desc(1488, 836, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED);
        t.record_image(
            img(0x310),
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D { width: mvec2.width, height: mvec2.height, depth: 1 },
                format: mvec2.format,
                usage: mvec2.usage,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            },
        );
        crate::ngx_ring::inject(vec![sr_record(1000, [0x110, 0x310, 0x200, 0, 0x600])]);
        let _ = submit(&mut t, &[vec![cb(11)]]);
        assert_eq!((t.generation, t.inputs.map(|i| i.colour.0)), (1, Some(img(0x110))));
        for colour in [0x110, 0x120] {
            t.begin(cb(11));
            crate::ngx_ring::inject(vec![sr_record(1001 + colour, [colour, 0x310, 0x200, 0, 0x600])]);
            t.launch(cb(11), Some(&param_block(&[h(colour), h(0x200), h(0x600)])));
            let (b, c) = t.take_inline_at().expect("a hold point");
            let point = t.inline_point(b, &c).unwrap_or_else(|r| panic!("{colour:#x}: refused, {r:?}"));
            assert_eq!(point.layout, read_only, "{colour:#x}: committed before the identification changed");
        }
    }

    /// What is kept at a new identification ([`Tracker::keep_watched_across`]): what was recorded on an image
    /// watched before and after it; an image watched before only (the old depth with nothing identified, an image
    /// of the old colour shape after the shape changed) is dropped, as its later barriers go unseen.
    #[test]
    fn a_new_identification_keeps_only_what_was_watched_throughout() {
        let (mut t, _) = held_tracker();
        let same = vk::Image::from_raw(0x110);
        let info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        t.record_image(same, &info);
        let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        for image in [colour(), same, vk::Image::from_raw(0x200)] {
            t.barrier(cb(1), ImageSync::to(image, read_only));
        }
        // Recorded, not yet submitted: kept as the commit's input, also for an image of the shape alone.
        t.barrier(cb(2), ImageSync::to(same, GENERAL));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        assert_eq!(t.committed.len(), 3, "the colour input, an image of its shape and the depth are watched");
        let watch = t.colour_watch;
        assert_eq!(watch, Some(ColourWatch { width: 1707, height: 960, format: vk::Format::R16G16B16A16_SFLOAT }));
        // Nothing identified, the shape still watched.
        t.keep_watched_across(None, watch);
        assert_eq!((t.committed.get(&colour()), t.committed.get(&same)), (Some(&read_only), Some(&read_only)));
        assert!(!t.committed.contains_key(&vk::Image::from_raw(0x200)), "the depth is no longer watched");
        assert_eq!(t.pending.get(&cb(2)).map(Vec::len), Some(1));
        // Another shape: nothing of the old one is kept.
        t.keep_watched_across(None, Some(ColourWatch { width: 1280, height: 720, format: vk::Format::R16G16B16A16_SFLOAT }));
        assert!(t.committed.is_empty() && t.pending.is_empty());
    }

    /// Stellar Blade (Unreal Engine 4 under vkd3d-proton, #51): DLSS's buffer has the game's same-layout barrier on
    /// the colour input (render target to shader resource) before its input launch. The split refuses that buffer;
    /// the hold inside it takes it, after the barrier, in the layout the barrier set. When the hold inside is not
    /// even tried in a buffer the split refuses, the reason is said once per reason; one begun before the
    /// identification is counted, and logged with the counts.
    #[test]
    fn a_same_layout_barrier_before_dlss_s_launch_is_left_to_the_hold_inside_and_a_refusal_says_why() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let rt_to_srv = ImageSync {
            src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw(),
            dst_access: vk::AccessFlags2::SHADER_READ.as_raw(),
            ..general_to_general(colour())
        };
        t.barrier(cb(2), rt_to_srv);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, &c).expect("held inside the buffer");
        assert_eq!((point.layout, point.guessed.colour, point.hazard), (GENERAL, None, Hazard::SameLayoutBarrier));
        t.inline_tried.insert(cb(2));
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, Some(Hazard::SameLayoutBarrier), "the split refuses it");
        assert_eq!(t.untried_line(cb(2)), None, "tried: its own refusal or the tally says the rest");
        // Launches whose parameters cannot be read: never a hold point, said once.
        t.barrier(cb(3), rt_to_srv);
        t.launch(cb(3), None);
        assert!(t.submit_ok(&[vec![cb(3)]]).is_none(), "matched to no evaluate: no hold point");
        assert!(t.untried_line(cb(3)).is_some_and(|l| l.contains("not readable")));
        assert_eq!(t.untried_line(cb(3)), None, "once");
        // A buffer begun before the identification: refused, and no longer silently.
        let early = cb(HELD_BEGUN + 1);
        let refusal = InlineRefusal::BegunBefore;
        assert!(refusal.quiet() && !refusal.said().is_empty());
        assert!(Refusal::Inside(refusal).text().contains("begun before"));
        let now = Instant::now();
        let first = t.quiet_refusal_line(refusal, now).expect("the first is logged");
        assert!(first.contains("1 buffer(s) begun before"), "{first}");
        assert_eq!(t.quiet_refusal_line(refusal, now + Duration::from_secs(1)), None, "counted");
        let next = t.quiet_refusal_line(refusal, now + QUIET_REFUSALS_EVERY + Duration::from_secs(1)).expect("then with the counts");
        assert!(next.contains("2 buffer(s) begun before") && next.contains("in the last 6.0 s"), "{next}");
        t.launch(early, Some(&params));
        assert_eq!(t.inline_point(early, &record_on(colour())).err(), Some(refusal));
    }

    /// The depth a hold reads is the one this buffer's input launch names, with that image's own
    /// layout: GTA V Enhanced alternates two depth images, and the identified one holds the previous
    /// frame's depth on every other frame.
    #[test]
    fn the_hold_reads_the_depth_its_evaluate_names() {
        let (mut t, params) = held_tracker();
        let other = vk::Image::from_raw(0x400);
        let info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
            format: vk::Format::D32_SFLOAT_S8_UINT,
            usage: vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        t.record_image(other, &info);
        t.record_view(vk::ImageView::from_raw(0x401), other);
        t.register(vk::ImageView::from_raw(0x401), Some(0x1400));
        let _ = t.refresh();
        // The identified inputs stay; a depth-only change is taken silently.
        assert_eq!(t.inputs.map(|i| i.colour.0), Some(colour()));
        let read_only = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        t.barrier(cb(1), ImageSync { old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, ..ImageSync::to(other, read_only) });
        let _ = t.submit_ok(&[vec![cb(1)]]);
        let named_other = param_block(&[0x1100, 0x1400, 0x1300]);
        t.begin(cb(2));
        crate::ngx_ring::inject(vec![sr_record(2, [0x100, 0x300, 0x400, 0, 0])]);
        t.launch(cb(2), Some(&named_other));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.inputs.map(|i| i.depth.0), Some(other));
        assert_eq!(scan.depth_layout, Some(read_only), "the named depth's own layout");
        // The next frame names the identified one again.
        t.begin(cb(2));
        crate::ngx_ring::inject(vec![sr_record(3, [0x100, 0x300, 0x200, 0, 0])]);
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.inputs.map(|i| i.depth.0), Some(vk::Image::from_raw(0x200)));
    }

    /// GTA V Enhanced's launch buffer with frame generation on (`NEURAL_FORGE_PROBE_DEPTH`,
    /// 2026-10-10): before the input launch, read-only `GENERAL -> GENERAL` barriers on the frame's
    /// depth (one of two the game alternates between) around a copy into DLSS's own depth, which the
    /// launch names and which has no layout known. The hold reads the frame's depth, and the barriers
    /// do not stop it; a barrier after a write does.
    #[test]
    fn the_hold_reads_the_depth_the_launch_buffer_copies_from() {
        let (mut t, _) = held_tracker();
        let info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
            format: vk::Format::D32_SFLOAT_S8_UINT,
            usage: vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let (frame_depth, ring) = (vk::Image::from_raw(0x400), vk::Image::from_raw(0x500));
        for (image, raw) in [(frame_depth, 0x400), (ring, 0x500)] {
            t.record_image(image, &info);
            t.record_view(vk::ImageView::from_raw(raw + 1), image);
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        let _ = t.refresh();
        assert_eq!(t.inputs.map(|i| i.colour.0), Some(colour()));
        let sync = |src: vk::AccessFlags, dst: vk::AccessFlags| ImageSync {
            src_access: u64::from(src.as_raw()),
            dst_access: u64::from(dst.as_raw()),
            ..general_to_general(frame_depth)
        };
        // An earlier frame left it in GENERAL.
        t.barrier(cb(1), ImageSync::to(frame_depth, GENERAL));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        let names_ring = param_block(&[0x1100, 0x1500, 0x1300]);
        t.begin(cb(2));
        t.barrier(cb(2), sync(vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ));
        t.barrier(cb(2), sync(vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ));
        crate::ngx_ring::inject(vec![sr_record(2, [0x100, 0x300, 0x500, 0, 0])]);
        t.launch(cb(2), Some(&names_ring));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.inputs.map(|i| i.depth.0), Some(frame_depth), "the copy's source, not DLSS's ring");
        assert_eq!((scan.depth_layout, scan.depth_hazard, scan.hazard), (Some(GENERAL), None, None));
        // Written in the buffer before the launch: not the frame's final depth, and no depth this hold.
        t.begin(cb(2));
        t.barrier(cb(2), sync(vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ));
        crate::ngx_ring::inject(vec![sr_record(3, [0x100, 0x300, 0x400, 0, 0])]);
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1400, 0x1300])));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.inputs.map(|i| i.depth.0), Some(frame_depth));
        assert_eq!(scan.depth_hazard, Some(Hazard::SameLayoutBarrier));
    }

    /// GTA V Enhanced's launch buffer with frame generation on, once DLSS's rings are known (a probe of the launch buffer,
    /// 2026-11-11): before the input launch, the frame's depth and the frame's motion vectors are copied into the ring
    /// images the record names, between read-only `GENERAL -> GENERAL` barriers on the sources and with none on the
    /// rings, whose layout is known from frames before. A hold in front of the buffer reads the copies' sources: the
    /// rings' images are written after it (they held the inputs of three frames back, and the edit cache rejected
    /// all of a cached hold's history). An image only read before the launch is not one, nor a copy after it.
    #[test]
    fn the_hold_reads_the_depth_and_motion_vectors_the_launch_buffer_copies_from_into_known_rings() {
        let (mut t, _) = held_tracker();
        let info = |format| vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
            format,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let image = vk::Image::from_raw;
        let (frame_depth, ring_depth, frame_mvec, ring_mvec, sampled_mvec) = (image(0x400), image(0x500), image(0x700), image(0x800), image(0x900));
        for (i, format) in [
            (frame_depth, vk::Format::D32_SFLOAT_S8_UINT),
            (ring_depth, vk::Format::D32_SFLOAT_S8_UINT),
            (frame_mvec, vk::Format::R16G16_SFLOAT),
            (ring_mvec, vk::Format::R16G16_SFLOAT),
            (sampled_mvec, vk::Format::R16G16_SFLOAT),
        ] {
            let raw = i.as_raw();
            t.record_image(i, &info(format));
            t.record_view(vk::ImageView::from_raw(raw + 1), i);
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        // GTA V Enhanced's exposure images: 1x1 R16F, the ring DLSS reads filled by a copy in the launch buffer.
        let (frame_exposure, ring_exposure) = (image(0xa00), image(0xb00));
        for i in [frame_exposure, ring_exposure] {
            t.record_image(i, &vk::ImageCreateInfo { extent: vk::Extent3D { width: 1, height: 1, depth: 1 }, ..info(vk::Format::R16_SFLOAT) });
            t.record_view(vk::ImageView::from_raw(i.as_raw() + 1), i);
            t.register(vk::ImageView::from_raw(i.as_raw() + 1), Some(i.as_raw() + 0x1000));
        }
        // Identified with its exposure image, so the exposure ring is watched as the depth and motion vectors are.
        crate::ngx_ring::inject(vec![sr_record(1, [0x100, 0x800, 0x500, 0, 0xb00])]);
        let _ = t.refresh();
        // Earlier frames left all of them in GENERAL.
        for i in [frame_depth, ring_depth, frame_mvec, ring_mvec, sampled_mvec, frame_exposure, ring_exposure] {
            t.barrier(cb(1), ImageSync::to(i, GENERAL));
        }
        let _ = t.submit_ok(&[vec![cb(1)]]);
        let sync = |i: vk::Image, src: vk::AccessFlags, dst: vk::AccessFlags| ImageSync {
            src_access: u64::from(src.as_raw()),
            dst_access: u64::from(dst.as_raw()),
            ..general_to_general(i)
        };
        let (shader, transfer) = (vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ);
        t.begin(cb(2));
        t.barrier(cb(2), sync(frame_depth, shader, transfer));
        t.copy(cb(2), frame_depth, ring_depth);
        t.barrier(cb(2), sync(frame_depth, transfer, shader));
        t.barrier(cb(2), sync(frame_mvec, shader, transfer));
        t.copy(cb(2), frame_mvec, ring_mvec);
        t.barrier(cb(2), sync(frame_mvec, transfer, shader));
        t.barrier(cb(2), sync(sampled_mvec, shader, transfer));
        t.copy(cb(2), frame_exposure, ring_exposure);
        crate::ngx_ring::inject(vec![sr_record(2, [0x100, 0x800, 0x500, 0, 0xb00])]);
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1500, 0x1800, 0x1b00])));
        // A copy into the ring after the launch (the next frame's, recorded ahead) is not this launch's.
        t.copy(cb(2), sampled_mvec, ring_mvec);
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        let inputs = scan.inputs.expect("identified");
        assert_eq!((inputs.depth.0, inputs.mvec.0), (frame_depth, frame_mvec), "the copies' sources, not DLSS's rings");
        assert_eq!(inputs.exposure_input, None, "an exposure image the buffer copies into: measured, neither the ring nor the copy's source");
        assert_eq!((scan.depth_layout, scan.mvec_layout), (Some(GENERAL), Some(GENERAL)));
        assert_eq!((scan.depth_hazard, scan.hazard), (None, None));
        // Without the copies the record's own are read, whatever else the buffer reads (a copy from the frame's motion
        // vectors into another image among them).
        t.begin(cb(2));
        t.barrier(cb(2), sync(frame_depth, shader, transfer));
        t.barrier(cb(2), sync(frame_mvec, shader, transfer));
        t.copy(cb(2), frame_mvec, sampled_mvec);
        crate::ngx_ring::inject(vec![sr_record(3, [0x100, 0x800, 0x500, 0, 0xb00])]);
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1500, 0x1800, 0x1b00])));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        let inputs = scan.inputs.expect("identified");
        assert_eq!((inputs.depth.0, inputs.mvec.0), (ring_depth, ring_mvec));
        assert_eq!(inputs.exposure_input.map(|e| e.0), Some(ring_exposure), "nothing copied into it, the one exposure named so far");
        // An evaluate on the same colour input naming another exposure image: they rotate, and none holds an exposure
        // in front of DLSS's buffer, from now on.
        for (seq, exposure) in [(4, 0xa00), (5, 0xb00)] {
            t.begin(cb(2));
            crate::ngx_ring::inject(vec![sr_record(seq, [0x100, 0x800, 0x500, 0, exposure])]);
            t.launch(cb(2), Some(&param_block(&[0x1100, 0x1500, 0x1800, exposure + 0x1000])));
            let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
            assert_eq!(scan.inputs.expect("identified").exposure_input, None, "exposure {exposure:#x}: measured");
        }
    }

    /// The depth NGX's record names wins over another depth image the buffer only reads before
    /// the launch (a history depth some pass samples): the launch's own is final at the hold. The
    /// read one is taken only where the named one is written in the buffer first (GTA V's ring, the
    /// test above).
    #[test]
    fn the_launchs_own_depth_wins_over_one_only_read_before_it() {
        let (mut t, _) = held_tracker();
        let info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1707, height: 960, depth: 1 },
            format: vk::Format::D32_SFLOAT_S8_UINT,
            usage: vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let (own, history) = (vk::Image::from_raw(0x400), vk::Image::from_raw(0x500));
        for (image, raw) in [(own, 0x400), (history, 0x500)] {
            t.record_image(image, &info);
            t.record_view(vk::ImageView::from_raw(raw + 1), image);
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        let _ = t.refresh();
        t.barrier(cb(1), ImageSync::to(own, GENERAL));
        t.barrier(cb(1), ImageSync::to(history, GENERAL));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        let read = |image| ImageSync {
            src_access: u64::from(vk::AccessFlags::SHADER_READ.as_raw()),
            dst_access: u64::from(vk::AccessFlags::TRANSFER_READ.as_raw()),
            ..general_to_general(image)
        };
        t.begin(cb(2));
        t.barrier(cb(2), read(history));
        crate::ngx_ring::inject(vec![sr_record(2, [0x100, 0x300, 0x400, 0, 0])]);
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1400, 0x1300])));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.inputs.map(|i| i.depth.0), Some(own), "the launch's own depth, not the one read before it");
        assert_eq!((scan.depth_layout, scan.depth_hazard), (Some(GENERAL), None));
    }

    /// A barrier on one aspect of a depth/stencil image (`separateDepthStencilLayouts`) leaves its
    /// aspects possibly apart: the hold neither copies nor transitions it from the tracked layout
    /// until a barrier on both says where they are. A depth-only format's depth barrier is the whole.
    #[test]
    fn a_barrier_on_one_aspect_of_depth_and_stencil_leaves_no_layout_to_copy_from() {
        let (mut t, params) = held_tracker();
        let depth = vk::Image::from_raw(0x200);
        let read_only = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        let aspect = |a: vk::ImageAspectFlags| ImageSync { aspect: a.as_raw(), ..ImageSync::to(depth, read_only) };
        t.barrier(cb(1), aspect(vk::ImageAspectFlags::DEPTH));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        t.begin(cb(2));
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.depth_layout, Some(ASPECTS_APART));
        assert_eq!(aux_copy_layout(scan.depth_layout), None, "no copy or transition from it");
        assert!(!depth_copy_layout(vk::Format::D32_SFLOAT_S8_UINT, ASPECTS_APART));
        t.barrier(cb(3), aspect(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL));
        let _ = t.submit_ok(&[vec![cb(3)]]);
        t.begin(cb(2));
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!(scan.depth_layout, Some(read_only), "both aspects together again");
        // The separate depth layouts are only taken for a format without stencil.
        assert!(!depth_copy_layout(vk::Format::D32_SFLOAT_S8_UINT, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL));
        assert!(depth_copy_layout(vk::Format::D32_SFLOAT, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL));
    }

    /// Stellar Blade (Unreal Engine 4 under vkd3d-proton, #51): `GENERAL -> GENERAL` barriers on the depth aspect
    /// alone of DLSS's D32_SFLOAT_S8_UINT depth, in a buffer in front of DLSS's in the same submission. They move
    /// neither aspect. Taken as leaving the aspects apart, the hold inside DLSS's buffer read the depth as GENERAL
    /// (what the last submission left) and its guard found it apart when the buffer ran, at every submit: every
    /// hold dropped the network's history. A transition of one aspect still leaves them apart, and a barrier that
    /// keeps its layout after it does not bring them together.
    #[test]
    fn a_same_layout_barrier_on_one_aspect_of_the_depth_moves_no_layout() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let depth = vk::Image::from_raw(0x200);
        let one = |aspect: vk::ImageAspectFlags, old: vk::ImageLayout, new: vk::ImageLayout| ImageSync {
            old_layout: old,
            aspect: aspect.as_raw(),
            src_access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
            dst_access: vk::AccessFlags2::SHADER_READ.as_raw(),
            ..ImageSync::to(depth, new)
        };
        t.barrier(cb(1), general_to_general(colour()));
        t.barrier(cb(1), general_to_general(depth));
        let _ = t.submit_ok(&[vec![cb(1)]]);
        // The buffer in front: the game's barriers on the depth aspect alone. DLSS's buffer: the hold inside.
        t.barrier(cb(2), one(vk::ImageAspectFlags::DEPTH, GENERAL, GENERAL));
        t.barrier(cb(3), one(vk::ImageAspectFlags::DEPTH, GENERAL, GENERAL));
        t.barrier(cb(3), general_to_general(colour()));
        t.launch(cb(3), Some(&params));
        let point = t.inline_point(cb(3), &record_on(colour())).expect("held inside the buffer");
        assert_eq!(point.hazard, Hazard::SameLayoutBarrier, "the split refuses it");
        let read = point.depth_input.expect("the depth is read");
        assert_eq!((read.image, read.layout, read.readable), (depth, Some(GENERAL), true));
        assert_eq!(point.guessed.depth, Some(GENERAL), "not set by DLSS's buffer: checked at the submit");
        // The guard's check at the submit: the depth is as guessed when DLSS's buffer runs.
        assert_eq!(t.layout_when_run(&[cb(2)], depth), Some(GENERAL));
        let scan = t.submit_ok(&[vec![cb(2), cb(3)]]).expect("launch-bearing");
        assert_eq!(scan.depth_layout, Some(GENERAL));
        assert_eq!(t.layout_when_run(&[], depth), Some(GENERAL), "and after the submission");
        // A transition of one aspect: apart, and a same-layout barrier on one aspect after it keeps them apart.
        let read_only = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        t.begin(cb(4));
        t.barrier(cb(4), one(vk::ImageAspectFlags::DEPTH, GENERAL, read_only));
        assert_eq!(t.layout_when_run(&[cb(4)], depth), Some(ASPECTS_APART));
        t.begin(cb(5));
        t.barrier(cb(5), one(vk::ImageAspectFlags::STENCIL, GENERAL, GENERAL));
        assert_eq!(t.layout_when_run(&[cb(4), cb(5)], depth), Some(ASPECTS_APART));
        let _ = t.submit_ok(&[vec![cb(4), cb(5)]]);
        assert_eq!(t.layout_when_run(&[], depth), Some(ASPECTS_APART));
        // Both aspects together again.
        t.begin(cb(6));
        t.barrier(cb(6), ImageSync { old_layout: ASPECTS_APART, ..ImageSync::to(depth, GENERAL) });
        assert_eq!(t.layout_when_run(&[cb(6)], depth), Some(GENERAL));
    }

    /// A same-layout barrier on one aspect of the depth that contradicts the layout tracked there (a render pass's
    /// final layout the tracking did not follow): only that aspect is known to be in it, so the aspects may be apart
    /// and the depth is not read; kept as tracked, the hold copied the depth in the wrong layout and its guard compared
    /// against the same wrong state. Where no layout is tracked, none is set.
    #[test]
    fn a_one_aspect_barrier_contradicting_the_tracked_layout_leaves_the_aspects_apart() {
        let copyable = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let (mut t, params) = held_tracker_with(copyable);
        let depth = vk::Image::from_raw(0x200);
        let one = |old: vk::ImageLayout, new: vk::ImageLayout| ImageSync {
            old_layout: old,
            aspect: vk::ImageAspectFlags::DEPTH.as_raw(),
            src_access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
            dst_access: vk::AccessFlags2::SHADER_READ.as_raw(),
            ..ImageSync::to(depth, new)
        };
        // Nothing tracked on the depth: a one-aspect barrier sets nothing.
        t.barrier(cb(1), one(GENERAL, GENERAL));
        assert_eq!(t.layout_when_run(&[cb(1)], depth), None);
        let _ = t.submit_ok(&[vec![cb(1)]]);
        assert_eq!(t.layout_when_run(&[], depth), None);
        // Tracked as the attachment layout; the game's one-aspect barrier says GENERAL.
        let attachment = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        t.barrier(cb(2), general_to_general(colour()));
        t.barrier(cb(2), ImageSync::to(depth, attachment));
        let _ = t.submit_ok(&[vec![cb(2)]]);
        assert_eq!(t.layout_when_run(&[], depth), Some(attachment));
        t.barrier(cb(3), one(GENERAL, GENERAL));
        assert_eq!(t.layout_when_run(&[cb(3)], depth), Some(ASPECTS_APART), "contradicted: apart");
        // The same barrier in DLSS's buffer before its launch: the hold there does not read the depth as tracked.
        t.barrier(cb(4), one(GENERAL, GENERAL));
        t.barrier(cb(4), general_to_general(colour()));
        t.launch(cb(4), Some(&params));
        let point = t.inline_point(cb(4), &record_on(colour())).expect("held inside the buffer");
        assert!(point.depth_input.is_none_or(|d| !d.readable), "the depth is not read: {:?}", point.depth_input);
        let _ = t.submit_ok(&[vec![cb(3), cb(4)]]);
        assert_eq!(t.layout_when_run(&[], depth), Some(ASPECTS_APART));
        // A barrier on both aspects says where they are again; then one that agrees with it keeps it.
        t.begin(cb(5));
        t.barrier(cb(5), ImageSync { old_layout: ASPECTS_APART, ..ImageSync::to(depth, GENERAL) });
        t.barrier(cb(5), one(GENERAL, GENERAL));
        assert_eq!(t.layout_when_run(&[cb(5)], depth), Some(GENERAL));
    }

    /// Depth and motion vectors are read only by a dump: a barrier on them before the launch stops
    /// a dump, never a hold. This is GTA V Enhanced's launch buffer with DLSS Frame Generation on
    /// (probe `v201-fg-probe-1`): a dispatch, then depth and motion vectors copied for frame
    /// generation between `GENERAL -> GENERAL` barriers (read -> transfer read, transfer write ->
    /// shader read), then SR's input launch. Nothing touches the colour or exposure input.
    #[test]
    fn barriers_on_depth_and_motion_vectors_stop_only_a_dump() {
        let (mut t, params) = held_tracker();
        let (depth, mvec) = (vk::Image::from_raw(0x200), vk::Image::from_raw(0x300));
        let sync = |image, src_access: vk::AccessFlags, dst_access: vk::AccessFlags| ImageSync {
            src_access: u64::from(src_access.as_raw()),
            dst_access: u64::from(dst_access.as_raw()),
            ..general_to_general(image)
        };
        t.barrier(cb(2), sync(mvec, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ));
        t.barrier(cb(2), sync(mvec, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ));
        t.barrier(cb(2), sync(depth, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ));
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.hazard, scan.dump_hazard), (None, Some(Hazard::SameLayoutBarrier)));
        // The model-mode hold still runs; it only leaves the depth out.
        assert_eq!(scan.depth_hazard, Some(Hazard::SameLayoutBarrier));
        assert!(scan.colour_in_general());
        // Barriers on the motion vectors alone: the depth is copied.
        t.begin(cb(2));
        t.barrier(cb(2), sync(mvec, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ));
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.hazard, scan.dump_hazard, scan.depth_hazard), (None, Some(Hazard::SameLayoutBarrier), None));
        // A barrier on the colour input as well: no hold either.
        t.begin(cb(2));
        t.barrier(cb(2), sync(depth, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ));
        t.barrier(cb(2), general_to_general(colour()));
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.hazard, scan.dump_hazard), (Some(Hazard::SameLayoutBarrier), Some(Hazard::SameLayoutBarrier)));
        // A transition of depth: a dump's hazard only, named as a transition.
        t.begin(cb(2));
        t.barrier(cb(2), ImageSync { old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, ..ImageSync::to(depth, GENERAL) });
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.hazard, scan.dump_hazard), (None, Some(Hazard::LayoutTransition)));
    }

    /// The order holds across `vkCmdExecuteCommands`: a barrier the primary recorded before
    /// executing the secondary that launches is before the launch; one after it is not.
    #[test]
    fn synchronization_before_an_executed_secondarys_launch_counts() {
        let (mut t, params) = held_tracker();
        t.launch(cb(5), Some(&params));
        t.barrier(cb(2), general_to_general(colour()));
        t.execute(cb(2), &[cb(5)]);
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, Some(Hazard::SameLayoutBarrier));
        t.begin(cb(2));
        t.execute(cb(2), &[cb(5)]);
        t.barrier(cb(2), general_to_general(colour()));
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, None);
        // The barrier in one secondary, the launch in the next.
        t.begin(cb(2));
        t.global(cb(6), Hazard::EventWait);
        t.execute(cb(2), &[cb(6), cb(5)]);
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, Some(Hazard::EventWait));
        t.begin(cb(2));
        t.execute(cb(2), &[cb(5), cb(6)]);
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("launch-bearing").hazard, None);
    }

    /// A submitted ownership transfer of the colour input is remembered as its owner, for the
    /// caller to compare with the family of the queue the hold would run on.
    #[test]
    fn the_colour_inputs_owning_queue_family_follows_submitted_transfers() {
        let (mut t, params) = held_tracker();
        t.launch(cb(2), Some(&params));
        t.barrier(cb(1), ImageSync { src_queue_family: 0, dst_queue_family: 3, ..general_to_general(colour()) });
        // In the same submit, ahead of the launch buffer: known to the scan before it is committed.
        let scan = t.scan(&[vec![cb(1)], vec![cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.colour_owner, scan.hazard), (Some(3), None));
        assert!(t.owners.is_empty(), "nothing is committed by a scan");
        t.commit_submitted([cb(1), cb(2)]);
        assert_eq!(t.scan(&[vec![cb(2)]]).and_then(|s| s.colour_owner), Some(3));
        t.forget_image(colour());
        assert!(t.owners.is_empty());
    }

    /// Dynamic rendering suspended in the buffer before the launch buffer and resumed in it: the
    /// split would put the layer's submits between the two.
    #[test]
    fn a_split_between_suspended_and_resumed_rendering_is_refused() {
        let (suspending, resuming) = (vk::RenderingFlags::SUSPENDING, vk::RenderingFlags::RESUMING);
        let (mut t, params) = held_tracker();
        t.begin_rendering(cb(1), suspending);
        t.begin_rendering(cb(2), resuming);
        t.launch(cb(2), Some(&params));
        let scan = t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing");
        assert_eq!((scan.index, scan.hazard), (1, Some(Hazard::SuspendedRendering)));
        // Suspended and resumed inside the launch buffer: nothing crosses the split.
        t.begin(cb(2));
        t.begin_rendering(cb(2), suspending);
        t.begin_rendering(cb(2), resuming);
        t.launch(cb(2), Some(&params));
        assert_eq!(t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing").hazard, None);
        // The launch buffer suspends and the next resumes: both are in the tail, together.
        t.begin(cb(2));
        t.launch(cb(2), Some(&params));
        t.begin_rendering(cb(2), suspending);
        t.begin_rendering(cb(3), resuming);
        assert_eq!(t.submit_ok(&[vec![cb(2), cb(3)]]).expect("launch-bearing").hazard, None);
        // Through secondaries: the resuming instance is in an executed secondary.
        t.begin(cb(2));
        t.begin_rendering(cb(7), resuming);
        t.execute(cb(2), &[cb(7)]);
        t.launch(cb(2), Some(&params));
        assert_eq!(t.submit_ok(&[vec![cb(1), cb(2)]]).expect("launch-bearing").hazard, Some(Hazard::SuspendedRendering));
        // The wrapper takes the lock for flagged instances only, and stays armed for them.
        let tracking = Tracking::default();
        tracking.begin_rendering(cb(1), vk::RenderingFlags::empty());
        assert!(!tracking.armed());
        tracking.begin_rendering(cb(1), suspending);
        assert!(tracking.armed());
        tracking.begin(cb(1));
        assert!(!tracking.armed());
    }

    /// Tracked layouts advance only from command buffers the next layer accepted: a scan commits
    /// nothing, a refused head leaves everything as it was, and after a refused tail only the
    /// head's buffers count.
    #[test]
    fn failed_submissions_do_not_advance_the_tracked_layouts() {
        let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let transfer = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let fixture = || {
            let (mut t, params) = held_tracker();
            // Head: cb 1 leaves the colour input in GENERAL. Tail: the launch buffer, then cb 3,
            // which moves it out of GENERAL after DLSS.
            t.barrier(cb(1), ImageSync::to(colour(), GENERAL));
            t.launch(cb(2), Some(&params));
            t.barrier(cb(2), ImageSync::to(colour(), read_only));
            t.barrier(cb(3), ImageSync::to(colour(), transfer));
            t
        };
        let batches = vec![batch1(&[], &[1, 2, 3], &[])];
        let cbs = vec![vec![cb(1), cb(2), cb(3)]];
        // What the hold reads comes from the buffers ahead of the launch buffer, uncommitted.
        let mut t = fixture();
        let scan = t.scan(&cbs).expect("launch-bearing");
        assert_eq!((scan.index, scan.colour_layout), (1, Some(GENERAL)));
        assert!(t.committed.is_empty(), "a scan commits nothing");
        // Submits `plan`'s parts, failing the calls numbered in `fail`; returns the result.
        let run = |t: &mut Tracker, fail: &[usize]| {
            let scan = t.scan(&cbs).expect("launch-bearing");
            let calls = std::cell::Cell::new(0);
            let submit = |_: &[Batch1], _: vk::Fence| {
                calls.set(calls.get() + 1);
                if fail.contains(&calls.get()) {
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                } else {
                    vk::Result::SUCCESS
                }
            };
            let mut accepted = Vec::new();
            let result = submit_around(plan(batches.clone(), scan.batch, scan.index), vk::Fence::null(), &submit, |_| false, &mut |part| accepted.extend(part));
            t.commit_submitted(accepted);
            result
        };
        // The head is refused: nothing executed, nothing tracked.
        let mut t = fixture();
        assert_eq!(run(&mut t, &[1]), vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(t.committed.is_empty());
        // The head is accepted, the tail refused twice (the second time is its retry): the head's
        // barrier only, and the call reported lost (its head cannot be taken back).
        let mut t = fixture();
        assert_eq!(run(&mut t, &[2, 3]), vk::Result::ERROR_DEVICE_LOST);
        assert_eq!(t.committed.get(&colour()), Some(&GENERAL), "the launch buffer and what follows never ran");
        // The tail refused once: accepted when submitted again, the last buffer's layout.
        let mut t = fixture();
        assert_eq!(run(&mut t, &[2]), vk::Result::SUCCESS);
        assert_eq!(t.committed.get(&colour()), Some(&transfer));
        // Both accepted: the last buffer's layout.
        let mut t = fixture();
        assert_eq!(run(&mut t, &[]), vk::Result::SUCCESS);
        assert_eq!(t.committed.get(&colour()), Some(&transfer));
        // An unsplit call that fails is never committed by its caller: the state stays put, and
        // the next scan still reads the last accepted layout.
        let mut t = fixture();
        assert!(t.scan(&cbs).is_some());
        assert!(t.committed.is_empty());
        t.commit_submitted([cb(1)]);
        assert_eq!(t.scan(&[vec![cb(2)]]).and_then(|s| s.colour_layout), Some(GENERAL));
    }

    /// A parameter buffer in vkd3d-proton's layout: some scalars, then the given 64-bit handles.
    fn param_block(handles: &[u64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1485u32.to_le_bytes());
        bytes.extend_from_slice(&836u32.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for h in handles {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        bytes
    }

    /// GTA V's registered set with DLSS Frame Generation on: SR's colour input, depth and motion
    /// vectors, the output, and FG's own images (an output-size RGBA16F storage image among them).
    /// Returns the tracker and the handles: (colour, sr output, fg colour, depth, mvec).
    fn fg_tracker() -> (Tracker, [u64; 5]) {
        let mut t = Tracker::default();
        let info = |w, h, format| vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: w, height: h, depth: 1 },
            format,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let set = [
            (0x100, 1485, 836, rgba, 0x14_0000_195f_u64),
            (0x200, 2560, 1440, rgba, 0x15_0000_43cd),
            (0x300, 2560, 1440, rgba, 0x15_0000_5000),
            (0x400, 1485, 836, vk::Format::D32_SFLOAT_S8_UINT, 0x14_0000_1962),
            (0x500, 1485, 836, vk::Format::R16G16_SFLOAT, 0x14_0000_215e),
        ];
        for (raw, w, h, format, handle) in set {
            t.record_image(vk::Image::from_raw(raw), &info(w, h, format));
            t.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
            t.register(vk::ImageView::from_raw(raw + 1), Some(handle));
        }
        crate::ngx_ring::inject(vec![sr_record(1, [0x100, 0x500, 0x400, 0x200, 0])]);
        assert!(t.refresh().unwrap().starts_with("colour input: image 0x100 (1485x836"));
        (t, set.map(|s| s.4))
    }

    /// The probe's layout line names each parameter word: views with their role (whole words and
    /// Unreal Engine 5's two packed halves), then integers and floats.
    #[test]
    fn probe_layout_names_each_word() {
        let (mut t, [colour, sr_out, _, depth, mvec]) = fg_tracker();
        let c = Some(vk::Image::from_raw(0x100));
        assert_eq!(t.describe_word(colour, c), "view:colour-in(1485x836,R16G16B16A16_SFLOAT)");
        assert_eq!(t.describe_word(depth, c), "view:depth(1485x836,D32_SFLOAT_S8_UINT)");
        assert_eq!(t.describe_word(mvec, c), "view:mvec(1485x836,R16G16_SFLOAT)");
        assert_eq!(t.describe_word(sr_out, c), "view:image(2560x1440,R16G16B16A16_SFLOAT)");
        assert_eq!(t.describe_word(colour, None), "view:image(1485x836,R16G16B16A16_SFLOAT)");
        t.record_view(vk::ImageView::from_raw(0x701), vk::Image::from_raw(0x400));
        t.register(vk::ImageView::from_raw(0x701), Some(0x0140_1c00));
        t.record_view(vk::ImageView::from_raw(0x702), vk::Image::from_raw(0x500));
        t.register(vk::ImageView::from_raw(0x702), Some(0x0320_1c23));
        assert_eq!(t.describe_word(0x0320_1c23_0140_1c00, c), "[view:depth(1485x836,D32_SFLOAT_S8_UINT)|view:mvec(1485x836,R16G16_SFLOAT)]");
        let words: Vec<String> = param_block(&[]).as_chunks::<8>().0.iter().map(|w| t.describe_word(u64::from_le_bytes(*w), c)).collect();
        assert_eq!(words, ["i(1485,836)", "f(0.5,0)"]);
        assert_eq!(t.describe_word(0, c), "0");
        assert_eq!(t.describe_word(0x7f12_3456_789a_bcde, c), "0x7f123456789abcde");
    }

    /// DLSS Frame Generation's launch-bearing buffers (they read depth, motion vectors and the output-size frame, never
    /// the render-size colour input) carry no Super Resolution evaluate and are forwarded; only Super Resolution's
    /// buffer, whose launch names the colour input of NGX's record, is the hold point, wherever the two sit in a submit
    /// or across submits.
    #[test]
    fn only_the_buffer_whose_launches_name_the_colour_input_is_held() {
        let (mut t, [colour, sr_out, fg_colour, depth, mvec]) = fg_tracker();
        // SR: the input kernel reads colour, depth and motion vectors; later kernels only scratch.
        t.launch(cb(1), Some(&param_block(&[colour, depth, mvec])));
        t.launch(cb(1), Some(&param_block(&[sr_out])));
        // FG: two buffers (GTA's 80- and 36-launch ones), depth, motion vectors and the output only.
        t.launch(cb(2), Some(&param_block(&[fg_colour, depth, mvec])));
        t.launch(cb(2), Some(&param_block(&[sr_out])));
        t.launch(cb(3), Some(&param_block(&[fg_colour, sr_out])));
        assert!(t.launch[&cb(1)].sr().is_some());
        assert!(t.launch[&cb(2)].ngx.is_none() && t.launch[&cb(3)].ngx.is_none());

        // FG's submits on their own (another queue in GTA V): not held, counted.
        assert!(t.submit_ok(&[vec![cb(2)]]).is_none(), "frame generation's buffer is not the hold point");
        assert!(t.submit_ok(&[vec![cb(3)]]).is_none());
        assert_eq!(t.foreign_submits, 2);
        // An FG buffer first in the same submit: the split point is SR's buffer behind it.
        let scan = t.submit_ok(&[vec![cb(2)], vec![cb(7), cb(1)]]).expect("SR's buffer is held");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!((t.evaluate_submits, t.evaluations, t.foreign_submits), (1, 1, 2));
        // SR's buffer through a secondary carries its references to the primary.
        t.begin(cb(1));
        t.launch(cb(11), Some(&param_block(&[colour])));
        t.execute(cb(1), &[cb(11)]);
        assert!(t.submit_ok(&[vec![cb(3), cb(1)]]).is_some_and(|s| s.index == 1));
        assert_eq!(t.evaluate_submits, 2);
    }

    #[test]
    fn launch_params_reads_cudas_extra_buffer_form_only() {
        let block = param_block(&[0x14_0000_195f]);
        let size = block.len() as u32;
        let size64 = block.len() as u64;
        let ptr = |v: usize| v as *const c_void;
        // vkd3d-proton's list: extraCount 1, END-terminated; size as u32 and as size_t.
        for size_ptr in [std::ptr::from_ref(&size).cast::<c_void>(), std::ptr::from_ref(&size64).cast::<c_void>()] {
            let extras = [ptr(1), block.as_ptr().cast(), ptr(2), size_ptr, ptr(0)];
            let got = unsafe { launch_params(extras.as_ptr(), 1) }.expect("readable");
            assert_eq!(got, &block[..]);
        }
        // Size first, then buffer.
        let extras = [ptr(2), std::ptr::from_ref(&size).cast(), ptr(1), block.as_ptr().cast(), ptr(0)];
        assert_eq!(unsafe { launch_params(extras.as_ptr(), 5) }, Some(&block[..]));
        // Refused: no extras, an unknown key, a missing size or buffer, zero or huge sizes.
        assert_eq!(unsafe { launch_params(std::ptr::null(), 1) }, None);
        assert_eq!(unsafe { launch_params(extras.as_ptr(), 0) }, None);
        let unknown = [ptr(3), ptr(0), ptr(0)];
        assert_eq!(unsafe { launch_params(unknown.as_ptr(), 1) }, None);
        let no_size = [ptr(1), block.as_ptr().cast(), ptr(0)];
        assert_eq!(unsafe { launch_params(no_size.as_ptr(), 1) }, None);
        let zero = 0u32;
        let zero_size = [ptr(1), block.as_ptr().cast(), ptr(2), std::ptr::from_ref(&zero).cast(), ptr(0)];
        assert_eq!(unsafe { launch_params(zero_size.as_ptr(), 1) }, None);
        let huge = 1u32 << 20;
        let huge_size = [ptr(1), block.as_ptr().cast(), ptr(2), std::ptr::from_ref(&huge).cast(), ptr(0)];
        assert_eq!(unsafe { launch_params(huge_size.as_ptr(), 1) }, None);
        // No terminator within four pairs: refused, nothing past the ninth entry is read.
        let endless = [
            ptr(1),
            block.as_ptr().cast(),
            ptr(1),
            block.as_ptr().cast(),
            ptr(1),
            block.as_ptr().cast(),
            ptr(1),
            block.as_ptr().cast(),
            ptr(1),
            block.as_ptr().cast(),
        ];
        assert_eq!(unsafe { launch_params(endless.as_ptr(), 1) }, None);
        // An explicit count that ends between a key and its value: the value is not read (only three
        // entries exist), and the list is refused. Four entries without the terminator are the list.
        let cut = [ptr(2), std::ptr::from_ref(&size).cast(), ptr(1)];
        assert_eq!(unsafe { launch_params(cut.as_ptr(), 3) }, None);
        let whole = [ptr(2), std::ptr::from_ref(&size).cast(), ptr(1), block.as_ptr().cast()];
        assert_eq!(unsafe { launch_params(whole.as_ptr(), 4) }, Some(&block[..]));
    }

    /// Only write bits count as a write in a barrier's source access: the transform feedback
    /// counter's read bit is not one, shader storage writes (sync2, bit 34) and the video, optical flow
    /// and micromap writes are.
    #[test]
    fn a_barrier_after_reads_only_does_not_follow_a_write() {
        use vk::AccessFlags2 as A;
        let after = |access: A| ImageSync { src_access: access.as_raw(), ..general_to_general(vk::Image::from_raw(7)) };
        for read in [A::TRANSFORM_FEEDBACK_COUNTER_READ_EXT, A::SHADER_STORAGE_READ, A::VIDEO_ENCODE_READ_KHR, A::SHADER_READ | A::TRANSFER_READ] {
            assert!(!after(read).may_follow_a_write(), "{read:?}");
        }
        for write in [
            A::SHADER_STORAGE_WRITE,
            A::VIDEO_DECODE_WRITE_KHR,
            A::VIDEO_ENCODE_WRITE_KHR,
            A::OPTICAL_FLOW_WRITE_NV,
            A::MICROMAP_WRITE_EXT,
            A::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
            A::MEMORY_WRITE,
        ] {
            assert!(after(write).may_follow_a_write(), "{write:?}");
        }
    }

    // ---- Identification by the input kernel's parameters (`Tracker::observe`). ----

    fn img(raw: u64) -> vk::Image {
        vk::Image::from_raw(raw)
    }

    /// The kernel handle [`tracker_with`] registers for image `raw`.
    fn h(raw: u64) -> u64 {
        raw + 0x1000
    }

    fn set_of(list: &[(u64, ImageDesc)]) -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        list.iter().map(|&(raw, d)| (raw, (img(raw), d))).collect()
    }

    /// One launch-bearing submit as [`Tracking::scan`] runs it: observe, refresh, scan, classify.
    /// Returns the scan and the lines it would log.
    fn submit(t: &mut Tracker, batches: &[Vec<vk::CommandBuffer>]) -> (Option<Scan>, Vec<String>) {
        let mut lines = t.observe(batches);
        lines.extend(t.refresh());
        let scan = t.submit_ok(batches);
        lines.extend(t.classify_line(scan.as_ref()));
        (scan, lines)
    }

    /// DLSS's launches recorded into a secondary that is executed into the submitted primary: the
    /// primary carries the secondary's camera jitter, DLSS's output kernel and image, and the colour
    /// input's attachment layout, beside its input launch. Before, only the input launch carried
    /// over: the hold ran without jitter, DLSS's output was never learnt, and the layout the
    /// secondary's rendering left never reached the committed state.
    #[test]
    fn a_secondarys_jitter_output_and_attachment_layouts_carry_over_to_its_primary() {
        use crate::ngx_ring::{end_of, inject, parse_record, tests::record_bytes, FEATURE_SR};
        let mut t = tracker_with(&gta_registered());
        let rec = parse_record(&record_bytes(1, FEATURE_SR, 1, [0x100, 0x300, 0x200, 0x400, 0], [-0.25, 0.1667], [1.0; 2], 1.0, 1.0, 1));
        inject(vec![rec]);
        assert!(t.refresh().is_some(), "identified by the record");
        let (primary, secondary) = (cb(1), cb(11));
        t.begin(primary);
        t.begin(secondary);
        t.attachments(secondary, &[(vk::ImageView::from_raw(0x101), vk::ImageLayout::GENERAL)]);
        t.launch(secondary, Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(secondary, Some(&param_block(&[h(0x100), h(0x400)])));
        assert_eq!((t.launch[&secondary].ngx, t.launch[&secondary].output_writes), (Some(rec), 1));
        t.execute(primary, &[secondary]);
        let refs = &t.launch[&primary];
        assert_eq!((refs.ngx, refs.output_writes), (Some(rec), 1), "the secondary's evaluate and its write of the output");
        assert_eq!(t.attached.get(&primary), Some(&vec![(img(0x100), vk::ImageLayout::GENERAL)]), "the secondary's attachment layout");
        inject(vec![rec, end_of(&rec, 2)]);
        for n in 0..3 {
            let scan = submit(&mut t, &[vec![primary]]).0.expect("launch-bearing");
            assert_eq!(scan.frame, DlssFrame { jitter: Some([-0.25, 0.1667]), reset: true }, "submit {n}");
            assert_eq!(t.committed.get(&img(0x100)), Some(&vk::ImageLayout::GENERAL), "submit {n}: committed from the primary");
        }
        assert_eq!(t.dlss_output.map(|o| (o.image, o.input, o.writes)), Some((img(0x400), img(0x100), 1)), "DLSS's output is learnt from the primary");
    }

    /// The post path serves a held game's frames only where every pass runs at present: `present`; `auto` and
    /// `before` while Ray Reconstruction runs and no hold after DLSS is being recorded (3.7.0's hand-off);
    /// `upscaled` only when, without its hold, it would fall back to the post path too.
    #[test]
    fn the_post_path_serves_only_frames_whose_passes_all_run_at_present() {
        use neural_forge_protocol::enums::placement::*;
        let one = |p: u32| (1, [p; neural_forge_protocol::MAX_PASSES]);
        // (setting, rr, after recorded) -> serves
        for (setting, rr, after, serves) in [
            (AUTO, false, false, false),
            (AUTO, true, false, true),
            (AUTO, true, true, false),
            (BEFORE, false, false, false),
            (BEFORE, true, true, true),
            (UPSCALED, false, false, false),
            (UPSCALED, false, true, false),
            (UPSCALED, true, false, true),
            (UPSCALED, true, true, false),
            (PRESENT, false, false, true),
            (PRESENT, true, true, true),
        ] {
            assert_eq!(post_serves_for(one(setting), rr, after), serves, "{} rr {rr} after {after}", NAMES[setting as usize]);
        }
        // Pass 2 at present behind pass 1 before the upscaler: it moves into the holds, the post path stays off.
        let mut s = [BEFORE; neural_forge_protocol::MAX_PASSES];
        s[1] = PRESENT;
        assert!(!post_serves_for((2, s), false, false));
    }

    /// A hold after DLSS's output that is refused (its point or its recording) eight times in a row makes the
    /// output count as not learnable for [`AFTER_RETRY`], so the passes placed there go where `auto` runs them
    /// meanwhile; then it is tried again. A hold recorded starts the count over and ends a refusal.
    #[test]
    fn the_hold_state_says_where_the_holds_are() {
        use neural_forge_protocol::enums::preupscale_state::{HOLDING, HOLDING_AFTER, HOLDING_BOTH};
        assert_eq!(hold_state_of(true, false), HOLDING);
        assert_eq!(hold_state_of(false, true), HOLDING_AFTER);
        assert_eq!(hold_state_of(true, true), HOLDING_BOTH);
        assert_eq!(hold_state_of(false, false), HOLDING, "a hold the plan does not explain stays before the upscaler");
    }

    #[test]
    fn a_hold_after_dlss_that_keeps_missing_gives_its_passes_back_for_a_while() {
        let mut t = Tracker::default();
        let t0 = Instant::now();
        assert!(!t.frame_at(t0).output, "no output learnt");
        let o = DlssOutput { image: img(0x400), input: img(0x100), depth: img(0x200), mvec: img(0x300), rr: true, writes: 1 };
        t.dlss_output = Some(o);
        assert!(t.frame_at(t0).output);
        for _ in 0..AFTER_MISSES - 1 {
            t.after_missed_at(t0);
        }
        t.after_recorded();
        for _ in 0..AFTER_MISSES - 1 {
            t.after_missed_at(t0);
        }
        assert!(t.frame_at(t0).output, "a recorded hold started the count over");
        t.after_missed_at(t0);
        assert!(!t.frame_at(t0).output, "{AFTER_MISSES} misses in a row");
        assert!(!t.frame_at(t0 + AFTER_RETRY - Duration::from_millis(1)).output, "until the retry");
        assert!(t.frame_at(t0 + AFTER_RETRY).output, "tried again after {AFTER_RETRY:?}");
        t.dlss_output = Some(DlssOutput { image: img(0x410), ..o });
        assert!(t.frame_at(t0).output, "another output is tried at once");
        t.dlss_output = Some(o);
        t.after_recorded();
        assert!(t.frame_at(t0).output, "a hold recorded ends the refusal");
    }

    #[test]
    fn capture_regions_pad_odd_sizes_with_the_edge() {
        assert_eq!(padded(1707, 960), (1708, 960));
        assert_eq!(padded(2227, 1253), (2228, 1254));
        assert_eq!(padded(1490, 838), (1490, 838));
        assert_eq!(capture_regions(16, 8).len(), 1, "even: one plain copy");
        let r = capture_regions(17, 9);
        assert_eq!(r.len(), 4);
        assert_eq!(r[1].buffer_offset, 17 * TEXEL, "the last column lands in column 17 of an 18-wide row");
        assert_eq!((r[1].image_offset.x, r[1].image_extent.width, r[1].image_extent.height), (16, 1, 9));
        assert_eq!(r[2].buffer_offset, 9 * 18 * TEXEL, "the last row lands in row 9");
        assert_eq!(r[3].buffer_offset, (9 * 18 + 17) * TEXEL);
        assert!(r.iter().all(|r| r.buffer_row_length == 18 && r.buffer_offset.is_multiple_of(TEXEL)));
        let w = writeback_region(17, 9);
        assert_eq!((w.buffer_row_length, w.image_extent.width, w.image_extent.height), (18, 17, 9), "the write-back crops the padding");
    }

    #[test]
    fn half_floats_and_the_preview_curve() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
        // One white (1.0) texel and one black, in a 2-texel padded row.
        let mut colour = Vec::new();
        for v in [0x3c00u16, 0x3c00, 0x3c00, 0x3c00, 0, 0, 0, 0x3c00] {
            colour.extend_from_slice(&v.to_le_bytes());
        }
        let png = preview_rgba8(&colour, 2, 1, 1);
        // 1/(1+1) = 0.5 linear -> 188 sRGB.
        assert_eq!(png, vec![188, 188, 188, 255]);
    }

    #[test]
    fn a_dump_writes_raw_files_a_sidecar_and_a_preview() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join(format!("target/test-scratch/preupscale-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (w, h) = (3u32, 1u32);
        let colour = vec![0u8; (4 * 2 * TEXEL) as usize];
        let mut frame = DumpFrame {
            width: w,
            height: h,
            colour,
            depth: Some((vec![0u8; 12], vk::Format::D32_SFLOAT_S8_UINT)),
            mvec: Some(vec![0u8; 12]),
            mvec_scale: [0.5, 0.75],
            frame: 42,
            exposure: Vec::new(),
        };
        let files = frame.write(&dir).unwrap();
        assert_eq!(files, vec!["colour.rgba16f", "depth.r32f", "mvec.rg16f", "colour-preview.png", "meta.json"], "no exposure.json without candidates");
        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        for needle in [
            "\"width\": 3",
            "\"height\": 1",
            "\"padded_width\": 4",
            "\"padded_height\": 2",
            "\"frame\": 42",
            "D32_SFLOAT_S8_UINT",
            "\"mvec_scale\": [0.5, 0.75]",
        ] {
            assert!(meta.contains(needle), "{needle} in {meta}");
        }
        let rgba32: Vec<u8> = [0.5f32, 2.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        frame.exposure = vec![
            ExposureValue { image: 0x600, format: vk::Format::R32G32B32A32_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), assumed: true, bytes: Some(rgba32) },
            ExposureValue {
                image: 0x610,
                format: vk::Format::R16_SFLOAT,
                layout: Some(vk::ImageLayout::GENERAL),
                assumed: false,
                bytes: Some(vec![0x00, 0x3c]),
            },
            ExposureValue { image: 0x620, format: vk::Format::R16_SFLOAT, layout: None, assumed: false, bytes: None },
        ];
        let files = frame.write(&dir).unwrap();
        assert_eq!(files.last().map(String::as_str), Some("exposure.json"));
        assert_eq!(std::fs::read_to_string(dir.join("meta.json")).unwrap(), meta, "meta.json does not change");
        let json = std::fs::read_to_string(dir.join("exposure.json")).unwrap();
        for needle in [
            "\"image\": \"0x600\", \"format\": \"R32G32B32A32_SFLOAT\", \"layout\": \"GENERAL\", \"layout_assumed\": true, \"raw_le_hex\": \"0000003f00000040000000000000803f\"",
            "\"values\": [5e-1, 2e0, 0e0, 1e0]",
            "\"raw_le_hex\": \"003c\", \"values\": [1e0]",
            "\"image\": \"0x620\", \"format\": \"R16_SFLOAT\", \"layout\": null, \"layout_assumed\": false, \"raw_le_hex\": null, \"values\": null",
        ] {
            assert!(json.contains(needle), "{needle} in {json}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- GPU: the hold machinery on a real (lavapipe or local) device. ----

    struct Gpu {
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        device: ash::Device,
        queue: vk::Queue,
        family: u32,
        pool: vk::CommandPool,
        _entry: ash::Entry,
    }

    impl Gpu {
        /// Whether a 2D optimal-tiling `format` image with `usage` can be created here.
        fn supports(&self, format: vk::Format, usage: vk::ImageUsageFlags) -> bool {
            // SAFETY: a plain property query on this instance's physical device.
            unsafe {
                self.instance.get_physical_device_image_format_properties(
                    self.physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    vk::ImageCreateFlags::empty(),
                )
            }
            .is_ok()
        }

        fn open(import: bool) -> Option<Self> {
            let (entry, instance, physical, device, queue, family) =
                if import { crate::composition::gpu::test_device_with_external_memory_host()? } else { crate::composition::gpu::test_device()? };
            let info = vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = unsafe { device.create_command_pool(&info, None) }.unwrap();
            Some(Self { instance, physical, device, queue, family, pool, _entry: entry })
        }

        fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.device;
            unsafe {
                let alloc = vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                let cmd = d.allocate_command_buffers(&alloc).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.queue).unwrap();
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        fn image(&self, width: u32, height: u32) -> (vk::Image, vk::DeviceMemory) {
            let d = &self.device;
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = unsafe { d.create_image(&info, None) }.unwrap();
            let reqs = unsafe { d.get_image_memory_requirements(image) };
            let props = unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
            let index = (0..props.memory_type_count)
                .find(|&i| {
                    reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .unwrap();
            let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None) }.unwrap();
            unsafe { d.bind_image_memory(image, memory, 0) }.unwrap();
            (image, memory)
        }

        fn barrier(cmd: vk::CommandBuffer, d: &ash::Device, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[b],
                )
            };
        }

        /// Uploads `bytes` (tightly packed RGBA16F) and leaves the image in GENERAL, like DLSS's input.
        fn upload(&self, image: vk::Image, width: u32, height: u32, bytes: &[u8]) {
            let staging = own_host_buffer(&self.device, &self.instance, self.physical, bytes.len() as u64).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr, bytes.len()) };
            let d = &self.device;
            self.one_shot(|cmd| {
                Self::barrier(cmd, d, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                let region =
                    vk::BufferImageCopy { image_subresource: colour_layers(), image_extent: vk::Extent3D { width, height, depth: 1 }, ..Default::default() };
                unsafe { d.cmd_copy_buffer_to_image(cmd, staging.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]) };
                Self::barrier(cmd, d, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
            });
            unsafe { staging.destroy(d) };
        }

        fn read(&self, image: vk::Image, width: u32, height: u32) -> Vec<u8> {
            let n = (width * height) as usize * TEXEL as usize;
            let staging = own_host_buffer(&self.device, &self.instance, self.physical, n as u64).unwrap();
            let d = &self.device;
            self.one_shot(|cmd| {
                let open = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                unsafe {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[open],
                        &[],
                        &[],
                    )
                };
                let region =
                    vk::BufferImageCopy { image_subresource: colour_layers(), image_extent: vk::Extent3D { width, height, depth: 1 }, ..Default::default() };
                unsafe { d.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, staging.buffer, &[region]) };
                let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                unsafe {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[to_host],
                        &[],
                        &[],
                    )
                };
            });
            let out = unsafe { std::slice::from_raw_parts(staging.ptr, n) }.to_vec();
            unsafe { staging.destroy(d) };
            out
        }
    }

    impl Drop for Gpu {
        fn drop(&mut self) {
            unsafe {
                self.device.device_wait_idle().unwrap();
                self.device.destroy_command_pool(self.pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    /// A distinct, finite half-float pattern per texel and channel.
    fn pattern(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for c in 0..4u32 {
                    out.extend_from_slice(&(0x3c00u16 + (x * 16 + y * 4 + c) as u16).to_le_bytes());
                }
            }
        }
        out
    }

    /// The resources of a hold recorded into DLSS's buffer are never timed, and create no timestamp
    /// pool to drop: a `GpuTimer` has no `Drop`, so one made and discarded leaks its `VkQueryPool`, which
    /// the validation layer reports when the device is destroyed. The split's resources are timed.
    #[test]
    fn inline_hold_resources_create_no_timer_pools() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let d = &gpu.device;
        // SAFETY: a live device; nothing of the resources is ever submitted.
        let inline = unsafe { Resources::build_inline(d, &gpu.instance, gpu.physical, gpu.family, 16, 8) }.expect("inline resources");
        assert!(inline.inline && inline.capture_timer.is_none() && inline.writeback_timer.is_none());
        let shm = scratch_shm("timers");
        // SAFETY: as above.
        let split = unsafe { Resources::build(d, &gpu.instance, gpu.physical, gpu.family, 16, 8, &shm, false) }.expect("split resources");
        let timestamps = unsafe { gpu.instance.get_physical_device_queue_family_properties(gpu.physical) }[gpu.family as usize].timestamp_valid_bits > 0;
        assert_eq!((split.capture_timer.is_some(), split.writeback_timer.is_some()), (timestamps, timestamps));
        // SAFETY: nothing submitted.
        unsafe {
            inline.destroy(d);
            split.destroy(d);
        }
    }

    /// Resources a failed submit left without a fence are let go of (leaked) on a live device, so the
    /// next hold builds new ones and the after-the-upscaler path does not wait on them for good.
    #[test]
    fn unfenced_hold_resources_are_rebuilt_on_a_live_device() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let d = &gpu.device;
        let shm = scratch_shm("unfenced");
        let mut session = Session::default();
        // SAFETY: a fresh device; nothing of the resources is ever submitted.
        assert!(unsafe { session.ensure(d, &gpu.instance, gpu.physical, gpu.family, 16, 8, &shm, false, placement::Stage::Before) });
        session.res.as_mut().unwrap().on_queue(gpu.queue);
        session.res.as_mut().unwrap().unfenced = true;
        assert!(session.res.as_ref().unwrap().pending());
        // SAFETY: as above.
        assert!(unsafe { session.ensure(d, &gpu.instance, gpu.physical, gpu.family, 16, 8, &shm, false, placement::Stage::Before) }, "built again");
        assert!(!session.res.as_ref().unwrap().pending(), "fresh resources");
        assert_eq!(session.left_behind.len(), 1, "the old ones set aside, not leaked");
        // The new ones' first fence on the same queue, seen signalled: the old ones' work is done.
        let res = session.res.as_mut().unwrap();
        res.on_queue(gpu.queue);
        // SAFETY: the resources' own idle fence, submitted once with nothing.
        unsafe { d.queue_submit(gpu.queue, &[], res.capture_fence) }.unwrap();
        res.capture_pending = true;
        session.bury(d);
        assert_eq!(session.left_behind.len(), 1, "no fence of the new ones seen yet");
        assert!(session.res.as_mut().unwrap().wait_idle(d, &mut None));
        session.bury(d);
        assert!(session.left_behind.is_empty(), "destroyed once a later fence on their queue signalled");
        // The split holds run before the upscaler only: no fenced resources are ever built into the
        // slot of the hold after DLSS's output, which `ensure_inline` destroys without a wait.
        // SAFETY: as above.
        assert!(!unsafe { session.ensure(d, &gpu.instance, gpu.physical, gpu.family, 32, 16, &shm, false, placement::Stage::Upscaled) });
        assert!(session.res_after.is_none() && session.res.as_ref().unwrap().matches(gpu.family, 16, 8), "nothing built, the split's left alone");
        // SAFETY: everything submitted was waited on.
        unsafe { session.destroy(d) };
    }

    /// A channel over a scratch file, which goes (with its directory) when the test is done with it.
    struct ScratchShm {
        shm: ShmClient,
        dir: std::path::PathBuf,
    }

    impl std::ops::Deref for ScratchShm {
        type Target = ShmClient;
        fn deref(&self) -> &ShmClient {
            &self.shm
        }
    }

    impl std::ops::DerefMut for ScratchShm {
        fn deref_mut(&mut self) -> &mut ShmClient {
            &mut self.shm
        }
    }

    impl Drop for ScratchShm {
        fn drop(&mut self) {
            // The mapping stays valid without its file.
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn scratch_shm(tag: &str) -> ScratchShm {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!(
            "target/test-scratch/preupscale-{}-{tag}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&dir.join("shm.bin").display().to_string()));
        ScratchShm { shm, dir }
    }

    // ---- The hand-back window. ----

    /// The post-upscaler hand-back: once a hold on a device has run the model, the post path
    /// stays off until DLSS has not run for HAND_BACK while Super Resolution still launches (DLAA, its input no
    /// longer identified), so a loading screen (no DLSS for a few to ~20 s) never hands back. A device that never
    /// got that far is never affected.
    #[test]
    fn the_post_path_takes_over_only_after_hand_back_without_dlss() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        // Super Resolution keeps launching after t0 without its input held.
        let launching = Some(SrLaunches { since: t0.checked_sub(Duration::from_secs(60)).unwrap_or(t0), last: s(40.0) });
        // Never held: the post path runs, DLSS or not.
        assert!(!post_off(false, Some(t0), launching, t0));
        assert!(!post_off(false, None, launching, t0));
        // Held, and the last frame was held at t0 (the game went to a loading screen).
        assert!(post_off(true, Some(t0), launching, s(0.6)), "0.6 s into a loading screen (the old 500 ms hand-back)");
        assert!(post_off(true, Some(t0), launching, s(20.0)), "a long loading screen");
        assert!(post_off(true, Some(t0), launching, s(HAND_BACK.as_secs_f32() - 0.01)));
        assert!(!post_off(true, Some(t0), launching, s(HAND_BACK.as_secs_f32() + 0.01)), "DLSS off for longer than HAND_BACK: the post path resumes");
        assert!(!post_off(true, None, launching, t0));
        // A clock read before the last held frame (another thread's) is not "long ago".
        assert!(post_off(true, Some(s(1.0)), launching, t0));

        // Through a Session: holds and DLSS submits refresh the window; skipped holds count too.
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = ShmClient::test_over_header(&header);
        let mut session = Session::default();
        session.saw_dlss();
        assert!(!post_off(session.engaged, session.last_hold, None, Instant::now()), "not before the first hold ran the network");
        let skipped = HoldResult { miss: Some("the native network is still loading"), waits_consumed: true, native: true, ..Default::default() };
        session.note(&shm, &skipped, Duration::from_millis(1), (64, 64));
        assert!(!session.engaged, "a hold that did not run the network does not engage the device");
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 1, "a held frame that went untouched is a miss");
        let held = HoldResult { waits_consumed: true, wrote_back: true, evaluated: true, native: true, ..Default::default() };
        session.note(&shm, &held, Duration::from_millis(10), (64, 64));
        assert!(session.engaged);
        assert_eq!(header.preupscale_state.load(Ordering::Relaxed), 2, "holding");
        let last = session.last_hold.expect("a held frame");
        assert!(post_off(session.engaged, session.last_hold, None, last + Duration::from_secs(10)));
    }

    /// A pause menu (#49): the game stops launching DLSS altogether. The post path stays off for PAUSE_HAND_BACK, not
    /// HAND_BACK: Cyberpunk 2077's menu shows the last frame (with the model's edit) under its menu, and handing it to
    /// the post path after 30 s ran the network over the menu at the output size and rebuilt it there and back. Super
    /// Resolution launching again (inputs not held: DLAA, a new set not yet identified) hands back after HAND_BACK as
    /// before; a launch within RECENT of the last frame is that frame's own and does not count.
    #[test]
    fn a_pause_without_any_dlss_launch_keeps_the_post_path_off_longer() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let run = |since: f32, last: f32| Some(SrLaunches { since: s(since), last: s(last) });
        // Nothing launched since the last held frame (the run that ended with it, or none at all): paused.
        for launch in [None, run(0.0, 0.0), run(0.0, 0.2)] {
            assert!(post_off(true, Some(t0), launch, s(HAND_BACK.as_secs_f32() + 1.0)), "paused, 31 s: off ({launch:?})");
            assert!(post_off(true, Some(t0), launch, s(PAUSE_HAND_BACK.as_secs_f32() - 0.01)), "paused: off until PAUSE_HAND_BACK");
            assert!(!post_off(true, Some(t0), launch, s(PAUSE_HAND_BACK.as_secs_f32() + 0.01)), "DLSS gone for good: the post path runs");
        }
        // Super Resolution goes on launching without its input held (DLAA): HAND_BACK from the last held frame.
        assert!(!post_off(true, Some(t0), run(0.0, 40.0), s(HAND_BACK.as_secs_f32() + 1.0)));
        assert_eq!(hand_back_at(t0, run(0.0, 40.0)), t0 + HAND_BACK);
        // The game resumes after 41 s in its pause menu: the run that begins there hands nothing back at once (#59),
        // only after HAND_BACK of launches with the input never held.
        let resumed = run(41.0, 41.3);
        assert!(post_off(true, Some(t0), resumed, s(41.3)), "the first frames after the menu");
        assert_eq!(hand_back_at(t0, resumed), s(41.0) + HAND_BACK);
        assert_eq!(hand_back_at(t0, None), t0 + PAUSE_HAND_BACK);
        // The run: launches under RECENT apart continue it, a gap starts another.
        let r = SrLaunches::after(None, t0);
        let r = SrLaunches::after(Some(r), s(0.1));
        assert_eq!((r.since, r.last), (t0, s(0.1)));
        let r = SrLaunches::after(Some(r), s(5.0));
        assert_eq!((r.since, r.last), (s(5.0), s(5.0)));
    }

    /// The hand-back counts from the last held frame, not from the last frame with DLSS's inputs identified: while
    /// Super Resolution's input kernel keeps launching with no frame held, the post path takes over after HAND_BACK,
    /// whether the inputs stopped being identified (no DLSS frame seen at all) or are identified and every frame is
    /// refused (DLAA); the status then says what happens (waiting for DLSS input, or identified with the reason), never
    /// paused. Only with no launch at all (a pause menu, a loading screen) does the model stay put for PAUSE_HAND_BACK,
    /// the status paused.
    #[test]
    fn the_post_path_takes_over_30_s_after_the_last_held_frame_unless_dlss_sends_nothing() {
        use neural_forge_protocol::enums::preupscale_state::{IDENTIFIED, PAUSED, WAITING};
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        // Super Resolution launching every frame from before t0 up to `to`.
        let launching = |to: f32| Some(SrLaunches { since: t0.checked_sub(Duration::from_secs(60)).unwrap_or(t0), last: s(to) });
        let mut session = Session { engaged: true, last_hold: Some(t0), last_dlss: Some(t0), ..Default::default() };
        let kept_off = |session: &Session, launches: Option<SrLaunches>, at: f32| session.post_kept_off(true, false, launches, None, s(at));
        let refused = Refusal::Text("the colour input was last transitioned out of GENERAL; not holding".into());

        // The inputs are no longer identified: launches go on, no DLSS frame is seen, nothing is held.
        assert!(kept_off(&session, launching(10.0), 10.0), "within HAND_BACK of the last held frame");
        assert_eq!(session.status_at(true, launching(10.0), s(10.0), None).0, WAITING, "the game plays: not paused");
        assert!(!kept_off(&session, launching(31.0), 31.0), "HAND_BACK after the last held frame: the post path takes over");
        assert_eq!(session.status_at(false, launching(31.0), s(31.0), None).0, WAITING);

        // DLAA: the inputs are identified and every frame is refused, so DLSS frames keep being seen.
        for at in [10.0, 31.0] {
            session.last_dlss = Some(s(at));
            assert_eq!(kept_off(&session, launching(at), at), at < 30.0, "DLAA at {at} s");
            let kept = kept_off(&session, launching(at), at);
            assert_eq!(session.status_at(kept, launching(at), s(at), Some(&refused)), (IDENTIFIED, refused.text()), "DLAA at {at} s: the reason, not paused");
        }

        // A pause menu: no launch since the last held frame. The model stays put for PAUSE_HAND_BACK, the status paused.
        session.last_dlss = Some(t0);
        let menu = Some(SrLaunches { since: t0, last: s(0.1) });
        for at in [1.0, 31.0, PAUSE_HAND_BACK.as_secs_f32() - 1.0] {
            assert!(kept_off(&session, menu, at), "paused at {at} s");
            assert_eq!(session.status_at(true, menu, s(at), Some(&refused)), (PAUSED, String::new()), "paused at {at} s");
        }
        let at = PAUSE_HAND_BACK.as_secs_f32() + 1.0;
        assert!(!kept_off(&session, menu, at), "DLSS gone for good: the post path runs");
        assert_eq!(session.status_at(false, menu, s(at), None).0, WAITING);
        // Nothing launched at all since the process saw DLSS: the same.
        assert!(kept_off(&session, None, 31.0));
        assert_eq!(session.status_at(true, None, s(31.0), None).0, PAUSED);
        // The game resumes from its menu after 41 s: no hand-back at once, and the first launches are not "paused".
        let resumed = Some(SrLaunches { since: s(41.0), last: s(41.2) });
        assert!(kept_off(&session, resumed, 41.2));
        assert_eq!(session.status_at(true, resumed, s(41.2), None).0, WAITING);
        // Not in model mode, or with the post path serving every pass: never kept off.
        assert!(!session.post_kept_off(false, false, None, None, s(1.0)));
        assert!(!session.post_kept_off(true, true, None, None, s(1.0)));
    }

    /// A DLSS game's loading screens and menus before DLSS's first frame get no post path (#59: GTA V's hangs after the
    /// post path built the network at 4K on its first loading screen, Cyberpunk 2077's spill). A game without a sign of
    /// DLSS keeps it; the post path serving every pass (placement present) is never kept off; the hand-back rules count
    /// from the sign: DLSS launching with nothing held for HAND_BACK, or nothing at all for PAUSE_HAND_BACK. Fails
    /// without the change: a session that never held kept nothing off.
    #[test]
    fn a_dlss_game_gets_no_post_path_before_dlss_first_frame() {
        use neural_forge_protocol::enums::preupscale_state::PAUSED;
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let session = Session::default();
        let kept_off = |serves: bool, launches: Option<SrLaunches>, sign: Option<Instant>, at: f32| session.post_kept_off(true, serves, launches, sign, s(at));
        // Loading screens and menus, no Super Resolution launch yet: off, and the status says paused with the game's DLSS.
        for at in [0.0, 10.0, 60.0, PAUSE_HAND_BACK.as_secs_f32() - 1.0] {
            assert!(kept_off(false, None, Some(t0), at), "before DLSS at {at} s");
        }
        assert_eq!(session.status_at(true, None, s(10.0), None).0, PAUSED);
        // DLSS never comes (the game runs another upscaler): the post path after PAUSE_HAND_BACK.
        assert!(!kept_off(false, None, Some(t0), PAUSE_HAND_BACK.as_secs_f32() + 1.0));
        // DLSS starts at 50 s and its frames are not held (DLAA, inputs not identified): HAND_BACK after that.
        let launching = |to: f32| Some(SrLaunches { since: s(50.0), last: s(to) });
        assert!(kept_off(false, launching(60.0), Some(t0), 60.0));
        assert!(!kept_off(false, launching(81.0), Some(t0), 81.0), "HAND_BACK of launches with nothing held");
        // A game with no sign of DLSS, or one whose frames were held already (`before_dlss` is then `None`): as before.
        assert!(!kept_off(false, None, None, 1.0));
        // The post path serves every pass (placement present): never kept off.
        assert!(!kept_off(true, None, Some(t0), 1.0));
        // Not in model mode: never.
        assert!(!session.post_kept_off(false, false, None, Some(t0), s(1.0)));
        assert!(pre_dlss_post_off(Some(t0), None, s(1.0)) && !pre_dlss_post_off(None, None, s(1.0)));
    }

    /// Ray Reconstruction's first frames do not hand the frames to the post path: only once it has run for
    /// RAY_RECONSTRUCTION_SETTLE (the hold after DLSS is learned from its first frames), and only while running.
    #[test]
    fn ray_reconstruction_hands_the_post_path_its_frames_only_once_settled() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let run = |since: f32, last: f32| Some(SrLaunches { since: s(since), last: s(last) });
        assert!(!ray_reconstruction_settled(None, s(1.0)));
        assert!(!ray_reconstruction_settled(run(0.0, 1.0), s(1.0)), "just started");
        assert!(!ray_reconstruction_settled(run(0.0, 1.9), s(1.95)));
        assert!(ray_reconstruction_settled(run(0.0, 2.5), s(2.5)), "settled");
        assert!(!ray_reconstruction_settled(run(0.0, 2.5), s(4.0)), "stopped");
        // Ray Reconstruction settled, holds after DLSS neither recorded nor tried: the post path serves; once they are, not.
        let auto = (1, [neural_forge_protocol::enums::placement::AUTO; neural_forge_protocol::MAX_PASSES]);
        assert!(post_serves_for(auto, true, false));
        assert!(!post_serves_for(auto, true, true));
        assert!(!post_serves_for(auto, false, false));
    }

    // ---- The HDR encode: a CPU reference and the GPU checks against it. ----

    /// Single to half precision, round to nearest even.
    pub(crate) fn f32_to_f16(v: f32) -> u16 {
        if v.is_nan() {
            return 0x7e00;
        }
        let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
        let a = f64::from(v.abs());
        if a >= 65520.0 {
            return sign | 0x7c00;
        }
        if a < 2f64.powi(-14) {
            // Subnormal (or zero) in units of 2^-24; 1024 rounds up into the smallest normal.
            return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
        }
        let mut e = a.log2().floor() as i32;
        if 2f64.powi(e) > a {
            e -= 1;
        }
        if 2f64.powi(e + 1) <= a {
            e += 1;
        }
        let mut m = ((a / 2f64.powi(e) - 1.0) * 1024.0).round_ties_even() as u32;
        if m == 1024 {
            m = 0;
            e += 1;
        }
        if e > 15 {
            return sign | 0x7c00;
        }
        sign | (((e + 15) as u16) << 10) | m as u16
    }

    #[test]
    fn half_conversion_round_trips_every_finite_half() {
        for h in 0u16..=0xffff {
            let v = f16_to_f32(h);
            if v.is_finite() {
                assert_eq!(f16_to_f32(f32_to_f16(v)).to_bits(), v.to_bits(), "{h:#06x} = {v}");
            }
        }
        assert_eq!(f32_to_f16(0.1), 0x2e66);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
    }

    /// The CPU reference of `preupscale_encode.comp` / `preupscale_decode.comp`, in f64, from the
    /// formulas in docs/PRE_UPSCALER_DESIGN.md, "E1b: the HDR encode" (the 8-bit path's edit).
    mod hdr_ref {
        const K: f64 = 5.770780;
        const KNEE: f64 = 0.75;

        fn srgb_oetf(c: f64) -> f64 {
            let c = c.clamp(0.0, 1.0);
            if c <= 0.0031308 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            }
        }

        fn srgb_eotf(s: f64) -> f64 {
            let s = s.clamp(0.0, 1.0);
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        }

        /// scene-linear -> model domain: `max(scene, 0) * e / w`, the shoulder above 0.75, sRGB.
        pub fn encode(scene: f32, e: f32, w: f32) -> f64 {
            let x = if scene >= 0.0 { f64::from(scene) } else { 0.0 };
            let v = x * f64::from(e) / f64::from(w);
            srgb_oetf(if v <= KNEE { v } else { KNEE + 0.25 * (1.0 - (-K * (v - KNEE)).exp()) })
        }

        /// model domain -> scene-linear: sRGB EOTF, clamped below 1 - 1e-4, the shoulder's inverse.
        pub fn decode(answer: f32, e: f32, w: f32) -> f64 {
            let y = srgb_eotf(f64::from(answer)).min(1.0 - 1e-4);
            let v = if y >= KNEE { KNEE - (1.0 - (y - KNEE) / 0.25).ln() / K } else { y };
            v * f64::from(w) / f64::from(e)
        }
    }

    /// A scene-linear RGBA16F test frame: a ramp from 0 to about 60 per channel at different rates
    /// (with the game's exposure and paper white 3, encoded values from 0 through the shoulder to
    /// its clamped top), a black texel, a sun texel (1000, 800, 30000) and a near-black one; alpha
    /// 0.25..0.75.
    fn hdr_pattern(width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut out = Vec::with_capacity(n * TEXEL as usize);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let rgb = match i {
                0 => [0.0, 0.0, 0.0],
                1 => [1000.0, 800.0, 30000.0],
                2 => [0.01, 0.02, 0.05],
                _ => [60.0 * t * t, 45.0 * t, 48.0 * t.sqrt()],
            };
            for v in [rgb[0], rgb[1], rgb[2], 0.25 + 0.5 * t] {
                out.extend_from_slice(&f32_to_f16(v as f32).to_le_bytes());
            }
        }
        out
    }

    /// One half of a tightly packed RGBA16F buffer `row` texels wide.
    fn half_at(buf: &[u8], row: u32, x: u32, y: u32, c: usize) -> u16 {
        let at = ((y * row + x) as usize * TEXEL as usize) + c * 2;
        u16::from_le_bytes([buf[at], buf[at + 1]])
    }

    fn proxy_bytes(shm: &ShmClient, pw: u32, ph: u32) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(shm.proxy_region(Slot::Primary).unwrap().0, (pw * ph) as usize * TEXEL as usize) }.to_vec()
    }

    /// Check (a): every texel of the padded proxy is the CPU encode of the colour input's texel
    /// (the edge texel for the padding) within 1e-3, alpha exactly 1. Returns how many channels in
    /// the real extent the encode clamped (>= 0.999).
    fn check_encode(proxy: &[u8], original: &[u8], w: u32, h: u32, e: f32, white: f32) -> usize {
        let (pw, ph) = padded(w, h);
        let mut clamped = 0;
        for y in 0..ph {
            for x in 0..pw {
                let (sx, sy) = (x.min(w - 1), y.min(h - 1));
                for c in 0..3 {
                    let scene = f16_to_f32(half_at(original, w, sx, sy, c));
                    let want = hdr_ref::encode(scene, e, white);
                    let got = f16_to_f32(half_at(proxy, pw, x, y, c));
                    assert!((f64::from(got) - want).abs() <= 1e-3, "encode at {x},{y} channel {c}: scene {scene} -> {got}, the CPU reference says {want}");
                    if got >= hdr::CLAMPED && x < w && y < h {
                        clamped += 1;
                    }
                }
                assert_eq!(half_at(proxy, pw, x, y, 3), 0x3c00, "alpha 1 at {x},{y}");
            }
        }
        clamped
    }

    /// The write-back: the whole original pixel where the encoded input's (`proxy`'s) peak channel
    /// reached 0.999, else per channel the CPU inverse of the answer (`answer`) within relative 2e-3, faded into
    /// the original from a peak of [`hdr::FADE_SRGB`] (smoothstep); alpha the original's.
    #[allow(clippy::too_many_arguments)]
    fn check_decode(after: &[u8], original: &[u8], proxy: &[u8], answer: &[u8], w: u32, h: u32, e: f32, white: f32) {
        let (pw, _) = padded(w, h);
        for y in 0..h {
            for x in 0..w {
                let peak = (0..3).map(|c| f16_to_f32(half_at(proxy, pw, x, y, c))).fold(f32::MIN, f32::max);
                for c in 0..3 {
                    let got = half_at(after, w, x, y, c);
                    let orig = half_at(original, w, x, y, c);
                    if peak >= hdr::CLAMPED {
                        assert_eq!(got, orig, "a clamped highlight keeps its whole original pixel at {x},{y} channel {c} (encoded peak {peak})");
                        continue;
                    }
                    let decoded = hdr_ref::decode(f16_to_f32(half_at(answer, pw, x, y, c)), e, white);
                    let t = f64::from(((peak - hdr::FADE_SRGB) / (hdr::CLAMPED - hdr::FADE_SRGB)).clamp(0.0, 1.0));
                    let t = t * t * (3.0 - 2.0 * t);
                    let want = decoded * (1.0 - t) + f64::from(f16_to_f32(orig)) * t;
                    let got = f64::from(f16_to_f32(got));
                    assert!((got - want).abs() <= 2e-3 * want.abs() + 1e-5, "decode at {x},{y} channel {c}: {got}, the CPU reference says {want}");
                }
                assert_eq!(half_at(after, w, x, y, 3), half_at(original, w, x, y, 3), "alpha is kept at {x},{y}");
            }
        }
    }

    /// A 1x1 R16_SFLOAT image holding `value`, in GENERAL: DLSS's exposure input.
    fn exposure_image(gpu: &Gpu, value: f32) -> (vk::Image, vk::DeviceMemory) {
        let d = &gpu.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16_SFLOAT)
            .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { d.create_image(&info, None) }.unwrap();
        let reqs = unsafe { d.get_image_memory_requirements(image) };
        let index = (0..32).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
        let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None) }.unwrap();
        unsafe { d.bind_image_memory(image, memory, 0) }.unwrap();
        let range =
            vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        gpu.one_shot(|cmd| unsafe {
            Gpu::barrier(cmd, d, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            d.cmd_clear_color_image(cmd, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [value, 0.0, 0.0, 0.0] }, &[range]);
            Gpu::barrier(cmd, d, image, vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL);
        });
        (image, memory)
    }

    fn exposure_aux(image: vk::Image) -> Option<Aux> {
        Some(Aux { image, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true })
    }

    /// Waits for the layer's own work, then frees `res` and the test's images.
    fn finish(gpu: &Gpu, mut res: Resources, images: &[(vk::Image, vk::DeviceMemory)]) {
        let mut gpu_ms = None;
        assert!(res.wait_idle(&gpu.device, &mut gpu_ms));
        unsafe {
            res.destroy(&gpu.device);
            for &(i, m) in images {
                gpu.device.destroy_image(i, None);
                gpu.device.free_memory(m, None);
            }
        }
    }

    /// Checks (a) and (b): roundtrip mode on a 17x9 scene-linear frame with the game's exposure.
    /// The proxy is the CPU reference encode within 1e-3 (padding included, alpha 1); the
    /// write-back gives the frame back: channels the encode clamped (>= 0.999) bit-exactly, every
    /// other one within relative 1e-3 plus what the half floats themselves lose (one step of the
    /// 16-bit proxy, carried through the inverse, plus one step of the 16-bit result). That bound
    /// rather than a flat 2e-3: lavapipe's (and Intel's) float-to-half store truncates, which alone makes
    /// midtones come back up to ~2.6e-3 off, and towards the shoulder's top the inverse steepens
    /// so one proxy step is several percent (the maxima per band are printed). The write-back is
    /// also checked against the CPU inverse of the proxy within 2e-3 (`check_decode`). No model is
    /// involved. A second hold on the result is again its own inverse. Imported and staged.
    #[test]
    fn roundtrip_encodes_like_the_cpu_reference_and_decodes_back_to_the_frame() {
        for import in [true, false] {
            let Some(gpu) = Gpu::open(import) else {
                eprintln!("preupscale roundtrip test (import={import}): no suitable Vulkan device, skipping");
                continue;
            };
            let (w, h) = (17u32, 9u32);
            let (pw, ph) = padded(w, h);
            let (image, memory) = gpu.image(w, h);
            let original = hdr_pattern(w, h);
            gpu.upload(image, w, h, &original);
            let (exposure, exposure_mem) = exposure_image(&gpu, 0.128);
            let shm = scratch_shm(if import { "rt-import" } else { "rt-staged" });
            let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, import) }.expect("resources");
            let white = 2.5;
            let target = Target {
                colour: image,
                width: w,
                height: h,
                depth: None,
                depth_buffer: None,
                region: None,
                mvec: None,
                mvec_scale: [1.0, 1.0],
                exposure: [None; MAX_EXPOSURE],
                exposure_input: exposure_aux(exposure),
                paper_white: white,
                lift_exposure: false,
                pre_exposed: false,
                identification: 1,
            };
            let (device, queue) = (&gpu.device, gpu.queue);
            let mut submits = Vec::new();
            let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                submits.push(which);
                device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
            };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 0, &mut submit) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{result:?}");
            assert!(!shm.has_pending_request(Slot::Primary), "roundtrip never requests an answer");
            let e = result.exposure.expect("exposure read");
            let proxy = proxy_bytes(&shm, pw, ph);
            let clamped = check_encode(&proxy, &original, w, h, e, white);
            let after = gpu.read(image, w, h);
            check_decode(&after, &original, &proxy, &proxy, w, h, e, white);
            let (mut strict, mut steep, mut kept) = (0f64, 0f64, 0);
            // How this device rounds into the 16-bit proxy (reported): to nearest, or towards zero.
            let (mut nearest, mut channels) = (0, 0);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let want = hdr_ref::encode(f16_to_f32(half_at(&original, w, x, y, c)), e, white) as f32;
                        channels += 1;
                        nearest += usize::from(half_at(&proxy, pw, x, y, c) == f32_to_f16(want));
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let x0 = f64::from(f16_to_f32(half_at(&original, w, x, y, c)));
                        let enc_bits = half_at(&proxy, pw, x, y, c);
                        let enc = f16_to_f32(enc_bits);
                        let got = f64::from(f16_to_f32(half_at(&after, w, x, y, c)));
                        // The decode keeps the whole pixel where its encoded peak channel clamped.
                        let peak = (0..3).map(|k| f16_to_f32(half_at(&proxy, pw, x, y, k))).fold(f32::MIN, f32::max);
                        if peak >= hdr::CLAMPED {
                            kept += usize::from(enc >= hdr::CLAMPED);
                            assert_eq!(got, x0);
                            continue;
                        }
                        // One step of the 16-bit proxy, in scene units, plus one step of the 16-bit
                        // result: what half floats alone can lose (either rounding mode).
                        let proxy_step = (hdr_ref::decode(f16_to_f32(enc_bits + 1), e, white) - hdr_ref::decode(enc, e, white)).abs();
                        let out_step = f64::from(f16_to_f32(f32_to_f16(x0 as f32) + 1)) - x0;
                        let rel = (got - x0).abs() / x0.max(1e-6);
                        assert!(
                            (got - x0).abs() <= 1e-3 * x0 + proxy_step + out_step + 1e-6,
                            "roundtrip at {x},{y} channel {c}: {x0} -> {got} (encoded {enc}, one proxy step = {proxy_step}, one result step = {out_step})"
                        );
                        if enc <= 0.9 {
                            strict = strict.max(rel);
                        } else {
                            steep = steep.max(rel);
                        }
                    }
                }
            }
            assert_eq!(kept, clamped);
            assert!(kept > 0, "the pattern has clamped highlights");
            // Stable: a second roundtrip on the written-back frame is again its own inverse.
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 1, &mut submit) };
            assert!(result.wrote_back, "{result:?}");
            assert_eq!(submits, vec![Which::Capture, Which::WriteBack, Which::Capture, Which::WriteBack]);
            let again = gpu.read(image, w, h);
            let proxy2 = proxy_bytes(&shm, pw, ph);
            check_decode(&again, &after, &proxy2, &proxy2, w, h, e, white);
            eprintln!(
                "preupscale roundtrip test (import={import}): e={e} white={white}; max relative error {strict:.2e} (encoded <= 0.9), {steep:.2e} (0.9 < encoded < 0.999); {kept} clamped channels kept exactly; {nearest}/{channels} proxy halves round-to-nearest of the CPU encode"
            );
            finish(&gpu, res, &[(image, memory), (exposure, exposure_mem)]);
        }
    }

    /// A split hold's write-back fence, noted with the network's loader (`Loader::pre_submitted`), is forgotten
    /// when the resources it belongs to go idle or are destroyed, also when no later hold of theirs comes: the
    /// loader asks a noted fence for its status later (another device's lend, the after-the-upscaler path's
    /// claim), and must never ask a destroyed one. Another resources' fence is left noted.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn destroyed_resources_leave_no_fence_noted_with_the_loader() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let d = &gpu.device;
        let shm = scratch_shm("prefence");
        let setup = native::Setup::for_test(&gpu._entry, &gpu.instance, gpu.physical, gpu.family);
        let loader: NativeLoader = Arc::new(native::Loader::for_test(setup, Vec::new(), Arc::default(), Box::new(|_, _| {})));
        // SAFETY: a fresh device; each submit below is waited on before the resources are destroyed.
        unsafe {
            let mut first = Resources::build(d, &gpu.instance, gpu.physical, gpu.family, 16, 8, &shm, false).expect("resources");
            let mut second = Resources::build(d, &gpu.instance, gpu.physical, gpu.family, 16, 8, &shm, false).expect("resources");
            // As a hold does: the loader set, the resources waited for, the write-back submitted and noted.
            let hold = |res: &mut Resources| {
                res.pre_loader = Some(loader.clone());
                assert!(res.wait_idle(d, &mut None));
                d.reset_fences(&[res.writeback_fence]).unwrap();
                d.queue_submit(gpu.queue, &[], res.writeback_fence).unwrap();
                res.writeback_pending = true;
                loader.pre_submitted(d, res.writeback_fence);
            };
            hold(&mut first);
            assert_eq!(loader.pre_fence(), Some(first.writeback_fence));
            d.queue_wait_idle(gpu.queue).unwrap();
            // The second resources' wait leaves the first one's fence noted.
            second.pre_loader = Some(loader.clone());
            assert!(second.wait_idle(d, &mut None));
            assert_eq!(loader.pre_fence(), Some(first.writeback_fence), "another resources' fence stays noted");
            // The first resources are destroyed (a new extent, `Session::ensure_in`) with no hold of theirs after.
            first.destroy(d);
            assert_eq!(loader.pre_fence(), None, "the destroyed fence is forgotten");
            // And a wait of the resources whose fence is noted forgets it too.
            hold(&mut second);
            d.queue_wait_idle(gpu.queue).unwrap();
            assert!(second.wait_idle(d, &mut None));
            assert_eq!(loader.pre_fence(), None);
            second.destroy(d);
        }
    }

    /// Check (d): a readable exposure image holding 0: model and roundtrip modes leave the frame
    /// untouched and never run the model (the capture runs, the write-back is skipped). No
    /// exposure image, or an unreadable one, is no longer a reason to leave the frame alone: the
    /// exposure is measured from the frame ([`ExposureSource::Auto`]), and those holds write back.
    #[test]
    fn a_zero_exposure_leaves_the_frame_untouched_and_a_missing_one_is_measured() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale exposure fail-open test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (9u32, 5u32);
        let (image, memory) = gpu.image(w, h);
        let original = hdr_pattern(w, h);
        gpu.upload(image, w, h, &original);
        let (zero, zero_mem) = exposure_image(&gpu, 0.0);
        let shm = scratch_shm("noexposure");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let (device, queue) = (&gpu.device, gpu.queue);
        let mut submits = Vec::new();
        let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            submits.push(which);
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
        };
        let base = Target {
            colour: image,
            width: w,
            height: h,
            depth: None,
            depth_buffer: None,
            region: None,
            mvec: None,
            mvec_scale: [1.0, 1.0],
            exposure: [None; MAX_EXPOSURE],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            lift_exposure: false,
            pre_exposed: false,
            identification: 1,
        };
        {
            let target = Target { exposure_input: exposure_aux(zero), ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 0, &mut submit) };
            assert!(result.waits_consumed && !result.wrote_back, "{result:?}");
            assert_eq!((result.exposure, result.exposure_source), (Some(0.0), Some(ExposureSource::Game)));
            assert!(result.miss.is_some_and(|m| m.contains("exposure value")), "{result:?}");
        }
        assert_eq!(submits, vec![Which::Capture], "a capture only, no write-back, with a zero one");
        assert!(!shm.has_pending_request(Slot::Primary));
        assert_eq!(gpu.read(image, w, h), original, "the frame is untouched");
        let unreadable = Some(Aux { readable: false, ..exposure_aux(zero).unwrap() });
        let unknown_layout = Some(Aux { layout: None, ..exposure_aux(zero).unwrap() });
        let mut plain = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
        };
        for exposure_input in [None, unreadable, unknown_layout] {
            let target = Target { exposure_input, ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 0, &mut plain) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{exposure_input:?}: {result:?}");
            assert_eq!(result.exposure_source, Some(ExposureSource::Auto), "{exposure_input:?}");
            assert!(result.exposure.is_some_and(hdr::exposure_ok), "{result:?}");
        }
        finish(&gpu, res, &[(image, memory), (zero, zero_mem)]);
    }

    /// The write-back stays within the finite half range. At an exposure of 5e-5 (a sunlit physically based
    /// game's) the shoulder's inverse of a fully clipped answer (1.0), and of one just below the top (0.985),
    /// lies past 65504: it stores 65504, never +Inf, which DLSS would spread through its history. A mid-range
    /// answer decodes as the CPU reference says, and the pixel whose encode clamped (an infinite channel) keeps
    /// the original. With an unusable exposure (0) the encode of that infinite channel is 0, as every other
    /// value, not the NaN of +Inf x 0, and nothing is written back. (Lavapipe's float-to-half conversion
    /// rounds toward zero, so an overflow lands on 65504 there anyway, and its clamp() returns 0 for a NaN:
    /// this pins the result every driver must give, NVIDIA's round-to-nearest one included.)
    /// The ACES transfer on the GPU against its CPU reference (`hdr::aces`): the encode of every texel, the
    /// decode of an edited answer (a tenth darker, linear), and the decode of the answer left as it was sent
    /// (the frame back), at the game's exposure and the default paper white.
    #[test]
    fn the_aces_transfer_matches_its_cpu_reference() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale ACES test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (6u32, 3u32);
        let (pw, ph) = padded(w, h);
        let d = &gpu.device;
        let (image, memory) = gpu.image(w, h);
        let colours = [
            [0.0, 0.0, 0.0],
            [0.002, 0.003, 0.004],
            [0.05, 0.1, 0.2],
            [0.3, 0.3, 0.3],
            [0.9, 0.5, 0.1],
            [2.0, 1.5, 1.0],
            [4.0, 0.2, 0.05],
            [0.1, 0.2, 6.0],
            [12.0, 10.0, 9.0],
            [1000.0, 800.0, 30000.0],
            [f32::NAN, 0.5, 0.5],
            [-1.0, 0.5, 2.0],
        ];
        let colour = |x: u32, y: u32| colours[((y * w + x) as usize) % colours.len()];
        let scene = frame(w, h, colour);
        let white = hdr::DEFAULT_PAPER_WHITE;
        let e = f16_to_f32(f32_to_f16(0.6));
        let exposure = e * hdr::ACES_WHITE / white;
        let original = |x: u32, y: u32, c: usize| f16_to_f32(half_at(&scene, w, x, y, c));
        let mut pass = unsafe { hdr::HdrPass::build(d, &gpu.instance, gpu.physical, w, h) }.expect("the HDR pass builds");
        pass.set_transfer(hdr::ACES);
        assert!(unsafe { pass.bind_colour(d, image) });
        let source = own_host_buffer_with(d, &gpu.instance, gpu.physical, u64::from(pw * ph) * 8, vk::BufferUsageFlags::TRANSFER_SRC).unwrap();
        // One hold with `answer` (padded, RGBA16F halves) as the network's: the frame after it and the encoded picture.
        let hold = |answer: &[u16]| {
            unsafe { std::ptr::copy_nonoverlapping(answer.as_ptr().cast::<u8>(), source.ptr, answer.len() * 2) };
            gpu.upload(image, w, h, &scene);
            gpu.one_shot(|cmd| unsafe {
                d.cmd_fill_buffer(cmd, pass.exposure_buffer(), 0, 4, u32::from(f32_to_f16(e)));
                let open = [pass.encoded_to_general(), pass.answer_to_general()];
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &open,
                );
                pass.record_encode(d, cmd, None, white);
                let encoded = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[encoded],
                    &[],
                    &[],
                );
                pass.record_decode(d, cmd, source.buffer, white, None);
            });
            (gpu.read(image, w, h), gpu.read(pass.encoded_image(), pw, ph))
        };

        // The encode, and an answer a tenth darker in linear light (the padding's texels as sent).
        let (_, encoded) = hold(&vec![0u16; (pw * ph * 4) as usize]);
        let mut edited = Vec::with_capacity((pw * ph * 4) as usize);
        for y in 0..ph {
            for x in 0..pw {
                let (sx, sy) = (x.min(w - 1), y.min(h - 1));
                for c in 0..3 {
                    let want = hdr::aces::encode(original(sx, sy, c), exposure);
                    let got = f16_to_f32(half_at(&encoded, pw, x, y, c));
                    assert!((got - want).abs() <= 1e-3, "encode at {x},{y} channel {c}: {} -> {got}, the CPU reference says {want}", original(sx, sy, c));
                    edited.push(f32_to_f16(hdr::aces::srgb_oetf(hdr::aces::srgb_eotf(got) * 0.9)));
                }
                edited.push(0x3c00);
            }
        }
        let check = |after: &[u8], answer: &dyn Fn(u32, u32, usize) -> f32, what: &str| {
            for y in 0..h {
                for x in 0..w {
                    let px = [0, 1, 2].map(|c| original(x, y, c));
                    let want = hdr::aces::decode(px, [0, 1, 2].map(|c| answer(x, y, c)), exposure, true);
                    for (c, want) in want.into_iter().enumerate() {
                        let got = f16_to_f32(half_at(after, w, x, y, c));
                        if want.is_nan() {
                            assert!(got.is_nan(), "{what} at {x},{y} channel {c}: {got}, the original's NaN kept");
                        } else {
                            assert!((got - want).abs() <= 2e-3 * want.abs() + 1e-5, "{what} at {x},{y} channel {c}: {got}, the CPU reference says {want}");
                        }
                    }
                    assert_eq!(half_at(after, w, x, y, 3), half_at(&scene, w, x, y, 3), "alpha is kept at {x},{y}");
                }
            }
        };
        let (after, _) = hold(&edited);
        check(&after, &|x, y, c| f16_to_f32(edited[((y * pw + x) * 4) as usize + c]), "the edited answer");
        // No edit: the frame back, as the CPU reference has it (the original within the halves' rounding).
        let sent: Vec<u16> = encoded.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        let (after, _) = hold(&sent);
        check(&after, &|x, y, c| f16_to_f32(half_at(&encoded, pw, x, y, c)), "the answer as sent");
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let (o, got) = (original(x, y, c), f16_to_f32(half_at(&after, w, x, y, c)));
                    if o.is_finite() && o * exposure > 1e-3 {
                        assert!((got - o).abs() <= 4e-3 * o, "no edit at {x},{y} channel {c}: {o} -> {got}");
                    }
                }
            }
        }
        unsafe {
            pass.destroy(d);
            source.destroy(d);
            d.destroy_image(image, None);
            d.free_memory(memory, None);
        }
    }

    #[test]
    fn the_decode_stores_finite_halves_at_a_low_exposure_and_an_infinity_encodes_to_0_without_one() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale half-range test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (2u32, 2u32);
        let (pw, ph) = padded(w, h);
        let d = &gpu.device;
        let (image, memory) = gpu.image(w, h);
        let scene = frame(w, h, |x, y| if (x, y) == (1, 1) { [f32::INFINITY, 1.0, 1.0] } else { [1.0, 2.0, 3.0] });
        let white = hdr::DEFAULT_PAPER_WHITE;
        let answers = [1.0f32, 0.985, 0.5];
        let answer: Vec<u16> = (0..pw * ph).flat_map(|_| [f32_to_f16(answers[0]), f32_to_f16(answers[1]), f32_to_f16(answers[2]), 0x3c00]).collect();
        let source = own_host_buffer_with(d, &gpu.instance, gpu.physical, u64::from(pw * ph) * 8, vk::BufferUsageFlags::TRANSFER_SRC).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(answer.as_ptr().cast::<u8>(), source.ptr, answer.len() * 2) };
        let mut pass = unsafe { hdr::HdrPass::build(d, &gpu.instance, gpu.physical, w, h) }.expect("the HDR pass builds");
        assert!(unsafe { pass.bind_colour(d, image) });
        // One hold's encode and decode at exposure `e`: the frame after it and the encoded picture.
        let hold = |e: f32| {
            gpu.upload(image, w, h, &scene);
            gpu.one_shot(|cmd| unsafe {
                d.cmd_fill_buffer(cmd, pass.exposure_buffer(), 0, 4, u32::from(f32_to_f16(e)));
                let open = [pass.encoded_to_general(), pass.answer_to_general()];
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &open,
                );
                pass.record_encode(d, cmd, None, white);
                let encoded = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[encoded],
                    &[],
                    &[],
                );
                pass.record_decode(d, cmd, source.buffer, white, None);
            });
            (gpu.read(image, w, h), gpu.read(pass.encoded_image(), pw, ph))
        };

        let (after, _) = hold(5e-5);
        let e = f16_to_f32(f32_to_f16(5e-5));
        for y in 0..h {
            for x in 0..w {
                for (c, &a) in answers.iter().enumerate() {
                    let got = half_at(&after, w, x, y, c);
                    if (x, y) == (1, 1) {
                        assert_eq!(got, half_at(&scene, w, x, y, c), "the clamped pixel keeps the original, channel {c}");
                        continue;
                    }
                    let want = hdr_ref::decode(f16_to_f32(f32_to_f16(a)), e, white);
                    if want > 65504.0 {
                        assert_eq!(got, 0x7bff, "answer {} at {x},{y} (the inverse {want}) stores the largest finite half", a);
                    } else {
                        let got = f64::from(f16_to_f32(got));
                        assert!((got - want).abs() <= 2e-3 * want, "answer {} at {x},{y}: {got}, the CPU reference says {want}", a);
                    }
                }
            }
        }
        assert!(hdr_ref::decode(f16_to_f32(f32_to_f16(0.985)), e, white) > 65504.0, "the test reaches past the half range below the top too");

        let (after, encoded) = hold(0.0);
        assert_eq!(after, scene, "nothing is written back without a usable exposure");
        for y in 0..ph {
            for x in 0..pw {
                for c in 0..3 {
                    assert_eq!(half_at(&encoded, pw, x, y, c), 0, "an unusable exposure encodes {x},{y} channel {c} as 0");
                }
            }
        }
        unsafe {
            pass.destroy(d);
            source.destroy(d);
            d.destroy_image(image, None);
            d.free_memory(memory, None);
        }
    }

    // ---- Auto-exposure: the CPU reference on synthetic frames, and the GPU pass against it. ----

    /// A tightly packed RGBA16F frame from a per-texel colour.
    fn frame(width: u32, height: u32, mut colour: impl FnMut(u32, u32) -> [f32; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height) as usize * TEXEL as usize);
        for y in 0..height {
            for x in 0..width {
                for v in colour(x, y).into_iter().chain([1.0]) {
                    out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
            }
        }
        out
    }

    /// Luma of a half-rounded grey `v` (what the frame holds).
    fn grey(v: f32) -> [f32; 3] {
        [v, v, v]
    }

    /// The exact trimmed log2-average luma of `texels` over the same samples as the shader (even x
    /// and y, finite, not black): sorted, the lowest and highest [`hdr::AUTO_TRIM`] dropped
    /// (fractionally at the cut), averaged in f64. What the histogram approximates.
    fn exact_trimmed_mean_log2(texels: &[u8], width: u32, height: u32) -> f64 {
        let mut logs = Vec::new();
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                let rgb: Vec<f64> = (0..3).map(|c| f64::from(f16_to_f32(half_at(texels, width, x, y, c)))).collect();
                if rgb.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                let l = 0.2126 * rgb[0].max(0.0) + 0.7152 * rgb[1].max(0.0) + 0.0722 * rgb[2].max(0.0);
                if l >= 2f64.powi(-24) {
                    logs.push(l.log2());
                }
            }
        }
        logs.sort_by(f64::total_cmp);
        let n = logs.len() as f64;
        let (lo, hi) = (f64::from(hdr::AUTO_TRIM) * n, n - f64::from(hdr::AUTO_TRIM) * n);
        let (mut w, mut sum) = (0.0, 0.0);
        for (i, l) in logs.iter().enumerate() {
            let (a, b) = ((i as f64).max(lo), (i as f64 + 1.0).min(hi));
            if b > a {
                w += b - a;
                sum += (b - a) * l;
            }
        }
        sum / w
    }

    /// A deterministic log-normal luma frame (grey) with the given median and log spread `sigma`
    /// (natural log): Box-Muller over a fixed LCG.
    fn log_normal(width: u32, height: u32, median: f64, sigma: f64) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        frame(width, height, |_, _| {
            let z = (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
            grey((median * (sigma * z).exp()) as f32)
        })
    }

    /// The reference's histogram and trimmed log-average against the exact sort-based trimmed mean,
    /// and its exposure against the formula, on synthetic HDR frames: uniform (e = key / L), a
    /// gradient, bright sky over dark ground (the geometric mean of the two), the same with NaN,
    /// infinite and negative texels (ignored / clamped), with a 0.5% sun (trimmed away), and black
    /// (no samples: e stays, or 1). Adaptation: 5% per frame in log2 towards the target.
    #[test]
    fn the_auto_exposure_reference_finds_the_trimmed_log_average() {
        let key = f64::from(hdr::AUTO_KEY);
        let check = |name: &str, texels: &[u8], w: u32, h: u32, want_mean: Option<f64>| {
            let got = hdr::auto_exposure_reference(texels, w, h, None);
            let exact = exact_trimmed_mean_log2(texels, w, h);
            assert!((f64::from(got.mean_log2) - exact).abs() < 0.01, "{name}: histogram mean log2 {} vs exact {exact}", got.mean_log2);
            if let Some(want) = want_mean {
                assert!((exact - want).abs() < 0.02, "{name}: exact trimmed mean log2 {exact}, expected {want}");
            }
            assert!((f64::from(got.target_log2_e) - (key.log2() - f64::from(got.mean_log2))).abs() < 1e-5, "{name}: target = log2(key) - mean");
            assert_eq!(got.log2_e, got.target_log2_e, "{name}: the first frame takes the target");
            got
        };
        let (w, h) = (96u32, 64u32);
        // Uniform: e = key / L exactly (to the histogram's 1/256-bin resolution).
        for l in [0.01f32, 1.0, 5.9, 3000.0] {
            let got = check("uniform", &frame(w, h, |_, _| grey(l)), w, h, Some(f64::from(f16_to_f32(f32_to_f16(l))).log2()));
            let e = got.log2_e.exp2();
            assert!((f64::from(e) - key / f64::from(l)).abs() <= 0.005 * key / f64::from(l), "uniform {l}: e {e}, want {}", key / f64::from(l));
        }
        // A horizontal gradient 0.5..50 and a log gradient (exact mean known).
        check("gradient", &frame(w, h, |x, _| grey(0.5 + 49.5 * x as f32 / (w - 1) as f32)), w, h, None);
        check("log gradient", &frame(w, h, |x, _| grey(2f32.powf(-3.0 + 8.0 * x as f32 / (w - 1) as f32))), w, h, None);
        // Bright sky over dark ground, half and half: the log-average is the geometric mean.
        let sky = |_: u32, y: u32| if y < h / 2 { [40.0, 50.0, 70.0] } else { [0.4, 0.5, 0.3] };
        let sky_luma = 0.2126f64 * 40.0 + 0.7152 * 50.0 + 0.0722 * 70.0;
        let ground_luma = 0.2126f64 * 0.4 + 0.7152 * 0.5 + 0.0722 * 0.3;
        let geo = (sky_luma.log2() + ground_luma.log2()) / 2.0;
        let sky_ground = frame(w, h, sky);
        let base = check("sky and ground", &sky_ground, w, h, Some(geo));
        // NaN, +Inf, -Inf and negative texels: the first three are skipped, negatives are clamped
        // to 0 (and a black sample is skipped). One bad texel per 16x8 block, on sampled positions.
        let dirty = frame(w, h, |x, y| match (x % 16, y % 8) {
            (0, 0) => [f32::NAN, 1.0, 1.0],
            (2, 0) => [1.0, f32::INFINITY, 1.0],
            (4, 0) => [f32::NEG_INFINITY, 1.0, 1.0],
            (6, 0) => [-5.0, -5.0, -5.0],
            _ => sky(x, y),
        });
        let got = check("with NaN/Inf/negative", &dirty, w, h, Some(geo));
        assert!(got.samples < base.samples, "the bad texels were not counted");
        assert!((got.mean_log2 - base.mean_log2).abs() < 0.02, "{} vs {}", got.mean_log2, base.mean_log2);
        // A 0.5% sun at 30000 (fewer than the trimmed 1%): trimmed away.
        let sunny = frame(w, h, |x, y| if y == 0 && x < 30 && x % 2 == 0 { grey(30000.0) } else { sky(x, y) });
        let got = check("with a sun", &sunny, w, h, None);
        assert!((got.mean_log2 - base.mean_log2).abs() < 0.03, "the sun moved the mean: {} vs {}", got.mean_log2, base.mean_log2);
        // Black: nothing to measure, e keeps the previous value (or 1 with none).
        let black = frame(w, h, |_, _| grey(0.0));
        assert_eq!(hdr::auto_exposure_reference(&black, w, h, None).log2_e, 0.0);
        assert_eq!(hdr::auto_exposure_reference(&black, w, h, Some(-2.5)).log2_e, -2.5);
        // Adaptation: 5% of the way in log2, frame after frame.
        let first = hdr::auto_exposure_reference(&sky_ground, w, h, None);
        let next = hdr::auto_exposure_reference(&sky_ground, w, h, Some(first.target_log2_e + 2.0));
        assert!((next.log2_e - (first.target_log2_e + 2.0 * 0.95)).abs() < 1e-5, "{next:?}");
    }

    /// The key against GTA V's own exposure (docs/PRE_UPSCALER_DESIGN.md, "E1b: the HDR encode"):
    /// dump A (Grove Street) scene luma p50 5.9 and p99 19.5 with the game's e 0.1282; dump B
    /// (Vinewood) p50 3.0 and p99 20.9 with e 0.1581. Log-normal frames with those medians and
    /// spreads (sigma = ln(p99 / p50) / 2.326) give an auto e within 30% of the game's.
    #[test]
    fn the_auto_exposure_key_matches_gtas_own_exposure_within_30_percent() {
        let (w, h) = (512u32, 288u32);
        for (name, median, p99, game_e) in [("A", 5.9f64, 19.5f64, 0.1282f64), ("B", 3.0, 20.9, 0.1581)] {
            let sigma = (p99 / median).ln() / 2.326;
            let texels = log_normal(w, h, median, sigma);
            let got = hdr::auto_exposure_reference(&texels, w, h, None);
            let e = f64::from(got.log2_e.exp2());
            let ratio = e / game_e;
            eprintln!("GTA dump {name}: median {median}, sigma {sigma:.3}: auto e {e:.4}, the game's {game_e}: ratio {ratio:.3}");
            assert!((0.7..=1.3).contains(&ratio), "dump {name}: auto e {e} vs the game's {game_e}");
        }
    }

    /// Runs one auto-exposure hold (roundtrip mode, no exposure image) on `texels` and returns it.
    fn auto_hold(gpu: &Gpu, res: &mut Resources, image: vk::Image, w: u32, h: u32, identification: u64) -> HoldResult {
        auto_hold_in(gpu, res, image, (w, h), None, identification)
    }

    /// [`auto_hold`] with DLSS's subrect: `region` at the image's corner ([`Target::region`]).
    fn auto_hold_in(gpu: &Gpu, res: &mut Resources, image: vk::Image, (w, h): (u32, u32), region: Option<(u32, u32)>, identification: u64) -> HoldResult {
        let target = Target {
            colour: image,
            width: w,
            height: h,
            depth: None,
            depth_buffer: None,
            region,
            mvec: None,
            mvec_scale: [1.0, 1.0],
            exposure: [None; MAX_EXPOSURE],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            lift_exposure: false,
            pre_exposed: false,
            identification,
        };
        let (device, queue) = (&gpu.device, gpu.queue);
        let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
        };
        let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, res, &target, Mode::Roundtrip, false, 0, &mut submit) };
        assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{result:?}");
        assert_eq!(result.exposure_source, Some(ExposureSource::Auto));
        result
    }

    /// The GPU auto-exposure (`preupscale_exposure.comp`, through a real hold) against the CPU
    /// reference: on a frame larger than one workgroup tile and not a multiple of it (200x130), for
    /// uniform, sky-and-ground (with NaN, Inf and negative texels) and log-normal frames, the
    /// sample count is exact, the mean log2 luma within 0.01, the exposure the half of the
    /// reference's. Then, on one identification, the next frame (4x brighter) moves 5% of the way
    /// (the persistent state), and a new identification takes the target at once.
    #[test]
    fn the_gpu_auto_exposure_matches_the_cpu_reference_adapts_and_resets() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale auto-exposure test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (200u32, 130u32);
        let (image, memory) = gpu.image(w, h);
        let shm = scratch_shm("auto");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let sky = |x: u32, y: u32| match (x % 16, y % 8) {
            (0, 0) => [f32::NAN, 1.0, 1.0],
            (2, 0) => [1.0, f32::INFINITY, 1.0],
            (4, 0) => [-5.0, -5.0, -5.0],
            _ if y < h / 3 => [40.0, 50.0, 70.0],
            _ => [0.4, 0.5, 0.3],
        };
        let frames = [("uniform", frame(w, h, |_, _| grey(5.9))), ("sky and ground", frame(w, h, sky)), ("log-normal", log_normal(w, h, 3.0, 0.83))];
        let mut identification = 10;
        for (name, texels) in &frames {
            gpu.upload(image, w, h, texels);
            // A new identification each: the first frame takes the target.
            identification += 1;
            let result = auto_hold(&gpu, &mut res, image, w, h, identification);
            let got = result.auto_exposure.expect("the state was read");
            let want = hdr::auto_exposure_reference(texels, w, h, None);
            assert_eq!(got.samples, want.samples, "{name}: samples");
            assert!(got.valid, "{name}");
            assert!((got.mean_log2 - want.mean_log2).abs() < 0.01, "{name}: GPU mean log2 {} vs CPU {}", got.mean_log2, want.mean_log2);
            assert!((got.log2_e - want.log2_e).abs() < 0.01, "{name}: GPU log2 e {} vs CPU {}", got.log2_e, want.log2_e);
            let e = result.exposure.unwrap();
            assert_eq!(e, f16_to_f32(f32_to_f16(got.log2_e.exp2())), "{name}: the exposure buffer holds the state's e as a half");
            eprintln!("auto-exposure {name}: {} samples, mean log2 luma {:.4} (CPU {:.4}), e {e:.5}", got.samples, got.mean_log2, want.mean_log2);
        }
        // Adaptation on the same identification: 4x brighter moves 5% of 2 EV.
        let (_, last) = &frames[2];
        let before = auto_hold(&gpu, &mut res, image, w, h, identification).auto_exposure.unwrap();
        let brighter = frame(w, h, |x, y| {
            let at = ((y * w + x) as usize) * TEXEL as usize;
            let v = f16_to_f32(u16::from_le_bytes([last[at], last[at + 1]]));
            grey(v * 4.0)
        });
        gpu.upload(image, w, h, &brighter);
        let after = auto_hold(&gpu, &mut res, image, w, h, identification).auto_exposure.unwrap();
        let want = hdr::auto_exposure_reference(&brighter, w, h, Some(before.log2_e));
        assert!((after.log2_e - want.log2_e).abs() < 0.01, "adapted: GPU {} vs CPU {}", after.log2_e, want.log2_e);
        assert!((before.log2_e - after.log2_e - 0.05 * 2.0).abs() < 0.02, "5% of 2 EV: {} -> {}", before.log2_e, after.log2_e);
        // A new identification starts over: the target at once.
        let reset = auto_hold(&gpu, &mut res, image, w, h, identification + 1).auto_exposure.unwrap();
        assert!((reset.log2_e - reset.target_log2_e).abs() < 1e-6 && (reset.log2_e - after.log2_e).abs() > 1.5, "{reset:?} after {after:?}");
        // DLSS's subrect: only the rectangle at the corner is measured, never the padding around it (here far brighter).
        let (rw, rh) = (120u32, 70u32);
        let inside = frame(rw, rh, sky);
        let padded = frame(w, h, |x, y| if x < rw && y < rh { sky(x, y) } else { grey(5000.0) });
        gpu.upload(image, w, h, &padded);
        let got = auto_hold_in(&gpu, &mut res, image, (w, h), Some((rw, rh)), identification + 2).auto_exposure.unwrap();
        let want = hdr::auto_exposure_reference(&inside, rw, rh, None);
        assert_eq!(got.samples, want.samples, "the rectangle's samples alone");
        assert!((got.log2_e - want.log2_e).abs() < 0.01, "the rectangle's exposure: GPU {} vs CPU {}", got.log2_e, want.log2_e);
        finish(&gpu, res, &[(image, memory)]);
    }

    /// The Model brightness setting's Auto where the game gives an exposure (`lift_exposure`): the capture copies
    /// the game's texel (1.0 here), the auto-exposure measures the frame and the e the encode uses is the measured
    /// one kept within 0.5x-16x of the game's, with the game's own in the buffer's high half for the
    /// plausibility check. A very dark frame (the measured e far above 16) gets 16, a very bright one 0.5, one in
    /// between its measured e; the source stays the game's.
    #[test]
    fn the_auto_brightness_lifts_the_game_s_exposure_within_its_bounds() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale lifted exposure test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (64u32, 48u32);
        let (image, memory) = gpu.image(w, h);
        let (exposure, exposure_mem) = exposure_image(&gpu, 1.0);
        let shm = scratch_shm("lift");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let (device, queue) = (&gpu.device, gpu.queue);
        // Each frame on a new identification: the measured e is the frame's target at once.
        for (identification, (name, luma, want)) in
            [("dark", 0.0005f32, Some(16.0f32)), ("bright", 50.0, Some(0.5)), ("between", 0.3, None)].into_iter().enumerate()
        {
            let texels = frame(w, h, |_, _| grey(luma));
            gpu.upload(image, w, h, &texels);
            let target = Target {
                colour: image,
                width: w,
                height: h,
                depth: None,
                depth_buffer: None,
                region: None,
                mvec: None,
                mvec_scale: [1.0, 1.0],
                exposure: [None; MAX_EXPOSURE],
                exposure_input: exposure_aux(exposure),
                paper_white: hdr::DEFAULT_PAPER_WHITE,
                lift_exposure: true,
                pre_exposed: false,
                identification: identification as u64 + 1,
            };
            let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
            };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 0, &mut submit) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{name}: {result:?}");
            assert_eq!(result.exposure_source, Some(ExposureSource::Game), "{name}");
            let measured = result.auto_exposure.expect("measured").log2_e.exp2();
            let e = result.exposure.unwrap();
            assert_eq!(result.game_exposure, Some(1.0), "{name}: the game's own e beside the lifted one");
            let want = want.unwrap_or_else(|| f16_to_f32(f32_to_f16(measured)));
            assert!((e - want).abs() <= want * 1e-3, "{name}: e {e}, want {want} (measured {measured})");
            assert!((hdr::LIFT_LO..=hdr::LIFT_HI).contains(&e), "{name}");
        }
        finish(&gpu, res, &[(image, memory), (exposure, exposure_mem)]);
    }

    /// #51, Clair Obscur: Expedition 33: a colour input the game already exposed (a flat 0.2 here) with the
    /// game's eye-adaptation texel beside it (0.0077) is held with the game's exposure, measured beside it, until
    /// the holds of [`PREEXPOSED_WINDOW`] found the frame far below the key; from then on the exposure is
    /// measured from the frame, unbounded by the game's: e 3, the encode's scale (e over the paper white) 1, where
    /// the game's put it at 0.0026. The game's texel is still read, into the exposure buffer's high half
    /// ([`hdr::AutoUse::Beside`]), for the test whether the frame still is pre-exposed. A scene-referred frame (GTA V: about 6, with its exposure of 0.1 or 1) and a
    /// dark one the game's exposure puts far below the key (Crimson Desert at night) keep the game's exposure, used
    /// exactly as it is (the measurement beside it writes nothing the encode reads).
    #[test]
    fn a_pre_exposed_colour_input_is_measured_and_a_scene_referred_one_keeps_the_game_s_exposure() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale pre-exposed test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (64u32, 48u32);
        let (image, memory) = gpu.image(w, h);
        let shm = scratch_shm("preexposed");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let (device, queue) = (&gpu.device, gpu.queue);
        let white = hdr::DEFAULT_PAPER_WHITE;
        for (identification, (name, luma, game, pre_exposed)) in [
            ("pre-exposed (Expedition 33)", 0.2f32, 0.0077f32, true),
            ("GTA V", 6.0, 0.1, false),
            ("GTA V, a brighter exposure", 6.0, 1.0, false),
            ("dark and scene-referred", 0.01, 0.0077, false),
        ]
        .into_iter()
        .enumerate()
        {
            let identification = identification as u64 + 1;
            let texels = frame(w, h, |_, _| grey(luma));
            let (exposure, exposure_mem) = exposure_image(&gpu, game);
            let mut session = Session::default();
            let mut game_holds = 0;
            let mut last = None;
            // A hold every 50 ms of the holds' clock: the window's holds, and five more.
            let step = Duration::from_millis(50);
            let window = (PREEXPOSED_WINDOW.as_millis() / step.as_millis()) as u32 + 1;
            assert!(window >= PREEXPOSED_HOLDS);
            let t0 = Instant::now();
            for k in 0..window + 5 {
                // Each hold on the game's next frame (the roundtrip's write-back changed this one).
                gpu.upload(image, w, h, &texels);
                let target = Target {
                    colour: image,
                    width: w,
                    height: h,
                    depth: None,
                    depth_buffer: None,
                    region: None,
                    mvec: None,
                    mvec_scale: [1.0, 1.0],
                    exposure: [None; MAX_EXPOSURE],
                    // As the device's hold: the game's exposure image only while it is trusted.
                    exposure_input: exposure_aux(exposure).filter(|_| session.game_exposure_trusted(identification)),
                    paper_white: white,
                    lift_exposure: false,
                    pre_exposed: session.pre_exposed_measured(identification),
                    identification,
                };
                let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                    device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
                };
                let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &target, Mode::Roundtrip, false, 0, &mut submit) };
                assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{name}: {result:?}");
                assert!(result.auto_exposure.is_some_and(|a| a.valid && a.samples > 0), "{name}: measured beside the game's: {result:?}");
                assert_eq!(result.exposure_source, Some(ExposureSource::Game), "{name}: the game's texel is read throughout");
                if !target.pre_exposed {
                    game_holds += 1;
                    let e = result.exposure.unwrap();
                    assert!((e - game).abs() <= game * 1e-3, "{name}: the game's exposure as it is, {e}");
                    assert_eq!(result.game_exposure, Some(e), "{name}: nothing in the high half");
                }
                session.check_exposure_at(identification, &result, t0 + step * k);
                last = Some(result);
            }
            let last = last.unwrap();
            assert!(session.game_exposure_trusted(identification), "{name}: never untrusted, only measured beside it");
            if pre_exposed {
                assert_eq!(game_holds, window, "{name}: the game's exposure until the window was over");
                assert!(session.pre_exposed_measured(identification), "{name}");
                let e = last.exposure.unwrap();
                let read = last.game_exposure.unwrap();
                assert!((read - game).abs() <= game * 1e-3, "{name}: the game's {read} in the high half beside the measured {e}");
                let want = hdr::auto_exposure_reference(&texels, w, h, None).log2_e.exp2();
                assert!((e - want).abs() <= want * 2e-3, "{name}: e {e}, the CPU reference's {want}");
                assert!((e / white - 1.0).abs() < 0.02, "{name}: the encode's scale {} (key {} over the frame's {luma})", e / white, hdr::AUTO_KEY);
            } else {
                assert_eq!(game_holds, window + 5, "{name}");
                assert!(!session.pre_exposed_measured(identification), "{name}");
            }
            unsafe {
                gpu.device.destroy_image(exposure, None);
                gpu.device.free_memory(exposure_mem, None);
            }
        }
        finish(&gpu, res, &[(image, memory)]);
    }

    /// A hold of [`Session::check_exposure`]'s pre-exposed test: a frame of log-average luma `luma` with the game's
    /// exposure `e`, the measurement beside it.
    fn pre_exposed_hold(luma: f32, e: f32) -> HoldResult {
        let target = hdr::AUTO_KEY.log2() - luma.log2();
        HoldResult {
            exposure_source: Some(ExposureSource::Game),
            exposure: Some(e),
            auto_exposure: Some(hdr::AutoState { log2_e: target, valid: true, target_log2_e: target, mean_log2: luma.log2(), samples: 100 }),
            ..HoldResult::default()
        }
    }

    /// The holds' clock for [`Session::check_exposure_at`]: a hold every `step`.
    struct HoldClock {
        t0: Instant,
        step: Duration,
        k: u32,
    }

    impl HoldClock {
        /// 60 holds a second.
        fn new() -> Self {
            Self { t0: Instant::now(), step: Duration::from_micros(16_667), k: 0 }
        }

        /// The holds a run of [`PREEXPOSED_WINDOW`] takes, its first included: the last of them can switch.
        fn window(&self) -> u32 {
            PREEXPOSED_WINDOW.as_nanos().div_ceil(self.step.as_nanos()) as u32 + 1
        }

        fn hold(&mut self, session: &mut Session, identification: u64, result: &HoldResult) {
            session.check_exposure_at(identification, result, self.t0 + self.step * self.k);
            self.k += 1;
        }

        fn holds(&mut self, session: &mut Session, identification: u64, n: u32, result: &HoldResult) {
            for _ in 0..n {
                self.hold(session, identification, result);
            }
        }

        /// Nothing held for `gap`.
        fn wait(&mut self, gap: Duration) {
            self.k += gap.as_nanos().div_ceil(self.step.as_nanos()) as u32;
        }
    }

    /// The window of [`Session::check_exposure`]'s pre-exposed test: holds past [`PREEXPOSED_STOPS`] with the frame
    /// near the key for [`PREEXPOSED_WINDOW`], [`PREEXPOSED_HOLDS`] of them at least, the game's exposure steady
    /// within [`PREEXPOSED_STEADY_STOPS`]; one between [`PREEXPOSED_CLEAR_STOPS`] and it keeps the run, one under
    /// it, a frame darker than [`PREEXPOSED_FRAME_STOPS`], a gap or another identification starts it over; a hold
    /// without a measurement changes nothing. The game's exposure stays trusted: only measured beside.
    #[test]
    fn the_pre_exposed_test_needs_its_window() {
        let far = pre_exposed_hold(0.2, 0.0077); // 8.6 stops below, the frame 1.6 stops under the key
        let between = pre_exposed_hold(hdr::AUTO_KEY / (4.5f32.exp2() * 0.0077), 0.0077); // 4.5 stops: a brighter frame, the exposure as it was
        let near = pre_exposed_hold(0.2, hdr::AUTO_KEY / 0.2 / 2.0); // 1 stop
        let dark = pre_exposed_hold(0.01, 0.0077); // far below, but so is the frame itself
        let mut session = Session::default();
        let mut clock = HoldClock::new();
        let window = clock.window();
        clock.holds(&mut session, 1, window - 2, &far);
        clock.hold(&mut session, 1, &between);
        clock.hold(&mut session, 1, &HoldResult { auto_exposure: None, ..pre_exposed_hold(0.2, 0.0077) });
        assert!(!session.pre_exposed_measured(1), "one short");
        clock.hold(&mut session, 1, &far);
        assert!(session.pre_exposed_measured(1), "the window, the one between and the one unmeasured not breaking it");
        assert!(session.game_exposure_trusted(1), "measured beside the game's, which stays read");
        assert!(!session.pre_exposed_measured(2), "only for its identification");
        for (name, breaker) in [("near", &near), ("dark", &dark)] {
            let mut session = Session::default();
            let mut clock = HoldClock::new();
            clock.holds(&mut session, 2, window - 1, &far);
            clock.hold(&mut session, 2, breaker);
            clock.hold(&mut session, 2, &far);
            assert!(!session.pre_exposed_measured(2), "{name} starts the run over");
            clock.hold(&mut session, 3, &far);
            assert!(!session.pre_exposed_measured(3), "{name}: another identification starts over");
        }
        // A gap longer than PREEXPOSED_GAP starts over: nothing was seen in it.
        let mut session = Session::default();
        let mut clock = HoldClock::new();
        clock.holds(&mut session, 4, window / 2, &far);
        clock.wait(PREEXPOSED_GAP * 2);
        clock.holds(&mut session, 4, window / 2 + 2, &far);
        assert!(!session.pre_exposed_measured(4), "a gap");
        clock.holds(&mut session, 4, window, &far);
        assert!(session.pre_exposed_measured(4), "a whole window after it");
        // Sparse holds: the window's time but fewer than PREEXPOSED_HOLDS.
        let mut session = Session::default();
        let mut clock = HoldClock { step: PREEXPOSED_GAP - Duration::from_millis(10), ..HoldClock::new() };
        clock.holds(&mut session, 5, clock.window() + 1, &far);
        assert!(clock.window() < PREEXPOSED_HOLDS && !session.pre_exposed_measured(5), "too few holds");
        clock.holds(&mut session, 5, PREEXPOSED_HOLDS, &far);
        assert!(session.pre_exposed_measured(5), "enough of them");
        // The game's exposure moving by more than PREEXPOSED_STEADY_STOPS over the window (a stop a second here,
        // far below throughout) is an adaptation, not a pre-exposed frame.
        let mut session = Session::default();
        let mut clock = HoldClock::new();
        for k in 0..window * 3 {
            let e = 0.0005 * (k as f32 / 60.0).exp2();
            clock.hold(&mut session, 6, &pre_exposed_hold(0.2, e));
        }
        assert!(!session.pre_exposed_measured(6), "a moving exposure");
        let mut session = Session::default();
        let mut clock = HoldClock::new();
        clock.holds(&mut session, 7, window * 4, &dark);
        clock.holds(&mut session, 8, window * 4, &near);
        assert!(!session.pre_exposed_measured(7) && !session.pre_exposed_measured(8));
    }

    /// #51: a scene-referred game walking from daylight into a dark tunnel (GTA V: e 0.15 on a frame of luma 6, then
    /// a frame of 0.1) puts the frame more than [`PREEXPOSED_STOPS`] below the key while its eye adaptation lags
    /// (here held for 1.5 s, which a count of 30 holds in a row tripped on) and then catches up: never measured.
    /// Expedition 33's pre-exposed frame (luma 0.2) with its eye-adaptation texel steady anywhere in 0.0077-0.019
    /// (a little noise on it) is, after [`PREEXPOSED_WINDOW`].
    #[test]
    fn an_eye_adaptation_catching_up_is_not_a_pre_exposed_frame() {
        let mut session = Session::default();
        let mut clock = HoldClock::new();
        clock.holds(&mut session, 1, 120, &pre_exposed_hold(6.0, 0.15));
        let mut e = 0.15f32;
        for k in 0..600 {
            // 1.5 s at the daylight exposure, then 4% of the way (log2) per hold towards the tunnel's.
            if k >= 90 {
                e = (e.log2() + 0.04 * (6.0f32.log2() - e.log2())).exp2();
            }
            clock.hold(&mut session, 1, &pre_exposed_hold(0.1, e));
            assert!(!session.pre_exposed_measured(1), "hold {k} in the tunnel, e {e}");
        }
        for (identification, game) in [(2, 0.0077f32), (3, 0.012), (4, 0.019)] {
            let mut session = Session::default();
            let mut clock = HoldClock::new();
            let window = clock.window();
            for k in 0..window {
                assert!(!session.pre_exposed_measured(identification), "{game}: hold {k}");
                let noise = [1.0, 1.08, 0.94, 1.03][k as usize % 4];
                clock.hold(&mut session, identification, &pre_exposed_hold(0.2, game * noise));
            }
            assert!(session.pre_exposed_measured(identification), "{game}");
        }
    }

    /// The switch to the measured frame is undone once the frame as given is no longer far below the game's
    /// exposure for [`PREEXPOSED_WINDOW`] (under [`PREEXPOSED_CLEAR_STOPS`], or a frame darker than
    /// [`PREEXPOSED_FRAME_STOPS`]); a hold past [`PREEXPOSED_STOPS`] starts that run over, one between leaves it.
    /// Each direction is logged once per change, at most once per [`PREEXPOSED_SAY_EVERY`]: the rest are counted.
    #[test]
    fn a_pre_exposed_frame_goes_back_to_the_game_s_exposure_once_it_clears() {
        let far = pre_exposed_hold(0.2, 0.0077);
        let between = pre_exposed_hold(hdr::AUTO_KEY / (4.5f32.exp2() * 0.0077), 0.0077);
        // A stop: a frame brighter by as much, the game's exposure as it was (a change of it by over EXPOSURE_JUMP_MAX
        // back and forth would be the jumps that untrust it).
        let near = pre_exposed_hold(hdr::AUTO_KEY / (2.0 * 0.0077), 0.0077);
        let dark = pre_exposed_hold(0.01, 0.0077);
        for (name, clear) in [("near", &near), ("dark", &dark)] {
            let mut session = Session::default();
            let mut clock = HoldClock::new();
            let window = clock.window();
            clock.holds(&mut session, 1, window, &far);
            assert!(session.pre_exposed_measured(1), "{name}");
            assert_eq!(session.exposure_preexposed_said[1], (Some(clock.t0 + clock.step * (window - 1)), 0), "{name}: logged");
            clock.holds(&mut session, 1, window - 1, clear);
            clock.hold(&mut session, 1, &far);
            assert!(session.pre_exposed_measured(1), "{name}: a hold far below starts the run over");
            clock.holds(&mut session, 1, window - 2, clear);
            clock.hold(&mut session, 1, &between);
            assert!(session.pre_exposed_measured(1), "{name}: one short, and one between leaves the run");
            clock.hold(&mut session, 1, clear);
            assert!(!session.pre_exposed_measured(1), "{name}: undone");
            assert!(session.exposure_preexposed_said[0].0.is_some(), "{name}: logged");
            // Back and forth within PREEXPOSED_SAY_EVERY: switched each time, logged once each way.
            for _ in 0..2 {
                clock.holds(&mut session, 1, window, &far);
                assert!(session.pre_exposed_measured(1), "{name}: again");
                clock.holds(&mut session, 1, window, clear);
                assert!(!session.pre_exposed_measured(1), "{name}: undone again");
            }
            assert_eq!(session.exposure_preexposed_said.map(|(_, n)| n), [2, 2], "{name}: the changes not logged");
            clock.wait(PREEXPOSED_SAY_EVERY);
            clock.holds(&mut session, 1, window, &far);
            assert_eq!(session.exposure_preexposed_said[1], (Some(clock.t0 + clock.step * (clock.k - 1)), 0), "{name}: logged again");
        }
    }

    /// Roundtrip with the auto-exposure is the identity the game-exposure roundtrip is: the encode
    /// is the CPU reference with the measured e (the half in the exposure buffer), and the decode,
    /// reading the same buffer, gives the frame back (clamped highlights exactly). Twice, with the
    /// frame changing between holds so e changes (the adaptation), so a decode using any other
    /// frame's e would show.
    #[test]
    fn roundtrip_with_auto_exposure_stays_an_identity() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale auto roundtrip test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (17u32, 9u32);
        let (pw, ph) = padded(w, h);
        let (image, memory) = gpu.image(w, h);
        let shm = scratch_shm("auto-rt");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let white = hdr::DEFAULT_PAPER_WHITE;
        let mut last_e = None;
        for scale in [1.0f32, 8.0] {
            let original: Vec<u8> = {
                let base = hdr_pattern(w, h);
                base.as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                    .flat_map(|(i, b)| {
                        let v = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                        let v = if i % 4 == 3 { v } else { v * scale };
                        f32_to_f16(v).to_le_bytes()
                    })
                    .collect()
            };
            gpu.upload(image, w, h, &original);
            let result = auto_hold(&gpu, &mut res, image, w, h, 1);
            let e = result.exposure.unwrap();
            assert_ne!(Some(e), last_e, "the exposure changed with the frame");
            last_e = Some(e);
            let proxy = proxy_bytes(&shm, pw, ph);
            let clamped = check_encode(&proxy, &original, w, h, e, white);
            let after = gpu.read(image, w, h);
            check_decode(&after, &original, &proxy, &proxy, w, h, e, white);
            // Back to the frame: within what the half floats lose (as the game-exposure roundtrip).
            let mut kept = 0;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let x0 = f64::from(f16_to_f32(half_at(&original, w, x, y, c)));
                        let enc_bits = half_at(&proxy, pw, x, y, c);
                        let enc = f16_to_f32(enc_bits);
                        let got = f64::from(f16_to_f32(half_at(&after, w, x, y, c)));
                        // The decode keeps the whole pixel where its encoded peak channel clamped.
                        let peak = (0..3).map(|k| f16_to_f32(half_at(&proxy, pw, x, y, k))).fold(f32::MIN, f32::max);
                        if peak >= hdr::CLAMPED {
                            kept += usize::from(enc >= hdr::CLAMPED);
                            assert_eq!(got, x0);
                            continue;
                        }
                        let proxy_step = (hdr_ref::decode(f16_to_f32(enc_bits + 1), e, white) - hdr_ref::decode(enc, e, white)).abs();
                        let out_step = f64::from(f16_to_f32(f32_to_f16(x0 as f32) + 1)) - x0;
                        assert!((got - x0).abs() <= 1e-3 * x0 + proxy_step + out_step + 1e-6, "auto roundtrip at {x},{y} channel {c}: {x0} -> {got} (e {e})");
                    }
                    assert_eq!(half_at(&after, w, x, y, 3), half_at(&original, w, x, y, 3));
                }
            }
            assert_eq!(kept, clamped);
            eprintln!("auto roundtrip x{scale}: e={e}, {kept} clamped channels kept");
        }
        finish(&gpu, res, &[(image, memory)]);
    }

    /// What the native hold copies DLSS's depth from: the slot buffer inside DLSS's buffer, else a
    /// readable depth image in a layout a copy can read from; nothing otherwise.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_depth_is_copied_only_where_it_can_be_read() {
        let depth = |readable, layout| Aux { image: vk::Image::null(), format: vk::Format::D32_SFLOAT_S8_UINT, layout, readable };
        let base = Target {
            colour: vk::Image::null(),
            width: 4,
            height: 4,
            depth: None,
            depth_buffer: None,
            region: None,
            mvec: None,
            mvec_scale: [1.0, 1.0],
            exposure: [None; MAX_EXPOSURE],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            lift_exposure: false,
            pre_exposed: false,
            identification: 1,
        };
        assert_eq!(native::depth_source(&base), None);
        let readable = Target { depth: Some(depth(true, Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL))), ..base };
        assert_eq!(native::depth_source(&readable), Some(vk::Format::D32_SFLOAT_S8_UINT));
        assert_eq!(native::depth_source(&Target { depth: Some(depth(false, Some(vk::ImageLayout::GENERAL))), ..base }), None, "no TRANSFER_SRC");
        assert_eq!(native::depth_source(&Target { depth: Some(depth(true, None)), ..base }), None, "no layout known");
        assert_eq!(native::depth_source(&Target { depth: Some(depth(true, Some(vk::ImageLayout::UNDEFINED))), ..base }), None);
        let inside = Target { depth_buffer: Some((vk::Buffer::null(), vk::Format::D16_UNORM)), ..base };
        assert_eq!(native::depth_source(&inside), Some(vk::Format::D16_UNORM));
        // Inside DLSS's buffer: the separate depth layouts only without stencil.
        assert!(depth_copy_layout(vk::Format::D32_SFLOAT, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL));
        assert!(!depth_copy_layout(vk::Format::D32_SFLOAT_S8_UINT, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL));
        assert!(depth_copy_layout(vk::Format::D32_SFLOAT_S8_UINT, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL));
        assert!(!depth_copy_layout(vk::Format::D32_SFLOAT_S8_UINT, vk::ImageLayout::UNDEFINED));
        assert!(depth_copy_layout(vk::Format::R32_SFLOAT, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
    }

    /// The pre-upscaler path makes the game's depth images copyable, as it does RG16F images.
    #[test]
    fn depth_images_are_made_copyable() {
        let info = |format| {
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        };
        let x86 = cfg!(target_arch = "x86_64");
        let src = if x86 { vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::empty() };
        assert_eq!(added_usage(&info(vk::Format::D32_SFLOAT_S8_UINT)), src);
        assert_eq!(added_usage(&info(vk::Format::D16_UNORM)), src);
        let already = vk::ImageCreateInfo { usage: vk::ImageUsageFlags::TRANSFER_SRC, ..info(vk::Format::D32_SFLOAT) };
        assert!(added_usage(&already).is_empty());
        // A sampled RGBA16F render target is a colour candidate: copied in and back by the hold inside DLSS's buffer.
        let both = if x86 { vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST } else { vk::ImageUsageFlags::empty() };
        let target = |format| vk::ImageCreateInfo {
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            mip_levels: 1,
            array_layers: 1,
            ..info(format)
        };
        assert_eq!(added_usage(&target(vk::Format::R16G16B16A16_SFLOAT)), both);
        assert_eq!(added_usage(&target(vk::Format::B10G11R11_UFLOAT_PACK32)), both);
        let storage = vk::ImageCreateInfo { usage: vk::ImageUsageFlags::STORAGE, ..target(vk::Format::R16G16B16A16_SFLOAT) };
        assert_eq!(added_usage(&storage), both, "a storage image is written into");
        // Textures are left as asked: only sampled, or with mip levels or layers.
        let texture =
            vk::ImageCreateInfo { usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST, ..target(vk::Format::R16G16B16A16_SFLOAT) };
        assert!(added_usage(&texture).is_empty(), "only sampled");
        assert!(added_usage(&vk::ImageCreateInfo { mip_levels: 6, ..target(vk::Format::R16G16B16A16_SFLOAT) }).is_empty(), "mip levels");
        assert!(added_usage(&vk::ImageCreateInfo { array_layers: 6, ..target(vk::Format::R16G16B16A16_SFLOAT) }).is_empty(), "layers");
    }

    /// A native capture on a real device: the depth aspect of a D32S8 image in
    /// DEPTH_STENCIL_ATTACHMENT_OPTIMAL into the depth copy (through a transition and back), and a
    /// cache dump's copy of the colour input before; then the write-back's dump copies (the colour
    /// input after, the motion vectors and the depth). Without a readable depth, nothing is copied.
    #[test]
    fn the_capture_copies_the_depth_and_a_dumps_colour_and_the_write_back_reads_them_back() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale depth copy test: no Vulkan device, skipping");
            return;
        };
        // Asked before anything is created, so a skip leaves nothing behind for the device's destruction.
        let depth_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !gpu.supports(vk::Format::D32_SFLOAT_S8_UINT, depth_usage) {
            eprintln!("preupscale depth copy test: no D32_SFLOAT_S8_UINT image here, skipping");
            return;
        }
        let d = &gpu.device;
        let (w, h) = (5u32, 3u32);
        let (image, memory) = gpu.image(w, h);
        let original = pattern(w, h);
        gpu.upload(image, w, h, &original);
        let depth_format = vk::Format::D32_SFLOAT_S8_UINT;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(depth_format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(depth_usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let depth = unsafe { d.create_image(&info, None) }.expect("a D32_SFLOAT_S8_UINT image, which the device supports");
        let reqs = unsafe { d.get_image_memory_requirements(depth) };
        let props = unsafe { gpu.instance.get_physical_device_memory_properties(gpu.physical) };
        let index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
        let depth_mem = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None) }.unwrap();
        unsafe { d.bind_image_memory(depth, depth_mem, 0) }.unwrap();
        let ds_range = aux_range(depth_format);
        let to = |old, new| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth)
                .subresource_range(ds_range)
        };
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)],
            );
            d.cmd_clear_depth_stencil_image(
                cmd,
                depth,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue { depth: 0.75, stencil: 3 },
                &[ds_range],
            );
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)],
            );
        });
        let shm = scratch_shm("depth-copy");
        let mut res = unsafe { Resources::build(d, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let texels = (w * h) as usize;
        let host = |bytes: usize| own_host_buffer(d, &gpu.instance, gpu.physical, bytes as u64).expect("host buffer");
        let (depth_copy, before, after, mvec_src, mvec_back, depth_back) =
            (host(texels * 4), host(texels * 8), host(texels * 8), host(texels * 4), host(texels * 4), host(texels * 4));
        let fill = |b: &HostBuffer, bytes: usize, v: u8| unsafe { std::ptr::write_bytes(b.ptr, v, bytes) };
        let read = |b: &HostBuffer, bytes: usize| unsafe { std::slice::from_raw_parts(b.ptr, bytes) }.to_vec();
        fill(&depth_copy, texels * 4, 0xaa);
        fill(&mvec_src, texels * 4, 0x3c);
        let readable = Target {
            colour: image,
            width: w,
            height: h,
            depth: Some(Aux { image: depth, format: depth_format, layout: Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL), readable: true }),
            depth_buffer: None,
            region: None,
            mvec: None,
            mvec_scale: [1.0, 1.0],
            exposure: [None; MAX_EXPOSURE],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            lift_exposure: false,
            pre_exposed: false,
            identification: 1,
        };
        let run = |res: &Resources, target: &Target, cache: Option<&CacheCopies>| unsafe {
            record_capture(d, res, target, None, None, Some(res.proxy.buffer), None, Some((depth_copy.buffer, None)), cache.map(|c| c.colour), None, None)
                .unwrap();
            record_writeback(d, res, target, res.proxy.buffer, None, cache, None, None).unwrap();
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            d.queue_submit(gpu.queue, &[vk::SubmitInfo::default().command_buffers(&[res.capture_cmd, res.writeback_cmd])], fence).unwrap();
            d.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            d.destroy_fence(fence, None);
        };
        // Not readable: the depth copy is left alone.
        run(&res, &Target { depth: Some(Aux { readable: false, ..readable.depth.unwrap() }), ..readable }, None);
        assert!(read(&depth_copy, texels * 4).iter().all(|&b| b == 0xaa), "an unreadable depth is copied");
        let cache = CacheCopies {
            colour: before.buffer,
            answer: after.buffer,
            mvec: Some((mvec_src.buffer, mvec_back.buffer)),
            depth: Some((depth_copy.buffer, depth_back.buffer)),
        };
        run(&res, &readable, Some(&cache));
        let quarter: Vec<u8> = (0..texels).flat_map(|_| 0.75f32.to_le_bytes()).collect();
        assert_eq!(read(&depth_copy, texels * 4), quarter, "the depth aspect as float32");
        assert_eq!(read(&before, texels * 8), original, "the colour input before, tightly packed");
        // The raw write-back put the proxy (the colour as captured) back: the answer equals it.
        assert_eq!(read(&after, texels * 8), original, "the colour input after the write-back");
        assert_eq!(read(&mvec_back, texels * 4), vec![0x3c; texels * 4]);
        assert_eq!(read(&depth_back, texels * 4), quarter);
        // The depth is back in its layout (validation checks this use).
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageLayout::GENERAL)],
            );
        });
        let mut gpu_ms = None;
        assert!(res.wait_idle(d, &mut gpu_ms));
        unsafe {
            res.destroy(d);
            for b in [&depth_copy, &before, &after, &mvec_src, &mvec_back, &depth_back] {
                b.destroy(d);
            }
            d.destroy_image(image, None);
            d.free_memory(memory, None);
            d.destroy_image(depth, None);
            d.free_memory(depth_mem, None);
        }
    }

    /// Dump mode on a real device: colour (padded), the depth aspect of a D32S8 image that sits in
    /// DEPTH_STENCIL_ATTACHMENT_OPTIMAL (read through a transition and put back), RG16F motion
    /// vectors in GENERAL, and two 1x1 exposure images (RGBA32F in GENERAL, R16F in
    /// SHADER_READ_ONLY_OPTIMAL, read through a transition); no write-back.
    #[test]
    fn a_dump_hold_reads_colour_depth_and_motion_vectors_without_writing_back() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale dump test: no Vulkan device, skipping");
            return;
        };
        // Asked before anything is created, so a skip leaves nothing behind for the device's destruction.
        let depth_format = vk::Format::D32_SFLOAT_S8_UINT;
        if !gpu.supports(depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST) {
            eprintln!("preupscale dump test: no D32_SFLOAT_S8_UINT image here, skipping");
            return;
        }
        let d = &gpu.device;
        let (w, h) = (5u32, 3u32);
        let (image, memory) = gpu.image(w, h);
        let original = pattern(w, h);
        gpu.upload(image, w, h, &original);
        let make_sized = |format: vk::Format, usage: vk::ImageUsageFlags, (w, h): (u32, u32)| {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = unsafe { d.create_image(&info, None) }.ok()?;
            let reqs = unsafe { d.get_image_memory_requirements(image) };
            let props = unsafe { gpu.instance.get_physical_device_memory_properties(gpu.physical) };
            let index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0)?;
            let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None) }.ok()?;
            unsafe { d.bind_image_memory(image, memory, 0) }.ok()?;
            Some((image, memory))
        };
        let make = |format: vk::Format, usage: vk::ImageUsageFlags| make_sized(format, usage, (w, h));
        let (exp32, exp32_mem) = make_sized(vk::Format::R32G32B32A32_SFLOAT, vk::ImageUsageFlags::STORAGE, (1, 1)).expect("1x1 RGBA32F image");
        let (exp16, exp16_mem) = make_sized(vk::Format::R16_SFLOAT, vk::ImageUsageFlags::SAMPLED, (1, 1)).expect("1x1 R16F image");
        let (depth, depth_mem) =
            make(depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT).expect("a D32_SFLOAT_S8_UINT image, which the device supports");
        let (mvec, mvec_mem) = make(vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT).expect("RG16F image");
        let ds_range = aux_range(depth_format);
        let colour_range = aux_range(vk::Format::R16G16_SFLOAT);
        let to = |image, range, old, new| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    to(depth, ds_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    to(mvec, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL),
                    to(exp32, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL),
                    to(exp16, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                ],
            );
            d.cmd_clear_color_image(cmd, exp32, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.5, 2.0, 0.0, 1.0] }, &[colour_range]);
            d.cmd_clear_color_image(cmd, exp16, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [3.0, 0.0, 0.0, 0.0] }, &[colour_range]);
            d.cmd_clear_depth_stencil_image(
                cmd,
                depth,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue { depth: 0.25, stencil: 7 },
                &[ds_range],
            );
            d.cmd_clear_color_image(cmd, mvec, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [1.0, -2.0, 0.0, 0.0] }, &[colour_range]);
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    to(depth, ds_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                    to(exp16, colour_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                ],
            );
        });
        let shm = scratch_shm("dump");
        let mut res = unsafe { Resources::build(d, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let target = Target {
            colour: image,
            width: w,
            height: h,
            depth: Some(Aux { image: depth, format: depth_format, layout: Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL), readable: true }),
            depth_buffer: None,
            region: None,
            mvec: Some(Aux { image: mvec, format: vk::Format::R16G16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }),
            mvec_scale: [1.0, 1.0],
            exposure: [
                Some((Aux { image: exp32, format: vk::Format::R32G32B32A32_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }, true)),
                Some((Aux { image: exp16, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL), readable: true }, false)),
                None,
                Some((Aux { image: exp16, format: vk::Format::R16_SFLOAT, layout: None, readable: true }, false)),
            ],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            lift_exposure: false,
            pre_exposed: false,
            identification: 1,
        };
        let queue = gpu.queue;
        let mut submits = Vec::new();
        let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            submits.push(which);
            d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)
        };
        let mut result = unsafe { run_hold(d, &gpu.instance, gpu.physical, &mut res, &target, Mode::Dump, true, 7, &mut submit) };
        assert_eq!(submits, vec![Which::Capture], "a dump never writes back");
        assert!(!result.wrote_back && result.miss.is_none(), "{result:?}");
        let frame = result.dump.take().expect("dump bytes");
        assert_eq!((frame.width, frame.height, frame.frame), (w, h, 7));
        let (pw, ph) = padded(w, h);
        assert_eq!(frame.colour.len(), (pw * ph) as usize * TEXEL as usize);
        assert_eq!(&frame.colour[..TEXEL as usize], &original[..TEXEL as usize]);
        let (depth_bytes, format) = frame.depth.expect("depth read");
        assert_eq!(format, depth_format);
        assert!(depth_bytes.as_chunks::<4>().0.iter().all(|v| f32::from_le_bytes(*v) == 0.25), "the depth aspect as float32");
        let mv = frame.mvec.expect("motion vectors read");
        assert!(mv.as_chunks::<4>().0.iter().all(|v| *v == [0x00, 0x3c, 0x00, 0xc0]), "RG16F (1, -2)");
        assert_eq!(frame.exposure.len(), 3, "one entry per candidate");
        let rgba32: Vec<u8> = [0.5f32, 2.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            frame.exposure[0],
            ExposureValue {
                image: exp32.as_raw(),
                format: vk::Format::R32G32B32A32_SFLOAT,
                layout: Some(vk::ImageLayout::GENERAL),
                assumed: true,
                bytes: Some(rgba32)
            }
        );
        assert_eq!(frame.exposure[1].bytes.as_deref(), Some(&[0x00, 0x42][..]), "R16F 3.0, read through a transition");
        assert_eq!(frame.exposure[1].layout, Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        assert_eq!((frame.exposure[2].layout, frame.exposure[2].bytes.as_ref()), (None, None), "no layout known, not read");
        assert_eq!(gpu.read(image, w, h), original, "the colour input is untouched");
        // The depth and R16F images are back in their layouts (validation checks this use).
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                all,
                all,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    to(depth, ds_range, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageLayout::GENERAL),
                    to(exp16, colour_range, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL),
                ],
            );
        });
        let mut gpu_ms = None;
        assert!(res.wait_idle(d, &mut gpu_ms));
        unsafe {
            res.destroy(d);
            for (i, m) in [(image, memory), (depth, depth_mem), (mvec, mvec_mem), (exp32, exp32_mem), (exp16, exp16_mem)] {
                d.destroy_image(i, None);
                d.free_memory(m, None);
            }
        }
    }

    /// A destroyed image leaves no exposure name behind, as colour input or as the exposure named: a new image with
    /// its handle starts afresh.
    #[test]
    fn forgetting_an_image_forgets_its_exposure_names() {
        let mut t = Tracker::default();
        let i = vk::Image::from_raw;
        t.exposure_names.insert(i(1), (i(2), false));
        t.exposure_names.insert(i(3), (i(4), true));
        t.exposure_names.insert(i(5), (i(6), false));
        t.forget_image(i(1));
        t.forget_image(i(4));
        assert_eq!(t.exposure_names.keys().copied().collect::<Vec<_>>(), [i(5)]);
    }
}
