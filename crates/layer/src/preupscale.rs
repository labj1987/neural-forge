//! `NEURAL_FORGE_PREUPSCALE`: run the model on the game's own DLSS input, before its upscaler
//! (docs/PRE_UPSCALER_DESIGN.md, "Implementation (layer)").
//!
//! Under vkd3d-proton + DXVK-NVAPI, DLSS Super Resolution reaches Vulkan as CUDA kernels
//! (`vkCmdCuLaunchKernelNVX`) on image views registered through `VK_NVX_image_view_handle`.
//! The probe (`crate::probe_ngx`, docs/PRE_UPSCALER_PROBE.md) showed that in GTA V Enhanced the
//! colour input is final when the launch-bearing submit starts and sits in `GENERAL`. So this
//! module:
//!
//! 1. tracks the registered views and marks launch-bearing command buffers ([`Tracking`]), noting
//!    whether their launches name the colour input ([`LaunchRefs`]: only those are held, so DLSS
//!    Frame Generation's launch-bearing submits go through untouched);
//! 2. identifies the colour input (and depth, motion vectors) from what DLSS Super Resolution's
//!    input kernel names in its parameters ([`input_launch`], [`Tracker::observe`]; also finds
//!    DLAA's output-size input), with the registered set's shapes as the fallback ([`identify`]);
//! 3. at a `vkQueueSubmit`/`vkQueueSubmit2` carrying a launch-bearing buffer, splits the call
//!    around that buffer ([`plan`]) and, between the two halves, runs its own capture submit,
//!    the model round trip and a write-back submit on the same queue ([`run_hold`]).
//!
//! Modes: `model` (the default with the variable unset; it only ever holds on a device that has the
//! NVX extensions DLSS needs and a submit carrying DLSS's input, every other device and frame keeps
//! the post-upscaler path), `off` (`NEURAL_FORGE_PREUPSCALE=off`: nothing here is reachable, no extra
//! hooks, no waits; the hooked-command list is exactly 1.1.0's), `dump`
//! (capture colour, depth, motion vectors and the 1x1 exposure images once and write them to
//! disk), `identity` (capture and write the same bytes back: the hold's own cost,
//! picture-neutral), `model` (the colour input is encoded for the model, [`hdr`], and the helper's
//! answer decoded back over it), `roundtrip` (the same encode and decode with the proxy itself as the
//! answer, no helper: the transform's neutrality). Every failure forwards the game's submit untouched.
//!
//! # The dependency chain of a hold
//!
//! The game's call is submitted as: `head` (batches before the launch batch, plus the launch
//! batch's command buffers before the launch buffer, with the launch batch's wait semaphores),
//! then the layer's capture batch `C`, then (after the CPU waited for `C`'s fence and, in model
//! mode, for the helper) the layer's write-back batch `W`, then `tail` (the launch buffer onward,
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
//!   fence before anyone reads the bytes (the helper only starts after `seq_req` is bumped).
//! - Helper (host writes into the answer region) -> `W`: the helper's writes complete before it
//!   publishes `seq_resp`, which the layer reads with acquire ordering before submitting `W`;
//!   queue submission makes earlier host writes visible to the device.
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

pub(crate) mod hdr;
pub(crate) mod inline;
#[cfg(target_arch = "x86_64")]
pub(crate) mod native;

/// The native backend's per-device network ([`native::Loader`]); none exists on 32-bit builds.
#[cfg(target_arch = "x86_64")]
pub(crate) type NativeLoader = native::Loader;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) enum NativeLoader {}

/// Whether model-mode holds on a device with `loader` go to the native backend.
pub(crate) fn native_on(loader: Option<&NativeLoader>) -> bool {
    #[cfg(target_arch = "x86_64")]
    return loader.is_some() && native::backend() == native::Backend::Native;
    #[cfg(not(target_arch = "x86_64"))]
    return loader.is_some();
}

/// A native hold ([`native::run_native_hold`]).
///
/// # Safety
/// As [`run_hold`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_native(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, res: &mut Resources, target: &Target,
    loader: &NativeLoader, jitter: Option<[f32; 2]>, submit: &mut dyn FnMut(Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>,
) -> HoldResult {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: forwarded.
    return unsafe { native::run_native_hold(device, instance, physical_device, res, target, loader, jitter, native::Conditioning::default(), submit) };
    #[cfg(not(target_arch = "x86_64"))]
    match *loader {}
}

/// The variable that selects the mode. Read through `neural_forge_protocol::env`.
pub(crate) const ENV: &str = "NEURAL_FORGE_PREUPSCALE";

/// How long after the last hold the device counts as holding (`preupscale_state` 2).
pub(crate) const RECENT: Duration = Duration::from_millis(500);

/// In model mode, on a device that has held, how long after the last DLSS submit seen the
/// post-upscaler path stays off (frames are presented as DLSS made them). Covers loading screens,
/// where DLSS does not run: handing back to the post path there rebuilt the helper's feature at the
/// output size and back at every loading screen (twice per screen, at 4K near full VRAM; see
/// "Robustness: failed feature builds" in docs/PRE_UPSCALER_DESIGN.md). Loading screens do not need
/// the model. Longer than this without DLSS (switched off, DLAA) and the post path runs as before.
pub(crate) const HAND_BACK: Duration = Duration::from_secs(30);

/// Model-mode holds in a row without a model answer (an echo, late, or none) that open the
/// [`Breaker`].
pub(crate) const BREAKER_MISSES: u32 = 8;

/// The largest value of the game's exposure image the HDR encode trusts. Measured exposures:
/// GTA V 0.15-0.34, Crimson Desert 0.02-0.06, Cyberpunk 2077 2.7-3.3; Wukong's DLSS-internal 1x1
/// read up to 59,456 ([`Session::check_exposure`]).
pub(crate) const EXPOSURE_TRUST_MAX: f32 = 1000.0;

/// How long an open [`Breaker`] forwards DLSS submits untouched before one probe hold.
pub(crate) const BREAKER_COOL_DOWN: Duration = Duration::from_secs(2);

/// The longest a model-mode hold waits for the helper's answer.
pub(crate) const ANSWER_BUDGET: Duration = Duration::from_millis(30);

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
    /// The HDR encode and its inverse with the answer := the proxy itself (no helper call): checks
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

    /// Whether the mode holds inside DLSS's command buffer where the split is refused
    /// ([`inline`]): every mode but `off`. A `dump` taken there has the colour input (and the
    /// exposure) only: the staging copy does not carry depth and motion vectors.
    pub(crate) fn holds_inline(self) -> bool {
        matches!(self, Mode::Model | Mode::Roundtrip | Mode::Identity | Mode::Dump)
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
    // The kernels' names ([`Kernel`]): only DLSS Super Resolution's input kernel identifies.
    VulkanCommand::CreateCuFunctionNvx,
    VulkanCommand::DestroyCuFunctionNvx,
    // Synchronization a split must not come before or between ([`Hazard`]): event waits, and
    // dynamic rendering suspended in one command buffer and resumed in the next.
    VulkanCommand::CmdWaitEvents,
    VulkanCommand::CmdWaitEvents2,
    VulkanCommand::CmdBeginRendering,
];

/// Whether kernel names gate identification and holds. Off: names differ between DLSS versions
/// (see [`Tracker::record_function`]).
const GATE_BY_KERNEL_NAME: bool = false;

/// What a CUDA kernel is, by the name `vkCreateCuFunctionNVX` gave it. An identification
/// precondition only (docs/PRE_UPSCALER_DESIGN.md, "DLSS Ray Reconstruction"): the colour input is
/// found by the registered handles a launch names, as before; the name says whether that launch is
/// DLSS Super Resolution's input kernel at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// DLSS Super Resolution's input kernel (`hiluma_engine_input_*` in GTA V,
    /// `cuda_engine_input_kernel_*` among NGX's modules): it reads the colour input, depth and
    /// motion vectors.
    SrInput,
    /// The network kernels of DLSS Ray Reconstruction (`custom_block*`, `k_central_block`,
    /// `k_initial_merge`, `custom_upsample*`; Resident Evil Requiem with ray tracing). Frame
    /// Generation shares some of these names (`custom_block0_convPre_kernel`, `k_initial_merge` in
    /// GTA V), so they only mean Ray Reconstruction while no SR input kernel launches.
    RayReconstruction,
    /// Anything else (SR's network and output kernels, FG's, NGX's helpers).
    Other,
}

impl Kernel {
    pub(crate) fn of(name: &str) -> Self {
        if name.starts_with("hiluma_engine_input") || name.starts_with("cuda_engine_input_kernel") {
            Self::SrInput
        } else if ["custom_block", "k_central_block", "k_initial_merge", "custom_upsample"].iter().any(|p| name.starts_with(p)) {
            Self::RayReconstruction
        } else {
            Self::Other
        }
    }
}

/// Whether a kernel is DLSS Frame Generation's by its name: `main_kernel` (Black Myth: Wukong's
/// frame-generation buffers, Resident Evil Requiem's) and the network kernels launched beside it
/// there (`k_conv_fp16_nhwc`, `k_pooling`, `k_upscale`, `k_element_wise`). Everything else reading
/// the colour input first may be the hold point, as before: DLSS Super Resolution's own kernels
/// (Cyberpunk 2077's `cuda_luma_convert_kernel`) and Crimson Desert's (`rr2_*` with Ray
/// Reconstruction on).
pub(crate) fn fg_kernel_name(name: &str) -> bool {
    name == "main_kernel" || ["k_conv_fp16_nhwc", "k_pooling", "k_upscale", "k_element_wise"].iter().any(|p| name.starts_with(p))
}

/// Launch-bearing submits within which an SR input kernel must have launched for the inputs to be
/// identified, once kernel names are known ([`Tracker::sr_running`]). With frame generation at 6x
/// there are about 6 per real frame, so this is about 10 real frames.
pub(crate) const SR_RECENT: u64 = 64;

/// [`COMMANDS`] when `on`, nothing otherwise.
pub(crate) fn commands(on: bool) -> &'static [VulkanCommand] {
    if on { COMMANDS } else { &[] }
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
/// depth-format image is registered at all then. Only taken when an input launch names no image of
/// [`DEPTH_FORMATS`].
const DEPTH_AS_COLOUR: [vk::Format; 2] = [vk::Format::R32_SFLOAT, vk::Format::R32_UINT];

const DEPTH_FORMATS: [vk::Format; 6] = [
    vk::Format::D16_UNORM,
    vk::Format::X8_D24_UNORM_PACK32,
    vk::Format::D32_SFLOAT,
    vk::Format::D16_UNORM_S8_UINT,
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D32_SFLOAT_S8_UINT,
];

fn has_stencil(format: vk::Format) -> bool {
    matches!(format, vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT)
}

/// Bytes per texel of the depth aspect copied out of a depth format.
fn depth_texel_bytes(format: vk::Format) -> u64 {
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

/// At most this many registered 1x1 images are kept as exposure candidates (GTA registers two).
pub(crate) const MAX_EXPOSURE: usize = 4;

/// Bytes reserved per exposure image in the dump readback (the largest texel).
const EXPOSURE_STRIDE: u64 = 16;

/// At most this many further colour candidates are kept beside the first in [`Inputs::others`], for
/// the identification line. Every candidate is in [`Tracker::candidates`].
pub(crate) const MAX_OTHERS: usize = 3;

/// The DLSS inputs among the registered images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inputs {
    /// The colour input: the candidate with the lowest handle. Only a launch-bearing buffer whose
    /// launches name it is held as reading it ([`LaunchRefs::kind`]).
    pub colour: (vk::Image, ImageDesc),
    /// The depth and motion-vector images at the colour input's extent.
    pub depth: (vk::Image, ImageDesc),
    pub mvec: (vk::Image, ImageDesc),
    /// How many colour candidates there were.
    pub candidates: usize,
    /// The further candidates' colour images, lowest handles first. A buffer whose input launch
    /// names one of them with the colour input's depth and motion vectors is held with it as the
    /// target ([`Tracker::retarget`]); one that names it otherwise stays forwarded, and is only
    /// counted ([`Tracker::other_candidate_buffers`]).
    pub others: [Option<(vk::Image, ImageDesc)>; MAX_OTHERS],
    /// What the candidates were judged smaller than: the swapchain's extent, or (no swapchain known
    /// on the device) the largest registered output-like image ([`output_candidate`]).
    pub output: (u32, u32),
    pub output_from_swapchain: bool,
    /// The registered 1x1 float images (DLSS's exposure input among them), lowest handles first.
    /// Only dump mode reads them.
    pub exposure: [Option<(vk::Image, ImageDesc)>; MAX_EXPOSURE],
    /// DLSS's exposure input, which the HDR encode multiplies in: the registered 1x1 R16_SFLOAT
    /// image the input kernel's command buffer names ([`Rule::Params`]), else the registered one
    /// (lowest handle among several). NGX's own 1x1 RGBA32F images are not it.
    pub exposure_input: Option<(vk::Image, ImageDesc)>,
    /// The exposure input is the one DLSS's own command buffer names (not just the registered one).
    pub exposure_named: bool,
    /// Which rule found the colour input.
    pub rule: Rule,
}

/// How the colour input was found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rule {
    /// A launch of DLSS's own command buffer names it together with exactly one depth image and
    /// motion vectors ([`InputLaunch`], [`Tracker::observe`]): what DLSS Super Resolution's input
    /// kernel reads. Works with DLAA (input extent == output extent).
    Params,
    /// The registered set alone ([`identify`]): the RGBA16F storage image beside depth and motion
    /// vectors that is smaller than the output. The fallback (unreadable parameters, other launch
    /// forms, before the parameters have settled).
    Size,
}


/// DLSS's motion-vector formats: RG16F (GTA V) or RG32F. RG16F first where both are at an extent.
const MVEC_FORMATS: [vk::Format; 2] = [vk::Format::R16G16_SFLOAT, vk::Format::R32G32_SFLOAT];

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

/// The largest registered output-like image (a 2D RGBA16F or R11G11B10 storage image): what the
/// colour input is compared against when no swapchain is known on the device (Cyberpunk 2077 under
/// vkd3d-proton logged "swapchain None").
fn output_candidate(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>) -> Option<(u32, u32)> {
    registered
        .values()
        .filter(|(_, d)| d.plain && OUTPUT_FORMATS.contains(&d.format) && d.usage.contains(vk::ImageUsageFlags::STORAGE))
        .map(|(_, d)| (d.width, d.height))
        .max_by_key(|&(w, h)| u64::from(w) * u64::from(h))
}

/// Whether `d` has the shape of DLSS's colour input: a 2D single-sample RGBA16F storage image.
fn colour_like(d: &ImageDesc) -> bool {
    d.plain && d.format == vk::Format::R16G16B16A16_SFLOAT && d.usage.contains(vk::ImageUsageFlags::STORAGE)
}

/// Whether an image of extent `(w, h)` is smaller than `output` (fits inside it, fewer texels).
fn smaller(w: u32, h: u32, (ow, oh): (u32, u32)) -> bool {
    w <= ow && h <= oh && u64::from(w) * u64::from(h) < u64::from(ow) * u64::from(oh)
}

/// The registered image at `(w, h)` with the first format of `formats` present there (lowest
/// handle).
fn at(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, formats: &[vk::Format], w: u32, h: u32) -> Option<(vk::Image, ImageDesc)> {
    formats
        .iter()
        .find_map(|&f| registered.values().find(|(_, d)| d.plain && d.format == f && (d.width, d.height) == (w, h)).copied())
}

/// The colour candidates: every registered RGBA16F storage image smaller than the output (the
/// swapchain, or without one the largest registered output-like image, [`output_candidate`]) with a
/// registered depth image ([`DEPTH_FORMATS`]) and motion-vector image ([`MVEC_FORMATS`]) at the
/// same extent. DLSS's inputs share the render extent; its output and NGX's scratch images have no
/// depth image beside them, and DLAA (render extent == output) is refused by the size test.
/// Deterministic: among several candidates, the lowest handles win (Crimson Desert registers two
/// or three; the others are kept in [`Inputs::others`] for the log). This is the size rule
/// ([`Rule::Size`]), the fallback: what DLSS's own input kernel names ([`Tracker::observe`],
/// [`Rule::Params`]) wins once it has settled.
pub(crate) fn identify(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, swapchain: Option<(u32, u32)>) -> Option<Inputs> {
    let (output, output_from_swapchain) = match swapchain {
        Some(e) => (e, true),
        None => (output_candidate(registered)?, false),
    };
    let candidates = size_candidates(registered, output);
    let colour = *candidates.first()?;
    let (w, h) = (colour.1.width, colour.1.height);
    let mut others = [None; MAX_OTHERS];
    for (slot, other) in others.iter_mut().zip(&candidates[1..]) {
        *slot = Some(*other);
    }
    Some(Inputs {
        colour,
        depth: at(registered, &DEPTH_FORMATS, w, h)?,
        mvec: at(registered, &MVEC_FORMATS, w, h)?,
        candidates: candidates.len(),
        others,
        output,
        output_from_swapchain,
        exposure: exposure_images(registered),
        exposure_input: registered_exposure_input(registered),
        exposure_named: false,
        rule: Rule::Size,
    })
}

/// The size rule's colour candidates ([`identify`]), lowest handles first: every one, where
/// [`Inputs::others`] keeps three for the log.
fn size_candidates(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, output: (u32, u32)) -> Vec<(vk::Image, ImageDesc)> {
    registered
        .values()
        .filter(|(_, d)| {
            colour_like(d)
                && smaller(d.width, d.height, output)
                && at(registered, &DEPTH_FORMATS, d.width, d.height).is_some()
                && at(registered, &MVEC_FORMATS, d.width, d.height).is_some()
        })
        .copied()
        .collect()
}

/// The registered 1x1 float images (DLSS's exposure input among them), lowest handles first.
fn exposure_images(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>) -> [Option<(vk::Image, ImageDesc)>; MAX_EXPOSURE] {
    let mut exposure = [None; MAX_EXPOSURE];
    for (slot, found) in exposure.iter_mut().zip(
        registered.values().filter(|(_, d)| d.plain && (d.width, d.height) == (1, 1) && exposure_texel_bytes(d.format).is_some()),
    ) {
        *slot = Some(*found);
    }
    exposure
}

/// Whether `d` is DLSS's exposure input's kind: a 2D single-sample 1x1 `R16_SFLOAT` image.
fn exposure_like(d: &ImageDesc) -> bool {
    d.plain && (d.width, d.height) == (1, 1) && d.format == vk::Format::R16_SFLOAT
}

/// The registered 1x1 R16_SFLOAT image (lowest handle): the exposure input when DLSS's buffer
/// names none.
fn registered_exposure_input(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>) -> Option<(vk::Image, ImageDesc)> {
    registered.values().find(|(_, d)| exposure_like(d)).copied()
}

/// What DLSS Super Resolution's input kernel reads, as one launch's parameters name it: the colour
/// input, depth and motion vectors. Found by [`input_launch`], kept per command buffer
/// ([`LaunchRefs::input`]) and decided on in [`Tracker::observe`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum InputLaunch {
    /// Exactly one depth image, motion vectors, and exactly one colour candidate at the depth's
    /// extent.
    Inputs { colour: vk::Image, depth: vk::Image, mvec: vk::Image },
    /// Exactly one depth image and motion vectors, but several colour candidates at its extent: not
    /// used, logged once.
    Ambiguous { depth: vk::Image, colours: Vec<(vk::Image, ImageDesc)> },
    /// Depth and motion vectors, but several depth images, or no colour candidate at the depth's
    /// extent (DLSS Frame Generation's launches in GTA V: the output-size frame beside the
    /// render-size depth).
    Unusable,
}

/// Whether `d` can be DLSS's colour input as its input kernel's parameters name it: a 2D
/// single-sample RGBA16F storage image, or an R11G11B10 one that is a storage or a sampled image
/// (Unreal Engine 5 hands DLSS a sampled `B10G11R11_UFLOAT` colour input: Black Myth: Wukong's
/// benchmark, 2026-10-05). The split holds RGBA16F only; the hold inside DLSS's buffer converts.
fn colour_candidate(d: &ImageDesc) -> bool {
    d.plain
        && match d.format {
            vk::Format::R16G16B16A16_SFLOAT => d.usage.contains(vk::ImageUsageFlags::STORAGE),
            vk::Format::B10G11R11_UFLOAT_PACK32 => d.usage.intersects(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED),
            _ => false,
        }
}

/// A sampled, copyable 2D single-sample RGBA16F image: DLSS's colour input in a launch naming no
/// [`colour_candidate`] at the depth's extent ([`input_launch`]); held inside DLSS's buffer only.
fn sampled_colour(d: &ImageDesc) -> bool {
    d.plain
        && d.format == vk::Format::R16G16B16A16_SFLOAT
        && d.usage.contains(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
}

/// Reads one launch's named registered images (`named`; `images` has their descriptions) the way
/// DLSS Super Resolution's input kernel names its inputs. `None` when the launch names no depth
/// image or no 2-channel float image (it is not an input launch). Otherwise exactly one depth image
/// ([`DEPTH_FORMATS`], or without one a single-channel 32-bit image, [`DEPTH_AS_COLOUR`]) and exactly
/// one colour candidate ([`colour_candidate`]) **at the depth's
/// extent** give [`InputLaunch::Inputs`]: DLSS's colour input and depth share the render extent
/// (with DLAA both are the output's), while DLSS Frame Generation reads the output-size frame
/// beside the render-size depth. The colour input need not be smaller than the output. The motion
/// vectors: at the depth's extent first, RG16F before RG32F, lowest handle.
pub(crate) fn input_launch(images: &HashMap<vk::Image, ImageDesc>, named: &[vk::Image]) -> Option<InputLaunch> {
    let of = |keep: fn(&ImageDesc) -> bool| -> Vec<(vk::Image, ImageDesc)> {
        let mut list: Vec<(vk::Image, ImageDesc)> = named.iter().filter_map(|i| images.get(i).filter(|d| keep(d)).map(|d| (*i, *d))).collect();
        list.sort_by_key(|(i, _)| i.as_raw());
        list
    };
    let mvecs = of(|d| d.plain && MVEC_FORMATS.contains(&d.format));
    let mut depths = of(|d| d.plain && DEPTH_FORMATS.contains(&d.format));
    if depths.is_empty() && !mvecs.is_empty() {
        // Depth handed over as a colour image (Streamline's copy): one at a motion-vector or colour
        // candidate's extent.
        let extents: Vec<(u32, u32)> = mvecs.iter().map(|(_, d)| (d.width, d.height)).chain(of(colour_candidate).iter().map(|(_, d)| (d.width, d.height))).collect();
        depths = of(|d| d.plain && DEPTH_AS_COLOUR.contains(&d.format)).into_iter().filter(|(_, d)| extents.contains(&(d.width, d.height))).collect();
    }
    if depths.is_empty() || mvecs.is_empty() {
        return None;
    }
    let [(depth, dd)] = depths[..] else {
        return Some(InputLaunch::Unusable);
    };
    let extent = (dd.width, dd.height);
    // In the order the launch's parameters name them (not by handle).
    let at_extent = |keep: fn(&ImageDesc) -> bool| -> Vec<(vk::Image, ImageDesc)> {
        named.iter().filter_map(|i| images.get(i).filter(|d| keep(d) && (d.width, d.height) == extent).map(|d| (*i, *d))).collect()
    };
    // Without one, a sampled RGBA16F image the layer can copy: GTA San Andreas - The Definitive
    // Edition (Unreal Engine 4's DLSS plugin) hands DLSS a sampled colour input (2026-10-06). Only as
    // a fallback, so the games whose launches name several RGBA16F images keep their pick.
    let mut colours = at_extent(colour_candidate);
    if colours.is_empty() {
        colours = at_extent(sampled_colour);
    }
    let mvec = mvecs
        .iter()
        .min_by_key(|(i, d)| ((d.width, d.height) != extent, MVEC_FORMATS.iter().position(|f| *f == d.format), i.as_raw()))
        .map(|(i, _)| *i)?;
    Some(match colours[..] {
        [] => InputLaunch::Unusable,
        [(colour, _)] => InputLaunch::Inputs { colour, depth, mvec },
        // Several: the first in parameter order is DLSS's colour input (Cyberpunk 2077: dumped, the
        // first is the rendered frame, the second a near-uniform dark buffer, 2026-10-05).
        // `NEURAL_FORGE_PREUPSCALE_PICK` takes another one, for diagnostics.
        _ => match colours.get(pick().unwrap_or(0)) {
            Some(&(colour, _)) => InputLaunch::Inputs { colour, depth, mvec },
            None => InputLaunch::Ambiguous { depth, colours },
        },
    })
}

/// `NEURAL_FORGE_PREUPSCALE_PICK=N` (diagnostics only): when an input launch names several colour
/// candidates at the depth's extent, take the Nth in parameter order (from 0) instead of the first.
/// With `NEURAL_FORGE_PREUPSCALE=dump` it shows which candidate is the game's frame; an N past the
/// candidates leaves the launch ambiguous (not used, logged once).
fn pick() -> Option<usize> {
    static PICK: LazyLock<Option<usize>> = LazyLock::new(|| std::env::var("NEURAL_FORGE_PREUPSCALE_PICK").ok().and_then(|v| v.parse().ok()));
    *PICK
}

/// One command buffer's evidence: its input launch's colour input, depth and motion vectors, and
/// the 1x1 R16_SFLOAT image (DLSS's exposure input) the buffer names, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Named {
    pub colour: vk::Image,
    pub depth: vk::Image,
    pub mvec: vk::Image,
    pub exposure: Option<vk::Image>,
    /// A later launch of a buffer whose input launch names this colour input names it again
    /// together with another colour candidate of its extent ([`LaunchRefs::output_pair`]): DLSS
    /// Super Resolution's output kernel, which reads the input and writes the output.
    pub output_pair: bool,
    /// A submitted launch-bearing buffer whose input launch does not name this colour input still
    /// names it: something besides an input kernel reads it (DLSS Frame Generation's other buffer
    /// reads FG's frame; nothing but SR's own buffer reads SR's input).
    pub foreign: bool,
}

impl Named {
    /// The evidence of an input launch naming `colour`, `depth` and `mvec`, nothing else known yet.
    pub(crate) fn new(colour: vk::Image, depth: vk::Image, mvec: vk::Image, exposure: Option<vk::Image>) -> Self {
        Self { colour, depth, mvec, exposure, output_pair: false, foreign: false }
    }

    /// Whether this output-size entry without an exposure image may still be DLSS Super
    /// Resolution's input (DLAA in a game that gives DLSS no exposure, Resident Evil Requiem): its
    /// buffer's output kernel names it with the output, and no other buffer names it.
    pub(crate) fn sr_without_exposure(&self) -> bool {
        self.output_pair && !self.foreign
    }
}

/// Launch-bearing submits without new evidence before [`Tracker::observe`] decides. With DLSS Frame
/// Generation each real frame has one Super Resolution buffer and two (GTA V's 3x) to about five
/// (Crimson Desert's 6x) FG submits, so this spans a few real frames: both kinds are seen before
/// anything is chosen.
pub(crate) const SETTLE_SUBMITS: u32 = 16;

/// At most this many different colour inputs are kept as evidence.
const MAX_NAMED: usize = 4;

/// The inputs as DLSS's own input kernel names them ([`Rule::Params`]); `None` when one of the
/// images is no longer registered. `size` is what the size rule found; its other candidates are
/// listed beside the colour input (and a forwarded buffer naming one is counted, as before).
fn by_params(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, n: Named, swapchain: Option<(u32, u32)>, size: Option<&Inputs>) -> Option<Inputs> {
    let get = |i: vk::Image| registered.get(&i.as_raw()).copied();
    let colour = get(n.colour)?;
    let rest: Vec<(vk::Image, ImageDesc)> = size
        .into_iter()
        .flat_map(|s| std::iter::once(s.colour).chain(s.others.iter().flatten().copied()))
        .filter(|o| o.0 != n.colour)
        .collect();
    let mut others = [None; MAX_OTHERS];
    for (slot, other) in others.iter_mut().zip(&rest) {
        *slot = Some(*other);
    }
    let (output, output_from_swapchain) = match swapchain {
        Some(e) => (e, true),
        None => (output_candidate(registered).unwrap_or((colour.1.width, colour.1.height)), false),
    };
    let named_exposure = n.exposure.and_then(get);
    // An output-size colour input whose buffer names no exposure (DLAA in a game that gives DLSS
    // none): a registered 1x1 R16F elsewhere is not DLSS's; the exposure is measured from the frame.
    let measured = named_exposure.is_none() && !smaller(colour.1.width, colour.1.height, output);
    Some(Inputs {
        colour,
        // Its own depth image, or (destroyed since: a game rotating its depth images) the registered
        // depth image at the colour input's extent.
        depth: get(n.depth).or_else(|| at(registered, &DEPTH_FORMATS, colour.1.width, colour.1.height))?,
        mvec: get(n.mvec)?,
        candidates: size.map_or(1, |s| s.candidates.max(1 + rest.len())),
        others,
        output,
        output_from_swapchain,
        exposure: exposure_images(registered),
        exposure_input: if measured { None } else { named_exposure.or_else(|| registered_exposure_input(registered)) },
        exposure_named: named_exposure.is_some(),
        rule: Rule::Params,
    })
}

/// Short names for the usage bits that matter here, for [`diagnose`].
fn usage_short(usage: vk::ImageUsageFlags) -> String {
    let names = [
        (vk::ImageUsageFlags::STORAGE, "storage"),
        (vk::ImageUsageFlags::SAMPLED, "sampled"),
        (vk::ImageUsageFlags::COLOR_ATTACHMENT, "colour"),
        (vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, "depth"),
        (vk::ImageUsageFlags::TRANSFER_SRC, "src"),
        (vk::ImageUsageFlags::TRANSFER_DST, "dst"),
    ];
    let list: Vec<&str> = names.iter().filter(|(bit, _)| usage.contains(*bit)).map(|(_, n)| *n).collect();
    if list.is_empty() { "-".to_string() } else { list.join("+") }
}

/// The longest [`diagnose`] text.
pub(crate) const DIAGNOSE_MAX: usize = 1500;

/// At most this many failed identifications per device are logged with [`diagnose`].
const MAX_DIAGNOSES: u32 = 16;

/// Why [`identify`] found nothing, compactly: what the output was taken to be, for every extent
/// with an RGBA16F storage image the first condition it failed, and the registered views grouped by
/// extent, format and usage (largest first, with counts). At most [`DIAGNOSE_MAX`] bytes.
pub(crate) fn diagnose(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, swapchain: Option<(u32, u32)>) -> String {
    let output = swapchain.or_else(|| output_candidate(registered));
    type Key = (std::cmp::Reverse<u64>, u32, u32, i32, u32, bool);
    let mut kinds: BTreeMap<Key, usize> = BTreeMap::new();
    for (_, d) in registered.values() {
        let key = (std::cmp::Reverse(u64::from(d.width) * u64::from(d.height)), d.width, d.height, d.format.as_raw(), d.usage.as_raw(), d.plain);
        *kinds.entry(key).or_default() += 1;
    }
    let mut extents: Vec<(u32, u32)> = registered.values().filter(|(_, d)| colour_like(d)).map(|(_, d)| (d.width, d.height)).collect();
    extents.sort_by_key(|&(w, h)| (std::cmp::Reverse(u64::from(w) * u64::from(h)), w, h));
    extents.dedup();
    const MAX_REASONS: usize = 8;
    let mut reasons: Vec<String> = extents
        .iter()
        .take(MAX_REASONS)
        .map(|&(w, h)| {
            let why = if output.is_none_or(|o| !smaller(w, h, o)) {
                "not smaller than the output"
            } else if at(registered, &DEPTH_FORMATS, w, h).is_none() {
                "no depth image at this extent"
            } else if at(registered, &MVEC_FORMATS, w, h).is_none() {
                "no R16G16/R32G32_SFLOAT motion vectors at this extent"
            } else {
                "qualifies"
            };
            format!("{w}x{h} {why}")
        })
        .collect();
    if extents.len() > MAX_REASONS {
        reasons.push(format!("... (+{} more)", extents.len() - MAX_REASONS));
    }
    let output_text = match (swapchain, output) {
        (Some(s), _) => format!("output: swapchain {}x{}", s.0, s.1),
        (None, Some(o)) => format!("output: no swapchain known, largest RGBA16F/R11G11B10 storage image {}x{}", o.0, o.1),
        (None, None) => "output: no swapchain known and no RGBA16F/R11G11B10 storage image to compare with".to_string(),
    };
    let mut line = format!(
        "{output_text}; RGBA16F storage extents: {}; registered: ",
        if reasons.is_empty() { "none".to_string() } else { reasons.join(", ") }
    );
    let total = kinds.len();
    for (k, (&(_, w, h, format, usage, plain), n)) in kinds.iter().enumerate() {
        let item = format!(
            "{}{w}x{h} {:?} {}{} x{n}",
            if k > 0 { ", " } else { "" },
            vk::Format::from_raw(format),
            usage_short(vk::ImageUsageFlags::from_raw(usage)),
            if plain { "" } else { " (not 2D single-sample)" }
        );
        let more = format!("{}... (+{} more)", if k > 0 { ", " } else { "" }, total - k);
        if line.len() + item.len() + more.len() > DIAGNOSE_MAX {
            line.push_str(&more);
            break;
        }
        line.push_str(&item);
    }
    line
}

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
}

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
        }
    }

    /// A queue-family ownership transfer (its release or its acquire: both name the same pair).
    pub(crate) fn transfers_ownership(&self) -> bool {
        self.src_queue_family != self.dst_queue_family
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

/// Per-device tracking state. Every hook that feeds it only does map operations under the lock.
#[derive(Default)]
pub(crate) struct Tracker {
    images: HashMap<vk::Image, ImageDesc>,
    views: HashMap<vk::ImageView, vk::Image>,
    /// Views registered through `vkGetImageViewHandle*NVX`/`vkGetImageViewAddressNVX`, and their image.
    registered: HashMap<vk::ImageView, vk::Image>,
    swapchains: HashMap<vk::SwapchainKHR, (u32, u32)>,
    /// The registered set or the swapchains changed since the last identification.
    dirty: bool,
    inputs: Option<Inputs>,
    /// What the last identification log line said, so a change is logged once.
    announced: Option<String>,
    /// Command buffers with a `vkCmdCuLaunchKernelNVX` recorded since their last begin (directly or
    /// through executed secondaries), with what their launches' parameters referenced.
    launch: HashMap<vk::CommandBuffer, LaunchRefs>,
    /// The values a view registration returned (`vkGetImageViewHandle*NVX` handles,
    /// `vkGetImageViewAddressNVX` addresses), for every registered view: what a kernel parameter
    /// buffer that reads one of them carries. A launch counts as someone else's only when it names
    /// a registered view and none of them is the colour input.
    keys: Vec<(u64, vk::ImageView, vk::Image)>,
    /// Launch-bearing submits whose buffers all had readable parameters none of which named the
    /// colour input (DLSS Frame Generation's, in GTA V), forwarded untouched.
    pub(crate) foreign_submits: u64,
    /// Held launch-bearing submits whose held buffer reads the colour input.
    pub(crate) colour_submits: u64,
    /// Forwarded (foreign) launch-bearing buffers whose launches name another colour candidate
    /// ([`Inputs::others`]) and not the colour input, and the first such candidate seen: evidence
    /// for a game that alternates DLSS's colour input between images (none is held for it).
    pub(crate) other_candidate_buffers: u64,
    other_named: Option<vk::Image>,
    /// Which first-of-a-kind classification lines were logged.
    said_kinds: u8,
    /// Failed identifications logged with the registered set ([`diagnose`]); at most
    /// [`MAX_DIAGNOSES`], later ones get the short line only.
    diagnoses: u32,
    /// Barriers recorded into each command buffer for the watched images, in recording order,
    /// awaiting submission.
    pending: HashMap<vk::CommandBuffer, Vec<ImageSync>>,
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
    /// The launch just recorded was the first of its command buffer to name a colour candidate:
    /// the buffer and that image ([`Self::inline_point`] decides on a hold inside the buffer there).
    inline_at: Option<(vk::CommandBuffer, vk::Image)>,
    /// Command buffers with a suspending or resuming dynamic render pass instance.
    rendering: HashMap<vk::CommandBuffer, Rendering>,
    /// Launch-bearing submits seen.
    evaluations: u64,
    /// The inputs changed in [`Self::refresh`] and no launch-bearing submit has been scanned since
    /// ([`Scan::identified_now`]).
    identified: bool,
    /// What submitted launch-bearing buffers' input launches named ([`InputLaunch::Inputs`]), one
    /// entry per colour input, at most [`MAX_NAMED`].
    named: Vec<Named>,
    /// Launch-bearing submits since [`Self::named`] last changed.
    named_quiet: u32,
    /// The evidence chosen once it settled ([`SETTLE_SUBMITS`]); [`Self::refresh`] prefers it to
    /// the size rule.
    named_pick: Option<Named>,
    /// Which of [`Self::observe`]'s once-only lines were logged.
    said_named: u8,
    /// Counts identifications: bumped whenever [`Self::refresh`] changes the inputs
    /// ([`Scan::identification`]).
    generation: u64,
    /// The CUDA kernels created on the device (`vkCreateCuFunctionNVX`), with what they are.
    functions: HashMap<vk::CuFunctionNVX, (Kernel, String)>,
    /// A kernel name was ever seen: from then on only DLSS Super Resolution's input kernel
    /// identifies ([`Self::sr_running`]). Without names (no `vkCreateCuFunctionNVX` seen) the rules
    /// work as before.
    names_known: bool,
    /// Launch-bearing submits seen by [`Self::observe`].
    launch_submits: u64,
    /// The last of those carrying an SR input kernel launch, and one carrying a Ray
    /// Reconstruction-family launch.
    last_sr: Option<u64>,
    last_rr: Option<u64>,
    /// The Ray Reconstruction-family kernels seen launching (at most 4 names, for the log).
    rr_names: Vec<String>,
    /// [`Self::sr_running`] as of the last [`Self::observe`] (a change re-runs the identification).
    sr_gate: bool,
    /// The size rule's choice among several candidates overridden by what DLSS's input kernel
    /// names ([`Self::observe`]: the same other candidate in [`SWITCH_AFTER`] consecutive input
    /// launches, or at once when nothing is identified yet). [`Self::refresh`] prefers it while its
    /// colour image is still one of the size rule's candidates.
    switched: Option<Named>,
    /// Every colour candidate of the size rule at the last [`Self::refresh`], lowest handles first
    /// ([`size_candidates`]): what the input kernel may be followed to ([`Self::switch_by_evidence`],
    /// [`Self::retarget`]). Crimson Desert registers up to ten at its render extent once it has
    /// loaded into the world from the title screen, and in play its input kernel reads one past the
    /// three [`Inputs::others`] keeps (2.0.6 regression run, 2026-10-05: never switched to, nothing
    /// held in play).
    pub(crate) candidates: Vec<(vk::Image, ImageDesc)>,
    /// Registered images whose last registered view was destroyed. They stay registered (in
    /// [`Self::registered_set`]) until the image itself is destroyed: Black Myth: Wukong's
    /// benchmark with DLSS Frame Generation on destroys and re-creates the views it hands DLSS every
    /// frame (about 65 a second, the images never), and dropping the image with its view flipped the
    /// identification every frame, so nothing was ever held (2026-10-05).
    kept: HashSet<vk::Image>,
    /// For the running tally line: registered images whose last view was destroyed (kept), and
    /// registered images destroyed.
    pub(crate) unregistered_by_view: u64,
    pub(crate) unregistered_by_image: u64,
    /// Changes of the depth image alone, and gaps without one, taken without re-identifying
    /// ([`Self::refresh`]).
    pub(crate) depth_changes: u64,
    /// Watched images attached in `vkCmdBeginRendering`, per command buffer, with their layout
    /// ([`Self::attachments`]).
    attached: HashMap<vk::CommandBuffer, Vec<(vk::Image, vk::ImageLayout)>>,
    /// Diagnostics: barriers seen on the identified colour input, the last one's new layout, and
    /// whether the in-buffer hold's "layout unknown" line was logged.
    colour_barriers: u64,
    last_colour_layout: Option<vk::ImageLayout>,
    said_layout_miss: bool,
    /// Diagnostics: the kernels seen making a buffer's first launch naming a colour candidate.
    said_first_kernels: Vec<String>,
    /// The probe's parameter-layout lines already written, by kernel name and buffer size
    /// ([`Self::probe_layout`]).
    layouts_said: Vec<(String, usize)>,
    /// The launch-bearing submit at which the current gap without a depth image began: the inputs
    /// are kept for at most [`SR_RECENT`] launch-bearing submits from it.
    depth_gap: Option<u64>,
    /// Consecutive input launches naming the same size-rule candidate other than the colour input.
    streak: Option<(Named, u32)>,
    /// Held launch-bearing submits whose buffer's input launch names another size-rule candidate
    /// (with the colour input's depth and motion vectors): held with that candidate as the target
    /// ([`Self::scan`]). Counted in [`Self::colour_submits`] too.
    pub(crate) retargeted_submits: u64,
    retarget_named: Option<vk::Image>,
    /// The last held submit's target colour image, and the evaluations at which it changed (the
    /// last [`ALTERNATION_WINDOW`]); more than [`ALTERNATION_FLIPS`] there is alternation, logged once.
    last_target: Option<vk::Image>,
    flips: std::collections::VecDeque<u64>,
    alternation: Option<(vk::Image, vk::Image)>,
}

/// Consecutive input launches naming the same other size-rule candidate before the colour input
/// is switched to it ([`Tracker::observe`]).
pub(crate) const SWITCH_AFTER: u32 = 8;

/// Alternation of the hold target between colour candidates is logged when it changes more than
/// [`ALTERNATION_FLIPS`] times within [`ALTERNATION_WINDOW`] held submits.
const ALTERNATION_WINDOW: u64 = 60;
const ALTERNATION_FLIPS: usize = 4;

/// Where a submit's first launch-bearing command buffer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Scan {
    pub batch: usize,
    pub index: usize,
    /// The colour input's committed layout just before the launch buffer, if any barrier on it was
    /// ever seen.
    pub colour_layout: Option<vk::ImageLayout>,
    pub inputs: Option<Inputs>,
    /// The depth and motion-vector images' committed layouts at the same point.
    pub depth_layout: Option<vk::ImageLayout>,
    pub mvec_layout: Option<vk::ImageLayout>,
    /// The exposure candidates' committed layouts, index for index with [`Inputs::exposure`].
    pub exposure_layouts: [Option<vk::ImageLayout>; MAX_EXPOSURE],
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
    /// Why a dump (which also reads depth, the motion vectors and the 1x1 exposure candidates)
    /// must not be taken in front of the launch buffer: a barrier on one of those images only.
    pub dump_hazard: Option<Hazard>,
    /// The queue family that owns the colour input, when a submitted ownership transfer said so.
    pub colour_owner: Option<u32>,
    /// The camera jitter DLSS was given for this frame, when its input kernel's parameters say
    /// ([`LaunchRefs::jitter`]; [`Self::jitter_px`] reads it).
    pub jitter: Option<u64>,
}

impl Scan {
    /// The camera jitter (x, y, render pixels), if known.
    pub(crate) fn jitter_px(&self) -> Option<[f32; 2]> {
        self.jitter.map(|v| [f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32)])
    }

    /// Whether the colour input can be taken to be in `GENERAL` at this submit: its last committed
    /// barrier left it there, or no barrier on it was seen since it was watched. Never on the
    /// submit that (re)identified the inputs ([`Self::identified_now`]): that one goes to DLSS
    /// untouched, and the next is held with known layouts.
    pub(crate) fn colour_in_general(&self) -> bool {
        !self.identified_now && self.colour_layout.is_none_or(|l| l == vk::ImageLayout::GENERAL)
    }
}

/// What one command buffer's `vkCmdCuLaunchKernelNVX` launches referenced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaunchRefs {
    /// A launch whose parameter buffer could not be read.
    pub opaque: bool,
    /// Registered images whose handle (or address) is in a launch's parameters.
    pub images: Vec<vk::Image>,
    /// The first launch (in recording order) that names a depth image and a 2-channel float image,
    /// read as DLSS Super Resolution's input kernel ([`input_launch`]): the first launch of SR's
    /// buffer.
    pub input: Option<InputLaunch>,
    /// A launch after the input launch names its colour input again together with another colour
    /// candidate of the same extent (DLSS Super Resolution's output kernel: input and output).
    pub output_pair: bool,
    /// Whether the input launch's kernel is DLSS Super Resolution's input kernel ([`Kernel`]);
    /// `None` when its name is not known.
    pub input_sr: Option<bool>,
    /// A launch of an SR input kernel, anywhere in the buffer.
    pub sr_input: bool,
    /// The Ray Reconstruction-family kernels launched in the buffer ([`Kernel::RayReconstruction`]),
    /// at most 4 distinct.
    pub rr: Vec<vk::CuFunctionNVX>,
    /// What the buffer held when its first launch was recorded.
    pub before_first: Option<Before>,
    /// The same for the first launch naming each image of [`Self::images`].
    pub before_named: Vec<(vk::Image, Before)>,
    /// The camera jitter the buffer's DLSS Super Resolution input launch was given: word 3 of a 310.x
    /// `hiluma_engine_input*` kernel's parameters, x in its low and y in its high 32 bits, render
    /// pixels (docs/NATIVE_BACKEND.md, "Motion vectors"). `None` for other kernels.
    pub jitter: Option<u64>,
}

/// The word of a 310.x SR input kernel's parameters that holds the camera jitter ([`LaunchRefs::jitter`]).
const HILUMA_JITTER_WORD: usize = 3;

/// What a command buffer held at the point a launch was recorded into it: how many of its
/// barriers on watched images come before the launch ([`Tracker`]'s `pending` list is in recording
/// order), and the global synchronization recorded by then.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Before {
    pub barriers: usize,
    pub global: Option<Hazard>,
}

/// Where a hold runs inside DLSS's own command buffer ([`Tracker::inline_point`],
/// [`inline`]): the colour input the buffer's first colour launch names, DLSS's exposure input as
/// the encode reads it, which identification they belong to, and what refused the split.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InlinePoint {
    pub colour: vk::Image,
    pub desc: ImageDesc,
    /// The colour input's layout at the hold (restored after it).
    pub layout: vk::ImageLayout,
    pub exposure_input: Option<Aux>,
    pub identification: u64,
    pub hazard: Hazard,
}

/// How a launch-bearing command buffer is treated at the submit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaunchKind {
    /// A launch reads the identified colour input: DLSS Super Resolution's buffer, the hold point.
    Colour,
    /// Every launch's parameters were read, they name registered views, and none of them is the
    /// colour input (DLSS Frame Generation's buffers in GTA V): forwarded untouched.
    Foreign,
    /// Not decidable (a launch's parameters were not readable or name no registered view at all,
    /// or no colour input is identified yet): held as before the distinction existed, so nothing
    /// that worked stops working, and a handle form the scan misses cannot turn DLSS Super
    /// Resolution's own buffer into a forwarded one.
    Unknown,
}

impl LaunchRefs {
    pub(crate) fn kind(&self, colour: Option<vk::Image>) -> LaunchKind {
        match colour {
            Some(c) if self.images.contains(&c) => LaunchKind::Colour,
            Some(_) if !self.opaque && !self.images.is_empty() => LaunchKind::Foreign,
            _ => LaunchKind::Unknown,
        }
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
        let before = self.registered.len();
        self.registered.retain(|_, i| *i != image);
        if self.kept.remove(&image) || self.registered.len() != before {
            self.dirty = true;
            self.unregistered_by_image += 1;
        }
        self.committed.remove(&image);
        self.owners.remove(&image);
        // Evidence naming a destroyed image is dropped; the rest settles again before it decides. Not
        // for its depth image: a game rotating its depth images (Black Myth: Wukong with frame
        // generation on, a new one every frame) would drop the evidence every frame; the depth is
        // taken from the registered set instead ([`by_params`]).
        let before = self.named.len();
        self.named.retain(|n| ![n.colour, n.mvec].contains(&image));
        for n in &mut self.named {
            if n.exposure == Some(image) {
                n.exposure = None;
            }
        }
        if self.named.len() != before {
            self.named_quiet = 0;
        }
        if self.named_pick.is_some_and(|n| [n.colour, n.mvec].contains(&image) || n.exposure == Some(image)) {
            self.named_pick = None;
            self.dirty = true;
        }
        if self.switched.is_some_and(|n| [n.colour, n.mvec].contains(&image) || n.exposure == Some(image)) {
            self.switched = None;
            self.dirty = true;
        }
        if self.streak.is_some_and(|(n, _)| [n.colour, n.mvec].contains(&image) || n.exposure == Some(image)) {
            self.streak = None;
        }
    }

    pub(crate) fn record_view(&mut self, view: vk::ImageView, image: vk::Image) {
        self.views.insert(view, image);
    }

    pub(crate) fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view);
        self.keys.retain(|k| k.1 != view);
        // The image stays registered ([`Self::kept`]): nothing changes for the identification.
        if let Some(image) = self.registered.remove(&view) {
            if !self.registered.values().any(|i| *i == image) {
                self.kept.insert(image);
                self.unregistered_by_view += 1;
            }
        }
    }

    /// A view registered through `vkGetImageViewHandle*NVX` / `vkGetImageViewAddressNVX`; `key` is
    /// the handle or address it returned (what a kernel that reads the view is given).
    pub(crate) fn register(&mut self, view: vk::ImageView, key: Option<u64>) {
        if let Some(&image) = self.views.get(&view) {
            let known = self.kept.remove(&image) || self.registered.values().any(|i| *i == image);
            if let Some(old) = self.registered.insert(view, image).filter(|&old| old != image) {
                // A view handle reused for another image: the old image keeps its registration.
                if !self.registered.values().any(|i| *i == old) {
                    self.kept.insert(old);
                }
            }
            if !known {
                self.dirty = true;
            }
            if let Some(key) = key.filter(|&k| k != 0) {
                if !self.keys.iter().any(|k| k.0 == key) {
                    self.keys.push((key, view, image));
                }
            }
        }
    }

    pub(crate) fn swapchain(&mut self, swapchain: vk::SwapchainKHR, extent: Option<(u32, u32)>) {
        match extent {
            Some(e) => self.swapchains.insert(swapchain, e),
            None => self.swapchains.remove(&swapchain),
        };
        self.dirty = true;
    }

    /// `vkCreateCuFunctionNVX` returned `function` for the kernel `name`.
    pub(crate) fn record_function(&mut self, function: vk::CuFunctionNVX, name: &str) {
        self.functions.insert(function, (Kernel::of(name), name.to_string()));
        // Kernel names only feed the logs, never the decision: Crimson Desert's DLSS Super
        // Resolution (a newer DLSS than GTA V's) launches `custom_block*`/`k_initial_merge`
        // kernels and no `hiluma_engine_input*`, so gating on names switched a working game off
        // (2026-10-03). `names_known` stays false until a reliable Ray Reconstruction signature
        // exists.
        self.names_known = GATE_BY_KERNEL_NAME;
    }

    pub(crate) fn forget_function(&mut self, function: vk::CuFunctionNVX) {
        self.functions.remove(&function);
    }

    /// Whether DLSS Super Resolution runs, as far as the identification is concerned: no kernel
    /// name was ever seen (the rules work as before), or an SR input kernel launched within the
    /// last [`SR_RECENT`] launch-bearing submits. Otherwise (DLSS Ray Reconstruction, or frame
    /// generation alone) nothing is identified, so nothing is held: the model must not run on Ray
    /// Reconstruction's noisy input.
    fn sr_running(&self) -> bool {
        !self.names_known || self.last_sr.is_some_and(|s| self.launch_submits.saturating_sub(s) < SR_RECENT)
    }

    /// [`Self::launch_kernel`] without the kernel (its name unknown).
    #[cfg(test)]
    pub(crate) fn launch(&mut self, command_buffer: vk::CommandBuffer, params: Option<&[u8]>) {
        self.launch_kernel(command_buffer, vk::CuFunctionNVX::null(), params);
    }

    /// `vkCmdCuLaunchKernelNVX` of `function` recorded into `command_buffer`; `params` is the
    /// launch's kernel parameter buffer when it could be read ([`launch_params`]), `None`
    /// otherwise. Every 8-byte word of it is compared with the registered views' handles and
    /// addresses. The buffer's first launch naming depth and motion vectors is kept as its input
    /// launch ([`input_launch`]), with whether its kernel is SR's input kernel ([`Kernel`]).
    pub(crate) fn launch_kernel(&mut self, command_buffer: vk::CommandBuffer, function: vk::CuFunctionNVX, params: Option<&[u8]>) {
        let kernel = self.functions.get(&function).map(|(k, _)| *k);
        let before = Before {
            barriers: self.pending.get(&command_buffer).map_or(0, Vec::len),
            global: self.global_sync.get(&command_buffer).copied(),
        };
        {
            let refs = self.launch.entry(command_buffer).or_default();
            refs.before_first.get_or_insert(before);
            refs.sr_input |= kernel == Some(Kernel::SrInput);
            if kernel == Some(Kernel::RayReconstruction) && refs.rr.len() < 4 && !refs.rr.contains(&function) {
                refs.rr.push(function);
            }
        }
        let Some(bytes) = params else {
            self.launch.entry(command_buffer).or_default().opaque = true;
            return;
        };
        if kernel == Some(Kernel::SrInput) {
            probe_params(bytes);
            let hiluma = self.functions.get(&function).is_some_and(|(_, name)| name.starts_with("hiluma_engine_input"));
            if let Some(word) = bytes.chunks_exact(8).nth(HILUMA_JITTER_WORD).filter(|_| hiluma) {
                let value = u64::from_le_bytes(word.try_into().unwrap_or_default());
                self.launch.entry(command_buffer).or_default().jitter.get_or_insert(value);
            }
        }
        let mut named: Vec<vk::Image> = Vec::new();
        for word in bytes.chunks_exact(8) {
            let value = u64::from_le_bytes(word.try_into().unwrap_or_default());
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
        let refs = self.launch.entry(command_buffer).or_default();
        // Whether this launch is the buffer's first one read as an input launch.
        let mut fresh_input = false;
        if let Some(InputLaunch::Inputs { colour, .. }) = refs.input {
            // A later launch naming the colour input with another candidate of its extent.
            let extent = self.images.get(&colour).map(|d| (d.width, d.height));
            if !refs.output_pair && named.contains(&colour) {
                refs.output_pair = named
                    .iter()
                    .any(|i| *i != colour && self.images.get(i).is_some_and(|d| colour_candidate(d) && Some((d.width, d.height)) == extent));
            }
        } else if refs.input.is_none() {
            fresh_input = true;
            refs.input = input_launch(&self.images, &named);
            if refs.input.is_some() {
                refs.input_sr = kernel.map(|k| k == Kernel::SrInput);
            }
            // DLSS Super Resolution's input kernel by name, but not read as an input launch: say once
            // what it named, so a new engine's input form shows in the log.
            let read_as = refs.input.clone();
            if kernel == Some(Kernel::SrInput) && !matches!(read_as, Some(InputLaunch::Inputs { .. })) && self.said_named & 4 == 0 {
                self.said_named |= 4;
                let list: Vec<String> = named
                    .iter()
                    .map(|i| match self.images.get(i) {
                        Some(d) => format!("{}x{} {:?} {}", d.width, d.height, d.format, usage_short(d.usage)),
                        None => "unregistered".into(),
                    })
                    .collect();
                let words: Vec<String> = bytes
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap_or_default()))
                    .filter(|&v| v != 0)
                    .take(64)
                    .map(|v| format!("{v:#x}"))
                    .collect();
                let mut keys: Vec<u64> = self.keys.iter().map(|k| k.0).collect();
                keys.sort_unstable();
                keys.dedup();
                crate::log!(
                    "[preupscale] DLSS's input kernel names {} registered image(s) ({}) and {} parameter word(s); read as {:?}; non-zero words [{}]; {} registered key(s), e.g. [{}]",
                    named.len(),
                    list.join(", "),
                    bytes.len() / 8,
                    read_as,
                    words.join(" "),
                    keys.len(),
                    keys.iter().rev().take(24).map(|k| format!("{k:#x}")).collect::<Vec<_>>().join(" ")
                );
                crate::logging::flush();
            }
        }
        let candidate = |i: &vk::Image| self.inputs.is_some_and(|n| n.colour.0 == *i || n.others.iter().flatten().any(|o| o.0 == *i));
        let had_colour = refs.images.iter().any(candidate);
        // The hold inside the buffer goes only before an input launch naming that colour image with
        // depth and motion vectors (DLSS Super Resolution's input kernel): with DLSS Frame
        // Generation on, Black Myth: Wukong's frame-generation buffers name the colour candidates
        // too, and a hold recorded into each of them took every staging slot within a frame
        // (2026-10-05).
        let input_colour = match refs.input {
            Some(InputLaunch::Inputs { colour, .. }) => Some(colour),
            _ => None,
        };
        // Or this input launch's own colour image, with the identified motion vectors, when it is
        // not a known candidate: Wukong's benchmark scene with frame generation on hands DLSS a new
        // colour (and depth) image every frame, registered after the last identification.
        let own_colour = match refs.input {
            // Smaller than the output only: at DLAA, frame generation's launch has SR's shape at the
            // output size.
            Some(InputLaunch::Inputs { colour, mvec, .. })
                if fresh_input
                    && (refs.input_sr == Some(true) || self.inputs.is_some_and(|n| n.mvec.0 == mvec))
                    && self.images.get(&colour).zip(self.inputs).is_some_and(|(d, n)| smaller(d.width, d.height, n.output)) =>
            {
                Some(colour)
            }
            _ => None,
        };
        // Diagnostics: the kernel of each buffer's first launch naming a colour candidate.
        if !had_colour && named.iter().any(candidate) && self.said_first_kernels.len() < 8 {
            let name = self.functions.get(&function).map_or("unnamed".to_string(), |(_, n)| n.clone());
            let kind = match &refs.input {
                Some(InputLaunch::Inputs { .. }) => "inputs",
                Some(InputLaunch::Unusable) => "unusable",
                Some(InputLaunch::Ambiguous { .. }) => "ambiguous",
                None => "none",
            };
            let key = format!("{name} ({kind}, input launch of this buffer: {})", input_colour.is_some());
            if !self.said_first_kernels.contains(&key) {
                crate::log!("[preupscale] first launch naming a colour candidate: kernel {key}");
                self.said_first_kernels.push(key);
            }
        }
        // The hold goes at the buffer's first launch naming a colour candidate unless that launch is
        // DLSS Frame Generation's by its kernel's name (`main_kernel` in Wukong's frame-generation
        // buffers, which took every staging slot) and not an input launch naming that image.
        let sr_kernel = !self.functions.get(&function).is_some_and(|(_, n)| fg_kernel_name(n));
        let first_colour = if had_colour {
            None
        } else {
            named.iter().copied().find(candidate).filter(|c| input_colour == Some(*c) || sr_kernel).or(own_colour)
        };
        for image in named {
            if !refs.images.contains(&image) {
                refs.images.push(image);
                refs.before_named.push((image, before));
            }
        }
        self.inline_at = first_colour.map(|c| (command_buffer, c));
        if crate::probe_ngx::enabled() {
            self.probe_layout(function, bytes, input_colour);
        }
    }

    /// With `NEURAL_FORGE_PROBE_NGX=1`: logs once per kernel and parameter-buffer size what each
    /// 8-byte word of the launch's parameters holds (docs/DLSS_KERNEL_CATALOGUE.md is built from
    /// these lines). `colour` is the buffer's identified colour input, if any.
    fn probe_layout(&mut self, function: vk::CuFunctionNVX, bytes: &[u8], colour: Option<vk::Image>) {
        const MAX_LAYOUTS: usize = 96;
        let name = self.functions.get(&function).map_or_else(|| "unnamed".to_string(), |(_, n)| n.clone());
        let id = (name, bytes.len());
        if self.layouts_said.len() >= MAX_LAYOUTS || self.layouts_said.contains(&id) {
            return;
        }
        let words: Vec<String> = bytes
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes(w.try_into().unwrap_or_default()))
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
            } else if MVEC_FORMATS.contains(&d.format) {
                "mvec"
            } else if self.inputs.is_some_and(|n| (d.width, d.height) == n.output) {
                "output-size"
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

    /// Whether a hold should run inside `cb`, right before the launch just recorded, which is the
    /// first of the buffer to name `colour` (a colour candidate): the inputs are identified (not on
    /// the submit that identified them), the colour input is an RGBA16F or R11G11B10 image the layer
    /// can transfer (`TRANSFER_SRC | TRANSFER_DST`) at the identified extent, its layout at that
    /// point is known (the buffer's own barriers, else the committed state; an RGBA16F storage image
    /// no barrier was seen on is taken as `GENERAL`, as the split does), no render pass instance is
    /// suspended there, and the split in front of the buffer cannot take it: the split would be
    /// refused ([`Self::launch_hazard`]), or the colour input is not an RGBA16F image in `GENERAL`
    /// (the split reads and writes it as a storage image). A buffer the split can take (GTA V's)
    /// gets nothing.
    pub(crate) fn inline_point(&self, cb: vk::CommandBuffer, colour: vk::Image) -> Option<InlinePoint> {
        let inputs = self.inputs?;
        if self.identified {
            return None;
        }
        let desc = *self.images.get(&colour)?;
        let copyable = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if !OUTPUT_FORMATS.contains(&desc.format)
            || !desc.plain
            || !desc.usage.contains(copyable)
            || (desc.width, desc.height) != (inputs.colour.1.width, inputs.colour.1.height)
        {
            return None;
        }
        if self.rendering.get(&cb).is_some_and(|r| r.last_suspending || r.open_resume) {
            return None;
        }
        let rgba16f = desc.format == vk::Format::R16G16B16A16_SFLOAT;
        let layout = self
            .pending
            .get(&cb)
            .and_then(|p| p.iter().rev().find(|s| s.image == colour).map(|s| s.new_layout))
            .or_else(|| self.attached.get(&cb).and_then(|a| a.iter().rev().find(|(i, _)| *i == colour).map(|(_, l)| *l)))
            .or_else(|| self.committed.get(&colour).copied())
            .or((rgba16f && desc.usage.contains(vk::ImageUsageFlags::STORAGE)).then_some(vk::ImageLayout::GENERAL))?;
        if !INLINE_LAYOUTS.contains(&layout) {
            return None;
        }
        let at = Inputs { colour: (colour, desc), ..inputs };
        let hazard = match self.launch_hazard(cb, Some(&at)).0 {
            Some(h) => h,
            None if rgba16f && layout == vk::ImageLayout::GENERAL => return None,
            None => Hazard::NotSplittable,
        };
        let exposure_input = inputs.exposure_input.map(|(image, d)| {
            let layout = self
                .pending
                .get(&cb)
                .and_then(|p| p.iter().rev().find(|s| s.image == image).map(|s| s.new_layout))
                .or_else(|| self.committed.get(&image).copied())
                .or(d.usage.contains(vk::ImageUsageFlags::STORAGE).then_some(vk::ImageLayout::GENERAL));
            Aux { image, format: d.format, layout, readable: d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) && (d.width, d.height) == (1, 1) }
        });
        Some(InlinePoint { colour, desc, layout, exposure_input, identification: self.generation, hazard })
    }

    /// Takes the mark [`Self::launch_kernel`] left for the launch just recorded.
    pub(crate) fn take_inline_at(&mut self) -> Option<(vk::CommandBuffer, vk::Image)> {
        self.inline_at.take()
    }

    /// Whether any command buffer carries a launch, recorded barriers or other synchronization.
    fn armed(&self) -> bool {
        !self.launch.is_empty() || !self.pending.is_empty() || !self.global_sync.is_empty() || !self.rendering.is_empty() || !self.attached.is_empty()
    }

    pub(crate) fn begin(&mut self, command_buffer: vk::CommandBuffer) {
        self.free(&[command_buffer]);
    }

    pub(crate) fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.launch.remove(cb);
            self.pending.remove(cb);
            self.global_sync.remove(cb);
            self.rendering.remove(cb);
            self.attached.remove(cb);
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
            // Where the secondary's recording starts in the primary's.
            let base = Before {
                barriers: self.pending.get(&primary).map_or(0, Vec::len),
                global: self.global_sync.get(&primary).copied(),
            };
            let shift = |b: Before| Before { barriers: base.barriers + b.barriers, global: base.global.or(b.global) };
            if let Some(c) = self.launch.get(secondary).cloned() {
                let refs = self.launch.entry(primary).or_default();
                refs.opaque |= c.opaque;
                if refs.before_first.is_none() {
                    refs.before_first = c.before_first.map(shift);
                }
                if refs.input.is_none() {
                    refs.input = c.input.clone();
                    refs.input_sr = c.input_sr;
                }
                refs.sr_input |= c.sr_input;
                for f in c.rr {
                    if refs.rr.len() < 4 && !refs.rr.contains(&f) {
                        refs.rr.push(f);
                    }
                }
                if c.input.is_some() && c.input == refs.input {
                    refs.output_pair |= c.output_pair;
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
            if let Some(hazard) = self.global_sync.get(secondary).copied() {
                self.global(primary, hazard);
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

    fn watched(&self, image: vk::Image) -> bool {
        self.inputs.is_some_and(|i| {
            i.colour.0 == image
                || i.others.iter().flatten().any(|o| o.0 == image)
                || i.depth.0 == image
                || i.mvec.0 == image
                || i.exposure.iter().flatten().any(|e| e.0 == image)
                || i.exposure_input.is_some_and(|e| e.0 == image)
        })
    }

    pub(crate) fn barrier(&mut self, command_buffer: vk::CommandBuffer, sync: ImageSync) {
        if self.inputs.is_some_and(|i| i.colour.0 == sync.image) {
            self.colour_barriers += 1;
            self.last_colour_layout = Some(sync.new_layout);
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
                self.committed.insert(sync.image, sync.new_layout);
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
    /// Returns `(hazard, dump_hazard)`. Every hold reads and writes the colour input and reads the
    /// exposure input; only a dump also reads depth, the motion vectors and every 1x1 exposure
    /// candidate, so a barrier on one of those alone stops only a dump. With DLSS Frame Generation
    /// on, GTA V Enhanced copies depth and motion vectors for it inside the launch buffer, between
    /// `GENERAL -> GENERAL` barriers, right before SR's input launch (docs/PRE_UPSCALER_DESIGN.md,
    /// "When a submit is not split").
    fn launch_hazard(&self, cb: vk::CommandBuffer, inputs: Option<&Inputs>) -> (Option<Hazard>, Option<Hazard>) {
        if self.rendering.get(&cb).is_some_and(|r| r.open_resume) {
            return (Some(Hazard::SuspendedRendering), None);
        }
        let Some(refs) = self.launch.get(&cb) else { return (None, None) };
        let colour = inputs.map(|i| i.colour.0);
        let Some(before) = colour.and_then(|c| refs.before_named.iter().find(|(i, _)| *i == c)).map(|(_, b)| *b).or(refs.before_first) else {
            return (None, None);
        };
        if before.global.is_some() {
            return (before.global, None);
        }
        // Unknown inputs: every watched image counts as read by every hold.
        let held = |image: vk::Image| inputs.is_none_or(|i| i.colour.0 == image || i.exposure_input.is_some_and(|e| e.0 == image));
        let dumped = |image: vk::Image| {
            inputs.is_some_and(|i| i.depth.0 == image || i.mvec.0 == image || i.exposure.iter().flatten().any(|e| e.0 == image))
        };
        let Some(recorded) = self.pending.get(&cb) else { return (None, None) };
        let before = &recorded[..before.barriers.min(recorded.len())];
        (
            before.iter().find(|s| held(s.image)).map(ImageSync::hazard),
            before.iter().find(|s| dumped(s.image)).map(ImageSync::hazard),
        )
    }

    /// The largest swapchain's extent.
    fn swapchain_extent(&self) -> Option<(u32, u32)> {
        self.swapchains.values().copied().max_by_key(|&(w, h)| u64::from(w) * u64::from(h))
    }

    /// Re-derives the inputs when the registered set or the swapchains changed. Returns a log line
    /// when the identification changed.
    pub(crate) fn refresh(&mut self) -> Option<String> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        let registered = self.registered_set();
        // What DLSS's own input kernel names, once settled; the size rule otherwise. Neither while
        // kernel names say DLSS Super Resolution is not running ([`Self::sr_running`]).
        let gate = self.sr_running();
        let size = identify(&registered, self.swapchain_extent()).filter(|_| gate);
        self.candidates = match (&size, self.swapchain_extent().or_else(|| output_candidate(&registered))) {
            (Some(_), Some(output)) => size_candidates(&registered, output),
            _ => Vec::new(),
        };
        // The input kernel's choice among the size rule's candidates ([`Self::switched`]) only while
        // its colour image still is one of them.
        if self.switched.is_some_and(|n| !self.is_candidate(n.colour)) {
            self.switched = None;
        }
        let inputs = self
            .switched
            .and_then(|n| by_params(&registered, n, self.swapchain_extent(), size.as_ref()))
            .or_else(|| self.named_pick.filter(|_| gate).and_then(|n| by_params(&registered, n, self.swapchain_extent(), size.as_ref())))
            .or(size);
        // The depth image is not part of what was identified: the model reads the colour input and
        // the exposure only, and the depth's layout only gates a dump ([`Scan::dump_hazard`]). Black
        // Myth: Wukong's benchmark with DLSS Frame Generation on hands DLSS a new depth image every
        // frame (16 in rotation, the colour input, motion vectors and exposure fixed): re-identifying
        // on it skipped every hold (the identifying submit is never held). A depth change alone is
        // taken silently.
        // Between one depth image's destruction and the next one's registration there is no depth
        // image at the render extent, and neither rule identifies anything. While the colour input
        // and motion vectors are still registered and DLSS Super Resolution runs, the inputs are
        // kept: the gap is not a new identification either. The destroyed depth has no layout any
        // more, so a dump cannot read it, and a hold never reads depth.
        let depth_gap = |old: &Inputs| {
            let alive = |image: vk::Image| registered.contains_key(&image.as_raw());
            gate && alive(old.colour.0)
                && alive(old.mvec.0)
                && at(&registered, &DEPTH_FORMATS, old.colour.1.width, old.colour.1.height).is_none()
        };
        let gap_open = self.depth_gap.is_none_or(|since| self.launch_submits.saturating_sub(since) < SR_RECENT);
        if inputs.is_none() && gap_open && self.inputs.is_some_and(|old| depth_gap(&old)) {
            if self.depth_gap.is_none() {
                self.depth_gap = Some(self.launch_submits);
                self.depth_changes += 1;
            }
            // Looked at again at the next launch-bearing submit, until the gap closes or expires.
            self.dirty = true;
            return None;
        }
        if inputs.is_some() {
            self.depth_gap = None;
        }
        let key = |i: Option<Inputs>| i.map(|i| (i.colour.0, i.mvec.0, i.exposure.map(|e| e.map(|e| e.0)), i.exposure_input.map(|e| e.0)));
        // The same inputs by the same rule: only the candidates beside them changed (a game creating
        // and destroying render-size images every frame, Black Myth: Wukong with frame generation on:
        // 97% of its log was this line). Taken silently, after the depth-only change below.
        let unchanged = inputs.is_some() && key(inputs) == key(self.inputs) && inputs.map(|i| i.rule) == self.inputs.map(|i| i.rule);
        if key(inputs) != key(self.inputs) {
            self.committed.clear();
            self.owners.clear();
            self.pending.clear();
            self.identified = true;
            self.generation += 1;
        } else if let (Some(new), Some(old)) = (inputs, self.inputs) {
            if new.depth.0 != old.depth.0 {
                self.inputs = inputs;
                self.depth_changes += 1;
                return None;
            }
        }
        if unchanged && self.announced.is_some() {
            self.inputs = inputs;
            return None;
        }
        self.inputs = inputs;
        let line = match inputs {
            Some(i) => format!(
                "colour input: image {} ({}x{} {:?} {:?}){}, depth {} {:?}, motion vectors {} {:?}{}, exposure input {}{}; swapchain {:?}{}; {}",
                hex(i.colour.0),
                i.colour.1.width,
                i.colour.1.height,
                i.colour.1.format,
                i.colour.1.usage,
                if i.candidates > 1 {
                    let mut others: Vec<String> = i.others.iter().flatten().map(|(image, d)| format!("{} {}x{}", hex(*image), d.width, d.height)).collect();
                    let listed = 1 + others.len();
                    if i.candidates > listed {
                        others.push(format!("{} more", i.candidates - listed));
                    }
                    match i.rule {
                        Rule::Size => format!(" (first of {} candidates; others {})", i.candidates, others.join(", ")),
                        Rule::Params => format!(" (the size rule's other candidates: {})", others.join(", ")),
                    }
                } else {
                    String::new()
                },
                hex(i.depth.0),
                i.depth.1.format,
                hex(i.mvec.0),
                i.mvec.1.format,
                if i.exposure.iter().any(Option::is_some) {
                    let list: Vec<String> = i.exposure.iter().flatten().map(|(image, d)| format!("{} {:?}", hex(*image), d.format)).collect();
                    format!(", 1x1 (exposure) {}", list.join(", "))
                } else {
                    String::new()
                },
                i.exposure_input.map_or_else(|| "none (no registered 1x1 R16_SFLOAT; measured from the frame)".to_string(), |(image, _)| hex(image)),
                if i.exposure_named { " (named by the same command buffer)" } else { "" },
                self.swapchain_extent(),
                if i.output_from_swapchain || i.rule == Rule::Params {
                    String::new()
                } else {
                    format!(", compared with the largest registered RGBA16F/R11G11B10 storage image ({}x{})", i.output.0, i.output.1)
                },
                match i.rule {
                    Rule::Params => "identified by the input kernel's parameters (one launch names it with the depth and motion vectors)",
                    Rule::Size => "identified by size",
                }
            ),
            None if !gate => format!(
                "no DLSS input: no DLSS Super Resolution input kernel (hiluma_engine_input*, cuda_engine_input_kernel*) launched in the last {SR_RECENT} launch-bearing submits{}; waiting",
                if self.rr_names.is_empty() { String::new() } else { format!(" (DLSS Ray Reconstruction's kernels launch: {})", self.rr_names.join(", ")) }
            ),
            None => format!(
                "no DLSS input among {} registered views (swapchain {:?}); waiting",
                registered.len(),
                self.swapchain_extent()
            ),
        };
        // A failed identification says why, with the registered set ([`diagnose`]), once per change
        // of it, for the first [`MAX_DIAGNOSES`] changes; after those only the short line, once per
        // change of the short line (the rule before the diagnosis existed).
        let detailed = gate && inputs.is_none() && !registered.is_empty() && self.diagnoses < MAX_DIAGNOSES;
        let line = if detailed { format!("{line}. {}", diagnose(&registered, self.swapchain_extent())) } else { line };
        if self.announced.as_deref() == Some(line.as_str()) {
            return None;
        }
        if detailed {
            self.diagnoses += 1;
        }
        self.announced = Some(line.clone());
        Some(line)
    }

    /// Gathers what the submit's launch-bearing buffers' input launches name ([`LaunchRefs::input`])
    /// and, once that evidence has not changed for [`SETTLE_SUBMITS`] launch-bearing submits,
    /// decides ([`Self::pick_named`]); a changed decision marks the inputs for [`Self::refresh`].
    /// Called before it at every launch-bearing submit. Returns once-only log lines.
    pub(crate) fn observe(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Vec<String> {
        let mut lines = Vec::new();
        let mut any = false;
        let mut changed = false;
        // This submit's input launches, in submission order, for the switch below.
        let mut evidence: Vec<Named> = Vec::new();
        for cb in batches.iter().flatten() {
            let Some(refs) = self.launch.get(cb) else { continue };
            any = true;
            match &refs.input {
                // With kernel names known, only SR's input kernel is evidence (not Ray
                // Reconstruction's or Frame Generation's first launch, which can have its shape).
                Some(InputLaunch::Inputs { .. }) if self.names_known && refs.input_sr != Some(true) => {}
                Some(InputLaunch::Inputs { colour, depth, mvec }) => {
                    let (colour, depth, mvec) = (*colour, *depth, *mvec);
                    let exposure = refs.images.iter().filter(|i| self.images.get(i).is_some_and(exposure_like)).min_by_key(|i| i.as_raw()).copied();
                    evidence.push(Named::new(colour, depth, mvec, exposure));
                    if let Some(n) = self.named.iter_mut().find(|n| n.colour == colour) {
                        if n.exposure.is_none() && exposure.is_some() {
                            n.exposure = exposure;
                            changed = true;
                        }
                        if !n.output_pair && refs.output_pair {
                            n.output_pair = true;
                            changed = true;
                        }
                    } else if self.named.len() < MAX_NAMED {
                        self.named.push(Named { output_pair: refs.output_pair, ..Named::new(colour, depth, mvec, exposure) });
                        changed = true;
                    }
                }
                Some(InputLaunch::Ambiguous { depth, colours }) if self.said_named & 1 == 0 => {
                    self.said_named |= 1;
                    let list: Vec<String> = colours.iter().map(|(i, d)| format!("{} {}x{} {:?}", hex(*i), d.width, d.height, d.format)).collect();
                    lines.push(format!(
                        "a CUDA launch names depth {} with several colour candidates at its extent, in parameter order ({}): not used to identify DLSS's colour input",
                        hex(*depth),
                        list.join(", ")
                    ));
                }
                _ => {}
            }
        }
        // A buffer whose input launch is not an entry's own but which names that entry's colour
        // input: something besides an input kernel reads it.
        for cb in batches.iter().flatten() {
            let Some(refs) = self.launch.get(cb) else { continue };
            let own = match &refs.input {
                Some(InputLaunch::Inputs { colour, .. }) => Some(*colour),
                _ => None,
            };
            for n in &mut self.named {
                if !n.foreign && own != Some(n.colour) && refs.images.contains(&n.colour) {
                    n.foreign = true;
                    changed = true;
                }
            }
        }
        if !any {
            return lines;
        }
        // Whether DLSS Super Resolution runs (by its input kernel's name) or Ray Reconstruction.
        self.launch_submits += 1;
        let now = self.launch_submits;
        for cb in batches.iter().flatten() {
            let Some(refs) = self.launch.get(cb) else { continue };
            if refs.sr_input {
                self.last_sr = Some(now);
            }
            if !refs.rr.is_empty() {
                self.last_rr = Some(now);
            }
            for f in &refs.rr {
                if let Some((_, name)) = self.functions.get(f) {
                    if self.rr_names.len() < 4 && !self.rr_names.contains(name) {
                        self.rr_names.push(name.clone());
                    }
                }
            }
        }
        let gate = self.sr_running();
        if gate != self.sr_gate {
            self.sr_gate = gate;
            self.dirty = true;
        }
        if !gate && self.last_rr == Some(now) && self.said_named & 64 == 0 {
            self.said_named |= 64;
            lines.push(format!(
                "DLSS Ray Reconstruction detected ({} launched, no DLSS Super Resolution input kernel); the model can't run before it (its input is the noisy ray-traced frame): nothing is identified or held, the model runs after the upscaler",
                self.rr_names.join(", ")
            ));
        }
        self.switch_by_evidence(&evidence, &mut lines);
        self.named_quiet = if changed { 0 } else { self.named_quiet.saturating_add(1) };
        // Decided when the evidence has just settled, and again when the registered set or the
        // swapchains change after that (the output the colour input is compared with may move).
        if self.named_quiet == SETTLE_SUBMITS || (self.named_quiet > SETTLE_SUBMITS && self.dirty) {
            let pick = self.pick_named(&mut lines);
            if pick != self.named_pick {
                self.named_pick = pick;
                self.dirty = true;
            }
        }
        lines
    }

    /// Overrides the size rule's choice among several colour candidates with the one DLSS's input
    /// kernel names (`evidence`: this submit's input launches). With nothing identified yet, at
    /// once; otherwise when the same other candidate is named by [`SWITCH_AFTER`] consecutive input
    /// launches (one naming the colour input resets the count, so a game alternating between two
    /// candidates never switches: [`Self::scan`] holds each submit with the one its buffer names).
    /// Only the size rule's candidates (render-size, smaller than the output, with depth and motion
    /// vectors at their extent) are switched between: a single candidate (GTA V) and DLAA (none)
    /// are never affected.
    fn switch_by_evidence(&mut self, evidence: &[Named], lines: &mut Vec<String>) {
        if evidence.is_empty() || !self.sr_running() {
            return;
        }
        let Some(current) = self.inputs else {
            // The first identification: what the input kernel names among the size rule's
            // candidates, rather than the lowest handle.
            let registered = self.registered_set();
            let Some(size) = identify(&registered, self.swapchain_extent()) else { return };
            let all = size_candidates(&registered, size.output);
            if let Some(n) = evidence.iter().find(|n| n.colour != size.colour.0 && all.iter().any(|c| c.0 == n.colour)) {
                self.switched = Some(*n);
                self.dirty = true;
            }
            return;
        };
        for n in evidence {
            if n.colour == current.colour.0 {
                self.streak = None;
                continue;
            }
            if !self.is_candidate(n.colour) {
                continue;
            }
            let count = match self.streak {
                Some((s, k)) if s.colour == n.colour => k + 1,
                _ => 1,
            };
            self.streak = Some((*n, count));
            if count >= SWITCH_AFTER {
                self.streak = None;
                self.switched = Some(*n);
                self.dirty = true;
                lines.push(format!(
                    "colour input switched to {} (DLSS's input kernel reads it; the {} had picked {})",
                    hex(n.colour),
                    match current.rule {
                        Rule::Size => "size rule",
                        Rule::Params => "input kernel's earlier parameters",
                    },
                    hex(current.colour.0)
                ));
                return;
            }
        }
    }

    /// Whether `image` is one of the size rule's colour candidates ([`Self::candidates`]).
    fn is_candidate(&self, image: vk::Image) -> bool {
        self.candidates.iter().any(|c| c.0 == image)
    }

    /// The registered images with their descriptions, keyed by raw handle (what [`identify`] reads).
    fn registered_set(&self) -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        self.registered
            .values()
            .chain(&self.kept)
            .filter_map(|&image| self.images.get(&image).map(|d| (image.as_raw(), (image, *d))))
            .collect()
    }

    /// The evidence to identify by. An entry whose colour input is smaller than the output (the
    /// swapchain; without one the largest registered RGBA16F/R11G11B10 storage image) counts as it
    /// is: DLSS Frame Generation's launches never name a render-size frame beside the render-size
    /// depth. An entry whose colour input is the output's size (DLAA, or frame generation at native
    /// resolution, whose launch names the output-size frame beside output-size depth and motion
    /// vectors) counts if its buffer also names a 1x1 R16_SFLOAT image: DLSS Super Resolution's
    /// exposure input, which frame generation does not take. Without one (a game that gives DLSS no
    /// exposure: Resident Evil Requiem at DLAA) it counts only with SR's own shape
    /// ([`Named::sr_without_exposure`]): a later launch of its buffer names it with the output (SR's
    /// output kernel), and no other launch-bearing buffer names it (FG's frame is read by FG's other
    /// buffer too) — and only when exactly one such entry exists. Among several counted entries,
    /// the only one whose buffer names the exposure; otherwise none, logged once, and the size rule
    /// decides.
    fn pick_named(&mut self, lines: &mut Vec<String>) -> Option<Named> {
        let describe = |t: &Self, list: &[Named]| -> String {
            let list: Vec<String> = list
                .iter()
                .map(|n| {
                    let size = t.images.get(&n.colour).map_or_else(String::new, |d| format!(" {}x{}", d.width, d.height));
                    format!("{}{size}{}", hex(n.colour), n.exposure.map_or_else(String::new, |e| format!(" with exposure {}", hex(e))))
                })
                .collect();
            list.join(", ")
        };
        let output = self.swapchain_extent().or_else(|| output_candidate(&self.registered_set()));
        let (mut counted, output_size): (Vec<Named>, Vec<Named>) = self.named.iter().partition(|n| {
            n.exposure.is_some() || self.images.get(&n.colour).zip(output).is_some_and(|(d, o)| smaller(d.width, d.height, o))
        });
        // Output-size entries without an exposure image: SR's shape (DLAA, no exposure given to
        // DLSS), counted when it is the only one; the rest (FG's frame) not used.
        let (sr_like, refused): (Vec<Named>, Vec<Named>) = output_size.iter().partition(|n| n.sr_without_exposure());
        if let [one] = sr_like[..] {
            counted.push(one);
            if self.said_named & 16 == 0 {
                self.said_named |= 16;
                lines.push(format!(
                    "a CUDA launch names an output-size colour image with depth and motion vectors and its command buffer names no 1x1 R16_SFLOAT exposure ({}): taken as DLSS Super Resolution's input at DLAA, because a later launch of the same buffer names it with the output (SR's output kernel) and no other launch-bearing buffer names it; the exposure is measured from the frame",
                    describe(self, &sr_like)
                ));
            }
        } else if sr_like.len() > 1 && self.said_named & 32 == 0 {
            self.said_named |= 32;
            lines.push(format!(
                "several output-size colour images have SR's shape without a 1x1 R16_SFLOAT exposure ({}): none used",
                describe(self, &sr_like)
            ));
        }
        let refused: Vec<Named> = if sr_like.len() > 1 { output_size.clone() } else { refused };
        if !refused.is_empty() && self.said_named & 8 == 0 {
            self.said_named |= 8;
            let why: Vec<String> = refused
                .iter()
                .map(|n| {
                    let mut reasons = Vec::new();
                    if !n.output_pair {
                        reasons.push("no later launch of its buffer names it with another output-size colour image");
                    }
                    if n.foreign {
                        reasons.push("another launch-bearing buffer names it too");
                    }
                    if reasons.is_empty() {
                        reasons.push("another output-size image has the same shape");
                    }
                    format!("{}: {}", hex(n.colour), reasons.join(", "))
                })
                .collect();
            lines.push(format!(
                "a CUDA launch names an output-size colour image with depth and motion vectors ({}), but its command buffer names no 1x1 R16_SFLOAT exposure: not DLSS Super Resolution's input (DLSS Frame Generation's frame, or SR without an exposure input); not used ({})",
                describe(self, &refused),
                why.join("; ")
            ));
        }
        match counted[..] {
            [] => None,
            [one] => Some(one),
            _ => {
                let with: Vec<Named> = counted.iter().filter(|n| n.exposure.is_some()).copied().collect();
                if let [one] = with[..] {
                    if self.said_named & 2 == 0 {
                        self.said_named |= 2;
                        lines.push(format!(
                            "input launches in different command buffers name different colour inputs ({}): {} chosen, the one whose buffer names the 1x1 R16_SFLOAT exposure",
                            describe(self, &counted),
                            hex(one.colour)
                        ));
                    }
                    Some(one)
                } else {
                    if self.said_named & 4 == 0 {
                        self.said_named |= 4;
                        lines.push(format!(
                            "input launches in different command buffers name different colour inputs ({}): none chosen by the input kernel's parameters, the size rule decides",
                            describe(self, &counted)
                        ));
                    }
                    None
                }
            }
        }
    }

    /// Finds the first launch-bearing command buffer in a submit (`batches`: each batch's command
    /// buffers, in order) and reads the watched images' layouts just before it: the committed ones
    /// with the barriers of the buffers ahead of it in this submit applied on top, in submission
    /// order. Nothing is committed here: the submit has not happened yet, and may fail
    /// ([`Self::commit_submitted`]).
    pub(crate) fn scan(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        if !self.armed() {
            return None;
        }
        // What this submit's buffers so far would leave behind.
        let mut proposed: HashMap<vk::Image, (vk::ImageLayout, Option<u32>)> = HashMap::new();
        let mut found: Option<Scan> = None;
        let mut any_foreign = false;
        let mut held_kind = None;
        let mut held_target: Option<(vk::Image, bool)> = None;
        for (bi, cbs) in batches.iter().enumerate() {
            for (ci, &cb) in cbs.iter().enumerate() {
                let colour = self.inputs.map(|i| i.colour.0);
                // With kernel names known, a buffer that launches no SR input kernel is never DLSS
                // Super Resolution's (Ray Reconstruction's, Frame Generation's): forwarded.
                let kind = self.launch.get(&cb).map(|r| if self.names_known && !r.sr_input { LaunchKind::Foreign } else { r.kind(colour) });
                // A buffer whose input launch names another of the size rule's candidates with the
                // colour input's depth and motion vectors: DLSS reads that candidate this frame
                // (a game alternating its input between images). Held with it as the target.
                let retarget = (kind == Some(LaunchKind::Foreign)).then(|| self.retarget(cb)).flatten();
                let kind = if retarget.is_some() { Some(LaunchKind::Colour) } else { kind };
                if kind == Some(LaunchKind::Foreign) {
                    any_foreign = true;
                    // Counted only: whether a forwarded buffer reads another colour candidate.
                    let other = self.inputs.and_then(|i| {
                        let refs = self.launch.get(&cb)?;
                        self.candidates.iter().map(|c| c.0).filter(|&c| c != i.colour.0).find(|c| refs.images.contains(c))
                    });
                    if let Some(other) = other {
                        self.other_candidate_buffers += 1;
                        self.other_named.get_or_insert(other);
                    }
                }
                if found.is_none() && kind.is_some_and(|k| k != LaunchKind::Foreign) {
                    held_kind = kind;
                    held_target = retarget.or(self.inputs.map(|i| i.colour)).map(|c| (c.0, retarget.is_some()));
                    let layout = |image: Option<vk::Image>| image.and_then(|i| proposed.get(&i).map(|p| p.0).or(self.committed.get(&i).copied()));
                    let inputs = self.inputs.map(|i| Inputs { colour: retarget.unwrap_or(i.colour), ..i });
                    let colour_owner = inputs.and_then(|i| proposed.get(&i.colour.0).and_then(|p| p.1).or(self.owners.get(&i.colour.0).copied()));
                    let (hazard, dump_hazard) = self.launch_hazard(cb, inputs.as_ref());
                    found = Some(Scan {
                        batch: bi,
                        index: ci,
                        colour_layout: layout(inputs.map(|i| i.colour.0)),
                        inputs,
                        depth_layout: layout(self.inputs.map(|i| i.depth.0)),
                        mvec_layout: layout(self.inputs.map(|i| i.mvec.0)),
                        exposure_layouts: std::array::from_fn(|k| layout(self.inputs.and_then(|i| i.exposure[k]).map(|e| e.0))),
                        exposure_input_layout: layout(self.inputs.and_then(|i| i.exposure_input).map(|e| e.0)),
                        evaluation: self.evaluations,
                        identified_now: self.identified,
                        identification: self.generation,
                        hazard,
                        dump_hazard,
                        colour_owner,
                        jitter: self.launch.get(&cb).and_then(|r| r.jitter),
                    });
                }
                if found.is_none() {
                    for sync in self.pending.get(&cb).into_iter().flatten() {
                        let owner = sync.transfers_ownership().then_some(sync.dst_queue_family).or(proposed.get(&sync.image).and_then(|p| p.1));
                        proposed.insert(sync.image, (sync.new_layout, owner));
                    }
                }
            }
        }
        if found.is_some() {
            self.evaluations += 1;
            self.identified = false;
            if held_kind == Some(LaunchKind::Colour) {
                self.colour_submits += 1;
            }
            if let Some((target, retargeted)) = held_target.filter(|_| held_kind == Some(LaunchKind::Colour)) {
                self.retargeted_submits += u64::from(retargeted);
                if retargeted {
                    self.retarget_named.get_or_insert(target);
                }
                self.note_target(target);
            }
        } else if any_foreign {
            self.foreign_submits += 1;
        }
        found
    }

    /// The size rule's other candidate `cb`'s input launch names, with the colour input's depth and
    /// motion vectors (so their layouts are watched), if any. With kernel names known, only an SR
    /// input kernel's launch.
    fn retarget(&self, cb: vk::CommandBuffer) -> Option<(vk::Image, ImageDesc)> {
        let i = self.inputs?;
        let refs = self.launch.get(&cb)?;
        if self.names_known && refs.input_sr != Some(true) {
            return None;
        }
        let Some(InputLaunch::Inputs { colour, mvec, .. }) = refs.input else { return None };
        // Any depth: a game may hand DLSS a new depth image every frame (Black Myth: Wukong with
        // frame generation on). The motion vectors must be the identified ones, unless the launch is
        // DLSS Super Resolution's input kernel by name: Wukong's benchmark scene names other motion
        // vectors than the ones identified in its intro, whose colour image lives on.
        if colour == i.colour.0 || (refs.input_sr != Some(true) && mvec != i.mvec.0) {
            return None;
        }
        // A candidate of the size rule, or (a new image every frame, registered after the last
        // identification) a colour candidate at the colour input's extent.
        self.candidates.iter().find(|c| c.0 == colour).copied().or_else(|| {
            self.images
                .get(&colour)
                .filter(|d| colour_candidate(d) && (d.width, d.height) == (i.colour.1.width, i.colour.1.height) && smaller(d.width, d.height, i.output))
                .map(|d| (colour, *d))
        })
    }

    /// Books a held submit's target colour image; logs (once) when it alternates.
    fn note_target(&mut self, target: vk::Image) {
        let now = self.evaluations;
        if let Some(last) = self.last_target.filter(|&l| l != target) {
            self.flips.push_back(now);
            while self.flips.front().is_some_and(|&f| now.saturating_sub(f) >= ALTERNATION_WINDOW) {
                self.flips.pop_front();
            }
            if self.flips.len() > ALTERNATION_FLIPS && self.alternation.is_none() {
                self.alternation = Some((last, target));
            }
        }
        self.last_target = Some(target);
    }

    /// A first-of-its-kind classification line, once per kind: a launch-bearing submit held because
    /// a launch reads the colour input; one forwarded because its launches name other images only;
    /// one held undecided (a launch's parameters could not be read, or no colour input yet: the rule
    /// before the distinction). `undecided_before`: held-undecided submits before this scan.
    fn classify_line(&mut self, scan: Option<&Scan>, undecided_before: u64) -> Option<String> {
        let undecided = self.evaluations - self.colour_submits;
        let (bit, line) = if scan.is_some() && self.colour_submits > 0 && self.said_kinds & 1 == 0 {
            (1, "a launch-bearing submit reads DLSS's colour input (its registered handle is in a launch's parameters): DLSS Super Resolution's, the hold point".to_string())
        } else if scan.is_none() && self.foreign_submits > 0 && self.said_kinds & 2 == 0 {
            (2, "a launch-bearing submit whose CUDA launches never name DLSS's colour input (DLSS Frame Generation's, or another NGX feature's) is forwarded untouched; only the one that reads it is held".to_string())
        } else if scan.is_some() && undecided > undecided_before && self.inputs.is_some() && self.said_kinds & 4 == 0 {
            (4, "a launch-bearing submit is held undecided: a launch's parameters were not readable (not in CUDA's buffer form), so it cannot be told from DLSS Super Resolution's".to_string())
        } else if scan.is_some() && self.retargeted_submits > 0 && self.said_kinds & 16 == 0 {
            (
                16,
                format!(
                    "a launch-bearing buffer's input launch names another colour candidate ({}) with the colour input's depth and motion vectors: held with that candidate (DLSS's input kernel reads it this frame)",
                    self.retarget_named.map_or_else(String::new, hex)
                ),
            )
        } else if let Some((a, b)) = self.alternation.filter(|_| self.said_kinds & 32 == 0) {
            (
                32,
                format!(
                    "DLSS's input kernel alternates between colour candidates {} and {} (more than {ALTERNATION_FLIPS} changes in {ALTERNATION_WINDOW} held submits): each submit is held with the candidate its input launch names",
                    hex(a),
                    hex(b)
                ),
            )
        } else if self.other_candidate_buffers > 0 && self.said_kinds & 8 == 0 {
            (
                8,
                format!(
                    "a forwarded launch-bearing buffer names another colour candidate ({}) and not the colour input {}: not held (counted in the tally)",
                    self.other_named.map_or_else(String::new, hex),
                    self.inputs.map_or_else(String::new, |i| hex(i.colour.0))
                ),
            )
        } else {
            return None;
        };
        self.said_kinds |= bit;
        Some(line)
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
}

static TRACKING: LazyLock<Mutex<HashMap<vk::Device, Arc<Tracking>>>> = LazyLock::new(Default::default);

impl Tracking {
    /// A new tracker for `device`, also reachable through [`tracking_for`] (the
    /// `vkGetImageViewHandle64NVX` wrapper has only the device handle).
    pub(crate) fn new_for(device: vk::Device) -> Arc<Self> {
        let tracking = Arc::new(Self { device: device.as_raw(), ..Self::default() });
        lock(&TRACKING).insert(device, tracking.clone());
        tracking
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
    pub(crate) fn launch(&self, command_buffer: vk::CommandBuffer, function: vk::CuFunctionNVX, params: Option<&[u8]>) -> Option<InlinePoint> {
        let mut t = self.lock();
        t.launch_kernel(command_buffer, function, params);
        self.rearm(&t);
        let (cb, colour) = t.take_inline_at()?;
        let point = t.inline_point(cb, colour);
        if point.is_none() && !t.said_layout_miss && t.images.get(&colour).is_some() {
            let own = t.pending.get(&cb).is_some_and(|p| p.iter().any(|s| s.image == colour)) || t.attached.get(&cb).is_some_and(|a| a.iter().any(|(i, _)| *i == colour));
            if !own && !t.committed.contains_key(&colour) {
                t.said_layout_miss = true;
                crate::log!(
                    "[preupscale] no hold inside DLSS's buffer: the colour input {}'s layout is unknown there ({} barriers on it seen since identification, last to {:?})",
                    hex(colour),
                    t.colour_barriers,
                    t.last_colour_layout
                );
                crate::logging::flush();
            }
        }
        point
    }

    /// `vkCreateCuFunctionNVX` created `function` for the kernel `name`.
    pub(crate) fn record_function(&self, function: vk::CuFunctionNVX, name: &str) {
        self.lock().record_function(function, name);
    }

    pub(crate) fn forget_function(&self, function: vk::CuFunctionNVX) {
        self.lock().forget_function(function);
    }

    /// `vkBeginCommandBuffer`: forgets the buffer's earlier recording. Nothing to do (no lock)
    /// while nothing is armed.
    pub(crate) fn begin(&self, command_buffer: vk::CommandBuffer) {
        if self.armed() {
            let mut t = self.lock();
            t.begin(command_buffer);
            self.rearm(&t);
        }
    }

    /// `vkFreeCommandBuffers`, as [`Self::begin`].
    pub(crate) fn free(&self, command_buffers: &[vk::CommandBuffer]) {
        if self.armed() {
            let mut t = self.lock();
            t.free(command_buffers);
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
        let (scan, line) = {
            let mut t = self.lock();
            if !t.armed() {
                return None;
            }
            let device = self.device;
            let observed = t.observe(batches);
            let line = t.refresh().map(|l| format!("{l} [device {device:#x}]"));
            self.watching.store(t.inputs.is_some(), Ordering::Relaxed);
            self.extent.store(t.inputs.map_or(0, |i| u64::from(i.colour.1.width) << 32 | u64::from(i.colour.1.height)), Ordering::Relaxed);
            let undecided_before = t.evaluations - t.colour_submits;
            let foreign_before = t.foreign_submits;
            let scan = t.scan(batches);
            let kind_line = t.classify_line(scan.as_ref(), undecided_before);
            // Every 3000th forwarded one: the running counts.
            let tally = (t.foreign_submits != foreign_before && t.foreign_submits.is_multiple_of(3000)).then(|| {
                format!(
                    "launch-bearing submits: {} held reading the colour input{}, {} held undecided, {} forwarded untouched (their launches never name the colour input){}; registered images: {} kept after their last view was destroyed, {} destroyed; {} depth-only changes",
                    t.colour_submits,
                    if t.retargeted_submits > 0 { format!(" ({} of them another candidate their input launch names)", t.retargeted_submits) } else { String::new() },
                    t.evaluations - t.colour_submits,
                    t.foreign_submits,
                    if t.other_candidate_buffers > 0 {
                        format!("; {} forwarded buffers named another colour candidate", t.other_candidate_buffers)
                    } else {
                        String::new()
                    },
                    t.unregistered_by_view,
                    t.unregistered_by_image,
                    t.depth_changes
                )
            });
            self.rearm(&t);
            (scan, observed.into_iter().map(Some).chain([line, kind_line, tally]))
        };
        for line in line.flatten() {
            crate::log!("[preupscale] {line}");
            crate::logging::flush();
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

/// One batch of a `vkQueueSubmit2` call, owned. `p_next` is the application's chain, kept on every
/// part (only structures safe to repeat are accepted, see [`parse2`]).
#[derive(Clone, Debug)]
pub(crate) struct Batch2 {
    pub flags: vk::SubmitFlags,
    pub p_next: *const c_void,
    pub waits: Vec<vk::SemaphoreSubmitInfo>,
    pub cbs: Vec<vk::CommandBufferSubmitInfo>,
    pub signals: Vec<vk::SemaphoreSubmitInfo>,
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
}

impl SplitBatch for Batch2 {
    type Wait = vk::SemaphoreSubmitInfo;
    fn split_at(mut self, k: usize) -> (Self, Self) {
        let rest = self.cbs.split_off(k);
        let suffix = Batch2 { flags: self.flags, p_next: self.p_next, waits: Vec::new(), cbs: rest, signals: std::mem::take(&mut self.signals) };
        (self, suffix)
    }
    fn take_waits(&mut self) -> Vec<vk::SemaphoreSubmitInfo> {
        std::mem::take(&mut self.waits)
    }
    fn put_waits(&mut self, waits: Vec<vk::SemaphoreSubmitInfo>) {
        self.waits.splice(0..0, waits);
    }
    fn own(waits: Vec<vk::SemaphoreSubmitInfo>, command_buffer: vk::CommandBuffer) -> Self {
        let waits = waits.into_iter().map(|w| vk::SemaphoreSubmitInfo { stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS, ..w }).collect();
        let cb = vk::CommandBufferSubmitInfo { command_buffer, ..Default::default() };
        Batch2 { flags: vk::SubmitFlags::empty(), p_next: std::ptr::null(), waits, cbs: vec![cb], signals: Vec::new() }
    }
    fn command_buffers(&self) -> Vec<vk::CommandBuffer> {
        self.cbs.iter().map(|c| c.command_buffer).collect()
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
pub(crate) fn submit_around<B: SplitBatch>(
    mut plan: Plan<B>, fence: vk::Fence, submit: &dyn Fn(&[B], vk::Fence) -> vk::Result, hold: impl FnOnce(&[B::Wait]) -> bool,
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
    let result = submit(&plan.tail, fence);
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
            waits: waits
                .iter()
                .zip(stages)
                .enumerate()
                .map(|(i, (&semaphore, &stage))| Wait1 { semaphore, stage, value: wait_values.map(|v| v[i]) })
                .collect(),
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
pub(crate) unsafe fn parse2(submits: &[vk::SubmitInfo2]) -> Result<Vec<Batch2>, &'static str> {
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
    pub infos: Vec<vk::SubmitInfo>,
    _timelines: Vec<vk::TimelineSemaphoreSubmitInfo>,
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
pub(crate) fn build2(batches: &[Batch2]) -> Vec<vk::SubmitInfo2> {
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

/// `(width, height)` rounded up to even: the helper's feature needs even sizes. The padding column
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
fn own_host_buffer_with(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, bytes: u64, usage: vk::BufferUsageFlags) -> Option<HostBuffer> {
    let info = vk::BufferCreateInfo::builder().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
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
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: valid allocation info.
    let Ok(memory) = (unsafe { device.allocate_memory(&alloc, None) }) else {
        // SAFETY: nothing bound yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };
    // SAFETY: freshly created, sized for each other; mapping a host-visible allocation.
    let mapped = unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .and_then(|()| device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()))
    };
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
/// else allocates a staged buffer of `bytes`.
///
/// # Safety
/// `region` must be valid for `capacity` bytes for the life of the result.
unsafe fn region_buffer(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, import: bool,
    region: *mut u8, capacity: usize, bytes: u64,
) -> Option<HostBuffer> {
    if import {
        if let Some(alignment) = crate::capture::min_imported_host_pointer_alignment(instance, physical_device) {
            let rounded = bytes.div_ceil(alignment) * alignment;
            if (region as u64).is_multiple_of(alignment) && rounded <= capacity as u64 {
                // SAFETY: `region` is valid for `capacity >= rounded` bytes and aligned.
                if let Some((buffer, memory)) = unsafe { crate::capture::import_host_buffer(device, instance, physical_device, region, rounded) } {
                    return Some(HostBuffer { buffer, memory, ptr: region, staged: false });
                }
            }
        }
    }
    own_host_buffer(device, instance, physical_device, bytes)
}

/// Everything one device's holds use. Built for one extent and queue family; rebuilt (after
/// draining) when either changes.
pub(crate) struct Resources {
    queue_family: u32,
    width: u32,
    height: u32,
    pool: vk::CommandPool,
    capture_cmd: vk::CommandBuffer,
    writeback_cmd: vk::CommandBuffer,
    capture_fence: vk::Fence,
    writeback_fence: vk::Fence,
    capture_timer: Option<crate::gpu_timer::GpuTimer>,
    writeback_timer: Option<crate::gpu_timer::GpuTimer>,
    /// Slot 0's proxy region (the capture's destination, identity's write-back source).
    proxy: HostBuffer,
    proxy_region: *mut u8,
    /// Slot 0's answer region (model's write-back source).
    answer: HostBuffer,
    answer_region: *mut u8,
    /// Dump-mode depth, motion-vector and exposure readbacks, built on first use.
    dump: Option<DumpBuffers>,
    /// The HDR encode and decode (model and roundtrip modes), built on first use.
    hdr: Option<hdr::HdrPass>,
    /// The native backend's frame resources (`native`), built on its first hold.
    #[cfg(target_arch = "x86_64")]
    native: Option<native::NativePass>,
    capture_pending: bool,
    writeback_pending: bool,
}

// SAFETY: the raw pointers are the SHM mapping (process-lifetime) or the layer's own mapped
// allocations, only ever used behind the device's `State` mutex.
unsafe impl Send for Resources {}

impl Resources {
    /// # Safety
    /// `shm` must stay open (its regions mapped) for the life of the result, which it does: the
    /// mapping is never unmapped.
    pub(crate) unsafe fn build(
        device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family: u32,
        width: u32, height: u32, shm: &ShmClient, import: bool,
    ) -> Option<Self> {
        let (pw, ph) = padded(width, height);
        let bytes = u64::from(pw) * u64::from(ph) * TEXEL;
        let (proxy_region, proxy_capacity) = shm.proxy_region(Slot::Primary)?;
        let (answer_region, answer_capacity) = shm.answer_region(Slot::Primary)?;
        if bytes > proxy_capacity as u64 || bytes > answer_capacity as u64 {
            return None;
        }
        // SAFETY: the regions are mapped for the life of the process.
        let proxy = unsafe { region_buffer(device, instance, physical_device, import, proxy_region, proxy_capacity, bytes) }?;
        // SAFETY: as above.
        let Some(answer) = (unsafe { region_buffer(device, instance, physical_device, import, answer_region, answer_capacity, bytes) }) else {
            // SAFETY: never used.
            unsafe { proxy.destroy(device) };
            return None;
        };
        let cleanup = |pool: Option<vk::CommandPool>, fences: &[vk::Fence]| {
            // SAFETY: none of these was ever submitted.
            unsafe {
                proxy.destroy(device);
                answer.destroy(device);
                for &f in fences {
                    device.destroy_fence(f, None);
                }
                if let Some(pool) = pool {
                    device.destroy_command_pool(pool, None);
                }
            }
        };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid create info on a live device.
        let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else {
            cleanup(None, &[]);
            return None;
        };
        let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2);
        // SAFETY: `pool` was just created; the loader's dispatch pointer is set on each buffer.
        let Ok(cmds) = (unsafe { crate::loader_data::allocate_commands(device, &alloc) }) else {
            cleanup(Some(pool), &[]);
            return None;
        };
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
            queue_family,
            width,
            height,
            pool,
            capture_cmd: cmds[0],
            writeback_cmd: cmds[1],
            capture_fence: fences[0],
            writeback_fence: fences[1],
            capture_timer: crate::gpu_timer::GpuTimer::new(device, instance, physical_device, queue_family),
            writeback_timer: crate::gpu_timer::GpuTimer::new(device, instance, physical_device, queue_family),
            proxy,
            proxy_region,
            answer,
            answer_region,
            dump: None,
            hdr: None,
            #[cfg(target_arch = "x86_64")]
            native: None,
            capture_pending: false,
            writeback_pending: false,
        })
    }

    pub(crate) fn matches(&self, queue_family: u32, width: u32, height: u32) -> bool {
        (self.queue_family, self.width, self.height) == (queue_family, width, height)
    }

    pub(crate) fn imported(&self) -> bool {
        !self.proxy.staged && !self.answer.staged
    }

    pub(crate) fn pending(&self) -> bool {
        self.capture_pending || self.writeback_pending
    }

    fn padded_bytes(&self) -> usize {
        let (pw, ph) = padded(self.width, self.height);
        pw as usize * ph as usize * TEXEL as usize
    }

    /// Waits (bounded) for every submission of the layer's own that is still pending. `true` when
    /// nothing is pending afterwards. Reads the write-back's timestamps into `writeback_gpu_ms`.
    pub(crate) fn wait_idle(&mut self, device: &ash::Device, writeback_gpu_ms: &mut Option<f32>) -> bool {
        let mut fences = Vec::new();
        if self.capture_pending {
            fences.push(self.capture_fence);
        }
        if self.writeback_pending {
            fences.push(self.writeback_fence);
        }
        if fences.is_empty() {
            return true;
        }
        // SAFETY: the layer's own fences, each submitted (only `pending` after a successful submit).
        let wait = unsafe { device.wait_for_fences(&fences, true, CLEANUP_WAIT.as_nanos() as u64) };
        if crate::note_fence_wait(wait, "preupscale::wait_idle").is_err() {
            return false;
        }
        if std::mem::take(&mut self.writeback_pending) {
            if let Some(timer) = self.writeback_timer.as_mut() {
                timer.read_if_pending(device);
                *writeback_gpu_ms = timer.take_reading().or(*writeback_gpu_ms);
            }
        }
        self.capture_pending = false;
        true
    }

    /// # Safety
    /// Nothing of this value's may still be pending on the GPU ([`Self::wait_idle`] returned true,
    /// or the device is idle).
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: forwarded; freeing the pool frees both command buffers.
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
            self.answer.destroy(device);
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
        }
    }
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
    pub depth: Option<Aux>,
    pub mvec: Option<Aux>,
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
    /// Which identification of DLSS's inputs this hold works on ([`Scan::identification`]): the
    /// auto-exposure starts its adaptation afresh when it changes.
    pub identification: u64,
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
    /// The answer was late or missing (counted in `preupscale_misses`).
    pub over_budget: bool,
    /// The helper answered in time with an echo of the frame, not the model's answer (no feature
    /// built, a failed evaluate): nothing is written back. Counted in `preupscale_misses`.
    pub echoed: bool,
    /// Model mode: the helper's answer was the model's and was written back.
    pub evaluated: bool,
    /// Model mode: the slot-0 request (`seq_req`) this hold started, so it reached the helper
    /// (whatever came back). Only such a hold engages the device ([`Session::note`]).
    pub request: Option<u32>,
    /// Model mode: the hold failed on the layer's side before any request reached the helper
    /// (an unusable exposure value, a submit it cannot parse, resources it cannot build, ...).
    /// Counted toward the breaker like a missing answer ([`Self::mark_local`]).
    pub local_failure: bool,
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
    /// With [`ExposureSource::Auto`]: what the auto-exposure measured.
    pub auto_exposure: Option<hdr::AutoState>,
    /// Where the hold's CPU time went (wall time on this thread).
    pub timing: HoldTiming,
    /// The native backend ran this hold (no helper request; `evaluated` means the network ran).
    pub native: bool,
}

impl HoldResult {
    /// A model-mode hold that failed on the layer's side before reaching the helper, for `why`.
    pub(crate) fn local(why: &'static str) -> Self {
        Self { miss: Some(why), local_failure: true, ..Default::default() }
    }

    /// Whether the post path must forget its own slot-0 answer (`capture::Inflight::forget_answer`)
    /// after this hold: it started a slot-0 request, so the answer region holds, or will hold once
    /// a late answer lands, this path's answer. Not only after a write-back: a request left in
    /// flight over budget would otherwise be read by the pipelined post path as its own answer.
    pub(crate) fn claims_slot0(&self) -> bool {
        self.request.is_some()
    }

    /// Sets [`Self::local_failure`] for a finished hold: in model mode, a miss with no request
    /// started and no answer missing (those are the helper's: `over_budget`, `echoed`).
    pub(crate) fn mark_local(&mut self, mode: Mode) {
        self.local_failure = mode == Mode::Model && !self.native && self.miss.is_some() && self.request.is_none() && !self.over_budget && !self.echoed;
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
    /// Model mode: `seq_req` bumped -> `seq_resp` seen.
    pub round_trip: Option<Duration>,
    /// Model mode: the helper's own wall time for this request (`helper_busy_us`), so
    /// `round_trip - helper_busy` is the hand-off: the helper noticing the request plus this
    /// thread noticing the answer.
    pub helper_busy: Option<Duration>,
    /// From the answer (or the capture, without a helper) to the write-back submit returning.
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
        vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED => None,
        l @ (vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => Some((l, false)),
        _ => Some((vk::ImageLayout::TRANSFER_SRC_OPTIMAL, true)),
    }
}

fn aux_range(format: vk::Format) -> vk::ImageSubresourceRange {
    let aspect = if DEPTH_FORMATS.contains(&format) {
        if has_stencil(format) { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH }
    } else {
        vk::ImageAspectFlags::COLOR
    };
    vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: vk::REMAINING_MIP_LEVELS, base_array_layer: 0, layer_count: vk::REMAINING_ARRAY_LAYERS }
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
fn capture_reads(target: &Target, bufs: Option<&DumpBuffers>, exposure_buffer: Option<vk::Buffer>, native_mvec: Option<vk::Buffer>) -> Vec<DumpRead> {
    let full = (target.width, target.height);
    let mut parts = Vec::new();
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

/// Records the capture into the proxy buffer at the padded size: the colour input (GENERAL) copied
/// raw, or with `hdr`, the exposure input read into the HDR pass's buffer (or, with no readable one,
/// [`ExposureSource::Auto`], measured from the colour input into the same buffer,
/// [`hdr::HdrPass::record_auto_exposure`]) and the colour input encoded for the model
/// ([`hdr::HdrPass::record_encode`]); in dump mode also the depth aspect, the
/// motion vectors and the 1x1 exposure candidates. Each auxiliary image is read in its committed
/// layout (transitioned to TRANSFER_SRC_OPTIMAL and back when that layout cannot be copied from).
///
/// # Safety
/// `res`'s capture command buffer must not be pending; with `hdr`, its colour view is bound to
/// `target.colour` and, for [`ExposureSource::Auto`], its auto-exposure is built.
unsafe fn record_capture(
    device: &ash::Device, res: &Resources, target: &Target, dump: Option<&DumpBuffers>, hdr: Option<&hdr::HdrPass>, native_mvec: Option<vk::Buffer>,
) -> ash::prelude::VkResult<()> {
    let auto = hdr.is_some() && ExposureSource::of(target) == ExposureSource::Auto;
    let cmd = res.capture_cmd;
    // SAFETY: the pool allows resetting buffers individually; the buffer is not pending.
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
    }
    let aux: Vec<DumpRead> = capture_reads(target, dump, hdr.filter(|_| !auto).map(hdr::HdrPass::exposure_buffer), native_mvec);
    let mut to_read: Vec<vk::ImageMemoryBarrier> = aux
        .iter()
        .filter(|r| r.transition)
        .map(|&DumpRead { aux: a, read, .. }| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(a.layout.unwrap_or_default())
                .new_layout(read)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(a.image)
                .subresource_range(aux_range(a.format))
                .build()
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
    let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(dst_access).build();
    // SAFETY: recording into the layer's own buffer; every handle is live.
    unsafe {
        device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, dst_stage, vk::DependencyFlags::empty(), &[open], &[], &to_read);
        if let Some(timer) = &res.capture_timer {
            timer.record_start(device, cmd);
        }
        for r in &aux {
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
        }
        match hdr {
            Some(pass) => {
                if auto {
                    // Unbuilt (the caller builds it first), the exposure buffer would keep the 0 it
                    // was cleared to, and nothing would be written back.
                    pass.record_auto_exposure(device, cmd);
                }
                pass.record_encode(device, cmd, res.proxy.buffer, target.paper_white);
            }
            None => device.cmd_copy_image_to_buffer(cmd, target.colour, vk::ImageLayout::GENERAL, res.proxy.buffer, &capture_regions(target.width, target.height)),
        }
    }
    let back: Vec<vk::ImageMemoryBarrier> = aux
        .iter()
        .filter(|r| r.transition)
        .map(|&DumpRead { aux: a, read, .. }| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(read)
                .new_layout(a.layout.unwrap_or_default())
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(a.image)
                .subresource_range(aux_range(a.format))
                .build()
        })
        .collect();
    // The auto-exposure's resolve wrote its state and the exposure value from a shader: those are
    // read by the host too.
    let (close_stage, close_access) = if auto {
        (vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
    } else {
        (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
    };
    let close = vk::MemoryBarrier::builder().src_access_mask(close_access).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ).build();
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
        if let Some(timer) = &res.capture_timer {
            timer.record_end(device, cmd);
        }
        device.end_command_buffer(cmd)
    }
}

/// Records the write-back into the colour input (GENERAL), cropping the padding: `source` (padded
/// RGBA16F) copied raw, or with `hdr`, copied into the HDR pass's answer image and decoded over the
/// colour input in place ([`hdr::HdrPass::record_decode`]). Closed by a barrier that makes it
/// visible to everything later on the queue.
///
/// # Safety
/// `res`'s write-back command buffer must not be pending; with `hdr`, the pass holds this hold's
/// capture and its colour view is bound to `target.colour`.
unsafe fn record_writeback(device: &ash::Device, res: &Resources, target: &Target, source: vk::Buffer, hdr: Option<&hdr::HdrPass>) -> ash::prelude::VkResult<()> {
    let cmd = res.writeback_cmd;
    let (dst_stage, dst_access, src_stage, src_access) = if hdr.is_some() {
        (
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
        )
    } else {
        (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
    };
    let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE).dst_access_mask(dst_access).build();
    let close = vk::MemoryBarrier::builder().src_access_mask(src_access).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
    let answer_image: Vec<vk::ImageMemoryBarrier> = hdr.map(hdr::HdrPass::answer_to_general).into_iter().collect();
    // SAFETY: the layer's own buffer, not pending; every handle is live.
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[open],
            &[],
            &answer_image,
        );
        if let Some(timer) = &res.writeback_timer {
            timer.record_start(device, cmd);
        }
        match hdr {
            Some(pass) => pass.record_decode(device, cmd, source, target.paper_white),
            None => device.cmd_copy_buffer_to_image(cmd, source, target.colour, vk::ImageLayout::GENERAL, &[writeback_region(target.width, target.height)]),
        }
        device.cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[close], &[], &[]);
        if let Some(timer) = &res.writeback_timer {
            timer.record_end(device, cmd);
        }
        device.end_command_buffer(cmd)
    }
}

/// One hold: capture the colour input into slot 0's proxy region, then per `mode` write the same
/// bytes back (identity), hand them to the helper and write its answer back (model), or keep them
/// for a dump. In model and roundtrip modes the capture encodes the frame for the model and the
/// write-back decodes the answer ([`hdr`]; roundtrip's answer is the encoded proxy itself); with no
/// readable exposure image the exposure is measured from the frame ([`ExposureSource::Auto`]), and
/// with an unusable exposure value nothing is written back. Both the encode and the decode read the
/// exposure from the same buffer, so they use the same value. `submit` makes the layer's two submissions on the game's queue (the capture one with
/// the moved wait semaphores). Never blocks longer than the capture's bounded fence wait plus, in
/// model mode, `budget`. The write-back is not waited on: its fence is checked at the next hold
/// ([`Resources::wait_idle`]).
///
/// # Safety
/// `res` must belong to `device` and match `target`'s extent; `target`'s images must be live on
/// `device`, the colour input in GENERAL; `submit` must submit on the queue the game's submit is
/// for, which the caller holds.
pub(crate) unsafe fn run_hold(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, res: &mut Resources, shm: &mut ShmClient,
    target: &Target, mode: Mode, dump: bool, frame: u64, budget: Duration,
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
        if source == ExposureSource::Auto {
            // SAFETY: `device` is live; nothing of the pass is pending (`wait_idle` above).
            if !unsafe { pass.ensure_auto(device, instance, physical_device) } {
                result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
                return result;
            }
            pass.auto_for(target.identification);
        }
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
    if unsafe { record_capture(device, res, target, dump_bufs, hdr_pass, None) }.is_err() {
        result.miss = Some("recording the capture failed");
        return result;
    }
    // SAFETY: the fence is idle (`wait_idle` above, or never submitted).
    if unsafe { device.reset_fences(&[res.capture_fence]) }.is_err() {
        result.miss = Some("resetting the capture fence failed");
        return result;
    }
    if crate::note_vk(submit(Which::Capture, res.capture_cmd, res.capture_fence)).is_err() {
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
    if crate::note_fence_wait(wait, "preupscale::capture").is_err() {
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
        let e = pass.exposure_value();
        result.exposure = Some(e);
        if result.exposure_source == Some(ExposureSource::Auto) {
            result.auto_exposure = pass.auto_state();
        }
        if !hdr::exposure_ok(e) {
            result.miss = Some("the exposure value is not usable (zero, negative or not finite)");
            return result;
        }
    }
    let source = match mode {
        Mode::Off => return result,
        Mode::Dump => {
            if dump {
                let n = target.width as usize * target.height as usize * 4;
                // SAFETY: the capture finished; each buffer holds at least `at + len` bytes (`n`,
                // `bytes`, `MAX_EXPOSURE * EXPOSURE_STRIDE`).
                let read = |buf: &HostBuffer, at: usize, len: usize| unsafe { std::slice::from_raw_parts(buf.ptr.add(at), len) }.to_vec();
                let reads = res.dump.as_ref().map(|bufs| capture_reads(target, Some(bufs), None, None)).unwrap_or_default();
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
        Mode::Model => {
            let (pw, ph) = padded(target.width, target.height);
            let before = shm.last_request(Slot::Primary);
            let answer = await_answer(shm, pw, ph, budget, &mut result.timing);
            // A request was started (`seq_req` bumped): it reached the helper.
            result.request = Some(shm.last_request(Slot::Primary)).filter(|&r| r != before);
            match answer {
                Answer::Model => result.evaluated = true,
                Answer::Echo => {
                    // The frame itself came back: nothing to write, and the breaker counts it.
                    result.miss = Some("the helper echoed the frame (no model answer)");
                    result.echoed = true;
                    return result;
                }
                Answer::Missed(why) => {
                    result.miss = Some(why);
                    result.over_budget = true;
                    return result;
                }
            }
            if res.answer.staged {
                // SAFETY: both valid for `bytes`; the helper finished writing before `seq_resp`.
                unsafe { std::ptr::copy_nonoverlapping(res.answer_region, res.answer.ptr, bytes) };
            }
            res.answer.buffer
        }
    };
    let t_writeback = Instant::now();
    let hdr_pass = if mode.hdr() { res.hdr.as_ref() } else { None };
    // SAFETY: the write-back buffer is idle (`wait_idle` above); in the HDR modes the pass holds this
    // hold's capture and its colour view.
    if unsafe { record_writeback(device, res, target, source, hdr_pass) }.is_err() {
        result.miss = Some("recording the write-back failed");
        return result;
    }
    // SAFETY: as for the capture fence.
    if unsafe { device.reset_fences(&[res.writeback_fence]) }.is_err() {
        result.miss = Some("resetting the write-back fence failed");
        return result;
    }
    if crate::note_vk(submit(Which::WriteBack, res.writeback_cmd, res.writeback_fence)).is_err() {
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

/// What came back for a model-mode hold's request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Answer {
    /// The model's answer, for this size.
    Model,
    /// In time and for this size, but the helper echoed the frame (`seq_eval` not this request).
    Echo,
    /// No usable answer: over budget, the helper stopped, the request could not be started, or an
    /// answer for another size.
    Missed(&'static str),
}

/// Sends slot 0's proxy (already written, `pw`x`ph` RGBA16F) to the helper and waits for the
/// answer, at most `budget` (or until the helper stops being alive), spinning. Fills `timing`'s
/// round trip and helper time.
pub(crate) fn await_answer(shm: &mut ShmClient, pw: u32, ph: u32, budget: Duration, timing: &mut HoldTiming) -> Answer {
    shm.set_frame_info(Slot::Primary, pw, ph, neural_forge_protocol::enums::proxy_format::RGBA16F);
    if !shm.begin_async_request(Slot::Primary) {
        return Answer::Missed("the request could not be started");
    }
    let start = Instant::now();
    let mut spins = 0u32;
    let mut why = "the helper did not answer";
    let answered = loop {
        match shm.poll_async_request(Slot::Primary) {
            Some(true) => break true,
            None => break false,
            Some(false) => {}
        }
        if start.elapsed() >= budget {
            why = "the answer was over budget";
            break false;
        }
        if !shm.helper_alive() {
            why = "the helper stopped answering";
            break false;
        }
        // Spin, not sleep: this is vkd3d-proton's submission thread, held anyway, and a
        // sleep's wake-up is latency added to every frame (see "Hand-off latency" in
        // docs/PRE_UPSCALER_DESIGN.md). Yielding keeps it polite to a busy core.
        relax(&mut spins);
    };
    timing.round_trip = Some(start.elapsed());
    if !answered {
        return Answer::Missed(why);
    }
    timing.helper_busy = shm.helper_busy();
    if shm.answered_dims() != Some((pw, ph)) {
        return Answer::Missed("the answer is for another size");
    }
    if shm.answer_evaluated() { Answer::Model } else { Answer::Echo }
}

/// The pre-upscaler path's circuit breaker (model mode). Closed, every DLSS submit is held. When
/// the helper reports no model (`model_up` 0) or [`BREAKER_MISSES`] holds in a row got no model
/// answer, it opens: submits are forwarded untouched with no wait for [`BREAKER_COOL_DOWN`], then
/// one probe hold is let through; a model answer closes it, anything else opens it again. Without
/// it a helper that could not build its feature still cost every frame a drained queue and the
/// full answer budget (the rig's stuck-feature run, docs/PRE_UPSCALER_DESIGN.md "Robustness: failed
/// feature builds"). Starts as a probe: the first hold decides.
#[derive(Debug)]
pub(crate) struct Breaker {
    state: BreakerState,
    /// Holds in a row without a model answer.
    misses: u32,
    /// When it opened (the first time since it was last closed).
    opened_at: Option<Instant>,
    /// Submits forwarded untouched while open, and probes that failed, since it opened.
    forwarded: u64,
    failed_probes: u32,
    last_log: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerState {
    Closed,
    Open { until: Instant },
    Probing,
}

impl Default for Breaker {
    fn default() -> Self {
        Self { state: BreakerState::Probing, misses: 0, opened_at: None, forwarded: 0, failed_probes: 0, last_log: None }
    }
}

impl Breaker {
    /// Whether this DLSS submit may be held, at `now`; `model_up` is the helper's flag. A `false`
    /// is counted as a submit forwarded untouched.
    pub(crate) fn allow(&mut self, now: Instant, model_up: bool) -> bool {
        match self.state {
            BreakerState::Closed if model_up => true,
            BreakerState::Closed => {
                self.open(now, "the helper reports no model (model_up=0)".to_string());
                self.forwarded += 1;
                false
            }
            BreakerState::Open { until } if now >= until => {
                self.state = BreakerState::Probing;
                true
            }
            BreakerState::Open { .. } => {
                self.forwarded += 1;
                false
            }
            BreakerState::Probing => true,
        }
    }

    /// A held submit's outcome: `true` when the model's answer was written back, `false` for an
    /// echo, a late answer or none, or a hold that failed before asking ([`HoldResult::local_failure`]).
    pub(crate) fn record(&mut self, now: Instant, model_answer: bool) {
        if model_answer {
            self.misses = 0;
            if self.state != BreakerState::Closed {
                if let Some(t) = self.opened_at.take() {
                    crate::log!(
                        "[preupscale] breaker closed: the helper answered with the model again; holding resumes after {:.1}s open ({} DLSS submits forwarded untouched, {} probes without a model answer)",
                        now.duration_since(t).as_secs_f32(),
                        self.forwarded,
                        self.failed_probes
                    );
                    crate::logging::flush();
                }
                self.state = BreakerState::Closed;
                self.last_log = None;
            }
            return;
        }
        self.misses = self.misses.saturating_add(1);
        match self.state {
            BreakerState::Probing => {
                let why = if self.opened_at.is_some() { "the probe hold got no model answer" } else { "the first hold got no model answer" };
                self.open(now, why.to_string());
            }
            BreakerState::Closed if self.misses >= BREAKER_MISSES => {
                self.open(now, format!("{} holds in a row got no model answer (echoed, late, or failed before reaching the helper)", self.misses));
            }
            _ => {}
        }
    }

    fn open(&mut self, now: Instant, why: String) {
        self.state = BreakerState::Open { until: now + BREAKER_COOL_DOWN };
        if self.opened_at.is_none() {
            self.opened_at = Some(now);
            self.forwarded = 0;
            self.failed_probes = 0;
            self.last_log = Some(now);
            crate::log!(
                "[preupscale] breaker open: {why}; DLSS submits go through untouched (no wait), one probe hold every {}s",
                BREAKER_COOL_DOWN.as_secs()
            );
            crate::logging::flush();
            return;
        }
        self.failed_probes += 1;
        if self.last_log.is_none_or(|t| now.duration_since(t) >= Duration::from_secs(30)) {
            self.last_log = Some(now);
            crate::log!(
                "[preupscale] breaker still open after {:.0}s: {why} ({} probes without a model answer, {} DLSS submits forwarded untouched)",
                self.opened_at.map_or(0.0, |t| now.duration_since(t).as_secs_f32()),
                self.failed_probes,
                self.forwarded
            );
            crate::logging::flush();
        }
    }

    /// Whether it has opened and not closed since: holds are paused (or one is probing).
    pub(crate) fn paused(&self) -> bool {
        self.opened_at.is_some()
    }

    /// Whether the next submit would be held at `now` without changing anything (tests).
    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.state == BreakerState::Closed
    }
}

/// One step of a busy wait: a few CPU spin hints, then a yield to any other runnable thread on
/// this core every [`YIELD_EVERY`] steps. Never sleeps (a sleep's wake-up granularity is what the
/// busy wait is there to avoid), so the caller bounds the wait itself.
pub(crate) fn relax(spins: &mut u32) {
    *spins = spins.wrapping_add(1);
    if (*spins).is_multiple_of(YIELD_EVERY) {
        std::thread::yield_now();
    } else {
        std::hint::spin_loop();
    }
}

/// [`relax`] yields on every this-many-th step.
pub(crate) const YIELD_EVERY: u32 = 16;

// ---- Dump files. ----

/// IEEE half to single precision.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = f32::from(h & 0x3ff);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        0x1f => {
            if mant == 0.0 { sign * f32::INFINITY } else { f32::NAN }
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
        if crate::dump::write_png(&dir.join("colour-preview.png"), &preview, self.width, self.height, false, false) {
            files.push("colour-preview.png".to_string());
        }
        let meta = format!(
            "{{\n  \"width\": {},\n  \"height\": {},\n  \"format\": \"R16G16B16A16_SFLOAT\",\n  \"padded_width\": {pw},\n  \"padded_height\": {ph},\n  \
             \"colour\": \"colour.rgba16f (padded_width x padded_height, little-endian half floats, scene-linear)\",\n  \
             \"depth\": {},\n  \"depth_format\": \"{}\",\n  \"mvec\": {},\n  \"mvec_format\": \"R16G16_SFLOAT\",\n  \"frame\": {}\n}}\n",
            self.width,
            self.height,
            depth_name.as_ref().map_or_else(|| "null".to_string(), |(n, ..)| format!("\"{n}\"")),
            depth_name.as_ref().map_or_else(|| "none".to_string(), |(.., f)| format!("{f:?}")),
            if self.mvec.is_some() { "\"mvec.rg16f\"" } else { "null" },
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
    /// [`HoldTiming`]'s phases in milliseconds, per hold that reached them; `handoff` is
    /// `round_trip - helper_busy` per answered hold.
    prep_ms: Vec<f32>,
    capture_wait_ms: Vec<f32>,
    /// The longest capture fence wait since the path started (the window's is the maximum of
    /// `capture_wait_ms`): the worst case [`FRAME_CAPTURE_WAIT`] has to leave room for.
    capture_wait_session_max_ms: f32,
    round_trip_ms: Vec<f32>,
    helper_busy_ms: Vec<f32>,
    handoff_ms: Vec<f32>,
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
    if secs > 0.0 { holds as f32 / secs } else { 0.0 }
}

impl Stats {
    fn book(&mut self, t: &HoldTiming) {
        let ms = |d: Duration| d.as_secs_f32() * 1000.0;
        let pairs = [
            (&mut self.prep_ms, t.prep),
            (&mut self.capture_wait_ms, t.capture_wait),
            (&mut self.round_trip_ms, t.round_trip),
            (&mut self.helper_busy_ms, t.helper_busy),
            (&mut self.writeback_cpu_ms, t.writeback),
        ];
        for (v, d) in pairs {
            if let Some(d) = d {
                v.push(ms(d));
            }
        }
        if let (Some(rt), Some(busy)) = (t.round_trip, t.helper_busy) {
            self.handoff_ms.push(ms(rt.saturating_sub(busy)));
        }
        if let Some(wait) = t.capture_wait {
            self.capture_wait_session_max_ms = self.capture_wait_session_max_ms.max(ms(wait));
        }
    }

    /// The phase medians, for the periodic line.
    fn phases(&self) -> String {
        format!(
            "phases ms (median): prep={:.2} capture_wait={:.2} round_trip={:.2} helper_busy={:.2} handoff={:.2} writeback={:.2}; capture_wait max={:.2} (session max {:.2}, bound {} ms)",
            median(&self.prep_ms),
            median(&self.capture_wait_ms),
            median(&self.round_trip_ms),
            median(&self.helper_busy_ms),
            median(&self.handoff_ms),
            median(&self.writeback_cpu_ms),
            self.capture_wait_ms.iter().copied().fold(0.0, f32::max),
            self.capture_wait_session_max_ms,
            FRAME_CAPTURE_WAIT.as_millis()
        )
    }

    fn clear_phases(&mut self) {
        for v in [&mut self.prep_ms, &mut self.capture_wait_ms, &mut self.round_trip_ms, &mut self.helper_busy_ms, &mut self.handoff_ms, &mut self.writeback_cpu_ms] {
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

/// One device's pre-upscaler state, kept in its `State` (so holds and presents are serialized by
/// the same lock).
#[derive(Default)]
pub(crate) struct Session {
    pub(crate) res: Option<Resources>,
    stats: Stats,
    last_hold: Option<Instant>,
    dumped: bool,
    /// Launch-bearing submits a due dump waited for the depth and motion-vector layouts.
    dump_waits: u32,
    said: HashSet<&'static str>,
    /// Model mode: whether to hold at all right now.
    pub(crate) breaker: Breaker,
    /// The last launch-bearing submit with DLSS's inputs identified (held or not).
    last_dlss: Option<Instant>,
    /// A model-mode hold on this device has asked the helper (whatever came back): the
    /// pre-upscaler path works here, so the post path stays off while DLSS runs, also while the
    /// breaker is open (otherwise the post path's output-size requests and the probes' render-size
    /// ones would rebuild the helper's feature back and forth, and no probe would ever find it
    /// built). Set only by a hold whose own request reached the helper ([`HoldResult::request`]),
    /// never by a slot 0 the post path left busy; cleared after [`BREAKER_MISSES`] holds in a row
    /// failed on the layer's side (the pre path cannot work here right now, so the post path
    /// takes over again).
    engaged: bool,
    /// The last slot-0 request a hold started, so a request still with the helper can be told to
    /// be this path's (an answer over budget) or the post path's.
    own_request: Option<u32>,
    /// Holds in a row that failed on the layer's side ([`HoldResult::local_failure`]).
    local_misses: u32,
    /// The identification and exposure source last logged ([`Self::note_exposure_source`]).
    exposure_said: Option<(u64, ExposureSource)>,
    /// The identification whose game exposure read implausibly high ([`Self::check_exposure`]): its
    /// holds measure the exposure from the frame instead.
    exposure_untrusted: Option<u64>,
    /// The native backend's network on this device, when it was created for it.
    pub(crate) native: Option<NativeLoader>,
    /// Consecutive frames still to dump after the one just taken ([`dump_frames`]).
    burst: u32,
}

/// `NEURAL_FORGE_PROBE_PARAMS=1`: logs every word of DLSS Super Resolution's input-kernel parameters
/// (hex, and its two halves as f32) for the first 2000 launches, to find the per-frame scalars
/// (the camera jitter) beside the documented ones (docs/DLSS_KERNEL_CATALOGUE.md).
fn probe_params(bytes: &[u8]) {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("NEURAL_FORGE_PROBE_PARAMS").is_ok_and(|v| v == "1"));
    static COUNT: AtomicU64 = AtomicU64::new(0);
    if !*ON || COUNT.fetch_add(1, Ordering::Relaxed) >= 2000 {
        return;
    }
    let words: Vec<String> = bytes
        .chunks_exact(8)
        .enumerate()
        .map(|(i, w)| {
            let v = u64::from_le_bytes(w.try_into().unwrap_or_default());
            format!("w{i}={v:016x}({},{})", f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32))
        })
        .collect();
    crate::log!("[probe-params] {}", words.join(" "));
}

/// `NEURAL_FORGE_PREUPSCALE_DUMP_HAZARD=ignore`: dump even when DLSS's launch buffer synchronizes
/// the depth or motion vectors before its launch ([`Scan::dump_hazard`]). Diagnostic only: those
/// images may then be read before the game finished them, which consecutive dumps show.
pub(crate) fn dump_ignores_hazard() -> bool {
    static IGNORE: LazyLock<bool> = LazyLock::new(|| std::env::var("NEURAL_FORGE_PREUPSCALE_DUMP_HAZARD").is_ok_and(|v| v == "ignore"));
    *IGNORE
}

/// `NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES`: how many consecutive DLSS frames each dump takes (default
/// 1). Two or more give frame pairs for checking the motion vectors' convention.
pub(crate) fn dump_frames() -> u32 {
    static FRAMES: LazyLock<u32> = LazyLock::new(|| {
        std::env::var("NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(1)
    });
    *FRAMES
}

impl Session {
    /// In model mode, whether the post-upscaler compose must stay off for this device's present
    /// ([`post_off`]).
    pub(crate) fn suppresses_post(&self) -> bool {
        mode() == Mode::Model && post_off(self.engaged, self.last_dlss, Instant::now())
    }

    /// A launch-bearing submit with DLSS's inputs identified reached the hold point.
    pub(crate) fn saw_dlss(&mut self) {
        self.last_dlss = Some(Instant::now());
        self.share_post_off();
    }

    /// Publishes, process-wide, until when this device keeps the post path off
    /// ([`post_off_by_any_device`]): a game whose DLSS runs on a device other than the one it
    /// presents on (no swapchain on the DLSS device, identified against an output-like image) must
    /// not get the model twice, before DLSS on one device and after it on the other.
    fn share_post_off(&self) {
        *lock(&POST_OFF_UNTIL) = post_off_until(mode() == Mode::Model, self.engaged, self.last_dlss);
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

    /// Logs `what` once per process-session for this device.
    pub(crate) fn say_once(&mut self, what: &'static str) {
        if self.said.insert(what) {
            crate::log!("[preupscale] {what}");
            crate::logging::flush();
        }
    }

    /// Waits for the layer's own pending work (bounded). `true` when none is left.
    pub(crate) fn drain(&mut self, device: &ash::Device) -> bool {
        let mut gpu = None;
        self.res.as_mut().is_none_or(|r| r.wait_idle(device, &mut gpu))
    }

    /// # Safety
    /// The device must be idle (teardown).
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        if let Some(res) = self.res.take() {
            // SAFETY: forwarded.
            unsafe { res.destroy(device) };
        }
    }

    /// Makes sure `res` is built for this extent and queue family, draining and rebuilding the old
    /// one only when nothing of it is pending. `false` when there is nothing usable.
    ///
    /// # Safety
    /// As [`Resources::build`].
    pub(crate) unsafe fn ensure(
        &mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family: u32,
        width: u32, height: u32, shm: &ShmClient, import: bool,
    ) -> bool {
        // A staged set serves either way; an imported one only where imports are wanted.
        if self.res.as_ref().is_some_and(|r| r.matches(queue_family, width, height) && (import || !r.imported())) {
            return true;
        }
        if let Some(mut old) = self.res.take() {
            let mut gpu = None;
            if !old.wait_idle(device, &mut gpu) {
                self.res = Some(old);
                return false;
            }
            // SAFETY: drained just above.
            unsafe { old.destroy(device) };
        }
        // SAFETY: forwarded.
        self.res = unsafe { Resources::build(device, instance, physical_device, queue_family, width, height, shm, import) };
        match &self.res {
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
    /// buffer names read 0.22 in some scenes and up to 59,456 in others, 2026-10-05). From then on
    /// that identification's holds measure the exposure from the frame. Logged once.
    pub(crate) fn check_exposure(&mut self, identification: u64, result: &HoldResult) {
        if result.exposure_source == Some(ExposureSource::Game)
            && result.exposure.is_some_and(|e| e > EXPOSURE_TRUST_MAX)
            && self.exposure_untrusted != Some(identification)
        {
            self.exposure_untrusted = Some(identification);
            crate::log!(
                "[preupscale] the game's exposure image read {:.1} (over {EXPOSURE_TRUST_MAX}): not an exposure; measuring it from the frame for this identification",
                result.exposure.unwrap_or_default()
            );
            crate::logging::flush();
        }
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
            self.last_hold = Some(now);
            self.last_dlss = Some(now);
        }
        if result.request.is_some() || (result.native && result.evaluated) {
            self.engaged = true;
            self.own_request = result.request;
            self.local_misses = 0;
        }
        // The breaker hears about holds that asked the helper (a model answer, or an echo, a late
        // answer or none) and about model-mode holds that failed on the layer's side before asking
        // (an unusable exposure value, a refused submit, no resources): those repeat every frame
        // in a game where they happen at all, each costing a capture or nothing for no model.
        // Contention for slot 0 is not reported here at all.
        if result.evaluated {
            self.breaker.record(now, true);
        } else if result.over_budget || result.echoed {
            self.breaker.record(now, false);
        } else if result.local_failure {
            self.breaker.record(now, false);
            self.local_misses = self.local_misses.saturating_add(1);
            if self.local_misses == BREAKER_MISSES && self.engaged {
                self.engaged = false;
                crate::log!(
                    "[preupscale] {BREAKER_MISSES} holds in a row failed before reaching the helper ({}); the post-upscaler compose runs again until a hold does",
                    result.miss.unwrap_or("?")
                );
                crate::logging::flush();
            }
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
        s.book(&result.timing);
        if let Some(e) = result.exposure.filter(|&e| hdr::exposure_ok(e)) {
            s.exposure.push(e);
            s.exposure_source = result.exposure_source;
        }
        if result.over_budget || result.echoed {
            s.misses += 1;
            s.window_misses += 1;
        }
        if let Some(why) = result.miss {
            s.unlogged_misses += 1;
            if s.last_miss_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(5)) {
                crate::log!("[preupscale] frame went to DLSS untouched: {why} ({} such since the last line, {} answers over budget or echoed in total)", s.unlogged_misses, s.misses);
                crate::logging::flush();
                s.last_miss_log = Some(Instant::now());
                s.unlogged_misses = 0;
            }
        }
        shm.publish_preupscale_hold(s.last_hold_ms, s.misses);
        shm.publish_preupscale_state(if holding { 2 } else if self.breaker.paused() && self.engaged { 3 } else { 1 }, extent.0, extent.1);
        if held && s.holds.is_multiple_of(SUMMARY_EVERY) {
            let (pw, ph) = padded(extent.0, extent.1);
            // The window's rate: frames the model ran on before the upscaler (the post path's
            // "composited" count does not include them). Loading screens inside a window lower it.
            let now = Instant::now();
            let rate = per_second(SUMMARY_EVERY, s.window_start.map_or(Duration::ZERO, |t| now.duration_since(t)));
            s.window_start = Some(now);
            crate::log!(
                "[preupscale] mode={} extent={}x{} (padded {pw}x{ph}) holds={} hold_ms median={:.2} capture_gpu_ms median={:.2} writeback_gpu_ms median={:.2} misses={} (total {}) holds_per_s={:.1}{}",
                mode().name(),
                extent.0,
                extent.1,
                s.holds,
                median(&s.hold_ms),
                median(&s.capture_gpu_ms),
                median(&s.writeback_gpu_ms),
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
            s.clear_phases();
            s.window_misses = 0;
        }
    }

    /// A DLSS submit found slot 0 with a request still in flight, so it is not held. When that
    /// request is this path's own (a hold's answer that ran over budget), it is booked as another
    /// late answer; when it is the post path's (it often has one pending before the device first
    /// holds), nothing is booked: no request of this path reached the helper, so neither the
    /// device's engagement nor the breaker may hear of it.
    pub(crate) fn note_busy_slot(&mut self, shm: &ShmClient, mode: Mode, extent: (u32, u32)) {
        if mode == Mode::Model && self.own_request.is_some() && shm.pending_request(Slot::Primary) == self.own_request {
            let result = HoldResult { miss: Some("an earlier request is still with the helper"), over_budget: true, ..Default::default() };
            self.note(shm, &result, Duration::ZERO, extent);
        }
    }

    /// Books a write-back GPU reading taken at a later drain.
    pub(crate) fn note_writeback_gpu(&mut self, ms: Option<f32>) {
        if let Some(ms) = ms {
            self.stats.writeback_gpu_ms.push(ms);
        }
    }

    /// Marks the dump as done (a pending one-shot request is consumed by the caller).
    pub(crate) fn dumped(&mut self) {
        self.burst = if self.burst > 0 { self.burst - 1 } else { dump_frames() - 1 };
        self.dumped = true;
    }

    /// The header state for a present: holding when a hold happened within [`RECENT`]; paused
    /// while the breaker is open, DLSS is still running and the device is engaged (the post path is off).
    pub(crate) fn publish_status(&self, shm: &ShmClient, extent: Option<(u32, u32)>) {
        let (w, h) = extent.unwrap_or((0, 0));
        let dlss_running = self.last_dlss.is_some_and(|t| t.elapsed() < RECENT);
        let state = if self.holding() {
            2
        } else if self.breaker.paused() && dlss_running && self.engaged {
            3
        } else {
            1
        };
        shm.publish_preupscale_state(state, w, h);
    }
}

/// Whether the post-upscaler path stays off at `now` in model mode: a hold on the device has asked
/// the helper (`engaged`) and DLSS ran within [`HAND_BACK`] (`last_dlss`).
pub(crate) fn post_off(engaged: bool, last_dlss: Option<Instant>, now: Instant) -> bool {
    engaged && last_dlss.is_some_and(|t| now.saturating_duration_since(t) < HAND_BACK)
}

/// Until when the device that holds DLSS's input keeps the post path off ([`post_off`] as of its
/// last DLSS submit or hold), shared by every device of the process; `None` when it does not.
static POST_OFF_UNTIL: Mutex<Option<Instant>> = Mutex::new(None);

/// [`POST_OFF_UNTIL`]'s value for a device in model mode (`model`): [`post_off`] at any `now` is
/// `now` before it.
fn post_off_until(model: bool, engaged: bool, last_dlss: Option<Instant>) -> Option<Instant> {
    last_dlss.filter(|_| model && engaged).map(|t| t + HAND_BACK)
}

/// Whether some device of the process keeps the post path off at `now` (its [`Session`]
/// suppresses it, [`Session::suppresses_post`]). On the device that holds this is the same as its
/// own [`Session::suppresses_post`]; it matters when DLSS runs on another device than the swapchain
/// (only ever set in model mode).
pub(crate) fn post_off_by_any_device(now: Instant) -> bool {
    lock(&POST_OFF_UNTIL).is_some_and(|until| now < until)
}

/// Whether a launch-bearing submit is held in `mode`, given the live switches: `None` forwards it
/// untouched; `Some(dump)` holds it, `dump` saying whether this hold writes a dump. In model mode the
/// live toggle (F11, the GUI, `shmctl set enabled 0`), `apply_model` off or the model reported
/// unavailable mean no hold at all (the post path then sees the toggle too and presents the game's
/// frame as it is), a helper that is not running is never waited for, and an open [`Breaker`]
/// forwards the submit untouched. `layouts_known`: the depth and motion-vector layouts are known (a
/// dump waits for them, [`Session::dump_ready`]).
pub(crate) fn gate(mode: Mode, session: &mut Session, shm: &mut ShmClient, layouts_known: bool) -> Option<bool> {
    match mode {
        Mode::Off => None,
        Mode::Dump => (session.dump_due(shm) && session.dump_ready(layouts_known)).then_some(true),
        Mode::Model => {
            (shm.model_wanted() && shm.helper_alive() && session.breaker.allow(Instant::now(), shm.model_up())).then_some(false)
        }
        Mode::Identity | Mode::Roundtrip => Some(false),
    }
}

/// Writes a dump off the submit thread and logs where it went.
pub(crate) fn write_dump_async(frame: DumpFrame) {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let dir = crate::dump::captures_dir().join(format!("preupscale-{stamp}"));
    let spawned = std::thread::Builder::new().name("nf-preupscale-dump".into()).spawn(move || {
        match frame.write(&dir) {
            Ok(files) => crate::log!("[preupscale] dump of the DLSS input ({}x{}, frame {}) written to {}: {}", frame.width, frame.height, frame.frame, dir.display(), files.join(", ")),
            Err(e) => crate::log!("[preupscale] dump to {} failed: {e}", dir.display()),
        }
        crate::logging::flush();
    });
    if spawned.is_err() {
        crate::log!("[preupscale] could not start the dump writer thread");
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
    fn the_summary_rate_is_holds_over_the_window() {
        assert_eq!(per_second(300, Duration::from_secs(5)), 60.0);
        assert_eq!(per_second(300, Duration::ZERO), 0.0);
    }

    #[test]
    fn relax_counts_its_steps_and_never_sleeps() {
        let mut spins = 0;
        let started = Instant::now();
        for _ in 0..(YIELD_EVERY * 1000) {
            relax(&mut spins);
        }
        assert_eq!(spins, YIELD_EVERY * 1000);
        // 16000 steps with 1000 yields: microseconds each at most, never a sleep's granularity.
        assert!(started.elapsed() < Duration::from_secs(1), "{:?}", started.elapsed());
        let mut near_wrap = u32::MAX;
        relax(&mut near_wrap);
        assert_eq!(near_wrap, 0, "the counter wraps instead of overflowing");
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
        assert!(commands(false).is_empty());
        assert!(commands(true).contains(&VulkanCommand::CmdCuLaunchKernelNvx));
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
        let batches = vec![
            batch1(&[(1, None)], &[10], &[(2, None)]),
            batch1(&[(3, Some(7)), (4, None)], &[20, 21], &[(5, Some(8))]),
            batch1(&[], &[30], &[]),
        ];
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
        let result = submit_around(p, game_fence, &submit, |waits| {
            let _ = submit(&[Batch1::own(waits.to_vec(), cb(100))], own_fence);
            let _ = submit(&[Batch1::own(Vec::new(), cb(101))], own_fence);
            true
        }, &mut |part| accepted.push(part));
        assert_eq!(result, vk::Result::SUCCESS);
        assert_eq!(accepted, vec![vec![cb(10)], vec![cb(20), cb(30)]], "the application's buffers of each accepted part, never the layer's own");
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (vec![batches[0].clone()], vk::Fence::null()), "head: the earlier batch, no fence");
        assert_eq!(calls[1].0[0].waits, vec![Wait1 { semaphore: sem(3), stage: vk::PipelineStageFlags::ALL_COMMANDS, value: Some(5) }], "the capture waits for what the launch batch waited for");
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
        assert_eq!(submit_around(plan(batches.clone(), 1, 0), game_fence, &failing, |_| { held = true; true }, &mut |_| reported = true), vk::Result::ERROR_DEVICE_LOST);
        assert!(!held);
        assert!(!reported, "a refused head reports nothing as accepted");
    }

    fn sem2(raw: u64, value: u64) -> vk::SemaphoreSubmitInfo {
        vk::SemaphoreSubmitInfo { semaphore: sem(raw), value, stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER, ..Default::default() }
    }
    fn cb2(raw: u64) -> vk::CommandBufferSubmitInfo {
        vk::CommandBufferSubmitInfo { command_buffer: cb(raw), ..Default::default() }
    }

    #[test]
    fn submit2_split_keeps_values_flags_and_chain_and_moves_waits() {
        let latency = vk::BaseInStructure { s_type: LATENCY_SUBMISSION_PRESENT_ID_NV, p_next: std::ptr::null() };
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
        let other = vk::SubmitInfo2 { p_next: std::ptr::null(), flags: vk::SubmitFlags::empty(), command_buffer_info_count: 1, p_command_buffer_infos: &cbs[0], wait_semaphore_info_count: 0, signal_semaphore_info_count: 0, ..info };
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
        assert_eq!(own.waits.iter().map(|w| (w.semaphore, w.value, w.stage_mask)).collect::<Vec<_>>(), vec![
            (sem(1), 41, vk::PipelineStageFlags2::ALL_COMMANDS),
            (sem(2), 42, vk::PipelineStageFlags2::ALL_COMMANDS)
        ]);
        assert!(own.p_next.is_null() && own.flags.is_empty(), "the layer's own batch carries nothing of the game's");
        assert!(p.tail[0].waits.is_empty());
        // An unknown chain is refused.
        let keyed = vk::BaseInStructure { s_type: vk::StructureType::WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR, p_next: std::ptr::null() };
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

    #[test]
    fn the_colour_input_is_identified_from_a_registered_set_like_gtas() {
        let set = gta_registered();
        let found = identify(&set, Some((2560, 1440))).expect("identified");
        assert_eq!(found.colour.0, vk::Image::from_raw(0x100));
        assert_eq!(found.depth.0, vk::Image::from_raw(0x200));
        assert_eq!(found.mvec.0, vk::Image::from_raw(0x300));
        assert_eq!(found.candidates, 1);
        assert_eq!(found.exposure.map(|e| e.map(|e| e.0.as_raw())), [Some(0x600), Some(0x610), None, None], "the 1x1 float images, not the 5120x2880 R16F");
        assert_eq!(found.exposure_input.map(|e| e.0.as_raw()), Some(0x610), "the exposure input is the 1x1 R16F, not NGX's RGBA32F");
        let mut no_exposure = set.clone();
        no_exposure.remove(&0x610);
        let found = identify(&no_exposure, Some((2560, 1440))).expect("still identified");
        assert_eq!(found.exposure_input, None, "no 1x1 R16F: no exposure input (the HDR modes then fail open)");
        assert!(found.output_from_swapchain && found.output == (2560, 1440) && found.others == [None; MAX_OTHERS]);
        let fallback = identify(&set, None).expect("no swapchain: compared with the largest RGBA16F storage image");
        assert_eq!((fallback.colour.0, fallback.depth.0, fallback.mvec.0), (found.colour.0, found.depth.0, found.mvec.0));
        assert_eq!((fallback.output, fallback.output_from_swapchain), ((2560, 1440), false));
        assert_eq!(identify(&set, Some((1707, 960))), None, "DLAA: the render extent is the output's");
        let mut no_mv = set.clone();
        no_mv.remove(&0x300);
        assert_eq!(identify(&no_mv, Some((2560, 1440))), None, "motion vectors are part of the signature");
        let mut no_depth = set.clone();
        no_depth.remove(&0x200);
        assert_eq!(identify(&no_depth, Some((2560, 1440))), None);
        // A second render-extent group (a preset change mid-run): the lowest handle wins, counted.
        let mut two = set;
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        two.insert(0x50, (vk::Image::from_raw(0x50), desc(1490, 838, rgba, vk::ImageUsageFlags::STORAGE)));
        two.insert(0x51, (vk::Image::from_raw(0x51), desc(1490, 838, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)));
        two.insert(0x52, (vk::Image::from_raw(0x52), desc(1490, 838, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT)));
        let found = identify(&two, Some((2560, 1440))).unwrap();
        assert_eq!((found.colour.0, found.depth.0, found.mvec.0, found.candidates), (vk::Image::from_raw(0x50), vk::Image::from_raw(0x51), vk::Image::from_raw(0x52), 2));
    }

    /// A tracker with `set` registered (view `raw + 1`, kernel handle `raw + 0x1000`) and, when
    /// given, one swapchain.
    fn tracker_with(set: &BTreeMap<u64, (vk::Image, ImageDesc)>, swapchain: Option<(u32, u32)>) -> Tracker {
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
        if let Some(e) = swapchain {
            t.swapchain(vk::SwapchainKHR::from_raw(9), Some(e));
        }
        t
    }

    /// (a) GTA V's registered set with its swapchain: the same inputs, line, scan target and
    /// classification as before the relaxed rule (colour 0x100, depth 0x200, motion vectors 0x300,
    /// one candidate, exposure input 0x610).
    #[test]
    fn gtas_set_keeps_its_identification_and_hold_target() {
        let mut t = tracker_with(&gta_registered(), Some((2560, 1440)));
        let line = t.refresh().expect("identified");
        assert!(line.starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT"), "{line}");
        assert!(!line.contains("candidates") && !line.contains("compared with"), "one candidate, the swapchain: {line}");
        assert!(line.ends_with("exposure input 0x610; swapchain Some((2560, 1440)); identified by size"), "{line}");
        t.launch(cb(1), Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        t.launch(cb(2), Some(&param_block(&[0x1400, 0x1200])));
        let scan = t.submit_ok(&[vec![cb(2), cb(1)]]).expect("SR's buffer is held");
        assert_eq!((scan.batch, scan.index), (0, 1));
        let inputs = scan.inputs.unwrap();
        assert_eq!(
            (inputs.colour.0.as_raw(), inputs.depth.0.as_raw(), inputs.mvec.0.as_raw(), inputs.candidates, inputs.exposure_input.map(|e| e.0.as_raw())),
            (0x100, 0x200, 0x300, 1, Some(0x610))
        );
        assert_eq!((t.colour_submits, t.foreign_submits, t.other_candidate_buffers), (1, 0, 0));
    }

    /// (b) Two render-size colour candidates (Crimson Desert registers two or three): the lowest
    /// handle is the colour input and a buffer naming it is held, as before. A buffer whose launches
    /// name only the other one, but not as an input launch (no depth and motion vectors beside it),
    /// stays forwarded, and is counted and logged once. One whose input launch names the other
    /// candidate with the colour input's depth and motion vectors is held with that candidate.
    #[test]
    fn a_buffer_naming_another_candidate_is_forwarded_and_counted() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
        let set: BTreeMap<u64, (vk::Image, ImageDesc)> = [
            (0x100, desc(1516, 852, rgba, storage)),
            (0x110, desc(1516, 852, rgba, storage)),
            (0x200, desc(1516, 852, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)),
            (0x300, desc(1516, 852, vk::Format::R16G16_SFLOAT, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
        ]
        .into_iter()
        .map(|(raw, d)| (raw, (vk::Image::from_raw(raw), d)))
        .collect();
        let mut t = tracker_with(&set, Some((2560, 1440)));
        let line = t.refresh().unwrap();
        assert!(line.starts_with("colour input: image 0x100 (1516x852") && line.contains("(first of 2 candidates; others 0x110 1516x852)"), "{line}");
        // The other candidate named without depth and motion vectors: forwarded, counted.
        t.launch(cb(1), Some(&param_block(&[0x1110, 0x1400])));
        assert_eq!(t.launch[&cb(1)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Foreign);
        assert!(t.submit_ok(&[vec![cb(1)]]).is_none(), "naming another candidate only: not held");
        assert_eq!((t.foreign_submits, t.other_candidate_buffers), (1, 1));
        let first = t.classify_line(None, 0).expect("the forwarded kind");
        assert!(first.starts_with("a launch-bearing submit whose CUDA launches never name"), "{first}");
        let other = t.classify_line(None, 0).expect("then the other candidate, once");
        assert!(other.contains("names another colour candidate (0x110) and not the colour input 0x100"), "{other}");
        assert!(t.classify_line(None, 0).is_none());
        // The colour input's buffer is held exactly as before; a buffer naming both is too.
        t.launch(cb(2), Some(&param_block(&[0x1100, 0x1200, 0x1300])));
        assert!(t.submit_ok(&[vec![cb(2)]]).is_some_and(|s| s.inputs.unwrap().colour.0 == vk::Image::from_raw(0x100)));
        t.launch(cb(3), Some(&param_block(&[0x1110, 0x1100])));
        assert!(t.submit_ok(&[vec![cb(3)]]).is_some_and(|s| s.inputs.unwrap().colour.0 == vk::Image::from_raw(0x100)));
        assert_eq!((t.colour_submits, t.other_candidate_buffers), (2, 1));
        // An input launch naming the other candidate with the same depth and motion vectors: held
        // with it as the target, not counted as forwarded.
        // Its layout is tracked like the colour input's (a barrier in an earlier buffer of the
        // submit).
        t.launch(cb(4), Some(&param_block(&[0x1110, 0x1200, 0x1300])));
        t.barrier(cb(5), ImageSync::to(vk::Image::from_raw(0x110), vk::ImageLayout::TRANSFER_SRC_OPTIMAL));
        let s = t.submit_ok(&[vec![cb(5), cb(4)]]).expect("held with the candidate its input launch names");
        assert_eq!(s.inputs.map(|i| (i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw())), Some((0x110, 0x200, 0x300)));
        assert_eq!((s.index, s.colour_layout), (1, Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)));
        assert_eq!((t.colour_submits, t.retargeted_submits, t.other_candidate_buffers), (3, 1, 1));
        let undecided = t.evaluations - t.colour_submits;
        assert!(t.classify_line(Some(&s), undecided).is_some_and(|l| l.starts_with("a launch-bearing submit reads DLSS's colour input")));
        let held = t.classify_line(Some(&s), undecided).expect("the retarget, once");
        assert!(held.starts_with("a launch-bearing buffer's input launch names another colour candidate (0x110)"), "{held}");
    }

    /// (c) A Cyberpunk-2077-like set: no swapchain known on the device, depth D32_SFLOAT without
    /// stencil, motion vectors RG32F, DLSS's output an R11G11B10 storage image. Identified against
    /// the output-like image; without any larger one (DLAA) not.
    #[test]
    fn without_a_swapchain_the_largest_output_like_image_is_the_comparison() {
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let mk = |list: &[(u64, ImageDesc)]| -> BTreeMap<u64, (vk::Image, ImageDesc)> { list.iter().map(|&(raw, d)| (raw, (vk::Image::from_raw(raw), d))).collect() };
        let inputs = [
            (0x100, desc(1708, 960, vk::Format::R16G16B16A16_SFLOAT, storage)),
            (0x200, desc(1708, 960, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)),
            (0x300, desc(1708, 960, vk::Format::R32G32_SFLOAT, storage)),
            (0x400, desc(1, 1, vk::Format::R16_SFLOAT, storage)),
            (0x500, desc(1708, 960, vk::Format::R8_UNORM, storage)),
        ];
        let mut set = inputs.to_vec();
        set.push((0x600, desc(2560, 1440, vk::Format::B10G11R11_UFLOAT_PACK32, storage)));
        let mut t = tracker_with(&mk(&set), None);
        let line = t.refresh().expect("identified");
        assert!(line.starts_with("colour input: image 0x100 (1708x960 R16G16B16A16_SFLOAT"), "{line}");
        assert!(line.contains("depth 0x200 D32_SFLOAT, motion vectors 0x300 R32G32_SFLOAT"), "{line}");
        assert!(line.ends_with("swapchain None, compared with the largest registered RGBA16F/R11G11B10 storage image (2560x1440); identified by size"), "{line}");
        assert_eq!(t.inputs.map(|i| (i.output, i.output_from_swapchain, i.exposure_input.map(|e| e.0.as_raw()))), Some(((2560, 1440), false, Some(0x400))));
        // An RGBA16F output does as well; a swapchain, once known, is what counts.
        let mut rgba_out = inputs.to_vec();
        rgba_out.push((0x600, desc(2560, 1440, vk::Format::R16G16B16A16_SFLOAT, storage)));
        assert!(identify(&mk(&rgba_out), None).is_some_and(|i| i.colour.0.as_raw() == 0x100 && i.candidates == 1));
        assert_eq!(identify(&mk(&rgba_out), Some((1708, 960))), None, "DLAA with a swapchain");
        // DLAA without a swapchain: nothing larger than the colour input, nothing identified.
        assert_eq!(identify(&mk(&inputs), None), None);
        // A larger image of another format (NGX's R16F scratch) is not an output.
        let mut scratch = inputs.to_vec();
        scratch.push((0x700, desc(5120, 2880, vk::Format::R16_SFLOAT, storage)));
        assert_eq!(identify(&mk(&scratch), None), None);
    }

    /// (d) Nothing qualifies: not identified, and the line says why, with the registered views
    /// grouped (extent, format, usage, count), once per change of the set, capped in length and
    /// in number.
    #[test]
    fn a_failed_identification_logs_the_registered_set_once_per_change() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE;
        // A colour input without STORAGE, a storage one without motion vectors beside it, depth.
        let set: BTreeMap<u64, (vk::Image, ImageDesc)> = [
            (0x100, desc(1708, 960, rgba, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT)),
            (0x110, desc(1708, 960, rgba, storage)),
            (0x200, desc(1708, 960, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)),
            (0x300, desc(1708, 960, vk::Format::R16G16_UNORM, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
            (0x410, desc(2560, 1440, rgba, storage)),
        ]
        .into_iter()
        .map(|(raw, d)| (raw, (vk::Image::from_raw(raw), d)))
        .collect();
        assert_eq!(identify(&set, None), None);
        let mut t = tracker_with(&set, None);
        let line = t.refresh().expect("the failure is logged");
        assert!(line.starts_with("no DLSS input among 6 registered views (swapchain None); waiting. output: no swapchain known, largest RGBA16F/R11G11B10 storage image 2560x1440"), "{line}");
        assert!(line.contains("RGBA16F storage extents: 2560x1440 not smaller than the output, 1708x960 no R16G16/R32G32_SFLOAT motion vectors at this extent"), "{line}");
        assert!(line.contains("registered: 2560x1440 R16G16B16A16_SFLOAT storage x2, "), "{line}");
        assert!(line.contains("1708x960 R16G16B16A16_SFLOAT sampled+colour x1") && line.contains("1708x960 D32_SFLOAT depth x1"), "{line}");
        assert!(t.refresh().is_none(), "nothing changed");
        // A change of the set (same count) is logged again.
        t.forget_image(vk::Image::from_raw(0x300));
        t.record_image(vk::Image::from_raw(0x310), &vk::ImageCreateInfo { image_type: vk::ImageType::TYPE_2D, extent: vk::Extent3D { width: 1708, height: 960, depth: 1 }, format: vk::Format::R16G16_SNORM, usage: storage, samples: vk::SampleCountFlags::TYPE_1, ..Default::default() });
        t.record_view(vk::ImageView::from_raw(0x311), vk::Image::from_raw(0x310));
        t.register(vk::ImageView::from_raw(0x311), None);
        let again = t.refresh().expect("a changed set is logged again");
        assert!(again.contains("R16G16_SNORM") && !again.contains("R16G16_UNORM"), "{again}");
        // Long sets are cut at the cap.
        let many: BTreeMap<u64, (vk::Image, ImageDesc)> = (0..400u64).map(|k| (0x1000 + k, (vk::Image::from_raw(0x1000 + k), desc(100 + k as u32, 50, rgba, storage)))).collect();
        let long = diagnose(&many, None);
        assert!(long.len() <= DIAGNOSE_MAX && long.ends_with(" more)"), "{} bytes: ...{}", long.len(), &long[long.len().saturating_sub(40)..]);
        // After MAX_DIAGNOSES detailed lines, only the short one (once per change of it).
        let mut t = tracker_with(&set, None);
        for k in 0..MAX_DIAGNOSES {
            t.dirty = true;
            t.announced = None;
            assert!(t.refresh().unwrap().contains("registered: "), "detailed {k}");
        }
        t.dirty = true;
        assert_eq!(t.refresh().as_deref(), Some("no DLSS input among 6 registered views (swapchain None); waiting"));
        t.dirty = true;
        assert!(t.refresh().is_none());
    }

    /// The process-wide post-path switch (for DLSS on another device than the swapchain, the case
    /// the output-like-image identification opens) agrees with the holding device's own at every
    /// moment, and is never set outside model mode.
    #[test]
    fn the_shared_post_off_deadline_matches_the_holding_devices_own() {
        let t0 = Instant::now();
        for engaged in [false, true] {
            for last in [None, Some(t0)] {
                let until = post_off_until(true, engaged, last);
                for ms in [0, 1, 29_999, 30_000, 30_001, 60_000] {
                    let now = t0 + Duration::from_millis(ms);
                    assert_eq!(until.is_some_and(|u| now < u), post_off(engaged, last, now), "engaged {engaged}, last {last:?}, +{ms} ms");
                }
                assert_eq!(post_off_until(false, engaged, last), None, "only model mode turns the post path off");
            }
        }
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
        for (raw, w, h, format) in [(0x100, 1707, 960, vk::Format::R16G16B16A16_SFLOAT), (0x200, 1707, 960, vk::Format::D32_SFLOAT_S8_UINT), (0x300, 1707, 960, vk::Format::R16G16_SFLOAT)] {
            t.record_image(vk::Image::from_raw(raw), &info(w, h, format, storage));
            t.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
            t.register(vk::ImageView::from_raw(raw + 1), Some(raw + 0x1000));
        }
        t.swapchain(vk::SwapchainKHR::from_raw(9), Some((2560, 1440)));
        let line = t.refresh().expect("the identification is logged");
        assert!(line.starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT"), "{line}");
        assert!(t.refresh().is_none(), "nothing changed, nothing logged");
        let colour = vk::Image::from_raw(0x100);
        // Recorded in one order, submitted in the other; a launch in a secondary.
        t.barrier(cb(1), ImageSync::to(colour, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        t.barrier(cb(2), ImageSync::to(colour, vk::ImageLayout::GENERAL));
        t.barrier(cb(2), ImageSync::to(vk::Image::from_raw(0x999), vk::ImageLayout::GENERAL));
        t.launch(cb(5), None);
        t.execute(cb(3), &[cb(5)]);
        let scan = t.submit_ok(&[vec![cb(1)], vec![cb(2), cb(3)]]).expect("launch-bearing");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!(scan.colour_layout, Some(vk::ImageLayout::GENERAL), "the layout as of submission order just before the launch");
        assert!(!t.committed.contains_key(&vk::Image::from_raw(0x999)), "only the inputs are watched");
        assert!(scan.identified_now && !scan.colour_in_general(), "the first launch-bearing submit since identification is not held");
        assert!(t.submit_ok(&[vec![cb(7)]]).is_none());
        t.begin(cb(3));
        assert!(t.submit_ok(&[vec![cb(3)]]).is_none(), "re-recorded: no longer launch-bearing");
        // A destroyed input image changes the set: a destroyed depth image only after SR_RECENT
        // launch-bearing submits without a new one (a game rotating its depth images has a moment
        // without one, which is no new identification).
        t.forget_image(vk::Image::from_raw(0x200));
        assert_eq!(t.refresh(), None);
        assert!(t.inputs.is_some());
        t.launch_submits += SR_RECENT;
        assert!(t.refresh().unwrap().starts_with("no DLSS input among 2 registered views"));
        assert!(t.inputs.is_none());
    }

    /// The submit that (re)identifies the inputs is not held: barriers recorded before the images
    /// were watched were never seen (the device hooks only record them while watching), and a new
    /// identification drops what was committed, so "no layout" there is not "GENERAL". The next
    /// launch-bearing submit, with no barrier seen while watched, is.
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
            for (raw, format) in [(base, vk::Format::R16G16B16A16_SFLOAT), (base + 0x100, vk::Format::D32_SFLOAT_S8_UINT), (base + 0x200, vk::Format::R16G16_SFLOAT)] {
                tr.record_image(vk::Image::from_raw(raw), &info(1707, 960, format));
                tr.record_view(vk::ImageView::from_raw(raw + 1), vk::Image::from_raw(raw));
                tr.register(vk::ImageView::from_raw(raw + 1), None);
            }
            tr.swapchain(vk::SwapchainKHR::from_raw(9), Some((2560, 1440)));
        };
        register(0x1000);
        t.launch(cb(1), vk::CuFunctionNVX::null(), None);
        let first = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(first.inputs.is_some() && first.identified_now && first.colour_layout.is_none());
        assert!(!first.colour_in_general(), "identified on this submit: nothing known, not held");
        let second = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(!second.identified_now && second.colour_layout.is_none());
        assert!(second.colour_in_general(), "watched for a submit and no barrier seen: GENERAL");
        // A re-identification (DLSS's inputs re-created) starts over.
        t.lock().forget_image(vk::Image::from_raw(0x1000));
        register(0x5000);
        let again = t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing");
        assert_eq!(again.inputs.map(|i| i.colour.0), Some(vk::Image::from_raw(0x5000)));
        assert!(again.identified_now && !again.colour_in_general());
        assert!(t.submit_ok(&[vec![cb(1)]]).expect("launch-bearing").colour_in_general());
        // A barrier out of GENERAL still refuses the hold.
        t.barriers(cb(2), [ImageSync::to(vk::Image::from_raw(0x5000), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)].into_iter(), None);
        assert!(!t.submit_ok(&[vec![cb(2), cb(1)]]).expect("launch-bearing").colour_in_general());
    }

    /// On a device with NVX but no DLSS nothing is ever armed, so the per-command-buffer and
    /// per-submit hooks stay a relaxed load; a launch arms it until its buffer is re-recorded.
    #[test]
    fn tracking_stays_disarmed_without_launches_and_disarms_after_the_launch_buffer_is_reset() {
        let t = Tracking::default();
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
        t.launch(cb(7), vk::CuFunctionNVX::null(), None);
        assert!(t.armed());
        assert!(t.submit_ok(&[vec![cb(6), cb(7)]]).is_some_and(|s| (s.batch, s.index) == (0, 1)));
        // A secondary's launch carries over to its primary.
        t.execute(cb(8), &[cb(7)]);
        t.begin(cb(7));
        assert!(t.armed(), "the primary still carries the launch");
        t.free(&[cb(8)]);
        assert!(!t.armed(), "every launch buffer was reset or freed");
        assert!(t.submit_ok(&[vec![cb(7), cb(8)]]).is_none());
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
        let boundary = vk::BaseInStructure { s_type: FRAME_BOUNDARY_EXT, p_next: std::ptr::null() };
        let latency = vk::BaseInStructure { s_type: LATENCY_SUBMISSION_PRESENT_ID_NV, p_next: std::ptr::from_ref(&boundary) };
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
    /// next is not the identifying one.
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
        t.swapchain(vk::SwapchainKHR::from_raw(9), Some((2560, 1440)));
        t.refresh().expect("identified");
        let params = param_block(&[0x1100, 0x1200, 0x1300]);
        t.launch(cb(900), Some(&params));
        assert!(t.submit_ok(&[vec![cb(900)]]).is_some_and(|s| s.identified_now));
        t.free(&[cb(900)]);
        (t, params)
    }

    fn colour() -> vk::Image {
        vk::Image::from_raw(0x100)
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

    /// Synchronization on a DLSS input inside the launch buffer, before the launch, is something
    /// the hold would run ahead of: each kind makes the submit one to forward untouched, also when
    /// the layout stays `GENERAL`.
    #[test]
    fn synchronization_before_the_launch_in_its_buffer_forbids_the_hold() {
        let transfer = ImageSync { src_queue_family: 0, dst_queue_family: 1, ..general_to_general(colour()) };
        let cases: [(&str, &dyn Fn(&mut Tracker), Hazard); 5] = [
            ("GENERAL -> GENERAL on the colour input", &|t| t.barrier(cb(2), general_to_general(colour())), Hazard::SameLayoutBarrier),
            ("a queue-family ownership transfer", &|t| t.barrier(cb(2), transfer), Hazard::QueueFamilyTransfer),
            ("a layout transition", &|t| t.barrier(cb(2), ImageSync { old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, ..ImageSync::to(colour(), GENERAL) }), Hazard::LayoutTransition),
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
        // An opaque launch (parameters unreadable): everything before the buffer's first launch.
        let (mut t, _) = held_tracker();
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), None);
        assert_eq!(t.submit_ok(&[vec![cb(2)]]).expect("held undecided").hazard, Some(Hazard::MemoryBarrier));
    }

    /// Cyberpunk 2077 with DLSS Frame Generation: Streamline copies the colour input and the depth
    /// into fresh images in DLSS's buffer, the depth as `R32_SFLOAT`, and DLSS's input kernel names
    /// those. Such a launch is DLSS's input launch, with the R32 image as its depth; a launch that
    /// also names a real depth image keeps that one.
    #[test]
    fn a_depth_handed_over_as_an_r32_colour_image_is_dlss_depth() {
        let desc = |format| ImageDesc { width: 1485, height: 835, format, usage: vk::ImageUsageFlags::STORAGE, plain: true };
        let (colour, r32, mvec, d32) = (vk::Image::from_raw(0x10), vk::Image::from_raw(0x20), vk::Image::from_raw(0x30), vk::Image::from_raw(0x40));
        let images: HashMap<vk::Image, ImageDesc> = [
            (colour, desc(vk::Format::R16G16B16A16_SFLOAT)),
            (r32, desc(vk::Format::R32_SFLOAT)),
            (mvec, desc(vk::Format::R16G16_SFLOAT)),
            (d32, desc(vk::Format::D32_SFLOAT)),
        ]
        .into_iter()
        .collect();
        assert_eq!(input_launch(&images, &[colour, r32, mvec]), Some(InputLaunch::Inputs { colour, depth: r32, mvec }));
        assert_eq!(input_launch(&images, &[colour, r32, d32, mvec]), Some(InputLaunch::Inputs { colour, depth: d32, mvec }));
        // An R32 image at another extent is not taken as the depth.
        let mut other = images.clone();
        other.insert(r32, ImageDesc { width: 640, height: 360, ..desc(vk::Format::R32_SFLOAT) });
        assert_eq!(input_launch(&other, &[colour, r32, mvec]), None);
        // No motion vectors: not an input launch.
        assert_eq!(input_launch(&images, &[colour, r32]), None);
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
        assert_eq!(t.take_inline_at(), Some((cb(2), colour())));
        let point = t.inline_point(cb(2), colour()).expect("held inside the buffer");
        assert_eq!((point.colour, point.hazard), (colour(), Hazard::MemoryBarrier));
        assert_eq!((point.desc.width, point.desc.height), (1707, 960));
        // A later launch of the same buffer naming it again: no second point.
        t.launch(cb(2), Some(&params));
        assert_eq!(t.take_inline_at(), None);
        // A buffer whose launch names the colour input without depth and motion vectors (DLSS Frame
        // Generation's, Black Myth: Wukong with frame generation on): not an input launch, no hold
        // inside it. Its later input launch is not the first to name the colour input either.
        let (fg, luma) = (vk::CuFunctionNVX::from_raw(0xf1), vk::CuFunctionNVX::from_raw(0xf2));
        t.record_function(fg, "main_kernel");
        t.record_function(luma, "cuda_luma_convert_kernel");
        t.global(cb(3), Hazard::MemoryBarrier);
        t.launch_kernel(cb(3), fg, Some(&param_block(&[0x1100])));
        assert_eq!(t.take_inline_at(), None);
        t.launch(cb(3), Some(&params));
        assert_eq!(t.take_inline_at(), None);
        // DLSS Super Resolution's own kernel reading the colour input before its input kernel
        // (Cyberpunk 2077's `cuda_luma_convert_kernel`): the hold goes there.
        t.global(cb(4), Hazard::MemoryBarrier);
        t.launch_kernel(cb(4), luma, Some(&param_block(&[0x1100])));
        assert_eq!(t.take_inline_at(), Some((cb(4), colour())));
        // GTA V's pattern: nothing before the launch, the split holds it.
        let (mut t, params) = held_tracker_with(copyable);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, c).is_none());
        // Barriers on depth only (GTA V with frame generation): the split holds it too.
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), general_to_general(vk::Image::from_raw(0x200)));
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, c).is_none());
        // Not copyable (STORAGE only): nothing.
        let (mut t, params) = held_tracker();
        t.global(cb(2), Hazard::MemoryBarrier);
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, c).is_none());
        // Left in SHADER_READ_ONLY_OPTIMAL by the buffer's own barrier: held inside, moved to the
        // transfer layouts and back to that one; in a layout outside those the hold handles: nothing.
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), ImageSync { old_layout: GENERAL, ..ImageSync::to(colour(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) });
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert_eq!(t.inline_point(b, c).map(|p| p.layout), Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        let (mut t, params) = held_tracker_with(copyable);
        t.barrier(cb(2), ImageSync { old_layout: GENERAL, ..ImageSync::to(colour(), vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL) });
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        assert!(t.inline_point(b, c).is_none());
        // A launch naming no colour candidate leaves no mark.
        let (mut t, _) = held_tracker_with(copyable);
        t.launch(cb(3), Some(&param_block(&[0x1200])));
        assert_eq!(t.take_inline_at(), None);
    }

    /// An implausible game exposure (Wukong's DLSS-internal 1x1 read 59,456) stops that
    /// identification from using the game's exposure image; plausible ones (GTA V's 0.22, Cyberpunk's
    /// 3.3) never do, and a measured exposure is not judged.
    #[test]
    fn an_implausible_game_exposure_switches_that_identification_to_the_measured_one() {
        let mut session = Session::default();
        let result = |source, e| HoldResult { exposure_source: Some(source), exposure: Some(e), ..HoldResult::default() };
        session.check_exposure(7, &result(ExposureSource::Game, 0.2195));
        session.check_exposure(7, &result(ExposureSource::Game, 3.32));
        session.check_exposure(7, &result(ExposureSource::Auto, 59456.0));
        assert!(session.game_exposure_trusted(7));
        session.check_exposure(7, &result(ExposureSource::Game, 59456.0));
        assert!(!session.game_exposure_trusted(7));
        assert!(session.game_exposure_trusted(8), "a new identification starts trusted");
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
        assert!(matches!(refs.input, Some(InputLaunch::Inputs { colour, .. }) if colour == vk::Image::from_raw(0x100)));
        // A word that is one 64-bit value is not split: a key equal to its low half is not named.
        let (mut t, _) = held_tracker();
        t.launch(cb(6), Some(&param_block(&[0x1100])));
        assert_eq!(t.launch[&cb(6)].images, vec![vk::Image::from_raw(0x100)]);
    }

    /// Unreal Engine 5 (Black Myth: Wukong's benchmark): DLSS's colour input is a sampled, not storage,
    /// `B10G11R11_UFLOAT` image. Its input kernel's parameters name it as the colour input; an
    /// RGBA16F image that is only sampled still is not a candidate. The split cannot hold an
    /// R11G11B10 input, so the hold goes inside the buffer even with nothing before the launch,
    /// in the layout the buffer's barrier left it in.
    #[test]
    fn a_sampled_r11g11b10_colour_input_is_identified_and_held_inside_the_buffer() {
        let r11 = |usage| ImageDesc { width: 1488, height: 836, format: vk::Format::B10G11R11_UFLOAT_PACK32, usage, plain: true };
        let rgba = |usage| ImageDesc { width: 1488, height: 836, format: vk::Format::R16G16B16A16_SFLOAT, usage, plain: true };
        assert!(colour_candidate(&r11(vk::ImageUsageFlags::SAMPLED)));
        assert!(colour_candidate(&r11(vk::ImageUsageFlags::STORAGE)));
        assert!(!colour_candidate(&r11(vk::ImageUsageFlags::COLOR_ATTACHMENT)));
        assert!(!colour_candidate(&rgba(vk::ImageUsageFlags::SAMPLED)));
        assert!(colour_candidate(&rgba(vk::ImageUsageFlags::STORAGE)));

        let mut t = Tracker::default();
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
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
        t.swapchain(vk::SwapchainKHR::from_raw(9), Some((2560, 1440)));
        let params = param_block(&[0x1100, 0x1200, 0x1300]);
        // Settle the identification by the input kernel's parameters.
        for _ in 0..(SETTLE_SUBMITS + 4) {
            t.launch(cb(900), Some(&params));
            let _ = submit(&mut t, &[vec![cb(900)]]);
            t.free(&[cb(900)]);
        }
        let inputs = t.inputs.expect("identified");
        assert_eq!((inputs.colour.0, inputs.colour.1.format), (vk::Image::from_raw(0x100), vk::Format::B10G11R11_UFLOAT_PACK32));
        t.barrier(cb(2), ImageSync { old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, ..ImageSync::to(vk::Image::from_raw(0x100), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) });
        let _ = submit(&mut t, &[vec![cb(2)]]);
        t.begin(cb(2));
        t.launch(cb(2), Some(&params));
        let (b, c) = t.take_inline_at().expect("first colour launch");
        let point = t.inline_point(b, c).expect("held inside the buffer");
        assert_eq!((point.layout, point.hazard), (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, Hazard::NotSplittable));
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
        assert!(scan.colour_in_general());
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
        // Submits `plan`'s parts, failing the call numbered `fail`; returns the result.
        let run = |t: &mut Tracker, fail: usize| {
            let scan = t.scan(&cbs).expect("launch-bearing");
            let calls = std::cell::Cell::new(0);
            let submit = |_: &[Batch1], _: vk::Fence| {
                calls.set(calls.get() + 1);
                if calls.get() == fail { vk::Result::ERROR_OUT_OF_DEVICE_MEMORY } else { vk::Result::SUCCESS }
            };
            let mut accepted = Vec::new();
            let result = submit_around(plan(batches.clone(), scan.batch, scan.index), vk::Fence::null(), &submit, |_| false, &mut |part| accepted.extend(part));
            t.commit_submitted(accepted);
            result
        };
        // The head is refused: nothing executed, nothing tracked.
        let mut t = fixture();
        assert_eq!(run(&mut t, 1), vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(t.committed.is_empty());
        // The head is accepted, the tail refused: the head's barrier only.
        let mut t = fixture();
        assert_eq!(run(&mut t, 2), vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert_eq!(t.committed.get(&colour()), Some(&GENERAL), "the launch buffer and what follows never ran");
        // Both accepted: the last buffer's layout.
        let mut t = fixture();
        assert_eq!(run(&mut t, 0), vk::Result::SUCCESS);
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
        t.swapchain(vk::SwapchainKHR::from_raw(9), Some((2560, 1440)));
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
        assert_eq!(t.describe_word(sr_out, c), "view:output-size(2560x1440,R16G16B16A16_SFLOAT)");
        assert_eq!(t.describe_word(colour, None), "view:image(1485x836,R16G16B16A16_SFLOAT)");
        t.record_view(vk::ImageView::from_raw(0x701), vk::Image::from_raw(0x400));
        t.register(vk::ImageView::from_raw(0x701), Some(0x0140_1c00));
        t.record_view(vk::ImageView::from_raw(0x702), vk::Image::from_raw(0x500));
        t.register(vk::ImageView::from_raw(0x702), Some(0x0320_1c23));
        assert_eq!(t.describe_word(0x0320_1c23_0140_1c00, c), "[view:depth(1485x836,D32_SFLOAT_S8_UINT)|view:mvec(1485x836,R16G16_SFLOAT)]");
        let words: Vec<String> = param_block(&[]).chunks_exact(8).map(|w| t.describe_word(u64::from_le_bytes(w.try_into().unwrap()), c)).collect();
        assert_eq!(words, ["i(1485,836)", "f(0.5,0)"]);
        assert_eq!(t.describe_word(0, c), "0");
        assert_eq!(t.describe_word(0x7f12_3456_789a_bcde, c), "0x7f123456789abcde");
    }

    /// DLSS Frame Generation's launch-bearing buffers (they read depth, motion vectors and the
    /// output-size frame, never the render-size colour input) are forwarded; only Super
    /// Resolution's buffer, whose input kernel names the colour input, is the hold point, wherever
    /// the two sit in a submit or across submits.
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
        assert_eq!(t.launch[&cb(1)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Colour);
        assert_eq!(t.launch[&cb(2)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Foreign);

        // FG's submits on their own (another queue in GTA V): not held, counted.
        assert!(t.submit_ok(&[vec![cb(2)]]).is_none(), "frame generation's buffer is not the hold point");
        assert!(t.submit_ok(&[vec![cb(3)]]).is_none());
        assert_eq!(t.foreign_submits, 2);
        // An FG buffer first in the same submit: the split point is SR's buffer behind it.
        let scan = t.submit_ok(&[vec![cb(2)], vec![cb(7), cb(1)]]).expect("SR's buffer is held");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!((t.colour_submits, t.evaluations, t.foreign_submits), (1, 1, 2));
        // SR's buffer through a secondary carries its references to the primary.
        t.begin(cb(1));
        t.launch(cb(11), Some(&param_block(&[colour])));
        t.execute(cb(1), &[cb(11)]);
        assert!(t.submit_ok(&[vec![cb(3), cb(1)]]).is_some_and(|s| s.index == 1));
        assert_eq!(t.colour_submits, 2);
    }

    /// Without readable parameters (another translation layer's launch form) or before the colour
    /// input is identified, a launch-bearing buffer is held as it was before the distinction:
    /// nothing that held before stops holding.
    #[test]
    fn unreadable_parameters_or_no_colour_input_keep_the_old_rule() {
        let (mut t, [_, sr_out, ..]) = fg_tracker();
        t.launch(cb(1), Some(&param_block(&[sr_out])));
        t.launch(cb(1), None);
        assert_eq!(t.launch[&cb(1)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Unknown);
        assert!(t.submit_ok(&[vec![cb(1)]]).is_some(), "an opaque launch could be SR's: held");
        assert_eq!((t.colour_submits, t.foreign_submits), (0, 0));
        let refs = LaunchRefs::default();
        assert_eq!(refs.kind(None), LaunchKind::Unknown, "no colour input identified yet");
        // A destroyed colour view drops its key: its handle no longer matches anything.
        t.forget_view(vk::ImageView::from_raw(0x101));
        assert!(t.keys.iter().all(|k| k.2 != vk::Image::from_raw(0x100)));
        // Every registered view is a key: a launch naming only depth is someone else's, but a
        // launch whose readable parameters name no registered view at all stays undecided (held),
        // so a handle form the scan misses cannot turn Super Resolution's own buffer into a
        // forwarded one.
        let (mut t, [_, _, _, depth, _]) = fg_tracker();
        t.launch(cb(5), Some(&param_block(&[depth])));
        assert_eq!(t.launch[&cb(5)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Foreign);
        t.launch(cb(6), Some(&param_block(&[0xdead_beef_0000])));
        assert_eq!(t.launch[&cb(6)].kind(Some(vk::Image::from_raw(0x100))), LaunchKind::Unknown);
        assert!(t.submit_ok(&[vec![cb(6)]]).is_some(), "names nothing registered: held as before");
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
        let endless = [ptr(1), block.as_ptr().cast(), ptr(1), block.as_ptr().cast(), ptr(1), block.as_ptr().cast(), ptr(1), block.as_ptr().cast(), ptr(1), block.as_ptr().cast()];
        assert_eq!(unsafe { launch_params(endless.as_ptr(), 1) }, None);
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
        let undecided = t.evaluations - t.colour_submits;
        let scan = t.submit_ok(batches);
        lines.extend(t.classify_line(scan.as_ref(), undecided));
        (scan, lines)
    }

    fn identification_lines(lines: &[String]) -> Vec<&String> {
        lines.iter().filter(|l| l.starts_with("colour input:") || l.starts_with("no DLSS input")).collect()
    }

    /// (a) GTA V with DLSS Frame Generation: SR's buffer (input kernel naming colour, depth and
    /// motion vectors, then the network, the output kernel and the exposure copy) and FG's two
    /// buffers (fg_tracker's shape: FG's output-size frame beside the render-size depth and motion
    /// vectors, then the output). Identified by size at the first submit exactly as before, then by
    /// the input kernel's parameters with the very same images (no re-identification, so no skipped
    /// hold); SR's buffer is held every frame and FG's are forwarded, every frame.
    #[test]
    fn gtas_sr_and_fg_buffers_keep_their_inputs_and_hold_target_by_the_input_kernel() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let mut set = gta_registered();
        set.insert(0x420, (img(0x420), desc(2560, 1440, rgba, storage)));
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x500), h(0x700)])));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x400), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x610)])));
        t.launch(cb(2), Some(&param_block(&[h(0x420), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x400)])));
        t.launch(cb(3), Some(&param_block(&[h(0x420), h(0x400)])));
        assert_eq!(t.launch[&cb(1)].input, Some(InputLaunch::Inputs { colour: img(0x100), depth: img(0x200), mvec: img(0x300) }));
        assert_eq!(t.launch[&cb(2)].input, Some(InputLaunch::Unusable), "FG's frame is not at the depth's extent: no evidence");
        assert_eq!(t.launch[&cb(3)].input, None, "no depth: not an input launch");
        let mut lines = Vec::new();
        for frame in 0..40 {
            let (sr, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let sr = sr.expect("SR's buffer is the hold point");
            assert_eq!((sr.batch, sr.index, sr.identified_now), (0, 0, frame == 0), "frame {frame}: identified once, at the first submit");
            let i = sr.inputs.unwrap();
            assert_eq!(
                (i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw(), i.candidates, i.exposure_input.map(|e| e.0.as_raw())),
                (0x100, 0x200, 0x300, 1, Some(0x610)),
                "frame {frame}"
            );
            for fg in [cb(2), cb(3)] {
                let (s, l) = submit(&mut t, &[vec![fg]]);
                lines.extend(l);
                assert!(s.is_none(), "frame {frame}: frame generation's buffers are forwarded");
            }
        }
        assert_eq!((t.colour_submits, t.foreign_submits, t.evaluations), (40, 80, 40));
        assert_eq!(t.named, vec![Named::new(img(0x100), img(0x200), img(0x300), Some(img(0x610)))], "the output kernel names the larger output: no same-extent output pair (that is DLAA's)");
        let ids = identification_lines(&lines);
        assert_eq!(ids.len(), 2, "{lines:#?}");
        assert!(ids[0].starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT") && ids[0].ends_with("exposure input 0x610; swapchain Some((2560, 1440)); identified by size"), "{}", ids[0]);
        assert!(ids[1].starts_with("colour input: image 0x100 (1707x960 R16G16B16A16_SFLOAT") && !ids[1].contains("candidates"), "{}", ids[1]);
        assert!(ids[1].ends_with("exposure input 0x610 (named by the same command buffer); swapchain Some((2560, 1440)); identified by the input kernel's parameters (one launch names it with the depth and motion vectors)"), "{}", ids[1]);
        assert!(lines.iter().all(|l| !l.contains("output-size colour image") && !l.contains("different colour inputs") && !l.contains("several colour candidates")), "{lines:#?}");

        // fg_tracker's own data: FG's first launch is not evidence, its buffers stay forwarded.
        let (mut t, [colour, sr_out, fg_colour, depth, mvec]) = fg_tracker();
        t.launch(cb(1), Some(&param_block(&[colour, depth, mvec])));
        t.launch(cb(1), Some(&param_block(&[sr_out])));
        t.launch(cb(2), Some(&param_block(&[fg_colour, depth, mvec])));
        t.launch(cb(2), Some(&param_block(&[sr_out])));
        t.launch(cb(3), Some(&param_block(&[fg_colour, sr_out])));
        assert_eq!(t.launch[&cb(2)].input, Some(InputLaunch::Unusable));
        for _ in 0..30 {
            assert!(submit(&mut t, &[vec![cb(1)]]).0.is_some_and(|s| s.inputs.unwrap().colour.0 == img(0x100)));
            assert!(submit(&mut t, &[vec![cb(2)]]).0.is_none());
            assert!(submit(&mut t, &[vec![cb(3)]]).0.is_none());
        }
        assert_eq!((t.named.len(), t.inputs.map(|i| i.rule)), (1, Some(Rule::Params)));
        assert_eq!((t.colour_submits, t.foreign_submits), (30, 60));
    }

    /// The DLAA set (Resident Evil Requiem: `UpscalingQuality_DLSS=MaxQuality`, FG on): DLSS's colour
    /// input, depth and motion vectors are all the swapchain's 2560x1440, beside a separate output
    /// image and FG's own output-size frame.
    fn dlaa_set() -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        set_of(&[
            (0x100, desc(2560, 1440, rgba, storage)),
            (0x200, desc(2560, 1440, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)),
            (0x300, desc(2560, 1440, vk::Format::R16G16_SFLOAT, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
            (0x500, desc(2560, 1440, rgba, storage)),
            (0x600, desc(1, 1, vk::Format::R32G32B32A32_SFLOAT, storage)),
            (0x610, desc(1, 1, vk::Format::R16_SFLOAT, storage)),
            (0x700, desc(1280, 720, vk::Format::R8_UNORM, storage)),
        ])
    }

    /// (b) DLAA with frame generation: the size rule refuses DLAA by design ("no DLSS input ...;
    /// waiting", as on the rig), and the input kernel's parameters identify it once settled. FG's
    /// launch names its output-size frame with the same depth and motion vectors, but its buffer
    /// names no exposure image, so it does not count (logged once). Held from the submit after
    /// identification on, FG forwarded; the hold is size-agnostic (no padding at 2560x1440).
    #[test]
    fn dlaa_is_identified_by_the_input_kernels_parameters_and_held() {
        let mut t = tracker_with(&dlaa_set(), Some((2560, 1440)));
        assert_eq!(identify(&dlaa_set(), Some((2560, 1440))), None, "the size rule refuses DLAA");
        // SR: input kernel, network, output kernel (names input and output: only the first launch counts), exposure copy.
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x700)])));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x400), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x610)])));
        // FG: its frame with depth and motion vectors, then the output.
        t.launch(cb(2), Some(&param_block(&[h(0x500), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x400)])));
        t.launch(cb(3), Some(&param_block(&[h(0x500), h(0x400), h(0x700)])));
        assert_eq!(t.launch[&cb(2)].input, Some(InputLaunch::Inputs { colour: img(0x500), depth: img(0x200), mvec: img(0x300) }), "FG's launch has SR's shape under DLAA");
        let mut lines = Vec::new();
        let mut first_held = None;
        for frame in 0..40 {
            let (sr, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let sr = sr.expect("launch-bearing");
            match sr.inputs {
                None => assert!(first_held.is_none(), "frame {frame}: never un-identified"),
                Some(i) => {
                    assert_eq!((i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw(), i.rule, i.exposure_input.map(|e| e.0.as_raw())), (0x100, 0x200, 0x300, Rule::Params, Some(0x610)));
                    if sr.identified_now {
                        assert!(first_held.is_none());
                    } else {
                        assert!(sr.colour_in_general());
                        first_held.get_or_insert(frame);
                    }
                }
            }
            for fg in [cb(2), cb(3)] {
                let (s, l) = submit(&mut t, &[vec![fg]]);
                lines.extend(l);
                assert!(s.and_then(|s| s.inputs).is_none(), "frame {frame}: frame generation's buffer is never held with inputs");
            }
        }
        // 16 quiet submits after the first: frame 5's second FG submit decides, frame 6 is the
        // identifying (unheld) SR submit, frame 7 the first hold.
        assert_eq!(first_held, Some(7));
        assert_eq!(t.colour_submits, 40 - 6, "SR submits reading the colour input from the identifying one on");
        let ids = identification_lines(&lines);
        assert_eq!(ids.len(), 2, "{lines:#?}");
        assert!(ids[0].starts_with("no DLSS input among 8 registered views (swapchain Some((2560, 1440))); waiting. output: swapchain 2560x1440"), "{}", ids[0]);
        assert!(ids[0].contains("2560x1440 not smaller than the output"), "{}", ids[0]);
        assert!(ids[1].starts_with("colour input: image 0x100 (2560x1440 R16G16B16A16_SFLOAT"), "{}", ids[1]);
        assert!(ids[1].ends_with("exposure input 0x610 (named by the same command buffer); swapchain Some((2560, 1440)); identified by the input kernel's parameters (one launch names it with the depth and motion vectors)"), "{}", ids[1]);
        let fg_line: Vec<&String> = lines.iter().filter(|l| l.contains("output-size colour image")).collect();
        assert_eq!(fg_line.len(), 1, "{lines:#?}");
        assert!(fg_line[0].contains("(0x500 2560x1440)"), "{}", fg_line[0]);
        // The hold's sizes: no padding at 2560x1440 (or at 4K DLAA), and 4K RGBA16F fits a region.
        assert_eq!((padded(2560, 1440), padded(3840, 2160)), ((2560, 1440), (3840, 2160)));
        assert!(3840 * 2160 * TEXEL as usize <= neural_forge_protocol::MAX_FRAME);
    }

    /// (c) Frame generation alone never identifies: GTA V's FG buffers (fg_tracker's data, their
    /// frame is not at the depth's extent), and FG at native resolution (frame, depth and motion
    /// vectors all output-size, which has SR's shape) without SR, whose buffers name no exposure.
    #[test]
    fn frame_generation_buffers_alone_never_identify() {
        let (mut t, [_, sr_out, fg_colour, depth, mvec]) = fg_tracker();
        t.launch(cb(2), Some(&param_block(&[fg_colour, depth, mvec])));
        t.launch(cb(2), Some(&param_block(&[sr_out])));
        t.launch(cb(3), Some(&param_block(&[fg_colour, sr_out])));
        for _ in 0..100 {
            assert!(submit(&mut t, &[vec![cb(2)]]).0.is_none());
            assert!(submit(&mut t, &[vec![cb(3)]]).0.is_none());
        }
        assert!(t.named.is_empty() && t.named_pick.is_none());
        assert_eq!(t.inputs.map(|i| (i.colour.0.as_raw(), i.rule)), Some((0x100, Rule::Size)), "the size rule's identification stays");
        // Native resolution: no SR buffer at all.
        let mut set = dlaa_set();
        set.remove(&0x100);
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(2), Some(&param_block(&[h(0x500), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x400)])));
        let mut lines = Vec::new();
        for _ in 0..100 {
            let (s, l) = submit(&mut t, &[vec![cb(2)]]);
            lines.extend(l);
            assert!(s.and_then(|s| s.inputs).is_none(), "FG's frame is never the colour input");
        }
        assert!(t.inputs.is_none() && t.named_pick.is_none());
        assert_eq!(t.named.len(), 1, "seen, but not counted");
        assert_eq!(lines.iter().filter(|l| l.contains("names no 1x1 R16_SFLOAT exposure")).count(), 1, "{lines:#?}");
    }

    /// (d) One launch naming two colour candidates at the depth's extent: the first in parameter
    /// order is the colour input (Cyberpunk 2077's case), also when its handle is the higher one.
    /// Ambiguity between buffers falls back to the size rule and is logged once: two buffers naming
    /// different colour inputs with neither (or both) naming the exposure.
    #[test]
    fn ambiguous_parameters_fall_back_to_the_size_rule_and_log_once() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let mut set = gta_registered();
        set.insert(0x110, (img(0x110), desc(1707, 960, rgba, vk::ImageUsageFlags::STORAGE)));
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(1), Some(&param_block(&[h(0x110), h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x610)])));
        assert!(matches!(t.launch[&cb(1)].input, Some(InputLaunch::Inputs { colour, .. }) if colour == img(0x110)), "the first in parameter order");

        // Two buffers, two colour inputs, no exposure named by either.
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x110), h(0x200), h(0x300)])));
        let mut lines = Vec::new();
        for _ in 0..40 {
            for b in [cb(1), cb(2)] {
                let (_, l) = submit(&mut t, &[vec![b]]);
                lines.extend(l);
            }
        }
        assert_eq!(t.named.len(), 2);
        assert_eq!((t.named_pick, t.inputs.map(|i| (i.colour.0.as_raw(), i.rule))), (None, Some((0x100, Rule::Size))));
        let said: Vec<&String> = lines.iter().filter(|l| l.contains("different colour inputs")).collect();
        assert_eq!(said.len(), 1, "{lines:#?}");
        assert!(said[0].contains("(0x100 1707x960, 0x110 1707x960): none chosen"), "{}", said[0]);
        // With exactly one of them naming the exposure, that one is chosen.
        t.launch(cb(1), Some(&param_block(&[h(0x610)])));
        for _ in 0..20 {
            for b in [cb(1), cb(2)] {
                submit(&mut t, &[vec![b]]);
            }
        }
        assert_eq!(t.inputs.map(|i| (i.colour.0.as_raw(), i.rule)), Some((0x100, Rule::Params)));
    }

    /// (e) Crimson Desert's set (two render-size candidates, the lowest handle the one DLSS's buffer
    /// names; multi frame generation with ~5 FG submits per frame): the same colour input and hold
    /// target, by size first and by the input kernel's parameters once settled, without a
    /// re-identification; the size rule's other candidate stays listed and counted as before.
    #[test]
    fn crimson_like_set_keeps_its_identification_and_hold_target() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
        let set = set_of(&[
            (0x100, desc(1516, 852, rgba, storage)),
            (0x110, desc(1516, 852, rgba, storage)),
            (0x200, desc(1516, 852, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)),
            (0x300, desc(1516, 852, vk::Format::R16G16_SFLOAT, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
            (0x500, desc(2560, 1440, rgba, storage)),
            (0x610, desc(1, 1, vk::Format::R16_SFLOAT, storage)),
        ]);
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x400), h(0x610)])));
        t.launch(cb(2), Some(&param_block(&[h(0x500), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x400)])));
        let mut lines = Vec::new();
        for frame in 0..20 {
            let (s, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let s = s.expect("held");
            assert_eq!(s.identified_now, frame == 0);
            assert_eq!(s.inputs.map(|i| (i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw())), Some((0x100, 0x200, 0x300)));
            for _ in 0..5 {
                let (s, l) = submit(&mut t, &[vec![cb(2)]]);
                lines.extend(l);
                assert!(s.is_none());
            }
        }
        assert_eq!((t.colour_submits, t.foreign_submits, t.other_candidate_buffers), (20, 100, 0));
        let ids = identification_lines(&lines);
        assert_eq!(ids.len(), 2, "{lines:#?}");
        assert!(ids[0].contains("(first of 2 candidates; others 0x110 1516x852)") && ids[0].ends_with("identified by size"), "{}", ids[0]);
        assert!(ids[1].starts_with("colour input: image 0x100 (1516x852") && ids[1].contains("(the size rule's other candidates: 0x110 1516x852)"), "{}", ids[1]);
        assert!(ids[1].ends_with("identified by the input kernel's parameters (one launch names it with the depth and motion vectors)"), "{}", ids[1]);
        // A buffer naming only the other candidate is still forwarded and counted.
        t.launch(cb(5), Some(&param_block(&[h(0x110), h(0x400)])));
        assert!(submit(&mut t, &[vec![cb(5)]]).0.is_none());
        assert_eq!(t.other_candidate_buffers, 1);
    }

    /// Crimson Desert's set as the rig logged it with this bug: three 1516x852 RGBA16F storage
    /// candidates (0x100, 0x110, 0x120), depth and motion vectors at that extent, the output and
    /// FG's output-size frame, DLSS's 1x1 exposure.
    fn crimson3_set() -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
        set_of(&[
            (0x100, desc(1516, 852, rgba, storage)),
            (0x110, desc(1516, 852, rgba, storage)),
            (0x120, desc(1516, 852, rgba, storage)),
            (0x200, desc(1516, 852, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)),
            (0x300, desc(1516, 852, vk::Format::R16G16_SFLOAT, storage)),
            (0x400, desc(2560, 1440, rgba, storage)),
            (0x500, desc(2560, 1440, rgba, storage)),
            (0x610, desc(1, 1, vk::Format::R16_SFLOAT, storage)),
        ])
    }

    /// Records SR's buffer `sr` reading `colour` (input kernel with depth and motion vectors, then
    /// the output kernel and the exposure copy) and FG's buffer `cb(90)` (its output-size frame
    /// beside depth and motion vectors, then the output).
    fn crimson_frame(t: &mut Tracker, sr: vk::CommandBuffer, colour: u64) {
        t.begin(sr);
        t.launch(sr, Some(&param_block(&[h(colour), h(0x200), h(0x300)])));
        t.launch(sr, Some(&param_block(&[h(0x400), h(0x610)])));
        t.begin(cb(90));
        t.launch(cb(90), Some(&param_block(&[h(0x500), h(0x200), h(0x300)])));
        t.launch(cb(90), Some(&param_block(&[h(0x400)])));
    }

    /// The rig's bug: the size rule took the lowest handle (0x100) while DLSS Super Resolution's
    /// input kernel reads another candidate, so almost nothing was held. Here an early input launch
    /// named a third candidate once (a stale entry: the settled evidence names two colour inputs
    /// and the parameters rule chooses none, as it must have on the rig), then SR reads `named`
    /// every frame, with ~5 FG submits per frame. Every SR submit is held with `named` (by the
    /// per-buffer target until the switch, then as the colour input), the switch is logged once,
    /// and only the switching submit is the "just identified" one. For both the middle and the
    /// highest handle.
    #[test]
    fn the_input_kernel_overrides_the_size_rules_pick_among_several_candidates() {
        for (named, stale) in [(0x110, 0x120), (0x120, 0x110)] {
            let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
            let line = t.refresh().expect("identified by size");
            assert!(line.starts_with("colour input: image 0x100 (1516x852") && line.contains("(first of 3 candidates; others 0x110 1516x852, 0x120 1516x852)"), "{line}");
            assert!(t.inputs.unwrap().others.iter().flatten().any(|o| o.0 == img(named)));
            // The stale input launch (one frame, a loading screen), then the steady state.
            crimson_frame(&mut t, cb(1), stale);
            submit(&mut t, &[vec![cb(1)]]);
            let mut lines = Vec::new();
            let mut identified_at = Vec::new();
            for frame in 0..40 {
                crimson_frame(&mut t, cb(1), named);
                let (s, l) = submit(&mut t, &[vec![cb(1)]]);
                lines.extend(l);
                let s = s.expect("SR's buffer is the hold point");
                assert_eq!(s.inputs.map(|i| (i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw())), Some((named, 0x200, 0x300)), "frame {frame}");
                if s.identified_now {
                    identified_at.push(frame);
                }
                for _ in 0..5 {
                    let (s, l) = submit(&mut t, &[vec![cb(90)]]);
                    lines.extend(l);
                    assert!(s.is_none(), "frame generation's buffers are forwarded");
                }
            }
            let switch = format!("colour input switched to {} (DLSS's input kernel reads it; the size rule had picked 0x100)", hex(img(named)));
            assert_eq!(lines.iter().filter(|l| **l == switch).count(), 1, "{lines:#?}");
            assert!(lines.iter().any(|l| l.contains("different colour inputs") && l.contains("none chosen")), "the settled evidence alone chooses nothing here: {lines:#?}");
            assert_eq!(identified_at, vec![SWITCH_AFTER as usize - 1], "re-identified once, at the switch (that one submit goes to DLSS untouched)");
            let i = t.inputs.unwrap();
            assert_eq!((i.colour.0.as_raw(), i.rule, i.exposure_input.map(|e| e.0.as_raw()), i.exposure_named), (named, Rule::Params, Some(0x610), true));
            let ids = identification_lines(&lines);
            assert!(ids.last().unwrap().starts_with(&format!("colour input: image {} (1516x852", hex(img(named)))), "{ids:#?}");
            assert!(ids.last().unwrap().ends_with("identified by the input kernel's parameters (one launch names it with the depth and motion vectors)"), "{ids:#?}");
            assert_eq!((t.colour_submits, t.retargeted_submits, t.other_candidate_buffers), (41, 1 + SWITCH_AFTER as u64 - 1, 0));
            assert!(t.alternation.is_none());
        }
    }

    /// Black Myth: Wukong's benchmark with DLSS Frame Generation on destroys and re-creates the views
    /// it hands DLSS every frame; the images live on. The identification must not change with the
    /// views (it changed every frame, and the submit that identifies is never held): an image stays
    /// registered after its last view is destroyed, until the image itself is.
    #[test]
    fn views_recreated_every_frame_keep_the_identification() {
        let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
        assert!(t.refresh().is_some_and(|l| l.starts_with("colour input: image 0x100 ")));
        for frame in 0..10u64 {
            // The colour input's only view goes, and a new one is created and registered.
            let old = vk::ImageView::from_raw(if frame == 0 { 0x101 } else { 0x9000 + frame - 1 });
            t.forget_view(old);
            assert_eq!(t.refresh(), None, "frame {frame}: a destroyed view changes nothing");
            assert_eq!(t.inputs.map(|i| i.colour.0), Some(img(0x100)));
            let new = vk::ImageView::from_raw(0x9000 + frame);
            t.record_view(new, img(0x100));
            t.register(new, Some(0x5000 + frame));
            assert_eq!(t.refresh(), None, "frame {frame}: re-registering a kept image changes nothing");
        }
        assert_eq!(t.unregistered_by_view, 10);
        // The image itself destroyed: the next candidate.
        t.forget_image(img(0x100));
        assert!(t.refresh().is_some_and(|l| l.starts_with("colour input: image 0x110 ")));
        assert_eq!(t.unregistered_by_image, 1);
    }

    /// Wukong's benchmark with frame generation on hands DLSS a new depth image every frame (the
    /// colour input, motion vectors and exposure stay). A depth change alone is taken without
    /// re-identifying: no log line, and the next submit is not the "just identified" one.
    #[test]
    fn a_new_depth_image_every_frame_is_not_a_new_identification() {
        let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
        assert!(t.refresh().is_some());
        let generation = t.generation;
        for frame in 0..16u64 {
            let (old, new) = (if frame == 0 { 0x200 } else { 0x7000 + frame - 1 }, 0x7000 + frame);
            t.forget_image(img(old));
            // The gap before the next depth image is registered: nothing changes.
            assert_eq!(t.refresh(), None, "frame {frame}: no depth image for a moment");
            assert_eq!(t.inputs.map(|i| i.colour.0), Some(img(0x100)));
            let info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                extent: vk::Extent3D { width: 1516, height: 852, depth: 1 },
                format: vk::Format::D32_SFLOAT_S8_UINT,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            t.record_image(img(new), &info);
            t.record_view(vk::ImageView::from_raw(new + 1), img(new));
            t.register(vk::ImageView::from_raw(new + 1), Some(new + 0x1000));
            assert_eq!(t.refresh(), None, "frame {frame}");
            let i = t.inputs.unwrap();
            assert_eq!((i.colour.0, i.depth.0), (img(0x100), img(new)), "frame {frame}");
        }
        assert_eq!((t.generation, t.depth_changes), (generation, 32));
        // The colour input destroyed is still a new identification.
        t.forget_image(img(0x100));
        assert!(t.refresh().is_some_and(|l| l.starts_with("colour input: image 0x110 ")));
        assert_eq!(t.generation, generation + 1);
    }

    /// The input kernel's choice survives the depth image's rotation: the switch to the candidate
    /// DLSS reads is kept when the depth it was seen with is destroyed and another one registered.
    #[test]
    fn the_switch_to_the_input_kernels_candidate_survives_a_new_depth_image() {
        let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
        assert!(t.refresh().is_some());
        crimson_frame(&mut t, cb(1), 0x120);
        submit(&mut t, &[vec![cb(1)]]);
        for _ in 0..40 {
            crimson_frame(&mut t, cb(1), 0x110);
            submit(&mut t, &[vec![cb(1)]]);
        }
        assert_eq!(t.inputs.map(|i| i.colour.0), Some(img(0x110)));
        t.forget_image(img(0x200));
        let info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width: 1516, height: 852, depth: 1 },
            format: vk::Format::D32_SFLOAT_S8_UINT,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        t.record_image(img(0x7000), &info);
        t.record_view(vk::ImageView::from_raw(0x7001), img(0x7000));
        t.register(vk::ImageView::from_raw(0x7001), Some(0x8000));
        assert_eq!(t.refresh(), None, "a new depth image is no new identification");
        let i = t.inputs.unwrap();
        assert_eq!((i.colour.0, i.depth.0), (img(0x110), img(0x7000)));
    }

    /// Crimson Desert loaded into the world from the title screen (2.0.6 regression run): ten
    /// 1516x852 candidates, and in play DLSS's input kernel reads the eighth. With only the first
    /// three kept beside the lowest handle it was never switched to and nothing was held; every
    /// candidate is kept now, so each SR submit is held with it and the switch happens once. The
    /// identification line still lists three others.
    #[test]
    fn the_input_kernel_is_followed_to_any_of_many_candidates() {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut set = crimson3_set();
        for k in 3..10u64 {
            let handle = 0x100 + 0x10 * k;
            set.insert(handle, (img(handle), desc(1516, 852, rgba, storage)));
        }
        let (named, stale) = (0x170, 0x110);
        let mut t = tracker_with(&set, Some((2560, 1440)));
        let line = t.refresh().expect("identified by size");
        assert!(line.contains("(first of 10 candidates; others 0x110 1516x852, 0x120 1516x852, 0x130 1516x852, 6 more)"), "{line}");
        crimson_frame(&mut t, cb(1), stale);
        submit(&mut t, &[vec![cb(1)]]);
        let mut lines = Vec::new();
        for frame in 0..40 {
            crimson_frame(&mut t, cb(1), named);
            let (s, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let s = s.expect("SR's buffer is the hold point");
            assert_eq!(s.inputs.map(|i| i.colour.0.as_raw()), Some(named), "frame {frame}");
            for _ in 0..5 {
                let (s, l) = submit(&mut t, &[vec![cb(90)]]);
                lines.extend(l);
                assert!(s.is_none(), "frame generation's buffers are forwarded");
            }
        }
        let switch = format!("colour input switched to {} (DLSS's input kernel reads it; the size rule had picked 0x100)", hex(img(named)));
        assert_eq!(lines.iter().filter(|l| **l == switch).count(), 1, "{lines:#?}");
        assert_eq!(t.inputs.unwrap().colour.0.as_raw(), named);
        assert_eq!(t.other_candidate_buffers, 0);
    }

    /// With SR's buffer in the first launch-bearing submit (its evidence available when the first
    /// identification happens), the colour input is what the input kernel names at once: no
    /// lowest-handle identification first, no switch.
    #[test]
    fn the_first_identification_prefers_the_input_kernels_candidate() {
        let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
        let mut lines = Vec::new();
        for frame in 0..20 {
            crimson_frame(&mut t, cb(1), 0x110);
            let (s, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let s = s.expect("held");
            assert_eq!((s.inputs.unwrap().colour.0.as_raw(), s.identified_now), (0x110, frame == 0), "frame {frame}");
            for _ in 0..5 {
                lines.extend(submit(&mut t, &[vec![cb(90)]]).1);
            }
        }
        let ids = identification_lines(&lines);
        assert_eq!(ids.len(), 1, "{ids:#?}");
        assert!(ids[0].starts_with("colour input: image 0x110 (1516x852") && ids[0].contains("(the size rule's other candidates: 0x100 1516x852, 0x120 1516x852)"), "{}", ids[0]);
        assert!(lines.iter().all(|l| !l.contains("switched")), "{lines:#?}");
        assert_eq!((t.colour_submits, t.retargeted_submits), (20, 0));
    }

    /// A game that alternates DLSS's colour input between two candidates frame by frame: every SR
    /// submit is held with the candidate its own input launch names, the colour input never
    /// switches (each launch naming it resets the count), and the alternation is logged once. Both
    /// with the identified colour input outside the pair and inside it.
    #[test]
    fn alternating_colour_inputs_are_each_held_with_their_own_image() {
        for pair in [[0x110, 0x120], [0x100, 0x110]] {
            let mut t = tracker_with(&crimson3_set(), Some((2560, 1440)));
            assert!(t.refresh().unwrap().starts_with("colour input: image 0x100"));
            let mut lines = Vec::new();
            for frame in 0..60 {
                let colour = pair[frame % 2];
                crimson_frame(&mut t, cb(1), colour);
                let (s, l) = submit(&mut t, &[vec![cb(1)]]);
                lines.extend(l);
                let s = s.expect("every SR submit is held");
                assert_eq!(s.inputs.map(|i| i.colour.0.as_raw()), Some(colour), "frame {frame}: held with the image its input launch names");
                assert_eq!(s.identified_now, frame == 0, "frame {frame}: never re-identified after the first");
                for _ in 0..5 {
                    lines.extend(submit(&mut t, &[vec![cb(90)]]).1);
                }
            }
            assert_eq!(t.inputs.unwrap().colour.0.as_raw(), 0x100, "{pair:x?}: no switch");
            assert!(lines.iter().all(|l| !l.contains("switched")), "{lines:#?}");
            let alternates: Vec<&String> = lines.iter().filter(|l| l.starts_with("DLSS's input kernel alternates between colour candidates")).collect();
            assert_eq!(alternates.len(), 1, "{lines:#?}");
            let expected_retargets = if pair[0] == 0x100 { 30 } else { 60 };
            assert_eq!((t.colour_submits, t.retargeted_submits, t.other_candidate_buffers), (60, expected_retargets, 0), "{pair:x?}");
        }
    }

    /// Resident Evil Requiem's registered set at DLAA (from the rig's log): four 2560x1440 RGBA16F
    /// storage images (0x100 DLSS's colour input, 0x110 its output, 0x120 and 0x130 others), depth
    /// and motion vectors at 2560x1440, eight 2560x1440 A2B10G10R10 storage images (the HDR10
    /// frames FG works on), R8/RGBA8 sRGB images, FG's 1280x720 images, and **no 1x1 exposure
    /// image**.
    fn re_set() -> BTreeMap<u64, (vk::Image, ImageDesc)> {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut list = vec![
            (0x100, desc(2560, 1440, rgba, storage)),
            (0x110, desc(2560, 1440, rgba, storage)),
            (0x120, desc(2560, 1440, rgba, storage)),
            (0x130, desc(2560, 1440, rgba, storage)),
            (0x200, desc(2560, 1440, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)),
            (0x300, desc(2560, 1440, vk::Format::R16G16_SFLOAT, storage)),
            (0x500, desc(2560, 1440, vk::Format::R8_UNORM, storage)),
            (0x510, desc(2560, 1440, vk::Format::R8G8B8A8_SRGB, vk::ImageUsageFlags::SAMPLED)),
        ];
        for k in 0..8 {
            list.push((0x400 + k * 0x10, desc(2560, 1440, vk::Format::A2B10G10R10_UNORM_PACK32, storage)));
        }
        for k in 0..4 {
            list.push((0x600 + k * 0x10, desc(1280, 720, if k % 2 == 0 { vk::Format::R8_UNORM } else { rgba }, storage)));
        }
        set_of(&list)
    }

    /// Records one frame's launches of an RE-like DLAA game without kernel names: SR (input kernel
    /// with colour, depth and motion vectors; network on FG-size scratch; output kernel naming input
    /// and output) in cb(1), and FG's two buffers: cb(2) naming its A2B10G10R10 frame with depth,
    /// motion vectors and its 1280x720 images, cb(3) its frames and images.
    fn re_launches(t: &mut Tracker) {
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch(cb(1), Some(&param_block(&[h(0x610), h(0x620)])));
        t.launch(cb(1), Some(&param_block(&[h(0x100), h(0x110), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x400), h(0x200), h(0x300), h(0x600), h(0x610)])));
        t.launch(cb(2), Some(&param_block(&[h(0x410), h(0x620)])));
        t.launch(cb(3), Some(&param_block(&[h(0x400), h(0x410), h(0x630)])));
    }

    /// Runs `frames` frames of SR's cb(1) then FG's cb(2) and cb(3) (FG 3x). Returns the frame of
    /// the first held SR submit and the lines logged; FG is never held with inputs.
    fn run_frames(t: &mut Tracker, frames: usize) -> (Option<usize>, Vec<String>) {
        let mut lines = Vec::new();
        let mut first_held = None;
        for frame in 0..frames {
            let (sr, l) = submit(t, &[vec![cb(1)]]);
            lines.extend(l);
            if sr.is_some_and(|s| s.inputs.is_some() && !s.identified_now) {
                first_held.get_or_insert(frame);
            }
            for fg in [cb(2), cb(3)] {
                let (s, l) = submit(t, &[vec![fg]]);
                lines.extend(l);
                assert!(s.and_then(|s| s.inputs).is_none(), "frame {frame}: frame generation's buffer is never held with inputs");
            }
        }
        (first_held, lines)
    }

    /// (f) Resident Evil Requiem-like DLAA without an exposure image: SR's buffer names no 1x1, so
    /// the output-size entry counts only by SR's shape: its output kernel names the input with the
    /// output, and no other buffer names it. Identified by the input kernel's parameters with no
    /// exposure input (measured from the frame), held from frame 7; FG's buffers (A2B10G10R10
    /// frames, 1280x720 images) are never evidence and never held. With FG naming an RGBA16F
    /// output-size image with depth and motion vectors too (a HUD-less colour), that entry is refused
    /// (its other buffer names it as well) and SR's is still chosen.
    #[test]
    fn re_like_dlaa_without_exposure_is_identified_by_srs_shape_and_fg_is_not() {
        let set = re_set();
        assert!(set.values().all(|(_, d)| !exposure_like(d)), "no 1x1 exposure image");
        let mut t = tracker_with(&set, Some((2560, 1440)));
        re_launches(&mut t);
        assert_eq!(t.launch[&cb(2)].input, Some(InputLaunch::Unusable), "FG's frame is A2B10G10R10: not a colour candidate");
        assert!(t.launch[&cb(1)].output_pair && !t.launch[&cb(2)].output_pair);
        let (first_held, lines) = run_frames(&mut t, 40);
        assert_eq!(first_held, Some(7));
        let i = t.inputs.expect("identified");
        assert_eq!((i.colour.0.as_raw(), i.depth.0.as_raw(), i.mvec.0.as_raw(), i.rule, i.exposure_input), (0x100, 0x200, 0x300, Rule::Params, None));
        assert_eq!(t.named, vec![Named { output_pair: true, ..Named::new(img(0x100), img(0x200), img(0x300), None) }]);
        let ids = identification_lines(&lines);
        assert!(ids.last().unwrap().contains("exposure input none (no registered 1x1 R16_SFLOAT; measured from the frame)"), "{}", ids.last().unwrap());
        assert!(ids.last().unwrap().ends_with("identified by the input kernel's parameters (one launch names it with the depth and motion vectors)"));
        assert_eq!(lines.iter().filter(|l| l.contains("taken as DLSS Super Resolution's input at DLAA")).count(), 1, "{lines:#?}");
        assert!(lines.iter().all(|l| !l.contains("not used")), "{lines:#?}");

        // FG names an RGBA16F output-size image with depth and motion vectors (SR's input shape),
        // and its other buffer names it too: refused, SR still chosen.
        let mut t = tracker_with(&set, Some((2560, 1440)));
        re_launches(&mut t);
        t.launch(cb(2), Some(&param_block(&[h(0x120), h(0x130)])));
        t.begin(cb(2));
        t.launch(cb(2), Some(&param_block(&[h(0x120), h(0x200), h(0x300), h(0x600)])));
        t.launch(cb(2), Some(&param_block(&[h(0x120), h(0x130)])));
        t.launch(cb(3), Some(&param_block(&[h(0x120), h(0x400)])));
        assert!(matches!(t.launch[&cb(2)].input, Some(InputLaunch::Inputs { colour, .. }) if colour == img(0x120)));
        assert!(t.launch[&cb(2)].output_pair, "FG's buffer has SR's whole shape here");
        let (first_held, lines) = run_frames(&mut t, 40);
        assert_eq!(first_held, Some(7));
        assert_eq!(t.inputs.map(|i| (i.colour.0.as_raw(), i.rule)), Some((0x100, Rule::Params)));
        let fg = t.named.iter().find(|n| n.colour == img(0x120)).expect("FG's entry is seen");
        assert!(fg.foreign && !fg.sr_without_exposure());
        let refused: Vec<&String> = lines.iter().filter(|l| l.contains("not used")).collect();
        assert_eq!(refused.len(), 1, "{lines:#?}");
        assert!(refused[0].contains("0x120: another launch-bearing buffer names it too"), "{}", refused[0]);
    }

    /// (g) Frame generation alone at native resolution, with SR's whole shape (its launch names an
    /// output-size RGBA16F frame with depth and motion vectors, a later launch that frame with
    /// another output-size image), never identifies: its other buffer names the frame too.
    #[test]
    fn frame_generation_with_srs_shape_but_a_second_reader_never_identifies() {
        let mut set = re_set();
        set.remove(&0x100);
        let mut t = tracker_with(&set, Some((2560, 1440)));
        t.launch(cb(2), Some(&param_block(&[h(0x120), h(0x200), h(0x300)])));
        t.launch(cb(2), Some(&param_block(&[h(0x120), h(0x130)])));
        t.launch(cb(3), Some(&param_block(&[h(0x120), h(0x400)])));
        let mut lines = Vec::new();
        for _ in 0..60 {
            for b in [cb(2), cb(3)] {
                let (s, l) = submit(&mut t, &[vec![b]]);
                lines.extend(l);
                assert!(s.and_then(|s| s.inputs).is_none());
            }
        }
        assert!(t.inputs.is_none() && t.named_pick.is_none());
        assert_eq!(t.named.iter().map(|n| (n.output_pair, n.foreign)).collect::<Vec<_>>(), vec![(true, true)]);
        assert_eq!(lines.iter().filter(|l| l.contains("another launch-bearing buffer names it too")).count(), 1, "{lines:#?}");
    }

    /// Kernel names as `vkCreateCuFunctionNVX` gives them: GTA V's SR input kernel, NGX's input
    /// kernel family, Ray Reconstruction's network (some names shared with Frame Generation), and
    /// everything else.
    #[test]
    fn kernels_are_told_apart_by_name() {
        for sr in ["hiluma_engine_input_depthinv_mvlo_hdr_v2_rel", "cuda_engine_input_kernel_rel_hdr_colvar_mvlo"] {
            assert_eq!(Kernel::of(sr), Kernel::SrInput, "{sr}");
        }
        for rr in ["custom_block0_conv0_kernel", "custom_block1_hf_kernel", "custom_upsample_hf_kernel", "k_initial_merge", "k_central_block"] {
            assert_eq!(Kernel::of(rr), Kernel::RayReconstruction, "{rr}");
        }
        for other in ["main_kernel", "k_conv_fp16_nhwc", "k_pooling", "dltss_pwin_enc0_layer", "hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel", "cuda_copy_exposure_kernel"] {
            assert_eq!(Kernel::of(other), Kernel::Other, "{other}");
        }
    }

    /// Creates the kernels of `names` on `t` (handles 0x9000 + index) and returns their handles.
    fn kernels(t: &mut Tracker, names: &[&str]) -> Vec<vk::CuFunctionNVX> {
        names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let f = vk::CuFunctionNVX::from_raw(0x9000 + k as u64);
                t.record_function(f, name);
                f
            })
            .collect()
    }

    /// (h) DLSS Ray Reconstruction (Resident Evil Requiem with ray tracing, at DLAA and at Balanced):
    /// SR's kernels are created but never launched; RR's network launches, its first launch naming
    /// an RGBA16F output-size image with depth and motion vectors and a later one that image with
    /// another (SR's whole shape), and at Balanced the registered set also has a render-size group
    /// the size rule would take (RR's noisy input). With kernel names known, neither rule identifies
    /// anything, no buffer is held, and "DLSS Ray Reconstruction detected" is logged once. Without
    /// the names (the same launches), SR's shape would have been taken: the names are what keep RR out.
    /// Kernel names never gate identification or holds: a newer DLSS's Super Resolution
    /// (Crimson Desert) launches `custom_block*` kernels and no `hiluma_engine_input*`, and the
    /// name gate switched it off. Recording a name must leave the decision to the image rules.
    #[test]
    fn kernel_names_do_not_gate_identification() {
        let mut t = Tracker::default();
        t.record_function(vk::CuFunctionNVX::from_raw(0x1), "custom_block0_conv0_c8_kernel");
        t.record_function(vk::CuFunctionNVX::from_raw(0x2), "k_initial_merge");
        assert!(!t.names_known, "names must not switch the gates on");
        assert!(t.sr_running(), "with names off, every buffer stays eligible as before");
    }

    #[test]
    #[ignore = "kernel-name gating is off (GATE_BY_KERNEL_NAME): Crimson Desert's DLSS Super Resolution launches custom_block* kernels; see Tracker::record_function"]
    fn ray_reconstruction_is_never_identified_or_held() {
        let mut set = re_set();
        set.remove(&0x100);
        // Balanced: RR's render-size inputs, which the size rule alone would identify.
        let b10 = vk::Format::B10G11R11_UFLOAT_PACK32;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        set.insert(0x700, (img(0x700), desc(1486, 836, vk::Format::R16G16B16A16_SFLOAT, storage)));
        set.insert(0x710, (img(0x710), desc(1486, 836, b10, storage)));
        set.insert(0x720, (img(0x720), desc(1486, 836, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)));
        set.insert(0x730, (img(0x730), desc(1486, 836, vk::Format::R16G16_SFLOAT, storage)));
        assert!(identify(&set, Some((2560, 1440))).is_some(), "the size rule alone takes RR's render-size input");
        let record = |t: &mut Tracker, f: &[vk::CuFunctionNVX]| {
            t.launch_kernel(cb(1), f[2], Some(&param_block(&[h(0x120), h(0x200), h(0x300), h(0x500)])));
            t.launch_kernel(cb(1), f[3], Some(&param_block(&[h(0x600), h(0x610)])));
            t.launch_kernel(cb(1), f[4], Some(&param_block(&[h(0x120), h(0x130)])));
            t.launch_kernel(cb(2), f[5], Some(&param_block(&[h(0x400), h(0x200), h(0x300), h(0x620)])));
            t.launch_kernel(cb(2), f[6], Some(&param_block(&[h(0x410), h(0x630)])));
        };
        let names = [
            "hiluma_engine_input_depthinv_mvlo_hdr_v2_rel",
            "cuda_engine_input_kernel_rel_hdr_colvar_mvlo",
            "custom_block0_conv0_kernel",
            "k_central_block",
            "custom_upsample_hf_kernel",
            "main_kernel",
            "k_conv_fp16_nhwc",
        ];
        let mut t = tracker_with(&set, Some((2560, 1440)));
        let f = kernels(&mut t, &names);
        record(&mut t, &f);
        assert_eq!(t.launch[&cb(1)].input_sr, Some(false), "RR's first launch is not SR's input kernel");
        let mut lines = Vec::new();
        for _ in 0..60 {
            for b in [cb(1), cb(2)] {
                let (s, l) = submit(&mut t, &[vec![b]]);
                lines.extend(l);
                assert!(s.is_none(), "with kernel names, a buffer launching no SR input kernel is never the hold point");
            }
        }
        assert!(t.inputs.is_none() && t.named.is_empty() && t.named_pick.is_none(), "{:?}", t.named);
        let rr: Vec<&String> = lines.iter().filter(|l| l.starts_with("DLSS Ray Reconstruction detected")).collect();
        assert_eq!(rr.len(), 1, "{lines:#?}");
        assert!(rr[0].contains("custom_block0_conv0_kernel, k_central_block, custom_upsample_hf_kernel launched"), "{}", rr[0]);
        let ids = identification_lines(&lines);
        assert!(ids.iter().all(|l| l.starts_with("no DLSS input")), "{ids:#?}");
        assert!(ids.last().unwrap().contains("no DLSS Super Resolution input kernel"), "{}", ids.last().unwrap());

        // The same launches without kernel names: SR's shape would be taken (what the names prevent).
        let mut t = tracker_with(&set, Some((2560, 1440)));
        record(&mut t, &[vk::CuFunctionNVX::null(); 7]);
        for _ in 0..60 {
            for b in [cb(1), cb(2)] {
                submit(&mut t, &[vec![b]]);
            }
        }
        assert!(t.inputs.is_some(), "without names the shape rules alone identify");
    }

    /// (i) GTA V with kernel names known (SR's input kernel `hiluma_engine_input_*` launching in its
    /// buffer, FG launching `main_kernel`, `custom_block0_convPre_kernel` and `k_initial_merge`, names
    /// it shares with Ray Reconstruction): exactly as without names (identified by size at the first
    /// submit, then by the parameters, SR held every frame, FG forwarded), and no Ray Reconstruction
    /// line. The same for the DLAA SR set (with an exposure image) and for RE-like DLAA without one.
    #[test]
    #[ignore = "kernel-name gating is off (GATE_BY_KERNEL_NAME): Crimson Desert's DLSS Super Resolution launches custom_block* kernels; see Tracker::record_function"]
    fn with_kernel_names_sr_games_keep_their_identification_and_hold_target() {
        let names = ["hiluma_engine_input_depthinv_mvlo_hdr_v2_rel", "dltss_pwin_enc0_layer", "hiluma_engine_output_depthinv_mvlo_hdr_max_v2_rel", "cuda_copy_exposure_kernel", "main_kernel", "custom_block0_convPre_kernel", "k_initial_merge"];
        // GTA V.
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let mut set = gta_registered();
        set.insert(0x420, (img(0x420), desc(2560, 1440, rgba, storage)));
        let mut t = tracker_with(&set, Some((2560, 1440)));
        let f = kernels(&mut t, &names);
        t.launch_kernel(cb(1), f[0], Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch_kernel(cb(1), f[1], Some(&param_block(&[h(0x500), h(0x700)])));
        t.launch_kernel(cb(1), f[2], Some(&param_block(&[h(0x100), h(0x400), h(0x200), h(0x300)])));
        t.launch_kernel(cb(1), f[3], Some(&param_block(&[h(0x610)])));
        t.launch_kernel(cb(2), f[5], Some(&param_block(&[h(0x420), h(0x200), h(0x300)])));
        t.launch_kernel(cb(2), f[4], Some(&param_block(&[h(0x400)])));
        t.launch_kernel(cb(3), f[6], Some(&param_block(&[h(0x420), h(0x400)])));
        let mut lines = Vec::new();
        for frame in 0..40 {
            let (sr, l) = submit(&mut t, &[vec![cb(1)]]);
            lines.extend(l);
            let sr = sr.expect("SR's buffer is the hold point");
            assert_eq!(sr.identified_now, frame == 0, "frame {frame}");
            assert_eq!(sr.inputs.map(|i| (i.colour.0.as_raw(), i.exposure_input.map(|e| e.0.as_raw()))), Some((0x100, Some(0x610))));
            for fg in [cb(2), cb(3)] {
                let (s, l) = submit(&mut t, &[vec![fg]]);
                lines.extend(l);
                assert!(s.is_none(), "frame {frame}: FG forwarded");
            }
        }
        assert_eq!((t.colour_submits, t.foreign_submits), (40, 80));
        assert_eq!(t.inputs.map(|i| i.rule), Some(Rule::Params));
        assert!(lines.iter().all(|l| !l.contains("Ray Reconstruction") && !l.contains("not used")), "{lines:#?}");

        // RE-like DLAA without exposure, SR running: identified as without names.
        let mut t = tracker_with(&re_set(), Some((2560, 1440)));
        let f = kernels(&mut t, &names);
        t.launch_kernel(cb(1), f[0], Some(&param_block(&[h(0x100), h(0x200), h(0x300)])));
        t.launch_kernel(cb(1), f[1], Some(&param_block(&[h(0x610), h(0x620)])));
        t.launch_kernel(cb(1), f[2], Some(&param_block(&[h(0x100), h(0x110), h(0x200), h(0x300)])));
        // FG with SR's whole shape and no second reader: only the names keep it out (it would make
        // two SR-like entries, and none would be chosen).
        t.launch_kernel(cb(2), f[5], Some(&param_block(&[h(0x120), h(0x200), h(0x300)])));
        t.launch_kernel(cb(2), f[4], Some(&param_block(&[h(0x120), h(0x130)])));
        t.launch_kernel(cb(3), f[6], Some(&param_block(&[h(0x400), h(0x410)])));
        assert_eq!(t.launch[&cb(2)].input_sr, Some(false));
        let (first_held, lines) = run_frames(&mut t, 40);
        assert_eq!(first_held, Some(7));
        assert_eq!(t.inputs.map(|i| (i.colour.0.as_raw(), i.rule, i.exposure_input)), Some((0x100, Rule::Params, None)));
        assert_eq!(t.named.len(), 1, "FG's launch is not evidence with names known: {:?}", t.named);
        assert!(lines.iter().all(|l| !l.contains("Ray Reconstruction") && !l.contains("none used")), "{lines:#?}");
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
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!("target/test-scratch/preupscale-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (w, h) = (3u32, 1u32);
        let colour = vec![0u8; (4 * 2 * TEXEL) as usize];
        let mut frame = DumpFrame {
            width: w,
            height: h,
            colour,
            depth: Some((vec![0u8; 12], vk::Format::D32_SFLOAT_S8_UINT)),
            mvec: Some(vec![0u8; 12]),
            frame: 42,
            exposure: Vec::new(),
        };
        let files = frame.write(&dir).unwrap();
        assert_eq!(files, vec!["colour.rgba16f", "depth.r32f", "mvec.rg16f", "colour-preview.png", "meta.json"], "no exposure.json without candidates");
        let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
        for needle in ["\"width\": 3", "\"height\": 1", "\"padded_width\": 4", "\"padded_height\": 2", "\"frame\": 42", "D32_SFLOAT_S8_UINT"] {
            assert!(meta.contains(needle), "{needle} in {meta}");
        }
        let rgba32: Vec<u8> = [0.5f32, 2.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        frame.exposure = vec![
            ExposureValue { image: 0x600, format: vk::Format::R32G32B32A32_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), assumed: true, bytes: Some(rgba32) },
            ExposureValue { image: 0x610, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), assumed: false, bytes: Some(vec![0x00, 0x3c]) },
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
        fn open(import: bool) -> Option<Self> {
            let (entry, instance, physical, device, queue, family) = if import {
                crate::composition::gpu::test_device_with_external_memory_host()?
            } else {
                crate::composition::gpu::test_device()?
            };
            let info = vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = unsafe { device.create_command_pool(&info, None) }.unwrap();
            Some(Self { instance, physical, device, queue, family, pool, _entry: entry })
        }

        fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.device;
            unsafe {
                let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                let cmd = d.allocate_command_buffers(&alloc).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                d.queue_submit(self.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.queue).unwrap();
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        fn image(&self, width: u32, height: u32) -> (vk::Image, vk::DeviceMemory) {
            let d = &self.device;
            let info = vk::ImageCreateInfo::builder()
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
                .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
                .unwrap();
            let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.unwrap();
            unsafe { d.bind_image_memory(image, memory, 0) }.unwrap();
            (image, memory)
        }

        fn barrier(cmd: vk::CommandBuffer, d: &ash::Device, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
            let b = vk::ImageMemoryBarrier::builder()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                .build();
            unsafe { d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]) };
        }

        /// Uploads `bytes` (tightly packed RGBA16F) and leaves the image in GENERAL, like DLSS's input.
        fn upload(&self, image: vk::Image, width: u32, height: u32, bytes: &[u8]) {
            let staging = own_host_buffer(&self.device, &self.instance, self.physical, bytes.len() as u64).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.ptr, bytes.len()) };
            let d = &self.device;
            self.one_shot(|cmd| {
                Self::barrier(cmd, d, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                let region = vk::BufferImageCopy { image_subresource: colour_layers(), image_extent: vk::Extent3D { width, height, depth: 1 }, ..Default::default() };
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
                let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).build();
                unsafe { d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[open], &[], &[]) };
                let region = vk::BufferImageCopy { image_subresource: colour_layers(), image_extent: vk::Extent3D { width, height, depth: 1 }, ..Default::default() };
                unsafe { d.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, staging.buffer, &[region]) };
                let to_host = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ).build();
                unsafe { d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[to_host], &[], &[]) };
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

    /// The fake helper's "model": swaps red and blue, bit-exactly.
    fn swap_rb(texels: &mut [u8]) {
        for t in texels.chunks_exact_mut(TEXEL as usize) {
            let (r, rest) = t.split_at_mut(2);
            r.swap_with_slice(&mut rest[2..4]);
        }
    }

    /// Another fake "model", in the encoded domain: `0.9 * v + 0.05` per colour channel, clamped
    /// to [0, 0.9995] as the real model's answers are; alpha 1.
    fn affine(texels: &mut [u8]) {
        for t in texels.chunks_exact_mut(TEXEL as usize) {
            for c in 0..3 {
                let v = f16_to_f32(u16::from_le_bytes([t[c * 2], t[c * 2 + 1]]));
                t[c * 2..c * 2 + 2].copy_from_slice(&f32_to_f16((0.9 * v + 0.05).clamp(0.0, 0.9995)).to_le_bytes());
            }
            t[6..8].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        }
    }

    /// A stand-in helper on the header: keeps its heartbeat moving and, unless muted, answers each
    /// slot-0 request by transforming the proxy region into the answer region at the size the
    /// layer published (marking it evaluated in `seq_eval`), or, with `echo` set, by copying the
    /// proxy back unchanged without marking it, as the real helper does with no feature built.
    struct FakeHelper {
        stop: Arc<AtomicBool>,
        mute: Arc<AtomicBool>,
        echo: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl FakeHelper {
        fn start(shm: &ShmClient, transform: fn(&mut [u8])) -> Self {
            let header = shm.test_header_ptr();
            let proxy = shm.proxy_region(Slot::Primary).unwrap().0 as usize;
            let answer = shm.answer_region(Slot::Primary).unwrap().0 as usize;
            let stop = Arc::new(AtomicBool::new(false));
            let mute = Arc::new(AtomicBool::new(false));
            let echo = Arc::new(AtomicBool::new(false));
            let (s, m, e) = (stop.clone(), mute.clone(), echo.clone());
            let thread = std::thread::spawn(move || {
                // SAFETY: the mapping outlives the thread (joined before the test ends).
                let hdr = unsafe { &*(header as *const neural_forge_protocol::ShmHeader) };
                hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);
                while !s.load(Ordering::Relaxed) {
                    hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
                    let req = hdr.seq_req.load(Ordering::Acquire);
                    if !m.load(Ordering::Relaxed) && req != 0 && hdr.seq_resp.load(Ordering::Relaxed) != req {
                        let (w, h) = (hdr.width.load(Ordering::Relaxed), hdr.height.load(Ordering::Relaxed));
                        assert_eq!(hdr.proxy_format.load(Ordering::Relaxed), neural_forge_protocol::enums::proxy_format::RGBA16F);
                        let n = w as usize * h as usize * TEXEL as usize;
                        // SAFETY: both regions are `MAX_FRAME`-sized and mapped for the process.
                        let mut texels = unsafe { std::slice::from_raw_parts(proxy as *const u8, n) }.to_vec();
                        let echoing = e.load(Ordering::Relaxed);
                        if !echoing {
                            transform(&mut texels);
                        }
                        unsafe { std::ptr::copy_nonoverlapping(texels.as_ptr(), answer as *mut u8, n) };
                        hdr.answered_w.store(w, Ordering::Relaxed);
                        hdr.answered_h.store(h, Ordering::Relaxed);
                        if !echoing {
                            hdr.seq_eval.store(req, Ordering::Relaxed);
                        }
                        hdr.seq_resp.store(req, Ordering::Release);
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
            });
            Self { stop, mute, echo, thread: Some(thread) }
        }
    }

    impl Drop for FakeHelper {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    fn scratch_shm(tag: &str) -> ShmClient {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .unwrap()
            .join(format!("target/test-scratch/preupscale-{}-{tag}-{}/shm.bin", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path.display().to_string()));
        shm
    }

    /// The live toggle switches the model off in the pre-upscaler path: with `enabled` off (what F11
    /// flips), `apply_model` off, or the helper gone, model mode holds nothing, and toggling back on
    /// holds again. The diagnostics ignore the toggle; off never holds.
    #[test]
    fn the_live_toggle_switches_model_mode_holds_off_and_on() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let mut shm = ShmClient::test_over_header(&header);
        let (enabled, apply_model) = (&header.enabled, &header.apply_model);
        // A running helper: its heartbeat moves before every check (the first sample only primes).
        header.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);
        let _ = shm.helper_alive();
        let beat = || {
            header.heartbeat.fetch_add(1, Ordering::Relaxed);
        };
        let mut session = Session::default();
        enabled.store(1, Ordering::Relaxed);
        apply_model.store(1, Ordering::Relaxed);
        beat();
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), Some(false), "on: held, no dump");
        // F11 (the hotkey poller flips exactly this field).
        enabled.store(0, Ordering::Relaxed);
        beat();
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), None, "toggled off: nothing held");
        assert_eq!(gate(Mode::Identity, &mut session, &mut shm, true), Some(false), "identity ignores the toggle");
        assert_eq!(gate(Mode::Roundtrip, &mut session, &mut shm, true), Some(false), "roundtrip ignores the toggle");
        assert_eq!(gate(Mode::Off, &mut session, &mut shm, true), None);
        enabled.store(1, Ordering::Relaxed);
        beat();
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), Some(false), "toggled back on: held again");
        apply_model.store(0, Ordering::Relaxed);
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), None, "apply_model off: nothing held");
        apply_model.store(1, Ordering::Relaxed);
        // The helper stops: its heartbeat freezes, and after the liveness window nothing is held.
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), None, "no helper: never waited for");
        beat();
        assert_eq!(gate(Mode::Model, &mut session, &mut shm, true), Some(false), "the helper is back: held again");
    }

    fn wait_for_helper(shm: &mut ShmClient) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !shm.helper_alive() {
            assert!(Instant::now() < deadline, "the fake helper's heartbeat is never seen");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // ---- The circuit breaker and the hand-back window. ----

    /// What [`HeaderHelper`] does with each slot-0 request.
    const SILENT: u32 = 0;
    const ECHO: u32 = 1;
    const MODEL: u32 = 2;

    /// A header-only stand-in helper (no pixel regions): keeps its heartbeat moving and answers
    /// each slot-0 request at once at the published size, as the model (`seq_eval`), as an echo,
    /// or not at all.
    struct HeaderHelper {
        stop: Arc<AtomicBool>,
        mode: Arc<std::sync::atomic::AtomicU32>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl HeaderHelper {
        fn start(header: &neural_forge_protocol::ShmHeader, mode: u32) -> Self {
            let ptr = std::ptr::from_ref(header) as usize;
            let stop = Arc::new(AtomicBool::new(false));
            let mode = Arc::new(std::sync::atomic::AtomicU32::new(mode));
            let (s, m) = (stop.clone(), mode.clone());
            let thread = std::thread::spawn(move || {
                // SAFETY: the header outlives the thread (joined on drop, before the test ends).
                let hdr = unsafe { &*(ptr as *const neural_forge_protocol::ShmHeader) };
                hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);
                while !s.load(Ordering::Relaxed) {
                    hdr.heartbeat.fetch_add(1, Ordering::Relaxed);
                    let req = hdr.seq_req.load(Ordering::Acquire);
                    let mode = m.load(Ordering::Relaxed);
                    if mode != SILENT && req != 0 && hdr.seq_resp.load(Ordering::Relaxed) != req {
                        hdr.answered_w.store(hdr.width.load(Ordering::Relaxed), Ordering::Relaxed);
                        hdr.answered_h.store(hdr.height.load(Ordering::Relaxed), Ordering::Relaxed);
                        if mode == MODEL {
                            hdr.seq_eval.store(req, Ordering::Relaxed);
                        }
                        hdr.seq_resp.store(req, Ordering::Release);
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
            });
            Self { stop, mode, thread: Some(thread) }
        }

        fn set(&self, mode: u32) {
            self.mode.store(mode, Ordering::Relaxed);
        }
    }

    impl Drop for HeaderHelper {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    /// One model-mode DLSS submit as the device hook sees it, minus the GPU: the breaker decides
    /// (at the simulated time `now`), and a held submit waits for the helper's answer within
    /// `budget`. Returns the answer (`None`: forwarded untouched) and the time spent waiting.
    fn submit_once(breaker: &mut Breaker, shm: &mut ShmClient, now: Instant, budget: Duration) -> (Option<Answer>, Duration) {
        if !breaker.allow(now, shm.model_up()) {
            return (None, Duration::ZERO);
        }
        let started = Instant::now();
        let answer = await_answer(shm, 64, 64, budget, &mut HoldTiming::default());
        let waited = started.elapsed();
        breaker.record(now, answer == Answer::Model);
        (Some(answer), waited)
    }

    /// The rig's stuck-feature case: the helper keeps answering in time, but only with echoes (no
    /// feature built), and still says `model_up=1`. Within [`BREAKER_MISSES`] holds the breaker
    /// opens; while it is open nothing is held and nothing waits; after the cool-down one probe goes
    /// through (an echo again: open again); once the model answers, the probe closes it and every
    /// submit is held again. With `model_up=0` it opens at once, and the probes still go out (so a
    /// helper that has not built yet is asked to).
    #[test]
    fn the_breaker_opens_on_echoes_stops_waiting_and_closes_when_the_model_answers() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        header.model_up.store(1, Ordering::Relaxed);
        let mut shm = ShmClient::test_over_header(&header);
        let helper = HeaderHelper::start(&header, MODEL);
        wait_for_helper(&mut shm);
        let budget = ANSWER_BUDGET;
        let mut breaker = Breaker::default();
        let t0 = Instant::now();

        // The first hold is a probe; a model answer closes the breaker.
        assert_eq!(submit_once(&mut breaker, &mut shm, t0, budget).0, Some(Answer::Model));
        assert!(breaker.is_closed() && !breaker.paused());

        // Echoes: held (and waited for, briefly) until the breaker opens, within BREAKER_MISSES.
        helper.set(ECHO);
        let mut held = 0;
        while let (Some(answer), _) = submit_once(&mut breaker, &mut shm, t0, budget) {
            assert_eq!(answer, Answer::Echo);
            held += 1;
            assert!(held <= BREAKER_MISSES, "the breaker never opened");
        }
        assert_eq!(held, BREAKER_MISSES, "opens after exactly BREAKER_MISSES echoed holds");
        assert!(breaker.paused());

        // Open: a frame's worth of submits for the whole cool-down, none held, no wait at all.
        let started = Instant::now();
        for i in 0..240u32 {
            let now = t0 + BREAKER_COOL_DOWN * i / 240;
            let (answer, waited) = submit_once(&mut breaker, &mut shm, now, budget);
            assert_eq!((answer, waited), (None, Duration::ZERO), "submit {i} at +{:?} was held", now - t0);
        }
        assert!(started.elapsed() < budget, "forwarding 240 submits took {:?}", started.elapsed());

        // After the cool-down: one probe, an echo again, so it opens again for another cool-down.
        let t1 = t0 + BREAKER_COOL_DOWN;
        assert_eq!(submit_once(&mut breaker, &mut shm, t1, budget).0, Some(Answer::Echo));
        assert_eq!(submit_once(&mut breaker, &mut shm, t1, budget).0, None, "a failed probe opens it again");
        assert_eq!(submit_once(&mut breaker, &mut shm, t1 + BREAKER_COOL_DOWN / 2, budget).0, None);

        // The helper recovers: the next probe gets the model's answer and holding resumes.
        helper.set(MODEL);
        let t2 = t1 + BREAKER_COOL_DOWN;
        assert_eq!(submit_once(&mut breaker, &mut shm, t2, budget).0, Some(Answer::Model));
        assert!(breaker.is_closed() && !breaker.paused());
        for _ in 0..20 {
            assert_eq!(submit_once(&mut breaker, &mut shm, t2, budget).0, Some(Answer::Model), "closed: every submit held");
        }

        // The helper says it has no model: open at once, without a single wait.
        header.model_up.store(0, Ordering::Relaxed);
        assert_eq!(submit_once(&mut breaker, &mut shm, t2, budget), (None, Duration::ZERO));
        assert!(breaker.paused());
        // The probe still goes out with model_up=0 (a helper that has not built yet builds on a
        // request), and a model answer closes it.
        header.model_up.store(1, Ordering::Relaxed);
        assert_eq!(submit_once(&mut breaker, &mut shm, t2 + BREAKER_COOL_DOWN, budget).0, Some(Answer::Model));
        assert!(breaker.is_closed());
    }

    /// A helper that has stopped answering in time (alive, but silent): each hold costs the whole
    /// budget, so the breaker caps the damage at BREAKER_MISSES budgets and then waits no more.
    #[test]
    fn the_breaker_opens_on_late_answers_after_breaker_misses_budgets() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        header.model_up.store(1, Ordering::Relaxed);
        let mut shm = ShmClient::test_over_header(&header);
        let helper = HeaderHelper::start(&header, MODEL);
        wait_for_helper(&mut shm);
        let budget = Duration::from_millis(5);
        let mut breaker = Breaker::default();
        let t0 = Instant::now();
        assert_eq!(submit_once(&mut breaker, &mut shm, t0, budget).0, Some(Answer::Model));
        helper.set(SILENT);
        // Only the misses that open the breaker are held, each waiting about one budget; every
        // submit after that goes through without a wait. (Not a total over all 100: a shared CI
        // runner's scheduling jitter on the eight bounded waits alone pushed that past its margin.)
        let mut held = 0;
        let mut longest = Duration::ZERO;
        for _ in 0..100 {
            let (answer, w) = submit_once(&mut breaker, &mut shm, t0, budget);
            match answer {
                Some(answer) => {
                    assert!(matches!(answer, Answer::Missed(_)), "{answer:?}");
                    held += 1;
                    longest = longest.max(w);
                }
                None => assert_eq!(w, Duration::ZERO, "a submit forwarded by the open breaker waited"),
            }
        }
        assert_eq!(held, BREAKER_MISSES);
        assert!(longest < budget * 4, "a held miss waited {longest:?} against a {budget:?} budget");
        drop(helper);
    }

    /// The post-upscaler hand-back: once a hold on a device has asked the helper, the post path
    /// stays off until DLSS has not run for HAND_BACK, so a loading screen (no DLSS for a few to
    /// ~20 s) never hands back, while DLSS switched off does after HAND_BACK. A device that never
    /// got that far is never affected.
    #[test]
    fn the_post_path_takes_over_only_after_hand_back_without_dlss() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        // Never held: the post path runs, DLSS or not.
        assert!(!post_off(false, Some(t0), t0));
        assert!(!post_off(false, None, t0));
        // Held, and DLSS was last seen at t0 (the game went to a loading screen).
        assert!(post_off(true, Some(t0), s(0.6)), "0.6 s into a loading screen (the old 500 ms hand-back)");
        assert!(post_off(true, Some(t0), s(20.0)), "a long loading screen");
        assert!(post_off(true, Some(t0), s(HAND_BACK.as_secs_f32() - 0.01)));
        assert!(!post_off(true, Some(t0), s(HAND_BACK.as_secs_f32() + 0.01)), "DLSS off for longer than HAND_BACK: the post path resumes");
        assert!(!post_off(true, None, t0));
        // A clock read before the last DLSS submit (another thread's) is not "long ago".
        assert!(post_off(true, Some(s(1.0)), t0));

        // Through a Session: holds and DLSS submits refresh the window; skipped holds count too.
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = ShmClient::test_over_header(&header);
        let mut session = Session::default();
        session.saw_dlss();
        assert!(!post_off(session.engaged, session.last_dlss, Instant::now()), "not before the first hold asked the helper");
        let skipped = HoldResult { miss: Some("the exposure value is not usable (zero, negative or not finite)"), waits_consumed: true, ..Default::default() };
        session.note(&shm, &skipped, Duration::from_millis(1), (64, 64));
        assert!(!session.engaged, "a hold that never reached the helper does not engage the device");
        let held = HoldResult { waits_consumed: true, wrote_back: true, evaluated: true, request: Some(1), ..Default::default() };
        session.note(&shm, &held, Duration::from_millis(10), (64, 64));
        assert!(session.engaged && session.breaker.is_closed());
        let last = session.last_dlss.expect("a hold is a DLSS submit");
        assert!(post_off(session.engaged, session.last_dlss, last + Duration::from_secs(10)));
        // An echoed first hold engages the device (the post path stays off, so its output-size
        // requests cannot fight the probes over the helper's feature) and opens the breaker.
        let mut fresh = Session::default();
        let echoed = HoldResult { waits_consumed: true, echoed: true, request: Some(1), ..Default::default() };
        fresh.note(&shm, &echoed, Duration::from_millis(1), (64, 64));
        assert!(fresh.engaged && fresh.breaker.paused(), "the first hold was an echo: engaged, paused");
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 1, "an echo is counted as a miss");
        assert_eq!(header.preupscale_state.load(Ordering::Relaxed), 3, "and the state says paused");
    }

    /// Before the device first holds, the post path often has a slot-0 request in flight. A DLSS
    /// submit that finds it is not held, and must neither engage the device (which would switch
    /// the post path off with no pre-path request ever made) nor feed the breaker. A request of
    /// this path's own still in flight (an answer over budget) is booked as a late answer.
    #[test]
    fn a_slot_the_post_path_left_busy_does_not_engage_the_device() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let mut shm = ShmClient::test_over_header(&header);
        let mut session = Session::default();
        session.saw_dlss();
        // The post path's request, unanswered (no helper here).
        assert!(shm.begin_async_request(Slot::Primary));
        for _ in 0..(2 * BREAKER_MISSES) {
            session.note_busy_slot(&shm, Mode::Model, (64, 64));
        }
        assert!(!session.engaged, "no request of the pre path reached the helper");
        assert!(!post_off(session.engaged, session.last_dlss, Instant::now()), "the post path keeps running");
        assert!(!session.breaker.paused(), "the breaker heard nothing");
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 0);

        // A hold of this path's own that ran over budget: its request is still in flight.
        let mut shm = ShmClient::test_over_header(&header);
        assert!(shm.begin_async_request(Slot::Primary));
        let late = HoldResult { waits_consumed: true, over_budget: true, request: shm.pending_request(Slot::Primary), miss: Some("the answer was over budget"), ..Default::default() };
        session.note(&shm, &late, Duration::from_millis(30), (64, 64));
        assert!(session.engaged, "a request that reached the helper engages the device");
        session.note_busy_slot(&shm, Mode::Model, (64, 64));
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 2, "the late answer and the busy slot it left");
        // Identity and the other diagnostics never book it.
        session.note_busy_slot(&shm, Mode::Identity, (64, 64));
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 2);
    }

    /// Holds that fail on the layer's side before asking the helper (an exposure value of 0 every
    /// frame, a submit whose pNext chain is refused, resources that cannot be built) count toward
    /// the breaker, so they stop costing a capture and a fence drain per frame; and after
    /// BREAKER_MISSES of them in a row an engaged device hands back to the post path (which would
    /// otherwise stay off for as long as DLSS runs while the model is never applied).
    #[test]
    fn holds_failing_before_the_helper_open_the_breaker_and_hand_back_to_the_post_path() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = ShmClient::test_over_header(&header);
        let extent = (64, 64);

        // Classification: only model-mode misses with no request and no missing answer are local.
        let mut exposure = HoldResult { waits_consumed: true, miss: Some("the exposure value is not usable (zero, negative or not finite)"), ..Default::default() };
        exposure.mark_local(Mode::Model);
        assert!(exposure.local_failure);
        let mut roundtrip = HoldResult { miss: Some("x"), ..Default::default() };
        roundtrip.mark_local(Mode::Roundtrip);
        assert!(!roundtrip.local_failure);
        let mut late = HoldResult { miss: Some("the answer was over budget"), over_budget: true, request: Some(3), ..Default::default() };
        late.mark_local(Mode::Model);
        assert!(!late.local_failure);
        // A request left in flight over budget claims slot 0's answer as much as a written-back one
        // (the pipelined post path would otherwise read the late answer as its own); a hold that
        // never asked does not.
        assert!(late.claims_slot0() && !late.wrote_back);
        assert!(HoldResult { wrote_back: true, evaluated: true, request: Some(4), ..Default::default() }.claims_slot0());
        assert!(!HoldResult { wrote_back: true, ..Default::default() }.claims_slot0(), "identity");
        assert!(!exposure.claims_slot0());
        let mut unstarted = HoldResult { miss: Some("the request could not be started"), over_budget: true, ..Default::default() };
        unstarted.mark_local(Mode::Model);
        assert!(!unstarted.local_failure, "the helper's side, counted as a late answer already");

        // A device that has run the model before the upscaler, then fails every hold locally.
        let mut session = Session::default();
        session.saw_dlss();
        let good = HoldResult { waits_consumed: true, wrote_back: true, evaluated: true, request: Some(1), ..Default::default() };
        session.note(&shm, &good, Duration::from_millis(10), extent);
        assert!(session.engaged && session.breaker.is_closed());
        for k in 1..BREAKER_MISSES {
            session.note(&shm, &exposure, Duration::from_millis(5), extent);
            session.saw_dlss();
            assert!(session.breaker.is_closed() && session.engaged, "only {k} in a row");
        }
        // A request that reaches the helper resets the streak.
        session.note(&shm, &good, Duration::from_millis(10), extent);
        for _ in 0..BREAKER_MISSES {
            session.note(&shm, &exposure, Duration::from_millis(5), extent);
            session.saw_dlss();
        }
        assert!(session.breaker.paused(), "BREAKER_MISSES local failures in a row open the breaker");
        assert!(!session.engaged && !post_off(session.engaged, session.last_dlss, Instant::now()), "and the post path takes over although DLSS still runs");
        assert_ne!(header.preupscale_state.load(Ordering::Relaxed), 3, "not 'paused before the upscaler': the post path runs");

        // A device that never asked the helper: a local failure neither engages it nor stalls.
        let mut fresh = Session::default();
        fresh.saw_dlss();
        let parse = HoldResult::local("a VkSubmitInfo pNext chain the hold does not handle");
        fresh.note(&shm, &parse, Duration::ZERO, extent);
        assert!(!fresh.engaged && fresh.breaker.paused(), "the first hold failed: probe again later, post path on");
        assert!(!post_off(fresh.engaged, fresh.last_dlss, Instant::now()));
    }

    // ---- The HDR encode: a CPU reference and the GPU checks against it. ----

    /// Single to half precision, round to nearest even.
    fn f32_to_f16(v: f32) -> u16 {
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
            if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
        }

        fn srgb_eotf(s: f64) -> f64 {
            let s = s.clamp(0.0, 1.0);
            if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
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

    fn answer_bytes(shm: &ShmClient, pw: u32, ph: u32) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(shm.answer_region(Slot::Primary).unwrap().0, (pw * ph) as usize * TEXEL as usize) }.to_vec()
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

    /// The write-back: per channel, the original where the encoded input (`proxy`) was >= 0.999,
    /// else the CPU inverse of the answer (`answer`) within relative 2e-3; alpha the original's.
    #[allow(clippy::too_many_arguments)]
    fn check_decode(after: &[u8], original: &[u8], proxy: &[u8], answer: &[u8], w: u32, h: u32, e: f32, white: f32) {
        let (pw, _) = padded(w, h);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let got = half_at(after, w, x, y, c);
                    let orig = half_at(original, w, x, y, c);
                    let enc = f16_to_f32(half_at(proxy, pw, x, y, c));
                    if enc >= hdr::CLAMPED {
                        assert_eq!(got, orig, "a clamped highlight keeps its original value at {x},{y} channel {c} (encoded {enc})");
                        continue;
                    }
                    let want = hdr_ref::decode(f16_to_f32(half_at(answer, pw, x, y, c)), e, white);
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
        let info = vk::ImageCreateInfo::builder()
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
        let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.unwrap();
        unsafe { d.bind_image_memory(image, memory, 0) }.unwrap();
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
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

    /// The hold on a 17x9 RGBA16F image in GENERAL (odd in both directions): identity leaves the
    /// image bit-identical and pads the proxy with the edge (the raw copy-through); model sends the
    /// HDR-encoded frame and writes the fake helper's answer (red and blue swapped, in the encoded
    /// domain) back through the inverse with the padding cropped and clamped highlights kept
    /// (check (c)); a helper that does not answer leaves the image untouched and the hold returns
    /// within its budget. Once with the SHM regions imported (zero-copy), once staged.
    #[test]
    fn a_hold_is_picture_neutral_in_identity_applies_the_answer_in_model_and_fails_open_on_timeout() {
        for import in [true, false] {
            let Some(gpu) = Gpu::open(import) else {
                eprintln!("preupscale hold test (import={import}): no suitable Vulkan device, skipping");
                continue;
            };
            let (w, h) = (17u32, 9u32);
            let (pw, ph) = padded(w, h);
            let (image, memory) = gpu.image(w, h);
            let original = hdr_pattern(w, h);
            gpu.upload(image, w, h, &original);
            assert_eq!(gpu.read(image, w, h), original, "the test's own upload round-trips");
            let (exposure, exposure_mem) = exposure_image(&gpu, 0.128);

            let mut shm = scratch_shm(if import { "import" } else { "staged" });
            let helper = FakeHelper::start(&shm, swap_rb);
            let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, import) }.expect("resources");
            if cfg!(target_pointer_width = "64") {
                assert_eq!(res.imported(), import, "imported exactly when the device can");
            }
            let white = hdr::DEFAULT_PAPER_WHITE;
            let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white, identification: 1 };
            let (device, queue) = (&gpu.device, gpu.queue);
            let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
            };

            // Identity: same bytes back, and the proxy region holds the padded capture.
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Identity, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(result.waits_consumed && result.wrote_back, "{result:?}");
            assert_eq!(result.miss, None);
            assert_eq!(result.request, None, "identity never asks the helper");
            assert_eq!(gpu.read(image, w, h), original, "identity is bit-identical");
            let proxy = proxy_bytes(&shm, pw, ph);
            let texel = |buf: &[u8], x: u32, y: u32, row: u32| buf[((y * row + x) as usize * TEXEL as usize)..][..TEXEL as usize].to_vec();
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(texel(&proxy, x, y, pw), texel(&original, x, y, w), "proxy texel {x},{y}");
                }
                assert_eq!(texel(&proxy, pw - 1, y, pw), texel(&original, w - 1, y, w), "padding column duplicates the edge, row {y}");
            }
            for x in 0..pw {
                assert_eq!(texel(&proxy, x, ph - 1, pw), texel(&proxy, x, h - 1, pw), "padding row duplicates the edge, column {x}");
            }

            // Model: the frame goes out encoded, the helper's answer comes back decoded, padding
            // cropped, clamped highlights kept.
            wait_for_helper(&mut shm);
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Model, false, 1, Duration::from_secs(10), &mut submit) };
            assert!(result.wrote_back, "{result:?}");
            assert_eq!(result.request, Some(shm.last_request(Slot::Primary)), "the request that reached the helper");
            assert_eq!(shm.answered_dims(), Some((pw, ph)), "the helper was asked for the padded size");
            let e = result.exposure.expect("the exposure value was read");
            assert!((e - 0.128).abs() < 1e-3, "the exposure image's value: {e}");
            let proxy = proxy_bytes(&shm, pw, ph);
            let clamped = check_encode(&proxy, &original, w, h, e, white);
            assert!(clamped > 0, "the pattern reaches the shoulder's top");
            let mut answer = proxy.clone();
            swap_rb(&mut answer);
            assert_eq!(answer_bytes(&shm, pw, ph), answer, "the fake helper answered");
            let after_model = gpu.read(image, w, h);
            assert_ne!(after_model, original, "the answer was applied");
            check_decode(&after_model, &original, &proxy, &answer, w, h, e, white);

            // Echo: the helper answers in time but with the frame itself (no model built): nothing
            // is written back, and the hold says so (the breaker counts it).
            helper.echo.store(true, Ordering::Relaxed);
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Model, false, 2, ANSWER_BUDGET, &mut submit) };
            assert!(result.echoed && !result.evaluated && !result.wrote_back && !result.over_budget, "{result:?}");
            assert_eq!(gpu.read(image, w, h), after_model, "an echo leaves the frame untouched");
            helper.echo.store(false, Ordering::Relaxed);

            // Timeout: the helper is alive but silent; the image is left alone, promptly.
            helper.mute.store(true, Ordering::Relaxed);
            let started = Instant::now();
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Model, false, 2, ANSWER_BUDGET, &mut submit) };
            let took = started.elapsed();
            assert!(!result.wrote_back && result.over_budget, "{result:?}");
            assert!(result.request.is_some() && shm.pending_request(Slot::Primary) == result.request, "its own request is still in flight");
            assert!(result.waits_consumed, "the capture itself went out (the waits are consumed either way)");
            assert!(took < Duration::from_secs(1), "a late answer must not hold the submit beyond its budget (took {took:?})");
            assert_eq!(gpu.read(image, w, h), after_model, "a missed answer leaves the frame untouched");

            eprintln!("preupscale hold test (import={import}): identity, model (HDR encode/decode, {clamped} clamped channels kept) and timeout checked");
            drop(helper);
            finish(&gpu, res, &[(image, memory), (exposure, exposure_mem)]);
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
    /// also checked against the CPU inverse of the proxy within 2e-3 (`check_decode`). No helper is
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
            let mut shm = scratch_shm(if import { "rt-import" } else { "rt-staged" });
            let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, import) }.expect("resources");
            let white = 2.5;
            let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white, identification: 1 };
            let (device, queue) = (&gpu.device, gpu.queue);
            let mut submits = Vec::new();
            let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                submits.push(which);
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
            };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Roundtrip, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{result:?}");
            assert!(!shm.has_pending_request(Slot::Primary), "roundtrip never calls the helper");
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
                        if enc >= hdr::CLAMPED {
                            kept += 1;
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
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Roundtrip, false, 1, ANSWER_BUDGET, &mut submit) };
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

    /// The hold inside DLSS's command buffer, end to end on one device with two queues: the
    /// application's buffer is parked at the recorded hold on queue 0 while the worker runs
    /// `run_hold` (roundtrip: encode, then decode of the encoded proxy) on the staging image on
    /// queue 1, then releases it; the buffer's copy back puts the result into the colour input
    /// before the buffer goes on. The frame must come back as roundtrip gives it (within the
    /// half-float steps the roundtrip test allows, loosely here), and the application's buffer must
    /// not finish before the release. Skips without a second queue (lavapipe).
    #[test]
    fn a_hold_inside_the_buffer_runs_on_the_side_queue_and_writes_back_through_the_staging_image() {
        let Some((entry, instance, physical, device, q0, q1, family, import)) = crate::composition::gpu::test_device_two_queues() else {
            eprintln!("inline hold test: no device with two queues in family 0, skipping");
            return;
        };
        let info = vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&info, None) }.unwrap();
        let gpu = Gpu { instance, physical, device, queue: q0, family, pool, _entry: entry };
        let (w, h) = (64u32, 36u32);
        let (image, memory) = gpu.image(w, h);
        let original = hdr_pattern(w, h);
        gpu.upload(image, w, h, &original);
        let (exposure, exposure_mem) = exposure_image(&gpu, 0.128);
        let shm = scratch_shm(if import { "inline-import" } else { "inline-staged" });
        let res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, import) }.expect("resources");
        let shared = Arc::new(Mutex::new((res, shm, None::<HoldResult>)));
        let (dev, inst, phys, side) = (gpu.device.clone(), gpu.instance.clone(), gpu.physical, q1);
        let held = shared.clone();
        let hold: inline::HoldFn = Box::new(move |job, queue| {
            assert_eq!(queue, side);
            let mut g = held.lock().unwrap();
            let (res, shm, out) = &mut *g;
            let target = Target {
                colour: job.image,
                width: w,
                height: h,
                depth: None,
                mvec: None,
                exposure: [None; MAX_EXPOSURE],
                exposure_input: job.exposure,
                paper_white: 2.5,
                identification: 1,
            };
            let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                dev.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
            };
            let result = unsafe { run_hold(&dev, &inst, phys, res, shm, &target, Mode::Roundtrip, false, 0, ANSWER_BUDGET, &mut submit) };
            let mut ms = None;
            assert!(res.wait_idle(&dev, &mut ms), "the write-back finished");
            *out = Some(result);
        });
        let device = Arc::new(gpu.device.clone());
        let inline = inline::Inline::start(device, &gpu.instance, gpu.physical, q1, family, family, hold).expect("worker");
        let d = &gpu.device;
        unsafe {
            let cb = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(gpu.pool).command_buffer_count(1)).unwrap()[0];
            let begin = vk::CommandBufferBeginInfo::default();
            d.begin_command_buffer(cb, &begin).unwrap();
            inline.begin(cb, begin.flags);
            let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE;
            let point = InlinePoint {
                colour: image,
                desc: ImageDesc { width: w, height: h, format: vk::Format::R16G16B16A16_SFLOAT, usage, plain: true },
                layout: vk::ImageLayout::GENERAL,
                exposure_input: exposure_aux(exposure),
                identification: 1,
                hazard: Hazard::MemoryBarrier,
            };
            assert!(inline.record(cb, point));
            d.end_command_buffer(cb).unwrap();
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let jobs = inline.jobs_for([cb], Some(family));
            d.queue_submit(q0, &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()], fence).unwrap();
            inline.dispatch(jobs);
            d.wait_for_fences(&[fence], true, 10_000_000_000).expect("the application's buffer finished");
            let result = shared.lock().unwrap().2.take().expect("the worker ran the hold");
            assert!(result.wrote_back && result.miss.is_none(), "{result:?}");
            let after = gpu.read(image, w, h);
            let (mut worst, mut compared) = (0f64, 0);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let x0 = f64::from(f16_to_f32(half_at(&original, w, x, y, c)));
                        let got = f64::from(f16_to_f32(half_at(&after, w, x, y, c)));
                        if x0 > 1e-3 && x0 < 5.0 {
                            worst = worst.max((got - x0).abs() / x0);
                            compared += 1;
                        }
                    }
                }
            }
            eprintln!("inline hold test (import={import}): {compared} channels, worst relative error {worst:.2e}");
            assert!(compared > 0 && worst < 0.05, "the frame did not come back through the staging image (worst {worst})");
            d.destroy_fence(fence, None);
            inline::destroy(d.handle());
        }
        let (res, _shm, _) = Arc::try_unwrap(shared).ok().expect("worker joined").into_inner().unwrap();
        finish(&gpu, res, &[(image, memory), (exposure, exposure_mem)]);
    }

    /// Check (c) with a non-trivial answer: the fake helper maps every encoded channel through
    /// `0.9 v + 0.05` (clamped to the model's [0, 0.9995]); the write-back is the CPU inverse of
    /// that answer, the clamped highlights the original. Even size (no padding), staged.
    #[test]
    fn model_mode_writes_back_the_inverse_of_the_answer() {
        let Some(gpu) = Gpu::open(false) else {
            eprintln!("preupscale model inverse test: no Vulkan device, skipping");
            return;
        };
        let (w, h) = (16u32, 6u32);
        let (image, memory) = gpu.image(w, h);
        let original = hdr_pattern(w, h);
        gpu.upload(image, w, h, &original);
        let (exposure, exposure_mem) = exposure_image(&gpu, 0.1581);
        let mut shm = scratch_shm("affine");
        let helper = FakeHelper::start(&shm, affine);
        wait_for_helper(&mut shm);
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let white = hdr::DEFAULT_PAPER_WHITE;
        let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white, identification: 1 };
        let (device, queue) = (&gpu.device, gpu.queue);
        let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
        };
        let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Model, false, 0, Duration::from_secs(10), &mut submit) };
        assert!(result.wrote_back, "{result:?}");
        let e = result.exposure.unwrap();
        let proxy = proxy_bytes(&shm, w, h);
        check_encode(&proxy, &original, w, h, e, white);
        let mut answer = proxy.clone();
        affine(&mut answer);
        assert_eq!(answer_bytes(&shm, w, h), answer);
        let after = gpu.read(image, w, h);
        check_decode(&after, &original, &proxy, &answer, w, h, e, white);
        // Not the identity: a midtone moved by the answer's change, through the inverse.
        let mid = f16_to_f32(half_at(&original, w, 8, 2, 1));
        let moved = f16_to_f32(half_at(&after, w, 8, 2, 1));
        assert!((moved - mid).abs() > 0.01 * mid, "{mid} -> {moved}");
        drop(helper);
        finish(&gpu, res, &[(image, memory), (exposure, exposure_mem)]);
    }

    /// Check (d): a readable exposure image holding 0: model and roundtrip modes leave the frame
    /// untouched and never call the helper (the capture runs, the write-back is skipped). No
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
        let mut shm = scratch_shm("noexposure");
        let helper = FakeHelper::start(&shm, swap_rb);
        wait_for_helper(&mut shm);
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let (device, queue) = (&gpu.device, gpu.queue);
        let mut submits = Vec::new();
        let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            submits.push(which);
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
        };
        let base = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: None, paper_white: hdr::DEFAULT_PAPER_WHITE, identification: 1 };
        for mode in [Mode::Model, Mode::Roundtrip] {
            let target = Target { exposure_input: exposure_aux(zero), ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, mode, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(result.waits_consumed && !result.wrote_back && !result.over_budget, "{mode:?}: {result:?}");
            assert_eq!((result.exposure, result.exposure_source), (Some(0.0), Some(ExposureSource::Game)));
            assert!(result.miss.is_some_and(|m| m.contains("exposure value")), "{result:?}");
        }
        assert_eq!(submits, vec![Which::Capture, Which::Capture], "captures only, no write-back, with a zero one");
        assert!(!shm.has_pending_request(Slot::Primary), "the helper was never asked");
        assert_eq!(gpu.read(image, w, h), original, "the frame is untouched");
        let unreadable = Some(Aux { readable: false, ..exposure_aux(zero).unwrap() });
        let unknown_layout = Some(Aux { layout: None, ..exposure_aux(zero).unwrap() });
        let mut plain = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
        };
        for exposure_input in [None, unreadable, unknown_layout] {
            let target = Target { exposure_input, ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Roundtrip, false, 0, ANSWER_BUDGET, &mut plain) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{exposure_input:?}: {result:?}");
            assert_eq!(result.exposure_source, Some(ExposureSource::Auto), "{exposure_input:?}");
            assert!(result.exposure.is_some_and(hdr::exposure_ok), "{result:?}");
        }
        drop(helper);
        finish(&gpu, res, &[(image, memory), (zero, zero_mem)]);
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
    fn auto_hold(gpu: &Gpu, res: &mut Resources, shm: &mut ShmClient, image: vk::Image, w: u32, h: u32, identification: u64) -> HoldResult {
        let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: None, paper_white: hdr::DEFAULT_PAPER_WHITE, identification };
        let (device, queue) = (&gpu.device, gpu.queue);
        let mut submit = |_: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
        };
        let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, res, shm, &target, Mode::Roundtrip, false, 0, ANSWER_BUDGET, &mut submit) };
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
        let mut shm = scratch_shm("auto");
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
            let result = auto_hold(&gpu, &mut res, &mut shm, image, w, h, identification);
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
        let before = auto_hold(&gpu, &mut res, &mut shm, image, w, h, identification).auto_exposure.unwrap();
        let brighter = frame(w, h, |x, y| {
            let at = ((y * w + x) as usize) * TEXEL as usize;
            let v = f16_to_f32(u16::from_le_bytes([last[at], last[at + 1]]));
            grey(v * 4.0)
        });
        gpu.upload(image, w, h, &brighter);
        let after = auto_hold(&gpu, &mut res, &mut shm, image, w, h, identification).auto_exposure.unwrap();
        let want = hdr::auto_exposure_reference(&brighter, w, h, Some(before.log2_e));
        assert!((after.log2_e - want.log2_e).abs() < 0.01, "adapted: GPU {} vs CPU {}", after.log2_e, want.log2_e);
        assert!((before.log2_e - after.log2_e - 0.05 * 2.0).abs() < 0.02, "5% of 2 EV: {} -> {}", before.log2_e, after.log2_e);
        // A new identification starts over: the target at once.
        let reset = auto_hold(&gpu, &mut res, &mut shm, image, w, h, identification + 1).auto_exposure.unwrap();
        assert!((reset.log2_e - reset.target_log2_e).abs() < 1e-6 && (reset.log2_e - after.log2_e).abs() > 1.5, "{reset:?} after {after:?}");
        finish(&gpu, res, &[(image, memory)]);
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
        let mut shm = scratch_shm("auto-rt");
        let mut res = unsafe { Resources::build(&gpu.device, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let white = hdr::DEFAULT_PAPER_WHITE;
        let mut last_e = None;
        for scale in [1.0f32, 8.0] {
            let original: Vec<u8> = {
                let base = hdr_pattern(w, h);
                base.chunks_exact(2).enumerate().flat_map(|(i, b)| {
                    let v = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    let v = if i % 4 == 3 { v } else { v * scale };
                    f32_to_f16(v).to_le_bytes()
                }).collect()
            };
            gpu.upload(image, w, h, &original);
            let result = auto_hold(&gpu, &mut res, &mut shm, image, w, h, 1);
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
                        if enc >= hdr::CLAMPED {
                            kept += 1;
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
        let d = &gpu.device;
        let (w, h) = (5u32, 3u32);
        let (image, memory) = gpu.image(w, h);
        let original = pattern(w, h);
        gpu.upload(image, w, h, &original);
        let make_sized = |format: vk::Format, usage: vk::ImageUsageFlags, (w, h): (u32, u32)| {
            let info = vk::ImageCreateInfo::builder()
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
            let memory = unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.ok()?;
            unsafe { d.bind_image_memory(image, memory, 0) }.ok()?;
            Some((image, memory))
        };
        let make = |format: vk::Format, usage: vk::ImageUsageFlags| make_sized(format, usage, (w, h));
        let (exp32, exp32_mem) = make_sized(vk::Format::R32G32B32A32_SFLOAT, vk::ImageUsageFlags::STORAGE, (1, 1)).expect("1x1 RGBA32F image");
        let (exp16, exp16_mem) = make_sized(vk::Format::R16_SFLOAT, vk::ImageUsageFlags::SAMPLED, (1, 1)).expect("1x1 R16F image");
        let depth_format = vk::Format::D32_SFLOAT_S8_UINT;
        let Some((depth, depth_mem)) = make(depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) else {
            eprintln!("preupscale dump test: no D32_SFLOAT_S8_UINT image here, skipping");
            return;
        };
        let (mvec, mvec_mem) = make(vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT).expect("RG16F image");
        let ds_range = aux_range(depth_format);
        let colour_range = aux_range(vk::Format::R16G16_SFLOAT);
        let to = |image, range, old, new| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .build()
        };
        let all = vk::PipelineStageFlags::ALL_COMMANDS;
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[], &[], &[
                to(depth, ds_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                to(mvec, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL),
                to(exp32, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL),
                to(exp16, colour_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            ]);
            d.cmd_clear_color_image(cmd, exp32, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.5, 2.0, 0.0, 1.0] }, &[colour_range]);
            d.cmd_clear_color_image(cmd, exp16, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [3.0, 0.0, 0.0, 0.0] }, &[colour_range]);
            d.cmd_clear_depth_stencil_image(cmd, depth, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearDepthStencilValue { depth: 0.25, stencil: 7 }, &[ds_range]);
            d.cmd_clear_color_image(cmd, mvec, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [1.0, -2.0, 0.0, 0.0] }, &[colour_range]);
            d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[], &[], &[
                to(depth, ds_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
                to(exp16, colour_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            ]);
        });
        let mut shm = scratch_shm("dump");
        let mut res = unsafe { Resources::build(d, &gpu.instance, gpu.physical, gpu.family, w, h, &shm, false) }.expect("resources");
        let target = Target {
            colour: image,
            width: w,
            height: h,
            depth: Some(Aux { image: depth, format: depth_format, layout: Some(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL), readable: true }),
            mvec: Some(Aux { image: mvec, format: vk::Format::R16G16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }),
            exposure: [
                Some((Aux { image: exp32, format: vk::Format::R32G32B32A32_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }, true)),
                Some((Aux { image: exp16, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL), readable: true }, false)),
                None,
                Some((Aux { image: exp16, format: vk::Format::R16_SFLOAT, layout: None, readable: true }, false)),
            ],
            exposure_input: None,
            paper_white: hdr::DEFAULT_PAPER_WHITE,
            identification: 1,
        };
        let queue = gpu.queue;
        let mut submits = Vec::new();
        let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
            submits.push(which);
            d.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
        };
        let mut result = unsafe { run_hold(d, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Dump, true, 7, ANSWER_BUDGET, &mut submit) };
        assert_eq!(submits, vec![Which::Capture], "a dump never writes back");
        assert!(!result.wrote_back && result.miss.is_none(), "{result:?}");
        let frame = result.dump.take().expect("dump bytes");
        assert_eq!((frame.width, frame.height, frame.frame), (w, h, 7));
        let (pw, ph) = padded(w, h);
        assert_eq!(frame.colour.len(), (pw * ph) as usize * TEXEL as usize);
        assert_eq!(&frame.colour[..TEXEL as usize], &original[..TEXEL as usize]);
        let (depth_bytes, format) = frame.depth.expect("depth read");
        assert_eq!(format, depth_format);
        assert!(depth_bytes.chunks_exact(4).all(|v| f32::from_le_bytes(v.try_into().unwrap()) == 0.25), "the depth aspect as float32");
        let mv = frame.mvec.expect("motion vectors read");
        assert!(mv.chunks_exact(4).all(|v| v == [0x00, 0x3c, 0x00, 0xc0]), "RG16F (1, -2)");
        assert_eq!(frame.exposure.len(), 3, "one entry per candidate");
        let rgba32: Vec<u8> = [0.5f32, 2.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(frame.exposure[0], ExposureValue { image: exp32.as_raw(), format: vk::Format::R32G32B32A32_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), assumed: true, bytes: Some(rgba32) });
        assert_eq!(frame.exposure[1].bytes.as_deref(), Some(&[0x00, 0x42][..]), "R16F 3.0, read through a transition");
        assert_eq!(frame.exposure[1].layout, Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        assert_eq!((frame.exposure[2].layout, frame.exposure[2].bytes.as_ref()), (None, None), "no layout known, not read");
        assert_eq!(gpu.read(image, w, h), original, "the colour input is untouched");
        // The depth and R16F images are back in their layouts (validation checks this use).
        gpu.one_shot(|cmd| unsafe {
            d.cmd_pipeline_barrier(cmd, all, all, vk::DependencyFlags::empty(), &[], &[], &[
                to(depth, ds_range, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageLayout::GENERAL),
                to(exp16, colour_range, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::GENERAL),
            ]);
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
}
