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
//! 2. identifies the colour input (and depth, motion vectors) from the registered set ([`identify`]);
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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle;
use vulkan_layer::LayerVulkanCommand as VulkanCommand;

use crate::shm::ShmClient;

pub(crate) mod hdr;

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

/// How long an open [`Breaker`] forwards DLSS submits untouched before one probe hold.
pub(crate) const BREAKER_COOL_DOWN: Duration = Duration::from_secs(2);

/// The longest a model-mode hold waits for the helper's answer.
pub(crate) const ANSWER_BUDGET: Duration = Duration::from_millis(30);

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
];

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

/// The DLSS inputs among the registered images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inputs {
    pub colour: (vk::Image, ImageDesc),
    pub depth: (vk::Image, ImageDesc),
    pub mvec: (vk::Image, ImageDesc),
    /// How many colour candidates there were (the lowest handle is taken).
    pub candidates: usize,
    /// The registered 1x1 float images (DLSS's exposure input among them), lowest handles first.
    /// Only dump mode reads them.
    pub exposure: [Option<(vk::Image, ImageDesc)>; MAX_EXPOSURE],
    /// DLSS's exposure input, which the HDR encode multiplies in: the registered 1x1 R16_SFLOAT
    /// image (lowest handle among several). NGX's own 1x1 RGBA32F images are not it.
    pub exposure_input: Option<(vk::Image, ImageDesc)>,
}

/// The colour input: the registered RGBA16F storage image whose extent equals a registered depth
/// image's and is smaller than the swapchain, with a registered RG16F (motion vector) image at the
/// same extent. DLSS's inputs share the render extent; its output and NGX's scratch images have no
/// depth image beside them, and DLAA (render extent == output) is refused by the swapchain test.
/// Deterministic: among several candidates, the lowest handles win.
pub(crate) fn identify(registered: &BTreeMap<u64, (vk::Image, ImageDesc)>, swapchain: Option<(u32, u32)>) -> Option<Inputs> {
    let (sw, sh) = swapchain?;
    let at = |format_ok: &dyn Fn(vk::Format) -> bool, w: u32, h: u32| {
        registered.values().find(|(_, d)| d.plain && format_ok(d.format) && (d.width, d.height) == (w, h)).copied()
    };
    let candidates: Vec<(vk::Image, ImageDesc)> = registered
        .values()
        .filter(|(_, d)| {
            d.plain
                && d.format == vk::Format::R16G16B16A16_SFLOAT
                && d.usage.contains(vk::ImageUsageFlags::STORAGE)
                && d.width <= sw
                && d.height <= sh
                && u64::from(d.width) * u64::from(d.height) < u64::from(sw) * u64::from(sh)
                && at(&|f| DEPTH_FORMATS.contains(&f), d.width, d.height).is_some()
                && at(&|f| f == vk::Format::R16G16_SFLOAT, d.width, d.height).is_some()
        })
        .copied()
        .collect();
    let colour = *candidates.first()?;
    let (w, h) = (colour.1.width, colour.1.height);
    let mut exposure = [None; MAX_EXPOSURE];
    for (slot, found) in exposure.iter_mut().zip(
        registered.values().filter(|(_, d)| d.plain && (d.width, d.height) == (1, 1) && exposure_texel_bytes(d.format).is_some()),
    ) {
        *slot = Some(*found);
    }
    let exposure_input = registered.values().find(|(_, d)| d.plain && (d.width, d.height) == (1, 1) && d.format == vk::Format::R16_SFLOAT).copied();
    Some(Inputs {
        colour,
        depth: at(&|f| DEPTH_FORMATS.contains(&f), w, h)?,
        mvec: at(&|f| f == vk::Format::R16G16_SFLOAT, w, h)?,
        candidates: candidates.len(),
        exposure,
        exposure_input,
    })
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
    /// Which first-of-a-kind classification lines were logged.
    said_kinds: u8,
    /// Layouts recorded into each command buffer for the watched images, awaiting submission.
    pending: HashMap<vk::CommandBuffer, Vec<(vk::Image, vk::ImageLayout)>>,
    /// The watched images' layouts as of the last submission (submission order, not recording order).
    committed: HashMap<vk::Image, vk::ImageLayout>,
    /// Launch-bearing submits seen.
    evaluations: u64,
    /// The inputs changed in [`Self::refresh`] and no launch-bearing submit has been scanned since
    /// ([`Scan::identified_now`]).
    identified: bool,
}

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
}

impl Scan {
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
        if self.registered.len() != before {
            self.dirty = true;
        }
        self.committed.remove(&image);
    }

    pub(crate) fn record_view(&mut self, view: vk::ImageView, image: vk::Image) {
        self.views.insert(view, image);
    }

    pub(crate) fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view);
        self.keys.retain(|k| k.1 != view);
        if self.registered.remove(&view).is_some() {
            self.dirty = true;
        }
    }

    /// A view registered through `vkGetImageViewHandle*NVX` / `vkGetImageViewAddressNVX`; `key` is
    /// the handle or address it returned (what a kernel that reads the view is given).
    pub(crate) fn register(&mut self, view: vk::ImageView, key: Option<u64>) {
        if let Some(&image) = self.views.get(&view) {
            if self.registered.insert(view, image) != Some(image) {
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

    /// `vkCmdCuLaunchKernelNVX` recorded into `command_buffer`; `params` is the launch's kernel
    /// parameter buffer when it could be read ([`launch_params`]), `None` otherwise. Every 8-byte
    /// word of it is compared with the registered views' handles and addresses.
    pub(crate) fn launch(&mut self, command_buffer: vk::CommandBuffer, params: Option<&[u8]>) {
        let refs = self.launch.entry(command_buffer).or_default();
        let Some(bytes) = params else {
            refs.opaque = true;
            return;
        };
        for word in bytes.chunks_exact(8) {
            let value = u64::from_le_bytes(word.try_into().unwrap_or_default());
            if value == 0 {
                continue;
            }
            for &(key, _, image) in &self.keys {
                if key == value && !refs.images.contains(&image) {
                    refs.images.push(image);
                }
            }
        }
    }

    /// Whether any command buffer carries a launch or recorded layouts.
    fn armed(&self) -> bool {
        !self.launch.is_empty() || !self.pending.is_empty()
    }

    pub(crate) fn begin(&mut self, command_buffer: vk::CommandBuffer) {
        self.launch.remove(&command_buffer);
        self.pending.remove(&command_buffer);
    }

    pub(crate) fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.launch.remove(cb);
            self.pending.remove(cb);
        }
    }

    pub(crate) fn execute(&mut self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        let carried: Vec<LaunchRefs> = secondaries.iter().filter_map(|cb| self.launch.get(cb)).cloned().collect();
        if !carried.is_empty() {
            let refs = self.launch.entry(primary).or_default();
            for c in carried {
                refs.opaque |= c.opaque;
                for image in c.images {
                    if !refs.images.contains(&image) {
                        refs.images.push(image);
                    }
                }
            }
        }
        let inherited: Vec<_> = secondaries.iter().filter_map(|cb| self.pending.get(cb)).flatten().copied().collect();
        if !inherited.is_empty() {
            self.pending.entry(primary).or_default().extend(inherited);
        }
    }

    fn watched(&self, image: vk::Image) -> bool {
        self.inputs.is_some_and(|i| {
            i.colour.0 == image
                || i.depth.0 == image
                || i.mvec.0 == image
                || i.exposure.iter().flatten().any(|e| e.0 == image)
                || i.exposure_input.is_some_and(|e| e.0 == image)
        })
    }

    pub(crate) fn barrier(&mut self, command_buffer: vk::CommandBuffer, image: vk::Image, layout: vk::ImageLayout) {
        if self.watched(image) {
            self.pending.entry(command_buffer).or_default().push((image, layout));
        }
    }

    fn commit(&mut self, command_buffer: vk::CommandBuffer) {
        if let Some(recorded) = self.pending.get(&command_buffer) {
            for &(image, layout) in recorded {
                self.committed.insert(image, layout);
            }
        }
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
        let registered: BTreeMap<u64, (vk::Image, ImageDesc)> = self
            .registered
            .values()
            .filter_map(|&image| self.images.get(&image).map(|d| (image.as_raw(), (image, *d))))
            .collect();
        let inputs = identify(&registered, self.swapchain_extent());
        let key = |i: Option<Inputs>| i.map(|i| (i.colour.0, i.depth.0, i.mvec.0, i.exposure.map(|e| e.map(|e| e.0)), i.exposure_input.map(|e| e.0)));
        if key(inputs) != key(self.inputs) {
            self.committed.clear();
            self.pending.clear();
            self.identified = true;
        }
        self.inputs = inputs;
        let line = match inputs {
            Some(i) => format!(
                "colour input: image {} ({}x{} {:?} {:?}){}, depth {} {:?}, motion vectors {} {:?}{}, exposure input {}; swapchain {:?}",
                hex(i.colour.0),
                i.colour.1.width,
                i.colour.1.height,
                i.colour.1.format,
                i.colour.1.usage,
                if i.candidates > 1 { format!(" (first of {} candidates)", i.candidates) } else { String::new() },
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
                i.exposure_input.map_or_else(|| "none (no registered 1x1 R16_SFLOAT)".to_string(), |(image, _)| hex(image)),
                self.swapchain_extent()
            ),
            None => format!(
                "no DLSS input among {} registered views (swapchain {:?}); waiting",
                registered.len(),
                self.swapchain_extent()
            ),
        };
        if self.announced.as_deref() == Some(line.as_str()) {
            return None;
        }
        self.announced = Some(line.clone());
        Some(line)
    }

    /// Finds the first launch-bearing command buffer in a submit (`batches`: each batch's command
    /// buffers, in order) and commits every buffer's recorded layouts in submission order, reading
    /// the watched images' layouts just before the launch buffer.
    pub(crate) fn scan(&mut self, batches: &[Vec<vk::CommandBuffer>]) -> Option<Scan> {
        if !self.armed() {
            return None;
        }
        let mut found: Option<Scan> = None;
        let mut any_foreign = false;
        let mut held_kind = None;
        for (bi, cbs) in batches.iter().enumerate() {
            for (ci, &cb) in cbs.iter().enumerate() {
                let colour = self.inputs.map(|i| i.colour.0);
                let kind = self.launch.get(&cb).map(|r| r.kind(colour));
                if kind == Some(LaunchKind::Foreign) {
                    any_foreign = true;
                }
                if found.is_none() && kind.is_some_and(|k| k != LaunchKind::Foreign) {
                    held_kind = kind;
                    let layout = |image: Option<vk::Image>| image.and_then(|i| self.committed.get(&i).copied());
                    found = Some(Scan {
                        batch: bi,
                        index: ci,
                        colour_layout: layout(self.inputs.map(|i| i.colour.0)),
                        inputs: self.inputs,
                        depth_layout: layout(self.inputs.map(|i| i.depth.0)),
                        mvec_layout: layout(self.inputs.map(|i| i.mvec.0)),
                        exposure_layouts: std::array::from_fn(|k| layout(self.inputs.and_then(|i| i.exposure[k]).map(|e| e.0))),
                        exposure_input_layout: layout(self.inputs.and_then(|i| i.exposure_input).map(|e| e.0)),
                        evaluation: self.evaluations,
                        identified_now: self.identified,
                    });
                }
                self.commit(cb);
            }
        }
        if found.is_some() {
            self.evaluations += 1;
            self.identified = false;
            if held_kind == Some(LaunchKind::Colour) {
                self.colour_submits += 1;
            }
        } else if any_foreign {
            self.foreign_submits += 1;
        }
        found
    }

    /// A first-of-its-kind classification line, once per kind: a launch-bearing submit held because
    /// a launch reads the colour input; one forwarded because its launches name other images only;
    /// one held undecided (a launch's parameters could not be read, or no colour input yet: the rule
    /// before the distinction). `undecided_before`: held-undecided submits before this scan.
    fn classify_line(&mut self, scan: Option<&Scan>, undecided_before: u64) -> Option<String> {
        let undecided = self.evaluations - self.colour_submits;
        let (bit, line) = if scan.is_some() && self.colour_submits > 0 && self.said_kinds & 1 == 0 {
            (1, "a launch-bearing submit reads DLSS's colour input (its registered handle is in a launch's parameters): DLSS Super Resolution's, the hold point")
        } else if scan.is_none() && self.foreign_submits > 0 && self.said_kinds & 2 == 0 {
            (2, "a launch-bearing submit whose CUDA launches never name DLSS's colour input (DLSS Frame Generation's, or another NGX feature's) is forwarded untouched; only the one that reads it is held")
        } else if scan.is_some() && undecided > undecided_before && self.inputs.is_some() && self.said_kinds & 4 == 0 {
            (4, "a launch-bearing submit is held undecided: a launch's parameters were not readable (not in CUDA's buffer form), so it cannot be told from DLSS Super Resolution's")
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
}

static TRACKING: LazyLock<Mutex<HashMap<vk::Device, Arc<Tracking>>>> = LazyLock::new(Default::default);

impl Tracking {
    /// A new tracker for `device`, also reachable through [`tracking_for`] (the
    /// `vkGetImageViewHandle64NVX` wrapper has only the device handle).
    pub(crate) fn new_for(device: vk::Device) -> Arc<Self> {
        let tracking = Arc::new(Self::default());
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
    pub(crate) fn launch(&self, command_buffer: vk::CommandBuffer, params: Option<&[u8]>) {
        let mut t = self.lock();
        t.launch(command_buffer, params);
        self.rearm(&t);
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

    /// Records image barriers' new layouts for the watched images.
    pub(crate) fn barriers(&self, command_buffer: vk::CommandBuffer, images: impl Iterator<Item = (vk::Image, vk::ImageLayout)>) {
        let mut t = self.lock();
        for (image, layout) in images {
            t.barrier(command_buffer, image, layout);
        }
        self.rearm(&t);
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
            let line = t.refresh();
            self.watching.store(t.inputs.is_some(), Ordering::Relaxed);
            self.extent.store(t.inputs.map_or(0, |i| u64::from(i.colour.1.width) << 32 | u64::from(i.colour.1.height)), Ordering::Relaxed);
            let undecided_before = t.evaluations - t.colour_submits;
            let foreign_before = t.foreign_submits;
            let scan = t.scan(batches);
            let kind_line = t.classify_line(scan.as_ref(), undecided_before);
            // Every 3000th forwarded one: the running counts.
            let tally = (t.foreign_submits != foreign_before && t.foreign_submits.is_multiple_of(3000)).then(|| {
                format!(
                    "launch-bearing submits: {} held reading the colour input, {} held undecided, {} forwarded untouched (their launches never name the colour input)",
                    t.colour_submits,
                    t.evaluations - t.colour_submits,
                    t.foreign_submits
                )
            });
            self.rearm(&t);
            (scan, [line, kind_line, tally])
        };
        for line in line.into_iter().flatten() {
            crate::log!("[preupscale] {line}");
            crate::logging::flush();
        }
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

#[cfg(test)]
impl Batch2 {
    fn command_buffers(&self) -> Vec<vk::CommandBuffer> {
        self.cbs.iter().map(|c| c.command_buffer).collect()
    }
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
pub(crate) fn submit_around<B: SplitBatch>(
    mut plan: Plan<B>, fence: vk::Fence, submit: &dyn Fn(&[B], vk::Fence) -> vk::Result, hold: impl FnOnce(&[B::Wait]) -> bool,
) -> vk::Result {
    if !plan.head.is_empty() {
        let result = submit(&plan.head, vk::Fence::null());
        if result != vk::Result::SUCCESS {
            return result;
        }
    }
    if !hold(&plan.capture_waits) {
        plan.restore_waits();
    }
    submit(&plan.tail, fence)
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
/// `VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT` (`VK_EXT_frame_boundary`): a debug-tool frame marker.
const FRAME_BOUNDARY_EXT: vk::StructureType = vk::StructureType::from_raw(1_000_375_001);

/// The application's `VkSubmitInfo2`s as owned batches. The pNext chain is kept as is on every part
/// of a split batch, so only structures that mean the same thing repeated are accepted (a latency
/// present id, a performance-query pass index, a frame boundary marker); anything else is refused.
///
/// # Safety
/// `submits` must be the application's own, valid for the duration of its call.
pub(crate) unsafe fn parse2(submits: &[vk::SubmitInfo2]) -> Result<Vec<Batch2>, &'static str> {
    let mut out = Vec::with_capacity(submits.len());
    for s in submits {
        let mut next = s.p_next.cast::<vk::BaseInStructure>();
        // SAFETY: a valid pNext chain of the application's own structure.
        while let Some(base) = unsafe { next.as_ref() } {
            if !matches!(base.s_type, LATENCY_SUBMISSION_PRESENT_ID_NV | FRAME_BOUNDARY_EXT | vk::StructureType::PERFORMANCE_QUERY_SUBMIT_INFO_KHR) {
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
        let (proxy_region, proxy_capacity) = shm.proxy_region(0)?;
        let (answer_region, answer_capacity) = shm.answer_region(0)?;
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
        let wait = unsafe { device.wait_for_fences(&fences, true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
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
    /// and roundtrip modes for the HDR encode; `None` when there is none (those modes then fail
    /// open). Its layout is the committed one, or `GENERAL` assumed for a storage image.
    pub exposure_input: Option<Aux>,
    /// The HDR encode's paper white ([`hdr::paper_white`]).
    pub paper_white: f32,
}

/// Which of the layer's two submissions [`run_hold`] asks the caller to make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Which {
    /// The capture: carries the moved wait semaphores.
    Capture,
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
    /// Model and roundtrip modes: the exposure value the capture read.
    pub exposure: Option<f32>,
    /// Where the hold's CPU time went (wall time on this thread).
    pub timing: HoldTiming,
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
        self.local_failure = mode == Mode::Model && self.miss.is_some() && self.request.is_none() && !self.over_budget && !self.echoed;
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
fn capture_reads(target: &Target, bufs: Option<&DumpBuffers>, exposure_buffer: Option<vk::Buffer>) -> Vec<DumpRead> {
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
/// raw, or with `hdr`, the exposure input read into the HDR pass's buffer and the colour input
/// encoded for the model ([`hdr::HdrPass::record_encode`]); in dump mode also the depth aspect, the
/// motion vectors and the 1x1 exposure candidates. Each auxiliary image is read in its committed
/// layout (transitioned to TRANSFER_SRC_OPTIMAL and back when that layout cannot be copied from).
///
/// # Safety
/// `res`'s capture command buffer must not be pending; with `hdr`, its colour view is bound to
/// `target.colour`.
unsafe fn record_capture(device: &ash::Device, res: &Resources, target: &Target, dump: Option<&DumpBuffers>, hdr: Option<&hdr::HdrPass>) -> ash::prelude::VkResult<()> {
    let cmd = res.capture_cmd;
    // SAFETY: the pool allows resetting buffers individually; the buffer is not pending.
    unsafe {
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
    }
    let aux: Vec<DumpRead> = capture_reads(target, dump, hdr.map(hdr::HdrPass::exposure_buffer));
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
            Some(pass) => pass.record_encode(device, cmd, res.proxy.buffer, target.paper_white),
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
    let close = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ).build();
    // SAFETY: as above.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
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
/// readable exposure image nothing is submitted, and with an unusable exposure value nothing is
/// written back. `submit` makes the layer's two submissions on the game's queue (the capture one with
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
        // The encode needs the game's exposure value; without a readable exposure image nothing is
        // captured and the frame goes to DLSS untouched.
        if !target.exposure_input.is_some_and(|a| a.readable && aux_copy_layout(a.layout).is_some()) {
            result.miss = Some("no readable exposure image (registered 1x1 R16_SFLOAT) for the HDR encode");
            return result;
        }
        if res.hdr.is_none() {
            // SAFETY: `device` is live.
            res.hdr = unsafe { hdr::HdrPass::build(device, instance, physical_device, res.width, res.height) };
        }
        let Some(pass) = res.hdr.as_mut() else {
            result.miss = Some("the HDR encode/decode pipelines could not be built");
            return result;
        };
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
    if unsafe { record_capture(device, res, target, dump_bufs, hdr_pass) }.is_err() {
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
    let wait = unsafe { device.wait_for_fences(&[res.capture_fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
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
                let reads = res.dump.as_ref().map(|bufs| capture_reads(target, Some(bufs), None)).unwrap_or_default();
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
            let before = shm.last_request(0);
            let answer = await_answer(shm, pw, ph, budget, &mut result.timing);
            // A request was started (`seq_req` bumped): it reached the helper.
            result.request = Some(shm.last_request(0)).filter(|&r| r != before);
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
    shm.set_frame_info(0, pw, ph, neural_forge_protocol::enums::proxy_format::RGBA16F);
    if !shm.begin_async_request(0) {
        return Answer::Missed("the request could not be started");
    }
    let start = Instant::now();
    let mut spins = 0u32;
    let mut why = "the helper did not answer";
    let answered = loop {
        match shm.poll_async_request(0) {
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
    }

    /// The phase medians, for the periodic line.
    fn phases(&self) -> String {
        format!(
            "phases ms (median): prep={:.2} capture_wait={:.2} round_trip={:.2} helper_busy={:.2} handoff={:.2} writeback={:.2}",
            median(&self.prep_ms),
            median(&self.capture_wait_ms),
            median(&self.round_trip_ms),
            median(&self.helper_busy_ms),
            median(&self.handoff_ms),
            median(&self.writeback_cpu_ms)
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
    }

    /// Whether a hold happened within [`RECENT`].
    pub(crate) fn holding(&self) -> bool {
        self.last_hold.is_some_and(|t| t.elapsed() < RECENT)
    }

    /// Whether a dump is due: the first hold after the input is identified, then whenever a
    /// one-shot `capture_request` is pending (the caller consumes it once the dump is taken).
    pub(crate) fn dump_due(&self, shm: &ShmClient) -> bool {
        !self.dumped || shm.capture_request_pending()
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
        if self.res.as_ref().is_some_and(|r| r.matches(queue_family, width, height)) {
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

    /// Books one hold's result; publishes the header fields and logs misses (sampled) and the
    /// periodic summary.
    pub(crate) fn note(&mut self, shm: &ShmClient, result: &HoldResult, cpu: Duration, extent: (u32, u32)) {
        let now = Instant::now();
        if result.wrote_back || result.dump.is_some() {
            self.last_hold = Some(now);
            self.last_dlss = Some(now);
        }
        if result.request.is_some() {
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
                "[preupscale] mode={} extent={}x{} (padded {pw}x{ph}) holds={} hold_ms median={:.2} capture_gpu_ms median={:.2} writeback_gpu_ms median={:.2} misses={} (total {}) holds_per_s={:.1}",
                mode().name(),
                extent.0,
                extent.1,
                s.holds,
                median(&s.hold_ms),
                median(&s.capture_gpu_ms),
                median(&s.writeback_gpu_ms),
                s.window_misses,
                s.misses,
                rate
            );
            crate::log!("[preupscale] {}", s.phases());
            crate::logging::flush();
            s.hold_ms.clear();
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
        if mode == Mode::Model && self.own_request.is_some() && shm.pending_request(0) == self.own_request {
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
        let result = submit_around(p, game_fence, &submit, |waits| {
            let _ = submit(&[Batch1::own(waits.to_vec(), cb(100))], own_fence);
            let _ = submit(&[Batch1::own(Vec::new(), cb(101))], own_fence);
            true
        });
        assert_eq!(result, vk::Result::SUCCESS);
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
        assert_eq!(submit_around(plan(batches[1..].to_vec(), 0, 0), game_fence, &submit, |_| false), vk::Result::SUCCESS);
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], (batches[1..].to_vec(), game_fence), "forwarded as it was");

        // A failing head is returned at once.
        let failing = |_: &[Batch1], _: vk::Fence| vk::Result::ERROR_DEVICE_LOST;
        let mut held = false;
        assert_eq!(submit_around(plan(batches.clone(), 1, 0), game_fence, &failing, |_| { held = true; true }), vk::Result::ERROR_DEVICE_LOST);
        assert!(!held);
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
            flags: vk::SubmitFlags::PROTECTED,
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
        assert_eq!((prefix.flags, prefix.p_next), (vk::SubmitFlags::PROTECTED, info.p_next), "both parts keep the flags and the repeatable chain");
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
        assert_eq!(identify(&set, None), None, "no swapchain, no comparison, no hold");
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
        t.barrier(cb(1), colour, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        t.barrier(cb(2), colour, vk::ImageLayout::GENERAL);
        t.barrier(cb(2), vk::Image::from_raw(0x999), vk::ImageLayout::GENERAL);
        t.launch(cb(5), None);
        t.execute(cb(3), &[cb(5)]);
        let scan = t.scan(&[vec![cb(1)], vec![cb(2), cb(3)]]).expect("launch-bearing");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!(scan.colour_layout, Some(vk::ImageLayout::GENERAL), "the layout as of submission order just before the launch");
        assert!(!t.committed.contains_key(&vk::Image::from_raw(0x999)), "only the inputs are watched");
        assert!(scan.identified_now && !scan.colour_in_general(), "the first launch-bearing submit since identification is not held");
        assert!(t.scan(&[vec![cb(7)]]).is_none());
        t.begin(cb(3));
        assert!(t.scan(&[vec![cb(3)]]).is_none(), "re-recorded: no longer launch-bearing");
        // A destroyed input image changes the set.
        t.forget_image(vk::Image::from_raw(0x200));
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
        t.launch(cb(1), None);
        let first = t.scan(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(first.inputs.is_some() && first.identified_now && first.colour_layout.is_none());
        assert!(!first.colour_in_general(), "identified on this submit: nothing known, not held");
        let second = t.scan(&[vec![cb(1)]]).expect("launch-bearing");
        assert!(!second.identified_now && second.colour_layout.is_none());
        assert!(second.colour_in_general(), "watched for a submit and no barrier seen: GENERAL");
        // A re-identification (DLSS's inputs re-created) starts over.
        t.lock().forget_image(vk::Image::from_raw(0x1000));
        register(0x5000);
        let again = t.scan(&[vec![cb(1)]]).expect("launch-bearing");
        assert_eq!(again.inputs.map(|i| i.colour.0), Some(vk::Image::from_raw(0x5000)));
        assert!(again.identified_now && !again.colour_in_general());
        assert!(t.scan(&[vec![cb(1)]]).expect("launch-bearing").colour_in_general());
        // A barrier out of GENERAL still refuses the hold.
        t.barriers(cb(2), [(vk::Image::from_raw(0x5000), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)].into_iter());
        assert!(!t.scan(&[vec![cb(2), cb(1)]]).expect("launch-bearing").colour_in_general());
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
        assert!(t.scan(&[vec![cb(1), cb(2)]]).is_none());
        assert_eq!(t.extent(), None);
        // DLSS records a launch: armed, and the submit carrying it is found.
        t.launch(cb(7), None);
        assert!(t.armed());
        assert!(t.scan(&[vec![cb(6), cb(7)]]).is_some_and(|s| (s.batch, s.index) == (0, 1)));
        // A secondary's launch carries over to its primary.
        t.execute(cb(8), &[cb(7)]);
        t.begin(cb(7));
        assert!(t.armed(), "the primary still carries the launch");
        t.free(&[cb(8)]);
        assert!(!t.armed(), "every launch buffer was reset or freed");
        assert!(t.scan(&[vec![cb(7), cb(8)]]).is_none());
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
        assert!(t.scan(&[vec![cb(2)]]).is_none(), "frame generation's buffer is not the hold point");
        assert!(t.scan(&[vec![cb(3)]]).is_none());
        assert_eq!(t.foreign_submits, 2);
        // An FG buffer first in the same submit: the split point is SR's buffer behind it.
        let scan = t.scan(&[vec![cb(2)], vec![cb(7), cb(1)]]).expect("SR's buffer is held");
        assert_eq!((scan.batch, scan.index), (1, 1));
        assert_eq!((t.colour_submits, t.evaluations, t.foreign_submits), (1, 1, 2));
        // SR's buffer through a secondary carries its references to the primary.
        t.begin(cb(1));
        t.launch(cb(11), Some(&param_block(&[colour])));
        t.execute(cb(1), &[cb(11)]);
        assert!(t.scan(&[vec![cb(3), cb(1)]]).is_some_and(|s| s.index == 1));
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
        assert!(t.scan(&[vec![cb(1)]]).is_some(), "an opaque launch could be SR's: held");
        assert_eq!((t.colour_submits, t.foreign_submits), (0, 0));
        let refs = LaunchRefs { opaque: false, images: vec![] };
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
        assert!(t.scan(&[vec![cb(6)]]).is_some(), "names nothing registered: held as before");
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
            let proxy = shm.proxy_region(0).unwrap().0 as usize;
            let answer = shm.answer_region(0).unwrap().0 as usize;
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
        let mut waited = Duration::ZERO;
        let mut held = 0;
        for _ in 0..100 {
            let (answer, w) = submit_once(&mut breaker, &mut shm, t0, budget);
            if let Some(answer) = answer {
                assert!(matches!(answer, Answer::Missed(_)), "{answer:?}");
                held += 1;
            }
            waited += w;
        }
        assert_eq!(held, BREAKER_MISSES);
        assert!(waited < budget * (BREAKER_MISSES + 2), "waited {waited:?} over 100 submits");
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
        assert!(shm.begin_async_request(0));
        for _ in 0..(2 * BREAKER_MISSES) {
            session.note_busy_slot(&shm, Mode::Model, (64, 64));
        }
        assert!(!session.engaged, "no request of the pre path reached the helper");
        assert!(!post_off(session.engaged, session.last_dlss, Instant::now()), "the post path keeps running");
        assert!(!session.breaker.paused(), "the breaker heard nothing");
        assert_eq!(header.preupscale_misses.load(Ordering::Relaxed), 0);

        // A hold of this path's own that ran over budget: its request is still in flight.
        let mut shm = ShmClient::test_over_header(&header);
        assert!(shm.begin_async_request(0));
        let late = HoldResult { waits_consumed: true, over_budget: true, request: shm.pending_request(0), miss: Some("the answer was over budget"), ..Default::default() };
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
        unsafe { std::slice::from_raw_parts(shm.proxy_region(0).unwrap().0, (pw * ph) as usize * TEXEL as usize) }.to_vec()
    }

    fn answer_bytes(shm: &ShmClient, pw: u32, ph: u32) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(shm.answer_region(0).unwrap().0, (pw * ph) as usize * TEXEL as usize) }.to_vec()
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
            let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white };
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
            assert_eq!(result.request, Some(shm.last_request(0)), "the request that reached the helper");
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
            assert!(result.request.is_some() && shm.pending_request(0) == result.request, "its own request is still in flight");
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
            let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white };
            let (device, queue) = (&gpu.device, gpu.queue);
            let mut submits = Vec::new();
            let mut submit = |which: Which, cmd: vk::CommandBuffer, fence: vk::Fence| unsafe {
                submits.push(which);
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence)
            };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, Mode::Roundtrip, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(result.waits_consumed && result.wrote_back && result.miss.is_none(), "{result:?}");
            assert!(!shm.has_pending_request(0), "roundtrip never calls the helper");
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
        let target = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: exposure_aux(exposure), paper_white: white };
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

    /// Check (d): no exposure image, an unreadable one, or one holding 0: model and roundtrip modes
    /// leave the frame untouched and never call the helper. Without a readable image nothing is
    /// submitted at all (the waits stay with the game's batch); with a zero value the capture runs
    /// and the write-back is skipped.
    #[test]
    fn a_missing_or_zero_exposure_leaves_the_frame_untouched() {
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
        let base = Target { colour: image, width: w, height: h, depth: None, mvec: None, exposure: [None; MAX_EXPOSURE], exposure_input: None, paper_white: hdr::DEFAULT_PAPER_WHITE };
        let unreadable = Some(Aux { readable: false, ..exposure_aux(zero).unwrap() });
        for (mode, exposure_input) in [(Mode::Model, None), (Mode::Roundtrip, None), (Mode::Model, unreadable)] {
            let target = Target { exposure_input, ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, mode, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(!result.waits_consumed && !result.wrote_back && !result.over_budget, "{mode:?}: {result:?}");
            assert!(result.miss.is_some_and(|m| m.contains("exposure")), "{result:?}");
        }
        for mode in [Mode::Model, Mode::Roundtrip] {
            let target = Target { exposure_input: exposure_aux(zero), ..base };
            let result = unsafe { run_hold(&gpu.device, &gpu.instance, gpu.physical, &mut res, &mut shm, &target, mode, false, 0, ANSWER_BUDGET, &mut submit) };
            assert!(result.waits_consumed && !result.wrote_back && !result.over_budget, "{mode:?}: {result:?}");
            assert_eq!(result.exposure, Some(0.0));
            assert!(result.miss.is_some_and(|m| m.contains("exposure value")), "{result:?}");
        }
        assert_eq!(submits, vec![Which::Capture, Which::Capture], "nothing without an exposure image; captures only, no write-back, with a zero one");
        assert!(!shm.has_pending_request(0), "the helper was never asked");
        assert_eq!(gpu.read(image, w, h), original, "the frame is untouched");
        drop(helper);
        finish(&gpu, res, &[(image, memory), (zero, zero_mem)]);
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
