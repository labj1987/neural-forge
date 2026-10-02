//! `NEURAL_FORGE_PROBE_NGX=1`: a diagnostic-only probe that watches a game's DLSS Super
//! Resolution work as it reaches Vulkan under vkd3d-proton/DXVK + DXVK-NVAPI, where DLSS
//! runs as CUDA kernels through `VK_NVX_binary_import` (`vkCmdCuLaunchKernelNVX`) on image
//! views registered through `VK_NVX_image_view_handle` (`vkGetImageViewHandleNVX`,
//! `vkGetImageViewAddressNVX`, and the newer `vkGetImageViewHandle64NVX`).
//!
//! Feasibility only (see `docs/PRE_UPSCALER_PROBE.md`): it answers whether the game's
//! internal render-resolution image is identifiable before its upscaler, and which submit
//! carries the upscaler's launches. It never changes a call: every hook forwards the
//! application's arguments unchanged and returns the next layer's result. Kernel parameter
//! pointers are opaque and never dereferenced; only their counts are logged.
//!
//! With the variable unset nothing here is reachable: [`probe_commands`] is not added to the
//! framework's hooked device commands, `entry_points` hands out the next layer's
//! `vkGetImageViewHandle64NVX` untouched, and the always-on hooks (`vkCreateImage`,
//! `vkQueueSubmit*`, `vkQueuePresentKHR`, ...) test [`enabled`] before touching any state.
//!
//! The state is process-wide behind one mutex: the game has one DLSS device, and handle
//! values are what the logs report anyway. Frames are counted between presents of any
//! swapchain the layer sees.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::CStr;
use std::sync::{LazyLock, Mutex, MutexGuard};

use ash::vk;
use ash::vk::Handle;
use vulkan_layer::LayerVulkanCommand as VulkanCommand;

use crate::probe_seq::{hex, Attachment, ImageBarrier, Recording};

/// The variable that turns the probe on. Read through `neural_forge_protocol::env`.
pub(crate) const ENV: &str = "NEURAL_FORGE_PROBE_NGX";

/// Every `SUMMARY_EVERY`th frame gets a summary line even when nothing changed.
const SUMMARY_EVERY: u64 = 60;
/// Caps on per-event lines, so a game that creates thousands of views or kernels cannot
/// flood the log.
const MAX_EVENT_LINES: u32 = 64;
/// Caps on how many distinct entries one frame's summary lists.
const MAX_LISTED: usize = 12;
/// Registered views count as settled this many frames after the last new one.
const STABLE_FRAMES: u64 = 30;
/// Full command sequences: the first `SEQ_FIRST` launch-bearing buffers ended after the views
/// settled, then one every `SEQ_EVERY` frames. The aggregate line follows `SEQ_EVERY` too.
const SEQ_FIRST: u32 = 3;
const SEQ_EVERY: u64 = 300;

/// Whether the probe is on for this process: only when the layer itself is live
/// (`NEURAL_FORGE_ENABLE`, not disabled, not a duplicate copy) and the variable is set.
/// Cached: it cannot change for the life of the process.
pub(crate) fn enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        // The variable first: with it unset, `layer_enabled` (and its duplicate-copy
        // decision) is not evaluated any earlier than it always was.
        let on = crate::env_flag(ENV) && crate::layer_enabled();
        if on {
            crate::log!("[probe-ngx] on in pid {} ({ENV}=1): NVX image-view and CUDA-launch hooks installed", std::process::id());
            crate::logging::flush();
        }
        on
    });
    *ENABLED
}

/// The device commands the probe adds to the framework's hooked set when it is on.
pub(crate) const PROBE_COMMANDS: &[VulkanCommand] = &[
    VulkanCommand::CreateImageView,
    VulkanCommand::DestroyImageView,
    VulkanCommand::GetImageViewHandleNvx,
    VulkanCommand::GetImageViewAddressNvx,
    VulkanCommand::CreateCuModuleNvx,
    VulkanCommand::CreateCuFunctionNvx,
    VulkanCommand::DestroyCuFunctionNvx,
    VulkanCommand::CmdCuLaunchKernelNvx,
    // The command sequence of launch-bearing buffers (`probe_seq`). `vkCmdCopyImage`,
    // `vkCmdBlitImage`, both pipeline barriers, `vkBeginCommandBuffer` and
    // `vkCmdExecuteCommands` are already in the default set.
    VulkanCommand::EndCommandBuffer,
    VulkanCommand::CreateFramebuffer,
    VulkanCommand::DestroyFramebuffer,
    VulkanCommand::CmdDraw,
    VulkanCommand::CmdDrawIndexed,
    VulkanCommand::CmdDrawIndirect,
    VulkanCommand::CmdDrawIndexedIndirect,
    VulkanCommand::CmdDrawIndirectCount,
    VulkanCommand::CmdDrawIndexedIndirectCount,
    VulkanCommand::CmdDrawMultiExt,
    VulkanCommand::CmdDrawMultiIndexedExt,
    VulkanCommand::CmdDrawMeshTasksExt,
    VulkanCommand::CmdDrawMeshTasksIndirectExt,
    VulkanCommand::CmdDrawMeshTasksIndirectCountExt,
    VulkanCommand::CmdDispatch,
    VulkanCommand::CmdDispatchIndirect,
    VulkanCommand::CmdDispatchBase,
    VulkanCommand::CmdExecuteGeneratedCommandsNv,
    VulkanCommand::CmdCopyImage2,
    VulkanCommand::CmdBlitImage2,
    VulkanCommand::CmdCopyBufferToImage,
    VulkanCommand::CmdCopyBufferToImage2,
    VulkanCommand::CmdClearColorImage,
    VulkanCommand::CmdClearDepthStencilImage,
    VulkanCommand::CmdClearAttachments,
    VulkanCommand::CmdResolveImage,
    VulkanCommand::CmdResolveImage2,
    VulkanCommand::CmdBeginRenderPass,
    VulkanCommand::CmdBeginRenderPass2,
    VulkanCommand::CmdEndRenderPass,
    VulkanCommand::CmdEndRenderPass2,
    VulkanCommand::CmdBeginRendering,
    VulkanCommand::CmdEndRendering,
];

/// [`PROBE_COMMANDS`] when `probe` is on, nothing otherwise.
pub(crate) fn probe_commands(probe: bool) -> &'static [VulkanCommand] {
    if probe { PROBE_COMMANDS } else { &[] }
}

/// `vkGetImageViewHandle64NVX` (`VK_NVX_image_view_handle` revision 3). Newer than the
/// pinned ash and `vulkan-layer`, so the framework passes it straight to the next layer and
/// `entry_points` wraps it itself.
pub(crate) const HANDLE64_NAME: &CStr = c"vkGetImageViewHandle64NVX";
pub(crate) type PfnGetImageViewHandle64Nvx = unsafe extern "system" fn(vk::Device, *const vk::ImageViewHandleInfoNVX) -> u64;

/// The next layer's `vkGetImageViewHandle64NVX`, per device, recorded when the layer's
/// `vkGetDeviceProcAddr` hands out the wrapper.
static HANDLE64_NEXT: LazyLock<Mutex<HashMap<vk::Device, PfnGetImageViewHandle64Nvx>>> = LazyLock::new(Default::default);

pub(crate) fn remember_handle64(device: vk::Device, next: PfnGetImageViewHandle64Nvx) {
    lock(&HANDLE64_NEXT).insert(device, next);
}

pub(crate) fn handle64_next(device: vk::Device) -> Option<PfnGetImageViewHandle64Nvx> {
    lock(&HANDLE64_NEXT).get(&device).copied()
}

pub(crate) fn forget_device(device: vk::Device) {
    lock(&HANDLE64_NEXT).remove(&device);
}

/// A poisoned mutex (a panic elsewhere while holding it) must not turn into a second panic
/// inside a hook the game called: the data is diagnostic, so keep using it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(Default::default);

fn state() -> MutexGuard<'static, State> {
    lock(&STATE)
}

fn emit(lines: Vec<String>) {
    if lines.is_empty() {
        return;
    }
    for line in &lines {
        crate::log!("[probe-ngx] {line}");
    }
    crate::logging::flush();
}

/// What `vkCreateImage` said about an image.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ImageDesc {
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
}

/// What a view covers: its image's extent at the view's base mip level, the view's own
/// format, and the image's usage. `None` for a view of an image the probe never saw created
/// (a swapchain image, or one created before the probe's first hook ran).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ViewDesc {
    width: u32,
    height: u32,
    format: i32,
    usage: u32,
}

impl std::fmt::Display for ViewDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} {:?} {:?}",
            self.width,
            self.height,
            vk::Format::from_raw(self.format),
            vk::ImageUsageFlags::from_raw(self.usage)
        )
    }
}

fn describe(view: Option<ViewDesc>) -> String {
    view.map_or_else(|| "unknown-view".to_string(), |v| v.to_string())
}

/// The shape of one `vkCmdCuLaunchKernelNVX`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct LaunchShape {
    function: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    params: usize,
    extras: usize,
}

impl std::fmt::Display for LaunchShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} grid={}x{}x{} block={}x{}x{} smem={} params={} extras={}",
            self.function, self.grid.0, self.grid.1, self.grid.2, self.block.0, self.block.1, self.block.2,
            self.shared_mem, self.params, self.extras
        )
    }
}

/// What happened between two presents.
#[derive(Default)]
struct Frame {
    launches: u32,
    /// Command buffers launches were recorded into, in first-seen order.
    cmdbufs: Vec<vk::CommandBuffer>,
    shapes: BTreeSet<LaunchShape>,
    /// Handle and address queries this frame.
    registrations: u32,
    /// Distinct view descriptions among them.
    registered: BTreeSet<Option<ViewDesc>>,
    /// `vkQueueSubmit`/`vkQueueSubmit2` calls (each call counts once) since the last present.
    submits: u32,
    /// `(submit index, queue, launches carried)` for each submit that carried a
    /// launch-bearing command buffer.
    launch_submits: Vec<(u32, vk::Queue, u32)>,
}

#[derive(Default)]
struct State {
    images: HashMap<vk::Image, ImageDesc>,
    views: HashMap<vk::ImageView, Option<ViewDesc>>,
    functions: HashMap<vk::CuFunctionNVX, String>,
    /// Launches recorded into each command buffer since it was last begun.
    launch_cmdbufs: HashMap<vk::CommandBuffer, u32>,
    /// Every view ever registered (handle or address query), for the cumulative count.
    ever_registered: HashSet<vk::ImageView>,
    frame: Frame,
    frames: u64,
    /// The distinct registered view extents of the previous frame.
    last_extents: BTreeSet<(u32, u32)>,
    any_launch_frame_logged: bool,
    /// Set by the first launch-bearing submit; reported (once) at the next present.
    first_launch_submit: Option<(u64, u32, vk::Queue)>,
    first_launch_submit_done: bool,
    module_lines: u32,
    function_lines: u32,
    registration_lines: u32,
    /// The image each live view was created on.
    view_images: HashMap<vk::ImageView, vk::Image>,
    /// Images of interest: those with a view registered through a handle/address query.
    interest: BTreeMap<vk::Image, ViewDesc>,
    /// The colour input among them (see [`identify_colour`]).
    colour: Option<vk::Image>,
    /// The frame of the last first-time view registration.
    last_new_registration: Option<u64>,
    framebuffers: HashMap<vk::Framebuffer, Vec<vk::ImageView>>,
    /// What every command buffer recorded since its last begin.
    recordings: HashMap<vk::CommandBuffer, Recording>,
    next_epoch: u64,
    /// The most recently recorded barrier on the colour input: buffer and description.
    colour_last_barrier: Option<(vk::CommandBuffer, String)>,
    seq_printed: u32,
    last_seq_frame: Option<u64>,
    /// Each launch-bearing buffer's epoch at its last submit.
    submitted_epochs: HashMap<vk::CommandBuffer, u64>,
    stats: SeqStats,
}

/// Counts over every launch-bearing command buffer ended after the views settled.
#[derive(Default)]
struct SeqStats {
    buffers: u32,
    classes: BTreeMap<&'static str, u32>,
    first_kernels: BTreeMap<String, u32>,
    flags: BTreeMap<String, u32>,
    layouts: BTreeMap<String, u32>,
    handles: HashSet<vk::CommandBuffer>,
    first_submits: u32,
    resubmitted_unchanged: u32,
    rerecorded: u32,
}

/// The colour input among the images of interest: the RGBA16F storage image at the extent of
/// a registered depth image (DLSS's inputs share the render extent, and the output and NGX's
/// scratch images don't have a depth image beside them). Returns the lowest-handle candidate
/// and how many there were.
fn identify_colour(interest: &BTreeMap<vk::Image, ViewDesc>) -> (Option<vk::Image>, usize) {
    const DEPTH: [vk::Format; 6] = [
        vk::Format::D16_UNORM,
        vk::Format::X8_D24_UNORM_PACK32,
        vk::Format::D32_SFLOAT,
        vk::Format::D16_UNORM_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
    ];
    let depth_extents: BTreeSet<(u32, u32)> = interest
        .values()
        .filter(|d| DEPTH.contains(&vk::Format::from_raw(d.format)))
        .map(|d| (d.width, d.height))
        .collect();
    let candidates: Vec<vk::Image> = interest
        .iter()
        .filter(|(_, d)| {
            d.format == vk::Format::R16G16B16A16_SFLOAT.as_raw()
                && vk::ImageUsageFlags::from_raw(d.usage).contains(vk::ImageUsageFlags::STORAGE)
                && depth_extents.contains(&(d.width, d.height))
        })
        .map(|(image, _)| *image)
        .collect();
    (candidates.first().copied(), candidates.len())
}

impl State {
    fn record_image(&mut self, image: vk::Image, info: &vk::ImageCreateInfo) {
        self.images.insert(
            image,
            ImageDesc { width: info.extent.width, height: info.extent.height, format: info.format, usage: info.usage },
        );
    }

    fn forget_image(&mut self, image: vk::Image) {
        self.images.remove(&image);
    }

    fn record_view(&mut self, view: vk::ImageView, info: &vk::ImageViewCreateInfo) {
        let mip = info.subresource_range.base_mip_level.min(31);
        let desc = self.images.get(&info.image).map(|image| ViewDesc {
            width: (image.width >> mip).max(1),
            height: (image.height >> mip).max(1),
            format: if info.format == vk::Format::UNDEFINED { image.format.as_raw() } else { info.format.as_raw() },
            usage: image.usage.as_raw(),
        });
        self.views.insert(view, desc);
        self.view_images.insert(view, info.image);
    }

    fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view);
        self.view_images.remove(&view);
    }

    fn register(&mut self, kind: &str, view: vk::ImageView, result: String) -> Vec<String> {
        let desc = self.views.get(&view).copied().flatten();
        self.frame.registrations += 1;
        self.frame.registered.insert(desc);
        let first = self.ever_registered.insert(view);
        let mut lines = Vec::new();
        if !first {
            return lines;
        }
        if self.registration_lines < MAX_EVENT_LINES {
            self.registration_lines += 1;
            let image = self.view_images.get(&view).map_or_else(|| "?".to_string(), |i| hex(*i));
            lines.push(format!("{kind} view={:#x} image={image} ({}) -> {result}", view.as_raw(), describe(desc)));
        }
        self.last_new_registration = Some(self.frames);
        if let (Some(desc), Some(image)) = (desc, self.view_images.get(&view).copied()) {
            self.interest.insert(image, desc);
            let (colour, candidates) = identify_colour(&self.interest);
            if colour != self.colour {
                self.colour = colour;
                if let Some(c) = colour {
                    lines.push(format!(
                        "colour input: image {} ({}){}, the registered RGBA16F storage image at the depth image's extent",
                        hex(c),
                        self.interest[&c],
                        if candidates > 1 { format!(" (first of {candidates} candidates)") } else { String::new() }
                    ));
                }
            }
        }
        lines
    }

    fn label(&self, image: vk::Image) -> String {
        if Some(image) == self.colour {
            format!("{}[COLOUR-IN]", hex(image))
        } else if let Some(desc) = self.interest.get(&image) {
            format!("{}[{}x{} {:?}]", hex(image), desc.width, desc.height, vk::Format::from_raw(desc.format))
        } else {
            hex(image)
        }
    }

    fn interesting(&self, image: vk::Image) -> bool {
        self.interest.contains_key(&image)
    }

    /// The views settled: some were registered and none new for [`STABLE_FRAMES`].
    fn settled(&self) -> bool {
        self.last_new_registration.is_some_and(|at| self.frames >= at + STABLE_FRAMES)
    }

    fn rec(&mut self, command_buffer: vk::CommandBuffer) -> &mut Recording {
        self.recordings.entry(command_buffer).or_default()
    }

    fn cmd_draw(&mut self, command_buffer: vk::CommandBuffer) {
        self.rec(command_buffer).draw();
    }

    fn cmd_dispatch(&mut self, command_buffer: vk::CommandBuffer) {
        self.rec(command_buffer).dispatch();
    }

    fn cmd_generated(&mut self, command_buffer: vk::CommandBuffer) {
        self.rec(command_buffer).generated();
    }

    fn cmd_clear_attachments(&mut self, command_buffer: vk::CommandBuffer) {
        self.rec(command_buffer).clear_attachments();
    }

    fn cmd_transfer(
        &mut self, command_buffer: vk::CommandBuffer, kind: &'static str,
        src: Option<(vk::Image, vk::ImageLayout)>, dst: Option<(vk::Image, vk::ImageLayout)>,
    ) {
        let interesting = src.is_some_and(|(i, _)| self.interesting(i)) || dst.is_some_and(|(i, _)| self.interesting(i));
        self.rec(command_buffer).transfer(kind, src, dst, interesting);
    }

    /// `views`: (view, layout, load op, depth) per attachment; views the probe never saw
    /// created are left out.
    fn cmd_begin_render(&mut self, command_buffer: vk::CommandBuffer, kind: &'static str, views: &[RenderView]) {
        let attachments: Vec<Attachment> = views
            .iter()
            .filter_map(|&(view, layout, load, depth)| {
                self.view_images.get(&view).map(|&image| Attachment { image, layout, load, depth })
            })
            .collect();
        let interesting = attachments.iter().any(|a| self.interesting(a.image));
        self.rec(command_buffer).begin_render(kind, attachments, interesting);
    }

    fn cmd_begin_render_pass(&mut self, command_buffer: vk::CommandBuffer, framebuffer: vk::Framebuffer, imageless: Option<Vec<vk::ImageView>>) {
        let views = imageless.or_else(|| self.framebuffers.get(&framebuffer).cloned()).unwrap_or_default();
        let views: Vec<RenderView> = views.into_iter().map(|v| (v, None, None, false)).collect();
        self.cmd_begin_render(command_buffer, "renderpass", &views);
    }

    fn cmd_end_render(&mut self, command_buffer: vk::CommandBuffer) {
        self.rec(command_buffer).end_render();
    }

    fn cmd_barriers(
        &mut self, command_buffer: vk::CommandBuffer, images: impl Iterator<Item = ImageBarrier>,
        memory: impl Iterator<Item = (u64, u64, u64, u64)>, buffers: u32,
    ) {
        for (src_stage, src_access, dst_stage, dst_access) in memory {
            self.rec(command_buffer).memory_barrier(src_stage, src_access, dst_stage, dst_access);
        }
        self.rec(command_buffer).buffer_barriers(buffers);
        for barrier in images {
            if Some(barrier.image) == self.colour {
                self.colour_last_barrier = Some((
                    command_buffer,
                    format!("{:?}->{:?} in cmdbuf {} (frame {})", barrier.old, barrier.new, hex(command_buffer), self.frames),
                ));
            }
            let interesting = self.interesting(barrier.image);
            self.rec(command_buffer).image_barrier(barrier, interesting);
        }
    }

    fn framebuffer(&mut self, framebuffer: vk::Framebuffer, views: Vec<vk::ImageView>) {
        self.framebuffers.insert(framebuffer, views);
    }

    /// At `vkEndCommandBuffer`: aggregate and maybe print a launch-bearing buffer's sequence.
    fn end(&mut self, command_buffer: vk::CommandBuffer) -> Vec<String> {
        let Some(rec) = self.recordings.get(&command_buffer) else { return Vec::new() };
        let launches = rec.total_launches();
        if launches == 0 || !self.settled() {
            return Vec::new();
        }
        let analysis = rec.analyse(self.colour);
        let flags = if rec.flags.is_empty() { "none".to_string() } else { format!("{:?}", rec.flags) };
        let first_kernel = analysis.first_launch.as_ref().map_or_else(|| "?".to_string(), |(_, k)| k.clone());
        let stats = &mut self.stats;
        stats.buffers += 1;
        *stats.classes.entry(analysis.class()).or_default() += 1;
        *stats.first_kernels.entry(first_kernel.clone()).or_default() += 1;
        *stats.flags.entry(flags.clone()).or_default() += 1;
        *stats.layouts.entry(analysis.layout_text()).or_default() += 1;
        let due = self.seq_printed < SEQ_FIRST || self.last_seq_frame.is_none_or(|at| self.frames >= at + SEQ_EVERY);
        if !due {
            return Vec::new();
        }
        self.seq_printed += 1;
        self.last_seq_frame = Some(self.frames);
        let rec = &self.recordings[&command_buffer];
        let cb = hex(command_buffer);
        let mut lines = vec![format!(
            "seq cb={cb} frame {} epoch {} flags={flags} launches={launches} entries={}{}: begin",
            self.frames,
            rec.epoch,
            rec.ops.len(),
            if rec.overflowed { " (entry cap reached, later commands only counted)" } else { "" }
        )];
        lines.extend(rec.lines(&|image| self.label(image)).into_iter().map(|line| format!("seq cb={cb} {line}")));
        let list = |items: &[String]| if items.is_empty() { "none".to_string() } else { items.join("; ") };
        let colour = self.colour.map_or_else(|| "not identified".to_string(), |c| self.label(c));
        let input_kernel = if first_kernel.contains("input") { "yes" } else { "no" };
        let elsewhere = rec.colour_elsewhere_at_first_launch.clone().unwrap_or_else(|| "none seen".into());
        lines.push(format!(
            "seq cb={cb} end: first launch {} {first_kernel} (input kernel: {input_kernel}); before it {} draws, {} dispatches, \
             {} renderings(other), {} transfers(other), {} mem-barriers with a write in src access",
            analysis.first_launch.as_ref().map_or_else(|| "-".to_string(), |(i, _)| format!("[{i}]")),
            analysis.draws_before,
            analysis.dispatches_before,
            analysis.renderings_before,
            analysis.transfers_before,
            analysis.write_memory_barriers_before,
        ));
        lines.push(format!(
            "seq cb={cb} colour input {colour} before the first launch: explicit writes: {}; barriers with write src access: {}; \
             transitions: {}; layout at first launch: {}; used as: {}; last barrier on it in another cmdbuf when the first launch was recorded: {elsewhere}",
            list(&analysis.explicit_writes),
            list(&analysis.barrier_writes),
            list(&analysis.transitions),
            analysis.layout_text(),
            analysis.usage_layout.as_ref().map_or_else(|| "-".to_string(), |(l, at)| format!("{l:?} at {at}")),
        ));
        lines
    }

    fn stats_line(&self) -> String {
        let s = &self.stats;
        let map = |m: &BTreeMap<String, u32>| m.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", ");
        let classes = s.classes.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", ");
        format!(
            "cmdbuf stats after frame {}: launch-bearing buffers ended={} distinct handles={} colour-input-before-first-launch={{{classes}}} \
             first kernel={{{}}} begin flags={{{}}} layout at first launch={{{}}} submits: first={} resubmitted-unchanged={} re-recorded={}",
            self.frames,
            s.buffers,
            s.handles.len(),
            map(&s.first_kernels),
            map(&s.flags),
            map(&s.layouts),
            s.first_submits,
            s.resubmitted_unchanged,
            s.rerecorded
        )
    }

    fn create_module(&mut self, data_size: usize, result: vk::Result) -> Vec<String> {
        if self.module_lines >= MAX_EVENT_LINES {
            return Vec::new();
        }
        self.module_lines += 1;
        vec![format!("vkCreateCuModuleNVX: {data_size} bytes -> {result:?}")]
    }

    fn create_function(&mut self, function: vk::CuFunctionNVX, name: String, result: vk::Result) -> Vec<String> {
        let mut lines = Vec::new();
        if self.function_lines < MAX_EVENT_LINES {
            self.function_lines += 1;
            lines.push(format!("vkCreateCuFunctionNVX: {name:?} -> {result:?} {:#x}", function.as_raw()));
        }
        if result == vk::Result::SUCCESS {
            self.functions.insert(function, name);
        }
        lines
    }

    fn launch(&mut self, command_buffer: vk::CommandBuffer, info: &vk::CuLaunchInfoNVX) {
        *self.launch_cmdbufs.entry(command_buffer).or_default() += 1;
        let function = self.functions.get(&info.function).cloned().unwrap_or_else(|| format!("fn{:#x}", info.function.as_raw()));
        let elsewhere = match &self.colour_last_barrier {
            Some((cb, _)) if *cb == command_buffer => None,
            other => other.as_ref().map(|(_, text)| text.clone()),
        };
        let rec = self.rec(command_buffer);
        if rec.launches == 0 {
            rec.colour_elsewhere_at_first_launch = elsewhere;
        }
        rec.launch(function.clone());
        let frame = &mut self.frame;
        frame.launches += 1;
        if !frame.cmdbufs.contains(&command_buffer) {
            frame.cmdbufs.push(command_buffer);
        }
        if frame.shapes.len() < MAX_LISTED {
            frame.shapes.insert(LaunchShape {
                function,
                grid: (info.grid_dim_x, info.grid_dim_y, info.grid_dim_z),
                block: (info.block_dim_x, info.block_dim_y, info.block_dim_z),
                shared_mem: info.shared_mem_bytes,
                params: info.param_count,
                extras: info.extra_count,
            });
        }
    }

    fn begin(&mut self, command_buffer: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) {
        self.launch_cmdbufs.remove(&command_buffer);
        self.next_epoch += 1;
        self.recordings.insert(command_buffer, Recording::new(flags, self.next_epoch));
    }

    fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.launch_cmdbufs.remove(cb);
            self.recordings.remove(cb);
            self.submitted_epochs.remove(cb);
        }
    }

    fn execute_secondary(&mut self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        let inherited: u32 = secondaries.iter().filter_map(|cb| self.launch_cmdbufs.get(cb)).sum();
        if inherited > 0 {
            *self.launch_cmdbufs.entry(primary).or_default() += inherited;
        }
        self.rec(primary).execute(u32::try_from(secondaries.len()).unwrap_or(u32::MAX), inherited);
    }

    fn submit(&mut self, queue: vk::Queue, command_buffers: impl Iterator<Item = vk::CommandBuffer>) -> Vec<String> {
        let index = self.frame.submits;
        self.frame.submits += 1;
        if self.launch_cmdbufs.is_empty() {
            return Vec::new();
        }
        let bearing: Vec<(vk::CommandBuffer, u32)> =
            command_buffers.filter_map(|cb| self.launch_cmdbufs.get(&cb).map(|n| (cb, *n))).collect();
        let carried: u32 = bearing.iter().map(|(_, n)| n).sum();
        if carried == 0 {
            return Vec::new();
        }
        for (cb, _) in &bearing {
            let epoch = self.recordings.get(cb).map_or(0, |r| r.epoch);
            let stats = &mut self.stats;
            stats.handles.insert(*cb);
            match self.submitted_epochs.insert(*cb, epoch) {
                None => stats.first_submits += 1,
                Some(previous) if previous == epoch => stats.resubmitted_unchanged += 1,
                Some(_) => stats.rerecorded += 1,
            }
        }
        self.frame.launch_submits.push((index, queue, carried));
        if self.first_launch_submit.is_none() && !self.first_launch_submit_done {
            self.first_launch_submit = Some((self.frames, index, queue));
            return vec![format!(
                "first launch-bearing submit: frame {} submit #{index} (0-based, counted from the previous present) on queue {:#x}, carrying {carried} launches",
                self.frames,
                queue.as_raw()
            )];
        }
        Vec::new()
    }

    fn present(&mut self, queue: vk::Queue) -> Vec<String> {
        let frame = std::mem::take(&mut self.frame);
        let n = self.frames;
        self.frames += 1;
        let mut lines = Vec::new();
        if let Some((at, index, submit_queue)) = self.first_launch_submit.take() {
            self.first_launch_submit_done = true;
            lines.push(format!(
                "first launch-bearing submit (frame {at}) was submit #{index} of {} before present on queue {:#x} ({} queue as the present)",
                frame.submits,
                queue.as_raw(),
                if submit_queue == queue { "same" } else { "a different" }
            ));
        }
        let extents: BTreeSet<(u32, u32)> = frame.registered.iter().flatten().map(|v| (v.width, v.height)).collect();
        let changed = extents != self.last_extents;
        self.last_extents = extents;
        let first_launches = frame.launches > 0 && !self.any_launch_frame_logged;
        if n.is_multiple_of(SUMMARY_EVERY) || changed || first_launches {
            if frame.launches > 0 {
                self.any_launch_frame_logged = true;
            }
            lines.push(summary_line(n, &frame, queue, self.ever_registered.len()));
        }
        if n.is_multiple_of(SEQ_EVERY) && self.stats.buffers > 0 {
            lines.push(self.stats_line());
        }
        lines
    }
}

fn summary_line(n: u64, frame: &Frame, queue: vk::Queue, ever: usize) -> String {
    let list = |items: Vec<String>| -> String {
        let more = items.len().saturating_sub(MAX_LISTED);
        let mut shown: Vec<String> = items.into_iter().take(MAX_LISTED).collect();
        if more > 0 {
            shown.push(format!("+{more} more"));
        }
        shown.join(", ")
    };
    let cmdbufs = list(frame.cmdbufs.iter().map(|cb| format!("{:#x}", cb.as_raw())).collect());
    let views = list(frame.registered.iter().map(|v| describe(*v)).collect());
    let submits = list(frame.launch_submits.iter().map(|(i, q, k)| format!("#{i}@{:#x}:{k}", q.as_raw())).collect());
    let shapes = list(frame.shapes.iter().map(ToString::to_string).collect());
    format!(
        "frame {n}: launches={} cmdbufs=[{cmdbufs}] views_registered={} ({views}) distinct_views_ever={ever} \
         submits={} launch_submits=[{submits}] present_queue={:#x} kernels=[{shapes}]",
        frame.launches,
        frame.registrations,
        frame.submits,
        queue.as_raw()
    )
}

// ---- Entry points for the hooks (each a no-op unless the probe is on). ----

pub(crate) fn on_create_image(image: vk::Image, info: &vk::ImageCreateInfo) {
    if enabled() {
        state().record_image(image, info);
    }
}

pub(crate) fn on_destroy_image(image: vk::Image) {
    if enabled() {
        state().forget_image(image);
    }
}

pub(crate) fn on_create_image_view(view: vk::ImageView, info: &vk::ImageViewCreateInfo) {
    state().record_view(view, info);
}

pub(crate) fn on_destroy_image_view(view: vk::ImageView) {
    state().forget_view(view);
}

pub(crate) fn on_view_handle(kind: &str, view: vk::ImageView, result: String) {
    let lines = state().register(kind, view, result);
    emit(lines);
}

pub(crate) fn on_create_cu_module(data_size: usize, result: vk::Result) {
    let lines = state().create_module(data_size, result);
    emit(lines);
}

pub(crate) fn on_create_cu_function(function: vk::CuFunctionNVX, info: &vk::CuFunctionCreateInfoNVX, result: vk::Result) {
    let name = if info.p_name.is_null() {
        "<null>".to_string()
    } else {
        // SAFETY: `pName` is a NUL-terminated string per `VkCuFunctionCreateInfoNVX`'s own
        // contract, valid for the duration of the application's call.
        let full = unsafe { CStr::from_ptr(info.p_name) }.to_string_lossy();
        full.chars().take(96).collect()
    };
    let lines = state().create_function(function, name, result);
    emit(lines);
}

pub(crate) fn on_destroy_cu_function(function: vk::CuFunctionNVX) {
    state().functions.remove(&function);
}

pub(crate) fn on_launch(command_buffer: vk::CommandBuffer, info: &vk::CuLaunchInfoNVX) {
    state().launch(command_buffer, info);
}

pub(crate) fn on_begin_command_buffer(command_buffer: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) {
    if enabled() {
        state().begin(command_buffer, flags);
    }
}

pub(crate) fn on_end_command_buffer(command_buffer: vk::CommandBuffer) {
    if enabled() {
        let lines = state().end(command_buffer);
        emit(lines);
    }
}

/// One render attachment as the hooks see it: view, layout (`None` for a render pass),
/// load op (`None` for a render pass), and whether it is the depth/stencil attachment.
pub(crate) type RenderView = (vk::ImageView, Option<vk::ImageLayout>, Option<vk::AttachmentLoadOp>, bool);

pub(crate) fn on_cmd_draw(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().cmd_draw(command_buffer);
    }
}

pub(crate) fn on_cmd_dispatch(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().cmd_dispatch(command_buffer);
    }
}

pub(crate) fn on_cmd_generated(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().cmd_generated(command_buffer);
    }
}

pub(crate) fn on_cmd_clear_attachments(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().cmd_clear_attachments(command_buffer);
    }
}

pub(crate) fn on_cmd_transfer(
    command_buffer: vk::CommandBuffer, kind: &'static str,
    src: Option<(vk::Image, vk::ImageLayout)>, dst: Option<(vk::Image, vk::ImageLayout)>,
) {
    if enabled() {
        state().cmd_transfer(command_buffer, kind, src, dst);
    }
}

pub(crate) fn on_cmd_begin_rendering(command_buffer: vk::CommandBuffer, views: &[RenderView]) {
    if enabled() {
        state().cmd_begin_render(command_buffer, "rendering", views);
    }
}

pub(crate) fn on_cmd_begin_render_pass(command_buffer: vk::CommandBuffer, framebuffer: vk::Framebuffer, imageless: Option<Vec<vk::ImageView>>) {
    if enabled() {
        state().cmd_begin_render_pass(command_buffer, framebuffer, imageless);
    }
}

pub(crate) fn on_cmd_end_render(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().cmd_end_render(command_buffer);
    }
}

/// Image barriers, memory barriers as `(src stage, src access, dst stage, dst access)`, and
/// the number of buffer barriers, all widened to sync2's 64-bit masks.
pub(crate) fn on_cmd_barriers(
    command_buffer: vk::CommandBuffer, images: impl Iterator<Item = ImageBarrier>,
    memory: impl Iterator<Item = (u64, u64, u64, u64)>, buffers: u32,
) {
    if enabled() {
        state().cmd_barriers(command_buffer, images, memory, buffers);
    }
}

pub(crate) fn on_create_framebuffer(framebuffer: vk::Framebuffer, views: Vec<vk::ImageView>) {
    if enabled() {
        state().framebuffer(framebuffer, views);
    }
}

pub(crate) fn on_destroy_framebuffer(framebuffer: vk::Framebuffer) {
    if enabled() {
        state().framebuffers.remove(&framebuffer);
    }
}

pub(crate) fn on_free_command_buffers(command_buffers: &[vk::CommandBuffer]) {
    if enabled() {
        state().free(command_buffers);
    }
}

pub(crate) fn on_execute_commands(primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
    if enabled() {
        state().execute_secondary(primary, secondaries);
    }
}

pub(crate) fn on_submit(queue: vk::Queue, command_buffers: impl Iterator<Item = vk::CommandBuffer>) {
    if enabled() {
        let lines = state().submit(queue, command_buffers);
        emit(lines);
    }
}

pub(crate) fn on_present(queue: vk::Queue) {
    if enabled() {
        let lines = state().present(queue);
        emit(lines);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb(raw: u64) -> vk::CommandBuffer { vk::CommandBuffer::from_raw(raw) }
    fn queue(raw: u64) -> vk::Queue { vk::Queue::from_raw(raw) }

    fn image_info(width: u32, height: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> vk::ImageCreateInfo {
        vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D { width, height, depth: 1 },
            format,
            usage,
            ..Default::default()
        }
    }

    fn view_info(image: vk::Image, format: vk::Format, mip: u32) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo {
            image,
            format,
            subresource_range: vk::ImageSubresourceRange { base_mip_level: mip, level_count: 1, layer_count: 1, ..Default::default() },
            ..Default::default()
        }
    }

    fn launch_info(function: u64, params: usize) -> vk::CuLaunchInfoNVX {
        vk::CuLaunchInfoNVX {
            function: vk::CuFunctionNVX::from_raw(function),
            grid_dim_x: 80,
            grid_dim_y: 45,
            grid_dim_z: 1,
            block_dim_x: 16,
            block_dim_y: 16,
            block_dim_z: 1,
            param_count: params,
            // Deliberately bogus: the probe must never dereference these.
            p_params: 0x10 as *const *const std::ffi::c_void,
            ..Default::default()
        }
    }

    #[test]
    fn the_probe_adds_no_commands_when_off() {
        assert!(probe_commands(false).is_empty());
        assert!(probe_commands(true).contains(&VulkanCommand::CmdCuLaunchKernelNvx));
        assert!(probe_commands(true).contains(&VulkanCommand::GetImageViewHandleNvx));
    }

    #[test]
    fn views_are_described_from_their_image_and_registrations_counted() {
        let mut s = State::default();
        let render = vk::Image::from_raw(0x100);
        let output = vk::Image::from_raw(0x200);
        s.record_image(render, &image_info(1707, 960, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT));
        s.record_image(output, &image_info(2560, 1440, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE));
        let (rv, ov, mipv, unknown) = (vk::ImageView::from_raw(0x10), vk::ImageView::from_raw(0x20), vk::ImageView::from_raw(0x30), vk::ImageView::from_raw(0x40));
        s.record_view(rv, &view_info(render, vk::Format::UNDEFINED, 0));
        s.record_view(ov, &view_info(output, vk::Format::R16G16B16A16_SFLOAT, 0));
        s.record_view(mipv, &view_info(output, vk::Format::R16G16B16A16_SFLOAT, 1));
        assert_eq!(s.views[&rv].unwrap().width, 1707);
        assert_eq!(s.views[&rv].unwrap().format, vk::Format::R16G16B16A16_SFLOAT.as_raw(), "an UNDEFINED view format falls back to the image's");
        assert_eq!((s.views[&mipv].unwrap().width, s.views[&mipv].unwrap().height), (1280, 720));
        let lines = s.register("vkGetImageViewHandleNVX", rv, "0x1".into());
        assert!(lines[0].contains("1707x960"), "{lines:?}");
        assert!(s.register("vkGetImageViewHandleNVX", rv, "0x1".into()).is_empty(), "a view is announced once");
        s.register("vkGetImageViewHandleNVX", ov, "0x2".into());
        s.register("vkGetImageViewHandleNVX", unknown, "0x3".into());
        assert_eq!(s.frame.registrations, 4);
        let lines = s.present(queue(0x9));
        let summary = lines.last().unwrap();
        assert!(summary.starts_with("frame 0: launches=0"), "{summary}");
        assert!(summary.contains("views_registered=4"), "{summary}");
        assert!(summary.contains("1707x960") && summary.contains("2560x1440") && summary.contains("unknown-view"), "{summary}");
        s.forget_view(rv);
        s.forget_image(render);
        assert!(!s.views.contains_key(&rv) && !s.images.contains_key(&render));
    }

    #[test]
    fn launches_are_attributed_to_command_buffers_and_submits() {
        let mut s = State::default();
        s.create_function(vk::CuFunctionNVX::from_raw(0x77), "dlss_sr_kernel".into(), vk::Result::SUCCESS);
        s.launch(cb(0xa), &launch_info(0x77, 5));
        s.launch(cb(0xa), &launch_info(0x77, 5));
        s.launch(cb(0xb), &launch_info(0x88, 3));
        // Submit #0 carries nothing, #1 carries cb a, #2 carries cb b inside a primary.
        assert!(s.submit(queue(0x1), [cb(0xc)].into_iter()).is_empty());
        let first = s.submit(queue(0x2), [cb(0xc), cb(0xa)].into_iter());
        assert!(first[0].contains("submit #1") && first[0].contains("carrying 2 launches"), "{first:?}");
        s.execute_secondary(cb(0xd), &[cb(0xb)]);
        assert!(s.submit(queue(0x2), [cb(0xd)].into_iter()).is_empty(), "only the first launch-bearing submit is announced");
        let lines = s.present(queue(0x2));
        assert!(lines[0].contains("submit #1 of 3") && lines[0].contains("same queue"), "{lines:?}");
        let summary = &lines[1];
        assert!(summary.contains("launches=3") && summary.contains("cmdbufs=[0xa, 0xb]"), "{summary}");
        assert!(summary.contains("launch_submits=[#1@0x2:2, #2@0x2:1]"), "{summary}");
        assert!(summary.contains("dlss_sr_kernel grid=80x45x1 block=16x16x1 smem=0 params=5 extras=0"), "{summary}");
        assert!(summary.contains("fn0x88"), "an unknown function is named by handle: {summary}");
        // A re-begun buffer no longer carries launches.
        s.begin(cb(0xa), vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        s.free(&[cb(0xd)]);
        assert!(s.submit(queue(0x2), [cb(0xa), cb(0xd)].into_iter()).is_empty());
        assert!(s.frame.launch_submits.is_empty());
    }

    #[test]
    fn summaries_are_sampled_and_follow_extent_changes() {
        let mut s = State::default();
        let image = vk::Image::from_raw(0x100);
        s.record_image(image, &image_info(1920, 1080, vk::Format::B8G8R8A8_UNORM, vk::ImageUsageFlags::SAMPLED));
        let view = vk::ImageView::from_raw(0x10);
        s.record_view(view, &view_info(image, vk::Format::B8G8R8A8_UNORM, 0));
        let q = queue(0x1);
        assert_eq!(s.present(q).len(), 1, "frame 0 is sampled");
        for _ in 1..10 {
            assert!(s.present(q).is_empty(), "quiet frames are not logged");
        }
        s.register("vkGetImageViewHandleNVX", view, "0x1".into());
        assert_eq!(s.present(q).len(), 1, "a new extent set is logged");
        assert_eq!(s.present(q).len(), 1, "and so is its disappearance");
        s.launch(cb(0x5), &launch_info(0x1, 1));
        assert_eq!(s.present(q).len(), 1, "the first frame with launches is logged");
        s.launch(cb(0x5), &launch_info(0x1, 1));
        assert!(s.present(q).is_empty(), "later launch frames only on the sampling cadence");
        while !s.frames.is_multiple_of(SUMMARY_EVERY) {
            s.present(q);
        }
        assert_eq!(s.present(q).len(), 1, "every 60th frame is logged");
    }

    /// GTA's DLSS inputs: colour, depth, MV at 1707x960, output and scratch elsewhere.
    fn register_dlss_inputs(s: &mut State) -> (vk::Image, vk::Image) {
        let rgba = vk::Format::R16G16B16A16_SFLOAT;
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let images = [
            (0x100, 1707, 960, rgba, storage | vk::ImageUsageFlags::COLOR_ATTACHMENT),
            (0x200, 1707, 960, vk::Format::D32_SFLOAT_S8_UINT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            (0x300, 1707, 960, vk::Format::R16G16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT),
            (0x400, 2560, 1440, rgba, storage),
            (0x500, 640, 384, rgba, storage),
        ];
        // Scratch and output first, so the colour input is not simply the first registered.
        for (raw, w, h, format, usage) in images.iter().rev() {
            let image = vk::Image::from_raw(*raw);
            s.record_image(image, &image_info(*w, *h, *format, *usage));
            let view = vk::ImageView::from_raw(raw + 1);
            s.record_view(view, &view_info(image, *format, 0));
            s.register("vkGetImageViewHandle64NVX", view, "handle 0x1".into());
        }
        (vk::Image::from_raw(0x100), vk::Image::from_raw(0x400))
    }

    #[test]
    fn the_colour_input_is_the_rgba16f_storage_image_at_the_depth_extent() {
        let mut s = State::default();
        let image = vk::Image::from_raw(0x100);
        s.record_image(image, &image_info(1707, 960, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE));
        s.record_view(vk::ImageView::from_raw(0x101), &view_info(image, vk::Format::R16G16B16A16_SFLOAT, 0));
        s.register("vkGetImageViewHandleNVX", vk::ImageView::from_raw(0x101), "0x1".into());
        assert_eq!(s.colour, None, "without a depth image beside it there is no render extent");
        let mut s = State::default();
        let (colour, _) = register_dlss_inputs(&mut s);
        assert_eq!(s.colour, Some(colour));
        assert!(s.label(colour).ends_with("[COLOUR-IN]"));
        assert!(s.label(vk::Image::from_raw(0x200)).contains("1707x960 D32_SFLOAT_S8_UINT"));
        assert_eq!(s.label(vk::Image::from_raw(0x999)), "0x999");
    }

    #[test]
    fn launch_bearing_sequences_are_printed_after_the_views_settle_and_then_sampled() {
        let mut s = State::default();
        let (colour, output) = register_dlss_inputs(&mut s);
        s.create_function(vk::CuFunctionNVX::from_raw(0x77), "cuda_engine_input_kernel_rel_hdr".into(), vk::Result::SUCCESS);
        s.create_function(vk::CuFunctionNVX::from_raw(0x78), "dltss_pwin_enc0_layer".into(), vk::Result::SUCCESS);
        let q = queue(0x9);
        let game = cb(0xa);
        let record = |s: &mut State| {
            s.begin(game, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            s.cmd_draw(game);
            s.cmd_dispatch(game);
            s.cmd_barriers(
                game,
                [ImageBarrier {
                    image: colour,
                    old: vk::ImageLayout::GENERAL,
                    new: vk::ImageLayout::GENERAL,
                    src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
                    src_access: vk::AccessFlags2::SHADER_WRITE.as_raw(),
                    dst_stage: vk::PipelineStageFlags2::ALL_COMMANDS.as_raw(),
                    dst_access: vk::AccessFlags2::SHADER_READ.as_raw(),
                }]
                .into_iter(),
                std::iter::empty(),
                0,
            );
            s.launch(game, &launch_info(0x77, 4));
            s.launch(game, &launch_info(0x78, 4));
            s.cmd_transfer(game, "copy", Some((output, vk::ImageLayout::GENERAL)), Some((vk::Image::from_raw(0x777), vk::ImageLayout::GENERAL)));
            s.end(game)
        };
        assert!(record(&mut s).is_empty(), "not before the views settle");
        for _ in 0..STABLE_FRAMES {
            s.present(q);
        }
        let lines = record(&mut s);
        assert!(lines[0].starts_with("seq cb=0xa frame 30 epoch 2 flags=ONE_TIME_SUBMIT launches=2 entries=5"), "{lines:?}");
        assert_eq!(lines[1], "seq cb=0xa [0] ... 1 draws, 1 dispatches");
        assert!(lines[2].starts_with("seq cb=0xa [1] barrier 0x100[COLOUR-IN] GENERAL->GENERAL src COMPUTE_SHADER:SHADER_WRITE"), "{}", lines[2]);
        assert_eq!(lines[3], "seq cb=0xa [2] LAUNCH #1 cuda_engine_input_kernel_rel_hdr");
        assert!(lines[5].contains("copy src=0x400[2560x1440 R16G16B16A16_SFLOAT] (GENERAL) dst=0x777 (GENERAL)"), "{}", lines[5]);
        assert!(lines[6].contains("first launch [2] cuda_engine_input_kernel_rel_hdr (input kernel: yes); before it 1 draws, 1 dispatches"), "{}", lines[6]);
        assert!(lines[7].contains("explicit writes: none; barriers with write src access: [1] src access SHADER_WRITE; transitions: none; layout at first launch: GENERAL (barrier [1])"), "{}", lines[7]);
        // Resubmission bookkeeping: submitted twice unchanged, then re-recorded.
        s.submit(q, [game].into_iter());
        s.submit(q, [game].into_iter());
        // Two more full sequences, then quiet until SEQ_EVERY frames later.
        assert!(!record(&mut s).is_empty());
        s.submit(q, [game].into_iter());
        assert!(!record(&mut s).is_empty());
        assert!(record(&mut s).is_empty(), "the fourth is not printed");
        while !s.frames.is_multiple_of(SEQ_EVERY) {
            s.present(q);
        }
        let lines = s.present(q);
        let stats = lines.last().unwrap();
        assert!(stats.starts_with("cmdbuf stats after frame 301: launch-bearing buffers ended=4 distinct handles=1"), "{stats}");
        assert!(stats.contains("colour-input-before-first-launch={barrier-write: 4}"), "{stats}");
        assert!(stats.contains("submits: first=1 resubmitted-unchanged=1 re-recorded=1"), "{stats}");
        assert!(record(&mut s).is_empty(), "the last print was at frame 30");
        while s.frames < STABLE_FRAMES + SEQ_EVERY {
            s.present(q);
        }
        assert!(!record(&mut s).is_empty(), "printed again once SEQ_EVERY frames passed");
    }

    #[test]
    fn event_lines_are_capped() {
        let mut s = State::default();
        for i in 0..(MAX_EVENT_LINES + 10) {
            s.create_module(16, vk::Result::SUCCESS);
            s.register("vkGetImageViewHandleNVX", vk::ImageView::from_raw(u64::from(i) + 1), "0".into());
        }
        assert_eq!(s.module_lines, MAX_EVENT_LINES);
        assert_eq!(s.registration_lines, MAX_EVENT_LINES);
        assert!(s.create_module(16, vk::Result::SUCCESS).is_empty());
        assert_eq!(s.ever_registered.len(), MAX_EVENT_LINES as usize + 10);
    }
}
