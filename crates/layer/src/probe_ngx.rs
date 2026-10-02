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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ffi::CStr;
use std::sync::{LazyLock, Mutex, MutexGuard};

use ash::vk;
use ash::vk::Handle;
use vulkan_layer::LayerVulkanCommand as VulkanCommand;

/// The variable that turns the probe on. Read through `neural_forge_protocol::env`.
pub(crate) const ENV: &str = "NEURAL_FORGE_PROBE_NGX";

/// Every `SUMMARY_EVERY`th frame gets a summary line even when nothing changed.
const SUMMARY_EVERY: u64 = 60;
/// Caps on per-event lines, so a game that creates thousands of views or kernels cannot
/// flood the log.
const MAX_EVENT_LINES: u32 = 64;
/// Caps on how many distinct entries one frame's summary lists.
const MAX_LISTED: usize = 12;

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
    }

    fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view);
    }

    fn register(&mut self, kind: &str, view: vk::ImageView, result: String) -> Vec<String> {
        let desc = self.views.get(&view).copied().flatten();
        self.frame.registrations += 1;
        self.frame.registered.insert(desc);
        let first = self.ever_registered.insert(view);
        if first && self.registration_lines < MAX_EVENT_LINES {
            self.registration_lines += 1;
            return vec![format!("{kind} view={:#x} ({}) -> {result}", view.as_raw(), describe(desc))];
        }
        Vec::new()
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
        let frame = &mut self.frame;
        frame.launches += 1;
        if !frame.cmdbufs.contains(&command_buffer) {
            frame.cmdbufs.push(command_buffer);
        }
        if frame.shapes.len() < MAX_LISTED {
            let function = self.functions.get(&info.function).cloned().unwrap_or_else(|| format!("fn{:#x}", info.function.as_raw()));
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

    fn begin(&mut self, command_buffer: vk::CommandBuffer) {
        self.launch_cmdbufs.remove(&command_buffer);
    }

    fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.launch_cmdbufs.remove(cb);
        }
    }

    fn execute_secondary(&mut self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        let inherited: u32 = secondaries.iter().filter_map(|cb| self.launch_cmdbufs.get(cb)).sum();
        if inherited > 0 {
            *self.launch_cmdbufs.entry(primary).or_default() += inherited;
        }
    }

    fn submit(&mut self, queue: vk::Queue, command_buffers: impl Iterator<Item = vk::CommandBuffer>) -> Vec<String> {
        let index = self.frame.submits;
        self.frame.submits += 1;
        if self.launch_cmdbufs.is_empty() {
            return Vec::new();
        }
        let carried: u32 = command_buffers.filter_map(|cb| self.launch_cmdbufs.get(&cb)).sum();
        if carried == 0 {
            return Vec::new();
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

pub(crate) fn on_begin_command_buffer(command_buffer: vk::CommandBuffer) {
    if enabled() {
        state().begin(command_buffer);
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
        s.begin(cb(0xa));
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
