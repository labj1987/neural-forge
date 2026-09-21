//! Per-device state and the [`DeviceHooks`] implementation: `vkCreateSwapchainKHR`/
//! `vkDestroySwapchainKHR` track swapchains, `vkGetDeviceQueue`/`vkGetDeviceQueue2`
//! learn which queue family a queue belongs to, `vkQueuePresentKHR` is where the real
//! capture/transport/write-back round trip (`crate::capture::run`) happens now.

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use vulkan_layer::{DeviceHooks, DeviceInfo, LayerResult, LayerVulkanCommand as VulkanCommand};

use crate::capture;
use crate::shm::ShmClient;
use crate::swapchain::{self, SwapchainState};

/// The one swapchain (across every device in this process) allowed to drive the
/// shared-memory channel. A process can present more than one swapchain -- the game
/// window and the Steam overlay, or, mid-resize, the old and new windows at once.
/// Routing all of them through one channel would make the helper rebuild its feature on
/// every size switch, and could hand one swapchain another's answer; the largest by
/// area is assumed to be the game, and the rest present untouched.
struct Primary {
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    area: u64,
}

static PRIMARY: Mutex<Option<Primary>> = Mutex::new(None);

fn claim_primary(device: vk::Device, swapchain: vk::SwapchainKHR, width: u32, height: u32) -> bool {
    let mut guard = PRIMARY.lock().unwrap();
    let area = u64::from(width) * u64::from(height);
    match &*guard {
        Some(p) if p.device == device && p.swapchain == swapchain => true,
        Some(p) if area <= p.area => false,
        _ => {
            *guard = Some(Primary { device, swapchain, area });
            true
        }
    }
}

fn release_primary(device: vk::Device, swapchain: vk::SwapchainKHR) {
    let mut guard = PRIMARY.lock().unwrap();
    if matches!(&*guard, Some(p) if p.device == device && p.swapchain == swapchain) {
        *guard = None;
    }
}

/// Resolves one function pointer through the next layer/driver's `vkGetDeviceProcAddr`,
/// or `None` if it isn't there. That's not a failure worth panicking over: a device
/// that never enabled `VK_KHR_swapchain` (a compute-only device, or any device an app
/// simply never presents from) legitimately has no `vkCreateSwapchainKHR` to resolve,
/// and such a device will also never have an app call it -- so a missing pointer here
/// just means this device's hooks quietly do nothing, not that anything is wrong. This
/// was caught by `examples/smoke.rs` creating a device with no extensions enabled at
/// all: the first version of this function panicked on exactly that, which would have
/// crashed every plain compute app the layer got loaded into.
///
/// Transmuting the result to `F` is sound exactly as far as the caller names the right
/// `PFN_vk*` type for `name` -- the same contract the equivalent C cast upstream's own
/// `next_dpa` calls carry.
///
/// # Safety
/// `get_proc` must be a valid `vkGetDeviceProcAddr` for `device`, and `F` must be the
/// PFN type matching `name`.
unsafe fn resolve<F: Copy>(get_proc: vk::PFN_vkGetDeviceProcAddr, device: vk::Device, name: &CStr) -> Option<F> {
    let p = unsafe { get_proc(device, name.as_ptr()) }?;
    // SAFETY: forwarded from the caller's own safety contract.
    Some(unsafe { std::mem::transmute_copy::<_, F>(&p) })
}

pub struct NeuralForgeDeviceInfo {
    _loader_data: crate::loader_data::Registration,
    device: Arc<ash::Device>,
    /// `None` only in the hypothetical case `create_device_info`'s own doc comment
    /// notes (a device created against an instance from before this layer loaded,
    /// which never happens for an implicit layer) -- capture is simply skipped
    /// (present passes through unmodified) whenever it is, rather than panicking.
    instance: Option<Arc<ash::Instance>>,
    surface_caps: Option<vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>,
    physical_device: vk::PhysicalDevice,
    next_create_swapchain_khr: Option<vk::PFN_vkCreateSwapchainKHR>,
    next_destroy_swapchain_khr: Option<vk::PFN_vkDestroySwapchainKHR>,
    next_queue_present_khr: Option<vk::PFN_vkQueuePresentKHR>,
    next_get_swapchain_images_khr: Option<vk::PFN_vkGetSwapchainImagesKHR>,
    next_get_device_queue: Option<vk::PFN_vkGetDeviceQueue>,
    next_get_device_queue2: Option<vk::PFN_vkGetDeviceQueue2>,
    state: Arc<Mutex<State>>,
    tracker: Mutex<TapTracker>,
}

#[derive(Default)]
struct State {
    swapchains: HashMap<vk::SwapchainKHR, SwapchainState>,
    shm: ShmClient,
    /// Which queue family a `VkQueue` handle belongs to -- learned by observing the
    /// app's own `vkGetDeviceQueue`/`vkGetDeviceQueue2` calls (see those hooks below),
    /// since Vulkan has no query that answers this for a handle after the fact. Needed
    /// to build a command pool for whatever queue `queue_present_khr` hands us.
    queue_families: HashMap<vk::Queue, u32>,
    capture: Option<capture::CaptureResources>,
    /// The Phase 2 non-blocking capture pipeline (`docs/ASYNC_CAPTURE_DESIGN.md`) `capture::run`
    /// uses for its own hot-path captures. Kept separate from `capture` above (which
    /// stays the single synchronous resource `run_sync` and `run`'s CPU-only
    /// write-back fallback still use) rather than sharing one resource type across
    /// both purposes -- a slot mid-flight for one would otherwise have to be safe to
    /// borrow for the other's completely different, fully-synchronous contract.
    capture_pipeline: Option<capture::CapturePipeline>,
    /// The Phase 3 zero-copy capture path (`docs/EXTERNAL_MEMORY_HOST_DESIGN.md`) --
    /// mutually exclusive with `capture_pipeline` above, never both active for the
    /// same device. One per protocol v3 wire slot (`docs/PROTOCOL_V3_DESIGN.md`), each
    /// importing that slot's own disjoint proxy region -- no write-write hazard
    /// between them (unlike two of the same slot, which `DirectCapture`'s own doc
    /// comment still explains). `capture::run` decides `DirectCapture` vs.
    /// `CapturePipeline` once per call, cheaply, from `external_memory_host` below
    /// plus a live alignment query -- so this stays populated (or not) correctly even
    /// if that decision's answer could somehow change mid-process, though in practice
    /// it never does.
    direct_capture: [Option<capture::DirectCapture>; 2],
    /// Whether `NeuralForgeInstanceHooks::create_device` (crate::lib) got
    /// `VK_EXT_external_memory_host` added to this device's own creation -- set once,
    /// at construction, from a side channel only that hook can populate (see its own
    /// doc comment for why `create_info` here would always say "no" regardless of
    /// what was actually enabled). `false` is the common case (no ordinary game
    /// requests this extension on its own); `capture::run` treats it as "keep using
    /// the staging-buffer path", not an error.
    external_memory_host: bool,
    gpu_compose: Option<crate::composition::gpu::GpuCompose>,
    /// Reused across frames by `capture::run` for its own pre-edit frame snapshot,
    /// instead of a fresh `frame_bytes`-sized heap allocation every single present
    /// call -- see that function's own doc comment on why the snapshot exists at all.
    /// At 4K RGBA8 that's a ~31.6MiB allocation avoided every frame; measured on
    /// `lordnikon` (2026-09-10) at ~78ms per fresh allocation+copy, a real, if not
    /// fully explained (a `perf stat` on the same machine at the same time showed the
    /// process 97% backend-bound with an IPC of 0.1 -- a severe memory-subsystem
    /// stall this allocation likely aggravates without being its root cause), cost.
    original_scratch: Vec<u8>,
    /// The pipelined redesign's own persistent state -- see `capture::run`'s own doc
    /// comment for why a round trip's original frame has to outlive the present call
    /// that sent it, across however many present calls it takes the helper to answer.
    /// One per protocol v3 wire slot: each slot's in-flight request has its own,
    /// completely independent original frame and dims.
    inflight: [capture::Inflight; 2],
    /// A single disabled-state evaluation reserves the helper's images and NGX
    /// feature before the game's working set fills available VRAM (see
    /// `capture::run`'s own doc comment on why) -- one flag for the whole process,
    /// not per-slot: it only ever uses wire slot 0.
    bootstrap_complete: bool,
    /// Reused across frames the same way `original_scratch` is, for the answer bytes
    /// `capture::run` reads back once a round trip resolves.
    answer_scratch: Vec<u8>,
    /// `working_scale`'s scaled proxy bytes -- reused across frames the same way
    /// `original_scratch` is, but at the (usually much smaller) model resolution
    /// rather than the swapchain's own. Empty and unused whenever `working_scale`
    /// is left at its default `1.0`.
    model_scratch: Vec<u8>,
    /// The original frame paired with whichever wire slot most recently produced the
    /// answer currently held in `last_answer` -- not per-slot, since only one answer
    /// is ever the "currently presented" one at a time (see `capture::Inflight`'s own
    /// doc comment for why this moved out of the per-slot array).
    raw_answer_base: Vec<u8>,
    /// Monotonic identity for `raw_answer_base`/`last_answer` together -- GPU
    /// composition uploads only when this changes.
    raw_answer_generation: u64,
    last_answer: Vec<u8>,
    /// `last_answer`'s own resolution -- see `capture::Inflight::proxy_dims`'s own
    /// doc comment. Meaningless while `last_answer` is empty; always set together
    /// with it otherwise.
    last_answer_dims: (u32, u32),
    hotkey: crate::hotkey::Poller,
    /// Per-swapchain-image relay semaphores -- see `queue_present_khr`'s own comment on
    /// why the application's present wait semaphores are relayed through the layer's
    /// own queue before any capture/compose work is submitted.
    relay_semaphores: crate::present_sync::PresentSemaphores,
}

/// Render-tap bookkeeping, deliberately kept out of the per-device `Mutex<State>`.
///
/// The command-recording hooks (`vkCmdPipelineBarrier*`, `vkCmdCopyImage`,
/// `vkCmdBlitImage`) fire from whichever thread the game records on, while the present
/// hook holds `State` across GPU fence waits; sharing one lock serialized the game's
/// recording against present. This has its own short-lived lock instead, and no hook
/// ever holds it across anything but map operations. Lock order, where both are taken:
/// `State` first, then this.
///
/// Layouts follow *submission* order, not recording order. A layout observed while a
/// command buffer is being recorded says nothing about the state of the image when
/// that buffer executes (buffers can be recorded on several threads, in any order, and
/// some are never submitted), so recorded transitions are parked per command buffer in
/// `pending` and only applied to `tapped_source_layouts` when `vkQueueSubmit*` actually
/// submits that buffer. Not tracked, as before: layout changes made implicitly by a
/// render pass's attachment descriptions.
#[derive(Default)]
struct TapTracker {
    /// Every image of every live swapchain, so a copy/blit destination can be
    /// recognised without touching `State`.
    swapchain_images: HashSet<vk::Image>,
    /// Passive transfer observations for swapchains which could not be admitted at
    /// creation.  This is diagnostic-only: it never changes a game command buffer.
    observed_swapchain_writes: HashSet<vk::Image>,
    /// Keyed by the *source* image (the game's own internal render target the render
    /// tap reads from), never by a swapchain image -- see `prune_orphaned_tap_source`'s
    /// doc comment for why every insertion here has to be paired with eventual removal,
    /// not left to grow for the process's whole lifetime. Holds only layouts already
    /// committed by a queue submission.
    tapped_source_layouts: HashMap<vk::Image, vk::ImageLayout>,
    tap_sources_by_destination: HashMap<vk::Image, vk::Image>,
    /// Layouts recorded into each command buffer for tap sources, in recording order,
    /// awaiting submission.
    pending: HashMap<vk::CommandBuffer, Vec<(vk::Image, vk::ImageLayout)>>,
}

impl TapTracker {
    fn is_source(&self, image: vk::Image) -> bool {
        self.tap_sources_by_destination.values().any(|&src| src == image)
    }

    /// The tap source and its last *submitted* layout for a swapchain image, if any.
    fn tap_for(&self, destination: vk::Image) -> Option<(vk::Image, vk::ImageLayout)> {
        let source = *self.tap_sources_by_destination.get(&destination)?;
        self.tapped_source_layouts.get(&source).map(|layout| (source, *layout))
    }

    /// Returns `true` the first time a given swapchain image is seen as a destination.
    fn observe_write(
        &mut self, command_buffer: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, dst: vk::Image,
    ) -> Option<bool> {
        if !self.swapchain_images.contains(&dst) {
            return None;
        }
        let first = self.observed_swapchain_writes.insert(dst);
        // A destination normally keeps the same source for its whole life (the
        // game doesn't usually re-target its own blit/copy calls frame to frame),
        // but if it ever does, the old source needs the same orphan check
        // `destroy_swapchain_khr` already does -- otherwise a source that's no
        // longer referenced by anything would sit in `tapped_source_layouts`
        // forever, the same unbounded-growth/stale-handle hazard
        // `prune_orphaned_tap_source`'s own doc comment explains.
        if let Some(previous_source) = self.tap_sources_by_destination.insert(dst, src) {
            if previous_source != src {
                prune_orphaned_tap_source(self, previous_source);
            }
        }
        // The copy/blit itself states the source's layout at this point in the buffer.
        self.pending.entry(command_buffer).or_default().push((src, src_layout));
        Some(first)
    }

    fn record_barrier(&mut self, command_buffer: vk::CommandBuffer, image: vk::Image, new_layout: vk::ImageLayout) {
        if self.is_source(image) {
            self.pending.entry(command_buffer).or_default().push((image, new_layout));
        }
    }

    fn begin_recording(&mut self, command_buffer: vk::CommandBuffer) {
        self.pending.remove(&command_buffer);
    }

    fn free(&mut self, command_buffers: &[vk::CommandBuffer]) {
        for cb in command_buffers {
            self.pending.remove(cb);
        }
    }

    fn execute_secondary(&mut self, primary: vk::CommandBuffer, secondaries: &[vk::CommandBuffer]) {
        if self.pending.is_empty() {
            return;
        }
        let inherited: Vec<_> = secondaries.iter().filter_map(|cb| self.pending.get(cb)).flatten().copied().collect();
        if !inherited.is_empty() {
            self.pending.entry(primary).or_default().extend(inherited);
        }
    }

    /// Applies each submitted command buffer's recorded layouts, in submission order.
    /// The recorded list is kept: a buffer can legally be submitted again unchanged.
    fn commit_submit(&mut self, command_buffers: impl Iterator<Item = vk::CommandBuffer>) {
        if self.pending.is_empty() {
            return;
        }
        for cb in command_buffers {
            let Some(recorded) = self.pending.get(&cb) else { continue };
            for &(image, layout) in recorded {
                if self.tap_sources_by_destination.values().any(|&src| src == image) {
                    self.tapped_source_layouts.insert(image, layout);
                }
            }
        }
    }
}

/// Removes `source`'s `tapped_source_layouts` entry once nothing in
/// `tap_sources_by_destination` still points at it.
///
/// Real bug, found 2026-09-16 after a live GTA session ran into a GPU-level hang
/// (`Xid 109 CTX_SWITCH_TIMEOUT`) during genuinely long real play, never reproduced by
/// any short test: `tapped_source_layouts` was insert-only -- `observe_swapchain_write`
/// added an entry for every distinct source image the render tap ever observed, for
/// the whole life of the process, and nothing ever removed one. `destroy_swapchain_khr`
/// already cleaned up `tap_sources_by_destination` (keyed by the *destination*
/// swapchain image, which really is bounded by swapchain lifetime), but never touched
/// `tapped_source_layouts` at all.
///
/// The real danger isn't just unbounded growth: Vulkan explicitly allows a destroyed
/// image's handle value to be reused for a later, completely unrelated image. Once the
/// game frees one of its own internal render targets and a new allocation happens to
/// reuse that same handle, `cmd_pipeline_barrier`/`cmd_pipeline_barrier2` (which check
/// every barrier's image against this map, for every image in the whole process, not
/// just ones this layer cares about) would silently start updating *our* stale entry
/// to track the new, unrelated resource's layout -- and if `tap_sources_by_destination`
/// still pointed some live swapchain's destination at that same stale handle, the
/// present hook could then issue capture/composition GPU commands against an image the
/// game is concurrently using for something else entirely, under completely wrong
/// layout assumptions. That kind of concurrent, layout-incoherent access is exactly the
/// class of thing that can wedge a GPU's scheduler -- a plausible, concrete mechanism
/// for a real hang, not merely a memory leak, and one that only needed enough real
/// playtime for a handle to actually get reused, which is why it never showed up in
/// `vkcube` or any short synthetic test.
fn prune_orphaned_tap_source(state: &mut TapTracker, source: vk::Image) {
    if !state.tap_sources_by_destination.values().any(|&src| src == source) {
        state.tapped_source_layouts.remove(&source);
    }
}

type CleanupState = (Arc<ash::Device>, Arc<Mutex<State>>);
static CLEANUP: std::sync::LazyLock<Mutex<HashMap<vk::Device, CleanupState>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Called only from vkDestroyDevice, before downstream device destruction.
/// Vulkan requires the caller to externally synchronize this device AND all its
/// queues here. That host-side contract makes a teardown-only device wait valid;
/// this is deliberately not done in a swapchain resize or per-frame hook.
/// # Safety
/// The caller must meet vkDestroyDevice's external synchronization requirements.
pub(crate) unsafe fn destroy_private_resources(handle: vk::Device) {
    let owned = CLEANUP.lock().unwrap().remove(&handle);
    if let Some((device, state)) = owned {
        let mut state = state.lock().unwrap();
        if state.capture.is_some() || state.capture_pipeline.is_some() || state.direct_capture.iter().any(Option::is_some) || state.gpu_compose.is_some() || !state.relay_semaphores.is_empty() {
            match unsafe { device.device_wait_idle() } {
                Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {
                    unsafe { capture::destroy(state.capture.take(), &device); }
                    unsafe { capture::destroy_pipeline(state.capture_pipeline.take(), &device); }
                    for slot in &mut state.direct_capture {
                        unsafe { capture::destroy_direct_capture(slot.take(), &device); }
                    }
                    if let Some(compose) = state.gpu_compose.take() {
                        unsafe { compose.destroy(&device); }
                    }
                    // SAFETY: device is idle (or lost) and the application's swapchains are
                    // gone, so no present still waits on a relay semaphore.
                    unsafe { state.relay_semaphores.destroy(&device); }
                }
                Err(error) => crate::log!("[layer] teardown wait failed: {:?}; cannot safely free pending resources", error),
            }
        }
        state.swapchains.clear();
        let mut primary = PRIMARY.lock().unwrap();
        if primary.as_ref().is_some_and(|p| p.device == handle) { *primary = None; }
        crate::log!("[layer] private device teardown complete {:?}", handle);
        crate::logging::flush();
    }
}

/// Submits a wait-only batch on `queue` that waits on the application's present wait
/// semaphores (at `ALL_COMMANDS`, so every later submission on the queue is ordered
/// after them) and signals a layer-owned semaphore for the real present to wait on
/// instead. One relay semaphore per swapchain image, like the layer's own present
/// semaphores: the image cannot be presented again before it has been re-acquired,
/// which is after the previous present's wait on the relay has been satisfied.
///
/// # Safety
/// `queue` must be the queue the present was requested on, externally synchronized for
/// the duration of the call, and `app_waits` the present's own wait semaphores.
unsafe fn relay_app_waits(
    device: &ash::Device,
    queue: vk::Queue,
    relay_semaphores: &mut crate::present_sync::PresentSemaphores,
    image: vk::Image,
    app_waits: &[vk::Semaphore],
) -> Option<vk::Semaphore> {
    let relay = relay_semaphores.get(image, || unsafe {
        device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()
    })?;
    let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; app_waits.len()];
    let submit = vk::SubmitInfo::builder()
        .wait_semaphores(app_waits)
        .wait_dst_stage_mask(&stages)
        .signal_semaphores(std::slice::from_ref(&relay))
        .build();
    // SAFETY: the batch references only live handles; `stages` and `app_waits` outlive
    // the call.
    unsafe { device.queue_submit(queue, &[submit], vk::Fence::null()) }.ok()?;
    Some(relay)
}

impl NeuralForgeDeviceInfo {
    pub fn new(
        instance: Option<Arc<ash::Instance>>,
        surface_caps: Option<vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>,
        physical_device: vk::PhysicalDevice,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
        create_info: &vk::DeviceCreateInfo,
    ) -> Self {
        let handle = device.handle();
        // SAFETY: `next_get_device_proc_addr` is the next layer/driver's own
        // `vkGetDeviceProcAddr`, handed to us by the layer framework for exactly this
        // device; each name below matches the `PFN_vk*` type requested.
        let (create, destroy, present, get_images, get_queue, get_queue2) = unsafe {
            (
                resolve::<vk::PFN_vkCreateSwapchainKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkCreateSwapchainKHR",
                ),
                resolve::<vk::PFN_vkDestroySwapchainKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkDestroySwapchainKHR",
                ),
                resolve::<vk::PFN_vkQueuePresentKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkQueuePresentKHR",
                ),
                resolve::<vk::PFN_vkGetSwapchainImagesKHR>(
                    next_get_device_proc_addr,
                    handle,
                    c"vkGetSwapchainImagesKHR",
                ),
                resolve::<vk::PFN_vkGetDeviceQueue>(next_get_device_proc_addr, handle, c"vkGetDeviceQueue"),
                resolve::<vk::PFN_vkGetDeviceQueue2>(next_get_device_proc_addr, handle, c"vkGetDeviceQueue2"),
            )
        };
        // Set by `NeuralForgeInstanceHooks::create_device` (crate::lib) before this
        // device existed at all -- see that function's own doc comment for why this
        // is the only way to learn it here, since `create_info` below always reflects
        // the app's *original*, un-injected request regardless of what actually got
        // enabled. Stored on `State` below; `capture::run` reads it every call to
        // decide between `DirectCapture` and `CapturePipeline` (see
        // docs/EXTERNAL_MEMORY_HOST_DESIGN.md).
        let external_memory_host = crate::take_external_memory_host_enabled(handle);
        crate::log!(
            "[layer] hooked device {:?} (swapchain support: {}, external_memory_host: {})",
            handle,
            create.is_some() && destroy.is_some() && present.is_some(),
            external_memory_host
        );
        // Explicit flush: a one-time-per-device event, not the per-frame hot path
        // `logging::log`'s modulo-64 throttle exists for -- worth the syscall so this
        // milestone survives a process killed by a signal before its normal exit path
        // (confirmed missing this session: a `timeout`-killed `vkcube` lost every log
        // line after this one, including real per-frame activity, purely because
        // nothing forced a flush past this first, coincidentally-flushed call).
        crate::logging::flush();
        let state = Arc::new(Mutex::new(State { external_memory_host, ..State::default() }));
        CLEANUP.lock().unwrap().insert(handle, (device.clone(), state.clone()));
        Self {
            // SAFETY: create_info is the loader chain for this newly created device.
            _loader_data: unsafe { crate::loader_data::register(handle, create_info) },
            device,
            instance,
            surface_caps,
            physical_device,
            next_create_swapchain_khr: create,
            next_destroy_swapchain_khr: destroy,
            next_queue_present_khr: present,
            next_get_swapchain_images_khr: get_images,
            next_get_device_queue: get_queue,
            next_get_device_queue2: get_queue2,
            state,
            tracker: Mutex::new(TapTracker::default()),
        }
    }

    /// The images backing `swapchain`, in the order the loader hands out indices for
    /// `VkPresentInfoKHR::pImageIndices` -- cached once at creation (see
    /// `create_swapchain_khr`) since the list never changes for a swapchain's lifetime.
    fn fetch_swapchain_images(&self, swapchain: vk::SwapchainKHR) -> Vec<vk::Image> {
        let Some(get_images) = self.next_get_swapchain_images_khr else { return Vec::new() };
        let handle = self.device.handle();
        let mut count = 0u32;
        // SAFETY: `get_images` was resolved from the next layer/driver's own proc-addr
        // table; the two-call enumeration pattern (count, then fill) is exactly what
        // the Vulkan spec requires for this function.
        if unsafe { get_images(handle, swapchain, &mut count, std::ptr::null_mut()) } != vk::Result::SUCCESS {
            return Vec::new();
        }
        let mut images = vec![vk::Image::null(); count as usize];
        // SAFETY: `images` has exactly `count` elements, matching what the first call
        // just reported.
        if unsafe { get_images(handle, swapchain, &mut count, images.as_mut_ptr()) } != vk::Result::SUCCESS {
            return Vec::new();
        }
        images
    }

    /// Records the application's transfer into each known swapchain image. A
    /// transfer command itself proves that the source image has the relevant read
    /// usage, making it a candidate for a later render-tap design. This hook is
    /// intentionally observational and always forwards the application command; the
    /// layout it learns is only *committed* when the command buffer is submitted
    /// (see [`TapTracker`]).
    fn observe_swapchain_write(
        &self, kind: &str, command_buffer: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout,
        dst: vk::Image, dst_layout: vk::ImageLayout, region_count: usize,
    ) {
        let first = self.tracker.lock().unwrap().observe_write(command_buffer, src, src_layout, dst);
        if first == Some(true) {
            crate::log!("[layer] observed game {} into swapchain: src={:?} {:?} dst={:?} {:?} regions={}",
                kind, src, src_layout, dst, dst_layout, region_count);
            crate::logging::flush();
        }
    }
}

impl DeviceInfo for NeuralForgeDeviceInfo {
    type HooksType = Self;
    type HooksRefType<'a> = &'a Self;

    fn hooked_commands() -> &'static [VulkanCommand] {
        &[
            VulkanCommand::CreateSwapchainKhr,
            VulkanCommand::DestroySwapchainKhr,
            VulkanCommand::QueuePresentKhr,
            VulkanCommand::GetDeviceQueue,
            VulkanCommand::GetDeviceQueue2,
            VulkanCommand::CmdCopyImage,
            VulkanCommand::CmdBlitImage,
            VulkanCommand::CmdPipelineBarrier,
            VulkanCommand::CmdPipelineBarrier2,
            VulkanCommand::DestroyImage,
            VulkanCommand::BeginCommandBuffer,
            VulkanCommand::FreeCommandBuffers,
            VulkanCommand::CmdExecuteCommands,
            VulkanCommand::QueueSubmit,
            VulkanCommand::QueueSubmit2,
        ]
    }

    fn hooks(&self) -> Self::HooksRefType<'_> {
        self
    }
}

impl DeviceHooks for NeuralForgeDeviceInfo {
    fn create_swapchain_khr(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> LayerResult<ash::prelude::VkResult<vk::SwapchainKHR>> {
        // No `VK_KHR_swapchain` on this device -- see `resolve()`'s doc comment. An app
        // that enabled the extension would never let this be `None`; let the framework's
        // own next-in-chain dispatch handle it exactly as if we weren't here.
        let Some(next_create) = self.next_create_swapchain_khr else {
            return LayerResult::Unhandled;
        };
        let eligible = crate::layer_enabled() && crate::ownership::eligible()
            && swapchain::is_supported_format(create_info.image_format)
            && create_info.image_extent.width <= neural_forge_protocol::MAX_W
            && create_info.image_extent.height <= neural_forge_protocol::MAX_H
            && swapchain::is_plausible_game_size(create_info.image_extent.width, create_info.image_extent.height);
        let adjusted = if eligible {
            self.instance.as_deref().and_then(|instance|
                self.surface_caps.and_then(|query| crate::surface_usage::prepare(instance, query, self.physical_device, create_info)))
        } else { None };
        let pass_through = adjusted.is_none();
        if pass_through && eligible {
            // Distinguishes *why* `prepare` returned `None`: `!candidate` means the
            // swapchain has an extension in `pNext`, non-empty `flags`, more than one
            // array layer, or an exotic present mode (most commonly `pNext` carrying
            // `VkSurfaceFullScreenExclusiveInfoEXT` under exclusive fullscreen) --
            // `prepare` never even queried surface capabilities in that case. `false`
            // means `candidate` passed but the surface itself doesn't support adding
            // TRANSFER_SRC/TRANSFER_DST, or the format/extent/sample-count combination
            // doesn't survive `vkGetPhysicalDeviceImageFormatProperties` with the
            // enlarged usage -- a real surface limitation, not a display-mode artifact.
            // When `candidate` itself rejected it, narrow down which check: `pNext`'s
            // leading `sType` (every `pNext` struct starts with `{sType, pNext}` per
            // the Vulkan spec, so reading it through `VkBaseInStructure` is valid for
            // any real extension struct), `flags`, and `image_array_layers` are each
            // reported directly instead of collapsing them into one boolean, since a
            // display-mode fix (e.g. leaving exclusive fullscreen, which is what adds
            // `VkSurfaceFullScreenExclusiveInfoEXT` to `pNext`) only helps if `pNext`
            // is actually the one that's non-null here.
            let p_next_type = (!create_info.p_next.is_null()).then(|| {
                // SAFETY: a non-null `pNext` on a `VkSwapchainCreateInfoKHR` the
                // application already passed to a real `vkCreateSwapchainKHR` call
                // must point at a valid extension struct, which per spec always
                // begins with `VkStructureType sType`.
                unsafe { (*create_info.p_next.cast::<vk::BaseInStructure>()).s_type }
            });
            crate::log!("[layer] capture admission declined for {}x{} fmt={:?} usage={:?} present_mode={:?} extended_semantics={} p_next_type={:?} flags={:?} array_layers={}",
                create_info.image_extent.width, create_info.image_extent.height,
                create_info.image_format, create_info.image_usage, create_info.present_mode,
                !crate::surface_usage::candidate(create_info), p_next_type, create_info.flags,
                create_info.image_array_layers);
        }
        let mut swapchain = vk::SwapchainKHR::null();
        let alloc_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: only image_usage changes in a private copy with verified support.
        let mut result = unsafe { next_create(self.device.handle(), adjusted.as_ref().unwrap_or(create_info), alloc_ptr, &mut swapchain) };
        // The usage the swapchain actually ends up with -- the enlarged one only if the
        // adjusted creation was the one that succeeded.
        let mut image_usage = adjusted.as_ref().unwrap_or(create_info).image_usage;
        let mut pass_through = pass_through;
        if result != vk::Result::SUCCESS && adjusted.is_some() {
            // The enlarged usage was rejected even though the surface and the format
            // both claimed to support it. Admission is a best-effort enhancement and
            // must never be the reason a game loses its swapchain, so the application's
            // own unmodified creation is tried once more -- with `oldSwapchain` cleared,
            // because the failed attempt above already retired it and passing a retired
            // handle again is invalid. This frame (and this swapchain) simply go
            // un-enhanced, which is the same outcome as declining admission outright.
            let mut fallback = *create_info;
            fallback.old_swapchain = vk::SwapchainKHR::null();
            crate::log!(
                "[layer] adjusted swapchain creation failed ({result:?}); retrying with the application's own usage -- \
                 this swapchain will be pass-through"
            );
            crate::logging::flush();
            swapchain = vk::SwapchainKHR::null();
            // SAFETY: `fallback` is the application's own create info with a cleared
            // `oldSwapchain`; the chain it carries is untouched and still valid.
            result = unsafe { next_create(self.device.handle(), &fallback, alloc_ptr, &mut swapchain) };
            pass_through = true;
            image_usage = fallback.image_usage;
        }
        if result != vk::Result::SUCCESS {
            return LayerResult::Handled(Err(result));
        }
        let hdr_kind = swapchain::detect_hdr_kind(create_info.image_format, create_info.image_color_space);
        // Cache even a pass-through swapchain's images. This lets the passive command
        // diagnostics identify a legal render-to-swapchain transfer without touching
        // the application's creation or recording path.
        let images = self.fetch_swapchain_images(swapchain);
        let state = SwapchainState {
            format: create_info.image_format,
            width: create_info.image_extent.width,
            height: create_info.image_extent.height,
            hdr_kind,
            pass_through,
            image_usage,
            images,
        };
        crate::log!(
            "[layer] swapchain {:?} {}x{} fmt={:?} hdr={} pass_through={} images={}",
            swapchain,
            state.width,
            state.height,
            state.format,
            state.hdr_kind,
            state.pass_through,
            state.images.len()
        );
        let mut layer_state = self.state.lock().unwrap();
        self.tracker.lock().unwrap().swapchain_images.extend(state.images.iter().copied());
        layer_state.swapchains.insert(swapchain, state);
        if !pass_through {
            if let Some(instance) = self.instance.as_deref() {
                layer_state.shm.prepare_motion_resources(instance, self.physical_device,
                    create_info.image_extent.width, create_info.image_extent.height,
                    swapchain::proxy_format_for(create_info.image_format));
            }
        }
        // Explicit flush, same reasoning as `new()`'s -- a one-time-per-swapchain
        // milestone, not the per-frame hot path.
        crate::logging::flush();
        LayerResult::Handled(Ok(swapchain))
    }

    fn get_device_queue(&self, queue_family_index: u32, queue_index: u32) -> LayerResult<vk::Queue> {
        let Some(next) = self.next_get_device_queue else { return LayerResult::Unhandled };
        let mut queue = vk::Queue::null();
        // SAFETY: `next` was resolved from the next layer/driver's own proc-addr
        // table; `queue_family_index`/`queue_index` are the caller's own, forwarded
        // unchanged.
        unsafe { next(self.device.handle(), queue_family_index, queue_index, &mut queue) };
        self.state.lock().unwrap().queue_families.insert(queue, queue_family_index);
        LayerResult::Handled(queue)
    }

    fn get_device_queue2(&self, queue_info: &vk::DeviceQueueInfo2) -> LayerResult<vk::Queue> {
        let Some(next) = self.next_get_device_queue2 else { return LayerResult::Unhandled };
        let mut queue = vk::Queue::null();
        // SAFETY: `next` was resolved from the next layer/driver's own proc-addr
        // table; `queue_info` is valid for the duration of this call (handed to us by
        // the loader for exactly this call).
        unsafe { next(self.device.handle(), queue_info, &mut queue) };
        self.state.lock().unwrap().queue_families.insert(queue, queue_info.queue_family_index);
        LayerResult::Handled(queue)
    }

    fn cmd_copy_image(
        &self, command_buffer: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout,
        dst: vk::Image, dst_layout: vk::ImageLayout, regions: &[vk::ImageCopy],
    ) -> LayerResult<()> {
        self.observe_swapchain_write("copy", command_buffer, src, src_layout, dst, dst_layout, regions.len());
        LayerResult::Unhandled
    }

    fn cmd_blit_image(
        &self, command_buffer: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout,
        dst: vk::Image, dst_layout: vk::ImageLayout, regions: &[vk::ImageBlit], _filter: vk::Filter,
    ) -> LayerResult<()> {
        self.observe_swapchain_write("blit", command_buffer, src, src_layout, dst, dst_layout, regions.len());
        LayerResult::Unhandled
    }

    fn cmd_pipeline_barrier(
        &self, command_buffer: vk::CommandBuffer, _src_stage: vk::PipelineStageFlags,
        _dst_stage: vk::PipelineStageFlags, _dependency: vk::DependencyFlags,
        _memory: &[vk::MemoryBarrier], _buffers: &[vk::BufferMemoryBarrier],
        images: &[vk::ImageMemoryBarrier],
    ) -> LayerResult<()> {
        // No logging here: this fires for every barrier the game records, from every
        // recording thread.
        let mut tracker = self.tracker.lock().unwrap();
        for barrier in images {
            tracker.record_barrier(command_buffer, barrier.image, barrier.new_layout);
        }
        LayerResult::Unhandled
    }

    fn cmd_pipeline_barrier2(
        &self, command_buffer: vk::CommandBuffer, info: &vk::DependencyInfo,
    ) -> LayerResult<()> {
        // SAFETY: the layer framework validated `info` for this application call;
        // the count/pointer pair is valid for the hook's duration.
        let images = unsafe { std::slice::from_raw_parts(info.p_image_memory_barriers,
            info.image_memory_barrier_count as usize) };
        let mut tracker = self.tracker.lock().unwrap();
        for barrier in images {
            tracker.record_barrier(command_buffer, barrier.image, barrier.new_layout);
        }
        LayerResult::Unhandled
    }

    fn begin_command_buffer(
        &self, command_buffer: vk::CommandBuffer, _begin_info: &vk::CommandBufferBeginInfo,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        self.tracker.lock().unwrap().begin_recording(command_buffer);
        LayerResult::Unhandled
    }

    fn free_command_buffers(
        &self, _command_pool: vk::CommandPool, command_buffers: &[vk::CommandBuffer],
    ) -> LayerResult<()> {
        self.tracker.lock().unwrap().free(command_buffers);
        LayerResult::Unhandled
    }

    fn cmd_execute_commands(
        &self, command_buffer: vk::CommandBuffer, secondaries: &[vk::CommandBuffer],
    ) -> LayerResult<()> {
        self.tracker.lock().unwrap().execute_secondary(command_buffer, secondaries);
        LayerResult::Unhandled
    }

    fn queue_submit(
        &self, _queue: vk::Queue, submits: &[vk::SubmitInfo], _fence: vk::Fence,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        let mut tracker = self.tracker.lock().unwrap();
        for submit in submits {
            if submit.command_buffer_count == 0 { continue; }
            // SAFETY: `p_command_buffers` is valid for `command_buffer_count` elements
            // for the duration of the application's own `vkQueueSubmit` call.
            let buffers = unsafe { std::slice::from_raw_parts(submit.p_command_buffers, submit.command_buffer_count as usize) };
            tracker.commit_submit(buffers.iter().copied());
        }
        LayerResult::Unhandled
    }

    fn queue_submit2(
        &self, _queue: vk::Queue, submits: &[vk::SubmitInfo2], _fence: vk::Fence,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        let mut tracker = self.tracker.lock().unwrap();
        for submit in submits {
            if submit.command_buffer_info_count == 0 { continue; }
            // SAFETY: `p_command_buffer_infos` is valid for `command_buffer_info_count`
            // elements for the duration of the application's own `vkQueueSubmit2` call.
            let infos = unsafe { std::slice::from_raw_parts(submit.p_command_buffer_infos, submit.command_buffer_info_count as usize) };
            tracker.commit_submit(infos.iter().map(|info| info.command_buffer));
        }
        LayerResult::Unhandled
    }

    fn destroy_swapchain_khr(
        &self,
        swapchain: vk::SwapchainKHR,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> LayerResult<()> {
        let Some(next_destroy) = self.next_destroy_swapchain_khr else {
            return LayerResult::Unhandled;
        };
        release_primary(self.device.handle(), swapchain);
        {
            let mut state = self.state.lock().unwrap();
            if let Some(old) = state.swapchains.remove(&swapchain) {
                {
                    let mut tracker = self.tracker.lock().unwrap();
                    for image in &old.images {
                        tracker.swapchain_images.remove(image);
                        tracker.observed_swapchain_writes.remove(image);
                        if let Some(source) = tracker.tap_sources_by_destination.remove(image) {
                            prune_orphaned_tap_source(&mut tracker, source);
                        }
                    }
                }
                if let Some(gpu) = &mut state.gpu_compose { gpu.retire_present_images(&old.images); }
                state.relay_semaphores.retire(&old.images);
            }
        }
        let alloc_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: same contract as `create_swapchain_khr` above.
        unsafe { next_destroy(self.device.handle(), swapchain, alloc_ptr) };
        LayerResult::Handled(())
    }

    /// The source-side half of the tap-source lifetime fix -- see
    /// `prune_orphaned_tap_source`'s doc comment for the destination-side half and the
    /// real bug behind both. Pruning only when a *destination* mapping goes away leaves
    /// the most dangerous case open: the game destroys one of its own render targets
    /// (a tap source) while the swapchain that referenced it lives on, the driver hands
    /// that same handle value to a later, unrelated image, and both maps here still
    /// name it as a live source in a known layout. Dropping it the moment the game
    /// destroys it closes that window at the only point it can actually be closed.
    /// Purely observational -- the app's own destroy is always forwarded unchanged.
    fn destroy_image(&self, image: vk::Image, _allocator: Option<&vk::AllocationCallbacks>) -> LayerResult<()> {
        let mut tracker = self.tracker.lock().unwrap();
        // A source can be registered (recorded) before any submission has committed a
        // layout for it, so check both.
        let had_layout = tracker.tapped_source_layouts.remove(&image).is_some();
        if had_layout || tracker.is_source(image) {
            tracker.tap_sources_by_destination.retain(|_, src| *src != image);
        }
        LayerResult::Unhandled
    }

    fn queue_present_khr(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        let Some(next_present) = self.next_queue_present_khr else {
            return LayerResult::Unhandled;
        };
        // Set by `capture::run` only when `composition::gpu::GpuCompose::dispatch_into_image_async`
        // wrote this frame's composited result asynchronously -- see that function's
        // own doc comment. When `Some`, the real present call below *must* wait on it,
        // or the presentation engine could display the image before the GPU work that
        // writes it has actually finished (a real, visible corruption bug, not a style
        // preference).
        let mut wait_semaphore: Option<vk::Semaphore> = None;
        // Set when the application's own present wait semaphores were consumed by a
        // layer-submitted relay batch (see below); the real present must then wait on
        // this instead of them.
        let mut relay_semaphore: Option<vk::Semaphore> = None;
        if crate::layer_enabled() {
            // SAFETY: `p_swapchains`/`p_image_indices`/`swapchain_count` are a valid,
            // parallel pair of slices for the duration of this call -- part of the
            // `VkPresentInfoKHR` the loader just handed us.
            let (swapchains, image_indices) = unsafe {
                (
                    std::slice::from_raw_parts(present_info.p_swapchains, present_info.swapchain_count as usize),
                    std::slice::from_raw_parts(present_info.p_image_indices, present_info.swapchain_count as usize),
                )
            };
            let mut state = self.state.lock().unwrap();
            for (&sc, &image_index) in swapchains.iter().zip(image_indices) {
                let Some(sw) = state.swapchains.get(&sc) else { continue };
                let Some(&image) = sw.images.get(image_index as usize) else { break };
                let tap = self.tracker.lock().unwrap().tap_for(image);
                if sw.pass_through && !sw.image_usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                    // The compose/write-back path lands its result with
                    // `vkCmdCopyBufferToImage`, which is only legal on an image created
                    // with TRANSFER_DST usage. An un-enlarged pass-through swapchain
                    // often lacks it; writing anyway is undefined behaviour on the GPU.
                    static SAID_USAGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID_USAGE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::log!("[layer] present skipped: swapchain is pass_through and was not created with TRANSFER_DST usage ({:?}) \
                                     -- the layer cannot legally write into it", sw.image_usage);
                        crate::logging::flush();
                    }
                    continue;
                }
                if sw.pass_through && !matches!(tap, Some((_, vk::ImageLayout::GENERAL))) {
                    // Why this frame produced no enhancement at all, said out loud.
                    //
                    // A pass-through swapchain can still be composed onto *if* the
                    // game's own render source (the image it blits into the swapchain)
                    // is tracked and currently in GENERAL. That is a layout
                    // coincidence: the tracked layout oscillates
                    // GENERAL -> TRANSFER_SRC_OPTIMAL -> GENERAL as the game records
                    // its own barriers, so whether present lands on a GENERAL means
                    // the difference between "the app enhances the frame" and "the app
                    // silently does nothing". Both have really happened on the same
                    // build minutes apart, which is exactly why this needs to be
                    // diagnosable from a log rather than guessed at.
                    //
                    // Logged once per distinct reason so it identifies the state
                    // without joining the per-barrier spam.
                    static REPORTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
                    let reason = match tap {
                        None => 0u32,
                        Some((_, layout)) => layout.as_raw().unsigned_abs().wrapping_add(1),
                    };
                    if REPORTED.swap(reason, std::sync::atomic::Ordering::Relaxed) != reason {
                        match tap {
                            None => crate::log!(
                                "[layer] present skipped: swapchain is pass_through and this image has no tracked render source \
                                 -- nothing to compose from, so this frame is untouched"
                            ),
                            Some((_, layout)) => crate::log!(
                                "[layer] present skipped: swapchain is pass_through and its render source is in {layout:?}, not GENERAL \
                                 -- capture only triggers on GENERAL, so this frame is untouched"
                            ),
                        }
                        crate::logging::flush();
                    }
                    continue;
                }
                if !claim_primary(self.device.handle(), sc, sw.width, sw.height) {
                    // Another swapchain (or another device) already holds the session's
                    // primary claim -- normal when a game keeps a second swapchain
                    // alive, but indistinguishable from a bug without saying so.
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::log!("[layer] present skipped: this swapchain ({}x{}) is not the session's primary claim", sw.width, sw.height);
                        crate::logging::flush();
                    }
                    break;
                }
                let Some(&queue_family) = state.queue_families.get(&queue) else {
                    // We've never seen this queue via a hooked `vkGetDeviceQueue`/
                    // `vkGetDeviceQueue2` call (e.g. an app using `VK_KHR_synchronization2`
                    // queue submission paths this layer doesn't intercept) -- no family
                    // to build a command pool on, so fail open rather than guess one.
                    //
                    // Silent until 2026-09-17, and silence here is indistinguishable
                    // from the app being broken: every present returns untouched while
                    // the layer looks perfectly healthy in every other respect.
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::log!(
                            "[layer] present skipped: the presenting queue was never seen through vkGetDeviceQueue/vkGetDeviceQueue2, \
                             so its family is unknown and no command pool can be built -- every frame will pass through untouched \
                             ({} queues known)",
                            state.queue_families.len()
                        );
                        crate::logging::flush();
                    }
                    break;
                };
                let width = sw.width;
                let height = sw.height;
                let proxy_format = swapchain::proxy_format_for(sw.format);
                let bgr_order = swapchain::is_bgr_order(sw.format);
                let (capture_image, capture_layout) = tap.unwrap_or((image, vk::ImageLayout::PRESENT_SRC_KHR));
                let State { shm, capture, capture_pipeline, direct_capture, external_memory_host, gpu_compose, original_scratch, model_scratch, inflight, bootstrap_complete, answer_scratch, raw_answer_base, raw_answer_generation, last_answer, last_answer_dims, hotkey, relay_semaphores, .. } = &mut *state;
                shm.poll_toggle_hotkey(hotkey);
                if shm.model_known_unavailable() {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::log!("[layer] present skipped: the helper reported the model permanently unavailable for this session");
                        crate::logging::flush();
                    }
                    // The helper has permanently disabled itself for this session
                    // (see `ngx::ensure_feature`'s one-shot design) -- nothing will
                    // ever evaluate a captured frame, so paying for the capture
                    // itself (a full image<->buffer round trip plus a whole-frame
                    // `memcpy`, every single present call) is pure waste. Skip
                    // straight to a real no-op present, matching what "fail-open"
                    // should actually cost: nothing.
                    break;
                }
                // The application's present waits (its "rendering finished" semaphores)
                // are what make the swapchain image safe to read. Every capture/compose
                // submission below goes onto `queue` *before* the real present is
                // called, so without this the layer could read or overwrite the image
                // while the game -- which may render on a different queue than it
                // presents on -- is still drawing into it.
                //
                // A wait in a `vkQueueSubmit` batch orders every later submission on
                // that queue (Vulkan spec, "Semaphore Waiting": the second
                // synchronization scope includes all commands later in submission
                // order), for the stages in the wait's destination mask. So one
                // wait-only relay batch, ahead of all the layer's own work, covers every
                // submit `capture::run` might make. It consumes the application's
                // binary semaphores, so it re-signals a layer-owned one that the real
                // present waits on in their place.
                // SAFETY: `p_wait_semaphores` is valid for `wait_semaphore_count`
                // elements for the duration of this call.
                let app_waits: &[vk::Semaphore] = if present_info.wait_semaphore_count == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(present_info.p_wait_semaphores, present_info.wait_semaphore_count as usize) }
                };
                if !app_waits.is_empty() {
                    // SAFETY: same queue-synchronization contract as `capture::run`
                    // below; `app_waits` are the application's own present waits.
                    match unsafe { relay_app_waits(&self.device, queue, relay_semaphores, image, app_waits) } {
                        Some(relay) => relay_semaphore = Some(relay),
                        None => {
                            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                crate::log!("[layer] present skipped: could not relay the application's present wait semaphores, \
                                             so no layer work can safely read the swapchain image");
                                crate::logging::flush();
                            }
                            break;
                        }
                    }
                }
                if let Some(instance) = &self.instance {
                    // SAFETY: `queue` is the same queue this present call was made on,
                    // externally synchronized for its duration by the same Vulkan rule
                    // that lets the caller call `vkQueuePresentKHR` on it at all right
                    // after this returns -- exactly this function's own safety
                    // contract. `image` is one of `sc`'s own images, currently
                    // `PRESENT_SRC_KHR` per `vkQueuePresentKHR`'s precondition on every
                    // image it's about to present.
                    unsafe {
                        wait_semaphore = capture::run(
                            &self.device,
                            instance,
                            self.physical_device,
                            queue,
                            queue_family,
                            capture_image,
                            capture_layout,
                            image,
                            width,
                            height,
                            proxy_format,
                            bgr_order,
                            capture,
                            capture_pipeline,
                            direct_capture,
                            *external_memory_host,
                            gpu_compose,
                            shm,
                            original_scratch,
                            model_scratch,
                            inflight,
                            bootstrap_complete,
                            answer_scratch,
                            raw_answer_base,
                            raw_answer_generation,
                            last_answer,
                            last_answer_dims,
                        );
                    }
                }
                break;
            }
        }

        // SAFETY: `present_info` is valid for the duration of this call; `next_present`
        // was resolved from the next layer/driver's own proc-addr table.
        let result = if relay_semaphore.is_some() || wait_semaphore.is_some() {
            // Never drop a dependency the application had: either its own wait
            // semaphores go through unchanged (the layer did no relay), or the relay
            // batch already waited on them and its semaphore stands in for them.
            // `capture::run`'s own compute work is an *additional* dependency.
            let mut combined: Vec<vk::Semaphore> = Vec::with_capacity(present_info.wait_semaphore_count as usize + 2);
            match relay_semaphore {
                Some(relay) => combined.push(relay),
                None if present_info.wait_semaphore_count > 0 => {
                    // SAFETY: `p_wait_semaphores` is a valid slice of `wait_semaphore_count`
                    // elements per `present_info`'s own contract, valid for this call's
                    // duration.
                    combined.extend_from_slice(unsafe {
                        std::slice::from_raw_parts(present_info.p_wait_semaphores, present_info.wait_semaphore_count as usize)
                    });
                }
                None => {}
            }
            if let Some(sem) = wait_semaphore {
                combined.push(sem);
            }
            // Copies every other field (`p_next`, `swapchain_count`, `p_swapchains`,
            // `p_image_indices`, `p_results`) unchanged from the app's own
            // `present_info` -- only the wait-semaphore list is actually different.
            let modified_info = vk::PresentInfoKHR { wait_semaphore_count: combined.len() as u32, p_wait_semaphores: combined.as_ptr(), ..*present_info };
            // SAFETY: `modified_info` is valid for the duration of this call --
            // `combined` (which it borrows from) outlives it; `next_present` was
            // resolved from the next layer/driver's own proc-addr table.
            unsafe { next_present(queue, &modified_info) }
        } else {
            // SAFETY: `present_info` is valid for the duration of this call;
            // `next_present` was resolved from the next layer/driver's own
            // proc-addr table.
            unsafe { next_present(queue, present_info) }
        };
        LayerResult::Handled(result.result())
    }
}

#[cfg(test)]
mod tap_source_lifetime_tests {
    use super::*;
    use ash::vk::Handle;

    fn image(raw: u64) -> vk::Image {
        vk::Image::from_raw(raw)
    }

    /// The exact scenario `prune_orphaned_tap_source`'s own doc comment describes:
    /// once nothing in `tap_sources_by_destination` points at a source any more, its
    /// `tapped_source_layouts` entry must actually go, not sit there for the rest of
    /// the process's life -- real 2026-09-16 bug, this guards against reintroducing it.
    #[test]
    fn prune_orphaned_tap_source_removes_a_truly_unreferenced_source() {
        let mut state = TapTracker::default();
        let source = image(1);
        state.tapped_source_layouts.insert(source, vk::ImageLayout::GENERAL);
        // No entry in `tap_sources_by_destination` points at `source` at all.
        prune_orphaned_tap_source(&mut state, source);
        assert!(!state.tapped_source_layouts.contains_key(&source));
    }

    #[test]
    fn prune_orphaned_tap_source_keeps_a_source_still_referenced_elsewhere() {
        let mut state = TapTracker::default();
        let source = image(1);
        state.tapped_source_layouts.insert(source, vk::ImageLayout::GENERAL);
        // A second, still-live destination also reads from this same source image --
        // pruning must not remove it out from under that live reference.
        state.tap_sources_by_destination.insert(image(2), source);
        prune_orphaned_tap_source(&mut state, source);
        assert!(state.tapped_source_layouts.contains_key(&source));
    }

    /// Simulates `destroy_swapchain_khr`'s own cleanup loop directly against `State`
    /// (its real trait method needs a live Vulkan device this test has no reason to
    /// stand up) -- proves a destroyed swapchain's own destination images no longer
    /// leave their source orphaned in `tapped_source_layouts` once nothing else
    /// references it, the actual leak this session found via a real GPU hang.
    #[test]
    fn destroying_the_only_swapchain_referencing_a_source_prunes_it() {
        let mut state = TapTracker::default();
        let source = image(10);
        let destination = image(20);
        state.tapped_source_layouts.insert(source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        state.tap_sources_by_destination.insert(destination, source);

        // What `destroy_swapchain_khr` does for each of the destroyed swapchain's own
        // images.
        if let Some(removed_source) = state.tap_sources_by_destination.remove(&destination) {
            prune_orphaned_tap_source(&mut state, removed_source);
        }

        assert!(!state.tap_sources_by_destination.contains_key(&destination));
        assert!(!state.tapped_source_layouts.contains_key(&source));
    }

    /// Two live swapchains sharing one source image (a real, normal case -- e.g. the
    /// same off-screen render target blitted into two different swapchains): only
    /// destroying *both* destinations should prune the shared source.
    #[test]
    fn a_source_shared_by_two_destinations_survives_until_both_are_gone() {
        let mut state = TapTracker::default();
        let source = image(10);
        state.tapped_source_layouts.insert(source, vk::ImageLayout::GENERAL);
        state.tap_sources_by_destination.insert(image(20), source);
        state.tap_sources_by_destination.insert(image(21), source);

        state.tap_sources_by_destination.remove(&image(20));
        prune_orphaned_tap_source(&mut state, source);
        assert!(state.tapped_source_layouts.contains_key(&source), "still referenced by image(21)");

        state.tap_sources_by_destination.remove(&image(21));
        prune_orphaned_tap_source(&mut state, source);
        assert!(!state.tapped_source_layouts.contains_key(&source));
    }

    /// `observe_swapchain_write`'s own re-target case: a destination that starts
    /// pointing at one source and later points at a different one must orphan-check
    /// the *old* source, the same way losing the destination entirely does.
    #[test]
    fn retargeting_a_destination_to_a_new_source_prunes_the_old_one_if_unreferenced() {
        let mut state = TapTracker::default();
        let destination = image(20);
        let old_source = image(1);
        let new_source = image(2);
        state.tapped_source_layouts.insert(old_source, vk::ImageLayout::GENERAL);
        state.tap_sources_by_destination.insert(destination, old_source);

        // What `observe_swapchain_write` does when a destination's source changes.
        state.tapped_source_layouts.insert(new_source, vk::ImageLayout::GENERAL);
        if let Some(previous_source) = state.tap_sources_by_destination.insert(destination, new_source) {
            if previous_source != new_source {
                prune_orphaned_tap_source(&mut state, previous_source);
            }
        }

        assert!(!state.tapped_source_layouts.contains_key(&old_source));
        assert!(state.tapped_source_layouts.contains_key(&new_source));
    }

    /// `destroy_image`'s own logic against `State`: the game destroying a *source*
    /// image (its own render target) while the swapchain that reads from it is still
    /// alive must drop both the layout entry and every destination mapping naming it
    /// -- the handle-reuse hazard `prune_orphaned_tap_source`'s doc comment describes,
    /// which destination-side pruning alone can never catch.
    #[test]
    fn destroying_a_source_image_drops_its_layout_and_every_mapping_to_it() {
        let mut state = TapTracker::default();
        let source = image(10);
        let unrelated_source = image(11);
        state.tapped_source_layouts.insert(source, vk::ImageLayout::GENERAL);
        state.tapped_source_layouts.insert(unrelated_source, vk::ImageLayout::GENERAL);
        state.tap_sources_by_destination.insert(image(20), source);
        state.tap_sources_by_destination.insert(image(21), source);
        state.tap_sources_by_destination.insert(image(22), unrelated_source);

        // What `destroy_image` does when the game frees `source`.
        if state.tapped_source_layouts.remove(&source).is_some() {
            state.tap_sources_by_destination.retain(|_, src| *src != source);
        }

        assert!(!state.tapped_source_layouts.contains_key(&source));
        assert!(!state.tap_sources_by_destination.contains_key(&image(20)));
        assert!(!state.tap_sources_by_destination.contains_key(&image(21)));
        // The unrelated source and its own destination are untouched.
        assert!(state.tapped_source_layouts.contains_key(&unrelated_source));
        assert_eq!(state.tap_sources_by_destination.get(&image(22)), Some(&unrelated_source));
    }
}

#[cfg(test)]
mod submit_order_tests {
    use super::*;
    use ash::vk::Handle;

    fn image(raw: u64) -> vk::Image { vk::Image::from_raw(raw) }
    fn cb(raw: u64) -> vk::CommandBuffer { vk::CommandBuffer::from_raw(raw) }

    fn tracker_with_swapchain() -> TapTracker {
        let mut t = TapTracker::default();
        t.swapchain_images.insert(image(100));
        t
    }

    /// The bug being fixed: a layout recorded into a command buffer must not be
    /// visible to the present hook until that buffer is actually submitted.
    #[test]
    fn recorded_layout_is_invisible_until_submit() {
        let mut t = tracker_with_swapchain();
        assert_eq!(t.observe_write(cb(1), image(1), vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image(100)), Some(true));
        assert_eq!(t.tap_for(image(100)), None);
        t.commit_submit([cb(1)].into_iter());
        assert_eq!(t.tap_for(image(100)), Some((image(1), vk::ImageLayout::TRANSFER_SRC_OPTIMAL)));
    }

    /// Two buffers recorded in one order and submitted in the other: the committed
    /// layout is the last *submitted* one.
    #[test]
    fn submission_order_wins_over_recording_order() {
        let mut t = tracker_with_swapchain();
        t.observe_write(cb(1), image(1), vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image(100));
        t.record_barrier(cb(2), image(1), vk::ImageLayout::GENERAL); // recorded second...
        t.commit_submit([cb(2), cb(1)].into_iter()); // ...submitted first
        assert_eq!(t.tap_for(image(100)).map(|x| x.1), Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL));
    }

    #[test]
    fn a_barrier_after_the_blit_in_the_same_buffer_is_the_final_layout() {
        let mut t = tracker_with_swapchain();
        t.observe_write(cb(1), image(1), vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image(100));
        t.record_barrier(cb(1), image(1), vk::ImageLayout::GENERAL);
        t.commit_submit([cb(1)].into_iter());
        assert_eq!(t.tap_for(image(100)).map(|x| x.1), Some(vk::ImageLayout::GENERAL));
    }

    #[test]
    fn unsubmitted_and_re_recorded_buffers_never_commit() {
        let mut t = tracker_with_swapchain();
        t.observe_write(cb(1), image(1), vk::ImageLayout::GENERAL, image(100));
        t.begin_recording(cb(1)); // buffer re-recorded: old contents are gone
        t.commit_submit([cb(1)].into_iter());
        assert_eq!(t.tap_for(image(100)), None);
    }

    #[test]
    fn barriers_on_unrelated_images_are_ignored() {
        let mut t = tracker_with_swapchain();
        t.record_barrier(cb(1), image(7), vk::ImageLayout::GENERAL);
        assert!(t.pending.is_empty());
    }

    #[test]
    fn writes_into_non_swapchain_images_are_ignored() {
        let mut t = tracker_with_swapchain();
        assert_eq!(t.observe_write(cb(1), image(1), vk::ImageLayout::GENERAL, image(555)), None);
        assert!(t.pending.is_empty() && t.tap_sources_by_destination.is_empty());
    }

    #[test]
    fn secondary_buffers_contribute_when_executed() {
        let mut t = tracker_with_swapchain();
        t.observe_write(cb(1), image(1), vk::ImageLayout::GENERAL, image(100));
        t.execute_secondary(cb(9), &[cb(1)]);
        t.commit_submit([cb(9)].into_iter());
        assert_eq!(t.tap_for(image(100)).map(|x| x.1), Some(vk::ImageLayout::GENERAL));
        t.free(&[cb(1), cb(9)]);
        assert!(t.pending.is_empty());
    }

    #[test]
    fn a_source_destroyed_before_submit_is_not_committed() {
        let mut t = tracker_with_swapchain();
        t.observe_write(cb(1), image(1), vk::ImageLayout::GENERAL, image(100));
        t.tap_sources_by_destination.clear(); // what destroy_image does to a source
        t.commit_submit([cb(1)].into_iter());
        assert!(t.tapped_source_layouts.is_empty());
    }

    /// Real (lavapipe) queue: the relay batch must consume the application's wait
    /// semaphore and hand back a semaphore that a later wait can consume, including
    /// when the same image is relayed again.
    #[test]
    fn relay_passes_the_application_wait_through_on_a_real_queue() {
        let Some((_entry, instance, _physical, device, queue, _family)) = crate::composition::gpu::test_device() else {
            eprintln!("relay test: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let mut relays = crate::present_sync::PresentSemaphores::default();
        unsafe {
            let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None).unwrap();
            for round in 0..3 {
                let app = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap();
                // The "game's" own submit signalling its present semaphore.
                device.queue_submit(queue, &[vk::SubmitInfo::builder().signal_semaphores(std::slice::from_ref(&app)).build()], vk::Fence::null()).unwrap();
                let relay = relay_app_waits(&device, queue, &mut relays, image(42), &[app]).expect("relay submit");
                let stage = vk::PipelineStageFlags::ALL_COMMANDS;
                let present_like = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&relay)).wait_dst_stage_mask(std::slice::from_ref(&stage)).build();
                device.queue_submit(queue, &[present_like], fence).unwrap();
                device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                device.reset_fences(&[fence]).unwrap();
                device.destroy_semaphore(app, None);
                assert!(!relays.is_empty(), "round {round}");
            }
            device.queue_wait_idle(queue).unwrap();
            relays.destroy(&device);
            device.destroy_fence(fence, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
