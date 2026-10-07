//! `VK_LAYER_neuralforge_neural` — the Linux-side Vulkan implicit layer.
//!
//! Hooks the swapchain lifecycle (`vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/
//! `vkQueuePresentKHR`) and exchanges frames with the helper over the shared-memory
//! transport defined in `neural_forge_protocol`. This crate never knows or cares whether the
//! helper on the other end of that mapping is running under Wine/Proton today or a
//! native Linux process later — that's the whole point of the seam.
//!
//! Built on Google's [`vulkan_layer`](https://github.com/google/vk-layer-for-rust)
//! crate, which supplies the actual `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr`
//! dispatch machinery, the `VkLayerInstanceCreateInfo`/`VkLayerDeviceCreateInfo`
//! chain-walk, and the loader-negotiation entry points — this crate only implements
//! [`vulkan_layer::DeviceHooks`] for the handful of functions it actually cares about;
//! everything else falls through to the next layer/driver automatically.
//!
//! Host shared-memory capture and GPU composition are implemented. Cross-process
//! ownership and executable filtering guard the channel; the swapchain size filter
//! additionally excludes small overlays. DMA-BUF remains experimental. See
//! docs/HARDWARE_VALIDATION.md for presentation-validation failures still under review.

mod breadcrumbs;
mod capture;
mod composition;
mod device;
mod ownership;
mod loader_data;
mod dump;
mod series;
mod gpu_timer;
mod logging;
mod hotkey;
mod shm;
mod swapchain;
mod surface_usage;
mod entry_points;
mod present_sync;
mod probe_ngx;
mod probe_seq;
mod preupscale;

// The native backend (crates/native), linked into the 64-bit layer; nothing calls it yet.
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(crate) use neural_forge_native as native;

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use ash::vk;
use std::sync::LazyLock;
use vulkan_layer::{
    auto_globalhooksinfo_impl, declare_introspection_queries, Global, GlobalHooks, InstanceHooks, InstanceInfo,
    Layer, LayerManifest, LayerResult, LayerVulkanCommand, VkLayerDeviceLink, VkLayerInstanceLink,
};

use device::NeuralForgeDeviceInfo;

/// What the layer keeps about one `VkInstance`: an `ash::Instance` to query
/// physical-device properties through when it builds capture resources, and the next
/// layer's surface-capabilities query.
#[derive(Clone)]
struct InstanceContext {
    instance: Arc<ash::Instance>,
    surface_caps: Option<vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>,
}

/// The owning instance of each physical device a device is being created on, so
/// `create_device_info` (which the `vulkan_layer` framework calls with only the physical
/// device, no way to reach the instance's own hooks) resolves it through the right
/// instance. Recorded by the owning instance's `create_device` hook, which the framework
/// calls first, and dropped with the instance (see `NeuralForgeInstanceHooks`'s `Drop`):
/// a launcher that creates a probe instance after the real one, or destroys one, never
/// has a device resolved through someone else's (or a dead) instance.
static PHYSICAL_DEVICE_INSTANCES: LazyLock<Mutex<HashMap<vk::PhysicalDevice, InstanceContext>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub const LAYER_NAME: &str = "VK_LAYER_neuralforge_neural";

/// Whether the pass should do anything at all. Off by default (`NEURAL_FORGE_ENABLE` unset)
/// so the layer is a true no-op for every game that hasn't opted in via its launch
/// options — checked once and cached, same as upstream, since it can't change for the
/// life of the process.
pub(crate) fn layer_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        env_flag("NEURAL_FORGE_ENABLE") && !env_flag("NEURAL_FORGE_DISABLE") && !duplicate_copy()
    });
    *ENABLED
}

/// The path of the shared object this code is running from.
fn layer_object_path() -> Option<String> {
    // SAFETY: `dladdr` only reads the address it is given; `info` is zero-initialised and
    // filled in on success, and `dli_fname` (when non-null) is a NUL-terminated string owned by
    // the loader for the life of the mapping.
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        if libc::dladdr(layer_object_path as *const std::ffi::c_void, &mut info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        Some(CStr::from_ptr(info.dli_fname).to_string_lossy().into_owned())
    }
}

/// Set when this copy of the layer has decided to be the live one. Exported unmangled so
/// another copy loaded into the same process can look it up with `dlsym` (see
/// [`duplicate_copy`]).
#[no_mangle]
pub static NEURAL_FORGE_LAYER_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True when a *different* copy of this layer is already live in the process.
///
/// A development manifest pointing at the build tree and an installed one pointing at the
/// install prefix are both honoured by the loader: two copies, two present hooks, two full
/// round trips and one shared-memory file with two writers racing on one sequence number. Only
/// the first copy to decide stays live; the rest go inert with one warning line. Each copy
/// finds the others as mappings of a same-named object at another path in `/proc/self/maps`
/// and asks each, through its exported [`NEURAL_FORGE_LAYER_LIVE`], whether it already went
/// live. (This used to be a process environment variable, but `setenv` inside a running,
/// multi-threaded game can race a concurrent `getenv` on another thread.) The copies decide in
/// hook-chain order, one after the other, so the first one reached wins.
fn duplicate_copy() -> bool {
    let Some(own) = layer_object_path() else { return false };
    let own = std::fs::canonicalize(&own).map_or(own, |p| p.to_string_lossy().into_owned());
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    if let Some(live) = other_copies(&maps, &own).into_iter().find(|path| copy_is_live(path)) {
        crate::log!("[layer] another copy is already loaded from {live}; this copy ({own}) stays inert. Remove one of the implicit-layer manifests.");
        return true;
    }
    NEURAL_FORGE_LAYER_LIVE.store(true, std::sync::atomic::Ordering::Relaxed);
    false
}

/// The paths of every mapped object in `maps` (the text of `/proc/self/maps`) with the same
/// file name as `own` but another path.
fn other_copies(maps: &str, own: &str) -> Vec<String> {
    let name = std::path::Path::new(own).file_name();
    let mut found: Vec<String> = Vec::new();
    for line in maps.lines() {
        // address perms offset dev inode path -- the path is everything after the fifth field.
        let mut rest = line;
        for _ in 0..5 {
            rest = rest.trim_start();
            rest = rest.find(char::is_whitespace).map_or("", |at| &rest[at..]);
        }
        let path = rest.trim();
        if path.starts_with('/') && path != own && std::path::Path::new(path).file_name() == name && !found.iter().any(|p| p == path) {
            found.push(path.to_string());
        }
    }
    found
}

/// Whether the copy of this layer mapped from `path` has gone live. Only looks at an object
/// that is already loaded (`RTLD_NOLOAD`): never loads anything.
fn copy_is_live(path: &str) -> bool {
    let Ok(c_path) = std::ffi::CString::new(path) else { return false };
    // SAFETY: `RTLD_NOLOAD` only returns a handle to an object that is already loaded (adding
    // a reference, released below) and never runs anyone's initializers.
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_NOLOAD) };
    if handle.is_null() {
        return false;
    }
    // SAFETY: the symbol, when present, is that copy's own `NEURAL_FORGE_LAYER_LIVE`, an
    // `AtomicBool` that lives as long as the object, which the handle keeps loaded.
    let live = unsafe {
        let symbol = libc::dlsym(handle, c"NEURAL_FORGE_LAYER_LIVE".as_ptr());
        !symbol.is_null() && (*symbol.cast::<std::sync::atomic::AtomicBool>()).load(std::sync::atomic::Ordering::Relaxed)
    };
    // SAFETY: releases the reference `dlopen` added above.
    unsafe { libc::dlclose(handle) };
    live
}

/// Latched by [`note_vk`] the first time a layer-issued Vulkan call reports `VK_ERROR_DEVICE_LOST`.
/// After that every call fails the same way, so the layer takes the fail-open path directly:
/// present untouched, submit nothing, log nothing more.
static DEVICE_LOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn device_lost() -> bool {
    DEVICE_LOST.load(std::sync::atomic::Ordering::Relaxed)
}

/// Passes a Vulkan result through unchanged, latching the layer inert on `DEVICE_LOST`.
pub(crate) fn note_vk<T>(result: ash::prelude::VkResult<T>) -> ash::prelude::VkResult<T> {
    if matches!(result, Err(vk::Result::ERROR_DEVICE_LOST)) && !DEVICE_LOST.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::log!("[layer] VK_ERROR_DEVICE_LOST; layer inert from here on");
        crate::logging::flush();
    }
    result
}

/// Generous-but-finite budget for a fence wait that guards this project's own GPU
/// work (compose dispatches, and the rare rebuild-drain waits in `capture.rs`). Every
/// such wait used to pass `u64::MAX`: a lost device already returns
/// `VK_ERROR_DEVICE_LOST` from the wait rather than hanging, so the case this bounds
/// is a driver that stalls *without* losing the device -- the call never returns, the
/// game's present thread parks, and nothing is logged, because there is no result for
/// `note_vk` to see. Five seconds is long enough that a genuinely busy dispatch (4K,
/// mode 2, several passes) never trips it, but short enough that a real stall becomes
/// a bounded, diagnosable stutter instead of an unexplained freeze.
///
/// Pattern and budget from PR #22 against DLSS5VKLayer (bmitch87), commit `4aa730c0`
/// ("Bound the fence waits, including the two in the game's present path") -- see
/// `ATTRIBUTION.md`.
pub(crate) const FENCE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Wraps a bounded (`FENCE_WAIT_TIMEOUT`) `wait_for_fences` result: marks `site` in
/// `breadcrumbs` (every call, not just failures -- these calls are rare enough per
/// frame that the trail is cheap and worth having regardless of outcome), passes the
/// result through [`note_vk`] as usual, and additionally logs once per process, by
/// name, plus a full breadcrumb dump, the first time one of these waits actually times
/// out -- so a real occurrence is diagnosable ("this site stalled, and here is what led
/// up to it") instead of indistinguishable from any other Vulkan error.
pub(crate) fn note_fence_wait(result: ash::prelude::VkResult<()>, site: &'static str) -> ash::prelude::VkResult<()> {
    crate::breadcrumbs::mark(site);
    if matches!(result, Err(vk::Result::TIMEOUT)) {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!(
                "[layer] fence wait timed out after {FENCE_WAIT_TIMEOUT:?} at {site} -- \
                 driver stall without device loss; failing open and continuing rather than hanging"
            );
            crate::breadcrumbs::dump("fence wait timeout");
        }
    }
    note_vk(result)
}

fn env_flag(name: &str) -> bool {
    neural_forge_protocol::env::flag(name)
}

/// Works around a crash inside Mesa's `device_select` implicit layer, confirmed
/// reproducible even with `vulkan-layer`'s own pristine `hello-world` example under
/// implicit activation on this machine's Mesa build (see `CLAUDE.md`'s "CRITICAL,
/// confirmed" section for the full bisection) -- so this is a bug in the interaction
/// between the pinned `vulkan-layer` commit and this Mesa build, not anything specific
/// to this crate. `vulkan_layer::Global::create_instance`'s default fallback path
/// (taken whenever `GlobalHooks::create_instance` is `Unhandled`) eagerly resolves all
/// three Vulkan 1.0 global entry points -- `vkCreateInstance`,
/// `vkEnumerateInstanceExtensionProperties`, `vkEnumerateInstanceLayerProperties` --
/// through the chained, `VK_NULL_HANDLE`-instance `vkGetInstanceProcAddr`
/// (`ash::vk::EntryFnV1_0::load`), even though only `vkCreateInstance` is ever actually
/// called afterward. Resolving `vkEnumerateInstanceExtensionProperties` that way
/// segfaults inside `libVkLayer_MESA_device_select.so` 100% of the time (confirmed via
/// `gdb`: `vkCreateInstance` resolves fine through the exact same chained pointer,
/// `vkEnumerateInstanceExtensionProperties` right after it does not). Hooking
/// `create_instance` ourselves and resolving only the one entry point this layer
/// actually needs avoids ever making the query that crashes.
#[derive(Default)]
struct NeuralForgeGlobalHooks;

#[auto_globalhooksinfo_impl]
impl GlobalHooks for NeuralForgeGlobalHooks {
    fn create_instance(
        &self,
        create_info: &vk::InstanceCreateInfo,
        layer_instance_link: &VkLayerInstanceLink,
        allocator: Option<&vk::AllocationCallbacks>,
        p_instance: *mut vk::Instance,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        // SAFETY: `layer_instance_link.pfnNextGetInstanceProcAddr` is the loader- or
        // next-layer-supplied chained `vkGetInstanceProcAddr`, valid for the duration of
        // this call; `VK_NULL_HANDLE` + a global-command name is the spec-mandated way
        // to query it before an instance exists.
        let create_instance = unsafe {
            (layer_instance_link.pfnNextGetInstanceProcAddr)(vk::Instance::null(), c"vkCreateInstance".as_ptr())
        };
        let create_instance: vk::PFN_vkCreateInstance = match create_instance {
            // SAFETY: a non-null `vkGetInstanceProcAddr(NULL, "vkCreateInstance")` result
            // is guaranteed by the Vulkan spec to have this exact signature.
            Some(fp) => unsafe { std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkCreateInstance>(fp) },
            None => return LayerResult::Handled(Err(vk::Result::ERROR_INITIALIZATION_FAILED)),
        };
        let allocator = allocator.map_or(std::ptr::null(), |allocator| allocator as *const _);
        // SAFETY: `create_info`/`p_instance` are the same, still-valid pointers the
        // framework was called with; `allocator` is either null or that same valid
        // pointer.
        LayerResult::Handled(unsafe { create_instance(create_info, allocator, p_instance) }.result())
    }
}

/// The one device extension this layer ever asks a game's own device creation to add,
/// for Phase 3's zero-copy capture path (`docs/ASYNC_CAPTURE_DESIGN.md`): importing the SHM
/// proxy region directly as device memory needs it on whichever device the layer's own
/// capture commands submit against -- the game's, not a private one, since the image
/// being copied is the game's own swapchain/render-tap source.
pub(crate) const EXTERNAL_MEMORY_HOST_EXTENSION: &CStr = c"VK_EXT_external_memory_host";

/// Devices [`NeuralForgeInstanceHooks::create_device`] actually added
/// [`EXTERNAL_MEMORY_HOST_EXTENSION`] to. `create_device_info` (the framework's own,
/// called right after with the *original*, un-injected `VkDeviceCreateInfo` regardless
/// of what a hooked `create_device` actually passed to the real driver -- there is no
/// other way to learn this) checks and removes its own device's entry here exactly
/// once. Same pattern as `device::CLEANUP`/`PHYSICAL_DEVICE_INSTANCES`: a small, short-lived,
/// mutex-guarded side table, not a source of truth kept around indefinitely.
static EXTERNAL_MEMORY_HOST_DEVICES: Mutex<Option<HashSet<vk::Device>>> = Mutex::new(None);

/// Whether a `VkDeviceCreateInfo` itself requests the extension `name`. A request with zero
/// extensions may legitimately leave `pp_enabled_extension_names` null (`vkcube` does), which
/// `slice::from_raw_parts` must never see.
pub(crate) fn requests_extension(create_info: &vk::DeviceCreateInfo, name: &CStr) -> bool {
    if create_info.enabled_extension_count == 0 || create_info.pp_enabled_extension_names.is_null() {
        return false;
    }
    // SAFETY: a non-null `pp_enabled_extension_names` is an array of `enabled_extension_count`
    // valid, NUL-terminated C strings per `VkDeviceCreateInfo`'s own contract.
    let requested = unsafe { std::slice::from_raw_parts(create_info.pp_enabled_extension_names, create_info.enabled_extension_count as usize) };
    requested.iter().any(|&requested| unsafe { CStr::from_ptr(requested) } == name)
}

/// Checks and clears whether `device` is one [`NeuralForgeInstanceHooks::create_device`]
/// created with [`EXTERNAL_MEMORY_HOST_EXTENSION`] enabled (added by the hook, or already
/// requested by the application).
pub(crate) fn take_external_memory_host_enabled(device: vk::Device) -> bool {
    EXTERNAL_MEMORY_HOST_DEVICES.lock().unwrap().as_mut().is_some_and(|set| set.remove(&device))
}

/// Devices created with the layer's own extra queue (the hold inside DLSS's command buffer,
/// `preupscale::inline`): its family and index, and the game's graphics family. Taken once by
/// `NeuralForgeDeviceInfo::new`, like [`EXTERNAL_MEMORY_HOST_DEVICES`].
static SIDE_QUEUES: Mutex<Option<HashMap<vk::Device, SideQueue>>> = Mutex::new(None);

/// Where the layer's own queue is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SideQueue {
    pub family: u32,
    pub index: u32,
    /// The first family the application requests with graphics and compute: where DLSS runs.
    pub app_family: u32,
}

/// Devices created with the native backend's additions (`preupscale::native`): its extensions and
/// features, and one more queue in the game's graphics family for the network's loading. Taken once by
/// `NeuralForgeDeviceInfo::new`.
#[cfg(target_arch = "x86_64")]
static NATIVE_DEVICES: Mutex<Option<HashMap<vk::Device, preupscale::native::Setup>>> = Mutex::new(None);

/// Checks and clears the native backend's setup for `device`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn take_native(device: vk::Device) -> Option<preupscale::native::Setup> {
    NATIVE_DEVICES.lock().unwrap().as_mut().and_then(|map| map.remove(&device))
}

/// `infos` with one more queue in the first family that has graphics and compute (the game's, where
/// DLSS runs): the family and the new queue's index, the extended infos, and the priorities they point
/// into. `None` when that family's request has flags or no queue to spare.
#[cfg(target_arch = "x86_64")]
fn native_queue_request(
    instance: &ash::Instance, physical_device: vk::PhysicalDevice, infos: &[vk::DeviceQueueCreateInfo],
) -> Option<(u32, u32, Vec<vk::DeviceQueueCreateInfo>, Box<Vec<f32>>)> {
    // SAFETY: `physical_device` belongs to `instance`.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let at = infos
        .iter()
        .position(|q| families.get(q.queue_family_index as usize).is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)))?;
    let q = infos[at];
    if !q.flags.is_empty() || q.queue_count >= families[q.queue_family_index as usize].queue_count || q.p_queue_priorities.is_null() {
        return None;
    }
    // SAFETY: `pQueuePriorities` holds `queueCount` floats.
    let mut priorities = Box::new(unsafe { std::slice::from_raw_parts(q.p_queue_priorities, q.queue_count as usize) }.to_vec());
    priorities.push(1.0);
    let mut extended = infos.to_vec();
    extended[at].queue_count = q.queue_count + 1;
    extended[at].p_queue_priorities = priorities.as_ptr();
    Some((q.queue_family_index, q.queue_count, extended, priorities))
}

/// Checks and clears the side queue [`NeuralForgeInstanceHooks::create_device`] added to `device`.
pub(crate) fn take_side_queue(device: vk::Device) -> Option<SideQueue> {
    SIDE_QUEUES.lock().unwrap().as_mut().and_then(|map| map.remove(&device))
}

/// The application's queue requests with one queue more for the layer, in a compute family without
/// graphics (NVIDIA's family 2): queues of the game's own graphics family share its GPU context,
/// and the layer's compute work there while the game's queue waits at the hold inside DLSS's buffer
/// faulted the channel (Xid 69, Crimson Desert, 2026-10-05). The application's request for that
/// family is extended by one when it has a queue to spare and is a plain one (no flags); without
/// one, a request for one queue is added. `priorities` backs the extended `pQueuePriorities`.
struct SideQueueRequest {
    infos: Vec<vk::DeviceQueueCreateInfo>,
    _priorities: Vec<f32>,
    queue: SideQueue,
}

fn side_queue_request(instance: &ash::Instance, physical_device: vk::PhysicalDevice, create_info: &vk::DeviceCreateInfo) -> Option<SideQueueRequest> {
    if create_info.queue_create_info_count == 0 || create_info.p_queue_create_infos.is_null() {
        return None;
    }
    // SAFETY: a non-null `pQueueCreateInfos` holds `queueCreateInfoCount` valid structures.
    let infos = unsafe { std::slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize) };
    // SAFETY: `physical_device` belongs to `instance`.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let app_family = infos
        .iter()
        .find(|q| families.get(q.queue_family_index as usize).is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)))?
        .queue_family_index;
    let family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS) && f.queue_count > 0)? as u32;
    let mut extended = infos.to_vec();
    let mut priorities: Vec<f32>;
    let index;
    match infos.iter().position(|q| q.queue_family_index == family) {
        Some(at) => {
            let q = infos[at];
            if !q.flags.is_empty() || q.queue_count >= families[family as usize].queue_count || q.p_queue_priorities.is_null() {
                return None;
            }
            // SAFETY: `pQueuePriorities` holds `queueCount` floats.
            priorities = unsafe { std::slice::from_raw_parts(q.p_queue_priorities, q.queue_count as usize) }.to_vec();
            priorities.push(1.0);
            extended[at].queue_count = q.queue_count + 1;
            extended[at].p_queue_priorities = priorities.as_ptr();
            index = q.queue_count;
        }
        None => {
            priorities = vec![1.0];
            extended.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&priorities).build());
            index = 0;
        }
    }
    Some(SideQueueRequest { infos: extended, _priorities: priorities, queue: SideQueue { family, index, app_family } })
}

/// Per-instance hooks, carrying that instance's own context. Dropped by the framework when
/// the application destroys the instance.
#[derive(Default)]
struct NeuralForgeInstanceHooks {
    ctx: Option<InstanceContext>,
}

impl NeuralForgeInstanceHooks {
    /// Records `physical_device` as belonging to this instance, for `create_device_info`.
    fn remember(&self, physical_device: vk::PhysicalDevice) {
        if let Some(ctx) = &self.ctx {
            PHYSICAL_DEVICE_INSTANCES.lock().unwrap().insert(physical_device, ctx.clone());
        }
    }
}

impl Drop for NeuralForgeInstanceHooks {
    fn drop(&mut self) {
        if let Some(ctx) = &self.ctx {
            let handle = ctx.instance.handle();
            PHYSICAL_DEVICE_INSTANCES.lock().unwrap().retain(|_, owner| owner.instance.handle() != handle);
        }
    }
}

impl InstanceHooks for NeuralForgeInstanceHooks {
    /// Adds [`EXTERNAL_MEMORY_HOST_EXTENSION`] to the game's own `vkCreateDevice` call
    /// when (and only when) it's safe to: the physical device actually advertises it
    /// and the app hasn't already requested it either way. Every other case returns
    /// [`LayerResult::Unhandled`], which hands the *unmodified* call straight to the
    /// framework's own default `create_device` path -- identical to this hook not
    /// existing at all. This function never changes device creation in any other way,
    /// and never makes a request that would otherwise have succeeded start failing:
    /// if creating the device with the extra extension is refused for any reason
    /// `vkEnumerateDeviceExtensionProperties` didn't predict, it retries with the
    /// exact, byte-identical original request before giving up.
    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
        layer_device_link: &VkLayerDeviceLink,
        allocator: Option<&vk::AllocationCallbacks>,
        p_device: &mut std::mem::MaybeUninit<vk::Device>,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        self.remember(physical_device);
        let Some(instance) = self.ctx.clone() else {
            return LayerResult::Unhandled;
        };
        // A `VkDeviceCreateInfo` requesting zero extensions may legitimately leave
        // `pp_enabled_extension_names` null (confirmed live: `vkcube` does exactly
        // this) -- `slice::from_raw_parts` requires a non-null pointer even for a
        // zero-length slice, unlike C's own more permissive "null + 0 means empty"
        // convention, so this has to be checked before it's ever dereferenced. Found
        // via `scripts/smoke-test.sh`'s debug-mode UB check aborting the process --
        // real, live UB in every release-mode run before this fix too, just silent.
        let requested: &[*const std::ffi::c_char] = if create_info.enabled_extension_count == 0 {
            &[]
        } else {
            // SAFETY: `create_info` is the framework's own, valid for this call;
            // `pp_enabled_extension_names` is a valid, non-null array of
            // `enabled_extension_count` C strings per its own contract as a
            // `VkDeviceCreateInfo`, `enabled_extension_count` just confirmed > 0.
            unsafe { std::slice::from_raw_parts(create_info.pp_enabled_extension_names, create_info.enabled_extension_count as usize) }
        };
        // SAFETY: resolved exactly like the framework's own default `create_device`
        // path resolves it (see `vulkan_layer::Global::create_device`) -- the next
        // layer/driver's real `vkCreateDevice`, through the instance this physical
        // device belongs to.
        let next_create_device: vk::PFN_vkCreateDevice = match unsafe {
            (layer_device_link.pfnNextGetInstanceProcAddr)(instance.instance.handle(), c"vkCreateDevice".as_ptr())
        } {
            // SAFETY: a non-null `vkGetInstanceProcAddr(instance, "vkCreateDevice")`
            // result is guaranteed by the Vulkan spec to have this exact signature.
            Some(f) => unsafe { std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkCreateDevice>(f) },
            None => return LayerResult::Unhandled,
        };
        let allocator_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // The hold inside DLSS's command buffer needs a queue of the layer's own beside the game's
        // (`preupscale::inline`): one more in the game's graphics family, on an NVIDIA device with
        // the pre-upscaler path on. A request with it that the driver refuses is made again without.
        // SAFETY: `physical_device` belongs to this instance.
        let nvidia = unsafe { instance.instance.get_physical_device_properties(physical_device) }.vendor_id == 0x10DE;
        let side = (nvidia && preupscale::active() && preupscale::inline::enabled())
            .then(|| side_queue_request(&instance.instance, physical_device, create_info))
            .flatten();
        // The native backend: the network's extensions and features and a loading queue of its own.
        // A request with them that the driver refuses is made again without (frames then go to DLSS
        // untouched in native mode, and the log says why).
        #[cfg(target_arch = "x86_64")]
        let native = nvidia
            && preupscale::active()
            && preupscale::native::backend() == preupscale::native::Backend::Native
            && preupscale::native::model_present();
        let create = |info: &vk::DeviceCreateInfo, p_device: &mut std::mem::MaybeUninit<vk::Device>| -> vk::Result {
            #[cfg(target_arch = "x86_64")]
            if native {
                let gipa = layer_device_link.pfnNextGetInstanceProcAddr;
                let base: Vec<vk::DeviceQueueCreateInfo> = match &side {
                    Some(sq) => sq.infos.clone(),
                    // SAFETY: `pQueueCreateInfos` holds `queueCreateInfoCount` valid structures (checked non-null).
                    None if !info.p_queue_create_infos.is_null() => unsafe { std::slice::from_raw_parts(info.p_queue_create_infos, info.queue_create_info_count as usize) }.to_vec(),
                    None => Vec::new(),
                };
                match native_queue_request(&instance.instance, physical_device, &base) {
                    None => log!("[native] no queue to spare in the game's graphics family for the network; native backend off on this device"),
                    Some((family, index, queues, _priorities)) => {
                        let mut with = *info;
                        with.queue_create_info_count = queues.len() as u32;
                        with.p_queue_create_infos = queues.as_ptr();
                        // SAFETY: the next layer's gipa for this instance; `with` and everything it
                        // points to outlive the extension, which outlives the call.
                        match unsafe { neural_forge_native::device_extend(gipa, instance.instance.handle(), physical_device, &with) } {
                            Err(why) => log!("[native] the device cannot run the network: {why}; native backend off on this device"),
                            Ok(extension) => {
                                // SAFETY: the extended request; `p_device` is the framework's out-parameter.
                                let result = unsafe { next_create_device(physical_device, &extension.info, allocator_ptr, p_device.as_mut_ptr()) };
                                drop(extension);
                                if result == vk::Result::SUCCESS {
                                    // SAFETY: written by the successful call.
                                    let device = unsafe { p_device.assume_init() };
                                    if let Some(sq) = &side {
                                        SIDE_QUEUES.lock().unwrap().get_or_insert_default().insert(device, sq.queue);
                                    }
                                    let setup = preupscale::native::Setup { gipa, instance: instance.instance.handle(), physical: physical_device, family, index };
                                    NATIVE_DEVICES.lock().unwrap().get_or_insert_default().insert(device, setup);
                                    log!("[native] device created with the network's extensions and features, loading queue {index} of family {family}");
                                    return result;
                                }
                                log!("[native] creating the device with the network's additions was refused ({result:?}); creating it without");
                            }
                        }
                    }
                }
            }
            if let Some(sq) = &side {
                let mut with = *info;
                with.queue_create_info_count = sq.infos.len() as u32;
                with.p_queue_create_infos = sq.infos.as_ptr();
                // SAFETY: `with` is the request with `sq`'s queue infos, which outlive the call;
                // `p_device` is the framework's out-parameter.
                let result = unsafe { next_create_device(physical_device, &with, allocator_ptr, p_device.as_mut_ptr()) };
                if result == vk::Result::SUCCESS {
                    // SAFETY: written by the successful call.
                    let device = unsafe { p_device.assume_init() };
                    SIDE_QUEUES.lock().unwrap().get_or_insert_default().insert(device, sq.queue);
                    return result;
                }
                log!("[preupscale] adding a queue for the hold inside DLSS's command buffer was refused ({result:?}); creating the device without it");
            }
            // SAFETY: the request as given; `p_device` as above.
            unsafe { next_create_device(physical_device, info, allocator_ptr, p_device.as_mut_ptr()) }
        };
        // The application already enables it (vkd3d-proton does): nothing to add, but the
        // device is created here, unmodified, so it can be recorded like one this hook
        // extended. `NeuralForgeDeviceInfo::new` must not look at the request's extension
        // list itself: on the framework's own (`Unhandled`) path that list has already been
        // freed by the time it is called, and reading it crashed the Rockstar launcher's GPU
        // process (0.1.96).
        if requests_extension(create_info, EXTERNAL_MEMORY_HOST_EXTENSION) {
            // `create_info` is the application's own request (with the side queue, if any).
            let result = create(create_info, p_device);
            if result.result().is_ok() {
                // SAFETY: `p_device` was just written by the successful call above.
                let device = unsafe { p_device.assume_init() };
                EXTERNAL_MEMORY_HOST_DEVICES.lock().unwrap().get_or_insert_default().insert(device);
            }
            return LayerResult::Handled(result.result());
        }
        // SAFETY: `physical_device` is the one this exact `vkCreateDevice` call is
        // for; `instance.instance` is its owning instance (these are this instance's
        // own hooks).
        let supported = unsafe { instance.instance.enumerate_device_extension_properties(physical_device) }
            .is_ok_and(|extensions| {
                extensions.iter().any(|extension| {
                    // SAFETY: `extension_name` is a NUL-terminated, driver-supplied C
                    // string per the Vulkan spec's own contract on this struct.
                    let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                    name == EXTERNAL_MEMORY_HOST_EXTENSION
                })
            });
        if !supported {
            #[cfg(target_arch = "x86_64")]
            let unhandled = side.is_none() && !native;
            #[cfg(not(target_arch = "x86_64"))]
            let unhandled = side.is_none();
            if unhandled {
                return LayerResult::Unhandled;
            }
            return LayerResult::Handled(create(create_info, p_device).result());
        }

        let mut names: Vec<*const std::ffi::c_char> = requested.to_vec();
        names.push(EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr());
        let mut extended = *create_info;
        extended.enabled_extension_count = names.len() as u32;
        extended.pp_enabled_extension_names = names.as_ptr();

        // SAFETY: `extended` borrows `names`, which outlives this call; `p_device` is
        // the framework's own out-parameter, valid for this call; `physical_device`
        // was validated above via a successful query against it.
        let result = create(&extended, p_device);
        if result.result().is_ok() {
            // SAFETY: `p_device` was just written by the successful call above.
            let device = unsafe { p_device.assume_init() };
            EXTERNAL_MEMORY_HOST_DEVICES.lock().unwrap().get_or_insert_default().insert(device);
            return LayerResult::Handled(Ok(()));
        }
        // The driver refused device creation with the extra extension for some reason
        // the earlier query didn't predict. Retry with the caller's own, completely
        // unmodified request -- this optimization attempt must never be the reason a
        // device creation that would otherwise have succeeded now fails.
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[layer] adding {EXTERNAL_MEMORY_HOST_EXTENSION:?} was refused ({result:?}); creating the device as requested, without zero-copy capture");
        }
        // Same as the call above, with `create_info` (the original request) this time.
        let result = create(create_info, p_device);
        LayerResult::Handled(result.result())
    }
}

impl InstanceInfo for NeuralForgeInstanceHooks {
    type HooksType = Self;
    type HooksRefType<'a> = &'a Self;

    fn hooked_commands() -> &'static [LayerVulkanCommand] {
        &[LayerVulkanCommand::CreateDevice]
    }

    fn hooks(&self) -> Self::HooksRefType<'_> {
        self
    }
}

#[derive(Default)]
struct NeuralForgeLayer(NeuralForgeGlobalHooks);

impl Layer for NeuralForgeLayer {
    type GlobalHooksInfo = NeuralForgeGlobalHooks;
    type InstanceInfo = NeuralForgeInstanceHooks;
    type DeviceInfo = NeuralForgeDeviceInfo;
    type InstanceInfoContainer = NeuralForgeInstanceHooks;
    type DeviceInfoContainer = NeuralForgeDeviceInfo;

    fn global_instance() -> impl Deref<Target = Global<Self>> + 'static {
        static GLOBAL: LazyLock<Global<NeuralForgeLayer>> = LazyLock::new(Default::default);
        &*GLOBAL
    }

    fn manifest() -> LayerManifest {
        let mut manifest = LayerManifest::default();
        manifest.name = LAYER_NAME;
        manifest.spec_version = vk::API_VERSION_1_1;
        manifest.implementation_version = 1;
        manifest.description = "NeuralForge neural rendering injection layer (Linux side)";
        manifest
    }

    fn global_hooks_info(&self) -> &Self::GlobalHooksInfo {
        &self.0
    }

    /// The framework's default (`DeviceInfo::hooked_commands`), plus the
    /// `NEURAL_FORGE_PROBE_NGX` probe's commands only when the probe is on, and the
    /// `NEURAL_FORGE_PREUPSCALE` tracking commands only when a mode is selected and, for a
    /// device's own table, only on a device that got the tracking (one with
    /// `VK_NVX_image_view_handle`). With both off (`NEURAL_FORGE_PREUPSCALE=off`) the list is exactly
    /// the default, so the framework hands out the next layer's pointers for every NVX entry point,
    /// as in 1.1.0; a device without NVX gets the default list under the default mode too.
    fn hooked_device_commands(
        &self,
        _instance_info: &Self::InstanceInfo,
        device_info: Option<&Self::DeviceInfo>,
    ) -> Box<dyn Iterator<Item = LayerVulkanCommand>> {
        device_commands(probe_ngx::enabled(), preupscale_hooks(preupscale::active(), device_info.map(NeuralForgeDeviceInfo::preupscale_tracked)))
    }

    fn create_instance_info(
        &self,
        _create_info: &vk::InstanceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        instance: Arc<ash::Instance>,
        next_get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    ) -> Self::InstanceInfoContainer {
        // Resolve below this layer: the physical-device handle supplied by the
        // framework belongs to that chain, not the loader's outer trampoline.
        let surface_caps = unsafe {
            next_get_instance_proc_addr(instance.handle(), c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR".as_ptr())
                .map(|p| std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(p))
        };
        NeuralForgeInstanceHooks { ctx: Some(InstanceContext { instance, surface_caps }) }
    }

    fn create_device_info(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self::DeviceInfoContainer {
        // The owning instance's `create_device` hook recorded it just before this call, so
        // this is always `Some` in practice; `None` only for a device created against an
        // instance from before this layer was loaded, which never happens for an implicit
        // layer, and then capture is simply skipped.
        let instance = PHYSICAL_DEVICE_INSTANCES.lock().unwrap().get(&physical_device).cloned();
        NeuralForgeDeviceInfo::new(instance.as_ref().map(|ctx| ctx.instance.clone()), instance.and_then(|ctx| ctx.surface_caps), physical_device, device, next_get_device_proc_addr, create_info)
    }
}

declare_introspection_queries!(entry_points::EntryPoints);

/// Whether the pre-upscaler path's commands are hooked: a mode is on and, for a device's own table
/// (`tracked` is `Some`), that device got the tracking. The instance-level table (`None`) follows
/// the mode alone; its hooks find no tracking on a device without it and forward the call.
fn preupscale_hooks(mode_on: bool, tracked: Option<bool>) -> bool {
    mode_on && tracked.unwrap_or(true)
}

/// The device commands the framework routes to this layer's hooks: the default set, then the
/// probe's and the pre-upscaler path's when they are on, each command once.
fn device_commands(probe: bool, preupscale: bool) -> Box<dyn Iterator<Item = LayerVulkanCommand>> {
    use vulkan_layer::DeviceInfo;
    let mut seen: Vec<LayerVulkanCommand> = Vec::new();
    for command in NeuralForgeDeviceInfo::hooked_commands().iter().chain(probe_ngx::probe_commands(probe)).chain(preupscale::commands(preupscale)) {
        if !seen.contains(command) {
            seen.push(command.clone());
        }
    }
    Box::new(seen.into_iter())
}

#[cfg(test)]
mod probe_command_tests {
    use super::*;
    use vulkan_layer::DeviceInfo;

    #[test]
    fn probe_off_hooks_exactly_the_default_set() {
        let off: Vec<_> = device_commands(false, false).collect();
        assert_eq!(off, NeuralForgeDeviceInfo::hooked_commands().to_vec());
        for command in probe_ngx::PROBE_COMMANDS {
            assert!(!off.contains(command), "{command:?} must not be hooked with the probe off");
        }
        let on: Vec<_> = device_commands(true, false).collect();
        assert_eq!(on.len(), off.len() + probe_ngx::PROBE_COMMANDS.len());
        assert!(probe_ngx::PROBE_COMMANDS.iter().all(|command| on.contains(command)));
        for command in preupscale::COMMANDS {
            assert!(!off.contains(command), "{command:?} must not be hooked with the pre-upscaler path off");
        }
        let pre: Vec<_> = device_commands(false, true).collect();
        assert_eq!(pre.len(), off.len() + preupscale::COMMANDS.len());
        assert!(preupscale::COMMANDS.iter().all(|command| pre.contains(command)));
        // Both: the union, each command once.
        let both: Vec<_> = device_commands(true, true).collect();
        let only_pre = preupscale::COMMANDS.iter().filter(|command| !probe_ngx::PROBE_COMMANDS.contains(command)).count();
        assert_eq!(both.len(), on.len() + only_pre);
        assert!(preupscale::COMMANDS.iter().all(|command| both.contains(command)));
        assert_eq!(both.iter().filter(|c| **c == LayerVulkanCommand::CmdBeginRendering).count(), 1, "each command once");
    }

    /// `NEURAL_FORGE_PREUPSCALE=off` hooks exactly 1.1.0's list everywhere; the default mode hooks
    /// the tracking commands only on a device that got the tracking (one with NVX).
    #[test]
    fn the_pre_upscaler_commands_follow_the_mode_and_the_device() {
        assert!(!preupscale_hooks(false, None));
        assert!(!preupscale_hooks(false, Some(true)));
        assert!(!preupscale_hooks(false, Some(false)));
        assert!(preupscale_hooks(true, None), "instance level: the mode alone");
        assert!(preupscale_hooks(true, Some(true)));
        assert!(!preupscale_hooks(true, Some(false)), "a device without NVX keeps the default list");
        assert_eq!(device_commands(false, preupscale_hooks(true, Some(false))).collect::<Vec<_>>(), NeuralForgeDeviceInfo::hooked_commands().to_vec());
        assert!(preupscale::wanted_on_device(true, true));
        assert!(!preupscale::wanted_on_device(true, false));
        assert!(!preupscale::wanted_on_device(false, true));
    }
}

#[cfg(test)]
mod requests_extension_tests {
    use super::*;

    #[test]
    fn finds_an_extension_the_application_requested_and_tolerates_a_null_list() {
        let other = c"VK_KHR_swapchain";
        let names = [other.as_ptr(), EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr()];
        let with = vk::DeviceCreateInfo { enabled_extension_count: 2, pp_enabled_extension_names: names.as_ptr(), ..Default::default() };
        assert!(requests_extension(&with, EXTERNAL_MEMORY_HOST_EXTENSION));
        let without = vk::DeviceCreateInfo { enabled_extension_count: 1, pp_enabled_extension_names: names.as_ptr(), ..Default::default() };
        assert!(!requests_extension(&without, EXTERNAL_MEMORY_HOST_EXTENSION));
        // Zero extensions with a null list, as `vkcube` passes it.
        assert!(!requests_extension(&vk::DeviceCreateInfo::default(), EXTERNAL_MEMORY_HOST_EXTENSION));
    }
}

#[cfg(test)]
mod instance_table_tests {
    use super::*;
    use ash::vk::Handle;

    /// A device is resolved through the instance that owns its physical device, not whichever
    /// instance was created last, and an instance's entries go when it is destroyed.
    #[test]
    fn devices_resolve_through_their_own_instance_and_forget_a_destroyed_one() {
        let Some(entry) = (unsafe { ash::Entry::load() }).ok() else {
            eprintln!("instance table test: no Vulkan loader, skipping");
            return;
        };
        let create = || unsafe { entry.create_instance(&vk::InstanceCreateInfo::builder(), None) }.ok().map(Arc::new);
        let (Some(real), Some(probe)) = (create(), create()) else {
            eprintln!("instance table test: no Vulkan ICD, skipping");
            return;
        };
        let hooks = |instance: &Arc<ash::Instance>| NeuralForgeInstanceHooks { ctx: Some(InstanceContext { instance: instance.clone(), surface_caps: None }) };
        // Stand-in handles: the table is keyed by whatever the framework hands over.
        let (real_gpu, probe_gpu) = (vk::PhysicalDevice::from_raw(0x5100), vk::PhysicalDevice::from_raw(0x5200));
        let real_hooks = hooks(&real);
        let probe_hooks = hooks(&probe);
        real_hooks.remember(real_gpu);
        probe_hooks.remember(probe_gpu);
        let owner = |gpu: vk::PhysicalDevice| PHYSICAL_DEVICE_INSTANCES.lock().unwrap().get(&gpu).map(|c| c.instance.handle());
        assert_eq!(owner(real_gpu), Some(real.handle()), "the real instance's device is not resolved through the later probe");
        assert_eq!(owner(probe_gpu), Some(probe.handle()));
        drop(probe_hooks);
        assert_eq!(owner(probe_gpu), None, "a destroyed instance's devices are forgotten");
        assert_eq!(owner(real_gpu), Some(real.handle()));
        drop(real_hooks);
        assert_eq!(owner(real_gpu), None);
        unsafe {
            probe.destroy_instance(None);
            real.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod duplicate_copy_tests {
    use super::*;

    #[test]
    fn other_copies_are_same_named_objects_at_other_paths() {
        let own = "/opt/nf/lib/neural-forge/libneural_forge_layer.so";
        let maps = "\
7f00-7f01 r--p 00000000 08:01 11 /opt/nf/lib/neural-forge/libneural_forge_layer.so
7f01-7f02 r-xp 00001000 08:01 11 /opt/nf/lib/neural-forge/libneural_forge_layer.so
7f10-7f11 r--p 00000000 08:01 22 /home/a/dev tree/target/release/libneural_forge_layer.so
7f11-7f12 r-xp 00001000 08:01 22 /home/a/dev tree/target/release/libneural_forge_layer.so
7f20-7f21 r--p 00000000 08:01 33 /usr/lib/libvulkan.so.1
7f30-7f31 rw-p 00000000 00:00 0
7f40-7f41 rw-p 00000000 00:00 0 [heap]
";
        assert_eq!(other_copies(maps, own), vec!["/home/a/dev tree/target/release/libneural_forge_layer.so".to_string()]);
        assert!(other_copies(maps, "/home/a/dev tree/target/release/libneural_forge_layer.so").contains(&own.to_string()));
    }

    #[test]
    fn an_object_that_is_not_loaded_is_not_live() {
        assert!(!copy_is_live("/nonexistent/libneural_forge_layer.so"));
    }
}

#[cfg(test)]
mod device_lost_tests {
    use super::*;

    #[test]
    fn note_vk_passes_results_through_and_latches_only_device_lost() {
        assert_eq!(note_vk::<()>(Ok(())), Ok(()));
        assert_eq!(note_vk::<()>(Err(vk::Result::ERROR_OUT_OF_DATE_KHR)), Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert!(!device_lost(), "an ordinary error must not latch the layer inert");
        assert_eq!(note_vk::<u32>(Err(vk::Result::ERROR_DEVICE_LOST)), Err(vk::Result::ERROR_DEVICE_LOST));
        assert!(device_lost());
        DEVICE_LOST.store(false, std::sync::atomic::Ordering::Relaxed); // other tests share the process
    }
}
