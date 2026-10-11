//! `VK_LAYER_neuralforge_neural` — the Linux-side Vulkan implicit layer.
//!
//! Hooks the swapchain lifecycle (`vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/
//! `vkQueuePresentKHR`), holds DLSS's input to run the model before the upscaler (the native
//! backend, `preupscale`), and after the upscaler exchanges frames with its in-process model server
//! over the shared-memory transport defined in `neural_forge_protocol`.
//!
//! Built on Google's [`vulkan_layer`](https://github.com/google/vk-layer-for-rust)
//! crate, which supplies the actual `vkGetInstanceProcAddr`/`vkGetDeviceProcAddr`
//! dispatch machinery, the `VkLayerInstanceCreateInfo`/`VkLayerDeviceCreateInfo`
//! chain-walk, and the loader-negotiation entry points — this crate only implements
//! [`vulkan_layer::DeviceHooks`] for the handful of functions it actually cares about;
//! everything else falls through to the next layer/driver automatically.
//!
//! Cross-process ownership and executable filtering guard the channel; the swapchain size
//! filter additionally excludes small overlays. See docs/HARDWARE_VALIDATION.md for
//! presentation-validation failures still under review.

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

// The native backend (crates/native), linked into the 64-bit layer (`preupscale::native`).
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
        crate::event!(neural_forge_protocol::state_log::DUPLICATE, "another copy is already loaded from {live}; this copy ({own}) stays inert");
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
        // Once, with the latch: the channel's `device_lost_at` and the state log say it without a log.
        crate::shm::note_device_lost();
        crate::event!(neural_forge_protocol::state_log::DEVICE_LOST, "VK_ERROR_DEVICE_LOST; the layer is inert from here on");
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
            crate::event!(
                neural_forge_protocol::state_log::FENCE_TIMEOUT,
                "fence wait timed out after {FENCE_WAIT_TIMEOUT:?} at {site}; stages, newest first: {}",
                crate::breadcrumbs::trail()
            );
        }
    }
    note_vk(result)
}

/// Why this process's device cannot run the network (`neural_forge_native::device_extend`'s refusal),
/// once one was refused: the Status tab's line (`layer_reason`, published when the channel opens,
/// `shm::ShmClient::open`) and the state log's. Set at device creation, never per frame.
static NATIVE_UNAVAILABLE: Mutex<Option<String>> = Mutex::new(None);

/// `device cannot run the network: <first missing>` from `device_extend`'s message ("the device lacks
/// X", or an exception's text).
fn cannot_run_line(why: &str) -> String {
    let first = why.strip_prefix("the device lacks ").unwrap_or(why);
    format!("{}{first}", neural_forge_protocol::state_log::CANNOT_RUN_PREFIX)
}

fn note_native_unavailable(why: &str) {
    let line = cannot_run_line(why);
    let mut said = NATIVE_UNAVAILABLE.lock().unwrap();
    // A launcher can create several devices on the same GPU: one line per different reason.
    if said.as_deref() != Some(line.as_str()) {
        crate::event!(neural_forge_protocol::state_log::NATIVE_UNAVAILABLE, "{line}");
        *said = Some(line);
    }
}

/// The status line for a device that cannot run the network, if this process created one.
pub(crate) fn native_unavailable() -> Option<String> {
    NATIVE_UNAVAILABLE.lock().unwrap().clone()
}

fn env_flag(name: &str) -> bool {
    neural_forge_protocol::env::flag(name)
}

/// Works around a crash inside Mesa's `device_select` implicit layer, confirmed
/// reproducible even with `vulkan-layer`'s own pristine `hello-world` example under
/// implicit activation on this machine's Mesa build (see the "CRITICAL,
/// confirmed" section of docs/history/development-before-neuralforge.md for the full bisection) -- so this is a bug in the interaction
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

/// The extension the layer adds to a game's own device creation for the zero-copy capture path (`docs/ASYNC_CAPTURE_DESIGN.md`): importing the SHM
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

/// Checks and clears the side queue [`NeuralForgeInstanceHooks::create_device`] added to `device`.
pub(crate) fn take_side_queue(device: vk::Device) -> Option<SideQueue> {
    SIDE_QUEUES.lock().unwrap().as_mut().and_then(|map| map.remove(&device))
}

/// The application's queue requests `infos` with `n` more queues for the layer, in a compute family without
/// graphics (NVIDIA's family 2): queues of the game's own graphics family share its GPU context, and the
/// layer's work there faulted the channel (Xid 69 in Crimson Desert with the hold inside DLSS's buffer,
/// 2026-10-05; Xid 69 and Xid 32 in GTA V with the network's uploads, 2026-10-07). Every such family is
/// considered, in order, and the first with room for `n` more is taken: the application's request for it is
/// extended when it is a plain one (no flags), or a request for `n` queues is added.
struct ExtraQueues {
    /// The first family the application requests with graphics and compute: where DLSS runs.
    app_family: u32,
    family: u32,
    /// The first of the new queues' indices.
    index: u32,
    infos: Vec<vk::DeviceQueueCreateInfo>,
    /// Backs the extended `pQueuePriorities`.
    _priorities: Vec<f32>,
}

impl ExtraQueues {
    fn side_queue(&self) -> SideQueue {
        SideQueue { family: self.family, index: self.index, app_family: self.app_family }
    }
}

fn extra_queues(families: &[vk::QueueFamilyProperties], infos: &[vk::DeviceQueueCreateInfo], n: u32) -> Option<ExtraQueues> {
    let app_family = infos
        .iter()
        .find(|q| families.get(q.queue_family_index as usize).is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)))?
        .queue_family_index;
    for (family, props) in families.iter().enumerate() {
        if !props.queue_flags.contains(vk::QueueFlags::COMPUTE) || props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            continue;
        }
        let family = family as u32;
        let mut extended = infos.to_vec();
        let (priorities, index) = match infos.iter().position(|q| q.queue_family_index == family) {
            Some(at) => {
                let q = infos[at];
                if !q.flags.is_empty() || q.queue_count + n > props.queue_count || q.p_queue_priorities.is_null() {
                    continue;
                }
                // SAFETY: `pQueuePriorities` holds `queueCount` floats.
                let mut priorities = unsafe { std::slice::from_raw_parts(q.p_queue_priorities, q.queue_count as usize) }.to_vec();
                priorities.extend(std::iter::repeat_n(1.0, n as usize));
                extended[at].queue_count = q.queue_count + n;
                extended[at].p_queue_priorities = priorities.as_ptr();
                (priorities, q.queue_count)
            }
            None => {
                if props.queue_count < n {
                    continue;
                }
                let priorities = vec![1.0; n as usize];
                extended.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&priorities).build());
                (priorities, 0)
            }
        };
        return Some(ExtraQueues { app_family, family, index, infos: extended, _priorities: priorities });
    }
    None
}

/// `create_info`'s queue requests, and the physical device's families.
fn queue_requests<'a>(instance: &ash::Instance, physical_device: vk::PhysicalDevice, create_info: &'a vk::DeviceCreateInfo) -> (&'a [vk::DeviceQueueCreateInfo], Vec<vk::QueueFamilyProperties>) {
    let infos = if create_info.queue_create_info_count == 0 || create_info.p_queue_create_infos.is_null() {
        &[][..]
    } else {
        // SAFETY: a non-null `pQueueCreateInfos` holds `queueCreateInfoCount` valid structures.
        unsafe { std::slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize) }
    };
    // SAFETY: `physical_device` belongs to `instance`.
    (infos, unsafe { instance.get_physical_device_queue_family_properties(physical_device) })
}

/// The loader's `VkLayerDeviceCreateInfo` (`vk_layer.h`), which `vulkan_layer` does not export:
/// the `VK_LAYER_LINK_INFO` node of a `VkDeviceCreateInfo`'s chain.
#[repr(C)]
struct LoaderDeviceCreateInfo {
    s_type: vk::StructureType,
    p_next: *const std::ffi::c_void,
    function: i32,
    layer_info: *mut VkLayerDeviceLink,
}

/// `VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO` and `VK_LAYER_LINK_INFO`.
const LOADER_DEVICE_CREATE_INFO: vk::StructureType = vk::StructureType::from_raw(48);
const LAYER_LINK_INFO: i32 = 0;

/// The link node of a `vkCreateDevice` request and the `pLayerInfo` the framework left in it for
/// the layer below. Every layer below advances `pLayerInfo` in place on each `vkCreateDevice` it
/// receives, so when one application call is tried more than once (with the layer's additions,
/// then without), each attempt must start from this value again, or the layer below reads its own
/// successor's link (a crash, or a layer skipped).
struct DeviceLink {
    node: *mut LoaderDeviceCreateInfo,
    info: *mut VkLayerDeviceLink,
}

impl DeviceLink {
    /// # Safety
    /// `create_info`'s chain is valid and its link node writable (the loader's own).
    unsafe fn find(create_info: &vk::DeviceCreateInfo) -> Option<Self> {
        let mut next = create_info.p_next.cast::<LoaderDeviceCreateInfo>();
        while !next.is_null() {
            // SAFETY: every node of a valid chain starts with `sType` and `pNext`; the fields past
            // them are read only on the loader's own node.
            let node = unsafe { &*next };
            if node.s_type == LOADER_DEVICE_CREATE_INFO && node.function == LAYER_LINK_INFO {
                return Some(Self { node: next.cast_mut(), info: node.layer_info });
            }
            next = node.p_next.cast();
        }
        None
    }

    fn restore(&self) {
        // SAFETY: the node `find` found, alive for the `vkCreateDevice` call this is made in.
        unsafe { (*self.node).layer_info = self.info };
    }
}

/// Calls the next layer's `vkCreateDevice` with `link` (if the request has one) put back first: the
/// one way this layer's device creation reaches the layer below, however many attempts it makes.
///
/// # Safety
/// As `vkCreateDevice`; `link` was found in `info`'s chain.
unsafe fn call_next_create_device(
    next: vk::PFN_vkCreateDevice, link: Option<&DeviceLink>, physical_device: vk::PhysicalDevice, info: &vk::DeviceCreateInfo,
    allocator: *const vk::AllocationCallbacks, p_device: &mut std::mem::MaybeUninit<vk::Device>,
) -> vk::Result {
    if let Some(link) = link {
        link.restore();
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { next(physical_device, info, allocator, p_device.as_mut_ptr()) }
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
    /// In a process the layer is on in and the target filter admits, adds what the layer needs to
    /// the game's own `vkCreateDevice` call: [`EXTERNAL_MEMORY_HOST_EXTENSION`] when the physical
    /// device advertises it, the side queue for the hold inside DLSS's command buffer, and the
    /// native backend's extensions, features and queues. Every other case returns
    /// [`LayerResult::Unhandled`], which hands the *unmodified* call straight to the framework's
    /// own default `create_device` path -- identical to this hook not existing at all. It never
    /// makes a request that would otherwise have succeeded start failing: an addition the driver
    /// refuses is dropped and the request made again, down to the application's own, every attempt
    /// through [`call_next_create_device`].
    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
        layer_device_link: &VkLayerDeviceLink,
        allocator: Option<&vk::AllocationCallbacks>,
        p_device: &mut std::mem::MaybeUninit<vk::Device>,
    ) -> LayerResult<ash::prelude::VkResult<()>> {
        self.remember(physical_device);
        // A process the layer is off in, or one the target filter excludes (launchers, overlays),
        // gets its device exactly as it asked for it: no queues, features, extensions or worker.
        if !(layer_enabled() && ownership::eligible()) {
            return LayerResult::Unhandled;
        }
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
        // SAFETY: the framework's copy of the request, whose chain is the loader's.
        let link = unsafe { DeviceLink::find(create_info) };
        // SAFETY: `vkCreateDevice` for this call's physical device, with the request each attempt
        // passes and the framework's out-parameter.
        let next_create_device = |info: &vk::DeviceCreateInfo, p_device: &mut std::mem::MaybeUninit<vk::Device>| unsafe {
            call_next_create_device(next_create_device, link.as_ref(), physical_device, info, allocator_ptr, p_device)
        };
        // The hold inside DLSS's command buffer needs a queue of the layer's own beside the game's
        // (`preupscale::inline`): one more in a compute family without graphics ([`extra_queues`]), on
        // an NVIDIA device with the pre-upscaler path on. A request with it that the driver refuses is
        // made again without.
        // SAFETY: `physical_device` belongs to this instance.
        let nvidia = unsafe { instance.instance.get_physical_device_properties(physical_device) }.vendor_id == 0x10DE;
        if !create_info.p_queue_create_infos.is_null() {
            // SAFETY: `pQueueCreateInfos` holds `queueCreateInfoCount` valid structures (checked non-null).
            let asked: Vec<String> = unsafe { std::slice::from_raw_parts(create_info.p_queue_create_infos, create_info.queue_create_info_count as usize) }
                .iter()
                .map(|q| format!("family {} x{}", q.queue_family_index, q.queue_count))
                .collect();
            log!("[queues] the application asks for {}", asked.join(", "));
        }
        let (app_queues, families) = queue_requests(&instance.instance, physical_device, create_info);
        // The hold inside DLSS's buffer: one queue of the layer's own.
        let side = (nvidia && preupscale::active() && preupscale::inline::enabled()).then(|| extra_queues(&families, app_queues, 1)).flatten();
        // The native backend: the network's extensions and features and a loading queue of its own.
        // A request with them that the driver refuses is made again without (frames then go to DLSS
        // untouched in native mode, and the log says why).
        #[cfg(target_arch = "x86_64")]
        let native = nvidia && preupscale::active();
        let create = |info: &vk::DeviceCreateInfo, p_device: &mut std::mem::MaybeUninit<vk::Device>| -> vk::Result {
            #[cfg(target_arch = "x86_64")]
            if native {
                let gipa = layer_device_link.pfnNextGetInstanceProcAddr;
                // Two more (the network's loading, and the after-the-upscaler path's frames), on top of the side queue.
                let base = side.as_ref().map_or(app_queues, |sq| &sq.infos[..]);
                match extra_queues(&families, base, 2) {
                    None => log!("[native] no compute queue to spare for the network's loading; native backend off on this device"),
                    Some(ExtraQueues { app_family: frame_family, family, index, infos: queues, .. }) => {
                        let mut with = *info;
                        with.queue_create_info_count = queues.len() as u32;
                        with.p_queue_create_infos = queues.as_ptr();
                        // SAFETY: the next layer's gipa for this instance; `with` and everything it
                        // points to outlive the extension, which outlives the call.
                        match unsafe { neural_forge_native::device_extend(gipa, instance.instance.handle(), physical_device, &with) } {
                            Err(why) => {
                                log!("[native] the device cannot run the network: {why}; native backend off on this device");
                                note_native_unavailable(&why);
                            }
                            Ok(extension) => {
                                let result = next_create_device(&extension.info, p_device);
                                drop(extension);
                                if result == vk::Result::SUCCESS {
                                    // SAFETY: written by the successful call.
                                    let device = unsafe { p_device.assume_init() };
                                    if let Some(sq) = &side {
                                        SIDE_QUEUES.lock().unwrap().get_or_insert_default().insert(device, sq.side_queue());
                                    }
                                    let setup = preupscale::native::Setup { gipa, instance: instance.instance.handle(), physical: physical_device, frame_family, family, index, post_index: index + 1 };
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
                // `with` is the request with `sq`'s queue infos, which outlive the call.
                let result = next_create_device(&with, p_device);
                if result == vk::Result::SUCCESS {
                    // SAFETY: written by the successful call.
                    let device = unsafe { p_device.assume_init() };
                    SIDE_QUEUES.lock().unwrap().get_or_insert_default().insert(device, sq.side_queue());
                    return result;
                }
                log!("[preupscale] adding a queue for the hold inside DLSS's command buffer was refused ({result:?}); creating the device without it");
            }
            next_create_device(info, p_device)
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

    #[test]
    fn the_cannot_run_line_names_the_first_missing_requirement() {
        assert_eq!(cannot_run_line("the device lacks VK_EXT_shader_float8"), "device cannot run the network: VK_EXT_shader_float8");
        assert_eq!(cannot_run_line("Vulkan 1.3 (the device reports 1.2)"), "device cannot run the network: Vulkan 1.3 (the device reports 1.2)");
        note_native_unavailable("the device lacks shaderInt64");
        assert_eq!(native_unavailable().as_deref(), Some("device cannot run the network: shaderInt64"));
        *NATIVE_UNAVAILABLE.lock().unwrap() = None;
    }
}

#[cfg(test)]
mod device_link_tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::ffi::c_char;

    thread_local! {
        static SEEN: RefCell<Vec<*mut VkLayerDeviceLink>> = const { RefCell::new(Vec::new()) };
        static REFUSE: Cell<u32> = const { Cell::new(0) };
        static CALLED: Cell<u32> = const { Cell::new(0) };
    }

    unsafe extern "system" fn gipa(_: vk::Instance, _: *const c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    unsafe extern "system" fn gdpa(_: vk::Device, _: *const c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    /// The layer below, as a real one behaves: it takes its link and advances `pLayerInfo` in place
    /// for the layer under it, then refuses the first `REFUSE` requests.
    unsafe extern "system" fn next_layer(
        _: vk::PhysicalDevice, info: *const vk::DeviceCreateInfo, _: *const vk::AllocationCallbacks, _: *mut vk::Device,
    ) -> vk::Result {
        CALLED.set(CALLED.get() + 1);
        // SAFETY: the test's own request. Without a link node (a request the hook made itself, in
        // the ineligible-process test) it is refused; unwinding out of here would abort.
        let Some(link) = (unsafe { DeviceLink::find(&*info) }) else {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        };
        // SAFETY: the test's node and links, alive for the call.
        unsafe {
            let node = &mut *link.node;
            SEEN.with_borrow_mut(|seen| seen.push(node.layer_info));
            if !node.layer_info.is_null() {
                node.layer_info = (*node.layer_info).pNext;
            }
        }
        let refuse = REFUSE.get();
        if refuse > 0 {
            REFUSE.set(refuse - 1);
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        vk::Result::SUCCESS
    }

    /// Every attempt of one application call reaches the layer below with the link the framework
    /// left for it, however many attempts were refused before it.
    #[test]
    fn every_attempt_sees_the_original_link() {
        let mut lower = VkLayerDeviceLink { pNext: std::ptr::null_mut(), pfnNextGetInstanceProcAddr: gipa, pfnNextGetDeviceProcAddr: gdpa };
        let mut below = VkLayerDeviceLink { pNext: &mut lower, pfnNextGetInstanceProcAddr: gipa, pfnNextGetDeviceProcAddr: gdpa };
        let original: *mut VkLayerDeviceLink = &mut below;
        let mut node = LoaderDeviceCreateInfo { s_type: LOADER_DEVICE_CREATE_INFO, p_next: std::ptr::null(), function: LAYER_LINK_INFO, layer_info: original };
        // Another structure ahead of the link node, as applications chain features.
        let mut features = vk::PhysicalDeviceFeatures2 { p_next: std::ptr::from_mut(&mut node).cast(), ..Default::default() };
        let info = vk::DeviceCreateInfo { p_next: std::ptr::from_mut(&mut features).cast(), ..Default::default() };
        // SAFETY: the chain above.
        let link = unsafe { DeviceLink::find(&info) }.expect("found past the features");
        assert_eq!(link.info, original);
        REFUSE.set(5);
        let mut p_device = std::mem::MaybeUninit::uninit();
        // Six attempts, as `create_device` can make: the first five refused, each with its own copy
        // of the request (the additions differ, the chain is the application's).
        let mut results = Vec::new();
        for _ in 0..6 {
            let with = info;
            // SAFETY: the fake layer below; the request and link above.
            results.push(unsafe { call_next_create_device(next_layer, Some(&link), vk::PhysicalDevice::null(), &with, std::ptr::null(), &mut p_device) });
        }
        assert_eq!(results.last(), Some(&vk::Result::SUCCESS));
        SEEN.with_borrow(|seen| {
            assert_eq!(seen.len(), 6);
            assert!(seen.iter().all(|&l| l == original), "an attempt saw an advanced link: {seen:?} (original {original:?})");
        });
    }

    /// In a process the layer is off in (`NEURAL_FORGE_ENABLE` is unset in tests), the hook hands the
    /// request to the framework untouched and never calls the layer below itself.
    #[test]
    fn an_ineligible_process_gets_its_device_unmodified() {
        let Some(entry) = (unsafe { ash::Entry::load() }).ok() else {
            eprintln!("ineligible device test: no Vulkan loader, skipping");
            return;
        };
        let Some(instance) = (unsafe { entry.create_instance(&vk::InstanceCreateInfo::builder(), None) }).ok().map(Arc::new) else {
            eprintln!("ineligible device test: no Vulkan ICD, skipping");
            return;
        };
        let Some(&physical) = unsafe { instance.enumerate_physical_devices() }.ok().as_ref().and_then(|d| d.first()) else {
            eprintln!("ineligible device test: no physical device, skipping");
            unsafe { instance.destroy_instance(None) };
            return;
        };
        assert!(!layer_enabled(), "tests run with the layer off");
        unsafe extern "system" fn gipa_next(_: vk::Instance, name: *const c_char) -> vk::PFN_vkVoidFunction {
            // SAFETY: a NUL-terminated name from the hook.
            (unsafe { CStr::from_ptr(name) } == c"vkCreateDevice")
                // SAFETY: the same signature as `vkCreateDevice`, which the caller transmutes back.
                .then(|| unsafe { std::mem::transmute::<vk::PFN_vkCreateDevice, unsafe extern "system" fn()>(next_layer) })
        }
        let link = VkLayerDeviceLink { pNext: std::ptr::null_mut(), pfnNextGetInstanceProcAddr: gipa_next, pfnNextGetDeviceProcAddr: gdpa };
        let priorities = [1.0];
        let queue = vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&priorities).build();
        let info = vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue)).build();
        let hooks = NeuralForgeInstanceHooks { ctx: Some(InstanceContext { instance: instance.clone(), surface_caps: None }) };
        let mut p_device = std::mem::MaybeUninit::uninit();
        CALLED.set(0);
        let result = hooks.create_device(physical, &info, &link, None, &mut p_device);
        assert!(matches!(result, LayerResult::Unhandled), "the framework forwards the application's own request");
        assert_eq!(CALLED.get(), 0, "the hook made no request of its own");
        drop(hooks);
        unsafe { instance.destroy_instance(None) };
    }
}

#[cfg(test)]
mod extra_queue_tests {
    use super::*;

    fn family(flags: vk::QueueFlags, queue_count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties { queue_flags: flags, queue_count, ..Default::default() }
    }

    const GRAPHICS: vk::QueueFlags = vk::QueueFlags::from_raw(vk::QueueFlags::GRAPHICS.as_raw() | vk::QueueFlags::COMPUTE.as_raw() | vk::QueueFlags::TRANSFER.as_raw());
    const COMPUTE: vk::QueueFlags = vk::QueueFlags::from_raw(vk::QueueFlags::COMPUTE.as_raw() | vk::QueueFlags::TRANSFER.as_raw());

    /// The first compute-only family is full (the application asks for all its queues): the next one with room
    /// is taken, not none.
    #[test]
    fn a_full_compute_family_is_passed_over_for_one_with_room() {
        let families = [family(GRAPHICS, 16), family(COMPUTE, 2), family(vk::QueueFlags::TRANSFER, 2), family(COMPUTE, 8)];
        let priorities = [1.0f32; 2];
        let app = [
            vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&priorities[..1]).build(),
            vk::DeviceQueueCreateInfo::builder().queue_family_index(1).queue_priorities(&priorities).build(),
        ];
        let q = extra_queues(&families, &app, 2).expect("family 3 has room");
        assert_eq!((q.app_family, q.family, q.index), (0, 3, 0));
        assert_eq!(q.infos.len(), 3);
        assert_eq!((q.infos[2].queue_family_index, q.infos[2].queue_count), (3, 2));
        // One more in family 1 would fit only without the application's two.
        assert!(extra_queues(&families[..3], &app, 1).is_none());
        // The application's own request for the family is extended when it has room.
        let one = [app[0], vk::DeviceQueueCreateInfo::builder().queue_family_index(1).queue_priorities(&priorities[..1]).build()];
        let q = extra_queues(&families, &one, 1).expect("family 1 has one to spare");
        assert_eq!((q.family, q.index, q.infos.len(), q.infos[1].queue_count), (1, 1, 2, 2));
        // SAFETY: the extended priorities hold the new count.
        assert_eq!(unsafe { std::slice::from_raw_parts(q.infos[1].p_queue_priorities, 2) }, &[1.0, 1.0]);
        // No graphics family requested: nothing to add beside.
        assert!(extra_queues(&families, &app[1..], 1).is_none());
    }
}
