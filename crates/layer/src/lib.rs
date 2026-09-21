//! `VK_LAYER_neuralforge_neural` — the Linux-side Vulkan implicit layer.
//!
//! Hooks the swapchain lifecycle (`vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/
//! `vkQueuePresentKHR`) and exchanges frames with the helper over the shared-memory
//! transport defined in `neuralforge_protocol`. This crate never knows or cares whether the
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

mod capture;
mod optical_flow;
mod composition;
mod device;
mod ownership;
mod loader_data;
mod dump;
mod logging;
mod hotkey;
mod shm;
mod swapchain;
mod surface_usage;
mod entry_points;
mod present_sync;

use std::collections::HashSet;
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

/// The most recently created `VkInstance`, so `create_device_info` (which the
/// `vulkan_layer` framework calls with no way to reach whatever `create_instance_info`
/// returned -- see that method's own doc comment) can still get an `ash::Instance` to
/// query physical-device memory properties from when it builds capture resources.
/// Games overwhelmingly create exactly one `VkInstance`; a plain "last one wins" slot
/// is the same simplification `device::PRIMARY` already makes for the analogous
/// one-swapchain-at-a-time assumption.
#[derive(Clone)]
struct InstanceContext {
    instance: Arc<ash::Instance>,
    surface_caps: Option<vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>,
}
static CURRENT_INSTANCE: Mutex<Option<InstanceContext>> = Mutex::new(None);

pub const LAYER_NAME: &str = "VK_LAYER_neuralforge_neural";

/// Whether the pass should do anything at all. Off by default (`NEURALFORGE_ENABLE` unset)
/// so the layer is a true no-op for every game that hasn't opted in via its launch
/// options — checked once and cached, same as upstream, since it can't change for the
/// life of the process.
pub(crate) fn layer_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        env_flag("NEURALFORGE_ENABLE") && !env_flag("NEURALFORGE_DISABLE")
    });
    *ENABLED
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
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
            Some(fp) => unsafe { std::mem::transmute(fp) },
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
/// once. Same pattern as `device::CLEANUP`/`CURRENT_INSTANCE`: a small, short-lived,
/// mutex-guarded side table, not a source of truth kept around indefinitely.
static EXTERNAL_MEMORY_HOST_DEVICES: Mutex<Option<HashSet<vk::Device>>> = Mutex::new(None);

/// Checks and clears whether `device` is one [`NeuralForgeInstanceHooks::create_device`]
/// added [`EXTERNAL_MEMORY_HOST_EXTENSION`] to.
pub(crate) fn take_external_memory_host_enabled(device: vk::Device) -> bool {
    EXTERNAL_MEMORY_HOST_DEVICES.lock().unwrap().as_mut().is_some_and(|set| set.remove(&device))
}

#[derive(Default)]
struct NeuralForgeInstanceHooks;

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
        let Some(instance) = CURRENT_INSTANCE.lock().unwrap().clone() else {
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
        // SAFETY: every element of `requested` is a valid, NUL-terminated C string for
        // the same reason as the slice itself.
        if requested.iter().any(|&name| unsafe { CStr::from_ptr(name) } == EXTERNAL_MEMORY_HOST_EXTENSION) {
            return LayerResult::Unhandled;
        }
        // SAFETY: `physical_device` is the one this exact `vkCreateDevice` call is
        // for; `instance.instance` is its owning instance (the only kind
        // `CURRENT_INSTANCE` ever stores).
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
            return LayerResult::Unhandled;
        }

        let mut names: Vec<*const std::ffi::c_char> = requested.to_vec();
        names.push(EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr());
        let mut extended = *create_info;
        extended.enabled_extension_count = names.len() as u32;
        extended.pp_enabled_extension_names = names.as_ptr();

        // SAFETY: resolved exactly like the framework's own default `create_device`
        // path resolves it (see `vulkan_layer::Global::create_device`) -- the next
        // layer/driver's real `vkCreateDevice`, through the instance this physical
        // device belongs to.
        let next_create_device: vk::PFN_vkCreateDevice = match unsafe {
            (layer_device_link.pfnNextGetInstanceProcAddr)(instance.instance.handle(), c"vkCreateDevice".as_ptr())
        } {
            // SAFETY: a non-null `vkGetInstanceProcAddr(instance, "vkCreateDevice")`
            // result is guaranteed by the Vulkan spec to have this exact signature.
            Some(f) => unsafe { std::mem::transmute(f) },
            None => return LayerResult::Unhandled,
        };
        let allocator_ptr = allocator.map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: `extended` borrows `names`, which outlives this call; `p_device` is
        // the framework's own out-parameter, valid for this call; `physical_device`
        // was validated above via a successful query against it.
        let result = unsafe { next_create_device(physical_device, &extended, allocator_ptr, p_device.as_mut_ptr()) };
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
        // SAFETY: same reasoning as the call above, with `create_info` (the original,
        // borrowed, unmodified request) this time.
        let result = unsafe { next_create_device(physical_device, create_info, allocator_ptr, p_device.as_mut_ptr()) };
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
        *CURRENT_INSTANCE.lock().unwrap() = Some(InstanceContext { instance, surface_caps });
        Default::default()
    }

    fn create_device_info(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
        _allocator: Option<&vk::AllocationCallbacks>,
        device: Arc<ash::Device>,
        next_get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    ) -> Self::DeviceInfoContainer {
        // `create_instance_info` always runs before `create_device_info` for the
        // instance a device is created against (the app must call `vkCreateInstance`
        // before `vkCreateDevice`), so this is always `Some` in practice; `unwrap_or`
        // only matters for a hypothetical device created against an instance from
        // before this layer was loaded, which never happens for an implicit layer.
        let instance = CURRENT_INSTANCE.lock().unwrap().clone();
        NeuralForgeDeviceInfo::new(instance.as_ref().map(|ctx| ctx.instance.clone()), instance.and_then(|ctx| ctx.surface_caps), physical_device, device, next_get_device_proc_addr, create_info)
    }
}

declare_introspection_queries!(entry_points::EntryPoints);
