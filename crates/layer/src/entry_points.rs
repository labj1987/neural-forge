//! Intercept destruction before the framework destroys the downstream device.
//! All other dispatch and loader bookkeeping remains in the pinned framework.
use ash::vk;
use std::ffi::{c_char, CStr};
use vulkan_layer::Global;
type Framework = Global<crate::NeuralForgeLayer>;
pub struct EntryPoints;

impl EntryPoints {
    pub unsafe extern "system" fn enumerate_instance_layer_properties(count: *mut u32, props: *mut vk::LayerProperties) -> vk::Result {
        unsafe { Framework::enumerate_instance_layer_properties(count, props) }
    }
    pub unsafe extern "system" fn enumerate_instance_extension_properties(name: *const c_char, count: *mut u32, props: *mut vk::ExtensionProperties) -> vk::Result {
        unsafe { Framework::enumerate_instance_extension_properties(name, count, props) }
    }
    pub unsafe extern "system" fn enumerate_device_layer_properties(pd: vk::PhysicalDevice, count: *mut u32, props: *mut vk::LayerProperties) -> vk::Result {
        unsafe { Framework::enumerate_device_layer_properties(pd, count, props) }
    }
    pub unsafe extern "system" fn enumerate_device_extension_properties(pd: vk::PhysicalDevice, name: *const c_char, count: *mut u32, props: *mut vk::ExtensionProperties) -> vk::Result {
        unsafe { Framework::enumerate_device_extension_properties(pd, name, count, props) }
    }
    pub unsafe extern "system" fn get_instance_proc_addr(instance: vk::Instance, name: *const c_char) -> vk::PFN_vkVoidFunction {
        let original = unsafe { Framework::get_instance_proc_addr(instance, name) };
        unsafe { Self::intercept(name, original) }
    }
    pub unsafe extern "system" fn get_device_proc_addr(device: vk::Device, name: *const c_char) -> vk::PFN_vkVoidFunction {
        let original = unsafe { Framework::get_device_proc_addr(device, name) };
        let original = unsafe { Self::intercept_probe(device, name, original, crate::probe_ngx::enabled() || crate::preupscale::active()) };
        unsafe { Self::intercept(name, original) }
    }
    /// `vkGetImageViewHandle64NVX` is newer than the pinned ash and `vulkan-layer`, so the
    /// framework hands out the next layer's pointer for it. With the `NEURAL_FORGE_PROBE_NGX`
    /// probe or a `NEURAL_FORGE_PREUPSCALE` mode on (and only then), wrap that pointer; a null
    /// result (extension absent) stays null.
    unsafe fn intercept_probe(device: vk::Device, name: *const c_char, original: vk::PFN_vkVoidFunction, probe: bool) -> vk::PFN_vkVoidFunction {
        let Some(next) = original.filter(|_| probe) else { return original };
        if unsafe { CStr::from_ptr(name) } != crate::probe_ngx::HANDLE64_NAME {
            return original;
        }
        // SAFETY: a non-null `vkGetDeviceProcAddr(device, "vkGetImageViewHandle64NVX")` has
        // this signature.
        crate::probe_ngx::remember_handle64(device, unsafe { std::mem::transmute::<unsafe extern "system" fn(), crate::probe_ngx::PfnGetImageViewHandle64Nvx>(next) });
        Some(unsafe { std::mem::transmute::<crate::probe_ngx::PfnGetImageViewHandle64Nvx, unsafe extern "system" fn()>(Self::get_image_view_handle64_nvx) })
    }
    unsafe extern "system" fn get_image_view_handle64_nvx(device: vk::Device, info: *const vk::ImageViewHandleInfoNVX) -> u64 {
        // Recorded when this wrapper was handed out for `device`; if it somehow was not, ask
        // the framework again (it forwards unknown names to the next layer).
        let next = crate::probe_ngx::handle64_next(device).or_else(|| {
            let p = unsafe { Framework::get_device_proc_addr(device, crate::probe_ngx::HANDLE64_NAME.as_ptr()) }?;
            Some(unsafe { std::mem::transmute::<unsafe extern "system" fn(), crate::probe_ngx::PfnGetImageViewHandle64Nvx>(p) })
        });
        let Some(next) = next else { return 0 };
        let handle = unsafe { next(device, info) };
        if let Some(info) = unsafe { info.as_ref() } {
            if crate::probe_ngx::enabled() {
                crate::probe_ngx::on_view_handle("vkGetImageViewHandle64NVX", info.image_view, format!("handle {handle:#x}"));
            }
            if let Some(tracking) = crate::preupscale::tracking_for(device) {
                tracking.lock().register(info.image_view);
            }
        }
        handle
    }
    unsafe fn intercept(name: *const c_char, original: vk::PFN_vkVoidFunction) -> vk::PFN_vkVoidFunction {
        // Preserve null results (including invalid instance/device queries). Keep
        // subsequent proc-address queries routed through this same interception.
        original?;
        match unsafe { CStr::from_ptr(name) }.to_bytes() {
            b"vkDestroyDevice" => Some(unsafe { std::mem::transmute::<vk::PFN_vkDestroyDevice, unsafe extern "system" fn()>(Self::destroy_device) }),
            b"vkGetInstanceProcAddr" => Some(unsafe { std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, unsafe extern "system" fn()>(Self::get_instance_proc_addr) }),
            b"vkGetDeviceProcAddr" => Some(unsafe { std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(Self::get_device_proc_addr) }),
            _ => original,
        }
    }
    unsafe extern "system" fn destroy_device(device: vk::Device, allocator: *const vk::AllocationCallbacks) {
        if device == vk::Device::null() { return; }
        // Ask the framework directly, bypassing this adapter. Its original function
        // must still remove device dispatch bookkeeping and forward the allocator.
        let original = unsafe { Framework::get_device_proc_addr(device, c"vkDestroyDevice".as_ptr()) };
        if let Some(original) = original {
            unsafe { crate::device::destroy_private_resources(device); }
            if crate::probe_ngx::enabled() || crate::preupscale::active() { crate::probe_ngx::forget_device(device); }
            if crate::preupscale::active() { crate::preupscale::forget_device(device); }
            let destroy: vk::PFN_vkDestroyDevice = unsafe { std::mem::transmute(original) };
            unsafe { destroy(device, allocator); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "system" fn dummy() {}
    #[test]
    fn intercept_preserves_null_and_unrelated_commands() {
        unsafe {
            assert!(EntryPoints::intercept(c"vkDestroyDevice".as_ptr(), None).is_none());
            assert_eq!(EntryPoints::intercept(c"vkQueuePresentKHR".as_ptr(), Some(dummy)).unwrap() as usize, dummy as *const () as usize);
            for name in [c"vkDestroyDevice", c"vkGetInstanceProcAddr", c"vkGetDeviceProcAddr"] {
                assert_ne!(EntryPoints::intercept(name.as_ptr(), Some(dummy)).unwrap() as usize, dummy as *const () as usize);
            }
        }
    }
    #[test]
    fn the_handle64_probe_wraps_only_when_on_and_only_a_present_entry_point() {
        use ash::vk::Handle;
        let device = vk::Device::from_raw(0xd00d);
        let name = crate::probe_ngx::HANDLE64_NAME.as_ptr();
        let same = |p: vk::PFN_vkVoidFunction| p.map(|f| f as usize);
        unsafe {
            // Off: the next layer's pointer, untouched, and nothing recorded.
            assert_eq!(same(EntryPoints::intercept_probe(device, name, Some(dummy), false)), same(Some(dummy)));
            assert!(crate::probe_ngx::handle64_next(device).is_none());
            // On, extension absent: still null.
            assert!(EntryPoints::intercept_probe(device, name, None, true).is_none());
            // On, unrelated names: untouched.
            for other in [c"vkGetImageViewHandleNVX", c"vkCmdCuLaunchKernelNVX", c"vkQueueSubmit"] {
                assert_eq!(same(EntryPoints::intercept_probe(device, other.as_ptr(), Some(dummy), true)), same(Some(dummy)));
            }
            // On, present: wrapped, and the next pointer remembered for forwarding.
            let wrapped = EntryPoints::intercept_probe(device, name, Some(dummy), true);
            assert_ne!(same(wrapped), same(Some(dummy)));
            assert!(crate::probe_ngx::handle64_next(device).is_some());
        }
        crate::probe_ngx::forget_device(device);
        assert!(crate::probe_ngx::handle64_next(device).is_none());
    }

    #[test]
    fn the_handle64_wrapper_forwards_arguments_and_result_unchanged() {
        use ash::vk::Handle;
        unsafe extern "system" fn next(device: vk::Device, info: *const vk::ImageViewHandleInfoNVX) -> u64 {
            device.as_raw() ^ unsafe { (*info).image_view.as_raw() }
        }
        let device = vk::Device::from_raw(0xbeef);
        crate::probe_ngx::remember_handle64(device, next);
        let info = vk::ImageViewHandleInfoNVX { image_view: vk::ImageView::from_raw(0x1234), ..Default::default() };
        // The probe is off in the test process and no tracker exists for this handle, so this is
        // pure forwarding.
        assert_eq!(unsafe { EntryPoints::get_image_view_handle64_nvx(device, &info) }, 0xbeef ^ 0x1234);
        crate::probe_ngx::forget_device(device);
    }
}
