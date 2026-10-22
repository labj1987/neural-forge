//! Read-only VRAM readings for the log (`VK_EXT_memory_budget`): what the process uses of the
//! device-local heaps and the budget the driver gives it, summed over those heaps. Measurement only:
//! nothing here allocates or changes what is allocated. Each network loader keeps its own [`Probe`]
//! (games create and destroy probe devices and instances first: a reading only ever goes to the
//! loader's own, live device); without the extension, or without a `vkGetPhysicalDeviceMemoryProperties2`
//! the instance may call ([`Properties2`]), there is none.

use std::ffi::CStr;

use ash::vk;

/// Which `vkGetPhysicalDeviceMemoryProperties2` an instance may call: the core one (Vulkan 1.1 asked
/// for at `vkCreateInstance`, and a 1.1 physical device), the KHR one (its extension enabled), or
/// none. A name `vkGetInstanceProcAddr` answers for is no permission to call it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Properties2 {
    #[default]
    None,
    Core,
    Khr,
}

impl Properties2 {
    /// From the application's `vkCreateInstance` call.
    pub(crate) fn of(info: &vk::InstanceCreateInfo) -> Self {
        // SAFETY: the application's create info, valid for the call: a non-null application info, and
        // `enabled_extension_count` NUL-terminated names when that is not 0.
        let (api, extensions) = unsafe {
            let api = if info.p_application_info.is_null() { 0 } else { (*info.p_application_info).api_version };
            let names: &[*const std::ffi::c_char] = if info.enabled_extension_count == 0 || info.pp_enabled_extension_names.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(info.pp_enabled_extension_names, info.enabled_extension_count as usize)
            };
            (api, names.iter().map(|&n| CStr::from_ptr(n)).collect::<Vec<_>>())
        };
        if at_least_1_1(api) {
            Self::Core
        } else if extensions.iter().any(|n| n.to_bytes() == b"VK_KHR_get_physical_device_properties2") {
            Self::Khr
        } else {
            Self::None
        }
    }
}

/// An `apiVersion` of 1.1 or later (0 is 1.0).
fn at_least_1_1(api: u32) -> bool {
    api != 0 && (vk::api_version_major(api), vk::api_version_minor(api)) >= (1, 1)
}

/// One device's budget query.
pub(crate) struct Probe {
    properties2: vk::PFN_vkGetPhysicalDeviceMemoryProperties2,
    physical: vk::PhysicalDevice,
}

/// The device's budget query, if it has `VK_EXT_memory_budget` and the instance may query it
/// (`allowed`: [`Properties2::of`] its creation). The entry points come from `gipa` and are checked: a
/// missing one gives none (ash's table would call a stub that panics, and the layer aborts on a panic).
pub(crate) fn probe(gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice, allowed: Properties2) -> Option<Probe> {
    // SAFETY: `gipa` is the next layer's for `instance`; the names are NUL-terminated.
    let get = |name: &CStr| unsafe { gipa(instance, name.as_ptr()) };
    let enumerate = get(c"vkEnumerateDeviceExtensionProperties")?;
    let properties = get(c"vkGetPhysicalDeviceProperties")?;
    // SAFETY: the instance-level functions those names give.
    let (enumerate, properties) = unsafe {
        (
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkEnumerateDeviceExtensionProperties>(enumerate),
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceProperties>(properties),
        )
    };
    // The core query also needs a 1.1 physical device.
    let mut props = vk::PhysicalDeviceProperties::default();
    // SAFETY: a plain query of the instance's own physical device.
    unsafe { properties(physical, &mut props) };
    let name = match allowed {
        Properties2::Core if at_least_1_1(props.api_version) => c"vkGetPhysicalDeviceMemoryProperties2",
        Properties2::Khr => c"vkGetPhysicalDeviceMemoryProperties2KHR",
        _ => return None,
    };
    // SAFETY: the instance-level function that name gives, which the instance may call (`allowed`).
    let properties2 = unsafe { std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceMemoryProperties2>(get(name)?) };
    let mut count = 0u32;
    // SAFETY: the two-call idiom, with a buffer of the size the first call gave.
    let extensions = unsafe {
        if enumerate(physical, std::ptr::null(), &mut count, std::ptr::null_mut()) != vk::Result::SUCCESS {
            return None;
        }
        let mut all = vec![vk::ExtensionProperties::default(); count as usize];
        let result = enumerate(physical, std::ptr::null(), &mut count, all.as_mut_ptr());
        if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
            return None;
        }
        all.truncate(count as usize);
        all
    };
    let budget = extensions.iter().any(|e| {
        // The name within its fixed-size array (the driver NUL-terminates it; nothing past the array is read).
        // SAFETY: c_char and u8 have the same size and alignment.
        let bytes = unsafe { std::slice::from_raw_parts(e.extension_name.as_ptr().cast::<u8>(), e.extension_name.len()) };
        CStr::from_bytes_until_nul(bytes).is_ok_and(|n| n.to_bytes() == b"VK_EXT_memory_budget")
    });
    budget.then_some(Probe { properties2, physical })
}

/// (used, budget) in MiB over the device-local heaps. The probe's device is live (the caller's).
pub(crate) fn reading(probe: &Probe) -> (u64, u64) {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: the device has the extension (checked in `probe`); a plain property query.
    unsafe { (probe.properties2)(probe.physical, &mut props) };
    let heaps = (props.memory_properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    let (mut used, mut total) = (0u64, 0u64);
    for (i, heap) in props.memory_properties.memory_heaps[..heaps].iter().enumerate() {
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            used += budget.heap_usage[i];
            total += budget.heap_budget[i];
        }
    }
    (used >> 20, total >> 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core query only for an instance created for Vulkan 1.1 or later, the KHR one only with its
    /// extension enabled, none otherwise (a 1.0 instance without the extension).
    #[test]
    fn the_query_follows_how_the_instance_was_created() {
        let app = |api| vk::ApplicationInfo { api_version: api, ..Default::default() };
        let (v10, v11, v13) = (app(vk::API_VERSION_1_0), app(vk::API_VERSION_1_1), app(vk::API_VERSION_1_3));
        let khr = [c"VK_KHR_get_physical_device_properties2".as_ptr()];
        let info = |a: Option<&vk::ApplicationInfo>, ext: &[*const std::ffi::c_char]| -> vk::InstanceCreateInfo<'static> {
            vk::InstanceCreateInfo {
                p_application_info: a.map_or(std::ptr::null(), |a| std::ptr::from_ref(a).cast()),
                enabled_extension_count: ext.len() as u32,
                pp_enabled_extension_names: if ext.is_empty() { std::ptr::null() } else { ext.as_ptr() },
                ..Default::default()
            }
        };
        assert_eq!(Properties2::of(&info(None, &[])), Properties2::None, "no application info: 1.0");
        assert_eq!(Properties2::of(&info(Some(&app(0)), &[])), Properties2::None, "0 is 1.0");
        assert_eq!(Properties2::of(&info(Some(&v10), &[])), Properties2::None);
        assert_eq!(Properties2::of(&info(Some(&v10), &khr)), Properties2::Khr);
        assert_eq!(Properties2::of(&info(None, &khr)), Properties2::Khr);
        assert_eq!(Properties2::of(&info(Some(&v11), &[])), Properties2::Core);
        assert_eq!(Properties2::of(&info(Some(&v13), &khr)), Properties2::Core);
    }
}
