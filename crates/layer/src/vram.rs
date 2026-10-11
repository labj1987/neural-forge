//! Read-only VRAM readings for the log (`VK_EXT_memory_budget`): what the process uses of the
//! device-local heaps and the budget the driver gives it, summed over those heaps. Measurement only:
//! nothing here allocates or changes what is allocated. Each network loader keeps its own [`Probe`]
//! (games create and destroy probe devices and instances first: a reading only ever goes to the
//! loader's own, live device); without the extension, or without `vkGetPhysicalDeviceMemoryProperties2`
//! (or its KHR name) from the next layer, there is none.

use std::ffi::CStr;

use ash::vk;

/// One device's budget query.
pub(crate) struct Probe {
    properties2: vk::PFN_vkGetPhysicalDeviceMemoryProperties2,
    physical: vk::PhysicalDevice,
}

/// The device's budget query, if it has `VK_EXT_memory_budget`. The entry points come from `gipa` and
/// are checked: a missing one gives none (ash's table would call a stub that panics, and the layer
/// aborts on a panic; a 1.0 instance has no `vkGetPhysicalDeviceMemoryProperties2`).
pub(crate) fn probe(gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice) -> Option<Probe> {
    // SAFETY: `gipa` is the next layer's for `instance`; the names are NUL-terminated.
    let get = |name: &CStr| unsafe { gipa(instance, name.as_ptr()) };
    let enumerate = get(c"vkEnumerateDeviceExtensionProperties")?;
    let properties2 = get(c"vkGetPhysicalDeviceMemoryProperties2").or_else(|| get(c"vkGetPhysicalDeviceMemoryProperties2KHR"))?;
    // SAFETY: the instance-level functions those names give.
    let (enumerate, properties2) = unsafe {
        (
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkEnumerateDeviceExtensionProperties>(enumerate),
            std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceMemoryProperties2>(properties2),
        )
    };
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
        // SAFETY: the driver NUL-terminates extension names.
        unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }.to_bytes() == b"VK_EXT_memory_budget"
    });
    budget.then_some(Probe { properties2, physical })
}

/// (used, budget) in MiB over the device-local heaps. The probe's device is live (the caller's).
pub(crate) fn reading(probe: &Probe) -> (u64, u64) {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::builder().push_next(&mut budget).build();
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
