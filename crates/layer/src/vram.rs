//! Read-only VRAM readings for the log (`VK_EXT_memory_budget`): what the process uses of the
//! device-local heaps and the budget the driver gives it, summed over those heaps. Measurement only:
//! nothing here allocates or changes what is allocated. The first NVIDIA device the network loads on
//! is the one read ([`init`]); without the extension the readings are empty.

use std::sync::OnceLock;

use ash::vk;

struct Probe {
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
}

static PROBE: OnceLock<Option<Probe>> = OnceLock::new();

/// Remembers the device to read, once per process, if it has `VK_EXT_memory_budget`.
pub(crate) fn init(instance: &ash::Instance, physical: vk::PhysicalDevice) {
    PROBE.get_or_init(|| {
        // SAFETY: `physical` belongs to `instance`; a plain property query.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let budget = extensions.iter().any(|e| {
            // SAFETY: the driver NUL-terminates extension names.
            unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }.to_bytes() == b"VK_EXT_memory_budget"
        });
        budget.then(|| Probe { instance: instance.clone(), physical })
    });
}

/// (used, budget) in MiB over the device-local heaps, or `None` without a probe.
pub(crate) fn reading() -> Option<(u64, u64)> {
    let probe = PROBE.get()?.as_ref()?;
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::builder().push_next(&mut budget).build();
    // SAFETY: the device has the extension (checked in `init`); a plain property query.
    unsafe { probe.instance.get_physical_device_memory_properties2(probe.physical, &mut props) };
    let heaps = props.memory_properties.memory_heap_count as usize;
    let (mut used, mut total) = (0u64, 0u64);
    for (i, heap) in props.memory_properties.memory_heaps[..heaps].iter().enumerate() {
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            used += budget.heap_usage[i];
            total += budget.heap_budget[i];
        }
    }
    Some((used >> 20, total >> 20))
}

/// ` vram=USED/BUDGET MiB` for a log line, or nothing without a probe.
pub(crate) fn note() -> String {
    reading().map_or_else(String::new, |(used, budget)| format!(" vram={used}/{budget} MiB"))
}
