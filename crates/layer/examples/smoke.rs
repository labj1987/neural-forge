//! A real-loader smoke test, not a unit test: creates an actual `VkInstance`/`VkDevice`
//! through the system Vulkan loader with the layer enabled, so it exercises the parts
//! `cargo test` cannot -- the manifest negotiation, `vulkan_layer`'s dispatch-table
//! wiring, and `NeuralForgeDeviceInfo::new`'s function-pointer resolution -- against a real
//! loader and ICD (lavapipe is enough; nothing here needs a GPU).
//!
//! Run via `scripts/smoke-test.sh`, which builds the layer, writes a manifest pointing
//! at the just-built `.so`, and sets the env vars this needs
//! (`VK_LAYER_PATH`/`NEURAL_FORGE_ENABLE`/`NEURAL_FORGE_LOG`) before running it. Running this
//! directly without that setup will simply not find the layer -- which is a fine
//! outcome too (it means the layer opted out cleanly), not a crash.

use ash::vk;

fn main() {
    // SAFETY: dynamically loads the system Vulkan loader (`libvulkan.so.1`); the usual
    // caveats of loading arbitrary shared libraries apply and are accepted here the
    // same way `ash-window`/every other `ash` consumer accepts them.
    let entry = unsafe { ash::Entry::load() }.expect("no Vulkan loader found");

    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is a valid, fully-populated `VkInstanceCreateInfo`.
    let instance = unsafe { entry.create_instance(&create_info, None) }.expect("vkCreateInstance failed");
    println!("smoke: instance created (layer negotiation, if any, already happened)");

    // SAFETY: `instance` was just created above and is destroyed at the end of main.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.expect("enumerate_physical_devices failed");
    let physical_device = *physical_devices
        .first()
        .expect("no Vulkan physical devices -- is a software ICD (lavapipe) installed?");
    // SAFETY: `physical_device` is one of the handles just enumerated above.
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
    println!("smoke: using physical device: {name:?}");

    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    // SAFETY: `device_create_info` is valid; queue family 0 exists on every physical
    // device (the Vulkan spec guarantees at least one queue family).
    let device =
        unsafe { instance.create_device(physical_device, &device_create_info, None) }.expect("vkCreateDevice failed");
    println!("smoke: device created -- if the layer is enabled, NeuralForgeDeviceInfo::new ran without panicking");

    // SAFETY: destroying in the reverse order of creation, each handle only once.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    println!("smoke: OK");
}
