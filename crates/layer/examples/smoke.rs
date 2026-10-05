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
    // `vkQueueSubmit2` needs synchronization2, where the device has it (lavapipe does).
    let mut supported = vk::PhysicalDeviceVulkan13Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::builder().push_next(&mut supported);
    // SAFETY: `physical_device` is live; the chain is a valid Vulkan 1.3 feature query.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
    let submit2 = props.api_version >= vk::API_VERSION_1_3 && supported.synchronization2 == vk::TRUE;
    let mut enabled = vk::PhysicalDeviceVulkan13Features::builder().synchronization2(submit2);
    let mut device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    if submit2 {
        device_create_info = device_create_info.push_next(&mut enabled);
    }
    // SAFETY: `device_create_info` is valid; queue family 0 exists on every physical
    // device (the Vulkan spec guarantees at least one queue family).
    let device =
        unsafe { instance.create_device(physical_device, &device_create_info, None) }.expect("vkCreateDevice failed");
    println!("smoke: device created -- if the layer is enabled, NeuralForgeDeviceInfo::new ran without panicking");

    // The image, view and submit paths the `NEURAL_FORGE_PROBE_NGX` probe hooks (it is off
    // unless smoke-test.sh's second pass sets it). Neither NVX extension is enabled here, so
    // their entry points must come back null, probe or not, and nothing may crash.
    // SAFETY: plain object creation on the device above, destroyed again below.
    unsafe {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width: 64, height: 32, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
        let image = device.create_image(&image_info, None).expect("vkCreateImage failed");
        let requirements = device.get_image_memory_requirements(image);
        let memory_type = requirements.memory_type_bits.trailing_zeros();
        let memory = device
            .allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(requirements.size).memory_type_index(memory_type), None)
            .expect("vkAllocateMemory failed");
        device.bind_image_memory(image, memory, 0).expect("vkBindImageMemory failed");
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, level_count: 1, layer_count: 1, ..Default::default() });
        let view = device.create_image_view(&view_info, None).expect("vkCreateImageView failed");
        for name in [c"vkGetImageViewHandleNVX", c"vkGetImageViewAddressNVX", c"vkCmdCuLaunchKernelNVX", c"vkCreateCuFunctionNVX"] {
            let p = (instance.fp_v1_0().get_device_proc_addr)(device.handle(), name.as_ptr());
            assert!(p.is_none(), "{name:?} resolved on a device that never enabled its extension");
        }
        let queue = device.get_device_queue(0, 0);
        device.queue_submit(queue, &[vk::SubmitInfo::default()], vk::Fence::null()).expect("empty vkQueueSubmit failed");
        device.queue_wait_idle(queue).expect("vkQueueWaitIdle failed");
        // A recorded command buffer through both submit entry points, each with a fence: the
        // layer issues the application's submits to the next layer itself (to see their result
        // before its tracking advances), so the fence must signal and the result come back.
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(0), None).expect("vkCreateCommandPool failed");
        let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).expect("vkAllocateCommandBuffers failed")[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).expect("vkCreateFence failed");
        let record = || {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).expect("vkBeginCommandBuffer failed");
            let barrier = vk::ImageMemoryBarrier::builder()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, level_count: 1, layer_count: 1, ..Default::default() });
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier.build()]);
            device.end_command_buffer(cmd).expect("vkEndCommandBuffer failed");
        };
        record();
        device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence).expect("vkQueueSubmit failed");
        device.wait_for_fences(&[fence], true, 5_000_000_000).expect("the vkQueueSubmit fence never signalled");
        if submit2 {
            device.reset_fences(&[fence]).expect("vkResetFences failed");
            record();
            let infos = [vk::CommandBufferSubmitInfo::builder().command_buffer(cmd).build()];
            device.queue_submit2(queue, &[vk::SubmitInfo2::builder().command_buffer_infos(&infos).build()], fence).expect("vkQueueSubmit2 failed");
            device.wait_for_fences(&[fence], true, 5_000_000_000).expect("the vkQueueSubmit2 fence never signalled");
        }
        println!("smoke: vkQueueSubmit{} with a fence OK", if submit2 { " and vkQueueSubmit2" } else { "" });
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_image_view(view, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
    println!("smoke: image, view and submit hooks OK; NVX entry points absent as expected");

    // SAFETY: destroying in the reverse order of creation, each handle only once.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    println!("smoke: OK");
}
