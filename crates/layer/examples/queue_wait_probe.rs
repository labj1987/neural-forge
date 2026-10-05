//! Can a second queue run while another queue of the same device waits on a host-set event?
//!
//! The hold inside DLSS's command buffer (`preupscale::inline`) pauses the game's queue at a
//! `vkCmdWaitEvents` on an event the host sets, and meanwhile runs its own work on a queue of the
//! layer's. This probe measures, per candidate queue, how long a trivial submission (a buffer fill)
//! takes to finish while queue (family 0, index 0) is parked at such a wait.
//!
//! ```text
//! cargo build --release -p neural-forge-layer --example queue_wait_probe
//! ./target/release/examples/queue_wait_probe
//! ```
use std::time::{Duration, Instant};

use ash::vk;

fn main() {
    // SAFETY: plain Vulkan setup through the system loader; every handle is used on this thread.
    unsafe {
        let entry = ash::Entry::load().expect("loader");
        let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app), None).expect("instance");
        let pd = *instance.enumerate_physical_devices().unwrap().first().expect("device");
        let props = instance.get_physical_device_properties(pd);
        println!("device: {}", std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy());
        let families = instance.get_physical_device_queue_family_properties(pd);
        for (i, f) in families.iter().enumerate() {
            println!("family {i}: {:?} x{}", f.queue_flags, f.queue_count);
        }
        // Family 0 with 2 queues; every other family with compute, 1 queue each.
        let prio = [1.0f32, 1.0];
        let mut infos = vec![vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&prio[..2.min(families[0].queue_count as usize)]).build()];
        let others: Vec<u32> = (1..families.len() as u32).filter(|&f| families[f as usize].queue_flags.contains(vk::QueueFlags::COMPUTE)).collect();
        for &f in &others {
            infos.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(f).queue_priorities(&prio[..1]).build());
        }
        let device = instance.create_device(pd, &vk::DeviceCreateInfo::builder().queue_create_infos(&infos), None).expect("device");
        let memory = instance.get_physical_device_memory_properties(pd);

        let parked = device.get_device_queue(0, 0);
        let event = device.create_event(&vk::EventCreateInfo::default(), None).unwrap();
        let pool0 = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(0), None).unwrap();
        let park_cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool0).command_buffer_count(1)).unwrap()[0];
        device.begin_command_buffer(park_cb, &vk::CommandBufferBeginInfo::default()).unwrap();
        let mb = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::HOST_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).build();
        device.cmd_wait_events(park_cb, &[event], vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::TRANSFER, &[mb], &[], &[]);
        device.end_command_buffer(park_cb).unwrap();

        let mut candidates: Vec<(String, u32, u32)> = Vec::new();
        if families[0].queue_count >= 2 {
            candidates.push(("same family (0, index 1)".into(), 0, 1));
        }
        for &f in &others {
            candidates.push((format!("family {f} (index 0)"), f, 0));
        }
        for (name, family, index) in candidates {
            let queue = device.get_device_queue(family, index);
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            let cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            let buffer = device.create_buffer(&vk::BufferCreateInfo::builder().size(1 << 20).usage(vk::BufferUsageFlags::TRANSFER_DST), None).unwrap();
            let reqs = device.get_buffer_memory_requirements(buffer);
            let index_m = (0..memory.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
            let mem = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index_m), None).unwrap();
            device.bind_buffer_memory(buffer, mem, 0).unwrap();
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_fill_buffer(cb, buffer, 0, vk::WHOLE_SIZE, 0x1234_5678);
            device.end_command_buffer(cb).unwrap();
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            // Baseline: the queue alone.
            let t = Instant::now();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, 2_000_000_000).unwrap();
            let alone = t.elapsed();
            device.reset_fences(&[fence]).unwrap();
            // Parked: queue (0, 0) waits on the event while this one works.
            device.reset_event(event).unwrap();
            let park_fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            device.queue_submit(parked, &[vk::SubmitInfo::builder().command_buffers(&[park_cb]).build()], park_fence).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            let t = Instant::now();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()], fence).unwrap();
            let r = device.wait_for_fences(&[fence], true, 2_000_000_000);
            let parked_time = t.elapsed();
            device.set_event(event).unwrap();
            let p = device.wait_for_fences(&[park_fence], true, 5_000_000_000);
            println!(
                "{name}: alone {:.3} ms; while (0,0) waits on a host event: {} after {:.1} ms; parked queue released: {:?}",
                alone.as_secs_f64() * 1e3,
                if r.is_ok() { "finished" } else { "DID NOT FINISH" },
                parked_time.as_secs_f64() * 1e3,
                p.map(|_| "ok")
            );
            if r.is_err() {
                // It may finish once released.
                let _ = device.wait_for_fences(&[fence], true, 5_000_000_000);
            }
            device.destroy_fence(park_fence, None);
            device.destroy_fence(fence, None);
            device.destroy_buffer(buffer, None);
            device.free_memory(mem, None);
            device.destroy_command_pool(pool, None);
        }
        device.device_wait_idle().unwrap();
        device.destroy_event(event, None);
        device.destroy_command_pool(pool0, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
