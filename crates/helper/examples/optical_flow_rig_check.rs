//! Real-hardware check of the helper's GPU motion-vector path, run under the same Wine/Proton
//! the helper uses: builds the device the way `main.rs::create_vulkan_context` does with an
//! optical-flow queue, then for many frames uploads a synthetic frame with a known per-frame
//! translation into a Color image, runs `GpuFlow::estimate` into an MVec image exactly as
//! `FrameResources::evaluate` does, and reads MVec back to check the vectors. Proves the
//! device comes up with the flow queue, the vectors are right, and nothing hangs.
//!
//! Usage (under the helper's runner): optical_flow_rig_check.exe [frames] [width] [height]
//! Exit status: 0 = pass; nonzero = a failed check, or 3 when a single frame stalls >10 s.
//! Every line also goes, flushed, to `optical_flow_rig_check.log` beside the executable:
//! Proton does not reliably pass a Windows program's stdout through.
use ash::vk;
use neural_forge_helper::optical_flow::{FlowQueue, GpuFlow};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static PROGRESS: AtomicU64 = AtomicU64::new(0);
static LOG: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();

macro_rules! say {
    ($($t:tt)*) => {{
        let line = format!($($t)*);
        println!("{line}");
        if let Some(f) = LOG.get() {
            use std::io::Write;
            let mut f = f.lock().unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }};
}

const STEP: u32 = 4;

fn noise_frame(w: u32, h: u32, shift: u32, out: &mut [u8]) {
    for y in 0..h {
        for x in 0..w {
            let sx = (x + w - shift % w) % w; // content moves right by `shift`
            let n = ((sx / 8).wrapping_mul(747796405) ^ (y / 8).wrapping_mul(2891336453)).wrapping_mul(277803737);
            let v = (n >> 24) as u8;
            let i = ((y * w + x) * 4) as usize;
            out[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Median motion over the interior of the MVec readback (every 16th pixel).
fn median_motion(mvec: &[u8], w: u32, h: u32) -> (f32, f32) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for y in (64..h - 64).step_by(16) {
        for x in (64..w - 64).step_by(16) {
            let i = ((y * w + x) * 4) as usize;
            xs.push(half_to_f32(u16::from_le_bytes([mvec[i], mvec[i + 1]])));
            ys.push(half_to_f32(u16::from_le_bytes([mvec[i + 2], mvec[i + 3]])));
        }
    }
    xs.sort_by(f32::total_cmp);
    ys.sort_by(f32::total_cmp);
    (xs[xs.len() / 2], ys[ys.len() / 2])
}

unsafe fn image(device: &ash::Device, mem: &vk::PhysicalDeviceMemoryProperties, w: u32, h: u32, format: vk::Format) -> vk::Image {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
    let image = device.create_image(&info, None).expect("image");
    let req = device.get_image_memory_requirements(image);
    let index = (0..mem.memory_type_count)
        .find(|&i| req.memory_type_bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .expect("device-local memory");
    let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(index), None).expect("image memory");
    device.bind_image_memory(image, memory, 0).expect("bind image");
    image
}

fn barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1).build())
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .build()
}

fn region(w: u32, h: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build())
        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .build()
}

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (frames, w, h) = (*args.first().unwrap_or(&500), *args.get(1).unwrap_or(&2560), *args.get(2).unwrap_or(&1440));
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        if let Ok(f) = std::fs::File::create(dir.join("optical_flow_rig_check.log")) {
            let _ = LOG.set(std::sync::Mutex::new(f));
        }
    }
    // Watchdog: a stalled frame is the failure this exists to catch, so fail loudly.
    std::thread::spawn(|| {
        let mut last = u64::MAX;
        loop {
            std::thread::sleep(Duration::from_secs(10));
            let now = PROGRESS.load(Ordering::Relaxed);
            if now == last {
                say!("STALL: no progress for 10 s at frame {now}");
                std::process::exit(3);
            }
            last = now;
        }
    });

    unsafe {
        let entry = ash::Entry::load().expect("vulkan-1.dll");
        let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app), None).expect("instance");
        let pd = instance.enumerate_physical_devices().unwrap().into_iter().find(|&p| instance.get_physical_device_properties(p).vendor_id == 0x10DE).expect("NVIDIA GPU");
        say!("gpu: {:?}", std::ffi::CStr::from_ptr(instance.get_physical_device_properties(pd).device_name.as_ptr()));
        let families = instance.get_physical_device_queue_family_properties(pd);
        let Some(family) = families.iter().position(|p| p.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV | vk::QueueFlags::TRANSFER)) else {
            say!("FAIL: no queue family with OPTICAL_FLOW_NV | TRANSFER");
            std::process::exit(1);
        };
        let family = family as u32;
        say!("flow queue family: {family}");
        let names = ["VK_NV_optical_flow", "VK_KHR_synchronization2", "VK_KHR_format_feature_flags2"].map(|n| std::ffi::CString::new(n).unwrap());
        let ptrs: Vec<*const std::ffi::c_char> = names.iter().map(|c| c.as_ptr()).collect();
        let mut queues = vec![vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
        if family != 0 {
            queues.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&[1.0]).build());
        }
        let mut of_on = vk::PhysicalDeviceOpticalFlowFeaturesNV::builder().optical_flow(true);
        let mut s2_on = vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);
        let info = vk::DeviceCreateInfo::builder().queue_create_infos(&queues).enabled_extension_names(&ptrs).push_next(&mut of_on).push_next(&mut s2_on);
        let device = instance.create_device(pd, &info, None).expect("device with optical-flow queue");
        let main_queue = device.get_device_queue(0, 0);
        let fq = FlowQueue { family, queue: device.get_device_queue(family, 0) };

        let mem = instance.get_physical_device_memory_properties(pd);
        let color = image(&device, &mem, w, h, vk::Format::B8G8R8A8_UNORM);
        let mvec = image(&device, &mem, w, h, vk::Format::R16G16_SFLOAT);
        let bytes = u64::from(w) * u64::from(h) * 4;
        let staging = device
            .create_buffer(&vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST), None)
            .expect("staging");
        let req = device.get_buffer_memory_requirements(staging);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED;
        let index = (0..mem.memory_type_count).find(|&i| req.memory_type_bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(host)).expect("host memory");
        let staging_memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(index), None).expect("staging memory");
        device.bind_buffer_memory(staging, staging_memory, 0).expect("bind staging");
        let mapped = std::slice::from_raw_parts_mut(device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("map").cast::<u8>(), bytes as usize);
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(0).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
        let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let run = |record: &dyn Fn()| {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            record();
            device.end_command_buffer(cmd).unwrap();
            device.reset_fences(&[fence]).unwrap();
            device.queue_submit(main_queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, 10_000_000_000).expect("fence");
        };
        let pipe = |b: &[vk::ImageMemoryBarrier]| {
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], b)
        };
        let range = vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1).build();
        // What `FrameResources::evaluate`'s upload does: Color from the frame, MVec cleared,
        // both left in SHADER_READ_ONLY_OPTIMAL.
        let upload = || {
            run(&|| {
                pipe(&[barrier(color, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL), barrier(mvec, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)]);
                device.cmd_copy_buffer_to_image(cmd, staging, color, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(w, h)]);
                device.cmd_clear_color_image(cmd, mvec, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]);
                pipe(&[
                    barrier(color, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    barrier(mvec, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                ]);
            });
        };
        let readback = || {
            run(&|| {
                pipe(&[barrier(mvec, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)]);
                device.cmd_copy_image_to_buffer(cmd, mvec, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, staging, &[region(w, h)]);
                pipe(&[barrier(mvec, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
            });
        };

        let mut flow = GpuFlow::new(&instance, &device, pd, 0, &fq, w, h, 1).expect("GPU flow session");
        say!("session: {w}x{h}");
        noise_frame(w, h, 0, mapped);
        upload();
        assert!(!flow.estimate(&device, main_queue, color, mvec, [1.0, 1.0]).expect("seed"), "first frame only seeds history");
        upload();
        assert!(flow.estimate(&device, main_queue, color, mvec, [1.0, 1.0]).expect("stationary"));
        readback();
        let (sx, sy) = median_motion(mapped, w, h);
        say!("stationary median=({sx:.2},{sy:.2}) expected (0,0)");
        let mut worst = 0f32;
        let mut total = Duration::ZERO;
        for i in 1..=frames {
            noise_frame(w, h, i * STEP, mapped);
            upload();
            let t = Instant::now();
            assert!(flow.estimate(&device, main_queue, color, mvec, [1.0, 1.0]).expect("estimate"));
            total += t.elapsed();
            PROGRESS.store(u64::from(i), Ordering::Relaxed);
            if i % 50 == 0 || i == 1 {
                readback();
                let (dx, dy) = median_motion(mapped, w, h);
                worst = worst.max((dx + STEP as f32).abs()).max(dy.abs());
                say!("frame {i}: median=({dx:.2},{dy:.2}) expected (-{STEP},0)");
            }
        }
        say!("frames={frames} avg_estimate={:.2}ms worst_error={worst:.2}px", total.as_secs_f64() * 1000.0 / f64::from(frames));
        flow.destroy(&device);
        let pass = sx.abs() < 0.5 && sy.abs() < 0.5 && worst < 1.0;
        say!("{}", if pass { "PASS" } else { "FAIL: vectors off" });
        std::process::exit(if pass { 0 } else { 1 });
    }
}
