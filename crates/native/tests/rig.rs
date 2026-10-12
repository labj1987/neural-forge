//! The network through the C API on a real GPU, against `dlss5vk parity --dump` of the same frame:
//! the device made with `device_extend`, the network opened on a queue of a compute-only family (as
//! the layer opens it), the recorded graph run from a graphics-family queue's primary (buffers shared
//! between the two), and the head compared byte for byte.
//!
//! Runs only with `NEURAL_FORGE_NATIVE_RIG=<dump dir>` (holding `features.f32` and `head.f32` of a
//! 1486x836 frame); `NEURAL_FORGE_NATIVE_MODEL` overrides the model directory (default: the user data
//! dir's `neural-forge/model`). The same variable runs the rebuild test (a small build, then a large one
//! on the same network), the fallback to barriers and the device-creation test.
//!
//! One network per process at a time (`nf_native_open` refuses a second): the tests take [`ONE`] so
//! the harness's threads run them one after another.

use ash::vk;
use neural_forge_native::{device_extend, device_supported, Network, OpenInfo};
use std::path::PathBuf;

const WIDTH: u32 = 1486;
const HEIGHT: u32 = 836;

/// Held by every test for its whole run: one network (and one rig device) at a time.
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    family: u32,
    /// A compute family without graphics, where the network loads (as in the layer).
    compute: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
}

fn gpu() -> Gpu {
    // SAFETY: the system Vulkan loader.
    let entry = unsafe { ash::Entry::load() }.expect("Vulkan loader");
    let app = vk::ApplicationInfo::builder().api_version(vk::make_api_version(0, 1, 3, 0));
    // SAFETY: valid create info.
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app), None) }.unwrap();
    // SAFETY: live instance.
    let physical = unsafe { instance.enumerate_physical_devices() }
        .unwrap()
        .into_iter()
        // SAFETY: live instance and device.
        .find(|&p| unsafe { instance.get_physical_device_properties(p) }.vendor_id == 0x10de)
        .expect("an NVIDIA GPU");
    // SAFETY: as above.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let family = families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)).unwrap() as u32;
    let compute = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .expect("a compute-only family") as u32;
    let gipa = entry.static_fn().get_instance_proc_addr;
    // SAFETY: the loader's gipa for this instance.
    unsafe { device_supported(gipa, instance.handle(), physical) }.expect("device support");
    let priorities = [1.0f32];
    let queues = [
        vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&priorities).build(),
        vk::DeviceQueueCreateInfo::builder().queue_family_index(compute).queue_priorities(&priorities).build(),
    ];
    let info = vk::DeviceCreateInfo::builder().queue_create_infos(&queues).build();
    // SAFETY: valid create info, outlives the extension.
    let extension = unsafe { device_extend(gipa, instance.handle(), physical, &info) }.expect("extend");
    // SAFETY: the extended create info is valid while `extension` lives.
    let device = unsafe { instance.create_device(physical, &extension.info, None) }.expect("vkCreateDevice with the network's additions");
    drop(extension);
    // SAFETY: as above.
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    Gpu { _entry: entry, instance, device, physical, family, compute, memory }
}

impl Gpu {
    fn host_buffer(&self, bytes: u64, usage: vk::BufferUsageFlags) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
        // SAFETY: live device; valid create infos.
        unsafe {
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::builder().size(bytes).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let index = (0..self.memory.memory_type_count)
                .find(|&i| req.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(want))
                .unwrap();
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(index), None).unwrap();
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast();
            (buffer, memory, ptr)
        }
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("NEURAL_FORGE_NATIVE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/neural-forge/model"))
}

/// A build after a smaller one on the same network: the kernels outlive the graph, and their split-K
/// scratch must be sized again for the larger graph, not kept from the first build.
#[test]
fn a_larger_build_after_a_smaller_one_on_one_network() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let model = model_dir();
    let g = gpu();
    let open = OpenInfo {
        gipa: g._entry.static_fn().get_instance_proc_addr,
        instance: g.instance.handle(),
        physical: g.physical,
        device: g.device.handle(),
        queue_family: g.compute,
        queue_index: 0,
        frame_family: g.family,
        model_dir: &model,
        chain: std::env::var("NEURAL_FORGE_NATIVE_CHAIN").map_or(true, |v| v != "0"),
        fence_timeout_ms: 5000,
        init_dispatchable: None,
    };
    // SAFETY: the device was made with device_extend; the compute queue is the network's alone.
    let mut net = unsafe { Network::open(&open) }.expect("open");
    // SAFETY: nothing of a previous build exists.
    let small = unsafe { net.build(1280, 720) }.expect("the 1280x720 build");
    eprintln!("1280x720: {small:?}");
    // SAFETY: the small build's graph was never submitted.
    let large = unsafe { net.build(3840, 2160) }.expect("the 3840x2160 build after the 1280x720 one");
    eprintln!("3840x2160: {large:?}");
    assert!(large.field_width >= 3840 && large.field_height >= 2160);
    let d = &g.device;
    // SAFETY: live device and handles; the graph's frame completes before anything is destroyed.
    unsafe {
        let queue = d.get_device_queue(g.family, 0);
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(g.family), None).unwrap();
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        d.cmd_fill_buffer(cmd, large.features, 0, vk::WHOLE_SIZE, 0);
        let to_compute = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[to_compute.build()], &[], &[]);
        d.cmd_execute_commands(cmd, &[net.graph_commands()]);
        d.end_command_buffer(cmd).unwrap();
        d.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence).unwrap();
        d.wait_for_fences(&[fence], true, 5_000_000_000).expect("the 3840x2160 graph completes");
        assert_eq!(net.chain_timeouts().0, 0);
        d.destroy_fence(fence, None);
        d.destroy_command_pool(pool, None);
        drop(net);
        d.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

#[test]
fn the_network_through_the_c_api_matches_dlss5vk() {
    let _one = one_at_a_time();
    let Some(dump) = std::env::var_os("NEURAL_FORGE_NATIVE_RIG").map(PathBuf::from) else {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    };
    let model = model_dir();
    let features = std::fs::read(dump.join("features.f32")).unwrap();
    let expected = std::fs::read(dump.join("head.f32")).unwrap();
    let g = gpu();
    let gipa = g._entry.static_fn().get_instance_proc_addr;
    let open = OpenInfo {
        gipa,
        instance: g.instance.handle(),
        physical: g.physical,
        device: g.device.handle(),
        queue_family: g.compute,
        queue_index: 0,
        frame_family: g.family,
        model_dir: &model,
        chain: std::env::var("NEURAL_FORGE_NATIVE_CHAIN").map_or(true, |v| v != "0"),
        fence_timeout_ms: 5000,
        init_dispatchable: None,
    };
    let t = std::time::Instant::now();
    // SAFETY: the device was made with device_extend; queue 1 is the network's alone.
    let mut net = unsafe { Network::open(&open) }.expect("open");
    let opened = t.elapsed();
    // SAFETY: nothing of a previous build exists.
    let frame = unsafe { net.build(WIDTH, HEIGHT) }.expect("build");
    eprintln!("open {:?}, build {:?}, frame {frame:?}", opened, t.elapsed() - opened);
    assert_eq!((frame.field_width, frame.field_height), (1536, 896));
    assert_eq!(features.len() as u64, u64::from(frame.field_width) * u64::from(frame.field_height) * 64);
    let head_bytes = u64::from(frame.field_width) * u64::from(frame.field_height) * 16;
    assert_eq!(expected.len() as u64, head_bytes);

    let d = &g.device;
    let (up, up_mem, up_ptr) = g.host_buffer(features.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
    let (down, down_mem, down_ptr) = g.host_buffer(head_bytes, vk::BufferUsageFlags::TRANSFER_DST);
    // SAFETY: mapped for at least that many bytes.
    unsafe { std::ptr::copy_nonoverlapping(features.as_ptr(), up_ptr, features.len()) };
    // SAFETY: live device and handles throughout; every buffer outlives the waits below.
    unsafe {
        let queue = d.get_device_queue(g.family, 0);
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(g.family), None).unwrap();
        let cmd = d
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1))
            .unwrap()[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let queries = d.create_query_pool(&vk::QueryPoolCreateInfo::builder().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap();
        let period = g.instance.get_physical_device_properties(g.physical).limits.timestamp_period as f64;
        for run in 0..5 {
            std::ptr::write_bytes(down_ptr, 0, head_bytes as usize);
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            d.cmd_copy_buffer(cmd, up, frame.features, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: features.len() as u64 }]);
            let to_compute = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[to_compute.build()], &[], &[]);
            d.cmd_reset_query_pool(cmd, queries, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            d.cmd_execute_commands(cmd, &[net.graph_commands()]);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
            let to_copy = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[to_copy.build()], &[], &[]);
            d.cmd_copy_buffer(cmd, frame.head, down, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: head_bytes }]);
            d.end_command_buffer(cmd).unwrap();
            d.reset_fences(&[fence]).unwrap();
            let t = std::time::Instant::now();
            d.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence).unwrap();
            d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
            let mut stamps = [0u64; 2];
            d.get_query_pool_results(queries, 0, 2, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
            eprintln!("run {run}: graph {:.3} ms GPU", (stamps[1] - stamps[0]) as f64 * period / 1e6);
            let got = std::slice::from_raw_parts(down_ptr, head_bytes as usize);
            let differing = got.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0.iter()).filter(|(a, b)| a != b).count();
            eprintln!("run {run}: {:?} incl. copies, {differing} of {} head values differ, chain timeouts {:?}", t.elapsed(), expected.len() / 4, net.chain_timeouts());
            assert_eq!(differing, 0, "the head differs from dlss5vk's");
            assert_eq!(net.chain_timeouts().0, 0);
        }
        // Back to back, the graph alone (the GPU at its working clocks, as dlss5vk bench measures).
        for primary in [false, true] {
        let mut times = Vec::new();
        for _ in 0..40 {
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            d.cmd_reset_query_pool(cmd, queries, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            if primary {
                net.record_graph(cmd).unwrap();
            } else {
                d.cmd_execute_commands(cmd, &[net.graph_commands()]);
            }
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
            d.end_command_buffer(cmd).unwrap();
            d.reset_fences(&[fence]).unwrap();
            d.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], fence).unwrap();
            d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
            let mut stamps = [0u64; 2];
            d.get_query_pool_results(queries, 0, 2, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
            times.push((stamps[1] - stamps[0]) as f64 * period / 1e6);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("graph back to back ({}), 40 runs: min {:.3} ms, median {:.3} ms", if primary { "recorded into the primary" } else { "secondary" }, times[0], times[20]);
        }
        assert_eq!(net.chain_timeouts().0, 0);
        d.destroy_query_pool(queries, None);
        d.destroy_fence(fence, None);
        d.destroy_command_pool(pool, None);
        drop(net);
        for (b, m) in [(up, up_mem), (down, down_mem)] {
            d.destroy_buffer(b, None);
            d.free_memory(m, None);
        }
        d.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// The fallback to barriers is a switch: the next build records the graph once, without counter
/// chaining, at the size it is asked for.
#[test]
fn the_fallback_to_barriers_builds_once_without_chaining() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let model = model_dir();
    let g = gpu();
    let open = OpenInfo {
        gipa: g._entry.static_fn().get_instance_proc_addr,
        instance: g.instance.handle(),
        physical: g.physical,
        device: g.device.handle(),
        queue_family: g.compute,
        queue_index: 0,
        frame_family: g.family,
        model_dir: &model,
        chain: true,
        fence_timeout_ms: 5000,
        init_dispatchable: None,
    };
    // SAFETY: the device was made with device_extend; the compute queue is the network's alone.
    let mut net = unsafe { Network::open(&open) }.expect("open");
    // SAFETY: nothing of a previous build exists, and nothing of one is ever submitted here.
    unsafe {
        net.build(640, 360).expect("the first build");
        net.fall_back_to_barriers();
        assert!(net.frame().is_none(), "the chained graph is dropped by the switch");
        let rebuilt = net.build(800, 448).expect("the build after the switch");
        assert_eq!(rebuilt.chained, 0, "recorded with barriers");
        assert!(rebuilt.field_width >= 800 && rebuilt.field_height >= 448, "at the size asked for, not the old one");
        drop(net);
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// `device_extend` only reads the application's create info: its own feature structure keeps its flags,
/// the copy the extended chain carries has the network's set, and the device is created with it. An
/// application enabling VK_EXT_buffer_device_address gets no additions.
#[test]
fn device_extend_copies_the_applications_chain() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let g = gpu();
    let gipa = g._entry.static_fn().get_instance_proc_addr;
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::builder().queue_family_index(g.family).queue_priorities(&priorities).build()];
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
    let info = vk::DeviceCreateInfo::builder().queue_create_infos(&queues).push_next(&mut v12).push_next(&mut v11).build();
    // SAFETY: valid create info, outliving the extension.
    let extension = unsafe { device_extend(gipa, g.instance.handle(), g.physical, &info) }.expect("extend");
    assert_eq!((v12.buffer_device_address, v12.shader_float16, v11.storage_buffer16_bit_access), (0, 0, 0), "the application's structures are untouched");
    assert!(std::ptr::eq(v11.p_next.cast::<vk::PhysicalDeviceVulkan12Features>(), &v12), "and so is their chain");
    // SAFETY: the extended chain is valid while `extension` lives; every node starts with sType and pNext.
    let (mut copied12, mut copied11) = (None, None);
    unsafe {
        let mut node = extension.info.p_next.cast::<vk::BaseInStructure>();
        while !node.is_null() {
            assert_ne!(node.cast::<vk::PhysicalDeviceVulkan12Features>(), std::ptr::addr_of!(v12), "not the application's own");
            match (*node).s_type {
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => copied12 = Some(*node.cast::<vk::PhysicalDeviceVulkan12Features>()),
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => copied11 = Some(*node.cast::<vk::PhysicalDeviceVulkan11Features>()),
                _ => {}
            }
            node = (*node).p_next;
        }
    }
    let (c12, c11) = (copied12.expect("the Vulkan 1.2 features' copy"), copied11.expect("the Vulkan 1.1 features' copy"));
    assert_eq!((c12.buffer_device_address, c12.shader_float16, c11.storage_buffer16_bit_access), (1, 1, 1));
    // SAFETY: the extended create info is valid while `extension` lives.
    let device = unsafe { g.instance.create_device(g.physical, &extension.info, None) }.expect("vkCreateDevice with the copied chain");
    drop(extension);
    let ext = [c"VK_EXT_buffer_device_address".as_ptr()];
    let with_ext = vk::DeviceCreateInfo::builder().queue_create_infos(&queues).enabled_extension_names(&ext).build();
    // SAFETY: valid create info.
    let refused = unsafe { device_extend(gipa, g.instance.handle(), g.physical, &with_ext) };
    assert!(refused.is_err_and(|e| e.contains("VK_EXT_buffer_device_address")));
    // SAFETY: nothing of either device is in use.
    unsafe {
        device.destroy_device(None);
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// The chain a layer gets from the loader: its link node (`VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO`,
/// 48) is copied like any other, and a structure this build does not know is kept, as the application's,
/// at the end of the chain instead of failing the call. A structure the network sets after such a one
/// is refused (it could only be written in place).
#[test]
fn device_extend_keeps_the_loaders_nodes_and_unknown_ones() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let g = gpu();
    let gipa = g._entry.static_fn().get_instance_proc_addr;
    let priorities = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::builder().queue_family_index(g.family).queue_priorities(&priorities).build()];
    // An unknown structure: an sType no registry has, followed by nothing the network sets.
    #[repr(C)]
    struct Unknown {
        s_type: vk::StructureType,
        p_next: *const std::ffi::c_void,
        payload: [u64; 4],
    }
    let unknown = Unknown { s_type: vk::StructureType::from_raw(0x7fff_0123), p_next: std::ptr::null(), payload: [7; 4] };
    // The loader's node, as vk_layer.h lays it out: sType, pNext, function, then a union of pointers.
    #[repr(C)]
    struct LoaderNode {
        s_type: vk::StructureType,
        p_next: *const std::ffi::c_void,
        function: u32,
        u: *const std::ffi::c_void,
    }
    let loader = LoaderNode { s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO, p_next: std::ptr::from_ref(&unknown).cast(), function: 0, u: 0x1234 as *const _ };
    let mut v12 = vk::PhysicalDeviceVulkan12Features { p_next: std::ptr::from_ref(&loader).cast_mut().cast(), ..Default::default() };
    let info = vk::DeviceCreateInfo { p_next: std::ptr::from_mut(&mut v12).cast(), ..vk::DeviceCreateInfo::builder().queue_create_infos(&queues).build() };
    // SAFETY: valid create info, outliving the extension; the chain is only read.
    let extension = unsafe { device_extend(gipa, g.instance.handle(), g.physical, &info) }.expect("the loader's node and an unknown one do not stop the additions");
    let (mut copied_loader, mut tail, mut v12_set) = (None, None, false);
    // SAFETY: the extended chain is valid while `extension` lives.
    unsafe {
        let mut node = extension.info.p_next.cast::<vk::BaseInStructure>();
        while !node.is_null() {
            match (*node).s_type {
                vk::StructureType::LOADER_DEVICE_CREATE_INFO => copied_loader = Some(node),
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => v12_set = (*node.cast::<vk::PhysicalDeviceVulkan12Features>()).buffer_device_address == 1,
                t if t == unknown.s_type => tail = Some(node),
                _ => {}
            }
            node = (*node).p_next;
        }
        let copied_loader = copied_loader.expect("the loader's node is in the chain");
        assert_ne!(copied_loader.cast::<LoaderNode>(), std::ptr::from_ref(&loader), "copied");
        assert_eq!((*copied_loader.cast::<LoaderNode>()).u, loader.u, "with its contents");
    }
    assert_eq!(tail.map(|t| t.cast::<Unknown>()), Some(std::ptr::from_ref(&unknown)), "the unknown structure is the application's own, linked as it is");
    assert!(v12_set);
    assert_eq!((v12.buffer_device_address, loader.p_next), (0, std::ptr::from_ref(&unknown).cast()), "nothing of the application's was written");
    drop(extension);
    // A structure the network sets after the unknown one: refused, not written.
    let mut after = vk::PhysicalDeviceVulkan12Features::default();
    let unknown_first = Unknown { s_type: unknown.s_type, p_next: std::ptr::from_mut(&mut after).cast(), payload: [0; 4] };
    let info = vk::DeviceCreateInfo { p_next: std::ptr::from_ref(&unknown_first).cast(), ..vk::DeviceCreateInfo::builder().queue_create_infos(&queues).build() };
    // SAFETY: as above.
    let refused = unsafe { device_extend(gipa, g.instance.handle(), g.physical, &info) };
    assert!(refused.is_err_and(|e| e.contains("after one this build does not know")));
    assert_eq!(after.buffer_device_address, 0);
    // SAFETY: nothing of the device is in use.
    unsafe {
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}
