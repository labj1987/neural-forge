//! The network through the C API on a real GPU, against `dlss5vk parity --dump` of the same frame:
//! the device made with `device_extend`, the network opened on a queue of a compute-only family (as
//! the layer opens it), the recorded graph run from a graphics-family queue's primary (buffers shared
//! between the two), and the head compared byte for byte.
//!
//! Runs only with `NEURAL_FORGE_NATIVE_RIG=<dump dir>` (holding `features.f32` and `head.f32` of a
//! 1486x836 frame); `NEURAL_FORGE_NATIVE_MODEL` overrides the model directory (default: the full
//! network's directory in the user data dir, `neural-forge/model/full`). The same variable runs the rebuild test (a small build, then a large one
//! on the same network), the fallback to barriers, the chain watchdog's count across builds, sizes that
//! cannot be built, the graph recorded into the application's buffer (`record_graph_at`) and the
//! device-creation tests. Each network drains the device before it closes ([`Drained`]). With
//! `NEURAL_FORGE_NATIVE_HEADS=<dir>` too, the heads at every size the layer builds are compared with a
//! saved run there (or saved, the first time): run once on a known-good build, then on the change.
//!
//! One network per process at a time (`nf_native_open` refuses a second): the tests take [`ONE`] so
//! the harness's threads run them one after another.

use ash::vk;
use neural_forge_native::{device_extend, device_supported, Frame, Network, OpenInfo, HOLD_SLOTS};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 1486;
const HEIGHT: u32 = 836;

/// Held by every test for its whole run: one network (and one rig device) at a time.
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// An instance kept for the whole process, never destroyed. Without it the loader unloads the NVIDIA
/// driver whenever a test destroys the last instance and loads it again for the next, and after about ten
/// such loads in one process the driver no longer loads at all ("libnvidia-tls.so: cannot allocate memory
/// in static TLS block"): the next test finds no NVIDIA GPU.
static KEEP_DRIVER_LOADED: std::sync::OnceLock<(ash::Entry, vk::Instance)> = std::sync::OnceLock::new();

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
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
    KEEP_DRIVER_LOADED.get_or_init(|| {
        // SAFETY: the system Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.expect("Vulkan loader");
        // SAFETY: valid create info.
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.unwrap();
        (entry, instance.handle())
    });
    // SAFETY: the system Vulkan loader.
    let entry = unsafe { ash::Entry::load() }.expect("Vulkan loader");
    // SAFETY: valid create info.
    let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.unwrap();
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
    unsafe { device_supported(gipa, instance.handle(), vk::API_VERSION_1_3, physical) }.expect("device support");
    let priorities = [1.0f32; 2];
    // The compute family's queue 0 is the network's own; queue 1 runs the recordings of that family.
    let queues = [
        vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities[..1]),
        vk::DeviceQueueCreateInfo::default().queue_family_index(compute).queue_priorities(&priorities),
    ];
    let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
    // SAFETY: valid create info, outlives the extension.
    let extension = unsafe { device_extend(gipa, instance.handle(), vk::API_VERSION_1_3, physical, &info) }.expect("extend");
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
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(bytes).usage(usage), None).unwrap();
            let req = self.device.get_buffer_memory_requirements(buffer);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let index = (0..self.memory.memory_type_count)
                .find(|&i| req.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(want))
                .unwrap();
            let memory = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None).unwrap();
            self.device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let ptr = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast();
            (buffer, memory, ptr)
        }
    }
}

/// Whether the networks are opened with counter chaining (`NEURAL_FORGE_NATIVE_CHAIN=0`: barriers).
fn chain_wanted() -> bool {
    std::env::var("NEURAL_FORGE_NATIVE_CHAIN").map_or(true, |v| v != "0")
}

/// A network that drains its device before it closes: dropped on any path (a failed assertion's unwinding
/// included), it waits for the device to be idle first, so nothing it recorded is pending when it is
/// destroyed (`Network::open`'s contract).
struct Drained<'a> {
    net: Network,
    device: &'a ash::Device,
}

impl Drop for Drained<'_> {
    fn drop(&mut self) {
        // SAFETY: the device is live (it outlives the network, whose field drops after this).
        let _ = unsafe { self.device.device_wait_idle() };
    }
}

impl std::ops::Deref for Drained<'_> {
    type Target = Network;
    fn deref(&self) -> &Network {
        &self.net
    }
}

impl std::ops::DerefMut for Drained<'_> {
    fn deref_mut(&mut self) -> &mut Network {
        &mut self.net
    }
}

/// The network on the rig's device, loading on the compute family's queue 0 (as the layer opens it).
fn open_network<'a>(g: &'a Gpu, model: &Path, chain: bool) -> Drained<'a> {
    let open = OpenInfo {
        gipa: g._entry.static_fn().get_instance_proc_addr,
        instance: g.instance.handle(),
        physical: g.physical,
        device: g.device.handle(),
        queue_family: g.compute,
        queue_index: 0,
        frame_family: g.family,
        model_dir: model,
        chain,
        fence_timeout_ms: 5000,
        init_dispatchable: None,
    };
    // SAFETY: the device was made with device_extend; the compute family's queue 0 is the network's alone.
    Drained { net: unsafe { Network::open(&open) }.expect("open"), device: &g.device }
}

/// Forgets a fall back to barriers when dropped: the tests after one that falls back (passing or
/// failing) open their networks as asked.
struct ForgetFallBack;

impl Drop for ForgetFallBack {
    fn drop(&mut self) {
        neural_forge_native::forget_fall_back();
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("NEURAL_FORGE_NATIVE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/neural-forge/model/full"))
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
    let mut net = open_network(&g, &model, chain_wanted());
    // SAFETY: nothing of a previous build exists.
    let small = unsafe { net.build(1280, 720) }.expect("the 1280x720 build");
    eprintln!("1280x720: {small:?}");
    // SAFETY: the small build's graph was never submitted.
    let large = unsafe { net.build(3840, 2160) }.expect("the 3840x2160 build after the 1280x720 one");
    eprintln!("3840x2160: {large:?}");
    assert_eq!(large.chained, u32::from(chain_wanted()), "chained as asked");
    assert!(large.field_width >= 3840 && large.field_height >= 2160);
    let d = &g.device;
    // SAFETY: live device and handles; the graph's frame completes before anything is destroyed.
    unsafe {
        let queue = d.get_device_queue(g.family, 0);
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family), None).unwrap();
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        d.cmd_fill_buffer(cmd, large.features, 0, vk::WHOLE_SIZE, 0);
        let to_compute = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[to_compute],
            &[],
            &[],
        );
        d.cmd_execute_commands(cmd, &[net.graph_commands()]);
        d.end_command_buffer(cmd).unwrap();
        d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
        d.wait_for_fences(&[fence], true, 5_000_000_000).expect("the 3840x2160 graph completes");
        assert_eq!(net.chain_timeouts().0, 0);
        d.destroy_fence(fence, None);
        d.destroy_command_pool(pool, None);
        drop(net);
        d.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// Two frame sizes built together (per-pass model resolution before the upscaler): both graphs run one
/// after the other in one submit, and the smaller one's head is the head of the same size built alone,
/// on the same features.
#[test]
fn two_sizes_built_together_each_give_the_head_of_their_size_alone() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let model = model_dir();
    let g = gpu();
    let mut net = open_network(&g, &model, chain_wanted());
    let d = &g.device;
    // Runs `graphs` (each with its features filled with a fixed pattern first) in one submit and returns each
    // head as read back.
    let run = |net: &Network, frames: &[(Frame, vk::CommandBuffer)]| -> Vec<Vec<f32>> {
        // SAFETY: live device and handles; the submit completes before anything is read or destroyed.
        unsafe {
            let queue = d.get_device_queue(g.family, 0);
            let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family), None).unwrap();
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let mut staging = Vec::new();
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            for (frame, graph) in frames {
                let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
                let (src, src_mem, src_ptr) = g.host_buffer(frame.features_bytes, usage);
                let values: Vec<f32> = (0..frame.features_bytes / 4).map(|i| ((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 0.25).collect();
                std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), src_ptr, values.len() * 4);
                let (dst, dst_mem, dst_ptr) = g.host_buffer(frame.head_bytes, usage);
                let all = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                d.cmd_copy_buffer(cmd, src, frame.features, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: frame.features_bytes }]);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[all],
                    &[],
                    &[],
                );
                d.cmd_execute_commands(cmd, &[*graph]);
                d.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[all],
                    &[],
                    &[],
                );
                d.cmd_copy_buffer(cmd, frame.head, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: frame.head_bytes }]);
                staging.push((src, src_mem, dst, dst_mem, dst_ptr, frame.head_bytes));
            }
            d.end_command_buffer(cmd).unwrap();
            d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            d.wait_for_fences(&[fence], true, 10_000_000_000).expect("the graphs complete");
            assert_eq!(net.chain_timeouts().0, 0);
            let heads = staging
                .iter()
                .map(|&(src, src_mem, dst, dst_mem, ptr, bytes)| {
                    let head = std::slice::from_raw_parts(ptr.cast::<f32>(), (bytes / 4) as usize).to_vec();
                    d.destroy_buffer(src, None);
                    d.free_memory(src_mem, None);
                    d.destroy_buffer(dst, None);
                    d.free_memory(dst_mem, None);
                    head
                })
                .collect();
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
            heads
        }
    };
    // SAFETY: nothing of a previous build exists.
    let both = unsafe { net.build_sizes(&[(960, 540), (480, 270)]) }.expect("the two sizes build together");
    assert!(both[0].field_width >= 960 && both[1].field_width >= 480 && both[1].field_width < both[0].field_width);
    assert!(both.iter().all(|f| f.chained == u32::from(chain_wanted())), "chained as asked");
    let together = run(&net.net, &[(both[0], net.graph_commands_at(0)), (both[1], net.graph_commands_at(1))]);
    assert!(together.iter().flatten().all(|v| v.is_finite()), "finite heads");
    // SAFETY: the two graphs' submit completed.
    let alone = unsafe { net.build(480, 270) }.expect("the small size alone");
    let single = run(&net.net, &[(alone, net.graph_commands())]);
    let worst = together[1].iter().zip(&single[0]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("480x270 built with 960x540 against alone: largest difference {worst}");
    assert!(worst <= 1e-3, "the smaller graph built beside the larger one gives its own size's head: {worst}");
    drop(net);
    // SAFETY: the network is closed; nothing is pending.
    unsafe {
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
    let t = std::time::Instant::now();
    let mut net = open_network(&g, &model, chain_wanted());
    let opened = t.elapsed();
    // SAFETY: nothing of a previous build exists.
    let frame = unsafe { net.build(WIDTH, HEIGHT) }.expect("build");
    eprintln!("open {:?}, build {:?}, frame {frame:?}", opened, t.elapsed() - opened);
    eprintln!("weights on the GPU: {:?}", net.weight_bytes());
    assert_eq!((frame.field_width, frame.field_height), (1536, 896));
    assert_eq!(frame.chained, u32::from(chain_wanted()), "chained as asked (a fall back left from another test would test only barriers)");
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
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family), None).unwrap();
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let queries = d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap();
        let period = g.instance.get_physical_device_properties(g.physical).limits.timestamp_period as f64;
        for run in 0..5 {
            std::ptr::write_bytes(down_ptr, 0, head_bytes as usize);
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            d.cmd_copy_buffer(cmd, up, frame.features, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: features.len() as u64 }]);
            let to_compute = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[to_compute],
                &[],
                &[],
            );
            d.cmd_reset_query_pool(cmd, queries, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            d.cmd_execute_commands(cmd, &[net.graph_commands()]);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
            let to_copy = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[to_copy],
                &[],
                &[],
            );
            d.cmd_copy_buffer(cmd, frame.head, down, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: head_bytes }]);
            d.end_command_buffer(cmd).unwrap();
            d.reset_fences(&[fence]).unwrap();
            let t = std::time::Instant::now();
            d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
            let mut stamps = [0u64; 2];
            d.get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
            eprintln!("run {run}: graph {:.3} ms GPU", (stamps[1] - stamps[0]) as f64 * period / 1e6);
            let got = std::slice::from_raw_parts(down_ptr, head_bytes as usize);
            let differing = got.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0.iter()).filter(|(a, b)| a != b).count();
            eprintln!(
                "run {run}: {:?} incl. copies, {differing} of {} head values differ, chain timeouts {:?}",
                t.elapsed(),
                expected.len() / 4,
                net.chain_timeouts()
            );
            assert_eq!(differing, 0, "the head differs from dlss5vk's");
            assert_eq!(net.chain_timeouts().0, 0);
        }
        // Back to back, the graph alone (the GPU at its working clocks, as dlss5vk bench measures).
        for primary in [false, true] {
            let mut times = Vec::new();
            for _ in 0..40 {
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                d.cmd_reset_query_pool(cmd, queries, 0, 2);
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
                if primary {
                    // SAFETY: `cmd` is a primary of the frame family being recorded here; the previous run's
                    // submit was waited on.
                    net.record_graph(cmd).unwrap();
                } else {
                    d.cmd_execute_commands(cmd, &[net.graph_commands()]);
                }
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
                d.end_command_buffer(cmd).unwrap();
                d.reset_fences(&[fence]).unwrap();
                d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
                d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
                let mut stamps = [0u64; 2];
                d.get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
                times.push((stamps[1] - stamps[0]) as f64 * period / 1e6);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "graph back to back ({}), 40 runs: min {:.3} ms, median {:.3} ms",
                if primary { "recorded into the primary" } else { "secondary" },
                times[0],
                times[20]
            );
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
    let _forget = ForgetFallBack;
    let model = model_dir();
    let g = gpu();
    let mut net = open_network(&g, &model, true);
    // SAFETY: nothing of a previous build exists, and nothing of one is ever submitted here.
    unsafe {
        assert_eq!(net.build(640, 360).expect("the first build").chained, 1, "recorded with chaining");
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

/// The head at every size the layer builds, on one network in the order the layer builds them (the
/// output size for the after-the-upscaler path first, then render sizes), with the dump's features
/// tiled over each field. With `NEURAL_FORGE_NATIVE_HEADS=<dir>` each head is compared byte for byte
/// with `head-<W>x<H>.f32` there, or written there when that file does not exist yet: the check that a
/// change to how the network is built (not what it computes) leaves every output byte as it was.
#[test]
fn the_heads_at_the_layers_sizes_match_a_saved_run() {
    let _one = one_at_a_time();
    let (Some(dump), Some(heads)) =
        (std::env::var_os("NEURAL_FORGE_NATIVE_RIG").map(PathBuf::from), std::env::var_os("NEURAL_FORGE_NATIVE_HEADS").map(PathBuf::from))
    else {
        eprintln!("NEURAL_FORGE_NATIVE_RIG or NEURAL_FORGE_NATIVE_HEADS unset: skipped");
        return;
    };
    let model = model_dir();
    let source = std::fs::read(dump.join("features.f32")).unwrap();
    let (source_width, source_height) = (1536usize, 896usize);
    assert_eq!(source.len(), source_width * source_height * 64);
    let g = gpu();
    let mut net = open_network(&g, &model, chain_wanted());
    let d = &g.device;
    let mut mismatches = Vec::new();
    for (width, height) in [(2560, 1440), (1486, 836), (3840, 2160), (2228, 1253)] {
        // SAFETY: the previous size's frame completed (waited on) before this build.
        let frame = unsafe { net.build(width, height) }.expect("build");
        assert_eq!(frame.chained, u32::from(chain_wanted()), "chained as asked: the saved heads are of the graph asked for");
        let (fw, fh) = (frame.field_width as usize, frame.field_height as usize);
        let mut features = vec![0u8; fw * fh * 64];
        for y in 0..fh {
            for x in 0..fw {
                let from = ((y % source_height) * source_width + (x % source_width)) * 64;
                features[(y * fw + x) * 64..][..64].copy_from_slice(&source[from..from + 64]);
            }
        }
        let head_bytes = (fw * fh * 16) as u64;
        let (up, up_mem, up_ptr) = g.host_buffer(features.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
        let (down, down_mem, down_ptr) = g.host_buffer(head_bytes, vk::BufferUsageFlags::TRANSFER_DST);
        // SAFETY: live device and handles; the frame completes before anything is destroyed.
        let got = unsafe {
            std::ptr::copy_nonoverlapping(features.as_ptr(), up_ptr, features.len());
            let queue = d.get_device_queue(g.family, 0);
            let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family), None).unwrap();
            let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            d.cmd_copy_buffer(cmd, up, frame.features, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: features.len() as u64 }]);
            let to_compute = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[to_compute],
                &[],
                &[],
            );
            d.cmd_execute_commands(cmd, &[net.graph_commands()]);
            let to_copy = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[to_copy],
                &[],
                &[],
            );
            d.cmd_copy_buffer(cmd, frame.head, down, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: head_bytes }]);
            d.end_command_buffer(cmd).unwrap();
            d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
            d.wait_for_fences(&[fence], true, 5_000_000_000).expect("the graph completes");
            let got = std::slice::from_raw_parts(down_ptr, head_bytes as usize).to_vec();
            d.destroy_fence(fence, None);
            d.destroy_command_pool(pool, None);
            for (b, m) in [(up, up_mem), (down, down_mem)] {
                d.destroy_buffer(b, None);
                d.free_memory(m, None);
            }
            got
        };
        assert_eq!(net.chain_timeouts().0, 0);
        let saved = heads.join(format!("head-{width}x{height}.f32"));
        match std::fs::read(&saved) {
            Ok(expected) => {
                let differing = got.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0.iter()).filter(|(a, b)| a != b).count();
                eprintln!(
                    "{width}x{height} (field {fw}x{fh}, chained {}): {differing} of {} head values differ from {}",
                    frame.chained,
                    got.len() / 4,
                    saved.display()
                );
                if differing != 0 || expected.len() != got.len() {
                    mismatches.push(format!("{width}x{height}"));
                }
            }
            Err(_) => {
                std::fs::write(&saved, &got).unwrap();
                eprintln!("{width}x{height} (field {fw}x{fh}): written to {}", saved.display());
            }
        }
    }
    drop(net);
    // SAFETY: nothing of the network is left.
    unsafe {
        d.destroy_device(None);
        g.instance.destroy_instance(None);
    }
    assert!(mismatches.is_empty(), "heads differ at {mismatches:?}");
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
    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(g.family).queue_priorities(&priorities)];
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
    let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).push_next(&mut v12).push_next(&mut v11);
    // SAFETY: valid create info, outliving the extension.
    let extension = unsafe { device_extend(gipa, g.instance.handle(), vk::API_VERSION_1_3, g.physical, &info) }.expect("extend");
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
    let with_ext = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&ext);
    // SAFETY: valid create info.
    let refused = unsafe { device_extend(gipa, g.instance.handle(), vk::API_VERSION_1_3, g.physical, &with_ext) };
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
    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(g.family).queue_priorities(&priorities)];
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
    let loader =
        LoaderNode { s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO, p_next: std::ptr::from_ref(&unknown).cast(), function: 0, u: 0x1234 as *const _ };
    let mut v12 = vk::PhysicalDeviceVulkan12Features { p_next: std::ptr::from_ref(&loader).cast_mut().cast(), ..Default::default() };
    let info = vk::DeviceCreateInfo { p_next: std::ptr::from_mut(&mut v12).cast(), ..vk::DeviceCreateInfo::default().queue_create_infos(&queues) };
    // SAFETY: valid create info, outliving the extension; the chain is only read.
    let extension = unsafe { device_extend(gipa, g.instance.handle(), vk::API_VERSION_1_3, g.physical, &info) }
        .expect("the loader's node and an unknown one do not stop the additions");
    let (mut copied_loader, mut tail, mut v12_set) = (None, None, false);
    // SAFETY: the extended chain is valid while `extension` lives.
    unsafe {
        let mut node = extension.info.p_next.cast::<vk::BaseInStructure>();
        while !node.is_null() {
            match (*node).s_type {
                vk::StructureType::LOADER_DEVICE_CREATE_INFO => copied_loader = Some(node),
                vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                    v12_set = (*node.cast::<vk::PhysicalDeviceVulkan12Features>()).buffer_device_address == 1
                }
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
    let info = vk::DeviceCreateInfo { p_next: std::ptr::from_ref(&unknown_first).cast(), ..vk::DeviceCreateInfo::default().queue_create_infos(&queues) };
    // SAFETY: as above.
    let refused = unsafe { device_extend(gipa, g.instance.handle(), vk::API_VERSION_1_3, g.physical, &info) };
    assert!(refused.is_err_and(|e| e.contains("after one this build does not know")));
    assert_eq!(after.buffer_device_address, 0);
    // SAFETY: nothing of the device is in use.
    unsafe {
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// Bytes of a fixed pattern of small values (the same for every run): features for a size the dump has none of.
fn pattern(bytes: u64) -> Vec<u8> {
    (0..bytes / 4).flat_map(|i| (((i * 2654435761 % 1000) as f32 / 1000.0 - 0.5) * 0.25).to_ne_bytes()).collect()
}

/// One submit on `queue` from `cmd` (begun here, which resets it: its pool allows that): `features` copied
/// into `frame.features`, then what `record` records (the network's commands), then the head copied back.
/// The head's bytes, once the submit completed.
///
/// # Safety
/// `cmd` is a primary of `queue`'s family, not pending; what `record` records is valid there.
unsafe fn run_head(g: &Gpu, queue: vk::Queue, cmd: vk::CommandBuffer, frame: &Frame, features: &[u8], record: impl FnOnce(vk::CommandBuffer)) -> Vec<u8> {
    assert_eq!(features.len() as u64, frame.features_bytes);
    let d = &g.device;
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let (up, up_mem, up_ptr) = g.host_buffer(frame.features_bytes, usage);
    let (down, down_mem, down_ptr) = g.host_buffer(frame.head_bytes, usage);
    // SAFETY: live device and handles (this function's contract); the submit completes before anything is read
    // or destroyed.
    unsafe {
        std::ptr::copy_nonoverlapping(features.as_ptr(), up_ptr, features.len());
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let all = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        let stages = vk::PipelineStageFlags::ALL_COMMANDS;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        d.cmd_copy_buffer(cmd, up, frame.features, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: frame.features_bytes }]);
        d.cmd_pipeline_barrier(cmd, stages, stages, vk::DependencyFlags::empty(), &[all], &[], &[]);
        record(cmd);
        d.cmd_pipeline_barrier(cmd, stages, stages, vk::DependencyFlags::empty(), &[all], &[], &[]);
        d.cmd_copy_buffer(cmd, frame.head, down, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: frame.head_bytes }]);
        d.end_command_buffer(cmd).unwrap();
        d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence).unwrap();
        d.wait_for_fences(&[fence], true, 10_000_000_000).expect("the submit completes");
        let head = std::slice::from_raw_parts(down_ptr, frame.head_bytes as usize).to_vec();
        d.destroy_fence(fence, None);
        for (b, m) in [(up, up_mem), (down, down_mem)] {
            d.destroy_buffer(b, None);
            d.free_memory(m, None);
        }
        head
    }
}

/// Measurement only, with `NEURAL_FORGE_NATIVE_PROFILE=1` too: the graph at 1486x836 on frame d1's features,
/// recorded straight into a primary with a GPU timestamp after every dispatch, 20 times; prints each kernel's
/// median time (per block and kernel, in recording order), each block's and the total. Timestamps after
/// chained dispatches that overlap give each the time from the previous one's end to its own: run with
/// `NEURAL_FORGE_NATIVE_CHAIN=0` too for each kernel alone.
#[test]
fn the_graph_profiled_per_kernel() {
    let (Some(dump), true) = (std::env::var_os("NEURAL_FORGE_NATIVE_RIG").map(PathBuf::from), std::env::var_os("NEURAL_FORGE_NATIVE_PROFILE").is_some()) else {
        eprintln!("NEURAL_FORGE_NATIVE_RIG or NEURAL_FORGE_NATIVE_PROFILE unset: skipped");
        return;
    };
    let _one = one_at_a_time();
    let features = std::fs::read(dump.join("features.f32")).unwrap();
    let g = gpu();
    let mut net = open_network(&g, &model_dir(), chain_wanted());
    // SAFETY: nothing of a previous build exists.
    let frame = unsafe { net.build(WIDTH, HEIGHT) }.expect("build");
    let d = &g.device;
    const RUNS: usize = 20;
    let mut runs: Vec<Vec<(String, f64)>> = Vec::new();
    // SAFETY: live device and handles; each submit completes (run_head waits) before its profile is read.
    unsafe {
        let queue = d.get_device_queue(g.family, 0);
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family), None).unwrap();
        let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        for _ in 0..RUNS + 2 {
            run_head(&g, queue, cmd, &frame, &features, |cmd| net.record_graph_profiled(cmd, 4096).expect("record"));
            runs.push(net.profile());
        }
        d.destroy_command_pool(pool, None);
    }
    // The first two warm the clocks and the caches.
    let runs = &runs[2..];
    assert!(runs.iter().all(|r| r.len() == runs[0].len() && !r.is_empty()), "every run times the same dispatches");
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let per_dispatch: Vec<(String, f64)> = (0..runs[0].len()).map(|i| (runs[0][i].0.clone(), median(runs.iter().map(|r| r[i].1).collect()))).collect();
    let total = median(runs.iter().map(|r| r.iter().map(|(_, ms)| ms).sum()).collect());
    let mut by_kernel: Vec<(String, f64, usize)> = Vec::new();
    for (label, ms) in &per_dispatch {
        let kernel = label.split_once(" | ").map_or(label.as_str(), |(_, k)| k).to_string();
        match by_kernel.iter_mut().find(|(k, _, _)| *k == kernel) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => by_kernel.push((kernel, *ms, 1)),
        }
    }
    by_kernel.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut by_block: Vec<(String, f64)> = Vec::new();
    for (label, ms) in &per_dispatch {
        let block = label.split_once(" | ").map_or("", |(b, _)| b).to_string();
        match by_block.last_mut() {
            Some(e) if e.0 == block => e.1 += ms,
            _ => by_block.push((block, *ms)),
        }
    }
    eprintln!("profile: {} dispatches, total {total:.3} ms (median of {RUNS}), chained {}", per_dispatch.len(), frame.chained);
    for (kernel, ms, n) in &by_kernel {
        eprintln!("kernel {ms:8.3} ms {:5.1}% x{n:<4} {kernel}", ms / total * 100.0);
    }
    for (block, ms) in &by_block {
        eprintln!("block {ms:8.3} ms {block}");
    }
    for (label, ms) in &per_dispatch {
        eprintln!("dispatch {ms:8.4} ms {label}");
    }
}

/// This process's use of the GPU's device-local heaps (`VK_EXT_memory_budget`), in bytes.
fn device_local_usage(g: &Gpu) -> u64 {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: live instance and physical device; the chain is valid for the call.
    unsafe { g.instance.get_physical_device_memory_properties2(g.physical, &mut props) };
    let heaps = props.memory_properties.memory_heaps;
    let count = props.memory_properties.memory_heap_count as usize;
    (0..count).filter(|&i| heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)).map(|i| budget.heap_usage[i]).sum()
}

/// Rebuilding the network for another size returns the last build's memory: ten switches between
/// 1486x836 and 2560x1440 end within 32 MiB of where the first pair did (AMDNR measured another
/// runtime keeping 75-350 MB per new size).
#[test]
fn rebuilds_at_other_sizes_return_their_memory() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let g = gpu();
    let mut net = open_network(&g, &model_dir(), chain_wanted());
    let mut after_pair = Vec::new();
    for _ in 0..10 {
        for (w, h) in [(1486, 836), (2560, 1440)] {
            // SAFETY: no graph of an earlier build was submitted.
            unsafe { net.build(w, h) }.expect("build");
        }
        after_pair.push(device_local_usage(&g));
    }
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    eprintln!("device-local use after each pair: {:?} MiB", after_pair.iter().map(|&b| mib(b).round()).collect::<Vec<_>>());
    let growth = after_pair[after_pair.len() - 1].saturating_sub(after_pair[0]);
    assert!(growth < 32 << 20, "ten rebuild pairs kept {:.0} MiB", mib(growth));
}

/// What sharing the activations saves (`NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS`, NATIVE_BACKEND.md "Activation
/// memory"): the device-local memory a build takes, with and without, for the size sets the layer builds at 1440p
/// and 4K (each output with DLSS's render size at Quality). Shared never takes more.
#[test]
fn shared_activations_take_less_memory() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let g = gpu();
    let mut net = open_network(&g, &model_dir(), chain_wanted());
    let mib = |b: u64| (b as f64 / (1u64 << 20) as f64).round();
    for sizes in [&[(2560, 1440), (1708, 960)][..], &[(3840, 2160), (2560, 1440)][..]] {
        let mut taken = [0u64; 2];
        for (i, shared) in [false, true].into_iter().enumerate() {
            let _shared = SharedActivations::set(shared);
            // SAFETY: no graph of the previous build was submitted. A tiny build first returns the last one's memory.
            unsafe { net.build(64, 64) }.expect("the small build");
            let base = device_local_usage(&g);
            // SAFETY: as above.
            unsafe { net.build_sizes(sizes) }.expect("build");
            taken[i] = device_local_usage(&g).saturating_sub(base);
        }
        eprintln!("{sizes:?}: own allocations {} MiB, shared {} MiB", mib(taken[0]), mib(taken[1]));
        assert!(taken[1] <= taken[0]);
    }
}

/// `NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS` set for the builds while this lives, and removed when it is dropped on
/// any path (a failed assertion's unwinding included): the tests after one that sets it build as by default.
struct SharedActivations;

impl SharedActivations {
    fn set(shared: bool) -> Self {
        // SAFETY: the variable is read by the builds of the test holding [`ONE`] (as every test here does), on its
        // thread; no other test runs meanwhile.
        unsafe { std::env::set_var("NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS", if shared { "1" } else { "0" }) };
        SharedActivations
    }
}

impl Drop for SharedActivations {
    fn drop(&mut self) {
        // SAFETY: as in `set`.
        unsafe { std::env::remove_var("NEURAL_FORGE_NATIVE_SHARED_ACTIVATIONS") };
    }
}

/// How many 4-byte values of `got` differ from `expected` (all of them when the lengths differ).
fn differing(got: &[u8], expected: &[u8]) -> usize {
    if got.len() != expected.len() {
        return got.len().max(expected.len()) / 4;
    }
    got.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0.iter()).filter(|(a, b)| a != b).count()
}

/// The hold inside DLSS's command buffer (`record_graph_at`): the graph recorded straight into a primary of
/// the frame family, through two hold slots, gives dlss5vk's head as the secondary does; the secondary still
/// does after them (the holds' descriptor sets are their own pools', not the frame recordings'); a hold slot
/// recorded again once its buffer completed (its pool reset) gives it again, and so does the recording for
/// the network's own family. A second size's hold gives that size's secondary's head. An unknown hold slot or
/// size is refused.
#[test]
fn the_graph_recorded_into_the_applications_buffer_gives_the_secondarys_head() {
    let _one = one_at_a_time();
    let Some(dump) = std::env::var_os("NEURAL_FORGE_NATIVE_RIG").map(PathBuf::from) else {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    };
    let model = model_dir();
    let features = std::fs::read(dump.join("features.f32")).unwrap();
    let expected = std::fs::read(dump.join("head.f32")).unwrap();
    let g = gpu();
    let mut net = open_network(&g, &model, chain_wanted());
    // SAFETY: nothing of a previous build exists.
    let frames = unsafe { net.build_sizes(&[(WIDTH, HEIGHT), (640, 360)]) }.expect("build");
    assert!(frames.iter().all(|f| f.chained == u32::from(chain_wanted())), "chained as asked");
    let d = &g.device;
    // SAFETY: live device and handles; every submit completes (run_head waits) before the next records over
    // what it used, and before anything is destroyed.
    unsafe {
        let queue = d.get_device_queue(g.family, 0);
        let own_queue = d.get_device_queue(g.compute, 1);
        let reset = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
        let pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.family).flags(reset), None).unwrap();
        let own_pool = d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(g.compute).flags(reset), None).unwrap();
        let cmds = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(3)).unwrap();
        let own = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(own_pool).command_buffer_count(1)).unwrap()[0];
        let check = |what: &str, net: &Network, head: &[u8]| {
            let n = differing(head, &expected);
            eprintln!("{what}: {n} of {} head values differ from dlss5vk's", expected.len() / 4);
            assert_eq!(n, 0, "{what}: the head differs from dlss5vk's");
            assert_eq!(net.chain_timeouts().0, 0, "{what}: no counter wait gave up");
        };
        let head = run_head(&g, queue, cmds[0], &frames[0], &features, |cmd| net.record_graph_at(0, 0, cmd).expect("hold slot 0"));
        check("hold slot 0", &net, &head);
        let head = run_head(&g, queue, cmds[1], &frames[0], &features, |cmd| net.record_graph_at(0, 1, cmd).expect("hold slot 1"));
        check("hold slot 1", &net, &head);
        let head = run_head(&g, queue, cmds[2], &frames[0], &features, |cmd| d.cmd_execute_commands(cmd, &[net.graph_commands_at(0)]));
        check("the secondary after the holds", &net, &head);
        let head = run_head(&g, queue, cmds[0], &frames[0], &features, |cmd| net.record_graph_at(0, 0, cmd).expect("hold slot 0 again"));
        check("hold slot 0 recorded again", &net, &head);
        let head = run_head(&g, own_queue, own, &frames[0], &features, |cmd| d.cmd_execute_commands(cmd, &[net.graph_commands_own_at(0)]));
        check("the recording for the network's own family", &net, &head);
        // The second size: its hold against its secondary, on the same features.
        let small = pattern(frames[1].features_bytes);
        let held = run_head(&g, queue, cmds[0], &frames[1], &small, |cmd| net.record_graph_at(1, HOLD_SLOTS - 1, cmd).expect("the second size's hold"));
        let secondary = run_head(&g, queue, cmds[1], &frames[1], &small, |cmd| d.cmd_execute_commands(cmd, &[net.graph_commands_at(1)]));
        assert!(held.as_chunks::<4>().0.iter().all(|v| f32::from_ne_bytes(*v).is_finite()), "a finite head");
        assert_eq!(differing(&held, &secondary), 0, "the second size's hold gives its secondary's head");
        // Refused before anything is recorded.
        d.begin_command_buffer(cmds[2], &vk::CommandBufferBeginInfo::default()).unwrap();
        let why = net.record_graph_at(0, HOLD_SLOTS, cmds[2]).expect_err("no hold slot past the last");
        assert!(why.contains("no such hold slot"), "{why}");
        let why = net.record_graph_at(2, 0, cmds[2]).expect_err("no third size");
        assert!(why.contains("not built for that size"), "{why}");
        d.end_command_buffer(cmds[2]).unwrap();
        d.destroy_command_pool(own_pool, None);
        d.destroy_command_pool(pool, None);
        drop(net);
        d.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// A counter-chain timeout in the graph's frames that nothing read before the next build (a size change
/// coming before the hold that checks the count) switches that build to barriers instead of failing it, and
/// leaves no count behind: every build after it succeeds.
#[test]
fn a_chain_timeout_left_from_the_frames_does_not_fail_the_next_builds() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let _forget = ForgetFallBack;
    let model = model_dir();
    let g = gpu();
    let mut net = open_network(&g, &model, true);
    // SAFETY: nothing of a build is ever submitted here.
    unsafe {
        assert_eq!(net.build(640, 360).expect("the first build").chained, 1, "recorded with chaining");
        net.fake_chain_timeout(false);
        assert_eq!(net.chain_timeout_count(), 1);
        let rebuilt = net.build(800, 448).expect("the build after a timeout in the frames");
        assert_eq!(rebuilt.chained, 0, "fell back to barriers");
        assert_eq!(net.chain_timeout_count(), 0, "no count left behind");
        assert_eq!(net.build(640, 360).expect("a build after that").chained, 0);
        drop(net);
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// A warm-up's own counter-chain timeout: the build falls back to barriers and records with them, and the
/// builds after it succeed.
#[test]
fn a_warm_up_chain_timeout_falls_back_to_barriers() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let _forget = ForgetFallBack;
    let model = model_dir();
    let g = gpu();
    let mut net = open_network(&g, &model, true);
    // SAFETY: nothing of a build is ever submitted here.
    unsafe {
        net.fake_chain_timeout(true);
        let frame = net.build_sizes(&[(800, 448), (640, 360)]).expect("built with barriers after the warm-up's timeout");
        assert!(frame.iter().all(|f| f.chained == 0), "recorded with barriers");
        assert_eq!(net.chain_timeout_count(), 0, "no count left behind");
        assert_eq!(net.build(640, 360).expect("a build after that").chained, 0);
        drop(net);
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}

/// Sizes that cannot be built (none, a zero width or height) are refused with the current build kept: a
/// refused call frees nothing the caller had no reason to drain first.
#[test]
fn invalid_sizes_keep_the_current_build() {
    if std::env::var_os("NEURAL_FORGE_NATIVE_RIG").is_none() {
        eprintln!("NEURAL_FORGE_NATIVE_RIG unset: skipped");
        return;
    }
    let _one = one_at_a_time();
    let model = model_dir();
    let g = gpu();
    let mut net = open_network(&g, &model, chain_wanted());
    // SAFETY: nothing of a build is ever submitted here.
    unsafe {
        let built = net.build(640, 360).expect("build");
        let graph = net.graph_commands();
        let cases: [&[(u32, u32)]; 4] = [&[], &[(0, 360)], &[(640, 0)], &[(640, 360), (0, 0)]];
        for sizes in cases {
            let why = net.build_sizes(sizes).expect_err("refused");
            assert!(why.starts_with("no "), "{sizes:?}: {why}");
            let kept = net.frame().expect("the build is kept");
            assert_eq!((kept.features, kept.head, kept.field_width, kept.field_height), (built.features, built.head, built.field_width, built.field_height));
            assert_eq!(net.graph_commands(), graph, "{sizes:?}: the recording is kept");
        }
        assert!(net.build(0, 0).is_err());
        assert!(net.frame().is_some(), "kept after a zero build too");
        drop(net);
        g.device.destroy_device(None);
        g.instance.destroy_instance(None);
    }
}
