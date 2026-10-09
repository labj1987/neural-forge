//! Frame-series capture: `capture_request = N` with N above 1 dumps the next N presented
//! frames as matched original/composited PNG pairs, for measuring how well the model's edit
//! lands frame by frame (`scripts/agreement.py`, `crates/layer/examples/pan.rs`).
//!
//! The one-shot dump (`capture_request = 1`, [`crate::dump`]) is untouched. It goes through
//! `capture::run_sync`, which is neither the default synchronous present's composition nor
//! the pipelined one, so it cannot compare the two. A series is path-agnostic instead: it reads
//! the swapchain image back just before `capture::run` (the game's frame) and again just after
//! it (exactly what is presented, whichever path composed it), and leaves the present path
//! itself alone.
//!
//! Cost while a series runs: two blocking image readbacks per present and two PNG encodes,
//! done on writer threads. The frames are paced a little slower than without a capture; the
//! writers are bounded, so a backlog blocks the present thread rather than growing memory
//! without limit. The series ends by waiting for every file to be written.

use ash::vk;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// What the writers hold at most before the present thread waits on them.
const BACKLOG_BYTES: usize = 1536 << 20;
const WRITERS: usize = 4;

struct Job {
    seq: u32,
    original: Vec<u8>,
    composited: Vec<u8>,
    width: u32,
    height: u32,
    bgr_order: bool,
}

/// Host-visible copy target for one readback.
struct Readback {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *const u8,
    coherent: bool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

struct Resources {
    pool: vk::CommandPool,
    queue_family: u32,
    bytes: u64,
    original: Readback,
    composited: Readback,
}

/// A running series: where it writes, how many frames are left, and its writers.
struct Active {
    dir: PathBuf,
    remaining: u32,
    seq: u32,
    started: std::time::Instant,
    index: Option<std::io::BufWriter<std::fs::File>>,
    sender: Option<mpsc::SyncSender<Job>>,
    writers: Vec<std::thread::JoinHandle<u32>>,
    /// Set by [`Series::before`] when the original readback is on the queue.
    original_pending: bool,
}

#[derive(Default)]
pub struct Series {
    active: Option<Active>,
    resources: Option<Resources>,
}

// SAFETY: the raw mapped pointers are only dereferenced on the thread holding the device
// state's `Mutex`, like every other Vulkan resource in that state.
unsafe impl Send for Series {}

/// The capture request this present takes, as a series length. Normally only a series
/// (`capture_request` above 1): the one-shot (1) belongs to `capture::run`. While frames are `held`
/// before the upscaler (`crate::preupscale`) the post path does not run, so the present takes the
/// one-shot as well, as a series of one frame; otherwise the request would stay pending.
pub fn take_request(shm: &crate::shm::ShmClient, held: bool) -> Option<u32> {
    shm.take_series_request().or_else(|| (held && shm.take_capture_request()).then_some(1))
}

impl Series {
    /// Starts a series of `frames` presents, written under `base/series-<ms>/`.
    pub fn start(&mut self, frames: u32, base: &Path) {
        if self.active.is_some() {
            crate::log!("[series] a series is already running; request for {frames} frames ignored");
            return;
        }
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
        let dir = base.join(format!("series-{stamp}"));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            crate::log!("[series] failed to create {}: {}", dir.display(), e);
            return;
        }
        let index = std::fs::File::create(dir.join("index.tsv")).ok().map(std::io::BufWriter::new);
        crate::log!("[series] capturing the next {frames} presents into {}", dir.display());
        crate::logging::flush();
        self.active = Some(Active {
            dir,
            remaining: frames,
            seq: 0,
            started: std::time::Instant::now(),
            index,
            sender: None,
            writers: Vec::new(),
            original_pending: false,
        });
    }

    pub fn running(&self) -> bool {
        self.active.is_some()
    }

    /// Queues the readback of `image` as this present's original. Call before anything of the
    /// layer's own is submitted for this present; `image` is in `PRESENT_SRC_KHR` and every
    /// earlier submission on `queue` (the application's, via the relay) is ordered before it.
    ///
    /// # Safety
    /// `queue` is externally synchronized for this call and the following [`Self::after`], and
    /// belongs to `queue_family`; `image` is a live `width`x`height` 4-byte-per-pixel image
    /// created with `TRANSFER_SRC` usage.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn before(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family: u32,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let Some(active) = self.active.as_mut() else { return };
        active.original_pending = false;
        let bytes = u64::from(width) * u64::from(height) * 4;
        if self.resources.as_ref().is_some_and(|r| r.bytes != bytes || r.queue_family != queue_family) {
            // SAFETY: every submission on these resources was waited on in `after`.
            unsafe { destroy_resources(self.resources.take(), device) };
        }
        if self.resources.is_none() {
            self.resources = unsafe { create_resources(device, instance, physical_device, queue_family, bytes) };
        }
        let Some(r) = self.resources.as_ref() else {
            crate::log!("[series] could not create readback resources; series abandoned");
            self.finish();
            return;
        };
        // SAFETY: per this function's contract.
        active.original_pending = unsafe { submit_copy(device, queue, &r.original, image, width, height, None) };
    }

    /// Reads `image` back as this present's composited result, waiting on `compose_done` (the
    /// semaphore `capture::run` returned, if any) first, and hands the pair to the writers.
    /// Returns `true` when `compose_done` was waited on here, so the real present must no
    /// longer wait on it (a binary semaphore is consumed by one wait).
    ///
    /// # Safety
    /// Same as [`Self::before`], called once after it for the same present.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn after(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        image: vk::Image,
        width: u32,
        height: u32,
        bgr_order: bool,
        compose_done: Option<vk::Semaphore>,
    ) -> bool {
        let Some(active) = self.active.as_mut() else { return false };
        let Some(r) = self.resources.as_ref() else { return false };
        if !active.original_pending {
            return false;
        }
        active.original_pending = false;
        // SAFETY: per this function's contract.
        let submitted = unsafe { submit_copy(device, queue, &r.composited, image, width, height, compose_done) };
        let original_done = unsafe { wait_and_reset(device, &r.original) };
        let composited_done = submitted && unsafe { wait_and_reset(device, &r.composited) };
        if !(original_done && composited_done) {
            crate::log!("[series] readback failed on frame {}; series abandoned", active.seq);
            self.finish();
            // A fence that never signalled leaves its resources in an unknown state: they are
            // leaked rather than freed under the GPU or reused with a pending fence.
            self.resources = None;
            return submitted;
        }
        let len = (u64::from(width) * u64::from(height) * 4) as usize;
        // SAFETY: both fences are signalled, the buffers are `len` bytes and stay mapped.
        let (original, composited) = unsafe {
            invalidate(device, &r.original);
            invalidate(device, &r.composited);
            (std::slice::from_raw_parts(r.original.ptr, len).to_vec(), std::slice::from_raw_parts(r.composited.ptr, len).to_vec())
        };
        let seq = active.seq;
        if let Some(index) = active.index.as_mut() {
            if seq == 0 {
                // `gpu_composed`: the GPU composition ran for this present (it returned a
                // semaphore). 0 is a frame that went out untouched, or the rare CPU fallback.
                let _ = writeln!(index, "seq\tms\tgpu_composed\twidth\theight");
            }
            let _ = writeln!(index, "{seq}\t{:.3}\t{}\t{width}\t{height}", active.started.elapsed().as_secs_f64() * 1000.0, u8::from(compose_done.is_some()));
        }
        if active.sender.is_none() {
            let bound = (BACKLOG_BYTES / (2 * len.max(1))).max(2);
            let (sender, receiver) = mpsc::sync_channel::<Job>(bound);
            let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
            for _ in 0..WRITERS {
                let receiver = receiver.clone();
                let dir = active.dir.clone();
                active.writers.push(std::thread::spawn(move || {
                    let mut failures = 0;
                    loop {
                        let job = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok(job) = job else { return failures };
                        if !write_job(&dir, &job) {
                            failures += 1;
                        }
                    }
                }));
            }
            active.sender = Some(sender);
        }
        if let Some(sender) = &active.sender {
            let _ = sender.send(Job { seq, original, composited, width, height, bgr_order });
        }
        active.seq += 1;
        active.remaining = active.remaining.saturating_sub(1);
        if active.remaining == 0 {
            self.finish();
        }
        submitted
    }

    /// Ends the series: waits for every queued pair to be written, then says where they are.
    fn finish(&mut self) {
        self.finish_within(None);
    }

    /// [`Self::finish`], but gives the writers at most `limit`: past it they are left to finish
    /// on their own threads (the process may exit first, losing the pairs still queued).
    fn finish_within(&mut self, limit: Option<std::time::Duration>) {
        let Some(mut active) = self.active.take() else { return };
        if let Some(mut index) = active.index.take() {
            let _ = index.flush();
        }
        drop(active.sender.take());
        let deadline = limit.map(|l| std::time::Instant::now() + l);
        while let Some(deadline) = deadline {
            if active.writers.iter().all(|w| w.is_finished()) || std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let (done, running): (Vec<_>, Vec<_>) = active.writers.drain(..).partition(|w| deadline.is_none() || w.is_finished());
        let failures: u32 = done.into_iter().map(|w| w.join().unwrap_or(1)).sum();
        if running.is_empty() {
            crate::log!("[series] wrote {} frame pairs into {} ({} failed)", active.seq, active.dir.display(), failures);
        } else {
            crate::log!(
                "[series] {} writers still busy at teardown: {} frame pairs captured into {}, the last ones may be missing",
                running.len(), active.seq, active.dir.display()
            );
        }
        crate::logging::flush();
    }

    /// # Safety
    /// Nothing submitted through this series may still be pending (it never is after
    /// [`Self::after`] returns).
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        // Device teardown holds the layer's state lock: a game quitting mid-series must not
        // wait out a long PNG backlog.
        self.finish_within(Some(std::time::Duration::from_secs(2)));
        unsafe { destroy_resources(self.resources.take(), device) };
    }
}

/// The file names one series frame is written under.
pub fn pair_names(seq: u32) -> (String, String) {
    (format!("{seq:06}-original.png"), format!("{seq:06}-composited.png"))
}

fn write_job(dir: &Path, job: &Job) -> bool {
    let (original, composited) = pair_names(job.seq);
    let a = crate::dump::write_png(&dir.join(original), &job.original, job.width, job.height, job.bgr_order, true);
    let b = crate::dump::write_png(&dir.join(composited), &job.composited, job.width, job.height, job.bgr_order, true);
    a && b
}

unsafe fn create_readback(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
    bytes: u64,
) -> Option<Readback> {
    let info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let find = |wanted: vk::MemoryPropertyFlags| {
        (0..mem_props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted))
    };
    // Cached host memory first: reading uncached memory back on the CPU is slow.
    let Some(type_index) = find(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| find(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)) else {
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };
    let coherent = mem_props.memory_types[type_index as usize].property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
    let memory = match unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) } {
        Ok(m) => m,
        Err(_) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return None;
        }
    };
    let cleanup = || unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
    };
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        cleanup();
        return None;
    }
    let Ok(ptr) = (unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }) else {
        cleanup();
        return None;
    };
    let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
    // The buffer is submitted through the loader's trampolines: it needs the device's loader data,
    // like every buffer the layer allocates.
    let Some(cmd) = unsafe { crate::loader_data::allocate_commands(device, &alloc) }.ok().and_then(|v| v.first().copied()) else {
        cleanup();
        return None;
    };
    let Ok(fence) = (unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }) else {
        cleanup();
        return None;
    };
    Some(Readback { buffer, memory, ptr: ptr.cast(), coherent, cmd, fence })
}

unsafe fn create_resources(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: u64,
) -> Option<Resources> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = unsafe { device.create_command_pool(&pool_info, None) }.ok()?;
    let original = unsafe { create_readback(device, &mem_props, pool, bytes) };
    let composited = unsafe { create_readback(device, &mem_props, pool, bytes) };
    match (original, composited) {
        (Some(original), Some(composited)) => Some(Resources { pool, queue_family, bytes, original, composited }),
        (a, b) => {
            for r in [a, b].into_iter().flatten() {
                unsafe { destroy_readback(device, r) };
            }
            unsafe { device.destroy_command_pool(pool, None) };
            None
        }
    }
}

unsafe fn destroy_readback(device: &ash::Device, r: Readback) {
    unsafe {
        device.destroy_fence(r.fence, None);
        device.destroy_buffer(r.buffer, None);
        device.free_memory(r.memory, None);
    }
}

unsafe fn destroy_resources(resources: Option<Resources>, device: &ash::Device) {
    let Some(r) = resources else { return };
    unsafe {
        destroy_readback(device, r.original);
        destroy_readback(device, r.composited);
        // Frees the command buffers with it.
        device.destroy_command_pool(r.pool, None);
    }
}

/// Records and submits "`image` (`PRESENT_SRC_KHR`) -> `r.buffer`, back to `PRESENT_SRC_KHR`",
/// optionally waiting on `wait` at the transfer stage, signalling `r.fence`.
unsafe fn submit_copy(device: &ash::Device, queue: vk::Queue, r: &Readback, image: vk::Image, width: u32, height: u32, wait: Option<vk::Semaphore>) -> bool {
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let to_src = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build();
    let to_present = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::empty())
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .build();
    let to_host = vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(r.buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        .build();
    let region = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width, height, depth: 1 })
        .build();
    let begin = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        if device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()).is_err() || device.begin_command_buffer(r.cmd, &begin).is_err() {
            return false;
        }
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
        device.cmd_copy_image_to_buffer(r.cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, r.buffer, &[region]);
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[], &[to_host], &[to_present]);
        if device.end_command_buffer(r.cmd).is_err() {
            return false;
        }
        let stage = [vk::PipelineStageFlags::TRANSFER];
        let waits: Vec<vk::Semaphore> = wait.into_iter().collect();
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stage[..waits.len()])
            .command_buffers(std::slice::from_ref(&r.cmd))
            .build();
        device.queue_submit(queue, &[submit], r.fence).is_ok()
    }
}

/// Waits (bounded) for `r`'s submission, then resets its fence for the next one.
unsafe fn wait_and_reset(device: &ash::Device, r: &Readback) -> bool {
    unsafe {
        let ok = device.wait_for_fences(&[r.fence], true, 5_000_000_000).is_ok();
        ok && device.reset_fences(&[r.fence]).is_ok()
    }
}

unsafe fn invalidate(device: &ash::Device, r: &Readback) {
    if !r.coherent {
        let range = vk::MappedMemoryRange::builder().memory(r.memory).offset(0).size(vk::WHOLE_SIZE).build();
        unsafe {
            let _ = device.invalidate_mapped_memory_ranges(&[range]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear(device: &ash::Device, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image, from: vk::ImageLayout, colour: [f32; 4]) {
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let barrier = |old, new, src, dst| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(src)
                .dst_access_mask(dst)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .build()
        };
        unsafe {
            let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier(from, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::MEMORY_READ, vk::AccessFlags::TRANSFER_WRITE)]);
            device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: colour }, &[range]);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[barrier(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::MEMORY_READ)]);
            device.end_command_buffer(cmd).unwrap();
            // No fence: queue order alone has to keep the series' readbacks around this.
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], vk::Fence::null()).unwrap();
        }
    }

    /// The one-shot stays `capture::run`'s on the post path, and is taken (as one frame) while
    /// frames are held before the upscaler; a series is taken either way, once.
    #[test]
    fn requests_are_taken_by_the_post_path_and_while_held() {
        let header = Box::new(neural_forge_protocol::ShmHeader::default());
        let shm = crate::shm::ShmClient::test_over_header(&header);
        let field = &header.capture_request;
        use std::sync::atomic::Ordering::Relaxed;
        assert_eq!(take_request(&shm, false), None);
        assert_eq!(take_request(&shm, true), None);
        field.store(1, Relaxed);
        assert_eq!(take_request(&shm, false), None, "the one-shot is left for capture::run");
        assert_eq!(field.load(Relaxed), 1);
        assert_eq!(take_request(&shm, true), Some(1), "held: the one-shot is served as one frame");
        assert_eq!(field.load(Relaxed), 0);
        assert_eq!(take_request(&shm, true), None, "served once");
        field.store(30, Relaxed);
        assert_eq!(take_request(&shm, true), Some(30));
        field.store(2, Relaxed);
        assert_eq!(take_request(&shm, false), Some(2));
        assert_eq!(field.load(Relaxed), 0);
    }

    /// A series of three presents where something (standing in for the composition) rewrites
    /// the image between `before` and `after`: each pair must hold the frame as it was before
    /// and after that write, the files must be named by sequence, and the series must end by
    /// itself after the requested count, with an index row per frame.
    ///
    /// Then, on the same device (one device per test keeps the suite's memory small): while frames are held before the upscaler the present composes nothing, so a
    /// one-shot request taken there reads the presented frame twice, both halves of its pair are the
    /// final picture, and the request is served (no longer left pending).
    #[test]
    fn series_reads_back_each_present_before_and_after_the_composition() {
        let Some((_entry, instance, physical_device, device, queue, family)) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let (width, height) = (16u32, 8u32);
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!("target/test-scratch/series-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let held_base = base.with_file_name(format!("series-held-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&held_base);
        unsafe {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = device.create_image(&info, None).unwrap();
            let reqs = device.get_image_memory_requirements(image);
            let props = instance.get_physical_device_memory_properties(physical_device);
            let type_index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None).unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();

            let mut series = Series::default();
            series.start(3, &base);
            assert!(series.running());
            let mut layout = vk::ImageLayout::UNDEFINED;
            for frame in 0..4u32 {
                let game = frame as f32 / 8.0;
                clear(&device, queue, pool, image, layout, [game, 0.0, 0.0, 1.0]);
                layout = vk::ImageLayout::PRESENT_SRC_KHR;
                let was_running = series.running();
                series.before(&device, &instance, physical_device, queue, family, image, width, height);
                clear(&device, queue, pool, image, layout, [game, 1.0, 0.0, 1.0]);
                series.after(&device, queue, image, width, height, true, None);
                assert_eq!(was_running, frame < 3, "the series must end after exactly 3 presents");
            }
            assert!(!series.running());

            // Held before the upscaler: the one-shot is taken by the present, as one frame.
            let header = Box::new(neural_forge_protocol::ShmHeader::default());
            let shm = crate::shm::ShmClient::test_over_header(&header);
            header.capture_request.store(1, std::sync::atomic::Ordering::Relaxed);
            for frame in 0..2u32 {
                if let Some(frames) = take_request(&shm, true) {
                    series.start(frames, &held_base);
                }
                // The game's frame, the model's edit already in it.
                clear(&device, queue, pool, image, layout, [0.25 + frame as f32 / 4.0, 0.5, 0.0, 1.0]);
                let was_running = series.running();
                series.before(&device, &instance, physical_device, queue, family, image, width, height);
                series.after(&device, queue, image, width, height, true, None);
                assert_eq!(was_running, frame == 0, "a one-shot is one frame");
            }
            assert_eq!(header.capture_request.load(std::sync::atomic::Ordering::Relaxed), 0, "the request was served");
            assert!(!series.running());
            series.destroy(&device);
            device.queue_wait_idle(queue).unwrap();
            device.destroy_command_pool(pool, None);
            device.destroy_image(image, None);
            device.free_memory(memory, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
        let dir = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
        assert!(dir.file_name().unwrap().to_string_lossy().starts_with("series-"));
        let decode = |name: &str| {
            let mut reader = png::Decoder::new(std::fs::File::open(dir.join(name)).unwrap()).read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size()];
            reader.next_frame(&mut buf).unwrap();
            buf
        };
        for seq in 0..3u32 {
            let (original, composited) = pair_names(seq);
            let red = (seq as f32 / 8.0 * 255.0).round() as u8;
            // Written as true RGBA (the image is BGRA): red carries the frame, green the "edit".
            let o = decode(&original);
            let c = decode(&composited);
            assert_eq!(&o[..4], &[red, 0, 0, 255], "frame {seq} original");
            assert_eq!(&c[..4], &[red, 255, 0, 255], "frame {seq} composited");
            assert!(o.chunks(4).all(|p| p == &o[..4]) && c.chunks(4).all(|p| p == &c[..4]));
        }
        assert!(!dir.join(pair_names(3).0).exists(), "a fourth present must not be captured");
        let index = std::fs::read_to_string(dir.join("index.tsv")).unwrap();
        assert_eq!(index.lines().count(), 4, "header plus one row per frame:\n{index}");
        let _ = std::fs::remove_dir_all(&base);

        let dir = std::fs::read_dir(&held_base).unwrap().next().unwrap().unwrap().path();
        let (original, composited) = pair_names(0);
        let (o, c) = (decode_at(&dir, &original), decode_at(&dir, &composited));
        assert_eq!(o, c, "held: both halves are the presented frame");
        // The first held frame (0.25, 0.5, 0, 1), within a level of the driver's rounding.
        assert!(o[0].abs_diff(64) <= 1 && o[1].abs_diff(128) <= 1 && o[2] == 0 && o[3] == 255, "{:?}", &o[..4]);
        assert!(!dir.join(pair_names(1).0).exists());
        let _ = std::fs::remove_dir_all(&held_base);
    }

    fn decode_at(dir: &std::path::Path, name: &str) -> Vec<u8> {
        let mut reader = png::Decoder::new(std::fs::File::open(dir.join(name)).unwrap()).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buf).unwrap();
        buf
    }
}
