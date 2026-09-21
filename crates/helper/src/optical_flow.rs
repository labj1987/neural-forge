//! Real motion vectors, estimated between consecutive proxy frames with
//! `VK_NV_optical_flow`.
//!
//! Unlike the (deleted) layer-side attempt this is adapted from, this runs on the
//! helper's own already-created Vulkan device -- no private device, no second
//! `vkCreateDevice` call, nothing created from inside the game's own process at a
//! swapchain-transition moment. That specific combination ("a private optical-flow
//! device created during a live game's swapchain transition") was the documented
//! trigger for a real driver crash the layer-side version was stubbed to avoid
//! (`crates/layer/src/shm.rs`'s `prepare_motion_resources`, still stubbed).
//!
//! This is the architecture DLSS5VKLayer's own AGPL-3.0 helper (`helper/main.cpp`)
//! actually uses: one `VkCtx`, created once at helper startup, doing both NGX
//! evaluation and optical flow -- read directly, not from docs, 2026-09-17 (see
//! `ATTRIBUTION.md`, `docs/GHOSTING_PLAN.md` step 4). This module is an independent
//! reimplementation of that same, generic `VK_NV_optical_flow` usage pattern (session
//! creation, grid negotiation, execute, NVIDIA's own documented signed-5.5
//! fixed-point flow-vector decode) -- not a port of their C++.
//!
//! Reference: <https://docs.vulkan.org/spec/latest/chapters/VK_NV_optical_flow/optical_flow.html>
use ash::vk;

/// Whichever queue family actually supports `VK_QUEUE_OPTICAL_FLOW_BIT_NV` --
/// resolved once at helper startup (`main.rs::create_vulkan_context`) alongside the
/// main queue, since Vulkan requires every queue a device will ever use to be
/// requested at `vkCreateDevice` time. `None` whenever no such family exists, or the
/// device lacks the extensions/features optical flow needs -- every caller treats
/// that exactly like a disabled feature (empty motion, not an error).
pub struct FlowQueue {
    pub family: u32,
    pub queue: vk::Queue,
}

pub struct OpticalFlow {
    api: vk::NvOpticalFlowFn,
    sync: ash::extensions::khr::Synchronization2,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    session: vk::OpticalFlowSessionNV,
    images: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    pub width: u32,
    pub height: u32,
    current: usize,
    previous: bool,
    grid: u32,
}
// Access is serialized by the helper's own single-threaded per-frame loop.
unsafe impl Send for OpticalFlow {}

impl OpticalFlow {
    /// `device`/`instance`/`pd` are the helper's own, already-created and already-live
    /// Vulkan context -- this never creates or owns a device. `flow_queue` is the
    /// family/queue `main.rs` already resolved at startup specifically for this.
    pub fn new(instance: &ash::Instance, device: &ash::Device, pd: vk::PhysicalDevice, flow_queue: &FlowQueue, width: u32, height: u32, quality: u32) -> Result<Self, String> {
        unsafe { Self::create(instance, device, pd, flow_queue, width, height, quality) }.map_err(|e| format!("{e:?}"))
    }

    unsafe fn create(instance: &ash::Instance, device: &ash::Device, pd: vk::PhysicalDevice, flow_queue: &FlowQueue, width: u32, height: u32, quality: u32) -> Result<Self, vk::Result> {
        let mut props = vk::PhysicalDeviceOpticalFlowPropertiesNV::default();
        instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::builder().push_next(&mut props));
        if width < props.min_width || width > props.max_width || height < props.min_height || height > props.max_height {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        let grid = [4, 2, 1, 8].into_iter().find(|&g| props.supported_output_grid_sizes.as_raw() & g != 0)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let api = vk::NvOpticalFlowFn::load(|name| std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr())));
        let sync = ash::extensions::khr::Synchronization2::new(instance, device);
        let mut flow = Self {
            api, sync, queue: flow_queue.queue, pool: vk::CommandPool::null(), cmd: vk::CommandBuffer::null(),
            session: vk::OpticalFlowSessionNV::null(), images: vec![], buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(), width, height, current: 0, previous: false, grid,
        };
        // This type intentionally has no `Drop` impl (it doesn't own `device`, unlike
        // the private-device version this is adapted from -- see this module's own
        // doc comment on why). That means every one of the several fallible steps
        // below, if it errors out through plain `?`, would otherwise leak whatever
        // Vulkan handles `flow` already holds -- worth avoiding given this is on a
        // lazy, retriable path (`main.rs` may call `new` again on a later frame after
        // a failure). `finish_building` does the actual sequential `?`-based setup;
        // any error from it is funneled through `destroy` here, in exactly one place,
        // before being returned -- `destroy` is already written to tolerate a
        // partially-populated `flow` (every Vulkan destroy call it makes is a
        // spec-guaranteed no-op on a still-`NULL_HANDLE`/never-touched field).
        if let Err(e) = Self::finish_building(&mut flow, instance, device, pd, flow_queue, width, height, level_for(quality)) {
            flow.destroy(device);
            return Err(e);
        }
        Ok(flow)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn finish_building(flow: &mut Self, instance: &ash::Instance, device: &ash::Device, pd: vk::PhysicalDevice, flow_queue: &FlowQueue, width: u32, height: u32, level: vk::OpticalFlowPerformanceLevelNV) -> Result<(), vk::Result> {
        let grid = flow.grid;
        let info = vk::OpticalFlowSessionCreateInfoNV::builder().width(width).height(height)
            .image_format(vk::Format::B8G8R8A8_UNORM).flow_vector_format(vk::Format::R16G16_S10_5_NV)
            .output_grid_size(vk::OpticalFlowGridSizeFlagsNV::from_raw(grid)).performance_level(level);
        (flow.api.create_optical_flow_session_nv)(device.handle(), &*info, std::ptr::null(), &mut flow.session).result()?;
        let mem = instance.get_physical_device_memory_properties(pd);
        for i in 0..3 {
            let output = i == 2;
            let (w, h) = if output { (width.div_ceil(grid), height.div_ceil(grid)) } else { (width, height) };
            let format = if output { vk::Format::R16G16_S10_5_NV } else { vk::Format::B8G8R8A8_UNORM };
            let mut usage = vk::OpticalFlowImageFormatInfoNV::builder().usage(if output { vk::OpticalFlowUsageFlagsNV::OUTPUT } else { vk::OpticalFlowUsageFlagsNV::INPUT });
            let info = vk::ImageCreateInfo::builder().image_type(vk::ImageType::TYPE_2D).format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 }).mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL).usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut usage);
            let image = device.create_image(&info, None)?;
            flow.images.push((image, vk::DeviceMemory::null(), vk::ImageView::null()));
            let req = device.get_image_memory_requirements(image);
            let type_index = memory_type(&mem, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(type_index), None)?;
            flow.images[i].1 = memory;
            device.bind_image_memory(image, memory, 0)?;
            let view = device.create_image_view(&vk::ImageViewCreateInfo::builder().image(image)
                .view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(subresource()), None)?;
            flow.images[i].2 = view;
        }
        flow.pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(flow_queue.family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)?;
        flow.cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(flow.pool)
            .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        flow.buffer = device.create_buffer(&vk::BufferCreateInfo::builder().size(u64::from(width) * u64::from(height) * 4)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST), None)?;
        let req = device.get_buffer_memory_requirements(flow.buffer);
        flow.memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size)
            .memory_type_index(memory_type(&mem, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?), None)?;
        device.bind_buffer_memory(flow.buffer, flow.memory, 0)?;
        flow.mapped = device.map_memory(flow.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?.cast();
        Ok(())
    }

    /// Returns full-resolution signed pixel displacements, current -> previous, one
    /// `[dx, dy]` per pixel in raster order -- exactly the shape
    /// `neuralforge_protocol::motion::encode` already takes (that function, and the
    /// `DLSSNR.MVec` upload path in `frame.rs` it feeds, have existed since before
    /// this module; this is simply the first real producer of the vectors they
    /// expect, so callers convert with `motion::encode(&vectors, motion_scale)`
    /// rather than this module duplicating that already-tested byte-packing itself).
    /// The first capture seeds history and returns `None` (no vectors yet, same as
    /// the game just having launched or a scene cut just having been observed and
    /// history cleared).
    pub fn estimate(&mut self, device: &ash::Device, bytes: &[u8], bgr: bool) -> Result<Option<Vec<[f32; 2]>>, String> {
        unsafe { self.run(device, bytes, bgr) }.map_err(|e| format!("{e:?}"))
    }

    unsafe fn run(&mut self, device: &ash::Device, bytes: &[u8], bgr: bool) -> Result<Option<Vec<[f32; 2]>>, vk::Result> {
        let n = self.width as usize * self.height as usize * 4;
        if bytes.len() != n {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, n);
        if !bgr {
            for p in std::slice::from_raw_parts_mut(self.mapped, n).chunks_exact_mut(4) {
                p.swap(0, 2);
            }
        }
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let input = self.images[self.current].0;
        self.barrier(input, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        device.cmd_copy_buffer_to_image(self.cmd, self.buffer, input, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(self.width, self.height)]);
        self.barrier(input, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
        if self.previous {
            for (binding, view) in [
                (vk::OpticalFlowSessionBindingPointNV::INPUT, self.images[self.current].2),
                (vk::OpticalFlowSessionBindingPointNV::REFERENCE, self.images[1 - self.current].2),
                (vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR, self.images[2].2),
            ] {
                (self.api.bind_optical_flow_session_image_nv)(device.handle(), self.session, binding, view, vk::ImageLayout::GENERAL).result()?;
            }
            self.barrier(self.images[2].0, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            // Captures may be several presents apart: disable implicit temporal hints.
            let execute = vk::OpticalFlowExecuteInfoNV::builder().flags(vk::OpticalFlowExecuteFlagsNV::DISABLE_TEMPORAL_HINTS);
            (self.api.cmd_optical_flow_execute_nv)(self.cmd, self.session, &*execute);
            self.barrier(self.images[2].0, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            device.cmd_copy_image_to_buffer(self.cmd, self.images[2].0, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.buffer,
                &[region(self.width.div_ceil(self.grid), self.height.div_ceil(self.grid))]);
            let barrier = [vk::MemoryBarrier2::builder().src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE).dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ).build()];
            self.sync.cmd_pipeline_barrier2(self.cmd, &vk::DependencyInfo::builder().memory_barriers(&barrier));
        }
        device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        device.queue_submit(self.queue, &[vk::SubmitInfo::builder().command_buffers(&cmds).build()], vk::Fence::null())?;
        device.queue_wait_idle(self.queue)?;
        let output = if self.previous {
            let grid_w = self.width.div_ceil(self.grid) as usize;
            // SAFETY: `self.mapped` is a live host-coherent mapping of at least
            // `grid_w * grid_h * 4` bytes (this buffer was sized for the full,
            // un-gridded frame, always >= the gridded flow output); the fence-free
            // `queue_wait_idle` above already confirmed the GPU's writes landed.
            let raw = std::slice::from_raw_parts(self.mapped, grid_w * self.height.div_ceil(self.grid) as usize * 4);
            Some(upsample_flow_grid(raw, grid_w, self.width, self.height, self.grid))
        } else {
            None
        };
        self.previous = true;
        self.current = 1 - self.current;
        Ok(output)
    }

    unsafe fn barrier(&self, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
        let barriers = [vk::ImageMemoryBarrier2::builder().image(image).old_layout(old).new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource()).src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE).build()];
        self.sync.cmd_pipeline_barrier2(self.cmd, &vk::DependencyInfo::builder().image_memory_barriers(&barriers));
    }

    /// # Safety
    /// `device` must be the same live device this session was created against; no
    /// submitted work referencing this session's handles may still be in flight
    /// (the last `estimate` call's own `queue_wait_idle` already guarantees that for
    /// every caller in this codebase, which never calls this concurrently with an
    /// in-flight `estimate`).
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            let _ = device.device_wait_idle();
            if self.session != vk::OpticalFlowSessionNV::null() {
                (self.api.destroy_optical_flow_session_nv)(device.handle(), self.session, std::ptr::null());
            }
            for &(image, memory, view) in &self.images {
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            if !self.mapped.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Whether `current` looks like a different scene than `previous` -- a hard cut
/// (level transition, cutscene, death/respawn) rather than motion within one scene.
/// Carrying a flow field across a cut hands the model a field describing content
/// that is no longer on screen, worse than handing it nothing -- DLSS5VKLayer's own
/// `helper/main.cpp` runs the same kind of check (`DetectSceneCut`, CPU, against the
/// proxy bytes it already has) for exactly this reason; this is an independent
/// reimplementation of the same generic technique (mean luma delta against a
/// threshold), not a port.
///
/// `previous`/`current` are both `width*height*4` BGRA8/RGBA8 proxy bytes (the same
/// format `OpticalFlow::estimate` itself reads). Deliberately coarse and cheap --
/// samples every 4th pixel rather than every pixel, since this only needs to catch
/// "the whole picture changed", not a precise measurement, and it runs on the
/// helper's own hot per-frame path.
pub fn is_scene_cut(previous: &[u8], current: &[u8], width: u32, height: u32, threshold: u8) -> bool {
    let n = (width as usize) * (height as usize) * 4;
    if previous.len() < n || current.len() < n || n == 0 {
        return false;
    }
    let mut sum_delta: i64 = 0;
    let mut samples: i64 = 0;
    let mut i = 0;
    while i + 4 <= n {
        // Unweighted average of the three colour channels -- a full luminance
        // formula is not worth the cost here; this only needs to catch "the whole
        // picture changed", not measure light. `bgr`/`rgb` channel order does not
        // matter for an unweighted average of the same three channels.
        let prev_luma = (previous[i] as i32 + previous[i + 1] as i32 + previous[i + 2] as i32) / 3;
        let curr_luma = (current[i] as i32 + current[i + 1] as i32 + current[i + 2] as i32) / 3;
        sum_delta += (curr_luma - prev_luma).unsigned_abs() as i64;
        samples += 1;
        i += 16; // every 4th pixel (4 bytes/pixel * 4)
    }
    if samples == 0 {
        return false;
    }
    (sum_delta / samples) as u8 >= threshold
}

/// `quality` is [`neuralforge_protocol::enums::mvec_quality`]'s own raw value
/// (`main.rs`'s caller passes `hdr.mvec_quality()` straight through) -- matched here
/// by value rather than importing the constants, since this is the only place in
/// this crate that needs them and the values are stable, documented protocol
/// constants (0 fast, 1 balanced, 2 quality), not a magic number guessed locally.
fn level_for(quality: u32) -> vk::OpticalFlowPerformanceLevelNV {
    match quality {
        neuralforge_protocol::enums::mvec_quality::FAST => vk::OpticalFlowPerformanceLevelNV::FAST,
        neuralforge_protocol::enums::mvec_quality::QUALITY => vk::OpticalFlowPerformanceLevelNV::SLOW,
        _ => vk::OpticalFlowPerformanceLevelNV::MEDIUM,
    }
}

fn memory_type(mem: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, vk::Result> {
    (0..mem.memory_type_count).find(|&i| bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(flags))
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
}

fn subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1).build()
}

fn region(width: u32, height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy::builder().image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build())
        .image_extent(vk::Extent3D { width, height, depth: 1 }).build()
}

/// NVOF's raw output is one `[i16; 2]` signed-5.5-fixed-point displacement per grid
/// cell (`raw`, `grid_w` cells wide), current -> previous, in pixels (the fixed-point
/// scale is NVIDIA's own documented `/32.0`, not a guess -- matches the layer-side
/// implementation this is adapted from, which was verified against the Vulkan spec's
/// own worked description of the format). Upsampled here (nearest -- one grid cell's
/// vector repeated across every pixel it covers, matching the deadzone/upscale
/// compute pass DLSS5VKLayer's own `helper/shaders/mvec_deadzone.comp` does on the
/// GPU, done here on the CPU for a first, simple, correct version) into one `[dx, dy]`
/// per pixel, raster order -- exactly what `neuralforge_protocol::motion::encode`
/// takes, so the actual `R16G16_SFLOAT` byte-packing (including that function's own
/// deadzone and scale handling) is that already-tested code, not duplicated here.
fn upsample_flow_grid(raw: &[u8], grid_w: usize, width: u32, height: u32, grid: u32) -> Vec<[f32; 2]> {
    let mut out = vec![[0.0f32; 2]; width as usize * height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let cell = (y / grid as usize) * grid_w + x / grid as usize;
            let offset = cell * 4;
            if offset + 4 <= raw.len() {
                out[y * width as usize + x] = [
                    i16::from_le_bytes([raw[offset], raw[offset + 1]]) as f32 / 32.0,
                    i16::from_le_bytes([raw[offset + 2], raw[offset + 3]]) as f32 / 32.0,
                ];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_repeats_one_grid_cell_across_every_pixel_it_covers() {
        // 2x2 pixels, grid=2 (one cell covers the whole thing): raw holds one
        // [dx,dy] = [1.0, -0.5] (32 and -16 in signed-5.5 fixed point).
        let raw = [32i16.to_le_bytes(), (-16i16).to_le_bytes()].concat();
        let out = upsample_flow_grid(&raw, 1, 2, 2, 2);
        assert_eq!(out.len(), 4);
        for v in out {
            assert!((v[0] - 1.0).abs() < 1e-6, "dx should be 1.0, got {}", v[0]);
            assert!((v[1] - -0.5).abs() < 1e-6, "dy should be -0.5, got {}", v[1]);
        }
    }

    #[test]
    fn upsample_handles_a_grid_that_does_not_evenly_divide_the_frame() {
        // 3x3 pixels, grid=2 -> a 2x2 cell grid (ceil(3/2)=2), the last row/column of
        // pixels reads the second (edge) cell -- must not panic or read out of bounds.
        let cells = [[10i16, 0i16], [0, 10], [-10, 0], [0, -10]];
        let raw: Vec<u8> = cells.iter().flat_map(|c| [c[0].to_le_bytes(), c[1].to_le_bytes()].concat()).collect();
        let out = upsample_flow_grid(&raw, 2, 3, 3, 2);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn upsample_zero_fills_when_the_raw_buffer_is_short() {
        // A short/corrupt raw buffer must not panic (out-of-bounds) -- zero-fill
        // instead, the same fail-open discipline as every other malformed-input path
        // in this codebase.
        let out = upsample_flow_grid(&[], 1, 2, 2, 2);
        assert_eq!(out, vec![[0.0, 0.0]; 4]);
    }

    #[test]
    fn upsampled_vectors_feed_directly_into_the_already_tested_motion_encode() {
        // The actual integration point: this module's own output, run through
        // `neuralforge_protocol::motion::encode` exactly as `main.rs` will call it,
        // must produce real, non-degenerate R16G16_SFLOAT bytes -- confirms the two
        // modules' shapes genuinely agree, not just that each compiles alone.
        let raw = [64i16.to_le_bytes(), 0i16.to_le_bytes()].concat(); // dx=2.0, dy=0.0
        let vectors = upsample_flow_grid(&raw, 1, 2, 2, 2);
        let bytes = neuralforge_protocol::motion::encode(&vectors, [1.0, 1.0]);
        assert_eq!(bytes.len(), vectors.len() * 4);
        // dx=2.0 at scale 1.0 -> half(2.0) = 0x4000, little-endian.
        assert_eq!(&bytes[0..2], &0x4000u16.to_le_bytes());
    }

    #[test]
    fn scene_cut_is_not_flagged_for_a_stable_or_slightly_moving_frame() {
        let (w, h) = (8u32, 8u32);
        let a = vec![100u8; (w * h * 4) as usize];
        let mut b = a.clone();
        // A small, uniform shift -- motion, not a cut.
        for p in b.chunks_exact_mut(4) {
            p[0] = p[0].saturating_add(5);
            p[1] = p[1].saturating_add(5);
            p[2] = p[2].saturating_add(5);
        }
        assert!(!is_scene_cut(&a, &a, w, h, 40), "identical frames must never be a cut");
        assert!(!is_scene_cut(&a, &b, w, h, 40), "a small uniform shift must not be a cut");
    }

    #[test]
    fn scene_cut_is_flagged_for_a_completely_different_frame() {
        let (w, h) = (8u32, 8u32);
        let a = vec![20u8; (w * h * 4) as usize];
        let b = vec![220u8; (w * h * 4) as usize];
        assert!(is_scene_cut(&a, &b, w, h, 40), "a frame that changed almost entirely must be a cut");
    }

    #[test]
    fn scene_cut_never_panics_on_mismatched_or_empty_buffers() {
        assert!(!is_scene_cut(&[], &[], 8, 8, 40));
        assert!(!is_scene_cut(&[1, 2, 3], &[1, 2, 3], 8, 8, 40));
    }

}
