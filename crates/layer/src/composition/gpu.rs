//! GPU dispatch of `shaders/compose.comp`, wired into `capture.rs`'s write-back as of
//! 2026-09-10 -- the real fix for [`super::apply`]'s CPU path's real, measured
//! performance cost (see that module's own doc comment, and the crate's `CLAUDE.md`
//! entry, for the actual numbers: ~800ms/frame single-threaded, ~97/10s even
//! multi-threaded on a 16-core machine, both far short of the no-composition
//! baseline). Same algorithm (`compose.comp` is the hand-translated GPU twin of the
//! exact `color.rs` functions `apply.rs` calls directly), same real-hardware
//! verification target -- this is a different, faster execution strategy for
//! identical math, not a rewrite of it.
//!
//! Three storage images bound as inputs (`u_original`, `u_proxy`, `u_model_answer`)
//! and one as output (`u_output`), all `R8G8B8A8_UNORM` -- matching the real, only-
//! currently-supported `RGBA8` proxy format (`RGBA16F` still falls back to
//! [`super::apply`]'s CPU path, same gap that path already has, unchanged by this
//! module). `u_original` and `u_proxy` are bound to the *same* image view: no separate
//! downscaled proxy exists yet (`capture.rs` sends the full captured frame as the
//! proxy), so uploading it twice would be pure waste, on the GPU exactly as it already
//! was on the CPU path.
//!
//! Two execution paths, both real and tested:
//! - [`GpuCompose::dispatch`]/[`GpuCompose::dispatch_into_image`]: one command buffer,
//!   one fence, fully synchronous (submit, then wait). Used for the rare cases that
//!   need the CPU to see the result before moving on (a pending `capture_request`
//!   dump) or as the correctness reference the async path below is checked against.
//! - [`GpuCompose::dispatch_into_image_async`] (added 2026-09-10): the real per-frame
//!   fast path. Double-buffered (`async_slots`, 2 of them) so the CPU never has to
//!   block on *this* frame's own compute work -- it submits with a signal semaphore
//!   and returns immediately, leaving `capture::run`/`device.rs` to chain that
//!   semaphore into the real present call's own wait list (a GPU-side dependency, not
//!   a CPU one) so the presentation engine -- not our code -- is what actually waits
//!   for the compute work to finish before the frame goes on screen. The only new CPU
//!   wait this introduces is on a slot's *own* fence, immediately before that same
//!   slot is reused two dispatches later -- by then the GPU has almost always long
//!   since finished, so that wait is normally instant. See its own doc comment for the
//!   full reasoning, including why this needed real thought about semaphore reuse
//!   safety, not just "make it not block."

use ash::vk;

const SPV: &[u8] = include_bytes!("../../shaders/compose.spv");
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const ASYNC_SLOTS: usize = 2;

/// The live composition settings the async present path needs, plus whether the
/// capture leg actually encoded the proxy for this answer (which decides the mode --
/// see `compose.comp`). Grouped so threading them does not add four positional
/// parameters to every function between `capture::run` and the dispatch.
#[derive(Clone, Copy)]
pub struct ComposeParams {
    pub colour_strength: f32,
    pub transfer_strength: f32,
    pub max_ratio: f32,
    /// True when the proxy that produced this answer went through
    /// [`crate::composition::encode_pass`]. False means the proxy is a bit-identical
    /// copy of the frame and the ratio transfer would self-cancel on it.
    pub proxy_encoded: bool,
}

#[repr(C)]
struct PushConstants {
    colour_strength: f32,
    transfer_strength: f32,
    max_ratio: f32,
    /// Matches `compose.comp`'s own `params.bgr_order` field exactly (same offset,
    /// same 4-byte size as the `f32`s above it -- no padding to worry about).
    bgr_order: u32,
    /// Matches `compose.comp`'s own `params.mode`: 0 selects the classic ratio-transfer
    /// (`UpgradeToneMap`), 1 selects the guarded-additive formula the async
    /// held-answer path needs instead -- see that shader's own doc comment on why.
    mode: u32,
}

struct Image {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl Image {
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract (device-destruction
        // time, or a same-size-class rebuild with no in-flight GPU work referencing
        // these handles -- every caller only ever calls this after a completed
        // `wait_for_fences` on the fence that guards this specific set of resources).
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| (type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(wanted))
}

fn create_storage_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> Option<Image> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(FORMAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage | vk::ImageUsageFlags::STORAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `info` is a valid `VkImageCreateInfo`.
    let image = unsafe { device.create_image(&info, None) }.ok()?;
    // SAFETY: `image` was just created, no memory bound yet.
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let Some(type_index) = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
    else {
        // SAFETY: `image` has no memory bound; nothing else references it.
        unsafe { device.destroy_image(image, None) };
        return None;
    };
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: `image` has no memory bound.
            unsafe { device.destroy_image(image, None) };
            return None;
        }
    };
    // SAFETY: `image`/`memory` were each just created, sized/typed for each other.
    if unsafe { device.bind_image_memory(image, memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return None;
    }
    let view_info = vk::ImageViewCreateInfo::builder().image(image).view_type(vk::ImageViewType::TYPE_2D).format(FORMAT).subresource_range(
        vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build(),
    );
    // SAFETY: `image` is bound to memory; `view_info` matches it.
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(v) => v,
        Err(_) => {
            // SAFETY: nothing else references `image`/`memory` yet.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_image(image, None);
            }
            return None;
        }
    };
    Some(Image { image, view, memory })
}

fn image_copy_region(width: u32, height: u32, offset: u64) -> vk::BufferImageCopy {
    vk::BufferImageCopy::builder()
        .buffer_offset(offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D { width, height, depth: 1 })
        .build()
}

fn full_subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build()
}

fn image_barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(full_subresource())
        .src_access_mask(src)
        .dst_access_mask(dst)
        .build()
}

struct Sized_ {
    width: u32,
    height: u32,
    original: Image,
    model_answer: Image,
    // The game frame that produced `model_answer`; used to carry its enhancement
    // delta onto newer game frames.
    proxy: Image,
    output: Image,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_capacity: vk::DeviceSize,
    // A device-local copy of the most recent raw model frame.  Re-presenting a
    // held answer must not re-upload 4K pixels from the CPU every swapchain present.
    cached_buffer: vk::Buffer,
    cached_memory: vk::DeviceMemory,
    current_buffer: vk::Buffer,
    current_memory: vk::DeviceMemory,
}

/// One independent, fully self-contained resource set: its own images/staging buffer
/// (sized on first use, rebuilt on resize), descriptor set, command buffer, and fence.
/// The sync path uses exactly one of these; the async path uses [`ASYNC_SLOTS`] of
/// them, alternating (see [`AsyncSlot`]); presentation semaphores follow images.
struct ComposeSlot {
    descriptor_set: vk::DescriptorSet,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    sized: Option<Sized_>,
    cached_generation: u64,
    /// `working_scale`'s compose-side scratch: an intermediate image at the answer's
    /// own (usually smaller) resolution, blitted up into `Sized_::model_answer`
    /// before the compute shader reads it -- the same `vkCmdBlitImage`/
    /// `VK_FILTER_LINEAR` mechanism `capture::ModelScratch` uses on the capture side,
    /// so the compute shader itself (and every image binding downstream of
    /// `model_answer`) is completely unaware scaling happened at all. Independent of
    /// `sized`'s own lifetime -- the answer's resolution and the output's resolution
    /// change on different triggers (a `working_scale` edit vs. a swapchain resize),
    /// so tying this to `ensure_sized`'s full rebuild would destroy/rebuild
    /// `original`/`proxy`/`output` for no reason whenever only one of them changes.
    /// `None` until the first scaled answer is ever composited on this slot.
    small_answer: Option<(Image, u32, u32)>,
}

impl ComposeSlot {
    fn new(device: &ash::Device, pool: vk::CommandPool, descriptor_pool: vk::DescriptorPool, descriptor_set_layout: vk::DescriptorSetLayout) -> Option<Self> {
        let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(descriptor_pool).set_layouts(std::slice::from_ref(&descriptor_set_layout));
        // SAFETY: `descriptor_pool` was sized by the caller for this and every other slot.
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }.ok()?[0];
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        // SAFETY: `pool` was created by the caller with `RESET_COMMAND_BUFFER`.
        let Ok(cmd) = (unsafe { crate::loader_data::allocate_commands(device, &cmd_alloc_info) }) else { return None };
        let cmd = cmd[0];
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: starting signaled means this slot's first real use never blocks on a
        // fence nothing has submitted work against yet.
        let Ok(fence) = (unsafe { device.create_fence(&fence_info, None) }) else { return None };
        Some(Self { descriptor_set, cmd, fence, sized: None, cached_generation: 0, small_answer: None })
    }

    /// (Re)builds this slot's own images/staging buffer if `width`/`height` changed
    /// (or this is the first use). Never touches any *other* slot's resources.
    fn ensure_sized(&mut self, device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32) -> bool {
        if let Some(s) = &self.sized {
            if s.width == width && s.height == height {
                return true;
            }
            // SAFETY: called only after this slot's own fence has been waited on
            // (every caller below does this before calling `ensure_sized`) -- never
            // while GPU work referencing these specific images might still be in
            // flight.
            unsafe {
                s.original.destroy(device);
                s.model_answer.destroy(device);
                s.proxy.destroy(device);
                s.output.destroy(device);
                device.destroy_buffer(s.staging_buffer, None);
                device.free_memory(s.staging_memory, None);
                device.destroy_buffer(s.cached_buffer, None);
                device.free_memory(s.cached_memory, None);
                device.destroy_buffer(s.current_buffer, None);
                device.free_memory(s.current_memory, None);
            }
            self.sized = None;
            self.cached_generation = 0;
        }

        let usage_in = vk::ImageUsageFlags::TRANSFER_DST;
        let usage_out = vk::ImageUsageFlags::TRANSFER_SRC;
        let (Some(original), Some(model_answer), Some(proxy), Some(output)) = (
            create_storage_image(device, mem_props, width, height, usage_in),
            create_storage_image(device, mem_props, width, height, usage_in),
            create_storage_image(device, mem_props, width, height, usage_in),
            create_storage_image(device, mem_props, width, height, usage_out),
        ) else {
            return false;
        };

        let frame_bytes = u64::from(width) * u64::from(height) * 4;
        let staging_capacity = frame_bytes * 2; // original + model_answer, uploaded together.
        let buf_info = vk::BufferCreateInfo::builder()
            .size(staging_capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buf_info` is valid.
        let Ok(staging_buffer) = (unsafe { device.create_buffer(&buf_info, None) }) else { return false };
        // SAFETY: `staging_buffer` was just created, not yet bound to memory.
        let reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let Some(type_index) =
            find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        else {
            // SAFETY: `staging_buffer` has no memory bound.
            unsafe { device.destroy_buffer(staging_buffer, None) };
            return false;
        };
        let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
        // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
        let Ok(staging_memory) = (unsafe { device.allocate_memory(&alloc, None) }) else {
            // SAFETY: same reasoning as above.
            unsafe { device.destroy_buffer(staging_buffer, None) };
            return false;
        };
        // SAFETY: `staging_buffer`/`staging_memory` were each just created, sized/typed
        // for each other.
        if unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }.is_err() {
            // SAFETY: neither is aliased anywhere else.
            unsafe {
                device.free_memory(staging_memory, None);
                device.destroy_buffer(staging_buffer, None);
            }
            return false;
        }
        // SAFETY: `staging_memory` is `HOST_VISIBLE`; mapping the whole allocation is
        // always in bounds.
        let Ok(staging_ptr) = (unsafe { device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }) else {
            // SAFETY: same reasoning as above.
            unsafe {
                device.free_memory(staging_memory, None);
                device.destroy_buffer(staging_buffer, None);
            }
            return false;
        };

        // Keep the held raw model frame in device-local memory.  The staging buffer
        // stays CPU-visible only for the infrequent helper-answer update; regular
        // presents copy this cached buffer straight to the swapchain image.
        let cache_info = vk::BufferCreateInfo::builder()
            .size(frame_bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let Ok(cached_buffer) = (unsafe { device.create_buffer(&cache_info, None) }) else {
            unsafe {
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
                original.destroy(device);
                model_answer.destroy(device);
                proxy.destroy(device);
                output.destroy(device);
            }
            return false;
        };
        let cached_reqs = unsafe { device.get_buffer_memory_requirements(cached_buffer) };
        let Some(cached_type) = find_memory_type(mem_props, cached_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_memory_type(mem_props, cached_reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
        else {
            unsafe {
                device.destroy_buffer(cached_buffer, None);
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
                original.destroy(device);
                model_answer.destroy(device);
                proxy.destroy(device);
                output.destroy(device);
            }
            return false;
        };
        let cached_alloc = vk::MemoryAllocateInfo::builder().allocation_size(cached_reqs.size).memory_type_index(cached_type);
        let Ok(cached_memory) = (unsafe { device.allocate_memory(&cached_alloc, None) }) else {
            unsafe {
                device.destroy_buffer(cached_buffer, None);
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
                original.destroy(device);
                model_answer.destroy(device);
                proxy.destroy(device);
                output.destroy(device);
            }
            return false;
        };
        if unsafe { device.bind_buffer_memory(cached_buffer, cached_memory, 0) }.is_err() {
            unsafe {
                device.free_memory(cached_memory, None);
                device.destroy_buffer(cached_buffer, None);
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
                original.destroy(device);
                model_answer.destroy(device);
                proxy.destroy(device);
                output.destroy(device);
            }
            return false;
        }

        let current_info = vk::BufferCreateInfo::builder()
            .size(frame_bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let Ok(current_buffer) = (unsafe { device.create_buffer(&current_info, None) }) else { return false };
        let current_reqs = unsafe { device.get_buffer_memory_requirements(current_buffer) };
        let Some(current_type) = find_memory_type(mem_props, current_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_memory_type(mem_props, current_reqs.memory_type_bits, vk::MemoryPropertyFlags::empty())) else { unsafe { device.destroy_buffer(current_buffer, None) }; return false };
        let current_alloc = vk::MemoryAllocateInfo::builder().allocation_size(current_reqs.size).memory_type_index(current_type);
        let Ok(current_memory) = (unsafe { device.allocate_memory(&current_alloc, None) }) else { unsafe { device.destroy_buffer(current_buffer, None) }; return false };
        if unsafe { device.bind_buffer_memory(current_buffer, current_memory, 0) }.is_err() { unsafe { device.free_memory(current_memory, None); device.destroy_buffer(current_buffer, None) }; return false; }

        let image_info = |view: vk::ImageView| vk::DescriptorImageInfo::builder().image_view(view).image_layout(vk::ImageLayout::GENERAL).build();
        let infos = [image_info(original.view), image_info(proxy.view), image_info(model_answer.view), image_info(output.view)];
        let writes: Vec<_> = (0..4u32)
            .map(|i| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&infos[i as usize]))
                    .build()
            })
            .collect();
        // SAFETY: `self.descriptor_set` was allocated for exactly this slot, matches
        // `writes`' layout (4 storage-image bindings); every image view is live for at
        // least as long as `self.sized` holds its owning `Image`.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        self.sized = Some(Sized_ { width, height, original, model_answer, proxy, output, staging_buffer, staging_memory, staging_ptr: staging_ptr.cast(), staging_capacity, cached_buffer, cached_memory, current_buffer, current_memory });
        true
    }

    /// `working_scale`'s compose-side counterpart to `capture::CapturePipeline`'s own
    /// `ensure_model_scratch`-equivalent lazy build/rebuild -- see `small_answer`'s
    /// own doc comment on why this is a separate image with its own independent
    /// lifetime rather than folded into `ensure_sized`.
    fn ensure_small_answer(&mut self, device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32) -> bool {
        if let Some((_, w, h)) = &self.small_answer {
            if *w == width && *h == height {
                return true;
            }
        }
        if let Some((old, _, _)) = self.small_answer.take() {
            // SAFETY: called only after this slot's own fence has been waited on --
            // every caller of this method (`present_temporal_delta_async`) does so
            // before reaching this point, mirroring `ensure_sized`'s identical
            // reasoning -- never while GPU work referencing this image might still
            // be in flight.
            unsafe { old.destroy(device) };
        }
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        // Only `TRANSFER_DST`/`TRANSFER_SRC` are actually needed (this image is never
        // bound to the compute shader, only blitted from) -- `create_storage_image`
        // bakes `STORAGE` into every image it builds regardless, accepted here as a
        // harmless simplification rather than a second, near-identical image builder
        // just to drop one usage flag.
        let Some(image) = create_storage_image(device, mem_props, width, height, usage) else { return false };
        self.small_answer = Some((image, width, height));
        true
    }

    fn begin_ensured(&mut self, device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32, original: &[u8], model_answer: &[u8]) -> Option<u64> {
        let frame_bytes = (u64::from(width) * u64::from(height) * 4) as usize;
        if original.len() < frame_bytes || model_answer.len() < frame_bytes {
            return None;
        }
        if !self.ensure_sized(device, mem_props, width, height) {
            return None;
        }
        let s = self.sized.as_ref().expect("just ensured above");
        if (s.staging_capacity as usize) < frame_bytes * 2 {
            return None;
        }
        // SAFETY: `s.staging_ptr` is a live mapping of at least `frame_bytes * 2`
        // bytes; `original`/`model_answer` were just confirmed at least `frame_bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(original.as_ptr(), s.staging_ptr, frame_bytes);
            std::ptr::copy_nonoverlapping(model_answer.as_ptr(), s.staging_ptr.add(frame_bytes), frame_bytes);
        }
        Some(frame_bytes as u64)
    }

    /// Records the shared portion of every dispatch variant onto `self.cmd` (already
    /// begun): upload `original`/`model_answer` into this slot's own images, run the
    /// compute shader, and copy `output` back into this slot's own `staging_buffer` at
    /// offset 0. Callers record whatever happens after this themselves (download to a
    /// CPU slice, vs. straight into another image).
    ///
    /// # Safety
    /// `self.cmd` must already be recording.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_upload_and_compute(&self, device: &ash::Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout, width: u32, height: u32, frame_bytes: u64, colour_strength: f32, transfer_strength: f32, max_ratio: f32, bgr_order: bool) {
        let s = self.sized.as_ref().expect("caller already ensured this");
        let region = |offset| image_copy_region(width, height, offset);
        // SAFETY: `self.cmd` is recording (forwarded from this function's own
        // contract); every image below was just (re)created by `ensure_sized` and is
        // still `UNDEFINED` (or is being deliberately discarded via `UNDEFINED` as
        // `oldLayout`, spec-legal and exactly what a fresh per-frame result needs --
        // see the crash-fix writeup in `CLAUDE.md` for why this specific pattern is
        // safe, unlike blindly assuming a *different* real prior layout).
        unsafe {
            let to_dst = [
                image_barrier(s.original.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
                image_barrier(s.model_answer.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
                image_barrier(s.proxy.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE),
            ];
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(0)]);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(frame_bytes)]);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.proxy.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(0)]);

            let to_general = [
                image_barrier(s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                image_barrier(s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                image_barrier(s.proxy.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                image_barrier(s.output.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::SHADER_WRITE),
            ];
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_general);

            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, std::slice::from_ref(&self.descriptor_set), &[]);
            let push = PushConstants { colour_strength, transfer_strength, max_ratio, bgr_order: bgr_order as u32, mode: 0 };
            let push_bytes = std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<PushConstants>());
            device.cmd_push_constants(self.cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            device.cmd_dispatch(self.cmd, width.div_ceil(8), height.div_ceil(8), 1);

            let to_src = image_barrier(s.output.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            device.cmd_copy_image_to_buffer(self.cmd, s.output.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, s.staging_buffer, &[region(0)]);
        }
    }

    /// Records the copy from this slot's own `staging_buffer` (offset 0, just written
    /// by `record_upload_and_compute`'s own final copy) straight into `target_image`,
    /// then restores `target_image` to `PRESENT_SRC_KHR`. Deliberately buffer-mediated,
    /// not a raw `vkCmdCopyImage` straight from `output` -- see
    /// [`GpuCompose::dispatch_into_image`]'s own doc comment for why (format-mismatch
    /// color corruption risk).
    ///
    /// Manages `target_image`'s own `PRESENT_SRC_KHR -> TRANSFER_DST_OPTIMAL ->
    /// PRESENT_SRC_KHR` round trip itself (2026-09-10) -- until this, the contract was
    /// "assumes already `TRANSFER_DST_OPTIMAL`", relying entirely on `capture::run`'s
    /// own stage 1 having *already* transitioned it that far as a side effect of its
    /// own unrelated readback. That coupling is exactly what made stage 1 and
    /// composition inseparable, which is what forced every single present call through
    /// a full, synchronous, helper-round-trip-gated capture+composite cycle in the
    /// first place (see `capture.rs`'s own doc comment on the pipelined redesign this
    /// enabled) -- a real Vulkan-layout bug waiting to happen the moment anything tried
    /// to call this without stage 1 having just run.
    ///
    /// # Safety
    /// `self.cmd` must be recording, with `record_upload_and_compute` already called on
    /// it this same recording, and `target_image` must currently be `PRESENT_SRC_KHR`
    /// (true of any image `vkQueuePresentKHR`'s own precondition hasn't been violated
    /// on, real swapchain images included).
    unsafe fn record_copy_into_image(&self, device: &ash::Device, width: u32, height: u32, frame_bytes: u64, target_image: vk::Image) {
        let s = self.sized.as_ref().expect("caller already ensured this");
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            let to_transfer_dst = image_barrier(
                target_image,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_dst],
            );
            let buffer_barrier = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s.staging_buffer)
                .offset(0)
                .size(frame_bytes)
                .build();
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[buffer_barrier], &[]);
            device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
            let to_present = image_barrier(target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
        }
    }

    /// Records a raw held-answer update (only when `update` is true) followed by a
    /// device-local buffer-to-swapchain copy.  This is deliberately shader-free: the
    /// helper answer is already encoded in the swapchain's byte order, and copying
    /// through a buffer preserves those bytes without assuming the target format.
    ///
    /// # Safety
    /// `self.cmd` is recording and `target_image` is currently `PRESENT_SRC_KHR`.
    unsafe fn record_cached_raw_into_image(&self, device: &ash::Device, width: u32, height: u32, frame_bytes: u64, update: bool, target_image: vk::Image) {
        let s = self.sized.as_ref().expect("caller already ensured this");
        unsafe {
            if update {
                let copy = vk::BufferCopy::builder().src_offset(0).dst_offset(0).size(frame_bytes).build();
                device.cmd_copy_buffer(self.cmd, s.staging_buffer, s.cached_buffer, &[copy]);
            }
            // This also covers reads in a later queue submission: queue order alone
            // orders execution, while this memory dependency makes the cached write
            // visible to its next transfer read.
            let ready = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(s.cached_buffer)
                .offset(0)
                .size(frame_bytes)
                .build();
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[ready], &[]);
            let to_transfer_dst = image_barrier(target_image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_transfer_dst]);
            device.cmd_copy_buffer_to_image(self.cmd, s.cached_buffer, target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
            let to_present = image_barrier(target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
        }
    }

    /// Records a lightweight temporal carry pass.  It reads the current swapchain
    /// image into device-local storage, applies the cached model delta, then writes
    /// the result back; no per-frame CPU readback or upload is involved.
    /// Despite the name (kept to avoid a bigger rename tonight -- see
    /// `ATTRIBUTION.md`/`GHOSTING_PLAN.md`), this no longer carries a "delta" across
    /// frames: `compose.comp`'s `carry_delta` branch was deleted 2026-09-17 after
    /// reading DLSS5VKLayer's own resolve function (AGPL-3.0), which documents trying
    /// exactly that reprojection technique and measuring it a dead end. What this
    /// dispatches now is the same ratio-transfer `UpgradeToneMap` path
    /// `record_upload_and_compute` already used -- `s.proxy`/`s.model_answer` are
    /// still whatever the most recent answer pair is (held between round trips,
    /// updated only when `update_cache` is true, unchanged from before), and
    /// `s.original` is still read fresh off `target_image` every single call
    /// (unchanged from before) -- this function's real remaining value versus just
    /// calling `record_upload_and_compute` directly is the caching/device-local-copy
    /// performance structure below (upload only on a new generation, otherwise reuse
    /// `s.cached_buffer`), not any longer a different composition strategy.
    ///
    /// `answer_width`/`answer_height` are the model answer's *own* resolution --
    /// usually equal to `width`/`height`, but smaller when `working_scale` is active
    /// (see [`ComposeSlot::small_answer`]'s own doc comment). When they differ, the
    /// staging buffer's second half (uploaded at `answer_width*answer_height*4`
    /// bytes, not the full `frame_bytes`) goes into `self.small_answer` first, then a
    /// GPU blit (`VK_FILTER_LINEAR`) grows it up into `s.model_answer` -- after which
    /// the rest of this function, and the compute shader it dispatches, reads
    /// `s.model_answer` exactly as it always has, completely unaware scaling
    /// happened at all.
    #[allow(clippy::too_many_arguments)]
    unsafe fn record_temporal_delta_into_image(&self, device: &ash::Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout, width: u32, height: u32, answer_width: u32, answer_height: u32, frame_bytes: u64, update_cache: bool, bgr_order: bool, target_image: vk::Image, compose: ComposeParams) {
        let s = self.sized.as_ref().expect("caller already ensured this");
        let cached_before = self.cached_generation != 0;
        let scaled_answer = answer_width != width || answer_height != height;
        unsafe {
            let target_to_src = image_barrier(target_image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[target_to_src]);
            device.cmd_copy_image_to_buffer(self.cmd, target_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, s.current_buffer, &[image_copy_region(width, height, 0)]);
            let current_ready = vk::BufferMemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).buffer(s.current_buffer).offset(0).size(frame_bytes).build();
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[current_ready], &[]);

            if update_cache {
                let old = if cached_before { vk::ImageLayout::GENERAL } else { vk::ImageLayout::UNDEFINED };
                let to_dst = [
                    image_barrier(s.proxy.image, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE),
                    image_barrier(s.model_answer.image, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE),
                ];
                device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_dst);
                device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.proxy.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
                if scaled_answer {
                    // `self.small_answer` was already ensured sized to
                    // `(answer_width, answer_height)` by `present_temporal_delta_async`
                    // before this call -- required precondition, matching every other
                    // `self.sized.as_ref().expect(...)` in this file.
                    let (small, _, _) = self.small_answer.as_ref().expect("caller already ensured this");
                    let small_to_dst = image_barrier(small.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
                    device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[small_to_dst]);
                    device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, small.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(answer_width, answer_height, frame_bytes)]);
                    let small_to_src = image_barrier(small.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ);
                    device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[small_to_src]);
                    let blit = vk::ImageBlit::builder()
                        .src_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
                        .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: answer_width as i32, y: answer_height as i32, z: 1 }])
                        .dst_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
                        .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: width as i32, y: height as i32, z: 1 }])
                        .build();
                    // `s.model_answer.image` was already transitioned to
                    // `TRANSFER_DST_OPTIMAL` in `to_dst` above -- the blit target here
                    // is the same layout the direct-copy branch below would have used.
                    device.cmd_blit_image(self.cmd, small.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
                } else {
                    device.cmd_copy_buffer_to_image(self.cmd, s.staging_buffer, s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, frame_bytes)]);
                }
                let to_general = [
                    image_barrier(s.proxy.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                    image_barrier(s.model_answer.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ),
                ];
                device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_general);
            }
            let current_old = if cached_before { vk::ImageLayout::GENERAL } else { vk::ImageLayout::UNDEFINED };
            let output_old = if cached_before { vk::ImageLayout::TRANSFER_SRC_OPTIMAL } else { vk::ImageLayout::UNDEFINED };
            let to_compute = [
                image_barrier(s.original.image, current_old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE),
                image_barrier(s.output.image, output_old, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_WRITE),
            ];
            // This batch prepares both the upload destination and the compute
            // output. SHADER_WRITE needs COMPUTE_SHADER in the destination scope.
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_compute);
            device.cmd_copy_buffer_to_image(self.cmd, s.current_buffer, s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
            let original_general = image_barrier(s.original.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[original_general]);
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, std::slice::from_ref(&self.descriptor_set), &[]);
            // Mode 2 when the capture leg encoded the proxy, mode 1 when it could not
            // (see `compose.comp`'s own comments on both). Mode 2 is the encoded-proxy
            // ratio transfer -- the formula upstream uses, which needs no motion mask
            // because only a dimensionless relighting factor is carried across the
            // round trip rather than spatially-fixed additive detail. Mode 1 remains
            // the fallback for devices where the proxy cannot be encoded, since
            // feeding an unencoded proxy to mode 2 would make its ratio identically
            // one (proxy == original) and discard the answer.
            //
            // The live settings are threaded now rather than hardcoded: mode 2 reads
            // transfer_strength as a power on the ratio and max_ratio as the guard, so
            // both genuinely decide the picture here.
            let push = PushConstants {
                colour_strength: compose.colour_strength,
                transfer_strength: compose.transfer_strength,
                max_ratio: compose.max_ratio,
                bgr_order: bgr_order as u32,
                mode: if compose.proxy_encoded { 2 } else { 1 },
            };
            let push_bytes = std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<PushConstants>());
            device.cmd_push_constants(self.cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            device.cmd_dispatch(self.cmd, width.div_ceil(8), height.div_ceil(8), 1);
            let output_src = image_barrier(s.output.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[output_src]);
            device.cmd_copy_image_to_buffer(self.cmd, s.output.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, s.cached_buffer, &[image_copy_region(width, height, 0)]);
            let output_ready = vk::BufferMemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).buffer(s.cached_buffer).offset(0).size(frame_bytes).build();
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[output_ready], &[]);
            let target_to_dst = image_barrier(target_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[target_to_dst]);
            device.cmd_copy_buffer_to_image(self.cmd, s.cached_buffer, target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
            let to_present = image_barrier(target_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
        }
    }

    /// # Safety
    /// No GPU work referencing this slot's handles may still be in flight.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            if let Some(s) = &self.sized {
                s.original.destroy(device);
                s.model_answer.destroy(device);
                s.proxy.destroy(device);
                s.output.destroy(device);
                device.destroy_buffer(s.staging_buffer, None);
                device.free_memory(s.staging_memory, None);
                device.destroy_buffer(s.cached_buffer, None);
                device.free_memory(s.cached_memory, None);
                device.destroy_buffer(s.current_buffer, None);
                device.free_memory(s.current_memory, None);
            }
            if let Some((small_answer, _, _)) = &self.small_answer {
                small_answer.destroy(device);
            }
            device.destroy_fence(self.fence, None);
        }
    }
}

/// One independently fenced command/resource slot. Two of these
/// alternate in [`GpuCompose::dispatch_into_image_async`] so a slot is never reused
/// until its *previous* use (two dispatches ago) has genuinely finished.
struct AsyncSlot {
    slot: ComposeSlot,
}

/// The size-independent pipeline state (shader module, descriptor/pipeline layouts,
/// the compute pipeline itself) plus every [`ComposeSlot`]/[`AsyncSlot`] that shares
/// it. Built once and reused across every resolution -- each slot's own `Sized_` is
/// what actually changes on a resize.
pub struct GpuCompose {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    pool: vk::CommandPool,
    sync: ComposeSlot,
    async_slots: [AsyncSlot; ASYNC_SLOTS],
    next_async_slot: usize,
    present_semaphores: crate::present_sync::PresentSemaphores,
}

// SAFETY: every field is either a plain Vulkan handle or (inside a slot's own
// `Sized_`) a `vkMapMemory` pointer into memory that slot owns exclusively -- never
// aliased outside the `Mutex<State>` this whole struct always lives behind in
// `NeuralForgeDeviceInfo`, same reasoning as `capture::CaptureResources`.
unsafe impl Send for GpuCompose {}

impl GpuCompose {
    /// Builds the size-independent pipeline state plus every compose slot (the sync
    /// one and [`ASYNC_SLOTS`] async ones). `None` on any failure -- callers treat
    /// that as "the GPU compose path isn't available", falling back to
    /// [`super::apply`]'s CPU path, never a reason to stop presenting frames.
    pub fn new(device: &ash::Device, queue_family: u32) -> Option<Self> {
        let bindings: Vec<_> = (0..4)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        // SAFETY: `layout_info` is valid.
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }.ok()?;

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .build();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        // SAFETY: `pipeline_layout_info` is valid; `descriptor_set_layout` was just
        // created above.
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) } {
            Ok(l) => l,
            Err(_) => {
                // SAFETY: nothing else references `descriptor_set_layout` yet.
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return None;
            }
        };

        let Ok(code) = ash::util::read_spv(&mut std::io::Cursor::new(SPV)) else {
            // SAFETY: neither `pipeline_layout` nor `descriptor_set_layout` is
            // referenced anywhere else yet.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };
        let module_info = vk::ShaderModuleCreateInfo::builder().code(&code);
        // SAFETY: `module_info.code` is a valid SPIR-V module (`compose.spv`, compiled
        // from `shaders/compose.comp` via `glslangValidator -V`, validated with
        // `spirv-val` before being committed).
        let shader_module = match unsafe { device.create_shader_module(&module_info, None) } {
            Ok(m) => m,
            Err(_) => {
                // SAFETY: same reasoning as the branch above.
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };
        let entry_point = c"main";
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage_info).layout(pipeline_layout);
        // SAFETY: `pipeline_info` is valid; `shader_module`/`pipeline_layout` were
        // just created above and outlive this call.
        let pipeline_result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None) };
        // SAFETY: the module is never referenced again after pipeline creation
        // (successful or not) -- the Vulkan spec allows destroying it immediately.
        unsafe { device.destroy_shader_module(shader_module, None) };
        let pipeline = match pipeline_result {
            Ok(pipelines) => pipelines[0],
            Err(_) => {
                // SAFETY: same reasoning as the branches above.
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };

        // One sync slot plus `ASYNC_SLOTS` async ones, each needing its own 4
        // storage-image descriptors.
        let total_sets = 1 + ASYNC_SLOTS as u32;
        let pool_size = vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(4 * total_sets).build();
        let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(total_sets).pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: `pool_info` is valid.
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(_) => {
                // SAFETY: nothing else references these yet.
                unsafe {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return None;
            }
        };

        let pool_create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_create_info` is valid.
        let Ok(cmd_pool) = (unsafe { device.create_command_pool(&pool_create_info, None) }) else {
            // SAFETY: nothing else references these yet.
            unsafe {
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            return None;
        };

        let cleanup_partial = |device: &ash::Device| {
            // SAFETY: called only on a failure path below, before this function has
            // returned `Some` -- nothing outside this function references any of
            // these yet.
            unsafe {
                device.destroy_command_pool(cmd_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
        };

        let Some(sync) = ComposeSlot::new(device, cmd_pool, descriptor_pool, descriptor_set_layout) else {
            cleanup_partial(device);
            return None;
        };

        let mut async_slots = Vec::with_capacity(ASYNC_SLOTS);
        for _ in 0..ASYNC_SLOTS {
            let Some(slot) = ComposeSlot::new(device, cmd_pool, descriptor_pool, descriptor_set_layout) else {
                // SAFETY: `sync` and every already-built async slot so far own no GPU
                // work in flight (nothing has been submitted yet at construction time).
                unsafe {
                    sync.destroy(device);
                    for s in &async_slots {
                        let s: &AsyncSlot = s;
                        s.slot.destroy(device);
                    }
                }
                cleanup_partial(device);
                return None;
            };
            async_slots.push(AsyncSlot { slot });
        }

        Some(Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            pool: cmd_pool,
            sync,
            async_slots: async_slots.try_into().unwrap_or_else(|_| unreachable!("pushed exactly ASYNC_SLOTS elements above")),
            next_async_slot: 0,
            present_semaphores: Default::default(),
        })
    }

    /// Runs `shaders/compose.comp` against `original`/`model_answer` (both `RGBA8`,
    /// `width`x`height`), writing the composited result back into `model_answer` in
    /// place -- the same signature and in-place-overwrite convention
    /// [`super::apply::apply_rgba8`] uses, so `capture.rs` can call either
    /// interchangeably. `false` (leaving `model_answer` untouched) on any failure,
    /// fails open exactly like every other stage of the capture path.
    ///
    /// Downloads the result to a CPU-visible slice and fully waits for it -- use this
    /// when something on the CPU actually needs to see the bytes (a pending
    /// `capture_request` dump in particular). [`Self::dispatch_into_image_async`] is
    /// the fast, non-blocking path for the common case where nothing does.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        original: &[u8],
        model_answer: &mut [u8],
        colour_strength: f32,
        transfer_strength: f32,
        max_ratio: f32,
        bgr_order: bool,
    ) -> bool {
        // SAFETY: `physical_device` is the device this instance was created against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let Some(frame_bytes) = self.sync.begin_ensured(device, &mem_props, width, height, original, model_answer) else { return false };

        // SAFETY: `self.sync.cmd` was allocated with `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.sync.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.sync.cmd` was just reset.
        if unsafe { device.begin_command_buffer(self.sync.cmd, &begin_info) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.cmd` was just begun above.
        unsafe {
            self.sync.record_upload_and_compute(
                device,
                self.pipeline,
                self.pipeline_layout,
                width,
                height,
                frame_bytes,
                colour_strength,
                transfer_strength,
                max_ratio,
                bgr_order,
            )
        };
        if unsafe { device.end_command_buffer(self.sync.cmd) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.fence` starts signaled or was reset+waited-on by this
        // same function's previous call.
        if unsafe { device.reset_fences(&[self.sync.fence]) }.is_err() {
            return false;
        }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.sync.cmd)).build();
        // SAFETY: `self.sync.cmd` was just recorded and ended above.
        if unsafe { device.queue_submit(queue, &[submit], self.sync.fence) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.fence` was just submitted against above.
        if unsafe { device.wait_for_fences(&[self.sync.fence], true, u64::MAX) }.is_err() {
            return false;
        }

        let s = self.sync.sized.as_ref().expect("ensured by begin_ensured above");
        // SAFETY: the fence wait above guarantees the download copy has completed;
        // `s.staging_ptr` is host-coherent (no explicit invalidate needed).
        unsafe { std::ptr::copy_nonoverlapping(s.staging_ptr, model_answer.as_mut_ptr(), frame_bytes as usize) };
        true
    }

    /// Same composition as [`Self::dispatch`], but writes the result directly into
    /// `target_image` (assumed already `TRANSFER_DST_OPTIMAL` -- exactly the layout
    /// `capture::run`'s own stage 1 already leaves the real swapchain image in) instead
    /// of downloading it to a CPU-visible slice, and in the *same* command
    /// buffer/submission as the compute dispatch itself, restoring `target_image` to
    /// `PRESENT_SRC_KHR` before returning. Still fully synchronous (blocks on this
    /// dispatch's own fence) -- [`Self::dispatch_into_image_async`] is the same
    /// write-into-image trick without that block.
    ///
    /// Deliberately routes through the sync slot's own `staging_buffer` rather than a
    /// raw `vkCmdCopyImage` straight from `output`: a raw image-to-image copy between
    /// two images of different formats is a byte-for-byte copy with no
    /// channel-swizzle, so it would silently corrupt colors if `target_image`'s real
    /// format ever differs from this module's own hardcoded `FORMAT` -- e.g. a real
    /// `B8G8R8A8` swapchain vs. this module's `R8G8B8A8`. A buffer-to-image copy has no
    /// format attached to the source either, so it's exactly as agnostic to a *size*/
    /// layout mismatch, the same reasoning `capture.rs`'s own original stage 2 already
    /// relied on. **This does NOT make channel order automatically correct, though**
    /// (a real, confirmed bug this comment used to claim otherwise about, 2026-09-11):
    /// the bytes in that buffer are only correct for `target_image`'s real component
    /// order because the shader itself now swizzles its output based on `bgr_order`
    /// (see `compose.comp`'s `params.bgr_order` doc comment) -- the buffer copy's own
    /// format-obliviousness was never the thing making this correct.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into_image(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        original: &[u8],
        model_answer: &[u8],
        colour_strength: f32,
        transfer_strength: f32,
        max_ratio: f32,
        bgr_order: bool,
        target_image: vk::Image,
    ) -> bool {
        // SAFETY: `physical_device` is the device this instance was created against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let Some(frame_bytes) = self.sync.begin_ensured(device, &mem_props, width, height, original, model_answer) else { return false };

        // SAFETY: `self.sync.cmd` was allocated with `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.sync.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.sync.cmd` was just reset.
        if unsafe { device.begin_command_buffer(self.sync.cmd, &begin_info) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.cmd` was just begun above.
        unsafe {
            self.sync.record_upload_and_compute(
                device,
                self.pipeline,
                self.pipeline_layout,
                width,
                height,
                frame_bytes,
                colour_strength,
                transfer_strength,
                max_ratio,
                bgr_order,
            );
            self.sync.record_copy_into_image(device, width, height, frame_bytes, target_image);
        }
        if unsafe { device.end_command_buffer(self.sync.cmd) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.fence` starts signaled or was reset+waited-on by this
        // same function's previous call.
        if unsafe { device.reset_fences(&[self.sync.fence]) }.is_err() {
            return false;
        }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.sync.cmd)).build();
        // SAFETY: `self.sync.cmd` was just recorded and ended above.
        if unsafe { device.queue_submit(queue, &[submit], self.sync.fence) }.is_err() {
            return false;
        }
        // SAFETY: `self.sync.fence` was just submitted against above.
        unsafe { device.wait_for_fences(&[self.sync.fence], true, u64::MAX) }.is_ok()
    }

    /// `answer_width`/`answer_height` are `answer`'s own resolution -- pass
    /// `(width, height)` for the pre-`working_scale` behavior (`answer` must then be
    /// exactly `width*height*4` bytes, same as always); a smaller pair triggers the
    /// GPU-blit upscale in `record_temporal_delta_into_image`. Deliberately does not
    /// support `answer_width*answer_height > width*height` (a `working_scale` above
    /// `1.0`, supersampling): the shared staging buffer this function uploads into is
    /// sized from `(width, height)` alone (`ensure_sized`, unchanged), so a genuinely
    /// larger answer would overflow it -- caught below and treated as "nothing to
    /// composite this frame" rather than risking that. `working_scale` below `1.0` is
    /// unaffected; only the above-`1.0` (supersampling) direction is out of scope for
    /// this compose-side path tonight.
    #[allow(clippy::too_many_arguments)]
    pub fn present_temporal_delta_async(
        &mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue: vk::Queue,
        width: u32, height: u32, answer_width: u32, answer_height: u32, base: &[u8], answer: &[u8], generation: u64, bgr_order: bool, target_image: vk::Image, compose: ComposeParams,
    ) -> Option<vk::Semaphore> {
        let semaphore = self.present_semaphores.get(target_image, || unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()
        })?;
        let idx = self.next_async_slot; self.next_async_slot = (self.next_async_slot + 1) % ASYNC_SLOTS;
        let slot = &mut self.async_slots[idx];
        if unsafe { device.wait_for_fences(&[slot.slot.fence], true, u64::MAX) }.is_err() { return None; }
        let bytes = (u64::from(width) * u64::from(height) * 4) as usize;
        let answer_bytes = (u64::from(answer_width) * u64::from(answer_height) * 4) as usize;
        if generation == 0 || base.len() < bytes || answer.len() < answer_bytes || answer_bytes > bytes { return None; }
        let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        if !slot.slot.ensure_sized(device, &props, width, height) { return None; }
        let scaled_answer = answer_width != width || answer_height != height;
        if scaled_answer && !slot.slot.ensure_small_answer(device, &props, answer_width, answer_height) { return None; }
        let update = slot.slot.cached_generation != generation;
        if update {
            let s = slot.slot.sized.as_ref().expect("ensured");
            unsafe { std::ptr::copy_nonoverlapping(base.as_ptr(), s.staging_ptr, bytes); std::ptr::copy_nonoverlapping(answer.as_ptr(), s.staging_ptr.add(bytes), answer_bytes); }
        }
        if unsafe { device.reset_command_buffer(slot.slot.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() { return None; }
        let begin = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if unsafe { device.begin_command_buffer(slot.slot.cmd, &begin) }.is_err() { return None; }
        unsafe { slot.slot.record_temporal_delta_into_image(device, self.pipeline, self.pipeline_layout, width, height, answer_width, answer_height, bytes as u64, update, bgr_order, target_image, compose); }
        if unsafe { device.end_command_buffer(slot.slot.cmd) }.is_err() || unsafe { device.reset_fences(&[slot.slot.fence]) }.is_err() { return None; }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&slot.slot.cmd)).signal_semaphores(std::slice::from_ref(&semaphore)).build();
        if unsafe { device.queue_submit(queue, &[submit], slot.slot.fence) }.is_err() { return None; }
        if update { slot.slot.cached_generation = generation; }
        Some(semaphore)
    }

    /// Presents a raw helper answer every frame while uploading it only when the
    /// helper produces a newer generation.  The normal compose path remains for
    /// callers that need its math; this path is for the held-answer presentation
    /// policy in `capture::run`.
    pub fn present_cached_raw_async(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        raw_answer: &[u8],
        generation: u64,
        target_image: vk::Image,
    ) -> Option<vk::Semaphore> {
        let semaphore = self.present_semaphores.get(target_image, || unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()
        })?;
        let idx = self.next_async_slot;
        self.next_async_slot = (self.next_async_slot + 1) % ASYNC_SLOTS;
        let async_slot = &mut self.async_slots[idx];
        if unsafe { device.wait_for_fences(&[async_slot.slot.fence], true, u64::MAX) }.is_err() {
            return None;
        }
        let frame_bytes = (u64::from(width) * u64::from(height) * 4) as usize;
        if raw_answer.len() < frame_bytes || generation == 0 {
            return None;
        }
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        if !async_slot.slot.ensure_sized(device, &mem_props, width, height) {
            return None;
        }
        let update = async_slot.slot.cached_generation != generation;
        if update {
            let s = async_slot.slot.sized.as_ref().expect("just ensured above");
            if s.staging_capacity < frame_bytes as u64 { return None; }
            unsafe { std::ptr::copy_nonoverlapping(raw_answer.as_ptr(), s.staging_ptr, frame_bytes); }
            async_slot.slot.cached_generation = generation;
        }
        if unsafe { device.reset_command_buffer(async_slot.slot.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() { return None; }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if unsafe { device.begin_command_buffer(async_slot.slot.cmd, &begin_info) }.is_err() { return None; }
        unsafe { async_slot.slot.record_cached_raw_into_image(device, width, height, frame_bytes as u64, update, target_image); }
        if unsafe { device.end_command_buffer(async_slot.slot.cmd) }.is_err() { return None; }
        if unsafe { device.reset_fences(&[async_slot.slot.fence]) }.is_err() { return None; }
        let submit = vk::SubmitInfo::builder()
            .command_buffers(std::slice::from_ref(&async_slot.slot.cmd))
            .signal_semaphores(std::slice::from_ref(&semaphore))
            .build();
        if unsafe { device.queue_submit(queue, &[submit], async_slot.slot.fence) }.is_err() { return None; }
        Some(semaphore)
    }

    /// The real per-frame fast path: same composition and same write-straight-into-
    /// `target_image` trick as [`Self::dispatch_into_image`], but **does not block the
    /// CPU** on this dispatch's own completion. Submits with a signal semaphore and
    /// returns it immediately; the caller (`capture::run`, then `device.rs`'s present
    /// hook) must add that semaphore to the *real* `vkQueuePresentKHR` call's own wait
    /// list, so the presentation engine -- a GPU-side dependency, not a CPU one --
    /// is what actually waits for this frame's compute work before displaying it.
    ///
    /// Double-buffered across [`ASYNC_SLOTS`] independent [`AsyncSlot`]s (own images,
    /// staging buffer, command buffer, fence) specifically so that never
    /// blocking on *this* call's own fence doesn't mean never blocking at all: the one
    /// necessary wait is on the slot's *own* fence, from its *previous* use
    /// ([`ASYNC_SLOTS`] dispatches ago) -- immediately before reusing its resources,
    /// not before returning this frame's result. By the time a slot comes back around,
    /// the GPU has almost always finished with it long ago (an entire other frame's
    /// worth of capture + SHM round trip has elapsed on the CPU in between), so that
    /// wait is normally instant; it exists purely so this can never race a slot's own
    /// still-in-flight prior work, not to reintroduce the per-frame block this method
    /// exists to remove.
    ///
    /// Presentation semaphores are keyed by acquired target image, independently
    /// of these command slots. Reacquisition orders reuse after the previous
    /// presentation's wait. Slot fences alone do not establish that ordering.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_into_image_async(
        &mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        width: u32,
        height: u32,
        original: &[u8],
        model_answer: &[u8],
        colour_strength: f32,
        transfer_strength: f32,
        max_ratio: f32,
        bgr_order: bool,
        target_image: vk::Image,
    ) -> Option<vk::Semaphore> {
        let semaphore = self.present_semaphores.get(target_image, || unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()
        })?;
        let idx = self.next_async_slot;
        self.next_async_slot = (self.next_async_slot + 1) % ASYNC_SLOTS;
        let async_slot = &mut self.async_slots[idx];

        // SAFETY: `async_slot.slot.fence` was signaled at creation, or by this same
        // slot's own previous dispatch -- waiting here (not before returning that
        // previous dispatch's own result) is exactly what makes this "async": the
        // block only ever happens `ASYNC_SLOTS` dispatches later, immediately before
        // this specific slot's resources are touched again, never on the frame that
        // just submitted them.
        if unsafe { device.wait_for_fences(&[async_slot.slot.fence], true, u64::MAX) }.is_err() {
            return None;
        }

        // SAFETY: `physical_device` is the device this instance was created against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let Some(frame_bytes) = async_slot.slot.begin_ensured(device, &mem_props, width, height, original, model_answer) else { return None };

        // SAFETY: `async_slot.slot.cmd` was allocated with `RESET_COMMAND_BUFFER`, and
        // the fence wait above guarantees any previous use of it has completed.
        if unsafe { device.reset_command_buffer(async_slot.slot.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return None;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `async_slot.slot.cmd` was just reset.
        if unsafe { device.begin_command_buffer(async_slot.slot.cmd, &begin_info) }.is_err() {
            return None;
        }
        // SAFETY: `async_slot.slot.cmd` was just begun above.
        unsafe {
            async_slot.slot.record_upload_and_compute(
                device,
                self.pipeline,
                self.pipeline_layout,
                width,
                height,
                frame_bytes,
                colour_strength,
                transfer_strength,
                max_ratio,
                bgr_order,
            );
            async_slot.slot.record_copy_into_image(device, width, height, frame_bytes, target_image);
        }
        if unsafe { device.end_command_buffer(async_slot.slot.cmd) }.is_err() {
            return None;
        }
        // SAFETY: the fence wait above guarantees this fence is not in the signaled
        // state from a still-pending wait -- safe to reset before resubmitting.
        if unsafe { device.reset_fences(&[async_slot.slot.fence]) }.is_err() {
            return None;
        }
        let submit = vk::SubmitInfo::builder()
            .command_buffers(std::slice::from_ref(&async_slot.slot.cmd))
            .signal_semaphores(std::slice::from_ref(&semaphore))
            .build();
        // SAFETY: `async_slot.slot.cmd` was just recorded and ended above. Not waiting
        // on `async_slot.slot.fence` here is the entire point of this method -- see
        // its own doc comment for why that's still sound.
        if unsafe { device.queue_submit(queue, &[submit], async_slot.slot.fence) }.is_err() {
            return None;
        }
        Some(semaphore)
    }

    pub fn retire_present_images(&mut self, images: &[vk::Image]) {
        self.present_semaphores.retire(images);
    }

    /// # Safety
    /// Must only be called at device-destruction time, with no submitted work
    /// referencing these handles still in flight.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            self.present_semaphores.destroy(device);
            self.sync.destroy(device);
            for async_slot in &self.async_slots {
                async_slot.slot.destroy(device);

            }
            device.destroy_command_pool(self.pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// A real `VkInstance`/`VkDevice`/queue through the system Vulkan loader --
/// `None` if no loader/ICD is available in this environment (this crate's own
/// `examples/smoke.rs` doc comment already establishes lavapipe is enough, no real
/// GPU needed, for exactly this kind of check). Shared by every `#[test]` below
/// rather than each standing up its own instance/device.
#[cfg(test)]
pub(crate) fn test_device() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32)> {
    // SAFETY: same reasoning as `examples/smoke.rs`'s identical call.
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is valid.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;
    // SAFETY: `instance` was just created and outlives every use of `physical_device`.
    let physical_device = *unsafe { instance.enumerate_physical_devices() }.ok()?.first()?;
    let queue_family = 0;
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(queue_family).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    // SAFETY: `device_create_info` is valid; every physical device has a family 0
    // (the Vulkan spec guarantees at least one queue family).
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
    // SAFETY: `device`/family/index 0 match what `device_create_info` just requested.
    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    Some((entry, instance, physical_device, device, queue, queue_family))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real correctness check for this whole module: dispatches
    /// `shaders/compose.comp` on a real (if software) Vulkan device and confirms
    /// its output matches [`super::super::apply::apply_rgba8`] -- the same
    /// already-real-hardware-verified CPU reference (see the crate's `CLAUDE.md`
    /// entry) -- to within a small per-channel tolerance (GPU and CPU `pow`/`cbrt`
    /// implementations are never bit-identical, only close). A real, non-uniform
    /// test image (not one flat color) so the tone-mapping/OkLab branches this
    /// algorithm actually has are exercised, not just the identity case.
    #[test]
    fn gpu_dispatch_matches_the_cpu_reference() {
        let Some((_entry, instance, physical_device, device, queue, _queue_family)) = test_device() else {
            eprintln!("gpu_dispatch_matches_the_cpu_reference: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, 0) else {
            eprintln!("gpu_dispatch_matches_the_cpu_reference: GpuCompose::new failed (e.g. no compute-capable queue), skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        // A real gradient plus a few deliberately out-of-band pixels, not a flat
        // color -- exercises the headroom/no-headroom branches, the OkLab hue
        // correction, and the gamut-compression path all at once.
        let original: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 37 % 256) as u8;
                [t, t.wrapping_add(64), t.wrapping_add(128), 255]
            })
            .collect();
        let model_answer: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 53 % 256) as u8;
                [t.wrapping_add(20), t, t.wrapping_add(200), 255]
            })
            .collect();

        let (colour_strength, transfer_strength, max_ratio) = (0.7, 0.8, 2.0);

        let mut gpu_result = model_answer.clone();
        let ok = gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut gpu_result, colour_strength, transfer_strength, max_ratio, false);
        assert!(ok, "GpuCompose::dispatch returned false");

        let mut cpu_result = model_answer.clone();
        super::super::apply::apply_rgba8(&original, &mut cpu_result, colour_strength, transfer_strength, max_ratio, 0, false);

        let mut max_diff = 0i32;
        for (g, c) in gpu_result.chunks_exact(4).zip(cpu_result.chunks_exact(4)) {
            for ch in 0..3 {
                max_diff = max_diff.max((i32::from(g[ch]) - i32::from(c[ch])).abs());
            }
        }
        assert!(max_diff <= 3, "GPU and CPU composition diverge by up to {max_diff} (expected <= 3): gpu={gpu_result:?} cpu={cpu_result:?}");

        // SAFETY: `gpu`'s own fence wait inside `dispatch` guarantees no GPU work
        // is in flight; nothing else references `device`/`instance`.
        unsafe {
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    #[test]
    fn bgr_order_produces_the_same_true_colors_as_rgb_order_on_swapped_bytes() {
        // Real regression test for the 2026-09-11 channel-swap bug: feeds the exact
        // same semantic colors as `gpu_dispatch_matches_the_cpu_reference` above, but
        // with R and B physically swapped in the byte layout (simulating a real
        // `B8G8R8A8` swapchain) and `bgr_order: true` on every call. If the swizzle in
        // `compose.comp`/`apply_rgba8` is correct, the *true* colors this produces
        // (read back through the swapped indices) must match `gpu_dispatch_matches_
        // the_cpu_reference`'s own RGB-order result exactly -- not just GPU agreeing
        // with CPU (both could be equally wrong the same way), but this whole
        // BGR-order run agreeing with that separate, independently-computed RGB-order
        // run.
        let Some((_entry, instance, physical_device, device, queue, _queue_family)) = test_device() else {
            eprintln!("bgr_order_produces_the_same_true_colors_as_rgb_order_on_swapped_bytes: no Vulkan loader/ICD, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, 0) else {
            eprintln!("bgr_order_produces_the_same_true_colors_as_rgb_order_on_swapped_bytes: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        // Same true colors as `gpu_dispatch_matches_the_cpu_reference`, but stored
        // B,G,R,A -- swap index 0 and 2 relative to that test's own byte arrays.
        let original_bgr: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 37 % 256) as u8;
                [t.wrapping_add(128), t.wrapping_add(64), t, 255]
            })
            .collect();
        let model_answer_bgr: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 53 % 256) as u8;
                [t.wrapping_add(200), t, t.wrapping_add(20), 255]
            })
            .collect();

        let (colour_strength, transfer_strength, max_ratio) = (0.7, 0.8, 2.0);

        let mut gpu_result = model_answer_bgr.clone();
        let ok = gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original_bgr, &mut gpu_result, colour_strength, transfer_strength, max_ratio, true);
        assert!(ok, "GpuCompose::dispatch returned false");

        let mut cpu_result = model_answer_bgr.clone();
        super::super::apply::apply_rgba8(&original_bgr, &mut cpu_result, colour_strength, transfer_strength, max_ratio, 0, true);

        // GPU and CPU must still agree with each other under bgr_order too.
        let mut max_diff = 0i32;
        for (g, c) in gpu_result.chunks_exact(4).zip(cpu_result.chunks_exact(4)) {
            for ch in 0..3 {
                max_diff = max_diff.max((i32::from(g[ch]) - i32::from(c[ch])).abs());
            }
        }
        assert!(max_diff <= 3, "GPU and CPU composition diverge by up to {max_diff} under bgr_order=true (expected <= 3)");

        // The true colors (read back through the swapped R/B indices) must match
        // `gpu_dispatch_matches_the_cpu_reference`'s own RGB-order CPU result exactly
        // -- proving the swizzle round-trips correctly end to end, not just that GPU
        // and CPU are consistently wrong the same way.
        let original_rgb: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 37 % 256) as u8;
                [t, t.wrapping_add(64), t.wrapping_add(128), 255]
            })
            .collect();
        let model_answer_rgb: Vec<u8> = (0..pixel_count)
            .flat_map(|i| {
                let t = (i * 53 % 256) as u8;
                [t.wrapping_add(20), t, t.wrapping_add(200), 255]
            })
            .collect();
        let mut reference_rgb = model_answer_rgb.clone();
        super::super::apply::apply_rgba8(&original_rgb, &mut reference_rgb, colour_strength, transfer_strength, max_ratio, 0, false);

        for (px_bgr, px_rgb_ref) in gpu_result.chunks_exact(4).zip(reference_rgb.chunks_exact(4)) {
            // px_bgr is [B, G, R, A] -- reorder to true [R, G, B] before comparing.
            let true_rgb = [px_bgr[2], px_bgr[1], px_bgr[0]];
            let reference = [px_rgb_ref[0], px_rgb_ref[1], px_rgb_ref[2]];
            for ch in 0..3 {
                assert!(
                    (i32::from(true_rgb[ch]) - i32::from(reference[ch])).abs() <= 3,
                    "bgr_order=true's true colors {true_rgb:?} must match the independent RGB-order reference {reference:?}"
                );
            }
        }

        // SAFETY: `gpu`'s own fence wait inside `dispatch` guarantees no GPU work
        // is in flight; nothing else references `device`/`instance`.
        unsafe {
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A second real dispatch at a *different* size than the first test used,
    /// through the *same* `GpuCompose` instance-creation path -- confirms
    /// `ensure_sized`'s rebuild-on-resize path (not just its happy-path "already
    /// the right size" branch) actually works, real device included.
    #[test]
    fn gpu_dispatch_handles_a_resize() {
        let Some((_entry, instance, physical_device, device, queue, _queue_family)) = test_device() else {
            eprintln!("gpu_dispatch_handles_a_resize: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, 0) else {
            eprintln!("gpu_dispatch_handles_a_resize: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        for &(width, height) in &[(4u32, 4u32), (16u32, 12u32)] {
            let pixel_count = (width * height) as usize;
            let original = vec![128u8; pixel_count * 4];
            let mut answer = vec![100u8; pixel_count * 4];
            let ok = gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut answer, 1.0, 1.0, 2.0, false);
            assert!(ok, "dispatch failed at {width}x{height}");
        }

        // SAFETY: same reasoning as the test above.
        unsafe {
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A small host-visible/host-coherent buffer pre-filled with `bytes`, for
    /// uploading test fixture content into an image via `cmd_copy_buffer_to_image`.
    /// Caller destroys/frees both returned handles once the upload's fence signals.
    fn build_upload_staging(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, bytes: &[u8]) -> (vk::Buffer, vk::DeviceMemory) {
        let buf_info = vk::BufferCreateInfo::builder().size(bytes.len() as u64).usage(vk::BufferUsageFlags::TRANSFER_SRC).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buf_info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let type_index = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };
        let ptr = unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.unwrap().cast::<u8>();
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        unsafe { device.unmap_memory(memory) };
        (buffer, memory)
    }

    #[test]
    fn present_temporal_delta_async_upscales_a_smaller_answer_and_rejects_an_oversized_one() {
        // `working_scale`'s compose-side counterpart to `capture.rs`'s own
        // `working_scale_sends_a_genuinely_smaller_proxy_and_still_composites` --
        // that test proves the *pipeline* end to end; this one proves this specific
        // function's own two documented behaviors directly: (1) a smaller `answer`
        // genuinely lands in the output (not silently dropped/left native), and (2)
        // an `answer` larger than the frame (the out-of-scope `working_scale > 1.0`
        // case, see this function's own doc comment) is safely rejected rather than
        // overflowing the shared staging buffer.
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("present_temporal_delta_async_upscales_a_smaller_answer_and_rejects_an_oversized_one: no Vulkan loader/ICD, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            eprintln!("present_temporal_delta_async_upscales_a_smaller_answer_and_rejects_an_oversized_one: GpuCompose::new failed, skipping");
            unsafe { device.destroy_device(None); instance.destroy_instance(None) };
            return;
        };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();

        let (width, height) = (16u32, 16u32);
        let (answer_width, answer_height) = (8u32, 8u32);
        let base = vec![50u8; (width * height * 4) as usize]; // what the model was shown ("proxy").
        let answer = vec![200u8; (answer_width * answer_height * 4) as usize]; // the model's answer: brighter than `base`, a real delta.
        let target = make_target_image(&device, &mem_props, width, height);
        // `record_temporal_delta_into_image` reads the shader's "current frame"
        // (`u_original`) straight off `target_image`'s own live content, *not* from
        // `base` -- the motion mask compares that against `base` ("proxy") to decide
        // how much of the model's delta to keep. For this test to actually exercise
        // the delta (not have the mask suppress it because the target starts
        // uninitialized, far from `base`), `target` needs `base`'s own bytes as its
        // starting content -- a stable/motionless frame, exactly the case
        // `stability` should hold near its maximum for.
        let staging = build_upload_staging(&device, &mem_props, &base);
        let cmd = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        unsafe {
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            let to_dst = image_barrier(target.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            device.cmd_copy_buffer_to_image(cmd, staging.0, target.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_region(width, height, 0)]);
            let to_present = image_barrier(target.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_buffer(staging.0, None);
            device.free_memory(staging.1, None);
        }

        // (1) A smaller answer must still be accepted and actually change the output.
        let sem = gpu.present_temporal_delta_async(&device, &instance, physical_device, queue, width, height, answer_width, answer_height, &base, &answer, 1, false, target.image, crate::composition::gpu::ComposeParams { colour_strength: 1.0, transfer_strength: 1.0, max_ratio: 2.0, proxy_encoded: false });
        assert!(sem.is_some(), "a genuinely smaller answer must still be composited, not rejected");
        let sem = sem.unwrap();
        let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
        unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build()], wait_fence).unwrap();
            device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
            device.destroy_fence(wait_fence, None);
        }
        // Present->transfer-src round trip so `read_back_image` (which assumes
        // `PRESENT_SRC_KHR`) can read it -- `present_temporal_delta_async` itself
        // already leaves `target` in `PRESENT_SRC_KHR` (see its own barrier at the
        // end of `record_temporal_delta_into_image`), so no extra transition needed.
        let out = read_back_image(&device, &mem_props, queue, pool, target.image, width, height);
        // `base` (50) is uniform, so the motion mask holds `stability` at its
        // maximum everywhere -- the composited pixel should sit meaningfully above
        // `base` and not still be exactly `base`, proving the (upscaled) answer
        // really reached the shader, not a silently-empty/no-op composite.
        let avg = out.chunks_exact(4).map(|p| p[0] as u32).sum::<u32>() / (out.len() as u32 / 4);
        assert!(avg > 60, "composited output (avg r={avg}) should be visibly brighter than the 50-valued base if the upscaled answer reached the shader");

        // (2) An answer *larger* than the frame (working_scale > 1.0, out of scope
        // for this function -- see its own doc comment) must be rejected, not
        // overflow the shared staging buffer.
        let big_answer = vec![200u8; ((width + 8) * (height + 8) * 4) as usize];
        let oversized = gpu.present_temporal_delta_async(&device, &instance, physical_device, queue, width, height, width + 8, height + 8, &base, &big_answer, 2, false, target.image, crate::composition::gpu::ComposeParams { colour_strength: 1.0, transfer_strength: 1.0, max_ratio: 2.0, proxy_encoded: false });
        assert!(oversized.is_none(), "an answer larger than the frame must be safely rejected, not overflow the staging buffer");

        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            target.destroy(&device);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    fn make_target_image(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32) -> Image {
        create_storage_image(device, mem_props, width, height, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .expect("failed to create the test's own target image")
    }

    /// Reads `target` (assumed already `PRESENT_SRC_KHR`, transitioning it to
    /// `TRANSFER_SRC_OPTIMAL` and back is not needed since the test throws the image
    /// away afterward) back into a fresh CPU buffer, blocking until done -- test-only
    /// verification plumbing; production code never needs to read the real swapchain
    /// image back.
    fn read_back_image(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image, width: u32, height: u32) -> Vec<u8> {
        let frame_bytes = (width * height * 4) as u64;
        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();

        let buf_info = vk::BufferCreateInfo::builder().size(frame_bytes).usage(vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buf_info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let type_index = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let ptr = unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.unwrap().cast::<u8>();

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_src = image_barrier(image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[image_copy_region(width, height, 0)]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        }

        let result = unsafe { std::slice::from_raw_parts(ptr, frame_bytes as usize) }.to_vec();
        unsafe {
            device.unmap_memory(memory);
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
        result
    }

    /// Puts a freshly-created (`UNDEFINED`) test image into `PRESENT_SRC_KHR` --
    /// `record_copy_into_image`'s real precondition (2026-09-10) now that it manages
    /// its own `PRESENT_SRC_KHR -> TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR` round trip
    /// instead of assuming the caller already left it in `TRANSFER_DST_OPTIMAL`, the
    /// same state any real swapchain image is already in per `vkQueuePresentKHR`'s own
    /// contract -- this stands in for that real precondition.
    fn transition_to_present_src(device: &ash::Device, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image) {
        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_present = image_barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::empty(), vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
    }

    /// The real point of `dispatch_into_image`: confirms it produces the *same*
    /// composited result as [`GpuCompose::dispatch`] when writing directly into a
    /// target image instead of a CPU slice -- real device, real image transitions,
    /// real merged submission, not just "doesn't return false."
    #[test]
    fn dispatch_into_image_matches_dispatch() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("dispatch_into_image_matches_dispatch: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            eprintln!("dispatch_into_image_matches_dispatch: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        let original: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 41 % 256) as u8; [t, t.wrapping_add(90), t.wrapping_add(30), 255] }).collect();
        let model_answer: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 61 % 256) as u8; [t.wrapping_add(10), t, t.wrapping_add(180), 255] }).collect();
        let (colour_strength, transfer_strength, max_ratio) = (0.6, 0.9, 2.0);

        // The reference: `dispatch`'s already-verified CPU-visible path.
        let mut expected = model_answer.clone();
        assert!(gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut expected, colour_strength, transfer_strength, max_ratio, false));

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");

        // A standalone target image, standing in for a real swapchain image --
        // `dispatch_into_image`'s own contract only requires `TRANSFER_DST_OPTIMAL`,
        // which `capture::run`'s stage 1 already guarantees for the real one.
        let target = make_target_image(&device, &mem_props, width, height);
        transition_to_present_src(&device, queue, pool, target.image);

        let mut model_answer_for_direct = model_answer.clone();
        let ok = gpu.dispatch_into_image(
            &device, &instance, physical_device, queue, width, height, &original, &mut model_answer_for_direct,
            colour_strength, transfer_strength, max_ratio, false, target.image,
        );
        assert!(ok, "dispatch_into_image returned false");

        let actual = read_back_image(&device, &mem_props, queue, pool, target.image, width, height);
        assert_eq!(actual, expected, "dispatch_into_image's target image content must match dispatch's CPU-visible result exactly (same command sequence, same inputs)");

        // SAFETY: `read_back_image`'s own fence wait guarantees no GPU work is in flight.
        unsafe {
            target.destroy(&device);
            device.destroy_command_pool(pool, None);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    #[test]
    fn cached_raw_present_preserves_bytes_and_reuses_the_device_cache() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("cached_raw_present_preserves_bytes_and_reuses_the_device_cache: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            unsafe { device.destroy_device(None); instance.destroy_instance(None); }
            return;
        };
        let (width, height) = (8u32, 8u32);
        let raw: Vec<u8> = (0..(width * height) as usize).flat_map(|i| {
            let t = (i * 47 % 256) as u8;
            [t.wrapping_add(200), t, t.wrapping_add(70), 255]
        }).collect();
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let targets = [make_target_image(&device, &mem_props, width, height), make_target_image(&device, &mem_props, width, height)];
        for target in &targets { transition_to_present_src(&device, queue, pool, target.image); }
        for target in &targets {
            let sem = gpu.present_cached_raw_async(&device, &instance, physical_device, queue, width, height, &raw, 1, target.image).expect("cached raw present failed");
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
            let stage = vk::PipelineStageFlags::ALL_COMMANDS;
            let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&stage)).build();
            unsafe {
                device.queue_submit(queue, &[submit], fence).unwrap();
                device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                device.destroy_fence(fence, None);
            }
            assert_eq!(read_back_image(&device, &mem_props, queue, pool, target.image, width, height), raw);
        }
        unsafe {
            for target in &targets { target.destroy(&device); }
            device.destroy_command_pool(pool, None);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// The real point of `dispatch_into_image_async`: confirms that after explicitly
    /// waiting on the semaphore it returns (standing in for what the real present call
    /// does), the target image holds the same real composited result the synchronous
    /// `dispatch_into_image` produces for identical inputs -- not just "returns a
    /// semaphore", but "the semaphore actually gates real, correct, completed work."
    #[test]
    fn dispatch_into_image_async_matches_dispatch_into_image() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("dispatch_into_image_async_matches_dispatch_into_image: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            eprintln!("dispatch_into_image_async_matches_dispatch_into_image: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        let original: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 29 % 256) as u8; [t, t.wrapping_add(50), t.wrapping_add(140), 255] }).collect();
        let model_answer: Vec<u8> = (0..pixel_count).flat_map(|i| { let t = (i * 71 % 256) as u8; [t.wrapping_add(5), t, t.wrapping_add(220), 255] }).collect();
        let (colour_strength, transfer_strength, max_ratio) = (0.5, 1.0, 2.0);

        let mut expected = model_answer.clone();
        assert!(gpu.dispatch(&device, &instance, physical_device, queue, width, height, &original, &mut expected, colour_strength, transfer_strength, max_ratio, false));

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");

        let target = make_target_image(&device, &mem_props, width, height);
        transition_to_present_src(&device, queue, pool, target.image);

        let sem = gpu.dispatch_into_image_async(
            &device, &instance, physical_device, queue, width, height, &original, &model_answer,
            colour_strength, transfer_strength, max_ratio, false, target.image,
        );
        let Some(sem) = sem else { panic!("dispatch_into_image_async returned None") };

        // Stand in for what the real present call does: wait on the returned
        // semaphore before touching the image at all. A trivial submit with no
        // command buffers, just a wait, is the simplest way to consume it here.
        let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
        let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
        unsafe {
            device.queue_submit(queue, &[submit], wait_fence).unwrap();
            device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
            device.destroy_fence(wait_fence, None);
        }

        let actual = read_back_image(&device, &mem_props, queue, pool, target.image, width, height);
        assert_eq!(actual, expected, "dispatch_into_image_async's target image content (after waiting on its semaphore) must match dispatch's result exactly");

        // SAFETY: the explicit semaphore wait above (via `wait_fence`) guarantees the
        // async dispatch's GPU work, including its own fence signal, has completed.
        unsafe {
            target.destroy(&device);
            device.destroy_command_pool(pool, None);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// Drives `dispatch_into_image_async` across more dispatches than there are async
    /// slots, confirming slot reuse (waiting on a slot's own fence from its previous
    /// use, two dispatches back) is actually safe against a real device -- not just
    /// "the first two calls work."
    #[test]
    fn dispatch_into_image_async_survives_many_slot_reuses() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("dispatch_into_image_async_survives_many_slot_reuses: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let Some(mut gpu) = GpuCompose::new(&device, queue_family) else {
            eprintln!("dispatch_into_image_async_survives_many_slot_reuses: GpuCompose::new failed, skipping");
            // SAFETY: nothing was created past the device/instance.
            unsafe {
                device.destroy_device(None);
                instance.destroy_instance(None);
            }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let pixel_count = (width * height) as usize;
        let original = vec![120u8; pixel_count * 4];
        let model_answer = vec![90u8; pixel_count * 4];
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");

        // Real code always chains each dispatch's semaphore into that same frame's
        // present call before the next dispatch could ever reuse (and re-signal) it --
        // reproduced here by waiting on each semaphore immediately, every iteration,
        // same real discipline `device.rs` follows, just without a real swapchain.
        for i in 0..(ASYNC_SLOTS * 3 + 1) {
            let target = make_target_image(&device, &mem_props, width, height);
            transition_to_present_src(&device, queue, pool, target.image);
            let sem = gpu.dispatch_into_image_async(&device, &instance, physical_device, queue, width, height, &original, &model_answer, 1.0, 1.0, 2.0, false, target.image);
            let Some(sem) = sem else { panic!("dispatch_into_image_async returned None on iteration {i}") };
            let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
            let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
            let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
            unsafe {
                device.queue_submit(queue, &[submit], wait_fence).unwrap();
                device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                device.destroy_fence(wait_fence, None);
                target.destroy(&device);
            }
        }

        // SAFETY: every iteration above already waited out its own GPU work.
        unsafe {
            device.destroy_command_pool(pool, None);
            gpu.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
