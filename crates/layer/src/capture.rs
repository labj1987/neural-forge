//! Milestone 4 phase A/B: capture the image `queue_present_khr` is about to present
//! into the shared-memory proxy region, run the round trip, and copy a result back
//! before the real present call.
//!
//! No `VK_EXT_external_memory_host` import yet -- every byte crosses an explicit CPU
//! `memcpy` between a host-visible/host-coherent staging buffer and the mapping
//! `ShmClient` owns. That is exactly the "staging copy" fallback
//! `crates/helper/src/shm.rs`'s own doc comment already describes as always-correct,
//! just not zero-copy; importing the mapping directly as device memory is a later
//! optimization on top of this, not a prerequisite for it working.
//!
//! Stage 1 (capture into a staging buffer) is one command buffer + one fence,
//! synchronous -- the CPU needs those bytes before it can even start the SHM round
//! trip, so there's no way around blocking on it. What happens after the round trip
//! is the original synchronous stage-2 write-back below (a synchronous GPU or CPU
//! compose into the staging bytes, then one more command buffer + fence wait).
//! The per-frame path (`run`) composes asynchronously through
//! `composition::gpu::GpuCompose::present_temporal_delta_async` instead.

use ash::vk;

use crate::shm::ShmClient;

pub struct CaptureResources {
    queue_family: u32,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    capacity: vk::DeviceSize,
}

// SAFETY: every field is either a plain Vulkan handle (as `Send`-safe as `ash::Device`
// itself already assumes) or `ptr`, a `vkMapMemory` pointer into memory this struct
// owns exclusively -- never aliased outside the `Mutex<State>` this always lives behind
// in `NeuralForgeDeviceInfo`.
unsafe impl Send for CaptureResources {}

impl CaptureResources {
    /// # Safety
    /// Must not be called while any submitted work referencing these handles might
    /// still be in flight -- callers only ever call this right after a successful
    /// `vkWaitForFences` on `self.fence`, or at device-destruction time.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// One command pool/buffer + fence + host-coherent staging buffer -- the resource
/// bundle both [`CaptureResources`] (a single one) and [`CapturePipeline`] (two, see
/// its own doc comment) are built from. Pulled out so there is exactly one place that
/// builds/unwinds this specific allocation sequence, not two copies that could drift.
struct CaptureBuffer {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    capacity: vk::DeviceSize,
}

// SAFETY: same reasoning as `CaptureResources`'s own impl below -- plain Vulkan
// handles plus a `vkMapMemory` pointer into memory this struct owns exclusively.
unsafe impl Send for CaptureBuffer {}

impl CaptureBuffer {
    /// # Safety
    /// Must not be called while any submitted work referencing these handles might
    /// still be in flight -- see every caller's own safety comment for how each
    /// upholds that.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// `working_scale`'s capture-side machinery: a small GPU image plus its own small
/// CPU-readable buffer, built only when the model's resolution differs from the
/// capture's own. [`submit_pipeline_capture`] blits the full-resolution `capture_image`
/// down into `image` -- a hardware resize unit (`vkCmdBlitImage`, `VK_FILTER_LINEAR`),
/// sub-millisecond, unlike the CPU resample this replaces (measured 315-546ms at GTA's
/// resolution on 2026-09-17, unusable on the present thread; see `docs/GHOSTING_PLAN.md`
/// step 1). `image` then copies into `buffer`, which is what actually crosses SHM as
/// the proxy -- smaller proxy, smaller `DLSSNR.Width`/`Height` at `CreateFeature`,
/// faster model evaluation, the whole point of `working_scale`.
///
/// Deliberately only `VK_FILTER_LINEAR`: a hardware blit has no concept of the
/// Lanczos/Catmull-Rom/Mitchell-Netravali/Kaiser kernels `scaling_downscaler` selects
/// (that field stays meaningful only for a hypothetical future compute-shader
/// implementation, not this one) -- a real, honest quality/speed tradeoff, not an
/// oversight.
///
/// Never touches [`CaptureBuffer`]'s own full-resolution buffer/copy at all: the
/// full-resolution bytes this slot always still produces are what `run` swaps into
/// `inflight[slot].original`, the compositor's motion-mask reference, which must stay
/// full-resolution (see `run`'s own doc comment on `raw_answer_base`).
struct ModelScratch {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    ptr: *mut u8,
    width: u32,
    height: u32,
    capacity: vk::DeviceSize,
    /// The encode's storage-image view and its descriptor set, both `None` when the
    /// encode isn't available on this device/format (see
    /// [`crate::composition::encode_pass::supports_storage`]) -- in which case the
    /// proxy crosses to the helper unencoded, exactly as it did before the encode
    /// existed, and the resolve is told so.
    ///
    /// The view is always `R8G8B8A8_UNORM` even when `image` is `B8G8R8A8_UNORM`: the
    /// image carries `MUTABLE_FORMAT` and the two are format-compatible (same 32-bit
    /// class), which makes `encode.comp`'s `rgba8` layout qualifier correct against
    /// the view while `bgr_order` carries what the channels actually mean.
    encode_view: Option<vk::ImageView>,
    encode_set: Option<vk::DescriptorSet>,
}

// SAFETY: same reasoning as `CaptureBuffer`'s own impl -- plain Vulkan handles plus a
// `vkMapMemory` pointer into memory this struct owns exclusively.
unsafe impl Send for ModelScratch {}

impl ModelScratch {
    /// # Safety
    /// Same contract as [`CaptureBuffer::destroy`]: no submitted work referencing
    /// `image`/`buffer` may still be in flight.
    unsafe fn destroy(&self, device: &ash::Device, encode: Option<&crate::composition::encode_pass::EncodePass>) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            // The set first: freeing it while the view it points at still exists is
            // the only valid order, and both must outlive any submission using them
            // (this function's own contract).
            if let (Some(set), Some(pass)) = (self.encode_set, encode) {
                pass.free_set(device, set);
            }
            if let Some(view) = self.encode_view {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(self.image, None);
            device.free_memory(self.image_memory, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.buffer_memory, None);
        }
    }
}

fn build_model_scratch(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
    encode: Option<&crate::composition::encode_pass::EncodePass>,
) -> Option<ModelScratch> {
    // The encode needs this image to be dispatchable over in place, which needs
    // `STORAGE` usage -- not guaranteed for `B8G8R8A8_UNORM`, so it is asked rather
    // than assumed. When it isn't available the image is built exactly as it was
    // before the encode existed and the proxy crosses unencoded; nothing fails.
    //
    // `MUTABLE_FORMAT` is what lets the view below be `R8G8B8A8_UNORM` over a BGRA
    // image (same 32-bit compatibility class), so `encode.comp`'s `rgba8` qualifier is
    // correct against the view rather than relying on a driver tolerating a mismatch.
    let want_encode = encode.is_some() && crate::composition::encode_pass::supports_storage(instance, physical_device, format);
    // Logged once per process, because "is the encode actually running" cannot be read
    // off the self-check below: on SDR content the encode's own effect is small, so an
    // unencoded proxy and a correctly encoded one produce similar deltas against the
    // reference. This line is the unambiguous signal.
    {
        static ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ANNOUNCED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            if want_encode {
                crate::log!("[encode] active: dispatching over the {width}x{height} scratch ({format:?}), composition uses mode 2");
            } else if encode.is_none() {
                crate::log!("[encode] unavailable: the encode pipeline itself failed to build, composition stays on mode 1");
            } else {
                crate::log!("[encode] unavailable: {format:?} has no STORAGE_IMAGE support on this device, composition stays on mode 1");
            }
            crate::logging::flush();
        }
    }
    let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    let mut create_flags = vk::ImageCreateFlags::empty();
    if want_encode {
        usage |= vk::ImageUsageFlags::STORAGE;
        create_flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    let image_info = vk::ImageCreateInfo::builder()
        .flags(create_flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: `image_info` is a valid `VkImageCreateInfo`.
    let Ok(image) = (unsafe { device.create_image(&image_info, None) }) else { return None };
    // SAFETY: `image` was just created, no memory bound yet.
    let image_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let image_type = (0..mem_props.memory_type_count)
        .find(|&i| image_reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .or_else(|| (0..mem_props.memory_type_count).find(|&i| image_reqs.memory_type_bits & (1 << i) != 0));
    let Some(image_type) = image_type else {
        // SAFETY: `image` has no memory bound; nothing else references it.
        unsafe { device.destroy_image(image, None) };
        return None;
    };
    let image_alloc = vk::MemoryAllocateInfo::builder().allocation_size(image_reqs.size).memory_type_index(image_type);
    // SAFETY: `image_alloc` is valid; `image_type` satisfies `image_reqs`.
    let Ok(image_memory) = (unsafe { device.allocate_memory(&image_alloc, None) }) else {
        // SAFETY: same reasoning as above.
        unsafe { device.destroy_image(image, None) };
        return None;
    };
    // SAFETY: `image`/`image_memory` were each just created, sized/typed for each other.
    if unsafe { device.bind_image_memory(image, image_memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(image_memory, None);
            device.destroy_image(image, None);
        }
        return None;
    }

    let bytes = vk::DeviceSize::from(width) * vk::DeviceSize::from(height) * 4;
    let buf_info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `buf_info` is valid.
    let Ok(buffer) = (unsafe { device.create_buffer(&buf_info, None) }) else {
        // SAFETY: `image`/`image_memory` are bound to each other, own nothing else yet.
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
        }
        return None;
    };
    // SAFETY: `buffer` was just created, no memory bound yet.
    let buf_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(buf_type) = pick_readback_memory_type(buf_reqs, &mem_props) else {
        // SAFETY: `buffer` has no memory bound; `image`/`image_memory` own nothing else.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
        }
        return None;
    };
    let buf_alloc = vk::MemoryAllocateInfo::builder().allocation_size(buf_reqs.size).memory_type_index(buf_type);
    // SAFETY: `buf_alloc` is valid; `buf_type` satisfies `buf_reqs`.
    let Ok(buffer_memory) = (unsafe { device.allocate_memory(&buf_alloc, None) }) else {
        // SAFETY: same reasoning as above.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
        }
        return None;
    };
    // SAFETY: `buffer`/`buffer_memory` were each just created, sized/typed for each other.
    if unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(buffer_memory, None);
            device.destroy_buffer(buffer, None);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
        }
        return None;
    }
    // SAFETY: `buffer_memory` is `HOST_VISIBLE` (`pick_readback_memory_type`'s own
    // contract); mapping the whole allocation is always in bounds.
    let Ok(ptr) = (unsafe { device.map_memory(buffer_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }) else {
        // SAFETY: same reasoning as the bind-failure branch above.
        unsafe {
            device.free_memory(buffer_memory, None);
            device.destroy_buffer(buffer, None);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
        }
        return None;
    };

    // The encode's view and descriptor set, last so every earlier failure path stays
    // exactly as it was. Both are optional: if either step fails the scratch is still
    // perfectly usable for the blit/download it existed for before the encode, so the
    // proxy simply crosses unencoded rather than the whole capture failing.
    let (encode_view, encode_set) = if want_encode {
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        // SAFETY: `image` is bound and was created with `MUTABLE_FORMAT` plus
        // `STORAGE` usage (both gated on `want_encode`); `R8G8B8A8_UNORM` is in the
        // same format-compatibility class as the image's own format.
        match unsafe { device.create_image_view(&view_info, None) } {
            Ok(view) => match encode.and_then(|e| e.allocate_set(device, view)) {
                Some(set) => (Some(view), Some(set)),
                None => {
                    // SAFETY: nothing was submitted against `view`; no set references it.
                    unsafe { device.destroy_image_view(view, None) };
                    (None, None)
                }
            },
            Err(_) => (None, None),
        }
    } else {
        (None, None)
    };

    Some(ModelScratch {
        image,
        image_memory,
        buffer,
        buffer_memory,
        ptr: ptr.cast(),
        width,
        height,
        capacity: buf_reqs.size,
        encode_view,
        encode_set,
    })
}

/// Picks the best `HOST_VISIBLE` memory type for a buffer the GPU writes and the CPU
/// then reads back on the present thread -- the exact readback-cost fix from
/// `capture_hot_path_cost_per_present` (87ms -> 5.7ms/present at 1440p): prefer
/// `HOST_CACHED` system memory that is *not* `DEVICE_LOCAL` (not NVIDIA's small PCIe
/// BAR, whose CPU reads are uncached and brutally slow), falling back to any
/// `HOST_VISIBLE|HOST_COHERENT` type only if no cached system-memory type exists.
/// Pulled out of [`build_capture_buffer`] so [`build_model_scratch`]'s own small
/// readback buffer (`working_scale`'s scaled proxy) gets the identical fix rather than
/// a copy that could quietly regress on its own.
fn pick_readback_memory_type(reqs: vk::MemoryRequirements, mem_props: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
    let cached = vk::MemoryPropertyFlags::HOST_CACHED;
    let coherent = vk::MemoryPropertyFlags::HOST_COHERENT;
    let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let usable = |i: u32| reqs.memory_type_bits & (1 << i) != 0;
    let flags = |i: u32| mem_props.memory_types[i as usize].property_flags;
    let pick = |pred: &dyn Fn(vk::MemoryPropertyFlags) -> bool| (0..mem_props.memory_type_count).find(|&i| usable(i) && pred(flags(i)));
    pick(&|f| f.contains(host_visible | cached | coherent) && !f.contains(device_local)).or_else(|| pick(&|f| f.contains(host_visible | coherent)))
}

fn build_capture_buffer(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family: u32, bytes: vk::DeviceSize) -> Option<CaptureBuffer> {
    let pool_info =
        vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: `device` is the live device this capture serves; `pool_info` is valid.
    let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else { return None };

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `pool` was just created above.
    let cmd = match unsafe { crate::loader_data::allocate_commands(device, &alloc_info) } {
        Ok(bufs) => bufs[0],
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet.
            unsafe { device.destroy_command_pool(pool, None) };
            return None;
        }
    };

    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    // SAFETY: starting signaled means the first use's own wait/poll never blocks (or
    // reports pending) on a fence nothing has submitted work against yet.
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(f) => f,
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet; freeing it also frees `cmd`.
            unsafe { device.destroy_command_pool(pool, None) };
            return None;
        }
    };

    let buf_info = vk::BufferCreateInfo::builder()
        .size(bytes)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `buf_info` is valid.
    let buffer = match unsafe { device.create_buffer(&buf_info, None) } {
        Ok(b) => b,
        Err(_) => {
            // SAFETY: neither `fence` nor `pool` owns `buffer` (it doesn't exist).
            unsafe {
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return None;
        }
    };
    // SAFETY: `buffer` was just created and is not yet bound to memory.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: `physical_device` is the device this capture serves; `instance` is its
    // owning instance (stored once at `vkCreateInstance`, see `crate::CURRENT_INSTANCE`).
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    // This buffer is written by the GPU and then *read back by the CPU* on the game's
    // own present thread every capture -- see `pick_readback_memory_type`'s own doc
    // comment for why the memory type chosen for it dominates that cost (the
    // 87ms->5.7ms/present fix). The Vulkan spec guarantees at least one
    // HOST_VISIBLE|HOST_COHERENT type, so that helper returning `None` would mean a
    // spec-non-compliant driver, not a real device limitation -- still handled as a
    // plain "skip capture" rather than assumed away.
    let Some(type_index) = pick_readback_memory_type(reqs, &mem_props) else {
        // SAFETY: `buffer` has no memory bound yet; nothing else owns `fence`/`pool`.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return None;
    };

    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` was just confirmed to satisfy `reqs`.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: same reasoning as the branch above.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return None;
        }
    };
    // SAFETY: `buffer`/`memory` were each just created above, sized/typed to satisfy
    // each other by construction.
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        // SAFETY: `memory` is not yet bound to anything that would make freeing it
        // unsound; `buffer` has no memory bound.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return None;
    }
    // SAFETY: `memory` is `HOST_VISIBLE` by the type selection above; mapping the
    // whole allocation is always in bounds.
    let ptr = match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
        Ok(p) => p.cast::<u8>(),
        Err(_) => {
            // SAFETY: same reasoning as the bind-failure branch above.
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                device.destroy_fence(fence, None);
                device.destroy_command_pool(pool, None);
            }
            return None;
        }
    };

    Some(CaptureBuffer { pool, cmd, fence, buffer, memory, ptr, capacity: reqs.size })
}

/// Builds (or rebuilds, if the queue family changed or `bytes` grew past what's
/// already allocated) the resources capture needs. `existing` is left `None` on any
/// failure -- every caller treats that as "skip capture this frame, present
/// unmodified", never a reason to stop trying on a later frame.
fn ensure(
    existing: &mut Option<CaptureResources>,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: vk::DeviceSize,
) -> bool {
    if let Some(r) = existing {
        if r.queue_family == queue_family && r.capacity >= bytes {
            return true;
        }
        // SAFETY: called between frames, never while `r.fence` might still be
        // unsignaled from an in-flight submission -- `queue_present_khr` only reaches
        // here after the previous frame's own capture fully completed.
        unsafe { r.destroy(device) };
        *existing = None;
    }
    let Some(b) = build_capture_buffer(device, instance, physical_device, queue_family, bytes) else { return false };
    *existing = Some(CaptureResources { queue_family, pool: b.pool, cmd: b.cmd, fence: b.fence, buffer: b.buffer, memory: b.memory, ptr: b.ptr, capacity: b.capacity });
    true
}

fn subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build()
}

fn barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource())
        .src_access_mask(src)
        .dst_access_mask(dst)
        .build()
}

/// What [`run`] is carrying forward from the round trip it most recently *sent* **on
/// one wire slot**, across as many present calls as the helper takes to answer it.
/// `original` holds the exact pixels captured at send time -- needed again once the
/// answer finally arrives, since composition combines the two -- alongside the
/// dimensions/format that capture was taken at, so a resolution change mid-flight is
/// detected (and the stale pair discarded) rather than composited against a
/// mismatched frame size.
///
/// Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) gives the wire two independent slots, so
/// `run` carries `[Inflight; 2]`, one per slot -- each slot's own in-flight capture is
/// completely independent of the other's. What *isn't* per-slot (moved out to `run`'s
/// own parameters instead, alongside `last_answer`): the single currently-presented
/// answer and the one-shot bootstrap flag, both of which are properties of the
/// process's presentation state as a whole, not of either slot specifically -- see
/// `run`'s own doc comment for why "whichever slot answers most recently wins" is the
/// same kind of bounded staleness tradeoff this module already accepts.
#[derive(Default)]
pub struct Inflight {
    original: Vec<u8>,
    /// Synchronous presents seen on this device (slot 0's counter), for the model interval.
    presents: u64,
    dims: Option<(u32, u32, u32)>,
    /// The proxy's own `(width, height)` at the moment this slot's outstanding request
    /// was actually sent -- `working_scale`'s answer comes back at whatever resolution
    /// the request was sent at, which is no longer always `dims`' own `(width,
    /// height)` once scaling is active. `None` means "sent at the swapchain's own
    /// resolution", the only possibility before `working_scale` existed -- callers
    /// resizing the answer buffer fall back to `dims` in that case, unchanged old
    /// behavior. Kept as a wholly separate field from `dims` rather than folding into
    /// it: `dims` also drives the swapchain-resize-detection comparison in `run`,
    /// which must keep comparing against the swapchain's own resolution regardless of
    /// what the proxy itself was scaled to.
    proxy_dims: Option<(u32, u32)>,
}

/// `working_scale × (width, height)`, rounded to the nearest even number (several
/// paths in this crate and the helper implicitly assume even dimensions are safe, not
/// a hazard) and floored at 64px per axis -- upstream (DLSS5VKLayer) hit and fixed a
/// real GPU hang from a smaller probe swapchain (`0.2.6-3`'s changelog), and this
/// project has no reason to retest a smaller floor itself. Returns `(width, height)`
/// unchanged whenever `scale` is not a real, useful value (non-finite, non-positive,
/// or close enough to `1.0` that scaling would buy nothing) -- callers comparing the
/// result against `(width, height)` for equality get exactly the "skip the entire
/// `working_scale` code path" behavior every caller had before it existed.
/// The most pixels the model is ever asked to work on: 4K-equivalent. Above this the model
/// either cannot be built at all (VRAM with a game already resident, or the model's own limits --
/// a 5760x3240 request failed to build on a 12 GB card) or is so slow that every answer is stale
/// on arrival. A game rendering above it is simply scaled down for the model, and the answer is
/// blitted back up, exactly as `working_scale` below 1.0 already is -- so any game resolution
/// works. `NEURAL_FORGE_MAX_MODEL_PIXELS` overrides it.
const DEFAULT_MAX_MODEL_PIXELS: u64 = 3840 * 2160;

fn max_model_pixels() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        neural_forge_protocol::env::var("NEURAL_FORGE_MAX_MODEL_PIXELS")
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v >= 64 * 64)
            .unwrap_or(DEFAULT_MAX_MODEL_PIXELS)
    })
}

/// The scale the model raster is actually built at: the requested `working_scale`, never above
/// 1.0 (the compose path rejects an answer larger than the frame, so supersampling is not
/// available), and never so large that the model input exceeds [`max_model_pixels`].
fn effective_working_scale(width: u32, height: u32, requested: f32) -> f32 {
    let mut scale = if requested.is_finite() && requested > 0.0 { requested.min(1.0) } else { 1.0 };
    let frame_pixels = (u64::from(width) * u64::from(height)).max(1) as f64;
    let cap = max_model_pixels() as f64;
    if frame_pixels * f64::from(scale) * f64::from(scale) > cap {
        scale = (cap / frame_pixels).sqrt() as f32;
    }
    scale
}

fn scaled_dims(width: u32, height: u32, scale: f32) -> (u32, u32) {
    if !scale.is_finite() || scale <= 0.0 || (scale - 1.0).abs() < 0.01 {
        // Full size, but never odd: the helper (and the model) only take even dimensions, and an
        // odd-width window (measured: a 2493x1408 window) was rejected outright, silently
        // presenting every frame without the effect. Dropping one row or column hands the model
        // a raster one pixel smaller, and the answer is scaled back to the exact frame size.
        return (width & !1, height & !1);
    }
    let axis = |v: u32| {
        let scaled = ((v as f32 * scale).round()).max(64.0) as u32;
        (scaled / 2) * 2 // force even
    };
    (axis(width), axis(height))
}

/// The `vk::Format` [`build_model_scratch`] should create its scratch image as, for a
/// given `proxy_format`/`bgr_order` pair -- `None` when `working_scale`'s GPU-blit
/// mechanism does not (yet) support this proxy format at all. Restricted to the two
/// 8-bit formats: an HDR float16 proxy's blit-filtering and byte-layout behavior
/// differ enough from the 8-bit case that this first cut deliberately does not attempt
/// it (matches the same `is_8bit` gate several other advanced paths in this crate and
/// the helper already use).
fn model_scratch_format(proxy_format: u32, bgr_order: bool) -> Option<vk::Format> {
    if !neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format) {
        return None;
    }
    Some(if bgr_order { vk::Format::B8G8R8A8_UNORM } else { vk::Format::R8G8B8A8_UNORM })
}

/// Real per-frame NR compute (a helper round trip through a Wine-hosted process, plus
/// whatever GPU work either side does) does not run at anywhere close to swapchain
/// present rate -- measured on real hardware (`lordnikon`, 2026-09-10, see
/// `CLAUDE.md`) at roughly 100-150ms end to end even once every other bottleneck
/// found that same session was fixed. [`run_sync`] (this crate's entire capture path
/// before this) called that round trip, and blocked waiting for it, from *inside*
/// every single present call -- meaning the game's own presentation rate could never
/// exceed the round trip's, even though the actual GPU compute involved is only a
/// few milliseconds. That coupling, not any single slow operation, was the real
/// cause of a reported ~2.8 fps at 4K with NR on, confirmed by removing this
/// project's layer entirely and watching the same game return to 99% GPU utilization
/// and a normal framerate.
///
/// This function decouples the two: it captures and sends a new frame only when no
/// round trip is currently in flight, checks on any in-flight one *without blocking*
/// (see [`ShmClient::poll_async_request`]), and applies whatever answer arrives to
/// whichever frame happens to be current at that moment -- not necessarily the one
/// that was captured alongside it. Every other frame (which, once the pipeline is
/// running, is most of them) touches `image` not at all and returns `None`
/// immediately, at effectively zero cost. The tradeoff this accepts, deliberately,
/// per Alex's own explicit authorization ("do it if it gives us the most frames when
/// NR is on"): the visible NR enhancement updates at whatever rate the round trip
/// actually achieves, not every frame, and is very occasionally composited against a
/// slightly newer frame than the one it was computed from (a few frames of temporal
/// staleness at most, bounded by the round trip's own duration) -- a real quality
/// tradeoff, not a free lunch, but one that keeps the game's own rendering and
/// presentation running at its true native rate instead of being held hostage by a
/// cross-process IPC round trip on every single frame.
///
/// Ordering inside a single call matters and is deliberate: capturing a new frame
/// (when due) always happens *before* compositing an answer that arrived this same
/// frame, because compositing overwrites `image` -- capturing after that would
/// capture this function's own composited output instead of the game's real
/// rendering, feeding a corrupted "original" into the next cycle.
///
/// Polls whatever capture is already in flight -- [`DirectCapture`] when `use_direct`,
/// otherwise [`CapturePipeline`] -- and, if one just completed, gets its bytes into
/// `original_scratch` and the SHM proxy region, then returns `true`. If nothing
/// completed this call, tries to submit a new capture into whichever strategy is
/// active instead (`false` either way). Shared by both of `run`'s capture call sites
/// (the disabled-bootstrap one-shot and the main enabled path below) -- they differ
/// only in what they do with a successful result afterward (`inflight` bookkeeping),
/// not in how a capture gets started or consumed.
///
/// For [`CapturePipeline`], "gets its bytes into `original_scratch` and the SHM proxy
/// region" means what it always has: copy staging memory into `original_scratch`, then
/// [`ShmClient::write_proxy`] copies that into the proxy region. For [`DirectCapture`],
/// the GPU write already landed the bytes in the proxy region directly -- no
/// `write_proxy` call needed, that copy is exactly what importing the region as device
/// memory removes -- but `original_scratch` still needs its own stable copy (read back
/// out of the now-written proxy region), because `inflight.original` (and the frame
/// hold/white-point meter) need bytes that survive whatever capture starts next and
/// overwrites that region, which the live proxy region itself can't provide once it's
/// shared, imported memory.
///
/// `model`, when `Some((model_width, model_height, format))`, additionally requests a
/// scaled proxy at that resolution (`working_scale`) -- see [`ModelScratch`]'s own doc
/// comment for the mechanism. Only the [`CapturePipeline`] (non-`use_direct`) branch
/// implements it; `use_direct` (the `VK_EXT_external_memory_host` zero-copy path)
/// ignores `model` and always sends the full-resolution proxy, same as before
/// `working_scale` existed -- that path is unverified to even be live on real hardware
/// (alignment checks fail it back to `CapturePipeline` on every device tested so far,
/// see `docs/HARDWARE_VALIDATION.md`), so it is not worth the same surgery until it is.
/// `original_scratch` is always the full-resolution capture regardless of `model` --
/// callers still swap it into `inflight[slot].original` unchanged; `model_scratch`
/// receives the scaled bytes only when `model` was requested and actually available
/// this poll (a rebuild-in-progress or first-ever call can still miss a poll, in which
/// case this falls back to sending the full-resolution proxy that frame, same
/// bounded-staleness fail-open discipline as everywhere else in this module).
///
/// Returns [`CaptureStep::Captured`] on a successful capture+send this call,
/// [`CaptureStep::Pending`] while a capture is in flight or was just submitted, and
/// [`CaptureStep::Failed`] when the capture resources could not be built or a submission
/// failed -- a caller waiting for this frame's capture must stop waiting on `Failed`, since
/// nothing is in flight that could ever complete. `Captured::sent` is the proxy's *actual*
/// resolution, which the caller must record (`Inflight::proxy_dims`) to size the eventual
/// answer correctly. This is `model`'s own request dims only when a scaled send genuinely
/// happened; every fallback above (an unavailable/failed scratch, `use_direct`, no `model`
/// requested at all) correctly reports the full-resolution `(width, height)` instead,
/// because it actually sent that -- callers must not re-derive this from `model`
/// themselves, only ever trust this return value.
///
/// A completed capture is only sent when it was taken at this frame's own `(width, height,
/// proxy_format)` and, when `submitted_after` is `Some`, submitted no earlier than that
/// instant (the synchronous present passes its own start: a capture a previous present
/// left in flight is of an older frame). Anything else is dropped and a fresh capture is
/// submitted in its place.
///
/// A direct-capture setup failure clears `external_memory_host`, so the rest of this
/// device's life uses the always-correct [`CapturePipeline`] path: there is no per-call
/// fallback, and retrying the import every present would only fail the same way.
///
/// `zc_target`, when `Some` (the synchronous present's zero-copy compose, `use_direct`
/// only), is the device-local capture target a newly submitted direct capture also copies
/// the frame into. A completed capture that did so leaves `original_scratch` empty and
/// reports `gpu_original`: the frame's stable copy is on the GPU, and the per-frame CPU
/// copy out of the proxy region is skipped. Every other completion copies as before.
#[allow(clippy::too_many_arguments)]
fn poll_or_submit_capture(
    slot: usize,
    use_direct: bool,
    external_memory_host: &mut bool,
    pipeline: &mut Option<CapturePipeline>,
    direct: &mut [Option<DirectCapture>; 2],
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    capture_image: vk::Image,
    capture_layout: vk::ImageLayout,
    width: u32,
    height: u32,
    proxy_format: u32,
    frame_bytes: u64,
    model: Option<(u32, u32, vk::Format)>,
    encode_push: crate::composition::encode_pass::EncodePush,
    shm: &mut ShmClient,
    original_scratch: &mut Vec<u8>,
    model_scratch: &mut Vec<u8>,
    zc_target: Option<vk::Buffer>,
    submitted_after: Option<std::time::Instant>,
) -> CaptureStep {
    let usable = |w: u32, h: u32, f: u32, submitted: std::time::Instant| {
        (w, h, f) == (width, height, proxy_format) && submitted_after.is_none_or(|after| submitted >= after)
    };
    if use_direct {
        let Some((host_ptr, capacity)) = shm.proxy_region(slot) else { return CaptureStep::Failed };
        // SAFETY: `host_ptr`/`capacity` describe `shm`'s own live proxy region for
        // this slot, valid for as long as `shm` stays open (the life of this process,
        // since the mapping is never unmapped -- see
        // `neural_forge_protocol::mapping::Mapping::header`'s own doc comment on the
        // equivalent GUI/CLI mapping); nothing else writes to it except through
        // `ShmClient::write_proxy`, which this branch never calls, and slot 0's/slot
        // 1's regions are disjoint (`docs/PROTOCOL_V3_DESIGN.md`), so the other slot's own
        // `DirectCapture` never touches these same bytes. Only the frame's own bytes are
        // imported, rounded up to the driver's import alignment (as the answer import is),
        // not the whole region: every imported page may be pinned. `run` only chooses
        // `use_direct` when that alignment query succeeded and the region is aligned to it.
        let import_bytes = min_imported_host_pointer_alignment(instance, physical_device)
            .map_or(capacity as u64, |alignment| frame_bytes.div_ceil(alignment) * alignment)
            .min(capacity as u64);
        if !unsafe {
            ensure_direct_capture(&mut direct[slot], device, instance, physical_device, queue_family, host_ptr, import_bytes)
        } {
            note_setup_failure();
            *external_memory_host = false;
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::log!("[layer] zero-copy capture setup failed; using the staging-buffer capture for the rest of this device's life");
                crate::logging::flush();
            }
            return CaptureStep::Failed;
        }
        let d = direct[slot].as_mut().expect("just ensured above");
        if let Some((w, h, f, target, submitted)) = poll_direct_capture(d, device) {
            if usable(w, h, f, submitted) {
                if target.is_some() && target == zc_target {
                    // The same submission wrote the frame into the capture target the compose
                    // reads, so there is nothing to copy out here.
                    original_scratch.clear();
                    shm.set_frame_info(slot, width, height, proxy_format);
                    return CaptureStep::Captured(Captured { sent: (width, height), gpu_original: true, copy_out: std::time::Duration::ZERO });
                }
                let t_copy = std::time::Instant::now();
                let n = capacity.min(frame_bytes as usize);
                original_scratch.clear();
                // SAFETY: `host_ptr` is `shm`'s own live proxy region, valid for at least
                // `capacity` bytes; `poll_direct_capture` returning `Some` just confirmed
                // this slot's fence signaled, making the GPU's writes to it visible to the
                // CPU (host-coherent memory backs every capture buffer in this module,
                // imported or not).
                original_scratch.extend_from_slice(unsafe { std::slice::from_raw_parts(host_ptr, n) });
                shm.set_frame_info(slot, width, height, proxy_format);
                return CaptureStep::Captured(Captured { sent: (width, height), gpu_original: false, copy_out: t_copy.elapsed() });
            }
            // A capture of another frame size, or of an earlier frame: dropped, and a fresh
            // one submitted below in its place.
        }
        if submit_direct_capture(d, device, queue, capture_image, capture_layout, width, height, proxy_format, zc_target) || d.pending.is_some() {
            CaptureStep::Pending
        } else {
            CaptureStep::Failed
        }
    } else {
        if !ensure_pipeline(pipeline, device, instance, physical_device, queue_family, frame_bytes) {
            note_setup_failure();
            return CaptureStep::Failed;
        }
        let p = pipeline.as_mut().expect("just ensured above");
        let t_copy = std::time::Instant::now();
        let (full, model_dims) = poll_pipeline_capture(p, slot, device, original_scratch, model_scratch);
        if full.is_some_and(|(w, h, f, submitted)| usable(w, h, f, submitted)) {
            if let Some((mw, mh)) = model_dims {
                // Encode self-check, once per process: the untouched frame and the
                // GPU-encoded proxy are both sitting in CPU memory right here, so the
                // GPU's answer can be compared against the arithmetic it was supposed
                // to perform -- real-hardware verification of the encode that needs
                // nobody to look at a screen. Only possible when the proxy is
                // pixel-aligned with the frame, which means a working scale of exactly
                // 1.0; at any other scale it logs nothing rather than comparing
                // different rasters. See `composition::encode::compare_to_reference`
                // for what a given delta does and does not prove.
                static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if (mw, mh) == (width, height) && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    match crate::composition::encode::compare_to_reference(
                        original_scratch,
                        model_scratch,
                        encode_push.bgr_order != 0,
                        encode_push.white_point,
                        encode_push.reversible_mode,
                    ) {
                        Some((max_delta, mean_delta, pixels)) => crate::log!(
                            "[encode] self-check vs CPU reference: max_delta={max_delta} mean_delta={mean_delta:.4} over {pixels} px                              (white_point={} reversible_mode={} bgr_order={}) -- 0-2 is rounding, large means the GPU path is wrong",
                            encode_push.white_point,
                            encode_push.reversible_mode,
                            encode_push.bgr_order,
                        ),
                        None => crate::log!("[encode] self-check skipped: proxy and frame are different rasters"),
                    }
                    crate::logging::flush();
                }
                shm.set_frame_info(slot, mw, mh, proxy_format);
                shm.write_proxy(slot, model_scratch);
                return CaptureStep::Captured(Captured { sent: (mw, mh), gpu_original: false, copy_out: t_copy.elapsed() });
            }
            shm.set_frame_info(slot, width, height, proxy_format);
            shm.write_proxy(slot, original_scratch);
            return CaptureStep::Captured(Captured { sent: (width, height), gpu_original: false, copy_out: t_copy.elapsed() });
        }
        // Nothing completed, or what completed was of another frame size or an earlier frame
        // (dropped: its slot is free again and a fresh capture goes in below).
        let submitted = submit_pipeline_capture(
            p,
            slot,
            device,
            instance,
            physical_device,
            queue,
            capture_image,
            capture_layout,
            width,
            height,
            proxy_format,
            model,
            encode_push,
        );
        if submitted || p.slots[slot].pending.is_some() {
            CaptureStep::Pending
        } else {
            CaptureStep::Failed
        }
    }
}

/// What one [`poll_or_submit_capture`] call achieved.
enum CaptureStep {
    /// A capture completed and was sent this call.
    Captured(Captured),
    /// A capture is in flight (possibly submitted by this very call).
    Pending,
    /// Setup or submission failed: nothing is in flight that could complete.
    Failed,
}

/// A capture [`poll_or_submit_capture`] completed and sent this call.
struct Captured {
    /// The proxy's actual resolution (see [`poll_or_submit_capture`]).
    sent: (u32, u32),
    /// The frame's stable copy is in the zero-copy capture target, not `original_scratch`.
    gpu_original: bool,
    /// How long getting the frame's bytes out onto the CPU took (zero when `gpu_original`).
    copy_out: std::time::Duration,
}

/// The synchronous present's stage timings for the frame being composed, for the `[sync]` log.
#[derive(Clone, Copy)]
struct SyncTiming {
    /// Submitting the capture until its completion was observed (the capture's GPU time plus
    /// the polling granularity), without the CPU copy-out.
    capture_wait: std::time::Duration,
    /// The CPU copy of the captured frame (0 on a zero-copy frame).
    copy_out: std::time::Duration,
    /// The white meter.
    meter: std::time::Duration,
    /// Sending the request until the answer was seen.
    wait_answer: std::time::Duration,
    /// What the helper published as its own upload + evaluate + readback for that answer, so
    /// `wait_answer - helper` is the handoff overhead.
    helper: std::time::Duration,
    zero_copy: bool,
}

thread_local! {
    /// The synchronous present's stage timings for the frame being composed, consumed by the
    /// timing log after the compose.
    static SYNC_TIMING: std::cell::Cell<Option<SyncTiming>> = const { std::cell::Cell::new(None) };
}

/// The frame's white level, for the proxy encode's divisor (upstream's meter: `dlssnr.hlsl`
/// mode 4 plus `meter_reduce.comp`). The peak linear luminance of each tile of a 64x64 grid,
/// then the 90th percentile across tiles: not the frame's mean, which is scene brightness and would
/// hand a dark scene a divisor that blows it out, and not its maximum, which one specular hit
/// decides. Refused (`None`) unless enough of the frame is lit (20% of tiles above a tenth of
/// the brightest) to say where white is. Sampled 8x8 per tile: this runs on the CPU over the
/// captured frame, every few frames.
fn meter_white(frame: &[u8], width: u32, height: u32, bgr: bool) -> Option<f32> {
    const GRID: u32 = 64;
    const SAMPLES: u32 = 8;
    static LUT: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        })
    });
    if width < GRID || height < GRID || frame.len() < (width * height * 4) as usize {
        return None;
    }
    let (ri, bi) = if bgr { (2, 0) } else { (0, 2) };
    let mut peaks = Vec::with_capacity((GRID * GRID) as usize);
    for ty in 0..GRID {
        let (y0, y1) = (ty * height / GRID, (ty + 1) * height / GRID);
        for tx in 0..GRID {
            let (x0, x1) = (tx * width / GRID, (tx + 1) * width / GRID);
            let mut peak = 0.0f32;
            for sy in 0..SAMPLES {
                let y = y0 + (y1 - y0) * (2 * sy + 1) / (2 * SAMPLES);
                for sx in 0..SAMPLES {
                    let x = x0 + (x1 - x0) * (2 * sx + 1) / (2 * SAMPLES);
                    let o = ((y * width + x) * 4) as usize;
                    let l = 0.2126 * LUT[frame[o + ri] as usize] + 0.7152 * LUT[frame[o + 1] as usize] + 0.0722 * LUT[frame[o + bi] as usize];
                    peak = peak.max(l);
                }
            }
            peaks.push(peak);
        }
    }
    let top = peaks.iter().copied().fold(0.0f32, f32::max);
    if top <= 1e-6 {
        return None;
    }
    let lit = peaks.iter().filter(|&&p| p > top * 0.10).count();
    if lit * 5 < peaks.len() {
        return None;
    }
    peaks.sort_by(f32::total_cmp);
    let white = peaks[(peaks.len() - 1) * 9 / 10];
    (white > 1e-4).then_some(white)
}

/// Frame hold: the captured frame (and the proxy sent for it) that every present keeps working on
/// while `hold_frame` is on, so a settings change is judged against the same picture.
struct HeldFrame {
    width: u32,
    height: u32,
    original: Vec<u8>,
    proxy: Vec<u8>,
    sent: (u32, u32),
}
static HELD: std::sync::Mutex<Option<HeldFrame>> = std::sync::Mutex::new(None);

/// The longest a present waits for its own frame's answer before going out untouched.
const SYNC_BUDGET: std::time::Duration = std::time::Duration::from_millis(250);

/// `NEURAL_FORGE_PIPELINED=1` restores the old pipelined present: never waits for the model,
/// higher frame rate, but each answer is applied to a later frame than the one it was computed
/// for, which ghosts whenever the camera moves.
fn pipelined_present() -> bool {
    #[cfg(test)]
    if TEST_PIPELINED.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| neural_forge_protocol::env::flag("NEURAL_FORGE_PIPELINED"))
}

/// Lets the pipelined-mode tests run that mode without touching the process environment.
#[cfg(test)]
static TEST_PIPELINED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Held by every test that depends on the present mode, so they never see each other's.
#[cfg(test)]
static TEST_MODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Lets the zero-copy tests run the same direct-capture present with the CPU copies, as the
/// reference the zero-copy output is compared against.
#[cfg(test)]
static TEST_NO_ZERO_COPY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes [`ensure_direct_capture`] fail, standing in for a driver that refuses the host
/// pointer import.
#[cfg(test)]
static TEST_FAIL_DIRECT_SETUP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn zero_copy_allowed() -> bool {
    #[cfg(test)]
    if TEST_NO_ZERO_COPY.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    true
}

/// The zero-copy compose's answer-region guard ([`crate::composition::gpu::GpuCompose::wait_answer_region_reads`]),
/// checked before every request is handed to the helper: `true` when nothing on the GPU still
/// reads the region the helper is about to overwrite.
fn answer_region_free(gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>, device: &ash::Device) -> bool {
    gpu_compose.as_mut().is_none_or(|gpu| gpu.wait_answer_region_reads(device))
}

/// Set when capture resource creation failed during the last [`run`]. The present hook reads
/// it with [`take_setup_failure`] to release this swapchain's primary claim: a claim held by
/// a swapchain that cannot drive the channel would keep every peer of equal or smaller area
/// from taking over.
static SETUP_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn note_setup_failure() {
    SETUP_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether capture resource creation failed since the last call. Clears the flag.
pub fn take_setup_failure() -> bool {
    SETUP_FAILED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// # Safety
/// Same contract as [`run_sync`]: `queue` must be the same queue `image`'s
/// presentation was requested on, with no concurrent use of it from another thread
/// for the duration of this call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn run(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    capture_image: vk::Image,
    capture_layout: vk::ImageLayout,
    image: vk::Image,
    width: u32,
    height: u32,
    proxy_format: u32,
    bgr_order: bool,
    resources: &mut Option<CaptureResources>,
    pipeline: &mut Option<CapturePipeline>,
    direct: &mut [Option<DirectCapture>; 2],
    external_memory_host: &mut bool,
    gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>,
    shm: &mut ShmClient,
    original_scratch: &mut Vec<u8>,
    model_scratch: &mut Vec<u8>,
    inflight: &mut [Inflight; 2],
    bootstrap_complete: &mut bool,
    answer_scratch: &mut Vec<u8>,
    raw_answer_base: &mut Vec<u8>,
    raw_answer_generation: &mut u64,
    last_answer: &mut Vec<u8>,
    last_answer_dims: &mut (u32, u32),
) -> Option<vk::Semaphore> {
    // Cheap enough to leave on every frame: this is what turns the bounded fence-wait
    // markers in `note_fence_wait` into a trail with frame boundaries in it, not just
    // an undated list of wait completions. See `breadcrumbs`' own doc comment.
    crate::breadcrumbs::mark("capture::run enter");
    let pipeline_start = std::time::Instant::now();
    // `composition_settings()` (and everything else below) only ever reads through an
    // already-open mapping -- nothing about it opens one. Every real path that DOES
    // open the mapping (`try_round_trip`/`begin_async_request`) lives later in this
    // same function, gated behind the `composition_settings()` check right below.
    // Real bug, found and fixed 2026-09-11 via a live `vkcube` bisection on
    // `lordnikon`: on a brand-new process the mapping is never open yet, so this used
    // to return `None` here on literally every single frame, forever -- this function
    // was being called every present call (confirmed real, not theoretical) but never
    // actually captured or sent a single frame, because it always bailed out before
    // ever reaching the code that would open the mapping in the first place.
    // `ShmClient::open` is cheap to call unconditionally (an immediate no-op once
    // already open, see its own early return), so there's no real cost to calling it
    // here up front instead of leaving each caller to remember to.
    shm.open();
    // Decided fresh each call rather than cached: `vkGetPhysicalDeviceProperties2` is
    // a cheap, purely local query (the driver already has this value on hand, no real
    // round trip), so there's no need for a whole extra piece of per-device state just
    // to memoize something this inexpensive. Checked against the *actual* runtime
    // proxy-region pointer, not just its constant offset within the mapping --
    // `mmap`'s returned address alignment is not something this project can assume
    // beyond the page size POSIX guarantees, and a wrong guess here would silently
    // stop `poll_or_submit_capture` from ever completing a capture again for the rest
    // of this device's life (`ensure_direct_capture` failing is the only signal, and
    // this function has no per-call fallback to `CapturePipeline` once `use_direct` is
    // decided) rather than fail open onto the always-correct staging-buffer path.
    // Both slots' regions must independently satisfy the driver's alignment (they're
    // both just offsets within the same mapping, so in practice this only ever
    // differs if the mapping's own base address doesn't -- but "in practice" is
    // exactly the kind of assumption this project's own history says to verify, not
    // guess) -- `DirectCapture`/`CapturePipeline` are chosen once for the whole
    // device, never per-slot (see `direct_capture`'s own doc comment in `device.rs`),
    // so a single combined decision is what `run` actually needs here.
    let use_direct = *external_memory_host
        && (0..2).all(|slot| {
            shm.proxy_region(slot).is_some_and(|(ptr, capacity)| {
                min_imported_host_pointer_alignment(instance, physical_device).is_some_and(|alignment| {
                    let alignment = alignment as usize;
                    alignment != 0 && (ptr as usize) % alignment == 0 && capacity % alignment == 0
                })
            })
        });
    let Some(settings) = shm.composition_settings() else { return None };
    // A pending `capture_request` (a real PNG dump to disk) needs *this* frame's own original
    // and answer read back onto the CPU -- same-frame correctness matters more than throughput
    // for a rare, deliberately-triggered one-shot dump, not the normal per-frame path this
    // function otherwise replaces.
    //
    // A game that renders into its own image and blits into the swapchain (`capture_image !=
    // image`, e.g. GTA V through vkd3d) has no same-frame original to dump: the dump would be
    // read back from someone else's memory. Those requests cannot be served here, and they must
    // never stop the layer either: returning early with the request still pending switched the
    // whole pipeline off, on every present, until something cleared it -- a single `shmctl
    // capture` (or the GUI's capture button) was enough. So the request is consumed and reported
    // once instead.
    //
    // `debug_view` has no such limitation any more: it flows through the normal
    // synchronous-present path below instead (that path already guarantees this frame's own
    // answer via the answered_w/h check), which works whether or not the game blits into its
    // swapchain and gives every debug view the full mode-2 pipeline (colour trust, ratio
    // smoothing) rather than `run_sync`'s classic-only one.
    let sync_debug = capture_image == image;
    if !sync_debug && shm.capture_request_pending() {
        shm.take_capture_request();
        static SAID_CAPTURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID_CAPTURE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[layer] capture request dropped: this game blits into the swapchain, so there is no same-frame original to dump");
            crate::logging::flush();
        }
    }
    if sync_debug && shm.capture_request_pending() {
        // `run_sync` does its own round trip, which lets the helper write the answer region.
        if !answer_region_free(gpu_compose, device) {
            return None;
        }
        return unsafe {
            run_sync(
                device,
                instance,
                physical_device,
                queue,
                queue_family,
                image,
                width,
                height,
                proxy_format,
                bgr_order,
                resources,
                gpu_compose,
                shm,
                original_scratch,
                last_answer,
            )
        };
    }
    // "Off keeps the whole pass running... and simply presents the clean frame" --
    // `ShmHeader::apply_model`'s own doc comment -- and the model being permanently
    // unavailable is the same "nothing will ever consume a captured frame" case
    // `ShmClient::model_known_unavailable`'s own doc comment already covers. Either
    // way, paying for a capture+round-trip cycle nobody will use is pure waste;
    // skip the whole pipeline and let the caller present `image` untouched.
    // `neural_enabled` (the GUI's own "Enabled" toggle, `ShmHeader::enabled`) is
    // included here too -- a real bug, found 2026-09-11: `ShmHeader::neural_enabled()`
    // existed and the GUI wrote to it, but nothing in this crate ever read it back,
    // so turning "Enabled" off in the GUI had no effect on anything real at all.
    let bytes_per_pixel = neural_forge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > crate::shm::region_capacity() {
        return None;
    }

    let disabled = !settings.apply_model || !settings.neural_enabled || shm.model_known_unavailable();
    // The helper cannot allocate its NGX images until it knows the game's actual
    // swapchain dimensions.  Waiting until the user enables NR often means GTA has
    // already consumed nearly all VRAM, making that first allocation fail forever.
    // Send exactly one frame while disabled to create the feature early, then discard
    // its answer.  This is deliberately one request per process, never a hidden
    // rendering loop while NR is switched off.
    if disabled {
        // The one-shot bootstrap only ever needs one wire slot -- deliberately always
        // slot 0, exactly like `run_sync`'s own single-slot debug path, so it never
        // depends on protocol v3's second slot existing at all.
        const SLOT: usize = 0;
        if *bootstrap_complete {
            return None;
        }
        if shm.has_pending_request(SLOT) {
            if shm.poll_async_request(SLOT) == Some(true) {
                *bootstrap_complete = true;
            }
            return None;
        }
        // Same non-blocking capture strategy as the enabled path below, used here too
        // so this one-shot warm-up capture never costs a blocking fence wait either --
        // it just takes a call or two longer to land (irrelevant for a once-per-process
        // bootstrap) instead of stalling the present it happens on. Deliberately always
        // full-resolution, `model: None`, regardless of `working_scale`: its answer is
        // discarded either way, and if scaling is active the first *real* request will
        // still need its own `CreateFeature` at the scaled resolution -- this bootstrap
        // only helps avoid the VRAM-full failure mode, it does not need to guess the
        // eventual real resolution correctly to do that.
        if let CaptureStep::Captured(_) = poll_or_submit_capture(
            SLOT, use_direct, external_memory_host, pipeline, direct, device, instance, physical_device, queue, queue_family,
            capture_image, capture_layout, width, height, proxy_format, frame_bytes, None,
            crate::composition::encode_pass::EncodePush {
                white_point: settings.white_point,
                bgr_order: u32::from(bgr_order),
                reversible_mode: settings.reversible_mode,
            },
            shm, original_scratch, model_scratch, None, None,
        ) {
            if answer_region_free(gpu_compose, device) && shm.begin_async_request(SLOT) {
                inflight[SLOT].dims = Some((width, height, proxy_format));
                inflight[SLOT].proxy_dims = None;
            }
        }
        return None;
    }

    // `working_scale`: `(model_width, model_height)` equals `(width, height)` whenever
    // scaling is off or the current proxy format can't be scaled (see
    // `model_scratch_format`'s own doc comment) -- `model_request` is `None` in either
    // case, and every path below behaves exactly as it did before `working_scale`
    // existed. Computed once per call, shared by every slot below (the swapchain's
    // resolution and the settings are the same for all of them this frame).
    let (model_width, model_height) = scaled_dims(width, height, effective_working_scale(width, height, settings.working_scale));
    // Requested unconditionally, not only when the scale actually reduces the raster:
    // the scratch is now also where the proxy encode runs (leg 1's `ENCODE`), and the
    // encode has to happen at every working scale, including exactly 1.0. At 1.0 the
    // blit into it is 1:1 -- a device-local full-rate blit, which is what buys the
    // encode a storage image to dispatch over. `None` only when the proxy format
    // itself cannot be scratched at all, which is the same condition as before.
    let model_request = model_scratch_format(proxy_format, bgr_order).map(|format| (model_width, model_height, format));
    // Direct capture writes the whole frame straight into shared memory: no resize, no encode.
    // It is only correct when the model works on exactly the frame, which rules it out for any
    // working_scale, the model pixel cap, and odd-sized frames (rounded to even above). Missing
    // this sent full-size and odd-size frames past every one of those limits on devices that
    // support host-memory import, and the helper rejected them.
    let use_direct = use_direct && (model_width, model_height) == (width, height);
    // The zero-copy compose (see the synchronous present below) also imports slot 0's answer
    // region: the same live alignment check as the proxy regions above, with the import sized to
    // the frame rounded up to that alignment (not the whole region: fewer pinned pages).
    let answer_import = if use_direct {
        shm.answer_region(0).and_then(|(ptr, capacity)| {
            let alignment = min_imported_host_pointer_alignment(instance, physical_device)?;
            let bytes = frame_bytes.div_ceil(alignment) * alignment;
            ((ptr as u64) % alignment == 0 && bytes <= capacity as u64).then_some((ptr, bytes))
        })
    } else {
        None
    };

    // Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`): the same poll-then-maybe-submit sequence
    // as before, just run once per wire slot instead of once total. Each slot is
    // completely independent -- slot 1 submitting a new capture never waits on slot
    // 0's own pending request, and vice versa, which is the entire point of having
    // two slots instead of one. If *both* slots answer within the same present call,
    // the second one processed simply overwrites `last_answer`/`raw_answer_base` --
    // the same "whichever is freshest wins" bounded-staleness tradeoff `run`'s own
    // doc comment already documents for a single slot, not a new one v3 introduces.
    // Set by the synchronous present when the model worked below the frame's size: the small
    // proxy it was shown is still in `model_scratch`, and the composition needs it for the
    // transfer modes.
    let mut compose_small_proxy = false;
    // Set by the synchronous present while frame hold is on: the composition works on the held
    // frame (`raw_answer_base`) instead of the live swapchain image.
    let mut compose_held = false;
    // Set when this present reuses the previous answer (model interval above 1).
    let mut carried_answer = false;
    // Set by the synchronous present when this frame's answer went the zero-copy way: it is in
    // the answer region and the frame in the capture target, and neither was copied to the CPU.
    let mut zc_fresh = false;
    if pipelined_present() {
        for slot in 0..2 {
            // Poll whatever was sent on some earlier frame *before* touching anything
            // else -- `inflight[slot]`'s current contents correspond to it, and must be
            // read (below) before a new capture this same frame (if one happens) is
            // allowed to replace them.
            let mut have_answer = false;
            // Captured *before* the submit branch below can overwrite
            // `inflight[slot].proxy_dims` with a brand-new request's own dims -- reading
            // it again after that point would describe the wrong request. `(width,
            // height)` is a safe placeholder while `have_answer` is `false`; nothing below
            // reads `answer_dims` unless `have_answer` is `true`, at which point it was
            // always actually set from the branch just below.
            let mut answer_dims = (width, height);
            if shm.has_pending_request(slot) {
                if shm.poll_async_request(slot) == Some(true) {
                    // The answer comes back at whatever resolution *this outstanding
                    // request* was actually sent at (`inflight[slot].proxy_dims`), not
                    // necessarily this frame's own `(width, height)` -- see
                    // `Inflight::proxy_dims`'s own doc comment. Falls back to the
                    // swapchain's own `frame_bytes` when `proxy_dims` is `None`, the only
                    // possibility before `working_scale` existed.
                    answer_dims = inflight[slot].proxy_dims.unwrap_or((width, height));
                    let (aw, ah) = answer_dims;
                    let answer_bytes = (u64::from(aw) * u64::from(ah) * bytes_per_pixel) as usize;
                    answer_scratch.resize(answer_bytes, 0);
                    shm.read_answer(slot, answer_scratch);
                    // Preserve the exact game frame supplied to the model before the next
                    // request replaces `inflight[slot].original`; the temporal GPU path
                    // uses it to carry only the model's enhancement delta onto current
                    // frames.
                    raw_answer_base.clear();
                    raw_answer_base.extend_from_slice(&inflight[slot].original);
                    have_answer = true;
                }
            }

            // Non-blocking capture (`docs/ASYNC_CAPTURE_DESIGN.md`, `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`):
            // poll whatever capture is already in flight for this slot -- never a queue/
            // fence wait -- before deciding whether to submit a new one on it. Same
            // per-slot wire-protocol constraint as before: only start a new round trip on
            // this slot (and only bother keeping a just-finished capture's bytes at all)
            // when nothing is already outstanding on it. A capture that finishes while a
            // request is *already* in flight on this slot is still polled here (freeing
            // its GPU buffer for reuse) but its bytes are simply not consumed -- the same
            // bounded temporal-staleness tradeoff `run`'s own doc comment already
            // accepts, not a new one. `poll_or_submit_capture` never submits in the same
            // call it successfully polls, so a single check here (rather than the two
            // separate ones a poll-then-maybe-submit split would need) already correctly
            // skips submitting a redundant capture on the same call a round trip just
            // started.
            if !shm.has_pending_request(slot) {
                if let CaptureStep::Captured(captured) = poll_or_submit_capture(
                    slot, use_direct, external_memory_host, pipeline, direct, device, instance, physical_device, queue, queue_family,
                    capture_image, capture_layout, width, height, proxy_format, frame_bytes, model_request,
                    crate::composition::encode_pass::EncodePush {
                        white_point: settings.white_point,
                        bgr_order: u32::from(bgr_order),
                        reversible_mode: settings.reversible_mode,
                    },
                    shm, original_scratch, model_scratch, None, None,
                ) {
                    let (sent_w, sent_h) = captured.sent;
                    if answer_region_free(gpu_compose, device) && shm.begin_async_request(slot) {
                        std::mem::swap(&mut inflight[slot].original, original_scratch);
                        inflight[slot].dims = Some((width, height, proxy_format));
                        // The proxy's *actual* sent dims, straight from
                        // `poll_or_submit_capture`'s own return -- not re-derived from
                        // `model_request` here, which would be wrong whenever that
                        // function fell back to full-resolution (a scratch build/resize
                        // failure, or `use_direct`) despite scaling being requested.
                        inflight[slot].proxy_dims = (sent_w != width || sent_h != height).then_some((sent_w, sent_h));
                    }
                }
            }

            if have_answer && inflight[slot].dims == Some((width, height, proxy_format)) && neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format) {
                // Retain the model's raw answer for continuous re-presentation below --
                // deliberately *not* run through `composition::gpu`/`composition::apply`'s
                // tone-map compositor. That compositor's `UpgradeToneMap` targets `original`'s
                // own luminance exactly whenever `original <= proxy`; this pipeline's `proxy
                // == original` (no real downscaled proxy exists yet -- see this crate's other
                // doc comments) makes that true on every pixel, which doesn't just dilute the
                // model's edit but actively fights it: a *stronger* raw answer gets *more*
                // aggressively cancelled by the same ratio-based rescale, confirmed by direct
                // measurement on `lordnikon` 2026-09-12 (maxing every tuning parameter nearly
                // doubled the raw model's own delta from original, then the compositor's
                // output delta *dropped* below the unmodified baseline). No tuning knob fixes
                // that; it's this pipeline's proxy/original conflation actively working
                // against the model's answer, not merely muting it.
                last_answer.clear();
                last_answer.extend_from_slice(answer_scratch);
                *last_answer_dims = answer_dims;
                *raw_answer_generation = raw_answer_generation.wrapping_add(1).max(1);
            }
        }

        // Re-present the most recently retained answer on *every* call, not only the
        // rare one a round trip happens to resolve on. Compositing only on that rare
        // frame (`image` left completely untouched every other frame, this function's
        // very first design) alternates "native" and "one processed frame" -- the
        // flicker this project has fought since v0.1.49/v0.1.50 (see this crate's other
        // doc comments) -- independently of whether that processed frame is stale.
        // Re-blitting the same held answer every frame instead removes the alternation:
        // displayed content is always "the model's edit," refreshed at the round trip's
        // own cadence rather than toggling against untouched frames in between.
        //
        // What that composite *does* with a stale answer against a moved current frame
        // changed 2026-09-17: `composition::gpu::record_temporal_delta_into_image` used
        // to carry the stale delta forward with a motion-based suppression mask
        // (`compose.comp`'s deleted `carry_delta` branch) -- this crate independently
        // arrived at the same reprojection-with-suppression technique DLSS5VKLayer's own
        // AGPL-3.0 source documents trying and measuring as a dead end. It now
        // re-anchors to the current frame with a plain ratio-transfer, no suppression,
        // matching upstream's own resolve function directly -- see
        // `ATTRIBUTION.md`/`docs/GHOSTING_PLAN.md`. The real fix for genuine motion-driven
        // staleness is still a downscaled-proxy-plus-motion-vector pipeline (this
        // crate's own doc comment on motion vectors being disabled), not attempted here.
    } else {
        // Synchronous present (the default): this frame is captured, the model answers *this*
        // frame, and the answer is composed onto *this* frame before it is presented -- the
        // way upstream works. An answer is therefore never applied to a frame the camera has
        // moved on from, which is what produced the ghosting (a pale copy of the old frame's
        // edges and text laid over the new one) in the pipelined mode.
        //
        // The wait is bounded: a frame whose answer is not back within `SYNC_BUDGET` is
        // presented untouched, and the late request is never waited on again, so a slow,
        // warming-up, restarted or missing helper can delay a frame by at most the budget and
        // can never hang the game.
        const SLOT: usize = 0;
        // Model interval: on the presents between model runs, carry the last answer onto this
        // frame instead of capturing and waiting (the pipelined mode's way, for one frame at a
        // time). With frame generation this spares the generated frames the model's wait.
        let present_no = inflight[SLOT].presents;
        inflight[SLOT].presents = present_no.wrapping_add(1);
        // An answer to carry is either the CPU pair or, after a zero-copy present, the GPU's.
        let answer_held = !last_answer.is_empty() || gpu_compose.as_ref().is_some_and(|gpu| gpu.holds_generation(*raw_answer_generation));
        let carry = settings.model_interval > 1
            && present_no % u64::from(settings.model_interval) != 0
            && !settings.hold_frame
            && answer_held
            && inflight[SLOT].dims == Some((width, height, proxy_format));
        if carry {
            carried_answer = true;
        }
        // No live helper (never started, stopped, killed, restarting): present untouched and
        // do not wait for anything. This is what keeps a missing helper from costing a stall.
        if !carry && !shm.helper_alive() {
            shm.publish_frame_timing(pipeline_start.elapsed(), false);
            return None;
        }
        if !carry {
            let deadline = std::time::Instant::now() + SYNC_BUDGET;
            // Left over from a frame that ran out of time (or from pipelined mode): its answer
            // belongs to an old frame. Discard it when it lands; never block on it.
            for slot in 0..2 {
                if shm.has_pending_request(slot) && shm.poll_async_request(slot) == Some(false) && slot == SLOT {
                    shm.publish_frame_timing(pipeline_start.elapsed(), false);
                    return None;
                }
            }
            let encode_push = crate::composition::encode_pass::EncodePush {
                white_point: settings.white_point,
                bgr_order: u32::from(bgr_order),
                reversible_mode: settings.reversible_mode,
            };
            let is_8bit = neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format);
            // Zero-copy compose (`docs/EXTERNAL_MEMORY_HOST_DESIGN.md`): with direct capture at the
            // model's own size, the frame and the answer stay on the GPU all the way into the
            // compose. The capture also copies the frame into a device-local target, the answer
            // region is imported, and neither crosses to the CPU -- no copy out of the proxy
            // region, no `read_answer`, no staging upload. Frame hold keeps the CPU copies (the held
            // frame lives on the CPU), and a present that cannot set this up keeps them too.
            let mut zc_target = None;
            if let Some((answer_ptr, import_bytes)) = answer_import.filter(|_| !settings.hold_frame && is_8bit && zero_copy_allowed()) {
                if gpu_compose.is_none() {
                    *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
                }
                if let Some(gpu) = gpu_compose.as_mut() {
                    if gpu.ensure_answer_import(device, instance, physical_device, answer_ptr, import_bytes) {
                        let capture_idle = direct[SLOT].as_ref().is_none_or(|d| d.pending.is_none());
                        zc_target = gpu.zero_copy_capture_target(device, instance, physical_device, queue, frame_bytes, capture_idle);
                    }
                }
            }
            let t_start = std::time::Instant::now();
            let mut sent = None;
            let mut gpu_original = false;
            let mut copy_out = std::time::Duration::ZERO;
            let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
            if !settings.hold_frame || held.as_ref().is_some_and(|h| (h.width, h.height) != (width, height)) {
                *held = None;
            }
            if let Some(h) = held.as_ref() {
                // Holding: the model is shown the held proxy again, no capture.
                let t_copy = std::time::Instant::now();
                original_scratch.clone_from(&h.original);
                model_scratch.clone_from(&h.proxy);
                shm.set_frame_info(SLOT, h.sent.0, h.sent.1, proxy_format);
                shm.write_proxy(SLOT, &h.proxy);
                sent = Some(h.sent);
                copy_out = t_copy.elapsed();
            }
            while sent.is_none() {
                match poll_or_submit_capture(
                    SLOT, use_direct, external_memory_host, pipeline, direct, device, instance, physical_device, queue, queue_family,
                    capture_image, capture_layout, width, height, proxy_format, frame_bytes, model_request,
                    encode_push, shm, original_scratch, model_scratch, zc_target, Some(t_start),
                ) {
                    CaptureStep::Captured(captured) => {
                        sent = Some(captured.sent);
                        gpu_original = captured.gpu_original;
                        copy_out = captured.copy_out;
                        break;
                    }
                    // Nothing is in flight and nothing will be: waiting out the budget would
                    // only cost this frame (and every later one) the whole budget for nothing.
                    CaptureStep::Failed => break,
                    CaptureStep::Pending => {}
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            let Some((sent_w, sent_h)) = sent else {
                shm.publish_frame_timing(pipeline_start.elapsed(), false);
                return None;
            };
            let zero_copy = zc_target.is_some() && gpu_original && (sent_w, sent_h) == (width, height);
            let holding = held.is_some();
            if settings.hold_frame && !holding {
                // Start holding this frame. What was sent is the scaled, encoded scratch unless direct
                // capture wrote the whole frame straight into shared memory.
                let t_copy = std::time::Instant::now();
                let proxy = if (sent_w, sent_h) == (width, height) && use_direct { original_scratch.clone() } else { model_scratch.clone() };
                *held = Some(HeldFrame { width, height, original: original_scratch.clone(), proxy, sent: (sent_w, sent_h) });
                copy_out += t_copy.elapsed();
            }
            drop(held);
            let t_captured = std::time::Instant::now();
            // The white meter: every 8th frame, smoothed (about a second to settle) so the proxy's
            // brightness never jumps from one frame to the next -- a jumping divisor would make the
            // model's input, and therefore its answer, flicker. Published whatever the source setting,
            // so the GUI can show it; only used when the source is Measured.
            {
                static METER_FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if !holding && METER_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 8 == 0 {
                    let frame: &[u8] = if zero_copy {
                        // Zero-copy leaves `original_scratch` empty; the same frame is in the proxy
                        // region. SAFETY: the capture that wrote it was just seen complete (its fence,
                        // host-coherent memory), no request is out yet, and the helper only ever
                        // reads this region.
                        shm.proxy_region(SLOT).map_or(&[], |(ptr, capacity)| unsafe { std::slice::from_raw_parts(ptr, capacity.min(frame_bytes as usize)) })
                    } else {
                        original_scratch
                    };
                    if let Some(white) = meter_white(frame, width, height, bgr_order) {
                        shm.publish_measured_white(white);
                    }
                }
            }
            let t_metered = std::time::Instant::now();
            if !answer_region_free(gpu_compose, device) || !shm.begin_async_request(SLOT) {
                shm.publish_frame_timing(pipeline_start.elapsed(), false);
                return None;
            }
            inflight[SLOT].dims = Some((width, height, proxy_format));
            inflight[SLOT].proxy_dims = (sent_w != width || sent_h != height).then_some((sent_w, sent_h));
            let helper_ms = loop {
                match shm.poll_async_request(SLOT) {
                    Some(true) => {
                        // Only this frame's own answer: the helper echoes the raster it answered.
                        // A mismatch is someone else's (or a stale) answer; present untouched. No echo
                        // at all is a helper from before 0.1.81, which is trusted as before.
                        if shm.answered_dims().is_some_and(|d| d != (sent_w, sent_h)) {
                            shm.publish_frame_timing(pipeline_start.elapsed(), false);
                            return None;
                        }
                        // Published before the answer, so it is this answer's.
                        break shm.helper_stage_ms();
                    }
                    Some(false) if std::time::Instant::now() < deadline => std::thread::sleep(std::time::Duration::from_micros(100)),
                    // Out of time, or the helper is gone: this frame goes out untouched.
                    // Deliberately no breadcrumb dump here, unlike the bounded fence
                    // waits: a slow/warming-up/restarting helper routinely exceeds
                    // `SYNC_BUDGET` (see this function's own comment above), so this
                    // path is expected and already explained -- dumping on every
                    // occurrence would spam the log instead of flagging something
                    // unusual, which is what `breadcrumbs` exists for.
                    _ => {
                        shm.publish_frame_timing(pipeline_start.elapsed(), false);
                        return None;
                    }
                }
            };
            let t_answered = std::time::Instant::now();
            if !is_8bit {
                return None;
            }
            let answer_bytes = (u64::from(sent_w) * u64::from(sent_h) * bytes_per_pixel) as usize;
            if zero_copy {
                // The answer stays in the answer region and the frame in the capture target; the
                // compose copies both into its own device-local pair. An empty `last_answer` marks
                // the CPU pair invalid, so nothing can present a stale CPU answer, and
                // `raw_answer_base` is not read on this path.
                last_answer.clear();
                zc_fresh = true;
            } else {
                // Swapped rather than copied: these are full frames (15 MB at 1440p), and nothing reads
                // the scratch buffers again before the next capture refills them.
                last_answer.resize(answer_bytes, 0);
                shm.read_answer(SLOT, last_answer);
                std::mem::swap(raw_answer_base, original_scratch);
            }
            *last_answer_dims = (sent_w, sent_h);
            compose_small_proxy = (sent_w, sent_h) != (width, height) && model_scratch.len() >= answer_bytes;
            compose_held = settings.hold_frame;
            let capture_total = t_captured - t_start;
            SYNC_TIMING.with(|t| {
                t.set(Some(SyncTiming {
                    capture_wait: capture_total.saturating_sub(copy_out),
                    copy_out,
                    meter: t_metered - t_captured,
                    wait_answer: t_answered - t_metered,
                    helper: std::time::Duration::try_from_secs_f32(helper_ms / 1000.0).unwrap_or_default(),
                    zero_copy,
                }))
            });
        }
        if carry {
            let answer_bytes = (u64::from(last_answer_dims.0) * u64::from(last_answer_dims.1) * bytes_per_pixel) as usize;
            compose_small_proxy = *last_answer_dims != (width, height) && model_scratch.len() >= answer_bytes;
        } else {
            // A new answer: the composition uploads it (a carried one is already on the GPU).
            *raw_answer_generation = raw_answer_generation.wrapping_add(1).max(1);
        }
    }

    // After a zero-copy present the answer exists only on the GPU (see `zc_fresh`).
    let answer_on_gpu = zc_fresh || gpu_compose.as_ref().is_some_and(|gpu| gpu.holds_generation(*raw_answer_generation));
    if last_answer.is_empty() && !answer_on_gpu {
        return None;
    }
    // Which composition formula the answer below is eligible for: mode 2 (the
    // encoded-proxy ratio transfer) only when the capture leg really did encode the
    // proxy this answer came from, mode 1 otherwise. See `compose.comp`.
    let proxy_encoded = pipeline.as_ref().is_some_and(CapturePipeline::proxy_encoded);
    // Cache each helper answer in device-local memory once, then copy that cached
    // frame to every presented swapchain image.  The prior path re-uploaded two 4K
    // CPU buffers and ran the compose shader on every present, which made the counter
    // read in the 50s while frame pacing felt like the teens.  This leaves only one
    // device-local transfer on ordinary presents and preserves the no-flicker held
    // answer policy.
    if gpu_compose.is_none() {
        *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
    }
    if let Some(gpu) = gpu_compose {
        let inputs = if answer_on_gpu {
            crate::composition::gpu::ComposeInputs::Gpu { fresh: zc_fresh }
        } else {
            crate::composition::gpu::ComposeInputs::Cpu {
                base: raw_answer_base,
                answer: last_answer,
                proxy_small: compose_small_proxy.then_some(model_scratch.as_slice()),
                held_original: compose_held.then_some(raw_answer_base.as_slice()),
            }
        };
        if let Some(sem) = gpu.present_temporal_delta_async(
            device,
            instance,
            physical_device,
            queue,
            width,
            height,
            last_answer_dims.0,
            last_answer_dims.1,
            inputs,
            *raw_answer_generation,
            bgr_order,
            image,
            crate::composition::gpu::ComposeParams {
                colour_strength: settings.colour_strength,
                transfer_strength: settings.transfer_strength,
                max_ratio: settings.max_ratio,
                // The guard exists for the pipelined mode's late answers; in the synchronous mode
                // the answer always matches the frame, and against a scaled-up proxy the guard
                // would read the enlargement's blur as motion.
                ghost_guard: if pipelined_present() || carried_answer { settings.ghost_guard } else { 0.0 },
                transfer: settings.transfer,
                model_small: false,
                white_point: settings.white_point,
                compare: settings.compare,
                colour_trust: settings.colour_trust,
                ratio_smooth: settings.ratio_smooth,
                debug_view: settings.debug_view,
                debug_scale: settings.debug_scale,
                proxy_encoded,
                reversible_mode: settings.reversible_mode,
            },
        ) {
            if let Some(t) = SYNC_TIMING.with(|t| t.take()) {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 300 == 5 {
                    let total = pipeline_start.elapsed();
                    crate::log!(
                        "[sync] {width}x{height}: total={total:.1?} capture_gpu={:.1?} copy_out={:.1?} meter={:.1?} wait_answer={:.1?} helper={:.1?} rest(readback+compose)={:.1?} zc={}",
                        t.capture_wait,
                        t.copy_out,
                        t.meter,
                        t.wait_answer,
                        t.helper,
                        total.saturating_sub(t.capture_wait + t.copy_out + t.meter + t.wait_answer),
                        t.zero_copy,
                    );
                    crate::logging::flush();
                }
            }
            shm.publish_frame_timing(pipeline_start.elapsed(), true);
            return Some(sem);
        }
    }
    // No GPU compose available at all (`GpuCompose::new` failed) -- last resort: the
    // same CPU-visible write-back every other fallback path in this module already
    // uses, blocking cost and all.
    if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
        return None;
    }
    let r = resources.as_ref().expect("just ensured above");
    write_bytes_to_image(device, r, queue, image, width, height, last_answer);
    shm.publish_frame_timing(pipeline_start.elapsed(), true);
    None
}

/// Records "copy `image` (in `initial_layout`) into `buffer`, restore `initial_layout`"
/// into `cmd` -- reset, begin, both barriers, the copy, end. Does not submit or wait;
/// [`submit_pipeline_capture`] (the only caller now that the old fully-synchronous
/// single-shot capture path is gone -- see `docs/ASYNC_CAPTURE_DESIGN.md`) does that
/// itself, deliberately without waiting. Pulled out on its own so a future second
/// caller shares the exact same recorded commands rather than a copy that could
/// drift apart -- not, today, because there already is one.
/// `model`, when `Some((scratch_image, scratch_buffer, model_width, model_height))`,
/// additionally blits `image` (already in its read layout for the main copy below)
/// down into `scratch_image` at `(model_width, model_height)` -- `VK_FILTER_LINEAR`, a
/// hardware resize unit, not the CPU resample `working_scale` originally tried and
/// measured too slow for this thread (see `ModelScratch`'s own doc comment) -- then
/// copies `scratch_image` into `scratch_buffer`. That buffer's bytes become the SHM
/// proxy in place of `buffer`'s full-resolution ones; `buffer` is still always filled
/// at `(width, height)` exactly as before `working_scale` existed, since the
/// compositor's motion-mask reference must stay full-resolution.
///
/// `extra_dst`, when `Some`, receives a second copy of `image` at `(width, height)`: the
/// zero-copy compose's device-local capture target (see `composition::gpu::GpuCompose`'s
/// `zero_copy_capture_target`), which the compose that follows reads as the answer's base
/// instead of a CPU copy of the frame.
#[allow(clippy::too_many_arguments)]
fn record_capture_commands(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    initial_layout: vk::ImageLayout,
    buffer: vk::Buffer,
    extra_dst: Option<vk::Buffer>,
    width: u32,
    height: u32,
    model: Option<(vk::Image, vk::Buffer, u32, u32)>,
    encode: Option<(
        &crate::composition::encode_pass::EncodePass,
        vk::DescriptorSet,
        crate::composition::encode_pass::EncodePush,
    )>,
) -> bool {
    // SAFETY: `cmd` was allocated from a pool created with `RESET_COMMAND_BUFFER`.
    if unsafe { device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return false;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` was just reset above.
    if unsafe { device.begin_command_buffer(cmd, &begin_info) }.is_err() {
        return false;
    }
    // The layout `image` is read in. A render-tap source is only ever captured while its
    // tracked layout is GENERAL, in which copies and blits may read it directly: it is read
    // there, with a plain dependency and no layout transition at all, so the game's own image
    // never has its layout rewritten by the layer -- even if the tracked layout were stale (a
    // render pass's implicit transition is not tracked), nothing is transitioned from it.
    // A swapchain image is presented in PRESENT_SRC_KHR, which a copy cannot read, so it goes
    // to TRANSFER_SRC_OPTIMAL and back.
    let read_layout = if initial_layout == vk::ImageLayout::GENERAL { vk::ImageLayout::GENERAL } else { vk::ImageLayout::TRANSFER_SRC_OPTIMAL };
    let to_transfer_src = barrier(
        image,
        initial_layout,
        read_layout,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_READ,
    );
    // SAFETY: `cmd` is in the recording state; `image` is the caller's own, currently
    // `initial_layout` per every caller's own contract on the image it passes in.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_src],
        );
    }
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
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
        .build();
    // SAFETY: `image` is in `read_layout` after the barrier above; `buffer` is sized to
    // at least `width*height*bytes_per_pixel` by whichever caller built it.
    unsafe {
        device.cmd_copy_image_to_buffer(cmd, image, read_layout, buffer, &[region]);
    }
    if let Some(extra_dst) = extra_dst {
        // Same source, same region, into device-local memory. The barrier after it is what a
        // later submission's read of this buffer needs: a pipeline barrier's second scope runs
        // on through every later command on the queue, so the compose that consumes this
        // capture sees the write without a barrier of its own (the same pattern as
        // `composition::gpu`'s cached buffer).
        let extra_ready = vk::BufferMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(extra_dst)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build();
        // SAFETY: `image` is still in `read_layout`; `extra_dst` is sized for at least
        // `width*height*4` bytes by whoever handed it over.
        unsafe {
            device.cmd_copy_image_to_buffer(cmd, image, read_layout, extra_dst, &[region]);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[extra_ready], &[]);
        }
    }
    if let Some((scratch_image, scratch_buffer, model_width, model_height)) = model {
        // `image` is still in `read_layout` from the copy above -- read from it
        // again for the blit, same source, no extra barrier needed on this side.
        // `scratch_image` starts from `UNDEFINED` every call: a blit fully overwrites
        // the whole image, so there is never any prior content worth preserving, and
        // `UNDEFINED` as `oldLayout` is valid regardless of the image's actual current
        // layout (the exact property that lets this skip tracking it across frames).
        let scratch_to_dst = barrier(scratch_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
        // SAFETY: `cmd` is recording; `scratch_image` is this slot's own, not aliased.
        unsafe { device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[scratch_to_dst]) };
        let blit = vk::ImageBlit::builder()
            .src_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
            .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: width as i32, y: height as i32, z: 1 }])
            .dst_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
            .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: model_width as i32, y: model_height as i32, z: 1 }])
            .build();
        // SAFETY: `image` is in `read_layout`; `scratch_image` was just
        // transitioned to `TRANSFER_DST_OPTIMAL`; both are 2D, single-mip, single-layer.
        unsafe { device.cmd_blit_image(cmd, image, read_layout, scratch_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR) };
        // The encode, in place over the scratch the blit just filled and before the
        // download below reads it -- this is leg 1's `ENCODE` step, and it is the whole
        // reason the proxy the model sees is not a bit-identical copy of the frame any
        // more (see `composition::encode`'s module doc comment).
        //
        // `GENERAL` is the only layout a storage image may be written through, so the
        // scratch goes `TRANSFER_DST -> GENERAL` for the dispatch and `GENERAL ->
        // TRANSFER_SRC` for the copy, rather than straight from one transfer layout to
        // the other. Both barriers carry the real stage/access pair for the direction
        // they guard, so the dispatch cannot start before the blit's writes are visible
        // and the copy cannot start before the dispatch's are.
        if let Some((pass, set, push)) = encode {
            let to_general = barrier(
                scratch_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            // SAFETY: `cmd` is recording; `scratch_image` was just written by the blit
            // and is this slot's own, not aliased.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
            // SAFETY: `cmd` is recording; `set` was allocated by this same `pass` and
            // points at exactly this `scratch_image`, whose extent is
            // `model_width`x`model_height`.
            unsafe { pass.record(device, cmd, set, model_width, model_height, push) };
            let to_src = barrier(
                scratch_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            // SAFETY: `cmd` is recording; the dispatch above wrote `scratch_image`.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_src],
                );
            }
        } else {
            let scratch_to_src = barrier(scratch_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::TRANSFER_READ);
            // SAFETY: `cmd` is recording; `scratch_image` was just written by the blit above.
            unsafe { device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[scratch_to_src]) };
        }
        let scratch_region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D { width: model_width, height: model_height, depth: 1 })
            .build();
        // SAFETY: `scratch_image` is `TRANSFER_SRC_OPTIMAL`; `scratch_buffer` was sized
        // for exactly `model_width*model_height*4` bytes by `build_model_scratch`.
        unsafe { device.cmd_copy_image_to_buffer(cmd, scratch_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, scratch_buffer, &[scratch_region]) };
    }
    // Restore `image` to exactly the layout this function found it in -- nothing is
    // guaranteed to touch `image` again this same frame, so leaving it in
    // `TRANSFER_DST_OPTIMAL` (a layout only valid mid-way through an image<->buffer
    // round trip) would be a real bug the moment the real present call, or the game's
    // own next use of a render-tap source, ran against it instead.
    let to_present = barrier(
        image,
        read_layout,
        initial_layout,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::empty(),
    );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
    }
    unsafe { device.end_command_buffer(cmd) }.is_ok()
}

/// Phase 2 (`docs/ASYNC_CAPTURE_DESIGN.md`): two [`CaptureBuffer`] slots so a new capture
/// submission never has to wait on the previous one's fence first. `run` only ever
/// calls [`poll_pipeline_capture`] (non-blocking: did an earlier submission finish?)
/// and [`submit_pipeline_capture`] (non-blocking: start a new one if a slot is free)
/// against this -- no queue wait, no fence wait, on the per-present path. The one
/// place this module *does* block on a pipeline fence is [`ensure_pipeline`]'s resize
/// path, which is not on that path.
pub struct CapturePipeline {
    queue_family: u32,
    slots: [PipelineSlot; 2],
    /// The proxy encode's pipeline (see [`crate::composition::encode_pass`]), built
    /// once on first use and shared by both slots -- each slot owns only its own
    /// descriptor set, allocated from this pass's pool. `None` means the encode is
    /// unavailable on this device, in which case every proxy crosses to the helper
    /// unencoded exactly as it did before the encode existed.
    encode: Option<crate::composition::encode_pass::EncodePass>,
}

impl CapturePipeline {
    /// Whether the proxies this pipeline produces actually go through the encode --
    /// which decides the composition mode (see `compose.comp`). Device- and
    /// format-stable in practice (it depends on `STORAGE` support for the proxy
    /// format, not on anything per-frame), so reading it from whichever slot has
    /// already built its scratch is enough.
    fn proxy_encoded(&self) -> bool {
        self.slots.iter().any(|s| s.model.as_ref().is_some_and(|m| m.encode_set.is_some()))
    }
}

struct PipelineSlot {
    buf: CaptureBuffer,
    /// `working_scale`'s scratch (see [`ModelScratch`]), built and resized lazily by
    /// [`submit_pipeline_capture`] -- `None` until the first scaled request, same
    /// "only pay for what's used" discipline as everything else lazy in this module.
    model: Option<ModelScratch>,
    /// `Some((width, height, proxy_format, model_dims))` for a submission whose fence
    /// has not yet been confirmed signaled by [`poll_pipeline_capture`]; `model_dims`
    /// is `Some((model_width, model_height))` exactly when that submission also
    /// recorded a scaled blit into `model`. Nothing may reset or reuse
    /// `buf.cmd`/`buf.buffer`/`buf.memory`/`model` while this is `Some` -- the exact
    /// invariant whose violation caused the 2026-09-12 UB regression documented in
    /// this project's history (`docs/history/development-before-neuralforge.md`).
    pending: Option<(u32, u32, u32, Option<(u32, u32)>, std::time::Instant)>,
}

/// Builds both slots if `existing` is `None`; rebuilds both (same capacity/queue-
/// family-change trigger as [`ensure`]) if either is undersized or the queue family
/// changed. Unlike [`ensure`], a slot reaching this function may legitimately still
/// be `pending` -- draining it is this function's job, not a precondition callers
/// have to uphold, since the entire point of the pipeline is that callers never wait
/// on a pending slot themselves.
fn ensure_pipeline(
    existing: &mut Option<CapturePipeline>,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: vk::DeviceSize,
) -> bool {
    if let Some(p) = existing.as_ref() {
        if p.queue_family == queue_family && p.slots.iter().all(|s| s.buf.capacity >= bytes) {
            return true;
        }
    } else {
        return build_pipeline(existing, device, instance, physical_device, queue_family, bytes);
    }
    // A rebuild is needed (capacity grew or the queue family changed -- the same two
    // triggers `ensure` already has). Resize/teardown is rare and not latency
    // sensitive, so this is one of the places in the pipeline that takes a real,
    // blocking wait -- never on the steady-state per-frame path. Bounded rather than
    // truly unbounded: see `crate::FENCE_WAIT_TIMEOUT`.
    let p = existing.as_ref().expect("checked above");
    for slot in &p.slots {
        if slot.pending.is_some() {
            // SAFETY: `slot.buf.fence` is this slot's own fence; waiting for it here,
            // before any destroy below touches the resources it guards, is exactly
            // what makes that destroy sound -- the "drain before rebuilding on a live
            // device" this pipeline's own design doc calls for.
            let wait = unsafe { device.wait_for_fences(&[slot.buf.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
            if crate::note_fence_wait(wait, "capture::ensure_pipeline rebuild drain").is_err() {
                // A real device error, or the bounded wait above finally timed out.
                // Either way, leave the existing pipeline exactly as it was rather
                // than guess it's safe to destroy -- next frame's `ensure_pipeline`
                // call tries again.
                return false;
            }
        }
    }
    let p = existing.take().expect("checked above");
    for slot in &p.slots {
        // SAFETY: every slot's fence was just confirmed signaled above -- that same
        // fence also guards any blit/copy/dispatch this slot's `model` scratch was
        // involved in, since all of them are recorded into and submitted on the same
        // command buffer.
        unsafe {
            slot.buf.destroy(device);
            if let Some(model) = &slot.model {
                model.destroy(device, p.encode.as_ref());
            }
        }
    }
    // After every set allocated from it has been freed above.
    // SAFETY: same fence reasoning as the slots themselves.
    if let Some(pass) = &p.encode {
        unsafe { pass.destroy(device) };
    }
    build_pipeline(existing, device, instance, physical_device, queue_family, bytes)
}

fn build_pipeline(
    existing: &mut Option<CapturePipeline>,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    bytes: vk::DeviceSize,
) -> bool {
    let Some(a) = build_capture_buffer(device, instance, physical_device, queue_family, bytes) else { return false };
    let Some(b) = build_capture_buffer(device, instance, physical_device, queue_family, bytes) else {
        // SAFETY: `a` was just built above; nothing has submitted work against it yet.
        unsafe { a.destroy(device) };
        return false;
    };
    *existing = Some(CapturePipeline {
        queue_family,
        slots: [PipelineSlot { buf: a, model: None, pending: None }, PipelineSlot { buf: b, model: None, pending: None }],
        // Fail-open: `None` just means no encode, and the composition is told.
        encode: crate::composition::encode_pass::EncodePass::new(device),
    });
    true
}

/// Non-blocking: checks the given slot for a submission whose fence has actually
/// signaled (`vkGetFenceStatus`, never `vkWaitForFences`) and, if so, copies its bytes
/// into `out` (and, if this submission also requested a scaled proxy, into
/// `out_model` too) and frees the slot. Returns `(full_res_dims, model_dims)`:
/// `full_res_dims` is `Some((width, height, proxy_format))` -- unchanged meaning from
/// before `working_scale` existed, the caller's full-resolution-capture-completed
/// signal -- and `model_dims`, independent of it, is `Some((model_width,
/// model_height))` exactly when `out_model` was actually populated this call. `None`
/// for `full_res_dims` (both left untouched) if nothing is signaled yet, or if this
/// slot's fence reported a real error (left `pending` forever rather than guessed
/// safe to reuse).
///
/// Indexed by `slot`, not pooled: protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) dedicates
/// `pipeline.slots[0]` to wire slot 0's captures and `pipeline.slots[1]` to wire slot
/// 1's, one-to-one, rather than handing either wire slot whichever GPU buffer happens
/// to be free. That mapping is sound precisely because a caller only ever submits a
/// new capture into `pipeline.slots[slot]` when wire slot `slot` itself has no
/// request outstanding (see `run`'s own orchestration) -- by the time that's true,
/// any earlier capture headed for this same wire slot has already been polled out
/// and sent, so this GPU slot is free too, not just "some" slot in a shared pool.
fn poll_pipeline_capture(
    pipeline: &mut CapturePipeline,
    slot: usize,
    device: &ash::Device,
    out: &mut Vec<u8>,
    out_model: &mut Vec<u8>,
) -> (Option<(u32, u32, u32, std::time::Instant)>, Option<(u32, u32)>) {
    let slot = &mut pipeline.slots[slot];
    let Some((width, height, proxy_format, model_dims, submitted)) = slot.pending else { return (None, None) };
    // SAFETY: `slot.buf.fence` belongs to this slot; a status query never touches
    // command-buffer/buffer/memory state, so it's sound to call regardless of
    // whether the submission this fence guards has actually completed yet. The same
    // fence guards `slot.model`'s blit/copy too (recorded into and submitted on the
    // same command buffer), so one status query covers both.
    match crate::note_vk(unsafe { device.get_fence_status(slot.buf.fence) }) {
        Ok(true) => {
            let bytes_per_pixel = neural_forge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
            let frame_bytes = (u64::from(width) * u64::from(height) * bytes_per_pixel) as usize;
            // SAFETY: `slot.buf.ptr` is a live host-coherent mapping of at least
            // `frame_bytes` bytes (`submit_pipeline_capture` only ever submits
            // into a slot `build_capture_buffer` already sized for this exact
            // `frame_bytes`); the fence just confirmed signaled means the GPU's
            // writes are visible to the CPU with no explicit flush/invalidate
            // needed (host-coherent memory, same as every other read of a
            // `CaptureBuffer::ptr` in this module).
            let captured = unsafe { std::slice::from_raw_parts(slot.buf.ptr, frame_bytes) };
            out.clear();
            out.extend_from_slice(captured);
            let model_result = match (model_dims, slot.model.as_ref()) {
                (Some((mw, mh)), Some(m)) if m.width == mw && m.height == mh => {
                    let model_bytes = (mw as usize) * (mh as usize) * 4;
                    if (m.capacity as usize) >= model_bytes {
                        // SAFETY: same reasoning as `captured` above -- `m.ptr` is a
                        // live host-coherent mapping of at least `model_bytes`
                        // (`build_model_scratch` sizes it to exactly `width*height*4`),
                        // and the fence just confirmed signaled covers this write too.
                        let model_captured = unsafe { std::slice::from_raw_parts(m.ptr, model_bytes) };
                        out_model.clear();
                        out_model.extend_from_slice(model_captured);
                        Some((mw, mh))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            slot.pending = None;
            (Some((width, height, proxy_format, submitted)), model_result)
        }
        Ok(false) => (None, None), // still in flight -- leave `pending`, check again next call
        Err(_) => (None, None),    // real device error -- leave `pending`; never guess reuse is safe
    }
}

/// Non-blocking: records and submits a new capture into the given slot, if it's free
/// (`pending: None`). `false` (no new capture this frame) if it's still pending or
/// recording/submission itself failed -- the caller already treats that as "skip
/// capture this frame", the same fail-open discipline as every other path in this
/// module. Never waits, never touches a slot that is still `pending`. See
/// [`poll_pipeline_capture`]'s own doc comment for why an indexed, dedicated slot per
/// wire slot is sound (not a free-pool search like this function had before v3).
/// `model`, when `Some((model_width, model_height, format))`, additionally builds (or
/// resizes) this slot's [`ModelScratch`] and records the scaled blit into the same
/// command buffer -- see that type's own doc comment. A scratch build/resize failure
/// is not fatal to the capture itself: this function still records and submits the
/// full-resolution capture exactly as it would with `model: None`, just without the
/// scaled proxy this one frame (the caller's own fallback, described on
/// [`poll_or_submit_capture`], sends the full-resolution proxy instead).
#[allow(clippy::too_many_arguments)]
fn submit_pipeline_capture(
    pipeline: &mut CapturePipeline,
    slot: usize,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    image: vk::Image,
    initial_layout: vk::ImageLayout,
    width: u32,
    height: u32,
    proxy_format: u32,
    model: Option<(u32, u32, vk::Format)>,
    encode_push: crate::composition::encode_pass::EncodePush,
) -> bool {
    // Split borrow: the slot is taken mutably while `encode` is read immutably, and
    // they are disjoint fields of the same struct.
    let CapturePipeline { slots, encode, .. } = pipeline;
    let encode = encode.as_ref();
    let slot = &mut slots[slot];
    if slot.pending.is_some() {
        return false;
    }
    let model_dims = if let Some((model_width, model_height, format)) = model {
        let needs_rebuild = slot.model.as_ref().is_none_or(|m| m.width != model_width || m.height != model_height);
        if needs_rebuild {
            if let Some(old) = slot.model.take() {
                // SAFETY: `slot.pending` is `None` here (checked above) -- this slot's
                // previous submission, if any, already had its fence confirmed
                // signaled by `poll_pipeline_capture` before `pending` was cleared, so
                // nothing submitted against the old scratch can still be in flight.
                unsafe { old.destroy(device, encode) };
            }
            slot.model = build_model_scratch(device, instance, physical_device, model_width, model_height, format, encode);
        }
        slot.model.as_ref().map(|m| (m.image, m.buffer, model_width, model_height))
    } else {
        None
    };
    // The encode runs only when this submission actually has a scratch with a set on
    // it; `encode_dispatch` is `None` otherwise and the proxy crosses unencoded.
    let encode_dispatch = slot
        .model
        .as_ref()
        .filter(|_| model_dims.is_some())
        .and_then(|m| m.encode_set.map(|set| (set, encode.expect("a set only exists when the pass does"))));
    if !record_capture_commands(
        device,
        slot.buf.cmd,
        image,
        initial_layout,
        slot.buf.buffer,
        None,
        width,
        height,
        model_dims,
        encode_dispatch.map(|(set, pass)| (pass, set, encode_push)),
    ) {
        return false;
    }
    // SAFETY: `slot.buf.fence` is `pending: None` here -- either never used yet
    // (starts signaled, see `build_capture_buffer`) or its previous signal was
    // already confirmed and consumed by `poll_pipeline_capture` -- so resetting it
    // now cannot race an in-flight wait on it.
    if unsafe { device.reset_fences(&[slot.buf.fence]) }.is_err() {
        return false;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&slot.buf.cmd)).build();
    // SAFETY: `slot.buf.cmd` was just recorded and ended by `record_capture_commands`
    // above. Deliberately not waited on here -- the entire point of this function:
    // the caller's present call returns immediately, and a later
    // `poll_pipeline_capture` call picks up the result once this fence actually
    // signals, exactly like `vkQueuePresentKHR` itself never waits on the work it
    // submits either.
    if crate::note_vk(unsafe { device.queue_submit(queue, &[submit], slot.buf.fence) }).is_err() {
        return false;
    }
    slot.pending = Some((width, height, proxy_format, model_dims.map(|(_, _, w, h)| (w, h)), std::time::Instant::now()));
    true
}

/// Waits for every capture the layer has submitted and not yet seen complete (the
/// pipeline's and the direct captures' pending slots). Only the layer's own fences: a
/// fence wait needs no queue synchronization, so this is legal from any hook on any
/// thread, unlike `vkDeviceWaitIdle`. Bounded by [`crate::FENCE_WAIT_TIMEOUT`]; `false`
/// if a wait failed or timed out. The slots stay pending, so the next poll consumes them
/// as usual.
pub fn wait_in_flight(pipeline: Option<&CapturePipeline>, direct: &[Option<DirectCapture>; 2], device: &ash::Device) -> bool {
    let mut fences: Vec<vk::Fence> = Vec::new();
    if let Some(p) = pipeline {
        fences.extend(p.slots.iter().filter(|s| s.pending.is_some()).map(|s| s.buf.fence));
    }
    fences.extend(direct.iter().flatten().filter(|d| d.pending.is_some()).map(|d| d.buf.fence));
    if fences.is_empty() {
        return true;
    }
    // SAFETY: every fence is the layer's own and was submitted (a slot is only `pending`
    // after a successful submit), so the wait cannot hang on a never-submitted fence.
    let wait = unsafe { device.wait_for_fences(&fences, true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    crate::note_fence_wait(wait, "capture::wait_in_flight").is_ok()
}

/// One full-resolution capture of `image` (in `layout`) through the real capture pipeline,
/// waited on: its bytes. For tests outside this module that need to check what the capture
/// reads from an image.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn test_capture_once(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue: vk::Queue, queue_family: u32,
    image: vk::Image, layout: vk::ImageLayout, width: u32, height: u32,
) -> Option<Vec<u8>> {
    let bytes = u64::from(width) * u64::from(height) * 4;
    let mut pipeline = None;
    if !ensure_pipeline(&mut pipeline, device, instance, physical_device, queue_family, bytes) {
        return None;
    }
    let p = pipeline.as_mut()?;
    let push = crate::composition::encode_pass::EncodePush { white_point: 1.0, bgr_order: 0, reversible_mode: 0 };
    let submitted = submit_pipeline_capture(p, 0, device, instance, physical_device, queue, image, layout, width, height, neural_forge_protocol::enums::proxy_format::RGBA8, None, push);
    // SAFETY: `queue` is the test's own; waiting for it idle completes the capture.
    unsafe { device.queue_wait_idle(queue) }.ok()?;
    let (mut out, mut model) = (Vec::new(), Vec::new());
    let (full, _) = poll_pipeline_capture(p, 0, device, &mut out, &mut model);
    // SAFETY: the queue is idle, so nothing references the pipeline any more.
    unsafe { destroy_pipeline(pipeline, device) };
    (submitted && full.is_some()).then_some(out)
}

/// # Safety
/// Must only be called at device-destruction time, with no submitted work referencing
/// these handles still in flight -- same contract as [`destroy`].
pub unsafe fn destroy_pipeline(pipeline: Option<CapturePipeline>, device: &ash::Device) {
    if let Some(p) = pipeline {
        for slot in &p.slots {
            // SAFETY: forwarded from this function's own contract.
            unsafe {
                slot.buf.destroy(device);
                if let Some(model) = &slot.model {
                    model.destroy(device, p.encode.as_ref());
                }
            }
        }
        // After every set allocated from it was freed by the loop above.
        // SAFETY: forwarded from this function's own contract.
        if let Some(pass) = &p.encode {
            unsafe { pass.destroy(device) };
        }
    }
}

/// Whether, and by how much, this device's driver requires host pointers imported via
/// `VK_EXT_external_memory_host` to be aligned (`VkPhysicalDeviceExternalMemoryHostPropertiesEXT::minImportedHostPointerAlignment`).
/// `None` if the query itself fails -- treated as "don't attempt import", the same
/// fail-open discipline as every other capability check in this module.
pub(crate) fn min_imported_host_pointer_alignment(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<vk::DeviceSize> {
    let mut ext_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut ext_props);
    // SAFETY: `physical_device` belongs to `instance`; `props2` is a freshly built,
    // valid out-parameter with the EXT struct chained into its `pNext`.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    (ext_props.min_imported_host_pointer_alignment > 0).then_some(ext_props.min_imported_host_pointer_alignment)
}

/// The Phase 3 zero-copy capture path (`docs/EXTERNAL_MEMORY_HOST_DESIGN.md`): a single
/// [`CaptureBuffer`] whose device memory is *imported* directly from the live SHM
/// proxy region (`ShmClient::proxy_region`), so `vkCmdCopyImageToBuffer` writes
/// straight into shared memory -- no staging buffer, no CPU copy on the way there.
///
/// Deliberately one slot, not two like [`CapturePipeline`]: the imported memory *is*
/// the one shared proxy region every caller of `ShmClient::write_proxy` ultimately
/// writes to. Two of these submitted concurrently would be two unsynchronized GPU
/// writes to the exact same destination bytes -- a real write-write hazard, not just
/// wasted work -- so there is no second slot to hide capture latency behind here. The
/// tradeoff for a direct write is capping this at one in-flight capture at a time;
/// [`run`] only ever uses one of [`DirectCapture`] or [`CapturePipeline`] for a given
/// device, never both, precisely so nothing else can also be writing to the same
/// region through the other path.
pub struct DirectCapture {
    buf: CaptureBuffer,
    /// Same meaning as `PipelineSlot::pending`, for this capture's own single slot, plus the
    /// zero-copy capture target the submission also copied the frame into, if any.
    pending: Option<(u32, u32, u32, Option<vk::Buffer>, std::time::Instant)>,
}

/// Builds (or rebuilds, on a capacity/queue-family change) the one slot
/// [`DirectCapture`] needs, importing `host_ptr`/`capacity` (the live SHM proxy
/// region) rather than allocating fresh device memory. `false` on any failure --
/// every caller already treats that as "fall back to `CapturePipeline`", never a
/// reason to stop trying on a later frame.
///
/// # Safety
/// `host_ptr` must be valid for `capacity` bytes for as long as `existing` holds
/// `Some` afterward, and nothing outside the resulting `DirectCapture`'s own command
/// buffer may write to those bytes while a capture against them is in flight.
unsafe fn ensure_direct_capture(
    existing: &mut Option<DirectCapture>,
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    host_ptr: *mut u8,
    capacity: vk::DeviceSize,
) -> bool {
    #[cfg(test)]
    if TEST_FAIL_DIRECT_SETUP.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    if let Some(d) = existing.as_ref() {
        if d.buf.capacity >= capacity && d.buf.ptr == host_ptr {
            return true;
        }
        // A resize/pointer change (the mapping is only ever established once per
        // process in practice, but this mirrors `ensure_pipeline`'s own rebuild
        // discipline rather than assuming that) needs the same drain-before-destroy
        // care: never touch a slot that might still be in flight.
        if d.pending.is_some() {
            // SAFETY: `d.buf.fence` is this slot's own fence; waiting for it before
            // any destroy below touches the resources it guards is exactly what makes
            // that destroy sound. Bounded, like every other rebuild wait in this
            // module -- rare, not latency sensitive, but not worth risking forever
            // over: see `crate::FENCE_WAIT_TIMEOUT`.
            let wait = unsafe { device.wait_for_fences(&[d.buf.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
            if crate::note_fence_wait(wait, "capture::ensure_direct_capture rebuild drain").is_err() {
                return false;
            }
        }
        let d = existing.take().expect("checked above");
        // SAFETY: the fence was just confirmed signaled above (or was never pending).
        unsafe { d.buf.destroy(device) };
    }
    // SAFETY: forwarded from this function's own contract.
    let Some(buf) = (unsafe { build_imported_capture_buffer(device, instance, physical_device, queue_family, host_ptr, capacity) }) else {
        return false;
    };
    *existing = Some(DirectCapture { buf, pending: None });
    true
}

/// Non-blocking, mirrors [`poll_pipeline_capture`] -- except there is nothing to copy
/// out: a signaled fence here means the bytes are already sitting in the SHM proxy
/// region this slot's memory was imported from. Returns the completed submission's own
/// `(width, height, proxy_format, capture_target, submitted_at)`, or `None` if nothing is signaled yet (or
/// the fence reported a real error, left `pending` forever rather than guessed safe to reuse).
fn poll_direct_capture(direct: &mut DirectCapture, device: &ash::Device) -> Option<(u32, u32, u32, Option<vk::Buffer>, std::time::Instant)> {
    let dims = direct.pending?;
    // SAFETY: `direct.buf.fence` belongs to this slot; a status query never touches
    // command-buffer/buffer/memory state, so it's sound regardless of whether the
    // submission this fence guards has actually completed yet.
    match crate::note_vk(unsafe { device.get_fence_status(direct.buf.fence) }) {
        Ok(true) => {
            direct.pending = None;
            Some(dims)
        }
        Ok(false) => None,
        Err(_) => None, // real device error -- leave `pending`; never guess reuse is safe
    }
}

/// Non-blocking, mirrors [`submit_pipeline_capture`] -- records and submits a new
/// capture into the one slot, if it's free (`pending: None`). `false` if it's still
/// pending or recording/submission itself failed. `original_dst`, when `Some`, is the
/// zero-copy capture target the same submission also copies the frame into (see
/// [`record_capture_commands`]).
#[allow(clippy::too_many_arguments)]
fn submit_direct_capture(
    direct: &mut DirectCapture,
    device: &ash::Device,
    queue: vk::Queue,
    image: vk::Image,
    initial_layout: vk::ImageLayout,
    width: u32,
    height: u32,
    proxy_format: u32,
    original_dst: Option<vk::Buffer>,
) -> bool {
    if direct.pending.is_some() {
        return false;
    }
    // `working_scale` is not wired into the dma-buf path -- see `poll_or_submit_capture`'s
    // own doc comment on why.
    if !record_capture_commands(device, direct.buf.cmd, image, initial_layout, direct.buf.buffer, original_dst, width, height, None, None) {
        return false;
    }
    // SAFETY: `direct.buf.fence` is `pending: None` here -- either never used yet
    // (starts signaled) or its previous signal was already confirmed and consumed by
    // `poll_direct_capture` -- so resetting it now cannot race an in-flight wait.
    if unsafe { device.reset_fences(&[direct.buf.fence]) }.is_err() {
        return false;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&direct.buf.cmd)).build();
    // SAFETY: `direct.buf.cmd` was just recorded and ended above. Deliberately not
    // waited on -- same reasoning as `submit_pipeline_capture`.
    if crate::note_vk(unsafe { device.queue_submit(queue, &[submit], direct.buf.fence) }).is_err() {
        return false;
    }
    direct.pending = Some((width, height, proxy_format, original_dst, std::time::Instant::now()));
    true
}

/// # Safety
/// Must only be called at device-destruction time, with no submitted work referencing
/// these handles still in flight -- same contract as [`destroy`]. Never unmaps or
/// otherwise touches the imported host pointer itself -- that memory belongs to
/// `ShmClient`, not this buffer, exactly like every other Vulkan-object-only destroy
/// in this module.
pub unsafe fn destroy_direct_capture(direct: Option<DirectCapture>, device: &ash::Device) {
    if let Some(d) = direct {
        // SAFETY: forwarded from this function's own contract.
        unsafe { d.buf.destroy(device) };
    }
}

/// Builds a [`CaptureBuffer`] whose device memory is *imported* from `host_ptr`
/// (`VK_EXT_external_memory_host`) rather than freshly allocated -- `ptr` in the
/// result is `host_ptr` itself, not a separate `vkMapMemory` mapping, so a capture
/// submitted against it writes straight into whatever `host_ptr` already points at.
/// `None` on any failure, exactly like [`build_capture_buffer`].
///
/// # Safety
/// `host_ptr` must be valid for `bytes` bytes and already aligned/sized to whatever
/// `min_imported_host_pointer_alignment` the caller queried -- this function does not
/// re-check either, only the driver does (at `vkAllocateMemory`, where a violation is
/// a validation error, not proactively caught here).
unsafe fn build_imported_capture_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    host_ptr: *mut u8,
    bytes: vk::DeviceSize,
) -> Option<CaptureBuffer> {
    let pool_info =
        vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: `device` is the live device this capture serves; `pool_info` is valid.
    let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else { return None };
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `pool` was just created above.
    let cmd = match unsafe { crate::loader_data::allocate_commands(device, &alloc_info) } {
        Ok(bufs) => bufs[0],
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet.
            unsafe { device.destroy_command_pool(pool, None) };
            return None;
        }
    };
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    // SAFETY: starting signaled means the first use's own poll never reports pending
    // for a fence nothing has submitted work against yet.
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(f) => f,
        Err(_) => {
            // SAFETY: `pool` owns no other resources yet; freeing it also frees `cmd`.
            unsafe { device.destroy_command_pool(pool, None) };
            return None;
        }
    };

    // SAFETY: forwarded from this function's own contract.
    let Some((buffer, memory)) = (unsafe { import_host_buffer(device, instance, physical_device, host_ptr, bytes) }) else {
        // SAFETY: neither `fence` nor `pool` owns any other resource yet.
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return None;
    };

    // No `vkMapMemory` here, unlike `build_capture_buffer`: `host_ptr` already *is*
    // the address this imported memory refers to -- that is the entire point of a
    // host-pointer import, and re-mapping it would be redundant at best.
    Some(CaptureBuffer { pool, cmd, fence, buffer, memory, ptr: host_ptr, capacity: bytes })
}

/// Imports `bytes` of host memory at `host_ptr` (`VK_EXT_external_memory_host`) and binds it
/// to a fresh `TRANSFER_SRC|TRANSFER_DST` buffer -- the query/buffer/import sequence shared by
/// [`build_imported_capture_buffer`] (the proxy region, written by a capture) and
/// `composition::gpu::GpuCompose`'s zero-copy compose (the answer region, read by a compose).
/// `None` on any failure, leaving nothing allocated. The memory is never mapped: `host_ptr`
/// already is the CPU's view of it, and freeing the returned memory never unmaps it.
///
/// # Safety
/// `host_ptr` must be valid for `bytes` bytes for as long as the returned memory lives, and
/// already aligned/sized to `min_imported_host_pointer_alignment` -- this function does not
/// re-check either, only the driver does (at `vkAllocateMemory`, where a violation is a
/// validation error, not proactively caught here).
pub(crate) unsafe fn import_host_buffer(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    host_ptr: *mut u8,
    bytes: vk::DeviceSize,
) -> Option<(vk::Buffer, vk::DeviceMemory)> {
    // Imported host memory has its own compatibility query -- separate from (and not
    // necessarily the same memory-type set as) `build_capture_buffer`'s own
    // HOST_VISIBLE|HOST_COHERENT search for a fresh allocation. Resolved by hand
    // (never `vk::ExtExternalMemoryHostFn::load`, which *panics* if the function
    // doesn't resolve -- found live, on real hardware: `vkEnumerateDeviceExtensionProperties`
    // and device creation both reporting the extension present does not guarantee
    // `vkGetDeviceProcAddr` resolves every one of its functions in this layered
    // context, and this whole module's fail-open discipline requires that to be a
    // normal "don't import" outcome, not an abort).
    // SAFETY: `device` is live; `name` is a valid, NUL-terminated C string.
    let get_memory_host_pointer_properties_ext = unsafe {
        instance.get_device_proc_addr(device.handle(), c"vkGetMemoryHostPointerPropertiesEXT".as_ptr())
    };
    let get_memory_host_pointer_properties_ext = get_memory_host_pointer_properties_ext?;
    // SAFETY: a non-null `vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT")`
    // result is guaranteed by the Vulkan spec to have this exact signature.
    let get_memory_host_pointer_properties_ext: vk::PFN_vkGetMemoryHostPointerPropertiesEXT =
        unsafe { std::mem::transmute(get_memory_host_pointer_properties_ext) };
    let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
    // SAFETY: `device` is live; `host_ptr` is valid for `bytes` bytes per this
    // function's own contract.
    let query_result = unsafe {
        get_memory_host_pointer_properties_ext(
            device.handle(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            host_ptr.cast(),
            &mut host_props,
        )
    };
    if query_result != vk::Result::SUCCESS {
        return None;
    }

    // A buffer that will be bound to *imported* memory must declare that up front:
    // VUID-vkBindBufferMemory-memory-02985 requires the external handle type used at
    // import time to already be set in the buffer's own `VkExternalMemoryBufferCreateInfo`
    // at creation -- found live, on real hardware, via `VK_LAYER_VALIDATE_SYNC=1`
    // (see docs/EXTERNAL_MEMORY_HOST_DESIGN.md), not caught by the local software ICD this
    // crate's tests otherwise run against.
    let mut external_info = vk::ExternalMemoryBufferCreateInfo::builder().handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let buf_info = vk::BufferCreateInfo::builder()
        .size(bytes)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_info);
    // SAFETY: `buf_info` is valid.
    let buffer = unsafe { device.create_buffer(&buf_info, None) }.ok()?;
    // SAFETY: `buffer` was just created and is not yet bound to memory.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    // SAFETY: `physical_device` is the device this capture serves; `instance` is its
    // owning instance.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    // Only a memory type both the buffer itself (`reqs`) and the imported pointer
    // (`host_props`) agree on is actually usable here.
    let compatible = reqs.memory_type_bits & host_props.memory_type_bits;
    let Some(type_index) = (0..mem_props.memory_type_count)
        .find(|&i| (compatible & (1 << i)) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted))
    else {
        // SAFETY: `buffer` has no memory bound yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };

    let mut import_info =
        vk::ImportMemoryHostPointerInfoEXT::builder().handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT).host_pointer(host_ptr.cast());
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(bytes).memory_type_index(type_index).push_next(&mut import_info);
    // SAFETY: `alloc` is valid; `type_index` was just confirmed to satisfy both
    // `reqs` and `host_props` above; `host_ptr`/`bytes` satisfy this function's own
    // safety contract on alignment/size.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: same reasoning as the branch above.
            unsafe { device.destroy_buffer(buffer, None) };
            return None;
        }
    };
    // SAFETY: `buffer`/`memory` were each just created above, sized/typed to satisfy
    // each other by construction.
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        // SAFETY: `memory` is not yet bound to anything that would make freeing it
        // unsound; `buffer` has no memory bound.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return None;
    }

    Some((buffer, memory))
}

/// Writes `bytes` (exactly `width*height*4` `RGBA8` bytes) into `image` via
/// `r`'s own staging buffer -- the CPU-composited last resort when no GPU compose
/// path is available at all. Fully synchronous; `image` assumed/left `PRESENT_SRC_KHR`,
/// same contract [`run_sync`]'s own stage 2 relies on. Best-effort: does nothing
/// observable on failure beyond leaving `image` unpresented-to this frame, same
/// fail-open discipline as every other stage in this module.
fn write_bytes_to_image(device: &ash::Device, r: &CaptureResources, queue: vk::Queue, image: vk::Image, width: u32, height: u32, bytes: &[u8]) {
    let frame_bytes = u64::from(width) * u64::from(height) * 4;
    if bytes.len() as u64 != frame_bytes {
        return;
    }
    // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes -- the same
    // invariant `run_sync`'s own stage 1/2 already rely on.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), r.ptr, bytes.len()) };
    // SAFETY: `r.cmd` was allocated with `RESET_COMMAND_BUFFER`.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return;
    }
    let to_dst = barrier(
        image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    unsafe {
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
    }
    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D { width, height, depth: 1 })
        .build();
    unsafe {
        device.cmd_copy_buffer_to_image(r.cmd, r.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
    }
    let to_present = barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
    unsafe {
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return;
    }
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    if crate::note_vk(unsafe { device.queue_submit(queue, &[submit], r.fence) }).is_err() {
        return;
    }
    let wait = unsafe { device.wait_for_fences(&[r.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    let _ = crate::note_fence_wait(wait, "capture::write_bytes_to_image");
}

/// Captures `image` into the proxy region, runs the shared-memory round trip, and
/// copies a result back into `image` before the caller's own present call. `resources`
/// is the per-device slot `queue_present_khr` owns (lazily built/rebuilt here).
///
/// Fails open on any error: returns without having touched `image` at all (still in
/// whatever layout the caller found it in, `PRESENT_SRC_KHR`) if anything along the way
/// doesn't work, so the caller can always fall back to presenting unmodified.
///
/// Always returns `None`: `image` is fully written and back in `PRESENT_SRC_KHR` when
/// this returns, safe to present with no extra wait. (The return type matches [`run`],
/// whose asynchronous compose does hand back a semaphore.)
///
/// # Safety
/// `queue` must be the same queue `image`'s presentation was requested on, with no
/// concurrent use of it from another thread for the duration of this call (the same
/// external-synchronization requirement `vkQueuePresentKHR` itself already places on
/// its own `queue` argument, which is what makes submitting here, from inside the
/// present hook, sound without any additional locking).
/// The old, fully-synchronous, one-frame-at-a-time path: capture *this* frame, block
/// on the helper round trip for *this* frame's own answer (up to a real timeout
/// budget), composite, write back -- all within the same present call. Kept
/// unchanged and still used for the two cases that genuinely need same-frame
/// correctness: a pending `capture_request` (its dump must show *this* frame's real
/// before/after, not some other frame's) and any non-zero `debug_view` (the
/// compare/split views are meaningless if original and answer come from different
/// moments). See [`run`]'s own doc comment for why every other case no longer goes
/// through here.
#[allow(clippy::too_many_arguments)]
unsafe fn run_sync(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    image: vk::Image,
    width: u32,
    height: u32,
    proxy_format: u32,
    bgr_order: bool,
    resources: &mut Option<CaptureResources>,
    gpu_compose: &mut Option<crate::composition::gpu::GpuCompose>,
    shm: &mut ShmClient,
    original_scratch: &mut Vec<u8>,
    _last_answer: &mut Vec<u8>,
) -> Option<vk::Semaphore> {
    let bytes_per_pixel = neural_forge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > crate::shm::region_capacity() {
        return None;
    }
    if !ensure(resources, device, instance, physical_device, queue_family, frame_bytes) {
        return None;
    }
    let r = resources.as_ref().expect("just ensured above");

    // Stage 1: image -> staging buffer.
    // SAFETY: `r.cmd` was allocated from `r.pool`, created with
    // `RESET_COMMAND_BUFFER`; resetting before every `begin_command_buffer` is exactly
    // what that flag exists to allow.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return None;
    }
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `r.cmd` was just reset above.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return None;
    }
    let to_transfer_src = barrier(
        image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_READ,
    );
    // SAFETY: `r.cmd` is in the recording state; `image` is the caller's own,
    // currently-`PRESENT_SRC_KHR` swapchain image per `vkQueuePresentKHR`'s contract.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_src],
        );
    }
    let copy_out = vk::BufferImageCopy::builder()
        .buffer_offset(0)
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
        .build();
    // SAFETY: `image` was just transitioned to `TRANSFER_SRC_OPTIMAL` above; `r.buffer`
    // was sized to at least `frame_bytes` by `ensure`.
    unsafe {
        device.cmd_copy_image_to_buffer(r.cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, r.buffer, &[copy_out]);
    }
    // Stage 1 hands the image back in PRESENT_SRC_KHR, the layout it found it in, so
    // every return between here and stage 2 (a failed wait, a failed stage-2 recording
    // or submit) still leaves it presentable. Stage 2 takes it from there itself.
    let back_to_present = barrier(
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::empty(),
    );
    // SAFETY: same reasoning as the first barrier above.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[back_to_present],
        );
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return None;
    }
    // SAFETY: `r.fence` starts signaled (see `ensure`) or was reset+waited-on by the
    // previous call to this function; `queue` is the caller's, externally synchronized
    // for the duration of this call per this function's own safety contract.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return None;
    }
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    let t_stage1_start = std::time::Instant::now();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if crate::note_vk(unsafe { device.queue_submit(queue, &[submit], r.fence) }).is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above. Bounded, not truly
    // unbounded: see `crate::FENCE_WAIT_TIMEOUT`.
    let wait = unsafe { device.wait_for_fences(&[r.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    if crate::note_fence_wait(wait, "capture::run_sync stage1").is_err() {
        return None;
    }
    let t_stage1 = t_stage1_start.elapsed();

    // CPU side: the captured bytes are now in `r.ptr` (host-coherent, no explicit
    // flush/invalidate needed). Hand them to the helper, then, if it actually
    // answered, overwrite `r.ptr` in place with that answer -- stage 2 below copies
    // whatever is sitting in `r.ptr` back into `image`, so this is what makes the
    // helper's answer (a real NGX evaluation, or the helper's own proxy-echo fallback
    // when the model isn't ready -- `neural_forge_helper::main`'s per-frame loop guarantees
    // the answer region is always the same size/format as the proxy either way)
    // actually reach the screen. A helper that never answers (not running, or the
    // round trip timed out) leaves `r.ptr` untouched -- it still holds the bytes
    // stage 1 just captured, so stage 2 below presents those unmodified, same fail-open
    // behavior as every other error path in this function.
    // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes (the memory
    // type/size `ensure` just built or confirmed already satisfies this call's own
    // `frame_bytes`).
    // Composition (below) needs the pre-edit frame after `read_answer` has already
    // overwritten `r.ptr` in place, so it has to be copied out now, before that
    // happens -- one extra `frame_bytes`-sized allocation/copy per frame, on top of
    // the two Vulkan transfers this function already does; not yet worth avoiding
    // ahead of proving the composition path correct at all.
    //
    // `captured` (a shared view of `r.ptr`) lives only inside this block, so it is
    // provably dead before the `&mut` view of the same memory is created for
    // `answer_dst` further down -- the two never coexist.
    const SLOT: usize = 0;
    let t_snapshot_start = std::time::Instant::now();
    let (t_snapshot, t_write_proxy) = {
        // SAFETY: `r.ptr` is a live mapping of at least `frame_bytes` bytes (see above).
        let captured = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
        original_scratch.clear();
        original_scratch.extend_from_slice(captured);
        let t_snapshot = t_snapshot_start.elapsed();
        let t_write_proxy_start = std::time::Instant::now();
        // `run_sync` is the synchronous debug/one-shot path (debug_view, a one-shot
        // capture_request dump) -- deliberately always slot 0, paired with
        // `try_round_trip`'s own blocking wait, never protocol v3's second slot.
        shm.set_frame_info(SLOT, width, height, proxy_format);
        shm.write_proxy(SLOT, captured);
        (t_snapshot, t_write_proxy_start.elapsed())
    };
    let original: &[u8] = original_scratch.as_slice();
    let t_roundtrip_start = std::time::Instant::now();
    let answered = shm.try_round_trip();
    let t_roundtrip = t_roundtrip_start.elapsed();
    let t_compose_start = std::time::Instant::now();
    if answered {
        // SAFETY: same reasoning as the read above; `ShmClient::read_answer` never
        // writes past the slice's length, which is exactly `frame_bytes` here.
        let answer_dst = unsafe { std::slice::from_raw_parts_mut(r.ptr, frame_bytes as usize) };
        shm.read_answer(SLOT, answer_dst);
        // Only `RGBA8` is handled -- `RGBA16F` still passes the helper's raw answer
        // through untouched (see `composition::apply`'s own doc comment for why, and
        // `neural_forge_protocol::enums::proxy_format` for the format codes).
        if neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format) {
            if let Some(settings) = shm.composition_settings() {
                if settings.apply_model && settings.neural_enabled {
                    // GPU dispatch (`composition::gpu`) only implements the normal
                    // composited case (`compose.comp` has no concept of `debug_view`
                    // at all) -- fails open to the CPU reference
                    // (`composition::apply::apply_rgba8`, which every mode already
                    // handles) whenever the GPU path isn't applicable, isn't
                    // available, or fails, same fail-open discipline as every other
                    // stage in this function. Always the synchronous, CPU-visible
                    // dispatch: this path exists for the capture dump, which needs the
                    // composited bytes in `r.ptr`.
                    let mut composed_sync = false;
                    if settings.debug_view == 0 {
                        if gpu_compose.is_none() {
                            *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
                        }
                        if let Some(gpu) = gpu_compose {
                            composed_sync = gpu.dispatch(
                                device,
                                instance,
                                physical_device,
                                queue,
                                width,
                                height,
                                original,
                                answer_dst,
                                settings.colour_strength,
                                settings.transfer_strength,
                                settings.max_ratio,
                                bgr_order,
                            );
                        }
                    }
                    if !composed_sync {
                        crate::composition::apply::apply_rgba8(
                            original,
                            answer_dst,
                            settings.colour_strength,
                            settings.transfer_strength,
                            settings.max_ratio,
                            settings.debug_view,
                            bgr_order,
                        );
                    }
                } else {
                    // "Off keeps the whole pass running... and simply presents the
                    // clean frame" -- ShmHeader::apply_model's own doc comment.
                    answer_dst.copy_from_slice(original);
                }
            }
        }
    }
    crate::log!("[capture] {}x{} {} bytes -> proxy; round trip answered={}", width, height, frame_bytes, answered);

    // Real `ShmHeader::capture_request` support: dump this frame's original and
    // final (post-composition, if any ran above) bytes to disk. Checked regardless of
    // `answered`/`proxy_format` so a request during a fail-open frame still produces a
    // (identical) matched pair rather than silently doing nothing -- `write_pair`
    // itself is the only place that would need to special-case a format it can't
    // encode, and today it always gets `RGBA8` bytes either way.
    if shm.take_capture_request() && neural_forge_protocol::enums::proxy_format::is_8bit(proxy_format) {
        // SAFETY: same reasoning as every other read of `r.ptr` in this function --
        // still a live mapping of at least `frame_bytes` bytes, and stage 2 below
        // hasn't started overwriting it yet.
        let current = unsafe { std::slice::from_raw_parts(r.ptr, frame_bytes as usize) };
        crate::dump::write_pair(&original, current, width, height, bgr_order);
    }

    // Stage 2: staging buffer (now holding the answer, if there was one -- otherwise
    // still the captured bytes) -> image.
    // SAFETY: `r.cmd` was ended above; the pool it came from allows re-recording.
    if unsafe { device.reset_command_buffer(r.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
        return None;
    }
    // SAFETY: `r.cmd` was just reset.
    if unsafe { device.begin_command_buffer(r.cmd, &begin_info) }.is_err() {
        return None;
    }
    let copy_in = copy_out;
    let to_transfer_dst = barrier(
        image,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
    );
    // SAFETY: `image` is `PRESENT_SRC_KHR` again after stage 1's final barrier (waited on
    // above); `r.buffer` (same host-coherent memory as `r.ptr`, which the CPU-side block
    // above may have just overwritten with the answer) holds exactly `frame_bytes` valid
    // bytes either way, matching `copy_in`'s own extent.
    unsafe {
        device.cmd_pipeline_barrier(r.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_transfer_dst]);
        device.cmd_copy_buffer_to_image(r.cmd, r.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy_in]);
    }
    let to_present = barrier(
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::empty(),
    );
    // SAFETY: restores the layout `vkQueuePresentKHR` requires before the caller's own
    // (real) present call runs right after this function returns.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
    }
    if unsafe { device.end_command_buffer(r.cmd) }.is_err() {
        return None;
    }
    // SAFETY: same reasoning as stage 1's own fence reset/submit/wait.
    if unsafe { device.reset_fences(&[r.fence]) }.is_err() {
        return None;
    }
    let submit2 = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&r.cmd)).build();
    let t_stage2_start = std::time::Instant::now();
    // SAFETY: `r.cmd` was just recorded and ended above.
    if crate::note_vk(unsafe { device.queue_submit(queue, &[submit2], r.fence) }).is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above. Waiting here (rather than
    // deferring to the next frame) keeps `image` fully write-back-complete and back in
    // `PRESENT_SRC_KHR` before this function returns, which is what the caller's own
    // immediately-following real present call requires.
    let wait = unsafe { device.wait_for_fences(&[r.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    if crate::note_fence_wait(wait, "capture::run_sync stage2").is_err() {
        return None;
    }
    let t_stage2 = t_stage2_start.elapsed();
    crate::log!(
        "[capture] timing stage1={:?} snapshot={:?} write_proxy={:?} roundtrip={:?} compose={:?} stage2={:?} total={:?}",
        t_stage1,
        t_snapshot,
        t_write_proxy,
        t_roundtrip,
        // `t_compose_start` was captured right after the round trip; `t_stage2_start`
        // right before stage 2's own submit -- the gap between them is exactly the
        // composition work (CPU reference or GPU dispatch), with no double-counting
        // against `t_stage2` below.
        t_stage2_start.duration_since(t_compose_start),
        t_stage2,
        t_stage1_start.elapsed(),
    );
    shm.publish_frame_timing(t_stage1_start.elapsed(), answered);

    None
}

/// # Safety
/// Must only be called at device-destruction time, with no submitted work referencing
/// these handles still in flight.
pub unsafe fn destroy(resources: Option<CaptureResources>, device: &ash::Device) {
    if let Some(r) = resources {
        // SAFETY: forwarded from this function's own contract.
        unsafe { r.destroy(device) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EXTERNAL_MEMORY_HOST_EXTENSION;
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Same shape as `composition::gpu::tests::test_device` -- a real (if software)
    /// Vulkan device via whatever loader/ICD is on this machine, `None` if there
    /// isn't one. Not shared with that module (private to it, and this crate has no
    /// shared test-support module yet); small enough that duplicating it costs less
    /// than inventing one.
    fn test_device() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32)> {
        // SAFETY: loads the system Vulkan loader; the usual caveats of loading an
        // arbitrary shared library apply and are accepted here the same way every
        // other `ash` consumer in this crate already does.
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
        // SAFETY: `device_create_info` is valid; every physical device has a family 0.
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
        // SAFETY: `device`/family/index 0 match what `device_create_info` just requested.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        Some((entry, instance, physical_device, device, queue, queue_family))
    }

    /// A standalone image standing in for a real swapchain image, already in
    /// `PRESENT_SRC_KHR` -- what `run`'s own contract requires of `image` on entry,
    /// same as any image `vkQueuePresentKHR`'s own precondition hasn't been violated
    /// on.
    fn make_present_src_image(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, width: u32, height: u32) -> (vk::Image, vk::DeviceMemory) {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None) }.expect("failed to create the test's own target image");
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let type_index = (0..mem_props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .expect("no suitable memory type for the test's own target image");
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_present = barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::empty(), vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
        (image, memory)
    }

    fn scratch_path(tag: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        format!("{}/neural-forge-capture-test-{}-{tag}-{n}/shm.bin", std::env::temp_dir().display(), std::process::id())
    }

    /// Same shape as `test_device`, except the device is created with
    /// `VK_EXT_external_memory_host` enabled when (and only when) the physical device
    /// actually advertises it -- `None` if there's no Vulkan loader/ICD at all, or a
    /// separate flag saying whether the extension actually ended up enabled, so
    /// `direct_capture_writes_straight_into_imported_host_memory` can skip itself
    /// cleanly on a machine (this dev sandbox's software ICD, most likely) that
    /// doesn't support it, rather than fail for a reason that has nothing to do with
    /// this crate's own code.
    fn test_device_with_external_memory_host() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32, bool)> {
        // SAFETY: same reasoning as `test_device`.
        let entry = unsafe { ash::Entry::load() }.ok()?;
        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;
        let physical_device = *unsafe { instance.enumerate_physical_devices() }.ok()?.first()?;
        let supported = unsafe { instance.enumerate_device_extension_properties(physical_device) }.is_ok_and(|extensions| {
            extensions.iter().any(|extension| {
                let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                name == EXTERNAL_MEMORY_HOST_EXTENSION
            })
        });
        let queue_family = 0;
        let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(queue_family).queue_priorities(&[1.0]).build()];
        let extension_names = [EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr()];
        let mut device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
        if supported {
            device_create_info = device_create_info.enabled_extension_names(&extension_names);
        }
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        Some((entry, instance, physical_device, device, queue, queue_family, supported))
    }

    /// Like `make_present_src_image`, but also fills the image with a known solid
    /// color before transitioning it to `PRESENT_SRC_KHR` -- `make_present_src_image`
    /// itself leaves its image's contents undefined, fine for tests that only check
    /// *that* a copy happened, not *what* it copied. This test needs the latter: the
    /// whole point is confirming the imported-memory capture lands the *right* bytes,
    /// not just *some* bytes.
    fn make_filled_present_src_image(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, width: u32, height: u32, color: [f32; 4]) -> (vk::Image, vk::DeviceMemory) {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None) }.expect("failed to create the test's own target image");
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let type_index = (0..mem_props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .expect("no suitable memory type for the test's own target image");
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_image_memory(image, memory, 0) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin_info).unwrap();
            let to_dst = barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            device.cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: color }, &[subresource()]);
            let to_present = barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
        (image, memory)
    }

    /// The actual point of `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`, exercised end to end
    /// against a real (if software) device: a capture submitted against a
    /// `DirectCapture` slot must land its bytes directly in the imported host
    /// pointer -- not a staging buffer, not something that merely runs without
    /// crashing, but the *exact* pixels the source image held. Skips itself (not a
    /// failure) if this machine's Vulkan device doesn't advertise
    /// `VK_EXT_external_memory_host` at all.
    #[test]
    fn direct_capture_writes_straight_into_imported_host_memory() {
        // Held so `TEST_FAIL_DIRECT_SETUP` (set by another test) is never seen here.
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some((_entry, instance, physical_device, device, queue, queue_family, supported)) = test_device_with_external_memory_host() else {
            eprintln!("direct_capture_writes_straight_into_imported_host_memory: no Vulkan loader/ICD, skipping");
            return;
        };
        if !supported {
            eprintln!("direct_capture_writes_straight_into_imported_host_memory: VK_EXT_external_memory_host not supported here, skipping");
            unsafe { device.destroy_device(None); instance.destroy_instance(None); }
            return;
        }
        let Some(alignment) = min_imported_host_pointer_alignment(&instance, physical_device) else {
            eprintln!("direct_capture_writes_straight_into_imported_host_memory: alignment query failed, skipping");
            unsafe { device.destroy_device(None); instance.destroy_instance(None); }
            return;
        };

        let (width, height) = (8u32, 8u32);
        let frame_bytes = (width * height * 4) as usize;
        // A real `mmap` (page-aligned, so a multiple of any real driver's alignment
        // requirement -- NVIDIA's own is 4096) standing in for the SHM proxy region
        // this path is actually meant to import; rounded up to `alignment` for
        // drivers that want more than a page.
        let region_len = frame_bytes.max(alignment as usize).div_ceil(alignment as usize) * alignment as usize;
        // SAFETY: a plain anonymous mapping, valid for the rest of this test; never
        // shared with another process, matching every other precondition
        // `ensure_direct_capture`'s own safety contract asks for.
        let host_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), region_len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        assert_ne!(host_ptr, libc::MAP_FAILED, "mmap for the test's own host region failed");
        let host_ptr = host_ptr.cast::<u8>();
        assert_eq!(host_ptr as usize % alignment as usize, 0, "mmap must hand back at least page-aligned memory");

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        // An arbitrary, checkable, non-zero pattern -- chosen so every channel lands
        // on an *exact* u8 value (100/150/200/255), not a half-integer boundary like
        // 0.5*255=127.5, where real drivers can reasonably round either way and a
        // fixed `.round()` in this test would be guessing which.
        let color = [100.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0, 1.0];
        let (image, image_memory) = make_filled_present_src_image(&device, &mem_props, queue, pool, width, height, color);

        let mut direct: Option<DirectCapture> = None;
        // SAFETY: `host_ptr` is valid for `region_len` bytes for the rest of this
        // test; nothing else writes to it. `region_len`, not `frame_bytes`, per
        // VUID-VkMemoryAllocateInfo-allocationSize-01745: the allocation itself must
        // be a multiple of `alignment` (4096 on this real NVIDIA driver) even though
        // the actual pixel copy below only ever touches the first `frame_bytes`.
        assert!(
            unsafe { ensure_direct_capture(&mut direct, &device, &instance, physical_device, queue_family, host_ptr, region_len as vk::DeviceSize) },
            "ensure_direct_capture should succeed with a supported, correctly aligned host pointer"
        );
        let d = direct.as_mut().unwrap();
        assert!(submit_direct_capture(d, &device, queue, image, vk::ImageLayout::PRESENT_SRC_KHR, width, height, neural_forge_protocol::enums::proxy_format::RGBA8, None));
        // A real wait (not `poll_direct_capture`'s own non-blocking check) is correct
        // here: this test cares whether the capture is *correct*, not whether `run`'s
        // own present-hook discipline of never blocking holds -- that's
        // `run_never_blocks_on_a_slow_helper_and_eventually_composites`'s job, not
        // this test's.
        crate::note_vk(unsafe { device.wait_for_fences(&[d.buf.fence], true, u64::MAX) }).expect("capture fence wait failed");
        assert_eq!(poll_direct_capture(d, &device).map(|(w, h, f, t, _)| (w, h, f, t)), Some((width, height, neural_forge_protocol::enums::proxy_format::RGBA8, None)));

        // SAFETY: the fence wait above confirms the GPU's writes to `host_ptr` are
        // complete and visible to the CPU (host-coherent memory).
        let captured = unsafe { std::slice::from_raw_parts(host_ptr, frame_bytes) };
        // `.round()`, not a bare `as u8` truncation: the real float->UNORM8 conversion
        // `vkCmdClearColorImage`/the copy actually perform rounds to nearest (found
        // live: 0.25 * 255 = 63.75, which truncation would wrongly expect as 63
        // against the real, correct 64).
        let expected: [u8; 4] = [(color[0] * 255.0).round() as u8, (color[1] * 255.0).round() as u8, (color[2] * 255.0).round() as u8, (color[3] * 255.0).round() as u8];
        for pixel in captured.chunks_exact(4) {
            assert_eq!(pixel, expected, "every captured pixel must match the source image's own fill color, read straight out of the imported host pointer");
        }

        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy_direct_capture(direct, &device);
            device.destroy_device(None);
            instance.destroy_instance(None);
            libc::munmap(host_ptr.cast(), region_len);
        }
    }

    /// Synchronous present (the default): with a helper that answers within the budget, the
    /// first present call captures, waits for that frame's own answer and composites it.
    #[test]
    fn synchronous_present_composites_the_same_frame_it_captured() {
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("run_never_blocks_on_a_slow_helper_and_eventually_composites: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("sync");

        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // A live helper (matters for `poll_async_request`'s timeout budget: the long
        // "steady state" one, not the short "nobody's listening" one, since this test
        // deliberately answers slower than that short budget).
        unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) }.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that only answers `HELPER_DELAY` after it sees a new request --
        // long enough that if `run` ever blocked waiting for it, a handful of calls
        // spaced much closer together than that would visibly take just as long.
        const HELPER_DELAY: Duration = Duration::from_millis(30);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = 0u32;
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                hdr.heartbeat.fetch_add(1, AtomicOrdering::Relaxed);
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    std::thread::sleep(HELPER_DELAY);
                    hdr.answered_w.store(hdr.width.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    hdr.answered_h.store(hdr.height.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let (width, height) = (8u32, 8u32);
        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut model_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut raw_answer_base = Vec::new();
        let mut raw_answer_generation = 0u64;
        let mut last_answer = Vec::new();
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;

        let mut got_semaphore = false;
        let mut iteration = 0u32;
        let mut slow_calls: Vec<Duration> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        // Real usage calls this once per present, indefinitely -- loop until either a
        // real composited result shows up or the deadline (comfortably several
        // `HELPER_DELAY`-long round trips) is exhausted, not a fixed iteration count.
        while Instant::now() < deadline {
            iteration += 1;
            let hdr_now = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let req_before = hdr_now.seq_req.load(AtomicOrdering::Relaxed);
            let call_start = Instant::now();
            // SAFETY: `image` is this test's own, currently `PRESENT_SRC_KHR`; `queue`
            // is used from this one thread only, exactly like `run`'s own contract
            // requires of the real present hook.
            let sem = unsafe {
                run(
                    &device,
                    &instance,
                    physical_device,
                    queue,
                    queue_family,
                    image,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    image,
                    width,
                    height,
                    proxy_format,
                    false,
                    &mut resources,
                    &mut pipeline,
                    &mut direct,
                    // This test's own device never enables `VK_EXT_external_memory_host`
                    // (see `test_device`'s minimal `DeviceCreateInfo`), so this must be
                    // `false` -- exercising `CapturePipeline`, the path this test
                    // actually validates. A `DirectCapture` equivalent needs its own
                    // test with the extension genuinely enabled, not this one lying
                    // about it.
                    &mut false,
                    &mut gpu_compose,
                    &mut shm,
                    &mut original_scratch,
                    &mut model_scratch,
                    &mut inflight,
                    &mut bootstrap_complete,
                    &mut answer_scratch,
                    &mut raw_answer_base,
                    &mut raw_answer_generation,
                    &mut last_answer,
                    &mut last_answer_dims,
                )
            };
            let call_time = call_start.elapsed();
            // Skip the very first call: it pays real one-time setup cost this test
            // doesn't otherwise isolate (`CapturePipeline`/`GpuCompose` first-use
            // allocation, first-touch driver/shader-cache warmup).
            //
            // The real invariant this guards is "`run()` never synchronously waits
            // for the helper's own answer". A single call occasionally running long
            // is not, by itself, evidence of that: GitHub's shared runners saw calls
            // up to 466ms (nearly *double* `HELPER_DELAY`) with no code change
            // involved, purely from real scheduler/software-rasterizer contention on
            // that infra -- worse than this test's own fake helper thread's sleep, so
            // no fixed per-call ceiling can both reject that noise and still allow a
            // real hardware/local run through. What a genuine synchronous-wait
            // regression looks like instead is *every* call converging near
            // `HELPER_DELAY`, not one noisy outlier -- so tally slow calls across the
            // whole loop and judge the pattern, not any single sample, below.
            if iteration > 1 && call_time >= HELPER_DELAY * 9 / 10 {
                slow_calls.push(call_time);
            }
            if let Some(sem) = sem {
                // The present that composited is the one that asked, and it left nothing in
                // flight: the answer it composited can only be for the frame it captured. (The
                // pipelined mode always leaves the next request outstanding.)
                let req_after = hdr_now.seq_req.load(AtomicOrdering::Relaxed);
                assert!(req_after > req_before, "the compositing present must itself have sent the request");
                assert_eq!(hdr_now.seq_resp.load(AtomicOrdering::Relaxed), req_after, "no request may be left in flight");
                got_semaphore = true;
                // Stand in for what the real present call does: wait on the semaphore
                // before the image is considered final, exactly like
                // `composition::gpu::tests`' own async tests already establish.
                let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
                let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
                unsafe {
                    device.queue_submit(queue, &[submit], wait_fence).unwrap();
                    device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                    device.destroy_fence(wait_fence, None);
                }
                break;
            }
            if !last_answer.is_empty() {
                got_semaphore = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(got_semaphore, "synchronous present must composite the answer");
        let _ = iteration;

        // Model interval 2: the next present carries that answer instead of asking again -- no new
        // request, but still a composited frame -- and the one after asks again.
        let hdr_now = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
        hdr_now.model_interval.store(2, AtomicOrdering::Relaxed);
        let mut asked = Vec::new();
        for _ in 0..4 {
            let before = hdr_now.seq_req.load(AtomicOrdering::Relaxed);
            let sem = unsafe {
                run(
                    &device, &instance, physical_device, queue, queue_family, image, vk::ImageLayout::PRESENT_SRC_KHR, image,
                    width, height, proxy_format, false, &mut resources, &mut pipeline, &mut direct, &mut false, &mut gpu_compose,
                    &mut shm, &mut original_scratch, &mut model_scratch, &mut inflight, &mut bootstrap_complete, &mut answer_scratch,
                    &mut raw_answer_base, &mut raw_answer_generation, &mut last_answer, &mut last_answer_dims,
                )
            };
            assert!(sem.is_some(), "every present is composited, carried or not");
            let sem = sem.unwrap();
            let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
            let stage = vk::PipelineStageFlags::ALL_COMMANDS;
            unsafe {
                device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&stage)).build()], wait_fence).unwrap();
                device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                device.destroy_fence(wait_fence, None);
            }
            asked.push(hdr_now.seq_req.load(AtomicOrdering::Relaxed) != before);
        }
        assert_eq!(asked.iter().filter(|&&a| a).count(), 2, "every other present asks the model: {asked:?}");
        assert!(asked.windows(2).all(|w| w[0] != w[1]), "asking and carrying alternate: {asked:?}");
        hdr_now.model_interval.store(1, AtomicOrdering::Relaxed);

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();

        // A capture pipeline slot can legitimately still be `pending` here (the test
        // loop can exit as soon as `last_answer` is non-empty, with no guarantee the
        // *next* speculative capture submission already resolved) -- wait for the
        // whole device idle first, the same real teardown precondition
        // `destroy_private_resources` relies on in production, before either
        // `destroy` call below touches anything.
        unsafe { device.device_wait_idle() }.unwrap();
        // SAFETY: every semaphore this test waited on has a completed, waited-for
        // fence behind it (the explicit wait above); the idle wait just above
        // confirms every capture-pipeline slot's own fence too; nothing else touched
        // `image`.
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
    /// Copies `bytes` into `image` (any layout, contents discarded) and leaves it `PRESENT_SRC_KHR`.
    fn upload_present_src(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image, width: u32, height: u32, bytes: &[u8]) {
        let (buffer, memory, ptr) = host_buffer(device, mem_props, bytes.len() as u64);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        one_shot(device, queue, pool, |cmd| unsafe {
            let to_dst = barrier(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            device.cmd_copy_buffer_to_image(cmd, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[full_region(width, height)]);
            let to_present = barrier(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
        });
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
    }

    /// Reads a `PRESENT_SRC_KHR` image back (leaving it `PRESENT_SRC_KHR`).
    fn read_back_present_src(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, queue: vk::Queue, pool: vk::CommandPool, image: vk::Image, width: u32, height: u32) -> Vec<u8> {
        let bytes = u64::from(width * height * 4);
        let (buffer, memory, ptr) = host_buffer(device, mem_props, bytes);
        one_shot(device, queue, pool, |cmd| unsafe {
            let to_src = barrier(image, vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_src]);
            device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[full_region(width, height)]);
            let back = barrier(image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::empty());
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[], &[], &[back]);
        });
        let out = unsafe { std::slice::from_raw_parts(ptr, bytes as usize) }.to_vec();
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        out
    }

    fn full_region(width: u32, height: u32) -> vk::BufferImageCopy {
        vk::BufferImageCopy::builder()
            .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build())
            .image_extent(vk::Extent3D { width, height, depth: 1 })
            .build()
    }

    fn host_buffer(device: &ash::Device, mem_props: &vk::PhysicalDeviceMemoryProperties, bytes: u64) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
        let info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = unsafe { device.create_buffer(&info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = (0..mem_props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted)).unwrap();
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }.unwrap();
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let ptr = unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.unwrap().cast::<u8>();
        (buffer, memory, ptr)
    }

    fn one_shot(device: &ash::Device, queue: vk::Queue, pool: vk::CommandPool, record: impl FnOnce(vk::CommandBuffer)) {
        let alloc_info = vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        unsafe { device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)) }.unwrap();
        record(cmd);
        unsafe {
            device.end_command_buffer(cmd).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build()], fence).unwrap();
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }
    }

    /// Removes a [`scratch_path`]'s directory when dropped, even when the test fails.
    struct RemoveScratch(String);
    impl Drop for RemoveScratch {
        fn drop(&mut self) {
            if let Some(dir) = std::path::Path::new(&self.0).parent() {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    fn test_game_frame(width: u32, height: u32) -> Vec<u8> {
        (0..width * height).flat_map(|i| { let t = (i * 37 % 200) as u8; [t + 30, 220 - t, t / 2 + 40, 255] }).collect()
    }

    /// One present, from the test's side: what `run` returned, and the image after the returned
    /// semaphore (if any) was waited on.
    struct Presented {
        composed: bool,
        asked: bool,
        /// The present left no CPU copy of the frame or the answer behind: the zero-copy path.
        cpu_pair_empty: bool,
        image: Vec<u8>,
    }

    /// Drives the synchronous present with direct capture (`VK_EXT_external_memory_host` enabled
    /// on a real device) against a fake helper whose answer is the proxy with RGB inverted, from
    /// the same starting frame every time: `warm-up` presents until the first composite, then
    /// `presents` more with `model_interval`. `zero_copy: false` runs the identical present with
    /// the CPU copies, the reference. `scribble` overwrites the answer and proxy regions before
    /// every carried present, standing in for a late answer landing, or anything else that
    /// writes the shared regions between model runs. `hold` turns frame hold on. `None` when the
    /// device has no `VK_EXT_external_memory_host`.
    ///
    /// `path` is the shm file, shared by every sequence of one test: imported host memory may be
    /// backed with real pages that are held until the process exits (measured: about 250 MB of
    /// tmpfs per file on Intel ANV when the whole proxy region was imported), so each test uses
    /// one file and removes it.
    fn direct_present_sequence(path: &str, zero_copy: bool, model_interval: u32, presents: usize, scribble: bool, hold: bool) -> Option<Vec<Presented>> {
        let (_entry, instance, physical_device, device, queue, queue_family, supported) = test_device_with_external_memory_host()?;
        if !supported {
            unsafe { device.destroy_device(None); instance.destroy_instance(None); }
            return None;
        }
        TEST_NO_ZERO_COPY.store(!zero_copy, AtomicOrdering::Relaxed);
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                TEST_NO_ZERO_COPY.store(false, AtomicOrdering::Relaxed);
                *HELD.lock().unwrap_or_else(|e| e.into_inner()) = None;
            }
        }
        let _reset = Reset;

        let (width, height) = (16u32, 16u32);
        let frame_bytes = (width * height * 4) as usize;
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(path));
        let hdr_ptr = shm.test_header_ptr();
        let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
        // An earlier sequence on the same file may have left these set.
        hdr.model_interval.store(1, AtomicOrdering::Relaxed);
        hdr.seq_resp.store(hdr.seq_req.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);
        // The measured white point would depend on the meter's process-wide phase.
        hdr.white_point_source.store(neural_forge_protocol::enums::white_point_source::MANUAL, AtomicOrdering::Relaxed);
        hdr.hold_frame.store(u32::from(hold), AtomicOrdering::Relaxed);
        let proxy_region = shm.proxy_region(0).unwrap().0 as usize;
        let answer_region = shm.answer_region(0).unwrap().0 as usize;

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the sequence returns).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = hdr.seq_req.load(AtomicOrdering::Relaxed);
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                hdr.heartbeat.fetch_add(1, AtomicOrdering::Relaxed);
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    std::sync::atomic::fence(AtomicOrdering::Acquire);
                    // SAFETY: both regions are at least `frame_bytes` long, and the layer neither
                    // reads the answer nor writes the proxy while a request is outstanding.
                    let proxy = unsafe { std::slice::from_raw_parts(proxy_region as *const u8, frame_bytes) };
                    let answer = unsafe { std::slice::from_raw_parts_mut(answer_region as *mut u8, frame_bytes) };
                    for (a, p) in answer.chunks_exact_mut(4).zip(proxy.chunks_exact(4)) {
                        a.copy_from_slice(&[255 - p[0], 255 - p[1], 255 - p[2], p[3]]);
                    }
                    hdr.helper_eval_ms_bits.store(1.5f32.to_bits(), AtomicOrdering::Relaxed);
                    hdr.answered_w.store(hdr.width.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    hdr.answered_h.store(hdr.height.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    std::sync::atomic::fence(AtomicOrdering::Release);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(500));
            }
        });

        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);
        let game_frame = test_game_frame(width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut model_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut raw_answer_base = Vec::new();
        let mut raw_answer_generation = 0u64;
        let mut last_answer = Vec::new();
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;

        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while out.len() < presents + 1 && Instant::now() < deadline {
            if out.len() == 1 {
                hdr.model_interval.store(model_interval, AtomicOrdering::Relaxed);
            }
            let carrying = !out.is_empty() && model_interval > 1 && inflight[0].presents % u64::from(model_interval) != 0;
            if scribble && carrying {
                // SAFETY: no request is outstanding (every present so far resolved its own).
                unsafe {
                    std::ptr::write_bytes(answer_region as *mut u8, 0x11, frame_bytes);
                    std::ptr::write_bytes(proxy_region as *mut u8, 0xee, frame_bytes);
                }
            }
            // The game draws its frame (the same one every time, so a carried answer still applies
            // in full rather than being faded out as motion).
            upload_present_src(&device, &mem_props, queue, pool, image, width, height, &game_frame);
            let before = hdr.seq_req.load(AtomicOrdering::Relaxed);
            // SAFETY: `image` is this test's own, currently `PRESENT_SRC_KHR`; one thread.
            let sem = unsafe {
                run(
                    &device, &instance, physical_device, queue, queue_family, image, vk::ImageLayout::PRESENT_SRC_KHR, image,
                    width, height, proxy_format, false, &mut resources, &mut pipeline, &mut direct, &mut true, &mut gpu_compose,
                    &mut shm, &mut original_scratch, &mut model_scratch, &mut inflight, &mut bootstrap_complete, &mut answer_scratch,
                    &mut raw_answer_base, &mut raw_answer_generation, &mut last_answer, &mut last_answer_dims,
                )
            };
            let Some(sem) = sem else {
                assert!(out.is_empty(), "every present after the first composite composes");
                std::thread::sleep(Duration::from_millis(2));
                continue;
            };
            let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
            let stage = vk::PipelineStageFlags::ALL_COMMANDS;
            unsafe {
                device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&stage)).build()], wait_fence).unwrap();
                device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                device.destroy_fence(wait_fence, None);
            }
            assert!(direct[0].is_some() && pipeline.is_none(), "the sequence must run on direct capture");
            out.push(Presented {
                composed: true,
                asked: hdr.seq_req.load(AtomicOrdering::Relaxed) != before,
                cpu_pair_empty: last_answer.is_empty() && original_scratch.is_empty(),
                image: read_back_present_src(&device, &mem_props, queue, pool, image, width, height),
            });
        }
        assert_eq!(out.len(), presents + 1, "the sequence ran out of time");
        let alignment = min_imported_host_pointer_alignment(&instance, physical_device).unwrap();
        assert_eq!(
            direct[0].as_ref().map(|d| d.buf.capacity),
            Some((frame_bytes as u64).div_ceil(alignment) * alignment),
            "the direct capture imports the frame's own bytes, not the whole region"
        );
        if hold {
            assert!(HELD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|h| h.original.len() == frame_bytes), "frame hold holds the frame on the CPU");
            assert!(!last_answer.is_empty() && !raw_answer_base.is_empty(), "frame hold composes the CPU pair");
            assert!(!gpu_compose.as_ref().unwrap().holds_generation(raw_answer_generation), "frame hold never takes the zero-copy path");
        }

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
        Some(out)
    }

    /// The synchronous present's zero-copy compose (direct capture, answer region imported,
    /// no CPU copy of the frame or the answer) presents byte-for-byte what the same present
    /// with the CPU copies does -- every model frame, and with model interval 2 every carried
    /// frame too, even when the answer and proxy regions are overwritten between model runs
    /// (a late answer landing during carried frames must not reach the picture).
    #[test]
    fn synchronous_present_zero_copy_matches_cpu_path() {
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = scratch_path("zero-copy");
        let _cleanup = RemoveScratch(path.clone());
        let Some(cpu) = direct_present_sequence(&path, false, 1, 3, false, false) else {
            eprintln!("synchronous_present_zero_copy_matches_cpu_path: no Vulkan device with VK_EXT_external_memory_host, skipping");
            return;
        };
        let zc = direct_present_sequence(&path, true, 1, 3, false, false).unwrap();
        assert!(cpu.iter().all(|p| p.composed && p.asked && !p.cpu_pair_empty), "the reference keeps the CPU copies");
        assert!(zc.iter().all(|p| p.composed && p.asked && p.cpu_pair_empty), "every present takes the zero-copy path");
        for (i, (c, z)) in cpu.iter().zip(&zc).enumerate() {
            assert_eq!(z.image, c.image, "present {i}: zero-copy output differs from the CPU path");
        }
        assert!(cpu.iter().all(|p| p.image != test_game_frame(16, 16)), "the answer must actually change the picture, or this proves nothing");

        let cpu = direct_present_sequence(&path, false, 2, 6, false, false).unwrap();
        let zc = direct_present_sequence(&path, true, 2, 6, true, false).unwrap();
        let asked: Vec<bool> = zc.iter().map(|p| p.asked).collect();
        assert_eq!(asked, cpu.iter().map(|p| p.asked).collect::<Vec<_>>(), "both runs ask and carry on the same presents");
        assert!(asked.iter().filter(|&&a| !a).count() >= 2, "model interval 2 must carry: {asked:?}");
        assert!(zc.iter().all(|p| p.cpu_pair_empty), "carried presents stay zero-copy too");
        for (i, (c, z)) in cpu.iter().zip(&zc).enumerate() {
            assert_eq!(z.image, c.image, "present {i} (asked: {}): zero-copy output differs from the CPU path", c.asked);
        }
        assert!(cpu.iter().all(|p| p.image != test_game_frame(16, 16)), "a carried present must change the picture too, or it proves nothing about its answer");
    }

    /// Frame hold keeps the CPU path even when everything else would take the zero-copy one: the
    /// held frame lives on the CPU, and holding starts from the CPU copy of the capture.
    #[test]
    fn frame_hold_uses_the_cpu_path_where_zero_copy_is_available() {
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = scratch_path("zero-copy-hold");
        let _cleanup = RemoveScratch(path.clone());
        let Some(held) = direct_present_sequence(&path, true, 1, 2, false, true) else {
            eprintln!("frame_hold_uses_the_cpu_path_where_zero_copy_is_available: no Vulkan device with VK_EXT_external_memory_host, skipping");
            return;
        };
        assert!(held.iter().all(|p| p.composed && !p.cpu_pair_empty), "frame hold composes from the CPU copies");
    }

    /// The synchronous one-shot path (`run_sync`, used for a capture dump): captures, waits for
    /// the helper's answer, composes it into the image and leaves the image presentable.
    /// Stage 1 now hands the image back in PRESENT_SRC_KHR before stage 2 takes it again, so an
    /// early return between the two can no longer leave it in TRANSFER_DST_OPTIMAL.
    #[test]
    fn run_sync_composes_and_leaves_the_image_presentable() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("run_sync_composes_and_leaves_the_image_presentable: no Vulkan loader/ICD, skipping");
            return;
        };
        let path = scratch_path("run-sync");
        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path));
        let hdr_ptr = shm.test_header_ptr();
        let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);
        let (width, height) = (16u32, 16u32);
        let frame_bytes = (width * height * 4) as usize;
        let proxy_region = shm.proxy_region(0).unwrap().0 as usize;
        let answer_region = shm.answer_region(0).unwrap().0 as usize;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = hdr.seq_req.load(AtomicOrdering::Relaxed);
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                hdr.heartbeat.fetch_add(1, AtomicOrdering::Relaxed);
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    std::sync::atomic::fence(AtomicOrdering::Acquire);
                    // SAFETY: both regions hold at least `frame_bytes`; the layer waits for the answer.
                    let proxy = unsafe { std::slice::from_raw_parts(proxy_region as *const u8, frame_bytes) };
                    let answer = unsafe { std::slice::from_raw_parts_mut(answer_region as *mut u8, frame_bytes) };
                    for (a, p) in answer.chunks_exact_mut(4).zip(proxy.chunks_exact(4)) {
                        a.copy_from_slice(&[255 - p[0], 255 - p[1], 255 - p[2], p[3]]);
                    }
                    std::sync::atomic::fence(AtomicOrdering::Release);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(500));
            }
        });
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);
        let game_frame = test_game_frame(width, height);
        upload_present_src(&device, &mem_props, queue, pool, image, width, height, &game_frame);
        let mut resources: Option<CaptureResources> = None;
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let (mut original_scratch, mut last_answer) = (Vec::new(), Vec::new());
        // SAFETY: `image` is the test's own, in PRESENT_SRC_KHR; one thread uses `queue`.
        let sem = unsafe {
            run_sync(
                &device, &instance, physical_device, queue, queue_family, image, width, height,
                neural_forge_protocol::enums::proxy_format::RGBA8, false, &mut resources, &mut gpu_compose, &mut shm,
                &mut original_scratch, &mut last_answer,
            )
        };
        assert!(sem.is_none(), "the synchronous path leaves nothing to wait on");
        let presented = read_back_present_src(&device, &mem_props, queue, pool, image, width, height);
        assert_ne!(presented, game_frame, "the answer was composed into the image");

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A completed capture is only sent for the frame it was taken of: one at another size, or
    /// (for the synchronous present) one submitted before this present began, is dropped and a
    /// fresh capture goes in its place.
    #[test]
    fn a_completed_capture_of_another_size_or_an_earlier_frame_is_not_sent() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("a_completed_capture_of_another_size_or_an_earlier_frame_is_not_sent: no Vulkan loader/ICD, skipping");
            return;
        };
        let path = scratch_path("stale-capture");
        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path));
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let (width, height) = (16u32, 16u32);
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);
        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let frame_bytes = u64::from(width * height * 4);
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let (mut original, mut model) = (Vec::new(), Vec::new());
        let mut step = |w: u32, h: u32, after: Option<Instant>| {
            let step = poll_or_submit_capture(
                0, false, &mut false, &mut pipeline, &mut direct, &device, &instance, physical_device, queue, queue_family,
                image, vk::ImageLayout::PRESENT_SRC_KHR, w, h, proxy_format, frame_bytes, None,
                crate::composition::encode_pass::EncodePush { white_point: 1.0, bgr_order: 0, reversible_mode: 0 },
                &mut shm, &mut original, &mut model, None, after,
            );
            unsafe { device.queue_wait_idle(queue) }.unwrap();
            match step {
                CaptureStep::Captured(c) => Some(c.sent),
                CaptureStep::Pending => None,
                CaptureStep::Failed => panic!("capture setup failed on lavapipe"),
            }
        };
        assert_eq!(step(8, 8, None), None, "submits an 8x8 capture");
        assert_eq!(step(width, height, None), None, "the 8x8 capture is not sent for a 16x16 frame");
        assert_eq!(step(width, height, None), Some((width, height)), "the replacement is");
        assert_eq!(step(width, height, None), None, "submits again");
        let present_began = Instant::now();
        assert_eq!(step(width, height, Some(present_began)), None, "a capture from before this present is not sent");
        assert_eq!(step(width, height, Some(present_began)), Some((width, height)), "the fresh one is");
        unsafe {
            device.device_wait_idle().unwrap();
            destroy_pipeline(pipeline, &device);
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// A capture setup that fails (here: the driver refusing the zero-copy host import) must
    /// not hold the synchronous present for its whole budget: nothing is in flight, so the
    /// frame goes out untouched at once. The failed import is latched, and the next presents
    /// capture through the staging-buffer pipeline and composite.
    #[test]
    fn a_failed_capture_setup_presents_at_once_and_falls_back_to_the_pipeline() {
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some((_entry, instance, physical_device, device, queue, queue_family, supported)) = test_device_with_external_memory_host() else {
            eprintln!("a_failed_capture_setup_presents_at_once_and_falls_back_to_the_pipeline: no Vulkan loader/ICD, skipping");
            return;
        };
        if !supported {
            eprintln!("a_failed_capture_setup_presents_at_once_and_falls_back_to_the_pipeline: VK_EXT_external_memory_host not supported here, skipping");
            unsafe { device.destroy_device(None); instance.destroy_instance(None); }
            return;
        }
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                TEST_FAIL_DIRECT_SETUP.store(false, AtomicOrdering::Relaxed);
            }
        }
        let _reset = Reset;
        TEST_FAIL_DIRECT_SETUP.store(true, AtomicOrdering::Relaxed);

        let path = scratch_path("direct-setup-fails");
        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path));
        let hdr_ptr = shm.test_header_ptr();
        let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = 0u32;
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                hdr.heartbeat.fetch_add(1, AtomicOrdering::Relaxed);
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    hdr.answered_w.store(hdr.width.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    hdr.answered_h.store(hdr.height.load(AtomicOrdering::Relaxed), AtomicOrdering::Relaxed);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(500));
            }
        });

        let (width, height) = (16u32, 16u32);
        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);
        upload_present_src(&device, &mem_props, queue, pool, image, width, height, &test_game_frame(width, height));

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let (mut original_scratch, mut model_scratch, mut answer_scratch, mut raw_answer_base, mut last_answer) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut raw_answer_generation = 0u64;
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;
        let mut external_memory_host = true;

        let mut present = |external_memory_host: &mut bool| unsafe {
            run(
                &device, &instance, physical_device, queue, queue_family, image, vk::ImageLayout::PRESENT_SRC_KHR, image,
                width, height, proxy_format, false, &mut resources, &mut pipeline, &mut direct, external_memory_host, &mut gpu_compose,
                &mut shm, &mut original_scratch, &mut model_scratch, &mut inflight, &mut bootstrap_complete, &mut answer_scratch,
                &mut raw_answer_base, &mut raw_answer_generation, &mut last_answer, &mut last_answer_dims,
            )
        };
        // Presents before the fake helper's first heartbeat go out untouched without trying
        // to capture; the first present that tries is the one that meets the failed setup.
        let deadline = Instant::now() + Duration::from_secs(2);
        let (first, took) = loop {
            let started = Instant::now();
            let first = present(&mut external_memory_host);
            let took = started.elapsed();
            if !external_memory_host || Instant::now() >= deadline {
                break (first, took);
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(!external_memory_host, "a failed zero-copy import is latched off for the device");
        assert!(first.is_none(), "a present whose capture could not be set up goes out untouched");
        // That first attempt also paid one-time setup (the compose pipeline), which on a loaded
        // software rasterizer can itself take hundreds of milliseconds. A setup that keeps
        // failing on later presents pays nothing but the failed attempt: each must come back
        // well inside the budget rather than waiting it out.
        let _ = took;
        for _ in 0..3 {
            let started = Instant::now();
            assert!(present(&mut true).is_none());
            let took = started.elapsed();
            assert!(took < SYNC_BUDGET / 2, "a failed capture setup must not wait out the {SYNC_BUDGET:?} budget (took {took:?})");
        }

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut composed = None;
        while composed.is_none() && Instant::now() < deadline {
            composed = present(&mut external_memory_host);
        }
        let sem = composed.expect("the staging-buffer pipeline must take over and composite");
        let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
        let stage = vk::PipelineStageFlags::ALL_COMMANDS;
        unsafe {
            device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&stage)).build()], wait_fence).unwrap();
            device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
            device.destroy_fence(wait_fence, None);
        }
        assert!(pipeline.is_some() && direct.iter().all(Option::is_none), "the fallback is the staging-buffer pipeline");

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// The real point of the pipelined redesign, exercised end to end against a real
    /// (if software) Vulkan device: `run` must never block a present call waiting on
    /// the helper, even when the helper genuinely takes far longer than one frame to
    /// answer -- and once it does answer, the result must actually reach `image` via
    /// a real, verifiable composited write (not just "a semaphore came back").
    #[test]
    fn run_never_blocks_on_a_slow_helper_and_eventually_composites() {
        let _mode = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        TEST_PIPELINED.store(true, std::sync::atomic::Ordering::Relaxed);
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                TEST_PIPELINED.store(false, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let _reset = Reset;
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("run_never_blocks_on_a_slow_helper_and_eventually_composites: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("blocks");

        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // A live helper (matters for `poll_async_request`'s timeout budget: the long
        // "steady state" one, not the short "nobody's listening" one, since this test
        // deliberately answers slower than that short budget).
        unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) }.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that only answers `HELPER_DELAY` after it sees a new request --
        // long enough that if `run` ever blocked waiting for it, a handful of calls
        // spaced much closer together than that would visibly take just as long.
        const HELPER_DELAY: Duration = Duration::from_millis(250);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = 0u32;
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                let req = hdr.seq_req.load(AtomicOrdering::Relaxed);
                if req != 0 && req != last_seen {
                    last_seen = req;
                    std::thread::sleep(HELPER_DELAY);
                    hdr.seq_resp.store(req, AtomicOrdering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let (width, height) = (8u32, 8u32);
        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut model_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut raw_answer_base = Vec::new();
        let mut raw_answer_generation = 0u64;
        let mut last_answer = Vec::new();
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;

        let mut got_semaphore = false;
        let mut iteration = 0u32;
        let mut slow_calls: Vec<Duration> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        // Real usage calls this once per present, indefinitely -- loop until either a
        // real composited result shows up or the deadline (comfortably several
        // `HELPER_DELAY`-long round trips) is exhausted, not a fixed iteration count.
        while Instant::now() < deadline {
            iteration += 1;
            let call_start = Instant::now();
            // SAFETY: `image` is this test's own, currently `PRESENT_SRC_KHR`; `queue`
            // is used from this one thread only, exactly like `run`'s own contract
            // requires of the real present hook.
            let sem = unsafe {
                run(
                    &device,
                    &instance,
                    physical_device,
                    queue,
                    queue_family,
                    image,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    image,
                    width,
                    height,
                    proxy_format,
                    false,
                    &mut resources,
                    &mut pipeline,
                    &mut direct,
                    // This test's own device never enables `VK_EXT_external_memory_host`
                    // (see `test_device`'s minimal `DeviceCreateInfo`), so this must be
                    // `false` -- exercising `CapturePipeline`, the path this test
                    // actually validates. A `DirectCapture` equivalent needs its own
                    // test with the extension genuinely enabled, not this one lying
                    // about it.
                    &mut false,
                    &mut gpu_compose,
                    &mut shm,
                    &mut original_scratch,
                    &mut model_scratch,
                    &mut inflight,
                    &mut bootstrap_complete,
                    &mut answer_scratch,
                    &mut raw_answer_base,
                    &mut raw_answer_generation,
                    &mut last_answer,
                    &mut last_answer_dims,
                )
            };
            let call_time = call_start.elapsed();
            // Skip the very first call: it pays real one-time setup cost this test
            // doesn't otherwise isolate (`CapturePipeline`/`GpuCompose` first-use
            // allocation, first-touch driver/shader-cache warmup).
            //
            // The real invariant this guards is "`run()` never synchronously waits
            // for the helper's own answer". A single call occasionally running long
            // is not, by itself, evidence of that: GitHub's shared runners saw calls
            // up to 466ms (nearly *double* `HELPER_DELAY`) with no code change
            // involved, purely from real scheduler/software-rasterizer contention on
            // that infra -- worse than this test's own fake helper thread's sleep, so
            // no fixed per-call ceiling can both reject that noise and still allow a
            // real hardware/local run through. What a genuine synchronous-wait
            // regression looks like instead is *every* call converging near
            // `HELPER_DELAY`, not one noisy outlier -- so tally slow calls across the
            // whole loop and judge the pattern, not any single sample, below.
            if iteration > 1 && call_time >= HELPER_DELAY * 9 / 10 {
                slow_calls.push(call_time);
            }
            if let Some(sem) = sem {
                got_semaphore = true;
                // Stand in for what the real present call does: wait on the semaphore
                // before the image is considered final, exactly like
                // `composition::gpu::tests`' own async tests already establish.
                let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
                let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
                unsafe {
                    device.queue_submit(queue, &[submit], wait_fence).unwrap();
                    device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                    device.destroy_fence(wait_fence, None);
                }
                break;
            }
            if !last_answer.is_empty() {
                got_semaphore = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(got_semaphore, "the pipeline must eventually composite a real answer within 5s of real time, not just avoid blocking forever");
        // A real synchronous-wait-on-the-helper regression makes *every* call take
        // close to `HELPER_DELAY`; isolated scheduler noise makes at most a rare one.
        // `iteration` counts the exempted first call too, so this ratio is
        // deliberately conservative (slightly stricter than "out of every later
        // call") rather than needing a second counter just to be exact about it.
        assert!(
            slow_calls.len() * 10 < iteration as usize,
            "{}/{iteration} run() calls took close to the helper's own {HELPER_DELAY:?} answer \
             delay ({slow_calls:?}) -- an isolated slow call is real-world scheduler noise, but \
             this many looks like run() is actually waiting on the helper again",
            slow_calls.len()
        );

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();

        // A capture pipeline slot can legitimately still be `pending` here (the test
        // loop can exit as soon as `last_answer` is non-empty, with no guarantee the
        // *next* speculative capture submission already resolved) -- wait for the
        // whole device idle first, the same real teardown precondition
        // `destroy_private_resources` relies on in production, before either
        // `destroy` call below touches anything.
        unsafe { device.device_wait_idle() }.unwrap();
        // SAFETY: every semaphore this test waited on has a completed, waited-for
        // fence behind it (the explicit wait above); the idle wait just above
        // confirms every capture-pipeline slot's own fence too; nothing else touched
        // `image`.
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    #[test]
    fn working_scale_sends_a_genuinely_smaller_proxy_and_still_composites() {
        // Confirms `working_scale`'s GPU-blit mechanism end to end on a real (or
        // software) Vulkan device: the proxy that actually reaches the wire is at the
        // scaled resolution (not the swapchain's own), and the whole pipeline still
        // reaches a real composited answer without crashing, hanging, or leaking --
        // the objective checkpoint `docs/GHOSTING_PLAN.md` step 1 calls for before this
        // lands on real hardware.
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("working_scale_sends_a_genuinely_smaller_proxy_and_still_composites: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("working_scale");

        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // SAFETY: `hdr_ptr` is this test's own live mapping, same technique the
        // sibling test above already uses.
        let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
        hdr.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);
        // 0.5 -- comfortably clear of `scaled_dims`' own 64px floor at this test's
        // resolution, so the assertions below are testing the scale factor, not the
        // floor.
        hdr.working_scale_bits.store(0.5f32.to_bits(), AtomicOrdering::Relaxed);

        // A fake helper that watches BOTH slots (protocol v3) and, on seeing a new
        // request, immediately asserts the proxy it was actually sent is at the
        // scaled resolution -- not the swapchain's own -- before answering. Any
        // mismatch fails the test from inside the helper thread via `sent_wrong_size`
        // rather than silently accepting whatever arrived, which a "does it
        // eventually composite" check alone would not catch.
        let (width, height) = (256u32, 192u32);
        let (expected_model_w, expected_model_h) = scaled_dims(width, height, 0.5);
        assert_ne!((expected_model_w, expected_model_h), (width, height), "0.5 at 256x192 must actually produce a smaller proxy, or this test proves nothing");
        let sent_wrong_size = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let (stop_clone, wrong_clone) = (Arc::clone(&stop), Arc::clone(&sent_wrong_size));
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last_seen = [0u32; 2];
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                for slot in 0..2 {
                    let req = hdr.seq_req_slot(slot).load(AtomicOrdering::Relaxed);
                    if req != 0 && req != last_seen[slot] {
                        last_seen[slot] = req;
                        let (w, h) = (hdr.width_slot(slot).load(AtomicOrdering::Relaxed), hdr.height_slot(slot).load(AtomicOrdering::Relaxed));
                        if (w, h) != (expected_model_w, expected_model_h) {
                            wrong_clone.store(true, AtomicOrdering::Relaxed);
                        }
                        hdr.seq_resp_slot(slot).store(req, AtomicOrdering::Relaxed);
                    }
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("failed to create the test's own command pool");
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut model_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut raw_answer_base = Vec::new();
        let mut raw_answer_generation = 0u64;
        let mut last_answer = Vec::new();
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;

        let mut composited = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !composited {
            // SAFETY: `image` is this test's own, currently `PRESENT_SRC_KHR`; `queue`
            // is used from this one thread only, exactly like `run`'s own contract.
            let sem = unsafe {
                run(
                    &device, &instance, physical_device, queue, queue_family, image, vk::ImageLayout::PRESENT_SRC_KHR, image, width, height,
                    proxy_format, false, &mut resources, &mut pipeline, &mut direct, &mut false, &mut gpu_compose, &mut shm, &mut original_scratch,
                    &mut model_scratch, &mut inflight, &mut bootstrap_complete, &mut answer_scratch, &mut raw_answer_base, &mut raw_answer_generation,
                    &mut last_answer, &mut last_answer_dims,
                )
            };
            if let Some(sem) = sem {
                let wait_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }.unwrap();
                let wait_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                let submit = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&sem)).wait_dst_stage_mask(std::slice::from_ref(&wait_stage)).build();
                unsafe {
                    device.queue_submit(queue, &[submit], wait_fence).unwrap();
                    device.wait_for_fences(&[wait_fence], true, u64::MAX).unwrap();
                    device.destroy_fence(wait_fence, None);
                }
                composited = true;
            } else if !last_answer.is_empty() {
                composited = true;
            }
            assert!(!sent_wrong_size.load(AtomicOrdering::Relaxed), "the helper observed a proxy request at the wrong resolution -- working_scale did not shrink it");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(composited, "the scaled pipeline must eventually composite a real answer within 5s, same as the unscaled path already does");

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();

        unsafe { device.device_wait_idle() }.unwrap();
        // SAFETY: every semaphore this test waited on has a completed, waited-for
        // fence behind it; the idle wait just above confirms every capture-pipeline
        // slot's own fence (including any `ModelScratch` this run built) too.
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// Measures, at a real GTA resolution, how much GPU-queue time one present's worth
    /// of `run`'s injected work actually costs -- the number behind the fps collapse.
    ///
    /// Not an assertion test and not a CI test: it needs a real GPU (a software ICD
    /// would report meaningless numbers) and is a benchmark, so it only runs when
    /// `NEURAL_FORGE_BENCH` is set in the environment, and only prints. The established
    /// way to use it (see `docs/HARDWARE_VALIDATION.md`) is to build the release test binary,
    /// copy it to `lordnikon`, and run it there with `NEURAL_FORGE_BENCH=1
    /// <bin> capture_hot_path_cost_per_present --nocapture --exact`.
    ///
    /// Why this is the right metric: `run` submits its capture copy and its compose onto
    /// the *game's own present queue* (see `submit_pipeline_capture`/
    /// `present_temporal_delta_async`), so every present drags that work through the same
    /// queue timeline as the game's real rendering. Timing a `device_wait_idle` right
    /// after each `run` call drains exactly that injected work and nothing else (this
    /// harness submits no competing "game" workload), so the reported per-present drain
    /// time is a clean lower bound on what the layer steals from the game every frame.
    /// At the 8x8 size the correctness test above uses this is microseconds and invisible;
    /// at 2560x1440 it is the real cost the hot-path fix has to bring down.
    #[test]
    fn capture_hot_path_cost_per_present() {
        if !neural_forge_protocol::env::is_set("NEURAL_FORGE_BENCH") {
            eprintln!("capture_hot_path_cost_per_present: set NEURAL_FORGE_BENCH=1 to run this GPU benchmark, skipping");
            return;
        }
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("capture_hot_path_cost_per_present: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("bench");

        let _cleanup = RemoveScratch(path.clone());
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path));
        let hdr_ptr = shm.test_header_ptr();
        unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) }.helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that answers as fast as it can see the request -- the real
        // steady state, where the layer has a fresh answer nearly every present and so
        // submits capture+compose work on nearly every call. That is the worst case for
        // per-present cost, which is exactly what we want to measure.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            let hdr = unsafe { &*(hdr_ptr as *mut neural_forge_protocol::ShmHeader) };
            let mut last = [0u32; 2];
            while !stop_clone.load(AtomicOrdering::Relaxed) {
                for slot in 0..2 {
                    let req = hdr.seq_req_slot(slot).load(AtomicOrdering::Relaxed);
                    if req != 0 && req != last[slot] {
                        last[slot] = req;
                        hdr.seq_resp_slot(slot).store(req, AtomicOrdering::Relaxed);
                    }
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        });

        let (width, height) = (2560u32, 1440u32);
        let proxy_format = neural_forge_protocol::enums::proxy_format::BGRA8;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.expect("bench command pool");
        let (image, image_memory) = make_present_src_image(&device, &mem_props, queue, pool, width, height);

        let mut resources: Option<CaptureResources> = None;
        let mut pipeline: Option<CapturePipeline> = None;
        let mut direct: [Option<DirectCapture>; 2] = [None, None];
        let mut gpu_compose: Option<crate::composition::gpu::GpuCompose> = None;
        let mut original_scratch = Vec::new();
        let mut model_scratch = Vec::new();
        let mut answer_scratch = Vec::new();
        let mut raw_answer_base = Vec::new();
        let mut raw_answer_generation = 0u64;
        let mut last_answer = Vec::new();
        let mut last_answer_dims = (0u32, 0u32);
        let mut inflight: [Inflight; 2] = Default::default();
        let mut bootstrap_complete = false;

        let warmup = 30u32;
        let measured = 200u32;
        // Two separate costs the layer imposes per present, measured apart so the fix
        // targets the right one: `cpu` is the wall time of the `run` call itself (the
        // work done synchronously on the game's own present thread -- full-frame host
        // readback/`memcpy`s and command recording), `gpu` is the drain of the work
        // `run` submitted onto the game's queue. The game's real per-present budget
        // pays for both.
        let mut cpu: Vec<Duration> = Vec::with_capacity(measured as usize);
        let mut gpu: Vec<Duration> = Vec::with_capacity(measured as usize);
        let mut submitted = 0u32;
        for iteration in 0..(warmup + measured) {
            let c = Instant::now();
            // SAFETY: `image` is this harness's own, treated as `PRESENT_SRC_KHR`;
            // `queue` is used from this one thread only, exactly `run`'s contract.
            let sem = unsafe {
                run(&device, &instance, physical_device, queue, queue_family, image,
                    vk::ImageLayout::PRESENT_SRC_KHR, image, width, height, proxy_format, false,
                    &mut resources, &mut pipeline, &mut direct, &mut false, &mut gpu_compose, &mut shm,
                    &mut original_scratch, &mut model_scratch, &mut inflight, &mut bootstrap_complete, &mut answer_scratch,
                    &mut raw_answer_base, &mut raw_answer_generation, &mut last_answer, &mut last_answer_dims)
            };
            let cpu_cost = c.elapsed();
            if sem.is_some() {
                submitted += 1;
            }
            // Drain exactly the work `run` just submitted onto the game's queue.
            let t = Instant::now();
            unsafe { device.device_wait_idle() }.unwrap();
            let gpu_cost = t.elapsed();
            if iteration >= warmup {
                cpu.push(cpu_cost);
                gpu.push(gpu_cost);
            }
            // A touch of spacing so the fake helper reliably flips a fresh answer
            // between presents, matching the steady state rather than starving itself.
            std::thread::sleep(Duration::from_millis(1));
        }

        let report = |label: &str, v: &mut Vec<Duration>| {
            v.sort_unstable();
            let mean = v.iter().sum::<Duration>() / v.len() as u32;
            println!(
                "  {label}: mean={mean:?} p50={:?} p95={:?} max={:?}",
                v[v.len() / 2], v[v.len() * 95 / 100], *v.last().unwrap()
            );
        };
        println!("capture_hot_path_cost_per_present @ {width}x{height} ({} samples, {submitted} composited a fresh answer):", cpu.len());
        report("cpu (run() on present thread)", &mut cpu);
        report("gpu (queue drain after run())", &mut gpu);

        stop.store(true, AtomicOrdering::Relaxed);
        helper.join().unwrap();
        unsafe { device.device_wait_idle() }.unwrap();
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(image_memory, None);
            device.destroy_command_pool(pool, None);
            destroy(resources, &device);
            destroy_pipeline(pipeline, &device);
            for slot in direct {
                destroy_direct_capture(slot, &device);
            }
            if let Some(gpu) = gpu_compose {
                gpu.destroy(&device);
            }
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod model_size_tests {
    use super::*;

    #[test]
    fn scale_never_exceeds_one_and_bad_values_mean_one() {
        assert_eq!(effective_working_scale(1920, 1080, 1.5), 1.0);
        assert_eq!(effective_working_scale(1920, 1080, 0.0), 1.0);
        assert_eq!(effective_working_scale(1920, 1080, f32::NAN), 1.0);
        assert_eq!(effective_working_scale(1920, 1080, 0.5), 0.5);
    }

    #[test]
    fn any_resolution_is_brought_under_the_model_pixel_cap() {
        for (w, h) in [(1280, 720), (2560, 1440), (3840, 2160), (5760, 3240), (7680, 4320), (5120, 1440), (3440, 1440), (7680, 2160)] {
            for requested in [0.5f32, 0.75, 1.0, 1.5, 2.0] {
                let (mw, mh) = scaled_dims(w, h, effective_working_scale(w, h, requested));
                assert!(
                    u64::from(mw) * u64::from(mh) <= DEFAULT_MAX_MODEL_PIXELS + u64::from(mw) + u64::from(mh),
                    "{w}x{h} at {requested} gave a {mw}x{mh} model raster, over the cap"
                );
                assert!(mw <= w && mh <= h, "{w}x{h} at {requested}: model raster {mw}x{mh} is larger than the frame");
                assert!(mw >= 64 && mh >= 64 && mw % 2 == 0 && mh % 2 == 0);
            }
        }
    }

    fn frame_of(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        (0..w * h).flat_map(|i| { let v = f(i % w, i / w); [v, v, v, 255] }).collect()
    }

    #[test]
    fn the_white_meter_reads_the_scene_not_a_highlight_or_the_dark() {
        let (w, h) = (256u32, 128u32);
        // A dim scene: every tile peaks at sRGB 150 (linear ~0.305), with one specular dot at 255.
        let mut dim = frame_of(w, h, |_, _| 150);
        let o = ((10 * w + 10) * 4) as usize;
        dim[o..o + 3].copy_from_slice(&[255, 255, 255]);
        let white = meter_white(&dim, w, h, false).expect("a lit scene is measured");
        assert!((white - 0.305).abs() < 0.01, "the dim scene's white, not the highlight: {white}");
        // Nearly all black, one lit corner: not enough of the frame to say where white is.
        let dark = frame_of(w, h, |x, y| if x < 20 && y < 20 { 200 } else { 0 });
        assert_eq!(meter_white(&dark, w, h, false), None);
    }

    #[test]
    fn odd_frames_get_an_even_model_raster() {
        assert_eq!(scaled_dims(2493, 1408, effective_working_scale(2493, 1408, 1.0)), (2492, 1408));
        assert_eq!(scaled_dims(1365, 767, effective_working_scale(1365, 767, 1.0)), (1364, 766));
        for (w, h) in [(2493u32, 1408u32), (1365, 767), (3441, 1441)] {
            for scale in [0.5f32, 0.75, 1.0] {
                let (mw, mh) = scaled_dims(w, h, effective_working_scale(w, h, scale));
                assert!(mw % 2 == 0 && mh % 2 == 0, "{w}x{h} at {scale}: {mw}x{mh}");
                assert!(neural_forge_protocol::frame_dims_valid(mw, mh, neural_forge_protocol::enums::proxy_format::RGBA8));
            }
        }
    }

    #[test]
    fn a_frame_under_the_cap_is_left_alone() {
        assert_eq!(scaled_dims(2560, 1440, effective_working_scale(2560, 1440, 1.0)), (2560, 1440));
        assert_eq!(scaled_dims(3840, 2160, effective_working_scale(3840, 2160, 1.0)), (3840, 2160));
        // The 8K frame that would not build: brought down to 4K-equivalent.
        assert_eq!(scaled_dims(5760, 3240, effective_working_scale(5760, 3240, 1.5)), (3840, 2160));
    }
}
