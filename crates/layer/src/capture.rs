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
//! depends on the settings and what's available: the common case (real GPU compose,
//! no debug dump pending) is `composition::gpu::GpuCompose::dispatch_into_image_async`
//! (2026-09-10) -- non-blocking, its own doc comment covers why that's sound. Every
//! other case (CPU compose, a pending `capture_request`, `RGBA16F`, no GPU available)
//! still falls back to the original synchronous stage-2 write-back below, one more
//! command buffer + fence wait, same as this whole function used to always do.

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
fn scaled_dims(width: u32, height: u32, scale: f32) -> (u32, u32) {
    if !scale.is_finite() || scale <= 0.0 || (scale - 1.0).abs() < 0.01 {
        return (width, height);
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
    if !neuralforge_protocol::enums::proxy_format::is_8bit(proxy_format) {
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
/// out of the now-written proxy region), because `inflight.original`/`prepare_motion`
/// need bytes that survive whatever capture starts next and overwrites that region,
/// which the live proxy region itself can't provide once it's shared, imported memory.
#[allow(clippy::too_many_arguments)]
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
/// Returns `Some((sent_width, sent_height))` -- the proxy's *actual* resolution, which
/// the caller must record (`Inflight::proxy_dims`) to size the eventual answer
/// correctly -- on a successful capture+send this call, `None` on "nothing to do this
/// frame". This is `model`'s own request dims only when a scaled send genuinely
/// happened; every fallback above (an unavailable/failed scratch, `use_direct`, no
/// `model` requested at all) correctly reports the full-resolution `(width, height)`
/// instead, because it actually sent that -- callers must not re-derive this from
/// `model` themselves, only ever trust this return value.
#[allow(clippy::too_many_arguments)]
fn poll_or_submit_capture(
    slot: usize,
    use_direct: bool,
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
) -> Option<(u32, u32)> {
    if use_direct {
        let (host_ptr, capacity) = shm.proxy_region(slot)?;
        // SAFETY: `host_ptr`/`capacity` describe `shm`'s own live proxy region for
        // this slot, valid for as long as `shm` stays open (the life of this process,
        // since the mapping is never unmapped -- see
        // `neuralforge_protocol::mapping::Mapping::header`'s own doc comment on the
        // equivalent GUI/CLI mapping); nothing else writes to it except through
        // `ShmClient::write_proxy`, which this branch never calls, and slot 0's/slot
        // 1's regions are disjoint (`docs/PROTOCOL_V3_DESIGN.md`), so the other slot's own
        // `DirectCapture` never touches these same bytes.
        if !unsafe {
            ensure_direct_capture(&mut direct[slot], device, instance, physical_device, queue_family, host_ptr, capacity as vk::DeviceSize)
        } {
            return None;
        }
        let d = direct[slot].as_mut().expect("just ensured above");
        if let Some(_dims) = poll_direct_capture(d, device) {
            let n = capacity.min(frame_bytes as usize);
            original_scratch.clear();
            // SAFETY: `host_ptr` is `shm`'s own live proxy region, valid for at least
            // `capacity` bytes; `poll_direct_capture` returning `Some` just confirmed
            // this slot's fence signaled, making the GPU's writes to it visible to the
            // CPU (host-coherent memory backs every capture buffer in this module,
            // imported or not).
            original_scratch.extend_from_slice(unsafe { std::slice::from_raw_parts(host_ptr, n) });
            shm.set_frame_info(slot, width, height, proxy_format);
            return Some((width, height));
        }
        submit_direct_capture(d, device, queue, capture_image, capture_layout, width, height, proxy_format);
        None
    } else {
        if !ensure_pipeline(pipeline, device, instance, physical_device, queue_family, frame_bytes) {
            return None;
        }
        let p = pipeline.as_mut().expect("just ensured above");
        let (full, model_dims) = poll_pipeline_capture(p, slot, device, original_scratch, model_scratch);
        if full.is_some() {
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
                return Some((mw, mh));
            }
            shm.set_frame_info(slot, width, height, proxy_format);
            shm.write_proxy(slot, original_scratch);
            return Some((width, height));
        }
        submit_pipeline_capture(
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
        None
    }
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
    external_memory_host: bool,
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
    let use_direct = external_memory_host
        && (0..2).all(|slot| {
            shm.proxy_region(slot).is_some_and(|(ptr, capacity)| {
                min_imported_host_pointer_alignment(instance, physical_device).is_some_and(|alignment| {
                    let alignment = alignment as usize;
                    alignment != 0 && (ptr as usize) % alignment == 0 && capacity % alignment == 0
                })
            })
        });
    let Some(settings) = shm.composition_settings() else { return None };
    // `debug_view`'s compare/split views and a pending `capture_request`'s dump both
    // need *this* frame's own original and answer, not whatever the async pipeline
    // below happens to have on hand -- same-frame correctness matters more than
    // throughput for either, and both are rare, deliberately-triggered cases (a
    // developer toggling a debug view, or a one-shot dump request), not the normal
    // per-frame path this function otherwise replaces.
    if capture_image != image && (settings.debug_view != 0 || shm.capture_request_pending()) { return None; }
    if settings.debug_view != 0 || shm.capture_request_pending() {
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
    let bytes_per_pixel = neuralforge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > neuralforge_protocol::MAX_FRAME {
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
        if poll_or_submit_capture(
            SLOT, use_direct, pipeline, direct, device, instance, physical_device, queue, queue_family,
            capture_image, capture_layout, width, height, proxy_format, frame_bytes, None,
            crate::composition::encode_pass::EncodePush {
                white_point: settings.white_point,
                bgr_order: u32::from(bgr_order),
                reversible_mode: settings.reversible_mode,
            },
            shm, original_scratch, model_scratch,
        ).is_some() {
            shm.prepare_motion(instance, physical_device, width, height, proxy_format, original_scratch);
            if shm.begin_async_request(SLOT) {
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
    let (model_width, model_height) = scaled_dims(width, height, settings.working_scale);
    // Requested unconditionally, not only when the scale actually reduces the raster:
    // the scratch is now also where the proxy encode runs (leg 1's `ENCODE`), and the
    // encode has to happen at every working scale, including exactly 1.0. At 1.0 the
    // blit into it is 1:1 -- a device-local full-rate blit, which is what buys the
    // encode a storage image to dispatch over. `None` only when the proxy format
    // itself cannot be scratched at all, which is the same condition as before.
    let model_request = model_scratch_format(proxy_format, bgr_order).map(|format| (model_width, model_height, format));

    // Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`): the same poll-then-maybe-submit sequence
    // as before, just run once per wire slot instead of once total. Each slot is
    // completely independent -- slot 1 submitting a new capture never waits on slot
    // 0's own pending request, and vice versa, which is the entire point of having
    // two slots instead of one. If *both* slots answer within the same present call,
    // the second one processed simply overwrites `last_answer`/`raw_answer_base` --
    // the same "whichever is freshest wins" bounded-staleness tradeoff `run`'s own
    // doc comment already documents for a single slot, not a new one v3 introduces.
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
            if let Some((sent_w, sent_h)) = poll_or_submit_capture(
                slot, use_direct, pipeline, direct, device, instance, physical_device, queue, queue_family,
                capture_image, capture_layout, width, height, proxy_format, frame_bytes, model_request,
            crate::composition::encode_pass::EncodePush {
                white_point: settings.white_point,
                bgr_order: u32::from(bgr_order),
                reversible_mode: settings.reversible_mode,
            },
            shm, original_scratch, model_scratch,
            ) {
                shm.prepare_motion(instance, physical_device, width, height, proxy_format, original_scratch);
                if shm.begin_async_request(slot) {
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

        if have_answer && inflight[slot].dims == Some((width, height, proxy_format)) && neuralforge_protocol::enums::proxy_format::is_8bit(proxy_format) {
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
    if last_answer.is_empty() {
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
        if let Some(sem) = gpu.present_temporal_delta_async(
            device,
            instance,
            physical_device,
            queue,
            width,
            height,
            last_answer_dims.0,
            last_answer_dims.1,
            raw_answer_base,
            last_answer,
            *raw_answer_generation,
            bgr_order,
            image,
            crate::composition::gpu::ComposeParams {
                colour_strength: settings.colour_strength,
                transfer_strength: settings.transfer_strength,
                max_ratio: settings.max_ratio,
                proxy_encoded,
            },
        ) {
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
/// additionally blits `image` (already `TRANSFER_SRC_OPTIMAL` for the main copy below)
/// down into `scratch_image` at `(model_width, model_height)` -- `VK_FILTER_LINEAR`, a
/// hardware resize unit, not the CPU resample `working_scale` originally tried and
/// measured too slow for this thread (see `ModelScratch`'s own doc comment) -- then
/// copies `scratch_image` into `scratch_buffer`. That buffer's bytes become the SHM
/// proxy in place of `buffer`'s full-resolution ones; `buffer` is still always filled
/// at `(width, height)` exactly as before `working_scale` existed, since the
/// compositor's motion-mask reference must stay full-resolution.
fn record_capture_commands(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    initial_layout: vk::ImageLayout,
    buffer: vk::Buffer,
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
    let to_transfer_src = barrier(
        image,
        initial_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
    // SAFETY: `image` was just transitioned to `TRANSFER_SRC_OPTIMAL` above; `buffer`
    // is sized to at least `width*height*bytes_per_pixel` by whichever caller built it.
    unsafe {
        device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[region]);
    }
    if let Some((scratch_image, scratch_buffer, model_width, model_height)) = model {
        // `image` is still `TRANSFER_SRC_OPTIMAL` from the copy above -- read from it
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
        // SAFETY: `image` is `TRANSFER_SRC_OPTIMAL`; `scratch_image` was just
        // transitioned to `TRANSFER_DST_OPTIMAL`; both are 2D, single-mip, single-layer.
        unsafe { device.cmd_blit_image(cmd, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, scratch_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR) };
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
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
    pending: Option<(u32, u32, u32, Option<(u32, u32)>)>,
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
    // sensitive, so this is the one place in the pipeline that takes a real, blocking
    // (unbounded, like every other fence wait this project keeps unbounded rather
    // than guessing a safe timeout) wait -- never on the steady-state per-frame path.
    let p = existing.as_ref().expect("checked above");
    for slot in &p.slots {
        if slot.pending.is_some() {
            // SAFETY: `slot.buf.fence` is this slot's own fence; waiting for it here,
            // before any destroy below touches the resources it guards, is exactly
            // what makes that destroy sound -- the "drain before rebuilding on a live
            // device" this pipeline's own design doc calls for.
            if unsafe { device.wait_for_fences(&[slot.buf.fence], true, u64::MAX) }.is_err() {
                // A real device error, not a timeout (there is no timeout above).
                // Leave the existing pipeline exactly as it was rather than guess it's
                // safe to destroy -- next frame's `ensure_pipeline` call tries again.
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
) -> (Option<(u32, u32, u32)>, Option<(u32, u32)>) {
    let slot = &mut pipeline.slots[slot];
    let Some((width, height, proxy_format, model_dims)) = slot.pending else { return (None, None) };
    // SAFETY: `slot.buf.fence` belongs to this slot; a status query never touches
    // command-buffer/buffer/memory state, so it's sound to call regardless of
    // whether the submission this fence guards has actually completed yet. The same
    // fence guards `slot.model`'s blit/copy too (recorded into and submitted on the
    // same command buffer), so one status query covers both.
    match unsafe { device.get_fence_status(slot.buf.fence) } {
        Ok(true) => {
            let bytes_per_pixel = neuralforge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
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
            (Some((width, height, proxy_format)), model_result)
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
    if unsafe { device.queue_submit(queue, &[submit], slot.buf.fence) }.is_err() {
        return false;
    }
    slot.pending = Some((width, height, proxy_format, model_dims.map(|(_, _, w, h)| (w, h))));
    true
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
fn min_imported_host_pointer_alignment(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<vk::DeviceSize> {
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
    /// Same meaning as `PipelineSlot::pending`, for this capture's own single slot.
    pending: Option<(u32, u32, u32)>,
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
            // that destroy sound. Unbounded, like every other rebuild wait in this
            // module -- rare, not latency sensitive, and there is no timeout to guess.
            if unsafe { device.wait_for_fences(&[d.buf.fence], true, u64::MAX) }.is_err() {
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
/// `(width, height, proxy_format)`, or `None` if nothing is signaled yet (or the fence
/// reported a real error, left `pending` forever rather than guessed safe to reuse).
fn poll_direct_capture(direct: &mut DirectCapture, device: &ash::Device) -> Option<(u32, u32, u32)> {
    let dims = direct.pending?;
    // SAFETY: `direct.buf.fence` belongs to this slot; a status query never touches
    // command-buffer/buffer/memory state, so it's sound regardless of whether the
    // submission this fence guards has actually completed yet.
    match unsafe { device.get_fence_status(direct.buf.fence) } {
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
/// pending or recording/submission itself failed.
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
) -> bool {
    if direct.pending.is_some() {
        return false;
    }
    // `working_scale` is not wired into the dma-buf path -- see `poll_or_submit_capture`'s
    // own doc comment on why.
    if !record_capture_commands(device, direct.buf.cmd, image, initial_layout, direct.buf.buffer, width, height, None, None) {
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
    if unsafe { device.queue_submit(queue, &[submit], direct.buf.fence) }.is_err() {
        return false;
    }
    direct.pending = Some((width, height, proxy_format));
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
    let Some(get_memory_host_pointer_properties_ext) = get_memory_host_pointer_properties_ext else {
        // SAFETY: neither `fence` nor `pool` owns any other resource yet.
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
        return None;
    };
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
        // SAFETY: neither `fence` nor `pool` owns any other resource yet.
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
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
    // owning instance.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    // Only a memory type both the buffer itself (`reqs`) and the imported pointer
    // (`host_props`) agree on is actually usable here.
    let compatible = reqs.memory_type_bits & host_props.memory_type_bits;
    let Some(type_index) = (0..mem_props.memory_type_count)
        .find(|&i| (compatible & (1 << i)) != 0 && mem_props.memory_types[i as usize].property_flags.contains(wanted))
    else {
        // SAFETY: `buffer` has no memory bound yet; nothing else owns `fence`/`pool`.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
        }
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

    // No `vkMapMemory` here, unlike `build_capture_buffer`: `host_ptr` already *is*
    // the address this imported memory refers to -- that is the entire point of a
    // host-pointer import, and re-mapping it would be redundant at best.
    Some(CaptureBuffer { pool, cmd, fence, buffer, memory, ptr: host_ptr, capacity: bytes })
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
    if unsafe { device.queue_submit(queue, &[submit], r.fence) }.is_err() {
        return;
    }
    let _ = unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) };
}

/// Captures `image` into the proxy region, runs the shared-memory round trip, and
/// copies a result back into `image` before the caller's own present call. `resources`
/// is the per-device slot `queue_present_khr` owns (lazily built/rebuilt here).
///
/// Fails open on any error: returns without having touched `image` at all (still in
/// whatever layout the caller found it in, `PRESENT_SRC_KHR`) if anything along the way
/// doesn't work, so the caller can always fall back to presenting unmodified.
///
/// Returns `Some(semaphore)` when (and only when)
/// `composition::gpu::GpuCompose::dispatch_into_image_async` was used: `image` is
/// already fully written with the composited result, but the GPU work that wrote it
/// is not guaranteed *complete* yet (that is the entire point of the "async" in its
/// name -- this function never blocks on it). The caller **must** add that semaphore
/// to the real present call's own wait-semaphore list before presenting `image` --
/// otherwise the presentation engine could display `image` before the compute work
/// finishes writing it, a real, visible corruption/tearing bug, not merely a style
/// preference. `None` in every other case means `image` is already fully complete and
/// correctly laid out (`PRESENT_SRC_KHR`) -- safe to present with no extra wait.
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
    let bytes_per_pixel = neuralforge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
    let frame_bytes = u64::from(width) * u64::from(height) * bytes_per_pixel;
    if frame_bytes == 0 || frame_bytes as usize > neuralforge_protocol::MAX_FRAME {
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
    let to_transfer_dst = barrier(
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::TRANSFER_READ,
        vk::AccessFlags::TRANSFER_WRITE,
    );
    // SAFETY: same reasoning as the first barrier above, transitioning for the
    // write-back this same command buffer will record in stage 2.
    unsafe {
        device.cmd_pipeline_barrier(
            r.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_dst],
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
    if unsafe { device.queue_submit(queue, &[submit], r.fence) }.is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
        return None;
    }
    let t_stage1 = t_stage1_start.elapsed();

    // CPU side: the captured bytes are now in `r.ptr` (host-coherent, no explicit
    // flush/invalidate needed). Hand them to the helper, then, if it actually
    // answered, overwrite `r.ptr` in place with that answer -- stage 2 below copies
    // whatever is sitting in `r.ptr` back into `image`, so this is what makes the
    // helper's answer (a real NGX evaluation, or the helper's own proxy-echo fallback
    // when the model isn't ready -- `neuralforge_helper::main`'s per-frame loop guarantees
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
        shm.prepare_motion(instance, physical_device, width, height, proxy_format, captured);
        (t_snapshot, t_write_proxy_start.elapsed())
    };
    let original: &[u8] = original_scratch.as_slice();
    let t_roundtrip_start = std::time::Instant::now();
    let answered = shm.try_round_trip();
    let t_roundtrip = t_roundtrip_start.elapsed();
    let t_compose_start = std::time::Instant::now();
    // `Some(sem)` only when `composition::gpu::GpuCompose::dispatch_into_image_async`
    // already wrote the fully composited result straight into `image` itself, on the
    // GPU's own timeline -- skips the capture_request dump (nothing useful to dump:
    // `r.ptr` still holds the *raw* answer, not the composited result, on this path)
    // and stage 2 (there is nothing left for it to do) below, returning early instead.
    // The caller (`device.rs`) must chain `sem` into the real present call -- see this
    // function's own doc comment and `dispatch_into_image_async`'s for why.
    let mut composed_async: Option<vk::Semaphore> = None;
    if answered {
        // SAFETY: same reasoning as the read above; `ShmClient::read_answer` never
        // writes past the slice's length, which is exactly `frame_bytes` here.
        let answer_dst = unsafe { std::slice::from_raw_parts_mut(r.ptr, frame_bytes as usize) };
        shm.read_answer(SLOT, answer_dst);
        // Only `RGBA8` is handled -- `RGBA16F` still passes the helper's raw answer
        // through untouched (see `composition::apply`'s own doc comment for why, and
        // `neuralforge_protocol::enums::proxy_format` for the format codes).
        if neuralforge_protocol::enums::proxy_format::is_8bit(proxy_format) {
            if let Some(settings) = shm.composition_settings() {
                if settings.apply_model && settings.neural_enabled {
                    // GPU dispatch (`composition::gpu`) only implements the normal
                    // composited case (`compose.comp` has no concept of `debug_view`
                    // at all) -- fails open to the CPU reference
                    // (`composition::apply::apply_rgba8`, which every mode already
                    // handles) whenever the GPU path isn't applicable, isn't
                    // available, or fails, same fail-open discipline as every other
                    // stage in this function.
                    let mut composed_sync = false;
                    // The helper's model output is the intended display-referred
                    // neural result. The legacy tone-map compositor was built for
                    // a clipped, downscaled proxy, but this pipeline feeds it the
                    // full original frame; it therefore collapses most of the model
                    // edit back toward the source image. Present the raw model result
                    // for normal rendering until that proxy pipeline exists.
                    if settings.debug_view == 0 {
                        if gpu_compose.is_none() {
                            *gpu_compose = crate::composition::gpu::GpuCompose::new(device, queue_family);
                        }
                        // Try the fast, non-blocking path first -- but only when
                        // nothing on the CPU needs to see the result afterward. A
                        // pending `capture_request` does (its dump needs real bytes
                        // in `r.ptr`), so that specific, rare, deliberately-triggered
                        // case still goes through the slower, fully-synchronous
                        // CPU-visible `dispatch` below, same as before this path
                        // existed.
                        // The first neural frame after enabling the feature can
                        // race the application's present transition on NVIDIA
                        // drivers. Keep composition CPU-visible until the
                        // async handoff is proven safe for live games.
                        if false && !shm.capture_request_pending() {
                            if let Some(gpu) = gpu_compose {
                                composed_async = gpu.dispatch_into_image_async(
                                    device,
                                    instance,
                                    physical_device,
                                    queue,
                                    width,
                                    height,
                                    &original,
                                    answer_dst,
                                    settings.colour_strength,
                                    settings.transfer_strength,
                                    settings.max_ratio,
                                    bgr_order,
                                    image,
                                );
                            }
                        }
                        if composed_async.is_none() {
                            if let Some(gpu) = gpu_compose {
                                composed_sync = gpu.dispatch(
                                    device,
                                    instance,
                                    physical_device,
                                    queue,
                                    width,
                                    height,
                                    &original,
                                    answer_dst,
                                    settings.colour_strength,
                                    settings.transfer_strength,
                                    settings.max_ratio,
                                    bgr_order,
                                );
                            }
                        }
                    }
                    if composed_async.is_none() && !composed_sync {
                        crate::composition::apply::apply_rgba8(
                            &original,
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
                    answer_dst.copy_from_slice(&original);
                }
            }
        }
    }
    crate::log!(
        "[capture] {}x{} {} bytes -> proxy; round trip answered={} composed_async={}",
        width,
        height,
        frame_bytes,
        answered,
        composed_async.is_some()
    );
    if let Some(sem) = composed_async {
        // `image` is already fully written (on the GPU's own timeline -- not
        // necessarily *complete* yet, that's the entire point) and back in
        // `PRESENT_SRC_KHR`. Nothing left to do this frame except hand `sem` up to
        // the caller so the real present call waits on it.
        crate::log!(
            "[capture] timing stage1={:?} snapshot={:?} write_proxy={:?} roundtrip={:?} compose(async-dispatch-only)={:?} stage2=skipped total={:?}",
            t_stage1,
            t_snapshot,
            t_write_proxy,
            t_roundtrip,
            t_compose_start.elapsed(),
            t_stage1_start.elapsed(),
        );
        shm.publish_frame_timing(t_stage1_start.elapsed(), true);
        return Some(sem);
    }

    // Real `ShmHeader::capture_request` support: dump this frame's original and
    // final (post-composition, if any ran above) bytes to disk. Checked regardless of
    // `answered`/`proxy_format` so a request during a fail-open frame still produces a
    // (identical) matched pair rather than silently doing nothing -- `write_pair`
    // itself is the only place that would need to special-case a format it can't
    // encode, and today it always gets `RGBA8` bytes either way.
    if shm.take_capture_request() && neuralforge_protocol::enums::proxy_format::is_8bit(proxy_format) {
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
    // SAFETY: `image` is currently `TRANSFER_DST_OPTIMAL` from stage 1's own final
    // barrier; `r.buffer` (same host-coherent memory as `r.ptr`, which the CPU-side
    // block above may have just overwritten with the answer) holds exactly
    // `frame_bytes` valid bytes either way, matching `copy_in`'s own extent.
    unsafe {
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
    if unsafe { device.queue_submit(queue, &[submit2], r.fence) }.is_err() {
        return None;
    }
    // SAFETY: `r.fence` was just submitted against above. Waiting here (rather than
    // deferring to the next frame) keeps `image` fully write-back-complete and back in
    // `PRESENT_SRC_KHR` before this function returns, which is what the caller's own
    // immediately-following real present call requires.
    if unsafe { device.wait_for_fences(&[r.fence], true, u64::MAX) }.is_err() {
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
        format!("{}/neuralforge-capture-test-{}-{tag}-{n}/shm.bin", std::env::temp_dir().display(), std::process::id())
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
        assert!(submit_direct_capture(d, &device, queue, image, vk::ImageLayout::PRESENT_SRC_KHR, width, height, neuralforge_protocol::enums::proxy_format::RGBA8));
        // A real wait (not `poll_direct_capture`'s own non-blocking check) is correct
        // here: this test cares whether the capture is *correct*, not whether `run`'s
        // own present-hook discipline of never blocking holds -- that's
        // `run_never_blocks_on_a_slow_helper_and_eventually_composites`'s job, not
        // this test's.
        unsafe { device.wait_for_fences(&[d.buf.fence], true, u64::MAX) }.expect("capture fence wait failed");
        assert_eq!(poll_direct_capture(d, &device), Some((width, height, neuralforge_protocol::enums::proxy_format::RGBA8)));

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

    /// The real point of the pipelined redesign, exercised end to end against a real
    /// (if software) Vulkan device: `run` must never block a present call waiting on
    /// the helper, even when the helper genuinely takes far longer than one frame to
    /// answer -- and once it does answer, the result must actually reach `image` via
    /// a real, verifiable composited write (not just "a semaphore came back").
    #[test]
    fn run_never_blocks_on_a_slow_helper_and_eventually_composites() {
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("run_never_blocks_on_a_slow_helper_and_eventually_composites: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("blocks");
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // A live helper (matters for `poll_async_request`'s timeout budget: the long
        // "steady state" one, not the short "nobody's listening" one, since this test
        // deliberately answers slower than that short budget).
        unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) }.helper_state.store(neuralforge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that only answers `HELPER_DELAY` after it sees a new request --
        // long enough that if `run` ever blocked waiting for it, a handful of calls
        // spaced much closer together than that would visibly take just as long.
        const HELPER_DELAY: Duration = Duration::from_millis(250);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) };
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
        let proxy_format = neuralforge_protocol::enums::proxy_format::RGBA8;
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
                    false,
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
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path), "test-only open_at should always succeed against a scratch path");
        let hdr_ptr = shm.test_header_ptr();
        // SAFETY: `hdr_ptr` is this test's own live mapping, same technique the
        // sibling test above already uses.
        let hdr = unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) };
        hdr.helper_state.store(neuralforge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);
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
            let hdr = unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) };
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

        let proxy_format = neuralforge_protocol::enums::proxy_format::RGBA8;
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
                    proxy_format, false, &mut resources, &mut pipeline, &mut direct, false, &mut gpu_compose, &mut shm, &mut original_scratch,
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
    /// `NEURALFORGE_BENCH` is set in the environment, and only prints. The established
    /// way to use it (see `docs/HARDWARE_VALIDATION.md`) is to build the release test binary,
    /// copy it to `lordnikon`, and run it there with `NEURALFORGE_BENCH=1
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
        if std::env::var_os("NEURALFORGE_BENCH").is_none() {
            eprintln!("capture_hot_path_cost_per_present: set NEURALFORGE_BENCH=1 to run this GPU benchmark, skipping");
            return;
        }
        let Some((_entry, instance, physical_device, device, queue, queue_family)) = test_device() else {
            eprintln!("capture_hot_path_cost_per_present: no Vulkan loader/ICD, skipping");
            return;
        };

        let path = scratch_path("bench");
        let mut shm = ShmClient::default();
        assert!(shm.test_open_at(&path));
        let hdr_ptr = shm.test_header_ptr();
        unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) }.helper_state.store(neuralforge_protocol::enums::helper_state::RUNNING, AtomicOrdering::Relaxed);

        // A fake helper that answers as fast as it can see the request -- the real
        // steady state, where the layer has a fresh answer nearly every present and so
        // submits capture+compose work on nearly every call. That is the worst case for
        // per-present cost, which is exactly what we want to measure.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let helper = std::thread::spawn(move || {
            let hdr = unsafe { &*(hdr_ptr as *mut neuralforge_protocol::ShmHeader) };
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
        let proxy_format = neuralforge_protocol::enums::proxy_format::BGRA8;
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
                    &mut resources, &mut pipeline, &mut direct, false, &mut gpu_compose, &mut shm,
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
