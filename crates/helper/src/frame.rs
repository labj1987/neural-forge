//! Milestone 4 phase B: the real per-frame resources `EvaluateFeature` needs -- Color
//! (the proxy the layer captured), Output (where the model writes its answer), MVec
//! (motion vectors), and Depth -- and the upload/evaluate/download sequence that binds
//! them.
//!
//! Motion comes from the native layer's optical-flow stage via shared memory.
//! A missing/reset history produces zero vectors; valid history is R16G16_SFLOAT.
//!
//! Depth scope, added 2026-09-10: `DLSSNR.Depth`/`DLSSNR.DepthInverted` are real,
//! confirmed-present parameters (`strings` against the real `nvngx_dlssnr.dll` turns up
//! `DLSSNR: EvaluateFeature Color=%p MVec=%p Depth=%p Output=%p ...`, naming exactly
//! four resources) this crate never bound before -- a real, plausible cause of a first
//! real visual check (see `CLAUDE.md`) turning up a solid-white `EvaluateFeature`
//! answer despite a `0x1` success code. There is no real depth buffer to
//! give it yet (`neural_forge_layer::capture` only ever captures the presented color image),
//! so this hands the model a constant, synthetic "far plane, no real depth" value --
//! an honest stand-in, not a real per-pixel depth buffer.
//!
//! Same staging-copy discipline as `neural_forge_layer::capture`: images are populated via
//! an explicit host-visible-buffer upload/download, not a zero-copy import.

use ash::vk;

use crate::abi::{self, NgxImageViewInfoVk, NgxResourceVk};

/// Host-observed timings for one completed helper evaluation. They include the
/// corresponding Vulkan fence waits, so they measure end-to-end stage latency rather
/// than merely command-recording cost.
#[derive(Clone, Copy, Debug)]
pub struct FrameTiming {
    pub upload: std::time::Duration,
    pub evaluate: std::time::Duration,
    pub download: std::time::Duration,
}

pub struct FrameResources {
    color_format: vk::Format,
    width: u32,
    height: u32,
    queue_family: u32,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,

    color_image: vk::Image,
    color_view: vk::ImageView,
    color_memory: vk::DeviceMemory,
    output_image: vk::Image,
    output_view: vk::ImageView,
    output_memory: vk::DeviceMemory,
    mvec_image: vk::Image,
    mvec_view: vk::ImageView,
    mvec_memory: vk::DeviceMemory,
    depth_image: vk::Image,
    depth_view: vk::ImageView,
    depth_memory: vk::DeviceMemory,

    /// Host-visible staging, sized to the larger of upload (Color/MVec) or download
    /// (Output) -- one buffer, reused sequentially, same simplification
    /// `neural_forge_layer::capture` makes for its own single staging buffer. Still used
    /// for MVec/Depth always, and for Color/Output too whenever `imported_proxy`/
    /// `imported_answer` below aren't available.
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,

    /// `VK_EXT_external_memory_host` imports of the SHM proxy/answer regions
    /// (`docs/EXTERNAL_MEMORY_HOST_DESIGN.md`'s helper-side half) -- when present,
    /// `run_transfer` copies Color directly from `imported_proxy`'s buffer and Output
    /// directly into `imported_answer`'s, skipping the staging-buffer CPU copies
    /// `evaluate` would otherwise do for those two. `None` (the common fallback, e.g.
    /// the mapping didn't land aligned, or the device lacks the extension) just means
    /// this instance uses the staging path for every resource, same as before this
    /// existed.
    imported_proxy: Option<(vk::Buffer, vk::DeviceMemory)>,
    imported_answer: Option<(vk::Buffer, vk::DeviceMemory)>,

    /// Whether `evaluate` has run at least once yet -- see `DLSSNR.Reset`'s own
    /// handling in `evaluate` for why this matters. `Cell`, not a plain `bool`:
    /// `evaluate` takes `&self` (this whole struct is a fixed, per-size-class set of
    /// GPU resources shared across every frame at that size, not something that needs
    /// `&mut` to use), so this is the one piece of real per-frame state that needs
    /// interior mutability to track from there.
    reset_done: std::cell::Cell<bool>,
}

// SAFETY: every field is a plain Vulkan handle or a `vkMapMemory` pointer into memory
// this struct owns exclusively -- never aliased outside the single-threaded frame loop
// in `main.rs` that owns this value.
unsafe impl Send for FrameResources {}

fn color_format(proxy: u32) -> Option<vk::Format> {
    use neural_forge_protocol::enums::proxy_format;
    match proxy {
        proxy_format::RGBA8 => Some(vk::Format::R8G8B8A8_UNORM),
        proxy_format::BGRA8 => Some(vk::Format::B8G8R8A8_UNORM),
        _ => None,
    }
}
const MVEC_FORMAT: vk::Format = vk::Format::R16G16_SFLOAT;
// `DLSSNR.Depth`/`DLSSNR.DepthInverted` exist in the real DLL's own string table
// (`EvaluateFeature Color=%p MVec=%p Depth=%p Output=%p` -- confirmed present via
// `strings` against the real binary, 2026-09-10) but were never bound here before --
// this crate had no depth buffer to give it and the real capture path
// (`neural_forge_layer::capture`) only ever captures the presented color image, never a
// depth attachment. A color-aspect (not a real `D32_SFLOAT` depth-aspect image, to
// avoid the different layout/aspect-mask rules those need) constant-far-plane image is
// a synthetic stand-in -- "no usable depth" as honestly as this crate can currently
// say it, not a real per-pixel depth buffer. See this module's own doc comment.
const DEPTH_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| (type_bits & (1 << i)) != 0 && props.memory_types[i as usize].property_flags.contains(wanted))
}

/// Mirrors `neural_forge_layer::capture`'s own function of the same name -- see
/// `docs/EXTERNAL_MEMORY_HOST_DESIGN.md` for why this needs checking on *this* device too,
/// not assumed from the Linux side's own query.
fn min_imported_host_pointer_alignment(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<vk::DeviceSize> {
    let mut ext_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut ext_props);
    // SAFETY: `physical_device` belongs to `instance`; `props2` is a freshly built,
    // valid out-parameter with the EXT struct chained into its `pNext`.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    (ext_props.min_imported_host_pointer_alignment > 0).then_some(ext_props.min_imported_host_pointer_alignment)
}

/// Imports `host_ptr`/`bytes` (the live SHM proxy or answer region) as a
/// `TRANSFER_SRC | TRANSFER_DST` buffer -- `None` on any failure, including the
/// device simply not exporting a compatible memory type for this exact pointer.
/// Callers already treat `None` as "keep using the staging path for this resource",
/// the same fail-open discipline `neural_forge_layer::capture`'s own import helper uses.
///
/// # Safety
/// `host_ptr` must be valid for `bytes` bytes, already aligned/sized to whatever
/// `min_imported_host_pointer_alignment` the caller queried, and must remain valid and
/// exclusively accessed by this buffer's own commands for as long as the returned
/// handles exist.
unsafe fn build_imported_buffer(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, host_ptr: *mut u8, bytes: vk::DeviceSize) -> Option<(vk::Buffer, vk::DeviceMemory)> {
    // Imported host memory has its own compatibility query, separate from (and not
    // necessarily the same memory-type set as) the plain HOST_VISIBLE|HOST_COHERENT
    // search `FrameResources::new`'s own staging buffer already does. Resolved by
    // hand (never `vk::ExtExternalMemoryHostFn::load`, which *panics* if the function
    // doesn't resolve -- see `neural_forge_layer::capture::build_imported_capture_buffer`'s
    // identical fix, found the same way, live on real hardware).
    // SAFETY: `device` is live; the name is a valid, NUL-terminated C string.
    let get_memory_host_pointer_properties_ext = unsafe { instance.get_device_proc_addr(device.handle(), c"vkGetMemoryHostPointerPropertiesEXT".as_ptr()) }?;
    // SAFETY: a non-null `vkGetDeviceProcAddr(device, "vkGetMemoryHostPointerPropertiesEXT")`
    // result is guaranteed by the Vulkan spec to have this exact signature.
    let get_memory_host_pointer_properties_ext: vk::PFN_vkGetMemoryHostPointerPropertiesEXT = unsafe { std::mem::transmute(get_memory_host_pointer_properties_ext) };
    let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
    // SAFETY: `device` is live; `host_ptr` is valid for `bytes` bytes per this
    // function's own contract.
    let query_result = unsafe {
        get_memory_host_pointer_properties_ext(device.handle(), vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT, host_ptr.cast(), &mut host_props)
    };
    if query_result != vk::Result::SUCCESS {
        return None;
    }

    // A buffer that will be bound to imported memory must declare that handle type up
    // front (VUID-vkBindBufferMemory-memory-02985) -- see
    // docs/EXTERNAL_MEMORY_HOST_DESIGN.md for where this was first found missing, on the
    // Linux side, via real-hardware validation.
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
    // SAFETY: `physical_device` is the device this buffer serves; `instance` is its
    // owning instance.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let compatible = reqs.memory_type_bits & host_props.memory_type_bits;
    let Some(type_index) = find_memory_type(&mem_props, compatible, wanted) else {
        // SAFETY: `buffer` has no memory bound yet.
        unsafe { device.destroy_buffer(buffer, None) };
        return None;
    };

    let mut import_info = vk::ImportMemoryHostPointerInfoEXT::builder().handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT).host_pointer(host_ptr.cast());
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(bytes).memory_type_index(type_index).push_next(&mut import_info);
    // SAFETY: `alloc` is valid; `type_index` was just confirmed to satisfy both `reqs`
    // and `host_props`; `host_ptr`/`bytes` satisfy this function's own safety contract.
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(_) => {
            // SAFETY: `buffer` has no memory bound yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return None;
        }
    };
    // SAFETY: `buffer`/`memory` were each just created above, sized/typed to satisfy
    // each other by construction.
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return None;
    }
    Some((buffer, memory))
}

fn create_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Option<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
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
    // SAFETY: `info` is a valid `VkImageCreateInfo`.
    let image = unsafe { device.create_image(&info, None) }.ok()?;
    // SAFETY: `image` was just created and has no memory bound yet.
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let Some(type_index) = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
    else {
        // SAFETY: `image` has no memory bound and is not referenced anywhere else.
        unsafe { device.destroy_image(image, None) };
        return None;
    };
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
    // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
    let memory = unsafe { device.allocate_memory(&alloc, None) }.ok().or_else(|| {
        // SAFETY: `image` has no memory bound; freeing it here is sound.
        unsafe { device.destroy_image(image, None) };
        None
    })?;
    // SAFETY: `image`/`memory` were each just created above, sized/typed for each other.
    if unsafe { device.bind_image_memory(image, memory, 0) }.is_err() {
        // SAFETY: neither is aliased anywhere else yet.
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return None;
    }
    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
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
    Some((image, view, memory))
}

/// Owns every Vulkan object `FrameResources::new` has created so far, so an early `?`
/// return frees them instead of leaking. `new` is retried on every capture tick while
/// it keeps failing (typically under memory pressure), so a leak on the failure path
/// compounds into exactly the VRAM exhaustion that caused the failure. `disarm` once
/// the fully built value takes ownership.
struct PartialResources<'a> {
    device: &'a ash::Device,
    undo: Vec<Undo>,
}

enum Undo {
    Image(vk::Image),
    View(vk::ImageView),
    Memory(vk::DeviceMemory),
    Buffer(vk::Buffer),
    Pool(vk::CommandPool),
    Fence(vk::Fence),
}

impl<'a> PartialResources<'a> {
    fn new(device: &'a ash::Device) -> Self {
        Self { device, undo: Vec::new() }
    }

    /// Takes ownership of an `(image, view, memory)` triple from `create_image`.
    fn track_image(&mut self, (image, view, memory): (vk::Image, vk::ImageView, vk::DeviceMemory)) -> (vk::Image, vk::ImageView, vk::DeviceMemory) {
        // Pushed memory-first so the reverse-order drop destroys view, image, memory.
        self.undo.push(Undo::Memory(memory));
        self.undo.push(Undo::Image(image));
        self.undo.push(Undo::View(view));
        (image, view, memory)
    }

    fn track(&mut self, item: Undo) {
        self.undo.push(item);
    }

    /// Ownership moves to the finished `FrameResources`; nothing is freed on drop.
    fn disarm(mut self) {
        self.undo.clear();
    }
}

impl Drop for PartialResources<'_> {
    fn drop(&mut self) {
        // SAFETY: each object was created on `device` by `new`, has never been
        // submitted to a queue (the only work `new` does is creation and binding), and
        // is destroyed exactly once, newest first.
        unsafe {
            for item in self.undo.drain(..).rev() {
                match item {
                    Undo::Image(h) => self.device.destroy_image(h, None),
                    Undo::View(h) => self.device.destroy_image_view(h, None),
                    Undo::Memory(h) => self.device.free_memory(h, None),
                    Undo::Buffer(h) => self.device.destroy_buffer(h, None),
                    Undo::Pool(h) => self.device.destroy_command_pool(h, None),
                    Undo::Fence(h) => self.device.destroy_fence(h, None),
                }
            }
        }
    }
}

impl FrameResources {
    /// Builds every resource `EvaluateFeature` needs for a `width`x`height` frame.
    /// `None` on any failure -- callers treat that as "skip evaluate this frame",
    /// mirroring `neural_forge_layer::capture`'s own fail-open discipline.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
        width: u32,
        height: u32,
        proxy_format: u32,
        proxy_region: (*mut u8, usize),
        answer_region: (*mut u8, usize),
    ) -> Option<Self> {
        let color_format = color_format(proxy_format)?;
        // SAFETY: `physical_device` is the device everything below is built against.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mut partial = PartialResources::new(device);

        let (color_image, color_view, color_memory) = partial.track_image(create_image(
            device,
            &mem_props,
            width,
            height,
            color_format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?);
        let (output_image, output_view, output_memory) = partial.track_image(create_image(
            device,
            &mem_props,
            width,
            height,
            color_format,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?);
        let (mvec_image, mvec_view, mvec_memory) = partial.track_image(create_image(
            device,
            &mem_props,
            width,
            height,
            MVEC_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?);
        let (depth_image, depth_view, depth_memory) = partial.track_image(create_image(
            device,
            &mem_props,
            width,
            height,
            DEPTH_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?);

        let pool_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `pool_info` is valid.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.ok()?;
        partial.track(Undo::Pool(pool));
        let alloc_info =
            vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        // SAFETY: `pool` was just created above.
        let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }.ok()?[0];
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: `fence_info` is valid.
        let fence = unsafe { device.create_fence(&fence_info, None) }.ok()?;
        partial.track(Undo::Fence(fence));

        // Upload holds Color plus R16G16_SFLOAT motion (4 bytes/pixel each).
        let staging_size = u64::from(width) * u64::from(height) * 8;
        let buf_info = vk::BufferCreateInfo::builder()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `buf_info` is valid.
        let staging_buffer = unsafe { device.create_buffer(&buf_info, None) }.ok()?;
        partial.track(Undo::Buffer(staging_buffer));
        // SAFETY: `staging_buffer` was just created, not yet bound to memory.
        let reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let type_index = find_memory_type(
            &mem_props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index);
        // SAFETY: `alloc` is valid; `type_index` satisfies `reqs`.
        let staging_memory = unsafe { device.allocate_memory(&alloc, None) }.ok()?;
        partial.track(Undo::Memory(staging_memory));
        // SAFETY: `staging_buffer`/`staging_memory` were each just created, sized/typed
        // for each other.
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }.ok()?;
        // SAFETY: `staging_memory` is `HOST_VISIBLE`; mapping the whole allocation is
        // always in bounds.
        let staging_ptr = unsafe { device.map_memory(staging_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.ok()?.cast::<u8>();

        // Try importing the live SHM proxy/answer regions as device memory
        // (`VK_EXT_external_memory_host`) -- `run_transfer` uses these directly
        // instead of the staging buffer above when available. Checked against the
        // *actual* runtime pointer, not just assumed: `shm::open`'s own aligned-mapping
        // attempt can fall back to an unaligned view, and this device might lack the
        // extension `WANTED_DEVICE_EXTENSIONS` only *requests*, never guarantees.
        let alignment = min_imported_host_pointer_alignment(instance, physical_device);
        let try_import = |region: (*mut u8, usize)| {
            let (ptr, capacity) = region;
            let alignment = alignment?;
            if ptr.is_null() || capacity == 0 || (ptr as usize) % alignment as usize != 0 || capacity as u64 % alignment != 0 {
                return None;
            }
            // SAFETY: `ptr`/`capacity` describe a live SHM region for as long as this
            // process's own mapping stays open (the life of the process); nothing else
            // writes to the proxy region while a request is outstanding, and nothing
            // else writes to the answer region except this same buffer's own download
            // copy -- the single-reader/single-writer discipline the wire protocol's
            // "one outstanding request at a time" rule already guarantees, the same
            // reasoning `neural_forge_layer::capture::DirectCapture` relies on for its own
            // single slot.
            unsafe { build_imported_buffer(device, instance, physical_device, ptr, capacity as vk::DeviceSize) }
        };
        let imported_proxy = try_import(proxy_region);
        let imported_answer = try_import(answer_region);
        crate::log!(
            "[frame] {width}x{height} resources: imported_proxy={} imported_answer={}",
            imported_proxy.is_some(),
            imported_answer.is_some()
        );

        // Nothing below can fail: the imports above already clean up after themselves
        // and report `None` for "not available".
        partial.disarm();
        Some(Self {
            color_format,
            width,
            height,
            queue_family,
            pool,
            cmd,
            fence,
            color_image,
            color_view,
            color_memory,
            output_image,
            output_view,
            output_memory,
            mvec_image,
            mvec_view,
            mvec_memory,
            depth_image,
            depth_view,
            depth_memory,
            staging_buffer,
            staging_memory,
            staging_ptr,
            imported_proxy,
            imported_answer,
            reset_done: std::cell::Cell::new(false),
        })
    }

    pub fn matches(&self, queue_family: u32, width: u32, height: u32, proxy_format: u32) -> bool {
        Some(self.color_format) == color_format(proxy_format) && self.queue_family == queue_family && self.width == width && self.height == height
    }

    /// Uploads `proxy` and motion, runs `EvaluateFeature`, downloads
    /// Output into `answer_out`. Returns timings for a completed evaluation, or `None`
    /// (leaving `answer_out` untouched) on a failure, including a guarded fault inside
    /// `EvaluateFeature` itself.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        evaluate_feature: abi::FnVkEvaluateFeature,
        feature: abi::NgxHandle,
        params: abi::NgxParameter,
        proxy: &[u8],
        motion: &[u8],
        motion_scale: [f32; 2],
        reset_history: bool,
        sharpness: f32,
        answer_out: &mut [u8],
    ) -> Option<FrameTiming> {
        static EVALUATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let evaluate_no = EVALUATES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pixel_count = (self.width as usize) * (self.height as usize);
        if proxy.len() < pixel_count * 4 || answer_out.len() < pixel_count * 4 {
            return None;
        }

        // Stage 1: upload proxy -> Color and motion -> MVec. Skipped for proxy when
        // `imported_proxy` is set: `proxy` is already a view into the exact memory
        // that buffer is imported from (`ShmMapping::frame_regions`/`proxy_and_answer_regions`
        // share the same base), so `run_transfer` below reads directly from it instead.
        if self.imported_proxy.is_none() {
            // SAFETY: `staging_ptr` is a live mapping of at least `pixel_count * 4`
            // bytes (this type's own construction sized it to exactly that).
            unsafe { std::ptr::copy_nonoverlapping(proxy.as_ptr(), self.staging_ptr, pixel_count * 4) };
        }
        unsafe {
            let dst = self.staging_ptr.add(pixel_count * 4);
            if motion.len() == pixel_count * 4 { std::ptr::copy_nonoverlapping(motion.as_ptr(),dst,motion.len()); }
            else { std::ptr::write_bytes(dst,0,pixel_count*4); }
        }
        let t_upload_start = std::time::Instant::now();
        if !self.run_transfer(device, queue, TransferKind::Upload) {
            return None;
        }
        let t_upload = t_upload_start.elapsed();

        // Stage 2: the real NGX call, guarded the same way every other DLL call in
        // this crate already is.
        let color_info = self.resource_info(self.color_view, self.color_image, self.color_format);
        let output_info = self.resource_info(self.output_view, self.output_image, self.color_format);
        let mvec_info = self.resource_info(self.mvec_view, self.mvec_image, MVEC_FORMAT);
        let depth_info = self.resource_info(self.depth_view, self.depth_image, DEPTH_FORMAT);
        // Parameter names guessed "for shape" from the same `DLSSNR.*` convention
        // `crates/helper/src/ngx.rs::create_feature_at` already uses for the scalar
        // parameters -- no public spec exists for this fictional feature's resource
        // bindings any more than for its scalars. Wrong names/slots here fail via the
        // guard below, not a crash, exactly like a wrong `CreateFeature` parameter did
        // before the v0.1.2 fix. `Depth`/`DepthInverted` and every `*Subrect*` name
        // below are confirmed present in the real DLL's own string table (`strings`,
        // 2026-09-10) -- not new guesses, the first ones checked against real evidence.
        let name = |n: &str| std::ffi::CString::new(n).unwrap();
        // SAFETY: `params` was allocated and validated by the caller (`ngx::load_and_init`).
        let t_eval = unsafe {
            let mut color = NgxResourceVk::from_image_view(color_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.Color").as_ptr(), std::ptr::from_mut(&mut color).cast());
            let mut output = NgxResourceVk::from_image_view(output_info, true);
            abi::ngx_set_ptr(params, name("DLSSNR.Output").as_ptr(), std::ptr::from_mut(&mut output).cast());
            abi::ngx_set_f32(params, name("DLSSNR.MVecScaleX").as_ptr(), motion_scale[0]);
            abi::ngx_set_f32(params, name("DLSSNR.MVecScaleY").as_ptr(), motion_scale[1]);
            // Only sharpness is written per evaluate: DoSharpening is enabled at create and this is
            // the per-frame amount it applies. Style, intensity, the local strengths and auto mask
            // are latched by the model at creation (`ngx::set_create_tuning`); writing them here
            // does nothing to a running feature and poisons the block for the next create.
            abi::ngx_set_f32(params, name("Sharpness").as_ptr(), sharpness.clamp(0.0, 1.0));
            let mut mvec = NgxResourceVk::from_image_view(mvec_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.MVec").as_ptr(), std::ptr::from_mut(&mut mvec).cast());
            let mut depth = NgxResourceVk::from_image_view(depth_info, false);
            abi::ngx_set_ptr(params, name("DLSSNR.Depth").as_ptr(), std::ptr::from_mut(&mut depth).cast());
            // Standard, non-reversed-Z convention (near=0, far=1) -- matches the
            // constant 1.0 ("far") the depth image is cleared to in `run_transfer`.
            abi::ngx_set_u32(params, name("DLSSNR.DepthInverted").as_ptr(), 0);

            // Every resource is the full frame at (0,0) -- no sub-rect windowing is
            // used anywhere in this crate yet.
            for resource in ["Color", "Output", "MVec", "Depth"] {
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectBaseX")).as_ptr(), 0);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectBaseY")).as_ptr(), 0);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectWidth")).as_ptr(), self.width);
                abi::ngx_set_u32(params, name(&format!("DLSSNR.{resource}SubrectHeight")).as_ptr(), self.height);
            }
            // `ngx::create_feature_at` sets `DLSSNR.Reset = 1` once, at creation, and
            // this crate never touched it again before now -- every single evaluate
            // call therefore told the model "no valid history, this is frame one" for
            // the life of the feature, a real, plausible cause of a first real visual
            // check (see CLAUDE.md) finding every frame produces the exact same
            // (solid white) output regardless of input: a temporal model's real,
            // history-dependent answer would only ever appear from the second frame
            // set to `Reset = 0` onward, which never happened before this. `1` only on
            // this feature's actual first `evaluate` call, `0` on every one after.
            let reset = u32::from(!self.reset_done.replace(true) || reset_history);
            abi::ngx_set_u32(params, name("DLSSNR.Reset").as_ptr(), reset);

            let t_eval_start = std::time::Instant::now();
            // NGX Vulkan evaluation records its GPU work into a caller-owned, live
            // command buffer, just as feature creation does. A null buffer can return
            // success while recording no output work, which leaves Output untouched.
            let Some(result) = self.run_evaluate(device, queue, || {
                crate::guard::guarded(
                    || evaluate_feature(self.cmd, feature, params, std::ptr::null()),
                    abi::result::FAIL_SEH,
                )
            }) else {
                return None;
            };
            let t_eval = t_eval_start.elapsed();
            let failed = !abi::succeeded(result.0) || result.1 != 0;
            // Bounded: one line per evaluate, forever, is real time under Wine. Failures always log.
            if failed || crate::logging::sampled(evaluate_no) {
                crate::log!("[ngx] EvaluateFeature -> {:#x} seh={:#x} took={:?}", result.0 as u32, result.1, t_eval);
            }
            if failed {
                return None;
            }
            t_eval
        };

        // Stage 3: download Output -> answer_out.
        let t_download_start = std::time::Instant::now();
        if !self.run_transfer(device, queue, TransferKind::Download) {
            return None;
        }
        let t_download = t_download_start.elapsed();
        if crate::logging::sampled(evaluate_no) {
            crate::log!(
                "[frame] timing upload={:?} eval={:?} download={:?} total={:?}",
                t_upload,
                t_eval,
                t_download,
                t_upload + t_eval + t_download
            );
        }
        // Skipped when `imported_answer` is set: `run_transfer`'s download copy just
        // wrote Output directly into the exact memory `answer_out` is a view of.
        if self.imported_answer.is_none() {
            // SAFETY: `staging_ptr` is a live mapping of at least `pixel_count * 4` bytes.
            unsafe { std::ptr::copy_nonoverlapping(self.staging_ptr, answer_out.as_mut_ptr(), pixel_count * 4) };
        }
        Some(FrameTiming {
            upload: t_upload,
            evaluate: t_eval,
            download: t_download,
        })
    }

    fn resource_info(&self, view: vk::ImageView, image: vk::Image, format: vk::Format) -> NgxImageViewInfoVk {
        NgxImageViewInfoVk {
            image_view: view,
            image,
            subresource_range: vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
            format,
            width: self.width,
            height: self.height,
        }
    }

    /// Records NGX evaluation into the same queue used for the resource upload and
    /// download, then waits for completion before Output is copied back to staging.
    fn run_evaluate<F>(&self, device: &ash::Device, queue: vk::Queue, evaluate: F) -> Option<(abi::NgxResult, u32)>
    where
        F: FnOnce() -> (abi::NgxResult, u32),
    {
        if unsafe { device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return None;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        if unsafe { device.begin_command_buffer(self.cmd, &begin_info) }.is_err() {
            return None;
        }
        let result = evaluate();
        if result.1 != 0 || unsafe { device.end_command_buffer(self.cmd) }.is_err() {
            return None;
        }
        if unsafe { device.reset_fences(&[self.fence]) }.is_err() {
            return None;
        }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.cmd)).build();
        if unsafe { device.queue_submit(queue, &[submit], self.fence) }.is_err() {
            return None;
        }
        if unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }.is_err() {
            return None;
        }
        Some(result)
    }

    fn run_transfer(&self, device: &ash::Device, queue: vk::Queue, kind: TransferKind) -> bool {
        // SAFETY: `self.cmd` was allocated from `self.pool`, created with
        // `RESET_COMMAND_BUFFER`.
        if unsafe { device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()) }.is_err() {
            return false;
        }
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `self.cmd` was just reset above.
        if unsafe { device.begin_command_buffer(self.cmd, &begin_info) }.is_err() {
            return false;
        }
        let region = |width: u32, height: u32| {
            vk::BufferImageCopy::builder()
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
                .build()
        };
        let sub = |aspect| {
            vk::ImageSubresourceRange::builder().aspect_mask(aspect).base_mip_level(0).level_count(1).base_array_layer(0).layer_count(1).build()
        };
        let img_barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(sub(vk::ImageAspectFlags::COLOR))
                .src_access_mask(src)
                .dst_access_mask(dst)
                .build()
        };
        match kind {
            TransferKind::Upload => {
                let to_dst_color = img_barrier(
                    self.color_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let to_dst_mvec = img_barrier(
                    self.mvec_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                let to_dst_depth = img_barrier(
                    self.depth_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                // SAFETY: `self.cmd` is recording; all three images were just created
                // (`UNDEFINED` matches their real, never-yet-transitioned layout).
                unsafe {
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_dst_color, to_dst_mvec, to_dst_depth],
                    );
                    // `imported_proxy`, when set, is the live SHM proxy region itself
                    // (see `FrameResources::new`) -- reading Color straight from it
                    // instead of `staging_buffer` is exactly what skipping `evaluate`'s
                    // own `proxy -> staging_ptr` copy above requires.
                    let color_src = self.imported_proxy.map_or(self.staging_buffer, |(buffer, _)| buffer);
                    device.cmd_copy_buffer_to_image(
                        self.cmd,
                        color_src,
                        self.color_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region(self.width, self.height)],
                    );
                    let mut motion_region = region(self.width,self.height);
                    motion_region.buffer_offset = u64::from(self.width)*u64::from(self.height)*4;
                    device.cmd_copy_buffer_to_image(self.cmd,self.staging_buffer,self.mvec_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[motion_region]);
                    // Depth: no real depth buffer captured yet either (see module doc
                    // comment) -- a constant 1.0 ("far plane", standard non-reversed-Z
                    // convention, matching `DLSSNR.DepthInverted = 0` below) rather
                    // than leaving it undefined.
                    device.cmd_clear_color_image(
                        self.cmd,
                        self.depth_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] },
                        &[sub(vk::ImageAspectFlags::COLOR)],
                    );
                    let to_shader = img_barrier(
                        self.color_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let mvec_to_shader = img_barrier(
                        self.mvec_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let depth_to_shader = img_barrier(
                        self.depth_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    let output_to_general = img_barrier(
                        self.output_image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::SHADER_WRITE,
                    );
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_shader, mvec_to_shader, depth_to_shader, output_to_general],
                    );
                }
            }
            TransferKind::Download => {
                let to_src = img_barrier(
                    self.output_image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
                // SAFETY: `self.cmd` is recording; `output_image` was left `GENERAL`
                // by the upload stage's own final barrier, matching what
                // `EvaluateFeature` (run on the CPU-side call in between, not this
                // command buffer) was told to expect as the storage image's layout.
                unsafe {
                    device.cmd_pipeline_barrier(
                        self.cmd,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_src],
                    );
                    // `imported_answer`, when set, is the live SHM answer region itself
                    // -- writing Output straight into it is exactly what skipping
                    // `evaluate`'s own `staging_ptr -> answer_out` copy afterward
                    // requires.
                    let answer_dst = self.imported_answer.map_or(self.staging_buffer, |(buffer, _)| buffer);
                    device.cmd_copy_image_to_buffer(
                        self.cmd,
                        self.output_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        answer_dst,
                        &[region(self.width, self.height)],
                    );
                }
            }
        }
        if unsafe { device.end_command_buffer(self.cmd) }.is_err() {
            return false;
        }
        // SAFETY: `self.fence` starts signaled or was reset+waited-on by this same
        // function's previous call.
        if unsafe { device.reset_fences(&[self.fence]) }.is_err() {
            return false;
        }
        let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&self.cmd)).build();
        // SAFETY: `self.cmd` was just recorded and ended above.
        if unsafe { device.queue_submit(queue, &[submit], self.fence) }.is_err() {
            return false;
        }
        // SAFETY: `self.fence` was just submitted against above.
        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }.is_ok()
    }

    /// # Safety
    /// Must only be called at process-teardown time, with no submitted work
    /// referencing these handles still in flight.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            device.destroy_fence(self.fence, None);
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);
            if let Some((buffer, memory)) = self.imported_proxy {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            if let Some((buffer, memory)) = self.imported_answer {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color_image, None);
            device.free_memory(self.color_memory, None);
            device.destroy_image_view(self.output_view, None);
            device.destroy_image(self.output_image, None);
            device.free_memory(self.output_memory, None);
            device.destroy_image_view(self.mvec_view, None);
            device.destroy_image(self.mvec_image, None);
            device.free_memory(self.mvec_memory, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

enum TransferKind {
    Upload,
    Download,
}

#[cfg(test)] mod format_tests {
    use super::*;
    use neural_forge_protocol::enums::proxy_format;
    #[test]
    #[ignore = "requires Vulkan under Wine on real hardware"]
    fn rgba_and_bgra_resources_recreate_on_format_change() {
        let entry = unsafe {ash::Entry::load()}.unwrap();
        let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let instance = unsafe {entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app),None)}.unwrap();
        let pd = unsafe {instance.enumerate_physical_devices()}.unwrap()[0];
        let q = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
        let device = unsafe {instance.create_device(pd,&vk::DeviceCreateInfo::builder().queue_create_infos(&q),None)}.unwrap();
        for format in [proxy_format::RGBA8,proxy_format::BGRA8] {
            // No real SHM mapping in this manual test -- null/zero-length regions,
            // which `try_import` inside `new` safely declines (falling back to the
            // staging path) rather than dereferencing.
            let f = FrameResources::new(&device,&instance,pd,0,512,512,format,(std::ptr::null_mut(),0),(std::ptr::null_mut(),0)).expect("real Color/Output/MVec resources");
            assert!(f.matches(0,512,512,format));
            assert!(!f.matches(0,512,512,if format == proxy_format::RGBA8 {proxy_format::BGRA8} else {proxy_format::RGBA8}));
            assert!(!f.matches(0,256,512,format));
            unsafe {f.destroy(&device)};
        }
        unsafe {device.destroy_device(None);instance.destroy_instance(None)};
    }
    #[test] fn ngx_formats_match_raw_bytes() {
        assert_eq!(color_format(proxy_format::RGBA8),Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(color_format(proxy_format::BGRA8),Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(color_format(proxy_format::RGBA16F),None);
        assert_eq!(color_format(999),None);
    }
}
