//! The GPU's optical flow for the after-the-upscaler history (`NEURAL_FORGE_POST_MOTION=flow`,
//! docs/NATIVE_BACKEND.md, "History after the upscaler"): `VK_NV_optical_flow` between the frame's proxy and
//! the previous frame's, whose vectors `native_post_preprocess.comp` reprojects each pass's history with.
//!
//! - Device creation (`lib.rs`): with `flow` selected and the device offering the extension, its
//!   `opticalFlow` feature and a queue family flagged `QUEUE_OPTICAL_FLOW_BIT_NV` the application does not
//!   request itself ([`queue_family`]), the request gets the extension, the feature and one queue of that
//!   family ([`FlowQueue`]); a request with them that the driver refuses is made again without.
//! - Per frame size and slot, a [`Session`]: two input images (`B8G8R8A8_UNORM`, the proxy's bytes copied in
//!   whatever its channel order: both frames are the same) used in turn as the frame and the reference, and
//!   one flow image (`R16G16_S10_5_NV`) at the coarsest output grid the device offers of 4x4, 2x2 and 1x1.
//!   The input is the frame and the reference the previous frame, so each vector points from where a pixel
//!   is to where it was (backward flow, what the reprojection wants).
//! - Per frame with a history: the proxy is copied into the frame's input image on the compute queue
//!   (signalling a semaphore), the flow runs on the optical-flow queue (waiting for it, signalling another),
//!   and the frame's compute work waits for that one and copies the vectors into a buffer the preprocess
//!   reads. The server thread waits for the frame's fence, as it does for every frame; the game never waits.
//!
//! The entry points are resolved by hand and checked (a name the loader does not resolve is "no flow", never
//! a panic). Anything missing makes `flow` run as `zero`, logged once with the reason.

use std::cell::Cell;
use std::ffi::{c_char, CStr};

use ash::vk;

/// The extension.
pub(crate) const EXTENSION: &CStr = c"VK_NV_optical_flow";
/// The input images' format: the proxy's four bytes as they are.
pub(crate) const INPUT_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
/// The flow vectors' format: x and y, signed 10.5 fixed point.
pub(crate) const FLOW_FORMAT: vk::Format = vk::Format::R16G16_S10_5_NV;

/// The optical-flow queue a device was created with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlowQueue {
    pub family: u32,
    pub index: u32,
}

/// The family to add the optical-flow queue in at device creation, or why there is none: `extension` and
/// `feature` are what the device offers, `requested` the device request's queues (the application's, with
/// the layer's own added).
pub(crate) fn queue_family(
    extension: bool,
    feature: bool,
    families: &[vk::QueueFamilyProperties],
    requested: &[vk::DeviceQueueCreateInfo],
) -> Result<u32, String> {
    if !extension {
        return Err("the device does not offer VK_NV_optical_flow".into());
    }
    if !feature {
        return Err("the device's opticalFlow feature is not supported".into());
    }
    let family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV) && f.queue_count > 0)
        .ok_or("no queue family offers optical flow")? as u32;
    if requested.iter().any(|q| q.queue_family_index == family) {
        return Err(format!("the application uses the optical-flow queue family {family} itself"));
    }
    Ok(family)
}

/// The output grid: the coarsest of 4x4, 2x2 and 1x1 the device offers (8x8 only when it offers nothing
/// finer), with its size in pixels.
pub(crate) fn grid(supported: vk::OpticalFlowGridSizeFlagsNV) -> Option<(vk::OpticalFlowGridSizeFlagsNV, u32)> {
    [
        (vk::OpticalFlowGridSizeFlagsNV::TYPE_4X4, 4),
        (vk::OpticalFlowGridSizeFlagsNV::TYPE_2X2, 2),
        (vk::OpticalFlowGridSizeFlagsNV::TYPE_1X1, 1),
        (vk::OpticalFlowGridSizeFlagsNV::TYPE_8X8, 8),
    ]
    .into_iter()
    .find(|(flag, _)| supported.contains(*flag))
}

/// What the device says about optical flow (`VkPhysicalDeviceOpticalFlowPropertiesNV` and the formats).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Capabilities {
    pub grids: vk::OpticalFlowGridSizeFlagsNV,
    pub min: (u32, u32),
    pub max: (u32, u32),
    pub inputs: Vec<vk::Format>,
    pub outputs: Vec<vk::Format>,
}

/// A session for `width` x `height`: its grid flag and size, or why not.
pub(crate) fn session_plan(caps: &Capabilities, width: u32, height: u32) -> Result<(vk::OpticalFlowGridSizeFlagsNV, u32), String> {
    if !caps.inputs.contains(&INPUT_FORMAT) {
        return Err(format!("the optical flow takes none of the proxy's formats (it takes {:?})", caps.inputs));
    }
    if !caps.outputs.contains(&FLOW_FORMAT) {
        return Err(format!("the optical flow writes no R16G16_S10_5 vectors (it writes {:?})", caps.outputs));
    }
    if width < caps.min.0 || height < caps.min.1 || width > caps.max.0 || height > caps.max.1 {
        return Err(format!("{width}x{height} is outside the optical flow's {}x{} to {}x{}", caps.min.0, caps.min.1, caps.max.0, caps.max.1));
    }
    grid(caps.grids).ok_or_else(|| "the optical flow offers no output grid".into())
}

/// The function a name resolves to, transmuted to `T`; `None` when it does not resolve.
///
/// # Safety
/// `T` is the function pointer type of `name`.
unsafe fn cast<T: Copy>(f: vk::PFN_vkVoidFunction) -> Option<T> {
    // SAFETY: a resolved Vulkan entry point has the signature its name gives (the caller's contract).
    f.map(|f| unsafe { std::mem::transmute_copy::<unsafe extern "system" fn(), T>(&f) })
}

/// Whether the physical device offers the `opticalFlow` feature, asked through `gipa` with the
/// features query `properties2` allows (none: not asked, `false`).
///
/// # Safety
/// `gipa` and `instance` are the next layer's for `physical`.
pub(crate) unsafe fn feature(
    gipa: vk::PFN_vkGetInstanceProcAddr,
    instance: vk::Instance,
    physical: vk::PhysicalDevice,
    properties2: crate::vram::Properties2,
) -> bool {
    let name: &CStr = match properties2 {
        crate::vram::Properties2::Core => c"vkGetPhysicalDeviceFeatures2",
        crate::vram::Properties2::Khr => c"vkGetPhysicalDeviceFeatures2KHR",
        crate::vram::Properties2::None => return false,
    };
    // SAFETY: the name's signature.
    let Some(features2) = (unsafe { cast::<vk::PFN_vkGetPhysicalDeviceFeatures2>(gipa(instance, name.as_ptr())) }) else { return false };
    let mut of = vk::PhysicalDeviceOpticalFlowFeaturesNV::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut of);
    // SAFETY: a valid chain, filled in by the call.
    unsafe { features2(physical, &mut f2) };
    of.optical_flow == vk::TRUE
}

/// [`Capabilities`] of `physical`, through `gipa`.
///
/// # Safety
/// As [`feature`].
pub(crate) unsafe fn capabilities(
    gipa: vk::PFN_vkGetInstanceProcAddr,
    instance: vk::Instance,
    physical: vk::PhysicalDevice,
    properties2: crate::vram::Properties2,
) -> Result<Capabilities, String> {
    let name: &CStr = match properties2 {
        crate::vram::Properties2::Core => c"vkGetPhysicalDeviceProperties2",
        crate::vram::Properties2::Khr => c"vkGetPhysicalDeviceProperties2KHR",
        crate::vram::Properties2::None => return Err("the instance has no properties2 query".into()),
    };
    // SAFETY: the names' signatures.
    let (properties2, formats) = unsafe {
        (
            cast::<vk::PFN_vkGetPhysicalDeviceProperties2>(gipa(instance, name.as_ptr())).ok_or("vkGetPhysicalDeviceProperties2 does not resolve")?,
            cast::<vk::PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV>(gipa(instance, c"vkGetPhysicalDeviceOpticalFlowImageFormatsNV".as_ptr()))
                .ok_or("vkGetPhysicalDeviceOpticalFlowImageFormatsNV does not resolve")?,
        )
    };
    let mut of = vk::PhysicalDeviceOpticalFlowPropertiesNV::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut of);
    // SAFETY: a valid chain, filled in by the call.
    unsafe { properties2(physical, &mut p2) };
    let list = |usage: vk::OpticalFlowUsageFlagsNV| -> Vec<vk::Format> {
        let info = vk::OpticalFlowImageFormatInfoNV::default().usage(usage);
        let mut count = 0u32;
        // SAFETY: the count query, then as many structures as it said.
        unsafe {
            if formats(physical, &info, &mut count, std::ptr::null_mut()) != vk::Result::SUCCESS {
                return Vec::new();
            }
            let mut out = vec![vk::OpticalFlowImageFormatPropertiesNV::default(); count as usize];
            let r = formats(physical, &info, &mut count, out.as_mut_ptr());
            if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                return Vec::new();
            }
            out.truncate(count as usize);
            out.iter().map(|f| f.format).collect()
        }
    };
    Ok(Capabilities {
        grids: of.supported_output_grid_sizes,
        min: (of.min_width, of.min_height),
        max: (of.max_width, of.max_height),
        inputs: list(vk::OpticalFlowUsageFlagsNV::INPUT),
        outputs: list(vk::OpticalFlowUsageFlagsNV::OUTPUT),
    })
}

/// Whether `info`'s chain already holds a structure of type `ty`.
///
/// # Safety
/// The chain is valid.
pub(crate) unsafe fn chains(info: &vk::DeviceCreateInfo, ty: vk::StructureType) -> bool {
    let mut next = info.p_next.cast::<vk::BaseInStructure>();
    while !next.is_null() {
        // SAFETY: every node of a valid chain starts with sType and pNext.
        let node = unsafe { &*next };
        if node.s_type == ty {
            return true;
        }
        next = node.p_next;
    }
    false
}

/// A device request with the optical flow added: the extension, the feature (prepended to the chain) and
/// one queue of `family`. Holds what the request points at.
pub(crate) struct Request<'a> {
    /// The application's create info, pointing into the fields below too: valid while `self` lives.
    pub info: vk::DeviceCreateInfo<'a>,
    _features: Box<vk::PhysicalDeviceOpticalFlowFeaturesNV<'static>>,
    _names: Vec<*const c_char>,
    _queues: Vec<vk::DeviceQueueCreateInfo<'a>>,
    _priority: Box<[f32; 1]>,
}

impl<'a> Request<'a> {
    /// `None` when the request already chains an optical-flow feature structure (the application's own:
    /// one structure of a type is all a chain may hold).
    ///
    /// # Safety
    /// `base` is a valid request that outlives the result.
    pub(crate) unsafe fn new(base: &vk::DeviceCreateInfo<'a>, family: u32) -> Option<Self> {
        // SAFETY: forwarded.
        if unsafe { chains(base, vk::StructureType::PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV) } {
            return None;
        }
        let mut features = Box::new(vk::PhysicalDeviceOpticalFlowFeaturesNV { optical_flow: vk::TRUE, ..Default::default() });
        features.p_next = base.p_next.cast_mut();
        // SAFETY: a valid request's arrays hold their counts (a zero count may come with a null pointer).
        let (mut names, mut queues) = unsafe {
            let names: Vec<*const c_char> = if base.enabled_extension_count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(base.pp_enabled_extension_names, base.enabled_extension_count as usize).to_vec()
            };
            let queues: Vec<vk::DeviceQueueCreateInfo<'a>> = if base.queue_create_info_count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(base.p_queue_create_infos, base.queue_create_info_count as usize).to_vec()
            };
            (names, queues)
        };
        // SAFETY: the names are NUL-terminated.
        if !names.iter().any(|&n| unsafe { CStr::from_ptr(n) } == EXTENSION) {
            names.push(EXTENSION.as_ptr());
        }
        let priority = Box::new([1.0f32]);
        // `priority` moves into the result (the box's contents stay put): the pointer is set directly.
        queues.push(vk::DeviceQueueCreateInfo { queue_family_index: family, queue_count: 1, p_queue_priorities: priority.as_ptr(), ..Default::default() });
        let mut info = *base;
        info.p_next = std::ptr::from_ref(&*features).cast();
        info.enabled_extension_count = names.len() as u32;
        info.pp_enabled_extension_names = names.as_ptr();
        info.queue_create_info_count = queues.len() as u32;
        info.p_queue_create_infos = queues.as_ptr();
        Some(Self { info, _features: features, _names: names, _queues: queues, _priority: priority })
    }
}

/// The extension's device entry points.
#[derive(Clone, Copy)]
pub(crate) struct Fns {
    create: vk::PFN_vkCreateOpticalFlowSessionNV,
    destroy: vk::PFN_vkDestroyOpticalFlowSessionNV,
    bind: vk::PFN_vkBindOpticalFlowSessionImageNV,
    execute: vk::PFN_vkCmdOpticalFlowExecuteNV,
}

impl Fns {
    /// # Safety
    /// `device` is live.
    unsafe fn load(instance: &ash::Instance, device: vk::Device) -> Result<Self, String> {
        let get = |name: &CStr| {
            // SAFETY: `device` is live; the name is NUL-terminated.
            unsafe { instance.get_device_proc_addr(device, name.as_ptr()) }
        };
        let missing = |name: &str| format!("{name} does not resolve");
        // SAFETY: each name's signature.
        unsafe {
            Ok(Self {
                create: cast(get(c"vkCreateOpticalFlowSessionNV")).ok_or_else(|| missing("vkCreateOpticalFlowSessionNV"))?,
                destroy: cast(get(c"vkDestroyOpticalFlowSessionNV")).ok_or_else(|| missing("vkDestroyOpticalFlowSessionNV"))?,
                bind: cast(get(c"vkBindOpticalFlowSessionImageNV")).ok_or_else(|| missing("vkBindOpticalFlowSessionImageNV"))?,
                execute: cast(get(c"vkCmdOpticalFlowExecuteNV")).ok_or_else(|| missing("vkCmdOpticalFlowExecuteNV"))?,
            })
        }
    }
}

/// The device's optical flow for the server: its entry points, queue, command buffers and semaphores.
pub(crate) struct FlowDevice {
    fns: Fns,
    caps: Capabilities,
    pub(crate) queue: vk::Queue,
    pub(crate) family: u32,
    /// The server's compute family (the images are shared between the two).
    compute_family: u32,
    pool: vk::CommandPool,
    /// The flow's command buffer (optical-flow family).
    pub(crate) cmd: vk::CommandBuffer,
    /// The upload's (compute family, from the server's pool, freed with it).
    pub(crate) upload: vk::CommandBuffer,
    /// Signalled by the upload, waited for by the flow; signalled by the flow, waited for by the frame.
    pub(crate) uploaded: vk::Semaphore,
    pub(crate) flowed: vk::Semaphore,
}

impl FlowDevice {
    /// # Safety
    /// `device` is live and was created with `queue` (the optical-flow queue) and `setup`'s instance;
    /// `compute_pool` is the server's command pool of `compute_family`.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        setup: &super::native::Setup,
        queue: FlowQueue,
        compute_family: u32,
        compute_pool: vk::CommandPool,
    ) -> Result<Self, String> {
        // SAFETY: forwarded.
        let fns = unsafe { Fns::load(instance, device.handle())? };
        // SAFETY: the setup's gipa and instance are the next layer's for its physical device.
        let caps = unsafe { capabilities(setup.gipa, setup.instance, setup.physical, setup.properties2)? };
        let mut made = Self {
            fns,
            caps,
            queue: vk::Queue::null(),
            family: queue.family,
            compute_family,
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            upload: vk::CommandBuffer::null(),
            uploaded: vk::Semaphore::null(),
            flowed: vk::Semaphore::null(),
        };
        // SAFETY: valid create infos; on a failure what exists is destroyed (nothing was submitted).
        let built = unsafe {
            (|| -> Result<(), String> {
                made.queue = device.get_device_queue(queue.family, queue.index);
                crate::loader_data::initialize_queue(device.handle(), made.queue).map_err(|e| format!("its queue: {e:?}"))?;
                made.pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(queue.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .map_err(|e| format!("its command pool: {e:?}"))?;
                made.cmd =
                    crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::default().command_pool(made.pool).command_buffer_count(1))
                        .map_err(|e| format!("{e:?}"))?[0];
                made.upload =
                    crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::default().command_pool(compute_pool).command_buffer_count(1))
                        .map_err(|e| format!("{e:?}"))?[0];
                made.uploaded = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| format!("{e:?}"))?;
                made.flowed = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| format!("{e:?}"))?;
                Ok(())
            })()
        };
        match built {
            Ok(()) => Ok(made),
            Err(why) => {
                // SAFETY: nothing of it was submitted.
                unsafe { made.destroy(device) };
                Err(why)
            }
        }
    }

    /// The session plan for a frame size ([`session_plan`]).
    pub(crate) fn plan(&self, width: u32, height: u32) -> Result<(vk::OpticalFlowGridSizeFlagsNV, u32), String> {
        session_plan(&self.caps, width, height)
    }

    /// # Safety
    /// Nothing of it is pending. The upload's command buffer goes with the server's pool.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; null handles are ignored.
        unsafe {
            device.destroy_semaphore(self.uploaded, None);
            device.destroy_semaphore(self.flowed, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// An image of a session, with its memory and view.
struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl Image {
    /// # Safety
    /// `device` is live.
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
        flow_usage: vk::OpticalFlowUsageFlagsNV,
        families: &[u32],
    ) -> Result<Self, String> {
        let mut of = vk::OpticalFlowImageFormatInfoNV::default().usage(flow_usage);
        let mut info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut of);
        info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let mut out = Self { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() };
        // SAFETY: valid create infos; on a failure what exists is destroyed.
        let made = unsafe {
            (|| -> ash::prelude::VkResult<()> {
                out.image = device.create_image(&info, None)?;
                let reqs = device.get_image_memory_requirements(out.image);
                let props = instance.get_physical_device_memory_properties(physical);
                let index = (0..props.memory_type_count)
                    .find(|&i| {
                        reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                    .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
                out.memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None)?;
                device.bind_image_memory(out.image, out.memory, 0)?;
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                out.view = device.create_image_view(
                    &vk::ImageViewCreateInfo::default().image(out.image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(range),
                    None,
                )?;
                Ok(())
            })()
        };
        match made {
            Ok(()) => Ok(out),
            Err(e) => {
                // SAFETY: never used.
                unsafe { out.destroy(device) };
                Err(format!("an optical-flow image ({format:?}, {width}x{height}): {e:?}"))
            }
        }
    }

    /// # Safety
    /// Nothing pending uses it; null handles are ignored.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// One frame size's optical flow on a slot.
pub(crate) struct Session {
    session: vk::OpticalFlowSessionNV,
    width: u32,
    height: u32,
    /// The grid's size in pixels, and the flow image's extent (cells).
    pub(crate) grid: u32,
    pub(crate) cells: (u32, u32),
    inputs: [Image; 2],
    flow: Image,
    /// The input image the next frame goes into (the other holds the previous frame).
    parity: Cell<usize>,
    /// The images are still in their initial layout.
    fresh: Cell<bool>,
    /// The previous flow belongs to this sequence: its hints may be used.
    hinted: Cell<bool>,
}

impl Session {
    /// # Safety
    /// `device` is live and `fd` its.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        fd: &FlowDevice,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let (grid_flag, grid) = fd.plan(width, height)?;
        let cells = (width.div_ceil(grid), height.div_ceil(grid));
        let families: Vec<u32> = if fd.family == fd.compute_family { vec![fd.family] } else { vec![fd.compute_family, fd.family] };
        let input_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        // SAFETY: `device` is live; each image is destroyed again if a later one fails.
        unsafe {
            let a = Image::new(device, instance, physical, INPUT_FORMAT, width, height, input_usage, vk::OpticalFlowUsageFlagsNV::INPUT, &families)?;
            let b = match Image::new(device, instance, physical, INPUT_FORMAT, width, height, input_usage, vk::OpticalFlowUsageFlagsNV::INPUT, &families) {
                Ok(b) => b,
                Err(e) => {
                    a.destroy(device);
                    return Err(e);
                }
            };
            let flow = match Image::new(
                device,
                instance,
                physical,
                FLOW_FORMAT,
                cells.0,
                cells.1,
                vk::ImageUsageFlags::TRANSFER_SRC,
                vk::OpticalFlowUsageFlagsNV::OUTPUT,
                &families,
            ) {
                Ok(f) => f,
                Err(e) => {
                    a.destroy(device);
                    b.destroy(device);
                    return Err(e);
                }
            };
            let info = vk::OpticalFlowSessionCreateInfoNV {
                width,
                height,
                image_format: INPUT_FORMAT,
                flow_vector_format: FLOW_FORMAT,
                output_grid_size: grid_flag,
                performance_level: vk::OpticalFlowPerformanceLevelNV::FAST,
                ..Default::default()
            };
            let mut session = vk::OpticalFlowSessionNV::null();
            let r = (fd.fns.create)(device.handle(), &info, std::ptr::null(), &mut session);
            if r != vk::Result::SUCCESS {
                for i in [&a, &b, &flow] {
                    i.destroy(device);
                }
                return Err(format!("vkCreateOpticalFlowSessionNV ({width}x{height}, grid {grid}): {r:?}"));
            }
            Ok(Self { session, width, height, grid, cells, inputs: [a, b], flow, parity: Cell::new(0), fresh: Cell::new(true), hinted: Cell::new(false) })
        }
    }

    /// A frame without history: the next flow must not use the previous one's hints.
    pub(crate) fn restart(&self) {
        self.hinted.set(false);
    }

    /// Records the upload of `proxy` (the frame's texels, `width` x `height`) into the frame's input image,
    /// with the images' first layout change on the first frame.
    ///
    /// # Safety
    /// `cmd` is recording on the compute family; nothing pending uses the frame's input image.
    pub(crate) unsafe fn record_upload(&self, device: &ash::Device, cmd: vk::CommandBuffer, proxy: vk::Buffer) {
        let range =
            vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let general = |image: vk::Image, old: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags| {
            vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(src)
                .dst_access_mask(dst)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let target = self.inputs[self.parity.get()].image;
        // SAFETY: the caller's contract.
        unsafe {
            let barriers: Vec<vk::ImageMemoryBarrier> = if self.fresh.get() {
                [self.inputs[0].image, self.inputs[1].image, self.flow.image]
                    .into_iter()
                    .map(|i| general(i, vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE))
                    .collect()
            } else {
                // The flow that read it two frames ago finished (that frame's fence); the copy only waits for it.
                vec![general(target, vk::ImageLayout::GENERAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)]
            };
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: self.width, height: self.height, depth: 1 },
                ..Default::default()
            };
            device.cmd_copy_buffer_to_image(cmd, proxy, target, vk::ImageLayout::GENERAL, &[region]);
        }
        self.fresh.set(false);
    }

    /// Binds the frame's input, the previous frame's (the reference) and the flow image, and records the flow
    /// into `cmd` (the optical-flow family's). Then the next frame goes into the other input image.
    ///
    /// # Safety
    /// Nothing pending uses the session; `cmd` is not pending.
    pub(crate) unsafe fn record_flow(&self, device: &ash::Device, fd: &FlowDevice) -> Result<(), String> {
        let (input, reference) = (self.inputs[self.parity.get()].view, self.inputs[1 - self.parity.get()].view);
        let bind = |point, view| {
            // SAFETY: the session and views are live and not in use.
            let r = unsafe { (fd.fns.bind)(device.handle(), self.session, point, view, vk::ImageLayout::GENERAL) };
            if r == vk::Result::SUCCESS {
                Ok(())
            } else {
                Err(format!("vkBindOpticalFlowSessionImageNV: {r:?}"))
            }
        };
        bind(vk::OpticalFlowSessionBindingPointNV::INPUT, input)?;
        bind(vk::OpticalFlowSessionBindingPointNV::REFERENCE, reference)?;
        bind(vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR, self.flow.view)?;
        let flags = if self.hinted.get() { vk::OpticalFlowExecuteFlagsNV::empty() } else { vk::OpticalFlowExecuteFlagsNV::DISABLE_TEMPORAL_HINTS };
        let info = vk::OpticalFlowExecuteInfoNV { flags, ..Default::default() };
        // SAFETY: the caller's contract.
        unsafe {
            device.reset_command_buffer(fd.cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| format!("{e:?}"))?;
            device
                .begin_command_buffer(fd.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|e| format!("{e:?}"))?;
            (fd.fns.execute)(fd.cmd, self.session, &info);
            device.end_command_buffer(fd.cmd).map_err(|e| format!("{e:?}"))?;
        }
        self.hinted.set(true);
        Ok(())
    }

    /// The upload went in: the next frame's goes into the other image.
    pub(crate) fn advance(&self) {
        self.parity.set(1 - self.parity.get());
    }

    /// Records the copy of the flow vectors into `out` (cells x 4 bytes), after the flow (the frame waits for
    /// its semaphore at the transfer stage).
    ///
    /// # Safety
    /// `cmd` is recording on the compute family.
    pub(crate) unsafe fn record_readback(&self, device: &ash::Device, cmd: vk::CommandBuffer, out: vk::Buffer) {
        let region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_extent: vk::Extent3D { width: self.cells.0, height: self.cells.1, depth: 1 },
            ..Default::default()
        };
        // SAFETY: the caller's contract.
        unsafe { device.cmd_copy_image_to_buffer(cmd, self.flow.image, vk::ImageLayout::GENERAL, out, &[region]) };
    }

    /// # Safety
    /// Nothing pending uses it.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device, fd: &FlowDevice) {
        // SAFETY: forwarded.
        unsafe {
            (fd.fns.destroy)(device.handle(), self.session, std::ptr::null());
            for i in self.inputs.iter().chain(std::iter::once(&self.flow)) {
                i.destroy(device);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties { queue_flags: flags, queue_count: 1, ..Default::default() }
    }

    /// `flow` falls back to `zero` with the reason when the device lacks anything it needs.
    #[test]
    fn the_flow_is_used_only_where_the_device_has_all_of_it() {
        let families = [
            family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
            family(vk::QueueFlags::COMPUTE),
            family(vk::QueueFlags::TRANSFER | vk::QueueFlags::OPTICAL_FLOW_NV),
        ];
        let app = [vk::DeviceQueueCreateInfo::default().queue_family_index(0)];
        assert_eq!(queue_family(true, true, &families, &app), Ok(2));
        assert!(queue_family(false, true, &families, &app).unwrap_err().contains("VK_NV_optical_flow"), "no extension (lavapipe)");
        assert!(queue_family(true, false, &families, &app).unwrap_err().contains("feature"));
        assert!(queue_family(true, true, &families[..2], &app).unwrap_err().contains("no queue family"));
        let theirs = [app[0], vk::DeviceQueueCreateInfo::default().queue_family_index(2)];
        assert!(queue_family(true, true, &families, &theirs).unwrap_err().contains("itself"));

        let caps = Capabilities {
            grids: vk::OpticalFlowGridSizeFlagsNV::TYPE_1X1 | vk::OpticalFlowGridSizeFlagsNV::TYPE_2X2 | vk::OpticalFlowGridSizeFlagsNV::TYPE_4X4,
            min: (32, 32),
            max: (8192, 8192),
            inputs: vec![vk::Format::B8G8R8A8_UNORM, vk::Format::R8_UNORM],
            outputs: vec![FLOW_FORMAT],
        };
        assert_eq!(session_plan(&caps, 2560, 1440), Ok((vk::OpticalFlowGridSizeFlagsNV::TYPE_4X4, 4)), "the RTX 5070's: the 4x4 grid");
        assert!(session_plan(&caps, 16, 1440).is_err(), "too small");
        assert!(session_plan(&Capabilities { inputs: vec![vk::Format::R8_UNORM], ..caps.clone() }, 2560, 1440).is_err());
        assert!(session_plan(&Capabilities { outputs: vec![], ..caps.clone() }, 2560, 1440).is_err());
        assert!(session_plan(&Capabilities { grids: vk::OpticalFlowGridSizeFlagsNV::empty(), ..caps.clone() }, 2560, 1440).is_err());
        assert_eq!(
            grid(vk::OpticalFlowGridSizeFlagsNV::TYPE_1X1 | vk::OpticalFlowGridSizeFlagsNV::TYPE_8X8),
            Some((vk::OpticalFlowGridSizeFlagsNV::TYPE_1X1, 1))
        );
        assert_eq!(grid(vk::OpticalFlowGridSizeFlagsNV::TYPE_8X8), Some((vk::OpticalFlowGridSizeFlagsNV::TYPE_8X8, 8)));
    }

    /// The device request with the flow added keeps the base's extensions, chain and queues, and adds the
    /// extension once, the feature in front of the chain and one queue; an application's own optical-flow
    /// feature structure means no addition (a chain holds one of a type).
    #[test]
    fn the_request_adds_the_extension_the_feature_and_a_queue() {
        let names = [c"VK_KHR_swapchain".as_ptr()];
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&priorities)];
        let base = vk::DeviceCreateInfo::default().enabled_extension_names(&names).queue_create_infos(&queues).push_next(&mut sync2);
        // SAFETY: a valid request, outliving the result.
        let r = unsafe { Request::new(&base, 5) }.expect("added");
        // SAFETY: the request's arrays hold their counts.
        let (n, q) = unsafe { (std::slice::from_raw_parts(r.info.pp_enabled_extension_names, 2), std::slice::from_raw_parts(r.info.p_queue_create_infos, 2)) };
        // SAFETY: NUL-terminated names.
        assert_eq!(unsafe { (CStr::from_ptr(n[0]), CStr::from_ptr(n[1])) }, (c"VK_KHR_swapchain", EXTENSION));
        assert_eq!((q[0].queue_family_index, q[1].queue_family_index, q[1].queue_count), (0, 5, 1));
        // SAFETY: the chain is valid.
        unsafe {
            assert!(chains(&r.info, vk::StructureType::PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV));
            assert!(chains(&r.info, vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES), "the base's chain follows");
            assert!(Request::new(&r.info, 5).is_none(), "already there");
        }
    }
}
