//! Real motion vectors, estimated between consecutive proxy frames with
//! `VK_NV_optical_flow`, entirely on the GPU.
//!
//! This runs on the helper's own already-created Vulkan device -- no private device, no
//! second `vkCreateDevice` call, nothing created from inside the game's own process at a
//! swapchain-transition moment. That specific combination ("a private optical-flow device
//! created during a live game's swapchain transition") was the documented trigger for a
//! real driver crash in the old layer-side version, which was removed.
//!
//! Per frame, three submissions chained by semaphores, with no CPU copy of any frame:
//! 1. main queue: blit the model's Color image (already uploaded) down by [`DOWNSCALE`]
//!    into this frame's flow input;
//! 2. optical-flow queue: estimate flow between this input and the previous one, copy the
//!    gridded result into a buffer;
//! 3. main queue: `flow_to_mvec.comp` turns it into full-resolution R16G16_SFLOAT motion
//!    (current -> previous, deadzone, units scale), copied into the MVec image.
//!
//! The same generic `VK_NV_optical_flow` usage pattern DLSS5VKLayer's own AGPL-3.0 helper
//! uses (session creation, grid negotiation, execute, NVIDIA's documented signed-5.5
//! fixed-point vector format) -- an independent implementation, not a port (see
//! `ATTRIBUTION.md`).
//!
//! Reference: <https://docs.vulkan.org/spec/latest/chapters/VK_NV_optical_flow/optical_flow.html>
use ash::vk;

/// Whichever queue family actually supports `VK_QUEUE_OPTICAL_FLOW_BIT_NV` --
/// resolved once at helper startup (`main.rs::create_vulkan_context`) alongside the
/// main queue, since Vulkan requires every queue a device will ever use to be
/// requested at `vkCreateDevice` time. `None` whenever no such family exists, or the
/// device lacks the extensions/features optical flow needs -- every caller treats
/// that exactly like a disabled feature (no motion, not an error).
pub struct FlowQueue {
    pub family: u32,
    pub queue: vk::Queue,
}

const SPV: &[u8] = include_bytes!("../shaders/flow_to_mvec.spv");

/// Flow runs on the frame scaled down by this much per axis: a quarter of the pixels, which
/// is where most of optical flow's cost goes. Vectors are scaled back up by the shader.
pub const DOWNSCALE: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    full: [u32; 2],
    grid_dims: [u32; 2],
    to_cell: [f32; 2],
    factor: [f32; 2],
    inv_scale: [f32; 2],
}

#[derive(Clone, Copy, Default)]
struct ImageRes {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

#[derive(Clone, Copy, Default)]
struct BufferRes {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

pub struct GpuFlow {
    api: vk::NvOpticalFlowFn,
    sync: ash::extensions::khr::Synchronization2,
    flow_queue: vk::Queue,
    main_pool: vk::CommandPool,
    flow_pool: vk::CommandPool,
    cmd_pre: vk::CommandBuffer,
    cmd_flow: vk::CommandBuffer,
    cmd_post: vk::CommandBuffer,
    sem_pre: vk::Semaphore,
    sem_flow: vk::Semaphore,
    fence: vk::Fence,
    session: vk::OpticalFlowSessionNV,
    inputs: [ImageRes; 2],
    output: ImageRes,
    flow_buffer: BufferRes,
    motion_buffer: BufferRes,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pub width: u32,
    pub height: u32,
    pub quality: u32,
    flow_width: u32,
    flow_height: u32,
    grid: u32,
    current: usize,
    previous: bool,
    /// Set when a wait timed out: the GPU may still own this session's resources, so
    /// `destroy` leaks them rather than risk freeing (or waiting on) in-flight work.
    stalled: bool,
    /// Set when a submission failed part-way: the session is drained (or `stalled`) and must not
    /// be used again, only destroyed.
    dead: bool,
}
// Access is serialized by the helper's own single-threaded per-frame loop.
unsafe impl Send for GpuFlow {}

impl GpuFlow {
    /// `device`/`instance`/`pd` are the helper's own live Vulkan context; `main_family` is the
    /// family the model's images and main queue belong to, `flow_queue` the optical-flow one.
    /// `width`/`height` are the model's (full) frame size.
    #[allow(clippy::too_many_arguments)]
    pub fn new(instance: &ash::Instance, device: &ash::Device, pd: vk::PhysicalDevice, main_family: u32, flow_queue: &FlowQueue, width: u32, height: u32, quality: u32) -> Result<Self, String> {
        let api = vk::NvOpticalFlowFn::load(|name| unsafe { std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr())) });
        let mut flow = Self {
            api,
            sync: ash::extensions::khr::Synchronization2::new(instance, device),
            flow_queue: flow_queue.queue,
            main_pool: vk::CommandPool::null(),
            flow_pool: vk::CommandPool::null(),
            cmd_pre: vk::CommandBuffer::null(),
            cmd_flow: vk::CommandBuffer::null(),
            cmd_post: vk::CommandBuffer::null(),
            sem_pre: vk::Semaphore::null(),
            sem_flow: vk::Semaphore::null(),
            fence: vk::Fence::null(),
            session: vk::OpticalFlowSessionNV::null(),
            inputs: [ImageRes::default(); 2],
            output: ImageRes::default(),
            flow_buffer: BufferRes::default(),
            motion_buffer: BufferRes::default(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader: vk::ShaderModule::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            width,
            height,
            quality,
            flow_width: 0,
            flow_height: 0,
            grid: 0,
            current: 0,
            previous: false,
            stalled: false,
            dead: false,
        };
        // Every fallible step goes through `build`; on any error everything created so far
        // is destroyed here in one place (`destroy` tolerates null handles).
        match unsafe { flow.build(instance, device, pd, main_family, flow_queue.family) } {
            Ok(()) => Ok(flow),
            Err(e) => {
                unsafe { flow.destroy(device) };
                Err(e)
            }
        }
    }

    unsafe fn build(&mut self, instance: &ash::Instance, device: &ash::Device, pd: vk::PhysicalDevice, main_family: u32, flow_family: u32) -> Result<(), String> {
        let mut props = vk::PhysicalDeviceOpticalFlowPropertiesNV::default();
        instance.get_physical_device_properties2(pd, &mut vk::PhysicalDeviceProperties2::builder().push_next(&mut props));
        let fits = |w: u32, h: u32| w >= props.min_width && w <= props.max_width && h >= props.min_height && h <= props.max_height;
        // Scale down when the result still fits the session limits; otherwise run at full size.
        let (fw, fh) = if fits(self.width / DOWNSCALE, self.height / DOWNSCALE) {
            (self.width / DOWNSCALE, self.height / DOWNSCALE)
        } else if fits(self.width, self.height) {
            (self.width, self.height)
        } else {
            return Err(format!("{}x{} is outside optical flow's {}x{}..{}x{}", self.width, self.height, props.min_width, props.min_height, props.max_width, props.max_height));
        };
        self.flow_width = fw;
        self.flow_height = fh;
        self.grid = [4, 2, 1, 8].into_iter().find(|&g| props.supported_output_grid_sizes.as_raw() & g != 0).ok_or("no supported output grid size")?;
        let (grid_w, grid_h) = (fw.div_ceil(self.grid), fh.div_ceil(self.grid));

        let info = vk::OpticalFlowSessionCreateInfoNV::builder()
            .width(fw)
            .height(fh)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .flow_vector_format(vk::Format::R16G16_S10_5_NV)
            .output_grid_size(vk::OpticalFlowGridSizeFlagsNV::from_raw(self.grid))
            .performance_level(level_for(self.quality));
        (self.api.create_optical_flow_session_nv)(device.handle(), &*info, std::ptr::null(), &mut self.session).result().map_err(err("session"))?;

        let mem = instance.get_physical_device_memory_properties(pd);
        let families = [main_family, flow_family];
        // Written on the main queue, read by the flow queue: shared, so no ownership transfers.
        let shared = |concurrent: bool| if concurrent && main_family != flow_family { (vk::SharingMode::CONCURRENT, &families[..]) } else { (vk::SharingMode::EXCLUSIVE, &families[..0]) };
        for i in 0..3 {
            let output = i == 2;
            let (w, h, format, usage, (mode, qf)) = if output {
                (grid_w, grid_h, vk::Format::R16G16_S10_5_NV, vk::OpticalFlowUsageFlagsNV::OUTPUT, shared(false))
            } else {
                (fw, fh, vk::Format::B8G8R8A8_UNORM, vk::OpticalFlowUsageFlagsNV::INPUT, shared(true))
            };
            let mut of_usage = vk::OpticalFlowImageFormatInfoNV::builder().usage(usage);
            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(mode)
                .queue_family_indices(qf)
                .push_next(&mut of_usage);
            let image = device.create_image(&image_info, None).map_err(err("flow image"))?;
            let slot = if output { &mut self.output } else { &mut self.inputs[i] };
            slot.image = image;
            let req = device.get_image_memory_requirements(image);
            slot.memory = device
                .allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(memory_type(&mem, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?), None)
                .map_err(err("flow image memory"))?;
            device.bind_image_memory(image, slot.memory, 0).map_err(err("bind flow image"))?;
            slot.view = device
                .create_image_view(&vk::ImageViewCreateInfo::builder().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(subresource()), None)
                .map_err(err("flow image view"))?;
        }

        let flow_bytes = u64::from(grid_w) * u64::from(grid_h) * 4;
        let motion_bytes = u64::from(self.width) * u64::from(self.height) * 4;
        let (mode, qf) = shared(true);
        self.flow_buffer = make_buffer(device, &mem, flow_bytes, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER, mode, qf)?;
        self.motion_buffer = make_buffer(device, &mem, motion_bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, vk::SharingMode::EXCLUSIVE, &[])?;

        let pool = |family| vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.main_pool = device.create_command_pool(&pool(main_family), None).map_err(err("main pool"))?;
        self.flow_pool = device.create_command_pool(&pool(flow_family), None).map_err(err("flow pool"))?;
        let alloc = |pool, n| vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(n);
        let main_cmds = device.allocate_command_buffers(&alloc(self.main_pool, 2)).map_err(err("main command buffers"))?;
        (self.cmd_pre, self.cmd_post) = (main_cmds[0], main_cmds[1]);
        self.cmd_flow = device.allocate_command_buffers(&alloc(self.flow_pool, 1)).map_err(err("flow command buffer"))?[0];
        self.sem_pre = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(err("semaphore"))?;
        self.sem_flow = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(err("semaphore"))?;
        self.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(err("fence"))?;

        let words = ash::util::read_spv(&mut std::io::Cursor::new(SPV)).map_err(|e| format!("shader: {e}"))?;
        self.shader = device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&words), None).map_err(err("shader module"))?;
        let bindings = [0, 1].map(|b| {
            vk::DescriptorSetLayoutBinding::builder().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build()
        });
        self.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).map_err(err("set layout"))?;
        let ranges = [vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<Push>() as u32).build()];
        let layouts = [self.set_layout];
        self.pipeline_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts).push_constant_ranges(&ranges), None).map_err(err("pipeline layout"))?;
        let entry = std::ffi::CString::new("main").expect("static name");
        let stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::COMPUTE).module(self.shader).name(&entry);
        let pipeline_info = [vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(self.pipeline_layout).build()];
        self.pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None).map_err(|(_, e)| format!("pipeline: {e:?}"))?[0];
        let sizes = [vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(2).build()];
        self.descriptor_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).map_err(err("descriptor pool"))?;
        self.descriptor_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(self.descriptor_pool).set_layouts(&layouts)).map_err(err("descriptor set"))?[0];
        let infos = [(self.flow_buffer.buffer, flow_bytes), (self.motion_buffer.buffer, motion_bytes)]
            .map(|(buffer, range)| [vk::DescriptorBufferInfo::builder().buffer(buffer).range(range).build()]);
        let writes = [0usize, 1].map(|i| {
            vk::WriteDescriptorSet::builder().dst_set(self.descriptor_set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&infos[i]).build()
        });
        device.update_descriptor_sets(&writes, &[]);
        Ok(())
    }

    /// Forget the previous frame (a scene cut): the next call only seeds history.
    pub fn reset(&mut self) {
        self.previous = false;
    }

    /// Estimates motion into `mvec` (the model's MVec image, `width`x`height`, R16G16_SFLOAT)
    /// from `color` (the model's Color image, same size). Both must be in
    /// `SHADER_READ_ONLY_OPTIMAL` and idle on `main_queue`, and are left that way. `scale` is
    /// the units scale from `neural_forge_protocol::motion::scales`. `Ok(false)` on the first
    /// frame after creation or `reset` (history seeded, `mvec` untouched), `Ok(true)` when
    /// `mvec` now holds this frame's motion.
    pub fn estimate(&mut self, device: &ash::Device, main_queue: vk::Queue, color: vk::Image, mvec: vk::Image, scale: [f32; 2]) -> Result<bool, vk::Result> {
        if self.stalled {
            return Err(vk::Result::TIMEOUT);
        }
        if self.dead {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        let input = self.inputs[self.current];
        unsafe {
            // 1. Main queue: scale the model's input down into this frame's flow input.
            begin(device, self.cmd_pre)?;
            self.barrier(self.cmd_pre, color, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            self.barrier(self.cmd_pre, input.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            let blit = vk::ImageBlit::builder()
                .src_subresource(layers())
                .src_offsets([vk::Offset3D::default(), offset(self.width, self.height)])
                .dst_subresource(layers())
                .dst_offsets([vk::Offset3D::default(), offset(self.flow_width, self.flow_height)])
                .build();
            device.cmd_blit_image(self.cmd_pre, color, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, input.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
            self.barrier(self.cmd_pre, input.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
            self.barrier(self.cmd_pre, color, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            device.end_command_buffer(self.cmd_pre)?;
            let pre = [self.cmd_pre];
            if !self.previous {
                // Seed frame: nothing to compare against yet.
                device.reset_fences(&[self.fence])?;
                device.queue_submit(main_queue, &[vk::SubmitInfo::builder().command_buffers(&pre).build()], self.fence)?;
                self.wait(device)?;
                self.previous = true;
                self.current = 1 - self.current;
                return Ok(false);
            }
            // From the first semaphore-signalling submit on, a failure leaves work (and a signalled
            // `sem_pre`) behind: drain both queues before reporting it, so nothing this session owns
            // is still in use when the caller drops it.
            if let Err(e) = self.submit_pair(device, main_queue, input, mvec, scale) {
                if !self.stalled {
                    let drained = device.queue_wait_idle(main_queue).is_ok() && device.queue_wait_idle(self.flow_queue).is_ok();
                    if !drained {
                        // Could not prove the queues idle: never touch or free this session again.
                        self.stalled = true;
                    }
                }
                self.dead = true;
                crate::log!("[mvec] optical-flow submission failed ({e:?}); session closed");
                return Err(e);
            }
        }
        self.current = 1 - self.current;
        Ok(true)
    }

    /// Steps 1b-3 of [`Self::estimate`]: everything from the first submit that signals `sem_pre`
    /// to the wait on the final fence.
    unsafe fn submit_pair(&mut self, device: &ash::Device, main_queue: vk::Queue, input: ImageRes, mvec: vk::Image, scale: [f32; 2]) -> Result<(), vk::Result> {
        let pre = [self.cmd_pre];
        unsafe {
            let signal_pre = [self.sem_pre];
            device.queue_submit(main_queue, &[vk::SubmitInfo::builder().command_buffers(&pre).signal_semaphores(&signal_pre).build()], vk::Fence::null())?;

            // 2. Flow queue: estimate between this input and the previous one.
            for (binding, view) in [
                (vk::OpticalFlowSessionBindingPointNV::INPUT, input.view),
                (vk::OpticalFlowSessionBindingPointNV::REFERENCE, self.inputs[1 - self.current].view),
                (vk::OpticalFlowSessionBindingPointNV::FLOW_VECTOR, self.output.view),
            ] {
                (self.api.bind_optical_flow_session_image_nv)(device.handle(), self.session, binding, view, vk::ImageLayout::GENERAL).result()?;
            }
            begin(device, self.cmd_flow)?;
            self.barrier(self.cmd_flow, self.output.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            // Captures may be several presents apart: disable implicit temporal hints.
            let execute = vk::OpticalFlowExecuteInfoNV::builder().flags(vk::OpticalFlowExecuteFlagsNV::DISABLE_TEMPORAL_HINTS);
            (self.api.cmd_optical_flow_execute_nv)(self.cmd_flow, self.session, &*execute);
            self.barrier(self.cmd_flow, self.output.image, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let (grid_w, grid_h) = (self.flow_width.div_ceil(self.grid), self.flow_height.div_ceil(self.grid));
            device.cmd_copy_image_to_buffer(self.cmd_flow, self.output.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.flow_buffer.buffer, &[region(grid_w, grid_h)]);
            device.end_command_buffer(self.cmd_flow)?;
            let flow_cmds = [self.cmd_flow];
            let signal_flow = [self.sem_flow];
            let wait_stage = [vk::PipelineStageFlags::ALL_COMMANDS];
            device.queue_submit(
                self.flow_queue,
                &[vk::SubmitInfo::builder().wait_semaphores(&signal_pre).wait_dst_stage_mask(&wait_stage).command_buffers(&flow_cmds).signal_semaphores(&signal_flow).build()],
                vk::Fence::null(),
            )?;

            // 3. Main queue: flow cells -> full-resolution motion -> MVec image.
            begin(device, self.cmd_post)?;
            device.cmd_bind_pipeline(self.cmd_post, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(self.cmd_post, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.descriptor_set], &[]);
            let push = Push {
                full: [self.width, self.height],
                grid_dims: [grid_w, grid_h],
                to_cell: [
                    self.flow_width as f32 / self.width as f32 / self.grid as f32,
                    self.flow_height as f32 / self.height as f32 / self.grid as f32,
                ],
                factor: [self.width as f32 / self.flow_width as f32, self.height as f32 / self.flow_height as f32],
                inv_scale: [1.0 / scale[0].max(1.0), 1.0 / scale[1].max(1.0)],
            };
            let bytes = std::slice::from_raw_parts((&push as *const Push).cast::<u8>(), std::mem::size_of::<Push>());
            device.cmd_push_constants(self.cmd_post, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_dispatch(self.cmd_post, self.width.div_ceil(16), self.height.div_ceil(16), 1);
            let written = [vk::MemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .build()];
            self.sync.cmd_pipeline_barrier2(self.cmd_post, &vk::DependencyInfo::builder().memory_barriers(&written));
            self.barrier(self.cmd_post, mvec, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_copy_buffer_to_image(self.cmd_post, self.motion_buffer.buffer, mvec, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(self.width, self.height)]);
            self.barrier(self.cmd_post, mvec, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            device.end_command_buffer(self.cmd_post)?;
            let post = [self.cmd_post];
            let post_stage = [vk::PipelineStageFlags::COMPUTE_SHADER];
            device.reset_fences(&[self.fence])?;
            device.queue_submit(
                main_queue,
                &[vk::SubmitInfo::builder().wait_semaphores(&signal_flow).wait_dst_stage_mask(&post_stage).command_buffers(&post).build()],
                self.fence,
            )?;
            // The post submission waits on the flow one, which waits on the pre one, so its
            // fence covers all three.
            self.wait(device)
        }
    }

    /// Bounded, like every other GPU wait in the helper: the layer waits on the helper every
    /// frame, so an unbounded wait here would freeze the game on a driver stall.
    unsafe fn wait(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        match device.wait_for_fences(&[self.fence], true, crate::ngx::FENCE_WAIT_TIMEOUT.as_nanos() as u64) {
            Ok(()) => Ok(()),
            Err(e) => {
                if e == vk::Result::TIMEOUT {
                    self.stalled = true;
                    crate::log!("[mvec] optical-flow fence wait timed out after {:?}; motion vectors off for this session", crate::ngx::FENCE_WAIT_TIMEOUT);
                }
                Err(e)
            }
        }
    }

    unsafe fn barrier(&self, cmd: vk::CommandBuffer, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
        let barriers = [vk::ImageMemoryBarrier2::builder()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource())
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .build()];
        self.sync.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::builder().image_memory_barriers(&barriers));
    }

    /// # Safety
    /// `device` must be the same live device this was created against, and no work of this
    /// session may be in flight (every `estimate` waits for its own work before returning).
    pub unsafe fn destroy(&self, device: &ash::Device) {
        if self.stalled {
            return;
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.shader, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.sem_pre, None);
            device.destroy_semaphore(self.sem_flow, None);
            device.destroy_command_pool(self.main_pool, None);
            device.destroy_command_pool(self.flow_pool, None);
            if self.session != vk::OpticalFlowSessionNV::null() {
                (self.api.destroy_optical_flow_session_nv)(device.handle(), self.session, std::ptr::null());
            }
            for res in self.inputs.iter().chain(std::iter::once(&self.output)) {
                device.destroy_image_view(res.view, None);
                device.destroy_image(res.image, None);
                device.free_memory(res.memory, None);
            }
            for res in [self.flow_buffer, self.motion_buffer] {
                device.destroy_buffer(res.buffer, None);
                device.free_memory(res.memory, None);
            }
        }
    }
}

/// A coarse luma thumbnail of a BGRA8/RGBA8 frame (every 8th pixel on each axis), kept
/// between frames for [`is_scene_cut`] instead of a copy of the whole frame.
pub fn luma_thumbnail(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    const STEP: usize = 8;
    let (w, h) = (width as usize, height as usize);
    if frame.len() < w * h * 4 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(w.div_ceil(STEP) * h.div_ceil(STEP));
    for y in (0..h).step_by(STEP) {
        for x in (0..w).step_by(STEP) {
            let i = (y * w + x) * 4;
            // Unweighted average of the three channels: only needs to catch "the whole picture
            // changed", and channel order does not matter for it.
            out.push(((u32::from(frame[i]) + u32::from(frame[i + 1]) + u32::from(frame[i + 2])) / 3) as u8);
        }
    }
    out
}

/// Whether two [`luma_thumbnail`]s look like different scenes -- a hard cut (level
/// transition, cutscene, death/respawn) rather than motion within one scene. Carrying a flow
/// field across a cut hands the model a field describing content no longer on screen, worse
/// than handing it nothing. DLSS5VKLayer's helper runs the same kind of check
/// (`DetectSceneCut`); this is an independent implementation of the generic technique (mean
/// luma delta against a threshold).
pub fn is_scene_cut(previous: &[u8], current: &[u8], threshold: u8) -> bool {
    if previous.len() != current.len() || current.is_empty() {
        return false;
    }
    let sum: u64 = previous.iter().zip(current).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
    sum / current.len() as u64 >= u64::from(threshold)
}

/// `quality` is [`neural_forge_protocol::enums::mvec_quality`]'s raw value.
fn level_for(quality: u32) -> vk::OpticalFlowPerformanceLevelNV {
    match quality {
        neural_forge_protocol::enums::mvec_quality::FAST => vk::OpticalFlowPerformanceLevelNV::FAST,
        neural_forge_protocol::enums::mvec_quality::QUALITY => vk::OpticalFlowPerformanceLevelNV::SLOW,
        _ => vk::OpticalFlowPerformanceLevelNV::MEDIUM,
    }
}

/// `map_err` helper: a Vulkan error with what was being created.
fn err(what: &'static str) -> impl Fn(vk::Result) -> String {
    move |e| format!("{what}: {e:?}")
}

unsafe fn begin(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), vk::Result> {
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
}

unsafe fn make_buffer(device: &ash::Device, mem: &vk::PhysicalDeviceMemoryProperties, size: u64, usage: vk::BufferUsageFlags, mode: vk::SharingMode, families: &[u32]) -> Result<BufferRes, String> {
    let buffer = device
        .create_buffer(&vk::BufferCreateInfo::builder().size(size).usage(usage).sharing_mode(mode).queue_family_indices(families), None)
        .map_err(|e| format!("buffer: {e:?}"))?;
    let req = device.get_buffer_memory_requirements(buffer);
    let memory = match memory_type(mem, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .and_then(|index| device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(index), None).map_err(|e| format!("buffer memory: {e:?}")))
    {
        Ok(memory) => memory,
        Err(e) => {
            device.destroy_buffer(buffer, None);
            return Err(e);
        }
    };
    if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        return Err(format!("bind buffer: {e:?}"));
    }
    Ok(BufferRes { buffer, memory })
}

fn memory_type(mem: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
    (0..mem.memory_type_count)
        .find(|&i| bits & (1 << i) != 0 && mem.memory_types[i as usize].property_flags.contains(flags))
        .ok_or_else(|| format!("no memory type with {flags:?}"))
}

fn subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1).build()
}

fn layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).build()
}

fn offset(width: u32, height: u32) -> vk::Offset3D {
    vk::Offset3D { x: width as i32, y: height as i32, z: 1 }
}

fn region(width: u32, height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy::builder().image_subresource(layers()).image_extent(vk::Extent3D { width, height, depth: 1 }).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    #[test]
    fn push_constants_match_the_shader_block() {
        // uvec2 full, uvec2 grid_dims, vec2 to_cell, vec2 factor, vec2 inv_scale (std430 push).
        assert_eq!(std::mem::size_of::<Push>(), 40);
    }

    #[test]
    fn thumbnail_samples_every_eighth_pixel_per_axis() {
        assert_eq!(luma_thumbnail(&frame(16, 16, 90), 16, 16), vec![90; 4]);
        assert_eq!(luma_thumbnail(&frame(17, 9, 90), 17, 9).len(), 3 * 2);
        assert!(luma_thumbnail(&[1, 2, 3], 16, 16).is_empty(), "a short frame must not panic");
    }

    #[test]
    fn scene_cut_is_not_flagged_for_a_stable_or_slightly_changed_frame() {
        let a = luma_thumbnail(&frame(32, 32, 100), 32, 32);
        let b = luma_thumbnail(&frame(32, 32, 105), 32, 32);
        assert!(!is_scene_cut(&a, &a, 40), "identical frames must never be a cut");
        assert!(!is_scene_cut(&a, &b, 40), "a small uniform change must not be a cut");
    }

    #[test]
    fn scene_cut_is_flagged_for_a_completely_different_frame() {
        let a = luma_thumbnail(&frame(32, 32, 20), 32, 32);
        let b = luma_thumbnail(&frame(32, 32, 220), 32, 32);
        assert!(is_scene_cut(&a, &b, 40));
    }

    #[test]
    fn scene_cut_never_panics_on_mismatched_or_empty_thumbnails() {
        assert!(!is_scene_cut(&[], &[], 40));
        assert!(!is_scene_cut(&[1, 2], &[1, 2, 3], 40));
    }
}
