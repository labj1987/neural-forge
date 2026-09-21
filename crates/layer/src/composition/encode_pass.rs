//! The GPU half of the proxy encode: the compute pipeline that runs
//! [`super::encode`]'s curves over the captured frame before it crosses to the helper.
//!
//! Deliberately its own pipeline rather than a mode of [`super::gpu::GpuCompose`]:
//! this runs on the *capture* leg (one dispatch over the model's raster, in place,
//! between the blit that fills the scratch image and the copy that downloads it),
//! while `GpuCompose` runs on the compose leg with a four-image descriptor set. They
//! share nothing but the device.
//!
//! Every failure path here is fail-open: `None`/`false` means "the proxy goes to the
//! helper unencoded, exactly as it did before this existed", never a reason to stop
//! presenting. The resolve is told which it got (see `compose.comp`'s `mode`), so an
//! unencoded proxy keeps the old composition math rather than being fed to one that
//! assumes an encode ran.

use ash::vk;

const SPV: &[u8] = include_bytes!("../../shaders/encode.spv");

/// Matches `encode.comp`'s `Params` block exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EncodePush {
    /// What the model should treat as white. The resolve undoes this divide.
    pub white_point: f32,
    /// Whether the bytes in the scratch image are B,G,R,A rather than canonical
    /// R,G,B,A -- the image is viewed as `R8G8B8A8_UNORM` regardless (so the shader's
    /// `rgba8` qualifier is spec-correct against the view), and this says what the
    /// channels actually mean. Same contract as `compose.comp`'s own `bgr_order`.
    pub bgr_order: u32,
    /// `neural_forge_protocol::enums::reversible_mode`.
    pub reversible_mode: u32,
}

/// The size-independent pipeline state. One per device; the per-image descriptor sets
/// are allocated from here by [`EncodePass::allocate_set`] and owned by the caller's
/// own scratch resources (freed with [`EncodePass::free_set`]).
pub struct EncodePass {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
}

// SAFETY: plain Vulkan handles, never aliased outside the lock its owner lives behind
// -- the same reasoning `GpuCompose`'s own impl carries.
unsafe impl Send for EncodePass {}

impl EncodePass {
    /// `None` on any failure, which the caller treats as "no encode available".
    pub fn new(device: &ash::Device) -> Option<Self> {
        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(std::slice::from_ref(&binding));
        // SAFETY: `layout_info` is valid.
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }.ok()?;

        let push_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<EncodePush>() as u32)
            .build();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));
        // SAFETY: valid info; `descriptor_set_layout` was just created.
        let pipeline_layout = match unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) } {
            Ok(l) => l,
            Err(_) => {
                // SAFETY: nothing references it yet.
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return None;
            }
        };

        let cleanup_layouts = |device: &ash::Device| {
            // SAFETY: neither handle is referenced anywhere else yet.
            unsafe {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
        };

        let Ok(code) = ash::util::read_spv(&mut std::io::Cursor::new(SPV)) else {
            cleanup_layouts(device);
            return None;
        };
        let module_info = vk::ShaderModuleCreateInfo::builder().code(&code);
        // SAFETY: `encode.spv` is a valid SPIR-V module, compiled from
        // `shaders/encode.comp` with `glslangValidator -V`.
        let module = match unsafe { device.create_shader_module(&module_info, None) } {
            Ok(m) => m,
            Err(_) => {
                cleanup_layouts(device);
                return None;
            }
        };
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry)
            .build();
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(pipeline_layout).build();
        // SAFETY: `pipeline_info` is valid; `module`/`pipeline_layout` outlive the call.
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None) };
        // SAFETY: the module is only needed for pipeline creation, which has returned.
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = match pipeline {
            Ok(p) => p[0],
            Err(_) => {
                cleanup_layouts(device);
                return None;
            }
        };

        // Sized for both wire slots plus headroom for rebuilds (a resolution change
        // frees and reallocates). `FREE_DESCRIPTOR_SET` so a rebuild can hand its set
        // back rather than leaking one per resize.
        let pool_size = vk::DescriptorPoolSize::builder().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(16).build();
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(16)
            .pool_sizes(std::slice::from_ref(&pool_size));
        // SAFETY: `pool_info` is valid.
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(_) => {
                // SAFETY: `pipeline` is not referenced anywhere else yet.
                unsafe { device.destroy_pipeline(pipeline, None) };
                cleanup_layouts(device);
                return None;
            }
        };

        Some(Self { descriptor_set_layout, pipeline_layout, pipeline, descriptor_pool })
    }

    /// One set bound to `view`, which must be an `R8G8B8A8_UNORM` view of the scratch
    /// image (see [`EncodePush::bgr_order`] for why the view's format is fixed even
    /// when the image's is BGRA).
    pub fn allocate_set(&self, device: &ash::Device, view: vk::ImageView) -> Option<vk::DescriptorSet> {
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        // SAFETY: `alloc_info` names this pass's own pool and layout.
        let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }.ok()?.first().copied()?;
        let image_info = vk::DescriptorImageInfo::builder().image_view(view).image_layout(vk::ImageLayout::GENERAL).build();
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info))
            .build();
        // SAFETY: `set` was just allocated; `view` outlives it (the caller owns both and
        // frees the set before destroying the view).
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        Some(set)
    }

    /// # Safety
    /// `set` must have come from this pass's own [`Self::allocate_set`], and nothing
    /// submitted against it may still be in flight.
    pub unsafe fn free_set(&self, device: &ash::Device, set: vk::DescriptorSet) {
        let _ = device.free_descriptor_sets(self.descriptor_pool, std::slice::from_ref(&set));
    }

    /// Records the dispatch into `cmd`, which must be recording, with the scratch image
    /// already in [`vk::ImageLayout::GENERAL`]. In place: every invocation reads and
    /// writes only its own pixel, so there is no hazard within the dispatch.
    ///
    /// # Safety
    /// `cmd` is recording; `set` is this pass's own and points at an image of exactly
    /// `width`x`height`.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        width: u32,
        height: u32,
        push: EncodePush,
    ) {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&set), &[]);
        let bytes = std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<EncodePush>());
        device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
        // `encode.comp`'s own `local_size`, and its bounds check covers the remainder.
        device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
    }

    /// # Safety
    /// Nothing submitted against this pass may still be in flight, and every set
    /// allocated from it must already be freed or unused.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

/// Whether `format` can actually be used as a storage image on this device, which is
/// what the encode dispatch needs. Not guaranteed for `B8G8R8A8_UNORM` by the spec
/// (only a short mandatory list is), so it is asked rather than assumed: `false` means
/// the caller leaves the proxy unencoded and the resolve keeps its old math.
pub fn supports_storage(instance: &ash::Instance, physical_device: vk::PhysicalDevice, format: vk::Format) -> bool {
    // SAFETY: `physical_device` came from this instance's own enumeration.
    let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    props.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
}
