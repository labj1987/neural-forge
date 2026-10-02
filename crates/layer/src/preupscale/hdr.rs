//! The pre-upscaler path's HDR encode and its inverse, on the GPU (docs/PRE_UPSCALER_DESIGN.md,
//! "E1b: the HDR encode" and "Implementation (layer)").
//!
//! The model treats its input as a display-referred [0, 1] picture whatever it is told (E1b), so
//! DLSS's scene-linear colour input is encoded before it is sent (`shaders/preupscale_encode.comp`:
//! the game's exposure multiplied in, divided by the paper white, the per-channel shoulder, sRGB)
//! and the answer is decoded on the way back (`shaders/preupscale_decode.comp`: the exact inverse,
//! keeping the original value where the encoded input was clamped at the shoulder's top).
//!
//! One [`HdrPass`] per [`super::Resources`] (one extent): two compute pipelines over one descriptor
//! set (the colour input's view, the padded encoded and answer images, the exposure buffer), built
//! on first use in a mode that needs it. Every failure is fail-open: the caller forwards the frame
//! untouched.

use ash::vk;

use super::HostBuffer;

const ENCODE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_encode.spv");
const DECODE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_decode.spv");

/// The variable that overrides the paper white (exposed units mapped to 1.0 before the shoulder).
pub(crate) const PAPER_WHITE_ENV: &str = "NEURAL_FORGE_PREUPSCALE_PAPER_WHITE";

/// E1b's paper white: the exposed median lands at 0.16-0.25 and under 1% of pixels clamp.
pub(crate) const DEFAULT_PAPER_WHITE: f32 = 3.0;

/// [`HdrPush::flags`] bit: keep the original scene value of a channel whose encoded input was at
/// the shoulder's top (>= 0.999), where the inverse cannot bring it back.
pub(crate) const KEEP_CLAMPED: u32 = 1;

/// The encoded value at and above which a channel counts as clamped (the decode keeps it).
#[cfg(test)]
pub(crate) const CLAMPED: f32 = 0.999;

/// Bytes of the exposure buffer (one R16_SFLOAT texel at offset 0, the rest padding).
const EXPOSURE_BYTES: u64 = 16;

/// The shaders' `local_size`.
const GROUP: u32 = 8;

/// The push constants of both `preupscale_encode.comp` and `preupscale_decode.comp`: their
/// `Params` blocks match this field for field (append only, in all three places).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HdrPush {
    /// The colour input's extent (`uvec2 size`).
    pub width: u32,
    pub height: u32,
    /// The proxy's padded extent (`uvec2 padded`).
    pub padded_width: u32,
    pub padded_height: u32,
    /// `float paper_white`.
    pub paper_white: f32,
    /// `uint flags`: [`KEEP_CLAMPED`].
    pub flags: u32,
}

/// `value` as a paper white: a finite number above 0 (up to 1000). `None` (unset or empty) is the
/// default; anything else unusable is an error.
pub(crate) fn parse_paper_white(value: Option<&str>) -> Result<f32, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(DEFAULT_PAPER_WHITE),
        Some(v) => match v.parse::<f32>() {
            Ok(w) if w.is_finite() && w > 0.0 && w <= 1000.0 => Ok(w),
            _ => Err(v.to_string()),
        },
    }
}

/// The paper white for this process ([`PAPER_WHITE_ENV`], default [`DEFAULT_PAPER_WHITE`]).
/// Cached; an unusable value is logged once and the default used.
pub(crate) fn paper_white() -> f32 {
    static WHITE: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        match parse_paper_white(neural_forge_protocol::env::var(PAPER_WHITE_ENV).as_deref()) {
            Ok(w) => {
                crate::log!("[preupscale] HDR encode: paper white {w} (exposure multiplied in, shoulder above 0.75, sRGB)");
                w
            }
            Err(other) => {
                crate::log!("[preupscale] {PAPER_WHITE_ENV}={other:?} is not a positive number; using {DEFAULT_PAPER_WHITE}");
                DEFAULT_PAPER_WHITE
            }
        }
    });
    *WHITE
}

/// Whether an exposure value can be divided by: finite and clearly positive (the shaders' own
/// `exposure_ok`).
pub(crate) fn exposure_ok(e: f32) -> bool {
    e > 1e-6 && e < 65504.0
}

/// A device-local RGBA16F storage image of the layer's own, with its view.
struct OwnImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl OwnImage {
    /// # Safety
    /// `device` is live.
    unsafe fn new(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32, usage: vk::ImageUsageFlags) -> Option<Self> {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info.
        let image = unsafe { device.create_image(&info, None) }.ok()?;
        // SAFETY: just created.
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        // SAFETY: plain property query.
        let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let pick = |want: vk::MemoryPropertyFlags| {
            (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want))
        };
        let memory = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL).or_else(|| pick(vk::MemoryPropertyFlags::empty())).and_then(|index| {
            let alloc = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index);
            // SAFETY: valid allocation info.
            unsafe { device.allocate_memory(&alloc, None) }.ok()
        });
        let Some(memory) = memory else {
            // SAFETY: nothing bound or submitted.
            unsafe { device.destroy_image(image, None) };
            return None;
        };
        // SAFETY: sized for each other by construction.
        let view = unsafe { device.bind_image_memory(image, memory, 0) }.ok().and_then(|()| {
            // SAFETY: `image` is bound and live.
            unsafe { colour_view(device, image) }
        });
        let Some(view) = view else {
            // SAFETY: nothing submitted uses either.
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return None;
        };
        Some(Self { image, memory, view })
    }

    /// # Safety
    /// Nothing submitted may still use it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// A plain 2D RGBA16F view of `image`'s first mip and layer.
///
/// # Safety
/// `image` is a live, plain 2D RGBA16F image of `device`.
unsafe fn colour_view(device: &ash::Device, image: vk::Image) -> Option<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    // SAFETY: forwarded.
    unsafe { device.create_image_view(&info, None) }.ok()
}

/// The encode and decode pipelines and what they work on, for one extent.
pub(crate) struct HdrPass {
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    encode: vk::Pipeline,
    decode: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// What the capture sent (padded): copied to the proxy region, read by the decode.
    encoded: OwnImage,
    /// The answer, copied in from the answer region (padded).
    answer: OwnImage,
    /// The exposure texel, copied in by the capture; host-visible so the CPU checks it.
    exposure: HostBuffer,
    /// The view of the colour input bound at the last [`Self::bind_colour`]. Re-created every hold
    /// (after the previous hold's work drained), so a destroyed and re-created image with a reused
    /// handle is never reached through a stale view.
    colour_view: Option<vk::ImageView>,
}

// SAFETY: plain handles and the layer's own mapped allocation, only used behind the device's
// `State` mutex (as `Resources`).
unsafe impl Send for HdrPass {}

impl HdrPass {
    /// `None` on any failure (the caller then forwards frames untouched).
    ///
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32) -> Option<Self> {
        let (padded_width, padded_height) = super::padded(width, height);
        let binding = |n: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build()
        };
        let bindings = [
            binding(0, vk::DescriptorType::STORAGE_IMAGE),
            binding(1, vk::DescriptorType::STORAGE_IMAGE),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
            binding(3, vk::DescriptorType::STORAGE_IMAGE),
        ];
        // Built piece by piece; `Partial` destroys whatever exists on an early return.
        let mut p = Partial::default_for(device);
        // SAFETY: valid create infos throughout; every handle created here is owned by `p` until
        // the end, where it moves into the result.
        unsafe {
            p.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<HdrPush>() as u32).build();
            p.pipeline_layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&p.set_layout)).push_constant_ranges(std::slice::from_ref(&push)), None)
                .ok()?;
            p.encode = compute_pipeline(device, p.pipeline_layout, ENCODE_SPV)?;
            p.decode = compute_pipeline(device, p.pipeline_layout, DECODE_SPV)?;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 3 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 },
            ];
            p.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).ok()?;
            p.set = device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(p.pool).set_layouts(std::slice::from_ref(&p.set_layout)))
                .ok()?
                .first()
                .copied()?;
            p.encoded = Some(OwnImage::new(device, instance, physical_device, padded_width, padded_height, vk::ImageUsageFlags::TRANSFER_SRC)?);
            p.answer = Some(OwnImage::new(device, instance, physical_device, padded_width, padded_height, vk::ImageUsageFlags::TRANSFER_DST)?);
            p.exposure = Some(super::own_host_buffer_with(
                device,
                instance,
                physical_device,
                EXPOSURE_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?);
        }
        let (encoded, answer, exposure) = (p.encoded.take()?, p.answer.take()?, p.exposure.take()?);
        // SAFETY: the mapping covers EXPOSURE_BYTES; a defined starting value (0: "no exposure").
        unsafe { std::ptr::write_bytes(exposure.ptr, 0, EXPOSURE_BYTES as usize) };
        let image_info = |view: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
        let (enc_info, ans_info) = (image_info(encoded.view), image_info(answer.view));
        let buf_info = [vk::DescriptorBufferInfo { buffer: exposure.buffer, offset: 0, range: EXPOSURE_BYTES }];
        let writes = [
            vk::WriteDescriptorSet::builder().dst_set(p.set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&enc_info).build(),
            vk::WriteDescriptorSet::builder().dst_set(p.set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&buf_info).build(),
            vk::WriteDescriptorSet::builder().dst_set(p.set).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&ans_info).build(),
        ];
        // SAFETY: the set is not in use; every view and buffer is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let pass = Self {
            width,
            height,
            padded_width,
            padded_height,
            set_layout: std::mem::take(&mut p.set_layout),
            pipeline_layout: std::mem::take(&mut p.pipeline_layout),
            encode: std::mem::take(&mut p.encode),
            decode: std::mem::take(&mut p.decode),
            pool: std::mem::take(&mut p.pool),
            set: p.set,
            encoded,
            answer,
            exposure,
            colour_view: None,
        };
        Some(pass)
    }

    /// The exposure buffer (the capture copies the texel to offset 0).
    pub(crate) fn exposure_buffer(&self) -> vk::Buffer {
        self.exposure.buffer
    }

    /// The exposure value the last capture read (call after its fence).
    pub(crate) fn exposure_value(&self) -> f32 {
        // SAFETY: the mapping covers EXPOSURE_BYTES and is host-coherent; the capture's fence was
        // waited on and its closing barrier made the transfer write available to the host.
        let bits = unsafe { std::ptr::read_volatile(self.exposure.ptr.cast::<[u8; 2]>()) };
        super::f16_to_f32(u16::from_le_bytes(bits))
    }

    /// Clears the exposure buffer to 0 (so a capture that did not read the exposure image leaves
    /// "unusable" behind, never the previous frame's value).
    pub(crate) fn clear_exposure(&self) {
        // SAFETY: as above; nothing pending uses the buffer (called after the hold's `wait_idle`).
        unsafe { std::ptr::write_bytes(self.exposure.ptr, 0, EXPOSURE_BYTES as usize) };
    }

    /// Points binding 0 at a fresh view of `colour`, destroying the previous one.
    ///
    /// # Safety
    /// Nothing submitted may still use the set or the previous view (the hold's `wait_idle` passed);
    /// `colour` is a live plain 2D RGBA16F storage image of `device`.
    pub(crate) unsafe fn bind_colour(&mut self, device: &ash::Device, colour: vk::Image) -> bool {
        if let Some(old) = self.colour_view.take() {
            // SAFETY: nothing pending uses it (contract).
            unsafe { device.destroy_image_view(old, None) };
        }
        // SAFETY: forwarded.
        let Some(view) = (unsafe { colour_view(device, colour) }) else {
            return false;
        };
        self.colour_view = Some(view);
        let info = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
        let write = vk::WriteDescriptorSet::builder().dst_set(self.set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&info).build();
        // SAFETY: the set is not in use (contract).
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        true
    }

    fn push(&self, paper_white: f32) -> HdrPush {
        HdrPush { width: self.width, height: self.height, padded_width: self.padded_width, padded_height: self.padded_height, paper_white, flags: KEEP_CLAMPED }
    }

    /// A barrier taking the encoded image from anything (its contents are rewritten) to `GENERAL`
    /// for the encode's writes. For the capture's opening barrier.
    pub(crate) fn encoded_to_general(&self) -> vk::ImageMemoryBarrier {
        to_general(self.encoded.image, vk::AccessFlags::SHADER_WRITE)
    }

    /// The same for the answer image and the write-back's copy into it.
    pub(crate) fn answer_to_general(&self) -> vk::ImageMemoryBarrier {
        to_general(self.answer.image, vk::AccessFlags::TRANSFER_WRITE)
    }

    /// Records, after the capture's exposure copy: the exposure write made visible to the shader,
    /// the encode dispatch over the padded extent, and the copy of the encoded image into `proxy`
    /// (tightly packed at the padded size). The caller's closing barrier covers the transfer.
    ///
    /// # Safety
    /// `cmd` is recording, the encoded image is in `GENERAL` (the caller's opening barrier),
    /// [`Self::bind_colour`] was called for this hold, and `proxy` holds the padded frame.
    pub(crate) unsafe fn record_encode(&self, device: &ash::Device, cmd: vk::CommandBuffer, proxy: vk::Buffer, paper_white: f32) {
        let exposure_in = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).build();
        let encoded_out = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).build();
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: self.padded_width,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: self.padded_width, height: self.padded_height, depth: 1 },
        };
        // SAFETY: forwarded; every handle is live.
        unsafe {
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[exposure_in], &[], &[]);
            self.dispatch(device, cmd, self.encode, paper_white, self.padded_width, self.padded_height);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[encoded_out], &[], &[]);
            device.cmd_copy_image_to_buffer(cmd, self.encoded.image, vk::ImageLayout::GENERAL, proxy, &[region]);
        }
    }

    /// Records, after the write-back's opening barrier: the copy of `source` (the padded answer,
    /// tightly packed) into the answer image, made visible to the shader, and the decode dispatch
    /// over the real extent, which writes the colour input in place. The caller closes with a
    /// `COMPUTE_SHADER/SHADER_WRITE -> ALL_COMMANDS` barrier.
    ///
    /// # Safety
    /// `cmd` is recording, the answer image is in `GENERAL` (the caller's opening barrier), the
    /// encoded image still holds this hold's capture, [`Self::bind_colour`] was called for it.
    pub(crate) unsafe fn record_decode(&self, device: &ash::Device, cmd: vk::CommandBuffer, source: vk::Buffer, paper_white: f32) {
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: self.padded_width,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: self.padded_width, height: self.padded_height, depth: 1 },
        };
        let answer_in = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).build();
        // SAFETY: forwarded; every handle is live.
        unsafe {
            device.cmd_copy_buffer_to_image(cmd, source, self.answer.image, vk::ImageLayout::GENERAL, &[region]);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[answer_in], &[], &[]);
            self.dispatch(device, cmd, self.decode, paper_white, self.width, self.height);
        }
    }

    /// # Safety
    /// `cmd` is recording.
    unsafe fn dispatch(&self, device: &ash::Device, cmd: vk::CommandBuffer, pipeline: vk::Pipeline, paper_white: f32, width: u32, height: u32) {
        let push = self.push(paper_white);
        // SAFETY: `HdrPush` is `repr(C)`, plain data; read as bytes for its own size.
        let bytes = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<HdrPush>()) };
        // SAFETY: forwarded; the set and pipeline are this pass's own.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.set), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_dispatch(cmd, width.div_ceil(GROUP), height.div_ceil(GROUP), 1);
        }
    }

    /// # Safety
    /// Nothing submitted may still use the pass (its owner's fences were waited on, or the device
    /// is idle).
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; destroying the pool frees the set.
        unsafe {
            if let Some(view) = self.colour_view {
                device.destroy_image_view(view, None);
            }
            self.encoded.destroy(device);
            self.answer.destroy(device);
            self.exposure.destroy(device);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.encode, None);
            device.destroy_pipeline(self.decode, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn to_general(image: vk::Image, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
        .build()
}

/// A compute pipeline from embedded SPIR-V (`scripts/check_shaders.py` ties it to its source).
///
/// # Safety
/// `device` is live; `layout` matches the shader's interface.
unsafe fn compute_pipeline(device: &ash::Device, layout: vk::PipelineLayout, spv: &[u8]) -> Option<vk::Pipeline> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).ok()?;
    // SAFETY: valid SPIR-V, compiled by `glslangValidator -V` and checked by spirv-val.
    let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code), None) }.ok()?;
    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(entry).build();
    let info = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(layout).build();
    // SAFETY: `module` and `layout` outlive the call.
    let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None) };
    // SAFETY: only needed for creation, which returned.
    unsafe { device.destroy_shader_module(module, None) };
    pipeline.ok().and_then(|p| p.first().copied())
}

/// [`HdrPass::build`]'s handles so far; destroys them unless moved out.
struct Partial<'a> {
    device: &'a ash::Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    encode: vk::Pipeline,
    decode: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    encoded: Option<OwnImage>,
    answer: Option<OwnImage>,
    exposure: Option<HostBuffer>,
}

impl<'a> Partial<'a> {
    fn default_for(device: &'a ash::Device) -> Self {
        Self {
            device,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            encode: vk::Pipeline::null(),
            decode: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            encoded: None,
            answer: None,
            exposure: None,
        }
    }
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        let d = self.device;
        // SAFETY: nothing here was ever submitted; null handles are ignored by the destroy calls.
        unsafe {
            if let Some(i) = self.encoded.take() {
                i.destroy(d);
            }
            if let Some(i) = self.answer.take() {
                i.destroy(d);
            }
            if let Some(b) = self.exposure.take() {
                b.destroy(d);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_pipeline(self.encode, None);
            d.destroy_pipeline(self.decode, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_push_constants_match_the_shaders_params_block() {
        // uvec2 size, uvec2 padded, float paper_white, uint flags: 24 bytes, std430-packed.
        assert_eq!(std::mem::size_of::<HdrPush>(), 24);
        assert_eq!(std::mem::offset_of!(HdrPush, padded_width), 8);
        assert_eq!(std::mem::offset_of!(HdrPush, paper_white), 16);
        assert_eq!(std::mem::offset_of!(HdrPush, flags), 20);
        for spv in [ENCODE_SPV, DECODE_SPV] {
            assert!(spv.len() > 20 && spv.len().is_multiple_of(4), "embedded SPIR-V");
        }
    }

    #[test]
    fn the_paper_white_parses_and_refuses_nonsense() {
        assert_eq!(parse_paper_white(None), Ok(3.0));
        assert_eq!(parse_paper_white(Some(" ")), Ok(3.0));
        assert_eq!(parse_paper_white(Some("2.5")), Ok(2.5));
        for bad in ["0", "-1", "nan", "inf", "white", "5000"] {
            assert!(parse_paper_white(Some(bad)).is_err(), "{bad}");
        }
        assert!(exposure_ok(0.128) && !exposure_ok(0.0) && !exposure_ok(f32::NAN) && !exposure_ok(f32::INFINITY) && !exposure_ok(-1.0));
    }
}
