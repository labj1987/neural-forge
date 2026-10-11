//! Reduced model resolution before the upscaler (`working_scale`, the Model resolution setting, below
//! 100%; docs/ARCHITECTURE.md 4.11). The network's cost and its VRAM go with its field's texels, so at
//! a scale `s` of the render extent the network runs on `s^2` of the pixels. Everything else of the
//! hold stays at the render extent: the encode, the motion pass, the decode and the edit cache.
//!
//! - In the capture (`C`), after the encode and the motion pass, `preupscale_reduce.comp` writes the
//!   network's input at its extent (the area average of the frame, encoded) and samples the
//!   accumulated motion into the network's own field ([`Reduced::record_reduce`]). The network's
//!   history lives at its extent; its vectors and jitter are scaled to it (`NativePass::frame`).
//! - After the network's frame (`N`), `preupscale_enlarge.comp` turns its answer into one at the
//!   render extent ([`Reduced::record_enlarge`]): the network's edit, the edit cache's log2 ratio,
//!   enlarged by a joint bilateral upsample guided by the full-resolution frame's luma and depth, and
//!   applied to the frame. The write-back decodes that answer with its own rules, and the edit cache
//!   captures it as it captures a full-resolution one.
//!
//! At 100% none of this exists: the native pass is built as before and the picture is unchanged.
//!
//! The setting's default (100%) is automatic on this path ([`effective_scale`]): 50% from
//! [`AUTO_PIXELS`] render pixels up (4K output), where it measured faster and no worse than 100%
//! (docs/LESSONS.md, "Reduced model resolution"; lowering it by default authorized by Alex on
//! 2026-10-11), and 100% below, where 75% already lost added detail. A lowered setting is used as set.

use ash::vk;

use super::hdr::{self, OwnImage};
use super::native::OwnBuffer;

const REDUCE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_reduce.spv");
const ENLARGE_SPV: &[u8] = include_bytes!("../../shaders/preupscale_enlarge.spv");

/// The enlarge's luma weight, in the encoded picture's units (upstream's JbuSigma).
pub(crate) const SIGMA_LUMA: f32 = 0.08;
/// The enlarge's depth weight, in stops of the pseudo-linear depth (the cache's bilateral sigma).
pub(crate) const SIGMA_DEPTH: f32 = 0.15;
/// The lowest scale (`ShmHeader::working_scale`'s bound).
pub(crate) const MIN_SCALE: f32 = 0.25;

/// The render pixels from which the default model resolution is [`AUTO_SCALE`]: above 1440p's
/// Balanced render size (1485x836, 1.24 Mpx), below 4K's (2228x1253, 2.79 Mpx).
pub(crate) const AUTO_PIXELS: u64 = 2_000_000;
/// The default model resolution from [`AUTO_PIXELS`] up.
pub(crate) const AUTO_SCALE: f32 = 0.5;

/// `NEURAL_FORGE_PREUPSCALE_FULL_RES=1`: the default stays 100% at every size (the A/B switch for
/// the automatic default; a lowered setting still applies).
fn full_res_forced() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_PREUPSCALE_FULL_RES").as_deref() == Some("1"));
    *ON
}

/// The model resolution this path runs at for a render extent and the setting (`working_scale`): the
/// setting when it is lowered (below 1), else [`AUTO_SCALE`] from [`AUTO_PIXELS`] render pixels up
/// (unless `full_res` keeps 100%), else 1.
pub(crate) fn scale_for(width: u32, height: u32, setting: f32, full_res: bool) -> f32 {
    if setting < 1.0 {
        setting
    } else if !full_res && u64::from(width) * u64::from(height) >= AUTO_PIXELS {
        AUTO_SCALE
    } else {
        1.0
    }
}

/// [`scale_for`] with `NEURAL_FORGE_PREUPSCALE_FULL_RES`.
pub(crate) fn effective_scale(width: u32, height: u32, setting: f32) -> f32 {
    scale_for(width, height, setting, full_res_forced())
}

/// `flags`: depth is reversed-Z (the edit cache's decision, `cache::F_REVERSE_Z`).
pub(crate) const F_REVERSE_Z: u32 = 1;
/// `flags`: the reduce copies the accumulated motion (it was written this hold).
pub(crate) const F_MOTION: u32 = 2;

/// The shaders' `local_size`.
const GROUP: u32 = 8;

/// The network's extent for a render extent at `scale`: each side rounded, at least 8 texels (or the
/// side itself). Any scale of 1 or more, or not a number, is the render extent itself.
pub(crate) fn model_extent(width: u32, height: u32, scale: f32) -> (u32, u32) {
    if scale.is_nan() || scale >= 1.0 {
        return (width, height);
    }
    let s = scale.max(MIN_SCALE);
    let side = |n: u32| ((n as f32 * s).round() as u32).clamp(n.min(8), n);
    (side(width), side(height))
}

/// The push constants of `preupscale_reduce.comp` and `preupscale_enlarge.comp`: their `Params` blocks
/// match this field for field (append only, in all three places).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReducedPush {
    /// `uvec2 size`: the render extent.
    pub size: [u32; 2],
    /// `uvec2 padded`: the render extent's padded size (the enlarged answer's rows).
    pub padded: [u32; 2],
    /// `uvec2 model`: the network's extent.
    pub model: [u32; 2],
    /// `uvec2 model_padded`: its padded size (its encoded input and its answer's rows).
    pub model_padded: [u32; 2],
    /// `float paper_white`.
    pub paper_white: f32,
    /// `uint depth_kind`: `cache::depth_kind`, 0 without depth.
    pub depth_kind: u32,
    /// `uint flags`: `F_*`.
    pub flags: u32,
    /// `float sigma_luma`, `float sigma_depth`.
    pub sigma_luma: f32,
    pub sigma_depth: f32,
}

/// What the reduce and the enlarge read and write this hold, besides the pass's own.
pub(crate) struct Bindings {
    /// The colour input's view (the HDR pass's, bound for this hold).
    pub colour: vk::ImageView,
    /// The HDR pass's encoded picture (render padded extent).
    pub encoded: vk::ImageView,
    pub exposure: vk::Buffer,
    /// The accumulated motion (`NativePass::acc`, render extent).
    pub acc: vk::Buffer,
    /// The network's answer (`NativePass`'s, model padded extent).
    pub answer: vk::Buffer,
    /// This hold's depth copy, if it has one.
    pub depth: Option<vk::Buffer>,
}

/// The network's side of a reduced model resolution for one render extent and one model extent: the
/// two shaders, the network's encoded input and motion at its extent, the enlarged answer, and the
/// edit cache's scratch when the network's features are too small for it.
pub(crate) struct Reduced {
    width: u32,
    height: u32,
    model_width: u32,
    model_height: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    reduce: vk::Pipeline,
    enlarge: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The network's encoded input (model padded extent).
    pub(super) encoded: OwnImage,
    /// The network's motion (model extent, f32 x and y).
    pub(super) acc: OwnBuffer,
    /// The enlarged answer (render padded extent, packed half4), what the write-back decodes.
    pub(super) answer: OwnBuffer,
    command_pool: vk::CommandPool,
    /// The enlarge, recorded every hold that runs the network.
    cmd: vk::CommandBuffer,
    /// The edit cache's buffers (`cache::ScratchLayout`) above interval 1: the network's input
    /// features at the model extent cannot hold them. Built on first use.
    scratch: Option<(OwnBuffer, u64)>,
}

// SAFETY: plain handles of the layer's own, only used behind the device's `State` mutex.
unsafe impl Send for Reduced {}

impl Reduced {
    /// `None` on any failure (whatever was made is destroyed).
    ///
    /// # Safety
    /// `device` is live; `family` is the queue family the holds are submitted on.
    pub(crate) unsafe fn build(
        device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, render: (u32, u32), model: (u32, u32),
    ) -> Option<Self> {
        let binding = |n: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build()
        };
        let (image, buffer) = (vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_BUFFER);
        let bindings = [
            binding(0, image),
            binding(1, image),
            binding(2, buffer),
            binding(3, image),
            binding(4, buffer),
            binding(5, buffer),
            binding(6, buffer),
            binding(7, buffer),
            binding(8, buffer),
        ];
        let (mpw, mph) = super::padded(model.0, model.1);
        let (pw, ph) = super::padded(render.0, render.1);
        let mut set_layout = vk::DescriptorSetLayout::null();
        let mut pipeline_layout = vk::PipelineLayout::null();
        let mut pipelines = [vk::Pipeline::null(); 2];
        let mut pool = vk::DescriptorPool::null();
        let mut command_pool = vk::CommandPool::null();
        let (mut encoded, mut acc, mut answer) = (None, None, None);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        // SAFETY: valid create infos; everything made here is destroyed below on failure.
        let made = unsafe {
            (|| -> Option<(vk::DescriptorSet, vk::CommandBuffer)> {
                set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
                let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<ReducedPush>() as u32).build();
                pipeline_layout = device
                    .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(std::slice::from_ref(&push)), None)
                    .ok()?;
                pipelines[0] = hdr::compute_pipeline(device, pipeline_layout, REDUCE_SPV)?;
                pipelines[1] = hdr::compute_pipeline(device, pipeline_layout, ENLARGE_SPV)?;
                let sizes = [vk::DescriptorPoolSize { ty: image, descriptor_count: 3 }, vk::DescriptorPoolSize { ty: buffer, descriptor_count: 6 }];
                pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).ok()?;
                let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(std::slice::from_ref(&set_layout))).ok()?[0];
                // TRANSFER_SRC: read back by the tests.
                encoded = Some(OwnImage::new(device, instance, physical_device, mpw, mph, vk::ImageUsageFlags::TRANSFER_SRC)?);
                acc = Some(OwnBuffer::new(device, instance, physical_device, u64::from(model.0) * u64::from(model.1) * 8, storage | vk::BufferUsageFlags::TRANSFER_SRC)?);
                answer = Some(OwnBuffer::new(device, instance, physical_device, u64::from(pw) * u64::from(ph) * 8, storage | vk::BufferUsageFlags::TRANSFER_SRC)?);
                command_pool = device
                    .create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                    .ok()?;
                let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                let cmd = crate::loader_data::allocate_commands(device, &alloc).ok()?[0];
                Some((set, cmd))
            })()
        };
        match (made, encoded.take(), acc.take(), answer.take()) {
            (Some((set, cmd)), Some(encoded), Some(acc), Some(answer)) => Some(Self {
                width: render.0,
                height: render.1,
                model_width: model.0,
                model_height: model.1,
                set_layout,
                pipeline_layout,
                reduce: pipelines[0],
                enlarge: pipelines[1],
                pool,
                set,
                encoded,
                acc,
                answer,
                command_pool,
                cmd,
                scratch: None,
            }),
            (_, encoded, acc, answer) => {
                // SAFETY: nothing here was submitted; null handles are ignored; the pools free their
                // sets and command buffers.
                unsafe {
                    if let Some(e) = encoded {
                        e.destroy(device);
                    }
                    for b in [acc, answer].iter().flatten() {
                        b.destroy(device);
                    }
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_descriptor_pool(pool, None);
                    for p in pipelines {
                        device.destroy_pipeline(p, None);
                    }
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                }
                None
            }
        }
    }

    /// The enlarge's command buffer ([`Self::record_enlarge`]).
    pub(crate) fn command(&self) -> vk::CommandBuffer {
        self.cmd
    }

    /// Model pixels per render pixel, per axis.
    pub(crate) fn ratio(&self) -> [f32; 2] {
        [self.model_width as f32 / self.width as f32, self.model_height as f32 / self.height as f32]
    }

    /// The edit cache's scratch of `bytes`, built on first use; `None` when it cannot be.
    ///
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn scratch(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, bytes: u64) -> Option<(vk::Buffer, u64)> {
        if self.scratch.as_ref().is_none_or(|(_, b)| *b < bytes) {
            if let Some((old, _)) = self.scratch.take() {
                // SAFETY: the caller's hold waited for every earlier hold's work.
                unsafe { old.destroy(device) };
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            // SAFETY: forwarded.
            self.scratch = unsafe { OwnBuffer::new(device, instance, physical_device, bytes, usage) }.map(|b| (b, bytes));
        }
        self.scratch.as_ref().map(|(b, n)| (b.buffer, *n))
    }

    /// The push constants for this hold.
    pub(crate) fn push(&self, paper_white: f32, depth_kind: u32, flags: u32) -> ReducedPush {
        let (pw, ph) = super::padded(self.width, self.height);
        let (mpw, mph) = super::padded(self.model_width, self.model_height);
        ReducedPush {
            size: [self.width, self.height],
            padded: [pw, ph],
            model: [self.model_width, self.model_height],
            model_padded: [mpw, mph],
            paper_white,
            depth_kind,
            flags,
            sigma_luma: SIGMA_LUMA,
            sigma_depth: SIGMA_DEPTH,
        }
    }

    /// Points the set at this hold's resources.
    ///
    /// # Safety
    /// Nothing submitted may still use the set (the hold's `wait_idle` passed); every handle is live.
    pub(crate) unsafe fn bind(&self, device: &ash::Device, b: &Bindings) {
        let image = |view| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
        let buffer = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (colour, encoded, small) = (image(b.colour), image(b.encoded), image(self.encoded.view));
        // Without depth the shaders do not read binding 8: any buffer will do.
        let (exposure, acc_full, acc_small, answer_small, answer_full, depth) =
            (buffer(b.exposure), buffer(b.acc), buffer(self.acc.buffer), buffer(b.answer), buffer(self.answer.buffer), buffer(b.depth.unwrap_or(self.acc.buffer)));
        let w = |binding, ty| vk::WriteDescriptorSet::builder().dst_set(self.set).dst_binding(binding).descriptor_type(ty);
        let (img, buf) = (vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_BUFFER);
        let writes = [
            w(0, img).image_info(&colour).build(),
            w(1, img).image_info(&encoded).build(),
            w(2, buf).buffer_info(&exposure).build(),
            w(3, img).image_info(&small).build(),
            w(4, buf).buffer_info(&acc_full).build(),
            w(5, buf).buffer_info(&acc_small).build(),
            w(6, buf).buffer_info(&answer_small).build(),
            w(7, buf).buffer_info(&answer_full).build(),
            w(8, buf).buffer_info(&depth).build(),
        ];
        // SAFETY: forwarded.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// # Safety
    /// `cmd` is recording; the set is bound to this hold's resources.
    unsafe fn dispatch(&self, device: &ash::Device, cmd: vk::CommandBuffer, pipeline: vk::Pipeline, push: &ReducedPush, extent: [u32; 2]) {
        // SAFETY: `ReducedPush` is `repr(C)`, plain data; read as bytes for its own size.
        let bytes = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(push).cast::<u8>(), std::mem::size_of::<ReducedPush>()) };
        // SAFETY: forwarded; the set and pipeline are this pass's own.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.set], &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_dispatch(cmd, extent[0].div_ceil(GROUP), extent[1].div_ceil(GROUP), 1);
        }
    }

    /// Records the reduce into the capture, after the encode and the motion pass: a barrier for what
    /// they wrote (the network's encoded input is rewritten whole: from `UNDEFINED`), the dispatch over
    /// the model padded extent. The network's frame opens with a barrier from everything before it.
    ///
    /// # Safety
    /// `cmd` is recording; [`Self::bind`] was called for this hold.
    pub(crate) unsafe fn record_reduce(&self, device: &ash::Device, cmd: vk::CommandBuffer, push: &ReducedPush) {
        let written = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let small = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.encoded.image)
            .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
            .build();
        // SAFETY: forwarded.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[small],
            );
            self.dispatch(device, cmd, self.reduce, push, push.model_padded);
        }
    }

    /// Records the enlarge into the pass's own command buffer, submitted right after the network's
    /// frame: opened by a barrier from everything before it, closed by one to everything after it (the
    /// write-back copies the enlarged answer).
    ///
    /// # Safety
    /// The command buffer is not pending (the hold's `wait_idle` passed); [`Self::bind`] was called.
    pub(crate) unsafe fn record_enlarge(&self, device: &ash::Device, push: &ReducedPush) -> ash::prelude::VkResult<vk::CommandBuffer> {
        let cmd = self.cmd;
        let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).build();
        let close = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
        // SAFETY: forwarded.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[open], &[], &[]);
            self.dispatch(device, cmd, self.enlarge, push, push.padded);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[close], &[], &[]);
            device.end_command_buffer(cmd)?;
        }
        Ok(cmd)
    }

    /// # Safety
    /// Nothing submitted may still use it.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; the pools free the set and the command buffer.
        unsafe {
            if let Some((b, _)) = &self.scratch {
                b.destroy(device);
            }
            self.encoded.destroy(device);
            self.acc.destroy(device);
            self.answer.destroy(device);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.reduce, None);
            device.destroy_pipeline(self.enlarge, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preupscale::{f16_to_f32, own_host_buffer_with, tests::f32_to_f16, HostBuffer};

    #[test]
    fn the_model_extent_is_the_render_extent_at_100_percent() {
        for s in [1.0, 1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(model_extent(2228, 1253, s), (2228, 1253), "scale {s}");
        }
        assert_eq!(model_extent(2228, 1253, 0.5), (1114, 627));
        assert_eq!(model_extent(2228, 1253, 0.75), (1671, 940));
        assert_eq!(model_extent(1485, 836, 0.1), (371, 209), "at least 25%");
        assert_eq!(model_extent(12, 6, 0.25), (8, 6), "at least 8 texels, never more than the side");
    }

    #[test]
    fn the_default_is_half_resolution_from_4k_and_full_below() {
        assert_eq!(scale_for(2228, 1253, 1.0, false), 0.5, "4K Balanced");
        assert_eq!(scale_for(1485, 836, 1.0, false), 1.0, "1440p Balanced");
        assert_eq!(scale_for(1920, 1080, 1.0, false), 0.5, "4K Performance");
        assert_eq!(scale_for(2228, 1253, 1.0, true), 1.0, "full resolution forced");
        assert_eq!(scale_for(2228, 1253, 0.75, false), 0.75, "a lowered setting as set");
        assert_eq!(scale_for(1485, 836, 0.75, true), 0.75);
        assert_eq!(model_extent(2228, 1253, scale_for(2228, 1253, 1.0, false)), (1114, 627));
    }

    const W: u32 = 16;
    const H: u32 = 16;
    const PAPER: f32 = 3.0;
    const EPS: f64 = 1.0 / 512.0;

    fn oetf(c: f64) -> f64 {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
    }

    fn eotf(s: f64) -> f64 {
        let s = s.clamp(0.0, 1.0);
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    }

    /// v (exposure-normalised) to the model domain and back (the encode's and decode's curves).
    fn enc(v: f64) -> f64 {
        oetf(if v <= 0.75 { v } else { 0.75 + 0.25 * (1.0 - (-5.770780 * (v - 0.75)).exp()) })
    }

    fn dec(a: f64) -> f64 {
        let y = eotf(a).min(1.0 - 1e-4);
        if y >= 0.75 { 0.75 - (1.0 - (y - 0.75) / 0.25).ln() / 5.770780 } else { y }
    }

    fn halves(values: impl Iterator<Item = f32>) -> Vec<u16> {
        values.map(f32_to_f16).collect()
    }

    struct Rig {
        gpu: (ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue, u32),
        pool: vk::CommandPool,
    }

    impl Rig {
        fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
            let d = &self.gpu.3;
            // SAFETY: the test's own pool, queue and device.
            unsafe {
                let cmd = d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).command_buffer_count(1)).unwrap()[0];
                d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                d.end_command_buffer(cmd).unwrap();
                self.submit(&[cmd]);
                d.free_command_buffers(self.pool, &[cmd]);
            }
        }

        fn submit(&self, cmds: &[vk::CommandBuffer]) {
            let d = &self.gpu.3;
            // SAFETY: recorded, idle command buffers of this device.
            unsafe {
                d.queue_submit(self.gpu.4, &[vk::SubmitInfo::builder().command_buffers(cmds).build()], vk::Fence::null()).unwrap();
                d.queue_wait_idle(self.gpu.4).unwrap();
            }
        }

        /// `texels` (RGBA16F rows of `w`) into `image`, left in GENERAL.
        fn upload(&self, staging: &HostBuffer, image: &OwnImage, w: u32, h: u32, texels: &[u16]) {
            let d = &self.gpu.3;
            // SAFETY: the staging buffer is mapped for at least the texels; the image is the test's.
            unsafe {
                std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
                self.one_shot(|cmd| {
                    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
                    let to_general = vk::ImageMemoryBarrier::builder()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.image)
                        .subresource_range(range)
                        .build();
                    d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
                    let region = vk::BufferImageCopy {
                        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                        image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                        ..Default::default()
                    };
                    d.cmd_copy_buffer_to_image(cmd, staging.buffer, image.image, vk::ImageLayout::GENERAL, &[region]);
                });
            }
        }

        fn read_image(&self, staging: &HostBuffer, image: &OwnImage, w: u32, h: u32) -> Vec<f32> {
            let d = &self.gpu.3;
            // SAFETY: as `upload`.
            unsafe {
                self.one_shot(|cmd| {
                    let region = vk::BufferImageCopy {
                        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                        image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                        ..Default::default()
                    };
                    d.cmd_copy_image_to_buffer(cmd, image.image, vk::ImageLayout::GENERAL, staging.buffer, &[region]);
                });
                std::slice::from_raw_parts(staging.ptr.cast::<u16>(), (w * h * 4) as usize).iter().map(|&h| f16_to_f32(h)).collect()
            }
        }

        fn read_buffer(&self, staging: &HostBuffer, buffer: vk::Buffer, bytes: u64) -> Vec<u8> {
            let d = &self.gpu.3;
            // SAFETY: as `upload`.
            unsafe {
                self.one_shot(|cmd| d.cmd_copy_buffer(cmd, buffer, staging.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]));
                std::slice::from_raw_parts(staging.ptr, bytes as usize).to_vec()
            }
        }
    }

    /// The reduce (the network's input: the area average, encoded; its motion: the render pixel at each
    /// model texel's centre) and the enlarge at 50% on lavapipe: with the network's answer equal to its
    /// input the frame comes back as it was; with an edit of +0.5 stop on one side of an edge and -0.5
    /// on the other, the pixels next to the edge take their own side's edit, told apart by luma, and
    /// with the same luma on both sides by depth, where a plain bilinear enlargement mixes the two.
    #[test]
    fn the_reduced_network_sees_the_area_average_and_its_edit_is_enlarged_along_edges() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, family) = (&gpu.1, gpu.2, &gpu.3, gpu.5);
        let (mw, mh) = model_extent(W, H, 0.5);
        assert_eq!((mw, mh), (8, 8));
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
            let rig = Rig { gpu: gpu.clone(), pool };
            let host = |bytes: u64| own_host_buffer_with(device, instance, physical, bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST).unwrap();
            let staging = host(u64::from(W * H) * 16);
            let exposure = host(16);
            std::ptr::copy_nonoverlapping(0x3c00u32.to_le_bytes().as_ptr(), exposure.ptr, 4);
            let acc_full = host(u64::from(W * H) * 8);
            let answer_small = host(u64::from(mw * mh) * 8);
            let depth = host(u64::from(W * H) * 4);
            let colour = OwnImage::new(device, instance, physical, W, H, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let encoded = OwnImage::new(device, instance, physical, W, H, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let r = Reduced::build(device, instance, physical, family, (W, H), (mw, mh)).expect("the reduced passes build");
            r.bind(device, &Bindings { colour: colour.view, encoded: encoded.view, exposure: exposure.buffer, acc: acc_full.buffer, answer: answer_small.buffer, depth: Some(depth.buffer) });
            let acc: Vec<f32> = (0..W * H).flat_map(|i| [i as f32 * 0.25, -(i as f32)]).collect();
            std::ptr::copy_nonoverlapping(acc.as_ptr().cast::<u8>(), acc_full.ptr, acc.len() * 4);

            // The frame: dark on the left half, bright on the right, with a fine texture; depth near on
            // the left (reversed-Z: larger is nearer), far on the right. `flat` gives both sides the same
            // colour so only depth tells them apart.
            let scene = |x: u32, y: u32, c: u32, flat: bool| -> f32 {
                let side = if flat || x < W / 2 { 0.15 } else { 1.2 };
                side * (1.0 + 0.05 * ((x * 3 + y * 5 + c) % 4) as f32)
            };
            let depth_of = |x: u32| if x < W / 2 { 0.5f32 } else { 0.05 };
            let d: Vec<f32> = (0..W * H).map(|i| depth_of(i % W)).collect();
            std::ptr::copy_nonoverlapping(d.as_ptr().cast::<u8>(), depth.ptr, d.len() * 4);
            let v_of = |x: u32, y: u32, c: u32, flat: bool| f64::from(f16_to_f32(f32_to_f16(scene(x, y, c, flat)))) / f64::from(PAPER);
            let load = |flat: bool| {
                let texels = halves((0..W * H).flat_map(|i| {
                    let (x, y) = (i % W, i / W);
                    [scene(x, y, 0, flat), scene(x, y, 1, flat), scene(x, y, 2, flat), 1.0]
                }));
                rig.upload(&staging, &colour, W, H, &texels);
                let enc_texels = halves((0..W * H).flat_map(|i| {
                    let (x, y) = (i % W, i / W);
                    [enc(v_of(x, y, 0, flat)) as f32, enc(v_of(x, y, 1, flat)) as f32, enc(v_of(x, y, 2, flat)) as f32, 1.0]
                }));
                rig.upload(&staging, &encoded, W, H, &enc_texels);
            };
            let flags = F_REVERSE_Z | F_MOTION;
            let reduce = |push: &ReducedPush| rig.one_shot(|cmd| r.record_reduce(device, cmd, push));
            let enlarge = |push: &ReducedPush| rig.submit(&[r.record_enlarge(device, push).unwrap()]);
            // The answer the enlarge wrote, as an edit in stops against the frame.
            let measured = |flat: bool| -> Vec<[f64; 3]> {
                let bytes = rig.read_buffer(&staging, r.answer.buffer, u64::from(W * H) * 8);
                let h: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                (0..W * H)
                    .map(|i| {
                        let (x, y) = (i % W, i / W);
                        std::array::from_fn(|c| ((dec(f64::from(f16_to_f32(h[i as usize * 4 + c]))) + EPS) / (v_of(x, y, c as u32, flat) + EPS)).log2())
                    })
                    .collect()
            };

            load(false);
            let push = r.push(PAPER, 1, flags);
            reduce(&push);
            // The network's input: each texel the 2x2 average of v, encoded.
            let small = rig.read_image(&staging, &r.encoded, mw, mh);
            for (i, x, y) in (0..mw * mh).map(|i| (i, i % mw, i / mw)) {
                for c in 0..3 {
                    let avg = (0..4).map(|k| v_of(2 * x + k % 2, 2 * y + k / 2, c, false)).sum::<f64>() / 4.0;
                    let got = f64::from(small[(i * 4 + c) as usize]);
                    assert!((got - enc(avg)).abs() < 1e-3, "model texel ({x}, {y}) channel {c}: {got} against {}", enc(avg));
                }
            }
            // Its motion: the render pixel at each model texel's centre, unchanged.
            let bytes = rig.read_buffer(&staging, r.acc.buffer, u64::from(mw * mh) * 8);
            let small_acc: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            for (i, x, y) in (0..mw * mh).map(|i| (i, i % mw, i / mw)) {
                let src = ((2 * y + 1) * W + 2 * x + 1) as usize;
                assert_eq!([small_acc[2 * i as usize], small_acc[2 * i as usize + 1]], [acc[2 * src], acc[2 * src + 1]], "model texel ({x}, {y})");
            }

            // An answer equal to the input: no edit, the frame as it was (within the half's rounding).
            let as_input = halves(small.iter().copied());
            std::ptr::copy_nonoverlapping(as_input.as_ptr().cast::<u8>(), answer_small.ptr, as_input.len() * 2);
            enlarge(&push);
            for (i, e) in measured(false).iter().enumerate() {
                assert!(e.iter().all(|e| e.abs() < 0.01), "pixel {i}: {e:?} stops with no edit");
            }

            // +0.5 stop on the left half of the model texels, -0.5 on the right.
            let side_edit = |x: u32| if x < mw / 2 { 0.5 } else { -0.5 };
            let edited = |small: &[f32]| {
                let texels = halves((0..mw * mh).flat_map(|i| {
                    let x = i % mw;
                    let px = |c: u32| enc(dec(f64::from(small[(i * 4 + c) as usize])) * 2f64.powf(side_edit(x))) as f32;
                    [px(0), px(1), px(2), 1.0]
                }));
                std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), answer_small.ptr, texels.len() * 2);
            };
            // Pixels 7 and 8 sit on either side of the edge, each a quarter of the way to the other side's
            // model texel: bilinear alone gives them +-0.25.
            let check = |m: &[[f64; 3]], what: &str| {
                for y in 1..H - 1 {
                    for x in [6, 7, 8, 9] {
                        let want = if x < W / 2 { 0.5 } else { -0.5 };
                        let e = m[(y * W + x) as usize];
                        assert!(e.iter().all(|e| (e - want).abs() < 0.03), "{what}: pixel ({x}, {y}) took {e:?} stops, its side's edit is {want}");
                    }
                }
            };
            edited(&small);
            // Luma alone (no depth) tells the sides apart.
            let plain = r.push(PAPER, 0, flags);
            enlarge(&plain);
            check(&measured(false), "told apart by luma");
            // The same colour on both sides: only depth tells them apart.
            load(true);
            reduce(&push);
            let small = rig.read_image(&staging, &r.encoded, mw, mh);
            edited(&small);
            enlarge(&push);
            check(&measured(true), "told apart by depth");
            // Without depth, the same frame is enlarged plainly: the edge pixels mix the two sides.
            enlarge(&plain);
            let m = measured(true);
            let e = m[(4 * W + 7) as usize][1];
            assert!((e - 0.25).abs() < 0.03, "without a guide, pixel (7, 4) is bilinear: {e} stops");

            r.destroy(device);
            colour.destroy(device);
            encoded.destroy(device);
            for b in [&staging, &exposure, &acc_full, &answer_small, &depth] {
                b.destroy(device);
            }
            device.destroy_command_pool(pool, None);
        }
    }
}
