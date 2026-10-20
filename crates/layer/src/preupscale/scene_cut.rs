//! The scene-cut detector of the native hold (docs/DLSSNR_PARAMETERS.md, "History reset"): when the
//! picture changes as a whole between two holds (a cut in a cutscene, a respawn, a camera switch), the
//! network's history describes a scene no longer on screen, and blending toward it ghosts the old scene
//! into the new one. The detector resets the history on such a frame, whether DLSS's motion vectors are
//! there or not.
//!
//! It is the 1.x helper's detector (`scene.rs`, removed with the helper in 3.0.0) moved onto the GPU: a
//! luma thumbnail of the encoded capture on an [`STEP`]-pixel step (each channel clamped to [0, 1] and
//! rounded to 8 bits, the three averaged), compared with the previous hold's; a cut when the mean
//! absolute difference is at least [`THRESHOLD`] of 255. `shaders/preupscale_scene_cut.comp` takes the
//! thumbnail and decides in the capture (`C`) of every hold, after the encode, and writes the decision
//! into the native pass's state buffer, where the frame shaders read it as "no history" on the same frame
//! (seed 0, no blend, for every pass). Nothing waits for it: the host reads the few result words at the
//! next hold, once the hold's fence was waited on anyway (`Resources::wait_idle`), for the log, the
//! counter and the CPU's side of the reset ([`effect`]).

use std::sync::LazyLock;

use ash::vk;

use super::hdr;
use super::native::OwnBuffer;
use super::HostBuffer;

const SPV: &[u8] = include_bytes!("../../shaders/preupscale_scene_cut.spv");

/// Every `STEP`th pixel on each axis goes into the thumbnail (the shader's `STEP`).
pub(crate) const STEP: u32 = 8;

/// A cut is a mean absolute difference of the two thumbnails of at least this, on 0-255: the 8-bit value
/// of the encoded capture, which is display-referred in [0, 1] (the encode's exposure, paper white,
/// shoulder and sRGB). The 1.x helper's threshold on the same encoded proxy (it quantised RGBA16F proxies
/// to 8 bits exactly so; commit `ebad222`), where GTA V gave 9 cuts in 7121 network frames
/// (docs/LESSONS.md, "False scene cuts: none, threshold kept").
pub(crate) const THRESHOLD: u32 = 40;

/// The test override: `off` (or `0`) switches the detector off, a number from 1 to 255 replaces
/// [`THRESHOLD`]. Read once, logged once.
pub(crate) const ENV: &str = "NEURAL_FORGE_SCENE_CUT";

/// The first dispatch's workgroup (the shader's `local_size_x`).
const GROUP: u32 = 256;

/// The threshold `raw` (the [`ENV`] value) asks for, 0 for off: unset, empty or `on` is [`THRESHOLD`];
/// anything else than `off`, `0` or 1 to 255 is an error.
pub(crate) fn parse_env(raw: Option<&str>) -> Result<u32, String> {
    match raw.map(str::trim) {
        None | Some("" | "on") => Ok(THRESHOLD),
        Some("off" | "0") => Ok(0),
        Some(other) => other.parse::<u32>().ok().filter(|t| (1..=255).contains(t)).ok_or_else(|| other.to_string()),
    }
}

/// This process's threshold ([`parse_env`]), 0 when the detector is off. Cached; logged once when it is
/// not the default.
pub(crate) fn threshold() -> u32 {
    static THRESHOLD_ONCE: LazyLock<u32> = LazyLock::new(|| match parse_env(neural_forge_protocol::env::var(ENV).as_deref()) {
        Ok(0) => {
            crate::log!("[preupscale] scene-cut detection off ({ENV}=off): only the other history resets apply");
            0
        }
        Ok(t) => {
            if t != THRESHOLD {
                crate::log!("[preupscale] scene-cut threshold {t} of 255 ({ENV}; the default is {THRESHOLD})");
            }
            t
        }
        Err(other) => {
            crate::log!("[preupscale] {ENV}={other:?} is not off or a threshold from 1 to 255; the default {THRESHOLD} stays");
            THRESHOLD
        }
    });
    *THRESHOLD_ONCE
}

/// The thumbnail's sample count for a render extent.
pub(crate) fn samples(width: u32, height: u32) -> u32 {
    width.div_ceil(STEP) * height.div_ceil(STEP)
}

/// The shader's push constants (field for field).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Push {
    size: [u32; 2],
    threshold: u32,
    compare: u32,
}

/// What the GPU decided for a hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reading {
    pub cut: bool,
    /// The mean absolute difference, 0-255 (0 on a hold with nothing to compare with).
    pub mean: u32,
}

/// What the hold that ran the detector did with its frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Recorded {
    /// The network ran (a refresh); else the edit cache carried its last answer (a cached hold).
    pub network: bool,
    /// The frame's parameters kept the history (the shaders' `frame.z`).
    pub history: bool,
}

/// What a hold's reading means for the CPU, read at the next hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    /// No cut, or one on a frame that had no history anyway.
    None,
    /// The network's frame was a cut and started afresh on the GPU: the CPU's seed restarts as after a
    /// reset, and the reset is logged and counted. Carries the mean difference.
    ResetOnGpu(u32),
    /// The cut was on a cached hold: nothing of the network ran on it, so its next frame resets
    /// (`HistoryGap::scene_cut`) and the edit cache refreshes rather than carry an edit across it.
    Latch(u32),
}

/// [`Effect`] of `reading` on the hold `recorded`.
pub(crate) fn effect(reading: Reading, recorded: Recorded) -> Effect {
    match (reading.cut, recorded) {
        (false, _) | (true, Recorded { network: true, history: false }) => Effect::None,
        (true, Recorded { network: true, history: true }) => Effect::ResetOnGpu(reading.mean),
        (true, Recorded { network: false, .. }) => Effect::Latch(reading.mean),
    }
}

/// The detector's pipelines and buffers for one extent, built on the first native hold.
pub(crate) struct ScenePass {
    width: u32,
    height: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// The thumbnail pass, then the decision.
    pipelines: [vk::Pipeline; 2],
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    thumb: OwnBuffer,
    partial: OwnBuffer,
    /// Host-visible: compared, cut, sum, samples (the shader's `Result`).
    result: HostBuffer,
    /// The encoded view and state buffer the set points at, with the passes they are of ([`Self::bind`]'s
    /// `made`): a rebuilt pass's view or buffer may come back under the old one's handle.
    bound: Option<(vk::ImageView, vk::Buffer, [u64; 2])>,
    /// A submitted hold wrote the thumbnail: the next one compares with it.
    have: bool,
    /// The last submitted hold whose result is still to be read.
    pending: Option<Recorded>,
}

// SAFETY: plain handles and the layer's own mapping, only used behind the device's `State` mutex.
unsafe impl Send for ScenePass {}

/// [`ScenePass::build`]'s handles so far; destroys them unless moved out.
struct Partial<'a> {
    device: &'a ash::Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    pool: vk::DescriptorPool,
    thumb: Option<OwnBuffer>,
    partial: Option<OwnBuffer>,
    result: Option<HostBuffer>,
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        let d = self.device;
        // SAFETY: nothing of these was ever submitted; null handles are ignored.
        unsafe {
            for &p in &self.pipelines {
                d.destroy_pipeline(p, None);
            }
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            for b in [&self.thumb, &self.partial].into_iter().flatten() {
                b.destroy(d);
            }
            if let Some(r) = &self.result {
                r.destroy(d);
            }
        }
    }
}

impl ScenePass {
    /// # Safety
    /// `device` is live.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, width: u32, height: u32) -> Option<Self> {
        let n = samples(width, height).max(1);
        let groups = n.div_ceil(GROUP);
        let types = [vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER];
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = types
            .iter()
            .enumerate()
            .map(|(i, &ty)| vk::DescriptorSetLayoutBinding::builder().binding(i as u32).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build())
            .collect();
        let mut p = Partial {
            device,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
            pool: vk::DescriptorPool::null(),
            thumb: None,
            partial: None,
            result: None,
        };
        // SAFETY: valid create infos; everything made is owned by `p` until it moves into the result.
        unsafe {
            p.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<Push>() as u32).build();
            p.pipeline_layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&p.set_layout)).push_constant_ranges(std::slice::from_ref(&push)), None)
                .ok()?;
            for decide in [0u32, 1] {
                let entry = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
                let data = decide.to_ne_bytes();
                let spec = vk::SpecializationInfo::builder().map_entries(&entry).data(&data).build();
                p.pipelines.push(hdr::compute_pipeline_with(device, p.pipeline_layout, SPV, Some(&spec))?);
            }
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 },
            ];
            p.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).ok()?;
            let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(p.pool).set_layouts(std::slice::from_ref(&p.set_layout))).ok()?[0];
            p.thumb = Some(OwnBuffer::new(device, instance, physical_device, u64::from(n) * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?);
            p.partial = Some(OwnBuffer::new(device, instance, physical_device, u64::from(groups) * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?);
            // TRANSFER_SRC: inside DLSS's buffer the hold copies the result out for the host itself.
            p.result = Some(super::own_host_buffer_with(device, instance, physical_device, 16, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)?);
            let result = p.result.take()?;
            std::ptr::write_bytes(result.ptr, 0, 16);
            let pipelines: [vk::Pipeline; 2] = std::mem::take(&mut p.pipelines).try_into().ok()?;
            Some(Self {
                width,
                height,
                set_layout: std::mem::take(&mut p.set_layout),
                pipeline_layout: std::mem::take(&mut p.pipeline_layout),
                pipelines,
                pool: std::mem::take(&mut p.pool),
                set,
                thumb: p.thumb.take()?,
                partial: p.partial.take()?,
                result,
                bound: None,
                have: false,
                pending: None,
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    /// Points the set at this hold's encoded capture and the native pass's state buffer (when they
    /// changed). `made` says which objects the handles name: the HDR pass whose view `encoded` is
    /// (`hdr::HdrPass::id`) and the native pass whose buffer `state` is (`native::NativePass::id`). The
    /// detector outlives the native pass (rebuilt for another model resolution), and a destroyed buffer's
    /// handle comes back for the next one made, so equal handles alone do not say the set still points at
    /// the right buffer.
    ///
    /// # Safety
    /// Nothing of this pass is pending; both handles are live.
    pub(crate) unsafe fn bind(&mut self, device: &ash::Device, encoded: vk::ImageView, state: vk::Buffer, made: [u64; 2]) {
        if self.bound == Some((encoded, state, made)) {
            return;
        }
        // Not while a hold recorded into DLSS's buffer with the set may run (the hold then misses).
        if !super::may_rewrite("the scene-cut detector's descriptor set") {
            return;
        }
        let image = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: encoded, image_layout: vk::ImageLayout::GENERAL }];
        let buf = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (thumb, partial, result, state_info) = (buf(self.thumb.buffer), buf(self.partial.buffer), buf(self.result.buffer), buf(state));
        let w = |binding, ty| vk::WriteDescriptorSet::builder().dst_set(self.set).dst_binding(binding).descriptor_type(ty);
        let writes = [
            w(0, vk::DescriptorType::STORAGE_IMAGE).image_info(&image).build(),
            w(1, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&thumb).build(),
            w(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&partial).build(),
            w(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&result).build(),
            w(4, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&state_info).build(),
        ];
        // SAFETY: the set is not in use (contract); every handle is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        self.bound = Some((encoded, state, made));
    }

    /// Records the thumbnail and the decision into the capture, after the encode: compared with the
    /// previous hold's thumbnail when a submitted hold wrote one.
    ///
    /// # Safety
    /// `cmd` is recording; [`Self::bind`] was called for this hold; `threshold` is above 0.
    pub(crate) unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, threshold: u32) {
        let push = Push { size: [self.width, self.height], threshold, compare: u32::from(self.have) };
        // SAFETY: `Push` is `repr(C)` plain data, read as bytes for its own size.
        let bytes = unsafe { std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<Push>()) };
        let groups = samples(self.width, self.height).max(1).div_ceil(GROUP);
        let to_compute = |src_stage, src| {
            let b = vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).build();
            // SAFETY: forwarded.
            unsafe { device.cmd_pipeline_barrier(cmd, src_stage, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[b], &[], &[]) };
        };
        // SAFETY: forwarded; the set, layout and pipelines are this pass's own.
        unsafe {
            // The encode's writes (and the capture's copies) before the thumbnail reads the encoded image.
            to_compute(vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, std::slice::from_ref(&self.set), &[]);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[0]);
            device.cmd_dispatch(cmd, groups, 1, 1);
            to_compute(vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[1]);
            device.cmd_dispatch(cmd, 1, 1, 1);
            // The result to the host (read at the next hold), the cut word to the frame shaders and
            // whatever runs next in the capture.
            let b = vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ | vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .build();
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    /// The capture that [`Self::record`] went into was submitted, for a hold that did `recorded`.
    pub(crate) fn submitted(&mut self, recorded: Recorded) {
        self.have = true;
        self.pending = Some(recorded);
    }

    /// The hold inside DLSS's buffer recorded the detector: the next hold compares with its thumbnail (if
    /// that buffer never runs, the comparison is with an older one, or nothing: at worst a cut is seen).
    pub(crate) fn recorded_inline(&mut self) {
        self.have = true;
    }

    /// The result buffer (16 bytes: compared, cut, sum, samples), which the hold inside DLSS's buffer copies
    /// out for the host.
    pub(crate) fn result_buffer(&self) -> vk::Buffer {
        self.result.buffer
    }

    /// The reading in a result's four words.
    pub(crate) fn reading_of(w: [u32; 4]) -> Reading {
        let [compared, cut, sum, n] = w;
        Reading { cut: compared != 0 && cut != 0, mean: if compared != 0 { sum / n.max(1) } else { 0 } }
    }

    /// The last submitted hold's reading, once its fence was waited on (`None` when there is none to
    /// read).
    pub(crate) fn take(&mut self) -> Option<(Reading, Recorded)> {
        let recorded = self.pending.take()?;
        let mut w = [0u32; 4];
        // SAFETY: host-coherent, mapped for 16 bytes; the hold that wrote it was waited on and its pass
        // closed with a barrier to the host.
        unsafe { std::ptr::copy_nonoverlapping(self.result.ptr.cast::<u32>(), w.as_mut_ptr(), 4) };
        Some((Self::reading_of(w), recorded))
    }

    /// # Safety
    /// Nothing of the pass may still be pending.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; destroying the pool frees the set.
        unsafe {
            for &p in &self.pipelines {
                device.destroy_pipeline(p, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            self.thumb.destroy(device);
            self.partial.destroy(device);
            self.result.destroy(device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preupscale::hdr::OwnImage;
    use crate::preupscale::own_host_buffer_with;

    #[test]
    fn the_override_switches_off_or_sets_the_threshold() {
        assert_eq!(parse_env(None), Ok(THRESHOLD));
        assert_eq!(parse_env(Some("")), Ok(THRESHOLD));
        assert_eq!(parse_env(Some("on")), Ok(THRESHOLD));
        assert_eq!(parse_env(Some("off")), Ok(0));
        assert_eq!(parse_env(Some(" 0 ")), Ok(0));
        assert_eq!(parse_env(Some("25")), Ok(25));
        assert_eq!(parse_env(Some("255")), Ok(255));
        assert_eq!(parse_env(Some("256")), Err("256".into()));
        assert_eq!(parse_env(Some("-3")), Err("-3".into()));
        assert_eq!(parse_env(Some("yes")), Err("yes".into()));
    }

    #[test]
    fn a_cut_resets_on_the_gpu_on_a_network_frame_and_is_carried_to_the_next_one_from_a_cached_hold() {
        let cut = Reading { cut: true, mean: 87 };
        let none = Reading { cut: false, mean: 3 };
        let network = Recorded { network: true, history: true };
        let fresh = Recorded { network: true, history: false };
        let cached = Recorded { network: false, history: false };
        assert_eq!(effect(cut, network), Effect::ResetOnGpu(87));
        assert_eq!(effect(cut, fresh), Effect::None, "a frame without history is not reset twice");
        assert_eq!(effect(cut, cached), Effect::Latch(87));
        for r in [network, fresh, cached] {
            assert_eq!(effect(none, r), Effect::None);
        }
    }

    /// The detector outlives the native pass whose state buffer it writes the cut into (rebuilt for another
    /// model resolution), and a destroyed buffer's handle comes back for the next one made: the set is written
    /// again when the passes behind the same handles changed (`made`), and only then. The test points the set
    /// at another buffer behind `bind`'s back, as a set left on a destroyed buffer whose handle the new one got:
    /// the same passes leave it there, another native or HDR pass puts it back on the buffer named.
    #[test]
    fn the_set_is_written_again_for_another_pass_under_the_same_handles() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, queue, family) = (&gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        const W: u32 = 16;
        const H: u32 = 8;
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).unwrap();
            let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).unwrap()[0];
            let one_shot = |record: &dyn Fn(vk::CommandBuffer)| {
                device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                device.end_command_buffer(cmd).unwrap();
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
                device.queue_wait_idle(queue).unwrap();
            };
            let encoded = OwnImage::new(device, instance, physical, W, H, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            one_shot(&|cmd| {
                let to_general = vk::ImageMemoryBarrier::builder().old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).image(encoded.image).subresource_range(range).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
                device.cmd_clear_color_image(cmd, encoded.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.5; 4] }, &[range]);
            });
            let [state, elsewhere] = [(); 2].map(|()| own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap());
            let mut pass = ScenePass::build(device, instance, physical, W, H).expect("the scene-cut pass builds");
            // The state word each hold writes (`cut`), set beforehand in both buffers: which one the hold wrote.
            let written = |pass: &mut ScenePass| -> [bool; 2] {
                for b in [&state, &elsewhere] {
                    std::ptr::write(b.ptr.cast::<u32>().add(2), 7);
                }
                one_shot(&|cmd| pass.record(device, cmd, THRESHOLD));
                pass.submitted(Recorded { network: true, history: true });
                let _ = pass.take();
                [&state, &elsewhere].map(|b| *b.ptr.cast::<u32>().add(2) != 7)
            };
            pass.bind(device, encoded.view, state.buffer, [1, 1]);
            assert_eq!(written(&mut pass), [true, false], "bound to the state buffer");
            let set = pass.set;
            let stale = || {
                let info = [vk::DescriptorBufferInfo { buffer: elsewhere.buffer, offset: 0, range: vk::WHOLE_SIZE }];
                let w = vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info).build();
                device.update_descriptor_sets(&[w], &[]);
            };
            stale();
            pass.bind(device, encoded.view, state.buffer, [1, 1]);
            assert_eq!(written(&mut pass), [false, true], "the same handles of the same passes: the set is not written");
            for made in [[1, 2], [2, 2]] {
                stale();
                pass.bind(device, encoded.view, state.buffer, made);
                assert_eq!(written(&mut pass), [true, false], "another pass ({made:?}) behind the same handles: the set is written again");
            }
            pass.destroy(device);
            encoded.destroy(device);
            state.destroy(device);
            elsewhere.destroy(device);
            device.destroy_command_pool(pool, None);
        }
    }

    /// The 1.x helper's `luma_thumbnail_rgba16f` and `is_scene_cut`, on the CPU: the shader's reference.
    fn cpu_thumbnail(frame: &[[f32; 3]], width: u32, height: u32) -> Vec<u32> {
        let q = |v: f32| if v.is_nan() { 0 } else { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32 };
        let mut out = Vec::new();
        for y in (0..height).step_by(STEP as usize) {
            for x in (0..width).step_by(STEP as usize) {
                let [r, g, b] = frame[(y * width + x) as usize];
                out.push((q(r) + q(g) + q(b)) / 3);
            }
        }
        out
    }

    fn cpu_sum(a: &[u32], b: &[u32]) -> u32 {
        a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y)).sum()
    }

    fn half(v: f32) -> u16 {
        crate::preupscale::tests::f32_to_f16(v)
    }

    /// The frame as the encoded image holds it: on the half grid.
    fn on_half_grid(frame: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
        frame.into_iter().map(|c| c.map(|v| crate::preupscale::f16_to_f32(half(v)))).collect()
    }

    /// The detector on lavapipe over a sequence of frames, against the helper's CPU detector: a still frame
    /// never trips it, a small change does not, a different picture trips it once (and the frame after it,
    /// the same picture again, does not), the decision reaches the native pass's state word, the first hold
    /// compares with nothing, and the threshold is the one passed.
    #[test]
    fn a_still_frame_never_cuts_and_a_new_picture_cuts_once() {
        let Some(gpu) = crate::composition::gpu::test_device() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        let (instance, physical, device, queue, family) = (&gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
        // An odd extent: the thumbnail's last column and row are partial; several first-pass workgroups.
        const W: u32 = 203;
        const H: u32 = 101;
        // SAFETY: a fresh device; every object below is used on its queue only and waited for.
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            let one_shot = |record: &dyn Fn(vk::CommandBuffer)| {
                let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1)).unwrap()[0];
                device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                record(cmd);
                device.end_command_buffer(cmd).unwrap();
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], vk::Fence::null()).unwrap();
                device.queue_wait_idle(queue).unwrap();
                device.free_command_buffers(pool, &[cmd]);
            };
            let (pw, ph) = (W + 1, H + 1);
            let encoded = OwnImage::new(device, instance, physical, pw, ph, vk::ImageUsageFlags::TRANSFER_DST).unwrap();
            let staging = own_host_buffer_with(device, instance, physical, u64::from(pw * ph) * 8, vk::BufferUsageFlags::TRANSFER_SRC).unwrap();
            let state = own_host_buffer_with(device, instance, physical, 16, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            one_shot(&|cmd| {
                let to_general = vk::ImageMemoryBarrier::builder().old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::GENERAL).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE).src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED).image(encoded.image).subresource_range(range).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
            });
            let mut pass = ScenePass::build(device, instance, physical, W, H).expect("the scene-cut pass builds");
            pass.bind(device, encoded.view, state.buffer, [1, 1]);
            // The frame's valid part (the padding stays whatever it is: never sampled).
            let upload = |frame: &[[f32; 3]]| {
                let texels: Vec<u16> = (0..ph).flat_map(|y| (0..pw).flat_map(move |x| {
                    let [r, g, b] = if x < W && y < H { frame[(y * W + x) as usize] } else { [0.0; 3] };
                    [half(r), half(g), half(b), half(1.0)]
                })).collect();
                std::ptr::copy_nonoverlapping(texels.as_ptr().cast::<u8>(), staging.ptr, texels.len() * 2);
                one_shot(&|cmd| {
                    let region = vk::BufferImageCopy { image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, image_extent: vk::Extent3D { width: pw, height: ph, depth: 1 }, ..Default::default() };
                    device.cmd_copy_buffer_to_image(cmd, staging.buffer, encoded.image, vk::ImageLayout::GENERAL, &[region]);
                });
            };
            let state_cut = || *state.ptr.cast::<u32>().add(2);
            let hold = |pass: &mut ScenePass, frame: &[[f32; 3]], threshold: u32| -> Reading {
                upload(frame);
                // The state word set beforehand: the pass writes it on every hold.
                std::ptr::write(state.ptr.cast::<u32>().add(2), 7);
                one_shot(&|cmd| pass.record(device, cmd, threshold));
                pass.submitted(Recorded { network: true, history: true });
                let (reading, recorded) = pass.take().expect("a reading");
                assert_eq!(recorded, Recorded { network: true, history: true });
                assert_eq!(state_cut(), u32::from(reading.cut), "the frame shaders' word is the decision");
                assert!(pass.take().is_none(), "one reading per hold");
                reading
            };
            // A textured picture with values outside [0, 1] and a NaN (the encode clamps; the helper's rule
            // anyway), and a different one: dark where it was bright.
            let a: Vec<[f32; 3]> = on_half_grid((0..W * H).map(|i| {
                let (x, y) = (i % W, i / W);
                let v = ((x * 7 + y * 13) % 97) as f32 / 96.0;
                if (x, y) == (8, 8) { [f32::NAN, 1.5, -0.25] } else { [v, 0.5 * v + 0.2, 1.0 - v] }
            }).collect());
            let b = on_half_grid(a.iter().map(|&[r, _, bl]| [1.0 - r.clamp(0.0, 1.0), 0.05, (bl * 0.3).clamp(0.0, 1.0)]).collect());
            let slightly = on_half_grid(a.iter().map(|&[r, g, bl]| [r + 0.02, g + 0.02, bl - 0.02]).collect());
            let (ta, tb, ts) = (cpu_thumbnail(&a, W, H), cpu_thumbnail(&b, W, H), cpu_thumbnail(&slightly, W, H));
            let n = samples(W, H);
            assert_eq!(ta.len() as u32, n);
            let mean_ab = cpu_sum(&ta, &tb) / n;
            assert!(mean_ab >= THRESHOLD, "the test's second picture is a cut by the helper's rule ({mean_ab})");
            assert!(cpu_sum(&ta, &ts) / n < THRESHOLD, "and the slight change is not");

            // The first hold has nothing to compare with.
            assert_eq!(hold(&mut pass, &a, THRESHOLD), Reading { cut: false, mean: 0 });
            // A still frame, many times.
            for _ in 0..5 {
                assert_eq!(hold(&mut pass, &a, THRESHOLD), Reading { cut: false, mean: 0 }, "a still frame never cuts");
            }
            assert_eq!(hold(&mut pass, &slightly, THRESHOLD), Reading { cut: false, mean: cpu_sum(&ta, &ts) / n }, "a slight change is no cut");
            assert!(!hold(&mut pass, &a, THRESHOLD).cut);
            // The new picture: one cut, at the helper's mean difference.
            assert_eq!(hold(&mut pass, &b, THRESHOLD), Reading { cut: true, mean: mean_ab }, "a new picture cuts");
            assert_eq!(hold(&mut pass, &b, THRESHOLD), Reading { cut: false, mean: 0 }, "once");
            // A threshold above the difference keeps the history; one at it cuts.
            assert!(!hold(&mut pass, &a, mean_ab + 1).cut, "under the threshold");
            assert!(hold(&mut pass, &b, mean_ab).cut, "at the threshold");

            pass.destroy(device);
            encoded.destroy(device);
            staging.destroy(device);
            state.destroy(device);
            device.destroy_command_pool(pool, None);
        }
    }
}
