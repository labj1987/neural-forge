//! The sharpening of the finished frame (docs/USER_GUIDE.md, "Sharpening"): FidelityFX CAS (contrast adaptive
//! sharpening, ported, MIT; ATTRIBUTION.md) and Clarity (local contrast of luma), both off by default, applied at
//! present to the swapchain image, after everything else the layer does to the frame, on every presented frame.
//!
//! **Why at present.** It is the one stage every game reaches whatever its path (the model before the upscaler,
//! after DLSS, on the finished frame, or not at all), and it is where CAS belongs: CAS wants a display-referred
//! picture with 1 at the display's peak ("It is best to run CAS after tonemapping", ffx_cas.h), which the frames
//! before the game's own post-processing (DLSS's input and output, scene-linear HDR) are not. The HUD is already
//! drawn there, so the HUD regions of the Model tab (`hud_regions`) apply: `keep` leaves its rectangle untouched,
//! `soften` scales the sharpening by its strength, over the same feather as the model's. The HUD detection is the
//! after-the-upscaler path's own and is not used here. Clarity's "respect depth" is not offered: the only depth the
//! layer has is DLSS's input depth at the render size, copied only inside a hold before the upscaler and only with
//! the edit cache on (ARCHITECTURE.md 4.9), of a frame that the game's post-processing, the HUD and frame
//! generation's frames have since moved away from; Clarity's blur stops at strong luma edges instead (`edges`).
//!
//! **How.** One submit on the present's queue, after the layer's other work for the frame: the image is copied
//! into a buffer as raw words, `shaders/present_sharpen.comp` writes the filtered words into a second buffer, and
//! they are copied back; a semaphore of its own (one a swapchain image, like the relay's) is what the present then
//! waits on. With both filters off nothing is recorded or allocated, and the frame is presented exactly as before.
//! The work of consecutive frames shares its buffers and is ordered by the queue (each submit opens with a barrier
//! from all earlier commands); a present on another queue first waits for the earlier submits.

use std::sync::LazyLock;

use ash::vk;
use neural_forge_protocol::{HudRegion, SharpenSettings};

const SPV: &[u8] = include_bytes!("../shaders/present_sharpen.spv");

/// The test override: `off`, or `key=value` pairs separated by commas (`cas`, `clarity`, `radius`, `midtones`,
/// `edges`), each replacing that setting for the whole process (clamped to its range). Read once, logged once.
pub(crate) const ENV: &str = "NEURAL_FORGE_SHARPEN";

/// Command buffers in flight (each frame's submit has its own).
const SLOTS: usize = 3;

/// The shader's workgroup side.
const GROUP: u32 = 8;

/// GPU readings per logged median.
const WINDOW: usize = 300;

/// The setting names the override's keys stand for, in [`Override::values`]' order.
const KEYS: [(&str, &str); 5] =
    [("cas", "sharpen_cas"), ("clarity", "clarity_strength"), ("radius", "clarity_radius"), ("midtones", "clarity_midtones"), ("edges", "clarity_edges")];

/// What [`ENV`] asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Override {
    /// Both filters off, whatever the settings say.
    pub off: bool,
    /// Each setting the override names (in [`KEYS`]' order), clamped to its range.
    pub values: [Option<f32>; 5],
}

impl Override {
    /// The settings `s` with the override applied.
    pub(crate) fn apply(&self, s: SharpenSettings) -> SharpenSettings {
        if self.off {
            return SharpenSettings { cas: 0.0, clarity: 0.0, ..s };
        }
        let v = |k: usize, d: f32| self.values[k].unwrap_or(d);
        SharpenSettings { cas: v(0, s.cas), clarity: v(1, s.clarity), radius: v(2, s.radius), midtones: v(3, s.midtones), edges: v(4, s.edges) }
    }

    fn is_none(&self) -> bool {
        *self == Self::default()
    }
}

/// [`ENV`]'s value parsed: unset or empty is no override; `off` (or `0`); else `key=value,...`. An unknown key or a
/// value that is not a number is an error (the whole override is then ignored).
pub(crate) fn parse_env(raw: Option<&str>) -> Result<Override, String> {
    let Some(raw) = raw.map(str::trim).filter(|r| !r.is_empty()) else { return Ok(Override::default()) };
    if raw == "off" || raw == "0" {
        return Ok(Override { off: true, ..Override::default() });
    }
    let mut o = Override::default();
    for pair in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair:?} is not key=value"))?;
        let k = KEYS.iter().position(|(k, _)| *k == key.trim()).ok_or_else(|| format!("unknown key {:?}", key.trim()))?;
        let v = value.trim().parse::<f32>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("{pair:?}: not a number"))?;
        let (lo, hi) = neural_forge_protocol::setting_bounds(KEYS[k].1).expect("a sharpening setting");
        o.values[k] = Some(v.clamp(lo, hi));
    }
    Ok(o)
}

/// This process's override ([`parse_env`]). Cached; logged once when set.
pub(crate) fn env_override() -> Override {
    static ONCE: LazyLock<Override> = LazyLock::new(|| {
        let raw = neural_forge_protocol::env::var(ENV);
        match parse_env(raw.as_deref()) {
            Ok(o) if o.off => {
                crate::log!("[sharpen] {ENV}=off: no sharpening, whatever the settings say");
                o
            }
            Ok(o) if o.is_none() => o,
            Ok(o) => {
                let named: Vec<String> = KEYS.iter().zip(o.values).filter_map(|((k, _), v)| v.map(|v| format!("{k} {v}"))).collect();
                crate::log!("[sharpen] {ENV}: {} (in place of the settings)", named.join(", "));
                o
            }
            Err(why) => {
                crate::log!("[sharpen] {ENV}={:?} ignored: {why} (off, or key=value pairs of cas, clarity, radius, midtones, edges)", raw.unwrap_or_default());
                Override::default()
            }
        }
    });
    *ONCE
}

/// The frame to sharpen: one of the present's swapchain images.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub image: vk::Image,
    /// The layout the image is in, and is left in (`PRESENT_SRC_KHR` at a present).
    pub layout: vk::ImageLayout,
    pub width: u32,
    pub height: u32,
    /// Blue first in the words (BGRA8, A2R10G10B10).
    pub bgr: bool,
    /// Ten bits a channel (A2B10G10R10 / A2R10G10B10: HDR10 or 10-bit SDR); else eight.
    pub ten_bit: bool,
    /// BT.2020 primaries (HDR10), for the luma weights; else BT.709 (8-bit and 10-bit SDR).
    pub bt2020: bool,
}

/// The shader's push constants (field for field).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Push {
    dims: [u32; 4],
    flags: [u32; 4],
    cas: [f32; 4],
    clarity: [f32; 4],
    misc: [f32; 4],
}

/// CAS's peak weight for a strength: `-s / lerp(8, 5, s)`, 0 at 0, CAS's strongest (-1/5) at 1.
pub(crate) fn cas_peak(strength: f32) -> f32 {
    let s = strength.clamp(0.0, 1.0);
    -s / (8.0 + (5.0 - 8.0) * s)
}

/// Clarity's radius in half-size pixels (the taps each side of the blur) for a frame `height` lines tall: the
/// setting is in pixels at 1440 lines.
pub(crate) fn half_radius(radius: f32, height: u32) -> u32 {
    ((radius * height as f32 / 1440.0 / 2.0).round() as u32).clamp(1, 32)
}

/// The edge stop's `1 / (2 sigma^2)` on luma (0-1) for `edges`: 0 (no stop) at 0, sigma 0.5 down to 0.03 above.
pub(crate) fn edge_stop(edges: f32) -> f32 {
    if edges <= 0.0 {
        return 0.0;
    }
    let sigma = 0.5 + (0.03 - 0.5) * edges.clamp(0.0, 1.0);
    1.0 / (2.0 * sigma * sigma)
}

/// A HUD region's feather in pixels at an output `height`, as the model's (`native_post::Hud::region_feather`).
pub(crate) fn region_feather(height: u32) -> f32 {
    (height as f32 / 360.0).round().max(2.0)
}

impl Push {
    pub(crate) fn of(frame: &Frame, s: &SharpenSettings, regions: u32, phase: u32) -> Self {
        let r = half_radius(s.radius, frame.height);
        let sigma = r as f32 / 2.0;
        Push {
            dims: [frame.width, frame.height, phase, regions],
            flags: [u32::from(frame.bgr) | (u32::from(frame.ten_bit) << 1) | (u32::from(frame.bt2020) << 2), u32::from(s.cas > 0.0), u32::from(s.clarity > 0.0), 0],
            cas: [cas_peak(s.cas), 0.0, 0.0, 0.0],
            clarity: [s.clarity, r as f32, s.midtones.clamp(0.0, 1.0), edge_stop(s.edges)],
            misc: [region_feather(frame.height), 1.0 / (2.0 * sigma * sigma), 0.0, 0.0],
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `repr(C)` plain data, read as bytes for its own size.
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

/// The regions as the shader reads them: per region (left, top, right, bottom) in pixels, then (mode, strength).
pub(crate) fn region_words(regions: &[HudRegion], width: u32, height: u32) -> Vec<[f32; 4]> {
    let (w, h) = (width as f32, height as f32);
    regions.iter().flat_map(|r| [[r.x * w, r.y * h, (r.x + r.w) * w, (r.y + r.h) * h], [r.mode as f32, r.strength, 0.0, 0.0]]).collect()
}

/// A buffer and its memory.
struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Mapped (host-visible ones only).
    ptr: *mut u8,
}

impl Buf {
    /// # Safety
    /// `device` is live.
    unsafe fn new(device: &ash::Device, props: &vk::PhysicalDeviceMemoryProperties, bytes: u64, usage: vk::BufferUsageFlags, host: bool) -> Option<Self> {
        let info = vk::BufferCreateInfo::default().size(bytes.max(16)).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
        // SAFETY: just created.
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let want = if host { vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT } else { vk::MemoryPropertyFlags::DEVICE_LOCAL };
        let index =
            (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want));
        // SAFETY: valid allocation info.
        let memory = index
            .and_then(|i| unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(i), None) }.ok());
        // SAFETY: sized for each other; nothing submitted uses either yet.
        let bound = memory.filter(|&m| unsafe { device.bind_buffer_memory(buffer, m, 0) }.is_ok());
        let ptr = match (bound, host) {
            // SAFETY: host-visible memory of this allocation, mapped whole.
            (Some(m), true) => unsafe { device.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.ok().map(|p| p.cast::<u8>()),
            (Some(_), false) => Some(std::ptr::null_mut()),
            (None, _) => None,
        };
        match (bound, ptr) {
            (Some(memory), Some(ptr)) => Some(Self { buffer, memory, ptr }),
            _ => {
                // SAFETY: nothing submitted uses either.
                unsafe {
                    if let Some(m) = memory {
                        device.free_memory(m, None);
                    }
                    device.destroy_buffer(buffer, None);
                }
                None
            }
        }
    }

    /// # Safety
    /// Nothing submitted may still use it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; freeing mapped memory unmaps it.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// The frame-sized buffers, shared by the slots.
struct Buffers {
    width: u32,
    height: u32,
    /// The frame's words, and the filtered ones.
    src: Buf,
    dst: Buf,
    /// The half-size luma, the row pass and the base.
    luma: Buf,
    rows: Buf,
    base: Buf,
}

impl Buffers {
    /// # Safety
    /// `device` is live.
    unsafe fn new(device: &ash::Device, props: &vk::PhysicalDeviceMemoryProperties, width: u32, height: u32) -> Option<Self> {
        let full = u64::from(width) * u64::from(height) * 4;
        let half = u64::from(width.div_ceil(2)) * u64::from(height.div_ceil(2)) * 4;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let copy = storage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut made: Vec<Buf> = Vec::with_capacity(5);
        for (bytes, usage) in [(full, copy), (full, copy), (half, storage), (half, storage), (half, storage)] {
            // SAFETY: forwarded.
            match unsafe { Buf::new(device, props, bytes, usage, false) } {
                Some(b) => made.push(b),
                None => {
                    for b in &made {
                        // SAFETY: never submitted.
                        unsafe { b.destroy(device) };
                    }
                    return None;
                }
            }
        }
        let mut it = made.into_iter();
        let mut next = || it.next().expect("five buffers");
        Some(Self { width, height, src: next(), dst: next(), luma: next(), rows: next(), base: next() })
    }

    /// # Safety
    /// Nothing submitted may still use them.
    unsafe fn destroy(&self, device: &ash::Device) {
        for b in [&self.src, &self.dst, &self.luma, &self.rows, &self.base] {
            // SAFETY: forwarded.
            unsafe { b.destroy(device) };
        }
    }
}

/// One frame's command buffer, fence, timer, regions and descriptor set.
struct Slot {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    timer: Option<crate::gpu_timer::GpuTimer>,
    regions: Buf,
    set: vk::DescriptorSet,
    /// Submitted since its fence was last seen signalled.
    submitted: bool,
}

/// Room for this many regions' words in a slot's buffer.
const REGION_WORDS: usize = neural_forge_protocol::MAX_HUD_REGIONS * 2;

/// The sharpening's pipeline and resources on one device, for presents on one queue family.
pub(crate) struct Sharpen {
    family: u32,
    props: vk::PhysicalDeviceMemoryProperties,
    pool: vk::CommandPool,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    slots: Vec<Slot>,
    next: usize,
    buffers: Option<Buffers>,
    /// The queue the last submit went to.
    last_queue: Option<vk::Queue>,
    /// The semaphore each swapchain image's present waits on (`device.rs` notes and retires them as the relay's).
    pub(crate) semaphores: crate::present_sync::PresentSemaphores,
    /// GPU milliseconds of the frames since the last logged median.
    readings: Vec<f32>,
    /// What was last logged as in use (the settings and the frame's size), to log a change once.
    said: Option<(SharpenSettings, u32, u32)>,
}

// SAFETY: plain handles and mappings of the layer's own, only used behind the device's `State` mutex.
unsafe impl Send for Sharpen {}

impl Sharpen {
    /// The pipeline, the pool and the slots for presents on `family`; `None` when the family cannot run compute
    /// or anything cannot be made (logged once by the caller).
    ///
    /// # Safety
    /// `device` is live and `physical_device` its own.
    pub(crate) unsafe fn new(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32) -> Option<Self> {
        // SAFETY: plain queries.
        let (families, props) =
            unsafe { (instance.get_physical_device_queue_family_properties(physical_device), instance.get_physical_device_memory_properties(physical_device)) };
        if !families.get(family as usize).is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
            return None;
        }
        let mut s = Self {
            family,
            props,
            pool: vk::CommandPool::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            slots: Vec::new(),
            next: 0,
            buffers: None,
            last_queue: None,
            semaphores: crate::present_sync::PresentSemaphores::default(),
            readings: Vec::with_capacity(WINDOW),
            said: None,
        };
        // SAFETY: forwarded; on failure everything made so far is destroyed (never submitted).
        if unsafe { s.build(device, instance, physical_device) }.is_none() {
            // SAFETY: nothing was submitted.
            unsafe { s.destroy(device) };
            return None;
        }
        Some(s)
    }

    /// # Safety
    /// As [`Self::new`].
    unsafe fn build(&mut self, device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<()> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        // SAFETY: valid create infos; every handle is kept in `self` as it is made.
        unsafe {
            self.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(self.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .ok()?;
            self.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let push = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(std::mem::size_of::<Push>() as u32);
            self.pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&self.set_layout))
                        .push_constant_ranges(std::slice::from_ref(&push)),
                    None,
                )
                .ok()?;
            let code = ash::util::read_spv(&mut std::io::Cursor::new(SPV)).ok()?;
            let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok()?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.pipeline_layout);
            let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None);
            device.destroy_shader_module(module, None);
            self.pipeline = pipeline.ok()?.first().copied()?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 * SLOTS as u32 }];
            self.descriptor_pool =
                device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(SLOTS as u32).pool_sizes(&sizes), None).ok()?;
            let layouts = [self.set_layout; SLOTS];
            let sets =
                device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptor_pool).set_layouts(&layouts)).ok()?;
            // Through the loader's dispatch data: these are dispatchable objects made below the loader's trampoline.
            let cmd_info = vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(SLOTS as u32);
            let cmds = crate::loader_data::allocate_commands(device, &cmd_info).ok()?;
            for (set, cmd) in sets.into_iter().zip(cmds) {
                let fence = device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).ok()?;
                let regions = match Buf::new(device, &self.props, (REGION_WORDS * 16) as u64, vk::BufferUsageFlags::STORAGE_BUFFER, true) {
                    Some(r) => r,
                    None => {
                        device.destroy_fence(fence, None);
                        return None;
                    }
                };
                let timer = crate::gpu_timer::GpuTimer::new(device, instance, physical_device, self.family);
                self.slots.push(Slot { cmd, fence, timer, regions, set, submitted: false });
            }
        }
        Some(())
    }

    /// Waits (bounded) for every submitted slot; false when one did not finish.
    pub(crate) fn wait_idle(&mut self, device: &ash::Device) -> bool {
        let fences: Vec<vk::Fence> = self.slots.iter().filter(|s| s.submitted).map(|s| s.fence).collect();
        if fences.is_empty() {
            return true;
        }
        // SAFETY: the slots' own fences.
        let r = unsafe { device.wait_for_fences(&fences, true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
        if crate::note_fence_wait(device.handle(), r, "sharpen::wait_idle").is_err() {
            return false;
        }
        for s in &mut self.slots {
            if s.submitted {
                s.submitted = false;
                if let Some(t) = &mut s.timer {
                    t.read_if_pending(device);
                    if let Some(ms) = t.take_reading() {
                        self.readings.push(ms);
                    }
                }
            }
        }
        true
    }

    /// Sharpens `frame` with `settings` (nothing when [`SharpenSettings::any`] is false), keeping `regions` (the HUD)
    /// out of it, in one submit on `queue` ordered after everything submitted on it before. Returns the semaphore the
    /// present must wait on, `None` when nothing was submitted (the frame is then presented as it is).
    ///
    /// # Safety
    /// `queue` is of this pass's family, externally synchronized for the call; `frame.image` is live, in
    /// `frame.layout`, with `TRANSFER_SRC` and `TRANSFER_DST` usage, and the application's rendering of it is
    /// ordered before this submit (its present waits relayed on `queue`, or none).
    pub(crate) unsafe fn apply(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        frame: &Frame,
        settings: &SharpenSettings,
        regions: &[HudRegion],
    ) -> Option<vk::Semaphore> {
        if !settings.any() || frame.width == 0 || frame.height == 0 {
            return None;
        }
        // Another queue than the last submit's: nothing orders this frame's use of the shared buffers after it.
        if self.last_queue.is_some_and(|q| q != queue) && !self.wait_idle(device) {
            return None;
        }
        if self.buffers.as_ref().is_none_or(|b| (b.width, b.height) != (frame.width, frame.height)) {
            if !self.wait_idle(device) {
                return None;
            }
            if let Some(old) = self.buffers.take() {
                // SAFETY: every slot that used them was waited for.
                unsafe { old.destroy(device) };
            }
            // SAFETY: forwarded.
            self.buffers = Some(unsafe { Buffers::new(device, &self.props, frame.width, frame.height) }?);
            let b = self.buffers.as_ref()?;
            for slot in &self.slots {
                let infos = [b.src.buffer, b.dst.buffer, b.luma.buffer, b.rows.buffer, b.base.buffer, slot.regions.buffer]
                    .map(|buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }]);
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(n, i)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(slot.set)
                            .dst_binding(n as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(i)
                    })
                    .collect();
                // SAFETY: no slot is in use (waited above); the buffers are live.
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
        }
        let semaphore = self.semaphores.get(frame.image, || unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok() })?;
        let idx = self.next;
        let slot = &mut self.slots[idx];
        if slot.submitted {
            // SAFETY: the slot's own fence.
            let r = unsafe { device.wait_for_fences(&[slot.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
            if crate::note_fence_wait(device.handle(), r, "sharpen::apply slot reuse").is_err() {
                return None;
            }
            slot.submitted = false;
            if let Some(t) = &mut slot.timer {
                t.read_if_pending(device);
                if let Some(ms) = t.take_reading() {
                    self.readings.push(ms);
                }
            }
        }
        let words = region_words(regions, frame.width, frame.height);
        let count = words.len().min(REGION_WORDS) / 2;
        // SAFETY: host-coherent, mapped, sized for `REGION_WORDS` words; the slot's last use was waited for.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), slot.regions.ptr, count * 2 * 16) };
        let b = self.buffers.as_ref()?;
        // SAFETY: forwarded from this function's contract; the slot's buffer is not in use.
        unsafe { record(device, slot, b, self.pipeline_layout, self.pipeline, frame, settings, count as u32) }.ok()?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&slot.cmd)).signal_semaphores(std::slice::from_ref(&semaphore));
        // SAFETY: the fence is signalled (waited above, or never submitted), the buffer recorded; `queue` per contract.
        unsafe {
            crate::note_vk(device.handle(), device.reset_fences(&[slot.fence])).ok()?;
            if crate::note_vk(device.handle(), device.queue_submit(queue, &[submit], slot.fence)).is_err() {
                // Nothing was submitted: the slot is free (its fence, unsignalled now, is reset again before its
                // next submit and never waited for until then).
                return None;
            }
        }
        slot.submitted = true;
        if let Some(t) = &mut slot.timer {
            t.mark_submitted();
        }
        self.next = (idx + 1) % SLOTS;
        self.last_queue = Some(queue);
        self.note(settings, frame);
        Some(semaphore)
    }

    /// Logs the settings in use when they (or the frame's size) change, and the GPU time's median every
    /// [`WINDOW`] readings.
    fn note(&mut self, s: &SharpenSettings, frame: &Frame) {
        let now = Some((*s, frame.width, frame.height));
        if self.said != now {
            self.said = now;
            crate::log!(
                "[sharpen] {}x{}: CAS {:.2}, Clarity {:.2} (radius {:.0}, midtones {:.2}, edges {:.2})",
                frame.width,
                frame.height,
                s.cas,
                s.clarity,
                s.radius,
                s.midtones,
                s.edges
            );
        }
        if self.readings.len() >= WINDOW {
            let mut r = std::mem::take(&mut self.readings);
            r.sort_by(f32::total_cmp);
            crate::log!("[sharpen] gpu_ms median={:.3} p95={:.3} ({} frames)", r[r.len() / 2], r[r.len() * 95 / 100], r.len());
            crate::logging::flush();
            r.clear();
            self.readings = r;
        }
    }

    /// The family this pass was built for.
    pub(crate) fn family(&self) -> u32 {
        self.family
    }

    /// # Safety
    /// Nothing of this pass may still be pending (the device idle, or every slot waited for), and no present still
    /// waits on its semaphores.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; destroying the pools frees their buffers and sets; null handles are ignored.
        unsafe {
            for s in &self.slots {
                device.destroy_fence(s.fence, None);
                s.regions.destroy(device);
                if let Some(t) = &s.timer {
                    t.destroy(device);
                }
            }
            if let Some(b) = &self.buffers {
                b.destroy(device);
            }
            self.semaphores.destroy(device);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// Records one frame's work into `slot`'s command buffer.
///
/// # Safety
/// As [`Sharpen::apply`]; the slot's buffer is not in use.
#[allow(clippy::too_many_arguments)]
unsafe fn record(
    device: &ash::Device,
    slot: &Slot,
    b: &Buffers,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    frame: &Frame,
    s: &SharpenSettings,
    regions: u32,
) -> ash::prelude::VkResult<()> {
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let image_barrier = |old, new, src, dst| {
        vk::ImageMemoryBarrier::default()
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(src)
            .dst_access_mask(dst)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(frame.image)
            .subresource_range(range)
    };
    let mem = |src, dst| vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst);
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D { width: frame.width, height: frame.height, depth: 1 },
    };
    let (full_x, full_y) = (frame.width.div_ceil(GROUP), frame.height.div_ceil(GROUP));
    let (half_x, half_y) = (frame.width.div_ceil(2).div_ceil(GROUP), frame.height.div_ceil(2).div_ceil(GROUP));
    let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
    let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    // SAFETY: per this function's contract.
    unsafe {
        device.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(slot.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        // Everything earlier on the queue (the game's frame, the layer's other work, the last frame's use of the
        // buffers) before the copy reads the image and writes the buffer.
        device.cmd_pipeline_barrier(
            slot.cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[mem(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ, vk::AccessFlags::TRANSFER_WRITE)],
            &[],
            &[image_barrier(frame.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::TRANSFER_READ)],
        );
        if let Some(t) = &slot.timer {
            t.record_start(device, slot.cmd);
        }
        device.cmd_copy_image_to_buffer(slot.cmd, frame.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, b.src.buffer, std::slice::from_ref(&copy));
        device.cmd_pipeline_barrier(
            slot.cmd,
            vk::PipelineStageFlags::TRANSFER,
            compute,
            vk::DependencyFlags::empty(),
            &[mem(vk::AccessFlags::TRANSFER_WRITE, rw)],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(slot.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(slot.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, std::slice::from_ref(&slot.set), &[]);
        let phases: &[u32] = if s.clarity > 0.0 { &[0, 1, 2, 3] } else { &[3] };
        for (n, &phase) in phases.iter().enumerate() {
            if n > 0 {
                device.cmd_pipeline_barrier(slot.cmd, compute, compute, vk::DependencyFlags::empty(), &[mem(vk::AccessFlags::SHADER_WRITE, rw)], &[], &[]);
            }
            let push = Push::of(frame, s, regions, phase);
            device.cmd_push_constants(slot.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push.bytes());
            if phase == 3 {
                device.cmd_dispatch(slot.cmd, full_x, full_y, 1);
            } else {
                device.cmd_dispatch(slot.cmd, half_x, half_y, 1);
            }
        }
        // The filtered words to the copy back, and the image from the copy's read to its write.
        device.cmd_pipeline_barrier(
            slot.cmd,
            compute | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[mem(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ)],
            &[],
            &[image_barrier(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::TRANSFER_WRITE,
            )],
        );
        device.cmd_copy_buffer_to_image(slot.cmd, b.dst.buffer, frame.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&copy));
        // Back to the image's layout for the present and anything later on the queue (a capture's readback).
        device.cmd_pipeline_barrier(
            slot.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[image_barrier(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                frame.layout,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            )],
        );
        if let Some(t) = &slot.timer {
            t.record_end(device, slot.cmd);
        }
        device.end_command_buffer(slot.cmd)
    }
}

#[cfg(test)]
mod tests;
