//! The native backend (docs/NATIVE_BACKEND.md, "Phase 1"): the network runs on the game's own device,
//! inside the game's own DLSS submit, instead of in the helper.
//!
//! A hold in native mode submits three command buffers in front of DLSS's launch buffer, with no CPU
//! wait between them and none for them:
//!
//! - `C`, the capture (`super::record_capture`): the exposure texel, the HDR encode into the encoded
//!   image, and a copy of DLSS's motion vectors into [`NativePass`]'s motion buffer;
//! - `N[parity]`, recorded once per build ([`NativePass::record`]): `native_preprocess.comp` (the
//!   network's 16 input lanes from the encoded proxy and the reprojected history), the network's
//!   recorded graph (a secondary from `neural-forge-native`), and `native_composite.comp` (the
//!   residual, the blend toward the history, the next history and the answer buffer);
//! - `W`, the write-back (`super::record_writeback`): the answer decoded over DLSS's colour input.
//!
//! The ordering is submission order on the game's queue plus each buffer's opening barrier
//! (`ALL_COMMANDS/MEMORY_WRITE` to what it reads), the same chain as the helper hold's
//! (`super`, "The dependency chain of a hold"), minus the host round trip.
//!
//! The network itself (model, kernels, the graph for one size) is loaded and built on a thread of its
//! own, on a queue the layer added to the device for it ([`Loader`]); until it is ready, frames go to
//! DLSS untouched.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use ash::vk;
use neural_forge_native as nn;
use neural_forge_protocol::history::{HistoryGap, Stale};
use neural_forge_protocol::rebuild::{BuildRetry, FailInject, Step};

use super::hdr::{self, OwnImage};
use super::{own_host_buffer_with, HostBuffer, Target};

const PREPROCESS_SPV: &[u8] = include_bytes!("../../shaders/native_preprocess.spv");
const COMPOSITE_SPV: &[u8] = include_bytes!("../../shaders/native_composite.spv");

/// `NEURAL_FORGE_BACKEND`: which runs the model before the upscaler (this branch only, for A/B runs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Native,
    Helper,
}

pub(crate) fn backend() -> Backend {
    static BACKEND: std::sync::LazyLock<Backend> = std::sync::LazyLock::new(|| match neural_forge_protocol::env::var("NEURAL_FORGE_BACKEND").as_deref() {
        Some("helper") => Backend::Helper,
        Some("native") | None | Some("") => Backend::Native,
        Some(other) => {
            crate::log!("[native] NEURAL_FORGE_BACKEND={other:?} is not native or helper; using native");
            Backend::Native
        }
    });
    *BACKEND
}

/// `NEURAL_FORGE_NATIVE_CHAIN=0`: barriers between the network's launches instead of counter chaining.
fn chain_wanted() -> bool {
    neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_CHAIN").is_none_or(|v| v != "0")
}

/// `extract-model`'s output: `$XDG_DATA_HOME/neural-forge/model`.
pub(crate) fn model_dir() -> PathBuf {
    crate::dump::captures_dir().with_file_name("model")
}

/// Whether the model directory is there at all (the hash check is the loader's).
pub(crate) fn model_present() -> bool {
    model_dir().join("manifest.json").is_file()
}

/// Where the network runs on a device: what `create_device` added for it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Setup {
    pub gipa: vk::PFN_vkGetInstanceProcAddr,
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    /// The game's graphics family, where DLSS runs and the frame work goes.
    pub frame_family: u32,
    /// The layer's own queue for the network (a compute family without graphics): the loader's
    /// uploads and warm-up only.
    pub family: u32,
    pub index: u32,
}

/// The network for one frame size, as the hold uses it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Built {
    pub width: u32,
    pub height: u32,
    pub frame: nn::Frame,
    pub graph: vk::CommandBuffer,
    /// Changes with every build: [`NativePass`] re-records its buffers when it does.
    pub generation: u64,
}

#[derive(Default)]
struct LoaderState {
    /// `None` while the worker has it out (opening, building), before it is opened, and after a
    /// re-initialisation closed it.
    network: Option<nn::Network>,
    /// Why the network is not usable right now (the Status tab shows it); cleared by a success.
    failed: Option<String>,
    built: Option<Built>,
    /// The size the holds want; the worker builds it.
    wanted: Option<(u32, u32)>,
    fall_back: bool,
    quit: bool,
    generation: u64,
    /// Opened at least once (a later `None` network means closed for a re-initialisation).
    opened: bool,
}

/// The network on one device, opened and built off the game's threads. Nothing is loaded until the
/// first hold asks for it ([`Self::ready`]): only the device DLSS runs on pays for the model. A
/// failed open or build is tried again on the helper's schedule (`neural_forge_protocol::rebuild`):
/// 0.5, 1 and 2 s, then a re-initialisation (the network closed and opened again), then every 30 s.
pub(crate) struct Loader {
    shared: Arc<(Mutex<LoaderState>, Condvar)>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    pending: Mutex<Option<(vk::Device, Setup)>>,
}

unsafe extern "C" fn init_dispatchable(_user: *mut std::ffi::c_void, device: vk::Device, object: *mut std::ffi::c_void) {
    // SAFETY: a dispatchable object of `device` the network just got from below the loader.
    let _ = unsafe { crate::loader_data::initialize_object(device, object) };
}

impl Loader {
    pub(crate) fn start(device: vk::Device, setup: Setup) -> Self {
        Self { shared: Arc::default(), worker: Mutex::new(None), pending: Mutex::new(Some((device, setup))) }
    }

    /// Starts the loading thread the first time.
    fn spawn(&self) {
        let Some((device, setup)) = self.pending.lock().unwrap().take() else { return };
        let shared = self.shared.clone();
        let worker = std::thread::Builder::new().name("nf-native-load".into()).spawn(move || Self::work(&shared, device, setup)).ok();
        if worker.is_none() {
            crate::log!("[native] could not start the loader thread; frames go to DLSS untouched");
            self.shared.0.lock().unwrap().failed = Some("the loader thread could not be started".into());
        }
        *self.worker.lock().unwrap() = worker;
    }

    fn open(device: vk::Device, setup: Setup) -> Result<nn::Network, String> {
        let dir = model_dir();
        if !model_present() {
            return Err(format!("no model at {}: extract it in the Setup tab (neural-forge-cli extract-model)", dir.display()));
        }
        let open = nn::OpenInfo {
            gipa: setup.gipa,
            instance: setup.instance,
            physical: setup.physical,
            device,
            queue_family: setup.family,
            queue_index: setup.index,
            frame_family: setup.frame_family,
            model_dir: &dir,
            chain: chain_wanted(),
            fence_timeout_ms: crate::FENCE_WAIT_TIMEOUT.as_millis() as u32,
            init_dispatchable: Some(init_dispatchable),
        };
        let t = Instant::now();
        // SAFETY: the device was created with the network's additions, and the queue is the loader's
        // alone (`crate::take_native`).
        let network = unsafe { nn::Network::open(&open) }.map_err(|why| format!("the model at {} could not be loaded: {why}", dir.display()))?;
        crate::log!("[native] model loaded and verified from {} in {} ms (chaining {})", dir.display(), t.elapsed().as_millis(), if chain_wanted() { "on" } else { "off" });
        Ok(network)
    }

    fn work(shared: &(Mutex<LoaderState>, Condvar), device: vk::Device, setup: Setup) {
        let (lock, wake) = shared;
        let mut retry = BuildRetry::default();
        let mut inject = neural_forge_protocol::env::var(FailInject::ENV).and_then(|v| FailInject::parse(&v));
        if let Some(f) = inject {
            crate::log!("[native] {}: the next {} network builds will fail on purpose", FailInject::ENV, f.remaining());
        }
        let mut state = lock.lock().unwrap();
        loop {
            // Work is wanted when the network is not open yet, a size is wanted that is not built, or
            // the graph is to fall back to barriers; it is due when the retry schedule says so.
            let wanted = |s: &LoaderState| !s.opened || s.fall_back || (s.wanted.is_some() && s.wanted != s.built.map(|b| (b.width, b.height)));
            if state.quit {
                return;
            }
            if !wanted(&state) {
                state = wake.wait(state).unwrap();
                continue;
            }
            let step = retry.step(Instant::now());
            if let Step::Wait(d) = step {
                state = wake.wait_timeout(state, d).unwrap().0;
                continue;
            }
            let mut network = state.network.take();
            let fall_back = std::mem::take(&mut state.fall_back);
            let size = state.wanted.or(state.built.map(|b| (b.width, b.height)));
            state.built = None;
            drop(state);
            if step == Step::ReinitThenBuild && network.take().is_some() {
                crate::log!("[native] {} failed attempts in a row: closing the network and loading it again", retry.streak());
            }
            let t = Instant::now();
            let result = match network.take() {
                Some(n) => Ok(n),
                None => Self::open(device, setup),
            }
            .and_then(|mut n| match size {
                None => Ok((n, None)),
                Some(_) if inject.as_mut().is_some_and(FailInject::should_fail) => {
                    network = Some(n);
                    Err(format!("{} asked this build to fail", FailInject::ENV))
                }
                Some((w, h)) => {
                    let built = if fall_back { n.fall_back_to_barriers() } else { n.build(w, h) };
                    match built {
                        Ok(frame) => Ok((n, Some((w, h, frame)))),
                        Err(why) => {
                            network = Some(n);
                            Err(format!("building the network for {w}x{h} failed: {why}"))
                        }
                    }
                }
            });
            state = lock.lock().unwrap();
            state.opened = true;
            match result {
                Ok((n, built)) => {
                    if let Some((w, h, frame)) = built {
                        state.generation += 1;
                        state.built = Some(Built { width: w, height: h, frame, graph: n.graph_commands(), generation: state.generation });
                        crate::log!(
                            "[native] network {}built for {w}x{h} (field {}x{}, chained {}) in {} ms",
                            if fall_back { "re" } else { "" },
                            frame.field_width,
                            frame.field_height,
                            frame.chained != 0,
                            t.elapsed().as_millis()
                        );
                        retry.succeeded();
                        if let Some(streak) = retry.take_recovered() {
                            crate::log!("[native] the network is up again after {streak} failed attempt(s)");
                        }
                        state.failed = None;
                    }
                    state.network = Some(n);
                }
                Err(why) => {
                    let wait = retry.failed(Instant::now());
                    crate::log!("[native] {why}; frames go to DLSS untouched, trying again in {} ms", wait.as_millis());
                    state.failed = Some(why);
                    state.network = network;
                    // A failed build leaves no graph; a fallback that failed is asked for again.
                    state.fall_back |= fall_back;
                }
            }
            crate::logging::flush();
        }
    }

    /// The network built for `width` x `height`, or `None` (and the build is asked for) while it is not.
    pub(crate) fn ready(&self, width: u32, height: u32) -> Option<Built> {
        self.spawn();
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if let Some(b) = state.built.filter(|b| (b.width, b.height) == (width, height)) {
            return Some(b);
        }
        if state.wanted != Some((width, height)) {
            state.wanted = Some((width, height));
            wake.notify_all();
        }
        None
    }

    /// Why the network is not usable right now, if a load or build failed.
    pub(crate) fn failed(&self) -> Option<String> {
        self.shared.0.lock().unwrap().failed.clone()
    }

    /// Counter waits that gave up since the last reset (read once the frames that ran the graph
    /// completed), and where.
    pub(crate) fn chain_timeouts(&self) -> Option<(u32, String)> {
        let state = self.shared.0.lock().unwrap();
        state.network.as_ref().map(nn::Network::chain_timeouts)
    }

    /// Rebuilds the graph with barriers. The caller has drained every frame that used the current one.
    pub(crate) fn fall_back(&self) {
        let (lock, wake) = &*self.shared;
        let mut state = lock.lock().unwrap();
        state.fall_back = true;
        state.built = None;
        wake.notify_all();
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        {
            let (lock, wake) = &*self.shared;
            lock.lock().unwrap().quit = true;
            wake.notify_all();
        }
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
        // The network (closed here) must not be pending: the device's teardown drained the layer's work.
        self.shared.0.lock().unwrap().network.take();
    }
}

/// The frame's parameters, `native_preprocess.comp`/`native_composite.comp`'s `Params` (std140, field
/// for field, append only).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Params {
    dims: [u32; 4],
    frame: [u32; 4],
    motion: [f32; 4],
    control: [f32; 4],
    more: [f32; 4],
}

/// The network's conditioning (the Model tab's settings that reach its input lanes).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conditioning {
    pub style: u32,
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// Below 0: follows `local_structure`.
    pub skin_structure: f32,
    pub auto_mask: bool,
}

impl From<neural_forge_protocol::PassTuning> for Conditioning {
    /// With the helper's clamps (`ngx.rs::NgxTuning`): the same values reach the network.
    fn from(t: neural_forge_protocol::PassTuning) -> Self {
        Self {
            style: t.style,
            intensity: t.intensity.clamp(0.0, 4.0),
            local_tone: t.local_tone.clamp(0.0, 4.0),
            local_structure: t.local_structure.clamp(0.0, 4.0),
            skin_structure: t.skin_structure.clamp(-1.0, 4.0),
            auto_mask: t.auto_mask != 0,
        }
    }
}

impl Default for Conditioning {
    fn default() -> Self {
        Self { style: 0, intensity: 1.0, local_tone: 1.0, local_structure: 1.0, skin_structure: -1.0, auto_mask: true }
    }
}

/// A device-local buffer of the layer's own.
pub(crate) struct OwnBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl OwnBuffer {
    /// # Safety
    /// `device` is live.
    unsafe fn new(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, bytes: u64, usage: vk::BufferUsageFlags) -> Option<Self> {
        let info = vk::BufferCreateInfo::builder().size(bytes).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let buffer = unsafe { device.create_buffer(&info, None) }.ok()?;
        // SAFETY: just created.
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        // SAFETY: plain query.
        let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let index = (0..props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        let memory = index.and_then(|index| {
            // SAFETY: valid allocation info.
            unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None) }.ok()
        });
        // SAFETY: sized for each other; on failure nothing was submitted.
        match memory.filter(|&m| unsafe { device.bind_buffer_memory(buffer, m, 0) }.is_ok()) {
            Some(memory) => Some(Self { buffer, memory }),
            None => {
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
        // SAFETY: forwarded.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

const PARAMS_BYTES: u64 = std::mem::size_of::<Params>() as u64;

/// The layer's side of the native frame for one extent: the two frame shaders, the history images,
/// DLSS's motion vectors' copy, the answer buffer, and the two recorded `N` buffers.
pub(crate) struct NativePass {
    width: u32,
    height: u32,
    padded_width: u32,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    preprocess: vk::Pipeline,
    composite: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: [vk::DescriptorSet; 2],
    /// `history[p]` is read by parity `p`'s frame and `history[1 - p]` written.
    history: [OwnImage; 2],
    pub(crate) mvec: OwnBuffer,
    answer: OwnBuffer,
    state: OwnBuffer,
    params: [HostBuffer; 2],
    command_pool: vk::CommandPool,
    /// `N[parity]`, then the one-time initialisation (history layouts, state cleared).
    commands: [vk::CommandBuffer; 3],
    /// GPU timestamps around `N[parity]` (`None` where timestamps are not supported).
    timers: [Option<crate::gpu_timer::GpuTimer>; 2],
    /// The parity of the last frame submitted, whose timer is read at the next hold.
    last_parity: Option<usize>,
    /// The [`Built::generation`] `commands` were recorded for, and whether the initialisation ran.
    recorded: Option<u64>,
    initialised: bool,
    /// The frame counter of the holds that ran the network, its seed since the last history reset,
    /// and the last frame's jitter.
    frames: u64,
    seed: u32,
    last_jitter: Option<[f32; 2]>,
    pub(crate) gap: HistoryGap<(u32, u32, u64)>,
}

// SAFETY: plain handles and the layer's own mappings, only used behind the device's `State` mutex.
unsafe impl Send for NativePass {}

impl NativePass {
    /// # Safety
    /// `device` is live; `family` is the queue family the frames are submitted on.
    pub(crate) unsafe fn build(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, family: u32, width: u32, height: u32) -> Option<Self> {
        let (padded_width, padded_height) = super::padded(width, height);
        let binding = |n: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build()
        };
        let bindings = [
            binding(0, vk::DescriptorType::STORAGE_IMAGE),
            binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
            binding(3, vk::DescriptorType::STORAGE_BUFFER),
            binding(4, vk::DescriptorType::UNIFORM_BUFFER),
            binding(5, vk::DescriptorType::STORAGE_BUFFER),
            binding(6, vk::DescriptorType::STORAGE_BUFFER),
            binding(7, vk::DescriptorType::STORAGE_IMAGE),
            binding(8, vk::DescriptorType::STORAGE_BUFFER),
            binding(9, vk::DescriptorType::STORAGE_BUFFER),
        ];
        // SAFETY: valid create infos throughout; on a failure part way, the handles made so far leak
        // (a few small objects, once per extent at most), never anything submitted.
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
            let pipeline_layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&set_layout)), None).ok()?;
            let preprocess = hdr::compute_pipeline(device, pipeline_layout, PREPROCESS_SPV)?;
            let composite = hdr::compute_pipeline(device, pipeline_layout, COMPOSITE_SPV)?;
            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .ok()?;
            let sizes = [
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 4 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 12 },
                vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 },
            ];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&sizes), None).ok()?;
            let layouts = [set_layout, set_layout];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts)).ok()?;
            let image = |usage| OwnImage::new(device, instance, physical_device, width, height, usage);
            let history = [image(vk::ImageUsageFlags::SAMPLED)?, image(vk::ImageUsageFlags::SAMPLED)?];
            let texels = u64::from(width) * u64::from(height);
            let mvec = OwnBuffer::new(device, instance, physical_device, texels * 4, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let answer_bytes = u64::from(padded_width) * u64::from(padded_height) * 8;
            let answer = OwnBuffer::new(device, instance, physical_device, answer_bytes, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let state = OwnBuffer::new(device, instance, physical_device, 16, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let params = [
                own_host_buffer_with(device, instance, physical_device, PARAMS_BYTES, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
                own_host_buffer_with(device, instance, physical_device, PARAMS_BYTES, vk::BufferUsageFlags::UNIFORM_BUFFER)?,
            ];
            let command_pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .ok()?;
            let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(3);
            let cmds = crate::loader_data::allocate_commands(device, &alloc).ok()?;
            Some(Self {
                width,
                height,
                padded_width,
                set_layout,
                pipeline_layout,
                preprocess,
                composite,
                sampler,
                pool,
                sets: [sets[0], sets[1]],
                history,
                mvec,
                answer,
                state,
                params,
                command_pool,
                commands: [cmds[0], cmds[1], cmds[2]],
                timers: [
                    crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family),
                    crate::gpu_timer::GpuTimer::new(device, instance, physical_device, family),
                ],
                last_parity: None,
                recorded: None,
                initialised: false,
                frames: 0,
                seed: 0,
                last_jitter: None,
                gap: HistoryGap::default(),
            })
        }
    }

    pub(crate) fn matches(&self, width: u32, height: u32) -> bool {
        (self.width, self.height) == (width, height)
    }

    pub(crate) fn answer(&self) -> vk::Buffer {
        self.answer.buffer
    }

    /// Points both sets at this frame's resources and records `N[0]`, `N[1]` and the initialisation
    /// for `built`, unless they already are. Nothing of them may be pending.
    ///
    /// # Safety
    /// `device` is live; `encoded` and `exposure` are the HDR pass's (live for as long as this pass).
    pub(crate) unsafe fn record(&mut self, device: &ash::Device, built: &Built, encoded: vk::ImageView, exposure: vk::Buffer) -> bool {
        if self.recorded == Some(built.generation) {
            return true;
        }
        let f = built.frame;
        for p in 0..2 {
            let set = self.sets[p];
            let general = |view| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }];
            let enc = general(encoded);
            let next = general(self.history[1 - p].view);
            let prev = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: self.history[p].view, image_layout: vk::ImageLayout::GENERAL }];
            let buf = |buffer, range| [vk::DescriptorBufferInfo { buffer, offset: 0, range }];
            let (mv, feat, head, params, state, expo, ans) = (
                buf(self.mvec.buffer, vk::WHOLE_SIZE),
                buf(f.features, vk::WHOLE_SIZE),
                buf(f.head, vk::WHOLE_SIZE),
                buf(self.params[p].buffer, PARAMS_BYTES),
                buf(self.state.buffer, vk::WHOLE_SIZE),
                buf(exposure, 4),
                buf(self.answer.buffer, vk::WHOLE_SIZE),
            );
            let w = |binding, ty| vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(binding).descriptor_type(ty);
            let writes = [
                w(0, vk::DescriptorType::STORAGE_IMAGE).image_info(&enc).build(),
                w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&prev).build(),
                w(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&mv).build(),
                w(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&feat).build(),
                w(4, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&params).build(),
                w(5, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&state).build(),
                w(6, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&expo).build(),
                w(7, vk::DescriptorType::STORAGE_IMAGE).image_info(&next).build(),
                w(8, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ans).build(),
                w(9, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&head).build(),
            ];
            // SAFETY: the sets are not in use (nothing of this pass is pending); every handle is live.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let ok = (0..2).all(|p| {
            // SAFETY: forwarded from this function's contract.
            unsafe { self.record_frame(device, p, built) }.is_ok()
        });
        // SAFETY: as above.
        let ok = ok && unsafe { self.record_init(device) }.is_ok();
        self.recorded = ok.then_some(built.generation);
        ok
    }

    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_frame(&self, device: &ash::Device, p: usize, built: &Built) -> ash::prelude::VkResult<()> {
        let cmd = self.commands[p];
        let f = built.frame;
        let compute = |src: vk::AccessFlags, dst: vk::AccessFlags| vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(dst).build();
        // SAFETY: the layer's own buffer, not pending (forwarded); every handle is live.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE))?;
            if let Some(timer) = &self.timers[p] {
                timer.record_start(device, cmd);
            }
            // `C`'s encode and motion-vector copy, and the previous frame's history write, before
            // anything here reads them.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::UNIFORM_READ)],
                &[],
                &[],
            );
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[p]], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.preprocess);
            device.cmd_dispatch(cmd, f.field_width.div_ceil(8), f.field_height.div_ceil(8), 1);
            // The features to the network (the graph opens with compute barriers of its own).
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            device.cmd_execute_commands(cmd, &[built.graph]);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            // The graph's secondary may have bound its own pipeline layout and sets: bind again.
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[self.sets[p]], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.composite);
            device.cmd_dispatch(cmd, self.width.div_ceil(8), self.height.div_ceil(8), 1);
            // The answer to `W`'s copy, the history to the next frame's samplers.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[compute(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
                &[],
                &[],
            );
            if let Some(timer) = &self.timers[p] {
                timer.record_end(device, cmd);
            }
            device.end_command_buffer(cmd)
        }
    }

    /// The previous frame's network GPU time, once its work has completed (the caller waited for it).
    pub(crate) fn take_gpu_ms(&mut self, device: &ash::Device) -> Option<f32> {
        let timer = self.timers[self.last_parity.take()?].as_mut()?;
        timer.read_if_pending(device);
        timer.take_reading()
    }

    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_init(&self, device: &ash::Device) -> ash::prelude::VkResult<()> {
        let cmd = self.commands[2];
        let to_general = |image| {
            vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                .build()
        };
        // SAFETY: as `record_frame`.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;
            device.cmd_fill_buffer(cmd, self.state.buffer, 0, vk::WHOLE_SIZE, 0);
            let filled = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE).build();
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[filled],
                &[],
                &[to_general(self.history[0].image), to_general(self.history[1].image)],
            );
            device.end_command_buffer(cmd)
        }
    }

    /// The command buffers for this frame, in submission order: the initialisation the first time,
    /// then `N[parity]`. Writes the frame's parameters.
    pub(crate) fn frame(&mut self, built: &Built, history: bool, jitter: Option<[f32; 2]>, c: Conditioning) -> Vec<vk::CommandBuffer> {
        let p = (self.frames & 1) as usize;
        self.seed = if history { self.seed.wrapping_add(1) } else { 0 };
        let delta = match (self.last_jitter, jitter) {
            (Some(prev), Some(cur)) if history => [prev[0] - cur[0], prev[1] - cur[1]],
            _ => [0.0, 0.0],
        };
        self.last_jitter = jitter;
        let f = built.frame;
        let params = Params {
            dims: [f.field_width, f.field_height, self.width, self.height],
            frame: [self.padded_width, self.seed, u32::from(history), p as u32],
            motion: [delta[0], delta[1], f.blend_scale, c.intensity],
            control: [c.style as f32, c.local_tone, c.local_structure, c.skin_structure],
            more: [if c.auto_mask { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        // SAFETY: host-coherent and mapped for PARAMS_BYTES; parity `p`'s previous use (two frames
        // ago) completed: the hold waits for the previous hold's work before it records anything.
        unsafe { std::ptr::copy_nonoverlapping(std::ptr::from_ref(&params).cast::<u8>(), self.params[p].ptr, PARAMS_BYTES as usize) };
        self.frames += 1;
        self.last_parity = Some(p);
        if let Some(timer) = self.timers[p].as_mut() {
            timer.mark_submitted();
        }
        let mut out = Vec::with_capacity(2);
        if !std::mem::replace(&mut self.initialised, true) {
            out.push(self.commands[2]);
        }
        out.push(self.commands[p]);
        out
    }

    /// # Safety
    /// Nothing of this pass may still be pending.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded; freeing the pools frees their sets and buffers.
        unsafe {
            for t in self.timers.iter().flatten() {
                t.destroy(device);
            }
            device.destroy_command_pool(self.command_pool, None);
            for h in &self.history {
                h.destroy(device);
            }
            self.mvec.destroy(device);
            self.answer.destroy(device);
            self.state.destroy(device);
            for b in &self.params {
                b.destroy(device);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.preprocess, None);
            device.destroy_pipeline(self.composite, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// `NEURAL_FORGE_NATIVE_FAIL=chain`: reports one counter-chain timeout at the 300th network frame, to
/// exercise the fallback to barriers on the rig. Unset (the default) does nothing.
fn fake_chain_timeout(frames: u64) -> Option<(u32, String)> {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_NATIVE_FAIL").as_deref() == Some("chain"));
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    (*ON && frames >= 300 && !DONE.swap(true, std::sync::atomic::Ordering::Relaxed)).then(|| (1, "NEURAL_FORGE_NATIVE_FAIL=chain".to_string()))
}

/// What the Status tab says about the native backend after a hold: running, or why not.
pub(crate) fn status(result: &super::HoldResult, loader: &Loader) -> String {
    if result.evaluated {
        return "native network running".to_string();
    }
    match (loader.failed(), result.miss) {
        (Some(why), _) => format!("native network not running: {why}"),
        (None, Some(miss)) => format!("native network: {miss}"),
        (None, None) => String::new(),
    }
}

/// Logs a history reset's reason (the helper logs the same reasons for NGX's).
pub(crate) fn note_reset(stale: Option<Stale>) {
    match stale {
        Some(Stale::Skipped) => crate::log!("[native] resetting the history (frames went to DLSS untouched since the last one)"),
        Some(Stale::Idle(gap)) => crate::log!("[native] resetting the history ({} ms since the last frame)", gap.as_millis()),
        Some(Stale::FormatChanged) => crate::log!("[native] resetting the history (another extent or identification)"),
        None => {}
    }
}

/// Every hold's view of the target and its time, for [`HistoryGap`].
pub(crate) fn format_of(target: &Target) -> (u32, u32, u64) {
    (target.width, target.height, target.identification)
}


/// One native hold (module docs): the capture with the motion-vector copy, the network, the write-back,
/// submitted on the game's queue through `submit` with no wait in between. The previous hold's work
/// is waited for (bounded) first, as every hold does. Frames the network is not ready for go to DLSS
/// untouched, with `miss` saying why.
///
/// # Safety
/// As `super::run_hold`; `loader` belongs to `device`, and `res` was built for the queue family the
/// loader's network was opened for.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_native_hold(
    device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, res: &mut super::Resources, target: &Target,
    loader: &Loader, jitter: Option<[f32; 2]>, conditioning: Conditioning,
    submit: &mut dyn FnMut(super::Which, vk::CommandBuffer, vk::Fence) -> ash::prelude::VkResult<()>,
) -> super::HoldResult {
    let mut result = super::HoldResult { native: true, ..Default::default() };
    let t_hold = Instant::now();
    let mut writeback_gpu = None;
    if !res.wait_idle(device, &mut writeback_gpu) {
        result.miss = Some("the previous hold's work is still pending");
        return result;
    }
    result.writeback_gpu_ms = writeback_gpu;
    result.network_gpu_ms = res.native.as_mut().and_then(|n| n.take_gpu_ms(device));
    // Every frame that ran the graph has completed: the chain watchdog's count is final for them.
    let fake = fake_chain_timeout(res.native.as_ref().map_or(0, |n| n.frames));
    if let Some((waits, at)) = loader.chain_timeouts().filter(|(n, _)| *n > 0).or(fake) {
        crate::log!("[native] {waits} counter-chain wait(s) of the network gave up ({at}); rebuilding it with barriers between its launches");
        crate::logging::flush();
        loader.fall_back();
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some("the network's counter chain timed out; it is being rebuilt with barriers");
        return result;
    }
    let Some(built) = loader.ready(target.width, target.height) else {
        if let Some(n) = res.native.as_mut() {
            n.gap.skipped();
        }
        result.miss = Some(if loader.failed().is_some() { "the native network could not be loaded or built" } else { "the native network is still loading" });
        return result;
    };
    // The HDR encode and decode, exactly as the helper hold uses them.
    let source = super::ExposureSource::of(target);
    result.exposure_source = Some(source);
    if res.hdr.is_none() {
        // SAFETY: `device` is live.
        res.hdr = unsafe { hdr::HdrPass::build(device, instance, physical_device, res.width, res.height) };
    }
    let Some(pass) = res.hdr.as_mut() else {
        result.miss = Some("the HDR encode/decode pipelines could not be built");
        return result;
    };
    if source == super::ExposureSource::Auto {
        // SAFETY: `device` is live; nothing of the pass is pending.
        if !unsafe { pass.ensure_auto(device, instance, physical_device) } {
            result.miss = Some("the auto-exposure pipeline could not be built (no exposure image to read)");
            return result;
        }
        pass.auto_for(target.identification);
    }
    pass.clear_exposure();
    // SAFETY: nothing of the pass is pending; the colour input is a live RGBA16F storage image.
    if !unsafe { pass.bind_colour(device, target.colour) } {
        result.miss = Some("a storage view of the colour input could not be created");
        return result;
    }
    let (encoded, exposure) = (pass.encoded_view(), pass.exposure_buffer());
    // The native pass for this extent, its buffers recorded for this build.
    let mut native = match res.native.take().filter(|n| n.matches(target.width, target.height)) {
        Some(n) => n,
        None => {
            // SAFETY: `device` is live; `res.queue_family` is the frames' family.
            match unsafe { NativePass::build(device, instance, physical_device, res.queue_family, target.width, target.height) } {
                Some(n) => n,
                None => {
                    result.miss = Some("the native frame resources could not be built");
                    return result;
                }
            }
        }
    };
    // SAFETY: nothing of the pass is pending (`wait_idle` above); the HDR pass outlives it in `res`.
    if !unsafe { native.record(device, &built, encoded, exposure) } {
        res.native = Some(native);
        result.miss = Some("recording the native frame failed");
        return result;
    }
    // The history needs DLSS's motion vectors copied this frame.
    let mvec = target.mvec.filter(|a| a.readable && a.layout.is_some());
    let hdr_pass = res.hdr.as_ref();
    // SAFETY: the capture buffer is idle; the colour view was bound above.
    if unsafe { super::record_capture(device, res, target, None, hdr_pass, mvec.map(|_| native.mvec.buffer)) }.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the capture failed");
        return result;
    }
    // SAFETY: as above, with the native pass's answer as the decode's source.
    if unsafe { super::record_writeback(device, res, target, native.answer(), hdr_pass) }.is_err() {
        res.native = Some(native);
        result.miss = Some("recording the write-back failed");
        return result;
    }
    native.gap.note_format(format_of(target));
    let stale = native.gap.begin(Instant::now());
    if native.frames > 0 {
        note_reset(stale);
    }
    let history = native.frames > 0 && stale.is_none() && mvec.is_some();
    let frame_cmds = native.frame(&built, history, jitter, conditioning);
    res.native = Some(native);
    result.timing.prep = Some(t_hold.elapsed());
    if crate::note_vk(submit(super::Which::Capture, res.capture_cmd, vk::Fence::null())).is_err() {
        result.miss = Some("the capture submit failed");
        return result;
    }
    result.waits_consumed = true;
    for cmd in frame_cmds {
        if crate::note_vk(submit(super::Which::Native, cmd, vk::Fence::null())).is_err() {
            // The capture ran; nothing reads its result. The decode is not submitted, so DLSS gets
            // the frame as it was.
            if let Some(n) = res.native.as_mut() {
                n.gap.skipped();
            }
            result.miss = Some("the network's submit failed");
            return result;
        }
    }
    // SAFETY: the layer's own fence, idle (`wait_idle` above, or never submitted).
    if unsafe { device.reset_fences(&[res.writeback_fence]) }.is_err() {
        result.miss = Some("resetting the write-back fence failed");
        return result;
    }
    if crate::note_vk(submit(super::Which::WriteBack, res.writeback_cmd, res.writeback_fence)).is_err() {
        result.miss = Some("the write-back submit failed");
        return result;
    }
    // The write-back's fence also covers the capture and the network, submitted before it on the
    // same queue.
    res.writeback_pending = true;
    if let Some(timer) = res.writeback_timer.as_mut() {
        timer.mark_submitted();
    }
    result.evaluated = true;
    result.wrote_back = true;
    result.timing.writeback = Some(t_hold.elapsed());
    result
}
