//! The native backend after the upscaler (docs/NATIVE_BACKEND.md, "2.1"): the requests the after-the-upscaler
//! path writes into shared memory for the model server are answered here, in the game's process, by the native
//! network, so the capture and the composition stay exactly as they are.
//!
//! [`PostServer`] is the model server, the after-the-upscaler path's request loop:
//! per slot, a new `seq_req` is read with its size and format; an 8-bit frame the model is wanted for runs
//! through `native_post_preprocess.comp`, the network (its recording for the layer's own compute family) and
//! `native_post_composite.comp`, on a compute queue of the layer's own, waited for (bounded); anything else is
//! echoed (failing open). Then `answered_w/h`, `seq_eval`, `server_busy_us` and `seq_resp` are published in that
//! order, and the heartbeat keeps the layer's `server_alive` true.
//!
//! With more than one pass (the Passes setting) the frame runs through the network again for each, every pass
//! after the first reading the previous pass's answer (copied into the slot's `scratch`) where the first reads
//! the frame; each pass has its own settings (the Model tab's with that pass's overrides).
//!
//! There are no motion vectors after the upscaler. By default (`NEURAL_FORGE_POST_MOTION=none`) every frame
//! runs without history, with a fixed seed (a seed that changed every frame, with nothing blending frames,
//! would shimmer). `zero` and `flow` give each pass a history of its own ([`super::post_history`],
//! [`super::post_flow`]): its previous output at the same pixel or where the GPU's optical flow says the pixel
//! was, blended toward as before the upscaler, rejected per pixel where the frame changed, reset as a whole
//! on a cut and the other reset rules, with the seed counting frames since the reset. It never runs the
//! network while the pre-upscaler path uses it, nor the other way round
//! ([`super::native::Loader::claim_post`]).
//!
//! The presented frame carries the game's HUD (docs/NATIVE_BACKEND.md, "The HUD at present"): the user's
//! regions (`hud_regions`) and the detection (`hud_detect`, on by default) give each pixel a scale of the edit
//! (`native_post_hud.comp`, [`Hud`]), which the last pass's composite applies and the preprocess's history
//! rejection honours.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;
use neural_forge_protocol::enums::proxy_format;
use neural_forge_protocol::Slot;

use super::native::{Built, Conditioning, Loader, OwnBuffer, Setup};
use super::post_flow;
use super::post_history::{self, Motion, SlotHistory};
use super::{region_buffer, HostBuffer};
use crate::shm::ShmView;

const PREPROCESS_SPV: &[u8] = include_bytes!("../../shaders/native_post_preprocess.spv");
const COMPOSITE_SPV: &[u8] = include_bytes!("../../shaders/native_post_composite.spv");
const FEEDBACK_SPV: &[u8] = include_bytes!("../../shaders/native_post_feedback.spv");
const HUD_SPV: &[u8] = include_bytes!("../../shaders/native_post_hud.spv");

/// The model server's smallest frame (`ngx::MIN_FEATURE_DIM`): smaller ones are echoed.
const MIN_DIM: u32 = 64;

/// The descriptor sets: each pass's preprocess and composite (2k, 2k + 1), then pass feedback's from the
/// frame and from the scratch.
const SETS: usize = 2 * neural_forge_protocol::MAX_PASSES + 2;

/// The set layout's bindings (the shaders': 0 input, 1 features or head, 2 answer, 3 the pass's history,
/// 4 the reprojected history, 5 the frame's proxy, 6 the previous frame's, 7 the optical flow, 8 the look-up
/// table, 9 the HUD's scale of the edit).
const BINDINGS: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PreprocessPush {
    dims: [u32; 4],
    frame: [u32; 4],
    control: [f32; 4],
    more: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CompositePush {
    dims: [u32; 4],
    frame: [u32; 4],
    control: [f32; 4],
    /// The colour of the edit ([`super::colour::PassColour`]; native_post_composite.comp's).
    colour: [f32; 4],
    colour_flags: [u32; 4],
    lut_scale: [f32; 4],
    lut_offset: [f32; 4],
    /// The HUD: x 1 applies its scale of the edit (the last pass).
    hud: [f32; 4],
}

impl CompositePush {
    fn with_colour(self, c: super::colour::PassColour) -> Self {
        Self { colour: c.colour, colour_flags: c.flags, lut_scale: c.lut_scale, lut_offset: c.lut_offset, ..self }
    }
}

/// Which server answers the after-the-upscaler requests in this process: the channel is the process's, so
/// two answerers (two devices' servers) would race for every request. Holds the owning device's handle, 0 for
/// none; the owner's worker clears it when it ends.
pub(crate) struct Owner(AtomicU64);

static OWNER: Owner = Owner::new();

impl Owner {
    pub(crate) const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Takes the channel for `token` (non-zero), unless another server holds it.
    pub(crate) fn claim(&self, token: u64) -> Option<Claim<'_>> {
        self.0.compare_exchange(0, token, Ordering::AcqRel, Ordering::Acquire).ok().map(|_| Claim(self))
    }
}

/// The channel held by one server, until dropped.
pub(crate) struct Claim<'a>(&'a Owner);

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0 .0.store(0, Ordering::Release);
    }
}

/// How a server's worker is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// Setting up.
    Starting = 0,
    /// Answering requests.
    Serving = 1,
    /// Ended for a reason that may pass (its pipelines could not be built, or another server holds the
    /// channel): another start is worth making later.
    Retry = 2,
    /// Ended because a frame did not finish in time: the device's work may still be running, so nothing
    /// starts on it again.
    Stalled = 3,
}

impl Status {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Serving,
            2 => Self::Retry,
            3 => Self::Stalled,
            _ => Self::Starting,
        }
    }
}

/// The after-the-upscaler requests' server on one device.
pub(crate) struct PostServer {
    quit: Arc<AtomicBool>,
    status: Arc<AtomicU8>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl PostServer {
    /// Starts the server's worker, which takes the channel ([`Owner`]), checks that nothing outside the
    /// process answers it (its heartbeat stands still), builds its pipelines and serves. Nothing here waits:
    /// this is called from the present. `None` only when the thread could not be spawned.
    pub(crate) fn start(device: ash::Device, instance: ash::Instance, physical: vk::PhysicalDevice, setup: Setup, import: bool, loader: Arc<Loader>, view: ShmView) -> Option<Self> {
        let quit = Arc::new(AtomicBool::new(false));
        let status = Arc::new(AtomicU8::new(Status::Starting as u8));
        let (q, st) = (quit.clone(), status.clone());
        let worker = std::thread::Builder::new()
            .name("nf-native-post".into())
            .spawn(move || {
                let end = |s: Status| st.store(s as u8, Ordering::Release);
                let Some(claim) = OWNER.claim(ash::vk::Handle::as_raw(device.handle()).max(1)) else {
                    crate::log!("[native] another device's server is answering the after-the-upscaler requests: it keeps them");
                    return end(Status::Retry);
                };
                let hdr = view.header();
                let before = hdr.server_heartbeat.load(Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(300));
                if hdr.server_heartbeat.load(Ordering::Relaxed) != before {
                    crate::log!("[native] something else is answering the after-the-upscaler requests: it keeps them");
                    drop(claim);
                    return end(Status::Retry);
                }
                // SAFETY: the device and the queue the setup names are live for this thread's life (joined in Drop,
                // before the device's teardown frees anything).
                let Some(mut gpu) = (unsafe { Gpu::new(&device, setup) }) else {
                    crate::log!("[native] the after-the-upscaler path's pipelines could not be built; it stays untouched for now");
                    drop(claim);
                    return end(Status::Retry);
                };
                crate::log!("[native] answering the after-the-upscaler path's requests on queue {} of family {}", setup.post_index, setup.family);
                crate::logging::flush();
                end(Status::Serving);
                serve(&device, &instance, physical, import, &loader, view, &q, &mut gpu);
                let stalled = gpu.stalled;
                // SAFETY: `serve` returns with nothing of `gpu` pending (or stalled, and then nothing is freed).
                unsafe { gpu.destroy(&device) };
                drop(claim);
                end(if stalled { Status::Stalled } else { Status::Retry });
            })
            .ok()?;
        Some(Self { quit, status, worker: Some(worker) })
    }

    pub(crate) fn status(&self) -> Status {
        Status::from_u8(self.status.load(Ordering::Acquire))
    }
}

impl Drop for PostServer {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// [`Gpu::lut`]: the table's entries in a host-visible buffer, and which table they are.
struct PostLut {
    buffer: HostBuffer,
    entries: usize,
    lut: Arc<super::colour::Lut>,
}

/// The per-request GPU objects.
struct Gpu {
    queue: vk::Queue,
    set_layout: vk::DescriptorSetLayout,
    layouts: [vk::PipelineLayout; 2],
    pipelines: [vk::Pipeline; 2],
    pool: vk::DescriptorPool,
    /// Each pass's preprocess and composite sets (2k, 2k + 1; the first pass's read the frame, the later ones
    /// `scratch`), then pass feedback's from the frame and from the scratch ([`SETS`]).
    sets: [vk::DescriptorSet; SETS],
    /// Pass feedback below 1 (`native_post_feedback.comp`): its pipeline layout and pipeline, built the first
    /// time it is.
    feedback: Option<(vk::PipelineLayout, vk::Pipeline)>,
    /// The colour of the edit's look-up table (binding 8 of the composite's sets), the last pass's.
    lut: Option<PostLut>,
    /// The HUD's pipeline and its one set (`native_post_hud.comp`), built the first time the HUD is wanted
    /// (`None` inside after a failed build: the frames run without it, logged once).
    hud: Option<Option<HudPipeline>>,
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Per slot: the proxy and answer buffers over the shared-memory regions, for one frame size.
    slots: [Option<SlotBuffers>; 2],
    /// What the sets point at.
    bound: Option<Bound>,
    /// A wait that timed out: the buffers it used may still be in use, so they are never touched again.
    stalled: bool,
    /// Where the history comes from (`NEURAL_FORGE_POST_MOTION`), and the per-pixel rejection's threshold.
    motion: Motion,
    reject: f32,
    /// Per slot: the history's reset rules and seeds.
    history: [SlotHistory; 2],
    /// The optical flow (`flow`), set up on its first frame.
    flow: FlowState,
    /// What `create_device` added (the optical-flow queue).
    setup: Setup,
    /// Tests: a flow field (grid, S10.5 vectors) the preprocess reads instead of the GPU's.
    #[cfg(test)]
    synthetic_flow: Option<(u32, Vec<u32>)>,
}

/// What [`Gpu::sets`] point at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bound {
    slot: usize,
    width: u32,
    height: u32,
    generation: u64,
    scratch: bool,
    passes: usize,
    /// How many passes' history buffers exist (0: none).
    history: usize,
    flow: bool,
    /// The look-up table's buffer (null: none).
    lut: vk::Buffer,
    /// The HUD's scale buffer (null: none).
    hud: vk::Buffer,
}

/// The optical flow's state on the server's device.
enum FlowState {
    /// Not wanted (`NEURAL_FORGE_POST_MOTION` is not `flow`).
    Off,
    /// Wanted, set up on the first frame that needs it.
    Pending,
    Ready(post_flow::FlowDevice),
    /// Not available, or given up (kept to destroy the sessions made with it): `zero` instead.
    Unavailable(Option<post_flow::FlowDevice>),
}

impl FlowState {
    fn unavailable(why: &str, fd: Option<post_flow::FlowDevice>) -> Self {
        note_flow_off(why);
        Self::Unavailable(fd)
    }

    fn device(&self) -> Option<&post_flow::FlowDevice> {
        match self {
            Self::Ready(fd) | Self::Unavailable(Some(fd)) => Some(fd),
            _ => None,
        }
    }

    /// A submit of the flow's failed: both queues (this thread's alone) drain, so nothing of the flow is
    /// pending, and `zero` runs from here on.
    fn give_up(&mut self, device: &ash::Device, compute: vk::Queue, why: &str) {
        if let Self::Ready(fd) = std::mem::replace(self, Self::Off) {
            // SAFETY: both queues are this thread's alone.
            unsafe {
                let _ = device.queue_wait_idle(compute);
                let _ = device.queue_wait_idle(fd.queue);
            }
            *self = Self::unavailable(why, Some(fd));
        }
    }
}

impl SlotBuffers {
    /// # Safety
    /// Nothing of it is pending; `flow` is the device its session was made with.
    unsafe fn destroy(self, device: &ash::Device, flow: Option<&post_flow::FlowDevice>) {
        // SAFETY: forwarded.
        unsafe {
            self.proxy.destroy(device);
            self.answer.destroy(device);
            if let Some(s) = self.scratch {
                s.destroy(device);
            }
            if let Some(h) = self.hud {
                h.destroy(device);
            }
            if let Some(h) = self.hist {
                h.previous.destroy(device);
                h.reproj.destroy(device);
                for p in h.passes {
                    p.destroy(device);
                }
                if let Some(f) = h.flow {
                    f.destroy(device);
                }
                if let (Some(session), Some(fd)) = (h.session, flow) {
                    session.destroy(device, fd);
                }
            }
        }
    }
}

struct SlotBuffers {
    width: u32,
    height: u32,
    proxy: HostBuffer,
    answer: HostBuffer,
    /// The previous pass's answer, for the next pass to read (built on the first frame with two or more).
    scratch: Option<OwnBuffer>,
    /// The history's (`zero` and `flow`), built on the first frame with one.
    hist: Option<HistBuffers>,
    /// The HUD's ([`Hud`]), built on the first frame that wants it.
    hud: Option<HudBuffers>,
}

/// One slot's history buffers, for its frame size.
struct HistBuffers {
    /// The previous frame's proxy (the per-pixel rejection's reference), copied at the end of each frame.
    previous: OwnBuffer,
    /// The preprocess's reprojected history for the composite (8 bytes a pixel).
    reproj: OwnBuffer,
    /// Each pass's history (8 bytes a pixel: RGB as halves), built as passes are.
    passes: Vec<OwnBuffer>,
    /// The optical flow's vectors as the preprocess reads them (host-visible), and its session (`flow`).
    flow: Option<HostBuffer>,
    session: Option<post_flow::Session>,
    /// The session could not be made at this size: zero motion (logged once).
    flow_failed: bool,
}

impl Gpu {
    /// # Safety
    /// `device` is live and has the queue `setup` names.
    unsafe fn new(device: &ash::Device, setup: Setup) -> Option<Self> {
        // Built piece by piece into `g`; on an early return its drop destroys whatever exists (the server is
        // started again after a failure, so a failed attempt must not leave anything behind).
        let mut g = Partial { device, gpu: Self::empty(setup) };
        let step = |n: u32| -> Option<()> {
            #[cfg(test)]
            if tests::FAIL_AT.get() == Some(n) {
                return None;
            }
            let _ = n;
            Some(())
        };
        // SAFETY: valid create infos throughout; every handle goes into `g` as soon as it exists.
        unsafe {
            step(0)?;
            g.gpu.queue = device.get_device_queue(setup.family, setup.post_index);
            crate::loader_data::initialize_queue(device.handle(), g.gpu.queue).ok()?;
            let binding = |n: u32| vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build();
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS).map(binding).collect();
            step(1)?;
            g.gpu.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
            let set_layout = g.gpu.set_layout;
            let layout = |bytes: usize| {
                let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).size(bytes as u32).build();
                device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&set_layout)).push_constant_ranges(std::slice::from_ref(&push)), None).ok()
            };
            step(2)?;
            g.gpu.layouts[0] = layout(std::mem::size_of::<PreprocessPush>())?;
            step(3)?;
            g.gpu.layouts[1] = layout(std::mem::size_of::<CompositePush>())?;
            step(4)?;
            g.gpu.pipelines[0] = super::hdr::compute_pipeline(device, g.gpu.layouts[0], PREPROCESS_SPV)?;
            step(5)?;
            g.gpu.pipelines[1] = super::hdr::compute_pipeline(device, g.gpu.layouts[1], COMPOSITE_SPV)?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: SETS as u32 * BINDINGS }];
            step(6)?;
            g.gpu.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(SETS as u32).pool_sizes(&sizes), None).ok()?;
            let set_layouts = [set_layout; SETS];
            step(7)?;
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(g.gpu.pool).set_layouts(&set_layouts)).ok()?;
            g.gpu.sets.copy_from_slice(&sets);
            step(8)?;
            g.gpu.command_pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(setup.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .ok()?;
            step(9)?;
            g.gpu.cmd = crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::builder().command_pool(g.gpu.command_pool).command_buffer_count(1)).ok()?[0];
            step(10)?;
            g.gpu.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
        }
        let motion = post_history::motion();
        g.gpu.motion = motion;
        g.gpu.reject = post_history::reject();
        g.gpu.flow = if motion == Motion::Flow { FlowState::Pending } else { FlowState::Off };
        Some(std::mem::replace(&mut g.gpu, Self::empty(setup)))
    }

    /// The objects it owns that are destroyed one by one (the sets and the command buffer go with their pools).
    #[cfg(test)]
    fn live(&self) -> usize {
        use ash::vk::Handle;
        let handles = [self.set_layout.as_raw(), self.layouts[0].as_raw(), self.layouts[1].as_raw(), self.pipelines[0].as_raw(), self.pipelines[1].as_raw(), self.pool.as_raw(), self.command_pool.as_raw(), self.fence.as_raw()];
        handles.iter().filter(|&&h| h != 0).count()
    }

    /// No objects (every handle null).
    fn empty(setup: Setup) -> Self {
        Self {
            queue: vk::Queue::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layouts: [vk::PipelineLayout::null(); 2],
            pipelines: [vk::Pipeline::null(); 2],
            pool: vk::DescriptorPool::null(),
            sets: [vk::DescriptorSet::null(); SETS],
            feedback: None,
            lut: None,
            hud: None,
            command_pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            slots: [None, None],
            bound: None,
            stalled: false,
            motion: Motion::None,
            reject: post_history::REJECT,
            history: [SlotHistory::default(), SlotHistory::default()],
            flow: FlowState::Off,
            setup,
            #[cfg(test)]
            synthetic_flow: None,
        }
    }

    /// # Safety
    /// Nothing of it is pending.
    unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.stalled {
            return;
        }
        // SAFETY: forwarded.
        unsafe {
            let flow = std::mem::replace(&mut self.flow, FlowState::Off);
            for s in self.slots.iter_mut().filter_map(Option::take) {
                s.destroy(device, flow.device());
            }
            if let Some(fd) = flow.device() {
                fd.destroy(device);
            }
            if let Some((layout, pipeline)) = self.feedback.take() {
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(layout, None);
            }
            if let Some(l) = self.lut.take() {
                l.buffer.destroy(device);
            }
            if let Some(Some(h)) = self.hud.take() {
                h.destroy(device);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_descriptor_pool(self.pool, None);
            for p in self.pipelines {
                device.destroy_pipeline(p, None);
            }
            for l in self.layouts {
                device.destroy_pipeline_layout(l, None);
            }
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// [`Gpu::new`]'s objects so far; destroys them unless taken out.
struct Partial<'a> {
    device: &'a ash::Device,
    gpu: Gpu,
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        #[cfg(test)]
        tests::DESTROYED.set(self.gpu.live());
        // SAFETY: nothing of a `Gpu` being built was ever submitted; null handles are ignored.
        unsafe { self.gpu.destroy(self.device) };
    }
}

/// The request each slot's server starts out having seen: the last one answered (`seq_resp`), so a request
/// posted before it started is served.
fn first_seen(hdr: &neural_forge_protocol::ShmHeader) -> [u32; Slot::COUNT] {
    Slot::ALL.map(|s| hdr.seq_resp_slot(s).load(Ordering::Acquire))
}

#[allow(clippy::too_many_arguments)]
fn serve(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, loader: &Loader, view: ShmView, quit: &AtomicBool, gpu: &mut Gpu) {
    let hdr = view.header();
    // A request the layer posted before this thread started (it posts in the present that starts the
    // server) is still unanswered: `seq_resp` behind `seq_req`. Starting from `seq_resp` serves it.
    // Starting from `seq_req` lost it, and a layer whose channel still showed the previous session's
    // server running waited on it for good: after the first game on a channel, the after-the-upscaler
    // path never answered again.
    let mut last = first_seen(hdr);
    let mut frames: u64 = 0;
    let mut last_request = Instant::now();
    while !quit.load(Ordering::Relaxed) && !gpu.stalled {
        if !view.usable() {
            // The process let the channel go (it stopped presenting): another game's layer may own
            // it now, and its requests are not this server's. Picked up again once this process
            // presents and takes the lease back.
            last = Slot::ALL.map(|s| hdr.seq_req_slot(s).load(Ordering::Acquire));
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        hdr.server_state.store(neural_forge_protocol::enums::server_state::RUNNING, Ordering::Relaxed);
        hdr.model_up.store(u32::from(loader.failed().is_none()), Ordering::Relaxed);
        for slot in Slot::ALL {
            let seq = hdr.seq_req_slot(slot).load(Ordering::Acquire);
            if seq == last[slot.index()] {
                continue;
            }
            last[slot.index()] = seq;
            last_request = Instant::now();
            process(device, instance, physical, import, loader, view, gpu, slot, seq, &mut frames);
        }
        hdr.server_heartbeat.fetch_add(1, Ordering::Relaxed);
        // The model server's idle policy, simplified: poll fast while requests come, slowly when none have for a while.
        if last_request.elapsed() < Duration::from_secs(1) {
            std::thread::sleep(Duration::from_micros(200));
        } else {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process(
    device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, loader: &Loader, view: ShmView, gpu: &mut Gpu, slot: Slot,
    seq: u32, frames: &mut u64,
) {
    let hdr = view.header();
    let t = Instant::now();
    let width = hdr.width_slot(slot).load(Ordering::Relaxed);
    let height = hdr.height_slot(slot).load(Ordering::Relaxed);
    let format = hdr.proxy_format_slot(slot).load(Ordering::Relaxed);
    let dims_ok = neural_forge_protocol::frame_dims_valid(width, height, format);
    let bytes = if dims_ok { proxy_format::bytes_per_pixel(format) * width as usize * height as usize } else { 0 };
    let wanted = dims_ok
        && proxy_format::is_8bit(format)
        && width >= MIN_DIM
        && height >= MIN_DIM
        && hdr.neural_enabled()
        && hdr.apply_model.load(Ordering::Relaxed) != 0;
    if dims_ok {
        // A size or format change resets the history (`post_history`).
        gpu.history[slot.index()].note_request(width, height, format);
    }
    let tunings = hdr.pass_tunings();
    let passes: Vec<Conditioning> = tunings.passes().iter().map(|&t| Conditioning::from(t)).collect();
    let feedback = tunings.feedback;
    // The colour of the edit, and its look-up table (read from config.ini's `edit_lut` in the background).
    let colour = tunings.colour;
    let lut = super::colour::current(&colour);
    // The HUD's regions and detection (the presented frame carries the game's HUD).
    let hud = hdr.hud_settings();
    // The claim is released whatever happens after it (a claim held while the network is not built at
    // this size would keep the pre-upscaler path off it for good), except after a frame that did not
    // finish in time: its work may still use the network, so the claim stays until its fence signals
    // (`Loader::post_stalled`, checked by the pre-upscaler path's claim).
    let evaluated = wanted && loader.claim_post() && {
        // The Network setting: a switch frees the current build here, once this path's previous frame
        // was waited for (`evaluate` waits for each) and the other path is kept off by the claim.
        let ok = loader.ready(width, height, hdr.network_choice()).is_some_and(|built| {
            // SAFETY: the regions are mapped for the process's life; this thread owns the slot's request now.
            unsafe { evaluate(device, instance, physical, import, Some(loader), (view.regions(slot).0, view.regions(slot).1, view.capacity()), gpu, slot, &built, width, height, format == proxy_format::BGRA8, &passes, feedback, &colour, lut.as_ref(), &hud) }
        });
        if gpu.stalled {
            loader.post_stalled(device, gpu.fence);
        } else {
            loader.release_post();
        }
        ok
    };
    if !evaluated {
        // The effect off, the model not ready, a failed frame: the next frame starts afresh.
        gpu.history[slot.index()].skipped();
    }
    if !evaluated && gpu.stalled && gpu.slots[slot.index()].as_ref().is_some_and(|b| !b.answer.staged) {
        // The frame that did not finish writes the shared answer region itself (imported, zero-copy) and may
        // still be running: no echo into it and no answer published. The client's deadline presents the frame
        // untouched.
        return;
    }
    let (proxy, answer) = view.regions(slot);
    if !evaluated && bytes > 0 {
        // Fail open, as the model server does: the frame itself comes back.
        // SAFETY: both regions hold at least `bytes` (`frame_dims_valid` bounds the frame by the region size).
        unsafe { std::ptr::copy_nonoverlapping(proxy, answer, bytes) };
    }
    hdr.seq_ok.store(seq, Ordering::Relaxed);
    // The model server's per-stage times, as the `[sync]` line and the Status tab read them: the whole frame on
    // the GPU, waited for, counts as the evaluate; nothing is uploaded or read back separately.
    if evaluated {
        let ms = |v: f32| v.to_bits();
        hdr.server_upload_ms_bits.store(ms(0.0), Ordering::Relaxed);
        hdr.server_eval_ms_bits.store(ms(t.elapsed().as_secs_f32() * 1000.0), Ordering::Relaxed);
        hdr.server_readback_ms_bits.store(ms(0.0), Ordering::Relaxed);
    }
    if slot == Slot::Primary {
        if evaluated {
            hdr.seq_eval.store(seq, Ordering::Relaxed);
        }
        hdr.answered_w.store(if dims_ok { width } else { 0 }, Ordering::Relaxed);
        hdr.answered_h.store(if dims_ok { height } else { 0 }, Ordering::Relaxed);
        hdr.server_busy_us.store(u32::try_from(t.elapsed().as_micros()).unwrap_or(u32::MAX), Ordering::Relaxed);
    }
    hdr.seq_resp_slot(slot).store(seq, Ordering::Release);
    *frames += 1;
    neural_forge_protocol::store64(&hdr.server_frames_lo, &hdr.server_frames_hi, *frames);
}

/// Puts `lut` in [`Gpu::lut`] when it is another table than the one there (a new buffer for another size: the
/// sets are bound again). `None` keeps what is there (the strength may come back up).
///
/// # Safety
/// Nothing of `gpu`'s is pending.
unsafe fn upload_lut(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, gpu: &mut Gpu, lut: Option<&Arc<super::colour::Lut>>) {
    let Some(l) = lut else { return };
    if gpu.lut.as_ref().is_some_and(|g| g.lut.generation == l.generation) {
        return;
    }
    let entries = l.entries();
    if gpu.lut.as_ref().is_some_and(|g| g.entries != entries.len()) {
        if let Some(old) = gpu.lut.take() {
            // SAFETY: not pending (contract).
            unsafe { old.buffer.destroy(device) };
        }
    }
    if gpu.lut.is_none() {
        let Some(buffer) = super::own_host_buffer_with(device, instance, physical, (entries.len() * 16) as u64, vk::BufferUsageFlags::STORAGE_BUFFER) else {
            crate::log!("[colour] no buffer for the look-up table after the upscaler; it is left out");
            return;
        };
        gpu.lut = Some(PostLut { buffer, entries: entries.len(), lut: Arc::clone(l) });
    }
    if let Some(g) = gpu.lut.as_mut() {
        // SAFETY: host-coherent, mapped and sized for `entries`; not in use (contract).
        unsafe { std::ptr::copy_nonoverlapping(entries.as_ptr().cast::<u8>(), g.buffer.ptr, entries.len() * 16) };
        g.lut = Arc::clone(l);
    }
}

/// One frame through the network, once per entry of `passes` (each pass's settings, at least one). `false`
/// (and the caller echoes) on anything short of a written answer. `regions` are the slot's proxy and answer
/// regions and their size; `loader` (`None` in tests) is told about counter-chain timeouts.
///
/// With a history ([`Gpu::motion`] not `none`; docs/NATIVE_BACKEND.md, "History after the upscaler") each
/// pass reads its own previous output (`HistBuffers::passes`) and writes it again, the frame's proxy is kept
/// for the next frame's rejection, and with the optical flow the frame's proxy goes into the flow session
/// first (its own submit, then the flow on its queue, then the frame waiting for it).
///
/// # Safety
/// The regions are live; nothing of `gpu`'s is pending.
#[allow(clippy::too_many_arguments)]
unsafe fn evaluate(
    device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, loader: Option<&Loader>, regions: (*mut u8, *mut u8, usize), gpu: &mut Gpu, slot: Slot,
    built: &Built, width: u32, height: u32, bgra: bool, passes: &[Conditioning], feedback: f32, colour: &neural_forge_protocol::ColourSettings,
    lut: Option<&Arc<super::colour::Lut>>, hud: &neural_forge_protocol::HudSettings,
) -> bool {
    let s = slot.index();
    let bytes = u64::from(width) * u64::from(height) * 4;
    let (proxy_region, answer_region, capacity) = regions;
    if gpu.slots[s].as_ref().is_none_or(|b| (b.width, b.height) != (width, height)) {
        if let Some(old) = gpu.slots[s].take() {
            // SAFETY: nothing pending (every request waits for its own work).
            unsafe { old.destroy(device, gpu.flow.device()) };
            gpu.bound = None;
        }
        // SAFETY: the regions are mapped for the process's life with room for `bytes`.
        let buffers = unsafe {
            (region_buffer(device, instance, physical, import, proxy_region, capacity, bytes), region_buffer(device, instance, physical, import, answer_region, capacity, bytes))
        };
        let (proxy, answer) = match buffers {
            (Some(proxy), Some(answer)) => (proxy, answer),
            (proxy, answer) => {
                // SAFETY: just created, never used.
                unsafe {
                    for b in [proxy, answer].into_iter().flatten() {
                        b.destroy(device);
                    }
                }
                return false;
            }
        };
        gpu.slots[s] = Some(SlotBuffers { width, height, proxy, answer, scratch: None, hist: None, hud: None });
    }
    let more = passes.len() > 1;
    if more && gpu.slots[s].as_ref().is_some_and(|b| b.scratch.is_none()) {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: `device` is live.
        let Some(scratch) = (unsafe { OwnBuffer::new(device, instance, physical, bytes, usage) }) else { return false };
        if let Some(b) = gpu.slots[s].as_mut() {
            b.scratch = Some(scratch);
        }
        gpu.bound = None;
    }
    let blend = feedback < 1.0 && passes.len() > 1;
    if blend && gpu.feedback.is_none() {
        let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).size(8).build();
        // SAFETY: valid create infos; the layout is destroyed again if the pipeline fails.
        let built = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(&[gpu.set_layout]).push_constant_ranges(&[push]), None).ok().and_then(|layout| {
                match super::hdr::compute_pipeline(device, layout, FEEDBACK_SPV) {
                    Some(pipeline) => Some((layout, pipeline)),
                    None => {
                        device.destroy_pipeline_layout(layout, None);
                        None
                    }
                }
            })
        };
        let Some(built) = built else { return false };
        gpu.feedback = Some(built);
    }
    if passes.is_empty() {
        return false;
    }
    let history = gpu.motion != Motion::None;
    // SAFETY: nothing of the slot's is pending.
    if history && !unsafe { ensure_history(device, instance, physical, gpu, s, width, height, passes.len()) } {
        return false;
    }
    // The HUD (module docs): its buffers, and whether the scene moves as a whole since the last frame (the
    // detection runs only then; a still frame holds every confidence).
    // SAFETY: nothing of the slot's is pending.
    let hud_on = (hud.detect || !hud.regions.is_empty()) && (unsafe { ensure_hud(device, instance, physical, gpu, s, width, height) } || {
        note_hud_off();
        false
    });
    let hud_moving = match gpu.slots[s].as_mut().and_then(|b| b.hud.as_mut()) {
        Some(h) if hud_on && hud.detect => {
            // SAFETY: the proxy region holds the frame's `bytes`, written before the request was posted.
            let thumb = post_history::thumbnail(unsafe { std::slice::from_raw_parts(proxy_region, bytes as usize) }, width, height);
            // The previous proxy is the history's while it keeps one, else the HUD's own: a switch has none yet.
            let moving = h.from_history == history && h.thumb.as_deref().is_some_and(|last| Hud::moving(last, &thumb));
            h.thumb = Some(thumb);
            h.from_history = history;
            let words = Hud::region_words(&hud.regions, width, height);
            // SAFETY: host-coherent, sized for every region; not in use (contract).
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), h.regions.ptr, words.len() * 16) };
            moving
        }
        Some(h) => {
            // Without the detection its thumbnails are not kept: when it is switched on, its first frame holds.
            h.thumb = None;
            let words = Hud::region_words(&hud.regions, width, height);
            // SAFETY: as above.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), h.regions.ptr, words.len() * 16) };
            false
        }
        None => false,
    };
    // SAFETY: nothing of `gpu`'s is pending (contract).
    unsafe { upload_lut(device, instance, physical, gpu, lut.filter(|_| colour.lut_strength > 0.0)) };
    let lut = gpu.lut.as_ref().filter(|_| colour.lut_strength > 0.0).map(|l| &*l.lut);
    let lut_buffer = gpu.lut.as_ref().map_or(vk::Buffer::null(), |l| l.buffer.buffer);
    let Some(b) = gpu.slots[s].as_ref() else { return false };
    let hist = b.hist.as_ref().filter(|_| history);
    // The flow's vectors: the session's (on a frame whose passes have a history) or, in the tests, the
    // synthetic field already in the buffer.
    let session = hist.and_then(|h| h.session.as_ref()).filter(|_| matches!(gpu.flow, FlowState::Ready(_)));
    let session_live = session.is_some();
    #[cfg(test)]
    let synthetic = gpu.synthetic_flow.as_ref().map(|(grid, _)| *grid);
    #[cfg(not(test))]
    let synthetic: Option<u32> = None;
    let mode = match gpu.motion {
        Motion::Flow if session_live || synthetic.is_some() => Motion::Flow,
        Motion::None => Motion::None,
        _ => Motion::Zero,
    };
    let grid = hist.and_then(|h| h.session.as_ref().map(|x| x.grid)).or(synthetic).unwrap_or(0);
    // The history's decision for this frame: the cut test on the frame's thumbnail, the reset rules, each
    // pass's seed.
    let decision = if history {
        // SAFETY: the proxy region holds the frame's `bytes`, written before the request was posted.
        let thumb = post_history::thumbnail(unsafe { std::slice::from_raw_parts(proxy_region, bytes as usize) }, width, height);
        let d = gpu.history[s].begin(Instant::now(), thumb, super::scene_cut::threshold(), (built.network, built.generation), passes.len());
        if let Some(reset) = d.reset {
            post_history::note_reset(reset);
        }
        d
    } else {
        post_history::Decision { reset: None, passes: vec![(false, 0); passes.len()] }
    };
    let any_history = decision.passes.iter().any(|&(has, _)| has);
    let f = built.frame;
    let scratch = b.scratch.as_ref().map(|o| o.buffer);
    let flow_buffer = hist.and_then(|h| h.flow.as_ref()).map(|x| x.buffer);
    let hud_buffers = b.hud.as_ref().filter(|_| hud_on);
    let hud_buffer = hud_buffers.map_or(vk::Buffer::null(), |h| h.scale.buffer);
    let key = Bound {
        slot: s,
        width,
        height,
        generation: built.generation,
        scratch: scratch.is_some(),
        passes: passes.len(),
        history: hist.map_or(0, |h| h.passes.len()),
        flow: flow_buffer.is_some(),
        lut: lut_buffer,
        hud: hud_buffer,
    };
    if gpu.bound != Some(key) {
        let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let proxy_i = info(b.proxy.buffer);
        let (feat_i, head_i, ans_i) = (info(f.features), info(f.head), info(b.answer.buffer));
        let scratch_i = info(scratch.unwrap_or(b.proxy.buffer));
        // Without a history the shaders read none of its bindings: the proxy stands in for them.
        let or_proxy = |buffer: Option<vk::Buffer>| info(buffer.unwrap_or(b.proxy.buffer));
        let reproj_i = or_proxy(hist.map(|h| h.reproj.buffer));
        let previous_i = or_proxy(hist.map(|h| h.previous.buffer));
        let flow_i = or_proxy(flow_buffer);
        // The look-up table, or the proxy as a stand-in until there is one (nothing reads it then).
        let lut_i = or_proxy(Some(lut_buffer).filter(|l| *l != vk::Buffer::null()));
        // The HUD's scale of the edit, or the proxy as a stand-in (nothing reads it without the HUD).
        let hud_i = or_proxy(Some(hud_buffer).filter(|h| *h != vk::Buffer::null()));
        let history_i: Vec<[vk::DescriptorBufferInfo; 1]> = (0..passes.len()).map(|k| or_proxy(hist.and_then(|h| h.passes.get(k)).map(|o| o.buffer))).collect();
        let w = |set, n, i: &[vk::DescriptorBufferInfo; 1]| vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i).build();
        let mut writes = Vec::new();
        for (k, history_k) in history_i.iter().enumerate() {
            // Every pass after the first reads the previous pass's answer from the scratch.
            let input = if k == 0 { &proxy_i } else { &scratch_i };
            let (pre, post) = (gpu.sets[2 * k], gpu.sets[2 * k + 1]);
            writes.extend([w(pre, 0, input), w(pre, 1, &feat_i), w(pre, 3, history_k), w(pre, 4, &reproj_i), w(pre, 5, &proxy_i), w(pre, 6, &previous_i), w(pre, 7, &flow_i), w(pre, 9, &hud_i)]);
            writes.extend([w(post, 0, input), w(post, 1, &head_i), w(post, 2, &ans_i), w(post, 3, history_k), w(post, 4, &reproj_i), w(post, 5, &proxy_i), w(post, 8, &lut_i), w(post, 9, &hud_i)]);
        }
        if scratch.is_some() {
            // Pass feedback: from the frame (pass 2) and from the scratch in place (after), into the scratch.
            let (from_frame, from_scratch) = (gpu.sets[SETS - 2], gpu.sets[SETS - 1]);
            writes.extend([w(from_frame, 0, &proxy_i), w(from_frame, 1, &ans_i), w(from_frame, 2, &scratch_i)]);
            writes.extend([w(from_scratch, 0, &scratch_i), w(from_scratch, 1, &ans_i), w(from_scratch, 2, &scratch_i)]);
        }
        // SAFETY: the sets are not in use; every buffer is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        gpu.bound = Some(key);
    }
    if b.proxy.staged {
        // SAFETY: both hold `bytes`; the staging buffer is host-coherent and not in use.
        unsafe { std::ptr::copy_nonoverlapping(proxy_region, b.proxy.ptr, bytes as usize) };
    }
    #[cfg(test)]
    if let (Some((_, field)), Some(out)) = (gpu.synthetic_flow.as_ref(), hist.and_then(|h| h.flow.as_ref())) {
        // SAFETY: the buffer was sized for the field; not in use.
        unsafe { std::ptr::copy_nonoverlapping(field.as_ptr().cast::<u8>(), out.ptr, field.len() * 4) };
    }
    // The HUD's set: this frame's proxy, the previous one (the history's, or the HUD's own), the confidence, the
    // scale and the regions.
    let hud_rec = match (hud_buffers, gpu.hud.as_ref().and_then(Option::as_ref)) {
        (Some(h), Some(pipe)) => {
            let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
            let previous = hist.map_or(h.previous.buffer, |x| x.previous.buffer);
            let infos = [info(b.proxy.buffer), info(previous), info(h.confidence.buffer), info(h.scale.buffer), info(h.regions.buffer), info(h.near.buffer)];
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(n, i)| vk::WriteDescriptorSet::builder().dst_set(pipe.set).dst_binding(n as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i).build())
                .collect();
            // SAFETY: the set is not in use (every frame waits for its own work); the buffers are live.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            let push = HudPush {
                dims: [width, height, 0, hud.regions.len() as u32],
                flags: [u32::from(hud.detect), 0, 0, 0],
                detect: [Hud::STILL, Hud::SHARP, Hud::RISE, Hud::FALL],
                shape: [Hud::THRESHOLD, 0.0, Hud::region_feather(height), 0.0],
            };
            Some((pipe.layout, pipe.pipeline, pipe.set, push, h.fresh, h.confidence.buffer, (hist.is_none() && hud.detect).then_some(h.previous.buffer)))
        }
        _ => None,
    };
    let blend_scale = f.blend_scale;
    let reject = gpu.reject;
    let push = |c: &Conditioning, (has, seed): (bool, u32)| {
        let pre = PreprocessPush {
            dims: [f.field_width, f.field_height, width, height],
            frame: [seed, u32::from(bgra), if has { mode.shader_mode() } else { 0 }, if history { grid } else { 0 }],
            control: [c.style as f32, c.local_tone, c.local_structure, c.skin_structure],
            more: [if c.auto_mask { 1.0 } else { 0.0 }, if history { reject } else { 0.0 }, if history && hud_rec.is_some() { 1.0 } else { 0.0 }, 0.0],
        };
        let post = CompositePush {
            dims: pre.dims,
            frame: [0, u32::from(bgra), u32::from(has), u32::from(history)],
            control: [c.style as f32, c.local_tone, c.intensity, if history { blend_scale } else { 0.0 }],
            ..Default::default()
        };
        (pre, post)
    };
    let runs = passes.len();
    let barrier = |src: vk::AccessFlags, dst: vk::AccessFlags| [vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(dst).build()];
    // The optical flow runs on frames whose passes have a history; every frame's proxy goes into the session
    // (the next frame's reference): in its own submit before the flow, else in the frame's.
    let flow_frame = mode == Motion::Flow && session_live && any_history;
    let cmd = gpu.cmd;
    // SAFETY: the layer's own buffer, not pending; every handle is live.
    let recorded: ash::prelude::VkResult<()> = unsafe {
        (|| {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
            if let Some((layout, pipeline, set, push, fresh, confidence, _)) = hud_rec {
                // The HUD's scale of the edit, before every pass (the preprocess and the last composite read it).
                if fresh {
                    // A new confidence starts at 0 everywhere: nothing is HUD before the detection saw it.
                    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE), &[], &[]);
                    device.cmd_fill_buffer(cmd, confidence, 0, vk::WHOLE_SIZE, 0);
                    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                }
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                let groups = (width.div_ceil(8), height.div_ceil(8));
                if hud_moving {
                    device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<HudPush>()));
                    device.cmd_dispatch(cmd, groups.0, groups.1, 1);
                    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ), &[], &[]);
                }
                // Phase 1 (the detection's rows) with the detection on, then phase 2 (the scale).
                for phase in if push.flags[0] != 0 { 1..3 } else { 2..3 } {
                    let at = HudPush { dims: [push.dims[0], push.dims[1], phase, push.dims[3]], ..push };
                    device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&at).cast::<u8>(), std::mem::size_of::<HudPush>()));
                    device.cmd_dispatch(cmd, groups.0, groups.1, 1);
                    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                }
            }
            if let (Some(session), Some(out)) = (session, flow_buffer) {
                if flow_frame {
                    // The frame waits for the flow's semaphore at the transfer stage.
                    session.record_readback(device, cmd, out);
                    device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ), &[], &[]);
                } else {
                    // No flow on this frame (no pass has a history): its proxy still becomes the next frame's
                    // reference, and the next flow starts without the previous one's hints.
                    session.restart();
                    session.record_upload(device, cmd, b.proxy.buffer);
                    session.advance();
                }
            }
            for (k, c) in passes.iter().enumerate() {
                let (pre, post) = push(c, decision.passes[k]);
                let post = CompositePush { hud: [if hud_rec.is_some() && k + 1 == runs { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0], ..post.with_colour(super::colour::PassColour::of(colour, k, k + 1 == runs, lut, true)) };
                // Every pass after the first reads the previous one's answer: copied into `scratch`, after the
                // composite wrote it and before this pass's shaders read it (and the next composite rewrites it).
                if k > 0 {
                    let Some(scratch) = scratch else { return Err(vk::Result::ERROR_INITIALIZATION_FAILED) };
                    match gpu.feedback.filter(|_| blend) {
                        // Feedback below 1: the scratch is the pass's input moved toward its answer.
                        Some((layout, pipeline)) => {
                            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                            let push: [u32; 2] = [width * height, feedback.to_bits()];
                            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[if k == 1 { gpu.sets[SETS - 2] } else { gpu.sets[SETS - 1] }], &[]);
                            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 8));
                            device.cmd_dispatch(cmd, (width * height).div_ceil(64), 1, 1);
                            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                        }
                        None => {
                            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE), &[], &[]);
                            device.cmd_copy_buffer(cmd, b.answer.buffer, scratch, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
                            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                        }
                    }
                }
                let (pre_set, post_set) = (gpu.sets[2 * k], gpu.sets[2 * k + 1]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gpu.pipelines[0]);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.layouts[0], 0, &[pre_set], &[]);
                device.cmd_push_constants(cmd, gpu.layouts[0], vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&pre).cast::<u8>(), std::mem::size_of::<PreprocessPush>()));
                device.cmd_dispatch(cmd, f.field_width.div_ceil(8), f.field_height.div_ceil(8), 1);
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
                device.cmd_execute_commands(cmd, &[built.graph_own]);
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ), &[], &[]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gpu.pipelines[1]);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.layouts[1], 0, &[post_set], &[]);
                device.cmd_push_constants(cmd, gpu.layouts[1], vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&post).cast::<u8>(), std::mem::size_of::<CompositePush>()));
                device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
            }
            let mut last = vk::PipelineStageFlags::COMPUTE_SHADER;
            if let Some(h) = hist {
                // The frame's proxy is the next frame's previous one, copied once every pass read this one.
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE), &[], &[]);
                device.cmd_copy_buffer(cmd, b.proxy.buffer, h.previous.buffer, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
                last |= vk::PipelineStageFlags::TRANSFER;
            }
            if let Some((.., Some(previous))) = hud_rec {
                // The HUD's own previous proxy (no history keeps one), for the next frame's detection.
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE), &[], &[]);
                device.cmd_copy_buffer(cmd, b.proxy.buffer, previous, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
                last |= vk::PipelineStageFlags::TRANSFER;
            }
            device.cmd_pipeline_barrier(cmd, last, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::HOST_READ), &[], &[]);
            device.end_command_buffer(cmd)
        })()
    };
    if recorded.is_err() {
        return false;
    }
    // The flow's two submits before the frame's: the proxy into the session, then the flow.
    let waits = if flow_frame {
        // SAFETY: nothing of the session's is pending.
        match unsafe { flow_submits(device, gpu, s) } {
            Ok(waits) => Some(waits),
            Err(why) => {
                gpu.flow.give_up(device, gpu.queue, &why);
                return false;
            }
        }
    } else {
        None
    };
    let wait_stages = [vk::PipelineStageFlags::TRANSFER];
    let mut submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd));
    if let Some(sem) = waits.as_ref() {
        submit = submit.wait_semaphores(std::slice::from_ref(sem)).wait_dst_stage_mask(&wait_stages);
    }
    // SAFETY: the fence is the layer's own and idle; the queue is this thread's alone.
    let submitted = unsafe { device.reset_fences(&[gpu.fence]).and_then(|()| device.queue_submit(gpu.queue, &[submit.build()], gpu.fence)) };
    if crate::note_vk(device.handle(), submitted).is_err() {
        if waits.is_some() {
            // The flow ran (or runs) with nothing waiting for it: its queue drains, and the flow is not used again.
            gpu.flow.give_up(device, gpu.queue, "the frame's submit after the flow failed");
        }
        return false;
    }
    // SAFETY: the fence was just submitted.
    let waited = unsafe { device.wait_for_fences(&[gpu.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    if crate::note_fence_wait(device.handle(), waited, "native::post").is_err() {
        gpu.stalled = true;
        crate::log!("[native] the after-the-upscaler network frame did not finish in time; the after-the-upscaler path is not served from here on");
        return false;
    }
    if let Some((loader, (waits, at))) = loader.and_then(|l| l.chain_timeouts().filter(|(n, _)| *n > 0).map(|t| (l, t))) {
        crate::log!("[native] {waits} counter-chain wait(s) gave up ({at}); rebuilding the network with barriers between its launches");
        loader.fall_back();
        return false;
    }
    if b.answer.staged {
        // SAFETY: both hold `bytes`; the work that wrote the staging buffer completed.
        unsafe { std::ptr::copy_nonoverlapping(b.answer.ptr, answer_region, bytes as usize) };
    }
    if hud_rec.is_some() {
        // The confidence was zeroed by this frame's work, which completed.
        if let Some(h) = gpu.slots[s].as_mut().and_then(|b| b.hud.as_mut()) {
            h.fresh = false;
        }
    }
    true
}

/// The flow's submits before a flow frame on slot `s`: the upload of the frame's proxy into the session
/// (compute queue, signalling `uploaded`), then the flow (its queue, waiting for that, signalling `flowed`).
/// The semaphore the frame waits for, or why a submit failed.
///
/// # Safety
/// Nothing of the session's or the flow device's is pending.
unsafe fn flow_submits(device: &ash::Device, gpu: &Gpu, s: usize) -> Result<vk::Semaphore, String> {
    let FlowState::Ready(fd) = &gpu.flow else { return Err("no flow device".into()) };
    let Some(b) = gpu.slots[s].as_ref() else { return Err("no buffers".into()) };
    let Some(session) = b.hist.as_ref().and_then(|h| h.session.as_ref()) else { return Err("no session".into()) };
    // SAFETY: the caller's contract; the buffers are the device's own and idle.
    unsafe {
        device.reset_command_buffer(fd.upload, vk::CommandBufferResetFlags::empty()).map_err(|e| format!("{e:?}"))?;
        device.begin_command_buffer(fd.upload, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| format!("{e:?}"))?;
        let host = [vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ).build()];
        device.cmd_pipeline_barrier(fd.upload, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &host, &[], &[]);
        session.record_upload(device, fd.upload, b.proxy.buffer);
        device.end_command_buffer(fd.upload).map_err(|e| format!("{e:?}"))?;
        session.record_flow(device, fd)?;
        let upload = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&fd.upload)).signal_semaphores(std::slice::from_ref(&fd.uploaded)).build();
        device.queue_submit(gpu.queue, &[upload], vk::Fence::null()).map_err(|e| format!("the upload's submit: {e:?}"))?;
        session.advance();
        let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
        let flow = vk::SubmitInfo::builder()
            .command_buffers(std::slice::from_ref(&fd.cmd))
            .wait_semaphores(std::slice::from_ref(&fd.uploaded))
            .wait_dst_stage_mask(&stage)
            .signal_semaphores(std::slice::from_ref(&fd.flowed))
            .build();
        device.queue_submit(fd.queue, &[flow], vk::Fence::null()).map_err(|e| format!("the flow's submit: {e:?}"))?;
    }
    Ok(fd.flowed)
}

/// Builds slot `s`'s history buffers for this size and `passes` passes (and the flow's device and
/// session with `flow`), as far as they are missing. `false` when a buffer could not be made (the frame is
/// echoed); the flow's failures make it `zero` instead, logged once.
///
/// # Safety
/// Nothing of the slot's is pending.
#[allow(clippy::too_many_arguments)]
unsafe fn ensure_history(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, gpu: &mut Gpu, s: usize, width: u32, height: u32, passes: usize) -> bool {
    let pixels = u64::from(width) * u64::from(height);
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let Some(b) = gpu.slots[s].as_mut() else { return false };
    if b.hist.is_none() {
        // SAFETY: `device` is live; each is destroyed again if the other fails.
        let made = unsafe {
            (
                OwnBuffer::new(device, instance, physical, pixels * 4, storage | vk::BufferUsageFlags::TRANSFER_DST),
                OwnBuffer::new(device, instance, physical, pixels * 8, storage),
            )
        };
        match made {
            (Some(previous), Some(reproj)) => b.hist = Some(HistBuffers { previous, reproj, passes: Vec::new(), flow: None, session: None, flow_failed: false }),
            (previous, reproj) => {
                // SAFETY: never used.
                unsafe {
                    for o in [previous, reproj].into_iter().flatten() {
                        o.destroy(device);
                    }
                }
                return false;
            }
        }
        gpu.bound = None;
    }
    let Some(h) = b.hist.as_mut() else { return false };
    while h.passes.len() < passes {
        // SAFETY: `device` is live.
        let Some(o) = (unsafe { OwnBuffer::new(device, instance, physical, pixels * 8, storage) }) else { return false };
        h.passes.push(o);
        gpu.bound = None;
    }
    #[cfg(test)]
    if let Some((_, field)) = gpu.synthetic_flow.as_ref() {
        if h.flow.is_none() {
            h.flow = super::own_host_buffer_with(device, instance, physical, field.len() as u64 * 4, storage);
            gpu.bound = None;
        }
        return h.flow.is_some();
    }
    if gpu.motion != Motion::Flow || h.flow_failed {
        return true;
    }
    if matches!(gpu.flow, FlowState::Pending) {
        gpu.flow = match gpu.setup.flow {
            None => FlowState::unavailable("the device was created without the optical flow (see the [queues] and [native] lines above)", None),
            // SAFETY: the device was created with this queue; the pool is the server's.
            Some(queue) => match unsafe { post_flow::FlowDevice::new(device, instance, &gpu.setup, queue, gpu.setup.family, gpu.command_pool) } {
                Ok(fd) => {
                    crate::log!("[native] after the upscaler: the history follows the optical flow (queue {} of family {})", queue.index, queue.family);
                    FlowState::Ready(fd)
                }
                Err(why) => FlowState::unavailable(&why, None),
            },
        };
    }
    let FlowState::Ready(fd) = &gpu.flow else { return true };
    if h.session.is_none() {
        // SAFETY: `device` is live and `fd` its.
        match unsafe { post_flow::Session::new(device, instance, physical, fd, width, height) } {
            Ok(session) => {
                let cells = u64::from(session.cells.0) * u64::from(session.cells.1);
                match super::own_host_buffer_with(device, instance, physical, cells * 4, storage | vk::BufferUsageFlags::TRANSFER_DST) {
                    Some(out) => {
                        crate::log!("[native] after the upscaler: optical flow at {width}x{height}, a {0}x{0} grid ({1}x{2} vectors)", session.grid, session.cells.0, session.cells.1);
                        h.session = Some(session);
                        h.flow = Some(out);
                        gpu.bound = None;
                    }
                    None => {
                        // SAFETY: never used.
                        unsafe { session.destroy(device, fd) };
                        h.flow_failed = true;
                        note_flow_off("no buffer for its vectors");
                    }
                }
            }
            Err(why) => {
                h.flow_failed = true;
                note_flow_off(&why);
            }
        }
    }
    true
}

/// The HUD at present (docs/NATIVE_BACKEND.md, "The HUD at present"; `native_post_hud.comp`): the detection's
/// constants. A pixel counts as still at a largest channel change of [`Hud::STILL`] or less, as sharp where nine
/// taps two pixels apart span [`Hud::SHARP`] of luma; its confidence rises by [`Hud::RISE`] on a moving frame
/// where it is both and falls by [`Hud::FALL`] where it is not, and above [`Hud::THRESHOLD`] it is HUD, the edit
/// kept off it and fading back over the two pixels around it (the shader's two passes of five taps). A frame
/// moves when at least [`Hud::MOVING_SHARE`] of its thumbnail cells changed by [`Hud::MOVING_LEVEL`] or more.
pub(crate) struct Hud;

impl Hud {
    pub(crate) const STILL: f32 = 2.0 / 255.0;
    pub(crate) const SHARP: f32 = 0.15;
    pub(crate) const RISE: f32 = 1.0 / 16.0;
    pub(crate) const FALL: f32 = 1.0 / 4.0;
    /// Nine moving frames in a row before a pixel is HUD: a new confidence detects nothing for eight.
    pub(crate) const THRESHOLD: f32 = 0.5;
    pub(crate) const MOVING_LEVEL: u8 = 4;
    pub(crate) const MOVING_SHARE: f32 = 0.3;

    /// A region's feather in pixels at an output `height`: 4 at 1440 lines, at least 2.
    pub(crate) fn region_feather(height: u32) -> f32 {
        (height as f32 / 360.0).round().max(2.0)
    }

    /// Whether the scene moves as a whole between the thumbnails `last` and `now` ([`post_history::thumbnail`]):
    /// enough of its cells changed. Still (or not comparable) holds every confidence.
    pub(crate) fn moving(last: &[u8], now: &[u8]) -> bool {
        if last.len() != now.len() || now.is_empty() {
            return false;
        }
        let changed = last.iter().zip(now).filter(|(a, b)| a.abs_diff(**b) >= Self::MOVING_LEVEL).count();
        changed as f32 >= Self::MOVING_SHARE * now.len() as f32
    }

    /// The regions as the shader reads them: per region its rectangle in pixels and (mode, strength, 0, 0).
    fn region_words(regions: &[neural_forge_protocol::HudRegion], width: u32, height: u32) -> Vec<[f32; 4]> {
        let (w, h) = (width as f32, height as f32);
        regions.iter().flat_map(|r| [[r.x * w, r.y * h, (r.x + r.w) * w, (r.y + r.h) * h], [r.mode as f32, r.strength, 0.0, 0.0]]).collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HudPush {
    dims: [u32; 4],
    flags: [u32; 4],
    detect: [f32; 4],
    shape: [f32; 4],
}

/// [`Gpu::hud`]: `native_post_hud.comp`'s layout, pipeline and its one set.
struct HudPipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl HudPipeline {
    /// # Safety
    /// `device` is live.
    unsafe fn build(device: &ash::Device) -> Option<Self> {
        let mut h = Self { set_layout: vk::DescriptorSetLayout::null(), layout: vk::PipelineLayout::null(), pipeline: vk::Pipeline::null(), pool: vk::DescriptorPool::null(), set: vk::DescriptorSet::null() };
        // SAFETY: valid create infos; on a failure what was made is destroyed (null handles are ignored).
        let built = unsafe {
            (|| {
                let binding = |n: u32| vk::DescriptorSetLayoutBinding::builder().binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build();
                let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..6).map(binding).collect();
                h.set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings), None).ok()?;
                let push = vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<HudPush>() as u32).build();
                h.layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(&[h.set_layout]).push_constant_ranges(&[push]), None).ok()?;
                h.pipeline = super::hdr::compute_pipeline(device, h.layout, HUD_SPV)?;
                let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 }];
                h.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes), None).ok()?;
                h.set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(h.pool).set_layouts(&[h.set_layout])).ok()?[0];
                Some(())
            })()
        };
        match built {
            Some(()) => Some(h),
            None => {
                // SAFETY: nothing of it was used.
                unsafe { h.destroy(device) };
                None
            }
        }
    }

    /// # Safety
    /// Nothing of it is pending.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: forwarded (null handles are ignored).
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// One slot's HUD buffers, for its frame size.
struct HudBuffers {
    /// Per pixel, f32: the detection's confidence (0-1), kept from frame to frame.
    confidence: OwnBuffer,
    /// Per pixel, f32: the scale of the edit (1 untouched), written every frame.
    scale: OwnBuffer,
    /// Per pixel, f32: the detection's distance along the row to a confident pixel (the shader's phase 1).
    near: OwnBuffer,
    /// The previous frame's proxy while the history keeps none (the history's own otherwise), copied at the end
    /// of each frame.
    previous: OwnBuffer,
    /// The regions as the shader reads them ([`Hud::region_words`]).
    regions: HostBuffer,
    /// The confidence is undefined until the next frame zeroes it.
    fresh: bool,
    /// The last frame's thumbnail ([`Hud::moving`]), and whether its previous proxy was the history's.
    thumb: Option<Vec<u8>>,
    from_history: bool,
}

impl HudBuffers {
    /// # Safety
    /// Nothing of it is pending.
    unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: forwarded.
        unsafe {
            self.confidence.destroy(device);
            self.scale.destroy(device);
            self.near.destroy(device);
            self.previous.destroy(device);
            self.regions.destroy(device);
        }
    }
}

/// Builds slot `s`'s HUD buffers for this size (and the pipeline, once) as far as they are missing. `false`
/// when something could not be made: the frame then runs without the HUD (logged once).
///
/// # Safety
/// Nothing of the slot's is pending.
unsafe fn ensure_hud(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, gpu: &mut Gpu, s: usize, width: u32, height: u32) -> bool {
    if gpu.hud.is_none() {
        // SAFETY: `device` is live.
        gpu.hud = Some(unsafe { HudPipeline::build(device) });
    }
    if !matches!(gpu.hud, Some(Some(_))) {
        return false;
    }
    let Some(b) = gpu.slots[s].as_mut() else { return false };
    if b.hud.is_some() {
        return true;
    }
    let pixels = u64::from(width) * u64::from(height);
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    // SAFETY: `device` is live; whatever was made is destroyed again if anything fails.
    let made = unsafe {
        (
            OwnBuffer::new(device, instance, physical, pixels * 4, storage | vk::BufferUsageFlags::TRANSFER_DST),
            OwnBuffer::new(device, instance, physical, pixels * 4, storage),
            OwnBuffer::new(device, instance, physical, pixels * 4, storage),
            OwnBuffer::new(device, instance, physical, pixels * 4, storage | vk::BufferUsageFlags::TRANSFER_DST),
            super::own_host_buffer_with(device, instance, physical, (neural_forge_protocol::MAX_HUD_REGIONS * 32) as u64, storage),
        )
    };
    match made {
        (Some(confidence), Some(scale), Some(near), Some(previous), Some(regions)) => {
            b.hud = Some(HudBuffers { confidence, scale, near, previous, regions, fresh: true, thumb: None, from_history: false });
            gpu.bound = None;
            true
        }
        (confidence, scale, near, previous, regions) => {
            // SAFETY: never used.
            unsafe {
                for o in [confidence, scale, near, previous].into_iter().flatten() {
                    o.destroy(device);
                }
                if let Some(r) = regions {
                    r.destroy(device);
                }
            }
            false
        }
    }
}

/// Logs (once per process) that the HUD's buffers or pipeline could not be built.
fn note_hud_off() {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::log!("[native] after the upscaler: the HUD's buffers or pipeline could not be built; frames run without the HUD regions and detection");
    }
}

/// Logs (once per process) why `flow` runs as `zero`.
fn note_flow_off(why: &str) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::log!("[native] after the upscaler: no optical flow ({why}); the history keeps zero motion");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Makes [`Gpu::new`] fail at this step.
        pub(super) static FAIL_AT: Cell<Option<u32>> = const { Cell::new(None) };
        /// How many objects the last failed [`Gpu::new`]'s guard held, all destroyed.
        pub(super) static DESTROYED: Cell<usize> = const { Cell::new(usize::MAX) };
    }

    /// A failed setup destroys everything it created before the failure, at every step, so starting the
    /// server again after one leaks nothing.
    #[test]
    fn a_failed_setup_destroys_what_it_created() {
        let Some((_entry, _instance, physical, device, _queue, family)) = crate::composition::gpu::test_device() else { return };
        unsafe extern "system" fn gipa(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
            None
        }
        let setup = Setup { gipa, instance: vk::Instance::null(), physical, frame_family: family, family, index: 0, post_index: 0, properties2: crate::vram::Properties2::None, flow: None };
        // Objects that exist when each step fails (the sets and the command buffer go with their pools).
        let before = [0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 7];
        for (step, &created) in before.iter().enumerate() {
            FAIL_AT.set(Some(step as u32));
            DESTROYED.set(usize::MAX);
            // SAFETY: a live test device whose queue 0 of `family` exists.
            assert!(unsafe { Gpu::new(&device, setup) }.is_none(), "step {step} did not fail");
            assert_eq!(DESTROYED.get(), created, "failing at step {step}");
        }
        FAIL_AT.set(None);
        // SAFETY: as above.
        let mut gpu = unsafe { Gpu::new(&device, setup) }.expect("builds without the injected failure");
        assert_eq!(gpu.live(), 8);
        // SAFETY: nothing of it was submitted.
        unsafe { gpu.destroy(&device) };
        unsafe { device.destroy_device(None) };
    }

    /// The colour of the edit after the upscaler (`native_post_composite.comp` with its colour bits), on the
    /// real pipeline and an RGBA8 proxy with a head the test writes: the step bits with neutral values give
    /// the plain answer bit for bit; saturation 0 leaves a grey change; the dither moves a level by at most one
    /// step (each pixel the level below or above the value), uses both over a flat area and averages to it,
    /// where without it every pixel is the same rounded level.
    #[test]
    fn the_colour_of_the_edit_after_the_upscaler_reshapes_and_dithers_the_8_bit_answer() {
        let Some((_entry, _instance, physical, device, _queue, family)) = crate::composition::gpu::test_device() else { return };
        let instance = &_instance;
        unsafe extern "system" fn gipa(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
            None
        }
        let setup = Setup { gipa, instance: vk::Instance::null(), physical, frame_family: family, family, index: 0, post_index: 0, properties2: crate::vram::Properties2::None, flow: None };
        const W: u32 = 16;
        const H: u32 = 16;
        // SAFETY: a live test device whose queue 0 of `family` exists; every object is waited for.
        unsafe {
            let mut gpu = Gpu::new(&device, setup).expect("builds");
            let host = |bytes: u64| super::super::own_host_buffer_with(&device, instance, physical, bytes, vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
            let (proxy, head, answer, lut, history) = (host(u64::from(W * H) * 4), host(u64::from(W * H) * 16), host(u64::from(W * H) * 4), host(64), host(u64::from(W * H) * 8));
            let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
            let (pi, hi, ai, li, si) = (info(proxy.buffer), info(head.buffer), info(answer.buffer), info(lut.buffer), info(history.buffer));
            let w = |n, i: &[vk::DescriptorBufferInfo; 1]| vk::WriteDescriptorSet::builder().dst_set(gpu.sets[1]).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i).build();
            // No history (frame.z and .w 0): its two bindings point at a buffer nothing reads.
            // Nor the frame's proxy and the HUD's scale (no HUD: p.hud.x 0).
            device.update_descriptor_sets(&[w(0, &pi), w(1, &hi), w(2, &ai), w(3, &si), w(4, &si), w(5, &pi), w(8, &li), w(9, &si)], &[]);
            let fill = |proxy_of: &dyn Fn(u32, u32, u32) -> u8, edit: &dyn Fn(u32, u32, u32) -> f32| {
                let p: Vec<u8> = (0..H).flat_map(|y| (0..W).flat_map(move |x| [proxy_of(x, y, 0), proxy_of(x, y, 1), proxy_of(x, y, 2), 255])).collect();
                std::ptr::copy_nonoverlapping(p.as_ptr(), proxy.ptr, p.len());
                let h: Vec<f32> = (0..H).flat_map(|y| (0..W).flat_map(move |x| [4.0 * edit(x, y, 0), 4.0 * edit(x, y, 1), 4.0 * edit(x, y, 2), 0.0])).collect();
                std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), head.ptr, h.len() * 4);
            };
            let run = |c: super::super::colour::PassColour| -> Vec<u8> {
                let push = CompositePush { dims: [W, H, W, H], frame: [0, 0, 0, 0], control: [0.0, 1.0, 1.0, 0.0], ..Default::default() }.with_colour(c);
                let cmd = gpu.cmd;
                device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gpu.pipelines[1]);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.layouts[1], 0, &[gpu.sets[1]], &[]);
                device.cmd_push_constants(cmd, gpu.layouts[1], vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&push).cast::<u8>(), std::mem::size_of::<CompositePush>()));
                device.cmd_dispatch(cmd, W.div_ceil(8), H.div_ceil(8), 1);
                let to_host = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ).build();
                device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[to_host], &[], &[]);
                device.end_command_buffer(cmd).unwrap();
                device.reset_fences(&[gpu.fence]).unwrap();
                device.queue_submit(gpu.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], gpu.fence).unwrap();
                device.wait_for_fences(&[gpu.fence], true, u64::MAX).unwrap();
                std::slice::from_raw_parts(answer.ptr, (W * H * 4) as usize).to_vec()
            };
            use super::super::colour::{PassColour, DITHER, GUARD, STEP};
            let flags = |f: u32| PassColour { colour: [1.0; 4], flags: [f, 0, 0, 0], lut_offset: [0.0, 0.0, 0.0, 0.05], ..PassColour::default() };
            // A varied proxy and an edit of up to 12/255 a channel, with chroma.
            let proxy_of = |x: u32, y: u32, c: u32| (60 + (x * 13 + y * 7 + c * 29) % 120) as u8;
            let edit = |x: u32, y: u32, c: u32| (((x * 5 + y * 3 + c * 2) % 7) as f32 - 3.0) * 4.0 / 255.0;
            fill(&proxy_of, &edit);
            let plain = run(PassColour::default());
            assert_eq!(run(flags(STEP | GUARD)), plain, "neutral values leave the 8-bit answer bit for bit");
            let at = |a: &[u8], x: u32, y: u32, c: u32| a[((y * W + x) * 4 + c) as usize];
            let grey = run(PassColour { colour: [0.0, 1.0, 1.0, 0.0], ..flags(STEP | GUARD) });
            for y in 0..H {
                for x in 0..W {
                    let d: Vec<i32> = (0..3).map(|c| i32::from(at(&grey, x, y, c)) - i32::from(proxy_of(x, y, c))).collect();
                    assert!((d[0] - d[1]).abs() <= 1 && (d[1] - d[2]).abs() <= 1, "saturation 0 leaves a grey change at {x},{y}: {d:?}");
                }
            }
            assert!((0..W).any(|x| (0..3).map(|c| i32::from(at(&plain, x, 0, c)) - i32::from(proxy_of(x, 0, c))).collect::<Vec<_>>().windows(2).any(|w| (w[0] - w[1]).abs() > 2)), "the test's edit has chroma");

            // The dither on a flat area whose answer lies a third of a step above level 100.
            fill(&|_, _, _| 100, &|_, _, _| 0.34 / 255.0);
            let level = |v: u8| f32::from(v);
            let undithered = run(PassColour::default());
            assert!(undithered.chunks(4).all(|p| p[..3] == [100, 100, 100]), "without the dither every pixel rounds to 100");
            let dithered = run(flags(DITHER));
            let values: Vec<f32> = dithered.chunks(4).flat_map(|p| p[..3].iter().map(|&v| level(v)).collect::<Vec<_>>()).collect();
            assert!(values.iter().all(|&v| v == 100.0 || v == 101.0), "a level below or above the value, never further");
            // The value on the CPU: the residual on the proxy, truncated to the half grid (a normal half here).
            let p = 100.0f32 / 255.0;
            let v = f32::from_bits((((4.0 * 0.34 / 255.0f32) * 0.03125 + (p * 0.125 - 0.0625)) * 8.0 + 0.5).to_bits() & !0x1fff) * 255.0;
            assert!(v > 100.2 && v < 100.4, "the test's value: {v}");
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            assert!((mean - v).abs() < 0.025, "the dither averages to the value: {mean} against {v}");
            assert_eq!(dithered.chunks(4).map(|p| p[3]).collect::<Vec<_>>(), vec![255; (W * H) as usize], "alpha is the proxy's");

            for b in [&proxy, &head, &answer, &lut, &history] {
                b.destroy(&device);
            }
            gpu.destroy(&device);
            device.destroy_device(None);
        }
    }

    /// The layer posts its first request in the present that starts the server: one still unanswered when
    /// the server starts is served, whatever the previous session left in the channel.
    #[test]
    fn a_request_posted_before_the_server_started_is_served() {
        let hdr = neural_forge_protocol::ShmHeader::default();
        hdr.init_defaults();
        // The previous session answered request 7; this session's layer posted 8 before the server started.
        hdr.seq_resp_slot(Slot::Primary).store(7, Ordering::Relaxed);
        hdr.seq_req_slot(Slot::Primary).store(8, Ordering::Relaxed);
        let seen = first_seen(&hdr);
        assert_ne!(seen[Slot::Primary.index()], hdr.seq_req_slot(Slot::Primary).load(Ordering::Relaxed), "request 8 is still to serve");
        assert_eq!(seen[Slot::Secondary.index()], hdr.seq_req_slot(Slot::Secondary).load(Ordering::Relaxed), "an answered slot has nothing to serve");
    }

    /// The valid extent and the network's field of the GPU tests (the field pads by mirroring).
    const W: u32 = 40;
    const H: u32 = 24;
    const FW: u32 = 48;
    const FH: u32 = 32;

    /// A test device with the server's objects, a network "graph" that does nothing (the head is what the
    /// test writes) and host-visible features and head buffers.
    struct Rig {
        device: ash::Device,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        gpu: Gpu,
        features: HostBuffer,
        head: HostBuffer,
        graph_pool: vk::CommandPool,
        built: Built,
        proxy: Vec<u8>,
        answer: Vec<u8>,
        /// The HUD settings the frames run with (the shipped default: the detection on, no region).
        hud: neural_forge_protocol::HudSettings,
        /// Keeps the Vulkan library loaded (dropped last).
        _entry: ash::Entry,
    }

    impl Rig {
        fn new() -> Option<Self> {
            let (entry, instance, physical, device, _queue, family) = crate::composition::gpu::test_device()?;
            unsafe extern "system" fn gipa(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
                None
            }
            let setup = Setup { gipa, instance: vk::Instance::null(), physical, frame_family: family, family, index: 0, post_index: 0, properties2: crate::vram::Properties2::None, flow: None };
            // SAFETY: a live test device; every object below is destroyed in Drop after the device idles.
            unsafe {
                let gpu = Gpu::new(&device, setup).expect("the server's objects build");
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
                let features = super::super::own_host_buffer_with(&device, &instance, physical, u64::from(FW * FH) * 64, usage).unwrap();
                let head = super::super::own_host_buffer_with(&device, &instance, physical, u64::from(FW * FH) * 16, usage).unwrap();
                let graph_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
                let graph = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(graph_pool).level(vk::CommandBufferLevel::SECONDARY).command_buffer_count(1)).unwrap()[0];
                let inheritance = vk::CommandBufferInheritanceInfo::default();
                device.begin_command_buffer(graph, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE).inheritance_info(&inheritance)).unwrap();
                device.end_command_buffer(graph).unwrap();
                let frame = neural_forge_native::Frame {
                    field_width: FW,
                    field_height: FH,
                    features: features.buffer,
                    features_bytes: u64::from(FW * FH) * 64,
                    head: head.buffer,
                    head_bytes: u64::from(FW * FH) * 16,
                    blend_scale: 0.5,
                    chained: 0,
                };
                let built = Built { width: W, height: H, frame, graph, graph_own: graph, generation: 1, network: neural_forge_protocol::enums::network::FULL };
                let bytes = (W * H * 4) as usize;
                let hud = neural_forge_protocol::HudSettings { detect: true, regions: Vec::new() };
                Some(Self { device, instance, physical, gpu, features, head, graph_pool, built, proxy: vec![0; bytes], answer: vec![0; bytes], hud, _entry: entry })
            }
        }

        /// The head the "network" answers with: RGB residual and blend logit per field pixel.
        fn write_head(&self, head: impl Fn(u32, u32) -> [f32; 4]) {
            let h: Vec<f32> = (0..FH).flat_map(|y| (0..FW).flat_map(move |x| (0..4).map(move |c| (x, y, c)))).map(|(x, y, c)| head(x, y)[c]).collect();
            // SAFETY: host-coherent, sized for the field's four lanes; nothing pending.
            unsafe { std::ptr::copy_nonoverlapping(h.as_ptr().cast::<u8>(), self.head.ptr, h.len() * 4) };
        }

        /// One frame of `proxy` (RGBA8 texels of the valid extent) through `passes`; the answer's texels.
        fn frame(&mut self, proxy: &[u32], passes: &[Conditioning]) -> Vec<u32> {
            for (i, t) in proxy.iter().enumerate() {
                self.proxy[i * 4..i * 4 + 4].copy_from_slice(&t.to_le_bytes());
            }
            let regions = (self.proxy.as_mut_ptr(), self.answer.as_mut_ptr(), self.proxy.len());
            // SAFETY: the regions live as long as the rig; nothing of the rig's is pending.
            let ok = unsafe { evaluate(&self.device, &self.instance, self.physical, false, None, regions, &mut self.gpu, Slot::Primary, &self.built, W, H, false, passes, 1.0, &neural_forge_protocol::ColourSettings::DEFAULT, None, &self.hud) };
            assert!(ok, "the frame was answered");
            self.answer.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect()
        }

        /// The 16 input lanes the last pass's preprocess wrote at field pixel (x, y).
        fn lanes(&self, x: u32, y: u32) -> [f32; 16] {
            let mut l = [0.0; 16];
            // SAFETY: host-coherent, sized for the field; the frame was waited for.
            unsafe { std::ptr::copy_nonoverlapping(self.features.ptr.cast::<f32>().add(((y * FW + x) * 16) as usize), l.as_mut_ptr(), 16) };
            l
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            // SAFETY: every frame was waited for; the device idles before anything is freed.
            unsafe {
                let _ = self.device.device_wait_idle();
                self.gpu.destroy(&self.device);
                self.device.destroy_command_pool(self.graph_pool, None);
                self.features.destroy(&self.device);
                self.head.destroy(&self.device);
                self.device.destroy_device(None);
            }
        }
    }

    /// A deterministic RGBA8 test picture (alpha varies too: it must come back as it was).
    fn picture(seed: u32) -> Vec<u32> {
        (0..W * H)
            .map(|i| {
                let (x, y) = (i % W, i / W);
                let r = (x * 6 + y * 3 + seed * 17) % 256;
                let g = (x * 2 + y * 9 + seed * 5 + 40) % 256;
                let b = (255 - x * 4 - y) % 256;
                let a = (x + y * 7) % 256;
                r | (g << 8) | (b << 16) | (a << 24)
            })
            .collect()
    }

    /// A residual and logit pattern of the "network".
    fn head_pattern(k: f32) -> impl Fn(u32, u32) -> [f32; 4] {
        move |x, y| [((x % 7) as f32 - 3.0) * 0.25 * k, ((y % 5) as f32 - 2.0) * 0.5, (((x + y) % 3) as f32 - 1.0) * 0.75, 0.0]
    }

    /// FNV-1a over the answer's texels and the exact-arithmetic lanes (3 to 15: the noise lanes 0 to 2 go
    /// through log2 and cos, whose last bits differ between Vulkan implementations) of every field pixel.
    fn digest(answer: &[u32], rig: &Rig) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u32| {
            for b in v.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        answer.iter().for_each(|&t| eat(t));
        for y in 0..FH {
            for x in 0..FW {
                rig.lanes(x, y)[3..].iter().for_each(|v| eat(v.to_bits()));
            }
        }
        h
    }

    /// The shipped after-the-upscaler frame (3.7.0, no history), as `NEURAL_FORGE_POST_MOTION=none` must
    /// keep it bit for bit: the answers and lanes of a one-pass and a three-pass frame with the Model tab's
    /// defaults and with an intensity and conditioning, pinned from 3.7.0's shaders.
    #[test]
    fn the_shipped_frame_stays_bit_for_bit() {
        let Some(mut rig) = Rig::new() else { return };
        rig.write_head(head_pattern(1.0));
        // Style 0: the style operator's exp2 and log2 differ in their last bits between Vulkan implementations.
        let styled = Conditioning { style: 0, intensity: 0.6, local_tone: 0.7, local_structure: 1.3, skin_structure: 0.5, auto_mask: false };
        let mut got = Vec::new();
        for passes in [vec![Conditioning::default()], vec![styled], vec![Conditioning::default(), styled, Conditioning::default()]] {
            for seed in [0, 1] {
                let answer = rig.frame(&picture(seed), &passes);
                got.push(digest(&answer, &rig));
                // The noise lanes: a fixed seed, every frame the same.
                let first = rig.lanes(5, 3);
                let _ = rig.frame(&picture(seed + 7), &passes);
                assert_eq!(rig.lanes(5, 3)[..3], first[..3], "the noise lanes do not change between frames");
                // Lanes 7 to 9 copy the proxy (4 to 6).
                for (x, y) in [(0, 0), (5, 3), (W - 1, H - 1), (FW - 1, FH - 1)] {
                    let l = rig.lanes(x, y);
                    assert_eq!(l[7..10], l[4..7], "no history at ({x}, {y})");
                }
            }
        }
        assert_eq!(got, SHIPPED, "the frame moved");
    }

    /// [`the_shipped_frame_stays_bit_for_bit`]'s digests, from 3.7.0's shaders (commit 140c7a2).
    const SHIPPED: [u64; 6] = [0xafea_81f1_9746_c790, 0xbaf4_583a_665d_da89, 0xb04e_20c6_c636_d0f5, 0x1d9c_9395_0c08_6565, 0x80ee_55d5_8453_dfd5, 0x1756_013e_f379_6ca2];

    impl Rig {
        /// A rig whose frames keep a history (`motion`), with the default rejection.
        fn with(motion: Motion) -> Option<Self> {
            let mut rig = Self::new()?;
            rig.gpu.motion = motion;
            rig.gpu.flow = if motion == Motion::Flow { FlowState::Pending } else { FlowState::Off };
            Some(rig)
        }
    }

    /// A head with logit 0: the blend weight is sigmoid(0) x the rig's blend scale 0.5 = 0.25.
    const WEIGHT: f32 = 0.25;

    fn channel(t: u32, c: usize) -> f32 {
        f32::from(t.to_le_bytes()[c]) / 255.0
    }

    /// `picture` with `delta` added to every colour channel inside the rectangle.
    fn changed(base: &[u32], (x0, y0, x1, y1): (u32, u32, u32, u32), delta: u8) -> Vec<u32> {
        base.iter()
            .enumerate()
            .map(|(i, &t)| {
                let (x, y) = (i as u32 % W, i as u32 / W);
                if (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                    let b = t.to_le_bytes();
                    u32::from_le_bytes([b[0].saturating_add(delta), b[1].saturating_add(delta), b[2].saturating_add(delta), b[3]])
                } else {
                    t
                }
            })
            .collect()
    }

    /// The frame `none` answers for `proxy` with `head` (no history: what a reset frame must give).
    fn without_history(proxy: &[u32], head: impl Fn(u32, u32) -> [f32; 4], passes: &[Conditioning]) -> Option<(Vec<u32>, Rig)> {
        let mut rig = Rig::new()?;
        rig.write_head(head);
        let a = rig.frame(proxy, passes);
        Some((a, rig))
    }

    /// `NEURAL_FORGE_POST_MOTION=zero`: a still sequence's second frame blends toward the first frame's
    /// output by the composite's weight; lanes 7 to 9 carry that output (centred).
    #[test]
    fn zero_motion_blends_a_still_frame_toward_the_previous_output() {
        let Some(mut rig) = Rig::with(Motion::Zero) else { return };
        let one = [Conditioning::default()];
        let p = picture(0);
        rig.write_head(head_pattern(1.0));
        let first = rig.frame(&p, &one);
        let l1 = rig.lanes(5, 3);
        assert_eq!(l1[7..10], l1[4..7], "the first frame has no history");
        rig.write_head(head_pattern(-1.0));
        let second = rig.frame(&p, &one);
        let Some((plain, _)) = without_history(&p, head_pattern(-1.0), &one) else { return };
        let mut moved = 0;
        for i in 0..(W * H) as usize {
            for c in 0..3 {
                let expected = plain[i].to_le_bytes()[c] as f32 / 255.0 * (1.0 - WEIGHT) + channel(first[i], c) * WEIGHT;
                assert!((channel(second[i], c) - expected).abs() <= 1.5 / 255.0, "pixel {i} channel {c}: {} against {expected}", channel(second[i], c));
            }
            moved += usize::from(second[i] != plain[i]);
            assert_eq!(second[i] >> 24, p[i] >> 24, "alpha comes back as it was");
        }
        assert!(moved > (W * H) as usize / 2, "the history moved most pixels ({moved})");
        for (x, y) in [(5, 3), (W - 1, H - 1), (17, 11)] {
            let l = rig.lanes(x, y);
            let t = first[(y * W + x) as usize];
            for c in 0..3 {
                let centred = (channel(t, c) - 0.5) * 0.125;
                assert!((l[7 + c] - centred).abs() <= 0.125 / 255.0 + 0.001, "lane {} at ({x}, {y})", 7 + c);
            }
        }
    }

    /// A pixel that changed by more than the threshold since the previous frame gets no history (its lanes
    /// and answer are a first frame's); one that changed by less keeps it. The threshold is the setting's.
    #[test]
    fn a_changed_pixel_is_rejected() {
        let one = [Conditioning::default()];
        let p = picture(0);
        let big = (2, 2, 10, 8); // +60 of 255: 0.235
        let small = (20, 10, 30, 20); // +20 of 255: 0.078
        let next = changed(&changed(&p, big, 60), small, 20);
        let Some((plain, _)) = without_history(&next, head_pattern(-1.0), &one) else { return };
        let run = |reject: f32| {
            let mut rig = Rig::with(Motion::Zero).unwrap();
            rig.gpu.reject = reject;
            rig.write_head(head_pattern(1.0));
            rig.frame(&p, &one);
            rig.write_head(head_pattern(-1.0));
            let a = rig.frame(&next, &one);
            (a, rig)
        };
        let at = |x: u32, y: u32| (y * W + x) as usize;
        for (reject, big_kept, small_kept) in [(0.12, false, true), (0.06, false, false), (0.25, true, true)] {
            let (a, rig) = run(reject);
            for (rect, kept) in [(big, big_kept), (small, small_kept)] {
                // x % 7 != 3: the two heads' residuals differ there, so a kept history shows.
                let (x, y) = (rect.0 + 2, rect.1 + 1);
                let l = rig.lanes(x, y);
                if kept {
                    assert_ne!(a[at(x, y)], plain[at(x, y)], "threshold {reject}: kept at ({x}, {y})");
                    assert_ne!(l[7..10], l[4..7]);
                } else {
                    assert_eq!(a[at(x, y)], plain[at(x, y)], "threshold {reject}: rejected at ({x}, {y})");
                    assert_eq!(l[7..10], l[4..7]);
                }
            }
            // Unchanged pixels keep their history.
            assert_ne!(a[at(35, 2)], plain[at(35, 2)], "threshold {reject}: an unchanged pixel");
        }
    }

    /// The cut test resets the whole frame: every pixel answers as a first frame (seed 0, no blend), and the
    /// frame after it has a history again, with seed 1.
    #[test]
    fn a_scene_cut_resets_the_history_and_the_seed() {
        let Some(mut rig) = Rig::with(Motion::Zero) else { return };
        let one = [Conditioning::default()];
        rig.write_head(head_pattern(1.0));
        rig.frame(&picture(0), &one);
        let noise0 = rig.lanes(5, 3);
        rig.frame(&picture(0), &one);
        let noise1 = rig.lanes(5, 3);
        assert_ne!(noise1[..3], noise0[..3], "seed 1");
        // Black (the picture's mean luma is far above 40 of 255). The per-pixel rejection is turned off, to
        // see the cut's own effect.
        rig.gpu.reject = 1.0;
        let cut: Vec<u32> = picture(0).iter().map(|&t| t & 0xff00_0000).collect();
        let a = rig.frame(&cut, &one);
        let Some((plain, reference)) = without_history(&cut, head_pattern(1.0), &one) else { return };
        assert_eq!(a, plain, "the cut frame answers as a first frame");
        assert_eq!(rig.lanes(5, 3), reference.lanes(5, 3), "every lane, the noise's seed 0 included");
        rig.frame(&cut, &one);
        assert_eq!(rig.lanes(5, 3)[..3], noise1[..3], "seed 1 after the cut");
        assert_ne!(rig.lanes(5, 3)[7..10], rig.lanes(5, 3)[4..7], "a history again");
    }

    /// The seed counts frames since the last reset, and a request that did not reach the network (the effect
    /// off) resets it.
    #[test]
    fn the_seed_counts_and_resets() {
        let Some(mut rig) = Rig::with(Motion::Zero) else { return };
        let one = [Conditioning::default()];
        rig.write_head(head_pattern(1.0));
        let mut noise = Vec::new();
        for _ in 0..3 {
            rig.frame(&picture(0), &one);
            noise.push(rig.lanes(5, 3)[..3].to_vec());
        }
        assert!(noise[0] != noise[1] && noise[1] != noise[2] && noise[0] != noise[2], "seeds 0, 1, 2");
        rig.gpu.history[0].skipped();
        rig.frame(&picture(0), &one);
        assert_eq!(rig.lanes(5, 3)[..3], noise[0][..], "seed 0 after the reset");
        assert_eq!(rig.lanes(5, 3)[7..10], rig.lanes(5, 3)[4..7], "and no history");
        rig.frame(&picture(0), &one);
        assert_eq!(rig.lanes(5, 3)[..3], noise[1][..], "seed 1");
        // `none` keeps the shipped fixed seed.
        let Some(mut none) = Rig::new() else { return };
        none.write_head(head_pattern(1.0));
        for _ in 0..3 {
            none.frame(&picture(0), &one);
            assert_eq!(none.lanes(5, 3)[..3], noise[0][..]);
        }
    }

    /// Every pass of a multi-pass frame keeps its own history: the last pass's lanes 7 to 9 carry its own
    /// previous output (the frame's answer), not the first pass's.
    #[test]
    fn every_pass_keeps_its_own_history() {
        let two = [Conditioning::default(), Conditioning::default()];
        let one = [Conditioning::default()];
        let Some(mut rig) = Rig::with(Motion::Zero) else { return };
        rig.write_head(head_pattern(1.0));
        let answer = rig.frame(&picture(0), &two);
        let Some(mut first_pass) = Rig::with(Motion::Zero) else { return };
        first_pass.write_head(head_pattern(1.0));
        let pass1 = first_pass.frame(&picture(0), &one);
        rig.frame(&picture(0), &two);
        let mut differ = 0;
        for (x, y) in (0..W).flat_map(|x| (0..H).map(move |y| (x, y))) {
            let l = rig.lanes(x, y);
            let (own, other) = (answer[(y * W + x) as usize], pass1[(y * W + x) as usize]);
            for c in 0..3 {
                let centred = (channel(own, c) - 0.5) * 0.125;
                assert!((l[7 + c] - centred).abs() <= 0.125 / 255.0 + 0.001, "pass 2's own history at ({x}, {y})");
            }
            differ += usize::from(own != other);
        }
        assert!(differ > (W * H) as usize / 2, "the passes' outputs differ ({differ})");
    }

    /// The S10.5 encoding of a flow vector (x low, y high).
    fn vector(x: f32, y: f32) -> u32 {
        let q = |v: f32| u32::from((v * 32.0).round() as i16 as u16);
        q(x) | (q(y) << 16)
    }

    /// `flow`: the history is reprojected by the optical flow's vectors (a synthetic field through the same
    /// shader path; lavapipe has no optical flow). A pan of 3 pixels with the matching flow keeps every pixel
    /// whose previous position is on screen, reading the first frame's output 3 pixels back; zero motion
    /// rejects them.
    #[test]
    fn the_flow_reprojects_the_history() {
        let one = [Conditioning::default()];
        // Strong horizontal structure, so a 3-pixel shift changes every pixel by far more than 0.12.
        let p: Vec<u32> = (0..W * H).map(|i| { let (x, y) = (i % W, i / W); let v = (x * 50 + y * 3) % 256; v | ((255 - v) << 8) | (((x * 13) % 256) << 16) | (0xff << 24) }).collect();
        let pan: Vec<u32> = (0..W * H).map(|i| { let (x, y) = (i % W, i / W); if x >= 3 { p[(y * W + x - 3) as usize] } else { p[(y * W) as usize] } }).collect();
        let grid = 4;
        let cells = W.div_ceil(grid) * H.div_ceil(grid);
        let Some(mut flow) = Rig::with(Motion::Flow) else { return };
        flow.gpu.synthetic_flow = Some((grid, vec![vector(-3.0, 0.0); cells as usize]));
        flow.write_head(head_pattern(1.0));
        flow.frame(&p, &one);
        flow.write_head(head_pattern(-1.0));
        let a = flow.frame(&pan, &one);
        // The reference: the same first frame, then still (its lanes 7 to 9 are the first output at each pixel).
        let Some(mut still) = Rig::with(Motion::Zero) else { return };
        still.write_head(head_pattern(1.0));
        still.frame(&p, &one);
        still.write_head(head_pattern(-1.0));
        still.frame(&p, &one);
        let Some((plain, _)) = without_history(&pan, head_pattern(-1.0), &one) else { return };
        for y in 0..H {
            for x in 0..W {
                let l = flow.lanes(x, y);
                if x >= 3 {
                    assert_eq!(l[7..10], still.lanes(x - 3, y)[7..10], "({x}, {y}) reads the output 3 pixels back");
                } else {
                    assert_eq!(l[7..10], l[4..7], "({x}, {y}): its previous position is off screen");
                    assert_eq!(a[(y * W + x) as usize], plain[(y * W + x) as usize]);
                }
            }
        }
        assert_ne!(a[(5 * W + 20) as usize], plain[(5 * W + 20) as usize], "the flow's history blends");
        // Zero motion on the same pan rejects the moved pixels.
        let Some(mut zero) = Rig::with(Motion::Zero) else { return };
        zero.write_head(head_pattern(1.0));
        zero.frame(&p, &one);
        zero.write_head(head_pattern(-1.0));
        let z = zero.frame(&pan, &one);
        assert_eq!(z[(5 * W + 20) as usize], plain[(5 * W + 20) as usize], "zero motion rejects it");
    }

    /// `flow` on a device without the optical flow (lavapipe: no extension, so `create_device` added no
    /// queue) runs as `zero`.
    #[test]
    fn flow_without_the_extension_runs_as_zero_motion() {
        let one = [Conditioning::default()];
        let run = |motion| {
            let mut rig = Rig::with(motion).unwrap();
            rig.write_head(head_pattern(1.0));
            rig.frame(&picture(0), &one);
            rig.write_head(head_pattern(-1.0));
            let a = rig.frame(&picture(0), &one);
            let unavailable = matches!(rig.gpu.flow, FlowState::Unavailable(None));
            (a, unavailable)
        };
        if Rig::new().is_none() {
            return;
        }
        let (flow, unavailable) = run(Motion::Flow);
        assert!(unavailable, "no flow device");
        assert_eq!(flow, run(Motion::Zero).0);
    }

    /// The HUD settings of a test: the detection and the regions (normalised to the test frame).
    fn hud(detect: bool, regions: &[(u32, u32, u32, u32, u32, f32)]) -> neural_forge_protocol::HudSettings {
        let regions = regions
            .iter()
            .map(|&(x, y, w, h, mode, strength)| neural_forge_protocol::HudRegion { x: x as f32 / W as f32, y: y as f32 / H as f32, w: w as f32 / W as f32, h: h as f32 / H as f32, mode, strength })
            .collect();
        neural_forge_protocol::HudSettings { detect, regions }
    }

    /// How far pixel (x, y)'s centre lies outside the rectangle (pixels, Chebyshev; 0 inside), as the shader
    /// measures it.
    fn outside(x: u32, y: u32, (x0, y0, w, h): (u32, u32, u32, u32)) -> f32 {
        let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
        let dx = (x0 as f32 - cx).max(cx - (x0 + w) as f32).max(0.0);
        let dy = (y0 as f32 - cy).max(cy - (y0 + h) as f32).max(0.0);
        dx.max(dy)
    }

    /// HUD regions at present: inside a `keep` rectangle the answer is the frame's own texel exactly (no edit),
    /// inside a `soften` one the edit is scaled by its strength, and outside the feather the answer is the one
    /// without regions bit for bit; with history the rejected history leaves no trace of the previous frame's
    /// edit under the region.
    #[test]
    fn hud_regions_keep_the_edit_off_or_scale_it_and_leave_the_rest() {
        let Some(mut rig) = Rig::new() else { return };
        let one = [Conditioning::default()];
        let p = picture(0);
        rig.write_head(head_pattern(1.0));
        rig.hud = hud(false, &[]);
        let plain = rig.frame(&p, &one);
        let keep = (8, 6, 8, 6);
        let soften = (26, 12, 8, 8);
        rig.hud = hud(false, &[(keep.0, keep.1, keep.2, keep.3, neural_forge_protocol::enums::hud_mode::KEEP, 0.0), (soften.0, soften.1, soften.2, soften.3, neural_forge_protocol::enums::hud_mode::SOFTEN, 0.5)]);
        let a = rig.frame(&p, &one);
        let feather = Hud::region_feather(H);
        let (mut edited, mut kept, mut softened) = (0, 0, 0);
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                let (dk, ds) = (outside(x, y, keep), outside(x, y, soften));
                if dk == 0.0 {
                    assert_eq!(a[i], p[i], "({x}, {y}) inside the keep region: the frame's texel");
                    kept += 1;
                    edited += usize::from(plain[i] != p[i]);
                } else if ds == 0.0 {
                    for c in 0..3 {
                        let (f, e) = (channel(p[i], c), channel(plain[i], c));
                        let want = f + 0.5 * (e - f);
                        assert!((channel(a[i], c) - want).abs() <= 1.0 / 255.0, "({x}, {y}) channel {c} softened: {} against {want}", channel(a[i], c));
                    }
                    softened += usize::from(plain[i] != p[i]);
                } else if dk >= feather && ds >= feather {
                    assert_eq!(a[i], plain[i], "({x}, {y}) outside the feather: unchanged");
                }
            }
        }
        assert_eq!(kept, 48);
        assert!(edited > 24 && softened > 32, "the test's edit reaches into both regions ({edited}, {softened})");
        // With a history the region's pixels never blend toward a history (rejected): the frame after one with
        // another edit answers there as a first frame does.
        let Some(mut hist) = Rig::with(Motion::Zero) else { return };
        hist.hud = rig.hud.clone();
        hist.write_head(head_pattern(-1.0));
        hist.frame(&p, &one);
        hist.write_head(head_pattern(1.0));
        hist.frame(&p, &one);
        for (x, y) in [(10, 8), (12, 9)] {
            let l = hist.lanes(x, y);
            assert_eq!(l[7..10], l[4..7], "({x}, {y}): no history under the region");
        }
        let l = hist.lanes(2, 20);
        assert_ne!(l[7..10], l[4..7], "a history outside it");
    }

    /// A 12x6 block of 2-pixel black and white cells (the HUD's text) over `background`.
    fn with_text(background: &[u32]) -> Vec<u32> {
        background
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let (x, y) = (i as u32 % W, i as u32 / W);
                if (14..26).contains(&x) && (2..8).contains(&y) {
                    let on = (x / 2 + y / 2) % 2 == 0;
                    (if on { 0x00ff_ffff } else { 0 }) | (t & 0xff00_0000)
                } else {
                    t
                }
            })
            .collect()
    }

    /// The HUD detection: a static, sharp block over a scene that changes everywhere is found after nine
    /// moving frames (none before: a new confidence detects nothing for eight), and from then on gets no edit,
    /// while the scene is answered bit for bit as without the detection. A fully still sequence (sharp edges
    /// everywhere, nothing moving) grows no mask.
    #[test]
    fn the_hud_detection_finds_static_text_over_a_moving_scene_and_not_in_a_still_one() {
        let Some(mut rig) = Rig::new() else { return };
        let Some(mut off) = Rig::new() else { return };
        let one = [Conditioning::default()];
        for r in [&mut rig, &mut off] {
            r.write_head(head_pattern(1.0));
        }
        rig.hud = hud(true, &[]);
        off.hud = hud(false, &[]);
        // Inside the block, away from its edge (the detection needs sharp taps two pixels apart).
        let inner: Vec<(u32, u32)> = (16..24).flat_map(|x| (4..6).map(move |y| (x, y))).collect();
        let far = |x: u32, y: u32| !(11..29).contains(&x) || !(0..11).contains(&y);
        for k in 0..12u32 {
            let p = with_text(&picture(k));
            let a = rig.frame(&p, &one);
            let b = off.frame(&p, &one);
            for y in 0..H {
                for x in 0..W {
                    if far(x, y) {
                        assert_eq!(a[(y * W + x) as usize], b[(y * W + x) as usize], "frame {k}: ({x}, {y}) is scene, answered as without the detection");
                    }
                }
            }
            let held = inner.iter().filter(|&&(x, y)| a[(y * W + x) as usize] == p[(y * W + x) as usize]).count();
            let edited = inner.iter().filter(|&&(x, y)| b[(y * W + x) as usize] != p[(y * W + x) as usize]).count();
            if k < 9 {
                assert_eq!(a, b, "frame {k}: nothing is HUD yet");
            } else {
                assert_eq!(held, inner.len(), "frame {k}: the text gets no edit");
                assert!(edited > inner.len() / 2, "frame {k}: the test's edit reaches the text ({edited})");
            }
        }
        // Still: the same picture over and over.
        let Some(mut still) = Rig::new() else { return };
        still.write_head(head_pattern(1.0));
        still.hud = hud(true, &[]);
        let p = with_text(&picture(3));
        let b = off.frame(&p, &one);
        for k in 0..20 {
            assert_eq!(still.frame(&p, &one), b, "still frame {k}: no mask grows");
        }
    }

    /// One server answers at a time; once it ends, another can take over.
    #[test]
    fn one_server_owns_the_channel_until_it_ends() {
        let owner = Owner::new();
        let first = owner.claim(1).expect("free");
        assert!(owner.claim(2).is_none(), "a second server must not answer too");
        drop(first);
        let second = owner.claim(2).expect("free again after the first ended");
        assert!(owner.claim(1).is_none());
        drop(second);
    }
}
