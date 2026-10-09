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
//! There are no motion vectors after the upscaler: every frame runs without history, with a fixed seed (a
//! seed that changed every frame, with nothing blending frames, would shimmer). It never runs the network
//! while the pre-upscaler path uses it, nor the other way round ([`super::native::Loader::claim_post`]).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;
use neural_forge_protocol::enums::proxy_format;
use neural_forge_protocol::Slot;

use super::native::{Built, Conditioning, Loader, Setup};
use super::{region_buffer, HostBuffer};
use crate::shm::ShmView;

const PREPROCESS_SPV: &[u8] = include_bytes!("../../shaders/native_post_preprocess.spv");
const COMPOSITE_SPV: &[u8] = include_bytes!("../../shaders/native_post_composite.spv");

/// The model server's smallest frame (`ngx::MIN_FEATURE_DIM`): smaller ones are echoed.
const MIN_DIM: u32 = 64;

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

/// The per-request GPU objects.
struct Gpu {
    queue: vk::Queue,
    set_layout: vk::DescriptorSetLayout,
    layouts: [vk::PipelineLayout; 2],
    pipelines: [vk::Pipeline; 2],
    pool: vk::DescriptorPool,
    sets: [vk::DescriptorSet; 2],
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Per slot: the proxy and answer buffers over the shared-memory regions, for one frame size.
    slots: [Option<SlotBuffers>; 2],
    /// What the sets point at: the slot, its buffers' size, and the build.
    bound: Option<(usize, u32, u32, u64)>,
    /// A wait that timed out: the buffers it used may still be in use, so they are never touched again.
    stalled: bool,
}

struct SlotBuffers {
    width: u32,
    height: u32,
    proxy: HostBuffer,
    answer: HostBuffer,
}

impl Gpu {
    /// # Safety
    /// `device` is live and has the queue `setup` names.
    unsafe fn new(device: &ash::Device, setup: Setup) -> Option<Self> {
        // Built piece by piece into `g`; on an early return its drop destroys whatever exists (the server is
        // started again after a failure, so a failed attempt must not leave anything behind).
        let mut g = Partial { device, gpu: Self::empty() };
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
            let bindings = [binding(0), binding(1), binding(2)];
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
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 }];
            step(6)?;
            g.gpu.pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&sizes), None).ok()?;
            let set_layouts = [set_layout, set_layout];
            step(7)?;
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(g.gpu.pool).set_layouts(&set_layouts)).ok()?;
            g.gpu.sets = [sets[0], sets[1]];
            step(8)?;
            g.gpu.command_pool = device
                .create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(setup.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .ok()?;
            step(9)?;
            g.gpu.cmd = crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::builder().command_pool(g.gpu.command_pool).command_buffer_count(1)).ok()?[0];
            step(10)?;
            g.gpu.fence = device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
        }
        Some(std::mem::replace(&mut g.gpu, Self::empty()))
    }

    /// The objects it owns that are destroyed one by one (the sets and the command buffer go with their pools).
    #[cfg(test)]
    fn live(&self) -> usize {
        use ash::vk::Handle;
        let handles = [self.set_layout.as_raw(), self.layouts[0].as_raw(), self.layouts[1].as_raw(), self.pipelines[0].as_raw(), self.pipelines[1].as_raw(), self.pool.as_raw(), self.command_pool.as_raw(), self.fence.as_raw()];
        handles.iter().filter(|&&h| h != 0).count()
    }

    /// No objects (every handle null).
    fn empty() -> Self {
        Self {
            queue: vk::Queue::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layouts: [vk::PipelineLayout::null(); 2],
            pipelines: [vk::Pipeline::null(); 2],
            pool: vk::DescriptorPool::null(),
            sets: [vk::DescriptorSet::null(); 2],
            command_pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            slots: [None, None],
            bound: None,
            stalled: false,
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
            for s in self.slots.iter_mut().filter_map(Option::take) {
                s.proxy.destroy(device);
                s.answer.destroy(device);
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

fn serve(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, loader: &Loader, view: ShmView, quit: &AtomicBool, gpu: &mut Gpu) {
    let hdr = view.header();
    let mut last = Slot::ALL.map(|s| hdr.seq_req_slot(s).load(Ordering::Acquire));
    let mut frames: u64 = 0;
    let mut last_request = Instant::now();
    while !quit.load(Ordering::Relaxed) && !gpu.stalled {
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
    let conditioning = Conditioning::from(hdr.global_tuning());
    // The claim is released whatever happens after it: a claim held while the network is not
    // built at this size would keep the pre-upscaler path off it for good.
    let evaluated = wanted && loader.claim_post() && {
        let ok = loader.ready(width, height).is_some_and(|built| {
            // SAFETY: the regions are mapped for the process's life; this thread owns the slot's request now.
            unsafe { evaluate(device, instance, physical, import, loader, view, gpu, slot, &built, width, height, format == proxy_format::BGRA8, conditioning) }
        });
        loader.release_post();
        ok
    };
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

/// One frame through the network. `false` (and the caller echoes) on anything short of a written answer.
///
/// # Safety
/// The regions are live; nothing of `gpu`'s is pending.
#[allow(clippy::too_many_arguments)]
unsafe fn evaluate(
    device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, import: bool, loader: &Loader, view: ShmView, gpu: &mut Gpu, slot: Slot, built: &Built,
    width: u32, height: u32, bgra: bool, c: Conditioning,
) -> bool {
    let s = slot.index();
    let bytes = u64::from(width) * u64::from(height) * 4;
    let (proxy_region, answer_region) = view.regions(slot);
    if gpu.slots[s].as_ref().is_none_or(|b| (b.width, b.height) != (width, height)) {
        if let Some(old) = gpu.slots[s].take() {
            // SAFETY: nothing pending (every request waits for its own work).
            unsafe {
                old.proxy.destroy(device);
                old.answer.destroy(device);
            }
            gpu.bound = None;
        }
        // SAFETY: the regions are mapped for the process's life with room for `bytes`.
        let buffers = unsafe {
            (
                region_buffer(device, instance, physical, import, proxy_region, view.capacity(), bytes),
                region_buffer(device, instance, physical, import, answer_region, view.capacity(), bytes),
            )
        };
        let (Some(proxy), Some(answer)) = buffers else { return false };
        gpu.slots[s] = Some(SlotBuffers { width, height, proxy, answer });
    }
    let Some(b) = gpu.slots[s].as_ref() else { return false };
    let f = built.frame;
    if gpu.bound != Some((s, width, height, built.generation)) {
        let info = |buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (proxy_i, feat_i, head_i, ans_i) = (info(b.proxy.buffer), info(f.features), info(f.head), info(b.answer.buffer));
        let w = |set, n, i: &[vk::DescriptorBufferInfo; 1]| vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(n).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(i).build();
        let writes = [
            w(gpu.sets[0], 0, &proxy_i),
            w(gpu.sets[0], 1, &feat_i),
            w(gpu.sets[1], 0, &proxy_i),
            w(gpu.sets[1], 1, &head_i),
            w(gpu.sets[1], 2, &ans_i),
        ];
        // SAFETY: the sets are not in use; every buffer is live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        gpu.bound = Some((s, width, height, built.generation));
    }
    if b.proxy.staged {
        // SAFETY: both hold `bytes`; the staging buffer is host-coherent and not in use.
        unsafe { std::ptr::copy_nonoverlapping(proxy_region, b.proxy.ptr, bytes as usize) };
    }
    let pre = PreprocessPush {
        dims: [f.field_width, f.field_height, width, height],
        frame: [0, u32::from(bgra), 0, 0],
        control: [c.style as f32, c.local_tone, c.local_structure, c.skin_structure],
        more: [if c.auto_mask { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
    };
    let post = CompositePush { dims: pre.dims, frame: pre.frame, control: [c.style as f32, c.local_tone, c.intensity, 0.0] };
    let barrier = |src: vk::AccessFlags, dst: vk::AccessFlags| [vk::MemoryBarrier::builder().src_access_mask(src).dst_access_mask(dst).build()];
    let cmd = gpu.cmd;
    // SAFETY: the layer's own buffer, not pending; every handle is live.
    let recorded: ash::prelude::VkResult<()> = unsafe {
        (|| {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gpu.pipelines[0]);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.layouts[0], 0, &[gpu.sets[0]], &[]);
            device.cmd_push_constants(cmd, gpu.layouts[0], vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&pre).cast::<u8>(), std::mem::size_of::<PreprocessPush>()));
            device.cmd_dispatch(cmd, f.field_width.div_ceil(8), f.field_height.div_ceil(8), 1);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE), &[], &[]);
            device.cmd_execute_commands(cmd, &[built.graph_own]);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ), &[], &[]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, gpu.pipelines[1]);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.layouts[1], 0, &[gpu.sets[1]], &[]);
            device.cmd_push_constants(cmd, gpu.layouts[1], vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(std::ptr::from_ref(&post).cast::<u8>(), std::mem::size_of::<CompositePush>()));
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::HOST_READ), &[], &[]);
            device.end_command_buffer(cmd)
        })()
    };
    if recorded.is_err() {
        return false;
    }
    // SAFETY: the fence is the layer's own and idle; the queue is this thread's alone.
    let submitted = unsafe {
        device.reset_fences(&[gpu.fence]).and_then(|()| device.queue_submit(gpu.queue, &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()], gpu.fence))
    };
    if crate::note_vk(submitted).is_err() {
        return false;
    }
    // SAFETY: the fence was just submitted.
    let waited = unsafe { device.wait_for_fences(&[gpu.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) };
    if crate::note_fence_wait(waited, "native::post").is_err() {
        gpu.stalled = true;
        crate::log!("[native] the after-the-upscaler network frame did not finish in time; the after-the-upscaler path is not served from here on");
        return false;
    }
    if let Some((waits, at)) = loader.chain_timeouts().filter(|(n, _)| *n > 0) {
        crate::log!("[native] {waits} counter-chain wait(s) gave up ({at}); rebuilding the network with barriers between its launches");
        loader.fall_back();
        return false;
    }
    if b.answer.staged {
        // SAFETY: both hold `bytes`; the work that wrote the staging buffer completed.
        unsafe { std::ptr::copy_nonoverlapping(b.answer.ptr, answer_region, bytes as usize) };
    }
    true
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
        let setup = Setup { gipa, instance: vk::Instance::null(), physical, frame_family: family, family, index: 0, post_index: 0 };
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
