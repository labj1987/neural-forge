//! The before-DLSS placement on a second GPU: the network's frame on the other device ([`PreRunner`], the
//! [`Runner`] of `super::offload_pre`'s worker).
//!
//! What the hold on the game's GPU sent (the frame's encode, the accumulated motion, the depth, the exposure and
//! the scene cut's word) goes into a native pass of this device's own, built as the game's device builds its
//! (`NativePass`: the model resolution, the control mask, the colour of the edit), with the network of a second
//! loader on that device (barriers, never counter chaining: `super::native_post`). One network frame, as the split
//! hold submits it (`N[parity]` on the frames' queue), and the answer copied into the answer region. What the CPU
//! decides per frame on the game's device (the history, the jitter, the passes' settings) comes in a queue keyed
//! by the frame's sequence number ([`Metas`]).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;

use super::native::{Graph, Loader, NativePass, Passes};
use super::offload_pre::{Layout, Ran, Runner};
use super::HostBuffer;

/// What the hold decided for the frame it sent as `seq`.
#[derive(Clone)]
pub(crate) struct FrameMeta {
    pub seq: u32,
    /// The render extent and each pass slot's extent (`native::pass_extents`).
    pub width: u32,
    pub height: u32,
    pub extents: [(u32, u32); neural_forge_protocol::MAX_PASSES],
    /// The sizes the network is built for, and the Network setting.
    pub sizes: Vec<(u32, u32)>,
    pub choice: u32,
    /// The hold had motion vectors, and nothing on the game's side asks for a history reset.
    pub history: bool,
    pub jitter: Option<[f32; 2]>,
    pub mvec_scale: [f32; 2],
    pub passes: Passes,
    /// The hold sent DLSS's depth (the control mask's sky rule reads it).
    pub depth: bool,
    pub lut: Option<Arc<super::colour::Lut>>,
}

/// The frames' parameters, from the hold to the worker.
pub(crate) type Metas = Arc<Mutex<VecDeque<FrameMeta>>>;

/// The pass's resources on the other device, for one render extent.
struct Frame {
    width: u32,
    height: u32,
    extents: [(u32, u32); neural_forge_protocol::MAX_PASSES],
    hdr: super::hdr::HdrPass,
    native: NativePass,
    /// DLSS's depth as the hold sent it: only for the hold before DLSS (after DLSS's output there is none).
    depth: Option<super::native::DepthCopy>,
}

pub(crate) struct PreRunner {
    game: vk::Device,
    instance: ash::Instance,
    off: Option<super::offload::Offload>,
    loader: Option<Arc<Loader>>,
    metas: Metas,
    queue: vk::Queue,
    pool: vk::CommandPool,
    /// The copies in, and the answer's copy out.
    cmds: [vk::CommandBuffer; 2],
    fence: vk::Fence,
    /// The regions on this device (imported, or staged copies of them), made on the first frame.
    regions: Option<(HostBuffer, HostBuffer)>,
    /// Both devices can import the regions' memory, and the regions' whole sizes.
    shared: bool,
    capacity: (usize, usize),
    frame: Option<Frame>,
    rule: super::offload::ServerRule,
    /// A frame did not finish: nothing of the device is freed.
    leaked: bool,
    said_loading: bool,
    /// Frames asked of the runner so far.
    frames: u64,
    /// The hold it serves (the log's words).
    at: super::offload_pre::Where,
    /// The pass count last said in the log (more than one pass asked).
    said_more: Option<usize>,
}

/// The passes a frame runs on the other device: every pass of the stage the hold sent (`asked`) when the later passes'
/// buffers are built there (`more`), else pass 1 alone (`native::runnable`, as on the game's GPU).
fn server_passes(asked: Passes, more: bool) -> Passes {
    super::native::runnable(asked, more)
}

/// One frame's buffers on the other device, in submission order: the copies in (`up`), pass 1 (`frame`: its first
/// frame's buffer and its parity's), the later passes with the copies between them (`later`:
/// `NativePass::later_sequence`), the answer's copy out (`down`).
fn frame_submits(
    up: vk::CommandBuffer,
    frame: &[Option<vk::CommandBuffer>; 2],
    later: &[vk::CommandBuffer],
    down: vk::CommandBuffer,
) -> Vec<vk::CommandBuffer> {
    let mut all = vec![up];
    all.extend(frame.iter().flatten().copied());
    all.extend_from_slice(later);
    all.push(down);
    all
}

// SAFETY: plain handles of the offload device and its own mappings, used on the worker's thread only.
unsafe impl Send for PreRunner {}

/// `NEURAL_FORGE_OFFLOAD_TEST_FAIL=<frame>` (a test switch): that frame of a hold's offload (before DLSS or after its output) fails on the other
/// GPU, so the fall-back to the game's GPU can be seen in a game.
fn test_fail_at() -> Option<u64> {
    static AT: std::sync::LazyLock<Option<u64>> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TEST_FAIL").and_then(|v| v.trim().parse().ok()).filter(|n| *n > 0));
    *AT
}

impl PreRunner {
    /// On the offload device made with the game's device `game` (`offload::take`): its loader, queue and command
    /// buffers. `Err` with why it cannot serve (the device is given back).
    pub(crate) fn start(
        instance: ash::Instance,
        game: vk::Device,
        shared: bool,
        capacity: (usize, usize),
        metas: Metas,
        at: super::offload_pre::Where,
    ) -> Result<(Self, String), String> {
        let off = match super::offload::take(game) {
            Some(Ok(off)) => off,
            Some(Err(why)) => return Err(why),
            None => return Err("no device was made for it with the game's device (or the finished-frame placement has it)".into()),
        };
        if off.standin {
            let why = "NEURAL_FORGE_OFFLOAD_STANDIN is a test of the finished-frame placement only".to_string();
            super::offload::give_back(game, off, false);
            return Err(why);
        }
        let name = off.name.clone();
        crate::log!("[offload] {}", off.plan);
        crate::log!(
            "[offload] {}: each frame's edit comes from the other GPU and is applied one frame later, carried by DLSS's motion vectors; {}",
            at.name(),
            if at == super::offload_pre::Where::Before { "nothing DLSS reads is delayed" } else { "DLSS's output is not delayed" }
        );
        crate::logging::flush();
        let device = &off.device;
        let setup = off.setup;
        // SAFETY: the device just made for the offload; its queue family and index are its setup's.
        let made = unsafe {
            let queue = device.get_device_queue(setup.family, setup.post_index);
            (|| {
                crate::loader_data::initialize_queue(device.handle(), queue).ok()?;
                let pool = device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(setup.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .ok()?;
                let cmds = crate::loader_data::allocate_commands(device, &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2));
                let fence = device.create_fence(&vk::FenceCreateInfo::default(), None);
                match (cmds, fence) {
                    (Ok(cmds), Ok(fence)) => Some((queue, pool, [cmds[0], cmds[1]], fence)),
                    (_, fence) => {
                        if let Ok(f) = fence {
                            device.destroy_fence(f, None);
                        }
                        device.destroy_command_pool(pool, None);
                        None
                    }
                }
            })()
        };
        let Some((queue, pool, cmds, fence)) = made else {
            super::offload::give_back(game, off, false);
            return Err("its command buffers could not be made on the other GPU".into());
        };
        // Barriers between the network's launches, never counter chaining (`native_post::PostServer::start_offload`).
        let loader = Arc::new(Loader::start(off.device.handle(), off.setup, super::native::HoldCounts::default(), false));
        loader.register(off.device.handle());
        let shared = shared && off.import;
        Ok((
            Self {
                game,
                instance,
                off: Some(off),
                loader: Some(loader),
                metas,
                queue,
                pool,
                cmds,
                fence,
                regions: None,
                shared,
                capacity,
                frame: None,
                rule: super::offload::ServerRule::default(),
                leaked: false,
                said_loading: false,
                frames: 0,
                at,
                said_more: None,
            },
            name,
        ))
    }

    /// One frame; `Err` with [`Ran::Skipped`] or [`Ran::Failed`].
    ///
    /// # Safety
    /// As [`Runner::run`].
    unsafe fn frame(&mut self, meta: &FrameMeta, out: *const u8, inn: *mut u8, layout: &Layout) -> Result<(), Ran> {
        let failed = |why: &str| Ran::Failed(why.to_string());
        let (Some(off), Some(loader)) = (self.off.as_ref(), self.loader.as_ref()) else { return Err(failed("the other GPU's device is gone")) };
        let (device, instance, physical, setup) = (&off.device, &self.instance, off.physical, off.setup);
        if loader.dead() {
            return Err(failed("the other GPU's device was lost"));
        }
        // The regions on this device.
        if self.regions.is_none() {
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            // SAFETY: the regions outlive the runner (the worker is stopped before the exchange is destroyed).
            let o = unsafe { super::region_buffer(device, instance, physical, self.shared, out.cast_mut(), self.capacity.0, layout.out_bytes, usage) };
            let i = unsafe { super::region_buffer(device, instance, physical, self.shared, inn, self.capacity.1, layout.in_bytes, usage) };
            match (o, i) {
                (Some(o), Some(i)) => {
                    crate::log!(
                        "[offload] {}: the frame crosses system memory, {} on the other GPU ({:.1} MB out, {:.1} MB back per frame)",
                        self.at.name(),
                        if o.staged || i.staged { "copied once more there" } else { "imported there without a copy" },
                        layout.out_bytes as f64 / 1048576.0,
                        layout.in_bytes as f64 / 1048576.0
                    );
                    self.regions = Some((o, i));
                }
                (o, i) => {
                    // SAFETY: just made, never used.
                    unsafe {
                        if let Some(b) = o {
                            b.destroy(device);
                        }
                        if let Some(b) = i {
                            b.destroy(device);
                        }
                    }
                    return Err(failed("the frame's regions could not be made on the other GPU"));
                }
            }
        }
        // The pass for this extent and these pass extents.
        if self.frame.as_ref().is_some_and(|f| (f.width, f.height, f.extents) != (meta.width, meta.height, meta.extents)) {
            if let Some(f) = self.frame.take() {
                // SAFETY: idle: every frame is waited for before `run` returns.
                unsafe {
                    f.native.destroy(device);
                    f.hdr.destroy(device);
                    if let Some(d) = f.depth {
                        d.destroy(device);
                    }
                }
            }
        }
        if self.frame.is_none() {
            // SAFETY: a live device; the frames' family is the setup's.
            let built = unsafe {
                let hdr = super::hdr::HdrPass::build(device, instance, physical, meta.width, meta.height);
                let native = NativePass::build_at(device, instance, physical, setup.family, meta.width, meta.height, meta.extents, false, false);
                // No depth crosses for the hold after DLSS's output: no buffer for it there.
                let wants_depth = layout.sizes.depth > 0;
                let depth =
                    if wants_depth { super::native::DepthCopy::build(device, instance, physical, setup.family, meta.width, meta.height, false) } else { None };
                match (hdr, native, depth) {
                    (Some(hdr), Some(native), depth) if depth.is_some() == wants_depth => {
                        Some(Frame { width: meta.width, height: meta.height, extents: meta.extents, hdr, native, depth })
                    }
                    (hdr, native, depth) => {
                        if let Some(x) = hdr {
                            x.destroy(device);
                        }
                        if let Some(x) = native {
                            x.destroy(device);
                        }
                        if let Some(x) = depth {
                            x.destroy(device);
                        }
                        None
                    }
                }
            };
            self.frame = built;
        }
        let Some(f) = self.frame.as_mut() else { return Err(failed("the network's frame resources could not be built on the other GPU")) };
        // The network, built for the frame's sizes.
        if !loader.claim_pre() {
            return Err(Ran::Skipped);
        }
        let Some(builts) = loader.ready_sizes(&meta.sizes, meta.choice, 0) else {
            if !std::mem::replace(&mut self.said_loading, true) {
                crate::log!("[offload] {}: the network is loading on the other GPU; the frames go on as they are until it runs", self.at.name());
                crate::logging::flush();
            }
            let why = loader.failed().or_else(|| loader.paused().then(|| "the VRAM guard paused it".to_string()));
            return Err(match self.rule.missed(Instant::now(), why.as_deref()) {
                Some(why) => Ran::Failed(why),
                None => Ran::Skipped,
            });
        };
        self.said_loading = false;
        let Some(own) = meta.sizes.iter().position(|s| *s == meta.extents[0]) else { return Err(failed("the network's sizes do not include the frame's")) };
        let built = builts[own];
        // Its own compute family: the graph recorded for it.
        let graphs: Vec<Graph> = builts.iter().map(|&b| Graph { built: b, graph: b.graph_own }).collect();
        // Every pass of the stage there, as on the game's GPU: the later passes' buffers built once (a failure runs one).
        // SAFETY: nothing of the pass is pending (every frame is waited for); the frames' family is the setup's.
        let more = meta.passes.count > 1 && unsafe { f.native.ensure_more(device, instance, physical, setup.family, meta.passes.count) };
        let passes = server_passes(meta.passes, more);
        if meta.passes.count > 1 && self.said_more != Some(passes.count) {
            if more {
                crate::log!("[offload] {}: all {} passes of the stage run on the other GPU", self.at.name(), passes.count);
            } else {
                crate::log!("[offload] {}: the later passes' buffers could not be made on the other GPU: it runs pass 1 alone", self.at.name());
            }
        }
        self.said_more = (meta.passes.count > 1).then_some(passes.count);
        let depth = f.depth.as_ref().filter(|_| meta.depth && layout.sizes.depth > 0).map(super::native::DepthCopy::buffer);
        // SAFETY: nothing of the pass is pending (every frame is waited for); every handle is this device's.
        unsafe {
            if !f.native.ensure_lanes(device, instance, physical, passes.count) {
                return Err(failed("the model resolution's buffers could not be built on the other GPU"));
            }
            f.native.prepare_colour(device, instance, physical, &passes, meta.lut.as_ref());
            if !f.native.record_graphs(device, instance, physical, &graphs, f.hdr.encoded_view(), f.hdr.exposure_buffer(), depth) {
                return Err(failed("recording the network's frame failed on the other GPU"));
            }
        }
        let history = f.native.begin_frame(built.network, meta.history);
        let (frame_cmds, later) = f.native.frame_graphs(&graphs, history, meta.jitter, meta.mvec_scale, &passes);
        let later = if more { later } else { Vec::new() };
        let sequence = f.native.later_sequence(&later);
        let (o, i) = self.regions.as_ref().expect("made above");
        let texels = u64::from(meta.width) * u64::from(meta.height);
        // SAFETY: the regions as the contract; the device's own command buffers, not pending.
        unsafe {
            if o.staged {
                std::ptr::copy_nonoverlapping(out, o.ptr, layout.out_bytes as usize);
            }
            // The exposure the encode used, and the scene cut's word.
            let side = out.add(layout.side_at as usize);
            f.hdr.set_exposure_bytes(std::slice::from_raw_parts(side, super::hdr::HdrPass::EXPOSURE_BYTES as usize));
            let [up, down] = self.cmds;
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let recorded = (|| -> ash::prelude::VkResult<()> {
                device.reset_command_buffer(up, vk::CommandBufferResetFlags::empty())?;
                device.begin_command_buffer(up, &begin)?;
                let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
                let to_general = vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(f.hdr.encoded_image())
                    .subresource_range(range);
                let host = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    up,
                    vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[host],
                    &[],
                    &[to_general],
                );
                let region = vk::BufferImageCopy { buffer_offset: layout.frame_at, ..f.hdr.padded_region() };
                device.cmd_copy_buffer_to_image(up, o.buffer, f.hdr.encoded_image(), vk::ImageLayout::GENERAL, &[region]);
                let copy = |dst: vk::Buffer, at: u64, to: u64, size: u64| {
                    device.cmd_copy_buffer(up, o.buffer, dst, &[vk::BufferCopy { src_offset: at, dst_offset: to, size }])
                };
                copy(f.native.acc.buffer, layout.motion_at, 0, (texels * 8).min(layout.sizes.motion));
                if let Some(depth) = depth {
                    copy(depth, layout.depth_at, 0, (texels * 4).min(layout.sizes.depth));
                }
                // The scene cut the game's GPU decided in its capture: the state's third word.
                copy(f.native.state_buffer(), layout.side_at + super::hdr::HdrPass::EXPOSURE_BYTES + 8, 8, 4);
                let copied = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ);
                device.cmd_pipeline_barrier(
                    up,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[copied],
                    &[],
                    &[],
                );
                device.end_command_buffer(up)?;
                device.reset_command_buffer(down, vk::CommandBufferResetFlags::empty())?;
                device.begin_command_buffer(down, &begin)?;
                let done = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                device.cmd_pipeline_barrier(
                    down,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[done],
                    &[],
                    &[],
                );
                device.cmd_copy_buffer(
                    down,
                    f.native.answer(),
                    i.buffer,
                    &[vk::BufferCopy { src_offset: 0, dst_offset: layout.answer_at, size: layout.sizes.answer }],
                );
                let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                device.cmd_pipeline_barrier(
                    down,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[to_host],
                    &[],
                    &[],
                );
                device.end_command_buffer(down)
            })();
            if recorded.is_err() {
                return Err(failed("recording the frame's copies failed on the other GPU"));
            }
            let all = frame_submits(up, &frame_cmds, &sequence, down);
            let started = Instant::now();
            if crate::note_vk(device.handle(), device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&all)], self.fence)).is_err() {
                f.native.gap.skipped();
                return Err(failed("the frame's submit failed on the other GPU"));
            }
            for cmd in frame_cmds.into_iter().flatten() {
                f.native.submitted(cmd);
            }
            f.native.later_submitted(later.len());
            loader.pre_submitted(device, self.fence);
            let waited = device.wait_for_fences(&[self.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64);
            if waited.is_err() {
                // Still running (or the device is lost): nothing it uses is freed (drain before free).
                self.leaked = true;
                return Err(failed("a frame on the other GPU did not finish"));
            }
            loader.pre_idle(self.fence);
            let _ = device.reset_fences(&[self.fence]);
            let took = started.elapsed();
            super::offload::STATS.server_frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            super::offload::STATS.server_us.fetch_add(took.as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            if i.staged {
                std::ptr::copy_nonoverlapping(i.ptr.add(layout.answer_at as usize), inn.add(layout.answer_at as usize), layout.sizes.answer as usize);
            }
            if let Some(why) = self.rule.ran(built.generation, took) {
                return Err(Ran::Failed(why));
            }
        }
        Ok(())
    }
}

impl Runner for PreRunner {
    unsafe fn run(&mut self, seq: u32, out: *const u8, inn: *mut u8, layout: &Layout) -> Ran {
        // This frame's parameters; older ones belong to frames that never ran.
        let meta = {
            let mut q = self.metas.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while q.front().is_some_and(|m| m.seq != seq) && q.iter().any(|m| m.seq == seq) {
                q.pop_front();
            }
            match q.front() {
                Some(m) if m.seq == seq => q.pop_front(),
                _ => None,
            }
        };
        let Some(meta) = meta else { return Ran::Skipped };
        // Tests: the frame with this number (from 1) fails, as a frame that cannot run there does.
        self.frames += 1;
        if test_fail_at() == Some(self.frames) {
            return Ran::Failed("the test's failure (NEURAL_FORGE_OFFLOAD_TEST_FAIL)".into());
        }
        // SAFETY: the contract.
        match unsafe { self.frame(&meta, out, inn, layout) } {
            Ok(()) => Ran::Answered,
            Err(ran) => {
                if let Some(f) = self.frame.as_mut() {
                    f.native.gap.skipped();
                }
                ran
            }
        }
    }

    fn abandoned(&mut self) {
        if let Some(f) = self.frame.as_mut() {
            f.native.gap.skipped();
        }
    }

    fn stuck(&self) -> bool {
        self.leaked
    }

    fn gave_up(&mut self) {
        // The process's one network goes back to the game's device: it is closed here (the loader's own drop does
        // it; retiring alone leaves it open, and the game's loader then finds it open elsewhere for good). Not after
        // a frame that never finished: nothing it uses is freed.
        if self.leaked {
            return;
        }
        if let (Some(off), Some(f)) = (self.off.as_ref(), self.frame.take()) {
            // SAFETY: idle: every frame was waited for.
            unsafe {
                f.native.destroy(&off.device);
                f.hdr.destroy(&off.device);
                if let Some(d) = f.depth {
                    d.destroy(&off.device);
                }
            }
        }
        if let Some(loader) = self.loader.take() {
            loader.retire();
            loader.stop_worker();
            drop(loader);
            crate::log!("[offload] {}: the network is closed on the other GPU; the game's GPU loads it", self.at.name());
            crate::logging::flush();
        }
    }
}

impl Drop for PreRunner {
    fn drop(&mut self) {
        let Some(off) = self.off.take() else { return };
        let loader = self.loader.take();
        if let Some(l) = loader.as_ref() {
            l.retire();
            l.stop_worker();
        }
        if self.leaked {
            crate::log!("[offload] the other GPU's last frame never finished: its network and device are left to the process's end");
            crate::logging::flush();
            if let Some(l) = loader {
                std::mem::forget(l);
            }
            super::offload::give_back(self.game, off, true);
            return;
        }
        // SAFETY: idle: every frame was waited for.
        unsafe {
            let device = &off.device;
            if let Some(f) = self.frame.take() {
                f.native.destroy(device);
                f.hdr.destroy(device);
                if let Some(d) = f.depth {
                    d.destroy(device);
                }
            }
            if let Some((o, i)) = self.regions.take() {
                o.destroy(device);
                i.destroy(device);
            }
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
        }
        drop(loader);
        // Destroyed with the game's device (inside its vkDestroyDevice, the loader's lock held).
        super::offload::give_back(self.game, off, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pass of the stage the hold sent runs on the other GPU once the later passes' buffers are there (#63: only
    /// pass 1 ran there, the others were dropped); without them pass 1 alone.
    #[test]
    fn the_other_gpu_runs_every_pass_of_the_stage() {
        let asked = Passes { count: 3, ..Passes::default() };
        assert_eq!(server_passes(asked, true).count, 3);
        assert_eq!(server_passes(asked, false).count, 1);
        assert_eq!(server_passes(Passes::default(), false).count, 1);
    }

    /// The later passes run after pass 1 and before the answer is copied out.
    #[test]
    fn the_answer_is_copied_out_after_the_last_pass() {
        let cb = |n: u64| <vk::CommandBuffer as ash::vk::Handle>::from_raw(n);
        let (up, down) = (cb(1), cb(9));
        assert_eq!(frame_submits(up, &[None, Some(cb(2))], &[], down), vec![up, cb(2), down]);
        assert_eq!(frame_submits(up, &[Some(cb(3)), Some(cb(2))], &[cb(5), cb(6), cb(7)], down), vec![up, cb(3), cb(2), cb(5), cb(6), cb(7), down]);
    }
}
