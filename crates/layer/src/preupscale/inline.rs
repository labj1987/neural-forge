//! The hold inside DLSS's own command buffer (2.0.4).
//!
//! The split hold (`super::submit_around`) runs the model in front of DLSS's launch buffer. That is
//! only correct when nothing in the buffer before DLSS's first colour launch produces or
//! synchronizes the colour input. Crimson Desert and Cyberpunk 2077 render the frame in the same
//! buffer (draws, dispatches and write barriers before DLSS's first launch), so the split is
//! refused there ([`super::Hazard`]). This module holds at the launch itself instead:
//!
//! - At record time ([`Inline::record`]), right before the buffer's first launch that names the
//!   colour input, the layer records into the application's buffer: the colour input copied into a
//!   layer-owned staging image, event `captured` set, a wait on event `release` (set by the host),
//!   the staging image copied back over the colour input, and both events reset. Only copies,
//!   barriers and event commands: nothing that binds pipelines or descriptors, so the
//!   application's (vkd3d-proton's) cached state is untouched.
//! - At the submit ([`Inline::jobs_for`], [`Inline::dispatch`]), each submitted buffer carrying
//!   such a hold becomes a job for this device's worker thread.
//! - The worker waits for `captured`, runs the ordinary hold ([`super::run_hold`]) on the staging
//!   image on the layer's own queue (`side queue`, added at device creation in the same family),
//!   and then sets `release`, always: late, failed or skipped holds leave the staging image as it
//!   was copied, so the copy back changes nothing. The GPU never waits longer than the hold's own
//!   bounds.
//!
//! A buffer the split can take (GTA V's) gets nothing recorded: [`super::Tracker::inline_point`]
//! only answers where the split would be refused.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use ash::vk;

use super::{Aux, InlinePoint};

/// Staging slots per device. One per command buffer recorded with a hold and not yet re-recorded,
/// so a few frames recorded ahead each have their own image and events.
pub(crate) const MAX_SLOTS: usize = 8;

/// A slot whose buffer has not been submitted for this long is taken back (its buffer's pool was
/// reset or destroyed without `vkBeginCommandBuffer`/`vkFreeCommandBuffers` reaching the layer).
const STALE: Duration = Duration::from_secs(10);

/// `NEURAL_FORGE_INLINE=off` turns the hold inside DLSS's buffer off (no side queue is added to
/// the device either); the split hold is unaffected.
pub(crate) fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("NEURAL_FORGE_INLINE").map_or(true, |v| v != "off" && v != "0"));
    *ON
}

/// `NEURAL_FORGE_INLINE=release` (diagnostics): record the hold into DLSS's buffers as usual, but
/// release every one as soon as `captured` is set, without running the hold.
fn release_only() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("NEURAL_FORGE_INLINE").is_ok_and(|v| v == "release"));
    *ON
}

/// One staging image (and a 1x1 copy of DLSS's exposure input) and its two events.
struct Slot {
    image: vk::Image,
    memory: vk::DeviceMemory,
    exposure: vk::Image,
    exposure_memory: vk::DeviceMemory,
    extent: (u32, u32),
    captured: vk::Event,
    release: vk::Event,
    owner: Option<vk::CommandBuffer>,
    point: Option<InlinePoint>,
    used: Instant,
}

/// One execution of a buffer carrying a hold: what the worker needs.
pub(crate) struct Job {
    pub image: vk::Image,
    pub point: InlinePoint,
    /// The copy of DLSS's exposure input the buffer made beside the colour input, for the encode
    /// (`None`: the exposure is measured from the frame).
    pub exposure: Option<Aux>,
    /// The queue family of the application's queue the buffer was submitted on.
    pub family: u32,
    /// The layer's queue's family, where the hold's resources live.
    pub side_family: u32,
    captured: vk::Event,
    release: vk::Event,
}

#[derive(Default)]
struct Slots {
    slots: Vec<Slot>,
    by_cb: HashMap<vk::CommandBuffer, usize>,
    simultaneous: HashSet<vk::CommandBuffer>,
}

/// What runs the hold for a job on the worker thread (`device.rs`): the device's state and the
/// helper. It is only called once `captured` is set; `release` is set by the worker afterwards.
pub(crate) type HoldFn = Box<dyn FnMut(&Job, vk::Queue) + Send>;

pub(crate) struct Inline {
    device: Arc<ash::Device>,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// The game's graphics family (where DLSS's buffers are submitted) and the layer's queue's.
    app_family: u32,
    side_family: u32,
    slots: Mutex<Slots>,
    jobs: Mutex<Option<mpsc::Sender<Job>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    stopping: Arc<AtomicBool>,
    said: Mutex<HashSet<String>>,
}

static INLINES: LazyLock<Mutex<HashMap<vk::Device, Arc<Inline>>>> = LazyLock::new(Default::default);

impl Inline {
    /// Starts the worker for `device`'s side queue (`queue` in `family`).
    /// Starts the worker for `device`'s side `queue` in `family`; DLSS's buffers run on `app_family`.
    pub(crate) fn start(
        device: Arc<ash::Device>, instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue: vk::Queue, family: u32, app_family: u32,
        mut hold: HoldFn,
    ) -> Option<Arc<Self>> {
        // SAFETY: `physical_device` is the device's own.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (tx, rx) = mpsc::channel::<Job>();
        let stopping = Arc::new(AtomicBool::new(false));
        let inline = Arc::new(Self {
            device: device.clone(),
            memory,
            app_family,
            side_family: family,
            slots: Mutex::new(Slots::default()),
            jobs: Mutex::new(Some(tx)),
            worker: Mutex::new(None),
            stopping: stopping.clone(),
            said: Mutex::new(HashSet::new()),
        });
        let worker = std::thread::Builder::new()
            .name("nf-inline-hold".into())
            .spawn(move || {
                for job in rx {
                    if !stopping.load(Ordering::Relaxed) && job.family == app_family && wait_set(&device, job.captured, super::FRAME_CAPTURE_WAIT, &stopping) && !release_only() {
                        hold(&job, queue);
                    }
                    // Always: the GPU waits on it. A hold that did not run or wrote nothing left the
                    // staging image as the buffer copied it, so the copy back changes nothing.
                    // SAFETY: the layer's own event, alive until `destroy`, which joins this thread
                    // first.
                    let _ = unsafe { device.set_event(job.release) };
                }
            })
            .ok()?;
        *inline.worker.lock().unwrap() = Some(worker);
        INLINES.lock().unwrap().insert(inline.device.handle(), inline.clone());
        crate::log!("[preupscale] hold inside DLSS's command buffer available: side queue in family {family}, DLSS's in {app_family} (NEURAL_FORGE_INLINE=off turns it off)");
        crate::logging::flush();
        Some(inline)
    }

    fn say_once(&self, what: String) {
        if self.said.lock().unwrap().insert(what.clone()) {
            crate::log!("[preupscale] {what}");
            crate::logging::flush();
        }
    }

    /// `vkBeginCommandBuffer`: the buffer is not pending, so a slot it held is free again.
    pub(crate) fn begin(&self, cb: vk::CommandBuffer, flags: vk::CommandBufferUsageFlags) {
        let mut s = self.slots.lock().unwrap();
        release(&mut s, cb);
        if flags.contains(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE) {
            s.simultaneous.insert(cb);
        } else {
            s.simultaneous.remove(&cb);
        }
    }

    /// `vkFreeCommandBuffers`.
    pub(crate) fn free(&self, cbs: &[vk::CommandBuffer]) {
        let mut s = self.slots.lock().unwrap();
        for cb in cbs {
            release(&mut s, *cb);
            s.simultaneous.remove(cb);
        }
    }

    /// Whether `cb` carries a hold (the split path leaves it alone).
    pub(crate) fn owns(&self, cb: vk::CommandBuffer) -> bool {
        self.slots.lock().unwrap().by_cb.contains_key(&cb)
    }

    /// Records the hold into `cb` right before the launch being recorded (the caller's hook returns
    /// `Unhandled`, so the launch follows these commands). `false` when it could not: a buffer that
    /// may be pending more than once at a time, no free slot, or a slot that could not be built.
    pub(crate) fn record(&self, cb: vk::CommandBuffer, point: InlinePoint) -> bool {
        let mut s = self.slots.lock().unwrap();
        if s.simultaneous.contains(&cb) {
            self.say_once("a DLSS buffer recorded with SIMULTANEOUS_USE is not held inside".into());
            return false;
        }
        if s.by_cb.contains_key(&cb) {
            return false;
        }
        let extent = (point.desc.width, point.desc.height);
        let now = Instant::now();
        let free = s.slots.iter().position(|sl| sl.owner.is_none()).or_else(|| s.slots.iter().position(|sl| now.duration_since(sl.used) > STALE));
        let index = match free {
            Some(i) => i,
            None if s.slots.len() < MAX_SLOTS => {
                // SAFETY: the device is live.
                let Some(slot) = (unsafe { self.build_slot(extent) }) else {
                    self.say_once("a staging image for the hold inside DLSS's buffer could not be built; not holding there".into());
                    return false;
                };
                s.slots.push(slot);
                s.slots.len() - 1
            }
            None => {
                self.say_once(format!("all {MAX_SLOTS} staging slots of the hold inside DLSS's buffer are in use; that buffer is not held"));
                return false;
            }
        };
        if let Some(old) = s.slots[index].owner.take() {
            s.by_cb.remove(&old);
        }
        if s.slots[index].extent != extent {
            // SAFETY: the slot is not owned by any buffer that could still be pending (`begin`
            // freed it, or it went unsubmitted for `STALE`); the image is rebuilt at the new size.
            unsafe { self.resize_slot(&mut s.slots[index], extent) };
            if s.slots[index].image == vk::Image::null() {
                self.say_once(format!("a staging image of {}x{} could not be built; that buffer is not held inside", extent.0, extent.1));
                return false;
            }
        }
        let slot = &mut s.slots[index];
        // DLSS's exposure input, copied beside the colour input when it can be read from where it is
        // (the layer's queue cannot use the game's image: another family owns it).
        let exposure = point
            .exposure_input
            .filter(|a| a.readable && a.format == vk::Format::R16_SFLOAT && matches!(a.layout, Some(vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL)))
            .map(|a| (a.image, a.layout.unwrap_or(vk::ImageLayout::GENERAL)));
        // SAFETY: recording into the application's buffer, which it is recording on this thread
        // right now (the hook is inside its `vkCmdCuLaunchKernelNVX`); every handle is live.
        unsafe {
            record_hold(&self.device, cb, point.colour, slot.image, extent, exposure.map(|(image, layout)| (image, layout, slot.exposure)), slot.captured, slot.release)
        };
        slot.point = Some(InlinePoint {
            exposure_input: exposure.map(|_| Aux { image: slot.exposure, format: vk::Format::R16_SFLOAT, layout: Some(vk::ImageLayout::GENERAL), readable: true }),
            ..point
        });
        slot.owner = Some(cb);
        slot.used = now;
        s.by_cb.insert(cb, index);
        drop(s);
        self.say_once(format!(
            "holding inside DLSS's command buffer, right before its first launch that reads the colour input (the split in front of it is refused: {:?})",
            point.hazard
        ));
        true
    }

    /// The jobs of a submission about to be made on a queue of `family`: one per buffer carrying a
    /// hold, in submission order.
    pub(crate) fn jobs_for(&self, cbs: impl IntoIterator<Item = vk::CommandBuffer>, family: Option<u32>) -> Vec<Job> {
        let mut s = self.slots.lock().unwrap();
        let now = Instant::now();
        let mut jobs = Vec::new();
        for cb in cbs {
            let Some(&i) = s.by_cb.get(&cb) else { continue };
            let slot = &mut s.slots[i];
            slot.used = now;
            if let Some(point) = slot.point {
                // A queue the layer never saw (`vkGetDeviceQueue*`) counts as another family: the
                // worker then only releases.
                jobs.push(Job {
                    image: slot.image,
                    exposure: point.exposure_input,
                    point,
                    family: family.unwrap_or(u32::MAX),
                    side_family: self.side_family,
                    captured: slot.captured,
                    release: slot.release,
                });
            }
        }
        jobs
    }

    /// Hands a submitted batch's jobs to the worker. Called only after the next layer accepted the
    /// submission (a refused one executes nothing and waits on nothing).
    pub(crate) fn dispatch(&self, jobs: Vec<Job>) {
        if jobs.is_empty() {
            return;
        }
        let tx = self.jobs.lock().unwrap();
        let Some(tx) = tx.as_ref() else {
            // Stopping: release at once so the GPU never waits on a host that left.
            for job in jobs {
                // SAFETY: the layer's own event.
                let _ = unsafe { self.device.set_event(job.release) };
            }
            return;
        };
        for job in jobs {
            if let Err(mpsc::SendError(job)) = tx.send(job) {
                // SAFETY: as above.
                let _ = unsafe { self.device.set_event(job.release) };
            }
        }
    }

    /// # Safety
    /// The device is live.
    unsafe fn build_slot(&self, extent: (u32, u32)) -> Option<Slot> {
        let d = &self.device;
        // SAFETY: valid create infos on a live device.
        unsafe {
            let captured = d.create_event(&vk::EventCreateInfo::default(), None).ok()?;
            let Ok(release) = d.create_event(&vk::EventCreateInfo::default(), None) else {
                d.destroy_event(captured, None);
                return None;
            };
            let mut slot = Slot {
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
                exposure: vk::Image::null(),
                exposure_memory: vk::DeviceMemory::null(),
                extent: (0, 0),
                captured,
                release,
                owner: None,
                point: None,
                used: Instant::now(),
            };
            self.resize_slot(&mut slot, extent);
            match self.image(vk::Format::R16_SFLOAT, (1, 1)) {
                Some((image, memory)) if slot.image != vk::Image::null() => {
                    slot.exposure = image;
                    slot.exposure_memory = memory;
                }
                other => {
                    if let Some((image, memory)) = other {
                        d.destroy_image(image, None);
                        d.free_memory(memory, None);
                    }
                    if slot.image != vk::Image::null() {
                        d.destroy_image(slot.image, None);
                        d.free_memory(slot.memory, None);
                    }
                    d.destroy_event(captured, None);
                    d.destroy_event(release, None);
                    return None;
                }
            }
            Some(slot)
        }
    }

    /// (Re)builds the slot's staging image at `extent`. Leaves a null image when that fails.
    ///
    /// # Safety
    /// Nothing pending may use the slot's current image.
    unsafe fn resize_slot(&self, slot: &mut Slot, extent: (u32, u32)) {
        if slot.image != vk::Image::null() {
            // SAFETY: not in use (contract).
            unsafe {
                self.device.destroy_image(slot.image, None);
                self.device.free_memory(slot.memory, None);
            }
        }
        slot.extent = extent;
        // SAFETY: the device is live.
        (slot.image, slot.memory) = unsafe { self.image(vk::Format::R16G16B16A16_SFLOAT, extent) }.unwrap_or_default();
    }

    /// A device-local image of the layer's: `TRANSFER_SRC | TRANSFER_DST | STORAGE` (the encode and
    /// decode use the staging image as a storage image), shared by DLSS's family and the layer's
    /// queue's family when they differ.
    ///
    /// # Safety
    /// The device is live.
    unsafe fn image(&self, format: vk::Format, extent: (u32, u32)) -> Option<(vk::Image, vk::DeviceMemory)> {
        let d = &self.device;
        let families = [self.app_family, self.side_family];
        let mut info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.0, height: extent.1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        info = if self.app_family == self.side_family {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        } else {
            info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families)
        };
        // SAFETY: valid create info on a live device.
        let image = match unsafe { d.create_image(&info, None) } {
            Ok(image) => image,
            Err(e) => {
                crate::log!("[preupscale] staging image {}x{} {format:?}: vkCreateImage failed ({e:?})", extent.0, extent.1);
                crate::logging::flush();
                return None;
            }
        };
        // SAFETY: `image` was just created.
        let reqs = unsafe { d.get_image_memory_requirements(image) };
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let index = (0..self.memory.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(local))
            .or_else(|| (0..self.memory.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0));
        let alloc = index.map(|index| vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index).build());
        // SAFETY: valid allocate info.
        let memory = alloc.and_then(|a| unsafe { d.allocate_memory(&a, None) }.inspect_err(|e| crate::log!("[preupscale] staging image: vkAllocateMemory of {} bytes failed ({e:?})", reqs.size)).ok());
        // SAFETY: a fresh image and an allocation of its required size and type.
        match memory.filter(|&m| unsafe { d.bind_image_memory(image, m, 0) }.is_ok()) {
            Some(memory) => Some((image, memory)),
            None => {
                // SAFETY: never used.
                unsafe {
                    d.destroy_image(image, None);
                    if let Some(m) = memory {
                        d.free_memory(m, None);
                    }
                }
                crate::logging::flush();
                None
            }
        }
    }
}

fn release(s: &mut Slots, cb: vk::CommandBuffer) {
    if let Some(i) = s.by_cb.remove(&cb) {
        let slot = &mut s.slots[i];
        slot.owner = None;
        slot.point = None;
    }
}

/// Polls `event` until it is set (`true`), `deadline` passes or the device stops (`false`).
fn wait_set(device: &ash::Device, event: vk::Event, deadline: Duration, stopping: &AtomicBool) -> bool {
    let start = Instant::now();
    let mut spins = 0u32;
    loop {
        // SAFETY: the layer's own event.
        match unsafe { device.get_event_status(event) } {
            Ok(true) => return true,
            Ok(false) => {}
            Err(_) => return false,
        }
        if start.elapsed() > deadline || stopping.load(Ordering::Relaxed) {
            return false;
        }
        spins += 1;
        if spins < 256 {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

/// The commands recorded into the application's buffer before its first colour launch.
///
/// # Safety
/// `cb` is being recorded on this thread; `colour` is in `GENERAL` at this point of the buffer and
/// has `TRANSFER_SRC | TRANSFER_DST`; `staging` is the slot's image of the colour input's extent.
unsafe fn record_hold(
    d: &ash::Device, cb: vk::CommandBuffer, colour: vk::Image, staging: vk::Image, extent: (u32, u32),
    exposure: Option<(vk::Image, vk::ImageLayout, vk::Image)>, captured: vk::Event, release: vk::Event,
) {
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let region = vk::ImageCopy {
        src_subresource: layers,
        src_offset: vk::Offset3D::default(),
        dst_subresource: layers,
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D { width: extent.0, height: extent.1, depth: 1 },
    };
    // Everything the buffer wrote before this point (the frame, rendered in this same buffer) is
    // visible to the copy; the staging image's old contents are not needed.
    let open = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE).build();
    let staging_general = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(staging)
        .subresource_range(range)
        .build();
    // DLSS's 1x1 exposure input, copied beside the colour input into the slot's own (`exposure`:
    // source, its layout, the slot's image).
    let exposure_general = exposure.map(|(_, _, own)| vk::ImageMemoryBarrier { image: own, ..staging_general });
    let exposure_region = vk::ImageCopy { extent: vk::Extent3D { width: 1, height: 1, depth: 1 }, ..region };
    // The copy made available before the host sees `captured` and the side queue reads it.
    let published = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
    // The side queue's write-back (made available by its fence, which the host waited on before
    // setting `release`) visible to the copy back.
    let acquire = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
        .build();
    // The copy back visible to DLSS's launch and everything after it.
    let close = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE).build();
    // SAFETY: the contract; every command is legal outside a render pass instance, where a CUDA
    // launch is.
    unsafe {
        let to_general: Vec<vk::ImageMemoryBarrier> = std::iter::once(staging_general).chain(exposure_general).collect();
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[open], &[], &to_general);
        d.cmd_copy_image(cb, colour, vk::ImageLayout::GENERAL, staging, vk::ImageLayout::GENERAL, &[region]);
        if let Some((source, layout, own)) = exposure {
            d.cmd_copy_image(cb, source, layout, own, vk::ImageLayout::GENERAL, &[exposure_region]);
        }
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[published], &[], &[]);
        d.cmd_set_event(cb, captured, vk::PipelineStageFlags::TRANSFER);
        d.cmd_wait_events(cb, &[release], vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::TRANSFER, &[acquire], &[], &[]);
        d.cmd_copy_image(cb, staging, vk::ImageLayout::GENERAL, colour, vk::ImageLayout::GENERAL, &[region]);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[close], &[], &[]);
        // Reset for the next execution once everything before has run (the wait included).
        d.cmd_reset_event(cb, captured, vk::PipelineStageFlags::ALL_COMMANDS);
        d.cmd_reset_event(cb, release, vk::PipelineStageFlags::ALL_COMMANDS);
    }
}

/// Stops `device`'s worker and frees the slots: called from `vkDestroyDevice` before anything else
/// of the device's is freed. Jobs still queued are released at once (the worker skips the hold
/// while stopping), the worker is joined, then the device is waited idle so no buffer still uses a
/// slot.
///
/// # Safety
/// The caller meets `vkDestroyDevice`'s external synchronization requirements.
pub(crate) unsafe fn destroy(device: vk::Device) {
    let Some(inline) = INLINES.lock().unwrap().remove(&device) else { return };
    inline.stopping.store(true, Ordering::Relaxed);
    inline.jobs.lock().unwrap().take();
    if let Some(worker) = inline.worker.lock().unwrap().take() {
        let _ = worker.join();
    }
    let d = &inline.device;
    // SAFETY: the application destroys the device and owns all its queues now; the worker (the side
    // queue's only user) has exited.
    match unsafe { d.device_wait_idle() } {
        Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {
            let mut s = inline.slots.lock().unwrap();
            for slot in s.slots.drain(..) {
                // SAFETY: idle (or lost): nothing uses them.
                unsafe {
                    if slot.image != vk::Image::null() {
                        d.destroy_image(slot.image, None);
                        d.free_memory(slot.memory, None);
                    }
                    d.destroy_image(slot.exposure, None);
                    d.free_memory(slot.exposure_memory, None);
                    d.destroy_event(slot.captured, None);
                    d.destroy_event(slot.release, None);
                }
            }
        }
        Err(e) => crate::log!("[preupscale] inline hold teardown wait failed: {e:?}; staging slots not freed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preupscale::{Hazard, ImageDesc};
    use std::sync::atomic::AtomicU32;

    /// The commands recorded into the application's buffer on a real device (lavapipe in CI): the
    /// GPU stops at the hold until the worker releases it, the colour input comes back exactly as
    /// it was (a hold that writes nothing changes nothing), both events are reset afterwards, and
    /// the same buffer submitted again is held again. The worker's hold here only waits; the side
    /// queue's encode, helper round trip and decode are `run_hold`'s, tested on their own.
    #[test]
    fn the_gpu_waits_at_the_hold_until_the_worker_releases_it_and_the_frame_round_trips() {
        let Some((_entry, instance, physical_device, device, queue, family)) = crate::composition::gpu::test_device() else { return };
        let device = Arc::new(device);
        let (w, h) = (64u32, 48u32);
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE;
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: test-only setup on a live device; every handle is destroyed at the end.
        unsafe {
            let colour = device.create_image(&info, None).unwrap();
            let reqs = device.get_image_memory_requirements(colour);
            let props = instance.get_physical_device_memory_properties(physical_device);
            let index = (0..props.memory_type_count).find(|&i| reqs.memory_type_bits & (1 << i) != 0).unwrap();
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(index), None).unwrap();
            device.bind_image_memory(colour, memory, 0).unwrap();
            let bytes = u64::from(w * h * 8);
            let readback = device
                .create_buffer(&vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST), None)
                .unwrap();
            let breqs = device.get_buffer_memory_requirements(readback);
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let bindex = (0..props.memory_type_count)
                .find(|&i| breqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(host))
                .unwrap();
            let bmemory = device.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(breqs.size).memory_type_index(bindex), None).unwrap();
            device.bind_buffer_memory(readback, bmemory, 0).unwrap();

            let holds = Arc::new(AtomicU32::new(0));
            let counted = holds.clone();
            let hold: HoldFn = Box::new(move |_job, _queue| {
                std::thread::sleep(Duration::from_millis(60));
                counted.fetch_add(1, Ordering::SeqCst);
            });
            let inline = Inline::start(device.clone(), &instance, physical_device, vk::Queue::null(), family, family, hold).expect("worker");

            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(family), None).unwrap();
            let cb = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(1))
                .unwrap()[0];
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let to_general = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(colour)
                .subresource_range(range)
                .build();
            let written = vk::MemoryBarrier::builder().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ).build();
            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cb, &begin).unwrap();
            inline.begin(cb, begin.flags);
            // The "frame", rendered in the same buffer before DLSS's launch.
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_general]);
            let clear = vk::ClearColorValue { float32: [1.0, 0.5, 0.25, 1.0] };
            device.cmd_clear_color_image(cb, colour, vk::ImageLayout::GENERAL, &clear, &[range]);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[written], &[], &[]);
            let point = InlinePoint {
                colour,
                desc: ImageDesc { width: w, height: h, format: vk::Format::R16G16B16A16_SFLOAT, usage, plain: true },
                exposure_input: None,
                identification: 0,
                hazard: Hazard::MemoryBarrier,
            };
            assert!(inline.record(cb, point));
            assert!(inline.owns(cb));
            // DLSS's launch stand-in: what it reads.
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                ..Default::default()
            };
            device.cmd_copy_image_to_buffer(cb, colour, vk::ImageLayout::GENERAL, readback, &[region]);
            device.end_command_buffer(cb).unwrap();

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let ptr = device.map_memory(bmemory, 0, bytes, vk::MemoryMapFlags::empty()).unwrap().cast::<u16>();
            for round in 1..=2u32 {
                std::ptr::write_bytes(ptr, 0, (bytes / 2) as usize);
                let jobs = inline.jobs_for([cb], Some(family));
                assert_eq!(jobs.len(), 1);
                let started = Instant::now();
                device.queue_submit(queue, &[vk::SubmitInfo::builder().command_buffers(&[cb]).build()], fence).unwrap();
                inline.dispatch(jobs);
                device.wait_for_fences(&[fence], true, 10_000_000_000).unwrap();
                assert!(started.elapsed() >= Duration::from_millis(50), "the GPU did not wait for the release ({:?})", started.elapsed());
                assert_eq!(holds.load(Ordering::SeqCst), round, "the worker ran the hold once per execution");
                let texels = std::slice::from_raw_parts(ptr, (w * h * 4) as usize);
                let expected = [0x3c00u16, 0x3800, 0x3400, 0x3c00]; // 1.0, 0.5, 0.25, 1.0 as f16
                assert!(texels.chunks_exact(4).all(|t| t == expected), "the colour input changed across a hold that wrote nothing");
                let slots = inline.slots.lock().unwrap();
                let slot = &slots.slots[slots.by_cb[&cb]];
                assert_eq!(device.get_event_status(slot.captured), Ok(false), "captured reset by the buffer");
                assert_eq!(device.get_event_status(slot.release), Ok(false), "release reset by the buffer");
                drop(slots);
                device.reset_fences(&[fence]).unwrap();
            }
            // Re-recording frees the slot.
            device.begin_command_buffer(cb, &begin).unwrap();
            inline.begin(cb, begin.flags);
            assert!(!inline.owns(cb));
            device.end_command_buffer(cb).unwrap();

            destroy(device.handle());
            device.unmap_memory(bmemory);
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            device.destroy_buffer(readback, None);
            device.free_memory(bmemory, None);
            device.destroy_image(colour, None);
            device.free_memory(memory, None);
        }
    }
}
