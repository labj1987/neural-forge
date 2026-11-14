//! The second GPU shows the frame (docs/NATIVE_BACKEND.md, "The second GPU shows the frame"): with the model on the
//! finished frame and a second GPU (`NEURAL_FORGE_OFFLOAD_GPU`), the setting `second_gpu_present` makes that GPU
//! present the model's output on a swapchain of its own. Each frame crosses to it once and nothing comes back, so
//! the game never waits for an answer; frames the link or the second GPU cannot carry are skipped.
//!
//! A surface takes one swapchain, and a swapchain belongs to one device. The real swapchain is therefore made on the
//! second device, and the game gets a swapchain the layer owns ([`Virtual`]): plain images on the game's device and
//! a handle of the layer's own, which the hooks of every command that names a swapchain answer themselves
//! (`crate::device`). The game's present becomes a hand-off: one copy of the image into host memory on the game's
//! queue, into one of [`SLOTS`] staging slots ([`Ring`]), and the call returns. A presenter thread takes the newest
//! slot, uploads it into the real swapchain's next image and presents it through a [`Shower`]: on the second device
//! that is the layer's own finished-frame path (its device has a record of its own, `crate::device`), so the capture,
//! the network, the composition and everything else of that path run there as they do on a game's device.
//!
//! Anything that fails before the swapchain exists leaves the game its own swapchain (the round trip, or the model
//! on its GPU); a failure while it runs makes the game's next acquire or present return `VK_ERROR_OUT_OF_DATE_KHR`,
//! and the swapchain it then creates is a real one ([`Shared::fail`]).

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle;

/// The staging slots between the game's present and the presenter: one the second GPU works on, one holding the
/// newest finished frame, one free to write, so a present never waits for either.
pub(crate) const SLOTS: usize = 3;

/// The device commands hooked for it, when a second GPU is asked for (`on`): everything else that names a swapchain.
/// Without `NEURAL_FORGE_OFFLOAD_GPU` none of them is hooked.
pub(crate) fn commands(on: bool) -> &'static [vulkan_layer::LayerVulkanCommand] {
    use vulkan_layer::LayerVulkanCommand as C;
    if on {
        &[
            C::GetSwapchainImagesKhr,
            C::AcquireNextImageKhr,
            C::AcquireNextImage2Khr,
            C::GetSwapchainStatusKhr,
            C::WaitForPresentKhr,
            C::ReleaseSwapchainImagesExt,
            C::SetLatencySleepModeNv,
            C::LatencySleepNv,
            C::SetLatencyMarkerNv,
            C::GetLatencyTimingsNv,
        ]
    } else {
        &[]
    }
}

/// Bytes per texel of the swapchain formats that can be handed over; `None` for any other (the game then keeps a
/// swapchain of its own).
pub(crate) fn texel_bytes(format: vk::Format) -> Option<u64> {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2R10G10B10_UNORM_PACK32 => Some(4),
        vk::Format::R16G16B16A16_SFLOAT => Some(8),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The hand-off.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    /// Claimed by a present that has not submitted its copy yet.
    Writing,
    /// Holds the frame with this sequence number (its copy submitted; the presenter waits for it).
    Ready(u64),
    /// The presenter works on it.
    Showing,
}

struct RingInner {
    slots: [SlotState; SLOTS],
    next: u64,
    shown: u64,
    skipped: u64,
    closed: bool,
}

/// What the presenter gets from [`Ring::take`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Take {
    /// The newest frame: its slot and sequence number.
    Frame(usize, u64),
    /// Nothing came within the wait.
    Idle,
    Closed,
}

/// The staging slots' states: which the game's present may write, which the presenter shows next. The newest frame
/// wins: a frame still waiting when a newer one is there is never shown, and counts as skipped.
pub(crate) struct Ring {
    inner: Mutex<RingInner>,
    wake: Condvar,
}

impl Ring {
    pub(crate) fn new() -> Self {
        Self { inner: Mutex::new(RingInner { slots: [SlotState::Free; SLOTS], next: 1, shown: 0, skipped: 0, closed: false }), wake: Condvar::new() }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RingInner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A slot for the present's frame: a free one, else the one holding the oldest waiting frame (that frame is
    /// skipped). `idle(k)`: nothing of slot `k` is still being written by the GPU, so it may be written again; a slot
    /// that is not idle is never given. `None`: no slot, the present's own frame is skipped.
    pub(crate) fn claim(&self, idle: impl Fn(usize) -> bool) -> Option<usize> {
        let mut r = self.lock();
        if r.closed {
            r.skipped += 1;
            return None;
        }
        let free = (0..SLOTS).find(|&k| r.slots[k] == SlotState::Free && idle(k));
        let pick = free.or_else(|| {
            let oldest = (0..SLOTS).filter_map(|k| if let SlotState::Ready(seq) = r.slots[k] { Some((seq, k)) } else { None }).filter(|&(_, k)| idle(k)).min();
            oldest.map(|(_, k)| k)
        });
        match pick {
            Some(k) => {
                if r.slots[k] != SlotState::Free {
                    r.skipped += 1;
                }
                r.slots[k] = SlotState::Writing;
                Some(k)
            }
            None => {
                r.skipped += 1;
                None
            }
        }
    }

    /// The claimed slot's copy was submitted: its frame waits for the presenter. Returns its sequence number.
    pub(crate) fn filled(&self, k: usize) -> u64 {
        let mut r = self.lock();
        let seq = r.next;
        r.next += 1;
        r.slots[k] = SlotState::Ready(seq);
        self.wake.notify_all();
        seq
    }

    /// The claimed slot's copy could not be submitted: free again, the frame skipped.
    pub(crate) fn abandon(&self, k: usize) {
        let mut r = self.lock();
        r.slots[k] = SlotState::Free;
        r.skipped += 1;
    }

    /// The newest waiting frame, waited for up to `wait`; older waiting frames are dropped (skipped).
    pub(crate) fn take(&self, wait: Duration) -> Take {
        let mut r = self.lock();
        let deadline = Instant::now() + wait;
        loop {
            if r.closed {
                return Take::Closed;
            }
            let newest = (0..SLOTS).filter_map(|k| if let SlotState::Ready(seq) = r.slots[k] { Some((seq, k)) } else { None }).max();
            if let Some((seq, k)) = newest {
                for j in 0..SLOTS {
                    if j != k && matches!(r.slots[j], SlotState::Ready(_)) {
                        r.slots[j] = SlotState::Free;
                        r.skipped += 1;
                    }
                }
                r.slots[k] = SlotState::Showing;
                return Take::Frame(k, seq);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Take::Idle;
            }
            r = self.wake.wait_timeout(r, left).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }

    /// The presenter is done with slot `k`: its frame was shown, or not (`shown`).
    pub(crate) fn done(&self, k: usize, shown: bool) {
        let mut r = self.lock();
        r.slots[k] = SlotState::Free;
        if shown {
            r.shown += 1;
        } else {
            r.skipped += 1;
        }
    }

    /// No more frames: the presenter ends, presents skip.
    pub(crate) fn close(&self) {
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Frames shown and skipped so far.
    pub(crate) fn counts(&self) -> (u64, u64) {
        let r = self.lock();
        (r.shown, r.skipped)
    }
}

/// One of the game's images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImageState {
    Free,
    /// The game has it (between its acquire and its present).
    Acquired,
    /// Presented: the copy `seq` into slot `slot` may still read it.
    Copying {
        slot: usize,
        seq: u64,
    },
}

/// The image the next acquire gives: the first from `cursor` on that is free or whose copy finished (`copied`).
pub(crate) fn pick_image(images: &[ImageState], cursor: usize, copied: impl Fn(usize, u64) -> bool) -> Option<usize> {
    (0..images.len()).map(|n| (cursor + n) % images.len()).find(|&i| match images[i] {
        ImageState::Free => true,
        ImageState::Acquired => false,
        ImageState::Copying { slot, seq } => copied(slot, seq),
    })
}

/// What a result of the real swapchain (its acquire or present) means for the game's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Shown,
    /// Shown, and the game is told the swapchain no longer matches the surface.
    Suboptimal,
    /// Not shown: the game must recreate the swapchain (the result it is given).
    Stale(vk::Result),
    /// Nothing was acquired in time: this frame is skipped, the next is tried.
    Busy,
    /// The second GPU cannot go on showing frames.
    Failed,
}

pub(crate) fn verdict(result: vk::Result) -> Verdict {
    match result {
        vk::Result::SUCCESS => Verdict::Shown,
        vk::Result::SUBOPTIMAL_KHR => Verdict::Suboptimal,
        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_SURFACE_LOST_KHR | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => {
            Verdict::Stale(result)
        }
        vk::Result::TIMEOUT | vk::Result::NOT_READY => Verdict::Busy,
        _ => Verdict::Failed,
    }
}

/// The Status tab's line while the second GPU shows the frames (`ShmHeader::second_gpu_present_line`).
pub(crate) fn showing_line(gpu: &str, shown: u64, skipped: u64, copy_ms: f64) -> String {
    let total = shown + skipped;
    let pct = if total == 0 { 0.0 } else { skipped as f64 * 100.0 / total as f64 };
    let gpu: String = gpu.chars().take(48).collect();
    format!("second GPU shows the frames: {gpu}; {shown} shown, {skipped} skipped ({pct:.0}%), copy {copy_ms:.1} ms")
}

/// The line when it does not (the frames come back to the game's GPU, or the model runs there).
pub(crate) fn not_showing_line(why: &str) -> String {
    let why: String = why.chars().take(120).collect();
    format!("second GPU does not show the frames: {why}")
}

// ---------------------------------------------------------------------------------------------------------------
// The second device's side.

/// Presents on the second device: the layer's finished-frame path there (`crate::device`), a plain present in tests.
pub(crate) trait Shower: Send + Sync {
    /// # Safety
    /// `info` is a valid create info for the shower's device.
    unsafe fn create_swapchain(&self, info: &vk::SwapchainCreateInfoKHR) -> ash::prelude::VkResult<vk::SwapchainKHR>;
    /// # Safety
    /// `queue` is the second device's present queue, used by this thread only; `info` is valid.
    unsafe fn present(&self, queue: vk::Queue, info: &vk::PresentInfoKHR) -> vk::Result;
    /// # Safety
    /// Nothing of `swapchain` is in use.
    unsafe fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR);
    /// # Safety
    /// `swapchain` is the shower's own, not in use by another thread.
    unsafe fn set_hdr_metadata(&self, swapchain: vk::SwapchainKHR, metadata: &vk::HdrMetadataEXT);
}

/// `VK_KHR_surface`'s queries, from the next layer's `vkGetInstanceProcAddr`.
#[derive(Clone, Copy)]
pub(crate) struct SurfaceFns {
    pub support: vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR,
    pub caps: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
    pub formats: vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR,
    pub modes: vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR,
}

impl SurfaceFns {
    /// # Safety
    /// `gipa` is a `vkGetInstanceProcAddr` that serves `instance`.
    pub(crate) unsafe fn load(gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance) -> Option<Self> {
        // SAFETY: as the contract; each name matches the PFN type it is cast to.
        unsafe {
            Some(Self {
                support: std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(gipa(
                    instance,
                    c"vkGetPhysicalDeviceSurfaceSupportKHR".as_ptr(),
                )?),
                caps: std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(gipa(
                    instance,
                    c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR".as_ptr(),
                )?),
                formats: std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(gipa(
                    instance,
                    c"vkGetPhysicalDeviceSurfaceFormatsKHR".as_ptr(),
                )?),
                modes: std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(gipa(
                    instance,
                    c"vkGetPhysicalDeviceSurfacePresentModesKHR".as_ptr(),
                )?),
            })
        }
    }
}

/// The second device, as the hand-off uses it.
pub(crate) struct Second {
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    /// Its present queue: the presenter thread's alone while a swapchain is shown.
    pub queue: vk::Queue,
    pub family: u32,
    pub surface: SurfaceFns,
    pub shower: Box<dyn Shower>,
    /// For the Status tab and the log.
    pub name: String,
    /// `VK_EXT_external_memory_host` is enabled on it: the staging can be host memory both devices import.
    pub import: bool,
    /// A swapchain of the game is being shown on it (one at a time: its queue is one presenter's).
    busy: AtomicBool,
    /// Why it stopped showing frames, for the session.
    failed: Mutex<Option<String>>,
    /// Where the Status line goes (the channel's header), once the channel is open.
    pub header: Option<&'static neural_forge_protocol::ShmHeader>,
}

// SAFETY: handles and function pointers; the queue is used by one thread at a time (`busy`).
unsafe impl Send for Second {}
unsafe impl Sync for Second {}

impl Second {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: ash::Device,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        queue: vk::Queue,
        family: u32,
        surface: SurfaceFns,
        shower: Box<dyn Shower>,
        name: String,
        import: bool,
        header: Option<&'static neural_forge_protocol::ShmHeader>,
    ) -> Self {
        Self { device, instance, physical, queue, family, surface, shower, name, import, busy: AtomicBool::new(false), failed: Mutex::new(None), header }
    }

    /// Why the second GPU stopped showing frames this session, if it did.
    pub(crate) fn failed(&self) -> Option<String> {
        self.failed.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    fn publish(&self, line: &str) {
        if let Some(h) = self.header {
            if h.second_gpu_present_line() != line {
                h.set_second_gpu_present_line(line);
            }
        }
    }

    /// The real swapchain for the game's `info`, replacing `old` (retired by the call, not destroyed): the game's
    /// format, colour space, extent, transform and alpha, its present mode where the second GPU has it (else FIFO),
    /// with the usage the upload and the finished-frame path need.
    ///
    /// # Safety
    /// `info` is the game's valid create info; `old` is null or a swapchain of this device not in use.
    unsafe fn create_swapchain(
        &self,
        info: &vk::SwapchainCreateInfoKHR,
        old: vk::SwapchainKHR,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::PresentModeKHR), String> {
        let s = &self.surface;
        let mut supported = vk::FALSE;
        // SAFETY: the instance's own physical device and the game's surface (an instance object).
        let r = unsafe { (s.support)(self.physical, self.family, info.surface, &mut supported) };
        if r != vk::Result::SUCCESS || supported == vk::FALSE {
            return Err(format!("{} cannot present to the game's window (is the monitor plugged into it?)", self.name));
        }
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        // SAFETY: as above.
        let r = unsafe { (s.caps)(self.physical, info.surface, &mut caps) };
        if r != vk::Result::SUCCESS {
            return Err(format!("the surface's capabilities on the second GPU could not be read ({r:?})"));
        }
        // SAFETY: the two-call enumeration of each query.
        let formats = unsafe { enumerate(|n, p| (s.formats)(self.physical, info.surface, n, p)) };
        if !formats.iter().any(|f| f.format == info.image_format && f.color_space == info.image_color_space) {
            return Err(format!("the second GPU does not offer {:?} with {:?} on this surface", info.image_format, info.image_color_space));
        }
        // SAFETY: as above.
        let modes = unsafe { enumerate(|n, p| (s.modes)(self.physical, info.surface, n, p)) };
        let mode = if modes.contains(&info.present_mode) { info.present_mode } else { vk::PresentModeKHR::FIFO };
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if !caps.supported_usage_flags.contains(usage) {
            return Err("the surface on the second GPU cannot be copied into".into());
        }
        let mut count = info.min_image_count.max(caps.min_image_count);
        if caps.max_image_count != 0 {
            count = count.min(caps.max_image_count);
        }
        let transform = if caps.supported_transforms.contains(info.pre_transform) { info.pre_transform } else { caps.current_transform };
        let alpha = if caps.supported_composite_alpha.contains(info.composite_alpha) {
            info.composite_alpha
        } else if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };
        // Nothing of the game's chain or flags: they are for its device (format lists, present-mode lists, full-screen
        // info), and the finished-frame path only takes a swapchain without them.
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(info.surface)
            .min_image_count(count)
            .image_format(info.image_format)
            .image_color_space(info.image_color_space)
            .image_extent(info.image_extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(transform)
            .composite_alpha(alpha)
            .present_mode(mode)
            .clipped(info.clipped != vk::FALSE)
            .old_swapchain(old);
        // SAFETY: a valid create info for the shower's device.
        let sc = unsafe { self.shower.create_swapchain(&ci) }.map_err(|e| format!("creating the swapchain on the second GPU failed ({e:?})"))?;
        let fns = ash::khr::swapchain::Device::new(&self.instance, &self.device);
        // SAFETY: the swapchain just made.
        match unsafe { fns.get_swapchain_images(sc) } {
            Ok(images) if !images.is_empty() => Ok((sc, images, mode)),
            other => {
                // SAFETY: just made, nothing of it used.
                unsafe { self.shower.destroy_swapchain(sc) };
                Err(format!("the second GPU's swapchain has no images ({:?})", other.err()))
            }
        }
    }
}

/// A two-call enumeration.
unsafe fn enumerate<T: Default + Clone>(f: impl Fn(*mut u32, *mut T) -> vk::Result) -> Vec<T> {
    let mut n = 0u32;
    if f(&mut n, std::ptr::null_mut()) != vk::Result::SUCCESS || n == 0 {
        return Vec::new();
    }
    let mut out = vec![T::default(); n as usize];
    let r = f(&mut n, out.as_mut_ptr());
    if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
        return Vec::new();
    }
    out.truncate(n as usize);
    out
}

fn memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, want: &[vk::MemoryPropertyFlags]) -> Option<u32> {
    want.iter().find_map(|&w| (0..props.memory_type_count).find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(w)))
}

/// A mapped host-visible buffer.
struct Stage {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
}

impl Stage {
    /// # Safety
    /// `physical` is `device`'s.
    unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        cached: bool,
    ) -> Result<Self, String> {
        // SAFETY: valid create infos; everything made is destroyed on failure.
        unsafe {
            let buffer = device
                .create_buffer(&vk::BufferCreateInfo::default().size(bytes).usage(usage), None)
                .map_err(|e| format!("a staging buffer failed ({e:?})"))?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            let props = instance.get_physical_device_memory_properties(physical);
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let want = if cached { vec![hv | vk::MemoryPropertyFlags::HOST_CACHED, hv] } else { vec![hv] };
            let made = memory_type(&props, reqs.memory_type_bits, &want)
                .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
                .and_then(|t| device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(t), None));
            let memory = match made {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(format!("staging memory failed ({e:?})"));
                }
            };
            let mapped = device.bind_buffer_memory(buffer, memory, 0).and_then(|()| device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()));
            match mapped {
                Ok(p) => Ok(Self { buffer, memory, ptr: p.cast() }),
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                    Err(format!("mapping staging memory failed ({e:?})"))
                }
            }
        }
    }

    /// # Safety
    /// Nothing submitted still uses it.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: as the contract (freeing the memory unmaps it).
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Host memory for one staging slot that both devices import (`VK_EXT_external_memory_host`): the game's GPU writes
/// the frame into it and the second GPU reads it from there, with no copy by the CPU.
struct Pages {
    ptr: *mut u8,
    len: usize,
}

impl Pages {
    fn new(len: usize) -> Option<Self> {
        // SAFETY: an anonymous private mapping of `len` bytes (page-aligned by the kernel).
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        (ptr != libc::MAP_FAILED).then_some(Self { ptr: ptr.cast(), len })
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: this mapping, made by `new`; every buffer imported from it was destroyed (or it was leaked instead).
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

/// One slot's staging on both devices from the same host pages, where both import host memory: the game's side, the
/// second device's and the pages. `None`: this slot is staged in a buffer per device and copied by the presenter.
///
/// # Safety
/// Both devices are live and have `VK_EXT_external_memory_host` enabled.
unsafe fn shared_stage(game: &GameDevice, second: &Second, bytes: u64) -> Option<(Stage, Stage, Pages)> {
    let a = crate::capture::min_imported_host_pointer_alignment(game.instance, game.physical)?;
    let b = crate::capture::min_imported_host_pointer_alignment(&second.instance, second.physical)?;
    let align = a.max(b).max(4096);
    let len = bytes.div_ceil(align) * align;
    let pages = Pages::new(len as usize)?;
    // SAFETY: the pages are `len` bytes, aligned to both devices' import alignment, and outlive both buffers.
    unsafe {
        let (buffer, memory) =
            crate::capture::import_host_buffer(game.device, game.instance, game.physical, pages.ptr, len, vk::BufferUsageFlags::TRANSFER_DST)?;
        let ours = Stage { buffer, memory, ptr: pages.ptr };
        match crate::capture::import_host_buffer(&second.device, &second.instance, second.physical, pages.ptr, len, vk::BufferUsageFlags::TRANSFER_SRC) {
            Some((buffer, memory)) => Some((ours, Stage { buffer, memory, ptr: pages.ptr }, pages)),
            None => {
                ours.destroy(game.device);
                None
            }
        }
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
}

fn layout_barrier(image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(color_range())
}

fn whole(extent: vk::Extent2D) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
}

/// What the game's side and the presenter share.
pub(crate) struct Shared {
    /// What the game's presents return (a `vk::Result`): success, or what the real swapchain last said.
    status: AtomicI32,
    /// The copies' time (µs) and count since the last line: host copy and upload to the second GPU.
    copy_us: AtomicU64,
    copies: AtomicU64,
    /// The last copy time published (µs).
    last_copy_us: AtomicU64,
    hdr: Mutex<Option<Metadata>>,
}

struct Metadata(vk::HdrMetadataEXT<'static>);
// SAFETY: plain numbers; its chain pointer is null.
unsafe impl Send for Metadata {}

impl Shared {
    fn new() -> Self {
        Self {
            status: AtomicI32::new(vk::Result::SUCCESS.as_raw()),
            copy_us: AtomicU64::new(0),
            copies: AtomicU64::new(0),
            last_copy_us: AtomicU64::new(0),
            hdr: Mutex::new(None),
        }
    }

    pub(crate) fn status(&self) -> vk::Result {
        vk::Result::from_raw(self.status.load(Ordering::Acquire))
    }

    /// A stale or failed swapchain stays so; suboptimal never replaces either.
    fn note(&self, result: vk::Result) {
        let now = self.status();
        if now == vk::Result::SUCCESS || (now == vk::Result::SUBOPTIMAL_KHR && result != vk::Result::SUCCESS) {
            self.status.store(result.as_raw(), Ordering::Release);
        }
    }

    /// The second GPU cannot go on: said once, the game's next acquire or present asks it for a new swapchain
    /// (`VK_ERROR_OUT_OF_DATE_KHR`), which is then made on its own device.
    fn fail(&self, second: &Second, why: String) {
        self.status.store(vk::Result::ERROR_OUT_OF_DATE_KHR.as_raw(), Ordering::Release);
        let mut failed = second.failed.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if failed.is_none() {
            crate::log!("[present2] {} no longer shows the frames: {why}; the game is asked for a new swapchain, made on its own GPU", second.name);
            crate::logging::flush();
            second.publish(&not_showing_line(&format!("given up ({why})")));
            *failed = Some(why);
        }
    }
}

/// The presenter's resources on the second device.
struct SecondSide {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    stages: Vec<Stage>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
}

// SAFETY: handles and mapped pointers into memory it owns, used by one thread at a time.
unsafe impl Send for SecondSide {}

impl SecondSide {
    /// # Safety
    /// The presenter was joined; the second device's queue is idle.
    unsafe fn destroy(self, second: &Second, keep_swapchain: bool) -> vk::SwapchainKHR {
        // SAFETY: as the contract.
        unsafe {
            let _ = second.device.queue_wait_idle(second.queue);
            for s in &self.stages {
                s.destroy(&second.device);
            }
            second.device.destroy_fence(self.fence, None);
            second.device.destroy_semaphore(self.acquired, None);
            second.device.destroy_command_pool(self.pool, None);
            if !keep_swapchain {
                second.shower.destroy_swapchain(self.swapchain);
            }
        }
        self.swapchain
    }
}

/// The game's side of one staging slot.
struct GameStage {
    stage: Stage,
    fence: vk::Fence,
}

struct Inner {
    images: Vec<ImageState>,
    cursor: usize,
    /// The command pool and one buffer per slot, for the queue family the game presents on.
    pool: Option<(u32, vk::CommandPool, Vec<vk::CommandBuffer>)>,
    /// Per slot: the frame its last submitted copy carries (0: none submitted yet).
    slot_seq: [u64; SLOTS],
    presenter: Option<std::thread::JoinHandle<SecondSide>>,
    /// The second device's side once the presenter was joined ([`Virtual::retire`]).
    second_side: Option<SecondSide>,
    retired: bool,
}

/// The layer's own queue on the game's device, for the acquires' semaphores and fences.
#[derive(Clone)]
pub(crate) struct Signal {
    pub queue: vk::Queue,
    pub lock: Arc<Mutex<()>>,
}

/// The swapchain the game gets: images on its device, and the hand-off to the second.
pub(crate) struct Virtual {
    game: ash::Device,
    /// The game's surface, which the real swapchain holds.
    pub surface: vk::SurfaceKHR,
    pub extent: vk::Extent2D,
    images: Vec<(vk::Image, vk::DeviceMemory)>,
    stages: Vec<GameStage>,
    /// The host pages of the slots both devices import (the others are copied by the presenter).
    pages: Mutex<Vec<Pages>>,
    bytes: u64,
    ring: Arc<Ring>,
    pub shared: Arc<Shared>,
    second: Arc<Second>,
    signal: Signal,
    inner: Mutex<Inner>,
}

// SAFETY: handles and mapped pointers into memory it owns; the mutable parts are behind `inner`.
unsafe impl Send for Virtual {}
unsafe impl Sync for Virtual {}

static VIRTUALS: RwLock<Vec<(vk::SwapchainKHR, Arc<Virtual>)>> = RwLock::new(Vec::new());
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// The layer's swapchain with this handle, if it is one.
pub(crate) fn lookup(swapchain: vk::SwapchainKHR) -> Option<Arc<Virtual>> {
    if LIVE.load(Ordering::Acquire) == 0 || swapchain == vk::SwapchainKHR::null() {
        return None;
    }
    VIRTUALS.read().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(h, _)| *h == swapchain).map(|(_, v)| v.clone())
}

/// Whether any swapchain is the layer's (the hooks' quick check).
pub(crate) fn any() -> bool {
    LIVE.load(Ordering::Acquire) != 0
}

/// The layer's swapchains whose real swapchain still holds `surface`.
pub(crate) fn on_surface(surface: vk::SurfaceKHR) -> Vec<Arc<Virtual>> {
    if !any() {
        return Vec::new();
    }
    VIRTUALS.read().unwrap_or_else(std::sync::PoisonError::into_inner).iter().filter(|(_, v)| v.surface == surface).map(|(_, v)| v.clone()).collect()
}

/// Every swapchain of the layer's.
pub(crate) fn all() -> Vec<Arc<Virtual>> {
    VIRTUALS.read().unwrap_or_else(std::sync::PoisonError::into_inner).iter().map(|(_, v)| v.clone()).collect()
}

/// What a swapchain of the game's needs from its device.
pub(crate) struct GameDevice<'a> {
    pub device: &'a ash::Device,
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub signal: Signal,
    /// `VK_EXT_external_memory_host` is enabled on it.
    pub import: bool,
}

impl Virtual {
    /// The game's swapchain for `info`, shown by `second`; `old`: the one it replaces (the game's `oldSwapchain`,
    /// one of the layer's), retired by the call. `Err`: why not; nothing is left behind, `old` is retired and its
    /// real swapchain destroyed, so the game's surface is free for a swapchain of its own device.
    ///
    /// # Safety
    /// `info` is the game's valid create info for `game.device`; `second` is not in use by anything else.
    pub(crate) unsafe fn create(
        game: &GameDevice,
        second: &Arc<Second>,
        info: &vk::SwapchainCreateInfoKHR,
        old: Option<&Arc<Virtual>>,
    ) -> Result<vk::SwapchainKHR, String> {
        // The old swapchain's presenter first: the second device's queue and the surface are then this call's.
        let mut old_side = old.and_then(|o| o.retire());
        let held = old.is_some();
        let refuse = |side: &mut Option<SecondSide>, why: String| {
            if let Some(side) = side.take() {
                // SAFETY: the old presenter was joined by `retire`.
                unsafe { side.destroy(second, false) };
            }
            if held {
                second.busy.store(false, Ordering::Release);
            }
            Err(why)
        };
        if let Some(why) = second.failed() {
            return refuse(&mut old_side, format!("given up earlier ({why})"));
        }
        let Some(texel) = texel_bytes(info.image_format) else {
            return refuse(&mut old_side, format!("the swapchain's format {:?} is not one it can carry", info.image_format));
        };
        if info.image_array_layers != 1 {
            return refuse(&mut old_side, "a swapchain with more than one layer".into());
        }
        if !held && second.busy.swap(true, Ordering::AcqRel) {
            return Err("the second GPU already shows another swapchain of the game".into());
        }
        let old_swapchain = old_side.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.swapchain);
        // SAFETY: the game's create info; the old real swapchain is idle.
        let made = unsafe { second.create_swapchain(info, old_swapchain) };
        // The old one goes either way: replaced, or retired by the failed attempt.
        if let Some(side) = old_side.take() {
            // SAFETY: the old presenter was joined by `retire`.
            unsafe { side.destroy(second, false) };
        }
        let (swapchain, second_images, mode) = match made {
            Ok(m) => m,
            Err(why) => {
                second.busy.store(false, Ordering::Release);
                return Err(why);
            }
        };
        let extent = info.image_extent;
        let bytes = u64::from(extent.width) * u64::from(extent.height) * texel;
        // SAFETY: both devices are live; everything made is destroyed on failure.
        let built = unsafe { Self::build(game, second, info, bytes, swapchain, second_images) };
        match built {
            Ok(v) => {
                let v = Arc::new(v);
                let handle = vk::SwapchainKHR::from_raw(Arc::as_ptr(&v) as u64);
                // SAFETY: the presenter owns the second side until it is joined.
                unsafe { v.start() };
                VIRTUALS.write().unwrap_or_else(std::sync::PoisonError::into_inner).push((handle, v.clone()));
                LIVE.fetch_add(1, Ordering::AcqRel);
                crate::log!(
                    "[present2] the game's swapchain {}x{} {:?} {:?} is shown by {}: {} images for the game, present mode {mode:?} there{}; each frame crosses once ({:.1} MB, {}), the game never waits",
                    extent.width,
                    extent.height,
                    info.image_format,
                    info.image_color_space,
                    second.name,
                    v.images.len(),
                    if mode == info.present_mode { String::new() } else { format!(" (the game asked for {:?})", info.present_mode) },
                    bytes as f64 / 1e6,
                    match v.shared_slots() {
                        SLOTS => "through host memory both GPUs import, no CPU copy",
                        0 => "through a buffer on each GPU and one CPU copy",
                        _ => "some slots through host memory both GPUs import, the others with one CPU copy",
                    }
                );
                crate::logging::flush();
                second.publish(&showing_line(&second.name, 0, 0, 0.0));
                Ok(handle)
            }
            Err(why) => {
                // SAFETY: nothing of the new swapchain was used.
                unsafe { second.shower.destroy_swapchain(swapchain) };
                second.busy.store(false, Ordering::Release);
                Err(why)
            }
        }
    }

    unsafe fn build(
        game: &GameDevice,
        second: &Arc<Second>,
        info: &vk::SwapchainCreateInfoKHR,
        bytes: u64,
        swapchain: vk::SwapchainKHR,
        second_images: Vec<vk::Image>,
    ) -> Result<Self, String> {
        let d = game.device;
        let extent = info.image_extent;
        let mut images: Vec<(vk::Image, vk::DeviceMemory)> = Vec::new();
        let mut stages: Vec<GameStage> = Vec::new();
        let mut second_stages: Vec<Stage> = Vec::new();
        // Dropped (unmapped) after `undo` destroyed the buffers imported from them, on every failure below.
        let mut pages: Vec<Pages> = Vec::new();
        // SAFETY: valid create infos on live devices; `undo` destroys what was made.
        unsafe {
            let undo = |images: &[(vk::Image, vk::DeviceMemory)], stages: &[GameStage], second_stages: &[Stage]| {
                for (i, m) in images {
                    d.destroy_image(*i, None);
                    d.free_memory(*m, None);
                }
                for s in stages {
                    s.stage.destroy(d);
                    d.destroy_fence(s.fence, None);
                }
                for s in second_stages {
                    s.destroy(&second.device);
                }
            };
            // The game's format list, if its swapchain's views may have other formats.
            let mut view_formats: Vec<vk::Format> = Vec::new();
            let mut node = info.p_next.cast::<vk::BaseInStructure>();
            while !node.is_null() {
                if (*node).s_type == vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO {
                    let list = &*node.cast::<vk::ImageFormatListCreateInfo>();
                    if list.view_format_count > 0 && !list.p_view_formats.is_null() {
                        view_formats = std::slice::from_raw_parts(list.p_view_formats, list.view_format_count as usize).to_vec();
                    }
                }
                node = (*node).p_next;
            }
            let mutable = info.flags.contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT);
            let families: &[u32] = if info.image_sharing_mode == vk::SharingMode::CONCURRENT && !info.p_queue_family_indices.is_null() {
                std::slice::from_raw_parts(info.p_queue_family_indices, info.queue_family_index_count as usize)
            } else {
                &[]
            };
            let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
            let mut ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(info.image_format)
                .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(info.image_usage | vk::ImageUsageFlags::TRANSFER_SRC)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .flags(if mutable { vk::ImageCreateFlags::MUTABLE_FORMAT } else { vk::ImageCreateFlags::empty() });
            ci = if families.len() > 1 {
                ci.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(families)
            } else {
                ci.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
            if mutable && !view_formats.is_empty() {
                ci = ci.push_next(&mut list);
            }
            let props = game.instance.get_physical_device_memory_properties(game.physical);
            let count = info.min_image_count.max(3) as usize;
            for _ in 0..count {
                let made = d.create_image(&ci, None).map_err(|e| format!("an image for the game failed ({e:?})")).and_then(|image| {
                    let reqs = d.get_image_memory_requirements(image);
                    let memory = memory_type(&props, reqs.memory_type_bits, &[vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty()])
                        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
                        .and_then(|t| d.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(t), None))
                        .and_then(|m| d.bind_image_memory(image, m, 0).map(|()| m).inspect_err(|_| d.free_memory(m, None)));
                    memory.map(|m| (image, m)).map_err(|e| {
                        d.destroy_image(image, None);
                        format!("memory for the game's images failed ({e:?})")
                    })
                });
                match made {
                    Ok(pair) => images.push(pair),
                    Err(why) => {
                        undo(&images, &stages, &second_stages);
                        return Err(why);
                    }
                }
            }
            for _ in 0..SLOTS {
                // Host pages both devices import, where both can: nothing is copied by the CPU. Else a buffer per
                // device, copied by the presenter.
                let both = if game.import && second.import { shared_stage(game, second, bytes) } else { None };
                let (ours, other) = match both {
                    Some((ours, other, host)) => {
                        pages.push(host);
                        (Ok(ours), Ok(other))
                    }
                    None => (
                        Stage::new(d, game.instance, game.physical, bytes, vk::BufferUsageFlags::TRANSFER_DST, true),
                        Stage::new(&second.device, &second.instance, second.physical, bytes, vk::BufferUsageFlags::TRANSFER_SRC, false),
                    ),
                };
                let made = ours.and_then(|stage| match d.create_fence(&vk::FenceCreateInfo::default(), None) {
                    Ok(fence) => Ok(GameStage { stage, fence }),
                    Err(e) => {
                        stage.destroy(d);
                        Err(format!("a fence failed ({e:?})"))
                    }
                });
                match (made, other) {
                    (Ok(a), Ok(b)) => {
                        stages.push(a);
                        second_stages.push(b);
                    }
                    (a, b) => {
                        let why = a.as_ref().err().or(b.as_ref().err()).cloned().unwrap_or_default();
                        if let Ok(a) = a {
                            stages.push(a);
                        }
                        if let Ok(b) = b {
                            second_stages.push(b);
                        }
                        undo(&images, &stages, &second_stages);
                        return Err(why);
                    }
                }
            }
            // The presenter's own: a pool for its queue's family, one buffer, a fence, the acquire's semaphore.
            let sd = &second.device;
            let side = (|| -> ash::prelude::VkResult<SecondSide> {
                let pool = sd.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(second.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?;
                let info = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                let made = crate::loader_data::allocate_commands(sd, &info).and_then(|cmds| {
                    let fence = sd.create_fence(&vk::FenceCreateInfo::default(), None)?;
                    let acquired = sd.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).inspect_err(|_| sd.destroy_fence(fence, None))?;
                    Ok((cmds[0], fence, acquired))
                });
                match made {
                    Ok((cmd, fence, acquired)) => Ok(SecondSide { swapchain, images: Vec::new(), stages: Vec::new(), pool, cmd, fence, acquired }),
                    Err(e) => {
                        sd.destroy_command_pool(pool, None);
                        Err(e)
                    }
                }
            })();
            let mut side = match side {
                Ok(s) => s,
                Err(e) => {
                    undo(&images, &stages, &second_stages);
                    return Err(format!("the presenter's resources on the second GPU failed ({e:?})"));
                }
            };
            side.images = second_images;
            side.stages = second_stages;
            Ok(Self {
                game: d.clone(),
                surface: info.surface,
                extent,
                inner: Mutex::new(Inner {
                    images: vec![ImageState::Free; images.len()],
                    cursor: 0,
                    pool: None,
                    slot_seq: [0; SLOTS],
                    presenter: None,
                    second_side: Some(side),
                    retired: false,
                }),
                images,
                stages,
                pages: Mutex::new(pages),
                bytes,
                ring: Arc::new(Ring::new()),
                shared: Arc::new(Shared::new()),
                second: second.clone(),
                signal: game.signal.clone(),
            })
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Starts the presenter with the second device's side.
    ///
    /// # Safety
    /// Called once, before the handle is given out.
    unsafe fn start(self: &Arc<Self>) {
        let mut inner = self.lock();
        let Some(side) = inner.second_side.take() else { return };
        let (ring, shared, second, game) = (self.ring.clone(), self.shared.clone(), self.second.clone(), self.game.clone());
        let sources: Vec<(usize, vk::Fence)> = self.stages.iter().map(|s| (s.stage.ptr as usize, s.fence)).collect();
        let (bytes, extent) = (self.bytes, self.extent);
        let spawned = std::thread::Builder::new().name("nf-present2".into()).spawn(move || {
            // SAFETY: the second device's queue and `side` are this thread's until it ends; the game's stages outlive
            // it (the swapchain joins it before it frees them).
            unsafe { present_loop(&game, &second, &ring, &shared, side, &sources, bytes, extent) }
        });
        match spawned {
            Ok(handle) => inner.presenter = Some(handle),
            Err(e) => {
                // The side went with the closure: nothing shows; the game is asked for a new swapchain.
                self.shared.fail(&self.second, format!("the presenter thread could not start ({e})"));
            }
        }
    }

    /// How many staging slots are host memory both devices import.
    pub(crate) fn shared_slots(&self) -> usize {
        self.pages.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len()
    }

    /// The game's images.
    pub(crate) fn images(&self) -> Vec<vk::Image> {
        self.images.iter().map(|(i, _)| *i).collect()
    }

    fn fence_done(&self, slot: usize) -> bool {
        // SAFETY: the slot's own fence on the live game device.
        unsafe { self.game.get_fence_status(self.stages[slot].fence) }.unwrap_or(true)
    }

    /// `vkAcquireNextImageKHR`: the next image the game may render into, its semaphore and fence signalled.
    pub(crate) fn acquire(&self, timeout: u64, semaphore: vk::Semaphore, fence: vk::Fence) -> ash::prelude::VkResult<u32> {
        let mut inner = self.lock();
        if inner.retired {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        match verdict(self.shared.status()) {
            Verdict::Stale(r) => return Err(r),
            Verdict::Failed => return Err(vk::Result::ERROR_OUT_OF_DATE_KHR),
            _ => {}
        }
        let started = Instant::now();
        let index = loop {
            let seqs = inner.slot_seq;
            let picked = pick_image(&inner.images, inner.cursor, |slot, seq| seqs[slot] != seq || self.fence_done(slot));
            if let Some(i) = picked {
                break i;
            }
            let pending: Vec<vk::Fence> =
                inner.images.iter().filter_map(|s| if let ImageState::Copying { slot, .. } = s { Some(self.stages[*slot].fence) } else { None }).collect();
            if pending.is_empty() || timeout == 0 {
                return Err(if timeout == 0 { vk::Result::NOT_READY } else { vk::Result::TIMEOUT });
            }
            let left = Duration::from_nanos(timeout).saturating_sub(started.elapsed()).min(crate::FENCE_WAIT_TIMEOUT);
            // SAFETY: the slots' fences on the live game device.
            match unsafe { self.game.wait_for_fences(&pending, false, left.as_nanos() as u64) } {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => return Err(vk::Result::TIMEOUT),
                Err(e) => return Err(e),
            }
        };
        if semaphore != vk::Semaphore::null() || fence != vk::Fence::null() {
            let signals = [semaphore];
            let submit = [vk::SubmitInfo::default().signal_semaphores(if semaphore == vk::Semaphore::null() { &[] } else { &signals })];
            let _queue = self.signal.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            // SAFETY: the layer's own queue, under its lock; the game's semaphore and fence are unsignalled, as its
            // acquire requires.
            unsafe { self.game.queue_submit(self.signal.queue, if semaphore == vk::Semaphore::null() { &[] } else { &submit }, fence) }?;
        }
        inner.images[index] = ImageState::Acquired;
        inner.cursor = (index + 1) % inner.images.len();
        Ok(index as u32)
    }

    /// `vkReleaseSwapchainImagesEXT`: the images go back unpresented.
    pub(crate) fn release(&self, indices: &[u32]) {
        let mut inner = self.lock();
        for &i in indices {
            if let Some(s) = inner.images.get_mut(i as usize) {
                *s = ImageState::Free;
            }
        }
    }

    /// The game's present of image `index` on `queue` (of `family`): the copy into a staging slot is submitted
    /// behind `waits`, and the call returns; with no slot (or a retired or stale swapchain) the frame is skipped
    /// and only the waits are consumed. `present_fence` (`VK_EXT_swapchain_maintenance1`) is signalled behind it.
    ///
    /// # Safety
    /// `queue` is the game's, externally synchronized by its present; `waits` are its present's semaphores.
    pub(crate) unsafe fn present(&self, queue: vk::Queue, family: Option<u32>, index: u32, waits: &[vk::Semaphore], present_fence: vk::Fence) -> vk::Result {
        let mut inner = self.lock();
        let d = &self.game;
        let Some(&(image, _)) = self.images.get(index as usize) else { return vk::Result::ERROR_VALIDATION_FAILED_EXT };
        let status = self.shared.status();
        let live = !inner.retired && matches!(verdict(status), Verdict::Shown | Verdict::Suboptimal);
        // SAFETY: the game device is live; the pool and buffers are this swapchain's own.
        let pool_ok = live && family.is_some_and(|family| unsafe { self.ensure_pool(&mut inner, family) });
        let seqs = inner.slot_seq;
        let slot = if pool_ok { self.ring.claim(|k| seqs[k] == 0 || self.fence_done(k)) } else { None };
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let mut result = vk::Result::SUCCESS;
        match slot {
            Some(k) => {
                let cmd = inner.pool.as_ref().map(|(_, _, cmds)| cmds[k]).unwrap_or_default();
                let fence = self.stages[k].fence;
                // SAFETY: slot `k` is idle (its last copy finished), so its buffer and fence may be used again.
                let submitted = unsafe {
                    (|| -> ash::prelude::VkResult<()> {
                        d.reset_fences(&[fence])?;
                        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                        let (present, src) = (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                        let to_src = [layout_barrier(image, present, src, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ)];
                        d.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &to_src,
                        );
                        d.cmd_copy_image_to_buffer(cmd, image, src, self.stages[k].stage.buffer, &[whole(self.extent)]);
                        let back = [layout_barrier(image, src, present, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::empty())];
                        d.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::ALL_COMMANDS,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &back,
                        );
                        d.end_command_buffer(cmd)?;
                        let cmds = [cmd];
                        let submit = [vk::SubmitInfo::default().wait_semaphores(waits).wait_dst_stage_mask(&stages).command_buffers(&cmds)];
                        d.queue_submit(queue, &submit, fence)
                    })()
                };
                match submitted {
                    Ok(()) => {
                        let seq = self.ring.filled(k);
                        inner.slot_seq[k] = seq;
                        inner.images[index as usize] = ImageState::Copying { slot: k, seq };
                    }
                    Err(e) => {
                        self.ring.abandon(k);
                        inner.images[index as usize] = ImageState::Free;
                        result = e;
                    }
                }
            }
            None => {
                if !waits.is_empty() {
                    let submit = [vk::SubmitInfo::default().wait_semaphores(waits).wait_dst_stage_mask(&stages)];
                    // SAFETY: the game's queue, synchronized by its present: its semaphores are consumed as a present does.
                    if let Err(e) = unsafe { d.queue_submit(queue, &submit, vk::Fence::null()) } {
                        result = e;
                    }
                }
                inner.images[index as usize] = ImageState::Free;
            }
        }
        if present_fence != vk::Fence::null() {
            // SAFETY: as above; an empty submit signals the fence behind the queue's work.
            if let Err(e) = unsafe { d.queue_submit(queue, &[], present_fence) } {
                result = e;
            }
        }
        if result != vk::Result::SUCCESS {
            return result;
        }
        if inner.retired {
            return vk::Result::ERROR_OUT_OF_DATE_KHR;
        }
        match verdict(status) {
            Verdict::Shown => vk::Result::SUCCESS,
            Verdict::Suboptimal => vk::Result::SUBOPTIMAL_KHR,
            Verdict::Stale(r) => r,
            Verdict::Busy | Verdict::Failed => vk::Result::ERROR_OUT_OF_DATE_KHR,
        }
    }

    /// The pool and the slots' command buffers for `family`.
    unsafe fn ensure_pool(&self, inner: &mut Inner, family: u32) -> bool {
        if let Some((f, _, _)) = &inner.pool {
            // Another family would need the copies of the first to finish before their buffers go: never seen (a
            // game presents on one queue); such presents are skipped.
            return *f == family;
        }
        let d = &self.game;
        // SAFETY: valid create infos on the live game device.
        unsafe {
            let Ok(pool) = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            ) else {
                return false;
            };
            let info = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(SLOTS as u32);
            match crate::loader_data::allocate_commands(d, &info) {
                Ok(cmds) => {
                    inner.pool = Some((family, pool, cmds));
                    true
                }
                Err(_) => {
                    d.destroy_command_pool(pool, None);
                    false
                }
            }
        }
    }

    /// The game's HDR metadata for it: given to the real swapchain before its next present.
    pub(crate) fn set_hdr_metadata(&self, metadata: &vk::HdrMetadataEXT) {
        let mut m = *metadata;
        m.p_next = std::ptr::null();
        // SAFETY: the copy's chain is null, so no lifetime of the caller's is kept.
        let m = unsafe { std::mem::transmute::<vk::HdrMetadataEXT<'_>, vk::HdrMetadataEXT<'static>>(m) };
        *self.shared.hdr.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Metadata(m));
    }

    /// No more frames from this swapchain: the presenter is stopped and joined. Returns the second device's side
    /// the first time (the real swapchain and the presenter's resources), for the caller to destroy or replace.
    fn retire(&self) -> Option<SecondSide> {
        self.ring.close();
        let (handle, side) = {
            let mut inner = self.lock();
            inner.retired = true;
            (inner.presenter.take(), inner.second_side.take())
        };
        match handle {
            Some(h) => h.join().ok(),
            None => side,
        }
    }

    /// Retires it and frees the second device's side with the real swapchain: the game's surface is free again.
    pub(crate) fn release_second(&self) {
        if let Some(side) = self.retire() {
            // SAFETY: the presenter was joined.
            unsafe { side.destroy(&self.second, false) };
            self.second.busy.store(false, Ordering::Release);
        }
    }

    /// [`Self::release_second`], if `second` is the device that shows it.
    pub(crate) fn release_second_of(&self, second: &Arc<Second>) {
        if Arc::ptr_eq(&self.second, second) {
            self.release_second();
        }
    }

    /// Frees the game's side.
    ///
    /// # Safety
    /// The game destroys the swapchain: nothing of its still uses the images.
    unsafe fn free_game(&self) {
        let d = &self.game;
        let mut inner = self.lock();
        // Drain before free: the copies still running read the images and write the buffers.
        let pending: Vec<vk::Fence> = (0..SLOTS).filter(|&k| inner.slot_seq[k] != 0).map(|k| self.stages[k].fence).collect();
        // SAFETY: as the contract; the fences are this swapchain's own.
        unsafe {
            if !pending.is_empty()
                && crate::note_fence_wait(d.handle(), d.wait_for_fences(&pending, true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64), "present2 teardown")
                    .is_err()
            {
                crate::log!("[present2] the copies of a destroyed swapchain did not finish: its staging is left to the device's end");
                // The pages too: the GPU may still write them.
                std::mem::forget(std::mem::take(&mut *self.pages.lock().unwrap_or_else(std::sync::PoisonError::into_inner)));
                return;
            }
            if let Some((_, pool, _)) = inner.pool.take() {
                d.destroy_command_pool(pool, None);
            }
            for s in &self.stages {
                s.stage.destroy(d);
                d.destroy_fence(s.fence, None);
            }
            for (i, m) in &self.images {
                d.destroy_image(*i, None);
                d.free_memory(*m, None);
            }
        }
        // The buffers imported from them are gone on both devices (the second side went first): unmapped.
        self.pages.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
    }
}

/// `vkDestroySwapchainKHR` for one of the layer's: `true` when `swapchain` was one.
///
/// # Safety
/// The game's destroy: nothing of the swapchain is in use by it.
pub(crate) unsafe fn destroy(swapchain: vk::SwapchainKHR) -> bool {
    let Some(v) = lookup(swapchain) else { return false };
    VIRTUALS.write().unwrap_or_else(std::sync::PoisonError::into_inner).retain(|(h, _)| *h != swapchain);
    LIVE.fetch_sub(1, Ordering::AcqRel);
    v.release_second();
    // SAFETY: as the contract.
    unsafe { v.free_game() };
    true
}

/// The presenter: shows the newest frame of `ring` on the second device until the ring closes, then gives its side
/// back.
#[allow(clippy::too_many_arguments)]
unsafe fn present_loop(
    game: &ash::Device,
    second: &Second,
    ring: &Ring,
    shared: &Shared,
    side: SecondSide,
    sources: &[(usize, vk::Fence)],
    bytes: u64,
    extent: vk::Extent2D,
) -> SecondSide {
    let fns = ash::khr::swapchain::Device::new(&second.instance, &second.device);
    let d = &second.device;
    let mut said = Instant::now();
    let mut logged = 0u64;
    let mut presents = 0u64;
    loop {
        let (k, _seq) = match ring.take(Duration::from_millis(100)) {
            Take::Closed => break,
            Take::Idle => continue,
            Take::Frame(k, seq) => (k, seq),
        };
        // A swapchain the surface has left takes no more frames: the game recreates it.
        if !matches!(verdict(shared.status()), Verdict::Shown | Verdict::Suboptimal) {
            ring.done(k, false);
            continue;
        }
        let (src, copied) = sources[k];
        // The copy on the game's GPU.
        // SAFETY: slot `k` is the presenter's until `done`; its fence is the slot's own.
        if let Err(e) = crate::note_fence_wait(
            game.handle(),
            unsafe { game.wait_for_fences(&[copied], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64) },
            "present2 copy",
        ) {
            ring.done(k, false);
            shared.fail(second, format!("the game's GPU did not finish copying a frame out ({e:?})"));
            continue;
        }
        if let Some(Metadata(m)) = shared.hdr.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
            // SAFETY: the real swapchain, this thread's.
            unsafe { second.shower.set_hdr_metadata(side.swapchain, &m) };
        }
        let t = Instant::now();
        // A slot both devices import is the same memory: nothing to copy.
        if src as *const u8 != side.stages[k].ptr.cast_const() {
            // SAFETY: both mappings are `bytes` long and distinct; the game's copy finished and nothing reads the
            // second device's buffer (its last upload was waited for).
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, side.stages[k].ptr, bytes as usize) };
        }
        // SAFETY: the real swapchain and the presenter's own objects, on this thread only.
        let shown = unsafe {
            (|| -> Result<bool, vk::Result> {
                let (index, suboptimal) = fns.acquire_next_image(side.swapchain, 1_000_000_000, side.acquired, vk::Fence::null())?;
                let image = side.images[index as usize];
                d.reset_fences(&[side.fence])?;
                d.begin_command_buffer(side.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                let dst = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                let to_dst = [layout_barrier(image, vk::ImageLayout::UNDEFINED, dst, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)];
                d.cmd_pipeline_barrier(
                    side.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_dst,
                );
                d.cmd_copy_buffer_to_image(side.cmd, side.stages[k].buffer, image, dst, &[whole(extent)]);
                let to_present = [layout_barrier(image, dst, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty())];
                d.cmd_pipeline_barrier(
                    side.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &to_present,
                );
                d.end_command_buffer(side.cmd)?;
                let (waits, stages, cmds) = ([side.acquired], [vk::PipelineStageFlags::TRANSFER], [side.cmd]);
                let submit = [vk::SubmitInfo::default().wait_semaphores(&waits).wait_dst_stage_mask(&stages).command_buffers(&cmds)];
                d.queue_submit(second.queue, &submit, side.fence)?;
                // The upload is the link's time: waited for here, so the slot is free for the game once this frame
                // is done and the present below needs no semaphore of the layer's.
                crate::note_fence_wait(d.handle(), d.wait_for_fences(&[side.fence], true, crate::FENCE_WAIT_TIMEOUT.as_nanos() as u64), "present2 upload")?;
                shared.copy_us.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                shared.copies.fetch_add(1, Ordering::Relaxed);
                let (swapchains, indices) = ([side.swapchain], [index]);
                let info = vk::PresentInfoKHR::default().swapchains(&swapchains).image_indices(&indices);
                presents += 1;
                if crate::preupscale::offload_server::test_fail_at() == Some(presents) {
                    // The test's failure (`NEURAL_FORGE_OFFLOAD_TEST_FAIL`): the fall-back, seen in a game.
                    return Err(vk::Result::ERROR_UNKNOWN);
                }
                match second.shower.present(second.queue, &info) {
                    vk::Result::SUCCESS => Ok(suboptimal),
                    vk::Result::SUBOPTIMAL_KHR => Ok(true),
                    e => Err(e),
                }
            })()
        };
        match shown {
            Ok(suboptimal) => {
                ring.done(k, true);
                if suboptimal {
                    shared.note(vk::Result::SUBOPTIMAL_KHR);
                }
            }
            Err(e) => {
                ring.done(k, false);
                match verdict(e) {
                    Verdict::Stale(r) => shared.note(r),
                    Verdict::Busy => {}
                    _ => shared.fail(second, format!("showing a frame failed ({e:?})")),
                }
            }
        }
        if said.elapsed() >= Duration::from_secs(1) {
            said = Instant::now();
            let (n, us) = (shared.copies.swap(0, Ordering::Relaxed), shared.copy_us.swap(0, Ordering::Relaxed));
            if let Some(mean) = us.checked_div(n) {
                shared.last_copy_us.store(mean, Ordering::Relaxed);
            }
            let (shown, skipped) = ring.counts();
            let copy_ms = shared.last_copy_us.load(Ordering::Relaxed) as f64 / 1000.0;
            if second.failed().is_none() {
                second.publish(&showing_line(&second.name, shown, skipped, copy_ms));
            }
            if shown >= logged + 3600 {
                logged = shown;
                crate::log!("[present2] {}", showing_line(&second.name, shown, skipped, copy_ms));
            }
        }
    }
    let (shown, skipped) = ring.counts();
    crate::log!("[present2] the swapchain's presenter ended: {shown} frames shown, {skipped} skipped");
    side
}

// ---------------------------------------------------------------------------------------------------------------
// Commands that name a swapchain and are newer than the layer framework (it hands out the next layer's pointer for
// them): wrapped through `vkGetDeviceProcAddr`, so the driver never sees a handle of the layer's. Found with
// vkd3d-proton, which asks `VK_EXT_present_timing` about every swapchain: NVIDIA's driver hung on the layer's handle.

type PfnSwapchainU32 = unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, u32) -> vk::Result;
type PfnSwapchainOut = unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, *mut std::ffi::c_void, *mut u64) -> vk::Result;
type PfnInfoOut = unsafe extern "system" fn(vk::Device, *const std::ffi::c_void, *mut std::ffi::c_void) -> vk::Result;
type PfnSwapchainInfo = unsafe extern "system" fn(vk::Device, vk::SwapchainKHR, *const std::ffi::c_void) -> vk::Result;
type PfnRelease = unsafe extern "system" fn(vk::Device, *const vk::ReleaseSwapchainImagesInfoEXT) -> vk::Result;

/// The wrapped commands, by name: the index is the command's place in [`NEXT`]'s keys.
const NEWER: [&[u8]; 6] = [
    b"vkSetSwapchainPresentTimingQueueSizeEXT",
    b"vkGetSwapchainTimingPropertiesEXT",
    b"vkGetSwapchainTimeDomainPropertiesEXT",
    b"vkGetPastPresentationTimingEXT",
    b"vkWaitForPresent2KHR",
    b"vkReleaseSwapchainImagesKHR",
];

/// The next layer's pointer per device (its dispatch key) and command.
static NEXT: RwLock<Vec<(usize, usize, usize)>> = RwLock::new(Vec::new());

/// A device's dispatch key (what every handle of one device shares, whichever layer wrapped it).
unsafe fn dispatch_key(device: vk::Device) -> usize {
    // SAFETY: a dispatchable handle points at its dispatch table pointer.
    unsafe { *(device.as_raw() as *const usize) }
}

fn next_of(device: vk::Device, which: usize) -> Option<usize> {
    // SAFETY: the application's live device.
    let key = unsafe { dispatch_key(device) };
    NEXT.read().unwrap_or_else(std::sync::PoisonError::into_inner).iter().find(|(k, w, _)| *k == key && *w == which).map(|(_, _, f)| *f)
}

/// `vkGetDeviceProcAddr`'s result for `name`, wrapped when it is one of [`NEWER`] and a second GPU is asked for (`on`).
///
/// # Safety
/// `original` is the next layer's pointer for `name` on `device`.
pub(crate) unsafe fn intercept(device: vk::Device, name: &[u8], original: vk::PFN_vkVoidFunction, on: bool) -> vk::PFN_vkVoidFunction {
    let Some(next) = original else { return original };
    let Some(which) = NEWER.iter().position(|n| *n == name).filter(|_| on && device != vk::Device::null()) else { return original };
    let own: [usize; 6] = [
        set_timing_queue_size as PfnSwapchainU32 as usize,
        get_timing_properties as PfnSwapchainOut as usize,
        get_time_domains as PfnSwapchainOut as usize,
        get_past_timing as PfnInfoOut as usize,
        wait_for_present2 as PfnSwapchainInfo as usize,
        release_images as PfnRelease as usize,
    ];
    if next as usize == own[which] {
        return original;
    }
    // SAFETY: the application's live device.
    let key = unsafe { dispatch_key(device) };
    let mut all = NEXT.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    all.retain(|(k, w, _)| !(*k == key && *w == which));
    all.push((key, which, next as usize));
    // SAFETY: the wrapper has the command's signature; the pointer is only called as that command.
    Some(unsafe { std::mem::transmute::<usize, unsafe extern "system" fn()>(own[which]) })
}

unsafe extern "system" fn set_timing_queue_size(device: vk::Device, swapchain: vk::SwapchainKHR, size: u32) -> vk::Result {
    if lookup(swapchain).is_some() {
        return vk::Result::SUCCESS;
    }
    match next_of(device, 0) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnSwapchainU32>(f)(device, swapchain, size) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

/// The layer's swapchain has no refresh cycle to report: not ready, as a swapchain that has not presented yet.
unsafe extern "system" fn get_timing_properties(device: vk::Device, swapchain: vk::SwapchainKHR, out: *mut std::ffi::c_void, counter: *mut u64) -> vk::Result {
    if lookup(swapchain).is_some() {
        return vk::Result::NOT_READY;
    }
    match next_of(device, 1) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnSwapchainOut>(f)(device, swapchain, out, counter) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

/// No time domains: `VkSwapchainTimeDomainPropertiesEXT::timeDomainCount` (after `sType` and `pNext`) is 0.
unsafe extern "system" fn get_time_domains(device: vk::Device, swapchain: vk::SwapchainKHR, out: *mut std::ffi::c_void, counter: *mut u64) -> vk::Result {
    if lookup(swapchain).is_some() {
        if !out.is_null() {
            // SAFETY: the application's struct: sType (4, padded to 8), pNext (8), then the count.
            unsafe { *out.cast::<u8>().add(16).cast::<u32>() = 0 };
        }
        return vk::Result::SUCCESS;
    }
    match next_of(device, 2) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnSwapchainOut>(f)(device, swapchain, out, counter) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

/// No past presentation timings: `VkPastPresentationTimingInfoEXT::swapchain` is at 24, and
/// `VkPastPresentationTimingPropertiesEXT::presentationTimingCount` at 32.
unsafe extern "system" fn get_past_timing(device: vk::Device, info: *const std::ffi::c_void, out: *mut std::ffi::c_void) -> vk::Result {
    // SAFETY: the application's struct: sType, pNext, flags (4, padded), then the swapchain.
    if !info.is_null() && lookup(unsafe { *info.cast::<u8>().add(24).cast::<vk::SwapchainKHR>() }).is_some() {
        if !out.is_null() {
            // SAFETY: the application's struct: sType, pNext, two counters, then the count.
            unsafe { *out.cast::<u8>().add(32).cast::<u32>() = 0 };
        }
        return vk::Result::SUCCESS;
    }
    match next_of(device, 3) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnInfoOut>(f)(device, info, out) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

/// A present of the layer's swapchain is handed over at once and never waited for.
unsafe extern "system" fn wait_for_present2(device: vk::Device, swapchain: vk::SwapchainKHR, info: *const std::ffi::c_void) -> vk::Result {
    if lookup(swapchain).is_some() {
        return vk::Result::SUCCESS;
    }
    match next_of(device, 4) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnSwapchainInfo>(f)(device, swapchain, info) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

/// `vkReleaseSwapchainImagesKHR` (the EXT command's promoted name, with the same struct).
unsafe extern "system" fn release_images(device: vk::Device, info: *const vk::ReleaseSwapchainImagesInfoEXT) -> vk::Result {
    // SAFETY: the application's valid struct.
    if let Some(v) = unsafe { info.as_ref() }.and_then(|i| lookup(i.swapchain).map(|v| (v, i))).map(|(v, i)| {
        if i.image_index_count > 0 && !i.p_image_indices.is_null() {
            // SAFETY: the application's array of that many indices.
            v.release(unsafe { std::slice::from_raw_parts(i.p_image_indices, i.image_index_count as usize) });
        }
    }) {
        let () = v;
        return vk::Result::SUCCESS;
    }
    match next_of(device, 5) {
        // SAFETY: the next layer's pointer for this command.
        Some(f) => unsafe { std::mem::transmute::<usize, PfnRelease>(f)(device, info) },
        None => vk::Result::ERROR_EXTENSION_NOT_PRESENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_frame_wins_and_the_rest_count_as_skipped() {
        let ring = Ring::new();
        assert_eq!(ring.take(Duration::ZERO), Take::Idle);
        // Three presents while the presenter is away: each has a slot.
        let slots: Vec<usize> = (0..3).map(|_| ring.claim(|_| true).expect("a free slot")).collect();
        let seqs: Vec<u64> = slots.iter().map(|&k| ring.filled(k)).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
        // A fourth replaces the oldest waiting frame.
        let k = ring.claim(|_| true).expect("the oldest waiting frame's slot");
        assert_eq!(k, slots[0]);
        assert_eq!(ring.filled(k), 4);
        assert_eq!(ring.counts(), (0, 1));
        // The presenter takes the newest; the two older ones are dropped.
        assert_eq!(ring.take(Duration::ZERO), Take::Frame(slots[0], 4));
        assert_eq!(ring.counts(), (0, 3));
        // While it shows that one, two more presents have slots, a third replaces the older of them.
        let a = ring.claim(|_| true).unwrap();
        ring.filled(a);
        let b = ring.claim(|_| true).unwrap();
        ring.filled(b);
        assert!(a != slots[0] && b != slots[0] && a != b, "never the slot being shown");
        assert_eq!(ring.claim(|_| true), Some(a));
        ring.filled(a);
        ring.done(slots[0], true);
        assert_eq!(ring.counts(), (1, 4));
        assert_eq!(ring.take(Duration::ZERO), Take::Frame(a, 7));
        ring.done(a, false);
        assert_eq!(ring.counts(), (1, 6), "the older one dropped, and the one that could not be shown");
    }

    #[test]
    fn a_slot_the_gpu_still_writes_is_never_given_and_the_frame_is_skipped() {
        let ring = Ring::new();
        for _ in 0..3 {
            let k = ring.claim(|_| true).unwrap();
            ring.filled(k);
        }
        // Every waiting frame's copy is still running: the present has no slot and its own frame is skipped.
        assert_eq!(ring.claim(|_| false), None);
        assert_eq!(ring.counts(), (0, 1));
        // Only slot 2's copy finished: it is the one replaced, though not the oldest.
        assert_eq!(ring.claim(|k| k == 2), Some(2));
        // A claimed slot whose copy could not be submitted is free again.
        ring.abandon(2);
        assert_eq!(ring.claim(|k| k == 2), Some(2));
        // Closed: nothing more is taken or given.
        ring.close();
        assert_eq!(ring.claim(|_| true), None);
        assert_eq!(ring.take(Duration::from_secs(5)), Take::Closed);
    }

    #[test]
    fn a_waiting_presenter_wakes_for_a_frame_and_for_the_close() {
        let ring = Arc::new(Ring::new());
        let r = ring.clone();
        let t = std::thread::spawn(move || (r.take(Duration::from_secs(30)), r.take(Duration::from_secs(30))));
        let k = ring.claim(|_| true).unwrap();
        let seq = ring.filled(k);
        while ring.claim(|_| true).is_some_and(|j| {
            ring.abandon(j);
            ring.lock().slots[k] != SlotState::Showing
        }) {
            std::thread::yield_now();
        }
        ring.close();
        assert_eq!(t.join().unwrap(), (Take::Frame(k, seq), Take::Closed));
    }

    #[test]
    fn an_acquire_gives_a_free_image_or_one_whose_copy_finished() {
        use ImageState::*;
        let images = [Acquired, Copying { slot: 0, seq: 5 }, Free, Copying { slot: 1, seq: 6 }];
        assert_eq!(pick_image(&images, 0, |_, _| false), Some(2));
        assert_eq!(pick_image(&images, 3, |_, _| false), Some(2), "round the end");
        assert_eq!(pick_image(&images, 0, |slot, _| slot == 0), Some(1));
        assert_eq!(pick_image(&images, 3, |slot, seq| slot == 1 && seq == 6), Some(3));
        assert_eq!(pick_image(&[Acquired, Copying { slot: 0, seq: 1 }], 0, |_, _| false), None, "the game waits for a copy");
        assert_eq!(pick_image(&[], 0, |_, _| true), None);
    }

    #[test]
    fn the_real_swapchains_results_reach_the_game_as_a_recreate_or_a_fall_back() {
        assert_eq!(verdict(vk::Result::SUCCESS), Verdict::Shown);
        assert_eq!(verdict(vk::Result::SUBOPTIMAL_KHR), Verdict::Suboptimal);
        assert_eq!(verdict(vk::Result::ERROR_OUT_OF_DATE_KHR), Verdict::Stale(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert_eq!(verdict(vk::Result::ERROR_SURFACE_LOST_KHR), Verdict::Stale(vk::Result::ERROR_SURFACE_LOST_KHR));
        assert_eq!(verdict(vk::Result::TIMEOUT), Verdict::Busy);
        assert_eq!(verdict(vk::Result::ERROR_DEVICE_LOST), Verdict::Failed);
        // Stale stays; suboptimal never hides it.
        let shared = Shared::new();
        shared.note(vk::Result::SUBOPTIMAL_KHR);
        assert_eq!(shared.status(), vk::Result::SUBOPTIMAL_KHR);
        shared.note(vk::Result::ERROR_OUT_OF_DATE_KHR);
        shared.note(vk::Result::SUBOPTIMAL_KHR);
        assert_eq!(shared.status(), vk::Result::ERROR_OUT_OF_DATE_KHR);
    }

    #[test]
    fn the_status_lines_fit_the_channel() {
        let long = "NVIDIA GeForce RTX 5080 (10de:2c02) at 0000:05:00.0, driver 620.70.06, a very long description indeed";
        let line = showing_line(long, 123_456_789, 98_765_432, 123.4);
        assert!(line.starts_with("second GPU shows the frames: NVIDIA GeForce RTX 5080"), "{line}");
        assert!(line.contains("98765432 skipped (44%)") && line.ends_with("copy 123.4 ms"), "{line}");
        assert!(line.len() < neural_forge_protocol::REASON_BYTES, "{}", line.len());
        assert_eq!(showing_line("X", 0, 0, 0.0), "second GPU shows the frames: X; 0 shown, 0 skipped (0%), copy 0.0 ms");
        let not = not_showing_line(&"why ".repeat(100));
        assert!(not.starts_with("second GPU does not show the frames: why") && not.len() < neural_forge_protocol::REASON_BYTES);
    }

    /// The commands newer than the framework answer for the layer's swapchain themselves and forward any other.
    #[test]
    fn the_newer_swapchain_commands_never_pass_the_layers_handle_on() {
        unsafe extern "system" fn next_timing(_: vk::Device, _: vk::SwapchainKHR, _: *mut std::ffi::c_void, _: *mut u64) -> vk::Result {
            vk::Result::ERROR_SURFACE_LOST_KHR
        }
        // A stand-in device handle: a pointer to its dispatch key.
        let key: usize = 0x1234;
        let device = vk::Device::from_raw(std::ptr::from_ref(&key) as u64);
        let next = unsafe { std::mem::transmute::<PfnSwapchainOut, unsafe extern "system" fn()>(next_timing) };
        unsafe {
            assert_eq!(intercept(device, b"vkGetSwapchainTimingPropertiesEXT", Some(next), false).map(|f| f as usize), Some(next as usize), "off: untouched");
            assert_eq!(intercept(device, b"vkQueueSubmit", Some(next), true).map(|f| f as usize), Some(next as usize), "another command");
            assert!(intercept(device, b"vkGetSwapchainTimingPropertiesEXT", None, true).is_none(), "an absent command stays absent");
            let wrapped = intercept(device, b"vkGetSwapchainTimingPropertiesEXT", Some(next), true).expect("wrapped");
            assert_ne!(wrapped as usize, next as usize);
            let wrapped = std::mem::transmute::<unsafe extern "system" fn(), PfnSwapchainOut>(wrapped);
            // Not the layer's: the next layer answers.
            let other = vk::SwapchainKHR::from_raw(0xdead);
            assert_eq!(wrapped(device, other, std::ptr::null_mut(), std::ptr::null_mut()), vk::Result::ERROR_SURFACE_LOST_KHR);
            // The struct offsets the answers are written at.
            let mut domains = [0xffff_ffffu32; 8];
            assert_eq!(
                get_time_domains(device, other, domains.as_mut_ptr().cast(), std::ptr::null_mut()),
                vk::Result::ERROR_EXTENSION_NOT_PRESENT,
                "never asked for"
            );
        }
    }

    #[test]
    fn only_formats_with_a_known_texel_are_carried() {
        assert_eq!(texel_bytes(vk::Format::B8G8R8A8_UNORM), Some(4));
        assert_eq!(texel_bytes(vk::Format::A2B10G10R10_UNORM_PACK32), Some(4));
        assert_eq!(texel_bytes(vk::Format::R16G16B16A16_SFLOAT), Some(8));
        assert_eq!(texel_bytes(vk::Format::R5G6B5_UNORM_PACK16), None);
    }

    // --- with two devices and a real swapchain (lavapipe's headless surface) ---

    /// A plain present on the second device, which first reads the image's first texel (what arrived), can be held
    /// (a slow second GPU) and can fail.
    struct Plain {
        fns: ash::khr::swapchain::Device,
        device: ash::Device,
        stage: Stage,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        seen: Arc<Mutex<Vec<[u8; 4]>>>,
        hold: Arc<Mutex<()>>,
        fail: Arc<AtomicBool>,
        metadata: Arc<AtomicUsize>,
    }

    // SAFETY: handles, and a mapped pointer into memory it owns, used by the presenter thread only.
    unsafe impl Send for Plain {}
    unsafe impl Sync for Plain {}

    impl Shower for Plain {
        unsafe fn create_swapchain(&self, info: &vk::SwapchainCreateInfoKHR) -> ash::prelude::VkResult<vk::SwapchainKHR> {
            let mut info = *info;
            info.image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
            unsafe { self.fns.create_swapchain(&info, None) }
        }

        unsafe fn present(&self, queue: vk::Queue, info: &vk::PresentInfoKHR) -> vk::Result {
            let _held = self.hold.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.fail.load(Ordering::Acquire) {
                return vk::Result::ERROR_DEVICE_LOST;
            }
            unsafe {
                let (swapchain, index) = (*info.p_swapchains, *info.p_image_indices);
                let image = self.fns.get_swapchain_images(swapchain).unwrap()[index as usize];
                let d = &self.device;
                d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let (present, src) = (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                let a = [layout_barrier(image, present, src, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_READ)];
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &a,
                );
                d.cmd_copy_image_to_buffer(self.cmd, image, src, self.stage.buffer, &[whole(vk::Extent2D { width: 1, height: 1 })]);
                let b = [layout_barrier(image, src, present, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::empty())];
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &b,
                );
                d.end_command_buffer(self.cmd).unwrap();
                let cmds = [self.cmd];
                d.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], vk::Fence::null()).unwrap();
                d.queue_wait_idle(queue).unwrap();
                let mut texel = [0u8; 4];
                std::ptr::copy_nonoverlapping(self.stage.ptr, texel.as_mut_ptr(), 4);
                self.seen.lock().unwrap().push(texel);
                match self.fns.queue_present(queue, info) {
                    Ok(false) => vk::Result::SUCCESS,
                    Ok(true) => vk::Result::SUBOPTIMAL_KHR,
                    Err(e) => e,
                }
            }
        }

        unsafe fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
            unsafe { self.fns.destroy_swapchain(swapchain, None) }
        }

        unsafe fn set_hdr_metadata(&self, _swapchain: vk::SwapchainKHR, _metadata: &vk::HdrMetadataEXT) {
            self.metadata.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct Rig {
        _entry: ash::Entry,
        instance: ash::Instance,
        physical: vk::PhysicalDevice,
        game: ash::Device,
        queue: vk::Queue,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        semaphores: [vk::Semaphore; 2],
        second: Arc<Second>,
        surface: vk::SurfaceKHR,
        seen: Arc<Mutex<Vec<[u8; 4]>>>,
        hold: Arc<Mutex<()>>,
        fail: Arc<AtomicBool>,
        metadata: Arc<AtomicUsize>,
        other: (ash::Device, vk::CommandPool, vk::Buffer, vk::DeviceMemory),
        import: bool,
    }

    const EXTENT: vk::Extent2D = vk::Extent2D { width: 64, height: 48 };

    impl Rig {
        /// `None` without a Vulkan device that has a headless surface and swapchains (lavapipe has both).
        fn new() -> Option<Self> {
            Self::with(false)
        }

        /// With `import`, both devices have `VK_EXT_external_memory_host` (`None` where it is not offered).
        fn with(import: bool) -> Option<Self> {
            unsafe {
                let entry = ash::Entry::load().ok()?;
                let offered = entry.enumerate_instance_extension_properties(None).ok()?;
                let has = |n: &std::ffi::CStr| offered.iter().any(|e| e.extension_name_as_c_str() == Ok(n));
                if !has(ash::khr::surface::NAME) || !has(ash::ext::headless_surface::NAME) {
                    return None;
                }
                let names = [ash::khr::surface::NAME.as_ptr(), ash::ext::headless_surface::NAME.as_ptr()];
                let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
                let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&names), None).ok()?;
                let physical = *instance.enumerate_physical_devices().ok()?.first()?;
                let surface = ash::ext::headless_surface::Instance::new(&entry, &instance)
                    .create_headless_surface(&vk::HeadlessSurfaceCreateInfoEXT::default(), None)
                    .ok()?;
                let fns = SurfaceFns::load(entry.static_fn().get_instance_proc_addr, instance.handle())?;
                let mut supported = vk::FALSE;
                let _ = (fns.support)(physical, 0, surface, &mut supported);
                if supported == vk::FALSE {
                    return None;
                }
                if import
                    && !instance
                        .enumerate_device_extension_properties(physical)
                        .is_ok_and(|e| e.iter().any(|x| x.extension_name_as_c_str() == Ok(crate::EXTERNAL_MEMORY_HOST_EXTENSION)))
                {
                    return None;
                }
                let device = || {
                    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(0).queue_priorities(&[1.0])];
                    let mut exts = vec![ash::khr::swapchain::NAME.as_ptr()];
                    if import {
                        exts.push(crate::EXTERNAL_MEMORY_HOST_EXTENSION.as_ptr());
                    }
                    instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&exts), None).ok()
                };
                let (game, other) = (device()?, device()?);
                let queue = game.get_device_queue(0, 0);
                let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(0).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                let pool = game.create_command_pool(&pool_info, None).ok()?;
                let cmd = game.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).ok()?[0];
                let semaphores = [game.create_semaphore(&Default::default(), None).ok()?, game.create_semaphore(&Default::default(), None).ok()?];
                let other_pool = other.create_command_pool(&pool_info, None).ok()?;
                let other_cmd =
                    other.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(other_pool).command_buffer_count(1)).ok()?[0];
                let stage = Stage::new(&other, &instance, physical, 4, vk::BufferUsageFlags::TRANSFER_DST, true).ok()?;
                let (seen, hold, fail, metadata) = (Arc::default(), Arc::default(), Arc::default(), Arc::default());
                let other_parts = (other.clone(), other_pool, stage.buffer, stage.memory);
                let shower = Plain {
                    fns: ash::khr::swapchain::Device::new(&instance, &other),
                    device: other.clone(),
                    stage,
                    pool: other_pool,
                    cmd: other_cmd,
                    seen: Arc::clone(&seen),
                    hold: Arc::clone(&hold),
                    fail: Arc::clone(&fail),
                    metadata: Arc::clone(&metadata),
                };
                let _ = shower.pool;
                let other_queue = other.get_device_queue(0, 0);
                let second = Arc::new(Second::new(
                    other,
                    instance.clone(),
                    physical,
                    other_queue,
                    0,
                    fns,
                    Box::new(shower),
                    "the test's second device".into(),
                    import,
                    None,
                ));
                Some(Self {
                    _entry: entry,
                    instance,
                    physical,
                    game,
                    queue,
                    pool,
                    cmd,
                    semaphores,
                    second,
                    surface,
                    seen,
                    hold,
                    fail,
                    metadata,
                    other: other_parts,
                    import,
                })
            }
        }

        fn info(&self, old: vk::SwapchainKHR) -> vk::SwapchainCreateInfoKHR<'static> {
            vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(4)
                .image_format(vk::Format::B8G8R8A8_UNORM)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(EXTENT)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .old_swapchain(old)
        }

        fn create(&self, info: &vk::SwapchainCreateInfoKHR, old: Option<&Arc<Virtual>>) -> Result<Arc<Virtual>, String> {
            let game = GameDevice {
                device: &self.game,
                instance: &self.instance,
                physical: self.physical,
                // The test's one queue, used by one thread: the game's and the layer's own.
                signal: Signal { queue: self.queue, lock: Arc::default() },
                import: self.import,
            };
            unsafe { Virtual::create(&game, &self.second, info, old) }.map(|h| lookup(h).expect("registered"))
        }

        fn handle(v: &Arc<Virtual>) -> vk::SwapchainKHR {
            vk::SwapchainKHR::from_raw(Arc::as_ptr(v) as u64)
        }

        /// The game's frame: acquire, clear the image to `shade`, present. Returns the present's result.
        fn frame(&self, v: &Virtual, shade: u8) -> vk::Result {
            unsafe {
                let d = &self.game;
                let index = match v.acquire(5_000_000_000, self.semaphores[0], vk::Fence::null()) {
                    Ok(i) => i,
                    Err(e) => return e,
                };
                let image = v.images()[index as usize];
                d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
                let dst = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                let a = [layout_barrier(image, vk::ImageLayout::UNDEFINED, dst, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)];
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &a,
                );
                let colour = vk::ClearColorValue { float32: [f32::from(shade) / 255.0, 0.0, 0.0, 1.0] };
                d.cmd_clear_color_image(self.cmd, image, dst, &colour, &[color_range()]);
                let b = [layout_barrier(image, dst, vk::ImageLayout::PRESENT_SRC_KHR, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty())];
                d.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &b,
                );
                d.end_command_buffer(self.cmd).unwrap();
                let (waits, stages, cmds, signals) = ([self.semaphores[0]], [vk::PipelineStageFlags::TRANSFER], [self.cmd], [self.semaphores[1]]);
                let submit =
                    [vk::SubmitInfo::default().wait_semaphores(&waits).wait_dst_stage_mask(&stages).command_buffers(&cmds).signal_semaphores(&signals)];
                d.queue_submit(self.queue, &submit, vk::Fence::null()).unwrap();
                let r = v.present(self.queue, Some(0), index, &[self.semaphores[1]], vk::Fence::null());
                // The test's one command buffer is recorded again for the next frame.
                d.queue_wait_idle(self.queue).unwrap();
                r
            }
        }

        fn wait_for(&self, v: &Virtual, frames: u64) {
            let t = Instant::now();
            while {
                let (shown, skipped) = v.ring.counts();
                shown + skipped < frames
            } {
                assert!(t.elapsed() < Duration::from_secs(20), "the presenter did not account for {frames} frames: {:?}", v.ring.counts());
                std::thread::sleep(Duration::from_millis(2));
            }
        }

        /// The shades shown so far (the red channel of each shown frame's first texel; BGRA).
        fn shades(&self) -> Vec<u8> {
            self.seen.lock().unwrap().iter().map(|t| t[2]).collect()
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            unsafe {
                let _ = self.game.device_wait_idle();
                for s in self.semaphores {
                    self.game.destroy_semaphore(s, None);
                }
                self.game.destroy_command_pool(self.pool, None);
                self.game.destroy_device(None);
                let (other, pool, buffer, memory) = &self.other;
                let _ = other.device_wait_idle();
                other.destroy_command_pool(*pool, None);
                other.destroy_buffer(*buffer, None);
                other.free_memory(*memory, None);
                other.destroy_device(None);
                ash::khr::surface::Instance::new(&self._entry, &self.instance).destroy_surface(self.surface, None);
                self.instance.destroy_instance(None);
            }
        }
    }

    #[test]
    fn every_frame_reaches_the_second_devices_swapchain_unchanged_when_it_keeps_up() {
        let Some(rig) = Rig::new() else { return };
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        assert!(v.images().len() >= 4);
        v.set_hdr_metadata(&vk::HdrMetadataEXT::default());
        for n in 0..8u64 {
            assert_eq!(rig.frame(&v, 10 + 20 * n as u8), vk::Result::SUCCESS);
            rig.wait_for(&v, n + 1);
        }
        assert_eq!(rig.shades(), (0..8).map(|n| 10 + 20 * n).collect::<Vec<u8>>());
        assert_eq!(v.ring.counts(), (8, 0));
        assert_eq!(v.shared_slots(), 0, "without host-memory import on both devices each slot is copied once");
        assert_eq!(rig.metadata.load(Ordering::Relaxed), 1, "the game's HDR metadata reached the real swapchain once");
        assert!(rig.second.failed().is_none());
        assert!(unsafe { destroy(Rig::handle(&v)) });
        assert!(lookup(Rig::handle(&v)).is_none());
        assert!(!rig.second.busy.load(Ordering::Acquire));
    }

    /// Where both devices import host memory, every slot is the same pages on both and the frames arrive the same,
    /// with nothing copied by the CPU.
    #[test]
    fn frames_cross_through_host_memory_both_devices_import() {
        let Some(rig) = Rig::with(true) else { return };
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        assert_eq!(v.shared_slots(), SLOTS);
        for n in 0..9u64 {
            assert_eq!(rig.frame(&v, 10 + 20 * n as u8), vk::Result::SUCCESS);
            rig.wait_for(&v, n + 1);
        }
        assert_eq!(rig.shades(), (0..9).map(|n| 10 + 20 * n).collect::<Vec<u8>>());
        assert!(unsafe { destroy(Rig::handle(&v)) });
        // Another swapchain gets and frees its own pages.
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("another");
        assert_eq!(v.shared_slots(), SLOTS);
        assert!(unsafe { destroy(Rig::handle(&v)) });
    }

    #[test]
    fn a_slow_second_device_skips_frames_and_never_holds_the_game() {
        let Some(rig) = Rig::new() else { return };
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        let held = rig.hold.lock().unwrap();
        let t = Instant::now();
        for n in 0..12u8 {
            assert_eq!(rig.frame(&v, 10 + 20 * n), vk::Result::SUCCESS, "frame {n}");
        }
        let took = t.elapsed();
        drop(held);
        rig.wait_for(&v, 12);
        let (shown, skipped) = v.ring.counts();
        let shades = rig.shades();
        assert_eq!(shown as usize, shades.len());
        assert!(shown <= 3 && skipped >= 9, "{shown} shown, {skipped} skipped in {took:?}");
        assert_eq!(*shades.last().unwrap(), 10 + 20 * 11, "the newest frame is the one shown last: {shades:?}");
        assert!(shades.windows(2).all(|w| w[0] < w[1]), "never an older frame after a newer one: {shades:?}");
        assert!(unsafe { destroy(Rig::handle(&v)) });
    }

    #[test]
    fn a_recreated_swapchain_replaces_the_real_one_and_the_old_one_is_out_of_date() {
        let Some(rig) = Rig::new() else { return };
        let old = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        assert_eq!(rig.frame(&old, 50), vk::Result::SUCCESS);
        rig.wait_for(&old, 1);
        let mut info = rig.info(Rig::handle(&old));
        info.image_extent = vk::Extent2D { width: 96, height: 64 };
        let new = rig.create(&info, Some(&old)).expect("the replacement");
        // The old one takes no more frames; what the game still holds of it can be released and destroyed.
        assert_eq!(old.acquire(0, vk::Semaphore::null(), vk::Fence::null()), Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert!(unsafe { destroy(Rig::handle(&old)) });
        assert_eq!(new.extent.width, 96);
        assert_eq!(rig.frame(&new, 90), vk::Result::SUCCESS);
        rig.wait_for(&new, 1);
        assert_eq!(rig.shades(), vec![50, 90]);
        // A second swapchain beside a live one is refused (one presenter has the second device's queue).
        let beside = rig.create(&rig.info(vk::SwapchainKHR::null()), None);
        assert!(beside.is_err_and(|why| why.contains("already shows")));
        assert!(unsafe { destroy(Rig::handle(&new)) });
    }

    #[test]
    fn a_failure_on_the_second_device_asks_the_game_for_a_new_swapchain_of_its_own() {
        let Some(rig) = Rig::new() else { return };
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        assert_eq!(rig.frame(&v, 30), vk::Result::SUCCESS);
        rig.wait_for(&v, 1);
        rig.fail.store(true, Ordering::Release);
        assert_eq!(rig.frame(&v, 60), vk::Result::SUCCESS, "handed over before the failure is known");
        rig.wait_for(&v, 2);
        assert!(rig.second.failed().is_some_and(|why| why.contains("ERROR_DEVICE_LOST")));
        assert_eq!(v.shared.status(), vk::Result::ERROR_OUT_OF_DATE_KHR);
        // The game's next acquire is told to recreate; a present of an image it still holds too, its waits consumed.
        assert_eq!(v.acquire(0, vk::Semaphore::null(), vk::Fence::null()), Err(vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert_eq!(rig.shades(), vec![30], "nothing was shown after the failure");
        // The recreate is refused, and leaves the surface free: a swapchain of the game's own device is made on it.
        let again = rig.create(&rig.info(Rig::handle(&v)), Some(&v));
        assert!(again.is_err_and(|why| why.contains("given up earlier")));
        assert!(!rig.second.busy.load(Ordering::Acquire));
        let fns = ash::khr::swapchain::Device::new(&rig.instance, &rig.game);
        let own = unsafe { fns.create_swapchain(&rig.info(vk::SwapchainKHR::null()), None) }.expect("the game's own swapchain on the freed surface");
        unsafe { fns.destroy_swapchain(own, None) };
        assert!(unsafe { destroy(Rig::handle(&v)) });
    }

    #[test]
    fn a_swapchain_the_second_device_cannot_show_is_refused_with_nothing_left_behind() {
        let Some(rig) = Rig::new() else { return };
        let mut info = rig.info(vk::SwapchainKHR::null());
        info.image_format = vk::Format::R5G6B5_UNORM_PACK16;
        assert!(rig.create(&info, None).is_err_and(|why| why.contains("format")));
        let mut info = rig.info(vk::SwapchainKHR::null());
        info.image_color_space = vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        info.image_format = vk::Format::A2B10G10R10_UNORM_PACK32;
        assert!(rig.create(&info, None).is_err_and(|why| why.contains("does not offer")));
        assert!(!rig.second.busy.load(Ordering::Acquire));
        // And the next one that can be shown is.
        let v = rig.create(&rig.info(vk::SwapchainKHR::null()), None).expect("the swapchain");
        assert!(unsafe { destroy(Rig::handle(&v)) });
    }
}
