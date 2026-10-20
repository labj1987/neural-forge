//! The semaphores the layer's presents wait on in place of (or beside) the application's, one per
//! swapchain image, and what tells when the last wait on one of them has completed.
//!
//! Presentation completion is associated with an acquired image, not a command buffer slot: an image's
//! semaphore is signalled again only for that image's next present, which cannot come before the image
//! was acquired again. When its swapchain is destroyed the semaphore is retired, and destroyed once the
//! last wait on it is known to be complete ([`LastUse`]):
//!
//! - [`Tracking::PresentFence`]: the layer chained a present fence of its own
//!   (`VK_KHR_swapchain_maintenance1`/`VK_EXT_swapchain_maintenance1`) to the present, and its
//!   [`Ticket`]'s fence was seen signalled.
//! - [`Tracking::AppFenced`]: the application chained its own present fence for the swapchain (DXVK and
//!   vkd3d-proton do on every present). That one stays the application's, and the layer adds nothing to
//!   the present: an application that fences its presents waits for those fences before it destroys
//!   the swapchain (the spec's `vkDestroySwapchainKHR` and `VkSwapchainPresentFenceInfoKHR`; DXVK's
//!   `Presenter::destroySwapchain` and vkd3d-proton's `dxgi_vk_swap_chain_drain_swapchain_fences` do),
//!   so the semaphores are free once the application destroys it.
//! - Otherwise ([`LastUse::Unknown`]) the semaphore is kept until device teardown: Vulkan gives no
//!   earlier point at which a wait by such a present is known to be complete.
//!
//! No present gets a submission of the layer's for this: one more batch on the present's queue per
//! present (signalling the application's wait semaphores again in the present's place) cost GTA V
//! Enhanced a fifth of its frame rate for the rest of the session after the loading screens it ran on
//! (3.8.3 and 3.8.4).
use ash::vk;
use std::collections::HashMap;

/// `VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR`, which `_EXT` aliases (the two structures are
/// the same).
pub const PRESENT_FENCE_INFO: vk::StructureType = vk::StructureType::SWAPCHAIN_PRESENT_FENCE_INFO_EXT;

/// What is known about the last wait on a present semaphore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastUse {
    /// Never waited on, or every wait known complete.
    Done,
    /// The last wait completes before this ticket's fence signals.
    Ticket(Ticket),
    /// Waited on by a present the application fenced for this semaphore's swapchain: complete once the
    /// application destroyed that swapchain, which is when the semaphore is retired.
    AppFenced,
    /// Waited on by a present without a fence: complete only once the device is idle.
    Unknown,
}

/// A fence the layer handed to a present or to its own batch, by a number never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket(u64);

#[derive(Clone, Copy, Debug)]
struct Entry {
    semaphore: vk::Semaphore,
    last: LastUse,
}

#[derive(Default)]
pub struct PresentSemaphores {
    active: HashMap<vk::Image, Entry>,
    retired: Vec<Entry>,
}

impl PresentSemaphores {
    pub fn get(&mut self, image: vk::Image, create: impl FnOnce() -> Option<vk::Semaphore>) -> Option<vk::Semaphore> {
        if let Some(entry) = self.active.get(&image) {
            return Some(entry.semaphore);
        }
        let semaphore = create()?;
        self.active.insert(image, Entry { semaphore, last: LastUse::Done });
        Some(semaphore)
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty() && self.retired.is_empty()
    }

    /// Records the wait the present just made (or the layer's batch in its place) on `semaphore`, one
    /// of this set's active ones. A later wait always replaces an earlier one: the semaphore is
    /// signalled again only after its previous wait completed.
    pub fn note(&mut self, semaphore: vk::Semaphore, last: LastUse) {
        if let Some(entry) = self.active.values_mut().chain(self.retired.iter_mut()).find(|e| e.semaphore == semaphore) {
            entry.last = last;
        }
    }

    pub fn retire(&mut self, images: &[vk::Image]) {
        for image in images {
            if let Some(entry) = self.active.remove(image) {
                self.retired.push(entry);
            }
        }
    }

    /// Retired semaphores still alive.
    pub fn retired(&self) -> usize {
        self.retired.len()
    }

    /// Of those, the ones kept until device teardown ([`LastUse::Unknown`]).
    pub fn kept_until_teardown(&self) -> usize {
        self.retired.iter().filter(|e| e.last == LastUse::Unknown).count()
    }

    /// Destroys every retired semaphore whose last wait is known complete (its ticket's fence was seen
    /// signalled by [`Tickets::poll`]), and says how many.
    ///
    /// # Safety
    /// The semaphores and `tickets`' fences are `device`'s; every wait on a retired semaphore was noted.
    pub unsafe fn reclaim(&mut self, device: &ash::Device, tickets: &Tickets) -> usize {
        let before = self.retired.len();
        self.retired.retain(|entry| {
            let free = match entry.last {
                LastUse::Done => true,
                LastUse::Ticket(ticket) => tickets.complete(ticket),
                // Retired: the application destroyed the swapchain, after its present fences signalled.
                LastUse::AppFenced => true,
                LastUse::Unknown => false,
            };
            if free {
                // SAFETY: retired (no present can signal or wait on it again) and its last wait is
                // complete.
                unsafe { device.destroy_semaphore(entry.semaphore, None) };
            }
            !free
        });
        before - self.retired.len()
    }

    /// # Safety
    /// Only at device teardown after queue completion and application swapchain
    /// destruction. No present may still reference these semaphores.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for entry in self.active.values().chain(self.retired.iter()) {
            unsafe { device.destroy_semaphore(entry.semaphore, None) };
        }
    }
}

/// The fences behind [`Ticket`]s, one per present the layer tracks: pending until seen signalled, then
/// reset and reused.
#[derive(Default)]
pub struct Tickets {
    next: u64,
    pending: Vec<(Ticket, vk::Fence)>,
    /// Fences of presents that failed in a way that may leave them unsignalled for good (out of memory,
    /// device lost): never polled again, never complete, destroyed at teardown.
    stranded: Vec<(Ticket, vk::Fence)>,
    spare: Vec<vk::Fence>,
}

impl Tickets {
    /// No fence was ever made.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.stranded.is_empty() && self.spare.is_empty()
    }

    /// Whether a fence is still waited for.
    #[cfg(test)]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// An unsignalled fence for one present: a spare one, or a new one. Given to a call, it becomes a
    /// [`Ticket`] ([`Self::submitted`], [`Self::stranded`]); given to none, it goes back ([`Self::unused`]).
    ///
    /// # Safety
    /// `device` is the one every fence here belongs to.
    pub unsafe fn take(&mut self, device: &ash::Device) -> Option<vk::Fence> {
        match self.spare.pop() {
            Some(fence) => Some(fence),
            // SAFETY: plain creation.
            None => unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.ok(),
        }
    }

    fn ticket(&mut self) -> Ticket {
        self.next += 1;
        Ticket(self.next)
    }

    /// The fence was given to a present or a submit that was made: its ticket, complete once the fence
    /// is seen signalled.
    pub fn submitted(&mut self, fence: vk::Fence) -> Ticket {
        let ticket = self.ticket();
        self.pending.push((ticket, fence));
        ticket
    }

    /// The fence was given to nothing (the call was not made): reused as it is, unsignalled.
    pub fn unused(&mut self, fence: vk::Fence) {
        self.spare.push(fence);
    }

    /// The fence was given to a present that may never signal it: its ticket never completes.
    pub fn stranded(&mut self, fence: vk::Fence) -> Ticket {
        let ticket = self.ticket();
        self.stranded.push((ticket, fence));
        ticket
    }

    /// Whether the ticket's fence was seen signalled.
    pub fn complete(&self, ticket: Ticket) -> bool {
        ticket.0 != 0 && ticket.0 <= self.next && !self.pending.iter().chain(&self.stranded).any(|&(t, _)| t == ticket)
    }

    /// Looks at every pending fence once (`vkGetFenceStatus`, no wait): a signalled one completes its
    /// ticket and is reset for reuse.
    ///
    /// # Safety
    /// `device` is the one every fence here belongs to.
    pub unsafe fn poll(&mut self, device: &ash::Device) {
        let mut k = 0;
        while k < self.pending.len() {
            let (_, fence) = self.pending[k];
            // SAFETY: a live fence of `device`.
            if unsafe { device.get_fence_status(fence) } == Ok(true) {
                self.pending.swap_remove(k);
                // SAFETY: signalled, so no queue operation still uses it.
                if unsafe { device.reset_fences(&[fence]) }.is_ok() {
                    self.spare.push(fence);
                } else {
                    // Complete all the same (it was seen signalled); not reused.
                    // SAFETY: as above.
                    unsafe { device.destroy_fence(fence, None) };
                }
            } else {
                k += 1;
            }
        }
    }

    /// # Safety
    /// Only at device teardown, once the device is idle.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for &fence in self.pending.iter().chain(&self.stranded).map(|(_, f)| f).chain(&self.spare) {
            // SAFETY: per this function's contract.
            unsafe { device.destroy_fence(fence, None) };
        }
    }
}

/// How a present the layer adds semaphores to is tracked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tracking {
    /// The layer chains a `VkSwapchainPresentFenceInfoKHR` of its own: the device has
    /// `swapchainMaintenance1` enabled and the application chained none.
    PresentFence,
    /// The application chained its own present fence for the swapchain the layer worked on
    /// ([`LastUse::AppFenced`]); the present goes on with nothing more of the layer's.
    AppFenced,
    /// Neither (no `swapchainMaintenance1`, or the application chained present fences but none for that
    /// swapchain, so the layer's cannot be added): the semaphores stay until device teardown.
    Untracked,
}

/// The application's present fences, for the swapchain at one slot of a present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppFence {
    /// No `VkSwapchainPresentFenceInfoKHR` in the present's chain.
    None,
    /// One, with a fence for that swapchain.
    ForSlot,
    /// One, without a fence for that swapchain (its entry `VK_NULL_HANDLE`, or fewer entries).
    NotForSlot,
}

/// `present_fences`: the device has `swapchainMaintenance1` enabled.
pub fn tracking(present_fences: bool, app: AppFence) -> Tracking {
    match app {
        AppFence::ForSlot => Tracking::AppFenced,
        AppFence::NotForSlot => Tracking::Untracked,
        AppFence::None if present_fences => Tracking::PresentFence,
        AppFence::None => Tracking::Untracked,
    }
}

/// The application's present fence for the swapchain at `slot` of `present_info` (read only).
///
/// # Safety
/// `present_info`'s chain is valid, and a `VkSwapchainPresentFenceInfoKHR` in it holds `swapchainCount`
/// fences.
pub unsafe fn app_present_fence(present_info: &vk::PresentInfoKHR, slot: usize) -> AppFence {
    let mut next = present_info.p_next.cast::<vk::BaseInStructure>();
    while !next.is_null() {
        // SAFETY: every node of a valid chain starts with `sType` and `pNext`; one of this type is the
        // fence structure.
        let node = unsafe { &*next };
        if node.s_type == PRESENT_FENCE_INFO {
            let info = unsafe { &*next.cast::<vk::SwapchainPresentFenceInfoEXT>() };
            if (slot as u32) >= info.swapchain_count || info.p_fences.is_null() {
                return AppFence::NotForSlot;
            }
            // SAFETY: `p_fences` holds `swapchain_count` fences.
            let fence = unsafe { *info.p_fences.add(slot) };
            return if fence == vk::Fence::null() { AppFence::NotForSlot } else { AppFence::ForSlot };
        }
        next = node.p_next;
    }
    AppFence::None
}

/// The layer's own present fences for a present of `swapchain_count` swapchains: `fence` for the one
/// at `slot`, none for the others. Chained in front of the application's chain (`p_next`), which then
/// holds none of its own ([`AppFence::None`]).
pub struct PresentFences {
    fences: Vec<vk::Fence>,
    pub info: vk::SwapchainPresentFenceInfoEXT,
}

impl PresentFences {
    pub fn new(swapchain_count: u32, slot: usize, fence: vk::Fence, p_next: *const std::ffi::c_void) -> Box<Self> {
        let mut fences = vec![vk::Fence::null(); swapchain_count as usize];
        fences[slot] = fence;
        let mut this = Box::new(Self { fences, info: vk::SwapchainPresentFenceInfoEXT::default() });
        this.info = vk::SwapchainPresentFenceInfoEXT {
            s_type: PRESENT_FENCE_INFO,
            p_next,
            swapchain_count,
            p_fences: this.fences.as_ptr(),
        };
        this
    }
}

/// Whether gamescope's WSI layer presents (`ENABLE_GAMESCOPE_WSI=1`, as DXVK reads it): see
/// [`present_fence_signals`].
pub fn gamescope_wsi() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("ENABLE_GAMESCOPE_WSI").is_some_and(|v| v == "1"));
    *ON
}

/// Whether the fence of a present that returned `result` is signalled once its queue operations are
/// done. They are still enqueued after `VK_ERROR_OUT_OF_DATE_KHR`, `VK_ERROR_SURFACE_LOST_KHR` and the
/// other swapchain errors (the spec's `vkQueuePresentKHR`), but not necessarily after running out of
/// memory or losing the device. `gamescope_wsi` (`ENABLE_GAMESCOPE_WSI=1`): gamescope's WSI layer does
/// not signal it when a queued present fails, so only a successful one counts (as in DXVK).
pub fn present_fence_signals(result: vk::Result, gamescope_wsi: bool) -> bool {
    if gamescope_wsi {
        return result.as_raw() >= 0;
    }
    !matches!(result, vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_DEVICE_LOST)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn semaphores(sems: &PresentSemaphores) -> Vec<u64> {
        sems.retired.iter().map(|e| e.semaphore.as_raw()).collect()
    }

    #[test]
    fn three_images_do_not_share_a_two_slot_semaphore_ring() {
        let mut sems = PresentSemaphores::default();
        let mut next = 10;
        for image in [2, 0, 1, 2, 0, 1, 2, 0] {
            let sem = sems.get(vk::Image::from_raw(image + 1), || { next += 1; Some(vk::Semaphore::from_raw(next)) }).unwrap();
            assert_eq!(sem.as_raw(), [12, 13, 11][image as usize]);
        }
        assert_eq!(next, 13);
        sems.retire(&[vk::Image::from_raw(1)]);
        assert_eq!(semaphores(&sems), [12]);
        // Recycled Vulkan image handles must not recycle an old present semaphore.
        assert_eq!(sems.get(vk::Image::from_raw(1), || Some(vk::Semaphore::from_raw(14))), Some(vk::Semaphore::from_raw(14)));
    }

    /// A later present's note replaces the earlier one, and the note goes with the semaphore into
    /// retirement.
    #[test]
    fn the_last_wait_is_what_retirement_keeps() {
        let mut sems = PresentSemaphores::default();
        let sem = sems.get(vk::Image::from_raw(1), || Some(vk::Semaphore::from_raw(5))).unwrap();
        sems.note(sem, LastUse::Unknown);
        sems.note(sem, LastUse::Ticket(Ticket(3)));
        sems.note(vk::Semaphore::from_raw(99), LastUse::Unknown);
        sems.retire(&[vk::Image::from_raw(1)]);
        assert_eq!(sems.retired[0].last, LastUse::Ticket(Ticket(3)));
        assert_eq!(sems.kept_until_teardown(), 0);
    }

    #[test]
    fn the_layer_chains_its_own_fence_only_where_the_application_chained_none() {
        assert_eq!(tracking(true, AppFence::None), Tracking::PresentFence);
        assert_eq!(tracking(true, AppFence::ForSlot), Tracking::AppFenced, "the application's fences stay its own");
        assert_eq!(tracking(false, AppFence::ForSlot), Tracking::AppFenced);
        assert_eq!(tracking(true, AppFence::NotForSlot), Tracking::Untracked, "no slot for the layer's: kept until teardown");
        assert_eq!(tracking(false, AppFence::None), Tracking::Untracked, "no swapchainMaintenance1: kept until teardown");

        // The layer's chain: its fence at the engaged swapchain's slot, nothing at the others', in front
        // of the application's chain.
        let app_next = vk::PresentRegionsKHR::default();
        let fences = PresentFences::new(3, 1, vk::Fence::from_raw(77), std::ptr::from_ref(&app_next).cast());
        assert_eq!(fences.info.s_type, vk::StructureType::from_raw(1_000_275_001));
        assert_eq!(fences.info.swapchain_count, 3);
        // SAFETY: `p_fences` holds `swapchain_count` fences, alive with `fences`.
        let set = unsafe { std::slice::from_raw_parts(fences.info.p_fences, 3) };
        assert_eq!(set, [vk::Fence::null(), vk::Fence::from_raw(77), vk::Fence::null()]);
        assert_eq!(fences.info.p_next, std::ptr::from_ref(&app_next).cast());

        // The application's own fence for a slot is found anywhere in its chain.
        let app_fences = [vk::Fence::null(), vk::Fence::from_raw(5)];
        let theirs = vk::SwapchainPresentFenceInfoEXT::builder().fences(&app_fences).build();
        let regions = vk::PresentRegionsKHR::default();
        let behind = vk::PresentRegionsKHR { p_next: std::ptr::from_ref(&theirs).cast(), ..Default::default() };
        let info = |next: *const std::ffi::c_void| vk::PresentInfoKHR { p_next: next, swapchain_count: 2, ..Default::default() };
        // SAFETY: each chain is of live locals.
        unsafe {
            assert_eq!(app_present_fence(&info(std::ptr::null()), 0), AppFence::None);
            assert_eq!(app_present_fence(&info(std::ptr::from_ref(&regions).cast()), 1), AppFence::None);
            assert_eq!(app_present_fence(&info(std::ptr::from_ref(&behind).cast()), 1), AppFence::ForSlot);
            assert_eq!(app_present_fence(&info(std::ptr::from_ref(&behind).cast()), 0), AppFence::NotForSlot, "its entry for the slot is null");
            assert_eq!(app_present_fence(&info(std::ptr::from_ref(&behind).cast()), 2), AppFence::NotForSlot, "fewer entries");
        }
    }

    #[test]
    fn a_present_fence_counts_where_its_queue_operations_were_enqueued() {
        for result in [vk::Result::SUCCESS, vk::Result::SUBOPTIMAL_KHR, vk::Result::ERROR_OUT_OF_DATE_KHR, vk::Result::ERROR_SURFACE_LOST_KHR, vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT] {
            assert!(present_fence_signals(result, false), "{result:?}");
        }
        for result in [vk::Result::ERROR_OUT_OF_HOST_MEMORY, vk::Result::ERROR_OUT_OF_DEVICE_MEMORY, vk::Result::ERROR_DEVICE_LOST] {
            assert!(!present_fence_signals(result, false), "{result:?}");
        }
        assert!(present_fence_signals(vk::Result::SUBOPTIMAL_KHR, true));
        assert!(!present_fence_signals(vk::Result::ERROR_OUT_OF_DATE_KHR, true), "gamescope's WSI layer leaves it unsignalled");
    }

    /// Real (lavapipe) fences and semaphores: a retired semaphore stays while its ticket's fence is
    /// unsignalled, and is destroyed once it signalled; one waited on by a present without a fence stays
    /// until teardown; the fence is reused. Under the validation layer, destroying a semaphore a pending
    /// wait still uses would be reported.
    #[test]
    fn retired_semaphores_are_freed_once_their_fence_signals_and_kept_until_then() {
        let Some((_entry, instance, _physical, device, queue, _family)) = crate::composition::gpu::test_device() else {
            eprintln!("present_sync test: no Vulkan loader/ICD in this environment, skipping");
            return;
        };
        let new_semaphore = || unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok() };
        let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut sems = PresentSemaphores::default();
        let mut tickets = Tickets::default();
        unsafe {
            // Image 1: a semaphore really signalled and waited on, the wait fenced by a ticket (as the
            // present fence or the layer's own batch would be).
            let waited = sems.get(vk::Image::from_raw(1), new_semaphore).unwrap();
            let first_fence = tickets.take(&device).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().signal_semaphores(&[waited]).build()], vk::Fence::null()).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(&[waited]).wait_dst_stage_mask(&stage).build()], first_fence).unwrap();
            let first = tickets.submitted(first_fence);
            sems.note(waited, LastUse::Ticket(first));
            // Image 2: a ticket whose fence nothing has signalled yet (a present still queued).
            let queued = sems.get(vk::Image::from_raw(2), new_semaphore).unwrap();
            let second_fence = tickets.take(&device).unwrap();
            let second = tickets.submitted(second_fence);
            sems.note(queued, LastUse::Ticket(second));
            // Image 3: waited on by a present without a fence.
            let unknown = sems.get(vk::Image::from_raw(3), new_semaphore).unwrap();
            sems.note(unknown, LastUse::Unknown);
            // Image 4: never waited on.
            sems.get(vk::Image::from_raw(4), new_semaphore).unwrap();
            // Image 5: waited on by a present the application fenced, which it waited for before it
            // destroyed the swapchain (here: the wait is a fenced submit, waited below).
            let app_fenced = sems.get(vk::Image::from_raw(5), new_semaphore).unwrap();
            let app_fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().signal_semaphores(&[app_fenced]).build()], vk::Fence::null()).unwrap();
            device.queue_submit(queue, &[vk::SubmitInfo::builder().wait_semaphores(&[app_fenced]).wait_dst_stage_mask(&stage).build()], app_fence).unwrap();
            sems.note(app_fenced, LastUse::AppFenced);
            device.wait_for_fences(&[app_fence], true, u64::MAX).unwrap();
            device.destroy_fence(app_fence, None);

            sems.retire(&[1, 2, 3, 4, 5].map(vk::Image::from_raw));
            assert_eq!(sems.retired(), 5);
            device.wait_for_fences(&[first_fence], true, u64::MAX).unwrap();
            tickets.poll(&device);
            assert!(tickets.complete(first) && !tickets.complete(second));
            assert_eq!(sems.reclaim(&device, &tickets), 3, "the fenced wait that completed, the application-fenced one, and the semaphore never waited on");
            assert_eq!(semaphores(&sems), [queued.as_raw(), unknown.as_raw()], "kept: an unsignalled fence, and no fence at all");
            tickets.poll(&device);
            assert_eq!(sems.reclaim(&device, &tickets), 0, "still unsignalled: still kept");

            // The queued present's operations complete: its fence signals.
            device.queue_submit(queue, &[], second_fence).unwrap();
            device.wait_for_fences(&[second_fence], true, u64::MAX).unwrap();
            tickets.poll(&device);
            assert!(!tickets.has_pending());
            assert_eq!(sems.reclaim(&device, &tickets), 1);
            assert_eq!(semaphores(&sems), [unknown.as_raw()]);
            assert_eq!(sems.kept_until_teardown(), 1, "a present without a fence: until teardown");
            // Both fences were reset for reuse.
            let reused = tickets.take(&device).unwrap();
            assert!(reused == first_fence || reused == second_fence);
            assert_eq!(device.get_fence_status(reused), Ok(false), "reset");
            // A present that may never signal its fence: its ticket never completes.
            let lost = tickets.stranded(reused);
            tickets.poll(&device);
            assert!(!tickets.complete(lost));

            device.queue_wait_idle(queue).unwrap();
            sems.destroy(&device);
            tickets.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// An X11 window for the WSI test, through libX11 loaded at run time; never mapped.
    struct X11Window {
        lib: *mut std::ffi::c_void,
        display: *mut std::ffi::c_void,
        window: std::ffi::c_ulong,
    }

    impl X11Window {
        /// On the X server `NEURAL_FORGE_TEST_DISPLAY` names (an Xvfb's, never the desktop's unasked).
        fn open() -> Option<Self> {
            use std::ffi::{c_char, c_int, c_uint, c_ulong, c_void, CStr, CString};
            let name = CString::new(neural_forge_protocol::env::var("NEURAL_FORGE_TEST_DISPLAY")?).ok()?;
            // SAFETY: plain dlopen/dlsym; each symbol is cast to its libX11 signature.
            unsafe {
                let lib = libc::dlopen(c"libX11.so.6".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
                if lib.is_null() {
                    return None;
                }
                let sym = |name: &CStr| libc::dlsym(lib, name.as_ptr());
                let open: unsafe extern "C" fn(*const c_char) -> *mut c_void = std::mem::transmute(sym(c"XOpenDisplay"));
                let screen: unsafe extern "C" fn(*mut c_void) -> c_int = std::mem::transmute(sym(c"XDefaultScreen"));
                let root: unsafe extern "C" fn(*mut c_void, c_int) -> c_ulong = std::mem::transmute(sym(c"XRootWindow"));
                type Create = unsafe extern "C" fn(*mut c_void, c_ulong, c_int, c_int, c_uint, c_uint, c_uint, c_ulong, c_ulong) -> c_ulong;
                let create: Create = std::mem::transmute(sym(c"XCreateSimpleWindow"));
                let sync: unsafe extern "C" fn(*mut c_void, c_int) -> c_int = std::mem::transmute(sym(c"XSync"));
                let display = open(name.as_ptr());
                if display.is_null() {
                    libc::dlclose(lib);
                    return None;
                }
                let window = create(display, root(display, screen(display)), 0, 0, 64, 64, 0, 0, 0);
                sync(display, 0);
                Some(Self { lib, display, window })
            }
        }
    }

    impl Drop for X11Window {
        fn drop(&mut self) {
            use std::ffi::{c_int, c_ulong, c_void};
            // SAFETY: as in `open`; the window and display are this one's.
            unsafe {
                let destroy: unsafe extern "C" fn(*mut c_void, c_ulong) -> c_int = std::mem::transmute(libc::dlsym(self.lib, c"XDestroyWindow".as_ptr()));
                let close: unsafe extern "C" fn(*mut c_void) -> c_int = std::mem::transmute(libc::dlsym(self.lib, c"XCloseDisplay".as_ptr()));
                destroy(self.display, self.window);
                close(self.display);
                libc::dlclose(self.lib);
            }
        }
    }

    /// A real swapchain (lavapipe's X11 WSI, with `VK_EXT_swapchain_maintenance1`): the layer's present
    /// fence, chained in front of the present's chain the way `queue_present_khr` chains it, signals
    /// after presents that wait on relay semaphores, and once the swapchain is destroyed those
    /// semaphores are freed. Needs an X server: `NEURAL_FORGE_TEST_DISPLAY` (an Xvfb's display, e.g.
    /// `:97`); skipped without one, as in CI.
    #[test]
    fn a_present_fence_frees_the_semaphores_of_a_destroyed_swapchain() {
        use std::ffi::CStr;
        let Some(x11) = X11Window::open() else {
            eprintln!("present fence WSI test: no NEURAL_FORGE_TEST_DISPLAY X server, skipping");
            return;
        };
        // SAFETY: the system loader.
        let Ok(entry) = (unsafe { ash::Entry::load() }) else { return };
        let offered = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
        let has = |list: &[vk::ExtensionProperties], name: &CStr| list.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name);
        let instance_names = [c"VK_KHR_surface", c"VK_KHR_xlib_surface", c"VK_KHR_get_surface_capabilities2", c"VK_EXT_surface_maintenance1"];
        if !instance_names.iter().all(|n| has(&offered, n)) {
            eprintln!("present fence WSI test: the loader lacks an instance extension, skipping");
            return;
        }
        let names: Vec<_> = instance_names.iter().map(|n| n.as_ptr()).collect();
        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        // SAFETY: valid create infos throughout; everything made is destroyed below, before its parent.
        unsafe {
            let instance = entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app_info).enabled_extension_names(&names), None).unwrap();
            let physical = instance.enumerate_physical_devices().unwrap()[0];
            let device_offered = instance.enumerate_device_extension_properties(physical).unwrap();
            let surface_fns = ash::extensions::khr::Surface::new(&entry, &instance);
            let surface = ash::extensions::khr::XlibSurface::new(&entry, &instance)
                .create_xlib_surface(&vk::XlibSurfaceCreateInfoKHR::builder().dpy(x11.display.cast()).window(x11.window), None)
                .unwrap();
            if !has(&device_offered, c"VK_EXT_swapchain_maintenance1") || !surface_fns.get_physical_device_surface_support(physical, 0, surface).unwrap_or(false) {
                eprintln!("present fence WSI test: no VK_EXT_swapchain_maintenance1 or no presenting family 0, skipping");
                surface_fns.destroy_surface(surface, None);
                instance.destroy_instance(None);
                return;
            }
            let device_names = [c"VK_KHR_swapchain".as_ptr(), c"VK_EXT_swapchain_maintenance1".as_ptr()];
            let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::builder().swapchain_maintenance1(true);
            let queues = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
            let device = instance
                .create_device(physical, &vk::DeviceCreateInfo::builder().queue_create_infos(&queues).enabled_extension_names(&device_names).push_next(&mut maintenance), None)
                .unwrap();
            {
                use std::io::Write;
                let _ = std::io::stderr().write_all(b"[neural-forge-gpu-test] device acquired\n");
            }
            let queue = device.get_device_queue(0, 0);
            let caps = surface_fns.get_physical_device_surface_capabilities(physical, surface).unwrap();
            let format = surface_fns.get_physical_device_surface_formats(physical, surface).unwrap()[0];
            let extent = if caps.current_extent.width == u32::MAX { vk::Extent2D { width: 64, height: 64 } } else { caps.current_extent };
            let alpha = [vk::CompositeAlphaFlagsKHR::OPAQUE, vk::CompositeAlphaFlagsKHR::INHERIT, vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED, vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED]
                .into_iter()
                .find(|a| caps.supported_composite_alpha.contains(*a))
                .unwrap();
            // Each swapchain command resolved by itself: ash's loader also asks the device for
            // `vkGetPhysicalDevicePresentRectanglesKHR`, an instance command (a validation warning).
            let resolve = |name: &CStr| (instance.fp_v1_0().get_device_proc_addr)(device.handle(), name.as_ptr()).unwrap();
            let create_swapchain: vk::PFN_vkCreateSwapchainKHR = std::mem::transmute(resolve(c"vkCreateSwapchainKHR"));
            let destroy_swapchain: vk::PFN_vkDestroySwapchainKHR = std::mem::transmute(resolve(c"vkDestroySwapchainKHR"));
            let get_images: vk::PFN_vkGetSwapchainImagesKHR = std::mem::transmute(resolve(c"vkGetSwapchainImagesKHR"));
            let acquire: vk::PFN_vkAcquireNextImageKHR = std::mem::transmute(resolve(c"vkAcquireNextImageKHR"));
            let queue_present: vk::PFN_vkQueuePresentKHR = std::mem::transmute(resolve(c"vkQueuePresentKHR"));
            let mut swapchain = vk::SwapchainKHR::null();
            let created = create_swapchain(
                    device.handle(),
                    &vk::SwapchainCreateInfoKHR::builder()
                        .surface(surface)
                        .min_image_count(caps.min_image_count.max(2))
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(alpha)
                        .present_mode(vk::PresentModeKHR::FIFO)
                        .clipped(true)
                        .build(),
                    std::ptr::null(),
                    &mut swapchain,
                );
            assert_eq!(created, vk::Result::SUCCESS);
            let mut count = 0;
            assert_eq!(get_images(device.handle(), swapchain, &mut count, std::ptr::null_mut()), vk::Result::SUCCESS);
            let mut images = vec![vk::Image::null(); count as usize];
            assert_eq!(get_images(device.handle(), swapchain, &mut count, images.as_mut_ptr()), vk::Result::SUCCESS);
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::builder().queue_family_index(0), None).unwrap();
            let buffers = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::builder().command_pool(pool).command_buffer_count(images.len() as u32))
                .unwrap();
            // The "game's" frame: each image moved to PRESENT_SRC_KHR, signalling its own present semaphore.
            let mut app_semaphores = Vec::new();
            for (&buffer, &image) in buffers.iter().zip(&images) {
                device.begin_command_buffer(buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
                let barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
                device.cmd_pipeline_barrier(buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[barrier.build()]);
                device.end_command_buffer(buffer).unwrap();
                app_semaphores.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap());
            }
            let acquired = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
            let mut relays = PresentSemaphores::default();
            let mut tickets = Tickets::default();
            let mut presented = std::collections::HashSet::new();
            for _ in 0..images.len() * 2 {
                let mut index = 0;
                let result = acquire(device.handle(), swapchain, u64::MAX, vk::Semaphore::null(), acquired, &mut index);
                assert!(matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR), "{result:?}");
                device.wait_for_fences(&[acquired], true, u64::MAX).unwrap();
                device.reset_fences(&[acquired]).unwrap();
                let (image, app) = (images[index as usize], app_semaphores[index as usize]);
                let render = vk::SubmitInfo::builder().command_buffers(&buffers[index as usize..=index as usize]).signal_semaphores(std::slice::from_ref(&app));
                device.queue_submit(queue, &[render.build()], vk::Fence::null()).unwrap();
                // The relay batch (`device::relay_app_waits`): the present waits on the layer's semaphore.
                let relay = relays.get(image, || device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).ok()).unwrap();
                let relay_batch = vk::SubmitInfo::builder().wait_semaphores(std::slice::from_ref(&app)).wait_dst_stage_mask(&stage).signal_semaphores(std::slice::from_ref(&relay));
                device.queue_submit(queue, &[relay_batch.build()], vk::Fence::null()).unwrap();
                let fence = tickets.take(&device).unwrap();
                let fences = PresentFences::new(1, 0, fence, std::ptr::null());
                let present = vk::PresentInfoKHR {
                    p_next: std::ptr::from_ref(&fences.info).cast(),
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &relay,
                    swapchain_count: 1,
                    p_swapchains: &swapchain,
                    p_image_indices: &index,
                    ..Default::default()
                };
                let result = queue_present(queue, &present);
                assert!(matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR), "{result:?}");
                assert!(present_fence_signals(result, false));
                relays.note(relay, LastUse::Ticket(tickets.submitted(fence)));
                presented.insert(index);
                // The present fence signals on its own (no idle wait): the relay can be signalled again.
                device.wait_for_fences(&[fence], true, 5_000_000_000).expect("the present fence signals");
            }
            tickets.poll(&device);
            assert!(!tickets.has_pending());
            // The application destroys the swapchain (its presents are done: their fences signalled).
            relays.retire(&images);
            assert_eq!(relays.retired(), presented.len());
            assert_eq!(relays.reclaim(&device, &tickets), presented.len(), "every relay semaphore freed");
            assert!(relays.is_empty());
            destroy_swapchain(device.handle(), swapchain, std::ptr::null());

            tickets.destroy(&device);
            device.destroy_fence(acquired, None);
            for semaphore in app_semaphores {
                device.destroy_semaphore(semaphore, None);
            }
            device.destroy_command_pool(pool, None);
            device.destroy_device(None);
            surface_fns.destroy_surface(surface, None);
            instance.destroy_instance(None);
        }
        drop(x11);
    }
}
