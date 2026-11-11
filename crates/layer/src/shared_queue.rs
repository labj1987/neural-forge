//! A queue of the application's that the layer's network shares (#49: Red Dead Redemption 2 asks for every queue of
//! NVIDIA's compute-only family, so none is left for the network's loading and the after-the-upscaler frames, and the
//! model ran nowhere in that game). The graphics family stays out of the question (`ExtraQueues` in `lib.rs`: the
//! layer's work on queues of the game's graphics family faulted the GPU), so the network takes one of the
//! application's compute-only queues and both sides synchronize on it, as Vulkan requires of a `VkQueue` used from
//! more than one thread (`vkQueueSubmit`, `vkQueueSubmit2`, `vkQueueWaitIdle`, `vkQueueBindSparse`,
//! `vkQueuePresentKHR` and the queue debug labels): the application's calls through the layer's hooks, the network's
//! own submits (`native_post`, and the C++ side through [`c_lock`]) each hold [`SharedQueue`]'s lock for the call
//! only, never while waiting on a fence. The lock is re-entrant per thread: a hook that submits the layer's own work on
//! the same queue in the application's call does not wait on itself.
//!
//! `NEURAL_FORGE_SHARE_QUEUE=off` keeps the old behaviour: no queue to spare is no network on that device.

use std::ffi::c_void;
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread::ThreadId;

use ash::vk;

/// One shared queue: its family and index, and the lock every user of it takes ([`Self::guard`]).
pub(crate) struct SharedQueue {
    pub family: u32,
    pub index: u32,
    /// The thread holding the lock and how many times it took it.
    state: Mutex<(Option<ThreadId>, u32)>,
    freed: Condvar,
}

impl std::fmt::Debug for SharedQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedQueue {{ family: {}, index: {} }}", self.family, self.index)
    }
}

impl SharedQueue {
    /// A device's shared queue lives as long as the process (a game creates a handful of devices; each keeps a few
    /// dozen bytes), so the hooks, the network's owner and its C++ side can hold it without counting.
    pub(crate) fn new(family: u32, index: u32) -> &'static Self {
        Box::leak(Box::new(Self { family, index, state: Mutex::new((None, 0)), freed: Condvar::new() }))
    }

    /// Takes the lock (again, on the thread holding it).
    pub(crate) fn lock(&self) {
        let me = std::thread::current().id();
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while s.0.is_some_and(|owner| owner != me) {
            s = self.freed.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
        s.0 = Some(me);
        s.1 += 1;
    }

    /// Lets go of one [`Self::lock`] of this thread's; nothing when this thread does not hold it.
    pub(crate) fn unlock(&self) {
        let me = std::thread::current().id();
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if s.0 != Some(me) {
            return;
        }
        s.1 -= 1;
        if s.1 == 0 {
            s.0 = None;
            drop(s);
            self.freed.notify_one();
        }
    }

    /// The lock until the guard is dropped.
    pub(crate) fn guard(&self) -> Guard<'_> {
        self.lock();
        Guard(self)
    }

    /// The pointer [`c_lock`] takes as its `user`.
    pub(crate) fn as_user(&'static self) -> *mut c_void {
        std::ptr::from_ref(self).cast_mut().cast()
    }
}

/// [`SharedQueue::guard`]'s lock.
pub(crate) struct Guard<'a>(&'a SharedQueue);

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.unlock();
    }
}

/// Takes `queue`'s lock where `shared` is that queue, else nothing.
pub(crate) fn guard_if(shared: Option<(vk::Queue, &'static SharedQueue)>, queue: vk::Queue) -> Option<Guard<'static>> {
    shared.filter(|(q, _)| *q == queue).map(|(_, s)| s.guard())
}

/// The network's C++ side's lock callback (`NfNativeOpen::queue_lock`): `lock` 1 takes, 0 lets go.
///
/// # Safety
/// `user` is [`SharedQueue::as_user`]'s.
pub(crate) unsafe extern "C" fn c_lock(user: *mut c_void, lock: i32) {
    // SAFETY: the caller's contract.
    let queue = unsafe { &*user.cast_const().cast::<SharedQueue>() };
    if lock != 0 {
        queue.lock();
    } else {
        queue.unlock();
    }
}

/// `NEURAL_FORGE_SHARE_QUEUE=off`: no sharing. Read once.
pub(crate) fn allowed() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_SHARE_QUEUE").as_deref() != Some("off"));
    *ON
}

/// The application's queue the network shares where no compute-only family has room for queues of its own: in the
/// first compute-only family (no graphics) the application asks for without flags, its last queue (the one least
/// likely to carry its own main compute work). `None` without such a request.
pub(crate) fn choose(families: &[vk::QueueFamilyProperties], infos: &[vk::DeviceQueueCreateInfo]) -> Option<(u32, u32)> {
    families.iter().enumerate().find_map(|(family, props)| {
        if !props.queue_flags.contains(vk::QueueFlags::COMPUTE) || props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            return None;
        }
        let q = infos.iter().find(|q| q.queue_family_index == family as u32)?;
        (q.flags.is_empty() && q.queue_count > 0).then_some((family as u32, q.queue_count - 1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties { queue_flags: flags, queue_count: count, ..Default::default() }
    }

    /// NVIDIA's families (graphics 16, transfer 2, compute-only 8) and Red Dead Redemption 2's request: family 0 x1,
    /// family 2 x8, family 1 x2 (2026-11-10). The last of family 2 is shared; never a graphics family's.
    #[test]
    fn the_last_queue_of_the_application_s_compute_only_family_is_chosen() {
        let g = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        let families = [family(g, 16), family(vk::QueueFlags::TRANSFER, 2), family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 8)];
        let p = [1.0f32; 8];
        let ask = |family: u32, count: u32| vk::DeviceQueueCreateInfo {
            queue_family_index: family,
            queue_count: count,
            p_queue_priorities: p.as_ptr(),
            ..Default::default()
        };
        assert_eq!(choose(&families, &[ask(0, 1), ask(2, 8), ask(1, 2)]), Some((2, 7)));
        assert_eq!(choose(&families, &[ask(0, 1), ask(1, 2)]), None, "no compute-only queue asked for: nothing to share, never the graphics family");
        let protected = vk::DeviceQueueCreateInfo { flags: vk::DeviceQueueCreateFlags::PROTECTED, ..ask(2, 8) };
        assert_eq!(choose(&families, &[ask(0, 1), protected]), None, "a protected queue is not shared");
    }

    /// The lock is re-entrant on its thread and excludes the others until the last guard is dropped.
    #[test]
    fn the_lock_is_re_entrant_and_exclusive() {
        let q = SharedQueue::new(2, 7);
        let outer = q.guard();
        let inner = q.guard();
        let inside = Arc::new(AtomicU32::new(0));
        std::thread::scope(|scope| {
            let (q2, inside2) = (q, inside.clone());
            let other = scope.spawn(move || {
                let _g = q2.guard();
                inside2.store(1, Ordering::SeqCst);
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert_eq!(inside.load(Ordering::SeqCst), 0, "held twice on this thread: the other waits");
            drop(inner);
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert_eq!(inside.load(Ordering::SeqCst), 0, "still held once");
            drop(outer);
            other.join().unwrap();
        });
        assert_eq!(inside.load(Ordering::SeqCst), 1);
        // The C side's callback takes and lets go of the same lock.
        // SAFETY: `q` outlives the calls.
        unsafe { c_lock(q.as_user(), 1) };
        let free = std::thread::scope(|scope| scope.spawn(|| q.state.lock().unwrap().0.is_none()).join().unwrap());
        assert!(!free, "taken through the callback");
        // SAFETY: as above.
        unsafe { c_lock(q.as_user(), 0) };
        assert!(q.state.lock().unwrap().0.is_none());
        q.unlock(); // not held: nothing
    }

    /// [`guard_if`] locks only the shared queue.
    #[test]
    fn only_the_shared_queue_is_guarded() {
        use ash::vk::Handle;
        let shared = (vk::Queue::from_raw(0x70), SharedQueue::new(2, 7));
        assert!(guard_if(Some(shared), vk::Queue::from_raw(0x70)).is_some());
        assert!(guard_if(Some(shared), vk::Queue::from_raw(0x71)).is_none());
        assert!(guard_if(None, vk::Queue::from_raw(0x70)).is_none());
    }
}
