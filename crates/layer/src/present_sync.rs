//! Presentation completion is associated with an acquired image, not a command
//! buffer slot. Keep retired semaphores alive until device teardown.
use ash::vk;
use std::collections::HashMap;

#[derive(Default)]
pub struct PresentSemaphores {
    active: HashMap<vk::Image, vk::Semaphore>,
    retired: Vec<vk::Semaphore>,
}
impl PresentSemaphores {
    pub fn get(&mut self, image: vk::Image, create: impl FnOnce() -> Option<vk::Semaphore>) -> Option<vk::Semaphore> {
        if let Some(sem) = self.active.get(&image) { return Some(*sem); }
        let sem = create()?;
        self.active.insert(image, sem);
        Some(sem)
    }
    pub fn is_empty(&self) -> bool { self.active.is_empty() && self.retired.is_empty() }
    pub fn retire(&mut self, images: &[vk::Image]) {
        for image in images {
            if let Some(sem) = self.active.remove(image) { self.retired.push(sem); }
        }
    }
    /// # Safety
    /// Only at device teardown after queue completion and application swapchain
    /// destruction. No present may still reference these semaphores.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for &sem in self.active.values().chain(self.retired.iter()) {
            unsafe { device.destroy_semaphore(sem, None); }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
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
        assert_eq!(sems.retired, [vk::Semaphore::from_raw(12)]);
        // Recycled Vulkan image handles must not recycle an old present semaphore.
        assert_eq!(sems.get(vk::Image::from_raw(1), || Some(vk::Semaphore::from_raw(14))), Some(vk::Semaphore::from_raw(14)));
    }
}
