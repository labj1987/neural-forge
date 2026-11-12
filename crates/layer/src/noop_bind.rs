//! `vkCmdBindDescriptorSets2` (and its KHR alias) calls that bind nothing are dropped before they reach the driver.
//!
//! GTA San Andreas - The Definitive Edition's DXVK (v3.1.1-174) records `vkCmdBindDescriptorSets2KHR` with a null
//! `layout`, no descriptor sets and no `pNext`. The call binds nothing, but the spec does not allow it (`layout` must be
//! a valid handle without a chained `VkPipelineLayoutCreateInfo`, and `descriptorSetCount` must be above 0), and
//! NVIDIA's driver (620.70.06) does not check: for a null layout it reads whatever its third argument register holds
//! as the layout object (`[rdx+0x214]`). Most of the time that register points at something readable; when it is 0
//! (seen every time Rockstar's Social Club overlay loads in that game), the driver faults on DXVK's recording thread
//! over and over, the game's render thread hangs, and Unreal closes the game two minutes later ("GameThread timed out
//! waiting for RenderThread"). Dropping the call changes nothing the application can see: it binds no set.
//!
//! The wrapper is handed out by `vkGetDeviceProcAddr` only where the device has the command (the next layer returns
//! it), in a process the layer is on; every other call goes through untouched. Its next-layer pointer is kept per
//! device, by the loader's dispatch key (a command buffer's is its device's).
//! `vkCmdPushConstants2` is not wrapped: its `size` must be above 0, so no push binds nothing.
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock};

use ash::vk;

/// `VkBindDescriptorSetsInfo` (Vulkan 1.4 / `VK_KHR_maintenance6`), newer than the pinned ash.
#[repr(C)]
pub(crate) struct BindDescriptorSetsInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const std::ffi::c_void,
    pub stage_flags: vk::ShaderStageFlags,
    pub layout: vk::PipelineLayout,
    pub first_set: u32,
    pub descriptor_set_count: u32,
    pub p_descriptor_sets: *const vk::DescriptorSet,
    pub dynamic_offset_count: u32,
    pub p_dynamic_offsets: *const u32,
}

type Pfn = unsafe extern "system" fn(vk::CommandBuffer, *const BindDescriptorSetsInfo);

/// The next layer's `vkCmdBindDescriptorSets2` of each device the wrapper was handed out for, by dispatch key.
static NEXT: RwLock<Option<HashMap<usize, Pfn>>> = RwLock::new(None);
/// Calls dropped in this process.
static DROPPED: AtomicU64 = AtomicU64::new(0);
static SAID: AtomicBool = AtomicBool::new(false);

/// Whether `name` is one of the wrapped command's names.
fn wrapped_name(name: &[u8]) -> bool {
    matches!(name, b"vkCmdBindDescriptorSets2" | b"vkCmdBindDescriptorSets2KHR")
}

/// Whether the call binds nothing: a null layout, no sets and no chained structure.
pub(crate) fn binds_nothing(info: &BindDescriptorSetsInfo) -> bool {
    info.layout == vk::PipelineLayout::null() && info.descriptor_set_count == 0 && info.p_next.is_null()
}

/// The loader's dispatch key of a dispatchable handle: its first pointer-sized word.
///
/// # Safety
/// `raw` is a live dispatchable handle.
unsafe fn key(raw: u64) -> usize {
    // SAFETY: every dispatchable object starts with the loader's dispatch pointer (contract).
    unsafe { *(raw as *const usize) }
}

/// `vkGetDeviceProcAddr`'s answer for `name` on `device`: the wrapper for the wrapped command when the next layer
/// has it (`original` non-null) and `on` (the layer works in this process), else `original` unchanged.
///
/// # Safety
/// `device` is live; `original` is the next layer's pointer for `name`.
pub(crate) unsafe fn intercept(device: vk::Device, name: &[u8], original: vk::PFN_vkVoidFunction, on: bool) -> vk::PFN_vkVoidFunction {
    use ash::vk::Handle;
    let Some(next) = original else { return original };
    if !on || !wrapped_name(name) || device == vk::Device::null() {
        return original;
    }
    // SAFETY: the next layer's pointer for this command, which has this signature.
    let next = unsafe { std::mem::transmute::<unsafe extern "system" fn(), Pfn>(next) };
    if std::ptr::fn_addr_eq(next, bind_descriptor_sets2 as Pfn) {
        return original;
    }
    // SAFETY: forwarded.
    let k = unsafe { key(device.as_raw()) };
    NEXT.write().unwrap_or_else(PoisonError::into_inner).get_or_insert_default().insert(k, next);
    // SAFETY: the wrapper has the command's signature.
    Some(unsafe { std::mem::transmute::<Pfn, unsafe extern "system" fn()>(bind_descriptor_sets2) })
}

/// The device is being destroyed: its entry goes (a later device may get the same dispatch key).
///
/// # Safety
/// `device` is live.
pub(crate) unsafe fn forget(device: vk::Device) {
    use ash::vk::Handle;
    if device == vk::Device::null() {
        return;
    }
    // SAFETY: forwarded.
    let k = unsafe { key(device.as_raw()) };
    let removed = NEXT.write().unwrap_or_else(PoisonError::into_inner).as_mut().and_then(|m| m.remove(&k)).is_some();
    let dropped = DROPPED.load(Ordering::Relaxed);
    if removed && dropped > 0 {
        crate::log!("[layer] {dropped} vkCmdBindDescriptorSets2 calls that bind nothing dropped in this process so far");
    }
}

/// Calls dropped in this process.
#[cfg(test)]
fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

fn next_for(k: usize) -> Option<Pfn> {
    NEXT.read().unwrap_or_else(PoisonError::into_inner).as_ref().and_then(|m| m.get(&k).copied())
}

unsafe extern "system" fn bind_descriptor_sets2(cb: vk::CommandBuffer, info: *const BindDescriptorSetsInfo) {
    use ash::vk::Handle;
    // SAFETY: the application's call: `info` is null or valid for the call.
    if let Some(i) = unsafe { info.as_ref() } {
        if binds_nothing(i) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            if !SAID.swap(true, Ordering::Relaxed) {
                crate::log!(
                    "[layer] the application binds no descriptor set with a null pipeline layout (vkCmdBindDescriptorSets2, stages {:?}): \
                     such calls bind nothing and are dropped (NVIDIA's driver reads the null layout)",
                    i.stage_flags
                );
                crate::logging::flush();
            }
            return;
        }
    }
    // SAFETY: `cb` is a live command buffer (the application's call).
    match next_for(unsafe { key(cb.as_raw()) }) {
        // SAFETY: the application's arguments, forwarded unchanged to the device's next layer.
        Some(next) => unsafe { next(cb, info) },
        None => {
            static SAID_NONE: AtomicBool = AtomicBool::new(false);
            if !SAID_NONE.swap(true, Ordering::Relaxed) {
                crate::log!("[layer] vkCmdBindDescriptorSets2 on a command buffer of a device the wrapper does not know: not forwarded");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::sync::Mutex;

    /// The calls the stand-in next layer received: (command buffer, layout, set count, chained).
    static CALLS: Mutex<Vec<(u64, u64, u32, bool)>> = Mutex::new(Vec::new());

    unsafe extern "system" fn next_layer(cb: vk::CommandBuffer, info: *const BindDescriptorSetsInfo) {
        let i = unsafe { &*info };
        CALLS.lock().unwrap().push((cb.as_raw(), i.layout.as_raw(), i.descriptor_set_count, !i.p_next.is_null()));
    }

    unsafe extern "system" fn other() {}

    fn info(layout: u64, sets: &[vk::DescriptorSet], p_next: *const std::ffi::c_void) -> BindDescriptorSetsInfo {
        BindDescriptorSetsInfo {
            s_type: vk::StructureType::from_raw(1_000_545_003),
            p_next,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            layout: vk::PipelineLayout::from_raw(layout),
            first_set: 0,
            descriptor_set_count: sets.len() as u32,
            p_descriptor_sets: if sets.is_empty() { std::ptr::null() } else { sets.as_ptr() },
            dynamic_offset_count: 0,
            p_dynamic_offsets: std::ptr::null(),
        }
    }

    #[test]
    fn only_a_null_layout_with_no_sets_and_no_chain_binds_nothing() {
        let set = [vk::DescriptorSet::from_raw(7)];
        let chain = vk::PipelineLayoutCreateInfo::default();
        let chained = std::ptr::from_ref(&chain).cast();
        assert!(binds_nothing(&info(0, &[], std::ptr::null())));
        assert!(!binds_nothing(&info(5, &[], std::ptr::null())), "a real layout");
        assert!(!binds_nothing(&info(0, &set, std::ptr::null())), "a set");
        assert!(!binds_nothing(&info(0, &[], chained)), "a chained VkPipelineLayoutCreateInfo (dynamicPipelineLayout)");
    }

    /// Through `intercept` and the wrapper, with dispatchable stand-ins whose first word is a shared dispatch key
    /// (as the loader makes a device's command buffers): a call that binds nothing is dropped and counted, a normal
    /// bind and one with a chained `VkPipelineLayoutCreateInfo` reach the next layer unchanged.
    #[test]
    fn the_wrapper_drops_only_a_call_that_binds_nothing() {
        let dispatch = Box::new(0u8);
        let key = std::ptr::from_ref(&*dispatch) as usize;
        let device_object = Box::new(key);
        let cb_object = Box::new(key);
        let device = vk::Device::from_raw(std::ptr::from_ref(&*device_object) as u64);
        let cb = vk::CommandBuffer::from_raw(std::ptr::from_ref(&*cb_object) as u64);
        let next: unsafe extern "system" fn() = unsafe { std::mem::transmute::<Pfn, unsafe extern "system" fn()>(next_layer) };
        unsafe {
            // Not handed out: the command absent, the layer off, or another name.
            assert!(intercept(device, b"vkCmdBindDescriptorSets2", None, true).is_none());
            assert_eq!(intercept(device, b"vkCmdBindDescriptorSets2", Some(next), false).map(|f| f as usize), Some(next as usize));
            assert_eq!(intercept(device, b"vkCmdPushConstants2", Some(other), true).map(|f| f as usize), Some(other as *const () as usize));
            let wrapper = intercept(device, b"vkCmdBindDescriptorSets2KHR", Some(next), true).unwrap();
            assert_ne!(wrapper as usize, next as usize);
            let wrapper = std::mem::transmute::<unsafe extern "system" fn(), Pfn>(wrapper);
            let before = dropped();
            wrapper(cb, &info(0, &[], std::ptr::null()));
            assert_eq!(dropped(), before + 1);
            let set = [vk::DescriptorSet::from_raw(7)];
            wrapper(cb, &info(9, &set, std::ptr::null()));
            let chain = vk::PipelineLayoutCreateInfo::default();
            wrapper(cb, &info(0, &[], std::ptr::from_ref(&chain).cast()));
            assert_eq!(dropped(), before + 1);
            let calls: Vec<_> = CALLS.lock().unwrap().iter().filter(|c| c.0 == cb.as_raw()).copied().collect();
            assert_eq!(calls, vec![(cb.as_raw(), 9, 1, false), (cb.as_raw(), 0, 0, true)]);
            forget(device);
            assert!(next_for(key).is_none());
        }
    }
}
