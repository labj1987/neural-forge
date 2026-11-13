//! `vkCmdBindDescriptorSets2` (and its KHR alias) calls that bind nothing reach the driver with a valid layout.
//!
//! DXVK (v3.1.1-174, in GE-Proton11) records `vkCmdBindDescriptorSets2` with a null `layout`, no descriptor sets and no
//! `pNext`. It does so on purpose: with VK_EXT_descriptor_heap (which it uses on NVIDIA 595.84 and newer) it makes that
//! call right before every `vkCmdCuLaunchKernelNVX` (`DxvkContext::launchCuKernelNVX`, "HACK: We want to use the
//! driver-managed descriptor heap here"), because setting descriptor-set state invalidates the application's heaps
//! and so puts the driver's own descriptors back for DLSS's CUDA kernels, which address images by
//! `vkGetImageViewHandleNVX` handles. Without the call DLSS runs on DXVK's heaps and its output is garbage: God of War
//! went black but for the HUD (2026-11-12).
//!
//! The call is not valid as recorded (`layout` must be a valid handle, `descriptorSetCount` above 0), and NVIDIA's
//! driver (620.70.06) does not check the layout: for a null one it reads whatever its third argument register holds as
//! the layout object (`[rdx+0x214]`). When that register is 0 (every time Rockstar's Social Club overlay loads in GTA
//! San Andreas - The Definitive Edition) the driver faults on DXVK's recording thread over and over, the render thread
//! hangs and Unreal closes the game two minutes later. So the call goes on with the layout replaced by an empty
//! pipeline layout the layer creates for the device (no set layouts, no push-constant ranges): the driver reads a real
//! layout object, and the call still binds no set and still switches the command buffer from the heaps to descriptor
//! sets as DXVK means it to. Only where the device has no such layout (its creation failed) is the call dropped, as
//! 4.1.1 did with every such call.
//!
//! The wrapper is handed out by `vkGetDeviceProcAddr` only where the device has the command (the next layer returns
//! it), in a process the layer is on; every other call goes through untouched. Its next-layer pointer and the empty
//! layout are kept per device, by the loader's dispatch key (a command buffer's is its device's).
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

/// One device's entry: the next layer's `vkCmdBindDescriptorSets2` (once the wrapper was handed out for it) and the
/// empty pipeline layout a call that binds nothing goes on with.
#[derive(Clone, Copy, Default)]
struct Entry {
    next: Option<Pfn>,
    empty: Option<Empty>,
}

/// The layer's empty pipeline layout of a device, and how to destroy it with the device.
#[derive(Clone, Copy)]
struct Empty {
    device: vk::Device,
    layout: vk::PipelineLayout,
    destroy: Option<vk::PFN_vkDestroyPipelineLayout>,
}

/// Each device's entry, by dispatch key.
static DEVICES: RwLock<Option<HashMap<usize, Entry>>> = RwLock::new(None);
/// Calls dropped in this process (no empty layout).
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Calls forwarded with the empty layout in this process.
static FORWARDED: AtomicU64 = AtomicU64::new(0);
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
    DEVICES.write().unwrap_or_else(PoisonError::into_inner).get_or_insert_default().entry(k).or_default().next = Some(next);
    // SAFETY: the wrapper has the command's signature.
    Some(unsafe { std::mem::transmute::<Pfn, unsafe extern "system" fn()>(bind_descriptor_sets2) })
}

/// The device was created (in a process the layer works in): the empty pipeline layout a call that binds nothing goes
/// on with. Destroyed with the device ([`forget`]).
///
/// # Safety
/// `device` is live and `device` is its next-layer dispatch.
pub(crate) unsafe fn device_created(device: &ash::Device) {
    // SAFETY: the contract; an empty create info.
    match unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None) } {
        // SAFETY: the device is live.
        Ok(layout) => unsafe { set_empty_layout(device.handle(), layout, Some(device.fp_v1_0().destroy_pipeline_layout)) },
        Err(e) => crate::log!("[layer] no empty pipeline layout ({e:?}): vkCmdBindDescriptorSets2 calls that bind nothing are dropped on this device"),
    }
}

/// Keeps `layout` (empty) for `device`'s calls that bind nothing; `destroy` destroys it with the device.
///
/// # Safety
/// `device` is live; `layout` is an empty pipeline layout of it.
unsafe fn set_empty_layout(device: vk::Device, layout: vk::PipelineLayout, destroy: Option<vk::PFN_vkDestroyPipelineLayout>) {
    use ash::vk::Handle;
    // SAFETY: forwarded.
    let k = unsafe { key(device.as_raw()) };
    DEVICES.write().unwrap_or_else(PoisonError::into_inner).get_or_insert_default().entry(k).or_default().empty = Some(Empty { device, layout, destroy });
}

/// The device is being destroyed: its entry goes (a later device may get the same dispatch key), and its empty layout.
///
/// # Safety
/// `device` is live, and none of its command buffers is pending (vkDestroyDevice's own requirement).
pub(crate) unsafe fn forget(device: vk::Device) {
    use ash::vk::Handle;
    if device == vk::Device::null() {
        return;
    }
    // SAFETY: forwarded.
    let k = unsafe { key(device.as_raw()) };
    let removed = DEVICES.write().unwrap_or_else(PoisonError::into_inner).as_mut().and_then(|m| m.remove(&k));
    if let Some(Empty { device, layout, destroy: Some(destroy) }) = removed.and_then(|e| e.empty) {
        // SAFETY: the layer's own layout of this device, used by no pending work any more.
        unsafe { destroy(device, layout, std::ptr::null()) };
    }
    let (forwarded, dropped) = (FORWARDED.load(Ordering::Relaxed), DROPPED.load(Ordering::Relaxed));
    if removed.is_some() && forwarded + dropped > 0 {
        crate::log!(
            "[layer] vkCmdBindDescriptorSets2 calls that bind nothing in this process so far: {forwarded} forwarded with an empty layout, {dropped} dropped"
        );
    }
}

/// Calls dropped in this process.
#[cfg(test)]
fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

fn entry_for(k: usize) -> Entry {
    DEVICES.read().unwrap_or_else(PoisonError::into_inner).as_ref().and_then(|m| m.get(&k).copied()).unwrap_or_default()
}

/// Calls forwarded with the empty layout.
#[cfg(test)]
fn forwarded() -> u64 {
    FORWARDED.load(Ordering::Relaxed)
}

unsafe extern "system" fn bind_descriptor_sets2(cb: vk::CommandBuffer, info: *const BindDescriptorSetsInfo) {
    use ash::vk::Handle;
    // SAFETY: `cb` is a live command buffer (the application's call).
    let entry = entry_for(unsafe { key(cb.as_raw()) });
    let Some(next) = entry.next else {
        static SAID_NONE: AtomicBool = AtomicBool::new(false);
        if !SAID_NONE.swap(true, Ordering::Relaxed) {
            crate::log!("[layer] vkCmdBindDescriptorSets2 on a command buffer of a device the wrapper does not know: not forwarded");
        }
        return;
    };
    // SAFETY: the application's call: `info` is null or valid for the call.
    if let Some(i) = unsafe { info.as_ref() }.filter(|i| binds_nothing(i)) {
        let forward = entry.empty.map(|e| e.layout);
        if !SAID.swap(true, Ordering::Relaxed) {
            crate::log!(
                "[layer] the application binds no descriptor set with a null pipeline layout (vkCmdBindDescriptorSets2, stages {:?}; DXVK does before \
                 DLSS's launches under VK_EXT_descriptor_heap): {} (NVIDIA's driver reads the null layout)",
                i.stage_flags,
                if forward.is_some() { "forwarded with an empty layout" } else { "dropped, the device has no empty layout" }
            );
            crate::logging::flush();
        }
        let Some(layout) = forward else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        FORWARDED.fetch_add(1, Ordering::Relaxed);
        let with_layout = BindDescriptorSetsInfo {
            s_type: i.s_type,
            p_next: i.p_next,
            stage_flags: i.stage_flags,
            layout,
            first_set: i.first_set,
            descriptor_set_count: i.descriptor_set_count,
            p_descriptor_sets: i.p_descriptor_sets,
            dynamic_offset_count: i.dynamic_offset_count,
            p_dynamic_offsets: i.p_dynamic_offsets,
        };
        // SAFETY: the application's call with a valid layout of its device in place of the null one.
        unsafe { next(cb, &with_layout) };
        return;
    }
    // SAFETY: the application's arguments, forwarded unchanged to the device's next layer.
    unsafe { next(cb, info) }
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
    /// (as the loader makes a device's command buffers). A call that binds nothing (DXVK's before DLSS's launches) goes
    /// on with the device's empty layout in place of the null one, nothing else changed; it is dropped only where the
    /// device has no empty layout. A normal bind and one with a chained `VkPipelineLayoutCreateInfo` reach the next
    /// layer unchanged.
    #[test]
    fn a_call_that_binds_nothing_goes_on_with_the_empty_layout() {
        let dispatch = Box::new(0u8);
        let key = std::ptr::from_ref(&*dispatch) as usize;
        let device_object = Box::new(key);
        let cb_object = Box::new(key);
        let device = vk::Device::from_raw(std::ptr::from_ref(&*device_object) as u64);
        let cb = vk::CommandBuffer::from_raw(std::ptr::from_ref(&*cb_object) as u64);
        let next: unsafe extern "system" fn() = unsafe { std::mem::transmute::<Pfn, unsafe extern "system" fn()>(next_layer) };
        let calls = || -> Vec<(u64, u64, u32, bool)> { CALLS.lock().unwrap().iter().filter(|c| c.0 == cb.as_raw()).copied().collect() };
        unsafe {
            // Not handed out: the command absent, the layer off, or another name.
            assert!(intercept(device, b"vkCmdBindDescriptorSets2", None, true).is_none());
            assert_eq!(intercept(device, b"vkCmdBindDescriptorSets2", Some(next), false).map(|f| f as usize), Some(next as usize));
            assert_eq!(intercept(device, b"vkCmdPushConstants2", Some(other), true).map(|f| f as usize), Some(other as *const () as usize));
            let wrapper = intercept(device, b"vkCmdBindDescriptorSets2KHR", Some(next), true).unwrap();
            assert_ne!(wrapper as usize, next as usize);
            let wrapper = std::mem::transmute::<unsafe extern "system" fn(), Pfn>(wrapper);
            // No empty layout (its creation failed): dropped, as before.
            let before = dropped();
            wrapper(cb, &info(0, &[], std::ptr::null()));
            assert_eq!(dropped(), before + 1);
            assert!(calls().is_empty());
            // With the device's empty layout: forwarded with it, still no set.
            set_empty_layout(device, vk::PipelineLayout::from_raw(0x42), None);
            let before = forwarded();
            wrapper(cb, &info(0, &[], std::ptr::null()));
            assert_eq!(forwarded(), before + 1);
            assert_eq!(calls(), vec![(cb.as_raw(), 0x42, 0, false)], "the call reaches the driver with a valid layout and binds no set");
            let set = [vk::DescriptorSet::from_raw(7)];
            wrapper(cb, &info(9, &set, std::ptr::null()));
            let chain = vk::PipelineLayoutCreateInfo::default();
            wrapper(cb, &info(0, &[], std::ptr::from_ref(&chain).cast()));
            assert_eq!(calls(), vec![(cb.as_raw(), 0x42, 0, false), (cb.as_raw(), 9, 1, false), (cb.as_raw(), 0, 0, true)]);
            forget(device);
            assert!(entry_for(key).next.is_none() && entry_for(key).empty.is_none());
        }
    }
}
