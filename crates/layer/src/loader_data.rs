//! Initialize dispatchable objects created below the loader trampoline.
//! Contract: Khronos Vulkan-Loader, LoaderLayerInterface.md, "Creating New
//! Dispatchable Objects". This changes dispatch setup, not GPU synchronization.
use ash::{prelude::VkResult, vk};
use std::{collections::HashMap, ffi::c_void, sync::{LazyLock, Mutex}};
use vk::Handle;

type SetLoaderData = unsafe extern "system" fn(vk::Device, *mut c_void) -> vk::Result;

// VkLayerDeviceCreateInfo's VK_LOADER_DATA_CALLBACK union member. The loader
// supplies this ABI in VkDeviceCreateInfo::pNext (vk_layer.h).
#[repr(C)]
struct LoaderCallback {
    s_type: vk::StructureType,
    p_next: *const vk::BaseInStructure,
    function: u32,
    setter: Option<SetLoaderData>,
}
static DEVICES: LazyLock<Mutex<HashMap<vk::Device, Option<SetLoaderData>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct Registration(vk::Device);
impl Drop for Registration {
    fn drop(&mut self) { DEVICES.lock().unwrap().remove(&self.0); }
}

/// # Safety
/// `info` is the live loader-supplied create chain for `device`.
pub unsafe fn register(device: vk::Device, info: &vk::DeviceCreateInfo) -> Registration {
    let mut node = info.p_next.cast::<vk::BaseInStructure>();
    let mut setter = None;
    while !node.is_null() {
        let base = unsafe { &*node };
        if base.s_type == vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            let callback = unsafe { &*node.cast::<LoaderCallback>() };
            if callback.function == 1 { setter = callback.setter; break; }
        }
        node = base.p_next;
    }
    DEVICES.lock().unwrap().insert(device, setter);
    Registration(device)
}

unsafe fn initialize(device: vk::Device, object: *mut c_void) -> VkResult<()> {
    let registration = DEVICES.lock().unwrap().get(&device).copied();
    match registration {
        Some(Some(setter)) => unsafe { setter(device, object) }.result(),
        Some(None) => {
            // Older loaders may omit the callback. Khronos specifies copying the
            // parent's first dispatch-pointer slot in that case. Both objects are
            // live dispatchable Vulkan handles; only their ABI-owned first slot is touched.
            unsafe { *object.cast::<*mut c_void>() = *(device.as_raw() as *const *mut c_void); }
            Ok(())
        }
        // Direct ash users in the GPU tests allocate through the ordinary loader
        // trampoline, which already initializes the returned command buffer.
        None => Ok(()),
    }
}

/// Sets the loader's dispatch data on a queue the layer got from the next layer's
/// `vkGetDeviceQueue` (its own side queue): queues the application gets are set by the loader's
/// trampoline, this one never passes through it.
///
/// # Safety
/// `queue` is a live queue of `device`, obtained below the loader trampoline.
/// Any dispatchable object of `device` obtained from below the loader (the native network's).
#[cfg(target_arch = "x86_64")]
pub unsafe fn initialize_object(device: vk::Device, object: *mut c_void) -> VkResult<()> {
    unsafe { initialize(device, object) }
}

pub unsafe fn initialize_queue(device: vk::Device, queue: vk::Queue) -> VkResult<()> {
    unsafe { initialize(device, queue.as_raw() as *mut c_void) }
}

/// # Safety
/// The device and allocation parameters satisfy vkAllocateCommandBuffers.
pub unsafe fn allocate_commands(device: &ash::Device, info: &vk::CommandBufferAllocateInfo) -> VkResult<Vec<vk::CommandBuffer>> {
    let commands = unsafe { device.allocate_command_buffers(info) }?;
    for cmd in &commands {
        if let Err(error) = unsafe { initialize(device.handle(), cmd.as_raw() as *mut c_void) } {
            unsafe { device.free_command_buffers(info.command_pool, &commands); }
            return Err(error);
        }
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "system" fn set_dispatch(device: vk::Device, object: *mut c_void) -> vk::Result {
        unsafe { *object.cast::<usize>() = *(device.as_raw() as *const usize); }
        vk::Result::SUCCESS
    }
    #[test]
    fn callback_is_found_after_unrelated_pnext_and_registration_is_removed() {
        let mut parent = [0x1234usize];
        let device = vk::Device::from_raw(parent.as_mut_ptr() as u64);
        let callback = LoaderCallback { s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            p_next: std::ptr::null(), function: 1, setter: Some(set_dispatch) };
        let base = vk::BaseInStructure { s_type: vk::StructureType::APPLICATION_INFO,
            p_next: (&callback as *const LoaderCallback).cast() };
        let info = vk::DeviceCreateInfo { p_next: (&base as *const vk::BaseInStructure).cast(), ..Default::default() };
        let registration = unsafe { register(device, &info) };
        let mut child = [0usize, 0x5678];
        unsafe { initialize(device, child.as_mut_ptr().cast()) }.unwrap();
        assert_eq!(child, [0x1234, 0x5678]);
        drop(registration);
        assert!(!DEVICES.lock().unwrap().contains_key(&device));
    }
    #[test]
    fn older_loader_fallback_initializes_only_the_dispatch_slot() {
        let mut parent = [0x9abcusize];
        let device = vk::Device::from_raw(parent.as_mut_ptr() as u64);
        let _registration = unsafe { register(device, &vk::DeviceCreateInfo::default()) };
        let mut child = [0usize, 0xdef0];
        unsafe { initialize(device, child.as_mut_ptr().cast()) }.unwrap();
        assert_eq!(child, [0x9abc, 0xdef0]);
    }
}
