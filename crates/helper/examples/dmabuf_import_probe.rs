//! Phase 4 feasibility probe, reverse direction, Wine/helper half
//! (`docs/DMABUF_TRANSPORT_DESIGN.md`). Pairs with
//! `crates/layer/examples/dmabuf_export_probe.rs`, which must already be running
//! (its own doc comment covers the forward-direction result this is following up on).
//!
//! Takes the exporter's printed `pid` and `fd` as two command-line arguments. Opens
//! `Z:\proc\<pid>\fd\<fd>` via `CreateFileW` -- Wine already transparently maps Unix
//! paths under `Z:\`, the same translation `crates/helper/src/shm.rs::windows_path`
//! relies on for the SHM file itself -- which performs a real `open()` against that
//! `/proc` path on the host, duplicating access to the *exact* dma-buf the exporter
//! holds open. That already gives a normal win32 `HANDLE` (unlike the forward
//! direction's probe, this doesn't need `wine_server_fd_to_handle` at all: `CreateFileW`
//! itself is the thing that would wrap a raw fd as a handle here). Then tries
//! `vkImportMemoryWin32HandleKHR`'s counterpart, `vkAllocateMemory` with
//! `VkImportMemoryWin32HandleInfoKHR{handleType: OPAQUE_WIN32}`, to see whether the
//! driver accepts a handle that did not come from its own `vkGetMemoryWin32HandleKHR`
//! export.
//!
//! Logs via `NEURALFORGE_LOG` (`log!`/`logging::flush()`) -- plain stdout does not
//! reliably reach the invoking shell through `proton run`, confirmed while writing the
//! forward-direction probe. Run under Wine, with the exporter already running and its
//! pid/fd in hand:
//! `cargo run --example dmabuf_import_probe --target x86_64-pc-windows-gnu -p neural-forge-helper -- <pid> <fd>`

use std::ffi::c_void;

use ash::vk;
use neural_forge_helper::{log, logging};

#[link(name = "kernel32")]
extern "system" {
    fn CreateFileW(
        filename: *const u16,
        access: u32,
        share_mode: u32,
        security: *const c_void,
        creation: u32,
        flags: u32,
        template_file: *mut c_void,
    ) -> *mut c_void;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_WRITE: u32 = 0x4000_0000;
const FILE_SHARE_READ: u32 = 0x1;
const FILE_SHARE_WRITE: u32 = 0x2;
const OPEN_EXISTING: u32 = 3;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

const WANTED_DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_external_memory", "VK_KHR_external_memory_win32"];
const BYTES: vk::DeviceSize = 4096; // must match crates/layer/examples/dmabuf_export_probe.rs's own allocation.

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(pid), Some(fd)) = (args.get(1), args.get(2)) else {
        log!("dmabuf_import_probe: usage: dmabuf_import_probe <exporter pid> <exporter fd>");
        logging::flush();
        return;
    };
    let proc_path = format!("Z:\\proc\\{pid}\\fd\\{fd}");
    log!("dmabuf_import_probe: opening {proc_path} via CreateFileW");
    logging::flush();

    // SAFETY: `proc_path` is a real, NUL-terminated (via `utf16`) wide string.
    let handle = unsafe {
        CreateFileW(
            utf16(&proc_path).as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE || handle.is_null() {
        log!("dmabuf_import_probe: FAIL -- CreateFileW could not open {proc_path} (exporter not running, wrong pid/fd, or permission denied)");
        logging::flush();
        return;
    }
    log!("dmabuf_import_probe: CreateFileW succeeded, handle={handle:?}, building a Vulkan device next");
    logging::flush();

    // SAFETY: dynamically loads `vulkan-1.dll`, same as every other `ash` consumer in this crate.
    let entry = unsafe { ash::Entry::load() }.expect("failed to load the Vulkan loader");
    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is a valid, fully-populated `VkInstanceCreateInfo`.
    let instance = unsafe { entry.create_instance(&create_info, None) }.expect("vkCreateInstance failed");
    // SAFETY: `instance` was just created above.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.expect("vkEnumeratePhysicalDevices failed");
    let physical_device = *physical_devices
        .iter()
        .find(|&&pd| unsafe { instance.get_physical_device_properties(pd) }.vendor_id == 0x10DE)
        .or(physical_devices.first())
        .expect("no physical device found");
    // SAFETY: `physical_device` is one of the handles just enumerated above.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap_or_default();
    let available_names: std::collections::HashSet<String> = available
        .iter()
        .filter_map(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }.to_str().ok().map(str::to_owned))
        .collect();
    let enabled: Vec<&str> = WANTED_DEVICE_EXTENSIONS.iter().copied().filter(|e| available_names.contains(*e)).collect();
    log!("dmabuf_import_probe: device extensions available: {enabled:?} (wanted {WANTED_DEVICE_EXTENSIONS:?})");
    logging::flush();
    let enabled_c: Vec<std::ffi::CString> = enabled.iter().map(|e| std::ffi::CString::new(*e).unwrap()).collect();
    let enabled_ptrs: Vec<*const std::ffi::c_char> = enabled_c.iter().map(|c| c.as_ptr()).collect();
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info).enabled_extension_names(&enabled_ptrs);
    // SAFETY: queue family 0 always exists; `enabled_ptrs` only names extensions just
    // confirmed present, and `enabled_c` (owning their bytes) outlives this call.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.expect("vkCreateDevice failed");

    // Resolved manually, not via a panicking `::load()` helper -- same reasoning as
    // every other manually-resolved extension function in this project since
    // `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`.
    let get_props_name = c"vkGetMemoryWin32HandlePropertiesKHR";
    let get_props_fp = unsafe { instance.get_device_proc_addr(device.handle(), get_props_name.as_ptr()) };
    let Some(get_props_fp) = get_props_fp else {
        log!("dmabuf_import_probe: FAIL -- vkGetMemoryWin32HandlePropertiesKHR did not resolve");
        logging::flush();
        return;
    };
    let get_props: vk::PFN_vkGetMemoryWin32HandlePropertiesKHR = unsafe { std::mem::transmute(get_props_fp) };

    let mut props = vk::MemoryWin32HandlePropertiesKHR::default();
    // SAFETY: `handle` was just confirmed non-null/valid above; `props` is a valid,
    // zeroed out-parameter of the right type.
    let result = unsafe { get_props(device.handle(), vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32, handle, &mut props) };
    if result != vk::Result::SUCCESS {
        log!("dmabuf_import_probe: FAIL -- vkGetMemoryWin32HandlePropertiesKHR returned {result:?} (this handle isn't recognized as importable memory at all)");
        logging::flush();
        return;
    }
    log!("dmabuf_import_probe: vkGetMemoryWin32HandlePropertiesKHR succeeded, memory_type_bits={:#x}", props.memory_type_bits);
    logging::flush();

    let Some(mem_type) = (0..32).find(|&i| props.memory_type_bits & (1 << i) != 0) else {
        log!("dmabuf_import_probe: FAIL -- memory_type_bits is empty, nothing to pick");
        logging::flush();
        return;
    };

    let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::builder().handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32).handle(handle);
    let alloc_info = vk::MemoryAllocateInfo::builder().allocation_size(BYTES).memory_type_index(mem_type).push_next(&mut import_info);
    // SAFETY: `alloc_info` is a valid `VkMemoryAllocateInfo` chaining a real handle
    // just confirmed importable (in the sense that `vkGetMemoryWin32HandlePropertiesKHR`
    // above didn't reject it) and a memory type index its own `memory_type_bits` allows.
    match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(memory) => {
            log!("dmabuf_import_probe: SUCCESS -- vkAllocateMemory imported the handle. The reverse direction works.");
            logging::flush();
            // SAFETY: `memory` was just allocated above; nothing else references it.
            unsafe { device.free_memory(memory, None) };
        }
        Err(e) => {
            log!("dmabuf_import_probe: FAIL -- vkAllocateMemory (import) returned {e:?}");
            logging::flush();
        }
    }

    // SAFETY: `device`/`instance` were created above; `handle` was opened above and
    // this probe is done with it either way.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
        CloseHandle(handle);
    }
    drop(entry);
}
