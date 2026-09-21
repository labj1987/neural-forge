//! Phase 4 feasibility probe (`docs/DMABUF_TRANSPORT_DESIGN.md`) -- not production code.
//!
//! **Result, real hardware, `lordnikon`, 2026-09-15: this specific mechanism does not
//! work.** `wine_server_handle_to_fd` (called against a real `vkGetMemoryWin32HandleKHR`
//! handle from a real exported buffer, NVIDIA driver 615.71.09, Proton-CachyOS)
//! returns `STATUS_OBJECT_TYPE_MISMATCH` (`0xC0000024`) -- a real, well-formed NTSTATUS,
//! not a crash or a guessed-wrong-signature fault (`guard::guarded` reported `seh=0`).
//! Read literally: the object behind this handle is not a type wineserver's generic
//! fd/handle bridge knows how to unwrap. See `docs/DMABUF_TRANSPORT_DESIGN.md` for the full
//! writeup and why this most likely means NVIDIA's own `OPAQUE_WIN32` external-memory
//! implementation for a Wine guest does not route through the kernel `dma_buf`
//! subsystem at all (a driver-private shared-surface token instead), not a fixable
//! bug in this probe. Kept in the repo as a real, reusable diagnostic in case a future
//! Wine/Proton/driver version changes this, or someone wants to try the reverse
//! direction (`wine_server_fd_to_handle` + `vkImportMemoryWin32HandleKHR`) -- not yet
//! attempted, see that design doc's own "not yet tried" section for why it was judged
//! low-probability enough not to chase immediately.
//!
//! The question this was written to answer: can this helper, a Windows PE binary
//! running under Wine, ever hand the Linux-native layer a real, dma-buf-importable
//! file descriptor for GPU memory it allocated? Windows-side app code only ever sees
//! an opaque win32 `HANDLE` from `vkGetMemoryWin32HandleKHR` -- there is no Vulkan API
//! that hands back a Unix fd directly. This probe allocates a real, exportable Vulkan
//! buffer, gets its win32 handle, then tries Wine's own (undocumented for app use, but
//! genuinely exported) `ntdll.dll` function `wine_server_handle_to_fd` to convert that
//! handle into a real Unix fd this process owns -- confirmed present in this project's
//! own target Proton build via `objdump -p ntdll.dll` before writing this, and Wine
//! does not treat it as a stable public API, so this is wrapped in `guard::guarded`
//! the same way every other externally-supplied, not-fully-trusted call in this crate
//! is.
//!
//! Logs its own PID and the resulting fd (if any) via `NEURALFORGE_LOG` -- plain
//! `println!`/stdout does not reliably reach the invoking shell through
//! `proton run` (confirmed empirically writing this probe: zero output arrived on a
//! piped stdout across 20+ real seconds of a successfully-running process; switching
//! to this crate's own `log!`/`logging::flush()` fixed it immediately, matching every
//! other diagnostic path in this crate). Then sleeps, so a Linux-side process can
//! inspect `/proc/<pid>/fd/<fd>` (readlink, `file`, or an actual
//! `VK_EXT_external_memory_dma_buf` import attempt) while this is still alive and the
//! memory is still allocated -- moot given the result above, but left in for whatever
//! variant is tried next. Run under Wine (needs `NEURALFORGE_LOG` set to see anything):
//! `cargo run --example dmabuf_probe --target x86_64-pc-windows-gnu -p neural-forge-helper`

use std::ffi::c_void;

use ash::vk;
use neural_forge_helper::{guard, log, logging};

#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcessId() -> u32;
    fn GetModuleHandleW(module_name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

/// Wine's own `ntdll.dll` export (`wine_server_handle_to_fd`), not part of the real
/// Windows ntdll -- signature from Wine's long-stable `include/wine/server.h`
/// (unchanged across many releases; used internally by winevulkan/wined3d for exactly
/// this class of interop). Returns an `NTSTATUS` (0 = success); `unix_fd` receives a
/// freshly duplicated fd this process now owns (must be closed separately -- Wine
/// does not tie its lifetime to the win32 handle). `options` receives file status
/// flags this probe doesn't need and passes a throwaway pointer for.
type WineServerHandleToFd = unsafe extern "system" fn(handle: *mut c_void, access: u32, unix_fd: *mut i32, options: *mut u32) -> i32;

const WANTED_DEVICE_EXTENSIONS: &[&str] =
    &["VK_KHR_external_memory", "VK_KHR_external_memory_win32", "VK_EXT_external_memory_host"];

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn create_vulkan_context() -> Option<(ash::Entry, ash::Instance, vk::PhysicalDevice, ash::Device, vk::Queue)> {
    // SAFETY: dynamically loads `vulkan-1.dll`, same as every other `ash` consumer in
    // this crate.
    let entry = unsafe { ash::Entry::load() }.ok()?;
    let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    // SAFETY: `create_info` is a valid, fully-populated `VkInstanceCreateInfo`.
    let instance = unsafe { entry.create_instance(&create_info, None) }.ok()?;
    // SAFETY: `instance` was just created above.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    let physical_device = *physical_devices
        .iter()
        .find(|&&pd| unsafe { instance.get_physical_device_properties(pd) }.vendor_id == 0x10DE)
        .or(physical_devices.first())?;
    // SAFETY: `physical_device` is one of the handles just enumerated above.
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap_or_default();
    let available_names: std::collections::HashSet<String> = available
        .iter()
        .filter_map(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }.to_str().ok().map(str::to_owned))
        .collect();
    let enabled: Vec<&str> = WANTED_DEVICE_EXTENSIONS.iter().copied().filter(|e| available_names.contains(*e)).collect();
    log!("dmabuf_probe: device extensions available: {enabled:?} (wanted {WANTED_DEVICE_EXTENSIONS:?})");
    logging::flush();
    let enabled_c: Vec<std::ffi::CString> = enabled.iter().map(|e| std::ffi::CString::new(*e).unwrap()).collect();
    let enabled_ptrs: Vec<*const std::ffi::c_char> = enabled_c.iter().map(|c| c.as_ptr()).collect();
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info).enabled_extension_names(&enabled_ptrs);
    // SAFETY: queue family 0 always exists; `enabled_ptrs` only names extensions just
    // confirmed present, and `enabled_c` (owning their bytes) outlives this call.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.ok()?;
    // SAFETY: `device` was just created with exactly one queue on family 0, index 0.
    let queue = unsafe { device.get_device_queue(0, 0) };
    Some((entry, instance, physical_device, device, queue))
}

fn main() {
    guard::install();

    // SAFETY: `GetCurrentProcessId` takes no arguments and cannot fail.
    let pid = unsafe { GetCurrentProcessId() };
    log!("dmabuf_probe: pid={pid}");
    logging::flush();

    let Some((entry, instance, physical_device, device, _queue)) = create_vulkan_context() else {
        log!("dmabuf_probe: FAIL -- could not create a Vulkan device");
    logging::flush();
        return;
    };

    // A small, exportable buffer -- big enough to be a plausible real allocation,
    // small enough this probe doesn't need to care about alignment/size limits the
    // way a real proxy/answer transfer eventually will.
    const BYTES: vk::DeviceSize = 4096;
    let mut export_info = vk::ExportMemoryWin32HandleInfoKHR::builder();
    let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::builder().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(BYTES)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_buffer_info);
    // SAFETY: `buffer_info` is a valid `VkBufferCreateInfo` with a real size/usage.
    let Ok(buffer) = (unsafe { device.create_buffer(&buffer_info, None) }) else {
        log!("dmabuf_probe: FAIL -- vkCreateBuffer");
    logging::flush();
        return;
    };
    // SAFETY: `buffer` was just created above.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let Some(mem_type) = (0..mem_props.memory_type_count).find(|&i| {
        reqs.memory_type_bits & (1 << i) != 0 && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }) else {
        log!("dmabuf_probe: FAIL -- no device-local memory type fits this buffer");
    logging::flush();
        return;
    };
    let mut export_alloc = vk::ExportMemoryAllocateInfo::builder().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type)
        .push_next(&mut export_alloc)
        .push_next(&mut export_info);
    // SAFETY: `alloc_info` is valid; `mem_type` was just confirmed to fit `reqs`.
    let Ok(memory) = (unsafe { device.allocate_memory(&alloc_info, None) }) else {
        log!("dmabuf_probe: FAIL -- vkAllocateMemory with export requested");
    logging::flush();
        return;
    };
    // SAFETY: `buffer`/`memory` were just created/allocated above, sized/typed to match.
    if unsafe { device.bind_buffer_memory(buffer, memory, 0) }.is_err() {
        log!("dmabuf_probe: FAIL -- vkBindBufferMemory");
    logging::flush();
        return;
    }
    log!("dmabuf_probe: allocated a {BYTES}-byte exportable buffer, resolving vkGetMemoryWin32HandleKHR by hand next");
    logging::flush();

    // Resolved manually, not via `ash::extensions::khr::ExternalMemoryWin32::new`
    // (whose `KhrExternalMemoryWin32Fn::load` panics on a resolution failure instead
    // of returning `None` -- the exact class of bug `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`
    // already found and fixed for a different extension's `::load()` helper).
    let get_win32_handle_name = c"vkGetMemoryWin32HandleKHR";
    // SAFETY: `entry`/`instance` are valid and live; the name is a real, NUL-terminated function name.
    let get_win32_handle_fp = unsafe { instance.get_device_proc_addr(device.handle(), get_win32_handle_name.as_ptr()) };
    let Some(get_win32_handle_fp) = get_win32_handle_fp else {
        log!("dmabuf_probe: FAIL -- vkGetMemoryWin32HandleKHR did not resolve (extension not really enabled?)");
    logging::flush();
        return;
    };
    let get_win32_handle: vk::PFN_vkGetMemoryWin32HandleKHR = unsafe { std::mem::transmute(get_win32_handle_fp) };

    let get_info = vk::MemoryGetWin32HandleInfoKHR::builder().memory(memory).handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
    let mut win32_handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: `get_info` is valid, points at memory just allocated with export requested.
    let result = unsafe { get_win32_handle(device.handle(), &*get_info, &mut win32_handle) };
    if result != vk::Result::SUCCESS || win32_handle.is_null() {
        log!("dmabuf_probe: FAIL -- vkGetMemoryWin32HandleKHR returned {result:?}, handle={win32_handle:?}");
    logging::flush();
        return;
    }
    log!("dmabuf_probe: got win32 handle {win32_handle:?}, resolving wine_server_handle_to_fd next");
    logging::flush();

    // SAFETY: `L"ntdll.dll"` is a real, NUL-terminated wide string; ntdll is always
    // already loaded in every Windows process, so `GetModuleHandleW` (which never
    // loads, only looks up an already-loaded module) is the right call here, not
    // `LoadLibraryW`.
    let ntdll = unsafe { GetModuleHandleW(utf16("ntdll.dll").as_ptr()) };
    if ntdll.is_null() {
        log!("dmabuf_probe: FAIL -- ntdll.dll not found (should be impossible)");
    logging::flush();
        return;
    }
    let fn_name = c"wine_server_handle_to_fd";
    // SAFETY: `ntdll` is a valid, loaded module handle; `fn_name` is NUL-terminated.
    let raw_fn = unsafe { GetProcAddress(ntdll, fn_name.as_ptr()) };
    let Some(raw_fn) = (!raw_fn.is_null()).then_some(raw_fn) else {
        log!("dmabuf_probe: FAIL -- wine_server_handle_to_fd not exported by this ntdll.dll (not running under Wine, or a Wine build old enough not to have it)");
    logging::flush();
        return;
    };
    // SAFETY: `raw_fn` is non-null, just resolved from `ntdll.dll` by exact name --
    // the only real risk left is whether its actual signature matches
    // `WineServerHandleToFd`'s guess, which is exactly what `guard::guarded` below is
    // for: a wrong guess here should fault, not corrupt memory or hang, and `guarded`
    // turns that fault into a normal, observable return value instead of a crash.
    let handle_to_fd: WineServerHandleToFd = unsafe { std::mem::transmute(raw_fn) };

    let mut unix_fd: i32 = -1;
    let mut options: u32 = 0;
    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    let (status, seh) = guard::guarded(
        || unsafe { handle_to_fd(win32_handle, GENERIC_READ | GENERIC_WRITE, &mut unix_fd, &mut options) },
        -1,
    );
    if seh != 0 {
        log!("dmabuf_probe: FAULTED calling wine_server_handle_to_fd -- SEH code {seh:#x}. The guessed signature is wrong; do not use this function without correcting it.");
    } else if status != 0 {
        log!("dmabuf_probe: wine_server_handle_to_fd returned NTSTATUS {status:#x} (non-zero -- treat as failure)");
    logging::flush();
    } else {
        log!("dmabuf_probe: SUCCESS -- pid={pid} unix_fd={unix_fd} options={options:#x}");
    logging::flush();
        log!("dmabuf_probe: from Linux, inspect with: ls -la /proc/{pid}/fd/{unix_fd} ; readlink /proc/{pid}/fd/{unix_fd}");
    }

    log!("dmabuf_probe: sleeping 60s so the fd/memory stay alive for inspection from the Linux side...");
    logging::flush();
    std::thread::sleep(std::time::Duration::from_secs(60));

    // SAFETY: `win32_handle` was returned by `vkGetMemoryWin32HandleKHR` above, per
    // that function's own documented contract that the caller owns and must close it.
    unsafe { CloseHandle(win32_handle) };
    // SAFETY: `buffer`/`memory`/`device`/`instance` were all created above; nothing
    // else references them (this probe never submits any GPU work against them).
    unsafe {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    drop(entry);
}
