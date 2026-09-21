//! Phase 4 feasibility probe, reverse direction (`docs/DMABUF_TRANSPORT_DESIGN.md`) --
//! the native-Linux half. Pairs with `crates/helper/examples/dmabuf_import_probe.rs`.
//!
//! The forward direction (a Wine-hosted `vkGetMemoryWin32HandleKHR` handle -> a real
//! Unix fd via `wine_server_handle_to_fd`) failed with `STATUS_OBJECT_TYPE_MISMATCH`
//! on real hardware -- see `crates/helper/examples/dmabuf_probe.rs`'s own doc comment
//! and `docs/DMABUF_TRANSPORT_DESIGN.md`. This is the untried reverse direction from that
//! doc's "not yet tried" section: the *layer* (full, native `VK_EXT_external_memory_dma_buf`
//! access) creates a real dma-buf fd and holds it open; a separate probe on the
//! helper side opens `Z:\proc\<this pid>\fd\<this fd>` via `CreateFileW` (Wine already
//! transparently maps Unix paths under `Z:\`) to get its own handle to the same
//! resource, then tries `vkImportMemoryWin32HandleKHR` with it.
//!
//! This process just exports and holds -- prints its own pid and fd, then sleeps.
//! Run: `cargo run --example dmabuf_export_probe -p neural-forge-layer`
//!
//! **Real-hardware result** (`lordnikon`, RTX 5070, driver 615.71.09): the pairing
//! `dmabuf_import_probe.rs`'s `CreateFileW` on `Z:\proc\<pid>\fd\<fd>` failed outright.
//! Traced below the Wine layer entirely: with this probe's fd confirmed still open,
//! a plain `cat`/Python `os.open()` on the identical `/proc/<pid>/fd/<fd>` path (no
//! Wine involved) also failed, with `ENXIO`. `readlink` on that fd entry shows
//! `/dmabuf:` -- dma-buf fds are anon-inode-backed (like `epoll`/`eventfd`), and Linux
//! does not support re-opening an anon-inode fd via `/proc/<pid>/fd/<N>`; only
//! `dup()` or `SCM_RIGHTS` fd-passing over a Unix socket can hand one to another
//! process. See `docs/DMABUF_TRANSPORT_DESIGN.md` for the full writeup.

use ash::vk;

const WANTED_DEVICE_EXTENSIONS: &[&str] =
    &["VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf"];

fn main() {
    // SAFETY: dynamically loads the system Vulkan loader, same as every other `ash`
    // consumer in this workspace.
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
    println!("dmabuf_export_probe: device extensions available: {enabled:?} (wanted {WANTED_DEVICE_EXTENSIONS:?})");
    if enabled.len() != WANTED_DEVICE_EXTENSIONS.len() {
        eprintln!("dmabuf_export_probe: FAIL -- not every wanted extension is available on this device");
        return;
    }
    let enabled_c: Vec<std::ffi::CString> = enabled.iter().map(|e| std::ffi::CString::new(*e).unwrap()).collect();
    let enabled_ptrs: Vec<*const std::ffi::c_char> = enabled_c.iter().map(|c| c.as_ptr()).collect();
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info).enabled_extension_names(&enabled_ptrs);
    // SAFETY: queue family 0 always exists; `enabled_ptrs` only names extensions just
    // confirmed present, and `enabled_c` (owning their bytes) outlives this call.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.expect("vkCreateDevice failed");

    const BYTES: vk::DeviceSize = 4096;
    let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::builder().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(BYTES)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external_buffer_info);
    // SAFETY: `buffer_info` is a valid `VkBufferCreateInfo` with a real size/usage.
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }.expect("vkCreateBuffer failed");
    // SAFETY: `buffer` was just created above.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mem_type = (0..mem_props.memory_type_count)
        .find(|&i| {
            reqs.memory_type_bits & (1 << i) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .expect("no device-local memory type fits this buffer");
    let mut export_alloc = vk::ExportMemoryAllocateInfo::builder().handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let alloc_info = vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(mem_type).push_next(&mut export_alloc);
    // SAFETY: `alloc_info` is valid; `mem_type` was just confirmed to fit `reqs`.
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.expect("vkAllocateMemory with dma-buf export failed");
    // SAFETY: `buffer`/`memory` were just created/allocated above, sized/typed to match.
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("vkBindBufferMemory failed");
    println!("dmabuf_export_probe: allocated a {BYTES}-byte dma-buf-exportable buffer, resolving vkGetMemoryFdKHR by hand next");

    // Resolved manually, not via `ash::extensions::khr::ExternalMemoryFd::new` (whose
    // `KhrExternalMemoryFdFn::load` panics on a resolution failure instead of
    // returning `None` -- the exact class of bug `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`
    // already found and fixed for a different extension's `::load()` helper).
    let get_fd_name = c"vkGetMemoryFdKHR";
    // SAFETY: `instance` is valid and live; the name is a real, NUL-terminated function name.
    let get_fd_fp = unsafe { instance.get_device_proc_addr(device.handle(), get_fd_name.as_ptr()) };
    let Some(get_fd_fp) = get_fd_fp else {
        eprintln!("dmabuf_export_probe: FAIL -- vkGetMemoryFdKHR did not resolve");
        return;
    };
    let get_fd: vk::PFN_vkGetMemoryFdKHR = unsafe { std::mem::transmute(get_fd_fp) };

    let get_info = vk::MemoryGetFdInfoKHR::builder().memory(memory).handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut fd: std::os::raw::c_int = -1;
    // SAFETY: `get_info` is valid, points at memory just allocated with dma-buf export requested.
    let result = unsafe { get_fd(device.handle(), &*get_info, &mut fd) };
    if result != vk::Result::SUCCESS || fd < 0 {
        eprintln!("dmabuf_export_probe: FAIL -- vkGetMemoryFdKHR returned {result:?}, fd={fd}");
        return;
    }

    let pid = std::process::id();
    println!("dmabuf_export_probe: SUCCESS -- pid={pid} fd={fd}");
    println!("dmabuf_export_probe: from the helper (under Wine), try: CreateFileW(L\"Z:\\\\proc\\\\{pid}\\\\fd\\\\{fd}\", ...)");
    println!("dmabuf_export_probe: sleeping 90s so the fd/memory stay alive for the helper-side import probe...");
    std::thread::sleep(std::time::Duration::from_secs(90));

    // SAFETY: `fd` was returned by `vkGetMemoryFdKHR` above, which transfers
    // ownership to the caller per the Vulkan spec -- this process is done with it.
    unsafe { libc::close(fd) };
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
