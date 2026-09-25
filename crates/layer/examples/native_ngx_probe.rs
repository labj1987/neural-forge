//! Native Linux NGX feasibility probe -- the first real experiment for the "native
//! Linux NGX helper" idea `docs/DMABUF_TRANSPORT_DESIGN.md` mentioned as a longer-term
//! direction. See `docs/NATIVE_NGX_HELPER_DESIGN.md` for the full writeup this probe's
//! result feeds into; this comment covers only what the probe itself does.
//!
//! NVIDIA's driver ships a genuine native Linux ELF library implementing the same
//! `NVSDK_NGX_VULKAN_*` C ABI as Windows' `nvngx.dll`/`nvngx_dlssnr.dll`:
//! `/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1` (confirmed present on `lordnikon`,
//! exporting `NVSDK_NGX_VULKAN_Init_Ext`/`CreateFeature`/`EvaluateFeature`/etc. via
//! `nm -D`). NVIDIA's own public SDK (`github.com/NVIDIA/DLSS`, fetched this session)
//! confirms this is deliberate, not incidental: `include/nvsdk_ngx_loader.h` defines
//! `NGX_CORE_LIBRARY_NAME "libnvidia-ngx.so.1"` for the non-Windows branch and loads it
//! with a plain `dlopen`, and `lib/Linux_x86_64/rel/` ships real per-feature snippet
//! `.so`s for every *officially released* feature (`libnvidia-ngx-dlss.so` = Super
//! Resolution, `-dlssd.so` = Ray Reconstruction, `-dlssg.so` = Frame Generation).
//!
//! This project's whole reason to exist is Feature 18 (`abi::FEATURE_DLSSNR`, "DLSS 5
//! Neural Rendering"), which is not one of those officially released features:
//! NVIDIA's own current public header (`nvsdk_ngx_defs.h`) lists it explicitly as
//! `NVSDK_NGX_Feature_Reserved18 = 18` -- reserved, unallocated, no shipped
//! implementation for it anywhere in the public SDK, on any platform. This probe exists
//! to answer the one question that isn't settled by reading headers alone: does Core's
//! own `CreateFeature` dispatch reject ID 18 immediately and cleanly (consistent with
//! "no snippet exists for this feature on Linux, full stop"), or does something more
//! interesting happen first? Real hardware, not inference from documentation.
//!
//! `libnvidia-ngx.so.1` is a real, stripped, undocumented-for-this-purpose proprietary
//! library being called with a partly-guessed calling sequence (NVIDIA's header gives
//! the signature; it does not confirm this project's specific argument choices are the
//! ones that satisfy Core's own internal validation) -- so, like every Windows-side NGX
//! call in this project, this call is wrapped in a native signal-based guard
//! ([`guarded`] below, `sigsetjmp`/`siglongjmp` around `SIGSEGV`/`SIGBUS`/`SIGILL`/
//! `SIGFPE`, mirroring `crates/helper/src/guard.rs`'s Windows VEH+`setjmp` design for
//! the same reason: a wrong guess should fault safely, not take the whole probe down
//! silently).
//!
//! One concrete ABI hazard this probe gets right on purpose: NVIDIA's header declares
//! `InApplicationDataPath` as `const wchar_t*`, and `wchar_t` is **4 bytes on Linux**
//! (glibc, UTF-32) vs. **2 bytes on Windows** (UTF-16) -- the existing Windows helper's
//! `abi::FnVkInitExt` (UTF-16) is not portable to this call as-is; this probe encodes
//! its own UTF-32 buffer instead of reusing that type.
//!
//! Run: `cargo run --example native_ngx_probe -p neural-forge-layer`

use std::ffi::{c_char, c_int, c_void, CString};

use ash::vk;

const NGX_CORE_LIBRARY_NAME: &str = "libnvidia-ngx.so.1";
/// `abi::FEATURE_DLSSNR` / `NVSDK_NGX_Feature_Reserved18`, duplicated here rather than
/// depending on `neural-forge-helper` (a Windows-only crate this native Linux example
/// can't link against).
const FEATURE_DLSSNR: i32 = 18;
const SIGNED_SNIPPET_APPLICATION_ID: u64 = 0x0876_232C;
/// `NVSDK_NGX_Version_API` as of the current public header (`nvsdk_ngx_defs.h`):
/// `0x0000013 << 12 | sizeof(size_t)/4`-style encoding upstream computes at compile
/// time via a macro this probe doesn't replicate -- reusing the exact constant this
/// project's own Windows helper already uses successfully (`abi::VERSION_API_14`) is
/// the one already-proven value on hand, not a fresh guess.
const VERSION_API_14: u32 = 0x14;

type NgxResult = i32;
mod result {
    use super::NgxResult;
    pub const FAIL_SEH: NgxResult = 0x8BAD_F00Du32 as NgxResult;
}
fn succeeded(r: NgxResult) -> bool {
    (r as u32 & 0xFFF0_0000) != 0xBAD0_0000
}

// ---------------------------------------------------------------------------------
// A minimal native signal guard -- sigsetjmp/siglongjmp around a synchronous hardware
// fault, the Linux-native counterpart to crates/helper/src/guard.rs's Windows VEH.
// Self-contained here rather than promoted to shared library code: this is this
// probe's only caller, same as every other diagnostic example in this repo.
// ---------------------------------------------------------------------------------

/// Deliberately over-sized, same reasoning as `guard.rs`'s own `JmpBuf`: glibc's real
/// `sigjmp_buf` on x86_64 is `__jmp_buf` (8 `long`s = 64 bytes) + a saved-mask flag +
/// `sigset_t` (128 bytes on Linux) = a little over 192 bytes total; 512 bytes leaves
/// generous headroom rather than pinning down the exact glibc-internal layout.
#[repr(C, align(16))]
struct SigJmpBuf([u8; 512]);

extern "C" {
    #[link_name = "__sigsetjmp"]
    fn sigsetjmp(env: *mut SigJmpBuf, savesigs: c_int) -> c_int;
    fn siglongjmp(env: *mut SigJmpBuf, val: c_int) -> !;
    fn sigaction(signum: c_int, act: *const KSigaction, oldact: *mut KSigaction) -> c_int;
}

const SIGSEGV: c_int = 11;
const SIGBUS: c_int = 7;
const SIGILL: c_int = 4;
const SIGFPE: c_int = 8;

/// Mirrors glibc's real `struct sigaction` on Linux x86_64 field-for-field (handler,
/// then `sigset_t sa_mask` [16 `u64`s = 128 bytes], then `int sa_flags`, then
/// `sa_restorer`) -- unlike [`SigJmpBuf`] above (an opaque buffer glibc manages
/// internally, where oversizing is safe), this struct is read directly by the real
/// `sigaction()` libc call, so its layout has to match exactly, not just be "big
/// enough". `flags` is left at 0 (no `SA_SIGINFO`) specifically so the handler union
/// resolves to the plain one-argument `sa_handler` form matching [`fault_handler`]'s
/// own signature -- `SA_SIGINFO` would call it as a three-argument `sa_sigaction`
/// instead. `restorer`/`mask` both stay zeroed: glibc's own `sigaction()` wrapper
/// manages the kernel return trampoline internally regardless of what's passed here
/// (the standard, portable pattern -- no C caller sets this by hand either), and an
/// all-zero `sa_mask` is an empty signal set, equivalent to `sigemptyset`.
#[repr(C)]
struct KSigaction {
    handler: extern "C" fn(c_int),
    mask: [u64; 16],
    flags: c_int,
    restorer: *const c_void,
}

thread_local! {
    static GUARD_JMP: std::cell::UnsafeCell<SigJmpBuf> = const { std::cell::UnsafeCell::new(SigJmpBuf([0; 512])) };
    static GUARD_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static GUARD_SIGNAL: std::cell::Cell<c_int> = const { std::cell::Cell::new(0) };
}

extern "C" fn fault_handler(sig: c_int) {
    if !GUARD_ACTIVE.with(std::cell::Cell::get) {
        // Not something this probe was watching for -- restore the default disposition
        // and re-raise so the process dies normally instead of looping.
        std::process::abort();
    }
    GUARD_SIGNAL.with(|c| c.set(sig));
    GUARD_ACTIVE.with(|a| a.set(false));
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    // SAFETY: `jmp_ptr` was armed by `sigsetjmp` on this same thread earlier in this
    // same call chain (guaranteed by `GUARD_ACTIVE` only being true in between) --
    // exactly `siglongjmp`'s precondition. Jumping out of a synchronous hardware fault
    // handler (SIGSEGV/SIGBUS/SIGILL/SIGFPE) like this is well-established and safe:
    // the faulting instruction is never going to usefully resume.
    unsafe { siglongjmp(jmp_ptr, 1) }
}

fn install_guard() {
    for sig in [SIGSEGV, SIGBUS, SIGILL, SIGFPE] {
        let act = KSigaction { handler: fault_handler, mask: [0; 16], flags: 0, restorer: std::ptr::null() };
        // SAFETY: `act` is a valid, fully-initialized `sigaction` struct matching
        // glibc's real layout (see `KSigaction`'s doc comment); `sig` is one of the
        // four fixed, valid signal numbers above.
        unsafe { sigaction(sig, &act, std::ptr::null_mut()) };
    }
}

fn guarded<F: FnOnce() -> R, R>(f: F, fail_value: R) -> (R, c_int) {
    GUARD_SIGNAL.with(|c| c.set(0));
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    // SAFETY: `jmp_ptr` is this thread's own thread-local storage, stable for the
    // life of the thread -- exactly what `sigsetjmp` requires. `savesigs = 1` so a
    // `siglongjmp` back here also restores the signal mask the fault handler ran under.
    let did_jump = unsafe { sigsetjmp(jmp_ptr, 1) };
    if did_jump == 0 {
        GUARD_ACTIVE.with(|a| a.set(true));
        let result = f();
        GUARD_ACTIVE.with(|a| a.set(false));
        (result, 0)
    } else {
        (fail_value, GUARD_SIGNAL.with(std::cell::Cell::get))
    }
}

// ---------------------------------------------------------------------------------
// dlopen/dlsym against libnvidia-ngx.so.1
// ---------------------------------------------------------------------------------

const RTLD_NOW: c_int = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

type FnVkInitExt = unsafe extern "C" fn(
    app_id: u64,
    app_data_path: *const u32, // wchar_t on Linux is 4 bytes -- see the module doc comment.
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    sdk_version: u32,
    feature_info: *const c_void,
) -> NgxResult;

type FnVkAllocateParameters = unsafe extern "C" fn(parameters: *mut *mut c_void) -> NgxResult;

type FnVkCreateFeature =
    unsafe extern "C" fn(cmd: vk::CommandBuffer, feature: i32, parameters: *mut c_void, handle: *mut *mut c_void) -> NgxResult;

fn utf32(s: &str) -> Vec<u32> {
    s.chars().map(|c| c as u32).chain(std::iter::once(0)).collect()
}

fn main() {
    install_guard();

    let lib_name = CString::new(NGX_CORE_LIBRARY_NAME).unwrap();
    // SAFETY: `lib_name` is a valid, NUL-terminated C string naming a real shared
    // object this probe expects to already be on the system's dynamic linker path
    // (installed by the NVIDIA driver package, confirmed present on `lordnikon`).
    let core = unsafe { dlopen(lib_name.as_ptr(), RTLD_NOW) };
    if core.is_null() {
        // SAFETY: dlerror() returns a pointer to a static/thread-local buffer valid
        // until the next dl* call; read immediately.
        let msg = unsafe { std::ffi::CStr::from_ptr(dlerror()) }.to_string_lossy().into_owned();
        eprintln!("native_ngx_probe: FAIL -- dlopen({NGX_CORE_LIBRARY_NAME}) failed: {msg}");
        return;
    }
    println!("native_ngx_probe: dlopen({NGX_CORE_LIBRARY_NAME}) succeeded, resolving exports");

    // SAFETY (both the macro body and every transmute below it feeds): `core` was just
    // confirmed non-null above; each `sym` is a valid, NUL-terminated C string; each
    // resulting `fp` is confirmed non-null before use -- the standard "trust the
    // dynamic linker's own resolution, verify only non-null-ness" pattern this project
    // uses on the Windows side too.
    macro_rules! resolve {
        ($name:literal) => {{
            let sym = CString::new($name).unwrap();
            let fp = dlsym(core, sym.as_ptr());
            if fp.is_null() {
                eprintln!(concat!("native_ngx_probe: FAIL -- ", $name, " did not resolve"));
                return;
            }
            fp
        }};
    }
    let init_ext: FnVkInitExt = unsafe { std::mem::transmute(resolve!("NVSDK_NGX_VULKAN_Init_Ext")) };
    let allocate_parameters: FnVkAllocateParameters = unsafe { std::mem::transmute(resolve!("NVSDK_NGX_VULKAN_AllocateParameters")) };
    let create_feature: FnVkCreateFeature = unsafe { std::mem::transmute(resolve!("NVSDK_NGX_VULKAN_CreateFeature")) };
    println!("native_ngx_probe: all three exports resolved");

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
    let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
    let device_create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info);
    // SAFETY: queue family 0 always exists on a real GPU's graphics-capable device.
    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.expect("vkCreateDevice failed");
    println!("native_ngx_probe: real Vulkan instance/device created, calling NVSDK_NGX_VULKAN_Init_Ext");

    let app_data_path = utf32(std::env::temp_dir().to_str().unwrap());
    let ((init_result,), seh) = guarded(
        || {
            // SAFETY: `init_ext` resolved above from a live module; `instance`/
            // `physical_device`/`device` are this probe's own, live Vulkan handles;
            // `app_data_path` is a valid, NUL-terminated UTF-32 buffer kept alive for
            // this call's duration.
            let r = unsafe {
                init_ext(SIGNED_SNIPPET_APPLICATION_ID, app_data_path.as_ptr(), instance.handle(), physical_device, device.handle(), VERSION_API_14, std::ptr::null())
            };
            (r,)
        },
        (result::FAIL_SEH,),
    );
    println!("native_ngx_probe: NVSDK_NGX_VULKAN_Init_Ext -> {:#x} (fault_signal={seh})", init_result as u32);
    if seh != 0 {
        println!("native_ngx_probe: STOPPING -- Init_Ext faulted, the guessed native ABI is wrong somewhere");
        return;
    }
    if !succeeded(init_result) {
        println!("native_ngx_probe: STOPPING -- Init_Ext did not fault but returned a real NGX failure code, not calling CreateFeature");
        return;
    }

    let ((alloc_result, params),) = guarded(
        || {
            let mut params: *mut c_void = std::ptr::null_mut();
            // SAFETY: `allocate_parameters` resolved above; `&mut params` is a valid
            // out-pointer for the call's duration.
            let r = unsafe { allocate_parameters(&mut params) };
            ((r, params),)
        },
        ((result::FAIL_SEH, std::ptr::null_mut()),),
    )
    .0;
    println!("native_ngx_probe: NVSDK_NGX_VULKAN_AllocateParameters -> {:#x} params={params:?}", alloc_result as u32);

    println!("native_ngx_probe: calling NVSDK_NGX_VULKAN_CreateFeature({FEATURE_DLSSNR}) with a null command buffer -- expecting a clean, real NGX rejection (not a fault) if Core's dispatch has no snippet for this feature ID on Linux at all");
    let ((create_result,), seh) = guarded(
        || {
            let mut handle: *mut c_void = std::ptr::null_mut();
            // SAFETY: `create_feature` resolved above; a null `VkCommandBuffer` is an
            // intentionally invalid argument -- this call is expected to reject the
            // feature ID before it would ever dereference the command buffer, and is
            // guarded in case that expectation is wrong.
            let r = unsafe { create_feature(vk::CommandBuffer::null(), FEATURE_DLSSNR, params, &mut handle) };
            (r,)
        },
        (result::FAIL_SEH,),
    );
    println!("native_ngx_probe: NVSDK_NGX_VULKAN_CreateFeature({FEATURE_DLSSNR}) -> {:#x} (fault_signal={seh})", create_result as u32);

    // Control experiment: `NVSDK_NGX_Feature_SuperSampling` (= 1, "DLSS" proper) is a
    // genuinely public, officially released feature -- NVIDIA's own public SDK ships a
    // real Linux snippet for it (`lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so`), but
    // this specific machine was never confirmed to have that snippet actually
    // discoverable by Core (only Core itself, `libnvidia-ngx.so.1`, was found in the
    // earlier filesystem search). Running the identical `CreateFeature` call against
    // this feature ID distinguishes two different explanations for Feature 18's own
    // result above: "Core's feature-dispatch mechanism is broken/unavailable in this
    // environment generally" (this call would fail identically) vs. "Core's dispatch
    // works fine, it specifically has no snippet for Feature 18" (this call fails
    // differently, or succeeds, since it's asking for something NVIDIA's SDK is
    // actually meant to support).
    const FEATURE_SUPER_SAMPLING: i32 = 1;
    println!("native_ngx_probe: control: calling NVSDK_NGX_VULKAN_CreateFeature({FEATURE_SUPER_SAMPLING}) (public 'DLSS' Super Sampling feature) for comparison");
    let ((control_result,), seh) = guarded(
        || {
            let mut handle: *mut c_void = std::ptr::null_mut();
            // SAFETY: same reasoning as the Feature 18 call above.
            let r = unsafe { create_feature(vk::CommandBuffer::null(), FEATURE_SUPER_SAMPLING, params, &mut handle) };
            (r,)
        },
        (result::FAIL_SEH,),
    );
    println!("native_ngx_probe: NVSDK_NGX_VULKAN_CreateFeature({FEATURE_SUPER_SAMPLING}) -> {:#x} (fault_signal={seh})", control_result as u32);

    // SAFETY: `device`/`instance` were created above; this probe never submits any
    // real GPU work against them.
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    drop(entry);
}
