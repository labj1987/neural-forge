//! The Windows side: Win32 declarations, the log, the ring writer, loading NVIDIA's core, `DllMain`.

use core::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::exports::{HOOKED, PASSTHROUGH, PASSTHROUGH_TARGETS};
use crate::ring::{self, Header, Record};

pub(crate) type Handle = *mut c_void;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const u8) -> *const c_void;
    fn CreateFileW(name: *const u16, access: u32, share: u32, security: *mut c_void, disposition: u32, flags: u32, template: Handle) -> Handle;
    fn CreateFileMappingW(file: Handle, security: *mut c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> Handle;
    fn MapViewOfFile(mapping: Handle, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
    fn CloseHandle(handle: Handle) -> i32;
    fn GetCurrentThreadId() -> u32;
    fn QueryPerformanceCounter(count: *mut i64) -> i32;
    fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    fn DisableThreadLibraryCalls(module: Handle) -> i32;
    fn GetModuleFileNameW(module: Handle, buf: *mut u16, len: u32) -> u32;
}

const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_WRITE: u32 = 0x4000_0000;
const FILE_SHARE_READ: u32 = 1;
const FILE_SHARE_WRITE: u32 = 2;
const CREATE_ALWAYS: u32 = 2;
const PAGE_READWRITE: u32 = 4;
const FILE_MAP_ALL_ACCESS: u32 = 0xf001f;
const INVALID_HANDLE: Handle = usize::MAX as Handle;

pub(crate) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn module_path(module: Handle) -> String {
    let mut buf = [0u16; 1024];
    let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    String::from_utf16_lossy(&buf[..n.min(buf.len())])
}

pub(crate) fn thread_id() -> u32 {
    unsafe { GetCurrentThreadId() }
}

struct Clock {
    start: i64,
    freq: f64,
}

fn clock() -> &'static Clock {
    static CLOCK: OnceLock<Clock> = OnceLock::new();
    CLOCK.get_or_init(|| {
        let (mut start, mut freq) = (0i64, 1i64);
        unsafe {
            QueryPerformanceCounter(&mut start);
            QueryPerformanceFrequency(&mut freq);
        }
        Clock { start, freq: freq.max(1) as f64 }
    })
}

pub(crate) fn now_ms() -> f64 {
    let mut t = 0i64;
    unsafe { QueryPerformanceCounter(&mut t) };
    (t - clock().start) as f64 * 1000.0 / clock().freq
}

/// The text log: `NEURAL_FORGE_NGX_LOG` (a Windows path, e.g. `Z:\tmp\ngx.log`), one line per call; off when unset.
fn log_file() -> &'static Option<Mutex<std::fs::File>> {
    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let path = std::env::var("NEURAL_FORGE_NGX_LOG").ok().filter(|p| !p.is_empty())?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(Mutex::new)
    })
}

pub(crate) fn logging() -> bool {
    log_file().is_some()
}

pub(crate) fn log(line: &str) {
    if let Some(file) = log_file() {
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "[{:11.3} tid={}] {line}", now_ms(), thread_id());
        }
    }
}

/// The Linux pid of this process, which the layer's `getpid()` gives: the first field of `/proc/self/stat`, read
/// through Wine's `Z:` drive.
fn unix_pid() -> Option<u32> {
    let stat = std::fs::read_to_string("Z:\\proc\\self\\stat").ok()?;
    stat.split_whitespace().next()?.parse().ok()
}

static RING: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

fn open_ring() -> Result<String, String> {
    let pid = unix_pid().ok_or("no /proc/self/stat")?;
    let path = format!("Z:\\dev\\shm\\neural-forge-ngx-{pid}.ring");
    let name = wide(&path);
    unsafe {
        let file = CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            core::ptr::null_mut(),
            CREATE_ALWAYS,
            0,
            core::ptr::null_mut(),
        );
        if file == INVALID_HANDLE {
            return Err(format!("{path}: cannot create"));
        }
        let mapping = CreateFileMappingW(file, core::ptr::null_mut(), PAGE_READWRITE, 0, ring::FILE_BYTES as u32, core::ptr::null());
        CloseHandle(file);
        if mapping.is_null() {
            return Err(format!("{path}: cannot map"));
        }
        let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, ring::FILE_BYTES) as *mut u8;
        CloseHandle(mapping);
        if view.is_null() {
            return Err(format!("{path}: cannot view"));
        }
        let header = &mut *(view as *mut Header);
        header.magic = ring::MAGIC;
        header.version = ring::VERSION;
        header.record_bytes = ring::RECORD_BYTES as u32;
        header.slots = ring::SLOTS;
        header.pid = pid;
        header.next.store(0, Ordering::Release);
        RING.store(view, Ordering::Release);
    }
    Ok(path)
}

/// Appends `record` (its `seq` is set here).
pub(crate) fn publish(mut record: Record) {
    let base = RING.load(Ordering::Acquire);
    if base.is_null() {
        return;
    }
    unsafe {
        let header = &*(base as *const Header);
        let n = header.next.fetch_add(1, Ordering::AcqRel);
        let slot = base.add(ring::HEADER_BYTES + (n % ring::SLOTS as u64) as usize * ring::RECORD_BYTES) as *mut Record;
        let seq = &*(core::ptr::addr_of_mut!((*slot).seq) as *const core::sync::atomic::AtomicU64);
        seq.store(0, Ordering::Release);
        core::sync::atomic::fence(Ordering::Release);
        record.seq = 0;
        record.t_ms = now_ms();
        record.thread = thread_id();
        core::ptr::write_volatile(slot, record);
        core::sync::atomic::fence(Ordering::Release);
        seq.store(n + 1, Ordering::Release);
    }
}

/// NVIDIA's core: `%NEURAL_FORGE_NGX_CORE%`, else `_nvngx.dll` in `NVIDIA_WINE_DLL_DIR` (a Linux path, through `Z:`).
fn core_path() -> String {
    if let Ok(p) = std::env::var("NEURAL_FORGE_NGX_CORE") {
        if !p.is_empty() {
            return p;
        }
    }
    let dir = std::env::var("NVIDIA_WINE_DLL_DIR").unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu/nvidia/wine".into());
    format!("Z:{}/_nvngx.dll", dir.trim_end_matches('/')).replace('/', "\\")
}

/// The hooked exports' targets in NVIDIA's core, in [`HOOKED`]'s order.
pub(crate) static HOOK_TARGETS: [AtomicPtr<c_void>; HOOKED.len()] = [const { AtomicPtr::new(core::ptr::null_mut()) }; HOOKED.len()];

pub(crate) fn target(name: &str) -> *const c_void {
    HOOKED.iter().position(|h| *h == name).map_or(core::ptr::null(), |i| HOOK_TARGETS[i].load(Ordering::Acquire))
}

/// Loads NVIDIA's core and resolves every export; `Err` if it cannot be loaded or lacks one of them.
fn resolve(this: Handle) -> Result<String, String> {
    let path = core_path();
    let wpath = wide(&path);
    let core = unsafe { LoadLibraryW(wpath.as_ptr()) };
    if core.is_null() {
        return Err(format!("NVIDIA's NGX core {path} did not load"));
    }
    if core == this {
        return Err(format!("{path} resolved to this wrapper (the full-path native override is missing)"));
    }
    let mut missing = Vec::new();
    let lookup = |name: &str| -> *const c_void {
        let cname: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        unsafe { GetProcAddress(core, cname.as_ptr()) }
    };
    for (i, name) in PASSTHROUGH.iter().enumerate() {
        let p = lookup(name);
        if p.is_null() {
            missing.push(*name);
        }
        unsafe { (*core::ptr::addr_of_mut!(PASSTHROUGH_TARGETS))[i] = p as usize };
    }
    for (i, name) in HOOKED.iter().enumerate() {
        let p = lookup(name);
        if p.is_null() {
            missing.push(*name);
        }
        HOOK_TARGETS[i].store(p as *mut c_void, Ordering::Release);
    }
    if missing.is_empty() {
        Ok(format!("{} at {core:p}", module_path(core)))
    } else {
        Err(format!("NVIDIA's NGX core {path} lacks {}", missing.join(", ")))
    }
}

/// # Safety
/// Called by the Windows loader.
#[no_mangle]
pub unsafe extern "system" fn DllMain(this: Handle, reason: u32, _reserved: *mut c_void) -> i32 {
    const DLL_PROCESS_ATTACH: u32 = 1;
    if reason != DLL_PROCESS_ATTACH {
        return 1;
    }
    DisableThreadLibraryCalls(this);
    let _ = clock();
    let exe = module_path(core::ptr::null_mut());
    match resolve(this) {
        Ok(core) => {
            let ring = open_ring();
            log(&format!("Neural Forge NGX wrapper {} in {exe} as {}; NVIDIA's core {core}; ring {ring:?}", env!("CARGO_PKG_VERSION"), module_path(this)));
            1
        }
        Err(e) => {
            // Failing the load makes NGX report itself unavailable: the game runs without DLSS rather than with a
            // half-forwarded core.
            log(&format!("Neural Forge NGX wrapper in {exe}: {e}"));
            0
        }
    }
}
