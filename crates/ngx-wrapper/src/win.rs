//! The Windows side: Win32 declarations, the log, the ring writer, loading NVIDIA's core, `DllMain`.

use core::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::exports::{missing_export, HOOKED, PASSTHROUGH, PASSTHROUGH_TARGETS, PASSTHROUGH_TRACED};
use crate::ring::{self, Header, Record};
use crate::rules::{self, Resolution};
use crate::trace;

pub(crate) type Handle = *mut c_void;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const u8) -> *const c_void;
    fn DeleteFileW(name: *const u16) -> i32;
    fn CreateFileW(name: *const u16, access: u32, share: u32, security: *mut c_void, disposition: u32, flags: u32, template: Handle) -> Handle;
    fn CreateFileMappingW(file: Handle, security: *mut c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> Handle;
    fn MapViewOfFile(mapping: Handle, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
    fn CloseHandle(handle: Handle) -> i32;
    fn GetCurrentThreadId() -> u32;
    fn QueryPerformanceCounter(count: *mut i64) -> i32;
    fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    fn DisableThreadLibraryCalls(module: Handle) -> i32;
    fn VirtualProtect(address: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
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

/// The text log: `NEURAL_FORGE_NGX_LOG` (a Windows path, e.g. `Z:\tmp\ngx.log`, or a Unix one, `/tmp/ngx.log`, taken
/// through `Z:`: [`trace::windows_path`]), one line per creation, evaluate and release; off when unset.
fn log_file() -> &'static Option<Mutex<std::fs::File>> {
    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let path = trace::windows_path(&std::env::var("NEURAL_FORGE_NGX_LOG").ok()?)?;
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
    trace(format_args!("log: {line}"));
}

// ---- The call trace (`crate::trace`, `NEURAL_FORGE_NGX_TRACE`). ----

/// The mapped trace file (null: tracing off), and how many bytes went into its ring.
static TRACE: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
static TRACE_WRITTEN: AtomicU64 = AtomicU64::new(0);
/// Whether passthrough calls go through the traced entries (set once, at load).
static TRACE_CALLS: AtomicBool = AtomicBool::new(false);

pub(crate) fn tracing() -> bool {
    !TRACE.load(Ordering::Relaxed).is_null()
}

/// Appends one line to the trace: a copy into the mapping, no system call, no lock (a line's bytes are reserved with
/// one atomic add, so threads never interleave inside a line).
pub(crate) fn trace(args: core::fmt::Arguments) {
    let base = TRACE.load(Ordering::Acquire);
    if base.is_null() {
        return;
    }
    let line = trace::Line::new(now_ms(), thread_id(), args);
    let bytes = line.bytes();
    let pos = TRACE_WRITTEN.fetch_add(bytes.len() as u64, Ordering::AcqRel);
    // SAFETY: the view is FILE_BYTES long and never unmapped; concurrent writers own disjoint byte ranges (a writer a
    // whole ring behind could overlap, which only garbles text).
    unsafe {
        let data = core::slice::from_raw_parts_mut(base.add(trace::HEADER_BYTES), trace::DATA_BYTES);
        trace::write_at(data, pos, bytes);
        let header = trace::header(pos + bytes.len() as u64);
        core::ptr::copy_nonoverlapping(header.as_ptr(), base, trace::HEADER_BYTES);
    }
}

/// Maps the trace file `NEURAL_FORGE_NGX_TRACE` names (a new file each run; `{pid}` in the name becomes the game's
/// Linux pid), if set.
fn open_trace() -> Option<Result<String, String>> {
    let path = trace::windows_path(&std::env::var(trace::ENV).ok()?)?;
    let path = trace::with_pid(&path, unix_pid());
    let name = wide(&path);
    // SAFETY: Win32 calls with valid arguments; the view stays mapped for the process's life.
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
            return Some(Err(format!("{path}: cannot create")));
        }
        let mapping = CreateFileMappingW(file, core::ptr::null_mut(), PAGE_READWRITE, 0, trace::FILE_BYTES as u32, core::ptr::null());
        CloseHandle(file);
        if mapping.is_null() {
            return Some(Err(format!("{path}: cannot map")));
        }
        let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, trace::FILE_BYTES) as *mut u8;
        CloseHandle(mapping);
        if view.is_null() {
            return Some(Err(format!("{path}: cannot view")));
        }
        let header = trace::header(0);
        core::ptr::copy_nonoverlapping(header.as_ptr(), view, trace::HEADER_BYTES);
        TRACE.store(view, Ordering::Release);
    }
    Some(Ok(path))
}

/// NVIDIA's passthrough functions while tracing (in [`PASSTHROUGH`]'s order); [`PASSTHROUGH_TARGETS`] then holds
/// the traced entries.
pub(crate) static mut REAL_TARGETS: [usize; PASSTHROUGH.len()] = [0; PASSTHROUGH.len()];

thread_local! {
    static STACK: core::cell::RefCell<trace::Stack> = const { core::cell::RefCell::new(trace::Stack::new()) };
}

/// The common thunk of the traced passthrough entries (`exports.rs`): r11 holds the export's index. It saves the
/// argument registers (rcx, rdx, r8, r9, xmm0-3), lets [`trace_enter`] log the call and replace the caller's return
/// address with [`trace_return`], restores them and jumps to NVIDIA's function with the caller's stack untouched
/// otherwise, so any signature passes through. (Not for an export a C++ exception crosses: the unwinder would meet
/// `trace_return` as a return address. NGX's C exports return a result code.)
///
/// # Safety
/// Reached only from a traced entry, at an export's first instruction.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn trace_thunk() {
    core::arch::naked_asm!(
        "push rcx",
        "push rdx",
        "push r8",
        "push r9",
        "sub rsp, 0x68",
        "movdqu [rsp + 0x20], xmm0",
        "movdqu [rsp + 0x30], xmm1",
        "movdqu [rsp + 0x40], xmm2",
        "movdqu [rsp + 0x50], xmm3",
        "mov rcx, r11",
        "lea rdx, [rsp + 0x88]",
        "lea r8, [rsp + 0x68]",
        "call {enter}",
        "movdqu xmm0, [rsp + 0x20]",
        "movdqu xmm1, [rsp + 0x30]",
        "movdqu xmm2, [rsp + 0x40]",
        "movdqu xmm3, [rsp + 0x50]",
        "add rsp, 0x68",
        "pop r9",
        "pop r8",
        "pop rdx",
        "pop rcx",
        "jmp rax",
        enter = sym trace_enter,
    )
}

/// Where a traced passthrough call returns: [`trace_leave`] logs its exit and gives the caller's return address.
///
/// # Safety
/// Reached only by the `ret` of a function [`trace_enter`] redirected.
#[unsafe(naked)]
unsafe extern "C" fn trace_return() {
    core::arch::naked_asm!(
        "push rax",
        "push rdx",
        "sub rsp, 0x30",
        "movdqu [rsp + 0x20], xmm0",
        "mov rcx, rax",
        "call {leave}",
        "mov r11, rax",
        "movdqu xmm0, [rsp + 0x20]",
        "add rsp, 0x30",
        "pop rdx",
        "pop rax",
        "jmp r11",
        leave = sym trace_leave,
    )
}

/// A traced passthrough call's entry: logs it (its first four arguments: rcx, rdx, r8, r9), keeps the caller's return
/// address and puts [`trace_return`] in its place, and gives NVIDIA's function. `regs` holds r9, r8, rdx, rcx.
extern "C" fn trace_enter(export: usize, ret_slot: *mut usize, regs: *const usize) -> usize {
    let name = PASSTHROUGH.get(export).copied().unwrap_or("?");
    // SAFETY: the thunk passes its own stack slots; REAL_TARGETS is written once, before any export can be called.
    let (real, args) = unsafe { ((*core::ptr::addr_of!(REAL_TARGETS))[export], [*regs.add(3), *regs.add(2), *regs.add(1), *regs]) };
    let t_ms = now_ms();
    let pushed = STACK.with(|s| {
        let Ok(mut s) = s.try_borrow_mut() else { return false };
        // SAFETY: the thunk's slot holding the caller's return address.
        let ret = unsafe { *ret_slot };
        if s.push(trace::Frame { export, ret, t_ms }) {
            // SAFETY: as above.
            unsafe { *ret_slot = trace_return as *const () as usize };
            true
        } else {
            false
        }
    });
    trace(format_args!("> {name}({:#x}, {:#x}, {:#x}, {:#x}){}", args[0], args[1], args[2], args[3], if pushed { "" } else { " [exit not traced]" }));
    real
}

/// A traced passthrough call's exit: logs its result (eax) and time, and gives the caller's return address.
extern "C" fn trace_leave(rax: usize) -> usize {
    let frame = STACK.with(|s| s.try_borrow_mut().ok().and_then(|mut s| s.pop()));
    let Some(f) = frame else {
        // Cannot happen: only a pushed call returns here. There is no return address to go back to.
        trace(format_args!("trace: a traced return without its call; aborting"));
        std::process::abort();
    };
    trace(format_args!("< {} = {:#x} ({:.3} ms)", PASSTHROUGH.get(f.export).copied().unwrap_or("?"), rax as u32, now_ms() - f.t_ms));
    f.ret
}

/// A hooked export's entry line (`hooks.rs`); its start time for [`trace_out`] when tracing.
pub(crate) fn trace_in(name: &str, args: &[usize]) -> Option<f64> {
    if !tracing() {
        return None;
    }
    match args {
        [a, b, c, d, e] => trace(format_args!("> {name}({a:#x}, {b:#x}, {c:#x}, {d:#x}, {e:#x})")),
        [a, b, c, d] => trace(format_args!("> {name}({a:#x}, {b:#x}, {c:#x}, {d:#x})")),
        [a] => trace(format_args!("> {name}({a:#x})")),
        _ => trace(format_args!("> {name}")),
    }
    Some(now_ms())
}

/// A hooked export's exit line.
pub(crate) fn trace_out(name: &str, start: Option<f64>, result: u32) {
    if let Some(t) = start {
        trace(format_args!("< {name} = {result:#x} ({:.3} ms)", now_ms() - t));
    }
}

#[link(name = "ntdll")]
extern "system" {
    fn NtQueryInformationProcess(process: Handle, class: u32, info: *mut c_void, len: u32, ret: *mut u32) -> i32;
}

/// Wine's `ProcessWineUnixPid` information class (`include/winternl.h`).
const PROCESS_WINE_UNIX_PID: u32 = 1101;

/// The Linux pid of this process, which the layer's `getpid()` gives, from Wine's `ProcessWineUnixPid`. (Not
/// `Z:\proc\self`: Wine's server opens files, so that names the wineserver.)
fn unix_pid() -> Option<u32> {
    let mut pid: i32 = 0;
    let current = -1isize as Handle;
    // SAFETY: an out-parameter of the size the class writes.
    let status = unsafe { NtQueryInformationProcess(current, PROCESS_WINE_UNIX_PID, &mut pid as *mut i32 as *mut c_void, 4, core::ptr::null_mut()) };
    (status == 0 && pid > 0).then_some(pid as u32)
}

static RING: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

fn open_ring() -> Result<String, String> {
    let pid = unix_pid().ok_or("Wine gave no unix pid (ProcessWineUnixPid)")?;
    let path = format!("Z:\\dev\\shm\\neural-forge-ngx-{pid}.ring");
    let name = wide(&path);
    unsafe {
        // A file left at this name by an earlier process with the same pid goes first: the layer takes only a ring
        // made since its process started, and a new file is one.
        DeleteFileW(name.as_ptr());
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

/// Loads NVIDIA's core and resolves every export: `Err` if it cannot be loaded or lacks an export the wrapper hooks;
/// a passthrough export it lacks jumps to [`missing_export`] (named in the returned text, for the log).
fn resolve(this: Handle) -> Result<String, String> {
    let path = core_path();
    let wpath = wide(&path);
    let t = now_ms();
    trace(format_args!("load: LoadLibraryW({path}) ..."));
    let core = unsafe { LoadLibraryW(wpath.as_ptr()) };
    trace(format_args!("load: LoadLibraryW returned {core:p} after {:.3} ms", now_ms() - t));
    if core.is_null() {
        return Err(format!("NVIDIA's NGX core {path} did not load"));
    }
    if core == this {
        return Err(format!("{path} resolved to this wrapper (the full-path native override is missing)"));
    }
    let lookup = |name: &str| -> *const c_void {
        let cname: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        unsafe { GetProcAddress(core, cname.as_ptr()) }
    };
    let stubbed = match rules::resolution(|name| !lookup(name).is_null()) {
        Resolution::MissingHooked(missing) => return Err(format!("NVIDIA's NGX core {path} lacks {}", missing.join(", "))),
        Resolution::Ok { stubbed } => stubbed,
    };
    let traced = tracing();
    TRACE_CALLS.store(traced, Ordering::Relaxed);
    for (i, name) in PASSTHROUGH.iter().enumerate() {
        let p = lookup(name);
        let p = if p.is_null() { missing_export as *const () as usize } else { p as usize };
        // SAFETY: written here only, under the loader lock, before any export can be called.
        unsafe {
            (*core::ptr::addr_of_mut!(REAL_TARGETS))[i] = p;
            (*core::ptr::addr_of_mut!(PASSTHROUGH_TARGETS))[i] = if traced { PASSTHROUGH_TRACED[i] as *const () as usize } else { p };
        }
        trace(format_args!("load: {name} -> {p:#x}{}", if p == missing_export as *const () as usize { " (missing: NVSDK_NGX_Result_Fail)" } else { "" }));
    }
    for (i, name) in HOOKED.iter().enumerate() {
        HOOK_TARGETS[i].store(lookup(name) as *mut c_void, Ordering::Release);
        trace(format_args!("load: {name} -> {:p} (hooked)", lookup(name)));
    }
    let mut text = format!("{} at {core:p}", module_path(core));
    if !stubbed.is_empty() {
        let hidden = hide_exports(this, &stubbed);
        text += &format!(
            "; lacks {} (taken out of this module's exports: {hidden} of {}; a caller that bound one before gets NVSDK_NGX_Result_Fail)",
            stubbed.join(", "),
            stubbed.len()
        );
    }
    Ok(text)
}

/// Takes `names` out of this module's export table (their addresses set to 0, which Wine's `GetProcAddress` answers
/// with NULL: `crate::pe`), so a game asking for an export NVIDIA's core lacks finds none, as from the core. How many
/// were taken out.
fn hide_exports(this: Handle, names: &[&str]) -> usize {
    // SAFETY: `this` is this module's mapped image; its headers are readable, and the slice covers SizeOfImage.
    let image = unsafe {
        let headers = core::slice::from_raw_parts(this as *const u8, 0x400);
        let Some(size) = crate::pe::size_of_image(headers) else { return 0 };
        core::slice::from_raw_parts(this as *const u8, size)
    };
    let mut hidden = 0;
    for name in names {
        let Some(at) = crate::pe::export_entry(image, name) else { continue };
        // SAFETY: a 4-byte entry inside the image; made writable for the write and put back.
        unsafe {
            let entry = (this as *mut u8).add(at) as *mut c_void;
            let mut old = 0u32;
            if VirtualProtect(entry, 4, PAGE_READWRITE, &mut old) != 0 {
                core::ptr::write_unaligned(entry as *mut u32, 0);
                VirtualProtect(entry, 4, old, &mut old);
                hidden += 1;
                trace(format_args!("load: {name} taken out of the exports (NVIDIA's core lacks it)"));
            }
        }
    }
    hidden
}

/// Loads NVIDIA's core and resolves the exports here, under the loader lock, which Microsoft's DllMain guidance
/// advises against (`LoadLibrary` from `DllMain`). It is sound here: the passthrough exports are bare jumps through
/// the resolved pointers, so they must be set before any export can be called, and no other point comes before
/// that; Wine's loader lock is recursive, so the nested load of NVIDIA's core (and its own `DllMain`) runs on this
/// thread without waiting; the core is loaded by its full path, overridden native, so it cannot be this module again
/// (checked: [`resolve`] refuses a load that returns this module); and nothing here waits on another thread.
/// The ring is opened here too, only where the layer is on ([`rules::ring_wanted`]).
///
/// # Safety
/// Called by the Windows loader.
#[no_mangle]
pub unsafe extern "system" fn DllMain(this: Handle, reason: u32, _reserved: *mut c_void) -> i32 {
    const DLL_PROCESS_ATTACH: u32 = 1;
    const DLL_PROCESS_DETACH: u32 = 0;
    if reason == DLL_PROCESS_DETACH {
        trace(format_args!("unload: DLL_PROCESS_DETACH ({})", if _reserved.is_null() { "FreeLibrary" } else { "process exit" }));
        return 1;
    }
    if reason != DLL_PROCESS_ATTACH {
        return 1;
    }
    DisableThreadLibraryCalls(this);
    let _ = clock();
    let exe = module_path(core::ptr::null_mut());
    let opened = open_trace();
    trace(format_args!(
        "load: Neural Forge NGX wrapper {} in {exe} as {}; trace {opened:?}; NVIDIA_WINE_DLL_DIR={:?} WINEDLLOVERRIDES={:?}",
        env!("CARGO_PKG_VERSION"),
        module_path(this),
        std::env::var("NVIDIA_WINE_DLL_DIR").ok(),
        std::env::var("WINEDLLOVERRIDES").ok()
    ));
    let t = now_ms();
    let resolved = resolve(this);
    trace(format_args!("load: resolved in {:.3} ms (passthrough calls traced: {})", now_ms() - t, TRACE_CALLS.load(Ordering::Relaxed)));
    match resolved {
        Ok(core) => {
            let ring = if rules::ring_wanted(|name| std::env::var(name).ok()) {
                open_ring()
            } else {
                Err("not created: the Neural Forge layer is not on in this process (NEURAL_FORGE_ENABLE)".into())
            };
            log(&format!("Neural Forge NGX wrapper {} in {exe} as {}; NVIDIA's core {core}; ring {ring:?}", env!("CARGO_PKG_VERSION"), module_path(this)));
            trace(format_args!("load: DllMain done"));
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
