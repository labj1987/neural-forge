//! VEH + `setjmp`/`longjmp` exception guard.
//!
//! MinGW has no `__try`/`__except` (those are MSVC-only C extensions) — which is
//! exactly why upstream's own C++ helper uses this technique instead of them: install a
//! process-wide Vectored Exception Handler that, when it fires on this thread while a
//! [`guarded`] call is in flight, performs a non-local jump straight back to the call
//! site via `longjmp` rather than letting the fault (or the OS's normal unwind search)
//! propagate any further. This is what stands between a bad call into
//! `nvngx_dlssnr.dll` — the whole reason this module exists — and this process going
//! down with it.
//!
//! Ported for shape from `core/guard.h`: the *mechanism*, and the reason for it
//! (MinGW's missing SEH intrinsics), are upstream's; this is a fresh Rust
//! implementation using `AddVectoredExceptionHandler` plus a hand-rolled `setjmp`/
//! `longjmp` FFI boundary, declared directly against `kernel32.dll`/the mingw CRT
//! rather than through a higher-level wrapper crate — these specific calls are old,
//! extremely stable parts of the Win32 ABI, and declaring them directly keeps this
//! module auditable without also depending on how some other crate happens to shape
//! its own wrapper for them.
//!
//! # The one discipline this depends on
//!
//! `longjmp` restores the CPU's registers (stack pointer, frame pointer, instruction
//! pointer) directly — it does **not** run any Rust `Drop` impl for a value that was
//! live in a stack frame between the [`guarded`] call and the point of the fault, the
//! same way a C++ destructor in that range would also be skipped by a raw `longjmp`.
//! Keep every [`guarded`] closure to what upstream's own `Guarded()` call sites are:
//! a single FFI call plus `Copy` locals, nothing that owns a heap allocation or a lock
//! that must be released for correctness.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

#[link(name = "kernel32")]
extern "system" {
    fn AddVectoredExceptionHandler(first: u32, handler: VectoredHandler) -> *mut c_void;
}

type VectoredHandler = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

/// <https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-exception_pointers>
#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut c_void,
}

/// <https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-exception_record>
/// Only the leading fields are named; the trailing `ExceptionInformation` array (whose
/// exact length depends on `NumberParameters`) is never read here, so it's left out
/// rather than guessed at.
#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut c_void,
    number_parameters: u32,
    exception_information: [usize; 15],
}

/// Byte offsets of `Rsp` and `Rip` inside the x86-64 `CONTEXT` -- fixed by the Windows ABI, and
/// read straight from the record rather than mirroring the whole 1232-byte structure.
const CONTEXT_RSP: usize = 0x98;
const CONTEXT_RIP: usize = 0xF8;

/// Exception codes Wine raises for every `OutputDebugStringA`/`W`. They are not faults; treating
/// one as a fault would longjmp out of a healthy call and, because feature creation is one-shot,
/// disable the model for the rest of the session.
const DBG_PRINTEXCEPTION_C: u32 = 0x4001_0006;
const DBG_PRINTEXCEPTION_WIDE_C: u32 = 0x4001_000A;

/// After this many caught faults on a thread the guard stands down and lets the process die
/// normally: a fault that keeps recurring is not something to keep absorbing.
const MAX_GUARDED_HITS: u32 = 24;

/// The modules whose faults are worth naming in the log, so a `rip` can be read as
/// `module+offset` rather than a bare address.
#[derive(Clone, Copy)]
pub enum Module {
    Snippet = 0,
    Core = 1,
    Nvapi = 2,
}
const MODULE_NAMES: [&str; 3] = ["nvngx_dlssnr.dll", "nvngx.dll", "nvapi64.dll"];
static RANGE_BASE: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
static RANGE_SIZE: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];

/// Records a loaded PE image's address range, from its own header's `SizeOfImage`.
///
/// # Safety
/// `base` must be the base of a currently-mapped PE image.
pub unsafe fn register_module(which: Module, base: *mut c_void) {
    if base.is_null() {
        return;
    }
    // SAFETY: `base` is a mapped PE image (caller's contract): the DOS header's `e_lfanew` at
    // 0x3C locates the NT headers, and `SizeOfImage` sits 80 bytes into them.
    let size = unsafe {
        let nt = base.cast::<u8>().add(std::ptr::read_unaligned(base.cast::<u8>().add(0x3C).cast::<u32>()) as usize);
        std::ptr::read_unaligned(nt.add(80).cast::<u32>()) as usize
    };
    RANGE_BASE[which as usize].store(base as usize, Ordering::Relaxed);
    RANGE_SIZE[which as usize].store(size, Ordering::Relaxed);
}

fn describe(addr: usize) -> (&'static str, usize) {
    for i in 0..MODULE_NAMES.len() {
        let base = RANGE_BASE[i].load(Ordering::Relaxed);
        let size = RANGE_SIZE[i].load(Ordering::Relaxed);
        if base != 0 && addr >= base && addr < base + size {
            return (MODULE_NAMES[i], addr - base);
        }
    }
    ("exe/other", addr)
}

/// Opaque `jmp_buf` storage, deliberately over-sized: MinGW-w64's real `jmp_buf` on
/// x86_64 is smaller than this, and a buffer larger than the real one is always safe
/// (it just wastes a little thread-local storage) — smaller would not be. The exact
/// size isn't pinned down more precisely than "generous" because that would need
/// checking against the real `<setjmp.h>` on this toolchain, which isn't installed on
/// this dev machine yet; `setjmp`/`longjmp` themselves are declared directly against
/// the mingw CRT below, so a size mismatch here would show up immediately as memory
/// corruption the first time this runs under Wine/Proton, not as a silent bug --
/// exactly the kind of thing to check for in this crate's first real runtime test.
#[repr(C, align(16))]
struct JmpBuf([u8; 256]);

extern "C" {
    /// The CRT's real `_setjmp(env, frame)`. The mingw `setjmp` macro passes
    /// `__builtin_frame_address(0)` as `frame`, which makes the matching `longjmp` *unwind* to
    /// that frame (running `RtlUnwindEx` over every frame in between). From inside a vectored
    /// exception handler that is not safe, so this always passes a null frame: `longjmp` then
    /// just restores registers.
    fn _setjmp(env: *mut JmpBuf, frame: *mut c_void) -> i32;
    fn longjmp(env: *mut JmpBuf, val: i32) -> !;
}

thread_local! {
    // `UnsafeCell`, not a plain `JmpBuf`: `setjmp` writes through the raw pointer this
    // hands out, and mutating through a pointer derived from a `&JmpBuf` with nothing
    // in between would be exactly the kind of aliasing violation `UnsafeCell` exists to
    // make legal -- same reasoning as `neural_forge_protocol::ShmHeader`'s seqlock-guarded
    // fields for the same underlying reason (shared, externally-written memory).
    static GUARD_JMP: UnsafeCell<JmpBuf> = UnsafeCell::new(JmpBuf([0; 256]));
    static GUARD_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static GUARD_CODE: Cell<u32> = const { Cell::new(0) };
    static GUARD_HITS: Cell<u32> = const { Cell::new(0) };
}

static INSTALL: Once = Once::new();

/// Installs the process-wide vectored exception handler. Idempotent; call once at
/// helper startup before any guarded NGX call. Every call to [`guarded`] before this
/// runs is not actually guarded at all — a fault would go uncaught.
pub fn install() {
    INSTALL.call_once(|| {
        // SAFETY: `veh_handler` matches `VectoredHandler`'s signature exactly. `first =
        // 1` puts it first in the chain, so a fault during a `guarded` call is seen
        // here before any handler installed by NGX/the driver/anything else gets a
        // chance to (mis)handle it.
        unsafe {
            AddVectoredExceptionHandler(1, veh_handler);
        }
    });
}

/// Runs `f`, catching any hardware exception (access violation, illegal instruction,
/// stack overflow, division fault — whatever the CPU raises) that occurs anywhere
/// inside it, including inside a call across the FFI boundary into `nvngx_dlssnr.dll`.
/// Returns `(f(), 0)` on success, or `(fail_value, exception_code)` — the raw SEH
/// `EXCEPTION_*` code — if a fault fired instead.
///
/// See the module doc comment for the one discipline this depends on: no local in `f`'s
/// call frames may need `Drop` to run for correctness.
pub fn guarded<F: FnOnce() -> R, R>(f: F, fail_value: R) -> (R, u32) {
    GUARD_CODE.with(|c| c.set(0));
    // SAFETY: `GUARD_JMP` is this thread's own thread-local storage, stable for the
    // life of the thread and never touched by any other thread; getting a raw pointer
    // to it and handing that pointer to `setjmp` is exactly what `setjmp` requires.
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    let did_jump = unsafe { _setjmp(jmp_ptr, std::ptr::null_mut()) };
    if did_jump == 0 {
        GUARD_ACTIVE.with(|a| a.set(true));
        let result = f();
        // Only reached if `f` returned normally -- a fault during `f` never gets here,
        // it jumps straight to the `else` branch below via `longjmp` instead.
        GUARD_ACTIVE.with(|a| a.set(false));
        (result, 0)
    } else {
        (fail_value, GUARD_CODE.with(|c| c.get()))
    }
}

unsafe extern "system" fn veh_handler(info: *mut ExceptionPointers) -> i32 {
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    if !GUARD_ACTIVE.with(Cell::get) {
        // Nothing we're watching is running on this thread right now -- not our fault
        // to handle, let the normal search (the debugger, the process's default
        // handler, ultimately termination) continue.
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: the OS guarantees `info` and `(*info).exception_record` are valid for the
    // duration of this call when it invokes a registered vectored handler.
    let record = unsafe { &*(*info).exception_record };
    let code = record.exception_code;
    if code == DBG_PRINTEXCEPTION_C || code == DBG_PRINTEXCEPTION_WIDE_C {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    GUARD_CODE.with(|c| c.set(code));
    let hits = GUARD_HITS.with(|h| {
        h.set(h.get() + 1);
        h.get()
    });
    if hits >= MAX_GUARDED_HITS {
        GUARD_ACTIVE.with(|a| a.set(false));
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: `context_record` is valid for this call; `Rsp`/`Rip` are at these fixed offsets.
    let (rip, rsp) = unsafe {
        let ctx = (*info).context_record.cast::<u8>();
        (
            std::ptr::read_unaligned(ctx.add(CONTEXT_RIP).cast::<usize>()),
            std::ptr::read_unaligned(ctx.add(CONTEXT_RSP).cast::<usize>()),
        )
    };
    // The return address at the top of the stack is only meaningful for a fault on a call
    // boundary, and `rsp` is only dereferenced when it looks like user-space memory.
    let ret = if rsp > 0x10000 && rsp < 0x7fff_ffff_ffff { unsafe { std::ptr::read_unaligned(rsp as *const usize) } } else { 0 };
    let fault = if record.number_parameters > 1 { record.exception_information[1] } else { 0 };
    let (rip_mod, rip_off) = describe(rip);
    let (ret_mod, ret_off) = describe(ret);
    crate::log!(
        "[veh] code={code:#x} rip={rip:#x} ({rip_mod}+{rip_off:#x}) ret={ret:#x} ({ret_mod}+{ret_off:#x}) fault={fault:#x} rw={} hit={hits}",
        if record.number_parameters > 0 { record.exception_information[0] } else { 0 }
    );
    GUARD_ACTIVE.with(|a| a.set(false));
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    // SAFETY: `jmp_ptr` was set up by a `setjmp` call on this same thread earlier in
    // this same call chain (guaranteed by `GUARD_ACTIVE` only being true between that
    // `setjmp` and the matching `guarded()` call returning) -- exactly the precondition
    // `longjmp` requires. This never returns; the thread resumes at that `setjmp` site.
    unsafe { longjmp(jmp_ptr, 1) }
}
