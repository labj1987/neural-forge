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
//! (MinGW's missing SEH intrinsics), are upstream's; this is a fresh implementation
//! using `AddVectoredExceptionHandler` declared directly against `kernel32.dll`.
//!
//! The `setjmp` site lives in C (`csrc/guard_shim.c`, built by `build.rs`), not Rust: a
//! function that returns twice needs the compiler to know it does (`returns_twice`), and
//! Rust has no way to say so. From Rust, `nf_guard_run` is an ordinary function that
//! returns once, with 0 (the closure finished) or 1 (the handler jumped back). Everything
//! that must survive the jump lives in memory the jump does not touch: the thread-locals
//! below and the `Slot` in [`guarded`]'s own frame, which sits above the C frame.
//!
//! # After a fault, stop calling
//!
//! The jump abandons the faulting DLL mid-call: any lock it held (its own, or the loader
//! lock) stays held. So the first caught fault latches [`faulted`] for the whole process,
//! and every later [`guarded`] call returns its fail value without running at all. That
//! includes teardown: `ngx::teardown` skips every DLL call and unload once the latch is
//! set, because releasing a feature through a DLL whose locks are held can hang.
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
//!
//! Never guard `LoadLibrary`: a fault inside a `DllMain` leaves the loader lock held, and
//! the next load would deadlock instead of failing. Such a fault is fatal either way.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
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

/// Storage for the C runtime's `jmp_buf`. [`install`] checks the real size
/// (`nf_guard_jmp_buf_size`, 256 bytes on x86_64 mingw-w64) fits.
#[repr(C, align(16))]
struct JmpBuf([u8; 256]);

type GuardBody = unsafe extern "C" fn(*mut c_void);

extern "C" {
    fn nf_guard_jmp_buf_size() -> usize;
    /// `csrc/guard_shim.c`: sets the jump point in `env`, sets `*active`, runs `body(ctx)`,
    /// clears `*active`. Returns 0 when `body` returned, 1 when [`nf_guard_jump`] came back.
    fn nf_guard_run(env: *mut JmpBuf, active: *mut i32, body: GuardBody, ctx: *mut c_void) -> i32;
    /// `longjmp(env, 1)`, without unwinding.
    fn nf_guard_jump(env: *mut JmpBuf) -> !;
}

thread_local! {
    // `UnsafeCell`: the C shim writes the jump point through a raw pointer to it.
    static GUARD_JMP: UnsafeCell<JmpBuf> = const { UnsafeCell::new(JmpBuf([0; 256])) };
    // Written by the C shim through a raw pointer (hence `Cell`), read by the handler.
    static GUARD_ACTIVE: Cell<i32> = const { Cell::new(0) };
    static GUARD_CODE: Cell<u32> = const { Cell::new(0) };
    static GUARD_HITS: Cell<u32> = const { Cell::new(0) };
}

/// The first caught fault's exception code, process-wide; 0 while nothing has faulted.
static FAULTED: AtomicU32 = AtomicU32::new(0);

static INSTALL: Once = Once::new();

/// Installs the process-wide vectored exception handler. Idempotent; call once at
/// helper startup before any guarded NGX call. Every call to [`guarded`] before this
/// runs is not actually guarded at all — a fault would go uncaught.
pub fn install() {
    INSTALL.call_once(|| {
        // SAFETY: a plain query of a C constant.
        let real = unsafe { nf_guard_jmp_buf_size() };
        assert!(real <= std::mem::size_of::<JmpBuf>(), "jmp_buf is {real} bytes, JmpBuf too small");
        // SAFETY: `veh_handler` matches `VectoredHandler`'s signature exactly. `first =
        // 1` puts it first in the chain, so a fault during a `guarded` call is seen
        // here before any handler installed by NGX/the driver/anything else gets a
        // chance to (mis)handle it.
        unsafe {
            AddVectoredExceptionHandler(1, veh_handler);
        }
    });
}

/// The exception code of the first fault [`guarded`] caught in this process, if any. Once
/// set, no further guarded call runs (see the module doc).
pub fn faulted() -> Option<u32> {
    match FAULTED.load(Ordering::Acquire) {
        0 => None,
        code => Some(code),
    }
}

/// Clears the process-wide fault latch. Only for `examples/guard_test.rs`, which checks that
/// the jump point survives many faults in a row; the helper itself never clears it.
#[doc(hidden)]
pub fn clear_fault_latch_for_test() {
    FAULTED.store(0, Ordering::Release);
}

struct Slot<F, R> {
    f: Option<F>,
    out: Option<R>,
}

unsafe extern "C" fn trampoline<F: FnOnce() -> R, R>(ctx: *mut c_void) {
    // SAFETY: `ctx` is the `Slot<F, R>` in `guarded`'s frame, live for this whole call.
    let slot = unsafe { &mut *ctx.cast::<Slot<F, R>>() };
    if let Some(f) = slot.f.take() {
        slot.out = Some(f());
    }
}

/// Runs `f`, catching any hardware exception (access violation, illegal instruction,
/// stack overflow, division fault — whatever the CPU raises) that occurs anywhere
/// inside it, including inside a call across the FFI boundary into `nvngx_dlssnr.dll`.
/// Returns `(f(), 0)` on success, or `(fail_value, exception_code)` — the raw SEH
/// `EXCEPTION_*` code — if a fault fired instead. Once any guarded call in the process
/// has faulted, `f` is not run at all and this returns `(fail_value, first_fault_code)`.
///
/// See the module doc comment for the one discipline this depends on: no local in `f`'s
/// call frames may need `Drop` to run for correctness.
pub fn guarded<F: FnOnce() -> R, R>(f: F, fail_value: R) -> (R, u32) {
    if let Some(code) = faulted() {
        return (fail_value, code);
    }
    GUARD_CODE.with(|c| c.set(0));
    let mut slot = Slot { f: Some(f), out: None };
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    let active_ptr = GUARD_ACTIVE.with(|a| a.as_ptr());
    // SAFETY: both pointers are this thread's own thread-locals, stable for the thread's life;
    // `slot` outlives the call and is only touched by `trampoline::<F, R>` on this thread.
    let jumped = unsafe { nf_guard_run(jmp_ptr, active_ptr, trampoline::<F, R>, std::ptr::from_mut(&mut slot).cast()) };
    match (jumped, slot.out) {
        (0, Some(out)) => (out, 0),
        _ => {
            let code = GUARD_CODE.with(Cell::get);
            // A jump always records a code; keep the result a failure even if it somehow did not.
            (fail_value, if code == 0 { abi_fail_seh() } else { code })
        }
    }
}

fn abi_fail_seh() -> u32 {
    crate::abi::result::FAIL_SEH as u32
}

unsafe extern "system" fn veh_handler(info: *mut ExceptionPointers) -> i32 {
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    if GUARD_ACTIVE.with(Cell::get) == 0 {
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
        GUARD_ACTIVE.with(|a| a.set(0));
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Latched before anything else can run: from here on no guarded call enters a DLL.
    let _ = FAULTED.compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
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
    GUARD_ACTIVE.with(|a| a.set(0));
    let jmp_ptr = GUARD_JMP.with(|j| j.get());
    // SAFETY: `GUARD_ACTIVE` is only nonzero between `nf_guard_run`'s `_setjmp` and its return,
    // on this same thread, so `jmp_ptr` holds a live jump point in a frame still on this stack.
    // This never returns; the thread resumes inside `nf_guard_run`, which returns 1.
    unsafe { nf_guard_jump(jmp_ptr) }
}
