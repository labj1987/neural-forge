//! The caller-identity spoof.
//!
//! `nvngx_dlssnr.dll` (and `nvngx.dll`, "core") verify their caller is `nvngx.dll`
//! before agreeing to do anything, by calling `GetModuleFileNameW` on their own caller
//! and checking the name it returns. This module patches this process's own imported
//! `GetModuleFileNameW` — specifically, the Import Address Table slot the loader wrote
//! when it resolved that import for the snippet/core module — so it reports
//! `"nvngx.dll"` back to them instead of this helper's own name, defeating that check.
//!
//! **This is the piece Alex explicitly accepted the legal exposure on** (DMCA §1201 /
//! NGX SDK EULA — a risk the original design review weighed and accepted). It is
//! implemented here exactly as designed, isolated in its own module with this comment
//! stating plainly what it does and why, so it is never mistaken for anything else
//! during later maintenance —
//! not to disguise it, but so nobody has to re-derive what this file is for from the
//! bytes alone.
//!
//! Ported for shape from `core/ngx_snippet.cpp`'s `FindImportedFunctionSlot`/
//! `InstallCallerSpoof`/`RemoveCallerSpoof`/`SpoofedGetModuleFileNameW` — the mechanism
//! is upstream's, this is a fresh Rust implementation of it. The PE import walk itself is
//! [`crate::pe`], which has no Win32 dependency and is unit-tested natively.

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

#[link(name = "kernel32")]
extern "system" {
    fn VirtualProtect(address: *mut c_void, size: usize, new_protect: u32, old_protect: *mut u32) -> i32;
    fn FlushInstructionCache(process: *mut c_void, base_address: *const c_void, size: usize) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn GetModuleFileNameW(module: *mut c_void, filename: *mut u16, size: u32) -> u32;
    fn GetModuleHandleW(module_name: *const u16) -> *mut c_void;
    fn SetLastError(code: u32);
}

/// This process's own main-executable module handle (`neural-forge-helper.exe`'s), resolved
/// once via `GetModuleHandleW(NULL)` — the same value the Win32 convention "pass NULL
/// for the calling process's own module" refers to, and the value equivalent to
/// upstream's `g_layerModule` (upstream is a DLL and gets this from `DllMain`; this
/// helper is an .exe, so it asks for it directly instead). [`spoofed_get_module_file_name_w`]
/// treats a query for *either* NULL *or* this exact handle as "asking about me" —
/// `nvngx_dlssnr.dll` may do either, depending on how it resolves its caller.
static THIS_MODULE: std::sync::atomic::AtomicPtr<c_void> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// The real `GetModuleFileNameW`'s address, captured from whichever IAT slot
/// [`install`] first patched, *not* re-resolved via this crate's own
/// `extern "system" { fn GetModuleFileNameW(...); }` declaration.
///
/// This matters even though [`install`] is only ever called against a *different*
/// loaded module (the snippet/core DLL, never this helper's own executable) whose IAT
/// is entirely separate from this helper's own: [`spoofed_get_module_file_name_w`]'s
/// pass-through path calling through this helper's own import instead would only be
/// safe as long as "the module I just patched is never the one my own call resolves
/// through" holds — true today, but not a guarantee this function should quietly rely
/// on when the fix is to just not need the guarantee at all. Confirmed the hard way:
/// an early version of this file called the plain import for pass-through, and a test
/// that (atypically, on purpose, to be self-contained) spoofed *this crate's own test
/// binary* recursed into itself and stack-overflowed under Wine within milliseconds —
/// the exact failure mode this static exists to make structurally impossible instead
/// of "impossible only because nothing does that yet."
static REAL_GET_MODULE_FILE_NAME_W: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

const PAGE_READWRITE: u32 = 0x04;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

pub use crate::pe::find_imported_function_slots;

/// The spoofed caller name every module this is installed against will see reported
/// for this process — i.e., what makes `nvngx_dlssnr.dll`/`nvngx.dll`'s caller-identity
/// check pass.
const AUTHORIZED_CALLER: &[u16] = &[
    b'n' as u16, b'v' as u16, b'n' as u16, b'g' as u16, b'x' as u16, b'.' as u16, b'd' as u16, b'l' as u16,
    b'l' as u16, 0,
];

/// Every slot [`install`] patched, each with its original value so it can be restored on
/// teardown. `AtomicPtr` (rather than a plain `usize`/an FFI call to
/// `InterlockedExchangePointer`, which isn't a real exported function to link against — it's
/// a compiler intrinsic on the C side) both performs the same atomic swap
/// `InterlockedExchangePointer` would and gives the original value back in one operation.
pub struct InstalledSpoof {
    patched: Vec<(*mut usize, usize)>,
}

// SAFETY: `slot` points into a loaded module's own read-write-after-VirtualProtect
// memory for the life of that module; nothing here assumes single-threaded access
// beyond what the atomic swap in `install`/`remove` already guarantees.
unsafe impl Send for InstalledSpoof {}

/// Patches every one of `module`'s `GetModuleFileNameW` IAT slots to call
/// [`spoofed_get_module_file_name_w`] instead of the real one. Returns `None` (and changes
/// nothing) if `module` has no such import — not every module necessarily imports it
/// directly — or if no slot could be patched.
///
/// # Safety
/// `module` must be a valid, currently-loaded module handle that stays loaded for as
/// long as the spoof is installed (i.e., until [`remove`] is called with the returned
/// handle).
pub unsafe fn install(module: *mut c_void) -> Option<InstalledSpoof> {
    // SAFETY: `GetModuleHandleW(NULL)` takes no ownership and cannot fail for a
    // running process asking about itself.
    let this_module = unsafe { GetModuleHandleW(std::ptr::null()) };
    THIS_MODULE.store(this_module, Ordering::Relaxed);

    // SAFETY: `module` is a loaded image (the caller's contract).
    let slots = unsafe { find_imported_function_slots(module, "GetModuleFileNameW") };
    let spoof = spoofed_get_module_file_name_w as *mut ();
    let mut patched = Vec::with_capacity(slots.len());
    for slot in slots {
        // SAFETY: `slot` points into `module`'s own mapped image, bounds-checked against its
        // `SizeOfImage` by the walker; an IAT entry is pointer-sized and pointer-aligned.
        if let Some(original) = unsafe { swap_slot(slot, spoof) } {
            if original == spoof as usize {
                // Already ours (the same module patched twice). Recording it would make the
                // spoof its own "real" function and recurse forever; leave it alone instead.
                crate::log!("[spoof] slot {slot:?} already holds the spoof; left as it is");
                continue;
            }
            // Every module's real `GetModuleFileNameW` import resolves to the same one real
            // implementation in the one real kernel32.dll loaded in this process, so any
            // unpatched slot's original value is the right pass-through target.
            REAL_GET_MODULE_FILE_NAME_W.store(original as *mut (), Ordering::Release);
            patched.push((slot, original));
        }
    }
    (!patched.is_empty()).then_some(InstalledSpoof { patched })
}

/// Atomically writes `value` into the IAT slot at `slot` and returns what was there, or `None`
/// (changing nothing) when the page could not be made writable. Writing back the value it
/// just read is a no-op, which is how an already-spoofed slot is left untouched.
///
/// # Safety
/// `slot` must be a pointer-sized, pointer-aligned IAT entry inside a loaded module.
unsafe fn swap_slot(slot: *mut usize, value: *mut ()) -> Option<usize> {
    let mut old_protect: u32 = 0;
    // SAFETY: one machine word's worth of protection change at a mapped address.
    if unsafe { VirtualProtect(slot.cast(), std::mem::size_of::<usize>(), PAGE_READWRITE, &mut old_protect) } == 0 {
        return None;
    }
    // SAFETY: `slot` is now writable, aligned and pointer-sized; nothing else in this process
    // writes this exact memory concurrently with this swap.
    let atomic_slot = unsafe { AtomicPtr::<()>::from_ptr(slot.cast()) };
    let current = atomic_slot.load(Ordering::Acquire);
    let original = if current == value { current } else { atomic_slot.swap(value, Ordering::AcqRel) };
    let mut discard = 0u32;
    // SAFETY: restoring whatever protection `VirtualProtect` reported was there before.
    unsafe {
        VirtualProtect(slot.cast(), std::mem::size_of::<usize>(), old_protect, &mut discard);
        FlushInstructionCache(GetCurrentProcess(), slot.cast(), std::mem::size_of::<usize>());
    }
    Some(original as usize)
}

/// Restores every slot this [`InstalledSpoof`] patched to its original value.
///
/// # Safety
/// The module it was installed against must still be loaded.
pub unsafe fn remove(spoof: InstalledSpoof) {
    for (slot, original) in spoof.patched {
        // SAFETY: same contract as in `install`.
        unsafe { swap_slot(slot, original as *mut ()) };
    }
}

type GetModuleFileNameWFn = unsafe extern "system" fn(*mut c_void, *mut u16, u32) -> u32;

fn real_get_module_file_name_w_fn(raw: *mut ()) -> Option<GetModuleFileNameWFn> {
    if raw.is_null() {
        return None;
    }
    // SAFETY: the only value ever stored in `REAL_GET_MODULE_FILE_NAME_W` is the
    // original IAT slot value `install()` swapped out -- the real
    // `GetModuleFileNameW`'s address, which has exactly this signature.
    Some(unsafe { std::mem::transmute_copy::<*mut (), GetModuleFileNameWFn>(&raw) })
}

/// The spoofed function itself: reports `"nvngx.dll"` when asked for *this process's*
/// module name (the only case that matters — a module asking `GetModuleFileNameW` about
/// itself, or about anything else, is unaffected and gets the real name), otherwise
/// behaves exactly like the real `GetModuleFileNameW`.
unsafe extern "system" fn spoofed_get_module_file_name_w(module: *mut c_void, filename: *mut u16, size: u32) -> u32 {
    // NULL is the Win32 convention for "the calling process's own module"; the exact
    // handle value set in `install()` is what a caller gets if it instead resolves its
    // caller's module explicitly (e.g. `GetModuleHandleExW` from a return address).
    // `nvngx_dlssnr.dll` may do either -- both mean the same thing, "asking about us".
    let is_this_process = module.is_null() || module == THIS_MODULE.load(Ordering::Relaxed);
    if !is_this_process {
        // The real function, captured before any patch -- never the plain import
        // below, which is exactly the slot this function itself might now be sitting
        // in (see `REAL_GET_MODULE_FILE_NAME_W`'s doc comment for why that's not just
        // a theoretical concern).
        let real = REAL_GET_MODULE_FILE_NAME_W.load(Ordering::Acquire);
        if let Some(real) = real_get_module_file_name_w_fn(real) {
            return unsafe { real(module, filename, size) };
        }
        // Only reachable if this function is somehow being called before any
        // `install()` ever ran to capture the real pointer -- shouldn't happen, but
        // falling through to the plain import is still strictly better than a null
        // function-pointer call.
        return unsafe { GetModuleFileNameW(module, filename, size) };
    }
    if filename.is_null() || size == 0 {
        unsafe { SetLastError(ERROR_INSUFFICIENT_BUFFER) };
        return 0;
    }
    let len = AUTHORIZED_CALLER.len() as u32 - 1; // excluding the NUL
    if size <= len {
        let copy_len = size.saturating_sub(1) as usize;
        // SAFETY: `filename` is valid for `size` u16s per the caller's contract
        // (matching the real `GetModuleFileNameW`'s own contract); `copy_len < size`.
        unsafe {
            std::ptr::copy_nonoverlapping(AUTHORIZED_CALLER.as_ptr(), filename, copy_len);
            *filename.add(copy_len) = 0;
        }
        unsafe { SetLastError(ERROR_INSUFFICIENT_BUFFER) };
        return size;
    }
    // SAFETY: `filename` is valid for at least `AUTHORIZED_CALLER.len()` u16s here
    // (`size > len`, checked above).
    unsafe {
        std::ptr::copy_nonoverlapping(AUTHORIZED_CALLER.as_ptr(), filename, AUTHORIZED_CALLER.len());
    }
    len
}
