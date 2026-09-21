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
//! NGX SDK EULA — see the review and
//! `/home/alex/.claude/plans/breezy-napping-waffle.md`). It is implemented here exactly
//! as designed, isolated in its own module with this comment stating plainly what it
//! does and why, so it is never mistaken for anything else during later maintenance —
//! not to disguise it, but so nobody has to re-derive what this file is for from the
//! bytes alone.
//!
//! Ported for shape from `core/ngx_snippet.cpp`'s `FindImportedFunctionSlot`/
//! `InstallCallerSpoof`/`RemoveCallerSpoof`/`SpoofedGetModuleFileNameW` — the mechanism
//! is upstream's, this is a fresh Rust implementation of it using raw PE parsing
//! against offsets from the (extremely stable, unchanged for decades) documented
//! `IMAGE_NT_HEADERS64`/`IMAGE_OPTIONAL_HEADER64` layout, rather than modeling every
//! field of those structs in Rust — a single misplaced field in a ~30-field struct
//! transcription would silently shift every offset after it, where hardcoded offsets
//! against long-stable documentation are directly checkable against that
//! documentation instead.

use std::ffi::{c_void, CStr};
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

/// Every byte offset below is from the documented, ABI-stable PE32+ layout
/// (`winnt.h`'s `IMAGE_DOS_HEADER`/`IMAGE_NT_HEADERS64`/`IMAGE_OPTIONAL_HEADER64`/
/// `IMAGE_IMPORT_DESCRIPTOR`), not derived from any struct definition in this crate.
mod pe {
    pub const DOS_E_MAGIC: usize = 0x00;
    pub const DOS_E_LFANEW: usize = 0x3C;
    pub const DOS_SIGNATURE: u16 = 0x5A4D; // "MZ"

    pub const NT_SIGNATURE: u32 = 0x0000_4550; // "PE\0\0"
    pub const NT_OPTIONAL_HDR64_MAGIC: u16 = 0x20B;
    /// `sizeof(IMAGE_FILE_HEADER)` (20 bytes) -- OptionalHeader starts right after it.
    pub const NT_OPTIONAL_HEADER_OFFSET: usize = 4 + 20;

    pub const OPT_MAGIC: usize = 0x00;
    pub const OPT_SIZE_OF_IMAGE: usize = 56;
    /// `DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]`, i.e. `112 + 1 * sizeof(IMAGE_DATA_DIRECTORY)`.
    pub const OPT_IMPORT_DIRECTORY: usize = 120;

    /// `sizeof(IMAGE_IMPORT_DESCRIPTOR)`.
    pub const IMPORT_DESCRIPTOR_SIZE: usize = 20;
    pub const IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK: usize = 0;
    pub const IMPORT_DESCRIPTOR_NAME: usize = 12;
    pub const IMPORT_DESCRIPTOR_FIRST_THUNK: usize = 16;

    /// `IMAGE_ORDINAL_FLAG64`: set in a thunk's value when the import is by ordinal
    /// rather than by name (no name to compare against, so such thunks are skipped).
    pub const ORDINAL_FLAG64: u64 = 0x8000_0000_0000_0000;
}

unsafe fn read_u16(base: *const u8, offset: usize) -> u16 {
    unsafe { base.add(offset).cast::<u16>().read_unaligned() }
}
unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
    unsafe { base.add(offset).cast::<u32>().read_unaligned() }
}
unsafe fn read_u64(base: *const u8, offset: usize) -> u64 {
    unsafe { base.add(offset).cast::<u64>().read_unaligned() }
}

/// Reads a NUL-terminated ASCII string at `base + rva`, bounded so a corrupt/malicious
/// RVA can't walk memory forever looking for a terminator that was never coming.
unsafe fn read_c_str_at_rva<'a>(base: *const u8, rva: u32, size_of_image: u32) -> Option<&'a CStr> {
    if rva == 0 || rva >= size_of_image {
        return None;
    }
    let ptr = unsafe { base.add(rva as usize) }.cast::<i8>();
    // SAFETY: bounded scan, same reasoning as the RVA check above -- this never reads
    // past `size_of_image` bytes from `base`, which is the whole mapped module.
    let max_len = (size_of_image - rva) as usize;
    let slice = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), max_len) };
    let end = slice.iter().position(|&b| b == 0)?;
    // SAFETY: `slice[..=end]` is exactly the bytes `CStr::from_ptr` would itself scan
    // (`end` is the position of the first, and by construction only, NUL byte in this
    // sub-slice), so this is equivalent to calling `CStr::from_ptr`, just pre-bounded.
    Some(unsafe { CStr::from_bytes_with_nul_unchecked(&slice[..=end]) })
}

/// Walks `module`'s own PE import table for `function_name` imported from
/// `KERNEL32.dll` (or one of its two ApiSet forwarders — a module built against a
/// modern SDK may import through those instead of the real DLL name), and returns a
/// pointer to the Import Address Table slot holding that function's resolved address.
///
/// Patching *that* slot is what makes every call site in `module` that calls the
/// function normally (which, for an imported function, always means "through the
/// IAT" — that's what an import is) see the patched value instead.
///
/// # Safety
/// `module` must be the base address of a currently-loaded, valid PE image (a real
/// `HMODULE`).
pub unsafe fn find_imported_function_slot(module: *mut c_void, function_name: &str) -> Option<*mut usize> {
    let base = module.cast::<u8>();

    // SAFETY: every read below is bounds-checked against `size_of_image` once it's
    // known, and the DOS/NT header reads are of a fixed, always-present region (every
    // valid PE image has these headers, unconditionally, at these fixed offsets).
    unsafe {
        if read_u16(base, pe::DOS_E_MAGIC) != pe::DOS_SIGNATURE {
            return None;
        }
        let nt_offset = read_u32(base, pe::DOS_E_LFANEW) as usize;
        if read_u32(base, nt_offset) != pe::NT_SIGNATURE {
            return None;
        }
        let opt_offset = nt_offset + pe::NT_OPTIONAL_HEADER_OFFSET;
        if read_u16(base, opt_offset + pe::OPT_MAGIC) != pe::NT_OPTIONAL_HDR64_MAGIC {
            return None;
        }
        let size_of_image = read_u32(base, opt_offset + pe::OPT_SIZE_OF_IMAGE);

        let import_dir_offset = opt_offset + pe::OPT_IMPORT_DIRECTORY;
        let import_rva = read_u32(base, import_dir_offset);
        let import_size = read_u32(base, import_dir_offset + 4);
        if import_rva == 0 || import_rva >= size_of_image {
            return None;
        }

        let mut desc_offset = import_rva as usize;
        let import_dir_end = (import_rva + import_size) as usize;
        while desc_offset + pe::IMPORT_DESCRIPTOR_SIZE <= import_dir_end {
            let name_rva = read_u32(base, desc_offset + pe::IMPORT_DESCRIPTOR_NAME);
            if name_rva == 0 {
                break;
            }
            let Some(lib_name) = read_c_str_at_rva(base, name_rva, size_of_image) else {
                break;
            };
            let is_kernel32 = lib_name
                .to_str()
                .map(|s| {
                    s.eq_ignore_ascii_case("KERNEL32.dll")
                        || s.eq_ignore_ascii_case("api-ms-win-core-libraryloader-l1-2-0.dll")
                        || s.eq_ignore_ascii_case("api-ms-win-core-libraryloader-l1-1-0.dll")
                })
                .unwrap_or(false);
            if is_kernel32 {
                let original_first_thunk = read_u32(base, desc_offset + pe::IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK);
                let first_thunk = read_u32(base, desc_offset + pe::IMPORT_DESCRIPTOR_FIRST_THUNK);
                if original_first_thunk != 0 && first_thunk != 0 {
                    if let Some(slot) = scan_thunks(base, original_first_thunk, first_thunk, size_of_image, function_name)
                    {
                        return Some(slot);
                    }
                }
            }
            desc_offset += pe::IMPORT_DESCRIPTOR_SIZE;
        }
    }
    None
}

/// # Safety
/// Same contract as [`find_imported_function_slot`].
unsafe fn scan_thunks(
    base: *const u8,
    original_first_thunk_rva: u32,
    first_thunk_rva: u32,
    size_of_image: u32,
    function_name: &str,
) -> Option<*mut usize> {
    let mut name_thunk_rva = original_first_thunk_rva;
    let mut addr_thunk_rva = first_thunk_rva;
    loop {
        if name_thunk_rva >= size_of_image || addr_thunk_rva >= size_of_image {
            return None;
        }
        // SAFETY: both RVAs just bounds-checked against `size_of_image` above.
        let thunk_value = unsafe { read_u64(base, name_thunk_rva as usize) };
        if thunk_value == 0 {
            return None; // end of this descriptor's thunk arrays
        }
        if thunk_value & pe::ORDINAL_FLAG64 == 0 {
            // The low 31 bits of a by-name thunk are the RVA of an IMAGE_IMPORT_BY_NAME
            // {Hint: u16, Name: [u8]} -- the name starts two bytes past that RVA.
            let import_by_name_rva = thunk_value as u32;
            // SAFETY: bounds-checked by `read_c_str_at_rva`.
            if let Some(name) = unsafe { read_c_str_at_rva(base, import_by_name_rva.wrapping_add(2), size_of_image) } {
                if name.to_str() == Ok(function_name) {
                    // SAFETY: this is the IAT slot itself -- the address the loader
                    // wrote the resolved function pointer into, and the address every
                    // call site compiled against this import actually calls through.
                    let slot = unsafe { base.add(addr_thunk_rva as usize) } as *mut usize;
                    return Some(slot);
                }
            }
        }
        name_thunk_rva += 8;
        addr_thunk_rva += 8;
    }
}

/// The spoofed caller name every module this is installed against will see reported
/// for this process — i.e., what makes `nvngx_dlssnr.dll`/`nvngx.dll`'s caller-identity
/// check pass.
const AUTHORIZED_CALLER: &[u16] = &[
    b'n' as u16, b'v' as u16, b'n' as u16, b'g' as u16, b'x' as u16, b'.' as u16, b'd' as u16, b'l' as u16,
    b'l' as u16, 0,
];

/// The slot's original value, so it can be restored on teardown. `AtomicPtr` (rather
/// than a plain `usize`/an FFI call to `InterlockedExchangePointer`, which isn't a real
/// exported function to link against — it's a compiler intrinsic on the C side) both
/// performs the same atomic swap `InterlockedExchangePointer` would and gives the
/// original value back in one operation.
pub struct InstalledSpoof {
    slot: *mut usize,
    original: usize,
}

// SAFETY: `slot` points into a loaded module's own read-write-after-VirtualProtect
// memory for the life of that module; nothing here assumes single-threaded access
// beyond what the atomic swap in `install`/`remove` already guarantees.
unsafe impl Send for InstalledSpoof {}

/// Patches `module`'s `GetModuleFileNameW` IAT slot to call [`spoofed_get_module_file_name_w`]
/// instead of the real one. Returns `None` (and changes nothing) if `module` has no
/// such import — not every module necessarily imports it directly.
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

    let slot = unsafe { find_imported_function_slot(module, "GetModuleFileNameW") }?;

    let mut old_protect: u32 = 0;
    // SAFETY: `slot` is a valid pointer into `module`'s own mapped image (returned by
    // `find_imported_function_slot`, which only ever returns offsets bounds-checked
    // against that image's own `SizeOfImage`); one machine word's worth of protection
    // change at that address is exactly what patching the slot needs.
    let unprotected = unsafe {
        VirtualProtect(
            slot.cast(),
            std::mem::size_of::<usize>(),
            PAGE_READWRITE,
            &mut old_protect,
        )
    } != 0;
    if !unprotected {
        return None;
    }

    // SAFETY: `slot` is now writable (the `VirtualProtect` above succeeded), points at
    // a properly aligned pointer-sized slot (an IAT entry is always pointer-sized and
    // pointer-aligned by construction), and nothing else in this process treats this
    // exact memory as anything other than "the value the loader put here" concurrently
    // with this swap.
    let atomic_slot = unsafe { AtomicPtr::<()>::from_ptr(slot.cast()) };
    let original = atomic_slot.swap(spoofed_get_module_file_name_w as *mut (), Ordering::AcqRel) as usize;
    // Every module's real `GetModuleFileNameW` import resolves to the same one real
    // implementation in the one real kernel32.dll loaded in this process, so it's
    // correct to overwrite this unconditionally on every `install()` call, whichever
    // module was just patched.
    REAL_GET_MODULE_FILE_NAME_W.store(original as *mut (), Ordering::Release);

    // SAFETY: restoring whatever protection `VirtualProtect` reported was there before
    // -- `old_protect` was populated by the call above.
    let mut discard = 0u32;
    unsafe {
        VirtualProtect(slot.cast(), std::mem::size_of::<usize>(), old_protect, &mut discard);
        FlushInstructionCache(GetCurrentProcess(), slot.cast(), std::mem::size_of::<usize>());
    }

    Some(InstalledSpoof { slot, original })
}

/// Restores the slot this [`InstalledSpoof`] patched to its original value.
///
/// # Safety
/// `spoof.slot` must still point at valid, writable-after-`VirtualProtect` memory --
/// true as long as the module it was installed against is still loaded.
pub unsafe fn remove(spoof: InstalledSpoof) {
    let mut old_protect: u32 = 0;
    // SAFETY: same contract as the `VirtualProtect` call in `install`.
    let unprotected = unsafe {
        VirtualProtect(
            spoof.slot.cast(),
            std::mem::size_of::<usize>(),
            PAGE_READWRITE,
            &mut old_protect,
        )
    } != 0;
    if !unprotected {
        return;
    }
    // SAFETY: same reasoning as the `from_ptr` call in `install` above.
    let atomic_slot = unsafe { AtomicPtr::<()>::from_ptr(spoof.slot.cast()) };
    atomic_slot.store(spoof.original as *mut (), Ordering::Release);
    let mut discard = 0u32;
    unsafe {
        VirtualProtect(spoof.slot.cast(), std::mem::size_of::<usize>(), old_protect, &mut discard);
        FlushInstructionCache(GetCurrentProcess(), spoof.slot.cast(), std::mem::size_of::<usize>());
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
