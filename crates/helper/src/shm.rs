//! The Windows side of the shared-memory mapping: opens the same file the Linux layer
//! does, translated to the path Wine exposes it under, and maps it the same
//! 64 KiB-aligned way for the same reason (`VK_EXT_external_memory_host`'s
//! `minImportedHostPointerAlignment`).
//!
//! Ported for shape from `helper/main.cpp`'s `ShmOpen` — same path translation, same
//! aligned-mapping retry loop — using raw `kernel32` declarations for the same reason
//! as the rest of this crate (see `spoof.rs`'s module doc comment).

use std::ffi::c_void;

use neural_forge_protocol::{answer_offset, answer_offset_slot, proxy_offset, proxy_offset_slot, shm_default_path, shm_total_bytes, MAX_FRAME};

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
    fn CreateDirectoryW(path: *const u16, security: *const c_void) -> i32;
    fn SetFilePointerEx(file: *mut c_void, distance: i64, new_pointer: *mut i64, method: u32) -> i32;
    fn SetEndOfFile(file: *mut c_void) -> i32;
    fn CreateFileMappingW(
        file: *mut c_void,
        security: *const c_void,
        protect: u32,
        max_size_high: u32,
        max_size_low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFileEx(
        mapping: *mut c_void,
        access: u32,
        offset_high: u32,
        offset_low: u32,
        size: usize,
        base_address: *mut c_void,
    ) -> *mut c_void;
    fn UnmapViewOfFile(base_address: *mut c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

const GENERIC_READ: u32 = 0x8000_0000;
const GENERIC_WRITE: u32 = 0x4000_0000;
const FILE_SHARE_READ: u32 = 0x1;
const FILE_SHARE_WRITE: u32 = 0x2;
const OPEN_ALWAYS: u32 = 4;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
const FILE_BEGIN: u32 = 0;
const PAGE_READWRITE: u32 = 0x04;
const FILE_MAP_READ: u32 = 0x0004;
const FILE_MAP_WRITE: u32 = 0x0002;
const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

pub struct ShmMapping {
    file: *mut c_void,
    mapping: *mut c_void,
    pub header: *mut neural_forge_protocol::ShmHeader,
}

// SAFETY: same reasoning as `neural_forge_layer::shm::ShmClient` (see its `unsafe impl Send`)
// -- every access through `header` goes through `ShmHeader`'s own atomics.
unsafe impl Send for ShmMapping {}

/// `NEURAL_FORGE_SHM` holds the POSIX path the Linux layer uses; Wine exposes the host
/// filesystem under `Z:\`, so `/tmp/neural-forge-1000/shm.bin` becomes `Z:\tmp\neural-forge-1000\shm.bin`.
fn windows_path(posix_path: &str) -> Vec<u16> {
    let translated: String = std::iter::once('Z').chain(std::iter::once(':')).chain(
        posix_path.chars().map(|c| if c == '/' { '\\' } else { c })
    ).collect();
    translated.encode_utf16().chain(std::iter::once(0)).collect()
}

fn parent_dir_utf16(path_utf16: &[u16]) -> Option<Vec<u16>> {
    let sep = '\\' as u16;
    let pos = path_utf16[..path_utf16.len().saturating_sub(1)].iter().rposition(|&c| c == sep)?;
    if pos == 0 {
        return None;
    }
    let mut dir: Vec<u16> = path_utf16[..pos].to_vec();
    dir.push(0);
    Some(dir)
}

/// Opens (creating if necessary) the mapping and maps its header region.
///
/// # Safety
/// Must only be called once per `ShmMapping` — this creates OS handles the returned
/// value owns and closes on [`ShmMapping::close`].
pub fn open() -> Option<ShmMapping> {
    let posix_path = neural_forge_protocol::env::var("NEURAL_FORGE_SHM").filter(|s| !s.is_empty()).unwrap_or_else(shm_default_path);
    if !neural_forge_protocol::isolated_path(&posix_path) { return None; }
    let win_path = windows_path(&posix_path);

    if let Some(dir) = parent_dir_utf16(&win_path) {
        // SAFETY: `dir` is a valid, NUL-terminated UTF-16 string for the call's
        // duration. The result is intentionally ignored: "already exists" and any
        // other failure are both handled the same way below (the `CreateFileW` call
        // simply fails if the directory genuinely isn't there).
        unsafe {
            CreateDirectoryW(dir.as_ptr(), std::ptr::null());
        }
    }

    // SAFETY: `win_path` is a valid, NUL-terminated UTF-16 string for the call's
    // duration; every other argument is a plain value, not a pointer needing its own
    // validity contract.
    let file = unsafe {
        CreateFileW(
            win_path.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if file == INVALID_HANDLE_VALUE {
        return None;
    }

    let total = neural_forge_protocol::shm_total_bytes() as i64;
    // SAFETY: `file` is a valid, open, writable file handle from the call above.
    let sized = unsafe {
        SetFilePointerEx(file, total, std::ptr::null_mut(), FILE_BEGIN) != 0 && SetEndOfFile(file) != 0
    };
    if !sized {
        unsafe { CloseHandle(file) };
        return None;
    }

    // SAFETY: `file` is valid and now sized to at least `shm_total_bytes()`.
    let mapping = unsafe { CreateFileMappingW(file, std::ptr::null(), PAGE_READWRITE, 0, 0, std::ptr::null()) };
    if mapping.is_null() {
        unsafe { CloseHandle(file) };
        return None;
    }

    // Windows hands out views at the allocation granularity, but the API doesn't
    // promise 64 KiB specifically, and Wine doesn't guarantee it either. The transport
    // import (`VK_EXT_external_memory_host`, on the Linux side) demands a 64 KiB-aligned
    // host pointer, so try aligned hint addresses first; a view that lands unaligned
    // just means this side keeps using staging copies instead of the zero-copy path --
    // not a failure, so the plain, unhinted mapping below is always the fallback.
    let mut base: *mut c_void = std::ptr::null_mut();
    const ALIGN: usize = 64 * 1024;
    const HINT_BASE: usize = 0x0000_2000_0000_0000;
    let map_size = shm_total_bytes();
    for i in 0..128usize {
        let hint = (HINT_BASE + i * (2 << 20)) as *mut c_void;
        // SAFETY: `mapping` is valid; a hinted address that the OS refuses is simply
        // not used (Windows either honors the hint exactly or fails the call outright
        // for `MapViewOfFileEx`, unlike Linux's `MAP_FIXED_NOREPLACE` semantics, but
        // trying several hints and moving on from any that fail is safe either way).
        let view = unsafe { MapViewOfFileEx(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, map_size, hint) };
        if !view.is_null() {
            if (view as usize) % ALIGN == 0 {
                base = view;
                break;
            }
            unsafe {
                UnmapViewOfFile(view);
            }
        }
    }
    if base.is_null() {
        // SAFETY: `mapping` is valid; requesting an OS-chosen address is always legal.
        base = unsafe { MapViewOfFileEx(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, map_size, std::ptr::null_mut()) };
    }
    if base.is_null() {
        unsafe {
            CloseHandle(mapping);
            CloseHandle(file);
        }
        return None;
    }

    let header = base.cast::<neural_forge_protocol::ShmHeader>();
    // SAFETY: just mapped above, `shm_total_bytes()` is large enough for `ShmHeader`
    // (enforced at compile time in `neural_forge_protocol`) plus both pixel regions.
    let hdr = unsafe { &*header };
    if !hdr.is_valid() {
        hdr.init_defaults();
    }

    Some(ShmMapping { file, mapping, header })
}

impl ShmMapping {
    fn pixel_base(&self) -> *mut u8 {
        self.header.cast::<u8>()
    }

    pub fn read_proxy(&self, out: &mut [u8]) -> usize {
        let n = out.len().min(MAX_FRAME);
        // SAFETY: `pixel_base()` is the start of this process's own mapping of the
        // full `shm_total_bytes()` region (see `open` above); `proxy_offset()..+n` is
        // in bounds for any `n <= MAX_FRAME` by that region's own definition.
        unsafe {
            std::ptr::copy_nonoverlapping(self.pixel_base().add(proxy_offset()), out.as_mut_ptr(), n);
        }
        n
    }

    /// Writes `bytes` (truncated to `MAX_FRAME`) into the answer region -- the model's
    /// raw output, for the layer's composition pass to read back.
    pub fn write_answer(&self, bytes: &[u8]) {
        let n = bytes.len().min(MAX_FRAME);
        // SAFETY: same reasoning as `read_proxy`, mirrored for the answer region.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pixel_base().add(answer_offset()), n);
        }
    }

    /// The given slot's proxy and answer regions' own addresses and capacity within
    /// this process's mapping -- for `frame::FrameResources`'s own
    /// `VK_EXT_external_memory_host` import (see its module doc comment), the only
    /// legitimate reason anything outside this module needs these addresses at all;
    /// every other caller already goes through
    /// [`Self::read_proxy`]/[`Self::write_answer`]/[`Self::frame_regions`].
    pub fn proxy_and_answer_regions(&self, slot: usize) -> ((*mut u8, usize), (*mut u8, usize)) {
        let base = self.pixel_base();
        // SAFETY: both stay within the `shm_total_bytes()` mapping `open` established,
        // same reasoning as `frame_regions`' own pointer arithmetic.
        unsafe { ((base.add(proxy_offset_slot(slot)), MAX_FRAME), (base.add(answer_offset_slot(slot)), MAX_FRAME)) }
    }

    /// Returns disjoint views of this slot's request's proxy and answer regions. The
    /// helper owns each request from observing that slot's `seq_req` until publishing
    /// its `seq_resp`, so it can write the answer directly into shared memory instead
    /// of copying through a second process-local frame buffer. Protocol v3
    /// (`docs/PROTOCOL_V3_DESIGN.md`) gives slot 0 and slot 1 disjoint regions, so this is
    /// still sound when the helper is processing both slots, one after another.
    ///
    /// # Safety
    /// The caller must only use these views while it owns the current request on this
    /// slot.
    pub unsafe fn frame_regions(&self, slot: usize, bytes: usize) -> (&[u8], &mut [u8]) {
        let n = bytes.min(MAX_FRAME);
        let base = self.pixel_base();
        unsafe {
            (
                std::slice::from_raw_parts(base.add(proxy_offset_slot(slot)), n),
                std::slice::from_raw_parts_mut(base.add(answer_offset_slot(slot)), n),
            )
        }
    }

    /// # Safety
    /// Must not be called while any other code still holds a reference derived from
    /// `self.header`.
    pub unsafe fn close(self) {
        // SAFETY: `header`/`mapping`/`file` were all produced by `open()` above and
        // are each closed/unmapped in the reverse order they were created.
        unsafe {
            UnmapViewOfFile(self.header.cast());
            CloseHandle(self.mapping);
            CloseHandle(self.file);
        }
    }
}
