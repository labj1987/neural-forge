//! PE32+ import-table walking for [`crate::spoof`]: finds the Import Address Table slots a
//! loaded module calls a `KERNEL32` function through.
//!
//! Pure pointer arithmetic over an image already in memory, with no Win32 calls, so it builds
//! on any target and its unit tests run natively against a synthetic image
//! (`cargo +stable test --target x86_64-unknown-linux-gnu -p neural-forge-helper --lib`).
//!
//! Every byte offset below is from the documented, ABI-stable PE32+ layout (`winnt.h`'s
//! `IMAGE_DOS_HEADER`/`IMAGE_NT_HEADERS64`/`IMAGE_OPTIONAL_HEADER64`/`IMAGE_IMPORT_DESCRIPTOR`)
//! rather than a Rust transcription of those structs: a single misplaced field in a ~30-field
//! struct would silently shift every offset after it, where hardcoded offsets are directly
//! checkable against the documentation. Every read is bounded by the image's own
//! `SizeOfImage`.

use std::ffi::{c_void, CStr};

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

/// The headers of a mapped image live in its first page; the NT headers up to and including
/// the import directory entry must fit there.
const HEADER_PAGE: usize = 4096;

/// `sizeof(IMAGE_IMPORT_DESCRIPTOR)`.
pub const IMPORT_DESCRIPTOR_SIZE: usize = 20;
pub const IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK: usize = 0;
pub const IMPORT_DESCRIPTOR_NAME: usize = 12;
pub const IMPORT_DESCRIPTOR_FIRST_THUNK: usize = 16;

/// `IMAGE_ORDINAL_FLAG64`: set in a thunk's value when the import is by ordinal rather than
/// by name (no name to compare against, so such thunks are skipped).
pub const ORDINAL_FLAG64: u64 = 0x8000_0000_0000_0000;

/// The import library names a `KERNEL32` function may come through: the DLL itself, or one of
/// the ApiSet forwarders a module built against a modern SDK imports instead.
const KERNEL32_NAMES: [&str; 3] =
    ["KERNEL32.dll", "api-ms-win-core-libraryloader-l1-2-0.dll", "api-ms-win-core-libraryloader-l1-1-0.dll"];

unsafe fn read_u16(base: *const u8, offset: usize) -> u16 {
    unsafe { base.add(offset).cast::<u16>().read_unaligned() }
}
unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
    unsafe { base.add(offset).cast::<u32>().read_unaligned() }
}
unsafe fn read_u64(base: *const u8, offset: usize) -> u64 {
    unsafe { base.add(offset).cast::<u64>().read_unaligned() }
}

/// Reads a NUL-terminated ASCII string at `base + rva`, bounded so a corrupt RVA can't walk
/// memory forever looking for a terminator that was never coming.
unsafe fn read_c_str_at_rva<'a>(base: *const u8, rva: u32, size_of_image: u32) -> Option<&'a CStr> {
    if rva == 0 || rva >= size_of_image {
        return None;
    }
    // SAFETY: bounded scan -- this never reads past `size_of_image` bytes from `base`, which
    // is the whole mapped module (the caller's contract).
    let slice = unsafe { std::slice::from_raw_parts(base.add(rva as usize), (size_of_image - rva) as usize) };
    let end = slice.iter().position(|&b| b == 0)?;
    CStr::from_bytes_with_nul(&slice[..=end]).ok()
}

/// Walks `module`'s own PE import table for `function_name` imported from `KERNEL32` (or one
/// of its ApiSet forwarders), through *every* import descriptor, and returns a pointer to each
/// Import Address Table slot holding that function's resolved address. A module can import
/// the same function through more than one descriptor; each one is a separate slot its call
/// sites may use.
///
/// # Safety
/// `module` must be the base address of a mapped PE image (a real `HMODULE`, or a buffer laid
/// out the same way) that is readable for its whole `SizeOfImage` and for at least its first
/// header page.
pub unsafe fn find_imported_function_slots(module: *mut c_void, function_name: &str) -> Vec<*mut usize> {
    let base = module.cast::<u8>();
    let mut slots = Vec::new();

    // SAFETY: the DOS/NT header reads stay inside the first header page (checked before each
    // read that depends on `e_lfanew`); everything after that is bounded by `size_of_image`.
    unsafe {
        if read_u16(base, DOS_E_MAGIC) != DOS_SIGNATURE {
            return slots;
        }
        let nt_offset = read_u32(base, DOS_E_LFANEW) as usize;
        let opt_offset = nt_offset + NT_OPTIONAL_HEADER_OFFSET;
        if opt_offset + OPT_IMPORT_DIRECTORY + 8 > HEADER_PAGE {
            return slots;
        }
        if read_u32(base, nt_offset) != NT_SIGNATURE || read_u16(base, opt_offset + OPT_MAGIC) != NT_OPTIONAL_HDR64_MAGIC {
            return slots;
        }
        let size_of_image = read_u32(base, opt_offset + OPT_SIZE_OF_IMAGE);
        let import_rva = read_u32(base, opt_offset + OPT_IMPORT_DIRECTORY);
        let import_size = read_u32(base, opt_offset + OPT_IMPORT_DIRECTORY + 4);
        if import_rva == 0 || import_rva >= size_of_image {
            return slots;
        }

        // The directory's own size, clipped to the image: a corrupt size can't walk past it.
        let import_dir_end = import_rva.saturating_add(import_size).min(size_of_image) as usize;
        let mut desc_offset = import_rva as usize;
        while desc_offset + IMPORT_DESCRIPTOR_SIZE <= import_dir_end {
            let name_rva = read_u32(base, desc_offset + IMPORT_DESCRIPTOR_NAME);
            if name_rva == 0 {
                break; // the all-zero terminator descriptor
            }
            let is_kernel32 = read_c_str_at_rva(base, name_rva, size_of_image)
                .and_then(|n| n.to_str().ok())
                .is_some_and(|n| KERNEL32_NAMES.iter().any(|k| n.eq_ignore_ascii_case(k)));
            if is_kernel32 {
                let original_first_thunk = read_u32(base, desc_offset + IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK);
                let first_thunk = read_u32(base, desc_offset + IMPORT_DESCRIPTOR_FIRST_THUNK);
                if original_first_thunk != 0 && first_thunk != 0 {
                    if let Some(slot) = scan_thunks(base, original_first_thunk, first_thunk, size_of_image, function_name) {
                        slots.push(slot);
                    }
                }
            }
            desc_offset += IMPORT_DESCRIPTOR_SIZE;
        }
    }
    slots
}

/// # Safety
/// Same contract as [`find_imported_function_slots`].
unsafe fn scan_thunks(base: *const u8, original_first_thunk_rva: u32, first_thunk_rva: u32, size_of_image: u32, function_name: &str) -> Option<*mut usize> {
    let (mut name_thunk, mut addr_thunk) = (original_first_thunk_rva as usize, first_thunk_rva as usize);
    let end = size_of_image as usize;
    while name_thunk + 8 <= end && addr_thunk + 8 <= end {
        // SAFETY: both thunks bounds-checked against `size_of_image` just above.
        let thunk_value = unsafe { read_u64(base, name_thunk) };
        if thunk_value == 0 {
            return None; // end of this descriptor's thunk arrays
        }
        if thunk_value & ORDINAL_FLAG64 == 0 {
            // The low 31 bits of a by-name thunk are the RVA of an IMAGE_IMPORT_BY_NAME
            // {Hint: u16, Name: [u8]} -- the name starts two bytes past that RVA.
            let import_by_name_rva = (thunk_value & 0x7FFF_FFFF) as u32;
            // SAFETY: bounds-checked by `read_c_str_at_rva`.
            if let Some(name) = unsafe { read_c_str_at_rva(base, import_by_name_rva.wrapping_add(2), size_of_image) } {
                if name.to_str() == Ok(function_name) {
                    // SAFETY: in bounds (checked above). This is the IAT slot itself -- the address
                    // the loader wrote the resolved function pointer into, and the address every
                    // call site compiled against this import actually calls through.
                    return Some(unsafe { base.add(addr_thunk) }.cast_mut().cast::<usize>());
                }
            }
        }
        name_thunk += 8;
        addr_thunk += 8;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal PE32+ image laid out in memory the way the loader maps one: DOS header, NT
    /// headers with only the fields the walker reads, and an import directory.
    struct Image {
        bytes: Vec<u8>,
    }

    const IMPORTS: u32 = 0x400; // import descriptors
    const SIZE: u32 = 0x2000;

    impl Image {
        fn new() -> Self {
            let mut img = Image { bytes: vec![0u8; SIZE as usize] };
            img.put_u16(DOS_E_MAGIC, DOS_SIGNATURE);
            img.put_u32(DOS_E_LFANEW, 0x80);
            img.put_u32(0x80, NT_SIGNATURE);
            let opt = 0x80 + NT_OPTIONAL_HEADER_OFFSET;
            img.put_u16(opt + OPT_MAGIC, NT_OPTIONAL_HDR64_MAGIC);
            img.put_u32(opt + OPT_SIZE_OF_IMAGE, SIZE);
            img.put_u32(opt + OPT_IMPORT_DIRECTORY, IMPORTS);
            img
        }
        fn put_u16(&mut self, at: usize, v: u16) {
            self.bytes[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
        fn put_u32(&mut self, at: usize, v: u32) {
            self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        fn put_u64(&mut self, at: usize, v: u64) {
            self.bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
        fn put_str(&mut self, at: usize, s: &str) {
            self.bytes[at..at + s.len()].copy_from_slice(s.as_bytes());
            self.bytes[at + s.len()] = 0;
        }
        fn set_directory_size(&mut self, size: u32) {
            self.put_u32(0x80 + NT_OPTIONAL_HEADER_OFFSET + OPT_IMPORT_DIRECTORY + 4, size);
        }
        /// Descriptor `index` importing `functions` (by name, `None` = by ordinal) from `dll`.
        /// Each descriptor gets its own 0x200-byte area for names and thunk arrays.
        fn descriptor(&mut self, index: usize, dll: &str, functions: &[Option<&str>]) -> u32 {
            let area = 0x800 + index * 0x200;
            let (name, ilt, iat, hints) = (area, area + 0x40, area + 0xA0, area + 0x100);
            self.put_str(name, dll);
            let mut hint = hints;
            for (i, f) in functions.iter().enumerate() {
                let value = match f {
                    Some(f) => {
                        self.put_str(hint + 2, f);
                        let v = hint as u64;
                        hint += 2 + f.len() + 1 + 1;
                        v
                    }
                    None => ORDINAL_FLAG64 | 7,
                };
                self.put_u64(ilt + i * 8, value);
                self.put_u64(iat + i * 8, 0x1111_0000 + i as u64);
            }
            let d = IMPORTS as usize + index * IMPORT_DESCRIPTOR_SIZE;
            self.put_u32(d + IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK, ilt as u32);
            self.put_u32(d + IMPORT_DESCRIPTOR_NAME, name as u32);
            self.put_u32(d + IMPORT_DESCRIPTOR_FIRST_THUNK, iat as u32);
            iat as u32
        }
        fn find(&mut self, function: &str) -> Vec<usize> {
            let base = self.bytes.as_mut_ptr();
            // SAFETY: `bytes` is SIZE bytes, laid out as a PE image with SizeOfImage = SIZE.
            unsafe { find_imported_function_slots(base.cast(), function) }.into_iter().map(|p| p as usize - base as usize).collect()
        }
    }

    #[test]
    fn finds_the_slot_by_name_skipping_ordinals_and_other_dlls() {
        let mut img = Image::new();
        img.descriptor(0, "USER32.dll", &[Some("GetModuleFileNameW")]);
        let iat = img.descriptor(1, "KERNEL32.dll", &[Some("Sleep"), None, Some("GetModuleFileNameW")]);
        img.set_directory_size(3 * IMPORT_DESCRIPTOR_SIZE as u32);
        assert_eq!(img.find("GetModuleFileNameW"), vec![iat as usize + 16]);
        assert_eq!(img.find("Sleep"), vec![iat as usize]);
        assert!(img.find("CreateFileW").is_empty());
    }

    #[test]
    fn continues_through_every_descriptor() {
        let mut img = Image::new();
        let a = img.descriptor(0, "kernel32.dll", &[Some("GetModuleFileNameW")]);
        let b = img.descriptor(1, "api-ms-win-core-libraryloader-l1-2-0.dll", &[Some("GetModuleFileNameW")]);
        img.set_directory_size(3 * IMPORT_DESCRIPTOR_SIZE as u32);
        assert_eq!(img.find("GetModuleFileNameW"), vec![a as usize, b as usize]);
    }

    #[test]
    fn a_directory_size_past_the_image_stops_at_the_image() {
        let mut img = Image::new();
        let a = img.descriptor(0, "KERNEL32.dll", &[Some("GetModuleFileNameW")]);
        // No terminator within the directory's claimed size, which runs far past SizeOfImage:
        // the walk must still end inside the image.
        for i in 1..((SIZE - IMPORTS) as usize / IMPORT_DESCRIPTOR_SIZE) {
            let d = IMPORTS as usize + i * IMPORT_DESCRIPTOR_SIZE;
            if d + IMPORT_DESCRIPTOR_SIZE > 0x800 {
                break;
            }
            img.put_u32(d + IMPORT_DESCRIPTOR_NAME, 0x7F0);
        }
        img.put_str(0x7F0, "x.dll");
        img.set_directory_size(u32::MAX - IMPORTS);
        assert_eq!(img.find("GetModuleFileNameW"), vec![a as usize]);
    }

    #[test]
    fn rejects_malformed_headers() {
        let mut img = Image::new();
        img.descriptor(0, "KERNEL32.dll", &[Some("GetModuleFileNameW")]);
        img.set_directory_size(2 * IMPORT_DESCRIPTOR_SIZE as u32);
        assert_eq!(img.find("GetModuleFileNameW").len(), 1);

        let mut bad = Image { bytes: img.bytes.clone() };
        bad.put_u32(DOS_E_LFANEW, 0x7FFF_FFF0); // NT headers outside the header page
        assert!(bad.find("GetModuleFileNameW").is_empty());

        let mut bad = Image { bytes: img.bytes.clone() };
        bad.put_u16(0x80 + NT_OPTIONAL_HEADER_OFFSET + OPT_MAGIC, 0x10B); // PE32, not PE32+
        assert!(bad.find("GetModuleFileNameW").is_empty());

        let mut bad = Image { bytes: img.bytes.clone() };
        bad.put_u32(0x80 + NT_OPTIONAL_HEADER_OFFSET + OPT_IMPORT_DIRECTORY, SIZE); // directory outside the image
        assert!(bad.find("GetModuleFileNameW").is_empty());
    }

    #[test]
    fn thunks_running_off_the_end_of_the_image_stop() {
        let mut img = Image::new();
        img.descriptor(0, "KERNEL32.dll", &[Some("Sleep")]);
        img.set_directory_size(2 * IMPORT_DESCRIPTOR_SIZE as u32);
        // Point the name thunks at the last 4 bytes of the image: an 8-byte read there would
        // run past it.
        img.put_u32(IMPORTS as usize + IMPORT_DESCRIPTOR_ORIGINAL_FIRST_THUNK, SIZE - 4);
        assert!(img.find("Sleep").is_empty());
    }
}
