//! Reading the wrapper's own export table in memory, to take out the exports NVIDIA's core lacks (`win.rs`,
//! [`crate::rules::Resolution`]): a caller that asks `GetProcAddress` for one then gets nothing, as it would from the
//! core, instead of the stub that answers `NVSDK_NGX_Result_Fail`. A caller that looks an entry point up and takes
//! another path when it is missing (a newer driver's function on an older driver) then takes the same path as without
//! the wrapper. Wine's `GetProcAddress` gives NULL for an export whose address is 0
//! (`dlls/ntdll/loader.c`, `find_ordinal_export`).
//!
//! Plain byte reading of a mapped PE image (no Windows), so it is tested on Linux.

fn u16_at(img: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(img.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(img: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(img.get(at..at + 4)?.try_into().ok()?))
}

/// The image's size from its PE32+ headers (`OptionalHeader.SizeOfImage`), for making the slice the others read.
pub fn size_of_image(headers: &[u8]) -> Option<usize> {
    let nt = u32_at(headers, 0x3c)? as usize;
    (u32_at(headers, nt)? == 0x4550 && u16_at(headers, nt + 24)? == 0x20b).then_some(())?;
    Some(u32_at(headers, nt + 24 + 56)? as usize)
}

/// The offset, in the mapped image `img`, of the export address table entry (a 4-byte RVA) of the export `name`.
pub fn export_entry(img: &[u8], name: &str) -> Option<usize> {
    let nt = u32_at(img, 0x3c)? as usize;
    if u32_at(img, nt)? != 0x4550 || u16_at(img, nt + 24)? != 0x20b {
        return None;
    }
    let dir = u32_at(img, nt + 24 + 112)? as usize;
    if dir == 0 {
        return None;
    }
    let functions = u32_at(img, dir + 20)? as usize;
    let names = u32_at(img, dir + 24)? as usize;
    let function_table = u32_at(img, dir + 28)? as usize;
    let name_table = u32_at(img, dir + 32)? as usize;
    let ordinal_table = u32_at(img, dir + 36)? as usize;
    for i in 0..names {
        // A name that cannot be read is skipped, not the end of the search.
        let Some(at) = u32_at(img, name_table + 4 * i) else { continue };
        let Some(bytes) = img.get(at as usize..at as usize + name.len() + 1) else { continue };
        if &bytes[..name.len()] == name.as_bytes() && bytes[name.len()] == 0 {
            let ordinal = u16_at(img, ordinal_table + 2 * i)? as usize;
            return (ordinal < functions).then_some(function_table + 4 * ordinal);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal PE32+ image with an export directory naming `names` (address i+1 each).
    fn image(names: &[&str]) -> Vec<u8> {
        let mut img = vec![0u8; 0x1000];
        let nt = 0x80usize;
        img[0x3c..0x40].copy_from_slice(&(nt as u32).to_le_bytes());
        img[nt..nt + 4].copy_from_slice(&0x4550u32.to_le_bytes());
        img[nt + 24..nt + 26].copy_from_slice(&0x20bu16.to_le_bytes());
        img[nt + 24 + 56..nt + 24 + 60].copy_from_slice(&0x1000u32.to_le_bytes());
        let dir = 0x200usize;
        img[nt + 24 + 112..nt + 24 + 116].copy_from_slice(&(dir as u32).to_le_bytes());
        let (funcs, name_ptrs, ords, strings) = (0x300usize, 0x400usize, 0x500usize, 0x600usize);
        let put = |img: &mut Vec<u8>, at: usize, v: u32| img[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut img, dir + 20, names.len() as u32);
        put(&mut img, dir + 24, names.len() as u32);
        put(&mut img, dir + 28, funcs as u32);
        put(&mut img, dir + 32, name_ptrs as u32);
        put(&mut img, dir + 36, ords as u32);
        let mut s = strings;
        for (i, n) in names.iter().enumerate() {
            put(&mut img, funcs + 4 * i, i as u32 + 1);
            put(&mut img, name_ptrs + 4 * i, s as u32);
            img[ords + 2 * i..ords + 2 * i + 2].copy_from_slice(&(i as u16).to_le_bytes());
            img[s..s + n.len()].copy_from_slice(n.as_bytes());
            s += n.len() + 1;
        }
        img
    }

    #[test]
    fn an_export_is_found_by_its_whole_name() {
        let img = image(&["NVSDK_NGX_VULKAN_Init", "NVSDK_NGX_VULKAN_Init_Ext2"]);
        assert_eq!(size_of_image(&img), Some(0x1000));
        assert_eq!(export_entry(&img, "NVSDK_NGX_VULKAN_Init"), Some(0x300));
        assert_eq!(export_entry(&img, "NVSDK_NGX_VULKAN_Init_Ext2"), Some(0x304));
        assert_eq!(export_entry(&img, "NVSDK_NGX_VULKAN_Ini"), None, "a prefix is not the name");
        assert_eq!(export_entry(&img, "NVSDK_NGX_VULKAN_Init_Ext"), None);
    }

    /// A name pointer past the image's end is skipped, and the names after it are still found.
    #[test]
    fn an_unreadable_name_is_skipped() {
        let mut img = image(&["A_FIRST", "B_SECOND"]);
        img[0x400..0x404].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        assert_eq!(export_entry(&img, "B_SECOND"), Some(0x304));
        assert_eq!(export_entry(&img, "A_FIRST"), None);
    }

    #[test]
    fn not_a_pe32_plus_image() {
        let mut img = image(&["A"]);
        img[0x80] = 0;
        assert_eq!(export_entry(&img, "A"), None);
        assert_eq!(size_of_image(&img), None);
        assert_eq!(export_entry(&[0u8; 16], "A"), None);
    }
}
