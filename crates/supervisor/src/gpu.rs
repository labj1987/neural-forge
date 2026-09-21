//! NVIDIA GPU detection via `/sys/bus/pci/devices` — no `lspci` dependency needed;
//! the kernel already exposes each PCI device's vendor/device/class IDs as plain
//! text files, which is what upstream's own bash fell back to when `lspci` wasn't
//! available. Since that fallback works unconditionally, it's simpler to make it the
//! only path here rather than also shelling out to `lspci` first.

/// Returns `(vendor_id, device_id)` as 16-bit hex values for the first NVIDIA
/// display-class (`0x03xxxx`) PCI device found, if any.
pub fn detect_nvidia_gpu() -> Option<(u32, u32)> {
    let entries = std::fs::read_dir("/sys/bus/pci/devices").ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let vendor = read_hex(&dir.join("vendor"))?;
        if vendor != 0x10de {
            continue;
        }
        let class = read_hex(&dir.join("class"))?;
        // Class codes are 0xCCSSPP (class, subclass, prog-if); 0x03 is "Display
        // controller".
        if (class >> 16) & 0xff != 0x03 {
            continue;
        }
        let device = read_hex(&dir.join("device"))?;
        return Some((vendor, device));
    }
    None
}

fn read_hex(path: &std::path::Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim().trim_start_matches("0x");
    u32::from_str_radix(trimmed, 16).ok()
}
