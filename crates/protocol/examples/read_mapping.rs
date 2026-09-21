//! One-off: opens an existing mapping at the path given as argv[1] and reports
//! whether it's a valid neural-forge mapping. Used to verify true cross-toolchain interop --
//! a mapping created by the Windows-side helper (built with mingw-w64, running under
//! Wine) read back correctly by Linux-side code (built with the system toolchain).
use std::os::fd::AsRawFd;

fn main() {
    let path = std::env::args().nth(1).expect("usage: read_mapping <path>");
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
    let map = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            neural_forge_protocol::HEADER_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    assert_ne!(map, libc::MAP_FAILED);
    let hdr = unsafe { &*(map as *const neural_forge_protocol::ShmHeader) };
    println!("is_valid: {}", hdr.is_valid());
    println!("magic: {:#x} (expected {:#x})", hdr.magic.load(std::sync::atomic::Ordering::Relaxed), neural_forge_protocol::SHM_MAGIC);
    println!("version: {} (expected {})", hdr.version.load(std::sync::atomic::Ordering::Relaxed), neural_forge_protocol::SHM_VERSION);
    println!("helper_state: {}", hdr.helper_state.load(std::sync::atomic::Ordering::Relaxed));
    println!("heartbeat: {}", hdr.heartbeat.load(std::sync::atomic::Ordering::Relaxed));
    println!("enabled: {}", hdr.enabled.load(std::sync::atomic::Ordering::Relaxed));
    println!("intensity: {}", f32::from_bits(hdr.intensity_bits.load(std::sync::atomic::Ordering::Relaxed)));
}
