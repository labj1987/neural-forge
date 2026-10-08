//! One-off: opens an existing mapping at the path given as argv[1] and reports
//! whether it's a valid neural-forge mapping (magic, version and size).
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
    println!("server_state: {}", hdr.server_state.load(std::sync::atomic::Ordering::Relaxed));
    println!("server_heartbeat: {}", hdr.server_heartbeat.load(std::sync::atomic::Ordering::Relaxed));
    println!("enabled: {}", hdr.enabled.load(std::sync::atomic::Ordering::Relaxed));
    println!("intensity: {}", f32::from_bits(hdr.intensity_bits.load(std::sync::atomic::Ordering::Relaxed)));
}
