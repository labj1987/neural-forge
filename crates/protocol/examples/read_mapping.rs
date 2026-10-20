//! One-off: looks at an existing mapping at the path given as argv[1] and reports whether it's a
//! valid neural-forge mapping (magic, version and size). Read-only, through
//! `mapping::open_existing`: a file that is not a mapping, or too short to be one, is reported, not
//! mapped, and nothing is written to the file.
use std::sync::atomic::Ordering;

use neural_forge_protocol::mapping::{self, OpenError};

fn main() {
    let path = std::env::args().nth(1).expect("usage: read_mapping <path>");
    let mapping = match mapping::open_existing(&path) {
        Ok(Some(mapping)) => mapping,
        Ok(None) => {
            println!("is_valid: false (no mapping at {path}, or one not initialised yet)");
            return;
        }
        Err(OpenError::WrongVersion { found }) => {
            println!("is_valid: false (version {found}, expected {})", neural_forge_protocol::SHM_VERSION);
            return;
        }
        Err(e) => {
            println!("is_valid: false ({e})");
            return;
        }
    };
    let hdr = mapping.header();
    println!("is_valid: {}", hdr.is_valid());
    println!("magic: {:#x} (expected {:#x})", hdr.magic.load(Ordering::Relaxed), neural_forge_protocol::SHM_MAGIC);
    println!("version: {} (expected {})", hdr.version.load(Ordering::Relaxed), neural_forge_protocol::SHM_VERSION);
    println!("server_state: {}", hdr.server_state.load(Ordering::Relaxed));
    println!("server_heartbeat: {}", hdr.server_heartbeat.load(Ordering::Relaxed));
    println!("enabled: {}", hdr.enabled.load(Ordering::Relaxed));
    println!("intensity: {}", f32::from_bits(hdr.intensity_bits.load(Ordering::Relaxed)));
}
