//! Library half of `neural-forge-helper`, existing so `examples/`/tests can exercise
//! individual modules (the SEH guard, the caller-identity spoof) directly under Wine
//! without needing a real `nvngx_dlssnr.dll` or a full helper run. `main.rs` is a thin
//! binary wrapper around this.
//!
//! The helper only runs on Windows (under Wine/Proton). The modules with no Win32 calls
//! (`abi`, `logging`, `pe`, `selfparam`) also build natively, so their unit tests run without
//! Wine: `cargo +stable test --target x86_64-unknown-linux-gnu -p neural-forge-helper --lib`.

pub mod abi;
#[cfg(windows)]
pub mod frame;
#[cfg(windows)]
pub mod guard;
pub mod logging;
#[cfg(windows)]
pub mod ngx;
#[cfg(windows)]
pub mod optical_flow;
pub mod pe;
pub mod selfparam;
#[cfg(windows)]
pub mod shm;
#[cfg(windows)]
pub mod spoof;
