//! Library half of `neural-forge-helper`, existing so `examples/`/tests can exercise
//! individual modules (the SEH guard, the caller-identity spoof) directly under Wine
//! without needing a real `nvngx_dlssnr.dll` or a full helper run. `main.rs` is a thin
//! binary wrapper around this.

pub mod abi;
pub mod frame;
pub mod guard;
pub mod logging;
pub mod ngx;
pub mod optical_flow;
pub mod selfparam;
pub mod shm;
pub mod spoof;
