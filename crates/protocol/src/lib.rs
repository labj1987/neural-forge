//! Shared-memory contract between the processes of neural-forge:
//!
//!   the Vulkan layer   in each game: holds DLSS's input and runs the model (the native backend),
//!                      or captures the frame after the upscaler and composes the model's answer,
//!                      which its in-process model server (`native_post`) computes
//!   the GTK4 GUI / CLI write settings and read status
//!
//! Everything here is plain atomics in a file mapping, so a process dying leaves the others reading
//! a consistent — if stale — picture. `ShmHeader` is `#[repr(C)]` and built entirely from
//! `AtomicU32` fields (the free-text fields are arrays of them), so its layout is fixed by the
//! field order alone.
//!
//! Layout of the mapping:
//!
//!   `[0, HEADER_BYTES)`                        `ShmHeader`
//!   `[HEADER_BYTES, +MAX_FRAME)`                slot 0's proxy (the layer's capture)
//!   `[HEADER_BYTES + MAX_FRAME, +MAX_FRAME)`    slot 0's answer (the model's output)
//!   `[HEADER_BYTES + MAX_FRAME*2, +MAX_FRAME)`  slot 1's proxy
//!   `[HEADER_BYTES + MAX_FRAME*3, +MAX_FRAME)`  slot 1's answer
//!
//! v3 (`docs/PROTOCOL_V3_DESIGN.md`) added the second slot so the layer can have two
//! requests outstanding at once.
//!
//! This is a from-scratch protocol for a from-scratch implementation — it is not wire
//! compatible with, and never attaches to, a mapping left behind by any other DLSS
//! neural-rendering project. `SHM_MAGIC` exists specifically so a stale mapping from
//! anything else is always rejected and reinitialized rather than half-read.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod enums;
mod header;
#[cfg(unix)]
pub mod mapping;
pub mod env;
pub mod history;
pub mod rebuild;
mod path;
pub mod persist;
#[cfg(unix)]
pub mod private_dir;
mod slot;
#[cfg(unix)]
pub mod state_log;

pub use header::{load64, setting_bounds, store64, ShmHeader, Tuning, SETTING_BOUNDS};
pub use path::{isolated_path, shm_default_path, shm_runtime_dir};
pub use slot::{next_request, Slot};

/// Identifies a neural-forge mapping. Bumped only if the protocol is ever forked into an
/// incompatible variant; a mismatch here means "not our mapping at all", not "an older
/// version of our mapping" — that distinction is `SHM_VERSION`'s job.
pub const SHM_MAGIC: u32 = u32::from_le_bytes(*b"NFR1");

/// The wire contract version (v2 added BGRA8 and a motion payload region; v3 adds a
/// second independent request/response slot — see `docs/PROTOCOL_V3_DESIGN.md`; v7
/// removes the motion payload region and `frame_mvec_valid` again, since motion is now
/// estimated inside the model server; v8 appends the layer's GPU timestamps; v9 the pre-upscaler path's status; v10 the model server's
/// per-request wall time, for the hold's hand-off breakdown; v11 `seq_eval`, which tells a model
/// answer from an echo; v12 drops `preset` and `sharpness`, which the 310.8 model never reads; v13
/// `native_running`, so the GUI can hide the helper-only settings while the native backend runs; v14
/// drops `scaling_downscaler`, which nothing read: the model never runs above the frame's size; v15
/// removes the Windows helper's fields (the per-pass settings, motion settings, rebuild spacing, its VRAM and
/// feature counts, its reason string, the DMA-BUF exchange) and renames what the in-process model
/// server still writes from `helper_*` to `server_*`; v16 appends `device_lost_at`, so the GUI and
/// `doctor` can say the GPU device was lost without a log).
/// The header layout version. A mismatch (matching magic, different version) means
/// another process in the chain is out of date; the GUI/CLI side refuses such a header
/// untouched (`mapping::OpenError::WrongVersion`) rather than half-read or reinitialize it.
pub const SHM_VERSION: u32 = 16;

pub const MAX_W: u32 = 7680;
pub const MAX_H: u32 = 4320;

/// Eight bytes a pixel: the float16 HDR proxy needs them, and the 8-bit path simply
/// uses the first half of each region. The mapping is file-backed and sparse, so an
/// SDR session never commits the second half.
pub const MAX_FRAME: usize = MAX_W as usize * MAX_H as usize * 8;

/// Whether a `(width, height, proxy_format)` triple read out of shared memory is one
/// the model server may size image resources from. The mapping is written by another
/// process, so none of it is trusted: dimensions must be non-zero, within
/// `MAX_W`/`MAX_H`, and even (the proxy's chroma-friendly grid), and the format one the
/// protocol defines.
pub fn frame_dims_valid(width: u32, height: u32, proxy_format: u32) -> bool {
    use enums::proxy_format::{BGRA8, RGBA16F, RGBA8};
    width != 0
        && height != 0
        && width <= MAX_W
        && height <= MAX_H
        && width.is_multiple_of(2)
        && height.is_multiple_of(2)
        && matches!(proxy_format, RGBA8 | RGBA16F | BGRA8)
}

pub const HEADER_BYTES: usize = 65536;

pub const REASON_BYTES: usize = 192;
pub const NAME_BYTES: usize = 128;

/// Total size of the mapping: header, slot 0's proxy/answer, and slot 1's proxy/answer.
pub const fn shm_total_bytes() -> usize {
    HEADER_BYTES + MAX_FRAME * 4
}

/// Byte offset of slot 0's proxy region (the frame the layer hands the model) within
/// the mapping. Both sides derive this the same way rather than hardcoding
/// `HEADER_BYTES` separately, so a future header resize can't silently desync them.
pub const fn proxy_offset() -> usize {
    HEADER_BYTES
}

/// Byte offset of slot 0's answer region (the model's raw output, for the
/// composition pass) within the mapping.
pub const fn answer_offset() -> usize {
    HEADER_BYTES + MAX_FRAME
}

/// Byte offset of slot 1's proxy region — the second, independent in-flight request
/// v3 adds. See `docs/PROTOCOL_V3_DESIGN.md`.
pub const fn proxy_b_offset() -> usize { HEADER_BYTES + MAX_FRAME * 2 }

/// Byte offset of slot 1's answer region.
pub const fn answer_b_offset() -> usize { HEADER_BYTES + MAX_FRAME * 3 }

/// The slot's proxy region: `proxy_offset()` or `proxy_b_offset()`.
pub const fn proxy_offset_slot(slot: Slot) -> usize {
    match slot {
        Slot::Primary => proxy_offset(),
        Slot::Secondary => proxy_b_offset(),
    }
}

/// The slot's answer region. See [`proxy_offset_slot`].
pub const fn answer_offset_slot(slot: Slot) -> usize {
    match slot {
        Slot::Primary => answer_offset(),
        Slot::Secondary => answer_b_offset(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_dims_validation_rejects_hostile_headers() {
        use super::*;
        assert!(frame_dims_valid(1920, 1080, enums::proxy_format::RGBA8));
        assert!(frame_dims_valid(MAX_W, MAX_H, enums::proxy_format::RGBA16F));
        assert!(!frame_dims_valid(0, 1080, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(1920, 0, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(MAX_W + 2, 1080, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(1920, MAX_H + 2, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(1921, 1080, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(1920, 1081, enums::proxy_format::RGBA8));
        assert!(!frame_dims_valid(1920, 1080, enums::proxy_format::UNKNOWN));
        assert!(!frame_dims_valid(1920, 1080, 99));
    }

    use super::*;

    #[test]
    fn offset_slot_helpers_match_the_named_functions() {
        assert_eq!(proxy_offset_slot(Slot::Primary), proxy_offset());
        assert_eq!(proxy_offset_slot(Slot::Secondary), proxy_b_offset());
        assert_eq!(answer_offset_slot(Slot::Primary), answer_offset());
        assert_eq!(answer_offset_slot(Slot::Secondary), answer_b_offset());
    }

    #[test]
    fn every_region_is_disjoint_and_fits_the_mapping() {
        let regions = [
            ("header", 0, HEADER_BYTES),
            ("proxy", proxy_offset(), MAX_FRAME),
            ("answer", answer_offset(), MAX_FRAME),
            ("proxy_b", proxy_b_offset(), MAX_FRAME),
            ("answer_b", answer_b_offset(), MAX_FRAME),
        ];
        for (i, (name_a, start_a, len_a)) in regions.iter().enumerate() {
            assert!(start_a + len_a <= shm_total_bytes(), "{name_a} region overruns shm_total_bytes()");
            for (name_b, start_b, len_b) in &regions[i + 1..] {
                let overlap = *start_a < start_b + len_b && *start_b < start_a + len_a;
                assert!(!overlap, "{name_a} and {name_b} regions overlap");
            }
        }
    }
}
