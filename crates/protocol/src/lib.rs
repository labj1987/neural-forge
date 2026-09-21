//! Shared-memory contract between the three processes that make up neural-forge:
//!
//!   the Linux Vulkan layer   captures the frame, runs the composition, presents the result
//!   the Windows helper       owns the NGX model and runs it (today under Wine/Proton; a
//!                            future native-Linux NGX helper drops in here unchanged, since
//!                            this crate is the whole contract either side needs to agree on)
//!   the GTK4 GUI / CLI       write settings and read status
//!
//! Everything here is plain atomics in a file mapping, so no side needs the others'
//! toolchain and a process dying leaves the others reading a consistent — if stale —
//! picture. `ShmHeader` is `#[repr(C)]` and built entirely from `AtomicU32` and
//! `UnsafeCell<[u8; N]>` fields specifically so its layout matches what a C11/C++
//! `std::atomic<uint32_t>` of the same field, in the same position, would produce —
//! that's what makes a single mmap'd region a valid contract between two different
//! toolchains/processes in the first place.
//!
//! Layout of the mapping:
//!
//!   `[0, HEADER_BYTES)`                        `ShmHeader`
//!   `[HEADER_BYTES, +MAX_FRAME)`                slot 0's proxy (the layer's capture)
//!   `[HEADER_BYTES + MAX_FRAME, +MAX_FRAME)`    slot 0's answer (the model's output)
//!   `[HEADER_BYTES + MAX_FRAME*2, +MAX_FRAME)`  the motion payload (shared, slot 0 only —
//!                                                see `docs/PROTOCOL_V3_DESIGN.md`)
//!   `[HEADER_BYTES + MAX_FRAME*3, +MAX_FRAME)`  slot 1's proxy
//!   `[HEADER_BYTES + MAX_FRAME*4, +MAX_FRAME)`  slot 1's answer
//!
//! v3 (`docs/PROTOCOL_V3_DESIGN.md`) added the second slot so the layer can have two
//! requests outstanding at once — never blocked with an idle wire slot while a
//! GPU-captured frame is ready to send, even though the helper still drains both
//! slots' NGX evaluation one at a time (see that doc for why the model side stays
//! serialized while the transport pipelines).
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
#[cfg(unix)]
pub mod compat;
pub mod env;
mod path;
pub mod persist;
pub mod motion;

pub use header::{load64, store64, PassControl, PassTuning, ShmHeader};
pub use path::{isolated_path, shm_default_path, shm_runtime_dir};

/// Identifies a neural-forge mapping. Bumped only if the protocol is ever forked into an
/// incompatible variant; a mismatch here means "not our mapping at all", not "an older
/// version of our mapping" — that distinction is `SHM_VERSION`'s job.
pub const SHM_MAGIC: u32 = u32::from_le_bytes(*b"NFR1");

/// The wire contract version (v2 added BGRA8 and a motion payload region; v3 adds a
/// second independent request/response slot — see `docs/PROTOCOL_V3_DESIGN.md`).
/// The header layout version. A mismatch (matching magic, different version) means
/// another process in the chain is out of date; callers should log loudly and
/// reinitialize rather than half-read a header laid out differently than they expect.
pub const SHM_VERSION: u32 = 4;

pub const MAX_W: u32 = 7680;
pub const MAX_H: u32 = 4320;

/// Eight bytes a pixel: the float16 HDR proxy needs them, and the 8-bit path simply
/// uses the first half of each region. The mapping is file-backed and sparse, so an
/// SDR session never commits the second half.
pub const MAX_FRAME: usize = MAX_W as usize * MAX_H as usize * 8;

/// Whether a `(width, height, proxy_format)` triple read out of shared memory is one
/// the helper may size image resources from. The mapping is written by another
/// process, so none of it is trusted: dimensions must be non-zero, within
/// `MAX_W`/`MAX_H`, and even (the proxy's chroma-friendly grid), and the format one the
/// protocol defines.
pub fn frame_dims_valid(width: u32, height: u32, proxy_format: u32) -> bool {
    use enums::proxy_format::{BGRA8, RGBA16F, RGBA8};
    width != 0
        && height != 0
        && width <= MAX_W
        && height <= MAX_H
        && width % 2 == 0
        && height % 2 == 0
        && matches!(proxy_format, RGBA8 | RGBA16F | BGRA8)
}

pub const HEADER_BYTES: usize = 65536;

/// The ceiling on how many times the model runs over one frame, and what the slider
/// offers unless the ceiling is lifted.
pub const MAX_PASSES: usize = 30;
pub const DEFAULT_MAX_PASSES: u32 = 5;

pub const REASON_BYTES: usize = 192;
pub const NAME_BYTES: usize = 128;

/// Total size of the mapping: header, slot 0's proxy/answer, the motion region, and
/// slot 1's proxy/answer.
pub const fn shm_total_bytes() -> usize {
    HEADER_BYTES + MAX_FRAME * 5
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

/// Full-resolution R16G16_SFLOAT motion for the same seq_req as slot 0's proxy.
/// Shared, not duplicated per slot: motion vectors are disabled in this project's
/// current known-good baseline (see `ShmHeader::mvec_enabled`'s own doc comment), so
/// there is no live per-slot motion payload to race on today — see
/// `docs/PROTOCOL_V3_DESIGN.md` for the rest of that reasoning.
pub const fn motion_offset() -> usize { HEADER_BYTES + MAX_FRAME * 2 }

/// Byte offset of slot 1's proxy region — the second, independent in-flight request
/// v3 adds. See `docs/PROTOCOL_V3_DESIGN.md`.
pub const fn proxy_b_offset() -> usize { HEADER_BYTES + MAX_FRAME * 3 }

/// Byte offset of slot 1's answer region.
pub const fn answer_b_offset() -> usize { HEADER_BYTES + MAX_FRAME * 4 }

/// `proxy_offset()`/`proxy_b_offset()` picked by an actual slot index, the same
/// `slot: usize` the layer and helper already use for `ShmHeader::seq_req_slot` and
/// friends -- one place to keep in sync instead of a `match` at every call site.
pub const fn proxy_offset_slot(slot: usize) -> usize {
    if slot == 0 { proxy_offset() } else { proxy_b_offset() }
}

/// `answer_offset()`/`answer_b_offset()` picked by slot index. See
/// [`proxy_offset_slot`].
pub const fn answer_offset_slot(slot: usize) -> usize {
    if slot == 0 { answer_offset() } else { answer_b_offset() }
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
        assert_eq!(proxy_offset_slot(0), proxy_offset());
        assert_eq!(proxy_offset_slot(1), proxy_b_offset());
        assert_eq!(answer_offset_slot(0), answer_offset());
        assert_eq!(answer_offset_slot(1), answer_b_offset());
    }

    #[test]
    fn every_region_is_disjoint_and_fits_the_mapping() {
        let regions = [
            ("header", 0, HEADER_BYTES),
            ("proxy", proxy_offset(), MAX_FRAME),
            ("answer", answer_offset(), MAX_FRAME),
            ("motion", motion_offset(), MAX_FRAME),
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
