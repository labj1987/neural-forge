//! The record format the wrapper writes and the Neural Forge layer reads (`crates/layer/src/ngx_ring.rs` parses the
//! same bytes; both sides test the offsets). One file per process, `/dev/shm/neural-forge-ngx-<pid>.ring` (the
//! Linux pid, which the layer gets from `getpid()` and the wrapper from Wine's `ProcessWineUnixPid`), mapped by both.
//!
//! Header (32 bytes): magic `NFNGXRG1`, version, record size, record count, pid, then the number of records
//! reserved so far (u64). Record `n` (0-based) lives in slot `n % count` after the header. A writer reserves `n`
//! with an atomic add, sets the slot's `seq` to 0, writes the fields, then sets `seq` to `n + 1` (release); a
//! reader takes a slot only while its `seq` reads `n + 1` before and after copying it.

/// The file's first 8 bytes.
pub const MAGIC: [u8; 8] = *b"NFNGXRG1";
/// The format's version ([`Header::version`]).
pub const VERSION: u32 = 1;
/// Records per file.
pub const SLOTS: u32 = 1024;
/// Bytes before the first record.
pub const HEADER_BYTES: usize = 32;
/// Bytes per record.
pub const RECORD_BYTES: usize = 128;
/// The whole file.
pub const FILE_BYTES: usize = HEADER_BYTES + SLOTS as usize * RECORD_BYTES;

/// [`Record::api`]
pub const API_VULKAN: u32 = 1;
pub const API_D3D11: u32 = 11;
pub const API_D3D12: u32 = 12;

/// The file header.
#[repr(C)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub record_bytes: u32,
    pub slots: u32,
    pub pid: u32,
    /// Records reserved so far.
    pub next: core::sync::atomic::AtomicU64,
}

/// One NGX call: an evaluate (`create_flags` from the feature's creation), or a creation (`seq` as for an evaluate,
/// `feature` set, images 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Record {
    /// `n + 1` for record `n` once written, 0 while being written.
    pub seq: u64,
    /// Milliseconds since the wrapper loaded.
    pub t_ms: f64,
    /// [`API_VULKAN`], [`API_D3D11`] or [`API_D3D12`].
    pub api: u32,
    /// NGX's handle id (not unique across features: Super Resolution and Frame Generation may both be 1).
    pub handle: u32,
    /// NGX's feature: 1 Super Resolution, 11 Frame Generation, 13 Ray Reconstruction (0 unknown).
    pub feature: u32,
    /// `DLSS.Feature.Create.Flags` from the creation: 1 IsHDR, 2 MVLowRes, 4 MVJittered, 8 DepthInverted,
    /// 64 AutoExposure.
    pub create_flags: u32,
    /// The VkImage handles of `Color`, `MotionVectors`, `Depth`, `Output` and `ExposureTexture` (0 if not given).
    pub colour: u64,
    pub mvec: u64,
    pub depth: u64,
    pub output: u64,
    pub exposure: u64,
    /// `Jitter.Offset.X/Y`, render pixels.
    pub jitter: [f32; 2],
    /// `MV.Scale.X/Y`: what turns the motion vectors into render pixels.
    pub mv_scale: [f32; 2],
    /// `DLSS.Pre.Exposure` (0 if not given) and `DLSS.Exposure.Scale` (0 if not given).
    pub pre_exposure: f32,
    pub exposure_scale: f32,
    /// `Reset`.
    pub reset: i32,
    /// `DLSS.Render.Subrect.Dimensions` (0 if not given), the colour input's and the output's extents.
    pub render: [u32; 2],
    pub colour_extent: [u32; 2],
    pub output_extent: [u32; 2],
    /// The calling thread (Windows thread id).
    pub thread: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    /// The offsets `crates/layer/src/ngx_ring.rs` reads.
    #[test]
    fn layout() {
        assert_eq!(size_of::<Header>(), HEADER_BYTES);
        assert_eq!(offset_of!(Header, next), 24);
        assert_eq!(size_of::<Record>(), RECORD_BYTES);
        let at = [
            (offset_of!(Record, seq), 0),
            (offset_of!(Record, t_ms), 8),
            (offset_of!(Record, api), 16),
            (offset_of!(Record, handle), 20),
            (offset_of!(Record, feature), 24),
            (offset_of!(Record, create_flags), 28),
            (offset_of!(Record, colour), 32),
            (offset_of!(Record, mvec), 40),
            (offset_of!(Record, depth), 48),
            (offset_of!(Record, output), 56),
            (offset_of!(Record, exposure), 64),
            (offset_of!(Record, jitter), 72),
            (offset_of!(Record, mv_scale), 80),
            (offset_of!(Record, pre_exposure), 88),
            (offset_of!(Record, exposure_scale), 92),
            (offset_of!(Record, reset), 96),
            (offset_of!(Record, render), 100),
            (offset_of!(Record, colour_extent), 108),
            (offset_of!(Record, output_extent), 116),
            (offset_of!(Record, thread), 124),
        ];
        for (got, want) in at {
            assert_eq!(got, want);
        }
    }
}
