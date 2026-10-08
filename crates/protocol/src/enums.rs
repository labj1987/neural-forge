//! The header's `u32`-valued fields group into small closed sets of meanings. These
//! are deliberately plain `u32` constants, not `#[repr(u32)] enum`s: the values live in
//! shared memory written by another process (possibly a stale or foreign one), and
//! reinterpreting an arbitrary `u32` as a Rust enum via a transmute is immediate
//! undefined behavior the moment the bytes don't match a declared variant. Read the
//! raw `u32` and clamp/validate it at the call site instead.

/// Where the white point comes from: the slider, or the calibration grid measured off
/// the untouched copy of the frame.
pub mod white_point_source {
    pub const MANUAL: u32 = 0;
    pub const MEASURED: u32 = 1;
}

/// What the swapchain holds, and therefore what the encode has to do to it before it
/// reaches the model. Getting this wrong encodes an encoded frame a second time, which
/// looks washed out and banded — so the default decides it from the format rather than
/// guessing.
pub mod colour_mode {
    /// 8-bit: display-referred. 10-bit and float: linear HDR.
    pub const AUTO: u32 = 0;
    /// Force display-referred; the encode becomes a pass-through.
    pub const DISPLAY: u32 = 1;
    /// Force linear HDR; the encode scales by the white point and sRGB-encodes.
    pub const LINEAR_HDR: u32 = 2;
}

/// Which proxy the model is shown, and how its answer is brought back — the reversible
/// composition modes. Modes 2 and 4 are known to flash on bright lights, which is why 0
/// is the default rather than a taste.
pub mod reversible_mode {
    /// Soft knee + composition — byte-identical to the original shipped behavior.
    pub const KNEE: u32 = 0;
    /// Unclipped Neutwo proxy + composition.
    pub const NEUTWO: u32 = 1;
    /// Neutwo proxy + pure-inverse replace — the model's answer straight back, no composition.
    pub const NEUTWO_REPLACE: u32 = 2;
    /// Hybrid proxy + composition — identity midtones, unclipped highlights.
    pub const HYBRID: u32 = 3;
    /// Hybrid proxy + replace.
    pub const HYBRID_REPLACE: u32 = 4;
    pub const COUNT: u32 = 5;
}

/// What the model server (`native_post`, the after-the-upscaler path's) is doing: the layer's post
/// path asks it for answers only while it runs. "Stopped" is what a freshly initialised header holds.
pub mod server_state {
    pub const MODEL_FAILED: u32 = 3;
    pub const RUNNING: u32 = 4;
    pub const STOPPED: u32 = 5;
}

/// The HDR input path. `hdr_mode` is the user's choice; `hdr_detected`/`hdr_kind` are
/// the layer's reading of the swapchain; `hdr_active` is what the layer actually
/// encoded the frame as; `proxy_format` is what the proxy actually is. The float path is only taken while both agree it exists.
pub mod hdr_mode {
    /// HDR when the swapchain is HDR.
    pub const AUTO: u32 = 0;
    /// Always the SDR proxy, whatever the swapchain.
    pub const OFF: u32 = 1;
    /// Float16 proxy even for an SDR swapchain — an A/B tool, not a preference.
    pub const FORCE: u32 = 2;
}

pub mod hdr_kind {
    /// 8-bit swapchain: already tone mapped.
    pub const NONE: u32 = 0;
    /// `R16G16B16A16_SFLOAT`: linear light, open range.
    pub const LINEAR_FP16: u32 = 1;
    /// 10-bit with a PQ/BT.2020 colour space: ST 2084 code.
    pub const PQ10: u32 = 2;
}

/// What the crossing images actually are, published by the model server: the model decides,
/// and the layer encodes to match rather than to hope.
pub mod proxy_format {
    pub const UNKNOWN: u32 = 0;
    pub const RGBA8: u32 = 1;
    pub const RGBA16F: u32 = 2;
    pub const BGRA8: u32 = 3;

    pub fn is_8bit(format: u32) -> bool {
        matches!(format, RGBA8 | BGRA8)
    }

    /// Bytes per pixel for a raw dump in this format -- shared by the layer (which
    /// writes the proxy region at this size) and the model server (which needs to know how
    /// many of the region's bytes are real for a given frame, not the full
    /// `MAX_FRAME`-sized reservation). Unknown formats are treated as the smaller,
    /// 8-bit encoding: a format this crate doesn't recognize should never have been
    /// written in the first place, and under-reading is safer than over-reading past
    /// what was actually captured.
    pub fn bytes_per_pixel(format: u32) -> usize {
        if format == RGBA16F {
            8
        } else {
            4
        }
    }
}

