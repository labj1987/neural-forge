//! The header's `u32`-valued fields group into small closed sets of meanings. These
//! are deliberately plain `u32` constants, not `#[repr(u32)] enum`s: the values live in
//! shared memory written by another process (possibly a stale or foreign one), and
//! reinterpreting an arbitrary `u32` as a Rust enum via a transmute is immediate
//! undefined behavior the moment the bytes don't match a declared variant. Read the
//! raw `u32` and clamp/validate it at the call site instead (see e.g.
//! [`ShmHeader::mvec_scale_mode`](crate::ShmHeader::mvec_scale_mode)).

/// Which fields a per-pass entry actually overrides. Sparse by design: a pass with no
/// entry, or an entry that does not name a field, follows the global setting — which is
/// what makes "pass 3 is gentler" expressible without restating everything else about
/// pass 3.
pub mod pass_override {
    pub const INTENSITY: u32 = 1 << 0;
    pub const LOCAL_STRUCTURE: u32 = 1 << 1;
    pub const LOCAL_TONE: u32 = 1 << 2;
    pub const SKIN_STRUCTURE: u32 = 1 << 3;
    pub const STYLE: u32 = 1 << 4;
    pub const PRESET: u32 = 1 << 5;
    pub const AUTO_MASK: u32 = 1 << 6;
    pub const SHARPNESS: u32 = 1 << 7;
}

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

/// The filter that brings the model's answer back down when it ran above native
/// resolution. Only consulted when the working scale is above 1.0.
pub mod downscaler {
    /// Reserved; not a downscaler this pipeline can run (it wants a different constant
    /// block and is an upscaler, not the averaging filter the down-leg needs). A header
    /// asking for it falls back to `LANCZOS3`.
    pub const FSR1: u32 = 0;
    pub const BICUBIC: u32 = 1;
    pub const CATMULL_ROM: u32 = 2;
    pub const LANCZOS2: u32 = 3;
    /// The default: the sharp one.
    pub const LANCZOS3: u32 = 4;
    pub const KAISER2: u32 = 5;
    pub const KAISER3: u32 = 6;
    pub const MAGIC: u32 = 7;
    pub const COUNT: u32 = 8;
}

/// What the helper has managed to do, for the GUI and for the layer's fail-open
/// decision. The layer reads this to decide whether anything is listening, so
/// "nobody" has to be the value a freshly initialized header holds — not a state that
/// also means "starting".
pub mod helper_state {
    pub const STARTING: u32 = 0;
    /// No NVIDIA device with the required extensions.
    pub const NO_VULKAN: u32 = 1;
    /// The NGX model DLL was not found.
    pub const NO_BINARIES: u32 = 2;
    pub const MODEL_FAILED: u32 = 3;
    pub const RUNNING: u32 = 4;
    pub const STOPPED: u32 = 5;
}

/// The HDR input path. `hdr_mode` is the user's choice; `hdr_detected`/`hdr_kind` are
/// the layer's reading of the swapchain; `hdr_active` is what the layer actually
/// encoded the frame as; `proxy_format` is what the helper actually built the crossing
/// images as. The float path is only taken while both agree it exists.
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

/// What the crossing images actually are, published by the helper: the model decides,
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
    /// writes the proxy region at this size) and the helper (which needs to know how
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

pub mod mvec_scale_mode {
    pub const NORMALIZED: u32 = 0;
    pub const PIXELS: u32 = 1;
    pub const UV01: u32 = 2;
}

/// What the optical-flow engine is asked for. Higher costs more of the frame's budget.
pub mod mvec_quality {
    pub const FAST: u32 = 0;
    pub const BALANCED: u32 = 1;
    pub const QUALITY: u32 = 2;
}
