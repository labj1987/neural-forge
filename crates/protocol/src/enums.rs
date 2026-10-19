//! The header's `u32`-valued fields group into small closed sets of meanings. These
//! are deliberately plain `u32` constants, not `#[repr(u32)] enum`s: the values live in
//! shared memory written by another process (possibly a stale or foreign one), and
//! reinterpreting an arbitrary `u32` as a Rust enum via a transmute is immediate
//! undefined behavior the moment the bytes don't match a declared variant. Read the
//! raw `u32` and clamp/validate it at the call site instead.

/// Where the model runs relative to DLSS's upscaler (`ShmHeader::preupscale_state`), as the layer
/// publishes it.
pub mod preupscale_state {
    /// After the upscaler (no DLSS, no NVX, or switched off).
    pub const OFF: u32 = 0;
    /// Before-the-upscaler tracking is on, but no DLSS input has been held lately.
    pub const WAITING: u32 = 1;
    /// Holding DLSS's input: the model runs before the upscaler.
    pub const HOLDING: u32 = 2;
}

/// Which of the networks `extract-model` writes the layer runs (`ShmHeader::network`, the Network
/// row of the Setup tab). Each has its own directory under the model directory
/// (`$XDG_DATA_HOME/neural-forge/model/<name>`).
pub mod network {
    /// `WEIGHTS_HT`, the network every build of the DLL carries: the default.
    pub const FULL: u32 = 0;
    /// `WEIGHTS_CC_SMALL_FP8` (build 310.9.16 on): faster, with a subtler effect.
    pub const SMALL: u32 = 1;
    pub const COUNT: u32 = 2;

    /// The network's directory name under the model directory, and its name on the command line.
    /// A value out of range is the full network.
    pub fn name(network: u32) -> &'static str {
        if network == SMALL { "small" } else { "full" }
    }
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


/// The bits of a pass's `overrides` setting (`pass<N>_overrides`): each names a Model-tab value the
/// pass takes from its own field instead. A pass with none follows the Model tab.
pub mod pass_override {
    pub const INTENSITY: u32 = 1;
    pub const LOCAL_TONE: u32 = 2;
    pub const LOCAL_STRUCTURE: u32 = 4;
    pub const SKIN_STRUCTURE: u32 = 8;
    pub const STYLE: u32 = 16;
    pub const AUTO_MASK: u32 = 32;
    /// The pass's own model resolution before the upscaler (`pass<N>_scale`) instead of
    /// `preupscale_scale`.
    pub const SCALE: u32 = 64;
    /// The colour of the edit (`ShmHeader::pass_colour`).
    pub const EDIT_SATURATION: u32 = 128;
    pub const EDIT_CONTRAST: u32 = 256;
    pub const EDIT_DETAIL: u32 = 512;
    pub const ALL: u32 = 1023;
}

/// The model resolution presets before the upscaler (`preupscale_scale`, `pass<N>_scale`): the network's
/// scale of DLSS's input size, by name. Full runs it at that size (the default); the others are DLSS's own
/// ratios, the network on a downscaled picture whose edit is brought back to full size.
pub mod model_scale {
    pub const PRESETS: [(&str, f32); 5] = [("Full", 1.0), ("Quality", 0.667), ("Balanced", 0.58), ("Performance", 0.5), ("Ultra Performance", 0.333)];
    /// The lowest scale a setting may hold.
    pub const MIN: f32 = 0.25;

    /// The preset `scale` is (within 0.001), as an index into [`PRESETS`].
    pub fn preset_of(scale: f32) -> Option<usize> {
        PRESETS.iter().position(|&(_, s)| (s - scale).abs() < 0.001)
    }
}
