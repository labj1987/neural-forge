//! The scene-linear RGBA16F proxy (the pre-upscaler path, `docs/PRE_UPSCALER_DESIGN.md`
//! section 3): what the helper needs to know about a frame's format class, and the small
//! amount of CPU-side float handling it does on such a frame.
//!
//! The model itself always gets the raw half floats. Only two consumers want an 8-bit
//! picture: optical flow (whose input format is `B8G8R8A8_UNORM`; the GPU twin of
//! [`tonemap_encode`] lives in `shaders/hdr_to_flow.comp`) and the scene-cut thumbnail
//! (`scene.rs`, CPU, which decodes halves with [`f16_to_f32`]).
//!
//! Pure bookkeeping and arithmetic with no Win32 or Vulkan calls, so it builds and tests
//! natively.

use neural_forge_protocol::enums::proxy_format;

/// Which kind of proxy a slot carries. Everything that is built for one class (the NGX
/// feature, the frame resources, the flow session, the model's history) is rebuilt or reset
/// when the class changes, exactly as for a size change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatClass {
    /// `RGBA8`/`BGRA8`: the display-referred frame, told to the model as SDR.
    Sdr8,
    /// `RGBA16F`: scene-linear HDR half floats, values may far exceed 1.0.
    Hdr16,
}

impl FormatClass {
    /// `None` for a format the helper cannot evaluate (unknown values from shared memory).
    pub fn of(format: u32) -> Option<Self> {
        match format {
            proxy_format::RGBA8 | proxy_format::BGRA8 => Some(Self::Sdr8),
            proxy_format::RGBA16F => Some(Self::Hdr16),
            _ => None,
        }
    }

    pub fn is_hdr(self) -> bool {
        self == Self::Hdr16
    }
}

/// Whether the helper can evaluate a proxy in this format at all.
pub fn supported(format: u32) -> bool {
    FormatClass::of(format).is_some()
}

/// What one NGX feature was built for. The model latches its size and its HDR/SDR mode at
/// creation, so a frame with a different key needs every feature rebuilt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FeatureKey {
    pub width: u32,
    pub height: u32,
    /// Created with `DLSSNR.Hdr=1, SDR=0` (an RGBA16F proxy) rather than `Hdr=0, SDR=1`.
    pub hdr: bool,
}

impl FeatureKey {
    pub fn new(width: u32, height: u32, hdr: bool) -> Self {
        Self { width, height, hdr }
    }
}

impl std::fmt::Display for FeatureKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{} hdr={}", self.width, self.height, u8::from(self.hdr))
    }
}

/// IEEE 754 binary16 (little-endian `u16` bits) to `f32`, exact for every value including
/// subnormals, infinities and NaN.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let out = match (exp, mant) {
        (0, 0) => sign,
        // Subnormal: mant * 2^-24, exactly representable in f32.
        (0, m) => {
            let v = m as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(out)
}

/// The 8-bit picture optical flow and the scene-cut thumbnail see for one scene-linear
/// channel value: Reinhard `x / (1 + x)` (so any brightness lands in [0, 1) and highlights
/// keep their texture), then the sRGB transfer function (so the 8 bits are spent where the
/// eye and the flow's block matcher need them). NaN and negatives map to 0, +Inf to 1.
///
/// `shaders/hdr_to_flow.comp`'s `encode` is the same function; keep them in step.
pub fn tonemap_encode(x: f32) -> f32 {
    if x.is_nan() || x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    let c = x / (1.0 + x);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// [`tonemap_encode`] quantised to a byte, rounding to nearest.
pub fn tonemap_u8(x: f32) -> u8 {
    (tonemap_encode(x) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// One RGBA16F pixel's three colour channels at byte offset `i` of `frame` (8 bytes per
/// pixel, little-endian halves), as `f32`. The caller checks `i + 6 <= frame.len()`.
pub fn rgb16f_at(frame: &[u8], i: usize) -> [f32; 3] {
    let half = |k: usize| f16_to_f32(u16::from_le_bytes([frame[i + 2 * k], frame[i + 2 * k + 1]]));
    [half(0), half(1), half(2)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_classes() {
        assert_eq!(FormatClass::of(proxy_format::RGBA8), Some(FormatClass::Sdr8));
        assert_eq!(FormatClass::of(proxy_format::BGRA8), Some(FormatClass::Sdr8));
        assert_eq!(FormatClass::of(proxy_format::RGBA16F), Some(FormatClass::Hdr16));
        assert_eq!(FormatClass::of(proxy_format::UNKNOWN), None);
        assert_eq!(FormatClass::of(99), None);
        assert!(FormatClass::Hdr16.is_hdr() && !FormatClass::Sdr8.is_hdr());
        assert!(supported(proxy_format::RGBA16F) && supported(proxy_format::RGBA8) && !supported(0));
        // The helper's bookkeeping and the protocol's byte count agree on 8 bytes per 16F pixel.
        assert_eq!(proxy_format::bytes_per_pixel(proxy_format::RGBA16F), 8);
    }

    #[test]
    fn feature_key_includes_hdr() {
        let sdr = FeatureKey::new(1708, 960, false);
        let hdr = FeatureKey::new(1708, 960, true);
        assert_ne!(sdr, hdr, "a format-class change must read as a different feature");
        assert_ne!(hdr, FeatureKey::new(1490, 838, true));
        assert_eq!(hdr, FeatureKey::new(1708, 960, true));
        assert_eq!(hdr.to_string(), "1708x960 hdr=1");
        assert_eq!(sdr.to_string(), "1708x960 hdr=0");
    }

    #[test]
    fn half_decode_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000) == 0.0 && f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x5640), 100.0);
        // Smallest subnormal and largest subnormal.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        // Smallest normal.
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0x7c01).is_nan());
    }

    #[test]
    fn half_decode_is_monotonic_over_all_positive_finite_values() {
        let mut prev = -1.0f32;
        for bits in 0u16..0x7c00 {
            let v = f16_to_f32(bits);
            assert!(v.is_finite() && v > prev, "bits {bits:#06x} -> {v} after {prev}");
            prev = v;
        }
    }

    #[test]
    fn tonemap_maps_open_range_into_unit_interval() {
        assert_eq!(tonemap_encode(0.0), 0.0);
        assert_eq!(tonemap_encode(-3.0), 0.0);
        assert_eq!(tonemap_encode(f32::NAN), 0.0);
        assert_eq!(tonemap_encode(f32::INFINITY), 1.0);
        // x = 1 -> 0.5 linear -> sRGB 0.7354.
        assert!((tonemap_encode(1.0) - 0.735_356_7).abs() < 1e-5);
        // The linear toe of the sRGB curve.
        assert!((tonemap_encode(0.001) - 12.92 * (0.001 / 1.001)).abs() < 1e-7);
        let mut prev = 0.0;
        for x in [0.0005f32, 0.01, 0.1, 0.5, 1.0, 4.0, 16.0, 100.0, 1000.0, 65504.0] {
            let y = tonemap_encode(x);
            assert!(y > prev && y < 1.0, "{x} -> {y}");
            prev = y;
        }
        assert_eq!(tonemap_u8(0.0), 0);
        assert_eq!(tonemap_u8(1.0), 188);
        assert_eq!(tonemap_u8(65504.0), 255);
        assert_eq!(tonemap_u8(f32::NAN), 0);
    }

    #[test]
    fn rgb_reads_little_endian_halves() {
        // R = 1.0, G = 2.0, B = 0.5, A = 1.0
        let px = [0x00, 0x3c, 0x00, 0x40, 0x00, 0x38, 0x00, 0x3c];
        assert_eq!(rgb16f_at(&px, 0), [1.0, 2.0, 0.5]);
    }
}
