//! The composition math: how much of the model's answer reaches the frame, and how.
//!
//! This is the part upstream's own `ATTRIBUTION.md` says is GPL-3.0-tainted in its
//! tree (derived from OptiScaler-family forks) — nothing here is read from upstream's
//! `dlssnr.hlsl`. Rederived independently from:
//!
//! - **The algorithm shape**: upstream's own `third_party/optiscaler/RenoDX_ATTRIBUTION.txt`
//!   describes, in plain English, what RenoDX's (MIT-licensed, clshortfuse) DLSS 5
//!   addon actually does — not code, a description of the technique (four
//!   paragraphs: the two-branch luminance/headroom rule, the OkLab hue correction, the
//!   luminance-only/full blend, the reversible neutral-axis gamut compression). That
//!   description is what [`upgrade_tone_map`] and [`gamut_compress_reversible`]
//!   implement; the Rust below is this crate's own expression of it.
//! - **The OkLab matrices**: Björn Ottosson's own published constants
//!   (<https://bottosson.github.io/posts/oklab/>), cross-checked against RenoDX's own
//!   `src/shaders/color/oklab.hlsl` (MIT) to confirm they match what RenoDX itself
//!   uses — these are explicitly *not* RenoDX's own work per the same attribution
//!   file ("not specific to RenoDX").
//! - **Everything else** (sRGB linear/XYZ, the Hunt-Pointer-Estevez LMS basis, BT.709
//!   luma weights): standard, textbook color science, the same status the
//!   attribution file gives the AP1/sRGB/PQ transforms it names.
//!
//! Pure math, deliberately: this module has no GPU/Vulkan dependency at all, so its
//! correctness is checkable with plain `#[test]`s (round-trips, known reference
//! values) rather than needing a real device. The GLSL compute shader this feeds is a
//! separate, hand-translated expression of the exact same algorithm — kept in sync by
//! hand, since GLSL and Rust can't literally share source.

/// BT.709/sRGB relative luminance weights (standard, ITU-R BT.709).
pub fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]
}

fn cbrt3(v: [f32; 3]) -> [f32; 3] {
    [v[0].cbrt(), v[1].cbrt(), v[2].cbrt()]
}

fn cube3(v: [f32; 3]) -> [f32; 3] {
    [v[0] * v[0] * v[0], v[1] * v[1] * v[1], v[2] * v[2] * v[2]]
}

// Björn Ottosson's published constants (see module doc comment) -- also present,
// unmodified, in RenoDX's own MIT-licensed `oklab.hlsl`.
const LINEAR_SRGB_TO_OKLAB_LMS: [[f32; 3]; 3] =
    [[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]];
const OKLAB_LMS_TO_OKLAB: [[f32; 3]; 3] =
    [[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]];
const OKLAB_TO_OKLAB_LMS: [[f32; 3]; 3] = [[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728], [1.0, -0.0894841775, -1.2914855480]];
const OKLAB_LMS_TO_LINEAR_SRGB: [[f32; 3]; 3] =
    [[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965], [-0.0041960863, -0.7034186147, 1.7076147010]];

pub fn oklab_from_linear_srgb(rgb: [f32; 3]) -> [f32; 3] {
    let lms = mat3_mul(&LINEAR_SRGB_TO_OKLAB_LMS, rgb);
    mat3_mul(&OKLAB_LMS_TO_OKLAB, cbrt3(lms))
}

pub fn linear_srgb_from_oklab(oklab: [f32; 3]) -> [f32; 3] {
    let lms = mat3_mul(&OKLAB_TO_OKLAB_LMS, oklab);
    mat3_mul(&OKLAB_LMS_TO_LINEAR_SRGB, cube3(lms))
}

// Standard IEC 61966-2-1 (sRGB) linear-RGB <-> CIE XYZ (D65), and the standard
// Hunt-Pointer-Estevez XYZ -> LMS basis (von Kries-style chromatic adaptation; see
// e.g. Fairchild, *Color Appearance Models*) -- none of this is RenoDX's; it's the
// same "specific to nobody" category upstream's own attribution file puts the AP1/PQ
// transforms in.
const LINEAR_SRGB_TO_XYZ: [[f32; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
const XYZ_TO_HPE_LMS: [[f32; 3]; 3] = [[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.0, 0.0, 1.0]];
const HPE_LMS_TO_XYZ: [[f32; 3]; 3] = [[1.9102329169, -1.1121299780, 0.2019182498], [0.3709500882, 0.6290542715, 0.0000000000], [0.0, 0.0, 1.0]];
/// The D65 white point's XYZ, normalized so `Y = 1` — used to build "a neutral color
/// of the same luminance as X" by scaling this along the white point's own ray.
const D65_WHITE_XYZ: [f32; 3] = [0.95047, 1.0, 1.08883];

fn linear_srgb_to_hpe_lms(rgb: [f32; 3]) -> [f32; 3] {
    mat3_mul(&XYZ_TO_HPE_LMS, mat3_mul(&LINEAR_SRGB_TO_XYZ, rgb))
}

fn hpe_lms_to_linear_srgb(lms: [f32; 3]) -> [f32; 3] {
    // XYZ -> linear sRGB is the matrix inverse of LINEAR_SRGB_TO_XYZ.
    const XYZ_TO_LINEAR_SRGB: [[f32; 3]; 3] = [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]];
    mat3_mul(&XYZ_TO_LINEAR_SRGB, mat3_mul(&HPE_LMS_TO_XYZ, lms))
}

/// The two-branch luminance/headroom rule plus the OkLab hue correction plus the
/// luminance-only/full blend — see the module doc comment for where this comes from.
///
/// - `original`: the frame before any model edit.
/// - `proxy`: what the model was actually shown (the tone-mapped/display-referred
///   encode of `original` — may have compressed away highlights `original` had).
/// - `model`: the model's raw answer, at the proxy's own exposure/encoding.
/// - `colour_strength`: 0 = luminance-only result (`original`'s hue, the model's
///   light); 1 = the fully hue-corrected result. Between: a blend of the two.
///
/// Returns linear-light RGB, **not yet gamut-compressed** — see
/// [`gamut_compress_reversible`], applied separately so a caller that wants the model's
/// raw transfer without the compression can skip it.
pub fn upgrade_tone_map(original: [f32; 3], proxy: [f32; 3], model: [f32; 3], colour_strength: f32) -> [f32; 3] {
    let original_luminance = luminance(original);
    let proxy_luminance = luminance(proxy).max(1e-6);
    let model_luminance = luminance(model).max(1e-6);

    // Below the proxy's own luminance, the original is already representable and is
    // the target outright. Above it, the proxy clipped highlights `original` actually
    // had; treat that clipped-away difference as headroom and hand it back on top of
    // whatever the model answered, rather than trying to make the model responsible
    // for light it was never shown.
    let target_luminance = if original_luminance <= proxy_luminance { original_luminance } else { model_luminance + (original_luminance - proxy_luminance) };

    let ratio = target_luminance / model_luminance;
    let rescaled = [model[0] * ratio, model[1] * ratio, model[2] * ratio];

    // The luminance-only result: the original's own hue and chroma, just relit to the
    // target luminance -- i.e. scale `original` itself by the same target/model
    // relationship, but taken relative to its own luminance instead of the model's.
    let original_ratio = target_luminance / original_luminance.max(1e-6);
    let luminance_only = [original[0] * original_ratio, original[1] * original_ratio, original[2] * original_ratio];

    if colour_strength <= 0.0 {
        return luminance_only;
    }

    // The fully upgraded colour: `rescaled`'s lightness (already at the target
    // luminance), the model's own hue and chroma *direction*, but only `rescaled`'s
    // chroma *magnitude* -- the model answers what direction the colour should shift,
    // the luminance-preserving rescale answers how far.
    let model_oklab = oklab_from_linear_srgb(model);
    let model_chroma = (model_oklab[1] * model_oklab[1] + model_oklab[2] * model_oklab[2]).sqrt();
    let (chroma_dir_a, chroma_dir_b) = if model_chroma > 1e-6 { (model_oklab[1] / model_chroma, model_oklab[2] / model_chroma) } else { (0.0, 0.0) };

    let rescaled_oklab = oklab_from_linear_srgb(rescaled);
    let rescaled_chroma = (rescaled_oklab[1] * rescaled_oklab[1] + rescaled_oklab[2] * rescaled_oklab[2]).sqrt();

    let corrected_oklab = [rescaled_oklab[0], chroma_dir_a * rescaled_chroma, chroma_dir_b * rescaled_chroma];
    let fully_upgraded = linear_srgb_from_oklab(corrected_oklab);

    let t = colour_strength.clamp(0.0, 1.0);
    [
        luminance_only[0] + (fully_upgraded[0] - luminance_only[0]) * t,
        luminance_only[1] + (fully_upgraded[1] - luminance_only[1]) * t,
        luminance_only[2] + (fully_upgraded[2] - luminance_only[2]) * t,
    ]
}

/// The reversible neutral-axis gamut compression: scale the whole colour by one factor
/// toward the neutral (gray) colour of the same luminance, just far enough that no
/// channel is negative — the identity when nothing is out of gamut, and (by
/// construction, since it's one scale toward one well-defined point) invertible by
/// scaling back out by the reciprocal, unlike clamping each channel independently
/// (which bends hue and cannot be undone once negative excursions are thrown away).
pub fn gamut_compress_reversible(rgb: [f32; 3]) -> [f32; 3] {
    let lms = linear_srgb_to_hpe_lms(rgb);
    let y = luminance(rgb);
    let neutral_xyz = [D65_WHITE_XYZ[0] * y, D65_WHITE_XYZ[1] * y, D65_WHITE_XYZ[2] * y];
    let neutral_lms = mat3_mul(&XYZ_TO_HPE_LMS, neutral_xyz);

    let mut t_max = 1.0f32;
    for c in 0..3 {
        if lms[c] < 0.0 {
            let delta = lms[c] - neutral_lms[c];
            if delta < 0.0 {
                let t_candidate = neutral_lms[c] / (neutral_lms[c] - lms[c]);
                t_max = t_max.min(t_candidate.max(0.0));
            }
        }
    }

    if t_max >= 1.0 {
        return rgb; // already in gamut -- exact identity, not just "close enough".
    }
    let compressed_lms = [
        neutral_lms[0] + (lms[0] - neutral_lms[0]) * t_max,
        neutral_lms[1] + (lms[1] - neutral_lms[1]) * t_max,
        neutral_lms[2] + (lms[2] - neutral_lms[2]) * t_max,
    ];
    hpe_lms_to_linear_srgb(compressed_lms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < eps)
    }

    #[test]
    fn hpe_lms_round_trips() {
        // `HPE_LMS_TO_XYZ` is a hand-derived matrix inverse of `XYZ_TO_HPE_LMS` --
        // exactly the kind of arithmetic worth checking by round-trip rather than
        // trusting the derivation. Epsilon looser than the OkLab round-trip test's:
        // `XYZ_TO_HPE_LMS`'s published constants are given to only 5 decimal places,
        // so an exact-inverse computation from them still carries ~1e-4 residual
        // error -- confirmed by every one of these colors landing consistently in the
        // 1e-4-to-1e-5 range with no outliers, which is what rounding error from the
        // input constants' own precision looks like, not a derivation mistake.
        for rgb in [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0; 3]] {
            let lms = linear_srgb_to_hpe_lms(rgb);
            let back = hpe_lms_to_linear_srgb(lms);
            assert!(approx_eq(rgb, back, 5e-4), "HPE LMS round trip failed for {rgb:?}: got {back:?} via {lms:?}");
        }
    }

    #[test]
    fn oklab_round_trips() {
        for rgb in [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]] {
            let oklab = oklab_from_linear_srgb(rgb);
            let back = linear_srgb_from_oklab(oklab);
            assert!(approx_eq(rgb, back, 1e-4), "round trip failed for {rgb:?}: got {back:?} via {oklab:?}");
        }
    }

    #[test]
    fn oklab_of_mid_gray_is_achromatic() {
        // A gray (equal R=G=B) must land with ~zero chroma (a, b) regardless of L.
        let oklab = oklab_from_linear_srgb([0.5, 0.5, 0.5]);
        assert!(oklab[1].abs() < 1e-4, "expected ~0 a for gray, got {}", oklab[1]);
        assert!(oklab[2].abs() < 1e-4, "expected ~0 b for gray, got {}", oklab[2]);
        assert!(oklab[0] > 0.0 && oklab[0] < 1.0);
    }

    #[test]
    fn luminance_of_white_is_one_and_black_is_zero() {
        assert!((luminance([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!(luminance([0.0, 0.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn upgrade_tone_map_below_headroom_targets_original_luminance() {
        // proxy_luminance is generous (2.0) so original (Y ~0.591 for [0.8,0.5,0.3])
        // never exceeds it -- exercises the "no headroom" branch.
        let original = [0.8, 0.5, 0.3];
        let proxy = [1.6, 1.0, 0.6]; // scaled up, luminance ~1.18
        let model = [0.4, 0.4, 0.4]; // model's raw answer, arbitrary
        let result = upgrade_tone_map(original, proxy, model, 0.0);
        // colour_strength = 0 => luminance-only: original's hue, target luminance.
        assert!((luminance(result) - luminance(original)).abs() < 1e-4);
    }

    #[test]
    fn upgrade_tone_map_above_headroom_adds_the_clipped_difference() {
        // original's luminance exceeds the proxy's -- the proxy clipped highlights.
        let original = [2.0, 2.0, 2.0]; // luminance 2.0
        let proxy = [1.0, 1.0, 1.0]; // luminance 1.0 (clipped by the encode)
        let model = [0.5, 0.5, 0.5]; // luminance 0.5
        let result = upgrade_tone_map(original, proxy, model, 0.0);
        // target = model_luminance + (original_luminance - proxy_luminance) = 0.5 + 1.0 = 1.5
        assert!((luminance(result) - 1.5).abs() < 1e-4, "got luminance {}", luminance(result));
    }

    #[test]
    fn upgrade_tone_map_colour_strength_zero_matches_original_hue() {
        let original = [0.9, 0.2, 0.2]; // strongly red
        let proxy = [0.9, 0.2, 0.2];
        let model = [0.2, 0.9, 0.2]; // strongly green -- very different hue
        let result = upgrade_tone_map(original, proxy, model, 0.0);
        let result_oklab = oklab_from_linear_srgb(result);
        let original_oklab = oklab_from_linear_srgb(original);
        let result_hue = result_oklab[2].atan2(result_oklab[1]);
        let original_hue = original_oklab[2].atan2(original_oklab[1]);
        assert!((result_hue - original_hue).abs() < 1e-3, "expected original's hue preserved at colour_strength=0");
    }

    #[test]
    fn gamut_compress_is_identity_when_already_in_gamut() {
        for rgb in [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.2, 0.4, 0.6], [0.0, 0.0, 0.0]] {
            let out = gamut_compress_reversible(rgb);
            assert!(approx_eq(rgb, out, 1e-4), "expected identity for in-gamut {rgb:?}, got {out:?}");
        }
    }

    #[test]
    fn gamut_compress_removes_negative_lms_channels() {
        // A strongly saturated, supersaturated color (components outside [0,1], as the
        // OkLab hue-correction step above can produce) can map to a negative LMS
        // channel; compression must remove it while preserving luminance exactly
        // (only relative RGB shifts toward gray, never toward brighter/darker).
        let out_of_gamut = [1.4, -0.3, -0.2];
        let compressed = gamut_compress_reversible(out_of_gamut);
        let lms = linear_srgb_to_hpe_lms(compressed);
        assert!(lms[0] >= -1e-4 && lms[1] >= -1e-4 && lms[2] >= -1e-4, "expected non-negative LMS, got {lms:?}");
        assert!(
            (luminance(compressed) - luminance(out_of_gamut)).abs() < 1e-3,
            "gamut compression must preserve luminance: {} vs {}",
            luminance(compressed),
            luminance(out_of_gamut)
        );
    }
}
