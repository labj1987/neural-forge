//! The proxy encode: what the model is actually shown.
//!
//! **Provenance, deliberately separate from [`super::color`]:** that module's own
//! header records that it is clean-room — "nothing here is read from upstream's
//! `dlssnr.hlsl`" — because upstream's `ATTRIBUTION.md` marks that shader as
//! GPL-3.0-tainted in their tree (derived from OptiScaler-family forks). This module
//! is **not** clean-room. It is a direct port of the encode curves in DLSS5VKLayer's
//! `layer_linux/src/dlssnr/dlssnr.hlsl`, which this project is licensed to take after
//! relicensing to AGPL-3.0-or-later (AGPL-3.0 and GPL-3.0 are mutually compatible for
//! combination; see the root `ATTRIBUTION.md` for the full record). Keeping it in its
//! own file is what stops [`super::color`]'s clean-room claim from becoming false.
//!
//! Upstream describes leg 1 of the pass as
//! `swapchain -> frame -> ENCODE -> proxy (+ the untouched keep)`. Until this module
//! existed this project had no encode at all: the proxy handed to the model was a
//! bit-identical copy of the frame. That had two measured consequences —
//!
//!   * the ratio transfer (upstream's own resolve, and this project's `compose.comp`
//!     mode 0) degenerates when `proxy == original`, because its rescale target
//!     collapses to the frame itself and the model's answer is discarded. Working
//!     around *that* is what put this project on an additive path gated by a
//!     per-pixel motion mask, and that mask re-derives from noisy frame deltas every
//!     frame, which is what shows up as shimmer;
//!   * the model is shown blown highlights rather than gradation. Upstream: "the model
//!     is never shown a field of flat white whose blown pixels flip between frames --
//!     unstable input is unstable output".
//!
//! Pure math with no Vulkan dependency, the same discipline [`super::color`] keeps, so
//! the GLSL in `shaders/encode.comp` can be checked against it with plain `#[test]`s.
//! The two are hand-kept in sync; GLSL and Rust cannot share source.

use neural_forge_protocol::enums::reversible_mode;

/// The knee point shared by the soft knee and the hybrid curve.
const KNEE: f32 = 0.75;

/// BT.709 luma, the same weights [`super::color::luminance`] uses.
fn luma(rgb: [f32; 3]) -> f32 {
    super::color::luminance(rgb)
}

fn peak(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2])
}

fn scale(rgb: [f32; 3], by: f32) -> [f32; 3] {
    [rgb[0] * by, rgb[1] * by, rgb[2] * by]
}

/// A soft knee instead of a hard ceiling: luminance above [`KNEE`] is rolled off
/// rather than clipped, so the model is never handed a field of flat white whose blown
/// pixels flip between frames.
///
/// The per-channel headroom afterwards is not cosmetic. The roll-off is on luminance,
/// in which blue carries seven percent, so a saturated blue can sit above 1.0 with a
/// low luminance, pass the knee untouched, and then be clipped by the sRGB encode's own
/// saturate. Clipping one channel of a triple is a hue rotation: upstream measured
/// exactly that as the green cast over every blue thing in GTA V — the sky, the denim,
/// the minimap. One scalar over the whole triple cannot move hue, so the peak channel
/// is brought to 1 that way instead.
pub fn soft_knee(display: [f32; 3]) -> [f32; 3] {
    let mut out = display;
    let display_luma = luma(out);
    if display_luma > KNEE {
        let rolled = KNEE + 0.25 * (1.0 - (-(display_luma - KNEE) / 0.25).exp());
        out = scale(out, rolled / display_luma);
    }
    let p = peak(out);
    if p > 1.0 {
        out = scale(out, 1.0 / p);
    }
    out
}

/// `[0, inf) -> [0, 1)` with no clip point.
fn neutwo(x: f32) -> f32 {
    x / (x * x + 1.0).sqrt()
}

/// The reversible proxy: unclipped and hue-preserving, applied as one scalar on the
/// peak channel so the three channels keep their ratios. The knee reaches its asymptote
/// within a stop of white, compressing highlight gradation into a band too thin for the
/// model to resolve; this keeps it.
pub fn neutwo_encode(v: [f32; 3]) -> [f32; 3] {
    let v = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)];
    let m = peak(v);
    if m <= 1e-6 {
        return v;
    }
    scale(v, neutwo(m) / m)
}

/// Identity below the knee — so midtones are exactly what the soft knee already gave —
/// and an unclipped roll of the excess above it, so highlights keep their gradation.
/// C1-continuous at the knee.
fn hybrid_curve(m: f32) -> f32 {
    if m <= KNEE {
        return m;
    }
    let e = (m - KNEE) / (1.0 - KNEE);
    KNEE + (1.0 - KNEE) * neutwo(e)
}

/// [`hybrid_curve`] applied as one scalar on the peak channel, hue preserved.
pub fn hybrid_encode(v: [f32; 3]) -> [f32; 3] {
    let v = [v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)];
    let m = peak(v);
    if m <= 1e-6 {
        return v;
    }
    scale(v, hybrid_curve(m) / m)
}

/// The whole encode, in the space the GLSL works in: linear-light frame in,
/// display-referred `[0, 1]` proxy out (the caller applies the sRGB encode).
///
/// `white_point` is what the model should treat as white; the resolve undoes the divide.
pub fn encode_proxy(frame_linear: [f32; 3], white_point: f32, mode: u32) -> [f32; 3] {
    let scale_by = 1.0 / white_point.max(1e-6);
    let normalized = scale(frame_linear, scale_by);
    match mode {
        reversible_mode::KNEE => soft_knee(normalized),
        m if m >= reversible_mode::HYBRID => hybrid_encode(normalized),
        _ => neutwo_encode(normalized),
    }
}

fn srgb_decode_byte(byte: u8) -> f32 {
    let c = f32::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode_byte(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The whole encode on one RGBA8 pixel, byte in and byte out -- the CPU twin of what
/// `shaders/encode.comp` does on the GPU, for exactly the same inputs.
pub fn reference_encode_pixel(pixel: [u8; 4], bgr_order: bool, white_point: f32, mode: u32) -> [u8; 4] {
    let (r, g, b) = if bgr_order { (pixel[2], pixel[1], pixel[0]) } else { (pixel[0], pixel[1], pixel[2]) };
    let linear = [srgb_decode_byte(r), srgb_decode_byte(g), srgb_decode_byte(b)];
    let display = encode_proxy(linear, white_point, mode);
    let out = [srgb_encode_byte(display[0]), srgb_encode_byte(display[1]), srgb_encode_byte(display[2])];
    if bgr_order {
        [out[2], out[1], out[0], pixel[3]]
    } else {
        [out[0], out[1], out[2], pixel[3]]
    }
}

/// Checks a GPU-produced proxy against this module's own CPU reference, pixel for
/// pixel, and reports `(max_channel_delta, mean_channel_delta, pixels_compared)`.
///
/// This is what makes the encode verifiable on real hardware without anyone looking at
/// a screen: the layer holds the untouched frame and the encoded proxy in CPU memory at
/// the same moment, so the GPU's answer can simply be compared to the arithmetic it was
/// supposed to perform. A max delta of 0-2 is rounding; anything large means the GPU
/// path is wrong (a channel-order mistake shows up immediately and hugely, since red
/// and blue diverge far more than one count).
///
/// `None` when the two buffers do not describe the same raster -- the proxy is only
/// pixel-aligned with the frame at a working scale of exactly 1.0, so a caller wanting
/// this check has to ask for that scale.
pub fn compare_to_reference(
    original_rgba: &[u8],
    proxy_rgba: &[u8],
    bgr_order: bool,
    white_point: f32,
    mode: u32,
) -> Option<(u8, f32, usize)> {
    if original_rgba.len() != proxy_rgba.len() || original_rgba.len() < 4 {
        return None;
    }
    let mut max_delta = 0u8;
    let mut total = 0u64;
    let mut channels = 0usize;
    for (frame, got) in original_rgba.chunks_exact(4).zip(proxy_rgba.chunks_exact(4)) {
        let want = reference_encode_pixel([frame[0], frame[1], frame[2], frame[3]], bgr_order, white_point, mode);
        // Alpha is carried through untouched by both paths, so only colour is compared.
        for i in 0..3 {
            let delta = want[i].abs_diff(got[i]);
            max_delta = max_delta.max(delta);
            total += u64::from(delta);
            channels += 1;
        }
    }
    if channels == 0 {
        return None;
    }
    Some((max_delta, total as f32 / channels as f32, channels / 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn hue_ratios(rgb: [f32; 3]) -> [f32; 3] {
        let p = peak(rgb);
        if p <= 0.0 {
            return [0.0; 3];
        }
        [rgb[0] / p, rgb[1] / p, rgb[2] / p]
    }

    #[test]
    fn the_knee_leaves_midtones_alone() {
        // Below the knee nothing is touched at all, so the proxy is the frame there and
        // the encode cannot cost midtone fidelity.
        for v in [0.0f32, 0.05, 0.2, 0.5] {
            let out = soft_knee([v, v, v]);
            assert!(close(out[0], v), "{v} moved to {}", out[0]);
        }
    }

    #[test]
    fn the_knee_rolls_off_rather_than_clipping_and_never_exceeds_one() {
        // The whole point: a blown input must come back as a distinguishable value
        // below 1, not as flat white. Two different very bright inputs must stay
        // different, which is what the model needs to see.
        let a = soft_knee([4.0, 4.0, 4.0]);
        let b = soft_knee([8.0, 8.0, 8.0]);
        assert!(a[0] < 1.0 && b[0] <= 1.0, "clipped: {a:?} {b:?}");
        assert!(a[0] < b[0] || close(a[0], b[0]), "not monotonic: {a:?} {b:?}");
    }

    #[test]
    fn a_saturated_blue_keeps_its_hue_instead_of_arriving_cyan() {
        // Upstream's measured GTA V failure: blue counts for 7% of luminance, so a
        // saturated blue above 1.0 passes the luminance roll-off untouched and then has
        // exactly one channel clipped by the encode -- which is a hue rotation, and read
        // on screen as a green cast over the sky, the denim and the minimap. The peak
        // scalar has to bring it inside without moving the ratios.
        let blue = [0.1f32, 0.2, 2.0];
        let out = soft_knee(blue);
        assert!(peak(out) <= 1.0 + 1e-6, "left the cube: {out:?}");
        let before = hue_ratios(blue);
        let after = hue_ratios(out);
        for i in 0..3 {
            assert!(close(before[i], after[i]), "hue moved on channel {i}: {before:?} -> {after:?}");
        }
    }

    #[test]
    fn neutwo_is_unclipped_and_strictly_monotonic_in_the_highlights() {
        // Where the knee's asymptote makes two bright scenes indistinguishable, Neutwo
        // has to keep them apart -- that is the reason it exists.
        let a = neutwo_encode([2.0, 2.0, 2.0])[0];
        let b = neutwo_encode([3.0, 3.0, 3.0])[0];
        assert!(a < 1.0 && b < 1.0, "Neutwo clipped: {a} {b}");
        assert!(b - a > 1e-3, "highlights collapsed together: {a} vs {b}");
    }

    #[test]
    fn neutwo_and_hybrid_preserve_hue_exactly() {
        let v = [0.3f32, 1.4, 0.05];
        for out in [neutwo_encode(v), hybrid_encode(v)] {
            let before = hue_ratios(v);
            let after = hue_ratios(out);
            for i in 0..3 {
                assert!(close(before[i], after[i]), "hue moved: {before:?} -> {after:?}");
            }
        }
    }

    #[test]
    fn the_hybrid_curve_is_identity_below_the_knee_and_continuous_across_it() {
        assert!(close(hybrid_curve(0.0), 0.0));
        assert!(close(hybrid_curve(0.5), 0.5));
        assert!(close(hybrid_curve(KNEE), KNEE));
        // C1-continuous: approaching the knee from above must arrive at the knee's own
        // value, with no step.
        let just_above = hybrid_curve(KNEE + 1e-4);
        assert!(just_above >= KNEE && just_above - KNEE < 1e-3, "step at the knee: {just_above}");
        // And above it, unclipped.
        assert!(hybrid_curve(100.0) < 1.0, "hybrid clipped");
    }

    #[test]
    fn the_white_point_divide_is_what_scales_the_frame() {
        // Half the white point means everything is twice as bright going in, which is
        // the control's whole job. Checked below the knee so the curve itself is the
        // identity and only the divide shows.
        let out = encode_proxy([0.1, 0.1, 0.1], 0.5, reversible_mode::KNEE);
        assert!(close(out[0], 0.2), "white point ignored: {out:?}");
    }

    #[test]
    fn a_black_pixel_encodes_to_black_under_every_curve() {
        for mode in [reversible_mode::KNEE, 1, reversible_mode::HYBRID] {
            let out = encode_proxy([0.0, 0.0, 0.0], 1.0, mode);
            assert!(out.iter().all(|c| close(*c, 0.0)), "mode {mode} moved black: {out:?}");
        }
    }

    #[test]
    fn comparing_a_reference_encode_against_itself_is_exact() {
        // The self-check has to be exact on its own output, or a real GPU delta could
        // not be distinguished from the check's own noise.
        let frame: Vec<u8> = (0..64u32).flat_map(|i| [(i * 3) as u8, (i * 5) as u8, (i * 7) as u8, 255]).collect();
        let encoded: Vec<u8> = frame
            .chunks_exact(4)
            .flat_map(|p| reference_encode_pixel([p[0], p[1], p[2], p[3]], false, 1.0, reversible_mode::KNEE))
            .collect();
        let (max, mean, pixels) = compare_to_reference(&frame, &encoded, false, 1.0, reversible_mode::KNEE).expect("same raster");
        assert_eq!(max, 0, "reference disagreed with itself");
        assert_eq!(mean, 0.0);
        assert_eq!(pixels, 64);
    }

    #[test]
    fn the_self_check_catches_a_swapped_channel_order() {
        // The single most likely GPU-side mistake, and the one that is invisible in
        // unit tests of the curves alone. Encoding with the wrong order has to show up
        // as a large delta, not a rounding-sized one.
        //
        // The pixels have to be *bright* and asymmetric for this to be a real test. On
        // an 8-bit display-referred frame with a white point of 1.0 the normalized
        // values never exceed 1.0, so the peak clamp never fires and the knee only
        // fires above 0.75 luminance -- below that the encode is the identity, under
        // which a channel swap is symmetric on the way in and out and cancels exactly.
        // (That is also why the encode changes little in an SDR midtone: its work is in
        // the highlights. See `compose.comp`'s mode 2, which is what makes the ratio
        // non-degenerate regardless of how much the encode moved.)
        // Where the encode is order-sensitive at all is narrower than it looks, and
        // worth stating because it bounds what this check can prove:
        //
        //   * below 0.75 luminance the knee is the identity, and a swap applied on the
        //     way in and undone on the way out cancels exactly;
        //   * once the peak clamp fires, the net scale is `1/peak` -- and peak is the
        //     max over the three channels, which does not care about their order, so a
        //     swap cancels exactly there too;
        //   * in between -- knee firing, peak still inside the cube -- the scale comes
        //     from luminance, which weights the channels unequally, and a swap shows.
        //
        // So the check catches an order mistake on bright in-gamut content and is blind
        // to one on dark or clipped content. That is a property of the encode, not of
        // the check; a small delta here is not proof the order is right.
        let frame: Vec<u8> = (0..32u32).flat_map(|i| [255u8, 250, (i * 2) as u8, 255]).collect();
        let wrong_order: Vec<u8> = frame
            .chunks_exact(4)
            .flat_map(|p| reference_encode_pixel([p[0], p[1], p[2], p[3]], true, 1.0, reversible_mode::KNEE))
            .collect();
        let (max, _, _) = compare_to_reference(&frame, &wrong_order, false, 1.0, reversible_mode::KNEE).expect("same raster");
        assert!(max > 4, "a swapped channel order only moved {max} counts -- the check would not catch it");
    }

    #[test]
    fn the_self_check_refuses_mismatched_rasters_rather_than_reporting_nonsense() {
        // At any working scale below 1.0 the proxy is smaller than the frame, and
        // comparing them pixel-for-pixel would be meaningless rather than merely
        // imprecise.
        assert!(compare_to_reference(&[0; 64], &[0; 32], false, 1.0, reversible_mode::KNEE).is_none());
    }

    #[test]
    fn the_encode_is_a_pure_function_so_the_resolve_can_reproduce_it() {
        // The resolve rebuilds the frame's own proxy rather than keeping the encoded one
        // at full resolution, which is only sound if the curve depends on nothing but
        // the pixel. Same input, same answer, every time.
        let v = [0.42, 0.91, 1.7];
        let first = encode_proxy(v, 1.0, reversible_mode::KNEE);
        for _ in 0..4 {
            assert_eq!(first, encode_proxy(v, 1.0, reversible_mode::KNEE));
        }
    }
}
