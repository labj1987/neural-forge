//! The supersampling down-leg's resampling filters: standard, well-published kernels
//! (Lanczos, Catmull-Rom, Mitchell-Netravali/"bicubic", Kaiser-windowed sinc), each
//! implemented directly from its mathematical definition — none of this needs, or
//! reads, upstream's `.spv`/`.h` pairs. Matches
//! [`neural_forge_protocol::enums::downscaler`]'s numbering (kept because it's a settings
//! contract, not an algorithm — see that module's doc comment).
//!
//! Each filter here is a 1D kernel; a 2D resample separably applies it along each
//! axis (standard practice for separable kernels — all of these are), which the GPU
//! compute shader that actually runs this does as two passes. This module is the
//! reference the shader is translated from and the thing `#[test]`s below check
//! against known closed-form values.

use neural_forge_protocol::enums::downscaler;

/// `sinc(x) = sin(pi*x) / (pi*x)`, with the removable singularity at 0 filled in.
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

/// The Lanczos kernel with `lobes` lobes on each side: a windowed sinc, zero outside
/// `[-lobes, lobes]`.
fn lanczos(x: f32, lobes: f32) -> f32 {
    if x.abs() >= lobes {
        0.0
    } else {
        sinc(x) * sinc(x / lobes)
    }
}

/// Catmull-Rom: the Mitchell-Netravali family member with `b=0, c=0.5` — interpolating
/// (passes through the input samples exactly), unlike Mitchell-Netravali's own default.
fn catmull_rom(x: f32) -> f32 {
    mitchell_netravali(x, 0.0, 0.5)
}

/// The Mitchell-Netravali two-parameter cubic filter family (Mitchell & Netravali,
/// *Reconstruction Filters in Computer Graphics*, 1988) — `b=1/3, c=1/3` is their own
/// recommended default ("bicubic" in the protocol's naming); `b=0, c=0.5` is
/// Catmull-Rom (see [`catmull_rom`]).
fn mitchell_netravali(x: f32, b: f32, c: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax.powi(3) + (-18.0 + 12.0 * b + 6.0 * c) * ax.powi(2) + (6.0 - 2.0 * b)) / 6.0
    } else if ax < 2.0 {
        ((-b - 6.0 * c) * ax.powi(3)
            + (6.0 * b + 30.0 * c) * ax.powi(2)
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn bicubic(x: f32) -> f32 {
    mitchell_netravali(x, 1.0 / 3.0, 1.0 / 3.0)
}

/// A Kaiser-windowed sinc with the given `beta` (higher = narrower main lobe, more
/// sidelobe suppression) and support radius `lobes`.
fn kaiser(x: f32, lobes: f32, beta: f32) -> f32 {
    if x.abs() >= lobes {
        return 0.0;
    }
    sinc(x) * kaiser_window(x / lobes, beta)
}

/// The Kaiser window itself, using the modified Bessel function `I0` (series
/// expansion — standard, e.g. Oppenheim & Schafer, *Discrete-Time Signal Processing*).
fn kaiser_window(t: f32, beta: f32) -> f32 {
    let arg = beta * (1.0 - t * t).max(0.0).sqrt();
    bessel_i0(arg) / bessel_i0(beta)
}

fn bessel_i0(x: f32) -> f32 {
    // I0(x) = sum_{k=0}^inf ( (x/2)^(2k) / (k!)^2 ). Converges quickly for the small
    // `beta` values (a handful) any reasonable Kaiser window uses; 24 terms is
    // overkill headroom, not a tuned minimum.
    let half_x_sq = (x * 0.5) * (x * 0.5);
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    for k in 1..24 {
        term *= half_x_sq / (k as f32 * k as f32);
        sum += term;
    }
    sum
}

/// The 1D kernel value at offset `x` (in samples) for the given
/// [`neural_forge_protocol::enums::downscaler`] value. Unsupported/unknown values (`FSR1`,
/// or anything out of range) fall back to `LANCZOS3`, matching the protocol's own
/// documented fallback behavior for a value this pipeline can't run.
pub fn kernel(x: f32, downscaler_kind: u32) -> f32 {
    match downscaler_kind {
        downscaler::BICUBIC => bicubic(x),
        downscaler::CATMULL_ROM => catmull_rom(x),
        downscaler::LANCZOS2 => lanczos(x, 2.0),
        downscaler::KAISER2 => kaiser(x, 2.0, 6.0),
        downscaler::KAISER3 => kaiser(x, 3.0, 8.0),
        downscaler::MAGIC => mitchell_netravali(x, 0.0, 0.6),
        _ => lanczos(x, 3.0), // LANCZOS3 and any unrecognized value both fall back here
    }
}

/// The kernel's support radius in samples — how far out a resample needs to gather
/// input taps from for this filter.
pub fn support_radius(downscaler_kind: u32) -> f32 {
    match downscaler_kind {
        downscaler::BICUBIC | downscaler::CATMULL_ROM | downscaler::MAGIC => 2.0,
        downscaler::LANCZOS2 => 2.0,
        downscaler::KAISER2 => 2.0,
        downscaler::KAISER3 => 3.0,
        _ => 3.0,
    }
}

/// One axis's worth of resample weights for every output sample: `taps[i]` is the
/// (first source index, weights starting there) pair for output index `i`. Shared by
/// both passes of [`resample_rgba8`] -- computed once per axis, not once per pixel.
struct AxisPlan {
    /// `first[i]` is the smallest source index output sample `i` reads from; the
    /// weights for it live at `weights[weight_offsets[i]..weight_offsets[i+1]]`.
    first: Vec<i32>,
    weight_offsets: Vec<usize>,
    weights: Vec<f32>,
}

fn plan_axis(src_len: u32, dst_len: u32, downscaler_kind: u32) -> AxisPlan {
    let src_len_f = src_len as f32;
    let dst_len_f = dst_len as f32;
    // Minifying stretches the kernel's own footprint by the minification ratio -- the
    // standard fix for a resize filter used to downsample (an unstretched kernel
    // would alias, sampling the source no more densely than the *output* asks for
    // rather than respecting how much *source* detail needs to be averaged away).
    // Magnifying uses the kernel at its native width: there's no source detail to
    // alias away, just more output samples than input ones.
    let scale = (src_len_f / dst_len_f).max(1.0);
    let radius = support_radius(downscaler_kind) * scale;
    let mut first = Vec::with_capacity(dst_len as usize);
    let mut weight_offsets = Vec::with_capacity(dst_len as usize + 1);
    let mut weights = Vec::new();
    weight_offsets.push(0);
    for i in 0..dst_len {
        // Sample centers align pixel *centers*, not edges (`+0.5 ... -0.5`) -- the
        // standard image-resize convention; aligning edges instead would visibly
        // shift content toward a corner on any non-integer scale factor.
        let center = (i as f32 + 0.5) * (src_len_f / dst_len_f) - 0.5;
        let lo = (center - radius).floor() as i32;
        let hi = (center + radius).ceil() as i32;
        let start_len = weights.len();
        let mut sum = 0.0f32;
        for tap in lo..=hi {
            let x = (tap as f32 - center) / scale;
            let w = kernel(x, downscaler_kind);
            weights.push(w);
            sum += w;
        }
        // Renormalize so the taps actually used (after the edge-clamp in the caller
        // folds outside-source taps back onto the nearest valid one) still sum to
        // exactly 1 -- otherwise a resize near the border measurably dims or
        // brightens that edge, and finite-precision kernel evaluation dims/brightens
        // very slightly everywhere even ignoring edges.
        if sum.abs() > 1e-6 {
            for w in &mut weights[start_len..] {
                *w /= sum;
            }
        }
        first.push(lo);
        weight_offsets.push(weights.len());
    }
    AxisPlan { first, weight_offsets, weights }
}

/// Resamples an interleaved RGBA8 buffer from `(src_w, src_h)` to `(dst_w, dst_h)`
/// using the [`neural_forge_protocol::enums::downscaler`] kernel named by
/// `downscaler_kind`, applied separably (horizontal pass, then vertical) exactly as
/// this module's own doc comment says every kernel here is meant to be used. Works in
/// either direction -- minifying (sending a smaller proxy to the model) or magnifying
/// (bringing its smaller answer back up to the frame's own resolution) -- since an
/// interpolating resize kernel is standard for both (this is not a downscale-only
/// operation despite the module's name). Edge taps clamp to the nearest valid source
/// pixel rather than reading out of bounds (standard "clamp to edge" resize
/// behavior). Identity-sized calls (`src_w==dst_w && src_h==dst_h`) skip all of the
/// above and copy the exact input bytes -- every caller's behavior before this
/// function existed, still exact when scaling is off.
///
/// `src` must hold at least `src_w * src_h * 4` bytes; a shorter buffer is treated as
/// all-zero padding rather than panicking, matching this crate's general "never crash
/// the game over a malformed frame" discipline. Returns exactly `dst_w * dst_h * 4`
/// bytes.
pub fn resample_rgba8(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, downscaler_kind: u32) -> Vec<u8> {
    let dst_len = (dst_w as usize) * (dst_h as usize) * 4;
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return vec![0u8; dst_len];
    }
    if src_w == dst_w && src_h == dst_h {
        let mut out = vec![0u8; dst_len];
        let n = out.len().min(src.len());
        out[..n].copy_from_slice(&src[..n]);
        return out;
    }
    let get = |x: i32, y: i32, c: usize| -> f32 {
        let x = x.clamp(0, src_w as i32 - 1) as usize;
        let y = y.clamp(0, src_h as i32 - 1) as usize;
        let idx = (y * src_w as usize + x) * 4 + c;
        src.get(idx).copied().unwrap_or(0) as f32
    };

    let plan_x = plan_axis(src_w, dst_w, downscaler_kind);
    let plan_y = plan_axis(src_h, dst_h, downscaler_kind);

    // Horizontal pass: (src_w, src_h) -> (dst_w, src_h), still in f32 (a resize
    // filter's weights are signed and can overshoot 0..255 mid-computation --
    // clamping only happens once, on the final byte write below).
    let mut mid = vec![0f32; (dst_w as usize) * (src_h as usize) * 4];
    for y in 0..src_h as i32 {
        for (x, &first) in plan_x.first.iter().enumerate() {
            let w_range = plan_x.weight_offsets[x]..plan_x.weight_offsets[x + 1];
            let weights = &plan_x.weights[w_range];
            let mut acc = [0f32; 4];
            for (t, &w) in weights.iter().enumerate() {
                let sx = first + t as i32;
                for c in 0..4 {
                    acc[c] += get(sx, y, c) * w;
                }
            }
            let out_idx = (y as usize * dst_w as usize + x) * 4;
            mid[out_idx..out_idx + 4].copy_from_slice(&acc);
        }
    }

    // Vertical pass: (dst_w, src_h) -> (dst_w, dst_h), reading `mid` with the same
    // edge-clamp convention as `get` above (min/max instead of `get`'s modulo-free
    // clamp since `mid` is already a plain, fully populated buffer, not the original
    // sparse-length-tolerant `src`).
    let mid_get = |x: usize, y: i32, c: usize| -> f32 {
        let y = y.clamp(0, src_h as i32 - 1) as usize;
        mid[(y * dst_w as usize + x) * 4 + c]
    };
    let mut out = vec![0u8; dst_len];
    for x in 0..dst_w as usize {
        for (y, &first) in plan_y.first.iter().enumerate() {
            let w_range = plan_y.weight_offsets[y]..plan_y.weight_offsets[y + 1];
            let weights = &plan_y.weights[w_range];
            let mut acc = [0f32; 4];
            for (t, &w) in weights.iter().enumerate() {
                let sy = first + t as i32;
                for c in 0..4 {
                    acc[c] += mid_get(x, sy, c) * w;
                }
            }
            let out_idx = (y * dst_w as usize + x) * 4;
            for c in 0..4 {
                out[out_idx + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolating_kernels_are_one_at_zero_and_zero_at_their_own_integer_taps() {
        // The defining property of an *interpolating* kernel: it reproduces the sample
        // it sits on exactly (1 at x=0) and doesn't leak into neighboring integer
        // sample positions within its support (0 at x=+-1, +-2, ... up to its radius).
        // Every sinc-based filter here (Lanczos, Kaiser: sinc(0)=1, sinc(integer)=0)
        // and Catmull-Rom (Mitchell-Netravali B=0,C=0.5) has this property.
        // `BICUBIC` (Mitchell-Netravali's own recommended B=1/3,C=1/3) and `MAGIC`
        // (B=0,C=0.6) are deliberately excluded: Mitchell-Netravali's defining
        // trade-off is trading exact interpolation for a smoother reconstruction, so
        // `f(0) = 1 - B/3 != 1` whenever `B != 0` is correct behavior, not a bug.
        for &d in &[downscaler::CATMULL_ROM, downscaler::LANCZOS2, downscaler::LANCZOS3, downscaler::KAISER2, downscaler::KAISER3]
        {
            assert!((kernel(0.0, d) - 1.0).abs() < 1e-5, "kernel {d} at 0 should be 1");
            let radius = support_radius(d) as i32;
            for i in 1..radius {
                let v = kernel(i as f32, d);
                assert!(v.abs() < 1e-4, "kernel {d} at integer tap {i} should be ~0, got {v}");
            }
        }
    }

    #[test]
    fn mitchell_netravali_default_is_deliberately_non_interpolating() {
        // f(0) = (6 - 2B) / 6 = 1 - B/3 for the general B,C filter -- confirms the
        // "trades interpolation for smoothness" property the test above relies on to
        // justify excluding BICUBIC/MAGIC, rather than just asserting a number.
        let b_bicubic = 1.0 / 3.0;
        assert!((kernel(0.0, downscaler::BICUBIC) - (1.0 - b_bicubic / 3.0)).abs() < 1e-5);
        assert!(kernel(0.0, downscaler::BICUBIC) < 1.0, "B=1/3 must not be interpolating");
    }

    #[test]
    fn catmull_rom_matches_known_closed_form_at_half_sample() {
        // Catmull-Rom's cardinal-spline blending weights at t=0.5 are the well-known
        // (-1/16, 9/16, 9/16, -1/16); this kernel's value at x=0.5 (distance from the
        // nearest sample to the query point) is that 9/16 = 0.5625, not 0.5 -- a
        // previous version of this test asserted 0.5 from a since-corrected
        // hand-computation and would have hidden a real bug in the other direction.
        assert!((catmull_rom(0.5) - 0.5625).abs() < 1e-5, "got {}", catmull_rom(0.5));
    }

    #[test]
    fn kernels_are_symmetric() {
        for &d in &[downscaler::BICUBIC, downscaler::CATMULL_ROM, downscaler::LANCZOS3, downscaler::KAISER3] {
            for x in [0.3f32, 0.7, 1.4, 2.1] {
                assert!(
                    (kernel(x, d) - kernel(-x, d)).abs() < 1e-5,
                    "kernel {d} should be symmetric, differs at x={x}"
                );
            }
        }
    }

    #[test]
    fn kernels_vanish_beyond_their_support_radius() {
        for &d in
            &[downscaler::BICUBIC, downscaler::CATMULL_ROM, downscaler::LANCZOS2, downscaler::LANCZOS3, downscaler::KAISER3]
        {
            let r = support_radius(d);
            assert_eq!(kernel(r + 0.5, d), 0.0, "kernel {d} should be exactly 0 beyond its support radius");
        }
    }

    #[test]
    fn unknown_downscaler_falls_back_to_lanczos3() {
        assert_eq!(kernel(0.5, downscaler::FSR1), kernel(0.5, downscaler::LANCZOS3));
        assert_eq!(kernel(0.5, 999), kernel(0.5, downscaler::LANCZOS3));
    }

    fn gradient_rgba8(w: u32, h: u32) -> Vec<u8> {
        (0..(w as usize * h as usize))
            .flat_map(|i| {
                let x = (i % w as usize) as u8;
                let y = (i / w as usize) as u8;
                [x, y, x.wrapping_add(y), 255]
            })
            .collect()
    }

    #[test]
    fn identity_resize_is_byte_for_byte_the_input() {
        let src = gradient_rgba8(37, 23);
        for &d in &[downscaler::LANCZOS3, downscaler::BICUBIC, downscaler::CATMULL_ROM] {
            let out = resample_rgba8(&src, 37, 23, 37, 23, d);
            assert_eq!(out, src, "identity resize must be an exact copy, kernel {d}");
        }
    }

    #[test]
    fn resizing_a_flat_colour_image_stays_that_colour() {
        // A constant-color source is the sharpest test of weight normalization: any
        // bug that leaves per-tap weights not summing to 1 (a clamped edge tap, a
        // stretched-kernel miscalculation) shows up as the output drifting off the
        // flat input color, brightest right at the border where edge-clamping bites
        // hardest.
        let (w, h) = (16u32, 12u32);
        let src: Vec<u8> = (0..(w as usize * h as usize)).flat_map(|_| [200u8, 100, 50, 255]).collect();
        for &d in &[downscaler::LANCZOS3, downscaler::BICUBIC, downscaler::CATMULL_ROM, downscaler::KAISER3] {
            for (dw, dh) in [(11u32, 9u32), (23, 17), (16, 12)] {
                let out = resample_rgba8(&src, w, h, dw, dh, d);
                assert_eq!(out.len(), (dw as usize) * (dh as usize) * 4);
                for px in out.chunks_exact(4) {
                    assert!(
                        (px[0] as i32 - 200).abs() <= 1 && (px[1] as i32 - 100).abs() <= 1 && (px[2] as i32 - 50).abs() <= 1,
                        "flat-color resize {w}x{h}->{dw}x{dh} kernel {d} drifted to {px:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn downscale_then_upscale_roundtrip_stays_close_to_the_original() {
        // Not a golden pixel-exact reference (a resize this general has no simple
        // closed form) -- a coarse but real correctness check: shrinking then
        // growing a smooth gradient back to its original size should reproduce
        // something close to the original, not garbage from a transposed axis or an
        // off-by-one in the tap window.
        let (w, h) = (64u32, 48u32);
        let src = gradient_rgba8(w, h);
        let small = resample_rgba8(&src, w, h, 48, 36, downscaler::LANCZOS3);
        let back = resample_rgba8(&small, 48, 36, w, h, downscaler::LANCZOS3);
        assert_eq!(back.len(), src.len());
        let max_diff = src.iter().zip(back.iter()).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs()).max().unwrap();
        assert!(max_diff <= 40, "downscale/upscale roundtrip drifted too far: max per-channel diff {max_diff}");
    }

    #[test]
    fn output_dimensions_are_always_exact() {
        for (sw, sh, dw, dh) in [(2560u32, 1440u32, 1920u32, 1080u32), (1920, 1080, 2560, 1440), (1, 1, 5, 5), (5, 5, 1, 1)] {
            let src = gradient_rgba8(sw, sh);
            let out = resample_rgba8(&src, sw, sh, dw, dh, downscaler::LANCZOS3);
            assert_eq!(out.len(), (dw as usize) * (dh as usize) * 4, "{sw}x{sh}->{dw}x{dh}");
        }
    }

    #[test]
    fn real_resolution_resample_timing() {
        // Not a pass/fail assertion -- this project's `working_scale` runs this
        // function inline on the game's present thread (see `capture::run`'s own
        // doc comments on why CPU cost there is exactly what caused the v0.1.64 fps
        // collapse), so the honest cost at real dimensions needs to be visible in
        // every test run, not just asserted "fast enough" against a guessed bound.
        let (w, h) = (2560u32, 1440u32);
        let (mw, mh) = (1920u32, 1080u32);
        let full = gradient_rgba8(w, h);
        let small = gradient_rgba8(mw, mh);
        let t = std::time::Instant::now();
        let down = resample_rgba8(&full, w, h, mw, mh, downscaler::LANCZOS3);
        let down_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let up = resample_rgba8(&small, mw, mh, w, h, downscaler::LANCZOS3);
        let up_ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(down.len(), (mw as usize) * (mh as usize) * 4);
        assert_eq!(up.len(), (w as usize) * (h as usize) * 4);
        println!("resample_rgba8 timing @ 2560x1440<->1920x1080 (debug build): down={down_ms:.2}ms up={up_ms:.2}ms");
    }
}
