//! Applies the composition math to a captured frame, on the CPU.
//!
//! This is a direct, mechanical port of `shaders/compose.comp`'s own `main()` --
//! `UpgradeToneMap` -> the transfer-ratio blend -> `GamutCompressReversible` -- built
//! on the exact same, already-`#[test]`-verified [`super::color`] primitives that
//! shader is hand-translated from. The orchestration below is new (previously nothing
//! called `color`'s functions outside `#[cfg(test)]` at all -- see the crate's
//! `CLAUDE.md` "composition" entry), but the algorithm itself is not: this file adds
//! no new math, only the per-pixel loop and sRGB encode/decode needed to run
//! `color`'s linear-light functions against real `RGBA8` bytes.
//!
//! GPU dispatch (the actual `compose.comp` shader) is still open work -- this CPU path
//! is the correctness-first way to get the algorithm actually reaching the presented
//! frame at all, not a replacement for eventually moving it to the GPU. Only `RGBA8`
//! is handled; `RGBA16F` (the HDR proxy format) still falls back to the caller's
//! existing raw-answer passthrough, same as before this file existed -- a real,
//! documented gap, not a silent wrong answer.

use super::color;

/// Composites `answer` (the model's raw `RGBA8` answer) against `original` (the frame
/// before any model edit, same format/size) in place.
///
/// No separate downscaled proxy exists yet (`capture.rs` sends the full captured frame
/// as the proxy the model is shown) -- `proxy` and `original` are the same picture
/// here, which is a real, documented gap (see `crates/layer/src/capture.rs`), not a
/// bug in this function.
///
/// `debug_view`: `0` the normal composited result; `1` the original frame, unmodified
/// (`answer` is overwritten with `original`); `2` the model's raw answer, unmodified
/// (this function is a no-op); `3` the composited result's difference from `original`,
/// amplified 4x and re-centered at mid-gray, so a subtle real edit is visible without
/// needing to A/B two screenshots. Matches `neural_forge_protocol::ShmHeader::debug_view`'s
/// own doc comment.
///
/// Both buffers must be `RGBA8` (4 bytes/pixel, sRGB-encoded, same length) -- callers
/// must not call this for any other `proxy_format`. A length mismatch processes
/// whichever is shorter and leaves the rest of `answer` untouched, rather than
/// panicking (fails open, same discipline as the rest of this crate's capture path).
pub fn apply_rgba8(
    original: &[u8],
    answer: &mut [u8],
    colour_strength: f32,
    transfer_strength: f32,
    max_ratio: f32,
    debug_view: u32,
    bgr_order: bool,
) {
    if debug_view == 2 {
        return;
    }
    let n = original.len().min(answer.len());
    if debug_view == 1 {
        answer[..n].copy_from_slice(&original[..n]);
        return;
    }

    // Guards the `1.0 / max_ratio` below against a pathological (<=1.0, e.g. from a
    // corrupt or adversarial SHM write) setting -- `ShmHeader::init_defaults` sets 2.0,
    // and the GUI has no control that can drive it below 1.0 in practice, but this
    // function has no way to know that's still true by the time it runs.
    let max_ratio = max_ratio.max(1.0 + 1e-4);
    let colour_strength = colour_strength.clamp(0.0, 1.0);
    let transfer_strength = transfer_strength.clamp(0.0, 1.0);

    // Each pixel is several `powf`/`cbrt` calls deep (sRGB decode/encode, two OkLab
    // conversions inside `upgrade_tone_map`) -- real, measured cost on real hardware:
    // ~1.94M pixels at 1080p took ~800ms single-threaded, dropping a real `vkcube` run
    // from ~24fps to ~1fps. Release vs. debug barely moved that number, confirming the
    // cost is the transcendental calls themselves, not un-optimized surrounding code.
    // Splitting the independent, per-pixel work across threads is a real, measured fix
    // for that -- every pixel here only ever reads its own `original`/`answer` bytes
    // and writes its own `answer` bytes, so there is no cross-pixel dependency to
    // serialize on. The real fix (dispatching `shaders/compose.comp` on the GPU
    // instead) is still open work; this is what makes the CPU path fast enough to
    // actually measure a real game against in the meantime.
    let threads = std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1).min(16);
    let pixels = n / 4;
    let chunk_pixels = pixels.div_ceil(threads).max(1);
    let chunk_bytes = chunk_pixels * 4;

    std::thread::scope(|scope| {
        for (orig_chunk, ans_chunk) in original[..n].chunks(chunk_bytes).zip(answer[..n].chunks_mut(chunk_bytes)) {
            scope.spawn(move || {
                // `original`/`answer` are raw captured/model bytes -- B,G,R,A order for
                // a `B8G8R8A8` swapchain, R,G,B,A for `R8G8B8A8` (see
                // `swapchain::is_bgr_order`'s doc comment for why this matters and how
                // it was found). Read/write through the real R/B slot indices so
                // `compose_pixel`'s own math -- which only ever deals in canonical
                // `[r, g, b]` triples -- never has to know which order the bytes came
                // in.
                let (r, b) = if bgr_order { (2, 0) } else { (0, 2) };
                for (orig_px, ans_px) in orig_chunk.chunks_exact(4).zip(ans_chunk.chunks_exact_mut(4)) {
                    let new_pixel = compose_pixel(
                        [orig_px[r], orig_px[1], orig_px[b]],
                        [ans_px[r], ans_px[1], ans_px[b]],
                        colour_strength,
                        transfer_strength,
                        max_ratio,
                        debug_view,
                    );
                    ans_px[r] = new_pixel[0];
                    ans_px[1] = new_pixel[1];
                    ans_px[b] = new_pixel[2];
                    // ans_px[3] (alpha): left as whatever the model's answer already had.
                }
            });
        }
    });
}

/// One pixel of the `compose.comp` pipeline: `UpgradeToneMap` -> the transfer-ratio
/// blend -> `GamutCompressReversible` -> (for `debug_view == 3` only) the amplified
/// diff. Pure and side-effect-free so [`apply_rgba8`] can call it from any thread.
fn compose_pixel(original: [u8; 3], model_answer: [u8; 3], colour_strength: f32, transfer_strength: f32, max_ratio: f32, debug_view: u32) -> [u8; 3] {
    let original_lin = [srgb_decode(original[0]), srgb_decode(original[1]), srgb_decode(original[2])];
    let model_lin = [srgb_decode(model_answer[0]), srgb_decode(model_answer[1]), srgb_decode(model_answer[2])];
    let proxy_lin = original_lin;

    let upgraded = color::upgrade_tone_map(original_lin, proxy_lin, model_lin, colour_strength);

    let inv_max_ratio = 1.0 / max_ratio;
    let safe_proxy = [proxy_lin[0].max(1e-4), proxy_lin[1].max(1e-4), proxy_lin[2].max(1e-4)];
    let ratio = [
        (upgraded[0] / safe_proxy[0]).clamp(inv_max_ratio, max_ratio),
        (upgraded[1] / safe_proxy[1]).clamp(inv_max_ratio, max_ratio),
        (upgraded[2] / safe_proxy[2]).clamp(inv_max_ratio, max_ratio),
    ];
    let transferred = [original_lin[0] * ratio[0], original_lin[1] * ratio[1], original_lin[2] * ratio[2]];
    let blended = [
        original_lin[0] + (transferred[0] - original_lin[0]) * transfer_strength,
        original_lin[1] + (transferred[1] - original_lin[1]) * transfer_strength,
        original_lin[2] + (transferred[2] - original_lin[2]) * transfer_strength,
    ];
    let mut result = color::gamut_compress_reversible(blended);

    if debug_view == 3 {
        result = [
            0.5 + (result[0] - original_lin[0]) * 4.0,
            0.5 + (result[1] - original_lin[1]) * 4.0,
            0.5 + (result[2] - original_lin[2]) * 4.0,
        ];
    }

    [srgb_encode(result[0]), srgb_encode(result[1]), srgb_encode(result[2])]
}

/// IEC 61966-2-1 sRGB EOTF (gamma-encoded byte -> linear light), the standard transfer
/// function 8-bit display-referred swapchain content is encoded with.
fn srgb_decode(byte: u8) -> f32 {
    let c = f32::from(byte) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The inverse of [`srgb_decode`]: linear light -> a gamma-encoded byte, clamping to
/// the representable `[0, 1]` range first (the composition math above can produce an
/// out-of-range value transiently, e.g. `debug_view == 3`'s amplification, before this
/// final encode).
fn srgb_encode(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba8(rgb: [u8; 3], width: usize, height: usize) -> Vec<u8> {
        (0..width * height).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect()
    }

    #[test]
    fn debug_view_2_is_a_no_op() {
        let original = solid_rgba8([200, 50, 50], 2, 2);
        let mut answer = solid_rgba8([10, 240, 10], 2, 2);
        let before = answer.clone();
        apply_rgba8(&original, &mut answer, 1.0, 1.0, 2.0, 2, false);
        assert_eq!(answer, before, "debug_view=2 must leave the raw model answer untouched");
    }

    #[test]
    fn debug_view_1_replaces_answer_with_original() {
        let original = solid_rgba8([200, 50, 50], 2, 2);
        let mut answer = solid_rgba8([10, 240, 10], 2, 2);
        apply_rgba8(&original, &mut answer, 1.0, 1.0, 2.0, 1, false);
        assert_eq!(answer, original, "debug_view=1 must show the original/proxy, not the model's answer");
    }

    #[test]
    fn transfer_strength_zero_leaves_the_original_untouched() {
        // blended = original + (transferred - original) * 0 = original for every
        // pixel, regardless of what the model answered -- an independent invariant of
        // the blend itself, not dependent on `upgrade_tone_map`'s own correctness.
        let original = solid_rgba8([180, 90, 30], 2, 2);
        let mut answer = solid_rgba8([5, 5, 200], 2, 2);
        apply_rgba8(&original, &mut answer, 1.0, 0.0, 2.0, 0, false);
        for (o, a) in original.chunks_exact(4).zip(answer.chunks_exact(4)) {
            for c in 0..3 {
                assert!((i32::from(o[c]) - i32::from(a[c])).abs() <= 1, "expected ~original at transfer_strength=0, got {a:?} vs {o:?}");
            }
        }
    }

    #[test]
    fn model_equal_to_original_is_near_identity() {
        // proxy == original == model here -> upgrade_tone_map's target luminance is
        // exactly original's own, the OkLab hue-correction has nothing to correct
        // (model's hue already equals original's), and the transfer ratio is ~1 --
        // the whole pipeline should reproduce `original` to within 8-bit rounding.
        let same = solid_rgba8([160, 120, 40], 2, 2);
        let mut answer = same.clone();
        apply_rgba8(&same, &mut answer, 1.0, 1.0, 2.0, 0, false);
        for (o, a) in same.chunks_exact(4).zip(answer.chunks_exact(4)) {
            for c in 0..3 {
                assert!((i32::from(o[c]) - i32::from(a[c])).abs() <= 1, "expected ~identity, got {a:?} vs {o:?}");
            }
        }
    }

    #[test]
    fn srgb_round_trips_every_byte_value() {
        for byte in 0..=255u8 {
            let back = srgb_encode(srgb_decode(byte));
            assert!((i32::from(back) - i32::from(byte)).abs() <= 1, "sRGB round trip failed for {byte}: got {back}");
        }
    }
}
