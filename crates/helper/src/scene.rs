//! Scene-cut detection on the CPU: a coarse luma thumbnail of each proxy frame, compared with
//! the previous one. Carrying a flow field across a cut hands the model a field describing
//! content no longer on screen, worse than handing it nothing.
//!
//! Works on both proxy classes: 8-bit frames are averaged as they are; RGBA16F frames are
//! decoded from half floats and put through the same tone map optical flow sees
//! ([`crate::hdr::tonemap_u8`]), so one threshold fits both.
//!
//! Pure arithmetic with no Win32 or Vulkan calls, so it builds and tests natively.

use crate::hdr::{rgb16f_at, tonemap_u8, FormatClass};

/// Every `STEP`th pixel on each axis goes into the thumbnail.
const STEP: usize = 8;

/// A coarse luma thumbnail of a BGRA8/RGBA8 frame (every 8th pixel on each axis), kept
/// between frames for [`is_scene_cut`] instead of a copy of the whole frame.
pub fn luma_thumbnail(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    if frame.len() < w * h * 4 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(w.div_ceil(STEP) * h.div_ceil(STEP));
    for y in (0..h).step_by(STEP) {
        for x in (0..w).step_by(STEP) {
            let i = (y * w + x) * 4;
            // Unweighted average of the three channels: only needs to catch "the whole picture
            // changed", and channel order does not matter for it.
            out.push(((u32::from(frame[i]) + u32::from(frame[i + 1]) + u32::from(frame[i + 2])) / 3) as u8);
        }
    }
    out
}

/// [`luma_thumbnail`] for an RGBA16F frame (8 bytes per pixel, little-endian halves): each
/// sampled channel is tone mapped to 8 bits first, then averaged the same way.
pub fn luma_thumbnail_rgba16f(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    if frame.len() < w * h * 8 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(w.div_ceil(STEP) * h.div_ceil(STEP));
    for y in (0..h).step_by(STEP) {
        for x in (0..w).step_by(STEP) {
            let [r, g, b] = rgb16f_at(frame, (y * w + x) * 8);
            out.push(((u32::from(tonemap_u8(r)) + u32::from(tonemap_u8(g)) + u32::from(tonemap_u8(b))) / 3) as u8);
        }
    }
    out
}

/// The thumbnail for a proxy of either class.
pub fn thumbnail(frame: &[u8], width: u32, height: u32, class: FormatClass) -> Vec<u8> {
    match class {
        FormatClass::Sdr8 => luma_thumbnail(frame, width, height),
        FormatClass::Hdr16 => luma_thumbnail_rgba16f(frame, width, height),
    }
}

/// Whether two thumbnails look like different scenes -- a hard cut (level transition,
/// cutscene, death/respawn) rather than motion within one scene. DLSS5VKLayer's helper runs
/// the same kind of check (`DetectSceneCut`); this is an independent implementation of the
/// generic technique (mean luma delta against a threshold).
pub fn is_scene_cut(previous: &[u8], current: &[u8], threshold: u8) -> bool {
    if previous.len() != current.len() || current.is_empty() {
        return false;
    }
    let sum: u64 = previous.iter().zip(current).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
    sum / current.len() as u64 >= u64::from(threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    /// A uniform RGBA16F frame, every channel `value` (given as half bits), alpha 1.0.
    fn frame16(width: u32, height: u32, value: u16) -> Vec<u8> {
        let px = [value.to_le_bytes(), value.to_le_bytes(), value.to_le_bytes(), 0x3c00u16.to_le_bytes()].concat();
        px.repeat((width * height) as usize)
    }

    #[test]
    fn thumbnail_samples_every_eighth_pixel_per_axis() {
        assert_eq!(luma_thumbnail(&frame(16, 16, 90), 16, 16), vec![90; 4]);
        assert_eq!(luma_thumbnail(&frame(17, 9, 90), 17, 9).len(), 3 * 2);
        assert!(luma_thumbnail(&[1, 2, 3], 16, 16).is_empty(), "a short frame must not panic");
    }

    #[test]
    fn scene_cut_is_not_flagged_for_a_stable_or_slightly_changed_frame() {
        let a = luma_thumbnail(&frame(32, 32, 100), 32, 32);
        let b = luma_thumbnail(&frame(32, 32, 105), 32, 32);
        assert!(!is_scene_cut(&a, &a, 40), "identical frames must never be a cut");
        assert!(!is_scene_cut(&a, &b, 40), "a small uniform change must not be a cut");
    }

    #[test]
    fn scene_cut_is_flagged_for_a_completely_different_frame() {
        let a = luma_thumbnail(&frame(32, 32, 20), 32, 32);
        let b = luma_thumbnail(&frame(32, 32, 220), 32, 32);
        assert!(is_scene_cut(&a, &b, 40));
    }

    #[test]
    fn scene_cut_never_panics_on_mismatched_or_empty_thumbnails() {
        assert!(!is_scene_cut(&[], &[], 40));
        assert!(!is_scene_cut(&[1, 2], &[1, 2, 3], 40));
    }

    #[test]
    fn rgba16f_thumbnail_decodes_and_tone_maps() {
        // 1.0 -> Reinhard 0.5 -> sRGB 0.7354 -> 188.
        assert_eq!(luma_thumbnail_rgba16f(&frame16(16, 16, 0x3c00), 16, 16), vec![188; 4]);
        // 0.0 -> 0; 65504 -> 255; NaN and -Inf -> 0, +Inf -> 255.
        assert_eq!(luma_thumbnail_rgba16f(&frame16(8, 8, 0x0000), 8, 8), vec![0]);
        assert_eq!(luma_thumbnail_rgba16f(&frame16(8, 8, 0x7bff), 8, 8), vec![255]);
        assert_eq!(luma_thumbnail_rgba16f(&frame16(8, 8, 0x7e00), 8, 8), vec![0]);
        assert_eq!(luma_thumbnail_rgba16f(&frame16(8, 8, 0xfc00), 8, 8), vec![0]);
        assert_eq!(luma_thumbnail_rgba16f(&frame16(8, 8, 0x7c00), 8, 8), vec![255]);
        // Same sampling grid as the 8-bit one.
        assert_eq!(luma_thumbnail_rgba16f(&frame16(17, 9, 0x3c00), 17, 9).len(), 3 * 2);
        // Too short for 8 bytes per pixel (an 8-bit-sized buffer) must not panic or read past it.
        assert!(luma_thumbnail_rgba16f(&frame(16, 16, 0), 16, 16).is_empty());
    }

    #[test]
    fn rgba16f_thumbnail_reads_per_pixel_channels() {
        // Pixel (0,0) R=4.0 G=0 B=0; everything else black.
        let mut f = frame16(8, 8, 0x0000);
        f[0..2].copy_from_slice(&0x4400u16.to_le_bytes());
        let expected = (u32::from(crate::hdr::tonemap_u8(4.0)) / 3) as u8;
        assert_eq!(luma_thumbnail_rgba16f(&f, 8, 8), vec![expected]);
    }

    #[test]
    fn hdr_scene_cut_uses_the_same_threshold() {
        // An exposure change within one scene (0.5 -> 0.6) is not a cut; dark -> bright is.
        let dim = thumbnail(&frame16(32, 32, 0x3800), 32, 32, FormatClass::Hdr16);
        let slightly = thumbnail(&frame16(32, 32, 0x38cd), 32, 32, FormatClass::Hdr16);
        let dark = thumbnail(&frame16(32, 32, 0x2000), 32, 32, FormatClass::Hdr16);
        let bright = thumbnail(&frame16(32, 32, 0x4900), 32, 32, FormatClass::Hdr16);
        assert!(!is_scene_cut(&dim, &slightly, 40));
        assert!(is_scene_cut(&dark, &bright, 40));
        assert_eq!(thumbnail(&frame(16, 16, 90), 16, 16, FormatClass::Sdr8), vec![90; 4]);
    }
}
