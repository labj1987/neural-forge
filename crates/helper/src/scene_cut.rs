//! Scene-cut detection on a coarse luma thumbnail of consecutive model frames.
//!
//! Two modes, chosen live through `ShmHeader::scene_cut_mode`
//! ([`neural_forge_protocol::enums::scene_cut_mode`]):
//!
//! - `FIXED` (0, the default): a cut whenever the mean absolute luma delta against the
//!   previous frame reaches a fixed threshold. Exactly the original check.
//! - `RUNNING_BASELINE` (1): the same statistic, judged against a running baseline of what this
//!   scene's frame-to-frame deltas usually are, so a detailed pan (high delta every frame) does
//!   not read as a cut while a jump well above the usual delta does. The design follows
//!   DLSS5VKLayer-Plus's description; this is an independent implementation.
//!
//! Pure CPU and platform-independent, so its tests run natively.

use neural_forge_protocol::enums::scene_cut_mode;

/// The cut threshold both modes use, in mean luma levels (0-255).
pub const THRESHOLD: u8 = 40;
/// The running baseline never counts as quieter than this fraction of the threshold, so a
/// near-static scene (baseline ~0) does not turn every small change into a candidate.
const FLOOR_FRACTION: f32 = 0.35;
/// A frame is a candidate when its delta is at least this multiple of the baseline.
const OVER_FACTOR: f32 = 2.5;
/// Baseline smoothing per frame: slow while a frame is a candidate (a cut must not teach the
/// baseline that cuts are normal), faster otherwise.
const RATE_OVER: f32 = 0.02;
const RATE_UNDER: f32 = 0.10;
/// Consecutive candidate frames that make a cut.
const CONFIRM_FRAMES: u32 = 2;

/// A coarse luma thumbnail of a BGRA8/RGBA8 frame (every 8th pixel on each axis), kept
/// between frames instead of a copy of the whole frame.
pub fn luma_thumbnail(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    const STEP: usize = 8;
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

/// The sum of absolute luma differences between two thumbnails and their length, or `None`
/// when they cannot be compared (different lengths, or empty). The one statistic both modes
/// share: its mean is the per-pixel luma delta.
pub fn delta_sum(previous: &[u8], current: &[u8]) -> Option<(u64, usize)> {
    if previous.len() != current.len() || current.is_empty() {
        return None;
    }
    let sum: u64 = previous.iter().zip(current).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
    Some((sum, current.len()))
}

/// Whether two [`luma_thumbnail`]s look like different scenes by the fixed rule -- a hard cut
/// (level transition, cutscene, death/respawn) rather than motion within one scene. Carrying a
/// flow field across a cut hands the model a field describing content no longer on screen,
/// worse than handing it nothing. DLSS5VKLayer's helper runs the same kind of check
/// (`DetectSceneCut`); this is an independent implementation of the generic technique (mean
/// luma delta against a threshold).
pub fn is_scene_cut(previous: &[u8], current: &[u8], threshold: u8) -> bool {
    delta_sum(previous, current).is_some_and(|(sum, len)| sum / len as u64 >= u64::from(threshold))
}

/// The running-baseline state machine (mode 1), fed one mean delta per frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunningBaseline {
    /// `None` until the first measured mean, which initialises it.
    baseline: Option<f32>,
    /// Consecutive candidate frames so far.
    over_run: u32,
}

impl RunningBaseline {
    /// Feeds one frame's mean delta; returns `(over, cut)`: whether this frame is a candidate,
    /// and whether it completes a cut.
    ///
    /// - `floor = 0.35 * threshold`
    /// - `over = mean >= 2.5 * max(baseline, floor) && mean >= threshold / 2`
    /// - two consecutive `over` frames fire a cut (on the second), and the run restarts
    /// - every frame, after the decision: `baseline += (mean - baseline) * (over ? 0.02 : 0.10)`
    pub fn observe(&mut self, mean: f32, threshold: u8) -> (bool, bool) {
        let threshold = f32::from(threshold);
        let baseline = *self.baseline.get_or_insert(mean);
        let floor = FLOOR_FRACTION * threshold;
        let over = mean >= OVER_FACTOR * baseline.max(floor) && mean >= threshold / 2.0;
        self.over_run = if over { self.over_run + 1 } else { 0 };
        let cut = self.over_run >= CONFIRM_FRAMES;
        if cut {
            self.over_run = 0;
        }
        let rate = if over { RATE_OVER } else { RATE_UNDER };
        self.baseline = Some(baseline + (mean - baseline) * rate);
        (over, cut)
    }

    pub fn baseline(&self) -> Option<f32> {
        self.baseline
    }
}

/// Both modes' state across frames. Fed every model frame motion is estimated for, whatever
/// the mode, so switching mode mid-session finds the running baseline already warm.
#[derive(Debug, Default)]
pub struct SceneCutDetector {
    /// The previous frame's thumbnail: mode 0 compares against it, exactly as before.
    previous: Vec<u8>,
    /// Mode 1's reference: the previous frame, except that it is held on a candidate frame
    /// until the cut is confirmed or the candidate run ends. A real cut is one picture change
    /// followed by frames of the new scene, which barely differ from each other; only against
    /// the held, pre-cut picture is the second frame of the new scene also a candidate.
    reference: Vec<u8>,
    /// Thumbnail dimensions `reference`/`running` belong to; a change resets mode 1.
    dims: (u32, u32),
    running: RunningBaseline,
}

impl SceneCutDetector {
    /// Drops every frame and all mode-1 state (motion switched off).
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The mode-1 baseline, if one has been measured since the last reset.
    pub fn baseline(&self) -> Option<f32> {
        self.running.baseline()
    }

    /// Takes this frame's thumbnail (of a `width`x`height` frame) and returns whether it starts a
    /// new scene under `mode` (an unknown mode reads as `FIXED`).
    pub fn observe(&mut self, thumb: Vec<u8>, width: u32, height: u32, mode: u32) -> bool {
        // Mode 0: bit-for-bit the original decision and `previous` handling.
        let fixed = is_scene_cut(&self.previous, &thumb, THRESHOLD);

        // Mode 1, tracked in every mode.
        let dims = (width.div_ceil(8), height.div_ceil(8));
        let mut running_cut = false;
        if thumb.is_empty() || dims != self.dims {
            self.dims = dims;
            self.running = RunningBaseline::default();
            self.reference.clone_from(&thumb);
        } else if let Some((sum, len)) = delta_sum(&self.reference, &thumb) {
            let mean = sum as f32 / len as f32;
            let (over, cut) = self.running.observe(mean, THRESHOLD);
            running_cut = cut;
            // Held while a candidate run is pending; otherwise (no candidate, or a confirmed
            // cut) the reference moves on to this frame.
            if !over || cut {
                self.reference.clone_from(&thumb);
            }
        } else {
            self.running = RunningBaseline::default();
            self.reference.clone_from(&thumb);
        }

        self.previous = thumb;
        if mode == scene_cut_mode::RUNNING_BASELINE {
            running_cut
        } else {
            fixed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_forge_protocol::enums::scene_cut_mode::{FIXED, RUNNING_BASELINE};

    /// Thumbnail dims the tests use (a 512x288 frame).
    const TW: usize = 64;
    const TH: usize = 36;
    const FW: u32 = (TW * 8) as u32;
    const FH: u32 = (TH * 8) as u32;

    /// xorshift: deterministic noise without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
    }

    /// A smooth world texture (sum of sines, period tens of thumbnail pixels) seen through a
    /// TW x TH window at horizontal offset `x0`.
    fn smooth_view(seed: f32, base: f32, x0: usize) -> Vec<u8> {
        (0..TH)
            .flat_map(|y| {
                (0..TW).map(move |x| {
                    let (x, y) = ((x + x0) as f32, y as f32);
                    let v = base + 40.0 * (x * 0.21 + seed).sin() + 25.0 * (y * 0.17 + x * 0.05 + seed * 2.0).sin();
                    v.clamp(0.0, 255.0) as u8
                })
            })
            .collect()
    }

    /// A high-detail world (independent noise per world pixel) seen through the window.
    fn noisy_world(seed: u64, width: usize) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..width * TH).map(|_| rng.byte()).collect()
    }
    fn noisy_view(world: &[u8], world_w: usize, x0: usize) -> Vec<u8> {
        (0..TH).flat_map(|y| (0..TW).map(move |x| world[y * world_w + x + x0])).collect()
    }

    /// Feeds `frames` and returns the indices where `mode` cut.
    fn cuts(det: &mut SceneCutDetector, frames: impl IntoIterator<Item = Vec<u8>>, mode: u32) -> Vec<usize> {
        frames.into_iter().enumerate().filter_map(|(i, t)| det.observe(t, FW, FH, mode).then_some(i)).collect()
    }

    /// Pan offsets advancing 1-3 px per frame, varying pseudo-randomly.
    fn pan_offsets(seed: u64, n: usize) -> Vec<usize> {
        let mut rng = Rng(seed);
        let mut x = 0;
        (0..n)
            .map(|_| {
                x += 1 + (rng.next() % 3) as usize;
                x
            })
            .collect()
    }

    #[test]
    fn running_baseline_never_cuts_a_steady_smooth_pan() {
        let mut det = SceneCutDetector::default();
        let frames = pan_offsets(7, 2000).into_iter().map(|x| smooth_view(0.0, 120.0, x));
        assert_eq!(cuts(&mut det, frames, RUNNING_BASELINE), Vec::<usize>::new());
    }

    #[test]
    fn running_baseline_never_cuts_a_detailed_pan_that_the_fixed_rule_cuts_every_frame() {
        let world_w = TW + 3 * 600 + 8;
        let world = noisy_world(11, world_w);
        let offsets = pan_offsets(3, 600);
        let frames: Vec<Vec<u8>> = offsets.iter().map(|&x| noisy_view(&world, world_w, x)).collect();
        let mut det = SceneCutDetector::default();
        assert_eq!(cuts(&mut det, frames.clone(), RUNNING_BASELINE), Vec::<usize>::new());
        // The case the running baseline exists for: the same pan is a "cut" every frame by the
        // fixed rule (mean delta ~85 > 40).
        let mut det = SceneCutDetector::default();
        assert_eq!(cuts(&mut det, frames, FIXED).len(), 599);
    }

    #[test]
    fn a_still_frame_with_noise_never_cuts() {
        for amplitude in [4u8, 16, 64, 255] {
            let base = smooth_view(1.0, 100.0, 0);
            let mut rng = Rng(5 + u64::from(amplitude));
            let frames: Vec<Vec<u8>> = (0..2000)
                .map(|_| base.iter().map(|&v| v.saturating_add((rng.byte() as u16 * u16::from(amplitude) / 255) as u8)).collect())
                .collect();
            let mut det = SceneCutDetector::default();
            assert_eq!(cuts(&mut det, frames, RUNNING_BASELINE), Vec::<usize>::new(), "noise amplitude {amplitude}");
        }
    }

    /// A pan of `n` frames over picture `seed`/`base`, starting at offset `x0`.
    fn pan(seed: f32, base: f32, rng_seed: u64, n: usize) -> Vec<Vec<u8>> {
        pan_offsets(rng_seed, n).into_iter().map(|x| smooth_view(seed, base, x)).collect()
    }

    #[test]
    fn a_hard_cut_is_detected_within_two_frames_and_the_baseline_recovers() {
        let mut det = SceneCutDetector::default();
        // Scene A, then a cut to an unrelated scene B, B pans, then a cut to C.
        let a = pan(0.0, 70.0, 1, 300);
        let b = pan(2.3, 190.0, 2, 400);
        let c = pan(4.1, 60.0, 3, 50);
        let frames: Vec<Vec<u8>> = a.into_iter().chain(b).chain(c).collect();
        let found = cuts(&mut det, frames, RUNNING_BASELINE);
        // The first frame of B is a candidate; the second (still B, against the held A) cuts.
        assert_eq!(found, vec![301, 701]);
        let baseline = det.baseline().unwrap();
        assert!(baseline < 20.0, "baseline recovered after the cuts: {baseline}");
    }

    #[test]
    fn hard_cuts_are_always_detected_in_mode_1() {
        // Many independent cuts at varying points of varying pans.
        for trial in 0..40u64 {
            let mut det = SceneCutDetector::default();
            let len_a = 20 + (trial as usize * 37) % 200;
            let a = pan(trial as f32 * 0.7, 60.0 + (trial % 3) as f32 * 10.0, trial, len_a);
            let b = pan(trial as f32 * 1.3 + 1.0, 180.0 + (trial % 4) as f32 * 5.0, trial + 100, 20);
            let found = cuts(&mut det, a.into_iter().chain(b), RUNNING_BASELINE);
            assert!(found == vec![len_a] || found == vec![len_a + 1], "trial {trial}: cuts at {found:?}, cut at {len_a}");
        }
    }

    #[test]
    fn a_one_frame_flash_is_not_a_cut_in_mode_1() {
        let mut det = SceneCutDetector::default();
        let mut frames = pan(0.0, 70.0, 9, 100);
        frames.insert(50, vec![250; TW * TH]);
        assert_eq!(cuts(&mut det, frames, RUNNING_BASELINE), Vec::<usize>::new());
    }

    #[test]
    fn a_size_change_resets_mode_1_without_a_cut() {
        let mut det = SceneCutDetector::default();
        cuts(&mut det, pan(0.0, 70.0, 1, 50), RUNNING_BASELINE);
        assert!(!det.observe(vec![200; 32 * 18], 256, 144, RUNNING_BASELINE));
        assert_eq!(det.baseline(), None, "new dims start a fresh baseline");
        assert!(!det.observe(vec![10; 32 * 18], 256, 144, RUNNING_BASELINE), "first measured mean only initialises");
        assert_eq!(det.baseline(), Some(190.0));
    }

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    #[test]
    fn thumbnail_samples_every_eighth_pixel_per_axis() {
        assert_eq!(luma_thumbnail(&frame(16, 16, 90), 16, 16), vec![90; 4]);
        assert_eq!(luma_thumbnail(&frame(17, 9, 90), 17, 9).len(), 3 * 2);
        assert!(luma_thumbnail(&[1, 2, 3], 16, 16).is_empty(), "a short frame must not panic");
    }

    #[test]
    fn fixed_rule_on_whole_frames() {
        let a = luma_thumbnail(&frame(32, 32, 100), 32, 32);
        let b = luma_thumbnail(&frame(32, 32, 105), 32, 32);
        let c = luma_thumbnail(&frame(32, 32, 220), 32, 32);
        assert!(!is_scene_cut(&a, &a, 40), "identical frames must never be a cut");
        assert!(!is_scene_cut(&a, &b, 40), "a small uniform change must not be a cut");
        assert!(is_scene_cut(&a, &c, 40));
        assert!(!is_scene_cut(&[], &[], 40));
        assert!(!is_scene_cut(&[1, 2], &[1, 2, 3], 40));
    }

    /// The original `is_scene_cut`, verbatim, as the reference for mode 0.
    fn original(previous: &[u8], current: &[u8], threshold: u8) -> bool {
        if previous.len() != current.len() || current.is_empty() {
            return false;
        }
        let sum: u64 = previous.iter().zip(current).map(|(&a, &b)| u64::from(a.abs_diff(b))).sum();
        sum / current.len() as u64 >= u64::from(threshold)
    }

    #[test]
    fn mode_0_gives_exactly_the_original_decision() {
        let mut rng = Rng(42);
        let mut seq: Vec<Vec<u8>> = vec![
            vec![100; 16],
            vec![100; 16],
            vec![140; 16],  // mean 40: cut
            vec![101; 16],  // mean 39: not
            vec![],         // empty: not
            vec![200; 16],  // after an empty: not
            vec![10; 8],    // length change: not
            vec![60; 8],    // 50: cut
        ];
        // Mean 39.5 floors to 39: not a cut by the original integer rule.
        seq.push(vec![20; 8]);
        seq.push([59u8, 60].repeat(4));
        for _ in 0..200 {
            let len = 4 + (rng.next() % 3) as usize;
            seq.push((0..len).map(|_| rng.byte()).collect());
        }
        // Interleave mode changes: mode 1's tracking must not leak into mode 0's answer.
        let mut det = SceneCutDetector::default();
        let mut prev: Vec<u8> = Vec::new();
        let mut seen = [false; 2];
        for (i, t) in seq.into_iter().enumerate() {
            let want = original(&prev, &t, 40);
            seen[usize::from(want)] = true;
            let mode = if i % 3 == 0 { FIXED } else { 7 }; // unknown modes read as FIXED
            assert_eq!(det.observe(t.clone(), 32, 32, mode), want, "frame {i}");
            prev = t;
        }
        assert_eq!(seen, [true, true], "the cases cover both answers");
    }

    #[test]
    fn running_baseline_rule() {
        let mut r = RunningBaseline::default();
        // First mean initialises: never over (mean >= 2.5x itself only at 0, and 0 < 20).
        assert_eq!(r.observe(4.0, 40), (false, false));
        assert_eq!(r.baseline(), Some(4.0));
        // floor = 14: 2.5 * 14 = 35 is the bar while the baseline is under the floor.
        assert_eq!(r.observe(34.9, 40), (false, false));
        assert!((r.baseline().unwrap() - (4.0 + 30.9 * 0.10)).abs() < 1e-4);
        let b = r.baseline().unwrap();
        assert_eq!(r.observe(60.0, 40), (true, false));
        assert!((r.baseline().unwrap() - (b + (60.0 - b) * 0.02)).abs() < 1e-4);
        assert_eq!(r.observe(60.0, 40), (true, true));
        // The run restarted after the cut: one more candidate alone does not cut.
        assert_eq!(r.observe(60.0, 40), (true, false));
        // A large baseline still needs mean >= threshold / 2: never true below 20.
        let mut r = RunningBaseline::default();
        r.observe(0.0, 40);
        assert!(!r.observe(19.9, 40).0);
    }
}
