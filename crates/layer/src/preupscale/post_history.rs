//! The after-the-upscaler path's history (docs/NATIVE_BACKEND.md, "History after the upscaler"): what the
//! native backend's frames there blend toward, selected for measurement by [`MOTION_ENV`].
//!
//! - `none` (the default): no history, every frame a first frame with a fixed seed, bit for bit as 3.7.0.
//! - `zero`: each pass's previous output at the same pixel (zero motion), the pre-upscaler composite's blend
//!   toward it, rejected per pixel where the frame changed there by more than [`REJECT_ENV`]'s threshold since
//!   the previous frame (`native_post_preprocess.comp`), and reset as a whole on a scene cut (the
//!   pre-upscaler detector's rule on a luma thumbnail of the 8-bit proxy: [`thumbnail`], [`is_cut`]), a size
//!   or format change, a request that did not reach the network (the effect off, the model not ready), a gap
//!   of more than [`HistoryGap::MAX_GAP`] or another network. The seed counts frames since the last reset,
//!   as before the upscaler.
//! - `flow`: as `zero`, with the history reprojected by the GPU's optical flow (`VK_NV_optical_flow`,
//!   `super::post_flow`); `zero` where the device has none.
//!
//! The idea of the per-pixel rejection comes from uzbekunknown/dlss-nr-on-intel (Apache-2.0,
//! `notes/phase72-preset-b.md`, "Flicker", and the daemon's `History` with `STALE = 0.12`); ideas only.

use std::sync::LazyLock;
use std::time::Instant;

use neural_forge_protocol::history::{HistoryGap, Stale};

/// The selector: `none`, `zero` or `flow`. Read once, logged once.
pub(crate) const MOTION_ENV: &str = "NEURAL_FORGE_POST_MOTION";

/// The per-pixel rejection threshold: the largest channel difference (0 to 1 of the 8-bit proxy's scale)
/// between a pixel and where it was in the previous frame above which it gets no history. Read once.
pub(crate) const REJECT_ENV: &str = "NEURAL_FORGE_POST_HISTORY_REJECT";

/// [`REJECT_ENV`]'s default: the reference daemon's `STALE`.
pub(crate) const REJECT: f32 = 0.12;

/// Where the history comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Motion {
    /// No history (the shipped behaviour).
    None,
    /// The previous output at the same pixel.
    Zero,
    /// The previous output where the optical flow says the pixel was.
    Flow,
}

impl Motion {
    /// The preprocess's history mode for a frame that has a history (`p.frame.z`).
    pub(crate) fn shader_mode(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Zero => 1,
            Self::Flow => 2,
        }
    }
}

/// [`MOTION_ENV`]'s value: unset or empty is `none`.
pub(crate) fn parse_motion(raw: Option<&str>) -> Result<Motion, String> {
    match raw.map(str::trim) {
        None | Some("" | "none") => Ok(Motion::None),
        Some("zero") => Ok(Motion::Zero),
        Some("flow") => Ok(Motion::Flow),
        Some(other) => Err(other.to_string()),
    }
}

/// This process's [`Motion`] ([`parse_motion`]); `none` when the value is not one of the three. Logged once
/// when it is not the default.
pub(crate) fn motion() -> Motion {
    static MOTION: LazyLock<Motion> = LazyLock::new(|| match parse_motion(neural_forge_protocol::env::var(MOTION_ENV).as_deref()) {
        Ok(Motion::None) => Motion::None,
        Ok(m) => {
            crate::log!(
                "[native] after the upscaler: history with {} ({MOTION_ENV}), rejection above {} ({REJECT_ENV})",
                if m == Motion::Zero { "zero motion" } else { "the optical flow" },
                reject()
            );
            m
        }
        Err(other) => {
            crate::log!("[native] {MOTION_ENV}={other:?} is not none, zero or flow; no history after the upscaler");
            Motion::None
        }
    });
    *MOTION
}

/// [`REJECT_ENV`]'s value: unset or empty is [`REJECT`]; else a number from 0 to 1 (1 never rejects a pixel
/// for a change, 0 rejects any change).
pub(crate) fn parse_reject(raw: Option<&str>) -> Result<f32, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(REJECT),
        Some(other) => other.parse::<f32>().ok().filter(|v| (0.0..=1.0).contains(v)).ok_or_else(|| other.to_string()),
    }
}

/// This process's rejection threshold ([`parse_reject`]); the default when the value is not one.
pub(crate) fn reject() -> f32 {
    static REJECT_ONCE: LazyLock<f32> = LazyLock::new(|| match parse_reject(neural_forge_protocol::env::var(REJECT_ENV).as_deref()) {
        Ok(v) => v,
        Err(other) => {
            crate::log!("[native] {REJECT_ENV}={other:?} is not a number from 0 to 1; the default {REJECT} stays");
            REJECT
        }
    });
    *REJECT_ONCE
}

/// The scene-cut thumbnail of an 8-bit proxy (`width` x `height` texels of four bytes, rows of `width`):
/// every [`super::scene_cut::STEP`]th pixel on each axis, the integer mean of its three colour channels
/// (the channel order does not matter), as the pre-upscaler detector's thumbnail of its encoded capture.
pub(crate) fn thumbnail(proxy: &[u8], width: u32, height: u32) -> Vec<u8> {
    let step = super::scene_cut::STEP as usize;
    let (w, h) = (width as usize, height as usize);
    let mut out = Vec::with_capacity(w.div_ceil(step) * h.div_ceil(step));
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            let at = (y * w + x) * 4;
            let t = proxy.get(at..at + 3).unwrap_or(&[0, 0, 0]);
            out.push(((u32::from(t[0]) + u32::from(t[1]) + u32::from(t[2])) / 3) as u8);
        }
    }
    out
}

/// The mean absolute difference of two thumbnails, 0-255 (rounded down); `None` when they differ in size.
pub(crate) fn mean_difference(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let sum: u64 = a.iter().zip(b).map(|(&x, &y)| u64::from(x.abs_diff(y))).sum();
    Some((sum / a.len() as u64) as u32)
}

/// A cut: the mean difference reaches `threshold` (0 never cuts), the pre-upscaler detector's rule.
pub(crate) fn is_cut(mean: u32, threshold: u32) -> bool {
    threshold > 0 && mean >= threshold
}

/// Why a frame's history was reset, for the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reset {
    /// [`HistoryGap`]'s reasons (a skipped request, a gap, a size or format change) and the cut.
    Stale(Stale),
    /// The network was rebuilt or switched.
    Network,
}

/// One pass's history on a slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PassState {
    /// Frames since the pass's last reset (the noise seed).
    seed: u32,
    /// The pass ran on the previous evaluate, so its history buffer holds that frame's output.
    valid: bool,
}

/// What a frame does with its history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Decision {
    /// Why the history was reset on this frame (`None` also on the first frame, which has none).
    pub reset: Option<Reset>,
    /// Per pass: whether it blends toward its history, and its noise seed.
    pub passes: Vec<(bool, u32)>,
}

/// One slot's history bookkeeping (no Vulkan): the reset rules and each pass's seed.
#[derive(Debug, Default)]
pub(crate) struct SlotHistory {
    gap: HistoryGap<(u32, u32, u32)>,
    thumb: Option<Vec<u8>>,
    network: Option<(u32, u64)>,
    passes: Vec<PassState>,
}

impl SlotHistory {
    /// Every request with valid dimensions: its size and format (a change resets the next frame).
    pub(crate) fn note_request(&mut self, width: u32, height: u32, format: u32) {
        self.gap.note_format((width, height, format));
    }

    /// A request that did not reach the network, or a frame that failed after [`Self::begin`]: the next
    /// frame starts afresh.
    pub(crate) fn skipped(&mut self) {
        self.gap.skipped();
        for p in &mut self.passes {
            p.valid = false;
        }
    }

    /// Before an evaluate with `passes` passes: `thumb` is the frame's [`thumbnail`], `threshold` the cut
    /// rule's, `network` the build (network, generation). Counts the seeds on, assuming the frame completes
    /// (else [`Self::skipped`]).
    pub(crate) fn begin(&mut self, now: Instant, thumb: Vec<u8>, threshold: u32, network: (u32, u64), passes: usize) -> Decision {
        let stale = self.gap.begin(now);
        let cut = self.thumb.as_deref().and_then(|last| mean_difference(last, &thumb)).filter(|&m| is_cut(m, threshold));
        self.thumb = Some(thumb);
        let switched = self.network.is_some_and(|n| n != network);
        self.network = Some(network);
        let reset = stale.map(Reset::Stale).or(switched.then_some(Reset::Network)).or(cut.map(|m| Reset::Stale(Stale::SceneCut(m))));
        // A pass that did not run on the previous frame (the Passes setting grew) has no history.
        self.passes.resize(passes, PassState::default());
        let decision = self
            .passes
            .iter_mut()
            .map(|p| {
                let has = reset.is_none() && p.valid;
                p.seed = if has { p.seed.wrapping_add(1) } else { 0 };
                p.valid = true;
                (has, p.seed)
            })
            .collect();
        Decision { reset, passes: decision }
    }
}

/// Logs a reset of the after-the-upscaler history.
pub(crate) fn note_reset(reset: Reset) {
    match reset {
        Reset::Stale(Stale::Skipped) => {
            crate::log!("[native] after the upscaler: resetting the history (requests went unanswered by the network since the last frame)")
        }
        Reset::Stale(Stale::Idle(gap)) => crate::log!("[native] after the upscaler: resetting the history ({} ms since the last frame)", gap.as_millis()),
        Reset::Stale(Stale::FormatChanged) => crate::log!("[native] after the upscaler: resetting the history (another size or format)"),
        Reset::Stale(Stale::SceneCut(mean)) => {
            crate::log!("[native] after the upscaler: resetting the history (scene cut: the picture's luma changed by {mean} of 255 on average)")
        }
        Reset::Network => crate::log!("[native] after the upscaler: resetting the history (the network was built again)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_selector_and_the_threshold_parse() {
        assert_eq!(parse_motion(None), Ok(Motion::None));
        assert_eq!(parse_motion(Some("")), Ok(Motion::None));
        assert_eq!(parse_motion(Some("none")), Ok(Motion::None));
        assert_eq!(parse_motion(Some(" zero ")), Ok(Motion::Zero));
        assert_eq!(parse_motion(Some("flow")), Ok(Motion::Flow));
        assert_eq!(parse_motion(Some("optical")), Err("optical".into()));
        assert_eq!(parse_reject(None), Ok(0.12));
        for v in ["0.06", "0.12", "0.25", "0", "1"] {
            assert_eq!(parse_reject(Some(v)), Ok(v.parse().unwrap()));
        }
        for v in ["-0.1", "1.5", "nan", "x"] {
            assert!(parse_reject(Some(v)).is_err(), "{v}");
        }
        assert_eq!((Motion::None.shader_mode(), Motion::Zero.shader_mode(), Motion::Flow.shader_mode()), (0, 1, 2));
    }

    /// The thumbnail and the cut rule are the pre-upscaler detector's: an 8-pixel step, the three channels'
    /// integer mean, a cut at a mean difference of 40.
    #[test]
    fn the_cut_test_is_the_scene_cut_detectors_rule() {
        let (w, h) = (20, 10);
        let frame = |v: [u8; 4]| -> Vec<u8> { (0..w * h).flat_map(|_| v).collect() };
        let a = thumbnail(&frame([10, 20, 30, 255]), w, h);
        assert_eq!(a, vec![20; 3 * 2], "every 8th pixel of 20 x 10: 3 x 2, the mean of r, g and b");
        let b = thumbnail(&frame([50, 60, 70, 0]), w, h);
        assert_eq!(mean_difference(&a, &b), Some(40));
        assert!(is_cut(40, 40));
        assert!(!is_cut(39, 40));
        assert!(!is_cut(255, 0), "0 is off");
        assert_eq!(mean_difference(&a, &a[1..]), None);
    }

    #[test]
    fn the_seed_counts_frames_since_a_reset_and_every_rule_resets_it() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let still = vec![100u8; 12];
        let mut h = SlotHistory::default();
        let net = (1, 1);
        h.note_request(64, 64, 1);
        let first = h.begin(ms(0), still.clone(), 40, net, 1);
        assert_eq!(first, Decision { reset: None, passes: vec![(false, 0)] }, "the first frame has no history");
        for (n, t) in [(1, 16), (2, 32), (3, 48)] {
            h.note_request(64, 64, 1);
            assert_eq!(h.begin(ms(t), still.clone(), 40, net, 1).passes, vec![(true, n)], "frame {n}");
        }
        // A cut: a frame without history, seed 0, then counting on.
        let cut = h.begin(ms(64), vec![150; 12], 40, net, 1);
        assert_eq!(cut, Decision { reset: Some(Reset::Stale(Stale::SceneCut(50))), passes: vec![(false, 0)] });
        assert_eq!(h.begin(ms(80), vec![150; 12], 40, net, 1).passes, vec![(true, 1)]);
        // Below the threshold, and with the detector off: kept.
        assert_eq!(h.begin(ms(96), vec![111; 12], 40, net, 1).passes, vec![(true, 2)]);
        assert_eq!(h.begin(ms(112), vec![0; 12], 0, net, 1).passes, vec![(true, 3)]);
        // The effect off (a request that did not reach the network).
        h.skipped();
        assert_eq!(h.begin(ms(128), vec![0; 12], 40, net, 1), Decision { reset: Some(Reset::Stale(Stale::Skipped)), passes: vec![(false, 0)] });
        // Another size, another format.
        h.note_request(64, 32, 1);
        assert_eq!(h.begin(ms(144), vec![0; 12], 40, net, 1).reset, Some(Reset::Stale(Stale::FormatChanged)));
        h.note_request(64, 32, 2);
        assert_eq!(h.begin(ms(160), vec![0; 12], 40, net, 1).reset, Some(Reset::Stale(Stale::FormatChanged)));
        // A gap of untouched frames.
        assert!(matches!(h.begin(ms(800), vec![0; 12], 40, net, 1).reset, Some(Reset::Stale(Stale::Idle(_)))));
        // Another network build.
        assert_eq!(h.begin(ms(816), vec![0; 12], 40, (1, 2), 1), Decision { reset: Some(Reset::Network), passes: vec![(false, 0)] });
        assert_eq!(h.begin(ms(832), vec![0; 12], 40, (1, 2), 1).passes, vec![(true, 1)]);
    }

    /// Every pass keeps its own history: a pass the previous frame did not run starts afresh.
    #[test]
    fn a_pass_that_did_not_run_last_frame_has_no_history() {
        let t0 = Instant::now();
        let mut h = SlotHistory::default();
        let f = vec![0u8; 4];
        h.begin(t0, f.clone(), 40, (0, 1), 1);
        assert_eq!(h.begin(t0 + Duration::from_millis(16), f.clone(), 40, (0, 1), 3).passes, vec![(true, 1), (false, 0), (false, 0)]);
        assert_eq!(h.begin(t0 + Duration::from_millis(32), f.clone(), 40, (0, 1), 2).passes, vec![(true, 2), (true, 1)]);
        // Pass 3 did not run on the previous frame: its history is two frames old.
        assert_eq!(h.begin(t0 + Duration::from_millis(48), f.clone(), 40, (0, 1), 3).passes, vec![(true, 3), (true, 2), (false, 0)]);
    }
}
