//! When the model's temporal history no longer belongs to the frame about to be evaluated.
//!
//! NGX keeps its own history inside each feature: the previous output, blended into the next
//! one by the model's own per-pixel weight. That history is only meaningful for the frame that
//! directly follows it. A feature that sat idle (the effect switched off and on again, the layer
//! passing frames through during a loading screen, a request that failed open) still holds the
//! last picture it produced, and without a reset the first frames after the gap are blended with
//! a scene that may be minutes old.
//!
//! The reference pipeline for this model resets on exactly these events: a first frame, a
//! reset, or a pixel whose previous position the model never saw all get blend weight zero.
//! See `docs/OPENDLSS_REVIEW.md`, rows "has-history flag" and "reset on idle".
//!
//! Pure bookkeeping with no Win32 or Vulkan calls, shared by the model server (NGX's history) and the
//! layer's native backend (its own history images). `F` is what counts as a change of input format
//! for the slot: the model server's 8-bit/HDR class, the layer's extent and identification.

use std::time::{Duration, Instant};

/// Per-slot record of when the model last evaluated, and whether a request since then went
/// without an evaluate.
#[derive(Debug)]
pub struct HistoryGap<F = ()> {
    last: Option<Instant>,
    broken: bool,
    /// The input format of the latest request on this slot, and whether it changed since the
    /// last evaluate.
    format: Option<F>,
    format_changed: bool,
}

impl<F> Default for HistoryGap<F> {
    fn default() -> Self {
        Self { last: None, broken: false, format: None, format_changed: false }
    }
}

/// Why a history reset was asked for, for the log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stale {
    /// A request on this slot reached the model server but not the model (effect off, model not
    /// ready, or a failed evaluate).
    Skipped,
    /// No evaluate for longer than [`HistoryGap::MAX_GAP`] (the layer stopped sending: pass
    /// through, warm-up, a long hitch).
    Idle(Duration),
    /// The slot's input format changed (the model server: between 8-bit and RGBA16F; the layer: another
    /// extent or another identification of DLSS's inputs): the history describes another picture.
    FormatChanged,
}

impl<F: Copy + PartialEq> HistoryGap<F> {
    /// The longest pause after which the previous answer still counts as the last frame. At
    /// 4K with the model on every 2nd frame the gap is about 100 ms, so this leaves a wide
    /// margin for ordinary frame-time spikes.
    pub const MAX_GAP: Duration = Duration::from_millis(500);

    /// A request on this slot that did not reach the model.
    pub fn skipped(&mut self) {
        self.broken = true;
    }

    /// Every request on this slot, with its proxy's format class (before deciding whether it
    /// reaches the model). A change from the previous request's class makes the next
    /// [`Self::begin`] report [`Stale::FormatChanged`].
    pub fn note_format(&mut self, class: F) {
        if self.format.is_some_and(|f| f != class) {
            self.format_changed = true;
        }
        self.format = Some(class);
    }

    /// Called once per evaluate, just before it runs. Returns why the model's history must be
    /// reset for it, or `None` when the previous evaluate on this slot was the frame before.
    /// The first evaluate of a session returns `None`: a new feature already resets itself.
    pub fn begin(&mut self, now: Instant) -> Option<Stale> {
        let stale = match self.last {
            None => None,
            Some(_) if self.format_changed => Some(Stale::FormatChanged),
            Some(_) if self.broken => Some(Stale::Skipped),
            Some(t) => {
                let gap = now.saturating_duration_since(t);
                (gap > Self::MAX_GAP).then_some(Stale::Idle(gap))
            }
        };
        self.broken = false;
        self.format_changed = false;
        self.last = Some(now);
        stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FormatClass {
        Sdr8,
        Hdr16,
    }

    #[test]
    fn first_evaluate_is_not_stale() {
        let mut g: HistoryGap = HistoryGap::default();
        assert_eq!(g.begin(Instant::now()), None);
    }

    #[test]
    fn steady_frames_keep_history() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        for i in 1..20u64 {
            // 4K at interval 2: one evaluate every ~100 ms.
            assert_eq!(g.begin(t0 + Duration::from_millis(100 * i)), None);
        }
    }

    #[test]
    fn skipped_request_resets_once() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        g.skipped();
        g.skipped();
        assert_eq!(g.begin(t0 + Duration::from_millis(16)), Some(Stale::Skipped));
        assert_eq!(g.begin(t0 + Duration::from_millis(32)), None);
    }

    #[test]
    fn skipped_before_first_evaluate_is_not_stale() {
        // Requests while the model was still building: the new feature resets on its own.
        let mut g: HistoryGap = HistoryGap::default();
        g.skipped();
        assert_eq!(g.begin(Instant::now()), None);
    }

    #[test]
    fn long_pause_resets() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        let later = t0 + HistoryGap::<()>::MAX_GAP + Duration::from_millis(1);
        assert!(matches!(g.begin(later), Some(Stale::Idle(_))));
        assert_eq!(g.begin(later + Duration::from_millis(16)), None);
    }

    #[test]
    fn pause_at_the_limit_keeps_history() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        assert_eq!(g.begin(t0 + HistoryGap::<()>::MAX_GAP), None);
    }

    #[test]
    fn format_change_resets_once() {
        use FormatClass::{Hdr16, Sdr8};
        let mut g = HistoryGap::<FormatClass>::default();
        let t0 = Instant::now();
        g.note_format(Sdr8);
        g.begin(t0);
        g.note_format(Sdr8);
        assert_eq!(g.begin(t0 + Duration::from_millis(16)), None, "same class keeps history");
        g.note_format(Hdr16);
        assert_eq!(g.begin(t0 + Duration::from_millis(32)), Some(Stale::FormatChanged));
        g.note_format(Hdr16);
        assert_eq!(g.begin(t0 + Duration::from_millis(48)), None);
        // Back again, through a request that was not evaluated: still reported as the format.
        g.note_format(Sdr8);
        g.skipped();
        assert_eq!(g.begin(t0 + Duration::from_millis(64)), Some(Stale::FormatChanged));
    }

    #[test]
    fn format_seen_before_first_evaluate_is_not_stale() {
        use FormatClass::{Hdr16, Sdr8};
        let mut g = HistoryGap::<FormatClass>::default();
        g.note_format(Sdr8);
        g.note_format(Hdr16);
        assert_eq!(g.begin(Instant::now()), None, "a new feature resets on its own");
    }
}
