//! When the model's temporal history no longer belongs to the frame about to be evaluated.
//!
//! The network keeps a history: the previous output, blended into the next one by the model's own
//! per-pixel weight. That history is only meaningful for the frame that directly follows it. A
//! network that sat idle (the effect switched off and on again, the layer
//! passing frames through during a loading screen, a request that failed open) still holds the
//! last picture it produced, and without a reset the first frames after the gap are blended with
//! a scene that may be minutes old.
//!
//! The reference pipeline for this model resets on exactly these events: a first frame, a
//! reset, or a pixel whose previous position the model never saw all get blend weight zero.
//! See `docs/OPENDLSS_REVIEW.md`, rows "has-history flag" and "reset on idle". The game's own reset of
//! DLSS's history, where the layer can read it ([`HistoryGap::game_reset`]), counts as one too.
//!
//! Pure bookkeeping with no Vulkan calls, used by the layer's native backend for its history images.
//! `F` is what counts as a change of input format for the slot (the extent and identification).

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
    /// A scene cut seen on a frame the network did not run on (the layer's edit cache carried the last
    /// answer to it), with its mean luma difference: the next evaluate starts afresh.
    cut: Option<u32>,
    /// The game reset DLSS's history for a frame the history has not been reset for yet: the next evaluate
    /// starts afresh.
    game: bool,
    /// The GPU already reset the history (the layer's hold guard, for a hold inside DLSS's buffer whose
    /// guessed input layout was wrong when the buffer ran): the next evaluate starts afresh.
    unsound: bool,
}

impl<F> Default for HistoryGap<F> {
    fn default() -> Self {
        Self { last: None, broken: false, format: None, format_changed: false, cut: None, game: false, unsound: false }
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
    /// The picture changed as a whole since the previous frame (the layer's scene-cut detector,
    /// `preupscale::scene_cut`): the mean absolute difference of the two frames' luma thumbnails, on
    /// 0-255. The history describes a scene no longer on screen.
    SceneCut(u32),
    /// The game reset DLSS's history for this frame (its NGX reset flag, as DLSS Super Resolution's input
    /// kernel was given it: `preupscale::reset_word`): a camera cut or a new scene the game knows of.
    GameReset,
    /// The GPU reset the history itself: the layer's hold guard found a hold inside DLSS's buffer
    /// recorded with a guessed input layout that was wrong when the buffer ran, and cleared what the
    /// holds carry (`preupscale::inline`). The next evaluate follows it.
    Unsound,
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

    /// A scene cut on a frame the history was kept through without an evaluate (the layer's cached
    /// holds), `mean` its luma difference: the next [`Self::begin`] reports [`Stale::SceneCut`]. A cut on
    /// a frame that evaluates is reset on the GPU for that frame and never comes here.
    pub fn scene_cut(&mut self, mean: u32) {
        self.cut = Some(mean);
    }

    /// The GPU's hold guard reset the history for a hold whose guessed input layout was wrong: the next
    /// [`Self::begin`] reports [`Stale::Unsound`], unless a reason above it in [`Self::pending`] does.
    pub fn unsound(&mut self) {
        self.unsound = true;
    }

    /// The game reset DLSS's history for the frame about to be held (or for one the history was kept through):
    /// the next [`Self::begin`] reports [`Stale::GameReset`], unless a reason above it in [`Self::pending`] does.
    pub fn game_reset(&mut self) {
        self.game = true;
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

    /// What [`Self::begin`] would return at `now`, without starting an evaluate: whether the history
    /// is still good for one.
    pub fn pending(&self, now: Instant) -> Option<Stale> {
        match self.last {
            None => None,
            Some(_) if self.format_changed => Some(Stale::FormatChanged),
            Some(_) if self.unsound => Some(Stale::Unsound),
            Some(_) if self.broken => Some(Stale::Skipped),
            Some(_) if self.cut.is_some() => self.cut.map(Stale::SceneCut),
            Some(_) if self.game => Some(Stale::GameReset),
            Some(t) => {
                let gap = now.saturating_duration_since(t);
                (gap > Self::MAX_GAP).then_some(Stale::Idle(gap))
            }
        }
    }

    /// A frame that kept the history in use without evaluating (the layer's edit cache: the frame got
    /// the last evaluate's correction carried to it). [`Self::MAX_GAP`] counts from it; a reset already
    /// asked for stays asked for.
    pub fn touch(&mut self, now: Instant) {
        if self.last.is_some() {
            self.last = Some(now);
        }
    }

    /// Called once per evaluate, just before it runs. Returns why the model's history must be
    /// reset for it, or `None` when the previous evaluate on this slot was the frame before.
    /// The first evaluate of a session returns `None`: a new feature already resets itself.
    pub fn begin(&mut self, now: Instant) -> Option<Stale> {
        let stale = self.pending(now);
        self.broken = false;
        self.format_changed = false;
        self.cut = None;
        self.game = false;
        self.unsound = false;
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
    fn touched_frames_keep_the_history_and_pending_does_not_consume_a_reset() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.touch(t0);
        assert_eq!(g.begin(t0), None, "a touch before the first evaluate does not start the clock");
        // Evaluates 400 ms apart with touched frames between: never idle.
        for i in 1..6u64 {
            g.touch(t0 + Duration::from_millis(400 * i - 200));
            assert_eq!(g.pending(t0 + Duration::from_millis(400 * i)), None);
            assert_eq!(g.begin(t0 + Duration::from_millis(400 * i)), None);
        }
        // Without the touches the same evaluates would reset.
        let mut h: HistoryGap = HistoryGap::default();
        h.begin(t0);
        assert!(matches!(h.pending(t0 + Duration::from_millis(600)), Some(Stale::Idle(_))));
        // A skip stays pending through touches and `pending`, and is consumed by the next begin.
        g.skipped();
        let t = t0 + Duration::from_millis(2100);
        g.touch(t);
        assert_eq!(g.pending(t), Some(Stale::Skipped));
        assert_eq!(g.pending(t), Some(Stale::Skipped));
        assert_eq!(g.begin(t), Some(Stale::Skipped));
        assert_eq!(g.pending(t), None);
    }

    #[test]
    fn a_scene_cut_resets_the_next_evaluate_once() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        // A cached frame saw the cut: the evaluate after it starts afresh, and only that one.
        g.scene_cut(87);
        g.touch(t0 + Duration::from_millis(16));
        assert_eq!(g.pending(t0 + Duration::from_millis(16)), Some(Stale::SceneCut(87)), "pending does not consume it");
        assert_eq!(g.begin(t0 + Duration::from_millis(32)), Some(Stale::SceneCut(87)));
        assert_eq!(g.begin(t0 + Duration::from_millis(48)), None);
        // With a skip as well it is one reset, reported as the skip.
        g.scene_cut(60);
        g.skipped();
        assert_eq!(g.begin(t0 + Duration::from_millis(64)), Some(Stale::Skipped));
        assert_eq!(g.begin(t0 + Duration::from_millis(80)), None);
        // Before the first evaluate a new feature resets on its own.
        let mut h: HistoryGap = HistoryGap::default();
        h.scene_cut(90);
        assert_eq!(h.begin(t0), None);
    }

    #[test]
    fn a_game_reset_resets_the_next_evaluate_once() {
        let mut g: HistoryGap = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        g.game_reset();
        assert_eq!(g.pending(t0 + Duration::from_millis(8)), Some(Stale::GameReset), "pending does not consume it");
        assert_eq!(g.begin(t0 + Duration::from_millis(16)), Some(Stale::GameReset));
        assert_eq!(g.begin(t0 + Duration::from_millis(32)), None, "once");
        // Kept through a touched (cached) frame, as a latched scene cut is.
        g.game_reset();
        g.touch(t0 + Duration::from_millis(48));
        assert_eq!(g.begin(t0 + Duration::from_millis(64)), Some(Stale::GameReset));
        // With a scene cut or a skip as well it is one reset, reported as the other.
        g.game_reset();
        g.scene_cut(70);
        assert_eq!(g.begin(t0 + Duration::from_millis(80)), Some(Stale::SceneCut(70)));
        assert_eq!(g.begin(t0 + Duration::from_millis(96)), None, "both consumed");
        g.game_reset();
        g.skipped();
        assert_eq!(g.begin(t0 + Duration::from_millis(112)), Some(Stale::Skipped));
        assert_eq!(g.begin(t0 + Duration::from_millis(128)), None);
        // Over an idle gap too: one reset, reported as the game's.
        g.game_reset();
        assert_eq!(g.begin(t0 + Duration::from_secs(2)), Some(Stale::GameReset));
        // Before the first evaluate a new feature resets on its own.
        let mut h: HistoryGap = HistoryGap::default();
        h.game_reset();
        assert_eq!(h.begin(t0), None);
        assert_eq!(h.begin(t0 + Duration::from_millis(16)), None, "and it is consumed there");
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

    /// The hold guard's reset is its own reason (not a skipped request: no frame went to DLSS untouched),
    /// pending until the next evaluate consumes it once; a format change still names itself first.
    #[test]
    fn an_unsound_hold_resets_once_with_its_own_reason() {
        let mut g: HistoryGap<u8> = HistoryGap::default();
        let t0 = Instant::now();
        g.begin(t0);
        g.unsound();
        assert_eq!(g.pending(t0), Some(Stale::Unsound));
        g.skipped();
        assert_eq!(g.begin(t0 + Duration::from_millis(16)), Some(Stale::Unsound));
        assert_eq!(g.begin(t0 + Duration::from_millis(32)), None, "once");
        g.note_format(1);
        g.note_format(2);
        g.unsound();
        assert_eq!(g.begin(t0 + Duration::from_millis(48)), Some(Stale::FormatChanged));
        assert_eq!(g.begin(t0 + Duration::from_millis(64)), None, "both consumed");
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
