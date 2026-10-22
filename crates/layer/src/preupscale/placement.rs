//! Where the network runs in a frame (the `placement` setting and each pass's `pass<N>_placement`;
//! docs/NATIVE_BACKEND.md, "Where the model runs").
//!
//! There are three stages, in the frame's order: before the upscaler (the hold inside DLSS's command buffer or
//! the split in front of it, at the render size), after DLSS's output (the hold right after DLSS's output
//! kernel, at the output size, before the game's post-processing and HUD), and at present (the
//! after-the-upscaler path on the presented frame, HUD included). [`stage_of`] resolves one pass's setting for
//! what the frame offers ([`Frame`]); [`Plan::of`] does every pass and makes the result sound:
//!
//! - Passes run in order, so a pass never runs at an earlier stage than the one before it: it is moved to that
//!   pass's stage.
//! - The present stage does not run beside the holds in one frame: the network is built for one set of sizes
//!   at a time, the after-the-upscaler path takes it only while no hold has used it for a second
//!   (`native::Loader::claim_post`), and its 8-bit proxy is made after the holds' frame is presented. Passes
//!   at present after passes in a hold move to the last hold's stage (after DLSS when its output is
//!   learnable, else before the upscaler).
//!
//! Each move is logged once. Each stage then runs one contiguous group of passes with their own settings and
//! its own histories (a hold's resources per stage: `Session::res` and `Session::res_after`).

use std::collections::HashSet;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{LazyLock, Mutex};

use neural_forge_protocol::enums::placement;
use neural_forge_protocol::MAX_PASSES;

/// A stage of the frame the network can run at, in the frame's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Stage {
    /// Before the upscaler, at DLSS's input (render) size.
    Before,
    /// After DLSS's output kernel, at the output size, before the game's HUD.
    Upscaled,
    /// At present, on the presented frame (the after-the-upscaler path).
    Present,
}

impl Stage {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Stage::Before => "before the upscaler",
            Stage::Upscaled => "after DLSS",
            Stage::Present => "at present",
        }
    }
}

/// What a frame offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    /// DLSS Ray Reconstruction runs (its input is the noisy frame: nothing is held before it).
    pub rr: bool,
    /// DLSS's output image is learnt (`Tracker::dlss_output`), so the hold after it can run.
    pub output: bool,
    /// DLSS's input is identified, so a hold before the upscaler can run (outside Ray Reconstruction).
    pub hook: bool,
}

/// The stage pass setting `setting` ([`placement`]) runs at in `frame`, and whether `upscaled` fell back
/// (the output is not learnable: it goes where `auto` would).
pub(crate) fn stage_of(setting: u32, frame: Frame) -> (Stage, bool) {
    let auto = || match frame {
        Frame { rr: true, output: true, .. } => Stage::Upscaled,
        Frame { rr: true, output: false, .. } => Stage::Present,
        Frame { hook: true, .. } => Stage::Before,
        _ => Stage::Present,
    };
    match setting {
        placement::BEFORE if frame.rr || !frame.hook => (Stage::Present, false),
        placement::BEFORE => (Stage::Before, false),
        placement::UPSCALED if frame.output => (Stage::Upscaled, false),
        placement::UPSCALED => (auto(), true),
        placement::PRESENT => (Stage::Present, false),
        _ => (auto(), false),
    }
}

/// Every pass's stage for one frame ([`Plan::of`]), and what had to move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub count: usize,
    pub stages: [Stage; MAX_PASSES],
    /// Lines for the log (once each): a fallback or a move made to keep the plan sound.
    pub notes: Vec<String>,
}

impl Plan {
    /// The plan for the first `count` passes' settings `settings` in `frame` (module docs).
    pub(crate) fn of(count: usize, settings: &[u32; MAX_PASSES], frame: Frame) -> Self {
        let count = count.clamp(1, MAX_PASSES);
        let mut stages = [Stage::Present; MAX_PASSES];
        let mut notes = Vec::new();
        for k in 0..count {
            let (stage, fell_back) = stage_of(settings[k], frame);
            if fell_back {
                notes.push(format!(
                    "pass {}: DLSS's output is not known in this game (or not yet), so the placement after DLSS runs {} instead, as auto does",
                    k + 1,
                    stage.name()
                ));
            }
            stages[k] = stage;
            if k > 0 && stages[k] < stages[k - 1] {
                notes.push(format!(
                    "pass {}: asked for {}, but pass {} runs {} and the passes run in order; it runs {} too",
                    k + 1,
                    stage.name(),
                    k,
                    stages[k - 1].name(),
                    stages[k - 1].name()
                ));
                stages[k] = stages[k - 1];
            }
        }
        // The present stage does not share a frame with the holds (module docs).
        if stages[0] != Stage::Present {
            let held =
                if frame.output { Stage::Upscaled } else { stages[..count].iter().copied().filter(|s| *s != Stage::Present).max().unwrap_or(Stage::Before) };
            for (k, stage) in stages.iter_mut().enumerate().take(count) {
                if *stage == Stage::Present {
                    notes.push(format!(
                        "pass {}: at present cannot follow passes held in the game's frame (the network serves one path at a time); it runs {}",
                        k + 1,
                        held.name()
                    ));
                    *stage = held;
                }
            }
        }
        Self { count, stages, notes }
    }

    /// The passes `stage` runs, as `first..end` (empty when none).
    pub(crate) fn range(&self, stage: Stage) -> (usize, usize) {
        let first = self.stages[..self.count].iter().position(|s| *s == stage);
        match first {
            Some(f) => (f, f + self.stages[f..self.count].iter().take_while(|s| **s == stage).count()),
            None => (0, 0),
        }
    }

    pub(crate) fn has(&self, stage: Stage) -> bool {
        self.stages[..self.count].contains(&stage)
    }

    /// Every pass runs at present: the holds stand down and the after-the-upscaler path serves the frame.
    pub(crate) fn all_present(&self) -> bool {
        !self.has(Stage::Before) && !self.has(Stage::Upscaled)
    }

    /// Logs the notes not logged before.
    pub(crate) fn say(&self) {
        static SAID: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);
        for note in &self.notes {
            if super::lock(&SAID).insert(note.clone()) {
                crate::log!("[placement] {note}");
                crate::logging::flush();
            }
        }
    }
}

/// Where one hold runs and which passes, decided when it is recorded (or at the split's submit), with the
/// sizes of both holds of the frame so that each asks the network for the same set of graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Hold {
    pub stage: Stage,
    /// The passes before the upscaler and after DLSS, as `first..end`.
    pub before: (usize, usize),
    pub upscaled: (usize, usize),
    /// DLSS's input (render) size and output size, where known ((0, 0) otherwise).
    pub render: (u32, u32),
    pub output: (u32, u32),
}

impl Hold {
    /// Every pass before the upscaler, at the hold's own size: what a hold was before placements.
    pub(crate) const ALL_BEFORE: Self = Self { stage: Stage::Before, before: (0, MAX_PASSES), upscaled: (0, 0), render: (0, 0), output: (0, 0) };

    /// The hold at `stage` of `plan`, with the frame's two sizes.
    pub(crate) fn of(plan: &Plan, stage: Stage, render: (u32, u32), output: (u32, u32)) -> Self {
        Self { stage, before: plan.range(Stage::Before), upscaled: plan.range(Stage::Upscaled), render, output }
    }

    /// This hold's passes, `first..end`.
    pub(crate) fn passes(&self) -> (usize, usize) {
        match self.stage {
            Stage::Upscaled => self.upscaled,
            _ => self.before,
        }
    }
}

/// The shared memory's header, once a device of the process opened it ([`watch`]): the recording side reads
/// the placements from it ([`settings`]).
static HEADER: AtomicPtr<neural_forge_protocol::ShmHeader> = AtomicPtr::new(std::ptr::null_mut());

/// The process's channel is open at `header` (mapped for the life of the process).
pub(crate) fn watch(header: &neural_forge_protocol::ShmHeader) {
    HEADER.store(std::ptr::from_ref(header).cast_mut(), Ordering::Release);
}

/// The pass count and each pass's placement (`auto` for every pass until the channel is open).
pub(crate) fn settings() -> (usize, [u32; MAX_PASSES]) {
    let header = HEADER.load(Ordering::Acquire);
    if header.is_null() {
        return (1, [placement::AUTO; MAX_PASSES]);
    }
    // SAFETY: a header [`watch`] stored: mapped for the life of the process.
    unsafe { (*header).placements() }
}

/// The plan for `frame` with the channel's settings, its notes logged once.
pub(crate) fn current(frame: Frame) -> Plan {
    let (count, settings) = settings();
    let plan = Plan::of(count, &settings, frame);
    plan.say();
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_forge_protocol::enums::placement::*;

    /// Every placement against Ray Reconstruction running or not and DLSS's output learnable or not (with the
    /// hook before the upscaler, which exists wherever Ray Reconstruction does not run and DLSS's input is
    /// identified), and without any hook at all.
    #[test]
    fn each_placement_resolves_to_its_stage_for_what_the_frame_offers() {
        use Stage::*;
        // (setting, rr, output) -> stage, fell back
        let table = [
            (AUTO, false, false, Before, false),
            (AUTO, false, true, Before, false),
            (AUTO, true, false, Present, false),
            (AUTO, true, true, Upscaled, false),
            (BEFORE, false, false, Before, false),
            (BEFORE, false, true, Before, false),
            (BEFORE, true, false, Present, false),
            (BEFORE, true, true, Present, false),
            (UPSCALED, false, false, Before, true),
            (UPSCALED, false, true, Upscaled, false),
            (UPSCALED, true, false, Present, true),
            (UPSCALED, true, true, Upscaled, false),
            (PRESENT, false, false, Present, false),
            (PRESENT, false, true, Present, false),
            (PRESENT, true, false, Present, false),
            (PRESENT, true, true, Present, false),
        ];
        for (setting, rr, output, stage, fell_back) in table {
            let frame = Frame { rr, output, hook: !rr };
            assert_eq!(stage_of(setting, frame), (stage, fell_back), "{} with rr {rr}, output {output}", NAMES[setting as usize]);
            // One pass: the plan is the stage itself, the fallback noted.
            let plan = Plan::of(1, &[setting; MAX_PASSES], frame);
            assert_eq!(plan.stages[0], stage);
            assert_eq!(plan.notes.len(), usize::from(fell_back), "{:?}", plan.notes);
            assert_eq!(plan.all_present(), stage == Present);
        }
        // No hook before the upscaler and no output (no DLSS, no NVX): everything at present.
        for setting in [AUTO, BEFORE, UPSCALED, PRESENT] {
            assert_eq!(stage_of(setting, Frame { rr: false, output: false, hook: false }).0, Present, "{}", NAMES[setting as usize]);
        }
        // A value out of range is auto.
        assert_eq!(stage_of(17, Frame { rr: true, output: true, hook: false }), (Upscaled, false));
    }

    #[test]
    fn passes_split_into_contiguous_groups_in_the_frames_order() {
        use Stage::*;
        let sr = Frame { rr: false, output: true, hook: true };
        let mut s = [AUTO; MAX_PASSES];
        s[0] = BEFORE;
        s[1] = UPSCALED;
        let plan = Plan::of(2, &s, sr);
        assert_eq!(&plan.stages[..2], &[Before, Upscaled]);
        assert!(plan.notes.is_empty());
        assert_eq!((plan.range(Before), plan.range(Upscaled), plan.range(Present)), ((0, 1), (1, 2), (0, 0)));
        let hold = Hold::of(&plan, Upscaled, (1707, 960), (2560, 1440));
        assert_eq!(hold.passes(), (1, 2));
        assert_eq!(Hold::of(&plan, Before, (1707, 960), (2560, 1440)).passes(), (0, 1));
        // Out of order: pass 2 before the upscaler after pass 1 after DLSS runs after DLSS too.
        s[0] = UPSCALED;
        s[1] = BEFORE;
        let plan = Plan::of(2, &s, sr);
        assert_eq!(&plan.stages[..2], &[Upscaled, Upscaled]);
        assert_eq!(plan.notes.len(), 1, "{:?}", plan.notes);
        // At present after a hold: to the hold after DLSS where the output is learnt, else before the upscaler.
        s[0] = BEFORE;
        s[1] = PRESENT;
        s[2] = PRESENT;
        let plan = Plan::of(3, &s, sr);
        assert_eq!(&plan.stages[..3], &[Before, Upscaled, Upscaled]);
        assert_eq!(plan.notes.len(), 2);
        let plan = Plan::of(3, &s, Frame { output: false, ..sr });
        assert_eq!(&plan.stages[..3], &[Before, Before, Before]);
        // Pass 1 at present: every later pass follows it there (the holds stand down).
        s[0] = PRESENT;
        s[1] = BEFORE;
        let plan = Plan::of(2, &s, sr);
        assert!(plan.all_present());
        // Ray Reconstruction: before the upscaler means at present, and auto after DLSS follows it there.
        s[0] = BEFORE;
        s[1] = AUTO;
        let plan = Plan::of(2, &s, Frame { rr: true, output: true, hook: false });
        assert_eq!(&plan.stages[..2], &[Present, Present]);
    }
}
