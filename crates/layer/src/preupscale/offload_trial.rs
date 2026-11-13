//! The measured choice between the second GPU and the game's own for the held placements (`super::offload_pre`, #63,
//! #70): the hold waits at its submit for the other GPU's answer to the frame before it, so once the other GPU is
//! slower than the game's frame it sets the frame rate. The present's late rule (`super::offload::LateRule`) never
//! sees that: the frames get as slow as the answers, so no answer is ever late. Whether the game's GPU does better
//! with the network on it is measured, not guessed: once a window of offloaded holds shows the game waiting for the
//! other GPU ([`Trial::offloaded`]), the model goes to the game's GPU for one trial ([`Trial::local`]); the frame
//! rate there decides, once per process ([`decide`]). A game's GPU that is no faster gives the frames back to the
//! other GPU (the network is built there again), and nothing is tried again.
//!
//! On the #63 reporter's machine (RTX 4090 with an RTX 5080 beside it, the frames crossing system memory at about
//! 3.5 GB/s): Cyberpunk 2077 with Ray Reconstruction at 2048x1536 ran at 27-36 fps with the 5080's 28.6 ms frames
//! and at its 75 fps cap with the network on the 4090 (7.2 ms); Red Dead Redemption 2 before DLSS at 1365x1024 ran
//! at 48.5 fps with the 5080's 15.1 ms frames.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The mean wait per hold at which the other GPU counts as setting the frame rate ([`Trial::offloaded`]), with
/// [`WAIT_SHARE`]: below it the waits are the jitter of a GPU that keeps up.
pub(crate) const WAIT_MIN: Duration = Duration::from_millis(2);
/// The share of the frame time the mean wait must reach as well.
pub(crate) const WAIT_SHARE: f64 = 0.10;
/// Local holds after the trial began that count for nothing: the network's build on the game's GPU, the first
/// frames of its history.
pub(crate) const LOCAL_WARM: u32 = 60;
/// Local holds measured once warm.
pub(crate) const LOCAL_WINDOW: usize = 300;
/// How much shorter the game's GPU's frames must be to keep the model there: within it the other GPU keeps it (the
/// user asked for it, and it leaves the game's GPU its time).
pub(crate) const MARGIN: f64 = 1.10;
/// A gap between two holds longer than this (a pause menu, a loading screen) is not a frame time.
const GAP: Duration = Duration::from_millis(250);

/// `NEURAL_FORGE_OFFLOAD_TRIAL`: `off` never tries the game's GPU (the second GPU keeps the model whatever the
/// frame rate); `on` tries it with `same` too (tests); unset: tried with a real second GPU, not with `same`.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let same = super::offload::Choice::current() == super::offload::Choice::Same;
        match neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TRIAL").as_deref() {
            Some("off") => false,
            Some("on") => true,
            Some(other) => {
                crate::log!("[offload] NEURAL_FORGE_OFFLOAD_TRIAL={other}: not on or off; the default holds");
                !same
            }
            None => !same,
        }
    });
    *ON
}

/// `NEURAL_FORGE_OFFLOAD_TRIAL_VERDICT=offload` (tests only): the trial's verdict is the other GPU whatever was
/// measured, so the way back to it can be run on one GPU.
fn forced_offload() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TRIAL_VERDICT").as_deref() == Some("offload"));
    *ON
}

/// What the trial found.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Verdict {
    /// The game's GPU is faster: the model stays there (why, for the log and the Status tab).
    Local(String),
    /// The other GPU is as fast or faster: the frames go back to it (why).
    Offload(String),
}

/// Where the model ends up after a trial whose offloaded frames came every `off` and local ones every `local`.
pub(crate) fn decide(off: Duration, local: Duration) -> bool {
    local.as_secs_f64() * MARGIN < off.as_secs_f64()
}

fn fps(d: Duration) -> f64 {
    if d.is_zero() {
        0.0
    } else {
        1.0 / d.as_secs_f64()
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// The median of `v` (`v` not empty).
fn median(v: &mut [Duration]) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

#[derive(Debug, Default)]
enum Phase {
    /// Offloaded, not tried yet.
    #[default]
    Watching,
    /// On the game's GPU for the trial: the offloaded frame time and wait it is measured against, and the local holds.
    Local { off: Duration, wait: Duration, warm: u32, last: Option<Instant>, intervals: Vec<Duration> },
    /// Decided for the process.
    Decided,
}

/// The process's trial ([`TRIAL`]); its own type so tests drive one of their own.
#[derive(Debug, Default)]
pub(crate) struct Trial {
    phase: Phase,
}

impl Trial {
    /// A window of offloaded holds: their median frame time `frame` and the mean wait per hold at the submit for the
    /// other GPU, `wait`. `Some(why)` starts the trial: the caller gives the other GPU up with `why`.
    pub(crate) fn offloaded(&mut self, frame: Duration, wait: Duration, enabled: bool) -> Option<String> {
        if !enabled || !matches!(self.phase, Phase::Watching) || frame.is_zero() {
            return None;
        }
        if wait < WAIT_MIN || wait.as_secs_f64() < frame.as_secs_f64() * WAIT_SHARE {
            return None;
        }
        self.phase = Phase::Local { off: frame, wait, warm: 0, last: None, intervals: Vec::with_capacity(LOCAL_WINDOW) };
        Some(format!(
            "trying the model on the game's GPU: the game waited {:.1} ms a frame for the other GPU's answer (a frame every {:.1} ms, {:.0} fps); the frame rate there decides where the model runs",
            ms(wait),
            ms(frame),
            fps(frame)
        ))
    }

    /// A hold that ran the model on the game's GPU at `now`. `Some` once the trial is decided.
    pub(crate) fn local(&mut self, now: Instant, forced_offload: bool) -> Option<Verdict> {
        let Phase::Local { off, wait, warm, last, intervals } = &mut self.phase else { return None };
        if *warm < LOCAL_WARM {
            *warm += 1;
            *last = Some(now);
            return None;
        }
        if let Some(t) = last.replace(now) {
            let d = now.saturating_duration_since(t);
            if d < GAP {
                intervals.push(d);
            }
        }
        if intervals.len() < LOCAL_WINDOW {
            return None;
        }
        let local = median(intervals);
        let (off, wait) = (*off, *wait);
        self.phase = Phase::Decided;
        let numbers = format!(
            "a frame every {:.1} ms ({:.0} fps) with the model on the game's GPU, every {:.1} ms ({:.0} fps) with it on the other GPU, which the game waited {:.1} ms a frame for",
            ms(local),
            fps(local),
            ms(off),
            fps(off),
            ms(wait)
        );
        Some(if decide(off, local) && !forced_offload {
            Verdict::Local(format!("the game's GPU is faster here ({numbers}): the model stays on it for the session"))
        } else {
            Verdict::Offload(format!("the other GPU is as fast or faster here ({numbers}): the model goes back to it"))
        })
    }

    /// The trial runs: the holds are on the game's GPU for it.
    pub(crate) fn running(&self) -> bool {
        matches!(self.phase, Phase::Local { .. })
    }
}

static TRIAL: Mutex<Trial> = Mutex::new(Trial { phase: Phase::Watching });

fn lock() -> std::sync::MutexGuard<'static, Trial> {
    TRIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// [`Trial::offloaded`] for the process.
pub(crate) fn offloaded(frame: Duration, wait: Duration) -> Option<String> {
    lock().offloaded(frame, wait, enabled())
}

/// Whether the process's trial runs ([`Trial::running`]).
pub(crate) fn running() -> bool {
    lock().running()
}

/// A hold ran the model on the game's GPU ([`Trial::local`]); a decided trial is logged and acted on: kept on the
/// game's GPU for the session, or the offload asked for again.
pub(crate) fn local_hold(now: Instant) {
    let verdict = {
        let mut t = lock();
        if !t.running() {
            return;
        }
        t.local(now, forced_offload())
    };
    match verdict {
        Some(Verdict::Local(why)) => {
            crate::log!("[offload] {why}");
            crate::logging::flush();
            super::offload::note_state(super::offload::State::FellBack(why));
        }
        Some(Verdict::Offload(why)) => {
            crate::log!("[offload] {why}");
            crate::logging::flush();
            super::offload::resume();
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: fn(u64) -> Duration = Duration::from_millis;

    /// Drives a trial to its verdict with local holds every `every`, after the warm-up.
    fn run_local(t: &mut Trial, every: Duration, forced: bool) -> Verdict {
        let mut now = Instant::now();
        for _ in 0..LOCAL_WARM {
            assert!(t.local(now, forced).is_none());
            now += every;
        }
        for i in 0..=LOCAL_WINDOW {
            if let Some(v) = t.local(now, forced) {
                assert_eq!(i, LOCAL_WINDOW - 1, "decided with the window's last interval");
                return v;
            }
            now += every;
        }
        panic!("no verdict")
    }

    #[test]
    fn a_gpu_that_keeps_up_is_not_tried() {
        let mut t = Trial::default();
        // The 5080 before DLSS in Cyberpunk 2077 at 1188x891: 9 ms frames, the game at its 75 fps cap, no wait.
        assert!(t.offloaded(MS(13), Duration::from_micros(300), true).is_none());
        // Waits of a few percent of the frame: jitter.
        assert!(t.offloaded(MS(40), MS(3), true).is_none());
        assert!(!t.running());
    }

    #[test]
    fn the_reporters_machine_keeps_the_model_on_the_game_gpu() {
        // #70: Red Dead Redemption 2 before DLSS, the 5080's 15.1 ms frames: holds every 20.6 ms, waiting ~13 ms.
        let mut t = Trial::default();
        let why = t.offloaded(Duration::from_micros(20_600), MS(13), true).expect("the trial starts");
        assert!(why.contains("13.0 ms a frame"), "{why}");
        assert!(t.running());
        // The 4090 with the network on it: about 60 fps.
        match run_local(&mut t, Duration::from_micros(16_600), false) {
            Verdict::Local(why) => assert!(why.contains("the game's GPU is faster"), "{why}"),
            v => panic!("{v:?}"),
        }
        assert!(!t.running());
        // Decided once: no second trial.
        assert!(t.offloaded(Duration::from_micros(20_600), MS(13), true).is_none());
    }

    #[test]
    fn a_slower_game_gpu_gives_the_frames_back() {
        let mut t = Trial::default();
        assert!(t.offloaded(MS(20), MS(8), true).is_some());
        // Local frames 10% shorter or less: the other GPU keeps the model.
        assert!(matches!(run_local(&mut t, MS(19), false), Verdict::Offload(_)));
        let mut t = Trial::default();
        assert!(t.offloaded(MS(20), MS(8), true).is_some());
        assert!(matches!(run_local(&mut t, MS(30), false), Verdict::Offload(_)));
    }

    #[test]
    fn the_test_switch_forces_the_way_back() {
        let mut t = Trial::default();
        assert!(t.offloaded(MS(36), MS(16), true).is_some());
        assert!(matches!(run_local(&mut t, MS(13), true), Verdict::Offload(_)));
    }

    #[test]
    fn off_never_tries() {
        let mut t = Trial::default();
        assert!(t.offloaded(MS(36), MS(16), false).is_none());
        assert!(!t.running());
    }

    #[test]
    fn pauses_and_the_warm_up_do_not_count_as_frames() {
        let mut t = Trial::default();
        assert!(t.offloaded(MS(36), MS(16), true).is_some());
        let mut now = Instant::now();
        // The build on the game's GPU: slow first holds.
        for _ in 0..LOCAL_WARM {
            assert!(t.local(now, false).is_none());
            now += MS(100);
        }
        // Holds every 13 ms with a menu's pause of 2 s in the middle.
        let mut verdict = None;
        for i in 0..400 {
            now += if i == 150 { Duration::from_secs(2) } else { MS(13) };
            if let Some(v) = t.local(now, false) {
                verdict = Some((i, v));
                break;
            }
        }
        let (i, v) = verdict.expect("decided");
        assert_eq!(i, LOCAL_WINDOW, "the pause's interval is not a frame: one more hold is needed");
        assert!(matches!(v, Verdict::Local(ref why) if why.contains("13.0 ms")), "{v:?}");
    }

    #[test]
    fn the_margin_is_ten_percent() {
        assert!(decide(MS(36), MS(13)));
        assert!(!decide(MS(20), MS(19)));
        assert!(decide(MS(20), MS(18)));
    }
}
