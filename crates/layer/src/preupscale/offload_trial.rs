//! The measured choice between the second GPU and the game's own for the held placements (`super::offload_pre`, #63,
//! #70): the hold waits at its submit for the other GPU's answer to the frame before it, so once the other GPU is
//! slower than the game's frame it sets the frame rate. The present's late rule (`super::offload::LateRule`) never
//! sees that: the frames get as slow as the answers, so no answer is ever late. Whether the game's GPU does better
//! with the network on it is measured, not guessed: once two windows of offloaded holds in a row show the game waiting
//! for the other GPU ([`Trial::offloaded`]), or one short window shows it waiting for most of each frame
//! ([`Trial::short_window`]), the held placements go to the game's GPU for one trial ([`Trial::local`]);
//! the frame rate there decides, once per process ([`decide`]). A game's GPU that is no faster, or cannot run the
//! model, gives the frames back to the other GPU, and nothing is tried again. The trial steers the held placements
//! ([`keeps_local`], read by `offload_pre::wanted`).
//!
//! The finished frame is tried too ([`finished_present`], #63): there the present shows each answer one frame late
//! and never waits for it, so the game's wait says nothing; a second GPU that drags the whole frame down (the reporter's
//! RTX 5080 on a PCIe 4.0 x4 link, also driving the monitor: 5-20 fps against 87-98 fps with the model off, while its
//! own frames took 7-16 ms and no present waited for an answer) went unnoticed by every rule. Two agreeing windows of
//! offloaded presents start the trial whatever the wait, once per process: the server on the other GPU ends for it
//! (`native_post`, without giving the other GPU up), the network runs on the game's GPU for the trial
//! ([`finished_local`]), and the faster keeps the model. The offload's own state (`super::offload::State`) is left
//! alone either way.
//!
//! On the #63 reporter's machine (RTX 4090 with an RTX 5080 beside it, the frames crossing system memory at about
//! 3.5 GB/s): Cyberpunk 2077 with Ray Reconstruction at 2048x1536 ran at 27-36 fps with the 5080's 28.6 ms frames
//! and at its 75 fps cap with the network on the 4090 (7.2 ms); Red Dead Redemption 2 before DLSS at 1365x1024 ran
//! at 48.5 fps with the 5080's 15.1 ms frames.

use std::time::{Duration, Instant};

/// The mean wait per hold at which the other GPU counts as setting the frame rate ([`Trial::offloaded`]), with
/// [`WAIT_SHARE`]: below it the waits are the jitter of a GPU that keeps up.
pub(crate) const WAIT_MIN: Duration = Duration::from_millis(2);
/// The share of the frame time the mean wait must reach as well.
pub(crate) const WAIT_SHARE: f64 = 0.10;
/// The share of the frame time the wait reaches where the other GPU sets the frame rate outright: the trial then
/// starts on two windows in a row even when their frame times differ ([`Trial::offloaded`]).
pub(crate) const WAIT_HEAVY: f64 = 0.50;
/// Holds in a short window of offloaded holds ([`Trial::short_window`]), a third of the summaries' 600: about 6 s at
/// the #63 reporter's 36 fps, where two 600-hold windows after the first took 50 s at the earliest (1800 frames).
pub(crate) const SHORT_HOLDS: u64 = 200;
/// The share of the frame time the game waits for the other GPU in a short window where that window alone starts the
/// trial: the other GPU clearly sets the frame rate (the #63 reporter's game waited 24.1 ms of its 27.5 ms frames).
/// Below it the two-window rule decides ([`WAIT_SHARE`], [`WAIT_HEAVY`]).
pub(crate) const WAIT_CLEAR: f64 = 0.60;
/// The share of a short window's holds that must have taken the other GPU's answer for the window to count: the
/// network runs there (not still loading), so the wait is its frames'.
pub(crate) const ANSWERED: f64 = 0.90;
/// How far two windows (two offloaded ones in a row, or the local window's halves) may differ and still agree.
pub(crate) const AGREE: f64 = 0.10;
/// Local holds after the trial began that count for nothing: the network's build on the game's GPU, the first
/// frames of its history (counted from the first hold that ran the model there).
pub(crate) const LOCAL_WARM: u32 = 60;
/// Frame times measured on the game's GPU once warm; extended while its two halves disagree, up to [`LOCAL_MAX`].
pub(crate) const LOCAL_WINDOW: usize = 300;
pub(crate) const LOCAL_MAX: usize = 900;
/// How much shorter the game's GPU's frames must be to keep the model there: within it the other GPU keeps it (the
/// user asked for it, and it leaves the game's GPU its time).
pub(crate) const MARGIN: f64 = 1.10;
/// A time between two holds longer than this (a pause menu, a loading screen) is no frame time.
pub(crate) const FRAME_GAP: Duration = Duration::from_millis(250);
/// How long the game's GPU has, from the trial's start, to run the model at all (its build), counted over holds that
/// keep coming; and how many holds in a row it may then miss. Past either the other GPU gets the frames back.
pub(crate) const BUILD_BUDGET: Duration = Duration::from_secs(15);
pub(crate) const MISSES: u32 = 30;
/// Frame times of the model on the game's GPU longer than [`FRAME_GAP`] in a row (below 4 frames a second) after which
/// the trial gives the frames back: no window would ever fill. Scattered ones (pauses, a throttled window) do not.
pub(crate) const SLOW: u32 = 30;
/// Holds of the trial, whether they ran the model or not, after which it gives the frames back unmeasured: runs of
/// a few frames between misses never fill a window, nor trip [`MISSES`].
pub(crate) const MAX_HOLDS: u32 = 2 * (LOCAL_WARM + 2 * LOCAL_MAX as u32);
/// The most the pauses between holds may move the build budget's start ([`BUILD_BUDGET`]): past it wall time counts,
/// so holds that all come a quarter second apart or more cannot keep the budget open for ever.
pub(crate) const PAUSES_MAX: Duration = Duration::from_secs(60);
/// Trials cut short by a change of what is held (another size, another pass count) before the offload keeps the
/// model for good.
pub(crate) const ABORTS: u32 = 3;
/// A window of offloaded finished-frame presents ([`finished_present`]): this many frame times, or as many as came in
/// [`FINISHED_SPAN`] once there are at least [`FINISHED_MIN`] (a slow second GPU fills a window in seconds too).
pub(crate) const FINISHED_WINDOW: usize = 120;
pub(crate) const FINISHED_SPAN: Duration = Duration::from_secs(5);
pub(crate) const FINISHED_MIN: usize = 10;
/// The longest time between two offloaded presents that still counts as a frame time ([`Presents::note`]): the gap
/// limit scales with the frames measured so far (three of them, at least [`FRAME_GAP`]), up to this, so a second GPU
/// at 3 fps with jitter still fills a window.
pub(crate) const FINISHED_GAP_MAX: Duration = Duration::from_secs(1);
/// After the verdict, the side that kept the model is tried once more ([`Trial::finished_step`]'s re-trial) when its
/// frame rate falls below this share of the one measured for it, in [`DRIFT_WINDOWS`] windows in a row.
pub(crate) const DRIFT_WINDOWS: u32 = 2;
/// How long a finished-frame trial may wait for the other GPU's server to hand over ([`Trial::handed_over`]) before it
/// is given up (the server ended another way, or the game's device was made again).
pub(crate) const HANDOVER_MAX: Duration = Duration::from_secs(20);

/// `NEURAL_FORGE_OFFLOAD_TRIAL`: `off` (or `0`, `false`) never tries the game's GPU (the second GPU keeps the model
/// whatever the frame rate); `on` (`1`, `true`) tries it with `same` too (tests); unset: tried with a real second GPU,
/// not with `same`.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let same = super::offload::Choice::current() == super::offload::Choice::Same;
        let value = neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TRIAL").map(|v| v.to_ascii_lowercase());
        parse_switch(value.as_deref()).unwrap_or_else(|| {
            if let Some(other) = value {
                crate::log!("[offload] NEURAL_FORGE_OFFLOAD_TRIAL={other}: not on or off; the default holds");
            }
            !same
        })
    });
    *ON
}

/// `on`/`off` and their spellings; `None` for anything else.
fn parse_switch(v: Option<&str>) -> Option<bool> {
    match v? {
        "on" | "1" | "true" => Some(true),
        "off" | "0" | "false" => Some(false),
        _ => None,
    }
}

/// `NEURAL_FORGE_OFFLOAD_TRIAL_VERDICT=offload` (tests only): the trial's verdict is the other GPU whatever was
/// measured, so the way back to it can be run on one GPU.
fn forced_offload() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_OFFLOAD_TRIAL_VERDICT").is_some_and(|v| v.eq_ignore_ascii_case("offload")));
    *ON
}

/// Whether a hold's miss is the network's build on the game's GPU, which has the build's budget and nothing else:
/// still loading (`native::LOADING`), a load or build that failed and is tried again (`native::NOT_BUILT`: a failure
/// from before the trial is still the loader's state while it builds), the process's one network still open on
/// another device (`native::OPEN_WAIT`), or the holds recorded into DLSS's buffers before the switch not done yet with
/// what the build changes (`native::WAITING`).
fn is_loading(miss: &str) -> bool {
    miss == super::native::LOADING || miss == super::native::NOT_BUILT || blocked_by(miss).is_some()
}

/// Why the build on the game's GPU cannot begin yet for a hold's `miss`, where the wait is on something other than the
/// game's GPU: the network still open on another device (`native::OPEN_WAIT`), or the holds recorded into DLSS's
/// buffers before the switch still running (`native::WAITING`). `None` for any other miss. A build budget that runs
/// out on such a wait is no verdict on the game's GPU ([`Trial::local`] aborts, and a later window tries again).
/// (#63: the hold switched to the game's GPU right after the other GPU's holds were recorded into DLSS's buffers;
/// the next 30 holds waited for those to be done and read as 30 misses of a model that runs: "could not run the
/// model", decided for the rest of the process, at 39 fps on the other GPU where the game's ran 103.)
fn blocked_by(miss: &str) -> Option<&'static str> {
    if miss == super::native::OPEN_WAIT {
        Some("the network was still open on another device")
    } else if miss == super::native::WAITING {
        Some("the holds recorded in DLSS's buffers before it were not done")
    } else {
        None
    }
}

/// What is held, which a window's frame times belong to: the hold's extent and the frame's pass count.
pub(crate) type Key = ((u32, u32), usize);

const UNMEASURED: &str = "the game's GPU could not be measured (too few frames in a row with the model on it): it goes back to the other GPU";

/// What a local hold brought the trial.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Verdict {
    /// The game's GPU is faster: the held placements stay there for the process (the log's line, and the Status tab's).
    Local { log: String, status: String },
    /// The other GPU is as fast or faster, or the game's GPU could not run the model: the frames go back to it (why).
    Offload(String),
    /// What is held changed under the trial: the frames go back to the other GPU, and a later window may try again.
    Aborted(String),
}

/// Where the model ends up after a trial whose offloaded frames came every `off` and local ones every `local`.
pub(crate) fn decide(off: Duration, local: Duration) -> bool {
    local.as_secs_f64() * MARGIN < off.as_secs_f64()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn fps(d: Duration) -> f64 {
    if d.is_zero() {
        0.0
    } else {
        1.0 / d.as_secs_f64()
    }
}

/// The mean of `v` with its lowest and highest tenth left out (`v` sorted here); zero for none. Not the median: with
/// the edit cache a local window's frames are two kinds (the network's and the cached), and the median picks one.
pub(crate) fn trimmed_mean(v: &mut [Duration]) -> Duration {
    if v.is_empty() {
        return Duration::ZERO;
    }
    v.sort_unstable();
    let cut = v.len() / 10;
    let kept = &v[cut..v.len() - cut];
    kept.iter().sum::<Duration>() / kept.len() as u32
}

/// Whether the game waited `wait` a hold for the other GPU's answer in frames every `frame`: at least [`WAIT_MIN`] and
/// [`WAIT_SHARE`] of the frame.
fn waits(frame: Duration, wait: Duration) -> bool {
    wait >= WAIT_MIN && wait.as_secs_f64() >= frame.as_secs_f64() * WAIT_SHARE
}

/// Whether a short window's holds, `taken` of which took the other GPU's answer and `untaken` found none, count as
/// answered ([`ANSWERED`]).
pub(crate) fn answered(taken: u64, untaken: u64) -> bool {
    taken > 0 && taken as f64 >= (taken + untaken) as f64 * ANSWERED
}

/// Whether two frame times agree within [`AGREE`].
fn agree(a: Duration, b: Duration) -> bool {
    let (a, b) = (a.as_secs_f64(), b.as_secs_f64());
    a > 0.0 && b > 0.0 && (a - b).abs() <= AGREE * a.max(b)
}

#[derive(Debug, Default)]
enum Phase {
    /// Offloaded, not tried yet; the last window that showed the game waiting, if the one before this did.
    #[default]
    Watching,
    /// On the game's GPU for the trial.
    Local(Local),
    /// The held placements stay on the game's GPU for the process (the Status line).
    KeptLocal(String),
    /// Decided for the other GPU.
    Decided,
}

#[derive(Debug)]
struct Local {
    /// Started by the finished frame's presents ([`Trial::finished_window`]), not by the holds.
    finished: bool,
    key: Key,
    off: Duration,
    wait: Duration,
    began: Instant,
    /// Holds that ran the model here (the warm-up counts them), every hold of the trial, the misses in a row since the
    /// last that ran, the last hold's time, and how far the pauses moved `began` (at most [`PAUSES_MAX`]).
    ran: u32,
    holds: u32,
    missed: u32,
    /// Since when the holds miss because the network is still loading on the game's GPU ([`is_loading`]): its build's
    /// budget ([`BUILD_BUDGET`]) counts from there (from the trial's start before anything ran).
    loading: Option<Instant>,
    /// The last of the build's holds that [`Self::loading`] times waited for something other than the game's GPU
    /// ([`blocked_by`], its words): a budget that then runs out is no verdict on the game's GPU.
    blocked: Option<&'static str>,
    last: Option<Instant>,
    paused: Duration,
    /// The last hold's time if it ran the model; `None` after a miss (an interval is only between two holds that ran).
    last_ran: Option<Instant>,
    /// Intervals in a row too long to be frame times.
    slow: u32,
    intervals: Vec<Duration>,
}

impl Local {
    /// A trial of `key` begun at `now` against the other GPU's frame time `off` and wait `wait`.
    fn new(finished: bool, key: Key, off: Duration, wait: Duration, now: Instant) -> Self {
        Self {
            finished,
            key,
            off,
            wait,
            began: now,
            ran: 0,
            holds: 0,
            missed: 0,
            loading: None,
            blocked: None,
            last: None,
            paused: Duration::ZERO,
            last_ran: None,
            slow: 0,
            intervals: Vec::with_capacity(LOCAL_MAX),
        }
    }
}

/// The process's trial ([`TRIAL`]); its own type so tests drive one of their own.
#[derive(Debug, Default)]
pub(crate) struct Trial {
    phase: Phase,
    /// The previous offloaded window that showed the game waiting (its key and frame time): a trial starts on two in
    /// a row that agree.
    waiting: Option<(Key, Duration)>,
    /// Whether the game waited for the other GPU for half its frame or more in that window ([`WAIT_HEAVY`]).
    waiting_heavy: bool,
    /// The key of the previous short window if the other GPU answered its holds ([`Trial::short_window`]): a short
    /// window counts only after one that was answered too, so the network's first frames there are left out.
    answered: Option<Key>,
    /// The last window showed the game waiting for the other GPU and no trial has started yet: the Status tab says the
    /// trial is measuring ([`Trial::pending`]).
    pending: bool,
    aborts: u32,
    /// The trial was started by the finished frame ([`Trial::finished_window`]): its local frames are the presents'.
    finished: bool,
    /// The finished frame's trial ended with the model back on the other GPU, not yet acted on ([`Trial::take_back`]).
    back: bool,
    /// The window of offloaded finished-frame presents being filled.
    presents: Presents,
    /// The other GPU's server ended for the finished frame's trial ([`Trial::handed_over`]): the game's GPU's presents
    /// count from then on.
    handed: bool,
    /// The verdict's two frame times ([`Self::local`]): the game's GPU's and the other GPU's.
    measured: Option<(Duration, Duration)>,
    /// The finished frame's verdict: whether the game's GPU kept the model, and the frame times measured on the game's
    /// GPU and on the other; the windows after it in a row in which the side that kept it is no longer the faster
    /// ([`Self::drifted`]); whether the one re-trial ran, and whether it runs for the game's GPU's earlier verdict
    /// (a tie then keeps it there).
    decided: Option<(bool, Duration, Duration)>,
    drift: u32,
    retried: bool,
    prefer_local: bool,
    /// Why the game's GPU's server has no answer for the finished frame's presents now, where its loader says
    /// ([`finished_present`]'s `miss`); `None`: no reason known (it answers, or is only late).
    finished_miss: Option<&'static str>,
    /// A trial was given up for want of a hand-over ([`HANDOVER_MAX`]): the other GPU's server may still end for it.
    orphan: bool,
}

/// Offloaded finished-frame presents as [`Trial::finished_window`] takes them: the key they were of, the last one
/// that showed an answer, when the window began, and the frame times so far.
#[derive(Debug, Default)]
struct Presents {
    key: Option<Key>,
    last: Option<Instant>,
    began: Option<Instant>,
    intervals: Vec<Duration>,
}

impl Presents {
    /// The effect is off or the path idle: the next present starts a new interval, and the pause is no frame time.
    fn pause(&mut self) {
        self.last = None;
    }

    /// An offloaded present of `key` at `now`, `composited` when it showed an answer; `Some` frame time
    /// ([`trimmed_mean`]) once a window is full.
    fn note(&mut self, key: Key, now: Instant, composited: bool) -> Option<Duration> {
        if self.key != Some(key) {
            *self = Self { key: Some(key), ..Self::default() };
        }
        if !composited {
            // A frame without the answer: the next interval would span it.
            self.last = None;
            return None;
        }
        let n = self.intervals.len() as u32;
        let limit = if n == 0 { FINISHED_GAP_MAX } else { (self.intervals.iter().sum::<Duration>() / n * 3).clamp(FRAME_GAP, FINISHED_GAP_MAX) };
        if let Some(d) = self.last.replace(now).map(|t| now.saturating_duration_since(t)) {
            if d < limit {
                self.intervals.push(d);
                self.began.get_or_insert(now);
            }
        }
        let n = self.intervals.len();
        let spanned = self.began.is_some_and(|b| now.saturating_duration_since(b) >= FINISHED_SPAN);
        if n >= FINISHED_WINDOW || (spanned && n >= FINISHED_MIN) {
            let frame = trimmed_mean(&mut self.intervals);
            self.intervals.clear();
            self.began = None;
            return Some(frame);
        }
        if spanned {
            // Too few frames in a row over the span (pauses, frames without the answer): start the window again.
            self.intervals.clear();
            self.began = None;
        }
        None
    }
}

impl Trial {
    /// A window of offloaded holds of `key`: their frame time `frame` ([`trimmed_mean`]) and the mean wait per hold at
    /// the submit for the other GPU, `wait`. `Some(why)` starts the trial: the caller stops its offloaded side.
    pub(crate) fn offloaded(&mut self, key: Key, frame: Duration, wait: Duration, enabled: bool, now: Instant) -> Option<String> {
        if !enabled || !matches!(self.phase, Phase::Watching) || frame.is_zero() {
            return None;
        }
        self.pending = waits(frame, wait);
        if !self.pending {
            self.waiting = None;
            self.waiting_heavy = false;
            return None;
        }
        let heavy = wait.as_secs_f64() >= frame.as_secs_f64() * WAIT_HEAVY;
        let last_heavy = std::mem::replace(&mut self.waiting_heavy, heavy);
        let before = self.waiting.replace((key, frame));
        let (_, last) = before.filter(|&(k, _)| k == key)?;
        // Two windows that agree are one scene, and its frame time is what the game's GPU is measured against. Where
        // the game waits for the other GPU for half its frame or more in both, two windows that differ start the
        // trial too, against the shorter of them (the other GPU at its best): after DLSS at 2048x1536 a second GPU
        // held the game to 37-48 fps, its two windows 21 and 27 ms apart, and no trial ever came (#63).
        if !agree(last, frame) && !(heavy && last_heavy) {
            // No start can follow from these windows: the Status tab has nothing to wait for.
            self.pending = false;
            return None;
        }
        let frame = if agree(last, frame) { frame } else { frame.min(last) };
        Some(self.start(key, frame, wait, now))
    }

    /// A short window of [`SHORT_HOLDS`] offloaded holds of `key` (its frame time `frame`, the mean wait per hold
    /// `wait`), `answered` when the other GPU answered its holds ([`answered`]). Where the game waited for the other
    /// GPU for [`WAIT_CLEAR`] of its frame or more, the window starts the trial on its own, provided the short window
    /// before it was answered too (of the same key): the #63 reporter's game waited 24.1 ms of its 27.5 ms frames, and
    /// two 600-hold windows after the first started the trial only 75 s in. A lighter wait leaves it to the two-window
    /// rule ([`Self::offloaded`]). `Some(why)` starts the trial, as there.
    pub(crate) fn short_window(&mut self, key: Key, frame: Duration, wait: Duration, answered: bool, enabled: bool, now: Instant) -> Option<String> {
        if !enabled || !matches!(self.phase, Phase::Watching) || frame.is_zero() {
            return None;
        }
        if !answered {
            self.answered = None;
            self.pending = false;
            return None;
        }
        let before = self.answered.replace(key);
        // The note shows only where a start can follow: a clear wait here, or a wait the two-window rule is measuring.
        let clear = waits(frame, wait) && wait.as_secs_f64() >= frame.as_secs_f64() * WAIT_CLEAR;
        self.pending = clear || (self.pending && waits(frame, wait));
        if before != Some(key) || !clear {
            return None;
        }
        Some(self.start(key, frame, wait, now))
    }

    /// Starts the held placements' trial of `key` against the other GPU's frames every `frame`, the game waiting `wait`
    /// a frame for them: the log's line.
    fn start(&mut self, key: Key, frame: Duration, wait: Duration, now: Instant) -> String {
        self.waiting = None;
        self.answered = None;
        self.pending = false;
        self.finished = false;
        self.phase = Phase::Local(Local::new(false, key, frame, wait, now));
        format!(
            "trying the model on the game's GPU: the game waited {:.1} ms a frame for the other GPU's answer (a frame every {:.1} ms, {:.0} fps); the frame rate there decides where the model runs",
            ms(wait),
            ms(frame),
            fps(frame)
        )
    }

    /// The game waits for the other GPU and the trial has not decided yet: it is measuring (the Status tab says so).
    pub(crate) fn pending(&self) -> bool {
        self.pending && matches!(self.phase, Phase::Watching)
    }

    /// A side of the held placements was built on the other GPU: windows from before it say nothing of its frames.
    fn side_built(&mut self) {
        self.waiting = None;
        self.waiting_heavy = false;
        self.answered = None;
        self.pending = false;
    }

    /// A window of offloaded finished-frame presents of `key`, a frame every `frame`: two in a row that agree start the
    /// trial whatever the game waited (it never waits for the answer there). `Some(why)` starts it: the caller ends the
    /// server on the other GPU, and the game's GPU runs the network from the next present ([`Self::finished_local`]).
    pub(crate) fn finished_window(&mut self, key: Key, frame: Duration, enabled: bool, now: Instant) -> Option<String> {
        if !enabled || !matches!(self.phase, Phase::Watching) || frame.is_zero() {
            return None;
        }
        let before = self.waiting.replace((key, frame));
        if !before.is_some_and(|(k, f)| k == key && agree(f, frame)) {
            return None;
        }
        self.waiting = None;
        self.finished = true;
        self.phase = Phase::Local(Local::new(true, key, frame, Duration::ZERO, now));
        Some(format!(
            "trying the model on the game's GPU: the finished frame came every {:.1} ms ({:.0} fps) with the model on the other GPU; the frame rate there decides where the model runs",
            ms(frame),
            fps(frame)
        ))
    }

    /// The finished frame's trial runs, or kept the model on the game's GPU: the finished frame's server runs there.
    pub(crate) fn finished_local(&self) -> bool {
        self.finished && self.keeps_local()
    }

    /// Whether the finished frame's trial just gave the model back to the other GPU (once: the caller ends the server
    /// on the game's GPU, and the other GPU's starts again).
    pub(crate) fn take_back(&mut self) -> bool {
        std::mem::take(&mut self.back)
    }

    /// The other GPU's server ended for the trial (`Session::ensure_post_server` saw it end): the game's GPU's presents
    /// are the trial's frames from now on.
    pub(crate) fn handed_over(&mut self, now: Instant) {
        let was = self.handed;
        self.handed = self.running();
        if !self.running() {
            // A hand-over after the trial was given up for want of one (`HANDOVER_MAX`): expected, not a failure.
            self.orphan = false;
        }
        if self.handed && !was {
            // The trial's budgets count from here: the game's GPU only now serves (its network is built from now on).
            if let Phase::Local(l) = &mut self.phase {
                l.began = now;
                l.last = None;
                l.last_ran = None;
            }
        }
    }

    /// The other GPU's server ending now is the hand-over a trial waited for (running), or one that a trial given up
    /// for want of it waited for ([`HANDOVER_MAX`]): no failure.
    pub(crate) fn expects_handover(&self) -> bool {
        self.finished_local() || self.orphan
    }

    /// The other GPU's server starts (again): a give-back not acted on is stale.
    pub(crate) fn clear_back(&mut self) {
        self.back = false;
    }

    /// The finished frame's Status line while the game's GPU runs the model for the trial or after it.
    pub(crate) fn finished_status_line(&self) -> Option<String> {
        if !self.finished {
            return None;
        }
        self.status_line()
    }

    /// A present of the finished-frame path ([`finished_present`]): `offloaded` while the other GPU's server answers,
    /// `composited` when it showed the model's answer, `live` while the effect is on and the path engaged, `gameplay`
    /// once DLSS evaluates (or the game has no DLSS). The line that started a trial, or a verdict.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finished_step(
        &mut self,
        key: Key,
        now: Instant,
        offloaded: bool,
        composited: bool,
        live: bool,
        gameplay: bool,
        enabled: bool,
        forced: bool,
    ) -> (Option<String>, Option<Verdict>) {
        if !live {
            // Effect off, or the path idle: a pause, as a gap between frames is.
            self.presents.pause();
            if let Phase::Local(l) = &mut self.phase {
                l.last_ran = None;
            }
            return (None, None);
        }
        if self.running() {
            // The other GPU's server never handed over (it ended another way, the game's device was made again): the
            // trial is given up, the model goes back to the other GPU, and a later window may try again.
            if self.finished && !self.handed {
                if let Phase::Local(l) = &self.phase {
                    if now.saturating_duration_since(l.began) > HANDOVER_MAX {
                        // The game's GPU never served it: nothing to end there, the other GPU's server starts again.
                        self.aborts += 1;
                        self.phase = if self.aborts >= ABORTS { Phase::Decided } else { Phase::Watching };
                        self.waiting = None;
                        self.back = false;
                        self.prefer_local = false;
                        self.orphan = true;
                        let why = "the trial on the game's GPU stopped: the other GPU's server did not hand over; the model goes back to it".to_string();
                        return (None, Some(Verdict::Aborted(why)));
                    }
                }
            }
            // The game's GPU's frames: only those of a finished-frame trial, once the other GPU's server is gone.
            if !self.finished || offloaded || !self.handed {
                return (None, None);
            }
            // Why the game's GPU's server did not answer, where its loader says (`native::post_miss`): the network
            // still another device's is an abort there too, and one off for the session a verdict.
            let miss = (!composited).then(|| self.finished_miss.unwrap_or("no answer from the game's GPU yet"));
            let verdict = self.local(key, now, composited, miss, forced);
            if verdict.is_some() {
                self.handed = false;
                // A re-trial cut short by a change of what is held keeps its tie rule for the next try.
                if !matches!(verdict, Some(Verdict::Aborted(_))) {
                    self.prefer_local = false;
                }
                self.presents = Presents::default();
                self.drift = 0;
                self.decided = match (&self.phase, self.measured) {
                    (Phase::KeptLocal(_), Some((local, off))) => Some((true, local, off)),
                    (Phase::Decided, Some((local, off))) if matches!(verdict, Some(Verdict::Offload(_))) => Some((false, local, off)),
                    _ => None,
                };
                self.measured = None;
            }
            return (None, verdict);
        }
        if !gameplay {
            // Menus and intros say nothing of the game's frames: no window, and none from before counts.
            self.presents = Presents::default();
            if matches!(self.phase, Phase::Watching) && self.decided.is_none() {
                self.waiting = None;
            }
            return (None, None);
        }
        match self.phase {
            Phase::Watching if offloaded => {
                let started = self.presents.note(key, now, composited).and_then(|frame| self.finished_window(key, frame, enabled, now));
                if started.is_some() {
                    self.presents = Presents::default();
                }
                (started, None)
            }
            Phase::KeptLocal(_) | Phase::Decided => (self.drifted(key, now, offloaded, composited, enabled), None),
            _ => {
                self.presents = Presents::default();
                (None, None)
            }
        }
    }

    /// After the finished frame's verdict, the side that kept the model is measured again, window by window, against
    /// the frame time measured for the other side. A window counts against it only when the other side would now win
    /// by the trial's own rule: the game's GPU slower than [`MARGIN`] times the other GPU's measured time, or the other
    /// GPU slower than [`MARGIN`] times the game's GPU's ([`decide`]). After [`DRIFT_WINDOWS`] such windows in a row the
    /// other side is tried once more, and never again: a game's GPU that kept the model gives it back to the other GPU,
    /// whose windows start the trial again (a tie then keeps the game's GPU); another GPU that kept it starts the trial
    /// on the game's GPU at once. Near-tie noise, and a slowdown that leaves the kept side ahead (an alt-tab, a
    /// throttled card, a heavier scene), spend nothing.
    fn drifted(&mut self, key: Key, now: Instant, offloaded: bool, composited: bool, enabled: bool) -> Option<String> {
        let (local, local_ms, off_ms) = self.decided?;
        if self.retried || !enabled || offloaded == local {
            self.presents = Presents::default();
            return None;
        }
        let window = self.presents.note(key, now, composited)?;
        let lost = if local { window.as_secs_f64() > off_ms.as_secs_f64() * MARGIN } else { decide(window, local_ms) };
        self.drift = if lost { self.drift + 1 } else { 0 };
        if self.drift < DRIFT_WINDOWS {
            return None;
        }
        self.retried = true;
        self.drift = 0;
        self.decided = None;
        self.presents = Presents::default();
        let (other, side) = if local { (off_ms, "the other GPU") } else { (local_ms, "the game's GPU") };
        let numbers = format!(
            "a frame every {:.1} ms ({:.0} fps), against {:.1} ms ({:.0} fps) measured with the model on {side}",
            ms(window),
            fps(window),
            ms(other),
            fps(other)
        );
        if local {
            self.phase = Phase::Watching;
            self.waiting = None;
            self.back = true;
            self.prefer_local = true;
            Some(format!("the game's GPU got slower with the model ({numbers}): the model goes back to the other GPU and is tried once more"))
        } else {
            self.phase = Phase::Watching;
            self.waiting = Some((key, window));
            let started = self.finished_window(key, window, enabled, now);
            started.map(|s| format!("the other GPU got slower with the model ({numbers}); {s}"))
        }
    }

    /// A hold of `key` at `now` during the trial, on the game's GPU: `ran` when the model ran on it, else `miss`, why
    /// not. `Some` once the trial is over.
    pub(crate) fn local(&mut self, key: Key, now: Instant, ran: bool, miss: Option<&str>, forced_offload: bool) -> Option<Verdict> {
        let Phase::Local(l) = &mut self.phase else { return None };
        if key != l.key {
            let why = format!(
                "the trial on the game's GPU stopped: the held frames changed ({}x{}, {} pass(es), from {}x{}, {}); the model goes back to the other GPU",
                key.0 .0, key.0 .1, key.1, l.key.0 .0, l.key.0 .1, l.key.1
            );
            self.aborts += 1;
            self.back = l.finished;
            self.handed = false;
            self.phase = if self.aborts >= ABORTS { Phase::Decided } else { Phase::Watching };
            return Some(Verdict::Aborted(why));
        }
        // A pause between holds (a menu, a loading screen) moves the budget's start: it counts holds that keep coming.
        let gap = l.last.map(|t| now.saturating_duration_since(t)).filter(|&d| d >= FRAME_GAP);
        if let Some(gap) = gap {
            let moved = gap.min(PAUSES_MAX.saturating_sub(l.paused));
            l.began += moved;
            l.paused += moved;
            // The build's budget too: a load that began before a menu has its 15 s of holds, not of wall time.
            if let Some(since) = l.loading.as_mut() {
                *since += moved;
            }
        }
        l.holds += 1;
        if l.holds > MAX_HOLDS {
            self.back = l.finished;
            self.phase = Phase::Decided;
            return Some(Verdict::Offload(UNMEASURED.into()));
        }
        if !ran {
            // A frame without the model: the next interval would span it.
            l.last_ran = None;
            // The network still loading on the game's GPU is its build, which has its budget and nothing else: such
            // holds are not misses of a model that runs. (#63: a hold recorded for the other GPU before the trial
            // reported "ran" after it began, and the 30 holds of the build's first quarter second then read as 30
            // misses: "could not run the model" 0.25 s into a load that takes one or two, the model sent straight
            // back, and the offload's side torn down and built again under it.)
            if miss == Some(super::native::DEAD) {
                // Off for the session on the game's GPU: nothing to wait for.
                self.back = l.finished;
                self.phase = Phase::Decided;
                return Some(Verdict::Offload(format!("the game's GPU could not run the model ({}): it goes back to the other GPU", super::native::DEAD)));
            }
            let failed = if miss.is_some_and(is_loading) {
                if let Some(why) = miss.and_then(blocked_by) {
                    l.blocked = Some(why);
                } else if l.blocked.take().is_some() {
                    // The network came free, and the game's GPU loads it (or fails to) from here: its own budget, and
                    // its own verdict. One early wait for the other device does not make 15 s of a build that cannot
                    // be made an abort, tried three times and blamed on the other device.
                    l.loading = Some(now);
                }
                let since = *l.loading.get_or_insert(if l.ran == 0 { l.began } else { now });
                now.saturating_duration_since(since) > BUILD_BUDGET
            } else {
                l.blocked = None;
                l.loading = None;
                l.missed += 1;
                if l.ran == 0 {
                    now.saturating_duration_since(l.began) > BUILD_BUDGET
                } else {
                    l.missed >= MISSES
                }
            };
            l.last = Some(now);
            if let Some(why) = l.blocked.filter(|_| failed) {
                // The build still waited for something other than the game's GPU when the budget ran out (the network
                // still another device's, the holds in DLSS's buffers still running): a matter of the order things
                // closed in, not of what the game's GPU can run. The frames go back, and a later window may try again.
                self.aborts += 1;
                self.back = l.finished;
                self.handed = false;
                self.phase = if self.aborts >= ABORTS { Phase::Decided } else { Phase::Watching };
                return Some(Verdict::Aborted(format!("the trial on the game's GPU stopped: {why}; the model goes back to the other GPU")));
            }
            if failed {
                self.back = l.finished;
                self.phase = Phase::Decided;
                return Some(Verdict::Offload(format!(
                    "the game's GPU could not run the model ({}): it goes back to the other GPU",
                    miss.unwrap_or("not built in time")
                )));
            }
            return None;
        }
        l.missed = 0;
        l.loading = None;
        l.blocked = None;
        l.ran += 1;
        l.last = Some(now);
        let previous = l.last_ran.replace(now);
        if l.ran <= LOCAL_WARM {
            return None;
        }
        match previous.map(|t| now.saturating_duration_since(t)) {
            Some(d) if d < FRAME_GAP => {
                l.intervals.push(d);
                l.slow = 0;
            }
            Some(_) => l.slow += 1,
            None => {}
        }
        // No window fills: the model runs below 4 frames a second here, or too few of its frames come in a row.
        if l.slow >= SLOW || l.ran > LOCAL_WARM + 2 * LOCAL_MAX as u32 {
            self.back = l.finished;
            self.phase = Phase::Decided;
            return Some(Verdict::Offload(UNMEASURED.into()));
        }
        let n = l.intervals.len();
        if n < LOCAL_WINDOW {
            return None;
        }
        // The window's halves must agree (a scene that changed under it is measured longer), up to the longest window.
        let mut first = l.intervals[..n / 2].to_vec();
        let mut second = l.intervals[n / 2..].to_vec();
        if !agree(trimmed_mean(&mut first), trimmed_mean(&mut second)) && n < LOCAL_MAX {
            return None;
        }
        let local = trimmed_mean(&mut l.intervals);
        let (off, wait, finished) = (l.off, l.wait, l.finished);
        self.measured = Some((local, off));
        let waited = if finished { String::new() } else { format!(", which the game waited {:.1} ms a frame for", ms(wait)) };
        let numbers = format!(
            "a frame every {:.1} ms ({:.0} fps) with the model on the game's GPU, every {:.1} ms ({:.0} fps) with it on the other GPU{waited}",
            ms(local),
            fps(local),
            ms(off),
            fps(off),
        );
        // A re-trial for the game's GPU's earlier verdict keeps it there unless the other GPU is now clearly faster.
        let local_wins = if std::mem::take(&mut self.prefer_local) { off.as_secs_f64() * MARGIN >= local.as_secs_f64() } else { decide(off, local) };
        if local_wins && !forced_offload {
            let status = format!("second GPU: not used: the game's GPU is faster ({:.1} ms vs {:.1} ms a frame)", ms(local), ms(off));
            self.phase = Phase::KeptLocal(status.clone());
            Some(Verdict::Local { log: format!("the game's GPU is faster here ({numbers}): the model stays on it until the game restarts"), status })
        } else {
            self.back = finished;
            self.phase = Phase::Decided;
            Some(Verdict::Offload(format!("the other GPU is as fast or faster here ({numbers}): the model goes back to it")))
        }
    }

    /// The trial runs: the held placements are on the game's GPU for it.
    pub(crate) fn running(&self) -> bool {
        matches!(self.phase, Phase::Local(_))
    }

    /// The held placements stay on the game's GPU: during the trial, and for good once it found the game's GPU faster.
    pub(crate) fn keeps_local(&self) -> bool {
        matches!(self.phase, Phase::Local(_) | Phase::KeptLocal(_))
    }

    /// The Status tab's line while the model runs on the game's GPU for the trial or after it.
    pub(crate) fn status_line(&self) -> Option<String> {
        match &self.phase {
            Phase::Local(_) => Some("second GPU: trying the game's GPU for a few seconds: the faster one keeps the model".into()),
            Phase::KeptLocal(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl Trial {
    const NEW: Self = Self {
        phase: Phase::Watching,
        waiting: None,
        waiting_heavy: false,
        answered: None,
        pending: false,
        aborts: 0,
        finished: false,
        back: false,
        presents: Presents { key: None, last: None, began: None, intervals: Vec::new() },
        handed: false,
        measured: None,
        decided: None,
        drift: 0,
        retried: false,
        prefer_local: false,
        finished_miss: None,
        orphan: false,
    };
}

/// The held placements' trial, and the finished frame's: each its own windows, verdict and Status line.
#[cfg(not(test))]
static TRIAL: std::sync::Mutex<Trial> = std::sync::Mutex::new(Trial::NEW);
#[cfg(not(test))]
static FINISHED: std::sync::Mutex<Trial> = std::sync::Mutex::new(Trial::NEW);

/// The process's trial of the held placements.
#[cfg(not(test))]
fn with<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    f(&mut TRIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// [`Trial::finished_local`] of [`FINISHED`], kept by [`with_finished`]: read by the other GPU's server every 200 µs
/// and by every present, without the lock.
#[cfg(not(test))]
static FINISHED_LOCAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The process's trial of the finished frame.
#[cfg(not(test))]
fn with_finished<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    let mut t = FINISHED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let r = f(&mut t);
    FINISHED_LOCAL.store(t.finished_local(), std::sync::atomic::Ordering::Release);
    r
}

#[cfg(test)]
thread_local! {
    /// Tests: each test thread's own trials, so a test that runs one is never reached by another test's holds
    /// (`Session::note` from tests that run in parallel).
    static TRIAL: std::cell::RefCell<Trial> = const { std::cell::RefCell::new(Trial::NEW) };
    static FINISHED: std::cell::RefCell<Trial> = const { std::cell::RefCell::new(Trial::NEW) };
}

#[cfg(test)]
fn with<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    TRIAL.with(|t| f(&mut t.borrow_mut()))
}

#[cfg(test)]
fn with_finished<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    FINISHED.with(|t| f(&mut t.borrow_mut()))
}

/// A side of the held placements was built on the other GPU: windows from before it say nothing of its frames.
pub(crate) fn side_built() {
    with(Trial::side_built);
}

/// [`Trial::short_window`] for the process.
pub(crate) fn short_window(key: Key, frame: Duration, wait: Duration, answered: bool) -> Option<String> {
    with(|t| t.short_window(key, frame, wait, answered, enabled(), Instant::now()))
}

/// The Status tab's words while the held placements' trial is pending ([`Trial::pending`]): appended to the offload's
/// line.
pub(crate) const PENDING: &str = "the game waits for it: measuring whether the game's GPU is faster";

/// [`PENDING`] while the process's trial is pending ([`Trial::pending`]).
pub(crate) fn pending_note() -> Option<&'static str> {
    with(|t| t.pending()).then_some(PENDING)
}

/// [`Trial::offloaded`] for the process.
pub(crate) fn offloaded(key: Key, frame: Duration, wait: Duration) -> Option<String> {
    with(|t| t.offloaded(key, frame, wait, enabled(), Instant::now()))
}

/// Whether the process's trial runs ([`Trial::running`]).
pub(crate) fn running() -> bool {
    with(|t| t.running())
}

/// Whether the held placements stay on the game's GPU ([`Trial::keeps_local`]).
pub(crate) fn keeps_local() -> bool {
    with(|t| t.keeps_local())
}

/// The trial's Status line ([`Trial::status_line`]).
pub(crate) fn status_line() -> Option<String> {
    with(|t| t.status_line())
}

/// A hold on the game's GPU while the trial runs ([`Trial::local`]); its verdict is logged. The held placements
/// follow [`keeps_local`] from the next hold on.
pub(crate) fn local_hold(key: Key, now: Instant, ran: bool, miss: Option<&str>) {
    let forced = forced_offload();
    // A finished-frame trial measures the presents, not the holds ([`finished_present`]).
    let verdict = with(|t| if t.running() { t.local(key, now, ran, miss, forced) } else { None });
    say(verdict);
}

/// Whether the finished frame runs on the game's GPU for the trial or after it ([`Trial::finished_local`]): the
/// finished frame's server starts there, not on the other GPU.
pub(crate) fn finished_local() -> bool {
    #[cfg(not(test))]
    return FINISHED_LOCAL.load(std::sync::atomic::Ordering::Acquire);
    #[cfg(test)]
    with_finished(|t| t.finished_local())
}

/// Whether the finished frame's trial just gave the model back to the other GPU ([`Trial::take_back`]).
pub(crate) fn take_back() -> bool {
    with_finished(Trial::take_back)
}

/// Whether a give-back of the finished frame's trial waits to be acted on (not taken).
pub(crate) fn back_pending() -> bool {
    with_finished(|t| t.back)
}

/// The other GPU's server ended for the finished frame's trial ([`Trial::handed_over`]).
pub(crate) fn handed_over() {
    with_finished(|t| t.handed_over(Instant::now()));
}

/// Whether the other GPU's server ending now is a hand-over ([`Trial::expects_handover`]).
pub(crate) fn expects_handover() -> bool {
    with_finished(|t| t.expects_handover())
}

/// The other GPU's server starts ([`Trial::clear_back`]).
pub(crate) fn clear_back() {
    with_finished(Trial::clear_back);
}

/// The finished frame's Status line while the game's GPU runs the model for the trial or after it.
pub(crate) fn finished_status_line() -> Option<String> {
    with_finished(|t| t.finished_status_line())
}

/// A present of the finished-frame path at `now`: `frame` the size of the swapchain it works on while it does (`None`:
/// the effect is off or the path idle on that swapchain, a pause), `offloaded` while the server on the other GPU
/// answers it (`super::offload::active`), `composited` when the frame showed the model's answer
/// ([`Trial::finished_step`]). Nothing without `NEURAL_FORGE_OFFLOAD_GPU`; `NEURAL_FORGE_OFFLOAD_TRIAL=off` tries
/// nothing. The caller passes `None` only for the swapchain the path works on: another window's present pauses nothing.
pub(crate) fn finished_present(frame: Option<(u32, u32)>, now: Instant, offloaded: bool, composited: bool, miss: Option<&'static str>) {
    if super::offload::Choice::current() == super::offload::Choice::Off {
        return;
    }
    // The finished frame runs the After DLSS group's passes (`Placements::after`).
    let key = (frame.unwrap_or_default(), super::placement::placements().after);
    let (enabled, forced) = (enabled(), forced_offload());
    let state = if frame.is_some() { in_gameplay(now) } else { Gameplay::No };
    let (started, verdict, waits) = with_finished(|t| {
        let gameplay = t.counts_as_gameplay(state);
        let waits = offloaded && !gameplay && matches!(t.phase, Phase::Watching) && t.decided.is_none();
        t.finished_miss = miss;
        let (started, verdict) = t.finished_step(key, now, offloaded, composited, frame.is_some(), gameplay, enabled, forced);
        (started, verdict, waits)
    });
    if waits {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[offload] the finished frame's trial waits for gameplay (DLSS running) before it measures the second GPU");
            crate::logging::flush();
        }
    }
    if let Some(line) = started {
        crate::log!("[offload] {line}");
        crate::logging::flush();
    }
    say(verdict);
}

/// How often [`in_gameplay`] reads NGX's ring again.
const GAMEPLAY_EVERY: Duration = Duration::from_millis(100);

/// Whether the game is in gameplay for the finished frame's windows ([`gameplay`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gameplay {
    /// DLSS runs, or the game has none (or never ran it past the hand-back): every window counts.
    Yes,
    /// DLSS ran and has not for [`super::PAUSE_HAND_BACK`]: switched off, or a long pause menu. Only the windows after a
    /// verdict count ([`Trial::counts_as_gameplay`]): a pause menu must never be a first trial's measurement.
    HandBack,
    /// A menu, a loading screen, an intro, or the effect off.
    No,
}

impl Trial {
    /// Whether windows count in `state`: [`Gameplay::HandBack`] only once a verdict was made (its drift windows).
    pub(crate) fn counts_as_gameplay(&self, state: Gameplay) -> bool {
        match state {
            Gameplay::Yes => true,
            Gameplay::HandBack => self.decided.is_some(),
            Gameplay::No => false,
        }
    }
}

/// [`gameplay`] now, read at most every [`GAMEPLAY_EVERY`].
fn in_gameplay(now: Instant) -> Gameplay {
    static LAST: std::sync::Mutex<Option<(Instant, Gameplay)>> = std::sync::Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((at, was)) = *last {
        if now.saturating_duration_since(at) < GAMEPLAY_EVERY {
            return was;
        }
    }
    // DLSS's launches seen without NGX's records (no wrapper) count as evaluates too.
    let evaluate = crate::ngx_ring::newest_evaluate_seen().max(super::sr_launches_last());
    let is = gameplay(evaluate, super::dlss_game::since(), now);
    *last = Some((now, is));
    is
}

/// Gameplay for the finished frame's windows: DLSS evaluates (`evaluate`, when its newest evaluate or launch was first
/// seen, within [`crate::ngx_ring::EVALUATING`]). A DLSS game that never ran DLSS (`sign`, when it was seen to be one)
/// is gameplay once the post path's own hand-back rule says so ([`super::PAUSE_HAND_BACK`] after the sign: Streamline
/// with another upscaler, Frame Generation only). Once DLSS has run, a long quiet is [`Gameplay::HandBack`]: DLSS
/// switched off counts after a verdict, but a long pause menu never starts the first trial. A game without DLSS is
/// gameplay throughout.
pub(crate) fn gameplay(evaluate: Option<Instant>, sign: Option<Instant>, now: Instant) -> Gameplay {
    let quiet_since = |t: Instant| now.saturating_duration_since(t);
    match evaluate {
        Some(e) if quiet_since(e) < crate::ngx_ring::EVALUATING => Gameplay::Yes,
        Some(e) if quiet_since(e) >= super::PAUSE_HAND_BACK => Gameplay::HandBack,
        Some(_) => Gameplay::No,
        None if sign.is_none_or(|s| quiet_since(s) >= super::PAUSE_HAND_BACK) => Gameplay::Yes,
        None => Gameplay::No,
    }
}

/// Logs a trial's verdict.
fn say(verdict: Option<Verdict>) {
    let line = match verdict {
        Some(Verdict::Local { log, .. }) => log,
        Some(Verdict::Offload(why) | Verdict::Aborted(why)) => why,
        None => return,
    };
    crate::log!("[offload] {line}");
    crate::logging::flush();
}

/// Tests: this thread's trial starts afresh.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    with(|t| *t = Trial::default());
    with_finished(|t| *t = Trial::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: fn(u64) -> Duration = Duration::from_millis;
    const KEY: Key = ((1485, 836), 1);

    /// A trial started from two agreeing windows of offloaded holds every `frame`, waiting `wait`.
    fn started(frame: Duration, wait: Duration) -> (Trial, Instant) {
        let mut t = Trial::default();
        let now = Instant::now();
        assert!(t.offloaded(KEY, frame, wait, true, now).is_none(), "one window starts nothing");
        assert!(t.offloaded(KEY, frame, wait, true, now).is_some(), "two that agree start the trial");
        assert!(t.running() && t.keeps_local());
        (t, now)
    }

    /// Drives a trial with local holds every `every` (the model running on each) to its verdict, from `now`.
    fn run_local(t: &mut Trial, mut now: Instant, every: Duration, forced: bool) -> (usize, Verdict) {
        for i in 0..2000 {
            if let Some(v) = t.local(KEY, now, true, None, forced) {
                return (i, v);
            }
            now += every;
        }
        panic!("no verdict")
    }

    #[test]
    fn a_gpu_that_keeps_up_is_not_tried() {
        let mut t = Trial::default();
        let now = Instant::now();
        // The 5080 before DLSS in Cyberpunk 2077 at 1188x891: 9 ms frames, the game at its 75 fps cap, no wait.
        for _ in 0..3 {
            assert!(t.offloaded(KEY, MS(13), Duration::from_micros(300), true, now).is_none());
        }
        // Waits of a few percent of the frame: jitter.
        for _ in 0..3 {
            assert!(t.offloaded(KEY, MS(40), MS(3), true, now).is_none());
        }
        assert!(!t.running());
    }

    #[test]
    fn two_windows_must_agree_to_start() {
        let mut t = Trial::default();
        let now = Instant::now();
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_none());
        assert!(t.offloaded(KEY, MS(30), MS(8), true, now).is_none(), "30 ms after 20 ms: not the same scene");
        assert!(t.offloaded(((2048, 1536), 1), MS(30), MS(8), true, now).is_none(), "another size");
        assert!(t.offloaded(KEY, MS(30), MS(8), true, now).is_none(), "the size changed between");
        assert!(t.offloaded(KEY, MS(31), MS(8), true, now).is_some());
    }

    #[test]
    fn the_reporters_machine_keeps_the_model_on_the_game_gpu() {
        // #70: Red Dead Redemption 2 before DLSS, the 5080's 15.1 ms frames: holds every 20.6 ms, waiting ~13 ms.
        let (mut t, now) = started(Duration::from_micros(20_600), MS(13));
        assert!(t.status_line().is_some_and(|l| l.contains("trying the game's GPU")));
        // The 4090 with the network on it: about 60 fps.
        let (i, v) = run_local(&mut t, now, Duration::from_micros(16_600), false);
        assert_eq!(i, LOCAL_WARM as usize + LOCAL_WINDOW - 1, "the warm-up, then the window");
        match v {
            Verdict::Local { log, status } => {
                assert!(log.contains("the game's GPU is faster") && log.contains("until the game restarts"), "{log}");
                assert_eq!(status, "second GPU: not used: the game's GPU is faster (16.6 ms vs 20.6 ms a frame)");
                assert!(status.len() < neural_forge_protocol::REASON_BYTES);
            }
            v => panic!("{v:?}"),
        }
        assert!(!t.running() && t.keeps_local());
        // Decided once: no second trial.
        assert!(t.offloaded(KEY, Duration::from_micros(20_600), MS(13), true, now).is_none());
        assert!(t.offloaded(KEY, Duration::from_micros(20_600), MS(13), true, now).is_none());
    }

    #[test]
    fn a_slower_game_gpu_gives_the_frames_back() {
        for local in [MS(19), MS(30)] {
            let (mut t, now) = started(MS(20), MS(8));
            assert!(matches!(run_local(&mut t, now, local, false).1, Verdict::Offload(_)), "{local:?}");
            assert!(!t.keeps_local() && t.status_line().is_none());
        }
    }

    #[test]
    fn the_test_switch_forces_the_way_back() {
        let (mut t, now) = started(MS(36), MS(16));
        assert!(matches!(run_local(&mut t, now, MS(13), true).1, Verdict::Offload(_)));
        assert!(!t.keeps_local());
    }

    #[test]
    fn off_never_tries() {
        let mut t = Trial::default();
        let now = Instant::now();
        for _ in 0..3 {
            assert!(t.offloaded(KEY, MS(36), MS(16), false, now).is_none());
        }
        assert!(!t.running());
    }

    /// After DLSS with Ray Reconstruction at 2048x1536 (#63): the game waited 20 and 27 ms of its 21 and 27 ms frames
    /// for the other GPU, and the two windows, more than a tenth apart, never started the trial. A wait of half the
    /// frame or more starts it, measured against the shorter window; a light wait still needs two that agree.
    #[test]
    fn a_game_held_up_for_most_of_its_frame_is_tried_though_its_windows_differ() {
        let now = Instant::now();
        let us = Duration::from_micros;
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, us(21_000), us(20_200), true, now).is_none(), "one window starts nothing");
        let why = t.offloaded(KEY, us(27_300), us(27_000), true, now).expect("the second, 30% longer, starts it");
        assert!(why.contains("a frame every 21.0 ms"), "against the shorter window: {why}");
        assert!(t.running());
        // Heavy in one window only (the other 8 of 20 ms): two scenes, nothing starts; a third that is heavy again does.
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_none());
        assert!(t.offloaded(KEY, us(27_300), us(27_000), true, now).is_none(), "only this window is heavy");
        assert!(!t.running());
        assert!(t.offloaded(KEY, us(21_000), us(20_200), true, now).is_some(), "two heavy windows in a row");
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, us(21_000), us(20_200), true, now).is_none());
        assert!(t.offloaded(KEY, MS(30), MS(8), true, now).is_none(), "heavy before, light now: nothing starts");
        // A light wait: windows that differ are two scenes, as before.
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_none());
        assert!(t.offloaded(KEY, MS(30), MS(8), true, now).is_none(), "8 of 30 ms: not held up for most of the frame");
        assert!(!t.running());
        // Another size between them starts nothing, however heavy.
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, MS(21), MS(20), true, now).is_none());
        assert!(t.offloaded(((2048, 1536), 1), MS(27), MS(26), true, now).is_none());
    }

    /// #63, test build 145: the trial's local phase found the process's network still open elsewhere (the
    /// after-the-upscaler path had taken it in a pause), and the loader's state, a failure from before the trial,
    /// read "could not be loaded or built" while it built: a verdict for the other GPU 0.4 s in. Both are the
    /// build's time. A network that never comes free ends the trial without a verdict, to be tried again; one that
    /// cannot be built is a verdict only after the build's budget.
    #[test]
    fn a_network_open_elsewhere_or_a_stale_failure_is_no_verdict() {
        use super::super::native::{NOT_BUILT, OPEN_WAIT};
        for miss in [OPEN_WAIT, NOT_BUILT] {
            let (mut t, mut now) = started(MS(22), MS(21));
            now += MS(3);
            assert!(t.local(KEY, now, true, None, false).is_none(), "a hold that ran");
            for k in 0..200 {
                now += MS(2);
                assert!(t.local(KEY, now, false, Some(miss), false).is_none(), "{miss}: hold {k}, 0.4 s in: no verdict");
            }
            assert!(t.running(), "{miss}: the trial goes on while the network comes");
            // It comes: the holds run and the trial measures.
            now += MS(13);
            assert!(t.local(KEY, now, true, None, false).is_none());
        }
        // Never free within the budget: aborted, not decided; the next windows may start another trial.
        let (mut t, mut now) = started(MS(22), MS(21));
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            verdict = t.local(KEY, now, false, Some(OPEN_WAIT), false);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Aborted(ref why)) if why.contains("still open on another device")), "{verdict:?}");
        assert!(!t.running() && !t.keeps_local());
        assert!(t.offloaded(KEY, MS(22), MS(21), true, now).is_none());
        assert!(t.offloaded(KEY, MS(22), MS(21), true, now).is_some(), "tried again on two more windows");
        // A network that cannot be built here: a verdict, once the build's budget is spent.
        let (mut t, mut now) = started(MS(22), MS(21));
        let began = now;
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            verdict = t.local(KEY, now, false, Some(NOT_BUILT), false);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("could not be loaded or built")), "{verdict:?}");
        assert!(now.saturating_duration_since(began) > BUILD_BUDGET, "only after the build's budget");
    }

    /// The game's GPU's holds at `now`, one every `every` that ran, until a verdict (at most the trial's longest).
    fn run_until_verdict(t: &mut Trial, now: &mut Instant, every: Duration) -> Option<Verdict> {
        for _ in 0..(LOCAL_WARM as usize + LOCAL_MAX + 10) {
            *now += every;
            if let Some(v) = t.local(KEY, *now, true, None, false) {
                return Some(v);
            }
        }
        None
    }

    /// #63, the reporter's log (RTX 4090 with an RTX 5080, Cyberpunk 2077 with Ray Reconstruction, after DLSS): the
    /// trial began right after the other GPU's holds were recorded into DLSS's buffers, and the holds on the game's GPU
    /// then waited for those to be done (`native::WAITING`). After a hold that ran, 30 of them read as 30 misses of a
    /// model that runs: "could not run the model (waiting for the holds ...)" 0.3 s in, decided for the process, and
    /// the model stayed on the 5080 at 39 fps where the 4090 had run it at 103. The wait is the build's: the trial
    /// goes on and reaches its verdict; one that never ends within the build's budget aborts and is tried again; a
    /// real failure after it still sends the model back.
    #[test]
    fn holds_still_running_in_dlss_buffers_at_the_switch_are_no_verdict() {
        use super::super::native::{DEAD, LOADING, NOT_BUILT, WAITING};
        let (mut t, mut now) = started(MS(25), MS(24));
        now += MS(3);
        assert!(t.local(KEY, now, true, None, false).is_none(), "a hold that ran");
        for k in 0..2 * MISSES {
            now += MS(10);
            assert!(t.local(KEY, now, false, Some(WAITING), false).is_none(), "hold {k} waiting for DLSS's buffers: no verdict");
        }
        for _ in 0..50 {
            now += MS(10);
            assert!(t.local(KEY, now, false, Some(LOADING), false).is_none(), "then the build");
        }
        assert!(t.running(), "the trial goes on");
        let verdict = run_until_verdict(&mut t, &mut now, Duration::from_micros(9_700));
        assert!(matches!(verdict, Some(Verdict::Local { .. })), "measured: the game's GPU is faster: {verdict:?}");
        assert!(t.keeps_local());
        // Holds that are never done within the build's budget: aborted, not decided, and the next short windows start
        // the trial again, which measures.
        let (mut t, mut now) = started(MS(25), MS(24));
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            verdict = t.local(KEY, now, false, Some(WAITING), false);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Aborted(ref why)) if why.contains("DLSS's buffers")), "{verdict:?}");
        assert!(!t.running() && !t.keeps_local());
        assert!(t.short_window(KEY, MS(25), MS(24), true, true, now).is_none());
        assert!(t.short_window(KEY, MS(25), MS(24), true, true, now).is_some(), "tried again");
        let verdict = run_until_verdict(&mut t, &mut now, Duration::from_micros(9_700));
        assert!(matches!(verdict, Some(Verdict::Local { .. })), "{verdict:?}");
        // A real failure of the game's GPU after the wait still sends the model back: a network off for the session at
        // once, one that cannot be built after its own budget, counted from the end of the wait.
        let (mut t, mut now) = started(MS(25), MS(24));
        for _ in 0..40 {
            now += MS(10);
            assert!(t.local(KEY, now, false, Some(WAITING), false).is_none());
        }
        let verdict = t.local(KEY, now + MS(10), false, Some(DEAD), false);
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("until the game restarts")), "{verdict:?}");
        let (mut t, mut now) = started(MS(25), MS(24));
        for _ in 0..40 {
            now += MS(10);
            assert!(t.local(KEY, now, false, Some(WAITING), false).is_none());
        }
        let done = now;
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            verdict = t.local(KEY, now, false, Some(NOT_BUILT), false);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("could not be loaded or built")), "{verdict:?}");
        assert!(now.saturating_duration_since(done) > BUILD_BUDGET, "after the build's own budget");
        assert!(t.short_window(KEY, MS(25), MS(24), true, true, now).is_none() && t.short_window(KEY, MS(25), MS(24), true, true, now).is_none(), "decided");
    }

    /// A network that is off for the session on the game's GPU is a verdict at once, not after the build's budget.
    #[test]
    fn a_dead_network_on_the_game_gpu_gives_the_frames_back_at_once() {
        let (mut t, mut now) = started(MS(22), MS(21));
        now += MS(3);
        let verdict = t.local(KEY, now, false, Some(super::super::native::DEAD), false);
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("until the game restarts")), "{verdict:?}");
        assert!(!t.running() && !t.keeps_local());
        assert!(t.offloaded(KEY, MS(22), MS(21), true, now).is_none() && t.offloaded(KEY, MS(22), MS(21), true, now).is_none(), "decided: not tried again");
    }

    /// The network open on another device, three trials in a row: the third abort decides for the other GPU (no
    /// trial for ever). A budget spent mostly waiting for the other device and ending on a failed build is still that
    /// wait's, not a verdict. On the finished frame's trial the abort hands the frames back (`take_back`).
    #[test]
    fn a_network_that_never_comes_free_is_tried_three_times_and_decides_nothing_about_the_gpu() {
        use super::super::native::{NOT_BUILT, OPEN_WAIT};
        let spend = |t: &mut Trial, now: &mut Instant, misses: &dyn Fn(u32) -> &'static str| -> Option<Verdict> {
            for k in 0..2000 {
                *now += MS(20);
                if let Some(v) = t.local(KEY, *now, false, Some(misses(k)), false) {
                    return Some(v);
                }
            }
            None
        };
        let (mut t, mut now) = started(MS(22), MS(21));
        for round in 1..=ABORTS {
            let verdict = spend(&mut t, &mut now, &|_| OPEN_WAIT);
            assert!(matches!(verdict, Some(Verdict::Aborted(_))), "round {round}: {verdict:?}");
            assert!(!t.running());
            let again = t.offloaded(KEY, MS(22), MS(21), true, now).or_else(|| t.offloaded(KEY, MS(22), MS(21), true, now));
            assert_eq!(again.is_some(), round < ABORTS, "round {round}: tried again until the third abort");
        }
        // The other device's for 14 s, then free, and the game's GPU cannot build it: its own budget from there (a
        // second 15 s at the most), and its own verdict, not an abort blamed on the other device.
        let (mut t, mut now) = started(MS(22), MS(21));
        let freed = now + MS(20) * 700;
        let verdict = spend(&mut t, &mut now, &|k| if k < 700 { OPEN_WAIT } else { NOT_BUILT });
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("could not be loaded or built")), "{verdict:?}");
        let built_for = now.saturating_duration_since(freed);
        assert!(built_for > BUILD_BUDGET && built_for < BUILD_BUDGET + Duration::from_secs(1), "the build's own budget: {built_for:?}");
        // One early wait, then a load that comes: no verdict from the wait, the trial measures.
        let (mut t, mut now) = started(MS(22), MS(21));
        for miss in [OPEN_WAIT, super::super::native::LOADING, super::super::native::LOADING] {
            now += MS(20);
            assert!(t.local(KEY, now, false, Some(miss), false).is_none());
        }
        now += MS(13);
        assert!(t.local(KEY, now, true, None, false).is_none() && t.running());
        // Still the other device's when the budget ends, after a load in between: aborted.
        let (mut t, mut now) = started(MS(22), MS(21));
        let verdict = spend(&mut t, &mut now, &|k| if k == 3 { NOT_BUILT } else { OPEN_WAIT });
        assert!(matches!(verdict, Some(Verdict::Aborted(_))), "{verdict:?}");
        // The finished frame's trial, through the presents (`finished_step`, with the loader's miss as
        // `finished_present` passes it): the network still the other device's is an abort there too, the game's GPU's
        // server is told to end and the other GPU's starts again; a network off for the session is a verdict at once.
        let finished = |miss: &'static str| -> (Trial, Option<Verdict>) {
            let mut t = Trial::default();
            let mut now = Instant::now();
            assert!(t.finished_window(KEY, MS(22), true, now).is_none());
            assert!(t.finished_window(KEY, MS(22), true, now).is_some(), "the finished frame's trial starts");
            t.handed_over(now);
            t.finished_miss = Some(miss);
            for _ in 0..2000 {
                now += MS(20);
                let (_, verdict) = t.finished_step(KEY, now, false, false, true, true, true, false);
                if verdict.is_some() {
                    return (t, verdict);
                }
            }
            (t, None)
        };
        let (mut t, verdict) = finished(OPEN_WAIT);
        assert!(matches!(verdict, Some(Verdict::Aborted(_))), "{verdict:?}");
        assert!(!t.finished_local(), "the finished frame is the other GPU's again");
        assert!(t.take_back(), "and the game's GPU's server is told to end, once");
        assert!(!t.take_back());
        let (t, verdict) = finished(super::super::native::DEAD);
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("until the game restarts")), "{verdict:?}");
        assert!(!t.finished_local());
    }

    /// #63's lost device began here: a hold that "ran" (recorded for the other GPU before the trial) and then the
    /// build's first holds, 30 of them in a quarter of a second, were taken for a model that stopped running. The
    /// build has its 15 s, however many holds come meanwhile, and the trial then measures.
    #[test]
    fn the_networks_build_on_the_game_gpu_is_not_thirty_misses() {
        let (mut t, mut now) = started(MS(36), MS(16));
        now += MS(3);
        assert!(t.local(KEY, now, true, None, false).is_none(), "a hold that ran");
        for k in 0..600 {
            now += MS(3);
            assert!(t.local(KEY, now, false, Some(super::super::native::LOADING), false).is_none(), "hold {k} of the build: no verdict");
        }
        assert!(t.running() && t.keeps_local(), "1.8 s into the build the trial still runs on the game's GPU");
        // Built: the holds run, and the trial measures as ever.
        now += MS(13);
        assert!(t.local(KEY, now, true, None, false).is_none());
        // A pause in the middle of a build (a menu) is not the build's time: 14 s of holds, 20 s of menu, 0.9 s more.
        let (mut t, mut now) = started(MS(36), MS(16));
        for _ in 0..700 {
            now += MS(20);
            assert!(t.local(KEY, now, false, Some(super::super::native::LOADING), false).is_none());
        }
        now += Duration::from_secs(20);
        for _ in 0..45 {
            assert!(t.local(KEY, now, false, Some(super::super::native::LOADING), false).is_none(), "the menu did not use the build's budget up");
            now += MS(20);
        }
        // A build that never ends still gives the frames back, after its budget.
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            verdict = t.local(KEY, now, false, Some(super::super::native::LOADING), false);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Offload(why)) if why.contains("still loading")), "after {BUILD_BUDGET:?} of loading");
        // Misses for another reason, once it ran, are still 30 in a row.
        let (mut t, mut now) = started(MS(36), MS(16));
        assert!(t.local(KEY, now, true, None, false).is_none());
        let mut after = 0;
        for k in 1..=MISSES {
            now += MS(13);
            if t.local(KEY, now, false, Some("the VRAM guard paused the network"), false).is_some() {
                after = k;
                break;
            }
        }
        assert_eq!(after, MISSES);
    }

    #[test]
    fn a_game_gpu_that_cannot_run_the_model_gives_the_frames_back() {
        // Never built: the budget runs out over holds that keep coming.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            if let Some(v) = t.local(KEY, now, false, Some(super::super::native::LOADING), false) {
                verdict = Some(v);
                break;
            }
        }
        match verdict.expect("decided") {
            Verdict::Offload(why) => assert!(why.contains("could not run the model (the native network is still loading)"), "{why}"),
            v => panic!("{v:?}"),
        }
        assert!(!t.keeps_local() && !t.running());
        // Built, then paused (the VRAM guard): 30 misses in a row.
        let (mut t, mut now) = started(MS(36), MS(16));
        for _ in 0..10 {
            now += MS(13);
            assert!(t.local(KEY, now, true, None, false).is_none());
        }
        for i in 1..=MISSES {
            now += MS(13);
            let v = t.local(KEY, now, false, Some("the VRAM guard paused the network"), false);
            assert_eq!(v.is_some(), i == MISSES);
        }
        assert!(!t.keeps_local());
    }

    #[test]
    fn a_pause_does_not_use_up_the_build_budget() {
        let (mut t, mut now) = started(MS(36), MS(16));
        for _ in 0..50 {
            now += MS(100);
            assert!(t.local(KEY, now, false, Some("loading"), false).is_none());
        }
        // A menu of 30 s, then the build finishes within a few holds.
        now += Duration::from_secs(30);
        assert!(t.local(KEY, now, false, Some("loading"), false).is_none());
        now += MS(100);
        assert!(t.local(KEY, now, true, None, false).is_none());
        assert!(t.running());
    }

    #[test]
    fn a_change_of_what_is_held_aborts_and_may_try_again() {
        let (mut t, now) = started(MS(36), MS(16));
        let v = t.local(((2560, 1440), 1), now, true, None, false);
        assert!(matches!(v, Some(Verdict::Aborted(_))), "{v:?}");
        assert!(!t.keeps_local());
        // Another two windows try again; after three aborts, never.
        for _ in 1..ABORTS {
            assert!(t.offloaded(KEY, MS(36), MS(16), true, now).is_none());
            assert!(t.offloaded(KEY, MS(36), MS(16), true, now).is_some());
            assert!(matches!(t.local(((1, 1), 2), now, true, None, false), Some(Verdict::Aborted(_))));
        }
        assert!(t.offloaded(KEY, MS(36), MS(16), true, now).is_none());
        assert!(t.offloaded(KEY, MS(36), MS(16), true, now).is_none());
    }

    #[test]
    fn two_kinds_of_frames_and_pauses_are_measured_fairly() {
        // The edit cache in motion: network frames of 20 ms and cached ones of 10 ms, alternating: a mean of 15 ms
        // (the median would say 10 or 20).
        let (mut t, mut now) = started(MS(20), MS(8));
        let mut verdict = None;
        for i in 0..2000u32 {
            now += if i == 200 {
                Duration::from_secs(2)
            } else if i % 2 == 0 {
                MS(20)
            } else {
                MS(10)
            };
            if let Some(v) = t.local(KEY, now, true, None, false) {
                verdict = Some(v);
                break;
            }
        }
        match verdict.expect("decided") {
            Verdict::Local { status, .. } => assert!(status.contains("(15.0 ms vs 20.0 ms"), "{status}"),
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn a_window_whose_halves_disagree_is_extended() {
        let (mut t, mut now) = started(MS(40), MS(16));
        let mut n = 0;
        let verdict = loop {
            n += 1;
            // 10 ms frames for the warm-up and the first half, then 20 ms: the halves disagree until the longest window.
            now += if n <= LOCAL_WARM as usize + LOCAL_WINDOW / 2 { MS(10) } else { MS(20) };
            if let Some(v) = t.local(KEY, now, true, None, false) {
                break v;
            }
            assert!(n < 5000);
        };
        assert!(n > LOCAL_WARM as usize + LOCAL_WINDOW + 1, "extended past the first window ({n})");
        assert!(matches!(verdict, Verdict::Local { .. }));
    }

    #[test]
    fn a_game_gpu_below_four_frames_a_second_gives_the_frames_back() {
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for _ in 0..1000 {
            now += MS(300);
            if let Some(v) = t.local(KEY, now, true, None, false) {
                verdict = Some(v);
                break;
            }
        }
        match verdict.expect("decided") {
            Verdict::Offload(why) => assert!(why.contains("could not be measured"), "{why}"),
            v => panic!("{v:?}"),
        }
        assert!(!t.running() && !t.keeps_local() && t.status_line().is_none());
    }

    #[test]
    fn runs_between_misses_end_unmeasured() {
        // One hold with the model, then 29 without, over and over: never 30 misses in a row, never a window.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for i in 0..(2 * MAX_HOLDS) {
            now += MS(10);
            if let Some(v) = t.local(KEY, now, i % 30 == 0, Some("miss"), false) {
                verdict = Some((i, v));
                break;
            }
        }
        let (i, v) = verdict.expect("decided");
        assert_eq!(i, MAX_HOLDS, "after the cap on the trial's holds");
        assert!(matches!(v, Verdict::Offload(ref why) if why.contains("could not be measured")), "{v:?}");
    }

    #[test]
    fn pauses_keep_the_build_budget_open_only_so_long() {
        // Never built, every hold a second apart: the pauses move the start by 60 s at most, then the budget runs out.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut decided_after = None;
        for i in 1..=200u32 {
            now += Duration::from_secs(1);
            if t.local(KEY, now, false, Some("loading"), false).is_some() {
                decided_after = Some(i);
                break;
            }
        }
        let n = decided_after.expect("decided");
        assert!((60..=80).contains(&n), "60 s of pauses, then 15 s of budget: {n}");
    }

    #[test]
    fn scattered_slow_frames_do_not_end_the_trial() {
        // 30 pauses spread over the window (every 10th interval 300 ms): measured, not given up.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for i in 0..5000u32 {
            now += if i > LOCAL_WARM && i % 10 == 0 { MS(300) } else { MS(13) };
            if let Some(v) = t.local(KEY, now, true, None, false) {
                verdict = Some(v);
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Local { .. })), "{verdict:?}");
    }

    #[test]
    fn frames_without_the_model_are_no_frame_times() {
        // The model on every second hold (misses between): 10 ms holds, but no interval between two that ran; the
        // measuring phase ends without a window, never with the misses' 10 ms taken for the model's frame time.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for i in 0..10_000u32 {
            now += MS(10);
            if let Some(v) = t.local(KEY, now, i % 2 == 0, Some("miss"), false) {
                verdict = Some(v);
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Offload(ref why)) if why.contains("could not be measured")), "{verdict:?}");
    }

    #[test]
    fn a_new_side_forgets_the_window_before_it() {
        let mut t = Trial::default();
        let now = Instant::now();
        assert!(t.offloaded(KEY, MS(36), MS(16), true, now).is_none());
        reset_for_test();
        with(|p| *p = std::mem::take(&mut t));
        side_built();
        assert!(offloaded(KEY, MS(36), MS(16)).is_none(), "the window before the side counts for nothing");
        assert!(offloaded(KEY, MS(36), MS(16)).is_some());
        reset_for_test();
    }

    /// #63: Cyberpunk 2077 with Ray Reconstruction, the model after DLSS on an RTX 5080 beside an RTX 4090: the game
    /// waited 24.1 ms of its 27.5 ms frames, and the trial started 75 s in. Where the game waits for most of its frame,
    /// one short window starts it, once the window before it was answered too (the network's first frames left out).
    #[test]
    fn a_game_waiting_most_of_its_frame_is_tried_after_one_short_window() {
        let now = Instant::now();
        let us = Duration::from_micros;
        let mut t = Trial::default();
        assert!(t.short_window(KEY, us(27_500), us(24_100), false, true, now).is_none(), "still loading: no answers");
        assert!(!t.pending());
        assert!(t.short_window(KEY, us(27_500), us(24_100), true, true, now).is_none(), "the first answered window is warm-up");
        assert!(t.pending(), "the Status tab says the trial measures");
        let why = t.short_window(KEY, us(27_500), us(24_100), true, true, now).expect("the next one starts it");
        assert!(why.contains("waited 24.1 ms") && why.contains("a frame every 27.5 ms"), "{why}");
        assert!(t.running() && t.keeps_local() && !t.pending());
        // Started: the long windows and the short ones start nothing more.
        assert!(t.short_window(KEY, us(27_500), us(24_100), true, true, now).is_none());
        assert!(t.offloaded(KEY, us(27_500), us(24_100), true, now).is_none());
        // The verdict as before: the 4090 at about 56 fps keeps the model.
        let (_, v) = run_local(&mut t, now, us(17_800), false);
        assert!(matches!(v, Verdict::Local { .. }), "{v:?}");
        // Exactly the share: 60% starts it, just below does not.
        let mut t = Trial::default();
        assert!(t.short_window(KEY, MS(20), MS(12), true, true, now).is_none());
        assert!(t.short_window(KEY, MS(20), MS(12), true, true, now).is_some());
    }

    /// Two long windows that disagree (and are not both heavy) can start nothing: the note must not stay.
    #[test]
    fn disagreeing_borderline_windows_clear_the_note() {
        let now = Instant::now();
        let mut t = Trial::default();
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_none());
        assert!(t.pending());
        assert!(t.offloaded(KEY, MS(26), MS(8), true, now).is_none(), "26 ms against 20 ms: not one scene");
        assert!(!t.pending(), "no start can follow");
    }

    /// A short window without the other GPU's answers leaves no note behind.
    #[test]
    fn an_unanswered_short_window_clears_the_note() {
        let us = Duration::from_micros;
        let now = Instant::now();
        let mut t = Trial::default();
        assert!(t.short_window(KEY, us(27_500), us(24_100), true, true, now).is_none());
        assert!(t.pending());
        assert!(t.short_window(KEY, us(27_500), us(24_100), false, true, now).is_none());
        assert!(!t.pending());
    }

    /// A borderline wait (a tenth to 60% of the frame) is never started by short windows: the two 600-hold windows
    /// decide, as before.
    #[test]
    fn a_borderline_wait_still_needs_two_windows() {
        let now = Instant::now();
        let mut t = Trial::default();
        for _ in 0..6 {
            assert!(t.short_window(KEY, MS(20), MS(8), true, true, now).is_none(), "8 of 20 ms: borderline");
        }
        assert!(!t.pending(), "short windows alone start nothing here: no note");
        assert!(!t.running());
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_none(), "one long window starts nothing");
        assert!(t.pending(), "it waits for a second long window: measuring");
        assert!(t.short_window(KEY, MS(20), MS(9), true, true, now).is_none());
        assert!(t.offloaded(KEY, MS(20), MS(8), true, now).is_some(), "two that agree do");
        // A heavy wait below the clear share (55%) is the two-window rule's too.
        let mut t = Trial::default();
        for _ in 0..4 {
            assert!(t.short_window(KEY, MS(20), MS(11), true, true, now).is_none());
        }
        assert!(!t.running());
        // No wait at all is not pending, and off tries nothing.
        let mut t = Trial::default();
        assert!(t.short_window(KEY, MS(13), us_300(), true, true, now).is_none());
        assert!(!t.pending());
        for _ in 0..3 {
            assert!(t.short_window(KEY, MS(36), MS(30), true, false, now).is_none());
        }
        assert!(!t.running() && !t.pending());
    }

    fn us_300() -> Duration {
        Duration::from_micros(300)
    }

    /// What resets the two-window rule resets the short one: a new side (the effect toggled, the hold built again), a
    /// window without the other GPU's answers (a pause, a load), another size, a window without the wait.
    #[test]
    fn an_interruption_resets_the_short_windows() {
        let now = Instant::now();
        let us = Duration::from_micros;
        let clear = |t: &mut Trial, answered: bool| t.short_window(KEY, us(27_500), us(24_100), answered, true, now);
        // A new side between the warm window and the clear one.
        reset_for_test();
        assert!(short_window(KEY, us(27_500), us(24_100), true).is_none());
        assert!(pending_note().is_some());
        side_built();
        assert!(pending_note().is_none(), "the new side's windows are measured afresh");
        assert!(short_window(KEY, us(27_500), us(24_100), true).is_none(), "warm-up again");
        assert!(short_window(KEY, us(27_500), us(24_100), true).is_some());
        reset_for_test();
        // A window without answers between.
        let mut t = Trial::default();
        assert!(clear(&mut t, true).is_none());
        assert!(clear(&mut t, false).is_none());
        assert!(clear(&mut t, true).is_none(), "warm-up again");
        assert!(clear(&mut t, true).is_some());
        // Another size between.
        let mut t = Trial::default();
        assert!(clear(&mut t, true).is_none());
        assert!(t.short_window(((2048, 1536), 1), us(27_500), us(24_100), true, true, now).is_none());
        assert!(clear(&mut t, true).is_none(), "the size changed between");
        assert!(clear(&mut t, true).is_some());
        // A window without the wait clears the pending line; the next clear one starts the trial (its key answered).
        let mut t = Trial::default();
        assert!(clear(&mut t, true).is_none());
        assert!(t.short_window(KEY, MS(13), us_300(), true, true, now).is_none());
        assert!(!t.pending());
        assert!(clear(&mut t, true).is_some());
        // A trial cut short by a change of what is held starts afresh: warm-up, then a clear window.
        let mut t = Trial::default();
        assert!(clear(&mut t, true).is_none());
        assert!(clear(&mut t, true).is_some());
        assert!(matches!(t.local(((2048, 1536), 1), now, true, None, false), Some(Verdict::Aborted(_))));
        assert!(!t.pending());
        assert!(clear(&mut t, true).is_none(), "warm-up again after the abort");
        assert!(clear(&mut t, true).is_some());
    }

    /// The pending note fits the Status channel whole after the offload's line, with the GPU described as the log
    /// does (Cyberpunk 2077 on an RTX 5070 cut it at "whether the gam"), and the full description stays where it fits.
    #[test]
    fn the_pending_line_fits_the_status_channel() {
        use super::super::offload_pre::Serving;
        let line_with_note = super::super::offload::line_with_note;
        for gpu in [
            "NVIDIA GeForce RTX 5070 [10de:2f04@0000:01:00.0], NVIDIA 620.70.06",
            "NVIDIA GeForce RTX 5080 Laptop GPU [10de:2c59@0000:01:00.0], NVIDIA 620.70.06",
        ] {
            for serving in [Serving { before: true, after: false }, Serving { before: false, after: true }, Serving { before: true, after: true }] {
                let line = line_with_note(gpu, serving, PENDING);
                assert!(line.len() < neural_forge_protocol::REASON_BYTES, "{} bytes: {line}", line.len());
                assert!(line.ends_with(PENDING) && line.contains(gpu.split_once(" [").unwrap().0), "{line}");
            }
        }
        let short = line_with_note("GPU [10de:2f04], NVIDIA 1", Serving { before: false, after: true }, PENDING);
        assert!(short.contains("GPU [10de:2f04], NVIDIA 1 after DLSS"), "{short}");
    }

    #[test]
    fn a_short_window_counts_as_answered_from_nine_tenths() {
        assert!(answered(90, 10) && answered(300, 0));
        assert!(!answered(89, 11) && !answered(0, 0) && !answered(0, 200));
    }

    #[test]
    fn the_switch_reads_its_spellings() {
        for (v, want) in [("on", Some(true)), ("1", Some(true)), ("true", Some(true)), ("off", Some(false)), ("0", Some(false)), ("false", Some(false))] {
            assert_eq!(parse_switch(Some(v)), want, "{v}");
        }
        assert_eq!(parse_switch(Some("maybe")), None);
        assert_eq!(parse_switch(None), None);
    }

    #[test]
    fn the_margin_is_ten_percent() {
        assert!(decide(MS(36), MS(13)));
        assert!(!decide(MS(20), MS(19)));
        assert!(decide(MS(20), MS(18)));
    }

    #[test]
    fn the_trimmed_mean_leaves_out_the_tails() {
        let mut v: Vec<Duration> = (0..10).map(|_| MS(10)).chain([MS(100)]).collect();
        assert_eq!(trimmed_mean(&mut v), MS(10));
        assert_eq!(trimmed_mean(&mut []), Duration::ZERO);
    }

    /// Presents of the finished frame from `now`, `every` apart (`jitter` alternately added and taken off), `offloaded`
    /// or not, each with the answer, live, in gameplay, until one starts the trial (the other GPU's server then hands
    /// over at once) or brings a verdict (or `n` of them): which present, and what.
    fn present_j(t: &mut Trial, now: &mut Instant, every: Duration, jitter: Duration, offloaded: bool, n: usize) -> (usize, Option<String>, Option<Verdict>) {
        for i in 0..n {
            let (started, verdict) = t.finished_step(KEY, *now, offloaded, true, true, true, true, false);
            *now += if i % 2 == 0 { every + jitter } else { every - jitter };
            if started.is_some() {
                t.handed_over(*now);
            }
            if started.is_some() || verdict.is_some() {
                return (i, started, verdict);
            }
        }
        (n, None, None)
    }

    fn present(t: &mut Trial, now: &mut Instant, every: Duration, offloaded: bool, n: usize) -> (usize, Option<String>, Option<Verdict>) {
        present_j(t, now, every, Duration::ZERO, offloaded, n)
    }

    #[test]
    fn the_finished_frame_is_tried_and_the_reporters_5_fps_go_back_to_the_game_gpu() {
        // #63: the finished frame at 5 fps with the model on the RTX 5080, and nothing ever waited for its answers.
        let mut t = Trial::default();
        let mut now = Instant::now();
        let (i, started, _) = present(&mut t, &mut now, MS(200), true, 1000);
        let started = started.expect("two windows of the same frame rate start the trial");
        assert!(started.contains("every 200.0 ms (5 fps)"), "{started}");
        assert!(i < 60, "two windows of 5 s at 5 fps, not two of 120 frames: {i}");
        assert!(t.running() && t.finished_local());
        assert_eq!(t.finished_status_line().as_deref(), Some("second GPU: trying the game's GPU for a few seconds: the faster one keeps the model"));
        // The other GPU's last answers count for nothing; the game's GPU at 80 fps keeps the model.
        assert!(present(&mut t, &mut now, MS(200), true, 50).2.is_none());
        let (_, _, verdict) = present(&mut t, &mut now, Duration::from_micros(12_500), false, 2000);
        match verdict {
            Some(Verdict::Local { log, status }) => {
                assert!(log.contains("80 fps) with the model on the game's GPU") && !log.contains("waited"), "{log}");
                assert!(status.contains("game's GPU is faster"), "{status}");
            }
            other => panic!("{other:?}"),
        }
        assert!(t.finished_local() && !t.take_back(), "kept on the game's GPU");
        assert!(t.finished_status_line().is_some_and(|l| l.starts_with("second GPU: not used")), "the Status line after the hand-over");
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_none(), "the other GPU's frames start nothing");
    }

    #[test]
    fn a_second_gpu_at_three_fps_with_jitter_is_tried() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        // 3 fps with 60 ms either way: frames 273 to 393 ms apart, past FRAME_GAP.
        let (_, started, _) = present_j(&mut t, &mut now, MS(333), MS(60), true, 200);
        assert!(started.is_some_and(|s| s.contains("(3 fps)")), "a window fills below 4 fps");
        let mut p = Presents::default();
        let at = Instant::now();
        assert!(p.note(KEY, at, true).is_none() && p.note(KEY, at + MS(900), true).is_none());
        assert_eq!(p.intervals, [MS(900)], "the first interval may be long");
        assert!(p.note(KEY, at + MS(900) + MS(2800), true).is_none());
        assert_eq!(p.intervals.len(), 1, "a pause past three frame times is no frame time");
    }

    #[test]
    fn a_second_gpu_that_keeps_up_gets_the_finished_frame_back() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(10), true, 1000).1.is_some());
        let (_, _, verdict) = present(&mut t, &mut now, MS(11), false, 3000);
        assert!(matches!(verdict, Some(Verdict::Offload(_))), "{verdict:?}");
        assert!(!t.finished_local() && t.take_back(), "the game's GPU's server ends once");
        assert!(!t.take_back());
        assert!(t.finished_status_line().is_none(), "the offload's own line again");
        assert!(present(&mut t, &mut now, MS(10), true, 1000).1.is_none(), "nothing is tried again while it keeps up");
    }

    #[test]
    fn menus_start_nothing_and_off_never_tries() {
        let mut t = Trial::default();
        let now = Instant::now();
        for i in 0..2000u64 {
            let (started, _) = t.finished_step(KEY, now + MS(10 * i), true, true, true, false, true, false);
            assert!(started.is_none(), "no DLSS evaluating: a menu or an intro");
        }
        let mut off = Trial::default();
        for i in 0..2000u64 {
            assert!(off.finished_step(KEY, now + MS(10 * i), true, true, true, true, false, false).0.is_none(), "NEURAL_FORGE_OFFLOAD_TRIAL=off");
        }
        let mut idle = Trial::default();
        for i in 0..2000u64 {
            assert!(idle.finished_step(KEY, now + MS(10 * i), false, true, true, true, true, false).0.is_none(), "the game's GPU's own frames");
        }
    }

    #[test]
    fn the_game_gpus_frames_count_only_once_the_other_gpus_server_handed_over() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        for i in 0..100_000u64 {
            let (started, _) = t.finished_step(KEY, now, true, true, true, true, true, false);
            now += MS(20);
            if started.is_some() {
                break;
            }
            assert!(i < 99_999);
        }
        for _ in 0..3000 {
            assert!(t.finished_step(KEY, now, false, true, true, true, true, false).1.is_none(), "before the hand-over");
            now += MS(5);
        }
        assert!(t.running());
        t.handed_over(now);
        assert!(present(&mut t, &mut now, MS(5), false, 3000).2.is_some_and(|v| matches!(v, Verdict::Local { .. })));
    }

    #[test]
    fn switching_the_effect_off_during_the_trial_is_a_pause_not_a_slow_frame() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(50), true, 1000).1.is_some());
        let mut verdict = None;
        for i in 0..3000 {
            // Every 40th frame starts ten seconds with the effect off (presents at 60 fps that the trial must not see).
            if i % 40 == 0 {
                for _ in 0..600 {
                    assert!(t.finished_step(KEY, now, false, false, false, true, true, false).1.is_none());
                    now += Duration::from_micros(16_667);
                }
            }
            verdict = t.finished_step(KEY, now, false, true, true, true, true, false).1;
            now += MS(10);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Local { .. })), "the game's GPU at 100 fps keeps the model: {verdict:?}");
    }

    #[test]
    fn a_stale_give_back_is_cleared_when_the_other_gpus_server_starts() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(10), true, 1000).1.is_some());
        assert!(matches!(present(&mut t, &mut now, MS(11), false, 3000).2, Some(Verdict::Offload(_))));
        t.clear_back();
        assert!(!t.take_back());
    }

    #[test]
    fn the_game_gpu_is_tried_again_only_once_it_is_no_faster_than_the_other_gpu_was() {
        // Kept on the game's GPU at 80 fps against the other GPU's 5 fps.
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_some());
        assert!(matches!(present(&mut t, &mut now, Duration::from_micros(12_500), false, 2000).2, Some(Verdict::Local { .. })));
        // A throttled card or a heavy scene at 10 fps: still twice the other GPU's rate, nothing spent.
        assert!(present(&mut t, &mut now, MS(100), false, 3000).1.is_none(), "a slowdown that still wins spends nothing");
        // A DLSS quality change: faster still.
        assert!(present(&mut t, &mut now, MS(8), false, 3000).1.is_none());
        // Slower than the other GPU was: tried once more.
        let (_, line, _) = present(&mut t, &mut now, MS(250), false, 2000);
        assert!(line.is_some_and(|l| l.contains("tried once more") && l.contains("200.0 ms (5 fps) measured with the model on the other GPU")));
        assert!(!t.finished_local() && t.take_back(), "back to the other GPU");
        // The other GPU at 5 fps, the game's GPU at 5 fps too: a tie keeps the earlier verdict, the game's GPU.
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_some());
        assert!(matches!(present(&mut t, &mut now, MS(200), false, 3000).2, Some(Verdict::Local { .. })), "a tie keeps the game's GPU");
        // No third trial, however slow it gets: no oscillation.
        assert!(present(&mut t, &mut now, MS(900), false, 3000).1.is_none());
        assert!(t.finished_local() && !t.take_back());
    }

    #[test]
    fn another_gpu_that_kept_the_model_is_tried_again_only_once_slower_than_the_game_gpu_was() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(10), true, 1000).1.is_some());
        assert!(matches!(present(&mut t, &mut now, MS(20), false, 3000).2, Some(Verdict::Offload(_))));
        assert!(t.take_back());
        // An alt-tab or a throttle that leaves it faster than the game's GPU was spends nothing.
        assert!(present(&mut t, &mut now, MS(18), true, 3000).1.is_none());
        let (_, line, _) = present(&mut t, &mut now, MS(30), true, 2000);
        assert!(line.is_some_and(|l| l.contains("the other GPU got slower") && l.contains("trying the model on the game's GPU")));
        assert!(t.running() && t.finished_local());
        t.handed_over(now);
        // A tie here keeps the other GPU, as the first verdict did.
        assert!(matches!(present(&mut t, &mut now, MS(30), false, 3000).2, Some(Verdict::Offload(_))));
        assert!(t.take_back());
        assert!(present(&mut t, &mut now, MS(100), true, 3000).1.is_none(), "tried once more only");
    }

    #[test]
    fn a_menu_then_gameplay_starts_the_trial_from_the_gameplay_windows() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        // A menu at 5 fps (no DLSS), then gameplay at 50 fps: the menu's window never pairs with gameplay's.
        for _ in 0..200 {
            assert!(t.finished_step(KEY, now, true, true, true, false, true, false).0.is_none());
            now += MS(200);
        }
        let (_, started, _) = present(&mut t, &mut now, MS(20), true, 1000);
        assert!(started.is_some_and(|s| s.contains("every 20.0 ms (50 fps)")));
    }

    #[test]
    fn the_held_and_finished_trials_are_apart() {
        reset_for_test();
        assert!(offloaded(KEY, MS(23), MS(5)).is_none() && offloaded(KEY, MS(23), MS(5)).is_some());
        assert!(keeps_local() && !finished_local() && finished_status_line().is_none());
        // The finished frame's windows go to its own trial: the held one keeps running, untouched.
        let mut now = Instant::now();
        let mut started = None;
        for _ in 0..1000 {
            started = with_finished(|t| t.finished_step(KEY, now, true, true, true, true, true, false).0);
            now += MS(10);
            if started.is_some() {
                break;
            }
        }
        assert!(started.is_some(), "the finished frame's own windows start its trial while the held one runs");
        assert!(finished_local() && with(|t| t.running() && !t.finished));
        // The holds' frames never feed the finished trial.
        local_hold(KEY, now, true, None);
        assert!(with_finished(|t| t.running()));
        reset_for_test();
    }

    #[test]
    fn gameplay_is_dlss_evaluating_or_the_post_paths_hand_back() {
        let now = Instant::now() + Duration::from_secs(1000);
        let ago = |s: u64| Some(now - Duration::from_secs(s));
        let ms_ago = |ms: u64| Some(now - MS(ms));
        assert_eq!(gameplay(ms_ago(500), ago(900), now), Gameplay::Yes, "DLSS evaluating (or its launches seen)");
        assert_eq!(gameplay(ago(2), ago(900), now), Gameplay::No, "DLSS paused: a menu");
        assert_eq!(gameplay(ago(301), ago(900), now), Gameplay::HandBack, "DLSS quiet past the hand-back: switched off, or a long pause");
        assert_eq!(gameplay(None, ago(10), now), Gameplay::No, "a DLSS game before its first evaluate");
        assert_eq!(gameplay(None, ago(301), now), Gameplay::Yes, "a DLSS game that never evaluates (another upscaler, Frame Generation only)");
        assert_eq!(gameplay(None, None, now), Gameplay::Yes, "a game without DLSS");
        // A pause menu of five minutes never measures a first trial; after a verdict, DLSS switched off counts.
        let mut t = Trial::default();
        assert!(!t.counts_as_gameplay(Gameplay::HandBack) && t.counts_as_gameplay(Gameplay::Yes) && !t.counts_as_gameplay(Gameplay::No));
        t.decided = Some((true, MS(10), MS(20)));
        assert!(t.counts_as_gameplay(Gameplay::HandBack));
    }

    #[test]
    fn near_ties_never_spend_the_re_trial() {
        // The other GPU kept it at 10.0 ms against the game's GPU's 9.5 ms (within the margin): noise around 10 ms.
        let mut t = Trial { phase: Phase::Decided, decided: Some((false, Duration::from_micros(9_500), MS(10))), ..Default::default() };
        let mut now = Instant::now();
        assert!(present_j(&mut t, &mut now, Duration::from_micros(10_300), MS(1), true, 5000).1.is_none());
        // The game's GPU kept it at 9.0 ms against the other GPU's 10.0 ms: noise up to 10.5 ms.
        let mut t = Trial { phase: Phase::KeptLocal("kept".into()), decided: Some((true, MS(9), MS(10))), ..Default::default() };
        let mut now = Instant::now();
        assert!(present_j(&mut t, &mut now, Duration::from_micros(10_500), MS(1), false, 5000).1.is_none());
        assert!(!t.retried);
    }

    #[test]
    fn a_late_hand_over_starts_the_build_budget_then() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        for _ in 0..100_000 {
            let (started, _) = t.finished_step(KEY, now, true, true, true, true, true, false);
            now += MS(20);
            if started.is_some() {
                break;
            }
        }
        // The hand-over comes 14 s later (the other GPU's server took long to end).
        for _ in 0..700 {
            assert!(t.finished_step(KEY, now, false, true, true, true, true, false).1.is_none());
            now += MS(20);
        }
        t.handed_over(now);
        // Then the network is built on the game's GPU: two seconds without the answer.
        for _ in 0..100 {
            let v = t.finished_step(KEY, now, false, false, true, true, true, false).1;
            assert!(v.is_none(), "the build budget counts from the hand-over: {v:?}");
            now += MS(20);
        }
        assert!(matches!(present(&mut t, &mut now, MS(5), false, 3000).2, Some(Verdict::Local { .. })));
    }

    #[test]
    fn a_trial_the_other_gpus_server_never_hands_over_ends() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        for _ in 0..100_000 {
            let (started, _) = t.finished_step(KEY, now, true, true, true, true, true, false);
            now += MS(20);
            if started.is_some() {
                break;
            }
        }
        assert!(t.running());
        let mut verdict = None;
        for _ in 0..2000 {
            verdict = t.finished_step(KEY, now, false, true, true, true, true, false).1;
            now += MS(20);
            if verdict.is_some() {
                break;
            }
        }
        assert!(matches!(verdict, Some(Verdict::Aborted(ref w)) if w.contains("did not hand over")), "{verdict:?}");
        assert!(!t.running() && !t.finished_local() && !t.take_back(), "nothing to end on the game's GPU");
        assert!(t.finished_status_line().is_none(), "no \"trying\" line for good");
        assert_eq!(t.aborts, 1, "counts toward the aborts");
        // The other GPU's server ending late is the hand-over it was: no failure; then nothing is expected.
        assert!(t.expects_handover());
        t.handed_over(now);
        assert!(!t.expects_handover());
    }

    #[test]
    fn a_re_trial_cut_short_keeps_its_tie_rule() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_some());
        assert!(matches!(present(&mut t, &mut now, Duration::from_micros(12_500), false, 2000).2, Some(Verdict::Local { .. })));
        assert!(present(&mut t, &mut now, MS(250), false, 2000).1.is_some_and(|l| l.contains("tried once more")));
        assert!(t.take_back() && t.prefer_local);
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_some());
        // Another size during the re-trial: aborted, and the tie rule stays for the next try.
        let other = ((1920, 1080), 1);
        assert!(matches!(t.finished_step(other, now, false, true, true, true, true, false).1, Some(Verdict::Aborted(_))));
        assert!(t.prefer_local);
    }

    #[test]
    fn a_window_of_scattered_presents_starts_again() {
        let mut p = Presents::default();
        let now = Instant::now();
        // Answers every other frame only: no interval between two that showed one.
        for i in 0..200u64 {
            assert!(p.note(KEY, now + MS(20 * i), i % 2 == 0).is_none());
        }
        assert!(p.intervals.is_empty());
    }
}
