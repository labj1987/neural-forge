//! The measured choice between the second GPU and the game's own for the held placements (`super::offload_pre`, #63,
//! #70): the hold waits at its submit for the other GPU's answer to the frame before it, so once the other GPU is
//! slower than the game's frame it sets the frame rate. The present's late rule (`super::offload::LateRule`) never
//! sees that: the frames get as slow as the answers, so no answer is ever late. Whether the game's GPU does better
//! with the network on it is measured, not guessed: once two windows of offloaded holds in a row show the game waiting
//! for the other GPU ([`Trial::offloaded`]), the held placements go to the game's GPU for one trial ([`Trial::local`]);
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
pub(crate) const FINISHED_MIN: usize = 20;

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
    last: Option<Instant>,
    paused: Duration,
    /// The last hold's time if it ran the model; `None` after a miss (an interval is only between two holds that ran).
    last_ran: Option<Instant>,
    /// Intervals in a row too long to be frame times.
    slow: u32,
    intervals: Vec<Duration>,
}

/// The process's trial ([`TRIAL`]); its own type so tests drive one of their own.
#[derive(Debug, Default)]
pub(crate) struct Trial {
    phase: Phase,
    /// The previous offloaded window that showed the game waiting (its key and frame time): a trial starts on two in
    /// a row that agree.
    waiting: Option<(Key, Duration)>,
    aborts: u32,
    /// The trial was started by the finished frame ([`Trial::finished_window`]): its local frames are the presents'.
    finished: bool,
    /// The finished frame's trial ended with the model back on the other GPU, not yet acted on ([`Trial::take_back`]).
    back: bool,
    /// The window of offloaded finished-frame presents being filled.
    presents: Presents,
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
        if let Some(d) = self.last.replace(now).map(|t| now.saturating_duration_since(t)) {
            if d < FRAME_GAP {
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
        if wait < WAIT_MIN || wait.as_secs_f64() < frame.as_secs_f64() * WAIT_SHARE {
            self.waiting = None;
            return None;
        }
        let before = self.waiting.replace((key, frame));
        if !before.is_some_and(|(k, f)| k == key && agree(f, frame)) {
            return None;
        }
        self.waiting = None;
        self.finished = false;
        self.phase = Phase::Local(Local {
            finished: false,
            key,
            off: frame,
            wait,
            began: now,
            ran: 0,
            holds: 0,
            missed: 0,
            last: None,
            paused: Duration::ZERO,
            last_ran: None,
            slow: 0,
            intervals: Vec::with_capacity(LOCAL_MAX),
        });
        Some(format!(
            "trying the model on the game's GPU: the game waited {:.1} ms a frame for the other GPU's answer (a frame every {:.1} ms, {:.0} fps); the frame rate there decides where the model runs",
            ms(wait),
            ms(frame),
            fps(frame)
        ))
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
        self.phase = Phase::Local(Local {
            finished: true,
            key,
            off: frame,
            wait: Duration::ZERO,
            began: now,
            ran: 0,
            holds: 0,
            missed: 0,
            last: None,
            paused: Duration::ZERO,
            last_ran: None,
            slow: 0,
            intervals: Vec::with_capacity(LOCAL_MAX),
        });
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
            self.phase = if self.aborts >= ABORTS { Phase::Decided } else { Phase::Watching };
            return Some(Verdict::Aborted(why));
        }
        // A pause between holds (a menu, a loading screen) moves the budget's start: it counts holds that keep coming.
        let gap = l.last.map(|t| now.saturating_duration_since(t)).filter(|&d| d >= FRAME_GAP);
        if let Some(gap) = gap {
            let moved = gap.min(PAUSES_MAX.saturating_sub(l.paused));
            l.began += moved;
            l.paused += moved;
        }
        l.holds += 1;
        if l.holds > MAX_HOLDS {
            self.back = l.finished;
            self.phase = Phase::Decided;
            return Some(Verdict::Offload(UNMEASURED.into()));
        }
        if !ran {
            l.missed += 1;
            // A frame without the model: the next interval would span it.
            l.last_ran = None;
            let failed = if l.ran == 0 { now.saturating_duration_since(l.began) > BUILD_BUDGET } else { l.missed >= MISSES };
            l.last = Some(now);
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
        let waited = if finished { String::new() } else { format!(", which the game waited {:.1} ms a frame for", ms(wait)) };
        let numbers = format!(
            "a frame every {:.1} ms ({:.0} fps) with the model on the game's GPU, every {:.1} ms ({:.0} fps) with it on the other GPU{waited}",
            ms(local),
            fps(local),
            ms(off),
            fps(off),
        );
        if decide(off, local) && !forced_offload {
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

#[cfg(not(test))]
static TRIAL: std::sync::Mutex<Trial> = std::sync::Mutex::new(Trial {
    phase: Phase::Watching,
    waiting: None,
    aborts: 0,
    finished: false,
    back: false,
    presents: Presents { key: None, last: None, began: None, intervals: Vec::new() },
});

/// The process's trial.
#[cfg(not(test))]
fn with<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    f(&mut TRIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

#[cfg(test)]
thread_local! {
    /// Tests: each test thread's own trial, so a test that runs one is never reached by another test's holds
    /// (`Session::note` from tests that run in parallel).
    static TRIAL: std::cell::RefCell<Trial> = std::cell::RefCell::new(Trial::default());
}

#[cfg(test)]
fn with<R>(f: impl FnOnce(&mut Trial) -> R) -> R {
    TRIAL.with(|t| f(&mut t.borrow_mut()))
}

/// A side of the held placements was built on the other GPU: windows from before it say nothing of its frames.
pub(crate) fn side_built() {
    with(|t| t.waiting = None);
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
    let verdict = with(|t| if t.running() && !t.finished { t.local(key, now, ran, miss, forced) } else { None });
    say(verdict);
}

/// Whether the finished frame runs on the game's GPU for the trial or after it ([`Trial::finished_local`]): the
/// finished frame's server starts there, not on the other GPU.
pub(crate) fn finished_local() -> bool {
    with(|t| t.finished_local())
}

/// Whether the finished frame's trial just gave the model back to the other GPU ([`Trial::take_back`]).
pub(crate) fn take_back() -> bool {
    with(Trial::take_back)
}

/// A present of the finished-frame path (`device`'s present, every one) of `dims` at `now`: `offloaded` while the
/// server on the other GPU answers it (`super::offload::active`), `composited` when the frame showed the model's
/// answer. Offloaded presents fill the windows that start the trial ([`Trial::finished_window`]); during the trial
/// the presents on the game's GPU are its frames ([`Trial::local`]). Nothing without `NEURAL_FORGE_OFFLOAD_GPU`.
pub(crate) fn finished_present(dims: (u32, u32), now: Instant, offloaded: bool, composited: bool) {
    if super::offload::Choice::current() == super::offload::Choice::Off {
        return;
    }
    let key = (dims, super::placement::settings().0);
    let forced = forced_offload();
    let enabled = enabled();
    let (started, verdict) = with(|t| finished_step(t, key, now, offloaded, composited, enabled, forced));
    if let Some(line) = started {
        crate::log!("[offload] {line}");
        crate::logging::flush();
    }
    say(verdict);
}

/// [`finished_present`] on `t`: the line that started the trial, or the trial's verdict.
fn finished_step(t: &mut Trial, key: Key, now: Instant, offloaded: bool, composited: bool, enabled: bool, forced: bool) -> (Option<String>, Option<Verdict>) {
    if t.running() {
        // The game's GPU's frames: only once the other GPU's server is gone (its last answers say nothing).
        let verdict =
            (t.finished && !offloaded).then(|| t.local(key, now, composited, (!composited).then_some("no answer from the game's GPU yet"), forced)).flatten();
        return (None, verdict);
    }
    if !offloaded || !matches!(t.phase, Phase::Watching) {
        t.presents = Presents::default();
        return (None, None);
    }
    let started = t.presents.note(key, now, composited).and_then(|frame| t.finished_window(key, frame, enabled, now));
    if started.is_some() {
        t.presents = Presents::default();
    }
    (started, None)
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

    #[test]
    fn a_game_gpu_that_cannot_run_the_model_gives_the_frames_back() {
        // Never built: the budget runs out over holds that keep coming.
        let (mut t, mut now) = started(MS(36), MS(16));
        let mut verdict = None;
        for _ in 0..2000 {
            now += MS(20);
            if let Some(v) = t.local(KEY, now, false, Some("the native network is still loading"), false) {
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

    /// Presents of the finished frame every `every` from `now`, `offloaded` or not, each with the answer, until one
    /// starts the trial or brings a verdict (or `n` of them): which present, and what.
    fn present(t: &mut Trial, now: &mut Instant, every: Duration, offloaded: bool, n: usize) -> (usize, Option<String>, Option<Verdict>) {
        for i in 0..n {
            let (started, verdict) = finished_step(t, KEY, *now, offloaded, true, true, false);
            *now += every;
            if started.is_some() || verdict.is_some() {
                return (i, started, verdict);
            }
        }
        (n, None, None)
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
        assert!(t.finished_local() && !t.take_back(), "kept on the game's GPU for good");
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_none(), "never tried again");
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
        assert!(present(&mut t, &mut now, MS(10), true, 1000).1.is_none(), "nothing is tried again");
    }

    #[test]
    fn finished_frame_presents_start_nothing_unless_offloaded_and_the_holds_leave_its_trial_alone() {
        let mut t = Trial::default();
        let mut now = Instant::now();
        assert!(present(&mut t, &mut now, MS(200), false, 1000).1.is_none(), "the game's GPU's own frames start nothing");
        let mut off = Trial::default();
        assert!(finished_step(&mut off, KEY, now, true, true, false, false).0.is_none());
        assert!(present(&mut t, &mut now, MS(200), true, 1000).1.is_some());
        // A held trial's frames never come from the presents, and a finished one's never from the holds.
        let (mut held, start) = started(MS(23), MS(5));
        assert!(finished_step(&mut held, KEY, start, false, true, true, false).1.is_none() && held.running() && !held.finished_local());
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
