//! Steps the network down when it no longer fits in VRAM, and back up once there is room again.
//!
//! When the game and other programs leave too little VRAM, the driver moves part of the network's memory
//! to system memory without telling anyone: its allocations still succeed, but every launch then reads
//! over PCIe. Measured on an RTX 5070 at 3840x2160 with 2 GiB held by another program (2026-10-29): the
//! network went from 12.5 to about 300 ms a frame, GTA V from 158 to about 20 shown fps, while the game
//! alone kept 222 under the same pressure. A tight budget alone is not that: with 1 GiB held the network
//! stayed resident (12.5 ms) and is left as it is.
//!
//! So a step down needs both signs at once: the device at its VRAM budget (`VK_EXT_memory_budget`; within
//! [`SEVERE_MARGIN_MIB`] of it when the network is ten times slower than its best or slower than half a second), and
//! the network's GPU time far above its best since it was built (or above [`SPILL_ABS_MS`]), for
//! [`SPILL_FOR`]; or a build that ended with the device already at its budget. A step down pauses the
//! network ([`Level::Paused`]: no network, frames go to DLSS untouched and the game gets the memory back).
//! The full network steps down to the small one first only where the small one needs at least
//! [`SMALL_STEP_MIN_MIB`] less: with NVIDIA's 310.9.16 networks it never does (45-46 MiB less at every size,
//! measured 2026-11-09 by the rig's `what_each_network_takes_at_the_tables_sizes`), and a step that frees so
//! little cannot end a spill, it only costs a rebuild before the pause. A step up waits until the budget has
//! room for the bigger network (as much as its last build took, plus [`ROOM_MARGIN_MIB`]) for [`ROOM_FOR`],
//! and at least the current backoff since the last step down, which doubles when a step up is followed by a
//! step down within [`FLAP_WINDOW`].
//!
//! The guard only decides ([`Guard::proposal`]); the loader applies a step where a switch of the Network
//! setting is applied, after the frames that used the network were drained. A change of the setting
//! starts the guard afresh. Pure logic: the readings and times come from the caller.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use neural_forge_protocol::enums::network;
use neural_forge_protocol::vram_need::{network_need_mib, SMALL_LESS_MIB};

/// Within this much of its budget the device counts as at its budget (NVIDIA reports the usage of a
/// process whose memory spilled as about its budget, not above it).
pub(crate) const OVER_MARGIN_MIB: u64 = 128;
/// The network's GPU time this many times its best since the build counts as spilled...
pub(crate) const SPILL_FACTOR: f32 = 4.0;
/// ...and so does any time above this.
pub(crate) const SPILL_ABS_MS: f32 = 150.0;
/// How long both signs have to hold before a step down.
pub(crate) const SPILL_FOR: Duration = Duration::from_secs(2);
/// The window the network's time is judged over (its median), and the samples it needs.
const WINDOW: Duration = Duration::from_secs(1);
const WINDOW_MIN: usize = 5;
/// A spill that slows the game below [`WINDOW_MIN`] holds a second (Cyberpunk 2077 at 4K with path tracing: 1-8 fps,
/// the network 311 ms) has fewer samples in [`WINDOW`]: the last [`WINDOW_MIN`] are kept as long as they are this
/// recent, and the newest counts as fresh within it.
const SLOW_WINDOW: Duration = Duration::from_secs(5);
/// Samples after a build before its best time counts (the first frames warm the clocks and caches).
const WARMUP: u32 = 10;
/// The small network is a step of its own only when it needs at least this much less than the full one at the
/// size the holds want: twice the budget's margin, below which the difference is lost in the reading.
pub(crate) const SMALL_STEP_MIN_MIB: u64 = 2 * OVER_MARGIN_MIB;
/// A step up needs room for the bigger network plus this, for this long.
pub(crate) const ROOM_MARGIN_MIB: u64 = 512;
pub(crate) const ROOM_FOR: Duration = Duration::from_secs(10);
/// What a network is taken to need when neither its size nor a build of it is known.
const UNKNOWN_NEED_MIB: u64 = 1536;
/// The wait after a step down before a step up, and its bounds when steps flap.
const BACKOFF_FIRST: Duration = Duration::from_secs(30);
const BACKOFF_MAX: Duration = Duration::from_secs(300);
pub(crate) const FLAP_WINDOW: Duration = Duration::from_secs(120);

/// Where the guard has the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Level {
    /// The network the Network setting chose.
    #[default]
    Chosen,
    /// The small network in place of the chosen full one.
    Small,
    /// No network: frames go to DLSS untouched.
    Paused,
}

#[derive(Debug, Default)]
pub(crate) struct Guard {
    level: Level,
    /// A step decided and not applied yet.
    proposal: Option<Level>,
    /// The network's GPU time per hold since the last build (time, ms), the last [`WINDOW`] of it.
    samples: VecDeque<(Instant, f32)>,
    seen: u32,
    best_ms: Option<f32>,
    /// Since when both signs of a spill have held.
    spill_since: Option<Instant>,
    /// Since when the budget has had room for a step up.
    room_since: Option<Instant>,
    down_at: Option<Instant>,
    up_at: Option<Instant>,
    backoff: Duration,
    /// What a fresh open and build of each network took on this device (MiB), and at which size.
    need: [Option<(u64, (u32, u32))>; 2],
    /// Why the last step down was taken, for the status line.
    why: Option<String>,
    /// The size the holds last wanted (or the last build's), for the needs a step down compares.
    size: Option<(u32, u32)>,
}

fn at_budget((used, budget): (u64, u64)) -> bool {
    used + OVER_MARGIN_MIB >= budget
}

/// A network this many times slower than its best...
const SEVERE_FACTOR: f32 = 10.0;
/// ...or slower than this...
const SEVERE_ABS_MS: f32 = 500.0;
/// ...counts as spilled within this much of the budget, not only within [`OVER_MARGIN_MIB`]: Cyberpunk 2077 at 4K with
/// Ray Reconstruction and path tracing had the network at 1.7 s a frame (best 31 ms) with the reading 217 MiB under
/// the budget (10189 of 10406 MiB), the game at 0.1-0.7 fps. No GPU is that much slower without a spill.
const SEVERE_MARGIN_MIB: u64 = 512;

fn near_budget((used, budget): (u64, u64)) -> bool {
    used + SEVERE_MARGIN_MIB >= budget
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

impl Guard {
    #[cfg(test)]
    pub(crate) fn level(&self) -> Level {
        self.level
    }

    /// The network to run for the Network setting `choice`: `None` while paused.
    pub(crate) fn target(&self, choice: u32) -> Option<u32> {
        match self.level {
            Level::Chosen => Some(choice),
            Level::Small => Some(network::SMALL),
            Level::Paused => None,
        }
    }

    /// Why the network is stepped down, while it is.
    pub(crate) fn why(&self) -> Option<&str> {
        (self.level != Level::Chosen).then_some(self.why.as_deref()).flatten()
    }

    /// The step decided and not applied yet.
    #[cfg(test)]
    pub(crate) fn proposal(&self) -> Option<Level> {
        self.proposal
    }

    /// Applies the decided step; returns it.
    /// A step down is decided and not applied yet.
    pub(crate) fn step_down_pending(&self) -> bool {
        match (self.level, self.proposal) {
            (_, Some(Level::Paused)) => self.level != Level::Paused,
            (Level::Chosen, Some(Level::Small)) => true,
            _ => false,
        }
    }

    pub(crate) fn apply(&mut self) -> Option<Level> {
        let level = self.proposal.take()?;
        self.level = level;
        self.samples.clear();
        self.seen = 0;
        self.best_ms = None;
        self.spill_since = None;
        self.room_since = None;
        Some(level)
    }

    /// The Network setting changed: the guard starts afresh with the chosen network (what each network
    /// needs is kept: it is the device's).
    pub(crate) fn reset(&mut self) {
        *self = Self { need: self.need, size: self.size, ..Self::default() };
    }

    /// What `net` needs at `size` (MiB): its last fresh build here scaled by the measured table
    /// (`network_need_mib`) from that build's size to this one, else the table's figure, else a fixed guess.
    fn estimate(&self, net: u32, size: Option<(u32, u32)>) -> u64 {
        let table = |(w, h): (u32, u32)| network_need_mib(w, h);
        match (self.need.get(net as usize).copied().flatten(), size) {
            (Some((mib, at)), Some(size)) => mib * table(size) / table(at).max(1),
            (Some((mib, _)), None) => mib,
            (None, Some(size)) => table(size).saturating_sub(if net == network::SMALL { SMALL_LESS_MIB } else { 0 }),
            (None, None) => UNKNOWN_NEED_MIB,
        }
    }

    /// Whether the small network is worth a step of its own in place of the full one: it needs at least
    /// [`SMALL_STEP_MIN_MIB`] less at the size the holds want (estimated as for a step up).
    fn small_frees_enough(&self) -> bool {
        self.estimate(network::SMALL, self.size) + SMALL_STEP_MIN_MIB <= self.estimate(network::FULL, self.size)
    }

    /// A build of `running` finished, for `size` (its first). `fresh_need`: the VRAM a fresh open and build
    /// took (MiB), when the network was opened for it; `after`: the (used, budget) reading after it. Returns a
    /// log line when the build already ended at the budget and a step down is decided.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn built(
        &mut self,
        running: u32,
        fresh_need: Option<u64>,
        size: (u32, u32),
        after: Option<(u64, u64)>,
        choice: u32,
        small_present: bool,
        now: Instant,
    ) -> Option<String> {
        if let (Some(need), Some(slot)) = (fresh_need, self.need.get_mut(running as usize)) {
            *slot = Some((need, size));
        }
        self.size = Some(size);
        self.samples.clear();
        self.seen = 0;
        self.best_ms = None;
        self.spill_since = None;
        let after = after.filter(|&r| at_budget(r))?;
        self.step_down(choice, small_present, now, format!("the device was at its VRAM budget after the build ({} of {} MiB)", after.0, after.1))
    }

    /// One hold's network GPU time.
    pub(crate) fn gpu_ms(&mut self, ms: f32, now: Instant) {
        if !ms.is_finite() || ms <= 0.0 {
            return;
        }
        self.seen += 1;
        self.samples.push_back((now, ms));
        // Older than WINDOW, unless it is one of the last WINDOW_MIN and within SLOW_WINDOW (a slow game).
        while self.samples.front().is_some_and(|&(t, _)| {
            let age = now.duration_since(t);
            age > SLOW_WINDOW || (age > WINDOW && self.samples.len() > WINDOW_MIN)
        }) {
            self.samples.pop_front();
        }
        if self.seen > WARMUP {
            self.best_ms = Some(self.best_ms.map_or(ms, |b| b.min(ms)));
        }
    }

    /// `NEURAL_FORGE_PROBE_VRAM=1`'s line: the level, the samples kept (their age and time) and the best time.
    pub(crate) fn probe(&self, now: Instant) -> String {
        let samples: Vec<String> = self.samples.iter().map(|&(t, ms)| format!("{:.1}s:{ms:.1}", now.saturating_duration_since(t).as_secs_f32())).collect();
        format!(
            "level {:?}, best {:?}, spill since {:?}, samples [{}]",
            self.level,
            self.best_ms,
            self.spill_since.map(|t| now.saturating_duration_since(t)),
            samples.join(" ")
        )
    }

    /// `NEURAL_FORGE_NATIVE_FAIL=vram`: a step down as if the network had spilled, to exercise the steps, the pause
    /// and the way back on the rig. Returns its log line.
    pub(crate) fn fake_spill(&mut self, choice: u32, small_present: bool, now: Instant) -> Option<String> {
        if self.proposal.is_some() {
            return None;
        }
        self.step_down(choice, small_present, now, "NEURAL_FORGE_NATIVE_FAIL=vram asked for a step down".to_string())
    }

    /// The loader's periodic check, with the device's (used, budget) reading (none without
    /// `VK_EXT_memory_budget`: the guard then never steps). `running` is the network open now, if built; `size`
    /// the size the holds want (the needs are estimated for it). Returns a log line when a step is decided.
    pub(crate) fn tick(
        &mut self,
        reading: Option<(u64, u64)>,
        running: Option<u32>,
        size: Option<(u32, u32)>,
        choice: u32,
        small_present: bool,
        now: Instant,
    ) -> Option<String> {
        if size.is_some() {
            self.size = size;
        }
        let reading = reading?;
        if self.proposal.is_some() {
            return None;
        }
        // A network running: is it spilling?
        if running.is_some() && self.level != Level::Paused {
            let fresh = self.samples.back().is_some_and(|&(t, _)| now.duration_since(t) <= SLOW_WINDOW);
            // Five samples give a median; a game the spill stalls may deliver fewer (a few network frames over seconds,
            // each well over a second in Cyberpunk 2077 with path tracing at 4K): two or more all above SPILL_ABS_MS are
            // as clear, and their smallest is taken. Not among the first WARMUP frames after a build: those warm the
            // clocks and caches, and two of them over SPILL_ABS_MS on a slower GPU are no spill.
            let times: Vec<f32> = self.samples.iter().map(|&(_, ms)| ms).collect();
            let window = match times.len() {
                _ if !fresh => None,
                n if n >= WINDOW_MIN => Some(median(&mut times.clone())),
                n if n >= 2 && self.seen > WARMUP && times.iter().all(|&ms| ms > SPILL_ABS_MS) => times.iter().copied().reduce(f32::min),
                _ => None,
            };
            let slow = window.is_some_and(|m| m > SPILL_ABS_MS || self.best_ms.is_some_and(|b| m > b * SPILL_FACTOR));
            let severe = window.is_some_and(|m| m > SEVERE_ABS_MS || self.best_ms.is_some_and(|b| m > b * SEVERE_FACTOR));
            if slow && (at_budget(reading) || severe && near_budget(reading)) {
                let since = *self.spill_since.get_or_insert(now);
                if now.duration_since(since) >= SPILL_FOR {
                    let m = window.unwrap_or_default();
                    let best = self.best_ms.map_or_else(|| "none yet".to_string(), |b| format!("{b:.1} ms"));
                    return self.step_down(
                        choice,
                        small_present,
                        now,
                        format!("the network took {m:.1} ms a frame (best {best}) with the device at its VRAM budget ({} of {} MiB)", reading.0, reading.1),
                    );
                }
            } else {
                self.spill_since = None;
            }
        }
        // Stepped down: room for the bigger network again?
        let up = match self.level {
            Level::Chosen => return None,
            Level::Small => Level::Chosen,
            Level::Paused if choice == network::FULL && small_present && self.small_frees_enough() => Level::Small,
            Level::Paused => Level::Chosen,
        };
        let want = match up {
            Level::Small => network::SMALL,
            _ => choice,
        };
        let need = self.estimate(want, size);
        // What the network open now frees when it is closed for the bigger one.
        let frees = running.map_or(0, |r| self.estimate(r, size));
        let room = reading.1.saturating_sub(reading.0) + frees;
        if room < need + ROOM_MARGIN_MIB {
            self.room_since = None;
            return None;
        }
        let since = *self.room_since.get_or_insert(now);
        if now.duration_since(since) < ROOM_FOR || self.down_at.is_some_and(|d| now.duration_since(d) < self.backoff) {
            return None;
        }
        self.proposal = Some(up);
        self.up_at = Some(now);
        Some(format!(
            "[native] VRAM guard: {} MiB free of the budget ({} of {} MiB used) for {} s: back to the {} network",
            reading.1.saturating_sub(reading.0),
            reading.0,
            reading.1,
            ROOM_FOR.as_secs(),
            network::name(want)
        ))
    }

    fn step_down(&mut self, choice: u32, small_present: bool, now: Instant, why: String) -> Option<String> {
        let down = match self.level {
            Level::Chosen if choice == network::FULL && small_present && self.small_frees_enough() => Level::Small,
            Level::Chosen | Level::Small => Level::Paused,
            Level::Paused => return None,
        };
        // A step down soon after a step up: the room did not last, so wait longer before the next one.
        self.backoff =
            if self.up_at.is_some_and(|u| now.duration_since(u) < FLAP_WINDOW) { (self.backoff * 2).clamp(BACKOFF_FIRST, BACKOFF_MAX) } else { BACKOFF_FIRST };
        self.down_at = Some(now);
        self.proposal = Some(down);
        let what = match down {
            Level::Small => "the small network runs in its place",
            _ => "the network is paused (frames go to DLSS untouched) until there is room again",
        };
        let line = format!("[native] VRAM guard: {why}; {what} (no step up for at least {} s)", self.backoff.as_secs());
        self.why = Some(match down {
            Level::Small => "too little VRAM for the full network: running the small one".to_string(),
            _ => "paused: too little VRAM for the network; it resumes once there is room".to_string(),
        });
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: u32 = network::FULL;
    const SMALL: u32 = network::SMALL;

    fn t(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    /// Holds at 60 fps of `ms` each from `from` to `to` (ms after `base`), with a tick every second at
    /// `reading`, the full network chosen; returns the first line a tick gave.
    fn run(g: &mut Guard, base: Instant, from: u64, to: u64, ms: f32, reading: (u64, u64), running: Option<u32>) -> Option<String> {
        let mut at = from;
        let mut line = None;
        while at < to {
            g.gpu_ms(ms, t(base, at));
            if at % 1000 < 17 {
                line = line.or_else(|| g.tick(Some(reading), running, Some((2228, 1253)), FULL, true, t(base, at)));
            }
            at += 16;
        }
        line
    }

    /// The 2 GiB case: the device at its budget and the network twenty times slower than its best: the
    /// full network steps down after two seconds of it, not before; to the small one, which in these
    /// builds (1500 against about 560 MiB) frees enough to be a step of its own.
    #[test]
    fn a_spilled_network_steps_down_to_the_small_one() {
        let base = Instant::now();
        let mut g = Guard::default();
        assert!(g.built(FULL, Some(1500), (2228, 1253), Some((5000, 8000)), FULL, true, base).is_none(), "room after the build");
        assert!(run(&mut g, base, 0, 5000, 12.5, (7000, 8000), Some(FULL)).is_none(), "normal frames");
        assert!(run(&mut g, base, 5000, 6500, 300.0, (7990, 8000), Some(FULL)).is_none(), "under two seconds of it");
        let line = run(&mut g, base, 6500, 10_000, 300.0, (7990, 8000), Some(FULL)).expect("a step down");
        assert!(line.contains("small network runs"), "{line}");
        assert_eq!(g.proposal(), Some(Level::Small));
        assert_eq!(g.apply(), Some(Level::Small));
        assert_eq!(g.target(FULL), Some(SMALL));
        assert!(g.why().unwrap().contains("running the small one"));
    }

    /// A spill bad enough to slow the game to a few frames a second (Cyberpunk 2077 at 4K with Ray Reconstruction
    /// and path tracing: 1-8 fps, the network 311 ms against its 32): under five holds a second never filled the
    /// one-second window, so the guard never stepped down. The last five holds within five seconds count.
    #[test]
    fn a_spill_that_slows_the_game_to_a_few_fps_still_steps_down() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 8000)), FULL, false, base);
        assert!(run(&mut g, base, 0, 5000, 32.0, (7000, 8000), Some(FULL)).is_none(), "normal frames");
        // Four holds a second of 311 ms, at the budget.
        let mut line = None;
        let mut at = 5000;
        while at < 12_000 && line.is_none() {
            g.gpu_ms(311.0, t(base, at));
            line = g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, at + 10));
            at += 250;
        }
        let line = line.expect("a step down");
        assert!(line.contains("paused"), "{line}");
        assert!(at <= 9000, "within about three seconds of the spill (the median turning, then SPILL_FOR), at {at}");
        // One hold a second, as slow: also stepped down.
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 8000)), FULL, false, base);
        run(&mut g, base, 0, 5000, 32.0, (7000, 8000), Some(FULL));
        let stepped = (5..20).any(|k| {
            g.gpu_ms(600.0, t(base, k * 1000));
            g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, k * 1000 + 500)).is_some()
        });
        assert!(stepped);
        // Two network frames five seconds apart, each over a second (holds that come only now and then, the game
        // stalling on each): stepped down too; one such frame alone is not enough.
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 8000)), FULL, false, base);
        run(&mut g, base, 0, 5000, 32.0, (7000, 8000), Some(FULL));
        g.gpu_ms(1325.0, t(base, 11_000));
        assert!(g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 11_500)).is_none(), "one slow frame");
        g.gpu_ms(1325.0, t(base, 14_000));
        let stepped = (0..6).any(|k| g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 14_100 + k * 1000)).is_some());
        assert!(stepped, "two slow frames within five seconds, at the budget");
        // The network 50 times slower than its best, 217 MiB under the budget (the reading in that game): a spill too.
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 10406)), FULL, false, base);
        run(&mut g, base, 0, 5000, 31.0, (7000, 10406), Some(FULL));
        let stepped = (0..8).any(|k| {
            g.gpu_ms(1700.0, t(base, 5000 + k * 1000));
            g.tick(Some((10189, 10406)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 5000 + k * 1000 + 500)).is_some()
        });
        assert!(stepped, "severe, near the budget");
        // As slow with a gigabyte to spare: not a spill.
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 10406)), FULL, false, base);
        run(&mut g, base, 0, 5000, 31.0, (7000, 10406), Some(FULL));
        let stepped = (0..8).any(|k| {
            g.gpu_ms(1700.0, t(base, 5000 + k * 1000));
            g.tick(Some((9400, 10406)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 5000 + k * 1000 + 500)).is_some()
        });
        assert!(!stepped, "severe with room");
        // Slow frames with room to spare stay as they are; and samples older than five seconds do not count.
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 8000)), FULL, false, base);
        for k in 0..5 {
            g.gpu_ms(311.0, t(base, k * 250));
        }
        assert!(g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 7000)).is_none(), "stale samples");
    }

    /// The first frames after a build warm the clocks and caches (WARMUP): two of them over SPILL_ABS_MS at the budget
    /// (a slower GPU's first frames at 4K) are no spill, however long they are the only ones; once the build is past
    /// its warm-up, two such frames are.
    #[test]
    fn two_slow_frames_right_after_a_build_are_no_spill() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (3840, 2160), Some((5000, 8000)), FULL, false, base);
        g.gpu_ms(400.0, t(base, 100));
        g.gpu_ms(400.0, t(base, 600));
        let stepped = (0..5).any(|k| g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 700 + k * 1000)).is_some());
        assert!(!stepped, "the build's first two frames");
        // Past the warm-up (frames at the build's own pace), two slow frames are a spill again.
        for k in 0..WARMUP as u64 {
            g.gpu_ms(30.0, t(base, 5000 + k * 16));
        }
        g.gpu_ms(400.0, t(base, 11_000));
        g.gpu_ms(400.0, t(base, 11_500));
        let stepped = (0..5).any(|k| g.tick(Some((7990, 8000)), Some(FULL), Some((3840, 2160)), FULL, false, t(base, 11_600 + k * 1000)).is_some());
        assert!(stepped, "two slow frames after the warm-up");
    }

    /// The 1 GiB case: at its budget but as fast as ever, the network is left as it is; and a GPU that is
    /// slow with room to spare (no spill) is never stepped down either.
    #[test]
    fn a_tight_budget_alone_or_a_slow_gpu_alone_changes_nothing() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((5000, 8000)), FULL, true, base);
        assert!(run(&mut g, base, 0, 20_000, 12.5, (7990, 8000), Some(FULL)).is_none(), "at the budget, not slow");
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((5000, 8000)), FULL, true, base);
        assert!(run(&mut g, base, 0, 20_000, 200.0, (4000, 8000), Some(FULL)).is_none(), "slow, with room");
        assert_eq!(g.level(), Level::Chosen);
    }

    /// A build that already ends at the budget is spilled from its first frame (its best time would be the
    /// spilled one): it steps down at once.
    #[test]
    fn a_build_at_the_budget_steps_down_at_once() {
        let base = Instant::now();
        let mut g = Guard::default();
        let line = g.built(FULL, Some(1400), (2228, 1253), Some((8024, 8039)), FULL, true, base).expect("a step down");
        assert!(line.contains("at its VRAM budget after the build"), "{line}");
        assert_eq!(g.apply(), Some(Level::Small));
    }

    /// The small network spilling too pauses the network; with the small one chosen (or not extracted) the
    /// first step down is the pause.
    #[test]
    fn the_small_network_spilling_pauses() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, true, base);
        g.apply();
        g.built(SMALL, Some(700), (2228, 1253), Some((7000, 8000)), FULL, true, t(base, 100));
        let line = run(&mut g, base, 200, 10_000, 250.0, (7995, 8000), Some(SMALL)).expect("a pause");
        assert!(line.contains("paused"), "{line}");
        assert_eq!(g.apply(), Some(Level::Paused));
        assert_eq!(g.target(FULL), None);
        let mut g = Guard::default();
        g.built(SMALL, Some(700), (2228, 1253), Some((8000, 8010)), SMALL, true, base);
        assert_eq!(g.apply(), Some(Level::Paused), "the small network chosen");
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, false, base);
        assert_eq!(g.apply(), Some(Level::Paused), "no small network extracted");
    }

    /// Paused, with a small network that frees enough (700 against 1500 MiB here): once the budget has room
    /// for the small network (its last build plus the margin) for ten seconds and the backoff has passed, it
    /// comes back; then the full one once there is room for it (counting what closing the small one frees).
    /// Not before.
    #[test]
    fn room_again_steps_back_up_one_level_at_a_time() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, true, base);
        g.apply();
        g.built(SMALL, Some(700), (2228, 1253), Some((8000, 8010)), FULL, true, t(base, 10));
        g.apply();
        assert_eq!(g.level(), Level::Paused);
        // 1000 MiB free: room for the small network (700 + 512)? No.
        assert!(g.tick(Some((7000, 8000)), None, Some((2228, 1253)), FULL, true, t(base, 60_000)).is_none());
        // 1300 MiB free from 60 s on: ten seconds of it.
        assert!(g.tick(Some((6700, 8000)), None, Some((2228, 1253)), FULL, true, t(base, 61_000)).is_none());
        assert!(g.tick(Some((6700, 8000)), None, Some((2228, 1253)), FULL, true, t(base, 65_000)).is_none(), "under ten seconds");
        let line = g.tick(Some((6700, 8000)), None, Some((2228, 1253)), FULL, true, t(base, 71_000)).expect("back to the small network");
        assert!(line.contains("back to the small network"), "{line}");
        assert_eq!(g.apply(), Some(Level::Small));
        // The small one running (700 MiB of it): the full one needs 1500 + 512, so 1400 free + 700 freed is enough.
        assert!(g.tick(Some((6600, 8000)), Some(SMALL), Some((2228, 1253)), FULL, true, t(base, 72_000)).is_none());
        let line = g.tick(Some((6600, 8000)), Some(SMALL), Some((2228, 1253)), FULL, true, t(base, 82_000)).expect("back to the full network");
        assert!(line.contains("back to the full network"), "{line}");
        assert_eq!(g.apply(), Some(Level::Chosen));
        assert_eq!(g.target(FULL), Some(FULL));
        assert!(g.why().is_none());
    }

    /// No step up during the backoff, however much room; a step down soon after a step up doubles it.
    #[test]
    fn steps_that_flap_wait_longer() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, true, base);
        g.apply();
        assert!(g.backoff == BACKOFF_FIRST);
        let roomy = (2000, 8000);
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 1_000)).is_none());
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 15_000)).is_none(), "ten seconds of room, but inside the backoff");
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 31_000)).is_some(), "after the backoff");
        g.apply();
        // Spilled again 20 s later: the backoff doubles.
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, true, t(base, 51_000));
        assert_eq!(g.backoff, BACKOFF_FIRST * 2);
        g.apply();
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 52_000)).is_none());
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 100_000)).is_none(), "49 s: inside the 60 s backoff");
        assert!(g.tick(Some(roomy), None, Some((2228, 1253)), FULL, true, t(base, 112_000)).is_some());
    }

    /// The full network's need recorded at the menus' 3840x2160 is scaled to the in-game size for a step up: with
    /// room for its 2228x1253 build (not for the 4K one) it comes back. Without any build of it, the measured table's
    /// figure for the size counts.
    #[test]
    fn needs_are_scaled_to_the_size_wanted() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1352), (3840, 2160), None, FULL, true, base);
        assert_eq!(g.estimate(FULL, Some((2228, 1253))), 1352 * 607 / 1415);
        assert_eq!(g.estimate(SMALL, Some((2228, 1253))), 607 - SMALL_LESS_MIB, "no build of it: the table");
        assert!(g.fake_spill(FULL, true, base).unwrap().contains("NEURAL_FORGE_NATIVE_FAIL=vram"));
        assert_eq!(g.apply(), Some(Level::Paused));
        // 1100 MiB free: room for the full one at 2228x1253 (580 + 512), not at 4K (1352 + 512).
        let s = Some((2228, 1253));
        assert!(g.tick(Some((6900, 8000)), None, s, FULL, true, t(base, 40_000)).is_none());
        assert!(g.tick(Some((6900, 8000)), None, s, FULL, true, t(base, 50_000)).unwrap().contains("back to the full network"));
        let mut g = Guard::default();
        g.built(FULL, Some(1352), (3840, 2160), None, FULL, true, base);
        g.fake_spill(FULL, true, base);
        g.apply();
        let s = Some((3840, 2160));
        assert!(g.tick(Some((6900, 8000)), None, s, FULL, true, t(base, 40_000)).is_none());
        assert!(g.tick(Some((6900, 8000)), None, s, FULL, true, t(base, 50_000)).is_none(), "no room for the 4K build");
    }

    /// NVIDIA's 310.9.16 networks as measured on the rig (2026-11-09): the small one takes only 45-46 MiB less than the
    /// full one at every size, so a spill pauses the full network at once (a step to the small one would free too little
    /// to end it) and room again brings the full one straight back, from the table alone and from real builds.
    #[test]
    fn a_small_network_that_frees_too_little_is_skipped() {
        let base = Instant::now();
        let s = (2560, 1440);
        let mut g = Guard::default();
        let line = g.built(FULL, None, s, Some((11_900, 11_950)), FULL, true, base).expect("a step down");
        assert!(line.contains("paused"), "{line}");
        assert_eq!(g.apply(), Some(Level::Paused));
        assert!(g.tick(Some((10_000, 11_950)), None, Some(s), FULL, true, t(base, 40_000)).is_none());
        let line = g.tick(Some((10_000, 11_950)), None, Some(s), FULL, true, t(base, 50_000)).expect("a step up");
        assert!(line.contains("back to the full network"), "{line}");
        assert_eq!(g.apply(), Some(Level::Chosen));
        // With both networks' real builds recorded (729 and 683 MiB) it is the same.
        let mut g = Guard::default();
        g.built(SMALL, Some(683), s, None, SMALL, true, base);
        g.built(FULL, Some(729), s, None, FULL, true, base);
        assert!(!g.small_frees_enough());
        assert!(g.fake_spill(FULL, true, base).unwrap().contains("paused"));
        // A small network that does free enough is still a step of its own.
        let mut g = Guard::default();
        g.built(SMALL, Some(729 - SMALL_STEP_MIN_MIB), s, None, SMALL, true, base);
        g.built(FULL, Some(729), s, None, FULL, true, base);
        assert!(g.small_frees_enough());
        assert!(g.fake_spill(FULL, true, base).unwrap().contains("small network runs"));
    }

    /// A change of the Network setting starts the guard afresh (keeping what each network needs); without
    /// budget readings it never steps.
    #[test]
    fn a_setting_change_resets_and_no_readings_never_step() {
        let base = Instant::now();
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), Some((8000, 8010)), FULL, true, base);
        g.apply();
        g.reset();
        assert_eq!((g.level(), g.proposal(), g.need[FULL as usize]), (Level::Chosen, None, Some((1500, (2228, 1253)))));
        let mut g = Guard::default();
        g.built(FULL, Some(1500), (2228, 1253), None, FULL, true, base);
        for i in 0..600 {
            g.gpu_ms(400.0, t(base, i * 16));
            assert!(g.tick(None, Some(FULL), Some((2228, 1253)), FULL, true, t(base, i * 16)).is_none());
        }
        assert_eq!(g.level(), Level::Chosen);
    }
}
