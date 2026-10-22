//! A fixed-size, mutex-protected ring of recent pipeline-stage markers on the game's
//! own present thread, dumped to the log only when a real stall is detected (a bounded
//! fence wait actually timing out, or the synchronous present's own answer-wait running
//! out) -- so a real freeze is diagnosable from `nf-layer.log` afterward instead of
//! guessed at from a game that just stopped responding. Marking is deliberately cheap
//! (one short lock, one `Instant::now()`) and always on: a diagnostic that has to be
//! switched on before the stall it would explain is not a diagnostic.
//!
//! Idea and shape from PR #22 against DLSS5VKLayer (bmitch87), commit `0b41c98c`,
//! itself framed from LCPD15/DXL's `FreezeWatchdog.h`/`.cpp` (AGPL-3.0) -- description
//! only, no code taken from either. This module is a fresh, much simpler
//! implementation (a `Mutex`-guarded array, not upstream's lock-free ring): the marker
//! rate here is "once per pipeline stage transition", not "once per hardware event", so
//! a brief lock is cheap enough not to need lock-free atomics. See `ATTRIBUTION.md`.

use std::sync::Mutex;
use std::time::Instant;

const CAPACITY: usize = 64;

type Slot = Option<(&'static str, Instant)>;

/// The ring itself: `CAPACITY` slots and the index of the next one to write.
struct Ring {
    next: usize,
    entries: [Slot; CAPACITY],
}

impl Ring {
    const fn new() -> Self {
        Ring { next: 0, entries: [None; CAPACITY] }
    }

    fn mark(&mut self, stage: &'static str, at: Instant) {
        self.entries[self.next % CAPACITY] = Some((stage, at));
        self.next = self.next.wrapping_add(1);
    }

    /// The marks, newest first.
    fn newest_first(&self) -> impl Iterator<Item = (&'static str, Instant)> + '_ {
        (0..CAPACITY).filter_map(|i| self.entries[(self.next + CAPACITY - 1 - i) % CAPACITY])
    }

    /// The stage names, newest first, on one line (`a < b < c`).
    fn trail(&self) -> String {
        self.newest_first().map(|(stage, _)| stage).collect::<Vec<_>>().join(" < ")
    }
}

static RING: Mutex<Ring> = Mutex::new(Ring::new());

/// Records that `stage` was just reached on the calling thread. Safe to call from any
/// thread (a game can present from more than one); entries interleave by wall-clock
/// order, not per-thread.
pub fn mark(stage: &'static str) {
    let Ok(mut ring) = RING.lock() else { return };
    ring.mark(stage, Instant::now());
}

/// Logs the ring's contents, newest first, each with how long ago it was marked
/// relative to the call to this function. `reason` names why a dump was worth taking.
pub fn dump(reason: &str) {
    let Ok(ring) = RING.lock() else { return };
    let now = Instant::now();
    let lines: Vec<String> =
        ring.newest_first().map(|(stage, at)| format!("  -{:>9.3}ms  {stage}", now.duration_since(at).as_secs_f64() * 1000.0)).collect();
    drop(ring);
    crate::log!("[breadcrumbs] {reason} -- last {} stage marker(s), newest first:", lines.len());
    for line in lines {
        crate::log!("{line}");
    }
    crate::logging::flush();
}

/// The ring's stage names, newest first, on one line (`a < b < c`): what the state log keeps of a dump.
pub fn trail() -> String {
    let Ok(ring) = RING.lock() else { return String::new() };
    ring.trail()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The global ring is marked by every test in the binary that reaches a marked stage, concurrently,
    // so the ring's behaviour is tested on a ring of the test's own.

    #[test]
    fn marks_are_read_back_newest_first() {
        let mut ring = Ring::new();
        let now = Instant::now();
        ring.mark("one", now);
        ring.mark("two", now);
        let stages: Vec<_> = ring.newest_first().map(|(s, _)| s).collect();
        assert_eq!(stages, ["two", "one"]);
    }

    #[test]
    fn a_full_ring_keeps_the_newest_marks() {
        let mut ring = Ring::new();
        let now = Instant::now();
        for _ in 0..CAPACITY {
            ring.mark("old", now);
        }
        ring.mark("second-newest", now);
        ring.mark("newest", now);
        let stages: Vec<_> = ring.newest_first().map(|(s, _)| s).collect();
        assert_eq!(stages.len(), CAPACITY);
        assert_eq!(&stages[..2], ["newest", "second-newest"]);
    }

    #[test]
    fn the_trail_lists_the_newest_mark_first() {
        let mut ring = Ring::new();
        let now = Instant::now();
        ring.mark("a", now);
        ring.mark("b", now);
        assert_eq!(ring.trail(), "b < a");
        assert_eq!(Ring::new().trail(), "");
    }

    #[test]
    fn dump_and_trail_do_not_panic_on_the_shared_ring() {
        mark("test:shared:51c2");
        dump("unit test");
        let _ = trail();
    }
}
