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

static RING: Mutex<(usize, [Slot; CAPACITY])> = Mutex::new((0, [None; CAPACITY]));

/// Records that `stage` was just reached on the calling thread. Safe to call from any
/// thread (a game can present from more than one); entries interleave by wall-clock
/// order, not per-thread.
pub fn mark(stage: &'static str) {
    let Ok(mut guard) = RING.lock() else { return };
    let (next, entries) = &mut *guard;
    entries[*next % CAPACITY] = Some((stage, Instant::now()));
    *next = next.wrapping_add(1);
}

/// Logs the ring's contents, newest first, each with how long ago it was marked
/// relative to the call to this function. `reason` names why a dump was worth taking.
pub fn dump(reason: &str) {
    let Ok(guard) = RING.lock() else { return };
    let (next, entries) = &*guard;
    let now = Instant::now();
    let mut lines = Vec::with_capacity(CAPACITY);
    for i in 0..CAPACITY {
        let idx = (*next + CAPACITY - 1 - i) % CAPACITY;
        if let Some((stage, at)) = entries[idx] {
            lines.push(format!("  -{:>9.3}ms  {stage}", now.duration_since(at).as_secs_f64() * 1000.0));
        }
    }
    drop(guard);
    crate::log!("[breadcrumbs] {reason} -- last {} stage marker(s), newest first:", lines.len());
    for line in lines {
        crate::log!("{line}");
    }
    crate::logging::flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_are_read_back_newest_first() {
        // The ring is a shared global, so give this test its own stage names --
        // nothing else in the test binary uses them -- rather than reset shared state
        // other tests might be mid-write to.
        mark("test:one:9f3a");
        mark("test:two:9f3a");
        let (next, entries) = &*RING.lock().unwrap();
        let newest = entries[(*next + CAPACITY - 1) % CAPACITY];
        let second = entries[(*next + CAPACITY - 2) % CAPACITY];
        assert_eq!(newest.map(|(s, _)| s), Some("test:two:9f3a"));
        assert_eq!(second.map(|(s, _)| s), Some("test:one:9f3a"));
    }

    #[test]
    fn dump_does_not_panic_on_an_empty_or_partial_ring() {
        dump("unit test");
    }
}
