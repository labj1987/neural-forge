//! When the native network is built again after it failed to build (re-initialising means closing and
//! opening the network again). The schedule was first made for NVIDIA's NGX feature in 2.x:
//!
//! A failed `CreateFeature` used to be retried on the rebuild spacing (250 ms) for ever, with
//! nothing in between: no backoff, no re-initialisation of NGX, and `model_up` left at 1. On the
//! rig (4K, VRAM full) one `0xbad00002` was followed by every later creation failing too, across
//! game launches, until the model server process was restarted (docs/PRE_UPSCALER_DESIGN.md,
//! "Robustness: failed feature builds"). [`BuildRetry`] is the schedule that replaced it:
//!
//! - after the 1st, 2nd and 3rd consecutive failure: wait [`BuildRetry::SHORT`] (0.5, 1, 2 s);
//! - the 4th attempt re-initialises NGX first (`Shutdown1` + `VULKAN_Init_Ext`), once;
//! - from then on: wait [`BuildRetry::LONG`] (30 s) between attempts, re-initialising again on
//!   every [`BuildRetry::REINIT_EVERY`]th of them (so at most every two minutes).
//!
//! A success ends the streak. The schedule does not reset on a change of size or HDR mode: those
//! change at every loading screen in some setups, and resetting there is what made the old retry
//! spin.
//!
//! [`FailInject`] is the `NEURAL_FORGE_FAIL_CREATE` fault injection used to exercise all of this
//! on the rig. Pure bookkeeping with no Vulkan calls.

use std::time::{Duration, Instant};

/// What to do about the network now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Not yet: the next attempt is due in this long.
    Wait(Duration),
    /// Build it.
    Build,
    /// Close and reopen the network, then build.
    ReinitThenBuild,
}

/// The retry schedule for the network (see the module doc).
#[derive(Debug, Default, Clone)]
pub struct BuildRetry {
    /// Consecutive failed builds.
    streak: u32,
    /// When the next attempt is due (`None`: now).
    next_at: Option<Instant>,
    /// The streak value at which the network was last re-initialised, so a step asked twice at the same
    /// streak re-initialises once.
    reinit_at: Option<u32>,
    /// Re-initialisations in this streak.
    reinits: u32,
    /// The streak a success just ended, until [`Self::take_recovered`].
    recovered: Option<u32>,
}

impl BuildRetry {
    /// The waits after the 1st, 2nd and 3rd consecutive failure.
    pub const SHORT: [Duration; 3] = [Duration::from_millis(500), Duration::from_secs(1), Duration::from_secs(2)];
    /// The wait once the short ones are used up.
    pub const LONG: Duration = Duration::from_secs(30);
    /// The attempt after this many consecutive failures re-initialises the network first.
    pub const REINIT_AFTER: u32 = 3;
    /// After the first re-initialisation, again every this many failures.
    pub const REINIT_EVERY: u32 = 4;

    /// What to do at `now`.
    pub fn step(&mut self, now: Instant) -> Step {
        if let Some(t) = self.next_at.filter(|t| now < *t) {
            return Step::Wait(t - now);
        }
        let due = self.streak >= Self::REINIT_AFTER && (self.streak - Self::REINIT_AFTER).is_multiple_of(Self::REINIT_EVERY);
        if due && self.reinit_at != Some(self.streak) {
            self.reinit_at = Some(self.streak);
            self.reinits += 1;
            return Step::ReinitThenBuild;
        }
        Step::Build
    }

    /// A build failed at `now`. Returns the wait before the next attempt.
    pub fn failed(&mut self, now: Instant) -> Duration {
        self.streak += 1;
        let wait = Self::SHORT.get(self.streak as usize - 1).copied().unwrap_or(Self::LONG);
        self.next_at = Some(now + wait);
        wait
    }

    /// A build succeeded: the streak is over.
    pub fn succeeded(&mut self) {
        if self.streak > 0 {
            self.recovered = Some(self.streak);
        }
        *self = Self { recovered: self.recovered, ..Self::default() };
    }

    /// Whether the last attempt failed (the model is not buildable right now).
    pub fn failing(&self) -> bool {
        self.streak > 0
    }

    /// Consecutive failures so far.
    pub fn streak(&self) -> u32 {
        self.streak
    }

    /// Re-initialisations of the network in this streak.
    pub fn reinits(&self) -> u32 {
        self.reinits
    }

    /// The length of the failure streak the last success ended, once.
    pub fn take_recovered(&mut self) -> Option<u32> {
        self.recovered.take()
    }
}

/// When the features released for a new frame key (a DLSS render-resolution or quality change, an
/// SDR/HDR switch) may be built again, given `build_after` (the previous build's time plus the
/// rebuild spacing, which `maintain_passes` sets after every build): at once, unless that build
/// was less than the spacing ago, so a key that keeps changing (a window being resized) still
/// builds at most once per spacing. `None` means now.
///
/// Nothing answers until the rebuild: every request meanwhile is echoed. Waiting a full spacing
/// after every change (250 ms) echoed ~15 frames at 60 fps while `model_up` said 1, more than the
/// layer's circuit breaker allows in a row, so the breaker opened for 2 s on every such change.
pub fn after_key_change(build_after: Option<Instant>, now: Instant) -> Option<Instant> {
    build_after.filter(|t| *t > now)
}

/// `NEURAL_FORGE_FAIL_CREATE=N` or `N@K`: let `K` creations through (default 0), then fail the next
/// `N` without building. A debug switch for the rig; unset (the default) does nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FailInject {
    skip: u32,
    fail: u32,
}

impl FailInject {
    pub const ENV: &'static str = "NEURAL_FORGE_FAIL_CREATE";

    /// `None` for anything but `N` or `N@K` with `N` > 0.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let (n, k) = match value.split_once('@') {
            Some((n, k)) => (n.trim().parse().ok()?, k.trim().parse().ok()?),
            None => (value.parse().ok()?, 0),
        };
        (n > 0).then_some(Self { skip: k, fail: n })
    }

    /// Called once per creation: whether this one is to fail.
    pub fn should_fail(&mut self) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            return false;
        }
        if self.fail > 0 {
            self.fail -= 1;
            return true;
        }
        false
    }

    /// Failures still to come.
    pub fn remaining(&self) -> u32 {
        self.fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame-key change long after the last build rebuilds at once (the layer's breaker would
    /// otherwise count a spacing's worth of echoes); one right after a build still waits for the
    /// spacing, so a key that keeps changing cannot rebuild every frame.
    #[test]
    fn a_key_change_rebuilds_at_once_unless_the_last_build_was_within_the_spacing() {
        let spacing = Duration::from_millis(250);
        let t0 = Instant::now();
        let after_build = Some(t0 + spacing);
        assert_eq!(after_key_change(after_build, t0 + Duration::from_secs(10)), None, "steady state: build now");
        assert_eq!(after_key_change(None, t0), None, "never built or discarded: build now");
        assert_eq!(after_key_change(after_build, t0 + Duration::from_millis(100)), Some(t0 + spacing), "just built: wait out the spacing");
        assert_eq!(after_key_change(after_build, t0 + spacing), None);
    }

    /// Drives the schedule like `maintain_passes` does, with a build that fails `fails` times and
    /// then succeeds, polling every 100 ms of simulated time. Returns the (time, step) of every
    /// attempt.
    fn run(fails: u32, polls: u32) -> (Vec<(Duration, Step)>, BuildRetry) {
        let t0 = Instant::now();
        let mut r = BuildRetry::default();
        let mut attempts = Vec::new();
        let mut left = fails;
        for i in 0..polls {
            let now = t0 + Duration::from_millis(100) * i;
            match r.step(now) {
                Step::Wait(_) => continue,
                step => {
                    attempts.push((now - t0, step));
                    if left > 0 {
                        left -= 1;
                        r.failed(now);
                    } else {
                        r.succeeded();
                        break;
                    }
                }
            }
        }
        (attempts, r)
    }

    #[test]
    fn short_backoff_then_one_reinit_then_long_backoff() {
        let (attempts, r) = run(6, 2000);
        let ms: Vec<u128> = attempts.iter().map(|(t, _)| t.as_millis()).collect();
        // 0, +0.5 s, +1 s, +2 s (with the re-init), then +30 s twice, then the success at +30 s.
        assert_eq!(ms, vec![0, 500, 1500, 3500, 33500, 63500, 93500]);
        let steps: Vec<Step> = attempts.iter().map(|(_, s)| *s).collect();
        assert_eq!(steps[3], Step::ReinitThenBuild, "the 4th attempt re-initialises NGX");
        assert!(steps.iter().enumerate().all(|(i, s)| i == 3 || *s == Step::Build), "{steps:?}");
        assert!(!r.failing());
    }

    #[test]
    fn a_long_streak_reinitialises_every_fourth_long_attempt_and_never_spins() {
        let (attempts, r) = run(u32::MAX, 20 * 60 * 10); // 20 simulated minutes
        assert!(r.failing());
        let reinit_at: Vec<usize> = attempts.iter().enumerate().filter(|(_, (_, s))| *s == Step::ReinitThenBuild).map(|(i, _)| i).collect();
        assert_eq!(&reinit_at[..3], &[3, 7, 11]);
        // Bounded: after the short phase, attempts are 30 s apart (the old code tried every 250 ms).
        for w in attempts[4..].windows(2) {
            assert_eq!(w[1].0 - w[0].0, BuildRetry::LONG);
        }
        assert!(attempts.len() < 45, "{} attempts in 20 minutes", attempts.len());
    }

    #[test]
    fn a_step_asked_twice_at_the_same_streak_reinitialises_once() {
        let t0 = Instant::now();
        let mut r = BuildRetry::default();
        for i in 0..3 {
            assert!(matches!(r.step(t0 + Duration::from_secs(10 * i)), Step::Build));
            r.failed(t0 + Duration::from_secs(10 * i));
        }
        let now = t0 + Duration::from_secs(40);
        assert_eq!(r.step(now), Step::ReinitThenBuild);
        // The caller did not get to build (say, the frame was below the floor): no second re-init.
        assert_eq!(r.step(now), Step::Build);
        assert_eq!(r.reinits(), 1);
    }

    #[test]
    fn success_ends_the_streak_and_reports_it_once() {
        let t0 = Instant::now();
        let mut r = BuildRetry::default();
        r.failed(t0);
        r.failed(t0);
        assert!(r.failing());
        assert_eq!(r.streak(), 2);
        assert!(matches!(r.step(t0), Step::Wait(_)));
        r.succeeded();
        assert!(!r.failing());
        assert_eq!(r.step(t0), Step::Build, "no wait left over from the streak");
        assert_eq!(r.take_recovered(), Some(2));
        assert_eq!(r.take_recovered(), None);
        // A success with no streak reports nothing.
        r.succeeded();
        assert_eq!(r.take_recovered(), None);
        // A fresh streak starts at the short waits again.
        assert_eq!(r.failed(t0), BuildRetry::SHORT[0]);
    }

    #[test]
    fn fail_inject_parses_and_counts() {
        assert_eq!(FailInject::parse(""), None);
        assert_eq!(FailInject::parse("0"), None);
        assert_eq!(FailInject::parse("x"), None);
        assert_eq!(FailInject::parse("3@"), None);
        let mut f = FailInject::parse(" 2 ").unwrap();
        assert_eq!((f.should_fail(), f.should_fail(), f.should_fail()), (true, true, false));
        let mut f = FailInject::parse("2@1").unwrap();
        assert_eq!(f.remaining(), 2);
        assert_eq!((f.should_fail(), f.should_fail(), f.should_fail(), f.should_fail()), (false, true, true, false));
        assert_eq!(f.remaining(), 0);
    }
}
