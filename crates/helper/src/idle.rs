//! How the helper's request loop waits between checks of `seq_req`.
//!
//! While requests are flowing (the layer holds a game submit for every answer in the
//! pre-upscaler path), a sleep between checks is latency on every frame: under Proton a 200 us
//! sleep takes about 255 us, so a request waited about 130 us on average before the helper saw
//! it. So for [`IdlePolicy::spin_for`] after the last request the loop only yields its time
//! slice between checks, and only once nothing has asked for that long does it go back to
//! sleeping (a game in a menu, the effect off, no game at all), which costs nothing.
//!
//! Pure policy, no Win32 or Vulkan, so it builds and tests natively.

use std::time::Duration;

/// What the loop does before its next check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Idle {
    /// Give up the time slice (`SwitchToThread`) and check again: requests are flowing.
    Yield,
    /// Sleep this long: nothing has asked for a while.
    Sleep(Duration),
}

#[derive(Clone, Copy, Debug)]
pub struct IdlePolicy {
    /// How long after the last request the loop keeps yielding instead of sleeping. Long enough
    /// to span the gap between two requests at the frame rates where it matters (a request per
    /// frame at 20+ fps, or per second frame at 40+).
    pub spin_for: Duration,
    /// The sleep once it has gone quiet.
    pub sleep: Duration,
}

impl Default for IdlePolicy {
    fn default() -> Self {
        Self { spin_for: Duration::from_millis(50), sleep: Duration::from_micros(200) }
    }
}

impl IdlePolicy {
    /// `since_request`: time since the loop last saw a request, `None` before the first one.
    pub fn next(&self, since_request: Option<Duration>) -> Idle {
        match since_request {
            Some(t) if t < self.spin_for => Idle::Yield,
            _ => Idle::Sleep(self.sleep),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yields_while_requests_flow_and_sleeps_once_quiet() {
        let p = IdlePolicy::default();
        assert_eq!(p.next(None), Idle::Sleep(Duration::from_micros(200)), "no request yet: sleep");
        assert_eq!(p.next(Some(Duration::ZERO)), Idle::Yield);
        assert_eq!(p.next(Some(Duration::from_millis(17))), Idle::Yield, "a frame's gap at 60 fps");
        assert_eq!(p.next(Some(Duration::from_millis(49))), Idle::Yield);
        assert_eq!(p.next(Some(Duration::from_millis(50))), Idle::Sleep(Duration::from_micros(200)));
        assert_eq!(p.next(Some(Duration::from_secs(3))), Idle::Sleep(Duration::from_micros(200)));
    }

    #[test]
    fn a_zero_window_always_sleeps() {
        let p = IdlePolicy { spin_for: Duration::ZERO, sleep: Duration::from_micros(200) };
        assert_eq!(p.next(Some(Duration::ZERO)), Idle::Sleep(Duration::from_micros(200)));
    }
}
