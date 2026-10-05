//! The wire's two request/response slots (`docs/PROTOCOL_V3_DESIGN.md`) as a type: a slot is
//! one of exactly two, so everything that picks a slot's fields, regions or pending state takes
//! a [`Slot`], and an index that is neither 0 nor 1 cannot be passed at all. A raw index (a
//! command-line argument, a loop counter) becomes one through [`Slot::new`], which refuses
//! anything else.

/// One of the protocol's two independent request/response slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Slot {
    /// Slot 0: `seq_req`/`seq_resp`, the first proxy and answer regions. The only slot with the
    /// answered-size echo, `seq_eval` and motion estimation.
    Primary,
    /// Slot 1: `seq_req_b`/`seq_resp_b`, the second proxy and answer regions.
    Secondary,
}

impl Slot {
    /// How many slots there are: the length of every per-slot array.
    pub const COUNT: usize = 2;
    /// Both slots, in index order.
    pub const ALL: [Slot; Slot::COUNT] = [Slot::Primary, Slot::Secondary];

    /// The slot with this index; `None` for anything but 0 and 1.
    pub const fn new(index: usize) -> Option<Self> {
        match index {
            0 => Some(Slot::Primary),
            1 => Some(Slot::Secondary),
            _ => None,
        }
    }

    /// 0 or 1: the position in a per-slot array of [`Slot::COUNT`] elements.
    pub const fn index(self) -> usize {
        match self {
            Slot::Primary => 0,
            Slot::Secondary => 1,
        }
    }
}

/// Prints the index, as the logs always have ("slot 0", "slot 1").
impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.index())
    }
}

/// The request number that follows `current` on a slot's `seq_req`. The counter is a `u32` and
/// wraps; 0 is never issued (it is what a freshly initialised header holds, and what "no request
/// yet" reads as in `seq_resp`, `seq_ok` and `seq_eval`), so after `u32::MAX` comes 1.
///
/// A slot has one request outstanding at a time, and the helper answers by storing the request's
/// own number in `seq_resp`: the request is answered when `seq_resp == request`, never "at least".
pub const fn next_request(current: u32) -> u32 {
    match current.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_two_slots_can_be_constructed() {
        assert_eq!(Slot::new(0), Some(Slot::Primary));
        assert_eq!(Slot::new(1), Some(Slot::Secondary));
        for index in [2, 3, 255, usize::MAX] {
            assert_eq!(Slot::new(index), None, "slot {index} does not exist");
        }
        for (index, slot) in Slot::ALL.into_iter().enumerate() {
            assert_eq!(slot.index(), index);
            assert_eq!(Slot::new(index), Some(slot));
            assert_eq!(slot.to_string(), index.to_string());
        }
        assert_eq!(Slot::ALL.len(), Slot::COUNT);
    }

    #[test]
    fn request_numbers_wrap_past_zero() {
        assert_eq!(next_request(0), 1);
        assert_eq!(next_request(41), 42);
        assert_eq!(next_request(u32::MAX - 1), u32::MAX);
        assert_eq!(next_request(u32::MAX), 1, "0 means no request and is skipped");
        let mut seq = u32::MAX - 2;
        let issued: Vec<u32> = (0..5).map(|_| { seq = next_request(seq); seq }).collect();
        assert_eq!(issued, [u32::MAX - 1, u32::MAX, 1, 2, 3]);
    }
}
