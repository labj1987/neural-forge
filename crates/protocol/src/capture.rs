//! Frame captures through the channel (v18): the GUI's capture button and `shmctl capture` ask, the
//! layer of the game that holds the channel serves the request and says what became of it.
//!
//! A request is one word, `capture_request = (id << 16) | frames`, written only when no other request
//! is pending ([`request`]), so a request and its id can never be torn apart by a second requester.
//! The layer zeroes the word when it takes the request and publishes an [`Outcome`] for that id:
//! running (a series in progress), written (with the files' paths), failed or cancelled, with a reason
//! code saying why a pair is not the effect's result or why nothing was written. A requester only
//! trusts an outcome that carries its own id.

use crate::ShmHeader;
use std::sync::atomic::Ordering;

/// The most frames one request may ask for (a series; one is the one-shot pair). At 4K a pair is
/// about 66 MB in flight, and the series reads the frame back twice per present.
pub const MAX_FRAMES: u32 = 1000;
/// Request ids run 1 to this and wrap; 0 is never handed out.
pub const MAX_ID: u32 = 0xffff;
/// The outcome's text field: the written files' paths, one per line.
pub const PATH_BYTES: usize = 1024;
/// Words in the header's result block: id, state, reason, done, total, composed, then the paths.
pub const RESULT_WORDS: usize = 6 + PATH_BYTES / 4;

const FRAMES_MASK: u32 = 0xffff;
const _: () = assert!(MAX_FRAMES <= FRAMES_MASK && PATH_BYTES.is_multiple_of(4));

/// What became of a request.
pub mod state {
    /// Nothing taken since the game's layer opened the channel.
    pub const IDLE: u32 = 0;
    /// Taken; a series is still capturing, or the files are still being written.
    pub const RUNNING: u32 = 1;
    /// Every file written (see the reason for a pair that is not the effect's result).
    pub const WRITTEN: u32 = 2;
    pub const FAILED: u32 = 3;
    /// Stopped by [`super::cancel`]; the pairs written before are kept.
    pub const CANCELLED: u32 = 4;
}

/// Why: for a written capture, whether its "composited" image really is the effect's result; for a
/// failed one, what stopped it. Values are a closed set; anything else reads as [`NONE`].
pub mod reason {
    pub const NONE: u32 = 0;
    /// The effect is switched off (the Enabled toggle or "Apply model edit"): both images are the game's frame.
    pub const EFFECT_OFF: u32 = 1;
    /// The model is unavailable this session: both images are the game's frame.
    pub const MODEL_UNAVAILABLE: u32 = 2;
    /// No model server is running: both images are the game's frame.
    pub const NO_MODEL_SERVER: u32 = 3;
    /// The model did not answer the frame in time, so it went out untouched.
    pub const NO_ANSWER: u32 = 4;
    /// The model runs before the upscaler: both images are the presented frame, which has its edit.
    pub const BEFORE_UPSCALER: u32 = 5;
    /// The effect did not compose the frame (or every frame of the series).
    pub const NOT_COMPOSED: u32 = 6;
    /// Only some frames of the series were composed by the effect.
    pub const PARTLY_COMPOSED: u32 = 7;
    /// The game's swapchain cannot be read back.
    pub const UNREADABLE: u32 = 10;
    /// An HDR or 10-bit swapchain: only 8-bit SDR frames are captured.
    pub const UNSUPPORTED_FORMAT: u32 = 11;
    /// Reading the frame back from the GPU failed.
    pub const READBACK_FAILED: u32 = 12;
    /// The files could not be written (see the game's log).
    pub const WRITE_FAILED: u32 = 13;
    /// Another capture was still running.
    pub const BUSY: u32 = 14;
    /// The request asked for no frames or more than [`super::MAX_FRAMES`].
    pub const BAD_COUNT: u32 = 15;
    /// The game ended (or its device went away) before the capture finished.
    pub const GAME_ENDED: u32 = 16;
    /// `NEURAL_FORGE_PREUPSCALE=dump`: DLSS's input was written instead of a pair.
    pub const DLSS_INPUT_DUMP: u32 = 17;

    /// One sentence for the GUI and `shmctl`, or `None` for [`NONE`] and unknown codes.
    pub fn describe(code: u32) -> Option<&'static str> {
        Some(match code {
            EFFECT_OFF => "the effect is switched off, so both images are the game's frame",
            MODEL_UNAVAILABLE => "the model is unavailable this session, so both images are the game's frame",
            NO_MODEL_SERVER => "no model server is running, so both images are the game's frame",
            NO_ANSWER => "the model did not answer in time, so the frame went out untouched",
            BEFORE_UPSCALER => "the model runs before the upscaler, so both images are the presented frame with its edit",
            NOT_COMPOSED => "the effect did not change the frame",
            PARTLY_COMPOSED => "the effect changed only some of the frames",
            UNREADABLE => "the game's swapchain cannot be read back",
            UNSUPPORTED_FORMAT => "the game presents HDR or 10-bit frames, and only 8-bit frames are captured",
            READBACK_FAILED => "reading the frame back from the GPU failed",
            WRITE_FAILED => "the files could not be written (the game's log says why)",
            BUSY => "another capture was still running",
            BAD_COUNT => "the request asked for no frames or too many",
            GAME_ENDED => "the game closed before the capture finished",
            DLSS_INPUT_DUMP => "the layer runs NEURAL_FORGE_PREUPSCALE=dump, so DLSS's input was written instead of a pair",
            _ => return None,
        })
    }
}

/// A taken or pending request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub id: u32,
    pub frames: u32,
}

impl Request {
    /// Whether the layer may serve it: 1 to [`MAX_FRAMES`] frames.
    pub fn valid(&self) -> bool {
        (1..=MAX_FRAMES).contains(&self.frames)
    }
}

/// The request word for `id` and `frames` (both already in range).
pub const fn encode_request(id: u32, frames: u32) -> u32 {
    ((id & MAX_ID) << 16) | (frames & FRAMES_MASK)
}

/// The request in a `capture_request` word, `None` for 0.
pub const fn decode_request(word: u32) -> Option<Request> {
    if word == 0 {
        None
    } else {
        Some(Request { id: word >> 16, frames: word & FRAMES_MASK })
    }
}

/// Why [`request`] asked for nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// No frames, or more than [`MAX_FRAMES`].
    BadCount,
    /// Another request is still waiting for a game to take it.
    Pending,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::BadCount => write!(f, "a capture takes 1 to {MAX_FRAMES} frames"),
            RequestError::Pending => write!(f, "another capture is still waiting for the game"),
        }
    }
}

/// Asks the game's layer for `frames` frames (1: the one-shot pair). Returns the request's id, which
/// its [`Outcome`] will carry.
pub fn request(hdr: &ShmHeader, frames: u32) -> Result<u32, RequestError> {
    if !(1..=MAX_FRAMES).contains(&frames) {
        return Err(RequestError::BadCount);
    }
    if hdr.capture_request.load(Ordering::Acquire) != 0 {
        return Err(RequestError::Pending);
    }
    let previous = hdr
        .capture_next_id
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n % MAX_ID + 1))
        .unwrap_or_default();
    let id = previous % MAX_ID + 1;
    // Release: a cancel of an earlier id written before this request is seen before it.
    hdr.capture_request
        .compare_exchange(0, encode_request(id, frames), Ordering::Release, Ordering::Relaxed)
        .map(|_| id)
        .map_err(|_| RequestError::Pending)
}

/// The request waiting for the layer, if any (not taken).
pub fn pending(hdr: &ShmHeader) -> Option<Request> {
    decode_request(hdr.capture_request.load(Ordering::Acquire))
}

/// What [`cancel`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cancelled {
    /// The request was still waiting: it is gone, and no game will ever take it.
    Withdrawn,
    /// The layer had taken it: it stops at its next present and says so in the outcome.
    Signalled,
}

/// Stops request `id`: withdraws it while it is still waiting for a game, else tells the layer to
/// stop it.
pub fn cancel(hdr: &ShmHeader, id: u32) -> Cancelled {
    let word = hdr.capture_request.load(Ordering::Acquire);
    if decode_request(word).is_some_and(|r| r.id == id) && hdr.capture_request.compare_exchange(word, 0, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
        return Cancelled::Withdrawn;
    }
    hdr.capture_cancel.store(id, Ordering::Release);
    Cancelled::Signalled
}

/// Takes the waiting request when `accept` says this caller serves it (the layer: one present path
/// serves one-shots, another series). Zeroes the word in the same atomic step, so a request is taken
/// once.
pub fn take(hdr: &ShmHeader, accept: impl Fn(&Request) -> bool) -> Option<Request> {
    let word = hdr.capture_request.load(Ordering::Acquire);
    let request = decode_request(word).filter(|r| accept(r))?;
    hdr.capture_request.compare_exchange(word, 0, Ordering::AcqRel, Ordering::Relaxed).ok().map(|_| request)
}

/// Whether the requester asked for capture `id` to stop. Id 0 is a capture the layer started on its
/// own (`NEURAL_FORGE_CAPTURE_AT`, never handed out by [`request`]): nothing can name it, and
/// `capture_cancel`'s 0 means "none", so it is never cancelled.
pub fn cancel_requested(hdr: &ShmHeader, id: u32) -> bool {
    id != 0 && hdr.capture_cancel.load(Ordering::Acquire) == id
}

/// What became of one request, as the layer publishes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub id: u32,
    /// A [`state`] value.
    pub state: u32,
    /// A [`reason`] value.
    pub reason: u32,
    /// Pairs written so far.
    pub done: u32,
    /// Pairs asked for.
    pub total: u32,
    /// How many of the written pairs the effect composed.
    pub composed: u32,
    /// The written files (a one-shot's two images, a series' directory), one per line; cut at
    /// [`PATH_BYTES`] - 1 bytes.
    pub paths: String,
}

impl Outcome {
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.paths.lines().filter(|l| !l.is_empty())
    }

    pub fn finished(&self) -> bool {
        matches!(self.state, state::WRITTEN | state::FAILED | state::CANCELLED)
    }

    /// One line for `shmctl` and the GUI's toast.
    pub fn message(&self) -> String {
        let why = reason::describe(self.reason);
        let frames = if self.total > 1 { format!(" ({} of {} frames)", self.done, self.total) } else { String::new() };
        match self.state {
            state::RUNNING => format!("Capturing{frames}"),
            state::WRITTEN => match why {
                Some(why) => format!("Captured{frames}, but {why}"),
                None => format!("Captured{frames}"),
            },
            state::CANCELLED => format!("Capture cancelled{frames}"),
            state::FAILED => format!("Capture failed: {}", why.unwrap_or("the layer did not say why")),
            _ => "No capture yet".into(),
        }
    }
}

/// Publishes `outcome` (the layer, which writes it from one place at a time).
pub fn publish(hdr: &ShmHeader, outcome: &Outcome) {
    let mut words = [0u32; RESULT_WORDS];
    words[..6].copy_from_slice(&[outcome.id, outcome.state, outcome.reason, outcome.done, outcome.total, outcome.composed]);
    let bytes = outcome.paths.as_bytes();
    // Cut at a character boundary, leaving room for the terminating NUL.
    let mut n = bytes.len().min(PATH_BYTES - 1);
    while n > 0 && !outcome.paths.is_char_boundary(n) {
        n -= 1;
    }
    for (i, chunk) in bytes[..n].chunks(4).enumerate() {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words[6 + i] = u32::from_le_bytes(word);
    }
    hdr.set_capture_result_words(&words);
}

/// The latest published outcome, `None` when none was ever published (or a write kept racing).
pub fn result(hdr: &ShmHeader) -> Option<Outcome> {
    let words = hdr.capture_result_words()?;
    let bytes: Vec<u8> = words[6..].iter().flat_map(|w| w.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let outcome = Outcome {
        id: words[0],
        state: words[1],
        reason: words[2],
        done: words[3],
        total: words[4],
        composed: words[5],
        paths: String::from_utf8_lossy(&bytes[..end]).into_owned(),
    };
    (outcome.state != state::IDLE).then_some(outcome)
}

/// A new game's layer opening the channel: a request left waiting from before belongs to no game that
/// is running now (it would fire, unasked, at this game's first frame), and an outcome left running
/// is a game that went away mid-capture. The same holds for a cache-dump request
/// (`shmctl dump-cache`) that no pre-upscaler hold has taken yet.
pub fn reset_session(hdr: &ShmHeader) {
    hdr.capture_request.store(0, Ordering::Release);
    hdr.preupscale_dump_request.store(0, Ordering::Relaxed);
    hdr.capture_cancel.store(0, Ordering::Relaxed);
    if let Some(mut stale) = result(hdr).filter(|o| o.state == state::RUNNING) {
        stale.state = state::FAILED;
        stale.reason = reason::GAME_ENDED;
        publish(hdr, &stale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_drops_requests_left_for_an_earlier_game() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.preupscale_dump_request.store(32, Ordering::Relaxed);
        h.capture_request.store(5, Ordering::Relaxed);
        reset_session(&h);
        assert_eq!(h.preupscale_dump_request.load(Ordering::Relaxed), 0);
        assert_eq!(h.capture_request.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_request_carries_its_id_and_count_in_one_word() {
        let h = ShmHeader::default();
        let id = request(&h, 1).unwrap();
        assert_eq!(pending(&h), Some(Request { id, frames: 1 }));
        assert_eq!(request(&h, 1), Err(RequestError::Pending), "one request waits at a time");
        assert_eq!(take(&h, |r| r.frames > 1), None, "a caller that does not serve it leaves it");
        assert_eq!(take(&h, |_| true), Some(Request { id, frames: 1 }));
        assert_eq!(take(&h, |_| true), None, "taken once");
        let next = request(&h, MAX_FRAMES).unwrap();
        assert_ne!(next, id);
        assert_eq!(take(&h, |_| true).map(|r| r.frames), Some(MAX_FRAMES));
    }

    #[test]
    fn counts_are_bounded_and_ids_skip_zero() {
        let h = ShmHeader::default();
        assert_eq!(request(&h, 0), Err(RequestError::BadCount));
        assert_eq!(request(&h, MAX_FRAMES + 1), Err(RequestError::BadCount));
        h.capture_next_id.store(MAX_ID, Ordering::Relaxed);
        assert_eq!(request(&h, 2), Ok(1), "wraps past the largest id to 1, never 0");
        assert!(!Request { id: 1, frames: 0 }.valid() && !Request { id: 1, frames: MAX_FRAMES + 1 }.valid());
        assert_eq!(decode_request(0), None);
        assert_eq!(decode_request(encode_request(MAX_ID, MAX_FRAMES)), Some(Request { id: MAX_ID, frames: MAX_FRAMES }));
    }

    #[test]
    fn cancel_withdraws_a_waiting_request_and_signals_a_taken_one() {
        let h = ShmHeader::default();
        let id = request(&h, 5).unwrap();
        assert_eq!(cancel(&h, id), Cancelled::Withdrawn);
        assert_eq!(pending(&h), None, "a withdrawn request never reaches a game");
        assert!(!cancel_requested(&h, id));
        let id = request(&h, 5).unwrap();
        take(&h, |_| true).unwrap();
        assert_eq!(cancel(&h, id), Cancelled::Signalled);
        assert!(cancel_requested(&h, id));
        // Cancelling an older id never withdraws a newer request.
        let newer = request(&h, 1).unwrap();
        assert_eq!(cancel(&h, id), Cancelled::Signalled);
        assert_eq!(pending(&h).map(|r| r.id), Some(newer));
    }

    #[test]
    fn an_outcome_round_trips_and_long_paths_are_cut_on_a_character() {
        let h = ShmHeader::default();
        assert_eq!(result(&h), None);
        let outcome = Outcome { id: 9, state: state::WRITTEN, reason: reason::NO_ANSWER, done: 1, total: 1, composed: 0, paths: "/a/1-original.png\n/a/1-composited.png".into() };
        publish(&h, &outcome);
        let read = result(&h).unwrap();
        assert_eq!(read, outcome);
        assert_eq!(read.paths().collect::<Vec<_>>(), ["/a/1-original.png", "/a/1-composited.png"]);
        assert!(read.finished());
        assert!(read.message().contains("did not answer"), "{}", read.message());
        let long = Outcome { paths: "é".repeat(PATH_BYTES), ..outcome };
        publish(&h, &long);
        let cut = result(&h).unwrap().paths;
        assert!(cut.len() < PATH_BYTES && cut.chars().all(|c| c == 'é'), "{}", cut.len());
    }

    #[test]
    fn a_new_session_drops_a_waiting_request_and_ends_a_running_capture() {
        let h = ShmHeader::default();
        publish(&h, &Outcome { id: 3, state: state::RUNNING, total: 10, done: 4, ..Outcome::default() });
        request(&h, 1).unwrap();
        h.capture_cancel.store(3, Ordering::Relaxed);
        reset_session(&h);
        assert_eq!(pending(&h), None, "a request from before this game never fires on it");
        assert!(!cancel_requested(&h, 3));
        let ended = result(&h).unwrap();
        assert_eq!((ended.state, ended.reason, ended.done), (state::FAILED, reason::GAME_ENDED, 4));
        // A finished outcome is left as it was.
        let written = Outcome { id: 4, state: state::WRITTEN, done: 1, total: 1, composed: 1, ..Outcome::default() };
        publish(&h, &written);
        reset_session(&h);
        assert_eq!(result(&h), Some(written));
    }

    #[test]
    fn every_reason_has_a_sentence() {
        for code in 1..=reason::DLSS_INPUT_DUMP {
            let known = reason::describe(code).is_some();
            assert_eq!(known, !(8..=9).contains(&code), "{code}");
        }
        assert_eq!(reason::describe(reason::NONE), None);
    }

    #[test]
    fn a_capture_the_layer_started_itself_is_never_cancelled() {
        // `NEURAL_FORGE_CAPTURE_AT`'s series carries id 0, and `capture_cancel`'s 0 means "none".
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(!cancel_requested(&h, 0), "a fresh header cancels nothing");
        h.capture_cancel.store(5, Ordering::Relaxed);
        assert!(cancel_requested(&h, 5));
        assert!(!cancel_requested(&h, 0));
    }
}
