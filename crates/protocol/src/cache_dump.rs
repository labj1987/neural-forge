//! The K-hold pre-upscaler cache dump's triggers, shared by the layer (which takes them) and the CLI
//! (which sets them): the channel's `preupscale_dump_request` (v17), a trigger file next to the
//! channel, and `NEURAL_FORGE_CACHE_DUMP_AT`. The dump itself is the layer's
//! (`preupscale::cache_dump`); its format is docs/ARCHITECTURE.md's "Pre-upscaler cache dump".

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Holds a request without a count dumps.
pub const DEFAULT_HOLDS: u32 = 16;
/// The most holds one dump takes (each hold is about 40 MB at 1440p's render size).
pub const MAX_HOLDS: u32 = 64;
/// The trigger file's name, in the channel's directory (`/tmp/neural-forge-$UID` by default): that
/// directory is the one a game inside Steam's container is known to see, since it maps the channel.
pub const TRIGGER_FILE: &str = "dump-cache";

/// The trigger file next to the channel at `shm_path`.
pub fn trigger_path(shm_path: &str) -> PathBuf {
    Path::new(shm_path).parent().unwrap_or(Path::new("/")).join(TRIGGER_FILE)
}

/// A hold count as the CLI and the trigger file give it: `None` for anything but 1 to
/// [`MAX_HOLDS`].
pub fn parse_holds(text: &str) -> Option<u32> {
    text.trim().parse::<u32>().ok().filter(|k| (1..=MAX_HOLDS).contains(k))
}

/// The trigger file's content: empty (or only whitespace) for [`DEFAULT_HOLDS`], else a count as
/// [`parse_holds`] takes it. `None`: the file is not a request (it is still removed).
pub fn parse_trigger(text: &str) -> Option<u32> {
    if text.trim().is_empty() {
        Some(DEFAULT_HOLDS)
    } else {
        parse_holds(text)
    }
}

/// `NEURAL_FORGE_CACHE_DUMP_AT=<resume>:<ms>[:K]`: dump K holds (default [`DEFAULT_HOLDS`]) starting
/// `<ms>` after the `<resume>`-th time DLSS frames resumed after a gap, as `NEURAL_FORGE_CAPTURE_AT`
/// counts them (a scripted benchmark's passes).
pub fn parse_at(text: &str) -> Option<(u32, Duration, u32)> {
    let mut parts = text.trim().split(':');
    let resume = parts.next()?.trim().parse().ok()?;
    let ms = parts.next()?.trim().parse().ok()?;
    let holds = match parts.next() {
        Some(k) => parse_holds(k)?,
        None => DEFAULT_HOLDS,
    };
    if parts.next().is_some() {
        return None;
    }
    Some((resume, Duration::from_millis(ms), holds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trigger_file_takes_a_count_or_nothing() {
        assert_eq!(parse_trigger(""), Some(DEFAULT_HOLDS));
        assert_eq!(parse_trigger(" \n"), Some(DEFAULT_HOLDS));
        assert_eq!(parse_trigger("8\n"), Some(8));
        assert_eq!(parse_trigger("64"), Some(64));
        assert_eq!(parse_trigger("65"), None);
        assert_eq!(parse_trigger("0"), None);
        assert_eq!(parse_trigger("lots"), None);
        assert_eq!(parse_trigger("-3"), None);
    }

    #[test]
    fn the_trigger_file_sits_next_to_the_channel() {
        assert_eq!(trigger_path("/tmp/neural-forge-1000/shm.bin"), PathBuf::from("/tmp/neural-forge-1000/dump-cache"));
    }

    #[test]
    fn the_scheduled_dump_takes_a_resume_a_delay_and_an_optional_count() {
        assert_eq!(parse_at("2:15000"), Some((2, Duration::from_millis(15000), DEFAULT_HOLDS)));
        assert_eq!(parse_at("1:500:32"), Some((1, Duration::from_millis(500), 32)));
        assert_eq!(parse_at("1:500:99"), None);
        assert_eq!(parse_at("1"), None);
        assert_eq!(parse_at("1:2:3:4"), None);
        assert_eq!(parse_at("x:2"), None);
    }
}
