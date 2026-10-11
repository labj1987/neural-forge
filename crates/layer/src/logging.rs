//! A minimal logging sink: `NEURAL_FORGE_LOG` names a file to append to, otherwise stderr.
//! One handle for the process, not one per call site -- opening (or looking up) the
//! sink on every log line would be wasteful on a hot path like the present hook.

use std::fmt::Arguments;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Stderr, Write};
use std::sync::{Mutex, OnceLock};

// Both variants are wrapped in `BufWriter`, not just the file one: logging from inside the
// game's own `vkQueuePresentKHR` override is the hottest of hot paths. Nothing sets `NEURAL_FORGE_LOG` for the game's own launch
// environment unless asked, so in a normal deployment `sink()` falls into `Stderr`, never `File`. A 2026-09-10
// `strace -e trace=write` on a live game process on the test machine confirmed the
// consequence directly at the syscall level: with the un-wrapped `Stderr` this used to
// be, a *single* `crate::log!` call fragmented into several separate blocking
// `write()` syscalls against a pipe (one per literal/formatted segment -- `Write`'s
// `write_fmt` doesn't coalesce them), every frame, forever -- a real, syscall-level-
// verified cause of multi-hundred-ms/frame stalls.
enum Sink {
    File(BufWriter<File>),
    Stderr(BufWriter<Stderr>),
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Sink::File(f) => f.write(buf),
            Sink::Stderr(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Sink::File(f) => f.flush(),
            Sink::Stderr(s) => s.flush(),
        }
    }
}

static SINK: OnceLock<Mutex<Sink>> = OnceLock::new();
// Flushing is still a real syscall -- doing it on every call would defeat buffering.
// Every 64th line keeps a live `tail -f`/console reasonably fresh without paying for
// a syscall on every single presented frame.
static FLUSH_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn sink() -> &'static Mutex<Sink> {
    SINK.get_or_init(|| {
        let sink = neural_forge_protocol::env::var("NEURAL_FORGE_LOG")
            .filter(|p| !p.is_empty() && neural_forge_protocol::isolated_path(p))
            .and_then(|path| OpenOptions::new().create(true).append(true).open(path).ok())
            .map(|f| Sink::File(BufWriter::new(f)))
            .unwrap_or_else(|| Sink::Stderr(BufWriter::new(std::io::stderr())));
        Mutex::new(sink)
    })
}

/// Writes one `[neural-forge-layer] ...` line. Never called directly -- use the
/// [`crate::log!`] macro so every call site gets the same prefix and newline handling.
pub fn log(args: Arguments<'_>) {
    // `NEURAL_FORGE_LOG_TIME=1`: each line starts with the wall-clock time (Unix seconds, milliseconds),
    // to line the log up with the kernel's (an Xid's time).
    static TIME: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_LOG_TIME").as_deref() == Some("1"));
    let Ok(mut sink) = sink().lock() else { return };
    let _ = if *TIME {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        writeln!(sink, "[neural-forge-layer] {}.{:03} {args}", t.as_secs(), t.subsec_millis())
    } else {
        writeln!(sink, "[neural-forge-layer] {args}")
    };
    if FLUSH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(64) {
        let _ = sink.flush();
    }
}

/// Forces a flush outside the modulo-64 throttle above. Confirmed missing and worth
/// having, 2026-09-11: a process killed by a signal (e.g. `timeout`'s default SIGTERM)
/// never runs the throttle's own eventual flush, silently losing every buffered line
/// since the last one -- including one-time milestones like device/swapchain creation
/// that matter far more than the steady-state per-frame logging the throttle exists
/// to protect. Call this after any one-shot milestone, never from the per-frame hot
/// path itself.
pub fn flush() {
    let Ok(mut sink) = sink().lock() else { return };
    let _ = sink.flush();
}

/// The state log's file (`neural_forge_protocol::state_log`), decided once: only in a process that
/// opted in (`NEURAL_FORGE_ENABLE=1`, read directly rather than through `layer_enabled`, whose own
/// first evaluation can write the duplicate-copy event), and never in this crate's tests, which must
/// not touch the real XDG state dir.
static STATE_LOG: std::sync::LazyLock<Option<std::path::PathBuf>> = std::sync::LazyLock::new(|| {
    if cfg!(test) || !neural_forge_protocol::env::flag("NEURAL_FORGE_ENABLE") {
        return None;
    }
    neural_forge_protocol::state_log::path(std::env::var("XDG_STATE_HOME").ok().as_deref(), std::env::var("HOME").ok().as_deref())
});

/// Appends one transition to the state log (see `neural_forge_protocol::state_log` for the grammar
/// and the file). Synchronous: an open, a `write` and a close. Never call it per frame: only where
/// the layer's state changes, and only once per change (every caller is latched or deduplicated).
/// Use the [`crate::event!`] macro.
pub fn event(kind: &'static str, message: &str) {
    let Some(path) = STATE_LOG.as_deref() else { return };
    // Serialises this process's writers, so a rotation and an append never race inside it.
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let line = neural_forge_protocol::state_log::format_line(now, std::process::id(), crate::ownership::process_name(), kind, message);
    if let Err(e) = neural_forge_protocol::state_log::append(path, &line, neural_forge_protocol::state_log::ROTATE_BYTES) {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log(format_args!("[layer] cannot write the state log {}: {e}", path.display()));
        }
    }
}

/// `event!(KIND, "format", args...)`: one state-log line (see [`crate::logging::event`]).
#[macro_export]
macro_rules! event {
    ($kind:expr, $($arg:tt)*) => {
        $crate::logging::event($kind, &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        $crate::logging::log(format_args!($($arg)*))
    };
}
