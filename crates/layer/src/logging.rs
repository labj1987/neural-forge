//! A minimal logging sink: `NEURALFORGE_LOG` names a file to append to, otherwise stderr.
//! One handle for the process, not one per call site -- opening (or looking up) the
//! sink on every log line would be wasteful on a hot path like the present hook.

use std::fmt::Arguments;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Stderr, Write};
use std::sync::{Mutex, OnceLock};

// Both variants are wrapped in `BufWriter`, not just the file one: this crate's own
// `set_frame_info`/`capture::run` log unconditionally once (now twice, with the added
// timing line) per frame, from inside the game's own `vkQueuePresentKHR` override --
// truly the hottest of hot paths. `NEURALFORGE_LOG` is only ever set for
// `neural-forge-helper.exe` by `neural_forge_supervisor::start()` (confirmed by grep) -- nothing
// sets it for the game's own launch environment, so in every real deployment this
// crate has ever run in, `sink()` falls into `Stderr`, never `File`. A 2026-09-10
// `strace -e trace=write` on a live game process on `lordnikon` confirmed the
// consequence directly at the syscall level: with the un-wrapped `Stderr` this used to
// be, a *single* `crate::log!` call fragmented into several separate blocking
// `write()` syscalls against a pipe (one per literal/formatted segment -- `Write`'s
// `write_fmt` doesn't coalesce them), every frame, forever -- a real, syscall-level-
// verified explanation for the multi-hundred-ms/frame stalls this session spent a long
// time chasing through GPU timing and helper-side-only logging fixes that (correctly
// diagnosed the same *pattern* elsewhere, but) touched the wrong sink to matter here.
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
        let sink = std::env::var("NEURALFORGE_LOG")
            .ok()
            .filter(|p| !p.is_empty() && neural_forge_protocol::isolated_path(p))
            .and_then(|path| OpenOptions::new().create(true).append(true).open(path).ok())
            .map(|f| Sink::File(BufWriter::new(f)))
            .unwrap_or_else(|| Sink::Stderr(BufWriter::new(std::io::stderr())));
        Mutex::new(sink)
    })
}

/// Writes one `[neuralforge-layer] ...` line. Never called directly -- use the
/// [`crate::log!`] macro so every call site gets the same prefix and newline handling.
pub fn log(args: Arguments<'_>) {
    let Ok(mut sink) = sink().lock() else { return };
    let _ = writeln!(sink, "[neuralforge-layer] {args}");
    if FLUSH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 64 == 0 {
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

#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        $crate::logging::log(format_args!($($arg)*))
    };
}
