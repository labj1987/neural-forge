//! A minimal logging sink: `NEURAL_FORGE_LOG` names a file to append to, otherwise stderr.
//! One handle for the process, not one per call site. Same shape as
//! `neural_forge_layer::logging` — kept as a separate copy rather than a shared crate since
//! it's this small and the two crates otherwise share nothing OS-specific here
//! (`std::env`/`std::fs`/`std::io` are already portable).

use std::fmt::Arguments;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Stderr, Write};
use std::sync::{Mutex, OnceLock};

// Both variants are wrapped in `BufWriter`: `std::fs::File`'s own `Write` impl issues
// one raw OS write syscall per call -- no internal buffering at all. This crate runs
// as a Windows guest binary under Wine/Proton, where a single such syscall against a
// real host-filesystem path has been measured, on real hardware, at ~150-180ms.
// `Sink::Stderr` used to skip the wrapper (reasoned as "already unbuffered and cheap
// to flush") -- wrong: `neural_forge_layer::logging`'s identical, un-wrapped `Stderr` was
// confirmed via `strace -e trace=write` on `lordnikon` (2026-09-10) to fragment a
// single `crate::log!` call into several separate blocking `write()` syscalls (one per
// literal/formatted segment) against a redirected pipe, every single frame -- the same
// risk applies here whenever this binary's stdout/stderr isn't a real terminal (i.e.
// whenever `NEURAL_FORGE_LOG` isn't set and Proton/Steam has redirected it, same as the
// layer's case), so both variants get the same fix rather than assuming Stderr is
// always cheap.
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
// Flushing is still a real syscall -- doing it on every call would defeat the whole
// point of buffering. Flushing every 64th line keeps a live `tail -f` reasonably
// fresh (well under a second behind at any real frame rate) while cutting the syscall
// count by ~64x against the pre-buffering baseline of one per log call.
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

/// Writes one `[neural-forge-helper] ...` line. Never called directly -- use the
/// [`crate::log!`] macro so every call site gets the same prefix and newline handling.
pub fn log(args: Arguments<'_>) {
    let Ok(mut sink) = sink().lock() else { return };
    let _ = writeln!(sink, "[neural-forge-helper] {args}");
    if FLUSH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 64 == 0 {
        let _ = sink.flush();
    }
}

/// Forces an immediate flush regardless of the every-64th-call counter above.
/// Real, confirmed cost of *not* having this (2026-09-10, `lordnikon`): a hang deep
/// inside a one-shot startup step (loading the NGX DLL) left every log line after
/// the very first one sitting in the buffer, never reaching disk, because the
/// process had to be killed (no flush-on-exit, and a `SIGKILL` gives it no chance
/// to run one anyway) before it ever logged 64 lines total -- exactly the situation
/// where the log is most needed. Call this after any one-shot startup milestone
/// (DLL load, feature creation, anything before the steady-state per-frame loop);
/// the buffering this works around exists for that loop's own hot-path calls, which
/// this is not one of.
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
