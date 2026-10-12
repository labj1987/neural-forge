//! A minimal logging sink: `NEURAL_FORGE_LOG` names a file to append to, otherwise stderr.
//! One writer thread for the process: a log line is formatted on the calling thread and queued,
//! so a slow sink (a stderr pipe nobody drains, a full or slow disk) never blocks the game's
//! present or submit threads. The file is rotated to `<path>.1` at [`ROTATE_BYTES`].

use std::fmt::Arguments;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Stderr, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// The log file's size at which it is moved to `<path>.1` (replacing an older `.1`) and started anew.
const ROTATE_BYTES: u64 = 64 << 20;
/// Lines queued for the writer at most; past that a line is dropped (and the drop counted) rather than
/// the caller waiting.
const QUEUE_LINES: usize = 4096;
/// How long [`flush`] waits for the writer to have written everything queued before it.
const FLUSH_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

// Both variants are wrapped in `BufWriter`: a 2026-09-10 `strace -e trace=write` on a live game
// showed an un-wrapped `Stderr` fragmenting a single line into several blocking `write()`
// syscalls (one per formatted segment).
enum Sink {
    File { file: BufWriter<File>, path: PathBuf, written: u64 },
    Stderr(BufWriter<Stderr>),
}

impl Sink {
    fn open() -> Self {
        neural_forge_protocol::env::var("NEURAL_FORGE_LOG")
            .filter(|p| !p.is_empty() && neural_forge_protocol::isolated_path(p))
            .and_then(|path| Self::file(PathBuf::from(path)))
            .unwrap_or_else(|| Sink::Stderr(BufWriter::new(std::io::stderr())))
    }

    fn file(path: PathBuf) -> Option<Self> {
        let file = OpenOptions::new().create(true).append(true).open(&path).ok()?;
        let written = file.metadata().map_or(0, |m| m.len());
        Some(Sink::File { file: BufWriter::new(file), path, written })
    }

    /// Writes one line, rotating the file first when it has reached `limit`.
    fn line(&mut self, line: &str, limit: u64) {
        if let Sink::File { file, path, written } = self {
            if *written >= limit {
                let _ = file.flush();
                let mut rotated = path.clone().into_os_string();
                rotated.push(".1");
                let _ = std::fs::rename(&*path, &rotated);
                if let Ok(fresh) = OpenOptions::new().create(true).append(true).open(&*path) {
                    *file = BufWriter::new(fresh);
                    *written = 0;
                }
            }
            *written += line.len() as u64;
        }
        let _ = match self {
            Sink::File { file, .. } => file.write_all(line.as_bytes()),
            Sink::Stderr(s) => s.write_all(line.as_bytes()),
        };
    }

    fn flush(&mut self) {
        let _ = match self {
            Sink::File { file, .. } => file.flush(),
            Sink::Stderr(s) => s.flush(),
        };
    }
}

/// A flush request: set and notified once everything queued before it is written.
type Done = Arc<(Mutex<bool>, Condvar)>;

enum Message {
    Line(String),
    Flush(Done),
}

enum Logger {
    Queued(SyncSender<Message>),
    /// No writer thread could be started: the caller writes.
    Direct(Mutex<Sink>),
}

static LOGGER: OnceLock<Logger> = OnceLock::new();
/// Lines dropped because the queue was full, said by the writer once it catches up.
static DROPPED: AtomicU64 = AtomicU64::new(0);

fn logger() -> &'static Logger {
    LOGGER.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel::<Message>(QUEUE_LINES);
        let spawned = std::thread::Builder::new().name("nf-log".into()).spawn(move || {
            let mut sink = Sink::open();
            while let Ok(message) = receiver.recv() {
                write_message(&mut sink, message);
                // Everything queued is written in one go; the file sees a flush when the queue is
                // empty, never per line.
                while let Ok(message) = receiver.try_recv() {
                    write_message(&mut sink, message);
                }
                sink.flush();
            }
        });
        match spawned {
            Ok(_) => Logger::Queued(sender),
            Err(_) => Logger::Direct(Mutex::new(Sink::open())),
        }
    })
}

fn write_message(sink: &mut Sink, message: Message) {
    let dropped = DROPPED.swap(0, Ordering::Relaxed);
    if dropped > 0 {
        sink.line(&format!("[neural-forge-layer] [log] {dropped} lines dropped: the log could not keep up\n"), ROTATE_BYTES);
    }
    match message {
        Message::Line(line) => sink.line(&line, ROTATE_BYTES),
        Message::Flush(done) => {
            sink.flush();
            let (flag, wake) = &*done;
            *flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
            wake.notify_all();
        }
    }
}

/// Writes one `[neural-forge-layer] ...` line. Never called directly -- use the
/// [`crate::log!`] macro so every call site gets the same prefix and newline handling.
pub fn log(args: Arguments<'_>) {
    // `NEURAL_FORGE_LOG_TIME=1`: each line starts with the wall-clock time (Unix seconds, milliseconds),
    // to line the log up with the kernel's (an Xid's time).
    static TIME: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| neural_forge_protocol::env::var("NEURAL_FORGE_LOG_TIME").as_deref() == Some("1"));
    let line = if *TIME {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        format!("[neural-forge-layer] {}.{:03} {args}\n", t.as_secs(), t.subsec_millis())
    } else {
        format!("[neural-forge-layer] {args}\n")
    };
    match logger() {
        Logger::Queued(sender) => {
            if let Err(TrySendError::Full(_)) = sender.try_send(Message::Line(line)) {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        Logger::Direct(sink) => {
            if let Ok(mut sink) = sink.lock() {
                sink.line(&line, ROTATE_BYTES);
            }
        }
    }
}

/// Waits (at most [`FLUSH_WAIT`]) until every line logged before it is written out. Call this after
/// any one-shot milestone: a process killed by a signal (e.g. `timeout`'s default SIGTERM) loses
/// what was still queued -- including one-time milestones like device/swapchain creation that matter
/// far more than steady-state logging.
pub fn flush() {
    match logger() {
        Logger::Queued(sender) => {
            let done: Done = Arc::default();
            if sender.try_send(Message::Flush(done.clone())).is_err() {
                return;
            }
            let (flag, wake) = &*done;
            let guard = flag.lock().unwrap_or_else(|e| e.into_inner());
            let _ = wake.wait_timeout_while(guard, FLUSH_WAIT, |written| !*written);
        }
        Logger::Direct(sink) => {
            if let Ok(mut sink) = sink.lock() {
                sink.flush();
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The file sink moves itself to `<path>.1` at the limit and starts anew: a long session's log
    /// is bounded to about twice the limit.
    #[test]
    fn the_log_file_is_rotated_at_its_limit() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!("target/test-scratch/log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layer.log");
        let mut sink = Sink::file(path.clone()).unwrap();
        for i in 0..10 {
            sink.line(&format!("line {i:02}\n"), 30);
        }
        sink.flush();
        let current = std::fs::read_to_string(&path).unwrap();
        let rotated = std::fs::read_to_string(dir.join("layer.log.1")).unwrap();
        assert!(current.len() <= 40 && rotated.len() <= 40, "{current:?} {rotated:?}");
        assert!(current.ends_with("line 09\n"), "{current:?}");
        assert!(rotated.contains("line 0") && !rotated.contains("line 09"), "{rotated:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A flush returns once what was logged before it is written, and never waits past its bound.
    #[test]
    fn flush_is_bounded() {
        log(format_args!("[test] a line before the flush"));
        let t = std::time::Instant::now();
        flush();
        assert!(t.elapsed() < FLUSH_WAIT + std::time::Duration::from_millis(500));
    }
}
