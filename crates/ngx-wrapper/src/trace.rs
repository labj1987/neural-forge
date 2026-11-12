//! The call trace (`NEURAL_FORGE_NGX_TRACE`; docs/PROTON_NGX.md, "The call trace"): a debugging aid that logs every
//! call into the wrapper's 75 exports with its thread, its first four arguments, its result and how long NVIDIA's
//! function took (entry and exit lines), and the steps of the wrapper's load, so a game that stalls or crashes with
//! the wrapper in shows which NGX call was running, on which thread, and for how long.
//!
//! The file is a fixed-size memory-mapped ring of text lines: writing a line is a copy into shared memory, with no
//! system call and no lock, and the kernel keeps the pages when the process dies, so the last lines before a crash
//! are there. A fixed-width header line names about where the next line goes (a hint: writers on several threads
//! rewrite it without ordering, so it can lag a line). Every line starts with its time, so `tr -d '\0' < FILE | sort
//! -n` is the way to put them in order, wrapped or not. Neither a line nor the header allocates or takes a lock.
//!
//! This module is the part that needs no Windows (tested on Linux); `win.rs` maps the file and `exports.rs` holds the
//! thunks that send passthrough calls through [`crate::win`]'s tracer.

use core::fmt::{self, Write as _};

/// The environment variable naming the trace file (a Windows path, or a Unix one: [`windows_path`]).
pub const ENV: &str = "NEURAL_FORGE_NGX_TRACE";

/// The whole file: the header line, then the ring.
pub const FILE_BYTES: usize = 64 << 20;
/// The header line's fixed width, newline included.
pub const HEADER_BYTES: usize = 128;
/// The ring of lines after the header.
pub const DATA_BYTES: usize = FILE_BYTES - HEADER_BYTES;
/// The longest line kept (a longer one is cut, its newline kept): room for the longest evaluate line the text log
/// writes (about 1.2 KB, `rules::tests::the_longest_eval_line_fits_a_trace_line`). On the stack of the calling thread.
pub const LINE_BYTES: usize = 2048;

/// A path variable's value as a Windows path: a Unix path (`/home/...`) through Wine's `Z:` drive (Z: is the Unix
/// root in every prefix Wine and Proton make), anything else as given. Quotes Steam's launch options may leave
/// around it are dropped. `None` when empty.
pub fn windows_path(value: &str) -> Option<String> {
    let v = value.trim().trim_matches(|c| c == '\'' || c == '"').trim();
    if v.is_empty() {
        return None;
    }
    Some(if v.starts_with('/') { format!("Z:{}", v.replace('/', "\\")) } else { v.to_string() })
}

/// `{pid}` in the path replaced by the game's Linux pid; without `{pid}`, `-<pid>` is put before the file's extension
/// (or at its end), so a launcher and its game, or two runs, never write the same file. A path that ends in a
/// separator names a folder: the file in it is `ngx-trace-<pid>.txt`.
pub fn with_pid(path: &str, pid: Option<u32>) -> String {
    let pid = pid.unwrap_or(0).to_string();
    if path.ends_with(['\\', '/']) {
        return format!("{path}ngx-trace-{pid}.txt");
    }
    if path.contains("{pid}") {
        return path.replace("{pid}", &pid);
    }
    let name_at = path.rfind(['\\', '/']).map_or(0, |i| i + 1);
    match path[name_at..].rfind('.').filter(|&dot| dot > 0) {
        Some(dot) => format!("{}-{pid}{}", &path[..name_at + dot], &path[name_at + dot..]),
        None => format!("{path}-{pid}"),
    }
}

/// A `fmt::Write` into a fixed buffer that cuts what does not fit (no allocation).
struct Cursor<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl fmt::Write for Cursor<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let n = s.len().min(self.buf.len() - self.len);
        self.buf[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// The header line for a ring whose next line goes at data offset `next % DATA_BYTES`, after `written` bytes in all.
pub fn header(written: u64) -> [u8; HEADER_BYTES] {
    let mut out = [b' '; HEADER_BYTES];
    let mut w = Cursor { buf: &mut out[..HEADER_BYTES - 1], len: 0 };
    let _ = write!(
        w,
        "neural-forge-ngx-trace 1 header={HEADER_BYTES} ring={DATA_BYTES} written={written:016} next={:010}{}",
        written % DATA_BYTES as u64,
        if written > DATA_BYTES as u64 { " wrapped" } else { "" }
    );
    out[HEADER_BYTES - 1] = b'\n';
    out
}

/// Copies `line` into the ring `data` at stream position `pos`, wrapping at its end.
pub fn write_at(data: &mut [u8], pos: u64, line: &[u8]) {
    if data.is_empty() {
        return;
    }
    let at = (pos % data.len() as u64) as usize;
    let first = line.len().min(data.len() - at);
    data[at..at + first].copy_from_slice(&line[..first]);
    let rest = &line[first..];
    let rest = &rest[..rest.len().min(data.len())];
    data[..rest.len()].copy_from_slice(rest);
}

/// One line, formatted on the stack (no allocation on the traced paths).
pub struct Line {
    buf: [u8; LINE_BYTES],
    len: usize,
}

impl Line {
    /// `t_ms` (since the wrapper loaded), the thread, the text, a newline.
    pub fn new(t_ms: f64, thread: u32, args: fmt::Arguments) -> Line {
        let mut buf = [0; LINE_BYTES];
        let mut w = Cursor { buf: &mut buf[..LINE_BYTES - 1], len: 0 };
        let _ = write!(w, "{t_ms:13.3} tid={thread:<6} ");
        let _ = w.write_fmt(args);
        let len = w.len;
        buf[len] = b'\n';
        Line { buf, len: len + 1 }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// The calls a thread is inside, for their exit lines: the export, the caller's return address the thunk replaced,
/// the entry time. A fixed array (no destructor in the thread-local).
#[derive(Clone, Copy)]
pub struct Frame {
    pub export: usize,
    pub ret: usize,
    pub t_ms: f64,
}

pub const MAX_DEPTH: usize = 64;

pub struct Stack {
    pub depth: usize,
    pub frames: [Frame; MAX_DEPTH],
}

impl Stack {
    pub const fn new() -> Stack {
        Stack { depth: 0, frames: [Frame { export: 0, ret: 0, t_ms: 0.0 }; MAX_DEPTH] }
    }

    /// Pushes a frame; `false` when full (that call's exit is then not traced, its return left alone).
    pub fn push(&mut self, f: Frame) -> bool {
        if self.depth == MAX_DEPTH {
            return false;
        }
        self.frames[self.depth] = f;
        self.depth += 1;
        true
    }

    pub fn pop(&mut self) -> Option<Frame> {
        self.depth = self.depth.checked_sub(1)?;
        Some(self.frames[self.depth])
    }
}

impl Default for Stack {
    fn default() -> Self {
        Stack::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_paths_go_through_z() {
        assert_eq!(windows_path("/home/u/ngx.log").as_deref(), Some(r"Z:\home\u\ngx.log"));
        assert_eq!(windows_path(r"Z:\tmp\ngx.log").as_deref(), Some(r"Z:\tmp\ngx.log"));
        assert_eq!(windows_path(r"'Z:\tmp\ngx.log'").as_deref(), Some(r"Z:\tmp\ngx.log"), "quotes left by a launch option");
        assert_eq!(windows_path(" \"/tmp/a b.log\" ").as_deref(), Some(r"Z:\tmp\a b.log"));
        assert_eq!(windows_path("C:/x.log").as_deref(), Some("C:/x.log"));
        assert_eq!(windows_path("  "), None);
        assert_eq!(windows_path("''"), None);
    }

    #[test]
    fn the_pid_is_put_in() {
        assert_eq!(with_pid(r"Z:\tmp\t-{pid}.txt", Some(42)), r"Z:\tmp\t-42.txt");
        assert_eq!(with_pid(r"Z:\tmp\t.txt", Some(42)), r"Z:\tmp\t-42.txt", "no {{pid}}: put before the extension");
        assert_eq!(with_pid(r"Z:\tmp.d\trace", Some(42)), r"Z:\tmp.d\trace-42", "a dot in a folder is no extension");
        assert_eq!(with_pid(r"Z:\tmp\.trace", Some(7)), r"Z:\tmp\.trace-7");
        assert_eq!(with_pid("{pid}", None), "0");
        assert_eq!(with_pid(r"Z:\home\u\", Some(9)), r"Z:\home\u\ngx-trace-9.txt", "a folder: the default name in it");
        assert_eq!(with_pid("Z:/tmp/", Some(9)), "Z:/tmp/ngx-trace-9.txt");
    }

    /// The traced paths allocate nothing (a heap lock at process exit could hang it): counted per thread.
    mod no_allocation {
        use super::super::*;
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        struct Counting;
        thread_local! {
            static COUNT: Cell<usize> = const { Cell::new(0) };
        }
        // SAFETY: forwards to the system allocator.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = COUNT.try_with(|c| c.set(c.get() + 1));
                unsafe { System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }
        }
        #[global_allocator]
        static ALLOC: Counting = Counting;

        #[test]
        fn a_line_and_the_header_allocate_nothing() {
            let before = COUNT.with(Cell::get);
            let h = header(DATA_BYTES as u64 + 77);
            let l = Line::new(12.5, 3, format_args!("> {}({:#x}, {:.3} ms)", "NVSDK_NGX_VULKAN_Init", 0x10u32, 1.25f64));
            let mut ring = [0u8; 64];
            write_at(&mut ring, 60, l.bytes());
            let mut s = Stack::new();
            s.push(Frame { export: 1, ret: 2, t_ms: 0.0 });
            assert_eq!(COUNT.with(Cell::get), before, "allocated");
            assert_eq!(h[HEADER_BYTES - 1], b'\n');
            let _ = String::from("counted");
            assert!(COUNT.with(Cell::get) > before, "the counter counts");
        }
    }

    #[test]
    fn the_header_is_fixed_width_and_names_the_next_line() {
        let h = header(0);
        assert_eq!(h.len(), HEADER_BYTES);
        assert_eq!(h[HEADER_BYTES - 1], b'\n');
        let text = String::from_utf8(h.to_vec()).unwrap();
        assert!(text.starts_with("neural-forge-ngx-trace 1 ") && text.contains("next=0000000000") && !text.contains("wrapped"), "{text}");
        let text = String::from_utf8(header(DATA_BYTES as u64 + 5).to_vec()).unwrap();
        assert!(text.contains("next=0000000005 wrapped"), "{text}");
        assert!(!text[..HEADER_BYTES - 1].contains('\n'));
    }

    #[test]
    fn lines_wrap_at_the_rings_end() {
        let mut ring = [0u8; 10];
        write_at(&mut ring, 0, b"abcdef");
        write_at(&mut ring, 6, b"ghijkl");
        assert_eq!(&ring, b"klcdefghij");
        write_at(&mut ring, 23, b"0123456789ABCD");
        assert_eq!(ring.len(), 10, "a line longer than the ring keeps its last bytes in place");
        write_at(&mut [], 3, b"x");
    }

    #[test]
    fn a_line_starts_with_its_time_and_ends_with_a_newline() {
        let l = Line::new(1234.5678, 77, format_args!("> NVSDK_NGX_VULKAN_Init(0x{:x})", 0x1234));
        let s = std::str::from_utf8(l.bytes()).unwrap();
        assert_eq!(s, "     1234.568 tid=77     > NVSDK_NGX_VULKAN_Init(0x1234)\n");
        let long = "x".repeat(5000);
        let l = Line::new(0.0, 1, format_args!("{long}"));
        assert_eq!(l.bytes().len(), LINE_BYTES);
        assert_eq!(*l.bytes().last().unwrap(), b'\n', "a cut line keeps its newline");
    }

    #[test]
    fn the_stack_holds_its_depth() {
        let mut s = Stack::new();
        for i in 0..MAX_DEPTH {
            assert!(s.push(Frame { export: i, ret: i * 2, t_ms: 0.0 }));
        }
        assert!(!s.push(Frame { export: 99, ret: 0, t_ms: 0.0 }), "full");
        assert_eq!(s.pop().map(|f| f.export), Some(MAX_DEPTH - 1));
        while s.pop().is_some() {}
        assert!(s.pop().is_none());
    }
}
