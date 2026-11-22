//! What the replaced shell scripts got from being shell: a command's exit status passed on (`set -e`), `${VAR:-default}`,
//! `command -v`, `mktemp -d` with its removal on the way out (`trap ... EXIT`), and an end by Ctrl-C, a hang-up or a
//! termination that still runs that clean-up.

use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};
use std::sync::atomic::{AtomicI32, Ordering};

/// A task's failure: the exit code, and the line for stderr unless the failed command has said it all itself.
#[derive(Debug)]
pub struct Fail {
    pub code: u8,
    pub message: Option<String>,
}

impl Fail {
    pub fn new(code: u8, message: impl Into<String>) -> Self {
        Fail { code, message: Some(message.into()) }
    }

    pub fn silent(code: u8) -> Self {
        Fail { code, message: None }
    }
}

impl From<String> for Fail {
    fn from(message: String) -> Self {
        Fail::new(1, message)
    }
}

impl From<&str> for Fail {
    fn from(message: &str) -> Self {
        Fail::new(1, message)
    }
}

pub type Res<T> = Result<T, Fail>;

/// A task's end: its exit code, a failure's message on stderr first.
pub fn finish(result: Res<ExitCode>) -> ExitCode {
    result.unwrap_or_else(|fail| {
        if let Some(message) = fail.message {
            eprintln!("{message}");
        }
        ExitCode::from(fail.code)
    })
}

/// The exit status a shell gives a command: its code, or 128 plus the signal that ended it.
pub fn code(status: ExitStatus) -> u8 {
    status.code().or_else(|| status.signal().map(|s| 128 + s)).unwrap_or(1) as u8
}

/// The message for a program that could not be started (a shell's "command not found" is 127).
pub fn not_started(cmd: &Command, e: std::io::Error) -> Fail {
    Fail::new(127, format!("{}: {e}", cmd.get_program().to_string_lossy()))
}

/// Runs `cmd` with this process's stdin, stdout and stderr; anything but success is the task's own failure, with the
/// command's exit status (`set -e`).
pub fn check(cmd: &mut Command) -> Res<()> {
    let status = cmd.status().map_err(|e| not_started(cmd, e))?;
    interrupted()?;
    if status.success() {
        Ok(())
    } else {
        Err(Fail::silent(code(status)))
    }
}

/// `${NAME:-}`: the variable when it is set and not empty.
pub fn var(name: &str) -> Option<String> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(|v| v.to_string_lossy().into_owned())
}

pub fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// `command -v NAME` for a program: the first executable of that name on PATH.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path).map(|dir| dir.join(name)).find(|p| is_executable(p))
}

/// `$(cat FILE 2>/dev/null)`: the file's text without its trailing newlines, empty when it cannot be read.
pub fn cat(path: &Path) -> String {
    std::fs::read_to_string(path).map(|t| t.trim_end_matches('\n').to_string()).unwrap_or_default()
}

/// A directory of this run's own, removed with everything in it when dropped.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `mktemp -d PREFIXXXXXXX`: a new private (mode 0700) directory whose name ends in six random characters.
pub fn mktemp_d(prefix: &Path) -> std::io::Result<TempDir> {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut last = None;
    for attempt in 0..100u32 {
        let mut random = [0u8; 6];
        if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut random)).is_err() {
            // No /dev/urandom: the clock and the process id still make a new name, and create refuses a taken one.
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
            let mixed = nanos ^ (u128::from(std::process::id()) << 64) ^ u128::from(attempt);
            random.copy_from_slice(&mixed.to_le_bytes()[..6]);
        }
        let suffix: String = random.iter().map(|b| CHARS[usize::from(*b) % CHARS.len()] as char).collect();
        let mut name = prefix.as_os_str().to_os_string();
        name.push(suffix);
        let dir = PathBuf::from(name);
        match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => return Ok(TempDir(dir)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("no free name")))
}

/// `mktemp -d`: a new private directory in TMPDIR (or /tmp).
pub fn mktemp() -> Res<TempDir> {
    let base = std::env::temp_dir();
    mktemp_d(&base.join("tmp.")).map_err(|e| Fail::new(1, format!("cannot create a temporary directory in {}: {e}", base.display())))
}

static SIGNAL: AtomicI32 = AtomicI32::new(0);

extern "C" fn on_signal(signal: libc::c_int) {
    SIGNAL.store(signal, Ordering::SeqCst);
}

/// From here on Ctrl-C, a hang-up and a termination no longer end the process at once: `interrupted` reports them, so
/// the task ends through its own clean-up (a shell's EXIT trap ran on these too).
pub fn trap_signals() {
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        // SAFETY: the handler only stores to an atomic.
        unsafe { libc::signal(signal, on_signal as *const () as libc::sighandler_t) };
    }
}

/// The failure of a run that was told to end (exit code 128 plus the signal), once `trap_signals` is on.
pub fn interrupted() -> Res<()> {
    match SIGNAL.load(Ordering::SeqCst) {
        0 => Ok(()),
        signal => Err(Fail::silent((128 + signal) as u8)),
    }
}

/// `sleep SECONDS`, ended early by a trapped signal.
pub fn sleep(seconds: f64) -> Res<()> {
    let end = std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds.max(0.0));
    loop {
        interrupted()?;
        let left = end.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Ok(());
        }
        std::thread::sleep(left.min(std::time::Duration::from_millis(100)));
    }
}

/// `timeout SECONDS command`: the command in its own process group, ended (SIGTERM to the group) when it runs longer.
/// Its exit status, 124 when it was ended.
pub fn timeout(seconds: u64, cmd: &mut Command) -> std::io::Result<u8> {
    use std::os::unix::process::CommandExt;
    let mut child = cmd.process_group(0).spawn()?;
    let end = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(code(status));
        }
        if std::time::Instant::now() >= end {
            kill(-(child.id() as i32), libc::SIGTERM);
            let _ = child.wait();
            return Ok(124);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// `$(timeout SECONDS command)`: as `timeout`, with what the command wrote to stdout.
pub fn timeout_output(seconds: u64, cmd: &mut Command) -> std::io::Result<(u8, String)> {
    use std::os::unix::process::CommandExt;
    let mut child = cmd.process_group(0).stdout(std::process::Stdio::piped()).spawn()?;
    let mut pipe = child.stdout.take();
    // Read on its own thread, so a command that writes more than a pipe holds is not stopped by it.
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(pipe) = pipe.as_mut() {
            let _ = pipe.read_to_end(&mut bytes);
        }
        bytes
    });
    let end = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    let code = loop {
        if let Some(status) = child.try_wait()? {
            break code(status);
        }
        if std::time::Instant::now() >= end {
            kill(-(child.id() as i32), libc::SIGTERM);
            let _ = child.wait();
            break 124;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let bytes = reader.join().unwrap_or_default();
    Ok((code, String::from_utf8_lossy(&bytes).into_owned()))
}

/// A word as a shell reads it back unchanged (`printf %q`'s purpose): in single quotes unless it is plain.
pub fn quote(word: &str) -> String {
    if !word.is_empty() && word.chars().all(|c| c.is_ascii_alphanumeric() || "_-./=:,+@%".contains(c)) {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

/// `date "+%F %T"` of a Unix time: the local date and time.
pub fn local_time(epoch: i64) -> String {
    // SAFETY: localtime_r fills the zeroed struct it is given.
    let tm = unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&epoch, &mut tm);
        tm
    };
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// `date +%s`.
pub fn epoch() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// `date "+%F %T"`.
pub fn now() -> String {
    local_time(epoch())
}

/// The number `sort -n` and awk read at the start of a text (0 when there is none).
pub fn leading_number(text: &str) -> f64 {
    let text = text.trim_start_matches([' ', '\t']);
    let mut end = 0;
    let mut dot = false;
    for (i, c) in text.char_indices() {
        if c.is_ascii_digit() || (c == '-' && i == 0) || (c == '.' && !dot) {
            dot |= c == '.';
            end = i + 1;
        } else {
            break;
        }
    }
    text[..end].parse().unwrap_or(0.0)
}

/// `kill -SIGNAL PID` (a negative pid is a process group); false when nothing was signalled.
pub fn kill(pid: i32, signal: libc::c_int) -> bool {
    // SAFETY: plain kill(2).
    unsafe { libc::kill(pid, signal) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mktemp_d_makes_a_private_directory_and_removes_it() {
        let base = std::env::temp_dir();
        let dir = mktemp_d(&base.join("nf-xtask-sh-test.")).unwrap();
        let path = dir.path().to_path_buf();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("nf-xtask-sh-test.") && name.len() == "nf-xtask-sh-test.".len() + 6, "{name}");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::write(path.join("file"), "x").unwrap();
        drop(dir);
        assert!(!path.exists());
    }

    #[test]
    fn quoted_words_survive_a_shell() {
        for word in ["", "plain-1.x=y", "a b", "it's", "WINEDLLOVERRIDES=xinput1_4=b;dinput8=b", "$HOME `x` \\ \"q\""] {
            let out = Command::new("sh").arg("-c").arg(format!("printf %s {}", quote(word))).output().unwrap();
            assert_eq!(String::from_utf8_lossy(&out.stdout), word, "{}", quote(word));
        }
    }

    #[test]
    fn leading_numbers() {
        assert_eq!(leading_number("118.2 fps"), 118.2);
        assert_eq!(leading_number("  -3x"), -3.0);
        assert_eq!(leading_number("[present] 5"), 0.0);
        assert_eq!(leading_number("1.2.3"), 1.2);
    }

    #[test]
    fn a_command_that_runs_too_long_is_ended() {
        assert_eq!(timeout(5, Command::new("sh").args(["-c", "exit 3"])).unwrap(), 3);
        let started = std::time::Instant::now();
        assert_eq!(timeout(1, Command::new("sleep").arg("30")).unwrap(), 124);
        assert!(started.elapsed().as_secs() < 10);
    }

    #[test]
    fn local_time_has_date_s_form() {
        let text = now();
        assert!(regex::Regex::new(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$").unwrap().is_match(&text), "{text}");
    }

    #[test]
    fn cat_drops_only_trailing_newlines() {
        let dir = mktemp().unwrap();
        std::fs::write(dir.path().join("f"), " a\nb \n\n").unwrap();
        assert_eq!(cat(&dir.path().join("f")), " a\nb ");
        assert_eq!(cat(&dir.path().join("missing")), "");
    }
}
