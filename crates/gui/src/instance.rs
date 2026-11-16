//! One running copy of the app per user login.
//!
//! The first copy listens on a socket in the user's runtime directory
//! (`XDG_RUNTIME_DIR`, one per login, shared by every display of that login).
//! A second launch finds it, hands over its activation token so the running
//! window can be raised, waits for the running copy's UI to say it has done so
//! (a window that does not answer is reported to the user), and exits: two windows would both write the
//! settings and both run the start-up install.
//!
//! Without `XDG_RUNTIME_DIR` there is no directory that is known to be the
//! user's alone, so the app runs without the single-instance check rather
//! than trust a socket in a shared place.

use async_channel::{unbounded as unbounded_channel, Receiver as UnboundedReceiver};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// One later launch asking for the running window to be shown.
pub struct Request {
    /// The activation token of that launch, when it had one (see `launch_token`).
    pub token: Option<String>,
    reply: UnixStream,
}

impl Request {
    /// Tells the launch that asked that the UI has handled its request: it waits for this line, and takes silence
    /// for a window that no longer responds.
    pub fn acknowledge(mut self) {
        let _ = self.reply.write_all(ACK.as_bytes());
    }
}

/// Requests from later launches to show the running window, and the word to them that the window is up.
pub struct Activations {
    requests: UnboundedReceiver<Request>,
    up: Arc<AtomicBool>,
}

impl Activations {
    /// The next request. Asking for one says the UI is up: from then on a later launch is not told to wait for a
    /// start-up any more.
    pub async fn next(&self) -> Option<Request> {
        self.up.store(true, Ordering::Relaxed);
        self.requests.recv().await.ok()
    }
}

/// What the running copy answers once its UI has handled a request.
const ACK: &str = "ok\n";

/// What the running copy's listener answers at once while its window is not up yet (the socket is bound before the
/// graphics start, which can take seconds on a cold start or a software renderer).
const STARTING: &str = "starting\n";

/// How long a later launch waits for the first answer, and how long for the UI's once it heard the window is starting.
const WAITS: Waits = Waits { first: Duration::from_secs(2), start_up: Duration::from_secs(15) };

#[derive(Clone, Copy)]
struct Waits {
    first: Duration,
    start_up: Duration,
}

/// What became of this launch.
pub enum Claim {
    /// This is the first copy; the receiver yields once per later launch.
    Primary(Activations),
    /// Another copy is still starting (it said so) and had not shown its window when this launch stopped waiting: it
    /// will, so this launch just ends.
    Starting,
    /// Another copy took the request and exited without answering, twice over: it is going away, and this launch
    /// ends without a word.
    Gone,
    /// Another copy is running and has shown its window.
    Shown,
    /// Another copy is running (its process id, where the socket tells) but its UI did not answer.
    NotResponding(Option<u32>),
}

fn runtime_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").filter(|dir| !dir.is_empty()).map(PathBuf::from)
}

/// What the desktop gave this launch to raise a window with: the Wayland
/// activation token, or the startup id (which desktops also fill with that
/// token).
fn launch_token() -> Option<String> {
    ["XDG_ACTIVATION_TOKEN", "DESKTOP_STARTUP_ID"].into_iter().filter_map(|name| std::env::var(name).ok()).find(|token| valid_token(token))
}

/// The longest token passed on. Real ones are a few dozen bytes; a Wayland
/// message holds at most 4096, and a request larger than that makes the
/// compositor drop the app's whole connection.
const MAX_TOKEN: usize = 256;

/// A token is a short run of printable ASCII: it travels as one line here
/// and as a string argument to the compositor, and comes from the
/// environment, which anything may have set.
fn valid_token(token: &str) -> bool {
    !token.is_empty() && token.len() <= MAX_TOKEN && token.bytes().all(|b| b.is_ascii_graphic())
}

/// Finds the running copy and asks it to show its window, or makes this the first copy.
pub fn claim() -> Claim {
    match runtime_dir() {
        Some(dir) => claim_at(&dir.join("neural-forge.sock"), launch_token(), WAITS),
        None => {
            eprintln!("neural-forge: XDG_RUNTIME_DIR is not set, so a second launch can't find this one");
            let (tx, rx) = unbounded_channel();
            // Kept open for the life of the process: the receiver never ends.
            std::mem::forget(tx);
            Claim::Primary(Activations { requests: rx, up: Arc::default() })
        }
    }
}

/// Holds `flock` on the lock file next to the socket, so that of two copies
/// started together only one looks at, removes and binds the socket at a time.
struct Lock(#[allow(dead_code)] std::fs::File);

fn lock(socket: &Path) -> Option<Lock> {
    let mut name = socket.as_os_str().to_owned();
    name.push(".lock");
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(PathBuf::from(name)).ok()?;
    // SAFETY: flock on a file descriptor this function owns; released when the file closes.
    (unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0).then_some(Lock(file))
}

/// Whether a failed connect means "nobody is listening" (a socket file left
/// by a copy that is gone, or no file at all), as opposed to a running copy
/// that could not be reached just now.
fn nobody_listening(error: &std::io::Error) -> bool {
    matches!(error.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound)
}

fn claim_at(path: &Path, token: Option<String>, waits: Waits) -> Claim {
    let _lock = lock(path);
    // Twice at most: a running copy that exits while this launch waits for its answer is looked for once more
    // (usually nobody listens then, and this launch becomes the first copy).
    for attempt in 0..2 {
        match UnixStream::connect(path) {
            Ok(mut stream) => {
                let _ = writeln!(stream, "activate {}", token.as_deref().unwrap_or(""));
                match await_answer(&stream, waits) {
                    Some(claim) => return claim,
                    None if attempt == 0 => continue,
                    None => return Claim::Gone,
                }
            }
            Err(e) if nobody_listening(&e) => {
                let _ = std::fs::remove_file(path);
                break;
            }
            Err(e) => {
                // Something answers to that name but not to us: leave it alone and run without the check.
                eprintln!("neural-forge: single-instance socket {}: {e}", path.display());
                break;
            }
        }
    }
    let (tx, rx) = unbounded_channel();
    let up: Arc<AtomicBool> = Arc::default();
    match UnixListener::bind(path) {
        Ok(listener) => {
            let up = Arc::clone(&up);
            std::thread::spawn(move || {
                for connection in listener.incoming().flatten() {
                    let mut request = read_request(connection);
                    // Until the UI is up, the launch that asks is told so here: the UI's own answer follows once the
                    // window is there to show.
                    if !up.load(Ordering::Relaxed) {
                        let _ = request.reply.write_all(STARTING.as_bytes());
                    }
                    if tx.send_blocking(request).is_err() {
                        break;
                    }
                }
            });
        }
        Err(e) => {
            // Without the socket the app still runs; it just can't be found by a second launch.
            eprintln!("neural-forge: single-instance socket {}: {e}", path.display());
            std::mem::forget(tx);
        }
    }
    Claim::Primary(Activations { requests: rx, up })
}

/// What the running copy made of this launch's request: its UI's answer (the window is shown), or silence. A copy
/// that says it is still starting is given longer, and is not called stuck if its window takes longer still. `None`
/// when the copy closed the connection without an answer: it has exited (or is exiting), which is not a stuck window.
fn await_answer(stream: &UnixStream, waits: Waits) -> Option<Claim> {
    let mut lines = BufReader::new(stream).take(64);
    // A line, `Some("")` for none in time, `None` for a closed connection.
    let mut read_line = |wait: Duration| {
        let _ = stream.set_read_timeout(Some(wait));
        let mut line = String::new();
        match lines.read_line(&mut line) {
            Ok(0) => None,
            // (A connection closed with the request still unread is reset, not ended.)
            Err(e) if matches!(e.kind(), std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe) => None,
            _ => Some(line),
        }
    };
    Some(match read_line(waits.first)?.as_str() {
        ACK => Claim::Shown,
        STARTING => match read_line(waits.start_up)?.as_str() {
            ACK => Claim::Shown,
            _ => Claim::Starting,
        },
        _ => Claim::NotResponding(peer_pid(stream)),
    })
}

/// The process on the other end of the socket (the running copy), from the kernel's record of who bound it.
fn peer_pid(stream: &UnixStream) -> Option<u32> {
    let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED writes one `ucred` into the buffer of that size, on a socket this function borrows.
    let rc = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, std::ptr::from_mut(&mut cred).cast(), &mut len) };
    (rc == 0 && cred.pid > 0).then_some(cred.pid as u32)
}

/// What to tell the user when the running copy does not answer.
pub fn not_responding_message(pid: Option<u32>) -> String {
    let which = pid.map(|pid| format!(" (process {pid})")).unwrap_or_default();
    format!("Neural Forge is already running but not responding. Close it{which} and start it again.")
}

/// The request a second launch sent: its token, if any (a launch that sends nothing readable still counts), and the
/// connection to answer it on.
fn read_request(connection: UnixStream) -> Request {
    let _ = connection.set_read_timeout(Some(Duration::from_secs(2)));
    let mut line = String::new();
    let _ = BufReader::new(&connection).take(8192).read_line(&mut line);
    Request { token: parse_request(&line), reply: connection }
}

fn parse_request(line: &str) -> Option<String> {
    let token = line.trim_end_matches('\n').strip_prefix("activate ")?;
    valid_token(token).then(|| token.to_string())
}

#[cfg(test)]
mod tests {
    use super::{claim_at, nobody_listening, not_responding_message, parse_request, valid_token, Activations, Claim, Waits, MAX_TOKEN};
    use std::time::Duration;

    const SOON: Waits = Waits { first: Duration::from_millis(300), start_up: Duration::from_millis(600) };
    const PATIENT: Waits = Waits { first: Duration::from_secs(5), start_up: Duration::from_secs(5) };

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("neural-forge-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn primary(claim: Claim) -> Activations {
        match claim {
            Claim::Primary(requests) => requests,
            _ => panic!("the first copy is the primary"),
        }
    }

    /// The running copy's UI taking one request, as `build_ui`'s loop does.
    fn ui_takes_one(first: &Activations) -> super::Request {
        first.up.store(true, std::sync::atomic::Ordering::Relaxed);
        first.requests.recv_blocking().unwrap()
    }

    /// A second launch at `path`, with a running copy whose UI answers each request.
    fn second_launch(path: &std::path::Path, first: &Activations, token: Option<&str>) -> (Claim, Option<String>) {
        std::thread::scope(|scope| {
            let ui = scope.spawn(|| {
                let request = ui_takes_one(first);
                let token = request.token.clone();
                request.acknowledge();
                token
            });
            let claim = claim_at(path, token.map(str::to_string), PATIENT);
            (claim, ui.join().unwrap())
        })
    }

    #[test]
    fn second_launch_reaches_the_first_with_its_token_and_hears_back() {
        let dir = scratch("instance");
        let path = dir.join("test.sock");
        let first = primary(claim_at(&path, None, SOON));
        let (claim, token) = second_launch(&path, &first, Some("token-1"));
        assert!(matches!(claim, Claim::Shown), "a second copy finds the first, is answered and stands down");
        assert_eq!(token, Some("token-1".to_string()), "and the first is told, with the token to raise its window with");
        let (claim, token) = second_launch(&path, &first, None);
        assert!(matches!(claim, Claim::Shown));
        assert_eq!(token, None, "no token: still a request");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_that_is_still_starting_is_waited_for_and_never_called_stuck() {
        let dir = scratch("starting");
        let path = dir.join("test.sock");
        // The first copy has bound its socket but its UI is not up yet (the graphics are still starting).
        let first = primary(claim_at(&path, None, SOON));
        // Its window comes up after the first wait has passed, within the start-up wait: the second launch hears "ok".
        let claim = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(450));
                ui_takes_one(&first).acknowledge();
            });
            claim_at(&path, None, SOON)
        });
        assert!(matches!(claim, Claim::Shown), "the start-up was waited for, past the first wait");
        // A start-up slower than any wait: the second launch ends quietly, it does not report a stuck window.
        let slow = scratch("starting-slow");
        let slow_path = slow.join("test.sock");
        let still_starting = primary(claim_at(&slow_path, None, SOON));
        assert!(matches!(claim_at(&slow_path, None, SOON), Claim::Starting));
        drop(still_starting);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&slow);
    }

    #[test]
    fn a_running_copy_whose_ui_does_not_answer_is_named() {
        let dir = scratch("stuck");
        let path = dir.join("test.sock");
        // The first copy's window was up (its UI took requests), and now takes none (a stuck window).
        let first = primary(claim_at(&path, None, SOON));
        first.up.store(true, std::sync::atomic::Ordering::Relaxed);
        match claim_at(&path, Some("token-2".into()), SOON) {
            Claim::NotResponding(pid) => {
                assert_eq!(pid, Some(std::process::id()), "the running copy's process, from the socket");
                assert_eq!(
                    not_responding_message(pid),
                    format!("Neural Forge is already running but not responding. Close it (process {}) and start it again.", std::process::id())
                );
            }
            _ => panic!("no answer in time is not a shown window"),
        }
        assert_eq!(not_responding_message(None), "Neural Forge is already running but not responding. Close it and start it again.");
        drop(first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_that_exits_while_a_second_launch_waits_is_not_called_stuck() {
        use std::os::unix::net::UnixListener;
        // A copy that takes the request and exits (its listener and socket file go with it): the second launch looks
        // again, finds nobody and becomes the first copy.
        let dir = scratch("gone");
        let path = dir.join("test.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let exiting = {
            let path = path.clone();
            std::thread::spawn(move || {
                let (connection, _) = listener.accept().unwrap();
                drop(listener);
                let _ = std::fs::remove_file(&path);
                drop(connection);
            })
        };
        let claim = claim_at(&path, None, PATIENT);
        exiting.join().unwrap();
        assert!(matches!(claim, Claim::Primary(_)), "the copy that exited left the place free");
        drop(claim);
        // A copy that keeps closing connections unanswered (it is shutting down): no "not responding", just an end.
        let dir2 = scratch("gone-twice");
        let path2 = dir2.join("test.sock");
        let listener = UnixListener::bind(&path2).unwrap();
        let closing = std::thread::spawn(move || {
            for _ in 0..2 {
                drop(listener.accept().unwrap());
            }
        });
        assert!(matches!(claim_at(&path2, None, PATIENT), Claim::Gone));
        closing.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn stale_socket_file_is_replaced() {
        let dir = scratch("stale");
        let path = dir.join("test.sock");
        // A socket nobody listens on any more.
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        assert!(path.exists());
        let first = primary(claim_at(&path, None, SOON));
        let (claim, _) = second_launch(&path, &first, None);
        assert!(matches!(claim, Claim::Shown), "and the new one answers");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_a_dead_socket_is_removed() {
        use std::io::{Error, ErrorKind};
        assert!(nobody_listening(&Error::from(ErrorKind::ConnectionRefused)));
        assert!(nobody_listening(&Error::from(ErrorKind::NotFound)));
        assert!(!nobody_listening(&Error::from(ErrorKind::PermissionDenied)));
        assert!(!nobody_listening(&Error::from(ErrorKind::WouldBlock)));
        // A path that is not a socket at all is not ours to delete.
        let dir = scratch("notsock");
        let path = dir.join("test.sock");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("keep"), b"x").unwrap();
        let _ = claim_at(&path, None, SOON);
        assert!(path.join("keep").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_copies_started_together_leave_one_primary() {
        let dir = scratch("race");
        let path = dir.join("test.sock");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || claim_at(&path, None, Waits { first: Duration::from_millis(50), start_up: Duration::from_millis(50) }))
            })
            .collect();
        let claims: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(claims.iter().filter(|c| matches!(c, Claim::Primary(_))).count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn requests_and_tokens_are_one_clean_line() {
        assert_eq!(parse_request("activate abc-123_x\n"), Some("abc-123_x".into()));
        assert_eq!(parse_request("activate \n"), None);
        assert_eq!(parse_request("activate\n"), None);
        assert_eq!(parse_request(""), None);
        assert_eq!(parse_request("something else\n"), None);
        assert!(!valid_token("a\nb") && !valid_token("") && !valid_token("a\u{7}"));
        assert!(!valid_token("a b") && !valid_token("tok\u{e9}n"), "printable ASCII only");
        assert!(valid_token(&"x".repeat(MAX_TOKEN)));
        assert!(!valid_token(&"x".repeat(MAX_TOKEN + 1)), "an oversized token must never reach the compositor");
        let long = format!("activate {}\n", "x".repeat(MAX_TOKEN + 1));
        assert_eq!(parse_request(&long), None);
    }
}
