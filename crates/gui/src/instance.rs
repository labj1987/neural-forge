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
use std::time::{Duration, Instant};

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
    up: UiUp,
    listening: Option<Listening>,
}

impl Activations {
    /// The handle that says the UI is up, set from the window's first drawn frame.
    pub fn up_signal(&self) -> UiUp {
        self.up.clone()
    }

    /// The next request.
    pub async fn next(&self) -> Option<Request> {
        self.requests.recv().await.ok()
    }
}

impl Drop for Activations {
    /// Requests nobody will handle any more are closed unanswered: the launches waiting on them hear the end of the
    /// connection at once (and look again) instead of waiting out the start-up wait.
    fn drop(&mut self) {
        self.requests.close();
        while self.requests.try_recv().is_ok() {}
    }
}

/// Says the first copy's UI is up (see `Activations::up_signal`). Set from the window's first drawn frame: a window
/// that has been created but never drawn is not up yet, and a later launch is told to wait for it.
#[derive(Clone, Default)]
pub struct UiUp(Arc<AtomicBool>);

impl UiUp {
    pub fn set(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_set(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The first copy's socket, closed (and its file removed) when the app is done with it: when its last window closes,
/// or when it can't open one. Without that, a later launch would connect to a copy that no longer reads its requests,
/// wait out the whole start-up wait, and call it stuck.
#[derive(Clone)]
struct Listening {
    listener: Arc<UnixListener>,
    path: PathBuf,
    closed: Arc<AtomicBool>,
}

impl Listening {
    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        // The file first: from here a launch finds nobody and becomes the first copy, instead of reaching a listener
        // that is about to stop.
        let _ = std::fs::remove_file(&self.path);
        // Wakes the listener thread's accept with an error and refuses connections from here on; the thread then ends.
        // SAFETY: shutdown on a socket this struct keeps open.
        unsafe { libc::shutdown(self.listener.as_raw_fd(), libc::SHUT_RDWR) };
    }
}

/// The first copy's socket, for `release`.
static PRIMARY: std::sync::OnceLock<Listening> = std::sync::OnceLock::new();

/// Stops answering later launches and removes the socket: call it when the app is about to end (its last window
/// closed, or none could be opened), so the next launch becomes the first copy at once.
pub fn release() {
    if let Some(listening) = PRIMARY.get() {
        listening.close();
    }
}

/// What the running copy answers once its UI has handled a request.
const ACK: &str = "ok\n";

/// What the running copy's listener answers at once while its window is not up yet (the socket is bound before the
/// graphics start, which can take seconds on a cold start or a software renderer).
const STARTING: &str = "starting\n";

/// How long a later launch waits for an answer.
#[derive(Clone, Copy)]
struct Waits {
    /// For any answer at all; also how long a copy that closes the connection unanswered (it is exiting) is asked
    /// again, counted from the first time it did.
    answer: Duration,
    /// For "ok" from a copy that said it is starting, counted from the request.
    start: Duration,
}

const WAITS: Waits = Waits { answer: Duration::from_secs(2), start: Duration::from_secs(15) };

/// How often a copy that closed the connection unanswered is asked again.
const RETRY_STEP: Duration = Duration::from_millis(75);

/// What became of this launch.
pub enum Claim {
    /// This is the first copy; the receiver yields once per later launch.
    Primary(Activations),
    /// Another copy (its process id, where the socket tells) is still starting (it said so) and had not shown its
    /// window when this launch stopped waiting. It has the request and shows its window when it is up.
    Starting(Option<u32>),
    /// Copies kept taking the request and closing the connection unanswered for the whole answer wait: they are going
    /// away, and this launch ends without a word.
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
        Some(dir) => {
            let claim = claim_at(&dir.join("neural-forge.sock"), launch_token(), WAITS);
            if let Claim::Primary(Activations { listening: Some(listening), .. }) = &claim {
                let _ = PRIMARY.set(listening.clone());
            }
            claim
        }
        None => {
            eprintln!("neural-forge: XDG_RUNTIME_DIR is not set, so a second launch can't find this one");
            let (tx, rx) = unbounded_channel();
            // Kept open for the life of the process: the receiver never ends.
            std::mem::forget(tx);
            Claim::Primary(Activations { requests: rx, up: UiUp::default(), listening: None })
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

/// The running copy closed the connection without an answer (an end of file, a reset or a broken pipe).
struct Closed;

fn claim_at(path: &Path, token: Option<String>, waits: Waits) -> Claim {
    // A copy that exits while it is being asked closes the connection unanswered. Its listener may take a moment more
    // to go away, so it is asked again in short steps until nobody listens (this launch becomes the first copy) or a
    // copy answers; if copies keep going away unanswered for the whole answer wait, this launch ends quietly rather
    // than call one stuck. That wait counts from the first closed connection: a copy may have said "starting" for
    // seconds before it went away.
    let mut deadline = None;
    loop {
        if let Ok(claim) = try_claim(path, token.as_deref(), waits) {
            return claim;
        }
        let deadline = *deadline.get_or_insert_with(|| Instant::now() + waits.answer);
        if Instant::now() >= deadline {
            return Claim::Gone;
        }
        std::thread::sleep(RETRY_STEP);
    }
}

fn try_claim(path: &Path, token: Option<&str>, waits: Waits) -> Result<Claim, Closed> {
    let lock = lock(path);
    match UnixStream::connect(path) {
        Ok(stream) => {
            // The wait for the answer needs no lock: other launches may look meanwhile.
            drop(lock);
            return hand_over(&stream, token, waits);
        }
        Err(e) if nobody_listening(&e) => {
            let _ = std::fs::remove_file(path);
        }
        Err(e) => {
            // Something answers to that name but not to us: leave it alone and run without the check.
            eprintln!("neural-forge: single-instance socket {}: {e}", path.display());
        }
    }
    let (tx, rx) = unbounded_channel();
    let up = UiUp::default();
    let listening = match UnixListener::bind(path) {
        Ok(listener) => {
            let listening = Listening { listener: Arc::new(listener), path: path.to_path_buf(), closed: Arc::default() };
            let (thread_listening, thread_up) = (listening.clone(), up.clone());
            std::thread::spawn(move || listen(&thread_listening, &thread_up, &tx));
            Some(listening)
        }
        Err(e) => {
            // Without the socket the app still runs; it just can't be found by a second launch.
            eprintln!("neural-forge: single-instance socket {}: {e}", path.display());
            std::mem::forget(tx);
            None
        }
    };
    Ok(Claim::Primary(Activations { requests: rx, up, listening }))
}

/// The first copy's listener thread: each later launch's request goes to the UI, answered "starting" at once while
/// the window is not up. Ends when the socket is released or nobody takes requests any more.
fn listen(listening: &Listening, up: &UiUp, tx: &async_channel::Sender<Request>) {
    for connection in listening.listener.incoming() {
        let connection = match connection {
            Ok(connection) => connection,
            // Closed by `release`: no more requests.
            Err(_) if listening.closed.load(Ordering::Acquire) => break,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        // Released between the accept and now: dropped unanswered, so that launch asks again and finds nobody.
        if listening.closed.load(Ordering::Acquire) {
            break;
        }
        let mut request = read_request(connection);
        // Until the UI is up, the launch that asks is told so here: the UI's own answer follows once the window is
        // there to show.
        if !up.is_set() {
            let _ = request.reply.write_all(STARTING.as_bytes());
        }
        if tx.send_blocking(request).is_err() {
            break;
        }
    }
}

/// Sends the request to the running copy and waits for its answer: "ok" within `waits.answer`, or "starting" within
/// that and then "ok" within `waits.start` of the request. A copy that said it is starting is not called stuck if its
/// window takes longer still. `Err(Closed)` when the running copy closed the connection without answering: it has
/// exited (or is exiting), which is not a stuck window, and the claim is worth another try.
fn hand_over(stream: &UnixStream, token: Option<&str>, waits: Waits) -> Result<Claim, Closed> {
    let pid = peer_pid(stream);
    let asked = Instant::now();
    let _ = stream.set_write_timeout(Some(waits.answer));
    if writeln!(&*stream, "activate {}", token.unwrap_or("")).is_err() {
        return Err(Closed);
    }
    let mut reader = BufReader::new(stream).take(64);
    let mut line = String::new();
    let _ = stream.set_read_timeout(Some(waits.answer));
    let mut read = reader.read_line(&mut line);
    let starting = read.is_ok() && line == STARTING;
    if starting {
        line.clear();
        let left = waits.start.saturating_sub(asked.elapsed()).max(Duration::from_millis(1));
        let _ = stream.set_read_timeout(Some(left));
        read = reader.read_line(&mut line);
    }
    match read {
        _ if line == ACK => Ok(Claim::Shown),
        // The end with nothing, or a reset (it closed before reading the request): the copy went away. A timeout is
        // another error.
        Ok(0) => Err(Closed),
        Err(e) if matches!(e.kind(), std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe) => Err(Closed),
        _ if starting => Ok(Claim::Starting(pid)),
        _ => Ok(Claim::NotResponding(pid)),
    }
}

/// The process on the other end of the socket (the running copy), from the kernel's record of who bound it.
fn peer_pid(stream: &UnixStream) -> Option<u32> {
    let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED writes one `ucred` into the buffer of that size, on a socket this function borrows.
    let rc = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, std::ptr::from_mut(&mut cred).cast(), &mut len) };
    (rc == 0 && cred.pid > 0).then_some(cred.pid as u32)
}

/// What to tell the user when the running copy said it is starting and its window has not come up in the time waited:
/// information, not an error (it has the request, and its window comes up on its own).
pub fn still_starting_message(pid: Option<u32>) -> String {
    let which = pid.map(|pid| format!(" (process {pid})")).unwrap_or_default();
    format!("Neural Forge is still starting{which}; its window will appear when it is ready.")
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
    use super::{claim_at, nobody_listening, not_responding_message, parse_request, still_starting_message, valid_token, Activations, Claim, Waits, MAX_TOKEN};
    use std::path::Path;
    use std::time::{Duration, Instant};

    const WAIT: Waits = Waits { answer: Duration::from_secs(2), start: Duration::from_secs(15) };

    /// Short waits: `answer` for any answer, `start` for "ok" after "starting".
    fn waits(answer: u64, start: u64) -> Waits {
        Waits { answer: Duration::from_millis(answer), start: Duration::from_millis(start) }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("neural-forge-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    impl Activations {
        fn recv_blocking(&self) -> super::Request {
            self.requests.recv_blocking().expect("the first copy is told")
        }

        /// What `release` does for the first copy of the process.
        fn close(&self) {
            self.listening.as_ref().expect("listening").close();
        }
    }

    fn primary(claim: Claim) -> Activations {
        match claim {
            Claim::Primary(requests) => requests,
            _ => panic!("the first copy is the primary"),
        }
    }

    /// The first copy at `path`, with its window drawn.
    fn up_primary(path: &Path) -> Activations {
        let first = primary(claim_at(path, None, WAIT));
        first.up_signal().set();
        first
    }

    /// A second launch at `path`, with a running copy whose UI answers each request.
    fn second_launch(path: &Path, first: &Activations, token: Option<&str>) -> (Claim, Option<String>) {
        std::thread::scope(|scope| {
            let ui = scope.spawn(|| {
                let request = first.recv_blocking();
                let token = request.token.clone();
                request.acknowledge();
                token
            });
            let claim = claim_at(path, token.map(str::to_string), WAIT);
            (claim, ui.join().unwrap())
        })
    }

    fn launch(path: &Path, waits: Waits) -> std::thread::JoinHandle<Claim> {
        let path = path.to_path_buf();
        std::thread::spawn(move || claim_at(&path, None, waits))
    }

    #[test]
    fn second_launch_reaches_the_first_with_its_token_and_hears_back() {
        let dir = scratch("instance");
        let path = dir.join("test.sock");
        let first = up_primary(&path);
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
        // The first copy has bound its socket but its window is not drawn yet (the graphics are still starting).
        let first = primary(claim_at(&path, None, WAIT));
        let second = launch(&path, waits(300, 5000));
        let request = first.recv_blocking();
        // Its window comes up well after the answer wait, within the start-up wait: the second launch hears "ok".
        std::thread::sleep(Duration::from_millis(800));
        first.up_signal().set();
        request.acknowledge();
        assert!(matches!(second.join().unwrap(), Claim::Shown), "the start-up was waited for, past the answer wait");
        // Once the window is drawn nothing says "starting": a stuck UI gets only the short wait.
        let started = Instant::now();
        let second = launch(&path, waits(300, 5000));
        let request = first.recv_blocking();
        assert!(matches!(second.join().unwrap(), Claim::NotResponding(Some(_))));
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
        drop(request);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_that_stays_starting_is_reported_as_starting_after_the_long_wait() {
        let dir = scratch("starting-slow");
        let path = dir.join("test.sock");
        let first = primary(claim_at(&path, None, WAIT));
        let started = Instant::now();
        let second = launch(&path, waits(200, 900));
        let request = first.recv_blocking();
        // Never answered: the window never comes up. The start-up wait counts from the request.
        let claim = second.join().unwrap();
        let waited = started.elapsed();
        match claim {
            Claim::Starting(pid) => {
                assert_eq!(pid, Some(std::process::id()));
                assert_eq!(
                    still_starting_message(pid),
                    format!("Neural Forge is still starting (process {}); its window will appear when it is ready.", std::process::id())
                );
            }
            _ => panic!("a copy that said it is starting is not stuck"),
        }
        assert!(waited >= Duration::from_millis(900) && waited < Duration::from_secs(3), "{waited:?}");
        assert_eq!(still_starting_message(None), "Neural Forge is still starting; its window will appear when it is ready.");
        drop(request);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn waiting_for_requests_does_not_say_the_window_is_up() {
        // The UI loop asks for requests before the window has drawn anything: a launch meanwhile still hears that this
        // copy is starting (and waits for it), not silence that would call it stuck.
        let dir = scratch("first-frame");
        let path = dir.join("test.sock");
        let first = primary(claim_at(&path, None, WAIT));
        use std::future::Future as _;
        let mut next = std::pin::pin!(first.next());
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(next.as_mut().poll(&mut context).is_pending());
        let second = launch(&path, waits(200, 600));
        let claim = second.join().unwrap();
        assert!(matches!(claim, Claim::Starting(Some(_))), "told \"starting\" until the first frame");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_running_copy_whose_ui_does_not_answer_is_named() {
        let dir = scratch("stuck");
        let path = dir.join("test.sock");
        // The first copy's window was drawn, and now its UI takes no requests (a stuck window).
        let first = up_primary(&path);
        let started = Instant::now();
        match claim_at(&path, Some("token-2".into()), waits(300, 5000)) {
            Claim::NotResponding(pid) => {
                assert_eq!(pid, Some(std::process::id()), "the running copy's process, from the socket");
                assert_eq!(
                    not_responding_message(pid),
                    format!("Neural Forge is already running but not responding. Close it (process {}) and start it again.", std::process::id())
                );
            }
            _ => panic!("no answer in time is not a shown window"),
        }
        assert!(started.elapsed() >= Duration::from_millis(300) && started.elapsed() < Duration::from_secs(2));
        assert_eq!(not_responding_message(None), "Neural Forge is already running but not responding. Close it and start it again.");
        drop(first);
        // An answer that is not "ok" is no answer either.
        let _ = std::fs::remove_file(&path);
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let wrong = std::thread::spawn(move || {
            use std::io::Write;
            let (mut stream, _) = listener.accept().unwrap();
            stream.write_all(b"busy\n").unwrap();
            stream
        });
        assert!(matches!(claim_at(&path, None, waits(300, 5000)), Claim::NotResponding(_)));
        drop(wrong.join());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_that_exits_while_asked_is_not_called_stuck() {
        let dir = scratch("exiting");
        let path = dir.join("test.sock");
        // A copy that takes the request and goes away without answering, in an app's order: the connection closes
        // first, the listener and its socket file a moment later. Asking again at once still reaches that listener
        // (its backlog); the launch keeps asking until it is gone.
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let exiting = {
            let path = path.clone();
            std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                drop(stream);
                std::thread::sleep(Duration::from_millis(300));
                let _ = std::fs::remove_file(&path);
                drop(listener);
            })
        };
        let claim = claim_at(&path, None, waits(2000, 5000));
        exiting.join().unwrap();
        assert!(matches!(claim, Claim::Primary(_)), "the copy that exited left the place free");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_copy_that_dies_after_saying_starting_still_gets_the_retries() {
        let dir = scratch("starting-dies");
        let path = dir.join("test.sock");
        // A copy says "starting", stays silent longer than the answer wait, then goes away (the connection first, the
        // socket a moment later): the retry wait counts from that first close, not from the request.
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let dying = {
            let path = path.clone();
            std::thread::spawn(move || {
                use std::io::Write;
                let (mut stream, _) = listener.accept().unwrap();
                stream.write_all(b"starting\n").unwrap();
                std::thread::sleep(Duration::from_millis(600));
                drop(stream);
                std::thread::sleep(Duration::from_millis(150));
                let _ = std::fs::remove_file(&path);
                drop(listener);
            })
        };
        let claim = claim_at(&path, None, waits(300, 5000));
        dying.join().unwrap();
        assert!(matches!(claim, Claim::Primary(_)), "retried after the copy went away, and became the first copy");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn copies_that_keep_exiting_while_asked_end_the_launch_quietly() {
        let dir = scratch("exiting-always");
        let path = dir.join("test.sock");
        // Every attempt finds a copy that takes the request and closes without answering, and keeps its socket: after
        // the answer wait, asked again in short steps, the launch leaves quietly.
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let closer = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut taken = 0;
                for stream in listener.incoming() {
                    drop(stream);
                    taken += 1;
                    if stop.load(std::sync::atomic::Ordering::Acquire) {
                        break;
                    }
                }
                taken
            })
        };
        let started = Instant::now();
        let claim = claim_at(&path, None, waits(500, 5000));
        let waited = started.elapsed();
        stop.store(true, std::sync::atomic::Ordering::Release);
        drop(std::os::unix::net::UnixStream::connect(&path));
        let taken = closer.join().unwrap();
        assert!(matches!(claim, Claim::Gone), "no stuck-copy message");
        assert!(taken > 2, "asked again in short steps: {taken}");
        assert!(waited >= Duration::from_millis(500) && waited < Duration::from_secs(2), "{waited:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launches_queued_at_a_copy_that_gives_up_start_at_once() {
        let dir = scratch("gives-up");
        let path = dir.join("test.sock");
        // The first copy's window never comes up (no window could be opened).
        let first = primary(claim_at(&path, None, WAIT));
        let started = Instant::now();
        let second = launch(&path, waits(2000, 15000));
        // Its request is queued (told "starting"); then the copy gives up as `ui::no_window` does: release, and drop
        // the requests.
        std::thread::sleep(Duration::from_millis(300));
        first.close();
        drop(first);
        let claim = second.join().unwrap();
        assert!(matches!(claim, Claim::Primary(_)), "not \"still starting\" beside the other copy's error");
        assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_released_copy_lets_the_next_launch_start_at_once() {
        let dir = scratch("released");
        let path = dir.join("test.sock");
        let first = primary(claim_at(&path, None, WAIT));
        first.close();
        assert!(!path.exists(), "the socket file is removed");
        let error = std::os::unix::net::UnixStream::connect(&path).unwrap_err();
        assert!(nobody_listening(&error), "{error}");
        first.close();
        let started = Instant::now();
        assert!(matches!(claim_at(&path, None, waits(2000, 15000)), Claim::Primary(_)), "the next launch is the first copy");
        assert!(started.elapsed() < Duration::from_secs(1), "{:?}", started.elapsed());
        drop(first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_connection_accepted_after_release_is_dropped_unanswered() {
        // The listener thread may have accepted a connection just before the release: it is closed unanswered (not
        // told "starting"), so that launch looks again at once.
        let dir = scratch("released-race");
        let path = dir.join("test.sock");
        let first = primary(claim_at(&path, None, WAIT));
        let listening = first.listening.clone().unwrap();
        listening.closed.store(true, std::sync::atomic::Ordering::Release);
        let mut stream = std::os::unix::net::UnixStream::connect(&path).unwrap();
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let mut answer = String::new();
        let _ = std::io::Read::read_to_string(&mut stream, &mut answer);
        assert_eq!(answer, "", "no \"starting\" from a released copy");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_socket_file_is_replaced() {
        let dir = scratch("stale");
        let path = dir.join("test.sock");
        // A socket nobody listens on any more.
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        assert!(path.exists());
        let first = up_primary(&path);
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
        let _ = claim_at(&path, None, waits(100, 100));
        assert!(path.join("keep").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_copies_started_together_leave_one_primary() {
        let dir = scratch("race");
        let path = dir.join("test.sock");
        let handles: Vec<_> = (0..8).map(|_| launch(&path, waits(100, 100))).collect();
        // All kept until counted: a dropped first copy would stop taking requests.
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
