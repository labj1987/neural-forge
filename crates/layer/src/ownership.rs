//! Session arbitration is independent of window size. The kernel releases the
//! channel lease on process exit/crash; no PID timeout can evict a live game.
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

fn basename(name: &str) -> String {
    name.rsplit(['/', '\\']).next().unwrap_or(name).to_ascii_lowercase()
}

/// The name the target filter judges a process by, from its arguments and `/proc/self/exe`. Under Wine
/// (the executable or argv[0] is `wine*`, e.g. `wine64-preloader`) that is the Windows `.exe` among the
/// arguments, else argv[0]; any other process is its own executable, so a wrapper carrying a game's `.exe`
/// as an argument (gamescope, a launcher script) is judged as itself.
fn judged_name(args: &[String], exe: &str) -> String {
    let wine = [exe, args.first().map_or("", String::as_str)].iter().any(|s| !s.is_empty() && basename(s).starts_with("wine"));
    if wine {
        args.iter().find(|s| s.to_ascii_lowercase().ends_with(".exe")).or_else(|| args.first()).map(|s| basename(s)).unwrap_or_default()
    } else if !exe.is_empty() {
        basename(exe)
    } else {
        args.first().map(|s| basename(s)).unwrap_or_default()
    }
}

/// Whether this process may claim the channel. An explicit `NEURAL_FORGE_TARGET_EXE` (the supervisor
/// writes the game's executable into every launch option it sets) is the final word: the process it
/// names is eligible even when its name is on the denylist (a game whose executable is literally
/// `Launcher.exe`), and no other process is. Only without a target does the denylist decide.
fn allowed(args: &[String], exe_path: &str, target: Option<&str>) -> bool {
    let exe = judged_name(args, exe_path);
    if exe.is_empty() {
        return false;
    }
    if let Some(names) = target.filter(|t| t.split(',').any(|name| !name.trim().is_empty())) {
        return names.split(',').map(str::trim).filter(|name| !name.is_empty()).any(|name| basename(name) == exe);
    }
    !neural_forge_protocol::processes::never_the_game(&exe)
}

/// This process's arguments and executable path.
fn this_process() -> (Vec<String>, String) {
    let args = std::fs::read("/proc/self/cmdline").unwrap_or_default()
        .split(|b| *b == 0).filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned()).collect::<Vec<_>>();
    let exe = std::fs::read_link("/proc/self/exe").map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    (args, exe)
}

static ELIGIBLE: LazyLock<bool> = LazyLock::new(|| {
    let (args, exe) = this_process();
    let target = neural_forge_protocol::env::var("NEURAL_FORGE_TARGET_EXE");
    let ok = allowed(&args, &exe, target.as_deref());
    if !ok { crate::log!("[ownership] process excluded from NeuralForge session"); }
    ok
});

pub fn eligible() -> bool { *ELIGIBLE }

/// A process the layer is enabled in but the target excludes made a game-sized swapchain: it is
/// showing a picture, and unless it is a launcher or helper (`processes::launcher_like`,
/// `processes::game_helper`) it may be the game the target missed. The state log says so once per
/// process, and `doctor` turns it into the fix (the executable to target).
pub fn note_excluded_swapchain(width: u32, height: u32) {
    static SAID: AtomicBool = AtomicBool::new(false);
    let Some(target) = neural_forge_protocol::env::var("NEURAL_FORGE_TARGET_EXE") else { return };
    let name = process_name();
    if eligible() || name.is_empty() || neural_forge_protocol::processes::launcher_like(name) || neural_forge_protocol::processes::game_helper(name) {
        return;
    }
    if !SAID.swap(true, Ordering::Relaxed) {
        crate::event!(neural_forge_protocol::state_log::TARGET_EXCLUDED, "{width}x{height} swapchain; NEURAL_FORGE_TARGET_EXE={target} leaves out {name}");
    }
}

/// This process's executable name as the target filter sees it (the Windows `.exe` for a Proton
/// game), for the GUI's "Game" row.
pub fn process_name() -> &'static str {
    static NAME: LazyLock<String> = LazyLock::new(|| {
        let (args, exe) = this_process();
        judged_name(&args, &exe)
    });
    &NAME
}

fn acquire(path: &str) -> std::io::Result<File> {
    let file = OpenOptions::new().read(true).write(true).create(true)
        .mode(0o600).custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(path)?;
    let meta = file.metadata()?;
    use std::os::unix::fs::MetadataExt;
    if !meta.is_file() || meta.uid() != unsafe { libc::getuid() } || meta.mode() & 0o077 != 0 {
        return Err(std::io::Error::other("unsafe ownership lock"));
    }
    // flock is tied to this open file description, not a reusable PID.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(file)
}

/// How long the owner may go without presenting before it lets the channel go. A game presents
/// every frame, its loading screens included; a launcher or helper that claimed the channel and then
/// sat minimized or closed its window stops, and must not keep the game it started from claiming it.
const LEASE_IDLE: std::time::Duration = std::time::Duration::from_secs(10);
/// How often a process without the lease tries for it again.
const RETRY_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

/// The lease, while this process holds it. Kept across swapchain recreation. The lock file is never
/// unlinked: doing so would let a new process lock a different inode.
struct Lease {
    path: String,
    _file: File,
}

static LEASE: Mutex<Option<Lease>> = Mutex::new(None);
/// Whether this process holds the lease: what every user of the channel checks (`shm::ShmClient`,
/// the model server's loop) before touching it, a relaxed load.
static HELD: AtomicBool = AtomicBool::new(false);
/// Milliseconds since [`epoch`] of this process's last present, and of its last attempt to claim.
static LAST_PRESENT_MS: AtomicU64 = AtomicU64::new(0);
static LAST_ATTEMPT_MS: AtomicU64 = AtomicU64::new(u64::MAX);
/// Bumped on every successful claim: a claim after a release is a new session on the channel.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// How many times this process has taken the lease (0: never).
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

fn epoch() -> std::time::Instant {
    static EPOCH: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);
    *EPOCH
}

fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64
}

/// Whether this process holds the channel's lease now. Released when it stops presenting
/// ([`LEASE_IDLE`]); a process that does not hold it must not read or write the channel.
pub fn held() -> bool {
    HELD.load(Ordering::Acquire)
}

/// Called on every present of this process: the lease follows presenting.
pub fn note_present() {
    LAST_PRESENT_MS.store(now_ms(), Ordering::Relaxed);
}

/// Takes (or keeps) the lease on `shm_path`'s channel. Cheap while held; a process that does not hold
/// it tries the lock at most once per [`RETRY_EVERY`].
pub fn claim(shm_path: &str) -> bool {
    if !eligible() { return false; }
    if held() {
        return LEASE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|l| l.path == shm_path);
    }
    let now = now_ms();
    let last = LAST_ATTEMPT_MS.load(Ordering::Relaxed);
    if last != u64::MAX && now.saturating_sub(last) < RETRY_EVERY.as_millis() as u64 {
        return false;
    }
    LAST_ATTEMPT_MS.store(now, Ordering::Relaxed);
    let mut lease = LEASE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = &*lease { return l.path == shm_path; }
    match acquire(&format!("{shm_path}.owner")) {
        Ok(file) => {
            crate::log!("[ownership] pid {} owns {}", std::process::id(), shm_path);
            *lease = Some(Lease { path: shm_path.to_owned(), _file: file });
            // A claim counts as presenting: the claim happens on a present (or a submit around one).
            note_present();
            GENERATION.fetch_add(1, Ordering::AcqRel);
            HELD.store(true, Ordering::Release);
            start_watchdog();
            true
        }
        Err(_) => false, // another process owns this channel; present untouched
    }
}

/// Releases the lease once this process has not presented for [`LEASE_IDLE`]. One thread per
/// process, started with the first claim; it only sleeps and compares two numbers.
fn start_watchdog() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let spawned = std::thread::Builder::new().name("nf-lease".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        release_if_idle(now_ms(), LEASE_IDLE);
    });
    if spawned.is_err() {
        // Without the watchdog the lease is kept until the process exits, as before.
        STARTED.store(false, Ordering::Relaxed);
        crate::log!("[ownership] could not start the lease watchdog; the lease is kept until this process exits");
    }
}

/// The watchdog's step: lets the lease go when the last present is more than `idle` ago.
fn release_if_idle(now: u64, idle: std::time::Duration) -> bool {
    if !held() || now.saturating_sub(LAST_PRESENT_MS.load(Ordering::Relaxed)) < idle.as_millis() as u64 {
        return false;
    }
    let mut lease = LEASE.lock().unwrap_or_else(|e| e.into_inner());
    HELD.store(false, Ordering::Release);
    if let Some(l) = lease.take() {
        crate::log!("[ownership] pid {} has not presented for {} s; released {}", std::process::id(), idle.as_secs(), l.path);
        crate::logging::flush();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Held by the lease tests: one spawns a child process, and a fork taken while another
    /// test holds a lease briefly shares that lease's open file description (and its lock).
    static LEASE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[test]
    fn launchers_cannot_claim_unless_explicitly_targeted() {
        for exe in ["explorer.exe", "Xalia.exe", "Launcher.exe", "SocialClubHelper.exe", "RockstarService.exe", "Rockstar Games Launcher.exe", "Social Club Helper.exe"] {
            let args = vec!["wine64-preloader".into(), format!("C:\\Rockstar Games\\{exe}")];
            assert!(!allowed(&args, "/usr/bin/wine64-preloader", None));
            assert!(!allowed(&args, "/usr/bin/wine64-preloader", Some("GTA5_Enhanced.exe")), "{exe}: another game is the target");
            // A game whose executable is literally one of these names works when it is the target.
            assert!(allowed(&args, "/usr/bin/wine64-preloader", Some(exe)), "{exe}, targeted");
        }
        // An empty or blank target is no target: the denylist decides.
        let launcher = vec!["wine64-preloader".to_string(), "C:\\x\\Launcher.exe".into()];
        assert!(!allowed(&launcher, "/usr/bin/wine64-preloader", Some(" , ")));
        assert!(allowed(&["Z:\\g\\Game.exe".into()], "/usr/bin/wine64-preloader", Some("")));
        assert!(allowed(&launcher, "/usr/bin/wine64-preloader", Some("Other.exe, Launcher.exe")));
        let wine = "/home/u/.steam/proton/files/bin/wine64-preloader";
        assert!(allowed(&["Z:\\games\\GTA5_Enhanced.exe".into()], wine, Some("GTA5_Enhanced.exe")));
        assert!(!allowed(&["another-game.exe".into()], wine, Some("GTA5_Enhanced.exe")));
        assert!(!allowed(&[], "", None));
    }

    /// Only Wine's `.exe` argument names the process: a wrapper carrying the game's `.exe` as an argument is
    /// judged as itself, gamescope is excluded, and a native binary is its own executable.
    #[test]
    fn wrappers_are_judged_by_their_own_executable() {
        let target = Some("GTA5_Enhanced.exe");
        // Wine with the preloader as argv[0], followed by the Windows exe.
        let preloader = vec!["/usr/bin/wine-preloader".to_string(), "Z:\\games\\GTA5_Enhanced.exe".into()];
        assert!(allowed(&preloader, "/usr/bin/wine-preloader", target));
        assert_eq!(judged_name(&preloader, "/usr/bin/wine-preloader"), "gta5_enhanced.exe");
        // gamescope carrying the game's command line.
        let gamescope = vec!["gamescope".to_string(), "-w".into(), "2560".into(), "--".into(), "wine".into(), "GTA5_Enhanced.exe".into()];
        assert!(!allowed(&gamescope, "/usr/bin/gamescope", target));
        assert!(!allowed(&gamescope, "/usr/bin/gamescope", None), "gamescope is never the game");
        // A native binary, with an argument that happens to end in .exe.
        let native = vec!["./mygame".to_string(), "--log=out.exe".into()];
        assert_eq!(judged_name(&native, "/opt/mygame/mygame"), "mygame");
        assert!(allowed(&native, "/opt/mygame/mygame", Some("mygame")));
        assert!(!allowed(&native, "/opt/mygame/mygame", Some("out.exe")));
    }
    /// The real launcher names (as Wine's `.exe` argument) never claim unless they are the target; a game still does.
    #[test]
    fn store_launchers_and_their_helpers_never_claim() {
        let wine = "/usr/bin/wine64-preloader";
        for exe in [
            "upc.exe", "UbisoftConnect.exe", "UplayWebCore.exe", "EADesktop.exe", "EALauncher.exe", "EABackgroundService.exe",
            "Battle.net.exe", "Agent.exe", "Battle.net Launcher.exe", "EpicGamesLauncher.exe", "EpicWebHelper.exe",
            "EOSOverlayRenderer-Win64-Shipping.exe", "GalaxyClient.exe", "GalaxyClient Helper.exe", "RiotClientServices.exe",
            "CrashReportClient.exe", "UnityCrashHandler64.exe", "QtWebEngineProcess.exe", "steamerrorreporter.exe",
        ] {
            let args = vec![wine.to_string(), format!("C:\\Program Files\\{exe}")];
            assert!(!allowed(&args, wine, None), "{exe}");
            assert!(!allowed(&args, wine, Some("Cyberpunk2077.exe")), "{exe}, another game targeted");
        }
        for game in ["GTA5_Enhanced.exe", "ACValhalla.exe", "Cyberpunk2077.exe", "eldenring.exe"] {
            assert!(allowed(&[wine.to_string(), format!("Z:\\g\\{game}")], wine, None), "{game}");
        }
    }

    /// The lease goes when the owner stops presenting, and not before.
    #[test]
    fn the_lease_is_released_after_presents_stop() {
        let _serial = LEASE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = std::env::temp_dir().join(format!("neural-forge-idle-test-{}", std::process::id()));
        let file = acquire(path.to_str().unwrap()).unwrap();
        *LEASE.lock().unwrap() = Some(Lease { path: "test".into(), _file: file });
        HELD.store(true, Ordering::Release);
        LAST_PRESENT_MS.store(1_000, Ordering::Relaxed);
        let idle = std::time::Duration::from_secs(10);
        assert!(!release_if_idle(10_999, idle), "presented within the window");
        assert!(held());
        assert!(release_if_idle(11_000, idle));
        assert!(!held() && LEASE.lock().unwrap().is_none());
        // The kernel lock went with it: another process (here another open) can take the channel.
        assert!(acquire(path.to_str().unwrap()).is_ok());
        assert!(!release_if_idle(99_000, idle), "nothing left to release");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn crashed_process_releases_lease() {
        let _serial = LEASE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::io::{BufRead, BufReader};
        use std::process::{Command, Stdio};
        let path = std::env::temp_dir().join(format!("neural-forge-crash-test-{}", std::process::id()));
        // Create with the same private permissions as production, then let an
        // independent process acquire the actual kernel lock.
        drop(acquire(path.to_str().unwrap()).unwrap());
        let mut child = Command::new("python3").args(["-c",
            "import fcntl,sys,time; f=open(sys.argv[1], 'r+'); fcntl.flock(f, fcntl.LOCK_EX); print('ready', flush=True); time.sleep(30)"])
            .arg(&path).stdout(Stdio::piped()).spawn().unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut ready).unwrap();
        let excluded = acquire(path.to_str().unwrap()).is_err();
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(ready.trim(), "ready");
        assert!(excluded);
        assert!(acquire(path.to_str().unwrap()).is_ok());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn lease_excludes_other_opens_and_recovers_after_close() {
        let _serial = LEASE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = std::env::temp_dir().join(format!("neural-forge-owner-test-{}", std::process::id()));
        let path = path.to_str().unwrap();
        let first = acquire(path).unwrap();
        assert!(acquire(path).is_err());
        drop(first);
        assert!(acquire(path).is_ok());
        std::fs::remove_file(path).unwrap();
    }
}
