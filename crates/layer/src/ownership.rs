//! Session arbitration is independent of window size. The kernel releases the
//! channel lease on process exit/crash; no PID timeout can evict a live game.
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
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

fn allowed(args: &[String], exe_path: &str, target: Option<&str>) -> bool {
    let exe = judged_name(args, exe_path);
    let compact = exe.replace([' ', '-', '_'], "");
    if ["rockstar", "socialclub", "xalia"].iter().any(|prefix| compact.starts_with(prefix)) {
        return false;
    }
    if exe.is_empty() || ["explorer.exe", "xalia.exe", "launcher.exe",
        "launcherpatcher.exe", "rockstarservice.exe", "rockstarlauncher.exe",
        "socialclubhelper.exe", "socialclub.exe", "steam.exe", "steamwebhelper.exe",
        "gameoverlayui.exe", "winedevice.exe", "services.exe", "gamescope"]
        .contains(&exe.as_str()) { return false; }
    target.map(|names| names.split(',').any(|name| basename(name.trim()) == exe))
        .unwrap_or(true)
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

// Keep the lease even during swapchain recreation. Never unlink
// the lock file: doing so would let a new process lock a different inode.
static LEASE: Mutex<Option<(String, File)>> = Mutex::new(None);
pub fn claim(shm_path: &str) -> bool {
    if !eligible() { return false; }
    let mut lease = LEASE.lock().unwrap();
    if let Some((path, _)) = &*lease { return path == shm_path; }
    match acquire(&format!("{shm_path}.owner")) {
        Ok(file) => {
            crate::log!("[ownership] pid {} owns {}", std::process::id(), shm_path);
            *lease = Some((shm_path.to_owned(), file)); true
        }
        Err(_) => false, // another process owns this channel; present untouched
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Held by the lease tests: one spawns a child process, and a fork taken while another
    /// test holds a lease briefly shares that lease's open file description (and its lock).
    static LEASE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    #[test]
    fn launchers_cannot_claim_even_when_explicitly_targeted() {
        for exe in ["explorer.exe", "Xalia.exe", "Launcher.exe", "SocialClubHelper.exe", "RockstarService.exe", "Rockstar Games Launcher.exe", "Social Club Helper.exe"] {
            let args = vec!["wine64-preloader".into(), format!("C:\\Rockstar Games\\{exe}")];
            assert!(!allowed(&args, "/usr/bin/wine64-preloader", None));
            assert!(!allowed(&args, "/usr/bin/wine64-preloader", Some(exe)));
        }
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
