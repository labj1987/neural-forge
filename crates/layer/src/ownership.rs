//! Session arbitration is independent of window size. The kernel releases the
//! channel lease on process exit/crash; no PID timeout can evict a live game.
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::{LazyLock, Mutex};

fn basename(name: &str) -> String {
    name.rsplit(['/', '\\']).next().unwrap_or(name).to_ascii_lowercase()
}

fn allowed(args: &[String], target: Option<&str>) -> bool {
    // Wine may expose wine-preloader as argv[0], followed by the Windows exe.
    let exe = args.iter().find(|s| s.to_ascii_lowercase().ends_with(".exe"))
        .or_else(|| args.first()).map(|s| basename(s)).unwrap_or_default();
    let compact = exe.replace([' ', '-', '_'], "");
    if ["rockstar", "socialclub", "xalia"].iter().any(|prefix| compact.starts_with(prefix)) {
        return false;
    }
    if exe.is_empty() || ["explorer.exe", "xalia.exe", "launcher.exe",
        "launcherpatcher.exe", "rockstarservice.exe", "rockstarlauncher.exe",
        "socialclubhelper.exe", "socialclub.exe", "steam.exe", "steamwebhelper.exe",
        "gameoverlayui.exe", "neural-forge-helper.exe", "neuralforge-helper.exe", "winedevice.exe", "services.exe"]
        .contains(&exe.as_str()) { return false; }
    target.map(|names| names.split(',').any(|name| basename(name.trim()) == exe))
        .unwrap_or(true)
}

static ELIGIBLE: LazyLock<bool> = LazyLock::new(|| {
    let args = std::fs::read("/proc/self/cmdline").unwrap_or_default()
        .split(|b| *b == 0).filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned()).collect::<Vec<_>>();
    let target = neural_forge_protocol::env::var("NEURAL_FORGE_TARGET_EXE");
    let ok = allowed(&args, target.as_deref());
    if !ok { crate::log!("[ownership] process excluded from NeuralForge session"); }
    ok
});

pub fn eligible() -> bool { *ELIGIBLE }

/// This process's executable name as the target filter sees it (the Windows `.exe` for a Proton
/// game), for the GUI's "Game" row.
pub fn process_name() -> &'static str {
    static NAME: LazyLock<String> = LazyLock::new(|| {
        let args = std::fs::read("/proc/self/cmdline").unwrap_or_default()
            .split(|b| *b == 0).filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned()).collect::<Vec<_>>();
        args.iter().find(|s| s.to_ascii_lowercase().ends_with(".exe")).or_else(|| args.first())
            .map(|s| s.rsplit(['/', '\\']).next().unwrap_or(s).to_owned()).unwrap_or_default()
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

// Keep the lease even during swapchain recreation and helper restart. Never unlink
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
            assert!(!allowed(&args, None));
            assert!(!allowed(&args, Some(exe)));
        }
        assert!(allowed(&["Z:\\games\\GTA5_Enhanced.exe".into()], Some("GTA5_Enhanced.exe")));
        assert!(!allowed(&["another-game.exe".into()], Some("GTA5_Enhanced.exe")));
        assert!(!allowed(&[], None));
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
