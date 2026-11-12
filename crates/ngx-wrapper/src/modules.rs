//! The NGX modules a game has loaded, and their file versions, for `doctor` and the report (docs/PROTON_NGX.md).
//!
//! After a feature is created ([`note_loaded`]) the wrapper lists the process's loaded modules, keeps the NGX ones
//! (NVIDIA's core, the wrapper itself, `nvngx.dll` and the DLSS snippets `nvngx_dlss.dll`, `nvngx_dlssg.dll`,
//! `nvngx_dlssd.dll`, `nvngx_dlssnr.dll`, ...: whichever path they were loaded from, the game's folder, the NGX models
//! folder an over-the-air update fills, the driver's folder) and writes their paths and file versions to
//! `<state>/neural-forge/ngx-modules/<game>-<exe>.txt` (and to the text log, when it is on). Neural Forge's `doctor`
//! and report read that file; the supervisor's `ngx_versions.rs` parses the format below. A set that did not change is
//! not written again, so a game that creates features all the time costs one module listing each.
//!
//! `<state>` is the supervisor's state directory ([`dir`]): Wine does not pass `HOME` into the Windows environment (it
//! becomes `WINEHOMEDIR`), so the home is read from there when neither `XDG_STATE_HOME` nor `HOME` is set. `<game>` is
//! the Steam app id Steam gives the game (`SteamAppId`, else `SteamGameId`), else a hash of the executable's path, so
//! two games whose executables share a name keep apart ([`state_file_name`]).
//!
//! The format, one file per game and executable, replaced by every run:
//!
//! ```text
//! neural-forge-ngx-modules 1
//! wrapper=4.0.0
//! exe=C:\...\game.exe
//! core<TAB>_nvngx.dll<TAB>570.133.0.0<TAB>Z:\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll
//! snippet<TAB>nvngx_dlss.dll<TAB>310.4.0.0<TAB>C:\...\nvngx_dlss.dll
//! ```
//!
//! The role is `wrapper`, `core` or `snippet`; a version that could not be read is `-`.

/// Whether a loaded module's file name is one of NGX's: `_nvngx.dll`, or `nvngx*.dll`.
pub fn is_ngx_module(file_name: &str) -> bool {
    let name = file_name.to_ascii_lowercase();
    name == "_nvngx.dll" || (name.starts_with("nvngx") && name.ends_with(".dll"))
}

/// The last component of a Windows or Unix path.
pub fn file_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// Whether the wrapper writes the file: only in a process the layer is switched on in, as for the ring.
pub fn wanted(var: impl Fn(&str) -> Option<String>) -> bool {
    var("NEURAL_FORGE_ENABLE").is_some_and(|v| v == "1") && !var("NEURAL_FORGE_DISABLE").is_some_and(|v| v == "1")
}

/// The state directory's `neural-forge/ngx-modules` as a Unix path (the supervisor's `paths::state_dir`, the same
/// rule: `XDG_STATE_HOME`, else `HOME`'s `.local/state`). In a Wine process `HOME` is not in the environment; Wine
/// gives the Unix home as `WINEHOMEDIR` instead ([`unix_home_from_wine`]). `None` when none of them names a Unix path.
pub fn dir(var: impl Fn(&str) -> Option<String>) -> Option<String> {
    let set = |name: &str| var(name).filter(|v| !v.is_empty());
    let home = || set("HOME").or_else(|| set("WINEHOMEDIR").and_then(|w| unix_home_from_wine(&w)));
    let state = set("XDG_STATE_HOME").or_else(|| home().map(|h| format!("{}/.local/state", h.trim_end_matches('/'))))?;
    Some(format!("{}/neural-forge/ngx-modules", state.trim_end_matches('/')))
}

/// The Unix path of Wine's `WINEHOMEDIR`, an NT path: `\??\Z:\home\user` (Z: is the Unix root in every prefix Wine
/// and Proton make) or `\??\unix\home\user` (no drive maps the home). `None` for another drive, whose Unix
/// location is not known here.
pub fn unix_home_from_wine(nt: &str) -> Option<String> {
    let rest = nt.trim_prefix(r"\??\");
    // Bytes, not `rest[..n]`: a slice inside a multi-byte character panics. The ASCII prefix matched, so `[n..]` is a boundary.
    let starts = |prefix: &str| rest.as_bytes().get(..prefix.len()).is_some_and(|b| b.eq_ignore_ascii_case(prefix.as_bytes()));
    let rest = if starts("z:") {
        &rest[2..]
    } else if starts("unix") {
        &rest[4..]
    } else {
        return None;
    };
    let path = rest.replace('\\', "/");
    path.starts_with('/').then_some(path)
}

/// The Windows-form path of the state directory ([`dir`]) the wrapper creates and writes in. With `XDG_STATE_HOME` or
/// `HOME` (Unix paths) it is `Z:` and the path, as ever. Without them it comes from `WINEHOMEDIR`: for `\??\Z:\...`
/// the same `Z:` path; for `\??\unix\...` or a home on another drive it is the NT path itself, `\\?\` and the rest
/// of `WINEHOMEDIR` and `\.local\state\neural-forge\ngx-modules`, which needs no `Z:` drive. `None` when none names a path.
pub fn win_dir(var: impl Fn(&str) -> Option<String>) -> Option<String> {
    let set = |name: &str| var(name).filter(|v| !v.is_empty());
    if set("XDG_STATE_HOME").is_some() || set("HOME").is_some() {
        return dir(var).map(|d| format!("Z:{d}").replace('/', "\\"));
    }
    let wine = set("WINEHOMEDIR")?;
    let rest = wine.trim_prefix(r"\??\");
    let bytes = rest.as_bytes();
    if bytes.get(..2).is_some_and(|b| b.eq_ignore_ascii_case(b"z:")) {
        return dir(var).map(|d| format!("Z:{d}").replace('/', "\\"));
    }
    let drive = bytes.get(1) == Some(&b':') && bytes.first().is_some_and(u8::is_ascii_alphabetic);
    let unix = bytes.get(..5).is_some_and(|b| b.eq_ignore_ascii_case(b"unix\\"));
    let home = rest.trim_end_matches('\\');
    // Something must follow the drive or `unix` (a bare `C:` or `unix` is no home).
    if !(drive && home.len() > 2 || unix && home.len() > 4) {
        return None;
    }
    Some(format!(r"\\?\{home}\.local\state\neural-forge\ngx-modules"))
}

/// The id the file is kept under for the game: the Steam app id Steam puts in the game's environment (`SteamAppId`,
/// else `SteamGameId`, which a non-Steam shortcut has), digits and not `0`; else `x` and a hash of the executable's
/// lower-case path, so executables of the same name in different games keep apart.
pub fn game_id(exe: &str, var: impl Fn(&str) -> Option<String>) -> String {
    let steam =
        |name: &str| var(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) && v.bytes().any(|b| b != b'0'));
    steam("SteamAppId").or_else(|| steam("SteamGameId")).unwrap_or_else(|| {
        // FNV-1a, 64 bits: stable across runs and builds (std's hasher is not).
        let hash = exe.to_ascii_lowercase().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
        format!("x{hash:016x}")
    })
}

/// The file's name for the game executable at `exe`: the [`game_id`], `-`, the executable's lower-case stem with
/// anything but letters, digits, `-`, `_` and `.` replaced, `.txt` after.
pub fn state_file_name(exe: &str, var: impl Fn(&str) -> Option<String>) -> String {
    let stem = file_name(exe);
    let stem = stem.rsplit_once('.').map_or(stem, |(s, _)| s);
    let clean: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c.to_ascii_lowercase() } else { '_' }).collect();
    format!("{}-{}.txt", game_id(exe, var), if clean.is_empty() { "game" } else { &clean })
}

/// Lists into a buffer that grows to what the lister says it needs: `fill` fills `buf` and returns how many entries
/// there are in all (more than fit when the buffer is short), `None` on failure (an empty list). Tries again while
/// the count grows between calls (a module loaded meanwhile), a few times, then keeps what fit.
pub fn grow_list<T: Copy>(zero: T, mut fill: impl FnMut(&mut [T]) -> Option<usize>) -> Vec<T> {
    let mut buf = vec![zero; 256];
    for _ in 0..8 {
        let Some(n) = fill(&mut buf) else { return Vec::new() };
        if n <= buf.len() {
            buf.truncate(n);
            return buf;
        }
        buf.resize(n + 32, zero);
    }
    buf
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub role: &'static str,
    pub path: String,
    /// `VS_FIXEDFILEINFO`'s file version, `None` when the file has none or it could not be read.
    pub version: Option<String>,
}

/// The four file-version parts as `a.b.c.d`.
pub fn version_text(ms: u32, ls: u32) -> String {
    format!("{}.{}.{}.{}", ms >> 16, ms & 0xffff, ls >> 16, ls & 0xffff)
}

/// The file's text (the module docs).
pub fn render(wrapper_version: &str, exe: &str, modules: &[Module]) -> String {
    let mut s = format!("neural-forge-ngx-modules 1\nwrapper={wrapper_version}\nexe={exe}\n");
    for m in modules {
        s += &format!("{}\t{}\t{}\t{}\n", m.role, file_name(&m.path), m.version.as_deref().unwrap_or("-"), m.path);
    }
    s
}

/// The text log's line for one module.
pub fn log_line(m: &Module) -> String {
    format!("NGX module {}: {} {}", m.role, m.version.as_deref().unwrap_or("no version"), m.path)
}

#[cfg(windows)]
pub use win::note_loaded;

#[cfg(windows)]
mod win {
    use super::*;
    use crate::win::{log, logging, wide, Handle};
    use core::ffi::c_void;
    use std::sync::Mutex;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn K32EnumProcessModules(process: Handle, modules: *mut Handle, bytes: u32, needed: *mut u32) -> i32;
        fn GetModuleFileNameW(module: Handle, buf: *mut u16, len: u32) -> u32;
        fn GetModuleHandleExW(flags: u32, address: *const c_void, out: *mut Handle) -> i32;
        fn LoadLibraryW(name: *const u16) -> Handle;
        fn GetProcAddress(module: Handle, name: *const u8) -> *const c_void;
    }

    const FROM_ADDRESS: u32 = 4;
    const UNCHANGED_REFCOUNT: u32 = 2;

    fn module_path(module: Handle) -> String {
        let mut buf = [0u16; 1024];
        let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        String::from_utf16_lossy(&buf[..n.min(buf.len())])
    }

    /// The modules the process has loaded: handle and path, all of them however many (`needed` says how many).
    fn loaded() -> Vec<(Handle, String)> {
        let handles = grow_list(core::ptr::null_mut::<c_void>(), |buf| {
            let mut needed = 0u32;
            let ok = unsafe { K32EnumProcessModules(GetCurrentProcess(), buf.as_mut_ptr(), size_of_val(buf) as u32, &mut needed) };
            (ok != 0).then_some(needed as usize / size_of::<Handle>())
        });
        handles.into_iter().map(|h| (h, module_path(h))).collect()
    }

    type SizeFn = unsafe extern "system" fn(*const u16, *mut u32) -> u32;
    type InfoFn = unsafe extern "system" fn(*const u16, u32, u32, *mut c_void) -> i32;
    type QueryFn = unsafe extern "system" fn(*const c_void, *const u16, *mut *mut c_void, *mut u32) -> i32;

    /// `VS_FIXEDFILEINFO`'s file version of `path`, through `version.dll` (loaded here, not imported: the wrapper's
    /// import table stays as it is).
    fn file_version(path: &str) -> Option<String> {
        unsafe {
            let lib = LoadLibraryW(wide("version.dll").as_ptr());
            if lib.is_null() {
                return None;
            }
            let sym = |name: &[u8]| GetProcAddress(lib, name.as_ptr());
            let (size, info, query) = (sym(b"GetFileVersionInfoSizeW\0"), sym(b"GetFileVersionInfoW\0"), sym(b"VerQueryValueW\0"));
            if size.is_null() || info.is_null() || query.is_null() {
                return None;
            }
            let size = core::mem::transmute::<*const c_void, SizeFn>(size);
            let info = core::mem::transmute::<*const c_void, InfoFn>(info);
            let query = core::mem::transmute::<*const c_void, QueryFn>(query);
            let wpath = wide(path);
            let mut handle = 0u32;
            let len = size(wpath.as_ptr(), &mut handle);
            if len == 0 {
                return None;
            }
            let mut block = vec![0u8; len as usize];
            if info(wpath.as_ptr(), 0, len, block.as_mut_ptr().cast()) == 0 {
                return None;
            }
            let (mut fixed, mut fixed_len) = (core::ptr::null_mut::<c_void>(), 0u32);
            if query(block.as_ptr().cast(), wide("\\").as_ptr(), &mut fixed, &mut fixed_len) == 0 || fixed.is_null() || (fixed_len as usize) < 16 {
                return None;
            }
            let words = fixed.cast::<u32>();
            if words.read_unaligned() != 0xfeef_04bd {
                return None;
            }
            Some(version_text(words.add(2).read_unaligned(), words.add(3).read_unaligned()))
        }
    }

    /// The paths last written, so an unchanged set costs the listing only.
    static LAST: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// Lists the NGX modules the process has loaded and, when the set changed since the last call, logs them and
    /// writes the state file (the module docs). Called after a feature was created: NGX loads the core before the
    /// first call, and a snippet when its feature is created. Never fails the game: a file that cannot be written is
    /// skipped.
    pub fn note_loaded() {
        let mut own = core::ptr::null_mut();
        unsafe { GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, note_loaded as *const c_void, &mut own) };
        let mut found: Vec<(Handle, String)> = loaded().into_iter().filter(|(_, p)| is_ngx_module(file_name(p))).collect();
        found.sort_by_key(|a| a.1.to_ascii_lowercase());
        let paths: Vec<String> = found.iter().map(|(_, p)| p.clone()).collect();
        // Held through the write, so two threads creating features at once cannot write an older set last.
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if *last == paths {
            return;
        }
        *last = paths;
        let modules: Vec<Module> = found
            .into_iter()
            .map(|(h, path)| {
                if h == own {
                    Module { role: "wrapper", version: Some(env!("CARGO_PKG_VERSION").to_string()), path }
                } else if file_name(&path).eq_ignore_ascii_case("_nvngx.dll") {
                    Module { role: "core", version: file_version(&path), path }
                } else {
                    Module { role: "snippet", version: file_version(&path), path }
                }
            })
            .collect();
        if logging() {
            for m in &modules {
                log(&log_line(m));
            }
        }
        let var = |name: &str| std::env::var(name).ok();
        if !wanted(var) {
            return;
        }
        let exe = module_path(core::ptr::null_mut());
        let Some(dir) = win_dir(var) else { return };
        let written = std::fs::create_dir_all(&dir).is_ok()
            && std::fs::write(format!("{dir}\\{}", state_file_name(&exe, var)), render(env!("CARGO_PKG_VERSION"), &exe, &modules)).is_ok();
        if !written {
            // Tried again at the next feature.
            last.clear();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Under Wine (`cargo test --target x86_64-pc-windows-gnu`, the runner in the cargo config): the module
        /// listing and the version read work on a real system DLL.
        #[test]
        fn lists_modules_and_reads_a_file_version() {
            let modules = loaded();
            assert!(modules.iter().any(|(_, p)| file_name(p).eq_ignore_ascii_case("kernel32.dll")), "{modules:?}");
            let kernel32 = modules.iter().find(|(_, p)| file_name(p).eq_ignore_ascii_case("kernel32.dll")).unwrap().1.clone();
            let version = file_version(&kernel32).unwrap_or_default();
            assert!(version.split('.').count() == 4 && version.split('.').all(|n| n.parse::<u32>().is_ok()), "{kernel32}: {version:?}");
            assert_eq!(file_version(r"C:\windows\system32\no-such-file.dll"), None);
        }

        /// Under Wine: a directory made and written through the `\\?\unix\...` NT form (no drive needed) is the Unix
        /// directory, and the `Z:` form of the same path reads it back.
        #[test]
        fn writes_through_the_unix_nt_path() {
            let unix_home = format!("/tmp/nf-ngx-wrapper-test-{}", std::process::id());
            let nt_home = format!(r"\??\unix{}", unix_home.replace('/', "\\"));
            let var = |name: &str| (name == "WINEHOMEDIR").then(|| nt_home.clone());
            let dir = win_dir(var).unwrap();
            assert!(dir.starts_with(r"\\?\unix\tmp\"), "{dir}");
            let name = state_file_name(r"C:\g\game.exe", var);
            let result = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(format!("{dir}\\{name}"), "ok"));
            let z_home = format!("Z:{unix_home}").replace('/', "\\");
            let back = std::fs::read_to_string(format!(r"{z_home}\.local\state\neural-forge\ngx-modules\{name}"));
            let _ = std::fs::remove_dir_all(&z_home);
            result.unwrap();
            assert_eq!(back.unwrap(), "ok");
        }

        /// Under Wine, with the process's real environment (Wine passes no `HOME`; run it with `env -u HOME -u
        /// XDG_STATE_HOME` to be sure): the state directory is found, through `WINEHOMEDIR`, and its home exists.
        #[test]
        fn finds_the_state_directory_without_home() {
            let var = |name: &str| std::env::var(name).ok();
            let found = dir(var).unwrap_or_else(|| panic!("no state directory: WINEHOMEDIR={:?}", std::env::var("WINEHOMEDIR")));
            if var("XDG_STATE_HOME").is_none() && var("HOME").is_none() {
                let home = unix_home_from_wine(&var("WINEHOMEDIR").unwrap()).unwrap();
                assert_eq!(found, format!("{home}/.local/state/neural-forge/ngx-modules"));
                assert!(std::fs::metadata(format!("Z:{home}").replace('/', "\\")).is_ok_and(|m| m.is_dir()), "{home}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
    }

    #[test]
    fn picks_the_ngx_modules_by_file_name() {
        for name in ["_nvngx.dll", "nvngx.dll", "nvngx_dlss.dll", "NVNGX_DLSSG.DLL", "nvngx_dlssd.dll", "nvngx_dlssnr.dll", "nvngx_deepdvc.dll"] {
            assert!(is_ngx_module(name), "{name}");
        }
        for name in ["sl.dlss.dll", "nvngx_dlss.dll.bak", "nvngx", "_nvngx.dll.old", "d3d12.dll"] {
            assert!(!is_ngx_module(name), "{name}");
        }
        assert_eq!(file_name(r"C:\Games\Cyberpunk 2077\bin\x64\nvngx_dlss.dll"), "nvngx_dlss.dll");
        assert_eq!(file_name("/usr/lib/x.dll"), "x.dll");
        assert_eq!(file_name("plain.dll"), "plain.dll");
    }

    #[test]
    fn writes_only_where_the_layer_is_on() {
        assert!(wanted(env(&[("NEURAL_FORGE_ENABLE", "1")])));
        assert!(!wanted(env(&[])));
        assert!(!wanted(env(&[("NEURAL_FORGE_ENABLE", "1"), ("NEURAL_FORGE_DISABLE", "1")])));
    }

    #[test]
    fn state_directory_and_file_names() {
        assert_eq!(dir(env(&[("XDG_STATE_HOME", "/s"), ("HOME", "/h")])).as_deref(), Some("/s/neural-forge/ngx-modules"));
        assert_eq!(dir(env(&[("XDG_STATE_HOME", ""), ("HOME", "/h/")])).as_deref(), Some("/h/.local/state/neural-forge/ngx-modules"));
        assert_eq!(dir(env(&[])), None);
        let steam = env(&[("SteamAppId", "1091500"), ("SteamGameId", "1091500")]);
        assert_eq!(state_file_name(r"Z:\mnt\Steam\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe", &steam), "1091500-cyberpunk2077.txt");
        assert_eq!(state_file_name(r"C:\x\GTA5_Enhanced.exe", env(&[("SteamAppId", "3240220")])), "3240220-gta5_enhanced.txt");
        assert_eq!(state_file_name(r"C:\x\we ird$.exe", env(&[("SteamAppId", "1")])), "1-we_ird_.txt");
        assert_eq!(state_file_name("", env(&[("SteamAppId", "1")])), "1-game.txt");
    }

    /// Wine passes no `HOME` into a game's Windows environment, only `WINEHOMEDIR`; the record must still be written.
    #[test]
    fn state_directory_from_wines_home_without_home() {
        let wine = env(&[("WINEHOMEDIR", r"\??\Z:\home\alex"), ("USERPROFILE", r"C:\users\steamuser")]);
        assert_eq!(dir(wine).as_deref(), Some("/home/alex/.local/state/neural-forge/ngx-modules"));
        assert_eq!(dir(env(&[("WINEHOMEDIR", r"\??\unix\home\a b\")])).as_deref(), Some("/home/a b/.local/state/neural-forge/ngx-modules"));
        assert_eq!(dir(env(&[("WINEHOMEDIR", r"\??\z:\home\alex"), ("XDG_STATE_HOME", "/s/")])).as_deref(), Some("/s/neural-forge/ngx-modules"));
        assert_eq!(dir(env(&[("WINEHOMEDIR", r"\??\Z:\home\alex"), ("HOME", "/h")])).as_deref(), Some("/h/.local/state/neural-forge/ngx-modules"));
        assert_eq!(unix_home_from_wine(r"\??\C:\users\steamuser"), None);
        assert_eq!(unix_home_from_wine(r"\??\Z:"), None);
        assert_eq!(unix_home_from_wine(r"\??\unix"), None);
    }

    /// A multi-byte character where the two- or four-byte prefix check ends must not panic (`rest[..2]` did).
    #[test]
    fn non_ascii_wine_home_does_not_panic() {
        for nt in [r"\??\é\home", r"\??\ü", "é", r"\??\日本語\home", r"\??\uniñ\x", r"\??\Zé:\x", ""] {
            assert_eq!(unix_home_from_wine(nt), None, "{nt}");
            assert_eq!(win_dir(env(&[("WINEHOMEDIR", nt)])), None, "{nt}");
        }
        assert_eq!(unix_home_from_wine(r"\??\Z:\home\élodie").as_deref(), Some("/home/élodie"));
        assert_eq!(unix_home_from_wine(r"\??\unix\home\日本").as_deref(), Some("/home/日本"));
    }

    /// The directory the wrapper writes through: `Z:` as ever, the NT path for `unix` and for another drive.
    #[test]
    fn windows_form_of_the_state_directory() {
        let w = |pairs: &[(&str, &str)]| win_dir(env(pairs));
        assert_eq!(w(&[("XDG_STATE_HOME", "/s/")]).as_deref(), Some(r"Z:\s\neural-forge\ngx-modules"));
        assert_eq!(w(&[("HOME", "/h")]).as_deref(), Some(r"Z:\h\.local\state\neural-forge\ngx-modules"));
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\Z:\home\alex")]).as_deref(), Some(r"Z:\home\alex\.local\state\neural-forge\ngx-modules"));
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\z:\home\alex"), ("HOME", "/h")]).as_deref(), Some(r"Z:\h\.local\state\neural-forge\ngx-modules"));
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\unix\home\a b\")]).as_deref(), Some(r"\\?\unix\home\a b\.local\state\neural-forge\ngx-modules"));
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\D:\users\alex")]).as_deref(), Some(r"\\?\D:\users\alex\.local\state\neural-forge\ngx-modules"));
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\C:")]), None);
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\unix")]), None);
        assert_eq!(w(&[("WINEHOMEDIR", r"\??\Z:")]), None);
        assert_eq!(w(&[]), None);
    }

    /// Executables of the same name in two games keep two records: by Steam app id, else by path.
    #[test]
    fn same_named_executables_of_two_games_keep_apart() {
        let a = state_file_name(r"Z:\games\A\Game.exe", env(&[("SteamAppId", "100")]));
        let b = state_file_name(r"Z:\games\B\Game.exe", env(&[("SteamAppId", "200")]));
        assert_ne!(a, b);
        assert_eq!(state_file_name(r"Z:\g\Game.exe", env(&[("SteamAppId", "0"), ("SteamGameId", "12345678901")])), "12345678901-game.txt");
        let (p, q) = (state_file_name(r"Z:\games\A\Game.exe", env(&[])), state_file_name(r"Z:\games\B\Game.exe", env(&[("SteamAppId", "x")])));
        assert_ne!(p, q);
        assert!(p.starts_with('x') && p.ends_with("-game.txt") && p.len() == "x".len() + 16 + "-game.txt".len(), "{p}");
        assert_eq!(p, state_file_name(r"z:\GAMES\a\game.EXE", env(&[])), "the path's case does not matter");
    }

    /// A process with more modules than the first buffer holds lists every one (no fixed cap).
    #[test]
    fn module_list_grows_to_what_is_needed() {
        let all: Vec<usize> = (1..=3000).collect();
        let listed = grow_list(0usize, |buf| {
            let n = buf.len().min(all.len());
            buf[..n].copy_from_slice(&all[..n]);
            Some(all.len())
        });
        assert_eq!(listed, all);
        assert_eq!(grow_list(0usize, |_| None), Vec::<usize>::new());
        assert_eq!(
            grow_list(0usize, |buf| {
                buf[0] = 7;
                Some(1)
            }),
            vec![7]
        );
    }

    #[test]
    fn renders_the_documented_format() {
        assert_eq!(version_text(310 << 16 | 4, 7), "310.4.0.7");
        let modules = [
            Module { role: "wrapper", path: r"C:\windows\system32\_nvngx.dll".into(), version: Some("4.0.0".into()) },
            Module { role: "snippet", path: r"C:\Game\nvngx_dlssg.dll".into(), version: Some("310.1.0.0".into()) },
            Module { role: "snippet", path: r"C:\Game\nvngx_dlss.dll".into(), version: None },
        ];
        assert_eq!(
            render("4.0.0", r"C:\Game\g.exe", &modules),
            "neural-forge-ngx-modules 1\nwrapper=4.0.0\nexe=C:\\Game\\g.exe\n\
             wrapper\t_nvngx.dll\t4.0.0\tC:\\windows\\system32\\_nvngx.dll\n\
             snippet\tnvngx_dlssg.dll\t310.1.0.0\tC:\\Game\\nvngx_dlssg.dll\n\
             snippet\tnvngx_dlss.dll\t-\tC:\\Game\\nvngx_dlss.dll\n"
        );
        assert_eq!(log_line(&modules[2]), r"NGX module snippet: no version C:\Game\nvngx_dlss.dll");
    }
}
