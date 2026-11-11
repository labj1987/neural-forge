//! The NGX modules a game has loaded, and their file versions, for `doctor` and the report (docs/PROTON_NGX.md).
//!
//! After a feature is created ([`note_loaded`]) the wrapper lists the process's loaded modules, keeps the NGX ones
//! (NVIDIA's core, the wrapper itself, `nvngx.dll` and the DLSS snippets `nvngx_dlss.dll`, `nvngx_dlssg.dll`,
//! `nvngx_dlssd.dll`, `nvngx_dlssnr.dll`, ...: whichever path they were loaded from, the game's folder, the NGX models
//! folder an over-the-air update fills, the driver's folder) and writes their paths and file versions to
//! `$XDG_STATE_HOME/neural-forge/ngx-modules/<exe>.txt` (and to the text log, when it is on). Neural Forge's `doctor`
//! and report read that file; the supervisor's `ngx_versions.rs` parses the format below. A set that did not change is
//! not written again, so a game that creates features all the time costs one module listing each.
//!
//! The format, one file per game executable, replaced by every run:
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

/// The state directory's `neural-forge/ngx-modules` (the supervisor's `paths::state_dir`, the same rule), `None` with
/// neither `XDG_STATE_HOME` nor `HOME`.
pub fn dir(var: impl Fn(&str) -> Option<String>) -> Option<String> {
    let set = |name: &str| var(name).filter(|v| !v.is_empty());
    let state = set("XDG_STATE_HOME").or_else(|| set("HOME").map(|h| format!("{}/.local/state", h.trim_end_matches('/'))))?;
    Some(format!("{state}/neural-forge/ngx-modules"))
}

/// The file's name for the game executable at `exe`: its lower-case stem, with anything but letters, digits, `-`, `_`
/// and `.` replaced, `.txt` after.
pub fn state_file_name(exe: &str) -> String {
    let stem = file_name(exe);
    let stem = stem.rsplit_once('.').map_or(stem, |(s, _)| s);
    let clean: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c.to_ascii_lowercase() } else { '_' }).collect();
    format!("{}.txt", if clean.is_empty() { "game" } else { &clean })
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

    /// The modules the process has loaded: handle and path.
    fn loaded() -> Vec<(Handle, String)> {
        let mut handles = vec![core::ptr::null_mut::<c_void>(); 1024];
        let mut needed = 0u32;
        let ok = unsafe { K32EnumProcessModules(GetCurrentProcess(), handles.as_mut_ptr(), (handles.len() * size_of::<Handle>()) as u32, &mut needed) };
        if ok == 0 {
            return Vec::new();
        }
        let count = (needed as usize / size_of::<Handle>()).min(handles.len());
        handles[..count].iter().map(|&h| (h, module_path(h))).collect()
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
        {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            if *last == paths {
                return;
            }
            *last = paths;
        }
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
        let Some(dir) = dir(var) else { return };
        let win = |p: &str| format!("Z:{p}").replace('/', "\\");
        let dir = win(&dir);
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = std::fs::write(format!("{dir}\\{}", state_file_name(&exe)), render(env!("CARGO_PKG_VERSION"), &exe, &modules));
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
        assert_eq!(state_file_name(r"Z:\mnt\Steam\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe"), "cyberpunk2077.txt");
        assert_eq!(state_file_name(r"C:\x\GTA5_Enhanced.exe"), "gta5_enhanced.txt");
        assert_eq!(state_file_name(r"C:\x\we ird$.exe"), "we_ird_.txt");
        assert_eq!(state_file_name(""), "game.txt");
    }

    #[test]
    fn renders_the_documented_format() {
        assert_eq!(version_text(310 << 16 | 4, 7), "310.4.0.7");
        let modules = [
            Module { role: "wrapper", path: r"C:\windows\system32\_nvngx.dll".into(), version: Some("4.0.0".into()) },
            Module { role: "snippet", path: r"C:\Game\nvngx_dlssg.dll".into(), version: Some("310.9.16.0".into()) },
            Module { role: "snippet", path: r"C:\Game\nvngx_dlss.dll".into(), version: None },
        ];
        assert_eq!(
            render("4.0.0", r"C:\Game\g.exe", &modules),
            "neural-forge-ngx-modules 1\nwrapper=4.0.0\nexe=C:\\Game\\g.exe\n\
             wrapper\t_nvngx.dll\t4.0.0\tC:\\windows\\system32\\_nvngx.dll\n\
             snippet\tnvngx_dlssg.dll\t310.9.16.0\tC:\\Game\\nvngx_dlssg.dll\n\
             snippet\tnvngx_dlss.dll\t-\tC:\\Game\\nvngx_dlss.dll\n"
        );
        assert_eq!(log_line(&modules[2]), r"NGX module snippet: no version C:\Game\nvngx_dlss.dll");
    }
}
