//! Where the wrapper loads NVIDIA's core from (docs/PROTON_NGX.md, "Loading NVIDIA's core"), and the copy it loads.
//!
//! The core's path must contain `nvngx.dll` in lower case ([`NGX_RUNTIME_MARK`]): NVIDIA's feature DLLs (`nvngx_dlss.dll`, `nvngx_dlssd.dll`, `nvngx_dlssg.dll`; 310.4 to 310.9 and the driver's
//! copies, read in Ghidra) check who calls each of their NGX exports: `GetModuleHandleExA(FROM_ADDRESS)` on the return
//! address, `GetModuleFileNameW`, and a case-sensitive `wcsstr(path, L"nvngx.dll")`; a caller whose path lacks it gets
//! `0xBAD00002` and the log line "Error: Not called from NGX runtime". So the core's path must contain `nvngx.dll` in
//! lower case (in any folder): as `_nvngx_core.dll`, `nvngx_core.dll` or `_NVNGX.DLL` DLSS never starts.
//! (Cyberpunk 2077, 2026-11-19: as `_nvngx_core.dll` no `CreateFeature`, the same as #51's "DLSS never starts" with
//! `nvngx_core.dll`). The wrapper warns when the core it loaded breaks that rule ([`name_warning`]: a core named by
//! `NEURAL_FORGE_NGX_CORE`). So the wrapper loads a copy of the
//! driver's core, `system32\neural-forge\_nvngx.dll`: a folder with nothing else in it, as stock Proton's core has no
//! DLSS feature DLL beside it in `system32`, where the driver's `nvidia/wine` folder holds NVIDIA's `nvngx_dlssg.dll`
//! and `nvngx_dlssd.dll` (Indiana Jones and the Great Circle crashed only with the core loaded from there, #51). Wine
//! would hand a module of that name to the wrapper (`_nvngx=b`), so the Neural Forge Proton copy also sets that full
//! path native (`proton_tool.rs`, `added_lines`); where it does not yet (a copy whose script is not made again yet), loading the copy
//! returns the wrapper itself, and the wrapper loads the driver's file instead.
//!
//! The part here needs no Windows (tested on Linux); `win.rs` asks Windows for the system folder and loads the core.

use std::io;
use std::path::{Path, PathBuf};

/// The folder in the prefix's `system32` that holds the copy.
pub const COPY_DIR: &str = "neural-forge";
/// The copy's file name: the core's own (see the module text).
pub const COPY_NAME: &str = "_nvngx.dll";
/// The environment variable that names the core to load instead (a Windows path, or a Linux one through `Z:`).
pub const ENV_CORE: &str = "NEURAL_FORGE_NGX_CORE";
/// What NVIDIA's feature DLLs look for, case-sensitively, in the path of the module that calls them: their caller must
/// be NGX's runtime (the module text).
pub const NGX_RUNTIME_MARK: &str = "nvngx.dll";

/// The warning for a core loaded from `path` when NVIDIA's feature DLLs will refuse its calls (its path lacks
/// [`NGX_RUNTIME_MARK`]): DLSS then never starts, with nothing else to say why.
pub fn name_warning(path: &str) -> Option<String> {
    (!path.contains(NGX_RUNTIME_MARK)).then(|| {
        format!(
            "NVIDIA's core {path} has no lower-case \"{NGX_RUNTIME_MARK}\" in its path: NVIDIA's DLSS DLLs refuse calls from it (0xBAD00002, \"Not called from NGX runtime\"), so DLSS never starts; name the file _nvngx.dll"
        )
    })
}

/// The driver folder Proton names in `NVIDIA_WINE_DLL_DIR` when it is not set (Debian and Ubuntu's layout).
pub const DEFAULT_DRIVER_DIR: &str = "/usr/lib/x86_64-linux-gnu/nvidia/wine";

/// Where NVIDIA's core comes from.
#[derive(Debug, PartialEq)]
pub enum Plan {
    /// `NEURAL_FORGE_NGX_CORE` names it: loaded from there as it is (for tests and diagnosis).
    Given(String),
    /// The driver's core (`from`, through `Z:`), copied to `to` (in the prefix's `system32`) and loaded from there.
    Copy { from: String, to: String },
}

/// The driver's core: `_nvngx.dll` in `NVIDIA_WINE_DLL_DIR` (a Linux path), through `Z:`.
pub fn driver_core(wine_dll_dir: Option<&str>) -> String {
    let dir = wine_dll_dir.map(str::trim).filter(|d| !d.is_empty()).unwrap_or(DEFAULT_DRIVER_DIR);
    format!("Z:{}\\_nvngx.dll", dir.trim_end_matches('/').replace('/', "\\"))
}

/// Where the core comes from: `NEURAL_FORGE_NGX_CORE` if set, else the driver's core copied into
/// `<system_dir>\neural-forge\`. `var` reads the environment.
pub fn plan(var: impl Fn(&str) -> Option<String>, system_dir: &str) -> Plan {
    if let Some(given) = var(ENV_CORE).as_deref().and_then(crate::trace::windows_path) {
        return Plan::Given(given);
    }
    let from = driver_core(var("NVIDIA_WINE_DLL_DIR").as_deref());
    let to = format!("{}\\{COPY_DIR}\\{COPY_NAME}", system_dir.trim_end_matches(['\\', '/']));
    Plan::Copy { from, to }
}

/// What [`place`] did.
#[derive(Debug, PartialEq)]
pub enum Placed {
    /// The copy was missing or differed from the driver's core, and was written.
    Written,
    /// The copy was already the driver's core, byte for byte.
    Unchanged,
}

/// Makes `to` a byte-for-byte copy of `from`: nothing is written when it already is one; otherwise the bytes go to a
/// file beside it named with `tag` (the process, so two processes never write the same file), which is then renamed
/// over `to`, so a reader never sees half a file. An error leaves `to` as it was (and no temporary file).
pub fn place(from: &Path, to: &Path, tag: &str) -> io::Result<Placed> {
    let bytes = std::fs::read(from)?;
    if std::fs::read(to).is_ok_and(|have| have == bytes) {
        return Ok(Placed::Unchanged);
    }
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = temporary(to, tag);
    let written = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, to));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    // Another process of the game may have put the same copy in place first (two starting at once), and Wine may
    // refuse the rename over its loaded file: that copy is as good.
    written.map(|()| Placed::Written).or_else(|e| if std::fs::read(to).is_ok_and(|have| have == bytes) { Ok(Placed::Unchanged) } else { Err(e) })
}

/// The temporary file [`place`] writes: `<to>.<tag>.tmp`.
fn temporary(to: &Path, tag: &str) -> PathBuf {
    let mut name = to.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{tag}.tmp"));
    to.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn the_driver_core_through_z() {
        assert_eq!(driver_core(Some("/run/host/usr/lib/x86_64-linux-gnu/nvidia/wine/")), r"Z:\run\host\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll");
        assert_eq!(driver_core(None), r"Z:\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll");
        assert_eq!(driver_core(Some("  ")), r"Z:\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll");
    }

    #[test]
    fn the_copy_goes_into_its_own_folder_in_system32() {
        let p = plan(env(&[("NVIDIA_WINE_DLL_DIR", "/usr/lib64/nvidia/wine")]), r"C:\windows\system32");
        assert_eq!(p, Plan::Copy { from: r"Z:\usr\lib64\nvidia\wine\_nvngx.dll".into(), to: r"C:\windows\system32\neural-forge\_nvngx.dll".into() });
        let Plan::Copy { to, .. } = plan(env(&[]), "C:\\windows\\system32\\") else { panic!() };
        assert_eq!(to, r"C:\windows\system32\neural-forge\_nvngx.dll");
    }

    /// The copy keeps the core's name, in a folder below `system32` (where Wine looks overrides up by the full path).
    #[test]
    fn the_copy_keeps_the_cores_name_in_a_folder_of_its_own() {
        assert_eq!(COPY_NAME, "_nvngx.dll");
        assert!(!COPY_DIR.is_empty() && !COPY_DIR.contains(['\\', '/']));
        assert!(crate::modules::is_ngx_module(COPY_NAME));
        assert!(COPY_NAME.contains(NGX_RUNTIME_MARK), "NVIDIA's DLSS DLLs take calls only from a path with lower-case nvngx.dll");
    }

    /// NVIDIA's feature DLLs refuse a caller whose path lacks lower-case `nvngx.dll`: such a core is warned about.
    #[test]
    fn a_core_path_without_lower_case_nvngx_dll_is_warned_about() {
        for ok in [r"C:\windows\system32\neural-forge\_nvngx.dll", r"Z:\usr\lib\nvidia\wine\_nvngx.dll", r"C:\NFC\nvngx.dll", r"C:\x\nvngx.dll.bak"] {
            assert_eq!(name_warning(ok), None, "{ok}");
        }
        for bad in [r"C:\nfc\_nvngx_core.dll", r"C:\nfcore\nvngx_core.dll", r"C:\windows\system32\_NVNGX.DLL", r"C:\x\_Nvngx.dll"] {
            assert!(name_warning(bad).is_some_and(|w| w.contains(bad) && w.contains("0xBAD00002")), "{bad}");
        }
        let Plan::Copy { to, .. } = plan(env(&[]), r"C:\windows\system32") else { panic!() };
        assert_eq!(name_warning(&to), None, "the wrapper's own copy");
    }

    #[test]
    fn a_given_core_wins_and_takes_a_linux_path() {
        assert_eq!(plan(env(&[(ENV_CORE, r"C:\nfcore\nvngx_core.dll")]), r"C:\windows\system32"), Plan::Given(r"C:\nfcore\nvngx_core.dll".into()));
        assert_eq!(
            plan(env(&[(ENV_CORE, "/usr/lib/x86_64-linux-gnu/nvidia/wine/_nvngx.dll")]), r"C:\windows\system32"),
            Plan::Given(r"Z:\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll".into())
        );
        assert!(matches!(plan(env(&[(ENV_CORE, " ")]), r"C:\windows\system32"), Plan::Copy { .. }), "empty: the copy");
    }

    struct Dir(PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn scratch(name: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("nf-core-copy-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    #[test]
    fn placing_writes_once_then_leaves_it() {
        let d = scratch("once");
        let from = d.0.join("_nvngx.dll");
        std::fs::write(&from, b"core 620.73").unwrap();
        let to = d.0.join("system32").join(COPY_DIR).join(COPY_NAME);
        assert_eq!(place(&from, &to, "1").unwrap(), Placed::Written);
        assert_eq!(std::fs::read(&to).unwrap(), b"core 620.73");
        assert_eq!(place(&from, &to, "1").unwrap(), Placed::Unchanged);
        // A new driver: the copy follows it.
        std::fs::write(&from, b"core 625.10, longer").unwrap();
        assert_eq!(place(&from, &to, "2").unwrap(), Placed::Written);
        assert_eq!(std::fs::read(&to).unwrap(), b"core 625.10, longer");
        let left: Vec<_> = std::fs::read_dir(to.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from(COPY_NAME)], "no temporary file left");
    }

    #[test]
    fn a_missing_driver_core_is_an_error_and_writes_nothing() {
        let d = scratch("missing");
        let to = d.0.join(COPY_DIR).join(COPY_NAME);
        assert!(place(&d.0.join("_nvngx.dll"), &to, "1").is_err());
        assert!(!to.exists());
    }

    #[test]
    fn a_failed_write_keeps_the_old_copy() {
        let d = scratch("fail");
        let from = d.0.join("_nvngx.dll");
        std::fs::write(&from, b"new").unwrap();
        // `to` is a folder: the rename over it fails.
        let to = d.0.join("taken");
        std::fs::create_dir_all(to.join("x")).unwrap();
        assert!(place(&from, &to, "7").is_err());
        assert!(to.join("x").is_dir());
        assert!(!d.0.join("taken.7.tmp").exists(), "the temporary file is removed");
    }
}
