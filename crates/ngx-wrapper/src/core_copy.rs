//! Where the wrapper loads NVIDIA's core from (docs/PROTON_NGX.md, "Loading NVIDIA's core"), and the copy it loads.
//!
//! NVIDIA's core treats the folder it was loaded from as a place of its own: when the game's NGX library asks for it
//! (a flag in the `InternalData` it passes), `NGXInitContext` adds that folder to where it loads DLSS's feature DLLs
//! from ("generic snippets"; driver 620.73's `_nvngx.dll`). Under stock Proton the core runs from `system32`, which
//! holds no feature DLLs; the driver's `nvidia/wine` folder holds NVIDIA's `nvngx_dlssg.dll` and `nvngx_dlssd.dll` next
//! to it. Indiana Jones and the Great Circle (#51) crashed only with the core loaded from the driver's folder: there a
//! feature reported itself supported that a byte-identical copy elsewhere reported unsupported. So the wrapper loads a
//! copy of the core from a folder of its own in the prefix's `system32`, which holds nothing else, as stock Proton's
//! core has no feature DLL beside it. The copy has another name than `_nvngx.dll`, which Wine would hand to the
//! wrapper again (`_nvngx=b`); the core reads only its folder, not its name.
//!
//! The part here needs no Windows (tested on Linux); `win.rs` asks Windows for the system folder and loads the core.

use std::io;
use std::path::{Path, PathBuf};

/// The folder in the prefix's `system32` that holds the copy.
pub const COPY_DIR: &str = "neural-forge";
/// The copy's file name: not `_nvngx.dll`, which `_nvngx=b` would load as the wrapper.
pub const COPY_NAME: &str = "_nvngx_core.dll";
/// The environment variable that names the core to load instead (a Windows path, or a Linux one through `Z:`).
pub const ENV_CORE: &str = "NEURAL_FORGE_NGX_CORE";
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
        assert_eq!(p, Plan::Copy { from: r"Z:\usr\lib64\nvidia\wine\_nvngx.dll".into(), to: r"C:\windows\system32\neural-forge\_nvngx_core.dll".into() });
        let Plan::Copy { to, .. } = plan(env(&[]), "C:\\windows\\system32\\") else { panic!() };
        assert_eq!(to, r"C:\windows\system32\neural-forge\_nvngx_core.dll");
    }

    /// The copy's name is not one Wine's `_nvngx=b` applies to, nor one the core takes for a feature (`nvngx_<name>`).
    #[test]
    fn the_copy_is_not_named_like_the_wrapper_or_a_feature() {
        assert!(!COPY_NAME.eq_ignore_ascii_case("_nvngx.dll"));
        assert!(!COPY_NAME.to_ascii_lowercase().starts_with("nvngx_"));
        assert!(crate::modules::is_ngx_module(COPY_NAME), "listed with NGX's modules");
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
