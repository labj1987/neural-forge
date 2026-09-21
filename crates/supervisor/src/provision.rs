//! Preparing the managed prefix for the system-Wine runner.
//!
//! A Proton runner brings its own DXVK and DXVK-NVAPI, and the helper leaves NVAPI to it
//! (`NEURAL_FORGE_SKIP_NVAPI`). Plain Wine brings neither, and NGX needs NVAPI to find the GPU,
//! which DXVK-NVAPI can only do through DXVK's `dxgi.dll`. So for `runner_type = "wine"` the
//! managed prefix gets both DLLs, a `dxvk.conf` reporting the NVIDIA vendor/device, and the
//! helper is started with native overrides for them.
//!
//! Ported from upstream DLSS5VKLayer's `dlssnr-helper` (`setup_prefix`,
//! `install_wine_runtime_dll`, `download_runtime_dll`, `ensure_dxvk_config`), with the same
//! source order: DLLs the user supplied, then the copies an installed Proton ships (open-source
//! runner components), then the pinned release archives, verified by SHA256 before anything is
//! extracted from them. Upstream's vendored `vulkan-1.dll` is not carried: Wine's own
//! `winevulkan` serves Vulkan, and nothing here needs a replacement for it.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::paths;

pub const DXVK_VERSION: &str = "3.1";
pub const DXVK_SHA256: &str = "30f9cc326874be344285582275446968cfa4c069db31ce56df312d6644179154";
pub const DXVK_NVAPI_VERSION: &str = "0.9.2";
pub const DXVK_NVAPI_SHA256: &str = "60c284223530d643c446c263f1e1a96c6de7b5ff21796219646da734d97a70d6";

/// The two DLLs the system-Wine prefix needs, in install order.
pub const RUNTIME_DLLS: [&str; 2] = ["nvapi64.dll", "dxgi.dll"];

pub fn dxvk_conf_path() -> String {
    format!("{}/dxvk.conf", paths::state_dir())
}

/// Where a runtime DLL goes in the managed prefix.
pub fn prefix_dll(name: &str) -> PathBuf {
    PathBuf::from(paths::prefix_dir()).join("drive_c/windows/system32").join(name)
}

/// Whether a real (non-placeholder) runtime DLL is installed in the managed prefix. `wineboot`
/// fills `system32` with Wine's own placeholder DLLs, so the file merely existing proves nothing.
pub fn prefix_dll_installed(name: &str) -> bool {
    std::fs::read(prefix_dll(name)).is_ok_and(|b| !b.windows(20).any(|w| w == b"Wine placeholder DLL"))
}

/// The environment the helper needs under system Wine.
pub fn wine_env() -> Vec<(String, String)> {
    vec![
        ("DXVK_ENABLE_NVAPI".to_string(), "1".to_string()),
        ("DXVK_CONFIG_FILE".to_string(), dxvk_conf_path()),
        ("WINEDLLOVERRIDES".to_string(), "nvapi64=n,b;dxgi=n,b".to_string()),
    ]
}

/// The pinned archive a DLL can be downloaded from: (url, sha256, member path inside it).
fn release_for(name: &str) -> Option<(String, &'static str, String)> {
    match name {
        "dxgi.dll" => Some((
            format!("https://github.com/doitsujin/dxvk/releases/download/v{DXVK_VERSION}/dxvk-{DXVK_VERSION}.tar.gz"),
            DXVK_SHA256,
            format!("dxvk-{DXVK_VERSION}/x64/dxgi.dll"),
        )),
        "nvapi64.dll" => Some((
            format!("https://github.com/jp7677/dxvk-nvapi/releases/download/v{DXVK_NVAPI_VERSION}/dxvk-nvapi-v{DXVK_NVAPI_VERSION}.tar.gz"),
            DXVK_NVAPI_SHA256,
            "./x64/nvapi64.dll".to_string(),
        )),
        _ => None,
    }
}

/// The DLL as shipped by an installed Proton build (`files/lib/wine/{dxvk,nvapi}/x86_64-windows`).
fn from_proton(name: &str) -> Option<PathBuf> {
    for runner in crate::runners::discover_proton() {
        let root = runner.path.parent().map(Path::to_path_buf).unwrap_or_default();
        for sub in ["files/lib/wine/dxvk/x86_64-windows", "files/lib/wine/nvapi/x86_64-windows"] {
            let candidate = root.join(sub).join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Downloads the pinned archive, checks its SHA256, and extracts `name` into the user's binaries
/// directory. `NEURAL_FORGE_AUTO_DOWNLOAD=0` turns this off.
fn download(name: &str) -> Result<PathBuf, String> {
    if neural_forge_protocol::env::var("NEURAL_FORGE_AUTO_DOWNLOAD").as_deref() == Some("0") {
        return Err(format!("automatic download disabled; put {name} in {}", paths::binaries_dir()));
    }
    let (url, digest, member) = release_for(name).ok_or_else(|| format!("no pinned release for {name}"))?;
    let tmp = PathBuf::from(paths::state_dir()).join(format!("download-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let cleanup = |r| {
        let _ = std::fs::remove_dir_all(&tmp);
        r
    };
    let archive = tmp.join("archive.tar.gz");
    let fetched = Command::new("curl").args(["--fail", "--location", "--silent", "--show-error", "--retry", "2", "--output"]).arg(&archive).arg(&url).status();
    if !fetched.is_ok_and(|s| s.success()) {
        return cleanup(Err(format!("could not download {url} (curl is required)")));
    }
    let bytes = std::fs::read(&archive).map_err(|e| e.to_string())?;
    let actual = sha256_hex(&bytes);
    if actual != digest {
        return cleanup(Err(format!("checksum mismatch for {url}: got {actual}, expected {digest}")));
    }
    let out = Command::new("tar").arg("-xOf").arg(&archive).arg(&member).output().ok();
    let Some(out) = out.filter(|o| o.status.success() && !o.stdout.is_empty()) else {
        return cleanup(Err(format!("{member} not found in {url}")));
    };
    let target = PathBuf::from(paths::binaries_dir()).join(name);
    let result = std::fs::create_dir_all(paths::binaries_dir()).and_then(|()| std::fs::write(&target, &out.stdout)).map(|()| target).map_err(|e| e.to_string());
    cleanup(result)
}

/// Where a runtime DLL comes from, in order: the configured binaries dir, the default binaries
/// dir, an installed Proton, the pinned download. Returns the source and a word for the log.
pub fn runtime_dll_source(cfg: &Config, name: &str) -> Result<(PathBuf, &'static str), String> {
    for dir in [cfg.binaries.clone(), paths::binaries_dir()] {
        let candidate = PathBuf::from(&dir).join(name);
        if !dir.is_empty() && candidate.is_file() {
            return Ok((candidate, "supplied"));
        }
    }
    if let Some(p) = from_proton(name) {
        return Ok((p, "from Proton"));
    }
    download(name).map(|p| (p, "downloaded"))
}

/// Writes `dxvk.conf` with the NVIDIA vendor/device (from the config, else detected), so DXVK
/// reports the real GPU to DXVK-NVAPI and NGX.
pub fn write_dxvk_conf(cfg: &Config) -> std::io::Result<()> {
    let (vendor, device) = if !cfg.dxvk_vendor.is_empty() && !cfg.dxvk_device.is_empty() {
        (cfg.dxvk_vendor.clone(), cfg.dxvk_device.clone())
    } else if let Some((v, d)) = crate::gpu::detect_nvidia_gpu() {
        (format!("{v:04x}"), format!("{d:04x}"))
    } else {
        (String::new(), String::new())
    };
    let text = if vendor.is_empty() { String::new() } else { format!("dxgi.customVendorId = {vendor}\ndxgi.customDeviceId = {device}\n") };
    std::fs::create_dir_all(paths::state_dir())?;
    std::fs::write(dxvk_conf_path(), text)
}

/// Prepares the managed prefix for `runner_type = "wine"` (a no-op for other runners): creates
/// it with `wineboot` if it has never been initialised, installs the runtime DLLs, writes
/// `dxvk.conf`. Returns one line per step for the caller to show.
pub fn prepare_wine_prefix(cfg: &Config) -> Result<Vec<String>, String> {
    if cfg.runner_type != "wine" {
        return Ok(Vec::new());
    }
    let mut notes = Vec::new();
    let prefix = PathBuf::from(paths::prefix_dir());
    std::fs::create_dir_all(&prefix).map_err(|e| e.to_string())?;
    if !prefix.join("drive_c").is_dir() {
        let wine = if cfg.runner_path.is_empty() { "wine".to_string() } else { cfg.runner_path.clone() };
        let status = Command::new(&wine)
            .arg("wineboot")
            .arg("--init")
            .env("WINEPREFIX", &prefix)
            .env("WINEDEBUG", "-all")
            // No Mono/Gecko install prompts for a prefix that only ever runs the helper.
            .env("WINEDLLOVERRIDES", "mscoree,mshtml=")
            .status()
            .map_err(|e| format!("could not run {wine} wineboot: {e}"))?;
        if !status.success() {
            return Err(format!("{wine} wineboot --init failed ({status})"));
        }
        notes.push(format!("created the managed Wine prefix {}", prefix.display()));
    }
    for name in RUNTIME_DLLS {
        let target = prefix_dll(name);
        let (source, how) = runtime_dll_source(cfg, name)?;
        let bytes = std::fs::read(&source).map_err(|e| e.to_string())?;
        if std::fs::read(&target).ok().as_deref() != Some(bytes.as_slice()) {
            std::fs::create_dir_all(target.parent().expect("has a parent")).map_err(|e| e.to_string())?;
            std::fs::write(&target, &bytes).map_err(|e| e.to_string())?;
            notes.push(format!("installed {name} ({how}: {})", source.display()));
        }
    }
    write_dxvk_conf(cfg).map_err(|e| e.to_string())?;
    Ok(notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_releases_name_the_upstream_archives() {
        let (url, digest, member) = release_for("dxgi.dll").unwrap();
        assert!(url.ends_with("/v3.1/dxvk-3.1.tar.gz") && digest.len() == 64 && member == "dxvk-3.1/x64/dxgi.dll");
        let (url, _, member) = release_for("nvapi64.dll").unwrap();
        assert!(url.ends_with("/v0.9.2/dxvk-nvapi-v0.9.2.tar.gz") && member == "./x64/nvapi64.dll");
        assert!(release_for("vulkan-1.dll").is_none());
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn wine_env_overrides_exactly_the_provisioned_dlls() {
        let env = wine_env();
        let overrides = &env.iter().find(|(k, _)| k == "WINEDLLOVERRIDES").unwrap().1;
        for name in RUNTIME_DLLS {
            assert!(overrides.contains(&format!("{}=n,b", name.trim_end_matches(".dll"))));
        }
    }
}
