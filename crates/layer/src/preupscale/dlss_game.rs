//! Whether this process is a DLSS game, known before DLSS's first frame (#49, #59).
//!
//! In placement Auto the after-the-upscaler path took a DLSS game's loading screens and menus before DLSS started:
//! it built the network at the output size (3840x2160 at 4K) and ran it over them. In GTA V Enhanced that build came
//! before the game created NGX's Vulkan devices in all three launches that then hung (the game's watchdog at
//! `+0x12c6eb`); in Cyberpunk 2077 with Ray Reconstruction its buffers stayed beside the holds' and the card spilled.
//! A DLSS game gets no post path before DLSS's first frame ([`super::pre_dlss_post_off`]).
//!
//! The signs, the first one seen wins:
//! - DLSS's own libraries mapped into the process ([`dlss_library`]): `nvngx_dlss.dll` and `nvngx_dlssd.dll`
//!   (NGX's Super Resolution and Ray Reconstruction), Streamline's `sl.dlss.dll` and `sl.dlss_d.dll`, and a native
//!   game's `libnvidia-ngx-dlss*.so`. Wine maps a PE image's headers from its file, so the DLL's path is in
//!   `/proc/self/maps` from the moment it is loaded; Streamline games load theirs at start-up, before their device.
//!   Read by a thread of its own ([`watch`]), never on a game thread.
//! - DLSS's CUDA kernels created through `VK_NVX_binary_import` ([`dlss_kernel_name`]), on any device.
//!
//! A game that ships DLSS but runs another upscaler gets the post path by the existing hand-back rules counted from
//! the sign: 5 minutes without any Super Resolution launch, 30 s of launches with nothing held.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// When and why the process was first seen to be a DLSS game.
static SIGN: Mutex<Option<(Instant, String)>> = Mutex::new(None);

/// When the process was first seen to be a DLSS game, if it was.
pub(crate) fn since() -> Option<Instant> {
    SIGN.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().map(|(at, _)| *at)
}

/// The process is a DLSS game (`why`); the first sign is kept and logged.
pub(crate) fn note(why: &str) {
    let mut sign = SIGN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if sign.is_some() {
        return;
    }
    *sign = Some((Instant::now(), why.to_string()));
    drop(sign);
    crate::log!(
        "[preupscale] the game uses DLSS ({why}): the after-the-upscaler path stays off until DLSS's first frame (loading screens and menus go untouched)"
    );
    crate::logging::flush();
}

/// The libraries whose mapping makes a DLSS game, by their lowercase file name: exact names, then prefixes.
const DLSS_DLLS: [&str; 4] = ["nvngx_dlss.dll", "nvngx_dlssd.dll", "sl.dlss.dll", "sl.dlss_d.dll"];
const DLSS_SO_PREFIX: &str = "libnvidia-ngx-dlss";

/// The first DLSS library among the mappings of `maps` (the text of `/proc/self/maps`), by its file name. Frame
/// Generation's own (`nvngx_dlssg.dll`, `sl.dlss_g.dll`) is not one: it does not upscale.
pub(crate) fn dlss_library(maps: &str) -> Option<String> {
    maps.lines().find_map(|line| {
        // The path is the sixth field and may contain spaces; it starts at the first '/' after the inode.
        let path = line.split_whitespace().nth(5).and_then(|_| line.find('/').map(|i| &line[i..]))?;
        let name = path.rsplit('/').next()?.to_ascii_lowercase();
        let so = name.starts_with(DLSS_SO_PREFIX) && !name.starts_with("libnvidia-ngx-dlssg") && name.contains(".so");
        (DLSS_DLLS.contains(&name.as_str()) || so).then_some(name)
    })
}

/// Whether a CUDA kernel created through `VK_NVX_binary_import` is DLSS Super Resolution's or Ray Reconstruction's by
/// its name: the input kernels and Ray Reconstruction's families ([`super::Kernel`]), Super Resolution's other known
/// prefixes (`hiluma_` in 310.x, `rrlite_` in Crimson Desert's 310.9, Cyberpunk 2077's `cuda_luma_convert_kernel`).
/// Frame Generation's and NGX's helpers are not.
pub(crate) fn dlss_kernel_name(name: &str) -> bool {
    super::Kernel::of(name) != super::Kernel::Other || ["hiluma_", "rrlite_", "cuda_luma_convert_kernel"].iter().any(|p| name.starts_with(p))
}

/// How often the watcher reads the mappings: often while a game starts, rarely after.
fn interval(running: Duration) -> Duration {
    if running < Duration::from_secs(180) {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(5)
    }
}

/// Starts the thread that looks for DLSS's libraries in the process ([`dlss_library`]), once per process. It ends once
/// the process is known to be a DLSS game.
pub(crate) fn watch() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    let started = Instant::now();
    let spawned = std::thread::Builder::new().name("nf-dlss-watch".into()).spawn(move || {
        while since().is_none() {
            if let Some(name) = std::fs::read_to_string("/proc/self/maps").ok().as_deref().and_then(dlss_library) {
                note(&format!("{name} is loaded"));
                break;
            }
            std::thread::sleep(interval(started.elapsed()));
        }
    });
    if let Err(e) = spawned {
        crate::log!("[preupscale] could not start the thread that looks for DLSS's libraries: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_dlss_libraries_in_the_mappings_and_nothing_else() {
        let line = |path: &str| format!("7f0000000000-7f0000001000 r--p 00000000 103:02 1234567                    {path}\n");
        let base = [
            line("/usr/lib/x86_64-linux-gnu/libc.so.6"),
            line("/mnt/Storage/Steam/steamapps/common/Grand Theft Auto V Enhanced/sl.interposer.dll"),
            line("/mnt/Storage/Steam/steamapps/common/Grand Theft Auto V Enhanced/nvngx_dlssg.dll"),
            line("/mnt/Storage/Steam/steamapps/common/Grand Theft Auto V Enhanced/sl.dlss_g.dll"),
            line("/usr/lib/x86_64-linux-gnu/nvidia/wine/nvngx.dll"),
            "7f0000002000-7f0000003000 rw-p 00000000 00:00 0                          [heap]\n".to_string(),
            "7f0000003000-7f0000004000 rw-p 00000000 00:00 0 \n".to_string(),
        ]
        .concat();
        assert_eq!(dlss_library(&base), None, "Frame Generation, the interposer and NGX's core are no sign");
        for (path, name) in [
            ("/mnt/Storage/Steam/steamapps/common/Grand Theft Auto V Enhanced/sl.dlss.dll", "sl.dlss.dll"),
            ("/mnt/Storage/Steam/steamapps/common/Cyberpunk 2077/bin/x64/NVNGX_DLSSD.DLL", "nvngx_dlssd.dll"),
            ("/mnt/Storage/Steam/steamapps/common/Red Dead Redemption 2/nvngx_dlss.dll", "nvngx_dlss.dll"),
            ("/usr/lib/x86_64-linux-gnu/libnvidia-ngx-dlss.so.580.95.05", "libnvidia-ngx-dlss.so.580.95.05"),
        ] {
            assert_eq!(dlss_library(&(base.clone() + &line(path))).as_deref(), Some(name), "{path}");
        }
        assert_eq!(dlss_library(&(base + &line("/usr/lib/libnvidia-ngx-dlssg.so.580.95.05"))), None);
    }

    #[test]
    fn dlss_kernels_by_name() {
        for name in ["hiluma_engine_input_kernel", "cuda_engine_input_kernel_x", "rr2_enc0_kernel", "rrlite_input", "cuda_luma_convert_kernel"] {
            assert!(dlss_kernel_name(name), "{name}");
        }
        for name in ["main_kernel", "k_conv_fp16_nhwc", "custom_block_3", "cuda_dldn_engine_luma_convert_kernel"] {
            assert!(!dlss_kernel_name(name), "{name}");
        }
    }
}
