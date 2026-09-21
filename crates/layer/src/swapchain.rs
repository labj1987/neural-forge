//! Per-swapchain state, and what a swapchain's format/colour space say about the light
//! it carries.

use ash::vk;
use neural_forge_protocol::enums::hdr_kind;

pub struct SwapchainState {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub hdr_kind: u32,
    /// True when this swapchain's format isn't one the pass can work with, or it's
    /// larger than the protocol's ceiling (`neural_forge_protocol::{MAX_W,MAX_H}`) -- it
    /// presents untouched either way.
    pub pass_through: bool,
    /// The `imageUsage` the swapchain was actually created with (the layer's enlarged
    /// usage only when admission succeeded). The write-back path needs `TRANSFER_DST`.
    pub image_usage: vk::ImageUsageFlags,
    /// The swapchain's own images, in `vkGetSwapchainImagesKHR` order -- index `i`
    /// here is exactly what `VkPresentInfoKHR::pImageIndices[i]` refers to. Fetched
    /// once at creation (see `device::NeuralForgeDeviceInfo::fetch_swapchain_images`).
    pub images: Vec<vk::Image>,
}

/// Decides when the layer may engage on a swapchain: only once the game itself has been
/// rendering steadily, and never while it is on a loading screen.
///
/// Measured on GTA V (Proton, vkd3d): composing while the game was still loading froze it -- the
/// game's GPU work stopped waiting on a fence that never signalled, and the game shut itself down
/// about a minute later -- while the identical setup engaged once the game was in the world ran
/// cleanly every time. Loading screens present slowly and in bursts; gameplay presents steadily.
///
/// The game's own frame time is the interval between presents minus the time the layer itself
/// spent in the previous present, so the model's cost (the synchronous wait) can never make a
/// healthy game look like a loading one.
#[derive(Default)]
pub struct Warmup {
    last_present: Option<std::time::Instant>,
    steady_since: Option<std::time::Instant>,
    engaged: bool,
    /// Consecutive loading-length frames seen while engaged.
    long_frames: u32,
}

impl Warmup {
    /// Before engaging: a game frame slower than this is not steady rendering (below ~15 fps).
    const STEADY_FRAME: std::time::Duration = std::time::Duration::from_millis(66);
    /// How long the game must render steadily before the layer first engages.
    const HOLD: std::time::Duration = std::time::Duration::from_secs(5);
    /// Once engaged, only loading-screen-length frames count against the game. Ordinary stutters
    /// (streaming, shader compiles, a busy CPU) are shorter, and dropping out on them made the
    /// effect vanish for seconds at random during play.
    const LOADING_FRAME: std::time::Duration = std::time::Duration::from_millis(400);
    /// And it takes several of them in a row: one long hitch is a stutter, a run is a loading screen.
    const LOADING_RUN: u32 = 3;

    /// Call on every present with the time the layer spent in the previous one. Returns whether
    /// the layer may do anything with this present.
    pub fn on_present(&mut self, now: std::time::Instant, layer_time_last: std::time::Duration) -> bool {
        let game_frame = self.last_present.map(|last| now.saturating_duration_since(last).saturating_sub(layer_time_last));
        self.last_present = Some(now);
        let Some(game_frame) = game_frame else { return false };
        if self.engaged {
            if game_frame > Self::LOADING_FRAME {
                self.long_frames += 1;
                if self.long_frames >= Self::LOADING_RUN {
                    self.engaged = false;
                    self.steady_since = None;
                    self.long_frames = 0;
                }
            } else {
                self.long_frames = 0;
            }
            return self.engaged;
        }
        if game_frame > Self::STEADY_FRAME {
            self.steady_since = None;
        } else if self.steady_since.is_none() {
            self.steady_since = Some(now);
        }
        self.engaged = self.steady_since.is_some_and(|t| now.saturating_duration_since(t) >= Self::HOLD);
        self.engaged
    }
}

/// Exclude known-small compositor/overlay swapchains (the Steam overlay was
/// observed at 1262x598). This is only a size heuristic; `ownership` separately
/// enforces process eligibility and an exclusive cross-process channel lease.
pub fn is_plausible_game_size(width: u32, height: u32) -> bool {
    u64::from(width) * u64::from(height) >= 1280 * 720
}

/// Whether this is a format the pass can capture, compose onto and present. SDR 8-bit only:
/// the 10-bit and float formats are *recognised* (see [`detect_hdr_kind`]) but not yet handled.
/// RGBA16F is captured but never composed back (there is no half-float compose fallback), and
/// the PQ transfer is not ported, so letting them through would hand the model an unencoded HDR
/// frame and write an SDR result over it. They present untouched instead.
///
/// Widen this list only together with the matching compose path (the tracked follow-up:
/// the float16 proxy, PQ10 transfer and half-float compose fallback).
pub fn is_supported_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_UNORM
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::R8G8B8A8_UNORM
            | vk::Format::R8G8B8A8_SRGB
    )
}

/// The `neural_forge_protocol::enums::proxy_format` this swapchain's raw
/// `vkCmdCopyImageToBuffer` dump actually is -- only meaningful for formats
/// [`is_supported_format`] already accepted. `proxy_format::bytes_per_pixel` is the
/// single source of truth for the byte count that goes with this; nothing here
/// duplicates it.
pub fn proxy_format_for(format: vk::Format) -> u32 {
    match format {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => neural_forge_protocol::enums::proxy_format::BGRA8,
        vk::Format::R16G16B16A16_SFLOAT => neural_forge_protocol::enums::proxy_format::RGBA16F,
        _ => neural_forge_protocol::enums::proxy_format::RGBA8,
    }
}

/// Whether raw capture bytes are B,G,R,A. The protocol now preserves this
/// format; composition still swizzles only its own pixel math, never SHM bytes.
pub fn is_bgr_order(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB)
}

/// What a swapchain's format and colour space together say about the light in the
/// frame. A float swapchain hands over linear light directly. A ten-bit swapchain in an
/// HDR10/PQ colour space carries ST 2084 code -- absolute nits, not just more precision
/// on an already-tone-mapped picture, which is what the same ten-bit format in an SDR
/// colour space would be.
pub fn detect_hdr_kind(format: vk::Format, color_space: vk::ColorSpaceKHR) -> u32 {
    if format == vk::Format::R16G16B16A16_SFLOAT {
        return hdr_kind::LINEAR_FP16;
    }
    let ten_bit = matches!(
        format,
        vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32
    );
    let pq = color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT;
    if ten_bit && pq {
        hdr_kind::PQ10
    } else {
        hdr_kind::NONE
    }
}

#[cfg(test)]
mod warmup_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn run(w: &mut Warmup, t0: Instant, frames: &[(u64, u64)]) -> Vec<bool> {
        // (ms since t0, layer ms spent in the previous present)
        frames.iter().map(|&(at, layer)| w.on_present(t0 + Duration::from_millis(at), Duration::from_millis(layer))).collect()
    }

    #[test]
    fn engages_only_after_five_seconds_of_steady_rendering() {
        let mut w = Warmup::default();
        let t0 = Instant::now();
        let frames: Vec<(u64, u64)> = (0..400).map(|i| (i * 16, 0)).collect(); // 60 fps
        let out = run(&mut w, t0, &frames);
        assert!(!out[..300].iter().any(|&b| b), "must not engage before 5 s");
        assert!(out[330..].iter().all(|&b| b), "must engage after 5 s of steady frames");
    }

    #[test]
    fn a_loading_screen_never_engages_and_disengages_a_running_game() {
        let mut w = Warmup::default();
        let t0 = Instant::now();
        // Loading: a present every 700 ms for a minute.
        let loading: Vec<(u64, u64)> = (0..90).map(|i| (i * 700, 0)).collect();
        assert!(!run(&mut w, t0, &loading).iter().any(|&b| b));
        // Steady gameplay engages; then a loading screen (a run of long frames) disengages.
        let base = 90 * 700;
        let mut frames: Vec<(u64, u64)> = (0..400).map(|i| (base + i * 16, 0)).collect();
        let end = base + 400 * 16;
        frames.extend((1..=4).map(|k| (end + k * 700, 0)));
        let out = run(&mut w, t0, &frames);
        assert!(out[399]);
        assert!(!out[403], "a run of loading-length frames must disengage");
    }

    #[test]
    fn ordinary_stutters_during_play_do_not_disengage() {
        let mut w = Warmup::default();
        let t0 = Instant::now();
        let mut frames: Vec<(u64, u64)> = (0..400).map(|i| (i * 16, 0)).collect();
        // Stutters of 100 ms, 300 ms and a single 600 ms hitch, each followed by normal frames.
        let mut t = 400 * 16;
        for gap in [100u64, 300, 600, 150] {
            t += gap;
            frames.push((t, 0));
            for _ in 0..30 {
                t += 16;
                frames.push((t, 0));
            }
        }
        let out = run(&mut w, t0, &frames);
        assert!(out[400..].iter().all(|&b| b), "stutters must not switch the effect off");
    }

    #[test]
    fn the_layers_own_wait_does_not_count_against_the_game() {
        let mut w = Warmup::default();
        let t0 = Instant::now();
        // 4K synchronous: 60 ms between presents, 45 ms of it inside the layer.
        let frames: Vec<(u64, u64)> = (0..200).map(|i| (i * 60, 45)).collect();
        assert!(*run(&mut w, t0, &frames).last().unwrap(), "slow because of the model is still steady");
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use neural_forge_protocol::enums::proxy_format;
    #[test]
    fn only_sdr_8bit_swapchains_are_composable() {
        for f in [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_SRGB] {
            assert!(is_supported_format(f));
        }
        for f in [vk::Format::A2B10G10R10_UNORM_PACK32, vk::Format::A2R10G10B10_UNORM_PACK32, vk::Format::R16G16B16A16_SFLOAT] {
            assert!(!is_supported_format(f), "{f:?} must present untouched until a compose path exists");
        }
    }
    #[test]
    fn raw_formats_preserve_channel_order_and_size() {
        for f in [vk::Format::B8G8R8A8_UNORM,vk::Format::B8G8R8A8_SRGB] {
            assert_eq!(proxy_format_for(f),proxy_format::BGRA8);
            assert_eq!(proxy_format::bytes_per_pixel(proxy_format_for(f)),4);
            assert!(proxy_format::is_8bit(proxy_format_for(f)));
        }
        assert_eq!(proxy_format_for(vk::Format::R8G8B8A8_UNORM),proxy_format::RGBA8);
        assert!(!proxy_format::is_8bit(proxy_format::RGBA16F));
    }
}
