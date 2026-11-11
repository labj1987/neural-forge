//! Per-swapchain state, and what a swapchain's format/colour space say about the light
//! it carries.

use ash::vk;
use neural_forge_protocol::enums::hdr_kind;

pub struct SwapchainState {
    pub format: vk::Format,
    /// The colour space it was created with (`imageColorSpace`): with the format, what the frame's light is.
    pub color_space: vk::ColorSpaceKHR,
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
    /// What its HDR verdict is made from (`crate::hdr_verdict`); the first HDR metadata is added when it comes.
    pub hdr: crate::hdr_verdict::Facts,
    /// [`Self::hdr`]'s verdict, made when it changes; empty for a swapchain that is not the game's own (too small,
    /// another process's, not on the NVIDIA GPU), which publishes none.
    pub hdr_text: String,
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
    /// How long the game must render steadily before the layer first engages
    /// (`NEURAL_FORGE_WARMUP_SECS` overrides; 0 engages at once, which is how the loading-screen
    /// freeze was reproduced).
    fn hold() -> std::time::Duration {
        static HOLD: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
        *HOLD.get_or_init(|| {
            let secs = neural_forge_protocol::env::var("NEURAL_FORGE_WARMUP_SECS").and_then(|v| v.parse::<u64>().ok()).unwrap_or(5);
            std::time::Duration::from_secs(secs)
        })
    }
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
        note_game_frame(now, game_frame);
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
        if Self::hold().is_zero() {
            self.engaged = true;
            return true;
        }
        if game_frame > Self::STEADY_FRAME {
            self.steady_since = None;
        } else if self.steady_since.is_none() {
            self.steady_since = Some(now);
        }
        self.engaged = self.steady_since.is_some_and(|t| now.saturating_duration_since(t) >= Self::hold());
        self.engaged
    }
}

/// The process's presents, for [`loading_now`]: ms since [`present_clock`]'s start plus one (0: none yet), of the
/// last present and of the start of the current run of frames shorter than [`Warmup::STEADY_FRAME`] (0: none).
static LAST_PRESENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STEADY_SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn present_clock(at: std::time::Instant) -> u64 {
    static START: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(std::time::Instant::now);
    at.saturating_duration_since(*START).as_millis() as u64 + 1
}

/// How long frames must have been short before the game counts as rendering steadily ([`loading_now`]), and how long
/// without a present counts as loading.
const STEADY_FOR_MS: u64 = 2000;
const NO_PRESENT_MS: u64 = 1000;

/// One game frame of any swapchain of the process ([`Warmup::on_present`]).
fn note_game_frame(now: std::time::Instant, game_frame: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    let t = present_clock(now);
    LAST_PRESENT.store(t, Relaxed);
    if game_frame > Warmup::STEADY_FRAME {
        STEADY_SINCE.store(0, Relaxed);
    } else {
        let _ = STEADY_SINCE.compare_exchange(0, t, Relaxed, Relaxed);
    }
}

/// Whether the game is loading rather than rendering steadily: it has presented, and either it has not for
/// [`NO_PRESENT_MS`] or its frames have not been shorter than [`Warmup::STEADY_FRAME`] (about 15 fps) for
/// [`STEADY_FOR_MS`]. The network's builds wait while it is (`preupscale::native::Loader`): each runs the network once
/// on the layer's own queue, and Hogwarts Legacy with ray tracing, loading at 12 fps, lost its GPU context to a
/// build's warm-up (Xid 109, CTX SWITCH TIMEOUT, twice in two runs, never without Neural Forge; 2026-10-23).
pub fn loading_now() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    loading_at(present_clock(std::time::Instant::now()), LAST_PRESENT.load(Relaxed), STEADY_SINCE.load(Relaxed))
}

fn loading_at(now: u64, last_present: u64, steady_since: u64) -> bool {
    if last_present == 0 {
        return false;
    }
    now.saturating_sub(last_present) > NO_PRESENT_MS || steady_since == 0 || now.saturating_sub(steady_since) < STEADY_FOR_MS
}

#[cfg(test)]
mod loading_tests {
    use super::loading_at;

    /// Nothing presented: not loading (a process that never presents builds at once). Frames short for 2 s: steady;
    /// for less, or a long frame since (0), or no present for a second: loading.
    #[test]
    fn the_game_loads_until_its_frames_have_been_short_for_two_seconds() {
        assert!(!loading_at(10_000, 0, 0), "never presented");
        assert!(!loading_at(10_000, 9_990, 7_000), "short frames for 3 s");
        assert!(loading_at(10_000, 9_990, 8_500), "for 1.5 s");
        assert!(loading_at(10_000, 9_990, 0), "a long frame just now");
        assert!(loading_at(10_000, 8_500, 2_000), "no present for 1.5 s");
    }
}

/// Exclude known-small compositor/overlay swapchains (the Steam overlay was
/// observed at 1262x598). This is only a size heuristic; `ownership` separately
/// enforces process eligibility and an exclusive cross-process channel lease.
pub fn is_plausible_game_size(width: u32, height: u32) -> bool {
    u64::from(width) * u64::from(height) >= 1280 * 720
}

/// Whether this is a swapchain the pass can capture, compose onto and present: the 8-bit SDR formats, and
/// the two 10-bit formats in the HDR10 colour space (`HDR10_ST2084_EXT`), whose frames the model server
/// converts to SDR for the network and whose composition carries the network's change back in linear light
/// (docs/NATIVE_BACKEND.md, "HDR on the finished frame (3.10.0)"; `NEURAL_FORGE_HDR_PRESENT=off` leaves them
/// out). A 10-bit swapchain in an SDR colour space, and the float16 (scRGB) formats, present untouched: the
/// first is not handled yet, and a float16 frame needs 8 bytes a pixel where the shared regions and the
/// composition hold 4.
pub fn is_supported_format(format: vk::Format, color_space: vk::ColorSpaceKHR) -> bool {
    is_sdr_8bit(format) || (detect_hdr_kind(format, color_space) == hdr_kind::PQ10 && crate::hdr_present::enabled())
}

fn is_sdr_8bit(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB)
}

/// The `neural_forge_protocol::enums::proxy_format` this swapchain's raw
/// `vkCmdCopyImageToBuffer` dump actually is -- only meaningful for swapchains
/// [`is_supported_format`] already accepted. `proxy_format::bytes_per_pixel` is the
/// single source of truth for the byte count that goes with this; nothing here
/// duplicates it.
pub fn proxy_format_for(format: vk::Format, color_space: vk::ColorSpaceKHR) -> u32 {
    use neural_forge_protocol::enums::proxy_format;
    match format {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => proxy_format::BGRA8,
        vk::Format::R16G16B16A16_SFLOAT => proxy_format::RGBA16F,
        vk::Format::A2B10G10R10_UNORM_PACK32 if detect_hdr_kind(format, color_space) == hdr_kind::PQ10 => proxy_format::PQ10_RGB,
        vk::Format::A2R10G10B10_UNORM_PACK32 if detect_hdr_kind(format, color_space) == hdr_kind::PQ10 => proxy_format::PQ10_BGR,
        _ => proxy_format::RGBA8,
    }
}

/// Whether raw capture bytes put blue first: B,G,R,A (the 8-bit BGRA formats), or blue in the low bits of
/// the packed word (`A2R10G10B10_UNORM_PACK32`). The protocol preserves this format; composition still
/// swizzles only its own pixel math, never SHM bytes.
/// Whether the swapchain's 8-bit format stores sRGB-encoded values that a blit decodes (`*_SRGB`).
pub fn is_srgb(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB)
}

pub fn is_bgr_order(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::A2R10G10B10_UNORM_PACK32)
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
    let ten_bit = matches!(format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32);
    let pq = color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT;
    if ten_bit && pq {
        hdr_kind::PQ10
    } else {
        hdr_kind::NONE
    }
}

/// The line logged for the first HDR metadata (`vkSetHdrMetadataEXT`) an application sends for a swapchain it
/// created with `format` in `color_space`. The metadata says nothing of how the frames are encoded; the colour
/// space does. A 10-bit swapchain declared sRGB that gets HDR10 metadata is either SDR (the metadata sent anyway:
/// DXVK and vkd3d-proton forward a game's `SetHDRMetaData` whatever its colour space) or PQ code in a swapchain
/// that does not say so, which the compositor then shows as sRGB (#51: Stellar Blade and Indiana Jones under
/// gamescope with HDR on in the game; gamescope logged the metadata, the layer only sRGB swapchains).
pub fn hdr_metadata_line(format: vk::Format, color_space: vk::ColorSpaceKHR, m: &vk::HdrMetadataEXT) -> String {
    let primaries = primaries_name(m);
    let ten_bit = matches!(format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32);
    let what = match detect_hdr_kind(format, color_space) {
        hdr_kind::PQ10 => "the swapchain is HDR10 (PQ), as the metadata says",
        hdr_kind::LINEAR_FP16 => "the swapchain is linear FP16",
        _ if ten_bit && color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR => {
            "the application created this swapchain as sRGB: its frames are SDR, or PQ-encoded HDR10 the swapchain does not declare \
             (then shown as sRGB); either way it is not an HDR10 swapchain to the layer, which presents it untouched"
        }
        _ => "the swapchain's colour space is not HDR10",
    };
    format!(
        "HDR metadata for a {format:?} {color_space:?} swapchain: {primaries} primaries, mastering luminance {:.4}-{:.0} nits, MaxCLL {:.0}, MaxFALL {:.0}; {what}",
        m.min_luminance, m.max_luminance, m.max_content_light_level, m.max_frame_average_light_level
    )
}

/// The metadata's display primaries by name ("BT.2020", "Display P3", "BT.709"), else in numbers.
pub fn primaries_name(m: &vk::HdrMetadataEXT) -> String {
    let near = |p: vk::XYColorEXT, x: f32, y: f32| (p.x - x).abs() < 0.005 && (p.y - y).abs() < 0.005;
    if near(m.display_primary_red, 0.708, 0.292) && near(m.display_primary_green, 0.170, 0.797) && near(m.display_primary_blue, 0.131, 0.046) {
        "BT.2020".to_string()
    } else if near(m.display_primary_red, 0.680, 0.320) && near(m.display_primary_green, 0.265, 0.690) && near(m.display_primary_blue, 0.150, 0.060) {
        "Display P3".to_string()
    } else if near(m.display_primary_red, 0.640, 0.330) && near(m.display_primary_green, 0.300, 0.600) && near(m.display_primary_blue, 0.150, 0.060) {
        "BT.709".to_string()
    } else {
        format!(
            "r ({:.3}, {:.3}) g ({:.3}, {:.3}) b ({:.3}, {:.3})",
            m.display_primary_red.x,
            m.display_primary_red.y,
            m.display_primary_green.x,
            m.display_primary_green.y,
            m.display_primary_blue.x,
            m.display_primary_blue.y
        )
    }
}

#[cfg(test)]
mod hdr_metadata_tests {
    use super::*;

    fn bt2020() -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT {
            display_primary_red: vk::XYColorEXT { x: 0.708, y: 0.292 },
            display_primary_green: vk::XYColorEXT { x: 0.17, y: 0.797 },
            display_primary_blue: vk::XYColorEXT { x: 0.131, y: 0.046 },
            white_point: vk::XYColorEXT { x: 0.3127, y: 0.329 },
            max_luminance: 1000.0,
            min_luminance: 0.001,
            max_content_light_level: 1000.0,
            max_frame_average_light_level: 400.0,
            ..Default::default()
        }
    }

    /// #51: gamescope logged BT.2020 metadata from Stellar Blade while the layer saw only sRGB swapchains.
    #[test]
    fn hdr10_metadata_on_an_srgb_swapchain_says_the_swapchain_is_not_hdr10() {
        let line = hdr_metadata_line(vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::SRGB_NONLINEAR, &bt2020());
        assert!(line.starts_with("HDR metadata for a A2B10G10R10_UNORM_PACK32 SRGB_NONLINEAR swapchain: BT.2020 primaries"), "{line}");
        assert!(line.contains("created this swapchain as sRGB") && line.contains("untouched"), "{line}");
        let pq = hdr_metadata_line(vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_ST2084_EXT, &bt2020());
        assert!(pq.contains("is HDR10 (PQ)") && pq.contains("1000 nits"), "{pq}");
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
    fn sdr_8bit_and_hdr10_swapchains_are_composable() {
        let srgb = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let hdr10 = vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        for f in [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_SRGB] {
            assert!(is_supported_format(f, srgb));
        }
        for f in [vk::Format::A2B10G10R10_UNORM_PACK32, vk::Format::A2R10G10B10_UNORM_PACK32] {
            assert!(is_supported_format(f, hdr10), "{f:?} in HDR10 has the HDR path");
            assert!(!is_supported_format(f, srgb), "{f:?} in an SDR colour space presents untouched");
            assert!(!is_supported_format(f, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT));
        }
        for cs in [srgb, hdr10, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT] {
            assert!(!is_supported_format(vk::Format::R16G16B16A16_SFLOAT, cs), "float16 (scRGB) presents untouched");
        }
    }
    #[test]
    fn raw_formats_preserve_channel_order_and_size() {
        let srgb = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let hdr10 = vk::ColorSpaceKHR::HDR10_ST2084_EXT;
        for f in [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB] {
            assert_eq!(proxy_format_for(f, srgb), proxy_format::BGRA8);
            assert_eq!(proxy_format::bytes_per_pixel(proxy_format_for(f, srgb)), 4);
            assert!(proxy_format::is_8bit(proxy_format_for(f, srgb)));
            assert!(is_bgr_order(f));
        }
        assert_eq!(proxy_format_for(vk::Format::R8G8B8A8_UNORM, srgb), proxy_format::RGBA8);
        assert!(!proxy_format::is_8bit(proxy_format::RGBA16F));
        let rgb = proxy_format_for(vk::Format::A2B10G10R10_UNORM_PACK32, hdr10);
        let bgr = proxy_format_for(vk::Format::A2R10G10B10_UNORM_PACK32, hdr10);
        assert_eq!((rgb, bgr), (proxy_format::PQ10_RGB, proxy_format::PQ10_BGR));
        for f in [rgb, bgr] {
            assert!(proxy_format::is_pq10(f) && !proxy_format::is_8bit(f));
            assert_eq!(proxy_format::bytes_per_pixel(f), 4, "the shared regions' size does not change");
        }
        assert!(!is_bgr_order(vk::Format::A2B10G10R10_UNORM_PACK32));
        assert!(is_bgr_order(vk::Format::A2R10G10B10_UNORM_PACK32));
        assert!(!proxy_format::is_pq10(proxy_format::RGBA8) && !proxy_format::is_pq10(proxy_format::BGRA8));
    }
}
