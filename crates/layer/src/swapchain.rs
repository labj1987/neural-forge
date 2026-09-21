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
