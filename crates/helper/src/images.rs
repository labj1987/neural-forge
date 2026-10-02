//! Usage flags of the helper's per-frame images (`frame.rs`), in a module that builds natively so
//! what each image is used for is checked by a unit test without Wine (a missing flag is a
//! validation error the helper's own tests could not see).

use ash::vk;

const fn or(a: vk::ImageUsageFlags, b: vk::ImageUsageFlags) -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::from_raw(a.as_raw() | b.as_raw())
}

/// Color, the model's input: the upload and the multi-pass chain copy write it (`TRANSFER_DST`),
/// the model and the HDR flow pass read it (`SAMPLED`), and on the 8-bit flow path
/// (`optical_flow::GpuFlow::estimate`) it is blitted down into the flow input (`TRANSFER_SRC`).
pub const COLOR_USAGE: vk::ImageUsageFlags = or(or(vk::ImageUsageFlags::SAMPLED, vk::ImageUsageFlags::TRANSFER_DST), vk::ImageUsageFlags::TRANSFER_SRC);

/// Output, the model's answer: written by the model (`STORAGE`), copied out to the answer region
/// and into Color for the next pass (`TRANSFER_SRC`).
pub const OUTPUT_USAGE: vk::ImageUsageFlags = or(vk::ImageUsageFlags::STORAGE, vk::ImageUsageFlags::TRANSFER_SRC);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_can_be_uploaded_sampled_and_blitted_from() {
        assert!(COLOR_USAGE.contains(vk::ImageUsageFlags::TRANSFER_DST), "upload and chain copy");
        assert!(COLOR_USAGE.contains(vk::ImageUsageFlags::SAMPLED), "the model and the HDR flow pass");
        assert!(COLOR_USAGE.contains(vk::ImageUsageFlags::TRANSFER_SRC), "the 8-bit flow path blits from it");
        assert!(OUTPUT_USAGE.contains(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC));
    }
}
