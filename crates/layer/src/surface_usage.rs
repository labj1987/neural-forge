//! Conservative admission for host capture. Never retry swapchain creation:
//! Vulkan retires oldSwapchain even when creation fails.
use ash::vk;

/// Structures that genuinely change what the swapchain's images *are*, and so have to
/// be understood before their usage is enlarged. Everything else in a `pNext` chain is
/// passed through untouched (this module only ever rewrites `image_usage`), so refusing
/// a whole swapchain for carrying any chain at all was too blunt.
///
/// Measured 2026-09-17: refusing every chain is what made this layer inert on GTA V.
/// DXVK creates that swapchain with an extension struct and non-empty flags, so
/// admission was declined, the swapchain stayed pass-through, and -- because the game
/// renders straight into the swapchain image as a colour attachment rather than
/// blitting into it -- there was no render source to tap either. The layer tracked
/// 793,394 barrier transitions and captured zero frames. Reading the frame at all
/// requires TRANSFER_SRC on the swapchain, which requires admitting it.
fn chain_is_understood(info: &vk::SwapchainCreateInfoKHR) -> bool {
    let mut next = info.p_next;
    while !next.is_null() {
        // SAFETY: every structure in a `pNext` chain begins with `{sType, pNext}` per
        // the Vulkan spec, and this chain is the application's own, valid for this call.
        let base = unsafe { &*next.cast::<vk::BaseInStructure>() };
        match base.s_type {
            // Multi-device layouts and exclusive full-screen control change how the
            // images are owned and presented. Not admitted.
            vk::StructureType::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR
            | vk::StructureType::SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT => return false,
            _ => {}
        }
        next = base.p_next.cast();
    }
    true
}

/// The view formats a `MUTABLE_FORMAT` swapchain declares, from its
/// `VkImageFormatListCreateInfo`. `None` when the chain has no such list.
fn view_formats(info: &vk::SwapchainCreateInfoKHR) -> Option<&[vk::Format]> {
    let mut next = info.p_next;
    while !next.is_null() {
        // SAFETY: as in `chain_is_understood`.
        let base = unsafe { &*next.cast::<vk::BaseInStructure>() };
        if base.s_type == vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO {
            // SAFETY: the sType identifies this struct; its array is the application's
            // own and valid for this call.
            let list = unsafe { &*next.cast::<vk::ImageFormatListCreateInfo>() };
            if list.view_format_count == 0 || list.p_view_formats.is_null() {
                return None;
            }
            return Some(unsafe { std::slice::from_raw_parts(list.p_view_formats, list.view_format_count as usize) });
        }
        next = base.p_next.cast();
    }
    None
}

pub fn candidate(info: &vk::SwapchainCreateInfoKHR) -> bool {
    // Flags that change the images themselves. `PROTECTED` images cannot be read back
    // at all, and `SPLIT_INSTANCE_BIND_REGIONS` is a multi-device layout. `MUTABLE_FORMAT`
    // is admitted only with the format list the spec requires for it: DXVK sets it on
    // every swapchain whose game renders through an sRGB view (GTA San Andreas DE), and
    // refusing it left that game untouched. The added TRANSFER usage copies raw bytes, so
    // the view formats do not change what is captured; `prepare` still checks every
    // listed format supports the enlarged usage. Unknown bits are allowed: this module
    // rewrites only `image_usage`, and the surface-capability and image-format-properties
    // checks below still have to pass for the enlarged usage.
    let unsupported_flags = vk::SwapchainCreateFlagsKHR::PROTECTED
        | vk::SwapchainCreateFlagsKHR::SPLIT_INSTANCE_BIND_REGIONS;
    let mutable_ok = !info.flags.contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT) || view_formats(info).is_some();
    !info.flags.intersects(unsupported_flags)
        && mutable_ok
        && chain_is_understood(info)
        && info.image_array_layers == 1
        && matches!(info.present_mode, vk::PresentModeKHR::FIFO | vk::PresentModeKHR::FIFO_RELAXED
            | vk::PresentModeKHR::IMMEDIATE | vk::PresentModeKHR::MAILBOX)
}

pub fn requested_usage(original: vk::ImageUsageFlags, supported: vk::ImageUsageFlags) -> Option<vk::ImageUsageFlags> {
    let required = original | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    supported.contains(required).then_some(required)
}

/// Returns a private copy; the caller's create info and extension chain are never modified.
/// Unsupported or unqueryable surfaces use the original creation path without capture.
pub fn prepare(instance: &ash::Instance, query: vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR, physical: vk::PhysicalDevice, info: &vk::SwapchainCreateInfoKHR)
    -> Option<vk::SwapchainCreateInfoKHR>
{
    if !candidate(info) { return None; }
    // SAFETY: query was resolved through the same next-layer instance dispatch
    // chain as physical. Never send that handle through the loader trampoline.
    let mut caps = vk::SurfaceCapabilitiesKHR::default();
    unsafe { query(physical, info.surface, &mut caps) }.result().ok()?;
    let usage = requested_usage(info.image_usage, caps.supported_usage_flags)?;
    // Surface usage alone does not guarantee this format's implied image creation
    // parameters support the enlarged usage combination (VUID-imageFormat-01778). A
    // mutable-format swapchain implies MUTABLE_FORMAT image creation, and every view
    // format it lists is checked too.
    let mutable = info.flags.contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT);
    let create_flags = if mutable { vk::ImageCreateFlags::MUTABLE_FORMAT } else { vk::ImageCreateFlags::empty() };
    let mut formats = vec![info.image_format];
    if mutable {
        formats.extend(view_formats(info)?.iter().copied().filter(|f| *f != info.image_format));
    }
    for format in formats {
        // SAFETY: plain query on the same physical device the swapchain is created for.
        let props = unsafe { instance.get_physical_device_image_format_properties(
            physical, format, vk::ImageType::TYPE_2D, vk::ImageTiling::OPTIMAL,
            usage, create_flags,
        ) }.ok()?;
        if info.image_extent.width > props.max_extent.width
            || info.image_extent.height > props.max_extent.height
            || props.max_array_layers < 1 || !props.sample_counts.contains(vk::SampleCountFlags::TYPE_1)
        { return None; }
    }
    let mut adjusted = *info;
    adjusted.image_usage = usage;
    Some(adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transfer_admission_preserves_application_usage() {
        let original = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let supported = original | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        assert_eq!(requested_usage(original, supported), Some(supported));
        assert_eq!(requested_usage(original, supported & !vk::ImageUsageFlags::TRANSFER_SRC), None);
        assert_eq!(requested_usage(original, supported & !vk::ImageUsageFlags::TRANSFER_DST), None);
        assert_eq!(requested_usage(original, supported & !vk::ImageUsageFlags::SAMPLED), None);
    }
    #[test]
    fn swapchains_that_redefine_their_own_images_are_not_captured() {
        let mut info = vk::SwapchainCreateInfoKHR::builder().image_array_layers(1).present_mode(vk::PresentModeKHR::FIFO).build();
        assert!(candidate(&info));
        // A present mode whose images are shared with the presentation engine.
        info.present_mode = vk::PresentModeKHR::SHARED_DEMAND_REFRESH;
        assert!(!candidate(&info));
        info.present_mode = vk::PresentModeKHR::FIFO;
        // Stereo.
        info.image_array_layers = 2;
        assert!(!candidate(&info));
        info.image_array_layers = 1;
        // Protected images cannot be read back at all.
        info.flags = vk::SwapchainCreateFlagsKHR::PROTECTED;
        assert!(!candidate(&info));
        // A mutable format without the format list the spec requires for it.
        info.flags = vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT;
        assert!(!candidate(&info));
        info.flags = vk::SwapchainCreateFlagsKHR::empty();
    }

    #[test]
    fn a_mutable_format_swapchain_with_its_format_list_is_admitted() {
        // DXVK creates GTA San Andreas DE's swapchain like this (UNORM storage, sRGB
        // views, plus the Reflex latency struct). Refusing it left the game untouched.
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let list = vk::ImageFormatListCreateInfo::builder().view_formats(&formats).build();
        let reflex = vk::BaseInStructure {
            s_type: vk::StructureType::from_raw(1_000_505_007),
            p_next: std::ptr::from_ref(&list).cast(),
        };
        let mut info = vk::SwapchainCreateInfoKHR::builder()
            .image_array_layers(1)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .image_format(vk::Format::B8G8R8A8_UNORM)
            .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
            .build();
        info.p_next = std::ptr::from_ref(&reflex).cast();
        assert!(candidate(&info));
        assert_eq!(view_formats(&info), Some(&formats[..]));

        let empty = vk::ImageFormatListCreateInfo::default();
        info.p_next = std::ptr::from_ref(&empty).cast();
        assert!(!candidate(&info), "an empty format list does not make a mutable format valid");
    }

    #[test]
    fn an_unknown_extension_in_the_chain_no_longer_makes_the_layer_inert() {
        // Refusing every `pNext` chain is what made this layer do nothing at all on
        // GTA V: DXVK attaches an extension struct newer than the headers this crate
        // builds against, so admission was declined, the swapchain stayed pass-through,
        // and the game -- which renders straight into the swapchain image rather than
        // blitting into it -- offered no render source to tap either. Since this module
        // rewrites `image_usage` and nothing else, an unrecognised struct in the
        // application's own chain is not a reason to give up on the frame.
        let mut info = vk::SwapchainCreateInfoKHR::builder().image_array_layers(1).present_mode(vk::PresentModeKHR::FIFO).build();
        let unknown = vk::BaseInStructure { s_type: vk::StructureType::from_raw(1_000_505_007), p_next: std::ptr::null() };
        info.p_next = std::ptr::from_ref(&unknown).cast();
        assert!(candidate(&info), "an unknown pNext struct must not disqualify capture");
    }

    #[test]
    fn structures_that_change_what_the_images_are_still_disqualify_the_chain() {
        let mut info = vk::SwapchainCreateInfoKHR::builder().image_array_layers(1).present_mode(vk::PresentModeKHR::FIFO).build();
        for s_type in [
            vk::StructureType::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
            vk::StructureType::SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
        ] {
            let refused = vk::BaseInStructure { s_type, p_next: std::ptr::null() };
            info.p_next = std::ptr::from_ref(&refused).cast();
            assert!(!candidate(&info), "{s_type:?} should not be admitted");
        }
    }

    #[test]
    fn the_whole_chain_is_walked_not_just_its_first_link() {
        // A refused struct hiding behind an allowed one still has to be found.
        let mut info = vk::SwapchainCreateInfoKHR::builder().image_array_layers(1).present_mode(vk::PresentModeKHR::FIFO).build();
        let deep = vk::BaseInStructure { s_type: vk::StructureType::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR, p_next: std::ptr::null() };
        let shallow = vk::BaseInStructure { s_type: vk::StructureType::from_raw(1_000_505_007), p_next: std::ptr::from_ref(&deep) };
        info.p_next = std::ptr::from_ref(&shallow).cast();
        assert!(!candidate(&info), "a refused struct deeper in the chain was missed");
    }
}
