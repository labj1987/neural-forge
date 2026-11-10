//! The transfer usage the pre-upscaler path adds to an application's images (`preupscale::added_usage`), kept
//! only where it leaves the image's memory requirements as the application would have had them.
//!
//! An engine that places its render targets in its own heaps (id Tech: Indiana Jones and the Great Circle,
//! #51) sizes and aliases them by the requirements it learns, and may have learned them for an image created
//! as it asked, or reuse what it learned for one create info for another. A driver is free to give an image
//! with `TRANSFER_SRC | TRANSFER_DST` a different size, alignment or set of memory types (a different
//! compression layout, for one). So each create info is judged once: the requirements of the image as asked
//! and with the added usage are compared, and on any difference the image is created exactly as asked (the
//! hold inside DLSS's buffer then cannot copy it, which it says). The judgement comes from the application's
//! own `vkGetDeviceImageMemoryRequirements` query where it makes one (answered for whichever create info the
//! image will be created with), else from creating the image both ways and comparing (Vulkan 1.0 only, so
//! valid on any device).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use ash::vk;

/// Create infos judged at most (the verdicts kept); past that each one is judged again when it comes.
const VERDICTS_MAX: usize = 4096;
/// Lines logged at most about images whose requirements change.
const SAID_MAX: usize = 8;
/// The view formats a kept verdict's format list may name at most; a longer list is judged every time.
const VIEW_FORMATS_MAX: usize = 64;

/// What a verdict is kept by: every field of the create info that can change the requirements, for a create
/// info without a `pNext` chain or whose chain is one `VkImageFormatListCreateInfo` (vkd3d-proton's placed
/// resources, created while the game runs: each one judged would create two images); one with any other chain is
/// judged every time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    flags: vk::ImageCreateFlags,
    image_type: vk::ImageType,
    format: vk::Format,
    extent: (u32, u32, u32),
    mip_levels: u32,
    array_layers: u32,
    samples: vk::SampleCountFlags,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    sharing_mode: vk::SharingMode,
    /// The format list's view formats, in its order (`None`: no chain).
    view_formats: Option<Vec<vk::Format>>,
}

impl Key {
    /// `info`'s chain is valid: the application's own, as `vkCreateImage` (or the requirements query) has it.
    fn of(info: &vk::ImageCreateInfo) -> Option<Self> {
        let view_formats = match info.p_next.cast::<vk::BaseInStructure>() {
            next if next.is_null() => None,
            next => {
                // SAFETY: a valid chain (above): its first node starts with the common header.
                let node = unsafe { &*next };
                if node.s_type != vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO || !node.p_next.is_null() {
                    return None;
                }
                // SAFETY: a node of this type is that structure.
                let list = unsafe { &*next.cast::<vk::ImageFormatListCreateInfo>() };
                let count = list.view_format_count as usize;
                if count > VIEW_FORMATS_MAX {
                    return None;
                }
                Some(if count == 0 || list.p_view_formats.is_null() {
                    Vec::new()
                } else {
                    // SAFETY: the application's array of `view_format_count` formats.
                    unsafe { std::slice::from_raw_parts(list.p_view_formats, count) }.to_vec()
                })
            }
        };
        Some(Key {
            flags: info.flags,
            image_type: info.image_type,
            format: info.format,
            extent: (info.extent.width, info.extent.height, info.extent.depth),
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
            samples: info.samples,
            tiling: info.tiling,
            usage: info.usage,
            sharing_mode: info.sharing_mode,
            view_formats,
        })
    }
}

/// One device's verdicts.
#[derive(Default)]
pub(crate) struct AddedUsage {
    /// Whether the added usage keeps the requirements, by the create info as the application asked.
    verdicts: Mutex<HashMap<Key, bool>>,
    /// The (format, usage) pairs already logged.
    said: Mutex<HashSet<(vk::Format, vk::ImageUsageFlags)>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn describe(r: &vk::MemoryRequirements) -> String {
    format!("{} bytes, alignment {}, memory types {:#x}", r.size, r.alignment, r.memory_type_bits)
}

impl AddedUsage {
    /// The verdict already reached for `asked`, if any.
    pub(crate) fn cached(&self, asked: &vk::ImageCreateInfo) -> Option<bool> {
        Key::of(asked).and_then(|k| lock(&self.verdicts).get(&k).copied())
    }

    /// Judges `copyable` (`asked` with the added usage) by the two images' requirements, keeps the verdict and
    /// logs a difference (once per format and usage). `true`: they are the same, the image gets the usage.
    pub(crate) fn judge(
        &self,
        asked: &vk::ImageCreateInfo,
        copyable: &vk::ImageCreateInfo,
        as_asked: vk::MemoryRequirements,
        with: vk::MemoryRequirements,
    ) -> bool {
        let same = (as_asked.size, as_asked.alignment, as_asked.memory_type_bits) == (with.size, with.alignment, with.memory_type_bits);
        if let Some(key) = Key::of(asked) {
            let mut verdicts = lock(&self.verdicts);
            if verdicts.len() < VERDICTS_MAX {
                verdicts.insert(key, same);
            }
        }
        if !same {
            let mut said = lock(&self.said);
            if said.len() < SAID_MAX && said.insert((asked.format, asked.usage)) {
                drop(said);
                crate::log!(
                    "[preupscale] adding {:?} to a {:?} {}x{} image (usage {:?}) changes its memory requirements (as asked: {}; with it: {}): created as the game asked, so the hold inside DLSS's buffer cannot copy such an image",
                    copyable.usage & !asked.usage,
                    asked.format,
                    asked.extent.width,
                    asked.extent.height,
                    asked.usage,
                    describe(&as_asked),
                    describe(&with)
                );
                crate::logging::flush();
            }
        }
        same
    }

    /// Creates the application's image, with the added usage (`copyable`) only where that keeps its memory
    /// requirements: by the verdict reached for `asked` before, else by creating it both ways, comparing the two
    /// (`requirements`) and destroying the one not kept. A create info the driver refuses with the usage is
    /// created as asked. Returns the result and whether the image got the usage.
    pub(crate) fn create(
        &self,
        asked: &vk::ImageCreateInfo,
        copyable: &vk::ImageCreateInfo,
        create: &dyn Fn(&vk::ImageCreateInfo) -> ash::prelude::VkResult<vk::Image>,
        destroy: &dyn Fn(vk::Image),
        requirements: &dyn Fn(vk::Image) -> vk::MemoryRequirements,
    ) -> (ash::prelude::VkResult<vk::Image>, bool) {
        match self.cached(asked) {
            Some(false) => return (create(asked), false),
            Some(true) => {
                return match create(copyable) {
                    Ok(image) => (Ok(image), true),
                    Err(_) => (create(asked), false),
                };
            }
            None => {}
        }
        let plain = match create(asked) {
            Ok(image) => image,
            Err(e) => return (Err(e), false),
        };
        let Ok(added) = create(copyable) else { return (Ok(plain), false) };
        if self.judge(asked, copyable, requirements(plain), requirements(added)) {
            destroy(plain);
            (Ok(added), true)
        } else {
            destroy(added);
            (Ok(plain), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn colour(p_next: *const std::ffi::c_void) -> (vk::ImageCreateInfo<'static>, vk::ImageCreateInfo<'static>) {
        let asked = vk::ImageCreateInfo {
            p_next,
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R16G16B16A16_SFLOAT,
            extent: vk::Extent3D { width: 32, height: 16, depth: 1 },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            // Indiana Jones and the Great Circle's colour input (#51).
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        };
        let copyable = vk::ImageCreateInfo { usage: asked.usage | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST, ..asked };
        (asked, copyable)
    }

    /// [`colour`] for a chain whose format list names more than one format: such an image is `MUTABLE_FORMAT`
    /// (VUID-VkImageCreateInfo-flags-04738).
    fn mutable(p_next: *const std::ffi::c_void) -> (vk::ImageCreateInfo<'static>, vk::ImageCreateInfo<'static>) {
        let (asked, copyable) = colour(p_next);
        let flags = vk::ImageCreateFlags::MUTABLE_FORMAT;
        (vk::ImageCreateInfo { flags, ..asked }, vk::ImageCreateInfo { flags, ..copyable })
    }

    /// On a real (lavapipe) device, through real images: where the requirements with the added usage differ
    /// (made to differ here: the driver's are the same) the image is created as asked and the other destroyed,
    /// the verdict is kept (the next one is created once, as asked) and said; where they are the same the image
    /// gets the usage, and the next one is created once, with it. A chain of one format list is kept by its formats
    /// too (the same list: one image; another list: judged again); any other chain is judged each time.
    #[test]
    fn the_added_usage_is_kept_only_where_the_memory_requirements_do_not_change() {
        let Some((_entry, instance, _physical, device, _queue, _family)) = crate::composition::gpu::test_device() else {
            eprintln!("added usage test: no Vulkan device, skipping");
            return;
        };
        let live: RefCell<HashMap<vk::Image, vk::ImageUsageFlags>> = RefCell::default();
        let creates = RefCell::new(0);
        let create = |info: &vk::ImageCreateInfo| {
            *creates.borrow_mut() += 1;
            let image = unsafe { device.create_image(info, None) }?;
            live.borrow_mut().insert(image, info.usage);
            Ok(image)
        };
        let destroy = |image: vk::Image| {
            live.borrow_mut().remove(&image);
            unsafe { device.destroy_image(image, None) };
        };
        let transfer = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        for differ in [true, false] {
            let usage = AddedUsage::default();
            let requirements = |image: vk::Image| {
                let mut r = unsafe { device.get_image_memory_requirements(image) };
                if differ && live.borrow()[&image].contains(transfer) {
                    r.size += 65536;
                }
                r
            };
            let (asked, copyable) = colour(std::ptr::null());
            *creates.borrow_mut() = 0;
            let (image, added) = usage.create(&asked, &copyable, &create, &destroy, &requirements);
            let image = image.expect("created");
            assert_eq!(added, !differ);
            assert_eq!(live.borrow().len(), 1, "the other one destroyed");
            assert_eq!(live.borrow()[&image], if differ { asked.usage } else { copyable.usage });
            assert_eq!((*creates.borrow(), usage.cached(&asked)), (2, Some(!differ)), "judged by creating both, kept");
            *creates.borrow_mut() = 0;
            let (again, added_again) = usage.create(&asked, &copyable, &create, &destroy, &requirements);
            assert_eq!((added_again, *creates.borrow()), (added, 1), "the kept verdict: one image");
            assert_eq!(live.borrow()[&again.unwrap()], live.borrow()[&image]);
            assert_eq!(usage.said.lock().unwrap().len(), usize::from(differ));
            // A chain of one format list (vkd3d-proton's placed resources): judged once per list, then kept.
            let formats = [vk::Format::R16G16B16A16_SFLOAT, vk::Format::R16G16B16A16_UINT];
            let list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
            let (chained, chained_copyable) = mutable(std::ptr::from_ref(&list).cast());
            assert_eq!(usage.cached(&chained), None, "not the verdict of the create info without the list");
            *creates.borrow_mut() = 0;
            let (third, third_added) = usage.create(&chained, &chained_copyable, &create, &destroy, &requirements);
            assert_eq!((*creates.borrow(), usage.cached(&chained), third_added), (2, Some(!differ), !differ), "judged");
            let same_formats = formats;
            let same_list = vk::ImageFormatListCreateInfo::default().view_formats(&same_formats);
            let (same, same_copyable) = mutable(std::ptr::from_ref(&same_list).cast());
            *creates.borrow_mut() = 0;
            let (fourth, fourth_added) = usage.create(&same, &same_copyable, &create, &destroy, &requirements);
            assert_eq!((*creates.borrow(), fourth_added), (1, !differ), "the same formats in another list: the kept verdict, one image");
            assert_eq!(live.borrow()[fourth.as_ref().unwrap()], live.borrow()[third.as_ref().unwrap()]);
            let other_formats = [vk::Format::R16G16B16A16_SFLOAT];
            let other_list = vk::ImageFormatListCreateInfo::default().view_formats(&other_formats);
            let (other, _) = colour(std::ptr::from_ref(&other_list).cast());
            assert_eq!(usage.cached(&other), None, "other formats: another verdict");
            // A list followed by anything else: judged each time, never kept.
            let mut stencil = vk::ImageStencilUsageCreateInfo::default().stencil_usage(vk::ImageUsageFlags::SAMPLED);
            stencil.p_next = std::ptr::from_ref(&list).cast();
            let (longer, longer_copyable) = mutable(std::ptr::from_ref(&stencil).cast());
            let mut followed = list;
            followed.p_next = std::ptr::from_ref(&stencil).cast();
            let (behind, _) = mutable(std::ptr::from_ref(&followed).cast());
            *creates.borrow_mut() = 0;
            let (fifth, _) = usage.create(&longer, &longer_copyable, &create, &destroy, &requirements);
            assert_eq!((*creates.borrow(), usage.cached(&longer), usage.cached(&behind)), (2, None, None));
            for image in [image, again.unwrap(), third.unwrap(), fourth.unwrap(), fifth.unwrap()] {
                destroy(image);
            }
            assert!(live.borrow().is_empty());
        }
        unsafe {
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }

    /// The application's own query decides as creating both does, and the verdict it reached is what the image
    /// is created with: any one of size, alignment or memory types differing is a difference.
    #[test]
    fn a_query_s_verdict_is_what_the_image_is_created_with() {
        let r = vk::MemoryRequirements { size: 1 << 20, alignment: 256, memory_type_bits: 0b111 };
        let (asked, copyable) = colour(std::ptr::null());
        let usage = AddedUsage::default();
        assert!(usage.judge(&asked, &copyable, r, r));
        assert_eq!(usage.cached(&asked), Some(true));
        for other in [
            vk::MemoryRequirements { size: r.size + 4096, ..r },
            vk::MemoryRequirements { alignment: 65536, ..r },
            vk::MemoryRequirements { memory_type_bits: 0b011, ..r },
        ] {
            let usage = AddedUsage::default();
            assert!(!usage.judge(&asked, &copyable, r, other), "{other:?}");
            assert_eq!(usage.cached(&asked), Some(false));
            let made = RefCell::new(Vec::new());
            let create = |info: &vk::ImageCreateInfo| {
                made.borrow_mut().push(info.usage);
                Ok(vk::Image::null())
            };
            let (_, added) = usage.create(&asked, &copyable, &create, &|_| unreachable!(), &|_| unreachable!());
            assert!(!added);
            assert_eq!(*made.borrow(), vec![asked.usage], "created once, as asked");
        }
    }
}
