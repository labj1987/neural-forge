//! `swapchainMaintenance1` on the game's device, for the present fences the layer chains to the
//! presents it adds semaphores to (`crate::present_sync`): `VK_KHR_swapchain_maintenance1` (else
//! `VK_EXT_swapchain_maintenance1`) and its feature, added at `vkCreateDevice` on an NVIDIA device that
//! offers them when the application did not enable them itself.
//!
//! Each device extension depends on an instance extension (`VK_KHR_surface_maintenance1`,
//! `VK_EXT_surface_maintenance1`), which depends on `VK_KHR_get_surface_capabilities2`. The layer adds
//! the ones the application did not enable to an instance that enables `VK_KHR_surface`
//! ([`instance_additions`]). It cannot ask which ones the drivers offer while the instance is created
//! (`vkEnumerateInstanceExtensionProperties` through the chain crashes Mesa's `device_select` layer,
//! see `NeuralForgeGlobalHooks`), and needs not: the loader's `vkCreateInstance` terminator passes each
//! driver only the extensions that driver offers, and a driver that offers a device extension offers
//! every extension that extension depends on (the spec's "Extension Dependencies"). So a device that
//! offers `VK_KHR_swapchain_maintenance1` belongs to a driver whose instance has
//! `VK_KHR_surface_maintenance1` enabled when the layer asked for it. An instance the drivers refuse with
//! the additions is created again as the application asked.
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::sync::Mutex;

use ash::vk;

pub const KHR_SWAPCHAIN: &CStr = c"VK_KHR_swapchain_maintenance1";
pub const EXT_SWAPCHAIN: &CStr = c"VK_EXT_swapchain_maintenance1";
pub const KHR_SURFACE: &CStr = c"VK_KHR_surface_maintenance1";
pub const EXT_SURFACE: &CStr = c"VK_EXT_surface_maintenance1";
const SURFACE: &CStr = c"VK_KHR_surface";
const SURFACE_CAPABILITIES_2: &CStr = c"VK_KHR_get_surface_capabilities2";
const SWAPCHAIN: &CStr = c"VK_KHR_swapchain";
/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR`, which `_EXT` aliases.
const FEATURES: vk::StructureType = vk::StructureType::PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;

/// The names of a create info's extension list (`count` C strings at `names`, or none).
///
/// # Safety
/// `names` holds `count` NUL-terminated strings, or `count` is 0.
pub unsafe fn extension_names<'a>(names: *const *const c_char, count: u32) -> Vec<&'a CStr> {
    if count == 0 || names.is_null() {
        return Vec::new();
    }
    // SAFETY: per this function's contract.
    unsafe { std::slice::from_raw_parts(names, count as usize) }.iter().map(|&n| unsafe { CStr::from_ptr(n) }).collect()
}

/// The instance extensions the layer adds to an application's `vkCreateInstance` (`requested`): for an
/// instance that presents (`VK_KHR_surface`) and enables neither surface maintenance extension, what
/// the device extension depends on that it lacks. One that enables either already has what one of the
/// device extensions needs (DXVK and vkd3d-proton enable them when the driver offers them).
pub fn instance_additions(requested: &[&CStr]) -> Vec<&'static CStr> {
    if !requested.contains(&SURFACE) || requested.contains(&KHR_SURFACE) || requested.contains(&EXT_SURFACE) {
        return Vec::new();
    }
    [SURFACE_CAPABILITIES_2, KHR_SURFACE, EXT_SURFACE].into_iter().filter(|name| !requested.contains(name)).collect()
}

/// Which of the surface extensions an instance was created with (the application's, or the layer's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstanceSupport {
    pub khr: bool,
    pub ext: bool,
}

impl InstanceSupport {
    pub fn of(enabled: &[&CStr]) -> Self {
        let capabilities = enabled.contains(&SURFACE_CAPABILITIES_2) && enabled.contains(&SURFACE);
        Self { khr: capabilities && enabled.contains(&KHR_SURFACE), ext: capabilities && enabled.contains(&EXT_SURFACE) }
    }
}

/// Instances created through the layer's `vkCreateInstance`, and what they were created with: taken by
/// `create_instance_info` (which sees only the application's request).
static INSTANCES: Mutex<Option<HashMap<vk::Instance, InstanceSupport>>> = Mutex::new(None);

pub fn note_instance(instance: vk::Instance, support: InstanceSupport) {
    INSTANCES.lock().unwrap().get_or_insert_default().insert(instance, support);
}

pub fn take_instance(instance: vk::Instance) -> InstanceSupport {
    INSTANCES.lock().unwrap().as_mut().and_then(|map| map.remove(&instance)).unwrap_or_default()
}

/// Devices created with `swapchainMaintenance1` enabled: taken by `NeuralForgeDeviceInfo::new`.
static DEVICES: Mutex<Option<std::collections::HashSet<vk::Device>>> = Mutex::new(None);

pub fn note_enabled(device: vk::Device) {
    DEVICES.lock().unwrap().get_or_insert_default().insert(device);
}

pub fn take_enabled(device: vk::Device) -> bool {
    DEVICES.lock().unwrap().as_mut().is_some_and(|set| set.remove(&device))
}

/// What the device offers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Offered {
    pub khr: bool,
    pub ext: bool,
    /// The `swapchainMaintenance1` feature.
    pub feature: bool,
}

/// What a `vkCreateDevice` request gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// The request goes as it is: `swapchainMaintenance1` is enabled by the application's own request,
    /// or (`Err`) not, and why.
    AsIs(Result<(), &'static str>),
    /// The feature structure goes in front of the request's chain, and `extension` (when the application
    /// enables neither) into its list.
    Add { extension: Option<&'static CStr> },
}

/// The plan for a request on an NVIDIA device: `requested`, its extensions; `app_feature`, the
/// `swapchainMaintenance1` of the features structure it chains, if any; `properties2`, whether the
/// instance allows the device extension's other dependency (Vulkan 1.1, or
/// `VK_KHR_get_physical_device_properties2`).
pub fn plan(requested: &[&CStr], app_feature: Option<bool>, offered: Offered, instance: InstanceSupport, properties2: bool) -> Plan {
    let app_extension = requested.contains(&KHR_SWAPCHAIN) || requested.contains(&EXT_SWAPCHAIN);
    match app_feature {
        // Its own structure is the application's to set: never written, never given twice.
        Some(true) if app_extension => return Plan::AsIs(Ok(())),
        Some(_) => return Plan::AsIs(Err("the application chains the feature's structure without enabling it")),
        None => {}
    }
    if !requested.contains(&SWAPCHAIN) {
        return Plan::AsIs(Err("the application does not enable VK_KHR_swapchain"));
    }
    if !properties2 {
        return Plan::AsIs(Err("the instance is Vulkan 1.0 without VK_KHR_get_physical_device_properties2"));
    }
    if !offered.feature {
        return Plan::AsIs(Err("the device does not offer it"));
    }
    if app_extension {
        return Plan::Add { extension: None };
    }
    if offered.khr && instance.khr {
        Plan::Add { extension: Some(KHR_SWAPCHAIN) }
    } else if offered.ext && instance.ext {
        Plan::Add { extension: Some(EXT_SWAPCHAIN) }
    } else {
        Plan::AsIs(Err("the instance lacks VK_KHR_surface_maintenance1 and VK_EXT_surface_maintenance1"))
    }
}

/// The `swapchainMaintenance1` of the features structure in a device request's chain, if it has one.
///
/// # Safety
/// `info`'s chain is valid.
pub unsafe fn app_feature(info: &vk::DeviceCreateInfo) -> Option<bool> {
    let mut next = info.p_next.cast::<vk::BaseInStructure>();
    while !next.is_null() {
        // SAFETY: every node of a valid chain starts with `sType` and `pNext`; one of this type is the
        // features structure.
        let node = unsafe { &*next };
        if node.s_type == FEATURES {
            return Some(unsafe { &*next.cast::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT>() }.swapchain_maintenance1 == vk::TRUE);
        }
        next = node.p_next;
    }
    None
}

/// What `physical` offers, asked through `gipa` (the features query `properties2` allows).
///
/// # Safety
/// `gipa` and `instance` are the next layer's for `physical`; `extensions` is its extension list.
pub unsafe fn offered(
    gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice, properties2: crate::vram::Properties2, extensions: &[&CStr],
) -> Offered {
    let khr = extensions.contains(&KHR_SWAPCHAIN);
    let ext = extensions.contains(&EXT_SWAPCHAIN);
    let name: &CStr = match properties2 {
        crate::vram::Properties2::Core => c"vkGetPhysicalDeviceFeatures2",
        crate::vram::Properties2::Khr => c"vkGetPhysicalDeviceFeatures2KHR",
        crate::vram::Properties2::None => return Offered { khr, ext, feature: false },
    };
    if !(khr || ext) {
        return Offered::default();
    }
    // SAFETY: the name's signature.
    let features2 = unsafe { gipa(instance, name.as_ptr()) }.map(|f| unsafe { std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkGetPhysicalDeviceFeatures2>(f) });
    let Some(features2) = features2 else { return Offered { khr, ext, feature: false } };
    let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut maintenance);
    // SAFETY: a valid chain, filled in by the call.
    unsafe { features2(physical, &mut f2) };
    Offered { khr, ext, feature: maintenance.swapchain_maintenance1 == vk::TRUE }
}

/// A device request with [`Plan::Add`]'s additions: a copy of the application's create info whose
/// chain starts with the feature structure (then the application's chain, untouched) and whose
/// extension list has the extension added. Alive until `vkCreateDevice` returned.
pub struct Addition<'a> {
    _names: Vec<*const c_char>,
    _features: Box<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>>,
    /// The application's create info, pointing into `_names` and `_features` too: valid while `self` lives.
    pub info: vk::DeviceCreateInfo<'a>,
}

impl<'a> Addition<'a> {
    /// # Safety
    /// `base` is a valid create info (its extension list included) whose pointers outlive the result.
    pub unsafe fn new(base: &vk::DeviceCreateInfo<'a>, extension: Option<&'static CStr>) -> Self {
        let mut names: Vec<*const c_char> = if base.enabled_extension_count == 0 || base.pp_enabled_extension_names.is_null() {
            Vec::new()
        } else {
            // SAFETY: per this function's contract.
            unsafe { std::slice::from_raw_parts(base.pp_enabled_extension_names, base.enabled_extension_count as usize) }.to_vec()
        };
        names.extend(extension.map(CStr::as_ptr));
        let features = Box::new(vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT {
            p_next: base.p_next.cast_mut(),
            swapchain_maintenance1: vk::TRUE,
            ..Default::default()
        });
        let mut info = *base;
        info.p_next = std::ptr::from_ref(&*features).cast();
        info.enabled_extension_count = names.len() as u32;
        info.pp_enabled_extension_names = names.as_ptr();
        Self { _names: names, _features: features, info }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: Offered = Offered { khr: true, ext: true, feature: true };
    const BOTH: InstanceSupport = InstanceSupport { khr: true, ext: true };

    #[test]
    fn the_instance_gets_what_the_device_extension_depends_on_only_when_it_presents() {
        assert!(instance_additions(&[]).is_empty(), "no surface: nothing");
        assert!(instance_additions(&[c"VK_KHR_get_physical_device_properties2"]).is_empty());
        assert_eq!(instance_additions(&[SURFACE, c"VK_KHR_xlib_surface"]), [SURFACE_CAPABILITIES_2, KHR_SURFACE, EXT_SURFACE]);
        // DXVK and vkd3d-proton enable them themselves: nothing added.
        assert!(instance_additions(&[SURFACE, SURFACE_CAPABILITIES_2, KHR_SURFACE, EXT_SURFACE]).is_empty());
        assert!(instance_additions(&[SURFACE, SURFACE_CAPABILITIES_2, EXT_SURFACE]).is_empty(), "either one is enough");
        assert_eq!(instance_additions(&[SURFACE, SURFACE_CAPABILITIES_2]), [KHR_SURFACE, EXT_SURFACE]);

        assert_eq!(InstanceSupport::of(&[SURFACE, SURFACE_CAPABILITIES_2, KHR_SURFACE]), InstanceSupport { khr: true, ext: false });
        assert_eq!(InstanceSupport::of(&[SURFACE, KHR_SURFACE, EXT_SURFACE]), InstanceSupport::default(), "a dependency missing");
    }

    #[test]
    fn the_device_gets_the_extension_and_feature_only_where_they_are_offered_and_not_enabled() {
        let app = [SWAPCHAIN];
        assert_eq!(plan(&app, None, ALL, BOTH, true), Plan::Add { extension: Some(KHR_SWAPCHAIN) }, "KHR first");
        assert_eq!(plan(&app, None, Offered { khr: false, ..ALL }, BOTH, true), Plan::Add { extension: Some(EXT_SWAPCHAIN) });
        assert_eq!(plan(&app, None, ALL, InstanceSupport { khr: false, ext: true }, true), Plan::Add { extension: Some(EXT_SWAPCHAIN) }, "the instance's dependency decides");
        assert!(matches!(plan(&app, None, ALL, InstanceSupport::default(), true), Plan::AsIs(Err(_))));
        assert!(matches!(plan(&app, None, Offered { feature: false, ..ALL }, BOTH, true), Plan::AsIs(Err(_))));
        assert!(matches!(plan(&app, None, ALL, BOTH, false), Plan::AsIs(Err(_))), "Vulkan 1.0 without properties2");
        assert!(matches!(plan(&[], None, ALL, BOTH, true), Plan::AsIs(Err(_))), "no swapchains");
        // The application enables it all itself (DXVK, vkd3d-proton): as it is.
        assert_eq!(plan(&[SWAPCHAIN, EXT_SWAPCHAIN], Some(true), ALL, BOTH, true), Plan::AsIs(Ok(())));
        // Its own structure, set to false: never written over, never given twice.
        assert!(matches!(plan(&[SWAPCHAIN, KHR_SWAPCHAIN], Some(false), ALL, BOTH, true), Plan::AsIs(Err(_))));
        // The extension without the structure: only the structure is added.
        assert_eq!(plan(&[SWAPCHAIN, KHR_SWAPCHAIN], None, ALL, BOTH, true), Plan::Add { extension: None });
    }

    #[test]
    fn the_addition_puts_the_feature_in_front_of_the_applications_chain_and_keeps_its_list() {
        let mut theirs = vk::PhysicalDeviceVulkan12Features::default();
        let theirs_at = std::ptr::from_ref(&theirs).cast::<std::ffi::c_void>();
        let names = [SWAPCHAIN.as_ptr()];
        let base = vk::DeviceCreateInfo::default().enabled_extension_names(&names).push_next(&mut theirs);
        // SAFETY: `base` and what it points to outlive `addition`.
        let addition = unsafe { Addition::new(&base, Some(KHR_SWAPCHAIN)) };
        // SAFETY: the addition's list and chain, alive with it.
        unsafe {
            assert_eq!(extension_names(addition.info.pp_enabled_extension_names, addition.info.enabled_extension_count), [SWAPCHAIN, KHR_SWAPCHAIN]);
            assert_eq!(app_feature(&addition.info), Some(true));
            assert_eq!(app_feature(&base), None);
            let first = &*addition.info.p_next.cast::<vk::BaseInStructure>();
            assert_eq!(first.s_type, FEATURES);
            assert_eq!(first.p_next.cast(), theirs_at, "the application's chain follows, untouched");
        }
        assert_eq!(base.enabled_extension_count, 1, "the application's request is only read");
    }
}
