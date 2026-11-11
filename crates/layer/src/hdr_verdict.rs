//! The HDR verdict: one line that says whether the game presents HDR and, when it does not, why, from what the
//! layer can see in the game's process. The swapchain's format and colour space (what the game asked for), whether
//! its surface offered an HDR colour space at all (`vkGetPhysicalDeviceSurfaceFormatsKHR`, asked once per swapchain
//! creation: where none is offered the game could not have chosen HDR, whatever its settings say), the first HDR
//! metadata the game sent (`vkSetHdrMetadataEXT`: a game with its HDR setting on sends it, also when its swapchain is
//! SDR), whether gamescope's WSI layer is loaded (in gamescope it decides which formats the surface offers) and
//! Proton's HDR switches (`PROTON_ENABLE_WAYLAND`, `PROTON_ENABLE_HDR`: on a Wayland desktop with HDR on, Proton's
//! Wayland driver offers HDR10 only with both).
//!
//! The verdict is logged when it changes and published on the channel (`ShmHeader::hdr_verdict`), where the
//! Status tab, `shmctl status`, `doctor` and Save report read it; the desktop's own output state (its HDR switch)
//! is added there by the front ends, which can ask the compositor (`neural_forge_supervisor::desktop_hdr`).
//! The "HDR under nested gamescope" report of #51 took three exchanges to answer from logs; this line answers it.

use ash::vk;
use neural_forge_protocol::enums::hdr_kind;

/// What the swapchain's surface offered, asked when it was created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// The query was not available or failed.
    Unknown,
    Offered {
        /// `HDR10_ST2084_EXT` with a 10-bit format (or any format: the colour space is what matters).
        hdr10: bool,
        /// `EXTENDED_SRGB_LINEAR_EXT` (scRGB, linear FP16).
        scrgb: bool,
    },
}

/// The first HDR metadata the game sent for the swapchain, in short.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    /// "BT.2020", "Display P3", "BT.709" or the primaries in numbers.
    pub primaries: String,
    pub max_luminance: f32,
    pub max_cll: f32,
}

/// Everything the verdict is made from, for one swapchain.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub surface: Surface,
    pub metadata: Option<Metadata>,
    /// gamescope's WSI layer (`libVkLayer_FROG_gamescope_wsi`) is loaded in the game.
    pub gamescope_wsi: bool,
    /// `PROTON_ENABLE_WAYLAND` and `PROTON_ENABLE_HDR`, `None` where unset (a game outside Proton has neither).
    pub proton_wayland: Option<bool>,
    pub proton_hdr: Option<bool>,
}

/// Whether an environment switch is on: set to anything but empty or `0`. `None` when unset.
fn switch(name: &str) -> Option<bool> {
    std::env::var_os(name).map(|v| !v.is_empty() && v != "0")
}

/// Whether gamescope's WSI layer is mapped into this process (`/proc/self/maps`).
fn gamescope_wsi_loaded() -> bool {
    std::fs::read_to_string("/proc/self/maps").is_ok_and(|maps| maps.contains("gamescope_wsi"))
}

impl Facts {
    /// The facts for a swapchain created with `format` in `color_space` on a surface that offered `surface`: the
    /// process's own (gamescope's layer, Proton's switches) read now.
    pub fn new(format: vk::Format, color_space: vk::ColorSpaceKHR, surface: Surface) -> Self {
        Self {
            format,
            color_space,
            surface,
            metadata: None,
            gamescope_wsi: gamescope_wsi_loaded(),
            proton_wayland: switch("PROTON_ENABLE_WAYLAND"),
            proton_hdr: switch("PROTON_ENABLE_HDR"),
        }
    }

    /// The conclusion in plain words, then the evidence: "game outputs HDR10 (swapchain A2B10G10R10_UNORM_PACK32
    /// HDR10_ST2084_EXT; surface offers HDR10; HDR metadata BT.2020, 1000 nits; gamescope WSI: no; Proton: Wayland
    /// on, HDR on)".
    pub fn verdict(&self) -> String {
        format!("{} ({})", self.conclusion(), self.evidence())
    }

    fn conclusion(&self) -> String {
        let kind = crate::swapchain::detect_hdr_kind(self.format, self.color_space);
        if kind == hdr_kind::PQ10 {
            return "game outputs HDR10".into();
        }
        if kind == hdr_kind::LINEAR_FP16 && self.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT {
            return "game outputs scRGB HDR (linear FP16), which Neural Forge presents untouched".into();
        }
        let offered_hdr = match self.surface {
            Surface::Offered { hdr10, scrgb } => Some(hdr10 || scrgb),
            Surface::Unknown => None,
        };
        let why_none = || -> String {
            if self.gamescope_wsi {
                "gamescope offered no HDR colour space: start gamescope with --hdr-enabled, on an HDR display".into()
            } else if self.proton_wayland == Some(true) && self.proton_hdr == Some(true) {
                "the window offered no HDR colour space: check that the desktop's HDR is on for this display".into()
            } else if self.proton_wayland.is_some() || self.proton_hdr.is_some() || std::env::var_os("STEAM_COMPAT_DATA_PATH").is_some() {
                "the window offered no HDR colour space: with the desktop's HDR on, PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1 \
                 in the launch options offer it (or gamescope with HDR)"
                    .into()
            } else {
                "the window offered no HDR colour space: check that the desktop's HDR is on for this display".into()
            }
        };
        match (&self.metadata, offered_hdr) {
            (Some(_), Some(true)) => "game asked for SDR although HDR metadata was sent: its HDR setting looks on, but it chose an SDR window \
                                      the surface could have made HDR10"
                .into(),
            (Some(_), Some(false)) => format!("game asked for SDR although HDR metadata was sent: {}", why_none()),
            (Some(_), None) => "game asked for SDR although HDR metadata was sent: its frames are SDR, or HDR10 in a window that does not say so".into(),
            (None, Some(true)) => "game outputs SDR; HDR10 was available (switch on the game's HDR setting for HDR)".into(),
            (None, Some(false)) => format!("game outputs SDR; {}", why_none()),
            (None, None) => "game outputs SDR".into(),
        }
    }

    fn evidence(&self) -> String {
        let surface = match self.surface {
            Surface::Offered { hdr10: true, scrgb: true } => "surface offers HDR10 and scRGB",
            Surface::Offered { hdr10: true, scrgb: false } => "surface offers HDR10",
            Surface::Offered { hdr10: false, scrgb: true } => "surface offers scRGB, not HDR10",
            Surface::Offered { hdr10: false, scrgb: false } => "surface offers no HDR colour space",
            Surface::Unknown => "surface formats not known",
        };
        let metadata = match &self.metadata {
            // Primaries that are none of the named ones are in the log's metadata line; here they would crowd out the rest.
            Some(m) => format!(
                "HDR metadata {}, {:.0} nits, MaxCLL {:.0}",
                if m.primaries.starts_with("r (") { "custom primaries" } else { &m.primaries },
                m.max_luminance,
                m.max_cll
            ),
            None => "no HDR metadata".into(),
        };
        let on = |v: Option<bool>| match v {
            Some(true) => "on",
            Some(false) => "off",
            None => "unset",
        };
        format!(
            "swapchain {:?} {:?}; {surface}; {metadata}; gamescope WSI: {}; Proton: Wayland {}, HDR {}",
            self.format,
            self.color_space,
            if self.gamescope_wsi { "yes" } else { "no" },
            on(self.proton_wayland),
            on(self.proton_hdr)
        )
    }
}

/// What `surface` offers to `physical_device`, through the next layer's `vkGetPhysicalDeviceSurfaceFormatsKHR`
/// (`query`): two calls, the count and the formats, the application's own surface handle. `Unknown` without the
/// query or when either call fails.
pub fn surface_offers(query: Option<vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>, physical_device: vk::PhysicalDevice, surface: vk::SurfaceKHR) -> Surface {
    let Some(query) = query else { return Surface::Unknown };
    if surface == vk::SurfaceKHR::null() {
        return Surface::Unknown;
    }
    let mut count = 0u32;
    // SAFETY: the next layer's own entry point, for a physical device and surface of this instance; the count call
    // passes a null array as the spec allows.
    if unsafe { query(physical_device, surface, &mut count, std::ptr::null_mut()) } != vk::Result::SUCCESS {
        return Surface::Unknown;
    }
    let mut formats = vec![vk::SurfaceFormatKHR::default(); count as usize];
    // SAFETY: as above, with room for `count` formats; INCOMPLETE (the list grew since) still fills what fits.
    let result = unsafe { query(physical_device, surface, &mut count, formats.as_mut_ptr()) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        return Surface::Unknown;
    }
    formats.truncate(count as usize);
    offered(&formats)
}

/// [`Surface::Offered`] for a surface's format list.
pub fn offered(formats: &[vk::SurfaceFormatKHR]) -> Surface {
    Surface::Offered {
        hdr10: formats.iter().any(|f| f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT),
        scrgb: formats.iter().any(|f| f.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT),
    }
}

/// The short form of `m` the verdict carries.
pub fn metadata(m: &vk::HdrMetadataEXT) -> Metadata {
    Metadata { primaries: crate::swapchain::primaries_name(m), max_luminance: m.max_luminance, max_cll: m.max_content_light_level }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(format: vk::Format, cs: vk::ColorSpaceKHR, surface: Surface) -> Facts {
        Facts { format, color_space: cs, surface, metadata: None, gamescope_wsi: false, proton_wayland: None, proton_hdr: None }
    }

    fn bt2020() -> Metadata {
        Metadata { primaries: "BT.2020".into(), max_luminance: 1000.0, max_cll: 1000.0 }
    }

    const TEN: vk::Format = vk::Format::A2B10G10R10_UNORM_PACK32;
    const HDR10: Surface = Surface::Offered { hdr10: true, scrgb: false };
    const NONE: Surface = Surface::Offered { hdr10: false, scrgb: false };

    #[test]
    fn an_hdr10_swapchain_is_hdr10_output() {
        let v = facts(TEN, vk::ColorSpaceKHR::HDR10_ST2084_EXT, HDR10).verdict();
        assert!(v.starts_with("game outputs HDR10 (swapchain A2B10G10R10_UNORM_PACK32 HDR10_ST2084_EXT; surface offers HDR10; no HDR metadata;"), "{v}");
        let fp16 = facts(vk::Format::R16G16B16A16_SFLOAT, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT, Surface::Offered { hdr10: true, scrgb: true }).verdict();
        assert!(fp16.starts_with("game outputs scRGB HDR"), "{fp16}");
    }

    #[test]
    fn metadata_on_an_sdr_swapchain_says_why() {
        // #51: Stellar Blade in nested gamescope without --hdr-enabled: metadata sent, sRGB swapchain, no HDR formats.
        let mut f = facts(TEN, vk::ColorSpaceKHR::SRGB_NONLINEAR, NONE);
        f.metadata = Some(bt2020());
        f.gamescope_wsi = true;
        let v = f.verdict();
        assert!(v.starts_with("game asked for SDR although HDR metadata was sent: gamescope offered no HDR colour space"), "{v}");
        assert!(v.contains("HDR metadata BT.2020, 1000 nits, MaxCLL 1000; gamescope WSI: yes"), "{v}");
        // The surface offered HDR10 and the game took sRGB anyway.
        f.surface = HDR10;
        assert!(f.verdict().starts_with("game asked for SDR although HDR metadata was sent: its HDR setting looks on, but it chose an SDR window"));
        f.surface = Surface::Unknown;
        assert!(f.verdict().contains("its frames are SDR, or HDR10 in a window that does not say so"));
    }

    #[test]
    fn sdr_output_names_the_missing_switch() {
        let mut f = facts(vk::Format::B8G8R8A8_UNORM, vk::ColorSpaceKHR::SRGB_NONLINEAR, NONE);
        f.proton_wayland = Some(false);
        assert!(f.verdict().contains("PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1"), "{}", f.verdict());
        f.proton_wayland = Some(true);
        f.proton_hdr = Some(true);
        assert!(f.verdict().contains("desktop's HDR is on"), "{}", f.verdict());
        assert!(f.verdict().ends_with("Proton: Wayland on, HDR on)"), "{}", f.verdict());
        f.surface = HDR10;
        assert!(f.verdict().starts_with("game outputs SDR; HDR10 was available"), "{}", f.verdict());
        f.surface = Surface::Unknown;
        assert!(f.verdict().starts_with("game outputs SDR (") && f.verdict().contains("surface formats not known"));
    }

    #[test]
    fn the_verdict_fits_the_channel() {
        let mut f = facts(TEN, vk::ColorSpaceKHR::SRGB_NONLINEAR, NONE);
        f.metadata = Some(Metadata { primaries: "r (0.000, 0.000) g (0.000, 0.000) b (0.000, 0.000)".into(), max_luminance: 10000.0, max_cll: 10000.0 });
        f.proton_wayland = Some(false);
        f.proton_hdr = Some(false);
        let v = f.verdict();
        assert!(v.len() < neural_forge_protocol::HDR_VERDICT_BYTES, "{} bytes: {v}", v.len());
        assert!(v.contains("HDR metadata custom primaries, 10000 nits"), "{v}");
        // The longest conclusion with the longest names, too.
        f.gamescope_wsi = false;
        f.surface = Surface::Offered { hdr10: false, scrgb: true };
        f.format = vk::Format::A2R10G10B10_UNORM_PACK32;
        f.color_space = vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD;
        let v = f.verdict();
        assert!(v.len() < neural_forge_protocol::HDR_VERDICT_BYTES, "{} bytes: {v}", v.len());
    }

    #[test]
    fn a_surface_list_is_read_by_colour_space() {
        let f = |cs| vk::SurfaceFormatKHR { format: TEN, color_space: cs };
        assert_eq!(offered(&[f(vk::ColorSpaceKHR::SRGB_NONLINEAR)]), NONE);
        assert_eq!(offered(&[f(vk::ColorSpaceKHR::SRGB_NONLINEAR), f(vk::ColorSpaceKHR::HDR10_ST2084_EXT)]), HDR10);
        assert_eq!(offered(&[f(vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT)]), Surface::Offered { hdr10: false, scrgb: true });
        assert_eq!(surface_offers(None, vk::PhysicalDevice::null(), vk::SurfaceKHR::null()), Surface::Unknown);
    }
}
