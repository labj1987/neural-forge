//! The wrapper's decisions that need no Windows: which NGX results are successes, which of an evaluate's parameters
//! it reads, whether it writes the ring, and what it does with exports NVIDIA's core lacks. Kept apart so they are
//! tested on Linux (the hooks that use them are Windows-only).

use crate::exports::{HOOKED, PASSTHROUGH};

/// `NVSDK_NGX_Result_Fail`: the code a failed NGX call's high 12 bits carry.
pub const NGX_RESULT_FAIL: u32 = 0xBAD0_0000;

/// `NVSDK_NGX_SUCCEED`: anything but a code in the `0xBAD00000` family.
pub fn ngx_succeeded(result: u32) -> bool {
    result & 0xFFF0_0000 != NGX_RESULT_FAIL
}

/// NGX's features whose images the layer uses: Super Resolution (1) and Ray Reconstruction (13).
pub const FEATURE_SR: u32 = 1;
pub const FEATURE_RR: u32 = 13;

/// `NVSDK_NGX_DLSS_Feature_Flags_AutoExposure`: DLSS measures exposure itself and ignores `ExposureTexture`.
pub const FLAG_AUTO_EXPOSURE: u32 = 64;

/// Whether an evaluate's images are looked up (each lookup calls into the resource the parameter names, which must
/// then be alive). Only for the features the layer reads images of; any other evaluate (Frame Generation, or a
/// feature whose creation the wrapper did not see, 0) is recorded without images.
pub fn reads_images(feature: u32) -> bool {
    matches!(feature, FEATURE_SR | FEATURE_RR)
}

/// Whether `ExposureTexture` is looked up: not with auto-exposure, where DLSS does not read it and a game need not
/// keep the parameter pointing at a live resource.
pub fn reads_exposure_texture(create_flags: u32) -> bool {
    create_flags & FLAG_AUTO_EXPOSURE == 0
}

/// Whether the wrapper creates its ring. Only in a process the Neural Forge layer is switched on in
/// (`NEURAL_FORGE_ENABLE=1`, the layer's `enable_environment`, and not `NEURAL_FORGE_DISABLE=1`), and not with
/// `NEURAL_FORGE_NGX_RING=off`: the layer is the ring's only reader and removes the file once it maps it, so in any
/// other process the file would be left in `/dev/shm` for nothing. `var` reads the environment.
pub fn ring_wanted(var: impl Fn(&str) -> Option<String>) -> bool {
    let flag = |name: &str| var(name).is_some_and(|v| v == "1");
    flag("NEURAL_FORGE_ENABLE") && !flag("NEURAL_FORGE_DISABLE") && var("NEURAL_FORGE_NGX_RING").is_none_or(|v| v.trim() != "off")
}

/// What becomes of NVIDIA's core's exports, given which names it has.
#[derive(Debug, PartialEq)]
pub enum Resolution {
    /// Every hooked export is there; the passthrough exports listed are not and answer `NVSDK_NGX_Result_Fail`
    /// ([`crate::exports::missing_export`]).
    Ok { stubbed: Vec<&'static str> },
    /// Hooked exports the core lacks: the wrapper refuses to load.
    MissingHooked(Vec<&'static str>),
}

/// The wrapper needs every export it hooks (it calls them with the caller's arguments and reads their results);
/// a passthrough export the core lacks (an older driver) is answered by a stub, as a game asking a core without it
/// would be refused.
pub fn resolution(has: impl Fn(&str) -> bool) -> Resolution {
    let missing: Vec<&'static str> = HOOKED.iter().copied().filter(|n| !has(n)).collect();
    if !missing.is_empty() {
        return Resolution::MissingHooked(missing);
    }
    Resolution::Ok { stubbed: PASSTHROUGH.iter().copied().filter(|n| !has(n)).collect() }
}

/// An image the evaluate's text line names: its VkImage, extent, VkFormat and layout.
pub fn image_text(key: &str, vk: u64, width: u32, height: u32, format: i32, layout: i32) -> String {
    format!(" {key}=vk:{vk:#x}({width}x{height} fmt={format} lay={layout})")
}

/// A parameter the evaluate's text line names by its bare pointer.
pub fn pointer_text(key: &str, ptr: usize) -> String {
    format!(" {key}={ptr:#x}")
}

/// The evaluate's text line (`EVAL ...`): the images (`images`, from [`image_text`] and [`pointer_text`]) first, then
/// every number of the record. The trace keeps it whole (`crate::trace::LINE_BYTES`; tested).
pub fn eval_line(api: &str, handle: usize, feature: &str, images: &str, rec: &crate::ring::Record) -> String {
    format!(
        "EVAL {api} h={}@{handle:#x} feat={feature}{images} jitter={},{} mv_scale={},{} pre_exposure={} exposure_scale={} reset={} render={}x{} bases=colour:{},{} mv:{},{} depth:{},{} output:{},{} out={}x{} flags={:#x}",
        rec.handle,
        rec.jitter[0],
        rec.jitter[1],
        rec.mv_scale[0],
        rec.mv_scale[1],
        rec.pre_exposure,
        rec.exposure_scale,
        rec.reset,
        rec.render[0],
        rec.render[1],
        rec.colour_base[0],
        rec.colour_base[1],
        rec.mvec_base[0],
        rec.mvec_base[1],
        rec.depth_base[0],
        rec.depth_base[1],
        rec.output_base[0],
        rec.output_base[1],
        rec.output_size[0],
        rec.output_size[1],
        rec.create_flags
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The longest evaluate line there can be (every image named with the largest numbers, every pointer, the longest
    /// float texts) reaches the trace whole: as the `log: ` line the trace keeps of it, with its time and thread.
    #[test]
    fn the_longest_eval_line_fits_a_trace_line() {
        let mut images = String::new();
        for key in ["Color", "MotionVectors", "Depth", "Output", "ExposureTexture"] {
            images += &image_text(key, u64::MAX, u32::MAX, u32::MAX, i32::MIN, i32::MIN);
        }
        for key in ["TransparencyMask", "DLSSG.Backbuffer", "DLSSG.HUDLess", "DLSSG.UI", "DLSSG.OutputInterpolated", "DLSSG.OutputReal"] {
            images += &pointer_text(key, usize::MAX);
        }
        // The longest f32 texts: the smallest subnormal (47 characters) and the most negative.
        let tiny = -f32::from_bits(1);
        let rec = crate::ring::Record {
            handle: u32::MAX,
            create_flags: u32::MAX,
            jitter: [tiny, -f32::MAX],
            mv_scale: [tiny, tiny],
            pre_exposure: tiny,
            exposure_scale: tiny,
            reset: i32::MIN,
            render: [u32::MAX; 2],
            colour_base: [u32::MAX; 2],
            mvec_base: [u32::MAX; 2],
            depth_base: [u32::MAX; 2],
            output_base: [u32::MAX; 2],
            output_size: [u32::MAX; 2],
            ..Default::default()
        };
        let text = eval_line("vulkan", usize::MAX, "SR", &images, &rec);
        let line = crate::trace::Line::new(f64::MAX.min(1e12), u32::MAX, format_args!("log: {text}"));
        let got = std::str::from_utf8(line.bytes()).unwrap();
        assert!(got.ends_with(&format!("{text}\n")), "cut at {} bytes: {} of the text's {}", crate::trace::LINE_BYTES, got.len(), text.len());
        assert!(got.contains("flags=0xffffffff"));
    }

    #[test]
    fn success_is_anything_outside_the_fail_family() {
        assert!(ngx_succeeded(1), "NVSDK_NGX_Result_Success");
        assert!(ngx_succeeded(0));
        for fail in [0xBAD0_0000, 0xBAD0_0001, 0xBAD0_0005, 0xBAD0_0010, 0xBAD0_FFFF, 0xBADF_FFFF] {
            assert!(!ngx_succeeded(fail), "{fail:#x}");
        }
        assert!(ngx_succeeded(0xBAE0_0000));
    }

    #[test]
    fn images_only_for_the_upscalers() {
        assert!(reads_images(FEATURE_SR));
        assert!(reads_images(FEATURE_RR));
        for other in [0, 11, 2, 12] {
            assert!(!reads_images(other), "{other}");
        }
    }

    #[test]
    fn no_exposure_texture_with_auto_exposure() {
        assert!(reads_exposure_texture(0));
        assert!(reads_exposure_texture(1 | 2 | 4 | 8));
        assert!(!reads_exposure_texture(FLAG_AUTO_EXPOSURE));
        assert!(!reads_exposure_texture(0x4b | FLAG_AUTO_EXPOSURE));
    }

    #[test]
    fn the_ring_only_where_the_layer_is_on() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        assert!(!ring_wanted(env(&[])), "a game the layer is not on");
        assert!(!ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "0")])));
        assert!(!ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "yes")])), "the layer reads exactly 1");
        assert!(ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "1")])));
        assert!(!ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "1"), ("NEURAL_FORGE_DISABLE", "1")])));
        assert!(ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "1"), ("NEURAL_FORGE_DISABLE", "0")])));
        assert!(!ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "1"), ("NEURAL_FORGE_NGX_RING", " off ")])));
        assert!(ring_wanted(env(&[("NEURAL_FORGE_ENABLE", "1"), ("NEURAL_FORGE_NGX_RING", "/tmp/r")])));
    }

    #[test]
    fn missing_passthrough_exports_are_stubbed() {
        assert_eq!(resolution(|_| true), Resolution::Ok { stubbed: vec![] });
        let gone = ["NVSDK_NGX_OTA_UPDATES_Install", "NVSDK_NGX_VULKAN_Init_Ext2"];
        assert_eq!(resolution(|n| !gone.contains(&n)), Resolution::Ok { stubbed: gone.to_vec() });
    }

    #[test]
    fn a_missing_hooked_export_refuses_the_load() {
        let r = resolution(|n| n != "NVSDK_NGX_VULKAN_CreateFeature1" && n != "NVSDK_NGX_CUDA_Init");
        assert_eq!(r, Resolution::MissingHooked(vec!["NVSDK_NGX_VULKAN_CreateFeature1"]));
        assert_eq!(resolution(|_| false), Resolution::MissingHooked(HOOKED.to_vec()));
    }
}
