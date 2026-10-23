//! `NEURAL_FORGE_PROBE_IMAGE=<kernel prefix>[+],...`: a diagnostic that copies every registered colour image a CUDA
//! launch names into a file, once per kernel name, after [`SKIP`] launches of it (the scene settled): before the
//! launch, or after it with a `+`. For seeing what an integration hands NGX and what NGX gives back (Metro Last
//! Light Redux runs NVIDIA's own NR after DLSS Super Resolution, 2026-10-23). The files are raw texels,
//! `<kernel>-<before|after>-<n>-<image>-<w>x<h>-<format>.raw` in a new `captures/probe-<ms>/`, written by a thread
//! a few seconds later (the frame has long finished by then). The host buffers are never destroyed (`HostBuffer`
//! has no destructor): the game may submit the recorded buffer again. Never set in normal use.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use ash::vk;
use ash::vk::Handle;

use super::ImageDesc;

/// Launches of a kernel let go by before its images are taken.
const SKIP: u32 = 300;

struct Spec {
    prefix: String,
    after: bool,
}

static SPECS: LazyLock<Vec<Spec>> = LazyLock::new(|| {
    neural_forge_protocol::env::var("NEURAL_FORGE_PROBE_IMAGE")
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| match s.strip_suffix('+') {
                    Some(p) => Spec { prefix: p.to_string(), after: true },
                    None => Spec { prefix: s.to_string(), after: false },
                })
                .collect()
        })
        .unwrap_or_default()
});

/// Per kernel name and side (after the launch or not): launches seen, and whether its images were taken.
type Seen = HashMap<(String, bool), (u32, bool)>;
static SEEN: LazyLock<Mutex<Seen>> = LazyLock::new(|| Mutex::new(HashMap::new()));

static DIR: LazyLock<std::path::PathBuf> = LazyLock::new(|| {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    crate::dump::captures_dir().join(format!("probe-{ms}"))
});

pub(crate) fn enabled() -> bool {
    !SPECS.is_empty()
}

/// Whether this launch of `name` is the one whose images are taken, `after` it or before.
pub(crate) fn due(name: &str, after: bool) -> bool {
    if !SPECS.iter().any(|s| s.after == after && name.starts_with(&s.prefix)) {
        return false;
    }
    let mut seen = SEEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = seen.entry((name.to_string(), after)).or_insert((0, false));
    entry.0 += 1;
    if entry.1 || entry.0 < SKIP {
        return false;
    }
    entry.1 = true;
    true
}

/// Bytes per texel of a colour format the probe copies.
fn texel_bytes(format: vk::Format) -> Option<u64> {
    match format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM | vk::Format::B10G11R11_UFLOAT_PACK32 | vk::Format::A2B10G10R10_UNORM_PACK32 => Some(4),
        f => super::exposure_texel_bytes(f),
    }
}

/// Records copies of `images` (those that are plain, transferable colour images, taken as in `GENERAL`) into
/// `cb`, and a thread that writes them out.
///
/// # Safety
/// `cb` is recording on this thread, outside a render pass; the images are live and in `GENERAL` (NGX's storage
/// images are).
pub(crate) unsafe fn record(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    cb: vk::CommandBuffer,
    name: &str,
    after: bool,
    images: &[(vk::Image, ImageDesc)],
) {
    let side = if after { "after" } else { "before" };
    let mut taken = Vec::new();
    let all = vk::PipelineStageFlags::ALL_COMMANDS;
    let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ);
    // SAFETY: recording (contract).
    unsafe { device.cmd_pipeline_barrier(cb, all, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
    for (n, &(image, d)) in images.iter().enumerate() {
        let Some(bytes) = texel_bytes(d.format).filter(|_| d.plain && d.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)) else { continue };
        let size = bytes * u64::from(d.width) * u64::from(d.height);
        let Some(host) = super::own_host_buffer_with(device, instance, physical, size, vk::BufferUsageFlags::TRANSFER_DST) else { continue };
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: d.width, height: d.height, depth: 1 },
        };
        // SAFETY: as above; the buffer is the probe's own.
        unsafe { device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::GENERAL, host.buffer, &[region]) };
        let file = format!("{name}-{side}-{n}-{:x}-{}x{}-{:?}.raw", image.as_raw(), d.width, d.height, d.format);
        taken.push((file, host.ptr as usize, size as usize));
    }
    let done = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ);
    // SAFETY: as above.
    unsafe {
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, all | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[done], &[], &[])
    };
    crate::log!("[probe-image] {name} {side}: {} image(s) copied, written to {} in a few seconds", taken.len(), DIR.display());
    let _ = std::thread::Builder::new().name("nf-probe-image".into()).spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(4));
        let _ = std::fs::create_dir_all(&*DIR);
        for (file, ptr, size) in taken {
            // SAFETY: a host-visible, coherent mapping of `size` bytes that is never unmapped.
            let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
            let _ = std::fs::write(DIR.join(&file), data);
        }
        crate::log!("[probe-image] written: {}", DIR.display());
    });
}
