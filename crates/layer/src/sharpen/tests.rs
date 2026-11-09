use super::*;
use neural_forge_protocol::enums::hud_mode;

#[test]
fn the_override_is_off_or_named_settings() {
    assert_eq!(parse_env(None), Ok(Override::default()));
    assert_eq!(parse_env(Some("  ")), Ok(Override::default()));
    assert_eq!(parse_env(Some("off")), Ok(Override { off: true, ..Override::default() }));
    assert_eq!(parse_env(Some("0")), Ok(Override { off: true, ..Override::default() }));
    let o = parse_env(Some("cas=0.5, clarity=3,radius=1")).unwrap();
    assert_eq!(o.values, [Some(0.5), Some(2.0), Some(2.0), None, None], "clamped to the settings' ranges");
    assert!(parse_env(Some("cas")).is_err());
    assert!(parse_env(Some("sharpness=1")).is_err());
    assert!(parse_env(Some("cas=lots")).is_err());
    assert!(parse_env(Some("cas=nan")).is_err());
    let s = SharpenSettings { cas: 0.2, clarity: 0.3, radius: 10.0, midtones: 0.1, edges: 0.9 };
    assert_eq!(o.apply(s), SharpenSettings { cas: 0.5, clarity: 2.0, radius: 2.0, ..s });
    assert!(!Override { off: true, ..o }.apply(s).any());
    assert_eq!(Override::default().apply(s), s);
}

#[test]
fn the_cas_strength_starts_from_nothing_and_ends_at_cas_s_strongest() {
    assert_eq!(cas_peak(0.0), 0.0);
    assert!((cas_peak(1.0) + 0.2).abs() < 1e-7, "CAS's -1/5 at sharpness 1");
    assert!((cas_peak(8.0 / 11.0) + 0.125).abs() < 1e-6, "CAS's own default (sharpness 0, -1/8)");
    assert!(cas_peak(0.3) < cas_peak(0.2), "stronger with the strength");
    assert_eq!(half_radius(16.0, 1440), 8);
    assert_eq!(half_radius(16.0, 2160), 12);
    assert_eq!(half_radius(64.0, 2160), 32, "at most 32 taps each side");
    assert_eq!(half_radius(2.0, 720), 1);
    assert_eq!(edge_stop(0.0), 0.0);
    assert!(edge_stop(1.0) > edge_stop(0.5));
}

// --- the shader's reference on the CPU ---------------------------------------------------------------------------

fn decode(word: u32, ten: bool) -> [f32; 3] {
    if ten {
        [word & 1023, (word >> 10) & 1023, (word >> 20) & 1023].map(|v| v as f32 / 1023.0)
    } else {
        [word & 255, (word >> 8) & 255, (word >> 16) & 255].map(|v| v as f32 / 255.0)
    }
}

fn encode(c: [f32; 3], word: u32, ten: bool) -> u32 {
    let c = c.map(|v| v.clamp(0.0, 1.0));
    if ten {
        let q = c.map(|v| (v * 1023.0).round() as u32);
        (word & 0xC000_0000) | q[0] | (q[1] << 10) | (q[2] << 20)
    } else {
        let q = c.map(|v| (v * 255.0).round() as u32);
        (word & 0xFF00_0000) | q[0] | (q[1] << 8) | (q[2] << 16)
    }
}

fn weights(f: &Frame) -> [f32; 3] {
    let w = if f.ten_bit { [0.2627, 0.6780, 0.0593] } else { [0.2126, 0.7152, 0.0722] };
    if f.bgr {
        [w[2], w[1], w[0]]
    } else {
        w
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The shader's frame, step for step, on the CPU.
fn reference(words: &[u32], f: &Frame, s: &SharpenSettings, regions: &[HudRegion]) -> Vec<u32> {
    let (w, h) = (f.width as i32, f.height as i32);
    let (hw, hh) = ((w + 1) / 2, (h + 1) / 2);
    let push = Push::of(f, s, regions.len() as u32, 3);
    let at = |x: i32, y: i32, sw: i32, sh: i32| (y.clamp(0, sh - 1) * sw + x.clamp(0, sw - 1)) as usize;
    let load = |x: i32, y: i32| decode(words[at(x, y, w, h)], f.ten_bit);
    let lw = weights(f);
    let mut base = vec![0.0f32; (hw * hh) as usize];
    if s.clarity > 0.0 {
        let mut luma = vec![0.0f32; base.len()];
        for y in 0..hh {
            for x in 0..hw {
                let (fx, fy) = (x * 2, y * 2);
                luma[at(x, y, hw, hh)] = 0.25 * (dot(load(fx, fy), lw) + dot(load(fx + 1, fy), lw) + dot(load(fx, fy + 1), lw) + dot(load(fx + 1, fy + 1), lw));
            }
        }
        let r = push.clarity[1] as i32;
        let blur = |src: &[f32], dx: i32, dy: i32| {
            let mut out = vec![0.0f32; src.len()];
            for y in 0..hh {
                for x in 0..hw {
                    let centre = src[at(x, y, hw, hh)];
                    let (mut sum, mut weight) = (0.0f32, 0.0f32);
                    for t in -r..=r {
                        let v = src[at(x + dx * t, y + dy * t, hw, hh)];
                        let d = v - centre;
                        let wt = (-((t * t) as f32) * push.misc[1] - d * d * push.clarity[3]).exp();
                        sum += v * wt;
                        weight += wt;
                    }
                    out[at(x, y, hw, hh)] = sum / weight.max(1e-6);
                }
            }
            out
        };
        let rows = blur(&luma, 1, 0);
        base = blur(&rows, 0, 1);
    }
    let mut out = vec![0u32; words.len()];
    for y in 0..h {
        for x in 0..w {
            let word = words[at(x, y, w, h)];
            let e = decode(word, f.ten_bit);
            let mut c = e;
            if s.cas > 0.0 {
                let sq = |v: [f32; 3]| v.map(|a| a * a);
                let n: Vec<[f32; 3]> =
                    [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)].iter().map(|&(dx, dy)| sq(load(x + dx, y + dy))).collect();
                let (a, b, cc, d, ee, ff, g, hh2, i) = (n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]);
                let k = 1;
                let mn = [b[k], d[k], ee[k], ff[k], hh2[k]].into_iter().fold(f32::INFINITY, f32::min);
                let mn2 = [mn, a[k], cc[k], g[k], i[k]].into_iter().fold(f32::INFINITY, f32::min);
                let mx = [b[k], d[k], ee[k], ff[k], hh2[k]].into_iter().fold(f32::NEG_INFINITY, f32::max);
                let mx2 = [mx, a[k], cc[k], g[k], i[k]].into_iter().fold(f32::NEG_INFINITY, f32::max);
                let (mn, mx) = (mn + mn2, mx + mx2);
                let amp = if mx > 0.0 { (mn.min(2.0 - mx) / mx).clamp(0.0, 1.0).sqrt() } else { 0.0 };
                let wg = amp * push.cas[0];
                let rcp = 1.0 / (1.0 + 4.0 * wg);
                c = [0, 1, 2].map(|j| ((b[j] * wg + d[j] * wg + ff[j] * wg + hh2[j] * wg + ee[j]) * rcp).clamp(0.0, 1.0).sqrt());
            }
            if s.clarity > 0.0 {
                let yv = dot(e, lw);
                let (qx, qy) = ((x as f32 + 0.5) * 0.5 - 0.5, (y as f32 + 0.5) * 0.5 - 0.5);
                let (x0, y0) = (qx.floor() as i32, qy.floor() as i32);
                let (fx, fy) = (qx - x0 as f32, qy - y0 as f32);
                let bv = |xx: i32, yy: i32| base[at(xx, yy, hw, hh)];
                let top = bv(x0, y0) + (bv(x0 + 1, y0) - bv(x0, y0)) * fx;
                let bottom = bv(x0, y0 + 1) + (bv(x0 + 1, y0 + 1) - bv(x0, y0 + 1)) * fx;
                let detail = yv - (top + (bottom - top) * fy);
                let mid = 1.0 + (4.0 * yv * (1.0 - yv) - 1.0) * push.clarity[2];
                c = c.map(|v| v + push.clarity[0] * detail * mid);
            }
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let feather = push.misc[0].max(1.0);
            let mut scale = 1.0f32;
            for r in regions {
                let (l, t, rr, bb) = (r.x * w as f32, r.y * h as f32, (r.x + r.w) * w as f32, (r.y + r.h) * h as f32);
                let inside = if r.mode == hud_mode::SOFTEN { r.strength } else { 0.0 };
                let dd = (l - px).max(px - rr).max(0.0).max((t - py).max(py - bb).max(0.0));
                let v = if dd >= feather { 1.0 } else { inside + (1.0 - inside) * (dd / feather) };
                scale = scale.min(v);
            }
            let c = [0, 1, 2].map(|j| e[j] + scale * (c[j] - e[j]));
            out[at(x, y, w, h)] = if c == e { word } else { encode(c, word, f.ten_bit) };
        }
    }
    out
}

// --- running the shader on a test device ---------------------------------------------------------------------------

struct TestImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

/// A device-local image of `format` holding `words`, in `GENERAL`.
unsafe fn upload(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    format: vk::Format,
    (width, height): (u32, u32),
    words: &[u32],
) -> TestImage {
    unsafe {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device.create_image(&info, None).unwrap();
        let reqs = device.get_image_memory_requirements(image);
        let index = (0..props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
            .unwrap();
        let memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(reqs.size).memory_type_index(index), None).unwrap();
        device.bind_image_memory(image, memory, 0).unwrap();
        let staging = Buf::new(device, props, (words.len() * 4) as u64, vk::BufferUsageFlags::TRANSFER_SRC, true).unwrap();
        std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), staging.ptr, words.len() * 4);
        one_shot(device, queue, pool, |cmd| {
            let range =
                vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let b = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[b],
            );
            device.cmd_copy_buffer_to_image(cmd, staging.buffer, image, vk::ImageLayout::GENERAL, &[region(width, height)]);
        });
        staging.destroy(device);
        TestImage { image, memory }
    }
}

fn region(width: u32, height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
        image_extent: vk::Extent3D { width, height, depth: 1 },
        ..Default::default()
    }
}

unsafe fn one_shot(device: &ash::Device, queue: vk::Queue, pool: vk::CommandPool, record: impl FnOnce(vk::CommandBuffer)) {
    unsafe {
        let cmd = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).unwrap()[0];
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).unwrap();
        record(cmd);
        device.end_command_buffer(cmd).unwrap();
        device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], vk::Fence::null()).unwrap();
        device.queue_wait_idle(queue).unwrap();
        device.free_command_buffers(pool, &[cmd]);
    }
}

/// The image's words, read back (after everything submitted on `queue`).
unsafe fn read(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    image: vk::Image,
    (width, height): (u32, u32),
) -> Vec<u32> {
    unsafe {
        let n = (width * height) as usize;
        let host = Buf::new(device, props, (n * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST, true).unwrap();
        one_shot(device, queue, pool, |cmd| {
            let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[b],
                &[],
                &[],
            );
            device.cmd_copy_image_to_buffer(cmd, image, vk::ImageLayout::GENERAL, host.buffer, &[region(width, height)]);
            let h = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[h], &[], &[]);
        });
        let mut out = vec![0u32; n];
        std::ptr::copy_nonoverlapping(host.ptr, out.as_mut_ptr().cast::<u8>(), n * 4);
        host.destroy(device);
        out
    }
}

/// A frame with edges, texture, flat areas, both extremes and every alpha (a fixed pseudo-random pattern).
fn pattern(width: u32, height: u32, ten: bool) -> Vec<u32> {
    let mut seed = 0x1234_5678u32;
    let max = if ten { 1023 } else { 255 };
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 24) % 48;
            let c = |shift: u32| -> u32 {
                let v = if x < width / 4 {
                    max / 3 // flat
                } else if (x / 5 + y / 7) % 2 == 0 {
                    max * 3 / 4 + noise.min(max / 4)
                } else {
                    noise + shift
                };
                v.min(max)
            };
            let alpha = if ten { (i % 4) << 30 } else { (i % 256) << 24 };
            let (r, g, b) = (c(0), c(8), if y % 9 == 0 { max } else { c(16) });
            if ten {
                alpha | r | (g << 10) | (b << 20)
            } else {
                alpha | r | (g << 8) | (b << 16)
            }
        })
        .collect()
}

fn channels(word: u32, ten: bool) -> [i32; 4] {
    if ten {
        [(word & 1023) as i32, ((word >> 10) & 1023) as i32, ((word >> 20) & 1023) as i32, (word >> 30) as i32]
    } else {
        [0, 8, 16, 24].map(|s| ((word >> s) & 255) as i32)
    }
}

/// Runs `cases` (settings and regions) through `Sharpen::apply` on a fresh image of `pattern` each, and checks
/// each result against the reference: within one code value per channel, alpha bit for bit, and every pixel the
/// reference leaves alone exactly the frame's.
fn check_against_reference(format: vk::Format, bgr: bool, ten: bool, cases: &[(SharpenSettings, Vec<HudRegion>)]) {
    let Some(gpu) = crate::composition::gpu::test_device() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    let (instance, physical, device, queue, family) = (&gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
    const W: u32 = 61;
    const H: u32 = 37;
    // SAFETY: a fresh device; everything is used on its one queue and waited for.
    unsafe {
        let props = instance.get_physical_device_memory_properties(physical);
        let fp = instance.get_physical_device_format_properties(physical, format);
        if !fp.optimal_tiling_features.contains(vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST) {
            eprintln!("{format:?} cannot be copied on this device: skipped");
            return;
        }
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
        let mut sharpen = Sharpen::new(device, instance, physical, family).expect("the pass builds");
        let words = pattern(W, H, ten);
        for (s, regions) in cases {
            let img = upload(device, &props, queue, pool, format, (W, H), &words);
            let frame = Frame { image: img.image, layout: vk::ImageLayout::GENERAL, width: W, height: H, bgr, ten_bit: ten };
            let applied = sharpen.apply(device, queue, &frame, s, regions);
            assert_eq!(applied.is_some(), s.any(), "submitted only with a filter on ({s:?})");
            if let Some(sem) = applied {
                // The semaphore was signalled; nothing waits on it here, so take it back with a wait-only submit.
                let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
                device.queue_submit(queue, &[vk::SubmitInfo::default().wait_semaphores(&[sem]).wait_dst_stage_mask(&stage)], vk::Fence::null()).unwrap();
            }
            let got = read(device, &props, queue, pool, img.image, (W, H));
            let want = if s.any() { reference(&words, &frame, s, regions) } else { words.clone() };
            let mut changed = 0;
            for (k, ((&g, &r), &o)) in got.iter().zip(&want).zip(&words).enumerate() {
                let (gc, rc, oc) = (channels(g, ten), channels(r, ten), channels(o, ten));
                assert_eq!(gc[3], oc[3], "alpha kept at {k} ({s:?})");
                if r == o {
                    assert_eq!(g, o, "the reference leaves pixel {k} alone, so must the shader ({s:?})");
                } else {
                    changed += 1;
                }
                for j in 0..3 {
                    assert!((gc[j] - rc[j]).abs() <= 1, "pixel {k} channel {j}: shader {} reference {} ({s:?})", gc[j], rc[j]);
                }
            }
            if s.any() {
                assert!(changed > (W * H) as usize / 10, "the filter changed the frame ({changed} pixels, {s:?})");
            }
            device.destroy_image(img.image, None);
            device.free_memory(img.memory, None);
        }
        assert!(sharpen.wait_idle(device));
        device.queue_wait_idle(queue).unwrap();
        sharpen.destroy(device);
        device.destroy_command_pool(pool, None);
    }
}

fn keep(x: f32, y: f32, w: f32, h: f32) -> HudRegion {
    HudRegion { x, y, w, h, mode: hud_mode::KEEP, strength: 0.5 }
}

/// Off is nothing submitted and the frame untouched; CAS alone, Clarity alone and both match the reference
/// within one 8-bit step; a flat area is left bit for bit; a `keep` region is the frame's own inside, a `soften`
/// one part of the change.
#[test]
fn the_shader_matches_its_reference_and_keeps_the_hud_regions_8_bit() {
    let cas = SharpenSettings { cas: 0.8, ..SharpenSettings::DEFAULT };
    let clarity = SharpenSettings { clarity: 1.0, radius: 12.0, midtones: 0.5, edges: 0.5, ..SharpenSettings::DEFAULT };
    let both = SharpenSettings { cas: 1.0, clarity: 0.6, radius: 6.0, midtones: 0.0, edges: 1.0 };
    let regions = vec![keep(0.5, 0.5, 0.3, 0.3), HudRegion { x: 0.0, y: 0.0, w: 0.5, h: 0.3, mode: hud_mode::SOFTEN, strength: 0.25 }];
    check_against_reference(
        vk::Format::R8G8B8A8_UNORM,
        false,
        false,
        &[(SharpenSettings::DEFAULT, vec![]), (cas, vec![]), (clarity, vec![]), (both, vec![]), (both, regions)],
    );
}

/// The same for an HDR10 frame with blue first (A2R10G10B10): ten bits a channel, the two alpha bits kept.
#[test]
fn the_shader_matches_its_reference_10_bit_blue_first() {
    let both = SharpenSettings { cas: 0.5, clarity: 0.8, radius: 16.0, midtones: 0.5, edges: 0.5 };
    check_against_reference(vk::Format::A2R10G10B10_UNORM_PACK32, true, true, &[(both, vec![]), (both, vec![keep(0.1, 0.1, 0.5, 0.5)])]);
}

/// Inside a `keep` region the frame is its own, bit for bit, whatever the filters do outside (checked directly,
/// not only through the reference).
#[test]
fn a_keep_region_is_left_bit_for_bit() {
    let r = reference(
        &pattern(40, 30, false),
        &Frame { image: vk::Image::null(), layout: vk::ImageLayout::GENERAL, width: 40, height: 30, bgr: false, ten_bit: false },
        &SharpenSettings { cas: 1.0, clarity: 2.0, radius: 8.0, midtones: 0.0, edges: 0.0 },
        &[keep(0.5, 0.0, 0.5, 1.0)],
    );
    let words = pattern(40, 30, false);
    for y in 0..30 {
        for x in 20..40 {
            assert_eq!(r[y * 40 + x], words[y * 40 + x], "({x}, {y})");
        }
    }
    assert!((0..30).any(|y| (0..16).any(|x| r[y * 40 + x] != words[y * 40 + x])), "changed outside");
}

/// The cost on the device the loader offers first (run on the RTX 5070 with the NVIDIA ICD, under the GPU lock):
/// 2560x1440, each setting 200 frames after 20 to warm up, the GPU time's median. With
/// `NEURAL_FORGE_SHARPEN_TEST_PNG` (an RGBA8 PNG) and `NEURAL_FORGE_SHARPEN_TEST_OUT` (a directory) it also
/// writes the frame through each setting as PNGs and prints what each changed: the mean change of the 8-bit
/// channels, and the mean absolute Laplacian of luma (local contrast at the finest scale) against the frame's.
#[test]
#[ignore = "a measurement: run explicitly on the target GPU"]
fn cost_at_1440p_and_effect_on_a_still_frame() {
    let Some(gpu) = crate::composition::gpu::test_device() else {
        eprintln!("no Vulkan device: skipped");
        return;
    };
    let (instance, physical, device, queue, family) = (&gpu.1, gpu.2, &gpu.3, gpu.4, gpu.5);
    // SAFETY: a plain query.
    let name =
        unsafe { instance.get_physical_device_properties(physical) }.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let input = std::env::var("NEURAL_FORGE_SHARPEN_TEST_PNG").ok();
    let out_dir = std::env::var("NEURAL_FORGE_SHARPEN_TEST_OUT").ok();
    let (w, h, words) = match &input {
        Some(path) => {
            let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            assert_eq!(info.color_type, png::ColorType::Rgba, "an RGBA8 PNG");
            let words: Vec<u32> = buf[..info.buffer_size()].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            (info.width, info.height, words)
        }
        None => (2560, 1440, pattern(2560, 1440, false)),
    };
    let cases = [
        ("cas-0.5", SharpenSettings { cas: 0.5, ..SharpenSettings::DEFAULT }),
        ("cas-1.0", SharpenSettings { cas: 1.0, ..SharpenSettings::DEFAULT }),
        ("clarity-0.5-r16", SharpenSettings { clarity: 0.5, ..SharpenSettings::DEFAULT }),
        ("clarity-0.5-r64", SharpenSettings { clarity: 0.5, radius: 64.0, ..SharpenSettings::DEFAULT }),
        ("cas-0.5-clarity-0.5-r16", SharpenSettings { cas: 0.5, clarity: 0.5, ..SharpenSettings::DEFAULT }),
    ];
    let luma = |words: &[u32]| -> Vec<f32> { words.iter().map(|&wd| dot(decode(wd, false), [0.2126, 0.7152, 0.0722])).collect() };
    let laplacian = |l: &[f32]| -> f64 {
        let (w, h) = (w as usize, h as usize);
        let mut s = 0.0f64;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let c = l[y * w + x];
                s += f64::from((4.0 * c - l[y * w + x - 1] - l[y * w + x + 1] - l[(y - 1) * w + x] - l[(y + 1) * w + x]).abs());
            }
        }
        s / ((w - 2) * (h - 2)) as f64
    };
    let base_lap = laplacian(&luma(&words));
    eprintln!("[sharpen-cost] device {name}, {w}x{h}");
    // SAFETY: a fresh device; everything is used on its one queue and waited for.
    unsafe {
        let props = instance.get_physical_device_memory_properties(physical);
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None).unwrap();
        let mut sharpen = Sharpen::new(device, instance, physical, family).expect("the pass builds");
        let img = upload(device, &props, queue, pool, vk::Format::R8G8B8A8_UNORM, (w, h), &words);
        let frame = Frame { image: img.image, layout: vk::ImageLayout::GENERAL, width: w, height: h, bgr: false, ten_bit: false };
        let stage = [vk::PipelineStageFlags::ALL_COMMANDS];
        for (label, s) in &cases {
            let mut readings = Vec::new();
            for n in 0..220 {
                let sem = sharpen.apply(device, queue, &frame, s, &[]).expect("submitted");
                device.queue_submit(queue, &[vk::SubmitInfo::default().wait_semaphores(&[sem]).wait_dst_stage_mask(&stage)], vk::Fence::null()).unwrap();
                assert!(sharpen.wait_idle(device));
                let r = std::mem::take(&mut sharpen.readings);
                if n >= 20 {
                    readings.extend(r);
                }
            }
            readings.sort_by(f32::total_cmp);
            let median = readings.get(readings.len() / 2).copied().unwrap_or(f32::NAN);
            let p95 = readings.get(readings.len() * 95 / 100).copied().unwrap_or(f32::NAN);
            // The effect: one more pass over the original frame.
            let fresh = upload(device, &props, queue, pool, vk::Format::R8G8B8A8_UNORM, (w, h), &words);
            let f = Frame { image: fresh.image, ..frame };
            let sem = sharpen.apply(device, queue, &f, s, &[]).expect("submitted");
            device.queue_submit(queue, &[vk::SubmitInfo::default().wait_semaphores(&[sem]).wait_dst_stage_mask(&stage)], vk::Fence::null()).unwrap();
            let got = read(device, &props, queue, pool, fresh.image, (w, h));
            let change: f64 = got
                .iter()
                .zip(&words)
                .map(|(&a, &b)| {
                    let (ca, cb) = (channels(a, false), channels(b, false));
                    f64::from((0..3).map(|j| (ca[j] - cb[j]).abs()).sum::<i32>()) / 3.0
                })
                .sum::<f64>()
                / got.len() as f64;
            let lap = laplacian(&luma(&got));
            eprintln!(
                "[sharpen-cost] {label}: gpu_ms median {median:.3} p95 {p95:.3} ({} frames); mean change {change:.2}/255, fine local contrast x{:.3}",
                readings.len(),
                lap / base_lap
            );
            if let (Some(dir), Some(_)) = (&out_dir, &input) {
                let path = std::path::Path::new(dir).join(format!("{label}.png"));
                let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
                let mut enc = png::Encoder::new(file, w, h);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let bytes: Vec<u8> = got.iter().flat_map(|wd| (wd | 0xFF00_0000).to_le_bytes()).collect();
                enc.write_header().unwrap().write_image_data(&bytes).unwrap();
            }
            device.destroy_image(fresh.image, None);
            device.free_memory(fresh.memory, None);
        }
        assert!(sharpen.wait_idle(device));
        device.queue_wait_idle(queue).unwrap();
        sharpen.destroy(device);
        device.destroy_image(img.image, None);
        device.free_memory(img.memory, None);
        device.destroy_command_pool(pool, None);
    }
}
