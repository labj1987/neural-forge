//! Real-hardware check of the helper's optical-flow path, run under the same Wine/Proton
//! the helper uses: builds the device the way `main.rs::create_vulkan_context` does with
//! an optical-flow queue, then estimates motion on synthetic frames with a known
//! per-frame translation for many frames. Proves the three things the game path needs:
//! the device comes up with the flow queue, the vectors are right, and nothing hangs.
//!
//! Usage (under the helper's runner): optical_flow_rig_check.exe [frames] [width] [height]
//! Exit status: 0 = pass; nonzero = a failed check, or 3 when a single frame stalls >10 s.
//! Every line also goes, flushed, to `optical_flow_rig_check.log` beside the executable:
//! Proton does not reliably pass a Windows program's stdout through.
use ash::vk;
use neural_forge_helper::optical_flow::{FlowQueue, OpticalFlow};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static PROGRESS: AtomicU64 = AtomicU64::new(0);
static LOG: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> = std::sync::OnceLock::new();

macro_rules! say {
    ($($t:tt)*) => {{
        let line = format!($($t)*);
        println!("{line}");
        if let Some(f) = LOG.get() {
            use std::io::Write;
            let mut f = f.lock().unwrap();
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }};
}

fn noise_frame(w: u32, h: u32, shift: u32) -> Vec<u8> {
    let mut f = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let sx = (x + w - shift % w) % w; // content moves right by `shift`
            let n = ((sx / 8).wrapping_mul(747796405) ^ (y / 8).wrapping_mul(2891336453)).wrapping_mul(277803737);
            let v = (n >> 24) as u8;
            let i = ((y * w + x) * 4) as usize;
            f[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    f
}

fn median_motion(v: &[[f32; 2]], w: u32, h: u32) -> (f32, f32) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for y in (64..h - 64).step_by(16) {
        for x in (64..w - 64).step_by(16) {
            let m = v[(y * w + x) as usize];
            xs.push(m[0]);
            ys.push(m[1]);
        }
    }
    xs.sort_by(f32::total_cmp);
    ys.sort_by(f32::total_cmp);
    (xs[xs.len() / 2], ys[ys.len() / 2])
}

fn main() {
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (frames, w, h) = (*args.first().unwrap_or(&3000), *args.get(1).unwrap_or(&1280), *args.get(2).unwrap_or(&720));
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        if let Ok(f) = std::fs::File::create(dir.join("optical_flow_rig_check.log")) {
            let _ = LOG.set(std::sync::Mutex::new(f));
        }
    }

    // Watchdog: a stalled estimate is the failure this exists to catch, so fail loudly.
    std::thread::spawn(|| {
        let mut last = u64::MAX;
        loop {
            std::thread::sleep(Duration::from_secs(10));
            let now = PROGRESS.load(Ordering::Relaxed);
            if now == last {
                say!("STALL: no progress for 10 s at frame {now}");
                std::process::exit(3);
            }
            last = now;
        }
    });

    unsafe {
        let entry = ash::Entry::load().expect("vulkan-1.dll");
        let app = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::builder().application_info(&app), None).expect("instance");
        let pd = instance.enumerate_physical_devices().unwrap().into_iter()
            .find(|&p| instance.get_physical_device_properties(p).vendor_id == 0x10DE).expect("NVIDIA GPU");
        let props = instance.get_physical_device_properties(pd);
        say!("gpu: {:?}", std::ffi::CStr::from_ptr(props.device_name.as_ptr()));

        // The same three conditions `find_flow_family` checks, reported separately.
        let exts: Vec<String> = instance.enumerate_device_extension_properties(pd).unwrap().iter()
            .map(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()).to_string_lossy().into_owned()).collect();
        let has = |n: &str| exts.iter().any(|e| e == n);
        say!("extensions: VK_NV_optical_flow={} VK_KHR_synchronization2={} VK_KHR_format_feature_flags2={}",
            has("VK_NV_optical_flow"), has("VK_KHR_synchronization2"), has("VK_KHR_format_feature_flags2"));
        let mut of = vk::PhysicalDeviceOpticalFlowFeaturesNV::default();
        let mut s2 = vk::PhysicalDeviceSynchronization2Features::default();
        instance.get_physical_device_features2(pd, &mut vk::PhysicalDeviceFeatures2::builder().push_next(&mut of).push_next(&mut s2));
        say!("features: opticalFlow={} synchronization2={}", of.optical_flow, s2.synchronization2);
        let families = instance.get_physical_device_queue_family_properties(pd);
        for (i, f) in families.iter().enumerate() {
            say!("queue family {i}: {:?} x{}", f.queue_flags, f.queue_count);
        }
        let Some(family) = families.iter().position(|p| p.queue_flags.contains(vk::QueueFlags::OPTICAL_FLOW_NV | vk::QueueFlags::TRANSFER)) else {
            say!("FAIL: no queue family with OPTICAL_FLOW_NV | TRANSFER");
            std::process::exit(1);
        };
        let family = family as u32;
        say!("flow queue family: {family}");

        let names: Vec<std::ffi::CString> = ["VK_NV_optical_flow", "VK_KHR_synchronization2", "VK_KHR_format_feature_flags2"]
            .iter().filter(|n| has(n)).map(|n| std::ffi::CString::new(*n).unwrap()).collect();
        let ptrs: Vec<*const std::ffi::c_char> = names.iter().map(|c| c.as_ptr()).collect();
        let mut queues = vec![vk::DeviceQueueCreateInfo::builder().queue_family_index(0).queue_priorities(&[1.0]).build()];
        if family != 0 {
            queues.push(vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&[1.0]).build());
        }
        let mut of_on = vk::PhysicalDeviceOpticalFlowFeaturesNV::builder().optical_flow(true);
        let mut s2_on = vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);
        let info = vk::DeviceCreateInfo::builder().queue_create_infos(&queues).enabled_extension_names(&ptrs)
            .push_next(&mut of_on).push_next(&mut s2_on);
        let device = instance.create_device(pd, &info, None).expect("device with optical-flow queue");
        let fq = FlowQueue { family, queue: device.get_device_queue(family, 0) };
        let mut flow = OpticalFlow::new(&instance, &device, pd, &fq, w, h, 0).expect("optical flow session");
        say!("session: {w}x{h}");

        const STEP: u32 = 4;
        assert!(flow.estimate(&device, &noise_frame(w, h, 0), true).unwrap().is_none(), "first frame has no history");
        let still = flow.estimate(&device, &noise_frame(w, h, 0), true).unwrap().expect("second frame");
        let (sx, sy) = median_motion(&still, w, h);
        say!("stationary median=({sx:.2},{sy:.2}) expected (0,0)");
        let mut worst = 0f32;
        let mut total = Duration::ZERO;
        for i in 1..=frames {
            let frame = noise_frame(w, h, i * STEP);
            let t = Instant::now();
            let v = flow.estimate(&device, &frame, true).unwrap().expect("history present");
            total += t.elapsed();
            let (dx, dy) = median_motion(&v, w, h);
            worst = worst.max((dx + STEP as f32).abs()).max(dy.abs());
            PROGRESS.store(i as u64, Ordering::Relaxed);
            if i % 50 == 0 || i == 1 {
                say!("frame {i}: median=({dx:.2},{dy:.2}) expected (-{STEP},0)");
            }
        }
        say!("frames={frames} avg_estimate={:.2}ms worst_error={worst:.2}px", total.as_secs_f64() * 1000.0 / frames as f64);
        flow.destroy(&device);
        device.destroy_device(None);
        instance.destroy_instance(None);
        let pass = sx.abs() < 0.5 && sy.abs() < 0.5 && worst < 1.0;
        say!("{}", if pass { "PASS" } else { "FAIL: vectors off" });
        std::process::exit(if pass { 0 } else { 1 });
    }
}
