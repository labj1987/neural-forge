//! The watched NGX exports: creation, evaluation and release of a feature, for D3D11, D3D12 and Vulkan. Each reads
//! what it needs through the parameter object NGX itself hands out, writes a record, and calls NVIDIA's function
//! with the caller's arguments.

use core::ffi::{c_char, c_void};
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::ring::{Record, API_D3D11, API_D3D12, API_VULKAN};
use crate::win::{log, logging, publish, target};

/// NGX's `NVSDK_NGX_Parameter` (an MSVC C++ object): its getters' vtable slots. MSVC groups overloads and lays
/// them out in reverse declaration order (measured: docs/PROTON_NGX.md).
#[repr(C)]
pub struct Parameter {
    vtbl: *const *const c_void,
}
const GET_VOID_POINTER: usize = 8;
const GET_D3D12_RESOURCE: usize = 9;
const GET_D3D11_RESOURCE: usize = 10;
const GET_I: usize = 11;
const GET_UI: usize = 12;
const GET_F: usize = 14;
const NGX_OK: u32 = 1;

type Getter = unsafe extern "C" fn(*mut Parameter, *const c_char, *mut c_void) -> u32;

unsafe fn get<T: Default>(p: *mut Parameter, slot: usize, key: &str) -> Option<T> {
    if p.is_null() {
        return None;
    }
    let mut name = [0u8; 96];
    let n = key.len().min(name.len() - 1);
    name[..n].copy_from_slice(&key.as_bytes()[..n]);
    let mut value = T::default();
    let f: Getter = core::mem::transmute(*(*p).vtbl.add(slot));
    (f(p, name.as_ptr() as *const c_char, &mut value as *mut T as *mut c_void) == NGX_OK).then_some(value)
}

unsafe fn get_f(p: *mut Parameter, key: &str) -> Option<f32> {
    get::<f32>(p, GET_F, key)
}

unsafe fn get_i(p: *mut Parameter, key: &str) -> Option<i32> {
    get::<i32>(p, GET_I, key)
}

unsafe fn get_ui(p: *mut Parameter, key: &str) -> Option<u32> {
    get::<u32>(p, GET_UI, key)
}

unsafe fn get_ptr(p: *mut Parameter, slot: usize, key: &str) -> *mut c_void {
    get::<usize>(p, slot, key).unwrap_or(0) as *mut c_void
}

/// `NVSDK_NGX_Handle`.
#[repr(C)]
pub struct NgxHandle {
    id: u32,
}

/// One image as the record names it.
#[derive(Clone, Copy, Default)]
struct Image {
    vk: u64,
    width: u32,
    height: u32,
    format: i32,
    layout: i32,
}

type Com = *mut *const *const c_void;

unsafe fn method<F: Copy>(obj: Com, slot: usize) -> F {
    core::mem::transmute_copy(&*(**obj).add(slot))
}

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

/// vkd3d-proton's `ID3D12DXVKInteropDevice1` and `ID3D12DXVKInteropDevice`.
const IID_INTEROP1: Guid = Guid(0x902d8115, 0x59eb, 0x4406, [0x95, 0x18, 0xfe, 0x00, 0xf9, 0x91, 0xee, 0x65]);
const IID_INTEROP: Guid = Guid(0x39da4e09, 0xbd1c, 0x4198, [0x9f, 0xae, 0x86, 0xbb, 0xe3, 0xbe, 0x41, 0xfd]);
/// DXVK's `IDXGIVkInteropSurface`.
const IID_DXGI_VK_INTEROP_SURFACE: Guid = Guid(0x5546cf8c, 0x77e7, 0x4341, [0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d]);

/// The interop device, found once (vkd3d-proton has one per process for a game's device).
struct Interop {
    device: usize,
    v1: bool,
}
static INTEROP: Mutex<Option<Option<Interop>>> = Mutex::new(None);

#[repr(C)]
#[derive(Default)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth: u16,
    mips: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

/// An `ID3D12Resource`'s VkImage, extent, VkFormat and the layout it has in `NON_PIXEL_SHADER_RESOURCE`.
unsafe fn d3d12_image(res: Com) -> Image {
    let mut img = Image { format: -1, layout: -1, ..Image::default() };
    if res.is_null() {
        return img;
    }
    let mut desc = ResourceDesc::default();
    let get_desc: unsafe extern "system" fn(Com, *mut ResourceDesc) -> *mut ResourceDesc = method(res, 10);
    get_desc(res, &mut desc);
    img.width = desc.width as u32;
    img.height = desc.height;
    let mut interop = INTEROP.lock().unwrap_or_else(|e| e.into_inner());
    if interop.is_none() {
        let get_device: unsafe extern "system" fn(Com, *const Guid, *mut usize) -> i32 = method(res, 7);
        let mut dev = 0usize;
        let found = if get_device(res, &IID_INTEROP1, &mut dev) == 0 && dev != 0 {
            Some(Interop { device: dev, v1: true })
        } else if get_device(res, &IID_INTEROP, &mut dev) == 0 && dev != 0 {
            Some(Interop { device: dev, v1: false })
        } else {
            None
        };
        log(&format!("interop: ID3D12DXVKInteropDevice{}", found.as_ref().map_or(" not available", |i| if i.v1 { "1" } else { "" })));
        *interop = Some(found);
    }
    if let Some(Some(i)) = interop.as_ref() {
        let dev = i.device as Com;
        let mut offset = 0u64;
        if i.v1 {
            let info1: unsafe extern "system" fn(Com, Com, *mut u64, *mut u64, *mut i32) -> i32 = method(dev, 13);
            info1(dev, res, &mut img.vk, &mut offset, &mut img.format);
        } else {
            let info: unsafe extern "system" fn(Com, Com, *mut u64, *mut u64) -> i32 = method(dev, 10);
            info(dev, res, &mut img.vk, &mut offset);
        }
        // D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE: the state DLSS takes its inputs in.
        let layout: unsafe extern "system" fn(Com, Com, u32, *mut i32) = method(dev, 9);
        layout(dev, res, 0x40, &mut img.layout);
    }
    img
}

#[repr(C)]
struct VkImageCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    image_type: i32,
    format: i32,
    extent: [u32; 3],
    mip_levels: u32,
    array_layers: u32,
    samples: u32,
    tiling: i32,
    usage: u32,
    sharing_mode: i32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
    initial_layout: i32,
}

/// An `ID3D11Resource`'s VkImage through DXVK's `IDXGIVkInteropSurface::GetVulkanImageInfo`.
unsafe fn d3d11_image(res: Com) -> Image {
    let mut img = Image { format: -1, layout: -1, ..Image::default() };
    if res.is_null() {
        return img;
    }
    let qi: unsafe extern "system" fn(Com, *const Guid, *mut usize) -> i32 = method(res, 0);
    let mut surface = 0usize;
    if qi(res, &IID_DXGI_VK_INTEROP_SURFACE, &mut surface) != 0 || surface == 0 {
        return img;
    }
    let surface = surface as Com;
    let mut info: VkImageCreateInfo = core::mem::zeroed();
    info.s_type = 14; // VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO
    let image_info: unsafe extern "system" fn(Com, *mut u64, *mut i32, *mut VkImageCreateInfo) -> i32 = method(surface, 4);
    image_info(surface, &mut img.vk, &mut img.layout, &mut info);
    img.width = info.extent[0];
    img.height = info.extent[1];
    img.format = info.format;
    let release: unsafe extern "system" fn(Com) -> u32 = method(surface, 2);
    release(surface);
    img
}

/// `NVSDK_NGX_Resource_VK` (image form).
#[repr(C)]
struct ResourceVk {
    view: u64,
    image: u64,
    aspect: u32,
    base_mip: u32,
    mips: u32,
    base_layer: u32,
    layers: u32,
    format: i32,
    width: u32,
    height: u32,
    kind: i32,
    read_write: u8,
}

unsafe fn vk_image(p: *mut Parameter, key: &str) -> Option<Image> {
    let r = get_ptr(p, GET_VOID_POINTER, key) as *const ResourceVk;
    if r.is_null() || (*r).kind != 0 {
        return None;
    }
    Some(Image { vk: (*r).image, width: (*r).width, height: (*r).height, format: (*r).format, layout: -1 })
}

#[derive(Clone, Copy, PartialEq)]
enum Api {
    D3D11,
    D3D12,
    Vulkan,
}

impl Api {
    fn code(self) -> u32 {
        match self {
            Api::D3D11 => API_D3D11,
            Api::D3D12 => API_D3D12,
            Api::Vulkan => API_VULKAN,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Api::D3D11 => "d3d11",
            Api::D3D12 => "d3d12",
            Api::Vulkan => "vulkan",
        }
    }
}

unsafe fn image(api: Api, p: *mut Parameter, key: &str) -> Option<Image> {
    match api {
        Api::Vulkan => vk_image(p, key),
        Api::D3D12 => Some(get_ptr(p, GET_D3D12_RESOURCE, key)).filter(|r| !r.is_null()).map(|r| d3d12_image(r as Com)),
        Api::D3D11 => Some(get_ptr(p, GET_D3D11_RESOURCE, key)).filter(|r| !r.is_null()).map(|r| d3d11_image(r as Com)),
    }
}

/// A created feature, told apart by its handle's address (NGX's ids repeat across features).
#[derive(Clone, Copy)]
struct Feature {
    handle: usize,
    id: u32,
    feature: u32,
    flags: u32,
    evaluates: u64,
}
static FEATURES: Mutex<Vec<Feature>> = Mutex::new(Vec::new());

fn feature_name(f: u32) -> &'static str {
    match f {
        1 => "SR",
        11 => "FG",
        13 => "RR",
        _ => "?",
    }
}

/// The names the text log lists beside the five the record carries.
const LOGGED_IMAGES: &[&str] = &["TransparencyMask", "DLSSG.Backbuffer", "DLSSG.HUDLess", "DLSSG.UI", "DLSSG.OutputInterpolated", "DLSSG.OutputReal"];

unsafe fn create(api: Api, feature: i32, p: *mut Parameter, out: *mut *mut NgxHandle, result: u32) {
    let flags = get_i(p, "DLSS.Feature.Create.Flags").unwrap_or(0) as u32;
    let handle = if out.is_null() { core::ptr::null_mut() } else { *out };
    let id = if handle.is_null() { 0 } else { (*handle).id };
    if !handle.is_null() {
        let mut list = FEATURES.lock().unwrap_or_else(|e| e.into_inner());
        list.retain(|f| f.handle != handle as usize);
        list.push(Feature { handle: handle as usize, id, feature: feature as u32, flags, evaluates: 0 });
    }
    let size = |a: &str, b: &str| (get_ui(p, a).unwrap_or(0), get_ui(p, b).unwrap_or(0));
    let (render, output) = (size("Width", "Height"), size("OutWidth", "OutHeight"));
    publish(Record {
        api: api.code(),
        handle: id,
        feature: feature as u32,
        create_flags: flags,
        render: [render.0, render.1],
        output_extent: [output.0, output.1],
        ..Record::default()
    });
    log(&format!(
        "CREATE {} feature={feature}({}) result={result:#x} h={id}@{handle:p} {}x{} -> {}x{} flags={flags:#x}",
        api.name(),
        feature_name(feature as u32),
        render.0,
        render.1,
        output.0,
        output.1
    ));
}

unsafe fn evaluate(api: Api, handle: *const NgxHandle, p: *mut Parameter) {
    let found = {
        let mut list = FEATURES.lock().unwrap_or_else(|e| e.into_inner());
        list.iter_mut().find(|f| f.handle == handle as usize).map(|f| {
            f.evaluates += 1;
            *f
        })
    };
    let mut rec = Record {
        api: api.code(),
        handle: if handle.is_null() { 0 } else { (*handle).id },
        feature: found.map_or(0, |f| f.feature),
        create_flags: found.map_or(0, |f| f.flags),
        ..Record::default()
    };
    let mut line = String::new();
    let mut take = |key: &str| -> Option<Image> {
        let img = image(api, p, key)?;
        if logging() {
            let _ = write!(line, " {key}=vk:{:#x}({}x{} fmt={} lay={})", img.vk, img.width, img.height, img.format, img.layout);
        }
        Some(img)
    };
    if let Some(i) = take("Color") {
        rec.colour = i.vk;
        rec.colour_extent = [i.width, i.height];
    }
    rec.mvec = take("MotionVectors").map_or(0, |i| i.vk);
    rec.depth = take("Depth").map_or(0, |i| i.vk);
    if let Some(i) = take("Output") {
        rec.output = i.vk;
        rec.output_extent = [i.width, i.height];
    }
    rec.exposure = take("ExposureTexture").map_or(0, |i| i.vk);
    if logging() {
        for key in LOGGED_IMAGES {
            take(key);
        }
    }
    rec.jitter = [get_f(p, "Jitter.Offset.X").unwrap_or(0.0), get_f(p, "Jitter.Offset.Y").unwrap_or(0.0)];
    rec.mv_scale = [get_f(p, "MV.Scale.X").unwrap_or(0.0), get_f(p, "MV.Scale.Y").unwrap_or(0.0)];
    rec.pre_exposure = get_f(p, "DLSS.Pre.Exposure").unwrap_or(0.0);
    rec.exposure_scale = get_f(p, "DLSS.Exposure.Scale").unwrap_or(0.0);
    rec.reset = get_i(p, "Reset").unwrap_or(0);
    rec.render = [get_ui(p, "DLSS.Render.Subrect.Dimensions.Width").unwrap_or(0), get_ui(p, "DLSS.Render.Subrect.Dimensions.Height").unwrap_or(0)];
    publish(rec);
    if logging() {
        log(&format!(
            "EVAL {} h={}@{handle:p} feat={}{line} jitter={},{} mv_scale={},{} pre_exposure={} exposure_scale={} reset={} render={}x{} flags={:#x}",
            api.name(),
            rec.handle,
            feature_name(rec.feature),
            rec.jitter[0],
            rec.jitter[1],
            rec.mv_scale[0],
            rec.mv_scale[1],
            rec.pre_exposure,
            rec.exposure_scale,
            rec.reset,
            rec.render[0],
            rec.render[1],
            rec.create_flags
        ));
    }
}

unsafe fn release(api: Api, handle: *mut NgxHandle) {
    let gone = {
        let mut list = FEATURES.lock().unwrap_or_else(|e| e.into_inner());
        let f = list.iter().find(|f| f.handle == handle as usize).copied();
        list.retain(|f| f.handle != handle as usize);
        f
    };
    log(&format!(
        "RELEASE {} h={} feat={} evaluates={}",
        api.name(),
        gone.map_or(0, |f| f.id),
        gone.map_or("?", |f| feature_name(f.feature)),
        gone.map_or(0, |f| f.evaluates)
    ));
}

type CreateFn = unsafe extern "C" fn(*mut c_void, i32, *mut Parameter, *mut *mut NgxHandle) -> u32;
type Create1Fn = unsafe extern "C" fn(*mut c_void, *mut c_void, i32, *mut Parameter, *mut *mut NgxHandle) -> u32;
type EvaluateFn = unsafe extern "C" fn(*mut c_void, *const NgxHandle, *mut Parameter, *mut c_void) -> u32;
type ReleaseFn = unsafe extern "C" fn(*mut NgxHandle) -> u32;

unsafe fn real<F: Copy>(name: &str) -> F {
    let p = target(name);
    core::mem::transmute_copy(&p)
}

macro_rules! hooks {
    ($api:expr, $create:ident, $evaluate:ident, $release:ident) => {
        /// # Safety
        /// NGX's contract.
        #[no_mangle]
        pub unsafe extern "C" fn $create(cmd: *mut c_void, feature: i32, p: *mut Parameter, out: *mut *mut NgxHandle) -> u32 {
            let r = real::<CreateFn>(stringify!($create))(cmd, feature, p, out);
            create($api, feature, p, out, r);
            r
        }
        /// # Safety
        /// NGX's contract.
        #[no_mangle]
        pub unsafe extern "C" fn $evaluate(cmd: *mut c_void, handle: *const NgxHandle, p: *mut Parameter, callback: *mut c_void) -> u32 {
            evaluate($api, handle, p);
            real::<EvaluateFn>(stringify!($evaluate))(cmd, handle, p, callback)
        }
        /// # Safety
        /// NGX's contract.
        #[no_mangle]
        pub unsafe extern "C" fn $release(handle: *mut NgxHandle) -> u32 {
            release($api, handle);
            real::<ReleaseFn>(stringify!($release))(handle)
        }
    };
}

hooks!(Api::D3D11, NVSDK_NGX_D3D11_CreateFeature, NVSDK_NGX_D3D11_EvaluateFeature, NVSDK_NGX_D3D11_ReleaseFeature);
hooks!(Api::D3D12, NVSDK_NGX_D3D12_CreateFeature, NVSDK_NGX_D3D12_EvaluateFeature, NVSDK_NGX_D3D12_ReleaseFeature);
hooks!(Api::Vulkan, NVSDK_NGX_VULKAN_CreateFeature, NVSDK_NGX_VULKAN_EvaluateFeature, NVSDK_NGX_VULKAN_ReleaseFeature);

/// # Safety
/// NGX's contract.
#[no_mangle]
pub unsafe extern "C" fn NVSDK_NGX_VULKAN_CreateFeature1(
    device: *mut c_void,
    cmd: *mut c_void,
    feature: i32,
    p: *mut Parameter,
    out: *mut *mut NgxHandle,
) -> u32 {
    let r = real::<Create1Fn>("NVSDK_NGX_VULKAN_CreateFeature1")(device, cmd, feature, p, out);
    create(Api::Vulkan, feature, p, out, r);
    r
}
