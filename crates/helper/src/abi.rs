//! The NGX Vulkan ABI: `NVSDK_NGX_Parameter`'s vtable layout, the Vulkan export
//! function signatures, result codes, and the Feature 18 (DLSS Neural Rendering)
//! constants.
//!
//! This is NVIDIA's own public-facing ABI shape — the same shape any NGX Vulkan
//! integration has to agree with to call into `nvngx_dlssnr.dll` at all — not
//! upstream's expression of it, so porting it precisely (rather than rederiving it
//! from scratch, which isn't even meaningfully possible for an ABI) is in bounds per
//! the plan's ground rule. Ported for shape from `core/ngx_abi.h`.
//!
//! Every `NVSDK_NGX_Result`-typed value here is a plain `i32`, not a Rust enum: these
//! values come back from a DLL we don't control, and reinterpreting an arbitrary `i32`
//! as a `#[repr(i32)] enum` via transmute is immediate undefined behavior the moment
//! the bytes don't match a declared variant (same reasoning as
//! `neural_forge_protocol::enums`). Check with [`succeeded`] or compare against the named
//! constants in [`result`] instead of matching on a constructed enum.
//!
//! Most of this module's ABI surface (the `NgxResourceVk` family, most of the
//! `result` codes, the parameter get/set helpers beyond what `ngx.rs` currently
//! calls) is genuinely unused right now, not forgotten: `EvaluateFeature`'s real
//! resource binding is milestone 4's job (see `ngx.rs`'s module doc comment), and it's
//! this module's whole ABI contract that milestone needs, not a subset to be
//! rediscovered later. Ported for shape once, used incrementally as the milestones
//! that need each piece land.
#![allow(dead_code)]

use std::ffi::c_void;

use ash::vk;

/// `NVSDK_NGX_Result`. See [`succeeded`] and [`result`].
pub type NgxResult = i32;

pub mod result {
    use super::NgxResult;

    pub const SUCCESS: NgxResult = 0x1;
    pub const FAIL: NgxResult = 0xBAD0_0000u32 as NgxResult;
    pub const FAIL_FAILURE: NgxResult = 0xBAD0_0001u32 as NgxResult;
    /// The caller-identity gate — what a spoofed/rejected caller sees.
    pub const FAIL_PLATFORM_ERROR: NgxResult = 0xBAD0_0002u32 as NgxResult;
    pub const FAIL_INCOMPATIBLE_TYPES: NgxResult = 0xBAD0_0003u32 as NgxResult;
    pub const FAIL_FEATURE_NOT_FOUND: NgxResult = 0xBAD0_0004u32 as NgxResult;
    pub const FAIL_INVALID_PARAMETER: NgxResult = 0xBAD0_0005u32 as NgxResult;
    pub const FAIL_SCRATCH_BUFFER_TOO_SMALL: NgxResult = 0xBAD0_0006u32 as NgxResult;
    pub const FAIL_NOT_INITIALIZED: NgxResult = 0xBAD0_0007u32 as NgxResult;
    pub const FAIL_UNSUPPORTED_INPUT_FORMAT: NgxResult = 0xBAD0_0008u32 as NgxResult;
    pub const FAIL_RW_FLAG_MISSING: NgxResult = 0xBAD0_0009u32 as NgxResult;
    pub const FAIL_MISSING_INPUT: NgxResult = 0xBAD0_000Au32 as NgxResult;
    /// Core's `CreateFeature(18)` without going through the signed-snippet route.
    pub const FAIL_UNABLE_TO_INITIALIZE_FEATURE: NgxResult = 0xBAD0_000Bu32 as NgxResult;
    pub const FAIL_OUT_OF_DATE: NgxResult = 0xBAD0_000Cu32 as NgxResult;
    pub const FAIL_OUT_OF_GPU_MEMORY: NgxResult = 0xBAD0_000Du32 as NgxResult;
    pub const FAIL_UNSUPPORTED_FORMAT: NgxResult = 0xBAD0_000Eu32 as NgxResult;
    pub const FAIL_UNABLE_TO_WRITE_TO_APP_DATA_PATH: NgxResult = 0xBAD0_000Fu32 as NgxResult;
    pub const FAIL_UNSUPPORTED_PARAMETER: NgxResult = 0xBAD0_0010u32 as NgxResult;
    pub const FAIL_DENIED: NgxResult = 0xBAD0_0011u32 as NgxResult;
    pub const FAIL_NOT_IMPLEMENTED: NgxResult = 0xBAD0_0012u32 as NgxResult;
    /// Not one of NVIDIA's own codes — this crate's own sentinel for "the call faulted
    /// and [`crate::guard::guarded`] caught it", mirroring upstream's identical use of
    /// this same bit pattern for the same purpose.
    pub const FAIL_SEH: NgxResult = 0x8BAD_F00Du32 as NgxResult;
}

/// `NVSDK_NGX_SUCCEED(r)`: true for [`result::SUCCESS`] and false for every `FAIL_*`
/// code, including ones not named above — NVIDIA reserves the whole `0xBADxxxxx` range
/// for failure, so this checks the range rather than enumerating every member of it.
pub fn succeeded(r: NgxResult) -> bool {
    (r as u32 & 0xFFF0_0000) != 0xBAD0_0000
}

pub const VERSION_API_13: u32 = 0x13;
pub const VERSION_API_14: u32 = 0x14;

/// Feature 18: DLSS Neural Rendering (DLSS 5 NR). Not in any public `NVSDK_NGX_Feature`
/// enum as of older SDKs — this is the one hard-coded feature ID this whole crate
/// exists to call.
pub const FEATURE_DLSSNR: i32 = 18;

pub const SIGNED_SNIPPET_APPLICATION_ID: u64 = 0x0876_232C;

pub mod feature_flags {
    pub const IS_HDR: u32 = 0x1;
    pub const DEPTH_INVERTED: u32 = 0x2;
    pub const DO_SHARPENING: u32 = 0x4;
    pub const AUTO_EXPOSURE: u32 = 0x8;
    pub const MV_LOW_RES: u32 = 0x10;
    pub const MV_JITTERED: u32 = 0x20;
    pub const RESET_RENDER_PROFILE: u32 = 0x100;
}

/// Opaque; NGX hands one of these back from `CreateFeature` and expects it back
/// unchanged for `EvaluateFeature`/`ReleaseFeature`. Never dereferenced by this crate.
pub type NgxHandle = *mut c_void;

// ---------------------------------------------------------------------------------
// NVSDK_NGX_Resource_VK — the canonical Vulkan resource layout NGX expects for
// DLSSNR.Color/Output/MVec/Depth.
// ---------------------------------------------------------------------------------

pub const RESOURCE_VK_TYPE_IMAGE_VIEW: i32 = 0;
pub const RESOURCE_VK_TYPE_BUFFER: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxImageViewInfoVk {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxBufferInfoVk {
    pub buffer: vk::Buffer,
    pub size_in_bytes: u32,
}

#[repr(C)]
#[derive(Copy)]
pub union NgxResourceVkPayload {
    pub image_view_info: NgxImageViewInfoVk,
    pub buffer_info: NgxBufferInfoVk,
}

// Written manually rather than derived: `Clone`'s derive macro generates code that
// calls `.clone()` on the active field, which is meaningless for a union (there is no
// single "active field" to know about). A plain bitwise copy is exactly right here
// since the type is `Copy`.
impl Clone for NgxResourceVkPayload {
    fn clone(&self) -> Self {
        *self
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxResourceVk {
    pub resource: NgxResourceVkPayload,
    pub resource_type: i32,
    pub read_write: bool,
}

impl NgxResourceVk {
    pub fn from_image_view(info: NgxImageViewInfoVk, read_write: bool) -> Self {
        Self {
            resource: NgxResourceVkPayload { image_view_info: info },
            resource_type: RESOURCE_VK_TYPE_IMAGE_VIEW,
            read_write,
        }
    }
}

// ---------------------------------------------------------------------------------
// NVSDK_NGX_Parameter — a C++ abstract interface (pure vtable), reached only via a
// pointer NGX itself hands us from AllocateParameters. Slots 9/10/13 are reserved
// padding in the real interface; they're named here so the slot numbering (and
// therefore every other slot's offset) stays correct, not because anything calls them.
// ---------------------------------------------------------------------------------

pub type CGetResult = unsafe extern "system" fn(*mut c_void, *const i8, *mut c_void) -> NgxResult;

#[repr(C)]
pub struct NgxParameterVtable {
    pub set_ptr: unsafe extern "system" fn(*mut c_void, *const i8, *mut c_void),
    pub set_u64: unsafe extern "system" fn(*mut c_void, *const i8, u64),
    pub set_f32: unsafe extern "system" fn(*mut c_void, *const i8, f32),
    pub set_f64: unsafe extern "system" fn(*mut c_void, *const i8, f64),
    pub set_u32: unsafe extern "system" fn(*mut c_void, *const i8, u32),
    pub set_i32: unsafe extern "system" fn(*mut c_void, *const i8, i32),
    pub get_f64: unsafe extern "system" fn(*mut c_void, *const i8, *mut f64) -> NgxResult,
    pub get_u64: unsafe extern "system" fn(*mut c_void, *const i8, *mut u64) -> NgxResult,
    pub get_ptr: unsafe extern "system" fn(*mut c_void, *const i8, *mut *mut c_void) -> NgxResult,
    pub get_reserved9: CGetResult,
    pub get_reserved10: CGetResult,
    pub get_i32: unsafe extern "system" fn(*mut c_void, *const i8, *mut i32) -> NgxResult,
    pub get_u32: unsafe extern "system" fn(*mut c_void, *const i8, *mut u32) -> NgxResult,
    pub get_reserved13: CGetResult,
    pub get_f32: unsafe extern "system" fn(*mut c_void, *const i8, *mut f32) -> NgxResult,
    pub reset: unsafe extern "system" fn(*mut c_void),
}

/// The object layout: a C++ object with no data members of its own, just the
/// compiler-inserted vtable pointer every polymorphic C++ object starts with. This is
/// the same "pointer to a pointer-to-function-table" shape as a COM interface, and for
/// the same reason it's reliably interoperable between MSVC (whatever built
/// `nvngx_dlssnr.dll`) and this mingw-built helper: neither compiler is free to lay out
/// a single, non-inherited vtable any other way on this platform.
#[repr(C)]
pub struct NgxParameterObj {
    pub vtable: *const NgxParameterVtable,
}

pub type NgxParameter = *mut NgxParameterObj;

macro_rules! ngx_param_setter {
    ($fn_name:ident, $slot:ident, $value_ty:ty) => {
        /// # Safety
        /// `param` must be a valid, live `NgxParameter` (from `AllocateParameters` or
        /// equivalent) and `name` a valid NUL-terminated C string.
        pub unsafe fn $fn_name(param: NgxParameter, name: *const i8, value: $value_ty) {
            let vtable = unsafe { &*(*param).vtable };
            unsafe { (vtable.$slot)(param.cast(), name, value) }
        }
    };
}

macro_rules! ngx_param_getter {
    ($fn_name:ident, $slot:ident, $value_ty:ty) => {
        /// # Safety
        /// Same contract as the setters above; `out` must be a valid pointer to write
        /// through.
        pub unsafe fn $fn_name(param: NgxParameter, name: *const i8, out: *mut $value_ty) -> NgxResult {
            let vtable = unsafe { &*(*param).vtable };
            unsafe { (vtable.$slot)(param.cast(), name, out) }
        }
    };
}

ngx_param_setter!(ngx_set_u64, set_u64, u64);
ngx_param_setter!(ngx_set_f32, set_f32, f32);
ngx_param_setter!(ngx_set_u32, set_u32, u32);
ngx_param_setter!(ngx_set_i32, set_i32, i32);
ngx_param_getter!(ngx_get_u32, get_u32, u32);
ngx_param_getter!(ngx_get_f32, get_f32, f32);

/// # Safety
/// `param` must be a valid, live `NgxParameter`; `name` a valid NUL-terminated C
/// string; `value` a pointer that outlives the parameter block's next use (NGX reads
/// it lazily, not necessarily at `Set` time) — for an [`NgxResourceVk`], that means the
/// struct it points to must not move or be dropped while the parameter block might
/// still read it.
pub unsafe fn ngx_set_ptr(param: NgxParameter, name: *const i8, value: *mut c_void) {
    let vtable = unsafe { &*(*param).vtable };
    unsafe { (vtable.set_ptr)(param.cast(), name, value) }
}

// ---------------------------------------------------------------------------------
// Vulkan export function signatures — resolved by name from `nvngx_dlssnr.dll`
// (the "signed snippet") and `nvngx.dll` ("core").
// ---------------------------------------------------------------------------------

/// `FeatureInfo`, always passed as `nullptr` here — its layout is otherwise opaque and
/// never constructed by this crate.
pub type FnVkInitExt = unsafe extern "system" fn(
    app_id: u64,
    app_data_path: *const u16,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    sdk_version: u32,
    feature_info: *const c_void,
) -> NgxResult;

pub type FnVkCreateFeature = unsafe extern "system" fn(
    cmd: vk::CommandBuffer,
    feature: i32,
    parameters: NgxParameter,
    handle: *mut NgxHandle,
) -> NgxResult;

/// The progress callback is always passed as `nullptr` here (this crate never streams
/// progress out of an evaluate call).
pub type FnVkEvaluateFeature = unsafe extern "system" fn(
    cmd: vk::CommandBuffer,
    handle: NgxHandle,
    parameters: NgxParameter,
    progress_callback: *const c_void,
) -> NgxResult;

pub type FnVkReleaseFeature = unsafe extern "system" fn(handle: NgxHandle) -> NgxResult;
pub type FnVkShutdown1 = unsafe extern "system" fn(device: vk::Device) -> NgxResult;

pub type FnVkAllocateParameters = unsafe extern "system" fn(parameters: *mut NgxParameter) -> NgxResult;
pub type FnVkDestroyParameters = unsafe extern "system" fn(parameters: NgxParameter) -> NgxResult;

/// `NVSDK_NGX_ENGINE_TYPE_CUSTOM` -- the only variant this crate ever needs (no game
/// engine to identify as), value confirmed against NVIDIA's own public
/// `NVIDIA/DLSS` GitHub repo (`include/nvsdk_ngx_defs.h`).
pub const ENGINE_TYPE_CUSTOM: i32 = 0;

/// `NVSDK_NGX_VULKAN_Init_ProjectID` -- the real, standard NGX Vulkan bootstrap for
/// any non-Unreal/Unity integration (`NVSDK_NGX_ENGINE_TYPE_CUSTOM`), confirmed
/// exported by the real `nvngx.dll` core on this machine (`strings`, 2026-09-10) and
/// distinct from [`FnVkInitExt`]/`NVSDK_NGX_VULKAN_Init_Ext`, the signed-snippet
/// route's own simplified (numeric `ApplicationId`, no `ProjectId`/`EngineType`)
/// entry point this crate used exclusively before. Signature confirmed against six
/// independent real-world callers (OptiScaler and its forks, all agreeing) rather
/// than from this crate's own guesswork -- `PFN_vkGetInstanceProcAddr`/
/// `PFN_vkGetDeviceProcAddr` in particular are real parameters `FnVkInitExt` omits
/// entirely, not something to improvise the layout of.
pub type FnVkInitProjectId = unsafe extern "system" fn(
    project_id: *const i8,
    engine_type: i32,
    engine_version: *const i8,
    app_data_path: *const u16,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    device: vk::Device,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    sdk_version: u32,
    feature_info: *const c_void,
) -> NgxResult;
