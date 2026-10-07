//! Native backend: the vendored OpenDLSS-NR host code (`third_party/opendlss-nr`), with its SPIR-V and
//! PTX kernels embedded at build time, behind a small C API (`cpp/nf_native.h`).
//!
//! `build.rs` compiles the kernels with the pinned toolchain from `scripts/fetch-native-tools.sh`, embeds
//! them, and links the C++ (libstdc++ included, statically) into this rlib. x86_64 Linux only.
//!
//! The layer drives it in three steps: [`device_extend`] at `vkCreateDevice` (the network's extensions
//! and features), [`Network::open`] and [`Network::build`] off the game's threads (model, kernels, the
//! graph for one size), then every frame it executes [`Network::graph_commands`] between its own
//! writes of [`Frame::features`] and reads of [`Frame::head`].

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;

use ash::vk;

mod ffi {
    use super::*;

    #[repr(C)]
    pub struct Open {
        pub gipa: vk::PFN_vkGetInstanceProcAddr,
        pub instance: vk::Instance,
        pub physical: vk::PhysicalDevice,
        pub device: vk::Device,
        pub queue_family: u32,
        pub queue_index: u32,
        pub frame_family: u32,
        pub model_dir: *const c_char,
        pub chain: u32,
        pub fence_timeout_ms: u32,
        pub init_dispatchable: Option<InitDispatchable>,
        pub init_user: *mut c_void,
    }

    extern "C" {
        pub fn nf_native_abi_version() -> u32;
        pub fn nf_native_asset_count(kind: *const c_char) -> u32;
        pub fn nf_native_device_supported(
            gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice, missing: *mut c_char, len: usize,
        ) -> u32;
        pub fn nf_native_device_extend(
            gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice, input: *const vk::DeviceCreateInfo,
            out: *mut vk::DeviceCreateInfo, state: *mut *mut c_void, err: *mut c_char, len: usize,
        ) -> u32;
        pub fn nf_native_device_restore(state: *mut c_void);
        pub fn nf_native_open(open: *const Open, err: *mut c_char, len: usize) -> *mut c_void;
        pub fn nf_native_build(n: *mut c_void, width: u32, height: u32, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_frame(n: *const c_void, frame: *mut super::Frame) -> u32;
        pub fn nf_native_graph_commands(n: *const c_void) -> vk::CommandBuffer;
        pub fn nf_native_graph_commands_own(n: *const c_void) -> vk::CommandBuffer;
        pub fn nf_native_record_graph(n: *mut c_void, primary: vk::CommandBuffer, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_chain_timeouts(n: *const c_void, at: *mut c_char, len: usize) -> u32;
        pub fn nf_native_reset_chain_timeouts(n: *mut c_void);
        pub fn nf_native_fall_back_to_barriers(n: *mut c_void, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_close(n: *mut c_void);
    }
}

const MESSAGE: usize = 512;

/// Sets the loader's dispatch pointer on a dispatchable object the network got from below the loader.
pub type InitDispatchable = unsafe extern "C" fn(user: *mut c_void, device: vk::Device, object: *mut c_void);

fn message(buf: &[c_char]) -> String {
    // SAFETY: the C side always NUL-terminates within the buffer (snprintf), and the buffer starts zeroed.
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// Version of the C API this build links (`nf_native_abi_version`).
pub fn abi_version() -> u32 {
    // SAFETY: takes no arguments and touches no state.
    unsafe { ffi::nf_native_abi_version() }
}

/// Kind of an embedded kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    /// A compiled GLSL compute shader.
    Spirv,
    /// A generated PTX kernel (`VK_NV_cuda_kernel_launch`).
    Ptx,
}

/// How many kernels of a kind are embedded.
pub fn asset_count(kind: AssetKind) -> u32 {
    let name = CString::new(match kind {
        AssetKind::Spirv => "spv",
        AssetKind::Ptx => "ptx",
    })
    .expect("no NUL in a literal");
    // SAFETY: `name` is a valid NUL-terminated string for the duration of the call.
    unsafe { ffi::nf_native_asset_count(name.as_ptr()) }
}

/// `Ok` when `physical` has every extension and feature the network needs, else the first missing one.
///
/// # Safety
/// `gipa` is a valid `vkGetInstanceProcAddr` for `instance`, which owns `physical`.
pub unsafe fn device_supported(gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice) -> Result<(), String> {
    let mut missing = [0 as c_char; MESSAGE];
    // SAFETY: forwarded from this function's contract; the buffer is MESSAGE bytes.
    match unsafe { ffi::nf_native_device_supported(gipa, instance, physical, missing.as_mut_ptr(), MESSAGE) } {
        1 => Ok(()),
        _ => Err(message(&missing)),
    }
}

/// The network's additions to a `VkDeviceCreateInfo`, alive until dropped (which puts back any flag it
/// set in the application's own structures). Drop it only after `vkCreateDevice` returned.
pub struct DeviceExtension {
    state: *mut c_void,
    /// The create info to pass on; its pointers stay valid while `self` lives.
    pub info: vk::DeviceCreateInfo,
}

impl Drop for DeviceExtension {
    fn drop(&mut self) {
        // SAFETY: `state` came from nf_native_device_extend and is released exactly once.
        unsafe { ffi::nf_native_device_restore(self.state) }
    }
}

/// Extends `info` with the network's extensions and features ([`DeviceExtension`]).
///
/// # Safety
/// As [`device_supported`]; `info` is a valid create info whose pointers outlive the result.
pub unsafe fn device_extend(
    gipa: vk::PFN_vkGetInstanceProcAddr, instance: vk::Instance, physical: vk::PhysicalDevice, info: &vk::DeviceCreateInfo,
) -> Result<DeviceExtension, String> {
    let mut out = vk::DeviceCreateInfo::default();
    let mut state = std::ptr::null_mut();
    let mut err = [0 as c_char; MESSAGE];
    // SAFETY: forwarded from this function's contract; out-pointers are to locals.
    let ok = unsafe { ffi::nf_native_device_extend(gipa, instance, physical, info, &mut out, &mut state, err.as_mut_ptr(), MESSAGE) };
    if ok == 1 {
        Ok(DeviceExtension { state, info: out })
    } else {
        Err(message(&err))
    }
}

/// What a frame binds: written by `nf_native_frame` (`#[repr(C)]` mirror of `NfNativeFrame`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Frame {
    pub field_width: u32,
    pub field_height: u32,
    /// f32 `[field_height][field_width][16]`, written by the caller before the graph.
    pub features: vk::Buffer,
    pub features_bytes: vk::DeviceSize,
    /// f32 `[field_height][field_width][4]`: the RGB residual and the blend logit.
    pub head: vk::Buffer,
    pub head_bytes: vk::DeviceSize,
    /// `block70.layer0.blend_scale`, clamped to [0, 1].
    pub blend_scale: f32,
    pub chained: u32,
}

/// Where the network runs.
pub struct OpenInfo<'a> {
    pub gipa: vk::PFN_vkGetInstanceProcAddr,
    pub instance: vk::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: vk::Device,
    /// The network's own queue (loading, warm-up): used by nothing else.
    pub queue_family: u32,
    pub queue_index: u32,
    /// The family of the queue the recorded graph is executed on (buffers are shared with it).
    pub frame_family: u32,
    pub model_dir: &'a Path,
    pub chain: bool,
    pub fence_timeout_ms: u32,
    /// Inside a layer: called on the network's queue and every command buffer it allocates.
    pub init_dispatchable: Option<InitDispatchable>,
}

/// The network on one device.
pub struct Network {
    raw: *mut c_void,
}

// SAFETY: the C++ object has no thread affinity; every use goes through `&mut self` or is read-only.
unsafe impl Send for Network {}

impl Network {
    /// Loads and verifies the model and builds the kernels. Seconds.
    ///
    /// # Safety
    /// The handles are live and belong together; `gipa` is the next layer's; the device was created with
    /// [`device_extend`]'s additions; the queue is used by nothing else while any call runs.
    pub unsafe fn open(info: &OpenInfo) -> Result<Self, String> {
        let dir = CString::new(info.model_dir.as_os_str().as_encoded_bytes()).map_err(|_| "a NUL in the model path".to_string())?;
        let open = ffi::Open {
            gipa: info.gipa,
            instance: info.instance,
            physical: info.physical,
            device: info.device,
            queue_family: info.queue_family,
            queue_index: info.queue_index,
            frame_family: info.frame_family,
            model_dir: dir.as_ptr(),
            chain: u32::from(info.chain),
            fence_timeout_ms: info.fence_timeout_ms,
            init_dispatchable: info.init_dispatchable,
            init_user: std::ptr::null_mut(),
        };
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: forwarded from this function's contract.
        let raw = unsafe { ffi::nf_native_open(&open, err.as_mut_ptr(), MESSAGE) };
        if raw.is_null() {
            Err(message(&err))
        } else {
            Ok(Self { raw })
        }
    }

    /// The graph for a `width` x `height` frame, run once (PTX compile, weight copies) and recorded.
    /// Nothing of a previous build may be pending.
    pub fn build(&mut self, width: u32, height: u32) -> Result<Frame, String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live.
        if unsafe { ffi::nf_native_build(self.raw, width, height, err.as_mut_ptr(), MESSAGE) } != 1 {
            return Err(message(&err));
        }
        self.frame().ok_or_else(|| "built, but no frame".to_string())
    }

    pub fn frame(&self) -> Option<Frame> {
        let mut frame = Frame::default();
        // SAFETY: `raw` is live; `frame` is a valid out-pointer.
        (unsafe { ffi::nf_native_frame(self.raw, &mut frame) } == 1).then_some(frame)
    }

    /// The recorded graph (a simultaneous-use secondary of `frame_family`).
    pub fn graph_commands(&self) -> vk::CommandBuffer {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_graph_commands(self.raw) }
    }

    /// The same graph as a secondary of the network's own queue family (`queue_family`). Never run at
    /// the same time as [`Self::graph_commands`]: they share the activations.
    pub fn graph_commands_own(&self) -> vk::CommandBuffer {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_graph_commands_own(self.raw) }
    }

    /// Records the graph straight into `primary` (for measurements; the frame path executes
    /// [`Self::graph_commands`]). No earlier straight recording may be pending.
    pub fn record_graph(&mut self, primary: vk::CommandBuffer) -> Result<(), String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live; `primary` is recording.
        match unsafe { ffi::nf_native_record_graph(self.raw, primary, err.as_mut_ptr(), MESSAGE) } {
            1 => Ok(()),
            _ => Err(message(&err)),
        }
    }

    /// Counter waits that gave up since the last reset, and where the first one was.
    pub fn chain_timeouts(&self) -> (u32, String) {
        let mut at = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live.
        let n = unsafe { ffi::nf_native_chain_timeouts(self.raw, at.as_mut_ptr(), MESSAGE) };
        (n, message(&at))
    }

    pub fn reset_chain_timeouts(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_reset_chain_timeouts(self.raw) }
    }

    /// Rebuilds the graph with barriers between launches (for the rest of the process).
    pub fn fall_back_to_barriers(&mut self) -> Result<Frame, String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live.
        if unsafe { ffi::nf_native_fall_back_to_barriers(self.raw, err.as_mut_ptr(), MESSAGE) } != 1 {
            return Err(message(&err));
        }
        self.frame().ok_or_else(|| "rebuilt, but no frame".to_string())
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        // SAFETY: `raw` came from nf_native_open; the owner guarantees nothing of it is pending.
        unsafe { ffi::nf_native_close(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_1() {
        assert_eq!(abi_version(), 1);
    }

    #[test]
    fn every_kernel_is_embedded() {
        // 13 shaders/*.comp; 78 PTX generator invocations (upstream build_shaders.ps1).
        assert_eq!(asset_count(AssetKind::Spirv), 13);
        assert_eq!(asset_count(AssetKind::Ptx), 78);
    }

    #[test]
    fn unknown_kind_has_no_assets() {
        let other = CString::new("dll").unwrap();
        // SAFETY: valid NUL-terminated string.
        assert_eq!(unsafe { ffi::nf_native_asset_count(other.as_ptr()) }, 0);
    }
}
