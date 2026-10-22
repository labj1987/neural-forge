//! Native backend: the network's host code (from OpenDLSS-NR, in `third_party/opendlss-nr`), with its SPIR-V and
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
        pub fn nf_native_hold_slots() -> u32;
        pub fn nf_native_device_supported(
            gipa: vk::PFN_vkGetInstanceProcAddr,
            instance: vk::Instance,
            instance_api_version: u32,
            physical: vk::PhysicalDevice,
            missing: *mut c_char,
            len: usize,
        ) -> u32;
        pub fn nf_native_device_extend(
            gipa: vk::PFN_vkGetInstanceProcAddr,
            instance: vk::Instance,
            instance_api_version: u32,
            physical: vk::PhysicalDevice,
            input: *const vk::DeviceCreateInfo,
            out: *mut vk::DeviceCreateInfo,
            state: *mut *mut c_void,
            err: *mut c_char,
            len: usize,
        ) -> u32;
        pub fn nf_native_device_restore(state: *mut c_void);
        pub fn nf_native_open(open: *const Open, stalled: *mut u32, err: *mut c_char, len: usize) -> *mut c_void;
        pub fn nf_native_build(n: *mut c_void, width: u32, height: u32, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_build_sizes(n: *mut c_void, sizes: *const u32, count: u32, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_frame(n: *const c_void, frame: *mut super::Frame) -> u32;
        pub fn nf_native_frame_at(n: *const c_void, slot: u32, frame: *mut super::Frame) -> u32;
        pub fn nf_native_graph_commands(n: *const c_void) -> vk::CommandBuffer;
        pub fn nf_native_graph_commands_own(n: *const c_void) -> vk::CommandBuffer;
        pub fn nf_native_graph_commands_at(n: *const c_void, slot: u32) -> vk::CommandBuffer;
        pub fn nf_native_graph_commands_own_at(n: *const c_void, slot: u32) -> vk::CommandBuffer;
        pub fn nf_native_record_graph(n: *mut c_void, primary: vk::CommandBuffer, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_record_graph_at(n: *mut c_void, slot: u32, hold: u32, primary: vk::CommandBuffer, err: *mut c_char, len: usize) -> u32;
        pub fn nf_native_chain_timeouts(n: *const c_void, at: *mut c_char, len: usize) -> u32;
        pub fn nf_native_reset_chain_timeouts(n: *mut c_void);
        pub fn nf_native_fall_back_to_barriers(n: *mut c_void);
        pub fn nf_native_forget_fall_back();
        pub fn nf_native_fake_chain_timeout(n: *mut c_void, in_warm_up: u32);
        pub fn nf_native_stalled(n: *const c_void) -> u32;
        pub fn nf_native_weight_bytes(n: *const c_void, raw_loaded: *mut u64, raw: *mut u64, matrices: *mut u64);
        pub fn nf_native_close(n: *mut c_void);
    }
}

const MESSAGE: usize = 512;

/// The hold slots [`Network::record_graph_at`] serves (`kHoldPools` in nf_native.cpp, which
/// `nf_native_hold_slots` reports: a test holds the two equal).
pub const HOLD_SLOTS: u32 = 16;

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

/// The hold slots the linked C side serves (`nf_native_hold_slots`): [`HOLD_SLOTS`].
pub fn hold_slots() -> u32 {
    // SAFETY: takes no arguments and touches no state.
    unsafe { ffi::nf_native_hold_slots() }
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

/// `Ok` when `physical` has every extension and feature the network needs and `instance` was created for
/// Vulkan 1.3 or later (`instance_api_version`: the application's `VkApplicationInfo::api_version`, 0 when it
/// gave none), else the first missing one. A device's API version is the lower of its instance's and the
/// GPU's, and the kernels need 1.3's: an older instance is refused, whatever the GPU supports.
///
/// # Safety
/// `gipa` is a valid `vkGetInstanceProcAddr` for `instance`, which owns `physical`.
pub unsafe fn device_supported(
    gipa: vk::PFN_vkGetInstanceProcAddr,
    instance: vk::Instance,
    instance_api_version: u32,
    physical: vk::PhysicalDevice,
) -> Result<(), String> {
    let mut missing = [0 as c_char; MESSAGE];
    // SAFETY: forwarded from this function's contract; the buffer is MESSAGE bytes.
    match unsafe { ffi::nf_native_device_supported(gipa, instance, instance_api_version, physical, missing.as_mut_ptr(), MESSAGE) } {
        1 => Ok(()),
        _ => Err(message(&missing)),
    }
}

/// The network's additions to a `VkDeviceCreateInfo`: a copy of the application's chain with the
/// network's flags set, and what it adds (the application's own structures are only read). Alive
/// until dropped; drop it only after `vkCreateDevice` returned.
pub struct DeviceExtension {
    state: *mut c_void,
    /// The create info to pass on; its pointers stay valid while `self` lives. They point into `state`, not at
    /// anything Rust borrows, hence `'static`: the struct must not outlive `self`.
    pub info: vk::DeviceCreateInfo<'static>,
}

impl Drop for DeviceExtension {
    fn drop(&mut self) {
        // SAFETY: `state` came from nf_native_device_extend and is released exactly once.
        unsafe { ffi::nf_native_device_restore(self.state) }
    }
}

/// Extends `info` with the network's extensions and features ([`DeviceExtension`]). Refused where
/// [`device_supported`] refuses (an instance older than Vulkan 1.3 included).
///
/// # Safety
/// As [`device_supported`]; `info` is a valid create info whose pointers outlive the result.
pub unsafe fn device_extend(
    gipa: vk::PFN_vkGetInstanceProcAddr,
    instance: vk::Instance,
    instance_api_version: u32,
    physical: vk::PhysicalDevice,
    info: &vk::DeviceCreateInfo,
) -> Result<DeviceExtension, String> {
    let mut out = vk::DeviceCreateInfo::default();
    let mut state = std::ptr::null_mut();
    let mut err = [0 as c_char; MESSAGE];
    // SAFETY: forwarded from this function's contract; out-pointers are to locals.
    let ok = unsafe { ffi::nf_native_device_extend(gipa, instance, instance_api_version, physical, info, &mut out, &mut state, err.as_mut_ptr(), MESSAGE) };
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
    /// The post block's `blend_scale` (`block70`, `block30` in the small network), clamped to [0, 1].
    pub blend_scale: f32,
    pub chained: u32,
}

/// [`Network::weight_bytes`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WeightBytes {
    pub raw_loaded: u64,
    pub raw: u64,
    pub matrices: u64,
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

/// Why [`Network::open`] failed.
#[derive(Debug, Clone)]
pub struct OpenError {
    pub message: String,
    /// A wait on the network's queue timed out ([`Network::stalled`]): its work may still run on the
    /// GPU, so the network is dead for this device.
    pub stalled: bool,
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// The network on one device.
pub struct Network {
    raw: *mut c_void,
}

// SAFETY: the C++ object has no thread affinity, and every use goes through `&mut self` or is
// read-only. What it shares with the rest of the process (volk's function table, the chaining switch,
// the asset loader) is process-global, which is safe only because `nf_native_open` admits one live
// network per process: nothing else repoints those while this one is used from another thread.
unsafe impl Send for Network {}

impl Network {
    /// Loads and verifies the model and builds the kernels. Seconds. One network per process at a time:
    /// fails while another is open.
    ///
    /// # Safety
    /// The handles are live and belong together; `gipa` is the next layer's; the device was created with
    /// [`device_extend`]'s additions; the queue is used by nothing else while any call runs. Dropping the
    /// network destroys everything it made (its recordings, activations, weights, pipelines and descriptor
    /// pools): when it is dropped, no submitted work that executes one of its recordings
    /// ([`Self::graph_commands`] and the other `graph_commands*`, or what [`Self::record_graph`] and
    /// [`Self::record_graph_at`] recorded) may still be pending, on any path out of the owner's scope (an
    /// early return or a panic's unwinding included). The handles the safe accessors return carry the same
    /// obligation: submitting one is the caller's unsafe Vulkan call.
    pub unsafe fn open(info: &OpenInfo) -> Result<Self, OpenError> {
        let dir = CString::new(info.model_dir.as_os_str().as_encoded_bytes())
            .map_err(|_| OpenError { message: "a NUL in the model path".to_string(), stalled: false })?;
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
        let mut stalled = 0;
        // SAFETY: forwarded from this function's contract; the out-pointers are to locals.
        let raw = unsafe { ffi::nf_native_open(&open, &mut stalled, err.as_mut_ptr(), MESSAGE) };
        if raw.is_null() {
            Err(OpenError { message: message(&err), stalled: stalled != 0 })
        } else {
            Ok(Self { raw })
        }
    }

    /// The graph for a `width` x `height` frame, run once (PTX compile, weight copies) and recorded. A zero
    /// size is refused with the current build kept. A counter-chain timeout left from the previous build's
    /// frames (not read with [`Self::chain_timeouts`] before), or one in this build's warm-up, switches to
    /// barriers as [`Self::fall_back_to_barriers`] does, and the build records with them
    /// ([`Frame::chained`] 0).
    ///
    /// # Safety
    /// Nothing of a previous build may be pending: no submitted work that executes its graph (either
    /// recording) or a straight [`Self::record_graph`] may still run, since the build frees them.
    pub unsafe fn build(&mut self, width: u32, height: u32) -> Result<Frame, String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live; nothing of the previous build is pending (this function's contract).
        if unsafe { ffi::nf_native_build(self.raw, width, height, err.as_mut_ptr(), MESSAGE) } != 1 {
            return Err(message(&err));
        }
        self.frame().ok_or_else(|| "built, but no frame".to_string())
    }

    /// [`Self::build`] for several frame sizes at once: one graph each (slot `i` for `sizes[i]`, slot 0 what
    /// [`Self::frame`] and [`Self::graph_commands`] return), sharing the weights and the kernels, run one after
    /// another (never at the same time). The frames in `sizes`' order. No sizes, or a zero one, are refused
    /// with the current build kept.
    ///
    /// # Safety
    /// As [`Self::build`].
    pub unsafe fn build_sizes(&mut self, sizes: &[(u32, u32)]) -> Result<Vec<Frame>, String> {
        let mut err = [0 as c_char; MESSAGE];
        let flat: Vec<u32> = sizes.iter().flat_map(|&(w, h)| [w, h]).collect();
        // SAFETY: `raw` is live; `flat` holds `sizes.len()` pairs; nothing of the previous build is pending.
        if unsafe { ffi::nf_native_build_sizes(self.raw, flat.as_ptr(), sizes.len() as u32, err.as_mut_ptr(), MESSAGE) } != 1 {
            return Err(message(&err));
        }
        (0..sizes.len() as u32).map(|i| self.frame_at(i).ok_or_else(|| "built, but no frame".to_string())).collect()
    }

    /// Slot `slot`'s frame ([`Self::build_sizes`]).
    pub fn frame_at(&self, slot: u32) -> Option<Frame> {
        let mut frame = Frame::default();
        // SAFETY: `raw` is live; `frame` is a valid out-pointer.
        (unsafe { ffi::nf_native_frame_at(self.raw, slot, &mut frame) } == 1).then_some(frame)
    }

    /// Slot `slot`'s recorded graph ([`Self::graph_commands`] for slot 0).
    pub fn graph_commands_at(&self, slot: u32) -> vk::CommandBuffer {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_graph_commands_at(self.raw, slot) }
    }

    /// Slot `slot`'s recording for the network's own queue family ([`Self::graph_commands_own`] for slot 0).
    pub fn graph_commands_own_at(&self, slot: u32) -> vk::CommandBuffer {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_graph_commands_own_at(self.raw, slot) }
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
    /// [`Self::graph_commands`]), with descriptor sets from the pool the warm-up uses, which this resets first.
    ///
    /// # Safety
    /// `primary` is a primary command buffer of `frame_family` in the recording state, recorded on this
    /// thread. No command buffer an earlier call recorded may still be submitted or pending (its descriptor
    /// sets are freed here), and none recorded by this one may be when the build is dropped (a rebuild,
    /// [`Self::fall_back_to_barriers`], drop).
    pub unsafe fn record_graph(&mut self, primary: vk::CommandBuffer) -> Result<(), String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live; `primary` is recording and nothing of an earlier call is pending (contract).
        match unsafe { ffi::nf_native_record_graph(self.raw, primary, err.as_mut_ptr(), MESSAGE) } {
            1 => Ok(()),
            _ => Err(message(&err)),
        }
    }

    /// Records slot `slot`'s graph ([`Self::build_sizes`]) straight into `primary`, a primary command buffer of
    /// `frame_family` being recorded (the application's: the hold inside DLSS's command buffer), between two
    /// compute barriers, with descriptor sets from hold slot `hold`'s own pool (`0..HOLD_SLOTS`), which this
    /// resets first. Binds compute pipelines and descriptor sets and pushes constants: the caller restores its
    /// own state afterwards. On an error part of the graph may be recorded.
    ///
    /// # Safety
    /// `primary` is recording on this thread. No command buffer recorded with hold slot `hold` may still be
    /// submitted or pending, and none recorded with this build may be, when the build is dropped (a rebuild,
    /// [`Self::fall_back_to_barriers`], drop).
    pub unsafe fn record_graph_at(&mut self, slot: u32, hold: u32, primary: vk::CommandBuffer) -> Result<(), String> {
        let mut err = [0 as c_char; MESSAGE];
        // SAFETY: `raw` is live; `primary` is recording (contract).
        match unsafe { ffi::nf_native_record_graph_at(self.raw, slot, hold, primary, err.as_mut_ptr(), MESSAGE) } {
            1 => Ok(()),
            _ => Err(message(&err)),
        }
    }

    /// A bounded wait on the network's queue timed out: its work may still run on the GPU, so the
    /// network takes no more work (every build fails) and leaks what it made when dropped instead of
    /// destroying it. Dead for this device.
    pub fn stalled(&self) -> bool {
        // SAFETY: `raw` is live.
        unsafe { ffi::nf_native_stalled(self.raw) != 0 }
    }

    /// GPU bytes of the weights: the records' raw buffers as loaded and now (after the first build only
    /// the scale vectors the kernels read through them), and the matrices built from the records.
    pub fn weight_bytes(&self) -> WeightBytes {
        let mut w = WeightBytes::default();
        // SAFETY: `raw` is live; the out-pointers are to locals.
        unsafe { ffi::nf_native_weight_bytes(self.raw, &mut w.raw_loaded, &mut w.raw, &mut w.matrices) };
        w
    }

    /// Counter waits that gave up since the last reset (no allocation: for every frame's check).
    pub fn chain_timeout_count(&self) -> u32 {
        // SAFETY: `raw` is live; a null `where` of length 0 is written nothing.
        unsafe { ffi::nf_native_chain_timeouts(self.raw, std::ptr::null_mut(), 0) }
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

    /// For tests of the watchdog: as if a counter wait gave up, in the last frame of the graph
    /// (`in_warm_up` false: the count [`Self::chain_timeouts`] reads) or in the next build's first warm-up.
    /// Cleared as a real one is.
    #[doc(hidden)]
    pub fn fake_chain_timeout(&mut self, in_warm_up: bool) {
        // SAFETY: `raw` is live; only host-side counters change.
        unsafe { ffi::nf_native_fake_chain_timeout(self.raw, u32::from(in_warm_up)) }
    }

    /// Switches to barriers between launches (for the rest of the process) and drops the graph recorded
    /// with counter chaining: the next [`Self::build`] records it with barriers, once, at the size it is
    /// asked for.
    ///
    /// # Safety
    /// As [`Self::build`]: nothing of a previous build may be pending.
    pub unsafe fn fall_back_to_barriers(&mut self) {
        // SAFETY: `raw` is live; nothing of the previous build is pending (this function's contract).
        unsafe { ffi::nf_native_fall_back_to_barriers(self.raw) }
    }
}

/// Forgets a [`Network::fall_back_to_barriers`] for networks opened from now on (the live one keeps what it
/// has): for a test of the fall back that shares its process with tests that chain.
pub fn forget_fall_back() {
    // SAFETY: only an atomic store.
    unsafe { ffi::nf_native_forget_fall_back() }
}

impl Drop for Network {
    fn drop(&mut self) {
        // SAFETY: `raw` came from nf_native_open and is closed once; nothing of it is pending at drop
        // ([`Network::open`]'s contract).
        unsafe { ffi::nf_native_close(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_5() {
        assert_eq!(abi_version(), 5);
    }

    /// The layer's slot count and the C++ side's pool count are one number.
    #[test]
    fn hold_slots_match_the_c_side() {
        assert_eq!(HOLD_SLOTS, hold_slots());
    }

    /// No instance function is looked up for this check: a gipa that answers nothing.
    unsafe extern "system" fn no_functions(_: vk::Instance, _: *const c_char) -> vk::PFN_vkVoidFunction {
        None
    }

    /// An instance created for Vulkan 1.0 (0 or 1.0), 1.1 or 1.2 is refused by name, before the device is
    /// looked at; 1.3 and later go on to the device's requirements (here: the first lookup, which fails).
    #[test]
    fn an_instance_older_than_1_3_is_refused() {
        let (instance, physical) = (vk::Instance::null(), vk::PhysicalDevice::null());
        for (api, version) in [(0, "1.0"), (vk::API_VERSION_1_0, "1.0"), (vk::API_VERSION_1_1, "1.1"), (vk::API_VERSION_1_2, "1.2")] {
            // SAFETY: the version is checked first; nothing is called through the handles.
            let why = unsafe { device_supported(no_functions, instance, api, physical) }.unwrap_err();
            assert!(why.contains(&format!("asks for Vulkan {version}")) && why.contains("1.3"), "{why}");
            let info = vk::DeviceCreateInfo::default();
            // SAFETY: as above; `info` is a valid, empty create info.
            let why = unsafe { device_extend(no_functions, instance, api, physical, &info) }.err().expect("refused");
            assert!(why.starts_with("the application's instance asks for Vulkan"), "{why}");
        }
        for api in [vk::API_VERSION_1_3, vk::make_api_version(0, 1, 4, 0), vk::make_api_version(0, 2, 0, 0)] {
            // SAFETY: `no_functions` answers every lookup with null, which the check reports before any call.
            let why = unsafe { device_supported(no_functions, instance, api, physical) }.unwrap_err();
            assert!(why.contains("vkGetPhysicalDeviceFeatures2"), "{why}");
        }
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
