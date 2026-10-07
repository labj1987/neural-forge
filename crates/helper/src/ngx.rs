//! Owns the `nvngx_dlssnr.dll` ("the signed snippet") lifecycle: load, install the
//! caller-identity spoof from `crate::spoof`, initialize NGX, create Feature 18.
//!
//! Ported for shape from `core/ngx_snippet.cpp`'s `NgxLoadAndInit`/`NgxCreatePass`/
//! `NgxTeardown` — the call sequence (which exports to resolve, which parameters to
//! set, in which order) is NVIDIA's own NGX contract, not upstream's expression of it;
//! this is a fresh Rust implementation of walking that contract.
//!
//! Milestone 3 scope (see the plan): load, spoof, init, and `CreateFeature(18)` — the
//! path that actually exercises the caller-identity spoof (NGX checks its caller
//! during init/create, not evaluate) and the SEH guard around a real call into
//! NVIDIA's DLL. `EvaluateFeature` needs bound `DLSSNR.Color`/`Output`/`MVec` Vulkan
//! image resources to mean anything, and building those is naturally paired with
//! milestone 4's composition pass (the same resources that pass reads/writes) — so
//! evaluate's parameter-binding (`NgxSetResources` upstream) is deferred there. The
//! call mechanics are structurally identical to create's (same guarded-call pattern,
//! same vtable parameter setting), so getting create working through the spoof
//! de-risks evaluate too, even unwired.

use std::ffi::{c_void, CString};
use std::time::{Duration, Instant};

use ash::vk;

use crate::abi::{self, NgxParameter};
use crate::guard::guarded;
pub use crate::hdr::FeatureKey;
use crate::selfparam;
use crate::spoof::{self, InstalledSpoof};

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryExW(filename: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
}

const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Generous-but-finite budget for the one `wait_for_fences` in this module that used
/// to pass `u64::MAX` (the setup command buffer in `create_feature_at`). A lost device
/// already returns `VK_ERROR_DEVICE_LOST` rather than hanging, so this was never
/// guarding against that; it was guarding against a driver stall that doesn't lose the
/// device, where the wait simply never returns. Same value and reasoning as
/// `neural_forge_layer`'s own `FENCE_WAIT_TIMEOUT`, ported here from PR #22 against
/// DLSS5VKLayer (bmitch87), commit `4aa730c0` -- see `ATTRIBUTION.md`.
pub(crate) const FENCE_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

fn resolve_bin_dir() -> Option<String> {
    // `std::env::var` already goes through the real `GetEnvironmentVariableW` on this
    // target -- no reason to hand-rolled that call the way `spoof.rs`'s PE parsing
    // genuinely needs to hand-roll PE-specific things.
    let dir = neural_forge_protocol::env::var("NEURAL_FORGE_BIN_DIR")?;
    if dir.is_empty() {
        return None;
    }
    Some(dir)
}

pub struct NgxSnippet {
    snippet: *mut c_void,
    snippet_spoof: Option<InstalledSpoof>,

    init_ext: Option<abi::FnVkInitExt>,
    create_feature: Option<abi::FnVkCreateFeature>,
    evaluate_feature: Option<abi::FnVkEvaluateFeature>,
    release_feature: Option<abi::FnVkReleaseFeature>,
    shutdown1: Option<abi::FnVkShutdown1>,

    params: NgxParameter,
    params_destroy: Option<abi::FnVkDestroyParameters>,
    /// True when `params` came from [`crate::selfparam::allocate`], not a DLL export —
    /// teardown must call [`crate::selfparam::destroy`] instead of `params_destroy` in
    /// that case (there's no real `DestroyParameters` call to make on our own object).
    self_params: bool,

    /// The Vulkan device NGX was initialized against — `Shutdown1` must be called with
    /// this exact device, not a null placeholder.
    device: vk::Device,

    pub disabled: bool,
    /// The NGX binaries are missing or unusable (no binaries folder, no `nvngx_dlssnr.dll` in it,
    /// or a snippet the caller-identity spoof cannot be installed on). Implies `disabled`.
    pub no_binaries: bool,

    /// One NGX feature per pass, in chain order. A slot with a null handle is a hole: a pass
    /// that failed to rebuild and is skipped by the chain until it builds again.
    passes: Vec<PassSlot>,
    /// The frame size and HDR mode every live feature was built for; a different incoming
    /// key (a new size, or the proxy switching between 8-bit and RGBA16F) rebuilds all.
    built: FeatureKey,
    /// The highest pass count the model would actually build at this size, once a later pass has
    /// failed to build. Not a fault: the chain simply runs at what fits.
    ceiling: Option<usize>,
    /// Builds are spaced: NGX creation is expensive and back-to-back creation can exhaust the
    /// driver's latches. `tuning_changed` re-arms on every new header value, so dragging a slider
    /// debounces rather than rebuilding at each tick.
    tuning_changed: Option<Instant>,
    build_after: Option<Instant>,
    /// The header values seen on the previous call, per pass.
    last_seen: Vec<NgxTuning>,
    ever_built: bool,
    /// When pass 0 (the model itself) is built again after it failed: short backoff, one
    /// re-initialisation of NGX, then a long backoff ([`crate::rebuild`]). Never fatal: the
    /// likely causes (VRAM full while the game loads or at 4K with frame generation, a size too
    /// large for the model) pass, and NGX itself can be left refusing every creation until it is
    /// re-initialised. `failing()` is what `model_up` 0 reports.
    retry: crate::rebuild::BuildRetry,
    /// `NEURAL_FORGE_FAIL_CREATE`: creations to fail on purpose (debug; unset does nothing).
    fail_inject: Option<crate::rebuild::FailInject>,
    /// What `VULKAN_Init_Ext` was called with, for a re-initialisation.
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    bin_dir: String,
    /// The DLL's `AllocateParameters`, to replace a DLL-owned parameter block after a
    /// re-initialisation.
    alloc: Option<abi::FnVkAllocateParameters>,
    /// A human-readable account of the latest failure, for the header's reason string. Taken by
    /// the main loop.
    failure_note: Option<String>,
}

/// One live (or holed) pass of the chain.
struct PassSlot {
    handle: abi::NgxHandle,
    /// What this feature was built with; the model latches these at creation, so a difference
    /// from what the header asks for now means a rebuild, not a parameter write.
    built: NgxTuning,
    /// Set on (re)build so the next evaluate tells the model its history is gone.
    needs_reset: bool,
    failures: u32,
}

/// The settings the model latches when a feature is created. Written into the parameter
/// block immediately before `CreateFeature` and at no other time -- writing them at
/// evaluate has no effect on a running feature and leaves the block holding stale values
/// for whatever creates a feature next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NgxTuning {
    pub style: u32,
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// -1 follows local structure; it is not a strength of zero.
    pub skin_structure: f32,
    pub auto_mask: u32,
    pub preset: u32,
}

impl Default for NgxTuning {
    fn default() -> Self {
        Self { style: 0, intensity: 1.0, local_tone: 1.0, local_structure: 1.0, skin_structure: -1.0, auto_mask: 1, preset: 0 }
    }
}

impl From<neural_forge_protocol::PassTuning> for NgxTuning {
    fn from(p: neural_forge_protocol::PassTuning) -> Self {
        Self {
            style: p.style,
            intensity: p.intensity.clamp(0.0, 4.0),
            local_tone: p.local_tone.clamp(0.0, 4.0),
            local_structure: p.local_structure.clamp(0.0, 4.0),
            skin_structure: p.skin_structure.clamp(-1.0, 4.0),
            auto_mask: u32::from(p.auto_mask != 0),
            preset: p.preset,
        }
    }
}

/// Writes the create-time block. Must run immediately before `CreateFeature`.
fn set_create_tuning(params: NgxParameter, t: &NgxTuning) -> u32 {
    let name = |n: &str| CString::new(n).unwrap();
    let ((), seh) = guarded(
        || {
            // SAFETY: `params` was allocated and validated in `load_and_init`.
            unsafe {
                abi::ngx_set_u32(params, name("DLSSNR.Hint.Render.Preset").as_ptr(), t.preset);
                abi::ngx_set_u32(params, name("DLSSNR.Style").as_ptr(), t.style);
                abi::ngx_set_f32(params, name("DLSSNR.Intensity").as_ptr(), t.intensity);
                abi::ngx_set_f32(params, name("DLSSNR.LocalToneStrength").as_ptr(), t.local_tone);
                abi::ngx_set_f32(params, name("DLSSNR.LocalStructureStrength").as_ptr(), t.local_structure);
                abi::ngx_set_f32(params, name("DLSSNR.SkinStructureStrength").as_ptr(), t.skin_structure);
                abi::ngx_set_u32(params, name("DLSSNR.UseAutoMask").as_ptr(), t.auto_mask);
            }
        },
        (),
    );
    if seh != 0 {
        crate::log!("[params] create tuning FAILED (seh={seh:#x})");
    } else {
        crate::log!(
            "[params] create tuning: preset={} style={} intensity={:.2} tone={:.2} structure={:.2} skin={:.2} automask={}",
            t.preset, t.style, t.intensity, t.local_tone, t.local_structure, t.skin_structure, t.auto_mask
        );
    }
    // A one-shot milestone, not the per-frame hot path: flushed so a killed helper still shows it.
    crate::logging::flush();
    seh
}

// SAFETY: every raw pointer/handle field here is either an opaque DLL/NGX handle
// (never dereferenced by this crate as anything other than an opaque token passed
// back to the same DLL) or a function pointer resolved once and never mutated after —
// nothing here assumes exclusive access beyond what `Mutex<NgxSnippet>` at the call
// site already guarantees.
unsafe impl Send for NgxSnippet {}

impl Default for NgxSnippet {
    fn default() -> Self {
        Self {
            snippet: std::ptr::null_mut(),
            snippet_spoof: None,
            init_ext: None,
            create_feature: None,
            evaluate_feature: None,
            release_feature: None,
            shutdown1: None,
            params: std::ptr::null_mut(),
            params_destroy: None,
            self_params: false,
            device: vk::Device::null(),
            disabled: false,
            no_binaries: false,
            passes: Vec::new(),
            built: FeatureKey::default(),
            ceiling: None,
            tuning_changed: None,
            build_after: None,
            last_seen: Vec::new(),
            ever_built: false,
            retry: crate::rebuild::BuildRetry::default(),
            fail_inject: None,
            instance: vk::Instance::null(),
            physical_device: vk::PhysicalDevice::null(),
            bin_dir: String::new(),
            alloc: None,
            failure_note: None,
        }
    }
}

/// # Safety
/// `module` must be a loaded module handle and `F` the function type matching `name`'s real
/// signature.
unsafe fn resolve_export<F: Copy>(module: *mut c_void, name: &str) -> Option<F> {
    let c_name = CString::new(name).ok()?;
    // SAFETY: `module` is a valid, loaded module handle (the caller's contract).
    let p = unsafe { GetProcAddress(module, c_name.as_ptr()) };
    if p.is_null() {
        return None;
    }
    // SAFETY: forwarded from this function's own contract -- `F` must be the type
    // matching `name`'s real signature.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
}

/// Loads `nvngx_dlssnr.dll`, installs the caller-identity spoof, initializes NGX, and
/// creates Feature 18. Every DLL call is wrapped in [`guarded`] — a fault anywhere in
/// here latches `disabled` rather than taking the whole helper down with it.
pub fn load_and_init(instance: vk::Instance, physical_device: vk::PhysicalDevice, device: vk::Device) -> NgxSnippet {
    let mut s = NgxSnippet { device, instance, physical_device, ..NgxSnippet::default() };
    s.fail_inject = neural_forge_protocol::env::var(crate::rebuild::FailInject::ENV).and_then(|v| crate::rebuild::FailInject::parse(&v));
    if let Some(f) = s.fail_inject {
        crate::log!("[ngx] {}: the next feature creations will fail on purpose ({} of them)", crate::rebuild::FailInject::ENV, f.remaining());
    }

    let Some(bin_dir) = resolve_bin_dir() else {
        return s.fail_no_binaries("NEURAL_FORGE_BIN_DIR is not set".to_string());
    };
    s.bin_dir = bin_dir.clone();
    let dll_file = format!("{bin_dir}\\nvngx_dlssnr.dll");
    if !std::path::Path::new(&dll_file).is_file() {
        return s.fail_no_binaries(format!("nvngx_dlssnr.dll not found in {bin_dir}"));
    }
    let dll_path = utf16(&dll_file);
    // SAFETY: `dll_path` is a valid NUL-terminated UTF-16 string.
    s.snippet = unsafe {
        LoadLibraryExW(
            dll_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    if s.snippet.is_null() {
        return s.fail("nvngx_dlssnr.dll failed to load".to_string());
    }
    // SAFETY: `s.snippet` is a just-loaded PE image.
    unsafe { crate::guard::register_module(crate::guard::Module::Snippet, s.snippet) };
    crate::log!("[ngx] nvngx_dlssnr.dll loaded at {:?}", s.snippet);
    crate::logging::flush();

    // SAFETY: `s.snippet` was just confirmed non-null and loaded above.
    unsafe {
        s.init_ext = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_Init_Ext");
        s.create_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_CreateFeature");
        s.evaluate_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_EvaluateFeature");
        s.release_feature = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_ReleaseFeature");
        s.shutdown1 = resolve_export(s.snippet, "NVSDK_NGX_VULKAN_Shutdown1");
    }
    if s.create_feature.is_none() || s.evaluate_feature.is_none() || s.release_feature.is_none() || s.shutdown1.is_none()
    {
        unsafe { FreeLibrary(s.snippet) };
        s.snippet = std::ptr::null_mut();
        return s.fail("nvngx_dlssnr.dll lacks the NGX Vulkan exports".to_string());
    }
    crate::log!("[ngx] snippet Vulkan exports resolved, installing caller-identity spoof next");
    crate::logging::flush();

    // SAFETY: `s.snippet` is a valid, currently-loaded module.
    s.snippet_spoof = unsafe { spoof::install(s.snippet) };
    if s.snippet_spoof.is_none() {
        return s.fail_no_binaries("could not install the caller-identity spoof on nvngx_dlssnr.dll (unexpected binaries)".to_string());
    }
    crate::log!("[ngx] caller-identity spoof installed, resolving NGX exports next");
    crate::logging::flush();

    // `nvngx_dlssnr.dll` is the only NGX binary this helper loads. Any dependency it
    // declares on another module (`nvngx.dll`) is satisfied by the Windows/Wine loader
    // when the snippet itself is loaded above; the helper does not load one itself, and
    // the model initialises and evaluates without one present (confirmed on real
    // hardware). NVAPI is left entirely to the runner (Proton's DXVK-NVAPI); the helper
    // never loads an `nvapi64.dll` of its own.

    // `NVSDK_NGX_VULKAN_Init_ProjectID` is exported but deliberately not called: the only
    // proven ProjectID init route is the D3D12 one, a different API family from this
    // helper's Vulkan device, and `VULKAN_Init_Ext` (below) is what every confirmed-working
    // run used (see CLAUDE.md's "First confirmed neural-rendering success").
    if unsafe { resolve_export::<abi::FnVkInitProjectId>(s.snippet, "NVSDK_NGX_VULKAN_Init_ProjectID") }.is_some() {
        crate::log!("[ngx] VULKAN_Init_ProjectID export present but deliberately not called, see ngx.rs doc comment");
        crate::logging::flush();
    }

    // SAFETY: `s.snippet` is a valid, loaded module.
    let alloc: Option<abi::FnVkAllocateParameters> = unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_AllocateParameters") };
    // SAFETY: `s.snippet` is a valid, loaded module.
    s.params_destroy = unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_DestroyParameters") };

    // Falls back to a self-implemented `NVSDK_NGX_Parameter` object
    // (`crate::selfparam`) whenever the DLL doesn't export `AllocateParameters` at
    // all, or its real call rejects -- not a corner case to treat as fatal: the
    // rejection is recovered exactly this way, going on to a real, successful
    // `CreateFeature(18)` afterward. See `selfparam`'s module doc comment for the
    // full evidence.
    let params = match alloc {
        Some(alloc) => {
            crate::log!("[ngx] calling AllocateParameters now");
            crate::logging::flush();
            let (alloc_result, seh) = guarded(
                || {
                    let mut params: NgxParameter = std::ptr::null_mut();
                    // SAFETY: `alloc` resolved above from a live module; `&mut params`
                    // is a valid out-pointer for the call's duration.
                    let r = unsafe { alloc(&mut params) };
                    (r, params)
                },
                (abi::result::FAIL_SEH, std::ptr::null_mut()),
            );
            let (alloc_code, params) = alloc_result;
            crate::log!("[ngx] AllocateParameters -> {:#x} seh={:#x}", alloc_code as u32, seh);
            crate::logging::flush();
            if abi::succeeded(alloc_code) && !params.is_null() {
                Some(params)
            } else {
                None
            }
        }
        None => {
            crate::log!("[ngx] no AllocateParameters export found on the snippet");
            crate::logging::flush();
            None
        }
    };
    let params = match params {
        Some(params) => params,
        None => {
            crate::log!("[ngx] falling back to a self-implemented NVSDK_NGX_Parameter object");
            crate::logging::flush();
            s.self_params = true;
            selfparam::allocate()
        }
    };
    s.params = params;
    s.alloc = alloc;

    params_self_test(params);

    let Some(init_ext) = s.init_ext else {
        return s.fail("nvngx_dlssnr.dll has no VULKAN_Init_Ext export".to_string());
    };
    crate::log!("[ngx] calling VULKAN_Init_Ext now");
    crate::logging::flush();
    let app_data_path = utf16(&bin_dir);
    let ((init_result,), seh) = guarded(
        || {
            // SAFETY: `init_ext` resolved above; `instance`/`physical_device`/`device`
            // are the caller's own, live Vulkan handles; `app_data_path` is a valid
            // NUL-terminated UTF-16 string kept alive for this call's duration.
            let r = unsafe {
                init_ext(
                    abi::SIGNED_SNIPPET_APPLICATION_ID,
                    app_data_path.as_ptr(),
                    instance,
                    physical_device,
                    device,
                    abi::VERSION_API_14,
                    std::ptr::null(),
                )
            };
            (r,)
        },
        (abi::result::FAIL_SEH,),
    );
    crate::log!("[ngx] VULKAN_Init_Ext -> {:#x} seh={:#x}", init_result as u32, seh);
    crate::logging::flush();
    if !abi::succeeded(init_result) {
        return s.fail(format!("VULKAN_Init_Ext failed ({:#x}, seh={seh:#x})", init_result as u32));
    }

    // Feature creation is deferred to `maintain_feature`, called once the per-frame
    // loop (`main.rs`) knows a real width/height -- there is no real frame to build it
    // at the size of yet at this point in startup.
    s
}

/// Round-trip self-test: sets scratch values of every type the helper writes through the
/// parameter vtable, then reads each straight back, before the block is used for anything real.
/// u32 alone passed under every candidate slot layout; f32, u64 and i32 only pass when the Set
/// and Get halves of the table sit where the DLL's own object has them. Diagnostic: logged, not
/// gated on.
fn params_self_test(params: NgxParameter) {
    let name = CString::new("DLSSNR.SelfTestProbe").unwrap();
    let n = name.as_ptr();
    let (results, seh) = guarded(
        || {
            // SAFETY: `params` is a live parameter block (a successful `AllocateParameters`, or
            // our own object); `n` is a NUL-terminated string that outlives every call.
            unsafe {
                let mut u = 0u32;
                abi::ngx_set_u32(params, n, 0x5a5a);
                let ru = abi::ngx_get_u32(params, n, &mut u);
                let mut f = 0f32;
                abi::ngx_set_f32(params, n, 0.625);
                let rf = abi::ngx_get_f32(params, n, &mut f);
                let mut q = 0u64;
                abi::ngx_set_u64(params, n, 0x1234_5678_9abc);
                let rq = abi::ngx_get_u64(params, n, &mut q);
                let mut i = 0i32;
                abi::ngx_set_i32(params, n, -42);
                let ri = abi::ngx_get_i32(params, n, &mut i);
                [
                    ("u32", ru, abi::succeeded(ru) && u == 0x5a5a),
                    ("f32", rf, abi::succeeded(rf) && f == 0.625),
                    ("u64", rq, abi::succeeded(rq) && q == 0x1234_5678_9abc),
                    ("i32", ri, abi::succeeded(ri) && i == -42),
                ]
            }
        },
        [("u32", abi::result::FAIL_SEH, false), ("f32", abi::result::FAIL_SEH, false), ("u64", abi::result::FAIL_SEH, false), ("i32", abi::result::FAIL_SEH, false)],
    );
    let summary: Vec<String> = results.iter().map(|(t, r, ok)| format!("{t}={}({:#x})", if *ok { "ok" } else { "FAIL" }, *r as u32)).collect();
    let all = seh == 0 && results.iter().all(|r| r.2);
    crate::log!("[ngx] params round-trip self-test: {} seh={seh:#x} -> {}", summary.join(" "), if all { "PASS" } else { "FAIL" });
    crate::logging::flush();
}

impl NgxSnippet {
    /// Ends `load_and_init` with the model off for the session and `note` as the reason.
    fn fail(mut self, note: String) -> Self {
        crate::log!("[ngx] {note}");
        crate::logging::flush();
        self.failure_note = Some(note);
        self.disabled = true;
        self
    }

    /// [`Self::fail`], reported as missing binaries rather than a model failure.
    fn fail_no_binaries(self, note: String) -> Self {
        let mut s = self.fail(note);
        s.no_binaries = true;
        s
    }

    pub fn evaluate_feature_fn(&self) -> Option<abi::FnVkEvaluateFeature> {
        self.evaluate_feature
    }

    pub fn params(&self) -> abi::NgxParameter {
        self.params
    }

    /// Whether the model is unavailable because its first feature failed to build and is waiting
    /// for its next attempt (as opposed to `disabled`, which is fatal for the session).
    pub fn build_failing(&self) -> bool {
        self.retry.failing()
    }

    /// Consecutive failed builds of the model's first feature.
    pub fn failed_builds(&self) -> u32 {
        self.retry.streak()
    }

    /// The length of the failure streak a successful build just ended, once.
    pub fn take_recovered(&mut self) -> Option<u32> {
        self.retry.take_recovered()
    }

    /// The latest failure note, once.
    pub fn take_failure_note(&mut self) -> Option<String> {
        self.failure_note.take()
    }

    /// The number of passes currently built (holes excluded).
    pub fn live_passes(&self) -> usize {
        self.passes.iter().filter(|p| !p.handle.is_null()).count()
    }

    /// The pass count the model would build at, once a later pass failed to (see `ceiling`).
    pub fn pass_ceiling(&self) -> Option<usize> {
        self.ceiling
    }

    /// The built passes' feature handles, in chain order.
    pub fn chain_handles(&self) -> Vec<abi::NgxHandle> {
        self.passes.iter().filter(|p| !p.handle.is_null()).map(|p| p.handle).collect()
    }

    /// Consumes the "history is gone" flag of the pass owning `handle`.
    pub fn take_needs_reset(&mut self, handle: abi::NgxHandle) -> bool {
        self.passes.iter_mut().find(|p| p.handle == handle).is_some_and(|p| std::mem::take(&mut p.needs_reset))
    }
}

/// Builds one feature. `Err` carries a short account of why not (an NGX result code, a Vulkan
/// step that failed), for the header's reason string.
fn create_feature_at(s: &mut NgxSnippet, device: &ash::Device, queue: vk::Queue, key: FeatureKey, tuning: &NgxTuning) -> Result<abi::NgxHandle, String> {
    let create_feature = s.create_feature.ok_or("no CreateFeature export")?;
    let FeatureKey { width, height, hdr: hdr_input } = key;
    if s.fail_inject.as_mut().is_some_and(crate::rebuild::FailInject::should_fail) {
        crate::log!(
            "[ngx] VULKAN_CreateFeature(18) not called: failing on purpose ({}, {} more) size={width}x{height} hdr={}",
            crate::rebuild::FailInject::ENV,
            s.fail_inject.map_or(0, |f| f.remaining()),
            u8::from(hdr_input)
        );
        crate::logging::flush();
        return Err("0xbad00002 (injected)".to_string());
    }
    let flags_env = neural_forge_protocol::env::var(crate::hdr::HDR_FLAGS_ENV).filter(|v| !v.trim().is_empty());
    let (flags, unknown) = crate::hdr::create_flags(hdr_input, flags_env.as_deref());
    let hdr = flags.hdr;
    crate::log!(
        "[ngx] feature {width}x{height} hdr={}: DLSSNR.Hdr={} DLSSNR.SDR={} AutoExposure={}{}",
        u8::from(hdr_input),
        u8::from(hdr),
        u8::from(!hdr),
        u8::from(flags.auto_exposure),
        flags_env.as_deref().map_or_else(String::new, |v| format!(" ({}={v:?}{})", crate::hdr::HDR_FLAGS_ENV, if unknown.is_empty() { String::new() } else { format!(", ignored {unknown:?}") }))
    );
    let name = |n: &str| CString::new(n).unwrap();
    let params = s.params;
    // Guarded like every other real call into the DLL below: `params`'s vtable is a
    // hand-ported ABI shape for a feature this crate has never had real hardware/DLL to
    // test against (see the module doc comment and CLAUDE.md's `ngx.rs` gotchas) --  if
    // a slot is misaligned relative to what the real driver's `nvngx.dll`/
    // `nvngx_dlssnr.dll` actually expects, calling through it faults, and an unguarded
    // fault here takes the whole helper down with no log line at all rather than
    // latching `disabled` the way every other DLL call in this file already does.
    let ((), seh) = guarded(
        || {
            // SAFETY: `params` was allocated and validated in `load_and_init` above.
            unsafe {
                abi::ngx_set_u32(params, name("DLSSNR.Width").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.Height").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.InputWidth").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.InputHeight").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.OutputWidth").as_ptr(), width);
                abi::ngx_set_u32(params, name("DLSSNR.OutputHeight").as_ptr(), height);
                abi::ngx_set_u32(params, name("DLSSNR.Upscaling").as_ptr(), 0);
                abi::ngx_set_f32(params, name("DLSSNR.Scale").as_ptr(), 1.0);
                abi::ngx_set_f32(params, name("DLSSNR.ScalingRatio").as_ptr(), 1.0);
                // 3 is `UltraPerformance` in the public enum (Balanced is 1). The 310.8 feature reads
                // neither key, nor most others in this block: docs/DLSSNR_PARAMETERS.md lists what
                // it does read (the `[params] read` log).
                abi::ngx_set_u32(params, name("PerfQualityValue").as_ptr(), 3);
                abi::ngx_set_u32(params, name("NVSDK_NGX_Parameter_PerfQualityValue").as_ptr(), 3);
                // The rest of this block: real parameter names confirmed present in
                // the DLL's own accepted-parameter string table (`objdump`/`strings`
                // on the actual DLL and on a real, working reference implementation's
                // own compiled helper this session installed and ran side by side --
                // never its source, per the project's own "shape not expression"
                // rule), not previously set here at all. `DLSSNR.Output.Width`/
                // `.Height` (dotted) do NOT exist in the real string table -- an
                // earlier version of this fix added them based on a mismatched
                // third-party reference and has been removed.
                abi::ngx_set_u32(params, name("DLSSNR.Enabled").as_ptr(), 1);
                abi::ngx_set_u32(params, name("DLSSNR.Reset").as_ptr(), 1);
                // Style, Intensity, the local strengths and UseAutoMask are deliberately absent:
                // they are the create-time tuning block, written by `set_create_tuning` right
                // before `CreateFeature` below so nothing later in this function can overwrite them.
                abi::ngx_set_u32(params, name("DLSSNR.AutoExposure").as_ptr(), u32::from(flags.auto_exposure));
                abi::ngx_set_f32(params, name("NVSDK_NGX_Parameter_ExposureScale").as_ptr(), 1.0);
                abi::ngx_set_f32(params, name("NVSDK_NGX_Parameter_PreExposure").as_ptr(), 1.0);
                // What the Color input is: the 8-bit display-referred swapchain proxy (SDR), or
                // the game's scene-linear RGBA16F frame from before its upscaler
                // (`docs/PRE_UPSCALER_DESIGN.md`), whose values run far above 1.0. The model
                // latches this at creation, so the feature key carries it and a change of
                // proxy class rebuilds the feature.
                abi::ngx_set_u32(params, name("DLSSNR.Hdr").as_ptr(), u32::from(hdr));
                abi::ngx_set_u32(params, name("DLSSNR.SDR").as_ptr(), u32::from(!hdr));
                abi::ngx_set_u32(params, name("Width").as_ptr(), width);
                abi::ngx_set_u32(params, name("Height").as_ptr(), height);
                abi::ngx_set_u32(params, name("CreationNodeMask").as_ptr(), 1);
                abi::ngx_set_u32(params, name("VisibilityNodeMask").as_ptr(), 1);
                let feature_flags = abi::feature_flags::DO_SHARPENING | if flags.auto_exposure { abi::feature_flags::AUTO_EXPOSURE } else { 0 };
                abi::ngx_set_u32(params, name("Feature_Flags").as_ptr(), feature_flags);
            }
        },
        (),
    );
    crate::log!("[ngx] set DLSSNR parameters seh={:#x}", seh);
    if seh != 0 {
        return Err(format!("setting the parameters faulted ({seh:#x})"));
    }

    // `NVSDK_NGX_VULKAN_CreateFeature`'s first parameter is a real, currently-
    // recording `VkCommandBuffer` -- the DLL records GPU-side setup work into it, and
    // the caller is responsible for ending/submitting/fence-waiting it afterward
    // (confirmed against the real, non-fictional NGX API shape: `abi::FnVkCreateFeature`
    // already declares this parameter; comparing against the actual upstream project's
    // own documented call sequence -- see this session's investigation -- shows it
    // passing a real command list here, then closing + executing + fence-waiting it,
    // never a null one). Passing `vk::CommandBuffer::null()` here previously was wrong:
    // the DLL then has nothing valid to record the setup work into and never gets
    // anything to wait on, which is consistent with the indefinite hang inside the
    // driver this was fixed after observing (see `guard.rs`/CLAUDE.md history).
    let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(0);
    // SAFETY: `device` is the live device this snippet was initialized against.
    let Ok(pool) = (unsafe { device.create_command_pool(&pool_info, None) }) else {
        crate::log!("[ngx] CreateFeature: failed to create the setup command pool");
        return Err("no setup command pool".to_string());
    };
    let alloc_info =
        vk::CommandBufferAllocateInfo::builder().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
    // SAFETY: `pool` was just created above.
    let cmd = match unsafe { device.allocate_command_buffers(&alloc_info) } {
        Ok(bufs) => bufs[0],
        Err(_) => {
            crate::log!("[ngx] CreateFeature: failed to allocate the setup command buffer");
            // SAFETY: `pool` owns no other resources yet.
            unsafe { device.destroy_command_pool(pool, None) };
            return Err("no setup command buffer".to_string());
        }
    };
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` was just allocated above, never previously recorded into.
    if unsafe { device.begin_command_buffer(cmd, &begin_info) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to begin the setup command buffer");
        // SAFETY: `pool` owns `cmd`; nothing else references either.
        unsafe { device.destroy_command_pool(pool, None) };
        return Err("the setup command buffer would not begin".to_string());
    }

    // Last, so nothing above can overwrite it.
    if set_create_tuning(params, tuning) != 0 {
        // SAFETY: `cmd` was begun above and never submitted; `pool` owns it.
        unsafe { device.destroy_command_pool(pool, None) };
        return Err("setting the tuning faulted".to_string());
    }

    let ((result, handle), seh) = guarded(
        || {
            let mut handle: abi::NgxHandle = std::ptr::null_mut();
            // SAFETY: `create_feature` resolved and validated in `load_and_init`;
            // `cmd` is a valid, currently-recording command buffer (begun just
            // above); `params`/`&mut handle` are valid.
            let r = unsafe { create_feature(cmd, abi::FEATURE_DLSSNR, params, &mut handle) };
            (r, handle)
        },
        (abi::result::FAIL_SEH, std::ptr::null_mut()),
    );
    crate::log!(
        "[ngx] VULKAN_CreateFeature(18) -> {:#x} seh={:#x} handle={:?} size={width}x{height} hdr={}",
        result as u32,
        seh,
        handle,
        u8::from(hdr)
    );

    if seh != 0 {
        // A fault during recording leaves `cmd`'s contents unknown/possibly
        // corrupted -- abandon it rather than risk submitting garbage GPU work.
        // SAFETY: `cmd` was never submitted; `pool` owns it and nothing else.
        unsafe { device.destroy_command_pool(pool, None) };
        return Err(format!("CreateFeature faulted ({seh:#x})"));
    }
    // SAFETY: `cmd` was successfully recorded into above (the guarded call above
    // returned without faulting, regardless of `result`'s own success/failure code --
    // the DLL may still have recorded partial setup work that needs a matching
    // end/submit either way, matching upstream's own "always close+execute" sequence).
    if unsafe { device.end_command_buffer(cmd) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to end the setup command buffer");
        unsafe { device.destroy_command_pool(pool, None) };
        // NGX may have created the feature before the buffer failed to end: release it rather than
        // leak its memory (nothing of it was submitted).
        release_handle(s, handle, 0);
        return Err("the setup command buffer would not end".to_string());
    }
    let fence_info = vk::FenceCreateInfo::builder();
    // SAFETY: `fence_info` is valid.
    let Ok(fence) = (unsafe { device.create_fence(&fence_info, None) }) else {
        crate::log!("[ngx] CreateFeature: failed to create the setup fence");
        unsafe { device.destroy_command_pool(pool, None) };
        release_handle(s, handle, 0);
        return Err("no setup fence".to_string());
    };
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build();
    // SAFETY: `cmd` was just ended above; `queue` is the caller's own, live queue.
    let submitted = unsafe { device.queue_submit(queue, &[submit], fence) }.is_ok();
    // Bounded, not `u64::MAX`: a driver that stalls here without losing the device
    // used to park this thread forever, with nothing to log -- see
    // `FENCE_WAIT_TIMEOUT`'s own doc comment.
    let wait = submitted.then(|| unsafe { device.wait_for_fences(&[fence], true, FENCE_WAIT_TIMEOUT.as_nanos() as u64) });
    let waited = matches!(wait, Some(Ok(())));
    crate::log!("[ngx] CreateFeature: setup command buffer submitted={submitted} waited={waited}");
    if submitted && matches!(wait, Some(Err(vk::Result::TIMEOUT))) {
        // The wait timed out with real work possibly still in flight: destroying the
        // fence or freeing `cmd` (by destroying the pool that owns it) right now would
        // race that work, which is exactly the hazard the old unbounded wait existed
        // to prevent. Leak both -- a one-time, anomalous leak in a process that will
        // usually just retry or eventually exit, not a routine cost -- and let this
        // build fail like any other `CreateFeature` failure below.
        crate::log!("[ngx] CreateFeature: setup fence wait timed out after {FENCE_WAIT_TIMEOUT:?}; abandoning the setup command pool and fence rather than risk freeing in-flight work");
        crate::logging::flush();
        return Err("the setup work did not finish".to_string());
    }
    // SAFETY: either the fence was just waited on successfully (work complete), or
    // submission itself failed (nothing in flight to wait for) -- both cases make
    // destroying these handles now sound. A timeout returns early above instead.
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
    }

    crate::logging::flush();
    if !abi::succeeded(result) || handle.is_null() || !waited {
        // A handle that came back with a failure code, or whose setup work did not run, is not a
        // usable feature; release it so nothing it holds stays allocated across the retries.
        release_handle(s, handle, 0);
        return Err(if abi::succeeded(result) { "the setup work was not submitted".to_string() } else { format!("{:#x}", result as u32) });
    }
    Ok(handle)
}

/// Releases one feature handle. Callers drain the device first (nothing may be in flight).
fn release_handle(s: &NgxSnippet, handle: abi::NgxHandle, pass: usize) {
    if handle.is_null() {
        return;
    }
    if let Some(release) = s.release_feature {
        let (result, seh) = guarded(|| unsafe { release(handle) }, abi::result::FAIL_SEH);
        crate::log!("[ngx] ReleaseFeature pass {pass} -> {:#x} seh={:#x}", result as u32, seh);
    }
}

/// The smallest frame the model is asked to build a feature for. A 1x1 request hung the driver
/// (Xid 109); nothing that small is a real game frame anyway.
pub const MIN_FEATURE_DIM: u32 = 64;

/// Drops every live feature so the next frame rebuilds from scratch (a header re-initialisation
/// invalidates whatever they were built against). Rebuilds after this are retries, not a first
/// attempt, so a failure never disables the model.
pub fn discard_features(s: &mut NgxSnippet, device: &ash::Device) {
    if s.live_passes() == 0 {
        return;
    }
    // SAFETY: nothing may be in flight when a feature is destroyed.
    let _ = unsafe { device.device_wait_idle() };
    release_all(s);
    s.build_after = Some(Instant::now());
}

fn release_all(s: &mut NgxSnippet) {
    let slots = std::mem::take(&mut s.passes);
    for (i, slot) in slots.iter().enumerate() {
        release_handle(s, slot.handle, i);
    }
}

/// Shuts NGX down and initialises it again on the same device (`Shutdown1`, then
/// `VULKAN_Init_Ext` with the arguments `load_and_init` used), after every feature is released.
/// The rig showed NGX refusing every `CreateFeature` with `0xbad00002` after one failure at full
/// VRAM, until the helper restarted, even once the VRAM was free again; a restart is exactly this
/// (and a new process). A DLL-owned parameter block belongs to the instance being shut down, so it
/// is destroyed and allocated again; the self-implemented one is ours and is kept. `false` when
/// the initialisation failed (the next attempt then fails too, and the schedule re-initialises
/// again later).
fn reinit(s: &mut NgxSnippet, device: &ash::Device) -> bool {
    crate::log!("[ngx] re-initialising NGX after {} failed builds in a row (Shutdown1, then VULKAN_Init_Ext)", s.retry.streak());
    crate::logging::flush();
    // SAFETY: nothing may be in flight when a feature is destroyed.
    let _ = unsafe { device.device_wait_idle() };
    release_all(s);
    s.ceiling = None;
    if let Some(shutdown1) = s.shutdown1 {
        let ngx_device = s.device;
        let (result, seh) = guarded(|| unsafe { shutdown1(ngx_device) }, abi::result::FAIL_SEH);
        crate::log!("[ngx] Shutdown1 -> {:#x} seh={:#x}", result as u32, seh);
        if seh != 0 {
            return false;
        }
    }
    if !s.self_params && !s.params.is_null() {
        if let Some(destroy) = s.params_destroy {
            let params = s.params;
            let (result, seh) = guarded(|| unsafe { destroy(params) }, abi::result::FAIL_SEH);
            crate::log!("[ngx] DestroyParameters -> {:#x} seh={:#x}", result as u32, seh);
        }
        s.params = std::ptr::null_mut();
    }
    let Some(init_ext) = s.init_ext else { return false };
    let app_data_path = utf16(&s.bin_dir);
    let (instance, physical_device, ngx_device) = (s.instance, s.physical_device, s.device);
    let ((init_result,), seh) = guarded(
        || {
            // SAFETY: the same live handles and arguments `load_and_init` passed; `app_data_path`
            // outlives the call.
            let r = unsafe {
                init_ext(abi::SIGNED_SNIPPET_APPLICATION_ID, app_data_path.as_ptr(), instance, physical_device, ngx_device, abi::VERSION_API_14, std::ptr::null())
            };
            (r,)
        },
        (abi::result::FAIL_SEH,),
    );
    crate::log!("[ngx] VULKAN_Init_Ext (re-initialisation) -> {:#x} seh={:#x}", init_result as u32, seh);
    if s.params.is_null() {
        let allocated = s.alloc.and_then(|alloc| {
            let ((code, params), seh) = guarded(
                || {
                    let mut params: NgxParameter = std::ptr::null_mut();
                    // SAFETY: `alloc` resolved from a live module; `&mut params` is a valid out-pointer.
                    let r = unsafe { alloc(&mut params) };
                    (r, params)
                },
                (abi::result::FAIL_SEH, std::ptr::null_mut()),
            );
            crate::log!("[ngx] AllocateParameters (re-initialisation) -> {:#x} seh={:#x}", code as u32, seh);
            (abi::succeeded(code) && !params.is_null()).then_some(params)
        });
        s.params = match allocated {
            Some(p) => p,
            None => {
                s.self_params = true;
                selfparam::allocate()
            }
        };
    }
    crate::logging::flush();
    abi::succeeded(init_result)
}

/// Gives up on a pass after this many consecutive failed builds.
const MAX_BUILD_FAILURES: u32 = 3;

/// Brings the chain of features into line with what the header asks for: one feature per
/// wanted pass, each built with that pass's own tuning.
///
/// `wanted` is compared by value, not through `tuning_seq` (a hint that a header reset returns
/// to zero and that persisted config never bumps). A retuned pass keeps answering with the tuning
/// it was built with until its replacement is due, and only that pass is replaced. Builds are
/// spaced by `settle_ms`; a value that keeps changing keeps postponing the rebuild. With a spacing
/// of 0 everything pending is built within this call.
///
/// Pass 0 is the model itself: a failure to build it, first time or rebuild, goes to the retry
/// schedule ([`crate::rebuild::BuildRetry`]: short backoff, one NGX re-initialisation, long
/// backoff), never disables the model, and leaves `build_failing()` true until it builds. A later
/// pass that will not build sets the ceiling instead: the chain runs at what fits. A later pass
/// that fails to *rebuild* is left a hole (skipped) and retried after the spacing.
///
/// `key` is the frame's size and HDR mode (an RGBA16F proxy builds with `DLSSNR.Hdr=1`); a
/// change of either rebuilds every pass, pass 0 at once unless the last build was within the
/// spacing ([`crate::rebuild::after_key_change`]): every request until then is echoed.
///
/// Returns the number of built passes afterwards.
pub fn maintain_passes(
    s: &mut NgxSnippet,
    device: &ash::Device,
    queue: vk::Queue,
    key: FeatureKey,
    wanted: &[NgxTuning],
    settle_ms: u32,
) -> usize {
    let FeatureKey { width, height, .. } = key;
    if s.disabled {
        return 0;
    }
    if let Some(code) = crate::guard::faulted() {
        // Every DLL call is refused from here on (see `guard`'s module doc): the model is gone
        // for this session, not merely unbuilt at this size.
        crate::log!("[ngx] a call into NGX faulted ({code:#x}); the model is off for this session");
        crate::logging::flush();
        s.failure_note = Some(format!("a call into NGX faulted ({code:#x}); restart the helper"));
        s.passes.clear();
        s.disabled = true;
        return 0;
    }
    if width < MIN_FEATURE_DIM || height < MIN_FEATURE_DIM {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[ngx] refusing feature at {width}x{height}: below the {MIN_FEATURE_DIM}x{MIN_FEATURE_DIM} floor");
        }
        // Whatever is live is for a different size and must not answer this frame.
        return 0;
    }
    let spacing = Duration::from_millis(u64::from(settle_ms));

    if s.live_passes() > 0 && s.built != key {
        crate::log!("[helper] frame {} -> {key}; rebuilding {} pass(es)", s.built, s.live_passes());
        // SAFETY: nothing may be in flight when a feature is destroyed.
        let _ = unsafe { device.device_wait_idle() };
        release_all(s);
        s.ceiling = None;
        // Due at once unless the last build was within the spacing (`rebuild::after_key_change`):
        // nothing answers until it is rebuilt.
        s.build_after = crate::rebuild::after_key_change(s.build_after, Instant::now());
    }

    s.last_seen.resize(wanted.len(), NgxTuning::default());
    // One action per call when spaced; every pending action when the spacing is 0.
    for _ in 0..(2 * wanted.len() + 2) {
        let now = Instant::now();
        for (i, w) in wanted.iter().enumerate() {
            if *w != s.last_seen[i] {
                s.last_seen[i] = *w;
                s.tuning_changed = Some(now);
            }
        }
        let target = wanted.len().min(s.ceiling.unwrap_or(usize::MAX)).max(1);

        // Extra passes the header no longer wants.
        if s.passes.len() > target {
            // SAFETY: nothing may be in flight when a feature is destroyed.
            let _ = unsafe { device.device_wait_idle() };
            for i in (target..s.passes.len()).rev() {
                let slot = s.passes.remove(i);
                release_handle(s, slot.handle, i);
            }
            s.build_after = Some(now + spacing);
            continue;
        }

        let due = s.build_after.is_none_or(|t| now >= t);
        let settled = s.tuning_changed.is_none_or(|t| now.duration_since(t) >= spacing);

        // The first feature is never debounced -- nothing is answering, so nothing to keep alive.
        let first_build = s.live_passes() == 0 && !s.ever_built;
        if !first_build && !due {
            break;
        }

        // A pass to (re)build: a hole first, then the first that differs from what it was built
        // with (only once the header value has settled), then the next missing one.
        let hole = s.passes.iter().position(|p| p.handle.is_null());
        let stale = if settled { s.passes.iter().enumerate().position(|(i, p)| !p.handle.is_null() && p.built != wanted[i]) } else { None };
        let missing = (s.passes.len() < target).then_some(s.passes.len());
        let Some(pass) = hole.or(stale).or(missing) else { break };
        // A stale pass keeps answering until its replacement is due, so it waits for the settle;
        // holes and missing passes have nothing to keep alive and go on the spacing alone.
        if Some(pass) == stale && !settled {
            break;
        }

        // Pass 0 is built on the retry schedule once it has failed (the spacing alone retried a
        // failing build every 250 ms for ever).
        let mut reinit_failed = false;
        if pass == 0 {
            match s.retry.step(now) {
                crate::rebuild::Step::Wait(_) => break,
                crate::rebuild::Step::ReinitThenBuild => reinit_failed = !reinit(s, device),
                crate::rebuild::Step::Build => {}
            }
        }

        if pass < s.passes.len() && !s.passes[pass].handle.is_null() {
            crate::log!("[helper] pass {pass} retuned; rebuilding it (spacing {settle_ms} ms)");
            crate::logging::flush();
            // SAFETY: nothing may be in flight when a feature is destroyed.
            let _ = unsafe { device.device_wait_idle() };
            let old = std::mem::replace(&mut s.passes[pass].handle, std::ptr::null_mut());
            release_handle(s, old, pass);
        }

        let created = create_feature_at(s, device, queue, key, &wanted[pass]);
        s.build_after = Some(Instant::now() + spacing);
        match created {
            Ok(handle) => {
                if pass == 0 {
                    s.retry.succeeded();
                }
                s.ever_built = true;
                s.built = key;
                let slot = PassSlot { handle, built: wanted[pass], needs_reset: true, failures: 0 };
                if pass < s.passes.len() {
                    s.passes[pass] = slot;
                } else {
                    s.passes.push(slot);
                }
            }
            Err(why) if pass == 0 => {
                // Not fatal and not retried every frame: the schedule decides when the next attempt
                // is, and whether NGX is re-initialised before it. Nothing answers meanwhile.
                let wait = s.retry.failed(Instant::now());
                let streak = s.retry.streak();
                crate::log!(
                    "[helper] the model would not build at {key} ({why}; {streak} failed in a row{}); passing frames through, next attempt in {:.1}s{}",
                    if reinit_failed { ", NGX re-initialisation failed" } else { "" },
                    wait.as_secs_f32(),
                    if streak == crate::rebuild::BuildRetry::REINIT_AFTER { " after re-initialising NGX" } else { "" }
                );
                crate::logging::flush();
                s.failure_note = Some(format!(
                    "model would not build at {width}x{height} ({why}, {streak} in a row); retrying in {}s{}",
                    wait.as_secs().max(1),
                    if s.retry.reinits() > 0 { " (NGX re-initialised); lower the resolution scale or free VRAM" } else { "" }
                ));
                if pass < s.passes.len() {
                    s.passes[pass].failures += 1;
                }
                return s.live_passes();
            }
            Err(_) if pass >= s.passes.len() => {
                // A later pass would not build: a ceiling, not a fault.
                crate::log!("[helper] pass {pass} would not build; holding the chain at {}", s.passes.len());
                crate::logging::flush();
                s.ceiling = Some(s.passes.len());
            }
            Err(_) => {
                let slot = &mut s.passes[pass];
                slot.failures += 1;
                crate::log!("[helper] pass {pass} rebuild failed ({}); skipping it until it builds", slot.failures);
                if slot.failures >= MAX_BUILD_FAILURES {
                    // Not coming back: drop the tail from here and hold the chain short.
                    s.passes.truncate(pass);
                    s.ceiling = Some(pass);
                }
            }
        }
        if settle_ms != 0 {
            break;
        }
    }
    s.live_passes()
}

/// Release the feature, shut down the snippet, restore the caller-identity spoof, and
/// unload both modules — in that order, matching upstream's verified teardown
/// sequence.
///
/// After a caught fault ([`crate::guard::faulted`]) nothing here calls into or unloads any
/// NVIDIA module: the faulting call may have left their locks held, so a release, a shutdown
/// or a `DllMain` detach could hang. The process is exiting anyway; only our own parameter
/// object is freed.
pub fn teardown(mut s: NgxSnippet) {
    if let Some(code) = crate::guard::faulted() {
        crate::log!("[ngx] teardown skipped after a caught fault ({code:#x}); leaving every NVIDIA module as it is");
        if s.self_params && !s.params.is_null() {
            // SAFETY: allocated by `selfparam::allocate` and never destroyed since. Nothing in the
            // DLLs runs again, so nothing can still read it.
            unsafe { selfparam::destroy(s.params) };
        }
        return;
    }
    release_all(&mut s);
    if let Some(shutdown1) = s.shutdown1 {
        let device = s.device;
        let (result, seh) = guarded(|| unsafe { shutdown1(device) }, abi::result::FAIL_SEH);
        crate::log!("[ngx] Shutdown1 -> {:#x} seh={:#x}", result as u32, seh);
    }
    if !s.params.is_null() {
        if s.self_params {
            // SAFETY: `s.params` was allocated by `selfparam::allocate` and never
            // destroyed since, exactly matching `destroy`'s contract.
            unsafe { selfparam::destroy(s.params) };
            crate::log!("[ngx] destroyed the self-implemented parameter object");
        } else if let Some(destroy) = s.params_destroy {
            let params = s.params;
            let (result, seh) = guarded(|| unsafe { destroy(params) }, abi::result::FAIL_SEH);
            crate::log!("[ngx] DestroyParameters -> {:#x} seh={:#x}", result as u32, seh);
        }
        s.params = std::ptr::null_mut();
    }
    if let Some(spoofed) = s.snippet_spoof.take() {
        // SAFETY: the snippet module is still loaded at this point.
        unsafe { spoof::remove(spoofed) };
    }
    if !s.snippet.is_null() {
        unsafe { FreeLibrary(s.snippet) };
    }
}
