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

/// `NEURAL_FORGE_SKIP_NVAPI` semantics: set to anything that doesn't start with `0`.
fn skip_nvapi() -> bool {
    neural_forge_protocol::env::var("NEURAL_FORGE_SKIP_NVAPI").is_some_and(|v| !v.is_empty() && !v.starts_with('0'))
}

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
    core: *mut c_void,
    snippet_spoof: Option<InstalledSpoof>,
    core_spoof: Option<InstalledSpoof>,

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

    /// `nvapi64.dll`, loaded only so a snippet that resolves NVAPI by name finds this
    /// copy; never called into. Null when the runner supplies NVAPI (`NEURAL_FORGE_SKIP_NVAPI`)
    /// or the load failed.
    nvapi: *mut c_void,

    pub feature: abi::NgxHandle,
    pub disabled: bool,

    /// What the live feature was built with -- the model latches these at creation, so a
    /// difference from what the header asks for now means a rebuild, not a parameter write.
    built_tuning: NgxTuning,
    /// The frame size the live feature was built for; a different incoming size means a rebuild.
    built_size: (u32, u32),
    /// The most recent header value seen; a new value re-arms the settle wait so dragging a
    /// slider debounces rather than rebuilding at every tick.
    last_seen_tuning: NgxTuning,
    tuning_changed: Option<Instant>,
    /// Builds are spaced: NGX creation is expensive and back-to-back creation can exhaust
    /// the driver's latches.
    build_after: Option<Instant>,
    rebuild_failures: u32,
    /// Set by a rebuild so the next evaluate tells the model its history is gone.
    needs_reset: bool,
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
            core: std::ptr::null_mut(),
            snippet_spoof: None,
            core_spoof: None,
            init_ext: None,
            create_feature: None,
            evaluate_feature: None,
            release_feature: None,
            shutdown1: None,
            params: std::ptr::null_mut(),
            params_destroy: None,
            self_params: false,
            device: vk::Device::null(),
            nvapi: std::ptr::null_mut(),
            feature: std::ptr::null_mut(),
            disabled: false,
            built_tuning: NgxTuning::default(),
            built_size: (0, 0),
            last_seen_tuning: NgxTuning::default(),
            tuning_changed: None,
            build_after: None,
            rebuild_failures: 0,
            needs_reset: false,
        }
    }
}

/// # Safety
/// `module`/`name` must be exactly what [`crate::spoof::find_imported_function_slot`]
/// and `GetProcAddress` require.
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
    let mut s = NgxSnippet::default();
    s.device = device;

    let Some(bin_dir) = resolve_bin_dir() else {
        crate::log!("[ngx] NEURAL_FORGE_BIN_DIR not set or nvngx_dlssnr.dll not found there");
        s.disabled = true;
        return s;
    };
    let dll_path = utf16(&format!("{bin_dir}\\nvngx_dlssnr.dll"));
    // SAFETY: `dll_path` is a valid NUL-terminated UTF-16 string.
    s.snippet = unsafe {
        LoadLibraryExW(
            dll_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    if s.snippet.is_null() {
        crate::log!("[ngx] LoadLibraryExW nvngx_dlssnr.dll failed");
        crate::logging::flush();
        s.disabled = true;
        return s;
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
        crate::log!("[ngx] snippet Vulkan exports incomplete");
        crate::logging::flush();
        unsafe { FreeLibrary(s.snippet) };
        s.snippet = std::ptr::null_mut();
        s.disabled = true;
        return s;
    }
    crate::log!("[ngx] snippet Vulkan exports resolved, installing caller-identity spoof next");
    crate::logging::flush();

    // SAFETY: `s.snippet` is a valid, currently-loaded module.
    s.snippet_spoof = unsafe { spoof::install(s.snippet) };
    if s.snippet_spoof.is_none() {
        crate::log!("[ngx] failed to install the caller-identity spoof on the snippet");
        crate::logging::flush();
        s.disabled = true;
        return s;
    }
    crate::log!("[ngx] caller-identity spoof installed, loading core (nvngx.dll) next");
    crate::logging::flush();

    // NVAPI. Never called into -- `nvapi64.dll` is loaded only so a snippet that resolves NVAPI
    // by name finds this copy. When the runner already supplies NVAPI (Proton's DXVK-NVAPI, which
    // the supervisor announces with NEURAL_FORGE_SKIP_NVAPI) it is skipped outright: forcing the
    // vendored copy in bypasses the DXVK-NVAPI override and can fault inside its DllMain. The
    // load is guarded either way, so a bad nvapi64 degrades to "no NVAPI" instead of taking the
    // helper down.
    if skip_nvapi() {
        crate::log!("[ngx] nvapi64.dll load skipped (runner supplies NVAPI)");
    } else {
        let path = utf16(&format!("{bin_dir}\\nvapi64.dll"));
        let (module, seh) = guarded(
            || {
                // SAFETY: `path` is a valid NUL-terminated UTF-16 string.
                unsafe {
                    LoadLibraryExW(
                        path.as_ptr(),
                        std::ptr::null_mut(),
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                    )
                }
            },
            std::ptr::null_mut(),
        );
        s.nvapi = module;
        // SAFETY: a non-null result is a just-loaded PE image (a null one is ignored).
        unsafe { crate::guard::register_module(crate::guard::Module::Nvapi, s.nvapi) };
        if s.nvapi.is_null() {
            crate::log!("[ngx] nvapi64.dll not loaded (seh={seh:#x}); continuing without it");
        } else {
            crate::log!("[ngx] nvapi64.dll loaded");
        }
    }

    // Core (nvngx.dll) must be loaded for the snippet's own Vulkan exports to resolve
    // at all -- confirmed via a real bisection on `lordnikon` (2026-09-11): with Core
    // deliberately kept unloaded, `NVSDK_NGX_VULKAN_AllocateParameters` couldn't be
    // found via `GetProcAddress` on the snippet either ("no AllocateParameters export
    // found on core or snippet", matching an identical, independently-observed note
    // earlier in this same investigation when `nvngx.dll` was genuinely missing from
    // this machine). That means `nvngx_dlssnr.dll`'s own `NVSDK_NGX_VULKAN_*` exports
    // are PE forwarders into Core, not a separate implementation -- there is no real
    // "prefer snippet vs. prefer core" choice to make for this call family; both
    // resolve to the exact same code either way. (Still resolving from `s.snippet`
    // first below, since that's harmless and matches how every export above this one
    // is resolved, but don't mistake it for a meaningful behavior switch.) This also
    // means the `0xbad00002` (`FAIL_PLATFORM_ERROR`) `AllocateParameters` now returns
    // (see the doc comment below on why `VULKAN_Init_ProjectID` isn't the cause) is a
    // real rejection from Core's own NGX runtime, reproduced identically against both
    // a freshly-recreated Wine prefix and the real, untouched, previously-working game
    // prefix -- not an environment/prefix difference, and not fixable by choosing a
    // different module to call through.
    let core_path = utf16(&format!("{bin_dir}\\nvngx.dll"));
    // SAFETY: `core_path` is a valid NUL-terminated UTF-16 string.
    s.core = unsafe {
        LoadLibraryExW(
            core_path.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    // SAFETY: a non-null result is a just-loaded PE image (a null one is ignored).
    unsafe { crate::guard::register_module(crate::guard::Module::Core, s.core) };
    crate::log!("[ngx] core (nvngx.dll) load: {}", if s.core.is_null() { "not found, degrading to snippet allocator" } else { "loaded" });
    crate::logging::flush();
    if !s.core.is_null() {
        // SAFETY: `s.core` just confirmed non-null.
        s.core_spoof = unsafe { spoof::install(s.core) };
        crate::log!("[ngx] core caller-identity spoof install: {}", if s.core_spoof.is_some() { "ok" } else { "FAILED (core will see our real caller identity)" });
        crate::logging::flush();
    }

    // Deliberately NOT calling `NVSDK_NGX_VULKAN_Init_ProjectID`, even though the DLL
    // exports it. Upstream's own reverse-engineered notes (extracted_pipeline_notes.md,
    // section 3.1) show the only *proven* ProjectID-based init route is
    // `NVSDK_NGX_D3D12_Init_with_ProjectID` against a dedicated D3D12 device -- a
    // different API family entirely from this helper's Vulkan device; the Vulkan
    // export exists on the DLL but was never exercised/proven by upstream at all
    // (section 11: "exists" as an export, not a tested route).
    //
    // Real, reproduced evidence this session (2026-09-11, `lordnikon`), not just
    // theory: calling `VULKAN_Init_ProjectID` against Core returned `0xbad00002`
    // (`FAIL_PLATFORM_ERROR`). Removing the call entirely was then tested in
    // isolation -- `AllocateParameters` (see the doc comment above on why it always
    // resolves to Core's own code regardless of which module you ask) still returns
    // the identical `0xbad00002` with `VULKAN_Init_ProjectID` never called at all, so
    // that call was NOT poisoning later calls the way an earlier pass of this
    // investigation first assumed -- it's simply irrelevant to the real, remaining
    // rejection. `VULKAN_Init_Ext` (below, snippet-only, no ProjectID involved) is
    // what every actually-confirmed-working run of this project used (see CLAUDE.md's
    // "First confirmed neural-rendering success") -- restoring that exact sequence,
    // not extending it with an unproven call, is what real evidence supports here.
    // `s.core` stays loaded/spoofed above because it's load-bearing for the snippet's
    // own exports to resolve at all (see above), not because anything calls into it
    // directly.
    if unsafe { resolve_export::<abi::FnVkInitProjectId>(s.snippet, "NVSDK_NGX_VULKAN_Init_ProjectID") }.is_some() {
        crate::log!("[ngx] VULKAN_Init_ProjectID export present but deliberately not called, see ngx.rs doc comment");
        crate::logging::flush();
    }

    // Snippet first, Core only as a fallback -- see the doc comment above `core_path`
    // for why this order, not Core-first, is the one actually proven to work.
    // SAFETY: `s.snippet` is a valid, loaded module.
    let alloc: Option<abi::FnVkAllocateParameters> = unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_AllocateParameters") };
    let alloc = alloc.or_else(|| {
        if s.core.is_null() {
            None
        } else {
            // SAFETY: `s.core` just confirmed non-null.
            unsafe { resolve_export(s.core, "NVSDK_NGX_VULKAN_AllocateParameters") }
        }
    });
    // SAFETY: `s.snippet` is a valid, loaded module.
    s.params_destroy = unsafe { resolve_export(s.snippet, "NVSDK_NGX_VULKAN_DestroyParameters") }.or_else(|| {
        if s.core.is_null() {
            None
        } else {
            // SAFETY: `s.core` just confirmed non-null.
            unsafe { resolve_export(s.core, "NVSDK_NGX_VULKAN_DestroyParameters") }
        }
    });

    // Falls back to a self-implemented `NVSDK_NGX_Parameter` object
    // (`crate::selfparam`) whenever the DLL doesn't export `AllocateParameters` at
    // all, or its real call rejects -- confirmed via a real side-by-side run against
    // upstream's own compiled helper on `lordnikon` (2026-09-11) that this is not a
    // corner case to treat as fatal: upstream hits the identical rejection from
    // Core's own allocator in this exact environment and recovers exactly this way,
    // going on to a real, successful `CreateFeature(18)` afterward. See
    // `selfparam`'s module doc comment for the full evidence.
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
            crate::log!("[ngx] no AllocateParameters export found on core or snippet");
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

    // Round-trip self-test: set a scratch value through the parameter vtable, then
    // read it straight back, before this parameter block is used for anything real.
    // Observed as a real step in a working reference implementation's own log output
    // (run side by side on this machine, never its source -- see this session's
    // investigation) right after its own successful AllocateParameters; this crate
    // never did anything like it. Purely diagnostic for now: logs whether the
    // written/read values match, doesn't gate anything on the result yet.
    {
        let probe_name = CString::new("DLSSNR.SelfTestProbe").unwrap();
        let (test_result, seh) = guarded(
            || {
                // SAFETY: `params` was just validated above as a live, non-null
                // parameter block from a successful `AllocateParameters`.
                unsafe {
                    abi::ngx_set_u32(params, probe_name.as_ptr(), 0x5a5a);
                    let mut readback: u32 = 0;
                    let r = abi::ngx_get_u32(params, probe_name.as_ptr(), &mut readback);
                    (r, readback)
                }
            },
            (abi::result::FAIL_SEH, 0),
        );
        crate::log!(
            "[ngx] params round-trip self-test -> {:#x} seh={:#x} readback={:#x}",
            test_result.0 as u32,
            seh,
            test_result.1
        );
        crate::logging::flush();
    }

    let Some(init_ext) = s.init_ext else {
        crate::log!("[ngx] snippet has no VULKAN_Init_Ext export");
        crate::logging::flush();
        s.disabled = true;
        return s;
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
        s.disabled = true;
        return s;
    }

    // Experimental: `NVSDK_NGX_VULKAN_GetFeatureRequirements` is a real export in the
    // DLL (confirmed via `objdump -p`) that nothing here has ever called. Real NGX
    // integrations call this before `CreateFeature`; skipping it is the leading
    // hypothesis for why `CreateFeature(18)` at a real size hangs indefinitely inside
    // the driver (`libnvidia-glcore.so`, confirmed via gdb) rather than returning or
    // faulting -- plausibly because some internal driver state this call would set up
    // never gets set up. `abi::FnVkGetFeatureRequirements`'s signature is a guess (no
    // `FeatureDiscoveryInfo`-shaped input, unlike NVIDIA's real public NGX SDK) since
    // this is a fictional feature with no spec to check the guess against -- guarded
    // the same as everything else here, purely to observe what it reports/does
    // without gating anything on the result yet.
    if let Some(get_requirements) =
        unsafe { resolve_export::<abi::FnVkGetFeatureRequirements>(s.snippet, "NVSDK_NGX_VULKAN_GetFeatureRequirements") }
    {
        let ((req_result, reqs), seh) = guarded(
            || {
                let mut reqs = abi::NgxFeatureRequirements {
                    version: abi::NgxSdkVersion { major: 0, minor: 0 },
                    feature_flags: 0,
                    min_gpu_mode: 0,
                    in_gpu_mode: 0,
                    min_cs_major_version: 0,
                    min_cs_minor_version: 0,
                };
                // SAFETY: `get_requirements` resolved above from the live snippet
                // module; `instance`/`physical_device` are the caller's own, live
                // handles; `&mut reqs` is a valid out-pointer for the call's duration.
                let r = unsafe { get_requirements(instance, physical_device, &mut reqs) };
                (r, reqs)
            },
            (abi::result::FAIL_SEH, abi::NgxFeatureRequirements {
                version: abi::NgxSdkVersion { major: 0, minor: 0 },
                feature_flags: 0,
                min_gpu_mode: 0,
                in_gpu_mode: 0,
                min_cs_major_version: 0,
                min_cs_minor_version: 0,
            }),
        );
        crate::log!(
            "[ngx] GetFeatureRequirements -> {:#x} seh={:#x} version={}.{} flags={:#x} min_gpu_mode={} in_gpu_mode={} min_cs={}.{}",
            req_result as u32,
            seh,
            reqs.version.major,
            reqs.version.minor,
            reqs.feature_flags,
            reqs.min_gpu_mode,
            reqs.in_gpu_mode,
            reqs.min_cs_major_version,
            reqs.min_cs_minor_version
        );
    } else {
        crate::log!("[ngx] snippet has no VULKAN_GetFeatureRequirements export");
    }

    // Feature creation is deferred to `maintain_feature`, called once the per-frame
    // loop (`main.rs`) knows a real width/height -- there is no real frame to build it
    // at the size of yet at this point in startup.
    s
}

impl NgxSnippet {
    pub fn evaluate_feature_fn(&self) -> Option<abi::FnVkEvaluateFeature> {
        self.evaluate_feature
    }

    pub fn params(&self) -> abi::NgxParameter {
        self.params
    }

    pub fn has_feature(&self) -> bool {
        !self.feature.is_null()
    }
}

fn create_feature_at(s: &mut NgxSnippet, device: &ash::Device, queue: vk::Queue, width: u32, height: u32, tuning: &NgxTuning) -> bool {
    let Some(create_feature) = s.create_feature else { return false };
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
                // Balanced (`NVSDK_NGX_PerfQuality_Value_Balanced` = 3), under both the bare and the
                // prefixed key: the model reads the bare one, and setting only the prefixed one
                // left it at MaxPerf.
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
                abi::ngx_set_u32(params, name("DLSSNR.AutoExposure").as_ptr(), 1);
                abi::ngx_set_f32(params, name("NVSDK_NGX_Parameter_ExposureScale").as_ptr(), 1.0);
                abi::ngx_set_f32(params, name("NVSDK_NGX_Parameter_PreExposure").as_ptr(), 1.0);
                // This helper always captures the swapchain as a plain 8-bit UNORM
                // proxy today (see `neural_forge_layer::capture`) regardless of the
                // swapchain's own HDR-ness -- SDR is the only honest hint to give
                // until the real HDR float16 path (mentioned in the project's own
                // README, not yet implemented) exists.
                abi::ngx_set_u32(params, name("DLSSNR.Hdr").as_ptr(), 0);
                abi::ngx_set_u32(params, name("DLSSNR.SDR").as_ptr(), 1);
                abi::ngx_set_u32(params, name("Width").as_ptr(), width);
                abi::ngx_set_u32(params, name("Height").as_ptr(), height);
                abi::ngx_set_u32(params, name("CreationNodeMask").as_ptr(), 1);
                abi::ngx_set_u32(params, name("VisibilityNodeMask").as_ptr(), 1);
                let flags = abi::feature_flags::DO_SHARPENING | abi::feature_flags::AUTO_EXPOSURE;
                abi::ngx_set_u32(params, name("Feature_Flags").as_ptr(), flags);
            }
        },
        (),
    );
    crate::log!("[ngx] set DLSSNR parameters seh={:#x}", seh);
    if seh != 0 {
        return false;
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
        return false;
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
            return false;
        }
    };
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` was just allocated above, never previously recorded into.
    if unsafe { device.begin_command_buffer(cmd, &begin_info) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to begin the setup command buffer");
        // SAFETY: `pool` owns `cmd`; nothing else references either.
        unsafe { device.destroy_command_pool(pool, None) };
        return false;
    }

    // Last, so nothing above can overwrite it.
    if set_create_tuning(params, tuning) != 0 {
        // SAFETY: `cmd` was begun above and never submitted; `pool` owns it.
        unsafe { device.destroy_command_pool(pool, None) };
        return false;
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
        "[ngx] VULKAN_CreateFeature(18) -> {:#x} seh={:#x} handle={:?} size={width}x{height}",
        result as u32,
        seh,
        handle
    );

    if seh != 0 {
        // A fault during recording leaves `cmd`'s contents unknown/possibly
        // corrupted -- abandon it rather than risk submitting garbage GPU work.
        // SAFETY: `cmd` was never submitted; `pool` owns it and nothing else.
        unsafe { device.destroy_command_pool(pool, None) };
        return false;
    }
    // SAFETY: `cmd` was successfully recorded into above (the guarded call above
    // returned without faulting, regardless of `result`'s own success/failure code --
    // the DLL may still have recorded partial setup work that needs a matching
    // end/submit either way, matching upstream's own "always close+execute" sequence).
    if unsafe { device.end_command_buffer(cmd) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to end the setup command buffer");
        unsafe { device.destroy_command_pool(pool, None) };
        return false;
    }
    let fence_info = vk::FenceCreateInfo::builder();
    // SAFETY: `fence_info` is valid.
    let Ok(fence) = (unsafe { device.create_fence(&fence_info, None) }) else {
        crate::log!("[ngx] CreateFeature: failed to create the setup fence");
        unsafe { device.destroy_command_pool(pool, None) };
        return false;
    };
    let submit = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cmd)).build();
    // SAFETY: `cmd` was just ended above; `queue` is the caller's own, live queue.
    let submitted = unsafe { device.queue_submit(queue, &[submit], fence) }.is_ok();
    let waited = submitted && unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }.is_ok();
    crate::log!("[ngx] CreateFeature: setup command buffer submitted={submitted} waited={waited}");
    // SAFETY: either the fence was just waited on (work complete), or submission
    // itself failed (nothing in flight to wait for) -- both cases make destroying
    // these handles now sound.
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
    }

    if !abi::succeeded(result) || handle.is_null() {
        return false;
    }
    crate::logging::flush();
    s.feature = handle;
    s.built_tuning = *tuning;
    s.built_size = (width, height);
    s.last_seen_tuning = *tuning;
    true
}

/// Releases the live feature, if any. Callers drain the device first (nothing may be
/// in flight against it).
fn release_feature(s: &mut NgxSnippet) {
    if s.feature.is_null() {
        return;
    }
    if let Some(release) = s.release_feature {
        let feature = s.feature;
        let (result, seh) = guarded(|| unsafe { release(feature) }, abi::result::FAIL_SEH);
        crate::log!("[ngx] ReleaseFeature -> {:#x} seh={:#x}", result as u32, seh);
    }
    s.feature = std::ptr::null_mut();
}

/// Consumes the "history is gone" flag a rebuild sets.
pub fn take_needs_reset(s: &mut NgxSnippet) -> bool {
    std::mem::take(&mut s.needs_reset)
}

/// The smallest frame the model is asked to build a feature for. A 1x1 request hung the driver
/// (Xid 109); nothing that small is a real game frame anyway.
pub const MIN_FEATURE_DIM: u32 = 64;

/// Drops the live feature so the next frame rebuilds it from scratch (a header re-initialisation
/// invalidates whatever the feature was built against). The rebuild is not a "first attempt", so
/// a failure retries rather than disabling the model.
pub fn discard_feature(s: &mut NgxSnippet, device: &ash::Device) {
    if !s.has_feature() {
        return;
    }
    // SAFETY: nothing may be in flight when the feature is destroyed.
    let _ = unsafe { device.device_wait_idle() };
    release_feature(s);
    s.build_after = Some(Instant::now());
}

/// Gives up on rebuilds after this many consecutive failures.
const MAX_REBUILD_FAILURES: u32 = 3;

/// Brings the live feature into line with what the header asks for, and builds it the
/// first time. `wanted` is compared by value, not through `tuning_seq` (a hint that a
/// header reset returns to zero, and that persisted config never bumps). A retuned
/// feature keeps answering with the tuning it was built with until the replacement is
/// ready; the replacement is built only after the wanted value has been stable for
/// `settle_ms` and builds are spaced by the same interval.
///
/// Returns whether a feature exists afterwards. A failure to build the *first* feature is
/// one-shot (`disabled`); a failed rebuild leaves no feature and retries after the spacing,
/// up to [`MAX_REBUILD_FAILURES`] times.
pub fn maintain_feature(
    s: &mut NgxSnippet,
    device: &ash::Device,
    queue: vk::Queue,
    width: u32,
    height: u32,
    wanted: NgxTuning,
    settle_ms: u32,
) -> bool {
    if s.disabled {
        return false;
    }
    if width < MIN_FEATURE_DIM || height < MIN_FEATURE_DIM {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::log!("[ngx] refusing feature at {width}x{height}: below the {MIN_FEATURE_DIM}x{MIN_FEATURE_DIM} floor");
        }
        // The live feature (if any) is for a different size and must not answer this frame.
        return false;
    }
    let now = Instant::now();
    let spacing = Duration::from_millis(u64::from(settle_ms));

    if s.has_feature() && s.built_size != (width, height) {
        crate::log!(
            "[helper] frame size {}x{} -> {width}x{height}; rebuilding the feature",
            s.built_size.0, s.built_size.1
        );
        // SAFETY: nothing may be in flight when the feature is destroyed.
        let _ = unsafe { device.device_wait_idle() };
        release_feature(s);
        s.build_after = Some(now + spacing);
    }

    if wanted != s.last_seen_tuning {
        s.last_seen_tuning = wanted;
        s.tuning_changed = Some(now);
    }

    if !s.has_feature() {
        // First build, or a retry after a failed rebuild: wait out the spacing between attempts
        // but not the debounce -- there is no feature answering, so nothing to keep alive.
        if s.build_after.is_some_and(|t| now < t) {
            return false;
        }
        let first = s.rebuild_failures == 0 && s.build_after.is_none();
        let ok = create_feature_at(s, device, queue, width, height, &wanted);
        if ok {
            s.rebuild_failures = 0;
            s.needs_reset = true;
        } else if first {
            // One attempt only: a failed first `CreateFeature` means a real fault or a clean
            // rejection, neither of which retrying next frame fixes (and retrying meant
            // redoing the command pool/buffer/fence setup on every captured frame).
            s.disabled = true;
        } else {
            s.rebuild_failures += 1;
            s.build_after = Some(now + spacing);
            if s.rebuild_failures >= MAX_REBUILD_FAILURES {
                crate::log!("[helper] rebuild failed {} times; giving up on the model", s.rebuild_failures);
                s.disabled = true;
            }
        }
        return ok;
    }

    if wanted == s.built_tuning {
        return true;
    }
    if s.tuning_changed.is_some_and(|t| now.duration_since(t) < spacing) || s.build_after.is_some_and(|t| now < t) {
        return true; // keep answering with the old tuning until the replacement is due
    }
    crate::log!("[helper] retuned; rebuilding the feature (spacing {settle_ms} ms)");
    crate::logging::flush();
    // SAFETY: the helper submits and fences every evaluate, so this only guards against
    // a stray submission; nothing may be in flight when the feature is destroyed.
    let _ = unsafe { device.device_wait_idle() };
    release_feature(s);
    // Marks this as a rebuild, not a first attempt: a failure now retries instead of disabling.
    s.build_after = Some(now + spacing);
    let ok = create_feature_at(s, device, queue, width, height, &wanted);
    if ok {
        s.rebuild_failures = 0;
        s.needs_reset = true;
    } else {
        s.rebuild_failures += 1;
        crate::log!("[helper] feature rebuild failed; retrying after the spacing");
    }
    ok
}

/// Release the feature, shut down the snippet, restore the caller-identity spoof, and
/// unload both modules — in that order, matching upstream's verified teardown
/// sequence.
pub fn teardown(mut s: NgxSnippet) {
    if !s.feature.is_null() {
        if let Some(release) = s.release_feature {
            let feature = s.feature;
            let (result, seh) = guarded(|| unsafe { release(feature) }, abi::result::FAIL_SEH);
            crate::log!("[ngx] ReleaseFeature -> {:#x} seh={:#x}", result as u32, seh);
        }
        s.feature = std::ptr::null_mut();
    }
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
    if let Some(spoofed) = s.core_spoof.take() {
        unsafe { spoof::remove(spoofed) };
    }
    if !s.nvapi.is_null() {
        unsafe { FreeLibrary(s.nvapi) };
    }
    if !s.core.is_null() {
        unsafe { FreeLibrary(s.core) };
    }
    if !s.snippet.is_null() {
        unsafe { FreeLibrary(s.snippet) };
    }
}
