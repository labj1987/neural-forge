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

/// Generous-but-finite budget for the one `wait_for_fences` in this module that used
/// to pass `u64::MAX` (the setup command buffer in `create_feature_at`). A lost device
/// already returns `VK_ERROR_DEVICE_LOST` rather than hanging, so this was never
/// guarding against that; it was guarding against a driver stall that doesn't lose the
/// device, where the wait simply never returns. Same value and reasoning as
/// `neural_forge_layer`'s own `FENCE_WAIT_TIMEOUT`, ported here from PR #22 against
/// DLSS5VKLayer (bmitch87), commit `4aa730c0` -- see `ATTRIBUTION.md`.
pub(crate) const FENCE_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

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

    pub disabled: bool,
    /// The NGX binaries are missing or unusable (no binaries folder, no `nvngx_dlssnr.dll` in it,
    /// or a snippet the caller-identity spoof cannot be installed on). Implies `disabled`.
    pub no_binaries: bool,

    /// One NGX feature per pass, in chain order. A slot with a null handle is a hole: a pass
    /// that failed to rebuild and is skipped by the chain until it builds again.
    passes: Vec<PassSlot>,
    /// The frame size every live feature was built for; a different incoming size rebuilds all.
    built_size: (u32, u32),
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
    /// The first feature failed to build at this size (and when). Not fatal: the model is
    /// unavailable *at that size* and is retried when the size changes, or after
    /// [`RETRY_FAILED_SIZE_AFTER`] -- the likely causes (not enough VRAM while the game is still
    /// loading, a size too large for the model) pass. Failing once used to disable the model for
    /// the rest of the session, so lowering the resolution scale afterwards did nothing.
    create_failed: Option<((u32, u32), Instant)>,
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
            disabled: false,
            no_binaries: false,
            passes: Vec::new(),
            built_size: (0, 0),
            ceiling: None,
            tuning_changed: None,
            build_after: None,
            last_seen: Vec::new(),
            ever_built: false,
            create_failed: None,
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
    let mut s = NgxSnippet { device, ..NgxSnippet::default() };

    let Some(bin_dir) = resolve_bin_dir() else {
        return s.fail_no_binaries("NEURAL_FORGE_BIN_DIR is not set".to_string());
    };
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
    crate::log!("[ngx] caller-identity spoof installed, loading core (nvngx.dll) next");
    crate::logging::flush();

    // NVAPI. Never called into -- `nvapi64.dll` is loaded only so a snippet that resolves NVAPI
    // by name finds this copy. When the runner already supplies NVAPI (Proton's DXVK-NVAPI, which
    // the supervisor announces with NEURAL_FORGE_SKIP_NVAPI) it is skipped outright: forcing the
    // vendored copy in bypasses the DXVK-NVAPI override and can fault inside its DllMain. The
    // load is deliberately not guarded: a fault inside a DllMain leaves the loader lock held, so
    // jumping out of it would only turn the crash into a deadlock at the next LoadLibrary.
    if skip_nvapi() {
        crate::log!("[ngx] nvapi64.dll load skipped (runner supplies NVAPI)");
    } else {
        let path = utf16(&format!("{bin_dir}\\nvapi64.dll"));
        // SAFETY: `path` is a valid NUL-terminated UTF-16 string.
        s.nvapi = unsafe {
            LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
        };
        if s.nvapi.is_null() {
            // Not in the binaries folder: the normal search path, which under system Wine finds the
            // DXVK-NVAPI copy the supervisor installed in the prefix (with its native override).
            let by_name = utf16("nvapi64.dll");
            // SAFETY: `by_name` is a valid NUL-terminated UTF-16 string.
            s.nvapi = unsafe { LoadLibraryExW(by_name.as_ptr(), std::ptr::null_mut(), 0) };
        }
        // SAFETY: a non-null result is a just-loaded PE image (a null one is ignored).
        unsafe { crate::guard::register_module(crate::guard::Module::Nvapi, s.nvapi) };
        if s.nvapi.is_null() {
            crate::log!("[ngx] nvapi64.dll not loaded; continuing without it");
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

    /// Whether the model is unavailable at the current size because its first feature failed to
    /// build (as opposed to `disabled`, which is fatal for the session).
    pub fn create_failed(&self) -> bool {
        self.create_failed.is_some()
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

fn create_feature_at(s: &mut NgxSnippet, device: &ash::Device, queue: vk::Queue, width: u32, height: u32, tuning: &NgxTuning) -> Option<abi::NgxHandle> {
    let create_feature = s.create_feature?;
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
        return None;
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
        return None;
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
            return None;
        }
    };
    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` was just allocated above, never previously recorded into.
    if unsafe { device.begin_command_buffer(cmd, &begin_info) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to begin the setup command buffer");
        // SAFETY: `pool` owns `cmd`; nothing else references either.
        unsafe { device.destroy_command_pool(pool, None) };
        return None;
    }

    // Last, so nothing above can overwrite it.
    if set_create_tuning(params, tuning) != 0 {
        // SAFETY: `cmd` was begun above and never submitted; `pool` owns it.
        unsafe { device.destroy_command_pool(pool, None) };
        return None;
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
        return None;
    }
    // SAFETY: `cmd` was successfully recorded into above (the guarded call above
    // returned without faulting, regardless of `result`'s own success/failure code --
    // the DLL may still have recorded partial setup work that needs a matching
    // end/submit either way, matching upstream's own "always close+execute" sequence).
    if unsafe { device.end_command_buffer(cmd) }.is_err() {
        crate::log!("[ngx] CreateFeature: failed to end the setup command buffer");
        unsafe { device.destroy_command_pool(pool, None) };
        return None;
    }
    let fence_info = vk::FenceCreateInfo::builder();
    // SAFETY: `fence_info` is valid.
    let Ok(fence) = (unsafe { device.create_fence(&fence_info, None) }) else {
        crate::log!("[ngx] CreateFeature: failed to create the setup fence");
        unsafe { device.destroy_command_pool(pool, None) };
        return None;
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
        return None;
    }
    // SAFETY: either the fence was just waited on successfully (work complete), or
    // submission itself failed (nothing in flight to wait for) -- both cases make
    // destroying these handles now sound. A timeout returns early above instead.
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
    }

    crate::logging::flush();
    if !abi::succeeded(result) || handle.is_null() {
        return None;
    }
    Some(handle)
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

/// How long a failed first build at one size blocks retrying that same size.
const RETRY_FAILED_SIZE_AFTER: Duration = Duration::from_secs(30);

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
/// A failure to build the very first feature is one-shot (`disabled`). A later pass that will not
/// build sets the ceiling instead: the chain runs at what fits. A pass that fails to *rebuild* is
/// left a hole (skipped) and retried after the spacing.
///
/// Returns the number of built passes afterwards.
pub fn maintain_passes(
    s: &mut NgxSnippet,
    device: &ash::Device,
    queue: vk::Queue,
    width: u32,
    height: u32,
    wanted: &[NgxTuning],
    settle_ms: u32,
) -> usize {
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

    if let Some((size, when)) = s.create_failed {
        if size == (width, height) && when.elapsed() < RETRY_FAILED_SIZE_AFTER {
            return 0; // still blocked at this size; the frame just passes through
        }
        // A different size, or long enough: try again from scratch.
        s.create_failed = None;
        s.ever_built = false;
        s.build_after = None;
    }

    if s.live_passes() > 0 && s.built_size != (width, height) {
        crate::log!(
            "[helper] frame size {}x{} -> {width}x{height}; rebuilding {} pass(es)",
            s.built_size.0, s.built_size.1, s.live_passes()
        );
        // SAFETY: nothing may be in flight when a feature is destroyed.
        let _ = unsafe { device.device_wait_idle() };
        release_all(s);
        s.ceiling = None;
        s.build_after = Some(Instant::now() + spacing);
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

        if pass < s.passes.len() && !s.passes[pass].handle.is_null() {
            crate::log!("[helper] pass {pass} retuned; rebuilding it (spacing {settle_ms} ms)");
            crate::logging::flush();
            // SAFETY: nothing may be in flight when a feature is destroyed.
            let _ = unsafe { device.device_wait_idle() };
            let old = std::mem::replace(&mut s.passes[pass].handle, std::ptr::null_mut());
            release_handle(s, old, pass);
        }

        let created = create_feature_at(s, device, queue, width, height, &wanted[pass]);
        s.build_after = Some(Instant::now() + spacing);
        match created {
            Some(handle) => {
                s.ever_built = true;
                s.built_size = (width, height);
                let slot = PassSlot { handle, built: wanted[pass], needs_reset: true, failures: 0 };
                if pass < s.passes.len() {
                    s.passes[pass] = slot;
                } else {
                    s.passes.push(slot);
                }
            }
            None if first_build => {
                // Not retried every frame (that would redo the whole command pool/fence setup
                // per captured frame), and not fatal either: blocked at this size until it
                // changes or the retry interval passes.
                crate::log!("[helper] the model would not build at {width}x{height}; passing frames through, retrying when the size changes or in {}s", RETRY_FAILED_SIZE_AFTER.as_secs());
                crate::logging::flush();
                s.failure_note = Some(format!("model would not build at {width}x{height}; lower the resolution scale"));
                s.create_failed = Some(((width, height), Instant::now()));
                return 0;
            }
            None if pass >= s.passes.len() => {
                // A later pass would not build: a ceiling, not a fault.
                crate::log!("[helper] pass {pass} would not build; holding the chain at {}", s.passes.len());
                crate::logging::flush();
                s.ceiling = Some(s.passes.len());
            }
            None => {
                let slot = &mut s.passes[pass];
                slot.failures += 1;
                crate::log!("[helper] pass {pass} rebuild failed ({}); skipping it until it builds", slot.failures);
                if slot.failures >= MAX_BUILD_FAILURES {
                    if pass == 0 {
                        crate::log!("[helper] pass 0 failed {} times; giving up on the model", slot.failures);
                        s.disabled = true;
                        return 0;
                    }
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
