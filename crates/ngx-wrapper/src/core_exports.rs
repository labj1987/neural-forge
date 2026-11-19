//! NVIDIA's NGX core (`_nvngx.dll`, the driver's `nvidia/wine` folder) as the NGX wrapper is built for it: every name
//! it exports, the ones the wrapper hooks and the ones whose code reads its caller. Shared with the Linux side
//! (`neural_forge_supervisor::ngx_core_exports`, through a `#[path]` module), whose `neural-forge-cli ngx-audit`
//! checks a driver's core against these lists (docs/PROTON_NGX.md, "After a driver update").

/// The driver whose core these lists were taken from and checked against (Ghidra and `ngx-audit`), and that core's
/// SHA-256: `ngx_audit`'s test scans it wherever that exact file is installed.
pub const CHECKED_DRIVER: &str = "620.73.04";
pub const CHECKED_CORE_SHA256: &str = "637a9390855b16c7f6c6a723e5fc97ab11bca20224064780f25dcff8b774892a";

/// Every name the core exports, sorted: a name's ordinal is its index here plus 1 (the core numbers its exports in
/// alphabetical order from 1, and the linker gives the wrapper's the same; `scripts/build-ngx-wrapper.sh` compares
/// both DLLs).
pub const CORE_EXPORTS: &[&str] = &[
    "NVSDK_NGX_CUDA_AllocateParameters",
    "NVSDK_NGX_CUDA_CreateFeature",
    "NVSDK_NGX_CUDA_CreateFeature1",
    "NVSDK_NGX_CUDA_DestroyParameters",
    "NVSDK_NGX_CUDA_EvaluateFeature",
    "NVSDK_NGX_CUDA_GetCapabilityParameters",
    "NVSDK_NGX_CUDA_GetDeviceCapabilityParameters",
    "NVSDK_NGX_CUDA_GetFeatureRequirements",
    "NVSDK_NGX_CUDA_GetParameters",
    "NVSDK_NGX_CUDA_GetScratchBufferSize",
    "NVSDK_NGX_CUDA_Init",
    "NVSDK_NGX_CUDA_Init_Ext",
    "NVSDK_NGX_CUDA_Init_Ext1",
    "NVSDK_NGX_CUDA_Init_ProjectID",
    "NVSDK_NGX_CUDA_ReleaseFeature",
    "NVSDK_NGX_CUDA_Shutdown",
    "NVSDK_NGX_CUDA_Shutdown1",
    "NVSDK_NGX_D3D11_AllocateParameters",
    "NVSDK_NGX_D3D11_CreateFeature",
    "NVSDK_NGX_D3D11_DestroyParameters",
    "NVSDK_NGX_D3D11_EvaluateFeature",
    "NVSDK_NGX_D3D11_GetCapabilityParameters",
    "NVSDK_NGX_D3D11_GetDeviceCapabilityParameters",
    "NVSDK_NGX_D3D11_GetFeatureRequirements",
    "NVSDK_NGX_D3D11_GetParameters",
    "NVSDK_NGX_D3D11_GetScratchBufferSize",
    "NVSDK_NGX_D3D11_Init",
    "NVSDK_NGX_D3D11_Init_Ext",
    "NVSDK_NGX_D3D11_Init_ProjectID",
    "NVSDK_NGX_D3D11_ReleaseFeature",
    "NVSDK_NGX_D3D11_Shutdown",
    "NVSDK_NGX_D3D11_Shutdown1",
    "NVSDK_NGX_D3D12_AllocateParameters",
    "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_DestroyParameters",
    "NVSDK_NGX_D3D12_EvaluateFeature",
    "NVSDK_NGX_D3D12_GetCapabilityParameters",
    "NVSDK_NGX_D3D12_GetDeviceCapabilityParameters",
    "NVSDK_NGX_D3D12_GetFeatureRequirements",
    "NVSDK_NGX_D3D12_GetParameters",
    "NVSDK_NGX_D3D12_GetScratchBufferSize",
    "NVSDK_NGX_D3D12_Init",
    "NVSDK_NGX_D3D12_Init_Ext",
    "NVSDK_NGX_D3D12_Init_ProjectID",
    "NVSDK_NGX_D3D12_ReleaseFeature",
    "NVSDK_NGX_D3D12_Shutdown",
    "NVSDK_NGX_D3D12_Shutdown1",
    "NVSDK_NGX_OTA_UPDATES_CheckForUpdate",
    "NVSDK_NGX_OTA_UPDATES_GetPath",
    "NVSDK_NGX_OTA_UPDATES_Install",
    "NVSDK_NGX_OTA_UPDATES_Register",
    "NVSDK_NGX_OTA_UPDATES_Unregister",
    "NVSDK_NGX_OTA_UPDATES_Update",
    "NVSDK_NGX_UpdateFeature",
    "NVSDK_NGX_VULKAN_AllocateParameters",
    "NVSDK_NGX_VULKAN_CreateFeature",
    "NVSDK_NGX_VULKAN_CreateFeature1",
    "NVSDK_NGX_VULKAN_DestroyParameters",
    "NVSDK_NGX_VULKAN_EvaluateFeature",
    "NVSDK_NGX_VULKAN_GetCapabilityParameters",
    "NVSDK_NGX_VULKAN_GetDeviceCapabilityParameters",
    "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements",
    "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements",
    "NVSDK_NGX_VULKAN_GetFeatureRequirements",
    "NVSDK_NGX_VULKAN_GetParameters",
    "NVSDK_NGX_VULKAN_GetScratchBufferSize",
    "NVSDK_NGX_VULKAN_Init",
    "NVSDK_NGX_VULKAN_Init_Ext",
    "NVSDK_NGX_VULKAN_Init_Ext2",
    "NVSDK_NGX_VULKAN_Init_ProjectID",
    "NVSDK_NGX_VULKAN_Init_ProjectID_Ext",
    "NVSDK_NGX_VULKAN_ReleaseFeature",
    "NVSDK_NGX_VULKAN_RequiredExtensions",
    "NVSDK_NGX_VULKAN_Shutdown",
    "NVSDK_NGX_VULKAN_Shutdown1",
];

/// Exports with a hook (`hooks.rs`).
pub const HOOKED: &[&str] = &[
    "NVSDK_NGX_D3D11_CreateFeature",
    "NVSDK_NGX_D3D11_EvaluateFeature",
    "NVSDK_NGX_D3D11_ReleaseFeature",
    "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_EvaluateFeature",
    "NVSDK_NGX_D3D12_ReleaseFeature",
    "NVSDK_NGX_VULKAN_CreateFeature",
    "NVSDK_NGX_VULKAN_CreateFeature1",
    "NVSDK_NGX_VULKAN_EvaluateFeature",
    "NVSDK_NGX_VULKAN_ReleaseFeature",
];

/// Passthrough exports whose code in NVIDIA's core finds the module that called it, from its return address
/// (`GetModuleHandleExA(FROM_ADDRESS)` and `GetModuleFileNameW`), and searches that module's folder for DLSS's feature
/// DLLs. The call trace leaves their return address alone (it logs their entry only): replaced by its own, the core
/// would take the wrapper in `system32` for the caller and miss the game's feature DLLs.
pub const READS_CALLER: &[&str] = &[
    "NVSDK_NGX_CUDA_GetFeatureRequirements",
    "NVSDK_NGX_CUDA_Init",
    "NVSDK_NGX_CUDA_Init_Ext",
    "NVSDK_NGX_CUDA_Init_Ext1",
    "NVSDK_NGX_CUDA_Init_ProjectID",
    "NVSDK_NGX_D3D11_GetFeatureRequirements",
    "NVSDK_NGX_D3D11_Init",
    "NVSDK_NGX_D3D11_Init_Ext",
    "NVSDK_NGX_D3D11_Init_ProjectID",
    "NVSDK_NGX_D3D12_GetFeatureRequirements",
    "NVSDK_NGX_D3D12_Init",
    "NVSDK_NGX_D3D12_Init_Ext",
    "NVSDK_NGX_D3D12_Init_ProjectID",
    "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements",
    "NVSDK_NGX_VULKAN_GetFeatureRequirements",
    "NVSDK_NGX_VULKAN_Init",
    "NVSDK_NGX_VULKAN_Init_Ext",
    "NVSDK_NGX_VULKAN_Init_Ext2",
    "NVSDK_NGX_VULKAN_Init_ProjectID",
    "NVSDK_NGX_VULKAN_Init_ProjectID_Ext",
];

/// Exports whose code ends in a jump through a pointer the audit's scan cannot follow, handing the target their
/// return address: the four `*_GetScratchBufferSize` jump to the DLSS feature's own function, from a table the core
/// fills at run time (`[table + 0x98 * feature + 0x2538]`, driver 620.73). The audit expects exactly these.
pub const UNFOLLOWED_TAIL_JUMPS: &[&str] = &[
    "NVSDK_NGX_CUDA_GetScratchBufferSize",
    "NVSDK_NGX_D3D11_GetScratchBufferSize",
    "NVSDK_NGX_D3D12_GetScratchBufferSize",
    "NVSDK_NGX_VULKAN_GetScratchBufferSize",
];
