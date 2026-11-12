//! NVIDIA's `_nvngx.dll` exports (driver 620.70, 75 names; the ordinals are their alphabetical order, which the
//! linker gives these too: `scripts/build-ngx-wrapper.sh` checks both against the driver's DLL when it is installed).
//! The ones Neural Forge watches are in `hooks.rs`; every other one is a jump through the pointer resolved from
//! NVIDIA's core at load (`win.rs`, `resolve`), or to [`missing_export`] when the core lacks it
//! ([`crate::rules::resolution`]).

/// Exports forwarded untouched.
macro_rules! passthrough {
    ($($name:ident)*) => {
        /// Their names, for resolving.
        pub(crate) const PASSTHROUGH: &[&str] = &[$(stringify!($name)),*];
        /// Their targets in NVIDIA's core, in [`PASSTHROUGH`]'s order.
        #[cfg(windows)]
        pub(crate) static mut PASSTHROUGH_TARGETS: [usize; PASSTHROUGH.len()] = [0; PASSTHROUGH.len()];
        /// The traced entries ([`crate::trace`]): with `NEURAL_FORGE_NGX_TRACE` set, [`PASSTHROUGH_TARGETS`] holds
        /// these, and NVIDIA's functions are in [`crate::win::REAL_TARGETS`].
        #[cfg(windows)]
        pub(crate) static PASSTHROUGH_TRACED: [unsafe extern "C" fn(); PASSTHROUGH.len()] = [$(traced::$name),*];
        passthrough!(@each 0usize; $($name)*);
        #[cfg(windows)]
        #[allow(non_snake_case)]
        mod traced {
            passthrough!(@traced 0usize; $($name)*);
        }
    };
    (@traced $i:expr; $name:ident $($rest:ident)*) => {
        /// The traced entry of this export: its index in r11, then the common thunk ([`crate::win::trace_thunk`]).
        ///
        /// # Safety
        /// NGX's contract for this function.
        #[unsafe(naked)]
        pub(super) unsafe extern "C" fn $name() {
            core::arch::naked_asm!("mov r11, {i}", "jmp {thunk}", i = const $i, thunk = sym crate::win::trace_thunk);
        }
        passthrough!(@traced $i + 1usize; $($rest)*);
    };
    (@traced $i:expr;) => {};
    (@each $i:expr; $name:ident $($rest:ident)*) => {
        /// NVIDIA's function of this name, reached by a jump.
        ///
        /// # Safety
        /// NGX's contract for this function.
        #[cfg(windows)]
        #[unsafe(naked)]
        #[no_mangle]
        pub unsafe extern "C" fn $name() {
            core::arch::naked_asm!("jmp qword ptr [rip + {t} + {off}]", t = sym PASSTHROUGH_TARGETS, off = const ($i * 8));
        }
        passthrough!(@each $i + 1usize; $($rest)*);
    };
    (@each $i:expr;) => {};
}

passthrough! {
    NVSDK_NGX_CUDA_AllocateParameters NVSDK_NGX_CUDA_CreateFeature NVSDK_NGX_CUDA_CreateFeature1 NVSDK_NGX_CUDA_DestroyParameters
    NVSDK_NGX_CUDA_EvaluateFeature NVSDK_NGX_CUDA_GetCapabilityParameters NVSDK_NGX_CUDA_GetDeviceCapabilityParameters
    NVSDK_NGX_CUDA_GetFeatureRequirements NVSDK_NGX_CUDA_GetParameters NVSDK_NGX_CUDA_GetScratchBufferSize NVSDK_NGX_CUDA_Init
    NVSDK_NGX_CUDA_Init_Ext NVSDK_NGX_CUDA_Init_Ext1 NVSDK_NGX_CUDA_Init_ProjectID NVSDK_NGX_CUDA_ReleaseFeature
    NVSDK_NGX_CUDA_Shutdown NVSDK_NGX_CUDA_Shutdown1
    NVSDK_NGX_D3D11_AllocateParameters NVSDK_NGX_D3D11_DestroyParameters NVSDK_NGX_D3D11_GetCapabilityParameters
    NVSDK_NGX_D3D11_GetDeviceCapabilityParameters NVSDK_NGX_D3D11_GetFeatureRequirements NVSDK_NGX_D3D11_GetParameters
    NVSDK_NGX_D3D11_GetScratchBufferSize NVSDK_NGX_D3D11_Init NVSDK_NGX_D3D11_Init_Ext NVSDK_NGX_D3D11_Init_ProjectID
    NVSDK_NGX_D3D11_Shutdown NVSDK_NGX_D3D11_Shutdown1
    NVSDK_NGX_D3D12_AllocateParameters NVSDK_NGX_D3D12_DestroyParameters NVSDK_NGX_D3D12_GetCapabilityParameters
    NVSDK_NGX_D3D12_GetDeviceCapabilityParameters NVSDK_NGX_D3D12_GetFeatureRequirements NVSDK_NGX_D3D12_GetParameters
    NVSDK_NGX_D3D12_GetScratchBufferSize NVSDK_NGX_D3D12_Init NVSDK_NGX_D3D12_Init_Ext NVSDK_NGX_D3D12_Init_ProjectID
    NVSDK_NGX_D3D12_Shutdown NVSDK_NGX_D3D12_Shutdown1
    NVSDK_NGX_OTA_UPDATES_CheckForUpdate NVSDK_NGX_OTA_UPDATES_GetPath NVSDK_NGX_OTA_UPDATES_Install NVSDK_NGX_OTA_UPDATES_Register
    NVSDK_NGX_OTA_UPDATES_Unregister NVSDK_NGX_OTA_UPDATES_Update NVSDK_NGX_UpdateFeature
    NVSDK_NGX_VULKAN_AllocateParameters NVSDK_NGX_VULKAN_DestroyParameters NVSDK_NGX_VULKAN_GetCapabilityParameters
    NVSDK_NGX_VULKAN_GetDeviceCapabilityParameters NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements
    NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements NVSDK_NGX_VULKAN_GetFeatureRequirements NVSDK_NGX_VULKAN_GetParameters
    NVSDK_NGX_VULKAN_GetScratchBufferSize NVSDK_NGX_VULKAN_Init NVSDK_NGX_VULKAN_Init_Ext NVSDK_NGX_VULKAN_Init_Ext2
    NVSDK_NGX_VULKAN_Init_ProjectID NVSDK_NGX_VULKAN_Init_ProjectID_Ext NVSDK_NGX_VULKAN_RequiredExtensions
    NVSDK_NGX_VULKAN_Shutdown NVSDK_NGX_VULKAN_Shutdown1
}

/// Exports with a hook (`hooks.rs`).
pub(crate) const HOOKED: &[&str] = &[
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

/// What a passthrough export NVIDIA's core lacks jumps to: `NVSDK_NGX_Result_Fail`, as NGX answers a call it cannot
/// serve. It takes no arguments, which is safe for every export's signature: on x64 Windows the caller passes
/// arguments in registers and its own stack space and cleans up after the call (no callee-popped arguments).
pub extern "C" fn missing_export() -> u32 {
    crate::rules::NGX_RESULT_FAIL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_export_fails() {
        assert_eq!(missing_export(), 0xBAD0_0000);
        assert!(!crate::rules::ngx_succeeded(missing_export()));
    }

    #[test]
    fn every_export_once() {
        let mut all: Vec<&str> = PASSTHROUGH.iter().chain(HOOKED).copied().collect();
        assert_eq!(all.len(), 75, "NVIDIA's core exports 75 names");
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 75, "a name twice");
    }
}
