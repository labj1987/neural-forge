//! Native backend: the vendored OpenDLSS-NR host code (`third_party/opendlss-nr`), with its SPIR-V and
//! PTX kernels embedded at build time, behind a small C API (`cpp/nf_native.h`).
//!
//! `build.rs` compiles the kernels with the pinned toolchain from `scripts/fetch-native-tools.sh`, embeds
//! them, and links the C++ (libstdc++ included, statically) into this rlib. x86_64 Linux only.

use std::ffi::CString;

mod ffi {
    use std::os::raw::c_char;

    extern "C" {
        pub fn nf_native_abi_version() -> u32;
        pub fn nf_native_asset_count(kind: *const c_char) -> u32;
    }
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

impl AssetKind {
    fn c_name(self) -> CString {
        CString::new(match self {
            AssetKind::Spirv => "spv",
            AssetKind::Ptx => "ptx",
        })
        .expect("no NUL in a literal")
    }
}

/// How many kernels of a kind are embedded.
pub fn asset_count(kind: AssetKind) -> u32 {
    let name = kind.c_name();
    // SAFETY: `name` is a valid NUL-terminated string for the duration of the call.
    unsafe { ffi::nf_native_asset_count(name.as_ptr()) }
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
