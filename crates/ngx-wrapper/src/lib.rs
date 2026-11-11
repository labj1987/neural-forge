//! Neural Forge's NGX wrapper for Proton (docs/PROTON_NGX.md).
//!
//! Proton copies NVIDIA's signed `_nvngx.dll` (the NGX core) into the prefix's `system32`. NGX's loader in the game
//! (or in Streamline's `sl.common.dll`) checks that file's signature and then loads it; in a Proton made by
//! `neural-forge-cli proton-tool create`, `WINEDLLOVERRIDES` says `_nvngx=b`, so Wine hands the loader this DLL, a
//! Wine builtin of the same name, instead. It loads NVIDIA's core by its absolute path from `NVIDIA_WINE_DLL_DIR`
//! (that path is overridden to native), forwards every export, and writes one record per creation and evaluate of
//! a DLSS feature (the Vulkan images of its inputs and output, the jitter, the motion vectors' scale, the reset,
//! the exposure values, the feature's creation flags) into a ring the Neural Forge layer in the same process maps
//! (`ring.rs`). It never changes a call's arguments or result.

pub mod exports;
pub mod ring;

#[cfg(windows)]
mod hooks;
#[cfg(windows)]
mod win;
