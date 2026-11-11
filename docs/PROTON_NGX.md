# DLSS's own parameters through Proton: the NGX wrapper

The layer works out DLSS's colour input, depth, motion vectors, output, exposure, camera jitter, motion vector units
and reset from the CUDA kernels DLSS launches (`preupscale.rs`, PRE_UPSCALER_DESIGN.md). Every new engine has needed new
rules there. The game itself hands NGX all of these by name, in the same process, on the Windows side of Proton. This
document is how Neural Forge reads them there, and how the layer uses them.

## How it fits together

```
game / sl.dlss.dll ──NGX's static loader──► C:\windows\system32\_nvngx.dll   (WinVerifyTrust on NVIDIA's signed file)
                                              │ Wine's load order: _nvngx=b
                                              ▼
                        Neural Forge's builtin _nvngx.dll (crates/ngx-wrapper)
                              │ forwards all 75 exports to NVIDIA's core, loaded from NVIDIA_WINE_DLL_DIR
                              │ one record per Create/EvaluateFeature ──► /dev/shm/neural-forge-ngx-<pid>.ring
                              ▼
            vkCmdCuLaunchKernelNVX ──► VK_LAYER_neuralforge_neural: ngx_ring.rs reads the ring
                                        record = source of truth; kernel rules = fallback and cross-check
```

- **Where NGX is loaded.** Proton copies NVIDIA's `_nvngx.dll` and `nvngx.dll` from the driver's `nvidia/wine`
  folder into the prefix's `system32` and sets `NVIDIA_WINE_DLL_DIR` (the `proton` script's `setup_prefix` and
  `init_session`). The prefix's registry names `C:\Windows\System32` as NGXCore's `FullPath`. NGX's static loader,
  linked into the game or into Streamline's `sl.common.dll`, checks that file's signature with `WinVerifyTrust` and
  loads it; NVIDIA's thin `nvngx.dll` is not used by it (Cyberpunk 2077, GTA V Enhanced, Crimson Desert, Red Dead
  Redemption 2).
- **Why a builtin passes the signature check.** Wine applies a DLL override to a module in the system directory by its
  base name (`dlls/ntdll/unix/loadorder.c`, `get_load_order`), and when the file found is a native PE and the load
  order says builtin, maps the builtin of that name instead (`dlls/ntdll/unix/loader.c`, `load_builtin`). The module
  keeps the requested path. So `WinVerifyTrust` reads NVIDIA's unchanged signed file, and `LoadLibrary` of that path
  returns the wrapper. A PE is a builtin when the 32 bytes at file offset 0x40 read `Wine builtin DLL`
  (`server/mapping.c`), which `scripts/build-ngx-wrapper.sh` writes after the build, as `winebuild --builtin` does.
- **Loading NVIDIA's core.** The wrapper loads `Z:<NVIDIA_WINE_DLL_DIR>/_nvngx.dll` by its full path. A full-path key
  is looked up before the base name, so the override `Z:\...\nvidia\wine\_nvngx=n` keeps that one native; without it
  Wine hands the wrapper back to itself. NVIDIA's snippets (`nvngx_dlss.dll` and the others) load as before.
- **What it records.** For D3D12 the `ID3D12Resource`s become VkImages through vkd3d-proton's
  `ID3D12DXVKInteropDevice1::GetVulkanResourceInfo1` (and the layout DLSS reads them in through
  `GetVulkanImageLayout`); for D3D11 through DXVK's `IDXGIVkInteropSurface::GetVulkanImageInfo`; Vulkan games hand
  NGX `NVSDK_NGX_Resource_VK` with the VkImage. winevulkan passes VkImage handles through unwrapped, so they are the
  handles the layer saw at `vkCreateImage`. Per evaluate: `Color`, `MotionVectors`, `Depth`, `Output`,
  `ExposureTexture`, `Jitter.Offset`, `MV.Scale`, `DLSS.Pre.Exposure`, `DLSS.Exposure.Scale`, `Reset`, the render
  subrect, the extents, and the feature (Super Resolution 1, Frame Generation 11, Ray Reconstruction 13) with its
  creation flags (`IsHDR`, `MVLowRes`, `MVJittered`, `DepthInverted`, `AutoExposure`).
- **The ring.** `crates/ngx-wrapper/src/ring.rs`: a 32-byte header, 1024 slots of 128 bytes. A writer reserves a slot
  with an atomic add, zeroes its `seq`, writes it, then sets `seq` to its number; the reader takes a slot only while
  `seq` reads the same before and after copying it. The file is named by the Linux pid (the wrapper asks Wine with
  `NtQueryInformationProcess(ProcessWineUnixPid)`; `Z:\proc\self` would name the wineserver, which opens files for
  Wine), which the layer gets from `getpid()`. The layer maps it read-only and removes the
  file, so nothing is left behind; `uninstall --purge` removes rings no layer opened. The record is written before
  NGX records the evaluate's launches, on the same thread, so the newest record naming a launch's images is that
  launch's evaluate.

## What the layer does with a record (`crates/layer/src/ngx_ring.rs`)

At a command buffer's first launch naming registered images, the newest Super Resolution or Ray Reconstruction record
whose colour input is among them belongs to that buffer:

- **Inputs:** the record's colour input, depth and motion vectors, where all three are registered images; the input
  rules' result is checked against them.
- **Ray Reconstruction:** by NGX's feature, not by kernel names: such a buffer is never evidence and never held before
  the upscaler, as before.
- **Jitter, reset, motion vector units:** the record's `Jitter.Offset`, `Reset` (through the same reset filter as the
  kernel's word) and `MV.Scale` (with the colour input's extent).
- **Output and exposure image:** the record's output replaces the rules' choice where the output kernel names it; its
  exposure image is the evidence's.
- **Pre-exposure:** where the encode uses the game's exposure as it is, the paper white is multiplied by
  `DLSS.Pre.Exposure / DLSS.Exposure.Scale` (the colour input is the scene times the pre-exposure; DLSS scales the
  exposure image's value). Encode and decode read the same paper white, so the write-back stays exact.
- **HDR:** `IsHDR` is checked: a game handing DLSS a display-referred input would show as a disagreement (the encode
  has no path for one yet).

Every check counts an agreement or a disagreement per fact. The first three disagreements of each are logged
(`[ngx] the kernel rules and NGX disagree on the ...`), a summary every 5 s (`[ngx] N records, M input launches
matched ...`), and `shmctl status` shows `ngx_records`, `ngx_matched`, `ngx_disagreements` and `ngx_disagree_kinds`
(bits in the order colour, depth, motion vectors, output, exposure image, jitter, motion vector units, reset, ray
reconstruction, HDR input). Without a ring (no wrapper, a native game, `NEURAL_FORGE_NGX_RING=off`) everything is as
before.

## The Proton copy (`crates/supervisor/src/proton_tool.rs`, `neural-forge-cli proton-tool`, the Setup tab)

Nothing of Proton is rebuilt. The app copies an installed Proton (GE-Proton, Proton-CachyOS, Valve's Proton or Proton
Experimental) into Steam's `compatibilitytools.d` as `<version>-NeuralForge`, with hard links where it is on the same
filesystem (no space), and writes three files new: `proton` with the two overrides after Valve's `NVIDIA_WINE_DLL_DIR`
block (refused unless that block is there exactly once), `compatibilitytool.vdf` with the new name, and
`files/lib/wine/x86_64-windows/_nvngx.dll`. The `version` file stays the source's, so a game's prefix is not upgraded
when it switches between the two. `PROTON_NO_NEURAL_FORGE_NGX=1` in a game's launch options turns the wrapper off for
it. When the source changes (a new GE-Proton in a "Latest" folder, a Proton Experimental update), the copy is listed as
outdated and `proton-tool update` (or Make again) makes a copy of the new version. Proton links many of a prefix's
`system32` files to the Proton that last ran it; a removed copy's prefixes are set up again by the next Proton a game
runs with.

Why a local copy and no published Proton: the wrapper is Neural Forge's own code (not derived from Wine; it links only
the mingw-w64 runtime and libgcc, whose notices ship in the AppImage), so the copy made on the user's machine conveys no
LGPL code. Publishing ready-made GE-Proton tarballs would make Neural Forge a distributor of Wine, vkd3d-proton, FFmpeg
and the rest, owing their complete corresponding source (LGPL-2.1 section 6; GE's "Source code" archives lack the
submodules). The `proton` script is BSD-3-Clause (Valve); its notice stays in the copy.

## Maintenance

- GE-Proton releases about three times a month (GE-Proton11-1 2026-06-24 to 11-12 2026-11-03: twelve). A copy is made
  again in seconds. The wrapper does not depend on the Wine version.
- What can break: Valve rewrites the `NVIDIA_WINE_DLL_DIR` block (its TODO says the copies into `system32` go once
  NGXCore's `NGXPath` is set): the patch then refuses, and the anchor needs updating. NVIDIA adds an export to its core:
  the wrapper lacks it and NGX's loader may refuse the core, so the game runs without DLSS. `build-ngx-wrapper.sh`
  compares the wrapper's exports (names and ordinals) with the driver's DLL wherever one is installed; a check in
  `doctor` against the user's driver is still to do. NGX starts verifying the loaded image instead of the file: the
  wrapper approach stops working (the fallback would be the in-memory export patch spike 1 used, from a builtin every
  game loads).

## Evidence

- Wine source: ValveSoftware/wine at `fda79a585939fb7512dc220a11249f9ffe825fcf` (GE-Proton11-12's submodule).
- A scratch prefix under the copy: `WinVerifyTrust(C:\windows\system32\_nvngx.dll)` = 0, `LoadLibrary` of that path
  returns the wrapper (its `EvaluateFeature` in the wrapper's module), NVIDIA's core loaded from
  `Z:\run\host\usr\lib\x86_64-linux-gnu\nvidia\wine\_nvngx.dll`, no export missing.
- Games (2026-11-10, RTX 5070, driver 620.70, 3840x2160): see MEASURED_RESULTS.md, "NGX wrapper".
