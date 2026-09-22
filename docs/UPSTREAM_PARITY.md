# Upstream parity

Compared against DLSS5VKLayer (bmitch87) **0.3.1-1**, commit `117c953`, shared-memory header
**v20** (2026-09-15). Earlier notes in `docs/history/` refer to 0.2.6-x; they are kept as written.
Every ported function is listed in [ATTRIBUTION.md](../ATTRIBUTION.md).

The two shared-memory protocols are **not wire-compatible and are not meant to be**. Neural
Forge's header is its own (version 5 as of 0.1.80). That both headers once happened to share a
1948-byte prefix is a coincidence, not a contract: never point one project's helper at the
other's mapping.

## Adopted (0.1.78)

| Upstream | Here |
|---|---|
| Create-time tuning block (`NgxSetCreateTuning`) and value-compared, debounced rebuilds (`MaintainPasses`) | `ngx::set_create_tuning`, `ngx::maintain_passes` |
| One NGX feature per pass, pass ceiling on build failure | `ngx::maintain_passes`, `frame::FrameResources::evaluate` chain |
| `PerfQualityValue` = Balanced; `DLSSNR_SKIP_NVAPI` | same, as `NEURAL_FORGE_SKIP_NVAPI` |
| Guard: ignore debug-print exceptions, cap hits, log rip/module ranges | `guard.rs` (with the non-unwinding `_setjmp`, which upstream does not use) |
| 64x64 minimum feature, rebuild on resize | `ngx::MIN_FEATURE_DIM`, `maintain_passes` |
| Present waits consumed once by the layer's first submit | relay batch (`device.rs`), drain before freeing swapchain resources, claim release on setup failure |
| Blocking, same-frame present (`ProcessPresent`) | synchronous present in `capture::run` (bounded wait, heartbeat check, fail-open) |
| NVIDIA vendor gate, duplicate-layer guard, DEVICE_LOST latch | `device.rs`, `lib.rs` |
| evdev + XInput2 hotkeys, `DLSSNR_TOGGLE_KEY`/`HOTKEY_BACKEND` | `hotkey.rs`, `NEURAL_FORGE_*` names |
| Hue trust (`kQuantFloor`) | `compose.comp` mode 2 |
| GUI: live reload, per-pass dialog, rebuild spacing, wider ranges | `crates/gui` |
| Compare views (side by side, wipe, swap, zoom) | `compose.comp` (0.1.80) |
| Colour trust and ratio smoothing, defaults 2 and 100% | `compose.comp` (0.1.80) |
| Transfer modes 0/1/2 (classic, matched residual, native + edit) | `compose.comp` + small-proxy enlargement in `composition/gpu.rs` (0.1.80) |
| White meter (tile peaks, 90th percentile, lit acceptance) and the Measured source | `capture::meter_white` on the CPU, smoothed (0.1.80) |
| Frame hold | synchronous present + `composition/gpu.rs` held-frame input (0.1.80) |
| `DEVELOPMENT.md` invariants | `CLAUDE.md`, "Composition invariants" |
| System-Wine prefix: DXVK 3.1 `dxgi.dll` + DXVK-NVAPI 0.9.2 (supplied, from Proton, or SHA256-pinned download), `dxvk.conf`, native overrides | `supervisor::provision` (0.1.81); verified end to end under Wine 10 on the rig |
| 32-bit layer (`VK_LAYER_neuralforge_neural_32`, its own manifest) | per-region shared-memory mapping capped at a 4K 8-bit frame on 32-bit; built and packaged by `build-appimage.sh`, tested in CI (0.1.83) |
| 16-bit multipass intermediates (`sdr16_multipass`) | `frame.rs` RGBA16F working images, falls back to 8-bit if refused (0.1.83) |
| `answered_w`/`answered_h` guard against another swapchain's answer | helper echoes, synchronous present checks (0.1.81) |
| Debug views 4/5 (colour-trust engagement, pre-bound colour) | `compose.comp` `debug_view`; views 0-3 (this project's own) and 4/5 now reach the GPU path, so they work for a game that blits into its swapchain too (0.1.84) |

## Deliberately not adopted

- Upstream ORs `TRANSFER_SRC|DST` into every swapchain unconditionally; Neural Forge keeps its
  admission checks (`surface_usage.rs`).
- Re-running `NgxLoadAndInit` on every resize (re-hooks the import table, leaks parameters) and
  quitting after one bad evaluate: Neural Forge rebuilds only the feature and fails open per frame.
- The meter's descriptor pool without a storage-buffer size, resetting a possibly unsignalled
  fence in `RestorePresent`, binding R8/A2R10/RGBA16F views to an `Rgba32f` storage
  declaration, and the env-parsed knobs that never reach the shader.
- Header v14+'s `layer_pid`: the GUI derives active/idle/gone from heartbeats instead, avoiding
  a header change for the same answer.
- Writing a `VK_INSTANCE_LAYERS` ordering snippet into `~/.config/environment.d/`: it would
  force-load the layer into every Vulkan program in the session. Documented as a per-game
  launch option instead (README, Troubleshooting).

## Not done yet

- **HDR / 10-bit / float16** swapchains: presented untouched. Needs the float16 proxy, PQ10
  transfer and half-float compose fallback (`composition.cpp` `Prepare`, `DetectHdrKind`).
- **A GPU-side white meter**: the meter runs on the CPU over the captured frame instead, which
  the synchronous present already holds -- measured fast enough in practice not to need the GPU.
- **The CPU downscale kernels** (`composition/downscale.rs`) remain unwired; the model is scaled
  with the hardware blit, and supersampling above 100% is not offered.
- **Upstream's vendored `vulkan-1.dll`** for system Wine: not carried; Wine's own `winevulkan`
  worked in the end-to-end test.
- **Motion vectors:** the GPU deadzone shader (`mvec_deadzone.comp`, two compile-time variants).
  Motion is off by default and the rig reports no optical-flow support, so it cannot be tested
  there.
- **dma-buf transport:** not wired (the GUI no longer offers the switch). A retest against system
  Wine + DXVK-NVAPI is now possible, since that runner works.
- **Frame generation:** with DLSS Frame Generation on, every presented frame -- generated ones
  included -- waits for its own model answer, so generation adds no frames (and half the model's
  work goes into generated frames). 0.1.83's "Model every Nth frame" (2 with
  frame generation) carries the answer to the frames between, measured +29% presented frames on the
  rig. The full fix is to enhance before frame generation (the game's render target, not the
  swapchain).
- **Why compositing during GTA V's loading screens stalls the game** is still unknown; the layer
  avoids it by waiting for 5 s of steady rendering (`swapchain::Warmup`). v0.1.85/0.1.86 found and
  fixed one concrete way this *class* of stall could happen (every `wait_for_fences` the layer or
  helper can hit on the game's own present/build path used to pass `u64::MAX` -- a driver that
  stalls without losing the device would park the thread forever with nothing logged; now bounded
  to 5s and logged via a breadcrumb trail, `crates/layer/src/breadcrumbs.rs`, see PR #22 against
  DLSS5VKLayer and `ATTRIBUTION.md`). Not confirmed as *the* cause -- if `nf-layer.log` ever shows
  a `fence wait timed out` line, that confirms it; if the freeze recurs with no such line, the
  cause is elsewhere (the helper's several `device_wait_idle()` calls and `frame.rs`'s per-frame
  evaluate wait are the next suspects, deliberately left unbounded in 0.1.86 -- see that version's
  changelog entry for why).
- **Running the model before the game's own upscaler** (on the internal render resolution)
  rather than on the upscaled output: a later performance idea.
