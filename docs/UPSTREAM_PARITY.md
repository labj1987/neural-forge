# Upstream parity

> **Note (2026-10-07, 3.0.0):** written before 3.0. Since 3.0 the model runs inside the layer
> (the native backend, [NATIVE_BACKEND.md](NATIVE_BACKEND.md)); the Windows helper, Wine, the
> runners, NGX at run time and the 32-bit layer are gone. What this document says about them is
> history; [ARCHITECTURE.md](ARCHITECTURE.md) describes the current design.

> **Note (2026-10-02, 2.0.0):** the "Frame generation" item below is resolved. With the model
> before DLSS Super Resolution (the 2.0 default), DLSS Frame Generation was measured on the rig at
> 53.0 real / 159 shown fps against 28.7 / 86 on the 1.x path
> ([PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md), "DLSS Frame Generation"). The other "Not done
> yet" items still stand.

Compared against DLSS5VKLayer (bmitch87) **0.3.1-1**, commit `117c953`, shared-memory header
**v20** (2026-09-15). Measured head-to-head against **0.3.1-2** (commit `9f43793`, 2026-09-26) on 2026-10-02: it still
cannot run GTA V Enhanced (the Rockstar Launcher's frames make its helper rebuild every frame), and in
Cyberpunk 2077 it matches Neural Forge 2.0 at the same work per frame (37.9 vs 38.3 fps). See
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md), "upstream 0.3.1-2 vs Neural Forge 2.0.0". Earlier notes in `docs/history/` refer to 0.2.6-x; they are kept as written.
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
| `DEVELOPMENT.md` invariants | `AGENTS.md`, "Composition invariants" |
| System-Wine prefix: DXVK 3.1 `dxgi.dll` + DXVK-NVAPI 0.9.2 (supplied, from Proton, or SHA256-pinned download), `dxvk.conf`, native overrides | `supervisor::provision` (0.1.81); verified end to end under Wine 10 on the rig |
| 32-bit layer (`VK_LAYER_neuralforge_neural_32`, its own manifest) | per-region shared-memory mapping capped at a 4K 8-bit frame on 32-bit; built and packaged by `build-appimage.sh`, tested in CI (0.1.83) |
| 16-bit multipass intermediates (`sdr16_multipass`) | `frame.rs` RGBA16F working images, falls back to 8-bit if refused (0.1.83) |
| `answered_w`/`answered_h` guard against another swapchain's answer | helper echoes, synchronous present checks (0.1.81) |
| Debug views 4/5 (colour-trust engagement, pre-bound colour) | `compose.comp` `debug_view`; views 0-3 (this project's own) and 4/5 now reach the GPU path, so they work for a game that blits into its swapchain too (0.1.84) |
| Reversible modes (soft knee, Neutwo, hybrid, and the two pure-inverse replace modes) | `encode.comp`; `compose.comp` rebuilds the frame's proxy with the encode's own curve (peak step included) and implements replace |

Capture source (1.0.0): an admitted swapchain is captured from its own image. Neural Forge's
render tap is used only on a pass-through swapchain, and only for a 1:1 copy or blit whose
source extent and format (recorded from `vkCreateImage`) match the swapchain; anything else
presents untouched. See `RENDER_TAP_DESIGN.md`.

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
- **Motion vectors:** upstream's `mvec_deadzone.comp` variants. Neural Forge estimates motion
  entirely on the GPU (`crates/helper/src/optical_flow.rs`: half-resolution optical flow, then
  `crates/helper/shaders/flow_to_mvec.comp` with a plain 0.5 px deadzone), validated on the rig
  (RTX 5070 under Proton, 0.78 ms per estimate at 2560x1440:
  `crates/helper/examples/optical_flow_rig_check.rs`); upstream's finer deadzone variants are not
  carried.
- **dma-buf transport:** not wired (the GUI no longer offers the switch). A retest against system
  Wine + DXVK-NVAPI is now possible, since that runner works.
- **Frame generation:** with DLSS Frame Generation on, every presented frame -- generated ones
  included -- waits for its own model answer, so generation adds no frames (and half the model's
  work goes into generated frames). 0.1.83's "Model every Nth frame" (2 with
  frame generation) carries the answer to the frames between, measured +29% presented frames on the
  rig. The full fix is to enhance before frame generation (the game's render target, not the
  swapchain). Done in 2.0: the pre-upscaler path (below) runs before DLSS Super Resolution and
  Frame Generation, and was measured at 53.0 real / 159 shown fps with DLSS FG 3x (see the note
  at the top).
- **Why compositing during GTA V's loading screens stalls the game** is still unknown; the layer
  avoids it by waiting for 5 s of steady rendering (`swapchain::Warmup`). v0.1.85-0.1.87 found and
  fixed one concrete way this *class* of stall could happen: every `wait_for_fences` the layer or
  helper can hit on the game's own present/build/evaluate path used to pass `u64::MAX` -- a driver
  that stalls without losing the device would park the thread forever with nothing logged. All of
  them are now bounded (5s) and logged via a breadcrumb trail on the layer side
  (`crates/layer/src/breadcrumbs.rs`); see PR #22 against DLSS5VKLayer and `ATTRIBUTION.md`. Not
  confirmed as *the* cause -- if `nf-layer.log` ever shows a `fence wait timed out` line, that
  confirms it. If the freeze recurs with no such line, the one remaining unbounded class is the
  helper's `device_wait_idle()` calls (`ngx.rs`): that Vulkan API has no
  timeout parameter at all, so bounding it needs a different mechanism (e.g. a watchdog thread)
  and a decision about whether it's safe to keep using a device once one such call is considered
  abandoned -- not attempted, deliberately, pending that decision. (`optical_flow.rs` no longer
  has one: its per-frame wait is a fence with `FENCE_WAIT_TIMEOUT`, and a stalled session is
  leaked rather than waited on.)
- **Running the model before the game's own upscaler:** done in 2.0 and the default where DLSS
  Super Resolution's input is found (`docs/PRE_UPSCALER_DESIGN.md`; `NEURAL_FORGE_PREUPSCALE=off`
  switches back). Not done yet there: an asynchronous hold (the capture still waits for the
  game's queued work, 3.6-4.4 ms), the game's own depth and motion vectors for the model (it
  still uses optical flow), and a probe of any game other than GTA V Enhanced.
