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
- **Debug views 4 and 5** (colour-bound and pre-bound views) and a GPU-side meter: the meter
  runs on the CPU over the captured frame instead, which the synchronous present already holds.
- **The CPU downscale kernels** (`composition/downscale.rs`) remain unwired; the model is scaled
  with the hardware blit, and supersampling above 100% is not offered.
- **32-bit layer:** builds, but the 1.3 GB mapping does not reliably fit a 32-bit address space;
  needs per-region mapping before it can ship.
- **System-Wine runner provisioning:** SHA256-pinned DXVK 3.1 `dxgi.dll` and dxvk-nvapi 0.9.2,
  `dxvk.conf`, a vendored `vulkan-1.dll` (upstream's `dlssnr-helper`). Proton runners do not
  need it. `cli doctor`'s DXVK check reports it missing until then.
- **16-bit multipass intermediates**, the GPU motion-vector deadzone shader
  (`mvec_deadzone.comp`, two compile-time variants), and a retest of the dma-buf export path.
- **Why compositing during GTA V's loading screens stalls the game** is unknown; the layer
  avoids it by waiting for 5 s of steady rendering (`swapchain::Warmup`).
- **Running the model before the game's own upscaler** (on the internal render resolution)
  rather than on the upscaled output: a later performance idea.
