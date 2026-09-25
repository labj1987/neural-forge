# Attribution

**As of 2026-09-17, this project is no longer clean-room with respect to
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87), AGPL-3.0.** Earlier
versions of NeuralForge deliberately avoided reading DLSS5VKLayer's own source (or
anything GPL/AGPL-licensed) specifically to keep this project MIT-licensed. That
boundary produced a worse result — months of re-deriving fixes from behavior and docs
alone, repeatedly landing on techniques upstream had already tried and rejected — for
no benefit anyone actually wanted, so it was dropped. DLSS5VKLayer's own `LICENSE`
file is AGPL-3.0, so once its source is read and adapted, AGPL-3.0 is a requirement of
that upstream license for the derived code, not merely a preference of this project.
This project's own license is therefore **AGPL-3.0-or-later** (see [LICENSE](LICENSE)
and [README.md](README.md#license)), and this file now records what is actually read,
adapted, or taken from DLSS5VKLayer's real source, not just its documented behavior.

## What's taken, and how

| Source | What's taken |
|---|---|
| [DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87), AGPL-3.0 — specifically `layer_linux/src/composition.cpp`, `composition.h`, and `dlssnr/dlssnr.hlsl` read directly | **The composition strategy**: no temporal accumulator/held-answer carried forward and blended against newer frames. DLSS5VKLayer's own source comment (`dlssnr.hlsl`, in the resolve function) states this directly: a reprojected-history accumulator was built, measured as a dead end twice ("the model re-decides its detail with the framing, so an old answer does not belong to a new frame and reprojecting it only moves where the disagreement lands"), and removed — replaced with re-anchoring the composition to the model's cached answer against the *current* frame, fresh, every single present, with no motion-based suppression of the edit at all. NeuralForge's own `carry_delta`/motion-mask compositor (`compose.comp`, `crates/layer/src/composition/gpu.rs::record_temporal_delta_into_image`, added v0.1.49–v0.1.65) was independently arrived at and is architecturally the same kind of reprojection-with-suppression technique DLSS5VKLayer's own source documents trying and rejecting — this is why it kept ghosting/flickering no matter how the suppression threshold was tuned. Replaced per this finding; see `CHANGELOG.md` for the version this landed in. |
| DLSS5VKLayer, `layer_linux/src/dlssnr/dlssnr.hlsl` — **ported, not clean-room** | **The proxy encode**, in `crates/layer/src/composition/encode.rs` and `shaders/encode.comp`: the soft knee (luminance roll-off above 0.75 plus the peak-channel headroom pass), the unclipped `Neutwo` reversible proxy, and the hybrid curve, along with the white-point divide that precedes them. Upstream's `ATTRIBUTION.md` marks `dlssnr.hlsl` as GPL-3.0-tainted in their own tree (derived from OptiScaler-family forks); combining it here is what NeuralForge's AGPL-3.0-or-later relicense permits (AGPL-3.0 and GPL-3.0 are mutually compatible for combination). Kept in its own module precisely so `composition/color.rs`'s clean-room claim stays true — nothing in *that* file is read from `dlssnr.hlsl`. Taken because this project had no encode at all: the proxy handed to the model was a bit-identical copy of the frame, which degenerates the ratio transfer (`proxy == original` collapses its rescale target onto the frame) and shows the model blown highlights instead of gradation. Upstream's own reason for the knee is the same artifact class this project spent v0.1.49–v0.1.72 fighting: "the model is never shown a field of flat white whose blown pixels flip between frames — unstable input is unstable output". The peak-channel headroom pass is likewise taken with its measured justification intact (a saturated blue above 1.0 clears the luminance knee, gets one channel clipped, and arrives hue-rotated — upstream measured this as a green cast over the sky, denim and minimap **in GTA V specifically**). |
| DLSS5VKLayer, `dlssnr.hlsl` — **ported** | **The composition guard** in `compose.comp`'s additive path: the `1/512` ratio floor applied above *and* below, plus the asymmetric `lift` (brightening room shrinking to none as a pixel approaches black) and `drop` (darkening room shrinking to none as it approaches white). Replaced a symmetric, epsilon-gated bound that upstream's source names as the cause of "the boiling: patches of lighter colour crawling over otherwise still geometry, worst where the picture is darkest". |
| DLSS5VKLayer, same files | The overall two-leg structure (capture leg encodes a proxy and downloads it; a helper round trip happens in between; a compose leg uploads the answer and recomposes onto the swapchain image) and the general shape of the luminance-ratio transfer (`ratio = originalLuma / proxyLuma` below the proxy's level, a headroom-preserving inverse above it, blended via `lerp` with a saturated strength, hue corrected in OkLab) — NeuralForge already had an equivalent, independently-derived version of this exact ratio math (`composition::gpu::UpgradeToneMap`/the "classic" non-`carry_delta` dispatch path, built earlier from the RenoDX design below) that had simply never been wired into the live per-present hot path. Confirming DLSS5VKLayer's own resolve function uses the same shape of math is what justified routing the hot path through it instead of `carry_delta`, rather than porting new math wholesale. |
| [DLSS5VKLayer PR #22](https://github.com/bmitch87/DLSS5VKLayer/pull/22) (not yet merged upstream at the time this was read), commit `4aa730c0` ("Bound the fence waits, including the two in the game's present path") — description read, no code copied (the two codebases' fence/wait plumbing don't share enough shape to port directly) | **The idea**, applied at every `wait_for_fences` call this layer can reach synchronously from the game's own present thread (`crates/layer/src/capture.rs`, `crates/layer/src/composition/gpu.rs`): pass a generous, finite timeout (`crate::FENCE_WAIT_TIMEOUT`, 5s) instead of `u64::MAX`, and treat a timeout exactly like the ordinary Vulkan error each call site already knew how to fail open from. The commit's own reasoning is what this project adopted: a lost device already surfaces as `VK_ERROR_DEVICE_LOST`, so an infinite wait was never guarding against that — it was guarding against a driver stall that doesn't lose the device, where the call simply never returns and nothing gets logged. Read while investigating NeuralForge's own long-unexplained freeze during GTA V's loading screens (`docs/UPSTREAM_PARITY.md`, "Not done yet"); landed in v0.1.85. Extended to the helper process's own `CreateFeature` setup wait (`crates/helper/src/ngx.rs::create_feature_at`) in v0.1.86, with its own recovery: a timeout there leaks the setup command pool/fence rather than free them, since a bounded wait's timeout (unlike the old unconditional wait) no longer guarantees nothing is still in flight. The helper's several `device_wait_idle` calls were deliberately left unbounded in v0.1.86 (and remain so in v0.1.87) — `device_wait_idle` has no timeout in the Vulkan API to begin with. `frame.rs`'s per-frame evaluate/transfer waits, also left unbounded in v0.1.86 for the same "load-bearing safety invariant" reason, were bounded in v0.1.87 once a way to preserve that invariant without a timeout swap alone was worked out: `FrameResources::stalled`, set on a genuine timeout, latches the whole instance so nothing touches its `cmd`/`fence` again and `main.rs`'s `retire_frame_resources` leaks it instead of destroying it on the next resize. |
| [DLSS5VKLayer PR #22](https://github.com/bmitch87/DLSS5VKLayer/pull/22), commit `0b41c98c` ("A frame trace that is still readable after it has been useful") — description read, no code copied (a lock-free ring buffer is not the implementation here; see below) | **The idea** of an always-on, per-process ring of recent pipeline-stage markers, dumped only when a stall is actually detected, so a real freeze is diagnosable from the log instead of guessed at: `crates/layer/src/breadcrumbs.rs` (v0.1.86). Upstream's own version (and this idea's ultimate source, credited in upstream's commit message) is a lock-free 64-entry ring with relaxed atomics, framed from LCPD15/DXL's `FreezeWatchdog.h`/`.cpp` (AGPL-3.0) — description only, no code taken from either LCPD15/DXL or DLSS5VKLayer. NeuralForge's implementation is a fresh, simpler one: a `Mutex`-guarded fixed array, not lock-free atomics, since this project's marker rate (once per pipeline-stage transition, not once per hardware event) doesn't need it. |

## Ported functions (upstream parity work, from 0.3.1-1 / commit 117c953)

Each function ported or re-implemented from DLSS5VKLayer's source is listed here with the
upstream file it came from. Nothing listed here lands in
`crates/layer/src/composition/color.rs`, which remains the one clean-room file.

| Function here | Upstream file / function |
|---|---|
| `crates/helper/src/ngx.rs::NgxTuning`, `set_create_tuning` | `core/ngx_snippet.cpp` `NgxTuning`, `NgxSetCreateTuning` (create-time-only tuning block) |
| `crates/helper/src/ngx.rs::maintain_feature` | `helper/main.cpp` `MaintainPasses` / `TuningFor` (compare by value, debounce by `rebuild_settle_ms`, destroy then recreate). Single-feature only until multipass lands. |
| `crates/helper/src/ngx.rs` per-evaluate `Sharpness` write, `PerfQualityValue = 3`, `NEURAL_FORGE_SKIP_NVAPI` handling | `core/ngx_snippet.cpp` `NgxSetSharpness`, `NgxCreatePass`, `NgxLoadAndInit` |
| `crates/helper/src/guard.rs`: DBG_PRINTEXCEPTION exclusion, guarded-hit cap (24), rip/fault/module-range logging (`register_module`, `describe`) | `core/guard.cpp` `GuardVeh`, `RegisterModuleRange`, `DescribeRange`. The setjmp side deliberately differs: upstream's unwinding `setjmp` form is not used here (`_setjmp(buf, NULL)`). |
| `crates/helper/src/ngx.rs`: 64x64 minimum feature size (`MIN_FEATURE_DIM`), rebuild on size change | `core/ngx_snippet.cpp` `kMinW`/`kMinH`; `helper/main.cpp` `EnsureNeural`. Upstream's re-run of `NgxLoadAndInit` on resize is not copied. |
| `crates/layer/shaders/compose.comp` mode 2 `hue_trust` | `layer_linux/src/dlssnr/dlssnr.hlsl` `hueTrust` / `kQuantFloor` |
| `crates/layer/src/capture.rs` synchronous present (capture, wait for this frame's answer, compose onto this frame) | `layer_linux/src/layer.cpp` `ProcessPresent`'s blocking round trip; the bounded wait, heartbeat liveness check and fail-open on a late answer are this project's |
| `crates/layer/shaders/compose.comp` compare views (side by side with letterbox and zoom, wipe, swap, divider) | `layer_linux/src/dlssnr/dlssnr.hlsl` `gCompareMode`/`gCompareSplit`/`gCompareZoom`/`gCompareSwap` |
| `crates/layer/shaders/compose.comp` mode 2 colour trust (`colour_trust`) and ratio smoothing (`ratio_smooth`) | `layer_linux/src/dlssnr/dlssnr.hlsl` `gColourTrust` (chroma-swing bound) and `gRatioSmooth` (`gainSmooth`/`gainSharp`); defaults 2 and 1 from `common/shm_protocol.h` |
| `crates/layer/shaders/compose.comp` transfer modes (`transfer`, `model_small`), `CubeScaleResidual`, `SoftKneeLuminance`; the small-proxy upload and enlargement in `composition/gpu.rs` | `layer_linux/src/dlssnr/dlssnr.hlsl` `gTransfer` 0/1/2, `CubeScaleResidual`, `SoftKnee`. Matched residual and its cube scaling are hhkbble's; native + edit is xenmods' DLSSNR-Cost-Scaler technique (MIT, no code copied), via upstream |
| `crates/layer/src/capture.rs` `meter_white` (tile peak luminance on a 64x64 grid, 90th percentile, lit-fraction acceptance) | `layer_linux/src/dlssnr/dlssnr.hlsl` mode 4 and `layer_linux/src/shaders/meter_reduce.comp`. Runs on the CPU over the synchronous present's captured frame, sampled 8x8 per tile every 8th frame, with smoothing upstream does not have |
| `crates/layer/shaders/compose.comp` debug views 4/5 (`debug_view`) and view 5's amplification (`debug_scale`) | `layer_linux/src/dlssnr/dlssnr.hlsl` `gDebugView == 4/5` (colour-trust engagement, pre-bound colour) and `gDebugScale`. Views 0-3 are this project's own (see `composition::apply`'s CPU reference), now also reachable through the GPU path for any game, including one that blits into its swapchain. `debug_scale_bits` was already a declared, initialised header field with no reader anywhere -- this is the first thing that reads it |
| `crates/layer/src/hotkey.rs` (`key_code_from_name`, the key-name table, evdev keyboard discovery/drain, XInput2 raw-key selection, backend order and `NEURAL_FORGE_HOTKEY_BACKEND`/`NEURAL_FORGE_TOGGLE_KEY` overrides) | `layer_linux/src/hotkey.cpp`, `hotkey.h` (`KeyCodeFromName`, `Hotkeys::OpenEvdev`/`RescanEvdev`/`OpenX11`/`PressedEvdev`/`PressedX11`) |
| `crates/layer/src/device.rs` NVIDIA vendor gate; `crates/layer/src/lib.rs` `duplicate_copy`, `note_vk`/`device_lost` | `layer_linux/src/layer.cpp` device creation vendor check (~l.914), `DuplicateLayerCopy` (~l.619), `NoteVk` (~l.1375). Swapchain admission (`surface_usage.rs`) is deliberately not upstream's: it does not OR `TRANSFER_SRC|DST` unconditionally. |
| `crates/layer/shaders/compose.comp` mode 2 ghost guard (`ghost_guard`) | The idea of taking the relighting ratio from the neighbourhood instead of the pixel is `dlssnr.hlsl`'s `gRatioSmooth`/`gainSmooth` (five taps, off by default). The stale-answer motion weighting around it (5x5 stride-3 worst-mismatch mask against the proxy, soft-thresholded) and the wider neighbourhood are this project's own, designed against measured doubling of sign text in GTA V at 4K. |

## Credits carried with the ported material

- **OptiScaler / Dagherbou** (GPL-3.0): upstream assigns `dlssnr.hlsl`'s lineage to
  [OptiScaler](https://github.com/cdozdil/OptiScaler) and
  [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR). The GPL-3.0
  text is `third_party/optiscaler/LICENSE`.
- **RenoDX / clshortfuse** (MIT): the composition design; the notice that must ship with any
  build is `third_party/optiscaler/RenoDX_ATTRIBUTION.txt`.
- **hhkbble**: the matched residual and its cube scaling (transfer mode 1), from a multi-pass
  pull request against the OptiScaler fork.
- **xenmods** ([DLSSNR-Cost-Scaler](https://github.com/xenmods/DLSSNR-Cost-Scaler), MIT): the
  native + edit technique (transfer mode 2). No code was copied.

`third_party/` and `LICENSE` ship inside the AppImage (`usr/share/doc/neural-forge/`), together
with `THIRD_PARTY_CRATES.md` (the statically linked Rust crates and their licences).

## What was previously taken clean-room (still accurate, unaffected by the above)

| Source | What's taken |
|---|---|
| [RenoDX](https://github.com/clshortfuse/renodx) (clshortfuse) — MIT | The color composition design: the two-branch luminance/headroom rule, the OkLab hue-correction step, and the reversible neutral-axis gamut compression. This is RenoDX's own DLSS 5 addon design, MIT-licensed, read and reimplemented directly from RenoDX's own public source. |
| Björn Ottosson ([bottosson.github.io/posts/oklab](https://bottosson.github.io/posts/oklab/)) | The OkLab conversion matrices, published as public reference constants. |
| Public domain / standard color science | sRGB transfer function, SMPTE ST.2084 (PQ) encode/decode, Hunt-Pointer-Estevez LMS conversion — textbook formulas, attributable to no one project. |
| Public domain / standard resampling literature | The Lanczos, Catmull-Rom, Mitchell-Netravali, and Kaiser-windowed-sinc kernels used for the supersampling downscale leg — implemented from their mathematical definitions. |

## What's still not taken

Since 0.1.80 the "native + edit" transfer mode is adopted (see the table above); its additive
rule and guard shape come from
[xenmods/DLSSNR-Cost-Scaler](https://github.com/xenmods/DLSSNR-Cost-Scaler), MIT, "no code is
copied", reached through DLSS5VKLayer's `dlssnr.hlsl`. The
NGX caller-identity spoof (`crates/helper/src/spoof.rs`) remains an independent
reimplementation of the same generic PE import-table-hook technique, not read from
DLSS5VKLayer's C++ — see [README.md's Legal section](README.md#legal) for why that
mechanism carries its own separate legal exposure regardless of either project's
license.

## License texts

This project's own code is AGPL-3.0-or-later — see [LICENSE](LICENSE). DLSS5VKLayer is
AGPL-3.0; adapting its composition strategy and confirming its transfer math directly
is why this project's license has to be AGPL-3.0 (upstream's own license requires it
for the derived code). RenoDX's MIT license applies to the design
this project's color composition math is independently derived from; xenmods'
DLSSNR-Cost-Scaler (MIT) is referenced above for provenance only, matching
DLSS5VKLayer's own attribution of it, and no code from it appears in this repository.
