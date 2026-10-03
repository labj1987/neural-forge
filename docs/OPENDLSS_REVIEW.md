# Review against OpenDLSS-NR (2026-10-01)

> **Note (2026-10-02):** two rows below are out of date. GPU timestamps now exist (1.1.0,
> `layer_capture_gpu_ms`/`layer_compose_gpu_ms`). And an HDR path was built: since 2.0 the model
> runs before DLSS on the game's HDR input, with the game's exposure value and a paper white of 3
> in an encode that follows this review's proxy shoulder ([PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md),
> "E1b"). The helper's separate per-stage fence waits became one wait per request in 2.0.

Neural Forge compared with [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (MIT,
commit `9d08f41`, docs `frame.md`, `execution.md`, `numerics.md`, `network.md`), an open
reimplementation of the DLSS 5 neural-rendering network and the pipeline around it. Read for
ideas only, alongside three closed or Windows-only projects:
[DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD) (release notes),
[OptiScaler_DLSSNR v0.1.2](https://github.com/Dagherbou/OptiScaler_DLSSNR/releases/tag/v0.1.2-dIssnr)
and [bgfx-dlss5-nr](https://github.com/andrewmd5/bgfx-dlss5-nr). No code or constants were
copied from any of them.

## The one structural difference

OpenDLSS-NR runs the network itself, so it builds the 16 input lanes, keeps the history, and
does the temporal blend in its own shaders. Neural Forge hands an 8-bit swapchain frame to
NVIDIA's runtime through NGX (`crates/helper/src/frame.rs`). Everything between "colour in"
and "answer out" happens inside NVIDIA's runtime, so most of `frame.md` describes work Neural
Forge never sees. Those rows are marked **not applicable**. What Neural Forge controls is what
it tells NGX (the parameters, the motion vectors, and `DLSSNR.Reset`), plus the capture before
the hand-off and the composite after it.

Status: **confirmed** means the reference practice was checked against Neural Forge's code and
the gap (or the match) is real. **Not applicable** means the item lives inside NVIDIA's runtime
for Neural Forge. **Unknown** means it can't be settled without NVIDIA's runtime source or a test
that doesn't exist yet.

## Fidelity: input, history and composite

| Item | OpenDLSS-NR | Neural Forge today | Gap | Recommendation | Risk | Evidence needed | Status |
|---|---|---|---|---|---|---|---|
| Proxy transform and shoulder | `v = scene / max(paperWhite, 0.05)`, per-channel shoulder above 0.75 (`0.75 + 0.25(1 - exp(-5.770780(v - 0.75)))`), sRGB, f16. Built from linear HDR scene radiance. | NGX is told `DLSSNR.SDR=1`, `Hdr=0`, `AutoExposure=1` and is handed the swapchain's 8-bit sRGB bytes (`ngx.rs` `create_feature_at`). At working scale 1 (Alex's setting) the frame goes over untouched (direct capture). Below 1, `encode.comp` applies a **luminance** soft knee with `exp(-(l - 0.75) / 0.25)` and a peak-channel step, over an already tone-mapped frame. | Different curve (luminance vs per-channel, slope 1 vs 1.44 at the knee), but it only runs below 100% model resolution. The 8-bit swapchain frame is already a display proxy, so a second shoulder over it is not what the reference describes either. | Keep. The luminance knee was adopted on purpose: a per-channel knee shifts hue on saturated highlights, which upstream measured as a green cast in GTA V. | Medium (changes every highlight) | A/B captures at 75% model resolution on saturated highlights | Confirmed (different, not adopted) |
| Feature lane layout | 16 lanes: noise, constant 1, proxy, reprojected history, style/128, local tone, structure triple. | Built by NVIDIA's runtime from the NGX tuning block (`DLSSNR.Style`, `LocalToneStrength`, `LocalStructureStrength`, `SkinStructureStrength`, `UseAutoMask`). Neural Forge's defaults (skin -1 = follow structure, auto-mask on) match the reference's `NrControls`. | None that Neural Forge can act on. | None. | n/a | n/a | Not applicable |
| History source (previous output, not previous scene) | Lanes 7-9 are the reprojected previous **output**. | Inside NGX. Neural Forge's own frames between model runs (model every Nth frame) reuse the last answer on the new frame in the composite; that is not the network's history. | None. | None. | n/a | n/a | Not applicable |
| History filter | Five-tap Catmull-Rom, clamped to the valid rectangle; bilinear softens over time. | Inside NGX. Neural Forge resamples only spatially (working scale below 1: GPU blit down, answer enlarged in the composite), never across time. | None for history. | None. | n/a | n/a | Not applicable |
| Truncation vs rounding of stored history | Truncated toward zero to the half grid. | Inside NGX. Neural Forge's own 8-bit stores round (Vulkan UNORM conversion). | None. | None. | n/a | n/a | Not applicable |
| Has-history flag / reset | A first frame, a reset, or a pixel whose previous position was off screen gets blend weight 0, and lanes 7-9 take the current proxy. A flag, separate from the motion value. | NGX takes only a whole-frame `DLSSNR.Reset`. Before this review it was set on a new feature and on a scene cut (motion on only). It was **not** set when the effect was switched off and on, when the layer passed frames through (warm-up, loading screens), or after a failed evaluate. The feature kept its last picture, and the next answers blended with it. | **Real gap.** | **Implemented:** `crates/helper/src/history.rs`. Reset NGX history and the optical-flow reference on the first evaluate after a request went unevaluated, or after more than 500 ms without an evaluate. | Low: a reset costs one frame without history, and fires only after a gap. | Unit tests (`history.rs`); GTA V benchmark on the rig: one `resetting model history (7848 ms since the last answer)` at a loading gap | Confirmed, fixed |
| Per-pixel off-screen history | Computed from the renderer's motion (previous position off screen means no history). | NGX has no per-pixel validity input. Neural Forge's motion vectors come from optical flow, which has no notion of off screen. Whether NVIDIA's runtime derives it from the motion vectors is not documented. | Can't be expressed through NGX. | None. | n/a | NVIDIA runtime behaviour | Unknown |
| blendScale | Learned cap on the history weight, 0.7397 in the shipped model. | Inside the model NGX loads. | None. | None. | n/a | n/a | Not applicable |
| Composite formula | `neural = clamp(proxy + rgb/4)`, blend with history by `sigmoid(logit) * blendScale`, then a luminance-ratio tone upgrade with Oklab hue transfer (`upgradeToneMap`). | The network-side blend is inside NGX. Neural Forge's own composite (after NGX) is the same luminance-ratio rule with Oklab hue transfer (`compose.comp` `UpgradeToneMap`, mode 0), plus guarded modes 1 and 2 for the direct and encoded paths. | None. The post-NGX math already matches the reference's `upgradeToneMap`. | Keep. | n/a | n/a | Confirmed (matches) |
| Style presets | Exposure, contrast and saturation offsets in an HSL operator, scaled by local tone. | `DLSSNR.Style` goes to NGX, which applies the preset itself. | None. | None. | n/a | n/a | Not applicable |
| Exposure / paper white | `paperWhite` divides the HDR scene. OptiScaler_DLSSNR reads the game's exposure texture to set it. | No paper-white parameter exists in NGX's parameter set. The swapchain frame is already exposed and tone mapped, so there is no exposure to recover. `AutoExposure=1` is set at creation. | None for SDR swapchains. Relevant only if an HDR path is ever built. | Note for the HDR follow-up. | n/a | n/a | Not applicable |

## Performance: hand-off and scheduling

| Item | OpenDLSS-NR | Neural Forge today | Gap | Recommendation | Risk | Evidence needed | Status |
|---|---|---|---|---|---|---|---|
| Command buffer pre-recording and parity | NR work pre-recorded once per (history parity, NR on/off) into four secondary command buffers. Parameters via `vkCmdUpdateBuffer`. | Layer: capture and compose command buffers are reset and re-recorded every present (`capture.rs` `record_capture_commands`, `gpu.rs` `record_temporal_delta_into_image`). Parameters go in push constants. Helper: NGX records its own evaluate into the helper's command buffer every frame, so it can't be pre-recorded. | Small. Recording a handful of copies and one dispatch costs microseconds, not the milliseconds the network costs. | Follow-up only if a profile shows recording time on the present thread. | Low | CPU profile of `capture::run` | Confirmed (not worth it now) |
| Descriptor and buffer reuse | Device-local buffers keyed by label and reused across re-records. Nothing allocated per frame. | No Vulkan object is created per frame on either side. Everything is cached by size or image and rebuilt only on resize. A few small Rust heap allocations per frame remain (helper: about 30 `CString`s and 16 `format!` names per pass; layer: two small `Vec`s). | Negligible. | None. | n/a | n/a | Confirmed (matches) |
| Barrier scope | Compute-to-compute only; transfer stages made the driver flush caches between dispatches, tens of µs each. | Neural Forge's work is mostly copies, so transfer barriers are inherent. The helper brackets NGX with `TRANSFER -> ALL_COMMANDS` and `ALL_COMMANDS -> TRANSFER`; the compose has 9-11 barriers. Several could be narrowed (`TRANSFER -> ALL_COMMANDS` before present to `BOTTOM_OF_PIPE`, `UNDEFINED` sources to `TOP_OF_PIPE`). | Small: tens of µs at most, against 12-13 ms of model time. | Follow-up, bundled with any future compose rework. | Low | GPU trace (Nsight) | Confirmed (not worth it now) |
| Separate submits and host waits | One command stream, no host synchronisation. | Helper: upload, optional flow, evaluate and download are separate submits, each with a blocking fence wait. Layer: capture and compose submits, a 100 µs fence-status poll and a 100 µs shared-memory poll. Measured hand-off overhead was about 0.2 ms per model frame (`[sync]`, 0.1.95). | About 0.2-0.4 ms per model frame, under 2% of a composite. | Follow-up. AGENTS.md forbids re-applying the reverted capture/composition fence changes, so this needs its own design. | Medium | Matched `[sync]` traces | Confirmed (follow-up) |
| Timestamp readback lag | Six GPU timestamps per frame, read two frames later, never stalling. | No GPU timestamps anywhere. All timings are host `Instant`s around waits Neural Forge already makes. | No stall exists to remove. GPU timestamps would only make the numbers more precise. | Follow-up for diagnostics only. | Low | n/a | Confirmed (no stall) |
| Model resolution scaling | Below about 768x768 launch- and occupancy-bound (2.1x the work in 1.04x the time); 1080p to 4K linear (2560x1440 12.6 ms, 1920x1080 7.77 ms on an RTX 4070 SUPER). | Working scale is applied in the layer (GPU blit down, answer enlarged in the composite). | None. See the measurement below. | Keep 100% (Alex's setting). | n/a | GTA V runs below | Confirmed (measured) |
| Queue priority | DLSS-NR-on-AMD's "priority queue mode" reports +9% under heavy load (no details). | The helper's queue has priority 1.0 and no global priority. The RTX 5070 driver exposes `VK_KHR_global_priority`. | None that can be closed: NVIDIA's Linux driver answered a HIGH request with `ERROR_NOT_PERMITTED_KHR` (tested on the rig, 615.71.09). Raising it needs elevated privileges for the helper's Wine process. | Dropped. Lowering the game's own queues to LOW from the layer is the only unprivileged variant; not tried. | Medium (changes the game's scheduling) | A/B with the game's queues at LOW | Confirmed (not possible unprivileged) |
| Fast arithmetic mode | DLSS-NR-on-AMD's Fast mode trades exactness for speed in its own kernels. OpenDLSS-NR has none (byte-identical output). | NGX `PerfQualityValue` = 3 (Balanced), the same as upstream. The kernels are NVIDIA's. | Can't be changed from outside the runtime. | None. | n/a | n/a | Not applicable |
| PTX counter chaining | Barrier-free chaining through device counters (NVIDIA-only, `VK_NV_cuda_kernel_launch`). | Neural Forge doesn't run kernels. | n/a | Out of scope. | n/a | n/a | Not applicable |

## Measurements

GTA V Enhanced built-in benchmark, pass 4 (the long free-roam pass), 2560x1440, frame generation
off, model every 2nd frame, motion vectors on. Runner: `gta-bench.sh` (unattended), real fps from
GTA's own frame-time file.

**Correction (later the same day): these runs had GTA's script mods loaded.** One of them, the
Enable All Interiors .NET script, holds GTA's main thread every frame and caps the game at about
63 fps with the GPU 45% busy. That cap, not ray tracing and not Neural Forge, is why NR looked
almost free in the first table below. Without the mods (see the last table), NR off is 93.6 and
NR on 61.6, the same proportion as the 1.0.0 baselines.

| Run | Real fps | GPU | Model evaluate | `[sync]` per model frame |
|---|---|---|---|---|
| 1.0.0, NR off | 62.8 | 45% | - | - |
| 1.0.0, NR on, model 100% | 58.6 | 85% | 9.7 ms | total 19.5-20.6 ms, zero-copy |
| 1.0.0, NR on, model 75% | 54.8 | 74% | 6.3 ms | total 19.3-21.0 ms, copy path (`copy_out` 2.8 ms + `rest` 1.6 ms) |
| 1.0.0, NR on, model 50% | 59.5 | 66% | 3.6 ms | total 14.2-15.6 ms, copy path |
| 1.0.1, NR on, model 100%, 4 runs | 58.1 / 58.5 / 58.1 / 58.4 | 84-85% | 10.1-10.3 ms | total 18.6-19.9 ms |

Cause of the NR-off ceiling, found by elimination (NR off, Smooth Motion off unless noted;
temporary `settings.xml` edits restored and hash-checked after each run; mods disabled for a run with
`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b` or by setting one file aside, all 36 mod files verified
identical afterwards):

| Run | Real fps | GPU |
|---|---|---|
| Mods loaded (Alex's setup) | 63.3 | 47% |
| Mods loaded, Smooth Motion on | 62.4 real / 124.8 displayed | 52% |
| Mods loaded: Reflex off / VSync on / native res / `descriptor_heap` / latency 3 / 4 images | 62.6-63.7 | 46-61% |
| Mods loaded, ray tracing off | 83.2 | 38% |
| **All mods off** | **93.6** | **67%** |
| Native trainer off | 63.1 | 46% |
| ScriptHookVDotNet off | 92.0 | 66% |
| **Enable All Interiors script off** | **93.0** | **67%** |
| iFruitAddon2 script off | 71.9 | 52% |
| No mods, NR on | 61.6 | 89% |
| No mods, render 1920x1080 | 93.2 | 52% |
| No mods, render 3840x2160 | 83.0 | 95% |

Without the mods the game is CPU-limited at about 93 fps (1080p and 1440p give the same frame rate)
and saturates the GPU at 4K. The GPU, driver, PCIe link, RAM (DDR4-3600, dual channel) and CPU
(7-Zip 88 GIPS) all check out.

**Model resolution.** The evaluate itself scales as OpenDLSS-NR predicts: 56% of the pixels take 64%
of the time and 25% take 36%, so the saving shrinks as the raster gets smaller. In Neural Forge the
frame rate doesn't follow even that. Below 100%, the layer leaves zero-copy direct capture for the
copy pipeline, whose CPU copies on the present thread cancel most of the saving (75% is slower than
100%). Model resolution is the wrong lever here. Model interval (skipping frames) is the one that
works, as both the reference and the 0.1.83 measurements say.

**Hand-off.** `wait_answer - helper` is 1.0-1.3 ms per model frame. About 0.7 ms of it is the
optical-flow estimate (logged as `[mvec] estimate 0.63-0.73 ms`, not part of the published helper
time). The rest is the two poll loops: a 200 µs sleep measures 255 µs under this Proton (probe run in the helper's own
prefix), and the layer polls every 100 µs. No per-frame Vulkan object creation was found on either
side.

**1.0.1 vs 1.0.0.** 58.1-58.5 against 58.6: no change beyond run-to-run noise. The reset fired once
per benchmark, at the loading gap between scenes (`resetting model history (7848 ms since the last
answer)`).

## Not changed, and why

- **Per-channel proxy shoulder.** Different from Neural Forge's luminance knee, but the knee only
  runs below 100% model resolution, and the per-channel curve shifts hue on saturated highlights.
- **Queue priority.** Refused by the driver without privileges (above).
- **Pre-recorded command buffers, narrower barriers, GPU timestamps.** Each is worth microseconds
  against a 10 ms model evaluate. Kept as follow-ups.
- **Replacing the poll loops.** At most about 0.3 ms per model frame, and AGENTS.md rules out
  re-applying the reverted fence changes. Needs its own design.
- **Fixing the copy path below 100% model resolution.** A real cost (about 4 ms per model frame),
  but a larger change than this review covers. Alex runs at 100%.
- **San Andreas DE and Crimson Desert benchmarks.** Not run: both need someone at the rig to get
  into gameplay, and the only shipped change adds no per-frame work (GTA V confirms it).
