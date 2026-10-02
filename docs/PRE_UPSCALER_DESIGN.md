# Running the model before the game's upscaler: design (needs Alex's go-ahead)

Status: **proposal, 2026-10-02. Nothing here is built.** It follows from the probe in
`docs/PRE_UPSCALER_PROBE.md`. The 2.0 plan said a positive probe stops the program here until
Alex decides; Phases 2 and 3 (working scale on the zero-copy path, the pipelined present) are
paused until then.

## What the probe established (GTA V Enhanced, 2560x1440, DLSS SR on)

- Under vkd3d-proton and DXVK-NVAPI, DLSS reaches Vulkan as CUDA kernels
  (`vkCmdCuLaunchKernelNVX`) on image views registered through `vkGetImageViewHandle64NVX`.
  A Vulkan layer sees all of it.
- DLSS's inputs are stable, registered once: colour **1707x960 R16G16B16A16_SFLOAT**
  (storage; scene-linear HDR, the kernels are the `hdr` variants), depth 1707x960 D32S8,
  motion vectors 1707x960 R16G16_SFLOAT, a 1x1 exposure value. Output 2560x1440 RGBA16F.
  (1707x960 is DLSS Quality; Alex's setting is Balanced, about 1490x838.)
- One DLSS evaluation per frame: 14 launches in one game command buffer, in one
  `vkQueueSubmit` on the present's queue, about 5 submits before the present.
- In 9006 of 9006 launch-bearing command buffers, nothing touches the colour input before
  the first launch (no copy, clear, attachment, barrier or write-access memory barrier); the
  buffer opens with one dispatch, then `hiluma_engine_input_depthinv_mvlo_hdr_v2_rel`, the
  DLSS input kernel. The image is in `GENERAL` layout throughout. vkd3d records every buffer
  fresh (no reuse, no special begin flags).

So the colour input is final when the launch-bearing submit starts, and the layer can do its
work **at that submit** without touching the game's command buffers.

## Why it is worth doing

| | Today (after the upscaler) | Before the upscaler |
|---|---|---|
| Pixels the model sees | 2560x1440 (3.69 Mpx) | ~1490x838 at Balanced (1.25 Mpx), 1707x960 at Quality |
| Model input | the finished 8-bit frame, HUD included, told "SDR" | the scene-linear HDR frame, no HUD, the input the model is designed for |
| Motion vectors and depth | estimated by optical flow | the game's own (available in the same submit) |
| Edit applied to | the displayed frame, by our composition | DLSS's input; DLSS upscales the enhanced frame |

Cost estimate per model frame at 1440p Balanced (evaluate interpolated from 6.2 ms at
1920x1080 and 10.2 ms at 2560x1440; not measured): evaluate ~4.2 ms, copies and hand-off
~2-2.5 ms, so ~6.5 ms of added GPU time against today's ~14.6 ms. With the game's own GPU
work at ~7.2 ms per frame (93.2 fps at 67%, DLSS Balanced already in that number):

- model every frame: ~13.7 ms per frame, **about 70 fps** (today: 41.9 every frame, 61.6 every
  2nd frame);
- 4K Balanced (2227x1253, 2.8 Mpx): evaluate ~8 ms, ~10.5 ms added against ~11.5 ms of game
  work: **about 45 fps every frame** (today 32.8 at every 2nd frame).

These are arithmetic on the 2.0 baseline, not measurements. The model's GPU time and the game's
are assumed additive (the same assumption the 2.0 plan makes).

## Design

### 1. Recognising the moment

With the feature on, the layer keeps the NVX hooks the probe uses (registration and launch
counting only, never touching kernel parameters):

- `vkGetImageViewHandle64NVX`/`vkGetImageViewHandleNVX`/`vkGetImageViewAddressNVX`: the
  registered-view table. The **colour input** is the registered RGBA16F storage image whose
  extent equals the registered depth image's and is smaller than the swapchain; depth and
  motion vectors are the registered D32S8 / RG16F images at the same extent. Keyed by extent,
  re-derived when the set changes (resolution or DLSS preset change).
- `vkCmdCuLaunchKernelNVX`: marks the command buffer as launch-bearing (first launch per
  buffer). Kernel names are only logged, never relied on.
- `vkQueueSubmit`/`vkQueueSubmit2`: a submit carrying a launch-bearing buffer is the hold
  point. If that buffer is not the first in its batch, the batch is split in two at it: the
  first part keeps the original wait semaphores, the second keeps the signal semaphores and
  the fence. (The probe did not record the buffer's position in its batch; the first build
  step logs it.)

### 2. What happens at the hold point (mechanism A)

On the game's queue, before forwarding the game's submit:

1. **Capture submit.** A full memory dependency on earlier work on the queue
   (`ALL_COMMANDS -> TRANSFER`), then copy colour (GENERAL, no layout change), the depth aspect
   and the motion vectors into the shared-memory regions imported as device memory (the
   existing zero-copy import, `DirectCapture`'s mechanism). Fence.
2. **Round trip.** Wait for the capture fence (bounded, `note_fence_wait`), bump `seq_req`,
   wait for the helper's answer (bounded; see 4).
3. **Write-back submit.** Copy the answer from the imported answer region into the colour
   input (GENERAL), then a barrier making it visible to everything after
   (`TRANSFER -> ALL_COMMANDS`).
4. **Forward** the game's submit unchanged.

Steps 1 and 3 are ordinary submits on the same queue, so submission order plus their barriers
order them against the game's work. No command is injected into a game buffer, no event is
waited on inside a buffer, nothing in the game's recording changes.

`vkQueueSubmit` here is called from vkd3d-proton's submission thread, not the game's render
thread. Waiting there stalls the GPU queue, not the game directly, until vkd3d's queue backs
up. The helper's model work runs on the GPU during that stall, so the stall is mostly the
model's own time rather than idle time. How much of the CPU hand-off (today ~1 ms) shows as
idle GPU is measured in step E2 below.

### 3. The model's side

- **Format.** The helper accepts an RGBA16F proxy (today it refuses one) and creates the
  feature with `DLSSNR.Hdr=1`, `SDR=0`. Its internal work format is already RGBA16F, so the
  upload becomes a plain copy.
- **Size.** 1707 is odd and the helper's feature needs even sizes: capture into a 1708-wide
  region with the last column duplicated, crop on write-back.
- **Depth and motion.** `DLSSNR.Depth` / `DepthInverted` and `DLSSNR.MVec` / `MVecScaleX/Y`
  exist in the feature's parameter set. First build: keep today's optical flow (run at the
  render size). Then try the game's own vectors, which needs their scale and sign convention:
  DLSS SR was given a scale by the game through NVAPI, which the layer can't see, so it is
  derived by comparing the game's vectors with our flow on a pan.
- **Exposure.** `AutoExposure=1` as today; the game's 1x1 exposure value is available if
  needed.
- **Composition.** The answer replaces the colour input directly (the equivalent of today's
  composition bypass): there is no 8-bit encode to undo, and DLSS's own temporal filter sits
  after it. The post-upscaler composition is off while this path is active.

### 4. Model interval: every frame

DLSS SR accumulates its history across frames. If only every second input were enhanced, DLSS
would blend enhanced and plain frames, halving the effect and making it shimmer. This path
therefore runs the model on **every frame**; the cost estimate above is for that. If an answer
is late (over budget), the frame goes to DLSS untouched and the miss is logged sampled; a
budget of about twice the measured round trip, never more than 30 ms.

### 5. Fail-open

Every failure (no registered views, extent mismatch, capture or write-back failure, helper not
running, answer over budget, protocol mismatch) forwards the game's submit untouched. At most
one submit is held at a time. A game or a frame without DLSS SR keeps today's post-upscaler
path unchanged, so nothing is lost for other games or for DLAA/native settings.

## What changes

- **Layer.** NVX tracking (from the probe, production-grade), the submit hold, capture and
  write-back of a 16F image, odd-width padding. The synchronous swapchain path stays as the
  fallback.
- **Protocol.** A proxy-format value for RGBA16F already exists; add regions or slot-b reuse
  for depth and motion vectors when they are used, and a flag saying which path a frame came
  from. `SHM_VERSION` bump.
- **Helper.** RGBA16F input, `Hdr=1`, padding-aware sizes, optional depth and game vectors.
- **GUI.** None; it is automatic when DLSS SR is detected (Alex never touches settings).

## Experiments before any build (all unattended, in order)

- **E1, HDR input.** Take one 1707x960 RGBA16F colour input from GTA (layer dump), feed it to the
  helper with `Hdr=1`: does NGX evaluate it, and is the answer finite, in range, and
  plausibly an enhancement (diff image)? If NGX refuses HDR or produces garbage, the design is
  re-cut around tone-mapping the input first.
- **E2, the hold alone.** Hold the DLSS submit and write back an unmodified copy (identity),
  model not called: frame cost of the hold, validation and sync validation clean, picture
  bit-identical. This bounds the layer's own overhead.
- **E3, the full loop**, model every frame, Quality and Balanced, three GTA benchmark runs each:
  real fps, GPU %, timestamps. Gate: at least today's 61.6 at Balanced with the model every
  frame.
- **Alex at the screen**, after E3: GTA with the new path vs today's, F11 toggling, daylight and
  night, judging whether the enhancement looks right on the HDR input and whether DLSS's
  temporal accumulation shimmers with it.

## Risks

- NGX's DLSS-NR feature may not accept HDR input, or may want a paper-white/exposure hint it
  does not expose (E1 decides).
- The model sees a jittered frame every frame (DLSS jitters the camera sub-pixel); its own
  temporal behaviour may fight DLSS's (E3 and Alex's eyes decide).
- vkd3d-proton back-pressure: holding its submission thread could stall the game's CPU if
  vkd3d's queue fills (E2/E3 timing shows it).
- Other games: only GTA V Enhanced is probed. DX12 games through Streamline under vkd3d should
  look the same; native Vulkan games call NGX through the same NVX extensions, but each needs a
  probe run before it is claimed.

## Decision for Alex

1. Go ahead with E1-E3 (unattended), then build this as 2.0 if E3's gate holds; or
2. Shelve it and continue with Phases 2 and 3 of the original plan (post-upscaler improvements:
   cheaper working scale at 4K, the pipelined present).

1.1.0 (measurement tools, timestamps, log rotation, the probe) ships either way.
