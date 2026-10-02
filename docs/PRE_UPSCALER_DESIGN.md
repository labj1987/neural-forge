# Running the model before the game's upscaler: design

Status: **approved by Alex 2026-10-02, HDR included. Layer and helper sides are built**, behind
`NEURAL_FORGE_PREUPSCALE` (off by default); see the two "Implementation" sections at the end.
E1 run on the rig 2026-10-02: NGX accepts `Hdr=1` but returns a clamped, broken answer for the raw scene-linear frame; E2-E3 not run (see "Rig results" at the end). Phases 2 and 3 of the original 2.0 plan are paused.

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

## Implementation (helper)

Built 2026-10-02 (section 3, "The model's side"). Nothing sends RGBA16F until the layer's half
lands; with 8-bit proxies the helper behaves exactly as before.

### What changed

- **Format.** `frame::color_format(RGBA16F)` is `R16G16B16A16_SFLOAT` (the multipass working
  format), so Color and Output are created in the proxy's own format and the upload (staging
  buffer or the imported shared-memory region) and the download into the answer region are
  plain copies at 8 bytes per pixel. `FrameResources` sizes its staging buffer and imports by
  `proxy_format::bytes_per_pixel`; for 8-bit proxies every size, copy and image is what it was.
- **HDR signalled to NGX at creation.** `ngx::create_feature_at` writes `DLSSNR.Hdr=1`,
  `DLSSNR.SDR=0` for an RGBA16F slot (`Hdr=0, SDR=1` for 8-bit), `AutoExposure=1` and
  `Feature_Flags` unchanged, and logs `[ngx] feature WxH hdr=1: DLSSNR.Hdr=1 DLSSNR.SDR=0
  AutoExposure=1` before `CreateFeature` (whose own line now ends in `hdr=1`). The features are
  keyed by `hdr::FeatureKey { width, height, hdr }`: `maintain_passes` rebuilds every pass when
  the key changes (`[helper] frame 1708x960 hdr=0 -> 1708x960 hdr=1; rebuilding N pass(es)`),
  and a failed first build blocks retries per key, not per size.
- **Motion vectors.** The flow input is built on the GPU (`GpuFlow::estimate` blits Color down
  into a `B8G8R8A8_UNORM` image). A session built for an HDR frame (`GpuFlow::new(.., hdr)`)
  first runs `shaders/hdr_to_flow.comp` over Color (per channel: NaN and negatives to 0,
  `x / (1 + x)`, sRGB encode) into a full-size `R8G8B8A8_UNORM` image, and the blit reads that.
  Sessions are keyed by size, quality and HDR-ness, so a format switch builds a new one (which
  also drops the flow's reference frame). `optical_flow_rig_check --hdr` exercises it on the rig.
- **Scene cuts.** `scene::thumbnail` (moved out of `optical_flow.rs` so it is tested natively)
  decodes an RGBA16F frame's halves (`hdr::f16_to_f32`) and applies the same tone map on the CPU
  (`hdr::tonemap_u8`) before averaging; the 40-level threshold is unchanged. Thumbnails of
  different classes are never compared.
- **History.** Everything in `history.rs` applies unchanged; in addition each request's format
  class goes through `HistoryGap::note_format`, and the first evaluate after a class change
  resets the model's history (`Stale::FormatChanged`, logged `resetting model history (proxy
  format changed ...)`). The rebuilt feature and frame resources reset it as well.
- The model input is always the raw scene-linear frame; the tone map only feeds the flow and
  the thumbnail. The GPU tone map and the CPU one agree exactly (0 LSB difference) on all 65536
  half bit patterns, checked on lavapipe during development.

### Running the model on a dumped frame (experiment E1)

`crates/protocol/examples/trigger_helper_roundtrip.rs` plays the layer against a running
helper. With `--rgba16f` it sends a raw frame (W*H*8 bytes, little-endian halves R, G, B, A),
as `RGBA16F` on slot 0, `--repeat` times (default 4: the first request after a size or format
change can come back as an echo while the feature builds). Each round prints whether the answer
was evaluated or echoed; for the last one it prints per-channel min/max/mean, NaN and Inf counts
for input and answer, and the mean |answer - input| per channel, and writes the answer (same
format and size) to `--out`. Odd sizes are padded by repeating the last column/row and cropped
back. Exit status 1 when the last answer was an echo.

The helper on the rig must be this build (`scripts/deploy-rig.sh`), started by the CLI with no
game running (the layer would drive slot 0 too), with the effect on:

```bash
cargo +stable build --release -p neural-forge-protocol --example trigger_helper_roundtrip
scp target/release/examples/trigger_helper_roundtrip lordnikon:/tmp/nf-roundtrip
ssh lordnikon 'export NEURAL_FORGE_SHM=/tmp/neural-forge-1000/shm.bin NEURAL_FORGE_UID=1000; /tmp/nf-roundtrip --rgba16f /tmp/gta-color-1708x960.rgba16f --width 1708 --height 960 --out /tmp/gta-color-1708x960.answer.rgba16f --repeat 8'
ssh lordnikon 'grep -E "\[ngx\] (feature|VULKAN_CreateFeature|EvaluateFeature)|resetting model history|\[frame\]" ~/.local/state/neural-forge/helper.log | tail -20'
```

What E1 reads from it: `[ngx] VULKAN_CreateFeature(18) -> 0x1 ... hdr=1` (NGX accepted the HDR
feature), `evaluated` rather than `ECHO`, no NaN/Inf in the answer, the answer's range of the
same order as the input's (scene-linear, not clamped to [0, 1] and not collapsed), and a mean
|answer - input| that is clearly non-zero but small next to the channel means.

## Implementation (layer)

Built 2026-10-02 in `crates/layer/src/preupscale.rs`, wired into `device.rs`, `lib.rs` and
`entry_points.rs`. Not yet run on the rig: everything below about GTA is what the code expects
from the probe, not a measurement.

### Modes

`NEURAL_FORGE_PREUPSCALE` (with `NEURAL_FORGE_ENABLE=1`):

| Value | What happens at the DLSS submit |
|---|---|
| unset / `off` | Nothing. The hooked-command list, the resolved entry points and every hot path are as before; no waits are added anywhere. |
| `dump` | Once (the first hold after the colour input is identified and the depth and motion-vector layouts are known from the game's barriers, normally a frame later; after 120 DLSS submits without them, colour only), and again for each `shmctl capture` (`capture_request=1`): capture colour, depth and motion vectors, forward the game's submit unchanged, and write `~/.local/share/neural-forge/captures/preupscale-<ms>/` with `colour.rgba16f` (padded size), `depth.r32f` (`depth.raw` for a non-float depth), `mvec.rg16f`, `meta.json` (width, height, padded size, formats, frame number), `colour-preview.png` (`x/(1+x)` per channel, sRGB-encoded) and `exposure.json` (every registered 1x1 float image, DLSS's exposure input among them: format, layout, raw bytes and values; a storage image with no barrier seen yet is read as `GENERAL`, flagged `layout_assumed`). Files are written off the submit thread. |
| `identity` | Capture the colour input into slot 0's proxy region, then copy the same bytes back into it. The helper is not called. Measures the hold's own cost; the picture must be unchanged. |
| `model` | Capture, hand the frame to the helper (`width` = padded width, `proxy_format` = RGBA16F), wait up to 30 ms (or until the helper stops being alive), copy the answer back into the colour input with the padding cropped. Every frame (`model_interval` is ignored). With `enabled` off (F11, the GUI, `shmctl set enabled 0`), `apply_model` off or the model reported unavailable, nothing is held. |

Any non-off mode turns on the NVX tracking (the probe's hooks, production-grade, without its
logging): `vkCreateImage`/`vkCreateImageView` descriptions, views registered through
`vkGetImageViewHandleNVX`/`vkGetImageViewHandle64NVX`/`vkGetImageViewAddressNVX`, command buffers
that record `vkCmdCuLaunchKernelNVX` (also through executed secondaries, cleared at
`vkBeginCommandBuffer`/`vkFreeCommandBuffers`), and the identified images' layouts from barriers,
committed in submission order. `NEURAL_FORGE_PROBE_NGX` still controls the probe's own lines.

### Identification

At a launch-bearing submit, when the registered set or the swapchains changed: the colour input is
the registered RGBA16F storage image (2D, single-sample) whose extent equals a registered depth
image's, with a registered RG16F image of the same extent, and smaller than the largest swapchain
(DLAA is refused that way). Several candidates: lowest handles. Logged once per change:
`[preupscale] colour input: image 0x... (1707x960 R16G16B16A16_SFLOAT ...), depth ..., motion vectors ...`
or `[preupscale] no DLSS input among N registered views ...; waiting`.

### The hold

At `vkQueueSubmit`/`vkQueueSubmit2`, the first command buffer marked launch-bearing is the hold
point (one hold per call, one at a time per device: the device's state lock is held throughout).
The call is re-issued through the next layer as: the batches before the launch batch plus, if the
launch buffer is not first in its batch, the buffers before it with the batch's wait semaphores
(no fence); the layer's capture batch (carrying the launch batch's wait semaphores when the launch
buffer was first, timeline values kept, stage masks widened to `ALL_COMMANDS`); the write-back
batch; then the launch buffer onward with the batch's signal semaphores, the call's later batches,
and the application's fence. Batches without a launch buffer are unchanged. `VkSubmitInfo` pNext
chains other than `VkTimelineSemaphoreSubmitInfo`, and `VkSubmitInfo2` chains other than a latency
present id, a performance-query pass or a frame-boundary marker, are not held (logged once).
The module doc comment of `preupscale.rs` has the full dependency-chain argument.

The capture copies the colour input (in `GENERAL`, no layout change) into slot 0's proxy region,
imported as a buffer (`VK_EXT_external_memory_host`, zero-copy; a mapped buffer of the layer's own
plus a CPU copy when the import is unavailable), at the padded size: an odd width gets its last
column duplicated into the padding column, an odd height its last row (4K Balanced is 2227x1253).
The capture fence is waited on (bounded, `note_fence_wait`); the write-back is not, its fence is
checked at the next hold or before the post-upscaler path next runs. Capture and write-back carry
GPU timestamps. A hold is skipped (frame forwarded untouched) when: the layer is not engaged yet
(loading screens), the colour input's last committed barrier left it outside `GENERAL`, a slot-0
request is still with the helper (the post path's, or a hold that ran over budget), the post
path's zero-copy capture is still writing slot 0, or anything fails.

In model mode, while a hold happened in the last 500 ms the post-upscaler compose is skipped for
that device's presents (the game's frame is presented as DLSS made it), so the model is not
applied twice. Without holds (DLSS off, DLAA, a game without DLSS, the toggle off) the post path
runs exactly as before.

### Running it on the rig

Select DLSS at Quality or Balanced (not DLAA) with frame generation off, then:

```bash
scripts/gta-bench.sh --host lordnikon preupscale-dump VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=dump 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
scripts/gta-bench.sh --host lordnikon preupscale-identity VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=identity 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
scripts/gta-bench.sh --host lordnikon preupscale-model VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=model 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

Model mode needs a helper that accepts an RGBA16F proxy (`DLSSNR.Hdr=1`); with one that refuses
it, every hold is a miss and the frame goes to DLSS untouched. For identity, the post-upscaler
path still runs unless NR is off (`--set enabled=0`), so set that to measure the hold alone.

What to look for in the launch log (`grep -F '[preupscale]'`):

- `[preupscale] mode <mode> in pid N` and `[preupscale] device ...: vkGetImageViewHandleNVX present, ..., vkQueueSubmit2 present`.
- The identification line (above).
- `[preupscale] resources for WxH (padded PWxPH) built: zero-copy (SHM regions imported)`.
- Model mode: `the model runs before the upscaler; the post-upscaler compose is off while frames are held`.
- Every 300 holds: `[preupscale] mode=... extent=WxH (padded ...) holds=N hold_ms median=... capture_gpu_ms median=... writeback_gpu_ms median=... misses=... (total ...)`.
- Misses, at most one line per 5 s: `[preupscale] frame went to DLSS untouched: <why> (...)`.
- Dump mode: `[preupscale] dump of the DLSS input (WxH, frame N) written to <dir>: colour.rgba16f, depth.r32f, mvec.rg16f, colour-preview.png, meta.json`.

`neural-forge-cli shmctl status` shows `preupscale_state` (0 off, 1 waiting for DLSS input,
2 holding), `preupscale_extent`, `preupscale_hold_ms` (the last hold's CPU time) and
`preupscale_misses` (shared-memory protocol 9).

### Not verified without NVX hardware

The tests run the split, the identification and the hold machinery itself (lavapipe and the local
GPU, with validation and synchronization validation clean), but no device here has
`VK_NVX_image_view_handle`/`VK_NVX_binary_import`, so the following is untested until the rig run:
that GTA's registrations and launches arrive as the probe saw them through these hooks; the launch
buffer's position in its batch; whether vkd3d-proton's launch-bearing `VkSubmitInfo2` carries a
pNext structure outside the accepted list; how long the capture fence wait is in practice (it also
drains everything queued before it); and whether vkd3d-proton's waits are ever wait-before-signal
on a timeline the same thread signals later (the bounded 5 s wait would then stall once and fail
open).

## Rig results (E1-E3)

Run 2026-10-02 on lordnikon (RTX, 2560x1440@288 HDR bt2100, GTA V Enhanced DLSS Balanced, frame
generation off, script mods off), main at 2aa02bf deployed with `scripts/deploy-rig.sh`
(protocol 9; the old v8 `shm.bin` had to be removed, nothing had it open). **E1 failed on the raw
scene-linear input, so E2 and E3 were not run**, as the plan says.

### E1: HDR input to the model

Dump run (`pu-dump-1`, `NEURAL_FORGE_PREUPSCALE=dump`, post path on as usual): the layer side
worked first time.

- `[preupscale] device ...: vkGetImageViewHandleNVX present, vkGetImageViewAddressNVX present,
  vkQueueSubmit2 present` on GTA's devices (absent on the launcher's).
- `[preupscale] colour input: image 0x791e6ea941c0 (1485x836 R16G16B16A16_SFLOAT TRANSFER_SRC |
  TRANSFER_DST | SAMPLED | STORAGE | COLOR_ATTACHMENT), depth ... D32_SFLOAT_S8_UINT, motion
  vectors ... R16G16_SFLOAT; swapchain Some((2560, 1440))` (logged twice: the depth image was
  re-registered once). Balanced is **1485x836**, padded to 1486x836.
- `resources for 1485x836 (padded 1486x836) built: zero-copy (SHM regions imported)`, then the
  dump (`frame 1`, about 40 s after launch, the benchmark's first scene: Grove Street towards
  downtown, daylight). The benchmark completed normally (61.2 fps pass 4, the post path at model
  every 2nd frame).

The dumped colour: luma p50 6.1, p90 13.0, p99 19.8, max 95 (scene-linear, pre-exposure; the
game's exposure value is not dumped). Alpha is not 1 (0..3, mean 0.72).

Round trip (`trigger_helper_roundtrip --rgba16f`, no game running): NGX **accepts** the HDR
feature: `[ngx] feature 1486x836 hdr=1: DLSSNR.Hdr=1 DLSSNR.SDR=0 AutoExposure=1`,
`VULKAN_CreateFeature(18) -> 0x1 ... size=1486x836 hdr=1`, and evaluates (first evaluates ~30 ms,
then **4.1 ms** steady at 1486x836). Note: with `--repeat 8` every answer was an echo, because the
size/format change rebuilds the feature after `rebuild_settle_ms` (250 ms) and 8 requests take
~90 ms; `--repeat 32+` is needed.

| Input sent | Answer range | Answer mean R/G/B | mean abs diff (display domain, x/(1+x)) | Verdict |
|---|---|---|---|---|
| raw scene-linear (as designed) | 0 .. 1.0 (clamped) | 0.27 / 0.26 / 0.26 vs input 6.0 / 6.5 / 8.2 | 0.53-0.55 | **garbage**: grey wash, detail smeared |
| scaled by 1/p99 luma (1/19.8) | 0 .. 0.9995 | 0.29 / 0.33 / 0.43 vs 0.30 / 0.33 / 0.41 | 0.044-0.048 | plausible, but 5.4% of pixels clip at 1.0 (highlights lost) |
| tone-mapped x/(1+x) per channel (0..1) | 0 .. 0.9995 | 0.74 / 0.76 / 0.75 vs 0.74 / 0.75 / 0.75 | 0.026-0.037 | plausible enhancement, no clipping |

No NaN/Inf in any answer. The answer's alpha is always 1.0. With `Hdr=1` the model's output is
clamped to [0, 1] in every case: it does not produce scene-linear values above 1, so the design's
"answer replaces the colour input directly" cannot work on the raw frame. The diffs of the two
working variants are structured (foliage, building edges, window detail, a mild local-contrast
change), not noise or a global shift.

Images (tone-mapped x/(1+x), sRGB; diff = |answer - input| x 8 in the tone-mapped domain) were made
on the dev machine and are not in the repo.

**Verdict: E1 fails as designed** (NGX takes `Hdr=1` but returns a clamped, broken picture for
scene-linear input). A tone-mapped input (x/(1+x) before the model, y/(1-y) on write-back, or the
same with an exposure scale) does give a plausible answer, so the design should be re-cut around
tone-mapping the input first (the plan's own fallback), with the inverse applied on write-back. Open
questions for the re-cut: whether to tone-map with the game's 1x1 exposure value, whether the
inverse (y/(1-y)) amplifies the model's changes in highlights too much (the blue channel's
scene-linear mean moved by -1.2 in the sky), and whether `Hdr=0, SDR=1` on the tone-mapped
input behaves better than `Hdr=1` (not tried).

### E2, E3

Not run (E1 decides; see above). The layer's identification and dump path are proven on GTA; the
hold, identity write-back and model mode are unmeasured.

## E1b: the HDR encode

Run 2026-10-02 on lordnikon (same settings as E1: Balanced 1485x836, mods off). Question: which
encode of the scene-linear DLSS input makes the model give a correct answer, and what is its
inverse for the write-back.

### What was added for it

- **Layer, dump mode** also reads every registered 1x1 float image (DLSS's exposure input among
  them) at the hold and writes `exposure.json` (format, layout, raw bytes, values). Everything
  else in the dump is unchanged.
- **Helper:** `NEURAL_FORGE_HDR_FLAGS` (debug, read at feature creation): `hdr` (default for an
  RGBA16F proxy: `Hdr=1, SDR=0`), `sdr` (`Hdr=0, SDR=1` for an RGBA16F proxy), `autoexp0`
  (`AutoExposure=0` and no `AUTO_EXPOSURE` feature flag). The `[ngx] feature` line logs the flags
  and the variable. `neural-forge-cli restart` passes its environment to the helper, so
  `NEURAL_FORGE_HDR_FLAGS=sdr neural-forge-cli restart` is enough; a plain restart undoes it.
- **`trigger_helper_roundtrip --rgba8 FILE`**: sends a raw 8-bit frame as an RGBA8 proxy (the
  8-bit reference).
- **`scripts/hdr_encode.py`** (numpy + Pillow; run on the rig): `info`, `encode`, `judge`
  (inverse, display PNGs of input and answer through the same display mapping, an x8 diff, and
  the statistics below), `montage`.

### The exposure value

GTA registers one **1x1 R16_SFLOAT** image (in `GENERAL`) and several 1x1 R32G32B32A32_SFLOAT
images (no `TRANSFER_SRC`, not readable; NGX's own). The R16F value is the game's exposure and
it adapts per scene: **0.1282** at Grove Street (dump A, frame 3), **0.1581** on Vinewood
Boulevard (dump B, frame 8397, taken with `shmctl capture` 219 s into the benchmark).

The convention is **multiply**: `exposed = scene * e`. Scene luma p50 5.9 / 3.0 becomes exposed
p50 0.75 / 0.47 (p99 2.5 / 3.3); dividing gives p50 46 / 19, which is nonsense.

### Encodes tried

All through the rig's real helper (`--repeat 64`, no game running), same frame, both dumps.
`w` is the paper white in exposed units (OpenDLSS-NR's `paperWhite`): `v = scene * e / w`.
"clamp" = pixels with any channel >= 0.999 in the model domain. "Edit" = mean |answer - input|
in a common display (opendlss shoulder at w = 3, sRGB), in 8-bit levels; "hi-grad" = edit on the
top 10% gradient pixels / edit elsewhere; "luma out/in" = scene-linear mean luma after the
inverse / before. A / B per cell.

| Encode | sent clamp % | answer clamp % | edit (8-bit) | hi-grad | signed edit B | luma out/in | identity round trip, px >1% off |
|---|---|---|---|---|---|---|---|
| E-a `opendlss`, w = 1 (exposure as is) | 39.9 / 31.4 | 1.8 / 1.1 | 17.9 / 15.5 | 0.67 / 0.64 | -0.070 / -0.053 | **0.79 / 0.77** | 38% |
| **E-a `opendlss`, w = 3** | 0.08 / 0.88 | 0.03 / 0.01 | **9.1 / 8.5** | 1.13 / 1.00 | -0.010 / -0.008 | 1.00 / 0.96 | 0.08% |
| E-b = E-a with `NEURAL_FORGE_HDR_FLAGS=sdr` | | | bit-identical answers to E-a | | | | |
| E-a, w = 3, `autoexp0` | | | bit-identical (A) | | | | |
| E-c `opendlss-linear`, w = 3 | 0.03 / 0.69 | 0.00 / 0.00 | 10.6 / 9.7 | 1.32 / 1.11 | +0.010 / +0.002 | 0.99 / 0.96 | 0.04% |
| E-c `opendlss-linear`, w = 1 | 36.3 / 30.4 | 0.4 / 0.08 | 17.1 / 16.4 | 0.80 / 0.69 | -0.053 / -0.052 | 0.81 / 0.77 | 34% |
| E-d `lumaknee` (encode.comp's SoftKnee), w = 3 | 0.70 / 14.6 | 0.02 / 0.01 | 9.0 / 8.3 | 1.13 / 1.00 | -0.009 / -0.005 | 1.00 / 0.99 | 0.64% |
| E-e `reinhard` x/(1+x) after exposure, w = 1 | 0 / 0 | 0 / 0 | 12.8 / 11.9 | 0.94 / 0.78 | **-0.026 / -0.027** | 0.94 / 0.90 | 0% |
| 8-bit reference: E-a w = 3 quantised, RGBA8 proxy | 0.11 / 1.0 | 0.04 / 0.03 | **9.1 / 8.4** | 1.16 / 1.00 | -0.010 / -0.008 | 1.00 / 0.96 | 26% (8-bit) |
| 8-bit reference at w = 1 | 41 / 32 | 2.8 / 1.8 | 19.4 / 16.2 | 0.65 / 0.63 | -0.074 / -0.053 | 0.76 / 0.74 | 58% (8-bit) |

Evaluate time 4.1-4.2 ms for every variant. No NaN/Inf; every answer is in [0, 0.9995].

**The creation flags do nothing.** `Hdr=1`/`SDR=0` vs `Hdr=0`/`SDR=1`, and `AutoExposure` 1 vs 0,
give **bit-identical** answers for the same input (checked pairwise on both dumps; the model is
deterministic: two runs of the same input are bit-identical too). The few pairs that differed
(1.3-2/255) were the runs where a feature rebuild echoed part of the 64 requests, so the model had
fewer history frames. So the model always treats its input as a display-referred [0, 1] picture
and clamps its output there, whatever it is told; the encode alone decides the result. (This
also explains E1's raw-input grey wash.) The 16F proxy with E-a w = 3 and the RGBA8 proxy of the
same picture differ by 1.1-1.2/255, i.e. quantisation plus history length.

### Visual verdict (images in the dev machine's scratchpad `pu2/`, not in the repo)

- **E-a w = 1** (the game's exposure straight into the OpenDLSS-NR shoulder): the encoded picture
  is blown out (a third of the pixels sit on the shoulder's top), and the model "fixes" that by
  pulling everything down: the answer is darker and greyer than the exposed picture, highlight
  texture is invented on the clipped billboard skin, and the write-back would lose 20-23% of the
  frame's mean luminance. Rejected; the exposure is right, the paper white is not.
- **E-a w = 3**: the input looks like a normal, well-exposed GTA frame (blue sky, correct reds),
  and the answer looks like the model's usual edit: a bit more local contrast and texture on
  skin, foliage and edges, greens slightly toward olive, reds slightly deeper. No grey wash, no
  hue shift, not washed out. Statistically indistinguishable from the 8-bit reference.
- **E-b**: identical to E-a (see above).
- **E-c (linear, w = 3)**: works, but the model sees a darker picture (sRGB-encoded midtones at
  ~0.23 instead of ~0.51), and its edit is larger, more edge-weighted and brightens slightly
  (+0.01 on every channel). Further from the 8-bit reference than E-a.
- **E-d (luminance knee, w = 3)**: same result as E-a where they agree, but the knee's peak divide
  pins 14.6% of dump B (the bright sky) at 1.0 in one channel and is not invertible there. At
  w = 3 the per-channel shoulder barely engages (p99 exposed ~0.9-1.1), so its hue risk is not
  visible; no reason to prefer the luminance knee.
- **E-e (x/(1+x), linear)**: no clamping and an exact inverse, but the model sees a flat, pale,
  dark-midtone picture: the answer is hazier, blue drops (-0.027), saturated reds turn toward
  crimson, and the frame loses 6-10% of its mean luminance. Rejected.

### Recommendation

The model input (RGBA16F, alpha 1):

```
e     = the game's 1x1 R16_SFLOAT exposure image (read on the GPU each frame; it adapts)
v     = max(scene, 0) * e / 3                                    (paper white 3, exposed units)
y     = v                                                        if v <= 0.75
        0.75 + 0.25 * (1 - exp(-5.770780 * (v - 0.75)))           otherwise   (per channel)
input = sRGB_OETF(y)
```

The write-back (per channel):

```
y     = sRGB_EOTF(clamp(answer, 0, 1))
y     = min(y, 1 - 1e-4)
v     = y                                                        if y < 0.75
        0.75 - ln(1 - (y - 0.75) / 0.25) / 5.770780              otherwise
scene' = v * 3 / e
```

The flags stay `Hdr=1, SDR=0, AutoExposure=1` (they make no difference). Open points for the
build: (1) the paper white 3 was picked so the exposed median lands at 0.16-0.25 and the clamp
stays under 1%; 2.5-4 would do as well, it wants one look in the game. (2) The inverse tops out at
v = 2.1 (y = 1 - 1e-4), i.e. scene = 6.3 / e (about 40-50 here), and half floats near 1.0 already
lose distinction above v of about 1.8: pixels whose input was at the shoulder's top (0.06-0.9%
here: sky, sun) cannot come back. The write-back should keep the original scene value where the
encoded input was >= 0.999. (3) The exposure image is read before
DLSS runs and was in `GENERAL` (the first dump had not seen a barrier on it yet, the second had).
