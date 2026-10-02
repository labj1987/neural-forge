# Running the model before the game's upscaler: design

Status: **approved by Alex 2026-10-02, HDR included. Layer and helper sides are built**, behind
`NEURAL_FORGE_PREUPSCALE` (off by default); see the two "Implementation" sections at the end.
Nothing is measured on the rig yet (E1-E3 below). Phases 2 and 3 of the original 2.0 plan are paused.

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
| `dump` | Once (the first hold after the colour input is identified and the depth and motion-vector layouts are known from the game's barriers, normally a frame later; after 120 DLSS submits without them, colour only), and again for each `shmctl capture` (`capture_request=1`): capture colour, depth and motion vectors, forward the game's submit unchanged, and write `~/.local/share/neural-forge/captures/preupscale-<ms>/` with `colour.rgba16f` (padded size), `depth.r32f` (`depth.raw` for a non-float depth), `mvec.rg16f`, `meta.json` (width, height, padded size, formats, frame number) and `colour-preview.png` (`x/(1+x)` per channel, sRGB-encoded). Files are written off the submit thread. |
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
