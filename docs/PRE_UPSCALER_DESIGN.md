# Running the model before the game's upscaler: design

Status: **the default since 2.0** (`model` mode with `NEURAL_FORGE_PREUPSCALE` unset, on devices with
`VK_NVX_image_view_handle`; `NEURAL_FORGE_PREUPSCALE=off` restores the 1.1.0 post-upscaler path
everywhere). Approved by Alex 2026-10-02, HDR included; see the two "Implementation" sections at the
end, and "2.0: the default" at the very end.
E1 run on the rig 2026-10-02: NGX accepts `Hdr=1` but returns a clamped, broken answer for the raw scene-linear frame (see "Rig results (E1-E3)"). E1b found the encode that works (the game's exposure, paper white 3, a per-channel shoulder, sRGB); the layer now applies it on the GPU before the model and inverts it on write-back (model mode, plus a `roundtrip` mode that checks the transform alone). E2-E3 run 2026-10-02 (see "Rig results (E2-E3)"): the hold alone is cheap, but the model every frame gives 50.5 fps at Balanced, **below the 61.6 gate**. Phases 2 and 3 of the original 2.0 plan are paused. The ~4.8 ms of "hand-off latency" turned out to be the helper's CPU scene-cut thumbnail under Wine; with it fixed and the helper's submissions chained, model every frame gives **66.1 fps** (see "Hand-off latency"), **above the gate**.

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
  first runs `shaders/hdr_to_flow.comp` over Color into a full-size `R8G8B8A8_UNORM` image, and
  the blit reads that. (Built as a tone map, `x / (1 + x)` then sRGB, for the raw frame; since
  the layer sends its encoded proxy, already display-referred in [0, 1], it only clamps: NaN to 0,
  the value clamped to [0, 1], see the fixes from the 2.0 review below.)
  Sessions are keyed by size, quality and HDR-ness, so a format switch builds a new one (which
  also drops the flow's reference frame). `optical_flow_rig_check --hdr` exercises it on the rig.
- **Scene cuts.** `scene::thumbnail` (moved out of `optical_flow.rs` so it is tested natively)
  decodes an RGBA16F frame's halves (`hdr::f16_to_f32`) and quantises them the same way on the
  CPU (`hdr::encoded_u8`, through a table, `hdr::encoded_u8_half`) before averaging; the 40-level
  threshold is unchanged and means the same on both classes. Thumbnails of different classes are
  never compared.
- **History.** Everything in `history.rs` applies unchanged; in addition each request's format
  class goes through `HistoryGap::note_format`, and the first evaluate after a class change
  resets the model's history (`Stale::FormatChanged`, logged `resetting model history (proxy
  format changed ...)`). The rebuilt feature and frame resources reset it as well.
- The model gets the proxy's half floats exactly as the layer sent them: since E1b that is the
  layer's encode (exposure, paper white, shoulder, sRGB), not the raw scene-linear frame. The 8-bit
  conversion only feeds the flow and the thumbnail. The GPU pass and the CPU function agree on all
  65536 half bit patterns within one step of float-to-UNORM rounding (3 of 196608 channels on
  lavapipe), checked by a layer test that runs the helper's shader
  (`preupscale::hdr::tests::the_helpers_flow_input_pass_quantises_the_encoded_proxy_without_a_tone_map`).

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

Built 2026-10-02 in `crates/layer/src/preupscale.rs` (the HDR encode and decode in
`crates/layer/src/preupscale/hdr.rs` and `crates/layer/shaders/preupscale_{encode,decode}.comp`, after
E1b), wired into `device.rs`, `lib.rs` and `entry_points.rs`. The hold, identity, the HDR encode and
roundtrip are not yet run on the rig: everything below about GTA is what the code expects from the
probe and E1/E1b, not a measurement.

### Modes

`NEURAL_FORGE_PREUPSCALE` (with `NEURAL_FORGE_ENABLE=1`):

| Value | What happens at the DLSS submit |
|---|---|
| `off` | Nothing. The hooked-command list, the resolved entry points and every hot path are as in 1.1.0; no waits are added anywhere. |
| `dump` | Once (the first hold after the colour input is identified and the depth and motion-vector layouts are known from the game's barriers, normally a frame later; after 120 DLSS submits without them, colour only), and again for each `shmctl capture` (`capture_request=1`): capture colour, depth and motion vectors, forward the game's submit unchanged, and write `~/.local/share/neural-forge/captures/preupscale-<ms>/` with `colour.rgba16f` (padded size), `depth.r32f` (`depth.raw` for a non-float depth), `mvec.rg16f`, `meta.json` (width, height, padded size, formats, frame number), `colour-preview.png` (`x/(1+x)` per channel, sRGB-encoded) and `exposure.json` (every registered 1x1 float image, DLSS's exposure input among them: format, layout, raw bytes and values; a storage image with no barrier seen yet is read as `GENERAL`, flagged `layout_assumed`). Files are written off the submit thread. |
| `identity` | Capture the colour input into slot 0's proxy region, then copy the same bytes back into it (raw, no encode). The helper is not called. Measures the hold's own cost; the picture must be unchanged. |
| `model` (also unset, the default since 2.0) | Capture the colour input **HDR-encoded** (see "The HDR encode and decode" below), hand it to the helper (`width` = padded width, `proxy_format` = RGBA16F), wait up to 30 ms (or until the helper stops being alive), decode the answer back over the colour input with the padding cropped. Every frame (`model_interval` is ignored). With `enabled` off (F11, the GUI, `shmctl set enabled 0`), `apply_model` off or the model reported unavailable, nothing is held. |
| `roundtrip` | The same encode and decode, with the answer := the encoded proxy itself; the helper is not called. Checks the transform's neutrality on the rig: the picture must come back unchanged except for half-float rounding and the clamped highlights (which are kept as they were). Ignores the toggle, like `identity`. |

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
(DLAA is refused that way). Several candidates: lowest handles. DLSS's **exposure input** is the
registered 1x1 `R16_SFLOAT` image (lowest handle if there are several; NGX's own 1x1 RGBA32F images,
which are not transfer sources, are never it). Logged once per change:
`[preupscale] colour input: image 0x... (1707x960 R16G16B16A16_SFLOAT ...), depth ..., motion vectors ..., exposure input 0x...`
(`exposure input none (no registered 1x1 R16_SFLOAT)` when missing) or
`[preupscale] no DLSS input among N registered views ...; waiting`. Without an exposure input,
model and roundtrip modes hold nothing (frames go to DLSS untouched), logged once: `no exposure
image (a registered 1x1 R16_SFLOAT) among DLSS's inputs; ...`.

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
(loading screens), DLSS's inputs were (re)identified at this very submit (no layout is known yet;
the next one is held), the colour input's last committed barrier left it outside `GENERAL`, a slot-0
request is still with the helper (the post path's, or a hold that ran over budget), the post
path's zero-copy capture is still writing slot 0, or anything fails.

### The HDR encode and decode (model, roundtrip)

The model treats its input as a display-referred [0, 1] picture (E1b), so the capture encodes and
the write-back inverts, on the GPU, in the layer's own two submissions (`preupscale/hdr.rs`,
`shaders/preupscale_encode.comp`, `shaders/preupscale_decode.comp`), with E1b's formulas exactly:

- **Capture** (one command buffer): `ALL_COMMANDS/MEMORY_WRITE -> TRANSFER|COMPUTE_SHADER` (the
  layer's padded encoded image `UNDEFINED -> GENERAL` in the same barrier); copy the exposure texel
  (in its committed layout, `GENERAL` assumed for a storage image no barrier was seen on,
  transitioned and put back otherwise) into a 16-byte host-visible storage buffer;
  `TRANSFER -> COMPUTE_SHADER`; the **encode** dispatch over the padded extent: reads the colour
  input through a storage view (`GENERAL`, read only), `v = max(scene, 0) * e / W`, per channel
  above 0.75 `0.75 + 0.25 (1 - exp(-5.770780 (v - 0.75)))`, sRGB OETF, alpha 1; the padding
  column/row reads the edge texel (as the raw copy duplicates it); `COMPUTE_SHADER -> TRANSFER`;
  copy the encoded image into slot 0's proxy region (imported, or the staged buffer); the usual
  `TRANSFER -> HOST` close. After the capture fence the CPU reads the exposure value back: zero,
  negative or not finite means no helper call and no write-back for that frame (`frame went to
  DLSS untouched: the exposure value is not usable ...`).
- **Write-back** (one command buffer): `ALL_COMMANDS|HOST -> TRANSFER|COMPUTE_SHADER` (the layer's
  padded answer image `UNDEFINED -> GENERAL`); copy the answer region (model) or the proxy region
  (roundtrip) into the answer image; `TRANSFER -> COMPUTE_SHADER`; the **decode** dispatch over the
  real extent (the padding is cropped): `y = sRGB EOTF(answer)`, `y = min(y, 1 - 1e-4)`, at or above
  0.75 `v = 0.75 - ln(1 - (y - 0.75) / 0.25) / 5.770780`, `scene' = v * W / e`; where the encoded
  input of that channel (read from the encoded image the capture wrote) was >= 0.999 the original
  scene value is kept; alpha is always the original. It reads and writes the colour input in place
  (each invocation its own pixel only); then `COMPUTE_SHADER/SHADER_WRITE ->
  ALL_COMMANDS/MEMORY_READ|MEMORY_WRITE` before DLSS. The decode also writes nothing when the
  exposure value is unusable (belt and braces; the CPU already skipped the submit).
- **Push constants** (`hdr::HdrPush`, both shaders' `Params`, 24 bytes, append only): `uvec2 size`
  (the colour input's extent), `uvec2 padded`, `float paper_white`, `uint flags` (bit 0: keep
  clamped highlights, always set).
- **Paper white** `W`: `NEURAL_FORGE_PREUPSCALE_PAPER_WHITE` (default **3.0**; a positive number up
  to 1000, anything else logs and uses 3). Logged once: `[preupscale] HDR encode: paper white 3 ...`.
- **Resources**: two padded RGBA16F device-local images (encoded, answer, ~10 MB each at 1486x836),
  the exposure buffer, two compute pipelines over one descriptor set, built on the first HDR hold
  for an extent and destroyed with the hold's other resources (after their fences). The storage
  view of the game's colour input is re-created every hold, after the previous hold's work drained,
  so a destroyed and re-created image with a reused handle is never reached through a stale view.
  Any failure to build them is a logged miss (frame untouched).
- **Timing**: `capture_gpu_ms` and `writeback_gpu_ms` span the whole command buffers, so in model and
  roundtrip modes they include the encode and the decode; roundtrip's medians minus identity's give
  the transform's own GPU cost.
- **Precision** (tests, lavapipe and an Intel iGPU, both of which truncate on the float-to-half
  store): roundtrip comes back within about 2.6e-3 relative where the encoded value is at most 0.9;
  towards the shoulder's top the inverse is steep and one step of the 16-bit proxy is worth up to
  ~4% of the scene value (encoded 0.99-0.999); channels at >= 0.999 come back bit-exact. Negative or
  NaN scene values encode as 0 and come back as 0.

`identity` is unchanged: a raw copy-through, the hold's own cost.

In model mode, while a hold happened in the last 500 ms the post-upscaler compose is skipped for
that device's presents (the game's frame is presented as DLSS made it), so the model is not
applied twice. Without holds (DLSS off, DLAA, a game without DLSS, the toggle off) the post path
runs exactly as before. (Since "Robustness: failed feature builds" below: once a hold on the
device has asked the helper, the compose stays off until DLSS has not run for 30 s, loading
screens included, and an open circuit breaker forwards DLSS submits untouched.)

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

```bash
scripts/gta-bench.sh --host lordnikon preupscale-roundtrip VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=roundtrip 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

Add `NEURAL_FORGE_PREUPSCALE_PAPER_WHITE=2.5` (or 4) to any of the model/roundtrip runs to try
another paper white.

Model mode needs a helper that accepts an RGBA16F proxy (`DLSSNR.Hdr=1`); with one that refuses
it, every hold is a miss and the frame goes to DLSS untouched. For identity and roundtrip, the
post-upscaler path still runs unless NR is off (`--set enabled=0`), so set that to measure the hold
alone.

What to look for in the launch log (`grep -F '[preupscale]'`):

- `[preupscale] mode <mode> in pid N` and `[preupscale] device ...: vkGetImageViewHandleNVX present, ..., vkQueueSubmit2 present`.
- The identification line (above).
- `[preupscale] resources for WxH (padded PWxPH) built: zero-copy (SHM regions imported)`.
- Model mode: `the model runs before the upscaler; the post-upscaler compose is off while frames are held`.
- Model and roundtrip: `[preupscale] HDR encode: paper white 3 (...)`, and the identification line
  names an `exposure input`.
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
open). For the HDR encode: that GTA's 1x1 R16F exposure image carries `TRANSFER_SRC` at every hold
(the E1b dump read it, so it did then), that a storage view of the colour input can be created
(its usage includes `STORAGE`; the probe saw `SAMPLED | STORAGE | ...`), the encode/decode GPU time
at 1486x836, and how NVIDIA rounds float-to-half stores (the tests' drivers truncate).

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

Not run at the time (E1 decides; see above); run after E1b, see "Rig results (E2-E3)" at the end.

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

## Rig results (E2-E3)

Run 2026-10-02 13:12-14:25 on lordnikon (2560x1440@288 HDR bt2100, scale 1.0; GTA V Enhanced, DLSS
Balanced 1485x836 unless noted, frame generation off, script mods off via
`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`), main at 840fc80 deployed with `scripts/deploy-rig.sh`,
helper restarted (`helper_state=4`). No remote-desktop session during any run. One benchmark
iteration per run with `scripts/gta-bench.sh`; numbers are pass 4 (117 s) from
`scripts/bench-report.py`. Hold columns are the median over the `[preupscale] mode=...` lines that
cover pass 4 (one line per 300 holds). Three runs exited early at Game Init (roundtrip, model 3,
Quality; no preupscale activity yet, no Xid) and were rerun after 5 minutes. No Xid/NVRM line in the
kernel log for the whole session.

| Config | Real fps | Displayed | GPU % | Power | Hold ms | Capture / write-back GPU ms | Misses (over budget) |
|---|---|---|---|---|---|---|---|
| `pu-nroff-1`: NR off (no layer) | 93.0 | 94.2 | 67 | 148 W | - | - | - |
| `pu-plain-cap`: layer loaded, `enabled=0`, no PREUPSCALE | 92.5 | 93.9 | 67 | 147 W | - | - | - |
| `pu-identity-1`: identity, `enabled=0` | 91.5 | 92.4 | 77 | 146 W | 4.28 | 0.68 / 0.63 | 0 |
| `pu-roundtrip-1`: roundtrip, `enabled=0` | 90.7 | 92.0 | 77 | 146 W | 4.21 | 0.69 / 0.65 | 0 (1 frame: exposure not usable, first hold) |
| `pu-post-1`: today's post path (model every 2nd frame) | **61.4** | 61.8 | 90 | 195 W | - | - | - |
| `pu-model-bal-1` | 50.5 | 50.8 | 69 | 151 W | 14.61 | 0.68 / 0.52 | 0 in pass 4 (60 total) |
| `pu-model-bal-2` | 50.4 | 50.4 | 68 | 150 W | 14.73 | 0.67 / 0.52 | 0 in pass 4 (85 total) |
| `pu-model-bal-3` | 50.6 | 50.7 | 68 | 150 W | 14.69 | 0.68 / 0.52 | 0 in pass 4 (93 total) |
| **model, Balanced, mean of 3** | **50.5** | 50.6 | 68 | 150 W | 14.7 | 0.68 / 0.52 | |
| `pu-model-q-1`: model, Quality (1707x960, padded 1708x960) | 43.5 | 43.6 | 66 | 150 W | 17.42 | 0.79 / 0.64 | ~14 around the pass 3/4 boundary (91 total) |

(Model mode's `composited/s 62.9` in `bench-report.py` comes from the loading screens between
passes, where no hold happens for 500 ms and the post path runs; it is not the pass-4 rate.)

### What the layer did

- Identification, every mode: `colour input: image ... (1485x836 R16G16B16A16_SFLOAT ...)`, depth
  D32_SFLOAT_S8_UINT, motion vectors R16G16_SFLOAT, `1x1 (exposure) ... R16_SFLOAT`, `exposure input
  0x...`, swapchain 2560x1440 (re-logged twice per run as NGX registers more 1x1 RGBA32F images);
  `resources for 1485x836 (padded 1486x836) built: zero-copy (SHM regions imported)`;
  `HDR encode: paper white 3` (also printed in identity mode, which does not encode).
- Model mode: `the model runs before the upscaler; the post-upscaler compose is off while frames
  are held`, alternating with `no hold in the last 500 ms` at each loading screen, as designed.
- The first hold of each run in model and roundtrip modes is skipped with `the exposure value is not
  usable` (the exposure image is still zero on the first DLSS frame); harmless.
- Over-budget answers (30 ms): 60-93 per model run, all in passes 0-3 and around the start of pass 4
  (scene changes), about 1% of ~7800 holds; **none in steady pass 4** at Balanced. No other errors.
- Helper per model frame (helper.log `[frame] timing`): upload 0.7 + evaluate 4.4 + download 0.5 =
  **5.7 ms** at 1486x836; 0.8 + 4.85 + 0.6 = 6.2 ms at 1708x960.

### E2 verdict: the hold alone is cheap

Identity costs 1.0 fps against the layer loaded and idle (92.5 -> 91.5; 1.5 against no layer at all),
i.e. about 0.1-0.2 ms per frame, single runs, so within about twice the run-to-run noise. Capture
and write-back are 0.65-0.7 ms of GPU each. The hold's CPU time (4.2 ms median) is mostly the
capture-fence wait, which drains the game's work queued before it, so it overlaps the game's own
GPU time rather than adding to it. Roundtrip (the HDR encode and decode, no model) is within noise
of identity: 0.01-0.02 ms more GPU per hold and 0.8 fps (one run each). The roundtrip and identity
pictures were not captured, so "picture unchanged" is not verified here; validation layers were
not run on the rig.

### E3 verdict: **gate fails**

Model every frame at Balanced: **50.5 fps** (three runs, 50.4-50.6) against the gate of 61.6 and
today's matched post path at **61.4** in the same session; Quality: 43.5. The frame time grows from
10.75 ms (NR off) to 19.8 ms, +9.1 ms, against the design's ~6.5 ms estimate. The hold is 14.7 ms:
~4.2 ms of it is the drain identity also has, ~5.7 ms is the helper's own GPU work, and the remaining
~4.8 ms is hand-off latency (fence waits, the shared-memory round trip, waking the helper). While the
submission thread is held, vkd3d-proton queues nothing new, so the GPU idles: GPU utilisation drops
to 68% (today's path runs at 90%) and power to 150 W. The cost is serial latency, not GPU throughput.
Even with zero hand-off the helper's 5.7 ms on top of ~10.9 ms would land near 60 fps, at the gate
rather than clearly above it (arithmetic, not measured). Making this path win needs the hold to stop
being synchronous (e.g. answer frame N-1's input while N renders, at the cost of a frame of latency
on the enhanced input) or a much cheaper model call.

### Picture (rough, unattended)

`shmctl capture --frames 2` cannot capture in model mode: the series capture lives in the post
path's present, which model mode skips while holding, so the request just stays pending (it had to be
cleared with `shmctl set capture_request 0`). Instead GTA's Xwayland window was grabbed with
GStreamer (`ximagesrc xid=<GTA window>`, 8-bit, MangoHud overlay included) 200 s after launch, in
`pu-model-bal-3` and in `pu-plain-cap` (layer loaded, `enabled=0`), plus the layer's own series
capture 5 s later in `pu-plain-cap` (original == composited, as expected with NR off). Images are in
the dev machine's scratchpad `pu3/` (`model-xgrab-{1,2}.png`, `plain-xgrab-{1,2}.png`,
`plain-layer-{0,1}-{original,composited}.png`), not in the repo; the rig copies were removed.

- The model-mode frames look like correct GTA frames: downtown from the air in daylight, blue sky,
  normal reds/greens on rooftops and billboards, crisp rooftop detail; no grey wash, no colour shift,
  no black or NaN blocks, no blown highlights (0.06% of pixels at 254-255, plain 0.07%).
- They are **not a like-for-like comparison** with plain: the scripted camera was at a different
  shot at the same delay (load times differ; the plain grab caught the hillside flight, the plain
  series 5 s later the downtown flight at a different angle). Mean levels are similar (model
  155/171/188, plain 148-151/167-170/182-191). Whether the model visibly enhances the frame after
  DLSS, and whether DLSS's temporal accumulation shimmers with it, is not decidable from these and is
  for Alex at the screen.

## Hand-off latency

Run 2026-10-02 14:40-15:55 on lordnikon, same settings as E2-E3 (2560x1440@288 HDR bt2100, GTA V
Enhanced DLSS Balanced 1485x836 padded 1486x836, frame generation off, script mods off, model
every frame, `NEURAL_FORGE_PREUPSCALE=model`), `scripts/gta-bench.sh` pass 4 via
`scripts/bench-report.py`. No remote-desktop session during any run; one early exit at Game Init
(`ho-before-1`, rerun after 5 minutes).

### Instrumentation (kept)

- Layer, after each 300-hold summary: `[preupscale] phases ms (median): prep= capture_wait=
  round_trip= helper_busy= handoff= writeback=`. `prep` is the hold's start to the capture submit
  returning, `capture_wait` the capture fence wait (which drains the game's queued work),
  `round_trip` `seq_req` bumped to `seq_resp` seen, `helper_busy` the helper's own wall time for
  that request (published in the new header field `helper_busy_us`, shared-memory protocol 10;
  an interval on the helper's clock, so Wine's clock never has to agree with the layer's),
  `handoff` = `round_trip - helper_busy` per hold (the helper noticing the request plus the layer
  noticing the answer), `writeback` the answer to the write-back submit returning.
- Helper, every 300 evaluated slot-0 requests: `[frame] stages ms (median) WxH: n= idle= setup=
  thumb= upload= flow= ngx_rec= eval= download= fence_waits= publish= busy=` (`idle` the loop's
  last wait step before the request, `thumb` the CPU scene-cut thumbnail, `ngx_rec` the CPU inside
  `EvaluateFeature`, `fence_waits` the helper's own fence waits within upload/eval/download).

### Before (main at 7ddc3d2 plus the instrumentation, `ho-before-1`)

50.5 fps, GPU 68%, hold 14.7-15.5 ms (medians per 300 holds over pass 4):

| Layer phase | ms | | Helper stage | ms |
|---|---|---|---|---|
| prep | 0.04 | | idle (200 us sleep under Proton) | 0.28 |
| capture_wait | 3.2-3.9 | | setup | 0.00 |
| round_trip | 11.3-11.5 | | **thumb** | **4.7-5.0** |
| helper_busy | 11.1-11.3 | | upload (submit + wait) | 0.65-0.75 |
| handoff | 0.17-0.19 | | flow (three submits + wait) | 0.44-0.48 |
| writeback | 0.04 | | ngx_rec | 0.19 |
| | | | eval (record + submit + wait) | 4.3 |
| | | | download (submit + wait) | 0.50 |
| | | | of which fence waits | 5.2 |
| | | | busy | 10.8-11.1 |

The "unexplained ~4.8 ms" was not a hand-off at all: the shared-memory hand-off both ways is
0.18 ms. It was the helper's **CPU scene-cut thumbnail** of the RGBA16F frame (every 8th pixel,
three channels tone mapped with `x / (1 + x)` and the sRGB curve, about 58000 `powf` calls): 0.6 ms
natively, but `powf` from the helper's Windows C runtime under Wine costs ~80 ns a call, and the
same code measured 4.8-5.3 ms under Wine on the dev machine. The fence waits (5.2 ms) are the GPU
work itself: upload, flow, evaluate and download add up to about that, so there was little wake-up
latency in them.

### What changed

- **Thumbnail through a table** (`hdr::tonemap_u8_half`, now `hdr::encoded_u8_half`): the tone map of all 65536 half bit
  patterns, built once at helper start (2.4 ms under Wine), then one lookup per channel.
  Bit-identical to the function (a test checks every half); the scene-cut threshold and behaviour
  are unchanged. 4.9 ms -> 0.13 ms on the rig (0.06 ms in the Wine benchmark).
- **One wait per request in the helper** (`FrameResources::evaluate`): command buffer A (upload)
  is submitted with its fence, the optical flow's three submissions follow on the same queues
  without a fence (`GpuFlow::estimate(.., FlowSync::Chained)`; `FlowSync::Wait` keeps the old
  behaviour for `optical_flow_rig_check`), command buffer B holds every pass's `EvaluateFeature`,
  the multipass steps and the download, and the helper waits once on both fences (bounded, the
  5 s `FENCE_WAIT_TIMEOUT`; a timeout latches the frame resources and the flow session as stalled,
  as before). A failure part-way (an evaluate refused at record time, a failed submit) queues an
  empty submission on B's fence and waits, so nothing is still running when the buffers are next
  reused. On the rig this saved little by itself (the waits were already mostly GPU time); it
  removes four CPU round trips per request and puts all the helper's GPU work back to back.
- **Helper loop**: for 50 ms after a request it yields between checks of `seq_req` instead of
  sleeping 200 us (`idle.rs`; back to sleeping once quiet, so an idle helper costs nothing).
  `idle` went from 0.28 ms to 0.00 ms.
- **Layer**: the answer wait spins (`spin_loop` hints, a yield every 16 checks) instead of
  sleeping 50 us between checks; still bounded by the 30 ms budget and the helper's heartbeat.
  `handoff` went from 0.18 ms to 0.00-0.01 ms.
- Not changed: the capture fence wait (still `wait_for_fences` with `FENCE_WAIT_TIMEOUT` and
  `note_fence_wait`; it is the drain of the game's own work, 3.6-4.4 ms, and shrinks only if the hold
  stops being synchronous). Nothing of the reverted capture/compose fence changes was reapplied.

### After (`ho-after-1..3`, this build)

| Run | Real fps | Displayed | GPU % | Power | Hold ms | capture_wait | round_trip | helper_busy | handoff |
|---|---|---|---|---|---|---|---|---|---|
| `ho-before-1` | 50.5 | 50.7 | 68 | 150 W | 14.7-15.5 | 3.2-3.9 | 11.3-11.5 | 11.1-11.3 | 0.18 |
| `ho-after-1` | 65.1 | 65.7 | 91 | 182 W | 9.9-10.5 | 3.7-4.4 | 6.0-6.1 | 5.9-6.1 | 0.00 |
| `ho-after-2` | 68.1 | 68.3 | 93 | 187 W | 10.0-10.4 | 3.6-3.9 | 6.04 | 6.0 | 0.00-0.01 |
| `ho-after-3` | 65.2 | 65.5 | 88 | 181 W | 10.5 | 4.4 | 6.04 | 6.0 | 0.00 |
| **mean of 3** | **66.1** | 66.5 | 91 | 183 W | ~10.2 | | | | |
| `ho-mvec0-1`: `--set mvec_enabled=0` | 67.0 | 67.7 | 92 | 185 W | 10.0 | 4.25 | 5.59 | 5.55 | 0.00 |
| `ho-post-1`: no PREUPSCALE (today's post path, model every 2nd frame) | 61.9 | 62.3 | 91 | 197 W | - | - | - | - | - |

Helper stages after: idle 0.00, thumb 0.12-0.15, upload (record + submit) 0.07, flow (submits)
0.10, ngx_rec 0.24-0.34, eval (record + submit) 0.27-0.37, the one wait 5.22-5.24, busy 5.85-5.98.
No over-budget answer in steady pass 4 (85-89 per run in total, at the loading screens and scene
changes, as before); no fence timeout or evaluate failure in the helper log.

**Optical flow** in this path costs about 0.45 ms of the hold (helper_busy 6.0 -> 5.55 ms with
motion vectors off, thumbnail included) and 1 fps in one run (67.0 against 65.1-68.1), within the
run-to-run spread. The default stays on: whether the vectors help the model's answer on DLSS's
jittered input is a picture question, not a timing one.

### Verdict

**The gate passes**: model every frame at Balanced, **66.1 fps** (65.1 / 68.1 / 65.2) against the
gate of 61.6 and today's post path (61.9 with this build, 61.4 before: no regression there). GPU utilisation is back
to 88-93%, so what remains is GPU work (the helper's ~5.2 ms and the layer's 1.3 ms of capture and
write-back on top of the game's ~10.9 ms), not waiting; the design's ~70 fps was arithmetic that
assumed the model's GPU time simply adds to the game's. The hold is still synchronous (the capture
wait drains the game's queued work, 3.6-4.4 ms), so going further means making it asynchronous.
In the post path the helper's stages are the same shape (2560x1440 RGBA8: thumb 0.18 ms, one wait
of 11.5 ms for the GPU work).
Run-to-run spread is about 3 fps (65.1-68.1), so a 1-2 fps difference between single runs is noise.

## 2.0: the default

`model` is what an unset `NEURAL_FORGE_PREUPSCALE` means (`preupscale::DEFAULT`). What changed with
that:

- **Per device.** The tracking (and the NVX hooks in the device's own hooked-command table) is only
  set up on a device whose next layer hands out `vkGetImageViewHandleNVX`, i.e. one that enabled
  `VK_NVX_image_view_handle` (logged `[preupscale] device ...: no VK_NVX_image_view_handle, ...; the
  model runs after the upscaler` otherwise). Such a device runs exactly the post path.
- **Cheap without DLSS.** On a device with NVX but no DLSS launches (a vkd3d-proton game without
  DLSS) the begin/free/execute/submit hooks read one relaxed atomic (`Tracking::armed`) and take no
  lock; the submit hook no longer copies the batch lists unless a launch-bearing buffer exists; the
  present reads the identified extent from an atomic. Image and view creation still record into the
  tracker (creation-time only). `capture_hot_path_cost_per_present` measures `capture::run`, which
  this does not touch.
- **`NEURAL_FORGE_PREUPSCALE=off`** keeps 1.1.0's hooked-command list (`probe_command_tests`), resolves
  no NVX entry point and logs no `[preupscale]` line (the smoke test's `off` pass checks the log).
- **The mode is read from the environment only** (`NEURAL_FORGE_ENABLE`, `NEURAL_FORGE_DISABLE`, the
  variable), not through `layer_enabled()`: it is first asked at instance creation, and the
  duplicate-copy decision must keep its 1.1.0 order. The hold checks `layer_enabled()` itself.
- **Captures while holding.** `shmctl capture` and `capture --frames N` are taken by the present hook
  while frames are held (they used to stay pending): the swapchain image is read back after the
  application's present waits (relayed), and each pair's original and composited are both the
  presented frame. The one-shot request becomes a series of one frame
  (`captures/series-<ms>/000000-{original,composited}.png`).
- **Logs.** The paper-white line only prints in model and roundtrip. Each 300-hold summary ends with
  `holds_per_s=` (the window's rate), which `scripts/bench-report.py` reports as "held before
  upscaler/s" instead of the post path's "composited/s" (which then only counts loading screens).
- **GUI.** The Status page's "Model placement" line: before the upscaler (holding WxH, misses), waiting
  for DLSS Super Resolution, or after the upscaler.

Left for the rig: Alex's on-screen judgement (daylight, night, F11), a three-run confirmation of the
default with no variable set, a `shmctl capture --frames 2` while holding, and a game without DLSS
(or DLAA) to see the device stay on the post path at its 1.1.0 frame rate.

## 4K and HDR output

Run 2026-10-02 15:59-17:10 on lordnikon, main at 346e280 (the build already on the rig;
`shmctl status` showed `helper_busy_ms`, so no redeploy). GTA V Enhanced, DLSS Balanced, frame
generation off, script mods off (`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`), `model_interval=2`,
`working_scale=1`, `mvec_enabled=1`. No remote-desktop session during any run. One benchmark
iteration per run with `scripts/gta-bench.sh`; numbers are pass 4 from `scripts/bench-report.py`.
Hold columns are the medians over the `[preupscale]` lines covering pass 4. Two runs exited early
at Game Init (`a4k-model-1`, `b-hdr-model-1`, both at about 90 s), and both reruns after 5 minutes
worked. No Xid at any point. The kernel log's only NVRM lines are 95 `NV_ERR_NO_MEMORY`, all at
16:48:59 during `a4k-model-sm-3`.

### A: 4K (3840x2160)

The desktop was switched temporarily to 3840x2160@144.000, scale 1.0, bt2100 with `gdctl set`
(non-persistent). GTA's settings.xml was set to 3840x2160 at 144 Hz. Both were restored
afterwards (see the end of this section).

| Run | Real fps | Displayed | GPU % | Power | Hold ms | Capture / write-back GPU ms | capture_wait / helper_busy ms | Misses |
|---|---|---|---|---|---|---|---|---|
| `a4k-nroff-1`: NR off (no layer) | 81.9 | 83.3 | 96 | 222 W | - | - | - | - |
| `a4k-post-1`: today's post path, model every 2nd frame | **28.7** | 28.9 | 93 | 195 W | - | - | `[sync]` helper 25.6 | - |
| `a4k-model-1` | 38.9 | 39.1 | 96 | 208 W | 17.2 (16.0-19.3) | 1.21 / 1.21 | 5.0-8.5 / 10.7-11.0 | 58 total, at scene changes |
| `a4k-model-2` | 39.1 | 39.2 | 96 | 209 W | 17.3 (16.0-19.3) | 1.21 / 1.21 | 5.0-8.5 / 10.7-11.0 | 58 total, at scene changes |
| **model, mean of 2** | **39.0** | 39.2 | 96 | 209 W | 17.3 | | | |
| `a4k-model-sm-3`: model + Smooth Motion (fresh helper) | 33.3 | **66.6** | 97 | 195 W | 18.6 | 1.12 / 1.18 | 5.2-8.4 / 10.9-11.7 | 0 in pass 4 (64 total) |
| `a4k-model-sm-1`: model + Smooth Motion | *invalid*: the model stopped partway (below) | 77.4 | 64 | 138 W | | | | 2713 |
| `a4k-model-sm-2`: model + Smooth Motion | *invalid*: the model was never built | 109.3 | 85 | 179 W | 9.5 | | helper_busy 2.2 (no evaluate) | 3521 |
| `a4k-model-3`: model, no SM (recovery check) | *invalid*: the model was never built | 60.8 | 81 | 176 W | 9.2 | | | 4270 |

What the layer did at 4K:

- Identification: `colour input: image ... (2228x1253 R16G16B16A16_SFLOAT ...)`, depth
  D32_SFLOAT_S8_UINT, motion vectors R16G16_SFLOAT, exposure input R16_SFLOAT 1x1, swapchain
  3840x2160 B8G8R8A8_UNORM. The resources line was `resources for 2228x1253 (padded 2228x1254)
  built: zero-copy`. DLSS's Balanced input at 4K is **2228**x1253, not 2227x1253 as assumed in
  "The hold", so only the height is padded.
- The first hold was skipped with `exposure value is not usable`, as at 1440p. `the model runs
  before the upscaler` alternated with `no hold in the last 500 ms` at each loading screen.
- Phases: prep 0.05, capture_wait 5.0-8.5 ms (the drain, up from 3.6-4.4 ms at 1440p),
  round_trip 10.7-11.0 ms with handoff 0.00. Helper stages at 2228x1254: thumb 0.23, the one wait
  10.2, busy 10.9 ms. In the post path the helper's busy time at 3840x2160 was 25.0-25.9 ms.
- Over-budget answers: 58 per model run, at loading screens and scene changes. Steady pass 4
  had a few at most.

### A verdict: at 4K the pre-upscaler path clearly wins

Model every frame before DLSS ran at **39.0 fps** (38.9 / 39.1). The post path, which runs the
model only every 2nd frame, ran at **28.7**: **+36%**, or 10.3 fps. The model's input is 2228x1254
instead of 3840x2160, about a third of the pixels, so the helper's GPU time per call drops from
~25 ms to ~10.9 ms. Frame time rises from 12.2 ms (NR off) to 25.6 ms. That is the helper's
~10.9 ms plus the layer's ~2.4 ms of capture and write-back. GPU utilisation stays at 96%, so
this is GPU work, not waiting. At 1440p the gain was 66.1 vs 61.9. At 4K it is much larger,
because the post path's cost grows with the output size while the pre-upscaler path's cost grows
with DLSS's render size. With Smooth Motion below the layer (fresh helper), real fps is 33.3 and
displayed is **66.6**. For comparison, a 2026-10-02 measurement on an older build with mods on gave
25.9 / 52.3 for NR on + SM; that was not re-measured on this build.

**VRAM is the limit at 4K with Smooth Motion.** The GPU sat at 11.7-11.9 GB of 12.2 GB through
the SM runs. In `a4k-model-sm-3`, 22 NGX feature creations at 3840x2160 failed together with the
kernel's 95 `NV_ERR_NO_MEMORY` lines at 16:48:59, then recovered. Without Smooth Motion, model
runs peak around 11.2 GB.

### Anomaly: NGX feature creation fails and stays failed

- **Trigger:** in model mode, the helper rebuilds its NGX feature at every loading screen.
  While there are no holds, the post path runs at 3840x2160 SDR; once holds resume, it runs at
  2228x1254 HDR (`frame 3840x2160 hdr=0 -> 2228x1254 hdr=1; rebuilding 1 pass(es)` and back).
  Each rebuild near full VRAM can fail.
- **What happened in `a4k-model-sm-1`:** a 3840x2160 recreation returned `VULKAN_CreateFeature(18)
  -> 0xbad00002`. From then on, **every** creation failed: 433 times in that run, at both sizes,
  with `pass 0 would not build; holding the chain at 0` and `evaluated=false` answers.
- **The failure outlived the game:** it continued through two further GTA launches,
  `a4k-model-sm-2` and `a4k-model-3`. The second of those ran without Smooth Motion, and its VRAM
  peak was the same as the runs that worked. It cleared only with `neural-forge-cli restart`.
- **Status didn't show it:** `shmctl status` kept saying `helper_state=4 (running)` and
  `model_up=1` the whole time.
- **The layer kept holding:** with no model, every hold still drained the queue, and most were
  counted as "the answer was over budget". The fps in those runs measures the hold without a
  model. The kernel logged nothing during `sm-1` and `sm-2`.
- **Not fixed here** (fixed since: "Robustness: failed feature builds" below). Two possible fixes: have the helper reinitialise NGX after repeated
  `0xbad00002`, and report "model not buildable" through shared memory so the layer stops
  holding. A third option is to avoid the 3840x2160 rebuild at loading screens in model mode.
- **Smaller oddities:**
  - `sm-1`'s benchmark.txt pass-4 average (54.1) disagrees with its frame-time file (39.3 fps).
  - Some helper `stages ms` lines labelled 2228x1254 carry ~25 ms medians. The label is the
    current size, but the 300-request window includes post-path frames.

### B: GTA's HDR output

**GTA V Enhanced on PC has no HDR output to turn on.**

- There is no HDR key in settings.xml or in either of its backups (`.bak-nf`, `.bak-restest`).
- Rockstar shipped HDR for the Enhanced edition on PS5 and Xbox Series only. PC players use
  Auto HDR, RTX HDR or the RenoDX mod.
- The profile's binary `pc_settings.bin` was not touched.
- The closest test was to let DXGI advertise HDR with `DXVK_HDR=1`, at 1440p on the default
  desktop (bt2100). settings.xml was unchanged and still hashed 8de35762....

| Run | Real fps | Displayed | GPU % | Power | Hold ms | capture_wait / helper_busy ms | Misses |
|---|---|---|---|---|---|---|---|
| `b-hdr-model-1`: `DXVK_HDR=1`, `PREUPSCALE=model` | 64.5 | 65.2 | 90 | 181 W | 9.6 (8.9-11.0) | 3.6-4.2 / 6.1 | 88 total, scene changes |
| `b-hdr-post-1`: `DXVK_HDR=1`, `PREUPSCALE=off` | 61.9 | 62.3 | 90 | 197 W | - | `[sync]` helper 11.6 | - |

- **Swapchain:** both runs used `2560x1440 fmt=B8G8R8A8_UNORM hdr=0 pass_through=false`, the
  same as without `DXVK_HDR`. The game never asks for a 10-bit/PQ or fp16 swapchain.
- **The post-upscaler path still composes:** because the swapchain is SDR, there is nothing to
  pass through. How the post path handles a real HDR swapchain (it should pass through untouched,
  per `swapchain::is_supported_format`) **is still unverified on the rig**. That needs a game
  with HDR output, probably through Proton's Wayland driver.
- **The pre-upscaler path is unaffected:** it identified 1485x836 RGBA16F and held as usual. It
  works on the render-resolution scene input, and the output format does not matter to it.
- **Fps:** 64.5 for the model path and 61.9 for the post path. These are within the spread of
  `ho-after-1..3` (66.1) and `ho-post-1` (61.9).
- **Picture:** GTA's Xwayland window was grabbed with GStreamer about 200 s after launch, twice,
  1 s apart. The images are in the dev machine's scratchpad `pu4/`, not in the repo. They show
  two correct-looking frames of the benchmark's jet flight along the highway through the hills:
  - Clear blue sky with soft cloud streaks, natural greens and tans, crisp power pylons and
    cables, and readable cars on the road.
  - No colour cast, no wash-out, and no black or NaN blocks.
  - Mean RGB 139/158/171 and 139/152/154.
  - These are 8-bit SDR grabs with the MangoHud overlay, from model mode, with no plain
    comparison frame. Like E3's grabs, they show that the pre-upscaler output is sane, not that
    it is better.

### Restored

- The desktop is back to 2560x1440@288.001, scale 1.0, bt2100 (`gdctl show`, `gdctl show -p`).
- settings.xml is byte-identical to the start (sha256 8de357621960...1a7a), and the backup made
  for this test was removed.
- Live settings: `enabled=1`, `working_scale=1`, `model_interval=2`, `mvec_enabled=1`.
- The helper was restarted at ~16:46 and is running (`helper_state=4`, `model_up=1`). GTA is not
  running.

## Robustness: failed feature builds

Fixes the anomaly in "4K and HDR output" ("NGX feature creation fails and stays failed"): one
`VULKAN_CreateFeature -> 0xbad00002` near full VRAM at 4K with Smooth Motion, after which every
creation failed (433 times, across later game launches, until `neural-forge-cli restart`), while
`shmctl status` said `model_up=1` and the layer kept holding every DLSS submit for the full 30 ms
budget.

### Cause

The helper log of that session (`helper.log` on the rig) shows three things:

- **No backoff, no escalation.** The failure was a *rebuild*: the 2228x1254 HDR feature released
  for a 3840x2160 SDR one when the post path took over at a loading screen. `maintain_passes`
  treated a failed pass 0 after an earlier success as "a later pass that will not build"
  (`pass 0 would not build; holding the chain at 0`, a ceiling of 0, clamped back to one wanted
  pass), so it retried on the rebuild spacing, every 250 ms, for ever: 433 identical attempts and
  never a re-initialisation of NGX. The one-shot `create_failed` (a 30 s block per key) only covered
  the very first build of a session.
- **NGX stayed refusing.** The attempts continued after GTA exited (VRAM free again) and through two
  more launches, at both sizes, all `0xbad00002`, while a fresh helper process (the restart) built
  the feature at once. So what refused was the helper's NGX instance, not the VRAM, and nothing in
  the helper ever shut NGX down and initialised it again.
- **Nobody was told.** `model_up` was only ever set to 1 (when a feature built), never back to 0,
  and the layer had no way to tell an echo from a model answer (`seq_ok` was written for every
  answer and read by nothing). So each hold drained the game's queue and waited for an answer that
  was the frame itself, often late because the helper was busy failing a creation.

The rebuilds themselves came from the layer: with no hold for 500 ms (every loading screen) the
post path took over and asked for a 3840x2160 SDR feature, and the next hold asked for 2228x1254
HDR again. That is two creations per loading screen, each near full VRAM.

### What changed

1. **Helper: no permanent failure** (`crates/helper/src/rebuild.rs`, `ngx.rs`, `main.rs`).
   - Pass 0 (the model itself), first build or rebuild, goes through a retry schedule,
     `rebuild::BuildRetry`. After the 1st, 2nd and 3rd consecutive failure it waits 0.5, 1 and 2 s.
     The 4th attempt first **re-initialises NGX**: `device_wait_idle`, release every feature,
     `Shutdown1`, destroy and allocate again a DLL-owned parameter block, then `VULKAN_Init_Ext` with
     the original arguments. After that it waits 30 s between attempts and re-initialises again on
     every 4th.
   - Never fatal: the old "pass 0 failed 3 times; giving up on the model" is gone. Never faster than
     the schedule, and not reset by a size or mode change.
   - `CreateFeature` failures carry their result code. A handle that came back with a failure, or
     whose setup work did not run, is released.
   - `model_up` is set to **0** while pass 0 cannot be built, and back to 1 by the next build.
     `helper_reason` says why (`model would not build at 1486x836 (0xbad00002, 4 in a row); retrying
     in 30s (NGX re-initialised); ...`) and is cleared on recovery (`[helper] the model built again
     at ... after N failed attempt(s)`). `shmctl status` now prints `helper_reason`.
   - `NEURAL_FORGE_FAIL_CREATE=N[@K]` (helper, debug, unset by default) lets K creations through and
     fails the next N without calling NGX.
   - Native unit tests, `rebuild::tests`: the schedule's timings, one re-initialisation then every
     4th, at most ~44 attempts in 20 minutes (the 250 ms spacing made 4800), recovery reported once,
     the injection's parsing.
2. **Layer: circuit breaker** (`preupscale::Breaker`).
   - Shared-memory protocol **11** appends `seq_eval` (offset 2020). The helper writes the slot-0
     request number there, before `seq_resp`, only when the model ran on it, so a hold can tell a
     model answer from an echo (`preupscale::await_answer` returns `Answer::{Model, Echo, Missed}`).
     An echo is not written back (miss `the helper echoed the frame (no model answer)`) and counts
     in `preupscale_misses`.
   - In model mode the breaker opens when the helper reports `model_up=0`, or when 8 holds in a row
     got no model answer (echo, late, none). DLSS submits are then forwarded untouched, with no
     capture and no wait, for 2 s; then one probe hold goes through. A model answer closes it;
     anything else opens it for another 2 s. It starts as a probe.
   - Logged on opening, every 30 s while open (`breaker still open after Ns: ... (N probes without a
     model answer, N DLSS submits forwarded untouched)`), and on closing.
   - `preupscale_state` 3 means paused. The GUI says "paused: DLSS's WxH input goes to DLSS
     untouched until the helper's model answers again"; `shmctl` says "paused: no model answer,
     forwarding untouched".
   - Tests, with a header-only fake helper: echoes open the breaker after exactly 8 holds; 240
     submits over the cool-down are forwarded with zero waiting; a failed probe re-opens it; a model
     answer closes it; a silent helper costs at most 8 budgets; `model_up=0` opens it at once and the
     probe still goes out. The GPU hold test also checks that an echo is not written back.
3. **Fewer rebuilds: a 30 s hand-back.**
   - Once a model-mode hold on a device has asked the helper (whatever came back), the post-upscaler
     compose stays off until no DLSS submit with identified inputs has been seen for **30 s**
     (`preupscale::HAND_BACK`; it was 500 ms since the last hold). Meanwhile frames are presented as
     DLSS made them. Loading screens get no model, and the helper's feature is no longer rebuilt at
     the output size and back at each one.
   - DLSS switched off for longer than that (DLAA, a game that stops using it) gives the post path
     back as before, logged (`no DLSS submit in the last 30 s (...); the post-upscaler compose runs
     as before`).
   - The DLSS submit is noted before the "presenting steadily" check, so a loading screen that still
     runs DLSS keeps the window open too.
   - "Asked the helper" rather than "held successfully" on purpose: while the breaker is open, the
     post path's output-size requests would otherwise rebuild the feature against the probes, and
     no probe would ever find it built.
   - Test: `post_off` across a 20 s loading screen, just under and just over 30 s, a device that
     never engaged, and through `Session::note` (a hold skipped before its request does not engage).
   - `NEURAL_FORGE_PREUPSCALE=off` is untouched: no tracking, so none of this is reachable
     (`probe_command_tests` and the smoke test's `off` pass are unchanged).

### Rig results

Run 2026-10-02 17:40-18:24 on lordnikon.

- This build was deployed with `scripts/deploy-rig.sh` (layer `eedf88b2c8f8`, helper
  `43c398d24fd8`). The stale protocol-10 `shm.bin` and `.owner` were removed first; nothing had
  them mapped. The committed build differs only in two log/reason strings (layer `74a4f101b7fc`,
  helper `78d5658dcee9`, deployed afterwards and left running).
- GTA V Enhanced, DLSS Balanced, script mods off, default mode (no `NEURAL_FORGE_PREUPSCALE`),
  `model_interval=2`, `working_scale=1`, `mvec_enabled=1`. Frame generation off unless noted. No
  remote-desktop session.
- A status poller (`shmctl status` and `nvidia-smi` VRAM every 0.5 s) ran beside each benchmark
  (`~/nf-spike/gta/<label>/poll.log`).
- One launch (`rb-c1440-1`) hung in the Rockstar Launcher before the game started: no game process
  after 10 minutes. It was killed by PID and rerun 5 minutes later.
- Kernel log: no Xid and no NVRM line from 17:30 to the end.

**(a) Fault injection, 1440p** (`rb-a1`). The helper was started with
`NEURAL_FORGE_FAIL_CREATE=6@2`. At 17:43:18, one second into pass 4, `shmctl set intensity 1.01`
retuned pass 0 to force a rebuild.

- Helper: the rebuild failed 6 times, at 0, +0.5, +1.5, +3.5 s (after `Shutdown1 -> 0x1` and
  `VULKAN_Init_Ext (re-initialisation) -> 0x1`), +33.5 and +63.5 s. It built at +93.5 s (`the model
  built again at 1486x836 hdr=1 after 6 failed attempt(s)`). The re-initialisation worked on the
  real NGX.
- Status: `model_up=0` from 17:43:19.5 (1 s after the retune) to 17:44:55.5, with the reason
  string. `preupscale_state=3` throughout. At 17:44:56 `model_up=1` and the reason was cleared.
- Layer: one echoed hold, then `breaker open: the helper reports no model (model_up=0)`. 48 probes
  got no model answer and 9460 DLSS submits were forwarded untouched; then `breaker closed ... after
  98.7s open`. Misses for the whole run: 51 (the probes plus the start).
- Frame rate (MangoHud; displayed = real here): **97.0 fps** while open (17:43:22-17:44:55), where
  NR off is 93.0-93.2 over pass 4 (`pu-nroff-1`, `p0-1440-nroff-1`). **66 fps** once holding
  resumed (17:44:55-17:45:13). Pass 4 as a whole: 91.5 real, 93.3 displayed, mostly breaker-open.
  Nothing waited 30 ms while the model was down.
- At the start of every run (here and in (b), (c)) the first hold was an echo: the helper was
  rebuilding from the post path's output-size SDR feature, with its 250 ms spacing. The breaker
  opened for 4.1 s at the first loading screen and closed on the second probe.

**(b) 4K + Smooth Motion, the scenario that triggered it.** The desktop was set temporarily to
3840x2160@144.000, scale 1.0, bt2100 with a non-persistent `gdctl set` (verified), and settings.xml
ScreenWidth/Height/RefreshRate to 3840/2160/144. Layers `VK_LAYER_neuralforge_neural:VK_LAYER_NV_present`
with `NVPRESENT_ENABLE_SMOOTH_MOTION=1`, two runs back to back.

| Run | Real fps | Displayed | GPU % | Power | Hold ms | Misses (run) | Creations | VRAM peak |
|---|---|---|---|---|---|---|---|---|
| `rb-b4k-sm-1` | 32.6 | 65.0 | 97 | 192 W | 19.3-20.0 | 2 | 2 | 11.86 GB |
| `rb-b4k-sm-2` | 35.3 | 70.7 | 96 | 203 W | 18.7-18.9 | 2 | 2 | 11.86 GB |
| before: `a4k-model-sm-3` (fresh helper) | 33.3 | 66.6 | 97 | 195 W | 18.6 | 64 | 2 per loading screen | 11.7-11.9 GB |

- No `0xbad00002`, no failed creation, never `model_up=0`. The breaker opened only at the start
  (4.1 s).
- The post path never took over at a loading screen. The helper built each size once per launch
  (3840x2160 SDR for the post path at the game's start, before the first hold, at low VRAM; then
  2228x1254 HDR) instead of twice per loading screen. The rebuild near full VRAM that started the
  stuck state no longer happens.

**(c) 1440p default mode, no variable** (desktop and settings.xml restored first):

| Run | Real fps | Displayed | GPU % | Power | Hold ms | Misses (run) | VRAM peak |
|---|---|---|---|---|---|---|---|
| `rb-c1440-2` | 65.2 | 65.5 | 90 | 181 W | 9.9-10.5 | 2 | 9.4 GB |
| `rb-c1440-3` | 64.7 | 65.5 | 90 | 181 W | 9.9-10.0 | 2 | 9.4 GB |
| **mean of 2** | **65.0** | 65.5 | 90 | 181 W | | | |

- Against 66.1 (65.1 / 68.1 / 65.2) in "Hand-off latency": within the ~3 fps run-to-run spread,
  so the breaker and the hand-back cost nothing measurable in steady state.
- Misses per run: 2, both at the start, against 85-89 before. Loading screens and scene changes
  no longer hand back to the post path, so nothing is rebuilt there and no hold waits for an
  answer that cannot come.

### Restored

- Desktop 2560x1440@288.001, scale 1.0, bt2100 (`gdctl show`, `gdctl show -p`).
- settings.xml sha256 8de357621960...1a7a, byte-identical; the backup was removed.
- `intensity` back to 1 after (a). Live settings `enabled=1`, `working_scale=1`, `model_interval=2`,
  `mvec_enabled=1`. The helper (this build, no fault injection) is running.

## Fixes from the 2.0 review

A code review of 2.0 (2026-10-02) found the following; each was confirmed in the code first and
has a test that fails without its fix. None of it is run on the rig yet.

- **A rebuild after a frame-key change opened the breaker** (`ngx::maintain_passes`,
  `rebuild::after_key_change`). Releasing the features for a new key (DLSS render resolution or
  quality, an SDR/HDR switch) set the next build a full spacing (250 ms) later, while `model_up`
  stayed 1: ~15 echoes at 60 fps, more than `BREAKER_MISSES`, so the breaker opened for 2 s on
  every such change (the echoed first hold of every rig run in "Robustness: failed feature builds"
  was this). The rebuild is now due at once unless the previous build was less than the spacing
  ago, so a key that keeps changing still builds at most once per spacing. `model_up` keeps its
  meaning (0 only while the model cannot be built): publishing 0 during a normal rebuild would
  open the breaker at once instead.
- **Double tone map of the encoded proxy** (`hdr_to_flow.comp`, `scene.rs`): see "Implementation
  (helper)" above; the flow input and the scene-cut thumbnail now quantise the layer's encode as
  it is.
- **The first hold after an identification assumed `GENERAL`** (`preupscale::Scan::colour_in_general`).
  Barriers are only recorded for watched images (the device hooks skip them until an input is
  identified), and a new identification drops what was committed, so on the submit that
  (re)identified DLSS's inputs every layout read "none seen" and the hold took the colour input
  (and the exposure image) to be in `GENERAL` without having watched it. That submit now goes to
  DLSS untouched; from the next one, "none seen" really means no barrier since it was watched.
- **Helper Color image usage** (`images.rs`): `TRANSFER_SRC` added; the 8-bit flow path blits from
  it (a validation error since before 1.1.0).
