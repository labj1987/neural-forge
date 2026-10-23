# What we tried, what worked, what didn't, and why

> **Note (2026-10-07, 3.0.0):** written before 3.0. Since 3.0 the model runs inside the layer
> (the native backend, [NATIVE_BACKEND.md](NATIVE_BACKEND.md)); the Windows helper, Wine, the
> runners, NGX at run time and the 32-bit layer are gone. What this document says about them is
> history; [ARCHITECTURE.md](ARCHITECTURE.md) describes the current design.

A chronological record of the decisions behind Neural Forge, from the first release (0.1.0,
2026-09-09) to 2.0.0 (2026-10-02). Each entry says what was tried, why, what happened (with
numbers where there are any), and where the evidence is. The detailed documents are the source;
this file is the map. When this file and a detailed document disagree, the detailed document's
measurement wins, and this file should be fixed.

All measurements are from the test machine (RTX 5070, NVIDIA 615.71.09) unless an entry says
otherwise. "History" means [history/development-before-neuralforge.md](history/development-before-neuralforge.md).

## Contents

1. [September 9-12: getting a model answer at all](#september-9-12-getting-a-model-answer-at-all)
2. [September 14-16: GTA V, transport and the first real numbers](#september-14-16-gta-v-transport-and-the-first-real-numbers)
3. [September 16-21: ghosting, and the switch to a synchronous present](#september-16-21-ghosting-and-the-switch-to-a-synchronous-present)
4. [September 21-25: motion vectors, zero copy, frame generation](#september-21-25-motion-vectors-zero-copy-frame-generation)
5. [October 1-2: measurement groundwork (1.0.1, 1.1.0)](#october-1-2-measurement-groundwork-101-110)
6. [October 2: the model before the upscaler (2.0)](#october-2-the-model-before-the-upscaler-20)
7. [October 11: the edit cache (3.3)](#october-11-the-edit-cache-33)
8. [October 13: the network's weights were on the GPU twice (3.4.1)](#october-13-the-networks-weights-were-on-the-gpu-twice-341)
9. [October 16: skipping the network's work on the padding (tried, not done)](#october-16-skipping-the-networks-work-on-the-padding-tried-not-done)
10. [Deliberately not done, or left open](#deliberately-not-done-or-left-open)
11. [Working practices that came out of all this](#working-practices-that-came-out-of-all-this)

## September 9-12: getting a model answer at all

### The layer crashed on its own activation path (0.1.5, fixed 0.1.6)

- **What:** with the layer enabled implicitly, every Vulkan program crashed in `vkCreateInstance`.
- **Why it happened:** the pinned `vulkan-layer` framework's default `create_instance` resolves
  all three global entry points through the next layer; resolving
  `vkEnumerateInstanceExtensionProperties` that way segfaults inside Mesa's `device_select` layer.
  The framework's own `hello-world` example crashed the same way.
- **Fix:** the layer implements `create_instance` itself and resolves only `vkCreateInstance`.
  5/5 clean runs locally, 3/3 on the rig.
- **Evidence:** History, "FIXED (2026-09-10, v0.1.6)".

### First real model answers (0.1.7)

- **What:** the helper's own NGX integration produced `EvaluateFeature -> 0x1` on 243 of 244
  frames of a 10-second `vkcube` run, with a legitimately signed `nvngx_dlssnr.dll`.
- **Evidence:** History, "First confirmed neural-rendering success".

### Blocking the present on the round trip (0.1.0-0.1.21) and the async pipeline (0.1.22)

- **What:** the first design captured a frame, waited for the helper's answer and composed it, all
  inside `vkQueuePresentKHR`.
- **Result:** the game's frame rate was capped by the round trip, about 100-150 ms per frame on the
  rig at the time.
- **Change (0.1.22):** capture only when no request is in flight, never block, and compose
  whatever answer arrives onto whichever frame is current. The maintainer chose this explicitly for frame
  rate. Its cost (an answer composed onto a newer frame) showed up later as flicker and ghosting.
- **Evidence:** CHANGELOG 0.1.22; History, "Real flicker root-cause".

### Full-frame copies at 70-130 ms each (0.1.18-0.1.21)

- **What:** per-stage timing showed every full-frame operation on the present thread (a heap
  copy, a write into shared memory, the round trip, the compose) costing 70-130 ms at 4K, about
  355 ms per frame in total.
- **Conclusion at the time:** `perf stat` showed the game 97% backend-bound with IPC 0.1, read as a
  memory-bandwidth condition outside the code's control. One clear waste (a fresh 31.6 MiB
  allocation every frame) was removed.
- **Later:** 0.1.64 found that the capture readback buffer was allocated in NVIDIA's uncached
  device-local BAR memory (see "The PCIe BAR readback" below). Whether that was the whole cause of
  the 0.1.21 numbers was never re-measured, but "a hardware condition, not a code bug" did not
  hold up.
- **Evidence:** CHANGELOG 0.1.18-0.1.22; [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md)
  (2026-09-16).

### `0xbad00002` from NGX's `AllocateParameters` (fixed 0.1.25)

- **What:** NGX refused to allocate its parameter block (`FAIL_PLATFORM_ERROR`), which disabled
  the helper.
- **Finding:** upstream's own compiled helper hit the identical error on the same machine and
  fell back to its own parameter object. NGX only needs a pointer with the right vtable layout.
- **Fix:** `crates/helper/src/selfparam.rs`, a self-implemented `NVSDK_NGX_Parameter`. Later (1.0.0)
  its vtable was laid out in the order NVIDIA's SDK header produces.
- **Evidence:** History, "NGX `FAIL_PLATFORM_ERROR` (`0xbad00002`) -- FIXED"; CHANGELOG 0.1.25, 1.0.0.

### Red and blue swapped on every `B8G8R8A8` swapchain (fixed 0.1.26)

- **What:** a uniform blue tint in live play. Every stage assumed RGBA byte order; the rig's
  swapchain is `B8G8R8A8_UNORM`, and Vulkan copies never reorder channels.
- **Fix:** the channel order is threaded through capture, composition and the shaders, with a test
  that compares true colours, not just GPU against CPU.
- **Evidence:** History, "Real red/blue channel swap".

### Bounded fence waits that made things worse (0.1.33-0.1.35, reverted)

- **What:** to fix a 9 fps collapse (196-274 fps with the effect off), the fence waits in capture
  and compose were bounded and guarded with non-blocking status checks.
- **Result:** both GTA games crashed on open. The changes were reverted in 0.1.35.
- **Lesson:** no Vulkan synchronization change ships without validation layers on, and the
  reverted fence changes are not to be reapplied (AGENTS.md, "Runtime constraints").
- **Also found that day:** upstream's apt package was still installed and loaded alongside, with
  the same trigger variable and the same shared-memory path; and the runtime directory's
  permissions depended on the umask of whoever created it first, which silently disabled the
  layer. Both fixed.
- **Evidence:** History, the three 2026-09-12 sections;
  [history/handoff-2026-09-12-fps-freeze-regression.md](history/handoff-2026-09-12-fps-freeze-regression.md).

## September 14-16: GTA V, transport and the first real numbers

### GTA's swapchain and the render tap (2026-09-14, 0.1.73-0.1.74)

- **What:** GTA's swapchain did not advertise `TRANSFER_SRC`, so a render-tap design read the
  image GTA blits into the swapchain ([RENDER_TAP_DESIGN.md](RENDER_TAP_DESIGN.md)).
- **Later finding (0.1.74):** admission had refused any swapchain whose create info carried an
  extension struct; DXVK and vkd3d-proton always attach one. The layer loaded, owned the session,
  tracked 793,394 barrier transitions and captured zero frames. GTA renders straight into the
  swapchain, so the tap never applied. Walking the chain fixed it.
- **Lesson:** every silent "does nothing" path now logs its reason once.
- **Evidence:** CHANGELOG 0.1.74; HARDWARE_VALIDATION.md, "GTA comparison gate".

### Zero-copy host memory (Phase 3, 2026-09-15)

- **What:** import the shared-memory regions as Vulkan memory with
  `VK_EXT_external_memory_host` on both sides, so the GPU copies straight into shared memory. The
  layer adds the extension at `vkCreateDevice`.
- **Result:** worked, but two real bugs were found only after everything had "validated clean": a
  null-pointer slice in device creation (undefined behaviour) and a panicking `ash` loader helper.
  Both were caught by `scripts/smoke-test.sh`'s debug build, not by validation layers or `cargo test`.
- **Evidence:** [EXTERNAL_MEMORY_HOST_DESIGN.md](EXTERNAL_MEMORY_HOST_DESIGN.md);
  HARDWARE_VALIDATION.md, Phase 3 entries.

### A second request slot (protocol v3, 0.1.59)

- **What:** two independent request/response slots, so a captured frame never waits for a free
  slot. The helper still evaluates one at a time: concurrent `EvaluateFeature` on one feature was
  never shown to be safe.
- **Evidence:** [PROTOCOL_V3_DESIGN.md](PROTOCOL_V3_DESIGN.md).

### DMA-BUF transport: blocked both ways (2026-09-15)

- **Why tried:** share GPU images between the layer and the helper with no trip through system RAM.
- **Helper to layer:** Wine guests only get `VK_KHR_external_memory_win32` handles. Wine's
  `wine_server_handle_to_fd` on such a handle returned `STATUS_OBJECT_TYPE_MISMATCH` (`0xC0000024`):
  it is a driver-private token, not a wineserver object.
- **Layer to helper:** a dma-buf fd is an anonymous inode, and Linux cannot re-open one through
  `/proc/<pid>/fd/<n>` from any process (`ENXIO`, confirmed without Wine). Only `SCM_RIGHTS` over a
  Unix socket could pass it, a different IPC than this project has.
- **Later context:** the model's evaluate (about 20 ms then) dwarfed transport (3-5 ms), so the
  loss was small.
- **Evidence:** [DMABUF_TRANSPORT_DESIGN.md](DMABUF_TRANSPORT_DESIGN.md); HARDWARE_VALIDATION.md,
  Phase 4 entries.

### A native Linux NGX helper: impossible today (2026-09-15)

- **Why tried:** no Wine, no cross-process handle problem.
- **Result:** NVIDIA's native `libnvidia-ngx.so.1` initialises cleanly (no caller spoof needed),
  but `CreateFeature(18)` returns `0xbad0000b`. Feature 18 is `NVSDK_NGX_Feature_Reserved18` in
  NVIDIA's public SDK; no Linux build of it exists anywhere this project looked.
- **Evidence:** [NATIVE_NGX_HELPER_DESIGN.md](NATIVE_NGX_HELPER_DESIGN.md).

### The PCIe BAR readback: 87 ms to 5.7 ms (0.1.64)

- **What:** the first real GTA sessions ran in the mid-to-high 20s fps. A hot-path benchmark
  (`capture_hot_path_cost_per_present`) showed 87 ms of CPU time per present.
- **Cause:** the capture readback buffer took the first host-visible memory type, which on NVIDIA
  is the small device-local BAR region; CPU reads from it are uncached over PCIe.
- **Fix:** prefer host-visible, host-cached, not device-local memory. CPU cost 87 ms to 5.7 ms
  (p50); GPU cost unchanged at about 1.6 ms.
- **Evidence:** HARDWARE_VALIDATION.md, 2026-09-16, "Measured it, found the real culprit".

### The GTA crash that was the driver (2026-09-16)

- **What:** GTA hung the GPU: `Xid 109` (`CTX_SWITCH_TIMEOUT`) then `Xid 119` (GSP timeout) in
  `journalctl -k`.
- **Finding:** a widely reported NVIDIA Linux driver bug under Proton, also seen in games with no
  DLSS layer at all. The rig needed a physical restart.
- **Lesson:** check `journalctl -k` for an `Xid` line before blaming the layer.
- **Evidence:** HARDWARE_VALIDATION.md, 2026-09-16 (later).

## September 16-21: ghosting, and the switch to a synchronous present

### Ghosting from the held answer (0.1.49-0.1.65)

- **What:** to stop flicker between enhanced and plain frames, the layer re-applied the last
  answer's edit to every frame until the next answer arrived (`carry_delta`), behind a per-pixel
  motion mask. An answer took about 26 ms, 3-4 frames at the time, so detail landed on content that
  had moved: ghosting.
- **Tried:** three mask variants live in GTA (0.1.65): v1 "a little bit less ghosting", v2 "closer
  to upstream" (shipped), v3 "visuals are worse" (reverted).
- **Conclusion:** the mask is a band-aid. A stale edit can only be shown (ghost) or dropped
  (enhancement vanishes in motion). Upstream's source, read after the relicense, documents the same
  reprojection-with-suppression approach as a dead end it removed.
- **Evidence:** [GHOSTING_PLAN.md](GHOSTING_PLAN.md) §1-2; [ATTRIBUTION.md](../ATTRIBUTION.md).

### Model resolution: CPU resample (rejected) and GPU blit (0.1.67)

- **Why tried:** upstream runs the model at about 0.75 scale, so the model is cheaper.
- **CPU resample:** 315 ms down and 546 ms up at 2560x1440 to 1920x1080. Never wired in.
- **GPU blit** (`vkCmdBlitImage`) into a small scratch image, and back up in the compose: helper
  evaluate about 26 ms to about 11.3 ms (p50) at 0.75 on the rig.
- **Later (1.0.1):** below 100% the layer leaves the zero-copy path for CPU copies, which cancel
  most of the saving: 58.1 fps at 0.75 against 61.6 at 1.0. Model resolution is the wrong lever;
  model interval (skipping frames) works. The maintainer runs at 100%.
- **Evidence:** GHOSTING_PLAN.md §1a, §1c; HARDWARE_VALIDATION.md, "2.0 baseline";
  [OPENDLSS_REVIEW.md](OPENDLSS_REVIEW.md), "Measurements".

### Synchronous present (0.1.78)

- **What:** on frame N's present, capture N, wait (at most 250 ms) for N's answer, compose it onto
  N, present. This is upstream's `ProcessPresent`.
- **Result:** no more ghosting; an answer is never applied to a later frame. Fail-open on a late,
  missing or restarting helper. The pipelined mode stays available as `NEURAL_FORGE_PIPELINED=1`.
- **Same release:** the layer engages only after 5 s of steady rendering and leaves on loading
  screens. Composing during GTA's loading screens had frozen the game.
- **Evidence:** CHANGELOG 0.1.78; GHOSTING_PLAN.md §1d.

### Loading-screen freezes and unbounded fence waits (0.1.85-0.1.87)

- **What:** every fence wait the layer or helper can hit was bounded at 5 s, with a breadcrumb
  trail dumped on a timeout. The idea came from DLSS5VKLayer PR #22.
- **Result:** not confirmed as the freeze's cause; no timeout line has been reported since. The
  5 s warm-up is what avoids the loading-screen freeze.
- **Evidence:** CHANGELOG 0.1.85-0.1.87; [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md), "Not done yet".

## September 21-25: motion vectors, zero copy, frame generation

### Motion vectors: from the layer to the helper (0.1.31, 0.1.69-0.1.98)

- **First version (0.1.31):** optical flow in the layer, on a private Vulkan device inside the
  game process. Creating that device during a swapchain transition crashed the driver, so it was
  stubbed out (commit `72d7a55`). For days the GUI's motion switch did nothing.
- **Second version (0.1.69):** optical flow in the helper, on the device it already has, like
  upstream. The opt-in gate did not cover device creation, and on the rig the helper hung after
  about 1000 frames, freezing the game (fixed 0.1.70).
- **Made real (0.1.93):** with motion on but no vectors, the helper had sent `DLSSNR.Reset=1` with
  every evaluate, so the model never used its history. The flow readback was uncached and read
  byte by byte: 216 ms to 4.2 ms per estimate at 1280x720.
- **On the GPU (0.1.95):** half-resolution flow and a compute shader into the model's motion image,
  0.75 ms per estimate at 2560x1440 instead of 16+ ms.
- **On by default (0.1.98):** cost 2-3% of the frame rate in GTA (62.4 to 61.0 fps at model every
  2nd frame).
- **Evidence:** GHOSTING_PLAN.md §1b, §4a, §4b; CHANGELOG 0.1.69-0.1.98.

### Zero copy in DX12 games (0.1.96, 0.1.97)

- **What:** vkd3d-proton enables `VK_EXT_external_memory_host` itself, and the layer only
  recognised the extension when it had added it, so GTA fell back to CPU copies (`zc=false`).
  Fixing that removed about 4.6 ms of copies per frame.
- **Regression:** 0.1.96 read the device's extension list after the framework had freed it, which
  crashed the Rockstar launcher. Fixed in 0.1.97.
- **Evidence:** CHANGELOG 0.1.96, 0.1.97.

### Frame generation (0.1.83, 0.1.99)

- **Model every Nth frame (0.1.83):** with DLSS Frame Generation on, every presented frame
  (generated ones too) waited for its own answer. At interval 2: about 54 presented fps against 42.
- **Smooth Motion below the layer (0.1.99):** NVIDIA's presentation-level generator, ordered below
  Neural Forge with `VK_INSTANCE_LAYERS`, so the model ran on real frames only: 116.4 shown fps
  against 60.9 at 1440p. lsfg-vk trailed it in every working mode, and one mode crashed GTA.
- **The layer order:** the Vulkan loader does not order implicit layers; without an explicit order
  both generators landed above Neural Forge.
- **Superseded in 2.0:** with the model before the upscaler, the game's own DLSS Frame Generation
  is better (53.0 real / 159 shown at 1440p), and the Smooth Motion switch was removed.
- **Evidence:** [FRAMEGEN_SPIKE.md](FRAMEGEN_SPIKE.md); CHANGELOG 0.1.83, 0.1.99, 2.0.0.

## October 1-2: measurement groundwork (1.0.1, 1.1.0)

### Review against OpenDLSS-NR (1.0.1)

- **What:** each stage of OpenDLSS-NR's documented pipeline compared with what Neural Forge
  controls through NGX.
- **Adopted:** the history rule. The model's history used to survive switching the effect off and
  on, loading screens and failed evaluates; the first answers after a pause blended with a picture
  that could be minutes old. The helper now resets history after a gap.
- **Tried and dropped:** a higher queue priority for the helper. NVIDIA's driver answered
  `ERROR_NOT_PERMITTED_KHR` to an unprivileged process.
- **Not worth it:** pre-recorded command buffers, narrower barriers, replacing the poll loops
  (microseconds against a 10 ms evaluate).
- **Evidence:** OPENDLSS_REVIEW.md.

### The GTA "NR-off ceiling" was a script mod

- **What:** with the effect off, GTA ran at about 63 fps with the GPU 45% busy, which made the
  effect look almost free.
- **First explanation:** ray tracing (commit `b443f68`). Wrong.
- **Found by elimination:** the Enable All Interiors .NET script holds GTA's main thread every
  frame. All mods off: 93.6 fps; that one script off: 93.0. Without mods the effect costs what it
  always did (61.6 fps on). Commit `5e0facd` corrected the review.
- **Since then:** every benchmark runs with script mods off
  (`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`).
- **Evidence:** OPENDLSS_REVIEW.md, "Cause of the NR-off ceiling".

### False scene cuts: none, threshold kept

- **Question:** does the fixed mean-luma threshold (40) reset the model's history on ordinary pans,
  as another DLSS 5 layer measured with its own fixed threshold?
- **Result:** 9 cuts in 7121 model evaluations (0.13%), at most 3 in the 117 s free-roam pass. The
  fixed threshold stays. A running-baseline detector had been built for the question (commit
  `a53d3f1`); it was taken out again before 1.1.0 and kept on the `scene-cut-baseline` branch in
  case the pipelined present ever shows false cuts.
- **Evidence:** HARDWARE_VALIDATION.md, "false scene cuts"; commit `68d0682`.

### The 2.0 baseline and the tools (1.1.0)

- 1.0.1, mods off, 1440p: 93.2 fps off, 61.6 on (model every 2nd frame), 41.9 every frame.
- 1.1.0 added GPU timestamps (capture 0.79 ms, compose 1.85 ms, no measurable cost), the
  unattended benchmark runner, log rotation, and the pan reproducer with its agreement metric.
- **Evidence:** HARDWARE_VALIDATION.md, "2.0 baseline", "1.1.0 against the 2.0 baseline".

## October 2: the model before the upscaler (2.0)

The full record is [PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md); the probe is
[PRE_UPSCALER_PROBE.md](PRE_UPSCALER_PROBE.md).

### The probe: DLSS is visible and holdable

- **Why:** run the model on DLSS's own input (render resolution, HDR, no HUD) instead of the
  finished frame. Idea from OptiScaler's pre-SR mod.
- **Result:** DLSS reaches Vulkan as NVX CUDA launches on registered views. Its colour input is
  identifiable (1707x960 `RGBA16F` at Quality), registered once, in `GENERAL`, and untouched before
  the first launch in 9006 of 9006 launch-bearing command buffers. So the layer can work at the
  submit, without injecting into game command buffers.

### E1: raw HDR input breaks the model

- **What:** a dumped 1486x836 scene-linear frame sent with `DLSSNR.Hdr=1`.
- **Result:** NGX accepts it and evaluates in 4.1 ms, but the answer is clamped to [0, 1]: a grey
  wash. Scaled or tone-mapped input gave plausible answers.

### E1b: the flags do nothing, the encode decides

- **What:** five encodes and the NGX flags, through the rig's real helper on two dumps.
- **Result:** `Hdr=1/SDR=0` against `Hdr=0/SDR=1`, and `AutoExposure` 1 against 0, give
  bit-identical answers. The model always treats its input as display-referred.
- **Chosen:** the game's 1x1 exposure value multiplied in, a paper white of 3, OpenDLSS-NR's
  per-channel shoulder above 0.75, sRGB. Statistically indistinguishable from the 8-bit
  reference. Paper white 1 blew out a third of the pixels and lost 20-23% of mean luminance on
  write-back; `x/(1+x)` lost 6-10% and turned hazy.
- **Evidence:** PRE_UPSCALER_DESIGN.md, "E1b"; `scripts/hdr_encode.py`.

### E2 and E3: the hold is cheap, but the gate failed at first

- **E2:** holding the DLSS submit and writing the same bytes back cost about 1 fps (92.5 to 91.5);
  capture and write-back are 0.65-0.7 ms of GPU each.
- **E3:** model every frame at Balanced: 50.5 fps (three runs), against a gate of 61.6. The hold was
  14.7 ms and the GPU only 68% busy, so the cost was waiting, not GPU work.

### The 4.9 ms "hand-off" was `powf` under Wine

- **Instrumentation:** a phase breakdown in the layer and a stage breakdown in the helper.
- **Finding:** the shared-memory hand-off was 0.18 ms. The missing time was the helper's CPU
  scene-cut thumbnail of the half-float frame: about 58,000 `powf` calls through the Windows C
  runtime under Wine, 4.8-5.3 ms (0.6 ms natively).
- **Fix:** a 64K-entry table built at start (bit-identical): 4.9 ms to 0.13 ms. Also one fence wait
  per request instead of one per stage, and yields instead of sleeps while busy.
- **Result:** 66.1 fps (65.1 / 68.1 / 65.2), hold about 10.2 ms, GPU 88-93%. The gate passed.
- **Evidence:** PRE_UPSCALER_DESIGN.md, "Hand-off latency"; commit `346e280`.

### 4K and HDR output

- **4K:** 39.0 fps before the upscaler against 28.7 after it (+36%). The model works on
  2228x1254 instead of 3840x2160, so the helper's time per call drops from about 25 ms to 10.9 ms.
- **HDR output:** GTA V Enhanced on PC has no HDR output to turn on. With `DXVK_HDR=1` the game
  still asked for an 8-bit swapchain. The after-the-upscaler path's handling of a real HDR
  swapchain is still untested.
- **Evidence:** PRE_UPSCALER_DESIGN.md, "4K and HDR output".

### The stuck NGX feature after `0xbad00002`

- **What:** at 4K with Smooth Motion, near full VRAM, one feature rebuild failed with
  `0xbad00002`. After that every build failed (433 attempts, through two later game launches)
  until the helper was restarted. Status still said `model_up=1`, and the layer held every DLSS
  frame for the full 30 ms.
- **Causes:** no backoff and no re-initialisation in the helper; NGX's instance stayed refusing
  while a fresh process built at once; `model_up` was never set back to 0; the layer could not tell
  an echo from a model answer. The rebuilds themselves came from switching to the
  after-the-upscaler path at every loading screen (two feature builds per screen near full VRAM).
- **Fixes:** a retry schedule with NGX re-initialisation (`rebuild.rs`); `model_up=0` with a reason;
  `seq_eval` (protocol 11) so the layer knows a model answer from an echo; a circuit breaker; and a
  30 s hand-back instead of 500 ms, so loading screens no longer switch paths.
- **Verified:** forced failures mid-benchmark kept the game at 97.0 fps and recovered after 6
  attempts, with a real NGX re-initialisation. 4K with Smooth Motion twice: 2 misses per run
  instead of 64, no failed build.
- **Evidence:** PRE_UPSCALER_DESIGN.md, "Robustness: failed feature builds"; commit `c467c65`.

### Frame generation's submits were held too

- **What:** DLSS Frame Generation also runs CUDA kernels, in two command buffers per real frame on
  another queue. The layer held every launch-bearing submit, so the model ran three times per real
  frame, twice on the wrong input: 22.4 real / 67.1 shown.
- **Fix:** hold only the command buffer whose launch parameters name DLSS's registered colour
  input. FG's buffers name other registered views and go through untouched; anything undecidable
  is held as before. Kernel names are never relied on.
- **Result:** 53.0 real / 159 shown against 28.7 / 86 on the after-the-upscaler path. The model
  runs once per real frame.
- **Evidence:** PRE_UPSCALER_DESIGN.md, "DLSS Frame Generation"; commits `6a9aab0`, `3a534d1`.

### Smooth Motion against the game's own frame generation

- Smooth Motion was the 1.x answer to frame generation (116.4 shown at 1440p). With the model
  before the upscaler, the game's own DLSS Frame Generation builds frames from enhanced frames, and
  NVIDIA recommends the in-game generator. The Setup tab's Smooth Motion switch was removed
  (commits `5b7901c`, `137f24e`).
- **Real play (2.0, an hour, DLSS FG 4x, mods on):** about 50 real / 195-199 shown fps, 0 misses in
  a 60 s sample. The maintainer: "everything is working beautifully".
- **Evidence:** HARDWARE_VALIDATION.md, "2.0.0".

### Fixes from the 2.0 review

A code review found six real problems, each with a test that fails without its fix: the
after-the-upscaler path's own request could engage the pre-upscaler path; a late pre-upscaler
request could be read as a pipelined post-path answer; a rebuild after a render-size change opened
the circuit breaker for 2 s; the helper tone-mapped the already-encoded proxy a second time for the
flow and the thumbnail (white came out at 188 of 255); the first hold after identification assumed
`GENERAL`; and the model's input image lacked `TRANSFER_SRC` usage. See PRE_UPSCALER_DESIGN.md,
"Fixes from the 2.0 review".

### GTA's own start-up crash

Every early exit during the 2.0 runs was an access violation at `GTA5_Enhanced.exe+0x12c6eb` during
"Game Init": 32 dumps on the rig, 11 of them on 2026-10-01 with 1.0.1, and other Game Init crashes
going back to January, before Neural Forge. It is the game's own; relaunching after a few minutes
works (HARDWARE_VALIDATION.md, "2.0.0").

**Corrected 2026-10-20:** it is not a crash at start-up but GTA's hang watchdog, which ends the game
when its update loop has not moved for 60 s, nearly always because a D3D12 fence it waits on never
completes ("October 20", below).

## October 11: the edit cache (3.3)

### 3.3 edit cache: what was tried

The edit cache (ARCHITECTURE.md 4.10) was ported from optiscaler-dlss5 with all of its anti-ghosting
parts. Each part was then measured offline with `scripts/cache_replay.py` on three GTA V dumps (16
holds each at 1485x836, the model on every hold: `gta-pan`, `gta-drive`, `gta-still`), the cache
mode carried 2, 3 and 4 holds, each part switched off in turn. Cells are err / p99.5 / worse% /
kept%: mean error and the per-hold 99.5th percentile (the strongest trails) in stops against the
model's own answer for that hold, the share of the frame worse than no edit by over 0.15 stop, and
how much of the model's effect is kept (100% = the model on every hold).

| dump | variant | lag 2 | lag 3 | lag 4 |
|---|---|---|---|---|
| pan | as ported | 0.0583/0.337/0.41/68.7 | 0.0746/0.433/0.86/58.7 | 0.0912/0.531/1.51/47.8 |
| pan | no fingerprint | 0.0592/0.316/0.83/66.1 | 0.0764/0.385/2.38/55.1 | 0.0926/0.448/4.42/44.9 |
| pan | no surroundings (context) | 0.0577/0.328/0.45/69.0 | 0.0735/0.421/0.95/60.0 | 0.0878/0.486/1.67/51.8 |
| pan | no surface fill | 0.0585/0.326/0.64/66.7 | 0.0770/0.454/1.84/55.7 | 0.0956/0.581/3.99/45.9 |
| pan | no Catmull-Rom | 0.0575/0.334/0.38/68.9 | 0.0741/0.431/0.82/58.9 | 0.0909/0.530/1.46/47.9 |
| pan | **without both (shipped)** | 0.0569/0.324/0.41/69.3 | 0.0729/0.419/0.90/60.2 | 0.0874/0.485/1.61/51.9 |
| drive | as ported | 0.1127/0.772/3.67/68.9 | 0.1200/0.820/3.96/66.9 | 0.1269/0.863/4.20/64.9 |
| drive | no fingerprint | 0.1126/0.755/4.12/68.9 | 0.1200/0.809/4.55/66.9 | 0.1269/0.853/4.92/65.0 |
| drive | no surroundings (context) | 0.1113/0.736/3.84/69.3 | 0.1182/0.775/4.17/67.4 | 0.1247/0.807/4.46/65.5 |
| drive | no surface fill | 0.1129/0.772/4.07/68.8 | 0.1206/0.835/4.54/66.6 | 0.1276/0.881/4.96/64.7 |
| drive | no Catmull-Rom | 0.1089/0.756/3.21/69.9 | 0.1163/0.807/3.49/67.9 | 0.1232/0.852/3.73/65.9 |
| drive | **without both (shipped)** | 0.1074/0.716/3.35/70.3 | 0.1143/0.756/3.68/68.4 | 0.1207/0.789/3.96/66.6 |
| still | as ported | 0.0838/0.504/2.64/60.4 | 0.0811/0.495/2.41/61.8 | 0.0859/0.516/2.80/59.4 |
| still | no fingerprint | 0.0846/0.505/2.89/59.9 | 0.0817/0.497/2.63/61.5 | 0.0868/0.520/3.07/58.9 |
| still | no surroundings (context) | 0.0838/0.504/2.66/60.4 | 0.0812/0.496/2.44/61.8 | 0.0859/0.517/2.83/59.4 |
| still | no surface fill | 0.0839/0.505/2.65/60.4 | 0.0812/0.496/2.43/61.8 | 0.0860/0.517/2.81/59.4 |
| still | no Catmull-Rom | 0.0808/0.490/2.14/61.8 | 0.0786/0.482/2.00/62.9 | 0.0830/0.502/2.32/60.7 |
| still | **without both (shipped)** | 0.0807/0.488/2.16/61.8 | 0.0786/0.481/2.02/62.9 | 0.0830/0.501/2.35/60.7 |

- **The fingerprint: kept.** Without it the area worse than no edit grows 2-3x at lag 4 on the pan
  (1.51 -> 4.42%) and by a sixth on the drive. Its p99.5 is lower without it on the pan (it rejects
  a carried edit and the frame goes without one), but that is the trade the fingerprint is for.
- **The same-surface fill: kept.** Without it the worse-than-no-edit area more than doubles on the
  pan at lag 4 (1.51 -> 3.99%) and kept drops 2 points.
- **The surroundings term of the fingerprint (upstream's context, a quarter-size mean log luma
  compared with the refresh's): removed.** Upstream added it for "the light an object left on the
  ground". Here it rejected more than it saved: without it trails (p99.5) and error are lower on the
  pan and the drive (pan lag 4 p99.5 0.531 -> 0.486, kept 47.8 -> 51.8%), for 0.1-0.2 point more
  worse-than-no-edit area; the still view does not change. Removing it also drops the quarter-size grid and
  its tent read from every hold.
- **The Catmull-Rom re-read of the history: removed.** Upstream re-reads the carried edit with a
  five-tap Catmull-Rom where all four bilinear taps are the same surface, against the blur that
  bilinear reads accumulate when the edit is re-carried hold after hold. Our cache never re-carries:
  every cached hold reads the refresh's own edit once, through the accumulated motion, so there is no
  blur to undo, and the sharpened read only adds error. Without it every number is better on all
  three dumps (still lag 2 worse% 2.64 -> 2.14), and the history pass does five fewer bilinear reads.
- With both removed the shader still matches the tool (the shader-against-tool test, within 1e-3
  stop) and interval 1 is unchanged bit for bit.

**Refresh crossfade and soft refresh: tried offline, removed after the live A/B below.** `cache_replay.py --cadence` runs the cache as the layer
does, the model on one hold in N and the cached holds between, and measures over time: the numbers
above for every hold after the first refresh (refresh holds included), p99.5 on the cached holds alone,
upstream's pulse probe (`measure_pulse.py`: the regional light, a 12-pixel Gaussian of the shown
picture's log luma ratio to the untouched frame, and its mean step from the hold before along the
motion, on refresh holds / on cached holds, the model's own step when it runs on every hold in
brackets), and flicker (upstream's `flicker_lab.py`: the motion-compensated change of the displayed
picture on pixels that stayed on their surface, minus the model's own on every hold, x1000; negative
is steadier than the model). Phase 0, each dump's 16 holds.

Without the walk the refresh holds pulse: the regional light steps 7x as far on them as on the cached
holds on the pan (0.044 against 0.006 stop) and 1.6x on the drive, because the cached holds hold one
answer still and the refresh jumps to the next. Upstream's keyframe crossfade (the shown edit walks
from what was on screen to the new answer across the interval, so it arrives when the model runs next)
with its soft refresh (pixels whose history failed the fingerprint walk too, from the last shown edit
one step back, or plainly where something came in front of a still view; only the background a move
reveals takes the answer at once) evens the pulse out (pan 0.023 / 0.021, drive 0.017 / 0.018, still
0.0085 / 0.0100, all at or below the model's own) and lowers flicker on every dump and interval (pan i2
+0.239 -> +0.010, drive -1.96 -> -2.93, still -1.25 -> -1.87). The price is lag: on the refresh
hold the picture is still half the previous answer at interval 2, so the error against each hold's own
answer goes up (p99.5 over all holds pan 0.139 -> 0.206, drive 0.357 -> 0.522, still 0.268 -> 0.386;
kept 7-9 points lower); on the cached holds alone it is the same at interval 2 and up to 0.03 higher at
3 and 4. Upstream turns its crossfade off before the upscaler (OptiScaler's `DlssNr_EditCache_Dx12.cpp`
around line 689, no reason given); the live runs below are after DLSS. Without soft refresh the
walk is weaker (pan i2 pulse 0.028 / 0.019, flicker +0.078). A walk capped at 2 holds is in between at
intervals 3 and 4 (the same at 2): less lag, more pulse, as upstream found when it chose the whole
interval. The walk over the whole interval with soft refresh shipped for a few hours (the capture,
history, apply and a refresh pass; `--no-crossfade`, `--no-soft-refresh` and `--xf-frames` in the
tool), until the live A/B after DLSS found nothing it improved there (below); it is gone from the
shaders, the layer and the tool.

| dump | run | err/p99.5/worse%/kept% | p99.5 cached | pulse refresh / cached (model) | flicker |
|---|---|---|---|---|---|
| pan | i2-base | 0.0220/0.139/0.10/88.9 | 0.261 | 0.0444 / 0.0062 (0.0265) | +0.239 |
| pan | i3-base | 0.0324/0.196/0.18/82.6 | 0.294 | 0.0594 / 0.0066 (0.0265) | +0.380 |
| pan | i4-base | 0.0479/0.275/0.43/75.2 | 0.344 | 0.0668 / 0.0073 (0.0265) | +0.634 |
| pan | i2-xf | 0.0327/0.203/0.11/83.7 | 0.261 | 0.0276 / 0.0189 (0.0265) | +0.078 |
| pan | i3-xf | 0.0453/0.277/0.24/76.4 | 0.304 | 0.0326 / 0.0153 (0.0265) | +0.202 |
| pan | i4-xf | 0.0586/0.347/0.64/70.3 | 0.367 | 0.0366 / 0.0140 (0.0265) | +0.501 |
| pan | i2-xf-soft | 0.0347/0.206/0.11/81.7 | 0.261 | 0.0230 / 0.0214 (0.0265) | +0.010 |
| pan | i3-xf-soft | 0.0502/0.285/0.27/72.8 | 0.306 | 0.0210 / 0.0178 (0.0265) | +0.167 |
| pan | i4-xf-soft | 0.0638/0.365/0.68/66.3 | 0.373 | 0.0182 / 0.0165 (0.0265) | +0.377 |
| pan | i2-xf2-soft | 0.0347/0.206/0.11/81.7 | 0.261 | 0.0230 / 0.0214 (0.0265) | +0.010 |
| pan | i3-xf2-soft | 0.0437/0.257/0.19/76.4 | 0.294 | 0.0306 / 0.0158 (0.0265) | +0.200 |
| pan | i4-xf2-soft | 0.0555/0.319/0.44/71.0 | 0.344 | 0.0345 / 0.0143 (0.0265) | +0.450 |
| drive | i2-base | 0.0531/0.357/1.59/85.0 | 0.669 | 0.0345 / 0.0214 (0.0190) | -1.962 |
| drive | i3-base | 0.0696/0.469/2.16/80.6 | 0.704 | 0.0408 / 0.0144 (0.0190) | -2.249 |
| drive | i4-base | 0.0864/0.576/2.70/76.1 | 0.720 | 0.0503 / 0.0112 (0.0190) | -2.677 |
| drive | i2-xf | 0.0761/0.510/1.73/78.7 | 0.669 | 0.0188 / 0.0181 (0.0190) | -2.910 |
| drive | i3-xf | 0.0922/0.618/2.35/74.4 | 0.699 | 0.0168 / 0.0146 (0.0190) | -3.236 |
| drive | i4-xf | 0.1027/0.680/2.84/71.6 | 0.717 | 0.0188 / 0.0123 (0.0190) | -3.381 |
| drive | i2-xf-soft | 0.0777/0.522/1.75/78.3 | 0.669 | 0.0174 / 0.0184 (0.0190) | -2.931 |
| drive | i3-xf-soft | 0.0944/0.633/2.38/73.9 | 0.699 | 0.0140 / 0.0153 (0.0190) | -3.282 |
| drive | i4-xf-soft | 0.1051/0.696/2.88/71.0 | 0.721 | 0.0137 / 0.0130 (0.0190) | -3.428 |
| drive | i2-xf2-soft | 0.0777/0.522/1.75/78.3 | 0.669 | 0.0174 / 0.0184 (0.0190) | -2.931 |
| drive | i3-xf2-soft | 0.0883/0.594/2.29/75.5 | 0.704 | 0.0204 / 0.0145 (0.0190) | -3.016 |
| drive | i4-xf2-soft | 0.0983/0.655/2.79/72.8 | 0.720 | 0.0254 / 0.0122 (0.0190) | -3.146 |
| still | i2-base | 0.0446/0.268/1.22/78.8 | 0.502 | 0.0185 / 0.0166 (0.0087) | -1.249 |
| still | i3-base | 0.0562/0.338/1.60/73.5 | 0.506 | 0.0193 / 0.0097 (0.0087) | -1.483 |
| still | i4-base | 0.0663/0.400/1.81/68.8 | 0.500 | 0.0184 / 0.0078 (0.0087) | -1.734 |
| still | i2-xf | 0.0634/0.385/1.29/69.9 | 0.502 | 0.0089 / 0.0103 (0.0087) | -1.867 |
| still | i3-xf | 0.0726/0.443/1.62/65.7 | 0.503 | 0.0068 / 0.0068 (0.0087) | -2.070 |
| still | i4-xf | 0.0772/0.471/1.76/63.7 | 0.494 | 0.0050 / 0.0058 (0.0087) | -2.142 |
| still | i2-xf-soft | 0.0641/0.386/1.29/69.6 | 0.502 | 0.0085 / 0.0100 (0.0087) | -1.872 |
| still | i3-xf-soft | 0.0732/0.445/1.62/65.5 | 0.502 | 0.0061 / 0.0066 (0.0087) | -2.080 |
| still | i4-xf-soft | 0.0776/0.472/1.75/63.5 | 0.493 | 0.0044 / 0.0058 (0.0087) | -2.150 |
| still | i2-xf2-soft | 0.0641/0.386/1.29/69.6 | 0.502 | 0.0085 / 0.0100 (0.0087) | -1.872 |
| still | i3-xf2-soft | 0.0695/0.420/1.64/67.2 | 0.506 | 0.0091 / 0.0071 (0.0087) | -1.934 |
| still | i4-xf2-soft | 0.0748/0.451/1.84/64.8 | 0.500 | 0.0084 / 0.0064 (0.0087) | -1.991 |

**Guided filter: kept. Aging: not added.** Upstream measured both slightly worse and ships them off
(edit-cache.md, "they also soften detail that was right"). Ours, the same definitions (aging: the high
band's confidence times 2^(-staleness / 8), staleness one per 4 px of each hold's motion since the
refresh; guided: He et al.'s fast guided filter of the cached edit on the hold's log luma, 2x2 means at
half size, a 3x3 window, eps 0.005, blended in by sat(staleness / 2)), run once through the cadence
replay with and without the walk:

- Aging changes almost nothing: p99.5 within 1% on every dump and interval, the worse-than-no-edit
  area 6-10% smaller on the drive only (i2 1.75 -> 1.65% with the walk), pulse and flicker the same.
  Too small to tell from these 16-hold dumps; not in the shader (`--aging` in the tool).
- The guided filter lowers error, trails and the worse-than-no-edit area and raises kept on every
  dump and interval: with the walk at interval 2, p99.5 pan 0.206 -> 0.193, drive 0.522 -> 0.489,
  still 0.386 -> 0.381; on the cached holds alone 0.261 -> 0.239, 0.669 -> 0.615, 0.502 -> 0.493;
  the worse-than-no-edit area drive 1.75 -> 1.29%, still 1.29 -> 1.17%; kept +0.4 to +1.0 point.
  With the walk its flicker is within 0.01 of the walk's alone (lower at interval 2: pan +0.010 ->
  +0.004, drive -2.93 -> -2.99, still -1.87 -> -1.91; pan +0.004 and +0.008 at 3 and 4); without the
  walk the drive's flicker rose by 0.15. Pulse unchanged. So ours disagree with
  upstream's and it went into the shaders: the motion pass accumulates the staleness, three
  half-size passes build the coefficients, the apply blends them in (`--no-guided` in the tool).

| dump | run | err/p99.5/worse%/kept% | p99.5 cached | pulse refresh / cached (model) | flicker |
|---|---|---|---|---|---|
| pan | i2-base | 0.0220/0.139/0.10/88.9 | 0.261 | 0.0444 / 0.0062 (0.0265) | +0.239 |
| pan | i4-base | 0.0479/0.275/0.43/75.2 | 0.344 | 0.0668 / 0.0073 (0.0265) | +0.634 |
| pan | i2-xf-soft | 0.0347/0.206/0.11/81.7 | 0.261 | 0.0230 / 0.0214 (0.0265) | +0.010 |
| pan | i4-xf-soft | 0.0638/0.365/0.68/66.3 | 0.373 | 0.0182 / 0.0165 (0.0265) | +0.377 |
| pan | i2-aging | 0.0219/0.138/0.10/88.9 | 0.260 | 0.0444 / 0.0063 (0.0265) | +0.238 |
| pan | i4-aging | 0.0477/0.275/0.41/75.3 | 0.343 | 0.0666 / 0.0074 (0.0265) | +0.630 |
| pan | i2-guided | 0.0209/0.127/0.07/89.3 | 0.239 | 0.0444 / 0.0062 (0.0265) | +0.244 |
| pan | i4-guided | 0.0461/0.260/0.34/75.9 | 0.325 | 0.0668 / 0.0073 (0.0265) | +0.646 |
| pan | i2-xf-soft-aging | 0.0345/0.205/0.10/81.7 | 0.260 | 0.0230 / 0.0214 (0.0265) | +0.006 |
| pan | i4-xf-soft-aging | 0.0636/0.364/0.65/66.4 | 0.373 | 0.0182 / 0.0165 (0.0265) | +0.372 |
| pan | i2-xf-soft-guided | 0.0334/0.193/0.08/82.2 | 0.239 | 0.0230 / 0.0214 (0.0265) | +0.004 |
| pan | i4-xf-soft-guided | 0.0623/0.352/0.58/67.0 | 0.358 | 0.0183 / 0.0164 (0.0265) | +0.385 |
| drive | i2-base | 0.0531/0.357/1.59/85.0 | 0.669 | 0.0345 / 0.0214 (0.0190) | -1.962 |
| drive | i4-base | 0.0864/0.576/2.70/76.1 | 0.720 | 0.0503 / 0.0112 (0.0190) | -2.677 |
| drive | i2-xf-soft | 0.0777/0.522/1.75/78.3 | 0.669 | 0.0174 / 0.0184 (0.0190) | -2.931 |
| drive | i4-xf-soft | 0.1051/0.696/2.88/71.0 | 0.721 | 0.0137 / 0.0130 (0.0190) | -3.428 |
| drive | i2-aging | 0.0525/0.353/1.50/85.1 | 0.662 | 0.0349 / 0.0221 (0.0190) | -1.976 |
| drive | i4-aging | 0.0847/0.566/2.44/76.4 | 0.708 | 0.0511 / 0.0113 (0.0190) | -2.782 |
| drive | i2-guided | 0.0498/0.328/1.16/85.9 | 0.615 | 0.0357 / 0.0238 (0.0190) | -1.811 |
| drive | i4-guided | 0.0816/0.526/1.96/77.2 | 0.658 | 0.0517 / 0.0119 (0.0190) | -2.702 |
| drive | i2-xf-soft-aging | 0.0770/0.517/1.65/78.5 | 0.662 | 0.0176 / 0.0187 (0.0190) | -2.977 |
| drive | i4-xf-soft-aging | 0.1038/0.689/2.61/71.3 | 0.713 | 0.0139 / 0.0130 (0.0190) | -3.546 |
| drive | i2-xf-soft-guided | 0.0737/0.489/1.29/79.3 | 0.615 | 0.0181 / 0.0197 (0.0190) | -2.991 |
| drive | i4-xf-soft-guided | 0.1026/0.657/2.27/71.5 | 0.675 | 0.0141 / 0.0132 (0.0190) | -3.617 |
| still | i2-base | 0.0446/0.268/1.22/78.8 | 0.502 | 0.0185 / 0.0166 (0.0087) | -1.249 |
| still | i4-base | 0.0663/0.400/1.81/68.8 | 0.500 | 0.0184 / 0.0078 (0.0087) | -1.734 |
| still | i2-xf-soft | 0.0641/0.386/1.29/69.6 | 0.502 | 0.0085 / 0.0100 (0.0087) | -1.872 |
| still | i4-xf-soft | 0.0776/0.472/1.75/63.5 | 0.493 | 0.0044 / 0.0058 (0.0087) | -2.150 |
| still | i2-aging | 0.0445/0.268/1.20/78.9 | 0.502 | 0.0186 / 0.0167 (0.0087) | -1.253 |
| still | i4-aging | 0.0661/0.399/1.75/68.9 | 0.499 | 0.0186 / 0.0077 (0.0087) | -1.760 |
| still | i2-guided | 0.0438/0.263/1.11/79.2 | 0.493 | 0.0186 / 0.0169 (0.0087) | -1.267 |
| still | i4-guided | 0.0641/0.385/1.51/69.7 | 0.482 | 0.0188 / 0.0078 (0.0087) | -1.852 |
| still | i2-xf-soft-aging | 0.0640/0.386/1.27/69.6 | 0.502 | 0.0085 / 0.0100 (0.0087) | -1.880 |
| still | i4-xf-soft-aging | 0.0774/0.472/1.69/63.6 | 0.493 | 0.0043 / 0.0057 (0.0087) | -2.178 |
| still | i2-xf-soft-guided | 0.0632/0.381/1.17/70.0 | 0.493 | 0.0085 / 0.0101 (0.0087) | -1.911 |
| still | i4-xf-soft-guided | 0.0759/0.460/1.48/64.2 | 0.479 | 0.0046 / 0.0057 (0.0087) | -2.283 |

**Live (GTA V Enhanced, 1440p, frame generation off, 2026-10-11).** The release build with all of the
above, `gta-bench.sh`, against the gate's runs of the cache as first ported (context and Catmull-Rom
in, no walk, no guided filter):

| run | real fps (pass 4) | passes 1-5 | cache_gpu_ms median | rejected |
|---|---|---|---|---|
| as ported, interval 2 | 77.7 | 81.8 / 81.1 / 90.2 / 94.1 / 81.3 | 0.509 | 3.2% |
| now, interval 2 | 75.4 | 82.4 / 80.0 / 88.3 / 93.2 / 79.4 | 0.758 | 2.6% |
| as ported, interval 3 | 64.6 | 64.9 / 71.7 / 80.0 / 87.0 / 68.5 | 0.543 | 3.6% |
| now, interval 3 | 79.7 | 85.6 / 85.7 / 97.0 / 101.3 / 85.1 | 0.789 | 2.9% |
| now without the guided passes, interval 3 | 80.0 | 87.4 / 85.8 / 99.4 / 105.5 / 85.2 | 0.689 | |

The cost per cached hold is read at interval 3, where two holds in three are cached and the median is
a cached hold's (at interval 2 it falls between the two kinds): +0.25 ms in all, of which the guided
filter's three half-size passes are 0.10 ms and the walk with the two removals the other 0.15; each
under the 0.2 ms a cached hold may cost. The fps differences are inside the run-to-run spread (about
3 fps; the as-ported interval-3 run ran at 69% GPU and is low for its own reasons). Fewer holds are
rejected without the surroundings test. No Xid.

At GTA's still moment (`NEURAL_FORGE_CAPTURE_AT`, bench-iq's 9 frames from the cut), interval 2:
flicker 1.197% against 1.030% with the model off on the same layer, added detail x1.130. There is no
frame-generation-off capture of the cache as first ported, so the walk's effect on flicker after DLSS
is not separated live; the model's own flicker every hold at this moment is the comparison still to
take.

**The walk, live after DLSS (GTA V Enhanced, 1440p, frame generation off, interval 2, 2026-10-11).**
A temporary `NEURAL_FORGE_CACHE_WALK=0` against the walk, with interval 1 and the model off as
references, the same layer (all edits 16-bit, below), at the still moment (bench-iq's 9 frames from
the cut, baseline the model-off run) and over pass 2's opening shot (Vinewood, the camera creeping
while the light goes from day to night; 160 presented frames from 3 s after the pass starts):

| run | real fps (pass 4) | still: flicker | still: added detail | pass 2: MAD vs interval 1 (all / edges) |
|---|---|---|---|---|
| walk on | 77.7 | 1.082% | x1.057 | 0.242 / 0.628 (again: 0.290 / 0.691) |
| walk off | 77.6 / 77.7 | 1.130% | x1.070 | 0.257 / 0.656 |
| interval 1 | 69.9 | 1.187% | x1.221 | 0 |
| model off | 90.4 | 0.976% | x1.000 | 0.394 / 0.844 |

MAD is the mean absolute log2-luma difference (stops, half resolution) to the interval-1 series after
aligning the frames by the offset of least difference; edges are the reference's top 10% gradient.
Two walk-on runs differ from each other by 0.190 / 0.533, more than walk on and off differ from
interval 1: the runs' timing differs more than the walk does. The walk's flicker is 4% lower, under
the 10% the metric resolves, and the PNGs at the sign's edges (frames 60 and 130 of the shot) show no
trail either way. Nothing it improves live, so it went (and with it the buffer the cache had of its
own). What does show at the still moment: interval 2 keeps a quarter of the model's added detail
(x1.06-1.07 against x1.22 every hold), the gap the refresh's carried edit loses to the current one: the history read, below.

**Where the cached holds lost the model's detail: the history read (2026-10-11).** At the still moment
interval 2 kept a quarter to a half of interval 1's added detail. The offline dumps put the loss in
one place. Per stage, the Laplacian energy of the carried luma edit over the model's own edit for
that hold (still dump: pixels whose accumulated motion is under 0.05 px; pan and drive: every pixel
on the chain), and the picture's detail before DLSS (log-luma Laplacian energy over the frame's):

| dump, lag | refresh's edit | nearest | bilinear | Catmull-Rom | depth-admitted bilinear (shipped then; since replaced by the strongest tap, below) | low band | after confidence | picture: model / nearest / bilinear |
|---|---|---|---|---|---|---|---|---|
| still, 1 | 1.004 | 1.001 | 0.178 | 0.359 | 0.179 | 0.050 | 0.194 | x1.037 / x1.021 / x1.003 |
| still, 2 | 0.999 | 0.998 | 0.268 | 0.520 | 0.269 | 0.049 | 0.259 | x1.029 / x1.019 / x1.004 |
| pan, 1 | 1.005 | 1.051 | 0.355 | 0.585 | 0.465 | 0.201 | 0.435 | x1.036 / x1.058 / x1.036 |
| pan, 2 | 1.002 | 1.070 | 0.356 | 0.583 | 0.479 | 0.200 | 0.427 | x1.032 / x1.059 / x1.036 |
| drive, 1 | 1.025 | 1.019 | 0.328 | 0.552 | 0.348 | 0.127 | 0.321 | x1.232 / x1.120 / x1.059 |
| drive, 2 | 1.049 | 1.027 | 0.331 | 0.551 | 0.350 | 0.122 | 0.314 | x1.233 / x1.121 / x1.060 |

- **The bilinear read keeps a fifth to a third of the high band.** The model's edit is near pixel
  scale, and a cached hold reads it at the jitter's sub-pixel offset: four taps blended at up to half
  a pixel average it away. The nearest texel keeps all of it, Catmull-Rom about half.
- **Nothing after it costs much.** The confidence takes at most 7% more (on still pixels it is
  0.94-0.95: colour check 0.999, 2 x valid - 1 1.00, fingerprint 0.94), so loosening the colour floor,
  the fingerprint or that term has nothing to win. The guided filter is off on a still view
  (staleness 0). The 16-bit edits: 3.9e-5 stop at most from the f32, 0 to 0.0006% of pixels at the
  +-2.5 stop range.
- **The rest is the model's own.** The refresh's high band correlates 0.14 (still), 0.00 (pan,
  drive) with the next hold's: the answer moves with the jitter, which is why kept stays near 60%
  whatever the read, and why a carried edit cannot be the next hold's.
- Catmull-Rom was removed earlier on per-hold error before DLSS, which a blur improves; detail after
  DLSS is the measure that shows the blur.

Live, GTA V Enhanced, 1440p, frame generation off, the still moment (bench-iq's 9 frames from the
cut, against one model-off run of the same layer) and pass 2's opening shot (160 frames from 4 s:
MAD to the first interval-1 run, all / edges, and the frame offset the alignment chose). Every run
through gta-bench with `model_interval` set; the kernel and the gain by temporary switches:

| run | real fps (pass 4) | still: added detail | still: flicker | pass 2: MAD (offset) |
|---|---|---|---|---|
| model off | 91.8 | x1.000 | 0.970% | 1.235 / 1.753 (+6) |
| interval 1 | 69.7 / 70.0 | x1.265 / x1.323 (mean x1.294) | 1.179 / 1.206% (1.193%) | reference / 0.171 / 0.335 (0) |
| interval 2, bilinear (as was) | 76.7 | x1.139 | 1.061% | 0.214 / 0.419 (0) |
| interval 2, Catmull-Rom | 75.8 | x1.167 | 1.276% | 0.296 / 0.546 (-6) |
| **interval 2, nearest (shipped)** | 77.0 / 76.5 / 77.8 | x1.229 / x1.186 / x1.216 (x1.210) | 1.119 / 1.104 / 1.064% (1.096%) | 0.328 / 0.556 (-6), 0.267 / 0.482 (-6), 0.292 / 0.538 (-4) |
| nearest, high band x1.1 | 76.6 / 75.9 | x1.146 / x1.168 (x1.157) | 1.119 / 1.242% (1.181%) | 0.161 / 0.326 (0), 0.333 / 0.579 (-8) |
| nearest, high band x1.2 | 77.4 / 77.4 | x1.192 / x1.185 (x1.189) | 1.042 / 1.125% (1.084%) | 0.284 / 0.523 (-1), 0.326 / 0.594 (-6) |

- **Nearest: kept.** The read takes the strongest depth-admitted tap's edit (the tap the fingerprint
  already came from); the four taps still give the weight and the colour check's luma. Added detail
  x1.210 over three runs against bilinear's x1.139 and interval 1's x1.294: 71% of interval 1's
  added detail, from 47%. Flicker lower than interval 1's in every run (1.06-1.12% against
  1.18-1.21%). The same fps.
- **Not the 95% asked for.** The metric's own spread is wide: two interval-1 runs differ by 0.058,
  three nearest runs by 0.043. What stays missing is the part of the model's answer that belongs to
  each hold's own jitter (the correlation above), which no read of an older answer can carry.
- **Catmull-Rom: no.** Less detail than nearest and flicker above interval 1's.
- **High-band gain: no.** 1.1 and 1.2 on top of nearest read x1.157 and x1.189, no more than without
  (inside the spread), and one 1.1 run's flicker was over interval 1's. Not added.
- **No trailing.** The pass-2 MAD follows the frame offset the alignment needs, not the variant:
  the runs that align at 0 read 0.16-0.21 (interval 1 again 0.171, a gain-1.1 run 0.161), those at
  -4 to -8 read 0.27-0.33, whatever they ran; GTA's camera is not at the same place in the same frame
  from run to run. The PNGs at frames 60 and 130 (the W and O of the sign, their legs) show the
  camera in a different place in each run and clean edges in all of them: no doubled edge, no trail.

**The adaptive cadence (Phase 4, 2026-10-11).** Upstream's cadence (`SpeedInterval`) picks the
interval from the camera's speed: long while still (2x the interval, up to 8), the interval while
moving, shorter in fast motion down to a floor of 2, with an early refresh when 10% of the carried
edit is rejected and a GPU budget as an alternative driver. Measured on our numbers, the
direction is the other way round, and of the rest only the regime itself stayed.

- **Speed.** Per hold, the median length of the motion vectors (the nearest surface's, render
  pixels): GTA's pan dump 4-7 px, the drive 5-10, the pass-2 creep (`gta-pan-64`) 1.1-1.6, Crimson
  Desert's foliage pan 1.0-1.6 (the same vector on two holds in a row: one game frame held twice
  at frame generation 6x), GTA's still shot 0.001 px. The pace is upstream's reference at motion
  priority 0.5, max(0.1% of the render width, 0.5 px) / 3 a hold: 0.495 px at 1485 wide, so a hold
  is still under 0.124 px. Every moving dump is far above it; the still shot is far below.
- **Direction: the network on every hold while still (taken), not a long interval (upstream's).**
  At the still moment a cached hold keeps 71% of the model's added detail (above: the per-jitter
  part of the answer no cache carries), and a still view is where a player looks closest; in
  motion the carried edit's error grows with the lag (trails p99.5 at interval 3 over 2: pan 0.131
  -> 0.185, drive 0.339 -> 0.437, `gta-pan-64` 0.335 -> 0.436), so the moving interval is 2. The
  still moments are a few percent of a benchmark's holds (1-7% here), so neither direction
  moves its fps: upstream's direction ran at the fixed interval's speed and loses the detail
  (below).
- **Hysteresis: 3 holds into the still regime, 20 out of it** (upstream's shorten-in-3,
  lengthen-in-20; still is the shorter interval here). On `gta-still-64` (still for 24 holds,
  creeping 0.01-0.4 px for 16, then accelerating to 4.5 px) the model runs on 56 of 64 holds:
  every hold until 20 holds after the creep passes the threshold, then every other one.
- **Still hold (a second threshold, three quarters of the pace, to leave): removed.** It keeps the
  still regime 4 holds longer on `gta-still-64` (model 92.2% of holds against 87.5%, kept 97.0
  against 96.0%), during the creep, and nothing else on any dump. The 20-hold exit already keeps a
  still view from flapping: a reading above the threshold has to repeat 20 times in a row. The
  live runs below had it; the bench-iq window is in the still regime either way.
- **Fast motion to interval 1: not added.** The carry is mostly accepted in fast motion (rejected
  3.4% of the drive's pixels at interval 2, 4.2% on the pan, 9.3% on Crimson Desert's foliage), so
  a refresh on every hold would buy little and cost the most fps exactly where the frame rate is
  felt.
- **Early refresh on the rejected share (upstream: 10%): not added.** At interval 2 the hold after
  a cached hold refreshes anyway, so the rule could only act at 3 or 4, which the trails above rule
  out.
- **GPU budget: not added.** Upstream's `ceil((refresh - cached) / (budget - cached))` sets the
  interval from cost alone: with a refresh at about 6.3 ms (the network 5.9, the write-back 0.4)
  and a cached hold's cache passes at 0.8-0.9 ms (1440p), a budget under about 3.6 ms gives 3 or
  more (the trails above), 3.6-6.3 ms gives 2 on every hold, a still view included, and more gives
  1 everywhere. It cannot meet the trail gate and the still gate at once; the regime does both.

Offline, `cache_replay.py --cadence N --adaptive` (`--still-interval` for upstream's direction),
err / p99.5 / worse% / kept% and the share of holds the model ran on:

| dump | fixed 2 | fixed 3 | adaptive, 2 in motion (shipped) | upstream's direction (8 still, 2 moving) |
|---|---|---|---|---|
| pan | 0.0215/0.131/0.09/89.1, 50% | 0.0314/0.185/0.14/83.0, 38% | = fixed 2 | = fixed 2 |
| drive | 0.0510/0.339/1.35/85.6, 50% | 0.0663/0.437/1.70/81.5, 38% | = fixed 2 | = fixed 2 |
| still | 0.0498/0.309/1.91/76.4, 50% | 0.0607/0.372/2.19/71.4, 38% | 0.0302/0.186/1.10/85.7, 69% | = fixed 2 (16 holds: never 20 still in a row) |
| Crimson Desert | 0.0653/0.505/2.28/72.8, 50% | 0.0831/0.628/2.82/65.4, 38% | = fixed 2 | = fixed 2 |
| `gta-pan-64` | 0.0421/0.335/1.44/88.3, 50% | 0.0581/0.436/1.93/83.3, 34% | = fixed 2 | = fixed 2 |
| `gta-drive-64` | 0.0443/0.276/1.18/84.2, 50% | 0.0590/0.363/1.45/79.1, 34% | = fixed 2 | = fixed 2 |
| `gta-still-64` | 0.0503/0.446/1.66/87.0, 50% | 0.0629/0.555/1.88/84.0, 34% | 0.0116/0.101/0.32/96.0, 88% | 0.0598/0.530/1.79/84.8, 38% |

The 64-hold dumps (`~/nf-spike/cache-dumps/gta-*-64` on the rig) are the bench's pass 2 at 4 s
and pass 5 at 67.5 s and 70 s. Upstream's direction on `gta-still-64` ran with the still hold
(below); the shipped column without it.

Live, GTA V Enhanced (`gta-bench.sh`, `model_interval` 2, the other two builds by temporary
patches never committed). Real fps: GTA's pass 4 (the long free-roam pass, where the still shot
is) and passes 1-3; p99 frame time over passes 1-3; the share of holds the network ran on and of
holds in the still regime from the layer's summaries. Still moment: bench-iq's flicker and added
detail over 9 frames from 31 frames after the cut, against a model-off run of the same layer
(each run's window and the baseline's at their own cut). Not the default 41: in the slower runs
the camera starts moving again 47-56 frames after the cut, and the default window caught it in
three runs (interval 1 x1.133 and x1.172, adaptive x1.104, flicker 1.36-1.93%).

| 1440p, frame generation off | pass 4 / passes 1-3 fps, p99 ms | network / still regime | still: added detail, flicker |
|---|---|---|---|
| interval 1 | 69.5 / 72.3, 15.9; 69.0 / 72.3, 16.5 | 100% | x1.327, 1.108%; x1.351, 1.120% |
| **adaptive (shipped)** | 74.5 / 79.6, 17.4; 73.4 / 80.2, 16.8 | 51.7 / 3.3%; 53.4 / 6.7% | x1.389, 1.044%; x1.309, 1.130% |
| fixed 2 | 75.0 / 79.3, 17.6 | 50.0% | x1.221, 1.016% |
| upstream's direction (8 still) | 75.2 / 79.7, 18.3 | 48.9 / 2.9% | x1.143, 1.099% |
| model off | 90.0 / 109.2, 14.1 | - | x1.000, 0.980% |

| configuration | adaptive: pass 4 / passes 1-3 fps, p99 ms | fixed 2 |
|---|---|---|
| 1440p, frame generation 4x | 52.4 / 54.4, 23.2 | 52.6 / 56.4, 23.8 |
| 4K, frame generation off | 43.5 / 44.3, 29.2; 42.5 / 46.2, 29.5 | 43.0 / 45.2, 29.1; 45.2 / 42.8, 29.1 |

- **Detail: adaptive keeps 103% of interval 1's added detail** (x1.349 against x1.339, means of
  two runs each; flicker 1.04-1.13% against 1.11-1.12%), fixed 2 keeps 65% (x1.221) and
  upstream's direction 42% (x1.143): it is at its long interval (8) through the window, which is
  mostly cached holds.
- **Speed: the fixed interval's.** The still regime is 1-7% of a benchmark's holds, so the
  network runs on 51-53% of them against 50%; every difference above is inside the run-to-run
  spread (about 3 fps). Upstream's direction runs on 49% and is no faster either.
- **Trails in motion: fixed 2's**, by construction (no moving dump reads as still) and offline.
- Runs left out: the first model-off, interval-1 and adaptive runs ran beside the offline replays,
  which pushed the rig into swap (48.7, 42.8 and 70.4 real fps); and two runs meant for frame
  generation 4x where it did not engage (1.0x: adaptive 76.0 and 74.8, fixed 76.9). No Xid in
  any run.

The shipped build (one threshold) once more, 1440p, frame generation off: 75.6 / 81.0 fps, p99
16.0 ms, the network on 51.0% of holds (2.1% in the still regime), added detail x1.343 at 0.978%
flicker. No Xid.

**The high band's confidence on foliage (Crimson Desert): measured, not added.** On Crimson
Desert's dump the low band alone is about as good as the whole cache from interval 2 up, and 9.3%
of its pixels are rejected against GTA's 3.4-4.2%. Two ways to lower the high band's confidence
with what predicts that, in the replay at interval 2 (err / p99.5 / worse% / kept%):

| variant | Crimson Desert | pan | drive | still |
|---|---|---|---|---|
| as is | 0.0653/0.505/2.28/72.8 | 0.0215/0.131/0.09/89.1 | 0.0510/0.339/1.35/85.6 | 0.0498/0.309/1.91/76.4 |
| low band only | 0.0650/0.530/1.19/72.4 | 0.0226/0.157/0.10/88.6 | 0.0598/0.412/1.54/83.1 | 0.0511/0.345/1.08/76.0 |
| x the 7x7 share accepted | 0.0656/0.508/2.12/72.6 | 0.0215/0.132/0.08/89.1 | 0.0510/0.340/1.32/85.6 | 0.0492/0.306/1.79/76.7 |
| x its square | 0.0658/0.513/2.03/72.5 | 0.0215/0.133/0.08/89.1 | 0.0510/0.341/1.29/85.6 | 0.0487/0.304/1.69/77.0 |
| x (1 - rejected share / 0.3), per hold | 0.0621/0.500/1.62/73.9 | 0.0213/0.131/0.08/89.1 | 0.0509/0.337/1.28/85.6 | 0.0481/0.299/1.62/77.2 |
| x (1 - rejected share / 0.6) | 0.0634/0.501/1.92/73.5 | 0.0214/0.131/0.08/89.1 | 0.0509/0.338/1.31/85.6 | 0.0489/0.303/1.76/76.9 |

The local share does nothing. The per-hold factor helps Crimson Desert (kept +1.1 points, the
worse-than-no-edit area 2.28 -> 1.62%) and leaves GTA's dumps as they were, but every number in
this replay improves when the high band shrinks (it is the per-jitter part that correlates 0-0.14
with the next hold's), and live the same kind of shrink (the bilinear read) cost the detail the
replay could not see. The factor scales GTA's carried high band by 0.86-0.89 and Crimson Desert's
by 0.69 on every cached hold in motion. Not added until a live Crimson Desert capture can show
whether the foliage looks better with it (the rejected count is already in the history pass, so
the apply could read it within the hold).

**The light pop-in limiter (Phase 5.1, upstream's anti pop-in, 2026-10-11): built, measured, removed.** The model sees only
the frame: a light entering it can make the model re-grade regions far from the light whose own
picture did not change, and the whole picture shifts in one hold. Upstream eases that: where the
frame's own regional light (a 3x3 tent of bilinear reads 12 px apart of its log luma) changed by
under 0.15 stop since the hold before, the shown edit's regional light (the same tent of the edit's
luma) may move at most a set rate (its default 0.75 stop a second) from what was shown there on the
hold before; where the frame changed (a light switched on right there, a cut, a revealed area) the
change passes. Ours ran it on a quarter-size grid on every chained hold, refresh or cached: the frame's tent and
each texel's position (its nearest-surface vector and the change of jitter) in the capture, the
edit's tent against the last shown one after the write-back, and the shift read bilinearly and
multiplied onto what the decode or the cached write-back wrote; the grid (28 bytes a quarter-size
texel) kept what was shown. The step was the rate times the smoothed hold time (0.9 / 0.1, each
clamped to 1/500..1/10 s). The shaders matched the replay tool's shift within 1.2e-5 stop.

Offline, `cache_replay.py --cadence 2 --adaptive` (the shipped cadence) and `--cadence 1` (the model
on every hold), the hold times the dumps' own (22-59 ms: dumping slows the game, and the game's
clock runs on). Jump: the regional light's step from the hold before along the motion (the pulse
probe's 12-pixel Gaussian of the shown picture's log luma ratio to the untouched frame), mean / p99
over the frame, averaged over the holds. Lag: the mean distance of the limited picture's regional
light from the unlimited one's (stops). Then err / p99.5 / kept% against the model's answer, as
above.

| dump | no limiter: jump, err/p99.5/kept% | 0.75 stop/s: jump, lag, err/p99.5/kept% | 1.5 stop/s: jump, lag, err/p99.5/kept% |
|---|---|---|---|
| drive | 0.0283 / 0.134, 0.0510/0.339/85.6 | 0.0263 / 0.122, 0.0038, 0.0550/0.382/84.5 | 0.0269 / 0.123, 0.0021, 0.0533/0.374/85.0 |
| `gta-drive-64` | 0.0212 / 0.086, 0.0443/0.276/84.2 | 0.0203 / 0.080, 0.0025, 0.0469/0.308/83.4 | 0.0207 / 0.081, 0.0011, 0.0454/0.298/83.9 |
| pan | 0.0241 / 0.148, 0.0215/0.131/89.1 | 0.0229 / 0.142, 0.0060, 0.0274/0.181/87.0 | 0.0234 / 0.143, 0.0032, 0.0247/0.172/88.0 |
| `gta-pan-64` | 0.0303 / 0.148, 0.0421/0.335/88.3 | 0.0271 / 0.132, 0.0068, 0.0483/0.387/86.4 | 0.0279 / 0.133, 0.0044, 0.0461/0.380/87.1 |
| still | 0.0132 / 0.050, 0.0302/0.186/85.7 | 0.0124 / 0.046, 0.0016, 0.0323/0.210/84.8 | 0.0129 / 0.048, 0.0007, 0.0310/0.202/85.4 |
| `gta-still-64` | 0.0144 / 0.074, 0.0116/0.101/96.0 | 0.0126 / 0.064, 0.0025, 0.0154/0.150/95.2 | 0.0134 / 0.067, 0.0013, 0.0134/0.138/95.6 |
| Crimson Desert | 0.0458 / 0.381, 0.0653/0.505/72.8 | 0.0405 / 0.328, 0.0038, 0.0677/0.541/71.9 | 0.0417 / 0.334, 0.0024, 0.0665/0.537/72.4 |
| still, model every hold | 0.0087 / 0.040 | 0.0072 / 0.032, 0.0016 | 0.0080 / 0.035, 0.0007 |
| `gta-still-64`, model every hold | 0.0125 / 0.063 | 0.0106 / 0.053, 0.0025 | 0.0115 / 0.056, 0.0013 |
| drive, model every hold | 0.0190 / 0.117 | 0.0169 / 0.109, 0.0042 | 0.0177 / 0.110, 0.0023 |

- **The gate: lower regional jumps on the drive, under 0.002 stop of lag on the still shots.**
  Upstream's 0.75 stop a second fails it: 0.0025 stop of lag on `gta-still-64` (its creep and
  acceleration). 1.5 passes: lag 0.0007 and 0.0013 on the still dumps (the same with the model on
  every hold), the drive's jumps 5% / 8% lower (mean / p99; the 64-hold drive 2% / 6%, the model on
  every hold 7% / 6%). Upstream's own offline result was in this range (Control's debris scene p99
  0.050 -> 0.033 stop for 0.0014 stop of lag).
- **Its cost offline is the model's own answer.** The limiter holds the regional light back from what the
  model gives on that hold, so every number measured against the model's answer is a little worse
  (drive p99.5 0.339 -> 0.374, kept -0.6 point; the pan's flicker, motion-compensated against the
  model's own, goes down: +0.254 -> +0.187).

Live (GTA V Enhanced, 1440p, frame generation off, the adaptive cadence, release builds with the
limiter at 1.5 stop/s and without it, three runs each in turns; the still moment from the new
shot's 31st frame against one model-off run of the same layer, 89.7 real fps, flicker 1.03%):

| run | real fps (pass 4) | still: flicker | still: added detail | cache_gpu_ms (median of the windows) |
|---|---|---|---|---|
| limiter | 72.6* / 74.3 / 75.1 | 1.134 / 1.001 / 1.066% (1.067%) | x1.230 / x1.300 / x1.285 (x1.272) | 1.03 / 1.14 / 1.14 |
| without | 60.6* / 74.4 / 75.2 | 1.155 / 1.046 / 1.048% (1.083%) | x1.283 / x1.382 / x1.382 (x1.349) | 0.93 / 0.96 / 0.98 |

(*: the first pair ran while another agent's image measurement held a core and pushed the rig into
swap.) No Xid.

- **Detail at the still moment: worse in every pair.** The limiter kept 78% of the added detail
  without it (x1.272 against x1.349), each of its runs below each run without it; the five earlier
  runs of the cadence without it read x1.31-1.39. Flicker the same (1.07 against 1.08%, under the
  10% the metric resolves), fps the same (74.3 / 75.1 against 74.4 / 75.2), the cache's GPU time
  +0.1 to 0.18 ms a hold.
- **Not kept: removed from the shaders, the layer and the replay tool.** The gate asked for detail
  and flicker at the still moment no worse; detail was. The shift is smooth (a quarter-size grid,
  36-pixel tents), so it should not touch pixel-scale detail by itself; what it holds back on the
  still shot is the model's regional re-grade while DLSS's history converges after the cut, and the
  measure reads less detail there. Upstream ships it on after DLSS, where the trade is different;
  before the upscaler, with the network on every hold while the camera is still, the jumps it would
  have cut were 5-8% of the drive's.

**The background model (upstream's CacheAsync, Phase 5.2): not attempted.** Upstream launches the
network on one frame on a compute queue of its own and lands its answer two frames later, carried to
then along the motion; the frames between show the cached edit. Its minimum interval is 3 (the launch
frame, the frame it runs beside, the landing), and it only paid at long intervals: Control at 1440p,
x8 in the background 25.3 fps against 24.8 in step (1% low 22.3 against 19.5), x3 in the background
21.7 against 23.8 in step, slower, because the network then shares the GPU with every frame (28 ms
wall on its queue against 16 at x8). The handoff made it conditional on our cadence leaving the cache
at interval 3 or more for a meaningful share of frames. It never does: the shipped cadence is 1 while
the camera is still and 2 while it moves (3 trails further, above), so 0% of holds are at 3 or more;
the network runs on 51-53% of GTA's benchmark holds (the still regime 2-7%). A second network
instance would also cost VRAM where there is none: upstream's pushed Control at 1440p to 11.6 of 12
GB and slowed every frame (24.2 to 19.8 fps), and at 4K with frame generation ours is already at the
card's edge without one ("3.3 edit cache: VRAM at 4K with frame generation").

**Low- and high-band gains (Phase 5.4): tried, not kept.** Upstream's `LowGain`/`HighGain` scale the
carried edit's two bands, a cheap stand-in for a stronger model. On top of the nearest-tap read, live
at GTA's still moment at interval 2 (the table above): a high-band gain of 1.1 read x1.146 / x1.168
added detail (mean x1.157) and 1.2 read x1.192 / x1.185 (x1.189), against x1.210 without a gain
(three runs, x1.186-1.229): no more detail than without, inside the runs' spread, and one 1.1 run's
flicker (1.242%) was above interval 1's (1.18-1.21%). The detail a cached hold lacks is the part of
the model's answer that follows each hold's jitter (the refresh's high band correlates 0.00-0.14 with
the next hold's), which scaling an older answer cannot bring back; and with the adaptive cadence a
still view runs the network on every hold, so there is no carried band left to scale where a player
looks closest. The Overdrive preset (intensity 1.30, local tone 1.25, local structure 1.45, skin
structure 1.20) would not gain from it for the same reason: it already raises the model's own
answer on every refresh, and a gain on the carried bands would only exaggerate the older answer on
the cached holds in motion, where the trail, not the strength, is what limits them. Not in the
shaders; `--low-gain` and `--high-gain` stay in the replay tool.

### 3.3 edit cache: reduced model resolution before the upscaler (built, removed)

Tried (handoff Phase 5.3): the network at a scale of DLSS's render extent (area-averaged input, history and
vectors at its size) with its edit enlarged to the render extent by a joint bilateral upsample guided by the
full-resolution luma and depth (upstream's JbuSigma 0.08; depth 0.15 stop), the decode and the edit cache
unchanged. The 1.x lesson (below 100% left the zero-copy path) no longer applied: the native network builds for
any size and the passes around it are the layer's own. Lowering the model resolution by default was authorized by
Alex on 2026-10-11 ("do whatever gives the best FPS and the best visuals, even if it does involve reducing the
resolution, as long as it doesn't hurt the actual picture"); the gate was the handoff's: added detail at least
95% of full resolution's.

Speed (GTA V Enhanced, 4K output, DLSS Balanced render 2228x1253, frame generation off, edit cache on): the
network 11.9 ms at 100%, 7.2 ms at 75% (1671x940), 3.8 ms at 50% (1114x627); real fps 44.0-44.7 / 49.5-50.6 /
53.7-56.6. Network memory 1.07 / 0.63 / 0.27 GB.

Picture: the first measurement used bench-iq's whole-scene added detail, which the traffic and pedestrians GTA
spawns swing by about 15% between runs of one build; on it 100% read x0.75-0.89 and 50% x1.03-1.08, as if the
smaller network were better. Measured on a fixed box of the still shot's building facade (no traffic in it;
log-luma Laplacian energy over model off, two runs each, the shot's frames 14-22 after its cut, and 10-18):

| Network at 4K | Added detail on the facade | Share of 100%'s added detail |
|---|---|---|
| 100% | x1.247, x1.191 (x1.197, x1.149) | 100% |
| 75% | x1.025, x1.030 (x1.011, x1.009) | about 12% |
| 50% (the final-validation run's own facade box) | x1.244 against 100%'s x1.881 | about 28% |

At 1440p 75% kept 88% on the whole-scene metric and 50% none. The 50% build also showed broad, soft diagonal
streaks on flat lit surfaces (the Vinewood sign's faces) that 100% shows only faintly. Every scale lost far more
than the 5% allowed, so the feature was removed (the shaders, `preupscale/reduced.rs`, the setting's effect before
the upscaler, the default rule); only a fix found while building it stayed: a pass for another extent is now
destroyed rather than dropped (dropping it leaked its images, buffers and pool). Measure detail on static areas
only; a whole-scene number moves with what the game spawns.

### 3.3 edit cache: VRAM at 4K with frame generation

At 4K with frame generation 4x the cache's gate failed: interval 2 ran at 19.1-22.5 real fps against
interval 1's 29-35 (earlier runs), the network at 15.8-17 ms instead of 12; VRAM was pinned at 11.86
of 12.2 GB in every 4K run (GTA fills what is free). The cache's buffers were 146 bytes a render pixel
(408 MB at 2228x1253), so they were the suspect.

- **Account and cut.** Of those 146 bytes, 68 lived across holds and the rest within one. All of it
  now lives in the network's input features (64 bytes a texel of its padded field), which only the
  network's frame uses: `native_preprocess.comp` rewrites every texel before the graph reads any.
  What crosses a network frame (the refresh's edit, read until the next refresh) is written after
  one and dead before the next; the walk's start was the one thing that crossed one, and it went with
  the walk. Edits are three 16-bit fixed-point stops in 8 bytes (the shaders stay within 8e-5 stop of
  the replay tool, was 4.5e-6; the test's tolerance is 1e-3); the fingerprint is read from the frame
  where it is needed and the step vector is gone; the guided filter's grids reuse the history read's
  and the pyramid's bytes. 56 bytes a render pixel of the features, 0 of the cache's own (8 for the
  few hours the walk still needed its start: 22 MB).
- **It was not the VRAM.** With the cache's own VRAM cut from 408 to 21 MB, the first runs at 4K FG
  4x still gave interval 2 20.0 real fps (network 16-19.8 ms, GPU 99% at 151 W) and interval 3 20.7
  against interval 1's 26.6 (network 12.5 ms, 214 W). VRAM in use was the same in all of them (median
  11.26 GB, peak 11.86 of 12.2: `nvidia-smi` each second). Later runs of the same layer at interval 2
  gave 28.0 real fps with the network at 12.1-13.0 ms (193 W): the same build is fast in some runs and
  slow in others. With the walk gone (the cache allocating nothing of its own) the 4K FG 4x runs
  spread wider still: interval 2 at 28.0 real fps with the network at 12.1-13.0 ms against interval
  1's 25.4 and 26.6 (12.0-12.5 ms) when the run is healthy, so the gate passes there; but three of the
  last six runs (two at interval 2, one at interval 1; with counter chaining and without) collapsed
  to 3.3-3.4 real fps, the network at 28-65 ms, the GPU at 21-22% and 60-67 W, while the kernel log said
  `NVRM: ctxBufPoolReserve: Failed to reserve memory ... Out of memory` about 2,000 times a minute for
  the whole run (none in the healthy runs). And interval 2's slower runs earlier (network 16-20 ms,
  ~150 W) had the same VRAM readings as the healthy ones. So at 4K with frame generation the process
  (GTA, DLSS frame generation, the network's 1.37 GB at the render size: about 294 MiB of weights and
  364 bytes a field texel) is at the edge of the card whatever the cache does, and which side of it a
  run lands on decides the result, not the interval. Frame generation also did not engage in three
  of the later runs (1.0x), which the gate has to repeat. Left open: the network's own VRAM (its
  activations, or a memory priority so the driver does not starve it), not the cache's. (The
  weights' second copy on the GPU is gone since 3.4.1: "The network's weights were on the GPU twice",
  below.)
- **The network's rebuild returns its memory.** Every run first builds the network at the output
  size for the after-the-upscaler path (the menus, before any DLSS frame), then at the render size
  for the hold. `VK_EXT_memory_budget`, now on the `[native] network built` lines: at 1440p the
  2560x1440 build adds 1666 MiB and the switch to 1485x836 frees 858-865 MiB, what the field sizes
  predict (about 294 MiB of weights and 364 bytes a field texel: 871 MiB) - nothing is held. At 4K
  the 3840x2160 build adds 3338 MiB (3.3 GB) and the switch frees 574-1235 MiB of a predicted 1970,
  read while GTA was at its budget (9.9 GB) and allocating; the 1440p match says it is GTA filling
  what was freed, not memory kept. The post path's build is not skippable: DLSS appears only once
  the benchmark starts, after the menus the after-the-upscaler path serves.

## October 13: the network's weights were on the GPU twice (3.4.1)

Idea from lmxxf's dlss5-on-amd-9070xt-porting 0.42 by Kien (MIT), which freed 216 MiB of held weight
capacity on AMD by compacting its own FP8 weight buffers (a different layout, the same principle). No code
was taken from it.

- **The duplicate.** OpenDLSS-NR's `nr::Model` uploaded each of the 153 tensor records whole as its `raw`
  buffer. The kernels read their matrices and attention priors only from the buffers that `fp8Matrix`,
  `fp8MatrixPermuted`, `f16Matrix` and `relativeBias` re-lay from the host copy; through `raw` they read
  only the scale vectors (per-channel f16 skip and blend scales, per-head f32 attention scales). A map of
  every read through `raw` (each `Kernels` entry point, the PTX generators and the GLSL shaders) found no
  matrix read there. Measured (the rig, `Network::weight_bytes`): the records take 147,683,904 bytes
  (140.8 MiB) on the GPU, the kernels need 100,160 of them (98 KiB), and the matrices take 145.7 MiB. So
  140.7 MiB were dead: about half of the "294 MiB of weights" above.
- **The design.** The matrix builders mark the record bytes they read; `Model::compactAux()` (patch 0008)
  replaces each `raw` with only the unmarked runs, each kept at its offset's alignment modulo 16, and
  `Tensor::auxOffset` maps a record offset to the compacted one at every kernel entry point that reads
  `raw` (it throws for a read overlapping a matrix's bytes, so a missed read fails loudly instead of
  reading garbage). The matrices are built while the first graph records, after that graph already
  holds `raw`'s address, so the compaction cannot happen at load: neural-forge-native runs it after the
  first build's warm-up, which already re-records the graph for the frames. Marking with a per-byte map
  lookup made the build 4.7 s instead of 0.37; one lookup per matrix gives 0.55 s (the 153 small uploads).
- **Bit-exact.** The heads against v3.4.0's at all four sizes the layer builds (1486x836, 2228x1253,
  2560x1440, 3840x2160; the rig test `the_heads_at_the_layers_sizes_match_a_saved_run`, the dump's
  features tiled over each field), with counter chaining on and off and with every PTX route off (the
  GLSL kernels): 0 values differ. The dump test is bit-exact too. An in-game capture comparison was not
  made: GTA's benchmark does not give the same frame at the same index from run to run.
- **In game (GTA V Enhanced, 1440p DLSS Balanced, frame generation 4x as set).** Opening and building
  the network adds 1536 MiB, against 1674 before (138 MiB less; the model load itself adds 148 MiB of
  which 140 are the records, compacted at the build). Real fps 56.3 / 58.4 / 56.9 against 56.5 / 56.8,
  the network at 6.56-6.67 ms against 6.57-6.59: unchanged. At 4K with frame generation 4x one of two
  runs of the patched build still collapsed with `ctxBufPoolReserve ... Out of memory` (3.2 real fps),
  so 140 MiB does not move GTA at 4K off the card's limit. 4K is no longer tested on the 12 GB test
  machine (AGENTS.md).

## October 16: skipping the network's work on the padding (tried, not done)

Idea from AMDLSS5 (`src/helper/runtime/pad_apara.h`, all rights reserved; only the idea was looked at,
no code or constants): the network runs on the padded field (1536x896 for DLSS Balanced's 1486x836,
docs/NATIVE_BACKEND.md, "Field padding"), so about 10% of every launch at that size covers rows and
columns the layer throws away. Propagate from the head back to the input how far each stage's output is
still needed, and launch only the windows inside that bound. With every stage trimmed by 10% the ceiling
looked like 0.5 ms of the network's 6.6.

- **The bottleneck sees every padded token.** The ViT (blocks 31-38) attends over all of level 5's
  tokens, the whole 24x16 grid at 1536x896; the attention kernels mask only the rows that round the
  token count up to a multiple of 64 (`token < tokens`). Level 5's rows 14-15 hold nothing but padding
  (836 / 64 = 13.1), and every output pixel attends to those 48 tokens. So each encoder stage (blocks
  0-30 and the transitions) and the ViT must run over the whole field, as they do now: a bound pulled
  back from the head stops at the bottleneck. This is also why the field size changes pixels inside
  the image.
- **What is left: the top two decoder levels.** Below the bottleneck only level 0 (768x448) and the field
  itself have whole windows of padding; at level 1 it is at most one window row of 28 in the shifted
  phases, and none deeper. Pulled back through the 8x8 windows of each phase (shifted windows are an
  offset grid with one more window per shifted axis, not a cyclic roll), at 1486x836 block 70 needs
  187x105 of its 193x113 windows (phase 4,4), and blocks 69, 68, 67 and 66 need 94x53 of 96x57, 95x53 of
  97x56, 95x54 of 97x57 and 95x54 of 96x56: 10% and about 7% fewer.
- **Measured.** A per-launch timestamp profile on the rig (the dump's features, 1486x836, barriers):
  the network takes 6.11 ms, the encoder and ViT 3.35 of it, block 70 0.32 and blocks 66-69 0.33 together.
  Block 70 walking 19,635 windows instead of 21,809 (a timing-only build, its output wrong) took
  0.2908 ms against 0.3215: 0.031 ms, the window count's share. Blocks 66-69 at their ratios add about
  0.023, so the trim can save at most about 0.054 ms, 0.9% of the network. The other sizes have less
  padding (2560x1440 has 32 rows of it: block 70 would skip 4 of 185 window rows).
- **Inside the noise, so not built.** The network's own time varies 6.56-6.67 ms between GTA runs
  (3.4.1, above) and the benchmark about 3 fps at 1440p; 0.054 ms of a 17.8 ms real frame (56 real fps
  with frame generation 4x) is about 0.2 fps. No patch was written, so the bit-exact gate, the GTA
  benchmark and the play checks were not run. Worth revisiting only if a network change keeps padded
  tokens out of the bottleneck (then the bound reaches the encoder and the 0.5 ms ceiling is real).

## October 19: passes again, natively (3.6.0)

Alex asked for a pass count like other apps offer (up to 10), whatever the measurements say about its use.
A one-off test on 2026-10-18 (a throwaway branch) ran the small network twice per frame; 3.6.0 ships it
as a setting with per-pass overrides, before and after the upscaler.

- **The edit cache keeps its edit in the network's input-feature buffer** (`cache::ScratchLayout`). The
  first test build ran pass 2 after the write-back, whose preprocess overwrote the edit pass 1 had just
  captured: cached holds then read garbage (rejected 100% of pixels in the cache status, against 3-5%
  normally). Every pass must run before the write-back, and the write-back must be the only thing that
  captures the edit. The shipped order is `C`, `N`, (`X` or `Y`, `N`) per later pass, `R`, `W`.
- **Anchor on the original frame.** Before the upscaler the encoded capture is saved before the first
  copy of an answer over it and put back after the last pass, so the write-back's clamped-highlight rule
  and the edit cache compare the last answer with the game's own frame. After the upscaler the shared
  proxy is never overwritten (later passes read a scratch copy), so the composition compares the whole
  chain with the original. `pan` after the upscaler: the edit is 191% of one pass's at 2 passes and 273%
  at 3 (agreement.py, ncc 0.96 / 0.93), so it grows instead of fading.
- **Each pass needs its own history** (an idea from y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG's notes,
  GPL-3.0, ideas only). With one shared history pass 1 blends toward the previous frame's multi-pass
  picture, which compounds frame after frame. Each later pass has its own pair of history images, built
  when the pass count first reaches it (about 20 MB per pass at 1486x836), and reads it only when it ran
  on the previous network frame. The history flag stays shared, so every reset (F11, a loading screen, a
  resolution change) resets every pass. The GPU test fails with the shared history.
- **Counter chaining across passes** is the same as across frames: each pass executes the whole recorded
  graph, with barriers between passes. `pan`, 2 passes, chaining against barriers: 60 of 60 frames
  bit-identical.
- **Memory** grows only by the later passes' history and parameter buffers: peak VRAM in GTA 1440p FG 4x
  was 9998 MiB at 1 pass, 10010-10072 at 2 and 10026 at 3 (the full network: 9986-10043 at 1, 10268 at 2).
- **The brightness guard** (Max ratio, after the upscaler) applies once to the final composition, so it
  caps the whole chain's edit. It is left as the user sets it rather than scaled with the pass count.
- **Not done:** per-pass ViT reuse and pass feedback (the RenoDX handoff's item 4 adds the latter).

Measured (mods off, FG 4x, three runs where the difference is small, pass 4 real fps): small 74.3 / 64.0
/ 58.1 at 1 / 2 / 3 passes, full 65.6 / 51.2 at 1 / 2. Two small passes and one full pass cost the same
within the 3 fps spread; two small passes look over-processed (2026-10-18 screenshots: gritty stone,
blotchy windows, streaky roads), one full pass does not.

### The after-the-upscaler path answered only the first game session (fixed in 3.6.0)

Found while testing passes with `pan`: on a channel a previous session had used, the after-the-upscaler
model never answered (3.5.3 too). The layer posts its first request in the present that starts the
native model server; the server's loop started from the current `seq_req`, so a request posted a moment
before the thread started was taken as seen and never served. On a fresh channel the layer gave up on it
(no heartbeat yet) and posted again; on a used one the header still showed the previous session's server
running, and the layer waited on the lost request for good. The server now starts from `seq_resp`
(`native_post::first_seen`): three sessions in a row all composed. `NEURAL_FORGE_PREUPSCALE=off` runs in
GTA after a first session showed nothing for the same reason.

### 4K with the small network (2026-10-19, Alex's go-ahead for this test)

One run of GTA at 3840x2160@144 VRR, FG 4x, Preset C, small network, one pass, texture quality one step
down (High to Normal) for the run only: 35.6 real / 142.4 shown fps, the network 7.47 ms at DLSS's
2228x1253 input (6.6 ms in the two runs where GTA did not engage frame generation, about its cost outside
a game at that size), peak VRAM 11863 of 12227 MiB, no miss, no Xid, no out-of-memory. The full network's
19 ms in game against 10.7 ms idle (3.4.1) is not seen with the small one: it fits, and slows by about
15% under frame generation's memory pressure without collapsing. Two passes at 4K were not run: there is
no memory left for them.

## October 19: the RenoDX handoff (3.7.0)

Users comparing NVIDIA's full and small networks found the small one much weaker (barely darkened crevices,
moved shadows, changed faces, sharpening, a "deep fried" moment after cuts). 3.7.0 makes the network a live
choice with the full one as the default, adds a scene-cut reset, a per-pixel control mask with a sky rule and
pass feedback. Measured in GTA V Enhanced at 1440p, FG 4x, Preset C, mods off, pass 4.

- **The network choice.** Full 65.2 real / 262.4 shown fps (65.3, 65.4, 65.1), the network 6.89 ms on the
  frames it runs; small 73.4 / 295.4 (72.7, 73.7, 73.8), 3.90 ms. Same as 3.6.0 within the spread (65.6 and
  74.3). A live switch in a running benchmark (full to small 30 s into the long pass, back 30 s later): each
  network built in 1.2-1.5 s, the history reset on each switch, no miss (the frames during the load go to
  DLSS untouched), no Xid. On the still shot the full network flickers less and adds more detail than the
  small one (0.75% against 1.27%, x1.53 against x1.32; one run each).
- **The control mask** (NATIVE_BACKEND.md, "The control mask"): 0.08 ms per network frame for the sky rule
  (6.89 against 6.81 ms full, 3.90 against 3.82 small), fps unchanged (65.2 against 65.2, 73.4 against 73.9).
  GTA V Enhanced is reversed-Z (the edit cache's verdict: about 17% of texels at exactly 0). On the still
  shot the sky rule halves the settled flicker (0.75% against 1.50%). The sky keeps its own blue there
  (without the rule the network greys it), but it is not identical to the effect off: a patch of open sky
  read 137/150/161 against 121/136/154 off and 142/150/158 without the rule. The mask leaves those pixels as
  they were (bit for bit in the GPU test), so what remains is the game's own response to the rest of the
  edited frame (its exposure) and DLSS's temporal blend near edges, not an edit of the sky.
- **Scene cuts.** Three resets in every benchmark run, all in the long free-roam pass at its scripted camera
  cuts (luma changes 52-68 of 255 against the threshold of 40), none in the other passes or on loading
  screens (no false positives seen in 15 runs). On the still shot after its cut, the full network with the
  reset against without it (`NEURAL_FORGE_SCENE_CUT=off`): 1.07% against 1.17% flicker just after the cut,
  0.75% against 2.29% once the shot has settled: without it the previous scene's history blends in.
- **Pass feedback**: lower values lower the added detail steadily, flicker does not move beyond noise; 1
  stays the default (NATIVE_BACKEND.md, "Pass feedback").
- **Bottleneck reuse (handoff item 5): not built.** The ViT bottleneck recomputed every other frame saves
  at most about a tenth of the network's time (the reference's figure). With the edit cache the network runs
  on about half the frames already, so the gain per frame is at most 0.19 ms of a 13.6 ms small-network
  frame (1.4%) and 0.34 ms of 15.3 ms for the full one (2.3%), both under the 3% the handoff set, and it
  needs a change to OpenDLSS-NR's graph. Measured network times from this table; the per-launch profile
  (October 16, above) puts the encoder and the ViT together at 3.35 of the full network's 6.11 ms at
  1486x836.
- **bench-iq.py's settle lands after the still shot when the capture slows the game.** The series capture
  reads every presented frame back, which slows the game, so GTA's still shot (about a second) holds fewer
  rendered frames than the 30 the default settle skips at FG 4x, and the window measured the camera moving
  again (20% "flicker"). The numbers above put a 9-frame window 40 presented frames (settled) and 8 frames
  (just after) after each series' own cut instead, with bench-iq.py's flicker and detail functions.

## October 19: placement, model resolution and history after the upscaler (3.8.0)

- **A history after the upscaler does not steady the edit.** Measured with `pan` (2492x1370, 60 frames;
  the edit's change between frames over its size): none 0.314, the previous output at the same pixel 0.355,
  reprojected by the GPU's optical flow 0.324 while panning at 6 px/frame; 0.249 against 0.296 on a still
  frame. Blending toward the network's own previous answer makes it change more, and the flow only brings
  it back near none for 2.4 ms more. It stays an env-var testing aid with `none` the default
  (NATIVE_BACKEND.md, "History after the upscaler").
- **A lower model resolution before the upscaler still loses the detail.** Rebuilt in 3.8.0 with the edit
  enlarged along the frame's edges, it measured in GTA V at 1440p (DLSS's input 1486x836) x1.01, x0.97 and
  x1.10 added detail on the facade box at Quality, Performance and Ultra Performance against x1.28-1.32 at
  Full, for 9-13% more real fps. The 3.3 lesson holds at 1440p too: a model on DLSS's input has few pixels
  to spare. After DLSS, on the 2560x1440 output (Ray Reconstruction), Performance keeps most of the detail
  at 150-153 fps shown against 118-120 at Full, so the setting is worth having there; Full stays the default.
- **A changed shared-memory layout needs a new version or a fresh file.** Two development builds with the
  same version but different layouts read the colour settings as 0, which zeroed the edit and made a
  measurement look like the model had no effect. Measure only on a freshly created shared memory after a
  layout change.

## October 20: why the unattended GTA benchmark was unreliable

Over 2026-10-19 and the morning of 2026-10-20 the agents' `gta-bench.sh` launches (120 of them, counted from
Steam's own "Adding process / Removing process" journal lines, which every launch produces) gave 108
results, 8 early exits, 2 launches stuck for the runner's full 20 minutes, and 2 launches that lasted under
70 s with no dump; and frame generation often missing. Each was traced on the rig (3.8.5, 1440p, frame
generation 4x and Preset C as Alex has it), with GTA's crash dumps, the kernel log, MangoHud's per-frame
logs, Streamline's own log and the game's memory read while it ran.

### The "+0x12c6eb crash" is GTA's hang watchdog

- **What:** every early exit left a dump at `GTA5_Enhanced.exe+0x12c6eb` or, during the benchmark, at
  `+0x951c50` or `+0xcfab00`. It was read as a start-up crash (October 2, below).
- **Found:** the executable is encrypted on disk, so the code was read from the running game
  (`/proc/<pid>/mem`). The faulting thread is a watchdog: once a second it compares a
  QueryPerformanceCounter stamp that the game's update loop writes (the function at `+0x12c8c0`, called from
  three places) with the one it saw before. When the stamp has not moved for 60 checks (the dword at
  `0x14286b710` is 60) it calls the handler of the current stage, a function that writes a code to address 0
  (`mov qword [0], 0x21e` at `+0x951c50`, `0x2ef` at `+0xcfab00`), or with no handler set writes to address 0
  itself (`mov dword [0], 0` at `+0x12c6eb`). The address names the stage, not a fault. Of the 110 dumps in
  the prefix (since January) 93 are the watchdog, and in 84 of those GTA's update thread waits at `+0x1e7c12a`,
  a loop of `ID3D12Fence::SetEventOnCompletion` (vtable slot 9) and 50 ms `WaitForSingleObject`: GPU work it
  submitted never completed.
- **The two clusters:** 2026-10-19 23:40-00:11, 5 exits in 8 launches, 4 of them with
  `Xid 109 (CTX SWITCH TIMEOUT)` 58 s before the dump: the hold-in-buffer development build writing through a
  destroyed view, already fixed by fd03e9b in 3.8.1 ("GTA V hang"). 2026-10-20 08:47-09:13, 3 exits in
  7 launches, no Xid, during an A/B of `kernel.split_lock_mitigate` on the 3.8.3/3.8.4 development builds:
  over that morning, launches with the mitigation on ended without a result 3 times in 7 (two watchdog exits
  and the stuck launch at 11:10), with it off 2 times in 24. GTA takes split-lock traps all the time (its own
  threads, Bink audio, Wine's dsound mixer; `journalctl -k`), and the mitigation makes each one sleep. The
  sysctl stays 0.
- **Not reproduced on 3.8.5 with the mitigation off:** none of the 34 organic launches of this
  investigation (Neural Forge on and off, the runner and Steam) stalled, exited early or got stuck.
- **Fix in the runner:** it reads the dumps a failed launch wrote and names the watchdog and its stage, copies
  `Xid` lines from the kernel log, notices a game that stopped presenting (MangoHud's per-frame log stops
  growing) and saves each game thread's state and wait channel, and launches again (`--tries`, default 3).
  Tested by stopping the game (SIGSTOP at 150 s: "stopped presenting at 168 s", relaunched, result) and by
  faulting it (below).

### The launches that never reached the benchmark

- **What:** two launches (07:06 and 11:10) ran the full 20 minutes with the game presenting and no
  benchmark. Their logs had been deleted, so the screen they showed is not known.
- **Reproduced:** a fault in one of GTA's threads during loading (SIGSEGV into the game at 120 s after
  the launch) is caught by GTA's own handler, which writes a dump (`0x80000003` at `+0x13b35e`) and keeps
  presenting a black screen with the loading spinner, at 50 fps, for good. The update loop keeps running, so
  the watchdog never fires. Faulted the same way in another launch, the game stopped presenting instead.
- **Fix in the runner:** a launch with the game running and no result after `NF_BENCH_STUCK_S` (480 s; a run
  takes 260-280 s) is ended and launched again; with Neural Forge in the launch the frame on screen is saved
  first. A launch where the game never appears within 120 s is retried the same way.

### Frame generation that stops in the benchmark

- **What:** with frame generation 4x in `settings.xml`, pass 4 ran at 1.0x in about half of the launches.
- **Found:** with Streamline's log on (registry `HKLM\Software\NVIDIA Corporation\Global\Streamline`,
  `LogLevel` 2, `LogPath`, for the test only), every launch that lost frame generation logged one line, at
  105-121 s, a scene change in passes 1-3, and none of the others did:
  `eDLSSGStatusFailReflexNotDetectedAtRuntime - sl.reflex must be enabled and active 6339 != 9414`.
  Frame generation came back for a moment at the next one or two scene changes and was off for pass 4.
  In `sl.dlss_g.dll` (`dlfgPresent.cpp`, `updateStatus`) the check runs on each present once DLSS-G has
  presented 100 frames: `sl.param.latency.frame` must equal `sl.param.reserved.frame`, the frame being
  presented. `sl.reflex.dll` sets the first on each PresentStart marker, and only while its Reflex mode is not
  Off. GTA turns Reflex off for its loading screens (the 2026-10-05 `SetSleepMode` sequence), so the check
  fails when DLSS-G is past its 100 frames before GTA has turned Reflex back on. In every failing launch the
  two frame numbers were about 3,000 apart (3,014-3,086), the frames presented with Reflex off. Which comes
  first is GTA's own timing.
- **Counts:** the runner kept frame generation in 9 of 15 launches with Neural Forge and in 4 of 11 without without; launched through Steam
  (`steam -applaunch 3240220` with the benchmark arguments, so Alex's launch options, the Steam overlay and
  the script mods) 3 of 4, the fourth with the same Streamline line. The runner's `env -i` environment is not
  the cause: the launch Alex uses fails the same way.
- **Tried, no help:** `DXVK_NVAPI_VKREFLEX=1` (dxvk-nvapi's Vulkan Reflex layer instead of vkd3d-proton's own
  `VK_NV_low_latency2` path; the layer loaded) lost it in 1 of 3 launches. dxvk-nvapi has no option that keeps
  Reflex on (its options: `DXVK_NVAPI_DRIVER_VERSION`, `_GPU_ARCH`, `_ALLOW_OTHER_DRIVERS`,
  `_DISABLE_ENTRYPOINTS`, `_VKREFLEX`, `_FAKE_VKREFLEX`, `_DRS_*`, logging), and none could: the check reads
  sl.reflex's own mode, which only GTA's `slReflexSetOptions` sets, and Streamline's production DLLs read no
  `sl.interposer.json` ("in development builds" only, their own message). The `SetSleepMode (Disabled)` /
  `(Enabled/2000us)` pairs at each scene change are in the launch log with `DXVK_NVAPI_LOG_LEVEL=info`.
- **Ruled out:** Neural Forge (above); the dlss-presetc timer, which replaces a DLL only when its hash
  differs and has not replaced GTA's since 2026-10-13 (its journal: 339 runs that changed nothing since
  2026-10-18); and, from 2026-10-05, window focus, VRAM and dynamic multi frame generation.
- **What the runner does:** frame generation is not retried (Alex, 2026-10-19). With frame generation on in
  `settings.xml` the runner prints "frame generation stopped at about N s" when the displayed rate falls to a
  third of its best for 20 s, about 125 s into the launch (the rule matched bench-report on all 26 runs
  checked), and `bench-report.py --fg` flags the run.

### The 20-minute wait

The runner waited up to 20 minutes for a result whatever happened, and a cancelled call left GTA running
beside the next one. It now ends a launch at 480 s, refuses to start while a GTA process is running, and kills
the Wine processes of GTA's prefix that the launch started.

## October 21: pipelining the hand-off (measured, not built)

al-sca's proton-crossapi-pipeline (https://github.com/al-sca/proton-crossapi-pipeline) found that mods
which hand each frame to a second device under Proton (DXVK and vkd3d-proton, two VkDevices joined by a
shared timeline semaphore) lose 5.8-7.5 ms per frame to idle waiting, a fixed cost: its serial mode took
18.59 ms for both 11.95 ms and 11.10 ms of real work. Presenting the whole previous frame (wait for pass
n-1 before overwriting the input, copy its result home, then hand frame n over) took the GPU from about
60% to 98%. The 2.x helper was that shape (its own device under Wine, 5.9-6.1 ms of round trip per hold,
PRE_UPSCALER_DESIGN.md, "Hand-off latency"), so the question was whether 3.x still is.

It is not: since 3.0 there is one device. Measured with `scripts/gta-bench.sh`, 2560x1440, GTA V
Enhanced, DLSS Balanced, settings as found (frame generation 4x, Preset C), script mods off, pass 4, one
run each; GTA turned frame generation off by itself in the first three (1.0x), so those three compare
directly. GPU from nvidia-smi each second, CPU per thread from `top -H` each second (100 = one core).

| Run | Real fps | GPU | Power | Busiest thread | Per-frame breakdown |
|---|---|---|---|---|---|
| Neural Forge not loaded | 93.6 | 63% | 145 W | 73% avg, 90% max | - |
| Model before the upscaler (default) | 76.4 | 84% | 188 W | 64% avg, 83% max | hold (CPU) 0.10 ms; network 5.77 ms on half the frames, edit cache 1.03 ms, write-back 0.44 ms (GPU) |
| `NEURAL_FORGE_PREUPSCALE=off` | 93.5 | 65% | 146 W | 73% avg | no network loaded (below) |
| Finished frame (`placement=present`), FG 4x | 12.8 (43.2 shown) | 96% | 223 W | 34% avg | capture wait 2.30, answer wait 17.65 (model 17.50), rest 0.05 ms; 43.6 composites/s |
| Finished frame, FG off | 35.7 | 96% | 214 W | 34% avg | capture wait 5.25 (the game's frame draining; capture 0.72 on the GPU), answer wait 17.55 (model 17.30), rest 0.06 ms; compose 1.83 ms GPU |

- **Before the upscaler there is no hand-off.** The network is recorded on the game's own queue between
  the game's work and DLSS; the submit is held 0.10 ms of CPU, and nothing waits on the host. The frame
  grew 2.41 ms (10.68 to 13.09) for about 3.8 ms of added GPU work per frame: less than the sum, because
  the GPU, 37% idle without Neural Forge (GTA's own CPU-side limits), takes the work into that idle time.
  There is no idle waiting to remove. Holding DLSS's whole submission one frame (keeping copies of its
  colour, depth, motion vectors, exposure and jitter) would add a frame of latency and video memory and
  could not shorten a frame whose layer work already runs in line on one queue.
- **On the finished frame the GPU is full.** The round trip outside the model is 0.05-0.06 ms (the model
  server polls every 200 us). With frame generation every presented frame, the generated ones too, runs
  the model (0.1.83's finding, `model_interval` its remedy), so 43.6 model frames a second fill the GPU.
  Without it the frame time, 28.0 ms, is the sum of the GPU work (game about 6.7, model 17.5, capture 0.8,
  compose 1.8: 26.8 ms, 96%). al-sca measured that the pipelined mode lands near the sum of both passes,
  not their maximum, so at most the remaining 4% (about 1 ms, 1.3 fps at 36 fps) could come back, below
  the run-to-run spread. Neither pipelined mode was built; `NEURAL_FORGE_PIPELINED` (answers composited
  onto later frames) stays as it is.
- **`NEURAL_FORGE_PREUPSCALE=off` had been loading no network since 3.0.** The native backend is created
  only in a mode other than off (`lib.rs`, `native`), and without it there is no model server, so the
  after-the-upscaler path presented every frame untouched: the run above equals the model off. The README
  and the docs still called it "the after-the-upscaler path everywhere"; they now say what it does, the
  layer logs it once, and the comparison with the model on the finished frame is `placement=present`.

## October 22: ideas from the other DLSS-NR projects

A survey of the other projects running DLSS 5's NR network (AMDNR by 3zwr1, lmxxf/Kien's and mochizuki0323's AMD
runtimes, TheAutomatic's and wilsjo2's OptiScaler forks, the SAOG0721 Magpie fork) and the tools around them (RenoDX,
ReShade, dlssg-to-fsr3, the FSR 4 overrides, HelixSR), read for what Neural Forge could use. What came of it:

- **Taken: activations placed by lifetime** (mochizuki0323's idea). About half the network's video memory
  (NATIVE_BACKEND.md, "Activation memory").
- **Already here:** the Natural/Cinematic grade after the network (mochizuki's `nr_style_grade.glsl`; ours matches NGX
  within a half step, NATIVE_BACKEND.md 2.2), an auto-exposure for games that hand DLSS none, motion accumulated over
  the frames the edit cache carries (AMDNR's `MotionAccumulateShader`), the sky from depth, counter chaining between
  launches (mochizuki's `NR_TCHAIN`), NaN from the head read as 0 in every composite (AMDNR's black-output guard), and a
  build that returns its memory when rebuilt for another size (AMDNR measured another runtime keeping 75-350 MB per
  size; the rig's `rebuilds_at_other_sizes_return_their_memory` holds ours flat).
- **Not taken, the noise lanes computed once** (mochizuki's `NR_NOISE_FIELD`): the noise is seeded per frame, and NGX's
  answer is reproduced bit for bit only with that seed. Computing it once gives up exactness for a negligible cost.
- **Not taken, the network one frame behind** (AMDNR's non-blocking lmxxf backend: frame N gets N-1's correction,
  reprojected, while N runs on another queue). Before the upscaler the hold already waits for nothing on the host
  (October 21, above): the network runs in line on the game's queue and the GPU takes it into the game's idle time.
  Moving it to another queue could only overlap it with the game's own work at a frame of latency, and TheAutomatic's
  A/B (`docs/decisions.md`, 09-21) found the previous frame's colour with the current frame's vectors blurred the whole
  picture.
- **Not taken, programmatic dependent launch** (`griddepcontrol`, lmxxf's home-made PDL): the launches go through
  `vkCmdCudaLaunchKernelNV`, which takes no CUDA launch attributes, and counter chaining already overlaps them.
- **Not taken, skipping the ViT when the picture has not changed** (lmxxf's `DLSS5_VIT_ADAPTIVE`, 5-6% in motion):
  TheAutomatic found it flickers badly with the network's history on, and ours is always on.
- **Not needed yet, pre-SR inputs smaller than their image** (wilsjo2's padded pre-SR): every game held so far (GTA V
  Enhanced, Crimson Desert, Cyberpunk 2077, Black Myth: Wukong, GTA San Andreas DE, Hogwarts Legacy) renders into an
  image of its render size. A game with dynamic resolution into a larger image would need it.
- **Not available in GTA V Enhanced: a reactive mask.** DLSS's input kernel names six images there (the 1485x836
  colour input, the 2560x1440 output, the motion vectors, depth, a 1x1 RGBA32F and a 640x384 RGBA16F image), none of
  them a one-channel mask, so there is no reactive mask to read (the layer now logs that list once per session,
  `DLSS's input kernel names ...`).
- **Left for later:** the game's reactive mask in the edit cache in games that pass one (AMDNR's `reactiveAt`), the frame
  generation HUD-less image as an exact HUD mask (dlssg-to-fsr3 reads `DLSSG.HUDLess`; a Vulkan layer under Proton sees
  only images, not NGX's parameters), a low-band temporal filter of the correction (SAOG's Magpie, lmxxf's
  `DLSS5_TEMPORAL_MODE`; DLSS5VKLayer measured an answer carried into newer frames as a dead end twice, ATTRIBUTION.md
  "The composition strategy", which argues against it unless the band split changes that), and HDR on the after-the-upscaler path (RenoDX's resource upgrade; HDR
  swapchains pass through untouched today).

## October 23: Metro Last Light Redux, NVIDIA's own NR as the reference

Metro Last Light Redux ships the same `nvngx_dlssnr.dll` the network is extracted from, run by NVIDIA's integration.
Three things it showed (NATIVE_BACKEND.md, "Metro Last Light Redux and NVIDIA's own NR"):

- **A game can break three assumptions at once:** four-channel motion vectors, several Vulkan devices with the network
  open on the wrong one, and an exposure image that is always 0. Each made the model silently do nothing; only the log
  and a within-frame dump showed it. A screenshot against the model off looked like a change because the scene moved.
- **Compare within one frame, not across launches.** A title scene that looks still differed by 5.5 of 255 between two
  shots, as much as the model's edit; numbers from separate launches contradicted each other (an edit correlation of
  0.89 and of -0.46 for the same setting). The probe's input and output of one frame, against our dump of a hold a few
  seconds later in the same session, gave stable numbers.
- **NVIDIA runs NR after DLSS on the game's tone-mapped image, linear.** Ours (before the upscaler, sRGB-encoded,
  exposure and shoulder) pulls the brighter half of the picture down where NVIDIA's does not, and adds less detail at
  the default brightness; about half the cost at the default placement, about twice NVIDIA's at the same one.

## Deliberately not done, or left open

- **Phases 2 and 3 of the original 2.0 plan** (a cheaper model resolution on the zero-copy path,
  and a pipelined present for the 1.x path). The model before the upscaler replaced them for DLSS
  games (PRE_UPSCALER_DESIGN.md, status line).
- **An asynchronous hold.** The capture still waits for the game's queued work (3.6-4.4 ms at
  1440p). Going further means answering frame N-1 while N renders, at a frame of latency.
- **The game's own depth and motion vectors for the model.** Optical flow is still used; the
  game's vectors need their scale and sign, which the layer cannot see.
- **Other games before the upscaler.** Only GTA V Enhanced has been probed.
- **HDR swapchains on the after-the-upscaler path.** Presented untouched; never tested with a game
  that has real HDR output.
- **The copy path below 100% model resolution.** About 4 ms per model frame of CPU copies. The maintainer
  runs at 100%.
- **The helper's `device_wait_idle` calls** stay unbounded: the Vulkan call has no timeout.
- **DMA-BUF and a native Linux helper:** closed (above).
- **Queue priority:** refused by the driver without privileges.
- **Building a frame generator into Neural Forge:** out of scope; the game's own works.
- **Skipping the network's work on the padding:** at most 0.054 ms of 6.1 at 1486x836, because the
  ViT attends over the padded tokens and the encoder must run over the whole field (October 16, above).
- **A before/after capture while the model runs before the upscaler.** The capture request (`shmctl
  capture`, the Debug tab's Capture button) is served after the upscaler as a matched pair: the game's
  frame and the composited result (`capture::run`, `dump::write_pair`). While frames are held before
  DLSS (the default for DLSS games), the post path does not run, and the present takes the request as a
  one-frame series (`series::take_request`): both PNGs are the presented frame, which already carries the
  model's edit, so there is no "before" image. A real pair needs the colour input read back before the
  network writes its answer, at render resolution, and the same frame's upscaled output; the
  `NEURAL_FORGE_PREUPSCALE=dump` mode reads the input but never writes back, so it is not that either.

## Working practices that came out of all this

- **Measure before and after every performance change**, with the unattended GTA benchmark and
  three runs where the difference is small. The run-to-run spread is about 3 fps at 1440p.
- **Run with script mods off**, no remote-desktop session connected (it costs about 8 fps,
  according to the runner's own warning), and MangoHud as the last layer so it counts shown frames.
- **Check what is deployed**, not what was built: compare hashes of the installed layer and helper
  (`scripts/deploy-rig.sh` does this). Several early "the fix did nothing" results were stale
  copies or a stale git clone on the rig.
- **Restart Steam to change its environment.** A running Steam client keeps the environment it
  started with, and a game's saved launch options apply regardless.
- **Run `scripts/smoke-test.sh` after touching device-creation code**; it caught what validation
  layers and unit tests did not.
- **Validation layers before any synchronization change**, and never reapply the reverted fence
  changes of 0.1.33-0.1.34.
- **Check `journalctl -k` for `Xid`** before blaming the layer for a GPU hang. A GTA dump at `+0x12c6eb`
  (or `+0x951c50`, `+0xcfab00`) is its hang watchdog: the update loop stalled for 60 s, so look for what
  the GPU never finished, not for a crash.
- **Read the game's own logs before deciding a failure is random.** Steam's journal lines give every
  launch's lifetime, GTA's dumps say where the game waited, and Streamline writes why frame generation
  turned off when its registry log switch is set.
- **Every silent no-op logs its reason once.**
