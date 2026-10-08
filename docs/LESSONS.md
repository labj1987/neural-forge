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
7. [Deliberately not done, or left open](#deliberately-not-done-or-left-open)
8. [Working practices that came out of all this](#working-practices-that-came-out-of-all-this)

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
  whatever answer arrives onto whichever frame is current. Alex chose this explicitly for frame
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
  reverted fence changes are not to be reapplied (CLAUDE.md, "Runtime constraints").
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
  model interval (skipping frames) works. Alex runs at 100%.
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
  a 60 s sample. Alex: "everything is working beautifully".
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
- **The copy path below 100% model resolution.** About 4 ms per model frame of CPU copies. Alex
  runs at 100%.
- **The helper's `device_wait_idle` calls** stay unbounded: the Vulkan call has no timeout.
- **DMA-BUF and a native Linux helper:** closed (above).
- **Queue priority:** refused by the driver without privileges.
- **Building a frame generator into Neural Forge:** out of scope; the game's own works.

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
- **Check `journalctl -k` for `Xid`** before blaming the layer for a GPU hang, and check for GTA's
  own `+0x12c6eb` crash before blaming it for an early exit.
- **Every silent no-op logs its reason once.**
