# NeuralForge target-machine validation — 2026-09-14

> **Note (2026-10-07, 3.0.0):** written before 3.0. Since 3.0 the model runs inside the layer
> (the native backend, [NATIVE_BACKEND.md](NATIVE_BACKEND.md)); the Windows helper, Wine, the
> runners, NGX at run time and the 32-bit layer are gone. What this document says about them is
> history; [ARCHITECTURE.md](ARCHITECTURE.md) describes the current design.

> **Note (2026-10-02):** entries are dated and kept as written; later entries correct earlier ones.
> One forward-looking line no longer holds: "2.0 baseline" says the copy path below 100% model
> resolution is "what Phase 2 removes". Phases 2 and 3 of the 2.0 plan were not built; running the
> model before the upscaler replaced them for DLSS games ([PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md)).
> A summary of the whole record is in [LESSONS.md](LESSONS.md).

Target: `alex@lordnikon`, RTX 5070, NVIDIA 615.71.09. Upstream package 0.3.0-1
remains installed. At inspection time GTA and the upstream helper were not running.
The saved upstream config retains passes=1, model_resolution=1, motion_enabled=0,
motion_quality=0. No upstream config, package, library, manifest, prefix or Steam
launch option was modified. Upstream config and both layer manifests passed a
before/after SHA-256 comparison.

NeuralForge is installed separately in `~/.local/share/neural-forge`, with its own
config, prefix, helper and `/tmp/neural-forge-1000/shm.bin`. The helper was started
and remains running. Its live settings retain working_scale=1, passes=1,
mvec_enabled=0, mvec_quality=0 and apply_model=1. The required NVIDIA DLLs were
copied by the explicit binary importer; nothing was moved from upstream.

The initial SSH-launched helper later exited while its header still reported RUNNING.
The cause of that exit is not established; header state alone is not a liveness check.
For continued validation it was relaunched with a transient user service,
`neuralforge-validation-helper.service` (oneshot with RemainAfterExit), keeping its
process tree outside the short-lived SSH session. This is not a boot-enabled service.
Verify the actual helper process and advancing counters before any benchmark.

The user service environment also inherited
`VK_INSTANCE_LAYERS=VK_LAYER_NV_dlssnr:VK_LAYER_NV_present`. This caused upstream's
NR layer to load into the compute helper. The supervisor now removes game-rendering
layer selections and activation flags from its child environment, preserving unrelated
layers such as validation and NV_present. The desktop manager's environment is unchanged.
A real child-process test verifies that separation.

Installer updates now replace files atomically. Previously they overwrote files in
place, which is unsafe when a running Wine helper or game has mapped an executable or
library. An integration test holds the old file open across an update and verifies
that it retains the old bytes while new opens see the replacement. No active process
is automatically stopped by the installer.

## Presentation test and dispatch fix

A 120-frame, 1280x720 Wayland `vkcube` smoke test with Khronos validation exits 0
without NeuralForge. With only NeuralForge loaded, the first run aborted with
SIGABRT at the very first `vkResetCommandBuffer` in `capture_pristine`, before
any helper frame was processed. GDB reproduced this and identified the reset call.

The layer allocates its private command buffers below the loader trampoline.
It was missing loader dispatch initialization for those buffers. The fix stores the
loader's device-data callback and initializes every private capture/composition
command buffer immediately after allocation, using Khronos's documented fallback
for an older loader without the callback. Direct-loader GPU tests are unaffected.
Two regression tests cover callback traversal/lifetime and the fallback dispatch slot.
No fence wait, queue timing or model setting was changed.

Reference: [Khronos loader interface, Creating New Dispatchable Objects](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md#creating-new-dispatchable-objects).

After this fix, the same NVIDIA `vkcube` test exits 0. Process maps confirm it loaded
only `libneural_forge_layer.so`, not upstream's `libVkLayer_NV_dlssnr.so`.
This confirms the abort is resolved; it does **not** establish valid end-to-end rendering.

## Correctness follow-up

The follow-up fixes make capture opt in only when the surface explicitly supports
both transfer-source and transfer-destination image usage. The layer queries the
next instance dispatch chain, never retries creation, and forwards the original
creation unchanged when a surface has an extended or unsupported configuration.
This matters because a failed replacement creation can retire an application's old
swapchain.

The layer now frees private capture and composition resources immediately before
the framework forwards `vkDestroyDevice`. This is the one teardown point where
Vulkan requires the application to externally synchronize the device and queues;
no per-frame or resize wait was added. Presentation semaphores are now assigned to
the acquired swapchain image rather than a rotating command-buffer slot. Retired
swapchain semaphores remain alive until device teardown. This follows Khronos's
[swapchain semaphore reuse guidance](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html).

On `lordnikon`, a 1,800-frame 2560x1440 Wayland `vkcube` run with NeuralForge,
the full model, host SHM, and `NEURAL_FORGE_DMABUF=0` exited normally in 19.6 seconds.
Khronos validation reported zero errors. The helper reported `model_up=1` and had
processed 192 frames at the time of the status capture. A separate 900-frame run
with synchronization validation enabled also exited normally with zero validation
errors and zero synchronization hazards. Both runs loaded only
`libneural_forge_layer.so`; upstream's NR layer was absent. The upstream config and
both installed upstream layer manifests still match their initial SHA-256 hashes.

Eleven non-fatal validation warnings remain from the pinned layer framework asking
`vkGetDeviceProcAddr` for instance-level commands. They do not come from the capture
path and are not yet resolved. There is still no measured GTA result or Feature 18
throughput claim. Smoke-test elapsed time is not game FPS or a performance result.

## GTA comparison gate

The upstream session was sampled for 61.478 seconds with the documented GTA baseline
and `DLSSNR_DMABUF=0`. Its layer counter advanced 4,540 frames (73.85 layer frames per
second); mean GPU utilization was 94.9%, mean VRAM allocation 5,851 MiB, mean board
power 232.3 W, and peak temperature 75 C. These counters are useful pipeline evidence,
but are not game FPS or a 1%-low result.

For the NeuralForge-only launch, Steam was restarted with `NEURAL_FORGE_ENABLE=1`,
`NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe`, `NEURAL_FORGE_DMABUF=0`, its isolated
implicit-layer path, and a per-session loader disable for `VK_LAYER_NV_dlssnr`.
NeuralForge loaded into the Rockstar processes and passed their swapchains through;
the explicit ownership filter did not let those processes acquire the session.
GTA itself reached 2560x1440, but reported only `TRANSFER_DST | COLOR_ATTACHMENT`
for its swapchain image usage. NeuralForge requires `TRANSFER_SRC` to copy the image
to its host transport. Since the surface capability query did not advertise it, the
layer retained the original swapchain and did no capture, resize, or helper work.
The upstream installation and configuration remain unchanged. A matched NeuralForge
benchmark is blocked until a legal GTA capture path is designed and validated.

A passive transfer probe subsequently found GTA's legal pre-present route: the game
blits `TRANSFER_SRC_OPTIMAL` render images into the `TRANSFER_DST_OPTIMAL` swapchain
images. The layer merely recorded those commands and forwarded them unchanged. The
candidate render-tap design is recorded in `RENDER_TAP_DESIGN.md`; it has not been
enabled for rendering or benchmarked.

The guarded render tap was then validated live. GTA's source images were observed
through Synchronization2 barriers as `GENERAL -> TRANSFER_SRC_OPTIMAL -> GENERAL`.
NeuralForge captures only after the source has returned to `GENERAL`, transitions it
to `TRANSFER_SRC_OPTIMAL` for its private copy, and restores `GENERAL`; the swapchain
remains a `TRANSFER_DST` output. Helper frames advanced from 219 to 336 on first use,
with `model_up=1` and `NEURAL_FORGE_DMABUF=0` throughout.

An initial 61.413-second NeuralForge interval advanced 475 layer frames (7.73 layer
frames per second), with 34.9% average GPU utilization, 5,235 MiB VRAM, 77.1 W mean
power, and 55 C maximum temperature. This cannot be compared as game FPS or against
the earlier upstream interval because the GTA scene and GPU workload were not held
constant. It does demonstrate that the current fully synchronous host transport is
the next performance bottleneck to instrument and pipeline.

A second, steady-state 61.379-second interval advanced 474 layer frames (7.72 layer
frames per second), with 35.6% average GPU utilization, 5,210 MiB VRAM, 76.9 W mean
power, and 53 C maximum temperature. The repeat confirms the synchronous pipeline
limit is reproducible rather than startup warm-up behavior.

The ownership filter was exercised with two temporary names for the same `vkcube`
binary. A process launched as `explorer.exe` was excluded, created only pass-through
swapchains, and left `helper_frames` unchanged. A process launched as
`GTA5_Enhanced.exe` with `NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe` acquired the
NeuralForge lease, used a non-pass-through swapchain, and advanced helper frames
from 484 to 530. These are process-filter tests, not a GTA launch. They confirm the
intended launcher exclusion and explicit-target path without touching Steam, GTA, or
the upstream install.

Keep the PR in draft until the review accepts these changes and the matched GTA
benchmark in PHASE1.md has been run. The user's helper, model-resolution and
DMA-BUF constraints remain in force; later optimization features are unimplemented.

## 2026-09-15 — eleven validation warnings: not reproduced; root cause identified

Repository renamed to `labj1987/neural-forge` on GitHub; local checkout's remote and
directory were updated to match, confirmed against the renamed repository. The Phase 1
PR above is merged.

Attempted to reproduce and fix the eleven non-fatal validation warnings from Phase 1
item 4 before continuing. A freshly built `libneural_forge_layer.so` was deployed
alongside the already-installed one (`~/nf-validate` via `VK_ADD_LAYER_PATH`, not
`VK_LAYER_PATH` -- the latter replaces rather than extends the default search path
and hides the system's own `VK_LAYER_KHRONOS_validation` manifest, which is why an
earlier attempt in this same session saw zero output and turned out not to have
validation loaded at all). With `VK_LAYER_KHRONOS_validation:VK_LAYER_neuralforge_neural`
confirmed active (`VK_LOADER_DEBUG=layer`), a 1280x720 Wayland `vkcube` run against the
currently-installed NeuralForge build produced **zero validation warnings or errors**,
with and without `VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT`, and `VK_LOADER_DEBUG=all`
showed nothing related to `vkGetDeviceProcAddr` beyond expected platform-surface-name
misses (Win32/Android/iOS/etc., irrelevant on this platform).

The likely source, found by reading the pinned `vulkan-layer` framework
(`google/vk-layer-for-rust` at `102d87cd`, `vulkan-layer/src/lib.rs` around its
`create_device` trampoline): it builds this layer's device dispatch table via
`ash::Device::load`, called with the *instance's* `get_instance_proc_addr` slot
replaced by `vkGetDeviceProcAddr` -- a deliberate trick to resolve the whole
`ash::Device` function table generically. The framework's own comment acknowledges
this can make the loader "complain about internal vkGetDeviceProcAddr called for
<function name>" for instance-level commands and calls it benign. This project's own
code (`crates/layer/src/device.rs`) only resolves six clearly device-level commands
itself and is not the source.

This was not reproduced live today, so it is not fixed. Either the specific
validation-layer version here (`1.4.341`) does not flag this pattern, or it only
surfaces under conditions this `vkcube` run did not match (GTA's actual Xwayland
surface path through Proton/winevulkan, rather than native Wayland). Re-verify against
a real GTA session, or against `vkcube` run through Xwayland specifically, before
concluding this needs a fork of the pinned framework -- patching a third-party git
dependency is a real undertaking and should not be started on an unreproduced report.

Also added `bench.sh` (since replaced by `scripts/gta-bench.sh`) for Phase 1 item 1 (the repeatable native/upstream/
neuralforge benchmark script). It restarts Steam under each mode's environment (an
already-running Steam client does not pick up a new shell's exported vars -- confirmed
the hard way in the 2026-09-14 session above), waits for `GTA5_Enhanced.exe`, and then
**stops and waits for a human to confirm the saved route/scene has been reached**
before starting the timed sample -- it cannot drive the car itself. The matched 3x
benchmark this phase's exit gate requires has still not been run.

## 2026-09-15 (later) -- Phase 2: non-blocking capture pipeline implemented and
## validated without real gameplay; the GTA measurement itself is not done

Implemented `ASYNC_CAPTURE_DESIGN.md`'s two-slot pipeline: `crates/layer/src/capture.rs`
gained `CapturePipeline` (two independent `CaptureBuffer` slots, each tracking its own
`pending: Option<(width, height, proxy_format)>`), `poll_pipeline_capture` (non-blocking
`vkGetFenceStatus`, never `vkWaitForFences`), and `submit_pipeline_capture` (record +
submit, never waits). `run`'s hot path now polls before deciding whether to submit,
gated on the same "only one outstanding wire request" rule the transport already had.
The one place this adds a real blocking wait is `ensure_pipeline`'s resize/queue-family-
change path -- draining whichever slot is still `pending` before destroying it, not on
the steady-state per-frame path. The old single-resource `CaptureResources`/`ensure`/
`capture_pristine` path is unchanged in behavior and still serves `run_sync` (the
`debug_view`/`capture_request` same-frame-correctness cases) and `run`'s CPU-only
write-back fallback -- deliberately kept as a separate resource from the new pipeline
rather than sharing one slot type across two different contracts. `capture_pristine`
itself (now dead once `run`'s two call sites moved to the pipeline) was removed; its
command-recording sequence was factored into `record_capture_commands`, shared with
the new pipeline's submit path.

Also added `NEURAL_FORGE_HELPER_DELAY_MS` (test-only) to `neural-forge-helper`: an
artificial per-response delay, read once at startup, applied right before
`seq_resp` is published -- the real-hardware equivalent of the existing Rust
integration test's fake in-process helper thread.

**Validated on `lordnikon`:**
- `cargo test -p neural-forge-layer` (all 45 native tests, 1 ignored) passes unchanged
  on the dev machine.
- The layer crate's release test binary was copied to `lordnikon` and run directly
  against the real NVIDIA driver with `VK_LAYER_KHRONOS_validation` and
  `VK_LAYER_VALIDATE_SYNC=1` (the current, non-deprecated sync-validation setting --
  `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` is deprecated and
  silently loses to it if both are set). All 45 tests passed, including
  `run_never_blocks_on_a_slow_helper_and_eventually_composites` (the test that
  specifically asserts a single `capture::run` call never takes anywhere near a slow
  helper's own delay). The only validation output was
  `VUID-VkImageMemoryBarrier-{old,new}Layout-parameter` on `PRESENT_SRC_KHR`, a known
  artifact of this test's own minimal device (created without `VK_KHR_swapchain`) --
  confirmed pre-existing and unrelated to this phase's change, not a synchronization
  hazard: no `SYNC-HAZARD-*` message appeared anywhere in either run.
- `neural-forge-helper.exe` with `NEURAL_FORGE_HELPER_DELAY_MS=250` set was run alone
  against `lordnikon`'s real Proton/Wine runner for 20+ seconds with no crash.
  A separate attempt earlier the same session, run immediately after starting a
  *second* helper instance against the same Wine prefix while the first was still
  live, did die silently within seconds -- reproduced once, and explained by
  concurrent processes sharing one Wine prefix (a known hazard, unrelated to this
  change) rather than the delay code itself once isolated. Recorded here in case it
  recurs: if so, it needs its own investigation, not an assumption this note already
  covers it.
- `vkcube` itself -- both before and after this phase's change, confirmed against the
  unmodified Phase 1 binaries as a control -- never advances `layer_frames` at all
  (`shmctl status` stays at 0) under `NEURAL_FORGE_ENABLE=1` regardless of which code is
  installed. This matches `RENDER_TAP_DESIGN.md`: `vkcube` never performs the
  `TRANSFER_SRC_OPTIMAL` -> swapchain blit the render tap looks for, so capture never
  engages for it, old pipeline or new. `vkcube` is therefore only useful here for
  confirming the pipeline doesn't corrupt or hang an *inactive* capture path, not for
  observing the async decoupling with a real captured frame and a real delayed
  answer end to end -- that needs the render tap actually active, which needs GTA.

**Not done, and not claimed**: no GTA session was run this phase (out of scope for
this session -- see PHASE1.md's own unmet benchmark gate above, still open). Phase 2
item 6 calls for "then on GTA. Report layer frames/sec and GTA fps against the Phase 1
table" -- that comparison, and confirmation that GTA fps recovers to within ~10% of
native with neural on, is still outstanding and needs a live session.

## 2026-09-15 (later still) -- Phase 3 step 1: device-extension injection

See `EXTERNAL_MEMORY_HOST_DESIGN.md` for the design. Summary: this project had never
hooked `InstanceHooks`/`InstanceInfo` before (`Layer::InstanceInfo` was the pinned
framework's own `StubInstanceInfo`, a real no-op) -- the game's own `vkCreateDevice`
call determines its device's extensions, and the only hook point that runs *before*
that call actually happens (letting a layer add one) lives on `InstanceHooks`, not the
`DeviceHooks` trait everything else in this crate already implements. Added
`NeuralForgeInstanceHooks::create_device`: adds `VK_EXT_external_memory_host` when the
physical device supports it and the app hasn't already requested it, retries with the
exact original request if the extended one is refused, and otherwise (the overwhelming
common case) returns `LayerResult::Unhandled` -- identical to not hooking at all.

**Validated on `lordnikon`** (RTX 5070, driver 615.71.09): `vkcube` at 1280x720 and at
2560x1440 (the real GTA render resolution) under `VK_LAYER_KHRONOS_validation` with
`VK_LAYER_VALIDATE_SYNC=1`. Both logged `external_memory_host: true` (confirming the
injection actually happened -- this driver does advertise the extension) with zero
validation errors and zero synchronization hazards; the capture pipeline kept advancing
`layer_frames` normally at both resolutions (474 in 20s at 2560x1440), no regression
from the pre-Phase-3 behavior. `cargo test` (all 45 layer tests, run 4x to rule out
flakiness after one unrelated one-off failure in an ownership test unrelated to this
change) stayed green throughout.

**Not validated**: a device that genuinely lacks the extension (this driver always has
it, so the "not supported" `Unhandled` branch is reviewed, not exercised live); GTA
itself. **Not done yet**: the actual memory import this unblocks -- `CapturePipeline`
still always allocates and copies through its own staging buffer regardless of
`external_memory_host`'s value, which today is logged and immediately discarded, not
stored or read anywhere yet.

## 2026-09-15 (later still) -- Phase 3 step 2: the actual zero-copy import, `DirectCapture`

See `EXTERNAL_MEMORY_HOST_DESIGN.md` for the full design. Summary: added
`capture::DirectCapture`, a single capture slot whose device memory is imported
directly from the live SHM proxy region, so a capture's `vkCmdCopyImageToBuffer`
writes straight into shared memory. `run` now shares one `poll_or_submit_capture`
helper across both `DirectCapture` and `CapturePipeline`, deciding per call (cheaply)
which to use.

Wrote a dedicated test (`capture::tests::direct_capture_writes_straight_into_imported_host_memory`)
that builds a device with the extension actually enabled, fills a source image with a
known color, captures it through `DirectCapture`, and asserts every captured byte
matches exactly -- not just "ran without crashing". **This test found two real bugs**
on first real-hardware run (`lordnikon`, `VK_LAYER_VALIDATE_SYNC=1`): a missing
`VkExternalMemoryBufferCreateInfo` on the buffer (a real production bug,
`VUID-vkBindBufferMemory-memory-02985`) and a misaligned `allocationSize` in the test
itself (`VUID-VkMemoryAllocateInfo-allocationSize-01745`, this NVIDIA driver's real
`minImportedHostPointerAlignment` is 4096) -- neither was caught by this project's
local software Vulkan ICD, which accepted both mistakes silently. Both fixed; the test
now passes on both the local software ICD and `lordnikon`'s real driver under full
synchronization validation, with zero hazards. The full test suite (47 tests, 1
ignored) stayed green on `lordnikon` throughout.

**Not done**: no GTA session. The real payoff this phase promises -- layer fps moving
toward upstream's ~74/s on `lordnikon` -- is still unmeasured; `DirectCapture` itself
has never run against a real game, only a synthetic single-frame test and (indirectly,
for device creation only) `vkcube`.

## 2026-09-15 (later still) -- Phase 3 step 3: helper-side import, and two real crashes
## found only after everything above had already validated clean

See `EXTERNAL_MEMORY_HOST_DESIGN.md` for the full account. Summary:

- Added the helper-side half of the zero-copy import (`FrameResources::imported_proxy`/
  `imported_answer` in `crates/helper/src/frame.rs`) -- confirmed live on `lordnikon`
  via a new tool (`crates/protocol/examples/trigger_helper_roundtrip.rs`) that drives a
  real request/response round trip against a real, running helper with no game
  involved: `imported_proxy=true imported_answer=true`, no crash across dozens of
  round trips. `EvaluateFeature` itself did not produce a real (non-echoed) answer for
  synthetic garbage pixel input -- confirmed as pre-existing by testing the identical
  input against the pre-Phase-3 helper build, which fails the same way.
- **Found two real, live bugs after the above already looked done** -- both survived a
  clean `cargo test` and multiple already-validated `vkcube` runs:
  1. Real undefined behavior in `NeuralForgeInstanceHooks::create_device`: calling
     `slice::from_raw_parts` on `pp_enabled_extension_names` without checking
     `enabled_extension_count == 0` first, which `vkcube` on this exact machine hits
     live (a null pointer is a legal C convention for "zero extensions" that Rust's
     `from_raw_parts` does not tolerate even at length zero). Caught by
     `scripts/smoke-test.sh`'s debug-mode UB checker aborting the process -- every
     earlier *release*-mode `vkcube` validation run this whole session had the
     identical UB and simply never visibly crashed, which is a worse outcome than a
     visible one, not a better one.
  2. `vk::ExtExternalMemoryHostFn::load(...)` panics (not gracefully fails) if the one
     function it's asked to resolve doesn't load -- hit live on `lordnikon`, inside
     `build_imported_capture_buffer`, on a device that had `external_memory_host: true`
     at creation. Fixed in both the layer and the helper by resolving
     `vkGetMemoryHostPointerPropertiesEXT` by hand via `get_device_proc_addr` (a real
     `Option`, checked explicitly) instead of the panicking `::load()` helper.
- Re-validated everything after both fixes: `scripts/smoke-test.sh`, the full
  `cargo test` workspace suite, and `vkcube` at 1280x720/2560x1440 under
  `VK_LAYER_KHRONOS_validation` + `VK_LAYER_VALIDATE_SYNC=1` (including one run with a
  real helper attached) -- all clean, no crash, zero validation errors or hazards.
  Also confirmed, while re-validating: `vkcube`'s own swapchain *is* admitted for this
  layer's plain (non-render-tap) capture path (`pass_through=false`) -- the earlier
  "capture never engages for `vkcube`" notes in this file were specifically about the
  render tap (GTA's own route), not a blanket statement; worth remembering next time a
  session reaches for that assumption.

**The lesson worth carrying forward, not just the two fixes**: "passed every test,
validated clean on real hardware multiple times" was true at the time and still missed
two real bugs, because neither Rust panics from a third-party crate's own internal
`::load()` helper nor a plain null-pointer UB gap are things Vulkan validation layers
or `cargo test` check for. Re-run `scripts/smoke-test.sh` specifically after touching
device-creation-adjacent code, not just a validated `vkcube` session -- it is the one
check in this project that catches this exact class of bug.

## 2026-09-15 (later still) -- Phase 3 step 4: protocol v3, a second independent
## request/response slot -- validated end to end on real hardware

See `PROTOCOL_V3_DESIGN.md` for the design; this entry is the real-hardware evidence.
`lordnikon`, RTX 5070, driver 615.71.09, a fresh `git clone` of each commit (not just
this dev machine's own software ICD):

- `cargo test -p neural-forge-layer` (48 tests): clean and deterministic across several
  repeated runs.
- `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1` against
  `capture::` tests: the same pre-existing cosmetic `VUID-VkImageMemoryBarrier-*`
  warning as the pre-v3 commit (confirmed identical via a second `git worktree` at the
  parent commit) -- not a regression, zero new validation errors or sync hazards.
- `scripts/smoke-test.sh`: clean, `external_memory_host: true`, no abort.
- `crates/protocol/examples/trigger_helper_roundtrip.rs` (extended with a slot
  argument) against a real running helper (Proton-CachyOS, the real
  `nvngx_dlssnr.dll`): both slots answered correctly when triggered *concurrently* --
  two processes launched at once, one per slot -- completing in well under a second
  total, repeated six times at two resolutions with no failures. The one informational
  mismatch seen (`seq_ok`, on one of six runs) is the expected, harmless consequence
  of that one field being deliberately shared rather than duplicated per slot (see
  `PROTOCOL_V3_DESIGN.md`) -- nothing reads it back, so it isn't a correctness gate.

**A discrepancy from the 2026-09-15 (later still) Phase 3 step 2 entry above, worth
recording rather than quietly overwriting**: that entry states `vkcube`'s own
swapchain *is* admitted for this layer's plain capture path (`pass_through=false`).
A `vkcube` run this session, same layer, same machine, under Wayland/Xwayland with
`VK_LAYER_VALIDATE_SYNC=1`, showed `pass_through=true` for both of its swapchains --
capture never engaged. Not re-investigated (out of scope for what this entry
validates -- the *wire protocol*, exercised directly via `trigger_helper_roundtrip.rs`
instead, deliberately bypassing the capture path entirely). Whether this is a real
regression, a Wayland-vs-whatever-surface-the-earlier-session-used difference, or a
driver/environment change since that entry was written is genuinely unknown -- flagged
here so a future session doesn't treat the older entry's `pass_through=false` claim as
still-current without checking again first.

**A real process lesson from this session, not a code bug**: the first two attempts
at the concurrent-slot test above looked like a serious cross-slot race (both
processes reporting the identical sequence number, one request left permanently
unanswered) until traced back to a stale deployed clone on `lordnikon` -- the
diagnostic tool's own slot-argument support had been pushed but not yet `git pull`ed
into the scratch clone used to build it, so both invocations silently raced for slot 0
alone. Re-confirmed clean immediately after pulling. Worth remembering the next time a
real-hardware result looks like a race: diff the deployed *source* against what was
actually pushed, not just re-run the same (possibly stale) deployed binary, before
concluding anything about the code itself.

## 2026-09-15 (later still) -- Phase 4: DMA-BUF transport, blocked on a real
## Wine/NVIDIA constraint, not an implementation gap

See `DMABUF_TRANSPORT_DESIGN.md` for the full writeup; this entry is the real-hardware
evidence. `lordnikon`, RTX 5070, driver 615.71.09, Proton-CachyOS:

`crates/helper/examples/dmabuf_probe.rs` built a real exportable Vulkan buffer inside
the Wine-hosted helper (`VkExportMemoryAllocateInfo`/`VkExternalMemoryBufferCreateInfo`
requesting `OPAQUE_WIN32`), got a real, non-null win32 handle from
`vkGetMemoryWin32HandleKHR` (resolved by hand, not via `ash`'s panicking `::load()`
convenience wrapper -- same fix class as `EXTERNAL_MEMORY_HOST_DESIGN.md`'s own), then
called Wine's `ntdll.dll` export `wine_server_handle_to_fd` (confirmed present in this
exact Proton build via `objdump -p ntdll.dll` before writing any code) against it,
wrapped in `guard::guarded`.

Result: no fault (`seh=0` -- the guessed Wine ABI is correct), but a real
`STATUS_OBJECT_TYPE_MISMATCH` (`0xC0000024`) NTSTATUS. Every step up to that call
succeeded; this is not a crash or a wrong-signature guess, it's Wine's own fd/handle
bridge telling this probe the handle's underlying object isn't one it can unwrap to a
Unix fd -- because a Vulkan external-memory win32 handle was never a wineserver-opened
object to begin with (see the design doc for the fuller reasoning, including the
likely NVIDIA-internal-shared-surface explanation for why there may be no real
`dma_buf` behind this handle type on this driver at all).

**Real process lesson from this session, worth remembering**: plain `println!`/stdout
from a Wine process launched via `proton run` did not reach the invoking shell at all,
even piped to a file, across 20+ real seconds of the process legitimately running
(confirmed via `user`/`sys` time in the shell's own `time` output) -- switching the
probe to this crate's own `log!`/`logging::flush()` (writing through `NEURAL_FORGE_LOG`
instead of stdout) fixed it immediately. This project's own `logging.rs` module
already exists for exactly this class of problem ("whenever this binary's stdout/
stderr isn't a real terminal... Proton/Steam has redirected it") -- use it for any
future Wine-side diagnostic tool from the start, don't lose time to a silent process
assuming `println!` is good enough first.

## 2026-09-15 (later still) -- Phase 4 reverse direction: also blocked, different
## reason, confirmed without Wine in the loop at all

Same session, immediate follow-up. `crates/layer/examples/dmabuf_export_probe.rs`
(native Linux) allocated a real `VK_EXT_external_memory_dma_buf` buffer, got a real fd
via `vkGetMemoryFdKHR`, and held it open. `crates/helper/examples/dmabuf_import_probe.rs`
(Windows, run under the same Proton-CachyOS/prefix setup as the forward-direction
probe) took that pid+fd and called `CreateFileW` on `Z:\proc\<pid>\fd\<fd>`.

Result: `CreateFileW` failed outright. Root-caused below Wine entirely -- with the
exporter's fd confirmed still open (`ls -la /proc/<pid>/fd/<fd>` showed a live entry),
a direct, non-Wine `cat`/Python `os.open()` on that exact path also failed, with
`ENXIO`. `readlink` on the fd entry showed `/dmabuf:`: dma-buf fds are anon-inode-backed
or the same reason `epoll`/`eventfd` fds are, and Linux does not support re-opening an
anon-inode fd via `/proc/<pid>/fd/<N>` from any process -- only `dup()` or `SCM_RIGHTS`
fd-passing over a Unix socket can hand one to another process. Confirmed this is not a
Wine quirk (Wine's own `Z:\` -> `/` mapping is real and otherwise works, per
`dosdevices/z: -> /`) before concluding anything about the driver or Vulkan layer at
all -- same "isolate the failure below the layer you suspect first" discipline as the
stale-git-clone lesson above. See `DMABUF_TRANSPORT_DESIGN.md` for the full writeup and
what a working transport would actually need (`SCM_RIGHTS`, not `/proc/pid/fd`).

Test processes/prefix wineserver cleaned up afterward; no leftover state on
`lordnikon`.

## 2026-09-16 -- Phase 2 item 6, first real GTA data: fps collapse confirmed. (The
## motion-vector explanation this entry originally offered was wrong -- see the
## correction at the end of it.)

Alex ran the first real GTA session against this exact codebase (v0.1.59, confirmed:
the AppImage Alex's desktop launcher runs, `~/AppImages/neuralforge.appimage`, hashes
byte-for-byte identical to the published `v0.1.59` GitHub release asset). FPS capped
at 120: neural rendering on dropped the game to the mid-to-high 20s, with real,
noticeable ghosting Alex reports as absent from upstream, alongside a genuinely
positive signal -- the lower fps itself "feels like it should," not laggy the way
earlier (pre-Phase-2) testing did, matching Phase 2's own design goal (the two-slot
non-blocking capture pipeline removing the present-hook's own fence wait).

**Real telemetry from that session** (`~/.local/state/neural-forge/helper.log`, both
protocol-v3 wire slots actively alternating -- frame numbers in the 1600s, confirming
Phase 3's double buffering was genuinely live, not just idle): `NVSDK_NGX_VULKAN_
EvaluateFeature` itself took a very consistent **~19-22ms per frame** end to end
(upload ~2-3ms, eval ~19-22ms, download ~0.6-3.6ms). This is the first real-gameplay
confirmation that the model's own evaluation cost, not host-transport, dominates the
per-frame budget -- directly answers the open question `DMABUF_TRANSPORT_DESIGN.md`'s
own recommendation left unresolved ("worth reassessing DMA-BUF... once that real
measurement exists and shows transport cost... is still the dominant cost left to
cut"): transport (upload+download, ~3-5ms) is small next to eval (~20ms) even before
any DMA-BUF-style savings, which is exactly why both DMA-BUF directions turning out to
be dead ends (see the two 2026-09-15 entries above) costs this project less than it
might have.

**Root-caused the ghosting, not yet re-confirmed live**: `config.ini` on `lordnikon`
had only ever persisted `set_enabled=1` -- nothing else -- meaning the session ran on
every other setting's real code default, and `ShmHeader::init_defaults` sets
`mvec_enabled` to `0` (off), exactly matching `PHASE1.md`'s own documented preserved
baseline ("motion_enabled=0, motion_quality=0"). No motion-vector input is a
well-understood, textbook cause of exactly this ghosting/smearing symptom during
camera or object motion in any temporally-reconstructing neural upscaler. **Not yet
verified**: whether turning `mvec_enabled` on actually resolves it live -- that's the
concrete next test, not attempted this session (needs Alex at the keyboard again).

**A real, separate GUI label bug found investigating the config**: the "Estimate
motion vectors" switch row's subtitle read "On by default" while the actual, documented
default is off -- fixed (now "Off by default"), see the same commit as the
`ViewSwitcher`/`ToolbarView` migration below.

**This does not yet satisfy Phase 1's own benchmark gate** ("GTA fps recovers to
within ~10% of native with neural on") -- a drop from a 120fps cap to the mid-to-high
20s is a large gap, not a 10% one. Whether that gap closes once motion vectors are on,
or reflects the real, currently-unoptimized cost of a `passes=1`/full-resolution
neural pass on this hardware, is still open. Phase 2's own exit gate (non-blocking
pipeline actually helping perceived smoothness at a real, lower fps) has real,
first-hand positive evidence now ("not laggy... like with testing before"); the raw
fps number itself is Phase 1's gate, not Phase 2's, and stays open pending a
motion-vectors-on retest.

**Correction, later the same day -- the motion-vector explanation above is wrong, and
so is the "not laggy" read.** Two things this entry got wrong, kept here rather than
rewritten so the reasoning trail stays honest:

1. `mvec_enabled` was never a live variable. `crates/layer/src/shm.rs::
   prepare_motion_resources` has had an unconditional early `return` in front of all
   of its real logic since 2026-09-14 (commit `72d7a55` -- a deliberate stub after a
   real NVIDIA driver crash when the private optical-flow device is created during a
   live swapchain transition). Motion-vector estimation has not run in *any* session,
   whatever the toggle said. Alex re-tested with the switch flipped on: ghosting
   unchanged, fps unchanged -- as it had to be, the code path is unreachable. The GUI
   now grays the whole Motion group out and says why (v0.1.62).
2. The actual mechanism behind the ghosting is documented in this project's own code
   (`crates/layer/src/capture.rs::run`, the "re-present the most recently retained
   answer on every call" block): the model cannot evaluate every frame (native ~330fps
   here vs. a ~25-30ms round trip per answer), so one answer's *delta* against the
   frame it was computed from gets re-applied, at full strength, to every presented
   frame until the next answer lands -- roughly 8-10 real frames at this session's
   rates. During camera or object motion that delta is misaligned with the frame it's
   applied to, which reads exactly as ghosting. That design was chosen deliberately
   (per the doc comment there, and Alex's own earlier authorization) to remove the
   native/processed *flicker* an earlier version had, and the comment already names
   the fix for the staleness it trades for: motion-vector reprojection -- the exact
   feature item 1 says is currently stubbed out. The two findings are one story.
3. On the second, longer session (v0.1.61, after the render-tap leak fix), Alex's own
   read was "input lag and stuttering", not the smoother feel reported here -- the
   earlier "feels right" line should not be taken as a settled Phase 2 result. Two
   other things were true during both sessions and confound the fps/lag numbers
   specifically: RustDesk *and* GNOME's own remote-desktop daemon were running at the
   same time (two independent screen-capture/encode pipelines on the same GPU), and
   the helper had been started by hand (v0.1.61 auto-starts it now). Whether the 30fps
   figure is the pipeline's real cost or partly that overhead is still not separated.

**Upstream comparison, finally done properly (and it overturns the "inherent to the
model" read above -- item 2 is a real NeuralForge bug, not a law of physics).** Every
earlier "upstream" run that day was invalid two ways at once: the Steam client was
never restarted between tests (it kept the first launch's `NEURAL_FORGE_ENABLE`
environment -- the `bench.sh` gotcha this file documents), *and* GTA's saved
Steam launch options bake in `NEURAL_FORGE_ENABLE=1 ... %command%`, which Steam applies
per-game regardless of the client environment. The fix was `NEURAL_FORGE_DISABLE=1`
(the layer manifest's own `disable_environment`, honored at the Vulkan-loader level
over any launch option) plus a genuine full Steam restart. Confirmed isolated via
Steam's own `console-linux.txt`: `[dlssnr-layer] ... RTX 5070 (inert=0 enabled=1)`,
zero `[neural-forge-layer]` lines for the whole session.

Real result, Alex's own eyes plus telemetry: **upstream has no ghosting and much
better fps.** The hard number that explains it -- upstream during real gameplay sat at
**98% GPU / 227 W** (the GPU saturated, running at native rate), while NeuralForge's
own earlier sessions sat at **~30-46% GPU / ~90-115 W** (the GPU *starved*, blocked
~60-70% of the time). And upstream's layer log has **no per-frame work at all** -- only
setup events -- where NeuralForge logs a barrier-tracking line on essentially every
frame (41,678 in one session).

That reframes the whole thing. Both NeuralForge symptoms are one root cause, and it is
NOT the model or the IPC latency (upstream runs the identical NGX model and doesn't
have either problem): **NeuralForge does expensive full-frame work on the game's own
queue on every present** -- a full-frame host readback + large `memcpy` to capture a
new frame whenever no round trip is in flight (most frames once the helper keeps up),
plus a GPU compose it makes the present wait on -- which throttles the game's rendering
down to ~30 fps. Upstream evidently leaves the great majority of frames as untouched
native passthrough (native cost, native fps, no staleness) and only pays capture/
composite cost on the frames a fresh answer is actually ready for. NeuralForge's
"re-present the held answer on every frame" design (added in v0.1.49/v0.1.50 to kill an
alternation *flicker*) is very likely the shared cause of BOTH the fps collapse (per-
frame cost) AND the ghosting (a stale answer re-applied across ~8-10 moving frames) --
a trade that upstream demonstrates you do not have to make. This is the real,
NeuralForge-specific lead to chase; the motion-vector/"inherent latency" framing in
item 2 above is superseded.

**Measured it, found the real culprit, and fixed most of it (v0.1.64).** Added a
resolution-realistic benchmark (`capture::tests::capture_hot_path_cost_per_present`,
env-gated behind `NEURAL_FORGE_BENCH`, runs on real hardware via the release test
binary) that drives `capture::run` at 2560x1440 with a keeping-up fake helper and
separately times the CPU cost (the `run` call on the game's present thread) and the GPU
cost (queue drain). Baseline on `lordnikon`:

- **CPU (run() on the present thread): p50 = 87 ms.**
- GPU (queue drain): p50 = 1.7 ms.

So the earlier "expensive work on the game's *queue*" read was wrong too -- the GPU
work is a rounding error. The ~87 ms is CPU-side, on the game's own present thread:
`run` reads the freshly-captured full frame back from a HOST_VISIBLE|HOST_COHERENT
staging buffer, and `build_capture_buffer` was picking the *first* such memory type,
which on NVIDIA is the small device-local BAR region -- CPU reads of it go uncached
over PCIe at well under 1 GB/s, ~87 ms for one 1440p frame. That single stall on the
present thread is the fps collapse: ~87 ms/present is an ~11 fps ceiling, and
interleaved with cheaper frames lands right at the ~30 fps observed in-game.

The fix is one memory-type preference in `build_capture_buffer`: prefer
HOST_VISIBLE|HOST_CACHED|HOST_COHERENT that is *not* DEVICE_LOCAL (cached system RAM,
fast CPU reads, still coherent so no invalidate needed), falling back to the old
selection only if no such type exists. Re-measured with the same benchmark:

- **CPU: p50 = 87 ms -> 5.7 ms (~15x).**
- GPU: unchanged (~1.6 ms).

Total per-present layer cost ~89 ms -> ~7 ms, i.e. an ~11 fps ceiling -> ~140 fps
ceiling from this alone. Full layer suite (54 tests) and the real-loader smoke test
still pass. **Still needs live GTA validation for the actual in-game fps and whether
ghosting improves** (the held-answer re-presentation is a separate axis this does not
touch) -- but the dominant cost is now measured and gone, not theorised. The benchmark
stays in the tree as the objective metric for any further hot-path work.

## 2026-09-16 (same day) -- GUI: migrated off deprecated `ViewSwitcherTitle`/`Bar`,
## caught a real layout bug via screenshot before it shipped

`CLAUDE.md`'s "Deliberately not done" item ("AdwViewSwitcherTitle/Bar, not
AdwToolbarView + AdwViewSwitcher + AdwBreakpoint... revisit only after checking the
actual CI-installed libadwaita version") got new information: a real CI run's own
`apt-get install` log confirmed GitHub Actions' `ubuntu-latest` ships libadwaita
**1.5.0** (well past the v1.4 minimum), and that same run's build output was already
emitting real deprecation warnings for `ViewSwitcherTitle`. Migrated
`crates/gui/src/ui.rs` to a plain `AdwViewSwitcher` in the header, `AdwToolbarView` for
the top/bottom bar layout, and an `AdwBreakpoint` to swap it for the bottom
`AdwViewSwitcherBar` below a width threshold -- the explicit replacement for what
`ViewSwitcherTitle`'s internal auto-collapse used to do implicitly.

**Real screenshot verification (this sandbox's X11 workaround, an isolated
`NeuralForgeVisualTest`-application-id scratch build) caught a genuine bug the first
pass shipped**: at the app's own natural default size (1340px wide -- the
`PreferencesPage` content's own natural width, not the coded 620px hint), the header's
`Wide`-policy `ViewSwitcher` rendered with all six tab labels truncated to a single
character each ("M...", "C...", "D..."...) -- plenty of raw window width, but not
enough left over for six full icon+label tabs once the header's own symmetric
title-centering and the About button ate into it. The originally-chosen breakpoint
(550sp) was far too low for this specifically content-heavy, six-tab window, leaving a
wide "dead zone" where the header switcher showed but had no room to render properly.
Fixed by raising the breakpoint to 1400sp (just above the content's own natural
width) and re-verified both states with real screenshots: the bottom `ViewSwitcherBar`
at the natural default size (all six tabs fully legible), and the header `ViewSwitcher`
at a genuinely wide size (tested both 1600px, still correctly showing the bottom bar,
and maximized ~2938px, correctly showing the header switcher -- this sandbox's
effective 2x display scale, `Xft.dpi=192`, means 1400sp maps to roughly 2800 real
pixels here, exactly the range those two results bracket). A plausible-looking fix
based on libadwaita's own docs alone would have shipped the truncated-label bug --
this is exactly the class of thing `feedback-screenshot-workaround-sandbox` (project
memory) says to get a real screenshot for before trusting a GTK layout change.

## 2026-09-16 (later) -- a real GTA crash traced to an NVIDIA driver bug external to
## this project, not a NeuralForge regression; `lordnikon` left down, needs a physical
## restart

Alex reproduced the previous session's GTA crash a second time (over RustDesk, at
work). Real evidence this time, not just an absent log:

- `nvidia-smi` returned `ERR!` across every field -- the driver could no longer query
  the GPU at all.
- `journalctl -k` (further back than `dmesg`'s own ring buffer, which had already
  rotated past the real event under a flood of secondary `NV_ERR_RESET_REQUIRED`
  assertion spam) showed the actual sequence: `NVRM: Xid ... 109, pid=<GTA5_Enhanced.exe's
  own pid>, ... errorString CTX SWITCH TIMEOUT` repeating every 4-5 seconds for over
  half a minute against the game's own GPU channels, followed by `Xid 119` -- the
  GPU's onboard GSP firmware itself timing out on its own internal heartbeat/RPC to
  the driver ("GSP-RM is slow", 45s timeout) -- which is what actually killed the GPU
  (`GPU_IN_FULLCHIP_RESET` required from that point on; this is a firmware-level fault,
  not something recoverable by any userspace/Vulkan-layer code, including this
  project's own).

**Traced to a known, unresolved, external NVIDIA Linux driver bug, not this
project**: `Xid 109 CTX_SWITCH_TIMEOUT` under Proton is a long-running, widely
reported issue (NVIDIA forums, `forums.developer.nvidia.com/t/xid109-ctx-switch-timeout-driver-crashes-in-many-applications/283722`,
and a matching RTX 5090/Blackwell report at `github.com/NVIDIA/open-gpu-kernel-modules/issues/1097`)
spanning driver branches from at least 545.x through 595.x (lordnikon runs 615.71.09,
newer than every version in those reports) and a wide range of GPUs (RTX 2080 through
5090) and completely unrelated games (CS2, Elden Ring, Apex Legends, Path of Exile,
Assassin's Creed Shadows, Crimson Desert) -- none of them running any DLSS/neural
rendering layer at all. NVIDIA staff acknowledged it internally ("bug 5052028") with
no fix shipped as of the driver versions discussed. This is strong evidence the crash
is an external, pre-existing driver/firmware bug lordnikon's session happened to hit,
not something this project's Vulkan hooking introduced -- worth remembering the next
time a GTA crash gets reported: check `journalctl -k` for a real `Xid` line before
assuming it's this project's own code, the same "verify below the layer you suspect"
discipline as the DMA-BUF and native-NGX investigations earlier in this file.

**Workarounds other users report** (none of them applied here yet, no hardware
access): driver downgrade to the 550.x branch helped some, though with no guarantee it
still applies to a Blackwell card on 615.71.09; `PROTON_HIDE_NVIDIA_GPU=1
PROTON_ENABLE_NVAPI=1` combined with Pyroveil (already present in lordnikon's real
Proton-CachyOS install at `.../files/share/pyroveil/`, so this may just be a launch-option
change, not a new install); lower in-game resolution reduced frequency for some users,
consistent with a timing/scheduling-pressure trigger rather than a hard deterministic
one.

**Machine state**: rebooted via `sudo reboot` over SSH at Alex's explicit request: the
GPU was already unusable (`ERR!`) and unrecoverable without at least a driver reload,
so nothing was lost by rebooting that wasn't already gone. The reboot did not
complete successfully -- `lordnikon` never came back on Tailscale or plain SSH after
several minutes of polling (genuine connection timeouts, not "connection refused",
meaning it never even came back on the network, let alone finished booting). Alex
confirmed it isn't visible in Tailscale from any path and will restart it by hand.
**Do not attempt further remote recovery of `lordnikon` in a future session without
Alex confirming it's back up first** -- the machine may be stuck at POST or otherwise
requires physical presence; blind SSH/reboot attempts against a host in this state
waste a session's time for no possible benefit.

## 2026-10-02 -- false scene cuts: none found, nothing changed

**Question:** `MotionState::prepare` (`crates/helper/src/main.rs`) calls
`optical_flow::is_scene_cut` with a fixed mean-luma threshold of 40 between consecutive model
frames. A cut resets NGX's history and the optical-flow reference, so a false cut costs a frame
without temporal history. Another DLSS 5 layer measured a fixed threshold tripping about four
times a second on steady pans, and at model interval 2 Neural Forge compares frames two presents
apart, so it could be worse off here.

**Setup:** LordNikon, Neural Forge 1.0.1, GTA V Enhanced built-in benchmark (all five passes,
pass 4 = 117 s of continuous free roam), 2560x1440 at 288 Hz with the HDR desktop on (GTA's
swapchain is still 8-bit, so NR composites normally), NR on, model every 2nd frame, model
resolution 100%, motion vectors on, Alex's mods with the fixed Enable All Interiors.

**Result:**

| | Count |
|---|---|
| Model evaluations (helper frame counter) | 7121 |
| Scene cuts logged (`[mvec] scene cut detected`) | 9 (0.13% of evaluations) |
| Of those, at most in pass 4 | 3 (at most one per 39 s of continuous play) |
| History resets from a pause (`resetting model history`, 1.0.1) | 1 (a 7.9 s loading gap) |

The cut count is exact. The times are not: the helper buffers its log, so lines reach the file in
bursts (six cuts within 90 ms, three within 20 ms, while model frames are about 36 ms apart). The
first burst falls before pass 4, during the benchmark's scripted camera cuts and loading; the
second is the only one that can be inside pass 4.

**Decision:** far below one cut per ten seconds of continuous play, so the fixed threshold stays.
No adaptive baseline was built. Real fps in this run: 55.2 (pass 4).

## 2026-10-02 -- 2.0 baseline (1.0.1, mods off)

The comparator for every 2.0 performance change. Neural Forge 1.0.1 as installed on LordNikon,
RTX 5070, driver 615.71.09. Desktop 2560x1440 at 288.001 Hz, scale 1.0, HDR on (bt2100).
GTA settings.xml: 2560x1440, RefreshRate 288, Windowed 0, VSync 0, FrameLimit 0, ReflexMode 2,
FrameGenType 0. GTA's script mods off (`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`), no remote
desktop session, MangoHud as the last layer, `scripts/gta-bench.sh`. NR off = Neural Forge's
layer not loaded. NR on = model every 2nd frame unless stated, working scale 1.0, motion vectors on.

One run per configuration: the existing no-mods runs already gave every number the 2.0 gates
use, so the planned three-run set was cut short. Both single runs land on the earlier
references (93.6 / 67% and 61.6 / 89%, `docs/OPENDLSS_REVIEW.md`), which is the harness check.

| Run (pass 4, ~116 s) | Real fps | Displayed | GPU | Power | Composited/s |
|---|---|---|---|---|---|
| NR off | 93.2 | 94.6 | 67% | 148 W | - |
| NR on, model every frame | 41.9 | 41.9 | 88% | 201 W | 43.0 |
| NR on, every 2nd frame | 61.6 | 62.0 | 89% | 195 W | 63.5 |
| NR on, every 2nd frame, working scale 0.75 | 58.1 | 58.5 | 78% | 166 W | 60.3 |

`[sync]` medians over pass 4 (ms per model frame) and the helper's own timings:

| Config | total | capture_gpu | copy_out | wait_answer | helper | rest | zc | helper upload / eval / download | flow |
|---|---|---|---|---|---|---|---|---|---|
| every frame, 1.0 | 18.4 | 5.60 | 0 | 12.7 | 11.6 | 0.05 | true | 0.78 / 9.90 / 0.60 | 0.72 |
| every 2nd, 1.0 | 19.0 | 5.95 | 0 | 12.75 | 11.55 | 0.06 | true | 0.86 / 10.21 / 0.68 | 0.74 |
| every 2nd, 0.75 | 19.5 | 6.30 | 2.9 | 8.3 | 7.5 | 1.7 | false | 0.68 / 6.17 / 0.40 | 0.53 |

- Working scale 0.75 leaves the zero-copy path: `copy_out` 2.9 ms plus `rest` 1.7 ms eat most
  of the 4 ms the smaller model saves, so it is 3.5 fps slower than 1.0 at 11 points less GPU.
  That is what Phase 2 removes.
- `capture_gpu` (submit to observed completion, which includes waiting for the game's own frame)
  is 5.6-6.3 ms in play against about 0.75 ms on the first `[sync]` line of each run.
- `wait_answer - helper` is about 1.1 ms: the optical flow (0.7 ms, not part of the published
  helper time) and the two poll loops.

4K is not re-run. On record (Alex's notes, real 4K at desktop scale 1.0, mods on with Enable All
Interiors patched, which matches mods off at 1440p): NR off 82.0 (GPU 93%; mods off 83.0 at 95%),
NR on every 2nd frame 32.8 (GPU 95%), NR on + Smooth Motion 25.9 real / 52.3 displayed (GPU 97%).
The older `r-nomods-nroff-4k` run on the rig read 93.0 at 68% GPU: it ran at the 1440p desktop and
is not a 4K number.

`capture_hot_path_cost_per_present` on LordNikon (release test binary at 9f2c084):

```
capture_hot_path_cost_per_present @ 2560x1440 (200 samples, 31 composited a fresh answer):
  cpu (run() on present thread): mean=54.201µs p50=221ns p95=990ns max=5.50823ms
  gpu (queue drain after run()): mean=62.796µs p50=36.68µs p95=89.218µs max=1.850415ms
```

## 2026-10-02 -- 1.1.0 against the 2.0 baseline

Same setup as the baseline above (1440p, mods off, model every 2nd frame, scale 1.0), one run:
**61.6 real fps, 61.9 displayed, GPU 90%, 196 W** against 61.6 / 89% on 1.0.1, so the always-on
GPU timestamps cost nothing measurable. `[sync]` medians: total 18.5 ms, capture_gpu 5.75,
wait_answer 12.65, helper 11.5, zc=true. The new timestamps: **gpu_capture 0.79 ms,
gpu_compose 1.85 ms** of the layer's own GPU work per capture and per compose.

`capture_hot_path_cost_per_present` on LordNikon at 1.1.0. Its fake helper now keeps a heartbeat,
so every present composites (229 fresh answers in 200 samples; at 1.0.1 only 31 did and the CPU
mean of 54 µs measured mostly presents that returned early, so the two are not comparable). It
runs the copy path (zc=false):

```
[sync] 2560x1440: total=5.5ms capture_gpu=1.3ms copy_out=2.0ms ... zc=false gpu_capture=1.08ms gpu_compose=1.56ms
cpu (run() on present thread): mean=5.586228ms p50=5.497712ms p95=6.248682ms max=7.70366ms
gpu (queue drain after run()): mean=1.713983ms p50=1.673638ms p95=1.849431ms max=1.883138ms
```

## 2026-10-02 -- 2.0.0

The model runs before DLSS Super Resolution by default (`docs/PRE_UPSCALER_DESIGN.md` has every
experiment). GTA V Enhanced, LordNikon, RTX 5070, driver 615.71.09, 2560x1440 at 288 Hz, HDR desktop,
DLSS Balanced (render 1485x836), `scripts/gta-bench.sh`, script mods off, pass 4:

| Build / config | Real fps | Shown fps | GPU |
|---|---|---|---|
| 1.0.1 baseline, model every 2nd frame (above) | 61.6 | 62.0 | 89% |
| 2.0 candidate e801204, default (model before the upscaler, every frame), 3 runs | 64.6 / 64.2 / 65.0, mean **64.6** | 65.3 / 64.4 / 65.6 | 89-92% |
| same build, `NEURAL_FORGE_PREUPSCALE=off` | 62.2 | 62.6 | 90% |
| 4K, 2.0 default vs 1.x path | 39.0 vs 28.7 | | 96% vs 93% |
| DLSS Frame Generation 3x, 2.0 default vs 1.x path | 53.0 vs 28.7 | 159 vs 86 | 96% vs 98% |

**Real play, 2.0 (137f24e, same code as 2.0.0), Alex at the screen, about an hour, mods on, DLSS
Frame Generation 4x.** A 60 s sample: ~50 real frames per second, every one held and run through the
model (48-50 holds/s), 195-199 fps shown, hold 9.7 ms median (capture wait 3.6, helper 5.9, hand-off
0.00), 0 misses in the sample and 28 over 29 minutes (loading screens and start-up), DLSS FG's
launch submits forwarded untouched (228,000 vs 88,655 held), GPU 96% (never below 94%), 9.5 of 12.2
GB VRAM, 205 W, 68 C, no Xid. Alex: "everything is working beautifully ... nothing I can complain
about", mods working.

**GTA's start-up crash.** Every early exit today is an access violation at
`GTA5_Enhanced.exe+0x12c6eb` during "Game Init" (32 dumps on this machine, 11 of them on 2026-10-01
with 1.0.1; other Game Init crashes go back to January, before Neural Forge). It is the game's own;
relaunching works.

## 2026-10-02 -- upstream 0.3.1-2 vs Neural Forge 2.0.0

DLSS5VKLayer 0.3.1-2 (release 2026-09-26, commit `9f43793`), installed on LordNikon from the
release tarball with `./install.sh --user` (no root): `~/.local/lib/dlssnr`, `~/.local/bin/dlssnr-*`,
implicit manifests `~/.local/share/vulkan/implicit_layer.d/VK_LAYER_NV_dlssnr.{x86_64,i686}.json`
(`enable_environment` `VKLayer_DLSS5=1`), a `dlssnr.desktop` launcher (not autostart). NGX DLLs
copied (not moved) from Neural Forge's binaries folder. The installer also writes
`~/.config/environment.d/dlssnr.conf` with `VK_INSTANCE_LAYERS=VK_LAYER_NV_dlssnr:VK_LAYER_NV_present`;
it was trashed right after the install, before any reload or login, so the session never saw it.
Checked inert: no `VK_*` in `systemctl --user show-environment`, nothing in autostart or systemd user
units, Steam launch options unchanged, and `VK_LOADER_DEBUG=layer vulkaninfo --summary` lists the
layer but never loads `libVkLayer_NV_dlssnr.so` unless `VKLayer_DLSS5=1` is set.

Upstream ran at its defaults: model every frame, after the upscaler, at 2560x1440, motion vectors on.
Only one model helper ran at a time.

**GTA V Enhanced: upstream still cannot run it.** `scripts/gta-bench.sh`, layers
`VK_LAYER_NV_dlssnr`, `VKLayer_DLSS5=1`, mods off, Alex's settings (DLSS Balanced, FrameGenType 1):

- Helper running from the start: 3 of 3 launches crashed at Game Init, about 95 s in, with the known
  access violation at `GTA5_Enhanced.exe+0x12c6eb`. A fourth with `DLSSNR_IDLE_REPAINT=0` did the
  same. The layer loads into the Rockstar Launcher as well as the game. The helper rebuilt its NGX
  feature on every frame, alternating between the launcher (`neural ready 1022x598`) and the game
  (`neural ready 2560x1440`). Each game frame was answered with `[shm] helper could not use frame N`
  (the answer was made for the other size), and each frame's present waited for a rebuild of about
  150 ms.
- Helper started 130 s after launch, once Game Init was over: the game ran, but at **2.7 fps
  displayed** (GPU 11%), against 377 fps on the loading screen just before. The helper logged 2,699
  rebuilds of the feature, alternating between the two sizes. It is the 0.3.0/0.3.1 failure, still
  there.
- Layer loaded with its helper stopped (the layer passes frames through): the benchmark completed,
  92.9 real / 94.3 shown, GPU 67%. That matches NR off, so the layer alone neither crashes the game
  nor costs anything. Two launches before this one stopped in the Rockstar Launcher ("Failed to
  connect to the Rockstar Games Library Service", then error 7002.1 "Launch validation had fatal
  error"). An NR-off launch straight after worked, as did the next layer-only launch. Cause unknown.
- For comparison, Neural Forge hit the same Game Init crash once in 4 launches, and NR off 0 in 2.

Neural Forge 2.0 and NR off in the same GTA settings (pass 4). DLSS Frame Generation did not always
engage. When it did, it gave 2x (`dlssFrameGenMode` 0), not 4x:

| Run | Real fps | Shown fps | GPU | FG engaged |
|---|---|---|---|---|
| Neural Forge 2.0 #1 | 68.2 | 68.7 | 94% | no |
| Neural Forge 2.0 #2 | 68.0 | 68.5 | 94% | no |
| Neural Forge 2.0 #3 | 56.9 | 114.2 | 96% | 2x |
| NR off #1 | 92.3 | 92.8 | 67% | no |
| NR off #2 | 89.1 | 180.3 | 87% | 2x |
| Upstream 0.3.1-2, helper from start (x4) | crashed at Game Init | | | |
| Upstream 0.3.1-2, helper started after Game Init | ~2.7 shown | | 11% | |

Neural Forge held 68.5 frames/s before the upscaler with FG off, and 57.1/s with FG on. The FG-off
pair was skipped because upstream has no GTA number to pair it with.

**Cyberpunk 2077: matched comparison.** It runs unattended under Proton:
`Cyberpunk2077.exe --launcher-skip -skipStartScreen -benchmark`, through the same SLR_4 entry point
as GTA with Proton-GE Latest. The game skips the menu, runs the 64 s benchmark scene and exits.
Results go to `Documents/CD Projekt Red/Cyberpunk 2077/benchmarkResults/<time>/summary.json`.
The runner and summariser are on the rig as `~/nf-spike/cp/cp-bench.sh` and `cp-report.py`.
Settings were Alex's, unchanged: 2560x1440 fullscreen, DLSS Auto, ray tracing on (reflections, sun
shadows, lighting Ultra), FG off, Reflex on.

In Cyberpunk, Neural Forge's pre-upscaler path finds no DLSS input
(`[preupscale] no DLSS input among 11 registered views`). It falls back to the after-upscaler path
at 2560x1440, the same place upstream works, so this is a like-for-like comparison.

| Config (benchmark scene, 64 s) | Real = shown fps | GPU | Power |
|---|---|---|---|
| Upstream 0.3.1-2 (model every frame), 3 runs | 38.0 / 37.9 / 37.9, mean **37.9** | 89-90% | 218 W |
| Neural Forge 2.0, model every frame (`model_interval=1`), 2 runs | 38.2 / 38.4, mean **38.3** | 95% | 219 W |
| Neural Forge 2.0, Alex's setting (`model_interval=2`), 3 runs | 51.9 / 51.8 / 52.1, mean **51.9** | 95-96% | 214 W |
| NR off, 2 runs | 93.0 / 85.3, mean 89.2 | 95% | 206-210 W |

At the same work per frame the two are equal (Neural Forge +0.4 fps). Neural Forge's default of
running the model every 2nd frame is 37% faster. Neural Forge's `[sync]` in Cyberpunk: helper 11.4 ms,
capture_gpu 11-24 ms (waiting for the game's frame), zc=true.

Left on the rig: upstream installed but inert, with its helper stopped and nothing global set.
The release tarball is kept in `~/Downloads`, and its `uninstall.sh --user` removes the install.
Neural Forge's helper is running with its settings as found (enabled=1, working_scale=1,
model_interval=2, mvec_enabled=1). GTA settings.xml is byte-identical (sha256 `6c687fff...addc`).
The desktop is untouched at 2560x1440@288, scale 1.0.

## 2026-10-05 -- 2.0.2

**Correction: 2.0.2 fails with DLSS Frame Generation on, which is how GTA is played.** The runs
below had `FrameGenType` 0. With Alex's settings as they are (`FrameGenType` 1, `dlssFrameGenMode`
2), 3 runs: **22.7 real fps, 90.5 shown** (frame generation 4x), no frame held. Every run logs
`the DLSS launch buffer has a same-layout barrier on a DLSS input before its launch; not holding`,
and the model ran after the upscaler on every shown frame (91 composited/s). Fixed in 2.0.3 (below).

2.0.2 (0720dfd, deployed with `scripts/deploy-rig.sh`) against 2.0.1's code (d79c82a, the build
installed on 2026-10-03; 2.0.1 only bumped the version after it), same day. LordNikon, RTX 5070,
driver 615.71.09, 2560x1440 at 288 Hz, HDR desktop, DLSS Balanced (render 1485x836), GTA settings.xml
as Alex has it except `FrameGenType` 0 for the runs (restored afterwards, sha256 identical),
`scripts/gta-bench.sh`, script mods off, no remote-desktop session, pass 4. Settings as found:
enabled=1, passes=1, working_scale=1, mvec_enabled=1.

| Config | 2.0.1 (1 run) | 2.0.2 (3 runs) | 2.0.2 mean | GPU |
|---|---|---|---|---|
| NR off (layer not loaded) | 92.4 | 92.1 / 92.2 / 91.7 | **92.0** | 67% |
| NR on, default (model before the upscaler) | 65.2 | 65.1 / 65.1 / 64.5 | **64.9** | 89% |
| Post-upscaler path (`NEURAL_FORGE_PREUPSCALE=off`) | 62.2 (2.0.0) | 62.1 / 63.0 / 62.1 | **62.4** | 90% |
| Layer loaded, doing nothing (`enabled=0`, `PREUPSCALE=off`) | - | 91.6 / 91.5 / 92.0 | **91.7** | 66% |

All within run-to-run noise. The layer issuing `vkQueueSubmit`/`vkQueueSubmit2` itself costs
nothing measurable: loaded and idle it gives 91.7 against 92.0 without it.

From every NR-on `launch.log` (2.0.2 runs 1-3, 2.0.1 run 1):

- Holds every DLSS frame: 65.5 / 65.7 / 66.5 held before the upscaler per second in pass 4 against
  65.1 / 65.1 / 64.5 real fps (2.0.1: 66.0 against 65.2). Hold 10.3-10.5 ms median (2.0.1:
  9.8-10.3), capture_gpu 0.62-0.63 ms, writeback 0.61-0.62 ms.
- No `not holding` line in any run, NR on or not: none of the new refusal reasons fires on GTA V.
- `[shm] attached /tmp/neural-forge-1000/shm.bin` once per run, no `[shm] refusing`.
- No `fence wait timed out`, no Vulkan error or validation line.
- One `frame went to DLSS untouched: the answer was over budget` and one `breaker open` / `breaker
  closed` at the first hold of each run (the model's first build); 2.0.1 logs the same lines at
  the same point.
- `capture_wait` session max: 13.42 / 6.16 / 14.08 ms (largest **14.08 ms**, against the 5000 ms
  bound). Benchmark runs only: loading screens are covered, alt-tab and resolution changes are
  not, so the bound is not changed yet.

## 2026-10-05 -- 2.0.3

**Why 2.0.2 refused GTA with frame generation on** (probe `v201-fg-probe-1`, NGX probe on 2.0.1,
`FrameGenType` 1): SR's launch buffer starts with a dispatch, then copies depth and motion vectors
for frame generation, each between `GENERAL -> GENERAL` barriers (`SHADER_READ -> TRANSFER_READ`
and back on the game's motion vectors, `TRANSFER_WRITE -> SHADER_READ` on the copies), then SR's
input launch. DLSS's depth and motion vectors are those copies. Nothing in the buffer touches the
colour or exposure input. A hold reads only those two (depth and motion vectors only in `dump`
mode), so 2.0.3 refuses a hold only for synchronization on them; a barrier on depth or motion
vectors stops only a dump.

2.0.3, Alex's settings as found (`FrameGenType` 1, `dlssFrameGenMode` 2), mods off, pass 4, 3 runs:

| Run | Real fps | Shown fps | Held/s | GPU |
|---|---|---|---|---|
| 2.0.3 #1 | 68.2 | 68.6 | 68.8 | 94% |
| 2.0.3 #2 | 68.1 | 68.3 | 68.7 | 94% |
| 2.0.3 #3 | 68.3 | 68.6 | 68.9 | 94% |
| 2.0.2 (3 runs, mean) | 22.7 | 90.5 | 0 | 97% |
| 2.0.1 (2 runs) | 68.2 / 67.7 | 68.5 / 68.0 | 67.8 / 68.4 | 94% |
| NR off, no layer | 91.3 | 92.5 | - | 66% |

- Every DLSS frame held; no `not holding` line; `[shm] attached` once, no `[shm] refusing`; no
  fence timeout, no Vulkan error. One over-budget miss and one breaker open/close at the first
  hold, as on 2.0.1 and 2.0.2. `capture_wait` session max 8.09 / 6.84 / 6.63 ms.
- **Frame generation was not shown in these benchmark launches**, with or without Neural Forge:
  the layer's own `[present]` count equals the real frame rate (about 60/s), so no generated frame
  was presented. NR off with no layer loaded showed none either, at `dlssFrameGenMode` 2 and at 0
  (91.8 real, 92.8 shown; 2.0.3 at mode 0: 67.8 real, 68.5 shown, every frame held). The only
  launches today that showed generated frames were 2.0.2's three (4x at 22.7 real). GTA's frame
  generation did not engage reliably in benchmark launches on 2026-10-02 either. Holds with
  generated frames shown are on record for the same hold on 2.0 (real play at 4x, about 50 real /
  195 shown; benchmark at 2x, 56.9 real / 114.2 shown).

**After a restart of LordNikon, frame generation engaged with 2.0.3.** Same settings as found, mods
off, no remote-desktop session, 4 launches:

| Run | Real fps | Shown fps | Held/s | GPU | Frame generation |
|---|---|---|---|---|---|
| 2.0.3 #1 | 68.2 | 68.7 | 68.9 | 94% | not shown |
| 2.0.3 #2 | 68.3 | 68.8 | 68.8 | 94% | not shown |
| 2.0.3 #3 | **49.5** | **198.1** | **49.5** | 96% | **4x** |
| NR off, no layer | 93.6 | 94.9 | - | 68% | not shown |

Run 3 holds every real frame with frame generation at 4x, matching 2.0's hour of real play (about
50 real / 195 shown). No `not holding`, no `[shm] refusing`, no fence timeout, no Vulkan error;
`capture_wait` session max 6.33 / 6.49 / 6.39 ms. Frame generation's submits went through untouched
(21,000 forwarded against 7,003 held).

## 2026-10-05 -- why GTA's frame generation engages in some benchmark launches only

Twelve launches with frame generation on in the settings (4x), 2.0.3 and NR off. GTA decides at a
loading screen whether to run DLSS Frame Generation at all: in launches where it does not show,
frame generation's CUDA submits stop (the layer's `launch-bearing submits` count stays at 3000, or
never reaches it), so no generated frame is made. Either it never starts, or it runs through passes
1-2 and stops at the pass 2 -> 3 load. Measured and ruled out:

- Neural Forge: NR off, no layer loaded, shows the same.
- Window focus: GTA held focus for a whole run that lost frame generation (`xprop` every 2 s).
- VRAM: peak 10.0 of 12.2 GB in two runs that lost it.
- Dynamic multi frame generation: not configured (no DRS override; `dlssFrameGenMode` 2 is a fixed
  4x).
- Reflex: `DXVK_NVAPI_LOG_LEVEL=info` shows the same `SetSleepMode` sequence (disabled on each
  loading screen, `Enabled/2000us` after) in engaged and non-engaged runs.

The cause inside GTA is not known. `scripts/bench-report.py --fg` now names a run where frame
generation did not engage and leaves it out, so a frame-generation number is always an engaged one.
With logging on, 2 of 3 engaged: 49.5 / 49.6 real, 198.5 / 198.6 shown, every frame held.

## 2026-10-05 -- 2.0.4: the hold inside DLSS's command buffer

**Crimson Desert renders its frame in DLSS's command buffer** (NGX probe `cd/probe-2`, 2.0.3): before
DLSS's first launch (`cuda_dldn_engine_luma_convert_kernel`) the buffer records 7 draws, 27 dispatches,
18 renderings and 23 barriers with a write in their source access. 2.0.2's rule refuses to split it
(`global memory barrier with a write in its source access before its launch`), so on 2.0.2/2.0.3 the
model never ran before the upscaler in Crimson Desert. Cyberpunk 2077's buffer has 2 dispatches and 6
write barriers before its first launch and is refused the same way.

**Getting the side queue right.** The hold's work on the layer's own queue while DLSS's buffer waits:

| Side queue | Result in Crimson Desert |
|---|---|
| Family 0 (the game's graphics family), zero-copy SHM import | no capture finished within 5 s; Xid 69 (class error, compute class `cec0`, offset `1b0c`) |
| Family 0, staged buffers | the same, plus 2x Xid 32 |
| Hold recorded, released at once, no work on the side queue (`NEURAL_FORGE_INLINE=release`) | no Xid: the recorded copies, events and wait are fine |
| Family 2 (compute only), staged, exposure copied in the buffer | **every frame held, 0 misses, no Xid, no fence timeout** |

A native probe (`examples/queue_wait_probe.rs`, RTX 5070) shows a second queue of family 0 and a
family-2 queue both finish work in 0.1-0.2 ms while queue (0,0) waits on a host event; the fault
needs the game's buffer (CUDA launches in flight) on the parked queue. The two-queue GPU test
(`a_hold_inside_the_buffer_runs_on_the_side_queue_and_writes_back_through_the_staging_image`) passes on
the RTX 5070 with staged buffers (worst relative error 2.71e-3). With host-memory imports in that
test process, a later 32 KB device-local allocation fails with `ERROR_OUT_OF_DEVICE_MEMORY` (also the
existing roundtrip test on that machine); the test can run staged with `NEURAL_FORGE_TEST_IMPORT=0`.

**Real play, Alex at the screen** (Steam launch, `NEURAL_FORGE_ENABLE=1 %command%`, 2560x1440, DLSS
Balanced 1516x852, frame generation with dynamic multi frame generation, Reflex on, SDR): holding,
24.7-31.3 frames held per second (every real frame), **148-164 fps shown**, hold 8.8-9.0 ms median
(capture wait 0.83-0.90, helper 5.6), 0 misses, no Xid. Ray Reconstruction switched on in game changed
nothing in the layer's view (same 1516x852 input held, 24.7/s, 148 shown); whether the game's Ray
Reconstruction then runs is not measured. Alex: "for the first time I'm actually noticing the neural
rendering being applied ... no real flickering ... with frame gen it's playable".

**GTA V and Cyberpunk 2077 on 2.0.4** (one run each, settings as found):

- GTA V Enhanced (frame generation on in its settings; it did not engage in pass 4 of this benchmark
  launch, as often): the split hold as before, 68.8 held/s against 68.1 real, 0 misses after the
  model's first build, no `not holding`, no hold inside the buffer, no fence timeout, no Xid.
- Cyberpunk 2077 (SDR, frame generation off as Alex has it, ray tracing on): held inside DLSS's buffer,
  54.1 held/s, hold 10.7 ms, 0 misses after the first build, no Xid. After about 11 s DLSS's launches
  named another of its two 1485x835 RGBA16F candidates (`a CUDA launch names depth ... with several
  colour candidates at its extent`); its buffers were then classed as not reading the colour input and
  forwarded, and 30 s later the post-upscaler compose took over again (46-55/s), as on 2.0.3. Not yet
  held for a whole session: the switch between the two candidates is the next fix.

## 2026-10-05 -- 2.0.5: Cyberpunk 2077 with frame generation

Frame generation turned on in Cyberpunk's settings (DLSS FG 3x; it had been off, which is not how
Alex plays). With it on, the NGX probe (`cp/cp-fgprobe-1`) shows Streamline copying DLSS's inputs in
DLSS's buffer before the launches: the colour into a fresh 1485x835 RGBA16F image and the depth into a
1485x835 `R32_SFLOAT` image; no depth-format image is registered. DLSS's input kernel names that R32
image and two 1485x835 RGBA16F images. `dump` through the hold inside the buffer with
`NEURAL_FORGE_PREUPSCALE_PICK=0` and `=1`: the first in parameter order is the rendered frame (a bar,
pool table), the second a near-uniform dark blue buffer.

2.0.5 (R32 depth accepted, first colour candidate in parameter order), benchmark `v204-cpfg-3`,
frame generation 3x, SDR: held inside DLSS's buffer for the whole run, 2,400 holds (about 39/s),
0 misses after the model's first build, 116.3 fps shown on average (107 minimum per Cyberpunk's own
summary: 118.0 average, 106.7 minimum), post-upscaler compose off (0.8/s), no fence timeout, no Xid.
2.0.4 in the same settings: never identified, 75 fps shown, the model after the upscaler.

## 2026-10-05 -- 2.0.6: Unreal Engine 5 (Black Myth: Wukong benchmark tool)

Steam launch, `NEURAL_FORGE_ENABLE=1 %command%`, Proton-GE Latest, the tool's own settings (DLSS,
super resolution 60, full ray tracing Medium, Very High), driven by a virtual Xbox controller
(`~/nf-spike/pad/nf-pad.py` on LordNikon: the tool ignores a synthetic mouse).

- Why nothing was held before: DLSS's input kernel (`cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi`)
  packs two 32-bit view handles per 8-byte parameter word (`0x3201c2301401c00`), so the layer
  matched only the one handle alone in its word (the output). Its colour input is a sampled
  1488x836 `B10G11R11_UFLOAT` image in `SHADER_READ_ONLY_OPTIMAL` in some scenes, an RGBA16F storage
  image in others; depth D32_SFLOAT_S8_UINT; the frame is rendered in DLSS's buffer before the
  launch.
- 2.0.6: held inside DLSS's buffer through the whole benchmark (over 7,800 holds per run, 19-90/s by
  scene), hold 8.4 ms, 0 misses after the model's first build, no Xid. The 1x1 DLSS's buffer names
  read 0.22 to 59,456 (DLSS's auto-exposure); over 1,000 the hold measures the exposure from the
  frame. Frames with the model on look natural against frames with it off (same scenes, different
  moments: the run with it off moves faster). The tool's counter: 36-42 fps with the model, 49-72
  without.
- Frame generation: the tool refuses it under Proton (hardware-accelerated GPU scheduling), with
  Proton Experimental and Proton-GE, `WINE_DISABLE_HARDWARE_SCHEDULING=0` and
  `WINE_ENABLE_HARDWARE_SCHEDULING=1` alike. Not Neural Forge's.

Regression on the final 2.0.6 build, one run each, with the settings as found (frame generation on
wherever the game has it):

- GTA V Enhanced (benchmark, `v206-gta-1`): split hold 69.0 held/s against 68.0 real fps, 0 misses,
  nothing inside DLSS's buffer (not needed), no fence timeout, no Xid.
- Cyberpunk 2077 (benchmark, FG 3x, `v206-cp-1`): held inside DLSS's buffer, about 2,700 holds at
  38.4/s, 116.5 fps shown on average (106.6 minimum), 0 misses, no Xid.
- Black Myth: Wukong benchmark tool: over 7,800 holds, 0 misses, no Xid.
- Crimson Desert (`v206-cd-2`): the title screen held (5,700 holds, 44/s, 0 misses); in game (Nas
  River) nothing was held and the model ran after the upscaler (75.6 fps composited, all presents),
  no fence timeout, no Xid. First put down to Ray Reconstruction (on since 17:18 that day): wrong.
  Alex's play with 2.0.5 and Ray Reconstruction on held every real frame (3,154 buffers held reading
  the colour input, 7 naming another candidate). In 2.0.6's run two buffers named different colour
  images, the size rule took the lowest handle, and in play DLSS's input kernel read a candidate
  past the three kept beside it (ten registered), so the switch to it never happened. Fixed in
  2.0.7.

## 2026-10-05 -- 2.0.7: Crimson Desert held in play again

One run per game on the final build, settings as found (frame generation on wherever the game has
it; Crimson Desert with Ray Reconstruction on):

- Crimson Desert (`v207-cd-3`, in game at Nas River, checked on the captured frame): the colour
  input switched once to the candidate DLSS's input kernel reads, 7,800 holds at 23/s, 141 fps
  shown, nothing composited after the upscaler, 0 misses, no fence timeout, no Xid. (`v207-cd-1`
  ended on the shader-compile screen: not counted.)
- GTA V Enhanced (`v207-gta-2`): 68.8 held/s against 68.0 real, 0 misses, no fence timeout.
- Cyberpunk 2077 (`v207-cp-2`, FG 3x): 115.3 fps shown on average (106.1 minimum), held inside
  DLSS's buffer at 38/s, 0 misses, no fence timeout.
- Black Myth: Wukong benchmark tool: an intermediate build that also widened where the hold goes
  inside DLSS's buffer to every candidate stopped holding 45 s into the benchmark (`v207-wk`); the
  final build (`v207-wk-2`) and the released 2.0.6 layer swapped in for an A/B (`v206-wk-ab`) both
  reached 6,600 holds at the same point, with the same uneven stretches during the benchmark.
- No Xid in any run.

## 2026-10-06 -- branch `wukong-fg-wip`: Black Myth: Wukong with DLSS Frame Generation

Frame generation in the benchmark tool unlocked under Proton with `HwSchMode`=2 (DWORD) under
`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` in its prefix (ProtonDB). With it on, the
tool destroys and re-creates the views it hands DLSS every frame (about 65 a second), and in the
benchmark scene hands DLSS a new colour and depth image every frame; its frame-generation buffers
(`main_kernel`) name the colour candidates too. 2.0.7 held nothing there (identification changed
every frame, the in-buffer staging slots were all taken by frame generation's buffers) and the
post-upscaler compose stayed off: no effect at all.

The branch: an image stays registered until it is destroyed (not with its last view); a new depth
image, or a short gap without one, is not a new identification; staging slots are freed on
`vkResetCommandPool`/`vkDestroyCommandPool`/`vkResetCommandBuffer`; the in-buffer hold never goes
before a frame-generation kernel's launch (by name); a buffer whose DLSS input-kernel launch names a
new colour image at the render extent is held with it; an unchanged identification is not logged
again (it was 97% of the log).

One run each on the final branch build, settings as found:

- Wukong benchmark tool, FG on, full run: the tool's results 69 fps average, 80 maximum, 29 minimum,
  62 low 5th; 12,000 holds at 48/s, 0 misses (FG on without the effect applied: 98 / 117 / 20 / 84).
  Log 214 lines per run (was 8,104).
- GTA V Enhanced (`wip-gta-2`, `wip-gta-4`): 68.4 and 68.7 held/s against 68.0 real, 0 misses
  (`wip-gta-3`'s passes 2-3 at 51 fps did not repeat).
- Cyberpunk 2077 (`wip-cp-3`, `wip-cp-4`, FG 3x): 115.7 and 117.0 fps shown on average, 38.7 held/s,
  0 misses.
- Crimson Desert (`wip-cd-2`, `wip-cd-3`, Nas River, Ray Reconstruction on): 146 and 141 fps shown,
  24 held/s, nothing composited after the upscaler, 0 misses.
- No fence timeout, no Xid in any run.

GTA San Andreas - The Definitive Edition (Unreal Engine 4's DLSS plugin, Super Resolution only, no
frame generation), checked on the branch: the effect is applied after the upscaler (60 fps,
every frame composited). DLSS's input kernel (`cuda_engine_input_kernel`, after
`cuda_luma_convert_kernel`) names a 1488x836 RGBA16F colour input that is sampled, not storage,
with a D32S8 depth at that extent and motion vectors at the output size. Taking a sampled RGBA16F
as the colour input (when no storage one is named) identified it, but nothing was held: the split
is refused (a write barrier before the launch) and the in-buffer hold has no layout for the colour
input (none of its transitions were seen as barriers; likely render-pass layout transitions). Not
kept on the branch: holding it needs render-pass layout tracking first.

## 2026-10-06 -- branch: every installed DLSS game held before the upscaler

GTA San Andreas DE is held now: its colour input is a sampled RGBA16F image (taken when a launch
names no storage colour candidate), and its layout comes from `vkCmdBeginRendering`'s attachments
(no barrier ever transitions it once identified: "0 barriers on it seen since identification").
Resident Evil Requiem (ray tracing High, FG 4x) is held through DLSS Super Resolution's input kernel
(`hiluma_engine_input_depthinv_mvlo_hdr_v2_rel`) with its B10G11R11 colour input.

The game's frame-rate cap: San Andreas DE resets it to 60 at every launch whatever its
GameUserSettings.ini say (both copies at `FrameRatePC=0`, also read-only, also with `FrameRate=0`:
the menu still shows 60). `scripts/game-reg.sh` sets Frame Rate to Unlocked in the menu on every run.

One run each on the final branch build (settings as found, frame generation on where the game has
it, no frame-rate cap):

- GTA San Andreas DE (`br-sa-3`, unlocked): 79 holds/s = its frame rate, 0 misses (capped at 60:
  60 holds/s).
- Resident Evil Requiem (`br-re-1`): 159.7 fps shown, 40.8 holds/s, 0 misses.
- GTA V Enhanced (`br-gta-2`, FG engaged at 4x): 49.7 real / 198.9 shown, 49.6 held/s, 0 misses.
- Cyberpunk 2077 (`br-cp-1`, FG 3x): 116.9 fps shown on average (106.2 minimum), 38.2 held/s.
- Crimson Desert (`br-cd-2`, Nas River): 139 fps shown, 23.5 held/s (`br-cd-1` ended on the
  shader-compile screen a new layer build brings: not counted).
- Black Myth: Wukong benchmark (`br-wk-1`, FG on): 34.7 held/s, 0 misses.
- No fence timeout, no Xid in any run.

Red Dead Redemption 2: not run. Its Vulkan renderer exits at startup on this driver with or without
the layer (known Proton issue); switched to DX12 (`system.xml.nf-bak` keeps the original), after
which the Rockstar Games Launcher did not start the game unattended (a dialog the unattended run
cannot see is the likely cause; not confirmed).
## 2026-10-06 -- Shadow Warrior 3: Definitive Edition (Unreal Engine 4), branch build

First launch (Proton Experimental): the game's own GPU benchmark failed under Proton and chose the
Low preset; NVIDIA DLSS was greyed out in Settings > Video until FidelityFX CAS was set to Off
(`PROTON_ENABLE_NVAPI=1` alone did not change it; not needed once CAS is off). Set: DLSS Balanced,
CAS Off, Frame Rate Limit No, V-Sync Off, Overall Quality High.

- In play (opening level): 83.9 holds/s = its frame rate, 0 misses, the game's exposure input
  (3.03); its menus pass a value over 1000 there, measured from the frame instead. Nothing composited
  after the upscaler. The opening cutscene runs without DLSS ("no DLSS submit"): the model runs after
  the upscaler there, as designed.
- Steam launch with only `NEURAL_FORGE_ENABLE=1 %command%`: held (97.4/s, main menu), 0 misses.
- No fence timeout, no Xid.

## 2026-10-06 -- The Witcher 3: Wild Hunt (next-gen, DX12): not rendering under Proton here

Steam launch `NEURAL_FORGE_ENABLE=1 %command% --launcher-skip` (after the EULA dialog): `witcher3.exe`
from `bin/x64_dx12` starts, opens a 2560x1440 window ("The Witcher 3", `xwininfo`), stays at
240-290% CPU and 3.8 GB, and never creates a swapchain (12 minutes). Direct launches: the same with
Proton Experimental (11.0-100, the prefix's) without Neural Forge's layer (no MangoHud frame log at
all), and with Proton-GE Latest. Not a Neural Forge problem; not pursued further.

## 2026-10-06 -- Metro Exodus Enhanced Edition (4A Engine, DX12), branch build

First launch (Proton Experimental, after the EULA): about 10 minutes of shader compilation after the
intro video, then the 3D main menu held before the upscaler: DLSS input 1280x800 (output 1920x1200,
`r_dlss_rx 0` = Quality), 81 fps, 0 misses. Settings found: fullscreen 1920x1200, Ultra, V-Sync off,
no fps limit. The fullscreen resolution list under Proton stops at 2560x1080 (no 2560x1440);
`r_res 2560x1440` in `user.cfg` was reset to 1280x720 by the game (and DLSS did not run there);
windowed 2560x1440 stopped at a 255x114 dialog on start that synthetic input could not answer.
Restored the original `user.cfg`. Not yet measured in play: the intro video runs for minutes on
every launch and does not take controller input.

## 2026-10-06 -- God of War (2018, DX11), branch build

The first DX11 game (DXVK): held before the upscaler with no change. Settings found: borderless
2560x1440, DLSS Quality (render 1708x960), V-Sync off, FPS limit off, Ultra. Menu: 38 holds/s.
In play (`gw-1`, Continue): 49.5 holds/s = its frame rate (no frame generation in this game),
0 misses, the game's exposure input, nothing composited after the upscaler, no fence timeout, no Xid.

Witcher 3 and Metro Exodus EE were removed from the machine and from the README's tested games
(neither ran far enough to test in play); their entries above stay as the record.

## 2026-10-06 -- Marvel's Spider-Man Remastered (DX12), branch build

First launch defaults: V-Sync on, frame generation off, dynamic resolution targeting 60 fps (DLSS
input 2560x1440 then, held at 37/s, 25 ms holds). Set (by Alex): V-Sync off, Reflex on, DLSS Frame
Generation, DLSS Super Resolution Quality. In play (F.E.A.S.T. centre, Continue): DLSS input
1712x960, 49.0 holds/s, 98.0 presents/s (frame generation 2x), 0 misses in the window, nothing
composited after the upscaler; the game's 1x1 exposure reads over 1000, so it is measured from the
frame. No fence timeout, no Xid.
