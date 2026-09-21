# NeuralForge target-machine validation — 2026-09-14

Target: `alex@lordnikon`, RTX 5070, NVIDIA 615.71.09. Upstream package 0.3.0-1
remains installed. At inspection time GTA and the upstream helper were not running.
The saved upstream config retains passes=1, model_resolution=1, motion_enabled=0,
motion_quality=0. No upstream config, package, library, manifest, prefix or Steam
launch option was modified. Upstream config and both layer manifests passed a
before/after SHA-256 comparison.

NeuralForge is installed separately in `~/.local/share/neuralforge`, with its own
config, prefix, helper and `/tmp/neuralforge-1000/shm.bin`. The helper was started
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
only `libneuralforge_layer.so`, not upstream's `libVkLayer_NV_dlssnr.so`.
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
the full model, host SHM, and `NEURALFORGE_DMABUF=0` exited normally in 19.6 seconds.
Khronos validation reported zero errors. The helper reported `model_up=1` and had
processed 192 frames at the time of the status capture. A separate 900-frame run
with synchronization validation enabled also exited normally with zero validation
errors and zero synchronization hazards. Both runs loaded only
`libneuralforge_layer.so`; upstream's NR layer was absent. The upstream config and
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

For the NeuralForge-only launch, Steam was restarted with `NEURALFORGE_ENABLE=1`,
`NEURALFORGE_TARGET_EXE=GTA5_Enhanced.exe`, `NEURALFORGE_DMABUF=0`, its isolated
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
with `model_up=1` and `NEURALFORGE_DMABUF=0` throughout.

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
`GTA5_Enhanced.exe` with `NEURALFORGE_TARGET_EXE=GTA5_Enhanced.exe` acquired the
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
item 4 before continuing. A freshly built `libneuralforge_layer.so` was deployed
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

Also added `scripts/bench.sh` for Phase 1 item 1 (the repeatable native/upstream/
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

Also added `NEURALFORGE_HELPER_DELAY_MS` (test-only) to `neural-forge-helper`: an
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
- `neural-forge-helper.exe` with `NEURALFORGE_HELPER_DELAY_MS=250` set was run alone
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
  (`shmctl status` stays at 0) under `NEURALFORGE_ENABLE=1` regardless of which code is
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
probe to this crate's own `log!`/`logging::flush()` (writing through `NEURALFORGE_LOG`
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

**Real telemetry from that session** (`~/.local/state/neuralforge/helper.log`, both
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
never restarted between tests (it kept the first launch's `NEURALFORGE_ENABLE`
environment -- the `scripts/bench.sh` gotcha this file documents), *and* GTA's saved
Steam launch options bake in `NEURALFORGE_ENABLE=1 ... %command%`, which Steam applies
per-game regardless of the client environment. The fix was `NEURALFORGE_DISABLE=1`
(the layer manifest's own `disable_environment`, honored at the Vulkan-loader level
over any launch option) plus a genuine full Steam restart. Confirmed isolated via
Steam's own `console-linux.txt`: `[dlssnr-layer] ... RTX 5070 (inert=0 enabled=1)`,
zero `[neuralforge-layer]` lines for the whole session.

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
env-gated behind `NEURALFORGE_BENCH`, runs on real hardware via the release test
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
