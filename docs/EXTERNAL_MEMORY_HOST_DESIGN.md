# Zero-copy host transport: device-extension injection

Phase 3's first goal (`ASYNC_CAPTURE_DESIGN.md`'s natural successor) is importing the
SHM proxy/answer regions directly as Vulkan device memory (`VK_EXT_external_memory_host`),
so a capture's `vkCmdCopyImageToBuffer` writes straight into shared memory instead of a
staging buffer the CPU then copies out of. That needs the extension enabled on
whichever `VkDevice` the capture commands submit against -- the game's own device, not
a private one, since the image being copied is the game's own swapchain/render-tap
source.

## The problem this solves

The game creates its own `VkDevice` by calling `vkCreateDevice` with whatever
extensions it wants; this layer only ever observes that call after the device already
exists (`Layer::create_device_info`, called by the pinned `vulkan-layer` framework with
a live `Arc<ash::Device>` already in hand). No ordinary game requests
`VK_EXT_external_memory_host` on its own -- it exists for exactly this kind of
external-interop use case, not normal rendering -- so without some way to add it, this
layer would either need its own private device (which can't touch the game's images
without a second, much larger external-memory *image* sharing scheme) or stay on the
staging-buffer path indefinitely.

## The fix: `InstanceHooks::create_device`

The pinned framework does have a hook for this, just not on the `DeviceHooks` trait
(everything in `device.rs` implements that one, and it only ever sees an
already-created device). It's on `InstanceHooks`, called *before* the framework's own
default `vkCreateDevice` forwarding -- see `vulkan_layer::Global::create_device` in the
pinned crate (`layer_trait/generated.rs`'s `InstanceHooks::create_device`, invoked from
`lib.rs`'s free-standing `create_device` trampoline). This project had never
implemented `InstanceHooks`/`InstanceInfo` before now (`Layer::InstanceInfo` was
`vulkan_layer::StubInstanceInfo`, a real no-op).

`crates/layer/src/lib.rs`'s new `NeuralForgeInstanceHooks::create_device`:

1. Returns `LayerResult::Unhandled` (the exact same as this hook not existing at all)
   whenever there's no safe extension to add: `VK_EXT_external_memory_host` already
   requested, or `vkEnumerateDeviceExtensionProperties` says the physical device
   doesn't actually support it. This is deliberately the overwhelmingly common return
   value -- the hook only ever takes over to add one specific extension, never to
   change anything else about device creation.
2. Otherwise, builds an extended extension list, resolves the real `vkCreateDevice`
   through `layer_device_link.pfnNextGetInstanceProcAddr` (the same resolution the
   framework's own default path uses), and calls it with the extended list.
3. If that creation is refused for any reason the earlier query didn't predict,
   retries with the exact, byte-identical original request before giving up -- this
   optimization must never be the reason a device creation that would otherwise have
   succeeded now fails.
4. On success, records the created `VkDevice` handle in a short-lived side table
   (`EXTERNAL_MEMORY_HOST_DEVICES`) `NeuralForgeDeviceInfo::new` (`device.rs`) checks
   and clears exactly once. This side channel exists because the framework's own
   subsequent `create_device_info` call is handed the *original*, un-injected
   `VkDeviceCreateInfo` regardless of what a hooked `create_device` actually passed to
   the real driver -- there's no other way for `device.rs` to learn what happened.

`State::external_memory_host: bool` carries this into `capture::run`'s own state,
read fresh every present call (cheap: `vkGetPhysicalDeviceProperties2` is a purely
local query) alongside a live check that the *actual* mmap'd proxy-region pointer
(not just its constant offset within the mapping) is aligned to whatever the driver's
own `minImportedHostPointerAlignment` requires. `false` either way just means the
capture pipeline keeps using the staging-buffer path it already has.

## The actual import path: `DirectCapture`

Implemented: `capture::DirectCapture`, a single capture slot (deliberately not two
like `CapturePipeline` -- see its own doc comment) whose device memory is *imported*
from `ShmClient::proxy_region()`'s live pointer via `VkImportMemoryHostPointerInfoEXT`,
so `vkCmdCopyImageToBuffer` writes straight into the SHM proxy region with no staging
buffer. `run`'s shared `poll_or_submit_capture` helper drives whichever of
`DirectCapture`/`CapturePipeline` is active; for the direct path, `ShmClient::write_proxy`
is never called (its one copy is exactly what importing removes) -- though
`original_scratch`/`inflight.original` still need their own one-copy readback out of
the now-written proxy region, since that Vec has to remain stable across whatever
capture starts next and overwrites the shared, imported memory (see `poll_or_submit_capture`'s
own doc comment). Net effect versus the pre-Phase-3 path: two CPU copies (staging ->
Vec -> SHM) become one (SHM -> Vec) for the capture direction.

Two real bugs found and fixed via real-hardware validation (not caught by this
project's local software ICD, which is more permissive than NVIDIA's driver +
validation layers here):
1. `VkBufferCreateInfo` for a buffer that will be bound to imported memory must chain
   `VkExternalMemoryBufferCreateInfo` with the same handle type used at import time
   (`VUID-vkBindBufferMemory-memory-02985`) -- missing entirely in the first version,
   caught immediately by `VK_LAYER_VALIDATE_SYNC=1` on `lordnikon`.
2. The import's `allocationSize` must be a multiple of `minImportedHostPointerAlignment`
   (`VUID-VkMemoryAllocateInfo-allocationSize-01745`) -- a *test* bug (passing the raw
   pixel byte count instead of the alignment-rounded region size), not the production
   code, but only visible once real hardware reported the real alignment (4096 on this
   NVIDIA driver) instead of the local software ICD's more forgiving behavior.

## The helper side: the same import, for upload/download

`crates/helper/src/frame.rs`'s `FrameResources` mirrors `DirectCapture` for the two
directions the helper itself owns: `imported_proxy` (Color's upload source) and
`imported_answer` (Output's download destination), built by `build_imported_buffer`
(the same query/buffer/import sequence as the layer's `build_imported_capture_buffer`,
duplicated rather than shared -- these are two separate crates for two separate
platforms). Unlike the layer side, the helper needed no extension-injection hack at
all: it creates its own `VkDevice` directly (`main.rs::create_vulkan_context`), and
`VK_EXT_external_memory_host` was already in `WANTED_DEVICE_EXTENSIONS` (added in an
earlier session, for reasons unrelated to this feature) and already being requested
whenever the driver advertises it. `crates/helper/src/shm.rs`'s own `open()` had
similarly already been try­ing to land the mapping at a 64 KiB-aligned address for
exactly this eventual use, falling back to an OS-chosen address otherwise -- both
pieces of groundwork just needed connecting to `FrameResources`, not building fresh.

`evaluate()` skips its `proxy -> staging_ptr` copy when `imported_proxy` is set (the
GPU reads Color directly from the live SHM proxy region) and its
`staging_ptr -> answer_out` copy when `imported_answer` is set (the GPU writes Output
directly into the live SHM answer region) -- the same one-copy-instead-of-two
reduction as the layer side, for the opposite direction.

The single-writer safety argument mirrors `DirectCapture`'s: the wire protocol's "one
outstanding request at a time" rule already guarantees nothing else writes to the
proxy region while the helper is reading it, and nothing else writes to the answer
region except this same download copy -- so importing both, even without a
double-buffered protocol, is sound today.

## Two real bugs, found via real-hardware validation

Neither was caught by this project's local software Vulkan ICD, which is more
permissive than NVIDIA's real driver + validation layers on `lordnikon`:

1. `VkBufferCreateInfo` for a buffer that will be bound to imported memory must chain
   `VkExternalMemoryBufferCreateInfo` with the same handle type used at import time
   (`VUID-vkBindBufferMemory-memory-02985`) -- missing entirely in the first version of
   both `build_imported_capture_buffer` (layer) and `build_imported_buffer` (helper).
2. The import's `allocationSize` must be a multiple of `minImportedHostPointerAlignment`
   (`VUID-VkMemoryAllocateInfo-allocationSize-01745`) -- a *test* bug (passing the raw
   pixel byte count instead of the alignment-rounded region size) in the layer-side
   test, not production code, but only visible once real hardware reported the real
   alignment (4096 on this NVIDIA driver) instead of the local software ICD's more
   forgiving behavior.

## A third bug, and a fourth: found only after those two were already fixed and "everything passed"

Both survived a clean `cargo test` and multiple validated `vkcube` runs on `lordnikon`
-- neither is a Vulkan validation-layer finding, which is exactly why they're recorded
separately here as their own lesson, not folded into the list above.

**Real, live undefined behavior in `NeuralForgeInstanceHooks::create_device`**, present
in every run (including every "successful" one) until `scripts/smoke-test.sh` happened
to be run again after the two bugs above: `std::slice::from_raw_parts(create_info.pp_enabled_extension_names,
create_info.enabled_extension_count as usize)` when `enabled_extension_count == 0` --
`vkcube` on this exact machine legitimately leaves `pp_enabled_extension_names` null in
that case (a valid, spec-permitted pattern; Vulkan's own C convention treats
null-plus-zero as "no extensions", same as an empty array), but `slice::from_raw_parts`
requires a non-null, aligned pointer *even for a zero-length slice* -- a real, if
narrow, gap between C and Rust's aliasing/pointer conventions. Debug builds' optional
UB checker caught it as a hard abort; every earlier *release*-mode `vkcube` run on
`lordnikon` this session had the identical UB and simply didn't visibly crash, which is
worse, not better -- undefined behavior having no visible symptom yet is not the same
as it being safe. Fixed by checking `enabled_extension_count == 0` and using `&[]`
before ever dereferencing the pointer, rather than trusting it's non-null because the
length says zero.

**`vk::ExtExternalMemoryHostFn::load(...)` panics if the function it's asked to
resolve doesn't load** -- found live on `lordnikon`: a real `vkcube` run where
`external_memory_host: true` at device creation (the extension genuinely enabled) still
hit `Unable to load get_memory_host_pointer_properties_ext` and aborted the whole
process, inside `build_imported_capture_buffer`. The extension being enabled at device
creation does not guarantee every one of its functions resolves via
`vkGetDeviceProcAddr` in every context -- this project doesn't know the exact reason
(plausibly something about this layer's own loader-dispatch machinery, plausibly a
driver quirk; not investigated further since the fix doesn't depend on knowing), but a
resolution failure has to be a normal, fail-open "don't import" outcome for a module
whose entire design philosophy is exactly that, not an abort. Both the layer's and the
helper's `build_imported_*` functions now resolve `vkGetMemoryHostPointerPropertiesEXT`
by hand via `get_device_proc_addr` (returns `Option`, checked explicitly) instead of
the panicking `::load()` helper. `crates/layer/src/optical_flow.rs`'s own
`NvOpticalFlowFn::load(...)` uses the identical panicking pattern and was not touched --
worth the same fix if `VK_NV_optical_flow` is ever seen behaving the same way live.

**The lesson, not just the fixes**: "passed every test, validated clean on real
hardware multiple times" was true and still missed two bugs that only a *different*
kind of exercise (a debug-mode UB check; a code path validation layers don't cover at
all, since neither VUID nor SYNC-HAZARD checking has anything to say about a Rust
panic) caught. Re-run `scripts/smoke-test.sh` specifically (not just `cargo test` or a
validated `vkcube` run) after touching anything in this file going forward.

## Zero-copy compose: the synchronous present without CPU frame copies

`DirectCapture` still left three full-frame CPU copies on every synchronous present: the
proxy region copied out into `original_scratch`, `ShmClient::read_answer` into
`last_answer`, and both copied again into `GpuCompose`'s staging buffer (about 59 MB per
frame at 1440p). When the synchronous present runs on direct capture with the model at the
frame's own size, frame hold off, an 8-bit format, a `GpuCompose`, and a slot-0 answer
region the driver can import, `capture::run` now skips all three:

- **Capture.** `record_capture_commands` records a second copy of the game image into a
  device-local *capture target* owned by `GpuCompose`, in the same command buffer that
  writes the proxy region. It ends with a `TRANSFER_WRITE -> TRANSFER_READ` buffer barrier.
  That barrier's second scope carries into later submissions on the queue, so the compose
  that reads the target needs nothing of its own. `original_scratch` is left empty. The
  white meter reads the proxy region directly: the capture fence has signaled, the memory is
  host-coherent, and the helper only ever reads it.
- **Answer.** `GpuCompose::ensure_answer_import` imports slot 0's answer region through
  `capture::import_host_buffer`, the query/buffer/import sequence extracted from
  `build_imported_capture_buffer`. Only the frame is imported, rounded up to
  `minImportedHostPointerAlignment`, not the whole 265 MB region. `read_answer` is skipped,
  and `last_answer` is cleared so no stale CPU answer can ever be presented.
- **Compose.** On the present that receives the answer (`ComposeInputs::Gpu { fresh: true }`),
  the compose command buffer copies the capture target into `gen_base` and the imported
  answer into `gen_answer`. Both are device-local and followed by their own buffer barriers.
  Every later compose of that generation reads only `gen_*`: a carried frame at
  `model_interval > 1`, or the other async slot. A late answer, the next capture or
  `run_sync` can overwrite the shared regions without changing the picture. A zero-copy
  compose for a generation `gen_*` do not hold returns `None`; it never reads shm.
- **Helper handoff.** The helper writes the answer region from another process on another
  `VkDevice`, which GPU ordering cannot reach. `GpuCompose` remembers the fence of the
  submission that read the region (`answer_reader`). `capture::run` waits on that fence
  before every `begin_async_request` and before `run_sync`. It is normally already signaled,
  because the next capture was queued behind that compose.
- **Resizes and queues.** The capture target and `gen_*` are rebuilt only when no direct
  capture is pending and both async compose slots have been waited. The import is replaced
  only after both slots have been waited. If a zero-copy compose or capture lands on a
  different queue than the last one, both slots are waited first. Otherwise the present takes
  the CPU path for that frame. Teardown already idles the device before `GpuCompose::destroy`,
  which frees the buffers and the import but never unmaps the region.

Anything that falls outside those conditions takes the CPU path exactly as before: pipelined
mode, working scale, frame hold, a failed import, a capture submitted without the target, or
a pending resize. The `[sync]` log line (every 300 composed frames) reports `zc=true/false`.
It splits the old `capture=` into `capture_gpu=` (submit to completion observed) and
`copy_out=` (0 on zero-copy frames). It reports the white meter as `meter=` instead of
counting it in `wait_answer=`, and adds `helper=`, the helper's published
upload + evaluate + readback, so `wait_answer - helper` is the handoff overhead.

Tests (lavapipe supports the extension, as does the Intel ANV driver on the dev machine):

- `composition::gpu::tests::gpu_inputs_compose_identically_to_cpu_inputs`: the fresh and
  carried zero-copy composes match the CPU inputs byte for byte, even after the imported
  region and the capture target have been overwritten.
- `capture::tests::synchronous_present_zero_copy_matches_cpu_path`: `run` end to end, with
  model interval 1 and 2, compared against the same direct-capture present with the CPU
  copies. The answer and proxy regions are scribbled before every carried frame.
- `capture::tests::frame_hold_uses_the_cpu_path_where_zero_copy_is_available`.

Direct capture imports the whole proxy region, which a driver may back with real pages
(about 250 MB of tmpfs per shm file on Intel ANV). The tests therefore share one shm file
per test and delete it afterwards.

**Not yet validated on hardware.** Before trusting it, run `scripts/smoke-test.sh`, `vkcube`
under `VK_LAYER_KHRONOS_validation` with sync validation, and GTA, where the `[sync]` line
should show `zc=true`.

## Validation

Real hardware, `lordnikon`, RTX 5070, driver 615.71.09, both via `vkcube` and via
dedicated tests copied to and run directly against the real driver -- all of the
following are *after* the two real-hardware VUID fixes and the two crash fixes above,
not before:

- `scripts/smoke-test.sh` (the one that caught the null-pointer UB) and the full
  `cargo test` workspace suite: clean.
- `vkcube` at 1280x720 and 2560x1440 (GTA's real render resolution) under
  `VK_LAYER_KHRONOS_validation:VK_LAYER_neuralforge_neural` with `VK_LAYER_VALIDATE_SYNC=1`:
  `external_memory_host: true`, `pass_through=false` (this layer's own capture *is*
  admitted for `vkcube`'s own swapchain -- earlier sessions' "capture never engages for
  `vkcube`" was about the render tap specifically, GTA's own capture route, not a
  blanket statement; `vkcube`'s surface does expose what plain swapchain capture
  needs), zero validation errors or hazards, no crash across multiple runs including
  one with a real helper attached.
- `capture::tests::direct_capture_writes_straight_into_imported_host_memory` (layer):
  builds a device with the extension actually enabled, fills a source image with a
  known, checkable color, captures it through `DirectCapture` into a real `mmap`'d
  host region, and asserts every captured byte matches the source exactly. Passes
  locally (this dev machine's software ICD also supports the extension) and on
  `lordnikon` under full synchronization validation.
- `scripts/protocol/examples/trigger_helper_roundtrip.rs` (new -- see its own doc
  comment) drove real request/response round trips against a real, running helper on
  `lordnikon` outside of any game: `[frame] 64x64 resources: imported_proxy=true
  imported_answer=true` confirmed the helper-side import succeeds on real Wine +
  Proton-CachyOS + the real NVIDIA driver, no crash across dozens of repeated round
  trips. `EvaluateFeature` itself did not produce a real answer for this synthetic
  garbage-pixel input (the round trip fails open and echoes the proxy through,
  confirmed via `helper_eval_ms` staying `0`) -- checked against the pre-Phase-3
  helper build too, which fails identically on the same synthetic input, so this is a
  pre-existing characteristic of feeding `EvaluateFeature` synthetic data, not
  something this phase's transport changes caused. Real pixel data (a real game frame)
  was not tested this way.

**Not yet validated**: a device where the extension genuinely isn't available (every
driver tested this session has it, so the `Unhandled`/fallback branches are reviewed,
not exercised live); a real answer from `EvaluateFeature` actually using the imported
buffers (blocked on the synthetic-input limitation above, needs a real game frame);
GTA itself, which needs a real session and is the only way to measure whether this
actually moves layer fps toward upstream's ~74/s on `lordnikon`.
