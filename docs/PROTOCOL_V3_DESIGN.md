# Protocol v3: a second independent request/response slot

Phase 3's remaining goal after `EXTERNAL_MEMORY_HOST_DESIGN.md`'s zero-copy host
import: the wire protocol has only ever supported one outstanding request at a time
(a single `seq_req`/`seq_resp` pair). Even with `CapturePipeline`'s two GPU capture
buffers (`ASYNC_CAPTURE_DESIGN.md`), a second, already-captured frame just sits idle
once captured -- the layer cannot send it to the helper until the first request's
answer comes back, because there is only one wire slot to send it on. v3 adds a
second, fully independent slot so the layer is never blocked with a ready frame and
nowhere to send it.

## What stays serialized, and why

The helper's `FrameResources::evaluate` calls NGX's `EvaluateFeature` synchronously
(upload, evaluate, download, each waiting on its own fence before the next starts) on
a single feature handle (`snippet.feature`, created once by `ngx::ensure_feature`).
Nothing in this project's own reverse-engineering of this NVIDIA API establishes that
a single NGX feature handle is safe to evaluate concurrently from two overlapping GPU
submissions -- DLSS-family NGX features are, as far as this project has ever
observed or documented, a strictly one-evaluate-in-flight-per-feature design. Betting
on undocumented concurrent-call safety, on a *reverse-engineered* feature, running
under Wine, is exactly the kind of guess this project's own history
(`docs/history/development-before-neuralforge.md`'s reverted fence-wait "fix",
`EXTERNAL_MEMORY_HOST_DESIGN.md`'s two crashes found only after "everything passed")
says not to make without real evidence.

So v3 does **not** attempt concurrent `EvaluateFeature` calls. The helper still
processes both slots' NGX evaluation one at a time, in whichever order their
`seq_req`/`seq_req_b` bumped. What v3 actually buys:

- The layer can have a second frame's bytes already sent to the helper the moment
  they're captured, instead of holding them in `CapturePipeline`'s GPU buffer (or, for
  `DirectCapture`, not being able to start a second zero-copy capture at all -- see
  below) until the first wire request resolves. That is real, measurable dead time
  removed from the layer's own present-hook cost, independent of whatever the helper's
  own NGX evaluation time is.
- The helper's *upload* for the next request can happen (staging copy, or nothing at
  all for the imported-memory path) while the *previous* request's evaluate/download
  is still the thing occupying the GPU queue -- still ordered, but the CPU side isn't
  idle waiting for a wire slot either.

## What's duplicated for slot 1, and what isn't

New `ShmHeader` fields (appended at the end, like every prior addition — see that
struct's own layout comment): `seq_req_b`, `seq_resp_b`, `width_b`, `height_b`,
`proxy_format_b`. New memory regions: `proxy_b_offset()`, `answer_b_offset()`,
each `MAX_FRAME` bytes, appended after the existing motion region. `SHM_VERSION`
bumped to 3 -- a mismatched-version mapping is rejected and reinitialized by both
sides' existing `is_valid()`/`open()` contract, so this is a clean break, not a
migration; both processes always ship from the same build.

**Not duplicated**, deliberately:

- `format` -- always 1, written once at `init_defaults` and never read anywhere in
  this workspace (confirmed by grep). A genuinely dead field; duplicating dead code
  adds surface area for zero behavioral value.
- `hdr_encode`, `answered_w`, `answered_h` -- declared, defaulted, round-tripped by
  `reset_persisted_settings`, but (also confirmed by grep) never actually read or
  written by any live layer/helper code path today. Same reasoning as `format`.
- `frame_mvec_valid`, `frame_mvec_scale_mode` -- real fields with real read/write
  code, but that code (`ShmClient::prepare_motion`,
  `ShmClient::prepare_motion_resources`) is itself unconditionally disabled (`return`
  as the first line, `#[allow(unreachable_code)]` below it) — motion vectors are off
  in this project's known-good baseline (see `mvec_enabled`'s own doc comment) because
  of a real NVIDIA driver crash during private-device creation, unrelated to this
  feature. Re-enabling motion and wiring a per-slot motion payload is a separate,
  future change; nothing here forecloses it (the fields still exist, singular, and a
  slot-1 motion payload can be added the same way slot 1's proxy/answer were, later).

## `DirectCapture` becomes two slots

`EXTERNAL_MEMORY_HOST_DESIGN.md` explains why `DirectCapture` was deliberately a
*single* slot despite `CapturePipeline` having two: importing the shared proxy region
directly as device memory only has one region to import into under protocol v2. With
two independent proxy regions now, `DirectCapture` gets a second slot, one import per
region, mirroring `CapturePipeline`'s own two-slot shape -- see that file's own doc
comment for the aliasing/write-hazard reasoning this mirrors.

## The helper side: two `FrameResources`, still one evaluate at a time

`neural_forge_helper::main`'s loop builds two `FrameResources` instances instead of
one -- one importing (or staging into) `proxy_region`/`answer_region`, the other
`proxy_b_region`/`answer_b_region`. Each loop iteration checks both `seq_req` and
`seq_req_b` for new work and processes whichever have changed, in the order noticed,
still fully sequentially (see "What stays serialized" above) -- never both at once,
just never idle-waiting on a wire slot that a captured frame could already be
occupying.

## Validation

Real hardware, `lordnikon`, RTX 5070, driver 615.71.09 -- a fresh clone at each
commit, not just this dev machine's own software ICD:

- `cargo test -p neural-forge-layer` (48 tests, including the new
  `the_two_slots_are_fully_independent`): clean, deterministic, ~0.5s, run several
  times in a row with no flakes.
- `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1 cargo
  test capture::`: the same pre-existing cosmetic validation warning both before and
  after this change (the tests' own device never enables `VK_KHR_swapchain`, so
  `PRESENT_SRC_KHR` barrier layouts trip `VUID-VkImageMemoryBarrier-*-parameter` --
  confirmed identical on the pre-v3 commit via a second worktree, so this is not a
  regression). Zero new validation errors or sync hazards.
- `scripts/smoke-test.sh` (the debug-mode UB checker that has caught real bugs this
  project's own validation layers couldn't -- see `EXTERNAL_MEMORY_HOST_DESIGN.md`):
  clean, `external_memory_host: true`, no abort, clean teardown.
- `crates/protocol/examples/trigger_helper_roundtrip.rs`, extended with the slot
  argument this design called for, against a real running helper
  (`neural-forge-cli start`, Proton-CachyOS, the real `nvngx_dlssnr.dll`): both slots
  answered correctly when triggered *concurrently* (two processes launched at once,
  slot 0 and slot 1), completing in well under a second total across six separate
  concurrent runs at two resolutions. `seq_resp`/`seq_resp_b` always resolved
  correctly; the one informational mismatch observed (`seq_ok` disagreeing with the
  requested number on one run) is the expected, harmless consequence of `seq_ok`
  being deliberately shared rather than duplicated (see "What's duplicated" above) --
  not a wire-protocol correctness issue, since nothing anywhere reads `seq_ok` back.
- One real process lesson, not a code bug: the first two "concurrent" test attempts
  looked like a serious cross-slot race (both processes reporting the same sequence
  number, one request going permanently unanswered) until re-checked against a
  freshly `git pull`ed clone on `lordnikon` -- the diagnostic tool's own slot
  argument hadn't been pulled yet, so both invocations were silently racing for slot
  0 alone. Confirmed by comparing `git log` on the remote clone before concluding
  anything about the actual code. Worth remembering next time a real-hardware result
  looks like a race: check the deployed *source*, not just the deployed binary,
  before trusting it as a finding.

GTA fps against this change is not measured as part of this work -- that needs the
user's own live session, same gate as everything else in Phase 3/4.
