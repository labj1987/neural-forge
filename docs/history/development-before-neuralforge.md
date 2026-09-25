> Historical record: pre-NeuralForge names and deployment instructions below are
> archival, not current instructions. Do not remove or modify upstream installations.
> See ../../PHASE1.md for current paths, safety constraints and the benchmark plan.

# 2026-09-12 (later still): v0.1.33/v0.1.34's Vulkan sync changes reverted -- made
# things worse, not better; handed off. Read this section before touching
# `capture.rs`/`composition/gpu.rs` fence-wait code again.

**v0.1.34 was deployed and the user tested it live. Result: worse than before, not
better.** Both GTA games now crash outright on open (not freeze -- an actual crash),
and Crimson Desert still needs a force-close. This is a regression from v0.1.34's own
changes, not a confirmation of anything this session believed it had fixed.

**What this session did in response**: reverted `crates/layer/src/capture.rs` and
`crates/layer/src/composition/gpu.rs` to their exact v0.1.32 state (`git checkout
bf6f773 -- <those two files>`, confirmed byte-identical to v0.1.31 for both --
v0.1.32 only touched `gui/src/ui.rs`). This undoes every fence-wait/UB change from the
two sections below. `crates/protocol/src/mapping.rs`'s permission fix (`set_permissions`
after `create_dir_all`) was kept -- it never touched Vulkan sync, is low-risk, and
fixes a real, separately-confirmed bug (see below). Rebuilt, full test suite green,
smoke test green, deployed to `lordnikon` as v0.1.35. **Not yet re-verified live by
the user as of this handoff** -- the state a next session inherits.

**Read this as a warning, not just a log entry**: this session's own fence-wait
"fixes" (bounded `wait_for_fences` + non-blocking `get_fence_status` entry checks, see
the two sections immediately below) were reasoned through carefully and match this
project's own established "always fail open" philosophy, backed by real measurements
(GTA V Enhanced FPS collapse, a `vkcube` near-total stall) -- and they made real
gameplay *crash* instead of merely running slowly. That gap between "the reasoning
seemed sound" and "the real-world result was worse" is exactly why:
1. **No fix in this area should ship again without Vulkan validation layers actually
   turned on** (`VK_LAYER_KHRONOS_validation`, `VK_INSTANCE_LAYERS` alongside this
   project's own layer, or `vkconfig`) during real testing -- this session never did
   this, for any of today's changes, and validation output would very likely have
   caught a synchronization mistake before it ever reached the user's real games.
2. A non-blocking `get_fence_status` check immediately before `reset_command_buffer`/
   resource-destroy, treating anything other than `Ok(true)` as "still in flight, skip
   this cycle," was this session's core safety argument for why bounding the waits
   was safe. That argument was never verified against the Vulkan spec's actual
   guarantees around `vkGetFenceStatus` racing a fence that signals *between* the
   check and the subsequent `vkResetCommandBuffer`/`vkQueueSubmit`, or against
   whatever this specific NVIDIA 615.71.09 driver actually does with a reused command
   pool/fence under contention -- a real, concrete gap in the reasoning, not
   necessarily *the* bug, but exactly the kind of thing validation layers exist to
   catch and this session skipped checking.
3. The two-stage discovery process itself (permission bug masked whether the
   capture-side fix worked at all; fixing it just exposed a *second* unbounded wait in
   compose) means there is no confidence the compose-side fix was the last one needed
   either, even before considering it made things worse. **Assume nothing about this
   pipeline's real synchronization correctness has been established by this session's
   own work; assume more coverage gaps exist even in code untouched today.**

**Concretely, for the next session**: the real, unresolved regression is still the
~22x FPS collapse (196-274fps neural-rendering-off vs. a steady 9fps on) documented
extensively below, now confirmed real, reproducible, and NOT caused by upstream's
layer conflict (already fixed) or motion vectors (ruled out via live A/B). The two
sections below this one contain the full reasoning trail for *why* `capture_pristine`
and `dispatch_into_image_async` were suspected -- worth reading for context and the
real measurements they're based on -- but their actual code fixes are reverted and
should be treated as a *disproven* hypothesis, not a starting point to reapply.
Suggested next steps, roughly in order of how much they'd de-risk any future attempt:
1. Get Vulkan validation layers running during a real (or at minimum `vkcube`) test
   session on `lordnikon` before writing any fix -- this alone might immediately
   surface the actual bug.
2. Consider whether the *real* per-frame cost is even a synchronization bug at all,
   versus genuine GPU/driver-side slowness on this specific hardware (an RTX 5070 on
   a `615.71.09` driver -- worth checking if that's a beta/early driver with known
   issues) that no amount of layer-side waiting logic can fix, only work around
   differently (e.g., lowering `passes`/working resolution, or the still-unmeasured
   cost of the new optical-flow private device even after being ruled out once via a
   single A/B toggle -- worth re-confirming, not just trusting that one earlier test).
3. If attempting the same class of fix again, test it against `vkcube` first (cheap,
   fast iteration, no Rockstar Games Launcher flakiness) *with validation layers on*,
   and only move to a real game after `vkcube` shows zero validation errors and a real
   throughput improvement -- this session went straight to real games both times and
   paid for it in wasted RGL-relaunch cycles either way.
4. Full environment/reproduction details, exact file locations, and every command
   needed to test on `lordnikon` are in this handoff's companion document (ask the
   user for its path if not already provided, or check the session's own summary to
   the user for it) -- SSH access, `dlssnr-cli` usage, log locations, and the
   Rockstar-Games-Launcher-flakiness gotcha (real, separate from this bug, costs
   significant time if not anticipated) are all there so they don't need
   re-discovering from scratch.

# 2026-09-12 (later still): two more real bugs found live -- a silent, session-wide
# permission bug that fully disabled neural rendering, and the v0.1.33 fix's own gap

**v0.1.33 was verified against real gameplay for the first time, and it worked --
briefly.** After `chmod 700 /tmp/dlssnr-1000` (the directory's mode was `0775`,
refused by `dlssnr_layer::shm::ensure_private_parent_dir`'s real security check) and a
clean game relaunch, real gameplay on `lordnikon` (GTA V Enhanced, menu 483fps, an
opening cinematic 472fps, live driving-around gameplay 487fps) showed **no freeze at
all** and no capture-side stalling. But `helper_frames` stayed at 0 the entire time --
the permission fix let `shm.bin` get touched, but the mapping never actually round-
tripped a frame, so this "clean" result never actually exercised the composition code
v0.1.33 touched. Before this could be re-verified with the pipeline actually engaged,
the user reported GTA V Enhanced, GTA San Andreas, *and* Crimson Desert all now
freezing solid (force-close required) with the layer enabled, recovering when the
`VKLayer_DLSS5=1` launch option was removed -- a real regression, confirmed across
multiple, unrelated games, meaning something in the shared layer code, not anything
game-specific.

**Bug 1, root cause of the permission failure (not just its symptom)**:
`dlssnr_protocol::mapping::open_at` (`crates/protocol/src/mapping.rs`, used by the CLI
and GUI -- the ones that actually create `/tmp/dlssnr-$UID/` for the first time each
boot, typically via `dlssnr-cli start` at login, well before any game touches it) used
plain `std::fs::create_dir_all(dir)` with no explicit mode -- meaning the resulting
permissions were `0o777 & !umask`, whatever umask the *first* process to ever create
the directory that boot happened to have, not necessarily private. `dlssnr_layer`'s own
`ensure_private_parent_dir` (a real, deliberate security check) then permanently
refuses to use a directory that isn't private, for the rest of that process's life, and
has no way to fix it -- so a permissive first-creation silently disabled neural
rendering for every game launched that boot, with the only visible evidence being a
`[shm] refusing ...` line in the layer's own log, which nothing sets `DLSSNR_LOG` to
capture for a real game launch. **Fixed**: `open_at` now explicitly
`set_permissions(dir, 0o700)` after `create_dir_all`, every call, not just on first
creation -- immune to umask, and self-heals a directory a previous, buggy build
already created wrong, no manual `chmod` ever needed again.

**Bug 2, the real freeze, found once the permission fix let the pipeline actually run
for the first time**: v0.1.33's own `capture_pristine` fix (bounded fence wait +
non-blocking entry check) was real and correct for *that specific* wait, but it was
not the only unbounded `wait_for_fences` on the hot path -- it was just the first one
the pipeline could ever reach, because every game launch before v0.1.34's permission
fix failed open before capture ever ran for real. `composition::gpu::GpuCompose`'s own
`dispatch_into_image_async` -- the primary per-frame compose path, not a rare
fallback -- has an *identical* unbounded entry wait (`wait_for_fences(..., u64::MAX)`,
waiting on an async slot's own previous dispatch before reusing it), called from
inside the present hook exactly like `capture_pristine`'s was. Once real frames
started actually reaching compose (thanks to bug 1's fix), this became the new
blocking point -- explaining both why v0.1.33 alone still froze on real gameplay, and
why the earlier permission-bug session never saw it (the pipeline could not get far
enough to reach this code at all). Same fix, same reasoning, applied to every fence
wait remaining on any path `capture::run` can reach for a normal frame:
`GpuCompose::dispatch_into_image_async`'s entry wait, `dispatch_into_image`/
`dispatch`'s own submission wait (both bounded to the same `ASYNC_SLOT_FENCE_TIMEOUT_NS`,
8ms), plus a non-blocking entry guard on `self.sync`'s shared fence before either of
those two functions ever resets or resizes it (the exact same "resetting a command
buffer whose previous submission hasn't finished is UB" risk `capture_pristine` had,
just on `self.sync` instead of `CaptureResources`) -- and `write_bytes_to_image`'s own
last-resort wait (`capture.rs`, bounded to `CAPTURE_FENCE_TIMEOUT_NS`). `run_sync`'s
two waits (`debug_view`/`capture_request` only, not the path any of this regression is
in) were deliberately left unbounded, same reasoning as before.

**Not yet re-verified live** -- this was shipped directly to unblock the user rather
than waiting for another RGL-cooperative relaunch (see the section above for how
unreliable that's been today). Full test suite and the smoke test are green. If a
freeze somehow persists after this, the next place to look is *why* a fresh
submission's own bounded wait isn't completing within 8ms on this hardware at all --
that would mean the underlying GPU-side cost itself, not any remaining unbounded CPU
wait, since every fence wait on the real gameplay path is now bounded.

# 2026-09-12 (later the same day): a real UB/hang bug found in `capture_pristine`,
# fixed -- the ~22x FPS regression's likely root cause, not fully confirmed live

**What was found**: `crates/layer/src/capture.rs`'s `capture_pristine` (the *synchronous*
stage-1 capture the async `run()` path still always pays for once per round-trip
cycle -- its own module doc comment already called this "no way around blocking on
it") unconditionally called `device.reset_command_buffer` on `r.cmd` every cycle, on
the sole assumption -- stated as a hard safety invariant in both its own and
`ensure()`'s doc comments -- that the *previous* cycle's submission against
`r.fence` had already been waited on to completion. That wait used `u64::MAX`
(unbounded). On real `lordnikon` hardware this did not reliably complete anywhere
near the "a few milliseconds" measured 2026-09-10: A/B'ing `enabled` twice on a real
GTA V Enhanced session (see the section above) showed FPS collapse from 196-274 to a
steady 9 the instant neural rendering turned on, with GPU utilization *dropping*
(21-26% vs. 99%) -- the signature of CPU-side blocking, not more real GPU work. A
`vkcube --width 1920 --height 1080` test under the same layer (needed to clear
`swapchain::is_plausible_game_size`'s 1280x720 floor, `vkcube`'s own default 500x500
window hits the `pass_through` path and never exercises this code at all) stalled
almost completely after its own very first frame -- single-threaded, so this is not
a multi-thread queue race, just this wait never returning promptly. Resetting a
command buffer whose previous submission has not actually finished is undefined
behavior per the Vulkan spec, independent of how slow that made things -- the
unbounded wait was not just a perf bug, it was silently relying on a completion
guarantee this code could not actually make on this hardware.

**The fix**: `capture_pristine` now does a non-blocking `get_fence_status` check at
entry and skips the whole cycle (same fail-open discipline as every other failure
path in this module) if the previous submission's fence is not yet signaled, rather
than resetting possibly-in-flight resources. `ensure()`'s resize/rebuild path (queue
family change or a capacity increase, destroying and recreating `CaptureResources`)
had the identical latent UB and got the identical guard. The one remaining
`wait_for_fences` in `capture_pristine` -- for the fresh submission this same call
just made -- is now bounded (`CAPTURE_FENCE_TIMEOUT_NS`, 8ms, generous relative to
the "a few ms" 2026-09-10 measurement) instead of `u64::MAX`, so a real stall costs
`vkQueuePresentKHR` at most that much once, fails open, and is safe to retry next
cycle precisely because the new entry check now correctly detects "still pending"
instead of blindly reusing the resources. `run_sync`'s own two `wait_for_fences`
calls (used only for `debug_view`/`capture_request`, not the path this regression is
in) were deliberately left unbounded -- not proven broken, and stage 2 specifically
writes into the swapchain image itself right before the real present call, where a
timeout would leave `image` in a genuinely unknown state rather than a safe "skip
this cycle."

**Verified**: full `cargo test` (default members) green, `scripts/smoke-test.sh`
green. **Not verified**: a second live GTA V Enhanced session against this build --
every relaunch attempt after the first successful one this session (a direct-Proton
bypass, and four separate `steam://rungameid` relaunches, including one that
eventually got far enough to spawn the real Rockstar Games Launcher/Social Club
process tree before it too fully exited) failed to reach sustained gameplay again,
Rockstar Games Launcher's own full process tree exiting completely each time --
this reproduced identically both before and after this fix was deployed, strongly
suggesting RGL itself refusing rapid successive relaunches (a real, separate,
external annoyance, not a `dlssnr` bug) rather than anything this change touched.
**So**: the reasoning for why this fix should substantially help is sound and
consistent with every real measurement gathered (the deployed layer during all of
today's measurements, including the one working GTA session, never queried this
code path with the fix in place), but the actual post-fix FPS number in a real
sustained game session is not yet confirmed. Re-test once RGL cooperates; if 9fps
somehow persists even with this fix, the next place to look is why the *fresh*
submission's own bounded wait is still not completing within 8ms, not this entry
guard (which only concerns *stale*, not fresh, work).

# 2026-09-12: GUI retabbed like upstream, and a real conflict found chasing
# "games freeze or crash" on `lordnikon`

- `gui/src/ui.rs`: the settings window is now an `AdwViewStack` of five tabs (Model,
  Motion, Composition — including the HDR white-point group, Debug, Status) instead
  of one long scrolling `AdwPreferencesPage`, matching upstream's own Qt tab layout.
  Uses `AdwViewSwitcherTitle` in the header bar plus an `AdwViewSwitcherBar` that
  reveals at narrow widths (`title-visible` property-bound between the two) — the
  standard adaptive pattern, verified by real screenshots at the default 620px width
  (bottom bar active) clicking through all five tabs. `ViewSwitcherTitle`/`Bar` are
  deprecated since libadwaita 1.4 in favor of `AdwBreakpoint`, kept anyway since the
  replacement needs a newer libadwaita than this project's `v1_4`/`v1_5` feature gate
  targets (see the gtk4/libadwaita CI-vs-local gotcha). Each tab's icon name
  (`applications-graphics-symbolic`, `camera-video-symbolic`, `view-paged-symbolic`,
  `edit-find-symbolic`, `network-transmit-receive-symbolic`) was checked against
  `/usr/share/icons/Adwaita` specifically, not whatever distro theme (Yaru/Numix) is
  active on the dev machine, after a first attempt with un-verified names rendered as
  GTK's broken-icon glyph — confirmed by screenshot, not just plausible-looking code.
- **Real, separate bug found while investigating "games either freeze or crash" on
  `lordnikon`**: upstream's real C++ package (`dlssnr` 0.2.6-3, apt, installed
  2026-09-11 for the earlier behavior-comparison session) was still installed and its
  `dlssnr-gui`/`dlssnr-helper` were still *running* at the same time as this
  rebuild's AppImage. Both implicit Vulkan layers register the same
  `enable_environment`/`disable_environment` trigger (`VKLayer_DLSS5`/
  `DLSSNR_DISABLE`) by design — this rebuild is meant to be a drop-in for upstream's
  own trigger — and both helpers target the exact same `/tmp/dlssnr-$UID/shm.bin`
  path. Any game launched with `VKLayer_DLSS5=1` therefore loaded *both* layers into
  the same process, both racing to init/write the same mapping (one in upstream's
  format, one in this project's v2 header) — a strong, sufficient explanation for
  freezes/crashes on its own, independent of anything in the new v0.1.31 optical-flow/
  SHM-v2 code. Fix: uninstalled the upstream apt package on `lordnikon`
  (`sudo apt remove dlssnr`) rather than renaming this project's trigger env vars,
  since the whole point of sharing them is drop-in compatibility with a machine that
  isn't also running upstream.
- **Live re-test after the uninstall, and a second, separate, real bug found**: with
  the conflict gone, launched GTA V Enhanced (already had `VKLayer_DLSS5=1
  %command%` in its own Steam launch options from an earlier session) for real via
  the actual running Steam client on `lordnikon`'s live desktop, with
  `dlssnr_helper` started through `dlssnr-cli start` first. **No freeze, no crash**
  — the game loaded past the legal splash straight into a resumed save, real
  rendering the whole way, and for the first time this project's own testing has
  ever seen, `NVSDK_NGX_VULKAN_EvaluateFeature` **succeeded repeatedly against real
  gameplay frames** (`-> 0x1`, not `0xbad0000b`/`0xbad00002`), each one logged with
  real timing (`upload`/`eval`/`download`, totalling roughly 10-23ms — well inside
  a 60fps budget on its own). But real, displayed FPS with neural rendering on was
  **9**, against **196-274 with it off** (`dlssnr-cli shmctl toggle enabled`,
  A/B'd twice, both directions reproduced instantly and consistently) — GPU
  utilization *dropped* with it on (21-26% enabled vs. 99% disabled), meaning this
  isn't the GPU doing more real work, something is stalling. Separately toggled
  `mvec_enabled` off (leaving `enabled=1`) to test the new optical-flow path
  specifically as a suspect, given the release notes' own "FPS impact ... has not
  been measured" caveat — **no change, still 9fps** — so this is not the optical
  flow work, and whatever it is predates v0.1.31. The ~22x frame-rate collapse is
  the strongest real candidate yet for what "games freeze or crash" actually was:
  not a hang or a crash in the literal sense, but a drop severe enough (9fps) to
  read as one. Left the running session with `enabled=0` (full FPS restored) rather
  than the persisted default, so the game stays playable; `dlssnr-cli shmctl toggle
  enabled` flips it back for further diagnosis.
  **Not yet done, and the obvious next step**: find where the stall actually is.
  The helper's own per-frame cost (10-23ms) doesn't explain a 9fps result (~111ms/
  frame) — the gap has to be on the *layer* side: capture, the SHM round-trip wait,
  or write-back. `capture.rs`'s async cross-frame pipelining (see the
  "Cross-frame async pipelining" section below) was specifically built to avoid a
  synchronous per-frame stall; whether it's actually the path being taken in this
  real game (vs. falling back to the synchronous path for some real-game-specific
  reason `vkcube` never exercised) hasn't been checked. `DLSSNR_LOG` wasn't set for
  this real launch (only `smoke-test.sh` sets it explicitly), so the layer's own
  log went to stderr, into whatever Steam/Proton did with it — capturing that
  directly (set `DLSSNR_LOG` in the game's own Steam launch options alongside
  `VKLayer_DLSS5=1`) is the fastest way to see which path a real frame is actually
  taking.

# 2026-09-11 follow-up implementation (unreleased)

This section supersedes the historical open-item descriptions below.

- SHM v2 preserves BGRA8 separately from RGBA8. Helper Color/Output formats and
  resource reuse now include the captured format. Layer composition retains its
  existing swizzle; raw SHM bytes are never swizzled twice.
- Native optical flow runs in the layer, on a private Vulkan device on the same
  physical GPU. It explicitly requests opticalFlow and synchronization2 and a
  queue supporting both OPTICAL_FLOW and TRANSFER; it does not alter the game's
  device create chain. Captured proxies are uploaded to two BGRA input images.
  These are consecutive **captured/model-input frames**, not necessarily adjacent
  presents in the existing asynchronous pipeline. Current-to-previous flow is
  expanded from the hardware grid, converted from signed fixed-point /32 to
  R16G16_SFLOAT, deadzoned below 0.5 pixels, and sent in a third sparse SHM region.
  Motion quality and units are honored. Unsupported devices fail open without
  repeatedly trying the same configuration. Resize, format/quality changes,
  disabling motion, and gaps over one second discard history; detected scene cuts
  discard vectors. The helper resets temporal history when expected vectors are
  absent. Optical flow uses host staging and waits on its private queue; FPS impact
  in real gameplay has not been measured. Async composition remains the default.
- HDR white-point source/manual value/scale/trim and a key-capture/clear widget are
  bound and persisted (36 settings). Escape cancels capture; focus loss cancels;
  Linux XKB hardware codes are converted to evdev codes. The UI explicitly says
  that existing HDR processing and in-game hotkey polling are still unimplemented;
  these changes implement the handoff's GUI-binding scope, not those larger paths.
- Validation: protocol/layer tests, GUI checks/tests, Windows helper cross-build,
  loader smoke test, and dedicated RTX 5070 optical-flow/SHM checks. The GPU
  translation check produced median (-8,0) for an 8-pixel right shift and stationary
  mean absolute component sum 0.02483368 pixels. No live-game A/B capture, HDR
  visual verification, release, or deployment was performed for this patch.
- All participating binaries must be rebuilt/restarted together for SHM v2.
  Header size is 1960; appended motion validity/units offsets are 1952/1956.
  The full mapping now reserves header + 3*MAX_FRAME (sparse). Never mix v1/v2
  processes on the same mapping. The existing mapping-open policy reinitializes
  mismatched headers; stop old participants before starting a v2 build.

# NeuralForge development history (pre-rename codename: dlssnr)

A from-scratch Rust/GTK4/libadwaita rebuild of
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (C++/Qt6): a Vulkan implicit
layer + Windows NGX helper that runs NVIDIA DLSS 5 Neural Rendering on Linux/Proton
games, forwarding frames to a Windows NGX helper running under Wine/Proton today (and,
per upstream's own stated roadmap, a native-Linux helper once NVIDIA ships one — the
whole point of the shared-memory seam below is that nothing on this side of it needs to
change when that happens). Distributed as a single AppImage built with appimagetool
directly against the system GTK 4/libadwaita, and it needs **no root/pkexec step at
all**: everything lives under `~/.local/share`, `~/.config`, `/tmp/dlssnr-$UID/`.

**Original implementation plan** (a design document kept outside the repo, since
retired; its conclusions are recorded here and in `ATTRIBUTION.md`). It covered the
licensing ground rules (upstream has no LICENSE file and admits GPL-3.0 contamination in parts of its own tree — nothing here
may be produced by reading upstream's source and translating it), the NGX
authorization-bypass mechanism this rebuild knowingly carries forward (accepted risk,
isolate it, never disguise it), and the per-crate design.

## Current state

Every crate has real code now (milestones 1-3 solid and real-tested; 4 phase A/B landed
2026-09-09 against real hardware — capture/transport/NGX-evaluate genuinely run every
frame now, but this project's own composition math still never reaches the presented
frame, see the `composition` section below for exactly what that does and doesn't
mean; 5 and 6 real and working, with the caveats below). Read each crate's own gotchas
section before touching it — "compiles" and "the plan says this milestone is done" are
not the same claim anywhere in this repo.

**2026-09-10: real, repeated `EvaluateFeature` success against a real, legitimately-
signed `nvngx_dlssnr.dll`, for the first time this project's own code has ever achieved
it** (previously only ever seen from upstream's C++ build) — see the "First confirmed
neural-rendering success" section below for the full test and what it does/doesn't
prove.

`helper` loads `nvngx_dlssnr.dll`, installs the caller-identity spoof, initializes NGX,
and creates Feature 18 (`ngx::load_and_init`) -- the path that actually exercises the
spoof and the SEH guard, wrapped in `crates/helper/src/guard.rs`'s VEH+`setjmp`/
`longjmp` mechanism. **As of 2026-09-09 there is a real per-frame `EvaluateFeature`
loop** (`crates/helper/src/frame.rs`, wired into `main.rs`) — see the `composition`
section below for what it does, what it caught and fixed on real hardware, and what's
still not proven end to end. As of the same date this is real-tested, not just
compiled: the cross-compile toolchain now exists on this dev machine (see below), and
three purpose-built examples (`examples/guard_test.rs`, `spoof_test.rs`,
`spoof_install_test.rs`) run the SEH guard and the caller-identity spoof for real under
Wine, plus the full `dlssnr_helper.exe` binary itself was run under Wine and its
shared-memory mapping read back correctly from Linux-built code (true cross-toolchain
interop, not just matching offsets on paper). One real bug was caught and fixed this
way — an infinite-recursion stack overflow that `cargo check` structurally could not
have found — see `helper` gotchas below.

`layer` hooks `vkCreateSwapchainKHR`/`vkDestroySwapchainKHR`/`vkQueuePresentKHR` via
Google's `vulkan_layer` crate and runs the real shared-memory round trip on present.
**As of 2026-09-09 `queue_present_khr` really does capture the presented image and
round-trip it** (`crates/layer/src/capture.rs`), and **as of 2026-09-10 the write-back
actually uses the helper's answer** instead of always re-presenting the untouched
capture — see the `composition` section below for exactly what that fix does and its
one real caveat (not yet verified against a real present cycle; this sandbox has no
display/swapchain to drive one through). Verified via
`cargo test -p dlssnr-layer` (the SHM round-trip state machine, including a simulated
echo-helper thread and the private-directory security check) and
`scripts/smoke-test.sh` (a real `VkInstance`/`VkDevice` through the system Vulkan
loader with the layer actually negotiated and inserted into the call chain — this is
what caught and fixed a real bug: `DlssnrDeviceInfo::new` originally panicked on any
device that didn't enable `VK_KHR_swapchain`, which would have crashed every
compute-only Vulkan app the layer got loaded into).

## Workspace layout

```
crates/
  protocol/   #[repr(C)] SHM wire format — the whole layer<->helper contract, no
              GTK/tokio deps, builds for both x86_64-unknown-linux-gnu and
              x86_64-pc-windows-gnu with identical layout.
  layer/      cdylib, x86_64-unknown-linux-gnu only. The Vulkan implicit layer.
  helper/     bin, x86_64-pc-windows-gnu only, runs under Wine/Proton. Never built or
              run natively on a Linux dev machine.
  supervisor/ lib, x86_64-unknown-linux-gnu only. Config/paths/process-supervision
              shared between `gui` and `cli` (extracted 2026-09-09) so both start/stop
              the helper through the same code.
  gui/        bin, gtk4 + libadwaita, flat src/*.rs modules.
  cli/        bin, the `dlssnr-cli` helper-manager (init/start/stop/status/doctor/
              runners/detect-gpu/import-binaries).
```

`helper` is excluded from the workspace's `default-members` (see root `Cargo.toml`), so
a **plain, flag-less** `cargo build`/`cargo test` on a Linux dev box never tries to link
Windows-only code against the host toolchain. **`--workspace`/`--all` ignore
`default-members` entirely** — that's documented Cargo behavior, not a bug here — so
`cargo build --workspace`/`cargo test --workspace` *will* try to build `helper` for the
host and fail at the link step the moment `helper` has any real `#[link(name =
"kernel32")]` code (confirmed: it linked fine back when `helper` was still an empty
stub with nothing to resolve, and fails now). **Use the plain, flag-less form for
"build/test everything except helper"** — `cargo build`, `cargo test`, `cargo check`
all correctly skip it that way. Reach for `-p dlssnr-helper` explicitly (with
`cargo check` for a quick pass, or the cross-compile toolchain below for the real
thing) rather than `--workspace` when `helper` itself is what you're working on.

`helper` needs `mingw-w64` installed (`apt install rustup gcc-mingw-w64-x86-64
binutils-mingw-w64-x86-64`, `rustup toolchain install stable`, `rustup target add
--toolchain stable-x86_64-unknown-linux-gnu x86_64-pc-windows-gnu`) plus a linker
override in `~/.cargo/config.toml` (`[target.x86_64-pc-windows-gnu] linker =
"x86_64-w64-mingw32-gcc"`, `ar = "x86_64-w64-mingw32-ar"`, `runner = "wine"`) — all now
set up on this dev machine (2026-09-09). **Important:** apt's `rustup` package
replaces `/usr/bin/cargo`/`/usr/bin/rustc` with its own toolchain-dispatch shims and
sets whichever toolchain you first `rustup toolchain install` as `default` — installing
a second (`stable`) toolchain just to get the `x86_64-pc-windows-gnu` target would have
silently switched the *whole project's* everyday Rust version out from under it. Fixed
by linking the pre-existing system install as its own named toolchain
(`rustup toolchain link system /usr/lib/rust-1.93`, then `rustup default system`) and
keeping `stable` around only for explicit cross-compiles
(`cargo +stable-x86_64-unknown-linux-gnu build --target x86_64-pc-windows-gnu -p
dlssnr-helper`) — plain `cargo`/`rustc` still resolve to 1.93.1 for everything else.
Real cross-compiling + linking against the mingw CRT is now verified working (see
`helper` gotchas below) — this is no longer "written but unverified."

## `protocol` gotchas

- **The three free-text fields** (`helper_reason`, `layer_reason`, `game_name`) are
  `UnsafeCell<[u8; N]>`, not plain byte arrays, specifically so mutating them through
  `&ShmHeader` (which is all any caller ever has — the header lives in an mmap'd region
  another process can write at any time, never behind a `&mut`) is defined behavior.
  Always go through `set_helper_reason`/`helper_reason`/etc., never touch the fields
  directly — they're private for exactly this reason. See `store_seq_guarded`/
  `load_seq_guarded` in `header.rs` for the seqlock-style protocol that makes a
  concurrent read-during-write never observe a torn string.
- **`ShmHeader` has no normal constructor.** Every real instance is a raw-pointer cast
  onto an existing mmap'd region (zero-filled by `ftruncate`, same as upstream's
  approach) — never a `ShmHeader { .. }` struct literal. `Default` exists only to give
  tests (and anything that wants a heap-allocated instance not backed by shared memory)
  the same all-zero starting point a fresh mapping already has; it is implemented via
  `mem::zeroed()` precisely because that's what a real mapping's bytes look like before
  `init_defaults()` runs, not because zero happens to be a convenient default. Don't
  "simplify" this into `#[derive(Default)]` — it doesn't compile (`[u8; 192]` has no
  `Default` impl in std beyond N=32), and even where it would, deriving would produce a
  *different* meaning than "matches a fresh mapping's raw bytes" for any field whose
  real, `init_defaults()`-assigned default isn't zero (most of the float fields
  default to `1.0`, not `0.0`).
- **The offset/size `const _: () = assert!(...)` block near the top of `header.rs`
  is load-bearing, not decorative.** If you add, remove, or reorder a field and one of
  these fails to compile, that's the check doing its job — bump `SHM_VERSION` in
  `lib.rs` in the same change, then recompute the asserted numbers (the easiest way:
  temporarily add a `#[test]` that prints `size_of`/`offset_of!` for the fields you
  need, run it, copy the numbers in, delete the test).
- **This protocol is not wire-compatible with upstream's**, on purpose — different
  magic (`SHM_MAGIC = "DSN1"` vs. upstream's `"GNR2"`), independent versioning starting
  at 1. A stale mapping from anything else must never be half-read as if it were ours.

## `layer` gotchas

- **`vulkan-layer` is a git dependency pinned to an exact commit**, not a crates.io
  release (it has none). `ash` in `crates/layer/Cargo.toml` is pinned to the *exact*
  version `vulkan-layer` itself depends on (`=0.37.3`) — Cargo treats different 0.x
  versions of the same crate as distinct, incompatible types, so a newer `ash` here
  would fail to compile against `vulkan_layer`'s trait signatures. If bumping the
  `vulkan-layer` git rev, check what `ash` version it depends on first and match it.
- **`panic = "abort"` is set workspace-wide** (root `Cargo.toml`, not per-crate — Cargo
  profiles other than a few per-package-overridable knobs apply to the whole build).
  Required for `dlssnr-layer`: it's a cdylib the Vulkan loader calls back into across a
  plain C ABI boundary, and an unwinding panic crossing that boundary is undefined
  behavior. `vulkan-layer`'s own example layers set the same thing for the same reason.
  `cargo test` still works fine with this set (confirmed on this toolchain).
- **Never resolve a next-in-chain device function pointer and assume it's there.**
  `device::resolve()` returns `Option<F>`, not `F` — a device that never enabled
  `VK_KHR_swapchain` (any compute-only app, or any device an app just never presents
  from) legitimately has no `vkCreateSwapchainKHR` to resolve. The first version of this
  panicked on a missing pointer; `scripts/smoke-test.sh`, which creates a device with no
  extensions enabled at all, caught it immediately. Every hook that depends on a
  resolved pointer checks for `None` first and returns `LayerResult::Unhandled` — that
  tells the `vulkan_layer` framework to fall through to its own next-dispatch exactly as
  if this crate weren't there, which is always correct when we have nothing useful to do
  anyway (an app that never enabled the extension will also never call the function).
- **`vulkan_layer`'s `DeviceInfo`/`InstanceInfo` traits don't hand `create_device_info`
  a way to reach whatever `create_instance_info` returned for the owning instance —
  worked around, not a blocker anymore.** `create_device_info` does get
  `vk::PhysicalDevice` directly (it always did; an earlier version of this note
  conflated that with the separate, real gap: no `ash::Instance` reference), but
  needed one to query physical-device memory properties when `capture.rs` builds its
  staging buffer. Fixed 2026-09-09 with `lib.rs`'s `static CURRENT_INSTANCE:
  Mutex<Option<Arc<ash::Instance>>>` — `create_instance_info` stashes the instance
  there, `create_device_info` reads it back. A documented "last one wins"
  simplification (games overwhelmingly create exactly one `VkInstance`), same spirit
  as `device::PRIMARY`'s one-swapchain-at-a-time assumption elsewhere in this crate.
- **Testing an implicit-type layer via `VK_LAYER_PATH` needs `VK_INSTANCE_LAYERS` too.**
  `VK_LAYER_PATH`/`VK_ADD_LAYER_PATH` only add manifests to the *explicit*-layer search;
  an implicit-type manifest found there is not auto-enabled the way a real one dropped
  into an actual `implicit_layer.d` directory would be (confirmed with
  `VK_LOADER_DEBUG=layer` — the manifest was found but never inserted into the call
  chain until `VK_INSTANCE_LAYERS=VK_LAYER_dlssnr_neural` was also set).
  `scripts/smoke-test.sh` sets both; a real install only ever needs
  `enable_environment` (see `data/VK_LAYER_dlssnr_neural.json`).
- **This sandbox has a real (software) Vulkan loader + ICD** (`libvulkan.so.1` +
  lavapipe, no `vulkaninfo` binary though) — `scripts/smoke-test.sh` genuinely creates a
  `VkInstance`/`VkDevice` through it, so it's worth running again after any change to
  `device.rs`/`lib.rs`, not just `cargo test`.

## `helper` gotchas

- **No `windows`/`windows-sys` dependency, on purpose.** Every WinAPI call
  (`LoadLibraryExW`, `GetProcAddress`, `VirtualProtect`, `AddVectoredExceptionHandler`,
  `CreateFileMappingW`, ...) is declared directly against `kernel32.dll` via
  `#[link(name = "kernel32")] extern "system" { ... }` blocks. These are among the
  oldest, most stable parts of the Win32 ABI (unchanged since Windows XP/Vista), and
  declaring them directly means correctness never depends on how some wrapper crate
  happens to shape its own bindings for them -- which mattered a lot while writing this
  with zero ability to compiler-check the Windows-specific paths (see below).
- **`cargo check -p dlssnr-helper` on the Linux host actually works and catches real
  bugs** -- `cargo check` only type-checks (`--emit=metadata`), never links, so it never
  needs to resolve the `kernel32` symbols the way `cargo build`/a real cross-compile
  would. It already caught two real mistakes (a private-struct-leaked-through-a-public-
  API error, an unused import) before any toolchain existed to build this for real.
  **What it does NOT catch**: anything about whether the raw offsets in `spoof.rs`'s
  `pe` module actually match a real PE image's layout, whether `setjmp`/`longjmp` even
  link successfully against the mingw CRT (see the next point), or any runtime
  behavior at all. Run it after every change to this crate regardless -- it's free and
  it's already proven itself worth running -- but it is not a substitute for the real
  cross-compile + Wine smoke test this crate still needs once the toolchain exists.
- **The IAT-patching recursion bug — the one real runtime bug found so far, and why
  `spoof.rs` has `REAL_GET_MODULE_FILE_NAME_W`.** The first version of
  `spoofed_get_module_file_name_w`'s pass-through branch (for a query about some module
  other than the spoofed one) called the plain `extern "system" fn GetModuleFileNameW`
  declared at the top of `spoof.rs`. That compiles fine and type-checks fine — the bug
  is purely about *which IAT slot that call resolves through at runtime*, invisible to
  the type system and therefore to `cargo check` no matter how carefully it's read.
  `examples/spoof_install_test.rs` (deliberately, for a self-contained test with no
  second DLL needed) installs the spoof against *its own* module — and every call to
  the plain import inside the same binary resolves through that binary's own single IAT
  slot for `GetModuleFileNameW`, which `install()` had just repointed at
  `spoofed_get_module_file_name_w` itself. Result: the pass-through branch called
  itself, forever, and the process stack-overflowed under Wine within milliseconds. Real
  usage (`ngx.rs` always patches a *different*, separately-loaded DLL — `nvngx_dlssnr.dll`
  or `nvngx.dll`, never this helper's own .exe) never hits this specific scenario, but
  relying on "the module I patch is never the one my own fallback call resolves through"
  as an unstated invariant was exactly the kind of latent landmine worth fixing outright
  rather than rationalizing away — matches what upstream's own C++ already does
  (`g_realGetModuleFileNameW`, captured once, used for every pass-through call,
  regardless of which module was patched). Fixed by capturing the *real* function
  pointer once in `install()` (before any patch) into a dedicated static and calling
  that for pass-through, never the plain import. **The general lesson**: when patching
  a well-known import in-process, never assume your own fallback call to "the real
  function" resolves anywhere other than through the exact slot you might have just
  patched — capture the original value explicitly instead of trusting which name you
  wrote in the source.
- **`guard.rs`'s `jmp_buf` (256 bytes, generously over-sized) and raw `setjmp`/`longjmp`
  linked and worked on the first real test** — `examples/guard_test.rs` triggers a real
  `0xC0000005` (`EXCEPTION_ACCESS_VIOLATION`) inside a `guard::guarded` closure (reading
  through a deliberately unmapped, non-null address — a *literal* null dereference gets
  intercepted by Rust's own debug-mode "unsafe precondition checks" and turned into a
  clean panic+abort before the CPU ever really faults, which was this test's own first,
  wrong version) and confirms the process survives, the guard returns the fail value,
  and execution continues normally afterward. Wine logs a benign
  `err:seh:RtlUnwindEx invalid frame` warning during this — expected and harmless: the
  whole point of the `longjmp`-based recovery is that it bypasses the *normal* SEH
  unwind bookkeeping Wine is noticing went missing, not that anything actually broke.
- **The PE-parsing offsets in `spoof.rs` are hardcoded from documented, ABI-stable
  struct layouts** (`IMAGE_DOS_HEADER`/`IMAGE_NT_HEADERS64`/`IMAGE_OPTIONAL_HEADER64`/
  `IMAGE_IMPORT_DESCRIPTOR`), not derived from modeling those structs as Rust types —
  deliberately, since a single misplaced field in the ~30-field `IMAGE_OPTIONAL_HEADER64`
  would silently shift every offset after it. **Confirmed correct against a real PE
  image**: `examples/spoof_test.rs` runs `find_imported_function_slot` against its own
  compiled `x86_64-pc-windows-gnu` binary (every such binary imports
  `GetModuleFileNameW` from `KERNEL32.dll` itself, so this needs no NVIDIA DLL at all)
  and confirms it finds a real, already-loader-resolved IAT slot — no NVIDIA DLL needed
  for this specific check.
- **As of 2026-09-09, `ngx.rs`/`frame.rs` really do call `EvaluateFeature` with real
  bound `DLSSNR.Color`/`Output`/`MVec` Vulkan resources** — the note this used to carry
  (`EvaluateFeature` needs real bound resources "which is milestone 4's job", a 1x1
  `CreateFeature` placeholder) is stale; see the `composition` section below for what's
  actually real now, what real hardware bugs it found and fixed (a driver-level hang,
  zero device extensions enabled, an unguarded crash), and what's still not proven
  (no legitimate `nvngx_dlssnr.dll` has ever been available to test against, so
  `CreateFeature`'s return is still a clean rejection, not a success).
- **`lib.rs` exists alongside `main.rs` specifically so `examples/` can exercise
  individual modules** (`guard`, `spoof`) directly without a full helper run — `main.rs`
  is now a thin binary wrapper around the `dlssnr_helper` library crate. Keep this
  split; it's what makes the Wine-based example tests above possible at all.

## FIXED (2026-09-10, v0.1.6): the layer crashed on its own real activation path

**Root cause found and fixed.** `vulkan_layer::Global::create_instance`'s default
fallback path (taken whenever `GlobalHooks::create_instance` returns `Unhandled`, which
is what `StubGlobalHooks` — what this crate used before this fix — always does) calls
`ash::vk::EntryFnV1_0::load`, which eagerly resolves *all three* Vulkan 1.0 global entry
points (`vkCreateInstance`, `vkEnumerateInstanceExtensionProperties`,
`vkEnumerateInstanceLayerProperties`) through the chained, `VK_NULL_HANDLE`-instance
`vkGetInstanceProcAddr` — even though a layer that only calls `entry.create_instance`
afterward (as this one does) never uses the other two. Resolving
`vkEnumerateInstanceExtensionProperties` that way segfaults inside
`libVkLayer_MESA_device_select.so` on this dev machine's Mesa build, 100% of the time;
resolving `vkCreateInstance` through the exact same chained pointer, one field earlier,
does not. **Confirmed via `gdb` this is not a dlssnr-specific bug at all**: building and
running `vulkan-layer`'s own pristine, unmodified `hello-world` example under the
identical implicit-activation scenario in this same sandbox produces the *exact same*
crash and backtrace (`libVkLayer_MESA_device_select.so` → `EntryFnV1_0::load` closure →
`vulkan_layer::Global::create_instance`, `vulkan-layer/src/lib.rs:710`/`716`) — this is a
bug in the pinned `vulkan-layer` commit's interaction with this Mesa build (worth filing
upstream at some point, but not blocking — see the fix below), not anything this
project's own code did wrong.

**The fix** (`crates/layer/src/lib.rs`): implemented `GlobalHooks::create_instance`
ourselves via a new `DlssnrGlobalHooks` type (replacing `StubGlobalHooks`), resolving
only `vkCreateInstance` through the chained `pfnNextGetInstanceProcAddr` and calling it
directly — the same pattern `vulkan-layer`'s own doc-comment example for a layer that
needs to intercept `vkCreateInstance` shows. This never makes the
`vkEnumerateInstanceExtensionProperties`/`vkEnumerateInstanceLayerProperties` queries
that crash, since this layer never needed them in the first place.

**Verified fixed, for real, not just compiled**:
- Locally (this sandbox has the identical `libVkLayer_MESA_device_select.so` present):
  5/5 clean runs of `crates/layer/examples/smoke` under real implicit activation
  (`VK_ADD_IMPLICIT_LAYER_PATH` + `VKLayer_DLSS5=1`, no `VK_INSTANCE_LAYERS`) — exit 0,
  full instance/device creation, every time. Previously 100% reproducible crash, 0/5.
- On `lordnikon` (real GPU/driver, the machine the original crash was found and
  bisected on): 3/3 clean `vkcube --width 1920 --height 1080` runs under the exact
  repro command line from the previous investigation
  (`VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_dlssnr`, `VKLayer_DLSS5=1`) — each ran the full
  8-second `timeout` (exit 124, not a crash), with the layer log showing real capture
  and shared-memory round trips happening every frame (`answered=false` is expected —
  no helper was running for this test, which fails open correctly, same as always).
- Full `cargo test` (workspace, all 4 native crates) stays green: 36 tests passed, 0
  failed. Both `scripts/smoke-test.sh` (explicit activation, the crate's existing
  regression test) and the mingw cross-compile check for `helper` still pass —
  confirming this change didn't regress either path.

**What this unblocks**: this was "the single most important thing to fix in this
project" as of 0.1.5 — nothing past `vkCreateInstance` could work while it held, on any
machine with Mesa's `device_select` present (the default on this dev setup and common
on real Linux desktops generally). The full bisection history that led here is kept
below for the record — it's what ruled out everything else first and narrowed this down
to "something about how we call the next layer during instance creation," which is
exactly where the real bug turned out to be.

### Original bisection (kept for the record; the crash above is now fixed)

**Upstream, for the record, fully works.** With a real, legitimately-signed
`nvngx_dlssnr.dll` in place (verified via `osslsigncode` — NVIDIA Corporation,
DigiCert chain, valid — after multiple wrong/tampered files were correctly rejected
earlier), upstream's real installed package (`dlssnr` 0.2.6-1) was run end-to-end for
the first time this project has ever seen: real `vkcube` frames, real
`VULKAN_CreateFeature(18)` success (`handle` non-null, real size), real
`EvaluateFeature`, real NV optical flow, `[helper] neural ready`. The one thing that
had to be fixed to get there wasn't code: `/tmp/dlssnr-1000` (the SHM runtime dir) was
`0775` instead of `0700` — group-writable, likely from an earlier session's shell
umask — and both upstream's and our own "refuse a non-private directory" security
check correctly rejected it. `chmod 700` fixed it immediately. **If DLSS5 NR ever
silently refuses to work on a real setup, check this first.**

**Our own layer does not work — it segfaults every time it's loaded the way it will
always actually be loaded.** `VK_LAYER_dlssnr_neural` (`crates/layer`, built from the
same commit released as `v0.1.4`) crashes 100% of the time when activated implicitly
via `VKLayer_DLSS5=1` (its real, only, intended activation mechanism — identical to
how upstream's own layer activates) in the presence of Mesa's `device_select` implicit
layer (`libVkLayer_MESA_device_select.so`) — which is present and active by default on
this machine, and is common enough on Linux desktops generally (anything with more
than one GPU, or some distros' default Vulkan setup) that this would very plausibly
also crash on a real player's machine, not just this dev box. **This is the single
most important thing to fix in this project right now** — nothing past
`vkCreateInstance` can work while this holds.

What's actually established, precisely, via direct `gdb` reproduction and a long,
systematic bisection (not inference) — **this took an entire extra investigation pass
past the first writeup below to get this far, so please read all of it before
re-testing any of the already-ruled-out hypotheses**:

- Crash signature: `SIGSEGV` inside `libVkLayer_MESA_device_select.so`, reached via
  `vulkan_layer::Global<DlssnrLayer>::create_instance` →
  `ash::vk::features::EntryFnV1_0::load` → the next layer's real `vkCreateInstance` —
  i.e., the crash is standard, correct framework code (Google's `vulkan_layer` crate,
  not anything we wrote) calling into Mesa, and Mesa's own code is what actually
  faults.
- **Ruled out: chain position/ordering, definitively.** `VK_LOADER_DEBUG=layer`
  confirms our layer sits *before* Mesa's `device_select` in the chain
  (`App → dlssnr_neural → device_select → Drivers`) when activated implicitly — i.e.
  we call *into* Mesa. Checked upstream's real, working layer in the identical
  implicit-activation scenario: **it sits in the exact same position** (also calls
  into Mesa the same way) **and does not crash.** So chain position alone isn't it.
- **Ruled out: `DlssnrDeviceInfo`/the device hooks entirely.** Temporarily swapped
  `type DeviceInfo`/`DeviceInfoContainer` to `StubDeviceInfo` (true no-op, matching the
  reference example) and stubbed `create_device_info` to `Default::default()`, keeping
  the real manifest and implicit activation — **still crashes, 3/3 runs.** Whatever
  this is, it has nothing to do with what device-level functions we hook.
- **Ruled out: the `CURRENT_INSTANCE` static/Mutex side effect in
  `create_instance_info`.** Removed the store entirely (pure `Default::default()`,
  byte-for-byte matching the reference example's own `create_instance_info`) — still
  crashes, 3/3.
- **Ruled out: every other module in this crate.** With `DeviceInfo` already stubbed,
  `composition`/`capture`/`device`/`shm`/`swapchain`/`logging` were fully dead code —
  commented out all six `mod` declarations (and the now-orphaned `use device::...`),
  producing a `.so` that is, in Rust-level shape, essentially identical to the
  reference example (same `Layer` impl shape, same `Stub*` types throughout, similar
  final size: 24.7 MB vs. the reference's 24.98 MB). **Still crashes, 3/3.**
- **Ruled out: `dlssnr-protocol` and `libc` as dependencies.** Removed both from
  `crates/layer/Cargo.toml` for this same minimal build (neither was even referenced
  anymore once the modules above were gone) — **still crashes, 3/3.**
- **Ruled out: `panic = "abort"` vs. `"unwind"`.** The reference example's own
  `Cargo.toml` sets `panic = "abort"` too, but `cargo tree` warns that setting is
  ignored there because it isn't the workspace root — checked the *real* root
  (`vk-layer-for-rust`'s own top-level `Cargo.toml`) and it sets the identical
  `panic = "abort"` for both profiles. Both builds use the same panic strategy after
  all.
- **Ruled out (as far as it's practical to pin): transitive dependency version skew.**
  `cargo tree` showed our workspace resolving newer patch versions of several of
  `vulkan-layer`'s own transitive deps than the reference example's isolated lockfile
  (`bytemuck` 1.25.2 vs. 1.16.1, `thiserror` 1.0.69 vs 1.0.61, `log` 0.4.34 vs 0.4.22,
  `once_cell` 1.21.4 vs 1.19.0, `quote`/`cfg-if`/`autocfg` similarly newer) — `ash` and
  `vulkan-layer` itself (same pinned git commit) were already identical either way.
  Pinned every one of these down to the reference's exact version via `cargo update -p
  <pkg> --precise <ver>` (`smallvec` and `proc-macro2`/`syn` couldn't be forced down —
  other workspace members' own minimum-version requirements blocked it) and rebuilt —
  **still crashes, 3/3.**
- **What's left, genuinely not yet tested**: the two dependency versions that
  couldn't be pinned down to match (`smallvec`, `proc-macro2`/`syn` — both are
  extremely unlikely candidates: `smallvec` is a data structure with no obvious reason
  to affect an unrelated crate's FFI boundary, and `proc-macro2`/`syn` only run at
  *compile* time generating code, not at runtime); building this crate in a
  completely standalone directory with no parent workspace at all (to rule out any
  workspace-resolution effect this bisection hasn't captured); and the possibility
  that this is a genuine bug in the pinned `vulkan-layer` crate commit itself that only
  reproduces with *this specific machine's* Mesa/driver build, which would need
  filing upstream with Google's repo to make further progress on.
- **All bisect edits were reverted after each test** — `crates/layer/src/lib.rs`,
  `crates/layer/Cargo.toml`, and `Cargo.lock` are all back to their real, correct,
  committed state (confirmed via `git diff --stat` showing nothing) — nothing about
  this investigation is reflected in the actual code. `spec_version = vk::API_VERSION_1_1`
  (a real, if inert, change from the first pass of this investigation) is still in
  place from before; see below.
- Reproduce with: real machine (needs actual Mesa `device_select` present — check
  `/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json` exists),
  `VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_dlssnr` (keeps upstream's real layer out of the
  way without touching its system-owned manifest), `VKLayer_DLSS5=1`, plain
  `vkcube --width 1920 --height 1080` (or under `gdb -batch -ex run -ex bt --args
  vkcube ...` for a fresh backtrace). Also worth knowing: `vkcube`'s default (Wayland)
  WSI mode doesn't create an X11-visible window and can't be screenshotted with
  `import`/`xdotool` the way everything else in this project's testing has been —
  `--width`/`--height` don't change that; process liveness/log output/gdb are the only
  ways to observe it, not a screenshot. When testing a hand-written manifest for any
  comparison layer, remember the Vulkan loader **requires both `enable_environment`
  and `disable_environment`** for a valid implicit-layer manifest — a manifest missing
  either gets silently skipped with only a `WARNING` in `VK_LOADER_DEBUG=all` output,
  easy to mistake for "this layer works fine" when it was actually never loaded at all
  (this cost real time in this investigation itself).

## First confirmed neural-rendering success (2026-09-10, `lordnikon`, real hardware)

**With the crash fix above landed, this project's own `helper` + `layer` produced real,
repeated, successful DLSS 5 Neural Rendering evaluations against a real,
legitimately-signed `nvngx_dlssnr.dll` for the first time.** Everything before this was
either simulated (unit tests), run against a bad/rejected DLL, or blocked outright by
the implicit-activation crash — this is the first time the full real path has actually
been exercised end to end with a model that can say yes.

**What was run**: a fresh cross-compiled release `dlssnr_helper.exe`, launched directly
under the real `Proton-CachyOS Latest` runner (bypassing `dlssnr-cli`/`dlssnr-gui`,
neither of which is deployed to `lordnikon` yet — this called `dlssnr_supervisor::start`'s
exact env var set by hand: `WINEPREFIX`/`STEAM_COMPAT_DATA_PATH` pointed at a fresh,
throwaway prefix, `STEAM_COMPAT_CLIENT_INSTALL_PATH` at the real Steam install,
`DLSSNR_BIN_DIR=Z:/home/alex/.local/share/dlssnr/binaries` at the real, hash-verified
NGX binaries already on that machine, `PROTON_ENABLE_NVAPI=1`/`DLSSNR_SKIP_NVAPI=1` as
`start()` itself sets), then real `vkcube` with `VK_LAYER_dlssnr_neural` activated the
same real, implicit way the crash fix above was verified with. **Both sides were pointed
at an isolated `DLSSNR_UID=rstest`** (`/tmp/dlssnr-rstest/`, not the real `/tmp/dlssnr-1000/`)
specifically so this test could never collide with `lordnikon`'s own real, working
upstream install, which happens to share this project's exact `~/.config/dlssnr`/
`~/.local/share/dlssnr` paths by design (see `crates/supervisor/src/paths.rs` — this
project deliberately mirrors upstream's own layout so it can be a drop-in alternative).

**What the helper's real log showed, in order**:
- `AllocateParameters -> 0x1`, `params round-trip self-test -> 0x1 ... readback=0x5a5a`
  — the parameter-vtable plumbing works.
- `VULKAN_Init_Ext -> 0x1` — NGX itself initializes cleanly against the real Vulkan
  device this helper's own `frame.rs`/`ngx.rs` set up.
- `GetFeatureRequirements -> 0xbad00005` — the diagnostic-only, not-gated-on call (see
  the `composition` section below) still fails; harmless, exactly as already documented.
- Once real frames started arriving from the layer: **`VULKAN_CreateFeature(18) -> 0x1
  seh=0x0 handle=0x2b7ac00 size=1920x1080`** — a real, non-null feature handle, the
  identical success shape previously only ever seen from upstream's own C++ build.
- **`EvaluateFeature -> 0x1` on 243 of 244 captured frames** (the one `evaluated=false`
  is frame 1, captured before `CreateFeature` had run yet — expected, not a failure).
  Zero evaluation failures across the whole 10-second `vkcube` run.
- The layer's own log agreed: `round trip answered=true` for every one of those frames.

**What this does and does not prove**: this confirms the full real path — capture,
SHM transport, `EvaluateFeature` with real bound Vulkan resources, the parameter
plumbing, the caller-identity spoof, the SEH guard — genuinely works end to end against
a real model on real hardware, repeatedly, not just once. It does **not** yet prove the
*visual* result is correct (this test never looked at a frame; `compare_mode`/
`debug_view` would be how to actually see the model's answer, and this project's own
composition math still isn't wired into the write-back — see `composition` below,
unchanged by this test) or that real optical flow is being fed in (`MVec` is still the
all-zero stand-in, also unchanged by this test, so the model was evaluated with "no
motion" input regardless of what `vkcube`'s own rotating cube was actually doing).
**Genuinely closed by this test**: whether this project's own from-scratch Rust NGX
integration can produce a real, successful, repeated model evaluation at all, on the
first machine that's ever had a legitimate DLL to test it against. That question is now
answered yes.

## First confirmed *correct visual output*, plus three real bugs found and fixed along
## the way, plus one important false alarm (2026-09-10, `lordnikon`)

A real dump of `EvaluateFeature`'s actual answer (via the new `ShmHeader::capture_request`
support, see `composition::apply`'s section below) initially showed a solid white
image with no visible structure at all -- alarming, since the success codes above only
prove the API call succeeded, not that the pixels it produced are meaningful. Chasing
that down the wrong way first, then the right way, is worth recording in full:

**Three real bugs found and fixed, via `strings` against the real `nvngx_dlssnr.dll`
(never upstream's source -- same "shape not expression" rule as everywhere else in this
project) and plain reasoning about temporal state, none of which turned out to be the
actual cause of the white image, but all three genuine, independently-justified
correctness fixes worth keeping regardless**:
1. **`DLSSNR.Depth`/`DLSSNR.DepthInverted` were never bound.** `strings` on the real DLL
   turns up `DLSSNR: EvaluateFeature Color=%p MVec=%p Depth=%p Output=%p ...` -- a real,
   confirmed-present fourth resource this crate never supplied. Fixed
   (`crates/helper/src/frame.rs`) with a synthetic, constant-"far-plane" depth image
   (no real depth buffer is captured by `dlssnr_layer::capture` yet, same honest
   stand-in status as MVec's all-zero motion), `DepthInverted = 0` (standard,
   non-reversed convention). Also added every resource's `*SubrectBaseX/Y/Width/Height`
   scalar (also confirmed present via `strings`), set to the full frame -- previously
   never set at all.
2. **`DLSSNR.Reset` was set to `1` once at feature creation and never touched again.**
   Every single `EvaluateFeature` call for the life of the feature was therefore
   telling the model "no valid history, this is frame one" -- plausible, on its own,
   for a temporal model to have a degenerate or placeholder first-frame output. Fixed
   (`frame.rs`'s new `reset_done: Cell<bool>`): `1` only on the feature's real first
   `evaluate` call, `0` on every one after.
3. **The device's `WANTED_DEVICE_EXTENSIONS` list (`main.rs`) was copied from a native
   Linux reference binary's own `strings` output without adjusting for platform.**
   `VK_EXT_external_memory_dma_buf`/`VK_KHR_external_memory_fd` are POSIX-specific and
   could never be exposed by a Windows Vulkan device even under perfect Wine
   emulation -- `dlssnr_helper.exe` is a Windows binary (this project's current,
   documented, interim architecture), so the correct ask is
   `VK_KHR_external_memory`/`VK_KHR_external_memory_win32`. Confirmed real: device
   extension count went from 6/9 to 8/9 once corrected, with `VK_KHR_external_memory_win32`
   actually present and enabled. Plausible relevance: the real DLL performs genuine
   CUDA-Vulkan interop internally (`cuSurfObjectGetResourceDesc` and similar, confirmed
   via the same `strings` pass), which is exactly the kind of operation a missing
   external-memory handle type would degrade.

**None of the three fixed it. The actual cause was a bug in this session's own new
diagnostic tool, not the pipeline**: `crates/layer/src/dump.rs`'s PNG writer passed the
`Output`/answer bytes through with whatever alpha channel they actually had -- which
turned out to be `0` (fully transparent) across the entire image, while the RGB channels
held real, structured, non-uniform data the whole time (confirmed by inspecting actual
pixel values, not just the rendered PNG: real varied colors like `(35,35,35)`/`(1,35,41)`,
not `(255,255,255)`). A PNG viewer renders zero-alpha content as blank/white against its
own background, which looks exactly like -- and was genuinely mistaken here for -- a
broken, degenerate model answer. **A real opaque-composite-mode swapchain present
(what every normal game uses, including `vkcube`) never reads the alpha channel at
all**, so this was never a bug that could have affected an actual displayed frame --
purely an artifact of how the new debug-dump tool chose to write its output file. Fixed
by forcing alpha to `255` before encoding (`write_png`'s own doc comment explains why).

**With that fixed, real visual confirmation**: the composited output is visually
near-identical to the original captured frame for this specific test scene (a plain,
correctly-exposed SDR rotating cube, no blown highlights) -- which is the *correct*
expected result for `upgrade_tone_map`'s own algorithm on content like this, not a sign
of nothing happening. Confirmed the model is doing real, non-trivial, structured work
regardless: a pixel diff between the original and composited PNGs shows a mean
per-channel difference of ~17/255 (max 65/255) across **100% of sampled pixels** --
substantial, structured change everywhere, not rounding noise and not a no-op. This is
the first time this project has visually confirmed its real, end-to-end,
capture-to-composition pipeline produces correct, coherent, non-degenerate output
against a real model on real hardware, not just "the API calls returned success."

**What this still doesn't prove**: whether the result is *better* than the original in
any measurable way (no reference/ground-truth comparison exists to score against), and
everything the "First confirmed neural-rendering success" section above already listed
as unaffected (real optical flow, a scene with actual blown highlights to exercise the
headroom-recovery branch) is still unaffected by this pass either.

## Composition math now actually reaches the frame (2026-09-10) -- CPU path, real
## measured performance cost, real measured fix

**`crates/layer/src/composition/apply.rs` (new) wires `color.rs`'s already-tested
`upgrade_tone_map`/`gamut_compress_reversible` into `capture.rs`'s real write-back**,
on the CPU: a direct, mechanical port of `shaders/compose.comp`'s own `main()`
orchestration (`UpgradeToneMap` -> the transfer-ratio blend ->
`GamutCompressReversible`), reading `colour_strength`/`transfer_strength`/`max_ratio`/
`debug_view`/`apply_model` fresh from the live SHM header every frame
(`ShmClient::composition_settings`, new). This is the thing the "composition" section
below has said was still open since milestone 4 phase A/B — as of this change, it no
longer is, for the `RGBA8` proxy format (the default; `RGBA16F` still passes the raw
answer through untouched, undocumented as a gap no longer, see `apply.rs`'s own doc
comment). `debug_view`'s four modes (0 composited, 1 original/proxy, 2 raw answer, 3
amplified diff) and `apply_model`'s off-switch are both real and wired, not just
modeled in the protocol.

**Real, measured performance cost, and a real, measured fix**: the naive
single-threaded version of this loop took ~800ms/frame at 1920x1080 on real hardware
(`lordnikon`) -- confirmed via real `vkcube` runs, not estimated: a real 10-second run
dropped from 244 captured/answered frames (no composition, the crash-fix verification
run) to 11-12 frames (composition, single-threaded, release build barely different
from debug -- confirming the cost is the transcendental `powf`/`cbrt` calls themselves,
several deep per pixel between two sRGB decodes and two OkLab conversions, not
un-optimized surrounding Rust). Every pixel's work is independent (reads its own
`original`/`answer` bytes, writes its own `answer` bytes, no cross-pixel state), so
`apply_rgba8` now splits the frame across `std::thread::available_parallelism()`
threads (capped at 16) via `std::thread::scope` -- confirmed via the same real
`vkcube` test: **97 frames in the same 10 seconds on this 16-core machine, an ~8x real
improvement**, not just a theoretical one.

**What this does and does not mean for real playability**: 97 frames/10s (~10fps) is
real and usable for continued testing, but still well short of the 244-frame (~24fps)
no-composition baseline on the same hardware -- this CPU path is what makes the
algorithm correct and provable now, not the final performance story. The
already-written, still-not-dispatched `shaders/compose.comp` (real GPU compute, not a
CPU loop across `powf` calls) is what closes that remaining gap; wiring it in is real,
scoped, open work, not done by this change. A second real optimization the current
code leaves on the table: `std::thread::scope` spawns fresh OS threads every single
frame rather than reusing a persistent pool -- a real, measurable cost this project
hasn't measured in isolation from the math itself yet.

**Verified**: `cargo test -p dlssnr-layer` covers `apply_rgba8`'s real invariants
(`debug_view` 1/2 short-circuit correctly, `transfer_strength=0` reproduces the
original within 8-bit rounding, `model == original` is near-identity, sRGB round-trips
every byte value) -- 5 new tests, all passing, plus the full existing suite (30 tests
workspace-wide) still green. Confirmed on real hardware: the round trip still succeeds
every frame with composition active (`round trip answered=true` throughout both the
single- and multi-threaded real runs above) -- this is a real behavior change to what
reaches the screen, not just new code that compiles.

## `shaders/compose.comp` is now really dispatched on the GPU (2026-09-10) -- a real
## shader bug found by a local test before it ever reached real hardware

**`crates/layer/src/composition/gpu.rs` (new)**: real Vulkan compute dispatch of
`shaders/compose.comp`, tried first in `capture.rs`'s write-back whenever
`debug_view == 0` (the shader has no concept of the other three debug modes at all;
those still fall back to [`apply.rs`](#composition-math-now-actually-reaches-the-frame-2026-09-10----cpu-path-real)'s
CPU path, which every mode already handles, same fail-open discipline as everywhere
else in this crate). `compose.comp` itself changed from `rgba16f` to `rgba8` storage
images with explicit `SrgbDecode`/`SrgbEncode` GLSL functions added -- storage-image
`imageLoad`/`imageStore` never apply an sRGB curve regardless of declared format (that
is exclusively a sampled-image-plus-sampler feature per the Vulkan spec), and the real,
only-currently-supported proxy format is `RGBA8`, not the float format the shader
originally assumed. Precompiled to SPIR-V with `glslangValidator -V` (validated with
`spirv-val`) and committed as `shaders/compose.spv`, embedded into the binary via
`include_bytes!` -- no shader-compiler dependency needed at build time, only when the
`.comp` source itself changes.

**A real shader bug was found and fixed before this ever touched real hardware**, by a
new local test (`composition::gpu::tests::gpu_dispatch_matches_the_cpu_reference`) that
needs nothing but a software Vulkan ICD (lavapipe, already relied on elsewhere in this
crate's own tests) — it dispatches the real shader and compares its output pixel-for-
pixel against [`apply.rs`](#composition-math-now-actually-reaches-the-frame-2026-09-10----cpu-path-real)'s
already-real-hardware-verified CPU reference. First run: diverged by up to 90/255 with
`colour_strength > 0` (fine at `colour_strength = 0`, isolating the bug to the OkLab
hue-correction path). Root cause: `OklabFromLinearSrgb`'s
`return OKLAB_LMS_TO_OKLAB * sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));` -- GLSL
evaluates same-precedence operators left to right, so this multiplies the matrix by
`sign(lms)` *first* (a `mat3 * vec3` producing some vector), then does a component-wise
`vec3 * vec3` against `pow(abs(lms), 1/3)` -- nothing like "matrix-multiply the
already-cube-rooted vector," which is what the Rust reference (and this same file's own
`LinearSrgbFromOklab`, which parenthesizes `(lms * lms * lms)` correctly) actually does.
Fixed by computing the cube root as its own complete vector first, then multiplying by
the matrix. **This would have shipped a visually wrong GPU composite for the default
`colour_strength = 1.0` setting** had the local test not caught it — exactly the kind
of bug a real-hardware-only testing strategy could easily have missed for a while (the
image still looks like *something*, not obviously broken, at a casual glance) and
exactly why this test was worth writing before trusting the port at all.

**Verified correct on real hardware after the fix**: a real `capture_request` dump
(same tool as [the section above](#first-confirmed-correct-visual-output-plus-three-real-bugs-found-and-fixed-along-the-way-plus-one-important-false-alarm-2026-09-10-lordnikon))
shows the same real, substantial, structured composition effect the CPU path already
proved (mean per-channel diff ~17/255 from the original, across the whole frame) — the
GPU path produces the same real answer, not just "doesn't crash."

**Real, measured, honest performance finding**: dispatching on the GPU (RTX 5070) is
faster than the multi-threaded CPU path but not by nearly as much as raw compute
throughput alone would suggest -- 118 frames in a real 10-second `vkcube` run, vs. 97
for the CPU path and 244 for no composition at all. The shader itself almost certainly
runs in microseconds on real hardware; the modest gain points at the *synchronous,
one-submit-one-wait-per-frame* dispatch pattern (`GpuCompose::dispatch` blocks on a
fence every single call, the same discipline `capture.rs`'s own two transfer stages
already use) as the real remaining cost -- three separate CPU-GPU round trips per
frame at this point (capture's own two stages plus this one), each paying real
kernel/driver synchronization overhead regardless of how fast the GPU work inside it
is.

**Verified**: the 2 new `#[cfg(test)]`s above (`gpu_dispatch_matches_the_cpu_reference`,
`gpu_dispatch_handles_a_resize` -- the latter confirms `ensure_sized`'s rebuild-on-
resize path, not just its happy path, works against a real device too) plus the full
existing suite (27 tests in this crate alone) still green. Both skip gracefully
(logging why, not failing) in an environment with no Vulkan loader/ICD at all, rather
than breaking a build that has no way to run them.

## Merged GPU compose + write-back into one submission (2026-09-10) -- real but
## smaller-than-hoped further gain, and what it reveals about the *actual* remaining cost

**`GpuCompose::dispatch_into_image` (new)**, tried first in `capture.rs`'s write-back
whenever nothing on the CPU needs to see the composited bytes afterward (i.e. no
`capture_request` dump is pending -- checked via a new non-consuming
`ShmClient::capture_request_pending`, since [`take_capture_request`](#first-confirmed-correct-visual-output-plus-three-real-bugs-found-and-fixed-along-the-way-plus-one-important-false-alarm-2026-09-10-lordnikon)
would wrongly consume a real request just to decide routing). Instead of downloading
the compute shader's answer to a CPU slice and having `capture.rs` upload it again in
a separate stage-2 submission, this writes the result straight into the real swapchain
image, in the *same* command buffer as the compute dispatch itself -- two GPU
submissions per frame instead of three, and no CPU round trip for the composited bytes
at all in the common case. **Deliberately still buffer-mediated, not a raw
`vkCmdCopyImage` from the compute output image straight into the swapchain image**:
that would be a byte-for-byte copy with no channel-swizzle, silently corrupting colors
the moment a real swapchain's format differs from this module's own hardcoded
`R8G8B8A8_UNORM` (e.g. a common `B8G8R8A8` swapchain) -- something this project has no
practical way to vary and test across in this environment. A buffer has no format
attached at all, so copying through one and letting the final `vkCmdCopyBufferToImage`
target the real image's own true format (exactly what `capture.rs`'s existing stage 2
already relied on) is correct regardless of what that real format turns out to be.
Verified byte-for-byte identical to the already-verified `dispatch` path by a new real
local test (`dispatch_into_image_matches_dispatch`) before ever touching real hardware,
same "test on lavapipe first" discipline as the shader-bug fix above.

**Verified correct and measured on real hardware**: every frame in a real `vkcube` run
took this fast path (`composed_directly=true` in the layer's own log) except the one
frame a real `capture_request` was pending for, which correctly fell back to the
slower, CPU-visible path -- confirmed via a real dump, still visually correct. Real
performance gain: **128 frames in the same 10-second run, up from 118** -- a real,
modest ~8% further improvement, smaller than the reduced submission count alone might
suggest.

**What this reveals**: the no-composition baseline (244 frames/10s) *also* does exactly
two GPU submissions per frame (capture's own stage 1 and stage 2 always run, whether or
not anything modifies the bytes in between) -- so submission *count* was never actually
the dominant remaining difference once this change made the composed path's count match
it. The real, larger remaining gap (244 vs. 128) is much more likely genuine GPU
bandwidth/work-volume: the composed path moves several times as many bytes through
VRAM/PCIe per frame (three extra images uploaded/downloaded around the compute
dispatch, on top of the capture round trip every path already pays) and runs a real
compute dispatch on top, not just synchronization overhead a smarter submission
strategy could hide.

## Cross-frame async pipelining (2026-09-10) -- explicitly authorized, real measured
## gain, and why it needed real thought about semaphore safety, not just "don't block"

The section above ended by flagging genuine cross-frame pipelining (submit frame N's
GPU work without blocking, only wait on frame N-1's before reusing its resources) as a
real architecture change trading in a frame of added latency -- a product decision, not
something to make unilaterally. Alex's explicit answer: **do it, if it gives the most
frames when NR is on** -- authorization for exactly that tradeoff, acted on immediately.

**`GpuCompose::dispatch_into_image_async` (new)**: the same write-straight-into-the-
swapchain-image trick as `dispatch_into_image`, but instead of blocking on this
dispatch's own fence, it submits with a **signal semaphore** and returns immediately.
`capture::run` now returns `Option<vk::Semaphore>` instead of `bool`, and
`device.rs`'s `queue_present_khr` chains that semaphore into the *real*
`vkQueuePresentKHR` call's own wait-semaphore list (combined with whatever the app
itself already provided, via `vk::PresentInfoKHR { wait_semaphore_count, p_wait_semaphores, ..*present_info }`
-- every other field copied through unchanged). This makes the correctness dependency
a GPU-side one (the presentation engine won't display the image until the compute work
signals), not a CPU-side block -- the CPU returns from the present hook and moves on to
the *next* frame's own capture + SHM round trip while this frame's compute work is
potentially still running on the GPU.

**This is real, correctness-critical synchronization, not just "remove the wait
call"**, and needed to be gotten right on the first real attempt, with no interactive
supervision to catch a subtle mistake:
- **Why not just skip the fence wait on the existing single-buffered `GpuCompose`
  state**: doing that alone would let a *second* dispatch call reset/rewrite a command
  buffer and staging memory a *first*, still-in-flight dispatch might still be reading
  from -- a real data race. Fixed with genuine double buffering: `ASYNC_SLOTS = 2`
  fully independent `AsyncSlot`s (own images, staging buffer, command buffer, fence,
  semaphore), alternating each call. The only wait this method makes is on a slot's
  *own* fence from its *previous* use (`ASYNC_SLOTS` dispatches back) -- immediately
  before touching that slot's resources again, never before returning the current
  call's own result. By the time a slot comes back around, an entire other frame's
  worth of capture + SHM round trip has elapsed on the CPU, so that wait is normally
  instant.
- **Binary semaphore reuse safety**: a signaled-but-not-yet-waited-on binary semaphore
  must never be signaled again (undefined behavior per the Vulkan spec if it is). Each
  slot's semaphore is only ever signaled by this method for that slot, and the very
  next thing that happens after it returns `Some(sem)` is `device.rs` unconditionally
  chaining `sem` into the real present call, every single frame (`queue_present_khr`'s
  own structure guarantees this) -- so a wait for it is always enqueued long before the
  same slot (and therefore the same semaphore) could ever be signaled again. Verified
  in practice, not just reasoned through: a new local test drives
  `dispatch_into_image_async` across `ASYNC_SLOTS * 3 + 1` iterations (real slot reuse,
  several times over), waiting on each returned semaphore exactly the way real code
  does, against a real device.
- **Deliberately did *not* attempt reprojection/stale-answer tricks** (using an older
  frame's neural answer composited onto a *newer* frame's own captured content, the way
  real temporal upscalers hide latency) to get an even bigger win: without real motion
  vectors (`MVec` is still the all-zero placeholder), that would produce real, visible
  ghosting/smearing on any moving content -- a genuine visual-quality regression, not
  just a synchronization detail. What's implemented here keeps every frame's presented
  image built from *that same frame's* own capture and *that same frame's* own model
  answer -- only *when* the CPU learns the GPU work is complete changed, never *which*
  data ends up on screen.

**Verified thoroughly on real hardware before trusting it**: no local test can exercise
the real present-call injection (needs a real swapchain), so this went straight to
`lordnikon` carefully -- a short 5s run first (checking specifically for hangs/crashes,
the real risk profile of getting Vulkan semaphore sync wrong), then a real 10s
measurement, then a real `capture_request` dump to confirm visual correctness, then a
45-second/601-frame stress run specifically to rule out a slot-reuse issue that might
only surface after many more cycles than a short run exercises. All clean: zero
crashes, zero hangs, zero fallbacks to the synchronous path (every single frame took
the async one except when a real `capture_request` was pending, which correctly used
the slower, synchronous, CPU-visible path instead), visually correct output.

**Real, measured, honest performance result**: **143 frames in a real 10-second
`vkcube` run, up from 128** (and confirmed consistent at ~134/10s pace over the full
45-second stress run) -- a further real gain, smaller than the jump from the CPU path
to GPU compute, confirming what the previous section's finding already predicted: most
of the remaining gap to the 244-frame no-composition baseline is genuine GPU
bandwidth/work-volume (more bytes moved, real compute time), which no amount of
smarter CPU-side scheduling removes. This is very likely close to the practical ceiling
for this architecture (single compute dispatch, buffer-mediated write-back, no real
optical flow yet) without a more fundamental change to how much data crosses the
capture/compose pipeline per frame -- a different, larger undertaking, not a
synchronization tweak.

**Verified**: 3 new `#[cfg(test)]`s (`dispatch_into_image_async_matches_dispatch_into_image`,
`dispatch_into_image_async_survives_many_slot_reuses`, plus the existing
`dispatch_into_image_matches_dispatch` still passing against the now-refactored
`ComposeSlot`-based internals) — 30 tests in this crate now, full workspace suite green.

## First real-game session, and a real production crash that everyone's own testing
## had been silently working around for a while (2026-09-10, `lordnikon`)

Alex ran a real, actual game (GTA San Andreas -- The Definitive Edition) via the real
`dlssnr.appimage` GUI for the first time this project has been tested against
something other than `vkcube`, and hit two real problems -- one a genuine bug in this
project's own code, one a pre-existing system configuration conflict, unrelated to
anything shipped here.

**Real bug, now fixed: the helper crashed on every single real start attempt.**
`dlssnr_supervisor::start()` (the function both `dlssnr-gui`'s Start button and
`dlssnr-cli start` call) never set `STEAM_COMPAT_CLIENT_INSTALL_PATH` -- Proton's own
launch script reads it directly out of the environment with no fallback
(`os.environ["STEAM_COMPAT_CLIENT_INSTALL_PATH"]`, a bare `KeyError` if unset) during
its own prefix setup, *before* it ever gets to running `dlssnr_helper.exe` at all. The
real irony: every one of this project's own manual SSH test sessions on `lordnikon`
(the ones that produced the "first confirmed neural-rendering success" and every
composition/GPU-dispatch verification since) set this exact variable by hand, every
single time, specifically because it's needed -- but that fix never made it back into
`start()` itself, so the *actual* production code path a real user hits was broken the
whole time this project's own testing kept working around it live, unnoticed until a
real user hit it on a real game. **Fixed** by adding `dlssnr_supervisor::paths::steam_install_dir`
(checks the same native/Flatpak/Snap Steam-root candidates `dlssnr-cli`'s own Proton
discovery already scans for `compatibilitytools.d`, returns the first real directory)
and wiring it into `start()`. 2 new tests (a real filesystem + env var override,
confirming both the found and not-found cases). A concrete lesson for this project's
own process, not just this one bug: a fix applied only in a throwaway test harness,
never the real code path, is not actually fixed.

**Not a bug in this project, found and reported, deliberately not touched without
asking**: `~/.config/environment.d/dlssnr.conf`, a systemd user-environment.d file
(almost certainly left behind by *upstream's own* installer at some point, given the
filename and that this project creates no such file anywhere) sets
`VK_INSTANCE_LAYERS="VK_LAYER_NV_dlssnr:VK_LAYER_NV_present"` -- globally, for every
Vulkan application in the whole graphical session, confirmed via
`systemctl --user show-environment` and by reading `/proc/<pid>/environ` for both the
real `dlssnr-gui` process and the real game process, both showing it. This explicitly,
unconditionally force-activates *upstream's* real layer for literally everything,
including a game whose own launch options only ever set `VKLayer_DLSS5=1` (this
project's own implicit-activation variable) with no
`VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_dlssnr` to counteract the global override --
unlike every one of this project's own `vkcube` tests, which always explicitly
disabled upstream's layer first. Plausible real consequences: this project's own
layer showing as "not attached" in its own GUI (upstream's real layer, not this
one, is what's actually active for real games on this machine), and -- if both
layers end up in the same chain simultaneously -- two entirely separate real neural
rendering passes running per frame, a real, substantial performance cost with an
entirely mundane explanation, not a bug in anything shipped here. This is the user's
own session-wide configuration (probably a leftover from originally setting up
upstream's own product on this machine), not this project's file to edit
unilaterally -- flagged for a real, deliberate decision (edit/remove the global
override, or just add the same per-game `VK_LOADER_LAYERS_DISABLE` this project's own
testing always uses to the affected game's own Steam launch options) rather than
touched on its own judgment.

## `composition` (milestone 4, phase A/B landed 2026-09-09 on `lordnikon`, real GPU —
## capture/transport/NGX-evaluate genuinely run every frame, and as of 2026-09-10 the
## helper's answer actually reaches the write-back too, not yet verified against a
## real present cycle; this project's own composition math still never gets applied.
## Reviewed and this section brought back in sync with the actual code on 2026-09-10.)

**What changed since the "GPU pipeline not wired up" note this section used to open
with**: that's no longer accurate. `queue_present_khr` now really does capture the
presented image, round-trip it through the helper, and the helper now really does call
NGX's `EvaluateFeature` against real bound Vulkan resources — all four commits
(`afbb408`, `16bf02e`, `072ea07`, `13de5cd`, 2026-09-09 17:10–19:57) came from a
session working directly against real hardware (RTX 5070, driver 615.71.09, machine
`lordnikon`), diagnosing real failures with real tools (`gdb` against a hung driver
call, `objdump`/`strings` against both the real `nvngx_dlssnr.dll` and a working
reference implementation's own compiled helper — binary inspection only, never
source, same "shape not expression" rule as everywhere else in this project). No new
unit tests came with this — none of it is meaningfully unit-testable without a real
GPU + a real, legitimately-signed NGX DLL (which this project still doesn't have, see
below) — verification here is real execution and log/gdb output, not `#[test]`.

- **`crates/layer/src/capture.rs`** (new, 439 lines): builds a per-device command
  pool/fence/host-visible staging buffer, and on `queue_present_khr` really does
  transition the about-to-be-presented image, copy it into the staging buffer, hand
  those bytes to `ShmClient::write_proxy`, round-trip through the helper, and copy
  something back into the image before the real present call. Fails open at every
  step (any Vulkan call failing just skips capture for that frame, presenting
  unmodified — never a reason to stop trying later frames).
- **Fixed 2026-09-10: the write-back now actually uses the helper's answer.** Until
  then, stage 2 of `capture::run` always copied `r.buffer` — which stage 1 filled with
  the *captured* bytes and nothing since had overwritten — back into the image; the
  real answer came back too (`shm.read_answer`) but only into a 16-byte diagnostic
  probe that got logged and discarded. Now, when the round trip answers, `r.ptr` (the
  same host-coherent memory `r.buffer` is bound to) gets overwritten in place with the
  full answer before stage 2's copy runs; a helper that never answers still leaves
  `r.ptr` holding the just-captured bytes, so the existing fail-open behavior is
  unchanged. **Still not verified end to end against a real present cycle** — this
  sandbox has no real display/swapchain to drive `queue_present_khr` through (the
  existing smoke test only creates a bare device, never a swapchain), and the other
  session's real-hardware testing on `lordnikon` predates this specific change.
  Reasoned through carefully (the SAFETY comments spell out exactly why re-reading
  `r.ptr`/`r.buffer` after the CPU-side overwrite is sound) and the full test suite
  stays green, but the next real verification of this path should happen on
  `lordnikon` against an actual game, not just asserted correct from here. Once this
  is confirmed working, `compose.comp`'s blend is the next, still fully separate,
  still-unstarted step — applying the model's raw answer directly (what this fix does)
  and blending it via this project's own composition math are two different things.
- **`crates/helper/src/frame.rs`** (new, 516 lines) + `ngx.rs` changes: real
  Color/Output/MVec Vulkan images, a real upload → `EvaluateFeature` → download
  sequence, wired into `main.rs`'s per-frame loop (watches `seq_req`, calls
  `ngx::ensure_feature` once a real size is known, evaluates, writes an answer back —
  echoing the proxy straight through on any failure so the transport still completes).
  MVec is always an all-zero image (no real `VK_NV_optical_flow` yet, an intentional,
  documented stand-in for "no motion", not a bug). The `DLSSNR.Color`/`.Output`/
  `.MVec` parameter names binding these images are **guessed** "for shape" the same
  way every other `DLSSNR.*` scalar parameter name here already was — no spec exists
  for a fictional feature's resource bindings either; a wrong guess fails via the SEH
  guard, not a crash.
- **Three real, hardware-diagnosed fixes landed alongside the above, each worth
  knowing about on its own:**
  - `v0.1.2` (`afbb408`/`16bf02e`): `create_feature_at()`'s parameter-setting calls
    were outside `guarded()`, the only DLL-touching block in `ngx.rs` that was — a
    real fault there silently killed the whole helper a couple seconds after a
    successful `VULKAN_Init_Ext`, no log line, no crash dialog. Now guarded like
    everything else.
  - `CreateFeature(18)` at a real size was hanging **indefinitely inside the NVIDIA
    driver itself** (`libnvidia-glcore.so`, confirmed with `gdb`), traced to passing
    `vk::CommandBuffer::null()` — the real API expects a live, currently-recording
    command buffer it records GPU-side setup work into, which the caller then
    ends/submits/fence-waits. Fixed in `create_feature_at` with a real
    pool/buffer/begin/end/submit/fence-wait around the call.
  - The helper's Vulkan device previously enabled **zero** device extensions.
    Comparing against a real, working reference implementation's own compiled helper
    (`strings`/`objdump` on the binary, confirmed present via
    `vkEnumerateDeviceExtensionProperties` before requesting) turned up a
    `WANTED_DEVICE_EXTENSIONS` list (`VK_EXT_external_memory_dma_buf`,
    `VK_KHR_buffer_device_address`, `VK_NVX_binary_import`, `VK_NVX_image_view_handle`,
    `VK_NV_optical_flow`, others) now requested when actually available — the leading
    suspect for why `CreateFeature` behaved inconsistently even after every other fix.
  - Also added, still experimental/diagnostic-only (logged, not gated on):
    `NVSDK_NGX_VULKAN_GetFeatureRequirements` (a real export nothing here had ever
    called before `CreateFeature`, which real NGX integrations call first) and a
    parameter round-trip self-test (`DLSSNR.SelfTestProbe`, set then read back through
    the same vtable `create_feature_at` uses, purely to confirm the plumbing works).
  - Self-correction worth knowing about: an earlier pass of this same work added
    `DLSSNR.Output.Width`/`.Height` parameters based on a mismatched third-party
    reference; confirmed absent from the real DLL's own string table and removed
    before landing.
- **`crates/layer/src/composition/color.rs` and `downscale.rs` are still real, tested,
  pure Rust and unchanged by any of this** (15 + 9 `#[test]`s — see git history for
  what they cover). **`crates/layer/shaders/compose.comp` is still never compiled,
  dispatched, or checked against that Rust reference, and nothing calls it.** This
  milestone's phase A/B was about proving the capture/transport/NGX-evaluate pipeline
  end to end, not about wiring in this project's own composition math — that's still
  entirely separate, still-open work, unaffected by anything above.
- **Not wired into device teardown, and checked (2026-09-10) that this isn't a quick
  fix**: `capture::destroy` exists (frees the command pool/staging buffer/memory) but
  nothing calls it. Looked into wiring it in properly this session: the pinned
  `vulkan-layer` commit's `DeviceHooks` trait has **no interceptable `destroy_device`
  command at all** (confirmed by reading its generated trait definition), and its
  framework-level `Global::destroy_device` calls the *real* `vkDestroyDevice` on the
  next layer/driver **before** dropping our `DeviceInfoContainer` -- so a naive
  `impl Drop for DlssnrDeviceInfo` that called `capture::destroy` there would be
  issuing Vulkan calls (`vkDestroyFence`/`vkFreeMemory`/etc.) against an
  **already-destroyed** `VkDevice`, which is genuine undefined behavior, not a fix.
  There is no safe hook point in this pinned crate version to run cleanup before real
  device destruction; doing this properly would mean patching/forking the git
  dependency to add one, a real but separate undertaking. Real but minor as a leak (the
  OS/driver reclaims GPU resources on process exit regardless) — left as a genuine,
  now-better-understood gap rather than a quick fix that would trade a harmless leak
  for real UB.
- **Still no legitimate `nvngx_dlssnr.dll` on hand anywhere** (see the earlier session
  transcript: the copies found were either the wrong model entirely or a
  signature-invalid, hash-mismatched file from an unofficial source, declined for use)
  — so none of the above has ever been confirmed against a real, working model
  evaluation; `CreateFeature`'s return code on `lordnikon` is still a rejection, just
  now a fast, clean one instead of a driver-level hang. That is real, measurable
  progress (a hang is strictly worse than a clean rejection), not proof the integration
  is fully correct.

## `gui` (milestone 5, real — settings read/render correctly; write-back visually
## unconfirmed for an environment reason, not a code one)

- **Every settings row in `ui.rs` genuinely binds to a live `dlssnr_protocol::ShmHeader`
  field** via `shm.rs`'s `bind_float`/`bind_bool`/`bind_u32` (thin wrappers around the
  exact same `AtomicU32::load`/`store` + `control_seq` bump already unit-tested in
  `dlssnr_protocol`). Confirmed by screenshot (see the session transcript) that every
  group (Model, Motion, Composition, Status) renders and every value shown matches
  `ShmHeader::init_defaults()`'s real defaults exactly — this is reading the live
  mapping, not a static mock.
- **Could not confirm the write-back path by actually clicking a widget.** `xdotool`
  clicks (window-relative *and* absolute-screen-coordinate, on the correct window ID)
  did not register on this window at all in this dev sandbox — confirmed by testing
  against the title bar's own close button, which also failed to close the window.
  That isolates it to this sandbox's X11/input-routing setup (matching the
  `_NET_WM_DESKTOP` warning `xdotool` printed), not a bug in the UI: the write path is
  the same trivial `AtomicU32::store` the read path already proved works, wired through
  GTK's own standard `connect_active_notify`/`connect_selected_notify`/
  `connect_value_changed` signal callbacks. Worth an actual click-test outside this
  sandbox before fully trusting it, but there is no specific reason to expect it's wrong.
- **`dlssnr_protocol::mapping`** (Linux-only, `cfg(unix)`) is the "just open the
  mapping and read/write settings" utility both `gui` and `cli` share, deliberately
  kept separate from `dlssnr_layer::shm::ShmClient` (which is entangled with the
  request/response round-trip state machine the GUI/CLI have no reason to depend on).
- **NGX binaries import row, added later (2026-09-09)**: the Status group's "NGX
  binaries" row has a real "Import…" button — this was missing when README.md first
  claimed it existed (a documentation bug, caught when Alex went looking for it in the
  running app and couldn't find it). `crates/gui/src/binaries.rs` holds the path
  (`XDG_DATA_HOME/dlssnr/binaries`, duplicated from `cli/src/paths.rs::binaries_dir`
  rather than shared — four lines, not worth a shared crate) and the copy logic, real-
  tested by `cargo test -p dlssnr-gui` (`import_from_copies_known_files_and_skips_unknown_ones`,
  confirms known DLLs are copied and unrelated files are not). The button opens a
  `gtk4::FileDialog::select_folder`, copies via that same function, and shows an
  `adw::Toast` with the result. **Same sandbox limitation as the write-back path
  above blocks confirming the actual click-through**: verified instead by temporarily
  reducing `build_ui` to just the Status group so it would render without needing the
  scroll this sandbox also can't inject, screenshotting it (row and subtitle render
  correctly, "nvngx_dlssnr.dll missing" reflecting the real absence of the file), then
  reverting that temporary reduction — the underlying `import_from` logic is what the
  test above actually exercises.

## Compared against a real, installed upstream instance (2026-09-10, on `lordnikon`)

Alex has upstream DLSS5VKLayer's real package (`dlssnr` 0.2.6-1, dpkg) installed on
another machine (`lordnikon`, RTX 5070, driver 615.71.09) with real Proton/Wine
runners and a real prior helper log to compare against. Used this to find and fix one
real gap in the port, without ever touching upstream's source (its installed
binaries/config/logs were read as *behavior* to compare against, same "shape not
expression" rule as everywhere else in this project — nothing here was learned by
reading upstream's C++).

**What was actually wrong, found and fixed**: our GUI's settings only ever lived in
the SHM mapping (`/tmp/dlssnr-$UID/shm.bin`), which does not survive a reboot —
`dlssnr_protocol::mapping::open_at` always calls `init_defaults()` on any mapping that
isn't already valid, with no path to restore prior tuning. Upstream's real,
installed `~/.config/dlssnr/config.ini` on lordnikon has every tunable persisted as
`set_<name>=<value>` lines and clearly reloads them (the file had `set_intensity=1`,
`set_passes=1`, etc. sitting there from a session that ended, presumably, well before
this one started). Fixed by adding `ShmHeader::persisted_settings`/
`apply_persisted_setting` (`crates/protocol/src/header.rs`) and a
`dlssnr_protocol::persist` module that round-trips those through the plain
`BTreeMap<String,String>` a config file already gets parsed into — `dlssnr-supervisor`'s
`Config` gained a `settings` field carrying whatever `set_*` lines it doesn't
otherwise recognize, and `gui/src/shm.rs`'s `bind_float`/`bind_u32`/`bind_bool` now
take a persistence key name and call through to it on every change.
`Mapping::freshly_created` is new too (`protocol/src/mapping.rs`) — `Shm::open` only
applies persisted settings on the call that actually created the mapping, not a warm
reattach to whatever a currently-running instance already has live. **Verified for
real, not just by inspection**: `shm::tests::a_setting_changed_through_bind_float_survives_a_simulated_reboot`
drives the actual production functions end to end (`Shm::open` → `bind_float`'s
setter → config.ini → delete the SHM file, simulating a reboot → `Shm::open` again →
confirms the value came back), plus the `persist` module's own round-trip tests in
`dlssnr-protocol`.

**What was checked and turned out fine, not worth changing**:
- The helper's caller-identity spoof already hooks the IAT of *two* separate loaded
  modules (`ngx.rs`'s two `spoof::install()` calls, for the snippet and the core NGX
  module) — confirmed this matches upstream's real helper log exactly, which shows
  "GetModuleFileNameW IAT hooked" twice at two different module base addresses during
  an actual run. Nothing to fix here; good, independent confirmation the port already
  does this right.
- `runners.rs`'s Proton scoring (CachyOS > exact-versioned GE-Proton > "GE-Proton
  Latest" alias > generic) matches the *relative ordering* `dlssnr-runner-probe --json`
  produced for real against lordnikon's actual `compatibilitytools.d` (scores
  10000000 / 9011000 / 9000000 respectively) — the exact score values differ (ours
  weren't designed to match upstream's numbers, just the ranking), and that's fine.
- The SEH guard really does matter in practice, not just in theory: upstream's real
  helper log shows the VEH catching a genuine `0xc0000005` access violation mid-
  `VULKAN_Init_with_ProjectID` and recovering cleanly (synthetic `0x8badf00d` return,
  then "init OK" right after) rather than crashing the whole helper process. Good
  validation that `guard.rs`'s whole reason for existing is a real, observed failure
  mode on real hardware, not a hypothetical one.

**Deliberately not chased further**: upstream's own real run on lordnikon also never
got DLSS5 NR actually working end-to-end (`VULKAN_CreateFeature(18)` returned
`0xbad00002`, `DLSSNR.Available=0`, fail-open kicked in) — this machine's
`nvngx_dlssnr.dll` is a hash-mismatched, signature-invalid file Alex obtained from an
unofficial source (see the earlier session transcript: verified via `osslsigncode`,
explicitly declined to use it for anything). That failure is upstream's own
integration also not working against *that specific file*, not evidence our port's
NGX call sequence is wrong — there's no legitimate model file on hand to actually
prove the happy path end-to-end yet on either implementation.

**Both gaps this section used to flag are fixed now (2026-09-10).**

**The GUI settings surface is much closer to complete**: `ui.rs` gained a new
"Compare and debug" group (`compare_mode`/`compare_split`/`compare_zoom`/
`compare_swap`/`debug_view`) and five new rows in Composition (`colour_mode`,
`transfer`, `unlock_passes`, `apply_model`, `hold_frame`) — real, tested, screenshotted
rows, not just protocol fields with nothing bound to them. This required extending
`ShmHeader::persisted_settings`/`apply_persisted_setting` from 21 to 31 entries first
(the function's own doc comment is explicit that both have to change together, or a
new row would appear to save but silently fail to survive a reboot) — every one of
these now round-trips through `config.ini` exactly like the rows that already existed.
**Still not bound**: `white_point_source`/`white_point_trim`/`white_point_scale` (HDR
white-point tuning, needs an HDR swapchain to be meaningful to test) and `toggle_key`
(a raw Linux key code; a real hotkey-capture widget is a separate, larger UI piece
than a row on an existing group). Verified by real screenshot, not just "compiles" --
caught and fixed one real bug this way: `AdwPreferencesGroup::title` is parsed as
Pango markup, and the first attempt at naming the new group "Compare & debug" broke it
outright (`GTK-WARNING: Failed to set text ... Entity did not end with a semicolon`) —
confirmed only by actually running the GUI and reading its own log, exactly the kind
of thing a type check can't catch. Renamed to "Compare and debug".

**`dlssnr-cli shmctl`** (`crates/cli/src/shmctl.rs`, new) is the real equivalent of
upstream's separate `dlssnr-shmctl` debug/introspection CLI (raw
`status`/`set`/`toggle`/`capture` against the live SHM header), which this port had no
equivalent of before. Covers all 31 `persisted_settings` (now including
`debug_view`/`apply_model`/`compare_mode`/`hold_frame`, moved there from a separate
"extra fields" list once the GUI work above needed them to persist too) plus
`capture_request` (a real one-shot trigger, correctly not a persisted setting), and a
`status` view that also surfaces `helper_state`/`model_up`/`helper_frames`. Same
"attach to the mapping and poke it" mechanism `cmd_config`/the GUI's own settings
binding already use, not new plumbing. 10 new tests on the pure resolve/store/toggle
logic (no real mapping needed for those), full suite still green.

## `supervisor` (added 2026-09-09: extracted from `cli` so the GUI can start/stop too)

`crates/cli/src/{paths,config,install_dir,process}.rs` moved verbatim into a new
`dlssnr-supervisor` lib crate (`git mv`, not rewritten) after Alex noticed the GUI had
no start/stop control at all — only the CLI did. Rather than duplicate the
runner-selection/env-var-construction logic `cmd_start` had (a real risk of drift
between two copies, unlike the four-line NGX-binaries path the GUI already duplicates
on purpose), it's now `dlssnr_supervisor::start(&Config) -> Result<StartedHelper,
StartError>` / `stop(Duration)` / `is_running()`, called identically by both `cli` and
`gui`. `cli`'s `cmd_start`/`cmd_stop` are now thin wrappers that just format
`StartError`'s `Display` output — confirmed byte-for-byte identical CLI output
before/after (`status`, `doctor`, `start`'s `HelperNotFound` error path, `stop`'s
no-helper-running no-op) by running each for real, plus the full test suite
(`cargo test`, workspace-wide) staying green, including
`process.rs`'s `stop_kills_the_whole_process_group_not_just_the_leader` moving over
still correctly `#[ignore]`d for the same sandbox reason documented below. `gui`'s
`binaries.rs` also lost its own duplicated `dir()` in favor of
`dlssnr_supervisor::paths::binaries_dir()`, now that a real shared crate exists for
exactly this.

The GUI's Status group's Helper row now has a Start/Stop button (`ui.rs`), keyed off
`dlssnr_supervisor::is_running()` (the actual pid-file check), not the SHM
`helper_state` the row's subtitle shows — those two can briefly disagree right after a
click. Confirmed rendering correctly via the same reduced-`build_ui`-then-screenshot
trick used for the NGX-import button (real screenshot: row present, labeled "Start",
matching the real "stopped" state) — **actual click-through is unverified**, same
sandbox input-routing limitation as everywhere else in this GUI. `stop()` blocks the
GTK main thread for up to 5s (graceful-then-SIGKILL) on click; deliberately not made
async since this GUI has no async runtime wired up at all (no `tokio`)
and adding one for one button wasn't judged worth it.

## `cli` (milestone 5, real and tested — including one real bug caught by an actual
## process-group kill test)

- **Every subcommand was actually run** (`init`/`config`/`runners`/`detect-gpu`/
  `status`/`doctor`/`import-binaries`), not just compiled — output matches what each
  is supposed to report (correctly fell back to system `wine` with no Proton compat
  tools present, correctly reported no NVIDIA GPU on this dev machine's actual iGPU,
  `doctor` correctly exits non-zero while `nvngx_dlssnr.dll`/the helper exe are
  missing, exactly as it should).
- **`process.rs`'s `stop_kills_the_whole_process_group_not_just_the_leader` test is
  `#[ignore]`d, for a confirmed environment reason, not because the code is wrong.**
  Direct reproduction (see the session transcript) showed `kill()` — direct pid,
  negative-pid process-group, with or without a prior `setsid()`, all report `Ok(())`
  — never actually reaches a child spawned via `std::process::Command` from a
  compiled Rust binary *in this specific dev sandbox*, while an identical signal to a
  plain shell background job (`sleep 30 &` from a bash tool call, no Rust involved)
  **does** get delivered and acted on in the same environment. That isolates the gap to
  this sandbox's handling of signals to `Command`-spawned children specifically — the
  actual `signal_group`/`stop` implementation is the standard, portable POSIX pattern
  and needs no change to work on a real desktop running the real `helper.exe` under
  Wine/Proton. Un-ignore and rerun outside this sandbox before shipping if that's ever
  worth re-confirming.
- **`start`/`stop` are otherwise unexercised against a real helper** (no
  `nvngx_dlssnr.dll` on this machine to start a real session with) — the process
  supervision mechanics are tested per the point above; the actual runner-invocation
  command line (`proton run <helper.exe>` with the right env vars) has not been run
  for real.

## Docs and release (done)

`README.md`/`ATTRIBUTION.md`/`LICENSE` written fresh — not copied from upstream's own
README/ATTRIBUTION.md, which were read (cloned to `/tmp/dlss5vklayer-review` in an
earlier session) only to know what to credit, per the same "shape not expression" rule
as the code. `ATTRIBUTION.md` explicitly separates what's really taken (DLSS5VKLayer's
architecture/protocol shape, RenoDX's MIT-licensed composition design, Ottosson's OkLab
constants) from what's deliberately not taken (the GPL-3.0 OptiScaler/DLSS-NR shader
code upstream's own `ATTRIBUTION.md` admits it carries) and from the one place this
project *does* knowingly reproduce upstream's approach on purpose (the NGX
caller-identity spoof) — that section says so plainly rather than blending it in.
`v0.1.0` tagged and pushed as a GitHub release with the AppImage + `.zsync` sidecar as
assets — repo and release are both public (confirmed via `gh repo view` before
creating it).

## Build process (milestone 6, real and verified end-to-end — this actually produced
## a working AppImage in this session)

`build-appimage.sh` builds the release binaries (native `protocol`/`layer`/`gui`/`cli`,
cross-compiled `helper`), assembles the `AppDir`, packs it with `appimagetool`, and
generates the `.zsync` sidecar with `zsyncmake` run directly (appimagetool's own zsync
generation silently no-ops on CI runners), with **no polkit/pkexec step at all** (this app needs one nowhere) and a `CARGO_HELPER`
environment override for this dev machine's own multi-toolchain setup (see "Current
state" above; a clean CI image needs no override, since it has only one toolchain to
begin with). **Actually run successfully this session**: produced a real, valid
`dlssnr-0.1.0-x86_64.AppImage` (confirmed `file`-typed as a real ELF PIE AppImage
runtime), which was then extracted and run (`--appimage-extract-and-run`) and launched
the real settings GUI with no errors — this is the one part of milestones 4-6 that got
genuinely full, real, end-to-end verification, precisely because it doesn't depend on
anything this dev machine lacks (a real NVIDIA GPU, `nvngx_dlssnr.dll`, working
`xdotool` input routing). The Vulkan layer manifest's `library_path` is correctly
rewritten from the checked-in `./libdlssnr_layer.so` (right for a manifest sitting
beside the `.so`) to the AppImage's real relative layout
(`../../lib/dlssnr/libdlssnr_layer.so`) during packaging — check this rewrite still
matches if the `AppDir` layout ever changes.

## Real-machine deploy gotcha: `~/.local/share/dlssnr/lib/libdlssnr_layer.so` is a
## SEPARATE, manually-maintained copy on `lordnikon` -- not something the app itself
## installs (2026-09-10)

Steam launches the game as a completely separate process tree from `dlssnr-gui`, so
`AppRun`'s `VK_ADD_LAYER_PATH` (scoped to `dlssnr-gui`'s own process tree) never
reaches it. For the game's own Vulkan loader to find this project's layer at all, a
*real* implicit-layer manifest has to exist somewhere the loader (or, on this
machine, Steam Linux Runtime's `pressure-vessel` container, which stages its own
snapshot of host `implicit_layer.d` entries) actually scans on the host --
`~/.local/share/vulkan/implicit_layer.d/dlssnr.json`, `"name": "VK_LAYER_dlssnr_neural"`,
pointing at `~/.local/share/dlssnr/lib/libdlssnr_layer.so`. That manifest and that
`.so` copy were placed by hand this session; nothing in `crates/gui`, `crates/cli`, or
`build-appimage.sh` creates, updates, or even references either path -- confirmed by
grep. **Every time the layer changes, that `.so` has to be copied out by hand** (e.g.
`scp target/release/libdlssnr_layer.so lordnikon:/tmp/... && ssh lordnikon mv /tmp/...
~/.local/share/dlssnr/lib/libdlssnr_layer.so` -- `mv` not `cp`, same "Text file busy"
reasoning as the AppImage itself, since the old `.so` may still be mapped into a
running game). Rebuilding and redeploying the AppImage alone does **not** update this
copy. Missing this step burned real time this session: the whole v0.1.17
buffered-logging fix was built, deployed as an AppImage, and "tested" by restarting
the game twice, with the game silently loading the stale pre-fix `.so` from this
separate path the entire time -- the FPS measurement that (falsely) ruled out logging
as a contributing cause was measuring the *old* code. A real fix belongs here: either
have `dlssnr-cli`/`dlssnr-gui` install/refresh this real path itself on every launch
(making it self-healing the way `install_dir.rs` already resolves the *AppImage's own*
paths), or find why `VK_ADD_LAYER_PATH`/implicit-layer discovery can't reach the game
process directly and drop this second install location entirely. Not yet done --
flagged here so it isn't rediscovered the hard way again.

## NGX `FAIL_PLATFORM_ERROR` (`0xbad00002`) -- FIXED (2026-09-11, `lordnikon`), plus a
## second, separate real bug found and fixed the same session: captured frames never
## actually reached the helper at all

**Both now fixed and verified end to end on real hardware.** A real `vkcube` run with
this fix showed `model_up=1`, `helper_frames=128` (a full 10-second run), real
`VULKAN_CreateFeature(18) -> 0x1`, and `EvaluateFeature -> 0x1` on essentially every
frame -- the first genuinely complete, working real-time NGX pipeline this project's
own code has produced since the app-removal/reinstall that broke it.

### Bug 1: `FAIL_PLATFORM_ERROR` from `AllocateParameters` -- Core's own allocator is
### unusable in this environment, and that's OK: it isn't actually needed

**Root cause, confirmed via a real side-by-side run against upstream's own compiled
helper** (recovered from its official GitHub release, `bmitch87/DLSS5VKLayer`'s
`0.2.6-1` tag -- never its source, same "shape not expression" rule as everywhere
else in this project): upstream hits the **identical** `0xbad00002` from Core's
`VULKAN_Init_with_ProjectID`/`VULKAN_Init_Ext`/`AllocateParameters` on this same
machine, with this same real `nvngx.dll`. This is a real, expected rejection in this
environment for *any* implementation, upstream included -- not a bug in this
project's caller-identity spoof or call sequence, and not a prefix/environment gap
(confirmed identical against both a freshly-recreated prefix and the real, untouched,
previously-working GTA San Andreas prefix). Upstream's own log shows the actual
recovery: when Core's allocator fails and the snippet doesn't export one either
(`nvngx_dlssnr.dll`'s own `NVSDK_NGX_VULKAN_*` exports are PE forwarders straight
into Core -- confirmed by deliberately keeping Core unloaded and watching
`GetProcAddress` fail to resolve them on the snippet too, so there's no real "call
via snippet instead of Core" workaround for this call family), it falls back to
**its own self-implemented, in-process `NVSDK_NGX_Parameter` object** -- NGX's real
entry points never actually require a parameter block the DLL itself allocated, just
a pointer matching the real vtable shape, which anyone can construct.

**The fix** (`crates/helper/src/selfparam.rs`, new): a `#[repr(C)]` object whose
first field is a vtable pointer (matching `abi::NgxParameterObj`'s real C++-ABI
layout) backed by a plain Rust `HashMap` for storage -- safe because C++ virtual
dispatch only ever touches the object opaquely through that one vtable pointer;
nothing on the DLL side assumes anything else about its layout. `ngx.rs` now falls
back to `selfparam::allocate()` whenever the real `AllocateParameters` call fails or
isn't exported, instead of disabling the whole session. Also removed an unproven
`NVSDK_NGX_VULKAN_Init_ProjectID` call added earlier the same day: real, reproduced
evidence (not the "poisons later calls" theory first assumed) showed removing it
entirely changed nothing about `AllocateParameters`' own rejection -- upstream's own
*proven* ProjectID route is `NVSDK_NGX_D3D12_Init_with_ProjectID` against a dedicated
D3D12 device, a different API family than this helper's Vulkan device entirely; the
Vulkan export exists on the DLL but was never exercised/proven by anyone.

**Verified**: a real manual helper run on `lordnikon` (bypassing `dlssnr-cli`/
`dlssnr-gui`, same technique as every prior real-hardware NGX test) showed
`AllocateParameters -> 0xbad00002` followed immediately by `falling back to a
self-implemented NVSDK_NGX_Parameter object`, a passing round-trip self-test, and
`VULKAN_Init_Ext -> 0x1` -- the helper no longer disables itself. `NgxSnippet` gained
a `self_params: bool` field so teardown calls `selfparam::destroy` instead of a real
`DestroyParameters` export for an object that was never one of the DLL's own
allocations.

### Bug 2: captured frames never reached the helper at all -- a real, separate,
### pre-existing bug found while verifying bug 1's fix, unrelated to NGX

**With bug 1 fixed, `vkcube` created a real swapchain and `capture::run` was
confirmed (via temporary bisection logging, since reverted) to be called on every
single present -- yet `helper_frames` stayed at 0 the entire run.** Root cause, found
by reading `capture::run`'s own first few lines: its very first check,
`shm.composition_settings()`, only ever *reads* through an already-open SHM mapping
(`ShmClient::header()`) -- it never opens one. The code that actually opens the
mapping (`ShmClient::open` inside `try_round_trip`/`begin_async_request`) lives
*later* in the same function, gated behind that first check. On a brand-new process
the mapping is never open yet, so `composition_settings()` returned `None` on
literally every single frame, forever, and the function always bailed out before
ever reaching the code that would open the mapping in the first place -- a real
chicken-and-egg ordering bug, almost certainly introduced somewhere during the
async-pipelining rewrite (0.1.16-0.1.22) when this function's checks got reordered
for early-exit efficiency without preserving the implicit "something must open the
mapping first" invariant the old code apparently had. Notably, this was NOT caught
by the existing test suite (`capture::tests::run_never_blocks_on_a_slow_helper_...`
still passes) -- worth knowing if writing a future regression test for this: a test
that manually pre-opens its own `ShmClient` before calling `capture::run` would not
catch this class of bug either.

**The fix**: `capture::run` (`crates/layer/src/capture.rs`) now calls `shm.open()`
unconditionally as its very first line, before `composition_settings()`.
`ShmClient::open` is already idempotent (an immediate no-op once already open, per
its own early `if self.header().is_some() { return true; }`), so there's no real
per-frame cost to calling it unconditionally instead of leaving callers to remember
to.

**A related, real logging bug found and fixed along the way, worth knowing about
separately**: diagnosing bug 2 was made harder than it should have been because
`crates/layer/src/logging.rs`'s modulo-64 flush throttle (added earlier for
per-frame hot-path performance, see its own doc comment) meant a `vkcube` process
killed by `timeout`'s default `SIGTERM` lost every buffered log line since the last
flush -- including one-time milestones like device/swapchain creation, which matter
far more than the steady-state per-frame logging the throttle exists to protect.
Fixed by adding a `logging::flush()` (both `layer` and this pattern already existed
in `helper`) called explicitly right after the "hooked device"/"swapchain created"
log lines in `device.rs` -- one-time events, not the hot path, so the extra flush
syscall costs nothing meaningful.

**How this was found**: a real side-by-side comparison against upstream's own
compiled helper (recovered from its official GitHub release) was the key that broke
the NGX investigation open, followed by temporary, real bisection logging directly
in the present hook (added, used, then fully reverted -- `git diff` confirms no
bisect leftovers) to pinpoint bug 2 once bug 1's fix exposed it. Neither bug could
plausibly have been found by re-reading the code alone; both needed a real `vkcube`
run against real hardware with the exact right diagnostic (ground truth from
`dlssnr-cli shmctl status`'s live SHM read, not just log output, which is what
caught that bug 2 was real and not just a logging artifact).

**Real incident this session, fixed, worth remembering**: a manual diagnostic run
intended to be read-only pointed a different Proton build (`Proton-CachyOS Latest`)
at the real GTA San Andreas prefix, which was actually associated with
`GE-Proton11-6`. Proton detected the version mismatch and ran a full `wineboot -u`
(regenerating the built-in Wine skeleton, rewriting `system.reg`/`user.reg`/
`userdef.reg`, downloading an FSR4 upscaler file) -- non-destructive in the end
(Wine's builtin-DLL regeneration preserves app-added registry keys), but the
diagnostic's `timeout N` killed the outer SSH/proton command without cleanly
stopping the `wineserver`/`winedevice.exe`/`xalia.exe` children it spawned, and that
orphaned `wineserver` (still bound to the prefix, under the wrong Proton build) then
blocked the game from launching normally through Steam until it was found (via
`/proc/<pid>/environ`'s `WINEPREFIX`) and killed (`wineserver -k` with `WINEPREFIX`
set to the exact prefix path, which for a Proton-managed prefix means the `pfx/`
subdirectory, not the `compatdata/<appid>` directory itself -- `wineserver -k`
silently no-ops against the wrong path with no error). **Lesson for next time**:
before running any Proton/Wine command against a prefix you don't manage, check
that prefix's own `version` file and use the exact same build; afterward, verify no
process is still alive with that `WINEPREFIX` in its environment, not just that the
outer command returned.

## Real red/blue channel swap on any `B8G8R8A8` swapchain -- FIXED (2026-09-11),
## found from a live user report during real gameplay with NGX finally working

Reported live, in real time, while playing GTA San Andreas with the NGX fix above
active: "getting flickering and it doesn't look like games when dlss5 is turned on."
Root cause, confirmed and fixed the same session -- **not related to the NGX
platform-error fix above at all**, a completely separate, pre-existing bug that
simply had no way to surface visually until NGX actually started working.

**What was wrong**: `vkCmdCopyImageToBuffer`/`vkCmdCopyBufferToImage` are raw,
format-preserving byte copies -- Vulkan never reorders channels during a copy, only
during a `vkCmdBlitImage` or a sampled (not storage) image read. This project's own
code (`swapchain::proxy_format_for`, `composition/apply.rs`'s CPU path,
`shaders/compose.comp`'s GPU path, and `dump.rs`'s debug PNG writer) all hardcoded an
`R,G,B,A` byte order for every 8-bit-per-channel format, on the reasoning that both
`R8G8B8A8` and `B8G8R8A8` are "4 bytes/pixel" (true for *size*, false for *channel
order*). This machine's real swapchain -- confirmed via real `vkcube` and real game
testing, not assumed -- is `B8G8R8A8_UNORM`, so every captured/composited/presented
pixel had its red and blue channels silently swapped. Confirmed visually via a real
`dlssnr-cli shmctl capture` dump during the live session: a uniform blue tint across
the *entire* frame (walls, character skin, everything), present identically in both
the "original" (pre-model) and "composited" dumps -- proving the swap happened
upstream of the model/composition math, not because of anything the model itself
produced.

**The fix**: `swapchain::is_bgr_order(format)` (new) detects the real channel order;
a `bgr_order: bool` now threads through the whole capture/composition pipeline --
`capture::run`/`run_sync`, `composition::apply::apply_rgba8` (swaps the R/B *byte
indices* it reads/writes, not the pixel math itself, which only ever deals in clean
`[r,g,b]` triples), `composition::gpu`'s three `dispatch*` functions and their shared
`PushConstants`, and `shaders/compose.comp` (a new `bgr_order` push-constant field,
swizzling `.bgr` on `imageLoad` and again on the final `imageStore` -- recompiled
with `glslangValidator`/validated with `spirv-val`, both confirmed available on this
dev machine as of this session), and `dump.rs`'s PNG writer. Two doc comments in
`composition/gpu.rs` that had asserted a buffer-mediated write-back is "correct
regardless of the real image format" were corrected -- true for size/layout
compatibility, false for channel order, exactly the bug this section documents.

**Verified for real, not just "compiles"**: a new, deliberately strict test
(`composition::gpu::tests::bgr_order_produces_the_same_true_colors_as_rgb_order_on_swapped_bytes`)
feeds the exact same semantic colors as the existing `gpu_dispatch_matches_the_cpu_reference`
test, but physically stored in swapped (BGR) byte order with `bgr_order: true`, and
asserts the *true* colors this produces (read back through the swapped indices)
match that separate, independently-computed RGB-order run's result -- not just that
the GPU and CPU paths agree with each other (which they could do while both being
consistently wrong the same way, exactly as they were before this fix). Full
workspace test suite (34 layer tests) stays green.

**Known, deferred, separate gap**: `crates/helper/src/frame.rs`'s `COLOR_FORMAT`
(the NGX `DLSSNR.Color`/`.Output` Vulkan resource format) is still hardcoded
`R8G8B8A8_UNORM` regardless of the real captured format -- this fix corrects what
the *layer* captures, composites, and presents, but the model itself may still
receive/produce mislabeled color data on a real `B8G8R8A8` swapchain. Fixing this
properly means threading the real channel order through the SHM protocol (a new
header field, a `SHM_VERSION` bump per `header.rs`'s own discipline) so `frame.rs`
can create its Color/Output images with the real matching `vk::Format` instead of a
hardcoded constant. Not yet done -- the visual symptom reported this session (a
uniform blue tint matching a pure channel swap, not a subtler model-confusion
artifact) was consistent enough with the layer-side bug alone that fixing it first
and re-checking with real gameplay was the right next step, not assuming both gaps
needed fixing before shipping anything.

**Also fixed the same session, unrelated**: the GUI's "Enabled" toggle
(`ShmHeader::enabled`, `neural_enabled()`) was never actually read by the capture
path at all -- turning it off in the GUI had no effect on anything real. Now gates
`capture::run` the same way `apply_model` already does.

## Real flicker root-cause, diagnosed and understood -- a known, deliberate tradeoff,
## not a new bug (2026-09-11, same live session as the channel-swap fix above)

With the channel-swap fix above deployed, Alex reported (still live, still playing):
"when I toggle neural rendering off the flickering stops and the fps goes up but
the game looks exactly the same with it on or off." Two separate real questions,
both answered with hard evidence this session, not guessed:

**What causes the flicker -- confirmed, not fixed by choice.** `capture::run`'s
async pipeline (0.1.22) captures a new frame only when no round trip is already in
flight; when a helper's answer *does* arrive, it composites `inflight.original`
(whatever frame was sent, possibly several real frames ago) against `image` (this
present call's own, freshly-rendered, possibly quite different content) -- exactly
the documented, deliberately-authorized tradeoff from 0.1.22's own writeup ("NR
visibly updates at whatever rate the round trip achieves... can be composited
against a slightly newer frame than the one it was computed from"). Verified by a
real, temporary diagnostic (added, tested, then fully reverted -- `git diff`
confirms no leftovers): a marker-file-gated flag that forced every frame through
`run_sync` (same-frame correctness, no `inflight` staleness) instead of the async
path. Live, real-time result while Alex watched: flicker gone, FPS dropped to
single digits (matching the pre-0.1.22 fully-synchronous profile). **Root cause
confirmed. Given the choice, Alex chose to keep the async default and accept the
flicker** rather than trade back the FPS gain -- this is a real, informed product
decision, not an open bug. If this comes up again, the fix already exists and is
already understood (force the synchronous path) -- it just isn't what Alex wants
by default. Don't silently re-implement it as a default without asking again.

**Why NR "looks the same" on/off -- confirmed real and working, just visually
subtle for this content, not silently broken.** A real pixel diff between an
original and composited capture from the same live session (sampled every 3rd
pixel, `PIL`, not eyeballed) showed **95% of the frame changed**, mean per-pixel
sum-of-absolute-channel-difference ~93.5/765 -- larger than the ~17/255-mean-diff
scene documented in the "First confirmed correct visual output" section above, not
smaller. NR is genuinely running and producing a real, structured, non-trivial
result on real gameplay content. It's just that `upgrade_tone_map`'s actual designed
effect -- highlight recovery + a gentle hue correction -- is a subtle global
tonal/exposure shift for a normally-exposed indoor scene with no blown highlights,
not a dramatic "AI-sharpened" look. A scene with real blown highlights (bright sky,
direct sun, headlights) would very likely show a much more obvious before/after --
untested this session, a real, cheap next step if this comes up again and a more
convincing demo is wanted.

**Still-open, separate, smaller-priority gaps, unaffected by any of the above**:
real motion vectors (still an all-zero placeholder) and the NGX model's own
Color/Output format (still hardcoded RGBA regardless of the real captured format,
see the channel-swap section above) are both still real, still open. Neither was
what caused the flicker Alex actually experienced and reported this session --
don't reach for either as "the fix" without new evidence pointing at them
specifically, the way the async-pipeline diagnostic above pointed at frame staleness.

## `dlssnr_supervisor::stop()` could leave an orphaned Wine-hosted helper running --
## FIXED (2026-09-11), found chasing "GTA V Enhanced has no effect and no
## performance cost" on `lordnikon`

Alex tested against a second real game (GTA V Enhanced) the same session: "no
flickering but doesn't have and[sic] effect on the look or performance." Zero
performance impact was the real tell -- if NR were genuinely running, even the
cheapest path costs *something*. Investigation found a real, separate,
process-supervision bug, not anything about GTA V Enhanced itself or the NGX/
composition pipeline covered elsewhere in this file.

**What was found**: `helper_state` read back `MODEL_FAILED` from a live
`dlssnr-cli shmctl status` even though the *current* helper process's own log
showed nothing but real, successful `EvaluateFeature -> 0x1` calls -- a genuine
contradiction, since `ensure_feature`'s "one-shot" design (see its own doc comment)
makes it structurally impossible for a helper that already has a working feature to
ever reach the `MODEL_FAILED`-setting code path again. The real explanation: a
**second, orphaned `dlssnr_helper.exe` process from earlier manual testing this
same session was still alive**, writing to the exact same `/tmp/dlssnr-1000/shm.bin`
mapping as the properly-started one -- two independent writers racing on one shared
header. Confirmed directly: `ps`/`/proc/<pid>/environ` showed a stray
`dlssnr_helper.exe` (started hours earlier for the NGX/BGR investigations above,
never cleanly killed) still bound to the same `DLSSNR_SHM` path.

**Root cause, confirmed by direct reproduction, not inferred**: `dlssnr-cli stop`
(→ `dlssnr_supervisor::process::stop`) sends `SIGTERM`/`SIGKILL` to the process
*group* the original `setsid()`'d launcher led, then declares success once that
one process-group-leader PID is dead -- it never checks whether the real,
Wine-hosted grandchild (the actual `dlssnr_helper.exe`, once wineserver takes it
over) is *also* gone. Reproduced live: after a `dlssnr-cli stop` that printed
"helper stopped" followed by `dlssnr-cli start`, the *old* Wine-hosted `.exe` was
still running (confirmed via `ps`) alongside the brand-new one. This is the same
class of problem `crates/supervisor/src/process.rs`'s own `#[ignore]`d test
(`stop_kills_the_whole_process_group_not_just_the_leader`) flagged as a *dev
sandbox limitation* -- but this reproduction happened on `lordnikon`, the real
target machine, not the sandbox. Wine/Proton's own process management genuinely
doesn't reliably keep every descendant inside the original process group; this
isn't purely a sandboxed-signal-delivery artifact.

**The fix** (`crates/supervisor/src/lib.rs`): `stop()` now also runs
`wineserver -k` against the exact configured `WINEPREFIX`
(`paths::prefix_dir()`) after the normal process-group kill -- the identical
manual recovery command this project's own real-hardware testing has used by hand
every single time this exact symptom came up (see the "Real incident this
session" notes elsewhere in this file). New `wineserver_binary(cfg)` resolves the
real wineserver binary next to whatever Proton build is configured
(`<runner dir>/files/bin/wineserver`, confirmed present at that exact relative
path on real installs) or falls back to `PATH` for a plain-Wine runner. Best-effort
(ignores errors) -- a plain Wine install with no `wineserver` on `PATH`, or nothing
left to kill, are not real failures worth surfacing.

**Verified**: 4 new tests for `wineserver_binary`'s own resolution logic (proton
with a real sibling, proton with none, plain wine falling back to `PATH`, no
runner configured at all) plus manual cleanup + a clean restart on `lordnikon`
confirmed `helper_state` reads back `RUNNING` again with exactly one helper
process alive. A second, unrelated but real bug was found and fixed the same
pass: `paths::tests::finds_a_real_steam_install_under_xdg_data_home` and
`returns_none_when_no_candidate_exists` raced on the same process-wide
`XDG_DATA_HOME` env var (confirmed genuinely intermittent under `cargo test`'s
workspace-wide scheduling, not hypothetical) -- fixed with a shared lock.

**One more real bug in this very fix, caught testing it before trusting it
(0.1.29)**: the first version of `stop()`'s new `wineserver -k` call passed
`paths::prefix_dir()` directly as `WINEPREFIX` -- but for `runner_type = "proton"`
that isn't the prefix Wine itself actually uses. Proton's own launch script
internally re-derives and uses `STEAM_COMPAT_DATA_PATH/pfx` (`start()` hands
`prefix_dir()` to Proton *as* `STEAM_COMPAT_DATA_PATH`), so the real, live
`WINEPREFIX` is always one level deeper. Caught immediately by testing the deployed
fix against the exact orphaned process it was meant to clean up: `wineserver -k`
with the bare prefix dir exited `1` and killed nothing; the identical command with
`/pfx` appended exited `0` and actually worked. Fixed with a new, deliberately pure
`real_wineprefix(cfg, prefix_dir)` function (takes the prefix as a parameter rather
than calling `paths::prefix_dir()` itself, specifically so it's testable without
touching the process-wide `XDG_DATA_HOME` env var the real function depends on) --
2 new tests. **Lesson for next time**: this exact `/pfx`-nesting gotcha was already
documented once earlier in this file (the "Real incident this session" note on
Proton version mismatches) -- write real, automated regression coverage for a
gotcha the first time it's found, not just a comment, or it silently costs a second
real bug later exactly like it did here.

**Not yet re-tested against GTA V Enhanced itself** after this fix and the
cleanup -- the immediate cause of "no effect" this session was the corrupted
shared state from the orphaned process, not necessarily anything specific to that
game, but that's inference, not confirmation. If it still shows no effect after a
clean helper restart, treat that as a fresh, unconfirmed report, not a re-run of
this same bug.

## `~/AppImages/dlssnr.appimage` is NOT the real, Gear-Lever-managed app -- a real
## deployment mistake this whole session, found and fixed (0.1.30)

**Read this before ever deploying a build to `lordnikon` by hand again.** Every
AppImage rebuild from v0.1.24 through v0.1.29 this session was deployed via
`scp`+`mv` to `~/AppImages/dlssnr.appimage` (a plain, unversioned filename) under
the assumption that was "the" app. It is not. The real, actually-integrated,
desktop-launched app is `~/AppImages/dlssnr.appimage_0_1_25.appimage` --
confirmed via its own `~/.local/share/applications/dlssnr.appimage_0_1_25.desktop`
launcher entry (`Exec=`/`TryExec=` both point at that exact versioned filename,
`X-AppImage-Version=0.1.25`). Gear Lever (`it.mijorus.gearlever`, installed as a
Flatpak) is what created this versioned-filename-plus-desktop-file pair when Alex
originally integrated the AppImage through it; a raw `scp`+`mv` to a *different*
path is invisible to it entirely. The two files are completely independent
(different inodes, different content) -- overwriting the wrong one all session
meant every GUI/CLI-level fix (the console-window fix, the orphaned-helper
`stop()` fix) never reached what Alex's own desktop icon actually launches, only
what this project's own manual SSH-based testing exercised. **This did not affect
the real game-visible fixes** (the NGX/color-channel work) -- those deploy the
Vulkan layer `.so` to a separate, always-correct path
(`~/.local/share/dlssnr/lib/libdlssnr_layer.so`, see the "Real-machine deploy
gotcha" section) that the game loads directly via its own Vulkan manifest,
independent of which GUI binary exists or which version it reports.

**Found via a real user report, not inspection**: "you broke something because I
cannot update dlssnr using gear lever" -- Gear Lever's own "check for updates" ran
against the real `dlssnr.appimage_0_1_25.appimage` and reported nothing newer,
despite v0.1.29 genuinely existing on GitHub. Investigation found two real,
separate problems: (1) this deployment mistake, meaning the *managed* file itself
had never moved past a build old enough to matter less, and (2) a real, separate
bug in `build-appimage.sh` itself -- see below.

**The `build-appimage.sh` bug, fixed**: `UPDATE_INFORMATION` embedded
`gh-releases-zsync|labj1987|Dlssnr|latest|...` (capital `D`) as the GitHub repo to
check -- the real repo is `labj1987/dlssnr` (lowercase). `gh api
repos/labj1987/Dlssnr/...` resolves this fine (GitHub's own API/web layer handles
the case mismatch), so this looked like it might be a red herring at first -- but
Gear Lever's own update-check client apparently does *not* handle it the same way,
matching the exact real symptom reported ("no updates found") rather than a
visible error. Fixed to match the repo's real casing exactly rather than relying
on any client's redirect behavior. Also gave `zsyncmake` a real, absolute `-u
<url>` (this exact release's GitHub download URL) instead of letting it default to
a bare relative filename in the `.zsync`'s own internal "URL:" header -- separate
metadata from `UPDATE_INFORMATION`, used by whatever client downloads the actual
new bytes once an update is found.

**What v0.1.30 actually fixes, precisely**: the corrected `UPDATE_INFORMATION`/
zsync-URL metadata is embedded in *this* release's own AppImage. For Gear Lever to
ever find it (or any future release) automatically, the *currently-managed* file
(`dlssnr.appimage_0_1_25.appimage`) needs to actually become a build that carries
this fix -- simply publishing v0.1.30 to GitHub does not retroactively fix what
the stale 0.1.25 file already has embedded. Check what was actually done to
`dlssnr.appimage_0_1_25.appimage` in this session's own real actions (in-place
content replacement, preserving the filename/desktop-entry Gear Lever already
knows about, was the plan discussed with Alex) before assuming Gear Lever's
update flow "just works" from here on without verifying it for real.

**Lesson for next time, plainly**: when deploying anything meant to reach a real
user-facing app on `lordnikon`, find out how that app is *actually* installed and
launched first (check `~/.local/share/applications/*.desktop` for the real
`Exec=` path) rather than assuming a plausible-looking file path is the right
target. This cost an entire session's worth of GUI/CLI-level fixes never reaching
the user until they happened to try updating and noticed.
