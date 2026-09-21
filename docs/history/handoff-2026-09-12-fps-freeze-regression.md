> Historical record: pre-NeuralForge names and deployment instructions below are
> archival, not current instructions. Do not remove or modify upstream installations.
> See ../../PHASE1.md for current paths, safety constraints and the benchmark plan.

# NeuralForge pre-rename handoff — 2026-09-12, FPS/freeze regression unresolved

## Current state

Historical checkout: `/home/alex/Projects/dlssnr` (now `/home/alex/Projects/neural-forge`). Current repository: `labj1987/neural-forge`.
Latest release: **v0.1.35** — https://github.com/labj1987/NeuralForge/releases/tag/v0.1.35
This is a **rollback release**. The underlying bug is still open.

Deployed live on the test machine `lordnikon` (SSH alias `lordnikon`, reachable from
this dev machine): the v0.1.35 AppImage at `/home/alex/AppImages/dlssnr.appimage`
(the real, desktop-launched app — confirmed via its `.desktop` file's `Exec=`), and
the matching layer at `/home/alex/.local/share/dlssnr/lib/libdlssnr_layer.so`
(a separate, manually-deployed copy the game actually loads — see "Deployment"
below). Both are current as of this handoff.

## The problem, precisely

With `dlssnr`'s Vulkan layer active (`enabled=1` in its settings, `VKLayer_DLSS5=1`
in a game's Steam launch options) and a real NVIDIA GPU (RTX 5070, driver
`615.71.09`), real gameplay is unplayable:

- **Confirmed, reproducible, real measurement**: GTA V Enhanced on `lordnikon` ran at
  **196–274 FPS with neural rendering off**, collapsing to a **steady 9 FPS with it
  on** — reproduced twice, both directions, instantly. GPU utilization *dropped*
  when it turned on (21–26% vs. 99%) — the signature of CPU-side blocking, not more
  real GPU work.
- **Ruled out**: motion vectors / optical flow specifically (toggling `mvec_enabled`
  off while `enabled` stayed on made no difference — still 9 FPS).
- **Ruled out**: a separate, real, already-fixed bug — upstream's own C++ package
  (`dlssnr` 0.2.6-3, apt) was still installed and running on `lordnikon` from an
  earlier comparison session, sharing the *same* `VKLayer_DLSS5` trigger and the
  *same* `/tmp/dlssnr-$UID/shm.bin` mapping. Uninstalled (`apt remove dlssnr`),
  confirmed gone. This was real but is **not** the current problem — it was fixed
  and verified separately before any of the below started.
- **This session's fence-wait "fix" attempt made it worse, not better**: after
  reverting (see below), the game goes back to running (slowly, ~9fps) rather than
  crashing outright. The crash was a regression from this session's own changes, now
  reverted. **The original 9fps-vs-274fps regression is still fully unresolved.**

## What's confirmed fixed and safe to keep

1. **Upstream package conflict** (environment, not code): `dlssnr` 0.2.6-3 apt
   package uninstalled from `lordnikon`. No code change; nothing to revert. Confirmed
   gone via `dpkg -l dlssnr` (shows `rc`, removed) and no files under
   `/usr/share/vulkan/implicit_layer.d/` matching `dlssnr`.
2. **GUI retabbed like upstream** (`crates/gui/src/ui.rs`, shipped in v0.1.32):
   cosmetic, unrelated to this bug, working, not reverted.
3. **Shared-memory directory permission bug** (`crates/protocol/src/mapping.rs`,
   kept in the revert): `dlssnr_protocol::mapping::open_at` used to create
   `/tmp/dlssnr-$UID/` with `std::fs::create_dir_all` and no explicit mode — the
   resulting permissions depended on whatever umask the *first* process to create it
   that boot happened to have. The Vulkan layer's own `ensure_private_parent_dir`
   security check (`crates/layer/src/shm.rs`) then **permanently refuses to use a
   non-private directory for the rest of that process's life**, silently disabling
   neural rendering with no visible error (only a `[shm] refusing ...` line in the
   layer's own log, which nothing captures for a real game launch since
   `DLSSNR_LOG` is never set for one). **Real bug, confirmed via `strace -f` on a
   live game process** showing that exact message flooding every frame. Fixed:
   `open_at` now explicitly `set_permissions(dir, 0o700)` every call, not just on
   first creation — immune to umask, self-heals a directory a previous build already
   got wrong. This fix is real, low-risk (never touches Vulkan sync), and kept.

## What was tried and reverted — read before retrying anything similar

`crates/layer/src/capture.rs` and `crates/layer/src/composition/gpu.rs` are back to
their **exact v0.1.31/v0.1.32 state** (`git checkout bf6f773 -- <those two files>` —
confirmed byte-identical between v0.1.31 and v0.1.32 for both). Everything below is
**reverted, not present in the current tree** — described here so it isn't
re-attempted blindly.

**The hypothesis** (plausible, but not proven correct): `capture_pristine`
(`capture.rs`) — the synchronous stage-1 GPU capture the async pipeline always pays
for once per round-trip cycle — unconditionally called `device.reset_command_buffer`
on every cycle, assuming (stated as a hard invariant in both its own and `ensure()`'s
doc comments) that the *previous* cycle's submission had already been waited on to
completion via an **unbounded** `wait_for_fences(..., u64::MAX)`. The working theory
was that this wait was not reliably completing quickly on this real hardware/driver,
and that resetting a command buffer whose previous submission hasn't finished is
undefined behavior regardless.

**What was actually done** (all reverted now):
- A non-blocking `get_fence_status` check at the entry of `capture_pristine`,
  skipping the whole cycle (treated as a normal failure, same as every other
  fail-open path in this module) if the fence isn't confirmed signaled, instead of
  resetting possibly-in-flight resources.
- The same guard added to `ensure()`'s resize/rebuild path.
- The one remaining wait in `capture_pristine` (for *this* cycle's own fresh
  submission) bounded to 8ms instead of `u64::MAX`.
- Once a **separate, real bug** (the permission issue above) was fixed, live testing
  showed the pipeline could for the first time actually reach the *compose* stage
  regularly, which exposed an **identical** unbounded wait in
  `GpuCompose::dispatch_into_image_async`'s own entry (`composition/gpu.rs`) — the
  primary per-frame compose path, not a rare fallback. The same bounded-wait +
  entry-guard treatment was applied there too, plus to `dispatch_into_image`/
  `dispatch`'s shared `self.sync` resource (same reset-while-in-flight risk, +
  `write_bytes_to_image`'s own last-resort wait).

**Why it's reverted**: live user testing of the deployed build with *all* of the
above in place showed **GTA games now crash outright on open** (not freeze — an
actual crash) and Crimson Desert still needed a force-close. Worse than the original
symptom (slow, not crashing), and not a validated fix for anything.

**The real gap in the reasoning, identified after the fact**: this whole
investigation never ran with **Vulkan validation layers enabled**
(`VK_LAYER_KHRONOS_validation`). The non-blocking `get_fence_status`-then-reset
pattern was reasoned through by hand against the Vulkan spec, not verified against
it with tooling that actually catches synchronization mistakes. There's a real,
unverified possibility of a race between the fence-status check and the subsequent
`vkResetCommandBuffer`/`vkQueueSubmit` (the fence could signal *between* the check
and the reset in a way the code doesn't account for), or a driver-specific quirk on
this exact NVIDIA `615.71.09` build that the check doesn't correctly handle. **Do not
re-attempt this class of fix without validation layers running first.**

## Suggested next steps, roughly in priority order

1. **Turn on Vulkan validation layers** (`VK_LAYER_KHRONOS_validation`, set via
   `VK_INSTANCE_LAYERS` alongside `VK_LAYER_dlssnr_neural`, or `vkconfig`) for any
   further investigation, starting with `vkcube` (cheap, fast, no launcher
   flakiness — see below) before ever touching a real game again.
2. **Re-confirm the actual bottleneck with validation on** — this session's own
   diagnosis (an unbounded `wait_for_fences` genuinely stalling) may or may not be
   correct; validation output plus proper GPU-side profiling (RenderDoc frame
   capture, or `nsys`/Nsight if available) would show definitively where the real
   per-frame cost is, rather than inferring it from FPS/GPU-utilization numbers
   alone.
3. **Consider that the cost might be genuine GPU/driver-side slowness**, not a
   synchronization bug at all — this is an RTX 5070 on driver `615.71.09`; worth
   checking whether that's a beta/early driver build with known issues independent
   of anything in this codebase.
4. **Re-verify the motion-vector/optical-flow ruling-out** — it was tested with a
   single live A/B toggle in one session; worth confirming with the private optical
   flow device's own resource lifecycle in mind (whether disabling `mvec_enabled`
   truly tears down the private device immediately, or only on the next
   session/reconnect).
5. If attempting a fix, **iterate against `vkcube` first**, with validation layers
   on, and only move to a real game once `vkcube` shows zero validation errors and a
   measured throughput improvement. This session went straight to real games twice
   and paid for it in wasted time both from Rockstar Games Launcher's own relaunch
   flakiness (below) and from shipping unverified fixes directly to the user.

## Environment and how to reproduce/test

**Test machine**: `lordnikon`, reachable via `ssh lordnikon` from this dev machine
(passwordless SSH already configured). Real desktop session (GNOME/Wayland +
Xwayland), NVIDIA RTX 5070, driver `615.71.09`. The user is often physically present
at this machine.

**Extracting the CLI for debugging** (the AppImage's `AppRun` always execs the GUI,
never the CLI, so extract it to invoke `dlssnr-cli` directly):
```bash
ssh lordnikon "mkdir -p /tmp/dlssnr-appimage-extract && cd /tmp/dlssnr-appimage-extract && /home/alex/AppImages/dlssnr.appimage --appimage-extract >/dev/null 2>&1"
```
Then the CLI is at `/tmp/dlssnr-appimage-extract/squashfs-root/usr/bin/dlssnr-cli`.
Useful subcommands: `doctor`, `status`, `start`/`stop`, `shmctl status` (dumps every
live setting plus `helper_state`/`model_up`/`helper_frames`), `shmctl toggle <name>`,
`shmctl set <name> <value>`, `shmctl capture` (one-shot forced capture dump).

**Helper log**: `~/.local/state/dlssnr/helper.log` on `lordnikon` — cumulative across
restarts (append-only), so `tail` the very end, not `grep` for old matches.

**Layer log**: the layer's own log goes to `DLSSNR_LOG` if set, otherwise stderr —
and nothing sets `DLSSNR_LOG` for a *real game* launch (only `dlssnr_supervisor::start()`
sets it for the *helper*). To see it for a real game:
- Cleanest: `strace -f -p <game_pid> -e trace=write -s 300 2>&1 | grep dlssnr` on an
  already-running game process (must use `-f` — the render/present thread is not the
  main thread).
- Alternative: temporarily append ` DLSSNR_LOG=/tmp/x.log` to the game's Steam
  launch option string in `~/.steam/steam/userdata/*/config/localconfig.vdf` (back
  it up first, revert right after triggering the launch — Steam periodically
  rewrites this file, so don't leave it edited).

**Vulkan layer deployment** (this project's own real gotcha, easy to get wrong):
the *managed*, actually-loaded layer library is
`~/.local/share/dlssnr/lib/libdlssnr_layer.so` on `lordnikon` — a **separate,
manually-maintained copy**, not something the AppImage installs itself. The AppImage
at `~/AppImages/dlssnr.appimage` is the real, desktop-launched GUI/CLI bundle
(confirmed via `~/.local/share/applications/dlssnr.desktop`'s `Exec=`). Deploying a
new build means **both**:
```bash
scp target/release/libdlssnr_layer.so lordnikon:/home/alex/.local/share/dlssnr/lib/libdlssnr_layer.so
# and, for a full release:
CARGO_HELPER="cargo +stable-x86_64-unknown-linux-gnu" bash build-appimage.sh
scp dlssnr-<version>-x86_64.AppImage lordnikon:/home/alex/AppImages/dlssnr.appimage.new
ssh lordnikon "mv ~/AppImages/dlssnr.appimage.new ~/AppImages/dlssnr.appimage && chmod +x ~/AppImages/dlssnr.appimage && sed -i 's/X-AppImage-Version=<old>/X-AppImage-Version=<new>/' ~/.local/share/applications/dlssnr.desktop"
```
An already-running game process won't pick up a new `.so` until relaunched (the old
one stays mapped from its original inode).

**Toolchain gotcha**: this dev machine has a non-rustup system Rust install kept as
the `system` rustup toolchain (default), with a separate `stable-x86_64-unknown-linux-gnu`
rustup toolchain added just for the `x86_64-pc-windows-gnu` cross-compile target
`build-appimage.sh` needs for the Windows helper. Always set
`CARGO_HELPER="cargo +stable-x86_64-unknown-linux-gnu"` when running
`build-appimage.sh` on this machine, or it fails with `toolchain 'system' does not
support components`.

**Games used for testing** (all have `VKLayer_DLSS5=1 %command%` in their Steam
launch options already): GTA V Enhanced (appid `3240220`), GTA San Andreas — The
Definitive Edition (appid `1547000`), Crimson Desert (appid `3321460`). Launch via
the already-running Steam client: `DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000
/home/alex/.local/share/Steam/ubuntu12_32/steam steam://rungameid/<appid>`.

**Rockstar Games Launcher flakiness — real, separate, costs significant time if not
anticipated**: GTA V Enhanced and San Andreas both launch through the full Rockstar
Games Launcher + Social Club stack (Electron-based, multiple helper/renderer
processes), which can take 1–3 minutes to cold-start and, on rapid successive
relaunches, sometimes fails to reach the actual game at all (its whole process tree
exits on its own after ~2 minutes). This is **unrelated to dlssnr** — reproduced
identically with the layer's own trigger env var completely removed. **Give it real
time (poll for the actual `GTA5_Enhanced.exe`/equivalent process, not just
`PlayGTAV.exe`/`Launcher.exe`), and don't relaunch immediately after a failed
attempt** — space relaunches out, or ask the user to launch it themselves
interactively, which has been noticeably more reliable than any automated
`steam://` trigger this session tried.

**Checking FPS/GPU state on a live game**: MangoHud is already enabled system-wide
and shows FPS/GPU%/CPU% in the top-left corner of every Vulkan game. Screenshot the
game window via `xdotool`/`import` (needs `DISPLAY=:0 XAUTHORITY=<the
Xwayland-session auth file, find via `ls /run/user/1000/.mutter-Xwaylandauth.*`>`).

## Full narrative writeup

`/home/alex/Projects/neural-forge/CLAUDE.md` (formerly `/home/alex/Projects/dlssnr/CLAUDE.md`) has the complete, dated, in-depth writeup of
every finding this session (and prior sessions) made, in far more detail than this
handoff — read the top few sections (2026-09-12 entries) for the full reasoning
trail behind everything summarized above, including the exact code-level detail of
what `capture_pristine`/`dispatch_into_image_async` do and why they were suspected.
