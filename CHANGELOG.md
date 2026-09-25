# Neural Forge changelog

One heading per released version, newest first. Versions 0.1.55 to 0.1.63 were
previously filed under "Unreleased" phase headings and are grouped by the release that
first shipped them; their phase is kept as a subheading.

## Unreleased

- Deferred (layer): retired present and relay semaphores (`present_sync.rs`,
  `GpuCompose::retire_present_images`) are still only freed at device teardown; freeing them
  earlier needs proof that the presentation engine's wait on them has completed, which core
  Vulkan cannot give without `VK_EXT_swapchain_maintenance1`.
- Deferred (layer): the unit-test target still carries clippy lints (mostly
  `chunks_exact` with a constant size in test helpers); the library and examples are clean.

## 0.1.99 — 2026-09-25

- **Smooth Motion switch** on the Setup page's Steam launch option. When it's on, the copied
  string adds `NVPRESENT_ENABLE_SMOOTH_MOTION=1` and
  `VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_NV_present:VK_LAYER_VALVE_steam_overlay_64`.
  - **Why the order line is needed:** the Vulkan loader does not order implicit layers, and
    without it NVIDIA's generator and the Steam overlay both loaded above this layer. Generated
    frames then skipped the effect, and the overlay's fps counter only counted real frames.
  - **Measured** in GTA V Enhanced's own benchmark at 2560x1440 on an RTX 5070 (model every
    2nd frame): 116.4 fps displayed against 60.9 without Smooth Motion, for 2.6 real fps.
  - **Details:** the switch is on by default and saved in `config.ini`. Measurements are in
    `docs/FRAMEGEN_SPIKE.md`.
- Loading a settings profile no longer wipes `config.ini` entries that are not model settings.

## 0.1.98 — 2026-09-24

- **Motion vectors are on by default.** Estimated on the GPU they now cost about 2-3% of the
  frame rate: in GTA V at 2560x1440 on an RTX 5070 (FG off, zero-copy active), 44.9 -> 43.4 fps
  with the model every frame and 62.4 -> 61.0 every 2nd frame, 0.65 ms per estimate. The switch
  on the Motion page still turns them off (effective the next time the helper starts).
- For reference, 0.1.97 in GTA V against 0.1.95 (which fell back to CPU copies there): 36.0 ->
  44.9 fps every frame, 50.4 -> 62.4 every 2nd frame.

## 0.1.97 — 2026-09-24

- **Fixed: 0.1.96 stopped the Rockstar Games Launcher (GTA V) from starting.** 0.1.96 learned
  whether a device had `VK_EXT_external_memory_host` by reading the device's extension list in
  its per-device setup, but on the layer framework's own device-creation path that list has
  already been freed by then; reading it crashed the launcher's GPU process, and the launcher
  exited with code 3 ("Unable to launch game"). The layer's device-creation hook now creates
  and records a device that already requests the extension itself, exactly as it does one it
  adds the extension to, and the per-device setup no longer reads the list at all. Verified by
  starting the launcher under the game's own Proton prefix with the layer on and off.

## 0.1.96 — 2026-09-24

- **Zero-copy capture and composition now work in DirectX 12 games (vkd3d-proton), such as
  GTA V.** vkd3d-proton enables `VK_EXT_external_memory_host` on its own, so the layer's
  device-creation hook correctly stood aside, but the layer only recognised the extension when
  it had added it itself; those devices were treated as not having it and fell back to the
  CPU-copy path (`[sync] ... zc=false`). The layer now also reads the application's own request.
  Measured on 0.1.95 in GTA V at 2560x1440, the copies this removes were ~4.6 ms of the ~24 ms
  per frame. If adding the extension is ever refused by the driver, the layer now says so once.

## 0.1.95 — 2026-09-24

- **Frames stay on the GPU between capture and composition.** When the direct capture path is
  active, the layer no longer copies each full frame through the CPU four times (the captured
  original, the helper's answer, and both again into the composition's staging buffer). The
  capture also copies the game's frame into a device-local image, the answer region is imported
  as Vulkan memory, and composition reads both on the GPU, including for frames that reuse an
  answer. Every other case keeps the previous path. Measured in GTA San Andreas DE at 2560x1440
  on an RTX 5070, model every frame: 43.8 -> 53.3 fps.
- **Motion vectors run entirely on the GPU.** The helper scales its already-uploaded frame to
  half resolution, runs optical flow on the flow queue, and converts the result into the model's
  motion image with a compute shader (`flow_to_mvec.comp`), with no frame copied through the CPU.
  0.75 ms per estimate at 2560x1440 instead of 16+ ms. San Andreas with motion vectors, model
  every frame: 18.2 -> 51.4 fps; every 2nd frame: 32.5 -> 83.5 fps.
- The helper no longer zero-fills and uploads a full frame of empty motion when motion vectors
  are off; it clears the motion image on the GPU. Scene cuts are detected from a small luma
  thumbnail instead of a copy of the previous frame.
- The layer's `[sync]` log line now splits the time: `capture_gpu`, `copy_out`, `meter`,
  `wait_answer`, `helper` (the helper's own stages) and `zc` (whether zero-copy was used).
- The layer's capture tests no longer leave shared-memory files in `/tmp`.

## 0.1.94 — 2026-09-24

- **The layer installs itself.** Every launch from the AppImage now copies its Vulkan layer and
  helper into `~/.local/share/neural-forge` when they differ from what is installed, so an
  updated AppImage never runs against a stale installed layer (which 0.1.93's shared-memory
  change would break). When everything is already current it writes nothing. The Setup tab's
  "Install layer for Steam games" button is gone; a failed install shows a message instead.

## 0.1.93 — 2026-09-24

- **The model's temporal history was reset on every frame.** With "Estimate motion vectors"
  on but no motion available (which was always the case), the helper sent `DLSSNR.Reset = 1`
  with every evaluate, so the model treated each frame as the first and never used its
  history. History now resets only on a detected scene cut.
- **Motion vectors work, from the helper.** The "Estimate motion vectors" switch is now the
  only switch (the `NEURAL_FORGE_MVEC_HELPER` variable is gone), and the Motion page's
  controls are live again with an accurate description. The switch takes effect the next
  time the helper starts: the optical-flow queue is only requested then.
- **Optical flow was ~50x slower than it needed to be.** Its readback buffer used uncached
  memory and was read byte by byte; it is now host-cached and copied out in one piece.
  Measured on an RTX 5070 under Proton: 216 ms -> 4.2 ms per estimate at 1280x720,
  16.6 ms at 2560x1440, with the vectors exact on synthetic motion.
- The optical-flow wait is now bounded (`FENCE_WAIT_TIMEOUT`) instead of `queue_wait_idle`;
  a stalled session turns motion off for that session instead of freezing the game.
- The "Motion units" setting was ignored (the helper read an internal field that was always
  "Pixels"); it is now honoured.
- The helper's startup log says exactly why optical flow is or isn't available.
- Removed the old layer-side motion path (disabled since 0.1.5x) and the shared-memory motion
  region and fields it used (`SHM_VERSION` 7).
- New `crates/helper/examples/optical_flow_rig_check.rs`: runs the helper's optical-flow path
  on real hardware under the helper's runner and checks the vectors and for stalls.
- Not yet measured in gameplay: whether motion vectors reduce ghosting, and their frame-rate
  cost at gameplay resolutions.

## 0.1.92 — 2026-09-24

- **New `scripts/check-stalls.sh [LOG]`** checks a layer log (default `$NEURAL_FORGE_LOG`, then
  `~/nf-layer.log`) for the bounded fence-wait timeouts and breadcrumb dumps, and summarizes
  engage/disengage transitions and the `[present]` frame rate. It exits 0 when no stalls are
  found, 1 when stalls are found and 2 when there is no log.
- Cleanup, not a bug fix: removed a dead assignment at the end of a swapchain-admission unit
  test (a leftover since 0.1.73) that caused the only compiler warning in the workspace.

## 0.1.91 — 2026-09-24

- **Removed all compatibility with the app's former names.** Releases no longer publish a
  legacy-named `NeuralForge-*` AppImage copy; `NEURALFORGE_*` environment variables are no
  longer read (use `NEURAL_FORGE_*`); the pre-0.1.77 `neuralforge` directory migration, the
  `/tmp/neuralforge-$UID` runtime link, the old `neuralforge-helper.exe` name, the legacy
  Vulkan manifest cleanup and `install.py archive-legacy-manifest` are gone. The guards that
  keep this app away from the upstream project's own files, variables and shared memory stay.

## 0.1.90 — 2026-09-24

- **Only one Neural Forge entry in the app menu.** Installing the layer for Steam games (the
  Setup tab's Install button, `neural-forge-cli install`, `scripts/install.py`) also wrote its
  own `.desktop` file, icon and AppStream metainfo pointing at the installed copy of the GUI,
  so anyone who had integrated the AppImage into their menu got a second, identical "Neural
  Forge" launcher. The install now copies only what Vulkan and the helper need; launching stays
  with the AppImage and whatever integrated it. The entry, icon and metainfo an older install
  wrote are removed on the next install, provided they are still exactly as it wrote them.

## 0.1.89 — 2026-09-23

- **GTA San Andreas – The Definitive Edition now gets the effect.** DXVK creates that game's
  swapchain with `MUTABLE_FORMAT` and a view-format list (UNORM storage, sRGB views) plus the
  Reflex latency struct. The capture admission check refused any mutable-format swapchain,
  so the game's window stayed pass-through; it renders straight into the swapchain, so there was
  no render source to tap either. The layer attached to nothing and the GUI showed no game. A
  mutable-format swapchain is now admitted when it carries the format list the spec requires,
  and the enlarged TRANSFER usage is checked against every listed view format with
  `MUTABLE_FORMAT` image creation. The copies move raw bytes, so the view formats don't change
  what is captured. Device groups and exclusive full-screen swapchains are still refused.

## 0.1.88 — 2026-09-23

- **The layer now logs the real presented frame rate every 5 s, effect on or off**
  (`[present] N fps (M/s composited by the effect)` in `nf-layer.log`). The existing
  `layer_frames` counter only counts captured frames, so it stops when the effect is off and
  could not show whether turning it off gave the frames back. First use, on the rig at 1440p
  with DLSS Frame Generation and Reflex on: GTA V runs at 37 fps with the effect on and ~100 fps
  with it off. The effect's off switch does restore the frame rate; watching over a
  remote-desktop stream hides the difference. Each presenting process gets its own line, so the
  Rockstar Social Club overlay (a separate process presenting ~50 fps, never composited) shows up
  as a second, unaffected rate.

## 0.1.87 — 2026-09-22

- **The helper's per-frame evaluate/transfer waits are bounded too, closing the gap 0.1.86 left
  open.** `crates/helper/src/frame.rs::FrameResources` reuses a single `cmd`/`fence` pair across
  every frame at a given size (no double-buffering, unlike the layer's own async slots), so
  bounding its two `wait_for_fences` calls needed more than a timeout swap: a genuine timeout now
  latches a new `stalled` flag on that instance, which `evaluate` checks before touching `cmd`
  again (refusing rather than racing whatever the abandoned wait was guarding), and which
  `matches` folds into its own answer so the normal resize-rebuild path in `main.rs` picks it up
  for free. The one new piece, `retire_frame_resources` (`main.rs`), destroys the old instance on
  a resize as before -- unless it was stalled, in which case it's deliberately leaked (a one-time,
  anomalous cost) rather than freed out from under GPU work that might still be running. Motion
  vectors' `device_wait_idle` calls (`ngx.rs`, `optical_flow.rs`) remain deliberately unbounded:
  `device_wait_idle` has no timeout in the Vulkan API at all, and motion vectors are off by
  default and untested on the rig (`docs/UPSTREAM_PARITY.md`), so that gap is real but lower
  priority. Continues v0.1.85/0.1.86; see ATTRIBUTION.md.

## 0.1.86 — 2026-09-22

- **The helper's own NGX feature-build wait is bounded too, and a breadcrumb trail now
  survives a stall.** 0.1.85 bounded every fence wait the *layer* can hit on the game's own
  present thread; this does the layer-process counterpart and adds a diagnostic. The helper's
  `CreateFeature` setup wait (`crates/helper/src/ngx.rs`) also used to pass `u64::MAX` -- bounded
  to the same 5s budget, but with its own recovery: a real timeout there means GPU work may still
  be in flight, so the setup command pool and fence are deliberately leaked (a rare, anomalous
  leak, not a routine cost) rather than freed out from under it. Several other unbounded waits in
  the helper (`device_wait_idle` in `ngx.rs`/`optical_flow.rs`, and `frame.rs`'s own per-frame
  evaluate wait) were deliberately **not** touched this pass: fixing them safely means changing a
  documented cross-file invariant ("nothing is in flight when a feature is destroyed"), not
  swapping a timeout value, and that's a bigger design decision than this pass was scoped for.
  New: `crates/layer/src/breadcrumbs.rs`, a small always-on ring of the last 64 pipeline-stage
  markers on the game's present thread (`capture::run` entry, every bounded fence wait), dumped to
  `nf-layer.log` only when one of those fence waits actually times out -- so a real stall is
  diagnosable from the log afterward instead of just "the game stopped responding". Idea from PR
  #22 against DLSS5VKLayer (bmitch87), commits `4aa730c0` and `0b41c98c`; see ATTRIBUTION.md.

## 0.1.85 — 2026-09-22

- **Every fence wait the layer can hit on the game's own present thread is now bounded (5s),
  instead of passing `u64::MAX`.** A lost device already returns `VK_ERROR_DEVICE_LOST` from an
  unbounded wait rather than hanging, so this never guarded against that; it guarded against a
  driver that stalls *without* losing the device, where the wait simply never returns, the game's
  present thread parks, and nothing gets logged because there is no result for anything to see.
  Seven call sites across `capture.rs` and `composition/gpu.rs` (the compose dispatches, their
  async-slot reuse waits, and the rare pipeline/direct-capture rebuild-on-resize drains) are
  affected; every one already had a safe fail-open return for an ordinary Vulkan error, so a
  timeout now takes that exact same path instead of blocking forever, and is logged once by site
  name if it ever actually happens. Prompted by reading PR #22 against DLSS5VKLayer (bmitch87),
  which documents the identical failure mode and fixes it the same way; a real occurrence here
  would be the first solid lead on why compositing during GTA V's loading screens has stalled the
  game in the past (`docs/UPSTREAM_PARITY.md`, "Not done yet") -- not confirmed as the cause, but
  no longer silently unrecoverable if it is.

## 0.1.84 — 2026-09-22

- **Debug views work in GTA V (and any other game that blits into its swapchain).** They used to
  be silently ignored there, gated behind a fallback path that only ever served games with no
  blit step -- the exact case GTA V isn't. All six now run through the same GPU pipeline compare
  views already use, so they get colour trust and ratio smoothing too, not just the classic
  formula. Two new views ported from upstream: colour trust's engagement (green passes whole, red
  held back) and the colour before that bound is applied, for comparing frame by frame -- plus a
  new "Debug view 5 amplification" slider (Debug tab) to make a subtle difference easier to see.

## 0.1.83 — 2026-09-21

- **Frame generation now adds frames.** "Model every Nth frame" (Model tab; header v6): at 2 the
  model runs on every other presented frame and the frames between reuse its last answer (with the
  ghost guard). With DLSS Frame Generation on in GTA V at 1440p: about 54 presented frames per
  second against 42.
- **Multipass keeps its precision.** Chains of passes go through 16-bit working images instead of
  8-bit ones between passes (falls back to 8-bit if the model refuses them).
- **32-bit games.** A 32-bit layer ships alongside the 64-bit one (`VK_LAYER_neuralforge_neural_32`,
  same `NEURAL_FORGE_ENABLE=1`), for frames up to 3840x2160.
- The launch-option builder no longer offers a DMA-BUF switch that did nothing.

## 0.1.82 — 2026-09-21

- A 0.1.81 layer refused every answer from an older helper (which does not report the size it
  answered), silently switching the effect off until the helper was updated. A helper that does
  not report is trusted again; a reported size that does not match is still refused.

## 0.1.81 — 2026-09-21

- **System Wine works as a runner.** With `runner_type=wine`, `neural-forge-cli setup` (and every
  helper start) prepares the managed prefix: DXVK's `dxgi.dll` and DXVK-NVAPI's `nvapi64.dll`
  from your binaries folder, an installed Proton, or SHA256-pinned DXVK 3.1 / DXVK-NVAPI 0.9.2
  downloads (`NEURAL_FORGE_AUTO_DOWNLOAD=0` to forbid), plus `dxvk.conf` with your GPU's IDs.
  Verified end to end under Wine 10. `doctor` checks the prefix's DLLs.
- **Answers are checked against the frame they are for.** The helper echoes the size it answered
  and the layer refuses a mismatched answer (another swapchain's, or from before a resize).

## 0.1.80 — 2026-09-21

Composition controls from upstream (section 9 of the parity work), each tested on a real GPU.

- **Compare views.** Side by side (letterboxed, zoom 1-2) or a wipe with a movable split, and swap.
  Both halves are the same frame, so this works in GTA V. Debug tab.
- **Colour trust** (default 2) bounds how far the model may shift a pixel's colour: edge fringing
  is shortened, never reversed. **Ratio smoothing** (default 100%) takes the relighting from a
  five-pixel neighbourhood, removing speckle. Composition tab.
- **Transfer modes** for a model working below 100%: classic, matched residual (default) and
  native + edit (sharpest text and edges). Classic rings at edges; the other two do not. Identical
  at 100%.
- **White meter.** With White point source set to Measured, the proxy is normalised by the frame's
  own measured white, so dark scenes reach the model well exposed. (Measured used to take the
  manual slider's value.) `shmctl status` shows the reading.
- **Frame hold.** Keeps working on one captured frame, so a settings change is judged against the
  same picture.
- **Highlights.** The composition now compares the model's answer with the frame encoded the same
  way the model saw it; comparing with the raw frame dimmed every highlight above the knee.
- **Settings survive.** Saved settings are re-applied whenever the shared memory was re-created by
  the game or the helper (a game started first, or an upgrade); they used to reset to defaults.
- The Game row shows the game's executable name while its frames are moving.
- The shared-memory header is version 5; run matching layer, helper and app.

## 0.1.79 — 2026-09-21

- **The app follows changes made elsewhere.** Every setting row updates within a second when a
  value is changed with `neural-forge-cli shmctl`, a loaded profile, Reset or another window; no
  more "restart Neural Forge to see it".
- **Per-pass settings.** A dialog (Model tab, Per-pass settings) lets each pass of the model
  override any Model value; overrides are saved (`set_pass_<n>_*` in `config.ini`).
- **Honest status.** Helper running / not running and the game's active / idle / not attached
  come from live heartbeats (a killed helper or closed game used to keep showing as running);
  the helper's reason is shown, and a version mismatch is called out. A button opens the helper
  log.
- **Upstream ranges.** Preset 0-15, intensity and local strengths 0-4, detail strength 0-4,
  highlight guard 1-30, compare zoom 1-2; model resolution is a percentage (25-100). New rows:
  rebuild spacing and ghost guard. The supersampling filter row is disabled (model resolution
  cannot exceed 100%).
- **`neural-forge-cli uninstall --purge`** also removes the config, imported DLLs, managed
  prefix, logs and `/tmp/neural-forge-$UID`.
- README: environment variable, CLI, file and troubleshooting reference;
  `docs/UPSTREAM_PARITY.md` records what was and was not carried over from upstream 0.3.1-1.

## 0.1.78 — 2026-09-21

Upstream parity (DLSS5VKLayer 0.3.1-1) and live testing on GTA V Enhanced.

- **No more ghosting.** Each frame now gets its own answer: the layer waits (at most 250 ms)
  for the model's answer to the frame it captured and composites it onto that frame. The
  pipelined mode, which applied answers to later frames and laid pale copies of old edges and
  text over moving scenes, is available with `NEURAL_FORGE_PIPELINED=1`.
- **No more freezes on launch.** Composing while GTA was on its loading screens froze the game.
  The layer now engages only after 5 s of steady rendering and steps back out on loading
  screens (ordinary stutters do not switch the effect off).
- **Start order no longer matters.** Start the helper before or after the game, or restart it
  mid-game; a helper whose heartbeat has stopped is never waited on.
- **Tuning controls work.** Style, intensity and the local strengths are applied when the
  model's feature is built (writing them each frame did nothing); changing one rebuilds the
  feature after `rebuild_settle_ms`. `PerfQualityValue` is Balanced.
- **Multiple passes.** One model feature per pass, each with its own tuning; a pass the driver
  will not build holds the chain at what fits.
- **Any resolution.** The model raster is capped at 3840x2160 pixels and always even; a failed
  build is retried when the size changes instead of disabling the model for the session.
- **Colour.** The model's hue is only trusted where it reported light, removing blue, green
  and pink specks in dark areas (upstream's hue trust).
- **Hotkey.** evdev plus XInput2 raw keys, loaded at run time (no libX11 link);
  `NEURAL_FORGE_TOGGLE_KEY` and `NEURAL_FORGE_HOTKEY_BACKEND` overrides.
- **Robustness.** Inert on non-NVIDIA devices and on a duplicate layer copy; latches off on
  `VK_ERROR_DEVICE_LOST`; drains before freeing swapchain resources; releases the primary
  swapchain claim when capture setup fails; a capture or debug request can no longer switch
  the layer off. HDR and 10-bit swapchains present untouched (SDR 8-bit only for now).
- **Helper.** Wine's `OutputDebugString` exceptions no longer count as faults; faults are
  capped and logged with module offsets; the non-unwinding `_setjmp` is used; features below
  64x64 are refused; `NEURAL_FORGE_SKIP_NVAPI` is honoured.
- **Licensing.** `project_license` is AGPL-3.0-or-later; `third_party/` notices,
  `THIRD_PARTY_CRATES.md` and `LICENSE` ship in the AppImage; the README's clean-room claim now
  matches ATTRIBUTION.md.
- The shared-memory header is version 4 (adds `ghost_guard`); run the matching layer, helper
  and GUI together.

## 0.1.77 — 2026-09-21

- **Runtime identifiers renamed to `neural-forge`.** Environment variables are now
  `NEURAL_FORGE_*` (the old `NEURALFORGE_*` spelling is still read as a fallback, by the
  layer, helper, GUI and CLI alike), the runtime dir is `/tmp/neural-forge-$UID`, config, data and
  state live in `neural-forge` directories, the install dir is `lib/neural-forge/`, the layer
  library is `libneural_forge_layer.so` and its manifest `neural_forge_layer.json`, and the log
  prefixes are `[neural-forge-layer]`/`[neural-forge-helper]`. The Vulkan layer name
  `VK_LAYER_neuralforge_neural` and the app ID are unchanged.
- **Automatic migration.** The GUI, every CLI command and `install` first rename the old
  `neuralforge` config/data/state directories (atomically, so the Wine prefix is moved, not
  copied; merged without overwriting if a new dir already exists) and rewrite recorded absolute
  paths (`config.ini`, `installation.json`, the installed manifest and `.desktop`, the Wine
  prefix registry and symlinks). A helper still running from the old layout is stopped first.
  Installing then removes the old `lib/neuralforge/` files, empty dirs and the old
  `VK_LAYER_neuralforge_neural.json` manifest.
- **Old layers keep working.** The old `/tmp/neuralforge-$UID` `shm.bin` and `shm.bin.owner` are
  hard-linked to the new ones, so a game that loaded an older layer still reaches the new helper.
- **Action needed:** the manifest now enables the layer with `NEURAL_FORGE_ENABLE=1` (the
  Vulkan loader honours only one variable). Update Steam launch options that still say
  `NEURALFORGE_ENABLE=1`; the GUI shows the new string. Restart games that were running during
  the upgrade.

## 0.1.76 — 2026-09-21

- **Rename to Neural Forge naming.** Repo, Cargo packages (`neural-forge-*`), executables
  (`neural-forge`, `neural-forge-cli`, `neural-forge-helper.exe`), the icon
  (`neural-forge.svg`) and the AppImage (`neural-forge-<version>-x86_64.AppImage`) now use
  hyphenated lowercase. Frozen on purpose: the app ID `io.github.labj1987.NeuralForge`
  (and its `.desktop`/appdata filenames), the layer `VK_LAYER_neuralforge_neural`, its
  manifest and `libneuralforge_layer.so` (under `lib/neuralforge/`), all `NEURALFORGE_*`
  variables, and the `neuralforge` config/data/state/`/tmp` paths. See CLAUDE.md
  "Naming convention".
- **Upgrading from an older install.** Installing this release removes the old
  `bin/neuralforge`, `bin/neuralforge-cli`, `neuralforge-helper.exe` and `neuralforge.svg`
  (only if unchanged since they were installed) and rewrites the desktop entry to launch
  `bin/neural-forge`. Helper lookup, stop and the layer's own-process exclusion accept
  both helper names. Anything of yours that launches the old `neuralforge` /
  `neuralforge-cli` path (shell aliases, symlinks, custom launchers) must be pointed at
  the new names. The release also publishes a `NeuralForge-<version>` copy of the AppImage
  so already-installed AppImages can still self-update by zsync.

## 0.1.75 — 2026-09-21

- **Synchronization fixes in the present path (likely relevant to the Xid 109 hangs).**
  The layer's capture/compose submissions no longer race the game's own rendering: the
  application's present wait semaphores are relayed through a wait-only batch on the
  presenting queue ahead of any layer work, and the real present waits on the layer's
  semaphore(s) instead. Render-tap source layouts are now tracked at
  `vkQueueSubmit`/`vkQueueSubmit2` time (recorded per command buffer, applied in
  submission order) rather than at recording time, so the barrier issued at present no
  longer uses a guessed `oldLayout`. A pass-through swapchain is only written to when it
  was actually created with `TRANSFER_DST`. Tap bookkeeping moved out of the per-device
  mutex (which present holds across fence waits) and the per-barrier logging was
  dropped. **Verified with unit tests and lavapipe only. This synchronization change,
  the encode and mode 2 are all still unverified on real hardware.**
- **Resource lifecycle and correctness.** The spawned helper is reaped (a helper that
  exited used to read as running forever as a zombie); the stop path checks the PID's
  command line before signaling its process group. The helper rejects zero, oversized,
  odd or unknown-format frame dimensions read from shared memory, and
  `FrameResources::new` frees everything it created on every early return (it used to
  leak VRAM on each failed retry). The shared-memory seqlock is now a real one (odd while
  writing) and `load64` no longer tears. Stopping the helper from the GUI no longer
  blocks the main thread. Installs record ownership as files land and remove files an
  older install shipped that the new one does not.
- The blocking round trip inside `vkQueuePresentKHR` is capped at one second; the hotkey
  poll runs at most every 50 ms and closes its X display; a shared and a mutable view of
  the mapped capture frame no longer coexist.
- **Packaging.** Design documents moved to `docs/`; committed SPIR-V is tied to its GLSL
  by a hash manifest checked in CI (`scripts/check_shaders.py`); `scripts/install.py`
  delegates to `neuralforge-cli` so there is a single install implementation; `once_cell`
  dropped; the release workflow runs the tests first; `appimagetool` is pinned to 1.9.1
  and checksum-verified; AppStream `<releases>` is generated from this changelog.
- The display name is now "Neural Forge" wherever it is shown to a person; internal
  identifiers are unchanged. Credits use the full name, the About dialog credits
  DLSS5VKLayer and states the license, and `ATTRIBUTION.md` now says AGPL-3.0 is
  required by upstream's own license.

## 0.1.74 — 2026-09-17

- **the layer was inert on real games, and now isn't.** Admission refused any
  swapchain whose create info carried a `pNext` chain or non-empty flags, left as a
  "validate extended creation semantics later" placeholder. Both games on the test rig
  hit it: DXVK and vkd3d-proton each attach an extension struct (sType 1000505007,
  newer than the headers this crate builds against). Admission was therefore always
  declined, the swapchain stayed pass-through, and GTA -- which renders straight into
  the swapchain image as a colour attachment rather than blitting into it -- offered no
  render source to tap either. The measured result was a layer that loaded, enabled
  itself, owned the session lease, reported healthy, tracked 793,394 barrier
  transitions and captured zero frames. Reading the frame at all needs TRANSFER_SRC on
  the swapchain, which needs admission. The chain is now walked and only structures
  that genuinely redefine the images are refused (format lists, device-group,
  full-screen-exclusive), with the same treatment for flags (protected, mutable-format,
  split-instance); anything unrecognised passes through untouched, because this module
  rewrites `image_usage` and nothing else. If the enlarged usage is rejected anyway,
  the application's own unmodified creation is retried once with `oldSwapchain`
  cleared, so admission can never cost a game its swapchain. Both 2560x1440 swapchains
  now report `pass_through=false`, on both backends.
- **Every silent "does nothing" now names itself.** Five paths used to skip compositing
  with no output at all -- a pass-through swapchain whose render source is missing or
  not in GENERAL, a swapchain that is not the session's primary claim, a presenting
  queue never seen through `vkGetDeviceQueue`, and a model the helper marked
  permanently unavailable. Each logs its reason once. This is what turned "the app does
  nothing" from a guess into a diagnosis.
- **The proxy encode is wired end to end** (`composition::encode_pass`, `encode.comp`):
  the scratch image gains STORAGE usage and a MUTABLE_FORMAT `R8G8B8A8_UNORM` view, the
  encode dispatches over it in place between the blit that fills it and the copy that
  downloads it, and the scratch is requested at every working scale including 1.0.
  `compose.comp` gains mode 2, the encoded-proxy ratio transfer: proxy and model are
  both in the encoded space, so their luminance quotient is dimensionless and only a
  broad relighting factor crosses the round trip -- which is why upstream needs no
  motion mask, and why the per-pixel mask that produced the shimmer is gone from that
  path. Mode 1 remains the fallback where the proxy cannot be encoded.
- **The encode verifies itself on real hardware.** The untouched frame and the
  GPU-encoded proxy sit in CPU memory at the same moment, so `compare_to_reference`
  checks the GPU's output against the exact arithmetic it should have performed and
  logs the max/mean delta, plus a one-shot line saying whether the encode dispatched at
  all. **Still unverified: no frame has yet been through the encode or mode 2 on real
  hardware.**
- **`scripts/rig-test.sh`**: builds both halves, deploys both atomically, forces the
  helper to launch from the installed path, runs the game, and reports fps, round-trip
  rate, timings, composition mode and any Xid -- read out of shared memory and the
  layer log rather than off a screen. Written after half the day's test rounds were
  wasted on the layer being hand-copied to the installed path while the helper ran from
  a stale AppImage.

## 0.1.70 — 2026-09-17

- **fix: real motion-vector device setup was NOT actually gated behind
  `NEURALFORGE_MVEC_HELPER`, and it hung the helper on real NVOF hardware.** v0.1.69
  claimed device creation was a no-op without the opt-in env var; it wasn't. Only the
  runtime `estimate_motion()` call checked the env var --
  `create_vulkan_context()`'s `find_flow_family()` probe, second-queue request, and
  `VkPhysicalDeviceOpticalFlowFeaturesNV`/`Synchronization2Features` chaining into
  `vkCreateDevice` ran unconditionally whenever the driver exposed an optical-flow
  queue, regardless of the env var. On real NVOF-capable hardware (confirmed live on
  an RTX 5070) this hung the helper silently after roughly a thousand frames with no
  error output, which -- because the layer calls the helper synchronously on every
  present, matching the upstream architecture this project adopted -- froze the game
  on the last composited frame at a crawl, whether the enhancement itself was toggled
  on or off. Never caught earlier because Wine (used for all pre-release testing of
  this feature) has no real NVOF-capable Vulkan driver and always reports the
  extension unavailable, so this exact code path never actually ran under test.
  Fixed in `crates/helper/src/main.rs` by gating `find_flow_family()` itself behind
  `NEURALFORGE_MVEC_HELPER`, so device creation is now genuinely byte-identical to
  pre-v0.1.69 behavior for anyone who hasn't opted in. See `docs/GHOSTING_PLAN.md` §4a.

## 0.1.69 — 2026-09-17

- **real motion vectors, built and cross-compiled, opt-in pending real-hardware
  validation.** New `crates/helper/src/optical_flow.rs`: `VK_NV_optical_flow` estimated
  between consecutive proxy frames, on the helper's own already-created device (never a
  private one, never touching the game process) -- the architecture DLSS5VKLayer's own
  AGPL-3.0 helper actually uses, confirmed by reading it directly. Scene cuts reset the
  session (CPU luma delta, independently reimplementing the same technique upstream's
  `DetectSceneCut` uses) instead of carrying a flow field across them. Feeds
  `frame::evaluate`'s existing `motion`/`reset_history` parameters, which have been wired
  and unused since before this module existed. **Correction (see v0.1.70 above): the
  claim below that device setup was fully gated behind the env var was wrong** --
  only the runtime estimation call was. Compiles clean (native and the real
  `x86_64-pc-windows-gnu` cross build, dev and release); 8 real unit tests pass under
  Wine; the actual `neuralforge-helper.exe`, run under Wine without a real NVIDIA GPU,
  starts cleanly and correctly falls back with zero effect on NGX -- which is exactly
  why Wine testing didn't catch the device-creation gap.

## 0.1.67 — 2026-09-17

- **`working_scale` is wired in, real: run the model at a fraction of the
  frame's resolution.** The earlier CPU-resample attempt (below) was measured too slow
  for the present thread and shelved; this is the GPU-blit version instead
  (`vkCmdBlitImage`, a hardware resize unit, sub-millisecond): `CapturePipeline` blits
  the captured frame down to the model's resolution before it crosses SHM as the
  proxy, and `composition::gpu` blits the helper's smaller answer back up before the
  existing, unmodified compose shader ever reads it. The full-resolution frame itself,
  and the compositor's motion-mask reference, are unaffected either way.
  `working_scale` above `1.0` (supersampling) is not wired into the compose side yet --
  only `<= 1.0` (downscaling the model's own work) runs end to end tonight. Measured
  live on `lordnikon` with `working_scale=0.75`, a real GTA session: model evaluation
  resolution 2560x1440 → 1920x1080, helper eval time (p50) **26 ms → 11.3 ms** (~2.3x),
  no Xid/driver errors, no Vulkan validation errors, layer stayed mapped into the game
  process. 61 layer tests pass (two new ones added: a real-Vulkan integration test
  confirming the proxy actually shrinks and the pipeline still composites, and a
  compose-level test confirming the upscaled answer actually reaches the shader and an
  oversized answer is safely rejected rather than overflowing the staging buffer).
  Visual correctness (does the enhancement still look right at the blit-upscaled
  resolution) still needs a live look, same as every visual check this project has
  ever needed. Full account in `docs/GHOSTING_PLAN.md` §1c.

## 0.1.66 — 2026-09-16

- **step 1/step 4 investigation, no behavior change (superseded above).** Attempted to wire
  `working_scale` (run the model at a fraction of the frame's resolution) into
  `capture::run`'s hot path via a CPU resample. Built and tested a real, separable
  resize (`composition::downscale::resample_rgba8`, finally putting the project's
  existing but previously-dead Lanczos/Catmull-Rom/Mitchell-Netravali/Kaiser kernels to
  use) — measured at GTA's real resolution it costs 315-546 ms, far worse than the
  87 ms bug fixed earlier tonight, and unusable inline on the present thread. Not wired
  in; nothing new deployed. The real fix needs a GPU blit (`vkCmdBlitImage`) inserted
  into the capture and compose pipelines instead of a CPU resize — left for a session
  where live testing on real hardware is possible. Separately investigated re-enabling
  motion vectors (`optical_flow.rs`): confirmed the documented crash trigger is a
  *second Vulkan device created from within the game's own process during its
  swapchain transition*, not motion vectors themselves being unsafe — so the
  helper-side design in `docs/GHOSTING_PLAN.md` (compute flow in the separate Windows
  helper process, which already owns its own independent device) is the right target,
  now for a verified reason. Both findings, and the corrected plan, are in
  `docs/GHOSTING_PLAN.md`.

## 0.1.65 — 2026-09-16

- **ghosting mitigation and the plan for the real fix.** Live GTA testing on
  v0.1.64 confirmed the fps fix ("great performance", 120s) with the enhancement applied,
  and ghosting still present. Tightened `compose.comp`'s `carry_delta` motion mask
  (0.005..0.032 threshold, cubic falloff) — the best of three variants tried live against
  GTA's built-in benchmark ("closer to upstream"); a variant scaling the threshold by the
  model's own edit strength made the picture worse and was reverted. This is a band-aid:
  the ghost is structural (an answer computed ~26 ms ago re-applied onto frames that have
  since moved). `docs/GHOSTING_PLAN.md` records what upstream does instead — synchronous
  per-frame presentation, the model at ~0.75 scale, optical-flow motion vectors with
  history — and the proposed steps (wire `working_scale`, a synchronous "Quality" mode,
  an explicit `DLSSNR.Reset` policy, real motion vectors, and closing the layer-deploy
  gap where AppImage updates never refresh the installed layer `.so`).

## 0.1.63 — 2026-09-16


### Phase 2

- First real GTA sessions against this pipeline (see `docs/HARDWARE_VALIDATION.md`'s
  2026-09-16 entries, including the correction at the end of the first): real
  telemetry shows `EvaluateFeature` taking a consistent ~15-25ms/frame -- the model's
  own eval cost, not transport, dominates. Real ghosting during motion; its actual
  mechanism is the deliberate "re-present the held answer every frame" design in
  `capture::run` (one answer's delta re-applied across ~8-10 real frames at native
  rate), not the motion-vector default -- motion-vector estimation has been stubbed
  out in `shm.rs` since 2026-09-14 and never ran. Alex's read of the longer session was
  "input lag and stuttering". Phase 1's own "~10% of native" fps gate is not met and
  stays open; upstream is still not validly compared (every same-day attempt was
  either accidentally still NeuralForge or crashed before gameplay).
- A repeat GTA crash (`Xid 109 CTX_SWITCH_TIMEOUT` -> `Xid 119` GSP firmware
  timeout -> full-chip GPU reset required). First read as a purely external NVIDIA
  driver bug (it is a widely reported one -- see `README.md`'s "Known issues"), but a
  real NeuralForge bug that could plausibly produce exactly this was then found and
  fixed in v0.1.61 (the render-tap source-image leak below), and a ~20-minute session
  on the fix ran clean. Treat the external-bug theory as unproven, not established.

## 0.1.60 — 2026-09-16


### Phase 6

- Migrated off the deprecated `AdwViewSwitcherTitle`/`AdwViewSwitcherBar` pairing to
  `AdwViewSwitcher` + `AdwToolbarView` + `AdwBreakpoint`, now that CI's real
  libadwaita version (1.5.0, confirmed against a real CI run's own build log) is
  past the v1.4 this needs -- closes the "Deliberately not done" item in `CLAUDE.md`.
  A real screenshot caught the first pass shipping a genuine layout bug (all six tab
  labels truncated to one character at the app's own natural window size); fixed by
  raising the breakpoint threshold, re-verified both states with real screenshots. See
  `docs/HARDWARE_VALIDATION.md`'s 2026-09-16 entry for the full detail.
- Fixed a stale GUI label: "Estimate motion vectors" said "On by default", the real
  default is off (`mvec_enabled=0`), matching `docs/PHASE1.md`'s documented baseline.

### Phase 4

- Investigated DMA-BUF transport (see `docs/DMABUF_TRANSPORT_DESIGN.md`): real hardware
  evidence (`lordnikon`, RTX 5070, driver 615.71.09) that a Wine-hosted Windows guest's
  `vkGetMemoryWin32HandleKHR` handle cannot be converted to a real Unix fd via Wine's
  own `wine_server_handle_to_fd` -- a well-formed `STATUS_OBJECT_TYPE_MISMATCH`, not a
  crash or a wrong-signature guess. This blocks the specific mechanism the protocol's
  already-reserved `proxy_pid`/`proxy_fd`/`answer_pid`/`answer_fd` fields imply, on a
  real Wine/NVIDIA-driver constraint, not a gap in this project's own code. No
  production code changed; `crates/helper/examples/dmabuf_probe.rs` (the diagnostic
  that found this) is kept for whatever's tried next.
- Investigated the reverse direction too (layer exports a dma-buf fd, helper opens it
  via `Z:\proc\<pid>\fd\<fd>`): also blocked, but for a different, more fundamental
  reason confirmed independent of Wine -- dma-buf fds are anon-inode-backed and Linux
  does not support re-opening one via `/proc/<pid>/fd/<N>` from any process (`ENXIO`),
  confirmed with a plain, non-Wine `cat`/`os.open()` before Wine was ever blamed. Both
  directions this document considered are now empirically closed, not just judged
  unlikely; a working transport would need real `SCM_RIGHTS` fd-passing over a Unix
  socket instead. No production code changed; `crates/layer/examples/dmabuf_export_probe.rs`
  and `crates/helper/examples/dmabuf_import_probe.rs` (the diagnostics that found this)
  are kept alongside `dmabuf_probe.rs`.
- Investigated the "skip Wine with a native Linux NGX helper" idea named as this
  project's longer-term direction (see `docs/NATIVE_NGX_HELPER_DESIGN.md`): NVIDIA does ship
  a genuine native Linux NGX runtime (`libnvidia-ngx.so.1`) that boots cleanly on real
  hardware with no caller-identity workaround needed, but no native Linux
  implementation of this project's target feature (DLSS 5 Neural Rendering,
  `NVSDK_NGX_Feature_Reserved18`) exists anywhere -- confirmed reserved/unallocated in
  NVIDIA's own current public SDK header, and the identical, well-known
  `FAIL_UNABLE_TO_INITIALIZE_FEATURE` result this project already recognized from the
  Windows side came back from the real native library too. A control experiment against
  a genuinely public feature (Super Resolution) confirmed this machine has no NGX
  snippet installed for anything, and a follow-up attempt to point Core at NVIDIA's own
  official redistributable Super Resolution `.so` via the documented `__NGX_CONF_FILE`
  mechanism didn't change the result either -- Core needs more than a file in the right
  directory to load a feature, not fully reverse-engineered this session. No production
  code changed; `crates/layer/examples/native_ngx_probe.rs` (the diagnostic that found
  this, including a small native `sigsetjmp`/`siglongjmp` signal guard, this project's
  Linux-native counterpart to the Windows helper's VEH-based one) is kept in the repo.

## 0.1.59 — 2026-09-15


### Phase 3

- Add protocol v3 (see `docs/PROTOCOL_V3_DESIGN.md`): a second, fully independent
  request/response wire slot, so the layer can have a captured frame already sent to
  the helper instead of idling a single wire slot while the previous answer is still
  pending. `CapturePipeline`/`DirectCapture` on the layer side and `FrameResources` on
  the helper side both become slot-indexed (one dedicated GPU resource set per wire
  slot); the single NGX feature/model stays deliberately serialized across both slots
  rather than betting on undocumented concurrent-evaluate safety for a
  reverse-engineered feature. Validated on `lordnikon`: `cargo test` (48/48, including
  a new test proving the two slots are fully independent), Khronos validation +
  synchronization validation (no new warnings versus the pre-v3 commit),
  `scripts/smoke-test.sh`, and a real running helper answering both slots correctly
  when triggered concurrently, repeatedly, in well under a second total. GTA fps
  against this change is not measured as part of this work.

## 0.1.58 — 2026-09-15


### Phase 6

- Add a Telemetry group to the Status tab: model/game/frame-rate rows and a live
  sparkline (last 5s of layer/helper-round-trip/model-eval time), all from fields
  already on `ShmHeader` -- no layer/Vulkan changes.

## 0.1.57 — 2026-09-15


### Phase 6

- First-run flow: with `nvngx_dlssnr.dll` missing, the window now opens directly on
  Setup (instead of Model) with a dismissible banner explaining why, rather than a
  silently fail-open app that never says why neural rendering isn't doing anything.
- Refresh `screenshots/` for the new Setup tab and update the README screenshot table
  and AppStream metainfo to match.

## 0.1.56 — 2026-09-15


### Phase 6

- Fix a real CI-only flake in `run_never_blocks_on_a_slow_helper_and_eventually_composites`:
  a fixed per-call timing ceiling (tuned against local/real-hardware timing) failed
  three times running on GitHub's shared runners, once on a call nearly *double* the
  helper's own simulated delay -- not evidence of a real blocking regression, just
  scheduler/software-rasterizer jitter this infra has and this machine doesn't. Now
  judges the pattern across the whole loop (few calls near the delay is noise, most
  of them is the real regression) instead of any single sample.
- Add a screenshot and the current release entry to the AppStream metainfo.
- Port `scripts/install.py`'s install/uninstall to Rust
  (`neuralforge_supervisor::install`; `neuralforge-cli install --appdir DIR` /
  `uninstall`) -- same hash-tracked, symlink-refusing, atomic-rename installer and
  `installation.json` record, confirmed to interoperate with `install.py` itself in
  both directions against a real `build-appimage.sh` output, not just a fixture.
  Runner discovery (Proton/Wine) moved from the CLI crate into
  `neuralforge-supervisor` so the GUI can reuse it too.
- Add the Setup tab: per-file NGX binaries status, a compatibility-tool picker, an
  "Install for Steam games" button (sourced from `$APPDIR` when running as an
  AppImage), and a Steam launch-option generator (target exe + DMA-BUF toggle -> the
  exact `NEURALFORGE_ENABLE=1 ... %command%` string, with a copy button).

## 0.1.55 — 2026-09-15


### Phase 1

- Rename the application and GitHub repository to NeuralForge; update binaries,
  Vulkan identity, private environment variables, paths, desktop metadata and releases.
- Isolate process ownership and add explicit game targeting; preserve upstream installs.
- Add safe installation/removal and identity-checked legacy manifest archival.
- Keep the host-transport GTA baseline; document benchmarks before later optimizations.
- Validate the host-SHM/full-model path at 2560x1440 on the RTX 5070: swapchain
  transfer usage is admitted only after capability checks, private device resources
  are released before device destruction, and present semaphores are image-scoped.
  The longer Vulkan and synchronization-validation smoke runs completed without
  validation errors. This is a correctness gate, not a GTA performance claim.
- Verify the target-process filter on the RTX 5070 with a real Vulkan test process:
  an `explorer.exe` name is excluded without advancing helper frames, while an
  explicitly targeted `GTA5_Enhanced.exe` name acquires the lease and advances them.
- Add pipeline timing telemetry and record the first GTA comparison gate: the upstream
  layer processed 4,540 frames in 61.5 seconds, while NeuralForge safely passed GTA
  through because its surface exposes `TRANSFER_DST | COLOR_ATTACHMENT`, not the
  `TRANSFER_SRC` usage required for legal capture. This is not a performance comparison.
- Add `scripts/bench.sh`, the repeatable native/upstream/neuralforge benchmark
  driver Phase 1 still needs run for real; it restarts Steam per mode and waits for a
  human to confirm the saved route before timing. Investigated the eleven Vulkan
  validation warnings Phase 1 flagged as outstanding; could not reproduce them on this
  machine's current validation-layer version under `vkcube`, so left unresolved rather
  than guessed at.

### Phase 2

- Implement the non-blocking two-slot capture pipeline (`docs/ASYNC_CAPTURE_DESIGN.md`):
  `run`'s present-hook capture submission no longer blocks on its own GPU fence --
  only a resize/queue-family change still takes a real (bounded to that rare event)
  wait. Validated on the RTX 5070 by running the layer crate's own test suite
  directly against the real driver with Khronos validation and synchronization
  validation active: 45/45 tests pass, zero synchronization hazards. Add
  `NEURALFORGE_HELPER_DELAY_MS` (test-only) to simulate a slow helper for this kind
  of validation. GTA fps has not yet been measured against this change -- `vkcube`
  cannot exercise the render tap at all (see `docs/HARDWARE_VALIDATION.md`), so this still
  needs a real session before Phase 2 can be called done.

### Phase 6

- Release Cargo profile: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`,
  `strip = true`. Measured on `libneuralforge_layer.so`: 1,774,464 -> 1,184,760 bytes
  (~33% smaller); full test suite still green.
- Add `ShmHeader::reset_persisted_settings` (`neuralforge-cli shmctl reset`, and a
  "Reset…" button on the GUI's Status tab): resets every user-tunable setting to its
  default while preserving the live helper/layer session -- seq words, status
  counters, DMA-BUF transport fields, HDR detection, motion-vector validity, the
  free-text reason/name fields. Upstream shipped a real bug here (PR #16), wiping the
  live session out from under a running process on every settings reset; a test
  (`reset_persisted_settings_changes_settings_but_preserves_the_live_session`) guards
  against reintroducing it.
- Add named settings profiles: `neuralforge-cli profile <list|save|load|delete>` and a
  "Save current as" / "Load profile" pair on the GUI's Status tab. Profiles are
  `[name]` sections of the same `set_<field>=<value>` lines `config.ini` itself
  stores, kept in their own `profiles.ini` so the flat, upstream-compatible
  `config.ini` format is untouched. `load` applies to the live session and re-snapshots
  the full setting set into `config.ini` so it survives a reboot too, the same pattern
  `reset_persisted_settings` already established.
- Move `HANDOFF_2026-09-12.md` into `docs/history/` alongside the other pre-rename
  archival record; trim `README.md` to what it does, requirements, install, usage,
  status, building and legal -- raw measurement evidence stays in
  `docs/HARDWARE_VALIDATION.md`.

### Phase 3

- Add device-extension injection (`NeuralForgeInstanceHooks::create_device`,
  see `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`): adds `VK_EXT_external_memory_host` to the
  game's own `vkCreateDevice` call when the physical device supports it and the app
  hasn't already requested it, the precondition Phase 3's zero-copy capture path
  needs. Falls back to the framework's default, unmodified path whenever there's
  nothing safe to add, and retries with the original request if the extended one is
  refused. Validated on the RTX 5070 at both 1280x720 and GTA's real 2560x1440 under
  Khronos validation + synchronization validation: injection confirmed active, zero
  hazards, no regression. The actual host-memory import this unblocks is not wired up
  yet -- this commit only adds the mechanism for getting the extension enabled.
- Implement the actual zero-copy import (`DirectCapture`, see
  `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`): a capture's `vkCmdCopyImageToBuffer` now writes
  straight into the SHM proxy region when the device extension is available, no
  staging buffer. A new test that checks captured bytes are exactly correct (not just
  "didn't crash") found two real bugs on its first real-hardware run: a missing
  `VkExternalMemoryBufferCreateInfo` on the buffer, and a misaligned allocation size --
  neither caught by this project's local software Vulkan ICD. Both fixed; the test now
  passes on both, on `lordnikon` under full synchronization validation. GTA fps still
  unmeasured -- the real payoff of this phase needs a live session.
- Add the helper-side half of the zero-copy import (`FrameResources::imported_proxy`/
  `imported_answer`, see `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`): `EvaluateFeature`'s
  Color/Output now read from and write to the live SHM regions directly when the
  device extension is available, no staging-buffer copy either direction. Confirmed
  live on `lordnikon` via a new `trigger_helper_roundtrip` tool that drives a real
  request/response round trip with no game involved.
- **Fixed two real, live bugs found only after everything above had already validated
  clean** (see `docs/HARDWARE_VALIDATION.md`'s own account): real undefined behavior in the
  device-extension-injection hook (`slice::from_raw_parts` on a pointer that's
  legitimately null when zero extensions are requested -- every earlier release-mode
  `vkcube` validation run this session had this same UB and simply never visibly
  crashed), and a real crash from `vk::ExtExternalMemoryHostFn::load(...)` panicking
  instead of failing open when a function doesn't resolve. Neither was caught by
  Vulkan validation layers or `cargo test` -- only `scripts/smoke-test.sh`'s
  debug-mode UB checker and a live `vkcube` crash surfaced them.

> Historical record: pre-NeuralForge names and deployment instructions below are
> archival, not current instructions. Do not remove or modify upstream installations.
> See docs/PHASE1.md for current paths, safety constraints and the benchmark plan.

## 0.1.30 — 2026-09-11

- **Fixed the AppImage update mechanism itself** — reported live: "I cannot update
  dlssnr using Gear Lever" (its own "check for updates" ran but reported nothing
  newer, despite this exact release existing). `build-appimage.sh` embedded
  `gh-releases-zsync|labj1987|Dlssnr|latest|...` (capital `D`) as the update-check
  info, but the real repository is `labj1987/dlssnr` (lowercase). GitHub's own
  API/web redirects resolve the case mismatch fine (confirmed directly), but Gear
  Lever's own update client apparently does not — matching the exact real-world
  symptom reported, not a hypothetical. Fixed to match the repo's real casing
  exactly, rather than relying on any client redirecting a mismatch correctly.
  Also gave `zsyncmake` a real, absolute `-u <url>` (this exact release's GitHub
  download URL) instead of letting it default to a bare relative filename in the
  `.zsync` sidecar's own internal "URL:" header — a second, separate piece of
  update metadata from `UPDATE_INFORMATION` above, used by whatever client
  actually downloads the new bytes once an update is found.
- **Separate, real mistake found and corrected the same session**: every AppImage
  rebuild from v0.1.24 through v0.1.29 was manually deployed to
  `~/AppImages/dlssnr.appimage` on `lordnikon` — a plain, unversioned file, *not*
  the one Gear Lever actually integrates and manages
  (`~/AppImages/dlssnr.appimage_0_1_25.appimage`, confirmed via its own `.desktop`
  launcher entry). The two are completely independent files (different inodes);
  none of those GUI/CLI-level fixes (the console-window fix, the orphaned-helper
  `stop()` fix) ever reached the copy the user's desktop icon actually launches.
  The game's own real rendering fixes were unaffected by this mistake — those load
  the Vulkan layer `.so` directly from a separate, already-correct path.

## 0.1.29 — 2026-09-11

- **Fixes a real bug in 0.1.28's own fix, caught testing it on `lordnikon` before
  trusting it**: `stop()`'s new `wineserver -k` call used `paths::prefix_dir()`
  directly as `WINEPREFIX`, but for `runner_type = "proton"` that's not the real
  prefix Wine itself uses — Proton's own launch script internally re-derives and
  uses `STEAM_COMPAT_DATA_PATH/pfx`. Confirmed directly against the exact orphaned
  process 0.1.28 was meant to clean up: `wineserver -k` with the bare prefix dir
  exits `1` and kills nothing; with `/pfx` appended, exits `0` and actually works.
  New `real_wineprefix` (a pure function, 2 new tests) computes the correct value
  per runner type — plain Wine has no such nesting and is unaffected.

## 0.1.28 — 2026-09-11

- **Fixed a real bug found while investigating "GTA V Enhanced has no effect and no
  performance cost"**: `dlssnr_supervisor::stop()`'s process-group kill could report
  success while the actual Wine-hosted `dlssnr_helper.exe` survived anyway, once
  wineserver took it over — Wine's own internal process management doesn't reliably
  stay inside the original `setsid()` process group. Confirmed via a real orphaned
  helper left running after a `stop()`/`start()` cycle on `lordnikon`: it kept
  writing to the same live SHM mapping as the newly-started helper, silently
  corrupting shared state (`helper_state` flapping between two independent writers,
  with no crash or error anywhere pointing at the real cause). `stop()` now also
  runs `wineserver -k` against the exact configured prefix afterward (best-effort,
  same recovery this project's own manual testing has used by hand every time this
  exact symptom came up) — closes the gap so a routine stop/restart no longer needs
  a human to notice and clean up the orphan by hand. New `wineserver_binary` helper
  (4 new tests) resolves the real wineserver binary next to whatever Proton build is
  configured.
- Fixed a real, confirmed-intermittent (not hypothetical) test race:
  `paths::tests::finds_a_real_steam_install_under_xdg_data_home` and
  `returns_none_when_no_candidate_exists` both mutated the process-wide
  `XDG_DATA_HOME` env var with no synchronization — Rust's default parallel test
  harness let them race for real (failed ~1 run in 3 under `cargo test`'s
  workspace-wide scheduling). Added a shared lock; 5/5 clean runs confirmed after.

## 0.1.27 — 2026-09-11

- **Hid the pointless console window `dlssnr_helper.exe` popped up on every real
  launch**, reported live: "it doesn't do anything." A plain Rust binary links as a
  CONSOLE-subsystem PE by default, so Wine allocates and shows one at startup even
  though this helper never had anything worth reading in it — real deployments
  always set `DLSSNR_LOG`, so its `Stderr` fallback is already unreachable in
  practice. Added `#![windows_subsystem = "windows"]` to `crates/helper/src/main.rs`
  — confirmed via `file` that the rebuilt binary now links `(GUI)` instead of
  `(console)`, matching upstream's own compiled helper's subsystem type.

## 0.1.26 — 2026-09-11

- **Fixed a real red/blue channel swap affecting every real game session**,
  reported live as "flickering... doesn't look like the game" while running
  GTA San Andreas with DLSS 5 NR on. Root cause: `vkCmdCopyImageToBuffer`/
  `vkCmdCopyBufferToImage` are raw, format-preserving byte copies — they never
  reorder channels — but this project's own code (`swapchain::proxy_format_for`,
  the composition math in `composition/apply.rs` and `shaders/compose.comp`,
  and the debug PNG dump) all hardcoded an `R,G,B,A` byte order regardless of
  the swapchain's real format. This machine's real swapchain (and this game's)
  is `B8G8R8A8_UNORM` — confirmed via real `vkcube`/game testing, not
  hypothetical — so every captured/composited/presented byte had red and blue
  silently swapped, visible as a uniform blue tint across real captured
  frames. Fixed by threading a real `bgr_order: bool` (from
  `swapchain::is_bgr_order`, new) through the whole capture/composition
  pipeline — both the CPU path (`apply_rgba8`) and the GPU compute path
  (`compose.comp`, recompiled with a new `bgr_order` push constant) now read
  and write through the correct channel indices. A real, deliberately strict
  regression test (`bgr_order_produces_the_same_true_colors_as_rgb_order_on_swapped_bytes`)
  verifies the *true* colors this produces match an independently-computed
  RGB-order reference exactly, not just that the GPU and CPU paths agree with
  each other (which they could do while both being equally wrong).
- Corrected two doc comments in `composition/gpu.rs` that had asserted a
  buffer-mediated write-back is "correct regardless of the real image format" —
  true for size/layout compatibility, false for channel order, which is
  exactly what caused the bug above.
- **Fixed a second, unrelated real bug found while investigating the above**:
  the GUI's "Enabled" toggle (`ShmHeader::enabled`, read via
  `ShmHeader::neural_enabled()`) was never actually checked by the capture
  path at all — turning it off in the GUI had no effect on anything. Now
  gates `capture::run` the same way `apply_model` already does.
- The NGX model's own `DLSSNR.Color`/`.Output` Vulkan resources
  (`crates/helper/src/frame.rs`) are still hardcoded `R8G8B8A8_UNORM`
  regardless of the real captured format — a real, separate, deferred gap:
  the model itself may still receive/produce mislabeled color data on a BGR
  swapchain, independent of the fix above (which corrects what the *layer*
  captures, composites, and presents). Not yet fixed; see CLAUDE.md.

## 0.1.25 — 2026-09-11

- **Real DLSS 5 Neural Rendering works again, end to end, for the first time since
  the app-removal/reinstall that broke it.** A real `vkcube` run on `lordnikon`
  showed `model_up=1`, `helper_frames=128` over a 10-second run, real
  `VULKAN_CreateFeature(18) -> 0x1`, and `EvaluateFeature -> 0x1` on essentially
  every frame. Two separate real bugs were found and fixed to get here — see
  CLAUDE.md's updated NGX section for the full writeup.
- **Fixed the `0xbad00002` (`FAIL_PLATFORM_ERROR`) NGX blocker.** A real
  side-by-side comparison against upstream's own compiled helper (recovered from its
  official GitHub release, `bmitch87/DLSS5VKLayer` `0.2.6-1`) showed upstream hits
  the identical rejection from Core's own allocator in this exact environment, and
  recovers with a self-implemented, in-process `NVSDK_NGX_Parameter` object instead
  of the DLL's own `AllocateParameters`. New `crates/helper/src/selfparam.rs`
  implements the same recovery: a real `NVSDK_NGX_Parameter`-shaped vtable object
  backed by a plain `HashMap`, used whenever the DLL's own allocator fails or isn't
  exported. Also removed an unproven `NVSDK_NGX_VULKAN_Init_ProjectID` call added
  earlier the same day — real testing showed it wasn't the cause of anything.
- **Fixed a second, separate, pre-existing bug found while verifying the first
  fix**: `capture::run`'s first line checked `shm.composition_settings()`, which
  only reads through an already-open SHM mapping — but nothing opens the mapping
  until *later* in the same function, gated behind that same check. On a brand-new
  process the mapping was never open, so the function always bailed out before ever
  capturing a single frame, on every present call, silently. Fixed by calling
  `shm.open()` (already idempotent) unconditionally as the function's first line.
- Fixed a related logging bug: the layer's per-frame log-flush throttle meant a
  `vkcube` process killed by `timeout`'s default `SIGTERM` lost every buffered log
  line since the last flush, including one-time startup milestones. Added explicit
  `logging::flush()` calls after device/swapchain creation (one-time events, not the
  per-frame hot path the throttle protects).
- Full workspace test suite stays green throughout (33 layer tests, full helper/
  supervisor/protocol suites).

## 0.1.24 — 2026-09-11

- **Real progress on the NGX `FAIL_PLATFORM_ERROR` (`0xbad00002`) blocker documented
  in `HANDOFF_NGX_PLATFORM_ERROR.md`**, though the underlying rejection itself is
  still not resolved. Removed the `NVSDK_NGX_VULKAN_Init_ProjectID` call added late
  in 0.1.23's session: upstream's own reverse-engineered notes show the only
  *proven* ProjectID-based init route is `NVSDK_NGX_D3D12_Init_with_ProjectID`
  against a dedicated D3D12 device — a different API family than this helper's
  Vulkan device — and the Vulkan-exports route this helper actually used was never
  exercised by upstream at all, just present as an export. A real bisection on
  `lordnikon` this session tested and ruled out the leading theory that this failed
  call was "poisoning" every later NGX call in the process: with the call removed
  entirely, `AllocateParameters` still returns the identical `0xbad00002`.
- **Real discovery, not a guess**: `nvngx_dlssnr.dll`'s (the "snippet") own
  `NVSDK_NGX_VULKAN_*` exports are PE forwarders straight into `nvngx.dll` (Core) —
  confirmed by deliberately keeping Core unloaded and observing
  `NVSDK_NGX_VULKAN_AllocateParameters` fail to resolve via `GetProcAddress` on the
  snippet either, with the identical wording already seen earlier in this
  investigation when `nvngx.dll` was genuinely missing from the machine. This means
  there was never a real "prefer snippet vs. prefer Core" choice for this call
  family — both always run the exact same Core code. The `0xbad00002` is a real
  rejection from Core's own NGX runtime, reproduced identically against both a
  freshly-recreated Wine prefix and the real, untouched, previously-working
  GTA San Andreas Proton prefix — ruling out an environment/prefix cause.
  `crates/helper/src/ngx.rs` now documents this clearly so it isn't re-investigated
  from scratch.
- Fixed a real incident this session: a manual diagnostic run against the game's own
  prefix (intended read-only) used a different Proton build than that prefix was
  actually associated with, triggering Wine's automatic version-upgrade path and
  leaving a stale `wineserver` holding the prefix locked, which then blocked the
  game from launching normally through Steam. Root-caused and fixed by killing the
  orphaned `wineserver`/`winedevice.exe`/`xalia.exe` processes still bound to that
  prefix; the prefix's own registry rewrite that came with the version bump appears
  otherwise harmless (Wine's builtin-DLL regeneration preserves app-added keys), and
  the game was confirmed launching again afterward.
- Remaining real next steps for the NGX blocker, in priority order: recover
  upstream's real compiled C++ helper binary (still installed as a system package
  before this session, `apt-cache policy`/`.deb` cache may still have it) and run it
  side-by-side against the exact same prefix/game session for a real reference
  comparison; failing that, a real debugger (`winedbg`) attached at the point of the
  `AllocateParameters` call to see exactly which internal Core check rejects it.

## 0.1.23 — 2026-09-10

- **Fixed a real crash-on-start after a from-scratch reinstall** (found immediately
  after reinstalling on `lordnikon` following the full-removal test): `start()`
  passes `paths::prefix_dir()` to Proton as both `WINEPREFIX` and
  `STEAM_COMPAT_DATA_PATH`, but nothing ever created that directory -- invisible on
  every normal run (Proton creates everything *inside* it on first successful init,
  so it's already there afterward), but fatal the moment it doesn't exist yet:
  Proton's own `setup_prefix()` fails opening `pfx.lock` with a `FileNotFoundError`,
  and the helper never starts. Fixed in two places for robustness: `start()` itself
  now creates the directory directly before building the runner's environment, and
  `paths::ensure_dirs()` (already responsible for config/data/state/binaries) now
  includes it too.

## 0.1.22 — 2026-09-10

- **The real architectural fix for the low-FPS problem, not another instrumentation
  pass.** Removing this project's layer entirely and re-testing confirmed the game
  itself and the GPU are completely healthy (99% GPU utilization, normal framerate,
  real clocks/power draw) -- the true cause was `capture::run` (the old function,
  renamed `run_sync` and now used only for `debug_view`/`capture_request`) blocking
  every single `vkQueuePresentKHR` call on a full helper round trip (real per-frame
  cost on `lordnikon`: ~100-150ms, a cross-process, Wine-hosted IPC call that can
  never be as fast as native in-process DLSS), capping the game's own presentation
  rate at the round trip's rate no matter how cheap the actual GPU work involved
  actually was.
- Rewrote the hot path (`capture::run`) as a real pipeline: captures and sends a new
  frame only when no round trip is already in flight
  (`ShmClient::has_pending_request`), polls any in-flight one without ever blocking
  (`ShmClient::begin_async_request`/`poll_async_request`, new), and applies whatever
  answer arrives to whichever frame happens to be current at that moment via the
  existing `dispatch_into_image_async` fast path. Every frame that isn't a capture or
  a fresh-answer frame (the large majority, once the pipeline is running) touches
  `image` not at all and returns immediately. Explicit, deliberate tradeoff (per
  Alex's own prior authorization, "do it if it gives us the most frames when NR is
  on"): NR visibly updates at whatever rate the round trip achieves, not every
  frame, and can be composited against a slightly newer frame than the one it was
  computed from -- a real quality cost, in exchange for the game's own rendering and
  presentation no longer being held hostage by a cross-process round trip on every
  single frame.
- Fixed a real, independently-necessary Vulkan layout bug this redesign exposed:
  `composition::gpu::GpuCompose`'s `record_copy_into_image` assumed its target image
  was already `TRANSFER_DST_OPTIMAL`, true only because the old code always ran a
  full capture (which left it there as a side effect) immediately before compositing
  in the same frame. Now that captures and composites can land on different frames
  entirely, that assumption no longer held. Fixed by having the function manage its
  own `PRESENT_SRC_KHR -> TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR` round trip
  unconditionally, matching the same state any real swapchain image is already
  guaranteed to be in.
- New tests: `ShmClient`'s non-blocking request API (answers-in-time and
  times-out-without-blocking cases), and a real end-to-end
  `capture::run_never_blocks_on_a_slow_helper_and_eventually_composites` test against
  a live (if software) Vulkan device with a deliberately slow fake helper -- asserts
  every individual `run()` call stays fast regardless, and that a real composited
  result still eventually lands. 33/33 layer tests pass; full workspace build and
  test suite clean.

## 0.1.21 — 2026-09-10

- **Root-caused the remaining ~350ms/frame stall to the hardware level on
  `lordnikon`.** 0.1.20's completed timing breakdown (stage1 ~5ms, snapshot ~78ms,
  write_proxy ~68ms, roundtrip ~128ms, compose ~72ms, all summing correctly to the
  ~355ms total) showed every full-frame-sized (33MB at 4K) operation costing a
  similar ~70-130ms regardless of what it actually was — a heap copy, a write into a
  shared mapping, a cross-process round trip, a GPU dispatch launch. Ruled out disk
  I/O (`/tmp` is genuine RAM-backed `tmpfs`, `Dirty`/`Writeback` near zero, no swap
  used), THP/compaction stalls (`enabled=madvise`, `thp_fault_alloc=0`,
  `compact_stall` static across a 3s sample), and a remote-desktop encoding
  bottleneck (confirmed a real physical HDMI/TV output at 3840x2160@144Hz VRR,
  `is-current=true`, on `seat0`/`tty2` — not primarily a remote session). `perf stat`
  on the live game process during actual play showed the real number: **97.0%
  backend-bound, IPC 0.1** — the CPU is stalling on the memory subsystem almost the
  entire time, a genuine low-level hardware/memory-bandwidth condition, not a logic
  bug in this codebase. Not something a code change can fix outright.
- **Fixed the one clear, unconditionally-correct waste found along the way**:
  `capture::run` allocated a fresh ~31.6MiB `Vec` from scratch every single frame
  (`captured.to_vec()`) just to snapshot the pre-edit frame for composition, instead
  of reusing one. Added `original_scratch: Vec<u8>` to `device.rs`'s per-device
  `State` (already `#[derive(Default)]`, already threaded through
  `queue_present_khr` alongside `shm`/`capture`/`gpu_compose`), reused via
  `clear()` + `extend_from_slice()` every frame. Doesn't fully explain the
  backend-bound stall above (a `perf stat` sample at the same time showed the same
  severe stall even accounting for this), but removes a real, unnecessary
  allocation from the hottest path in the codebase regardless.

## 0.1.20 — 2026-09-10

- **Closed a ~150ms/frame gap in `capture.rs`'s own timing instrumentation.** 0.1.18's
  stage1/roundtrip/compose/stage2 timing summed to only ~208ms on `lordnikon` against
  a measured ~355ms total -- real data, but with an unaccounted gap exactly where two
  full-frame-sized (33MB at 4K) operations sat untimed: `captured.to_vec()` (a fresh
  heap allocation + copy of the whole frame, taken so composition has an unedited
  reference after `read_answer` overwrites the original in place) and
  `ShmClient::write_proxy` (a second full-frame copy into the shared-memory region).
  Added `t_snapshot`/`t_write_proxy` timing around both and included them in both
  timing log lines, so the per-frame timeline is now fully accounted for with no
  remaining gap -- next real capture should show exactly which of these two copies
  (if either) is the actual ~150ms cost, rather than leaving it as an inferred gap.

## 0.1.19 — 2026-09-10

- **Found the real, syscall-verified cause of the per-frame stall via `strace`,**
  after 0.1.16-0.1.18's buffered-file-logging fixes made no measurable difference
  on `lordnikon` (a separate deploy gap meant those builds were never actually
  running in the game at all — see below). `strace -e trace=write` on the live game
  process showed a single `crate::log!()` call in `dlssnr_layer::logging` (called
  once, now twice with 0.1.18's added timing line, per frame from inside the game's
  own `vkQueuePresentKHR` override) fragmenting into several separate blocking
  `write()` syscalls against a redirected pipe — because `DLSSNR_LOG` is only ever
  set for `dlssnr_helper.exe` (by `dlssnr_supervisor::start()`), never for the
  game's own Steam-launched environment, so the layer's sink has always silently
  fallen into the un-wrapped `Stderr` branch in every real deployment, completely
  bypassing 0.1.17's `BufWriter` fix (which only wrapped `Sink::File`). Fixed by
  wrapping `Stderr` in `BufWriter` too, in both `dlssnr_layer::logging` (the actual
  cause here) and `dlssnr_helper::logging` (same latent gap, fixed for consistency
  even though the helper has always had `DLSSNR_LOG` set in practice).
- **Documented a separate, real deploy gap in `CLAUDE.md`** that silently
  invalidated the 0.1.17 and 0.1.18 field tests on `lordnikon`: the game loads the
  layer from a real, fixed path (`~/.local/share/dlssnr/lib/libdlssnr_layer.so`,
  referenced by a real Vulkan implicit-layer manifest under
  `~/.local/share/vulkan/implicit_layer.d/`) that was set up by hand earlier this
  session and that nothing in `dlssnr-gui`/`dlssnr-cli`/`build-appimage.sh` ever
  installs or refreshes — rebuilding and redeploying the AppImage alone does not
  update it. Both prior "restart the game and re-measure" tests silently re-ran the
  stale pre-fix `.so` the whole time. Flagged as an open item: this path should
  either become self-installing/self-updating, or be replaced entirely once it's
  understood why `VK_ADD_LAYER_PATH` (which `AppRun` does set, correctly, for
  `dlssnr-gui`'s own process tree) can't reach a Steam-launched game process, which
  runs as a fully separate process tree.

## 0.1.18 — 2026-09-10

- **Fixed the GUI's "Layer: not attached" status being wrong 100% of the time,
  regardless of whether the layer was actually attached.** `ShmHeader::layer_attached`
  (and every other `layer_*` telemetry field: `layer_heartbeat`, `layer_frames_lo/hi`,
  `layer_width`/`layer_height`/`layer_format`) was declared, reset to 0 by
  `init_defaults`, and read by `crates/gui/src/ui.rs`'s status row — but nothing
  anywhere in `dlssnr-layer` ever wrote `1` to it (confirmed by grep). Found while
  investigating a real report of "still low fps, also still saying layer is not
  attached" on `lordnikon`: `/proc/<pid>/maps` and a live, steadily-advancing helper
  frame counter both proved the real layer was loaded and actively processing frames
  the whole time the GUI displayed "not attached" — the status readout, not the
  pipeline, was broken. Fixed by having `ShmClient::set_frame_info` (already called
  once per frame from `capture.rs`) write all of these fields every frame, not just
  once at `open()` — which also matters because `open_at`'s own idempotent early
  return means a layer that was already attached before a helper restart resets the
  shared header never calls it again to re-set a one-shot flag.
- **Added per-stage timing to the layer's own capture path** (`capture.rs::run`):
  stage 1 (image→buffer GPU readback), the full helper round trip, composition
  (CPU reference or GPU dispatch, sync or async), and stage 2 (buffer→image
  write-back) each logged separately per frame. The 0.1.16/0.1.17 investigation
  measured the *helper*'s own GPU work at ~3ms/frame and ruled out logging I/O as
  the cause of the real, still-unexplained ~350ms/frame gap after buffering fixed a
  real (separately confirmed via 93%+ iowait on one CPU core) but apparently
  non-dominant I/O bug in both crates' loggers — this closes the one remaining
  unmeasured segment of the pipeline: the layer's own native-side Vulkan work,
  which nothing before this could distinguish from "the game's own render time" by
  external observation (GPU utilization, iowait) alone.

## 0.1.17 — 2026-09-10

- **Found and fixed the real cause of the low-FPS reports on `lordnikon`**: the
  0.1.16 timing instrumentation showed real GPU work costing only ~3ms/frame, yet
  the measured frame interval was ~357ms, with the GPU sitting at 2% utilization
  and one CPU core pegged at 92% iowait — a pure I/O stall, not a compute one. Root
  cause: `dlssnr_helper::logging`/`dlssnr_layer::logging`'s sink wrote to a plain
  `std::fs::File`, which has no internal buffering — every single `crate::log!` call
  (2-3 times per frame in each crate's per-frame hot path) issued its own raw OS
  write syscall, immediately followed by an explicit `.flush()`. On the helper side
  specifically, that syscall runs inside a Windows-guest binary under Wine/Proton,
  where a single such write against a real host-filesystem path measured at
  ~150-180ms — accounting for essentially the entire missing frame time on its own.
  This was already present in every prior release, not something 0.1.16 introduced.
  Fixed by wrapping both sinks' file handle in `BufWriter` and flushing only every
  64th call instead of every call, in both `crates/helper/src/logging.rs` (the
  Wine-hosted side, the dominant cost) and `crates/layer/src/logging.rs` (the native
  Linux side, same anti-pattern, much cheaper per-call but still a real syscall on
  every presented frame for no reason). The layer-side fix only takes effect on the
  next game relaunch (the `.so` is already loaded in the running game process); the
  helper-side fix takes effect on helper restart alone.

## 0.1.16 — 2026-09-10

- **Diagnosing real low-FPS reports on `lordnikon`** (GTA San Andreas – The
  Definitive Edition, real RTX 5070, upstream's conflicting package/layer fully
  purged this session): confirmed via `/proc/<pid>/maps` that this project's own
  `libdlssnr_layer.so` (not upstream's) is loaded directly inside the game's real
  Vulkan process and has the SHM buffer mapped — the layer/helper wiring itself is
  correct. The helper's own log showed real `NVSDK_NGX_Result_Success` answers every
  call, but only ~2.8 evaluated frames/sec at 3840x2160 — each `queue_present_khr`
  blocks on a full synchronous round trip to the helper (`shm.rs::try_round_trip`),
  and the helper's own per-frame path (`frame.rs::evaluate`) does three sequential,
  fence-blocking GPU round trips (CPU→GPU upload, `EvaluateFeature`, GPU→CPU
  download) with no per-stage timing to show which one actually dominates. Added
  `Instant`-based timing around all three stages, logged per frame
  (`[frame] timing upload=... eval=... download=... total=...`), to get real numbers
  instead of guessing before attempting any fix — a guess here risks "fixing" the
  wrong stage in a closed, three-layer-translated (Windows guest → Wine → native
  Vulkan) NGX call this project can't step through with a debugger.

## 0.1.15 — 2026-09-10

- **Fixed a real, 100%-reproducible crash on every single helper start attempt**
  with `runner_type = "proton"`: `dlssnr_supervisor::start()` never set
  `STEAM_COMPAT_CLIENT_INSTALL_PATH`, which Proton's own launch script reads directly
  out of the environment with no fallback (`KeyError` otherwise) during its own
  prefix setup, before ever getting to run the helper .exe. Found running a real game
  (GTA San Andreas – The Definitive Edition) on `lordnikon` — every manual SSH test
  this project's own history has done set this by hand for exactly this reason, but
  the fix never made it back into the actual production code path the GUI's "Start"
  button and `dlssnr-cli start` both use. Added `dlssnr_supervisor::paths::steam_install_dir`
  (checks the same native/Flatpak/Snap candidates `dlssnr-cli`'s own Proton discovery
  already scans, picks the first that's a real directory) and wired it into `start()`.
  2 new tests.
- Also found (not a code bug, a real system-configuration conflict, not touched
  without asking): a systemd `~/.config/environment.d/dlssnr.conf` — almost certainly
  left behind by upstream's own installer — globally forces
  `VK_INSTANCE_LAYERS=VK_LAYER_NV_dlssnr:...` for every Vulkan app in the session,
  which very plausibly explains a real game showing this project's own layer as "not
  attached" (upstream's real layer is what's actually active) and could itself cause a
  real performance hit if both layers end up running their own separate neural
  rendering pass on the same game simultaneously.

## 0.1.14 — 2026-09-10

- **Cross-frame async pipelining** (`GpuCompose::dispatch_into_image_async`, new):
  submits the compute dispatch + write-back with a signal semaphore and returns
  immediately instead of blocking on its own fence. `capture::run` now returns
  `Option<vk::Semaphore>`; `device.rs`'s present hook chains it into the real
  `vkQueuePresentKHR` call's own wait-semaphore list (combined with the app's own,
  never replacing them) so the presentation engine — not our code — waits for the GPU
  work before displaying the frame. Explicitly authorized: "do it if it gives us the
  most frames when NR is on."
- Genuinely double-buffered (`ASYNC_SLOTS = 2`, independent images/staging
  buffer/command buffer/fence/semaphore each) to avoid a real data race between two
  in-flight dispatches; the only wait is on a slot's *own* fence from its *previous*
  use, immediately before reuse, never before returning the current result. Verified
  the binary-semaphore reuse discipline is sound both by reasoning (a wait is always
  chained into that same frame's present call before the same slot could ever be
  reused) and by a new test driving it across many more iterations than there are
  slots, against a real device.
- Deliberately did not add reprojection/stale-answer tricks to chase a bigger win:
  without real motion vectors, that would cause real visible ghosting on moving
  content. Every frame's presented image still comes from that same frame's own
  capture and model answer — only *when* the CPU learns the work is done changed.
- Verified thoroughly on real hardware before trusting it: a short run first
  (watching specifically for hangs/crashes), then a real 10s measurement, then a real
  `capture_request` dump (still visually correct), then a 45-second/601-frame stress
  run to rule out a slot-reuse issue only surfacing after many cycles. Zero crashes,
  zero hangs, zero fallbacks except when a capture_request was genuinely pending.
- Real gain: 143 frames/10s, up from 128 (~134/10s sustained over 45s). Documented
  this is likely close to the practical ceiling for the current architecture — the
  remaining gap to the 244-frame no-composition baseline is real GPU bandwidth/work
  volume, not something more scheduling cleverness can remove.
- 3 new tests, 30 tests in this crate now, full suite green.

## 0.1.13 — 2026-09-10

- **Merged GPU compose + write-back into one submission** (`GpuCompose::dispatch_into_image`,
  new): writes the composited result straight into the real swapchain image, in the
  same command buffer as the compute dispatch, when nothing needs to see the bytes on
  the CPU (no pending `capture_request` dump — checked via a new non-consuming
  `ShmClient::capture_request_pending`). Two GPU submissions per frame instead of
  three, no CPU round-trip for the composited bytes in the common case. Deliberately
  routes through an intermediate buffer rather than a raw image-to-image copy, since
  the latter would silently corrupt colors if a real swapchain's format ever differs
  from this module's own hardcoded format (nothing here can vary/test that in this
  environment) — a buffer has no format attached, so the final copy always targets the
  real image's own true format, exactly like the existing stage 2 it replaces.
- Verified byte-for-byte identical to the already-verified separate-dispatch path by a
  new local test, before ever touching real hardware. Verified correct and measured on
  real hardware after: every frame took the fast path except the one a real
  `capture_request` was pending for, which correctly fell back and still dumped a
  visually correct frame. Real gain: 128 frames/10s, up from 118.
- Documented what this reveals: the no-composition baseline already does the same
  number of submissions per frame, so the larger remaining gap (244 vs. 128) is real
  GPU bandwidth/work-volume, not submission count — genuinely closing it further needs
  cross-frame pipelining, which trades in a frame of latency and is a real product
  decision, not attempted here.
- 1 new test, full suite green.

## 0.1.12 — 2026-09-10

- **`shaders/compose.comp` is now really dispatched on the GPU** (`crates/layer/src/composition/gpu.rs`,
  new), tried first in the write-back whenever `debug_view == 0`, falling back to the
  CPU path otherwise. The shader itself moved from `rgba16f` to `rgba8` storage images
  with explicit sRGB decode/encode added (storage-image loads never apply an sRGB
  curve regardless of format), precompiled to SPIR-V and embedded via `include_bytes!`.
- **A real shader bug was found and fixed before it ever touched real hardware**, by a
  new local test that needs only a software Vulkan ICD (lavapipe): `OklabFromLinearSrgb`
  multiplied a matrix by `sign(lms)` before the cube root instead of after, due to GLSL
  operator precedence — diverged from the CPU reference by up to 90/255 with the
  default `colour_strength = 1.0`. Fixed and reverified.
- Verified correct on real hardware after the fix (same real, structured composition
  effect as the CPU path). Real, honest performance finding: 118 frames/10s on the GPU
  vs. 97 for multi-threaded CPU and 244 for no composition — a real but modest gain,
  since the current synchronous one-submit-one-wait-per-frame pattern (three CPU-GPU
  round trips per frame now) is the real remaining cost, not the shader itself.
  Pipelining that is genuinely still-open work.
- 2 new tests (GPU-vs-CPU parity, resize handling), both skip gracefully with no
  Vulkan ICD present rather than breaking the build. Full suite green.

## 0.1.11 — 2026-09-10

- **Expanded the GUI settings surface**: a new "Compare and debug" group
  (`compare_mode`, `compare_split`, `compare_zoom`, `compare_swap`, `debug_view`) and
  five new Composition rows (`colour_mode`, `transfer`, `unlock_passes`,
  `apply_model`, `hold_frame`) — all real, tested, and verified by screenshot. Still
  not bound: white-point HDR tuning and the raw hotkey (`toggle_key`), which needs a
  proper key-capture widget.
- Extended `ShmHeader::persisted_settings`/`apply_persisted_setting` from 21 to 31
  entries so every new row actually survives a reboot through `config.ini`, not just
  appears to save.
- Found and fixed a real bug while verifying the above by actually running the GUI:
  `AdwPreferencesGroup::title` is parsed as Pango markup, and naming the new group
  "Compare & debug" broke it outright. Renamed to "Compare and debug".
- `dlssnr-cli shmctl` now covers all 31 persisted settings (previously 21, with
  `debug_view`/`apply_model`/`compare_mode`/`hold_frame` handled as a separate
  non-persisted special case — folded into the main list now that the GUI needs them
  to persist too).

## 0.1.10 — 2026-09-10

- **`dlssnr-cli shmctl`** (`crates/cli/src/shmctl.rs`, new): the real equivalent of
  upstream's separate `dlssnr-shmctl` debug/introspection tool this project had none
  of before — `status` (all 21 persisted settings plus live `helper_state`/`model_up`/
  `helper_frames`/`debug_view`/`apply_model`/`compare_mode`/`hold_frame`/
  `capture_request`), `set <name> <value>`, `toggle <name>`, and `capture [view]` (the
  same real frame-dump this project used to first visually verify its composition
  output). 10 new tests on the pure resolve/store/toggle logic; full suite green.

## 0.1.9 — 2026-09-10

- Real `ShmHeader::capture_request` support (`crates/layer/src/dump.rs`, new):
  writes a matched before/after PNG pair on request. Plus
  `crates/protocol/examples/trigger_capture.rs`, a small manual tool to trigger one
  (and optionally set `debug_view`) against a running instance from the outside.
- Three real bugs found (via `strings` against the real `nvngx_dlssnr.dll`) and fixed
  in the helper's NGX evaluation, all independently justified, none the actual cause
  of what they were found while chasing (see below): `DLSSNR.Depth`/`DepthInverted`
  and every resource's `*Subrect*` scalar were never bound; `DLSSNR.Reset` was set
  once at creation and never toggled to 0 for subsequent frames; the device extension
  list was copied from a native-Linux reference binary's own strings output without
  adjusting for this crate's actual Windows/Wine platform
  (`VK_EXT_external_memory_dma_buf`/`VK_KHR_external_memory_fd` -> `VK_KHR_external_memory`/
  `VK_KHR_external_memory_win32`; device extension count went 6/9 -> 8/9).
- Found and fixed the actual bug behind an initially-alarming solid-white dumped model
  answer: it was this session's own new PNG dump tool passing through a real answer's
  alpha channel (0 across the whole image) unmodified, which a PNG viewer renders as
  blank/transparent — not a broken model output. A real opaque-composite-mode present
  never reads alpha at all, so this could never have affected an actual displayed
  frame. Fixed by forcing alpha to 255 before encoding.
- **First real visual confirmation the full pipeline produces correct output**: a
  pixel diff between a real captured frame and its real composited answer shows a
  mean per-channel difference of ~17/255 across 100% of sampled pixels — real,
  structured work, not a no-op or garbage. Full writeup in CLAUDE.md.

## 0.1.8 — 2026-09-10

- **This project's own composition math now actually reaches the presented frame**
  (`crates/layer/src/composition/apply.rs`, new): a CPU port of `shaders/compose.comp`'s
  pipeline, built on the already-tested `upgrade_tone_map`/`gamut_compress_reversible`
  functions, wired into `capture.rs`'s real write-back for the `RGBA8` proxy format.
  `debug_view` (composited/original/raw-answer/amplified-diff) and `apply_model`'s
  off-switch are both real now, read live from the SHM header every frame.
- Real, measured performance finding: the naive single-threaded version cost ~800ms/
  frame at 1080p on real hardware (244 -> 11 frames in a real 10s `vkcube` run).
  Parallelized the (fully independent, per-pixel) work across threads —
  ~97 frames in the same real 10s run, an ~8x measured improvement. Still short of the
  244-frame no-composition baseline; real GPU dispatch of `compose.comp` remains the
  actual fix for game-ready framerates and is still open work.
- 5 new tests for `apply_rgba8`'s real invariants; full suite (30 tests) still green.
  Verified on real hardware: round trip still succeeds every frame with composition
  active.

## 0.1.7 — 2026-09-10

- **First confirmed real DLSS 5 Neural Rendering success from this project's own
  code.** With the 0.1.6 crash fix in place, ran a fresh `dlssnr_helper.exe` under real
  Proton + real `vkcube`/`VK_LAYER_dlssnr_neural` (implicit activation) against the
  real, legitimately-signed `nvngx_dlssnr.dll` on `lordnikon`. Result: `VULKAN_CreateFeature(18)
  -> 0x1` (real non-null handle, 1920x1080) and `EvaluateFeature -> 0x1` on 243/244
  captured frames (the one miss is frame 1, before `CreateFeature` had run) — zero
  evaluation failures across a full 10-second run. This is the exact success shape
  previously only ever produced by upstream's C++ build. Doesn't yet prove the visual
  output is correct (composition math still isn't wired in, real optical flow still
  isn't) — see CLAUDE.md's new "First confirmed neural-rendering success" section for
  exactly what this does and doesn't close.
- Repo visibility changed to private.

## 0.1.6 — 2026-09-10

- **Fixed the critical layer crash documented in 0.1.5**: `VK_LAYER_dlssnr_neural`
  segfaulted 100% of the time under real, implicit activation (`VKLayer_DLSS5=1`) in
  the presence of Mesa's `device_select` implicit layer. Root cause, found by
  reproducing the crash locally against Google's own pristine, unmodified
  `vulkan-layer` `hello-world` example (same crash, same backtrace — proving this was
  never a dlssnr-specific bug): `vulkan_layer::Global::create_instance`'s default
  fallback path eagerly resolves all three Vulkan 1.0 global entry points
  (`vkCreateInstance`/`vkEnumerateInstanceExtensionProperties`/
  `vkEnumerateInstanceLayerProperties`) through the chained, `VK_NULL_HANDLE`-instance
  `vkGetInstanceProcAddr`, even though a layer that doesn't hook those extra two never
  calls them afterward. Resolving `vkEnumerateInstanceExtensionProperties` that way
  segfaults inside `libVkLayer_MESA_device_select.so` on this Mesa build 100% of the
  time; `vkCreateInstance` resolves fine through the exact same chained pointer right
  before it. Worked around by implementing `GlobalHooks::create_instance` ourselves
  (`crates/layer/src/lib.rs`'s new `DlssnrGlobalHooks`) and resolving only the one
  entry point this layer actually needs, never making the query that crashes.
  Verified fixed: 5/5 clean runs locally (this sandbox has the identical Mesa
  `device_select` present) and 3/3 clean real `vkcube` runs on `lordnikon` (real
  GPU/driver, the machine the original crash was found on) — capturing and
  round-tripping real frames the whole time, no crash. Full workspace test suite and
  both smoke tests (explicit and implicit activation) still green.

## 0.1.5 — 2026-09-10

- No fix in this release — documenting a critical, confirmed bug found while testing
  against real hardware with a legitimate NGX DLL for the first time: this project's
  Vulkan layer segfaults 100% of the time when loaded the way it will always actually
  be loaded (implicit activation via `VKLayer_DLSS5=1`), in the presence of Mesa's
  `device_select` implicit layer. Upstream, tested side by side on the same machine,
  works correctly end to end. See CLAUDE.md for the full, gdb-verified writeup — what's
  ruled out, what isn't yet, and how to reproduce it.
- Runtime SHM directory permissions note: if DLSS5 NR silently refuses to work, check
  `/tmp/dlssnr-$UID` is `0700` (owner-only) — both this project's and upstream's own
  security check correctly reject a group-writable runtime directory.

## 0.1.4 — 2026-09-10

- The write-back now uses the helper's actual answer instead of always
  re-presenting the untouched capture. Not yet verified against a real present
  cycle (no display/swapchain in this dev sandbox) — the underlying change is
  small and carefully reasoned through, but real confirmation still needs to
  happen against an actual game on real hardware.

## 0.1.3 — 2026-09-10

- Settings changed in the GUI now survive a reboot. Previously they lived only in the
  SHM mapping under `/tmp`, which doesn't persist — found by comparing against a real,
  installed upstream instance's `config.ini`, which does persist them.

## 0.1.2 — 2026-09-09

- Fixes a real crash found on first-ever real-hardware testing: `create_feature_at()`
  set every `DLSSNR.*` parameter through the raw NGX vtable outside any `guarded()`
  call, the only DLL-touching code in `ngx.rs` that wasn't. On a real GPU + real
  `nvngx_dlssnr.dll`/`nvngx.dll`, a fault in that block took the whole helper down
  silently a couple seconds after a successful `VULKAN_Init_Ext`, with no log line and
  no crash dialog. Now guarded like every other call in the file, so a fault latches
  `disabled` and logs `seh=...` instead of killing the process.

## 0.1.1 — 2026-09-09

- Adds a Start/Stop button for the helper to the GUI's Status group. Backed by a new
  shared `dlssnr-supervisor` crate (extracted from `dlssnr-cli`) so both the CLI and
  GUI start/stop the helper through the same code instead of duplicating it.
- Adds an About dialog to the GUI (there wasn't one before) crediting Claude Code
  (Anthropic) in its acknowledgements, matching GreenLight/KernelPop/SteamPunk.
- Adds a "NGX binaries" import button to the GUI's Status group — previously only
  `dlssnr-cli import-binaries` could do this.

## 0.1.0 — 2026-09-09

- First release.
