# Neural Forge developer guidance

Repository: https://github.com/labj1987/neural-forge

Read [docs/PHASE1.md](docs/PHASE1.md) for the current namespace, installation contract and
target-ownership rules. Benchmarks run with `scripts/gta-bench.sh`
([docs/RUNNING_AND_MEASURING.md](docs/RUNNING_AND_MEASURING.md)) at 2560x1440, 288 Hz, VRR (the desktop's HDR as
it is set; HDR in the game only when asked for), the only configuration games are tested in: never at 4K, which the test machine's 12 GB card cannot hold steadily (GTA with frame
generation collapses at its memory limit); the old Phase 1 plan is in
[docs/history/phase1-benchmark-plan.md](docs/history/phase1-benchmark-plan.md). The former app name was dlssnr; upstream DLSS5VKLayer remains a
separate application and must not be modified or uninstalled by this project.

All documentation is indexed in [docs/README.md](docs/README.md); start with docs/ARCHITECTURE.md and docs/LESSONS.md.

## Naming convention

- Display name (anything a person reads: window titles, About, docs prose): **Neural Forge**.
- Hyphenated lowercase `neural-forge` for everything a machine or filesystem names: the
  repo, Cargo package names (`neural-forge-cli`, `neural-forge-protocol`, ...), binaries
  (`neural-forge`, `neural-forge-cli`), the AppImage
  (`neural-forge-<version>-x86_64.AppImage`), the icon (`neural-forge.svg`).
- **Frozen, never rename** (renaming breaks layer registration): the app ID
  `io.github.labj1987.NeuralForge` (and the `.desktop`/appdata files named after it) and the
  Vulkan layer name `VK_LAYER_neuralforge_neural`.
- Since 0.1.77 everything else is hyphenated/underscored `neural-forge`: `NEURAL_FORGE_*` env
  vars (read through `neural_forge_protocol::env`), `/tmp/neural-forge-$UID`, the `neural-forge` XDG config/data/state
  dirs, `lib/neural-forge/`, `libneural_forge_layer.so` (`[lib] name = "neural_forge_layer"`),
  its manifest `neural_forge_layer.json`, and the `[neural-forge-layer]` log prefix. The manifest's `enable_environment` is `NEURAL_FORGE_ENABLE`.
- No backward compatibility with old names or formats (`dlssnr`, `neuralforge`, `NEURALFORGE_*`,
  `NeuralForge-*` AppImages, the 2.x helper's config keys): every install already uses the current
  ones; config files are cleaned on the machines directly, not by migration code. Files a previous
  install shipped but this one no longer does are cleaned by the installer's record-based
  stale-file removal. Never run the CLI or `install` against real XDG dirs in tests: point
  `XDG_*_HOME` at a scratch dir.

## Build and test

- Run `bash scripts/fetch-native-tools.sh` once before building: it fetches the pinned glslang,
  Vulkan-Headers, volk and OpenDLSS-NR source (with `third_party/opendlss-nr/patches/` applied, each
  exactly) that `crates/native` builds with into `tools/native/`; the build is offline after that.
  OpenDLSS-NR is not vendored: change it through a patch there (UPSTREAM.md lists each with its reason).
- CI (`.github/workflows/check.yml`, run by ci.yml on every push and pull request and by release.yml
  on a tag) runs the tests, the layer's tests again under the Khronos validation layer (which must
  report nothing; one known violation is filtered until it is fixed), clippy, every script's
  self-test, the shader check, the AppImage build and its installed-layer smoke test. A new script with a self-test is added there. The weekly pin check
  (`scripts/check_pins.py`, pin-check.yml) keeps one issue about pins behind upstream; it must never
  comment or open more than that one (each notification is an email).
- `cargo test` and `cargo build --release` build every crate (x86_64 Linux only; there is no
  Windows helper and no 32-bit layer since 3.0).
- `bash build-appimage.sh` packages Neural Forge.
- The layer's GPU tests run on lavapipe: set `VK_DRIVER_FILES` to the `lvp_icd*.json` manifests, as
  CI does. On a machine with an NVIDIA GPU, never run them without it. Under validation:
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` (CI's step shows the settings); a test device
  enables what the code under test needs (`composition::gpu::test_device_extensions`), destroys what
  it created before the device, and records over real images, never stand-in handles.
- Run `python3 scripts/check_namespace.py`, `python3 scripts/test_install.py`,
  and `bash scripts/smoke-test.sh` for namespace, installation and Vulkan checks.
- Shaders: `python3 scripts/check_shaders.py --update` rebuilds the committed SPIR-V with the pinned
  glslang only (never one on PATH) and needs `spirv-val`.
- Every component compiled or linked into the shipped binaries must have its licence text in the
  AppImage: `scripts/gen_third_party_crates.py --licences` collects them from the crates' and the
  native sources at build time and fails on one without.
- Toggle key: a press is ignored only while another X client is known to have the focus
  (`hotkey.rs`); where the focus cannot be told (gamescope's nested Xwayland holds only the game, a
  bare X server without `_NET_ACTIVE_WINDOW`) it acts, and `NEURAL_FORGE_HOTKEY_GLOBAL=1` makes it act
  even then. Keep that: a key read from `/dev/input` otherwise fires while the user types elsewhere.
- Launch options: the supervisor always writes `NEURAL_FORGE_TARGET_EXE` (the game's own executables,
  `steam::target_for`), and an explicit target wins over the layer's built-in exclusions.
- `extract-model` gates on the network's shape, not the DLL's version: it takes any build of
  `nvngx_dlssnr.dll` whose weights resource holds exactly the tensors in
  `crates/supervisor/src/model_shape.rs` (names and byte lengths): `WEIGHTS_HT` for the full network
  (`EXPECTED`, 71 blocks), `WEIGHTS_CC_SMALL_FP8` for the small one (`EXPECTED_SMALL`, 31 blocks,
  `--network small`). The graph picks its block ranges from the model's block count
  (`patches/0009`, `nr::Topology`); a third network needs a new `Topology` and stage table. That table changes together with
  OpenDLSS-NR's graph: regenerate it with `scripts/gen_model_shape.py` when the pinned commit
  (`third_party/opendlss-nr/UPSTREAM.md`) moves to a new network. `VERIFIED` only marks the networks and builds compared with
  NVIDIA's runtime; it gates nothing.

## README

- The README describes the newest release only: what it does, how to install and use it, its
  current measurements and known issues, all in the present tense. No version history ("since 3.0",
  "3.4.1 drops ...", tables comparing releases, notes for settings made with an older version,
  screenshots captioned with an old version). Users run the latest release; history belongs in
  CHANGELOG.md and the docs.
- Every release re-reads the whole README and updates or removes what the release changed, including
  the screenshots and the measured results.
- Screenshots are always the dark theme, 1240x1400 (the window at 2x). A headless run (Xvfb) has no
  portal and falls back to light: set `ADW_DEBUG_COLOR_SCHEME=prefer-dark`, and look at every image
  before committing it.

## Changelog

- One `## X.Y.Z — YYYY-MM-DD` heading per released version, newest first. No entries for builds
  that were never released.
- Write each entry for the people using the app: what changed for them and anything they need to
  do. Leave out implementation detail (file paths, flags, internal names, CI and packaging changes)
  unless a user needs it to act.
- The release page is the version's section written out in full (`scripts/release_notes.py`), never
  a link to the changelog. The release fails if the section is missing.
- The AppStream `<releases>` list is generated from the headings
  (`scripts/sync_appdata_releases.py`, run by `build-appimage.sh` and checked in CI). Don't edit it
  by hand.

## Runtime constraints

Use only Neural Forge-owned paths, `NEURAL_FORGE_*` variables, and the
`VK_LAYER_neuralforge_neural` identity. Keep NVIDIA DLL names, NGX exports and
`DLSSNR.*` parameters unchanged. Do not copy or move ambiguous upstream config,
shared memory, or Wine prefixes. Import DLLs explicitly into Neural Forge's data dir.

Target-process filtering and the kernel ownership lease must remain effective before
any process can resize or write a channel. Preserve the GTA baseline: the model before the
upscaler on the native backend, the network on every frame while the camera is still and on every
second frame in motion with the edit cache carrying it between (docs/ARCHITECTURE.md 4.10;
`NEURAL_FORGE_EDIT_CACHE=off` runs it every frame), model_resolution=1. Since 3.0 the network runs in
the layer (`crates/native`, `crates/layer/src/preupscale/native.rs`; docs/NATIVE_BACKEND.md); the
2.x Windows helper and everything around it (runners, Wine prefix, DMA-BUF, NGX) are removed and
not to be brought back. Do not lower model resolution or turn the model off by default without
explicit user authorization.

Do not reapply the reverted capture/composition fence changes. Validate actual GPU
operations and measure before and after every performance change. The matched comparator
is the released 2.0.10 against the native backend (docs/NATIVE_BACKEND.md, Phase 3), taken with
`scripts/gta-bench.sh` and mods off. Keep the Rust implementation independent;
review licenses before source reuse.

Current target-machine evidence and unresolved Vulkan errors are in
[docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md).

## Pre-upscaler path

Since 2.0 the model runs **before** DLSS Super Resolution by default (`crates/layer/src/preupscale.rs`,
docs/PRE_UPSCALER_DESIGN.md): on a device with `VK_NVX_image_view_handle`, the layer identifies
DLSS's registered colour input, holds the game's DLSS submit, runs the network on the
render-resolution frame (every frame while the camera is still, every second frame in motion with the
edit cache carrying it between; with DLSS's motion vectors and the jitter for the history) and
writes the answer back before DLSS runs. Games that record their frame in DLSS's own command buffer
are held inside it (`preupscale/inline.rs`, on the layer's side compute queue). The colour input is
what DLSS's own input kernel's parameters name with depth and motion vectors (which also finds DLAA's
output-size input), with the older size rule as the fallback (docs/PRE_UPSCALER_DESIGN.md,
"Identification by the input kernel's parameters (DLAA)"). DLSS Ray
Reconstruction is never held: its input is the noisy ray-traced frame, so while its kernels
(`rr2_*`, `cuda_dldn_engine_swin_*`, `cuda_dldn_engine_hkpn_*`; names from `vkCreateCuFunctionNVX`)
launch nothing is identified and the model runs after the upscaler ("DLSS
Ray Reconstruction"). Only that positive sign gates: Super Resolution needs no kernel name (Crimson
Desert's is `rrlite_*`, and `custom_block*`/`k_initial_merge` are Frame Generation's); without a readable exposure image the
exposure is measured from the frame ("Auto-exposure when the game gives DLSS none"). Everything else (no NVX, no DLSS, native)
keeps the post-upscaler path. Only the launch-bearing command buffer whose
kernel parameters name the identified colour input is held, so DLSS Frame Generation's submits go
through untouched (docs/PRE_UPSCALER_DESIGN.md, "DLSS Frame Generation"). `NEURAL_FORGE_PREUPSCALE=off` is the A/B
and rollback switch; `dump`, `identity` and `roundtrip` are diagnostics. Rule: with
`NEURAL_FORGE_PREUPSCALE=off` the post path must stay byte-identical to 1.1.0 (the hooked-command
list is the default set, no NVX entry points resolved, nothing of `preupscale` reachable), and a
device without NVX must not get the tracking under the default either; the tests in
`lib.rs::probe_command_tests` and the smoke test's `off` pass guard this.

## Historical evidence

[Pre-rename development notes](docs/history/development-before-neuralforge.md) retain
old commands, measured failures and toolchain investigations as historical evidence.
Those old deployment recipes are not current instructions. The historical GTA handoff
is [docs/history/handoff-2026-09-12-fps-freeze-regression.md](docs/history/handoff-2026-09-12-fps-freeze-regression.md).

## Composition invariants (checklist for every change to the pass)

Adopted from upstream DLSS5VKLayer's `DEVELOPMENT.md`; check each before committing shader or
composition changes.

- **Default-identical toggles.** A new control's default must leave the shipped picture bit
  for bit as it was (or the change says in its commit why the picture moves). Transfer modes
  are identical at 100% model resolution; compare off is a no-op.
- **Push constants append-only, struct == shader.** `composition::gpu::PushConstants` and
  `compose.comp`'s `Params` block match field for field, and new fields go at the end of both.
  The same for `ShmHeader`: append, bump `SHM_VERSION`, update the pinned offsets.
- **The encode is reproduced wherever the proxy is compared.** Anything that compares the
  model's answer with the frame must first put the frame through the same white divide and
  knee the encode used (`SoftKneeLuminance(original / white_point)`), passthrough included.
- **Drain before free.** Nothing the GPU may still read is destroyed until its fence (or the
  device) has been waited on: slot resources, swapchain images, the network's buffers.
- **Tested on lavapipe, and the test fails without the change.** Every composition behaviour has
  a GPU test (`composition::gpu::tests::compose_once` makes one short); run it once with the
  change reverted to see it fail.
