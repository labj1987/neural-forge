# Neural Forge developer guidance

Repository: https://github.com/labj1987/neural-forge

Read [docs/PHASE1.md](docs/PHASE1.md) for the current namespace, installation contract and
benchmark plan. The former app name was dlssnr; upstream DLSS5VKLayer remains a
separate application and must not be modified or uninstalled by this project.

## Naming convention

- Display name (anything a person reads: window titles, About, docs prose): **Neural Forge**.
- Hyphenated lowercase `neural-forge` for everything a machine or filesystem names: the
  repo, Cargo package names (`neural-forge-cli`, `neural-forge-protocol`, ...), binaries
  (`neural-forge`, `neural-forge-cli`, `neural-forge-helper.exe`), the AppImage
  (`neural-forge-<version>-x86_64.AppImage`), the icon (`neural-forge.svg`).
- **Frozen, never rename** (renaming breaks layer registration): the app ID
  `io.github.labj1987.NeuralForge` (and the `.desktop`/appdata files named after it) and the
  Vulkan layer name `VK_LAYER_neuralforge_neural`.
- Since 0.1.77 everything else is hyphenated/underscored `neural-forge`: `NEURAL_FORGE_*` env
  vars (read through `neural_forge_protocol::env`), `/tmp/neural-forge-$UID`, the `neural-forge` XDG config/data/state
  dirs, `lib/neural-forge/`, `libneural_forge_layer.so` (`[lib] name = "neural_forge_layer"`),
  its manifest `neural_forge_layer.json`, and the `[neural-forge-layer]`/`[neural-forge-helper]`
  log prefixes. The manifest's `enable_environment` is `NEURAL_FORGE_ENABLE`.
- No backward compatibility with the old names (`dlssnr`, `neuralforge`, `NEURALFORGE_*`,
  `NeuralForge-*` AppImages): every install already uses the current ones. Files a previous
  install shipped but this one no longer does are cleaned by the installer's record-based
  stale-file removal. Never run the CLI or `install` against real XDG dirs in tests: point
  `XDG_*_HOME` at a scratch dir.

## Build and test

- `cargo test` and `cargo build --release` build the native default members.
- Do not use `--workspace` on Linux: the helper targets Windows only.
- `cargo +stable build --release --target x86_64-pc-windows-gnu -p neural-forge-helper`
  builds the helper when the cross target is installed in the stable toolchain.
- `CARGO_HELPER='cargo +stable' bash build-appimage.sh` packages Neural Forge.
- Run `python3 scripts/check_namespace.py`, `python3 scripts/test_install.py`,
  and `bash scripts/smoke-test.sh` for namespace, installation and Vulkan checks.

## Runtime constraints

Use only NeuralForge-owned paths, `NEURAL_FORGE_*` variables, and the
`VK_LAYER_neuralforge_neural` identity. Keep NVIDIA DLL names, NGX exports and
`DLSSNR.*` parameters unchanged. Do not copy or move ambiguous upstream config,
shared memory, or Wine prefixes. Import DLLs explicitly into NeuralForge's data dir.

Target-process filtering and the kernel ownership lease must remain effective before
any process can resize or write a channel. Preserve the GTA baseline: helper enabled,
passes=1, model_resolution=1, motion disabled/quality 0, host SHM transport.
DMA-BUF remains experimental, and both transport directions tried so far are blocked
on real, confirmed-on-hardware constraints (a Wine/NVIDIA-driver handle-type mismatch
one way, a plain Linux anon-inode-fd limitation the other), not just unimplemented --
see `docs/DMABUF_TRANSPORT_DESIGN.md` before touching it again. The "skip Wine with a native
Linux NGX helper" idea that would have sidestepped both is also closed, for an
unrelated reason (no native Linux implementation of this project's target NGX feature
exists anywhere, confirmed on real hardware) -- see `docs/NATIVE_NGX_HELPER_DESIGN.md`
before touching that either. Do not lower model resolution or disable the helper
without explicit user authorization.

Do not reapply the reverted capture/composition fence changes. Validate actual GPU
operations and establish matched upstream/NeuralForge measurements before performance
changes. Keep the Rust implementation independent; review licenses before source reuse.

Current target-machine evidence and unresolved Vulkan errors are in
[docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md).

## Deliberately not done

One item from the completion plan's Phase 6 was considered and intentionally left
as-is; don't re-raise it without new information:

- **Hotkey capture does not filter non-keyboard evdev devices** (Phase 6 item 6, as
  literally worded). Investigated 2026-09-15: neither the layer's in-game hotkey
  polling (`crates/layer/src/hotkey.rs`, X11 `XQueryKeymap`) nor the GUI's
  hotkey-capture row (`crates/gui/src/ui.rs`'s `hotkey_row`, GDK key-press events) does
  raw `/dev/input/eventN` enumeration at all -- both are already inherently
  keyboard-scoped by construction (`XQueryKeymap` only ever reports keyboard state;
  GDK key-press events only fire for keyboard input). The item doesn't map onto this
  architecture without inventing a new raw-evdev capture mechanism neither mechanism
  currently has any reason to need.

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
  device) has been waited on: slot resources, swapchain images, NGX features.
- **Tested on lavapipe, and the test fails without the change.** Every composition behaviour has
  a GPU test (`composition::gpu::tests::compose_once` makes one short); run it once with the
  change reverted to see it fail.
