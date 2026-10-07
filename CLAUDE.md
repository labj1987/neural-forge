# Neural Forge developer guidance

Repository: https://github.com/labj1987/neural-forge

Read [docs/PHASE1.md](docs/PHASE1.md) for the current namespace, installation contract and
benchmark plan. The former app name was dlssnr; upstream DLSS5VKLayer remains a
separate application and must not be modified or uninstalled by this project.

All documentation is indexed in [docs/README.md](docs/README.md); start with docs/ARCHITECTURE.md and docs/LESSONS.md.

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
passes=1, model_resolution=1, motion vectors on (quality FAST, the default since 0.1.98), host SHM transport.
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
operations and measure before and after every performance change. The matched comparator
is the 1.0.1 baseline in docs/HARDWARE_VALIDATION.md ("2.0 baseline"), taken with
`scripts/gta-bench.sh` and mods off; upstream was removed from the rig (see
docs/FRAMEGEN_SPIKE.md, "What broke and why"). Keep the Rust implementation independent;
review licenses before source reuse.

Current target-machine evidence and unresolved Vulkan errors are in
[docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md).

## Working with NVIDIA's binaries

The line is between interface and behaviour, which are fair game, and NVIDIA's code, which is not
read. It applies to `nvngx.dll`, `nvngx_dlss*.dll`, `nvngx_dlssnr.dll` and game executables alike.

Do, freely:

- Run the DLLs through their API and observe what they do: outputs, timing, A/B runs, the helper's
  `[params] read` log (every key the feature reads, `selfparam.rs`), and API-boundary tracing of
  exports, imports or Vulkan calls (for example Frida).
- Log what the layer sees of DLSS through Vulkan: NVX registrations, kernel names
  (`vkCreateCuFunctionNVX`), launch parameter blocks and how their words map to views per DLSS
  version (`NEURAL_FORGE_PROBE_NGX=1`), and record those layouts in `docs/` as tables.
- Read a DLL's interface metadata: PE headers, export and import tables, the version resource,
  `strings` for parameter and kernel names, and `cuobjdump --list-elf`/`--list-ptx` for the names
  of embedded kernels.
- Use NVIDIA's public SDK headers and docs, and open-source consumers of the same features (check
  the licence before taking code; record it in ATTRIBUTION.md).

Don't:

- Disassemble or decompile NVIDIA's code: no Ghidra, IDA, Hopper or REA on these DLLs, no
  `cuobjdump --dump-ptx`/`--dump-sass` or `nvdisasm` on their kernels. The NGX licence forbids it,
  and behaviour observed from outside has answered every question so far.
- Commit NVIDIA bytes: DLLs, PTX or SASS, weights, byte excerpts.
- Reimplement NVIDIA's kernels or network from their code.
- Build, extend or repair anything that gets past NVIDIA's access, licensing or integrity checks.
  The caller-identity spoof (`crates/helper/src/spoof.rs`, README "Legal") stays as it is: agents do
  not change its behaviour, and if a driver or DLL update breaks it, Alex fixes it.

The reverse-engineering tools on the test machine (docs/RUNNING_AND_MEASURING.md, section 10) are
for everything else: this project's own binaries and open-source ones.

## Pre-upscaler path

Since 2.0 the model runs **before** DLSS Super Resolution by default (`crates/layer/src/preupscale.rs`,
docs/PRE_UPSCALER_DESIGN.md): on a device with `VK_NVX_image_view_handle`, the layer identifies
DLSS's registered colour input, holds the game's DLSS submit, sends the HDR-encoded render-resolution
frame to the helper every frame and decodes the answer back before DLSS runs. The colour input is
what DLSS's own input kernel's parameters name with depth and motion vectors (which also finds DLAA's
output-size input), with the older size rule as the fallback (docs/PRE_UPSCALER_DESIGN.md,
"Identification by the input kernel's parameters (DLAA)"). Once kernel names are known
(`vkCreateCuFunctionNVX`), nothing is identified unless DLSS SR's input kernel launches, so DLSS Ray
Reconstruction is never held ("DLSS Ray Reconstruction"); without a readable exposure image the
exposure is measured from the frame ("Auto-exposure when the game gives DLSS none"). Everything else (no NVX, no DLSS, native)
keeps the post-upscaler path. Only the launch-bearing command buffer whose
kernel parameters name the identified colour input is held, so DLSS Frame Generation's submits go
through untouched (docs/PRE_UPSCALER_DESIGN.md, "DLSS Frame Generation"). `NEURAL_FORGE_PREUPSCALE=off` is the A/B
and rollback switch; `dump`, `identity` and `roundtrip` are diagnostics. Rule: with
`NEURAL_FORGE_PREUPSCALE=off` the post path must stay byte-identical to 1.1.0 (the hooked-command
list is the default set, no NVX entry points resolved, nothing of `preupscale` reachable), and a
device without NVX must not get the tracking under the default either; the tests in
`lib.rs::probe_command_tests` and the smoke test's `off` pass guard this.

## Deliberately not done

One item from the completion plan's Phase 6 was considered and intentionally left
as-is; don't re-raise it without new information:

- **Hotkey capture does not filter non-keyboard evdev devices** (Phase 6 item 6, as
  literally worded). Note (2026-10-02): since 0.1.78 the layer's hotkey does read evdev
  (`hotkey.rs`, ported from upstream), and it only opens devices that report keys A-Z, so
  the item is covered; the 2026-09-15 reasoning below described the older X11 code.
  Investigated 2026-09-15: neither the layer's in-game hotkey
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
