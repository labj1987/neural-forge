# NeuralForge developer guidance

Repository: https://github.com/labj1987/NeuralForge

Read [docs/PHASE1.md](docs/PHASE1.md) for the current namespace, installation contract and
benchmark plan. The former app name was dlssnr; upstream DLSS5VKLayer remains a
separate application and must not be modified or uninstalled by this project.

## Build and test

- `cargo test` and `cargo build --release` build the native default members.
- Do not use `--workspace` on Linux: the helper targets Windows only.
- `cargo +stable build --release --target x86_64-pc-windows-gnu -p neuralforge-helper`
  builds the helper when the cross target is installed in the stable toolchain.
- `CARGO_HELPER='cargo +stable' bash build-appimage.sh` packages NeuralForge.
- Run `python3 scripts/check_namespace.py`, `python3 scripts/test_install.py`,
  and `bash scripts/smoke-test.sh` for namespace, installation and Vulkan checks.

## Runtime constraints

Use only NeuralForge-owned paths, `NEURALFORGE_*` variables, and the
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
