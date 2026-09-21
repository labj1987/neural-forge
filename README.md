# Neural Forge

Neural Forge is a Linux Vulkan implicit layer plus Windows helper that forwards presented frames to
NVIDIA's DLSS 5 Neural Rendering model, running the model itself under Wine/Proton.
Written in Rust with GTK4 and libadwaita. A from-scratch rebuild of
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer)'s architecture, not a fork —
see [ATTRIBUTION.md](ATTRIBUTION.md) for exactly what that means and where the ideas
came from.

This is experimental, personal-use software. It works around an authorization check in
NVIDIA's proprietary NGX DLL to run the model outside its intended integration path —
see [Legal](#legal) before you use it.

Repository: [labj1987/neural-forge](https://github.com/labj1987/neural-forge).

## Screenshots

| Settings | Setup |
|---|---|
| ![Neural Forge settings window with Model, Motion, Composition, Debug, Status, and Setup tabs](screenshots/settings.png) | ![Setup tab: NGX binaries status, compatibility tool picker, Steam install and launch-option generator](screenshots/setup.png) |

## What it does

- A Vulkan implicit layer hooks presentation for native Linux and Proton games. It
  sends bounded captured frames to a Windows helper over private shared memory and
  presents untouched frames whenever an answer is not ready.
- Fail-open: if the helper isn't running or the model fails to initialize, the layer
  just presents the original frame — nothing about the game's rendering depends on it.
- The Windows-side helper runs NVIDIA's `nvngx_dlssnr.dll` (Feature 18) under Wine or a
  Proton build. The experimental `VK_NV_optical_flow` motion path is disabled in
  the known-good baseline.
- HDR-aware capture and composition support float16/PQ paths when a compatible
  swapchain exposes them; the validated GTA baseline is SDR B8G8R8A8.
- A composition pass blends the model's output back into the frame — tone/structure/
  skin/sharpness controls, a reversible neutral-axis proxy mode, and a choice of
  resampling filters (Lanczos, Catmull-Rom, Mitchell-Netravali, Kaiser-windowed sinc)
  for the supersampling leg. This math is rederived independently from public sources,
  not ported from any GPL-licensed code — see ATTRIBUTION.md.
- GTK4/libadwaita settings app for all of the above, live-bound to the running layer
  over the same shared-memory segment.
- A CLI (`neural-forge-cli`) for runner discovery, starting/stopping the helper, status,
  diagnostics, importing the NVIDIA NGX DLLs, and raw settings introspection
  (`shmctl status`/`set`/`toggle`/`capture`) — no bash script, no root step.
- Everything lives under `~/.local/share`, `~/.config`, and `/tmp/neuralforge-$UID/`. No
  polkit, no pkexec, no privileged install step at all.

## Requirements

- x86_64 Linux, NVIDIA GPU and driver, Vulkan loader.
- A Wine install or a Steam compatibility tool that bundles DXVK-NVAPI (e.g.
  Proton-CachyOS, Proton-GE) to run the Windows-side helper. Valve's stock Proton
  builds don't bundle DXVK-NVAPI, so they aren't a supported runner.
- NVIDIA's own NGX DLLs, which this project doesn't and can't ship — see below.

## Install

Download the AppImage from [Releases](https://github.com/labj1987/neural-forge/releases):

```bash
chmod +x neural-forge-*-x86_64.AppImage
./neural-forge-*-x86_64.AppImage
```

For Steam games launched separately from the GUI, install the extracted AppDir into
persistent user storage so Vulkan can find the layer after the AppImage exits:

```bash
python3 scripts/install.py install --appdir build-appimage/AppDir
```

The GUI is `neural-forge`; the CLI is `neural-forge-cli`; the Windows helper is
`neural-forge-helper.exe`. Config, data, state, runtime, control mapping and helper
prefix keep their `neuralforge` locations (unchanged by the 0.1.76 rename). Upstream DLSS5VKLayer can remain
installed; Neural Forge neither migrates ambiguous upstream state nor changes its
files, configuration, launch options, helper, or runtime. See
[docs/PHASE1.md](docs/PHASE1.md) for executable targeting, migration and uninstall.

## Usage

`nvngx_dlssnr.dll` is NVIDIA's proprietary model binary and isn't included here. Get it
from your own NVIDIA driver/SDK install and import it with:

```bash
neural-forge-cli import-binaries /path/to/dlls
```

or from the GUI's binaries import flow. Files are copied into
`$XDG_DATA_HOME/neuralforge/binaries`; restart the helper afterward.

Add `NEURALFORGE_ENABLE=1` (and, for a specific target executable in a multi-process
game, `NEURALFORGE_TARGET_EXE=<name>.exe`) to a game's Steam launch options to
activate the layer. GUI and layer share live settings over the same shared-memory
segment; `neural-forge-cli shmctl status/set/toggle/capture` covers the same controls
from a terminal.

## Status

Validated on GTA V Enhanced (RTX 5070, driver `615.71.09`): the render tap correctly
captures GTA's own render target while leaving Rockstar Launcher, Social Club, Wine
Explorer, Xalia and overlays pass-through, and the full model/helper round-trip runs
end to end. The current synchronous host-SHM transport is a correctness baseline, not
a performance result — see [docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md) for
measurements, [docs/RENDER_TAP_DESIGN.md](docs/RENDER_TAP_DESIGN.md) for the capture
constraints, and [docs/ASYNC_CAPTURE_DESIGN.md](docs/ASYNC_CAPTURE_DESIGN.md) and
[docs/EXTERNAL_MEMORY_HOST_DESIGN.md](docs/EXTERNAL_MEMORY_HOST_DESIGN.md) for the zero-copy
transport work in progress.

## Known issues

**A GPU hang with an NVIDIA `Xid 109` (`CTX_SWITCH_TIMEOUT`) or `Xid 119` error.**
Versions before 0.1.61 had a real bug that could plausibly cause exactly this (a
render-tap bookkeeping leak that could act on a reused image handle) — update first.
If it still happens on 0.1.61 or later, it may be the separate, long-running, widely
reported NVIDIA Linux driver bug under Proton — see
[nvidia forums thread 283722](https://forums.developer.nvidia.com/t/xid109-ctx-switch-timeout-driver-crashes-in-many-applications/283722)
and [NVIDIA/open-gpu-kernel-modules#1097](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1097) —
affecting a wide range of GPUs (RTX 2080 through 5090) and a wide range of games with
no DLSS/neural rendering involved at all (CS2, Elden Ring, Apex Legends, Path of Exile,
Crimson Desert), across driver branches NVIDIA has not yet fixed. Check `journalctl -k`
for a real `Xid` line before assuming Neural Forge caused a crash — see
[docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md)'s 2026-09-16 entry for how this was
diagnosed and what other users report as partial workarounds (driver downgrade to the
550.x branch, `PROTON_HIDE_NVIDIA_GPU=1 PROTON_ENABLE_NVAPI=1` with Pyroveil, or a
lower in-game resolution).

## Building from source

```bash
cargo build --release          # protocol, layer, gui, cli (native Linux)
cargo +stable build --release --target x86_64-pc-windows-gnu -p neural-forge-helper
./build-appimage.sh            # packs everything into an AppImage
```

Needs `mingw-w64` and the GTK4/libadwaita dev packages; see `build-appimage.sh` for the
exact package list. `CLAUDE.md` covers toolchain gotchas in detail if you're
cross-compiling the Windows helper on a machine with its own non-rustup Rust install.

## Legal

`nvngx_dlssnr.dll` checks which module is calling into it and refuses to run outside
its intended host application. The helper here spoofs that check (an IAT hook on the
caller-identity query) so the model will initialize at all under a generic Vulkan
helper process. That is a deliberate design choice, not an accident, and it likely
falls under DMCA §1201 (circumventing an access control) and/or breaches NVIDIA's NGX
EULA, depending on jurisdiction and how you use it. There's no license grant here for
that mechanism and none implied — use it at your own legal risk, for personal,
non-commercial use.

As of 2026-09-17 this project directly reads and adapts source from
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (AGPL-3.0) — see
[ATTRIBUTION.md](ATTRIBUTION.md) for what's taken and from where. Earlier versions of
this composition pipeline were clean-room (no GPL/AGPL code read or ported); that
boundary no longer holds. Upstream's own license is AGPL-3.0, which is why this
project's license is AGPL-3.0-or-later as well: a requirement for the adapted code, not
just a preference (below).

## License

This project's own code is licensed under the **GNU Affero General Public License
v3.0 or later (AGPL-3.0-or-later)** — see [LICENSE](LICENSE). It links against and
depends on NVIDIA's proprietary NGX SDK/DLLs at runtime, which are not covered by that
license and are not redistributed here.

## Acknowledgements

Development assistance: Claude Code (Anthropic) and Codex (OpenAI).
