# NeuralForge Phase 1

NeuralForge is the Rust application in `labj1987/neural-forge`. The GitHub repository
has been renamed from `labj1987/dlssnr`. Historical handoffs are evidence, not deployment instructions.
No installed upstream package or game setting is changed by this work.
See [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for the current RTX 5070
validation evidence before GTA benchmarking.

## Namespace contract

| Surface | NeuralForge |
|---|---|
| GUI / CLI / Windows helper | `neural-forge`, `neural-forge-cli`, `neural-forge-helper.exe` |
| Vulkan identity / library (frozen) | `VK_LAYER_neuralforge_neural`, `libneural_forge_layer.so` |
| Activation / opt-out | `NEURAL_FORGE_ENABLE=1`, `NEURAL_FORGE_DISABLE=1` |
| Other private environment variables | `NEURAL_FORGE_*`; no `DLSSNR_*` aliases |
| Desktop / application ID | `io.github.labj1987.NeuralForge` |
| Configuration | `$XDG_CONFIG_HOME/neural-forge/config.ini` |
| Data / managed prefix | `$XDG_DATA_HOME/neural-forge/{binaries,prefix}` |
| State | `$XDG_STATE_HOME/neural-forge/helper.log` |
| Runtime / control / lease | `/tmp/neural-forge-$UID/{shm.bin,helper.pid,shm.bin.owner}` |
| Wire magic | `NFR1` (layout v2 retained) |
| AppImage | `neural-forge-0.1.54-x86_64.AppImage` |

XDG variables fall back to the usual directories under HOME. Shared memory stays
under /tmp for Steam pressure-vessel visibility. Old runtime/config paths are not
imported. Explicit legacy SHM/log paths are refused or reset to NeuralForge defaults.
External `nvngx_dlssnr.dll`, other NVIDIA DLL names, exported NGX functions,
`DLSSNR.*` parameters and driver environment variables keep their original spelling.

## Installation and safe legacy handling

Build with `CARGO_HELPER='cargo +stable' bash build-appimage.sh` when the Windows
cross target lives in the stable toolchain. The AppImage launches the GUI directly.
To make the layer available to separately launched Steam games, install the extracted
AppDir into persistent user storage (an AppImage mount alone cannot do that):

```sh
python3 scripts/install.py install --appdir build-appimage/AppDir
```

The installer places binaries under `$XDG_DATA_HOME/neural-forge/bin`; use that full
path or add it to PATH. It writes a persistent manifest with an absolute library path,
plus the desktop/icon/metainfo files. It refuses unknown or modified destinations.
Updates replace files atomically, preserving mapped binaries in running processes.
`python3 scripts/install.py uninstall` removes only unchanged, hash-recorded files;
it retains config, binaries imported by the user, prefix, logs and runtime data.
It never calls a package manager or stops another app.

An optional, explicit migration archives only this repository's old manifest after
checking its unique layer name, library basename, and activation/disable keys:

```sh
python3 scripts/install.py archive-legacy-manifest --legacy-manifest /explicit/path/to/VK_LAYER_dlssnr_neural.json
```

It saves a `.disabled` backup in NeuralForge's data directory. It does not migrate
shared `~/.config/dlssnr`, old `/tmp/dlssnr-*`, Wine prefixes, DLLs, or desktop files.
No ownership evidence exists for those shared files. Import NVIDIA binaries explicitly
using `neural-forge-cli import-binaries DIR`. There are no old-name executable aliases.

## Target ownership

Known Wine desktop, Xalia, Rockstar, Social Club, Steam and helper executables are
excluded before swapchain setup. `NEURAL_FORGE_TARGET_EXE` accepts a case-insensitive,
comma-separated list of executable basenames; configure it for the actual game exe.
For GTA Enhanced use `GTA5_Enhanced.exe` after confirming that process name locally.
The Steam AppID alone is insufficient: launchers inherit it too. The helper launcher
also strips game-rendering layer selections from its child environment, preserving
unrelated diagnostic layers and leaving the parent/desktop environment unchanged.

The first eligible process to open a channel takes a nonblocking kernel file lease.
Other processes cannot attach, write frames or change its dimensions. The lease lasts
until process exit, survives swapchain recreation and helper restart, and releases on
crash without PID reuse/timeout guessing. Never unlink `shm.bin.owner` while a game
runs. Within one process the existing largest-swapchain rule is retained.
An unknown launcher can win first in unrestricted mode: use the explicit target for
GTA. Multiple simultaneous games need distinct explicit private SHM channels and
separate helper management; the current GUI manages one channel per user.

## Preserved baseline and benchmark gate

Upstream remains installed at 0.3.0-1. Its known-good GTA Enhanced launch option is:

```text
VKLayer_DLSS5=1 DLSSNR_DMABUF=0 %command%
```

Do not change this saved baseline. A separate NeuralForge test launch uses:

```text
NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_DMABUF=0 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%
```

NeuralForge currently uses host SHM transport; `NEURAL_FORGE_DMABUF` is reserved and
has no zero-copy implementation -- and, per `DMABUF_TRANSPORT_DESIGN.md`, real
hardware evidence this session says the underlying mechanism the current Wine-hosted
helper would need is blocked at the driver/Wine level, not just unbuilt. DMA-BUF
remains experimental and requires a separate explicit retest because upstream hung at
4K. Never enable both activation flags for
one benchmark process. Co-installation does not mean double injection is useful.

Keep helper enabled, passes=1, model_resolution=1, motion_enabled=0,
motion_quality=0. New mappings default to motion disabled/quality 0 (`mvec_enabled=0`,
`mvec_quality=0` in this Rust protocol). No model-resolution or helper enablement
changes were made in Phase 1.
Record RTX 5070 / driver 615.71.09, 2560x1440 at 288 Hz, GNOME scale 100%,
Steam AppID 3240220, the exact Proton build, game build, and DLL hashes.

1. Before testing, save config/launch options and record actual helper settings.
   Confirm only the intended layer is mapped into the game and each app's helper
   uses its own prefix, runtime path and advancing counters. Try Explorer, Xalia,
   Rockstar and Social Club while GTA runs; dimensions and owner must stay stable.
2. Validate Vulkan operations using validation layers in a separate smoke run, then
   test the same saved GTA scene/route. Do not mix validation overhead into timing.
3. Measure native baseline, upstream host transport, and NeuralForge host transport
   with identical settings and visual mode. Warm up, alternate order, repeat at least
   three 60-second captures. Record average/1% low FPS, frame-time percentiles,
   GPU utilization, VRAM, helper frames and matched screenshots. Stop on corruption,
   hangs or validation errors; do not compensate by lowering model resolution.
4. No performance improvement is claimed by Phase 1. The old roughly 9 FPS result
   remains unresolved until this matched comparison is run on the target machine.

## Later phases — prepared, not implemented

First add full pipeline instrumentation: capture GPU time, host readback/copy,
request wait/age, helper upload/evaluation/download, composition/writeback, present,
queue waits, allocations and feature rebuilds. Use correlated frame IDs and bounded
logging; quantify instrumentation overhead. Fix measured parity/correctness problems
before adding quality/performance features. Do not reapply reverted fence changes.

Then evaluate matched residual/native+edit with truly matched inputs and HDR/color
handling; independent X/Y neural scaling; delayed feature retirement with GPU-completion
proof and bounded caches; in-place resolve/VRAM reuse with aliasing validation;
depth-aware silhouettes only once reliable game depth exists; and an opt-in adaptive
FPS governor with hysteresis, rate limits, min/max bounds and stable frame pacing.
Each needs image-quality and performance acceptance tests before deployment.

[DLSSNR-Cost-Scaler releases](https://github.com/xenmods/DLSSNR-Cost-Scaler/releases)
are behavioral references: v1.0.4 discusses GPU/in-place optimizations; v1.0.5 adds
asymmetric scaling and depth-aware protection. Review exact tagged source and license
before any reuse. Governor and retirement details require their own design review.
Keep the clean Rust implementation; no code was copied from those projects here.
