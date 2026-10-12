# Neural Forge Phase 1

> **Note (2026-10-07, 3.0.0):** the namespace, installation and target-ownership sections are
> current for 3.0, which has no Windows helper, Wine prefix or runner. The Phase 1 benchmark plan
> moved to [history/phase1-benchmark-plan.md](history/phase1-benchmark-plan.md); benchmarks now run
> unattended with `scripts/gta-bench.sh` ([RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md)), and
> the matched comparator is 2.0.10 against the native backend ([NATIVE_BACKEND.md](NATIVE_BACKEND.md),
> Phase 3).

Neural Forge is the Rust application in `labj1987/neural-forge`. The GitHub repository
has been renamed from `labj1987/dlssnr`. Historical handoffs are evidence, not deployment instructions.
No installed upstream package or game setting is changed by this work.
See [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for the current RTX 5070
validation evidence before GTA benchmarking.

## Namespace contract

| Surface | Neural Forge |
|---|---|
| GUI / CLI | `neural-forge`, `neural-forge-cli` |
| Vulkan identity / library (frozen) | `VK_LAYER_neuralforge_neural`, `libneural_forge_layer.so` |
| Activation / opt-out | `NEURAL_FORGE_ENABLE=1`, `NEURAL_FORGE_DISABLE=1` |
| Other private environment variables | `NEURAL_FORGE_*`; no `DLSSNR_*` aliases |
| Desktop / application ID | `io.github.labj1987.NeuralForge` |
| Configuration | `$XDG_CONFIG_HOME/neural-forge/{config.ini,profiles.ini}` |
| Data | `$XDG_DATA_HOME/neural-forge/{bin,lib,model,binaries,captures}` |
| Runtime / control / lease | `/tmp/neural-forge-$UID/{shm.bin,shm.bin.owner}` |
| Wire magic | `NFR1` (layout v2 retained) |
| AppImage | `neural-forge-<version>-x86_64.AppImage` |

XDG variables fall back to the usual directories under HOME. Shared memory stays
under /tmp for Steam pressure-vessel visibility. Old runtime/config paths are not
imported. Explicit upstream (`dlssnr`) SHM/log paths are refused or reset to Neural Forge defaults.
External `nvngx_dlssnr.dll`, other NVIDIA DLL names, exported NGX functions,
`DLSSNR.*` parameters and driver environment variables keep their original spelling.

## Installation

Build with `bash build-appimage.sh` (after `bash scripts/fetch-native-tools.sh` once). The
AppImage launches the GUI directly and, each time it starts, installs its layer into persistent
user storage, since Steam games launched later cannot see inside an AppImage mount. From a
source build, install the extracted AppDir the same way:

```sh
python3 scripts/install.py install --appdir build-appimage/AppDir
```

The installer places the binaries under `$XDG_DATA_HOME/neural-forge/bin` and the layer under
`lib/neural-forge/`, and writes the layer manifest with an absolute library path to
`$XDG_DATA_HOME/vulkan/implicit_layer.d/`. It writes no desktop, icon or metainfo files: menu
integration belongs to whatever integrated the AppImage. It refuses unknown or modified
destinations, replaces files atomically (running processes keep their mapped copies), and removes
files a previous version installed that this one no longer ships, if they are unchanged.
`python3 scripts/install.py uninstall` removes only unchanged, hash-recorded files; it keeps
config, the extracted model, an imported DLL and runtime data (`neural-forge-cli uninstall --purge`
removes those too). It
never calls a package manager or stops another app.

Neural Forge does not migrate or touch upstream's shared `~/.config/dlssnr`, `/tmp/dlssnr-*`,
DLLs, or desktop files. Extract the model explicitly from the user's own `nvngx_dlssnr.dll`
(`neural-forge-cli extract-model`, or the GUI's Setup tab). There are no old-name executable
aliases.

## Target ownership

`NEURAL_FORGE_TARGET_EXE` accepts a case-insensitive, comma-separated list of executable
basenames, trimmed. When it is set, exactly those processes are eligible, even one whose name is on
the built-in list below (a game whose executable is literally `Launcher.exe` works when it is the
target). The supervisor always sets it when it switches a game on (GUI, `neural-forge-cli enable`):
the program Steam's own launch entry for the game starts (`appcache/appinfo.vdf`, the entries for
the installed branch), the other executables beside it (a store bootstrapper such as `PlayGTAV.exe`
starts `GTA5_Enhanced.exe` from the same folder; anti-cheat's `start_protected_game.exe` its
neighbour), an Unreal Engine game's `<Project>/Binaries/Win64` binary, and from the whole install
folder (six folders deep) every executable at least a quarter the size of the largest, which finds a
game whose launcher has a folder of its own (ESO's `game/client/eso64.exe`, FINAL FANTASY XIV's
`game/ffxiv_dx11.exe`). Redistributable installers, anti-cheat services, crash reporters and
launchers (bootstrappers, updaters, patchers) are left out; a launch entry that is a launcher stays
only when nothing else qualifies. A target chosen by hand (`enable GAME --target-exe`, or the Setup
tab's per-game editor, which offers the folder's executables best first) is kept in `config.ini`
(`target_exe_<appid>`) and wins. When a process the target leaves out creates a game-sized swapchain, the layer
writes a `target-excluded` line to the state log (once per process; never for launchers, bootstrappers,
updaters or helpers), and `doctor` reports the newest one, unless a process engaged in that session or
a later one, as a game that ran as an executable its target missed, with the command that targets
it. The Steam AppID
alone is insufficient: launchers inherit it too.

Without a target, Wine's own processes, Steam's, store launchers and their helpers (Rockstar, Social
Club, Ubisoft Connect, EA app, Battle.net, Epic, GOG Galaxy, Riot and others), crash reporters and
gamescope are excluded before swapchain setup (`neural_forge_protocol::processes`), and the first
other process to present steadily wins.

The first eligible process to open a channel takes a nonblocking kernel file lease.
Other processes cannot attach, write frames or change its dimensions. The lease survives
swapchain recreation and releases on crash or exit without PID reuse/timeout guessing; it is also
given up after 10 s without a present, so a process that claimed it and stopped presenting cannot
keep the game from it. Never unlink `shm.bin.owner` while a game runs. Within one process the
existing largest-swapchain rule is retained. Multiple simultaneous games need distinct explicit
private SHM channels (`NEURAL_FORGE_SHM`); the GUI manages one channel per user.
