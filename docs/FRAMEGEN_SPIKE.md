# Frame generation spike: Smooth Motion vs lsfg-vk (2026-09-25)

Question: can a presentation-level frame generator sit *below* Neural Forge so that
generated frames carry NR and the model only runs on real frames, and what does it cost?

Test rig: RTX 5070, NVIDIA 615.71.09, Ubuntu 26.04, Vulkan loader 1.4.341 (host), Steam
Linux Runtime 4 (GTA V Enhanced via Proton-CachyOS), 2560x1440 at 120 Hz with VRR.
Neural Forge 0.1.98 at the usual settings: NR on, model every 2nd frame, working scale
1.0, motion vectors on.

Generators tested: NVIDIA Smooth Motion (`VK_LAYER_NV_present`, driver-shipped) and
lsfg-vk 2.0.0 (`VK_LAYER_LSFGVK_frame_generation`, installed separately from its official
build, with Lossless Scaling's `lsfg-vk` branch). lsfg-vk is CC BY-NC-ND 4.0; nothing from it
is used in or shipped with Neural Forge. It runs only as its own installed layer.

## Layer order

### What the loader documents

- Implicit layers sit closest to the application, followed by layers named in
  `VK_INSTANCE_LAYERS`, then layers the application enables
  ([LoaderApplicationInterface.md, "Overall Layer Ordering"](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderApplicationInterface.md#overall-layer-ordering)).
- There is no defined order *between* implicit layers: within a manifest folder, "the order
  contents are read by the loader in each directory is random due to the behavior of
  readdir" ([LoaderLayerInterface.md, Linux Layer Discovery](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md)).
  No manifest field controls it; upstream still tracks this as
  [Vulkan-Loader#328](https://github.com/KhronosGroup/Vulkan-Loader/issues/328).
- `VK_INSTANCE_LAYERS` is ordered: the first name is closest to the application
  (LoaderApplicationInterface.md).
- The loader settings file ([LoaderSettingsFile.md](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderSettingsFile.md))
  can also fix the order, but it applies to every Vulkan program on the machine and is not
  needed here.
- The authoritative order is the `vkCreateDevice layer callstack setup to:` block printed
  by `VK_LOADER_DEBUG=layer`, read top (application) to bottom (driver)
  ([LoaderDebugging.md](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDebugging.md)).

Inside pressure-vessel, the host's implicit manifests are copied into one folder under
numbered names (NVIDIA's `nvidia_layers.json` was `00`/`01`, Neural Forge `02`), so they
all land under the same unordered readdir.

### What was measured

| Launch | Device chain (application → driver) |
|---|---|
| GTA, `NEURAL_FORGE_ENABLE=1 NVPRESENT_ENABLE_SMOOTH_MOTION=1` | NV_present → neuralforge → steam_overlay → … (**wrong**) |
| GTA, same plus `VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_NV_present` | steam_overlay → neuralforge → NV_present (**correct**) |
| vkcube in SLR 4, `NEURAL_FORGE_ENABLE=1 LSFGVK_PROFILE=…` | LSFGVK → neuralforge (**wrong**, both runs) |
| vkcube in SLR 4, plus `VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_LSFGVK_frame_generation` | neuralforge → LSFGVK (**correct**) |

Both generators land *above* Neural Forge by default, so the launch option must always set
the order explicitly. No change to Neural Forge's manifest or installer can fix this: the
loader offers no per-layer ordering field.

Smooth Motion is refused inside `SocialClubHelper.exe` (its log reports `Vulkan: No`, and the
executable name is hardcoded in `libnvidia-present`). For `GTA5_Enhanced.exe` it reports
`Vulkan: Yes`, `System support: 1. Device support: 1`, and wraps the 2560x1440 swapchain.

## Measurement method

Two counters bracket the generator in the same chain:
`VK_LAYER_MANGOHUD_overlay_x86_64` (above: the application's real fps, plus a frame cap via
`fps_limit`) and `VK_LAYER_MESA_overlay` with `output_file` (below: every frame actually
presented, generated frames included). The loader chain was checked on every run.
Counters that sit above the generator (the game's own counter, the Steam overlay, the Neural
Forge `[present]` line when the order is correct) never see generated frames.

Two traps found along the way:
- The Mesa overlay segfaults vkcube on native Wayland when `output_file` is set; X11 (xcb) works.
- MangoHud writes no log with `no_display` set.

All vkcube runs are 2560x1440, 10 s logging windows. GPU load is `nvidia-smi` sampled every 250 ms.

## Results

### Generator alone (vkcube)

| Config | Real fps | Displayed fps | GPU |
|---|---|---|---|
| no generator, capped 50 | 50.0 | 50.0 | 2%, 23 W |
| Smooth Motion, capped 50 | 50.0 | 100.0 | 7%, 42 W |
| lsfg-vk 2x, capped 50 | 50.0 | 100.1 | 9%, 38 W |
| lsfg-vk installed, no profile, capped 50 | 50.0 | 50.0 | 3%, 23 W |
| Smooth Motion, FIFO uncapped | 60.0 | 119.9 | 9%, 44 W |
| lsfg-vk 2x, FIFO uncapped | 60.0 | 120.0 | 12%, 42 W |
| no generator, immediate uncapped | 9461 | — | 98% |
| Smooth Motion, immediate uncapped | 1027 | 2038 | 98% |

Under FIFO at 120 Hz both generators hold the application to half the refresh rate (60) and
present 120. lsfg-vk forces FIFO by default (`override_present_mode = true`).

### Generator cost per real frame at 2560x1440

| Generator | Cost | Source |
|---|---|---|
| Smooth Motion 2x | ~0.87 ms | vkcube immediate: 1000/1027 − 1000/9461 ms |
| lsfg-vk 2x | 1.06 ms | `lsfg-vk-cli benchmark -w 2560 -h 1440 -m 2` |
| lsfg-vk 2x performance | 0.66 ms | `… -m 2 -p` |
| lsfg-vk 2x flow 0.5 | 0.47 ms | `… -m 2 -f 0.5` |
| lsfg-vk 2x performance, flow 0.5 | **0.36 ms** | `… -m 2 -p -f 0.5` |
| lsfg-vk 3x performance | 1.07 ms | `… -m 3 -p` |

The two sources use different methods, so treat Smooth Motion vs lsfg-vk as approximate. All
of them are small next to the Neural Forge model (`helper` ≈ 11.5–13.5 ms per model frame).

### Neural Forge + generator (vkcube, FIFO, 120 Hz)

| Config | Real fps | Displayed fps | NF composites | GPU |
|---|---|---|---|---|
| NF only | 120 (FIFO-bound) | 119.4 | 120/s | 76%, 162 W |
| NF + Smooth Motion | 60.6 | 117.7 | 58/s | 42%, 107 W |
| NF + lsfg-vk 2x | 61.0 | 119.7 | — | 47%, 113 W |
| NF + lsfg-vk 2x performance | 64.1 | 119.2 | — | 21%, 55 W |
| NF + lsfg-vk 2x flow 0.5 | 61.3 | 119.7 | — | 43%, 106 W |
| NF + lsfg-vk 2x performance, flow 0.5 | 61.4 | 119.9 | — | 42%, 105 W |
| NF + lsfg-vk 3x performance | 40.7 | 119.9 | — | 32%, 87 W |

With the generator below it, Neural Forge composites only real frames. The display still
gets ~120 fps while the model runs about half as often, so GPU load falls from 76% to
~42%. The GPU-load differences between lsfg-vk modes are within sampling noise at this load;
use the benchmark costs above to rank them.

### GTA V Enhanced (Neural Forge + Smooth Motion, correct order)

Recorded 08:27–08:34 while the session was being streamed over GNOME Remote Desktop (one NVENC
session, ~37 fps), which adds the same small load to every configuration.

| | Real fps (NF `[present]`, above the generator) | NF `[sync]` total | GPU |
|---|---|---|---|
| NR on | mostly 55–60 (range 44–85) | 19–22 ms | 91%, 193 W average |
| NR off (08:32:27–32) | 81–84 | — | — |

Displayed fps in GTA was not captured. The Mesa counter's file is shared by the game and
`SocialClubHelper.exe`, and the two processes overwrote each other. There is no same-scene
GTA baseline without a generator, and no GTA run with lsfg-vk. Image quality and input
latency were not measured.

## What broke and why

- **Wrong order by default.** Both generators sat above Neural Forge until
  `VK_INSTANCE_LAYERS` fixed the order (see above).
- **Upstream dlssnr interference.** The upstream package's
  `~/.config/environment.d/dlssnr.conf` forced
  `VK_INSTANCE_LAYERS=VK_LAYER_NV_dlssnr:VK_LAYER_NV_present` into every program in the
  session. Upstream has been removed from the rig. A generator-derived variable survives
  `systemctl --user unset-environment`; it only clears with `systemctl --user daemon-reload`.
  Programs started before that (GNOME Shell, and a Steam started from it) keep the old value
  until the next login.
- **Shared log files.** `NVPRESENT_LOG_FILE` and the Mesa `output_file` are opened by every
  Vulkan process in the launch (game, Social Club helper) and overwrite each other.
  Smooth Motion's log to stderr (journal) is reliable.

## Recommendation

- Put a frame generator below Neural Forge. It removes most of the model's cost from the
  displayed frame rate: at 120 Hz the model only has to keep up with 60 real fps.
- Prefer **lsfg-vk 2x, performance mode, flow scale 0.5**. It is the cheapest measured mode
  (0.36 ms per real frame at 1440p versus ~0.9 ms for Smooth Motion) and reached 119.9
  displayed with Neural Forge on vkcube.
- Smooth Motion is the no-install fallback.
- Still to confirm in GTA with the two-counter method:
  - displayed fps;
  - a no-generator baseline in the same scene.

Launch options (the order in `VK_INSTANCE_LAYERS` is required):

```text
NEURAL_FORGE_ENABLE=1 LSFGVK_PROFILE=2x-perf-fs50 VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_LSFGVK_frame_generation %command%
NEURAL_FORGE_ENABLE=1 NVPRESENT_ENABLE_SMOOTH_MOTION=1 VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_NV_present %command%
```

lsfg-vk is inactive unless `LSFGVK_PROFILE` names a profile: every profile in
`~/.config/lsfg-vk/conf.toml` has an empty `active_in`. `DISABLE_LSFGVK=1` stops the layer
from loading at all.

Building a frame generator into Neural Forge is out of scope for this spike.
