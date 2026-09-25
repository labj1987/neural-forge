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

### GTA V Enhanced: in-game benchmark, unattended

GTA was launched without Steam's launch options:
- through the same SLR 4 `_v2-entry-point` and Proton-CachyOS that Steam uses, with Steam
  running;
- with a chosen environment per run;
- with `-benchmark -benchmarkIterations 1 -benchmarkFrameTimes`
  ([Rockstar's parameter list](https://support.rockstargames.com/articles/2VjbVziQCiTiiVhDbmnexc/full-list-of-command-line-parameters-for-grand-theft-auto-v-on-pc)).

The game writes `Documents\Rockstar Games\GTAV Enhanced\Benchmarks\Benchmark-*.txt` and
per-pass frame times, then exits by itself. In-game settings were left as they were:
2560x1440, `VSync=1`, `FrameLimit=0`, game frame generation off.

How the numbers were taken:
- **Real fps:** pass-4 frame count ÷ the sum of its frame times, from GTA's own file.
- **Displayed fps:** MangoHud placed *below* the generator, logging per frame. It writes one
  file per process, so `GTA5_Enhanced_*.csv` is not clobbered by `SocialClubHelper.exe`.
  The count is frames within pass 4's wall-clock window ÷ window length.
- **Instantaneous fps:** not used. A generator presents a generated frame and a real frame
  back to back, so averaging per-frame fps overstates the displayed rate.
- **Alignment check:** with no generator, displayed matched real within 0.6 fps in every run.

Pass 4 is the long free-roam pass (about 117 s).

| Config | Real fps | Displayed fps | NF composites/s | GPU |
|---|---|---|---|---|
| No Neural Forge | 92.6 | 93.2 | — | 66%, 141 W |
| NF only (run 1 / run 2) | 60.6 / 60.6 | 60.8 / 60.9 | 62.3 | 88%, 188 W |
| **NF + Smooth Motion** | **58.0** | **116.4** | 59.0 | 91%, 189 W |
| NF + lsfg-vk 2x | 54.1 | 109.0 | 54.6 | 87%, 188 W |
| NF + lsfg-vk 2x performance | 55.6 | 112.1 | 55.8 | 89%, 186 W |
| NF + lsfg-vk 2x flow 0.5 | 56.1 | 112.5 | 56.0 | 87%, 183 W |
| NF + lsfg-vk 2x performance + flow 0.5 | — | — | — | crashes at Game Init, 3/3 |
| lsfg-vk 2x performance + flow 0.5 alone | 59.1 | 118.6 | — | 47%, 111 W |

Averages across all 5 passes follow the same order.

What these numbers show:
- Neural Forge costs the game 32 real fps here (92.6 → 60.6).
- A generator below it recovers the displayed rate. Smooth Motion gives the most: 116.4
  displayed for 2.6 real fps.
- Every working lsfg-vk mode costs 4.5–6.5 real fps with Neural Forge and ends up 4–7
  displayed fps behind Smooth Motion. The per-frame costs measured on vkcube do not carry
  over to this game with the model running.

The performance mode + flow 0.5 crash:
- Each setting runs fine with Neural Forge on its own, and the combination runs fine without
  Neural Forge.
- Together they crash GTA while it initialises, with no exception in the game process and a
  "Game Init" crash context.
- It was not investigated further, because that mode is not needed.

Not measured: input latency and image quality.

### Refresh rate: 120 Hz vs 288 Hz

Every run above used the TV at 2560x1440, 119.998 Hz + VRR. The TV also runs 288.001 Hz + VRR,
so the set was repeated there.

vkcube under vsync with Smooth Motion:
- at 120 Hz: 60 real / 120 displayed;
- at 288 Hz: 144.2 real / 288.0 displayed.

The ceiling is half the refresh rate.

GTA benchmark, pass 4 at 288 Hz:

| Config | Real fps | Displayed fps | GPU |
|---|---|---|---|
| No Neural Forge | 85.0 | 85.8 | 73%, 143 W |
| NF only | 59.9 | 60.4 | 90%, 190 W |
| NF + Smooth Motion, NR on | 59.7 | 119.8 | 93%, 195 W |
| NF + Smooth Motion, NR off | 91.7 | 185.1 | 76%, 154 W |

Neural Forge's ~60 real fps is its own limit, not refresh pacing: it is the same at 120 Hz
(60.6) and 288 Hz (59.9). With NR on, the refresh rate makes no difference: Smooth Motion
doubles about 60 to about 120 either way. With NR off, the 185 generated-plus-real frames only
reach the screen at 288 Hz; at 120 Hz the display caps them at 120.

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
  Smooth Motion's log to stderr (journal) and MangoHud's per-process logs are reliable.
- **lsfg-vk performance mode + flow 0.5 with Neural Forge** crashes GTA at startup (see above).
- `LSFGVK_ENV=1` switches lsfg-vk to reading every setting from environment variables, which
  silently ignores the selected profile's values.

## Recommendation

**Use Smooth Motion below Neural Forge for GTA.** In the game's own benchmark it shows 116.4 fps
against 60.9 for Neural Forge alone, for 2.6 real fps and no extra install. lsfg-vk works as
well, but it trails Smooth Motion in every mode that runs, and its cheapest mode crashes
alongside Neural Forge.

```text
NEURAL_FORGE_ENABLE=1 NVPRESENT_ENABLE_SMOOTH_MOTION=1 VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_NV_present %command%
```

The order in `VK_INSTANCE_LAYERS` is required. Without it the generator sits above Neural
Forge.

lsfg-vk alternative (2x flow 0.5, the best working lsfg-vk mode):

```text
NEURAL_FORGE_ENABLE=1 LSFGVK_PROFILE=2x-fs50 VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural:VK_LAYER_LSFGVK_frame_generation %command%
```

lsfg-vk is inactive unless `LSFGVK_PROFILE` names a profile: every profile in
`~/.config/lsfg-vk/conf.toml` has an empty `active_in`. `DISABLE_LSFGVK=1` stops the layer
from loading at all.

Building a frame generator into Neural Forge is out of scope for this spike.
