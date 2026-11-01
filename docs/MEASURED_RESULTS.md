# Measured results

The current measurements on the test machine, RTX 5070 (12 GB) with Proton, each with its date. How they were
taken: [RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md); the dated record of every test:
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md).

## GTA V Enhanced, before the upscaler

GTA V Enhanced's built-in benchmark on the test machine: RTX 5070 (12 GB), 2560x1440, DLSS Balanced,
DLSS Frame Generation 4x with Preset C, ray tracing on, script mods off, pass 4 (about 117 s), default
settings (keep the game's sky on, one pass). Real fps is the game's own frames; shown fps counts the
generated ones too. The full network: one run, 2026-10-28; the off row: one run, 2026-10-16; the small
network: three runs, 2026-10-19 (the 2026-10-28 runs of those two are the frame-generation-off ones
below: the game turned frame generation off in them).

| Model | Real fps | Shown fps | Network GPU time on the frames it runs | Weights in video memory |
|---|---|---|---|---|
| Off (Neural Forge not loaded) | 76.8 | 309.0 | - | - |
| Full network | 62.1 | 249.4 | 6.8 ms | 140 MB |
| Small network | 73.4 | 295.4 | 3.9 ms | 108 MB |

With frame generation off (2026-10-28, one run each, real fps): off 88.0, small network 76.9 (3.4 ms),
full network 75.1 (6.1 ms).

The network runs on every frame while the camera is still and on every second frame while it moves (the
edit cache carries it between), so about half the frames pay its time in this benchmark.

**Model resolution** before the upscaler, in the same game with frame generation off (DLSS's input
1486x836; added detail measured on a fixed patch of a building's facade, against the model off):

| Model resolution | Real fps | Network GPU time | Added detail |
|---|---|---|---|
| Off | 90.3 | - | x1.00 |
| Full | 77.1 | 6.05 ms | x1.28-1.32 |
| Quality | 84.4 | 3.20 ms | x1.01 |
| Performance | 85.2 | 3.04 ms | x0.97-0.98 |
| Ultra Performance | 86.8 | 2.65 ms | x1.10 |

At DLSS's input size the smaller model resolutions keep little of the detail the model adds: before the
upscaler, use Full.

## GTA V Enhanced at 4K

The same benchmark at 3840x2160 (DLSS's input 2228x1253), 2026-10-29, three runs each (median), the
game's frame rate limit at 120 for the frame generation runs:

| Model | Frame generation 4x: real fps | Shown fps | Frame generation off: real fps | Video memory peak |
|---|---|---|---|---|
| Off (Neural Forge not loaded) | 62.6 | 251.9 | 84.3 | 10.4 GB |
| Full network | 39.5 | 158.5 | 46.3 | 11.8 GB |

The network takes 12.9 ms here against 6.5 ms at 1440p: about 1.9 times, for 2.25 times the pixels. It
costs the game more at 4K than that because at 1440p the game leaves the GPU partly idle (65 % busy with
the model off, frame generation off), and most of the network's time fits in that idle time; at 4K the
GPU is already 95 % busy.

## Crimson Desert, after DLSS (Ray Reconstruction)

Crimson Desert with DLSS Ray Reconstruction and frame generation on, 2560x1440, placement Auto: the model
runs after DLSS on its 2560x1440 output. Shown fps counts the generated frames; network GPU time is per frame
the model runs on. No missed frames and no GPU fault in any run.

| Network, model resolution | Shown fps | Network GPU time |
|---|---|---|
| Model off | 183 | - |
| Full network, Full | 120-122 | 15.8 ms |
| Small network, Full | 140 | 9.2 ms |
| Full network, Quality | 143 | 8.3 ms |
| Full network, Performance | 150-153 | 5.4 ms |
| Full network, Ultra Performance | 161 | 3.8 ms |
| Full network on the finished frame (Placement: Finished frame) | 36 | - |

At the output size, Quality keeps the detail the model adds at Full and Performance most of it; Ultra
Performance is a little softer. With Ray Reconstruction off, the same game runs 176 fps shown with the
model before the upscaler (Auto), 133 with it after DLSS (Placement: After DLSS) and 41 on the finished
frame. With the game's HDR on (Proton-GE, Wayland) the model after DLSS runs as in SDR: the same holds per
second and the same size of change.

Hogwarts Legacy with Ray Reconstruction (switched on in its menu) and frame generation 4x, Full: 137 fps
shown with the model after DLSS (25 on the finished frame), 15.2 ms network GPU time.

## The network on its own

The network's time per frame on its own on an idle GPU at 1486x836 (DLSS's input in GTA V at 1440p, DLSS
Balanced; `crates/native/tests/rig.rs`): full 5.35 ms run back to back, small about 2.85 ms (scaled from
the per-kernel profile, which has no back-to-back run for the small network). In the game it
also waits for the game's own GPU work. Games
are tested at 2560x1440 only (see the README's [Known issues](../README.md#known-issues) for 4K). Details and the record of
every measurement: [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) and
[NATIVE_BACKEND.md](NATIVE_BACKEND.md).
