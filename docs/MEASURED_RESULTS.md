# Measured results

The current measurements on the test machine, RTX 5070 (12 GB) with Proton, at 3840x2160 (4K), each with its
date and settings; games that have HDR run with it on. The GPU draws 2.25 times the pixels of 2560x1440 at 4K,
so every frame rate here is lower than at 1440p, with or without the model: the earlier 1440p measurements are
kept at the end, marked as such, and are not comparable with the 4K ones. How they were taken:
[RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md); the dated record of every test:
[HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md).

## GTA V Enhanced at 4K, before the upscaler

GTA V Enhanced's built-in benchmark on the test machine: RTX 5070 (12 GB), 3840x2160, DLSS Balanced (DLSS's
input 2228x1253), DLSS Frame Generation 4x with Preset C, ray tracing on, the game's frame rate limit at 120
(it keeps frame generation on; the game stays under it), script mods off, pass 4 (about 117 s), default
settings (keep the game's sky on, one pass). GTA V has no HDR. Real fps is the game's own frames; shown fps
counts the generated ones too. One run each, 2026-11-10.

| Model | Real fps | Shown fps | Network GPU time on the frames it runs | Video memory peak | Weights in video memory |
|---|---|---|---|---|---|
| Off (Neural Forge not loaded) | 58.1 | 233.3 | - | 10.3 GB | - |
| Full network | 36.7 | 147.5 | 13.1 ms | 11.8 GB | 140 MB |
| Small network | 41.0 | 165.0 | 7.6 ms | 11.8 GB | 108 MB |

With frame generation off (4K, 2026-11-10, one run each, real fps): off 79.0, small network 51.3 (6.8 ms),
full network 44.9 (11.9 ms).

The network runs on every frame while the camera is still and on every second frame while it moves (the
edit cache carries it between), so about half the frames pay its time in this benchmark. It takes 13.1 ms
on the 4K input against 6.8 ms on the 1440p one (1486x836): about 1.9 times, for 2.25 times the pixels. It
costs the game more at 4K than that because at 1440p the game leaves the GPU partly idle (65 % busy with the
model off, frame generation off), and most of the network's time fits in that idle time; at 4K the GPU is
already about 90 % busy with the model off. With the model on, the card's 12 GB are close to full.

## Crimson Desert at 4K with HDR, after DLSS (Ray Reconstruction)

Crimson Desert with DLSS Ray Reconstruction and its dynamic frame generation on, 3840x2160 (DLSS Balanced, a
2272x1280 render), the game's HDR on (HDR10; Proton-GE with `PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1`),
placement Auto: the model runs after DLSS on its 3840x2160 output. Shown fps is the frames presented per
second (the generated ones included) over 60 s in the world; network GPU time is per frame the model runs on.
One run each, 2026-11-10; no missed frames and no GPU fault in any run.

| Network, model resolution | Shown fps | Network GPU time | Video memory peak |
|---|---|---|---|
| Model off | 105 | - | 9.8 GB |
| Full network, Full | 70 | 32.6 ms | 11.8 GB |
| Small network, Full | 79 | 19.4 ms | 11.6 GB |
| Full network, Quality | 81 | 17.8 ms | 11.6 GB |
| Full network, Performance | 84 | 10.2 ms | 11.2 GB |
| Full network, Ultra Performance | 87 | 5.9 ms | 11.1 GB |
| Full network on the finished frame (Placement: Finished frame) | 14 | - | 11.8 GB |

At the output size, Quality keeps the detail the model adds at Full and Performance most of it; Ultra
Performance is a little softer (compared at 1440p). With Ray Reconstruction off, the same game runs 102 fps
shown with the model before the upscaler (Auto; 12.9 ms on the 2272x1280 input), 68 with it after DLSS
(Placement: After DLSS) and 15 on the finished frame, against 124 with the model off. The model runs on
an SDR-mapped copy of the HDR frame, and its change is put back onto the HDR picture.

## Cyberpunk 2077 at 4K with HDR

Cyberpunk 2077's built-in benchmark (`-benchmark`, about 64 s) at 3840x2160, DLSS Auto (a 1920x1080 render),
DLSS Frame Generation 3x, ray tracing Ultra, the game's HDR on (HDR10 PQ; Proton-GE with
`PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1`), placement Auto. Shown fps is the benchmark's own average,
generated frames included; the model off is the layer loaded with the model switched off. One run each,
2026-11-10.

| Ray Reconstruction, path tracing | Where the model runs | Model off: shown fps | Model on: shown fps | Network GPU time | Video memory peak, model on |
|---|---|---|---|---|---|
| Off, off | Before the upscaler (1920x1080) | 113.5 | 89.8 | 9.8 ms | 11.8 GB |
| On, off | After DLSS (3840x2160) | 115.0 | 53.3 and 44.7 (two runs) | 32 ms | 11.8 GB |
| On, on | After DLSS (3840x2160) | 67.2 | 11 | - | 11.8 GB |

The model held every frame it was given, with no missed frame. With Ray Reconstruction the network runs on
the 3840x2160 output and the card's 12 GB are full: both runs fell to 15-17 fps for stretches. With path
tracing as well the game runs out of video memory and stays at a few frames per second (the README's
[Known issues](../README.md#known-issues)). A lower model resolution or the small network needs less video
memory (Crimson Desert above); these were not measured in Cyberpunk 2077.

## The network on its own

The network's time per frame on its own on an idle GPU at 1486x836 (DLSS's input in GTA V at 1440p, DLSS
Balanced; `crates/native/tests/rig.rs`): full 5.35 ms run back to back, small about 2.85 ms (scaled from
the per-kernel profile, which has no back-to-back run for the small network). In the game it
also waits for the game's own GPU work: at 4K the full network takes 13.1 ms on GTA V's 2228x1253 input,
12.9 ms on Crimson Desert's 2272x1280 one and about 32 ms on a 3840x2160 output after DLSS. Details and the
record of every measurement: [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) and
[NATIVE_BACKEND.md](NATIVE_BACKEND.md).

## Earlier measurements at 1440p

At 2560x1440, before the 4K measurements above; not comparable with them.

### GTA V Enhanced at 1440p, before the upscaler

The same benchmark at 2560x1440 (DLSS's input 1486x836), without the frame rate limit. The full network:
one run, 2026-10-28; the off row: one run, 2026-10-16; the small network: three runs, 2026-10-19.

| Model | Real fps | Shown fps | Network GPU time on the frames it runs |
|---|---|---|---|
| Off (Neural Forge not loaded) | 76.8 | 309.0 | - |
| Full network | 62.1 | 249.4 | 6.8 ms |
| Small network | 73.4 | 295.4 | 3.9 ms |

With frame generation off (1440p, 2026-10-28, one run each, real fps): off 88.0, small network 76.9
(3.4 ms), full network 75.1 (6.1 ms).

**Model resolution** before the upscaler, in the same game at 1440p with frame generation off (DLSS's input
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

### Crimson Desert and Hogwarts Legacy at 1440p, after DLSS (Ray Reconstruction)

Crimson Desert with DLSS Ray Reconstruction and frame generation on, 2560x1440, SDR, placement Auto: the
model after DLSS on its 2560x1440 output.

| Network, model resolution | Shown fps | Network GPU time |
|---|---|---|
| Model off | 183 | - |
| Full network, Full | 120-122 | 15.8 ms |
| Small network, Full | 140 | 9.2 ms |
| Full network, Quality | 143 | 8.3 ms |
| Full network, Performance | 150-153 | 5.4 ms |
| Full network, Ultra Performance | 161 | 3.8 ms |
| Full network on the finished frame (Placement: Finished frame) | 36 | - |

With Ray Reconstruction off at 1440p: 176 fps shown with the model before the upscaler (Auto), 133 with it
after DLSS and 41 on the finished frame.

Hogwarts Legacy at 1440p with Ray Reconstruction (switched on in its menu) and frame generation 4x, Full:
137 fps shown with the model after DLSS (25 on the finished frame), 15.2 ms network GPU time.
