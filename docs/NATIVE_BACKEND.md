# Native model backend (candidate 3.0)

The plan: run the neural rendering network inside the layer, on the game's own device, with
[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (MIT), and retire the Windows helper, the
Wine/Proton runner, the shared-memory hand-off, the NGX DLL at run time and the caller-identity spoof.
Branch `native-backend` only; nothing merges, tags or releases until Alex has judged the results.

This file is written as the work goes. Every attempt, including the failed ones, is in "Attempt
log" so the next one starts from it.

## Phase 0: go or no-go (2026-10-07)

**Verdict: go, with two cautions.** (Alex: go, 2026-10-07.) All three gates hold on the test machine:

| Gate | Needed | Measured |
|---|---|---|
| Model directory loads and benches | yes | yes: `nr::Model` loads it with SHA-256 verification on; bench runs at all four sizes |
| Network time at 1485x836 (1440p Balanced input) | <= 5.5 ms | **5.496-5.500 ms** (minimum of 40, three runs) |
| Network time at 2228x1253 (4K Balanced input) | <= 13 ms | **10.68-10.70 ms** |
| Output matches the helper's | within rounding | **bit-exact** on six GTA V frames (single frame, no history) |

The cautions:

1. **1485x836 passes by 0-4 µs.** The median there is 5.58 ms, and with counter chaining off the
   minimum is 5.93 ms. On an idle GPU the network alone is about 0.7 ms slower than the helper's
   in-game 4.8 ms at that size. The native path saves the hold's other costs (`capture_wait` 3.6-4.4
   ms, optical flow about 0.45 ms, the `C`/`W` copies 0.68/0.52 ms GPU; ARCHITECTURE.md 4.8), so
   the frame rate can still come out ahead, but success criterion 2 (fps equal or better at 1440p)
   is not settled until Phase 3 measures it in game. At 4K the network is clearly faster than the
   helper (10.7 vs 13.4 ms).
2. **Only the single-frame network is proven equal.** The comparison runs the first frame after a
   history reset on both sides. With history, NGX does the reprojection itself from Neural Forge's
   optical flow; the native path will do it from the game's DLSS motion vectors with OpenDLSS-NR's
   demo rules (frame.md). Those can differ, and nothing in Phase 0 compares them. Phase 1's capture
   pairs are where that shows.

### 0.1 Driver capabilities

RTX 5070 (12 GB), driver 615.71.09, Vulkan 1.4.351.

| Extension / feature | Present |
|---|---|
| `VK_KHR_cooperative_matrix` (rev 2) | yes |
| `VK_NV_cooperative_matrix2` (all seven features) | yes |
| `VK_EXT_shader_float8` (`shaderFloat8`, `shaderFloat8CooperativeMatrix`) | yes |
| `VK_NV_cuda_kernel_launch` (rev 2) | yes |
| `VK_KHR_pipeline_executable_properties` | yes |
| `VK_KHR_shader_clock` | yes |
| `VK_NV_shader_sm_builtins` | yes |
| `maintenance4` | yes |
| Cooperative matrix E4M3 x E4M3 with f16 accumulation (K 32) | yes (also f32 accumulation, and E5M2) |

The cooperative-matrix component types come from dlss5vk's `DLSS5VK_LIST_EXTENSIONS=1` listing of
the coopmat2 flexible-dimension table (workgroup and subgroup scope, 32-256 invocations). Because
`VK_NV_cuda_kernel_launch` is present, the PTX route is used and the GLSL fallback numbers were not
needed.

### 0.2 OpenDLSS-NR: the contracts

Commit **`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`** (the same one OPENDLSS_REVIEW.md read; four
commits, no releases). Read in full: README.md, docs/network.md, numerics.md, weights.md,
execution.md, frame.md, demo/README.md, checked against the code.

**Input**: f32 `[field rows][16]` over the padded field:

| Lane | Content |
|---|---|
| 0-2 | Gaussian noise, Box-Muller from a hash of the padded-field `(x, y)` and `seed`, each rounded to half |
| 3 | 1.0 |
| 4-6 | the centred proxy `f16((f16(c) - 0.5) * 0.125)`; proxy `c` = sRGB of the shoulder of `scene / paperWhite` (shoulder above 0.75: `0.75 + 0.25 (1 - exp(-5.770780 (v - 0.75)))`), rounded to half |
| 7-9 | the centred reprojected previous **output** (Catmull-Rom, five taps, clamped to the valid rectangle, no colour clamp), or a copy of 4-6 where there is no history (first frame, reset, or the pixel came from off screen) |
| 10 | style / 128 (0, 1 natural, 2 cinematic) |
| 11 | local tone |
| 12-14 | auto-mask on: `1, skin (< 0 means structure), structure`; off: `structure, -1, -1` |
| 15 | 0 |

Seed: the demo uses its frame counter. The padding (right and bottom) mirrors the image
(`2·valid - x - 2`) with its own noise.

**Output**: f32 `[field rows][4]`: an RGB residual on the proxy and one temporal-blend logit. The
composite: `neural = clamp(proxy + rgb / 4, 0, 1)`, then, with history,
`weight = clamp(sigmoid(logit) · clamp(blendScale, 0, 1), 0, 1)` and `neural = lerp(neural,
history, weight)`; `history' = truncate-to-half(neural)`. `blendScale` is the model's one
`block70.layer0.blend_scale` value (0.7397). The demo then applies style, intensity and
`upgradeToneMap` (luminance ratio plus Oklab hue), the same rule as Neural Forge's own composite.

**Field padding**: per axis, `align = 2^reductions(valid)` (64, or 128 when level 0 needs the extra
step), `field = max(320, alignUp(valid, align))`, plus one more `align` on the width when both axes
are multiples of `4·align`; any valid size of at least 33 px is accepted. The field decides the
window grid, so it changes pixels inside the image and cannot be chosen by the host. For the sizes
here: 1485x836 and 1486x836 -> 1536x896, 2228x1253 -> 2304x1280, 2560x1440 -> 2560x1472,
3840x2160 -> 3840x2176.

**Adopting a device** (demo `NrPass`): `vk::Context(instance, physical, device, family,
queueIndex)` loads every function through volk's process-global pointers and repoints them at that
device; it creates its own command pool, two 8192-set descriptor pools, a 256 MiB host-visible
staging buffer, and destroys only its own objects. Per frame the demo records four pre-recorded
secondary command buffers (`[history parity][NR off, on]`) into the host's primary buffer, ordered
by barriers alone: no semaphores, no fences. Everything outside the frame (uploads, the warm-up
run, resize, the barrier fallback) uses `vkQueueWaitIdle` or `vkDeviceWaitIdle` and assumes the
renderer is idle; on a one-queue family it shares the renderer's queue on that assumption. Only one
network frame may be in flight: activations, chain counters and the status word are single
instances. Device requirements (`vk::DeviceRequirements`): the four kernel extensions above plus
three tooling ones that no shader uses, and `shaderInt16/64`, 8/16-bit storage, `shaderFloat16`,
`shaderInt8`, the Vulkan memory model with device scope, `hostQueryReset`, `bufferDeviceAddress`,
`subgroupSizeControl`, `computeFullSubgroups`, `synchronization2`, `maintenance4`, and
robustBufferAccess **off** (about 15% cost).

What this means for Phase 1, all of which the demo never had to solve: calls must go through the
layer's next-layer dispatch, not volk; the out-of-frame `WaitIdle`s must become bounded waits under
the layer's queue lock; the chain-timeout status word is read without a fence and the fallback is a
process-wide switch; and the inputs come from DLSS's registered images (HDR colour, `RG16F` motion,
the 1x1 exposure), not from Filament.

### 0.3 dlss5vk on Ubuntu 26.04

In the scratch clone (`~/scratch/OpenDLSS-NR/linux/`, not in this repo): `CMakeLists.txt`,
`build.sh`, `run_ptx.py`, `linux-port.patch` and `load_check.cpp`.

- Pins from `scripts/fetch_tools.ps1`: glslang 16.6.0 (official Linux release binary),
  Vulkan-Headers `v1.4.363`, volk `vulkan-sdk-1.4.357.0`. CMake 3.31.12 portable (the machine has
  no system cmake), Ninja 1.13.2 and GCC 15.2 from the system.
- Shaders: `glslang -V --target-env vulkan1.3 -I shaders`, one `.spv` per `.comp` (13). PTX: the
  generator invocations of `build_shaders.ps1` (78 files) under Python with `-I`, through
  `run_ptx.py`, because `-I` drops the script's directory from the import path.
- Source changes for GCC (`linux-port.patch`): `#include <cstring>` in `vk_context.h`; the nested
  `Kernels::Chain` struct moved to namespace scope as `KernelChain` (GCC refuses a default argument
  of a nested class with default member initialisers), with `using Chain = KernelChain;` left in
  the class. Nothing else.
- `load_check` (not upstream): constructs `nr::Model` with hash verification on, which `bench`
  does not (`runBench` passes `false`).

### 0.4 The model directory

The imported `nvngx_dlssnr.dll` is **build 310.8.0.0** (`VS_FIXEDFILEINFO`), the only build the
graph implements.

**Where the weights are.** Only interface metadata and data were read: the PE section table, the
resource directory, and one data resource's own framing. No code in the DLL was read. The DLL has
two resources, `RT_VERSION` and `RT_RCDATA "WEIGHTS_HT"` (147,695,410 bytes, byte entropy 6.55
bits: stored, neither compressed nor encrypted). Its layout:

| Field | Bytes |
|---|---|
| total length (equals the resource size) | u64 |
| then 153 records, end to end: | |
| name length `n` | u64 |
| name `blockB.layerL.parameter` | `n` |
| record length `a`, counted from the end of this field | u64 |
| `a` again | u64 |
| tensor length | u64 |
| kind, always 1 | u32 |
| the tensor | tensor length |
| trailer (u64 0, u64 1, u32 tensor length / 2), not read | 20 |

The 153 tensors are `block0..70.layer0..4.layer` plus `block70.layer0.blend_scale`, 147,683,778
bytes (140.84 MiB) in total.

**The extractor**: `neural-forge-cli extract-model DIR` (`crates/supervisor/src/model.rs`; `DIR`
holds the DLL or is the DLL). It checks the build (anything but 310.8.0 is refused with the build it
found), walks the records refusing any surprise (a total that disagrees, a kind other than 1,
differing record lengths, a tensor overrunning its record, a name outside the pattern, other than
153 tensors, blocks other than exactly 0-70), and writes `paths::data_dir()/model`:
`manifest.json` plus `model/<stage>.e4m3`, each file through a temporary name and a rename. Eleven
stages by resolution: `encoder32` (blocks 0-4), `encoder64` (5-8), `encoder128` (9-14),
`encoder256` (15-22), `encoder512` (23-30), `vit` (31-38), `decoder512` (39-47), `decoder256`
(48-55), `decoder128` (56-61), `decoder64` (62-65), `decoder32` (66-70); tensors 16-byte aligned
in each file. The manifest also records the source DLL's build and SHA-256. Stage hashes are
upper-case hex, because `nr::Model` compares its own upper-case digest as a string. The output
directory is ignored in `.gitignore` (`/model/`, `*.e4m3`) in case it is ever written inside the
tree. Five unit tests run it on a synthetic PE image. Extraction takes 0.26 s.

**Checks, in the order the handoff set:**

1. Every tensor's length against OpenDLSS-NR's layouts (`fusedLayout`, `preFusedLayout`,
   `upsampleFusedLayout`, `postFusedLayout`, and the split-512, ViT and transition slices the graph
   reads; script `~/scratch/probe/layoutcheck.py`): 142 of 153 equal the layout's sum exactly. Ten
   (block 4, block 30 `layer4`, blocks 31-38 `layer0`) are the sum plus 16 bytes, and those 16
   bytes are zero in all ten: the trailing padding weights.md lists. `blend_scale` (2 bytes)
   decodes to 0.73975, the 0.7397 OpenDLSS-NR names for the shipped model.
2. `nr::Model` loads the directory with SHA-256 verification on (`load_check`): ok.
3. `dlss5vk bench` runs on it at all four sizes (0.5), and `parity` is bit-exact against the
   helper (0.6).

### 0.5 Network time on the test machine

`dlss5vk bench`, idle GPU (only the desktop and remote-desktop daemon, 0% utilisation, 854 MiB in
use before), minimum and median over 40 frames, three runs. VRAM is the peak of `nvidia-smi`
sampled every 100 ms during the run, minus the idle 854 MiB (the bench's whole process: weights,
activations, staging).

| Valid size | Field | Min, ms (3 runs) | Median, ms | `DLSS5VK_CHAIN=0` min / median | VRAM | Dispatches |
|---|---|---|---|---|---|---|
| 1485x836 | 1536x896 | 5.498 / 5.500 / 5.496 | 5.58 | 5.934 / 5.994 | +945 MiB | 249 |
| 2228x1253 | 2304x1280 | 10.690 / 10.698 / 10.683 | 10.90 | 10.941 / 11.172 | +1511 MiB | 249 |
| 2560x1440 | 2560x1472 | 13.408 / 13.361 / 13.351 | 13.76-13.89 | 13.656 / 13.979 | +1810 MiB | 249 |
| 3840x2160 | 3840x2176 | 30.017 / 30.046 / 30.059 | 31.31-31.36 | 30.101 / 31.262 | +3474 MiB | 248 |

No chain watchdog timeout in any run. About 6% slower than upstream's RTX 4070 SUPER at 1440p and
4K (12.6 / 29.3 ms), which fits the smaller GPU. Chaining is worth 0.44 ms at 1485x836, 0.25-0.3
ms at the middle sizes, nothing measurable at 4K. The sm_89 PTX is JIT-compiled by the driver on
first use, and every module loaded (each run builds the full PTX route). `parity` resubmits the
production schedule three times and compares it with the same graph under barriers; it agreed on
every frame in 0.6, so repeated runs are identical on this GPU.

### 0.6 Against the helper

Six GTA V Enhanced frames from earlier `NEURAL_FORGE_PREUPSCALE=dump` captures (DLSS's colour input
at 1485x836 with the game's exposure; frames 3, 3722, 5089, 7679, 8892, 10011). Two older dumps
without `exposure.json` were skipped. No game was run for this. For each frame:

1. Encode it exactly as the layer does (`scripts/hdr_encode.py encode DUMP opendlss --white 3`:
   the game's exposure, paper white 3, the shoulder, sRGB, half) and pad it to 1486 columns by
   repeating the last one, as the layer does for an odd width.
2. Send it to the helper with `trigger_helper_roundtrip --rgba16f` 32 times (the feature builds),
   wait 2 s (the helper resets the model's history after 500 ms without an answer), then once more:
   that answer is NGX's first frame after a reset, with no history. Helper settings as Alex has
   them: style 0, intensity 1, tone 1, structure 1, skin -1, auto-mask on, preset 0.
3. Run the same 1486x836 bytes through `dlss5vk parity` as a proxy fixture (conditioning style 0,
   tone 1, structure 1, skin -1, auto-mask on, seed 0) with the helper's answer as the f32 native
   output.

**Result: bit-exact on all six** (`composed RGB vs native output: bit-exact (3726888)`): every
half-float of `truncate-to-half(clamp(proxy + rgb/4))` equals the helper's answer. So NGX's answer
is the network's `neural` value, before any blend or tone upgrade, and the native network computes
it identically.

What had to match, found on the way (frame `d1`, 1485x836):

| Run | PSNR vs helper | Within one 8-bit code | Edit-field correlation |
|---|---|---|---|
| 1485 columns, seed 0 | 50.7 dB | 94.3% | 0.9957 |
| 1485 columns, seeds 1, 2, 3, 7, 42 | 47.2-47.5 dB | 85.6-86.2% | 0.990 |
| dlss5vk seed 0 against dlss5vk seed 1 | 47.5 dB | 85.7% | 0.991 |
| **1486 columns (the bytes NGX got), seed 0** | bit-exact | 100% | 1.0 |

- **The seed.** NGX's first frame after a reset uses the noise of seed 0. Any other seed differs
  from it by as much as two seeds differ from each other. Whether NGX's seed then counts frames
  (as the demo's does) is not tested here; with history in play it matters only for matching
  NGX's exact noise, not for the look.
- **The valid size.** The layer sends 1486 columns; with 1485 the field is the same (1536x896) but
  the extra real column changes the window contents, and the ViT spreads that over the whole frame
  (50.7 dB). The native path will see the true 1485 and does not need the padding column at all.
  The difference between the two is far below anything visible (50.7 dB, mean 0.002 on 0..1).

Files: `~/scratch/cmp/` on the test machine (`fx-dNw-s0/` fixtures, `cmp-*.png`: proxy, helper,
native and the difference x8 side by side). Not in the repo: they hold game frames and model
output.

### Not in this repo, on purpose

NVIDIA's weights are never committed, uploaded or packaged. The model directory lives in the user
data directory, the DLL stays where `import-binaries` put it, and the scratch clone, fixtures and
captures stay on the test machine.

## Phase 1: the network inside the layer (2026-10-07)

**Status: working in GTA V before the upscaler.** Every DLSS frame goes through the network on the
game's own device; the helper is not running. Failure paths tested. Open items at the end of this
section.

### 1.1 Vendoring and the build

- `third_party/opendlss-nr/`: upstream `src/` (minus `main.cpp` and `verify.cpp`), `shaders/`,
  `scripts/ptx/`, `LICENSE`, `NOTICE`, at `9d08f41`. `UPSTREAM.md` lists every local change, each also
  a patch in `patches/`: 0001 and 0002 the two GCC fixes, 0003 the asset loader (kernels come from
  memory, not a directory), 0004 the adoption inside a layer (next layer's proc addresses, bounded
  waits, no `vkDeviceWaitIdle` on an adopted device, a dispatchable-init hook, `CONCURRENT` buffers
  over two queue families). ATTRIBUTION.md has the row.
- `crates/native` (`neural-forge-native`): `build.rs` compiles the 13 GLSL kernels with the pinned
  glslang and runs the 78 PTX generator invocations of upstream's `build_shaders.ps1` at build time,
  embeds both (`.incbin`), and compiles the vendored C++ with the glue (`cpp/nf_native.cpp`) into the
  rlib. Nothing prebuilt is committed. `scripts/fetch-native-tools.sh` fetches glslang 16.6.0,
  Vulkan-Headers v1.4.363 and volk vulkan-sdk-1.4.357.0 (SHA-256 pinned) into `tools/native/`
  (ignored); CI runs it. A clean native build takes 3.5 s.
- libstdc++ is linked statically and nothing of the C++ is exported: the layer's `NEEDED` list is
  unchanged (`libgcc_s`, `libm`, `libc`, `ld-linux`), its dynamic symbols too. volk is compiled as C++
  in its own namespace (`VOLK_NAMESPACE`): as C its `vkGetInstanceProcAddr` and friends clashed with
  the layer's exported entry points. The layer grew from 2.0 MB to 8.9 MB (5.5 MB of it PTX). It now
  needs glibc 2.38 (GCC 15's libstdc++ and the vendored objects use `__isoc23_strtol`).
- The 32-bit layer does not link the native crate (`[target.'cfg(target_arch = "x86_64")']`); see 2.4.
- `crates/native/tests/rig.rs` (runs with `NEURAL_FORGE_NATIVE_RIG=<dlss5vk dump>`): a device made
  with the layer's additions, the network loaded on a compute-only family's queue and executed from
  a graphics queue, the head compared with dlss5vk's for GTA frame d1: **bit-exact**, three runs
  identical, no chain timeouts, 5.51 ms back to back (dlss5vk: 5.50 ms). The recorded secondary costs
  nothing against recording the graph straight into the primary (5.51 vs 5.60 ms).

### 1.2 Device setup

`NeuralForgeInstanceHooks::create_device` (`crates/layer/src/lib.rs`), on an NVIDIA device with the
pre-upscaler path on and `NEURAL_FORGE_BACKEND` native: `nf_native_device_extend` checks every
requirement and merges the network's features into the application's own chain. A flag whose Vulkan
1.1/1.2/1.3 aggregate structure the application chains is set there (and put back after the call);
otherwise the feature's own structure is prepended to a copy, and the extension added. Extensions
added: `VK_NV_cuda_kernel_launch`, `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
`VK_EXT_shader_float8`, and the promoted ones the device lists. The three tooling extensions
upstream enables (pipeline executable properties, shader clock, SM built-ins) are not: no kernel
uses them. robustBufferAccess is left as the game asked. A device that lacks a requirement, or a
request the driver refuses, is created exactly as the game asked, and the log names the first
missing requirement.

### 1.3 The game's device, and which queue does what

- **Loading** (the model, the weights' device copies, the kernels, the first run that makes the
  driver compile the PTX, the graph's recording): a queue the layer adds in a compute family without
  graphics (family 2 on the RTX 5070), used by the network's loader thread only. Buffers are
  `CONCURRENT` over that family and the game's graphics family. The first version used a queue of
  the game's graphics family and faulted the game's channel (Xid 32, device lost) while GTA rendered;
  the layer had already met this with its side queue (Xid 69, Crimson Desert). Nothing else uses the
  loader's queue, so no lock is needed there; every wait on it is bounded (5 s).
- **Every frame**: the game's own queue, inside the game's own DLSS submit, which the layer already
  splits (`preupscale::submit_around`, under the game's external synchronization of that queue: the
  layer is inside the game's `vkQueueSubmit` call). Order: the capture `C` (exposure, encode, the
  motion vectors' copy), `N[parity]` (preprocess, the recorded graph, composite), the write-back `W`
  (decode over the colour input), then DLSS. Each opens with a barrier from `ALL_COMMANDS/MEMORY_WRITE`
  to what it reads; the ordering argument is the helper hold's, minus the host round trip. No
  semaphores, and no CPU wait in the frame: the only wait is the next hold's bounded wait for the
  previous hold's `W` fence (which covers `C` and `N`, submitted before it on the same queue).
- **One network frame in flight**: activations, chain counters and the status word are single
  instances; frames on one queue in submission order never overlap, and the previous hold's work is
  complete before parameters for the same parity are written.
- **The network's functions** are loaded through the next layer's `vkGetInstanceProcAddr`, so its calls
  never pass through this layer; its queue and command buffers get the loader's dispatch pointer
  (`loader_data::initialize_object`).
- **Lazily**: the model is loaded on the first hold of the device DLSS runs on (GTA creates a dozen
  devices at start; loading on each cost 0.5 s and VRAM). Load 0.53-0.66 s with hash verification,
  build for 1485x836 0.73-1.15 s.

### 1.4 The frame path

`crates/layer/src/preupscale/native.rs`, shaders `native_preprocess.comp` and `native_composite.comp`
(ported from OpenDLSS-NR's demo shaders: lane layout, half roundings, noise hash, Catmull-Rom filter
and composite bit for bit; ATTRIBUTION.md).

- **Proxy**: the layer's existing encode (`preupscale_encode.comp`, `hdr.rs`): one implementation.
- **History rule**: the helper's (`history.rs`), moved into `neural-forge-protocol` and made generic,
  used by both: a reset after a frame that went to DLSS untouched, after 500 ms without a frame, or
  on another extent or identification. Plus per pixel: no history where the previous position is
  outside the frame, and none after a frame whose exposure was unusable (a GPU flag the composite
  writes).
- **Motion vectors** (measured on consecutive GTA V frames, `NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES`):
  DLSS's are in render pixels, y down, current to previous (`prev = x + mv`, scale +1 on both axes
  by a 7x7 search). They exclude the camera jitter: the warp error falls a further 20% with a global
  sub-pixel offset that changes every frame. The jitter DLSS is given is word 3 of the 310.x input
  kernel's parameters (x low, y high; found with `NEURAL_FORGE_PROBE_PARAMS=1`): a 23-frame Halton
  (2,3) sequence in ±0.5 px. Three consecutive pairs fit only `prev = x + mv + (jitter[t-1] -
  jitter[t])`, measured (-0.25, 0.31), (0.44, -0.50), (-0.94, 0.38) against (-0.25, 0.33), (0.50,
  -0.56), (-0.91, 0.33). Other DLSS versions' input kernels have no known jitter word: they get
  `jitter = 0` (history off by up to a pixel, which the network's blend weight then has to absorb).
- **Seed**: 0 on a frame without history, then counting, so the first frame after a reset is NGX's
  exactly (0.6).
- **Settings**: Phase 1 runs the defaults (style 0, tone 1, structure 1, skin -1, auto-mask, intensity
  1), the values Alex runs. The Model tab's mapping is 2.2.
- **Cost per frame** in GTA V at 1485x836: the hold's CPU time 0.08 ms (the helper path: about
  10 ms, of which 3.6-4.4 ms draining the game's queue); `W` on the GPU 0.11-0.12 ms; the network
  frame `N` (preprocess, network, composite) **6.46-6.66 ms** on the GPU (median per 300 holds, the
  `network_gpu_ms` field of the `[preupscale]` summary, `nat-3`), against 5.50 ms for the network alone
  on the idle GPU: sharing the GPU with the game costs about 1 ms, as the handoff expected.

### 1.5 The switch

`NEURAL_FORGE_BACKEND=native` (the default on this branch) or `helper`. With `helper` nothing of the
native path is reachable: no extensions added, no loader, the helper hold exactly as on main. For
A/B runs only; Phase 5 deletes it.

### 1.6 Failure paths, tested in GTA V

| Case | How | What happened |
|---|---|---|
| Network not ready | first frames | frames to DLSS untouched ("the native network is still loading"), one frame in the first run |
| Build fails | `NEURAL_FORGE_FAIL_CREATE=3` | retried after 0.5, 1, 2 s (the helper's schedule, now shared), the 4th attempt closed and reopened the network, built, "up again after 3 failed attempt(s)"; frames untouched meanwhile; no Xid |
| Chain watchdog | `NEURAL_FORGE_NATIVE_FAIL=chain` | the timeout read at the next hold, the graph rebuilt with barriers in 56 ms (`chained false`), the history reset, frames untouched for that time; no Xid |
| Model missing | model directory moved aside for a run | frames untouched all run; status "native network not running: no model at …: extract it in the Setup tab (neural-forge-cli extract-model)"; retried 0.5, 1, 2 s, then every 30 s; directory put back |
| Hash mismatch | not run in game: `load_check` refuses a stage whose digest differs (0.4), and the loader logs and retries like a missing model | |

The Status tab's pre-upscaler row now ends with the layer's status line (`layer_reason`): "native
network running", or "native network not running: <why>". `shmctl status` prints it.

### First numbers (not the Phase 3 measurement)

GTA V Enhanced benchmark, 1440p DLSS Balanced, script mods off, pass 4:

| Run | Real fps | GPU | Power | Notes |
|---|---|---|---|---|
| helper (`hlp-cap-1`) | 68.3 | 93% | 185 W | same day, same settings |
| native (`nat-2`) | 70.4 | 96% | 218 W | |
| native (`nat-cap-1`) | 70.2 | 96% | 218 W | a build ran on the CPU during passes 2-3 |
| native, after the fallback to barriers, frame generation 4x (`fail-1`) | 49.2 real, 197 shown | 96% | 215 W | |
| effect off (no model, `nomodel-1`), frame generation 4x | 74.6 real, 301 shown | 95% | 200 W | |
| native, chained, frame generation 4x (`nat-3`) | 50.2 real, 201 shown | 98% | 218 W | network frame 6.5 ms on the GPU |

The 2 fps between helper and native is inside the run-to-run spread (about 3 fps): not a result
yet. Frame generation engaged only in `fail-1`; the others ran at 1.0x, which the benchmark does in
about half its launches.

### Open items from Phase 1

- **The same moment, native and helper**: captures at fixed seconds after launch were seconds apart
  (the loading time varies), so in-game pictures could not be diffed; both look alike. Phase 3 needs
  captures keyed to the game's frame count.
- **The temporal path is not compared with NGX's.** NGX reprojects with the helper's optical flow;
  the native path with DLSS's motion vectors and the jitter. Watch for ghosting or shimmer in Phase 3
  and in Alex's play.
- **One frame of the layer's work in flight**: each hold waits (bounded) for the previous hold's
  write-back fence before recording, as the helper hold does. With the CPU no longer waiting on the
  GPU inside the hold, this is now the one place the game's CPU can wait for its own GPU work.
  Double-buffering `C`/`W` would remove it; measure in Phase 3 first.
- **Inside DLSS's command buffer** (Crimson Desert, Cyberpunk 2077's `inline` hold): ported after Phase 4
  ("Phase 4b").

## Phase 2: the rest of the app on the native path

### 2.1 After the upscaler

Games without DLSS Super Resolution keep the after-the-upscaler path: the 8-bit swapchain capture into slot
0/1's proxy region and the composition from the answer region, both unchanged. What answers the requests is
new: `preupscale::native_post::PostServer`, a thread in the game's process that does what the helper's
request loop did (`process_request`), minus NGX. Per slot, a new `seq_req` with an 8-bit frame the model is
wanted for runs through `native_post_preprocess.comp` (the 16 lanes from the RGBA8/BGRA8 proxy), the network
(its graph recorded a second time, for the layer's compute family) and `native_post_composite.comp` (the
residual, style operator and intensity, 8 bits out in the proxy's channel order), on a second compute queue
of the layer's own, waited for (bounded). Anything else is echoed, as the helper fails open. Then
`answered_w/h`, `seq_eval`, the helper's stage times, `helper_busy_us` and `seq_resp`, in the helper's order;
the heartbeat and `helper_state` keep the layer's `helper_alive` true. The regions are imported as host
memory where the device allows (staged copies otherwise).

- **No history**: there are no motion vectors after the upscaler, so every frame is a first frame, with a
  fixed seed (a changing seed with nothing blending frames would shimmer). The helper's NGX had a history
  fed by its optical flow. **For Alex**: worth a look for shimmer in a game on this path.
- **Never both paths at once**: the pre-upscaler hold and this server share the network's frame state; each
  refuses for a second after the other used it (`Loader::claim_pre`, `claim_post`).
- **A helper process that is running wins**: the server does not start if the helper's heartbeat moves.
- The loader is now created on every NVIDIA device with the network's additions, not only on devices with
  DLSS's tracking: a game without DLSS has no `VK_NVX_image_view_handle`.

Tested with the `pan` reproducer (2492x1370 window, no DLSS, helper stopped): the server answered every
request, the network built for 2492x1370 in 1.3 s, 67-68 fps with every frame composited, the composited
picture the original with the model's edit (mean 3.7/255, no channel swap). The same run through the
helper: 75-79 fps. The answer takes 15.7 ms native against 11.1 ms through the helper (NGX's evaluate 10.8
ms): **NVIDIA's network is about 20% faster than OpenDLSS-NR's at the same size on this GPU** (also at
1485x836: about 4.8 ms in game against 5.5 ms idle), presumably its kernels built for Blackwell against
upstream's sm_89 PTX that the driver compiles here. On the pre-upscaler path the native backend still comes
out even or ahead because the hold's round trip is gone; after the upscaler, where the helper already ran
off the game's queue, it costs about 12% of the frame rate in `pan`.

### 2.2 The Model tab on the network

What each control does with NGX (the helper) and natively. "Checked" is the same frame (GTA frame d1, the
first after a history reset) through the helper with the setting changed, against dlss5vk with the same
setting: bit-exact, or within one half-float step where the native side applies a post-network operator
(the 0.0005 maximum is the numpy port used for the check, not the shader).

| Control | With NGX (helper) | Native | Checked |
|---|---|---|---|
| Style (Default, Natural, Cinematic) | `DLSSNR.Style` | lane 10 (`style / 128`) **and** the Natural/Cinematic operator after the network (exposure, contrast, saturation in HSL, scaled by local tone; OpenDLSS-NR's `nrStyle`), in `native_composite.comp` | style 1 and 2: NGX = network + operator, within one half step; the network alone differs (max 0.12) |
| Intensity | `DLSSNR.Intensity` | `proxy + intensity * (styled - proxy)`, truncated to half, in the composite | 0.5: within one half step; the network alone differs (max 0.13) |
| Local tone | `DLSSNR.LocalToneStrength` | lane 11 (and the style operator's strength) | 0.5: bit-exact |
| Local structure | `DLSSNR.LocalStructureStrength` | lanes 12-14 | 0.5 with skin 2: bit-exact |
| Skin structure (-1 follows structure) | `DLSSNR.SkinStructureStrength` | lane 13 | as above |
| Auto mask | `DLSSNR.UseAutoMask` | lanes 12-14 (off: structure, -1, -1) | off: bit-exact |
| Preset | `DLSSNR.Hint.Render.Preset` | none | preset 1 gives NGX's preset-0 answer byte for byte: it does nothing on 310.8.0, with either backend. **Removed (SHM v12); the helper always sends 0.** |
| Sharpness | `Sharpness`, which the 310.8 feature never reads (DLSSNR_PARAMETERS.md) | none | No effect on either backend. **Removed (SHM v12).** |
| Passes, per-pass settings | the helper chains one NGX feature per pass | none: the network runs once per frame with the Model tab's values; per-pass overrides are ignored | **Alex (2026-10-07): not wanted on native.** The GUI shows Passes, Per-pass settings and Unlock pass limit only while a game runs on the helper (`native_running`, SHM v13). |
| Motion: estimate motion vectors, units, quality | the helper's optical flow feeds NGX's `MVec` | none: DLSS's own motion vectors and jitter | **Alex (2026-10-07): helper only.** The Motion tab is shown only while a game runs on the helper. |

Every other Model and Composition control (model interval, working scale, detail strength, colour
strength, highlight guard, white point, transfer mode, compare, debug) belongs to the after-the-upscaler
composition, which is unchanged, and has no network-side meaning.

The native hold reads the Model tab's settings from the header every frame (`ShmClient::global_tuning`, no per-pass overrides),
with the helper's clamps, so a change applies at the next frame (the helper rebuilt its feature after a
settle delay instead).

### 2.3 The Setup tab

The NGX import group, the runner picker and the Status page's "NGX binaries" row are gone, and with them
`crates/gui/src/binaries.rs`. In their place one group, "Neural rendering model": a row saying "present,
from build 310.8.0.0" or "missing", and "Extract…", which asks for `nvngx_dlssnr.dll` and runs the same
extractor as `neural-forge-cli extract-model` on a worker thread (a wrong build is refused with the build it
found). The first-run banner now fires on a missing model. The Status page shows the model's state. The
helper's runner stays in `config.ini` (and `neural-forge-cli`) for the A/B runs on this branch.

### 2.4 The 32-bit layer

**The native path cannot run there as built.** A 32-bit process on driver 615.71.09 gets
`VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_EXT_shader_float8` and
`VK_KHR_buffer_device_address`, but **not `VK_NV_cuda_kernel_launch`** (checked with a 32-bit extension
query on the RTX 5070), so none of the PTX kernels can launch. What it would take:

- the GLSL route only (every `DLSS5VK_PTX_*` family off, no counter chaining): 7.56 ms at 1485x836 and
  19.7 ms at 2560x1440 on the idle GPU (+37% and +48% against PTX). The fully unfused reference route is
  77 ms and 220 ms: not usable;
- an i686 build of the vendored C++ (the `cc` target, a 32-bit static libstdc++) and a requirement list
  without `VK_NV_cuda_kernel_launch`;
- a smaller staging window (256 MiB mapped in a 32-bit address space is too much) and the model's host
  copy dropped after upload (141 MiB).

Not built, as the handoff asks.

## Phase 3: measurements (cut short)

GTA V Enhanced benchmark, pass 4 (116 s), script mods off, real fps from GTA's frame times. Alex stopped the
matrix as too long once the answer was clear; what was measured:

| Configuration | Driver | Native | Helper |
|---|---|---|---|
| 2560x1440, DLSS Balanced, frame generation off | 615.71.09 | 70.7, 71.5, 71.6 (GPU 96%, 218 W) | 65.4, 69.9, 69.3 (GPU 88-95%, 185 W) |
| same | 615.78.08 | 70.9, 71.8 (one launch exited early) | 69.8, 69.8, 69.5 |
| 2560x1440, DLSS Balanced, frame generation 4x (engaged) | 615.78.08 | 49.7 real, 199.5 shown | 50.1 real, 200.4 shown |
| same settings, frame generation did not engage | 615.78.08 | 70.2, 70.5 | 68.6, 68.3 |
| 3840x2160, DLSS Balanced, frame generation off | 615.78.08 | 21.9 (GPU 99% at only 157 W) | not measured |

- **1440p**: native 71.3 against helper 68.2 on the old driver, 71.4 against 69.7 on the new. About 2-3 fps,
  at the edge of the run-to-run spread (about 3 fps): equal or slightly ahead, not a demonstrated gain.
  With frame generation 4x they are level (49.7 against 50.1, one run each).
- **4K: native is probably worse.** One run, 21.9 fps. The network frame took 18.8-19.5 ms on the GPU in game
  (10.7 ms on the idle GPU; 6.5 against 5.5 at 1440p), and the GPU ran at 99% busy but only 157 W: it is
  waiting on memory, most likely VRAM (12 GB) with 4K, ray tracing and the network's 1.5 GB at this size.
  The network was also built for 3840x2160 (3.4 GB) on the menus before DLSS started (the after-the-upscaler
  path) and rebuilt smaller after. NGX's evaluate at 4K Balanced was 10.7-11 ms through the helper
  (ARCHITECTURE.md 4.8). A helper run at 4K was not taken.
- Not done: the 30-minute session, chaining off in game, the matched native/helper/off captures (the
  frame-keyed trigger works now: `NEURAL_FORGE_CAPTURE_AT=5:20000` captured a 4K frame), more frame
  generation launches. No Xid, no watchdog timeout, no hang in any native run of the day (about 15 runs),
  including the injected failures.

## Phase 4: report

### What works

- The network runs inside the layer, on the game's device, in the game's DLSS submit: no helper, no Wine, no
  shared-memory round trip, no NGX DLL at run time, no spoof on the native path. Bit-exact with NGX's own
  answer for the same input frame (0.6), also through the layer's C API and on driver 615.78.08.
- Every Model-tab setting that does something with NGX does the same natively (2.2), checked against NGX.
- The after-the-upscaler path runs on it too (2.1), the Setup tab is one extract step (2.3).
- Failure paths: a missing model, failed builds, a chain timeout: frames go to DLSS untouched and the Status
  tab says why; nothing hung or crashed once the loading moved off the game's graphics queue family.

### What did not carry over, or is worse

- **4K is probably slower** (above). Likely VRAM; needs a look before 3.0 if 4K matters.
- **After the upscaler: about 12% fewer fps** in `pan` (67 against 77): NVIDIA's network is about 20%
  faster than OpenDLSS-NR's on this GPU at the same size, and the helper already ran off the game's queue there.
  And that path has **no history** (no motion vectors), where NGX had the helper's optical flow.
- **The temporal path differs from NGX's**: DLSS's motion vectors plus the jitter instead of optical flow. Not
  compared in game; whether DLSS's motion vectors are read one frame stale is open (inconclusive on a still
  scene).
- **Games held inside DLSS's command buffer** (Crimson Desert, Cyberpunk 2077): native since Phase 4b.
- **32-bit**: not possible as built (no `VK_NV_cuda_kernel_launch` in a 32-bit process).
- **Passes, the motion controls**: helper only, hidden in the GUI while native runs (SHM v13). Preset and
  sharpness did nothing on either backend and are removed (SHM v12).
- The layer grew to 8.9 MB and needs glibc 2.38.

### Recommendation

Not 3.0 yet. At 1440p, the configuration Alex plays, native is at least as fast as the helper with a
bit-exact picture and a much simpler setup, which is what the plan asked for. But 4K looks worse, the
after-the-upscaler path is slower and loses its history, and two game families still need the helper, so
removing the helper (Phase 5) now would make those cases worse. Suggested next steps, smallest first: one
4K helper run to confirm the 4K gap; if VRAM is the cause, don't build the network at output size on the
menus and drop the model's host and raw copies after upload; then decide whether the inline hold and the
post path are worth porting or whether 3.0 keeps the helper for them. Alex should also play GTA V on this
branch once to judge the temporal look (ghosting, shimmer) against the helper.

**Alex's answer (2026-10-07):** 4K and the after-the-upscaler path don't matter. He plays at 1440p, 288 Hz,
VRR, HDR, and the games he plays now and later have an upscaler. Passes and the motion controls are not
wanted on native (2.2). The games held inside DLSS's command buffer (Crimson Desert, Cyberpunk 2077) still
need the helper, so whether Phase 5 removes it is still his call. Then: "port the mode now, all games
should work on the native backend when everything is said and done" (Phase 4b).

## Phase 4b: the hold inside DLSS's command buffer

Games that render their frame in DLSS's own command buffer (Crimson Desert, Cyberpunk 2077, Black Myth:
Wukong) are held inside that buffer (PRE_UPSCALER_DESIGN.md, "Inside DLSS's command buffer"): the buffer
copies the colour input to a staging image and waits on an event while the layer's worker runs the hold
on the layer's side queue. That hold now runs the native network instead of asking the helper:

- **Queue and graph.** The side queue is in the network's compute family (the loader's), so the frames
  use the graph recorded for that family (`Built::graph_own`, as the after-the-upscaler path does);
  `run_native_hold` picks the graph by the hold's queue family and refuses any other family.
- **Motion vectors.** The buffer also copies DLSS's RG16F motion vectors (from `GENERAL`,
  `TRANSFER_SRC_OPTIMAL`, `SHADER_READ_ONLY_OPTIMAL` or `COLOR_ATTACHMENT_OPTIMAL`, moved to the
  transfer layout and back when needed) into a slot image of their own, shared with the side queue's
  family; the native capture reads that copy. Without it (another format, an unknown layout) the
  network runs with no history.
- **Jitter.** The launch's parameters are read at record time; the job takes the jitter at the submit.
- **Waiting.** The worker waits for the write-back as before, then releases the buffer: the network's
  frame is inside the time the game's GPU waits at the hold.
- No helper, no slot 0: the slot-0 gates are skipped in native mode, as in the split hold.
- **The post path stays off.** The worker needs the device's state, which the after-the-upscaler path
  holds while its frame waits on the GPU, and the GPU is parked at the hold. The first in-game run
  (`cdn-5`): the network was still loading at the first holds, so the device never engaged, the post
  path took over, and every hold after that gave up on the lock (139 every 5 s, 27 fps). Now each job
  keeps the post path off (`preupscale::keep_post_off`) and the loader's pre-upscaler claim fresh before
  taking the lock (`Loader::note_pre`). A line every 5 s counts what the hold did with its jobs.

### Result (Crimson Desert, 2026-10-07)

One run (`cdn-7`, title screen, E, 60 s in game, Ray Reconstruction on, settings as found), driver
615.78.08:

- Held inside DLSS's buffer at 52/s, every real frame (52.0 fps presented), 0 misses, no fence timeout,
  no Xid. Network 5.88 ms on the GPU at 1516x852, hold 7.0 ms (the worker's time, inside the GPU's wait).
- Motion vectors copied (RG16F, in `GENERAL` there).
- **No jitter**: the game runs DLSS Ray Reconstruction (`rr2_*` kernels); the jitter word is only known
  for the 310.x Super Resolution input kernel (`hiluma_engine_input*`, word 3). The history is
  reprojected with the motion vectors alone, so off by the jitter difference (under a pixel). Open:
  find the word in RR's encoder parameters (`NEURAL_FORGE_PROBE_PARAMS`).

### Against the helper (2026-10-07, released 2.0.10 against this branch)

One run per game and backend through the same direct launch, 60 s measured, settings as found. Cyberpunk 2077
is not installed on the rig.

| | Native | Helper (2.0.10) |
|---|---|---|
| Crimson Desert, in game | 52 fps presented, the model on every frame (52 held/s), 0 misses | 85 fps presented, the model on 44.5 of them a second (about half) |
| Crimson Desert, model switched off | 85.2 fps | 85.4 fps |
| Wukong benchmark, results screen | 107 fps presented = 54 real, every one held; frame generation 2x | 164 fps presented = about 82 real, 32 held/s (39%); frame generation 2x |
| Wukong benchmark, first half / second half | 66-76 / 63-81 fps presented | 64-73 / 86-115 fps presented |

- **The helper skips frames there, native does not.** The helper's hold inside DLSS's buffer gives a frame up
  silently whenever slot 0 is still busy (the request, the zero-copy capture or the compose), so the game runs
  faster with the model on only some of its frames. Native runs the model on every frame and the game's GPU
  waits for it inside the frame: about 7 ms a frame at 1516x852 (network 5.9 ms).
- **Frame generation is not affected by the backend.** Crimson Desert generates no frames under Proton with
  either backend or with the model off (one DLSS submit per present, 85 fps either way), although its options
  ask for dynamic multi frame generation (also with the dynamic option off: 51.8 fps native). Wukong's
  frame generation doubles native's frames as it does the helper's; GTA V's 4x does too (Phase 3, about 200
  presented from 50).
- **Fixed on the way:**
  - The after-the-upscaler path kept its claim on the network when the network was not built at its size, so
    the paths fought over the build. In Wukong (`ab-wk-nat`) the network then faulted the game's channel
    (Xid 13, class `cec0`, then Xid 32) at the first hold that ran it; with the claim always released, the
    benchmark ran through (`wk-nat-2`, 29,400 holds, 0 misses, no Xid).
  - The native hold ignored the on/off toggle (F11, the GUI, `apply_model`): it now goes to DLSS untouched when
    the model is off, as the helper's does (checked in Crimson Desert: "skipped (the model is switched off)").
- The pictures were not compared.

## Decided before Phase 1

1. **The extractor in a public repository.** `extract-model` reads NVIDIA's weights out of the DLL
   so they can run outside NGX. It reads only resource data (no code, nothing decrypted, no check
   bypassed), which the "Working with NVIDIA's binaries" rules in AGENTS.md/CLAUDE.md allow as
   written. But running the weights outside NVIDIA's runtime is a different use than those rules
   were written for, and the NGX licence's terms on it are yours to judge. Nothing has been pushed;
   the commit is on the local `native-backend` branch only.
   *Alex said move on to Phase 1 (2026-10-07). Still nothing pushed: pushing is Phase 4.*
2. **The 1440p margin.** The network alone is about 0.7 ms slower than the helper's evaluate at
   1485x836 on an idle GPU. Phase 3 will say whether the removed hand-off costs make up for it in
   game; if they do not, success criterion 2 fails at 1440p and passes at 4K.

## Attempt log

| Date | What | Result |
|---|---|---|
| 2026-10-07 | `apt install -y python3-pefile` on the test machine | Refused: the SSH session is user `alex`, not root (the handoff assumed root). Used a venv in `~/scratch/venv` instead. |
| 2026-10-07 | tmux for long jobs | Not installed on the test machine; `setsid nohup` and `screen` used instead. |
| 2026-10-07 | Record walk with the kind as a u8 and the tensor at +25 | Wrong: every tensor came out three bytes early (each started `00 00 00`, each trailer started with three data bytes). Lengths still matched the layouts, so the length check alone did not catch it. The kind is a u32; the tensor starts at +28. |
| 2026-10-07 | Stage hashes in lower-case hex | `nr::Model` refused them (`stage SHA-256 mismatch`): its `sha256.h` prints upper case and compares strings. Changed to upper case. |
| 2026-10-07 | `hdr_encode.py encode` output sent straight to the helper | Refused: the script crops to 1485 columns; the layer pads to 1486. Added the padding step. |
| 2026-10-07 | Comparison at 1485 columns | 50.7 dB, not bit-exact; at 1486 (NGX's bytes) bit-exact. See 0.6. |
| 2026-10-07 | The network's loading queue in the game's graphics family (queue 5 of family 0) | Xid 32 on GTA's channel and device lost during the load, GTA exited. Moved to a compute-only family's queue with `CONCURRENT` buffers: no Xid since. |
| 2026-10-07 | The model loaded at every device creation | GTA creates a dozen devices at start; each loaded the model. Now loaded on the first hold. |
| 2026-10-07 | The graph's head buffer on binding 3 of the composite, the features' binding in the preprocess | One set per parity cannot bind both; the head moved to binding 9. |
| 2026-10-07 | The native composite's `.spv` not rebuilt after an edit | The first committed layer had the old binding; `check_shaders.py` caught it. Rebuilt. |
| 2026-10-07 | The layer's GPU tests in parallel on the test machine | 10 fail on main and 16 on the branch with `ERROR_OUT_OF_DEVICE_MEMORY` or "failed to create the test's own target image"; the 6 extra pass one at a time (`--test-threads=1`). A machine limit, not a regression. |
| 2026-10-07 | In-game captures at fixed seconds after launch, native and helper | The moments were seconds apart (loading time varies): not comparable pixel by pixel. |
| 2026-10-07 | Driver 615.71.09 -> 615.78.08 (Alex's request, installed with GreenLight's install script, rebooted) between the first and the second Phase 3 pass | Same capabilities (32-bit still has no `VK_NV_cuda_kernel_launch`); the rig test still bit-exact; network time unchanged (1485x836 5.505 ms, 2228x1253 10.672, 2560x1440 13.349, 3840x2160 30.018; chaining off 5.947, 10.939, 13.731, 30.118). Phase 3 runs again in full on 615.78.08; the 615.71.09 pass of 1440p without frame generation is kept as a reference. |
| 2026-10-07 | `NEURAL_FORGE_CAPTURE_AT=5:20000` in the first Phase 3 runs | Nothing captured: the trigger never fired. Resumes are now logged when it is set; the captures move to dedicated runs. |
| 2026-10-07 | Stale motion vectors? (DLSS's buffer synchronizes them before its launch) | Consecutive dumps 393-396: inconclusive, the scene barely moves (median 0.33 px). Open. |

## Checklist

- [x] 0.1 driver capabilities recorded
- [x] 0.2 other repo read, contracts summarized, commit SHA recorded
- [x] 0.3 dlss5vk builds on Ubuntu 26.04
- [x] 0.4 model directory extracted, loads with hash verification
- [x] 0.5 bench numbers on the rig at four sizes
- [x] 0.6 output compared against the helper
- [x] Phase 0 report written, go or no-go stated, stopped for Alex
- [x] 1.1 sources vendored, built from build.rs
- [x] 1.2 device extensions and features added in the layer
- [x] 1.3 game device adopted, queue use documented
- [x] 1.4 native frame path before the upscaler works in GTA V
- [x] 1.5 backend switch in place
- [x] 1.6 failure paths tested (missing model, watchdog, allocation failure)
- [x] 2.1 after-the-upscaler path native
- [x] 2.2 settings mapping table written, unmapped controls listed
- [x] 2.3 Setup tab reduced to extract-model
- [x] 2.4 32-bit layer answer recorded
- [x] Phase 3 benchmarks (cut short on Alex's request; long session not run)
- [x] Phase 4 report written, branch pushed, stopped for Alex
- [ ] Phase 5 (only on Alex's yes)

Blockers: none. Waiting for Alex: Phase 5.
