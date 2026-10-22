# Native model backend (3.0)

The plan: run the neural rendering network inside the layer, on the game's own device, with
[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (MIT), and retire the Windows helper, the
Wine/Proton runner, the shared-memory hand-off, the NGX DLL at run time and the caller-identity spoof.
Branch `native-backend` only; nothing merges, tags or releases until the maintainer has judged the results.

This file is written as the work goes. Every attempt, including the failed ones, is in "Attempt
log" so the next one starts from it.

## Phase 0: go or no-go (2026-10-07)

**Verdict: go, with two cautions.** (the maintainer: go, 2026-10-07.) All three gates hold on the test machine:

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
  Vulkan-Headers `v1.4.365`, volk `vulkan-sdk-1.4.363.0`. CMake 3.31.12 portable (the machine has
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

The imported `nvngx_dlssnr.dll` is **build 310.8.0.0** (`VS_FIXEDFILEINFO`), the build whose output
was compared with NVIDIA's runtime (0.6). Since 3.1.0 the extractor takes any build carrying the same
network (the shape check, below).

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
holds the DLL or is the DLL). It walks the records refusing any surprise in their framing (a total
that disagrees, a kind other than 1, differing record lengths, a tensor overrunning its record, a
name outside the pattern), checks the network's shape (below), and writes `paths::data_dir()/model`:
`manifest.json` plus `model/<stage>.e4m3`, each file through a temporary name and a rename. (Since 3.7.0
that layout is one network's directory, `model/full` or `model/small`: `extract-model` writes every
network the DLL carries, the whole model directory replaced as one, and the `network` setting picks the
one the layer loads; "The small network" below.) Eleven
stages by resolution: `encoder32` (blocks 0-4), `encoder64` (5-8), `encoder128` (9-14),
`encoder256` (15-22), `encoder512` (23-30), `vit` (31-38), `decoder512` (39-47), `decoder256`
(48-55), `decoder128` (56-61), `decoder64` (62-65), `decoder32` (66-70); tensors 16-byte aligned
in each file. The manifest also records the source DLL's build and SHA-256. Stage hashes are
upper-case hex, because `nr::Model` compares its own upper-case digest as a string. The output
directory is ignored in `.gitignore` (`/model/`, `*.e4m3`) in case it is ever written inside the
tree. Extraction takes 0.26 s.

**The shape check (3.1.0).** Until 3.0.0 anything but build 310.8.0 was refused. That gate was the
project's own, not a technical limit, so the extractor now checks the network instead:
`check_shape` compares the records with `crates/supervisor/src/model_shape.rs`, a table of the 153
tensors the graph implements, each name and byte length (147,683,778 bytes in all). The table was
generated by `scripts/gen_model_shape.py` from the `tensors` array (`name`, `byteLength`) of the
manifest extracted from build 310.8.0.0; it holds no tensor bytes and no hashes of them. It describes
the network the graph in `third_party/opendlss-nr` builds, and changes with it. A build whose
records hold the same set of names, each the same length, is extracted whatever its version. Any
other is refused before anything is written, with a report: tensors, blocks and bytes found against
expected; names expected but missing; names present but not expected (a repeated name included);
names with a different length (`name: found N, expected M`), ten of each; and "This DLL carries a
different network. Neural Forge needs new graph code for it." A retrain that keeps the architecture
passes; `blend_scale` is read from the weights (`nf_native.cpp`), so a new value is picked up.
`VERIFIED` (the full network of 310.8.0 and 310.9.16, the small one of 310.9.16) lists the networks and builds whose output was compared bit for bit with
NVIDIA's runtime; the manifest's `source.verified` says whether the source build is one, the CLI,
the Setup tab and the layer's load log line show it, and it gates nothing. A manifest without the
field was written by 3.0.0, from 310.8.0. Fifteen unit tests run the walk, the shape check and the
extractor on synthetic PE images and blobs, with a small table in place of the real one.

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
   that answer is NGX's first frame after a reset, with no history. Helper settings as the maintainer has
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

- OpenDLSS-NR: upstream `src/` (minus `main.cpp` and `verify.cpp`), `shaders/`, `scripts/ptx/`,
  `LICENSE`, `NOTICE`, at `9d08f41`. Vendored in `third_party/opendlss-nr/` until 3.4.0; from 3.4.0 to
  3.8.6 `scripts/fetch-native-tools.sh` fetched the commit (SHA-256 pinned) into `tools/native/OpenDLSS-NR`
  and applied the patches kept in `third_party/opendlss-nr/patches/`. After 3.8.6 the source was taken
  back into `third_party/opendlss-nr/` with the ten patches applied, and is this project's to change:
  upstream (unchanged since 2026-09-21) is no longer tracked, and `UPSTREAM.md` there lists the ten
  changes with their reasons. The first four: 0001 and 0002 the two GCC fixes, 0003 the asset loader (kernels come from
  memory, not a directory), 0004 the adoption inside a layer (next layer's proc addresses, bounded
  waits, no `vkDeviceWaitIdle` on an adopted device, a dispatchable-init hook, `CONCURRENT` buffers
  over two queue families). ATTRIBUTION.md has the row.
- `crates/native` (`neural-forge-native`): `build.rs` compiles the 13 GLSL kernels with the pinned
  glslang and runs the 78 PTX generator invocations of upstream's `build_shaders.ps1` at build time,
  embeds both (`.incbin`), and compiles the network's C++ with the glue (`cpp/nf_native.cpp`) into the
  rlib. Nothing prebuilt is committed. `scripts/fetch-native-tools.sh` fetches glslang 16.6.0,
  Vulkan-Headers v1.4.365 and volk vulkan-sdk-1.4.363.0 (SHA-256 pinned) into `tools/native/`
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
missing requirement. An instance the application created for Vulkan 1.2 or older (its
`VkApplicationInfo::apiVersion`, which the layer keeps from `vkCreateInstance`) is refused the same
way: a device's API version is the lower of its instance's and the GPU's, and below 1.3 neither the
kernels' SPIR-V 1.6 nor the 1.3 core commands the vendored code calls are there (DXVK and
vkd3d-proton create 1.3 instances).

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
  complete before parameters for the same parity are written. Inside DLSS's command buffer (3.8.1) the
  hold is recorded into that buffer instead, the parameters with it (`vkCmdUpdateBuffer`), so buffers
  recorded ahead each carry their own; the network's graph is recorded straight into the buffer with
  descriptor sets from that buffer's slot (`nf_native_record_graph_at`), and the build is not rebuilt
  while a buffer recorded with it may still run (PRE_UPSCALER_DESIGN.md, "The hold inside DLSS's
  command buffer").
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
  writes). Since 3.7.0 also on a scene cut (`preupscale::scene_cut`, `Stale::SceneCut`): the 1.x
  helper's detector (an 8-pixel-step luma thumbnail of the encoded capture, a mean absolute difference
  of 40 of 255 from the previous hold's) on the GPU, two dispatches in `C` after the encode, whether
  motion vectors are on or not. Its decision is the state buffer's third word (`cut`, beside `ok[2]`),
  which both frame shaders read as no history on that same frame, so every pass (`MorePasses` too) starts
  afresh with seed 0. The host reads the 16-byte result at the next hold, after the wait it does anyway:
  a cut on a network frame logs `[native] resetting the history (scene cut: ...)`, is counted (SHM v20
  `scene_cuts`) and restarts the CPU's seed (`restart_seed`) so the next frame's is 1, as after a reset;
  a cut on a cached hold (no network ran on it) marks `HistoryGap::scene_cut`, so the next network frame
  resets (and is logged and counted then), and drops the edit cache's stored edit so that frame
  refreshes. `NEURAL_FORGE_SCENE_CUT=off|<threshold>` is the test override.
- **Motion vectors** (measured on consecutive GTA V frames, `NEURAL_FORGE_PREUPSCALE_DUMP_FRAMES`):
  DLSS's are in render pixels, y down, current to previous (`prev = x + mv`, scale +1 on both axes
  by a 7x7 search). They exclude the camera jitter: the warp error falls a further 20% with a global
  sub-pixel offset that changes every frame. The jitter DLSS is given is word 3 of the 310.x input
  kernel's parameters (x low, y high; found with `NEURAL_FORGE_PROBE_PARAMS=1`): a 23-frame Halton
  (2,3) sequence in ±0.5 px. Three consecutive pairs fit only `prev = x + mv + (jitter[t-1] -
  jitter[t])`, measured (-0.25, 0.31), (0.44, -0.50), (-0.94, 0.38) against (-0.25, 0.33), (0.50,
  -0.56), (-0.91, 0.33). Other DLSS versions' input kernels have no known jitter word: they get
  `jitter = 0` (history off by up to a pixel, which the network's blend weight then has to absorb).
  Motion vectors at the output's extent (Hogwarts Legacy's `mvhi` ones) are in output pixels: the
  hold blits them down to the colour input's extent and both shaders multiply them by render over
  output per axis (`Params.more.yz`; PRE_UPSCALER_DESIGN.md, "High-resolution motion vectors").
- **Seed**: 0 on a frame without history, then counting, so the first frame after a reset is NGX's
  exactly (0.6).
- **Settings**: Phase 1 runs the defaults (style 0, tone 1, structure 1, skin -1, auto-mask, intensity
  1), the values the maintainer runs. The Model tab's mapping is 2.2.
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
| Chain watchdog before a rebuild | rig: `a_chain_timeout_left_from_the_frames_does_not_fail_the_next_builds`, `a_warm_up_chain_timeout_falls_back_to_barriers` | a count the frames left that no hold read before a rebuild for another size, or a warm-up's own, switches that build to barriers (`chained 0`); each warm-up starts from a count of 0, so the count never fails a later build |
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
  and in the maintainer's play.
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

- **History, selectable**: there are no motion vectors after the upscaler. By default
  (`NEURAL_FORGE_POST_MOTION=none`) every frame is a first frame, with a fixed seed (a changing seed with
  nothing blending frames would shimmer). `zero` and `flow` give the path a history (zero motion, or the
  GPU's optical flow, as the helper's NGX had): "History after the upscaler" below.
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

Measured again on 2026-10-22, with the network as it runs now and without the helper in the way (3.8.x, 1486x836,
the rig's back-to-back timing): 5.35 ms for the full network against NGX's 5.1 ms (the helper's busy time, an
upper bound), and 3.15 ms for the small one against 3.6 ms. Generating the PTX for `sm_120` instead of `sm_89`
changes nothing (5.383-5.387 against 5.386-5.408 ms median, bit-exact), so the target was not the cause; where
the time goes is in DLSS_KERNEL_CATALOGUE.md, "Row tiles, the PTX target and where the time goes".

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
| Passes, per-pass settings | the helper chained one NGX feature per pass | the network runs `passes` times per network frame (1 to 10, SHM v19), each pass after the first on the previous pass's answer and with its own parameters (the Model tab's values with that pass's overrides) | Back in 3.6.0 at the maintainer's request (2026-10-19), after 3.0.0 removed them with the helper. Before the upscaler (`native::MorePasses`) the answer is copied over the encoded capture between passes, and the capture put back for the write-back; after it (`native_post`) each pass reads the previous answer from a scratch buffer. Each later pass has its own history (`MorePasses::history`, built when the pass count first reaches it; a pass reads it only if it ran on the previous frame). GTA V Enhanced, 1440p, FG 4x, mods off: small 74.3 / 64.0 / 58.1 real fps at 1 / 2 / 3 passes, full 65.6 / 51.2 at 1 / 2 (HARDWARE_VALIDATION.md, 2026-10-19). |
| Motion: estimate motion vectors, units, quality | the helper's optical flow feeds NGX's `MVec` | none: DLSS's own motion vectors and jitter | **The maintainer (2026-10-07): helper only.** The Motion tab is shown only while a game runs on the helper. |

Every other Model and Composition control (model interval, working scale, detail strength, colour
strength, highlight guard, white point, transfer mode, compare, debug) belongs to the after-the-upscaler
composition, which is unchanged, and has no network-side meaning.

The native hold reads the Model tab's settings and every pass's overrides from the header every frame (`ShmClient::pass_tunings`),
with the helper's clamps, so a change applies at the next frame (the helper rebuilt its feature after a
settle delay instead).

### 2.3 The Setup tab

The NGX import group, the runner picker and the Status page's "NGX binaries" row are gone, and with them
`crates/gui/src/binaries.rs`. In their place one group, "Neural rendering model": a row saying "present,
from build 310.8.0.0" or "missing", and "Extract…", which asks for `nvngx_dlssnr.dll` and runs the same
extractor as `neural-forge-cli extract-model` on a worker thread (since 3.1.0 a different network is refused
with the shape report: its summary line in a toast, all of it on stderr; an unverified build is marked so in the
row). The first-run banner now fires on a missing model. The Status page shows the model's state. The
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

GTA V Enhanced benchmark, pass 4 (116 s), script mods off, real fps from GTA's frame times. The maintainer stopped the
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
  faster than OpenDLSS-NR's on this GPU at the same size, and the helper already ran off the game's queue there. (Measured
  again on 2026-10-22: the full network about 0.25 ms behind NGX at 1486x836, the small one ahead; 2.1.)
  And that path has **no history** (no motion vectors), where NGX had the helper's optical flow.
- **The temporal path differs from NGX's**: DLSS's motion vectors plus the jitter instead of optical flow. Not
  compared in game; whether DLSS's motion vectors are read one frame stale is open (inconclusive on a still
  scene).
- **Games held inside DLSS's command buffer** (Crimson Desert, Cyberpunk 2077): native since Phase 4b.
- **32-bit**: not possible as built (no `VK_NV_cuda_kernel_launch` in a 32-bit process).
- **The motion controls**: helper only, hidden in the GUI while native runs (SHM v13). Preset and
  sharpness did nothing on either backend and are removed (SHM v12). Passes and per-pass settings run
  natively since 3.6.0.
- The layer grew to 8.9 MB and needs glibc 2.38.

### Recommendation

Not 3.0 yet. At 1440p, the configuration the maintainer plays, native is at least as fast as the helper with a
bit-exact picture and a much simpler setup, which is what the plan asked for. But 4K looks worse, the
after-the-upscaler path is slower and loses its history, and two game families still need the helper, so
removing the helper (Phase 5) now would make those cases worse. Suggested next steps, smallest first: one
4K helper run to confirm the 4K gap; if VRAM is the cause, don't build the network at output size on the
menus and drop the model's host and raw copies after upload; then decide whether the inline hold and the
post path are worth porting or whether 3.0 keeps the helper for them. The maintainer should also play GTA V on this
branch once to judge the temporal look (ghosting, shimmer) against the helper.

**The maintainer's answer (2026-10-07):** 4K and the after-the-upscaler path don't matter. He plays at 1440p, 288 Hz,
VRR, HDR, and the games he plays now and later have an upscaler. Passes and the motion controls are not
wanted on native (2.2). The games held inside DLSS's command buffer (Crimson Desert, Cyberpunk 2077) still
need the helper, so whether Phase 5 removes it is still his call. Then: "port the mode now, all games
should work on the native backend when everything is said and done" (Phase 4b).

## Phase 4b: the hold inside DLSS's command buffer

> Since 3.8.1 the hold inside DLSS's buffer no longer waits on an event the host sets, and no longer
> runs on a queue of the layer's: the whole hold, the network's graph included, is recorded into DLSS's
> buffer as it is recorded, on the game's graphics family (PRE_UPSCALER_DESIGN.md, "The hold inside
> DLSS's command buffer"). What follows is the 3.0 design it replaced, kept for its measurements.

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
  reprojected with the motion vectors alone, so off by the jitter difference (under a pixel). Found
  later (below). Since 3.1.4 Ray Reconstruction's input is not held at all (PRE_UPSCALER_DESIGN.md,
  "DLSS Ray Reconstruction"): with it on, Crimson Desert gets the model after the upscaler.

### Against the helper (2026-10-07, released 2.0.10 against this branch)

One run per game and backend through the same direct launch, 60 s measured, settings as found (Cyberpunk 2077
is out of scope).

| | Native | Helper (2.0.10) |
|---|---|---|
| Crimson Desert in game (Nas River, `ig-cd-*`) | 148.5 fps presented, 24.7 held/s, 0 misses | 146.3 fps presented, 24.3 held/s, 0 misses |
| Wukong benchmark, results screen | 107 fps presented = 54 real, every one held; frame generation 2x | 164 fps presented = about 82 real, 32 held/s (39%); frame generation 2x |
| Wukong benchmark, first half / second half | 66-76 / 63-81 fps presented | 64-73 / 86-115 fps presented |

- **Crimson Desert: parity.** Frame generation works with both (about 6 presented frames per held one). The
  pictures (captures 20 s apart at the same spot) match in tone, contrast and detail; the mean colour differs
  by 1-3 levels in 255 with the camera drift between captures; no ghosting on the falcon's wings or the
  character. An earlier set of runs (`ab-cd-*`, `cd-*-off*`, `cd-nat-fixed`) never left the title screen
  (the E key was not taken) and showed no frame generation there with either backend: not a driver problem.
- **Wukong: the helper skips frames there, native does not.** The helper's hold inside DLSS's buffer gives a
  frame up silently whenever slot 0 is still busy, so the game runs faster with the model on only some of its
  frames. Native runs the model on every frame and the game's GPU waits for it inside the frame.
- **Jitter.** Found with `NEURAL_FORGE_PROBE_KERNEL` (a dump of a named kernel's parameters): Wukong's
  `cuda_engine_input_kernel*` keeps it in bytes 80-87, Crimson Desert's Ray Reconstruction encoder
  `rr2_enc0_kernel` in bytes 400-407 (the previous frame's in 392-399); both a Halton(2,3) sequence minus 0.5 in
  render pixels. Both holds then ran with it. Since 3.1.4 Ray Reconstruction is never held, so the
  `rr2_enc0_kernel` word is no longer read.
  With Ray Reconstruction off, Crimson Desert's Super Resolution (`rrlite_*`) keeps it in
  `rrlite_enc0_*`'s bytes 168-175 (the previous frame's in 176-183), the same encoding (3000
  launches probed, 2026-10-09).
- **Fixed on the way:**
  - The after-the-upscaler path kept its claim on the network when the network was not built at its size,
    so the paths fought over the build. The claim is now always released.
  - `keep_post_off` shared the post path's "off until" with the session, which rewrote it at every present:
    the post path still started before the first holds. It has its own slot now.
  - The native hold ignored the on/off toggle (F11, the GUI, `apply_model`): it now goes to DLSS untouched
    when the model is off, as the helper's does (checked in Crimson Desert).
- **Open: an intermittent GPU fault in Wukong.** Xid 13 (`Subchannel Mismatch`, class `cec0`, a compute
  channel of the game) then Xid 32, in the first frames the hold inside DLSS's buffer runs the network (a frame
  is submitted and never completes; timestamped with `NEURAL_FORGE_LOG_TIME`). Never in Crimson Desert, never
  with the helper. Short runs (`ab.sh wks`, two minutes each):

  | Variant (counter chaining kept, `NEURAL_FORGE_NATIVE_INLINE_CHAIN=1`, to make it frequent) | Faults |
  |---|---|
  | as is | 5 of 8 |
  | frame generation off in the game | 0 of 4 |
  | the network's launches skipped (`NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`) | 0 of 6 |
  | the network waits for Frame Generation's compute submits made before it (timeline) | 4 of 6 |
  | ... and those submits wait while the network runs (mutual exclusion) | 3 of 6 |
  | ... for every submit outside the graphics family | 3 of 6 |
  | the network's input cleared of NaN and clamped | 2 of 6 |

  With barriers between the network's launches (the default now) 1 of 5 full runs faulted, and 4 later ones
  did not. Frame generation on also makes DLSS's input alternate every frame between an RGBA16F and an
  R11G11B10 image (only R11G11B10 with it off). Not overlap with Frame Generation's GPU work (the exclusion
  rows), not a queue collision (the game only takes its own queues: `[queues]` lines), not bad input. The
  ordering code is not kept; the diagnostics are. Next: what in the network's own work differs on the
  alternating frames, and whether it reproduces on the previous driver (615.71.09).
- **Wukong's motion vectors are not copied** (the image is not readable for the copy at the hold, although
  `TRANSFER_SRC` is now added to RG16F images when the native backend is on), so the network runs there
  without history. The history no longer resets when the colour input alternates (keyed on the extent only).

## Phase 5: the helper removed (3.0.0, 2026-10-07)

The maintainer (2026-10-07), after the game comparison above: "push for the 3.0.0 release, where you remove the
helper and everything associated with it". Done on `release-3.0`:

- **Gone:** `crates/helper` and its scripts (`helper-test.sh`, `rig-test.sh`), the supervisor's
  process, runner, provisioning, install-dir and GPU-detection modules, the CLI's `setup`, `start`,
  `stop`, `restart`, `runners` and `detect-gpu`, the GUI's helper start/stop, the Windows target in the
  AppImage build and CI, `NEURAL_FORGE_BACKEND` (native is the only model mode), the helper hold's
  round trip, circuit breaker and slot-0 gates in `preupscale.rs`, and the examples that drove the
  helper.
- **Gone with it:** the 32-bit layer (2.4: without the helper it only passed frames through), its
  manifest, build and CI test step.
- **Settings:** passes, per-pass settings, the pass limit, rebuild spacing and the motion controls
  (2.2) are removed, not hidden. The channel is SHM v15: those fields, the helper's VRAM, feature
  counts, reason string and the DMA-BUF exchange removed; what the in-process model server writes
  renamed `server_*` (`server_state`, `server_heartbeat`, `server_busy_us`, ...).
- **Kept:** the shared-memory channel (settings, status, and the after-the-upscaler request slots,
  answered by `native_post::PostServer`); the extractor; `binaries=` in `config.ini` (where an
  imported DLL lives). `config.ini` keeps only `binaries=`, `shm=` and the `set_` settings, and the
  layer applies the saved settings itself when it creates the channel. The maintainer's machines get their old
  keys removed by hand; there is no migration code.
- **Smoke test of the 3.0.0 AppImage** (installed on the test machine, driver 615.78.08, one run each):
  GTA V benchmark pass 4 69.1 fps, every frame held (native before: 70.9-71.8; spread about 3 fps);
  Crimson Desert in game 151.6 fps presented, 25.9 held/s inside DLSS's buffer, 0 misses (the first run
  after the update compiled the game's shaders and stayed on its title screen: 50.5 held/s, 0 misses);
  Lords of the Fallen (Unreal Engine 5, first launch, its privacy-policy screen: not accepted for the maintainer)
  held inside DLSS's buffer at 109.4/s, R11G11B10 input at 1488x836, network 6.19 ms, 0 misses. No Xid,
  no fence timeout in any run.
- **Checks:** every crate builds without warnings, `cargo test` (layer on lavapipe) passes, as do
  `test_install.py`, `check_shaders.py` and `check_namespace.py`.

## The small network (3.5.0, 2026-10-16)

Build 310.9.16.0 of `nvngx_dlssnr.dll` (266 MB, from a game update the maintainer received) carries a second
weights resource next to `WEIGHTS_HT`: `RT_RCDATA "WEIGHTS_CC_SMALL_FP8"`, 113,893,343 bytes, in the same record
framing (0.4). Its `WEIGHTS_HT` is byte-identical to 310.8.0's (the eleven stage files have the same SHA-256).
As in 0.4, only the PE headers, the resource directory and the records were read; no code in the DLL.

**What it is, from the records alone.** 77 tensors, blocks 0-30, 113,887,490 bytes, `block30.layer0.blend_scale`
last but one. Every block has exactly the names and byte lengths of a block type of the full network, in the
same order of types, with fewer repeats per stage:

| Stage | Full (blocks) | Small (blocks) |
|---|---|---|
| pre (block 0), encoder 32 | 0, 1-4 | 0, 1-2 |
| encoder 64 / 128 / 256 | 5-8 / 9-14 / 15-22 | 3-4 / 5-6 / 7-8 |
| encoder 512 (split blocks) | 23-30 | 9-10 |
| ViT | 31-38 | 11-18 |
| decoder 512 (transition, split blocks) | 39, 40-47 | 19, 20-21 |
| decoder 256 / 128 / 64 / 32 | 48-55 / 56-61 / 62-65 / 66-69 | 22-23 / 24-25 / 26-27 / 28-29 |
| post | 70 | 30 |

The first and last block of each stage carry the same transition weights as the full network's (block 2 is
block 4's 22,720 bytes, block 10 has block 30's ViT projection, block 19 is block 39's 525,312, and so on).
`model_shape::EXPECTED_SMALL` is the table; a unit test checks that every small block has a full block's shape.

**The graph.** `patches/0009` (UPSTREAM.md, 9) takes the block ranges from `nr::Topology`, chosen by the
model's block count. Nothing else changes: the same kernels, layouts and chaining. The window phase advances
per block at each level as in the full network, so the small network's encoder stages use phases 0-1 and its
decoder stages continue from 2. That rule is upstream's, found for the full network; for the small one it is
the graph's assumption, checked only by the picture below. The extractor (`model.rs`) takes `Network::Full`
or `Network::Small` (`extract-model DIR --network small`, the Setup tab's Network row) and records it in the
manifest (`network`); `nf_native.cpp` reads `blend_scale` from the last block.

**Both networks, chosen live (3.7.0).** `extract-model` (and the Setup tab's Extract button) writes every
network the DLL carries into its own directory, `model/full` and `model/small` (`model::extract_all`;
`--network` writes one and leaves the other). The `network` setting (SHM v20, `enums::network`, the
Setup tab's Network row, persisted like the other settings) chooses which one the layer loads, the full
one by default; when the chosen one is not extracted the other is loaded and the log says so. A change
reaches the running game at its next hold: `Loader::ready` hands out no build of another network, the
loader thread closes the open network and opens the chosen one (drained first, as for a
re-initialisation), and the first frame of the new network starts the history afresh (`note_network`).

**Checks** (`dlss5vk` built from the pinned commit with the same change, the 0.6 fixtures, RTX 5070):

- The full network through the changed graph: bit-exact against NGX's answer on frame d1w (as in 0.6).
- The small network on frames d1w, d4w-d8w (1486x836): no non-finite value, the same answer over three
  production submissions, and the same answer with barriers instead of chaining, on every frame.
- Against the full network on d1w (both composed as in 0.6): PSNR 36.19 dB, mean difference 0.0109 on 0..1.
  Against the input, the full network's answer is at 30.29 dB and the small one's at 32.97 dB: the small
  network changes the frame less. Side by side, the small network's picture is coherent and follows the full
  one's (no blocking, seams or colour shift).
- GPU time (`dlss5vk bench`, 200 frames at GTA's DLSS input 1707x960, median): full 7.03 ms over 249
  dispatches, small 3.81 ms over 125.
- Against NVIDIA's runtime: see "Against NGX (3.5.2)" below.

**In GTA V Enhanced** (3.5.0 on the test machine, `gta-bench.sh`, 1440p, the maintainer's settings with
frame generation 4x, one run each, pass 4):

| Model | Real fps | Shown fps | Network (`server`, median) | Weights on the GPU |
|---|---|---|---|---|
| Full network, 3.4.1 (same day) | 58.6 | 234.9 | 16.1 ms | 140 MiB |
| Full network, 3.5.0 | 56.8 | 228.1 | 16.0 ms | 140 MiB |
| Small network, 3.5.0 | 63.7 | 256.2 | 9.95 ms | 108 MiB |

The full network's time is unchanged by the graph change; the earlier 3.4.x runs on 2026-10-13 gave 56.3 and
57.6 real fps. The small network held 64.1 frames/s before the upscaler with no misses logged.

### Against NGX (3.5.2, 2026-10-17)

The released 2.0.10 helper, NGX with build 310.9.16 of the DLL, in its own XDG directories and channel, no
game running, driven by `trigger_helper_roundtrip` as in 0.6 (frame d1w: 32 requests, 2 s, one more, so
the answer is NGX's first frame after a reset). Only the parameter object the helper hands NGX was
watched; no code in the DLL was read. The DLL reads no new key that selects a network; three keys it
reads and the helper never sets are new against 310.8 (`DLSSNR.MaskBlendMode`, `DLSSNR.UICorrection`,
`DLSS.Indicator.Invert.X/Y.Axis`). `DLSSNR.Hint.Render.Preset` picks the network:

| Preset | First answer after a reset | `dlss5vk parity`, small network | `dlss5vk parity`, full network | Helper busy (1486x836, median of 40) |
|---|---|---|---|---|
| 0 (the default; 2 and 4 give the same first frame) | one answer | **bit-exact (3726888)** | 99.3% differ | 3.6 ms |
| 1 | 310.8.0's full answer, byte for byte | 99.3% differ | **bit-exact (3726888)** | 5.1 ms |

So NGX runs the small network at its default preset from build 310.9.16, and the native small network
computes it identically (the window phases of patch 0009 included). 3.5.2 marks the small and full
networks of 310.9.16 verified and makes the small one the default extraction where the DLL has it.
Presets 2 and 4 give different answers from 0 over a run with history; not followed further.

GTA V Enhanced on the helper with that model (2.0.10 installed, preset 0, `gta-bench.sh`, the maintainer's
settings with frame generation 4x and FG Preset C, one run, pass 4): 63.5 real fps, the model held
before the upscaler at 1486x836, 61 holds/s, 0 misses, helper busy 4.2 ms.

## The control mask (3.7.0, 2026-10-19)

A per-pixel control of the network's edit before the upscaler, after NVIDIA's `DLSSNR.ControlMask`. The
native backend reimplements the network, so it implements the mask's semantics directly instead of an NGX
parameter. The semantics follow GrimsVerk's static analysis of both runtimes and their working ControlMask
implementation (https://grimsverk.github.io/nr-runtime-research/static.html and controlmask.html; ideas
only, no code):

| Channel | Acts on | Here |
|---|---|---|
| r | the final blend: `out = orig + intensity x r x (NR - orig)` | `native_composite.comp` (NGX saturates `intensity x r`; here the product is used as it is, so an intensity above 1 still extrapolates and an all-ones mask is exactly no mask) |
| g | network input channel 11 = local tone x g (the Style grade is not scaled) | `native_preprocess.comp` |
| b | auto mask off: channel 12 = structure x b; on: NVIDIA's skin and structure channels (13, 14) take min(channel, structure x b), ControlMask's blend mode 2, the only one here, so NVIDIA's skin handling is kept and the mask only lowers structure | `native_preprocess.comp` (a b of 1 leaves every channel as it was) |
| a | unused | |

`native_mask.comp` builds it every network frame at the model's working size (one RGBA8 texel per pixel of
DLSS's input), in pass 1's `N` before the preprocess, from a neutral 1, 1, 1 by these rules in order, each
only lowering a value:

- **Keep the game's sky** (`mask_sky`, on by default): r = 0 where the depth is within `mask_sky_margin`
  (default 0.00001, the sky only; 0.001 takes in distant scenery) of the far plane, back to 1 at twice the
  margin, averaged over the 5x5 depth texels around the pixel so silhouettes do not flicker. It reads the edit
  cache's depth copy, in the convention the cache decides from its first holds (`infer_reverse_z`; GTA V
  Enhanced is reversed-Z, the cache's verdict), so it is off until then and without the edit cache
  (`NEURAL_FORGE_EDIT_CACHE=off` copies no depth). After the upscaler there is no depth: the mask is before
  the upscaler only.
- **Dark areas** (`mask_dark`, off): below the brightness `mask_dark_threshold` (luma of the encoded capture,
  0-1), r and b fade toward `mask_dark_intensity` and `mask_dark_structure`, fully at black.
- **Fast motion** (`mask_motion`, off): above `mask_motion_threshold` render pixels per frame (the motion the
  network's history reads), r and b fade toward `mask_motion_intensity` and `mask_motion_structure`, fully at
  twice the threshold.

With no rule on, the mask pass writes nothing and no shader reads it: the frame is bit for bit as without the
mask (`the_control_mask_keeps_the_sky_lowers_structure_and_changes_nothing_when_all_ones` also checks an
all-ones mask against none). Every pass of a multi-pass frame reads the same mask, built from the original
frame, unless the passes run at different model resolutions: then the first pass at each resolution builds
that resolution's mask from its own input ("Model resolution before the upscaler").

## Pass feedback (3.7.0, 2026-10-19)

`pass_feedback` (0 to 1, default 1) sets the next pass's colour input to the previous pass's input moved
toward its answer: input + feedback x (answer - input), the idea of janblade/OptiScaler-F5-DLSSNR-Multipass's
`DlssNr.PassFeedback` (v0.1.18 notes, ideas only). At 1 the copy between passes stays as it was (bit for
bit); below 1 `native_feedback.comp` blends in place in the encoded image before the upscaler (`X`, `Y`), and
`native_post_feedback.comp` into the scratch after it (from the frame for pass 2, in place after).

GTA V Enhanced, 1440p, FG 4x, the small network, the still shot after its cut (one run each; flicker and
added detail with bench-iq.py's functions on a window placed 40 presented frames after each series' own cut,
against the effect off at 0.931%):

| Passes | Feedback 1.0 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|
| 2 | 0.78%, x2.05 | 0.77%, x1.92 | 0.80%, x1.87 | 1.50%, x1.37 |
| 3 | 0.92%, x2.51 | 1.01%, x2.12 | 0.89%, x1.97 | 0.67%, x1.90 |

(One pass: 1.27%, x1.32.) Lower feedback lowers the added detail steadily (the over-processing of extra
passes); flicker does not move beyond the run-to-run noise. No value is clearly better, so 1 stays the
default.

## History after the upscaler (selectable, 2026-10-19)

Without a history every frame after the upscaler is a first frame, which costs temporal stability
(shimmer). `NEURAL_FORGE_POST_MOTION` selects one, for measurement (read once, logged once). `none` stays the
default: measured, neither history is steadier than none ("Measured" below), so it is a testing aid:

| Value | History |
|---|---|
| `none` (default) | None: lanes 7-9 copy the proxy, seed 0, no blend. Bit for bit as 3.7.0 (`the_shipped_frame_stays_bit_for_bit` pins the answers and lanes from 3.7.0's shaders). |
| `zero` | Each pass's previous output at the same pixel. |
| `flow` | Each pass's previous output where the GPU's optical flow (`VK_NV_optical_flow`) says the pixel was, read with the pre-upscaler preprocess's five-tap Catmull-Rom filter. `zero` where the device has no optical flow (logged once, with the reason). |

With a history the after-the-upscaler shaders do what the pre-upscaler ones do with theirs:
`native_post_preprocess.comp` puts the centred previous output into lanes 7-9 and the noise seed counts frames
since the last reset; `native_post_composite.comp` blends toward it by sigmoid(head logit) x blend scale and
keeps the result (before the style and intensity, as half floats) as the pass's next history. Every pass of a
multi-pass frame has its own history buffer; a pass that did not run on the previous frame (the Passes setting
grew) starts afresh. What gets no history:

- **Per pixel**: where the frame changed there by more than `NEURAL_FORGE_POST_HISTORY_REJECT` (0 to 1,
  default 0.12) since the previous frame (the largest channel difference of the 8-bit proxies, in the
  encoded 0-1 scale; with the flow against the previous frame where the pixel was, bilinear), or where the
  previous position is off screen. The idea is uzbekunknown/dlss-nr-on-intel's (Apache-2.0,
  `notes/phase72-preset-b.md` "Flicker", the daemon's `History` with `STALE = 0.12`; ideas only).
- **The whole frame** (`post_history::SlotHistory`): a scene cut (the pre-upscaler detector's rule on the
  CPU: a luma thumbnail of the proxy on an 8-pixel step, a cut at a mean difference of 40 of 255;
  `NEURAL_FORGE_SCENE_CUT` applies here too), another size or format, a request that did not reach the
  network (the effect off, the model not ready, a failed frame), more than 500 ms since the previous frame,
  another network build. Each reset is logged (`[native] after the upscaler: resetting the history (...)`).

**The optical flow** (`post_flow`). With `flow` selected, `create_device` adds the extension, its
`opticalFlow` feature and one queue of the family flagged `QUEUE_OPTICAL_FLOW_BIT_NV` (family 5 on the RTX
5070, its only queue), where the device has all three and the application does not use that family itself;
a request with them that the driver refuses is made again without. The RTX 5070's driver offers output grids
of 1x1, 2x2 and 4x4 for 32x32 to 8192x8192, inputs `B8G8R8A8_UNORM`, `R8_UNORM` and NV12, vectors
`R16G16_S10_5_NV` (queried 2026-10-19). Per slot and frame size a session at the proxy's size with the 4x4
grid, performance level fast: two input images used in turn (the proxy's four bytes copied in as they are,
RGBA or BGRA: both frames are the same) and one flow image. The frame's proxy is the input and the previous
frame's the reference, so a vector points from where a pixel is to where it was. A frame with a history is
three submits: the proxy into the session on the compute queue (signalling a semaphore), the flow on the
optical-flow queue (waiting for it, signalling another), and the frame, which waits for that one at its
transfer stage, copies the vectors into a host-visible buffer the preprocess reads (interpolated between the
grid cells' centres) and is waited for by the server thread, as every frame is. A submit of the flow's that
fails gives it up (both queues drain; `zero` from then on). The entry points are resolved by hand and checked.

**Tests**: lavapipe has no optical flow, so the tests feed the same shader path a synthetic flow field (`the_flow_reprojects_the_history`: a 3-pixel pan with the matching flow reads the first
frame's output 3 pixels back) and check the fallback (`flow_without_the_extension_runs_as_zero_motion`). The
flow's direction (input to reference) is the extension's documented meaning.

**The GPU work it adds** (per slot, at W x H; nothing with `none`):

| | `zero` | `flow` (on top of `zero`) |
|---|---|---|
| Buffers | previous proxy 4 B/px, reprojected history 8 B/px, one history per pass 8 B/px (1440p, one pass: 74 MB) | two input images 4 B/px, the flow image and its host-visible copy 4 B per 4x4 cell (1440p: 30 MB + 2 x 0.9 MB) |
| Per pass | the preprocess reads 8 B (history) + 8 B (two proxies) and writes 8 B per pixel; the composite reads and writes 8 B per pixel; same dispatches as before | the preprocess's history read is the five-tap filter (up to 20 texel reads) and four flow reads |
| Per frame | one buffer copy of the proxy (4 B/px); the cut thumbnail on the CPU (1/64 of the pixels) | one buffer-to-image copy of the proxy, the optical flow at the proxy's size, one image-to-buffer copy of the vectors, two more submits and their semaphores |

**Measured** (RTX 5070, 2026-10-19; `pan` at 2492x1370, a 60-frame series each; the number is the edit's mean
change between frames over its mean size, motion-compensated, so lower is steadier):

| Case | `none` | `zero` | `flow` |
|---|---|---|---|
| Pan, 6 px/frame | 0.314 | 0.355 | 0.324 |
| Still | 0.249 | 0.296 | |

The model server took 17.1 ms with `none`, 17.5 with `zero` and 19.5 with `flow`. The optical flow ran on the
RTX 5070 (queue 0 of family 5, 623x343 vectors for the 4x4 grid) with no fault, and the reprojection follows
the pan (`flow` is close to `none` while moving, where `zero` is worse). Neither history makes the edit
steadier than none, even on a still frame, where `zero` is the exact previous output: the network's answer
blended toward its own previous answer changes more from frame to frame, not less. The default stays `none`
and both histories stay a testing aid.

## Model resolution before the upscaler (3.8.0, 2026-10-19)

The Model tab's **Model resolution** (`preupscale_scale`) sets the size the network runs at before the
upscaler, as a scale of DLSS's input size: **Full** 1 (the default), **Quality** 0.667, **Balanced** 0.58,
**Performance** 0.5, **Ultra Performance** 0.333 (`enums::model_scale`; `shmctl set` takes any value from
0.25 to 1). Each pass can override it (`pass<N>_scale` with the `SCALE` override bit, the per-pass dialog's
Model resolution). Each side is rounded and kept at 8 texels or more (`native::model_extent`).

**Full is the frame as it was, bit for bit.** At 1 nothing of the following is built or recorded: one
graph at the render extent, the same `N` buffers, the same parameters
(`at_full_model_resolution_the_frame_is_bit_for_bit_as_before` runs two passes and three frames against a
pass built the old way). The default is not lowered: the 3.3 attempt at a reduced resolution before the
upscaler kept 12% (75%) to 28% (50%) of the added detail on a static facade at 4K and showed soft diagonal
streaks on flat lit surfaces at 50% (LESSONS.md, "3.3 edit cache: reduced model resolution before the
upscaler (built, removed)"), and lowering it by default needs the picture not to suffer. Below Full the
setting is the user's choice of speed over detail.

A pass below the render extent runs inside its own `N` buffer, around the network:

- **Reduce** (`native_reduce.comp`, first): the pass's input at its extent, each texel the area (box)
  average of the render pixels it covers, taken in the encoded domain the network reads (the encode's
  picture, or the previous pass's answer), so the picture keeps its look and only loses pixels. The
  motion (the accumulated field, or DLSS's packed vectors with the edit cache off) and the depth copy (as
  f32 device depth) are taken at the render pixel under each texel's centre.
- **The network** at that extent, with its own history images there. The vectors keep their units; the
  pass's parameters scale the vectors' scale, the jitter's step and the control mask's motion threshold by
  the extent's ratio to the render extent, so the history reprojects in the pass's own pixels.
- **Enlarge** (`native_enlarge.comp`, last): the network's **edit**, its answer minus its input at its
  extent in the encoded domain, is brought to the render extent by a joint bilateral upsample guided by the
  full-resolution input, and added to that input: each render pixel takes the four texels around it, each
  by its bilinear weight times `exp(-|guide - input|^2 / 0.08^2)` over the three encoded channels
  (`GUIDE_SIGMA`, upstream's JBU sigma), never below 1e-6 of the bilinear weight. The full-resolution
  frame's own detail is kept as it is; only the change is enlarged, and an edge in the frame keeps each
  side's edit. The result is written to the answer buffer at the render extent, where the write-back
  decodes it and the next pass reads it as before.

Everything after the passes is at the render extent: the decode and its clamped-highlight rule, the edit
cache (which captures the full-size edit, so its cached holds are unchanged; its buffers live in pass 1's
input features when they fit, else in a buffer of the native pass's own of the same layout, 56 bytes a
render pixel), the copies between passes and pass feedback, a cache dump. The control mask is built at each
extent by the first pass there. The scene-cut word is shared by every pass.

**Per pass**: every pass downsamples its own input, the previous pass's full-size answer, to its own
extent, and enlarges its edit back; passes at the same extent share one graph and one lane (input,
motion, depth, mask and answer buffers at that extent). `each_pass_runs_at_its_own_model_resolution`
checks pass 1 at full size and pass 2 at half: pass 2's input lanes are the 2x2 means of pass 1's answer.

**The network for several sizes**: the loader builds one graph per distinct extent the passes run at, pass
1's first, together (`nn::Network::build_sizes`, `nf_native_build_sizes`: a warm-up each, the largest
first so the split-K scratch is sized for it, then every recording in descriptor pool 1). The graphs share
the weights and kernels and run one after another, never at once. Every build of a set carries one
generation. VRAM goes with the field: the network's activations are about 364 bytes a field texel, so a
pass at 0.5 needs about a quarter of Full's, and each further size adds its own. Build time is one warm-up
per size.

**Changing it**: a change of any pass's model resolution builds the native pass again (its extents are
fixed for its life, `take_other_extents`), so every history starts afresh at its new extent, the edit cache
refreshes, and the loader builds the new set of sizes while the frames go to DLSS untouched (a moment, as
for a switch of the network). `another_model_resolution_starts_the_history_afresh` checks it.
`at_half_model_resolution_the_edit_comes_back_at_full_size_without_crossing_an_edge` checks the reduce
(the network's input is the area average) and the enlarge (a constant edit comes back with its own
magnitude on every pixel; opposite edits on either side of a hard edge stay on their sides, where plain
bilinear gives the pixel next to the edge half the other side's). The rig test
`two_sizes_built_together_each_give_the_head_of_their_size_alone` (`NEURAL_FORGE_NATIVE_RIG`, an NVIDIA
GPU) checks two sizes built together against one built alone.

**Measured** (RTX 5070, 1440p, 2026-10-19; HARDWARE_VALIDATION.md, "3.8.0"):

- **Before the upscaler, GTA V Enhanced** (frame generation off; DLSS's input 1486x836; the added detail on
  the still shot's facade box, windows 10 and 14 frames after its cut, against the model off): Full 77.1
  real fps and x1.28-1.32 (3.7.0: 76.6, x1.24-1.26), Quality 84.4 and x1.01, Performance 85.2 and
  x0.97-0.98, Ultra Performance 86.8 and x1.10; the model off 90.3. The network 6.05 / 3.20 / 3.04 / 2.65 ms.
  At DLSS's small input the reduced sizes keep little of the added detail, as the 3.3 attempt found at 4K.
- **After DLSS, Crimson Desert with Ray Reconstruction** (frame generation on, the model at the 2560x1440
  output): shown fps Full 118-120, Quality 143.4, Performance 150-153, Ultra Performance 160.9 against 183.2
  with the model off; the network 15.8 / 8.25 / 5.42 / 3.79 ms. At the output size Performance runs the
  network at 1280x720, close to what Full runs at before the upscaler in GTA, and keeps most of the added
  detail.

Full stays the default.

## The colour of the edit (3.8.0, 2026-10-19)

Controls that act on the network's **change** to the frame rather than on the frame: the edit e = answer -
input of a pass, in the domain the pass works in (before the upscaler the encoded picture,
`preupscale_encode.comp`'s per-channel shoulder and sRGB curve of the scene-linear input; after it the 8-bit
proxy's own values). The idea of shaping the change, per pass or for the whole stack, with an adaptive look-up
table on top, comes from another DLSS-NR app's feature list; the design here is this project's own.

| Control | Setting (default) | Step |
|---|---|---|
| Detail | `edit_detail` (1) | e = low + detail x (e - low), low the edit's 3x3 binomial blur ([1 2 1] squared / 16, edges clamped) |
| Saturation | `edit_saturation` (1) | l = BT.709 luma of e (0.2126, 0.7152, 0.0722; on the encoded values), e = contrast x l + saturation x (e - l): the grey part scaled by contrast, the part of no luma by saturation |
| Contrast | `edit_contrast` (1) | (the same line) |
| Look-up table | `edit_lut_strength` (0, off), `edit_lut_adapt` (0.05); the file is `edit_lut` in `config.ini` | on the last pass's result r: r = mix(r, LUT(r), strength x min(1, max channel abs(r - input) / adapt)), trilinear in a 3D `.cube` table (2 to 65 points, `DOMAIN_MIN`/`MAX` honoured) |
| Gamut guard | `edit_gamut_guard` (on) | where input + e leaves 0-1 in a channel, e is shortened along itself (r = input + t x e, the largest t <= 1 that fits); without it each channel clamps |
| Dither | `edit_dither` (off) | after the upscaler only: an 8x8 Bayer offset in (-0.5, 0.5) of the 8-bit step added before the answer is rounded |

The order is detail, saturation and contrast, the table, the guard, then the clamp to 0-1 and the half-float
truncation (before the upscaler) or the 8-bit rounding (after it). The three scales apply to every pass (the
Model tab's value, "the whole stack"), and a pass may override any of them in the per-pass dialog
(`pass<N>_edit_saturation`, `_contrast`, `_detail`, override bits 128, 256 and 512 of `pass<N>_overrides`);
the guard and the dither apply to every pass that has a step, the table to the last pass of the frame only, so
it grades the stack's result once.

**Why the guard is on by default.** The encode maps scene-linear 0 to 0 in each channel and the shoulder's top to
1, so an encoded channel below 0 is a colour outside BT.709's primaries in linear terms and one above 1 cannot be
written back; clamping one channel turns the hue (the decode clamps per channel too). Shortening the whole change
keeps its direction. In range, t is 1 and the guard does nothing (the test compares a guarded and an unguarded
in-range pixel bit for bit); with the scales at 1 the network's own answer is already clamped by the composite
and the stage is not recorded at all, so the guard on changes no default picture. On the test's red pixel at
saturation 2, the clamp turns the change's green-to-red ratio from -0.26 to -0.52 and the guard keeps it within
0.01.

**Where it runs.** Before the upscaler, `native_colour.comp` runs after each pass's composite in that pass's
`N`, in two dispatches (phase 0 the edit's low-pass into its own buffer, phase 1 the steps in place in the
answer), sharing the frame shaders' set layout (bindings 12, the low-pass, and 13, the table) and `Params` (four
vec4s appended). The history (`u_next`) is written by the composite before it, so it stays the network's own
answer and the next frame reprojects what the network made, never the reshaped picture; the edit cache captures
the reshaped answer, so the frames in between carry it. The stage is recorded only while some pass of the frame
reshapes its edit or the table applies (`colour::wanted`; the buffers are recorded again when that changes), and
a pass whose values are neutral returns at once: its answer is untouched, bit for bit. After the upscaler the
steps are inline in `native_post_composite.comp` (the detail's low-pass computes the eight neighbours' answers
there, so it costs about nine times the composite while detail is not 1); with no bit set the shader takes the
path it took before.

**The look-up table.** It is not in the shared memory: a 65-point table is 4.4 MB and the header is 64 KiB of
settings. `edit_lut` in `config.ini` names the file (the GUI's Colour group and `neural-forge-cli shmctl lut
FILE|none` set it, after checking it parses, and bump `edit_lut_seq`); the layer reads it on a thread of its own
(`preupscale::colour::current`, through `persist::read_config`) when `edit_lut_seq` moves on while the strength
is above 0, and the previous table (or none) stays in use until it is read. Each path uploads it into a
host-visible buffer of its own. The table's strength adapts to the edit: a pixel the network left alone keeps
its colour at any strength, and the full strength is reached at an edit of `edit_lut_adapt`.

Tests on lavapipe (`the_colour_of_the_edit_reshapes_the_answer_and_leaves_the_history`,
`the_colour_of_the_edit_after_the_upscaler_reshapes_and_dithers_the_8_bit_answer`, and the settings' own):
neutral values give the answer bit for bit with the stage recorded or not, and the history lane is the network's
answer; saturation 0 leaves a grey change of the edit's luma, contrast 0 one of no luma; detail 0 removes a
checkerboard and keeps the broad change; the guard as above; the table at strength 0 changes nothing, at full
strength it is the table where the edit is large and nothing where there is none; the dither gives each pixel the
level below or above its value and averages to it. Each fails with its step taken out of the shader (checked
once).

Not measured in a game yet: what the controls cost on the RTX 5070 (the low-pass and the steps are two
full-frame dispatches per pass that has a step) and how they look.

## Where the model runs (3.8.0, 2026-10-19)

The network can run at three stages of a frame (`preupscale::placement`), in the frame's order:

| Stage | Where | Size | Picture |
|---|---|---|---|
| Before the upscaler | the hold inside DLSS's command buffer, or the split in front of it (Phase 4b, PRE_UPSCALER_DESIGN.md) | DLSS's input (render) size | the scene in HDR, no HUD |
| After DLSS | the hold right after DLSS's output kernel, inside DLSS's command buffer | the output size | DLSS's upscaled (and with Ray Reconstruction denoised) frame, before the game's post-processing and HUD |
| At present | the after-the-upscaler path (2.1), `native_post` | the presented size | the 8-bit proxy of the finished frame, HUD included |

**The setting.** `placement` (Model tab, Placement; `shmctl set placement auto|before|upscaled|present`) picks
the stage, and a pass can have its own (`pass<N>_placement` with override bit `PLACEMENT`, 1024, the per-pass
dialog's Placement). Each pass's setting resolves for what the frame offers (`placement::stage_of`): Ray
Reconstruction running, DLSS's output learnt, and DLSS's input identified (the hook before the upscaler):

| Setting | Super Resolution, output learnt | Super Resolution, no output | Ray Reconstruction, output learnt | Ray Reconstruction, no output | No hook (no DLSS, no NVX) |
|---|---|---|---|---|---|
| `auto` (default) | before | before | after DLSS | at present | at present |
| `before` | before | before | at present | at present | at present |
| `upscaled` | after DLSS | before (as `auto`, logged) | after DLSS | at present (as `auto`, logged) | at present |
| `present` | at present | at present | at present | at present | at present |

`auto` is the default and keeps every game where it was: before the upscaler, after DLSS while Ray
Reconstruction runs (whose input is the noisy frame, never held), at present where neither hold exists.
`before` with Ray Reconstruction is 3.7.0's behaviour: its frames go to the after-the-upscaler path. DLSS's
output is learnt when eight submitted DLSS buffers in a row name the same output kernel and image ("DLSS's
output:" in the log); it counts as not learnable once eight holds after it in a row are refused (its
format, usage or layout, a buffer that cannot carry a hold; not a buffer that already holds one or busy
staging slots, which pass), logged once, and is tried again 5 s later, so `upscaled` runs where `auto` would
meanwhile instead of not at all. A hold recorded starts the count over. (A first version kept the refusal
for as long as DLSS kept that output: in Crimson Desert with Ray Reconstruction on, the misses while the game
loads were enough to send every frame of the session to the after-the-upscaler path.)

**Per pass.** Passes run in order, and each pass whose placement resolves to a stage runs there
(`placement::Plan`). Two rules keep the plan sound, each move logged once (`[placement]`):

- A pass never runs at an earlier stage than the pass before it: it moves to that pass's stage (pass 2
  `before` after pass 1 `upscaled` runs after DLSS).
- The present stage does not share a frame with the holds: passes at present after held passes move to the
  last hold's stage (after DLSS where the output is learnt, else before the upscaler). A pass 1 at present
  takes every pass with it. The two paths cannot run one frame together: the network is built for one set of
  sizes at a time and the after-the-upscaler path takes it only once no hold has used it for a second
  (`Loader::claim_post`, `PATH_SWITCH_MS`), and its 8-bit proxy is made after the held frame is presented,
  so a mixed frame would have the paths take the network from each other every frame.

So a frame runs one contiguous group of passes before the upscaler and the next after DLSS, or every pass
at present. Each stage's group is a `Passes` of its own (`Passes::stage`): its pass 1 is the frame's first
pass at that stage, with that pass's settings, model resolution and colour; pass feedback acts within the
group; the look-up table grades only the stack's last pass. Each hold carries its stage and both groups'
ranges from the moment it is recorded (`placement::Hold`, in `InlinePoint::hold`; the split hold takes its
plan at the submit, `Tracking::split_hold`), so the two holds of one buffer always split the passes the
same way.

**Two holds in one buffer.** A command buffer can carry the hold before the upscaler and the hold after
DLSS's output (`inline::Slots::by_cb` holds a list): each has its own staging slot and events, the jobs
are handed to the worker in the order recorded, and the worker runs them one after the other, waiting for
each hold's work before it releases the buffer, which is the order the GPU reaches them. A buffer carrying
only the hold after DLSS is still split in front of where the split applies (GTA V), and the split's work
on the game's queue is done before DLSS's buffer reaches the event the hold after DLSS waits for. The hold
after DLSS records the next layer's launch first, then the hold behind it. Its jitter is zero: DLSS's output
is not jittered (the edit cache keeps its interval there).

**Each stage has its own resources** (`Session::res` before the upscaler, `Session::res_after` after
DLSS): the HDR encode, the native pass with each pass's history, the edit cache, the scene-cut detector and
the control mask, each at its stage's size, so the two groups never share a history or a cut decision.

**Sizes.** The hold after DLSS's native pass is built at the output size, and the model resolution
(`preupscale_scale`, `pass<N>_scale`) applies there too, as a scale of the output size. Both holds of a frame
ask the network for the same set of sizes (`native::frame_sizes`: the group before the upscaler's sizes at
the render size, then the group after DLSS's at the output size, each once), so neither rebuilds it for the
other: one build of several sizes (`nf_native_build_sizes`), whose graphs share the weights, kernels and
split-K scratch and run one after another on one queue. With every pass before the upscaler the set is the
one size it always was. Each size adds its own activations (about 364 bytes a field texel): at 1440p a pass
after DLSS at Full is about 1.3 GB, with a group before the upscaler at DLSS Quality another 0.6 GB.

**After DLSS there is no depth at the output size** (`after_point` gives none): the control mask's sky rule
is dropped there (`Mask::for_frame` drops it without a depth copy), so `mask_sky` keeps no sky after DLSS;
the dark and motion rules work (the motion vectors are resampled to the output size), the edit cache runs
without its depth tests, and the scene-cut detector is the stage's own. The stage runs before the game's HUD,
so it needs no HUD handling.

**The after-the-upscaler path** serves a held game's frames only while every pass of the plan runs at
present (`post_serves`): `present`, or `auto`/`before`/`upscaled` resolving there (Ray Reconstruction without
a hold after DLSS recorded within a second, which is 3.7.0's hand-off). Then the holds stand down (no hold is
recorded or split, the frames go to DLSS untouched) and the post path runs as in any game without DLSS.

Tests on lavapipe: `each_placement_resolves_to_its_stage_for_what_the_frame_offers` (the table above),
`passes_split_into_contiguous_groups_in_the_frames_order` (the two rules),
`each_stage_records_only_its_own_passes` (two passes, before and after DLSS: each stage's pass runs alone
with its own intensity at its own size, and both holds ask for the same two sizes),
`the_post_path_serves_only_frames_whose_passes_all_run_at_present`,
`a_hold_after_dlss_that_keeps_missing_gives_its_passes_back`, and the settings' own.

Measured in Crimson Desert (RTX 5070, 1440p, frame generation on; HARDWARE_VALIDATION.md, "3.8.0"): with
Ray Reconstruction on at `auto` (the hold after DLSS) 118-120 fps shown with the full network, 140 with the
small one, against 36 on the after-the-upscaler path and 183 with the model off, with no misses; with it off
at `upscaled` (Super Resolution, the network at the output size) 133 fps, and at `present` 41. Not measured:
two passes split across the two holds. Hogwarts Legacy ran Super Resolution with its Ray Reconstruction
setting on, so it did not exercise the hold after DLSS.

## The HUD at present (3.8.0, 2026-10-19)

The presented frame carries the game's HUD, which the network's edit smears (text and icons gain the
network's texture and lose their edges). Two things keep the edit off it on the after-the-upscaler path
(`native_post`, `native_post_hud.comp`); the stages before the upscaler and after DLSS run before the HUD is
drawn and need neither.

**Regions** (`hud_regions` in `config.ini`, `set_hud_regions=x,y,w,h[,keep|soften[,strength]];...` in 0-1 of
the frame; the Model tab's HUD group; `shmctl set hud_regions "..."`): up to eight rectangles, each `keep` (no
edit inside) or `soften` (the edit times its strength, 0-1, default 0.5). Outside a rectangle the edit comes
back linearly over a feather of `height / 360` pixels (4 at 1440 lines, at least 2), so there is no seam.
The header holds them as numbers (`hud_region_count`, `hud_region[8]`); `config.ini` holds the text, the one
setting that is not a number (`persist::HUD_REGIONS`).

**Detection** (`hud_detect`, on by default): per pixel, a confidence (0-1) kept from frame to frame. On a
frame where the scene moves as a whole (at least 30% of the frame's 8-pixel thumbnail cells changed by 4 of
255 or more since the last frame, measured on the CPU from the proxy the request carries), a pixel whose
colour changed by 2/255 or less (largest channel) against the previous frame's proxy and whose neighbourhood
(nine taps two pixels apart) spans 0.15 of luma or more gains 1/16; every other pixel loses 1/4. Above 0.5 a
pixel is HUD, and the edit is kept off it and fades back over two pixels around it (1, 2/3, 1/3). On a still
frame nothing is dispatched: every confidence holds, so a still scene (a paused game, a menu) never grows a
mask, and a HUD found while moving stays found. The previous proxy is the history's own copy while the
history is on (`NEURAL_FORGE_POST_MOTION`), else the HUD's, copied at the end of each frame.

**Why on by default.** A new confidence starts at 0 for every pixel (the buffer is zeroed on its first
frame), and nine moving frames in a row are the least that make a pixel HUD, so the first frames after it is
made detect nothing, and where nothing is detected the frame is exactly what it was without the detection
(the scale is 1 everywhere and the composite takes its old path; `the_shipped_frame_stays_bit_for_bit` runs
with it on). Static, sharp, while the scene moves is the HUD's signature; the scene itself moves with the
camera.

**How it is applied.** Before the passes, phase 0 (the detection, on moving frames), phases 1 and 2 (the
scale of the edit per pixel: the detection's, feathered by the distance to the nearest confident pixel, and
each region's, the smallest) write a float a pixel. The
last pass's composite moves its answer back toward the frame's own proxy by that scale, after the colour of
the edit and the dither, so the whole stack's edit is scaled and a `keep` pixel is the frame's texel exactly.
With the history on, the preprocess also rejects the history wherever the scale is below 1 (as the per-pixel
change rule does), so HUD text that changes never ghosts.

**Cost.** Three dispatches of 8x8 groups over the frame, (2560 / 8) x (1440 / 8) = 57,600 groups each at
1440p: phase 0 (the detection, only on moving frames) reads nine proxy texels two pixels apart, the previous
texel and the confidence and writes the confidence; phase 1 (only with the detection on) reads five
confidences along the row and writes a distance; phase 2 reads five distances down the column and the
regions and writes the scale. The 5x5 feather in one pass (25 taps) cost five times as much on lavapipe, and
a shared-memory tile of the taps more still there, so the distance is found in two passes. The buffers are
4 bytes a pixel each (confidence, distance, scale, and the own previous proxy without the history: 59 MB at
1440p). Measured on lavapipe (CPU) at 2560x1440, median of seven: phase 0 8.9 ms, phase 1 4.1 ms, phase 2
7.0 ms (two regions), against 8.8 ms for one pass's composite there; on the RTX 5070 each is a few tens of
megabytes of memory traffic, about 0.05 to 0.1 ms (an estimate, not measured).

Tests on lavapipe: `hud_regions_keep_the_edit_off_or_scale_it_and_leave_the_rest` (the frame's texel exactly
inside a keep rectangle, half the edit within a 1/255 step in a soften one at 0.5, the answer without regions
bit for bit outside the feather, no history under a region), and
`the_hud_detection_finds_static_text_over_a_moving_scene_and_not_in_a_still_one` (a block of 2-pixel cells over
a scene that changes everywhere: nothing for nine frames, then no edit on the block and the scene bit for bit as
without the detection; twenty still frames grow nothing).

Measured in Crimson Desert at `present` (2026-10-19): the detection kept the minimap ring, the compass and
the HUD text untouched (by eye on a capture). Still to see: GTA V at `present` with the HUD on screen while
driving and while standing: whether the minimap, the weapon wheel and the money text are found while moving and stay
clean, what a first-person weapon model or a third-person character that stays in place does (it can be found
as HUD while the camera turns), and what the detection costs on the RTX 5070.

## Decided before Phase 1

1. **The extractor in a public repository.** `extract-model` reads NVIDIA's weights out of the DLL
   so they can run outside NGX. It reads only resource data (no code, nothing decrypted, no check
   bypassed). But running the weights outside NVIDIA's runtime is a different use than reading
   them, and the NGX licence's terms on it are yours to judge. Nothing has been pushed;
   the commit is on the local `native-backend` branch only.
   *the maintainer said move on to Phase 1 (2026-10-07). Still nothing pushed: pushing is Phase 4.*
2. **The 1440p margin.** The network alone is about 0.7 ms slower than the helper's evaluate at
   1485x836 on an idle GPU. Phase 3 will say whether the removed hand-off costs make up for it in
   game; if they do not, success criterion 2 fails at 1440p and passes at 4K.

## Attempt log

| Date | What | Result |
|---|---|---|
| 2026-10-07 | `apt install -y python3-pefile` on the test machine | Refused: the SSH session is a non-root user, not root (the handoff assumed root). Used a venv in `~/scratch/venv` instead. |
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
| 2026-10-07 | Driver 615.71.09 -> 615.78.08 (the maintainer's request, installed with a separate driver install script, rebooted) between the first and the second Phase 3 pass | Same capabilities (32-bit still has no `VK_NV_cuda_kernel_launch`); the rig test still bit-exact; network time unchanged (1485x836 5.505 ms, 2228x1253 10.672, 2560x1440 13.349, 3840x2160 30.018; chaining off 5.947, 10.939, 13.731, 30.118). Phase 3 runs again in full on 615.78.08; the 615.71.09 pass of 1440p without frame generation is kept as a reference. |
| 2026-10-07 | `NEURAL_FORGE_CAPTURE_AT=5:20000` in the first Phase 3 runs | Nothing captured: the trigger never fired. Resumes are now logged when it is set; the captures move to dedicated runs. |
| 2026-10-07 | Stale motion vectors? (DLSS's buffer synchronizes them before its launch) | Consecutive dumps 393-396: inconclusive, the scene barely moves (median 0.33 px). Open. |

## Checklist

- [x] 0.1 driver capabilities recorded
- [x] 0.2 other repo read, contracts summarized, commit SHA recorded
- [x] 0.3 dlss5vk builds on Ubuntu 26.04
- [x] 0.4 model directory extracted, loads with hash verification
- [x] 0.5 bench numbers on the rig at four sizes
- [x] 0.6 output compared against the helper
- [x] Phase 0 report written, go or no-go stated, stopped for the maintainer
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
- [x] Phase 3 benchmarks (cut short on the maintainer's request; long session not run)
- [x] Phase 4 report written, branch pushed, stopped for the maintainer
- [x] Phase 4b hold inside DLSS's command buffer native
- [x] Phase 5 helper removed (3.0.0)

Open: Lords of the Fallen in play (its privacy policy needs accepting first) and Remnant II (installing);
the Wukong startup fault (Unreal Engine 5, frame generation on), Wukong's motion vectors not
copyable, Spider-Man and Shadow Warrior 3 measured in play (menus only so far).
