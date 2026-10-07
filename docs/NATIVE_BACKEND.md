# Native model backend (candidate 3.0)

The plan: run the neural rendering network inside the layer, on the game's own device, with
[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (MIT), and retire the Windows helper, the
Wine/Proton runner, the shared-memory hand-off, the NGX DLL at run time and the caller-identity spoof.
Branch `native-backend` only; nothing merges, tags or releases until Alex has judged the results.

This file is written as the work goes. Every attempt, including the failed ones, is in "Attempt
log" so the next one starts from it.

## Phase 0: go or no-go (2026-10-07)

**Verdict: go, with two cautions.** All three gates hold on the test machine:

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

## For Alex to decide before Phase 1

1. **The extractor in a public repository.** `extract-model` reads NVIDIA's weights out of the DLL
   so they can run outside NGX. It reads only resource data (no code, nothing decrypted, no check
   bypassed), which the "Working with NVIDIA's binaries" rules in AGENTS.md/CLAUDE.md allow as
   written. But running the weights outside NVIDIA's runtime is a different use than those rules
   were written for, and the NGX licence's terms on it are yours to judge. Nothing has been pushed;
   the commit is on the local `native-backend` branch only.
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

## Checklist

- [x] 0.1 driver capabilities recorded
- [x] 0.2 other repo read, contracts summarized, commit SHA recorded
- [x] 0.3 dlss5vk builds on Ubuntu 26.04
- [x] 0.4 model directory extracted, loads with hash verification
- [x] 0.5 bench numbers on the rig at four sizes
- [x] 0.6 output compared against the helper
- [x] Phase 0 report written, go or no-go stated, stopped for Alex
- [ ] 1.1 sources vendored, built from build.rs
- [ ] 1.2 device extensions and features added in the layer
- [ ] 1.3 game device adopted, queue use documented
- [ ] 1.4 native frame path before the upscaler works in GTA V
- [ ] 1.5 backend switch in place
- [ ] 1.6 failure paths tested (missing model, watchdog, allocation failure)
- [ ] 2.1 after-the-upscaler path native
- [ ] 2.2 settings mapping table written, unmapped controls listed
- [ ] 2.3 Setup tab reduced to extract-model
- [ ] 2.4 32-bit layer answer recorded
- [ ] Phase 3 benchmarks and long session done
- [ ] Phase 4 report written, branch pushed, stopped for Alex
- [ ] Phase 5 (only on Alex's yes)

Blockers: none technical. Phase 1 waits for Alex's go and the two decisions above.
