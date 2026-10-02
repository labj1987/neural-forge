# Pre-upscaler probe (`NEURAL_FORGE_PROBE_NGX`)

**Status: hooks built, rig run done 2026-10-02.** Diagnostic only, nothing user-facing.

## Question

Can the layer see the game's own DLSS Super Resolution work, so that the model could one
day run on the game's internal render-resolution image *before* its upscaler instead of on
the upscaled swapchain image?

Under Proton, a D3D12 game's DLSS reaches Vulkan through vkd3d-proton and DXVK-NVAPI as
CUDA kernels: `VK_NVX_binary_import` (`vkCreateCuModuleNVX`, `vkCreateCuFunctionNVX`,
`vkCmdCuLaunchKernelNVX`) on image views registered through `VK_NVX_image_view_handle`
(`vkGetImageViewHandleNVX`, `vkGetImageViewHandle64NVX`, `vkGetImageViewAddressNVX`).
Those are the calls the probe watches.

## What the probe does

With `NEURAL_FORGE_PROBE_NGX=1` (and the layer enabled), the layer:

- records every image's extent, format and usage at `vkCreateImage`, and every view's
  image, base mip and format at `vkCreateImageView`;
- logs the first registration of each view through `vkGetImageViewHandleNVX`,
  `vkGetImageViewHandle64NVX` or `vkGetImageViewAddressNVX`: the view's extent, format and
  usage, and the returned handle or address;
- logs each `vkCreateCuModuleNVX` (binary size) and `vkCreateCuFunctionNVX` (kernel name);
- counts `vkCmdCuLaunchKernelNVX` per frame with the command buffers they were recorded
  into, their grid and block dimensions, shared memory, and parameter and extra counts.
  The parameter pointers are never read;
- numbers each `vkQueueSubmit`/`vkQueueSubmit2` call between presents and notes the ones
  that carry a launch-bearing command buffer (secondaries count through
  `vkCmdExecuteCommands`).

Every hook forwards the call unchanged and returns the next layer's result. With the
variable unset, none of these entry points is intercepted: the framework's hooked-command
list is exactly the default one, and the layer hands out the next layer's pointers.

Interception: the pinned `vulkan-layer` framework already generates hooks for every NVX
command in ash 0.37.3. The probe adds them to the list returned by
`Layer::hooked_device_commands` only when the variable is set. `vkGetImageViewHandle64NVX`
is newer than ash 0.37.3, so the framework doesn't know it; `entry_points.rs` wraps the
next layer's pointer for it in `vkGetDeviceProcAddr`, again only with the probe on.

## Running it on the rig

Before the run, select DLSS as GTA's upscaler at a preset that renders below the output
resolution (Quality or Balanced, not DLAA), with frame generation off. Then run the
unattended benchmark runner (`scripts/gta-bench.sh`):

```bash
scripts/gta-bench.sh --host lordnikon probe-ngx VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PROBE_NGX=1 NEURAL_FORGE_LOG=/tmp/neural-forge-probe-ngx.log
```

`NEURAL_FORGE_LOG` puts the layer's lines in a file. Without it they go to the game's
stderr, which is the Proton log. If the file doesn't show up on the host, look for the lines
in the Proton log instead. Each process logs its own `on in pid N` line, so lines from the
game process can be told apart from any other process that loads the layer.

```bash
grep -F '[probe-ngx]' /tmp/neural-forge-probe-ngx.log | head -200
```

## What to look for

1. `device ...: vkGetImageViewHandleNVX present, ... vkCreateCuFunctionNVX present`. If both
   say `absent` on GTA's device, DLSS doesn't run through these extensions here, and the
   rest of the log is empty.
2. `vkCreateCuFunctionNVX: "<name>"` lines: the DLSS kernel names.
3. `vkGetImageViewHandleNVX view=0x... (WxH FORMAT USAGE) -> handle 0x...` lines: the
   images DLSS was handed. Expect a group at the internal render extent (color, depth,
   motion vectors) and one at the output extent (2560x1440).
4. `first launch-bearing submit: frame F submit #i ... on queue Q` and then, at the next
   present, `was submit #i of S before present on queue P (same|a different queue)`.
5. `frame N: launches=K cmdbufs=[...] views_registered=M (...) submits=S launch_submits=[#i@Q:k]
   kernels=[...]`. This is logged on frame 0, every 60th frame, every frame where the set of
   registered view extents changes, and on the first frame with launches.

## Verdict questions

1. **Is the input image identifiable?** Yes if the registered views include one at the
   internal render extent with a color format, and one at the output extent with storage
   usage. Also check that the registered views are the same from frame to frame (stable
   handles), or at least stable in extent and format.
2. **Is the launch in a submit the layer can hold or split?** `launch_submits` shows which
   submit carries the launches and how many submits follow before the present. The layer
   can hold a submit in `vkQueueSubmit`, but it can't split a command buffer the game has
   already recorded. So:
   - If the DLSS launches are in their own submit, after the submit that renders the scene,
     the layer can put its own capture and write-back submits in between.
   - If they share a command buffer with the scene rendering, the layer would have to
     inject commands while the buffer is recorded (at `vkCmdCuLaunchKernelNVX`). Any wait
     for the model would then have to be recorded inside the game's own command buffer.
3. **What would a synchronous model call there cost in the two-process design?** The
   game's submit thread would block for the capture, the shared-memory hand-off, the
   helper's evaluate, and the write-back, all at the internal resolution. Today the full
   round trip at 2560x1440 is about 19-20 ms per model frame (`[sync]` total, 1.0.x, see
   `docs/OPENDLSS_REVIEW.md`), and evaluate alone is about 10 ms. The model's time scales
   roughly linearly with pixel count above 1080p, so a Quality render extent (about 44% of
   the output's pixels) should cost less. The submit can't be released until the answer
   is back, though, so the GPU idles for that time unless the submits after it (from
   `launch_submits` and `submits`) can overlap. The rig run should report these numbers:
   the launch submit's index, the submits left in the frame, and the render extent.

## Rig result

Setup: lordnikon, build `abd56a9` (main, deployed with `scripts/deploy-rig.sh`; version label
1.0.1, shared memory protocol v8), desktop 2560x1440@288 HDR bt2100, NR on (enabled=1,
working_scale=1, model_interval=2), GTA script mods off (`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`).
GTA `settings.xml`: only `dlssQuality` changed, 1 -> 2. `ResScalingType` was already 4 (DLSS;
a past native run had 0) and `FrameGenType` was already 0. The menu offers DLSS Performance,
Balanced, Quality and DLAA. The probe's registered extent (1707x960 = 2560x1440 / 1.5)
confirms that 2 = Quality, so the old value 1 was Balanced. The file was restored afterwards,
sha256-verified. Run:

```bash
scripts/gta-bench.sh --host lordnikon probe-ngx-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PROBE_NGX=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

The benchmark completed: passes 61.0 / 60.1 / 62.5 / 65.3 / 64.0 fps, pass 4 real 59.5,
displayed 59.9, GPU 90% 197 W. `[sync]` median: total 18.60 ms, capture_gpu 5.70,
wait_answer 12.70, helper 11.50. The run is NR on + DLSS Quality, so it isn't comparable
with the baseline. There were no validation messages, no device-lost errors and no NVIDIA
Xid lines in the kernel log. The only kernel lines were the usual split-lock traps from
`GTA5_Enhanced.e` at exit.

Key lines (game process pid 64451; the launcher process, pid 63959, logs its own frames on
queue `0x555565cfcd40` and never launches anything):

```
[probe-ngx] device 0x74c8500db880: vkGetImageViewHandleNVX present, vkGetImageViewAddressNVX present, vkCreateCuFunctionNVX present
[probe-ngx] vkCreateCuFunctionNVX: "cuda_engine_input_kernel_rel_hdr_colvar_mvlo" -> SUCCESS 0x74c83dca04f0
[probe-ngx] vkCreateCuFunctionNVX: "cuda_engine_output_kernel_rel_hdr_gauss5x5" -> SUCCESS 0x74c83de34390
[probe-ngx] vkGetImageViewHandle64NVX view=0x74caceaf2af0 (1707x960 R16G16B16A16_SFLOAT TRANSFER_SRC | TRANSFER_DST | SAMPLED | STORAGE | COLOR_ATTACHMENT) -> handle 0x140000195f
[probe-ngx] vkGetImageViewHandle64NVX view=0x74caccd378a0 (1707x960 D32_SFLOAT_S8_UINT TRANSFER_SRC | TRANSFER_DST | SAMPLED | DEPTH_STENCIL_ATTACHMENT) -> handle 0x1400001962
[probe-ngx] vkGetImageViewHandle64NVX view=0x74c795ee17d0 (1707x960 R16G16_SFLOAT TRANSFER_SRC | TRANSFER_DST | SAMPLED | COLOR_ATTACHMENT) -> handle 0x140000215e
[probe-ngx] vkGetImageViewHandle64NVX view=0x74c6b46647d0 (2560x1440 R16G16B16A16_SFLOAT TRANSFER_SRC | TRANSFER_DST | SAMPLED | STORAGE | COLOR_ATTACHMENT) -> handle 0x15000043cd
[probe-ngx] first launch-bearing submit: frame 3438 submit #20 (0-based, counted from the previous present) on queue 0x74c85030e990, carrying 14 launches
[probe-ngx] first launch-bearing submit (frame 3438) was submit #20 of 26 before present on queue 0x74c85030e990 (same queue as the present)
[probe-ngx] frame 15300: launches=14 cmdbufs=[0x74c81eb6d290] views_registered=0 () distinct_views_ever=22 submits=20 launch_submits=[#14@0x74c85030e990:14] present_queue=0x74c85030e990 kernels=[dltss_pwin_dec0_layer grid=81x49x1 block=32x1x1 ..., dltss_pwin_enc0_layer grid=81x49x1 ..., hiluma_engine_input_depthinv_mvlo_hdr_v2_rel grid=160x24x1 block=16x16x1 ...]
```

a. **Entry points and kernels.** Yes. Every NVX entry point is present on GTA's devices.
   Some extra devices in the same processes report `absent`. They are not the ones that launch anything; the probe doesn't record which devices they are. NGX creates its
   modules from about 8 KB to 200 KB: `cuda_clear_buffer_kernel`,
   `cuda_reduce_sum_kernel`, `cuda_histogram_kernel`, `cuda_auto_exposure_kernel`,
   `cuda_luma_convert_kernel`, `cuda_auto_exposure_copy_kernel`, the
   `cuda_engine_input_kernel_rel_{ldr,hdr}_{colvar,mvdiff}_{mvhi,mvlo}` and
   `cuda_engine_output_kernel_rel_{ldr,hdr}_gauss{3x3,5x5}` families. The per-frame launches
   are `dltss_pwin_enc0..4_layer` and `dltss_pwin_dec0..5_layer` (the DLSS transformer
   network) plus one `hiluma_engine_input_depthinv_mvlo_hdr_v2_rel`.
b. **Launches and submits.** Each frame has 14 launches (one DLSS evaluation), all in
   one command buffer and all in one submit: 152 of the logged frames show `launches=14`,
   and every `launch_submits` entry is `:14`. Two frames show 28 launches over 2 command
   buffers and one shows 4. Those count launches at record time, and recording crosses the
   present boundary. The submit still carries exactly 14. There are about 20 submits per
   frame (from 14 to 31). The launch submit is #14-#16 in a typical frame. In 157 of 159
   frames it is followed by exactly 5 more submits before the present; the other two frames
   had 6 and 7. It is on the same queue as the present (`0x74c85030e990`).
c. **Registered views.** 22 distinct views were registered, all within frames 3438-3441.
   After that, every frame shows `views_registered=0` and `distinct_views_ever=22`. The
   handles are stable, created once and reused. Distinct (extent, format, usage):
   - 1707x960 R16G16B16A16_SFLOAT, storage: **colour input** (HDR; the kernels are the
     `hdr` variants).
   - 1707x960 D32_SFLOAT_S8_UINT, depth-stencil: **depth**.
   - 1707x960 R16G16_SFLOAT, colour attachment: **motion vectors** (render resolution,
     `mvlo`).
   - 2560x1440 R16G16B16A16_SFLOAT, storage: **output** (several views; NGX also keeps
     history at this size).
   - 1x1 R32G32B32A32_SFLOAT and 1x1 R16_SFLOAT: exposure.
   - 640x384 R16G16B16A16_SFLOAT and 5120x2880 R16_SFLOAT, storage: these look like NGX
     scratch (5120x2880 is twice the output; the hiluma kernel's grid covers 2560x384).

   So there is a group at the internal render extent (1707x960) and one at 2560x1440,
   as expected.
d. **Separable or not.** All 14 launches sit in a single command buffer per frame,
   recorded by vkd3d-proton from the game's own D3D12 command list. NGX's
   `EvaluateFeature` records into the command list the game passes it. The probe doesn't
   count draws or dispatches in that buffer, so this run can't prove what else the buffer
   holds. Two things are known: the launches aren't in a submit of their own that the
   layer could slot work in front of without splitting a buffer, and about 14-16 submits
   come before it, so most scene rendering is in earlier submits. Holding a submit is
   possible. Putting the model *between* the scene and the DLSS kernels only works if
   the buffer starts with the DLSS work. Otherwise it needs commands injected at
   `vkCmdCuLaunchKernelNVX` record time. Counting the non-launch commands recorded into
   the launch-bearing buffer, before and after the first launch, would settle this.
e. **Stability.** The benchmark completed on the first try, with no crash, no device lost,
   no validation output and no Xid. The fps is in the normal range for NR on.

## Verdict

1. **Identifiable: yes.** The 1707x960 R16G16B16A16_SFLOAT storage view is the colour
   input, next to 1707x960 depth and RG16F motion vectors, and the 2560x1440 RGBA16F
   storage view is the output. They are registered once and stable for the whole run.
2. **Holdable: yes. Splittable: not shown, probably no.** The launches are in one submit
   per frame, on the present queue, 5 submits before the present. The layer can hold that
   submit. But all 14 launches are inside one game command buffer that may also hold the
   game's work right before DLSS. Getting the model in front of DLSS would most likely
   mean injecting at `vkCmdCuLaunchKernelNVX` record time (a split barrier and wait recorded
   into the game's buffer), not just holding the submit. This is unconfirmed until the
   buffer's other commands are counted.
3. **Synchronous cost at 1707x960 (estimate).** 1707x960 is 1.64 Mpx, 44.5% of
   2560x1440. Helper evaluate, interpolated linearly between 6.2 ms at 1920x1080 and
   10.2 ms at 2560x1440, comes to about 5.1 ms (pure pixel scaling from 1440p gives 4.5 ms).
   The `[sync]` round trip of 19.0 ms per model frame at 1.0, scaled by pixel count, gives
   about 8.5 ms. Not all of it scales (hand-off and wakeup overheads are fixed), so expect
   about 8.5-10 ms per model frame. During that time the launch submit and the 5 submits
   behind it on the same queue can't go to the GPU, so most of it would show up as a
   stall. At about 60 fps (16.7 ms frames) and model_interval=2, that is roughly 4-5 ms
   added per frame on average, close to today's swapchain-path cost. The gain would be
   applying the model before DLSS, not speed. This is an estimate, not a measurement.
