# Pre-upscaler probe (`NEURAL_FORGE_PROBE_NGX`)

**Status: hooks built, rig run pending.** Diagnostic only, nothing user-facing.

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
unattended benchmark runner (`gta-bench.sh`, kept on the rig, not in this repo):

```bash
~/nf-spike/gta-bench.sh probe-ngx VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PROBE_NGX=1 NEURAL_FORGE_LOG=/tmp/neural-forge-probe-ngx.log
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
