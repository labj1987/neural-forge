# The second GPU shows the frame (one-way offload, #63, 2026-11-14)

Code: `crates/layer/src/gpu2_present.rs` (the hand-off, both sides) and `crates/layer/src/device.rs` (`gpu2_*`: the
hooks, the second device's record).

Requested in #63. With the model on the finished frame, the second GPU runs the model and presents the frame
itself. Each frame crosses the link once and the game never waits for an answer; frames the link or the second GPU
cannot carry are skipped (the newest frame wins).

## Mechanism

A Vulkan surface takes one swapchain, and a swapchain belongs to one device. So the real swapchain has to live on
the second device, and the game has to get something else to render into.

1. **The game gets a swapchain the layer owns** (`gpu2_present::Virtual`). `vkCreateSwapchainKHR` on the game's
   device creates no swapchain there: the layer makes plain images of the swapchain's format, extent, usage and
   flags on the game's device and hands out a handle of its own. `vkGetSwapchainImagesKHR`,
   `vkAcquireNextImageKHR`/`2KHR`, `vkQueuePresentKHR`, `vkDestroySwapchainKHR`, `vkSetHdrMetadataEXT`,
   `vkReleaseSwapchainImagesEXT`, `vkGetSwapchainStatusKHR`, `vkWaitForPresentKHR` and the `VK_NV_low_latency2` calls
   that name a swapchain are answered by the layer for that handle and never reach the driver.
2. **The real swapchain is on the second device**, created for the game's surface with the game's own format, colour
   space, extent, present mode, composite alpha and transform (so HDR10/scRGB, VRR and tearing behave as the game
   asked), after `vkGetPhysicalDeviceSurfaceSupportKHR` and the surface's formats and modes were asked of the second
   GPU.
3. **Present is a hand-off.** The game's `vkQueuePresentKHR` submits one copy on the game's queue (image to a
   host-memory staging buffer, after the present's wait semaphores) and returns. Three staging slots: one the second
   GPU is working on, one holding the newest finished frame, one free to write. A frame that finds no slot replaces
   the waiting one or is dropped; both count as skipped.
4. **A presenter thread** takes the newest slot once its copy finished, uploads it into the next image of the real
   swapchain on the second device and presents it through the layer's own finished-frame path, run on that device.
   The second device gets a device record of its own (`NeuralForgeDeviceInfo`), so capture, network, composition,
   HDR path, sharpening, hotkey, captures and Status work there exactly as they do on a game's device. Each slot's
   staging is host memory both devices import (`VK_EXT_external_memory_host`), so the game's GPU writes the frame
   where the second GPU reads it and the CPU copies nothing; where either device cannot import, the slot is a mapped
   buffer per device and the presenter copies it once.

Rejected: a headless shadow swapchain for the game (NVIDIA's driver has no `VK_EXT_headless_surface`: asked on the
5070, no queue family presents to one); keeping the game's swapchain and adding a second one on the same surface
(`VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`).

## Semantics

- **Acquire**: an image that is neither acquired nor still being copied out; waits (up to the timeout) for a copy to
  finish when there is none. The game's semaphore and fence are signalled by an empty submit on the layer's own queue.
  That queue is the after-the-upscaler server's, and its lock here is the swapchain's own: this assumes the game's
  device has no real swapchain of its own beside the layer's (whose finished-frame server would use the same queue
  under another lock). One window per game is what the mode supports.
- **Present result**: `VK_SUCCESS`, or what the real swapchain last said (`SUBOPTIMAL`, `OUT_OF_DATE`,
  `SURFACE_LOST`), which stays until the game recreates the swapchain. The present's wait semaphores are always
  consumed; present fences (`VK_EXT_swapchain_maintenance1`) are signalled by the copy's submit.
- **Recreate**: the old swapchain's presenter is stopped and drained, the new real swapchain is created with the
  old one as `oldSwapchain`, the old one is destroyed. The second device's record and the network stay.
- **Present modes**: the real swapchain has the game's mode (or FIFO where the second GPU has not got it). The game
  is never throttled by it: with FIFO the display rate limits what is shown, not what the game renders.
- **Frame Generation**: its frames are ordinary presents of the game's swapchain and are handed off like any other.
- **Low latency (Reflex)**: sleep requests return at once, markers are dropped, timings report none.
- **Wayland and X11**: nothing of the surface is touched; only the device that owns its swapchain changes.
- **Commands newer than the layer framework** that name a swapchain (`VK_EXT_present_timing`'s four,
  `vkWaitForPresent2KHR`, `vkReleaseSwapchainImagesKHR`) are wrapped through `vkGetDeviceProcAddr`: vkd3d-proton asks
  the present timing of every swapchain, and NVIDIA's driver hung on the layer's handle (the first game run). A
  command that names a swapchain and is in neither list would reach the driver with a handle it does not know.

## Fallback

- Before the swapchain exists (the setting off, no second device, another placement, the second GPU cannot present
  to the surface, a format it has not got, any creation failure): the game's swapchain is created on its own device
  as always, and the second GPU is used the round-trip way. The log and the Status tab say why.
- While it runs (a present or upload fails, the second device is lost): the presenter stops, the game's next
  acquire/present returns `VK_ERROR_OUT_OF_DATE_KHR`, and the swapchain it then creates is a real one on its own
  device. The mode stays off for the session. No frame is shown black: the game recreates before it shows another.
  The second device's record is torn down and its device given back, so the round trip can use it.
  A present whose frames cannot be copied out (its queue's family unknown, or another than the first present's) and a
  timeout after the real swapchain's image was acquired are failures of this kind too, not frames to skip.
- A recreate that names a real swapchain of the game's own as the old one (the mode was switched on while the game
  ran) is refused before the second GPU's device is taken: that swapchain has the window. The mode starts with the
  next game start.
- Whenever the second GPU's side shows nothing any more (the first swapchain refused, a fall-back), its device goes
  back to `preupscale::offload`, so the round trip can take it.
  `NEURAL_FORGE_OFFLOAD_TEST_FAIL=N` fails the second device's `N`th present, to see this in a game.

## Settings and Status

- `second_gpu_present` (0/1, GUI: Model tab, "Second GPU shows the frame"), with `NEURAL_FORGE_OFFLOAD_GPU` and
  every pass at Placement "Finished frame".
- Status: "second GPU shows the frames: <GPU>; N shown, M skipped (P%), copy X ms" or why not.

## Tests (lavapipe)

Hand-off ring (newest wins, skipped counts), acquire/present image states, the newer commands' wrappers, and full
runs with a real swapchain on a second lavapipe device (headless surface): frames arrive unchanged (copied, and
through imported host memory), skip under a slow presenter, recreate, failure falls back with `OUT_OF_DATE` and
leaves the surface free, a swapchain that cannot be shown is refused.

## Measured (2026-11-14, one GPU)

The test machine has one GPU, so the second device is a second device of the same RTX 5070
(`NEURAL_FORGE_OFFLOAD_GPU=same`): it shows that the path works, not what a second card gains. Cyberpunk 2077's
benchmark through Steam, 2560x1440, HDR10 (Proton's Wayland driver), Frame Generation on, the full network at full
resolution on the finished frame: the real swapchain was created on the second device as A2B10G10R10 HDR10 with the
game's HDR metadata, the game recreated its swapchain once at start, and over the run the game presented 10626
frames (about 97 a second in the benchmark) of which 4084 were shown and 6542 skipped (62%); the second device
showed about 20 to 26 frames a second in the benchmark, each taking about 50 ms for the model there (the game and
the model share the one card); the copy took 3 to 7 ms a frame; video memory peaked at 10.3 GB of 12. The recording
(vw, 18 s) has no black or alternating frames and its contact sheet is the benchmark's picture; it flags 6 frames
of colour change at scene cuts and one 205 ms stretch without a new frame.

Why 62% were skipped on one card: the presenter shows a frame only when the model is done with it, and the model
at 2560x1440 took about 50 ms there because it shares the card with the game. So it showed 17 to 20 frames a second
against the game's 97 presents (about 80% skipped in the benchmark, almost none in the menus before the model
engages). The copy (2 to 6 ms) and the swapchain's FIFO at 288 Hz were not the limit. On a second card the shown
rate is one over the upload plus that card's own model time.

Two more runs the same day: with `NEURAL_FORGE_OFFLOAD_TEST_FAIL=1500` the second device's 1500th present failed,
the game recreated its swapchain, got one of its own device, the second device's record was torn down and the round
trip took over on it for the remaining 70 s (no black frame in the recording across the switch, one 382 ms stretch
without a new frame). And with the staging as host memory both devices import: 3887 shown, 5823 skipped, the same
picture (its recording starts in the game's own black loading screen).

Not tested: two different cards (the copy over a real link, a monitor on the second card,
`vkGetPhysicalDeviceSurfaceSupportKHR` refusing the surface), X11/Xwayland surfaces, scRGB swapchains, a resize
while it runs.
