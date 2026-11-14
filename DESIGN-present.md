# The second GPU shows the frame (one-way offload)

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
   HDR path, sharpening, hotkey, captures and Status work there exactly as they do on a game's device. The staging
   memory is imported on both devices (`VK_EXT_external_memory_host`) where both allow it, else copied once on the
   presenter thread.

Rejected: a headless shadow swapchain for the game (NVIDIA's driver has no `VK_EXT_headless_surface`: asked on the
5070, no queue family presents to one); keeping the game's swapchain and adding a second one on the same surface
(`VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`).

## Semantics

- **Acquire**: an image that is neither acquired nor still being copied out; waits (up to the timeout) for a copy to
  finish when there is none. The game's semaphore and fence are signalled by an empty submit on the layer's own queue.
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

## Fallback

- Before the swapchain exists (the setting off, no second device, another placement, the second GPU cannot present
  to the surface, a format it has not got, any creation failure): the game's swapchain is created on its own device
  as always, and the second GPU is used the round-trip way. The log and the Status tab say why.
- While it runs (a present or upload fails, the second device is lost): the presenter stops, the game's next
  acquire/present returns `VK_ERROR_OUT_OF_DATE_KHR`, and the swapchain it then creates is a real one on its own
  device. The mode stays off for the session. No frame is shown black: the game recreates before it shows another.

## Settings and Status

- `second_gpu_present` (0/1, GUI: Model tab, "Second GPU shows the frame"), with `NEURAL_FORGE_OFFLOAD_GPU` and
  every pass at Placement "Finished frame".
- Status: "second GPU shows the frames: <GPU>; N shown, M skipped (P%), copy X ms" or why not.

## Tests (lavapipe)

Hand-off ring (newest wins, skipped counts), acquire/present image states, a full run with a real swapchain on a
second lavapipe device (headless surface): frames arrive unchanged, skip under a slow presenter, recreate, failure
falls back with `OUT_OF_DATE`.
