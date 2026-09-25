# NeuralForge GTA render-tap design

## Current state (1.0.0)

This document records the original GTA investigation. Two things have changed since:

- GTA's swapchain has been admitted since 0.1.73: the earlier refusal came from DXVK's
  extension chain on the create info, not from missing usage, and the surface does
  advertise `TRANSFER_SRC`. GTA renders straight into the swapchain image, so it never used
  the tap. Admitted swapchains are always captured from their own image.
- The tap is used only on pass-through swapchains, and only when the observed copy or blit
  is a 1:1 write from the origin at the swapchain's extent with a matching format (a raw copy
  also accepts the sRGB/UNORM twin). The layer records source extents and formats from
  `vkCreateImage`. A `GENERAL` source is read in `GENERAL` with no layout transition. Any
  other source presents untouched, as rule 5 below requires.

GTA V Enhanced on the target system creates a 2560x1440 B8G8R8A8 swapchain with
`TRANSFER_DST | COLOR_ATTACHMENT`. The surface does not advertise `TRANSFER_SRC`,
so NeuralForge must not add it or read a swapchain image as a capture source.

The passive layer probe observed the game's own blits from two images in
`TRANSFER_SRC_OPTIMAL` into those swapchain images in `TRANSFER_DST_OPTIMAL`. Those
source images are the only currently demonstrated legal capture candidates.

## Required implementation rules

1. Observe the game's copy/blit into a known primary swapchain image and retain only
   the source image, source layout, destination swapchain image, and extent for the
   current presented frame.
2. At present, submit NeuralForge work on the same queue only after the application's
   present wait semaphores have completed. The source must not be read before the
   game blit finishes.
3. Copy the source image while it is in `TRANSFER_SRC_OPTIMAL` into private host-read
   memory. Do not transition or otherwise mutate the game's source image.
4. When an answer is available, write it to the swapchain image using its supported
   `TRANSFER_DST` usage, restore `PRESENT_SRC_KHR`, and add NeuralForge's completion
   semaphore to the real present call.
5. If the source is unavailable, changes size/format, lacks the required layout, or
   any synchronization condition cannot be proven, pass through the exact original
   present. No fallback may guess image ownership or layout.

## Validation gate

Validate the feature first with Khronos synchronization validation at 2560x1440,
`NEURAL_FORGE_DMABUF=0`, the full helper and model-resolution baseline. Confirm that
Rockstar, Social Club, Xalia, Explorer, and overlays remain pass-through and that
only `GTA5_Enhanced.exe` acquires the NeuralForge lease. Only then collect a matched
upstream/NeuralForge layer-throughput sample; it is still not a substitute for a
repeatable in-game FPS and 1%-low benchmark route.
