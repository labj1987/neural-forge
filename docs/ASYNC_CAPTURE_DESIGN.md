# Asynchronous host capture design

The current GTA render tap is correct but waits for its capture fence inside the
present hook before copying staging bytes to shared memory. The repeated live result
of about 7.7 layer frames per second identifies that wait as the next bottleneck.

## Bounded two-slot pipeline

Each slot owns a command buffer, host-coherent staging buffer, fence, dimensions, and
source-layout restoration command. At most one slot is submitted and one is ready for
shared-memory upload. The layer must never reuse or resize a slot until its fence has
signaled.

1. At present, poll the older slot's fence without waiting. If it is signaled, copy
   its mapped bytes into SHM and begin the helper request only if no request is already
   outstanding.
2. If a free slot exists and no capture is pending, record the legal
   `GENERAL -> TRANSFER_SRC_OPTIMAL -> GENERAL` source copy and submit it. Return to
   the real present immediately.
3. While the helper works, present the untouched game output. When its answer arrives,
   use the existing output write-back path and its completion semaphore.
4. If both slots are busy, the source mapping is absent, dimensions differ, a fence
   reports an error, or the helper has a request outstanding, skip capture and present
   unchanged. No queue wait, source-layout guess, or slot reuse is allowed.

## Validation

First exercise this with `vkcube`, Khronos synchronization validation, and an
artificially delayed helper. Then validate the GTA render tap at 2560x1440 with
`NEURAL_FORGE_DMABUF=0`, full helper/model resolution, explicit GTA ownership, and
launcher pass-through. Compare layer throughput only after the same scene and timing
route are repeatable.
