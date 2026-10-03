# Neural Forge documentation

Every document in this folder, one line each. New to the project: read
[ARCHITECTURE.md](ARCHITECTURE.md), then [LESSONS.md](LESSONS.md). The user-facing guide is the
top-level [README](../README.md).

Older documents are kept as they were written, with a dated note at the top where something in
them is no longer true. When a document and the code disagree, the code wins; when two documents
disagree, the more recent measurement wins.

## How it works

- [ARCHITECTURE.md](ARCHITECTURE.md): the processes, the shared-memory protocol (versions 1-11),
  both frame paths step by step, the helper's per-request work, and the timing of each stage.
- [PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md): the 2.0 design for running the model before
  DLSS Super Resolution, and the full record of its experiments (E1-E3, hand-off latency, 4K,
  robustness, frame generation).
- [EXTERNAL_MEMORY_HOST_DESIGN.md](EXTERNAL_MEMORY_HOST_DESIGN.md): zero-copy transport by
  importing shared memory as Vulkan memory (`VK_EXT_external_memory_host`), on both sides.
- [PROTOCOL_V3_DESIGN.md](PROTOCOL_V3_DESIGN.md): the second request/response slot, and why the
  model still runs one request at a time.
- [RENDER_TAP_DESIGN.md](RENDER_TAP_DESIGN.md): capturing from the image a game blits into its
  swapchain, and the rules for when that is allowed.
- [ASYNC_CAPTURE_DESIGN.md](ASYNC_CAPTURE_DESIGN.md): the two-slot non-blocking capture pipeline
  (the default present before 0.1.78; the synchronous present replaced it).

## Decisions and history

- [LESSONS.md](LESSONS.md): what was tried, what worked, what didn't, and why, from 0.1.0 to 2.0.0.
- [GHOSTING_PLAN.md](GHOSTING_PLAN.md): why the held answer ghosted, what upstream does instead,
  the model-resolution blit and the move of optical flow into the helper.
- [FRAMEGEN_SPIKE.md](FRAMEGEN_SPIKE.md): Smooth Motion and lsfg-vk below the layer, the Vulkan
  layer-order problem, and the 1.x frame generation measurements (superseded by 2.0).
- [DMABUF_TRANSPORT_DESIGN.md](DMABUF_TRANSPORT_DESIGN.md): why DMA-BUF sharing between the layer
  and the Wine helper is blocked in both directions.
- [NATIVE_NGX_HELPER_DESIGN.md](NATIVE_NGX_HELPER_DESIGN.md): why a native Linux NGX helper cannot
  run Feature 18.
- [OPENDLSS_REVIEW.md](OPENDLSS_REVIEW.md): Neural Forge compared stage by stage with OpenDLSS-NR's
  documented pipeline, the history reset rule, and the GTA script-mod finding.
- [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md): what was and was not carried over from DLSS5VKLayer
  0.3.1-1.
- [PHASE1.md](PHASE1.md): the namespace contract, installation contract and target ownership rules
  (its benchmark plan is historical).
- [history/development-before-neuralforge.md](history/development-before-neuralforge.md): the
  development log from before the rename (codename dlssnr), 2026-09-09 to 09-12.
- [history/handoff-2026-09-12-fps-freeze-regression.md](history/handoff-2026-09-12-fps-freeze-regression.md):
  the hand-off after the reverted fence-wait changes.

## Measurements

- [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md): every test on the RTX 5070 machine, dated,
  including the 2.0 baseline and the 2.0.0 results.
- [PRE_UPSCALER_PROBE.md](PRE_UPSCALER_PROBE.md): what DLSS looks like from a Vulkan layer in GTA V
  Enhanced (registered inputs, launches, submits, the launch command buffer's contents).
- [../CHANGELOG.md](../CHANGELOG.md): every release with its measured effect.

## Testing and tooling

- [RUNNING_AND_MEASURING.md](RUNNING_AND_MEASURING.md): building, deploying to a test machine, the
  unattended GTA benchmark, the pan reproducer and agreement metric, the diagnostic modes and
  probe, driving the helper without a game, reading the logs and `shmctl status`, validation
  layers, checking the picture by eye, and the release checklist.

## For contributors and agents

- [../CLAUDE.md](../CLAUDE.md) (identical to [../AGENTS.md](../AGENTS.md)): naming rules, build and
  test commands, runtime constraints, the pre-upscaler rules and the composition invariants.
- [../ATTRIBUTION.md](../ATTRIBUTION.md): what was taken from DLSS5VKLayer and others, function by
  function.
- [../THIRD_PARTY_CRATES.md](../THIRD_PARTY_CRATES.md): the statically linked Rust crates and their
  licences.
