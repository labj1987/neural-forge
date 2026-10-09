# Neural Forge documentation

Every document in this folder, one line each. New to the project: read
[ARCHITECTURE.md](ARCHITECTURE.md), then [LESSONS.md](LESSONS.md). The user-facing guide is the
top-level [README](../README.md).

Older documents are kept as they were written, with a dated note at the top where something in
them is no longer true. Up to 2.0.10 the model ran in a Windows helper under Wine through NVIDIA's
NGX runtime; 3.0 removed the helper, so what older documents say about it, the runners, the
Wine prefix or NGX is history. When a document and the code disagree, the code wins; when two documents
disagree, the more recent measurement wins.

## How it works

- [ARCHITECTURE.md](ARCHITECTURE.md): the layer, the shared-memory channel (versions 1-15), both
  frame paths step by step, the network the layer runs, and the timing of each stage.
- [NATIVE_BACKEND.md](NATIVE_BACKEND.md): how the network moved into the layer for 3.0 (OpenDLSS-NR,
  the model directory and its extractor, the device setup, the hold inside DLSS's buffer), and every
  measurement against NVIDIA's runtime through the 2.x helper.
- [PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md): the 2.0 design for running the model before
  DLSS Super Resolution, and the full record of its experiments (E1-E3, hand-off latency, 4K,
  robustness, frame generation).
- [EXTERNAL_MEMORY_HOST_DESIGN.md](EXTERNAL_MEMORY_HOST_DESIGN.md): zero-copy transport by
  importing shared memory as Vulkan memory (`VK_EXT_external_memory_host`), on both sides (written
  for the 2.x helper; the after-the-upscaler path's model server imports it the same way).
- [PROTOCOL_V3_DESIGN.md](PROTOCOL_V3_DESIGN.md): the second request/response slot, and why the
  model still runs one request at a time.
- [RENDER_TAP_DESIGN.md](RENDER_TAP_DESIGN.md): capturing from the image a game blits into its
  swapchain, and the rules for when that is allowed.
- [ASYNC_CAPTURE_DESIGN.md](ASYNC_CAPTURE_DESIGN.md): the two-slot non-blocking capture pipeline
  (the default present before 0.1.78; the synchronous present replaced it).

## Decisions and history

- [DLSSNR_EXPERIMENTS.md](DLSSNR_EXPERIMENTS.md): API-based preset comparison tooling and the
  validation gates for real depth and game motion-vector inputs.

- [LESSONS.md](LESSONS.md): what was tried, what worked, what didn't, and why, from 0.1.0 to 2.0.0.
- [GHOSTING_PLAN.md](GHOSTING_PLAN.md): why the held answer ghosted, what upstream does instead,
  the model-resolution blit and the move of optical flow into the helper.
- [FRAMEGEN_SPIKE.md](FRAMEGEN_SPIKE.md): Smooth Motion and lsfg-vk below the layer, the Vulkan
  layer-order problem, and the 1.x frame generation measurements (superseded by 2.0).
- [DMABUF_TRANSPORT_DESIGN.md](DMABUF_TRANSPORT_DESIGN.md): why DMA-BUF sharing between the layer
  and the 2.x Wine helper was blocked in both directions.
- [NATIVE_NGX_HELPER_DESIGN.md](NATIVE_NGX_HELPER_DESIGN.md): why a native Linux NGX helper cannot
  run Feature 18 (which is why 3.0 runs an open implementation of the network instead).
- [OPENDLSS_REVIEW.md](OPENDLSS_REVIEW.md): Neural Forge compared stage by stage with OpenDLSS-NR's
  documented pipeline, the history reset rule, and the GTA script-mod finding.
- [DLSS_KERNEL_CATALOGUE.md](DLSS_KERNEL_CATALOGUE.md): DLSS's kernel names and input-kernel parameter
  layouts for every DLSS version on the test machine (2.2.11 to 310.9.1), which kernels are SR, FG and
  Ray Reconstruction, and what the layer could stop guessing.
- [DLL_FINDINGS_AUDIT.md](DLL_FINDINGS_AUDIT.md): 2.0.10 and the native backend checked against a
  static analysis of `nvngx_dlssnr.dll` 310.8, item by item, with the recommended changes and the
  runtime tests that would settle what the code cannot.
- [DLSSNR_PARAMETERS.md](DLSSNR_PARAMETERS.md): which NGX parameters the Neural Rendering feature
  reads, the five the helper never sets, the helper writes nothing reads, and when the history
  reset is set, against the public SDK and the open-source consumers.
- [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md): what was and was not carried over from DLSS5VKLayer
  0.3.1-1.
- [PHASE1.md](PHASE1.md): the namespace contract, installation contract and target ownership rules.
- [history/phase1-benchmark-plan.md](history/phase1-benchmark-plan.md): the 2.x-era Phase 1 benchmark
  plan and later phases, moved out of PHASE1.md.
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
  probe, reading the logs and `shmctl status`, validation
  layers, checking the picture by eye, the release checklist, and the reverse-engineering toolkit
  installed on the test machine.
- [RE_TOOLKIT.md](RE_TOOLKIT.md): agent-driven reverse-engineering tools surveyed (what each attaches
  to, headless or GUI, licence) and why each was or was not installed.

## For contributors and agents

- [../AGENTS.md](../AGENTS.md): naming rules, build and
  test commands, runtime constraints, the pre-upscaler rules and the composition invariants.
- [../ATTRIBUTION.md](../ATTRIBUTION.md): what was taken from DLSS5VKLayer and others, function by
  function.
- [../THIRD_PARTY_CRATES.md](../THIRD_PARTY_CRATES.md): the statically linked Rust crates and their
  licences.
