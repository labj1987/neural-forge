# Phase 1 benchmark plan (historical)

> Moved out of [../PHASE1.md](../PHASE1.md), which keeps the namespace, installation and
> target-ownership contract. Everything below is from the 2.x era (Windows helper, Wine prefix,
> upstream co-installed at 0.3.0-1) and is not current. Benchmarks now run unattended with
> `scripts/gta-bench.sh` ([../RUNNING_AND_MEASURING.md](../RUNNING_AND_MEASURING.md)); the matched
> comparator is 2.0.10 against the native backend ([../NATIVE_BACKEND.md](../NATIVE_BACKEND.md), Phase 3).

## Preserved baseline and benchmark gate

Upstream remains installed at 0.3.0-1. Its known-good GTA Enhanced launch option is:

```text
VKLayer_DLSS5=1 DLSSNR_DMABUF=0 %command%
```

Do not change this saved baseline. A separate Neural Forge test launch uses:

```text
NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_TARGET_EXE=GTA5_Enhanced.exe %command%
```

Neural Forge currently uses host SHM transport; `NEURAL_FORGE_DMABUF` is reserved and
has no zero-copy implementation -- and, per `DMABUF_TRANSPORT_DESIGN.md`, real
hardware evidence this session says the underlying mechanism the current Wine-hosted
helper would need is blocked at the driver/Wine level, not just unbuilt. DMA-BUF
remains experimental and requires a separate explicit retest because upstream hung at
4K. Never enable both activation flags for
one benchmark process. Co-installation does not mean double injection is useful.

Keep helper enabled, passes=1, model_resolution=1. Since 0.1.98 new mappings default
to motion on at the Fast quality (`mvec_enabled=1`, `mvec_quality=0` in this Rust
protocol): measured at about 2-3% of the frame rate in GTA V (see CHANGELOG.md's 0.1.98
entry). Record the motion setting with every benchmark, since the Phase 1 baseline was
taken with motion off. No model-resolution or helper enablement changes were made in
Phase 1.
Record RTX 5070 / driver 615.71.09, 2560x1440 at 288 Hz, GNOME scale 100%,
Steam AppID 3240220, the exact Proton build, game build, and DLL hashes.

1. Before testing, save config/launch options and record actual helper settings.
   Confirm only the intended layer is mapped into the game and each app's helper
   uses its own prefix, runtime path and advancing counters. Try Explorer, Xalia,
   Rockstar and Social Club while GTA runs; dimensions and owner must stay stable.
2. Validate Vulkan operations using validation layers in a separate smoke run, then
   test the same saved GTA scene/route. Do not mix validation overhead into timing.
3. Measure native baseline, upstream host transport, and Neural Forge host transport
   with identical settings and visual mode. Warm up, alternate order, repeat at least
   three 60-second captures. Record average/1% low FPS, frame-time percentiles,
   GPU utilization, VRAM, helper frames and matched screenshots. Stop on corruption,
   hangs or validation errors; do not compensate by lowering model resolution.
4. No performance improvement is claimed by Phase 1. The old roughly 9 FPS result
   remains unresolved until this matched comparison is run on the target machine.

## Later phases — prepared, not implemented

First add full pipeline instrumentation: capture GPU time, host readback/copy,
request wait/age, helper upload/evaluation/download, composition/writeback, present,
queue waits, allocations and feature rebuilds. Use correlated frame IDs and bounded
logging; quantify instrumentation overhead. Fix measured parity/correctness problems
before adding quality/performance features. Do not reapply reverted fence changes.

Then evaluate matched residual/native+edit with truly matched inputs and HDR/color
handling; independent X/Y neural scaling; delayed feature retirement with GPU-completion
proof and bounded caches; in-place resolve/VRAM reuse with aliasing validation;
depth-aware silhouettes only once reliable game depth exists; and an opt-in adaptive
FPS governor with hysteresis, rate limits, min/max bounds and stable frame pacing.
Each needs image-quality and performance acceptance tests before deployment.

[DLSSNR-Cost-Scaler releases](https://github.com/xenmods/DLSSNR-Cost-Scaler/releases)
are behavioral references: v1.0.4 discusses GPU/in-place optimizations; v1.0.5 adds
asymmetric scaling and depth-aware protection. Review exact tagged source and license
before any reuse. Governor and retirement details require their own design review.
Keep the clean Rust implementation; no code was copied from those projects here.
