# OpenDLSS-NR, taken into Neural Forge

The network Neural Forge runs is the implementation from OpenDLSS-NR by maan, MIT:

- Upstream: https://github.com/maanHimself/OpenDLSS-NR
- Taken from commit `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` (2026-09-21, upstream's newest when it was
  taken in on 2026-10-22).
- License: MIT (`LICENSE`, `NOTICE` here; both ship in the AppImage).

Until 3.8.6, `scripts/fetch-native-tools.sh` downloaded that commit at build time, trimmed it to what is
listed under "Included" and applied ten patches. Since then this directory holds that result (the commit,
trimmed, with the ten changes below) and is this project's own code: upstream is no longer tracked.
`crates/native/build.rs` (neural-forge-native) builds from here. No compiled SPIR-V or PTX is committed:
both are generated at build time and embedded into the library.

## Changing it

- Edit the files here directly. This directory stays MIT, changes included: keep `LICENSE` and `NOTICE`.
- Record notable changes in `CHANGELOG.md`. Anything ported from a later upstream commit is credited by that
  commit.
- A new network needs `nr::Topology` in `src/nr_graph.*` extended, and `crates/supervisor/src/model_shape.rs`
  regenerated with `scripts/gen_model_shape.py`.

## Included

- `src/`: the host code: `kernels.cpp/.h`, `nr_graph.cpp/.h`, `nr_model.cpp/.h`, `vk_context.cpp/.h`,
  `numeric.h`. Upstream's `json.h` and `sha256.h` were removed in Neural Forge 3.9.4: the model directory is
  read and checked by `crates/native/src/model_file.rs`, and `nr::Model` takes the tensor records it read.
- `shaders/`: every GLSL compute shader (`*.comp`) and include (`*.glsl`).
- `scripts/ptx/`: the PTX generators (`ptxgen.py`, `swin.py`, the `*_e4m3.py` kernels, `test_fast_divmod.py`).
- `NOTICE`, `LICENSE`.

## Left out when it was taken in

- `src/main.cpp` (the `dlss5vk` command-line tool) and `src/verify.cpp`, which only that tool uses
  (`runVerify`). `src/reference.*` stays: `ref::siluTable()` feeds `Kernels::setSiluTable`.
- `demo/`, `ports/`, `docs/`, `third_party/` (Filament patch), and the PowerShell/Node build scripts
  (`scripts/*.ps1`, `scripts/*.mjs`, `scripts/*.py` outside `scripts/ptx/`). `build.rs` reproduces
  `scripts/build_shaders.ps1`: the same glslang flags and the same PTX generator invocations, except that the
  128-channel FFN kernels (`ffn_e4m3_C128_R4`, `_proj`) are generated with a 64-register cap (`build.rs` says why).

## Changes made before it was taken in

Each was a patch, applied in this order on top of the commit, until 3.8.6.

1. **Include cstring** (was `patches/0001-include-cstring.patch`): `src/vk_context.h` includes `<cstring>`. It uses `memcpy`
   and only compiled with MSVC, whose headers pull `<cstring>` in transitively; GCC's do not.
2. **Kernel chain namespace scope** (was `patches/0002-kernel-chain-namespace-scope.patch`): `Kernels::Chain` moves to namespace scope as
   `nr::KernelChain`, with `using Chain = KernelChain;` in the class. GCC rejects `Chain()` as a default
   argument inside the class while the nested struct with default member initializers is still incomplete
   (MSVC accepts it). Same type, same layout, no caller changes.
3. **Asset loader** (was `patches/0003-asset-loader.patch`): an asset hook in namespace `nr` (`src/vk_context.h/.cpp`):
   `using AssetLoader = bool (*)(const char* kind, const char* name, const uint8_t** data, size_t* size);`
   with `void setAssetLoader(AssetLoader)` and `AssetLoader assetLoader()`. When a loader is set, every
   SPIR-V read (`vk::Context::loadShaderModule`, kind `"spv"`) and every PTX probe and read in
   `src/kernels.cpp` (kind `"ptx"`) asks it, by file name without the extension, instead of opening
   `<shaderDir>/<name>.spv` or `$NEURAL_FORGE_NR_PTX_DIR/<name>.ptx` (upstream's `DLSS5VK_PTX_DIR`). Unset, behaviour is unchanged. Reason: a
   Vulkan layer is a single `.so` loaded into a game process and cannot rely on a kernel directory next to
   it; neural-forge-native embeds the kernels and installs a loader (`nf_native::installAssetLoader`).
4. **Layer adoption** (was `patches/0004-layer-adoption.patch`): a second adopting `vk::Context` constructor for use inside a Vulkan
   layer (`src/vk_context.h/.cpp`). It loads every function through the `vkGetInstanceProcAddr` it is given
   (the next layer's, `volkInitializeCustom`) instead of the loader, so the network's own calls never pass
   through the layer that issues them; falls back to the KHR/EXT spellings of `vkCmdPipelineBarrier2`,
   `vkGetBufferDeviceAddress` and `vkResetQueryPool` on a device created below 1.3; bounds every wait it
   makes (`endAndSubmit`, `waitIdle`) with a fence and a timeout, throwing on expiry instead of hanging; on an
   adopted device waits only for its own queue in the destructor, never `vkDeviceWaitIdle`; and calls an
   optional `InitDispatchable` hook on its queue and every command buffer it allocates, so a layer can set
   the loader's dispatch pointer on objects it got from below the loader; and, given a second queue family
   (the application's, where the recorded graph runs), creates every buffer `VK_SHARING_MODE_CONCURRENT` over
   its own family and that one, so the network can load on a queue of another family without ownership
   transfers. The original constructors are unchanged.
5. **Split scratch release** (was `patches/0005-split-scratch-release.patch`): `Kernels::releaseScratch()` (`src/kernels.h/.cpp`) destroys the split-K
   scratch, so the next recording sizes it again. It was sized once, from the first graph's split shapes, and
   `Kernels` outlives graph rebuilds: a later, larger graph threw "split-K partials exceed the scratch buffer".
   neural-forge-native calls it when it drops a graph, when nothing recorded holds the scratch's address.
6. **Constructor cleanup** (was `patches/0006-constructor-cleanup.patch`): constructors that throw release what they made. `nr::Model`'s constructor
   and both adopting `vk::Context` constructors (with `initCommon`, which they call) catch any exception, run the
   destructor's cleanup (`release()`, which the destructors now call too) and rethrow: no destructor runs for a
   constructor that throws, and on an adopted device every object left behind leaked on the application's device.
   A tensor's buffer is owned by the model before its upload, and `createBuffer` destroys the buffer (and any
   memory) when allocating, binding or mapping it fails.
7. **Stalled context** (was `patches/0007-stalled-context.patch`): a `vk::Context` whose bounded wait timed out (`endAndSubmit`, `waitIdle`)
   latches `stalled()`. The work it gave up on may still run on the GPU, so a stalled context takes no more work
   (`beginCommands`, `endAndSubmit` and `waitIdle` throw) and destroys nothing: `destroyBuffer`, `destroyPipeline`
   and `destroyCuda*` forget the handle, and the destructors of `Context`, `Model`, `Kernels` and `Graph` return at
   once, leaking what they own; `Context`'s also when its own last wait for its queue is the one that timed out. Before, a timeout threw and the owners' destructors then freed buffers the
   timed-out submit could still be using. neural-forge-native reports the flag (`nf_native_stalled`) and the
   layer stops using the network on that device.
8. **Compact aux** (was `patches/0008-compact-aux.patch`): every tensor record was uploaded whole as its `raw` buffer, but the kernels read only
   its scale vectors through it (the per-channel f16 skip and blend scales, the per-head f32 attention scales); its
   matrices and attention priors are read from the re-laid buffers that `fp8Matrix`, `fp8MatrixPermuted`, `f16Matrix` and
   `relativeBias` build from the host copy. So the packed weights were on the GPU twice. Those four now mark the record
   bytes they read, and `Model::compactAux()` replaces each `raw` with only the unmarked bytes, every run of them kept at
   its offset's alignment modulo 16. `Tensor::auxOffset` maps a record offset to the compacted buffer; every kernel entry
   point that reads `raw` (`gemmFp8`, `fusedBlock32`, `expertFfnPtx`, `postBlend`, `upsampleResidual`, `qkvAttention`,
   `windowNormalize`, `globalNormalize`, `globalNormalizePtx`, `globalAttention`) passes its offsets through it, and it
   throws for a read that overlaps a matrix's bytes. Until `compactAux` runs the mapping is the identity.
   neural-forge-native calls it after the warm-up run of a network's first build, when every matrix exists and nothing
   recorded with the whole records can still run. Measured: the records' 140.8 MiB on the GPU become 98 KiB, and the
   network's output is bit-identical (every size the layer builds, chained and not, the GLSL routes too).
9. **Small network topology** (was `patches/0009-small-network-topology.patch`): the graph's stage boundaries were hard-coded for the 71-block
   network. `nr::Topology` holds them (the last 32-channel encoder block, the first and last block of each fused
   encoder and decoder stage, the split-512 stages, the ViT range, the 512 decoder's transition and the post block)
   and `Topology::forBlockCount` gives the full network's (71 blocks) or the small one's (31: `WEIGHTS_CC_SMALL_FP8`
   in build 310.9.16, the same block types with fewer per stage); `Graph` takes it from `Model::blockCount()` and
   refuses any other count, as the removed `kBlockCount` check did. Every block number in `record` and `encodeVit`
   comes from it, as do the boundary capture names and `referenceBoundaryNames` (parity fixtures only; a member
   now, since the list depends on the network). The full network's output is unchanged: bit-exact against NGX's
   answer on fixture d1w (docs/NATIVE_BACKEND.md, 0.6).
10. **External descriptor pool** (was `patches/0010-external-descriptor-pool.patch`): `vk::Context::useDescriptorPool(pool)` (`src/vk_context.h`)
   makes the context allocate its descriptor sets from a pool of the caller's until the next call, and returns
   the pool it used before. The context has two pools of its own, reset in rotation, so the sets of a graph
   recorded twice in a row would be freed while the first recording may still run. neural-forge-native records
   the graph straight into the game's command buffer for the hold inside DLSS's buffer
   (`nf_native_record_graph_at`), several of which can be recorded before any runs: each recording takes its sets
   from a pool of the layer's hold slot, reset only when that slot records again.
