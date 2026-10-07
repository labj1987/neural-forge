# OpenDLSS-NR (vendored)

- Upstream: https://github.com/maanHimself/OpenDLSS-NR
- Commit: `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`
- License: MIT (`LICENSE`, `NOTICE`)

Built by `crates/native/build.rs` (neural-forge-native) with the toolchain pinned in
`scripts/fetch-native-tools.sh`. No compiled SPIR-V or PTX is committed: both are generated at build time
and embedded into the library.

## Included

- `src/`: the host code: `kernels.cpp/.h`, `nr_graph.cpp/.h`, `nr_model.cpp/.h`, `vk_context.cpp/.h`,
  `json.h`, `numeric.h`, `sha256.h`.
- `shaders/`: every GLSL compute shader (`*.comp`) and include (`*.glsl`).
- `scripts/ptx/`: the PTX generators (`ptxgen.py`, `swin.py`, the `*_e4m3.py` kernels, `test_fast_divmod.py`).
- `NOTICE`, `LICENSE`.

## Not included

- `src/main.cpp` (the `dlss5vk` command-line tool), and `src/verify.cpp`, `src/reference.cpp`,
  `src/reference.h`, which only that tool uses (`runVerify` and the CPU reference).
- `demo/`, `ports/`, `docs/`, `third_party/` (Filament patch), and the PowerShell/Node build scripts
  (`scripts/*.ps1`, `scripts/*.mjs`, `scripts/*.py` outside `scripts/ptx/`). `build.rs` reproduces
  `scripts/build_shaders.ps1`: the same glslang flags and the same PTX generator invocations.

## Local changes

Each one is also a patch in `patches/`, relative to this directory and applied in order
(`git apply --directory=third_party/opendlss-nr patches/NNNN-*.patch` on a fresh copy of upstream).

1. `patches/0001-include-cstring.patch`: `src/vk_context.h` includes `<cstring>`. It uses `memcpy`
   and only compiled with MSVC, whose headers pull `<cstring>` in transitively; GCC's do not.
2. `patches/0002-kernel-chain-namespace-scope.patch`: `Kernels::Chain` moves to namespace scope as
   `nr::KernelChain`, with `using Chain = KernelChain;` in the class. GCC rejects `Chain()` as a default
   argument inside the class while the nested struct with default member initializers is still incomplete
   (MSVC accepts it). Same type, same layout, no caller changes.
3. `patches/0003-asset-loader.patch`: an asset hook in namespace `nr` (`src/vk_context.h/.cpp`):
   `using AssetLoader = bool (*)(const char* kind, const char* name, const uint8_t** data, size_t* size);`
   with `void setAssetLoader(AssetLoader)` and `AssetLoader assetLoader()`. When a loader is set, every
   SPIR-V read (`vk::Context::loadShaderModule`, kind `"spv"`) and every PTX probe and read in
   `src/kernels.cpp` (kind `"ptx"`) asks it, by file name without the extension, instead of opening
   `<shaderDir>/<name>.spv` or `$DLSS5VK_PTX_DIR/<name>.ptx`. Unset, behaviour is unchanged. Reason: a
   Vulkan layer is a single `.so` loaded into a game process and cannot rely on a kernel directory next to
   it; neural-forge-native embeds the kernels and installs a loader (`nf_native::installAssetLoader`).
