// The SPIR-V and PTX kernels build.rs embeds. The definitions are generated (OUT_DIR/nf_assets.cpp): every
// third_party/opendlss-nr/shaders/*.comp as kind "spv" and every PTX generator output as kind "ptx", each
// named by its file name without the extension, as OpenDLSS-NR's nr::AssetLoader expects.
#pragma once

#include <cstddef>
#include <cstdint>

namespace nf_native {

// nr::AssetLoader-compatible lookup: false when there is no such asset. The bytes live for the process.
bool findAsset(const char* kind, const char* name, const uint8_t** data, size_t* size);

// Number of embedded assets of a kind ("spv" or "ptx").
uint32_t assetCount(const char* kind);

// Routes every shader and PTX read of the vendored library through findAsset (nr::setAssetLoader). Call it
// before the first nr::Kernels is constructed.
void installAssetLoader();

}  // namespace nf_native
