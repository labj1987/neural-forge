// Stub of the native backend's C API: enough to link the vendored code and check the embedded assets.
#include "nf_native.h"

#include "nf_assets.h"

extern "C" uint32_t nf_native_abi_version(void) { return 1; }

extern "C" uint32_t nf_native_asset_count(const char* kind) { return nf_native::assetCount(kind); }
