// C API of the native backend (crates/native). Rust declares the same functions in src/lib.rs.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version of this C API; bumped whenever a signature or a contract below changes.
uint32_t nf_native_abi_version(void);

// How many embedded assets of a kind ("spv" or "ptx") the library carries; 0 for any other kind.
uint32_t nf_native_asset_count(const char* kind);

#ifdef __cplusplus
}
#endif
