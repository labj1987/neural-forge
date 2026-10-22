#!/usr/bin/env bash
# Regenerates the layer framework's generated sources (third_party/vk-layer-for-rust: the vk_layer.h binding,
# layer_trait/generated.rs and global_simple_intercept/generated.rs) from the Vulkan headers they are built
# against, or with --check verifies that the committed files are what the generator gives (CI runs that).
# The headers are Vulkan-Headers at commit 3df77fb3 (1.3.251, the registry ash 0.37.3 matches), fetched
# once, SHA-256 checked, into tools/native/ (or NEURAL_FORGE_NATIVE_TOOLS). Needs libclang (bindgen; set
# LIBCLANG_PATH if it is not found) and the pinned toolchain's rustfmt.
#   bash scripts/vk-layer-codegen.sh [--check]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FW="$ROOT/third_party/vk-layer-for-rust"
TOOLS="${NEURAL_FORGE_NATIVE_TOOLS:-$ROOT/tools/native}"
URL="https://github.com/KhronosGroup/Vulkan-Headers/archive/3df77fb3e4e0961f7f01de9a9ae20dfdcfc4910a.tar.gz"
SHA="43427cd151bba1f64d4c857f467a879dfc8c28cc06fc4d4453b133cd96dcba54"
HEADERS="$TOOLS/Vulkan-Headers-vk-layer"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

if [ ! -e "$HEADERS/registry/vk.xml" ] || [ "$(cat "$HEADERS/.fetched-sha256" 2>/dev/null)" != "$SHA" ]; then
  mkdir -p "$TOOLS"
  stage="$(mktemp -d "$TOOLS/.fetch-vk-layer-headers.XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  echo "Vulkan-Headers (vk-layer): downloading $URL"
  curl -fsSL --retry 3 -o "$stage/archive.tar.gz" "$URL"
  got="$(sha256sum "$stage/archive.tar.gz" | cut -d' ' -f1)"
  [ "$got" = "$SHA" ] || { echo "Vulkan-Headers (vk-layer): SHA-256 mismatch: expected $SHA, got $got" >&2; exit 1; }
  tar -xzf "$stage/archive.tar.gz" -C "$stage"
  top=("$stage"/Vulkan-Headers-*)
  echo "$SHA" >"${top[0]}/.fetched-sha256"
  rm -rf "$HEADERS"
  mv "${top[0]}" "$HEADERS"
  rm -rf "$stage"
  trap - EXIT
fi

# The generator reads the headers from third_party/Vulkan-Headers inside the framework (git-ignored).
mkdir -p "$FW/third_party"
ln -sfn "$HEADERS" "$FW/third_party/Vulkan-Headers"
trap 'rm -f "$FW/third_party/Vulkan-Headers"; rmdir "$FW/third_party" 2>/dev/null || true' EXIT

if [ -z "${LIBCLANG_PATH:-}" ] && ! ldconfig -p 2>/dev/null | grep -q 'libclang\.so '; then
  for dir in $(ls -d /usr/lib/llvm-*/lib 2>/dev/null | sort -V -r); do
    if ls "$dir"/libclang*.so* >/dev/null 2>&1; then export LIBCLANG_PATH="$dir"; break; fi
  done
fi

cd "$FW"
RUST_LOG=error cargo run --quiet --locked --release -p vulkan-layer-scripts --bin codegen -- "$@"
