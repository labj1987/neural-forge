#!/usr/bin/env bash
# Downloads the pinned tools the native backend (crates/native) builds with into tools/native/ (git-ignored):
#   glslang 16.6.0 (Linux x86_64 release binary), Vulkan-Headers v1.4.363, volk vulkan-sdk-1.4.357.0.
# Every archive is checked against the SHA-256 pinned below before it is unpacked. Re-running skips a tool whose
# unpacked copy carries the stamp of the same pinned archive, so it is safe to run on every build.
#   bash scripts/fetch-native-tools.sh            (NEURAL_FORGE_NATIVE_TOOLS=<dir> to unpack elsewhere)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${NEURAL_FORGE_NATIVE_TOOLS:-$ROOT/tools/native}"
mkdir -p "$TOOLS"

# name | url | sha256 | file that must exist after unpacking (relative to tools/native/<name>)
PINS=(
  "glslang|https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz|2c34071f56ecf39d16233294d3739cf43f981fd39080f2fd9dccb782101191ce|bin/glslang"
  "Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.363.tar.gz|cbaf687d3c59b9666fe080e7f8396b6b4f4d344a768ee6b337c59852f4526b68|include/vulkan/vulkan_core.h"
  "volk|https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.357.0.tar.gz|6400c7b23e24d17e4f04bac49b55b06c4e87677d33398e90344743ec73560ca6|volk.c"
)

for pin in "${PINS[@]}"; do
  IFS='|' read -r name url sha check <<<"$pin"
  dest="$TOOLS/$name"
  if [ -e "$dest/$check" ] && [ "$(cat "$dest/.fetched-sha256" 2>/dev/null)" = "$sha" ]; then
    echo "$name: up to date ($dest)"
    continue
  fi
  stage="$(mktemp -d "$TOOLS/.fetch-$name.XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  echo "$name: downloading $url"
  curl -fsSL --retry 3 -o "$stage/archive.tar.gz" "$url"
  got="$(sha256sum "$stage/archive.tar.gz" | cut -d' ' -f1)"
  if [ "$got" != "$sha" ]; then
    echo "$name: SHA-256 mismatch: expected $sha, got $got" >&2
    exit 1
  fi
  mkdir "$stage/x"
  tar -xzf "$stage/archive.tar.gz" -C "$stage/x"
  # GitHub tag tarballs have one top-level directory; the glslang release archive does not.
  top="$stage/x"
  if [ ! -e "$top/$check" ]; then
    entries=("$top"/*)
    [ "${#entries[@]}" -eq 1 ] && top="${entries[0]}"
  fi
  [ -e "$top/$check" ] || { echo "$name: $check not found in the archive" >&2; exit 1; }
  echo "$sha" >"$top/.fetched-sha256"
  rm -rf "$dest"
  mv "$top" "$dest"
  rm -rf "$stage"
  trap - EXIT
  echo "$name: verified and unpacked into $dest"
done

"$TOOLS/glslang/bin/glslang" --version | head -n 1
grep -m1 '#define VK_HEADER_VERSION ' "$TOOLS/Vulkan-Headers/include/vulkan/vulkan_core.h"
echo "native tools ready in $TOOLS"
