#!/usr/bin/env bash
# Downloads the pinned sources and tools the native backend (crates/native) builds with into tools/native/
# (git-ignored):
#   glslang 16.6.0 (Linux x86_64 release binary), Vulkan-Headers v1.4.365, volk vulkan-sdk-1.4.363.0, and
#   OpenDLSS-NR at commit 9d08f41 with this project's patches (third_party/opendlss-nr/patches/) applied.
# Every archive is checked against the SHA-256 pinned below before it is unpacked; every patch must apply
# cleanly. Re-running skips a tool whose unpacked copy carries the stamp of the same pinned archive (and, for
# OpenDLSS-NR, the same patches), so it is safe to run on every build, and a filled tools/native/ builds
# offline. scripts/check_pins.py reads the pins below.
#   bash scripts/fetch-native-tools.sh            (NEURAL_FORGE_NATIVE_TOOLS=<dir> to unpack elsewhere)
#   bash scripts/fetch-native-tools.sh --self-test
set -euo pipefail

# True when patch's output reports a hunk that applied only with an offset or fuzz. Only its hunk lines
# are read: a patched file's path (`patching file src/offset_table.cpp`) says nothing about how it applied.
drifted() {
  grep -qE '^Hunk #[0-9]+ succeeded at [0-9]+ (with fuzz|\(offset)' <<<"$1"
}

if [ "${1:-}" = "--self-test" ]; then
  drifted 'patching file src/kernels.cpp' && { echo "self-test: a clean patch counted as drifted" >&2; exit 1; }
  drifted $'patching file src/offset_table.cpp\npatching file shaders/fuzz.comp' &&
    { echo "self-test: a path with offset or fuzz in it counted as drifted" >&2; exit 1; }
  drifted $'patching file src/a.cpp\nHunk #2 succeeded at 120 (offset 3 lines).' ||
    { echo "self-test: an offset hunk passed" >&2; exit 1; }
  drifted $'patching file src/a.cpp\nHunk #1 succeeded at 7 with fuzz 1.' ||
    { echo "self-test: a fuzzed hunk passed" >&2; exit 1; }
  drifted $'patching file src/a.cpp\nHunk #3 succeeded at 40 with fuzz 2 (offset -1 lines).' ||
    { echo "self-test: a fuzzed and offset hunk passed" >&2; exit 1; }
  echo "fetch-native-tools self-test ok"
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${NEURAL_FORGE_NATIVE_TOOLS:-$ROOT/tools/native}"
PATCHES="$ROOT/third_party/opendlss-nr/patches"
mkdir -p "$TOOLS"

# name | url | sha256 | file that must exist after unpacking (relative to tools/native/<name>)
PINS=(
  "glslang|https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz|2c34071f56ecf39d16233294d3739cf43f981fd39080f2fd9dccb782101191ce|bin/glslang"
  "Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.365.tar.gz|ed832b3292cce324ebba6522fd2d3004bf1b797fa49bfbcad0b302ff23d56abc|include/vulkan/vulkan_core.h"
  "volk|https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz|1547d8d74395d4048fb3f4a6313da56db626e89b55db238d0d7e8944c8a645f3|volk.c"
  "OpenDLSS-NR|https://github.com/maanHimself/OpenDLSS-NR/archive/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1.tar.gz|1301bfc9ee8e3868e9df579d7c9bfeb537a82b4be97c164e0c7925661d787125|src/vk_context.cpp"
)

# What the native backend uses of OpenDLSS-NR (third_party/opendlss-nr/UPSTREAM.md): everything else in the
# archive is removed before the patches are applied, so the tree is exactly what is built and shipped.
OPENDLSS_KEEP=(src shaders scripts/ptx LICENSE NOTICE)
OPENDLSS_DROP=(src/main.cpp src/verify.cpp)

# The stamp a tool's unpacked copy carries: the archive's SHA-256, and for OpenDLSS-NR the patches' too.
stamp_for() {
  local name="$1" sha="$2"
  if [ "$name" = "OpenDLSS-NR" ]; then
    printf '%s\n' "$sha"
    (cd "$PATCHES" && sha256sum -- *.patch)
  else
    printf '%s\n' "$sha"
  fi
}

# Trims an unpacked OpenDLSS-NR to what is used and applies the patches in order; fails on any that does
# not apply cleanly (a hunk with an offset or fuzz is a failure too).
prepare_opendlss() {
  local top="$1" kept
  kept="$(mktemp -d "$TOOLS/.keep.XXXXXX")"
  for path in "${OPENDLSS_KEEP[@]}"; do
    mkdir -p "$kept/$(dirname "$path")"
    mv "$top/$path" "$kept/$path"
  done
  rm -rf "$top"
  mv "$kept" "$top"
  for path in "${OPENDLSS_DROP[@]}"; do
    rm -f "$top/$path"
  done
  for patch in "$PATCHES"/*.patch; do
    out="$(cd "$top" && patch -p1 --forward --batch --fuzz=0 --no-backup-if-mismatch < "$patch" 2>&1)" || {
      echo "OpenDLSS-NR: $(basename "$patch") does not apply:" >&2
      echo "$out" >&2
      exit 1
    }
    if drifted "$out"; then
      echo "OpenDLSS-NR: $(basename "$patch") applies only with an offset or fuzz:" >&2
      echo "$out" >&2
      exit 1
    fi
    echo "OpenDLSS-NR: applied $(basename "$patch")"
  done
}

for pin in "${PINS[@]}"; do
  IFS='|' read -r name url sha check <<<"$pin"
  dest="$TOOLS/$name"
  stamp="$(stamp_for "$name" "$sha")"
  if [ -e "$dest/$check" ] && [ "$(cat "$dest/.fetched-sha256" 2>/dev/null)" = "$stamp" ]; then
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
  if [ "$name" = "OpenDLSS-NR" ]; then
    prepare_opendlss "$top"
  fi
  echo "$stamp" >"$top/.fetched-sha256"
  rm -rf "$dest"
  mv "$top" "$dest"
  rm -rf "$stage"
  trap - EXIT
  echo "$name: verified and unpacked into $dest"
done

"$TOOLS/glslang/bin/glslang" --version | head -n 1
grep -m1 '#define VK_HEADER_VERSION ' "$TOOLS/Vulkan-Headers/include/vulkan/vulkan_core.h"
echo "native tools ready in $TOOLS"
