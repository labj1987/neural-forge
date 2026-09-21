#!/usr/bin/env bash
# Builds the layer, points a scratch Vulkan layer manifest at it, and runs
# crates/layer/examples/smoke.rs through the real system Vulkan loader with the layer
# enabled. See that file's doc comment for what this actually proves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo build -p neural-forge-layer

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

SO_PATH="$(pwd)/target/debug/libneural_forge_layer.so"
sed "s#\./libneural_forge_layer\.so#$SO_PATH#" data/neural_forge_layer.json \
    > "$SCRATCH/neural_forge_layer.json"

# VK_LAYER_PATH only makes the loader consider a manifest for *explicit* enabling --
# it does not add it to the implicit_layer.d search path, so an implicit-type layer
# found this way still needs to be named explicitly to actually get inserted into the
# call chain. That's a test-harness-only difference: a real install drops this manifest
# into an actual implicit_layer.d directory instead, where enable_environment alone is
# enough (see data/neural_forge_layer.json and build-appimage.sh once that exists).
export VK_LAYER_PATH="$SCRATCH"
export VK_INSTANCE_LAYERS=VK_LAYER_neuralforge_neural
export NEURAL_FORGE_ENABLE=1
export NEURAL_FORGE_UID="smoketest-$$"
export NEURAL_FORGE_LOG="$SCRATCH/layer.log"

echo "==> manifest: $SCRATCH/neural_forge_layer.json"
echo "==> layer library: $SO_PATH"
echo

cargo run --example smoke -p neural-forge-layer

echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
