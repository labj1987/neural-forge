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
# An installed copy's implicit manifest (~/.local/share/vulkan/implicit_layer.d) has the same
# layer name and wins over VK_LAYER_PATH, so on a machine with Neural Forge installed the
# smoke test would otherwise exercise the installed library, not this build. Point the
# per-user data dir at the scratch dir so only the manifest above is found.
export XDG_DATA_HOME="$SCRATCH/xdg-data"
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
if grep -q '\[probe-ngx\]' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the NGX probe logged without NEURAL_FORGE_PROBE_NGX" >&2
    exit 1
fi

# Second pass with the diagnostic NGX probe on (docs/PRE_UPSCALER_PROBE.md): the local ICD has
# no VK_NVX_* extensions, so this proves the probe's hooks install and stay harmless there.
export NEURAL_FORGE_PROBE_NGX=1
export NEURAL_FORGE_LOG="$SCRATCH/layer-probe.log"
echo
echo "==> again with NEURAL_FORGE_PROBE_NGX=1"
cargo run --example smoke -p neural-forge-layer
echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
if ! grep -q '\[probe-ngx\] on in pid' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: NEURAL_FORGE_PROBE_NGX=1 did not turn the probe on" >&2
    exit 1
fi

# Third pass with the pre-upscaler path in model mode (docs/PRE_UPSCALER_DESIGN.md, "Implementation
# (layer)"): the local ICD has no VK_NVX_* extensions and no DLSS, so this proves the tracking hooks
# install, nothing is held, and the probe's logging stays off.
unset NEURAL_FORGE_PROBE_NGX
export NEURAL_FORGE_PREUPSCALE=model
export NEURAL_FORGE_LOG="$SCRATCH/layer-preupscale.log"
echo
echo "==> again with NEURAL_FORGE_PREUPSCALE=model"
cargo run --example smoke -p neural-forge-layer
echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
if ! grep -q '\[preupscale\] mode model in pid' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: NEURAL_FORGE_PREUPSCALE=model did not turn the pre-upscaler path on" >&2
    exit 1
fi
if grep -q '\[probe-ngx\]' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the NGX probe logged with only NEURAL_FORGE_PREUPSCALE set" >&2
    exit 1
fi
if grep -q 'frame went to DLSS untouched\|resources for' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the pre-upscaler path held a submit on a device without DLSS" >&2
    exit 1
fi
