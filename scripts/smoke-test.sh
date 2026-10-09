#!/usr/bin/env bash
# Builds the layer, points a scratch Vulkan layer manifest at it, and runs
# crates/layer/examples/smoke.rs through the real system Vulkan loader with the layer
# enabled. See that file's doc comment for what this actually proves.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# NEURAL_FORGE_SMOKE_SO names a layer library to test instead of building the debug one (the
# release workflow points it at the library the AppImage installed).
SO_PATH="${NEURAL_FORGE_SMOKE_SO:-}"
if [[ -z "$SO_PATH" ]]; then
    cargo build --locked -p neural-forge-layer
    SO_PATH="$(pwd)/target/debug/libneural_forge_layer.so"
fi
test -f "$SO_PATH"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

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

cargo run --locked --example smoke -p neural-forge-layer

echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
if grep -q '\[probe-ngx\]' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the NGX probe logged without NEURAL_FORGE_PROBE_NGX" >&2
    exit 1
fi
# The default is the pre-upscaler path in model mode (docs/PRE_UPSCALER_DESIGN.md): the local ICD
# has no VK_NVX_* extensions and no DLSS, so it must load cleanly, give the device no tracking, and
# hold nothing.
if ! grep -q '\[preupscale\] mode model (default) in pid' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: with NEURAL_FORGE_PREUPSCALE unset the pre-upscaler path is not the default" >&2
    exit 1
fi
if ! grep -q 'no VK_NVX_image_view_handle, so no DLSS input to find' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: a device without NVX did not stay on the post-upscaler path" >&2
    exit 1
fi
if grep -q 'frame went to DLSS untouched\|resources for\|HDR encode' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the default held a submit on a device without DLSS" >&2
    exit 1
fi

# Pass with NEURAL_FORGE_PREUPSCALE=off: the A/B and rollback switch, 1.1.0's behaviour. Nothing of
# the pre-upscaler path may run or log.
export NEURAL_FORGE_PREUPSCALE=off
export NEURAL_FORGE_LOG="$SCRATCH/layer-off.log"
echo
echo "==> again with NEURAL_FORGE_PREUPSCALE=off"
cargo run --locked --example smoke -p neural-forge-layer
echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
if ! grep -q '\[layer\] hooked device' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the layer did not run with NEURAL_FORGE_PREUPSCALE=off" >&2
    exit 1
fi
if grep -q '\[preupscale\]' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: the pre-upscaler path ran with NEURAL_FORGE_PREUPSCALE=off" >&2
    exit 1
fi
unset NEURAL_FORGE_PREUPSCALE

# Second pass with the diagnostic NGX probe on (docs/PRE_UPSCALER_PROBE.md): the local ICD has
# no VK_NVX_* extensions, so this proves the probe's hooks install and stay harmless there.
export NEURAL_FORGE_PROBE_NGX=1
export NEURAL_FORGE_LOG="$SCRATCH/layer-probe.log"
echo
echo "==> again with NEURAL_FORGE_PROBE_NGX=1"
cargo run --locked --example smoke -p neural-forge-layer
echo
echo "==> layer log ($NEURAL_FORGE_LOG):"
cat "$NEURAL_FORGE_LOG" 2>/dev/null || echo "(no log written -- the layer never ran)"
if ! grep -q '\[probe-ngx\] on in pid' "$NEURAL_FORGE_LOG" 2>/dev/null; then
    echo "FAIL: NEURAL_FORGE_PROBE_NGX=1 did not turn the probe on" >&2
    exit 1
fi

# Third pass with the pre-upscaler path's model mode named explicitly (docs/PRE_UPSCALER_DESIGN.md,
# "Implementation (layer)"): the local ICD has no VK_NVX_* extensions and no DLSS, so this proves
# the mode is taken, nothing is held, and the probe's logging stays off.
unset NEURAL_FORGE_PROBE_NGX
export NEURAL_FORGE_PREUPSCALE=model
export NEURAL_FORGE_LOG="$SCRATCH/layer-preupscale.log"
echo
echo "==> again with NEURAL_FORGE_PREUPSCALE=model"
cargo run --locked --example smoke -p neural-forge-layer
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
