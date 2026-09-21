#!/usr/bin/env bash
# Cross-compiles neural-forge-helper and its examples for x86_64-pc-windows-gnu, then runs
# the example tests under Wine. See crates/helper/examples/*.rs for what each one
# actually checks, and CLAUDE.md's "helper gotchas" for the bug this already caught.
#
# Needs: mingw-w64 + a rustup toolchain with the x86_64-pc-windows-gnu target (see
# CLAUDE.md for the exact setup), and `wine`. Uses the `stable-x86_64-unknown-linux-gnu`
# rustup toolchain explicitly (`+stable-x86_64-unknown-linux-gnu`) -- never the plain
# `cargo`, which stays pinned to this project's regular toolchain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TOOLCHAIN="+stable-x86_64-unknown-linux-gnu"
TARGET="x86_64-pc-windows-gnu"

echo "==> cross-compiling neural-forge-helper + examples"
cargo "$TOOLCHAIN" build --target "$TARGET" -p neural-forge-helper \
    --bin neural-forge-helper --example guard_test --example spoof_test --example spoof_install_test

SCRATCH="$(mktemp -d)"
export WINEPREFIX="$SCRATCH/prefix"
export WINEDEBUG=-all
trap 'WINEPREFIX="$SCRATCH/prefix" wineserver -k 2>/dev/null || true; WINEPREFIX="$SCRATCH/prefix" wineserver -w 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT
BIN_DIR="target/$TARGET/debug"
FAIL=0
for example in guard_test spoof_test spoof_install_test; do
    echo
    echo "==> running $example under wine"
    if ! timeout 15 wine "$BIN_DIR/examples/$example.exe"; then
        status=$?
        echo "!!! $example FAILED (exit $status)"
        FAIL=1
    fi
done

echo
echo "==> running the full neural-forge-helper.exe binary for a few seconds (expect a clean"
echo "    fail-open: no nvngx_dlssnr.dll is available on this dev machine)"
NEURALFORGE_LOG="$SCRATCH/helper.log" NEURALFORGE_UID="helper-test-$$" \
    timeout 8 wine "$BIN_DIR/neural-forge-helper.exe" > "$SCRATCH/stdout.log" 2>&1 || true
echo "--- helper log ---"
cat "$SCRATCH/helper.log" 2>/dev/null || echo "(no log written)"
rm -rf "/tmp/neural-forge-helper-test-$$"

if [ "$FAIL" -ne 0 ]; then
    echo
    echo "==> one or more example tests FAILED -- see above"
    exit 1
fi
echo
echo "==> all helper example tests passed"
