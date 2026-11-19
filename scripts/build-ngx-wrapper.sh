#!/usr/bin/env bash
# build-ngx-wrapper.sh [OUT]: builds the NGX wrapper for Proton (crates/ngx-wrapper, x86_64-pc-windows-gnu) and
# writes it to OUT (default target/ngx-wrapper/_nvngx.dll) stamped as a Wine builtin: the 32 bytes at file offset
# 0x40 read "Wine builtin DLL", what `winebuild --builtin` writes and what Wine's loader checks before it serves a
# DLL from its builtin directories (docs/PROTON_NGX.md). With NVIDIA's driver DLL installed, also checks that the
# wrapper exports every name of it with the same ordinal.
#   --self-test   stamp and check a minimal PE header, no build
# Needs the Rust target x86_64-pc-windows-gnu (rustup target add) and the mingw-w64 linker (gcc-mingw-w64-x86-64).
set -euo pipefail
cd "$(dirname "$0")/.."

# The stamp, its check and the export listing are `cargo xtask wine-builtin` (crates/xtask/src/wine_builtin.rs).
stamp() {
    cargo xtask wine-builtin stamp "$1"
}

check_stamp() {
    cargo xtask wine-builtin check "$1"
}

if [ "${1:-}" = "--self-test" ]; then
    cargo xtask wine-builtin --self-test || { echo "self-test: the Wine builtin stamp failed"; exit 1; }
    echo "build-ngx-wrapper.sh self-test: ok"
    exit 0
fi

out=${1:-target/ngx-wrapper/_nvngx.dll}
target_dir="${CARGO_TARGET_DIR:-$PWD/target}/ngx-wrapper-build"
cargo build --release --locked --manifest-path crates/ngx-wrapper/Cargo.toml --target x86_64-pc-windows-gnu --target-dir "$target_dir"
mkdir -p "$(dirname "$out")"
cp "$target_dir/x86_64-pc-windows-gnu/release/_nvngx.dll" "$out.tmp"
stamp "$out.tmp"
check_stamp "$out.tmp"
mv -f "$out.tmp" "$out"

exports() {
    x86_64-w64-mingw32-objdump -p "$1" | cargo xtask wine-builtin exports
}
nvidia=""
for d in /usr/lib/x86_64-linux-gnu/nvidia/wine /usr/lib64/nvidia/wine /usr/lib/nvidia/wine; do [ -f "$d/_nvngx.dll" ] && nvidia="$d/_nvngx.dll" && break; done
if [ -n "$nvidia" ] && command -v x86_64-w64-mingw32-objdump >/dev/null; then
    # Both listings are captured first: a failing objdump or `wine-builtin exports` stops the script here (set -e),
    # where inside <(...) it would only look like an empty listing.
    nvidia_exports=$(exports "$nvidia")
    our_exports=$(exports "$out")
    count=$(printf '%s\n' "$our_exports" | grep -c . || true)
    if [ "$count" -eq 0 ]; then
        echo "error: no exports found in $out" >&2; exit 1
    fi
    if [ "$nvidia_exports" = "$our_exports" ]; then
        echo "exports: all $count of $nvidia, same ordinals"
    else
        echo "exports differ from $nvidia:"; diff <(printf '%s\n' "$nvidia_exports") <(printf '%s\n' "$our_exports") | head -20; exit 1
    fi
fi
echo "$out: $(stat -c %s "$out") bytes, Wine builtin"
