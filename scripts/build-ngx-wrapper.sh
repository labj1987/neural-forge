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

stamp() {
    python3 -I - "$1" <<'PY'
import struct, sys
path = sys.argv[1]
b = bytearray(open(path, 'rb').read())
if b[:2] != b'MZ':
    sys.exit(f'{path}: not a PE file')
lfanew = struct.unpack_from('<I', b, 0x3c)[0]
if lfanew < 0x60:
    sys.exit(f'{path}: no room for the Wine builtin signature (e_lfanew {lfanew:#x})')
b[0x40:0x60] = b'Wine builtin DLL'.ljust(32, b'\0')
open(path, 'wb').write(b)
PY
}

check_stamp() {
    python3 -I -c 'import sys; b = open(sys.argv[1], "rb").read(0x60); sys.exit(0 if b[0x40:0x50] == b"Wine builtin DLL" and b[0x50:0x60] == bytes(16) else 1)' "$1"
}

if [ "${1:-}" = "--self-test" ]; then
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    python3 -I -c 'import struct, sys; b = bytearray(0x100); b[:2] = b"MZ"; struct.pack_into("<I", b, 0x3c, 0x80); open(sys.argv[1], "wb").write(b)' "$tmp/pe.dll"
    stamp "$tmp/pe.dll"; check_stamp "$tmp/pe.dll" || { echo "self-test: stamp not found"; exit 1; }
    python3 -I -c 'import struct, sys; b = bytearray(0x100); b[:2] = b"MZ"; struct.pack_into("<I", b, 0x3c, 0x40); open(sys.argv[1], "wb").write(b)' "$tmp/small.dll"
    if (stamp "$tmp/small.dll") 2>/dev/null; then echo "self-test: a header without room was stamped"; exit 1; fi
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
    x86_64-w64-mingw32-objdump -p "$1" | python3 -I -c '
import re, sys
for m in re.finditer(r"\+base\[\s*(\d+)\]\s+[0-9a-f]{4}\s+(NVSDK\S+)", sys.stdin.read()): print(m.group(2), m.group(1))'
}
nvidia=""
for d in /usr/lib/x86_64-linux-gnu/nvidia/wine /usr/lib64/nvidia/wine /usr/lib/nvidia/wine; do [ -f "$d/_nvngx.dll" ] && nvidia="$d/_nvngx.dll" && break; done
if [ -n "$nvidia" ] && command -v x86_64-w64-mingw32-objdump >/dev/null; then
    if diff <(exports "$nvidia") <(exports "$out") >/dev/null; then
        echo "exports: all $(exports "$out" | wc -l) of $nvidia, same ordinals"
    else
        echo "exports differ from $nvidia:"; diff <(exports "$nvidia") <(exports "$out") | head -20; exit 1
    fi
fi
echo "$out: $(stat -c %s "$out") bytes, Wine builtin"
