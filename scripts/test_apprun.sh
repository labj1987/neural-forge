#!/usr/bin/env bash
# Tests data/AppRun (the AppImage's entry point) in a scratch AppDir, with stand-ins for ldd, the dialog tools and the
# GUI: a GUI whose libraries resolve is started with its arguments; one with missing libraries is not, and the
# libraries and packages are named on stderr and in the first dialog tool that can show them.
# usage: bash scripts/test_apprun.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/app/usr/bin" "$tmp/bin"
cp "$ROOT/data/AppRun" "$tmp/app/AppRun"
printf '#!/bin/sh\necho "gui ran: $*"\n' > "$tmp/app/usr/bin/neural-forge"
chmod 755 "$tmp/app/usr/bin/neural-forge"
# Only the tools AppRun uses, so the machine's own zenity, kdialog or notify-send are never found.
for tool in dirname readlink awk sort paste sed cat; do ln -s "$(command -v "$tool")" "$tmp/bin/$tool"; done
fail() { echo "test_apprun: $*" >&2; exit 1; }
stub() { printf '#!/bin/sh\n%s\n' "$2" > "$tmp/bin/$1"; chmod 755 "$tmp/bin/$1"; }
run() { env -i PATH="$tmp/bin" "$@" "$BASH" "$tmp/app/AppRun" --flag > "$tmp/out" 2> "$tmp/err"; }

# Libraries resolve: the GUI starts with the arguments.
stub ldd 'printf "\tlibadwaita-1.so.0 => /usr/lib/libadwaita-1.so.0 (0x1)\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x2)\n"'
run || fail "resolved libraries: AppRun failed"
grep -qx "gui ran: --flag" "$tmp/out" || fail "resolved libraries: the GUI did not run with its arguments"

# No ldd: nothing to check, the GUI starts.
rm "$tmp/bin/ldd"
run || fail "no ldd: AppRun failed"
grep -qx "gui ran: --flag" "$tmp/out" || fail "no ldd: the GUI did not run"

# Missing libraries, no display: stderr names them and the packages, the GUI does not run, the exit is non-zero.
stub ldd 'printf "\tlibgtk-4.so.1 => not found\n\tlibadwaita-1.so.0 => not found\n\tlibadwaita-1.so.0 => not found\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x2)\n"'
if run; then fail "missing libraries: AppRun exited 0"; fi
[ ! -s "$tmp/out" ] || fail "missing libraries: the GUI ran"
grep -q "not installed: libadwaita-1.so.0, libgtk-4.so.1\.$" "$tmp/err" || fail "missing libraries: not named once each: $(cat "$tmp/err")"
for pkg in "apt install libgtk-4-1 libadwaita-1-0" "dnf install gtk4 libadwaita" "pacman -S gtk4 libadwaita" "zypper install libgtk-4-1 libadwaita-1-0"; do
    grep -qF "$pkg" "$tmp/err" || fail "missing libraries: '$pkg' not said"
done

# With a display: zenity shows it.
stub zenity "echo zenity >> '$tmp/shown'"
stub kdialog "echo kdialog >> '$tmp/shown'"
stub notify-send "echo notify-send >> '$tmp/shown'"
if run WAYLAND_DISPLAY=wayland-0; then fail "dialog: AppRun exited 0"; fi
[ "$(cat "$tmp/shown")" = zenity ] || fail "dialog: expected zenity alone, got $(cat "$tmp/shown")"
# A zenity that cannot run (127) falls through to kdialog; without kdialog, notify-send.
rm "$tmp/shown"; stub zenity "exit 127"
if run DISPLAY=:0; then fail "kdialog: AppRun exited 0"; fi
[ "$(cat "$tmp/shown")" = kdialog ] || fail "kdialog: got $(cat "$tmp/shown")"
rm "$tmp/shown" "$tmp/bin/kdialog"
if run DISPLAY=:0; then fail "notify-send: AppRun exited 0"; fi
[ "$(cat "$tmp/shown")" = notify-send ] || fail "notify-send: got $(cat "$tmp/shown")"
echo "test_apprun: ok"
