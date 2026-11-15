#!/usr/bin/env bash
# Tests data/AppRun (the AppImage's entry point) in a scratch AppDir, with stand-ins for ldd, the dialog tools and the
# two GUI copies (usr/bin: the host's GTK; usr/bin/bundled: the bundled one):
# - the host copy runs, with no bundle variables, when `ldd -r` finds every library and symbol;
# - the bundled copy runs, with them, when a library or symbol version is missing, a symbol is undefined, or there is
#   no ldd or a failing one; NEURAL_FORGE_GTK=host|bundled forces either; one stderr line names the choice;
# - on the bundled path, a missing host-only library stops it, named on stderr and in the first dialog tool that can
#   show it.
# usage: bash scripts/test_apprun.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/app/usr/bin/bundled" "$tmp/bin"
cp "$ROOT/data/AppRun" "$tmp/app/AppRun"
for copy in bin bin/bundled; do
    printf '#!/bin/sh\necho "%s ran: $*"\nenv | grep -E "^(PATH|GTK_EXE_PREFIX|GTK_PATH|GIO_MODULE_DIR|GIO_EXTRA_MODULES|GDK_PIXBUF_MODULE_FILE|GSETTINGS_SCHEMA_DIR|NEURAL_FORGE_BUNDLE_ENV|NEURAL_FORGE_ORIG_[A-Z_]*)=" || true\n' \
        "$copy" > "$tmp/app/usr/$copy/neural-forge"
    chmod 755 "$tmp/app/usr/$copy/neural-forge"
done
# Only the tools AppRun uses, so the machine's own zenity, kdialog or notify-send are never found.
for tool in dirname readlink awk sort paste sed cat grep env; do ln -s "$(command -v "$tool")" "$tmp/bin/$tool"; done
fail() { echo "test_apprun: $*" >&2; exit 1; }
stub() { printf '#!/bin/sh\n%s\n' "$2" > "$tmp/bin/$1"; chmod 755 "$tmp/bin/$1"; }
run() { env -i PATH="$tmp/bin" "$@" "$BASH" "$tmp/app/AppRun" --flag > "$tmp/out" 2> "$tmp/err"; }
# ldd: `ldd -r` (the host copy) prints $tmp/ldd-host with @BIN@ as the path it was given (the loader names the object
# that needs a symbol that way), plain ldd (the bundled copy) prints $tmp/ldd-bundled.
lddstub="if [ \"\$1\" = -r ]; then sed \"s#@BIN@#\$2#\" '$tmp/ldd-host'; else cat '$tmp/ldd-bundled'; fi"
stub ldd "$lddstub"
resolved=$'\tlibadwaita-1.so.0 => /usr/lib/libadwaita-1.so.0 (0x1)\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x2)'
printf '%s\n' "$resolved" > "$tmp/ldd-host"
printf '%s\n' "$resolved" > "$tmp/ldd-bundled"
ran_host() {
    grep -qx "bin ran: --flag" "$tmp/out" || fail "$1: the host copy did not run with its arguments: $(cat "$tmp/out" "$tmp/err")"
    if grep -E '^(GTK|GIO|GDK|GSETTINGS)' "$tmp/out"; then fail "$1: the host copy got bundle variables: $(cat "$tmp/out")"; fi
    # PATH has the AppImage's usr/bin first, and its earlier value is saved for the programs the GUI starts.
    grep -qx "PATH=$tmp/app/usr/bin:$tmp/bin" "$tmp/out" || fail "$1: PATH: $(cat "$tmp/out")"
    grep -qx "NEURAL_FORGE_BUNDLE_ENV=PATH" "$tmp/out" || fail "$1: only PATH is listed: $(cat "$tmp/out")"
    grep -qx "NEURAL_FORGE_ORIG_PATH=$tmp/bin" "$tmp/out" || fail "$1: PATH not saved: $(cat "$tmp/out")"
    grep -q "^neural-forge: GTK: host (" "$tmp/err" || fail "$1: no 'GTK: host' line: $(cat "$tmp/err")"
}
ran_bundled() {
    grep -qx "bin/bundled ran: --flag" "$tmp/out" || fail "$1: the bundled copy did not run with its arguments: $(cat "$tmp/out" "$tmp/err")"
    grep -qx "GTK_EXE_PREFIX=$tmp/app/usr" "$tmp/out" || fail "$1: no GTK_EXE_PREFIX for the bundled copy: $(cat "$tmp/out")"
    grep -qx "GIO_MODULE_DIR=$tmp/app/usr/lib/gio/modules" "$tmp/out" || fail "$1: no GIO_MODULE_DIR: $(cat "$tmp/out")"
    grep -qx "NEURAL_FORGE_BUNDLE_ENV=PATH GTK_EXE_PREFIX GTK_PATH GDK_PIXBUF_MODULE_FILE GIO_MODULE_DIR GIO_EXTRA_MODULES GSETTINGS_SCHEMA_DIR" "$tmp/out" \
        || fail "$1: no NEURAL_FORGE_BUNDLE_ENV: $(cat "$tmp/out")"
    grep -q "^neural-forge: GTK: bundled (" "$tmp/err" || fail "$1: no 'GTK: bundled' line: $(cat "$tmp/err")"
}

# The host's libraries resolve everything: the host copy, with nothing of the bundle's.
run || fail "host OK: AppRun failed"
ran_host "host OK"
[ "$(wc -l < "$tmp/err")" -eq 1 ] || fail "host OK: more than one stderr line: $(cat "$tmp/err")"

# The host lacks a library: the bundled copy, and the line names the library.
printf '\tlibadwaita-1.so.0 => not found\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x2)\n' > "$tmp/ldd-host"
run || fail "host lacks a library: AppRun failed"
ran_bundled "host lacks a library"
grep -q "the host lacks libadwaita-1.so.0" "$tmp/err" || fail "host lacks a library: not named: $(cat "$tmp/err")"

# A symbol version the host's library does not have (an older libadwaita): bundled.
printf '%s\n' "$resolved" "./neural-forge: /usr/lib/libadwaita-1.so.0: version \`LIBADWAITA_1_5' not found (required by ./neural-forge)" > "$tmp/ldd-host"
run || fail "missing symbol version: AppRun failed"
ran_bundled "missing symbol version"
grep -q "the host lacks LIBADWAITA_1_5" "$tmp/err" || fail "missing symbol version: not named: $(cat "$tmp/err")"

# Undefined symbols the GUI needs (a host GTK older than the one it was built with), in the loader's three forms:
# bundled. A host library's own unresolved symbol does not count.
printf '%s\n' "$resolved" "undefined symbol: adw_about_dialog_new	(@BIN@)" "undefined symbol: gtk_file_dialog_new, version GTK_4	(@BIN@)" \
    "symbol adw_toast_new version LIBADWAITA_1 not defined in file libadwaita-1.so.0 with link time reference	(@BIN@)" \
    "undefined symbol: g_private_helper	(/usr/lib/libfoo.so.1)" > "$tmp/ldd-host"
run || fail "undefined symbols: AppRun failed"
ran_bundled "undefined symbols"
grep -q "lack 3 symbols" "$tmp/err" || fail "undefined symbols: not counted: $(cat "$tmp/err")"
printf '%s\n' "$resolved" "undefined symbol: g_private_helper	(/usr/lib/libfoo.so.1)" "undefined symbol: x	(@BIN@-other)" > "$tmp/ldd-host"
run || fail "a host library's own symbol: AppRun failed"
ran_host "a host library's own symbol"

# A GTK_PATH and GIO_EXTRA_MODULES the user had point at host modules: the bundled copy runs without them, and their
# values are saved for children.
printf '\tlibadwaita-1.so.0 => not found\n' > "$tmp/ldd-host"
run GTK_PATH=/usr/lib/gtk GIO_EXTRA_MODULES=/usr/lib/gio || fail "host module paths: AppRun failed"
ran_bundled "host module paths"
if grep -qE '^(GTK_PATH|GIO_EXTRA_MODULES)=' "$tmp/out"; then fail "host module paths: still set: $(cat "$tmp/out")"; fi
grep -qx "NEURAL_FORGE_ORIG_GTK_PATH=/usr/lib/gtk" "$tmp/out" || fail "host module paths: GTK_PATH not saved: $(cat "$tmp/out")"
grep -qx "NEURAL_FORGE_ORIG_GIO_EXTRA_MODULES=/usr/lib/gio" "$tmp/out" || fail "host module paths: GIO_EXTRA_MODULES not saved: $(cat "$tmp/out")"
# A stale saved value from another environment is not passed on.
run NEURAL_FORGE_ORIG_FOO=x || fail "stale saved value: AppRun failed"
if grep -q NEURAL_FORGE_ORIG_FOO "$tmp/out"; then fail "stale saved value: kept"; fi

# A GSETTINGS_SCHEMA_DIR the user had is kept behind the bundle's, and its value saved for children.
run GSETTINGS_SCHEMA_DIR=/home/u/schemas || fail "user schemas: AppRun failed"
grep -qx "GSETTINGS_SCHEMA_DIR=$tmp/app/usr/share/glib-2.0/schemas:/home/u/schemas" "$tmp/out" || fail "user schemas: $(cat "$tmp/out")"
grep -qx "NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR=/home/u/schemas" "$tmp/out" || fail "user schemas: not saved: $(cat "$tmp/out")"
if grep -q "^NEURAL_FORGE_ORIG_GTK_EXE_PREFIX=" "$tmp/out"; then fail "user schemas: an unset variable got an ORIG value"; fi

# Overrides: host even though the host lacks symbols; bundled even though the host resolves.
run NEURAL_FORGE_GTK=host || fail "forced host: AppRun failed"
ran_host "forced host"
grep -q "NEURAL_FORGE_GTK=host" "$tmp/err" || fail "forced host: the reason is not the override: $(cat "$tmp/err")"
printf '%s\n' "$resolved" > "$tmp/ldd-host"
run NEURAL_FORGE_GTK=bundled || fail "forced bundled: AppRun failed"
ran_bundled "forced bundled"
grep -q "NEURAL_FORGE_GTK=bundled" "$tmp/err" || fail "forced bundled: the reason is not the override: $(cat "$tmp/err")"
# An unknown value decides as if unset, and says so.
run NEURAL_FORGE_GTK=sideways || fail "unknown override: AppRun failed"
ran_host "unknown override"
grep -q "NEURAL_FORGE_GTK=sideways is neither" "$tmp/err" || fail "unknown override: not said: $(cat "$tmp/err")"

# ldd itself fails (the host's dynamic loader cannot run the check): bundled.
stub ldd "if [ \"\$1\" = -r ]; then echo 'ldd: exited with unknown exit code (139)' >&2; exit 1; else cat '$tmp/ldd-bundled'; fi"
run || fail "failing ldd: AppRun failed"
ran_bundled "failing ldd"
grep -q "ldd could not check" "$tmp/err" || fail "failing ldd: not said: $(cat "$tmp/err")"

# No ldd: nothing can be checked, so the bundled copy.
rm "$tmp/bin/ldd"
run || fail "no ldd: AppRun failed"
ran_bundled "no ldd"
grep -q "no ldd" "$tmp/err" || fail "no ldd: not said: $(cat "$tmp/err")"

# The bundled path, with host-only libraries missing and no display: stderr names them once each and how to install
# them, the GUI does not run, the exit is non-zero.
stub ldd "$lddstub"
printf '\tlibadwaita-1.so.0 => not found\n' > "$tmp/ldd-host"
printf '\tlibEGL.so.1 => not found\n\tlibfontconfig.so.1 => not found\n\tlibEGL.so.1 => not found\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x2)\n' > "$tmp/ldd-bundled"
if run; then fail "missing libraries: AppRun exited 0"; fi
grep -q "ran:" "$tmp/out" && fail "missing libraries: a GUI ran"
grep -q "not installed: libEGL.so.1, libfontconfig.so.1\.$" "$tmp/err" || fail "missing libraries: not named once each: $(cat "$tmp/err")"
grep -qF "sudo dnf install 'libEGL.so.1()(64bit)' 'libfontconfig.so.1()(64bit)'" "$tmp/err" || fail "missing libraries: no install hint: $(cat "$tmp/err")"

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
