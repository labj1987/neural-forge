#!/usr/bin/env bash
# Tests data/AppRun (the AppImage's entry point) in a scratch AppDir, with stand-ins for ldd, ldconfig, the dynamic
# loader, the dialog tools and the GUI:
# - when the system has everything, the GUI runs with its arguments and exactly the environment AppRun was given;
# - it does not run, and stderr says why (once per library, with the install command for the distribution), when a
#   linked library is missing, glibc is too old, a symbol the GUI needs is undefined, or the libraries it loads at
#   run time are missing (libwayland-client only on a Wayland session; Vulkan's or EGL's loader, either is enough);
# - without ldd, or when ldd fails without saying why, the dynamic loader is asked;
# - with a display, the first tool that can show the message does (zenity, kdialog, notify-send, the notification
#   service over D-Bus, xmessage).
# usage: bash scripts/test_apprun.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/app/usr/bin" "$tmp/bin" "$tmp/lib" "$tmp/tmpdir"
cp "$ROOT/data/AppRun" "$tmp/app/AppRun"
GUI="$tmp/app/usr/bin/neural-forge"
# The GUI stand-in: its arguments one per line, and its environment, sorted.
printf '#!/bin/sh\nprintf "ran:%%s\\n" "$@"\nenv | sort > "%s/env"\n' "$tmp" > "$GUI"
chmod 755 "$GUI"
# Only the tools AppRun uses, so the machine's own ldd, zenity, kdialog or notify-send are never found.
for tool in dirname readlink sort tail head sed cat env mktemp rm; do ln -s "$(command -v "$tool")" "$tmp/bin/$tool"; done
fail() { echo "test_apprun: $*" >&2; exit 1; }
stub() { printf '#!/bin/sh\n%s\n' "$2" > "$tmp/bin/$1"; chmod 755 "$tmp/bin/$1"; }
# ldd -r prints $tmp/ldd with @BIN@ as the path it was given (the loader names the object that needs a symbol that
# way) and exits with $tmp/ldd-rc; ldconfig -p prints $tmp/cache. The loader stand-in (LIBCHECK_LOADER) prints
# $tmp/loader.
stub ldd "[ \"\$1\" = -r ] || exit 2; sed \"s#@BIN@#\$2#g\" '$tmp/ldd'; exit \$(cat '$tmp/ldd-rc')"
stub ldconfig "[ \"\$1\" = -p ] || exit 2; cat '$tmp/cache'"
# shellcheck disable=SC2016 # the stand-in's own $1
printf '#!/bin/sh\nsed "s#@BIN@#$1#g" "%s/loader"\n' "$tmp" > "$tmp/loader-stub"; chmod 755 "$tmp/loader-stub"
linked=$'\tlinux-vdso.so.1 (0x1)\n\tlibxcb.so.1 => /usr/lib/libxcb.so.1 (0x2)\n\tlibxkbcommon.so.0 => /usr/lib/libxkbcommon.so.0 (0x3)\n\tlibxkbcommon-x11.so.0 => /usr/lib/libxkbcommon-x11.so.0 (0x4)\n\tlibgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x5)\n\tlibc.so.6 => /usr/lib/libc.so.6 (0x6)'
cache_line() { printf '\t%s (libc6,x86-64) => /usr/lib/%s\n' "$1" "$1"; }
full_cache() { echo "9 libs found in cache \`/etc/ld.so.cache'"; for l in libc.so.6 libxcb.so.1 libvulkan.so.1 libEGL.so.1 libwayland-client.so.0; do cache_line "$l"; done; }
reset() {
    printf '%s\n' "$linked" > "$tmp/ldd"; echo 0 > "$tmp/ldd-rc"; full_cache > "$tmp/cache"
    printf '%s\n' "$linked" > "$tmp/loader"; rm -f "$tmp/shown" "$tmp/env"
    stub ldd "[ \"\$1\" = -r ] || exit 2; sed \"s#@BIN@#\$2#g\" '$tmp/ldd'; exit \$(cat '$tmp/ldd-rc')"
}
printf 'ID=ubuntu\nID_LIKE=debian\n' > "$tmp/ubuntu"; printf 'ID=fedora\n' > "$tmp/fedora"; printf 'ID=gentoo\n' > "$tmp/gentoo"
# run VAR=VALUE...: AppRun with only these variables (and PATH), the GUI given two arguments, one with a space.
run() {
    rm -f "$tmp/out" "$tmp/err" "$tmp/env"
    env -i PATH="$tmp/bin" TMPDIR="$tmp/tmpdir" LIBCHECK_OS_RELEASE="$tmp/ubuntu" LIBCHECK_GLIBC_HAVE=2.35 "$@" "$BASH" "$tmp/app/AppRun" --flag "a b" > "$tmp/out" 2> "$tmp/err"
}
ran() {
    [ "$(cat "$tmp/out")" = $'ran:--flag\nran:a b' ] || fail "$1: the GUI did not run with its arguments: $(cat "$tmp/out" "$tmp/err")"
    [ ! -s "$tmp/err" ] || fail "$1: AppRun wrote to stderr: $(cat "$tmp/err")"
}
refused() {
    [ ! -s "$tmp/out" ] || fail "$1: the GUI ran: $(cat "$tmp/out")"
    [ ! -e "$tmp/env" ] || fail "$1: the GUI ran"
    grep -q "^Neural Forge: Neural Forge can't start" "$tmp/err" || fail "$1: no message on stderr: $(cat "$tmp/err")"
    [ "$(wc -l < "$tmp/err")" -eq 1 ] || fail "$1: more than one stderr line: $(cat "$tmp/err")"
}
says() { grep -qF -- "$2" "$tmp/err" || fail "$1: stderr lacks '$2': $(cat "$tmp/err")"; }

# Everything is there: the GUI runs, and its environment is exactly the one AppRun got (the shells add only PWD,
# SHLVL and _). No PATH change, no markers.
reset
run FOO="x y" XDG_RUNTIME_DIR=/run/user/1 WAYLAND_DISPLAY=wayland-0 LD_LIBRARY_PATH=/opt/lib || fail "all fine: AppRun failed: $(cat "$tmp/err")"
ran "all fine"
expected="$(printf '%s\n' "FOO=x y" "LD_LIBRARY_PATH=/opt/lib" "LIBCHECK_GLIBC_HAVE=2.35" "LIBCHECK_OS_RELEASE=$tmp/ubuntu" "PATH=$tmp/bin" "TMPDIR=$tmp/tmpdir" \
    "WAYLAND_DISPLAY=wayland-0" "XDG_RUNTIME_DIR=/run/user/1" | sort)"
got="$(grep -vE '^(PWD|SHLVL|_|OLDPWD)=' "$tmp/env")"
[ "$got" = "$expected" ] || fail "all fine: the environment changed: $(diff <(echo "$expected") <(echo "$got") || true)"
# A system library's own unresolved symbol, and one of another object, are not the GUI's business.
printf '%s\n' "$linked" "undefined symbol: xcb_private	(/usr/lib/libxcb.so.1)" "undefined symbol: x	(@BIN@-other)" > "$tmp/ldd"
run || fail "foreign symbols: AppRun failed: $(cat "$tmp/err")"
ran "foreign symbols"

# A linked library is missing (named twice by ldd): refused, named once, with the distribution's install command.
reset
printf '%s\n' "$linked" $'\tlibxkbcommon-x11.so.0 => not found' $'\tlibxkbcommon-x11.so.0 => not found' > "$tmp/ldd"
echo 1 > "$tmp/ldd-rc"
if run; then fail "missing library: AppRun exited 0"; fi
refused "missing library"
says "missing library" "these system libraries are missing: libxkbcommon-x11.so.0."
says "missing library" "Install them with: sudo apt install libxkbcommon-x11-0. Then start Neural Forge again."
if run LIBCHECK_OS_RELEASE="$tmp/fedora"; then fail "missing library (Fedora): AppRun exited 0"; fi
says "missing library (Fedora)" "Install them with: sudo dnf install 'libxkbcommon-x11.so.0()(64bit)'."
if run LIBCHECK_OS_RELEASE="$tmp/fedora" LIBCHECK_OSTREE_BOOTED="$tmp/fedora"; then fail "missing library (Fedora Silverblue): AppRun exited 0"; fi
says "missing library (Fedora Silverblue)" "Install them with: rpm-ostree install 'libxkbcommon-x11.so.0()(64bit)', and restart the computer."
if run LIBCHECK_OS_RELEASE="$tmp/gentoo"; then fail "missing library (unknown system): AppRun exited 0"; fi
says "missing library (unknown system)" "Install the packages that provide them with your package manager, then start Neural Forge again."
# A library the table does not know: the command that finds its package.
printf '%s\n' "$linked" $'\tlibfoo.so.3 => not found' $'\tlibxcb.so.1 => not found' > "$tmp/ldd"
if run; then fail "unknown library: AppRun exited 0"; fi
says "unknown library" "Install them with: sudo apt install libxcb1. Find the package that provides each of the others with: apt-file search libfoo.so.3."

# glibc older than the build's: the newest version needed, and the system's.
reset
printf '%s\n' "$linked" "@BIN@: /lib/libc.so.6: version \`GLIBC_2.38' not found (required by @BIN@)" \
    "@BIN@: /lib/libc.so.6: version \`GLIBC_2.39' not found (required by @BIN@)" > "$tmp/ldd"
echo 1 > "$tmp/ldd-rc"
if run; then fail "old glibc: AppRun exited 0"; fi
refused "old glibc"
says "old glibc" "it needs glibc 2.39 or newer; this system has 2.35."

# Symbols the GUI needs that the system's libraries lack, in the loader's forms: refused, the first three named.
reset
printf '%s\n' "$linked" "undefined symbol: xkb_a	(@BIN@)" "undefined symbol: xkb_b, version V_1	(@BIN@)" \
    "symbol xkb_c version V_2 not defined in file libxkbcommon.so.0 with link time reference	(@BIN@)" "undefined symbol: xkb_d	(@BIN@)" \
    "undefined symbol: xcb_private	(/usr/lib/libxcb.so.1)" > "$tmp/ldd"
if run; then fail "missing symbols: AppRun exited 0"; fi
refused "missing symbols"
says "missing symbols" "the system's libraries are too old: they lack xkb_a, xkb_b, xkb_c and 1 more. Update the system"

# The run-time libraries. No Vulkan and no EGL loader: refused, both named, either is enough.
reset
{ echo "x libs"; cache_line libc.so.6; cache_line libwayland-client.so.0; } > "$tmp/cache"
if run; then fail "no graphics loader: AppRun exited 0"; fi
refused "no graphics loader"
says "no graphics loader" "missing: libvulkan.so.1, libEGL.so.1. "
says "no graphics loader" "one of libvulkan.so.1 and libEGL.so.1 is enough"
says "no graphics loader" "sudo apt install libvulkan1 libegl1"
# Either one is enough, from the cache or from LD_LIBRARY_PATH.
{ echo "x libs"; cache_line libc.so.6; cache_line libEGL.so.1; } > "$tmp/cache"
run || fail "EGL only: AppRun failed: $(cat "$tmp/err")"
ran "EGL only"
{ echo "x libs"; cache_line libc.so.6; } > "$tmp/cache"
: > "$tmp/lib/libvulkan.so.1"
run LD_LIBRARY_PATH="/nonexistent:$tmp/lib" || fail "Vulkan from LD_LIBRARY_PATH: AppRun failed: $(cat "$tmp/err")"
ran "Vulkan from LD_LIBRARY_PATH"
# libwayland-client: needed on a Wayland session only.
{ echo "x libs"; cache_line libc.so.6; cache_line libvulkan.so.1; } > "$tmp/cache"
run DISPLAY=:0 LIBCHECK_DIALOG=none || fail "X11 without libwayland-client: AppRun failed: $(cat "$tmp/err")"
ran "X11 without libwayland-client"
if run WAYLAND_DISPLAY=wayland-0 LIBCHECK_DIALOG=none; then fail "Wayland without libwayland-client: AppRun exited 0"; fi
refused "Wayland without libwayland-client"
says "Wayland without libwayland-client" "missing: libwayland-client.so.0."
says "Wayland without libwayland-client" "sudo apt install libwayland-client0"
# A cache that does not list glibc (libraries kept elsewhere, as on NixOS): nothing is said about run-time libraries.
{ echo "0 libs"; cache_line libfoo.so.1; } > "$tmp/cache"
run WAYLAND_DISPLAY=wayland-0 || fail "no usable cache: AppRun failed: $(cat "$tmp/err")"
ran "no usable cache"

# No ldd: the dynamic loader is asked (LIBCHECK_LOADER stands in for running the GUI under LD_TRACE_LOADED_OBJECTS).
reset
rm "$tmp/bin/ldd"
run LIBCHECK_LOADER="$tmp/loader-stub" || fail "no ldd: AppRun failed: $(cat "$tmp/err")"
ran "no ldd"
printf '%s\n' "$linked" $'\tlibxcb.so.1 => not found' > "$tmp/loader"
if run LIBCHECK_LOADER="$tmp/loader-stub"; then fail "no ldd, missing library: AppRun exited 0"; fi
refused "no ldd, missing library"
says "no ldd, missing library" "missing: libxcb.so.1."
# ldd fails without naming a problem: the loader is asked, and only what it names stops the GUI.
reset
printf 'ldd: exited with unknown exit code (139)\n' > "$tmp/ldd"; echo 1 > "$tmp/ldd-rc"
run LIBCHECK_LOADER="$tmp/loader-stub" || fail "failing ldd, clean loader: AppRun failed: $(cat "$tmp/err")"
ran "failing ldd, clean loader"
printf '%s\n' "$linked" "@BIN@: /lib/libc.so.6: version \`GLIBC_2.40' not found (required by @BIN@)" > "$tmp/loader"
if run LIBCHECK_LOADER="$tmp/loader-stub"; then fail "failing ldd, loader names glibc: AppRun exited 0"; fi
says "failing ldd, loader names glibc" "needs glibc 2.40 or newer"

# The dialog: none without a display, or with LIBCHECK_DIALOG=none; else zenity; a zenity that cannot run (127)
# falls through to kdialog; without kdialog, notify-send.
reset
printf '%s\n' "$linked" $'\tlibxcb.so.1 => not found' > "$tmp/ldd"
stub zenity "echo zenity >> '$tmp/shown'"
stub kdialog "echo \"kdialog \$2\" >> '$tmp/shown'"
stub notify-send "echo \"notify-send \$*\" >> '$tmp/shown'"
if run; then fail "no display: AppRun exited 0"; fi
[ ! -e "$tmp/shown" ] || fail "no display: a dialog was shown: $(cat "$tmp/shown")"
if run WAYLAND_DISPLAY=wayland-0 LIBCHECK_DIALOG=none; then fail "dialogs off: AppRun exited 0"; fi
[ ! -e "$tmp/shown" ] || fail "dialogs off: a dialog was shown: $(cat "$tmp/shown")"
if run DISPLAY=:0; then fail "zenity: AppRun exited 0"; fi
[ "$(cat "$tmp/shown")" = zenity ] || fail "zenity: expected zenity alone, got $(cat "$tmp/shown")"
refused "zenity"
# A zenity that cannot open the display ends with 1, as a closed dialog does, and says why on stderr: the next tool
# shows the message. Its stderr is read once it has ended, not to the end of a pipe that something it left running
# still holds; the temporary file it went to is removed.
rm "$tmp/shown"
stub zenity "echo '(zenity:42): Gtk-WARNING **: cannot open display: :0' >&2; $(command -v sleep) 5 & exit 1"
started=$SECONDS
if run DISPLAY=:0; then fail "zenity without a display: AppRun exited 0"; fi
((SECONDS - started < 4)) || fail "zenity without a display: waited for what it left running"
grep -q '^kdialog Neural Forge can.t start' "$tmp/shown" || fail "zenity without a display: got $(cat "$tmp/shown")"
[ -z "$(ls -A "$tmp/tmpdir")" ] || fail "zenity without a display: left $(ls -A "$tmp/tmpdir")"
rm "$tmp/shown"; stub zenity "exit 127"
if run DISPLAY=:0; then fail "kdialog: AppRun exited 0"; fi
grep -q '^kdialog Neural Forge can.t start' "$tmp/shown" || fail "kdialog: got $(cat "$tmp/shown")"
[ "$(wc -l < "$tmp/shown")" -eq 1 ] || fail "kdialog: more than one dialog: $(cat "$tmp/shown")"
rm "$tmp/shown" "$tmp/bin/kdialog"
if run WAYLAND_DISPLAY=wayland-0; then fail "notify-send: AppRun exited 0"; fi
grep -q '^notify-send -u critical -t 0 -a Neural Forge Neural Forge Neural Forge can.t start' "$tmp/shown" \
    || fail "notify-send (critical, no expiry): got $(cat "$tmp/shown")"
[ "$(wc -l < "$tmp/shown")" -eq 1 ] || fail "notify-send: more than one message: $(cat "$tmp/shown")"
# A desktop with none of the three (a stock COSMIC): the notification service is asked over D-Bus, with the message
# as the body; where that fails too (no service), xmessage on an X display.
rm "$tmp/shown" "$tmp/bin/notify-send"
stub gdbus "echo \"gdbus \$*\" >> '$tmp/shown'"
stub xmessage "echo xmessage >> '$tmp/shown'"
if run WAYLAND_DISPLAY=wayland-0; then fail "gdbus: AppRun exited 0"; fi
grep -q "^gdbus call --session --dest org.freedesktop.Notifications .*Notify -- 'Neural Forge' uint32 0 '' 'Neural Forge' 'Neural Forge can\\\\'t start.* @as \[\] {'urgency': <byte 2>} int32 0\$" "$tmp/shown" \
    || fail "gdbus: got $(cat "$tmp/shown")"
[ "$(wc -l < "$tmp/shown")" -eq 1 ] || fail "gdbus: more than one message: $(cat "$tmp/shown")"
rm "$tmp/shown"; stub gdbus "exit 1"
if run DISPLAY=:0; then fail "xmessage: AppRun exited 0"; fi
[ "$(cat "$tmp/shown")" = xmessage ] || fail "xmessage: got $(cat "$tmp/shown")"
rm "$tmp/shown"
if run WAYLAND_DISPLAY=wayland-0; then fail "no tool: AppRun exited 0"; fi
[ ! -e "$tmp/shown" ] || fail "xmessage without an X display: $(cat "$tmp/shown")"
echo "test_apprun: ok"
