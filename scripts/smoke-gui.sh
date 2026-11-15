#!/usr/bin/env bash
# smoke-gui.sh APPDIR [host|bundled...]: starts the extracted AppImage's GUI (APPDIR/AppRun) on a virtual X display
# (Xvfb, its own D-Bus session) with each GTK copy (default: both) and checks that it reaches its main loop: its window
# is mapped (xdotool), it is still running, and its stderr names the copy and holds no loader or GTK error. Everything
# it writes goes to a scratch HOME, XDG directories and XDG_RUNTIME_DIR, never the user's.
# Needs xvfb-run, dbus-run-session and xdotool.
#   --self-test   check the stderr verdicts, no GUI
set -euo pipefail

# verdict PATH STDERR-FILE: why the run's stderr fails, or nothing.
verdict() {
    grep -q "^neural-forge: GTK: $1 (" "$2" || { echo "no 'GTK: $1' line"; return 0; }
    if grep -E 'error while loading shared libraries|undefined symbol|-CRITICAL \*\*|-ERROR \*\*|Gtk-WARNING|Gdk-WARNING|GLib-GIO-WARNING|Failed to load module' "$2"; then
        echo "errors on stderr"
    fi
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
    printf 'neural-forge: GTK: bundled (x)\n' > "$t/ok"
    [ -z "$(verdict bundled "$t/ok")" ] || { echo "self-test: a clean run failed"; exit 1; }
    [ -n "$(verdict host "$t/ok")" ] || { echo "self-test: the wrong copy passed"; exit 1; }
    printf 'neural-forge: GTK: bundled (x)\n(x:1): Gtk-CRITICAL **: boom\n' > "$t/bad"
    [ -n "$(verdict bundled "$t/bad")" ] || { echo "self-test: a GTK error passed"; exit 1; }
    echo "smoke-gui.sh self-test: ok"
    exit 0
fi

appdir="$(cd "$1" && pwd)"; shift
[ $# -gt 0 ] || set -- host bundled
for tool in xvfb-run dbus-run-session xdotool; do command -v "$tool" >/dev/null || { echo "smoke-gui: $tool is missing" >&2; exit 1; }; done
fail=0
for copy in "$@"; do
    s="$(mktemp -d)"
    mkdir -p "$s/home" "$s/run"; chmod 700 "$s/run"
    # The run: start the GUI, wait up to 30 s for its window, check it still runs, then end it.
    # shellcheck disable=SC2016 # the inner script's own variables
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$s/home" XDG_RUNTIME_DIR="$s/run" XDG_CONFIG_HOME="$s/home/.config" \
        XDG_DATA_HOME="$s/home/.local/share" XDG_STATE_HOME="$s/home/.local/state" XDG_CACHE_HOME="$s/home/.cache" \
        NEURAL_FORGE_GTK="$copy" GDK_BACKEND=x11 LANG=C.UTF-8 \
        xvfb-run -a -s "-screen 0 1920x1080x24" dbus-run-session -- bash -c '
            "$1/AppRun" > "$2/out" 2> "$2/err" & gui=$!
            for _ in $(seq 60); do
                xdotool search --onlyvisible --name "^Neural Forge$" >/dev/null 2>&1 && { echo mapped > "$2/state"; break; }
                kill -0 "$gui" 2>/dev/null || break
                sleep 0.5
            done
            sleep 2
            kill -0 "$gui" 2>/dev/null && echo running >> "$2/state"
            kill "$gui" 2>/dev/null; wait "$gui" 2>/dev/null || true' _ "$appdir" "$s" 2> "$s/session" || true
    state="$(cat "$s/state" 2>/dev/null | paste -sd' ' -)"
    why="$(verdict "$copy" "$s/err")"
    if [ "$state" != "mapped running" ] || [ -n "$why" ]; then
        echo "smoke-gui: $copy: FAILED (${state:-no window}; ${why:-stderr clean})" >&2
        sed 's/^/  /' "$s/err" >&2
        fail=1
    else
        echo "smoke-gui: $copy: window mapped, main loop running, stderr clean"
    fi
    rm -rf "$s"
done
exit "$fail"
