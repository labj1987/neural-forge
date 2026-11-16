#!/usr/bin/env bash
# smoke-gui.sh APPDIR: starts the extracted AppImage's GUI (APPDIR/AppRun) on a virtual X display (Xvfb, its own D-Bus
# session) and checks that it reaches its main loop: its window ("Neural Forge", app id io.github.labj1987.NeuralForge)
# is mapped (xdotool), it is still running, and its stderr holds no loader error, no panic and no line the GUI logged
# at WARN or ERROR level (it logs "neural-forge: LEVEL [target] message"). Xvfb has no GPU: the window is drawn on
# lavapipe (VK_DRIVER_FILES set to Mesa's lvp_icd*.json when there is one, as CI does for the layer's tests), never on
# the machine's own GPU. Everything it writes goes to a scratch HOME, XDG directories and a private XDG_RUNTIME_DIR,
# never the user's. Needs xvfb-run (xvfb), dbus-run-session (dbus-daemon) and xdotool.
#   --self-test   check the stderr verdicts, no GUI
set -euo pipefail

# verdict STDERR-FILE: why the run's stderr fails (the offending lines, then a summary), or nothing.
verdict() {
    # What the bare session lacks is not the GUI's fault: Xvfb has no XInput pointer device (the toolkit reports that at
    # error level and works without), and a D-Bus session with no desktop portal and no accessibility bus makes the
    # toolkit's D-Bus crate warn that those services are unknown.
    if grep -v -e "Found no xinput mouse pointers" -e "WARN \[zbus::proxy\] Failed to populate properties cache via GetAll: org.freedesktop.DBus.Error.ServiceUnknown" "$1" \
        | grep -E "error while loading shared libraries|undefined symbol|panicked|can't start|^neural-forge: (WARN|ERROR) "; then
        echo "errors on stderr"
    fi
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
    printf 'neural-forge: INFO [neural_forge::app] window opened\nneural-forge: DEBUG [x] WARN in a message is fine\n' > "$t/ok"
    [ -z "$(verdict "$t/ok")" ] || { echo "self-test: a clean run failed"; exit 1; }
    printf 'neural-forge: ERROR [gpui_linux::linux::x11::client] Found no xinput mouse pointers.\n' > "$t/xvfb"
    printf 'neural-forge: WARN [zbus::proxy] Failed to populate properties cache via GetAll: org.freedesktop.DBus.Error.ServiceUnknown: The name org.a11y.Bus was not provided by any .service files. Property change streams will not produce values.\n' >> "$t/xvfb"
    [ -z "$(verdict "$t/xvfb")" ] || { echo "self-test: what the bare session lacks failed a run"; exit 1; }
    printf 'neural-forge: WARN [zbus::proxy] something else\n' > "$t/zbus"
    [ -n "$(verdict "$t/zbus")" ] || { echo "self-test: another D-Bus warning passed"; exit 1; }
    for bad in 'neural-forge: WARN [neural_forge::status] no channel' 'neural-forge: ERROR [gpui] boom' \
        "thread 'main' panicked at src/main.rs:1:1:" \
        './neural-forge: error while loading shared libraries: libxcb.so.1: cannot open shared object file' \
        './neural-forge: symbol lookup error: ./neural-forge: undefined symbol: xkb_x' \
        "Neural Forge: Neural Forge can't start because these system libraries are missing: libEGL.so.1."; do
        printf 'neural-forge: INFO [x] started\n%s\n' "$bad" > "$t/bad"
        [ -n "$(verdict "$t/bad")" ] || { echo "self-test: passed: $bad"; exit 1; }
    done
    echo "smoke-gui.sh self-test: ok"
    exit 0
fi

[ $# -eq 1 ] || { echo "usage: smoke-gui.sh APPDIR | --self-test" >&2; exit 2; }
appdir="$(cd "$1" && pwd)"
for tool in xvfb-run dbus-run-session xdotool; do command -v "$tool" >/dev/null || { echo "smoke-gui: $tool is missing" >&2; exit 1; }; done
icds="$(find /usr/share/vulkan/icd.d -maxdepth 1 -name 'lvp_icd*.json' 2>/dev/null | sort | paste -sd: -)"
[ -n "$icds" ] || echo "smoke-gui: no lavapipe ICD in /usr/share/vulkan/icd.d (mesa-vulkan-drivers); the GUI picks its own renderer" >&2
s="$(mktemp -d)"
trap 'rm -rf "$s"' EXIT
mkdir -p "$s/home" "$s/run"; chmod 700 "$s/run"
# The run: start the GUI, wait up to 30 s for its window, check it still runs 2 s later, then end it.
# shellcheck disable=SC2016 # the inner script's own variables
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$s/home" XDG_RUNTIME_DIR="$s/run" XDG_CONFIG_HOME="$s/home/.config" \
    XDG_DATA_HOME="$s/home/.local/share" XDG_STATE_HOME="$s/home/.local/state" XDG_CACHE_HOME="$s/home/.cache" \
    LANG=C.UTF-8 ${icds:+VK_DRIVER_FILES="$icds"} \
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
state="$( { cat "$s/state" 2>/dev/null || true; } | paste -sd' ' -)"
why="$(verdict "$s/err")"
if [ "$state" != "mapped running" ] || [ -n "$why" ]; then
    echo "smoke-gui: FAILED (${state:-no window}; ${why:-stderr clean})" >&2
    echo "  the GUI's stderr:" >&2
    sed 's/^/    /' "$s/err" >&2 || true
    echo "  the session's (Xvfb, D-Bus):" >&2
    sed 's/^/    /' "$s/session" >&2 || true
    exit 1
fi
echo "smoke-gui: window mapped, main loop running, stderr clean"
