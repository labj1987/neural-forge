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
    # error level and works without).
    if grep -v -e "Found no xinput mouse pointers" "$1" \
        | grep -E "error while loading shared libraries|undefined symbol|panicked|can't start|^neural-forge: (WARN|ERROR) "; then
        echo "errors on stderr"
    fi
    return 0
}

# read_state FILE: what a start wrote of itself, on one line ("mapped running" for a window that showed and a GUI still
# running 2 s later; "running" for no window; nothing for a GUI that ended).
read_state() { { cat "$1" 2>/dev/null || true; } | paste -sd' ' -; }

# outcome STATE WHY: pass (window mapped, GUI running, stderr clean), retry (no window, but the GUI was running and its
# stderr clean: the one case that gets a second start) or fail.
outcome() {
    if [ "$1" = "mapped running" ] && [ -z "$2" ]; then echo pass
    elif [ "$1" = running ] && [ -z "$2" ]; then echo retry
    else echo fail; fi
}

if [ "${1:-}" = "--self-test" ]; then
    t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
    # A start as the run records it: the window mapped, then the GUI still running.
    printf 'mapped\nrunning\n' > "$t/state"
    [ "$(read_state "$t/state")" = "mapped running" ] || { echo "self-test: a recorded start read as '$(read_state "$t/state")'"; exit 1; }
    [ "$(outcome "$(read_state "$t/state")" "")" = pass ] || { echo "self-test: a working GUI did not pass"; exit 1; }
    [ "$(outcome "$(read_state "$t/missing")" "")" = fail ] || { echo "self-test: a GUI that ended passed"; exit 1; }
    [ "$(outcome running "")" = retry ] || { echo "self-test: no window with a running GUI is not retried"; exit 1; }
    [ "$(outcome running "neural-forge: ERROR x")" = fail ] || { echo "self-test: an error on stderr is retried"; exit 1; }
    [ "$(outcome "mapped running" "neural-forge: ERROR x")" = fail ] || { echo "self-test: an error on stderr passed"; exit 1; }
    # The whole run against a stand-in GUI (a window with the GUI's title), where the tools for one are installed.
    if [ -z "${SMOKE_GUI_NESTED:-}" ] && command -v xmessage >/dev/null && command -v xvfb-run >/dev/null && command -v xdotool >/dev/null \
        && command -v dbus-run-session >/dev/null; then
        mkdir -p "$t/app"
        printf '#!/bin/sh\nexec xmessage -title "Neural Forge" stand-in\n' > "$t/app/AppRun"; chmod +x "$t/app/AppRun"
        SMOKE_GUI_NESTED=1 bash "$0" "$t/app" > "$t/run" 2>&1 || { echo "self-test: a working stand-in GUI failed:"; cat "$t/run"; exit 1; }
        printf '#!/bin/sh\nexec sleep 100\n' > "$t/app/AppRun"
        if SMOKE_GUI_NESTED=1 SMOKE_GUI_WAIT=4 bash "$0" "$t/app" > "$t/run" 2>&1; then echo "self-test: a GUI without a window passed"; exit 1; fi
        grep -q "starting it once more" "$t/run" || { echo "self-test: no second start for a GUI without a window:"; cat "$t/run"; exit 1; }
    fi
    printf 'neural-forge: INFO [neural_forge::app] window opened\nneural-forge: DEBUG [x] WARN in a message is fine\n' > "$t/ok"
    [ -z "$(verdict "$t/ok")" ] || { echo "self-test: a clean run failed"; exit 1; }
    printf 'neural-forge: ERROR [gpui_linux::linux::x11::client] Found no xinput mouse pointers.\n' > "$t/xvfb"
    [ -z "$(verdict "$t/xvfb")" ] || { echo "self-test: what the bare session lacks failed a run"; exit 1; }
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
state=""
why=""
# One start: the GUI, up to 30 s for its window, a check that it still runs 2 s later, then its end.
start_once() {
    rm -f "$s/state" "$s/out" "$s/err"
    # shellcheck disable=SC2016 # the inner script's own variables
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$s/home" XDG_RUNTIME_DIR="$s/run" XDG_CONFIG_HOME="$s/home/.config" \
        XDG_DATA_HOME="$s/home/.local/share" XDG_STATE_HOME="$s/home/.local/state" XDG_CACHE_HOME="$s/home/.cache" \
        LANG=C.UTF-8 ${icds:+VK_DRIVER_FILES="$icds"} \
        xvfb-run -a -s "-screen 0 1920x1080x24" dbus-run-session -- bash -c '
            "$1/AppRun" > "$2/out" 2> "$2/err" & gui=$!
            for _ in $(seq "$3"); do
                xdotool search --onlyvisible --name "^Neural Forge$" >/dev/null 2>&1 && { echo mapped > "$2/state"; break; }
                kill -0 "$gui" 2>/dev/null || break
                sleep 0.5
            done
            sleep 2
            kill -0 "$gui" 2>/dev/null && echo running >> "$2/state"
            kill "$gui" 2>/dev/null; wait "$gui" 2>/dev/null || true' _ "$appdir" "$s" "${SMOKE_GUI_WAIT:-60}" 2> "$s/session" || true
    state="$(read_state "$s/state")"
    why="$(verdict "$s/err")"
}
start_once
tries="first start"
# The toolkit's X11 code has been seen to hang at start on Xvfb before its window shows (the process alive, nothing
# on stderr): that one case gets a second start, and the output says so.
if [ "$(outcome "$state" "$why")" = retry ]; then
    echo "smoke-gui: no window on the first start (the process was alive, stderr clean): starting it once more"
    start_once
    tries="second start, the first showed no window"
fi
if [ "$(outcome "$state" "$why")" != pass ]; then
    echo "smoke-gui: FAILED on the $tries (${state:-no window}; ${why:-stderr clean})" >&2
    echo "  the GUI's stderr:" >&2
    sed 's/^/    /' "$s/err" >&2 || true
    echo "  the session's (Xvfb, D-Bus):" >&2
    sed 's/^/    /' "$s/session" >&2 || true
    exit 1
fi
echo "smoke-gui: window mapped, main loop running, stderr clean ($tries)"
