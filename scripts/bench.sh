#!/usr/bin/env bash
# Repeatable GTA V Enhanced benchmark for the matched native/upstream/neural-forge
# comparison required by docs/PHASE1.md ("Preserved baseline and benchmark gate", item 2).
# Run this ON THE TARGET MACHINE (lordnikon), not the dev box that builds the layer --
# it needs the real Steam client, the real GPU, and the real game.
#
# This script cannot drive the car. It launches Steam with the right mode's
# environment, waits for the real game process, then STOPS AND WAITS for you to
# confirm you have reached the saved route/scene (see docs/HARDWARE_VALIDATION.md for
# which one) before it starts the timed sample. Only the sample window itself needs
# to be unattended.
#
# Usage: scripts/bench.sh <native|upstream|neuralforge> [duration_seconds]
#
# Needs: steam, nvidia-smi, xdotool (to trigger MangoHud's own toggle_logging hotkey
# on the game window -- this repo does not control MangoHud, so it drives the exact
# hotkey already configured in ~/.config/MangoHud/MangoHud.conf rather than guessing
# at an autostart config key). For upstream/neural-forge modes, neural-forge-cli must
# already be on PATH (see docs/PHASE1.md's install step) for the layer-side telemetry
# sample; native mode skips that half.
#
# Output: scripts/../bench-out/<mode>-<timestamp>/ containing raw samples and a
# one-line summary appended to scripts/../bench-out/summary.csv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:?usage: bench.sh <native|upstream|neuralforge> [duration_seconds]}"
DURATION="${2:-60}"
STEAM_APPID=3240220
GAME_EXE="GTA5_Enhanced.exe"
MANGOHUD_HOTKEY="shift+F2" # matches toggle_logging=Shift_L+F2 in MangoHud.conf
MANGOHUD_OUT="$HOME" # matches output_folder=/home/alex in MangoHud.conf

case "$MODE" in
    native)
        LAUNCH_ENV=(env -u NEURAL_FORGE_ENABLE -u VKLayer_DLSS5 NEURAL_FORGE_DISABLE=1)
        ;;
    upstream)
        # docs/PHASE1.md's preserved baseline. Do not change this launch option.
        LAUNCH_ENV=(env -u NEURAL_FORGE_ENABLE VKLayer_DLSS5=1 DLSSNR_DMABUF=0)
        ;;
    neuralforge)
        LAUNCH_ENV=(env -u VKLayer_DLSS5 NEURAL_FORGE_ENABLE=1 \
            "NEURAL_FORGE_TARGET_EXE=$GAME_EXE")
        ;;
    *)
        echo "unknown mode: $MODE (want native|upstream|neuralforge)" >&2
        exit 1
        ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="bench-out/${MODE}-${STAMP}"
mkdir -p "$OUT_DIR"
echo "==> mode=$MODE duration=${DURATION}s out=$OUT_DIR"

# Steam only applies a launch-time environment to processes it spawns AFTER it itself
# was started with that environment -- re-exporting vars in this shell does nothing to
# an already-running Steam client (confirmed the hard way in the 2026-09-14 session;
# see docs/HARDWARE_VALIDATION.md's "GTA comparison gate"). So always restart Steam fresh
# under the mode's environment rather than assuming inheritance.
echo "==> stopping any running Steam client"
steam -shutdown >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    pgrep -x steam >/dev/null 2>&1 || break
    sleep 1
done
pkill -x steam 2>/dev/null || true
sleep 2

echo "==> starting Steam under the $MODE environment"
"${LAUNCH_ENV[@]}" steam -silent >"$OUT_DIR/steam.log" 2>&1 &
STEAM_PID=$!
for _ in $(seq 1 60); do
    pgrep -x steam >/dev/null 2>&1 && break
    sleep 1
done
sleep 5 # let the client finish its own startup handshake before we ask it to launch anything

echo "==> launching AppID $STEAM_APPID"
"${LAUNCH_ENV[@]}" steam "steam://rungameid/$STEAM_APPID" >/dev/null 2>&1 || true

echo "==> waiting for $GAME_EXE (up to 180s -- Rockstar Games Launcher/Social Club can be slow)"
GAME_PID=""
for _ in $(seq 1 180); do
    GAME_PID="$(pgrep -f "$GAME_EXE" | head -1 || true)"
    [ -n "$GAME_PID" ] && break
    sleep 1
done
if [ -z "$GAME_PID" ]; then
    echo "!!! $GAME_EXE never appeared -- aborting this run, nothing sampled" >&2
    exit 1
fi
echo "==> $GAME_EXE is up (pid $GAME_PID)"

echo
echo "==> reach the saved benchmark route now, then press Enter to start the ${DURATION}s sample"
echo "    (see docs/HARDWARE_VALIDATION.md for which saved scene/route this run must match)"
read -r _

WINDOW_ID="$(xdotool search --name "Grand Theft Auto" | head -1 || true)"
if [ -n "$WINDOW_ID" ]; then
    xdotool windowactivate "$WINDOW_ID" 2>/dev/null || true
    xdotool key --window "$WINDOW_ID" "$MANGOHUD_HOTKEY" # start MangoHud logging
    sleep 0.3
else
    echo "!!! could not find the game window via xdotool -- MangoHud logging must be" >&2
    echo "    started by hand (Shift_L+F2) right now, within the next few seconds" >&2
fi

# Layer/helper telemetry sample, same pattern as the ad hoc collector used
# 2026-09-14 (see /tmp/collect_neuralforge.py on this machine). Native mode has no
# layer in the loop, so it only gets the GPU-side sample.
SAMPLE_JSON="$OUT_DIR/samples.jsonl"
: >"$SAMPLE_JSON"
START="$(date +%s.%N)"
for ((i = 0; i < DURATION; i++)); do
    GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "")"
    if [ "$MODE" != "native" ] && command -v neural-forge-cli >/dev/null 2>&1; then
        STATUS="$(neural-forge-cli shmctl status 2>/dev/null || echo "")"
    else
        STATUS=""
    fi
    printf '{"t":%s,"gpu":"%s","status":"%s"}\n' \
        "$(date +%s.%N)" "$GPU" "${STATUS//\"/\'}" >>"$SAMPLE_JSON"
    sleep 1
done
END="$(date +%s.%N)"

if [ -n "$WINDOW_ID" ]; then
    xdotool key --window "$WINDOW_ID" "$MANGOHUD_HOTKEY" # stop MangoHud logging
else
    echo "!!! stop MangoHud logging by hand now (Shift_L+F2) if you started it by hand" >&2
fi

echo "==> sample window done ($(python3 -c "print(f'{$END-$START:.1f}')")s actual)"

# MangoHud writes <output_folder>/MangoHud-<exe>-<timestamp>.csv on log stop, matching
# whatever the process was named at launch. Grab the newest one written since START.
MH_CSV="$(find "$MANGOHUD_OUT" -maxdepth 1 -iname 'MangoHud-*.csv' -newermt "@$START" 2>/dev/null | sort | tail -1 || true)"
if [ -n "$MH_CSV" ]; then
    cp "$MH_CSV" "$OUT_DIR/"
    echo "==> MangoHud log: $OUT_DIR/$(basename "$MH_CSV")"
else
    echo "!!! no MangoHud CSV found under $MANGOHUD_OUT since sample start -- FPS/1%-low" >&2
    echo "    for this run must be filled in by hand from the on-screen overlay" >&2
fi

# One summary line: mode, duration, MangoHud avg/1%-low fps if we found a CSV, plus
# mean GPU util/VRAM/power from our own sample. Real fps/1%-low computation from the
# MangoHud CSV (its own header format) is intentionally left to whoever reviews this
# run rather than guessed here -- do not fabricate numbers this script cannot verify.
SUMMARY_CSV="bench-out/summary.csv"
[ -f "$SUMMARY_CSV" ] || echo "mode,timestamp,duration_s,mangohud_csv,sample_jsonl,out_dir" >"$SUMMARY_CSV"
echo "$MODE,$STAMP,$DURATION,${MH_CSV:-},$SAMPLE_JSON,$OUT_DIR" >>"$SUMMARY_CSV"
echo "==> appended to $SUMMARY_CSV"
echo "==> raw samples: $SAMPLE_JSON"
