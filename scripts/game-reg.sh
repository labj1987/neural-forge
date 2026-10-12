#!/bin/bash
# game-reg.sh <label> <appid> <exe-pattern> <boot-s> <steps...>: one run of a Steam game with the layer
# (its own launch options): launch, wait <boot-s>, send controller steps (a step "W<n>" waits n s),
# measure 60 s in play, capture the presented frame, quit. Prints holds, presents, timeouts, Xid.
set -euo pipefail
if [ $# -lt 4 ]; then
    echo "usage: game-reg.sh <label> <appid> <exe-pattern> <boot-s> <steps...>" >&2
    exit 2
fi
label=$1; appid=$2; pat=$3; boot=$4; shift 4
case $label in ''|.|..|*/*) echo "error: label must be a plain name (no '/', not empty)" >&2; exit 2;; esac
uid=$(id -u)
out=$HOME/nf-spike/runs/$label; mkdir -p "$out"; C=$HOME/.local/share/neural-forge/bin/neural-forge-cli
pad(){ timeout 5 sh -c 'echo "$1" > /tmp/nf-pad.fifo' sh "$1" || echo "warning: controller step $1 failed" >&2; sleep 1; }
start=$(date "+%F %T"); echo "$start" > "$out/start"

# Ends the game however this script ends (a failed step under set -e included): `pgrep -f` also
# matches this script (and the shell running it), whose own command line holds the pattern, so this
# process, everything above it and anything it forked itself are skipped.
quit_game() {
    local mine=" " p=$$
    while [ "$p" -gt 1 ]; do
        mine="$mine$p "
        p=$(ps -o ppid= -p "$p" | tr -d ' ')
        [ -n "$p" ] || break
    done
    for p in $(pgrep -f -- "$pat" || true); do
        case $mine in *" $p "*) continue;; esac
        [ -e "/proc/$p" ] || continue
        [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "$$" ] && continue
        kill -TERM "$p" 2>/dev/null || true
    done
}
trap quit_game EXIT
"$C" shmctl set enabled 1 >/dev/null
XA=$(ls -t "/run/user/$uid"/.mutter-Xwaylandauth.* | head -1)
DISPLAY=:0 XAUTHORITY=$XA timeout 10 steam -applaunch "$appid" >/dev/null 2>&1 || true
sleep "$boot"
for s in "$@"; do case $s in W*) sleep "${s#W}";; *) pad "$s";; esac; done
m=$(date "+%F %T"); sleep 60
# This run's capture only: a file older than the marker is an earlier run's, never copied as this one's.
marker=$out/.capture-marker; touch "$marker"
if timeout 40 "$C" shmctl capture >"$out/capture.txt" 2>&1; then
    shot=$(find "$HOME/.local/share/neural-forge/captures" -newer "$marker" -name '*-original.png' 2>/dev/null | sort | head -1)
    if [ -n "$shot" ]; then cp "$shot" "$out/end.png"; else echo "warning: the capture wrote no original frame" >&2; fi
else
    echo "warning: capture failed: $(tail -1 "$out/capture.txt")" >&2
fi
rm -f "$marker"
journalctl --user -u steam-relaunch --since "$start" -o cat > "$out/launch.log" || true

quit_game
sleep 8
L=$out/launch.log; W=$(journalctl --user -u steam-relaunch --since "$m" --until "$(date "+%F %T")" -o cat || true)
echo "== $label"
echo "$W" | grep "mode=model" | tail -1 | sed -E "s/.*(holds=[0-9]+).*(misses=[0-9]+ \(total [0-9]+\)).*(holds_per_s=[0-9.]+).*/\1 \2 \3/" || true
echo "$W" | grep "\[present\]" | sed -E "s/.*\] ([0-9.]+) fps \(([0-9.]+)\/s.*/\1 \2/" | awk '{f+=$1; c+=$2; n++} END {if (n) printf "presents %.1f/s composited %.1f/s over %d windows\n", f/n, c/n, n}' || true
echo "timeouts $(grep -c "fence wait timed out" "$L" || true) Xid $(journalctl -k --since "$start" --no-pager | grep -c Xid || true)"
