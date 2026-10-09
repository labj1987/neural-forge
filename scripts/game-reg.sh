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
"$C" shmctl set enabled 1 >/dev/null
XA=$(ls -t "/run/user/$uid"/.mutter-Xwaylandauth.* | head -1)
DISPLAY=:0 XAUTHORITY=$XA timeout 10 steam -applaunch "$appid" >/dev/null 2>&1 || true
sleep "$boot"
for s in "$@"; do case $s in W*) sleep "${s#W}";; *) pad "$s";; esac; done
m=$(date "+%F %T"); sleep 60
timeout 10 "$C" shmctl capture >/dev/null 2>&1 || true; sleep 3
d=$(ls -td "$HOME"/.local/share/neural-forge/captures/series-* "$HOME"/.local/share/neural-forge/captures/*-original.png 2>/dev/null | head -1 || true)
if [ -d "$d" ]; then cp "$d/000000-original.png" "$out/end.png"; elif [ -f "$d" ]; then cp "$d" "$out/end.png"; fi
journalctl --user -u steam-relaunch --since "$start" -o cat > "$out/launch.log" || true

# `pgrep -f` also matches this script (and the shell running it): its own command line holds the pattern.
# Skip this process, everything above it and anything it forked itself (the $(...) running pgrep).
mine=" "
p=$$
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
sleep 8
L=$out/launch.log; W=$(journalctl --user -u steam-relaunch --since "$m" --until "$(date "+%F %T")" -o cat || true)
echo "== $label"
echo "$W" | grep "mode=model" | tail -1 | sed -E "s/.*(holds=[0-9]+).*(misses=[0-9]+ \(total [0-9]+\)).*(holds_per_s=[0-9.]+).*/\1 \2 \3/" || true
echo "$W" | grep "\[present\]" | sed -E "s/.*\] ([0-9.]+) fps \(([0-9.]+)\/s.*/\1 \2/" | awk '{f+=$1; c+=$2; n++} END {if (n) printf "presents %.1f/s composited %.1f/s over %d windows\n", f/n, c/n, n}' || true
echo "timeouts $(grep -c "fence wait timed out" "$L" || true) Xid $(journalctl -k --since "$start" --no-pager | grep -c Xid || true)"
