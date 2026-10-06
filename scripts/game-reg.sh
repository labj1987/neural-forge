#!/bin/bash
# game-reg.sh <label> <appid> <exe-pattern> <boot-s> <steps...>: one run of a Steam game with the layer
# (its own launch options): launch, wait <boot-s>, send controller steps (a step "W<n>" waits n s),
# measure 60 s in play, capture the presented frame, quit. Prints holds, presents, timeouts, Xid.
set -u
label=$1; appid=$2; pat=$3; boot=$4; shift 4
out=~/nf-spike/runs/$label; mkdir -p $out; C=~/.local/share/neural-forge/bin/neural-forge-cli
pad(){ timeout 5 sh -c "echo $1 > /tmp/nf-pad.fifo"; sleep 1; }
start=$(date "+%F %T"); echo "$start" > $out/start
$C shmctl set enabled 1 >/dev/null
XA=$(ls -t /run/user/1000/.mutter-Xwaylandauth.* | head -1); DISPLAY=:0 XAUTHORITY=$XA timeout 10 steam -applaunch $appid >/dev/null 2>&1
sleep $boot
for s in "$@"; do case $s in W*) sleep ${s#W};; *) pad $s;; esac; done
m=$(date "+%F %T"); sleep 60
timeout 10 $C shmctl capture >/dev/null 2>&1; sleep 3
d=$(ls -td ~/.local/share/neural-forge/captures/series-* ~/.local/share/neural-forge/captures/*-original.png 2>/dev/null | head -1)
[ -d "$d" ] && cp "$d/000000-original.png" $out/end.png || cp "$d" $out/end.png
journalctl --user -u steam-relaunch --since "$start" -o cat > $out/launch.log
for p in $(pgrep -f "$pat"); do kill -TERM $p; done; sleep 8
L=$out/launch.log; W=$(journalctl --user -u steam-relaunch --since "$m" --until "$(date "+%F %T")" -o cat)
echo "== $label"
echo "$W" | grep "mode=model" | tail -1 | sed -E "s/.*(holds=[0-9]+).*(misses=[0-9]+ \(total [0-9]+\)).*(holds_per_s=[0-9.]+).*/\1 \2 \3/"
echo "$W" | grep "\[present\]" | sed -E "s/.*\] ([0-9.]+) fps \(([0-9.]+)\/s.*/\1 \2/" | awk "{f+=\$1; c+=\$2; n++} END {if (n) printf \"presents %.1f/s composited %.1f/s over %d windows\n\", f/n, c/n, n}"
echo "timeouts $(grep -c "fence wait timed out" $L) Xid $(journalctl -k --since "$start" --no-pager | grep -c Xid)"
