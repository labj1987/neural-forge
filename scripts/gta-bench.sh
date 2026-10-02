#!/bin/bash
# Unattended GTA V Enhanced built-in benchmark with a chosen layer environment.
#
# Runs on the rig itself, or from the dev machine with --host, which pipes this script
# over ssh. Steam must be running on the rig. The game is launched through the same
# Steam Linux Runtime entry point and Proton that Steam uses, with only the environment
# given here (Steam's launch options are bypassed), and exits by itself after one
# benchmark iteration. Results land in $NF_BENCH_DIR/<label> on the rig: GTA's
# Benchmark/FrameTimes files, MangoHud's per-frame CSV (MangoHud is always the last
# layer, i.e. below any frame generator, so it counts displayed frames), nvidia-smi
# samples, and the launch log, which holds the layer's [sync] and [present] lines.
# Summarise with scripts/bench-report.py.
#
# usage: gta-bench.sh [--host HOST] [--set key=value]... <label> <VK layers above MangoHud> [env...]
#
#   --set key=value  set a live Neural Forge setting with `neural-forge-cli shmctl set`
#                    before the launch (e.g. --set working_scale=0.66 --set model_interval=2)
#                    and put the previous value back after the run, confirmed with
#                    `shmctl status`. The model resolution must never be left below 1.0.
#
# Examples (NR off, then NR on at model interval 2, both with GTA's script mods off):
#   gta-bench.sh --host lordnikon p0-nroff-1 '' 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
#   gta-bench.sh --host lordnikon --set model_interval=2 p0-nron-1 VK_LAYER_neuralforge_neural \
#       NEURAL_FORGE_ENABLE=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
#
# Rig paths can be overridden with NF_BENCH_STEAM_LIBRARY (the library holding GTA),
# NF_BENCH_STEAM_ROOT (the Steam client), NF_BENCH_PROTON and NF_BENCH_DIR.
# GTA sometimes exits early at Game Init, with or without Neural Forge; the script says
# "launch exited early" and leaves no benchmark.txt. Wait five minutes and run it again.
set -u

if [ "${1:-}" = "--host" ]; then
    host=$2; shift 2
    args=$(printf '%q ' "$@")
    exec ssh "$host" "bash -s -- $args" < "$0"
fi

sets=()
while [ "${1:-}" = "--set" ]; do sets+=("$2"); shift 2; done
if [ $# -lt 2 ]; then
    sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
label=$1; mid=$2; shift 2

S=${NF_BENCH_STEAM_LIBRARY:-/mnt/Storage/Steam}
C=${NF_BENCH_STEAM_ROOT:-$HOME/.local/share/Steam}
PROTON=${NF_BENCH_PROTON:-$C/compatibilitytools.d/Proton-CachyOS Latest}
GAME="$S/steamapps/common/Grand Theft Auto V Enhanced"
SLR=$S/steamapps/common/SteamLinuxRuntime_4
DOCS="$S/steamapps/compatdata/3240220/pfx/drive_c/users/steamuser/Documents/Rockstar Games/GTAV Enhanced/Benchmarks"
CLI=$HOME/.local/share/neural-forge/bin/neural-forge-cli
shmctl() { NEURAL_FORGE_SHM=/tmp/neural-forge-$(id -u)/shm.bin NEURAL_FORGE_UID=$(id -u) "$CLI" shmctl "$@"; }
status_of() { shmctl status | sed -n "s/^$1=//p"; }

out=${NF_BENCH_DIR:-$HOME/nf-spike/gta}/$label; rm -rf "$out"; mkdir -p "$out"

restore=()
for kv in "${sets[@]}"; do
    k=${kv%%=*}
    restore+=("$k=$(status_of "$k")")
    shmctl set "$k" "${kv#*=}" >/dev/null || { echo "shmctl set $kv failed" >&2; exit 1; }
    echo "set $k=$(status_of "$k")" | tee -a "$out/settings"
done
put_back() {
    for kv in "${restore[@]}"; do
        k=${kv%%=*}
        shmctl set "$k" "${kv#*=}" >/dev/null
        echo "restored $k=$(status_of "$k")"
    done
}
trap put_back EXIT
if [ -n "$(ss -tn state established '( sport = :3390 )' 2>/dev/null | tail -n +2)" ]; then
    echo "warning: a remote desktop session is connected (port 3390); it costs about 8 fps" | tee -a "$out/settings"
fi

before=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
layers=${mid:+$mid:}VK_LAYER_MANGOHUD_overlay_x86_64
uid=$(id -u)
start=$(date +%s); date "+%F %T" > "$out/start"
setsid env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin LANG="${LANG:-C.UTF-8}" \
  DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/$uid \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$uid/bus \
  XAUTHORITY="$(ls -t /run/user/$uid/.mutter-Xwaylandauth.* | head -1)" \
  SteamAppId=3240220 SteamGameId=3240220 SteamOverlayGameId=3240220 STEAM_COMPAT_APP_ID=3240220 \
  STEAM_COMPAT_CLIENT_INSTALL_PATH="$C" STEAM_COMPAT_DATA_PATH="$S/steamapps/compatdata/3240220" \
  STEAM_COMPAT_INSTALL_PATH="$GAME" STEAM_COMPAT_LIBRARY_PATHS="$S/steamapps:$C/steamapps" \
  STEAM_COMPAT_TOOL_PATHS="$PROTON:$SLR" STEAM_COMPAT_SHADER_PATH="$S/steamapps/shadercache/3240220" \
  VK_INSTANCE_LAYERS="$layers" VK_LOADER_DEBUG=layer \
  MANGOHUD_CONFIG="output_folder=$out,autostart_log=1,log_duration=3600,log_interval=0" \
  "$@" \
  "$SLR/_v2-entry-point" --verb=waitforexitandrun -- "$PROTON/proton" waitforexitandrun "$GAME/PlayGTAV.exe" \
  -benchmark -benchmarkIterations 1 -benchmarkFrameTimes > "$out/launch.log" 2>&1 &
pg=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > "$out/gpu.csv" 2>/dev/null &
smi=$!
echo "launched pgid=$pg"
for i in $(seq 1 240); do  # up to 20 min for launcher + load + benchmark
  sleep 5
  new=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
  if [ -n "$new" ] && [ "$new" != "$before" ]; then echo "result $new after $(( $(date +%s)-start ))s"; sleep 5; break; fi
  if ! kill -0 $pg 2>/dev/null; then echo "launch exited early after $(( $(date +%s)-start ))s"; break; fi
  [ $((i % 12)) = 0 ] && echo "  waiting $(( $(date +%s)-start ))s: game=$(pgrep -c -x GTA5_Enhanced.e)"
done
sleep 3; new=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
date "+%F %T" > "$out/end"; kill $smi
G=$(pgrep -x GTA5_Enhanced.e); [ -n "$G" ] && kill -TERM $G; sleep 15
kill -TERM -- -$pg 2>/dev/null; sleep 5; kill -KILL -- -$pg 2>/dev/null
if [ -n "$new" ] && [ "$new" != "$before" ]; then
  ts=${new##*/Benchmark-}; ts=${ts%.txt}; cp "$new" "$out/benchmark.txt"
  for f in "$DOCS"/FrameTimes-Pass*-$ts.txt; do cp "$f" "$out/${f##*/FrameTimes-}"; done
fi
ls "$out"
