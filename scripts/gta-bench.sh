#!/bin/bash
# Unattended GTA V Enhanced built-in benchmark with a chosen layer environment.
#
# Runs on the rig itself, or from another machine with --host, which pipes this script
# over ssh. Steam must be running on the rig. The game is launched through the same
# Steam Linux Runtime entry point and Proton that Steam uses, with only the environment
# given here (Steam's launch options are bypassed), and exits by itself after one
# benchmark iteration. Results land in $NF_BENCH_DIR/<label> on the rig: GTA's
# Benchmark/FrameTimes files, MangoHud's per-frame CSV (MangoHud is always the last
# layer, i.e. below any frame generator, so it counts displayed frames), nvidia-smi
# samples, and the launch log, which holds the layer's [sync] and [present] lines.
# Summarise with scripts/bench-report.py.
#
# usage: gta-bench.sh [--host HOST] [--tries N] [--set key=value]... <label> <VK layers above MangoHud> [env...]
#        gta-bench.sh --self-test
#
#   --set key=value  set a live Neural Forge setting with `neural-forge-cli shmctl set`
#                    before the launch (e.g. --set working_scale=0.66 --set model_interval=2)
#                    and put the previous value back after the run, confirmed with
#                    `shmctl status`. The model resolution must never be left below 1.0.
#   --tries N        launches allowed for one result (default 3). A launch that does not end in a
#                    result is classified, its evidence kept in <label>/attempt-K/, and the game is
#                    launched again: an early exit (GTA's crash dump read: its hang watchdog, which
#                    kills the game after its main loop has stalled for 60 s, or another fault; an
#                    Xid in the kernel log names a GPU hang), a launch where the game never started,
#                    a game that stopped presenting (thread states saved), and a stuck launch (the
#                    game presenting with no result after NF_BENCH_STUCK_S, default 480 s; a normal
#                    run takes 260-280 s). Every attempt is listed in <label>/attempts.
#
# Frame generation is not retried. GTA turns it off by itself in some launches (Streamline reports
# eDLSSGStatusFailReflexNotDetectedAtRuntime at a benchmark scene change, with or without Neural Forge,
# launched by Steam or by this script; docs/LESSONS.md). With frame generation on in settings.xml the
# runner says when the displayed rate fell to a third of its best for 20 s ("frame generation
# stopped"), writes it to <label>/fg, and bench-report.py gives the measured multiplier.
#
# Examples (NR off, then NR on at model interval 2, both with GTA's script mods off):
#   gta-bench.sh --host rig p0-nroff-1 '' 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
#   gta-bench.sh --host rig --set model_interval=2 p0-nron-1 VK_LAYER_neuralforge_neural \
#       NEURAL_FORGE_ENABLE=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
#
# Rig paths can be overridden with NF_BENCH_STEAM_LIBRARY (the library holding GTA),
# NF_BENCH_STEAM_ROOT (the Steam client), NF_BENCH_PROTON, NF_BENCH_DIR, and NF_BENCH_CLI (the
# neural-forge-cli --set uses; the installed one by default, a development build's when the
# layer under test has a newer shared-memory version than the installed CLI can open).
# A frame series the layer captured during the run (NEURAL_FORGE_CAPTURE_AT=<resume>:<ms>:<frames>,
# for scripts/bench-iq.py) is moved into the run's directory.
set -u

# The checks that read GTA's crash dumps and MangoHud's log, in one Python program (sent along
# with --host). gta_py dumps <CrashLogs dir> <epoch>: one line per dump written since <epoch>.
# gta_py fg <MangoHud csv> <cap fps>: "stopped <s> <shown> <best>" or "kept". gta_py self-test.
gta_py() {
    python3 -I - "$@" <<'PY'
import glob, os, struct, sys

WATCHDOG = 0x14012c6eb  # GTA5_Enhanced.exe (2026-09-15 build): the instruction after its hang watchdog's call


def dump_cause(data):
    """What a GTA minidump says: the faulting site and whether it is the hang watchdog."""
    if len(data) < 32 or data[:4] != b"MDMP":
        return "empty dump (the game died while writing it)"
    _, _, nstreams, rva_dir = struct.unpack_from("<IIII", data, 0)
    streams = {}
    for i in range(nstreams):
        kind, _, rva = struct.unpack_from("<III", data, rva_dir + 12 * i)
        streams[kind] = rva
    if 6 not in streams:
        return "dump without an exception record"
    rva = streams[6]
    code, addr = struct.unpack_from("<I", data, rva + 8)[0], struct.unpack_from("<Q", data, rva + 24)[0]
    ctx = struct.unpack_from("<I", data, rva + 8 + 156)[0]
    rsp = struct.unpack_from("<Q", data, ctx + 0x98)[0]
    ret = None
    for kind in (5, 9):  # the memory lists: the word at rsp is the return address of a call
        if kind not in streams:
            continue
        r = streams[kind]
        if kind == 5:
            n = struct.unpack_from("<I", data, r)[0]
            ranges = [struct.unpack_from("<QII", data, r + 4 + 16 * i) for i in range(n)]
        else:
            n, off = struct.unpack_from("<QQ", data, r)
            ranges = []
            for i in range(n):
                start, size = struct.unpack_from("<QQ", data, r + 16 + 16 * i)
                ranges.append((start, size, off))
                off += size
        for start, size, at in ranges:
            if start <= rsp < start + size - 8:
                ret = struct.unpack_from("<Q", data, at + rsp - start)[0]
    site = f"GTA5_Enhanced.exe+{addr - 0x140000000:#x}" if 0x140000000 <= addr < 0x146000000 else hex(addr)
    if addr == WATCHDOG:
        return f"GTA's hang watchdog (main loop stalled 60 s, {site})"
    if ret == WATCHDOG:
        return f"GTA's hang watchdog (main loop stalled 60 s, stage handler {site})"
    return f"exception {code:#010x} at {site}"


def dumps(folder, since):
    for path in sorted(glob.glob(os.path.join(folder, "*.dmp")), key=os.path.getmtime):
        if os.path.getmtime(path) >= since:
            print(f"{os.path.basename(path)}: {dump_cause(open(path, 'rb').read())}")


def fg_rule(times, cap):
    """Frame generation stopped: the displayed rate over 10 s windows (every 5 s) under a third of the best
    window (ignoring loading-screen bursts above cap) for 20 s, once the best passed 150 fps."""
    best, low, t = 0.0, 0, 10.0
    while times and t <= times[-1]:
        n = sum(1 for x in times if t - 10 <= x < t) / 10
        if best < n <= cap:
            best = n
        if best >= 150 and n < best / 3:
            low += 1
            if low == 4:
                return f"stopped {t - 25:.0f} {n:.0f} {best:.0f}"
        else:
            low = 0
        t += 5
    return "kept"


def fg(csv, cap):
    rows = [r for r in open(csv, errors="replace").read().splitlines()[3:] if r.count(",") > 3]
    times = []
    for r in rows:
        try:
            times.append(int(r.split(",")[-1]) / 1e9)
        except ValueError:
            pass
    print(fg_rule(times, float(cap)))


def self_test():
    def minidump(addr, top_of_stack):
        # header, a stream directory of 2 (exception, memory list), the CONTEXT, one stack range
        body = bytearray(0x600)
        struct.pack_into("<4sIIIIIQ", body, 0, b"MDMP", 0xA793, 2, 32, 0, 0, 0)
        struct.pack_into("<III", body, 32, 6, 168, 0x40)
        struct.pack_into("<III", body, 44, 5, 20, 0x100)
        struct.pack_into("<IIIIQQ", body, 0x40, 1, 0, 0xC0000005, 0, 0, addr)
        struct.pack_into("<II", body, 0x40 + 8 + 152, 1232, 0x200)
        struct.pack_into("<Q", body, 0x200 + 0x98, 0x4710FEA8)
        struct.pack_into("<IQII", body, 0x100, 1, 0x4710FEA8, 16, 0x500)
        struct.pack_into("<Q", body, 0x500, top_of_stack)
        return bytes(body)
    assert dump_cause(minidump(WATCHDOG, 0)).startswith("GTA's hang watchdog (main loop stalled 60 s, GTA5_Enhanced.exe+0x12c6eb")
    assert "stage handler GTA5_Enhanced.exe+0x951c50" in dump_cause(minidump(0x140951C50, WATCHDOG))
    assert dump_cause(minidump(0x140123456, 0x140000000)) == "exception 0xc0000005 at GTA5_Enhanced.exe+0x123456"
    assert dump_cause(b"").startswith("empty dump")
    frames = lambda start, end, fps: [start + i / fps for i in range(int((end - start) * fps))]
    assert fg_rule(frames(0, 70, 60) + frames(70, 260, 300), 576) == "kept"
    assert fg_rule(frames(0, 70, 60) + frames(70, 75, 2000) + frames(75, 260, 300), 576) == "kept"
    lost = fg_rule(frames(0, 70, 60) + frames(70, 130, 300) + frames(130, 260, 80), 576)
    assert lost.startswith("stopped 13") and lost.endswith(" 80 300"), lost
    # a dip of a few seconds (a scene change) is not a stop
    assert fg_rule(frames(0, 70, 60) + frames(70, 110, 300) + frames(110, 118, 80) + frames(118, 260, 300), 576) == "kept"
    print("gta-bench self-test ok")


cmd = sys.argv[1]
if cmd == "dumps":
    dumps(sys.argv[2], float(sys.argv[3]))
elif cmd == "fg":
    fg(sys.argv[2], sys.argv[3])
elif cmd == "self-test":
    self_test()
PY
}

if [ "${1:-}" = "--host" ]; then
    host=$2; shift 2
    args=$(printf '%q ' "$@")
    exec ssh "$host" "bash -s -- $args" < "$0"
fi

if [ "${1:-}" = "--self-test" ]; then
    bash -n "$0" 2>/dev/null || [ ! -f "$0" ] || { echo "syntax error in $0" >&2; exit 1; }
    gta_py self-test || exit 1
    for bad in '' . .. a/b; do
        out=$(NF_BENCH_DIR=/nonexistent "$0" "$bad" '' 2>&1); rc=$?
        [ $rc = 2 ] && [[ $out == *"label must be"* ]] || { echo "label '$bad' accepted" >&2; exit 1; }
    done
    echo "gta-bench label checks ok"
    exit 0
fi

sets=(); tries=3
while :; do
    case "${1:-}" in
        --set) sets+=("$2"); shift 2;;
        --tries) tries=$2; shift 2;;
        *) break;;
    esac
done
if [ $# -lt 2 ]; then
    sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
label=$1; mid=$2; shift 2
# The label names a directory that is deleted and recreated below: never empty, never a path.
case $label in
    ''|.|..|*/*) echo "error: label must be a plain name (not empty, no '/')" >&2; exit 2;;
esac
case $tries in ''|*[!0-9]*|0) echo "error: --tries takes a count of 1 or more" >&2; exit 2;; esac

C=${NF_BENCH_STEAM_ROOT:-$HOME/.local/share/Steam}
S=${NF_BENCH_STEAM_LIBRARY:-$C}
PROTON=${NF_BENCH_PROTON:-$C/compatibilitytools.d/Proton-CachyOS Latest}
GAME="$S/steamapps/common/Grand Theft Auto V Enhanced"
SLR=$S/steamapps/common/SteamLinuxRuntime_4
PFX="$S/steamapps/compatdata/3240220"
USERDIR="$PFX/pfx/drive_c/users/steamuser"
DOCS="$USERDIR/Documents/Rockstar Games/GTAV Enhanced/Benchmarks"
CRASH="$USERDIR/AppData/Local/Rockstar Games/GTAV Enhanced/CrashLogs"
SETTINGS="$USERDIR/Documents/Rockstar Games/GTAV Enhanced/settings.xml"
CLI=${NF_BENCH_CLI:-$HOME/.local/share/neural-forge/bin/neural-forge-cli}
STUCK_S=${NF_BENCH_STUCK_S:-480}
shmctl() { NEURAL_FORGE_SHM=/tmp/neural-forge-$(id -u)/shm.bin NEURAL_FORGE_UID=$(id -u) timeout 30 "$CLI" shmctl "$@"; }
status_of() { shmctl status | sed -n "s/^$1=//p"; }

for exe in GTA5_Enhanced.e PlayGTAV.exe; do
    if pgrep -x "$exe" >/dev/null; then
        echo "error: GTA is already running ($exe, pid $(pgrep -x "$exe" | tr '\n' ' ')); not launching a second one" >&2
        exit 1
    fi
done
[ -x "$SLR/_v2-entry-point" ] || { echo "error: no Steam Linux Runtime 4 at $SLR (set NF_BENCH_STEAM_LIBRARY)" >&2; exit 1; }

out=${NF_BENCH_DIR:-$HOME/nf-spike/gta}/$label; rm -rf "$out"; mkdir -p "$out"

restore=()
# Puts every --set back and checks it took: a value left changed (the CLI can no longer open the
# channel, say) is an error and fails the run, whatever its result.
put_back() {
    local kv k want got failed=0
    for kv in "${restore[@]}"; do
        k=${kv%%=*}; want=${kv#*=}
        shmctl set "$k" "$want" >/dev/null 2>&1 || true
        got=$(status_of "$k")
        if [ "$got" = "$want" ]; then
            echo "restored $k=$got"
        else
            echo "error: $k is ${got:-unreadable} after the run, not the previous $want: set it back by hand" >&2
            failed=1
        fi
    done
    [ $failed = 0 ] || exit 1
}
# Installed before the first set, so a failed later --set still puts back the earlier ones.
trap put_back EXIT
for kv in "${sets[@]}"; do
    k=${kv%%=*}
    old=$(status_of "$k")
    # Without the value it has now it could not be put back.
    [ -n "$old" ] || { echo "error: shmctl status shows no $k; not changing it" >&2; exit 1; }
    restore+=("$k=$old")
    shmctl set "$k" "${kv#*=}" >/dev/null || { echo "shmctl set $kv failed" >&2; exit 1; }
    echo "set $k=$(status_of "$k")" | tee -a "$out/settings"
done
if [ -n "$(ss -tn state established '( sport = :3390 )' 2>/dev/null | tail -n +2)" ]; then
    echo "warning: a remote desktop session is connected (port 3390); it costs about 8 fps" | tee -a "$out/settings"
fi
fg_on=0
grep -q '<FrameGenType value="1"' "$SETTINGS" 2>/dev/null && fg_on=1
refresh=$(sed -n 's/.*<RefreshRate value="\([0-9]*\)".*/\1/p' "$SETTINGS" 2>/dev/null)

layers=${mid:+$mid:}VK_LAYER_MANGOHUD_overlay_x86_64
uid=$(id -u)
G=; pg=; smi=

# Kills what one launch left: the game, the launch's process group, and the Wine processes of
# GTA's prefix started since the launch (winedevice.exe spins a core each when left behind).
end_launch() {
    local since=$1 p
    [ -n "$smi" ] && kill "$smi" 2>/dev/null; smi=
    G=$(pgrep -x GTA5_Enhanced.e); [ -n "$G" ] && kill -TERM $G; sleep 15
    [ -n "$pg" ] && { kill -TERM -- -$pg 2>/dev/null; sleep 5; kill -KILL -- -$pg 2>/dev/null; }
    for p in $(pgrep -u "$uid" -f '\.exe|wineserver|wine64|wine-preloader'); do
        [ "$p" = "$$" ] && continue
        grep -qa "STEAM_COMPAT_DATA_PATH=$PFX" /proc/$p/environ 2>/dev/null || continue
        [ $(( $(date +%s) - $(ps -o etimes= -p $p 2>/dev/null || echo 999999) )) -ge "$since" ] || continue
        kill -KILL $p 2>/dev/null
    done
}

# Every thread of the game: name, state, kernel wait channel and CPU ticks, for a game that stopped
# presenting (readable without privileges).
thread_states() {
    local t
    for t in /proc/$1/task/*; do
        printf '%s %s state=%s wchan=%s utime=%s stime=%s\n' "${t##*/}" "$(cat $t/comm 2>/dev/null)" \
            "$(awk '{print $3}' $t/stat 2>/dev/null)" "$(cat $t/wchan 2>/dev/null)" \
            "$(awk '{print $14}' $t/stat 2>/dev/null)" "$(awk '{print $15}' $t/stat 2>/dev/null)"
    done
}

# One launch. Sets $why to "result" or to what went wrong.
launch_once() {
    local before new i now csv age stalled_at= note_fg=0 fgline at shown best seen=
    before=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
    start=$(date +%s); date "+%F %T" > "$out/start"
    setsid env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin LANG="${LANG:-C.UTF-8}" \
      DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/$uid \
      DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$uid/bus \
      XAUTHORITY="$(ls -t /run/user/$uid/.mutter-Xwaylandauth.* | head -1)" \
      SteamAppId=3240220 SteamGameId=3240220 SteamOverlayGameId=3240220 STEAM_COMPAT_APP_ID=3240220 \
      STEAM_COMPAT_CLIENT_INSTALL_PATH="$C" STEAM_COMPAT_DATA_PATH="$PFX" \
      STEAM_COMPAT_INSTALL_PATH="$GAME" STEAM_COMPAT_LIBRARY_PATHS="$S/steamapps:$C/steamapps" \
      STEAM_COMPAT_TOOL_PATHS="$PROTON:$SLR" STEAM_COMPAT_SHADER_PATH="$S/steamapps/shadercache/3240220" \
      VK_INSTANCE_LAYERS="$layers" VK_LOADER_DEBUG=layer \
      MANGOHUD_CONFIG="output_folder=$out,autostart_log=1,log_duration=3600,log_interval=0" \
      "$@" \
      "$SLR/_v2-entry-point" --verb=waitforexitandrun -- "$PROTON/proton" waitforexitandrun "$GAME/PlayGTAV.exe" \
      -benchmark -benchmarkIterations 1 -benchmarkFrameTimes > "$out/launch.log" 2>&1 < /dev/null &
    pg=$!
    nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > "$out/gpu.csv" 2>/dev/null &
    smi=$!
    echo "launched pgid=$pg"
    why=
    for i in $(seq 1 $(( STUCK_S / 5 ))); do
        sleep 5; now=$(date +%s)
        new=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
        if [ -n "$new" ] && [ "$new" != "$before" ]; then echo "result $new after $(( now - start ))s"; why=result; sleep 5; break; fi
        G=$(pgrep -x GTA5_Enhanced.e | head -1)
        if ! kill -0 $pg 2>/dev/null; then
            if [ -n "$stalled_at" ]; then why="the game stopped presenting at ${stalled_at}s and exited after $(( now - start ))s"
            else why="launch exited early after $(( now - start ))s"; fi
            break
        fi
        if [ -z "$G" ]; then
            if [ -n "$seen" ]; then
                # The game is gone but the launch is not (the launcher or its crash reporter lingers).
                [ $(( now - seen )) -ge 30 ] && { why="the game exited after $(( seen - start ))s"; break; }
            elif [ $(( now - start )) -ge 120 ]; then
                why="the game did not start within $(( now - start ))s (the launcher is still running)"; break
            fi
            continue
        fi
        seen=$now
        csv=$(ls -t "$out"/GTA5_Enhanced_*[0-9].csv 2>/dev/null | head -1)
        [ -n "$csv" ] || continue
        # MangoHud writes a line per presented frame: a log that stops growing is a game that stopped presenting.
        age=$(( now - $(stat -c %Y "$csv") ))
        if [ $age -ge 15 ] && [ -z "$stalled_at" ]; then
            stalled_at=$(( now - start - age ))
            echo "  the game stopped presenting at ${stalled_at}s; GTA's own watchdog ends it 60 s later"
            thread_states $G > "$out/stalled-threads.txt"
        elif [ $age -ge 90 ]; then
            why="the game stopped presenting at ${stalled_at}s and was still stalled after $(( now - start ))s"; break
        elif [ $age -lt 15 ] && [ -n "$stalled_at" ]; then
            echo "  presenting again after $(( now - start ))s"; stalled_at=
        fi
        if [ $fg_on = 1 ] && [ $note_fg = 0 ] && [ $(( i % 2 )) = 0 ]; then
            fgline=$(gta_py fg "$csv" $(( 2 * ${refresh:-288} )) 2>/dev/null)
            case $fgline in stopped*)
                read -r _ at shown best <<< "$fgline"
                echo "  frame generation stopped at about ${at}s (displayed $shown fps, $best before); GTA's own, not retried" | tee "$out/fg"
                note_fg=1;;
            esac
        fi
        [ $((i % 12)) = 0 ] && echo "  waiting $(( now - start ))s: game=$(pgrep -c -x GTA5_Enhanced.e)"
    done
    if [ -z "$why" ]; then
        why="stuck: no benchmark result after $(( $(date +%s) - start ))s with the game running"
        # What the game shows: the layer's next presented frame, when Neural Forge is in the launch.
        case $mid in *VK_LAYER_neuralforge_neural*) shmctl capture > "$out/stuck-capture.txt" 2>&1;; esac
        csv=$(ls -t "$out"/GTA5_Enhanced_*[0-9].csv 2>/dev/null | head -1)
        [ -n "$csv" ] && echo "displayed fps at the end: $(tail -n 300 "$csv" | cut -d, -f1 | sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}')" >> "$out/stuck-capture.txt"
    fi
    sleep 3
    date "+%F %T" > "$out/end"
    end_launch "$start"
    if [ "$why" != result ]; then
        new=$(ls "$DOCS"/Benchmark-* 2>/dev/null | sort | tail -1)
        [ -n "$new" ] && [ "$new" != "$before" ] && why=result
    fi
    if [ "$why" = result ]; then
        ts=${new##*/Benchmark-}; ts=${ts%.txt}; cp "$new" "$out/benchmark.txt"
        for f in "$DOCS"/FrameTimes-Pass*-$ts.txt; do cp "$f" "$out/${f##*/FrameTimes-}"; done
        return
    fi
    # What the system and the game recorded about the failed launch.
    journalctl -k --since "@$start" --no-pager 2>/dev/null | grep -E 'Xid|NVRM|segfault|Out of memory|oom-kill' > "$out/kernel.txt"
    gta_py dumps "$CRASH" "$start" > "$out/crash.txt" 2>&1
    [ -s "$out/crash.txt" ] && why="$why; $(paste -sd ';' "$out/crash.txt")"
    grep -q Xid "$out/kernel.txt" && why="$why; GPU hang: $(grep -o 'Xid.*' "$out/kernel.txt" | head -1)"
}

env_args=("$@")
attempt=0
while :; do
    attempt=$((attempt + 1))
    launch_once "${env_args[@]}"
    echo "$attempt $(date '+%F %T') $why" >> "$out/attempts"
    [ "$why" = result ] && break
    echo "attempt $attempt: $why"
    mkdir -p "$out/attempt-$attempt"
    for f in "$out"/*; do [ -f "$f" ] && [ "${f##*/}" != attempts ] && [ "${f##*/}" != settings ] && mv "$f" "$out/attempt-$attempt/"; done
    [ $attempt -ge "$tries" ] && { echo "no result after $attempt launches"; break; }
    sleep 15
done
sed -n 's/.*\[capture\] capturing the next [0-9]* presents into //p' "$out/launch.log" 2>/dev/null | while read -r d; do
  [ -d "$d" ] && mv "$d" "$out/"
done
ls "$out"
[ "$why" = result ]
