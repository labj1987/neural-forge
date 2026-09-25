#!/usr/bin/env bash
# Automated end-to-end test against the real rig, with no human in the loop.
#
# Why this exists: every earlier round of testing needed Alex to install something,
# launch the game, look at the screen and report back, and half of those rounds were
# wasted because the two halves of the system came from different builds -- the layer
# hand-copied to the installed path while the helper ran from whatever stale AppImage
# happened to be mounted. This script builds both halves, installs both halves,
# forces the helper to come from the installed path (NEURAL_FORGE_INSTALL_DIR wins in
# `supervisor::install_dir`'s search order), runs the game, and reads the answers out
# of the shared-memory header and the layer's own log.
#
# What it can verify without anyone looking at a screen:
#   * the helper's own self-tests, locally, before anything is deployed
#     (scripts/helper-test.sh: native unit tests, guard_test, spoof_test, spoof_install_test)
#   * whether the proxy encode is actually dispatching  ([encode] active)
#   * whether the GPU encode matches the CPU reference  ([encode] self-check)
#   * which composition mode the resolve picked
#   * real presented frames per second, from the layer's own frame counter
#   * helper round-trip timings, and whether any Xid landed in the kernel log
#
# What it still cannot judge: how the picture *looks*. Shimmer is temporal and
# subjective; this measures everything around it so that a human look is the last step
# rather than every step.
#
# Usage: scripts/rig-test.sh [host] [seconds_to_sample]
set -uo pipefail

HOST="${1:-lordnikon}"
SAMPLE="${2:-20}"
APP_ID=3240220
INSTALL_LIB="/home/alex/.local/share/neural-forge/lib/neural-forge"
SHM="/tmp/neural-forge-1000/shm.bin"

say() { printf '\n== %s\n' "$*"; }

# The CLI ships inside the AppImage mount, whose path changes on every app launch.
# Resolved once, remotely, rather than hardcoded.
remote_cli() {
    ssh "$HOST" 'ls -d /tmp/.mount_neural*/usr/bin/neural-forge-cli ~/.local/share/neural-forge/bin/neural-forge-cli 2>/dev/null | head -1'
}

sh_remote() { ssh "$HOST" "$@"; }

say "helper self-tests on this machine (native unit tests, then the SEH guard and the spoof under Wine)"
# Local and cheap: a helper that fails these would only fail on the rig in a harder-to-read way.
bash "$(dirname "${BASH_SOURCE[0]}")/helper-test.sh" || exit 1

say "building and installing both halves through the installer (scripts/deploy-rig.sh)"
# Never copy files into the installed path directly: the installer refuses to overwrite
# files its record says it didn't write, so a direct copy breaks the next real install.
bash "$(dirname "${BASH_SOURCE[0]}")/deploy-rig.sh" "$HOST" || exit 1

CLI="$(remote_cli)"
if [ -z "$CLI" ]; then
    echo "!! no neural-forge-cli found in any AppImage mount -- is Neural Forge installed on $HOST?" >&2
    exit 1
fi

say "stopping the previous game and helper"
# `pgrep -f X[.]exe` rather than `pgrep -f X.exe`: the pattern as written does not
# match this very command's own argv, which is how an earlier run of this killed its
# own ssh session. Kill by PID, never `pkill -f`.
sh_remote "P=\$(pgrep -f 'GTA5_Enhanced[.]exe' | head -1); [ -n \"\$P\" ] && { kill -TERM \$P; sleep 6; kill -KILL \$P 2>/dev/null; }; true"
sh_remote "export NEURAL_FORGE_SHM=$SHM NEURAL_FORGE_UID=1000; '$CLI' stop >/dev/null 2>&1; true"

say "starting the helper from the INSTALLED path (not the AppImage's own copy)"
sh_remote "export NEURAL_FORGE_SHM=$SHM NEURAL_FORGE_UID=1000 NEURAL_FORGE_INSTALL_DIR=$INSTALL_LIB; '$CLI' start 2>&1 | head -5"

say "test settings"
# working_scale 1.0 is required for the encode self-check: the proxy is only
# pixel-aligned with the frame at exactly 1.0, and the check refuses to compare
# different rasters rather than reporting nonsense.
sh_remote "export NEURAL_FORGE_SHM=$SHM NEURAL_FORGE_UID=1000; for kv in 'enabled 1' 'apply_model 1' 'working_scale 1.0'; do '$CLI' shmctl set \$kv; done"

say "launching the game"
sh_remote "export DISPLAY=:0 WAYLAND_DISPLAY=wayland-0 XDG_RUNTIME_DIR=/run/user/1000 \
             DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
             XAUTHORITY=\$(ls -t /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1); \
           setsid nohup /home/alex/.local/share/Steam/ubuntu12_32/steam -applaunch $APP_ID >/tmp/rig-test-launch.log 2>&1 & sleep 3; echo launched"

say "waiting for the Rockstar launcher to actually start the game"
# Measured: the launcher's own startup plus its cloud-save sync took over six minutes,
# which used to be charged against the game's load time and made the wait below time
# out roughly a minute after the game finally started. The two waits are separate now.
sh_remote "for i in \$(seq 1 120); do
    if pgrep -f 'GTA5_Enhanced[.]exe' >/dev/null 2>&1; then echo \"game process up after \$((i*5))s\"; exit 0; fi
    sleep 5
  done; echo 'TIMED OUT: the launcher never started the game'; exit 1"

say "waiting for the layer to present frames (up to 10 minutes from game start)"
# Frames flow in GTA's menu too: this does NOT mean gameplay. Gameplay needs someone at
# the rig to choose Story Mode, so treat unattended numbers as menu numbers.
# The layer's own frame counter moving is the only reliable "we are really rendering and
# really capturing" signal -- the game can be up, windowed and burning CPU while capture
# never triggers (see the pass_through/GENERAL gate in device.rs).
sh_remote "export NEURAL_FORGE_SHM=$SHM NEURAL_FORGE_UID=1000
  start=\$('$CLI' shmctl status | awk -F= '/^layer_frames=/{print \$2}')
  for i in \$(seq 1 120); do
    sleep 5
    now=\$('$CLI' shmctl status | awk -F= '/^layer_frames=/{print \$2}')
    if [ \"\$now\" != \"\$start\" ]; then echo \"frames flowing after \$((i*5))s (layer_frames \$start -> \$now)\"; exit 0; fi
  done
  echo 'TIMED OUT: no presented frames'; exit 1"
FRAMES_OK=$?

say "sampling for ${SAMPLE}s"
sh_remote "export NEURAL_FORGE_SHM=$SHM NEURAL_FORGE_UID=1000
  a=\$('$CLI' shmctl status | awk -F= '/^layer_frames=/{print \$2}')
  h0=\$('$CLI' shmctl status | awk -F= '/^helper_frames=/{print \$2}')
  sleep $SAMPLE
  b=\$('$CLI' shmctl status | awk -F= '/^layer_frames=/{print \$2}')
  h1=\$('$CLI' shmctl status | awk -F= '/^helper_frames=/{print \$2}')
  echo \"presented fps: \$(( (b - a) / $SAMPLE ))\"
  echo \"helper round trips/s: \$(( (h1 - h0) / $SAMPLE ))\"
  '$CLI' shmctl status | grep -E 'helper_state|model_up|helper_upload_ms|helper_eval_ms|helper_readback_ms|layer_ms|working_scale|max_ratio|transfer_strength|colour_strength|reversible_mode'"

say "why capture did or did not trigger"
sh_remote "journalctl --since '25 min ago' -o cat 2>/dev/null | grep -E 'present skipped' | tail -3 || echo 'no skip reasons logged (capture triggered normally)'"

say "encode verification, straight from the layer's own log"
sh_remote "journalctl --since '15 min ago' -o cat 2>/dev/null | grep -E '\[encode\]' | tail -5 || echo 'no [encode] lines -- the encode never ran'"

say "composition admission and mode"
sh_remote "journalctl --since '15 min ago' -o cat 2>/dev/null | grep -E 'capture admission declined|swapchain .* pass_through' | tail -4"

say "GPU health (any Xid means a driver-level fault, not a composition bug)"
sh_remote "journalctl -k --since '15 min ago' 2>/dev/null | grep -iE 'xid|gpu has fallen' | tail -5 || echo 'no Xid errors'"

say "done (game left running; pass 'kill' as the third argument to stop it)"
if [ "${3:-}" = "kill" ]; then
    sh_remote "P=\$(pgrep -f 'GTA5_Enhanced[.]exe' | head -1); [ -n \"\$P\" ] && kill -TERM \$P; true"
fi
exit $FRAMES_OK
