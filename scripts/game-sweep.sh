#!/bin/bash
# The pre-release game sweep: runs the tested games installed on this machine through their unattended routes
# (scripts/game-sweep.toml) and prints a pass/fail table from the layer's status, its log and the kernel log, against
# what data/games.toml says of each game. The work is done by crates/supervisor/examples/game_sweep.rs, built here.
# Run it on the test machine, in tmux (a run takes minutes per game), with the build under test:
#
#   scripts/game-sweep.sh --layer target/release/libneural_forge_layer.so --cli target/release/neural-forge-cli
#
# Options (passed through):
#   --only LABEL,...  only these runs
#   --layer SO        put this layer in place of the installed one for the sweep (restored, hash-checked)
#   --cli PATH        the neural-forge-cli whose `shmctl status` reads the layer's channel (the installed one by
#                     default; the build's own when --layer is a build with another channel version)
#   --out DIR         where the logs, status samples and results.txt go (default ~/nf-sweep/<unix time>)
#   --lock FILE       the GPU lock taken for the whole sweep (default $NF_GPU_LOCK, else ~/scratch/gpu.lock)
#   --vw PATH         the screen recorder (`vw record`/`vw analyze`, default ~/scratch/vw/bin/vw): each run is recorded for
#                     10 s once at its placement and passes only when the analysis flags nothing (colour flashes,
#                     alternating frames, black frames, freezes); look at each run's <label>-vw/sheet.jpg too
#   --no-record       no recording: every run then fails on its picture
#   --dry-run         print each run's launch options, settings edits and expectations; change nothing
#   --config FILE     another run list
#
# It shuts Steam down to write the games' launch options and starts it again, edits the game settings a run names,
# and puts all of it back afterwards, also after an error or Ctrl-C. Exit status 0 only when every run passed and
# everything was restored.
#
# usage: game-sweep.sh [options]
#        game-sweep.sh --self-test
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
if [ "${1:-}" = "--self-test" ]; then
    bash -n "$0" || { echo "syntax error in $0" >&2; exit 1; }
    (cd "$repo" && cargo run -q -p neural-forge-supervisor --example game-sweep -- --self-test) || exit 1
    exit 0
fi
case "${1:-}" in -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;; esac
(cd "$repo" && cargo build -q --release -p neural-forge-supervisor --example game-sweep) || exit 1
exec "${CARGO_TARGET_DIR:-$repo/target}/release/examples/game-sweep" "$@"
