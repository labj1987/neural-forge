#!/usr/bin/env bash
# Checks a layer log for GPU stalls the layer caught and recovered from instead of freezing
# the game: bounded fence-wait timeouts and the breadcrumb dump logged with each one. Also
# summarizes engage/disengage transitions (loading-screen boundaries) and the presented
# frame rate.
#
# Usage: scripts/check-stalls.sh [LOG]
#   LOG defaults to $NEURAL_FORGE_LOG, then ~/nf-layer.log (the file the game was launched
#   with NEURAL_FORGE_LOG pointing at).
# Exit status: 0 = no stalls found, 1 = stalls found, 2 = no log to check.
set -uo pipefail

LOG="${1:-${NEURAL_FORGE_LOG:-$HOME/nf-layer.log}}"
if [ ! -f "$LOG" ]; then
  echo "no log at $LOG -- launch the game with NEURAL_FORGE_LOG=$LOG first, or pass the log path" >&2
  exit 2
fi

count() { grep -cF -- "$1" "$LOG"; }

timeouts=$(count '[layer] fence wait timed out')
dumps=$(count '[breadcrumbs] ')

echo "== fence wait timeouts: $timeouts =="
if [ "$timeouts" -gt 0 ]; then
  # Each timeout is followed by its breadcrumb dump: the stage markers that led up to it.
  grep -n -A 70 -F '[layer] fence wait timed out' "$LOG" | head -90
fi
echo "== breadcrumb dumps: $dumps =="
echo
echo "== engage/disengage transitions (loading-screen boundaries) =="
echo "$(count 'game is rendering steadily; engaging') engage, $(count 'passing frames through untouched') disengage"
echo
echo "== presented frame rate =="
present=$(grep -F '[present] ' "$LOG")
if [ -n "$present" ]; then
  echo "$(wc -l <<<"$present") sample(s); lowest and latest:"
  sed -E 's/.*\[present\] ([0-9.]+) fps.*/\1/' <<<"$present" | sort -n | head -1 | sed 's/^/  lowest: /;s/$/ fps/'
  tail -1 <<<"$present" | sed 's/.*\[present\] /  latest: /'
else
  echo "no [present] lines (the layer logs one every 5 s while presenting)"
fi

if [ "$timeouts" -gt 0 ] || [ "$dumps" -gt 0 ]; then
  echo
  echo "STALLS FOUND"
  exit 1
fi
echo
echo "no stalls found"
