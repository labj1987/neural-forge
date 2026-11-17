#!/usr/bin/env bash
# Tests scripts/game-reg.sh with stand-ins for the CLI, steam, sleep and journalctl, under a scratch HOME and
# XDG_RUNTIME_DIR (it never starts a game):
# - a game already running (a process matching the pattern) is refused and left running, and `enabled` is not set;
# - a run puts the layer's `enabled` back as it was, and quits only the game it started;
# - Steam gets the session's XAUTHORITY, else GNOME's Xwayland's, else ~/.Xauthority, else none.
# usage: bash scripts/test_game_reg.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
game=""
cleanup() { [ -z "$game" ] || kill "$game" 2>/dev/null || true; rm -rf "$tmp"; }
trap cleanup EXIT
mkdir -p "$tmp/bin" "$tmp/home" "$tmp/run"
fail() { echo "test_game_reg: $*" >&2; exit 1; }
stub() { printf '#!/bin/sh\n%s\n' "$2" > "$tmp/bin/$1"; chmod 755 "$tmp/bin/$1"; }
# The CLI: `enabled` kept in $tmp/enabled, every call logged.
stub cli "echo \"\$*\" >> '$tmp/cli.log'
case \"\$2\" in
status) echo \"enabled=\$(cat '$tmp/enabled')\";;
set) echo \"\$4\" > '$tmp/enabled';;
capture) exit 1;;
esac"
# steam starts the "game" (a process whose command line holds the pattern) and says which X authority it got.
pat="nf-game-reg-test-$$"
stub steam "echo \"\${XAUTHORITY-unset}\" > '$tmp/xauth'; (exec -a $pat-run /bin/sleep 300) >/dev/null 2>&1 &"
stub sleep "exit 0"
stub journalctl "exit 0"
run() { env "$@" HOME="$tmp/home" XDG_RUNTIME_DIR="$tmp/run" NF_REG_CLI="$tmp/bin/cli" PATH="$tmp/bin:$PATH" bash "$ROOT/scripts/game-reg.sh" t1 123 "$pat" 0; }

# A game already running: refused, still running, nothing set.
(exec -a "$pat-mine" /bin/sleep 300) &
game=$!
echo 0 > "$tmp/enabled"
if out=$(run 2>&1); then fail "a run started while the game was running: $out"; fi
[[ $out == *"already running"* ]] || fail "no reason given: $out"
kill -0 "$game" 2>/dev/null || fail "the game that was already running was closed"
[ ! -e "$tmp/cli.log" ] || fail "the CLI was called: $(cat "$tmp/cli.log")"
kill "$game"; wait "$game" 2>/dev/null || true; game=""

# A run: enabled back to 0, the game it started quit, XAUTHORITY passed on.
run XAUTHORITY="$tmp/session-auth" > /dev/null 2>&1 || fail "the run failed"
[ "$(cat "$tmp/enabled")" = 0 ] || fail "enabled left at $(cat "$tmp/enabled")"
grep -qx 'shmctl set enabled 1' "$tmp/cli.log" || fail "enabled was not set for the run"
[ "$(cat "$tmp/xauth")" = "$tmp/session-auth" ] || fail "steam got $(cat "$tmp/xauth"), not XAUTHORITY"
for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -f -- "$pat-run" > /dev/null || break; /bin/sleep 0.2; done
if pgrep -f -- "$pat-run" > /dev/null; then pkill -f -- "$pat-run"; fail "the game the run started was not quit"; fi

# No XAUTHORITY: GNOME's Xwayland's newest, else ~/.Xauthority, else none (and no failure).
touch "$tmp/run/.mutter-Xwaylandauth.A"; /bin/sleep 0.01; touch "$tmp/run/.mutter-Xwaylandauth.B"; touch "$tmp/home/.Xauthority"
run -u XAUTHORITY > /dev/null 2>&1 || fail "the run failed with GNOME's X authority"
[ "$(cat "$tmp/xauth")" = "$tmp/run/.mutter-Xwaylandauth.B" ] || fail "steam got $(cat "$tmp/xauth"), not the newest mutter file"
rm "$tmp/run/.mutter-Xwaylandauth."*
run -u XAUTHORITY > /dev/null 2>&1 || fail "the run failed with ~/.Xauthority"
[ "$(cat "$tmp/xauth")" = "$tmp/home/.Xauthority" ] || fail "steam got $(cat "$tmp/xauth"), not ~/.Xauthority"
rm "$tmp/home/.Xauthority"
run -u XAUTHORITY > /dev/null 2>&1 || fail "the run failed without any X authority"
[ "$(cat "$tmp/xauth")" = unset ] || fail "steam got $(cat "$tmp/xauth") with no X authority anywhere"
pkill -f -- "$pat-run" || true
echo "test_game_reg: ok"
