#!/usr/bin/env bash
# Checks the dependency graph with cargo-deny (deny.toml): RustSec advisories, licences and sources. Runs
# the pinned cargo-deny release below, fetched once (SHA-256 checked) into tools/ (or NEURAL_FORGE_NATIVE_TOOLS);
# scripts/check_pins.py reads the pin. The advisory database is fetched fresh on every run.
#   bash scripts/cargo-deny.sh [cargo-deny check arguments]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${NEURAL_FORGE_NATIVE_TOOLS:-$ROOT/tools/native}"
CARGO_DENY_VERSION="0.20.2"
URL="https://github.com/EmbarkStudios/cargo-deny/releases/download/$CARGO_DENY_VERSION/cargo-deny-$CARGO_DENY_VERSION-x86_64-unknown-linux-musl.tar.gz"
SHA="9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f"
DEST="$TOOLS/cargo-deny"

if [ ! -x "$DEST/cargo-deny" ] || [ "$(cat "$DEST/.fetched-sha256" 2>/dev/null)" != "$SHA" ]; then
  mkdir -p "$TOOLS"
  stage="$(mktemp -d "$TOOLS/.fetch-cargo-deny.XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  echo "cargo-deny: downloading $URL"
  curl -fsSL --retry 3 -o "$stage/archive.tar.gz" "$URL"
  got="$(sha256sum "$stage/archive.tar.gz" | cut -d' ' -f1)"
  [ "$got" = "$SHA" ] || { echo "cargo-deny: SHA-256 mismatch: expected $SHA, got $got" >&2; exit 1; }
  tar -xzf "$stage/archive.tar.gz" -C "$stage"
  top=("$stage"/cargo-deny-*/)
  echo "$SHA" >"${top[0]}/.fetched-sha256"
  rm -rf "$DEST"
  mv "${top[0]}" "$DEST"
  rm -rf "$stage"
  trap - EXIT
fi

cd "$ROOT"
"$DEST/cargo-deny" --locked check "$@"
