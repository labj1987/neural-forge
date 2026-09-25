#!/usr/bin/env bash
# Builds the AppImage from the working tree and puts exactly that build where the rig runs it:
# the launcher copy (~/AppImages/neural-forge.appimage) and the installed layer/helper, then
# proves the installed files are byte-identical to the ones inside the AppImage. Leaves nothing
# behind: no backup of the previous AppImage (releases are the rollback) and no temp files.
#
# Usage: scripts/deploy-rig.sh [host]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HOST="${1:-lordnikon}"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
APPIMAGE="neural-forge-$VERSION-x86_64.AppImage"

echo "==> building $APPIMAGE"
CARGO_HELPER="${CARGO_HELPER:-cargo +stable}" bash build-appimage.sh >/tmp/deploy-rig-build.log 2>&1 || { tail -20 /tmp/deploy-rig-build.log; exit 1; }

echo "==> copying to $HOST"
scp -q "$APPIMAGE" "$HOST:/tmp/nf-deploy.AppImage"
# The local build only exists to be copied; releases come from CI, so don't let builds pile up here.
rm -rf "$APPIMAGE" "$APPIMAGE.zsync" build-appimage
ssh "$HOST" "VERSION=$VERSION bash -s" <<'REMOTE'
set -eo pipefail
  mkdir -p ~/AppImages
  cp /tmp/nf-deploy.AppImage ~/AppImages/neural-forge.appimage.new && chmod +x ~/AppImages/neural-forge.appimage.new
  mv -f ~/AppImages/neural-forge.appimage.new ~/AppImages/neural-forge.appimage
  # A version-specific extraction directory, so nothing else's squashfs-root in /tmp is
  # touched or reused.
  X=/tmp/nf-deploy-$VERSION
  rm -rf "$X" && mkdir -p "$X"
  (cd "$X" && ~/AppImages/neural-forge.appimage --appimage-extract >/dev/null)
  "$X/squashfs-root/usr/bin/neural-forge-cli" install --appdir "$X/squashfs-root" | tail -1
  L=~/.local/share/neural-forge/lib/neural-forge; S=$X/squashfs-root/usr/lib/neural-forge
  ok=1
  for f in libneural_forge_layer.so helper/neural-forge-helper.exe; do
    a=$(sha256sum "$L/$f" | cut -d" " -f1); b=$(sha256sum "$S/$f" | cut -d" " -f1)
    if [ "$a" = "$b" ]; then echo "  $f  ${a:0:12}  installed == AppImage"; else echo "  $f MISMATCH"; ok=0; fi
  done
  rm -rf "$X" /tmp/nf-deploy.AppImage
  # Keep the menu entry's version metadata (written when the AppImage was integrated) in step.
  E=~/.local/share/applications/neural-forge.desktop
  [ -f "$E" ] && sed -i "s/^X-AppImage-Version=.*/X-AppImage-Version=$VERSION/" "$E"
  update-desktop-database ~/.local/share/applications 2>/dev/null || true
  [ $ok = 1 ]
REMOTE
rm -f /tmp/deploy-rig-build.log
echo "==> deployed and verified"
