set -eo pipefail
  # A version-specific extraction directory, so nothing else's squashfs-root in /tmp is
  # touched or reused. Removed with the copied AppImage however the script ends.
  X=/tmp/nf-deploy-$VERSION
  trap 'rm -rf "$X" /tmp/nf-deploy.AppImage' EXIT
  mkdir -p ~/AppImages
  cp /tmp/nf-deploy.AppImage ~/AppImages/neural-forge.appimage.new && chmod +x ~/AppImages/neural-forge.appimage.new
  mv -f ~/AppImages/neural-forge.appimage.new ~/AppImages/neural-forge.appimage
  rm -rf "$X" && mkdir -p "$X"
  (cd "$X" && ~/AppImages/neural-forge.appimage --appimage-extract >/dev/null)
  "$X/squashfs-root/usr/bin/neural-forge-cli" install --appdir "$X/squashfs-root" | tail -1
  # Where the CLI installs: $XDG_DATA_HOME/neural-forge, ~/.local/share when it is unset.
  D="${XDG_DATA_HOME:-$HOME/.local/share}"
  I=$D/neural-forge; S=$X/squashfs-root/usr
  ok=1
  for f in lib/neural-forge/libneural_forge_layer.so bin/neural-forge bin/neural-forge-cli; do
    a=$(sha256sum "$I/$f" 2>/dev/null | cut -d" " -f1) || a=""
    b=$(sha256sum "$S/$f" | cut -d" " -f1)
    if [ -z "$a" ]; then echo "  $f MISSING from $I"; ok=0
    elif [ "$a" = "$b" ]; then echo "  $f  ${a:0:12}  installed == AppImage"
    else echo "  $f MISMATCH"; ok=0; fi
  done
  # Keep the menu entry's version metadata (written when the AppImage was integrated) in step.
  E=$D/applications/neural-forge.desktop
  [ -f "$E" ] && sed -i "s/^X-AppImage-Version=.*/X-AppImage-Version=$VERSION/" "$E"
  update-desktop-database "$D/applications" 2>/dev/null || true
  [ $ok = 1 ]
