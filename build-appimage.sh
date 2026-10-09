#!/usr/bin/env bash
# build-appimage.sh — build the Neural Forge AppImage.
# Run from the repo root on Ubuntu (the GitHub Actions runner), as an ordinary user: it writes
# only inside the checkout. Only its from-scratch dependency install below needs root.
#
# Packaging is appimagetool run directly on a hand-built AppDir, plus zsyncmake for the
# update sidecar. GTK 4 and libadwaita come from the host system and are not bundled.
set -euo pipefail

# LIBDIR/LIB/MANIFEST are the layer's install identity (VK_LAYER_neuralforge_neural, libneural_forge_layer.so,
# lib/neural-forge/); NAME is the user-facing binary/AppImage name. See AGENTS.md "Naming convention".
LIBDIR="neural-forge"
LIB="libneural_forge_layer.so"
MANIFEST="neural_forge_layer.json"
NAME="neural-forge"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
ARCH="x86_64"
BUILD_DIR="build-appimage"
APPDIR="$BUILD_DIR/AppDir"

echo "==> Building $NAME $VERSION AppImage"

# ── Build dependencies ────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1 || ! pkg-config --exists gtk4 2>/dev/null; then
    echo "==> Installing build dependencies"
    # Tolerate an unrelated third-party repo (e.g. a runner image's preinstalled
    # Google Chrome source) failing to refresh -- apt falls back to its cached index
    # for that repo and still refreshes everything else; only `apt-get install`
    # failing on a package we actually need should be fatal.
    apt-get update -qq || true
    # g++ brings the static libstdc++ that crates/native links; curl and python3 are what
    # scripts/fetch-native-tools.sh and the crate's build script run.
    apt-get install -y -qq cargo rustc g++ curl python3 libgtk-4-dev libadwaita-1-dev \
        pkg-config wget desktop-file-utils zsync
fi
# The pinned glslang, Vulkan-Headers and volk that crates/native builds with (SHA-256-checked,
# skipped when already in tools/native/).
bash scripts/fetch-native-tools.sh
# ── Release build ─────────────────────────────────────────────────────
echo "==> cargo build --release (protocol/layer/native/gui/cli)"
cargo build --release --locked

# ── AppDir layout ─────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/lib/$LIBDIR" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
         "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/vulkan/implicit_layer.d"

cp "target/release/$NAME"                      "$APPDIR/usr/bin/"
cp "target/release/$NAME-cli"                       "$APPDIR/usr/bin/"
cp "target/release/$LIB"                            "$APPDIR/usr/lib/$LIBDIR/"
sed "s#\./libneural_forge_layer\.so#../../../lib/$LIBDIR/$LIB#" \
    "data/$MANIFEST" > "$APPDIR/usr/share/vulkan/implicit_layer.d/$MANIFEST"
cp data/io.github.labj1987.NeuralForge.desktop                               "$APPDIR/usr/share/applications/"
cp data/icon.svg                                   "$APPDIR/usr/share/icons/hicolor/scalable/apps/$NAME.svg"
# The <releases> list is generated from CHANGELOG.md's version headings, and fails the
# build if the newest one is not this workspace version.
python3 scripts/sync_appdata_releases.py
cp data/io.github.labj1987.NeuralForge.appdata.xml       "$APPDIR/usr/share/metainfo/"

# Licence notices that must ship with any build: this project's own AGPL text, the GPL-3.0 text
# and RenoDX MIT notice for the OptiScaler-derived shader lineage, and the statically linked
# crates' licence listing (regenerated here so it cannot drift from Cargo.lock).
DOCDIR="$APPDIR/usr/share/doc/$NAME"
mkdir -p "$DOCDIR"
python3 scripts/gen_third_party_crates.py
cp LICENSE ATTRIBUTION.md THIRD_PARTY_CRATES.md "$DOCDIR/"
cp -r third_party "$DOCDIR/"

# Top-level AppImage requirements
cp data/io.github.labj1987.NeuralForge.desktop "$APPDIR/"
cp data/icon.svg "$APPDIR/$NAME.svg"

# ── AppRun ────────────────────────────────────────────────────────────
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/neural-forge" "$@"
APPRUN
chmod 755 "$APPDIR/AppRun"

# ── appimagetool ──────────────────────────────────────────────────────
# Pinned to a released version and verified by checksum (the `continuous` tag moves
# under us, so two builds of the same commit could use different tools). To bump:
# pick a release at https://github.com/AppImage/appimagetool/releases, and take the
# x86_64 asset's sha256 from that release page.
APPIMAGETOOL_VERSION="1.9.1"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
TOOL="$BUILD_DIR/appimagetool-$APPIMAGETOOL_VERSION"
if [[ ! -f "$TOOL" ]] || ! echo "$APPIMAGETOOL_SHA256  $TOOL" | sha256sum -c --status; then
    echo "==> Downloading appimagetool $APPIMAGETOOL_VERSION"
    wget -q -O "$TOOL" \
        "https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-x86_64.AppImage"
    if ! echo "$APPIMAGETOOL_SHA256  $TOOL" | sha256sum -c --status; then
        echo "error: appimagetool $APPIMAGETOOL_VERSION does not match the pinned checksum" >&2
        rm -f "$TOOL"
        exit 1
    fi
    chmod +x "$TOOL"
fi

# The runtime appimagetool puts in front of the squashfs. Without --runtime-file it downloads
# the moving `continuous` build at pack time, so it is pinned and checked the same way.
# To bump: pick a release at https://github.com/AppImage/type2-runtime/releases and take the
# sha256 of its runtime-x86_64 asset (download it and run sha256sum).
RUNTIME_VERSION="20251108"
RUNTIME_SHA256="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"
RUNTIME="$BUILD_DIR/runtime-x86_64-$RUNTIME_VERSION"
if [[ ! -f "$RUNTIME" ]] || ! echo "$RUNTIME_SHA256  $RUNTIME" | sha256sum -c --status; then
    echo "==> Downloading type2-runtime $RUNTIME_VERSION"
    wget -q -O "$RUNTIME" \
        "https://github.com/AppImage/type2-runtime/releases/download/$RUNTIME_VERSION/runtime-x86_64"
    if ! echo "$RUNTIME_SHA256  $RUNTIME" | sha256sum -c --status; then
        echo "error: type2-runtime $RUNTIME_VERSION does not match the pinned checksum" >&2
        rm -f "$RUNTIME"
        exit 1
    fi
fi

echo "==> Packing AppImage"
OUT="neural-forge-$VERSION-$ARCH.AppImage"

# Use the canonical renamed repository for release updates.
UPDATE_INFORMATION="gh-releases-zsync|labj1987|neural-forge|latest|neural-forge-*-x86_64.AppImage.zsync"
VERSION="$VERSION" ARCH="$ARCH" "$TOOL" --appimage-extract-and-run \
    --runtime-file "$RUNTIME" -u "$UPDATE_INFORMATION" "$APPDIR" "$OUT"

echo "==> Done: $OUT"
ls -lh "$OUT"

# appimagetool's built-in zsync generation silently no-ops on GitHub Actions runners,
# so the sidecar is built directly instead. Fatal in CI (CI is set): the AppImage's
# UPDATE_INFORMATION points at a .zsync, so a release without one cannot update. A
# local build without zsyncmake only warns.
#
# `-u <url>` here is a *second*, different piece of update metadata than
# `UPDATE_INFORMATION` above: it's the .zsync file's own internal "URL:" header,
# read by whatever HTTP client actually fetches the new AppImage bytes once a zsync
# client has decided (via UPDATE_INFORMATION's gh-releases-zsync scheme) that an
# update exists. Without it, zsyncmake defaults to a bare relative filename, which
# only resolves correctly if a client does real relative-URL resolution against
# wherever it fetched this .zsync from -- not guaranteed. Point it at this exact
# release's real, absolute GitHub download URL instead of relying on that.
ZSYNC_URL="https://github.com/labj1987/neural-forge/releases/download/v$VERSION/$OUT"
echo "==> Generating .zsync sidecar"
if ! command -v zsyncmake >/dev/null 2>&1; then
    if [[ -n "${CI:-}" ]]; then
        echo "error: zsyncmake not found (install the zsync package)" >&2
        exit 1
    fi
    echo "==> WARNING: zsyncmake not found — continuing without .zsync"
elif zsyncmake -u "$ZSYNC_URL" -o "$OUT.zsync" "$OUT"; then
    echo "==> .zsync generated: $OUT.zsync"
elif [[ -n "${CI:-}" ]]; then
    echo "error: zsyncmake failed" >&2
    exit 1
else
    echo "==> WARNING: zsyncmake failed — continuing without .zsync"
fi
