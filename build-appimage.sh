#!/usr/bin/env bash
# build-appimage.sh — build the Neural Forge AppImage.
# Run from the repo root on Ubuntu (the GitHub Actions runner), as an ordinary user: it writes
# only inside the checkout (and, on a machine without Rust, rustup's ~/.cargo and ~/.rustup).
# Only its from-scratch system package install below needs root, through sudo.
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
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi
# Every tool the build runs (here, in scripts/fetch-native-tools.sh, the crates' build scripts and the
# Python generators), so a missing one is installed before the long release build, not found after it.
missing_deps() {
    local tool
    for tool in g++ curl patch python3 pkg-config desktop-file-validate zsyncmake; do
        command -v "$tool" >/dev/null 2>&1 || { echo "$tool"; return 0; }
    done
    pkg-config --exists gtk4 libadwaita-1 2>/dev/null || { echo "gtk4/libadwaita-1"; return 0; }
    return 1
}
if missing="$(missing_deps)"; then
    echo "==> Installing build dependencies ($missing is missing)"
    # Tolerate an unrelated third-party repo (e.g. a runner image's preinstalled
    # Google Chrome source) failing to refresh -- apt falls back to its cached index
    # for that repo and still refreshes everything else; only `apt-get install`
    # failing on a package we actually need should be fatal.
    $SUDO apt-get update -qq || true
    # g++ brings the static libstdc++ that crates/native links; curl, patch and python3 are what
    # scripts/fetch-native-tools.sh and the crate's build script run.
    $SUDO apt-get install -y -qq g++ curl patch python3 libgtk-4-dev libadwaita-1-dev \
        pkg-config desktop-file-utils zsync
    if missing="$(missing_deps)"; then
        echo "error: $missing is still missing after installing the build dependencies" >&2
        exit 1
    fi
fi
# Rust from rustup, at the version rust-toolchain.toml pins (the distribution's cargo is too old for
# Cargo.lock and the code): rustup reads the file and installs that toolchain on first use.
if ! command -v rustup >/dev/null 2>&1; then
    echo "==> Installing rustup (the toolchain comes from rust-toolchain.toml)"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)"
rustup run "$TOOLCHAIN" rustc --version >/dev/null 2>&1 || rustup toolchain install "$TOOLCHAIN"
# The pinned glslang, Vulkan-Headers, volk and patched OpenDLSS-NR that crates/native builds with
# (SHA-256-checked, skipped when already in tools/native/).
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

RELEASE="${CARGO_TARGET_DIR:-target}/release"
cp "$RELEASE/$NAME"                                 "$APPDIR/usr/bin/"
cp "$RELEASE/$NAME-cli"                             "$APPDIR/usr/bin/"
cp "$RELEASE/$LIB"                                  "$APPDIR/usr/lib/$LIBDIR/"
sed "s#\./libneural_forge_layer\.so#../../../lib/$LIBDIR/$LIB#" \
    "data/$MANIFEST" > "$APPDIR/usr/share/vulkan/implicit_layer.d/$MANIFEST"
cp data/io.github.labj1987.NeuralForge.desktop                               "$APPDIR/usr/share/applications/"
cp data/icon.svg                                   "$APPDIR/usr/share/icons/hicolor/scalable/apps/$NAME.svg"
# The <releases> list is generated from CHANGELOG.md's version headings, and fails the
# build if the newest one is not this workspace version.
python3 scripts/sync_appdata_releases.py
cp data/io.github.labj1987.NeuralForge.appdata.xml       "$APPDIR/usr/share/metainfo/"

# Licence notices that must ship with any build: this project's own AGPL text, the GPL-3.0 text
# and RenoDX MIT notice for the OptiScaler-derived shader lineage (third_party/optiscaler), the
# OpenDLSS-NR patches and their reasons, the statically linked crates' listing (regenerated here so
# it cannot drift from Cargo.lock), and the full licence and notice texts of every linked crate and
# of OpenDLSS-NR, volk and the Vulkan-Headers compiled into the layer, collected from their sources.
DOCDIR="$APPDIR/usr/share/doc/$NAME"
mkdir -p "$DOCDIR"
python3 scripts/gen_third_party_crates.py
python3 scripts/gen_third_party_crates.py --licences "$DOCDIR/THIRD_PARTY_LICENSES.txt"
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
# Kept in .cache/, outside the build directory this script wipes, so a second build reuses it.
TOOL_DIR=".cache"
mkdir -p "$TOOL_DIR"
TOOL="$TOOL_DIR/appimagetool-$APPIMAGETOOL_VERSION"
if [[ ! -f "$TOOL" ]] || ! echo "$APPIMAGETOOL_SHA256  $TOOL" | sha256sum -c --status; then
    echo "==> Downloading appimagetool $APPIMAGETOOL_VERSION"
    curl -fsSL --retry 3 -o "$TOOL" \
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
RUNTIME="$TOOL_DIR/runtime-x86_64-$RUNTIME_VERSION"
if [[ ! -f "$RUNTIME" ]] || ! echo "$RUNTIME_SHA256  $RUNTIME" | sha256sum -c --status; then
    echo "==> Downloading type2-runtime $RUNTIME_VERSION"
    curl -fsSL --retry 3 -o "$RUNTIME" \
        "https://github.com/AppImage/type2-runtime/releases/download/$RUNTIME_VERSION/runtime-x86_64"
    if ! echo "$RUNTIME_SHA256  $RUNTIME" | sha256sum -c --status; then
        echo "error: type2-runtime $RUNTIME_VERSION does not match the pinned checksum" >&2
        rm -f "$RUNTIME"
        exit 1
    fi
fi

echo "==> Packing AppImage"
OUT="neural-forge-$VERSION-$ARCH.AppImage"

# Reproducible packing: the same commit packs to the same bytes (and the same .zsync), so a published
# release can be checked against a local rebuild. Every file in the AppDir gets the commit's time and
# umask-independent permissions, and SOURCE_DATE_EPOCH gives appimagetool's mksquashfs the same
# filesystem time (appimagetool also packs every file root-owned).
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct 2>/dev/null || true)}"
if [[ -z "$SOURCE_DATE_EPOCH" ]]; then
    echo "error: no git commit to take the AppImage's timestamps from; set SOURCE_DATE_EPOCH" >&2
    exit 1
fi
export SOURCE_DATE_EPOCH
chmod -R u+rwX,go+rX,go-w "$APPDIR"
find "$APPDIR" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +

# Use the canonical renamed repository for release updates.
UPDATE_INFORMATION="gh-releases-zsync|labj1987|neural-forge|latest|neural-forge-*-x86_64.AppImage.zsync"
VERSION="$VERSION" ARCH="$ARCH" "$TOOL" --appimage-extract-and-run \
    --runtime-file "$RUNTIME" -u "$UPDATE_INFORMATION" "$APPDIR" "$OUT"

# The .zsync records the AppImage's mtime (its MTime: header): the commit's too, so it is reproducible.
touch -d "@$SOURCE_DATE_EPOCH" "$OUT"
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
