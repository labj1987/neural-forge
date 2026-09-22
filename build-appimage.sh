#!/usr/bin/env bash
# build-appimage.sh — build the Neural Forge AppImage.
# Run from the repo root on Ubuntu (matches GreenLight/KernelPop/SteamPunk's own CI
# assumption). Run as root in CI.
#
# Unlike GreenLight/KernelPop, this app needs no polkit/pkexec step at all -- every
# path it touches (~/.local/share, ~/.config, /tmp/neural-forge-$UID/) is already
# user-owned, so AppRun just execs the GUI directly.
set -euo pipefail

# LIBDIR/LIB/MANIFEST are the layer's install identity (VK_LAYER_neuralforge_neural, libneural_forge_layer.so,
# lib/neural-forge/); NAME is the user-facing binary/AppImage name. See CLAUDE.md "Naming convention".
LIBDIR="neural-forge"
LIB="libneural_forge_layer.so"
MANIFEST="neural_forge_layer.json"
NAME="neural-forge"
VERSION="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
ARCH="x86_64"
BUILD_DIR="build-appimage"
APPDIR="$BUILD_DIR/AppDir"
WIN_TARGET="x86_64-pc-windows-gnu"

# On a from-scratch CI image, `cargo`/`rustup` are already the only Rust toolchain, so
# plain `cargo` and `rustup target add` are correct there. On this repo's own dev
# machine specifically, a pre-existing non-rustup toolchain was deliberately kept as
# the `system` default (see CLAUDE.md's "helper gotchas") and the cross-compile
# toolchain lives under a separate `stable-x86_64-unknown-linux-gnu` rustup toolchain
# instead -- override CARGO_HELPER via the environment if building there.
CARGO_HELPER="${CARGO_HELPER:-cargo}"

echo "==> Building $NAME $VERSION AppImage"

# ── Build dependencies ────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1 || ! pkg-config --exists gtk4 2>/dev/null; then
    echo "==> Installing build dependencies"
    # Tolerate an unrelated third-party repo (e.g. a runner image's preinstalled
    # Google Chrome source) failing to refresh -- apt falls back to its cached index
    # for that repo and still refreshes everything else; only `apt-get install`
    # failing on a package we actually need should be fatal.
    apt-get update -qq || true
    apt-get install -y -qq cargo rustc libgtk-4-dev libadwaita-1-dev \
        pkg-config libssl-dev wget file desktop-file-utils zsync \
        mingw-w64 gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64 gcc-multilib
fi
# Add the target to whichever toolchain `$CARGO_HELPER` will actually invoke, not
# necessarily the default one -- on a clean CI image there's exactly one toolchain and
# `+something` never appears in `$CARGO_HELPER`, so this reduces to the obvious
# `rustup target add`. On this repo's own dev machine, the default toolchain is a
# linked, non-rustup-managed one that `rustup target add` can't touch at all (see
# CLAUDE.md) -- `CARGO_HELPER="cargo +stable-..."` there means the target has to be
# added to *that* toolchain instead.
rustup_toolchain_arg=""
for word in $CARGO_HELPER; do
    case "$word" in
        +*) rustup_toolchain_arg="--toolchain ${word#+}" ;;
    esac
done
# The 32-bit layer, for 32-bit games (a separate layer name and manifest, as upstream does).
I686_TARGET="i686-unknown-linux-gnu"
if ! rustup target list --installed ${rustup_toolchain_arg:+$rustup_toolchain_arg} 2>/dev/null | grep -q "$I686_TARGET"; then
    # shellcheck disable=SC2086
    rustup target add $rustup_toolchain_arg "$I686_TARGET"
fi
if ! rustup target list --installed ${rustup_toolchain_arg:+$rustup_toolchain_arg} 2>/dev/null | grep -q "$WIN_TARGET"; then
    # shellcheck disable=SC2086 -- word-splitting $rustup_toolchain_arg is intentional here.
    rustup target add $rustup_toolchain_arg "$WIN_TARGET"
fi

# ── Release build ─────────────────────────────────────────────────────
echo "==> cargo build --release (protocol/layer/gui/cli)"
cargo build --release

echo "==> $CARGO_HELPER build --release --target $WIN_TARGET -p neural-forge-helper"
$CARGO_HELPER build --release --target "$WIN_TARGET" -p neural-forge-helper

echo "==> $CARGO_HELPER build --release --target $I686_TARGET -p neural-forge-layer"
$CARGO_HELPER build --release --target "$I686_TARGET" -p neural-forge-layer

# ── AppDir layout ─────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/lib/$LIBDIR/helper" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
         "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/vulkan/implicit_layer.d"

cp "target/release/$NAME"                      "$APPDIR/usr/bin/"
cp "target/release/$NAME-cli"                       "$APPDIR/usr/bin/"
cp "target/release/$LIB"                            "$APPDIR/usr/lib/$LIBDIR/"
cp "target/$WIN_TARGET/release/${NAME}-helper.exe"  "$APPDIR/usr/lib/$LIBDIR/helper/"
mkdir -p "$APPDIR/usr/lib/$LIBDIR/i686"
cp "target/$I686_TARGET/release/$LIB" "$APPDIR/usr/lib/$LIBDIR/i686/"
sed "s#\./i686/libneural_forge_layer\.so#../../../lib/$LIBDIR/i686/$LIB#" \
    data/neural_forge_layer_i686.json > "$APPDIR/usr/share/vulkan/implicit_layer.d/neural_forge_layer_i686.json"
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
# The Vulkan layer manifest ships inside the AppImage's own read-only tree, so the
# loader needs an explicit path to it -- there is no writable implicit_layer.d this
# install owns to drop it into (this app needs no root/install step at all).
export VK_ADD_LAYER_PATH="$HERE/usr/share/vulkan/implicit_layer.d${VK_ADD_LAYER_PATH:+:$VK_ADD_LAYER_PATH}"
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

echo "==> Packing AppImage"
OUT="neural-forge-$VERSION-$ARCH.AppImage"

# Use the canonical renamed repository for release updates.
UPDATE_INFORMATION="gh-releases-zsync|labj1987|neural-forge|latest|neural-forge-*-x86_64.AppImage.zsync"
VERSION="$VERSION" ARCH="$ARCH" "$TOOL" --appimage-extract-and-run \
    -u "$UPDATE_INFORMATION" "$APPDIR" "$OUT"

echo "==> Done: $OUT"
ls -lh "$OUT"

# appimagetool's built-in zsync generation silently no-ops on some CI runners (see
# KernelPop's CLAUDE.md); build the sidecar directly instead. Non-fatal.
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
if zsyncmake -u "$ZSYNC_URL" "$OUT"; then
    echo "==> .zsync generated: $OUT.zsync"
else
    echo "==> WARNING: zsyncmake failed — continuing without .zsync"
fi

# ── Legacy-name compatibility assets ──────────────────────────────────
# AppImages released before the rename embed update information that matches
# `NeuralForge-*-x86_64.AppImage.zsync` on this repo's releases (GitHub redirects the
# old repo name). Publish a byte-identical copy under the legacy name, with its own
# .zsync, so those installs can still find and fetch this release. Drop once no
# pre-rename installs remain.
LEGACY="NeuralForge-$VERSION-$ARCH.AppImage"
cp "$OUT" "$LEGACY"
LEGACY_URL="https://github.com/labj1987/neural-forge/releases/download/v$VERSION/$LEGACY"
if zsyncmake -u "$LEGACY_URL" "$LEGACY"; then
    echo "==> legacy-name copy + .zsync generated: $LEGACY"
else
    echo "==> WARNING: zsyncmake failed for the legacy-name copy"
fi
