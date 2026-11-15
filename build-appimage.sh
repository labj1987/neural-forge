#!/usr/bin/env bash
# build-appimage.sh — build the Neural Forge AppImage.
# Run from the repo root on Ubuntu (the GitHub Actions runner), as an ordinary user: it writes
# only inside the checkout (and, on a machine without Rust, rustup's ~/.cargo and ~/.rustup).
# Only its from-scratch system package install below needs root, through sudo.
#
# Packaging is appimagetool run directly on a hand-built AppDir, plus zsyncmake for the
# update sidecar. The GUI ships twice: usr/bin/neural-forge uses the host's GTK 4 and libadwaita,
# usr/bin/bundled/neural-forge the GTK runtime bundled here (bundle_gtk_runtime); AppRun picks one.
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

# ── Bundled GTK 4 / libadwaita runtime ───────────────────────────────
# The AppImage carries GTK 4, libadwaita and everything they link, taken from the build machine
# (the release runner, Ubuntu 24.04, glibc 2.39), for desktops that ship neither (a stock KDE or
# Hyprland install). Only the GUI's second copy, usr/bin/bundled/neural-forge, uses it: AppRun runs the
# host copy (usr/bin/neural-forge, unpatched) whenever the host's own GTK and libadwaita resolve,
# so an up-to-date desktop keeps its newer libadwaita (accent colours and the like). The CLI, the
# Vulkan layer and the NGX wrapper are never patched and never load a bundled library.
# What stays on the host is the AppImage project's excludelist
# (github.com/AppImageCommunity/pkg2appimage, file `excludelist` at commit 15a64c2, 2024-11-03):
# glibc, libstdc++/libgcc_s, the GL/EGL/GLX/glvnd and DRM/GBM stack, libX11/libxcb, Wayland's
# client library, fontconfig/freetype/harfbuzz/fribidi, zlib, expat and the rest of that list,
# plus our own additions below: the Vulkan loader (its drivers and layers are the host's) and
# Wayland's other client libraries (they pair with the host's libwayland-client and Mesa).
# Nothing is found through LD_LIBRARY_PATH: every bundled ELF file gets an rpath relative to
# itself ($ORIGIN), so the programs the GUI starts never load a bundled library.
BUNDLE_EXCLUDE=(
    ld-linux.so.2 ld-linux-x86-64.so.2 libanl.so.1 libBrokenLocale.so.1 libcidn.so.1 libc.so.6
    libdl.so.2 libm.so.6 libmvec.so.1 libnss_compat.so.2 libnss_dns.so.2 libnss_files.so.2
    libnss_hesiod.so.2 libnss_nisplus.so.2 libnss_nis.so.2 libpthread.so.0 libresolv.so.2
    librt.so.1 libthread_db.so.1 libutil.so.1 libstdc++.so.6 libGL.so.1 libEGL.so.1
    libGLdispatch.so.0 libGLX.so.0 libOpenGL.so.0 libdrm.so.2 libglapi.so.0 libgbm.so.1
    libxcb.so.1 libX11.so.6 libX11-xcb.so.1 libwayland-client.so.0 libasound.so.2
    libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libcom_err.so.2 libexpat.so.1
    libgcc_s.so.1 libgpg-error.so.0 libICE.so.6 libSM.so.6 libusb-1.0.so.0 libuuid.so.1
    libz.so.1 libjack.so.0 libpipewire-0.3.so.0 libxcb-dri3.so.0 libxcb-dri2.so.0
    libfribidi.so.0 libgmp.so.10
    # Ours:
    libvulkan.so.1 libwayland-egl.so.1 libwayland-cursor.so.0 libwayland-server.so.0
)

bundle_excluded() { [[ " ${BUNDLE_EXCLUDE[*]} " == *" $1 "* ]]; }

# bundle_ldd FILE: prints "soname path" for each library FILE loads (its whole closure, as ldd
# resolves it right now); fails, naming them, when one cannot be found.
bundle_ldd() {
    local out
    out="$(ldd "$1")" || return 1
    if grep -q '=> not found' <<<"$out"; then
        echo "==> ERROR: $1 needs libraries this machine does not have:" >&2
        grep '=> not found' <<<"$out" >&2
        return 1
    fi
    awk '$2 == "=>" && $3 ~ /^\// { print $1, $3 }' <<<"$out"
}

# bundle_gtk_runtime APPDIR BINARY...: puts the GTK 4 + libadwaita runtime the given binaries
# (paths relative to APPDIR/usr, already copied there) need into APPDIR/usr:
#   lib/                                the libraries (rpath $ORIGIN)
#   lib/gdk-pixbuf-2.0/2.10.0/          image loaders (SVG icons) and their loaders.cache
#   lib/gio/modules/                    the dconf GSettings backend, so settings live in the
#                                       host's dconf like every other GTK app's
#   lib/gtk-4.0/4.0.0/immodules/        the IBus input method module (X11 sessions; on Wayland
#                                       GTK speaks to the input method itself)
#   share/glib-2.0/schemas/             GTK's and the desktop's GSettings schemas, compiled
#   share/doc/neural-forge/bundled-libraries/   the bundled packages' copyright files
# The layer's directory, lib/neural-forge/, is not touched (the installer copies all of it).
# AppRun's bundle_env block points GTK at them (GTK_EXE_PREFIX, GDK_PIXBUF_MODULE_FILE,
# GIO_MODULE_DIR, GSETTINGS_SCHEMA_DIR), and the GUI gives every program it starts the host's
# values of those back (neural_forge_supervisor::host_env).
bundle_gtk_runtime() {
    local appdir="$1"; shift
    local usr="$appdir/usr" syslib pixver pixdir loaders giomods immods f soname path pkg
    local work="$appdir/.."
    syslib="$(pkg-config --variable=libdir gtk4)"
    pixver="$(pkg-config --variable=gdk_pixbuf_binary_version gdk-pixbuf-2.0)"
    pixdir="$(pkg-config --variable=gdk_pixbuf_moduledir gdk-pixbuf-2.0)"
    # AppRun's bundle_env names the loader cache's directory: it must be this gdk-pixbuf's binary version.
    if ! grep -qF "/usr/lib/gdk-pixbuf-2.0/$pixver/loaders.cache" data/AppRun; then
        echo "==> ERROR: data/AppRun's GDK_PIXBUF_MODULE_FILE does not name gdk-pixbuf-2.0/$pixver" >&2
        return 1
    fi
    loaders="$usr/lib/gdk-pixbuf-2.0/$pixver/loaders"
    giomods="$usr/lib/gio/modules"
    immods="$usr/lib/gtk-4.0/4.0.0/immodules"
    local modules=("$pixdir/libpixbufloader-svg.so" "$syslib/gio/modules/libdconfsettings.so"
        "$syslib/gtk-4.0/4.0.0/immodules/libim-ibus.so")
    # GIO 2.80 starts every program it launches itself (a file manager for FileLauncher, without an OpenURI portal)
    # through gio-launch-desktop, looked up at its build path (/usr/lib/x86_64-linux-gnu/glib-2.0, absent outside
    # Debian and Ubuntu), then in PATH: AppRun puts usr/bin first in PATH. It links only libc.
    local launcher="$syslib/glib-2.0/gio-launch-desktop"
    for f in "${modules[@]}" "$launcher" "$syslib/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders"; do
        [[ -e "$f" ]] && continue
        echo "==> ERROR: $f is missing (packages librsvg2-common, dconf-gsettings-backend, ibus-gtk4, libgdk-pixbuf2.0-bin)" >&2
        return 1
    done
    for f in patchelf glib-compile-schemas gio-querymodules dpkg; do
        command -v "$f" >/dev/null 2>&1 || { echo "==> ERROR: $f is not installed" >&2; return 1; }
    done

    echo "==> Bundling GTK 4 and libadwaita for the GUI's second copy"
    mkdir -p "$usr/lib" "$loaders" "$giomods" "$immods" "$usr/share/glib-2.0/schemas"
    cp "$pixdir"/*.so "$loaders/"
    cp "$syslib/gio/modules/libdconfsettings.so" "$giomods/"
    cp "$syslib/gtk-4.0/4.0.0/immodules/libim-ibus.so" "$immods/"
    # The real one goes in usr/libexec; usr/bin gets data/gio-launch-desktop, which first gives the launched program
    # the environment it would have had without the AppImage (PATH, the NEURAL_FORGE_* markers).
    mkdir -p "$usr/libexec"
    cp "$launcher" "$usr/libexec/"
    install -m 755 data/gio-launch-desktop "$usr/bin/gio-launch-desktop"
    printf '%s\n' "$pixdir"/*.so "${modules[@]:1}" "$launcher" > "$work/bundle-sources"
    bundle_ldd "$usr/libexec/gio-launch-desktop" > "$work/bundle-closure" || return 1
    while read -r soname path; do
        bundle_excluded "$soname" || { echo "==> ERROR: gio-launch-desktop needs $soname, which is not the host's" >&2; return 1; }
    done < "$work/bundle-closure"

    # The libraries: the ldd closure of the binaries and the modules, minus the excludelist.
    local elves=()
    for f in "$@"; do elves+=("$usr/$f"); done
    elves+=("$loaders"/*.so "$giomods"/*.so "$immods"/*.so)
    for f in "${elves[@]}"; do
        bundle_ldd "$f" > "$work/bundle-closure" || return 1
        while read -r soname path; do
            bundle_excluded "$soname" && continue
            if [[ ! -e "$usr/lib/$soname" ]]; then
                cp -L "$path" "$usr/lib/$soname"
                echo "$path" >> "$work/bundle-sources"
            fi
        done < "$work/bundle-closure"
    done

    # rpath, not LD_LIBRARY_PATH. DT_RPATH (--force-rpath), not DT_RUNPATH, so that a
    # LD_LIBRARY_PATH the user has set cannot swap in a host library the bundle wasn't built with.
    # The binaries' rpath also names the loader directory: loaders.cache lists bare file names,
    # which dlopen() finds there wherever the AppImage is mounted. The binaries live two directories
    # below usr/ (usr/bin/bundled/).
    for f in "$@"; do
        patchelf --force-rpath --set-rpath "\$ORIGIN/../../lib:\$ORIGIN/../../lib/gdk-pixbuf-2.0/$pixver/loaders" "$usr/$f"
    done
    # shellcheck disable=SC2016 # $ORIGIN is for the dynamic linker, not the shell
    for f in "$usr/lib"/*.so*; do patchelf --force-rpath --set-rpath '$ORIGIN' "$f"; done
    # shellcheck disable=SC2016
    for f in "$loaders"/*.so "$immods"/*.so; do patchelf --force-rpath --set-rpath '$ORIGIN/../../..' "$f"; done
    # shellcheck disable=SC2016
    for f in "$giomods"/*.so; do patchelf --force-rpath --set-rpath '$ORIGIN/../..' "$f"; done

    # Check: everything bundled now loads its libraries from the AppDir, or from the host only
    # when the excludelist says so.
    local inside bad=0
    inside="$(realpath "$usr/lib")"
    for f in "${elves[@]}" "$usr/lib"/*.so*; do
        bundle_ldd "$f" > "$work/bundle-closure" || return 1
        while read -r soname path; do
            bundle_excluded "$soname" && continue
            [[ "$(realpath "$path")" == "$inside/"* ]] && continue
            echo "==> ERROR: $f loads $soname from $path, not from the AppDir" >&2
            bad=1
        done < "$work/bundle-closure"
    done
    rm -f "$work/bundle-closure"
    [[ $bad -eq 0 ]] || return 1

    local cache
    cache="$("$syslib/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders" "$loaders"/*.so)"
    cache="${cache//\"$(realpath "$loaders")\//\"}"
    cache="${cache//\"$loaders\//\"}"
    printf '%s\n' "$cache" > "$loaders/../loaders.cache"
    if grep -q '^"/.*\.so"$' "$loaders/../loaders.cache"; then
        echo "==> ERROR: loaders.cache still names absolute paths" >&2
        return 1
    fi
    gio-querymodules "$giomods"

    # GSettings schemas: GTK's own (its file, colour and emoji choosers abort without them) and the
    # desktop ones GTK reads when there is no settings portal.
    cp /usr/share/glib-2.0/schemas/org.gtk.gtk4.*.gschema.xml \
       /usr/share/glib-2.0/schemas/org.gnome.desktop.*.xml "$usr/share/glib-2.0/schemas/"
    glib-compile-schemas --strict "$usr/share/glib-2.0/schemas"

    # The bundled libraries' copyright and licence notices, from their packages.
    local docs="$usr/share/doc/$NAME/bundled-libraries"
    mkdir -p "$docs"
    while read -r path; do
        pkg="$(dpkg -S "$path" 2>/dev/null || dpkg -S "$(realpath "$path")" 2>/dev/null \
            || dpkg -S "${path#/usr}" 2>/dev/null || true)"
        pkg="${pkg%%:*}"
        if [[ -z "$pkg" || ! -f "/usr/share/doc/$pkg/copyright" ]]; then
            echo "==> ERROR: no copyright notice found for $path" >&2
            return 1
        fi
        cp "/usr/share/doc/$pkg/copyright" "$docs/$pkg.copyright"
        echo "$pkg" >> "$work/bundle-packages"
    done < <(sort -u "$work/bundle-sources")
    rm -f "$work/bundle-sources"
    # The licence texts those notices refer to by path (Debian keeps the common ones in /usr/share/common-licenses).
    local lic
    mkdir -p "$docs/common-licenses"
    while read -r lic; do
        [[ -f "$lic" ]] || { echo "==> ERROR: a bundled library's copyright file names $lic, which is missing" >&2; return 1; }
        cp "$lic" "$docs/common-licenses/"
    done < <(grep -ohE '/usr/share/common-licenses/[A-Za-z0-9.+_-]*[A-Za-z0-9+_-]' "$docs"/*.copyright | sort -u)
    # Which packages, at which version, from which source package.
    {
        echo "The libraries and modules in usr/lib, and usr/libexec/gio-launch-desktop, are these Ubuntu 24.04 packages' files,"
        echo "unmodified apart from the run-time search path (rpath) patchelf set on the libraries and modules."
        echo "Their sources are the Ubuntu source packages named here, at these versions."
        echo
        printf 'package\tversion\tsource package\n'
        # shellcheck disable=SC2016 # dpkg-query's format, not the shell's
        sort -u "$work/bundle-packages" | xargs dpkg-query -W -f '${Package}\t${Version}\t${source:Package}\n'
    } > "$docs/PACKAGES.txt"
    rm -f "$work/bundle-packages"
    echo "==> Bundled $(find "$usr/lib" -path "$usr/lib/$LIBDIR" -prune -o -name '*.so*' -print | wc -l) libraries and modules"
}

# ── Build dependencies ────────────────────────────────────────────────
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi
# Every tool the build runs (here, in scripts/fetch-native-tools.sh, the crates' build scripts and the
# Python generators), so a missing one is installed before the long release build, not found after it.
missing_deps() {
    local tool
    for tool in g++ curl patch python3 pkg-config desktop-file-validate zsyncmake patchelf glib-compile-schemas gio-querymodules; do
        command -v "$tool" >/dev/null 2>&1 || { echo "$tool"; return 0; }
    done
    pkg-config --exists gtk4 libadwaita-1 2>/dev/null || { echo "gtk4/libadwaita-1"; return 0; }
    # What bundle_gtk_runtime copies besides the libraries.
    local f
    for f in "$(pkg-config --variable=gdk_pixbuf_moduledir gdk-pixbuf-2.0)/libpixbufloader-svg.so" \
        "$(pkg-config --variable=libdir gtk4)/gio/modules/libdconfsettings.so" \
        "$(pkg-config --variable=libdir gtk4)/gtk-4.0/4.0.0/immodules/libim-ibus.so" \
        "$(pkg-config --variable=libdir gtk4)/gdk-pixbuf-2.0/gdk-pixbuf-query-loaders" \
        /usr/share/glib-2.0/schemas/org.gnome.desktop.interface.gschema.xml; do
        [[ -e "$f" ]] || { echo "$f"; return 0; }
    done
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
    # patchelf and the rest are for the bundled GTK runtime (bundle_gtk_runtime).
    $SUDO apt-get install -y -qq g++ curl patch python3 libgtk-4-dev libadwaita-1-dev \
        pkg-config desktop-file-utils zsync \
        patchelf librsvg2-common dconf-gsettings-backend ibus-gtk4 libgdk-pixbuf2.0-bin gsettings-desktop-schemas
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
# The pinned glslang, Vulkan-Headers and volk that crates/native builds with (SHA-256-checked,
# skipped when already in tools/native/).
bash scripts/fetch-native-tools.sh
# ── Release build ─────────────────────────────────────────────────────
echo "==> cargo build --release (protocol/layer/native/gui/cli)"
cargo build --release --locked

# The NGX wrapper for Proton (docs/PROTON_NGX.md): a Windows DLL, stamped as a Wine builtin.
bash scripts/build-ngx-wrapper.sh "${CARGO_TARGET_DIR:-target}/ngx-wrapper/_nvngx.dll"

# ── AppDir layout ─────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/bin/bundled" \
         "$APPDIR/usr/lib/$LIBDIR" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
         "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/vulkan/implicit_layer.d"

RELEASE="${CARGO_TARGET_DIR:-target}/release"
# The GUI twice: as built (the host's GTK) and with the bundled GTK runtime (AppRun picks one).
cp "$RELEASE/$NAME"                                 "$APPDIR/usr/bin/"
cp "$RELEASE/$NAME"                                 "$APPDIR/usr/bin/bundled/"
bundle_gtk_runtime "$APPDIR" "bin/bundled/$NAME"
if [[ -n "$(patchelf --print-rpath "$APPDIR/usr/bin/$NAME")" ]]; then
    echo "==> ERROR: usr/bin/$NAME must load the system's libraries, but it has an rpath" >&2
    exit 1
fi
cp "$RELEASE/$NAME-cli"                             "$APPDIR/usr/bin/"
cp "$RELEASE/$LIB"                                  "$APPDIR/usr/lib/$LIBDIR/"
mkdir -p "$APPDIR/usr/lib/$LIBDIR/proton"
cp "${CARGO_TARGET_DIR:-target}/ngx-wrapper/_nvngx.dll" "$APPDIR/usr/lib/$LIBDIR/proton/"
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
# network's MIT licence, notice and origin from OpenDLSS-NR (third_party/opendlss-nr) and the layer
# framework's Apache-2.0 licence and origin (third_party/vk-layer-for-rust), both without their
# source, the statically linked crates' listing (regenerated here so it cannot drift from
# Cargo.lock), and the full licence and notice texts of every linked crate and of the network, volk
# and the Vulkan-Headers compiled into the layer, collected from their sources.
DOCDIR="$APPDIR/usr/share/doc/$NAME"
mkdir -p "$DOCDIR"
python3 scripts/gen_third_party_crates.py
python3 scripts/gen_third_party_crates.py --licences "$DOCDIR/THIRD_PARTY_LICENSES.txt"
cp LICENSE ATTRIBUTION.md THIRD_PARTY_CRATES.md "$DOCDIR/"
mkdir -p "$DOCDIR/third_party/opendlss-nr" "$DOCDIR/third_party/vk-layer-for-rust"
cp -r third_party/optiscaler "$DOCDIR/third_party/"
cp third_party/opendlss-nr/{LICENSE,NOTICE,UPSTREAM.md} "$DOCDIR/third_party/opendlss-nr/"
cp third_party/vk-layer-for-rust/{LICENSE,UPSTREAM.md} "$DOCDIR/third_party/vk-layer-for-rust/"
# The NGX wrapper links the mingw-w64 runtime and libgcc (GCC Runtime Library Exception) statically: their
# notices from the build machine's packages.
mkdir -p "$DOCDIR/third_party/mingw-w64"
for pkg in mingw-w64-common gcc-mingw-w64-base; do
    [ -f "/usr/share/doc/$pkg/copyright" ] || { echo "missing /usr/share/doc/$pkg/copyright (the NGX wrapper's runtime notices)" >&2; exit 1; }
    cp "/usr/share/doc/$pkg/copyright" "$DOCDIR/third_party/mingw-w64/$pkg.copyright"
done

# Top-level AppImage requirements
cp data/io.github.labj1987.NeuralForge.desktop "$APPDIR/"
cp data/icon.svg "$APPDIR/$NAME.svg"

# ── AppRun ────────────────────────────────────────────────────────────
# data/AppRun: starts the host-GTK GUI when the host's GTK and libadwaita resolve, else the bundled one after
# checking the libraries it takes from the host (scripts/test_apprun.sh).
cp data/AppRun "$APPDIR/AppRun"
chmod 755 "$APPDIR/AppRun"

# ── appimagetool ──────────────────────────────────────────────────────
# Pinned to a released version and verified by checksum (the `continuous` tag moves
# under us, so two builds of the same commit could use different tools). The weekly pin
# check (scripts/check_pins.py --bump) moves the version and its sha256 to each new release.
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
# The weekly pin check (scripts/check_pins.py --bump) moves the version and the sha256 of the
# runtime-x86_64 asset to each new release.
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
