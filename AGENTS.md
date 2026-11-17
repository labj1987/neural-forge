# Neural Forge developer guidance

Repository: https://github.com/labj1987/neural-forge

Read [docs/PHASE1.md](docs/PHASE1.md) for the current namespace, installation contract and
target-ownership rules. Benchmarks run with `scripts/gta-bench.sh`
([docs/RUNNING_AND_MEASURING.md](docs/RUNNING_AND_MEASURING.md)) on the maintainer's desktop, which stays at
2560x1440, 288 Hz, VRR with HDR on: never change the display mode for a test, set the game's own resolution. The
game sweep sets it to 2560x1440 (since 2026-11-12: at 4K the 12 GB card runs out of video memory with Ray
Reconstruction and the model after DLSS); `gta-bench.sh` sets none and uses GTA's `settings.xml` as found. Games that have HDR are tested with their HDR on; video memory is sampled. Every release is
checked in Cyberpunk 2077, GTA V Enhanced, Red Dead Redemption 2 and Crimson Desert; the old Phase 1 plan is in
[docs/history/phase1-benchmark-plan.md](docs/history/phase1-benchmark-plan.md). The former app name was dlssnr; upstream DLSS5VKLayer remains a
separate application and must not be modified or uninstalled by this project.

All documentation is indexed in [docs/README.md](docs/README.md); start with docs/ARCHITECTURE.md and docs/LESSONS.md.

## Naming convention

- Display name (anything a person reads: window titles, About, docs prose): **Neural Forge**.
- Hyphenated lowercase `neural-forge` for everything a machine or filesystem names: the
  repo, Cargo package names (`neural-forge-cli`, `neural-forge-protocol`, ...), binaries
  (`neural-forge`, `neural-forge-cli`), the AppImage
  (`neural-forge-<version>-x86_64.AppImage`), the icon (`neural-forge.svg`).
- **Frozen, never rename** (renaming breaks layer registration): the app ID
  `io.github.labj1987.NeuralForge` (and the `.desktop`/appdata files named after it) and the
  Vulkan layer name `VK_LAYER_neuralforge_neural`.
- Since 0.1.77 everything else is hyphenated/underscored `neural-forge`: `NEURAL_FORGE_*` env
  vars (read through `neural_forge_protocol::env`), `/tmp/neural-forge-$UID`, the `neural-forge` XDG config/data/state
  dirs, `lib/neural-forge/`, `libneural_forge_layer.so` (`[lib] name = "neural_forge_layer"`),
  its manifest `neural_forge_layer.json`, and the `[neural-forge-layer]` log prefix. The manifest's `enable_environment` is `NEURAL_FORGE_ENABLE`.
- No backward compatibility with old names or formats (`dlssnr`, `neuralforge`, `NEURALFORGE_*`,
  `NeuralForge-*` AppImages, the 2.x helper's config keys): every install already uses the current
  ones; config files are cleaned on the machines directly, not by migration code. Files a previous
  install shipped but this one no longer does are cleaned by the installer's record-based
  stale-file removal. Never run the CLI or `install` against real XDG dirs in tests: point
  `XDG_*_HOME` at a scratch dir.

## Build and test

- Run `bash scripts/fetch-native-tools.sh` once before building: it fetches the pinned glslang,
  Vulkan-Headers and volk that `crates/native` builds with into `tools/native/`; the build is offline
  after that. The network's source (C++, GLSL, PTX generators) is this project's own, in
  `third_party/opendlss-nr`: edit it there directly. That directory stays MIT (keep its `LICENSE` and
  `NOTICE`); record notable changes in CHANGELOG.md, and credit anything ported from a later upstream
  OpenDLSS-NR commit by that commit. `UPSTREAM.md` there says where it came from.
- The layer framework (`vulkan-layer`, Google's vk-layer-for-rust, Apache-2.0) is this project's own too, in
  `third_party/vk-layer-for-rust`: its own workspace, excluded from the root one. Edit it there directly and
  mark each file changed for the first time with a "Modified for Neural Forge" note (Apache-2.0 asks for one);
  never edit its generated files by hand (`bash scripts/vk-layer-codegen.sh`). Its `UPSTREAM.md` has the checks.
- CI (`.github/workflows/check.yml`, run by ci.yml on every push and pull request and by release.yml
  on a tag) runs the tests, the layer's tests again under the Khronos validation layer (which must
  report nothing, with nothing filtered), clippy, `cargo fmt --check`, every script's
  self-test, the shader check, the dependency check, the AppImage build and its installed-layer smoke test. A new script with a self-test is added there.
- Formatting: `cargo fmt --all` before committing (rustfmt.toml: 160 columns), and in
  `third_party/vk-layer-for-rust` too, which keeps its own 100-column config.
- Dependencies: `bash scripts/cargo-deny.sh` (deny.toml) checks RustSec advisories, licences and sources. A
  new advisory fails CI: update the crate, or add an `ignore` entry in deny.toml with the reason it does not
  apply. A new licence must be added to the allow-list on purpose, after checking it is AGPL-compatible. The weekly pin check
  (`scripts/check_pins.py`, pin-check.yml) bumps every pin Dependabot cannot see that is behind upstream
  (`check_pins.py --bump`, which rewrites every place the pin is named: add a new place to its rewrite rules
  and self-test), runs check.yml on the bump and fast-forwards main to it only if that passes (pins only:
  no version bump, no release). main is branch-protected and requires the status "check / build" on the
  commit: a workflow's own check runs are filed under the commit it started from, so the land job records
  the passed check.yml result on the bump commit (Checks API, linked to the run) before it pushes. Vulkan-Headers and volk are pinned to the same `vulkan-sdk-*` tag. It keeps
  one issue about pins behind upstream, left open only when a bump failed; it must never comment or open
  more than that one (each notification is an email).
- `cargo test` and `cargo build --release` build every crate (x86_64 Linux only; there is no
  Windows helper and no 32-bit layer since 3.0).
- `bash build-appimage.sh` packages Neural Forge. The AppImage bundles no libraries: the GUI ships once,
  `usr/bin/neural-forge` as cargo built it, next to the CLI, the layer and the NGX wrapper. Its toolkit (GPUI) links
  only glibc, libgcc_s, libxcb, libxkbcommon and libxkbcommon-x11, and loads `libwayland-client.so.0` (Wayland
  sessions) and `libvulkan.so.1` or `libEGL.so.1` by name at run time, all from the system. The only shared object
  in the AppImage is the layer, in `usr/lib/neural-forge/` (the installer copies all of it); CI checks that. Build
  dependencies are the toolkit's headers (`libxcb1-dev libxkbcommon-dev libxkbcommon-x11-dev libfontconfig1-dev`).
  Release builds run on Ubuntu 24.04, whose glibc is the floor the README states (CI's glibc baseline step).
- `data/AppRun` checks the system, then `exec`s the GUI with its arguments, and changes no environment variable: the
  GUI and every program it starts see the environment the AppImage was started with (start programs with a plain
  `std::process::Command`). Its libcheck refuses to start, with the reason on stderr and in a dialog (zenity, else
  kdialog, else notify-send) that names the distribution's install command, when `ldd -r` reports a missing library,
  a missing symbol version (a glibc older than the build's) or an undefined symbol of the GUI, or when the run-time
  libraries are missing (`ldconfig -p` or `LD_LIBRARY_PATH`: libwayland-client on a Wayland session, one of the
  Vulkan and EGL loaders); without ldd it asks the dynamic loader. `scripts/test_apprun.sh` tests it with stand-ins;
  `scripts/smoke-gui.sh APPDIR` starts the extracted AppImage's GUI on Xvfb (drawn on lavapipe) and fails on an
  unmapped window, a loader error, a panic or a WARN/ERROR log line.
- The layer's GPU tests run on lavapipe: set `VK_DRIVER_FILES` to the `lvp_icd*.json` manifests, as
  CI does. On a machine with an NVIDIA GPU, never run them without it. Under validation:
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` and `--test-threads=1` (CI's step shows the settings;
  validation layers before 1.4 crash when tests create and use devices in parallel); a test device
  enables what the code under test needs (`composition::gpu::test_device_extensions`), destroys what
  it created before the device, and records over real images, never stand-in handles.
- Leave lavapipe's vector width and CPU features at their defaults. With `LP_NATIVE_VECTOR_WIDTH=128` on an
  AVX CPU, Mesa 26.0.8's llvmpipe crashes in its own JIT code in every test that runs the native frame (a
  32-byte aligned store to a 16-byte aligned slot of its stack), and Mesa 25.2.8's does with 128 and
  `GALLIUM_OVERRIDE_CPU_CAPS=sse4.1` (a jump into non-executable memory). The same shaders pass at the
  defaults (also under GPU-assisted validation), and each Mesa passes the setting the other crashes with.
- Tests must not depend on how fast the machine runs them: the hold's history gap
  (`HistoryGap::MAX_GAP`, 500 ms) reads `native::hold_now`, which the inline test rig stops.
- Run `python3 scripts/check_namespace.py`, `python3 scripts/test_install.py`,
  and `bash scripts/smoke-test.sh` for namespace, installation and Vulkan checks.
- Shaders: `python3 scripts/check_shaders.py --update` rebuilds the committed SPIR-V with the pinned
  glslang only (never one on PATH) and needs `spirv-val`.
- Every component compiled or linked into the shipped binaries must have its licence text in the
  AppImage: `scripts/gen_third_party_crates.py --licences` collects them from the crates' and the
  native sources at build time and fails on one without.
- Toggle key: a press is ignored only while another X client is known to have the focus
  (`hotkey.rs`); where the focus cannot be told (gamescope's nested Xwayland holds only the game, a
  bare X server without `_NET_ACTIVE_WINDOW`) it acts, and `NEURAL_FORGE_HOTKEY_GLOBAL=1` makes it act
  even then. Keep that: a key read from `/dev/input` otherwise fires while the user types elsewhere.
- Launch options: the supervisor always writes `NEURAL_FORGE_TARGET_EXE` (the game's own executables,
  `steam::target_for`), and an explicit target wins over the layer's built-in exclusions.
- `extract-model` gates on the network's shape, not the DLL's version: it takes any build of
  `nvngx_dlssnr.dll` whose weights resource holds exactly the tensors in
  `crates/supervisor/src/model_shape.rs` (names and byte lengths): `WEIGHTS_HT` for the full network
  (`EXPECTED`, 71 blocks), `WEIGHTS_CC_SMALL_FP8` for the small one (`EXPECTED_SMALL`, 31 blocks,
  `--network small`). The graph picks its block ranges from the model's block count
  (`nr::Topology` in `third_party/opendlss-nr/src/nr_graph.*`); a third network needs a new `Topology` and stage table. That table
  changes together with the graph: regenerate it with `scripts/gen_model_shape.py` when the graph takes a new network. `VERIFIED` only marks the networks and builds compared with
  NVIDIA's runtime; it gates nothing.

## Rust

- Toolchain: Rust 1.100 or newer. `rust-toolchain.toml` pins the exact release that local, CI and release builds use; `rust-version = "1.100"` in Cargo.toml is the minimum.
- Dependency cooldown: `.cargo/config.toml` sets `[registry] global-min-publish-age = "3 days"`, so resolution only takes crate versions at least 3 days old (Dependabot's cargo `cooldown` matches). To take a newer version early for a security advisory: `CARGO_RESOLVER_INCOMPATIBLE_PUBLISH_AGE=allow cargo update -p <crate> --precise <version>`. Until that version is 3 days old, a plain `cargo update` moves it back down, so run that round's `cargo update` with the same variable set (or update the other crates with `-p`).
- Never compare tool versions as text or floats ("1.100" sorts before "1.99" as text): split into integers, or use `sort -V`.
- Never read Cargo's intermediate layout under `target/` (`build/`, `deps/`, `incremental/`, `.fingerprint/`): it is not a stable interface and changed in Cargo 1.100. Use the final artifact paths, `OUT_DIR` inside build.rs, or `cargo build --message-format=json`.
- Cargo's own lints are on (`[lints.cargo]`, `unused_dependencies = "warn"`) and CI sets `CARGO_BUILD_WARNINGS=deny`, so a lint warning fails CI. A direct dependency kept only to turn on a feature of a crate already in the graph stays in `[dependencies]` with a comment saying why: the lint does not flag it, and moving it to a `cfg(false)` table would turn the feature off. (`[target.'cfg(false)'.dependencies]` is only for a dependency listed just to pin a transitive crate's version.)

## README

- The README stays short: what it is, requirements, install, usage, the known issues and
  troubleshooting a user will actually hit, then Legal, License and Acknowledgements. Every other
  user-facing detail (settings, tested games, environment variables, CLI, files) goes in
  docs/USER_GUIDE.md, and measurements in docs/MEASURED_RESULTS.md.
- The README, the user guide and the measured results describe the newest release only, in the
  present tense. No version history ("since 3.0",
  "3.4.1 drops ...", tables comparing releases, notes for settings made with an older version,
  screenshots captioned with an old version). Users run the latest release; history belongs in
  CHANGELOG.md and the docs.
- Every release re-reads all three and updates or removes what the release changed, including the
  screenshots and the measured results. A new environment variable gets a row in the user guide's table.
- Screenshots are always the dark theme, 2880x1800: the window at its default size on a large screen
  (1440x900, the tabs in the title bar) at 2x. On a private X display (Xvfb at 3200x2000, `dbus-run-session`):
  `GPUI_X11_SCALE_FACTOR=2` for the scale, `NEURAL_FORGE_GUI_THEME=dark` (a headless session has no desktop preference to
  follow), `NEURAL_FORGE_GUI_OPEN` for the tab or dialog and `NEURAL_FORGE_GUI_SCROLL` for a tab's lower part, the lavapipe
  ICD in `VK_DRIVER_FILES`, and the window alone captured (`import -window`). Give it a scratch `HOME`, XDG directories
  and a private `XDG_RUNTIME_DIR` (the GUI's startup updates the NGX wrapper in the Proton copies it finds through
  `$HOME`, and a second session's helpers in the real runtime directory disturb the desktop's). Look at every image
  before committing it.

## UI (GPUI Kit)

- The toolkit is `gpui-kit`, pinned to an exact version in `crates/gui/Cargo.toml` (0.9.x releases weekly and pins GPUI
  itself as the `gpui-pre` snapshot crates). It draws with wgpu: Vulkan first, OpenGL ES through EGL second; Mesa's
  software drivers count. Bump it on purpose, then re-run the checks below. Its sources are the reference:
  `~/.cargo/registry/src/*/gpui-component-*`, `gpui-base-*`, `gpui-pre-*`.
- Shape (`crates/gui/src`): `ui.rs` has the toolkit set-up, the pieces pages are made of (`group`, `row`, `Spin`,
  `menu_button`, `ask`, `inform`, `toast`) and `MainView` with the title bar, tabs and periodic updates; `pages.rs` the
  five pages, the "Settings for" bar, the per-pass dialog (`PassDialog`), the target editor (`TargetEditor`), About and
  the error window; `text.rs` every text and decision that needs no window, with the tests; `keymap.rs` and `keytap.rs` the
  toggle key's names and its line to the display server's keyboard; `shm.rs` the saving of settings; `instance.rs`/`activation.rs` the single instance.
- The rows are drawn from the header: a switch or menu row reads its field when it is rendered and writes it with
  `MainView::set_bits`; nothing is kept in step by hand. A spin row is a `Spin` (a number field created once, in
  `add_setting_spins`, `FLOAT_SPINS`/`INT_SPINS`): a new float setting gets a line there and a `spin_row` in its page.
  The toolkit's number field steps its own text by 1 unless told not to: `Spin::new` turns that off (`set_step(None)`) and
  moves the value itself on the step event. After anything writes the header call `MainView::changed`. What else can change it (shmctl, a profile, a game's
  session) is picked up by `tick`, once a second, which compares `Live` (every status text and the settings'
  snapshot) with what is shown and redraws only on a difference.
- The window must not cost a game anything. No continuous redraw: `cx.notify()` only from an edit, from `tick` on a
  difference, and from `sample_telemetry` (5 Hz) only while the Status tab shows a graph with samples in it. Before a
  release that touches the GUI, measure the open, idle window beside a game (`nvidia-smi pmon` and `--query-compute-apps`
  for its GPU time and memory, `top` for its CPU). `main` removes `NEURAL_FORGE_ENABLE` from its own environment
  before anything else, so the layer is never loaded into the window's own Vulkan instance: check
  `/proc/<pid>/maps` for `libneural_forge_layer` after a change there.
- Work off the UI thread goes through `MainView::run` (a closure on the background executor, its result delivered to
  the view). Nothing that reads Steam's files, runs a program or waits may run in `render` or a click handler.
- Dialogs: always through `window.open_dialog` with `.overlay(false)` (the toolkit's dimming layer also paints the
  client-side shadow margin, a dark frame around the window on KDE and GNOME), the content in `named_group(..)`, the
  buttons in `dialog_buttons()` inside that group (not the toolkit's footer), a button given the focus when it opens
  (`ask` gives it to Cancel: every question here removes or risks something), and the text above the buttons capped
  with `dialog_room(window)` and scrollable: the toolkit cuts off a dialog taller than the window. Run what a button
  starts with `window.defer` after `close_dialog`, so a follow-up dialog is not the one that gets closed. Dialog
  builders run on every frame (`Fn`): one-shot state travels in `Rc<RefCell<Option<..>>>`, and content that changes
  while open is its own view.
- Enter in a dialog: the toolkit sends Enter to the dialog's `on_ok` before the focused button sees it. Every dialog
  therefore sets `on_ok` and acts as the focused button (`enter_key`), returning false (stay open) anywhere else; the
  per-pass dialog never closes on Enter (it belongs to its number fields).
- Nothing may show a message before the window has the toolkit's root view (that is, from `MainView::new`):
  `push_notification` panics there. `show_toast` prints to stderr instead, and what must be said at start runs from
  `MainView::started`.
- Screen readers: GPUI reports only elements with an id and a role. Use `label(id, text)` for text (never a bare
  string child in a dialog or row), `accessibility_label` on icon buttons, switches, number fields and buttons that
  repeat per row, and `toast()` for messages (it also feeds the `Announcer` live region; the toolkit's notifications
  are not announced). Ids are scoped by the nearest element with an id, so each `row` has its own.
- Fonts: `ui::init` names installed families for the UI and fixed-width fonts before the toolkit starts; without that
  GPUI warns about ".SystemUIFont" on every Linux system. Nothing may log at warn level on a normal start
  (`NEURAL_FORGE_GUI_LOG=info` shows the renderer picked). On Xvfb GPUI logs "Found no xinput mouse pointers" at error
  level, which `scripts/smoke-gui.sh` lets through (the D-Bus crate's warning that a service is unknown in a bare
  session is no longer shown: the logger caps zbus at error).
- The file dialog (portal) has no file-type filter in GPUI: what was picked is checked afterwards
  (`cube::load`, `extract_all`). A menu is a button with a popup menu (`menu_button`), not a searchable list.
- The toggle key: GPUI gives a key's name on the current layout, not its hardware code, and the layer reads evdev
  codes. `keytap.rs` keeps one second line to the display server's keyboard for the window's life (`KeyTap::open`): on
  Wayland a guest registry and seat on the toolkit's connection, with a `wl_keyboard` taken only while the row waits
  for a key (it receives the press with its evdev code) or for the moment it takes to read the keymap; on X11 a
  private connection that selects XInput 2.1 raw key presses while the row waits. `keymap.rs` (xkbcommon) names a
  stored code on the session's layout and, where no direct report comes, finds the key that carries a name on it (the
  fallback: it cannot tell the number pad from the main block). The layout in use: X11's XKB state group; on Wayland
  the group of a modifiers event if the compositor sends one to a new keyboard (Hyprland does not), else what the
  last bound press showed (`layout_of_press`). The keymap is read at start, when the window gets the focus and when a
  key is waited for. Never ask a seat for a keyboard before it has said it has one: a headless GNOME Shell 50
  segfaulted on that. Wayland things are tried in a private compositor or a VM first, never first on the desktop in
  use; key codes are tested with real ones (QEMU `sendkey` in a VM, `ydotool` on a session): KP_1 must store 79, and
  on a French layout the key in Q's place 16, shown as "A".
- One window per login (`instance.rs`): a second launch sends its request and waits 2 s for an answer. Until the
  window has drawn its first frame (`UiUp`, set from `window.on_next_frame` in `build_ui`, not from the UI loop's
  first wait for a request), the listener thread answers "starting" at once and the launch waits up to 15 s from its
  request for the UI's "ok"; if the window is not up by then it says the first copy is still starting, as information
  (`tell_without_window`: zenity `--info`, kdialog `--msgbox`, a notification of normal urgency with the desktop's own
  expiry) and exits 0. Once the window is up, 2 s without "ok" is reported as the running copy not responding, an error
  (`say_without_window`: stderr and the first of zenity, kdialog, notify-send, the notification service over D-Bus
  and xmessage, the order AppRun uses too) with exit 1. A dialog program counts as shown only if it ended with 0 or 1
  and its stderr (a private unlinked temporary file, read after it exits, never a pipe read to its end) does not say
  it could not open the display. A copy that closes the connection unanswered (end of file, reset, broken pipe) is
  going away: the launch asks again every 75 ms for up to 2 s from the first such close, and takes its place once
  nobody listens, never calling it stuck. `instance::release` removes the socket file, then shuts the listener down
  (in that order); it runs when the last window closes and at the start of `no_window`, which also drops the
  `Activations` so launches queued there hear the end at once; a connection accepted after the release is dropped
  unanswered. `main` installs the layer before that check. Tests of the messages run stand-in programs on a private
  `PATH` with `DISPLAY=:9999`, no `WAYLAND_DISPLAY` and a private `XDG_RUNTIME_DIR`, never a real dialog.
- A dialog's continuation runs through `Once` from every way out: the toolkit's `close_dialog` (what a button calls)
  does not call the dialog's `on_close`.
- The logger caps what each source may show (`CAPS` in `main.rs`, whatever `NEURAL_FORGE_GUI_LOG` asks for): the
  toolkit's chatty dependencies (tracing, naga, ...) from warn up; zbus (its "Failed to populate properties cache"
  warning for a service a desktop lacks) and `gpui::window::a11y` (its harmless "set_focus called more than once in a
  single frame" with a screen reader on) from error up.
- The window's first size is fitted to the screen (`default_window_size`). On Wayland the toolkit reports a
  fractionally scaled screen too small and the window's scale arrives only after it is up, so the fit is made again
  then (`refit`, `refit_armed` in `text.rs`: Wayland only, one screen, the window still at its first size, within its
  first 3 s; a moment after the scale is known and once more if KWin's pending configure undoes it). A toolkit quirk
  seen on KDE at 150%: after maximize and restore the window comes back at the first, too-small size, because the
  toolkit restores to the bounds it remembers from before the refit (gpui-pre-linux `wayland/window.rs`,
  `state.window_bounds`); not patched here.
- Clicks, the wheel and typing can be tested on the private X display with `xdotool` (window-relative positions in
  device pixels, twice the logical ones at 2x), reading the scratch `config.ini` afterwards for what was saved.
- Before a release, on GNOME, KDE Plasma, Hyprland and COSMIC: start, every tab, a number field (type, step, Enter),
  a menu, the per-pass dialog, a question (Reset), the file dialog, light/dark switch, a second start; and with no
  GPU: lavapipe (`VK_DRIVER_FILES` at the `lvp_icd` manifest), OpenGL ES only (`VK_DRIVER_FILES=/dev/null`), no
  driver (the app's own message), no loader (AppRun's message). Retake the screenshots when a page's look changes.

## Tested games and the sweep

- `data/games.toml` is the per-game compatibility list the app is built with (`crates/supervisor/src/compat.rs`):
  each tested game's engine, API, DLSS features, where the model runs with Ray Reconstruction off and on, its
  HDR route, setup notes and the version it was last verified in. The Status and Setup tabs, `games --tested`,
  `doctor`, Save report and the game sweep read it. A newly tested game gets an entry (and a row in the user guide's
  "Tested games"); a change that moves the model elsewhere in a game updates its entry in the same commit.
- Before a release that touches the layer (above all `preupscale`), run the game sweep on the test machine with
  the release build: `scripts/game-sweep.sh --layer <build>/libneural_forge_layer.so --cli <build>/neural-forge-cli`
  (docs/RUNNING_AND_MEASURING.md, section 11), in tmux; it takes the GPU lock and puts back what it changes. Every
  run must pass (each run holds the GPU lock only while it runs), and each is recorded (`vw`, `--vw`): a run passes
  only when the recording flags nothing and its contact sheet looks right against the model off; log numbers alone never pass; record the passing games' `verified`/`verified_on` in `data/games.toml`. Runs set the game's own
  resolution to 2560x1440 (the desktop stays at 2560x1440, 288 Hz, VRR), with HDR for games that have it.

## Test builds

- A fix for an issue someone else reported goes to them before it reaches `main`: push it on its own branch,
  carrying the version bump and CHANGELOG section it will be released with, open a PR (CI must pass), then run
  `gh workflow run release.yml --ref BRANCH -f issue=N`. The Build and Release workflow builds that branch
  exactly as a release and publishes it as a pre-release named "Test build for #N (BRANCH)" with the tag
  `test-BRANCH-RUN` (never `v*`, never the latest release, so the AppImage updater does not offer it).
- Link the pre-release in the issue and ask the reporter to try it. Merge and release only once they confirm
  it fixes the problem (or say plainly why you release without that). After the release, delete the test
  pre-releases and their tags (`gh release delete TAG --cleanup-tag --yes`).

## Changelog

- One `## X.Y.Z — YYYY-MM-DD` heading per released version, newest first. No entries for builds
  that were never released.
- Write each entry for the people using the app: what changed for them and anything they need to
  do. Leave out implementation detail (file paths, flags, internal names, CI and packaging changes)
  unless a user needs it to act.
- The release page is the version's section written out in full (`scripts/release_notes.py`), never
  a link to the changelog. The release fails if the section is missing.
- The AppStream `<releases>` list is generated from the headings
  (`scripts/sync_appdata_releases.py`, run by `build-appimage.sh` and checked in CI). Don't edit it
  by hand.

## Runtime constraints

Use only Neural Forge-owned paths, `NEURAL_FORGE_*` variables, and the
`VK_LAYER_neuralforge_neural` identity. Keep NVIDIA DLL names, NGX exports and
`DLSSNR.*` parameters unchanged. Do not copy or move ambiguous upstream config,
shared memory, or Wine prefixes. Import DLLs explicitly into Neural Forge's data dir.

Target-process filtering and the kernel ownership lease must remain effective before
any process can resize or write a channel. Preserve the GTA baseline: the model before the
upscaler on the native backend, the network on every frame while the camera is still and on every
second frame in motion with the edit cache carrying it between (docs/ARCHITECTURE.md 4.10;
`NEURAL_FORGE_EDIT_CACHE=off` runs it every frame), model_resolution=1. Since 3.0 the network runs in
the layer (`crates/native`, `crates/layer/src/preupscale/native.rs`; docs/NATIVE_BACKEND.md); the
2.x Windows helper and everything around it (runners, Wine prefix, DMA-BUF, NGX) are removed and
not to be brought back. Do not lower model resolution or turn the model off by default without
explicit user authorization. The one Windows-side piece is the NGX wrapper for Proton (`crates/ngx-wrapper`,
docs/PROTON_NGX.md): it never runs NGX or the network, it forwards NVIDIA's NGX core unchanged and records the
parameters the game gives DLSS for the layer (`ngx_ring.rs`). It ships only inside Proton copies made on the user's
machine (`proton_tool.rs`), never as a published Proton build. Its records are the layer's only source of what DLSS
works on: do not bring back rules that infer DLSS's images, jitter, units, reset or Ray Reconstruction from its
kernels.

Do not reapply the reverted capture/composition fence changes. Validate actual GPU
operations and measure before and after every performance change. The matched comparator
is the released 2.0.10 against the native backend (docs/NATIVE_BACKEND.md, Phase 3), taken with
`scripts/gta-bench.sh` and mods off. Keep the Rust implementation independent;
review licenses before source reuse.

Current target-machine evidence and unresolved Vulkan errors are in
[docs/HARDWARE_VALIDATION.md](docs/HARDWARE_VALIDATION.md).

## Pre-upscaler path

Since 2.0 the model runs **before** DLSS Super Resolution by default (`crates/layer/src/preupscale.rs`,
docs/PRE_UPSCALER_DESIGN.md): on a device with `VK_NVX_image_view_handle`, the layer holds the game's DLSS submit,
runs the network on the render-resolution frame (every frame while the camera is still, every second frame in motion
with the edit cache carrying it between; with DLSS's motion vectors and the jitter for the history) and writes the
answer back before DLSS runs. Games that record their frame in DLSS's own command buffer are held inside it: the whole
hold is recorded into that buffer, and the application's compute state is bound again after it
(`preupscale/inline.rs`, `preupscale/restore.rs`).

Everything about DLSS comes from NGX's records (`ngx_ring.rs`, docs/PROTON_NGX.md): the NGX wrapper writes one before
NVIDIA's `EvaluateFeature` and an end record after it returns. A command buffer belongs to an evaluate when one of
its launches names the evaluate's colour input (the registered views' handles in the launch's parameters) between the
two; Super Resolution's is the hold point (the split in front of it, or inside it at that launch), and the images,
exposure, jitter, `MV.Scale` and reset of a hold are its record's. The identification (the images watched for their
layouts) is the newest Super Resolution record on the device. DLSS Ray Reconstruction (NGX's feature 13) is never held
before the upscaler: its input is the noisy ray-traced frame, so the model runs after DLSS's output there (the hold
after DLSS goes after the evaluate's last write of its output, as many writes as the last ended evaluate made) or on
the finished frame. DLSS Frame Generation's buffers carry no Super Resolution evaluate and go through untouched. No
rule reads DLSS's kernels (names or parameter words) to decide anything: without records (no Neural Forge Proton,
`NEURAL_FORGE_NGX_RING=off`) nothing is held before or after DLSS, and the status says the wrapper is missing.
Everything else (no NVX, no DLSS, native) keeps the post-upscaler path. `NEURAL_FORGE_PREUPSCALE=off` is the rollback
switch: since 3.0 it also loads no network (the native backend needs a mode on), so the post path presents every
frame untouched; the A/B with the model on the finished frame is `placement=present`. `dump`, `identity` and
`roundtrip` are diagnostics. Rule: with `NEURAL_FORGE_PREUPSCALE=off` the post path must stay byte-identical to 1.1.0
(the hooked-command list is the default set, no NVX entry points resolved, nothing of `preupscale` reachable), and a
device without NVX must not get the tracking under the default either; the tests in `lib.rs::probe_command_tests` and
the smoke test's `off` pass guard this.

## Historical evidence

[Pre-rename development notes](docs/history/development-before-neuralforge.md) retain
old commands, measured failures and toolchain investigations as historical evidence.
Those old deployment recipes are not current instructions. The historical GTA handoff
is [docs/history/handoff-2026-09-12-fps-freeze-regression.md](docs/history/handoff-2026-09-12-fps-freeze-regression.md).

## Composition invariants (checklist for every change to the pass)

Adopted from upstream DLSS5VKLayer's `DEVELOPMENT.md`; check each before committing shader or
composition changes.

- **Default-identical toggles.** A new control's default must leave the shipped picture bit
  for bit as it was (or the change says in its commit why the picture moves). Transfer modes
  are identical at 100% model resolution; compare off is a no-op.
- **Push constants append-only, struct == shader.** `composition::gpu::PushConstants` and
  `compose.comp`'s `Params` block match field for field, and new fields go at the end of both.
  The same for `ShmHeader`: append, bump `SHM_VERSION`, update the pinned offsets.
- **The encode is reproduced wherever the proxy is compared.** Anything that compares the
  model's answer with the frame must first put the frame through the same white divide and
  knee the encode used (`SoftKneeLuminance(original / white_point)`), passthrough included.
- **Drain before free.** Nothing the GPU may still read is destroyed until its fence (or the
  device) has been waited on: slot resources, swapchain images, the network's buffers.
- **Tested on lavapipe, and the test fails without the change.** Every composition behaviour has
  a GPU test (`composition::gpu::tests::compose_once` makes one short); run it once with the
  change reverted to see it fail.

## Shared memory

The owner's Claude sessions (local, cloud and chat) share notes in the Google Drive folder "Claude Memory", one doc per topic; this repo's doc is "app: neural-forge". If you have the Google Drive connector:

- Before starting work, read that doc and the "prefs:" docs for decisions, things already tried or ruled out, and the release state.
- After meaningful work (a release, a decision, a failed attempt worth not repeating, an open issue), update the doc. The connector can't edit a doc's text: create a replacement doc with the same title in that folder, then trash the old one. The folder's "README - how this folder works" doc has the conventions.
