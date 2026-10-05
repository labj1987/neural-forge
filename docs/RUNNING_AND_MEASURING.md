# Running and measuring

How to build Neural Forge, put a build on a test machine, measure it, and read what it reports.
Everything here is how the numbers in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) and
[PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md) were taken. The test machine in the examples is
reached as `ssh lordnikon` (RTX 5070, NVIDIA 615.71.09, Steam with GTA V Enhanced and
Proton-CachyOS); substitute your own host.

## 1. Build and test

Native crates (protocol, layer, GUI, CLI, supervisor). Don't use `--workspace`: the helper targets
Windows only.

```bash
cargo build --release
```

The Windows helper, cross-compiled (needs `mingw-w64` and the `x86_64-pc-windows-gnu` target in the
stable rustup toolchain; [CLAUDE.md](../CLAUDE.md) has the toolchain notes):

```bash
cargo +stable build --release --target x86_64-pc-windows-gnu -p neural-forge-helper
```

The AppImage (layer 64- and 32-bit, helper, GUI, CLI):

```bash
CARGO_HELPER='cargo +stable' bash build-appimage.sh
```

Checks, in the order CI runs them:

```bash
cargo test
```

```bash
python3 scripts/test_install.py && python3 scripts/check_shaders.py && python3 scripts/sync_appdata_releases.py --check
```

```bash
bash scripts/smoke-test.sh && python3 scripts/check_namespace.py
```

- `scripts/smoke-test.sh` runs `crates/layer/examples/smoke.rs` through the real Vulkan loader
  with the layer enabled, in a debug build. Run it after any change near device creation: it
  caught a null-pointer slice and a panicking loader helper that validation layers and unit tests
  missed (EXTERNAL_MEMORY_HOST_DESIGN.md). It also checks that `NEURAL_FORGE_PREUPSCALE=off` logs
  no `[preupscale]` line and that the probe stays off unless asked.
- `scripts/helper-test.sh` cross-compiles the helper's examples and runs its self-tests under Wine
  (`guard_test`, `spoof_test`, `spoof_install_test`).
- GPU tests run on lavapipe in CI. `NEURAL_FORGE_REQUIRE_VULKAN=1` makes a test fail instead of
  skipping when no Vulkan device exists.
- Tests never touch real XDG directories: point `XDG_*_HOME` at a scratch directory when running
  the CLI or `install` by hand.

## 2. Deploy to a test machine

```bash
scripts/deploy-rig.sh lordnikon
```

It builds the AppImage from the working tree, copies it to `~/AppImages/neural-forge.appimage` on
the host, installs the layer and helper from it with `neural-forge-cli install`, and prints the
SHA-256 of the installed `libneural_forge_layer.so` and `neural-forge-helper.exe` next to the
AppImage's copies. It fails if they differ. It leaves no local build and no backup behind.

Then restart the helper on the host so it runs the new build, and check it:

```bash
ssh lordnikon '~/.local/share/neural-forge/bin/neural-forge-cli restart && ~/.local/share/neural-forge/bin/neural-forge-cli shmctl status | head -12'
```

(`install` puts the CLI in `~/.local/share/neural-forge/bin`; the shorter `neural-forge-cli` in the
commands below assumes that directory is on the PATH.)

**After a shared-memory protocol change** (`SHM_VERSION`), the helper refuses an old `shm.bin`.
Stop the helper, make sure nothing has the file open (no game running), then remove it:

```bash
ssh lordnikon 'neural-forge-cli stop; D=/tmp/neural-forge-$(id -u); fuser "$D/shm.bin" || rm "$D/shm.bin" "$D/shm.bin.owner"'
```

`scripts/rig-test.sh [host] [seconds]` is the older end-to-end check: it builds and deploys both
halves, launches the game, and reports frame rate, round-trip timings, composition mode and any
`Xid`, read from shared memory and the layer log.

## 3. The unattended GTA benchmark

`scripts/gta-bench.sh` runs GTA V Enhanced's built-in benchmark with a chosen layer environment,
with no one at the machine. It runs on the host, or from the dev machine with `--host`. Steam must
be running on the host. The game is launched through the same Steam Linux Runtime entry point and
Proton that Steam uses, with only the environment given (Steam's launch options are bypassed), and
`-benchmark -benchmarkIterations 1 -benchmarkFrameTimes`. It exits by itself.

```text
gta-bench.sh [--host HOST] [--set key=value]... <label> <VK layers above MangoHud> [env...]
```

Effect off (no Neural Forge layer), then on, both with GTA's script mods off:

```bash
scripts/gta-bench.sh --host lordnikon nroff-1 '' 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
scripts/gta-bench.sh --host lordnikon nron-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

The 1.x path for comparison:

```bash
scripts/gta-bench.sh --host lordnikon post-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=off 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

Summarise, and average three runs of one configuration:

```bash
scripts/bench-report.py --host lordnikon --mean nron-1 nron-2 nron-3
```

**What the numbers are.** Pass 4 is the long free-roam pass, about 117 s. Real fps is the pass-4
frame count divided by the sum of its frame times, from GTA's own frame-time file. Shown fps is
MangoHud's per-frame log counted over pass 4's wall-clock window; MangoHud is always the last
layer, so it sits below any frame generator and counts generated frames. GPU utilisation and
power come from `nvidia-smi` over the same window. The report also gives the median of every
field of the layer's `[sync]` lines, and "held before upscaler/s" (from `holds_per_s`) when the
model ran before the upscaler. `scripts/bench-report.py --self-test` checks its parsing.

**Rules that matter for the numbers:**

- **Script mods off** (`WINEDLLOVERRIDES=xinput1_4=b;dinput8=b`). The Enable All Interiors script
  caps GTA at about 63 fps with the GPU half idle (OPENDLSS_REVIEW.md).
- **No remote-desktop session connected.** The runner warns when one is open on port 3390; it
  costs about 8 fps.
- **Three runs** when the difference you are measuring is under about 3 fps (the run-to-run spread
  at 1440p).
- **Model resolution is never left lowered.** `--set working_scale=0.66` sets it with
  `neural-forge-cli shmctl set` for one run and restores the previous value afterwards, printing
  `shmctl status` both times. Any live setting works the same way (`--set model_interval=2`).
- **Early exit at "Game Init".** GTA sometimes crashes at `GTA5_Enhanced.exe+0x12c6eb` during
  start-up, with or without Neural Forge. The runner says "launch exited early" and writes no
  `benchmark.txt`. Wait five minutes and run it again.
- **Frame generation does not always engage.** With DLSS Frame Generation on, check shown fps:
  some launches present only real frames. GTA's setting is `FrameGenType` in `settings.xml`
  (0 off, 1 on).

**4K runs.** GTA renders at the desktop's size under Proton, so switch the host's desktop to
3840x2160 at scale 1.0 first, temporarily, with GNOME's `gdctl set` (not persistent), and set
`ScreenWidth`, `ScreenHeight` and `RefreshRate` in GTA's `settings.xml`. Afterwards put both back
and check: `gdctl show` for the desktop, and `sha256sum` of `settings.xml` against the value taken
before the change. Remove any backup made for the test.

Results land in `$NF_BENCH_DIR/<label>` on the host (default `~/nf-spike/gta`): GTA's Benchmark
and FrameTimes files, MangoHud's CSV, `nvidia-smi` samples, and `launch.log` with the layer's lines.

## 4. Where the edit lands: the pan reproducer and agreement

`vkcube`'s background never moves, so it cannot show an edit landing on the wrong frame.
`crates/layer/examples/pan.rs` presents a detailed picture moving `--px-per-frame` right and half
that down every frame, with the frame number stamped in the top-left corner. Frame N is the same
picture in every run. The layer engages on it like on a game (helper running,
`NEURAL_FORGE_ENABLE=1`).

```bash
cargo build --release -p neural-forge-layer --example pan
```

```bash
NEURAL_FORGE_ENABLE=1 ./target/release/examples/pan --width 2560 --height 1440 --px-per-frame 6 --grain 0.3 --frames 1100 --capture-at 900 --capture-frames 120
```

```bash
NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PIPELINED=1 ./target/release/examples/pan --width 2560 --height 1440 --px-per-frame 6 --grain 0.3 --frames 1100 --capture-at 900 --capture-frames 120
```

Other flags: `--contrast C`, `--present-mode fifo|immediate|mailbox`, `--frames 0` (run until
closed), `--save DIR` (every frame as a PNG). `--capture-at`/`--capture-frames` make pan request a
capture series right before a given frame, so every run captures the same frames.

`scripts/agreement.py` (needs Pillow) matches two runs' frames by their stamped number and reports
the correlation of the two edit fields (`composited - original`; 1.0 means the same edit in the
same place) and the test run's edit strength on the original's sharpest 10% of pixels, as a
percentage of the reference's:

```bash
python3 scripts/agreement.py REF_SERIES_DIR TEST_SERIES_DIR
```

```bash
python3 scripts/agreement.py --runs ref1:test1 ref2:test2 ref3:test3
```

Two synchronous runs may not score exactly 1.0 if the model's answer to the same frame varies.
Compare one synchronous run against another first: that score is the ceiling for any other path.

**Capturing frames from a game.** `neural-forge-cli shmctl capture --frames N` captures the next N
presented frames as `~/.local/share/neural-forge/captures/series-<ms>/<seq>-{original,composited}.png`
plus an `index.tsv`. `capture` with no `--frames` is a one-shot dump. While the model runs before
the upscaler there is no separate original (the enhanced frame went through DLSS), so both images
of each pair are the presented frame.

## 5. Diagnosing the model before the upscaler

### The probe

`NEURAL_FORGE_PROBE_NGX=1` logs the game's DLSS work as it reaches Vulkan: NVX registrations,
kernel names, launches per command buffer, which submit carries them, and the command sequence
inside the launch-bearing buffer. Set GTA to DLSS Quality or Balanced (not DLAA) with frame
generation off:

```bash
scripts/gta-bench.sh --host lordnikon probe-ngx-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PROBE_NGX=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
ssh lordnikon "grep -F '[probe-ngx]' ~/nf-spike/gta/probe-ngx-1/launch.log | head -200"
```

What to look for, and what GTA showed, is in [PRE_UPSCALER_PROBE.md](PRE_UPSCALER_PROBE.md). Run it
once on any new game before claiming the model before the upscaler works there.

### The modes

`NEURAL_FORGE_PREUPSCALE` selects what happens at the DLSS submit:

| Value | What it does | Use |
|---|---|---|
| unset or `model` | Encode, model, decode, every frame (the default). | Normal play. |
| `off` | Nothing; the 1.x path everywhere, byte-identical to 1.1.0. | A/B comparisons, rollback. |
| `dump` | Once (and again on each `shmctl capture`): write DLSS's colour, depth, motion vectors and the 1x1 exposure images to `~/.local/share/neural-forge/captures/preupscale-<ms>/` (`colour.rgba16f`, `depth.r32f`, `mvec.rg16f`, `exposure.json`, `meta.json`, `colour-preview.png`). The frame goes on untouched. | Getting real DLSS input for offline experiments. |
| `identity` | Capture and write the same bytes back, raw; no helper. | The hold's own cost. Set `--set enabled=0` so the 1.x path does not run too. |
| `roundtrip` | The HDR encode and decode with the encoded frame as the answer; no helper. | That the transform alone leaves the picture unchanged. |

```bash
scripts/gta-bench.sh --host lordnikon pu-identity-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=identity 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

`NEURAL_FORGE_PREUPSCALE_PAPER_WHITE=2.5` (or 4) tries another paper white in `model` or
`roundtrip` mode.

### Driving the helper without a game

`crates/protocol/examples/trigger_helper_roundtrip.rs` plays the layer against a running helper.
With `--rgba16f` it sends a raw frame (for example a `dump`'s `colour.rgba16f`) as an RGBA16F
request, prints whether each answer was a model evaluation or an echo, and statistics of input and
answer:

```bash
cargo +stable build --release -p neural-forge-protocol --example trigger_helper_roundtrip
```

```bash
scp target/release/examples/trigger_helper_roundtrip lordnikon:/tmp/nf-roundtrip
```

```bash
ssh lordnikon 'export NEURAL_FORGE_SHM=/tmp/neural-forge-1000/shm.bin NEURAL_FORGE_UID=1000; /tmp/nf-roundtrip --rgba16f /tmp/colour.rgba16f --width 1486 --height 836 --out /tmp/answer.rgba16f --repeat 64'
```

Run it with the helper started by the CLI and no game running (the layer would drive slot 0 too).
Use `--repeat 32` or more: the first requests after a size or format change are echoed while the
feature builds. `--rgba8 FILE` sends an 8-bit frame instead; with no file it sends one synthetic
64x64 frame on the slot given. `scripts/hdr_encode.py` (numpy, Pillow) encodes a dump for the model
and judges the answer (`info`, `encode`, `judge`, `montage`); it is how E1b chose the encode.

`crates/helper/examples/optical_flow_rig_check.rs` runs the helper's optical-flow path on real
hardware under the helper's runner and checks the vectors and for stalls; `--hdr` checks the
RGBA16F input pass.

### Fault injection

Helper variables are read when the helper starts. `neural-forge-cli restart` passes its own
environment to the helper, so prefix the restart; a plain restart afterwards undoes it:

```bash
ssh lordnikon 'NEURAL_FORGE_FAIL_CREATE=6@2 neural-forge-cli restart'
```

- `NEURAL_FORGE_FAIL_CREATE=N[@K]`: let K model builds through, then fail the next N without
  calling NGX. Exercises the retry schedule, the NGX re-initialisation, `model_up=0` and the
  layer's circuit breaker. Force a rebuild mid-run with a tuning change, for example
  `neural-forge-cli shmctl set intensity 1.01` (and set it back to 1 afterwards).
- `NEURAL_FORGE_HDR_FLAGS=hdr|sdr|autoexp0`: NGX creation flags for RGBA16F frames.
- `NEURAL_FORGE_HELPER_DELAY_MS=N`: delay every answer.

## 6. Reading the logs

**Where they are.** The layer logs to `NEURAL_FORGE_LOG` if set, otherwise to the game's stderr
(the Proton log; in `gta-bench.sh` runs, `launch.log`). The helper logs to
`~/.local/state/neural-forge/helper.log` (moved to `helper.log.1` past 20 MB at a helper start).
`scripts/check-stalls.sh [LOG]` checks a layer log for fence-wait timeouts and breadcrumb dumps
and summarises engage/disengage transitions and the frame rate.

### Layer lines

| Line | When | What it says |
|---|---|---|
| `[present] N fps (M/s composited by the effect) over Ts` | every 5 s, per presenting process | The real presented frame rate, effect on or off. |
| `[sync] WxH: total= capture_gpu= copy_out= meter= wait_answer= helper= rest(readback+compose)= zc= gpu_capture= gpu_compose=` | every 300 composed frames (after the upscaler) | Per model frame: total time on the present thread; capture submit to observed completion (includes the game's own frame); CPU copy out (0 with zero copy); the white meter; the wait for the answer; the helper's own time; readback and compose; whether zero copy was used; the layer's own GPU time from timestamps. |
| `[preupscale] mode <mode> (default) in pid N` | once | The mode for this process. |
| `[preupscale] device ...: vkGetImageViewHandleNVX present, ...` or `no VK_NVX_image_view_handle, ...` | per device | Whether the device can hold at all. |
| `[preupscale] colour input: image 0x... (WxH R16G16B16A16_SFLOAT ...), depth ..., motion vectors ..., exposure input 0x...` | on (re)identification | What the layer recognised as DLSS's inputs. |
| `[preupscale] resources for WxH (padded PWxPH) built: zero-copy (SHM regions imported)` | per extent | The hold's resources. |
| `[preupscale] mode=model extent=WxH ... holds=N hold_ms median= capture_gpu_ms median= writeback_gpu_ms median= misses= (total ) holds_per_s=` | every 300 holds | The hold's cost and rate. `holds_per_s` equals real fps when every frame is held. |
| `[preupscale] phases ms (median): prep= capture_wait= round_trip= helper_busy= handoff= writeback=; capture_wait max= (session max , bound  ms)` | after each summary | Where the hold's time goes (see [ARCHITECTURE.md](ARCHITECTURE.md), 4.8). `capture_wait max` is the longest capture fence wait of the window and of the session, next to the bound it is waited with (`preupscale::FRAME_CAPTURE_WAIT`): the numbers to collect across loading, resolution changes, alt-tab and shutdown before that bound is changed. |
| `[preupscale] frame went to DLSS untouched: <why>` | at most once per 5 s | A miss: exposure not usable, answer late, an echo, a failure. |
| `[preupscale] breaker open: ...`, `breaker still open after Ns ...`, `breaker closed ...` | on change, every 30 s while open | The circuit breaker. |
| `[preupscale] launch-bearing submits: N held reading the colour input, N held undecided, N forwarded untouched` | every 3000 forwarded submits | Frame generation's submits being left alone. |
| `[hotkey] watching N keyboard(s) through evdev` / `watching XInput2 raw keys on :0` / `no way to read the keyboard here ...` | first poll | Which keyboard backend the toggle key uses. |
| `[layer] fence wait timed out after 5s at <site> ...` | on a timeout | A driver stall without device loss; a breadcrumb dump follows. |

### Helper lines

| Line | What it says |
|---|---|
| `[ngx] feature WxH hdr=1: DLSSNR.Hdr=1 DLSSNR.SDR=0 AutoExposure=1` and `VULKAN_CreateFeature(18) -> 0x1 ... size=WxH hdr=1` | A feature build and its result (`0x1` is success; `0xbad00002` and others are failures). |
| `[helper] frame A -> B; rebuilding N pass(es)` | A size or format change. |
| `[helper] the model built again at ... after N failed attempt(s)` | Recovery after failed builds. |
| `[frame] stages ms (median) WxH: n= idle= setup= thumb= upload= flow= ngx_rec= eval= download= fence_waits= publish= busy=` | Every 300 evaluated slot-0 requests: the helper's own time per stage. `upload`, `flow` and `eval` are record-and-submit times; `fence_waits` is the one wait for the GPU work; `busy` is the whole request. |
| `resetting model history (...)` | The model's history was reset after a gap, a failed evaluate or a format change. |
| `[mvec] scene cut detected` | A scene cut reset the flow and history. |

### `neural-forge-cli shmctl status`

The `# live status` block:

| Field | Meaning |
|---|---|
| `helper_state` | 0 starting, 1 no Vulkan, 2 no binaries, 3 model failed, 4 running, 5 stopped (named in the output). |
| `model_up` | 1 when the model is built; 0 while it cannot be. |
| `helper_reason` | Why the model is not up (empty when it is). |
| `helper_frames` | Requests the helper has answered. |
| `helper_upload_ms`, `helper_eval_ms`, `helper_readback_ms` | Record-and-submit times of the last request, and its one wait. |
| `helper_busy_ms` | The helper's whole time for its last slot-0 request. |
| `layer_frames`, `layer_ms` | Frames the layer captured, and its last per-frame time. |
| `layer_capture_gpu_ms`, `layer_compose_gpu_ms` | The layer's own GPU time from timestamps (after the upscaler). |
| `preupscale_state` | 0 off, 1 waiting for DLSS input, 2 holding, 3 paused. |
| `preupscale_extent`, `preupscale_hold_ms`, `preupscale_misses` | DLSS's render size, the last hold's CPU time, and holds whose answer did not make it. |
| `layer_measured_white`, `layer_composition_up`, `capture_request` | The white meter, whether the compose is up, a pending capture. |

The `# settings` block lists every live setting by name; `shmctl set NAME VALUE` and
`shmctl toggle NAME` change them.

## 7. Validation layers and the loader

- Load Khronos validation with synchronization checks alongside the layer:
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1`. The older
  `VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` is deprecated and silently loses
  to it.
- To test a layer build without installing it, use `VK_ADD_LAYER_PATH`, not `VK_LAYER_PATH`: the
  latter replaces the search path and hides the system's validation layer manifest.
- `VK_LOADER_DEBUG=layer` prints the layer order (`vkCreateDevice layer callstack setup to:`,
  application at the top). The loader does not order implicit layers among themselves; use
  `VK_INSTANCE_LAYERS` to force an order (FRAMEGEN_SPIKE.md).
- The capture hot-path benchmark runs a real GPU: build the layer's release test binary, copy it
  to the host, and run it with `NEURAL_FORGE_BENCH=1` and the test name
  `capture_hot_path_cost_per_present`. Results are in HARDWARE_VALIDATION.md.

## 8. Checking the picture by eye

Numbers cannot say whether the picture looks right; Alex's eyes decide. The 2.0 check (about 15
minutes, GTA V Enhanced, launch option `NEURAL_FORGE_ENABLE=1 %command%`, DLSS Balanced):

1. Frame generation off. Story Mode, daylight, a street with signs and trees. Press F11 a few
   times. Does "on" look better (skin, foliage, edges, signs), and are the colours right?
2. Turn the camera quickly left and right. Any second copy of edges or text, smearing, or shimmer
   that isn't there with F11 off?
3. At night with headlights and street lights: blown-out lights, colour fringes, flicker?
4. Frame generation on in GTA's settings, at 2x, 3x and 4x: does it look and feel right with F11
   on? (Check the fps counter that frame generation actually engaged.)
5. Optional A/B with the 1.x path: add `NEURAL_FORGE_PREUPSCALE=off` before `%command%`, relaunch,
   look at the same spots, then remove it.

Report better / same / worse for 1-3, how 2x/3x/4x look, and anything odd. The 2.0 result: an hour
of play at DLSS Frame Generation 4x, "everything is working beautifully" (HARDWARE_VALIDATION.md,
"2.0.0").

## 9. Release checklist

1. Bump `version` in the workspace `Cargo.toml` (`[workspace.package]`); `Cargo.lock` follows on
   the next build.
2. Add a `## X.Y.Z — YYYY-MM-DD` heading to `CHANGELOG.md`, with an em dash (`—`), not a hyphen:
   `scripts/sync_appdata_releases.py` only recognises that form. Move the release's entries out of
   "Unreleased".
3. Regenerate the AppStream releases list from the changelog, and check it:

   ```bash
   python3 scripts/sync_appdata_releases.py && python3 scripts/sync_appdata_releases.py --check
   ```

4. Run the checks in section 1.
5. Commit, then tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` runs on `v*` tags:
   tests, the shader check, the smoke test, then builds the AppImage and its `.zsync` and publishes
   the release. The build fails rather than publish without the `.zsync`.
6. Deploy the release to the test machine and confirm the installed hashes (section 2).
