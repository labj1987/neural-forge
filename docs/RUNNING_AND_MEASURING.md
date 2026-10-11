# Running and measuring

How to build Neural Forge, put a build on a test machine, measure it, and read what it reports.
Everything here is how the numbers in [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) and
[PRE_UPSCALER_DESIGN.md](PRE_UPSCALER_DESIGN.md) were taken. The test machine in the examples is
reached as `ssh <host>` (RTX 5070, NVIDIA 615.78.08, Steam with GTA V Enhanced and
Proton-CachyOS); substitute your own host. Up to 2.0.10 the model ran in a Windows helper; sections
that measured it say so.

## 1. Build and test

Once, the pinned tools the native backend (`crates/native`) builds with:

```bash
bash scripts/fetch-native-tools.sh
```

Every crate (protocol, layer, native, GUI, CLI, supervisor):

```bash
cargo build --release
```

The AppImage (layer, GUI, CLI):

```bash
bash build-appimage.sh
```

Checks, in the order CI runs them:

```bash
VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | paste -sd:) cargo test
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
- GPU tests run on lavapipe, in CI and locally with `VK_DRIVER_FILES` as above. Without it, on a
  machine with an NVIDIA GPU, they run on the real device, in parallel, and can run out of memory. `NEURAL_FORGE_REQUIRE_VULKAN=1` makes a test fail instead of
  skipping when no Vulkan device exists.
- Tests never touch real XDG directories: point `XDG_*_HOME` at a scratch directory when running
  the CLI or `install` by hand.

## 2. Deploy to a test machine

```bash
scripts/deploy-rig.sh the test machine
```

It builds the AppImage from the working tree, copies it to `~/AppImages/neural-forge.appimage` on
the host, installs it with `neural-forge-cli install`, and prints the SHA-256 of the installed
`libneural_forge_layer.so` next to the AppImage's copy. It fails if they differ. It leaves no local
build and no backup behind. The next game started uses the new layer; check it with:

```bash
ssh <host> '~/.local/share/neural-forge/bin/neural-forge-cli doctor && ~/.local/share/neural-forge/bin/neural-forge-cli shmctl status | head -12'
```

(`install` puts the CLI in `~/.local/share/neural-forge/bin`; the shorter `neural-forge-cli` in the
commands below assumes that directory is on the PATH.)

**After a shared-memory protocol change** (`SHM_VERSION`), the GUI and CLI refuse the old
`shm.bin` until a game with the new layer has started: the layer re-creates it.

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
scripts/gta-bench.sh --host <host> nroff-1 '' 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
scripts/gta-bench.sh --host <host> nron-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

The 1.x path for comparison:

```bash
scripts/gta-bench.sh --host <host> post-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=off 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

Summarise, and average three runs of one configuration:

```bash
scripts/bench-report.py --host <host> --mean nron-1 nron-2 nron-3
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
  `benchmark.txt`. Run it again straight away.
- **Test as it is played: frame generation on.** Release checks run with GTA's `settings.xml` as
  the maintainer has it (`FrameGenType` 1, `dlssFrameGenMode` 0/1/2 = 2x/3x/4x). Frame-generation-off runs
  are comparisons only; 2.0.2 passed them and failed with frame generation on.
- **Automated runs check that the model is applied; frame generation is checked in real play.** One
  run per game per change, with GTA's settings as the maintainer has them: every DLSS frame held
  (`holds_per_s` equal to the real frame rate), model answers written back (0 misses, no echoes), no
  `not holding` refusal, no `fence wait timed out`, no Xid. GTA decides at each loading screen whether
  to run DLSS Frame Generation and in benchmark launches it often does not, with or without Neural
  Forge (2026-10-05: about half of the launches; not window focus, VRAM, Reflex or Neural Forge, see
  HARDWARE_VALIDATION.md), so a run is never repeated for it. `bench-report.py` gives each run's
  multiplier, and `--fg` names a run where it did not engage and leaves it out of `--mean`. Whether
  frame generation works and how the game feels is the maintainer's check, playing.

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
picture in every run. The layer engages on it like on a game (`NEURAL_FORGE_ENABLE=1`); its frames
are answered by the layer's own model server.

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
scripts/gta-bench.sh --host <host> probe-ngx-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PROBE_NGX=1 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

```bash
ssh <host> "grep -F '[probe-ngx]' ~/nf-spike/gta/probe-ngx-1/launch.log | head -200"
```

What to look for, and what GTA showed, is in [PRE_UPSCALER_PROBE.md](PRE_UPSCALER_PROBE.md). Run it
once on any new game before claiming the model before the upscaler works there.

### The modes

`NEURAL_FORGE_PREUPSCALE` selects what happens at the DLSS submit:

| Value | What it does | Use |
|---|---|---|
| unset or `model` | Encode, the network, decode, every frame (the default). | Normal play. |
| `off` | Nothing; the 1.x path everywhere, byte-identical to 1.1.0. | A/B comparisons, rollback. |
| `dump` | Once (and again on each `shmctl capture`): write DLSS's colour, depth, motion vectors and the 1x1 exposure images to `~/.local/share/neural-forge/captures/preupscale-<ms>/` (`colour.rgba16f`, `depth.r32f`, `mvec.rg16f`, `exposure.json`, `meta.json`, `colour-preview.png`). The frame goes on untouched. | Getting real DLSS input for offline experiments. |
| `identity` | Capture and write the same bytes back, raw; no network. | The hold's own cost. Set `--set enabled=0` so the 1.x path does not run too. |
| `roundtrip` | The HDR encode and decode with the encoded frame as the answer; no network. | That the transform alone leaves the picture unchanged. |

```bash
scripts/gta-bench.sh --host <host> pu-identity-1 VK_LAYER_neuralforge_neural NEURAL_FORGE_ENABLE=1 NEURAL_FORGE_PREUPSCALE=identity 'WINEDLLOVERRIDES=xinput1_4=b;dinput8=b'
```

`NEURAL_FORGE_PREUPSCALE_PAPER_WHITE=2.5` (or 4) tries another paper white in `model` or
`roundtrip` mode.

### Offline tools

`scripts/hdr_encode.py` (numpy, Pillow) encodes a `dump` for the model and judges an answer
(`info`, `encode`, `judge`, `montage`); it is how E1b chose the encode. `crates/native/tests/rig.rs`
runs the network on a dumped frame on the real device (`NEURAL_FORGE_NATIVE_RIG=<dump dir>`,
`NEURAL_FORGE_NATIVE_MODEL` for another model directory) and compares the head with a reference.

### Fault injection

Set in the game's launch environment (for `gta-bench.sh`, as extra arguments):

- `NEURAL_FORGE_FAIL_CREATE=N[@K]`: let K network builds through, then fail the next N. Exercises
  the retry schedule and the network's re-initialisation; frames go to DLSS untouched meanwhile.
- `NEURAL_FORGE_NATIVE_FAIL=chain`: report one counter-chain timeout at the 300th network frame
  (the graph is rebuilt with barriers, the history reset).
- `NEURAL_FORGE_NATIVE_SKIP_GRAPH=1`: record everything but the network's launches.

### The hold inside DLSS's command buffer (2.0.4)

Where DLSS's buffer renders its own input (Crimson Desert, Cyberpunk 2077) the hold runs inside the
buffer, with its work on a queue the layer adds in a compute-only family (PRE_UPSCALER_DESIGN.md,
"The hold inside DLSS's command buffer"). Its lines: `hold inside DLSS's command buffer available:
side queue in family N, DLSS's in M` per device, then `holding inside DLSS's command buffer ...` once;
the 300-hold summaries are the ordinary ones. `NEURAL_FORGE_INLINE=off` turns it off;
`NEURAL_FORGE_INLINE=release` records it but releases every hold at once without running it (to tell
a fault in the recorded commands from one in the side queue's work).

`crates/layer/examples/queue_wait_probe.rs` measures, natively, whether each candidate queue finishes
work while queue (0, 0) waits on a host-set event. The two-queue GPU test
(`a_hold_inside_the_buffer_runs_on_the_side_queue_and_writes_back_through_the_staging_image`) skips on
a device with one queue (lavapipe); on the test machine run the release test binary with
`NEURAL_FORGE_TEST_IMPORT=0` (with host-memory imports in that process, later device allocations
fail there).

## 6. Reading the logs

**Where they are.** Transitions and errors only (never per-frame lines) also go to the state log, `$XDG_STATE_HOME/neural-forge/layer.log`, which `neural-forge-cli doctor` reads. The full log goes to `NEURAL_FORGE_LOG` if set, otherwise to the game's stderr
(the Proton log; in `gta-bench.sh` runs, `launch.log`). `NEURAL_FORGE_LOG_TIME=1` puts a Unix
timestamp on every line (to match them with `journalctl -k`).
`scripts/check-stalls.sh [LOG]` checks a layer log for fence-wait timeouts and breadcrumb dumps
and summarises engage/disengage transitions and the frame rate.

### Layer lines

| Line | When | What it says |
|---|---|---|
| `[present] N fps (M/s composited by the effect) over Ts` | every 5 s, per presenting process | The real presented frame rate, effect on or off. |
| `[sync] WxH: total= capture_gpu= copy_out= meter= wait_answer= server= rest(readback+compose)= zc= gpu_capture= gpu_compose=` | every 300 composed frames (after the upscaler) | Per model frame: total time on the present thread; capture submit to observed completion (includes the game's own frame); CPU copy out (0 with zero copy); the white meter; the wait for the answer; the model server's own time; readback and compose; whether zero copy was used; the layer's own GPU time from timestamps. |
| `[preupscale] mode <mode> (default) in pid N` | once | The mode for this process. |
| `[preupscale] device ...: vkGetImageViewHandleNVX present, ...` or `no VK_NVX_image_view_handle, ...` | per device | Whether the device can hold at all. |
| `[preupscale] colour input: image 0x... (WxH R16G16B16A16_SFLOAT ...), depth ..., motion vectors ..., exposure input 0x...` | on (re)identification | What the layer recognised as DLSS's inputs. |
| `[preupscale] resources for WxH (padded PWxPH) built: zero-copy (SHM regions imported)` | per extent | The hold's resources. |
| `[preupscale] mode=model extent=WxH ... holds=N hold_ms median= capture_gpu_ms median= writeback_gpu_ms median= network_gpu_ms median= misses= (total ) holds_per_s=` | every 300 holds | The hold's cost and rate; `network_gpu_ms` is the network frame's GPU time. `holds_per_s` equals real fps when every frame is held. |
| `[preupscale] inside DLSS's buffer, last 5.0 s: held N, ...` | every 5 s while holding inside DLSS's buffer | What the hold did with its jobs (held, skipped and why). |
| `[preupscale] phases ms (median): prep= capture_wait= writeback=; capture_wait max= (session max , bound  ms)` | after each summary | Where the hold's CPU time goes. `capture_wait max` is the longest capture fence wait of the window and of the session, next to the bound it is waited with (`preupscale::FRAME_CAPTURE_WAIT`): the numbers to collect across loading, resolution changes, alt-tab and shutdown before that bound is changed. |
| `[preupscale] frame went to DLSS untouched: <why>` | at most once per 5 s | A miss: exposure not usable, the network not ready, a failure. |
| `[native] device created with the network's extensions and features, ...` or `the device cannot run the network: <why>` | per device | Whether the device can run the network. |
| `[native] model loaded and verified from <dir> in N ms`, `network built for WxH (field ..., chained ...) in N ms` | on the first hold, per extent | Loading and building the network. |
| `[native] <why>; frames go to DLSS untouched, trying again in N ms`, `the network is up again after N failed attempt(s)` | on a failed build | The retry schedule. |
| `[native] resetting the history (...)` | on a reset | A gap, another extent, or untouched frames. |
| `[native] answering the after-the-upscaler path's requests on queue N of family M` | once per device | The model server for the after-the-upscaler path is up. |
| `[preupscale] launch-bearing submits: N held reading the colour input, N held undecided, N forwarded untouched` | every 3000 forwarded submits | Frame generation's submits being left alone. |
| `[hotkey] watching N keyboard(s) through evdev` / `watching XInput2 raw keys on :0` / `no way to read the keyboard here ...` | first poll | Which keyboard backend the toggle key uses. |
| `[layer] fence wait timed out after 5s at <site> ...` | on a timeout | A driver stall without device loss; a breadcrumb dump follows. |

### `neural-forge-cli shmctl status`

The `# live status` block:

| Field | Meaning |
|---|---|
| `server_state` | The after-the-upscaler model server: 3 model failed, 4 running, 5 stopped (named in the output). |
| `model_up` | 1 when the network is built; 0 while it cannot be. |
| `server_frames` | Requests the model server has answered. |
| `server_upload_ms`, `server_eval_ms`, `server_readback_ms` | Its stage times for the last request. |
| `server_busy_ms` | Its whole time for its last slot-0 request. |
| `layer_reason` | The native network's status line ("native network running", or why not). |
| `native_running` | 1 while the layer runs the network before the upscaler. |
| `layer_frames`, `layer_ms` | Frames the layer captured, and its last per-frame time. |
| `layer_capture_gpu_ms`, `layer_compose_gpu_ms` | The layer's own GPU time from timestamps (after the upscaler). |
| `preupscale_state` | 0 off, 1 waiting for DLSS input, 2 holding. |
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

Numbers cannot say whether the picture looks right; the maintainer's eyes decide. The 2.0 check (about 15
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

## 10. Reverse-engineering toolkit

Installed on the test machine 2026-10-06. The decompilers below are for this project's own binaries
and open-source ones, not for NVIDIA's DLLs or kernels. The wider survey is [RE_TOOLKIT.md](RE_TOOLKIT.md).

| Tool | Where | Registered as | Health check |
|---|---|---|---|
| REA 4.1.0 (MCP + CLI) | `~/.local/bin/rea`; MCP runs `npx -y rea-agents@4.1.0 mcp` | `rea` (Claude Code, user scope) | `rea doctor --format json` |
| Ghidra 12.1.4 | `/opt/ghidra`, JDK 21 at `/usr/lib/jvm/java-21-openjdk-amd64` | (used by both MCP servers) | `rea doctor --provider ghidra --format json` |
| ghidra-headless-mcp 0.1.0 | `/opt/ghidra-headless-mcp`, venv in `.venv` | `ghidra` (Claude Code, user scope) | `/opt/ghidra-headless-mcp/.venv/bin/ghidra_cli --fake-backend call health.ping` |
| Hopper 6 demo (REA's optional provider) | `/opt/hopper/bin/Hopper`, run by REA on a private Xvfb display | (through REA) | `rea doctor --provider hopper --format json` |
| cuobjdump, nvdisasm 13.4.92 | `/opt/cuda-binutils-13.4`, linked from `/usr/local/bin` | none | `cuobjdump --version` |

Notes:

- **CUDA binutils come from NVIDIA's redistributable tarballs**
  (`developer.download.nvidia.com/compute/cuda/redist/`, SHA-256 checked against
  `redistrib_13.4.2.json`), not from Ubuntu's `nvidia-cuda-toolkit`. That package is CUDA 12.4,
  too old for Blackwell's `sm_120`, and it installs `libnvidia-compute-595`/`-590` and
  `nvidia-kernel-common-595`, which would sit on top of the runfile driver installed separately. Do
  not install it.
- **REA needs a provider named** when more than one supports a target: set
  `REA_ANALYSIS_PROVIDER=ghidra` (the `--provider` flag exists on `doctor` but not on
  `inspect-artifact`). REA's Ghidra path rejects Windows DLLs; the Hopper demo cannot save and
  stops after 30 minutes.
- **Ghidra's first analysis of a binary is slow**: `program.open` on the 2 MB Rust coreutils
  `/bin/ls` took about 15 minutes of auto-analysis. Pass `--update-analysis false` to open without
  it when only symbols, strings or bytes are needed. For multi-step work start the persistent server
  (`ghidra_cli --ghidra-install-dir /opt/ghidra server start`, then `server stop`) so the JVM starts
  once.
- **Neither Ghidra MCP server needs a GUI.** `rea` and `ghidra` appear in `claude mcp list` from any
  directory.
