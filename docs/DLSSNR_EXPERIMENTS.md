# Feature 18 improvement experiments

2026-10-07. Baseline reviewed: `5f8b7eb` (2.0.8 plus the parameter-read audit).
This work uses interface strings and API observations only. No NVIDIA code or weights were
decompiled, extracted or added to this repository.

## Preset comparison

The uploaded 310.8 DLL's interface diagnostics explicitly describe unavailable presets falling
back to the shipping default. A successful creation therefore cannot establish that two preset
numbers select different networks, so presets 0-15 were swept through the helper on a real
encoded pre-upscaler dump, with no game running. The sweep script (`scripts/preset-sweep.py`) was
removed together with the Preset setting once the result below showed it does nothing; it is in
the git history.

## Result: presets 0-15 on 310.8 (2026-10-06)

Run on the test machine (RTX 5070, driver 615.71.09) at 23:00, no game running, pass 0 overrides
clear, with the installed helper and `nvngx_dlssnr.dll` from the CLI's `binaries` directory
(SHA-256 `e16bcf15e16e13f5...`). Output: `~/nf-spike/presets/sweep-2` (`metadata.json`,
`results.json`, `output-groups.json`, one CSV, log and raw answer per trial).

**The frame.** A GTA V Enhanced pre-upscaler dump: a city street with people, signs, palms and
buildings, 1485x836, DLSS's exposure 0.180 (`~/nf-spike/presets/dump-street`, kept with its depth
and motion vectors). Encoded as the layer does (`hdr_encode.py encode --convention mul --white 3`,
`opendlss`), SHA-256 `aa05d7635f6f4324...`. GTA only dumps with its frame generation off: with it on,
GTA copies depth and motion vectors inside DLSS's buffer behind same-layout barriers, which stops a
dump (but not a model hold). `FrameGenType` was set to 0 for the dump run and the original
`settings.xml` put back, hash checked.

**Procedure.** Presets 0-15, three trials each in rotated order, the helper restarted before
every trial, 160 requests per trial (32 warmup evaluations, then 64 measured).

| | All 16 presets |
|---|---|
| Feature built with the requested preset | yes, 48 of 48 (`[params] create tuning: preset=N`, `VULKAN_CreateFeature(18) -> 0x1`; the feature reads `DLSSNR.Hint.Render.Preset`) |
| Requests evaluated (`seq_eval`) | 160 of 160 in every trial, no echoes |
| Answer differs from the input | every request (mean absolute change 0.034 in encoded units) |
| Final answer | **one SHA-256 for all 48 trials** (`faf3a490f91c...`): every preset gives preset 0's answer, bit for bit, across fresh sessions |
| Helper busy time, median per trial | 5.77-5.90 ms for every preset (trial-to-trial spread about 0.1 ms, no preset outside it) |
| Helper evaluate (CPU recording) time, median | 0.13-0.15 ms |
| Request round trip, median | 10.1 ms (includes the tool's 5 ms polling) |
| VRAM estimate | not populated by the helper |

**Reading.** On this DLL and this frame, the preset number changes neither the picture nor the
cost: 1-15 behave exactly like 0, which fits every value falling back to the one shipping network.
It is still a still frame: a preset that differed only in how it treats motion or history would not
show here. Since the cost is identical there is no fps to gain, so no game benchmark was run and
default preset 0 stays. Run the sweep again when a new `nvngx_dlssnr.dll` arrives; a second output
hash is the signal to do the moving-scene and game-benchmark comparison.

**Failures on the way.**
- The first run (`sweep-1`) used the CLI from `target/release`, which found no helper beside
  itself. Its `restart` stopped the installed helper and could not start it, so the first trial
  failed. Restoration put the settings back but could not restart the helper. The script now
  refuses up front (`helper_exe=missing`), and the procedure sets `NEURAL_FORGE_INSTALL_DIR`.
- Two GTA dump attempts with frame generation on wrote no dump (above), and one ran with the
  helper stopped, where nothing is held at all.

## Real depth: next integration experiment

`preupscale::Inputs` already identifies depth and motion images. The inline staging path explicitly
carries colour and exposure only; `frame.rs` binds constant-far `R32_SFLOAT` depth and helper optical
flow. This is the concrete missing input, not an absent NGX parameter.

A complete depth implementation needs capture at the same hold point as colour, depth-aspect
layout/ownership handling (including sampled/attachment depth and R32 colour copies), a protocol
version with per-slot depth metadata/storage, helper upload, validated inversion and subrects,
and history reset when source semantics change. Preserve synthetic-depth fallback for unidentified
or incompatible inputs. Do not infer inversion solely from format or kernel naming.

Start with a dump-only probe and establish near/far conventions on the rig before enabling it.
Then compare the same moving scene with synthetic versus real depth while keeping optical flow
and all model settings fixed. Capture/upload adds cost; depth is a visual-quality hypothesis,
not a demonstrated speed improvement.

Game motion vectors come after depth, with separately calibrated sign, scale, jitter and image
extent. The recorded optical-flow cost is roughly 0.45 ms on the measured pre-upscaler path;
transporting game vectors must cost less to establish a performance benefit.

## Validation limits

Local validation passed the six Python regression tests, the namespace check and 52 Rust tests
across the CLI, protocol and protocol examples (`cargo +stable test -p neural-forge-protocol
-p neural-forge-cli --all-targets`). The Python tests exercise echo/warmup classification,
exclusive channel ownership, preset-override rejection, restoration on failure/interruption, a
refused preset not ending the sweep, and the channel, DLL and helper checks. The GPU/API run is
above. No depth transport, dynamic-resolution
feature reuse, default preset change or release is enabled by this work.
