# DLSS Neural Rendering parameters: what the feature reads, what the helper sets

2026-10-06. Which NGX parameters Feature 18 (`nvngx_dlssnr.dll`, the 310.8 copy imported into the
app's binaries directory) actually reads; which of those the helper never sets; what the helper
sets that nothing reads; and when the history reset is set. Compiled from public sources and from
watching the feature's reads through the helper's own parameter object. **Nothing here comes from
disassembling or decompiling an NVIDIA DLL** (the DLSS and NGX SDK licence prohibits it).

## Bottom line

- **NVIDIA has published no parameters for this feature.** The public NGX SDK lists value 18 as
  `NVSDK_NGX_Feature_Reserved18`; Streamline 2.14.1 declares `kFeatureDLSS_NR = 1004` and its
  uplift buffer types, but ships no `sl_dlss_nr.h`, no plugin source and no guide.
- **The feature reads 41 keys** (plus the helper's own self-test key), the same set for an 8-bit
  and an RGBA16F frame. The helper sets 36 of them.
- **The 5 it reads and the helper never sets are optional resources:** `DLSSNR.ControlMask`,
  `DLSSNR.UI`, `DLSSNR.UIAlpha`, `DLSSNR.Backbuffer`, `DLSSNR.BidirectionalDistortionField`. Left
  unset, the feature runs without them. Every open-source consumer leaves them null too.
- **A history reset exists, and it works:** `DLSSNR.Reset`, read as an integer on every evaluate.
  The helper sets it at every point where the history stops belonging to the frame (table below).
  Its one gap is scene cuts while motion vectors are off, which is not the shipped configuration.
- **18 of the helper's writes are never read.** Among them is `Feature_Flags`, whose bits the
  helper had wrong against the public SDK. A black-box A/B with the bits corrected gave
  byte-identical answers, because nothing reads the key. `Sharpness` is not read either, so the
  per-pass sharpness setting has no effect on this DLL.
- **Nothing found qualifies for a release.** No change found here raises fps or moves the picture
  (see "Candidates considered").

## Sources

| Source | What it gave | Read at |
|---|---|---|
| [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) v310.9.1 | Generic NGX keys and their types, `NVSDK_NGX_DLSS_Feature_Flags` bits (`include/nvsdk_ngx_defs.h:288-300`), `NVSDK_NGX_PerfQuality_Value` (`:250-259`), the "Scene Transitions" reset guidance (Programming Guide 310.6.0, section 3.13). No Feature 18 parameters. | `374959484e79` |
| [NVIDIA-RTX/Streamline](https://github.com/NVIDIA-RTX/Streamline) v2.14.1 | `kFeatureDLSS_NR = 1004` (`include/sl_core_types.h:253`), `kBufferTypeUpliftInputColor/OutputColor/ControlMask` (`:209-215`), `sl::Constants::reset` (`include/sl_consts.h:234-235`). No NR header. | `2122257e0fce` |
| [bmitch87/DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer), AGPL-3.0 | Its helper's create and evaluate writes (`core/ngx_snippet.cpp`), reset rules (`helper/main.cpp`). | `ab722b091071` |
| [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR), GPL-3.0, and its multi-pass forks (hhkbble, y4my4my4m) | D3D12 and Vulkan forwarder writes, game-driven reset. | `973761621353` |
| RenoDX's DLSS 5 add-on, MIT, via [PEQHUB/RenoDX-DLSS5-Generic](https://github.com/PEQHUB/RenoDX-DLSS5-Generic) | Evaluate-time steering, reset epochs. | `1b7d6787817b` |
| [maanHimself/OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR), MIT, `docs/` only | Reset and history semantics. | `9d08f4184bbc` |
| [RedDukeDev/dlss5-nr-amd-custom](https://github.com/RedDukeDev/dlss5-nr-amd-custom), MIT: its NGX host's parameter writes and `docs/DESIGN.md` only | Writes, reset rules. | `058e8a998cfb` |
| **This repository's helper**, the `[params] read` log (`crates/helper/src/selfparam.rs::note_read`) | Every key the DLLs read from the parameter object, its type, and whether it was set. | this commit |

Only source code and documents of the projects above were read, and no code was taken from any of
them. Their own claims about what the DLL contains (string lists) are not used here.

### Why the read log is possible

On this machine `AllocateParameters` fails (`0xbad00002`), so the helper hands NGX its own
`NVSDK_NGX_Parameter` object (`selfparam.rs`, since 2026-09-11). Every key the feature reads
therefore goes through the helper's own `Get*` functions. Since 2026-10-06 they log each key once
per helper run: `[params] read <key> as <type>: set|unset`. This observes the API boundary from the
helper's side; the DLL is not inspected.

How it was measured: the helper, started by the CLI with no game running, was driven by
`trigger_helper_roundtrip` (RUNNING_AND_MEASURING.md, "Driving the helper without a game") with a
1280x720 frame from `screenshots/gta-v-street.jpg`, 48 requests, once as RGBA8 and once as RGBA16F
(created with `DLSSNR.Hdr=1`). Defaults otherwise: one pass, motion vectors on, auto mask on,
preset 0, style 0, strengths 1, skin -1. A key the feature reads only in a state not exercised here
(more passes, a UI image bound) would not appear.

## What the feature reads

"When" is when the helper writes it: C before `CreateFeature`, E before every `EvaluateFeature`.
The consumers' column gives where they agree or differ.

| Key | Read as | Helper sets | When | Consumers |
|---|---|---|---|---|
| `DLSSNR.Width`, `DLSSNR.Height` | u32 | frame size | C | all set it |
| `DLSSNR.ScalingRatio` | f32 | 1.0 | C | 1.0 everywhere |
| `DLSSNR.Hint.Render.Preset` | i32 | 0 (setting `preset`) | C | 0 by default ("the model chooses") |
| `DLSSNR.Style` | u32 | 0 (setting `style`) | C | 0 Default, 1 Natural, 2 Cinematic |
| `DLSSNR.Intensity` | f32 | 1.0, clamped 0-4 | C | default 1 |
| `DLSSNR.LocalToneStrength` | f32 | 1.0, clamped 0-4 | C | default 1 |
| `DLSSNR.LocalStructureStrength` | f32 | 1.0, clamped 0-4 | C | default 1 |
| `DLSSNR.SkinStructureStrength` | f32 | -1, clamped -1..4 | C | -1 = follow local structure |
| `DLSSNR.UseAutoMask` | i32 | 1 (setting `auto_mask`) | C | DLSS5VKLayer and OptiScaler on, RenoDX off |
| `CreationNodeMask`, `VisibilityNodeMask` | u32 | 1 | C | 1 |
| `DLSSNR.Color`, `DLSSNR.Output`, `DLSSNR.MVec`, `DLSSNR.Depth` | pointer | the frame's images | E | all |
| `DLSSNR.{Color,Output,MVec,Depth}Subrect{BaseX,BaseY,Width,Height}` | i32 | 0, 0, width, height | E | the same; OptiScaler and RenoDX use the game's offsets for MVec and Depth |
| `DLSSNR.MVecScaleX`, `DLSSNR.MVecScaleY` | f32 | the motion scale for the flow's units | E | each has its own units |
| `DLSSNR.DepthInverted` | i32 | 0 (depth cleared to 1.0, far) | E | DLSS5VKLayer 1; the game-hosted ones follow the game |
| `DLSSNR.Reset` | i32 | see "History reset" | C (1) and E | all |
| `DLSSNR.ControlMask` | pointer | **never** | | null everywhere |
| `DLSSNR.UI`, `DLSSNR.UIAlpha` | pointer | **never** | | null everywhere |
| `DLSSNR.Backbuffer` | pointer | **never** | | null, except the AMD project, which binds its output |
| `DLSSNR.BidirectionalDistortionField` | pointer | **never** | | null (DLSS5VKLayer only, explicitly) |

The style and strength parameters are written at creation only, and a change rebuilds the feature
(`ngx.rs::set_create_tuning`, `maintain_feature`). DLSS5VKLayer, OptiScaler and the AMD project do
the same. RenoDX writes them at evaluate and only resets history on a change. The read log cannot
tell the two apart: the keys are read either way.

### The five the helper never sets

All five are optional inputs. With no value the feature runs without them, as it has in every
helper session so far. Two are worth knowing about:

- **`DLSSNR.UI` and `DLSSNR.UIAlpha`**: a separate UI layer so the model leaves the HUD alone. On
  the pre-upscaler path the HUD is drawn after DLSS, so the frame the model sees has none, and
  there is nothing to bind. On the post-upscaler path the HUD is in the frame, but this project has
  no separate UI image to give.
- **`DLSSNR.ControlMask`**: Streamline documents its equivalent as an optional "4-channel control
  mask consumed by uplift passes". What the channels mean is not public. Binding one would be a new
  feature with no specification behind it.

## What the helper writes that the feature never reads

Present in `ngx.rs::create_feature_at` or `frame.rs`, absent from the read log on both paths:

| Key | Helper writes | Note |
|---|---|---|
| `Feature_Flags` | `DoSharpening \| AutoExposure` | Not the SDK key either (that is `DLSS.Feature.Create.Flags`). The helper's bit constants were also wrong against `nvsdk_ngx_defs.h`: it wrote `0x4 \| 0x8`, which the public enum reads as `MVJittered \| DepthInverted`. Corrected (`abi.rs`). An A/B of 4 alternating runs (control, corrected, control, corrected) gave byte-identical answers, because nothing reads the key. |
| `PerfQualityValue`, `NVSDK_NGX_Parameter_PerfQualityValue` | 3 | The code calls 3 "Balanced"; in the public enum 3 is `UltraPerformance` (Balanced is 1). Moot here: the feature reads neither key. |
| `Sharpness` | per pass, 0-1 | Not read, so **the per-pass sharpness setting does nothing on this DLL**. The public SDK marks sharpening deprecated ("Sharpness is not supported"). |
| `DLSSNR.Hdr`, `DLSSNR.SDR`, `DLSSNR.AutoExposure` | by proxy class | Not read on either path. The feature key still carries `hdr`, so a class change rebuilds the feature, which also resets history. That is still worth doing. |
| `DLSSNR.Enabled` | 1 | |
| `DLSSNR.InputWidth/Height`, `DLSSNR.OutputWidth/Height` | frame size | |
| `DLSSNR.Upscaling`, `DLSSNR.Scale` | 0, 1.0 | |
| `Width`, `Height` | frame size | |
| `NVSDK_NGX_Parameter_ExposureScale`, `NVSDK_NGX_Parameter_PreExposure` | 1.0 | Macro names, not the strings they expand to (`DLSS.Exposure.Scale`, `DLSS.Pre.Exposure`). Neither form is read. |

Each write costs well under a microsecond, so removing them would change neither fps nor the
picture. They are left in place; this table is the record.

## What consumers set that this feature does not read

These are disagreements between the consumers that this DLL does not settle in anyone's favour,
because it reads none of them:

- `DLSSNR.UICorrection`: OptiScaler sets 1 ("the model's own default"), the AMD project warns that 1
  without UI textures returns black, and DLSS5VKLayer and RenoDX set 0.
- `DLSSNR.Output.Width/Height`, `DLSSNRComputeScalingRatioCallback`, `DLSSNR.GlobalToneStrength`,
  bare `Reset`, the jitter keys, `MotionVectors`, `OutWidth`/`OutHeight`,
  `DLSS.Indicator.Invert.X/Y.Axis`, and the real `DLSS.Feature.Create.Flags`, `DLSS.Pre.Exposure`
  and `DLSS.Exposure.Scale` strings.

The scaling-ratio callback may matter for an upscaling (`ScalingRatio` below 1) configuration,
which RenoDX reports is rejected without it. The helper always runs at 1:1.

## History reset

`DLSSNR.Reset` exists, is read on every evaluate (as an integer), and is the feature's only reset
key. Bare `Reset`, which DLSS5VKLayer mirrors, is not read. The public NGX guide's rule for the
generic key: non-zero "for the first frame after a major transition", and release and recreate the
feature, not reset, when the resolution or formats change.

| Event | Helper | DLSS5VKLayer | OptiScaler | RenoDX | AMD project |
|---|---|---|---|---|---|
| Feature created or rebuilt (size, format, tuning, pass count) | reset (`create_feature_at` sets 1; the pass's `needs_reset`) | reset | reset | reset | reset |
| First evaluate of a new resource set (size or format change) | reset (`frame.rs`, `first`) | reset | reset | reset | reset |
| Requests went unevaluated (effect off, model not ready, failed open) | reset (`history::Stale::Skipped`) | no (passes through) | no | reset (toggle advances the epoch) | n/a |
| Pause over 500 ms | reset (`Stale::Idle`) | not handled | not handled | not handled | n/a |
| Proxy format changed | reset (`Stale::FormatChanged`) | reset (via rebuild) | reset | reset | reset |
| Scene cut detected from the picture | reset, **only while motion vectors are on** (`scene.rs`, threshold 40 on an 8-pixel-step luma thumbnail) | no model reset; drops the flow history and zeroes motion (threshold 55, two frames) | no detector | no detector | n/a |
| The game's own NGX reset flag | not visible | not visible | reset | reset | reset |

Findings:

- **Resolution changes are covered twice**: the feature is rebuilt (the key includes the size), and
  the new resource set's first evaluate resets.
- **Cuts with motion vectors off are not detected.** `MotionState::prepare` returns "no cut" as soon
  as motion is off, so the thumbnail is never taken. Motion vectors are on by default (quality FAST
  since 0.1.98) and in the GTA baseline, so this only affects a configuration nobody runs. Taking
  the thumbnail regardless would close it at the cost of a CPU pass over every 64th pixel.
- **The game's own reset is out of reach.** OptiScaler, RenoDX and the AMD project sit inside the
  game's NGX calls and forward its `Reset`. Neural Forge sees DLSS from a Vulkan layer, where the
  game's NGX parameter block is not visible; the picture-based detector is its substitute.
- **The helper resets more than DLSS5VKLayer does** (on cuts, gaps and pauses). Those rules came
  from OpenDLSS-NR's documented "history counts only for the frame that directly follows it"
  (OPENDLSS_REVIEW.md) and are kept.

## The layer side (`preupscale.rs`)

No public source describes DLSS Super Resolution's own kernels or their launch parameters: the
public SDK stops at the NGX API, and none of the consumers above touches the upscaler's internals.
The layer's identification therefore stays as it is: the input kernel's launch parameters naming
colour with depth and motion (`input_launch`, `by_params`), the size rule as the fallback
(`identify`, `size_candidates`), and the 1x1 exposure rule (`exposure_images`,
`registered_exposure_input`). It has held for nine games. This is a loss of convenience, not a
blocker: a new engine still needs `NEURAL_FORGE_PROBE_NGX=1` once (RUNNING_AND_MEASURING.md,
section 5).

## Candidates considered for a release

Alex's bar: a change ships only if it raises fps or improves image quality.

| Candidate | Small | Risk-free | Measurable with `gta-bench.sh` | Raises fps or improves the picture | Shipped |
|---|---|---|---|---|---|
| Correct `Feature_Flags` bits | yes | yes | no difference to measure | no: byte-identical answers | committed, unreleased |
| Remove the 18 unread writes | yes | yes | no | no | no |
| Bind `UI`, `UIAlpha` or `ControlMask` | no (no source image; mask meaning not public) | no | | unknown | no |
| Scene-cut reset with motion vectors off | yes | yes | no (the bench runs with motion on) | only in a configuration nobody runs | no |
| Hide or remove the sharpness setting | yes | yes | no | no (it already does nothing) | no; Alex's call, since it is a GUI change |

No 2.0.9.
