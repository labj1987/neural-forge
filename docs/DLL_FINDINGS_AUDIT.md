# Audit: Neural Forge against the nvngx_dlssnr.dll findings

Date: 2026-10-07. A read-only code audit: no code changed, nothing run apart from one `sha256sum`.

Refs read:
- `main` = `origin/main` at `328d084` (2.0.10).
- `native-backend` = `origin/native-backend` at `0d212f0` (the OpenDLSS-NR backend).

Citations are `ref:path:line` at those commits. Uncommitted work on LordNikon's `native-backend` working copy was
not audited.

The reference facts come from a static analysis (REA + Ghidra) of one DLL build. They are taken as given; this
audit only checks the code against them.

## 1. DLL identity check

The helper loads `<bin_dir>\nvngx_dlssnr.dll` (`main:crates/helper/src/ngx.rs:276-284`). `bin_dir` is
`NEURAL_FORGE_BIN_DIR`, which the supervisor sets from config.ini's `binaries=` line, or from
`<data_dir>/binaries` when that line is absent (`main:crates/supervisor/src/lib.rs:262,270`;
`main:crates/supervisor/src/paths.rs:57-58`).

On LordNikon, config.ini has `binaries=/home/alex/.local/share/neural-forge/binaries`. The file there:

| | Value |
|---|---|
| Path | `/home/alex/.local/share/neural-forge/binaries/nvngx_dlssnr.dll` |
| Size | 165,840,496 bytes |
| SHA-256 | `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e` |

**It matches the analysed build (310.8.0.0).** The conclusions below apply to the deployed DLL.

## 2. Audit items

### 2.1 Motion vectors and depth

**What NGX gets.** Both paths go through the helper's `FrameResources::evaluate`, so both get the same thing:

- **MVec: the helper's own optical flow, never the game's.**
  - The upload clears MVec to 0 (`main:crates/helper/src/frame.rs:1038-1045`).
  - The NV optical flow then overwrites it, computed on the uploaded colour (`frame.rs:668-690`).
  - `flow_to_mvec.comp` converts the flow to full-resolution RG16F (`main:crates/helper/shaders/flow_to_mvec.comp:23-32`).
  - Flow runs only for slot 0, and only with `mvec_enabled` on (`main:crates/helper/src/main.rs:495,545`). Otherwise MVec stays 0.
  - On the pre-upscaler path the RGBA16F proxy is quantised to 8 bits for the flow estimate (`main.rs:492-494`).
- **Depth: constant.** An R32_SFLOAT image (`frame.rs:159`), cleared to 1.0 ("far") on every upload (`frame.rs:1046-1056`).
- **Bound resources and subrects.** `DLSSNR.MVec` and `DLSSNR.Depth` are bound at `frame.rs:764-767`. Color, Output, MVec and
  Depth get full-frame subrects (`frame.rs:774-779`).
- **Unbound resources.** ControlMask, UI, UIAlpha, Backbuffer and BidirectionalDistortionField are never set, so
  they parse as null.
- **No motion region in the protocol.** The SHM protocol has carried none since v7 (`main:crates/protocol/src/lib.rs:17-27`).

**Scalars.** All three are set explicitly on every evaluate:

| Parameter | DLL fallback | Sent | Where |
|---|---|---|---|
| `DLSSNR.DepthInverted` | 1 | **0** (matches the constant "far" 1.0 clear) | `main:crates/helper/src/frame.rs:768-770` |
| `DLSSNR.MVecScaleX/Y` | 1.0 / 1.0 | `motion_scale` from `protocol::motion::scales(mvec_scale_mode, w, h)`; the default mode PIXELS gives 1.0 / 1.0 | `frame.rs:757-758`; `main:crates/helper/src/main.rs:413`; `main:crates/protocol/src/motion.rs:3-9`; default at `main:crates/protocol/src/header.rs:526` |

These lines are the same on `native-backend`.

**The game's own motion vectors and depth (pre-upscaler path).** The layer has them, but feeds them to NGX nowhere:

- **Found.** The layer identifies DLSS SR's depth and motion-vector images from the input kernel's launch
  parameters (`Inputs`, `main:crates/layer/src/preupscale.rs:415-423`; rule at `preupscale.rs:451-453,753-754`; size fallback at `509-531`).
  Motion vectors are accepted as RG16F or RG32F (`preupscale.rs:462-463`).
- **Usable for a hold.** The hold's `Target` carries both (`main:crates/layer/src/device.rs:1157-1169`).
  - They count as readable only at the colour input's extent, which is the proxy's extent.
  - Motion vectors count only when RG16F.
- **Copied only in dump mode.**
  - `capture_reads` copies them only when dump buffers exist (`preupscale.rs:3663-3668`, `3890`).
  - Dump buffers exist only in `Mode::Dump` (`gate`, `preupscale.rs:4863-4871`).
- **Model mode sends the proxy alone.** The request carries only the padded RGBA16F proxy (`await_answer`, `preupscale.rs:4093-4094`).
- **The native backend reads them for its own history.** It copies DLSS's motion vectors every frame
  (`native-backend:crates/layer/src/preupscale/native.rs:968-972`) and applies them with the jitter delta
  (`native-backend:crates/layer/shaders/native_preprocess.comp:73-76`). Depth is not used. The NGX/helper path on
  that branch still gets optical flow and constant depth.

**Where they would have to go:**
- **Capture.** Extend `record_capture` to copy them in model mode (`main:crates/layer/src/preupscale.rs:3697`).
  `native-backend` already gives it a motion-vector buffer (`native-backend:crates/layer/src/preupscale.rs:3784`).
- **Protocol.** Add a motion and depth region to the mapping, with format flags. That needs an
  `SHM_VERSION` bump (`main:crates/protocol/src/header.rs:389-402`).
- **Helper upload.** Upload them instead of the clear and the flow (`main:crates/helper/src/frame.rs:1038-1056,668-690`), and
  skip the flow when they are present (`main.rs:495,545`).
- **Scalars.** Set `MVecScale` and `DepthInverted` to match the real data (`frame.rs:757-758,770`).

**Size of the change.** About 5-6 files and 250-400 lines plus tests: preupscale.rs, device.rs, protocol
lib.rs/header.rs, and helper main.rs/frame.rs.

**Details any such change must handle:**
- RG32F vectors are refused today.
- Depth arrives as D32 or D24 (depth aspect) or as R32 copies (`preupscale.rs:366-387`), but the helper's
  depth image is R32F colour.
- The proxy is padded to even sizes (`preupscale.rs:3109-3110`), so the vectors and depth must be padded the same way.
- Game vectors usually exclude jitter, and the DLSSNR request carries no jitter.

**Match against the reference facts.** The code sets every relevant scalar explicitly, so no fallback applies.
`DepthInverted=0` overrides the fallback of 1 on purpose. The real data DLSSNR could use is available but not
forwarded.

**Verdict: change recommended (image quality).** This is conditional on the DLL's temporal path actually
responding to MVec (section 4, test 1). On the post-upscaler path there is no game data to forward, so optical flow
remains the only source there.

### 2.2 Model resolution setting

**Trace.**
- **GUI.** The "Model resolution" row (25-100%) is bound to the `working_scale` header field (`main:crates/gui/src/ui.rs:352-353`).
- **Persistence.** It is saved as `set_working_scale` (`main:crates/gui/src/shm.rs:80-83`).
- **Header.** The field is `working_scale_bits` (`main:crates/protocol/src/header.rs:173`, default 1.0 at `:466`, named-key table
  at `:660,:713`). The layer reads it at `main:crates/layer/src/shm.rs:344`.
- **Layer, post-upscaler capture.**
  - The scale is capped at 1.0 and at 3840x2160 worth of pixels (`main:crates/layer/src/capture.rs:620-628`).
  - The model size is rounded to even, with a 64 px floor (`capture.rs:630-640`), and applied at `capture.rs:1291`.
  - A LINEAR `vkCmdBlitImage` scales the frame into a model-sized image (`capture.rs:1911-1929`).
  - The request carries the reduced size (`capture.rs:888,1534`).
- **Helper.** It builds the feature at the request's size, with `DLSSNR.Upscaling=0`
  (`main:crates/helper/src/main.rs:390-391`; `main:crates/helper/src/ngx.rs:577-583`).
- **Compose.** The compose blits the answer back up to the frame (`main:crates/layer/src/composition/gpu.rs:875-895,943`).

**What the DLL sees.**
- **No size hint.** The setting never becomes ScalingRatio or PerfQualityValue. The DLL only sees a smaller frame.
- **Constant ScalingRatio.** The helper always sends `DLSSNR.Scale=1.0` and `DLSSNR.ScalingRatio=1.0` (`ngx.rs:584-585`). This matches what the
  DLL forces anyway.
- **PerfQualityValue outside the supported set.** The helper always sends `PerfQualityValue=3` and
  `NVSDK_NGX_Parameter_PerfQualityValue=3` (`ngx.rs:589-590`). 3 is outside the set the DLL's scaling callback
  supports (0, 1, 2, 4, 5). The comment there says 310.8 reads neither key. The reference facts say the callback
  returns "unsupported" for 3, so the comment and the facts disagree on whether the value is consulted; see section 4.
- **Same lines on `native-backend`** (`native-backend:crates/helper/src/ngx.rs:583-584`).

**Pre-upscaler path: the setting is a no-op.**
- **Model mode.** This is the default mode when `NEURAL_FORGE_PREUPSCALE` is unset
  (`main:crates/layer/src/preupscale.rs:200`). It sends DLSS's input at its padded size (`preupscale.rs:4026-4028,4093-4094`).
- **No reference.** `working_scale` is not referenced anywhere in the pre-upscaler module.
- **Native pre-upscaler path.** On `native-backend` it builds at DLSS's input size and ignores the setting too
  (`native-backend:crates/layer/src/preupscale/native.rs:915`).
- **Native post-upscaler path.** It honours the setting through the same capture code
  (`native-backend:crates/layer/src/preupscale/native_post.rs:220-221,234`).

**Stale comments and labels.**
- `main:crates/layer/src/shm.rs:53-64` calls `working_scale` "genuinely unused by any caller", which is wrong (`capture.rs:1291`).
- `main:crates/protocol/src/header.rs:169-172,197-198` describes supersampling above 1.0 and a `scaling_downscaler` read above 1.0.
  - The scale is capped at 1.0 (`capture.rs:621`).
  - `scaling_downscaler` is dead code (`shm.rs:66-67`).
  - The filter is a fixed LINEAR (`capture.rs:1929`). So the Downscaler setting (config `set_scaling_downscaler=4`) is a no-op.
- The GUI row's label (`ui.rs:353`) does not say it has no effect before the upscaler.

**Match against the reference facts.** The DLL fact (ScalingRatio ignored) does not affect the setting, because
the setting never relied on ScalingRatio.

**Verdict: change recommended (maintenance only).**
- **Post-upscaler path:** functional on this DLL build, through layer-side resizing.
- **Pre-upscaler path:** a no-op. That is the default, for both NGX and native.
- **Downscaler setting:** a no-op everywhere.

### 2.3 Reset and history

**Where Reset is set (helper, `main`).**
- **At creation:** `DLSSNR.Reset=1` (`main:crates/helper/src/ngx.rs:601`).
- **On every evaluate, per pass:** `first || reset_history || pass.reset` (`main:crates/helper/src/frame.rs:785-786`).
  - **`first`:** the first evaluate on a new `FrameResources` (`frame.rs:713`, `:115`).
  - **`reset_history`:** `scene_cut || stale.is_some()` (`main:crates/helper/src/main.rs:531`). `stale` comes from `HistoryGap::begin`
    (`main.rs:521`; `main:crates/helper/src/history.rs:68-82`), with these causes:
    - `FormatChanged`: 8-bit vs HDR class (`history.rs:58`, `main.rs:410`).
    - `Skipped`: a request was not evaluated (`main.rs:553-554`).
    - `Idle`: more than 500 ms since the last evaluate (`history.rs:48,73-75`).
  - **`pass.reset`:** set once after a pass is built or rebuilt (`main.rs:541`; `ngx.rs:533-534,1006`).
  - **Missing motion:** deliberately does not reset (`main.rs:518-520`).

**Events.**

| Event | Helper path (`main`) | Native path (`native-backend`) |
|---|---|---|
| Swapchain recreation, same size | **resets.** The warmup is dropped (`main:crates/layer/src/device.rs:1707`); no requests are sent for 5 s (`main:crates/layer/src/swapchain.rs:54,70`), so the next evaluate is `Idle`. The feature is kept. With `NEURAL_FORGE_WARMUP_SECS=0` and a gap of 500 ms or less: does not reset (`swapchain.rs:84-86`). | **resets** (`Idle`; `native-backend:crates/layer/src/device.rs:1731`) |
| Resolution change | **recreates the feature.** Every pass is released through `NVSDK_NGX_VULKAN_ReleaseFeature` and created at the new key (`ngx.rs:921-930,760-766,997-1006`); `FrameResources` is rebuilt (`main.rs:502-511`); Reset=1. A handle is never kept with new extents. | **recreates.** `Resources`, the `NativePass` and the network are rebuilt (`native-backend:crates/layer/src/preupscale.rs:4759-4772`; `native.rs:949-961,324-335`); no history on the first frame |
| Effect off then on (enabled, F11, apply_model), gap over 500 ms | **resets** (`Idle`; no requests while off, `main:crates/layer/src/capture.rs:1237-1257`) | **resets** (`Idle`) |
| Same, gap of 500 ms or less | **does not reset** (or `Skipped` if a request arrived while the model was off: resets) | **does not reset** |
| "Model every Nth frame" skips | **does not reset.** No request on carried presents (`capture.rs:1473-1487`); the gap stays under 500 ms, so history is N frames old. Not applied before the upscaler at all. | not applicable (not applied) |
| Warmup gate ending, first engage | **resets** (a new feature: Reset at create, `first`, `pass.reset`) | **resets** (`frames == 0`) |
| Warmup re-engage after a loading screen | **resets** (`Idle`, 5 s or more) | **resets** (`Idle`) |
| Tuning change | **recreates that pass** (`ngx.rs:968,988-1006`), Reset=1 | not traced |
| Frames without motion vectors | does not reset | **resets** on each such frame (`native-backend:crates/layer/src/preupscale/native.rs:988`) |

**Match against the reference facts.** The DLL invalidates history itself on a size change. The helper recreates
the feature instead and also sets Reset=1, which is a superset of what the DLL does. Reset is set explicitly on
every evaluate, so the fallback of 0 never applies.

**Verdict: no change needed.** Recreating the feature on a resolution change costs a rebuild (pass 0 is echoed
until it is done, `ngx.rs:884-886,929`). Keeping the handle would rely on the DLL's own resize branch, which is
untested here; see section 4.

### 2.4 Native backend parity

**What the helper sends (`main`) against the DLL fallbacks.**

| Control | DLL fallback | Helper sends | When |
|---|---|---|---|
| Style | 0 | 0 by default | creation only (`main:crates/helper/src/ngx.rs:183,157`) |
| Intensity | 1.0 | 1.0, clamped 0-4 | creation only (`ngx.rs:184,165`) |
| LocalToneStrength | 1.0 | 1.0, clamped 0-4 | creation only (`ngx.rs:185,166`) |
| LocalStructureStrength | 1.0 | 1.0, clamped 0-4 | creation only (`ngx.rs:186,167`) |
| SkinStructureStrength | -1.0 | -1.0, clamped -1..4 | creation only (`ngx.rs:187,168`) |
| UseAutoMask | **0** | **1** | creation only (`ngx.rs:188,169`) |
| Enabled | 1 | 1 | creation (`ngx.rs:600`) |
| Reset | 0 | 1 at creation, then per event | `ngx.rs:601`; `frame.rs:785-786` |
| DepthInverted | 1 | **0** | every evaluate (`frame.rs:770`) |
| MVecScaleX/Y | 1.0/1.0 | the flow's scale, default 1.0/1.0 | every evaluate (`frame.rs:757-758`) |
| UICorrection | 0 | never set (0 applies) | n/a |
| ControlMask | unbound | never bound | n/a |
| Preset | read at creation | 0 (`native-backend` hard-codes 0) | creation |
| Runtime callback | optional | none registered (the only callback is a null progress callback, `main:crates/helper/src/abi.rs:292`) | n/a |

The style and strength values are written into the shared parameter block (`ngx.rs:565`) only before
`CreateFeature` (`ngx.rs:139-142,602-604,669`); the comment calls them latched at creation (`frame.rs:759-762`). The
reference facts say the DLL re-reads them on every evaluate. With the block keeping the last values written, both
hold for one pass. With more than one pass, every pass's evaluate reads whatever the last-built pass wrote, so
per-pass overrides of the passes built earlier would be lost (section 4, test 3). Alex runs one pass.

**The native path (`native-backend`).**
- **Defaults.** `Conditioning::default` is style 0, intensity 1, tone 1, structure 1, skin -1, auto-mask on
  (`native-backend:crates/layer/src/preupscale/native.rs:413-416`). `From<PassTuning>` uses the helper's clamps (`native.rs:399-410`).
- **Source of the values.** They come from `global_tuning()` (`native-backend:crates/layer/src/preupscale.rs:132`;
  `native_post.rs:231`), which ignores per-pass overrides; the helper uses `resolve_pass(i)` (`main:crates/helper/src/main.rs:441`).
- **When they apply.** Written every frame (`native.rs:789-795`), so a change applies on the next frame; nothing is
  latched at creation.
- **Lanes** (`native-backend:crates/layer/shaders/native_preprocess.comp:135-142`):
  - Lane 10 is style/128 and lane 11 is tone.
  - Auto-mask on: lane 12 = 1, lane 13 = skin (structure when skin < 0), lane 14 = structure.
  - Auto-mask off: lane 12 = structure, lanes 13 and 14 = -1. These are the sentinels.
  - The vendored OpenDLSS-NR uses the same lanes (`native-backend:third_party/opendlss-nr/shaders/preprocess.comp:83-88`;
    the `autoMask ? 1 : -1` push constant in `third_party/opendlss-nr/src/kernels.cpp:520-522` means the same under its `> 0` test).
- **Composite.** Style 1 and 2 apply the `nrStyle` operator scaled by tone, then blend by intensity
  (`native-backend:crates/layer/shaders/native_composite.comp:88-92,171-178`).
- **No equivalents.** There is no Preset, ControlMask, UICorrection, Enabled, DepthInverted, depth input or MVecScale.
  Motion is DLSS's vectors in render pixels plus the jitter delta (`native_preprocess.comp:73-76`).
- **App defaults.** Both paths start from the same values: style 0, intensity 1, tone 1, structure 1, skin -1,
  auto-mask 1 (`main:crates/protocol/src/header.rs:59-71,431-452`; `native-backend:crates/protocol/src/header.rs:53-64,425-446`).

**Mismatches.**

| Control | DLL fallback | Helper | Native | Effect |
|---|---|---|---|---|
| UseAutoMask | 0 | 1 (explicit) | on | The fallback never applies; helper and native agree |
| Skin -1, auto-mask on | falls back to structure | -1 (the DLL resolves it) | lane 13 = structure | Same meaning |
| Auto-mask off | sentinels | user's skin value (the DLL replaces it) | lanes 13-14 = -1, lane 12 = structure | Same, provided the DLL's sentinel is -1 |
| DepthInverted | 1 | 0, with constant depth 1.0 | no depth | Matters only if the DLL's network sees depth beyond the 16 lanes |
| MVecScale | 1.0/1.0 | the flow's scale | none (DLSS vectors + jitter) | A different motion source by design; frames with history only |
| Per-pass values | n/a | `resolve_pass(i)` | `global_tuning()` | Differ only with more than one pass or pass overrides |
| When controls apply | every evaluate (per the facts) | written at creation, rebuild on change | every frame | Same values once applied; the helper rebuilds first |

**Match against the reference facts.** With the app defaults, no value the helper sends falls through to a DLL
fallback that native interprets differently. UseAutoMask is the one control whose fallback differs from the app
default, and the helper always sets it.

**What has been measured.** Native against NGX on GTA V frame d1, first frame after a reset, so no history
(`native-backend:docs/NATIVE_BACKEND.md:20,212-226,448-460`):
- Defaults: bit-exact on six frames.
- Tone 0.5, structure 0.5 with skin 2, and auto-mask off: bit-exact.
- Style 1 and 2, and intensity 0.5: within one half step.

**Not covered by those measurements:**
- skin -1 with structure other than 1
- auto-mask off with non-default skin or structure
- style with tone other than 1
- other intensity values
- any frame with history
- more than one pass

**Doc error.** `native-backend:docs/NATIVE_BACKEND.md:470` says the native hold reads pass 0's settings through
`ShmClient::pass_tuning`. The code uses `global_tuning()` (`native-backend:crates/layer/src/preupscale.rs:132`).

**Conclusion.** For identical inputs and the app defaults, the defaults alone do not make native differ from the
helper. The difference that remains is temporal: DLSS vectors plus jitter on native, against optical flow, constant
depth and `DepthInverted=0` on the helper. That only affects frames with history.

**Verdict: change recommended (maintenance only).** Fix the doc line. No change to native defaults.

## 3. Recommended changes, by expected benefit

1. **Feed the game's motion vectors (and depth) to NGX on the pre-upscaler path. Image quality.**
   - Only if section 4, test 1 shows DLSSNR's history responds to MVec.
   - Size: 5-6 files, 250-400 lines plus tests, and an SHM version bump (2.1).
   - It also removes the optical flow's cost on that path: one NV optical-flow run and an 8-bit conversion per frame.
     That is a possible fps gain, not measured.
   - Moot if the native backend replaces the helper, because native already uses DLSS's vectors.
2. **Label the Model resolution setting "after the upscaler only", or make the pre-upscaler path use it. Maintenance.**
   - Correct the stale comments: `main:crates/layer/src/shm.rs:53-64` and `main:crates/protocol/src/header.rs:169-172,197-198`.
   - Remove or relabel the Downscaler setting, which is a no-op.
   - Size: 3-4 files, under 30 lines for labels and comments. Making the pre-upscaler path honour the setting is a
     larger change and would trade picture for speed in the default mode.
3. **Send a supported PerfQualityValue (0, 1, 2, 4 or 5) instead of 3. Maintenance.**
   - Only if section 4, test 4 shows 3 is consulted.
   - Size: 1 file, 2 lines (`main:crates/helper/src/ngx.rs:589-590`).
4. **Fix `native-backend:docs/NATIVE_BACKEND.md:470`** (`pass_tuning` should be `global_tuning`). Maintenance. 1 line.
5. **Multi-pass shared parameter block. Image quality, multi-pass only.**
   - If section 4, test 3 confirms the risk, write each pass's style and strengths before its evaluate, not only at
     its creation.
   - Size: 1-2 files, about 20 lines.
   - No effect at 1 pass.

## 4. Not determined from the code, and the runtime test on LordNikon that settles each

1. **Whether DLSSNR 310.8's history responds to MVec at all.** Pre-upscaler path in GTA V, a steady pan, captures
   with "Estimate motion vectors" on and off. Identical output, or no change in ghosting, means real vectors would
   not help either.
2. **Units, sign and depth convention of the game's data.** One `NEURAL_FORGE_PREUPSCALE=dump` capture during a
   known camera pan; inspect `mvec.rg16f` (pixels or normalised, current to previous or the reverse) and the depth
   file (reversed-Z or not). Also check whether DLSSNR multiplies or divides by MVecScale: an A/B of the PIXELS and
   NORMALISED modes on a pan.
3. **Whether Style and the strengths are re-read at evaluate in practice, and the multi-pass effect.**
   - Two passes, an intensity override on pass 0 only.
   - Compare against a one-pass run with pass 0's settings.
   - Also read the helper's `[params] read` log for evaluate-time reads.
4. **Whether PerfQualityValue=3 is consulted.** Helper roundtrip on a fixed frame with 3 and with 1. Byte-identical
   answers, and no "unsupported" in the log, mean it is ignored.
5. **The DLL's auto-mask-off sentinels and the skin fallback.** Re-run the 2.2 parity check (helper roundtrip
   against `dlss5vk`, frame d1, after a reset) with (a) structure 0.5, skin -1, auto-mask on, and (b) structure 0.5,
   skin 2, auto-mask off. Bit-exact on both confirms native's lanes.
6. **Whether Depth or DepthInverted affect the picture.** Helper A/B with `DepthInverted` 0 and 1 on the same frame.
7. **Whether a runtime callback overrides controls.** The helper registers none, and the callback's effect is not
   visible from the code. The `[params] read` log in tests 3 and 5 would show any override.
8. **Keeping the feature on a resolution change instead of recreating it.** Whether the DLL's resize branch
   (resize and invalidate history) gives a correct first frame. Change the game's resolution with a test build that
   keeps the handle, and compare the first answers. Not worth doing unless rebuild time matters.
9. **Native: tuning-change rebuild and the network's internal state on reset.** Not traced in the time box.
