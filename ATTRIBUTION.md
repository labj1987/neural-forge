# Attribution

**As of 2026-09-17, this project is no longer clean-room with respect to
[DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87), AGPL-3.0.** Earlier
versions of NeuralForge deliberately avoided reading DLSS5VKLayer's own source (or
anything GPL/AGPL-licensed) specifically to keep this project MIT-licensed. That
boundary produced a worse result — months of re-deriving fixes from behavior and docs
alone, repeatedly landing on techniques upstream had already tried and rejected — for
no benefit anyone actually wanted, so it was dropped. DLSS5VKLayer's own `LICENSE`
file is AGPL-3.0, so once its source is read and adapted, AGPL-3.0 is a requirement of
that upstream license for the derived code, not merely a preference of this project.
This project's own license is therefore **AGPL-3.0-or-later** (see [LICENSE](LICENSE)
and [README.md](README.md#license)), and this file now records what is actually read,
adapted, or taken from DLSS5VKLayer's real source, not just its documented behavior.

## What's taken, and how

| Source | What's taken |
|---|---|
| [DLSS5VKLayer](https://github.com/bmitch87/DLSS5VKLayer) (bmitch87), AGPL-3.0 — specifically `layer_linux/src/composition.cpp`, `composition.h`, and `dlssnr/dlssnr.hlsl` read directly | **The composition strategy**: no temporal accumulator/held-answer carried forward and blended against newer frames. DLSS5VKLayer's own source comment (`dlssnr.hlsl`, in the resolve function) states this directly: a reprojected-history accumulator was built, measured as a dead end twice ("the model re-decides its detail with the framing, so an old answer does not belong to a new frame and reprojecting it only moves where the disagreement lands"), and removed — replaced with re-anchoring the composition to the model's cached answer against the *current* frame, fresh, every single present, with no motion-based suppression of the edit at all. NeuralForge's own `carry_delta`/motion-mask compositor (`compose.comp`, `crates/layer/src/composition/gpu.rs::record_temporal_delta_into_image`, added v0.1.49–v0.1.65) was independently arrived at and is architecturally the same kind of reprojection-with-suppression technique DLSS5VKLayer's own source documents trying and rejecting — this is why it kept ghosting/flickering no matter how the suppression threshold was tuned. Replaced per this finding; see `CHANGELOG.md` for the version this landed in. |
| DLSS5VKLayer, `layer_linux/src/dlssnr/dlssnr.hlsl` — **ported, not clean-room** | **The proxy encode**, in `crates/layer/src/composition/encode.rs` and `shaders/encode.comp`: the soft knee (luminance roll-off above 0.75 plus the peak-channel headroom pass), the unclipped `Neutwo` reversible proxy, and the hybrid curve, along with the white-point divide that precedes them. Upstream's `ATTRIBUTION.md` marks `dlssnr.hlsl` as GPL-3.0-tainted in their own tree (derived from OptiScaler-family forks); combining it here is what NeuralForge's AGPL-3.0-or-later relicense permits (AGPL-3.0 and GPL-3.0 are mutually compatible for combination). Kept in its own module precisely so `composition/color.rs`'s clean-room claim stays true — nothing in *that* file is read from `dlssnr.hlsl`. Taken because this project had no encode at all: the proxy handed to the model was a bit-identical copy of the frame, which degenerates the ratio transfer (`proxy == original` collapses its rescale target onto the frame) and shows the model blown highlights instead of gradation. Upstream's own reason for the knee is the same artifact class this project spent v0.1.49–v0.1.72 fighting: "the model is never shown a field of flat white whose blown pixels flip between frames — unstable input is unstable output". The peak-channel headroom pass is likewise taken with its measured justification intact (a saturated blue above 1.0 clears the luminance knee, gets one channel clipped, and arrives hue-rotated — upstream measured this as a green cast over the sky, denim and minimap **in GTA V specifically**). |
| DLSS5VKLayer, `dlssnr.hlsl` — **ported** | **The composition guard** in `compose.comp`'s additive path: the `1/512` ratio floor applied above *and* below, plus the asymmetric `lift` (brightening room shrinking to none as a pixel approaches black) and `drop` (darkening room shrinking to none as it approaches white). Replaced a symmetric, epsilon-gated bound that upstream's source names as the cause of "the boiling: patches of lighter colour crawling over otherwise still geometry, worst where the picture is darkest". |
| DLSS5VKLayer, same files | The overall two-leg structure (capture leg encodes a proxy and downloads it; a helper round trip happens in between; a compose leg uploads the answer and recomposes onto the swapchain image) and the general shape of the luminance-ratio transfer (`ratio = originalLuma / proxyLuma` below the proxy's level, a headroom-preserving inverse above it, blended via `lerp` with a saturated strength, hue corrected in OkLab) — NeuralForge already had an equivalent, independently-derived version of this exact ratio math (`composition::gpu::UpgradeToneMap`/the "classic" non-`carry_delta` dispatch path, built earlier from the RenoDX design below) that had simply never been wired into the live per-present hot path. Confirming DLSS5VKLayer's own resolve function uses the same shape of math is what justified routing the hot path through it instead of `carry_delta`, rather than porting new math wholesale. |

## Ported functions (upstream parity work, from 0.3.1-1 / commit 117c953)

Each function ported or re-implemented from DLSS5VKLayer's source is listed here with the
upstream file it came from. Nothing listed here lands in
`crates/layer/src/composition/color.rs`, which remains the one clean-room file.

| Function here | Upstream file / function |
|---|---|
| `crates/helper/src/ngx.rs::NgxTuning`, `set_create_tuning` | `core/ngx_snippet.cpp` `NgxTuning`, `NgxSetCreateTuning` (create-time-only tuning block) |
| `crates/helper/src/ngx.rs::maintain_feature` | `helper/main.cpp` `MaintainPasses` / `TuningFor` (compare by value, debounce by `rebuild_settle_ms`, destroy then recreate). Single-feature only until multipass lands. |
| `crates/helper/src/ngx.rs` per-evaluate `Sharpness` write, `PerfQualityValue = 3`, `NEURAL_FORGE_SKIP_NVAPI` handling | `core/ngx_snippet.cpp` `NgxSetSharpness`, `NgxCreatePass`, `NgxLoadAndInit` |
| `crates/helper/src/guard.rs`: DBG_PRINTEXCEPTION exclusion, guarded-hit cap (24), rip/fault/module-range logging (`register_module`, `describe`) | `core/guard.cpp` `GuardVeh`, `RegisterModuleRange`, `DescribeRange`. The setjmp side deliberately differs: upstream's unwinding `setjmp` form is not used here (`_setjmp(buf, NULL)`). |
| `crates/helper/src/ngx.rs`: 64x64 minimum feature size (`MIN_FEATURE_DIM`), rebuild on size change | `core/ngx_snippet.cpp` `kMinW`/`kMinH`; `helper/main.cpp` `EnsureNeural`. Upstream's re-run of `NgxLoadAndInit` on resize is not copied. |

## What was previously taken clean-room (still accurate, unaffected by the above)

| Source | What's taken |
|---|---|
| [RenoDX](https://github.com/clshortfuse/renodx) (clshortfuse) — MIT | The color composition design: the two-branch luminance/headroom rule, the OkLab hue-correction step, and the reversible neutral-axis gamut compression. This is RenoDX's own DLSS 5 addon design, MIT-licensed, read and reimplemented directly from RenoDX's own public source. |
| Björn Ottosson ([bottosson.github.io/posts/oklab](https://bottosson.github.io/posts/oklab/)) | The OkLab conversion matrices, published as public reference constants. |
| Public domain / standard color science | sRGB transfer function, SMPTE ST.2084 (PQ) encode/decode, Hunt-Pointer-Estevez LMS conversion — textbook formulas, attributable to no one project. |
| Public domain / standard resampling literature | The Lanczos, Catmull-Rom, Mitchell-Netravali, and Kaiser-windowed-sinc kernels used for the supersampling downscale leg — implemented from their mathematical definitions. |

## What's still not taken

DLSS5VKLayer's own `dlssnr.hlsl` documents (inline, at its "native + edit" transfer
mode) that one specific technique — an additive edit rule and a guard shape for it —
comes from [xenmods/DLSSNR-Cost-Scaler](https://github.com/xenmods/DLSSNR-Cost-Scaler),
MIT, "no code is copied." NeuralForge has not adopted that specific mode
(`transfer == 2`/"native + edit") and this file will be updated if that changes. The
NGX caller-identity spoof (`crates/helper/src/spoof.rs`) remains an independent
reimplementation of the same generic PE import-table-hook technique, not read from
DLSS5VKLayer's C++ — see [README.md's Legal section](README.md#legal) for why that
mechanism carries its own separate legal exposure regardless of either project's
license.

## License texts

This project's own code is AGPL-3.0-or-later — see [LICENSE](LICENSE). DLSS5VKLayer is
AGPL-3.0; adapting its composition strategy and confirming its transfer math directly
is why this project's license has to be AGPL-3.0 (upstream's own license requires it
for the derived code). RenoDX's MIT license applies to the design
this project's color composition math is independently derived from; xenmods'
DLSSNR-Cost-Scaler (MIT) is referenced above for provenance only, matching
DLSS5VKLayer's own attribution of it, and no code from it appears in this repository.
