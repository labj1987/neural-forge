//! The pre-upscaler edit cache's reference and offline replay: the algorithm the layer's
//! `crates/layer/shaders/preupscale_cache*.comp` shaders (and `crates/layer/src/preupscale/cache.rs`)
//! mirror step for step, the replay that measures it on a dump (the `cache-replay` binary), and the
//! synthetic dumps and exports the layer's shader tests compare with.
//!
//! Keep in step with the shaders. Every function in [`common`] names the shader step it stands for;
//! when one of the two changes, change the other, so that what `cache-replay` measures offline is
//! what the layer does in a game.
//!
//! The cache runs the model one hold in N. On the holds between, only the model's correction (the
//! edit) is carried along the motion vectors and multiplied onto the game's fresh frame; geometry is
//! never reprojected. One carry, as the shader does it on a cached hold:
//!
//! 1. The edit of the refresh hold: per channel log2((v_answer + eps_v) / (v_colour + eps_v)) in the
//!    exposure-normalised value v = max(scene, 0) * exposure / paper_white, clamped (luma within
//!    +-max_luma_edit stops, each channel within +-chroma_stops of the clamped luma).
//! 2. Where each pixel of this hold was on the refresh hold: the accumulated motion, updated on every
//!    hold since the refresh ([`common::acc_step`], the GPU's rule: this hold's vector, taken from the
//!    nearest surface of the 3x3 depth neighbourhood, plus the previous accumulated field read
//!    bilinearly where that vector and the change of jitter lead), plus the change of jitter since the
//!    refresh. ([`common::chain_step`], "walk", re-reads each older hold's vectors at the running
//!    position instead.)
//! 3. The history read there: four bilinear taps, each admitted by relative depth against this pixel
//!    (full weight to depth_tol, none at twice it), renormalised, for the weight and the luma the
//!    colour check reads; the edit is the strongest tap's own, not their blend (a bilinear blend at
//!    the jitter's sub-pixel offset keeps a fifth to a third of the edit's detail; docs/LESSONS.md,
//!    "3.3 edit cache: what was tried").
//! 4. Validation: the colour check on log luma against this hold's 3x3 range (noise-aware), which
//!    only dims the high band (lerp(0.6, 1, v)); the fingerprint (normalised log luma and two log
//!    chroma ratios against the 3x3 range), which rejects the edit.
//! 5. The band split: a coverage-weighted pyramid (1/4, 1/16, 1/64) of the validated edit, read
//!    jointly-bilaterally on depth and luma, coarse to fine, fading to no edit where nothing of this
//!    surface is near (surface fill); final edit = low + confidence * (history - low).
//!
//! Over time ([`replay::run_cadence`], which the shaders mirror too): a refresh hold shows the
//! model's answer, its cached holds carry it (steps 3-5), with a guided filter by staleness.
//!
//! Every step is computed in f32 and the means the metrics take in f64, the precision the shaders'
//! numbers were first measured with (a numpy reference, which this crate replaced).
//!
//! Ported from optiscaler-dlss5 by Skynizz (<https://github.com/Skynizz/optiscaler-dlss5>), commit
//! 6bfc4389c25054ef6747df6e6b1b470e9541c4da: `OptiScaler/shaders/dlssnr/precompile/dlssnr_cache.hlsl`
//! (LinDepth, PrintOf, PrintMatch, ClampEdit, NearestSurfaceMotion, ReadHistory, FetchLevel,
//! SurfaceLow, modes 1, 3, 4, 5) and `tools/dlssnr_cache/` (`nrcache_common.py`,
//! `measure_ghosting.py`, `measure_reprojection.py`, `measure_pulse.py`, `flicker_lab.py`: the
//! reference and the metrics). Upstream is GPL-3.0; this port is distributed under this project's
//! AGPL-3.0-or-later (see ATTRIBUTION.md). Where upstream's Python tools and its shader disagree,
//! this follows the shader. Upstream's surroundings term of the fingerprint (ContextAt, mode 12) and
//! its Catmull-Rom re-read of the history are left out: both measured worse on our dumps
//! (docs/LESSONS.md, "3.3 edit cache: what was tried").

pub mod common;
pub mod pyfmt;
pub mod replay;

#[cfg(test)]
mod tests;
