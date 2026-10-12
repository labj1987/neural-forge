#!/usr/bin/env python3
"""The pre-upscaler edit cache, in numpy: the reference the GPU shader mirrors.

Keep in step with crates/layer/shaders/preupscale_cache*.comp. Every function below names the
shader step it stands for; when one of the two changes, change the other, so that what
scripts/cache_replay.py measures offline is what the layer does in a game.

The cache runs the model one hold in N. On the holds between, only the model's correction (the
edit) is carried along the motion vectors and multiplied onto the game's fresh frame; geometry is
never reprojected. One carry here, as the shader does it on a cached hold:

  1. The edit of the refresh hold: per channel log2((v_answer + eps_v) / (v_colour + eps_v)) in the
     exposure-normalised value v = max(scene, 0) * exposure / paper_white, clamped (luma within
     +-max_luma_edit stops, each channel within +-chroma_stops of the clamped luma).
  2. Where each pixel of this hold was on the refresh hold: the accumulated motion, updated on every
     hold since the refresh (acc_step, the GPU's rule: this hold's vector, taken from the nearest
     surface of the 3x3 depth neighbourhood, plus the previous accumulated field read bilinearly
     where that vector and the change of jitter lead), plus the change of jitter since the refresh.
     (chain_step, "walk", re-reads each older hold's vectors at the running position instead.)
  3. The history read there: four bilinear taps, each admitted by relative depth against this
     pixel (full weight to depth_tol, none at twice it), renormalised, for the weight and the luma
     the colour check reads; the edit is the strongest tap's own, not their blend (a bilinear blend
     at the jitter's sub-pixel offset keeps a fifth to a third of the edit's detail; docs/LESSONS.md,
     "3.3 edit cache: what was tried").
  4. Validation: the colour check on log luma against this hold's 3x3 range (noise-aware), which
     only dims the high band (lerp(0.6, 1, v)); the fingerprint (normalised log luma and two log
     chroma ratios against the 3x3 range), which rejects the edit.
  5. The band split: a coverage-weighted pyramid (1/4, 1/16, 1/64) of the validated edit, read
     jointly-bilaterally on depth and luma, coarse to fine, fading to no edit where nothing of this
     surface is near (surface fill); final edit = low + confidence * (history - low).

Over time (cache_replay.py's run_cadence, which the shaders mirror too): a refresh hold shows the
model's answer, its cached holds carry it (steps 3-5), with a guided filter by staleness.

Ported from optiscaler-dlss5 by Skynizz (https://github.com/Skynizz/optiscaler-dlss5), commit
6bfc4389c25054ef6747df6e6b1b470e9541c4da: OptiScaler/shaders/dlssnr/precompile/dlssnr_cache.hlsl
(LinDepth, PrintOf, PrintMatch, ClampEdit, NearestSurfaceMotion, ReadHistory, FetchLevel,
SurfaceLow, modes 1, 3, 4, 5) and tools/dlssnr_cache/nrcache_common.py,
measure_ghosting.py, measure_reprojection.py (the numpy mirror and the metrics). Upstream is
GPL-3.0; this port is distributed under this project's AGPL-3.0-or-later (see ATTRIBUTION.md).
Where upstream's Python tools and its shader disagree, this follows the shader. Upstream's
surroundings term of the fingerprint (ContextAt, mode 12) and its Catmull-Rom re-read of the
history are left out: both measured worse on our dumps (docs/LESSONS.md, "3.3 edit cache: what was
tried").

numpy only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
EPS_V = 2.0 ** -9
F32 = np.float32


# --------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------

@dataclass
class Params:
    """Every knob of the carry, with the shader's defaults."""

    eps_v: float = EPS_V            # ratio floor in exposure-normalised v (docs/ARCHITECTURE.md 4.9)
    max_luma_edit: float = 1.5      # ClampEdit luma limit, stops; 0 turns the whole clamp off
    chroma_stops: float = 1.0       # ClampEdit: each channel within this of the clamped luma
    depth_tol: float = 0.10         # relative depth: full weight inside, none at twice it
    colour_tol: float = 0.50        # stops; soft to twice it
    noise_aware: bool = True        # colour check against this hold's 3x3 range, not the pixel
    colour_floor: float = 0.6       # a colour failure dims the high band to this, not to zero
    high_decay: float = 1.0         # per-carry confidence factor (shader 1.0)
    fingerprint: bool = True        # reject where the frame no longer looks like the refresh hold
    print_tol: float = 0.25         # fingerprint tolerance, stops; soft to twice it
    surface_fill: bool = True       # low band from this surface only, fading to no edit
    bilateral: bool = True          # push-pull path: first level read jointly-bilaterally
    sigma_depth: float = 0.15       # bilateral sigma, log2 depth (x(1+level) in the surface fill)
    sigma_luma: float = 1.0         # bilateral sigma, stops of luma (likewise)
    nearest_surface: bool = True    # motion from the nearest depth in the 3x3 neighbourhood
    jitter: bool = True             # add the change of jitter to the motion
    chain_sample: str = "bilinear"  # how a chained step reads the next hold's vectors (walk only)
    chain: str = "recurrence"       # "recurrence": the GPU's accumulated field; "walk": chain_step
    low_gain: float = 1.0
    high_gain: float = 1.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------------------------------
# The dump (docs/ARCHITECTURE.md 4.9, v1)
# --------------------------------------------------------------------------------------------------

DUMP_FORMAT = "neural-forge-cache-dump"
DUMP_VERSION = 1


def exposure_ok(e) -> bool:
    """The decode's own rule: an exposure outside this range leaves the frame untouched."""
    return e is not None and 1e-6 < float(e) < 65504.0


@dataclass
class Hold:
    """One dumped hold, as stored: scene-linear colour and answer, vectors in render pixels.

    A hold the layer could not have cached (no jitter known, no vectors copied) says so: the layer
    refreshes on it, and so do the tools (`Frame.carryable`)."""

    index: int
    jitter: np.ndarray        # (2,), render pixels (zeros when the manifest's jitter is null)
    exposure: float           # 0 when the manifest's exposure is null: unusable
    colour: np.ndarray        # H x W x 3 f32, scene-linear, as stored (not sanitised)
    answer: np.ndarray        # H x W x 3 f32
    mvec: np.ndarray          # H x W x 2 f32, render pixels (mvec_scale applied); zeros when not valid
    depth: np.ndarray | None  # H x W f32 device depth, None when the dump or this hold has none
    jitter_known: bool = True
    mvec_valid: bool = True


@dataclass
class Dump:
    path: str
    manifest: dict
    width: int
    height: int
    paper_white: float
    mvec_scale: tuple
    has_depth: bool
    frames: list = field(default_factory=list)  # the manifest's frame entries

    def __len__(self) -> int:
        return len(self.frames)

    def _read(self, name: str, dtype: str, channels: int) -> np.ndarray:
        path = os.path.join(self.path, name)
        data = np.fromfile(path, dtype=dtype)
        want = self.width * self.height * channels
        if data.size != want:
            raise ValueError(f"{path}: {data.size} values, expected {want} ({self.width}x{self.height}x{channels})")
        shape = (self.height, self.width, channels) if channels > 1 else (self.height, self.width)
        return data.reshape(shape).astype(F32)

    def hold(self, k: int) -> Hold:
        f = self.frames[k]
        files = f["files"]
        mvec_valid = f.get("mvec_valid", True) is not False
        mv = self._read(files["mvec"], "<f2", 2) * np.array(self.mvec_scale, F32)
        if not mvec_valid:
            mv = np.zeros_like(mv)
        depth_valid = f.get("depth_valid", True) is not False
        depth = self._read(files["depth"], "<f4", 1) if self.has_depth and depth_valid and files.get("depth") else None
        # null: unknown jitter (no carry from or to this hold), unusable exposure.
        jitter = f.get("jitter", [0.0, 0.0])
        exposure = f.get("exposure", 0.0)
        return Hold(index=int(f.get("index", k)), jitter=np.array(jitter if jitter is not None else [0.0, 0.0], F32),
                    exposure=float(exposure) if exposure is not None else 0.0,
                    colour=self._read(files["colour"], "<f2", 4)[..., :3],
                    answer=self._read(files["answer"], "<f2", 4)[..., :3], mvec=mv, depth=depth,
                    jitter_known=jitter is not None, mvec_valid=mvec_valid)


def load_dump(path: str) -> Dump:
    with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as fh:
        m = json.load(fh)
    if m.get("format") != DUMP_FORMAT:
        raise ValueError(f"{path}: not a {DUMP_FORMAT} (format {m.get('format')!r})")
    if int(m.get("version", 0)) != DUMP_VERSION:
        raise ValueError(f"{path}: dump version {m.get('version')}, this tool reads {DUMP_VERSION}")
    frames = list(m["frames"])
    if len(frames) != int(m.get("holds", len(frames))):
        raise ValueError(f"{path}: manifest says {m.get('holds')} holds but lists {len(frames)} frames")
    return Dump(path=path, manifest=m, width=int(m["width"]), height=int(m["height"]),
                paper_white=float(m.get("paper_white", 3.0)), mvec_scale=tuple(m.get("mvec_scale", [1.0, 1.0])),
                has_depth=m.get("depth_format", "none") != "none", frames=frames)


def write_dump(path: str, holds: list, paper_white: float = 3.0, game: str = "synthetic",
               reverse_z=None, extra: dict | None = None) -> None:
    """Writes holds (list of Hold) as a v1 dump, files cropped to the real extent as the layer does."""
    os.makedirs(path, exist_ok=True)
    h, w = holds[0].colour.shape[:2]
    has_depth = holds[0].depth is not None
    frames = []
    for k, hd in enumerate(holds):
        sub = f"hold-{k:02d}"
        os.makedirs(os.path.join(path, sub), exist_ok=True)

        def rgba(x):
            return np.concatenate([x, np.ones((h, w, 1), F32)], -1).astype("<f2")

        rgba(hd.colour).tofile(os.path.join(path, sub, "colour.rgba16f"))
        rgba(hd.answer).tofile(os.path.join(path, sub, "answer.rgba16f"))
        hd.mvec.astype("<f2").tofile(os.path.join(path, sub, "mvec.rg16f"))
        files = {"colour": f"{sub}/colour.rgba16f", "answer": f"{sub}/answer.rgba16f", "mvec": f"{sub}/mvec.rg16f"}
        if has_depth:
            hd.depth.astype("<f4").tofile(os.path.join(path, sub, "depth.r32f"))
            files["depth"] = f"{sub}/depth.r32f"
        frames.append({"index": k, "hold": 1000 + k, "time_ms": 1_000_000 + 16 * k,
                       "jitter": [float(hd.jitter[0]), float(hd.jitter[1])], "exposure": float(hd.exposure),
                       "history": k > 0, "files": files})
    manifest = {"format": DUMP_FORMAT, "version": DUMP_VERSION, "holds": len(holds), "width": w, "height": h,
                "path": "split", "paper_white": paper_white, "mvec_scale": [1.0, 1.0],
                "depth_format": "D32_SFLOAT_S8_UINT" if has_depth else "none", "reverse_z": reverse_z,
                "game": game, "frames": frames}
    if extra:
        manifest.update(extra)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


# The share of depth texels that must be neither 0 nor 1 for depth to tell its convention (the layer's
# cache::DEPTH_GEOMETRY_MIN): cleared depth alone, or depth never written, says nothing.
DEPTH_GEOMETRY_MIN = 0.05


def infer_reverse_z(depths: list) -> tuple:
    """Which way the dump's depth runs. Returns (reverse_z or None, reason).

    Reversed-Z puts the near plane at 1 and the far plane (sky, cleared depth) at 0. First the cleared
    value: a real share of texels at exactly 0 (and few at 1) is reversed-Z, the other way forward.
    Failing that, the ground: the bottom of a game frame is nearer than its top, so depth growing
    toward the bottom is reversed-Z. Holds with less drawn geometry than DEPTH_GEOMETRY_MIN are left
    out, as the layer leaves them out."""
    depths = [x for x in depths
              if 1.0 - float(np.mean(np.nan_to_num(x, nan=0.5) == 0.0)) - float(np.mean(np.nan_to_num(x, nan=0.5) == 1.0))
              >= DEPTH_GEOMETRY_MIN]
    if not depths:
        return None, f"no hold has {100 * DEPTH_GEOMETRY_MIN:.0f}% of its depth drawn (neither 0 nor 1); cannot tell"
    d = np.concatenate([np.nan_to_num(x, nan=0.5).ravel() for x in depths])
    z0, z1 = float(np.mean(d == 0.0)), float(np.mean(d == 1.0))
    if max(z0, z1) > 0.005 and (z0 > 2 * z1 or z1 > 2 * z0):
        rev = z0 > z1
        return rev, f"{100 * z0:.1f}% of depth texels are 0, {100 * z1:.1f}% are 1 (cleared = far)"
    tops, bottoms = [], []
    for x in depths:
        q = max(1, x.shape[0] // 4)
        tops.append(np.median(x[:q]))
        bottoms.append(np.median(x[-q:]))
    top, bottom = float(np.median(tops)), float(np.median(bottoms))
    if abs(bottom - top) <= 1e-9 * max(abs(top), abs(bottom), 1e-30):
        return None, f"depth the same at the top and bottom of the frame ({top:.6g}); cannot tell"
    return bottom > top, f"median depth top quarter {top:.6g}, bottom quarter {bottom:.6g} (the bottom is nearer)"


# --------------------------------------------------------------------------------------------------
# The edit (docs/ARCHITECTURE.md 4.9-4.10; shader FreshEdit / ClampEdit / apply)
# --------------------------------------------------------------------------------------------------

def luma(x: np.ndarray) -> np.ndarray:
    return (x[..., :3] @ LUMA).astype(F32)


def to_v(scene: np.ndarray, exposure: float, paper_white: float) -> np.ndarray:
    """Exposure-normalised value: v = max(scene, 0) * e / W. NaN and Inf count as 0, as the encode does."""
    s = np.nan_to_num(scene, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.maximum(s, 0.0) * (exposure / paper_white)).astype(F32)


def clamp_edit(e: np.ndarray, p: Params, scale: float = 1.0) -> np.ndarray:
    """ClampEdit: the luma within +-max_luma_edit * max(scale, 1), each channel within chroma_stops of it."""
    if not (p.max_luma_edit > 0 and math.isfinite(p.max_luma_edit)):
        return e.astype(F32, copy=False)
    limit = p.max_luma_edit * max(scale, 1.0)
    l = e @ LUMA
    lc = np.clip(l, -limit, limit)
    e = e + (lc - l)[..., None]
    return np.clip(e, (lc - p.chroma_stops)[..., None], (lc + p.chroma_stops)[..., None]).astype(F32)


def edit_of(v_colour: np.ndarray, v_answer: np.ndarray, p: Params) -> np.ndarray:
    """The model's edit in v: log2((v_answer + eps) / (v_colour + eps)) per channel, clamped."""
    e = np.log2((np.maximum(v_answer, 0) + p.eps_v) / (np.maximum(v_colour, 0) + p.eps_v)).astype(F32)
    return clamp_edit(e, p)


def apply_edit(v_colour: np.ndarray, edit: np.ndarray, p: Params) -> np.ndarray:
    """v_out = (v_colour + eps) * 2^edit - eps, at least 0."""
    return np.maximum((np.maximum(v_colour, 0) + p.eps_v) * np.exp2(edit) - p.eps_v, 0).astype(F32)


def lin_depth(depth: np.ndarray | None, reverse_z: bool, shape) -> np.ndarray:
    """LinDepth: proportional to view depth (exact up to a constant for reversed-Z). Only ratios are used.
    Without depth every pixel reads 1, so every depth test passes."""
    if depth is None:
        return np.ones(shape, F32)
    d = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    return (1.0 / np.maximum(d if reverse_z else 1.0 - d, 1e-7)).astype(F32)


# --------------------------------------------------------------------------------------------------
# Sampling, in pixel units (a pixel centre is i + 0.5; the shader's uv times the size)
# --------------------------------------------------------------------------------------------------

def pixel_centres(h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(F32)
    return np.stack([xs + 0.5, ys + 0.5], -1)


def bilinear_taps(pos: np.ndarray, w: int, h: int):
    """The four taps of a bilinear read at pos (pixel units), edge-clamped, in the shader's order
    (k & 1 = x, k >> 1 = y). Yields (flat index, x index, y index, weight)."""
    px = pos[..., 0] - 0.5
    py = pos[..., 1] - 0.5
    x0 = np.floor(px)
    y0 = np.floor(py)
    fx = (px - x0).astype(F32)
    fy = (py - y0).astype(F32)
    x0 = x0.astype(np.intp)
    y0 = y0.astype(np.intp)
    for k in range(4):
        ox, oy = k & 1, k >> 1
        xi = np.clip(x0 + ox, 0, w - 1)
        yi = np.clip(y0 + oy, 0, h - 1)
        wt = (fx if ox else 1 - fx) * (fy if oy else 1 - fy)
        yield yi * w + xi, xi, yi, wt


def gather(img: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat = img.reshape(-1, img.shape[2]) if img.ndim == 3 else img.reshape(-1)
    return flat[idx]


def sample_bilinear(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """A linear-clamp sampler read at pos (pixel units)."""
    h, w = img.shape[:2]
    out = None
    for idx, _, _, wt in bilinear_taps(pos, w, h):
        v = gather(img, idx)
        term = v * (wt[..., None] if v.ndim == 3 else wt)
        out = term if out is None else out + term
    return out.astype(F32)


def sample_nearest(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xi = np.clip(np.floor(pos[..., 0]).astype(np.intp), 0, w - 1)
    yi = np.clip(np.floor(pos[..., 1]).astype(np.intp), 0, h - 1)
    return gather(img, yi * w + xi)


def neighbourhood(img: np.ndarray) -> np.ndarray:
    """The 3x3 neighbourhood, edge-clamped, stacked on a new first axis in the shader's scan order
    (dy outer, dx inner)."""
    h, w = img.shape[:2]
    pad = [(1, 1), (1, 1)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, mode="edge")
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def level_taps(h: int, w: int, lh: int, lw: int):
    """Bilinear taps into an lh x lw grid at the h x w frame's pixel centres (uv * size - 0.5), which is
    separable: yields (flat index, weight) as full-frame arrays."""
    px = (np.arange(w, dtype=F32) + 0.5) / w * lw - 0.5
    py = (np.arange(h, dtype=F32) + 0.5) / h * lh - 0.5
    x0, y0 = np.floor(px), np.floor(py)
    fx, fy = (px - x0).astype(F32), (py - y0).astype(F32)
    x0, y0 = x0.astype(np.intp), y0.astype(np.intp)
    for k in range(4):
        ox, oy = k & 1, k >> 1
        xi = np.clip(x0 + ox, 0, lw - 1)
        yi = np.clip(y0 + oy, 0, lh - 1)
        wx = fx if ox else 1 - fx
        wy = fy if oy else 1 - fy
        yield yi[:, None] * lw + xi[None, :], (wy[:, None] * wx[None, :]).astype(F32)


# --------------------------------------------------------------------------------------------------
# Per-hold inputs, computed once (what mode 16 and the capture produce)
# --------------------------------------------------------------------------------------------------

@dataclass
class Frame:
    index: int
    usable: bool              # exposure usable: the decode wrote an answer
    jitter_known: bool        # the manifest's jitter was not null
    mvec_valid: bool          # its vectors were copied (the layer's motion pass had them)
    has_depth: bool           # this hold had depth
    jitter: np.ndarray
    v_colour: np.ndarray      # H x W x 3
    edit: np.ndarray          # H x W x 3, the model's (clamped) edit: "true" for this hold
    lin: np.ndarray           # H x W pseudo-linear depth
    log_luma: np.ndarray      # H x W, log2(luma(v) + eps): normalised (paper white is 1 in v)
    prt: np.ndarray           # H x W x 3 fingerprint at capture: log luma, log r/g, log b/g
    prt_lo: np.ndarray        # H x W x 3, 3x3 range of this hold's print
    prt_hi: np.ndarray
    mv: np.ndarray            # H x W x 2, render pixels to the previous hold (nearest-surface if on)
    disp_answer: np.ndarray   # H x W x 3, display(answer)
    base: float               # mean |display(colour) - display(answer)|: the model's whole effect


def print_of(v: np.ndarray, p: Params) -> np.ndarray:
    """PrintOf with white = 1 (v is already exposure-normalised)."""
    q = np.maximum(v, 0) + p.eps_v
    return np.stack([np.log2(luma(np.maximum(v, 0)) + p.eps_v), np.log2(q[..., 0] / q[..., 1]),
                     np.log2(q[..., 2] / q[..., 1])], -1).astype(F32)


def nearest_surface_motion(mv: np.ndarray, lin: np.ndarray) -> np.ndarray:
    """NearestSurfaceMotion / mode 16: each pixel takes the vector of the nearest depth in its 3x3
    neighbourhood (the first in scan order on a tie)."""
    h, w = lin.shape
    best = np.argmin(neighbourhood(lin), axis=0)
    dy = best // 3 - 1
    dx = best % 3 - 1
    ys, xs = np.mgrid[0:h, 0:w]
    yi = np.clip(ys + dy, 0, h - 1)
    xi = np.clip(xs + dx, 0, w - 1)
    return mv[yi, xi]


def display(v: np.ndarray) -> np.ndarray:
    """The reference's neutral tone map for comparing pictures: Reinhard on luma, then gamma 1/2.2
    (paper white is 1 in v)."""
    scale = 1.0 / (1.0 + luma(v))
    return (np.clip(v * scale[..., None], 0, 1) ** (1 / 2.2)).astype(F32)


def prepare(hold: Hold, dump_white: float, reverse_z: bool, p: Params) -> Frame:
    h, w = hold.colour.shape[:2]
    usable = exposure_ok(hold.exposure)
    e = hold.exposure if usable else 0.0
    vc = to_v(hold.colour, e, dump_white)
    va = to_v(hold.answer, e, dump_white)
    lin = lin_depth(hold.depth, reverse_z, (h, w))
    prt3 = print_of(vc, p)
    nb = neighbourhood(prt3)
    mv = np.nan_to_num(hold.mvec, nan=0.0, posinf=0.0, neginf=0.0)
    if p.nearest_surface and hold.depth is not None:
        mv = nearest_surface_motion(mv, lin)
    da = display(va)
    return Frame(index=hold.index, usable=usable, jitter_known=hold.jitter_known, mvec_valid=hold.mvec_valid,
                 has_depth=hold.depth is not None, jitter=hold.jitter, v_colour=vc, edit=edit_of(vc, va, p),
                 lin=lin, log_luma=prt3[..., 0].copy(), prt=prt3,
                 prt_lo=nb.min(axis=0), prt_hi=nb.max(axis=0), mv=mv.astype(F32), disp_answer=da,
                 base=float(np.abs(display(vc) - da).mean()))


# --------------------------------------------------------------------------------------------------
# Motion chaining (mode 5's job, done one hold at a time)
# --------------------------------------------------------------------------------------------------

def chain_step(pos: np.ndarray, cur: Frame, prev: Frame, p: Params) -> np.ndarray:
    """Where a point at pos (pixel units) in hold `cur` was in hold `prev`: cur's vectors read at pos,
    plus the change of jitter (prev = x + mv + jitter[t-1] - jitter[t])."""
    read = sample_bilinear if p.chain_sample == "bilinear" else sample_nearest
    out = pos + read(cur.mv, pos)
    if p.jitter:
        out = out + (prev.jitter - cur.jitter)
    return out.astype(F32)


def on_screen(pos: np.ndarray, h: int, w: int) -> np.ndarray:
    return (pos[..., 0] >= 0) & (pos[..., 0] <= w) & (pos[..., 1] >= 0) & (pos[..., 1] <= h)


def sample_bilinear_taken(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """A bilinear read that skips taps of weight 0, so that a NaN there does not spread (a NaN in a
    tap that counts does). The shader's own loop over the four taps."""
    h, w = img.shape[:2]
    out = np.zeros(pos.shape[:-1] + img.shape[2:], F32)
    for idx, _, _, wt in bilinear_taps(pos, w, h):
        v = gather(img, idx)
        wt_b = wt[..., None] if v.ndim == 3 else wt
        out = out + np.where(wt_b > 0, v * wt_b, 0)
    return out.astype(F32)


def acc_step(acc_prev: np.ndarray | None, cur: Frame, prev_jitter: np.ndarray, p: Params) -> np.ndarray:
    """preupscale_cache_motion.comp: the accumulated motion of hold `cur` back to the last refresh,
    acc_t(x) = mv_t(x) + acc_{t-1}(x + mv_t(x) + jitter[t-1] - jitter[t]), the previous field read
    bilinearly (`acc_prev` None: the hold right after a refresh, whose field is zero). Jitter is left
    out of the field; it telescopes to jitter[refresh] - jitter[t] (chained_position). NaN where the
    step's position leaves the frame or reads a NaN: the chain is lost there."""
    h, w = cur.mv.shape[:2]
    p1 = pixel_centres(h, w) + cur.mv
    if p.jitter:
        p1 = p1 + (prev_jitter - cur.jitter)
    acc = cur.mv.copy() if acc_prev is None else cur.mv + sample_bilinear_taken(acc_prev, p1)
    acc[~on_screen(p1, h, w)] = np.nan
    return acc.astype(F32)


def chained_position(acc: np.ndarray, cur: Frame, src: Frame, p: Params) -> tuple:
    """Where each pixel of `cur` was on the refresh hold `src`, from the accumulated field: (pos, on).
    Off the chain (NaN) the position is the pixel itself, and `on` is false."""
    h, w = acc.shape[:2]
    on = np.isfinite(acc).all(axis=-1)
    pos = pixel_centres(h, w) + np.where(on[..., None], acc, 0)
    if p.jitter:
        pos = pos + (src.jitter - cur.jitter)
    return pos.astype(F32), on


# --------------------------------------------------------------------------------------------------
# One carry: refresh hold `src` -> this hold `dst`, at chained positions `pos`
# --------------------------------------------------------------------------------------------------

def can_source(src: Frame) -> bool:
    """`cache::State::went_through`: a refresh stores an edit for the holds after it only with a usable
    exposure and a known jitter."""
    return src.usable and src.jitter_known


def can_carry(src: Frame, dst: Frame) -> bool:
    """`cache::State::plan`: a hold is cached from `src` only with a usable exposure, a known jitter,
    vectors copied and the refresh's presence of depth; any other hold refreshes."""
    return can_source(src) and dst.usable and dst.jitter_known and dst.mvec_valid and dst.has_depth == src.has_depth


def soft(d: np.ndarray, tol: float) -> np.ndarray:
    return np.clip((2.0 * tol - d) / max(tol, 1e-4), 0, 1)


def outside(v, lo, hi):
    return np.maximum(np.maximum(lo - v, v - hi), 0)


def read_history(src: Frame, pos: np.ndarray, lin_c: np.ndarray, p: Params) -> dict:
    """ReadHistory: four bilinear taps of the refresh hold's history at pos, each admitted by relative
    depth; the edit and the fingerprint from the strongest tap."""
    h, w = lin_c.shape
    conf = np.zeros((h, w), F32)
    ll = np.zeros((h, w), F32)
    wsum = np.zeros((h, w), F32)
    best_w = np.zeros((h, w), F32)
    best_i = np.zeros((h, w), np.intp)
    tol = p.depth_tol
    for idx, _, _, wb in bilinear_taps(pos, w, h):
        g = gather(src.lin, idx)
        rel = np.abs(g - lin_c) / np.maximum(np.minimum(g, lin_c), 1e-7)
        wd = np.clip((2.0 * tol - rel) / max(tol, 1e-6), 0, 1)
        wt = (wb * wd).astype(F32)
        conf += wt  # the refresh stored confidence 1 everywhere
        ll += gather(src.log_luma, idx) * wt
        wsum += wt
        better = wt > best_w
        best_w = np.where(better, wt, best_w)
        best_i = np.where(better, idx, best_i)
    ok = wsum > 1e-4
    d = np.where(ok, wsum, 1.0)
    conf /= d
    ll /= d
    has = best_w > 0
    edit = np.where(has[..., None], gather(src.edit, best_i), 0)
    prt = np.where(has[..., None], gather(src.prt, best_i), 0)
    meta_y = has.astype(F32)  # validity stored at refresh: 1
    return dict(edit=edit.astype(F32), conf=conf, log_luma=ll, valid=np.clip(wsum, 0, 1), prt=prt, meta_y=meta_y)


def validate(hist: dict, dst: Frame, on: np.ndarray, p: Params) -> tuple:
    """Mode 1's tests against this hold. Returns (v_colour, validity)."""
    if p.noise_aware:
        diff = outside(hist["log_luma"], dst.prt_lo[..., 0], dst.prt_hi[..., 0])
    else:
        diff = np.abs(dst.log_luma - hist["log_luma"])
    v_colour = soft(diff, max(p.colour_tol, 1e-3))
    if not p.fingerprint:
        return v_colour, np.ones_like(v_colour)
    src = hist["prt"]
    m = soft(outside(src[..., 0], dst.prt_lo[..., 0], dst.prt_hi[..., 0]), p.print_tol)
    m = m * soft(np.maximum(outside(src[..., 1], dst.prt_lo[..., 1], dst.prt_hi[..., 1]),
                            outside(src[..., 2], dst.prt_lo[..., 2], dst.prt_hi[..., 2])), p.print_tol)
    validity = np.where(on, np.clip(hist["meta_y"], 0, 1) * m, 0)
    return v_colour, validity.astype(F32)


def reduce4(values: np.ndarray, weight: np.ndarray) -> tuple:
    """ReduceToFirstLevel / mode 3: coverage-weighted 4x4 means; coverage = sum(weight) / 16."""
    h, w = weight.shape
    H, W = (h + 3) // 4, (w + 3) // 4
    pw = np.zeros((H * 4, W * 4), F32)
    pw[:h, :w] = weight
    pv = np.zeros((H * 4, W * 4) + values.shape[2:], F32)
    pv[:h, :w] = values
    den = pw.reshape(H, 4, W, 4).sum(axis=(1, 3))
    num = (pv * pw.reshape(pw.shape + (1,) * (values.ndim - 2))).reshape((H, 4, W, 4) + values.shape[2:]).sum(axis=(1, 3))
    dd = den.reshape(den.shape + (1,) * (values.ndim - 2))
    mean = np.where(dd > 1e-6, num / np.maximum(dd, 1e-6), 0)
    return mean.astype(F32), (den / 16.0).astype(F32)


def build_levels(edit: np.ndarray, weight: np.ndarray, guide: np.ndarray) -> list:
    """L0 (1/4) from the full frame, L1 (1/16) and L2 (1/64) from the level above, each with its
    guide (log2 depth, log luma) carried coverage-weighted. Returns [(edit, coverage, guide)]."""
    levels = []
    vals = np.concatenate([edit, guide], -1)
    wgt = weight
    for _ in range(3):
        m, c = reduce4(vals, wgt)
        levels.append((m[..., :3], c, m[..., 3:5]))
        vals, wgt = m, c
    return levels


def low_band(levels: list, guide: np.ndarray, p: Params) -> np.ndarray:
    """Mode 4's low band: SurfaceLow (same-surface fill, fading to no edit) or the push-pull pyramid."""
    h, w = guide.shape[:2]
    log_d, log_l = guide[..., 0], guide[..., 1]

    def bilateral(lvl, sd, sl, use_guide=True):
        e, c, g = levels[lvl]
        lh, lw = c.shape
        acc = np.zeros((h, w, 3), F32)
        ws = np.zeros((h, w), F32)
        for idx, wb in level_taps(h, w, lh, lw):
            wt = wb * gather(c, idx)
            if use_guide:
                gt = gather(g, idx)
                dd = (gt[..., 0] - log_d) / sd
                dl = (gt[..., 1] - log_l) / sl
                wt = wt * np.exp(-(dd * dd + dl * dl))
            acc += gather(e, idx) * wt[..., None]
            ws += wt
        return acc, ws

    if p.surface_fill:
        low = np.zeros((h, w, 3), F32)
        have = np.zeros((h, w), F32)
        for lvl in (2, 1, 0):
            acc, ws = bilateral(lvl, p.sigma_depth * (1 + lvl), p.sigma_luma * (1 + lvl))
            a = np.clip(ws * 4.0, 0, 1)
            est = np.where(ws[..., None] > 1e-6, acc / np.maximum(ws, 1e-6)[..., None], low)
            low = low + (est - low) * a[..., None]
            have = have + (1 - have) * a
        return (low * have[..., None]).astype(F32)

    def fetch(lvl):
        acc, ws = bilateral(lvl, 1.0, 1.0, use_guide=False)
        return np.where(ws[..., None] > 1e-6, acc / np.maximum(ws, 1e-6)[..., None], 0), ws

    e3, w3 = fetch(2)
    e2, w2 = fetch(1)
    low = np.where(w3[..., None] > 1e-6, e3, 0)
    low = low + (e2 - low) * np.clip(w2 * 4, 0, 1)[..., None]
    acc1, w1 = bilateral(0, p.sigma_depth, p.sigma_luma, use_guide=p.bilateral)
    e1 = np.where(w1[..., None] > 1e-6, acc1 / np.maximum(w1, 1e-6)[..., None], low)
    low = low + (e1 - low) * np.clip(w1 * 4, 0, 1)[..., None]
    return low.astype(F32)


def carry(src: Frame, dst: Frame, pos: np.ndarray, on: np.ndarray, p: Params) -> dict:
    """The refresh hold's edit carried to dst at the chained positions, every mode at once.

    Returns the edits for "naive", "low" and "cache" (and the diagnostics a self-check reads)."""
    h, w = dst.lin.shape
    hist = read_history(src, pos, dst.lin, p)
    onf = on.astype(F32)
    for k in ("edit", "conf", "log_luma", "valid", "meta_y"):
        hist[k] = hist[k] * (onf[..., None] if hist[k].ndim == 3 else onf)
    v_colour, validity = validate(hist, dst, on, p)

    weight = hist["valid"] * validity
    doubt = (p.colour_floor + (1 - p.colour_floor) * v_colour) * np.clip(2 * hist["valid"] - 1, 0, 1) \
        * p.high_decay * validity
    conf = np.clip(hist["conf"] * doubt, 0, 1)

    guide = np.stack([np.log2(np.maximum(dst.lin, 1e-7)), dst.log_luma], -1).astype(F32)
    levels = build_levels(hist["edit"], weight, guide)
    low = low_band(levels, guide, p)
    scale = max(p.low_gain, p.high_gain)
    cache = clamp_edit(p.low_gain * low + p.high_gain * conf[..., None] * (hist["edit"] - low), p, scale)
    naive = sample_bilinear(src.edit, pos) * onf[..., None]
    return dict(naive=naive, low=clamp_edit(p.low_gain * low, p, scale), cache=cache,
                valid=hist["valid"], validity=validity, v_colour=v_colour, conf=conf, history=hist["edit"],
                low_band=low)


# --------------------------------------------------------------------------------------------------
# Metrics (the reference's measure_ghosting.score and measure_reprojection's kept)
# --------------------------------------------------------------------------------------------------

def pair_metrics(shown: np.ndarray, dst: Frame, p: Params, mask: np.ndarray | None = None) -> dict:
    """One hold's numbers for one shown edit. err/p99.5 in stops of luma against the model's edit for
    this hold; worse = share of area where the shown edit is worse than no edit by over 0.15 stop;
    pic = mean |display(shown) - display(model)| (kept is computed over holds from pic and base)."""
    e = np.abs((shown - dst.edit) @ LUMA)
    excess = np.maximum(0, e - np.abs(dst.edit @ LUMA))
    d = np.abs(display(apply_edit(dst.v_colour, shown, p)) - dst.disp_answer).mean(axis=-1)
    if mask is not None:
        e, excess, d = e[mask], excess[mask], d[mask]
    return dict(err=float(e.mean()), p995=float(np.percentile(e, 99.5)), worse=float((excess > 0.15).mean()),
                harm=float(excess.mean()), pic=float(d.mean()))


def base_metric(dst: Frame, mask: np.ndarray | None = None) -> float:
    d = np.abs(display(dst.v_colour) - dst.disp_answer).mean(axis=-1)
    return float(d[mask].mean()) if mask is not None else float(d.mean())
