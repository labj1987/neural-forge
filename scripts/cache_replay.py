#!/usr/bin/env python3
"""Replays the pre-upscaler edit cache offline on a K-hold dump, and measures what it keeps.

A cache dump (lead format v1: manifest.json plus raw colour/answer/mvec/depth per hold, written by
the layer with the model running on every hold) has the model's real answer for every hold. For
each lag L (1, 2, 3, 4, 8 by default) and every hold t >= L, the cache is refreshed at t-L (the edit
from that hold's colour and answer), carried to t along the motion chained through the holds
between (--chain recurrence, the default: the layer's accumulated motion field; --chain walk: each
older hold's vectors re-read at the running position), validated against hold t's colour and depth, and compared with the model's real answer
for t. The algorithm is scripts/cache_common.py, the reference the layer's
preupscale_cache*.comp shaders mirror.

Modes (all on the same holds):

  none    no edit at all: the frame without the model. The floor.
  naive   the whole edit reprojected (bilinear), no validation: what a simple cache would do.
  low     the cache's low band only (validated pyramid, same-surface fill), no high band.
  cache   the full algorithm: validated history, band split, high band by confidence.

Metrics (as upstream's measure_ghosting.py / measure_reprojection.py define them):

  err     mean |luma(shown edit) - luma(true edit)| in stops, per hold, averaged. Lower is better.
  p99.5   the per-hold 99.5th percentile of that, averaged: the strongest trails.
  worse   share of the area where the shown edit is worse than no edit by over 0.15 stop.
  kept    1 - mean|display(shown) - display(model)| / mean|display(original) - display(model)|,
          display = Reinhard on luma, then gamma 1/2.2. 100% = the model on every hold.

"true" is the model's edit for that hold, clamped as the cache clamps every edit; "display(model)"
is the model's real answer, so kept is below 100% where the clamp binds (the self-check reports
that ceiling).

Usage:
  cache_replay.py DUMP [--lags 1 2 3 4 8] [--json OUT] [--oracle] [parameters]
  cache_replay.py DUMP --cadence 2 3 4 [--adaptive] [--phase K] [--aging] [--no-guided] [--json OUT]
  cache_replay.py DUMP --cadence N --export-cadence DIR
  cache_replay.py DUMP --self-check
  cache_replay.py --make-synthetic DIR --kind still|pan|depth-edge|regrade|jitter
  cache_replay.py DUMP --export-carry DIR [--reverse-z yes|no]

Ported in part from optiscaler-dlss5 by Skynizz (https://github.com/Skynizz/optiscaler-dlss5),
commit 6bfc4389c25054ef6747df6e6b1b470e9541c4da: tools/dlssnr_cache/measure_reprojection.py,
measure_ghosting.py (the strategies and the metrics), measure_pulse.py (the regional light step)
and flicker_lab.py (the motion-compensated flicker). Upstream is GPL-3.0; this port is
distributed under this project's AGPL-3.0-or-later (see ATTRIBUTION.md).

Needs Python 3 and numpy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_common as cc  # noqa: E402

MODES = ("none", "naive", "low", "cache")
METRICS = ("err", "p995", "worse", "harm", "kept")


# --------------------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------------------

class Frames:
    """The dump's holds, prepared on demand, keeping only the few a chain still needs."""

    def __init__(self, dump: cc.Dump, reverse_z: bool, p: cc.Params, keep: int):
        self.dump, self.reverse_z, self.p, self.keep = dump, reverse_z, p, keep
        self.cache = {}

    def __getitem__(self, k: int) -> cc.Frame:
        if k not in self.cache:
            self.cache[k] = cc.prepare(self.dump.hold(k), self.dump.paper_white, self.reverse_z, self.p)
            for old in [j for j in self.cache if j < k - self.keep]:
                del self.cache[old]
        return self.cache[k]


def resolve_reverse_z(dump: cc.Dump, choice: str) -> tuple:
    """The depth convention: forced, from the manifest, or inferred from the first holds' depth."""
    if choice in ("yes", "no"):
        return choice == "yes", "forced on the command line"
    if dump.manifest.get("reverse_z") is not None:
        return bool(dump.manifest["reverse_z"]), "from the manifest"
    if not dump.has_depth:
        return True, "no depth in the dump (depth tests pass everywhere)"
    depths = [dump.hold(k).depth for k in range(min(3, len(dump)))]
    rev, why = cc.infer_reverse_z(depths)
    if rev is None:
        return True, f"inferred: unknown ({why}); assuming reversed-Z"
    return rev, f"inferred: {why}"


def shown_edits(src: cc.Frame, dst: cc.Frame, pos, on, p: cc.Params) -> tuple:
    out = cc.carry(src, dst, pos, on, p)
    shown = {"none": np.zeros_like(dst.edit), "naive": out["naive"], "low": out["low"], "cache": out["cache"]}
    return shown, out


def replay(dump: cc.Dump, lags, p: cc.Params, reverse_z: bool, oracle: bool = False, modes=MODES,
           log=None) -> dict:
    """Every lag x mode x metric. With oracle, each hold is refreshed from itself (no carry)."""
    lags = sorted(set(int(x) for x in lags))
    max_lag = max(lags)
    frames = Frames(dump, reverse_z, p, max_lag + 1)
    h, w = dump.height, dump.width
    centres = cc.pixel_centres(h, w)
    acc = {L: {m: {k: [] for k in ("err", "p995", "worse", "harm", "pic")} for m in modes} for L in lags}
    base = {L: [] for L in lags}
    skipped = 0

    for t in range(len(dump)):
        dst = frames[t]
        if not dst.usable:
            skipped += 1
            continue
        pos, on = centres, np.ones((h, w), bool)
        for L in range(0, min(max_lag, t) + 1):
            if L > 0 and not oracle and p.chain == "walk":
                pos = cc.chain_step(pos, frames[t - L + 1], frames[t - L], p)
                on = on & cc.on_screen(pos, h, w)
            if L not in lags:
                continue
            src = dst if oracle or L == 0 else frames[t - L]
            if not src.usable:
                continue
            # The layer refreshes on any hold it cannot cache (cc.can_carry): no carry over one.
            if L > 0 and not oracle and not all(cc.can_carry(src, frames[k]) for k in range(t - L + 1, t + 1)):
                continue
            if L > 0 and not oracle and p.chain == "recurrence":
                pos, on = carried_position(frames, t, L, p)
            if oracle or L == 0:
                shown, _ = shown_edits(src, dst, centres, np.ones((h, w), bool), p)
            else:
                shown, _ = shown_edits(src, dst, pos, on, p)
            base[L].append(dst.base)
            for m in modes:
                r = cc.pair_metrics(shown[m], dst, p)
                for k, v in r.items():
                    acc[L][m][k].append(v)
        if log:
            log(f"  hold {t + 1}/{len(dump)}")

    results = {}
    for L in lags:
        n = len(base[L])
        results[L] = {}
        for m in modes:
            a = acc[L][m]
            if n == 0:
                results[L][m] = {k: float("nan") for k in METRICS} | {"samples": 0}
                continue
            b = float(np.mean(base[L]))
            results[L][m] = {"err": float(np.mean(a["err"])), "p995": float(np.mean(a["p995"])),
                             "worse": 100 * float(np.mean(a["worse"])), "harm": 1000 * float(np.mean(a["harm"])),
                             "kept": 100 * (1 - float(np.mean(a["pic"])) / b) if b > 0 else float("nan"),
                             "samples": n}
    return {"results": results, "unusable_holds": skipped}


def carried_position(frames, t: int, lag: int, p: cc.Params) -> tuple:
    """The recurrence (cc.acc_step) from a refresh at t - lag up to hold t: (pos, on)."""
    acc = None
    for k in range(t - lag + 1, t + 1):
        acc = cc.acc_step(acc, frames[k], frames[k - 1].jitter, p)
    return cc.chained_position(acc, frames[t], frames[t - lag], p)


def export_carry(dump: cc.Dump, p: cc.Params, reverse_z: bool, out: str) -> int:
    """For the layer's shader test: the refresh at hold 0 carried to every later hold by the
    recurrence, as long as that carry lasts (the first hold the run refreshes on, or leaves out for an
    unusable exposure, ends it). Writes per hold t >= 1 carried to `cache-TT.f32` (H x W x 3, the shown
    edit: mode "cache" with the guided filter, as run_cadence shows it), `acc-TT.f32` (H x W x 2, the
    accumulated motion, NaN off the chain) and `weight-TT.f32` (H x W, valid x validity), and
    `holds.txt` (per hold: jitter x, jitter y, exposure), all f32 little-endian. Returns the number of
    holds carried to."""
    os.makedirs(out, exist_ok=True)
    frames = Frames(dump, reverse_z, p, len(dump))
    with open(os.path.join(out, "holds.txt"), "w", encoding="utf-8") as fh:
        for k in range(len(dump)):
            f = frames[k]
            fh.write(f"{float(f.jitter[0])!r} {float(f.jitter[1])!r} {float(dump.hold(k).exposure)!r}\n")
    src = frames[0]
    acc = None
    carried = 0
    for t, _, refresh, shown, dst, _ in run_cadence(dump, frames, Cadence(interval=len(dump) + 1), p):
        if t == 0:
            if not refresh:
                break
            continue
        # The run restarted (a refresh) or left a hold out: hold 0's carry ends here.
        if refresh or t != carried + 1:
            break
        acc = cc.acc_step(acc, dst, frames[t - 1].jitter, p)
        pos, on = cc.chained_position(acc, dst, src, p)
        r = cc.carry(src, dst, pos, on, p)
        shown.astype("<f4").tofile(os.path.join(out, f"cache-{t:02d}.f32"))
        acc.astype("<f4").tofile(os.path.join(out, f"acc-{t:02d}.f32"))
        (r["valid"] * r["validity"]).astype("<f4").tofile(os.path.join(out, f"weight-{t:02d}.f32"))
        carried = t
    return carried


def print_table(results: dict, out=None) -> None:
    out = out or sys.stdout
    print(f"  {'lag':>3}  {'mode':<6} {'err':>8} {'p99.5':>7} {'worse%':>7} {'harm':>7} {'kept%':>7} {'n':>4}", file=out)
    for L, per in results.items():
        for m, r in per.items():
            print(f"  {L:>3}  {m:<6} {r['err']:8.4f} {r['p995']:7.3f} {r['worse']:7.2f} {r['harm']:7.2f} "
                  f"{r['kept']:7.2f} {r['samples']:>4}", file=out)


# --------------------------------------------------------------------------------------------------
# Cadence: the cache as it runs, one refresh every N holds, and what it does over time
# --------------------------------------------------------------------------------------------------

@dataclass
class Cadence:
    """How a run at a fixed interval shows its edits. The defaults are the layer's (the shaders mirror
    run_cadence with them): each refresh shows the model's answer at once, the guided filter on.
    Aging is an experiment, measured and not in the shaders; so was the walk to each refresh's answer
    (upstream's keyframe crossfade with soft refresh), removed after the live runs (docs/LESSONS.md,
    "3.3 edit cache: what was tried").

    adaptive    the layer's cadence: `interval` while the camera moves, `still_interval` while it is
                still (Regime);
    aging       the high band's confidence times 2^(-staleness / half_life), staleness = one per
                stale_px of motion since the refresh;
    guided      a fast guided filter (He et al.) of the cached edit on this hold's log luma, blended in
                by guided_strength x sat(staleness / 2) (staleness: as for aging; the motion pass
                accumulates it on the GPU)."""

    interval: int
    phase: int = 0
    adaptive: bool = False
    still_interval: int = 1
    aging: bool = False
    half_life: float = 8.0
    guided: bool = True
    guided_strength: float = 1.0
    guided_eps: float = 0.005
    stale_px: float = 4.0

    def label(self) -> str:
        parts = [f"i{self.interval}"]
        if self.adaptive:
            parts.append(f"adaptive{self.still_interval}" if self.still_interval != 1 else "adaptive")
        if self.aging:
            parts.append("aging")
        if not self.guided:
            parts.append("noguided")
        return "-".join(parts) if len(parts) > 1 else parts[0] + "-base"


class Regime:
    """The layer's adaptive cadence (`cache::Regime`, which mirrors this): a hold reads as still when
    its median pixel moves less than a quarter of the pace, max(0.1% of the render width, 0.5 px) / 3
    a hold (upstream's speed reference at motion priority 0.5). A regime change waits for consecutive
    holds that want it: 3 toward the shorter interval, 20 toward the longer, so the cadence never
    hops. The motion pass counts the pixels (`still_counts`) and a hold's counts decide the holds
    after it."""

    TO_SHORTER = 3
    TO_LONGER = 20

    def __init__(self, cad: "Cadence"):
        self.cad, self.still, self.against = cad, False, 0

    def interval(self) -> int:
        return self.cad.still_interval if self.still else self.cad.interval

    def observe(self, counts: tuple) -> None:
        still, n = counts
        if n == 0:
            return
        wants = 2 * still > n
        if wants == self.still:
            self.against = 0
            return
        self.against += 1
        shorter = (self.cad.still_interval < self.cad.interval) == wants
        if self.against >= (self.TO_SHORTER if shorter else self.TO_LONGER):
            self.still, self.against = wants, 0


def still_counts(mv: np.ndarray) -> tuple:
    """The motion pass's STAT_STILL and STAT_MOTION_N: pixels whose vector (the nearest surface's,
    render pixels) is shorter than a quarter of the pace, and all of them."""
    h, w = mv.shape[:2]
    pace = np.float32(max(0.001 * w, 0.5) / 3.0)
    s = np.sqrt((mv.astype(np.float32) ** 2).sum(axis=-1))
    return int((s < np.float32(0.25) * pace).sum()), h * w


def step_position(prev: cc.Frame, cur: cc.Frame, p: cc.Params) -> np.ndarray:
    """Where each pixel of `cur` was on the hold before it (pixel units): its vector plus the change of
    jitter."""
    h, w = cur.lin.shape
    pos = cc.pixel_centres(h, w) + cur.mv
    if p.jitter:
        pos = pos + (prev.jitter - cur.jitter)
    return pos.astype(np.float32)


def box3(x: np.ndarray) -> np.ndarray:
    """3x3 mean, edge-clamped (scipy's uniform_filter(size 3, mode nearest))."""
    return cc.neighbourhood(x).mean(axis=0)


def guided_fast(I: np.ndarray, P: np.ndarray, eps: float) -> np.ndarray:
    """Upstream's modes 13-15 (measure_ghosting.guided_fast): 2x2 means at half size, a 3x3 window
    there, a and b tent-blurred and read back bilinearly, rebuilt = a I + b, per channel."""
    h, w = I.shape
    H, W = (h + 1) // 2, (w + 1) // 2

    def half(x):
        x = np.pad(x, [(0, H * 2 - h), (0, W * 2 - w)], mode="edge")
        return x.reshape(H, 2, W, 2).mean(axis=(1, 3))

    def tent(x):
        p = np.pad(x, 1, mode="edge")
        x = 0.25 * p[:-2, 1:-1] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[2:, 1:-1]
        p = np.pad(x, 1, mode="edge")
        return 0.25 * p[1:-1, :-2] + 0.5 * p[1:-1, 1:-1] + 0.25 * p[1:-1, 2:]

    mI = box3(half(I))
    var = np.maximum(box3(half(I * I)) - mI * mI, 0)
    centres = (cc.pixel_centres(h, w) * np.array([W / w, H / h], np.float32)).astype(np.float32)
    out = np.empty_like(P)
    for ch in range(3):
        mp = box3(half(P[..., ch]))
        a = (box3(half(I * P[..., ch])) - mI * mp) / (var + eps)
        b = mp - a * mI
        out[..., ch] = cc.sample_bilinear(tent(a).astype(np.float32), centres) * I + \
            cc.sample_bilinear(tent(b).astype(np.float32), centres)
    return out


def run_cadence(dump: cc.Dump, frames: Frames, cad: Cadence, p: cc.Params):
    """Holds phase.. with the model on every interval-th (adaptive: the regime's interval, decided by
    the holds before): yields (hold, since, shown edit, frame). A hold without a usable exposure
    restarts the run (the next hold refreshes)."""
    regime = Regime(cad) if cad.adaptive else None
    src = acc = prev = stale = None
    since = 0
    for t in range(cad.phase, len(dump)):
        dst = frames[t]
        n = regime.interval() if regime else cad.interval
        # The motion pass counts nothing on a hold without vectors (F_NO_MVEC).
        if regime and dst.mvec_valid:
            regime.observe(still_counts(dst.mv))
        if not dst.usable:
            src = prev = None
            continue
        consecutive = prev is not None and prev.index == dst.index - 1
        # Any hold the layer cannot cache refreshes (cc.can_carry: jitter, vectors, depth presence).
        refresh = src is None or not consecutive or since + 1 >= n or not cc.can_carry(src, dst)
        step = step_position(prev, dst, p) if consecutive else None
        if refresh:
            # A refresh without a known jitter stores no edit: the next hold refreshes too.
            src, acc, since = (dst if cc.can_source(dst) else None), None, 0
            stale = np.zeros(dst.lin.shape, np.float32)
            shown = dst.edit
        else:
            since += 1
            acc = cc.acc_step(acc, dst, prev.jitter, p)
            pos, on = cc.chained_position(acc, dst, src, p)
            r = cc.carry(src, dst, pos, on, p)
            moved = np.linalg.norm(step - cc.pixel_centres(*dst.lin.shape), axis=-1)
            stale = cc.sample_bilinear(stale, step) + np.clip(moved / cad.stale_px, 0, 1)
            shown = r["cache"]
            if cad.aging or cad.guided:
                low, conf = r["low_band"], r["conf"]
                if cad.aging:
                    conf = conf * np.exp2(-stale / cad.half_life)
                shown = cc.clamp_edit(low + conf[..., None] * (r["history"] - low), p)
                if cad.guided:
                    q = guided_fast(dst.log_luma, shown, cad.guided_eps)
                    wgt = cad.guided_strength * np.clip(stale / 2.0, 0, 1)
                    shown = cc.clamp_edit(shown + (q - shown) * wgt[..., None], p)
        yield t, since, refresh, shown.astype(np.float32), dst, step
        prev = dst


def export_cadence(dump: cc.Dump, cad: Cadence, p: cc.Params, reverse_z: bool, out: str) -> int:
    """For the layer's shader test: every hold's shown edit of a cadence run, `shown-TT.f32` (H x W x 3
    f32 little-endian), and `holds.txt` (per hold: jitter x, jitter y, and the motion pass's
    still_counts). Returns the holds written."""
    os.makedirs(out, exist_ok=True)
    frames = Frames(dump, reverse_z, p, 3)
    k = 0
    with open(os.path.join(out, "holds.txt"), "w", encoding="utf-8") as fh:
        for t, _, _, shown, dst, _ in run_cadence(dump, frames, cad, p):
            still, n = still_counts(dst.mv)
            fh.write(f"{float(dst.jitter[0])!r} {float(dst.jitter[1])!r} {still} {n}\n")
            shown.astype("<f4").tofile(os.path.join(out, f"shown-{t:02d}.f32"))
            k += 1
    return k


def half_res(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:h, :w]
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    p = np.pad(x, ((0, 0), (r, r)), mode="edge")
    x = sum(k[i] * p[:, i:i + x.shape[1]] for i in range(2 * r + 1))
    p = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + x.shape[0], :] for i in range(2 * r + 1)).astype(np.float32)


def regional_light(dst: cc.Frame, edit: np.ndarray, p: cc.Params) -> np.ndarray:
    """measure_pulse.py's regional light: at half size, the shown picture's log2 luma ratio to the
    untouched frame, Gaussian sigma 6 (12 px at full size)."""
    y_orig = half_res(cc.luma(dst.v_colour))
    y_shown = half_res(cc.luma(cc.apply_edit(dst.v_colour, edit, p)))
    return gaussian(np.log2((y_shown + p.eps_v) / (y_orig + p.eps_v)).astype(np.float32), 6.0)


def regional_step(now: np.ndarray, before: np.ndarray, step: np.ndarray) -> np.ndarray:
    """The regional light's change from the hold before, moved here along the motion (half size)."""
    pos = np.stack([half_res(step[..., 0]), half_res(step[..., 1])], -1) * 0.5
    return now - cc.sample_bilinear(before, pos.astype(np.float32))


def cadence_metrics(dump: cc.Dump, cad: Cadence, p: cc.Params, reverse_z: bool) -> dict:
    """One cadence run's numbers.

    err/p99.5/worse/kept: as the lag replay, against the model's answer on every hold after the first
    refresh (the refresh holds included); *_cached on the cached holds only.
    pulse: measure_pulse.py's mean |regional light step| (stops) by holds since the refresh (0 = the
    refresh hold), the model's own (its answer on every hold) beside it; pulse_excess = the refresh
    holds' mean minus the cached holds'.
    flicker: flicker_lab.py's motion-compensated |picture_t - picture_t-1| (display values) on pixels
    that stayed on their surface, minus the model's own, x1000.
    refresh_share: the holds the model ran on, % of all."""
    frames = Frames(dump, reverse_z, p, 3)
    acc = {k: [] for k in ("err", "p995", "worse", "pic", "base")}
    cached = {k: [] for k in ("err", "p995", "worse", "pic", "base")}
    pulse, model_pulse, flick = {}, [], []
    prev = None
    holds = refreshes = 0
    for t, since, refresh, shown, dst, step in run_cadence(dump, frames, cad, p):
        holds += 1
        refreshes += int(refresh)
        reg = regional_light(dst, shown, p)
        reg_true = regional_light(dst, dst.edit, p)
        pic = cc.display(cc.apply_edit(dst.v_colour, shown, p))
        if prev is not None and step is not None and prev[0] == t - 1:
            r = cc.pair_metrics(shown, dst, p)
            for k in ("err", "p995", "worse", "pic"):
                acc[k].append(r[k])
                if not refresh:
                    cached[k].append(r[k])
            acc["base"].append(dst.base)
            if not refresh:
                cached["base"].append(dst.base)
            s = float(np.abs(regional_step(reg, prev[1], step)).mean())
            pulse.setdefault(0 if refresh else since, []).append(s)
            model_pulse.append(float(np.abs(regional_step(reg_true, prev[2], step)).mean()))
            lin_then = cc.sample_bilinear(prev[5], step)
            same = np.abs(np.log2(np.maximum(lin_then, 1e-6)) - np.log2(np.maximum(dst.lin, 1e-6))) < 0.1
            dc = np.abs(pic - cc.sample_bilinear(prev[3], step)).mean(axis=-1)
            dv = np.abs(dst.disp_answer - cc.sample_bilinear(prev[4], step)).mean(axis=-1)
            flick.append(float((dc - dv)[same].mean()) if same.any() else 0.0)
        prev = (t, reg, reg_true, pic, dst.disp_answer, dst.lin)

    def summary(a):
        if not a["err"]:
            return {}
        return {"err": float(np.mean(a["err"])), "p995": float(np.mean(a["p995"])),
                "worse": 100 * float(np.mean(a["worse"])),
                "kept": 100 * (1 - float(np.mean(a["pic"])) / float(np.mean(a["base"]))), "samples": len(a["err"])}

    by_since = {str(k): float(np.mean(v)) for k, v in sorted(pulse.items())}
    others = [x for k, v in pulse.items() if k != 0 for x in v]
    return {"label": cad.label(), "cadence": dataclasses.asdict(cad), "all": summary(acc), "cached": summary(cached),
            "pulse": by_since, "pulse_model": float(np.mean(model_pulse)) if model_pulse else float("nan"),
            "pulse_excess": float(np.mean(pulse.get(0, [np.nan])) - np.mean(others)) if others else float("nan"),
            "flicker": 1000 * float(np.mean(flick)) if flick else float("nan"),
            "refresh_share": 100 * refreshes / max(holds, 1)}


def print_cadence(rows: list, out=None) -> None:
    out = out or sys.stdout
    print(f"  {'run':<20} {'err':>7} {'p99.5':>6} {'worse%':>6} {'kept%':>6} | {'cached err':>10} {'p99.5':>6} | "
          f"{'pulse run/others (model)':>26} {'excess':>8} | {'flicker':>7} | {'model%':>6}", file=out)
    for r in rows:
        a, c = r["all"], r["cached"] or {"err": float("nan"), "p995": float("nan")}
        pz = r["pulse"]
        others = [v for k, v in pz.items() if k != "0"]
        print(f"  {r['label']:<20} {a['err']:7.4f} {a['p995']:6.3f} {a['worse']:6.2f} {a['kept']:6.1f} | {c['err']:10.4f} "
              f"{c['p995']:6.3f} | {pz.get('0', float('nan')):8.4f} / {np.mean(others) if others else float('nan'):6.4f} "
              f"({r['pulse_model']:.4f}) {r['pulse_excess']:+8.4f} | {r['flicker']:7.3f} | {r['refresh_share']:6.1f}", file=out)


# --------------------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------------------

def self_check(dump: cc.Dump, p: cc.Params, reverse_z: bool, out=None) -> bool:
    """Lag 0 is the identity for every mode's carry, and a lag-1 replay refreshed at the hold itself
    (--oracle, clamp off) keeps 100% of the effect with zero error."""
    out = out or sys.stdout
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}", file=out)

    frames = Frames(dump, reverse_z, p, 2)
    usable = [k for k in range(len(dump)) if frames[k].usable]
    check("usable holds", len(usable) >= 2, f"{len(usable)} of {len(dump)} have a usable exposure")
    if len(usable) < 2:
        return False

    h, w = dump.height, dump.width
    centres = cc.pixel_centres(h, w)
    for k in usable[:3]:
        f = frames[k]
        shown, d = shown_edits(f, f, centres, np.ones((h, w), bool), p)
        dev = max(float(np.abs(d[x] - 1).max()) for x in ("valid", "validity", "v_colour", "conf"))
        check(f"lag 0 hold {k}: carry admits everything", dev < 1e-5, f"max |weight - 1| {dev:.2e}")
        dh = float(np.abs(d["history"] - f.edit).max())
        check(f"lag 0 hold {k}: history read is the edit", dh < 1e-5, f"max diff {dh:.2e} stop")
        dn = float(np.abs(shown["naive"] - f.edit).max())
        check(f"lag 0 hold {k}: naive is the edit", dn < 1e-5, f"max diff {dn:.2e} stop")
        dc = float(np.abs(shown["cache"] - cc.clamp_edit(f.edit, p)).max())
        check(f"lag 0 hold {k}: cache is the edit", dc < 1e-4, f"max diff {dc:.2e} stop")
        guide = np.stack([np.log2(np.maximum(f.lin, 1e-7)), f.log_luma], -1).astype(np.float32)
        lo = cc.low_band(cc.build_levels(f.edit, np.ones((h, w), np.float32), guide), guide, p)
        dl = float(np.abs(shown["low"] - cc.clamp_edit(lo, p)).max())
        check(f"lag 0 hold {k}: low is the edit's low band", dl < 1e-5, f"max diff {dl:.2e} stop")
        check(f"lag 0 hold {k}: none is no edit", not shown["none"].any(), "all zero")

    raw = cc.Params(**{**p.as_dict(), "max_luma_edit": 0.0})
    r = replay(dump, [1], raw, reverse_z, oracle=True, modes=("none", "naive", "cache"))["results"][1]
    for m in ("naive", "cache"):
        check(f"lag 1 oracle {m}", r[m]["err"] < 1e-4 and r[m]["kept"] > 99.99,
              f"err {r[m]['err']:.2e} stop, kept {r[m]['kept']:.4f}% over {r[m]['samples']} holds")
    ceiling = replay(dump, [1], p, reverse_z, oracle=True, modes=("none", "cache"))["results"][1]["cache"]["kept"]
    print(f"  info  with the edit clamp at {p.max_luma_edit} stop, the oracle keeps {ceiling:.2f}% "
          "(the clamp's own ceiling)", file=out)
    return ok


# --------------------------------------------------------------------------------------------------
# Synthetic dumps
# --------------------------------------------------------------------------------------------------

def synthetic_holds(kind: str, w: int = 96, h: int = 64, n: int = 6) -> tuple:
    """Holds with known motion and a known "model" edit. Returns (holds, paper_white, info).

    still       nothing moves.
    pan         the camera pans 2.5 px right and 1 px down per hold, jitter +-0.25 px: with the jitter
                change the motion is a whole number of pixels, so a correct carry is exact.
    depth-edge  a near box moves 5 px right per hold over a still far background (sky on top); where
                it uncovers the background the carried high band must be rejected.
    regrade     the pan, with the model re-grading the whole picture 0.15 stop brighter each hold:
                every refresh's answer differs from the last.
    jitter      nothing moves but the jitter (sub-pixel, a different offset each hold), and the edit
                has a near-pixel-scale term: the history read must carry it whole."""
    paper_white, exposure = 3.0, 1.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs, ys = xs + 0.5, ys + 0.5

    def scene(X, Y, obj):
        """v of the colour and the model's edit at world (X, Y); obj picks the box's look."""
        detail = 0.25 * np.sin(1.3 * X) * np.sin(1.1 * Y)
        if obj:
            base = np.stack([0.45 + 0.1 * np.sin(X / 5), 0.25 + 0.05 * np.cos(Y / 4), 0.12 + 0 * X], -1)
            el = 0.7 + 0.2 * np.sin(0.9 * X + 1) * np.cos(0.8 * Y)
            chroma = np.stack([0.1 + 0 * X, 0 * X, -0.1 + 0 * X], -1)
        else:
            base = np.stack([0.20 + 0.08 * np.sin(X / 13), 0.28 + 0.08 * np.sin(Y / 11), 0.30 + 0.05 * np.cos((X + Y) / 17)], -1)
            el = -0.4 + 0.4 * np.sin(X / 17 + Y / 23) + 0.25 * np.sin(0.9 * X + 1) * np.cos(0.7 * Y)
            chroma = np.stack([-0.05 + 0 * X, 0.05 * np.sin(X / 9), 0.15 + 0 * X], -1)
        v = base * np.exp2(detail)[..., None]
        return v, el[..., None] + chroma - (chroma @ cc.LUMA.astype(np.float64))[..., None]

    holds, info = [], {"kind": kind}
    for t in range(n):
        if kind in ("pan", "regrade"):
            cam, jit = np.array([2.5 * t, 1.0 * t]), np.array([0.25 if t % 2 == 0 else -0.25, 0.0])
        elif kind == "still":
            cam, jit = np.zeros(2), np.zeros(2)
        elif kind == "depth-edge":
            cam, jit = np.zeros(2), np.zeros(2)
        elif kind == "jitter":
            cam = np.zeros(2)
            jit = np.array([[0.0, 0.0], [0.3, -0.2], [-0.35, 0.15], [0.1, 0.4], [-0.2, -0.3], [0.45, 0.05]][t % 6])
        else:
            raise ValueError(f"unknown synthetic kind {kind!r}")
        X, Y = xs - jit[0] + cam[0], ys - jit[1] + cam[1]
        v, edit = scene(X, Y, False)
        depth = 0.002 + 0.03 * np.clip(Y, 0, None) / h  # reversed-Z: the bottom is nearer
        mv = np.zeros((h, w, 2))
        if kind in ("pan", "regrade") and t > 0:
            mv[...] = [2.5, 1.0]
        if kind == "regrade":
            edit = edit + 0.15 * t
        if kind == "jitter":
            edit = edit + 0.3 * (np.cos(2.9 * X) * np.cos(2.7 * Y))[..., None]
        if kind == "depth-edge":
            sky = ys < 8
            v[sky] = [0.3, 0.45, 0.8]
            edit[sky] = [0.0, 0.1, 0.3]
            depth = np.where(sky, 0.0, 0.01 + 0.01 * ys / h)
            x0 = 10 + 5 * t
            box = (xs >= x0) & (xs < x0 + 20) & (ys >= 20) & (ys < 44)
            vb, eb = scene(xs - x0, ys - 20, True)
            v[box], edit[box], depth = vb[box], eb[box], np.where(box, 0.4, depth)
            if t > 0:
                mv[box] = [-5.0, 0.0]
            info.setdefault("box_x", []).append(x0)
            info["box_rows"] = [20, 44]
            info["box_width"] = 20
        answer = np.maximum((v + cc.EPS_V) * np.exp2(edit) - cc.EPS_V, 0)
        to_scene = paper_white / exposure
        holds.append(cc.Hold(index=t, jitter=jit.astype(np.float32), exposure=exposure,
                             colour=(v * to_scene).astype(np.float32), answer=(answer * to_scene).astype(np.float32),
                             mvec=mv.astype(np.float32), depth=depth.astype(np.float32)))
    return holds, paper_white, info


def make_synthetic(path: str, kind: str) -> dict:
    holds, white, info = synthetic_holds(kind)
    cc.write_dump(path, holds, paper_white=white, game=f"synthetic-{kind}", extra={"synthetic": info})
    return info


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------

def params_from(args) -> cc.Params:
    return cc.Params(eps_v=args.eps_v, max_luma_edit=args.max_luma_edit, chroma_stops=args.chroma_stops,
                     depth_tol=args.depth_tol, colour_tol=args.colour_tol, noise_aware=not args.no_noise_aware,
                     colour_floor=args.colour_floor, high_decay=args.high_decay, fingerprint=not args.no_fingerprint,
                     print_tol=args.print_tol, surface_fill=not args.no_surface_fill, bilateral=not args.no_bilateral,
                     sigma_depth=args.sigma_depth, sigma_luma=args.sigma_luma,
                     nearest_surface=not args.no_nearest_surface, jitter=not args.no_jitter,
                     chain_sample=args.chain_sample, chain=args.chain, low_gain=args.low_gain, high_gain=args.high_gain)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", help="the dump directory (manifest.json)")
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--oracle", action="store_true", help="refresh every hold from itself (no carry)")
    ap.add_argument("--json", metavar="OUT", help="also write the results as JSON")
    ap.add_argument("--self-check", action="store_true", help="lag-0 identity and the lag-1 oracle; non-zero exit on failure")
    ap.add_argument("--reverse-z", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--make-synthetic", metavar="DIR", help="write a synthetic dump to DIR and exit")
    ap.add_argument("--kind", choices=("still", "pan", "depth-edge", "regrade", "jitter"), default="pan")
    ap.add_argument("--export-carry", metavar="DIR", help="write the refresh at hold 0 carried to every later hold (the layer's shader test) and exit")
    ap.add_argument("--quiet", action="store_true")
    c = ap.add_argument_group("cadence: the cache run at a fixed interval (pulse, flicker; experiments not in the shaders)")
    c.add_argument("--cadence", type=int, nargs="+", metavar="N", help="run with the model every N holds instead of the lag replay")
    c.add_argument("--phase", type=int, default=0, help="the first refresh hold")
    c.add_argument("--adaptive", action="store_true", help="the layer's adaptive cadence: N while the camera moves")
    c.add_argument("--still-interval", type=int, default=1, help="(experiment) with --adaptive: the interval while still; the layer uses 1")
    c.add_argument("--aging", action="store_true")
    c.add_argument("--half-life", type=float, default=8.0)
    c.add_argument("--no-guided", action="store_true", help="no guided filter of the cached edits")
    c.add_argument("--guided-strength", type=float, default=1.0, help="(experiment) the shaders use 1")
    c.add_argument("--guided-eps", type=float, default=0.005)
    c.add_argument("--export-cadence", metavar="DIR", help="with one --cadence N: write every hold's shown edit (the layer's shader test) and exit")
    g = ap.add_argument_group("cache parameters (defaults are the shader's)")
    d = cc.Params()
    g.add_argument("--eps-v", type=float, default=d.eps_v)
    g.add_argument("--max-luma-edit", type=float, default=d.max_luma_edit, help="0 = no edit clamp")
    g.add_argument("--chroma-stops", type=float, default=d.chroma_stops)
    g.add_argument("--depth-tol", type=float, default=d.depth_tol)
    g.add_argument("--colour-tol", type=float, default=d.colour_tol)
    g.add_argument("--no-noise-aware", action="store_true")
    g.add_argument("--colour-floor", type=float, default=d.colour_floor)
    g.add_argument("--high-decay", type=float, default=d.high_decay)
    g.add_argument("--no-fingerprint", action="store_true")
    g.add_argument("--print-tol", type=float, default=d.print_tol)
    g.add_argument("--no-surface-fill", action="store_true")
    g.add_argument("--no-bilateral", action="store_true")
    g.add_argument("--sigma-depth", type=float, default=d.sigma_depth)
    g.add_argument("--sigma-luma", type=float, default=d.sigma_luma)
    g.add_argument("--no-nearest-surface", action="store_true")
    g.add_argument("--no-jitter", action="store_true")
    g.add_argument("--chain", choices=("recurrence", "walk"), default=d.chain,
                   help="recurrence: the layer's accumulated motion field (default); walk: each older hold's vectors re-read at the running position")
    g.add_argument("--chain-sample", choices=("bilinear", "nearest"), default=d.chain_sample, help="walk only")
    g.add_argument("--low-gain", type=float, default=d.low_gain)
    g.add_argument("--high-gain", type=float, default=d.high_gain)
    args = ap.parse_args(argv)

    if args.make_synthetic:
        info = make_synthetic(args.make_synthetic, args.kind)
        print(f"wrote a synthetic {args.kind} dump to {args.make_synthetic} ({json.dumps(info)})")
        return 0
    if not args.dump:
        ap.error("a dump directory is needed (or --make-synthetic)")
    if min(args.lags) < 0:
        ap.error("lags are 0 or more")

    p = params_from(args)
    dump = cc.load_dump(args.dump)
    rev, why = resolve_reverse_z(dump, args.reverse_z)
    print(f"{args.dump}: {len(dump)} holds, {dump.width}x{dump.height}, game {dump.manifest.get('game')}, "
          f"depth {dump.manifest.get('depth_format')}")
    print(f"reverse_z: {'yes' if rev else 'no'} ({why})")

    if args.export_carry:
        n = export_carry(dump, p, rev, args.export_carry)
        print(f"wrote {n} carried hold(s) to {args.export_carry}")
        return 0

    if args.self_check:
        ok = self_check(dump, p, rev)
        print("self-check " + ("passed" if ok else "FAILED"))
        return 0 if ok else 1

    if args.cadence:
        if min(args.cadence) < 2:
            ap.error("--cadence intervals are 2 or more")
        def cadence(n):
            return Cadence(interval=n, phase=args.phase, adaptive=args.adaptive, still_interval=args.still_interval, aging=args.aging, half_life=args.half_life, guided=not args.no_guided,
                           guided_strength=args.guided_strength, guided_eps=args.guided_eps)

        if args.export_cadence:
            if len(args.cadence) != 1:
                ap.error("--export-cadence takes one --cadence interval")
            k = export_cadence(dump, cadence(args.cadence[0]), p, rev, args.export_cadence)
            print(f"wrote {k} hold(s) to {args.export_cadence}")
            return 0
        t0 = time.time()
        rows = [cadence_metrics(dump, cadence(n), p, rev) for n in args.cadence]
        print(f"cadence, {time.time() - t0:.0f} s")
        print_cadence(rows)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump({"dump": os.path.abspath(args.dump), "params": p.as_dict(), "runs": rows}, fh, indent=1, allow_nan=True)
            print(f"wrote {args.json}")
        return 0

    t0 = time.time()
    log = None if args.quiet else (lambda s: print(s, file=sys.stderr, flush=True))
    out = replay(dump, args.lags, p, rev, oracle=args.oracle, modes=args.modes, log=log)
    if out["unusable_holds"]:
        print(f"{out['unusable_holds']} hold(s) had no usable exposure and were left out")
    print(f"{'oracle (no carry)' if args.oracle else 'carried'}, {time.time() - t0:.0f} s")
    print_table(out["results"])

    if args.json:
        doc = {"dump": os.path.abspath(args.dump), "holds": len(dump), "width": dump.width, "height": dump.height,
               "reverse_z": rev, "reverse_z_reason": why, "oracle": args.oracle, "params": p.as_dict(),
               "unusable_holds": out["unusable_holds"],
               "results": {str(L): per for L, per in out["results"].items()}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, allow_nan=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
