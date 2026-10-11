#!/usr/bin/env python3
"""Flicker and added detail of gta-bench.sh runs, from their presented-frame series.

usage: bench-iq.py [--baseline MODEL_OFF_RUN_DIR] [--frames N] [--start K] [--scan] RUN_DIR...
       bench-iq.py --self-test

A run's series is the `series-<ms>/` directory gta-bench.sh moves into the run directory: N
consecutive presented frames taken with NEURAL_FORGE_CAPTURE_AT=<resume>:<ms>:<frames> (the
layer's frame-series capture, docs/RUNNING_AND_MEASURING.md). The camera must be still there; what
moves anyway (people walking, particles) is what the exclusions below are for. Every number is
taken on the frames as presented (8-bit sRGB PNGs, linearised), with Y the Rec.709 luma.

Flicker: per pixel d = |log2(Y_t + f) - log2(Y_t-1 + f)| between consecutive frames, f = 0.05 x the
geometric mean of the first frame's Y; the mean of the smallest 95% of d (the 5% that change most
are moving objects), averaged over the pairs of the window (6 frames: 5 pairs), shown as
100 x (2^mean - 1) %. The same measure as the reference's live benchmark.

Added detail (with --baseline, a run with the layer loaded and the model off, captured at the same
moment): the mean energy of the 4-neighbour Laplacian of log2(Y + f) over the static pixels of the
window, with the model divided by without it (x1.15 = 15% more local contrast). Static: the pixel's
and its four neighbours' largest frame-to-frame change within the window is under STATIC_STOPS in
both runs. f is the
baseline's for both, so the two logs are on one scale.

The window is the 6 frames from SETTLE frames after the series' first cut (`anchor`; the first
frames when there is no cut): GTA's camera starts a pass up to about a second earlier or later from
run to run, so the series is captured across the cut and the cut fixes the moment. --start K / --frames N choose the window by hand.
--scan prints the mean frame-to-frame change of every pair and the stillest N-frame window, for
choosing the capture moment. With several RUN_DIRs the spread of their flicker (max / min - 1) is
printed too: the stability gate for repeated model-off runs is a spread of 10% or less.
"""
import glob
import os
import re
import sys

WINDOW = 6
EXCLUDE = 0.05  # the share of pixels with the largest change left out of the flicker mean
FLOOR = 0.05  # x geometric mean luma
STATIC_STOPS = 0.05
# A pair whose trimmed change is over this (stops) can be a cut, and the window starts SETTLE frames
# after the first one (see `anchor`).
CUT_STOPS = 0.05
SETTLE = 40  # largest frame-to-frame change (stops) of a pixel counted as static for detail
REC709 = (0.2126, 0.7152, 0.0722)
# The darkest non-black 8-bit sRGB code, linear: black pixels count as this in the geometric mean.
DARKEST = 1 / 255 / 12.92


def np():
    import numpy

    return numpy


def series_dir(run):
    """The run's series directory: `series-*` inside it, else the one its launch log names."""
    dirs = sorted(d for d in glob.glob(os.path.join(run, "series-*")) if os.path.isdir(d))
    if dirs:
        return dirs[0]
    if os.path.isdir(run) and os.path.basename(os.path.normpath(run)).startswith("series-"):
        return run
    log = os.path.join(run, "launch.log")
    if os.path.exists(log):
        m = re.findall(r"\[series\] capturing the next \d+ presents into (\S+)", open(log, errors="replace").read())
        if m and os.path.isdir(m[-1]):
            return m[-1]
    return None


def frame_files(directory):
    """The series' presented frames in order. On either present path `composited` is what was
    presented (the same as `original` when nothing was composited after the upscaler)."""
    return sorted(glob.glob(os.path.join(directory, "*-composited.png")))


def srgb_to_linear(v):
    n = np()
    return n.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_luma(path):
    from PIL import Image

    n = np()
    rgb = n.asarray(Image.open(path).convert("RGB"), dtype=n.float32) / n.float32(255)
    return srgb_to_linear(rgb) @ n.array(REC709, dtype=n.float32)


def floor_of(y):
    n = np()
    return FLOOR * float(n.exp(n.mean(n.log(n.maximum(y, DARKEST)))))


def log_frames(lumas, floor):
    n = np()
    return [n.log2(y + floor) for y in lumas]


def pair_changes(logs):
    n = np()
    return [n.abs(b - a) for a, b in zip(logs, logs[1:])]


def trimmed_mean(d):
    """Mean of the smallest (1 - EXCLUDE) of `d`."""
    n = np()
    flat = d.ravel()
    keep = int(len(flat) * (1 - EXCLUDE))
    return float(n.partition(flat, keep - 1)[:keep].mean()) if keep else 0.0


def percent(stops):
    return 100.0 * (2.0**stops - 1.0)


def flicker(lumas):
    """Flicker of consecutive frames' luma, in stops (`percent` converts)."""
    logs = log_frames(lumas, floor_of(lumas[0]))
    pairs = pair_changes(logs)
    return sum(trimmed_mean(d) for d in pairs) / len(pairs)


def laplacian(l):
    n = np()
    lap = n.zeros_like(l)
    lap[1:-1, 1:-1] = 4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]
    return lap


def static_mask(logs):
    """Pixels whose largest frame-to-frame change is under STATIC_STOPS, together with their four
    neighbours (the Laplacian reads them), the border left out."""
    n = np()
    still = n.max(n.stack(pair_changes(logs)), axis=0) < STATIC_STOPS
    out = n.zeros_like(still)
    out[1:-1, 1:-1] = still[1:-1, 1:-1] & still[:-2, 1:-1] & still[2:, 1:-1] & still[1:-1, :-2] & still[1:-1, 2:]
    return out


def detail(lumas, base_lumas):
    """Added detail: Laplacian energy of log luma with the model over without it, on the pixels
    static in both windows. None when no pixel is static in both."""
    n = np()
    floor = floor_of(base_lumas[0])
    logs, base = log_frames(lumas, floor), log_frames(base_lumas, floor)
    mask = static_mask(logs) & static_mask(base)
    if not mask.any():
        return None
    energy = lambda ls: float(n.mean([(laplacian(l)[mask] ** 2).mean() for l in ls]))
    return energy(logs) / energy(base)


def anchor(t, frames=WINDOW):
    """The window's first frame, from `t`, the trimmed change of each pair of the series: SETTLE frames after the series' first cut, or 0 without a cut. A cut
    is a pair whose trimmed change is over CUT_STOPS and over 3 times the pair before it (a camera
    that starts or stops moving changes gradually). Counting from the cut fixes the moment: GTA's
    benchmark camera starts a pass up to about a second earlier or later from run to run, so the
    series is captured across the cut. SETTLE leaves out the frames right after it, where DLSS's
    history is still converging (their change decays over about 30 frames). None when the series
    ends too soon after the cut."""
    for i in range(1, len(t)):
        if t[i] > CUT_STOPS and t[i] > 3 * t[i - 1]:
            start = i + 1 + SETTLE
            return start if start + frames <= len(t) + 1 else None
    return 0


def pair_stats(files):
    """(mean, trimmed mean) of the change of each consecutive pair of `files`, two frames in memory
    at a time (a series of 400 frames at 1440p does not fit as a whole)."""
    n = np()
    prev, floor, out = None, None, []
    for f in files:
        y = load_luma(f)
        floor = floor_of(y) if floor is None else floor
        l = n.log2(y + floor)
        if prev is not None:
            d = n.abs(l - prev)
            out.append((float(d.mean()), trimmed_mean(d)))
        prev = l
    return out


def window(run, start=None, frames=WINDOW):
    """The run's `frames` lumas from `start` (None: from the cut, `anchor`), or an error string."""
    d = series_dir(run)
    if d is None:
        return f"no series in {run}"
    files = frame_files(d)
    if start is None:
        start = anchor([t for _, t in pair_stats(files)], frames) if len(files) > frames else 0
        if start is None:
            return f"{d}: {len(files)} frames, the series ends before {SETTLE} + {frames} frames after its cut"
    if len(files) < start + frames:
        return f"{d}: {len(files)} frames, the window needs {start + frames}"
    return [load_luma(f) for f in files[start : start + frames]]


def scan(run, frames=WINDOW):
    """Mean change (stops) of every consecutive pair of the whole series, and the stillest window."""
    d = series_dir(run)
    stats = pair_stats(frame_files(d)) if d else []
    if not stats:
        return f"{run}: no series of 2 frames or more"
    trimmed = [t for _, t in stats]
    out = [f"{run}: {len(stats) + 1} frames in {d}", "  pair  mean%  flicker%"]
    out += [f"  {i:4d} {percent(m):6.3f} {percent(t):8.3f}" for i, (m, t) in enumerate(stats)]
    k = frames - 1
    if len(trimmed) >= k:
        best = min(range(len(trimmed) - k + 1), key=lambda s: sum(trimmed[s : s + k]))
        out.append(f"  stillest {frames}-frame window: frames {best}..{best + k}, flicker {percent(sum(trimmed[best:best + k]) / k):.3f}%")
    cut = anchor(trimmed, frames)
    out.append(f"  window after the cut: {'none (the series ends too soon)' if cut is None else f'frames {cut}..{cut + k}'}")
    return "\n".join(out)


def summary(run, baseline=None, start=None, frames=WINDOW):
    """`flicker X% detail xY` for bench-report.py, or None when the run has no series."""
    if series_dir(run) is None:
        return None
    lumas = window(run, start, frames)
    if isinstance(lumas, str):
        return lumas
    text = f"flicker {percent(flicker(lumas)):.2f}%"
    if baseline:
        base = window(baseline, start, frames)
        if isinstance(base, str):
            return f"{text} (detail: {base})"
        if base[0].shape != lumas[0].shape:
            return f"{text} (detail: the baseline is {base[0].shape[1]}x{base[0].shape[0]})"
        x = detail(lumas, base)
        text += " detail -" if x is None else f" detail x{x:.2f}"
    return text


def main(argv):
    opts = {"--baseline": None, "--frames": WINDOW, "--start": None}
    runs, want_scan = [], False
    it = iter(argv)
    for a in it:
        if a in opts:
            opts[a] = next(it, None)
        elif a == "--scan":
            want_scan = True
        else:
            runs.append(a)
    if not runs:
        print(__doc__, file=sys.stderr)
        return 2
    frames, baseline = int(opts["--frames"]), opts["--baseline"]
    start = None if opts["--start"] is None else int(opts["--start"])
    if want_scan:
        for run in runs:
            print(scan(run, frames))
        return 0
    values = []
    for run in runs:
        lumas = window(run, start, frames)
        if isinstance(lumas, str):
            print(f"{run}: {lumas}")
            continue
        f = flicker(lumas)
        values.append(f)
        line = f"{run}: flicker {percent(f):.3f}% ({frames} frames from {'the cut' if start is None else start})"
        if baseline:
            base = window(baseline, start, frames)
            x = detail(lumas, base) if not isinstance(base, str) else None
            line += f", detail x{x:.3f}" if x is not None else f", detail - ({base if isinstance(base, str) else 'no static pixels'})"
        print(line)
    if len(values) > 1:
        print(f"flicker spread over {len(values)} runs: max / min - 1 = {100 * (max(values) / min(values) - 1):.1f}%" if min(values) > 0 else "flicker spread: a run has no flicker")
    return 0 if values else 1


def _self_test():
    n = np()
    rng = n.random.default_rng(1)
    h, w = 120, 160
    # A textured scene (luma 0.05..0.6) whose texture sits well above the 0.05 x geomean floor.
    base = 0.05 + 0.55 * rng.random((h, w))
    still = [base.copy() for _ in range(WINDOW)]
    assert flicker(still) == 0.0, "identical frames: no flicker"
    # A global luma wobble of +-2%: every pair changes by log2(1.02 / 0.98) stops minus what the
    # floor takes off; no pixel is a mover, so the 5% exclusion keeps a uniform change as it is.
    wobble = [base * (1.02 if i % 2 else 0.98) for i in range(WINDOW)]
    f0 = floor_of(wobble[0])
    want = float(n.mean(n.abs(n.log2(base * 1.02 + f0) - n.log2(base * 0.98 + f0))))
    got = flicker(wobble)
    assert abs(got - want) < 2e-3 * want, (got, want)
    assert 1.5 < percent(got) < 4.0, percent(got)
    # Without the floor's damping it is exactly the wobble.
    assert abs(trimmed_mean(n.full((10, 10), n.log2(1.02 / 0.98))) - n.log2(1.02 / 0.98)) < 1e-12
    # A bright block (4% of the pixels) moving across an otherwise still scene is excluded.
    moving = []
    for i in range(WINDOW):
        y = base.copy()
        y[10:34, 10 + 8 * i : 42 + 8 * i] = 1.0
        moving.append(y)
    assert (24 * 32) / (h * w) < EXCLUDE
    assert flicker(moving) < 1e-3, percent(flicker(moving))
    # Detail: identical runs give x1.00; a sharpened run more, a blurred one less.
    assert abs(detail(still, still) - 1.0) < 1e-12
    blur = (base + n.roll(base, 1, 0) + n.roll(base, -1, 0) + n.roll(base, 1, 1) + n.roll(base, -1, 1)) / 5
    assert detail([blur] * WINDOW, still) < 0.9
    sharp = n.clip(base + 0.3 * (base - blur), 1e-3, None)
    assert detail([sharp] * WINDOW, still) > 1.1
    # The moving block is not static in the moving run: the detail ratio ignores it.
    assert abs(detail(moving, still) - 1.0) < 1e-12
    changes = lambda ls: [trimmed_mean(p) for p in pair_changes(log_frames(ls, floor_of(ls[0])))]
    # The window starts SETTLE frames after a cut, here into a black frame then the shot.
    other = 0.05 + 0.55 * rng.random((h, w))
    black = n.full((h, w), 1e-4)
    shot = [other * (1 + 0.001 * (i % 2)) for i in range(SETTLE + WINDOW)]
    assert anchor(changes(still + [black, black] + shot)) == WINDOW + SETTLE
    assert anchor(changes(still + [black] + shot[:SETTLE])) is None, "the series ends too soon"
    assert anchor(changes(still)) == 0, "no cut: from the start"
    pan = [n.roll(base, 3 * i, 1) for i in range(4)]
    assert anchor(changes(pan + still)) == 0, "a camera that stops is not a cut"
    print("bench-iq self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        _self_test()
    else:
        sys.exit(main(sys.argv[1:]))
