#!/usr/bin/env python3
"""Summarise gta-bench.sh runs.

usage: bench-report.py [--host HOST] [--mean] [--fg] <label>...
       bench-report.py --self-test

Per run: the real fps of each benchmark pass (GTA's own Benchmark file), pass 4 in detail
(real fps from its frame-time file; displayed fps from MangoHud's per-frame log, counted
over pass 4's wall-clock window, which is right below a frame generator because MangoHud
is the last layer; the layer's composited rate, or, when the model ran before the upscaler
(the pre-upscaler path's [preupscale] summary lines are in the log), the rate of frames it
held and enhanced there instead, since those frames are not "composited" and the
composited figure then only counts loading screens; GPU utilisation and power from nvidia-smi
over the same window), and the median of every field of the layer's [sync] lines (one per
300 composed frames, first line dropped as warm-up). --mean adds the mean of the pass-4
numbers over all labels given, which is how three runs of one configuration are reported.

Each run also gets its pass-4 frame-generation multiplier (displayed / real). GTA decides at each
loading screen whether to run DLSS Frame Generation, and in benchmark launches it often does not,
with or without Neural Forge (docs/RUNNING_AND_MEASURING.md). --fg is for runs with frame
generation on in GTA's settings: a run under FG_ENGAGED is reported as "frame generation did not
engage", is left out of --mean, and the exit status is 1, so the run is repeated.

Reads $NF_BENCH_DIR (default ~/nf-spike/gta) on the machine it runs on; --host runs it on
the rig over ssh.
"""
import datetime as dt
import glob
import os
import re
import statistics
import subprocess
import sys

# Displayed / real at or above this means frame generation presented generated frames (2x is
# the lowest multiplier GTA offers).
FG_ENGAGED = 1.5

UNITS = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def ms(value):
    m = re.fullmatch(r"([\d.]+)(ns|µs|us|ms|s)", value)
    return float(m.group(1)) * UNITS[m.group(2)] if m else None


def sync_medians(log):
    rows = []
    for line in log.splitlines():
        if "[sync]" not in line:
            continue
        fields = {}
        for k, v in re.findall(r"([\w()+]+)=(\S+)", line):
            t = ms(v)
            if t is not None:
                fields[k] = t
            else:
                fields[k] = v
        rows.append(fields)
    rows = rows[1:] if len(rows) > 1 else rows
    if not rows:
        return "-"
    out = []
    for k in rows[0]:
        vals = [r[k] for r in rows if k in r]
        if all(isinstance(v, float) for v in vals):
            out.append(f"{k}={statistics.median(vals):.2f}")
        else:
            out.append(f"{k}={'/'.join(sorted(set(map(str, vals))))}")
    return f"{len(rows)} lines: " + " ".join(out)


def held_per_second(log):
    """Median `holds_per_s` of the pre-upscaler path's summary lines (one per 300 holds), or None
    when the log has none (the post-upscaler path ran, or a build before the rate was logged)."""
    rates = sorted(
        float(x) for x in re.findall(r"\[preupscale\] mode=\S+ .* holds_per_s=([\d.]+)", log) if float(x) > 0
    )
    return rates[len(rates) // 2] if rates else None


def rate_text(composited, held):
    """The layer's own rate for the report line. When frames were held before the upscaler the
    composited figure is not the effect's rate, so it is labelled as what it is."""
    fmt = lambda v: "NA" if v is None else f"{v:.1f}"
    if held is None:
        return f"composited/s {fmt(composited)}"
    return f"held before upscaler/s {fmt(held)} (composited/s {fmt(composited)}: frames not held, e.g. loading screens)"


def report(label, base):
    d = os.path.join(base, label)
    if not os.path.exists(f"{d}/benchmark.txt"):
        print(f"{label}: no benchmark.txt (early exit?)")
        return None
    b = open(f"{d}/benchmark.txt").read().splitlines()
    avgs = [float(line.split(",")[3]) for line in b[1:6]]
    p4s = glob.glob(f"{d}/Pass4-*.txt")
    if not p4s:
        print(f"{label}: no Pass4 frame-time file")
        return None
    p4 = p4s[0]
    ts = re.search(r"Pass4-(.+)\.txt", p4).group(1)
    end = dt.datetime.strptime(ts, "%y-%m-%d-%H-%M-%S")
    ft = [float(line.split()[1]) for line in open(p4).read().splitlines()[1:] if line.strip()]
    dur = sum(ft) / 1000
    real4 = len(ft) / dur
    w0, w1 = end - dt.timedelta(seconds=dur - 5), end - dt.timedelta(seconds=5)
    disp = None
    for m in glob.glob(f"{d}/GTA5_Enhanced_*[0-9].csv"):
        st = dt.datetime.strptime(re.search(r"(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)", m).group(1), "%Y-%m-%d_%H-%M-%S")
        rows = open(m).read().splitlines()[3:]
        t = [st + dt.timedelta(microseconds=int(r.split(",")[-1]) / 1000) for r in rows if r.strip()]
        n = sum(1 for x in t if w0 <= x <= w1)
        if n:
            disp = n / (w1 - w0).total_seconds()
    gpu = power = None
    if os.path.exists(f"{d}/gpu.csv"):
        u = []
        for line in open(f"{d}/gpu.csv"):
            p = [x.strip() for x in line.split(",")]
            try:
                t = dt.datetime.strptime(p[0][:19], "%Y/%m/%d %H:%M:%S")
            except (ValueError, IndexError):
                continue
            if w0 <= t <= w1:
                u.append((float(p[1]), float(p[2])))
        if u:
            gpu = sum(x for x, _ in u) / len(u)
            power = sum(y for _, y in u) / len(u)
    log = open(f"{d}/launch.log", errors="replace").read()
    comp = sorted(float(x) for x in re.findall(r"\[present\] [\d.]+ fps \(([\d.]+)/s composited", log) if float(x) > 0)
    nf = comp[len(comp) // 2] if comp else None
    held = held_per_second(log)
    settings = open(f"{d}/settings").read().split() if os.path.exists(f"{d}/settings") else []
    fmt = lambda v, f: "NA" if v is None else f.format(v)
    fg = disp / real4 if disp else None
    print(
        f"{label}: passes {' / '.join(f'{a:.1f}' for a in avgs)} | pass4 ({dur:.0f}s) real {real4:.1f}, "
        f"displayed {fmt(disp, '{:.1f}')} (frame generation {fmt(fg, '{:.1f}x')}), {rate_text(nf, held)}, "
        f"GPU {fmt(gpu, '{:.0f}%')} {fmt(power, '{:.0f}W')}"
        + (f" | {' '.join(s for s in settings if '=' in s)}" if settings else "")
    )
    print(f"    [sync] {sync_medians(log)}")
    return real4, disp, gpu, fg


def main(argv):
    if len(argv) > 1 and argv[0] == "--host":
        with open(__file__, "rb") as f:
            sys.exit(subprocess.run(["ssh", argv[1], "python3", "-", *argv[2:]], stdin=f).returncode)
    mean = "--mean" in argv
    want_fg = "--fg" in argv
    labels = [a for a in argv if a not in ("--mean", "--fg")]
    if not labels:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    base = os.path.expanduser(os.environ.get("NF_BENCH_DIR", "~/nf-spike/gta"))
    results = [(label, r) for label, r in ((label, report(label, base)) for label in labels) if r]
    missed = [label for label, r in results if want_fg and not engaged(r[3])]
    for label in missed:
        print(f"{label}: frame generation did not engage (displayed / real under {FG_ENGAGED}x); run it again")
    results = [r for label, r in results if label not in missed]
    if mean and results:
        avg = lambda xs: sum(xs) / len(xs) if xs else None
        real = avg([r[0] for r in results])
        disp = avg([r[1] for r in results if r[1] is not None])
        gpu = avg([r[2] for r in results if r[2] is not None])
        print(
            f"mean of {len(results)}: real {real:.1f}"
            + (f", displayed {disp:.1f}" if disp else "")
            + (f", GPU {gpu:.0f}%" if gpu else "")
        )
    if missed:
        sys.exit(1)


def engaged(fg):
    """Whether a run's displayed / real ratio shows frame generation presenting frames."""
    return fg is not None and fg >= FG_ENGAGED


def _self_test():
    log = (
        "[neural-forge-layer] [present] 66.0 fps (0.0/s composited by the effect) over 5.0s\n"
        "[neural-forge-layer] [present] 30.0 fps (29.0/s composited by the effect) over 5.0s\n"
        "[neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=300 hold_ms median=10.20 "
        "capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=65.8\n"
        "[neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=600 hold_ms median=10.10 "
        "capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=66.2\n"
        "[neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=900 hold_ms median=10.10 "
        "capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=40.0\n"
    )
    assert held_per_second(log) == 65.8, held_per_second(log)
    assert held_per_second("[present] 61.0 fps (61.0/s composited by the effect) over 5.0s") is None
    assert rate_text(61.0, None) == "composited/s 61.0"
    assert engaged(3.99) and engaged(1.97) and not engaged(1.01) and not engaged(None)
    assert rate_text(29.0, 65.8).startswith("held before upscaler/s 65.8 (composited/s 29.0")
    print("bench-report self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        _self_test()
    else:
        main(sys.argv[1:])
