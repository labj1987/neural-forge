#!/usr/bin/env python3
"""How well the model's edit lands, compared between two capture runs of the same frames.

Each run is a directory of matched ``<key>-original.png`` / ``<key>-composited.png`` pairs,
as a frame series writes them (``neural-forge-cli shmctl capture --frames N``, see
``crates/layer/src/series.rs``). The reference run is the synchronous present; the test run
is the path under test (e.g. a pipelined present). Both should come from
``crates/layer/examples/pan.rs``, which stamps its frame counter into the top-left corner:
frames are matched across runs by that decoded counter, never by capture order, so the two
runs need not start on the same frame.

Per matched frame, ``edit = composited - original`` (per channel, signed). Reported:

* ``ncc``: zero-mean normalised cross-correlation of the reference and test edit fields
  (per-channel means removed, channels pooled). 1.0 = the test path puts the same edit in
  the same place; ~0 = unrelated. Frames whose reference or test edit is zero everywhere
  (the frame went out untouched) have no defined ncc and are counted as ``flat``.
* ``detail %``: mean |edit| on high-gradient pixels (top 10% of the original's luma
  gradient magnitude), test as a percentage of reference, summed over the run's frames.
  Below 100 means the test path's edit is weaker (or misplaced) where the detail is.

The 16-pixel band holding the frame stamp and a 1-pixel border are excluded.

Dependencies: Python 3 and Pillow (``python3-pil``); no numpy.

    scripts/agreement.py REF_DIR TEST_DIR
    scripts/agreement.py --runs REF1:TEST1 REF2:TEST2 REF3:TEST3
    scripts/agreement.py --self-test
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import random
import re
import sys
import tempfile

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageMath
except ImportError:  # pragma: no cover
    sys.exit("agreement.py needs Pillow (apt install python3-pil)")

# Frame stamp geometry: mirrors crates/layer/examples/pan.rs.
BLOCK = 8
MAGIC = (1, 0, 1, 1)
STAMP_BITS = 32
EXCLUDE_TOP = 2 * BLOCK
DETAIL_FRACTION = 0.10
# Originals of the same frame index from two runs are bit-identical when pan.rs ran with the
# same flags; a larger mean difference means the runs are not comparable.
ORIGINAL_TOLERANCE = 0.5

PAIR = re.compile(r"^(?P<key>.+)-original\.png$")


def lam(fn, **images):
    return ImageMath.lambda_eval(lambda e: fn(e), **images)


def decode_stamp(img: Image.Image) -> int | None:
    """The pan.rs frame counter in the top-left corner, or None when there is no stamp."""
    if img.width < (len(MAGIC) + STAMP_BITS + 1) * BLOCK or img.height < BLOCK:
        return None
    luma = img.convert("L")
    bits = [1 if luma.getpixel((i * BLOCK + BLOCK // 2, BLOCK // 2)) > 127 else 0 for i in range(len(MAGIC) + STAMP_BITS + 1)]
    if tuple(bits[: len(MAGIC)]) != MAGIC:
        return None
    value_bits = bits[len(MAGIC) : len(MAGIC) + STAMP_BITS]
    value = 0
    for b in value_bits:
        value = (value << 1) | b
    if sum(value_bits) % 2 != bits[-1]:
        return None
    return value


def list_pairs(directory: str) -> list[tuple[str, str, str]]:
    pairs = []
    for name in sorted(os.listdir(directory)):
        m = PAIR.match(name)
        if not m:
            continue
        composited = os.path.join(directory, f"{m['key']}-composited.png")
        if os.path.exists(composited):
            pairs.append((m["key"], os.path.join(directory, name), composited))
    return pairs


def index_run(directory: str, by_seq: bool) -> tuple[dict[int, tuple[str, str]], int]:
    """Frame index -> (original, composited) paths, and how many pairs had no readable stamp."""
    frames: dict[int, tuple[str, str]] = {}
    unreadable = 0
    for key, original, composited in list_pairs(directory):
        if by_seq:
            try:
                index = int(key)
            except ValueError:
                unreadable += 1
                continue
        else:
            with Image.open(original) as im:
                index = decode_stamp(im)
            if index is None:
                unreadable += 1
                continue
        frames.setdefault(index, (original, composited))
    return frames, unreadable


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def channels_f(img: Image.Image) -> list[Image.Image]:
    return [band.convert("F") for band in img.split()]


def mean_f(img: Image.Image) -> float:
    """Exact-enough mean of an F image (box reduction in C)."""
    return float(img.reduce((img.width, img.height)).getpixel((0, 0)))


def crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    return (1, EXCLUDE_TOP, width - 1, height - 1)


def detail_mask(original: Image.Image) -> Image.Image:
    """1.0 on the top DETAIL_FRACTION of luma gradient magnitude (cropped), else 0.0."""
    luma = original.convert("L").convert("F")
    gx = lam(lambda e: e["r"] - e["l"], r=ImageChops.offset(luma, -1, 0), l=ImageChops.offset(luma, 1, 0))
    gy = lam(lambda e: e["d"] - e["u"], d=ImageChops.offset(luma, 0, -1), u=ImageChops.offset(luma, 0, 1))
    g2 = lam(lambda e: e["x"] * e["x"] + e["y"] * e["y"], x=gx, y=gy).crop(crop_box(*original.size))
    values = array.array("f", g2.tobytes())
    stride = max(1, len(values) // 200_000)
    sample = sorted(values[::stride])
    threshold = sample[min(len(sample) - 1, int(len(sample) * (1.0 - DETAIL_FRACTION)))]
    # Strictly above when ties at the threshold would otherwise take in a flat area.
    if threshold <= 0.0:
        return lam(lambda e: e["g"] > 0.0, g=g2).convert("F")
    return lam(lambda e: e["g"] >= threshold, g=g2).convert("F")


def frame_metrics(ref_pair: tuple[str, str], test_pair: tuple[str, str]) -> dict:
    ref_o, ref_c = load_rgb(ref_pair[0]), load_rgb(ref_pair[1])
    test_o, test_c = load_rgb(test_pair[0]), load_rgb(test_pair[1])
    if not (ref_o.size == ref_c.size == test_o.size == test_c.size):
        return {"skip": "size"}
    box = crop_box(*ref_o.size)
    orig_diff = sum(mean_f(lam(lambda e: abs(e["a"] - e["b"]), a=a, b=b).crop(box)) for a, b in zip(channels_f(ref_o), channels_f(test_o))) / 3
    if orig_diff > ORIGINAL_TOLERANCE:
        return {"skip": "original"}
    mask = detail_mask(ref_o)
    mask_mean = mean_f(mask)
    cov = var_r = var_t = 0.0
    det_r = det_t = 0.0
    n = (box[2] - box[0]) * (box[3] - box[1])
    for ro, rc, to, tc in zip(channels_f(ref_o), channels_f(ref_c), channels_f(test_o), channels_f(test_c)):
        er = lam(lambda e: e["c"] - e["o"], c=rc, o=ro).crop(box)
        et = lam(lambda e: e["c"] - e["o"], c=tc, o=to).crop(box)
        mr, mt = mean_f(er), mean_f(et)
        # Centred before the products: keeps the float32 box means well conditioned.
        cr = lam(lambda e: e["a"] - mr, a=er)
        ct = lam(lambda e: e["a"] - mt, a=et)
        cov += mean_f(lam(lambda e: e["a"] * e["b"], a=cr, b=ct))
        var_r += mean_f(lam(lambda e: e["a"] * e["a"], a=cr))
        var_t += mean_f(lam(lambda e: e["a"] * e["a"], a=ct))
        det_r += mean_f(lam(lambda e: abs(e["a"]) * e["m"], a=er, m=mask))
        det_t += mean_f(lam(lambda e: abs(e["a"]) * e["m"], a=et, m=mask))
    ncc = cov / math.sqrt(var_r * var_t) if var_r > 1e-9 and var_t > 1e-9 else None
    # Mean |edit| per channel sample over the masked pixels.
    scale = 1.0 / (3.0 * mask_mean) if mask_mean > 0 else 0.0
    return {"ncc": ncc, "detail_ref": det_r * scale, "detail_test": det_t * scale, "pixels": n}


def compare_runs(ref_dir: str, test_dir: str, by_seq: bool, per_frame: bool) -> dict:
    ref, ref_bad = index_run(ref_dir, by_seq)
    test, test_bad = index_run(test_dir, by_seq)
    common = sorted(set(ref) & set(test))
    result = {
        "ref": ref_dir,
        "test": test_dir,
        "matched": 0,
        "ref_only": len(set(ref) - set(test)),
        "test_only": len(set(test) - set(ref)),
        "unstamped": ref_bad + test_bad,
        "original_mismatch": 0,
        "flat": 0,
        "frames": [],
    }
    nccs = []
    det_r = det_t = 0.0
    for index in common:
        m = frame_metrics(ref[index], test[index])
        if m.get("skip") == "original" or m.get("skip") == "size":
            result["original_mismatch"] += 1
            continue
        result["matched"] += 1
        if m["ncc"] is None:
            result["flat"] += 1
        else:
            nccs.append(m["ncc"])
        det_r += m["detail_ref"]
        det_t += m["detail_test"]
        if per_frame:
            ncc = "   n/a" if m["ncc"] is None else f"{m['ncc']:6.3f}"
            print(f"  frame {index:6d}  ncc {ncc}  |edit| on detail ref {m['detail_ref']:7.3f} test {m['detail_test']:7.3f}")
        result["frames"].append({"index": index, **{k: m[k] for k in ("ncc", "detail_ref", "detail_test")}})
    result["ncc_mean"] = sum(nccs) / len(nccs) if nccs else None
    result["ncc_min"] = min(nccs) if nccs else None
    result["detail_pct"] = 100.0 * det_t / det_r if det_r > 0 else None
    return result


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def report(results: list[dict], per_frame_json: bool) -> dict:
    print(f"{'run':>3}  {'matched':>7}  {'skipped':>7}  {'flat':>4}  {'ncc mean':>8}  {'ncc min':>7}  {'detail %':>8}")
    for i, r in enumerate(results, 1):
        skipped = r["ref_only"] + r["test_only"] + r["unstamped"] + r["original_mismatch"]
        print(f"{i:>3}  {r['matched']:>7}  {skipped:>7}  {r['flat']:>4}  {fmt(r['ncc_mean'], '8.4f')}  {fmt(r['ncc_min'], '7.4f')}  {fmt(r['detail_pct'], '8.1f')}")
        detail = []
        for k in ("ref_only", "test_only", "unstamped", "original_mismatch"):
            if r[k]:
                detail.append(f"{k}={r[k]}")
        if detail:
            print(f"     skipped: {', '.join(detail)}")
    nccs = [r["ncc_mean"] for r in results if r["ncc_mean"] is not None]
    dets = [r["detail_pct"] for r in results if r["detail_pct"] is not None]
    summary = {
        "ncc_mean": sum(nccs) / len(nccs) if nccs else None,
        "detail_pct": sum(dets) / len(dets) if dets else None,
        "runs": [
            {k: v for k, v in r.items() if k != "frames" or per_frame_json}
            for r in results
        ],
    }
    if len(results) > 1:
        print(f"avg  {'':>7}  {'':>7}  {'':>4}  {fmt(summary['ncc_mean'], '8.4f')}  {'':>7}  {fmt(summary['detail_pct'], '8.1f')}")
    print(json.dumps(summary, separators=(",", ":")))
    return summary


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def stamp(img: Image.Image, index: int) -> None:
    bits = list(MAGIC) + [(index >> (STAMP_BITS - 1 - i)) & 1 for i in range(STAMP_BITS)]
    bits.append(sum(bits[len(MAGIC) :]) % 2)
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(bits):
        c = (255, 255, 255) if b else (0, 0, 0)
        draw.rectangle((i * BLOCK, 0, i * BLOCK + BLOCK - 1, BLOCK - 1), fill=c)


def synth_original(rng: random.Random, size: tuple[int, int], index: int) -> Image.Image:
    w, h = size
    layers = []
    for _ in range(3):
        noise = Image.frombytes("L", (w // 4, h // 4), rng.randbytes((w // 4) * (h // 4))).resize(size, Image.BICUBIC)
        fine = Image.frombytes("L", size, rng.randbytes(w * h)).filter(ImageFilter.GaussianBlur(0.8))
        layers.append(Image.blend(noise, fine, 0.35))
    img = Image.merge("RGB", layers)
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rng.randrange(w), rng.randrange(EXCLUDE_TOP, h)
        r = rng.randrange(6, 30)
        c = tuple(rng.randrange(256) for _ in range(3))
        draw.rectangle((x, y, x + r, y + r // 2), fill=c) if rng.random() < 0.5 else draw.ellipse((x, y, x + r, y + r), fill=c)
    draw.rectangle((0, h * 3 // 4, w // 3, h), fill=(128, 128, 128))  # a flat area
    # Keep away from 0/255 so a +-40 edit never clips.
    img = Image.eval(img, lambda v: 48 + v * 160 // 255)
    stamp(img, index)
    return img


def apply_edit(original: Image.Image, edits: list[Image.Image]) -> Image.Image:
    out = [lam(lambda e: e["o"] + e["d"], o=o, d=d).convert("L") for o, d in zip(channels_f(original), edits)]
    return Image.merge("RGB", out)


def model_edit(original: Image.Image) -> list[Image.Image]:
    """A structured, content-dependent 'model edit': local contrast boost plus a colour tint."""
    blurred = original.filter(ImageFilter.GaussianBlur(2.0))
    edits = []
    for k, (o, b) in enumerate(zip(channels_f(original), channels_f(blurred))):
        edits.append(lam(lambda e: (e["o"] - e["b"]) * 0.8 + (3.0 if k == 0 else -2.0), o=o, b=b))
    return edits


def write_pair(directory: str, seq: int, original: Image.Image, composited: Image.Image) -> None:
    original.save(os.path.join(directory, f"{seq:06d}-original.png"))
    composited.save(os.path.join(directory, f"{seq:06d}-composited.png"))


def self_test() -> int:
    rng = random.Random(1234)
    size = (384, 216)
    frames = 6
    first_index = 1000
    with tempfile.TemporaryDirectory(prefix="agreement-selftest-") as tmp:
        dirs = {name: os.path.join(tmp, name) for name in ("ref", "same", "shifted1", "shifted", "noise", "partial")}
        for d in dirs.values():
            os.makedirs(d)
        for f in range(frames):
            index = first_index + f
            original = synth_original(rng, size, index)
            edit = model_edit(original)
            ref = apply_edit(original, edit)
            write_pair(dirs["ref"], f, original, ref)
            write_pair(dirs["same"], f + 7, original, ref)  # different capture numbering, same frames
            write_pair(dirs["shifted1"], f, original, apply_edit(original, [ImageChops.offset(e, 1, 0) for e in edit]))
            shifted = [ImageChops.offset(e, 3, 2) for e in edit]
            write_pair(dirs["shifted"], f, original, apply_edit(original, shifted))
            noise = [Image.frombytes("L", size, rng.randbytes(size[0] * size[1])).convert("F") for _ in range(3)]
            noise = [lam(lambda e: (e["n"] - 127.5) * 0.12, n=n) for n in noise]
            write_pair(dirs["noise"], f, original, apply_edit(original, noise))
            if f != 2:
                write_pair(dirs["partial"], f, original, ref)

        same = compare_runs(dirs["ref"], dirs["same"], False, False)
        shifted1 = compare_runs(dirs["ref"], dirs["shifted1"], False, False)
        shifted = compare_runs(dirs["ref"], dirs["shifted"], False, False)
        noise = compare_runs(dirs["ref"], dirs["noise"], False, False)
        partial = compare_runs(dirs["ref"], dirs["partial"], False, False)
        print(f"identical runs:         ncc {same['ncc_mean']:.4f}  detail {same['detail_pct']:.1f}%  matched {same['matched']}")
        print(f"edit shifted by (1,0):  ncc {shifted1['ncc_mean']:.4f}  detail {shifted1['detail_pct']:.1f}%")
        print(f"edit shifted by (3,2):  ncc {shifted['ncc_mean']:.4f}  detail {shifted['detail_pct']:.1f}%")
        print(f"noise-only edit:        ncc {noise['ncc_mean']:.4f}  detail {noise['detail_pct']:.1f}%")
        print(f"one frame missing:      matched {partial['matched']}  ref_only {partial['ref_only']}")
        checks = [
            ("stamps decode to the right frame indices", sorted(f["index"] for f in same["frames"]) == list(range(first_index, first_index + frames))),
            ("identical runs give ncc ~1", abs(same["ncc_mean"] - 1.0) < 1e-3),
            ("identical runs give detail ~100%", abs(same["detail_pct"] - 100.0) < 0.5),
            ("a shifted edit gives clearly lower ncc", shifted["ncc_mean"] < same["ncc_mean"] - 0.3),
            ("a 1 px shift scores between identical and a 3 px shift", shifted["ncc_mean"] < shifted1["ncc_mean"] < same["ncc_mean"] - 0.05),
            ("a noise-only edit gives ncc ~0", abs(noise["ncc_mean"]) < 0.05),
            ("a missing frame is skipped and counted", partial["matched"] == frames - 1 and partial["ref_only"] == 1),
        ]
        failed = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ref", nargs="?", help="reference capture directory (synchronous path)")
    parser.add_argument("test", nargs="?", help="test capture directory (path under test)")
    parser.add_argument("--runs", nargs="+", metavar="REF:TEST", help="several run pairs to average")
    parser.add_argument("--by-seq", action="store_true", help="match frames by capture file number instead of the pan.rs stamp")
    parser.add_argument("--per-frame", action="store_true", help="print every frame's numbers (and include them in the JSON)")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic self-test")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    pairs = []
    if args.runs:
        for item in args.runs:
            ref, sep, test = item.rpartition(":")
            if not sep or not ref or not test:
                parser.error(f"--runs takes REF:TEST pairs, got {item!r}")
            pairs.append((ref, test))
    if args.ref and args.test:
        pairs.insert(0, (args.ref, args.test))
    if not pairs:
        parser.error("give REF TEST, --runs REF:TEST ..., or --self-test")
    for ref, test in pairs:
        for d in (ref, test):
            if not os.path.isdir(d):
                parser.error(f"not a directory: {d}")
    results = []
    for ref, test in pairs:
        if args.per_frame:
            print(f"{ref} vs {test}:")
        results.append(compare_runs(ref, test, args.by_seq, args.per_frame))
    summary = report(results, args.per_frame)
    return 0 if summary["ncc_mean"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
