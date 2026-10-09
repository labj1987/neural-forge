#!/usr/bin/env python3
"""Encodes a dumped scene-linear DLSS input for the model, and judges the model's answer.

The pre-upscaler experiment E1b (docs/PRE_UPSCALER_DESIGN.md, "E1b: the HDR encode"). The
layer's dump mode (NEURAL_FORGE_PREUPSCALE=dump) writes colour.rgba16f (padded size,
little-endian halves, scene-linear, pre-exposure), meta.json and exposure.json (every
registered 1x1 float image at the DLSS submit). This script:

  info   DUMP                       exposure values and the colour's statistics under each
                                    exposure convention
  encode DUMP ENC OUT               writes the frame to send (RGBA16F, or RGBA8 for `sdr8`),
                                    unpadded
  judge  DUMP ENC ANSWER OUTDIR     applies ENC's inverse to the answer, writes
                                    <tag>-input.png, <tag>-answer.png and <tag>-diff.png and
                                    prints the statistics (one JSON line, also <tag>.json)
  montage OUTDIR OUT TAG...         side by side: input | answer | diff, one row per tag

Encodes (E = the exposure scale: the exposure value under --convention, divided by --white):

  opendlss         v = scene * E; per channel, above 0.75: 0.75 + 0.25 (1 - exp(-5.770780 (v - 0.75)));
                   then the sRGB OETF. Inverse: sRGB EOTF; for y in [0.75, 1): v = 0.75 -
                   ln(1 - (y - 0.75) / 0.25) / 5.770780 (y clamped below 1 - 1e-4); / E.
  opendlss-linear  the same without the sRGB step.
  lumaknee         v = scene * E; on luminance (Rec.709 weights), above 0.75: L' = 0.75 + 0.25
                   (1 - exp(-(L - 0.75) / 0.25)), rgb *= L' / L, then the peak channel brought to 1
                   by one scalar if above (the layer's encode.comp SoftKnee); then sRGB. The inverse
                   undoes the luminance knee; the peak divide is lossy and not undone.
  reinhard         v = scene * E; per channel v / (1 + v), linear. Inverse y / (1 - y) / E.
  sdr8             opendlss quantised to RGBA8 (round to nearest): the 8-bit reference, sent as an
                   RGBA8 proxy (Hdr=0, SDR=1).
  raw              the scene-linear frame as dumped (no encode; display as opendlss).

The display mapping for both pictures is the encode's own forward curve after the exposure, then
sRGB (so a perfect identity answer gives identical PNGs); `ref_*` statistics use one display for
every encode (opendlss's shoulder at --ref-white, then sRGB), so edit magnitudes compare across
encodes and paper whites.

Needs Python 3, numpy and Pillow (the rig has both; run it there).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

K = 5.770780
KNEE = 0.75
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
ENCODES = ("opendlss", "opendlss-linear", "lumaknee", "reinhard", "sdr8", "raw")


# ---- Transfer functions. ----

def srgb_oetf(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_eotf(s):
    s = np.clip(s, 0.0, 1.0)
    return np.where(s <= 0.04045, s / 12.92, np.power((s + 0.055) / 1.055, 2.4))


def shoulder(v):
    v = np.maximum(v, 0.0)
    return np.where(v <= KNEE, v, KNEE + 0.25 * (1.0 - np.exp(-K * (v - KNEE))))


def shoulder_inv(y):
    y = np.clip(y, 0.0, 1.0 - 1e-4)
    return np.where(y < KNEE, y, KNEE - np.log(1.0 - (y - KNEE) / 0.25) / K)


def luma(rgb):
    return rgb @ LUMA


def lumaknee(v):
    v = np.maximum(v, 0.0)
    l = luma(v)
    rolled = KNEE + 0.25 * (1.0 - np.exp(-(l - KNEE) / 0.25))
    scale = np.where(l > KNEE, rolled / np.maximum(l, 1e-12), 1.0)
    v = v * scale[..., None]
    peak = v.max(axis=-1)
    return v / np.maximum(peak, 1.0)[..., None]


def lumaknee_inv(y):
    y = np.clip(y, 0.0, 1.0)
    l = luma(y)
    lc = np.clip(l, 0.0, 1.0 - 1e-4)
    orig = KNEE - 0.25 * np.log(1.0 - (lc - KNEE) / 0.25)
    scale = np.where(l > KNEE, orig / np.maximum(l, 1e-12), 1.0)
    return y * scale[..., None]


def reinhard(v):
    v = np.maximum(v, 0.0)
    return v / (1.0 + v)


def reinhard_inv(y):
    y = np.clip(y, 0.0, 1.0 - 1e-4)
    return y / (1.0 - y)


# Per encode: curve (exposed scene -> [0, 1] model domain before any sRGB), its inverse, and
# whether the model input is sRGB-encoded.
CURVES = {
    "opendlss": (shoulder, shoulder_inv, True),
    "opendlss-linear": (shoulder, shoulder_inv, False),
    "lumaknee": (lumaknee, lumaknee_inv, True),
    "reinhard": (reinhard, reinhard_inv, False),
    "sdr8": (shoulder, shoulder_inv, True),
    "raw": (shoulder, shoulder_inv, True),
}


# ---- Files. ----

def load_dump(dump):
    meta = json.load(open(os.path.join(dump, "meta.json")))
    w, h, pw, ph = meta["width"], meta["height"], meta["padded_width"], meta["padded_height"]
    raw = np.fromfile(os.path.join(dump, "colour.rgba16f"), dtype="<f2").reshape(ph, pw, 4)
    colour = raw[:h, :w].astype(np.float64)
    exposure = None
    path = os.path.join(dump, "exposure.json")
    if os.path.exists(path):
        exposure = json.load(open(path))
    return meta, colour, exposure


def clean(rgb):
    return np.nan_to_num(np.maximum(rgb, 0.0), nan=0.0, posinf=65504.0)


def exposure_scale(exposure, args):
    """The scale the scene is multiplied by before the curve, from --exposure, or the chosen
    exposure image's first channel and --convention (mul: scene * e, div: scene / e), divided by
    --white (OpenDLSS-NR's paperWhite, in exposed units)."""
    if args.exposure is not None:
        return float(args.exposure) / args.white
    if not exposure:
        sys.exit("no exposure.json in the dump; pass --exposure")
    images = [i for i in exposure["images"] if i.get("values")]
    if not images:
        sys.exit("exposure.json has no read values; pass --exposure")
    e = float(images[min(args.exposure_image, len(images) - 1)]["values"][0])
    return (e if args.convention == "mul" else 1.0 / e) / args.white


def write_f16(path, rgb):
    h, w, _ = rgb.shape
    out = np.empty((h, w, 4), dtype="<f2")
    out[..., :3] = np.clip(rgb, 0.0, 65504.0).astype("<f2")
    out[..., 3] = 1.0
    out.tofile(path)


def png(path, rgb01):
    Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB").save(path)


# ---- Encode and decode. ----

def encode(enc, scene, e):
    curve, _, srgb = CURVES[enc]
    if enc == "raw":
        return scene
    y = curve(scene * e)
    return srgb_oetf(y) if srgb else y


def decode(enc, answer, e):
    """The answer back in scene-linear units."""
    _, inv, srgb = CURVES[enc]
    if enc == "raw":
        return answer
    y = srgb_eotf(answer) if srgb else answer
    return inv(y) / e


def display(enc, scene, e):
    """The encode's own display: exposure, its curve, sRGB."""
    curve, _, _ = CURVES[enc]
    return srgb_oetf(curve(scene * e))


def ref_display(scene, e):
    return srgb_oetf(shoulder(scene * e))


# ---- Commands. ----

def pct(a, q):
    return float(np.percentile(a, q))


def cmd_info(args):
    meta, colour, exposure = load_dump(args.dump)
    scene = clean(colour[..., :3])
    l = luma(scene)
    print(f"{args.dump}: {meta['width']}x{meta['height']} frame {meta['frame']}")
    print(f"  scene luma p1={pct(l, 1):.4g} p50={pct(l, 50):.4g} p90={pct(l, 90):.4g} p99={pct(l, 99):.4g} max={l.max():.4g}")
    print(f"  alpha min={colour[..., 3].min():.3g} mean={colour[..., 3].mean():.3g} max={colour[..., 3].max():.3g}")
    if not exposure:
        print("  no exposure.json")
        return
    for i, img in enumerate(exposure["images"]):
        print(f"  exposure[{i}] {img['image']} {img['format']} layout={img['layout']} assumed={img['layout_assumed']} raw={img['raw_le_hex']} values={img['values']}")
        if img.get("values"):
            v = float(img["values"][0])
            if v > 0 and math.isfinite(v):
                for name, s in (("scene * e", v), ("scene / e", 1.0 / v)):
                    el = l * s
                    print(f"    {name}: exposed luma p50={pct(el, 50):.4g} p90={pct(el, 90):.4g} p99={pct(el, 99):.4g}; "
                          f"above the 0.75 knee {100 * np.mean(el > KNEE):.1f}%")


def cmd_encode(args):
    _, colour, exposure = load_dump(args.dump)
    scene = clean(colour[..., :3])
    e = exposure_scale(exposure, args) if args.enc != "raw" else 1.0
    y = encode(args.enc, scene, e)
    if args.enc == "sdr8":
        h, w, _ = y.shape
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = (np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        out[..., 3] = 255
        out.tofile(args.out)
    else:
        write_f16(args.out, y)
    # The identity round trip (f16 or 8-bit storage included): what the write-back would lose
    # with no model at all.
    if args.enc == "sdr8":
        stored = np.round(np.clip(y, 0, 1) * 255.0) / 255.0
    else:
        stored = np.clip(y, 0, 65504).astype(np.float16).astype(np.float64)
    back = decode(args.enc, stored, e)
    rel = np.abs(back - scene) / np.maximum(scene, 1e-3)
    l = luma(scene)
    print(json.dumps({
        "enc": args.enc, "exposure_scale": e, "out": args.out,
        "encoded_max": float(y.max()), "encoded_p50": pct(y, 50),
        "encoded_ge_0.999_pct": 100 * float(np.mean(np.any(y >= 0.999, axis=-1))),
        "identity_roundtrip_relerr_gt_1pct_pct": 100 * float(np.mean(np.any(rel > 0.01, axis=-1))),
        "identity_roundtrip_relerr_gt_1pct_on_luma_top1pct": 100 * float(np.mean(np.any(rel > 0.01, axis=-1)[l >= pct(l, 99)])),
    }))


def gradient_mask(disp, top):
    l = luma(disp)
    gx = np.zeros_like(l)
    gy = np.zeros_like(l)
    gx[:, 1:-1] = l[:, 2:] - l[:, :-2]
    gy[1:-1, :] = l[2:, :] - l[:-2, :]
    g = np.hypot(gx, gy)
    return g >= np.percentile(g, 100 - top)


def edit_stats(prefix, din, dout, mask):
    d = dout - din
    ad = np.abs(d)
    dl = luma(dout) - luma(din)
    # Chroma: the edit with its luminance part taken out.
    chroma = np.abs(d - dl[..., None]).mean()
    hi = ad[mask].mean()
    lo = ad[~mask].mean()
    return {
        f"{prefix}mean_abs_edit": float(ad.mean()),
        f"{prefix}mean_abs_edit_8bit": float(ad.mean() * 255),
        f"{prefix}mean_signed_edit_rgb": [float(x) for x in d.reshape(-1, 3).mean(axis=0)],
        f"{prefix}mean_abs_luma_edit": float(np.abs(dl).mean()),
        f"{prefix}mean_abs_chroma_edit": float(chroma),
        f"{prefix}edit_on_high_gradient": float(hi),
        f"{prefix}edit_elsewhere": float(lo),
        f"{prefix}high_gradient_ratio": float(hi / max(lo, 1e-9)),
        f"{prefix}p99_abs_edit": pct(ad.max(axis=-1), 99),
    }


def cmd_judge(args):
    _, colour, exposure = load_dump(args.dump)
    scene = clean(colour[..., :3])
    h, w, _ = scene.shape
    e = exposure_scale(exposure, args)
    tag = args.tag or args.enc
    os.makedirs(args.outdir, exist_ok=True)
    if args.enc == "sdr8":
        raw = np.fromfile(args.answer, dtype=np.uint8).reshape(h, w, 4)
        answer = raw[..., :3].astype(np.float64) / 255.0
        sent = np.round(encode("opendlss", scene, e) * 255.0) / 255.0
    else:
        raw = np.fromfile(args.answer, dtype="<f2").reshape(h, w, 4)
        answer = raw[..., :3].astype(np.float64)
        sent = np.clip(encode(args.enc, scene, e), 0, 65504).astype(np.float16).astype(np.float64)
    finite = np.isfinite(answer).all(axis=-1)
    answer = np.nan_to_num(answer, nan=0.0, posinf=65504.0, neginf=0.0)
    back = decode(args.enc, answer, e)
    back_sent = decode(args.enc, sent, e)
    din = display(args.enc, back_sent, e)
    dout = display(args.enc, back, e)
    ref = e * args.white / args.ref_white
    rin = ref_display(back_sent, ref)
    rout = ref_display(back, ref)
    mask = gradient_mask(din, args.top)
    clamp_hi = np.any(answer >= 0.999, axis=-1)
    sent_hi = np.any(sent >= 0.999, axis=-1)
    stats = {
        "tag": tag, "enc": args.enc, "exposure_scale": e,
        "answer_min": float(answer.min()), "answer_max": float(answer.max()),
        "nonfinite_px": int((~finite).sum()),
        "answer_mean_rgb": [float(x) for x in answer.reshape(-1, 3).mean(axis=0)],
        "sent_mean_rgb": [float(x) for x in sent.reshape(-1, 3).mean(axis=0)],
        "answer_at_clamp_pct": 100 * float(clamp_hi.mean()),
        "sent_at_clamp_pct": 100 * float(sent_hi.mean()),
        "answer_at_zero_pct": 100 * float(np.any(answer <= 0.0, axis=-1).mean()),
        "scene_mean_luma_in_out": [float(luma(back_sent).mean()), float(luma(back).mean())],
    }
    stats.update(edit_stats("", din, dout, mask))
    stats.update(edit_stats("ref_", rin, rout, mask))
    png(os.path.join(args.outdir, f"{tag}-input.png"), din)
    png(os.path.join(args.outdir, f"{tag}-answer.png"), dout)
    png(os.path.join(args.outdir, f"{tag}-diff.png"), np.abs(dout - din) * args.gain)
    with open(os.path.join(args.outdir, f"{tag}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


def cmd_montage(args):
    rows = []
    for tag in args.tags:
        imgs = [Image.open(os.path.join(args.outdir, f"{tag}-{k}.png")) for k in ("input", "answer", "diff")]
        if args.crop:
            x, y, cw, ch = (int(v) for v in args.crop.split(","))
            imgs = [i.crop((x, y, x + cw, y + ch)) for i in imgs]
        if args.scale != 1.0:
            imgs = [i.resize((int(i.width * args.scale), int(i.height * args.scale)), Image.LANCZOS) for i in imgs]
        rows.append(imgs)
    cw, ch = rows[0][0].size
    pad = 6
    sheet = Image.new("RGB", (3 * cw + 2 * pad, len(rows) * ch + (len(rows) - 1) * pad), (40, 40, 40))
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            sheet.paste(img, (c * (cw + pad), r * (ch + pad)))
    sheet.save(args.out)
    print(f"wrote {args.out}: rows {', '.join(args.tags)}; columns input | answer | diff")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def exposure_args(sp):
        sp.add_argument("--convention", choices=("mul", "div"), default="mul", help="scene * e (mul) or scene / e (div)")
        sp.add_argument("--exposure-image", type=int, default=0, help="which read exposure.json image (default 0)")
        sp.add_argument("--exposure", type=float, help="override: the scale itself")
        sp.add_argument("--white", type=float, default=1.0,
                        help="paper white in exposed units: the scale is divided by it (default 1: exposed 1.0 is white)")

    sp = sub.add_parser("info")
    sp.add_argument("dump")
    sp = sub.add_parser("encode")
    sp.add_argument("dump")
    sp.add_argument("enc", choices=ENCODES)
    sp.add_argument("out")
    exposure_args(sp)
    sp = sub.add_parser("judge")
    sp.add_argument("dump")
    sp.add_argument("enc", choices=ENCODES)
    sp.add_argument("answer")
    sp.add_argument("outdir")
    sp.add_argument("--tag")
    sp.add_argument("--gain", type=float, default=8.0, help="diff amplification (default 8)")
    sp.add_argument("--top", type=float, default=10.0, help="high-gradient pixels: the top N%% (default 10)")
    sp.add_argument("--ref-white", type=float, default=3.0,
                    help="paper white of the common reference display behind the ref_* statistics (default 3)")
    exposure_args(sp)
    sp = sub.add_parser("montage")
    sp.add_argument("outdir")
    sp.add_argument("out")
    sp.add_argument("tags", nargs="+")
    sp.add_argument("--crop", help="x,y,w,h before scaling")
    sp.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args()
    {"info": cmd_info, "encode": cmd_encode, "judge": cmd_judge, "montage": cmd_montage}[args.cmd](args)


if __name__ == "__main__":
    main()
