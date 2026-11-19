#![doc = include_str!("hdr_encode/doc.txt")]
// The pixel loops walk runtime-sized buffers in channel-sized chunks; `as_chunks` would only add a remainder to handle.
#![allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) mod args;
pub(crate) mod pil;
pub(crate) mod pyfmt;

use args::{Conv, Kind, Nargs, Opt, Outcome, Parsed, Pos, Spec};
use pyfmt::{
    f16_to_f64, f64_to_f16, fmt_f, fmt_g, json_dumps, json_value_repr, json_value_str, maximum, mean, mean_columns3, np_max, np_min, percentile, str_repr, Json,
};
use serde_json::Value;
use std::path::Path;
use std::process::ExitCode;

const K: f64 = 5.770780;
const KNEE: f64 = 0.75;
const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];
const ENCODES: &[&str] = &["opendlss", "opendlss-linear", "lumaknee", "reinhard", "sdr8", "raw"];
/// The layer's paper white (preupscale::hdr::DEFAULT_PAPER_WHITE) and its decode's clamped threshold.
const DEFAULT_PAPER_WHITE: f64 = 3.0;
const CLAMPED: f64 = 0.999;
/// The encodes the layer's decode rule (the clamped-keep) belongs to.
const KEEPS_CLAMPED: &[&str] = &["opendlss", "opendlss-linear", "sdr8"];

const HELP_TOP: &str = include_str!("hdr_encode/help_top.txt");
const HELP_INFO: &str = include_str!("hdr_encode/help_info.txt");
const HELP_ENCODE: &str = include_str!("hdr_encode/help_encode.txt");
const HELP_JUDGE: &str = include_str!("hdr_encode/help_judge.txt");
const HELP_MONTAGE: &str = include_str!("hdr_encode/help_montage.txt");
const COMMANDS: &[&str] = &["info", "encode", "judge", "selftest", "montage"];

/// What an uncaught Python exception or `sys.exit(message)` becomes: the message on stderr, status 1.
type Fail = String;

/// preupscale::hdr::exposure_ok: a value the layer divides by.
fn exposure_ok(e: f64) -> bool {
    e.is_finite() && 1e-6 < e && e < 65504.0
}

// ---- Transfer functions. ----

fn srgb_oetf(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_eotf(s: f64) -> f64 {
    let s = s.clamp(0.0, 1.0);
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn shoulder(v: f64) -> f64 {
    let v = maximum(v, 0.0);
    if v <= KNEE {
        v
    } else {
        KNEE + 0.25 * (1.0 - (-K * (v - KNEE)).exp())
    }
}

fn shoulder_inv(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0 - 1e-4);
    if y < KNEE {
        y
    } else {
        KNEE - (1.0 - (y - KNEE) / 0.25).ln() / K
    }
}

fn luma(p: &[f64]) -> f64 {
    p[0] * LUMA[0] + p[1] * LUMA[1] + p[2] * LUMA[2]
}

fn luma_of(a: &[f64]) -> Vec<f64> {
    a.chunks_exact(3).map(luma).collect()
}

fn lumaknee(v: [f64; 3]) -> [f64; 3] {
    let v = v.map(|x| maximum(x, 0.0));
    let l = luma(&v);
    let rolled = KNEE + 0.25 * (1.0 - (-(l - KNEE) / 0.25).exp());
    let scale = if l > KNEE { rolled / maximum(l, 1e-12) } else { 1.0 };
    let v = v.map(|x| x * scale);
    let peak = np_max(&v);
    let d = maximum(peak, 1.0);
    v.map(|x| x / d)
}

fn lumaknee_inv(y: [f64; 3]) -> [f64; 3] {
    let y = y.map(|x| x.clamp(0.0, 1.0));
    let l = luma(&y);
    let lc = l.clamp(0.0, 1.0 - 1e-4);
    let orig = KNEE - 0.25 * (1.0 - (lc - KNEE) / 0.25).ln();
    let scale = if l > KNEE { orig / maximum(l, 1e-12) } else { 1.0 };
    y.map(|x| x * scale)
}

fn reinhard(v: f64) -> f64 {
    let v = maximum(v, 0.0);
    v / (1.0 + v)
}

fn reinhard_inv(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0 - 1e-4);
    y / (1.0 - y)
}

/// Per encode: the curve (exposed scene -> [0, 1] model domain before any sRGB), its inverse, and whether the
/// model input is sRGB-encoded.
fn curve(enc: &str, v: [f64; 3]) -> [f64; 3] {
    match enc {
        "lumaknee" => lumaknee(v),
        "reinhard" => v.map(reinhard),
        _ => v.map(shoulder),
    }
}

fn curve_inv(enc: &str, y: [f64; 3]) -> [f64; 3] {
    match enc {
        "lumaknee" => lumaknee_inv(y),
        "reinhard" => y.map(reinhard_inv),
        _ => y.map(shoulder_inv),
    }
}

fn is_srgb(enc: &str) -> bool {
    !matches!(enc, "opendlss-linear" | "reinhard")
}

fn px(a: &[f64], i: usize) -> [f64; 3] {
    [a[i], a[i + 1], a[i + 2]]
}

fn map_px(a: &[f64], f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    for i in (0..a.len()).step_by(3) {
        out.extend_from_slice(&f(px(a, i)));
    }
    out
}

// ---- Files. ----

struct Dump {
    meta: Value,
    /// (h, w, 4) float64 colour
    colour: Vec<f64>,
    h: usize,
    w: usize,
    exposure: Option<Value>,
}

fn read_json(path: &Path) -> Result<Value, Fail> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn meta_dim(meta: &Value, key: &str) -> Result<usize, Fail> {
    meta.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("KeyError or bad value: {}", str_repr(key)))
}

fn load_dump(dump: &str) -> Result<Dump, Fail> {
    let dir = Path::new(dump);
    let meta = read_json(&dir.join("meta.json"))?;
    let (w, h, pw, ph) = (meta_dim(&meta, "width")?, meta_dim(&meta, "height")?, meta_dim(&meta, "padded_width")?, meta_dim(&meta, "padded_height")?);
    let path = dir.join("colour.rgba16f");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let count = bytes.len() / 2;
    if count != ph * pw * 4 {
        return Err(format!("ValueError: cannot reshape array of size {count} into shape ({ph},{pw},4)"));
    }
    let (eh, ew) = (h.min(ph), w.min(pw));
    let mut colour = Vec::with_capacity(eh * ew * 4);
    for y in 0..eh {
        for x in 0..ew * 4 {
            let o = (y * pw * 4 + x) * 2;
            colour.push(f16_to_f64(u16::from_le_bytes([bytes[o], bytes[o + 1]])));
        }
    }
    let epath = dir.join("exposure.json");
    let exposure = if epath.exists() { Some(read_json(&epath)?) } else { None };
    Ok(Dump { meta, colour, h: eh, w: ew, exposure })
}

/// `np.nan_to_num(np.maximum(rgb, 0.0), nan=0.0, posinf=65504.0)` of the first three channels.
fn clean_rgb(colour: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(colour.len() / 4 * 3);
    for p in colour.chunks_exact(4) {
        for &x in &p[..3] {
            let v = maximum(x, 0.0);
            out.push(if v.is_nan() {
                0.0
            } else if v == f64::INFINITY {
                65504.0
            } else {
                v
            });
        }
    }
    out
}

struct ExposureArgs {
    white: f64,
    exposure: Option<f64>,
    div: bool,
    exposure_image: i64,
    keep_clamped: bool,
}

fn json_to_float(v: &Value) -> Result<f64, Fail> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => pyfmt::parse_float(s).ok_or_else(|| format!("ValueError: could not convert string to float: {}", str_repr(s))),
        other => Err(format!("TypeError: float() argument must be a string or a real number, not {}", json_value_repr(other))),
    }
}

/// The scale the scene is multiplied by before the curve, from --exposure, or the chosen exposure image's first
/// channel and --convention (mul: scene * e, div: scene / e), divided by --white (OpenDLSS-NR's paperWhite, in
/// exposed units).
fn exposure_scale(exposure: Option<&Value>, a: &ExposureArgs) -> Result<f64, Fail> {
    if !(a.white.is_finite() && a.white > 0.0) {
        return Err(format!("--white {} is not a positive paper white", pyfmt::float_repr(a.white)));
    }
    let e = if let Some(x) = a.exposure {
        x
    } else {
        let exposure = match exposure {
            Some(v) if !is_falsy(v) => v,
            _ => return Err("no exposure.json in the dump; pass --exposure".into()),
        };
        let images = exposure.get("images").and_then(Value::as_array).ok_or("KeyError: 'images'")?;
        let images: Vec<&Value> = images.iter().filter(|i| i.get("values").is_some_and(|v| !is_falsy(v))).collect();
        if images.is_empty() {
            return Err("exposure.json has no read values; pass --exposure".into());
        }
        let n = images.len() as i64;
        let mut idx = a.exposure_image.min(n - 1);
        if idx < 0 {
            idx += n;
        }
        if idx < 0 {
            return Err("IndexError: list index out of range".into());
        }
        let first = images[idx as usize].get("values").and_then(|v| v.get(0)).ok_or("IndexError or TypeError reading values[0]")?;
        let e = json_to_float(first)?;
        if a.div {
            if e != 0.0 {
                1.0 / e
            } else {
                f64::INFINITY
            }
        } else {
            e
        }
    };
    if !exposure_ok(e) {
        return Err(format!("the exposure {} is not usable (zero, negative or not finite); the layer writes nothing back with it", pyfmt::float_repr(e)));
    }
    Ok(e / a.white)
}

fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

fn write_f16(path: &str, rgb: &[f64]) -> Result<(), Fail> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 8);
    for p in rgb.chunks_exact(3) {
        for &c in p {
            out.extend_from_slice(&f64_to_f16(c.clamp(0.0, 65504.0)).to_le_bytes());
        }
        out.extend_from_slice(&0x3c00u16.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("{path}: {e}"))
}

/// `(np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)`
fn to_u8(c: f64) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn write_png(path: &Path, rgb01: &[f64], w: usize, h: usize) -> Result<(), Fail> {
    let data: Vec<u8> = rgb01.iter().map(|&c| to_u8(c)).collect();
    pil::write_rgb_png(path, w, h, &data)
}

// ---- Encode and decode. ----

fn encode(enc: &str, scene: &[f64], e: f64) -> Vec<f64> {
    if enc == "raw" {
        return scene.to_vec();
    }
    let srgb = is_srgb(enc);
    map_px(scene, |p| {
        let y = curve(enc, p.map(|x| x * e));
        if srgb {
            y.map(srgb_oetf)
        } else {
            y
        }
    })
}

/// The answer back in scene-linear units. Given what was sent (`sent`, the encoded input) and the `original`
/// scene, the layer's rule for its encodes: the original pixel is kept where the encoded peak channel was clamped
/// (>= CLAMPED), so no pixel mixes kept and decoded channels.
fn decode(enc: &str, answer: &[f64], e: f64, keep: Option<(&[f64], &[f64])>) -> Vec<f64> {
    if enc == "raw" {
        return answer.to_vec();
    }
    let srgb = is_srgb(enc);
    let mut back = map_px(answer, |p| {
        let y = if srgb { p.map(srgb_eotf) } else { p };
        curve_inv(enc, y).map(|x| x / e)
    });
    if let Some((sent, original)) = keep {
        if KEEPS_CLAMPED.contains(&enc) {
            for i in (0..back.len()).step_by(3) {
                if np_max(&sent[i..i + 3]) >= CLAMPED {
                    back[i..i + 3].copy_from_slice(&original[i..i + 3]);
                }
            }
        }
    }
    back
}

/// The encode's own display: exposure, its curve, sRGB.
fn display(enc: &str, scene: &[f64], e: f64) -> Vec<f64> {
    map_px(scene, |p| curve(enc, p.map(|x| x * e)).map(srgb_oetf))
}

fn ref_display(scene: &[f64], e: f64) -> Vec<f64> {
    map_px(scene, |p| p.map(|x| srgb_oetf(shoulder(x * e))))
}

// ---- Commands. ----

fn pct(a: &[f64], q: f64) -> f64 {
    percentile(a, q)
}

fn frac(flags: impl Iterator<Item = bool>) -> f64 {
    let (mut n, mut hit) = (0usize, 0usize);
    for f in flags {
        n += 1;
        hit += usize::from(f);
    }
    if n == 0 {
        f64::NAN
    } else {
        hit as f64 / n as f64
    }
}

fn any_px(a: &[f64], pred: impl Fn(f64) -> bool) -> Vec<bool> {
    a.chunks_exact(3).map(|p| p.iter().any(|&x| pred(x))).collect()
}

fn alpha_of(colour: &[f64]) -> Vec<f64> {
    colour.chunks_exact(4).map(|p| p[3]).collect()
}

fn cmd_info(dump: &str) -> Result<(), Fail> {
    let d = load_dump(dump)?;
    let scene = clean_rgb(&d.colour);
    let l = luma_of(&scene);
    let get = |k: &str| d.meta.get(k).map(json_value_str).ok_or_else(|| format!("KeyError: {}", str_repr(k)));
    println!("{dump}: {}x{} frame {}", get("width")?, get("height")?, get("frame")?);
    let g = |x: f64| fmt_g(x, 4);
    println!("  scene luma p1={} p50={} p90={} p99={} max={}", g(pct(&l, 1.0)), g(pct(&l, 50.0)), g(pct(&l, 90.0)), g(pct(&l, 99.0)), g(np_max(&l)));
    let alpha = alpha_of(&d.colour);
    println!("  alpha min={} mean={} max={}", fmt_g(np_min(&alpha), 3), fmt_g(mean(&alpha), 3), fmt_g(np_max(&alpha), 3));
    let exposure = match &d.exposure {
        Some(v) if !is_falsy(v) => v,
        _ => {
            println!("  no exposure.json");
            return Ok(());
        }
    };
    let images = exposure.get("images").and_then(Value::as_array).ok_or("KeyError: 'images'")?;
    for (i, img) in images.iter().enumerate() {
        let f = |k: &str| img.get(k).map(json_value_str).ok_or_else(|| format!("KeyError: {}", str_repr(k)));
        println!(
            "  exposure[{i}] {} {} layout={} assumed={} raw={} values={}",
            f("image")?,
            f("format")?,
            f("layout")?,
            f("layout_assumed")?,
            f("raw_le_hex")?,
            f("values")?
        );
        if let Some(values) = img.get("values").filter(|v| !is_falsy(v)) {
            let v = json_to_float(values.get(0).ok_or("IndexError or TypeError reading values[0]")?)?;
            if v > 0.0 && v.is_finite() {
                for (name, s) in [("scene * e", v), ("scene / e", 1.0 / v)] {
                    let el: Vec<f64> = l.iter().map(|x| x * s).collect();
                    println!(
                        "    {name}: exposed luma p50={} p90={} p99={}; above the 0.75 knee {}%",
                        g(pct(&el, 50.0)),
                        g(pct(&el, 90.0)),
                        g(pct(&el, 99.0)),
                        fmt_f(100.0 * frac(el.iter().map(|&x| x > KNEE)), 1)
                    );
                }
            }
        }
    }
    Ok(())
}

fn flt(x: f64) -> Json {
    Json::Float(x)
}

fn cmd_encode(a: &Parsed, ex: &ExposureArgs) -> Result<(), Fail> {
    let (dump, enc, out) = (a.str("dump").unwrap_or(""), a.str("enc").unwrap_or(""), a.str("out").unwrap_or(""));
    let d = load_dump(dump)?;
    let scene = clean_rgb(&d.colour);
    let e = if enc != "raw" { exposure_scale(d.exposure.as_ref(), ex)? } else { 1.0 };
    let y = encode(enc, &scene, e);
    if enc == "sdr8" {
        let mut bytes = Vec::with_capacity(y.len() / 3 * 4);
        for p in y.chunks_exact(3) {
            bytes.extend(p.iter().map(|&c| to_u8(c)));
            bytes.push(255);
        }
        std::fs::write(out, bytes).map_err(|e| format!("{out}: {e}"))?;
    } else {
        write_f16(out, &y)?;
    }
    // The identity round trip (f16 or 8-bit storage included): what the write-back would lose with no model at all.
    let stored: Vec<f64> = if enc == "sdr8" {
        y.iter().map(|&c| (c.clamp(0.0, 1.0) * 255.0).round_ties_even() / 255.0).collect()
    } else {
        y.iter().map(|&c| f16_to_f64(f64_to_f16(c.clamp(0.0, 65504.0)))).collect()
    };
    let keep = ex.keep_clamped.then_some((stored.as_slice(), scene.as_slice()));
    let back = decode(enc, &stored, e, keep);
    let rel: Vec<f64> = back.iter().zip(&scene).map(|(b, s)| (b - s).abs() / maximum(*s, 1e-3)).collect();
    let l = luma_of(&scene);
    let bad = any_px(&rel, |x| x > 0.01);
    let top = pct(&l, 99.0);
    let sel = bad.iter().zip(&l).filter(|(_, &lv)| lv >= top).map(|(&b, _)| b);
    let obj = Json::Obj(vec![
        ("enc".into(), Json::Str(enc.into())),
        ("exposure_scale".into(), flt(e)),
        ("out".into(), Json::Str(out.into())),
        ("encoded_max".into(), flt(np_max(&y))),
        ("encoded_p50".into(), flt(pct(&y, 50.0))),
        ("encoded_ge_0.999_pct".into(), flt(100.0 * frac(any_px(&y, |x| x >= 0.999).into_iter()))),
        ("identity_roundtrip_relerr_gt_1pct_pct".into(), flt(100.0 * frac(bad.iter().copied()))),
        ("identity_roundtrip_relerr_gt_1pct_on_luma_top1pct".into(), flt(100.0 * frac(sel))),
    ]);
    println!("{}", json_dumps(&obj, None));
    Ok(())
}

fn gradient_mask(disp: &[f64], w: usize, h: usize, top: f64) -> Result<Vec<bool>, Fail> {
    let l = luma_of(disp);
    let mut g = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let gx = if x >= 1 && x + 1 < w { l[y * w + x + 1] - l[y * w + x - 1] } else { 0.0 };
            let gy = if y >= 1 && y + 1 < h { l[(y + 1) * w + x] - l[(y - 1) * w + x] } else { 0.0 };
            g[y * w + x] = gx.hypot(gy);
        }
    }
    let q = 100.0 - top;
    if !(0.0..=100.0).contains(&q) {
        return Err("ValueError: Percentiles must be in the range [0, 100]".into());
    }
    let t = pct(&g, q);
    Ok(g.iter().map(|&v| v >= t).collect())
}

fn edit_stats(prefix: &str, din: &[f64], dout: &[f64], mask: &[bool]) -> Vec<(String, Json)> {
    let d: Vec<f64> = dout.iter().zip(din).map(|(o, i)| o - i).collect();
    let ad: Vec<f64> = d.iter().map(|x| x.abs()).collect();
    let dl: Vec<f64> = luma_of(dout).iter().zip(luma_of(din)).map(|(o, i)| o - i).collect();
    // Chroma: the edit with its luminance part taken out.
    let chroma: Vec<f64> = d.chunks_exact(3).zip(&dl).flat_map(|(p, l)| p.iter().map(move |c| (c - l).abs())).collect();
    let gather = |want: bool| -> Vec<f64> { ad.chunks_exact(3).zip(mask).filter(|(_, &m)| m == want).flat_map(|(p, _)| p.iter().copied()).collect() };
    let hi = mean(&gather(true));
    let lo = mean(&gather(false));
    let abs_dl: Vec<f64> = dl.iter().map(|x| x.abs()).collect();
    let peaks: Vec<f64> = ad.chunks_exact(3).map(np_max).collect();
    let m_ad = mean(&ad);
    let denom = if 1e-9 > lo { 1e-9 } else { lo };
    let k = |name: &str| format!("{prefix}{name}");
    vec![
        (k("mean_abs_edit"), flt(m_ad)),
        (k("mean_abs_edit_8bit"), flt(m_ad * 255.0)),
        (k("mean_signed_edit_rgb"), Json::List(mean_columns3(&d).iter().map(|&x| flt(x)).collect())),
        (k("mean_abs_luma_edit"), flt(mean(&abs_dl))),
        (k("mean_abs_chroma_edit"), flt(mean(&chroma))),
        (k("edit_on_high_gradient"), flt(hi)),
        (k("edit_elsewhere"), flt(lo)),
        (k("high_gradient_ratio"), flt(hi / denom)),
        (k("p99_abs_edit"), flt(pct(&peaks, 99.0))),
    ]
}

struct JudgeArgs {
    tag: Option<String>,
    gain: f64,
    top: f64,
    ref_white: f64,
}

fn cmd_judge(a: &Parsed, ex: &ExposureArgs, ja: &JudgeArgs) -> Result<(), Fail> {
    let (dump, enc, answer_path, outdir) =
        (a.str("dump").unwrap_or(""), a.str("enc").unwrap_or(""), a.str("answer").unwrap_or(""), a.str("outdir").unwrap_or(""));
    let d = load_dump(dump)?;
    let scene = clean_rgb(&d.colour);
    let (h, w) = (d.h, d.w);
    let e = exposure_scale(d.exposure.as_ref(), ex)?;
    let tag = match ja.tag.as_deref() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => enc.to_string(),
    };
    std::fs::create_dir_all(outdir).map_err(|e| format!("{outdir}: {e}"))?;
    let bytes = std::fs::read(answer_path).map_err(|e| format!("{answer_path}: {e}"))?;
    let (answer_raw, sent): (Vec<f64>, Vec<f64>) = if enc == "sdr8" {
        if bytes.len() != h * w * 4 {
            return Err(format!("ValueError: cannot reshape array of size {} into shape ({h},{w},4)", bytes.len()));
        }
        let ans = bytes.chunks_exact(4).flat_map(|p| p[..3].iter().map(|&b| b as f64 / 255.0)).collect();
        let sent = encode("opendlss", &scene, e).iter().map(|&c| (c * 255.0).round_ties_even() / 255.0).collect();
        (ans, sent)
    } else {
        let count = bytes.len() / 2;
        if count != h * w * 4 {
            return Err(format!("ValueError: cannot reshape array of size {count} into shape ({h},{w},4)"));
        }
        let ans = bytes.chunks_exact(8).flat_map(|p| (0..3).map(|c| f16_to_f64(u16::from_le_bytes([p[2 * c], p[2 * c + 1]])))).collect();
        let sent = encode(enc, &scene, e).iter().map(|&c| f16_to_f64(f64_to_f16(c.clamp(0.0, 65504.0)))).collect();
        (ans, sent)
    };
    let finite: Vec<bool> = answer_raw.chunks_exact(3).map(|p| p.iter().all(|x| x.is_finite())).collect();
    let answer: Vec<f64> = answer_raw
        .iter()
        .map(|&x| {
            if x.is_nan() || x == f64::NEG_INFINITY {
                0.0
            } else if x == f64::INFINITY {
                65504.0
            } else {
                x
            }
        })
        .collect();
    let keep = ex.keep_clamped.then_some((sent.as_slice(), scene.as_slice()));
    let back = decode(enc, &answer, e, keep);
    let back_sent = decode(enc, &sent, e, keep);
    let din = display(enc, &back_sent, e);
    let dout = display(enc, &back, e);
    if ja.ref_white == 0.0 {
        return Err("ZeroDivisionError: float division by zero".into());
    }
    let rf = e * ex.white / ja.ref_white;
    let rin = ref_display(&back_sent, rf);
    let rout = ref_display(&back, rf);
    let mask = gradient_mask(&din, w, h, ja.top)?;
    let clamp_hi = any_px(&answer, |x| x >= 0.999);
    let sent_hi = any_px(&sent, |x| x >= 0.999);
    let zero = any_px(&answer, |x| x <= 0.0);
    let rgb = |v: &[f64]| Json::List(mean_columns3(v).iter().map(|&x| flt(x)).collect());
    let mut stats: Vec<(String, Json)> = vec![
        ("tag".into(), Json::Str(tag.clone())),
        ("enc".into(), Json::Str(enc.into())),
        ("exposure_scale".into(), flt(e)),
        ("answer_min".into(), flt(np_min(&answer))),
        ("answer_max".into(), flt(np_max(&answer))),
        ("nonfinite_px".into(), Json::Int(finite.iter().filter(|f| !**f).count() as i64)),
        ("answer_mean_rgb".into(), rgb(&answer)),
        ("sent_mean_rgb".into(), rgb(&sent)),
        ("answer_at_clamp_pct".into(), flt(100.0 * frac(clamp_hi.into_iter()))),
        ("sent_at_clamp_pct".into(), flt(100.0 * frac(sent_hi.into_iter()))),
        ("answer_at_zero_pct".into(), flt(100.0 * frac(zero.into_iter()))),
        ("scene_mean_luma_in_out".into(), Json::List(vec![flt(mean(&luma_of(&back_sent))), flt(mean(&luma_of(&back)))])),
    ];
    stats.extend(edit_stats("", &din, &dout, &mask));
    stats.extend(edit_stats("ref_", &rin, &rout, &mask));
    let dir = Path::new(outdir);
    write_png(&dir.join(format!("{tag}-input.png")), &din, w, h)?;
    write_png(&dir.join(format!("{tag}-answer.png")), &dout, w, h)?;
    let diff: Vec<f64> = dout.iter().zip(&din).map(|(o, i)| (o - i).abs() * ja.gain).collect();
    write_png(&dir.join(format!("{tag}-diff.png")), &diff, w, h)?;
    let stats = Json::Obj(stats);
    let jpath = dir.join(format!("{tag}.json"));
    std::fs::write(&jpath, json_dumps(&stats, Some(1))).map_err(|e| format!("{}: {e}", jpath.display()))?;
    println!("{}", json_dumps(&stats, None));
    Ok(())
}

fn allclose(a: &[f64], b: &[f64], rtol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-8 + rtol * y.abs())
}

/// The curves invert, the clamped-keep rule keeps whole pixels, and an unusable exposure is refused.
fn selftest() -> Result<(), Fail> {
    let scene = [0.05, 0.2, 0.4, 1.0, 2.0, 0.3, 40.0, 0.5, 0.1, 9.0, 9.0, 9.0];
    let e = 1.5 / DEFAULT_PAPER_WHITE;
    let sent = encode("opendlss", &scene, e);
    let back = decode("opendlss", &sent, e, None);
    let ok: Vec<bool> = sent.chunks_exact(3).map(|p| np_max(p) < CLAMPED).collect();
    let pick = |a: &[f64], want: bool| -> Vec<f64> { a.chunks_exact(3).zip(&ok).filter(|(_, &o)| o == want).flat_map(|(p, _)| p.iter().copied()).collect() };
    if !allclose(&pick(&back, true), &pick(&scene, true), 1e-6) {
        return Err(format!("AssertionError: {back:?} {scene:?}"));
    }
    // A pixel whose peak channel clamped keeps all three channels of the original, whatever the answer.
    let answer: Vec<f64> = sent.iter().map(|x| x * 0.9).collect();
    let kept = decode("opendlss", &answer, e, Some((&sent, &scene)));
    let (kc, sc) = (pick(&kept, false), pick(&scene, false));
    if kc.is_empty() || kc != sc {
        return Err(format!("AssertionError: {kept:?}"));
    }
    if !allclose(&pick(&kept, true), &pick(&decode("opendlss", &answer, e, None), true), 1e-5) {
        return Err("AssertionError".into());
    }
    for bad in [0.0, -1.0, f64::INFINITY, f64::NAN, 70000.0] {
        if exposure_ok(bad) {
            return Err(format!("AssertionError: {}", pyfmt::float_repr(bad)));
        }
    }
    if !(exposure_ok(1.0) && exposure_ok(2f64.powi(-14))) {
        return Err("AssertionError".into());
    }
    let mut a = ExposureArgs { white: DEFAULT_PAPER_WHITE, exposure: Some(0.0), div: false, exposure_image: 0, keep_clamped: true };
    if exposure_scale(None, &a).is_ok() {
        return Err("AssertionError: an exposure of 0 was taken".into());
    }
    a.exposure = Some(1.5);
    if exposure_scale(None, &a) != Ok(0.5) {
        return Err("AssertionError".into());
    }
    println!("hdr_encode self-test: ok");
    Ok(())
}

fn cmd_montage(a: &Parsed, crop: Option<&str>, scale: f64) -> Result<(), Fail> {
    let (outdir, out) = (a.str("outdir").unwrap_or(""), a.str("out").unwrap_or(""));
    let tags = a.list("tags");
    let mut rows: Vec<Vec<pil::Rgb>> = Vec::new();
    for tag in &tags {
        let mut imgs = Vec::new();
        for k in ["input", "answer", "diff"] {
            imgs.push(pil::read_rgb_png(&Path::new(outdir).join(format!("{tag}-{k}.png")))?);
        }
        if let Some(c) = crop {
            let parts: Vec<&str> = c.split(',').collect();
            let nums: Vec<i64> = parts
                .iter()
                .map(|p| pyfmt::parse_int(p).ok_or_else(|| format!("ValueError: invalid literal for int() with base 10: {}", str_repr(p))))
                .collect::<Result<_, _>>()?;
            let [x, y, cw, ch] = nums[..] else {
                return Err(if nums.len() > 4 {
                    "ValueError: too many values to unpack (expected 4)".into()
                } else {
                    format!("ValueError: not enough values to unpack (expected 4, got {})", nums.len())
                });
            };
            imgs = imgs.iter().map(|i| i.crop(x, y, x + cw, y + ch)).collect::<Result<_, _>>()?;
        }
        if scale != 1.0 {
            imgs = imgs.iter().map(|i| i.resize_lanczos(((i.w as f64) * scale) as i64, ((i.h as f64) * scale) as i64)).collect::<Result<_, _>>()?;
        }
        rows.push(imgs);
    }
    let (cw, ch) = (rows[0][0].w, rows[0][0].h);
    let pad = 6;
    let mut sheet = pil::Rgb::filled(3 * cw + 2 * pad, rows.len() * ch + (rows.len() - 1) * pad, [40, 40, 40]);
    for (r, imgs) in rows.iter().enumerate() {
        for (c, img) in imgs.iter().enumerate() {
            sheet.paste(img, (c * (cw + pad)) as i64, (r * (ch + pad)) as i64);
        }
    }
    sheet.save(Path::new(out))?;
    println!("wrote {out}: rows {}; columns input | answer | diff", tags.join(", "));
    Ok(())
}

fn exposure_opts(opts: &mut Vec<Opt>) {
    opts.push(Opt { names: &["--convention"], dest: "convention", kind: Kind::One(Conv::Str), choices: Some(&["mul", "div"]) });
    opts.push(Opt { names: &["--exposure-image"], dest: "exposure_image", kind: Kind::One(Conv::Int), choices: None });
    opts.push(Opt { names: &["--exposure"], dest: "exposure", kind: Kind::One(Conv::Float), choices: None });
    opts.push(Opt { names: &["--white"], dest: "white", kind: Kind::One(Conv::Float), choices: None });
    opts.push(Opt { names: &["--no-keep-clamped"], dest: "keep_clamped", kind: Kind::Flag(false), choices: None });
}

fn help_opt() -> Opt {
    Opt { names: &["-h", "--help"], dest: "help", kind: Kind::Help, choices: None }
}

fn pos(dest: &'static str) -> Pos {
    Pos { dest, nargs: Nargs::One, choices: None }
}

fn exposure_args(p: &Parsed) -> ExposureArgs {
    ExposureArgs {
        white: p.float("white").unwrap_or(DEFAULT_PAPER_WHITE),
        exposure: p.float("exposure"),
        div: p.str("convention") == Some("div"),
        exposure_image: p.int("exposure_image").unwrap_or(0),
        keep_clamped: p.vals.get("keep_clamped").is_none_or(|v| matches!(v, args::Val::Bool(true))),
    }
}

fn top_error(msg: &str) -> ExitCode {
    args::error("xtask hdr-encode", HELP_TOP, msg);
    ExitCode::from(2)
}

pub fn main(argv: &[String]) -> ExitCode {
    let mut extras: Vec<String> = Vec::new();
    let mut i = 0;
    let mut cmd = None;
    while i < argv.len() {
        let t = argv[i].as_str();
        i += 1;
        if t == "-h" || (t.len() > 2 && t.starts_with("--h") && "--help".starts_with(t)) {
            print!("{HELP_TOP}");
            return ExitCode::SUCCESS;
        }
        if t.starts_with('-') && t.len() > 1 && !t.contains(' ') && t != "--" {
            extras.push(t.to_string());
            continue;
        }
        cmd = Some(t);
        break;
    }
    let Some(cmd) = cmd else { return top_error("the following arguments are required: cmd") };
    if !COMMANDS.contains(&cmd) {
        return top_error(&format!("argument cmd: invalid choice: {} (choose from {})", str_repr(cmd), COMMANDS.join(", ")));
    }
    let rest = &argv[i..];
    let (prog, help) = match cmd {
        "info" => ("xtask hdr-encode info", HELP_INFO),
        "encode" => ("xtask hdr-encode encode", HELP_ENCODE),
        "judge" => ("xtask hdr-encode judge", HELP_JUDGE),
        "montage" => ("xtask hdr-encode montage", HELP_MONTAGE),
        _ => ("xtask hdr-encode selftest", ""),
    };
    // `selftest` takes no arguments; its usage is argparse's one line.
    let selftest_help = "usage: xtask hdr-encode selftest [-h]\n\noptions:\n  -h, --help  show this help message and exit\n";
    let mut opts = vec![help_opt()];
    let mut positionals = Vec::new();
    match cmd {
        "info" => positionals.push(pos("dump")),
        "encode" | "judge" => {
            if cmd == "judge" {
                opts.push(Opt { names: &["--tag"], dest: "tag", kind: Kind::One(Conv::Str), choices: None });
                opts.push(Opt { names: &["--gain"], dest: "gain", kind: Kind::One(Conv::Float), choices: None });
                opts.push(Opt { names: &["--top"], dest: "top", kind: Kind::One(Conv::Float), choices: None });
                opts.push(Opt { names: &["--ref-white"], dest: "ref_white", kind: Kind::One(Conv::Float), choices: None });
            }
            exposure_opts(&mut opts);
            positionals.push(pos("dump"));
            positionals.push(Pos { dest: "enc", nargs: Nargs::One, choices: Some(ENCODES) });
            if cmd == "encode" {
                positionals.push(pos("out"));
            } else {
                positionals.push(pos("answer"));
                positionals.push(pos("outdir"));
            }
        }
        "montage" => {
            opts.push(Opt { names: &["--crop"], dest: "crop", kind: Kind::One(Conv::Str), choices: None });
            opts.push(Opt { names: &["--scale"], dest: "scale", kind: Kind::One(Conv::Float), choices: None });
            positionals.push(pos("outdir"));
            positionals.push(pos("out"));
            positionals.push(Pos { dest: "tags", nargs: Nargs::Plus, choices: None });
        }
        _ => {}
    }
    let spec = Spec { prog, help: if cmd == "selftest" { selftest_help } else { help }, opts, pos: positionals };
    let parsed = match args::parse(&spec, rest) {
        Outcome::Parsed(p) => p,
        Outcome::Exit(c) => return ExitCode::from(c),
    };
    extras.extend(parsed.extras.iter().cloned());
    if !extras.is_empty() {
        return top_error(&format!("unrecognized arguments: {}", extras.join(" ")));
    }
    let result = match cmd {
        "info" => cmd_info(parsed.str("dump").unwrap_or("")),
        "encode" => cmd_encode(&parsed, &exposure_args(&parsed)),
        "judge" => cmd_judge(
            &parsed,
            &exposure_args(&parsed),
            &JudgeArgs {
                tag: parsed.str("tag").map(String::from),
                gain: parsed.float("gain").unwrap_or(8.0),
                top: parsed.float("top").unwrap_or(10.0),
                ref_white: parsed.float("ref_white").unwrap_or(3.0),
            },
        ),
        "montage" => cmd_montage(&parsed, parsed.str("crop"), parsed.float("scale").unwrap_or(1.0)),
        _ => selftest(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selftest_passes() {
        selftest().unwrap();
    }

    #[test]
    fn unusable_exposures_are_refused() {
        for bad in [0.0, -1.0, f64::INFINITY, f64::NAN, 70000.0] {
            assert!(!exposure_ok(bad), "{bad}");
        }
        assert!(exposure_ok(1.0) && exposure_ok(2f64.powi(-14)));
    }

    #[test]
    fn curves_invert() {
        for enc in ["opendlss", "opendlss-linear", "reinhard"] {
            let scene = [0.05, 0.2, 0.4];
            let sent = encode(enc, &scene, 0.5);
            let back = decode(enc, &sent, 0.5, None);
            assert!(allclose(&back, &scene, 1e-6), "{enc}: {back:?}");
        }
    }

    #[test]
    fn clamped_pixels_keep_the_original() {
        let scene = [40.0, 0.5, 0.1];
        let sent = encode("opendlss", &scene, 0.5);
        let kept = decode("opendlss", &sent.iter().map(|x| x * 0.9).collect::<Vec<_>>(), 0.5, Some((&sent, &scene)));
        assert_eq!(kept, scene);
    }

    #[test]
    fn exposure_scale_divides_by_white() {
        let a = ExposureArgs { white: 3.0, exposure: Some(1.5), div: false, exposure_image: 0, keep_clamped: true };
        assert_eq!(exposure_scale(None, &a), Ok(0.5));
        let z = ExposureArgs { exposure: Some(0.0), ..a };
        assert!(exposure_scale(None, &z).is_err());
    }

    #[test]
    fn python_float_repr() {
        let cases =
            [(1e-5, "1e-05"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (0.0001, "0.0001"), (-2.5, "-2.5"), (1.5e300, "1.5e+300"), (f64::INFINITY, "inf")];
        for (x, s) in cases {
            assert_eq!(pyfmt::float_repr(x), s);
        }
        assert_eq!(fmt_g(0.000123456, 4), "0.0001235");
        assert_eq!(fmt_g(1234567.0, 4), "1.235e+06");
        assert_eq!(fmt_g(100.0, 4), "100");
    }

    #[test]
    fn f16_round_trip_and_ties() {
        assert_eq!(f64_to_f16(1.0), 0x3c00);
        assert_eq!(f64_to_f16(65520.0), 0x7c00);
        assert_eq!(f64_to_f16(65504.0), 0x7bff);
        assert_eq!(f64_to_f16(1.0 + 2f64.powi(-11)), 0x3c00);
        assert_eq!(f64_to_f16(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        for h in 0..0x7c00u16 {
            assert_eq!(f64_to_f16(f16_to_f64(h)), h);
        }
    }
}
