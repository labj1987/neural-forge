//! The pre-upscaler edit cache: the port of scripts/cache_common.py (the reference the layer's
//! preupscale_cache*.comp shaders mirror). Every step is computed in f32 as numpy computes it on
//! float32 arrays, and the means the metrics take are f64, as the Python's are.

use std::path::{Path, PathBuf};

use serde_json::Value;

pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
pub const EPS_V: f64 = 1.0 / 512.0;
pub const DUMP_FORMAT: &str = "neural-forge-cache-dump";
pub const DUMP_VERSION: i64 = 1;
pub const DEPTH_GEOMETRY_MIN: f64 = 0.05;

/// An H x W x C float32 image, row-major, channels innermost.
#[derive(Clone, Debug)]
pub struct Img {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

impl Img {
    pub fn new(h: usize, w: usize, c: usize, v: f32) -> Img {
        Img { h, w, c, d: vec![v; h * w * c] }
    }
    pub fn from_fn(h: usize, w: usize, c: usize, mut f: impl FnMut(usize, usize) -> f32) -> Img {
        let mut d = Vec::with_capacity(h * w * c);
        for i in 0..h * w {
            for k in 0..c {
                d.push(f(i, k));
            }
        }
        Img { h, w, c, d }
    }
    pub fn px(&self, i: usize) -> &[f32] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
    pub fn n(&self) -> usize {
        self.h * self.w
    }
    /// Elementwise with another image of the same shape.
    pub fn zip(&self, o: &Img, f: impl Fn(f32, f32) -> f32) -> Img {
        Img { h: self.h, w: self.w, c: self.c, d: self.d.iter().zip(&o.d).map(|(&a, &b)| f(a, b)).collect() }
    }
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Img {
        Img { h: self.h, w: self.w, c: self.c, d: self.d.iter().map(|&a| f(a)).collect() }
    }
    /// Each channel combined with the per-pixel value of a one-channel image.
    pub fn zip_px(&self, o: &Img, f: impl Fn(f32, f32) -> f32) -> Img {
        Img::from_fn(self.h, self.w, self.c, |i, k| f(self.d[i * self.c + k], o.d[i]))
    }
    pub fn channel(&self, k: usize) -> Img {
        Img::from_fn(self.h, self.w, 1, |i, _| self.d[i * self.c + k])
    }
    pub fn stack(chs: &[&Img]) -> Img {
        let c: usize = chs.iter().map(|x| x.c).sum();
        let mut d = Vec::with_capacity(chs[0].n() * c);
        for i in 0..chs[0].n() {
            for x in chs {
                d.extend_from_slice(x.px(i));
            }
        }
        Img { h: chs[0].h, w: chs[0].w, c, d }
    }
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.d.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

/// Every knob of the carry, with the shader's defaults.
#[derive(Clone, Debug)]
pub struct Params {
    pub eps_v: f64,
    pub max_luma_edit: f64,
    pub chroma_stops: f64,
    pub depth_tol: f64,
    pub colour_tol: f64,
    pub noise_aware: bool,
    pub colour_floor: f64,
    pub high_decay: f64,
    pub fingerprint: bool,
    pub print_tol: f64,
    pub surface_fill: bool,
    pub bilateral: bool,
    pub sigma_depth: f64,
    pub sigma_luma: f64,
    pub nearest_surface: bool,
    pub jitter: bool,
    pub chain_sample: String,
    pub chain: String,
    pub low_gain: f64,
    pub high_gain: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            eps_v: EPS_V,
            max_luma_edit: 1.5,
            chroma_stops: 1.0,
            depth_tol: 0.10,
            colour_tol: 0.50,
            noise_aware: true,
            colour_floor: 0.6,
            high_decay: 1.0,
            fingerprint: true,
            print_tol: 0.25,
            surface_fill: true,
            bilateral: true,
            sigma_depth: 0.15,
            sigma_luma: 1.0,
            nearest_surface: true,
            jitter: true,
            chain_sample: "bilinear".into(),
            chain: "recurrence".into(),
            low_gain: 1.0,
            high_gain: 1.0,
        }
    }
}

impl Params {
    pub fn as_json(&self) -> crate::pyfmt::J {
        use crate::pyfmt::J::{Bool, Float, Str};
        crate::pyfmt::J::obj(vec![
            ("eps_v", Float(self.eps_v)),
            ("max_luma_edit", Float(self.max_luma_edit)),
            ("chroma_stops", Float(self.chroma_stops)),
            ("depth_tol", Float(self.depth_tol)),
            ("colour_tol", Float(self.colour_tol)),
            ("noise_aware", Bool(self.noise_aware)),
            ("colour_floor", Float(self.colour_floor)),
            ("high_decay", Float(self.high_decay)),
            ("fingerprint", Bool(self.fingerprint)),
            ("print_tol", Float(self.print_tol)),
            ("surface_fill", Bool(self.surface_fill)),
            ("bilateral", Bool(self.bilateral)),
            ("sigma_depth", Float(self.sigma_depth)),
            ("sigma_luma", Float(self.sigma_luma)),
            ("nearest_surface", Bool(self.nearest_surface)),
            ("jitter", Bool(self.jitter)),
            ("chain_sample", Str(self.chain_sample.clone())),
            ("chain", Str(self.chain.clone())),
            ("low_gain", Float(self.low_gain)),
            ("high_gain", Float(self.high_gain)),
        ])
    }
}

// ------------------------------------------------------------------------------------------------
// The dump
// ------------------------------------------------------------------------------------------------

/// The decode's own rule: an exposure outside this range leaves the frame untouched.
pub fn exposure_ok(e: f64) -> bool {
    1e-6 < e && e < 65504.0
}

pub struct Hold {
    pub index: i64,
    pub jitter: [f32; 2],
    pub exposure: f64,
    pub colour: Img,
    pub answer: Img,
    pub mvec: Img,
    pub depth: Option<Img>,
    pub jitter_known: bool,
    pub mvec_valid: bool,
}

pub struct Dump {
    pub path: PathBuf,
    pub manifest: Value,
    pub width: usize,
    pub height: usize,
    pub paper_white: f64,
    pub mvec_scale: [f32; 2],
    pub has_depth: bool,
    pub frames: Vec<Value>,
}

fn f16_bits(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

impl Dump {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    fn read(&self, name: &str, half: bool, channels: usize) -> Result<Img, String> {
        let path = self.path.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = if half { 2 } else { 4 };
        let want = self.width * self.height * channels;
        if bytes.len() / size != want {
            return Err(format!("{}: {} values, expected {want} ({}x{}x{channels})", path.display(), bytes.len() / size, self.width, self.height));
        }
        let d = bytes.chunks_exact(size).map(|b| if half { f16_bits(b) } else { f32::from_le_bytes([b[0], b[1], b[2], b[3]]) }).collect();
        Ok(Img { h: self.height, w: self.width, c: channels, d })
    }

    pub fn hold(&self, k: usize) -> Result<Hold, String> {
        let f = &self.frames[k];
        let files = &f["files"];
        let file = |key: &str| files[key].as_str().ok_or_else(|| format!("hold {k}: no {key} file"));
        let mvec_valid = f.get("mvec_valid") != Some(&Value::Bool(false));
        // As preupscale_cache_motion.comp's half_vector: NaN is the largest finite half, Inf the largest of its sign.
        let raw = self.read(file("mvec")?, true, 2)?;
        let mut mv = Img::from_fn(raw.h, raw.w, 2, |i, c| {
            let x = raw.d[i * 2 + c];
            let x = if x.is_nan() { 65504.0 } else { x.clamp(-65504.0, 65504.0) };
            x * self.mvec_scale[c]
        });
        if !mvec_valid {
            mv = Img::new(raw.h, raw.w, 2, 0.0);
        }
        let depth_valid = f.get("depth_valid") != Some(&Value::Bool(false));
        let depth = match files.get("depth").and_then(Value::as_str) {
            Some(name) if self.has_depth && depth_valid && !name.is_empty() => Some(self.read(name, false, 1)?),
            _ => None,
        };
        let jitter = match f.get("jitter") {
            None => Some([0.0, 0.0]),
            Some(Value::Null) => None,
            Some(j) => Some([json_f64(&j[0]) as f32, json_f64(&j[1]) as f32]),
        };
        let exposure = match f.get("exposure") {
            None | Some(Value::Null) => 0.0,
            Some(e) => json_f64(e),
        };
        let rgb = |img: Img| Img::from_fn(img.h, img.w, 3, |i, c| img.d[i * 4 + c]);
        Ok(Hold {
            index: f.get("index").and_then(Value::as_i64).unwrap_or(k as i64),
            jitter: jitter.unwrap_or([0.0, 0.0]),
            exposure,
            colour: rgb(self.read(file("colour")?, true, 4)?),
            answer: rgb(self.read(file("answer")?, true, 4)?),
            mvec: mv,
            depth,
            jitter_known: jitter.is_some(),
            mvec_valid,
        })
    }
}

pub fn json_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

pub fn load_dump(path: &Path) -> Result<Dump, String> {
    let text = std::fs::read_to_string(path.join("manifest.json")).map_err(|e| format!("{}: {e}", path.display()))?;
    let m: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let p = path.display();
    if m.get("format").and_then(Value::as_str) != Some(DUMP_FORMAT) {
        return Err(format!("{p}: not a {DUMP_FORMAT} (format {})", m.get("format").map_or("None".into(), Value::to_string)));
    }
    if m.get("version").and_then(Value::as_i64).unwrap_or(0) != DUMP_VERSION {
        return Err(format!("{p}: dump version {}, this tool reads {DUMP_VERSION}", m.get("version").map_or("None".into(), Value::to_string)));
    }
    let frames = m["frames"].as_array().cloned().ok_or_else(|| format!("{p}: no frames"))?;
    let holds = m.get("holds").and_then(Value::as_u64).map_or(frames.len(), |h| h as usize);
    if frames.len() != holds {
        return Err(format!("{p}: manifest says {holds} holds but lists {} frames", frames.len()));
    }
    let scale = m.get("mvec_scale").map_or([1.0, 1.0], |s| [json_f64(&s[0]) as f32, json_f64(&s[1]) as f32]);
    Ok(Dump {
        path: path.to_path_buf(),
        width: m["width"].as_u64().ok_or("no width")? as usize,
        height: m["height"].as_u64().ok_or("no height")? as usize,
        paper_white: m.get("paper_white").map_or(3.0, json_f64),
        mvec_scale: scale,
        has_depth: m.get("depth_format").and_then(Value::as_str).unwrap_or("none") != "none",
        frames,
        manifest: m,
    })
}

/// Writes holds as a v1 dump (scripts/cache_common.py's write_dump), the manifest as json.dump(indent=1).
pub fn write_dump(path: &Path, holds: &[Hold], paper_white: f64, game: &str, extra: Vec<(String, crate::pyfmt::J)>) -> std::io::Result<()> {
    use crate::pyfmt::J;
    std::fs::create_dir_all(path)?;
    let (h, w) = (holds[0].colour.h, holds[0].colour.w);
    let has_depth = holds[0].depth.is_some();
    let half = |x: &[f32]| -> Vec<u8> { x.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect() };
    let rgba = |x: &Img| -> Vec<u8> { half(&Img::from_fn(h, w, 4, |i, c| if c < 3 { x.d[i * 3 + c] } else { 1.0 }).d) };
    let mut frames = Vec::new();
    for (k, hd) in holds.iter().enumerate() {
        let sub = format!("hold-{k:02}");
        std::fs::create_dir_all(path.join(&sub))?;
        std::fs::write(path.join(&sub).join("colour.rgba16f"), rgba(&hd.colour))?;
        std::fs::write(path.join(&sub).join("answer.rgba16f"), rgba(&hd.answer))?;
        std::fs::write(path.join(&sub).join("mvec.rg16f"), half(&hd.mvec.d))?;
        let mut files = vec![
            ("colour".to_string(), J::Str(format!("{sub}/colour.rgba16f"))),
            ("answer".to_string(), J::Str(format!("{sub}/answer.rgba16f"))),
            ("mvec".to_string(), J::Str(format!("{sub}/mvec.rg16f"))),
        ];
        if let Some(d) = &hd.depth {
            std::fs::write(path.join(&sub).join("depth.r32f"), d.to_le_bytes())?;
            files.push(("depth".to_string(), J::Str(format!("{sub}/depth.r32f"))));
        }
        frames.push(J::obj(vec![
            ("index", J::Int(k as i64)),
            ("hold", J::Int(1000 + k as i64)),
            ("time_ms", J::Int(1_000_000 + 16 * k as i64)),
            ("jitter", J::List(vec![J::Float(hd.jitter[0] as f64), J::Float(hd.jitter[1] as f64)])),
            ("exposure", J::Float(hd.exposure)),
            ("history", J::Bool(k > 0)),
            ("files", J::Obj(files)),
        ]));
    }
    let mut manifest = vec![
        ("format", J::Str(DUMP_FORMAT.into())),
        ("version", J::Int(DUMP_VERSION)),
        ("holds", J::Int(holds.len() as i64)),
        ("width", J::Int(w as i64)),
        ("height", J::Int(h as i64)),
        ("path", J::Str("split".into())),
        ("paper_white", J::Float(paper_white)),
        ("mvec_scale", J::List(vec![J::Float(1.0), J::Float(1.0)])),
        ("depth_format", J::Str(if has_depth { "D32_SFLOAT_S8_UINT" } else { "none" }.into())),
        ("reverse_z", J::Null),
        ("game", J::Str(game.into())),
        ("frames", J::List(frames)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect::<Vec<_>>();
    manifest.extend(extra);
    std::fs::write(path.join("manifest.json"), crate::pyfmt::dumps(&J::Obj(manifest), Some(1)))
}

fn nan_to(x: f32, nan: f32, pos: f32, neg: f32) -> f32 {
    if x.is_nan() {
        nan
    } else if x == f32::INFINITY {
        pos
    } else if x == f32::NEG_INFINITY {
        neg
    } else {
        x
    }
}

/// numpy's median of f32 values (the mean of the two middle ones, in f32).
fn median_f32(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Which way the dump's depth runs: (reverse_z or None, reason).
pub fn infer_reverse_z(depths: &[Img]) -> (Option<bool>, String) {
    use crate::pyfmt::fmt_g;
    let share = |x: &Img, v: f32| x.d.iter().filter(|&&d| nan_to(d, 0.5, d, d) == v).count() as f64 / x.d.len() as f64;
    let depths: Vec<&Img> = depths.iter().filter(|x| 1.0 - share(x, 0.0) - share(x, 1.0) >= DEPTH_GEOMETRY_MIN).collect();
    if depths.is_empty() {
        return (None, format!("no hold has {:.0}% of its depth drawn (neither 0 nor 1); cannot tell", 100.0 * DEPTH_GEOMETRY_MIN));
    }
    let total: usize = depths.iter().map(|x| x.d.len()).sum();
    let count = |v: f32| depths.iter().map(|x| x.d.iter().filter(|&&d| nan_to(d, 0.5, d, d) == v).count()).sum::<usize>() as f64 / total as f64;
    let (z0, z1) = (count(0.0), count(1.0));
    if z0.max(z1) > 0.005 && (z0 > 2.0 * z1 || z1 > 2.0 * z0) {
        return (Some(z0 > z1), format!("{:.1}% of depth texels are 0, {:.1}% are 1 (cleared = far)", 100.0 * z0, 100.0 * z1));
    }
    let (mut tops, mut bottoms) = (Vec::new(), Vec::new());
    for x in &depths {
        let q = (x.h / 4).max(1);
        tops.push(median_f32(x.d[..q * x.w].to_vec()));
        bottoms.push(median_f32(x.d[(x.h - q) * x.w..].to_vec()));
    }
    let (top, bottom) = (median_f32(tops) as f64, median_f32(bottoms) as f64);
    if (bottom - top).abs() <= 1e-9 * top.abs().max(bottom.abs()).max(1e-30) {
        return (None, format!("depth the same at the top and bottom of the frame ({}); cannot tell", fmt_g(top, 6)));
    }
    (Some(bottom > top), format!("median depth top quarter {}, bottom quarter {} (the bottom is nearer)", fmt_g(top, 6), fmt_g(bottom, 6)))
}

// ------------------------------------------------------------------------------------------------
// The edit
// ------------------------------------------------------------------------------------------------

pub fn luma_px(x: &[f32]) -> f32 {
    x[0] * LUMA[0] + x[1] * LUMA[1] + x[2] * LUMA[2]
}

pub fn luma(x: &Img) -> Img {
    Img::from_fn(x.h, x.w, 1, |i, _| luma_px(x.px(i)))
}

pub fn to_v(scene: &Img, exposure: f64, paper_white: f64) -> Img {
    let k = (exposure / paper_white) as f32;
    scene.map(|s| nan_to(s, 0.0, 0.0, 0.0).max(0.0) * k)
}

/// ClampEdit: the luma within +-max_luma_edit * max(scale, 1), each channel within chroma_stops of it.
pub fn clamp_edit(e: &Img, p: &Params, scale: f64) -> Img {
    if !(p.max_luma_edit > 0.0 && p.max_luma_edit.is_finite()) {
        return e.clone();
    }
    let limit = (p.max_luma_edit * scale.max(1.0)) as f32;
    let cs = p.chroma_stops as f32;
    let mut out = e.clone();
    for i in 0..e.n() {
        let px = &mut out.d[i * 3..i * 3 + 3];
        let l = luma_px(px);
        let lc = l.clamp(-limit, limit);
        for v in px.iter_mut() {
            *v = (*v + (lc - l)).max(lc - cs).min(lc + cs);
        }
    }
    out
}

pub fn edit_of(vc: &Img, va: &Img, p: &Params) -> Img {
    let eps = p.eps_v as f32;
    clamp_edit(&va.zip(vc, |a, c| ((a.max(0.0) + eps) / (c.max(0.0) + eps)).log2()), p, 1.0)
}

pub fn apply_edit(vc: &Img, edit: &Img, p: &Params) -> Img {
    let eps = p.eps_v as f32;
    vc.zip(edit, |c, e| ((c.max(0.0) + eps) * e.exp2() - eps).max(0.0))
}

pub fn lin_depth(depth: Option<&Img>, reverse_z: bool, h: usize, w: usize) -> Img {
    match depth {
        None => Img::new(h, w, 1, 1.0),
        Some(d) => d.map(|x| {
            let x = nan_to(x, 0.0, 1.0, 0.0);
            1.0 / (if reverse_z { x } else { 1.0 - x }).max(1e-7)
        }),
    }
}

// ------------------------------------------------------------------------------------------------
// Sampling, in pixel units (a pixel centre is i + 0.5)
// ------------------------------------------------------------------------------------------------

pub fn pixel_centres(h: usize, w: usize) -> Img {
    Img::from_fn(h, w, 2, |i, c| if c == 0 { (i % w) as f32 + 0.5 } else { (i / w) as f32 + 0.5 })
}

/// The four taps of a bilinear read at (x, y), edge-clamped, in the shader's order: (flat index, weight).
#[inline]
pub fn taps(x: f32, y: f32, w: usize, h: usize) -> [(usize, f32); 4] {
    let (px, py) = (x - 0.5, y - 0.5);
    let (x0, y0) = (px.floor(), py.floor());
    let (fx, fy) = (px - x0, py - y0);
    let clampi = |v: f32, o: i64, n: usize| ((v as i64).saturating_add(o)).clamp(0, n as i64 - 1) as usize;
    let mut out = [(0, 0.0); 4];
    for (k, o) in out.iter_mut().enumerate() {
        let (ox, oy) = (k & 1, k >> 1);
        let xi = clampi(x0, ox as i64, w);
        let yi = clampi(y0, oy as i64, h);
        let wt = (if ox == 1 { fx } else { 1.0 - fx }) * (if oy == 1 { fy } else { 1.0 - fy });
        *o = (yi * w + xi, wt);
    }
    out
}

/// A linear-clamp sampler read at pos (pixel units).
pub fn sample_bilinear(img: &Img, pos: &Img) -> Img {
    let c = img.c;
    let mut out = Img::new(pos.h, pos.w, c, 0.0);
    for i in 0..pos.n() {
        let t = taps(pos.d[i * 2], pos.d[i * 2 + 1], img.w, img.h);
        for k in 0..c {
            let mut v = img.d[t[0].0 * c + k] * t[0].1;
            for &(idx, wt) in &t[1..] {
                v += img.d[idx * c + k] * wt;
            }
            out.d[i * c + k] = v;
        }
    }
    out
}

pub fn sample_nearest(img: &Img, pos: &Img) -> Img {
    let c = img.c;
    Img::from_fn(pos.h, pos.w, c, |i, k| {
        let xi = (pos.d[i * 2].floor() as i64).clamp(0, img.w as i64 - 1) as usize;
        let yi = (pos.d[i * 2 + 1].floor() as i64).clamp(0, img.h as i64 - 1) as usize;
        img.d[(yi * img.w + xi) * c + k]
    })
}

/// The 3x3 neighbourhood of pixel (y, x), edge-clamped, in scan order (dy outer, dx inner): flat indexes.
#[inline]
pub fn nb_index(y: usize, x: usize, h: usize, w: usize) -> [usize; 9] {
    let mut out = [0; 9];
    let mut n = 0;
    for dy in [-1i64, 0, 1] {
        for dx in [-1i64, 0, 1] {
            let yi = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
            let xi = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
            out[n] = yi * w + xi;
            n += 1;
        }
    }
    out
}

/// The 3x3 minimum and maximum of every channel.
pub fn nb_range(img: &Img) -> (Img, Img) {
    let (mut lo, mut hi) = (img.clone(), img.clone());
    for i in 0..img.n() {
        let nb = nb_index(i / img.w, i % img.w, img.h, img.w);
        for k in 0..img.c {
            let (mut a, mut b) = (f32::INFINITY, f32::NEG_INFINITY);
            let mut nan = false;
            for &j in &nb {
                let v = img.d[j * img.c + k];
                nan |= v.is_nan();
                a = a.min(v);
                b = b.max(v);
            }
            // numpy's min/max propagate NaN.
            lo.d[i * img.c + k] = if nan { f32::NAN } else { a };
            hi.d[i * img.c + k] = if nan { f32::NAN } else { b };
        }
    }
    (lo, hi)
}

/// The 3x3 mean, edge-clamped (numpy's mean over the stacked neighbourhood: summed in scan order, then / 9).
pub fn box3(x: &Img) -> Img {
    Img::from_fn(x.h, x.w, 1, |i, _| {
        let nb = nb_index(i / x.w, i % x.w, x.h, x.w);
        let mut s = 0.0f32;
        for &j in &nb {
            s += x.d[j];
        }
        s / 9.0
    })
}

pub fn print_of(v: &Img, p: &Params) -> Img {
    let eps = p.eps_v as f32;
    Img::from_fn(v.h, v.w, 3, |i, k| {
        let px = v.px(i);
        let q = |c: usize| px[c].max(0.0) + eps;
        match k {
            0 => (luma_px(&[px[0].max(0.0), px[1].max(0.0), px[2].max(0.0)]) + eps).log2(),
            1 => (q(0) / q(1)).log2(),
            _ => (q(2) / q(1)).log2(),
        }
    })
}

/// NearestSurfaceMotion: each pixel takes the vector of the nearest depth in its 3x3 neighbourhood.
pub fn nearest_surface_motion(mv: &Img, lin: &Img) -> Img {
    let mut out = mv.clone();
    for i in 0..lin.n() {
        let nb = nb_index(i / lin.w, i % lin.w, lin.h, lin.w);
        let mut best = 0;
        for k in 1..9 {
            if lin.d[nb[k]] < lin.d[nb[best]] || (lin.d[nb[k]].is_nan() && !lin.d[nb[best]].is_nan()) {
                best = k;
            }
        }
        out.d[i * 2] = mv.d[nb[best] * 2];
        out.d[i * 2 + 1] = mv.d[nb[best] * 2 + 1];
    }
    out
}

/// Reinhard on luma, then gamma 1/2.2.
pub fn display(v: &Img) -> Img {
    let g = (1.0f64 / 2.2) as f32;
    Img::from_fn(v.h, v.w, 3, |i, k| {
        let s = 1.0 / (1.0 + luma_px(v.px(i)));
        (v.d[i * 3 + k] * s).clamp(0.0, 1.0).powf(g)
    })
}

/// Mean of f64 values with numpy's pairwise summation (blocks of 8 unrolled, 128-wide leaves).
pub fn np_sum_f64(a: &[f64]) -> f64 {
    if a.len() < 8 {
        let mut s = -0.0;
        for &x in a {
            s += x;
        }
        s
    } else if a.len() <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= a.len() {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut s = (r[0] + r[1]) + (r[2] + r[3]) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            s += x;
        }
        s
    } else {
        let mut n2 = a.len() / 2;
        n2 -= n2 % 8;
        np_sum_f64(&a[..n2]) + np_sum_f64(&a[n2..])
    }
}

pub fn np_mean_f64(a: &[f64]) -> f64 {
    np_sum_f64(a) / a.len() as f64
}

/// numpy's f32 pairwise sum (the same tree, in f32).
pub fn np_sum_f32(a: &[f32]) -> f32 {
    if a.len() < 8 {
        let mut s = -0.0f32;
        for &x in a {
            s += x;
        }
        s
    } else if a.len() <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= a.len() {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut s = (r[0] + r[1]) + (r[2] + r[3]) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[i..] {
            s += x;
        }
        s
    } else {
        let mut n2 = a.len() / 2;
        n2 -= n2 % 8;
        np_sum_f32(&a[..n2]) + np_sum_f32(&a[n2..])
    }
}

/// mean |disp - disp_answer| of each pixel's channel mean, in f64 (picture_error).
pub fn picture_error(disp: &Img, da: &Img, mask: Option<&[bool]>) -> f64 {
    let d: Vec<f64> = (0..disp.n())
        .filter(|&i| mask.is_none_or(|m| m[i]))
        .map(|i| {
            let mut s = 0.0f64;
            for k in 0..3 {
                s += (disp.d[i * 3 + k] - da.d[i * 3 + k]).abs() as f64;
            }
            s / 3.0
        })
        .collect();
    np_mean_f64(&d)
}

// ------------------------------------------------------------------------------------------------
// Per-hold inputs
// ------------------------------------------------------------------------------------------------

pub struct Frame {
    pub index: i64,
    pub usable: bool,
    pub jitter_known: bool,
    pub mvec_valid: bool,
    pub has_depth: bool,
    pub jitter: [f32; 2],
    pub v_colour: Img,
    pub edit: Img,
    pub lin: Img,
    pub log_luma: Img,
    pub prt: Img,
    pub prt_lo: Img,
    pub prt_hi: Img,
    pub mv: Img,
    pub disp_answer: Img,
    pub base: f64,
}

pub fn prepare(hold: &Hold, dump_white: f64, reverse_z: bool, p: &Params) -> Frame {
    let (h, w) = (hold.colour.h, hold.colour.w);
    let usable = exposure_ok(hold.exposure);
    let e = if usable { hold.exposure } else { 0.0 };
    let vc = to_v(&hold.colour, e, dump_white);
    let va = to_v(&hold.answer, e, dump_white);
    let lin = lin_depth(hold.depth.as_ref(), reverse_z, h, w);
    let prt = print_of(&vc, p);
    let (prt_lo, prt_hi) = nb_range(&prt);
    let mut mv = hold.mvec.map(|x| nan_to(x, 0.0, 0.0, 0.0));
    if p.nearest_surface && hold.depth.is_some() {
        mv = nearest_surface_motion(&mv, &lin);
    }
    let da = display(&va);
    let base = picture_error(&display(&vc), &da, None);
    Frame {
        index: hold.index,
        usable,
        jitter_known: hold.jitter_known,
        mvec_valid: hold.mvec_valid,
        has_depth: hold.depth.is_some(),
        jitter: hold.jitter,
        edit: edit_of(&vc, &va, p),
        v_colour: vc,
        lin,
        log_luma: prt.channel(0),
        prt,
        prt_lo,
        prt_hi,
        mv,
        disp_answer: da,
        base,
    }
}

// ------------------------------------------------------------------------------------------------
// Motion chaining
// ------------------------------------------------------------------------------------------------

fn jitter_delta(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Where a point at pos in hold `cur` was in hold `prev` (the walk).
pub fn chain_step(pos: &Img, cur: &Frame, prev: &Frame, p: &Params) -> Img {
    let read = if p.chain_sample == "bilinear" { sample_bilinear(&cur.mv, pos) } else { sample_nearest(&cur.mv, pos) };
    let dj = jitter_delta(prev.jitter, cur.jitter);
    Img::from_fn(pos.h, pos.w, 2, |i, c| {
        let o = pos.d[i * 2 + c] + read.d[i * 2 + c];
        if p.jitter {
            o + dj[c]
        } else {
            o
        }
    })
}

pub fn on_screen_px(x: f32, y: f32, h: usize, w: usize) -> bool {
    x >= 0.0 && x <= w as f32 && y >= 0.0 && y <= h as f32
}

pub fn on_screen(pos: &Img) -> Vec<bool> {
    (0..pos.n()).map(|i| on_screen_px(pos.d[i * 2], pos.d[i * 2 + 1], pos.h, pos.w)).collect()
}

/// A bilinear read that skips taps of weight 0, so a NaN there does not spread.
pub fn sample_bilinear_taken(img: &Img, pos: &Img) -> Img {
    let c = img.c;
    let mut out = Img::new(pos.h, pos.w, c, 0.0);
    for i in 0..pos.n() {
        for (idx, wt) in taps(pos.d[i * 2], pos.d[i * 2 + 1], img.w, img.h) {
            for k in 0..c {
                out.d[i * c + k] += if wt > 0.0 { img.d[idx * c + k] * wt } else { 0.0 };
            }
        }
    }
    out
}

/// preupscale_cache_motion.comp: acc_t(x) = mv_t(x) + acc_{t-1}(x + mv_t(x) + jitter[t-1] - jitter[t]).
pub fn acc_step(acc_prev: Option<&Img>, cur: &Frame, prev_jitter: [f32; 2], p: &Params) -> Img {
    let (h, w) = (cur.mv.h, cur.mv.w);
    let dj = jitter_delta(prev_jitter, cur.jitter);
    let p1 = Img::from_fn(h, w, 2, |i, c| {
        let centre = if c == 0 { (i % w) as f32 + 0.5 } else { (i / w) as f32 + 0.5 };
        let v = centre + cur.mv.d[i * 2 + c];
        if p.jitter {
            v + dj[c]
        } else {
            v
        }
    });
    let mut acc = match acc_prev {
        None => cur.mv.clone(),
        Some(a) => cur.mv.zip(&sample_bilinear_taken(a, &p1), |m, s| m + s),
    };
    for i in 0..h * w {
        if !on_screen_px(p1.d[i * 2], p1.d[i * 2 + 1], h, w) {
            acc.d[i * 2] = f32::NAN;
            acc.d[i * 2 + 1] = f32::NAN;
        }
    }
    acc
}

/// Where each pixel of `cur` was on the refresh hold `src`, from the accumulated field: (pos, on).
pub fn chained_position(acc: &Img, cur: &Frame, src: &Frame, p: &Params) -> (Img, Vec<bool>) {
    let (h, w) = (acc.h, acc.w);
    let on: Vec<bool> = (0..h * w).map(|i| acc.d[i * 2].is_finite() && acc.d[i * 2 + 1].is_finite()).collect();
    let dj = jitter_delta(src.jitter, cur.jitter);
    let pos = Img::from_fn(h, w, 2, |i, c| {
        let centre = if c == 0 { (i % w) as f32 + 0.5 } else { (i / w) as f32 + 0.5 };
        let v = centre + if on[i] { acc.d[i * 2 + c] } else { 0.0 };
        if p.jitter {
            v + dj[c]
        } else {
            v
        }
    });
    (pos, on)
}

// ------------------------------------------------------------------------------------------------
// One carry
// ------------------------------------------------------------------------------------------------

pub fn can_source(src: &Frame) -> bool {
    src.usable && src.jitter_known
}

pub fn can_carry(src: &Frame, dst: &Frame) -> bool {
    can_source(src) && dst.usable && dst.jitter_known && dst.mvec_valid && dst.has_depth == src.has_depth
}

#[inline]
fn soft(d: f32, tol: f64) -> f32 {
    ((2.0 * tol) as f32 - d) / (tol.max(1e-4) as f32)
}

#[inline]
fn soft_c(d: f32, tol: f64) -> f32 {
    clip01(soft(d, tol))
}

#[inline]
fn clip01(x: f32) -> f32 {
    if x.is_nan() {
        x
    } else {
        x.clamp(0.0, 1.0)
    }
}

#[inline]
fn outside(v: f32, lo: f32, hi: f32) -> f32 {
    np_max(np_max(lo - v, v - hi), 0.0)
}

/// numpy's maximum: NaN wins.
#[inline]
fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

pub struct History {
    pub edit: Img,
    pub conf: Vec<f32>,
    pub log_luma: Vec<f32>,
    pub valid: Vec<f32>,
    pub prt: Img,
    pub meta_y: Vec<f32>,
}

/// ReadHistory: four bilinear taps at pos, each admitted by relative depth; edit and print from the strongest tap.
pub fn read_history(src: &Frame, pos: &Img, lin_c: &Img, p: &Params) -> History {
    let (h, w) = (lin_c.h, lin_c.w);
    let n = h * w;
    let (mut conf, mut ll, mut wsum) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let mut edit = Img::new(h, w, 3, 0.0);
    let mut prt = Img::new(h, w, 3, 0.0);
    let mut meta_y = vec![0.0f32; n];
    let two_tol = (2.0 * p.depth_tol) as f32;
    let div = p.depth_tol.max(1e-6) as f32;
    for i in 0..n {
        let (mut best_w, mut best_i) = (0.0f32, 0usize);
        for (idx, wb) in taps(pos.d[i * 2], pos.d[i * 2 + 1], w, h) {
            let g = src.lin.d[idx];
            let lc = lin_c.d[i];
            let rel = (g - lc).abs() / g.min(lc).max(1e-7);
            let wd = clip01((two_tol - rel) / div);
            let wt = wb * wd;
            conf[i] += wt;
            ll[i] += src.log_luma.d[idx] * wt;
            wsum[i] += wt;
            if wt > best_w {
                best_w = wt;
                best_i = idx;
            }
        }
        let d = if wsum[i] > 1e-4 { wsum[i] } else { 1.0 };
        conf[i] /= d;
        ll[i] /= d;
        if best_w > 0.0 {
            edit.d[i * 3..i * 3 + 3].copy_from_slice(src.edit.px(best_i));
            prt.d[i * 3..i * 3 + 3].copy_from_slice(src.prt.px(best_i));
            meta_y[i] = 1.0;
        }
    }
    let valid = wsum.iter().map(|&x| clip01(x)).collect();
    History { edit, conf, log_luma: ll, valid, prt, meta_y }
}

/// Mode 1's tests against this hold: (v_colour, validity).
pub fn validate(hist: &History, dst: &Frame, on: &[bool], p: &Params) -> (Vec<f32>, Vec<f32>) {
    let n = dst.lin.n();
    let mut vcol = vec![0.0f32; n];
    let mut validity = vec![1.0f32; n];
    let ctol = p.colour_tol.max(1e-3);
    for i in 0..n {
        let lo = dst.prt_lo.px(i);
        let hi = dst.prt_hi.px(i);
        let diff = if p.noise_aware { outside(hist.log_luma[i], lo[0], hi[0]) } else { (dst.log_luma.d[i] - hist.log_luma[i]).abs() };
        vcol[i] = soft_c(diff, ctol);
        if p.fingerprint {
            let s = hist.prt.px(i);
            let m = soft_c(outside(s[0], lo[0], hi[0]), p.print_tol);
            let m = m * soft_c(np_max(outside(s[1], lo[1], hi[1]), outside(s[2], lo[2], hi[2])), p.print_tol);
            validity[i] = if on[i] { clip01(hist.meta_y[i]) * m } else { 0.0 };
        }
    }
    (vcol, validity)
}

/// ReduceToFirstLevel: coverage-weighted 4x4 means; coverage = sum(weight) / 16.
pub fn reduce4(values: &Img, weight: &[f32]) -> (Img, Vec<f32>) {
    let (h, w, c) = (values.h, values.w, values.c);
    let (hh, ww) = (h.div_ceil(4), w.div_ceil(4));
    let mut mean = Img::new(hh, ww, c, 0.0);
    let mut cov = vec![0.0f32; hh * ww];
    for by in 0..hh {
        for bx in 0..ww {
            // numpy sums axis 1 (rows of the block) then axis 3: for each y, x adds; order (y outer, x inner).
            let mut den = 0.0f32;
            let mut num = vec![0.0f32; c];
            for dy in 0..4 {
                for dx in 0..4 {
                    let (y, x) = (by * 4 + dy, bx * 4 + dx);
                    let (wt, inside) = if y < h && x < w { (weight[y * w + x], true) } else { (0.0, false) };
                    den += wt;
                    for (k, nk) in num.iter_mut().enumerate() {
                        let v = if inside { values.d[(y * w + x) * c + k] } else { 0.0 };
                        *nk += v * wt;
                    }
                }
            }
            let j = by * ww + bx;
            for k in 0..c {
                mean.d[j * c + k] = if den > 1e-6 { num[k] / den.max(1e-6) } else { 0.0 };
            }
            cov[j] = den / 16.0;
        }
    }
    (mean, cov)
}

pub struct Level {
    pub edit: Img,
    pub cov: Vec<f32>,
    pub guide: Img,
}

pub fn build_levels(edit: &Img, weight: &[f32], guide: &Img) -> Vec<Level> {
    let mut vals = Img::stack(&[edit, guide]);
    let mut wgt = weight.to_vec();
    let mut levels = Vec::new();
    for _ in 0..3 {
        let (m, c) = reduce4(&vals, &wgt);
        levels.push(Level {
            edit: Img::from_fn(m.h, m.w, 3, |i, k| m.d[i * 5 + k]),
            cov: c.clone(),
            guide: Img::from_fn(m.h, m.w, 2, |i, k| m.d[i * 5 + 3 + k]),
        });
        vals = m;
        wgt = c;
    }
    levels
}

/// The separable taps of an lh x lw grid at the frame's pixel centres: per axis (index, weight) pairs.
fn axis_taps(n: usize, ln: usize) -> Vec<[(usize, f32); 2]> {
    (0..n)
        .map(|i| {
            let p = (i as f32 + 0.5) / n as f32 * ln as f32 - 0.5;
            let p0 = p.floor();
            let f = p - p0;
            let i0 = p0 as i64;
            let cl = |v: i64| v.clamp(0, ln as i64 - 1) as usize;
            [(cl(i0), 1.0 - f), (cl(i0 + 1), f)]
        })
        .collect()
}

/// Mode 4's low band: SurfaceLow (same-surface fill) or the push-pull pyramid.
pub fn low_band(levels: &[Level], guide: &Img, p: &Params) -> Img {
    let (h, w) = (guide.h, guide.w);
    let bilateral = |lvl: usize, sd: f64, sl: f64, use_guide: bool| -> (Img, Vec<f32>) {
        let l = &levels[lvl];
        let (lh, lw) = (l.edit.h, l.edit.w);
        let (tx, ty) = (axis_taps(w, lw), axis_taps(h, lh));
        let (sd, sl) = (sd as f32, sl as f32);
        let mut acc = Img::new(h, w, 3, 0.0);
        let mut ws = vec![0.0f32; h * w];
        for (y, tyy) in ty.iter().enumerate() {
            for (x, txx) in tx.iter().enumerate() {
                let i = y * w + x;
                for k in 0..4 {
                    let (ox, oy) = (k & 1, k >> 1);
                    let (xi, wx) = txx[ox];
                    let (yi, wy) = tyy[oy];
                    let idx = yi * lw + xi;
                    let mut wt = (wy * wx) * l.cov[idx];
                    if use_guide {
                        let g = l.guide.px(idx);
                        let dd = (g[0] - guide.d[i * 2]) / sd;
                        let dl = (g[1] - guide.d[i * 2 + 1]) / sl;
                        wt *= (-(dd * dd + dl * dl)).exp();
                    }
                    for c in 0..3 {
                        acc.d[i * 3 + c] += l.edit.d[idx * 3 + c] * wt;
                    }
                    ws[i] += wt;
                }
            }
        }
        (acc, ws)
    };
    let est = |acc: &Img, ws: &[f32], fallback: &Img| -> Img {
        Img::from_fn(h, w, 3, |i, c| if ws[i] > 1e-6 { acc.d[i * 3 + c] / ws[i].max(1e-6) } else { fallback.d[i * 3 + c] })
    };
    if p.surface_fill {
        let mut low = Img::new(h, w, 3, 0.0);
        let mut have = vec![0.0f32; h * w];
        for lvl in [2usize, 1, 0] {
            let f = (1 + lvl) as f64;
            let (acc, ws) = bilateral(lvl, p.sigma_depth * f, p.sigma_luma * f, true);
            let e = est(&acc, &ws, &low);
            for i in 0..h * w {
                let a = clip01(ws[i] * 4.0);
                for c in 0..3 {
                    let l = low.d[i * 3 + c];
                    low.d[i * 3 + c] = l + (e.d[i * 3 + c] - l) * a;
                }
                have[i] += (1.0 - have[i]) * a;
            }
        }
        return low.zip_px(&Img { h, w, c: 1, d: have }, |l, a| l * a);
    }
    let zero = Img::new(h, w, 3, 0.0);
    let (a3, w3) = bilateral(2, 1.0, 1.0, false);
    let e3 = est(&a3, &w3, &zero);
    let (a2, w2) = bilateral(1, 1.0, 1.0, false);
    let e2 = est(&a2, &w2, &zero);
    let mut low = Img::from_fn(h, w, 3, |i, c| if w3[i] > 1e-6 { e3.d[i * 3 + c] } else { 0.0 });
    for i in 0..h * w {
        let a = clip01(w2[i] * 4.0);
        for c in 0..3 {
            let l = low.d[i * 3 + c];
            low.d[i * 3 + c] = l + (e2.d[i * 3 + c] - l) * a;
        }
    }
    let (a1, w1) = bilateral(0, p.sigma_depth, p.sigma_luma, p.bilateral);
    let e1 = est(&a1, &w1, &low);
    for i in 0..h * w {
        let a = clip01(w1[i] * 4.0);
        for c in 0..3 {
            let l = low.d[i * 3 + c];
            low.d[i * 3 + c] = l + (e1.d[i * 3 + c] - l) * a;
        }
    }
    low
}

pub struct Carried {
    pub naive: Img,
    pub low: Img,
    pub cache: Img,
    pub valid: Vec<f32>,
    pub validity: Vec<f32>,
    pub v_colour: Vec<f32>,
    pub conf: Vec<f32>,
    pub history: Img,
    pub low_band: Img,
}

/// The guide the band split reads: log2 depth and log luma.
pub fn guide_of(f: &Frame) -> Img {
    Img::from_fn(f.lin.h, f.lin.w, 2, |i, c| if c == 0 { f.lin.d[i].max(1e-7).log2() } else { f.log_luma.d[i] })
}

/// The refresh hold's edit carried to dst at the chained positions, every mode at once.
pub fn carry(src: &Frame, dst: &Frame, pos: &Img, on: &[bool], p: &Params) -> Carried {
    let n = dst.lin.n();
    let mut hist = read_history(src, pos, &dst.lin, p);
    for i in 0..n {
        let o = if on[i] { 1.0 } else { 0.0 };
        for c in 0..3 {
            hist.edit.d[i * 3 + c] *= o;
        }
        hist.conf[i] *= o;
        hist.log_luma[i] *= o;
        hist.valid[i] *= o;
        hist.meta_y[i] *= o;
    }
    let (v_colour, validity) = validate(&hist, dst, on, p);
    let cf = p.colour_floor as f32;
    let one_cf = (1.0 - p.colour_floor) as f32;
    let hd = p.high_decay as f32;
    let weight: Vec<f32> = (0..n).map(|i| hist.valid[i] * validity[i]).collect();
    let conf: Vec<f32> = (0..n)
        .map(|i| {
            let doubt = (cf + one_cf * v_colour[i]) * clip01(2.0 * hist.valid[i] - 1.0) * hd * validity[i];
            clip01(hist.conf[i] * doubt)
        })
        .collect();
    let guide = guide_of(dst);
    let levels = build_levels(&hist.edit, &weight, &guide);
    let low = low_band(&levels, &guide, p);
    let scale = p.low_gain.max(p.high_gain);
    let (lg, hg) = (p.low_gain as f32, p.high_gain as f32);
    let mixed = Img::from_fn(dst.lin.h, dst.lin.w, 3, |i, c| {
        let l = low.d[i * 3 + c];
        lg * l + (hg * conf[i]) * (hist.edit.d[i * 3 + c] - l)
    });
    let cache = clamp_edit(&mixed, p, scale);
    let s = sample_bilinear(&src.edit, pos);
    let naive = Img::from_fn(s.h, s.w, 3, |i, c| s.d[i * 3 + c] * if on[i] { 1.0 } else { 0.0 });
    Carried { naive, low: clamp_edit(&low.map(|x| lg * x), p, scale), cache, valid: hist.valid, validity, v_colour, conf, history: hist.edit, low_band: low }
}

// ------------------------------------------------------------------------------------------------
// Metrics
// ------------------------------------------------------------------------------------------------

pub struct PairMetrics {
    pub err: f64,
    pub p995: f64,
    pub worse: f64,
    pub harm: f64,
    pub pic: f64,
}

/// numpy's percentile (linear), on f32 data, interpolated in f32 as numpy 2 does for a float32 array.
pub fn percentile_f32(v: &[f32], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    if s.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let idx = q / 100.0 * (s.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    let t = (idx - lo as f64) as f32;
    let (a, b) = (s[lo], s[hi]);
    let diff = b - a;
    // numpy's _lerp: a + diff*t, or b - diff*(1-t) where t >= 0.5.
    (if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }) as f64
}

pub fn pair_metrics(shown: &Img, dst: &Frame, p: &Params) -> PairMetrics {
    let n = shown.n();
    let e: Vec<f32> = (0..n)
        .map(|i| luma_px(&[shown.d[i * 3] - dst.edit.d[i * 3], shown.d[i * 3 + 1] - dst.edit.d[i * 3 + 1], shown.d[i * 3 + 2] - dst.edit.d[i * 3 + 2]]).abs())
        .collect();
    let excess: Vec<f32> = (0..n).map(|i| np_max(0.0, e[i] - luma_px(dst.edit.px(i)).abs())).collect();
    let pic = picture_error(&display(&apply_edit(&dst.v_colour, shown, p)), &dst.disp_answer, None);
    let ef: Vec<f64> = e.iter().map(|&x| x as f64).collect();
    let xf: Vec<f64> = excess.iter().map(|&x| x as f64).collect();
    let worse: Vec<f64> = excess.iter().map(|&x| if x > 0.15 { 1.0 } else { 0.0 }).collect();
    PairMetrics { err: np_mean_f64(&ef), p995: percentile_f32(&e, 99.5), worse: np_mean_f64(&worse), harm: np_mean_f64(&xf), pic }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_weights_sum_to_one_and_clamp() {
        let t = taps(0.25, 3.75, 4, 4);
        let s: f32 = t.iter().map(|x| x.1).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert_eq!(t[0].0, 3 * 4); // x clamped to 0, y floor(3.25) = 3
    }

    #[test]
    fn clamp_keeps_luma_within_limit() {
        let e = Img { h: 1, w: 1, c: 3, d: vec![4.0, 4.0, 4.0] };
        let c = clamp_edit(&e, &Params::default(), 1.0);
        assert!((luma_px(&c.d) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn pairwise_sum_matches_sequential_on_small_input() {
        assert_eq!(np_sum_f64(&[1.0, 2.0, 3.0]), 6.0);
        let v: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(np_sum_f64(&v), 499500.0);
    }
}
