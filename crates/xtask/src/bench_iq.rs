//! `cargo xtask bench-iq` (it replaced scripts/bench-iq.py, flag for flag). The text below is also the usage it prints.
//!
//! The numbers are made the way numpy makes them, so the printed digits match: float32 luma, numpy's pairwise sum for
//! every mean, and numpy's own float32 `log`/`exp` kernels. The one thing not reproducible is the order of the values
//! `np.partition` leaves in front of the trimmed mean (an implementation detail of numpy's sort), which can move the
//! float32 sum in its last bits.
#![doc = include_str!("bench_iq/usage.txt")]

use std::fs::File;
use std::io::BufReader;
use std::ops::{Add, Div, Mul, Sub};
use std::process::ExitCode;
use std::sync::LazyLock;

use regex::Regex;

pub(crate) mod pyish;
mod selftest;

use crate::py::repr_str;
use pyish::{fixed, glob_dir, is_dir, join, normpath_basename, py_int, py_sum};

const USAGE: &str = include_str!("bench_iq/usage.txt");

pub(crate) const WINDOW: i64 = 9;
/// The share of pixels with the largest change left out of the flicker mean.
const EXCLUDE: f64 = 0.05;
/// x geometric mean luma.
const FLOOR: f64 = 0.05;
/// Largest frame-to-frame change (stops) of a pixel counted as static for detail.
const STATIC_STOPS: f64 = 0.05;
/// A pair whose trimmed change is over this (stops) can be a cut, and the window starts SETTLE frames after the first one.
const CUT_STOPS: f64 = 0.05;
/// The window starts on the new shot's 31st rendered frame: DLSS's history converges over about 30 rendered frames, and
/// GTA's camera starts moving again 47-56 frames after the still shot's cut in the slower runs. Counted in rendered
/// frames: the series holds presented frames, `settle` converts.
pub(crate) const SETTLE: i64 = 30;
const REC709: (f32, f32, f32) = (0.2126, 0.7152, 0.0722);
/// The darkest non-black 8-bit sRGB code, linear: black pixels count as this in the geometric mean.
const DARKEST: f64 = 1.0 / 255.0 / 12.92;

/// The layer's line for a series capture (crates/layer/src/series.rs; cargo xtask gta-bench reads the same). The directory is the
/// rest of the line, spaces included.
static SERIES_LOG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)\[capture\] capturing the next \d+ presents into (.+?)[ \t\r]*$").expect("static regex"));

/// What went wrong, split the way bench-report needs it: a missing or unreadable frame (`Os`) is caught there and printed on
/// stdout, anything else ends the program with the message on stderr.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum IqError {
    Os(String),
    Other(String),
}

impl std::fmt::Display for IqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IqError::Os(s) | IqError::Other(s) => f.write_str(s),
        }
    }
}

type Res<T> = Result<T, IqError>;

fn other<T>(msg: &str) -> Res<T> {
    Err(IqError::Other(msg.to_string()))
}

/// The float type of the arrays: float32 for real frames (like numpy's `float32` luma), float64 in the self-test scenes.
pub(crate) trait Fl: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {
    const ZERO: Self;
    /// A Python float or int used with an array of this type (numpy converts it to the array's type).
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn exp(self) -> Self;
    fn abs(self) -> Self;
    fn is_nan(self) -> bool;
}

impl Fl for f32 {
    const ZERO: f32 = 0.0;
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn ln(self) -> f32 {
        np_ln_f32(self)
    }
    fn log2(self) -> f32 {
        f32::log2(self)
    }
    fn exp(self) -> f32 {
        np_exp_f32(self)
    }
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl Fl for f64 {
    const ZERO: f64 = 0.0;
    fn from_f64(v: f64) -> f64 {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    fn log2(self) -> f64 {
        f64::log2(self)
    }
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

/// numpy's float32 `log` (the AVX2/FMA3 kernel in numpy/_core/src/umath/loops_exponent_log.dispatch.c.src, which the
/// AVX512F kernel matches): its own rational polynomial, not libm, so the same one is used to get the same bits.
// The coefficients are numpy's decimal literals, kept as written.
#[allow(clippy::excessive_precision)]
fn np_ln_f32(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::NAN;
    }
    if x_in < 0.0 {
        return -f32::NAN;
    }
    if x_in == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x_in == f32::INFINITY {
        return f32::INFINITY;
    }
    if x_in < f32::MIN_POSITIVE {
        // denormal: numpy scales it by 2^100 first; libm is within the kernel's own error here
        return x_in.ln();
    }
    let bits = x_in.to_bits();
    let mut exponent = ((bits >> 23) as i32 - 0x7e) as f32;
    let mut x = f32::from_bits((bits & 0x7f_ffff) | (126 << 23));
    if x <= std::f32::consts::FRAC_1_SQRT_2 {
        x += x;
        exponent -= 1.0;
    }
    x -= 1.0;
    let num = 2.589979117907922693523e-02_f32.mul_add(x, 3.808837741388407920751e-01);
    let num = num.mul_add(x, 1.480000633576506585156e+00);
    let num = num.mul_add(x, 2.112677543073053063722e+00);
    let num = num.mul_add(x, 9.999999999999998702752e-01);
    let num = num.mul_add(x, 0.0);
    let den = 5.875095403124574342950e-03_f32.mul_add(x, 1.546476374983906719538e-01);
    let den = den.mul_add(x, 9.864942958519418960339e-01);
    let den = den.mul_add(x, 2.453006071784736363091e+00);
    let den = den.mul_add(x, 2.612677543073109236779e+00);
    let den = den.mul_add(x, 1.0);
    exponent.mul_add(std::f32::consts::LN_2, num / den)
}

/// numpy's float32 `exp` (same file as `np_ln_f32`).
#[allow(clippy::excessive_precision)]
fn np_exp_f32(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::NAN;
    }
    if x_in >= 88.722_84 {
        return f32::INFINITY;
    }
    if x_in <= -103.972_08 {
        return 0.0;
    }
    let magic = 12_582_912.0_f32; // 0x1.8p23, rounds to the nearest integer
    let q = (x_in * std::f32::consts::LOG2_E + magic) - magic;
    if q <= -125.0 {
        // the kernel builds denormals specially; libm is within its error there
        return x_in.exp();
    }
    let x = q.mul_add(-6.93145752e-1_f32, x_in);
    let x = q.mul_add(-1.42860677e-6_f32, x);
    let num = 5.082762527590693718096e-04_f32.mul_add(x, 6.757896990527504603057e-03);
    let num = num.mul_add(x, 5.114512081637298353406e-02);
    let num = num.mul_add(x, 2.473615434895520810817e-01);
    let num = num.mul_add(x, 7.257664613233124478488e-01);
    let num = num.mul_add(x, 9.999999999980870924916e-01);
    let den = 2.159509375685829852307e-02_f32.mul_add(x, -2.742335390411667452936e-01);
    let den = den.mul_add(x, 1.0);
    f32::from_bits((num / den).to_bits().wrapping_add(((q as i32) << 23) as u32))
}

/// A frame's luma, row by row.
#[derive(Clone, Debug)]
pub(crate) struct Img<T> {
    pub h: usize,
    pub w: usize,
    pub d: Vec<T>,
}

impl<T: Fl> Img<T> {
    fn like(&self, d: Vec<T>) -> Img<T> {
        Img { h: self.h, w: self.w, d }
    }
}

/// numpy's `add.reduce` on a contiguous array: pairwise summation in blocks of eight.
fn pairwise<T: Fl>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut res = T::ZERO;
        for &x in a {
            res = res + x;
        }
        res
    } else if n <= 128 {
        let mut r = [T::ZERO; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj = *rj + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[n - (n % 8)..] {
            res = res + x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `array.mean()`.
fn np_mean<T: Fl>(a: &[T]) -> T {
    pairwise(a) / T::from_f64(a.len() as f64)
}

/// The series' directory: `series-*` inside the run, else the run itself when it is one, else the one its launch log names.
pub(crate) fn series_dir(run: &str) -> Option<String> {
    let dirs = glob_dir(run, |n| n.starts_with("series-"));
    if let Some(d) = dirs.into_iter().find(|d| is_dir(d)) {
        return Some(d);
    }
    if is_dir(run) && normpath_basename(run).starts_with("series-") {
        return Some(run.to_string());
    }
    let log = join(run, "launch.log");
    if pyish::exists(&log) {
        let text = pyish::read_text_lossy(&log).ok()?;
        let found = SERIES_LOG.captures_iter(&text).last()?;
        let dir = found[1].to_string();
        if is_dir(&dir) {
            return Some(dir);
        }
    }
    None
}

/// The series' presented frames in order. On either present path `composited` is what was presented (the same as
/// `original` when nothing was composited after the upscaler).
fn frame_files(directory: &str) -> Vec<String> {
    glob_dir(directory, |n| n.ends_with("-composited.png"))
}

/// The 8-bit codes' linear values, as numpy computes `srgb_to_linear` on float32.
fn srgb_table() -> [f32; 256] {
    let mut t = [0.0_f32; 256];
    for (code, out) in t.iter_mut().enumerate() {
        let v = code as f32 / 255.0_f32;
        *out = if v <= 0.04045_f32 { v / 12.92_f32 } else { ((v + 0.055_f32) / 1.055_f32).powf(2.4_f32) };
    }
    t
}

/// The frame's Rec.709 luma of the linearised 8-bit sRGB pixels (`Image.convert("RGB")`, then numpy float32).
fn load_luma(path: &str) -> Res<Img<f32>> {
    let file = File::open(path).map_err(|e| IqError::Os(pyish::os_error(&e, path)))?;
    let unidentified = || IqError::Os(format!("cannot identify image file {}", repr_str(path)));
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|_| unidentified())?;
    let mut buf = vec![0; reader.output_buffer_size().ok_or_else(unidentified)?];
    let info = reader.next_frame(&mut buf).map_err(|_| IqError::Os("broken data stream when reading image file".to_string()))?;
    let (channels, gray) = match info.color_type {
        png::ColorType::Grayscale => (1, true),
        png::ColorType::GrayscaleAlpha => (2, true),
        png::ColorType::Rgb => (3, false),
        png::ColorType::Rgba => (4, false),
        png::ColorType::Indexed => return Err(unidentified()),
    };
    let lin = srgb_table();
    let (w, h) = (info.width as usize, info.height as usize);
    let (kr, kg, kb) = REC709;
    let mut d = Vec::with_capacity(w * h);
    for px in buf[..w * h * channels].chunks_exact(channels) {
        let (r, g, b) = if gray { (px[0], px[0], px[0]) } else { (px[0], px[1], px[2]) };
        d.push(lin[r as usize] * kr + lin[g as usize] * kg + lin[b as usize] * kb);
    }
    Ok(Img { h, w, d })
}

/// The log-mean luma of the first frame times FLOOR: the additive floor of the log.
fn floor_of<T: Fl>(y: &Img<T>) -> f64 {
    let dark = T::from_f64(DARKEST);
    let logs: Vec<T> = y.d.iter().map(|&v| (if v >= dark || v.is_nan() { v } else { dark }).ln()).collect();
    FLOOR * np_mean(&logs).exp().to_f64()
}

fn log_frames<T: Fl>(lumas: &[Img<T>], floor: f64) -> Vec<Img<T>> {
    let f = T::from_f64(floor);
    lumas.iter().map(|y| y.like(y.d.iter().map(|&v| (v + f).log2()).collect())).collect()
}

fn pair_diff<T: Fl>(a: &Img<T>, b: &Img<T>) -> Res<Img<T>> {
    if (a.h, a.w) != (b.h, b.w) {
        return other("the frames differ in size");
    }
    Ok(a.like(a.d.iter().zip(&b.d).map(|(&x, &y)| (y - x).abs()).collect()))
}

fn pair_changes<T: Fl>(logs: &[Img<T>]) -> Res<Vec<Img<T>>> {
    logs.windows(2).map(|p| pair_diff(&p[0], &p[1])).collect()
}

/// Mean of the smallest (1 - EXCLUDE) of `d`.
pub(crate) fn trimmed_mean<T: Fl>(d: &[T]) -> f64 {
    let keep = (d.len() as f64 * (1.0 - EXCLUDE)) as usize;
    if keep == 0 {
        return 0.0;
    }
    let mut v = d.to_vec();
    v.select_nth_unstable_by(keep - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    np_mean(&v[..keep]).to_f64()
}

pub(crate) fn percent(stops: f64) -> f64 {
    100.0 * (2.0_f64.powf(stops) - 1.0)
}

/// Flicker of consecutive frames' luma, in stops (`percent` converts).
pub(crate) fn flicker<T: Fl>(lumas: &[Img<T>]) -> Res<f64> {
    let Some(first) = lumas.first() else { return other("no frames to compare") };
    let logs = log_frames(lumas, floor_of(first));
    let pairs = pair_changes(&logs)?;
    if pairs.is_empty() {
        return other("division by zero");
    }
    let each: Vec<f64> = pairs.iter().map(|d| trimmed_mean(&d.d)).collect();
    Ok(py_sum(&each) / pairs.len() as f64)
}

fn laplacian<T: Fl>(l: &Img<T>) -> Vec<T> {
    let (h, w) = (l.h, l.w);
    let mut lap = vec![T::ZERO; h * w];
    let four = T::from_f64(4.0);
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            lap[i] = four * l.d[i] - l.d[i - w] - l.d[i + w] - l.d[i - 1] - l.d[i + 1];
        }
    }
    lap
}

/// Pixels whose largest frame-to-frame change is under STATIC_STOPS, together with their four neighbours (the Laplacian
/// reads them), the border left out.
fn static_mask<T: Fl>(logs: &[Img<T>]) -> Res<Vec<bool>> {
    let pairs = pair_changes(logs)?;
    let Some(first) = pairs.first() else { return other("no frames to stack") };
    let (h, w) = (first.h, first.w);
    let limit = T::from_f64(STATIC_STOPS);
    let mut still = vec![true; h * w];
    for (i, s) in still.iter_mut().enumerate() {
        let mut m = pairs[0].d[i];
        for p in &pairs[1..] {
            let v = p.d[i];
            if v > m || v.is_nan() {
                m = v;
            }
        }
        *s = m < limit;
    }
    let mut out = vec![false; h * w];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            out[i] = still[i] && still[i - w] && still[i + w] && still[i - 1] && still[i + 1];
        }
    }
    Ok(out)
}

/// Added detail: Laplacian energy of log luma with the model over without it, on the pixels static in both windows. None
/// when no pixel is static in both.
pub(crate) fn detail<T: Fl>(lumas: &[Img<T>], base_lumas: &[Img<T>]) -> Res<Option<f64>> {
    let (Some(first), Some(base_first)) = (lumas.first(), base_lumas.first()) else { return other("no frames to compare") };
    if (first.h, first.w) != (base_first.h, base_first.w) {
        return other("the frames differ in size");
    }
    let floor = floor_of(base_first);
    let (logs, base) = (log_frames(lumas, floor), log_frames(base_lumas, floor));
    let mask = static_mask(&logs)?.iter().zip(static_mask(&base)?).map(|(&a, b)| a && b).collect::<Vec<bool>>();
    if !mask.iter().any(|&m| m) {
        return Ok(None);
    }
    let energy = |ls: &[Img<T>]| -> f64 {
        let means: Vec<T> = ls
            .iter()
            .map(|l| {
                let lap = laplacian(l);
                let sq: Vec<T> = lap.iter().zip(&mask).filter(|(_, &m)| m).map(|(&v, _)| v * v).collect();
                np_mean(&sq)
            })
            .collect();
        np_mean(&means).to_f64()
    };
    let (e, b) = (energy(&logs), energy(&base));
    if b == 0.0 {
        return other("division by zero");
    }
    Ok(Some(e / b))
}

/// The window's frames for frame-generation multiplier `fg`: whole cycles, at least 8 pairs.
pub(crate) fn window_frames(fg: i64) -> i64 {
    let fg = fg.max(1);
    -((-8_i64).div_euclid(fg)) * fg + 1
}

/// SETTLE rendered frames, in presented frames at multiplier `fg`.
pub(crate) fn settle(fg: i64) -> i64 {
    SETTLE * fg.max(1)
}

/// `(start, cut)`: the window's first frame, from `t`, the trimmed change of each pair of the series, `settle(fg)`
/// presented frames after the series' first cut (`cut` true), or `(0, false)` without a cut. A cut is a pair whose trimmed
/// change is over CUT_STOPS and over 3 times the pair before it (a camera that starts or stops moving changes gradually).
/// Counting from the cut fixes the moment: GTA's benchmark camera starts a pass up to about a second earlier or later from
/// run to run, so the series is captured across the cut. SETTLE leaves out the frames right after it, where DLSS's history
/// is still converging (their change decays over about 30 frames; the window starts on the new shot's 31st frame). None
/// when the series ends too soon after the cut.
pub(crate) fn anchor(t: &[f64], frames: i64, fg: i64) -> Option<(i64, bool)> {
    for i in 1..t.len() {
        if t[i] > CUT_STOPS && t[i] > 3.0 * t[i - 1] {
            let start = i as i64 + 1 + settle(fg);
            return if start + frames <= t.len() as i64 + 1 { Some((start, true)) } else { None };
        }
    }
    Some((0, false))
}

/// (mean, trimmed mean) of the change of each consecutive pair of `files`, two frames in memory at a time (a series of 400
/// frames at 1440p does not fit as a whole).
fn pair_stats(files: &[String]) -> Res<Vec<(f64, f64)>> {
    let mut prev: Option<Img<f32>> = None;
    let mut floor: Option<f64> = None;
    let mut out = Vec::new();
    for f in files {
        let y = load_luma(f)?;
        let fl = *floor.get_or_insert_with(|| floor_of(&y));
        let flt = f32::from_f64(fl);
        let l = y.like(y.d.iter().map(|&v| (v + flt).log2()).collect());
        if let Some(p) = &prev {
            let d = pair_diff(p, &l)?;
            out.push((np_mean(&d.d).to_f64(), trimmed_mean(&d.d)));
        }
        prev = Some(l);
    }
    Ok(out)
}

/// Python's `seq[start:stop]` as a range of a list of `len` items.
pub(crate) fn py_slice(len: usize, start: i64, stop: i64) -> std::ops::Range<usize> {
    let norm = |i: i64| -> usize {
        let i = if i < 0 { i + len as i64 } else { i };
        i.clamp(0, len as i64) as usize
    };
    let (a, b) = (norm(start), norm(stop));
    a..b.max(a)
}

/// A window of luma frames, or the one-line reason there is none.
pub(crate) enum Window {
    Frames(Vec<Img<f32>>, String),
    Msg(String),
}

/// `(lumas, where)`: the run's `frames` lumas from `start` (None: from the cut, `anchor`) and where the window is ("from
/// the cut", "from the series' start (no cut)", "from frame K"), or an error string.
pub(crate) fn window(run: &str, start: Option<i64>, frames: i64, fg: i64) -> Res<Window> {
    let Some(d) = series_dir(run) else { return Ok(Window::Msg(format!("no series in {run}"))) };
    let files = frame_files(&d);
    let (start, place) = match start {
        None => {
            let found = if files.len() as i64 > frames {
                let trimmed: Vec<f64> = pair_stats(&files)?.into_iter().map(|(_, t)| t).collect();
                anchor(&trimmed, frames, fg)
            } else {
                Some((0, false))
            };
            let Some((start, cut)) = found else {
                return Ok(Window::Msg(format!("{d}: {} frames, the series ends before {} + {frames} frames after its cut", files.len(), settle(fg))));
            };
            (start, if cut { "from the cut".to_string() } else { "from the series' start (no cut)".to_string() })
        }
        Some(s) => (s, format!("from frame {s}")),
    };
    if (files.len() as i64) < start + frames {
        return Ok(Window::Msg(format!("{d}: {} frames, the window needs {}", files.len(), start + frames)));
    }
    let lumas = files[py_slice(files.len(), start, start + frames)].iter().map(|f| load_luma(f)).collect::<Res<Vec<_>>>()?;
    Ok(Window::Frames(lumas, place))
}

/// Mean change (stops) of every consecutive pair of the whole series, and the stillest window.
fn scan(run: &str, frames: i64, fg: i64) -> Res<String> {
    let d = series_dir(run);
    let stats = match &d {
        Some(d) => pair_stats(&frame_files(d))?,
        None => Vec::new(),
    };
    let Some(d) = d.filter(|_| !stats.is_empty()) else { return Ok(format!("{run}: no series of 2 frames or more")) };
    let trimmed: Vec<f64> = stats.iter().map(|&(_, t)| t).collect();
    let mut out = vec![format!("{run}: {} frames in {d}", stats.len() + 1), "  pair  mean%  flicker%".to_string()];
    for (i, &(m, t)) in stats.iter().enumerate() {
        out.push(format!("  {i:4} {:>6} {:>8}", fixed(percent(m), 3), fixed(percent(t), 3)));
    }
    let k = frames - 1;
    if k < 1 {
        return other("division by zero");
    }
    if trimmed.len() as i64 >= k {
        let ku = k as usize;
        let mut best = (0, f64::INFINITY);
        for s in 0..=trimmed.len() - ku {
            let sum = py_sum(&trimmed[s..s + ku]);
            if s == 0 || sum < best.1 {
                best = (s, sum);
            }
        }
        let best_sum = py_sum(&trimmed[best.0..best.0 + ku]);
        out.push(format!("  stillest {frames}-frame window: frames {}..{}, flicker {}%", best.0, best.0 + ku, fixed(percent(best_sum / k as f64), 3)));
    }
    match anchor(&trimmed, frames, fg) {
        None => out.push("  window after the cut: none (the series ends too soon)".to_string()),
        Some((_, false)) => out.push(format!("  no cut: the window is frames 0..{k}")),
        Some((a, true)) => out.push(format!("  window after the cut: frames {a}..{}", a + k)),
    }
    Ok(out.join("\n"))
}

/// `flicker X% detail xY` for bench-report, or None when the run has no series. `fg`: the run's frame-generation
/// multiplier (presented per rendered frame).
pub(crate) fn summary(run: &str, baseline: Option<&str>, start: Option<i64>, frames: Option<i64>, fg: i64) -> Res<Option<String>> {
    if series_dir(run).is_none() {
        return Ok(None);
    }
    let frames = frames.filter(|&f| f != 0).unwrap_or_else(|| window_frames(fg));
    let (lumas, place) = match window(run, start, frames, fg)? {
        Window::Msg(m) => return Ok(Some(m)),
        Window::Frames(l, p) => (l, p),
    };
    let mut text = format!("flicker {}%", fixed(percent(flicker(&lumas)?), 2));
    if place.starts_with("from the series' start") {
        text += " (no cut)";
    }
    if let Some(baseline) = baseline.filter(|b| !b.is_empty()) {
        let base = match window(baseline, start, frames, fg)? {
            Window::Msg(m) => return Ok(Some(format!("{text} (detail: {m})"))),
            Window::Frames(b, _) => b,
        };
        if (base[0].h, base[0].w) != (lumas[0].h, lumas[0].w) {
            return Ok(Some(format!("{text} (detail: the baseline is {}x{})", base[0].w, base[0].h)));
        }
        match detail(&lumas, &base)? {
            None => text += " detail -",
            Some(x) => text += &format!(" detail x{}", fixed(x, 2)),
        }
    }
    Ok(Some(text))
}

/// The argument of an option that takes one, from `it`.
fn int_opt(v: &Option<String>) -> Result<i64, String> {
    match v {
        None => Err("an integer is missing".to_string()),
        Some(s) => py_int(s).ok_or_else(|| format!("not an integer: {}", repr_str(s))),
    }
}

fn run(argv: &[String]) -> Result<ExitCode, String> {
    let (mut baseline, mut frames_opt, mut start_opt, mut fg_opt) = (None, None, None, Some("1".to_string()));
    let mut runs: Vec<&str> = Vec::new();
    let mut want_scan = false;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--baseline" => baseline = it.next().cloned(),
            "--frames" => frames_opt = Some(it.next().cloned()),
            "--start" => start_opt = Some(it.next().cloned()),
            "--fg" => fg_opt = it.next().cloned(),
            "--scan" => want_scan = true,
            other => runs.push(other),
        }
    }
    if runs.is_empty() {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(2));
    }
    let fg = int_opt(&fg_opt)?;
    let frames = match frames_opt.flatten().filter(|s| !s.is_empty()) {
        Some(s) => int_opt(&Some(s))?,
        None => window_frames(fg),
    };
    let start = match start_opt {
        None | Some(None) => None,
        Some(s) => Some(int_opt(&s)?),
    };
    let ie = |e: IqError| e.to_string();
    if want_scan {
        for run in &runs {
            println!("{}", scan(run, frames, fg).map_err(ie)?);
        }
        return Ok(ExitCode::SUCCESS);
    }
    let mut values: Vec<f64> = Vec::new();
    let mut base_cache: Option<Window> = None;
    for run in &runs {
        let (lumas, place) = match window(run, start, frames, fg).map_err(ie)? {
            Window::Msg(m) => {
                println!("{run}: {m}");
                continue;
            }
            Window::Frames(l, p) => (l, p),
        };
        let f = flicker(&lumas).map_err(ie)?;
        values.push(f);
        let mut line = format!("{run}: flicker {}% ({frames} frames {place})", fixed(percent(f), 3));
        if let Some(b) = baseline.as_deref().filter(|b| !b.is_empty()) {
            if base_cache.is_none() {
                base_cache = Some(window(b, start, frames, fg).map_err(ie)?);
            }
            match base_cache.as_ref() {
                Some(Window::Msg(m)) => line += &format!(", detail - ({m})"),
                Some(Window::Frames(base, _)) => match detail(&lumas, base).map_err(ie)? {
                    Some(x) => line += &format!(", detail x{}", fixed(x, 3)),
                    None => line += ", detail - (no static pixels)",
                },
                None => {}
            }
        }
        println!("{line}");
    }
    if values.len() > 1 {
        let (max, min) = (values.iter().copied().fold(f64::MIN, f64::max), values.iter().copied().fold(f64::MAX, f64::min));
        if min > 0.0 {
            println!("flicker spread over {} runs: max / min - 1 = {}%", values.len(), fixed(100.0 * (max / min - 1.0), 1));
        } else {
            println!("flicker spread: a run has no flicker");
        }
    }
    Ok(if values.is_empty() { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

pub fn main(args: &[String]) -> ExitCode {
    if args == ["--self-test"] {
        return match selftest::run() {
            Ok(()) => {
                println!("bench-iq self-test ok");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("bench-iq self-test failed: {e}");
                ExitCode::from(1)
            }
        };
    }
    match run(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("bench-iq: {e}");
            ExitCode::from(1)
        }
    }
}
