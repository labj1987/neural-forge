//! The offline replay: the lag replay and its metrics, the cadence run,
//! the shader tests' exports, the self-check and the synthetic dumps.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

use crate::common::{self as cc, Dump, Frame, Img, Params};
use crate::pyfmt::{fmt_e, fmt_f, fmt_f_signed, repr, J};

pub const MODES: [&str; 4] = ["none", "naive", "low", "cache"];

/// The dump's holds, prepared on demand, keeping only the few a chain still needs.
pub struct Frames<'a> {
    dump: &'a Dump,
    reverse_z: bool,
    p: Params,
    keep: usize,
    cache: HashMap<usize, Rc<Frame>>,
}

impl<'a> Frames<'a> {
    pub fn new(dump: &'a Dump, reverse_z: bool, p: &Params, keep: usize) -> Self {
        Frames { dump, reverse_z, p: p.clone(), keep, cache: HashMap::new() }
    }

    pub fn get(&mut self, k: usize) -> Result<Rc<Frame>, String> {
        if let Some(f) = self.cache.get(&k) {
            return Ok(f.clone());
        }
        let f = Rc::new(cc::prepare(&self.dump.hold(k)?, self.dump.paper_white, self.reverse_z, &self.p));
        self.cache.insert(k, f.clone());
        let keep = self.keep;
        self.cache.retain(|&j, _| j + keep >= k);
        Ok(f)
    }
}

/// The depth convention: forced, from the manifest, or inferred from the first holds' depth.
pub fn resolve_reverse_z(dump: &Dump, choice: &str) -> Result<(bool, String), String> {
    if choice == "yes" || choice == "no" {
        return Ok((choice == "yes", "forced on the command line".into()));
    }
    match dump.manifest.get("reverse_z") {
        Some(serde_json::Value::Null) | None => {}
        Some(v) => return Ok((v.as_bool().unwrap_or(v.as_f64().is_some_and(|x| x != 0.0)), "from the manifest".into())),
    }
    if !dump.has_depth {
        return Ok((true, "no depth in the dump (depth tests pass everywhere)".into()));
    }
    let mut depths = Vec::new();
    for k in 0..dump.len().min(3) {
        if let Some(d) = dump.hold(k)?.depth {
            depths.push(d);
        }
    }
    Ok(match cc::infer_reverse_z(&depths) {
        (None, why) => (true, format!("inferred: unknown ({why}); assuming reversed-Z")),
        (Some(rev), why) => (rev, format!("inferred: {why}")),
    })
}

pub(crate) fn shown_edits(src: &Frame, dst: &Frame, pos: &Img, on: &[bool], p: &Params) -> (BTreeMap<&'static str, Img>, cc::Carried) {
    let out = cc::carry(src, dst, pos, on, p);
    let mut shown = BTreeMap::new();
    shown.insert("none", Img::new(dst.edit.h, dst.edit.w, 3, 0.0));
    shown.insert("naive", out.naive.clone());
    shown.insert("low", out.low.clone());
    shown.insert("cache", out.cache.clone());
    (shown, out)
}

#[derive(Clone, Debug)]
pub struct ModeResult {
    pub err: f64,
    pub p995: f64,
    pub worse: f64,
    pub harm: f64,
    pub kept: f64,
    pub samples: usize,
}

impl ModeResult {
    fn json(&self) -> J {
        J::obj(vec![
            ("err", J::Float(self.err)),
            ("p995", J::Float(self.p995)),
            ("worse", J::Float(self.worse)),
            ("harm", J::Float(self.harm)),
            ("kept", J::Float(self.kept)),
            ("samples", J::Int(self.samples as i64)),
        ])
    }
}

pub struct ReplayOut {
    pub results: Vec<(usize, Vec<(String, ModeResult)>)>,
    pub unusable_holds: usize,
}

/// The recurrence from a refresh at t - lag up to hold t: (pos, on).
pub(crate) fn carried_position(frames: &mut Frames, t: usize, lag: usize, p: &Params) -> Result<(Img, Vec<bool>), String> {
    let mut acc: Option<Img> = None;
    for k in t + 1 - lag..=t {
        let (cur, prev) = (frames.get(k)?, frames.get(k - 1)?);
        acc = Some(cc::acc_step(acc.as_ref(), &cur, prev.jitter, p));
    }
    let acc = acc.ok_or("empty chain")?;
    let (cur, src) = (frames.get(t)?, frames.get(t - lag)?);
    Ok(cc::chained_position(&acc, &cur, &src, p))
}

/// Every lag x mode x metric. With oracle, each hold is refreshed from itself (no carry).
pub fn replay(dump: &Dump, lags: &[usize], p: &Params, reverse_z: bool, oracle: bool, modes: &[String], log: bool) -> Result<ReplayOut, String> {
    let mut lags = lags.to_vec();
    lags.sort_unstable();
    lags.dedup();
    let max_lag = *lags.last().ok_or("no lags")?;
    let mut frames = Frames::new(dump, reverse_z, p, max_lag + 1);
    let (h, w) = (dump.height, dump.width);
    let centres = cc::pixel_centres(h, w);
    let all_on = vec![true; h * w];
    type Acc = Vec<(f64, f64, f64, f64, f64)>;
    let mut acc: HashMap<(usize, String), Acc> = HashMap::new();
    let mut base: HashMap<usize, Vec<f64>> = lags.iter().map(|&l| (l, Vec::new())).collect();
    let mut skipped = 0;
    for t in 0..dump.len() {
        let dst = frames.get(t)?;
        if !dst.usable {
            skipped += 1;
            continue;
        }
        let (mut pos, mut on) = (centres.clone(), all_on.clone());
        for l in 0..=max_lag.min(t) {
            if l > 0 && !oracle && p.chain == "walk" {
                let (cur, prev) = (frames.get(t - l + 1)?, frames.get(t - l)?);
                pos = cc::chain_step(&pos, &cur, &prev, p);
                let os = cc::on_screen(&pos);
                on.iter_mut().zip(os).for_each(|(a, b)| *a &= b);
            }
            if !lags.contains(&l) {
                continue;
            }
            let src = if oracle || l == 0 { dst.clone() } else { frames.get(t - l)? };
            if !src.usable {
                continue;
            }
            if l > 0 && !oracle {
                let mut ok = true;
                for k in t + 1 - l..=t {
                    let f = frames.get(k)?;
                    ok &= cc::can_carry(&src, &f);
                }
                if !ok {
                    continue;
                }
            }
            if l > 0 && !oracle && p.chain == "recurrence" {
                (pos, on) = carried_position(&mut frames, t, l, p)?;
            }
            let (shown, _) = if oracle || l == 0 { shown_edits(&src, &dst, &centres, &all_on, p) } else { shown_edits(&src, &dst, &pos, &on, p) };
            base.get_mut(&l).ok_or("lag")?.push(dst.base);
            for m in modes {
                let r = cc::pair_metrics(&shown[m.as_str()], &dst, p);
                acc.entry((l, m.clone())).or_default().push((r.err, r.p995, r.worse, r.harm, r.pic));
            }
        }
        if log {
            eprintln!("  hold {}/{}", t + 1, dump.len());
        }
    }
    let mut results = Vec::new();
    for &l in &lags {
        let n = base[&l].len();
        let mut per = Vec::new();
        for m in modes {
            let a = acc.get(&(l, m.clone())).cloned().unwrap_or_default();
            if n == 0 {
                per.push((m.clone(), ModeResult { err: f64::NAN, p995: f64::NAN, worse: f64::NAN, harm: f64::NAN, kept: f64::NAN, samples: 0 }));
                continue;
            }
            let b = cc::np_mean_f64(&base[&l]);
            let col = |f: fn(&(f64, f64, f64, f64, f64)) -> f64| cc::np_mean_f64(&a.iter().map(f).collect::<Vec<_>>());
            per.push((
                m.clone(),
                ModeResult {
                    err: col(|x| x.0),
                    p995: col(|x| x.1),
                    worse: 100.0 * col(|x| x.2),
                    harm: 1000.0 * col(|x| x.3),
                    kept: if b > 0.0 { 100.0 * (1.0 - col(|x| x.4) / b) } else { f64::NAN },
                    samples: n,
                },
            ));
        }
        results.push((l, per));
    }
    Ok(ReplayOut { results, unusable_holds: skipped })
}

pub fn results_json(results: &[(usize, Vec<(String, ModeResult)>)]) -> J {
    J::Obj(results.iter().map(|(l, per)| (l.to_string(), J::Obj(per.iter().map(|(m, r)| (m.clone(), r.json())).collect()))).collect())
}

pub fn print_table(results: &[(usize, Vec<(String, ModeResult)>)]) {
    println!("  {:>3}  {:<6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>4}", "lag", "mode", "err", "p99.5", "worse%", "harm", "kept%", "n");
    for (l, per) in results {
        for (m, r) in per {
            println!(
                "  {l:>3}  {m:<6} {} {} {} {} {} {:>4}",
                fmt_f(r.err, 8, 4),
                fmt_f(r.p995, 7, 3),
                fmt_f(r.worse, 7, 2),
                fmt_f(r.harm, 7, 2),
                fmt_f(r.kept, 7, 2),
                r.samples
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Cadence
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Cadence {
    pub interval: usize,
    pub phase: usize,
    pub adaptive: bool,
    pub still_interval: usize,
    pub aging: bool,
    pub half_life: f64,
    pub guided: bool,
    pub guided_strength: f64,
    pub guided_eps: f64,
    pub stale_px: f64,
}

impl Cadence {
    pub fn new(interval: usize) -> Cadence {
        Cadence {
            interval,
            phase: 0,
            adaptive: false,
            still_interval: 1,
            aging: false,
            half_life: 8.0,
            guided: true,
            guided_strength: 1.0,
            guided_eps: 0.005,
            stale_px: 4.0,
        }
    }

    pub fn label(&self) -> String {
        let mut parts = vec![format!("i{}", self.interval)];
        if self.adaptive {
            parts.push(if self.still_interval != 1 { format!("adaptive{}", self.still_interval) } else { "adaptive".into() });
        }
        if self.aging {
            parts.push("aging".into());
        }
        if !self.guided {
            parts.push("noguided".into());
        }
        if parts.len() > 1 {
            parts.join("-")
        } else {
            format!("{}-base", parts[0])
        }
    }

    fn json(&self) -> J {
        J::obj(vec![
            ("interval", J::Int(self.interval as i64)),
            ("phase", J::Int(self.phase as i64)),
            ("adaptive", J::Bool(self.adaptive)),
            ("still_interval", J::Int(self.still_interval as i64)),
            ("aging", J::Bool(self.aging)),
            ("half_life", J::Float(self.half_life)),
            ("guided", J::Bool(self.guided)),
            ("guided_strength", J::Float(self.guided_strength)),
            ("guided_eps", J::Float(self.guided_eps)),
            ("stale_px", J::Float(self.stale_px)),
        ])
    }
}

/// The layer's adaptive cadence (`cache::Regime`).
pub(crate) struct Regime {
    pub(crate) cad: Cadence,
    pub(crate) still: bool,
    pub(crate) against: usize,
}

impl Regime {
    const TO_SHORTER: usize = 3;
    const TO_LONGER: usize = 20;

    pub(crate) fn interval(&self) -> usize {
        if self.still {
            self.cad.still_interval
        } else {
            self.cad.interval
        }
    }

    pub(crate) fn observe(&mut self, (still, n): (usize, usize)) {
        if n == 0 {
            return;
        }
        let wants = 2 * still > n;
        if wants == self.still {
            self.against = 0;
            return;
        }
        self.against += 1;
        let shorter = (self.cad.still_interval < self.cad.interval) == wants;
        if self.against >= if shorter { Self::TO_SHORTER } else { Self::TO_LONGER } {
            self.still = wants;
            self.against = 0;
        }
    }
}

/// The motion pass's STAT_STILL and STAT_MOTION_N.
pub fn still_counts(mv: &Img) -> (usize, usize) {
    let pace = ((0.001 * mv.w as f64).max(0.5) / 3.0) as f32;
    let lim = 0.25f32 * pace;
    let still = (0..mv.n()).filter(|&i| (mv.d[i * 2] * mv.d[i * 2] + mv.d[i * 2 + 1] * mv.d[i * 2 + 1]).sqrt() < lim).count();
    (still, mv.n())
}

fn step_position(prev: &Frame, cur: &Frame, p: &Params) -> Img {
    let (h, w) = (cur.lin.h, cur.lin.w);
    let dj = [prev.jitter[0] - cur.jitter[0], prev.jitter[1] - cur.jitter[1]];
    Img::from_fn(h, w, 2, |i, c| {
        let centre = if c == 0 { (i % w) as f32 + 0.5 } else { (i / w) as f32 + 0.5 };
        let v = centre + cur.mv.d[i * 2 + c];
        if p.jitter {
            v + dj[c]
        } else {
            v
        }
    })
}

/// 2x2 means at half size, edge-padded.
fn half(x: &Img) -> Img {
    let (hh, ww) = (x.h.div_ceil(2), x.w.div_ceil(2));
    Img::from_fn(hh, ww, 1, |i, _| {
        let (y, xx) = (i / ww, i % ww);
        let at = |dy: usize, dx: usize| x.d[(2 * y + dy).min(x.h - 1) * x.w + (2 * xx + dx).min(x.w - 1)];
        (at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)) / 4.0
    })
}

/// A [1 2 1]/4 tent blur, vertical then horizontal, edge-clamped.
fn tent(x: &Img) -> Img {
    let (h, w) = (x.h, x.w);
    let v = Img::from_fn(h, w, 1, |i, _| {
        let (y, xx) = (i / w, i % w);
        0.25 * x.d[y.saturating_sub(1) * w + xx] + 0.5 * x.d[i] + 0.25 * x.d[(y + 1).min(h - 1) * w + xx]
    });
    Img::from_fn(h, w, 1, |i, _| {
        let (y, xx) = (i / w, i % w);
        0.25 * v.d[y * w + xx.saturating_sub(1)] + 0.5 * v.d[i] + 0.25 * v.d[y * w + (xx + 1).min(w - 1)]
    })
}

/// Upstream's fast guided filter of P (3 channels) on I.
pub(crate) fn guided_fast(i_img: &Img, p_img: &Img, eps: f64) -> Img {
    let (h, w) = (i_img.h, i_img.w);
    let (hh, ww) = (h.div_ceil(2), w.div_ceil(2));
    let m_i = cc::box3(&half(i_img));
    let ii = i_img.map(|x| x * x);
    let var = cc::box3(&half(&ii)).zip(&m_i, |a, m| (a - m * m).max(0.0));
    let sx = (ww as f64 / w as f64) as f32;
    let sy = (hh as f64 / h as f64) as f32;
    let centres = Img::from_fn(h, w, 2, |i, c| if c == 0 { ((i % w) as f32 + 0.5) * sx } else { ((i / w) as f32 + 0.5) * sy });
    let eps = eps as f32;
    let mut out = p_img.clone();
    for ch in 0..3 {
        let pc = p_img.channel(ch);
        let mp = cc::box3(&half(&pc));
        let ip = i_img.zip(&pc, |a, b| a * b);
        let bip = cc::box3(&half(&ip));
        let a = Img::from_fn(hh, ww, 1, |k, _| (bip.d[k] - m_i.d[k] * mp.d[k]) / (var.d[k] + eps));
        let b = Img::from_fn(hh, ww, 1, |k, _| mp.d[k] - a.d[k] * m_i.d[k]);
        let sa = cc::sample_bilinear(&tent(&a), &centres);
        let sb = cc::sample_bilinear(&tent(&b), &centres);
        for k in 0..h * w {
            out.d[k * 3 + ch] = sa.d[k] * i_img.d[k] + sb.d[k];
        }
    }
    out
}

/// One cadence hold: (t, since, refresh, shown, dst, step).
pub struct CadenceHold {
    pub t: usize,
    pub since: usize,
    pub refresh: bool,
    pub shown: Img,
    pub dst: Rc<Frame>,
    pub step: Option<Img>,
}

/// The cache as it runs, one refresh every N holds; `f` returns false to stop.
pub fn run_cadence(
    dump: &Dump,
    frames: &mut Frames,
    cad: &Cadence,
    p: &Params,
    mut f: impl FnMut(CadenceHold, &mut Frames) -> Result<bool, String>,
) -> Result<(), String> {
    let mut regime = cad.adaptive.then(|| Regime { cad: cad.clone(), still: false, against: 0 });
    let (mut src, mut acc, mut prev): (Option<Rc<Frame>>, Option<Img>, Option<Rc<Frame>>) = (None, None, None);
    let mut stale: Option<Img> = None;
    let mut since = 0;
    for t in cad.phase..dump.len() {
        let dst = frames.get(t)?;
        let n = regime.as_ref().map_or(cad.interval, Regime::interval);
        if let Some(r) = regime.as_mut() {
            if dst.mvec_valid {
                r.observe(still_counts(&dst.mv));
            }
        }
        if !dst.usable {
            src = None;
            prev = None;
            continue;
        }
        let consecutive = prev.as_ref().is_some_and(|pv| pv.index == dst.index - 1);
        let refresh = match &src {
            None => true,
            Some(s) => !consecutive || since + 1 >= n || !cc::can_carry(s, &dst),
        };
        let step = if consecutive { prev.as_ref().map(|pv| step_position(pv, &dst, p)) } else { None };
        let shown;
        if refresh {
            src = cc::can_source(&dst).then(|| dst.clone());
            acc = None;
            since = 0;
            stale = Some(Img::new(dst.lin.h, dst.lin.w, 1, 0.0));
            shown = dst.edit.clone();
        } else {
            since += 1;
            let s = src.clone().ok_or("no refresh")?;
            let pv = prev.clone().ok_or("no previous hold")?;
            let a = cc::acc_step(acc.as_ref(), &dst, pv.jitter, p);
            let (pos, on) = cc::chained_position(&a, &dst, &s, p);
            acc = Some(a);
            let r = cc::carry(&s, &dst, &pos, &on, p);
            let st = step.as_ref().ok_or("no step")?;
            let (h, w) = (dst.lin.h, dst.lin.w);
            let moved = Img::from_fn(h, w, 1, |i, _| {
                let dx = st.d[i * 2] - ((i % w) as f32 + 0.5);
                let dy = st.d[i * 2 + 1] - ((i / w) as f32 + 0.5);
                (dx * dx + dy * dy).sqrt()
            });
            let sp = cad.stale_px as f32;
            let st_new = cc::sample_bilinear(stale.as_ref().ok_or("no staleness")?, st).zip(&moved, |s, m| s + (m / sp).clamp(0.0, 1.0));
            let mut sh = r.cache;
            if cad.aging || cad.guided {
                let (low, mut conf) = (r.low_band, r.conf);
                if cad.aging {
                    let hl = cad.half_life as f32;
                    conf.iter_mut().zip(&st_new.d).for_each(|(c, s)| *c *= (-s / hl).exp2());
                }
                let mixed = Img::from_fn(h, w, 3, |i, c| {
                    let l = low.d[i * 3 + c];
                    l + conf[i] * (r.history.d[i * 3 + c] - l)
                });
                sh = cc::clamp_edit(&mixed, p, 1.0);
                if cad.guided {
                    let q = guided_fast(&dst.log_luma, &sh, cad.guided_eps);
                    let gs = cad.guided_strength as f32;
                    let wgt: Vec<f32> = st_new.d.iter().map(|&s| gs * (s / 2.0).clamp(0.0, 1.0)).collect();
                    let mixed = Img::from_fn(h, w, 3, |i, c| {
                        let s = sh.d[i * 3 + c];
                        s + (q.d[i * 3 + c] - s) * wgt[i]
                    });
                    sh = cc::clamp_edit(&mixed, p, 1.0);
                }
            }
            stale = Some(st_new);
            shown = sh;
        }
        let go = f(CadenceHold { t, since, refresh, shown, dst: dst.clone(), step }, frames)?;
        if !go {
            return Ok(());
        }
        prev = Some(dst);
    }
    Ok(())
}

/// The refresh at hold 0 carried to every later hold by the recurrence (the layer's shader test).
pub fn export_carry(dump: &Dump, p: &Params, reverse_z: bool, out: &Path) -> Result<usize, String> {
    let io = |e: std::io::Error| e.to_string();
    std::fs::create_dir_all(out).map_err(io)?;
    let mut frames = Frames::new(dump, reverse_z, p, dump.len());
    let mut text = String::new();
    for k in 0..dump.len() {
        let f = frames.get(k)?;
        let e = dump.hold(k)?.exposure;
        text += &format!("{} {} {}\n", repr(f.jitter[0] as f64), repr(f.jitter[1] as f64), repr(e));
    }
    std::fs::write(out.join("holds.txt"), text).map_err(io)?;
    let src = frames.get(0)?;
    let mut acc: Option<Img> = None;
    let mut carried = 0;
    let cad = Cadence::new(dump.len() + 1);
    run_cadence(dump, &mut frames, &cad, p, |hold, frames| {
        let t = hold.t;
        if t == 0 {
            return Ok(hold.refresh);
        }
        if hold.refresh || t != carried + 1 {
            return Ok(false);
        }
        let a = cc::acc_step(acc.as_ref(), &hold.dst, frames.get(t - 1)?.jitter, p);
        let (pos, on) = cc::chained_position(&a, &hold.dst, &src, p);
        let r = cc::carry(&src, &hold.dst, &pos, &on, p);
        std::fs::write(out.join(format!("cache-{t:02}.f32")), hold.shown.to_le_bytes()).map_err(io)?;
        std::fs::write(out.join(format!("acc-{t:02}.f32")), a.to_le_bytes()).map_err(io)?;
        let wgt: Vec<u8> = r.valid.iter().zip(&r.validity).flat_map(|(a, b)| (a * b).to_le_bytes()).collect();
        std::fs::write(out.join(format!("weight-{t:02}.f32")), wgt).map_err(io)?;
        acc = Some(a);
        carried = t;
        Ok(true)
    })?;
    Ok(carried)
}

/// Every hold's shown edit of a cadence run (the layer's shader test).
pub fn export_cadence(dump: &Dump, cad: &Cadence, p: &Params, reverse_z: bool, out: &Path) -> Result<usize, String> {
    let io = |e: std::io::Error| e.to_string();
    std::fs::create_dir_all(out).map_err(io)?;
    let mut frames = Frames::new(dump, reverse_z, p, 3);
    let mut k = 0;
    let mut text = String::new();
    run_cadence(dump, &mut frames, cad, p, |hold, _| {
        let (still, n) = still_counts(&hold.dst.mv);
        text += &format!("{} {} {still} {n}\n", repr(hold.dst.jitter[0] as f64), repr(hold.dst.jitter[1] as f64));
        std::fs::write(out.join(format!("shown-{:02}.f32", hold.t)), hold.shown.to_le_bytes()).map_err(io)?;
        k += 1;
        Ok(true)
    })?;
    std::fs::write(out.join("holds.txt"), text).map_err(io)?;
    Ok(k)
}

fn half_res(x: &Img) -> Img {
    let (h, w) = (x.h / 2, x.w / 2);
    Img::from_fn(h, w, 1, |i, _| {
        let (y, xx) = (i / w * 2, i % w * 2);
        let at = |dy: usize, dx: usize| x.d[(y + dy) * x.w + xx + dx];
        0.25 * (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1))
    })
}

fn gaussian(x: &Img, sigma: f64) -> Img {
    let r = (3.0 * sigma) as usize;
    let k: Vec<f32> = (0..=2 * r).map(|i| (-0.5 * ((i as f64 - r as f64) / sigma).powi(2)).exp() as f32).collect();
    let s = cc::np_sum_f32(&k);
    let k: Vec<f32> = k.iter().map(|v| v / s).collect();
    let (h, w) = (x.h, x.w);
    let horiz = Img::from_fn(h, w, 1, |i, _| {
        let (y, xx) = (i / w, i % w);
        let mut acc = 0.0f32;
        for (j, kj) in k.iter().enumerate() {
            let xs = (xx as i64 + j as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
            acc += kj * x.d[y * w + xs];
        }
        acc
    });
    Img::from_fn(h, w, 1, |i, _| {
        let (y, xx) = (i / w, i % w);
        let mut acc = 0.0f32;
        for (j, kj) in k.iter().enumerate() {
            let ys = (y as i64 + j as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
            acc += kj * horiz.d[ys * w + xx];
        }
        acc
    })
}

fn regional_light(dst: &Frame, edit: &Img, p: &Params) -> Img {
    let eps = p.eps_v as f32;
    let yo = half_res(&cc::luma(&dst.v_colour));
    let ys = half_res(&cc::luma(&cc::apply_edit(&dst.v_colour, edit, p)));
    gaussian(&ys.zip(&yo, |s, o| ((s + eps) / (o + eps)).log2()), 6.0)
}

fn regional_step(now: &Img, before: &Img, step: &Img) -> Img {
    let sx = half_res(&step.channel(0));
    let sy = half_res(&step.channel(1));
    let pos = Img::from_fn(sx.h, sx.w, 2, |i, c| if c == 0 { sx.d[i] * 0.5 } else { sy.d[i] * 0.5 });
    now.zip(&cc::sample_bilinear(before, &pos), |a, b| a - b)
}

fn mean_abs_f32(x: &Img) -> f64 {
    let a: Vec<f32> = x.d.iter().map(|v| v.abs()).collect();
    (cc::np_sum_f32(&a) / a.len() as f32) as f64
}

#[derive(Default)]
struct Series {
    err: Vec<f64>,
    p995: Vec<f64>,
    worse: Vec<f64>,
    pic: Vec<f64>,
    base: Vec<f64>,
}

impl Series {
    fn summary(&self) -> J {
        if self.err.is_empty() {
            return J::Obj(vec![]);
        }
        let m = cc::np_mean_f64;
        J::obj(vec![
            ("err", J::Float(m(&self.err))),
            ("p995", J::Float(m(&self.p995))),
            ("worse", J::Float(100.0 * m(&self.worse))),
            ("kept", J::Float(100.0 * (1.0 - m(&self.pic) / m(&self.base)))),
            ("samples", J::Int(self.err.len() as i64)),
        ])
    }
}

pub struct CadenceRow {
    pub json: J,
    label: String,
    all: J,
    cached: J,
    pub(crate) pulse: Vec<(usize, f64)>,
    pulse_model: f64,
    pulse_excess: f64,
    flicker: f64,
    refresh_share: f64,
}

/// One cadence run's numbers.
pub fn cadence_metrics(dump: &Dump, cad: &Cadence, p: &Params, reverse_z: bool) -> Result<CadenceRow, String> {
    let mut frames = Frames::new(dump, reverse_z, p, 3);
    let (mut all, mut cached) = (Series::default(), Series::default());
    let mut pulse: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let (mut model_pulse, mut flick) = (Vec::new(), Vec::new());
    let mut prev: Option<(usize, Img, Img, Img, Img, Img)> = None;
    let (mut holds, mut refreshes) = (0usize, 0usize);
    run_cadence(dump, &mut frames, cad, p, |hold, _| {
        holds += 1;
        refreshes += usize::from(hold.refresh);
        let dst = &hold.dst;
        let reg = regional_light(dst, &hold.shown, p);
        let reg_true = regional_light(dst, &dst.edit, p);
        let pic = cc::display(&cc::apply_edit(&dst.v_colour, &hold.shown, p));
        if let (Some(pv), Some(step)) = (&prev, &hold.step) {
            if pv.0 + 1 == hold.t {
                let r = cc::pair_metrics(&hold.shown, dst, p);
                for s in [&mut all].into_iter().chain((!hold.refresh).then_some(&mut cached)) {
                    s.err.push(r.err);
                    s.p995.push(r.p995);
                    s.worse.push(r.worse);
                    s.pic.push(r.pic);
                    s.base.push(dst.base);
                }
                let s = mean_abs_f32(&regional_step(&reg, &pv.1, step));
                pulse.entry(if hold.refresh { 0 } else { hold.since }).or_default().push(s);
                model_pulse.push(mean_abs_f32(&regional_step(&reg_true, &pv.2, step)));
                let lin_then = cc::sample_bilinear(&pv.5, step);
                let then_pic = cc::sample_bilinear(&pv.3, step);
                let then_ans = cc::sample_bilinear(&pv.4, step);
                let mut sel = Vec::new();
                for i in 0..dst.lin.n() {
                    let same = (lin_then.d[i].max(1e-6).log2() - dst.lin.d[i].max(1e-6).log2()).abs() < 0.1;
                    if same {
                        let mean3 = |a: &Img, b: &Img| {
                            ((a.d[i * 3] - b.d[i * 3]).abs() + (a.d[i * 3 + 1] - b.d[i * 3 + 1]).abs() + (a.d[i * 3 + 2] - b.d[i * 3 + 2]).abs()) / 3.0
                        };
                        sel.push(mean3(&pic, &then_pic) - mean3(&dst.disp_answer, &then_ans));
                    }
                }
                flick.push(if sel.is_empty() { 0.0 } else { (cc::np_sum_f32(&sel) / sel.len() as f32) as f64 });
            }
        }
        prev = Some((hold.t, reg, reg_true, pic, dst.disp_answer.clone(), dst.lin.clone()));
        Ok(true)
    })?;
    let by_since: Vec<(usize, f64)> = pulse.iter().map(|(k, v)| (*k, cc::np_mean_f64(v))).collect();
    let others: Vec<f64> = pulse.iter().filter(|(k, _)| **k != 0).flat_map(|(_, v)| v.clone()).collect();
    let pulse_model = if model_pulse.is_empty() { f64::NAN } else { cc::np_mean_f64(&model_pulse) };
    let pulse_excess = if others.is_empty() { f64::NAN } else { cc::np_mean_f64(pulse.get(&0).map_or(&[f64::NAN][..], |v| v)) - cc::np_mean_f64(&others) };
    let flicker = if flick.is_empty() { f64::NAN } else { 1000.0 * cc::np_mean_f64(&flick) };
    let refresh_share = 100.0 * refreshes as f64 / holds.max(1) as f64;
    let (a, c) = (all.summary(), cached.summary());
    let json = J::obj(vec![
        ("label", J::Str(cad.label())),
        ("cadence", cad.json()),
        ("all", a.clone()),
        ("cached", c.clone()),
        ("pulse", J::Obj(by_since.iter().map(|(k, v)| (k.to_string(), J::Float(*v))).collect())),
        ("pulse_model", J::Float(pulse_model)),
        ("pulse_excess", J::Float(pulse_excess)),
        ("flicker", J::Float(flicker)),
        ("refresh_share", J::Float(refresh_share)),
    ]);
    Ok(CadenceRow { json, label: cad.label(), all: a, cached: c, pulse: by_since, pulse_model, pulse_excess, flicker, refresh_share })
}

fn get(o: &J, k: &str) -> f64 {
    match o {
        J::Obj(items) => items.iter().find(|(x, _)| x == k).map_or(f64::NAN, |(_, v)| if let J::Float(f) = v { *f } else { f64::NAN }),
        _ => f64::NAN,
    }
}

pub fn print_cadence(rows: &[CadenceRow]) {
    println!(
        "  {:<20} {:>7} {:>6} {:>6} {:>6} | {:>10} {:>6} | {:>26} {:>8} | {:>7} | {:>6}",
        "run", "err", "p99.5", "worse%", "kept%", "cached err", "p99.5", "pulse run/others (model)", "excess", "flicker", "model%"
    );
    for r in rows {
        let others: Vec<f64> = r.pulse.iter().filter(|(k, _)| *k != 0).map(|(_, v)| *v).collect();
        let p0 = r.pulse.iter().find(|(k, _)| *k == 0).map_or(f64::NAN, |(_, v)| *v);
        println!(
            "  {:<20} {} {} {} {} | {} {} | {} / {} ({}) {} | {} | {}",
            r.label,
            fmt_f(get(&r.all, "err"), 7, 4),
            fmt_f(get(&r.all, "p995"), 6, 3),
            fmt_f(get(&r.all, "worse"), 6, 2),
            fmt_f(get(&r.all, "kept"), 6, 1),
            fmt_f(get(&r.cached, "err"), 10, 4),
            fmt_f(get(&r.cached, "p995"), 6, 3),
            fmt_f(p0, 8, 4),
            fmt_f(if others.is_empty() { f64::NAN } else { cc::np_mean_f64(&others) }, 6, 4),
            fmt_f(r.pulse_model, 0, 4),
            fmt_f_signed(r.pulse_excess, 8, 4),
            fmt_f(r.flicker, 7, 3),
            fmt_f(r.refresh_share, 6, 1)
        );
    }
}

// ------------------------------------------------------------------------------------------------
// Self-check
// ------------------------------------------------------------------------------------------------

fn max_abs_diff(a: &[f32], b: impl Fn(usize) -> f32) -> f64 {
    let mut m = 0.0f32;
    for (i, &x) in a.iter().enumerate() {
        let d = (x - b(i)).abs();
        if d.is_nan() {
            return f64::NAN;
        }
        m = m.max(d);
    }
    m as f64
}

pub fn self_check(dump: &Dump, p: &Params, reverse_z: bool) -> Result<bool, String> {
    let mut ok = true;
    let mut check = |name: String, cond: bool, detail: String| {
        ok &= cond;
        println!("  {}  {name}: {detail}", if cond { "PASS" } else { "FAIL" });
    };
    let mut frames = Frames::new(dump, reverse_z, p, 2);
    let mut usable = Vec::new();
    for k in 0..dump.len() {
        if frames.get(k)?.usable {
            usable.push(k);
        }
    }
    check("usable holds".into(), usable.len() >= 2, format!("{} of {} have a usable exposure", usable.len(), dump.len()));
    if usable.len() < 2 {
        return Ok(false);
    }
    let (h, w) = (dump.height, dump.width);
    let centres = cc::pixel_centres(h, w);
    let on = vec![true; h * w];
    for &k in usable.iter().take(3) {
        let f = frames.get(k)?;
        let (shown, d) = shown_edits(&f, &f, &centres, &on, p);
        let dev = [&d.valid, &d.validity, &d.v_colour, &d.conf].iter().map(|x| max_abs_diff(x, |_| 1.0)).fold(f64::NEG_INFINITY, f64::max);
        check(format!("lag 0 hold {k}: carry admits everything"), dev < 1e-5, format!("max |weight - 1| {}", fmt_e(dev, 2)));
        let dh = max_abs_diff(&d.history.d, |i| f.edit.d[i]);
        check(format!("lag 0 hold {k}: history read is the edit"), dh < 1e-5, format!("max diff {} stop", fmt_e(dh, 2)));
        let dn = max_abs_diff(&shown["naive"].d, |i| f.edit.d[i]);
        check(format!("lag 0 hold {k}: naive is the edit"), dn < 1e-5, format!("max diff {} stop", fmt_e(dn, 2)));
        let ce = cc::clamp_edit(&f.edit, p, 1.0);
        let dc = max_abs_diff(&shown["cache"].d, |i| ce.d[i]);
        check(format!("lag 0 hold {k}: cache is the edit"), dc < 1e-4, format!("max diff {} stop", fmt_e(dc, 2)));
        let guide = cc::guide_of(&f);
        let lo = cc::clamp_edit(&cc::low_band(&cc::build_levels(&f.edit, &vec![1.0; h * w], &guide), &guide, p), p, 1.0);
        let dl = max_abs_diff(&shown["low"].d, |i| lo.d[i]);
        check(format!("lag 0 hold {k}: low is the edit's low band"), dl < 1e-5, format!("max diff {} stop", fmt_e(dl, 2)));
        check(format!("lag 0 hold {k}: none is no edit"), shown["none"].d.iter().all(|&x| x == 0.0), "all zero".into());
    }
    let raw = Params { max_luma_edit: 0.0, ..p.clone() };
    let modes: Vec<String> = ["none", "naive", "cache"].iter().map(|s| s.to_string()).collect();
    let r = replay(dump, &[1], &raw, reverse_z, true, &modes, false)?;
    for m in ["naive", "cache"] {
        let x = &r.results[0].1.iter().find(|(n, _)| n == m).ok_or("mode")?.1;
        check(
            format!("lag 1 oracle {m}"),
            x.err < 1e-4 && x.kept > 99.99,
            format!("err {} stop, kept {}% over {} holds", fmt_e(x.err, 2), fmt_f(x.kept, 0, 4), x.samples),
        );
    }
    let modes: Vec<String> = ["none", "cache"].iter().map(|s| s.to_string()).collect();
    let c = replay(dump, &[1], p, reverse_z, true, &modes, false)?;
    let ceiling = c.results[0].1.iter().find(|(n, _)| n == "cache").ok_or("mode")?.1.kept;
    println!("  info  with the edit clamp at {} stop, the oracle keeps {}% (the clamp's own ceiling)", repr(p.max_luma_edit), fmt_f(ceiling, 0, 2));
    Ok(ok)
}

// ------------------------------------------------------------------------------------------------
// Synthetic dumps
// ------------------------------------------------------------------------------------------------

/// v of the colour and the model's edit at world (x, y), in f64 as numpy computes them.
fn scene(x: f64, y: f64, obj: bool) -> ([f64; 3], [f64; 3]) {
    let detail = 0.25 * (1.3 * x).sin() * (1.1 * y).sin();
    let (base, el, chroma) = if obj {
        ([0.45 + 0.1 * (x / 5.0).sin(), 0.25 + 0.05 * (y / 4.0).cos(), 0.12], 0.7 + 0.2 * (0.9 * x + 1.0).sin() * (0.8 * y).cos(), [0.1, 0.0, -0.1])
    } else {
        (
            [0.20 + 0.08 * (x / 13.0).sin(), 0.28 + 0.08 * (y / 11.0).sin(), 0.30 + 0.05 * ((x + y) / 17.0).cos()],
            -0.4 + 0.4 * (x / 17.0 + y / 23.0).sin() + 0.25 * (0.9 * x + 1.0).sin() * (0.7 * y).cos(),
            [-0.05, 0.05 * (x / 9.0).sin(), 0.15],
        )
    };
    let l = cc::LUMA.map(f64::from);
    let cl = chroma[0] * l[0] + chroma[1] * l[1] + chroma[2] * l[2];
    let g = detail.exp2();
    ([base[0] * g, base[1] * g, base[2] * g], [el + chroma[0] - cl, el + chroma[1] - cl, el + chroma[2] - cl])
}

/// Holds with known motion and a known "model" edit, as synthetic_holds writes them.
/// The holds, the paper white and the manifest's `synthetic` entry.
type Synthetic = (Vec<cc::Hold>, f64, Vec<(String, J)>);

pub fn synthetic_holds(kind: &str) -> Result<Synthetic, String> {
    let (w, h, n) = (96usize, 64usize, 6usize);
    let (paper_white, exposure) = (3.0, 1.5);
    let mut holds = Vec::new();
    let mut info = vec![("kind".to_string(), J::Str(kind.into()))];
    let mut box_x = Vec::new();
    for t in 0..n {
        let tf = t as f64;
        let (cam, jit) = match kind {
            "pan" | "regrade" => ([2.5 * tf, 1.0 * tf], [if t % 2 == 0 { 0.25 } else { -0.25 }, 0.0]),
            "still" | "depth-edge" => ([0.0, 0.0], [0.0, 0.0]),
            "jitter" => ([0.0, 0.0], [[0.0, 0.0], [0.3, -0.2], [-0.35, 0.15], [0.1, 0.4], [-0.2, -0.3], [0.45, 0.05]][t % 6]),
            _ => return Err(format!("unknown synthetic kind {kind:?}")),
        };
        let mut colour = Img::new(h, w, 3, 0.0);
        let mut answer = Img::new(h, w, 3, 0.0);
        let mut mvec = Img::new(h, w, 2, 0.0);
        let mut depth = Img::new(h, w, 1, 0.0);
        let x0 = 10 + 5 * t;
        for i in 0..h * w {
            let (xs, ys) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let (xx, yy) = (xs - jit[0] + cam[0], ys - jit[1] + cam[1]);
            let (mut v, mut edit) = scene(xx, yy, false);
            let mut d = 0.002 + 0.03 * yy.max(0.0) / h as f64;
            let mut mv = [0.0f64, 0.0];
            if (kind == "pan" || kind == "regrade") && t > 0 {
                mv = [2.5, 1.0];
            }
            if kind == "regrade" {
                edit = edit.map(|e| e + 0.15 * tf);
            }
            if kind == "jitter" {
                let j = 0.3 * ((2.9 * xx).cos() * (2.7 * yy).cos());
                edit = edit.map(|e| e + j);
            }
            if kind == "depth-edge" {
                let sky = ys < 8.0;
                if sky {
                    v = [0.3, 0.45, 0.8];
                    edit = [0.0, 0.1, 0.3];
                }
                d = if sky { 0.0 } else { 0.01 + 0.01 * ys / h as f64 };
                let x0f = x0 as f64;
                if xs >= x0f && xs < x0f + 20.0 && (20.0..44.0).contains(&ys) {
                    (v, edit) = scene(xs - x0f, ys - 20.0, true);
                    d = 0.4;
                    if t > 0 {
                        mv = [-5.0, 0.0];
                    }
                }
            }
            let to_scene = paper_white / exposure;
            for c in 0..3 {
                let a = ((v[c] + cc::EPS_V) * edit[c].exp2() - cc::EPS_V).max(0.0);
                colour.d[i * 3 + c] = (v[c] * to_scene) as f32;
                answer.d[i * 3 + c] = (a * to_scene) as f32;
            }
            mvec.d[i * 2] = mv[0] as f32;
            mvec.d[i * 2 + 1] = mv[1] as f32;
            depth.d[i] = d as f32;
        }
        if kind == "depth-edge" {
            box_x.push(J::Int(x0 as i64));
        }
        holds.push(cc::Hold {
            index: t as i64,
            jitter: [jit[0] as f32, jit[1] as f32],
            exposure,
            colour,
            answer,
            mvec,
            depth: Some(depth),
            jitter_known: true,
            mvec_valid: true,
        });
    }
    if kind == "depth-edge" {
        info.push(("box_x".into(), J::List(box_x)));
        info.push(("box_rows".into(), J::List(vec![J::Int(20), J::Int(44)])));
        info.push(("box_width".into(), J::Int(20)));
    }
    Ok((holds, paper_white, info))
}

pub fn make_synthetic(path: &Path, kind: &str) -> Result<J, String> {
    let (holds, white, info) = synthetic_holds(kind)?;
    let info = J::Obj(info);
    cc::write_dump(path, &holds, white, &format!("synthetic-{kind}"), vec![("synthetic".into(), info.clone())]).map_err(|e| e.to_string())?;
    Ok(info)
}
