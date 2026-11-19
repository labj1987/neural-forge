//! Tests of the replay and the reference on synthetic dumps (ported from scripts/test_cache_replay.py;
//! its command-line test is tests/cli.rs).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use crate::common::{self as cc, Dump, Frame, Hold, Img, Params};
use crate::replay::{self as cr, Cadence, Frames, Regime, MODES};

// ------------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------------

/// A scratch dir under target/test-scratch, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir: PathBuf =
            [env!("CARGO_MANIFEST_DIR"), "..", "..", "target", "test-scratch", &format!("cache-replay-{name}-{}-{n}", std::process::id())].iter().collect();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The synthetic dump of `kind` written under a fresh scratch dir.
fn synthetic(name: &str, kind: &str) -> (Scratch, Dump) {
    let s = Scratch::new(name);
    cr::make_synthetic(&s.join(kind), kind).unwrap();
    let dump = cc::load_dump(&s.join(kind)).unwrap();
    (s, dump)
}

fn crop(x: &Img, h: usize, w: usize) -> Img {
    Img::from_fn(h, w, x.c, |i, k| x.d[((i / w) * x.w + i % w) * x.c + k])
}

/// The first n holds of a synthetic kind, cropped to the top-left w x h (the Python's
/// synthetic_holds(kind, w=, h=, n=)).
fn small_holds(kind: &str, w: usize, h: usize, n: usize) -> Vec<Hold> {
    let (holds, _, _) = cr::synthetic_holds(kind).unwrap();
    holds
        .into_iter()
        .take(n)
        .map(|k| Hold {
            colour: crop(&k.colour, h, w),
            answer: crop(&k.answer, h, w),
            mvec: crop(&k.mvec, h, w),
            depth: k.depth.as_ref().map(|d| crop(d, h, w)),
            ..k
        })
        .collect()
}

fn write_small(dir: &Path, holds: &[Hold]) {
    cc::write_dump(dir, holds, 3.0, "synthetic", vec![]).unwrap();
}

/// Applies `change` to the dump's manifest and writes it back.
fn rewrite_manifest(path: &Path, change: impl FnOnce(&mut Value)) {
    let file = path.join("manifest.json");
    let mut m: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    change(&mut m);
    std::fs::write(file, serde_json::to_string(&m).unwrap()).unwrap();
}

/// numpy's assert_allclose (rtol 1e-7) on the selected elements.
fn assert_allclose(got: &[f32], want: &[f32], atol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let (a, b) = (a as f64, b as f64);
        assert!((a - b).abs() <= atol + 1e-7 * b.abs(), "element {i}: {a} vs {b} (atol {atol})");
    }
}

type Shown = std::collections::BTreeMap<&'static str, Img>;

/// Hold t's shown edits per mode, carried from hold t - lag along the layer's chain (the
/// accumulated field's recurrence, cc::acc_step), with the frame used.
fn carried(dump: &Dump, t: usize, lag: usize, p: &Params) -> (Shown, std::rc::Rc<Frame>) {
    let mut frames = Frames::new(dump, true, p, lag + 1);
    let (pos, on) = cr::carried_position(&mut frames, t, lag, p).unwrap();
    let (src, dst) = (frames.get(t - lag).unwrap(), frames.get(t).unwrap());
    let (shown, _) = cr::shown_edits(&src, &dst, &pos, &on, p);
    (shown, dst)
}

fn kept(shown: &Img, dst: &Frame, p: &Params, mask: &[bool]) -> f64 {
    let pic = cc::picture_error(&cc::display(&cc::apply_edit(&dst.v_colour, shown, p)), &dst.disp_answer, Some(mask));
    let base = cc::picture_error(&cc::display(&dst.v_colour), &dst.disp_answer, Some(mask));
    100.0 * (1.0 - pic / base)
}

fn all_modes() -> Vec<String> {
    MODES.iter().map(|m| m.to_string()).collect()
}

fn mode<'a>(out: &'a cr::ReplayOut, lag: usize, m: &str) -> &'a cr::ModeResult {
    &out.results.iter().find(|(l, _)| *l == lag).unwrap().1.iter().find(|(n, _)| n == m).unwrap().1
}

fn replay(dump: &Dump, lags: &[usize], p: &Params) -> cr::ReplayOut {
    cr::replay(dump, lags, p, true, false, &all_modes(), false).unwrap()
}

/// (t, since, refresh) of a cadence run.
fn cadence_refreshes(dump: &Dump, cad: &Cadence, p: &Params) -> Vec<(usize, bool)> {
    let mut frames = Frames::new(dump, true, p, 3);
    let mut got = Vec::new();
    cr::run_cadence(dump, &mut frames, cad, p, |h, _| {
        got.push((h.t, h.refresh));
        Ok(true)
    })
    .unwrap();
    got
}

fn mask_of(h: usize, w: usize, f: impl Fn(usize, usize) -> bool) -> Vec<bool> {
    (0..h * w).map(|i| f(i / w, i % w)).collect()
}

// ------------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------------

#[test]
fn loader_round_trip() {
    let (_s, dump) = synthetic("roundtrip", "depth-edge");
    let (holds, white, _) = cr::synthetic_holds("depth-edge").unwrap();
    assert_eq!((dump.len(), dump.width, dump.height, dump.paper_white), (6, 96, 64, white));
    for k in [0, 3] {
        let (got, want) = (dump.hold(k).unwrap(), &holds[k]);
        let half = |x: &Img| x.map(|v| half::f16::from_f32(v).to_f32());
        assert_eq!(got.colour.d, half(&want.colour).d);
        assert_eq!(got.answer.d, half(&want.answer).d);
        assert_eq!(got.mvec.d, half(&want.mvec).d);
        assert_eq!(got.depth.unwrap().d, want.depth.as_ref().unwrap().d);
        assert_eq!(got.jitter, want.jitter);
        assert_eq!(got.exposure, want.exposure);
    }
}

#[test]
fn loader_rejects_wrong_size() {
    let s = Scratch::new("wrongsize");
    write_small(&s.0, &small_holds("still", 16, 8, 2));
    let mut f = std::fs::OpenOptions::new().append(true).open(s.join("hold-01/mvec.rg16f")).unwrap();
    std::io::Write::write_all(&mut f, &[0, 0]).unwrap();
    drop(f);
    assert!(cc::load_dump(&s.0).unwrap().hold(1).is_err());
}

#[test]
fn reverse_z_inferred() {
    for kind in ["still", "depth-edge"] {
        let (_s, dump) = synthetic("revz", kind);
        let (rev, why) = cr::resolve_reverse_z(&dump, "auto").unwrap();
        assert!(rev, "{kind}: {why}");
    }
    let (_s, dump) = synthetic("revz-flipped", "still");
    let flipped = dump.hold(0).unwrap().depth.unwrap().map(|x| 1.0 - x);
    assert!(!cc::infer_reverse_z(&[flipped]).0.unwrap_or(false));
}

#[test]
fn cleared_depth_alone_tells_nothing() {
    // A loading screen's all-zero depth is no evidence (the layer's cache::DEPTH_GEOMETRY_MIN);
    // with one real frame beside it the real frame decides.
    let blank = Img::new(8, 16, 1, 0.0);
    assert_eq!(cc::infer_reverse_z(std::slice::from_ref(&blank)).0, None);
    let (_s, dump) = synthetic("cleared", "depth-edge");
    let real = dump.hold(0).unwrap().depth.unwrap();
    assert_eq!(cc::infer_reverse_z(&[blank.clone(), real.clone()]).0, Some(true));
    assert_eq!(cc::infer_reverse_z(&[blank, real.map(|x| 1.0 - x)]).0, Some(false));
}

#[test]
fn null_fields_load_and_refresh() {
    // A manifest's null jitter or exposure and a hold without vectors or depth load (no NaN, no
    // error), and the cadence refreshes on such holds as the layer does.
    let s = Scratch::new("nulls");
    write_small(&s.0, &small_holds("pan", 32, 16, 6));
    rewrite_manifest(&s.0, |m| {
        m["frames"][1]["jitter"] = Value::Null;
        m["frames"][3]["exposure"] = Value::Null;
        m["frames"][5]["mvec_valid"] = Value::Bool(false);
    });
    let dump = cc::load_dump(&s.0).unwrap();
    let (h1, h3, h5) = (dump.hold(1).unwrap(), dump.hold(3).unwrap(), dump.hold(5).unwrap());
    assert!(!h1.jitter_known);
    assert!(h1.jitter.iter().all(|x| x.is_finite()));
    assert_eq!(h3.exposure, 0.0);
    assert!(!h5.mvec_valid);
    assert!(!h5.mvec.d.iter().any(|&x| x != 0.0));
    let p = Params::default();
    // 0 refreshes; 1 has no jitter: refreshes and stores nothing, so 2 refreshes; 3 is left out
    // (unusable exposure); 4 refreshes after it; 5 has no vectors: refreshes.
    assert_eq!(cadence_refreshes(&dump, &Cadence::new(4), &p), [(0, true), (1, true), (2, true), (4, true), (5, true)]);
    let out = replay(&dump, &[1], &p);
    assert_eq!(mode(&out, 1, "cache").samples, 0, "no hold here could be cached from the one before");

    let s = Scratch::new("nulls-depth");
    write_small(&s.0, &small_holds("depth-edge", 32, 16, 3));
    rewrite_manifest(&s.0, |m| m["frames"][2]["depth_valid"] = Value::Bool(false));
    let dump = cc::load_dump(&s.0).unwrap();
    assert!(dump.hold(2).unwrap().depth.is_none());
    let got: Vec<bool> = cadence_refreshes(&dump, &Cadence::new(4), &Params::default()).into_iter().map(|(_, r)| r).collect();
    assert_eq!(got, [true, false, true], "another presence of depth than the refresh's refreshes");
}

#[test]
fn export_carry_ends_where_the_carry_from_hold_0_ends() {
    let s = Scratch::new("export");
    let out = s.join("out");
    write_small(&s.join("d"), &small_holds("pan", 32, 16, 6));
    rewrite_manifest(&s.join("d"), |m| m["frames"][3]["exposure"] = Value::from(0.0));
    let dump = cc::load_dump(&s.join("d")).unwrap();
    assert_eq!(cr::export_carry(&dump, &Params::default(), true, &out).unwrap(), 2);
    let mut names: Vec<String> =
        std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|f| f.starts_with("cache-")).collect();
    names.sort();
    assert_eq!(names, ["cache-01.f32", "cache-02.f32"]);
    let (_s2, pan) = synthetic("export-pan", "pan");
    assert_eq!(cr::export_carry(&pan, &Params::default(), true, &out).unwrap(), 5);
}

#[test]
fn the_recurrence_accumulates_the_motion_back_to_the_refresh() {
    // The shipped chain over several holds (acc_prev read): in the pan every hold moves 2.5 px right
    // and 1 down, so three holds after the refresh the field is three steps, wherever the chain stays
    // on screen, and NaN where it left.
    let (_s, dump) = synthetic("recurrence", "pan");
    let p = Params::default();
    let mut frames = Frames::new(&dump, true, &p, 4);
    let mut acc: Option<Img> = None;
    for k in 3..=5 {
        let (cur, prev) = (frames.get(k).unwrap(), frames.get(k - 1).unwrap());
        acc = Some(cc::acc_step(acc.as_ref(), &cur, prev.jitter, &p));
    }
    let acc = acc.unwrap();
    let (h, w) = (dump.height, dump.width);
    let on: Vec<bool> = (0..h * w).map(|i| acc.d[i * 2].is_finite() && acc.d[i * 2 + 1].is_finite()).collect();
    for i in (0..h * w).filter(|&i| on[i]) {
        assert_allclose(&acc.d[i * 2..i * 2 + 2], &[7.5, 3.0], 1e-5);
    }
    assert!((0..h - 4).all(|y| (0..w - 8).all(|x| on[y * w + x])));
    assert!(!on[h * w - 1], "the bottom right corner's chain leaves the frame");
    let (pos, ok) = cr::carried_position(&mut frames, 5, 3, &p).unwrap();
    assert_eq!(ok, on);
    let centres = cc::pixel_centres(h, w);
    let (j2, j5) = (frames.get(2).unwrap().jitter, frames.get(5).unwrap().jitter);
    for i in (0..h * w).filter(|&i| on[i]) {
        let want = [centres.d[i * 2] + 7.5 + (j2[0] - j5[0]), centres.d[i * 2 + 1] + 3.0 + (j2[1] - j5[1])];
        assert_allclose(&pos.d[i * 2..i * 2 + 2], &want, 1e-5);
    }
}

#[test]
fn still_keeps_everything() {
    let (_s, dump) = synthetic("still", "still");
    let out = replay(&dump, &[2], &Params::default());
    let r = mode(&out, 2, "cache");
    assert!(r.kept > 99.9);
    assert!(r.err < 1e-3);
    // No edit keeps nothing. pic and base are the same float64 mean of the same pixels (cc::picture_error), so
    // this is 0 on every numpy and SIMD level; checked on both sides.
    assert!(mode(&out, 2, "none").kept.abs() < 1e-6);
}

#[test]
fn a_vector_that_is_not_finite_leaves_the_frame() {
    // As preupscale_cache_motion.comp: an infinite vector in the dump's RG16F copy (an RG32F vector past
    // the half's range) is the largest finite half of its sign and a NaN one the largest finite half, so
    // the chain is lost there (NaN), where both were read as zero motion.
    let (mut holds, white, _) = cr::synthetic_holds("still").unwrap();
    let s = Scratch::new("infinite");
    let mut mv = Img::new(64, 96, 2, 0.0);
    let w = 96;
    mv.d[(5 * w + 7) * 2] = f32::INFINITY;
    mv.d[(9 * w + 11) * 2 + 1] = f32::NEG_INFINITY;
    mv.d[(13 * w + 17) * 2] = f32::NAN;
    holds.truncate(2);
    holds[1].mvec = mv;
    let scale = crate::pyfmt::J::List(vec![crate::pyfmt::J::Float(2272.0), crate::pyfmt::J::Float(-1280.0)]);
    cc::write_dump(&s.0, &holds, white, "synthetic", vec![("mvec_scale".into(), scale)]).unwrap();
    let dump = cc::load_dump(&s.0).unwrap();
    let got = dump.hold(1).unwrap().mvec;
    let at = |y: usize, x: usize| [got.d[(y * w + x) * 2], got.d[(y * w + x) * 2 + 1]];
    assert_eq!(at(5, 7), [65504.0 * 2272.0, 0.0]);
    assert_eq!(at(9, 11), [0.0, 65504.0 * 1280.0]);
    let p = Params { nearest_surface: false, ..Params::default() }; // each texel's own vector
    let f = cc::prepare(&dump.hold(1).unwrap(), dump.paper_white, true, &p);
    let acc = cc::acc_step(None, &f, [0.0, 0.0], &p);
    let nan = |y: usize, x: usize| acc.d[(y * w + x) * 2].is_nan() && acc.d[(y * w + x) * 2 + 1].is_nan();
    assert!(nan(5, 7) && nan(9, 11));
    assert!(nan(13, 17));
    assert_eq!((0..acc.n()).filter(|&i| acc.d[i * 2].is_nan() || acc.d[i * 2 + 1].is_nan()).count(), 3);
}

#[test]
fn history_read_keeps_a_pixel_scale_edit() {
    // A checkerboard edit read at a sub-pixel offset: the strongest tap's own edit comes through
    // whole, where a bilinear blend of the four taps keeps a fraction of it.
    let (_s, dump) = synthetic("history", "still");
    let p = Params::default();
    let hold = dump.hold(0).unwrap();
    let mut src = cc::prepare(&hold, dump.paper_white, true, &p);
    let lin = cc::prepare(&hold, dump.paper_white, true, &p).lin;
    let (h, w) = (lin.h, lin.w);
    src.edit = Img::from_fn(h, w, 3, |i, _| (((i % w + i / w) % 2) as f32) * 0.6 - 0.3);
    let centres = cc::pixel_centres(h, w);
    let pos = Img::from_fn(h, w, 2, |i, c| centres.d[i * 2 + c] + if c == 0 { 0.4 } else { -0.3 });
    let read = cc::read_history(&src, &pos, &lin, &p).edit;
    let blend = cc::sample_bilinear(&src.edit, &pos);
    let (mut blend_sum, mut edit_sum, mut count) = (0.0f64, 0.0f64, 0usize);
    for i in (0..h * w).filter(|&i| (2..h - 2).contains(&(i / w)) && (2..w - 2).contains(&(i % w))) {
        assert_eq!(read.px(i), src.edit.px(i));
        for k in 0..3 {
            blend_sum += blend.d[i * 3 + k].abs() as f64;
            edit_sum += src.edit.d[i * 3 + k].abs() as f64;
            count += 1;
        }
    }
    assert!(blend_sum / (count as f64) < 0.3 * edit_sum / (count as f64));
}

#[test]
fn pan_stays_aligned() {
    let (_s, dump) = synthetic("pan", "pan");
    let p = Params::default();
    let (h, w) = (dump.height, dump.width);
    for lag in [1, 3] {
        let (shown, dst) = carried(&dump, 5, lag, &p);
        // away from where content enters
        let mask = mask_of(h, w, |y, x| (2..h - 2 - lag - 2).contains(&y) && (2..w - 2 - 3 * lag - 2).contains(&x));
        for m in ["naive", "cache"] {
            let k = kept(&shown[m], &dst, &p, &mask);
            assert!(k > 99.0, "lag {lag} {m}: {k}");
        }
    }
    // Without the change of jitter the carry lands half a pixel off: measurably worse.
    let (shown_nj, dst) = carried(&dump, 5, 1, &Params { jitter: false, ..Params::default() });
    let (shown_j, _) = carried(&dump, 5, 1, &p);
    let mask = mask_of(h, w, |y, x| (4..h - 6).contains(&y) && (4..w - 8).contains(&x));
    assert!(kept(&shown_nj["naive"], &dst, &p, &mask) < kept(&shown_j["naive"], &dst, &p, &mask) - 1.0);
}

#[test]
fn depth_edge_rejects_the_uncovered_background() {
    let (_s, dump) = synthetic("depth-edge", "depth-edge");
    let p = Params::default();
    let (h, w) = (dump.height, dump.width);
    let (t, lag) = (5, 2);
    let (shown, dst) = carried(&dump, t, lag, &p);
    let syn = &dump.manifest["synthetic"];
    let (r0, r1) = (syn["box_rows"][0].as_f64().unwrap(), syn["box_rows"][1].as_f64().unwrap());
    let (then, now) = (syn["box_x"][t - lag].as_f64().unwrap(), syn["box_x"][t].as_f64().unwrap());
    // Covered by the box at t - lag, background at t (2 px margins off the edges).
    let mask = mask_of(h, w, |y, x| {
        let (ys, xs) = (y as f64 + 0.5, x as f64 + 0.5);
        ys >= r0 + 2.0 && ys < r1 - 2.0 && xs >= then + 2.0 && xs < now - 2.0
    });
    assert!(mask.iter().filter(|&&m| m).count() > 50);
    let err = |m: &str| {
        let (mut sum, mut n) = (0.0f64, 0usize);
        for i in (0..h * w).filter(|&i| mask[i]) {
            let d: Vec<f32> = (0..3).map(|k| shown[m].d[i * 3 + k] - dst.edit.d[i * 3 + k]).collect();
            sum += cc::luma_px(&d).abs() as f64;
            n += 1;
        }
        sum / n as f64
    };
    let (naive, cache) = (err("naive"), err("cache"));
    assert!(naive > 0.5, "naive {naive} cache {cache}");
    assert!(cache < 0.5 * naive, "naive {naive} cache {cache}");
    let whole = replay(&dump, &[lag], &p);
    assert!(mode(&whole, lag, "cache").worse < mode(&whole, lag, "naive").worse);
}

#[test]
fn self_check_passes() {
    for kind in ["still", "pan", "depth-edge"] {
        let (_s, dump) = synthetic("selfcheck", kind);
        assert!(cr::self_check(&dump, &Params::default(), true).unwrap(), "{kind}");
    }
}

#[test]
fn lag_zero_is_identity_for_every_mode() {
    let (_s, dump) = synthetic("lag0", "depth-edge");
    let out = replay(&dump, &[0], &Params { max_luma_edit: 0.0, ..Params::default() });
    for m in ["naive", "cache"] {
        let r = mode(&out, 0, m);
        assert!(r.err < 1e-6, "{m}");
        assert!(r.kept > 99.999, "{m}");
    }
}

#[test]
fn cadence_base_is_the_lag_replay() {
    // At interval 2 every cached hold is a lag-1 carry from the refresh before it, so the cadence
    // run's cached holds give the lag replay's lag-1 numbers on the same holds.
    let (_s, dump) = synthetic("cadence-base", "depth-edge");
    let p = Params::default();
    let mut frames = Frames::new(&dump, true, &p, 3);
    let cad = Cadence { guided: false, ..Cadence::new(2) };
    let mut shape = (0, 0);
    cr::run_cadence(&dump, &mut frames, &cad, &p, |hold, frames| {
        let (t, dst) = (hold.t, &hold.dst);
        assert_eq!(hold.refresh, t % 2 == 0);
        assert_eq!(hold.since, t % 2);
        shape = (hold.shown.h, hold.shown.w);
        if hold.refresh {
            assert_eq!(hold.shown.d, dst.edit.d);
            return Ok(true);
        }
        let prev = frames.get(t - 1)?;
        let acc = cc::acc_step(None, dst, prev.jitter, &p);
        let (pos, on) = cc::chained_position(&acc, dst, &prev, &p);
        let want = cc::carry(&prev, dst, &pos, &on, &p).cache;
        assert_allclose(&hold.shown.d, &want.d, 1e-6);
        Ok(true)
    })
    .unwrap();
    assert_eq!(shape, (dump.height, dump.width));
}

#[test]
fn adaptive_cadence_refreshes_every_hold_once_still() {
    // The regime: 3 still holds and every hold after them refreshes; a moving camera keeps the
    // interval. The layer's cache::Regime mirrors this.
    let p = Params::default();
    let cases: [(&str, [bool; 6]); 2] = [("still", [true, false, true, true, true, true]), ("pan", [true, false, true, false, true, false])];
    for (kind, want) in cases {
        let (_s, dump) = synthetic("adaptive", kind);
        let got: Vec<bool> = cadence_refreshes(&dump, &Cadence { adaptive: true, ..Cadence::new(2) }, &p).into_iter().map(|(_, r)| r).collect();
        assert_eq!(got, want, "{kind}");
    }
    let mut r = Regime { cad: Cadence { adaptive: true, ..Cadence::new(2) }, still: false, against: 0 };
    for counts in [(60, 100), (60, 100), (10, 100), (60, 100), (60, 100)] {
        r.observe(counts);
    }
    assert!(!r.still, "a moving hold restarts the count");
    r.observe((60, 100));
    assert_eq!(r.interval(), 1);
    for _ in 0..30 {
        r.observe((10, 100));
        r.observe((60, 100));
    }
    assert!(r.still, "a still hold restarts the count out of it");
    for _ in 0..19 {
        r.observe((10, 100));
    }
    assert!(r.still);
    r.observe((10, 100));
    assert_eq!(r.interval(), 2);
}

#[test]
fn cadence_metrics_has_a_pulse_per_hold_since_the_refresh() {
    let (_s, dump) = synthetic("pulse", "pan");
    let r = cr::cadence_metrics(&dump, &Cadence::new(3), &Params::default(), true).unwrap();
    let keys: Vec<usize> = r.pulse.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, [0, 1, 2]);
}

#[test]
fn guided_filter_keeps_a_linear_edit() {
    // A deterministic pseudo-normal guide (a sum of LCG uniforms); numpy's generator is not reproduced.
    let mut state = 1u64;
    let mut uniform = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let guide = Img::from_fn(20, 30, 1, |_, _| (0..12).map(|_| uniform()).sum::<f32>() - 6.0);
    let edit = Img::from_fn(20, 30, 3, |i, k| match k {
        0 => 0.5 * guide.d[i] + 0.1,
        1 => -0.2 * guide.d[i],
        _ => 0.3,
    });
    let out = cr::guided_fast(&guide, &edit, 1e-6);
    let worst = (0..20 * 30)
        .filter(|&i| (2..18).contains(&(i / 30)) && (2..28).contains(&(i % 30)))
        .flat_map(|i| (0..3).map(move |k| (i, k)))
        .map(|(i, k)| (out.d[i * 3 + k] - edit.d[i * 3 + k]).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-2, "max deviation {worst}");
}
