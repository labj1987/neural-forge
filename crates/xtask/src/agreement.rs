#![doc = include_str!("agreement/doc.txt")]
// The pixel loops walk runtime-sized buffers in channel-sized chunks; `as_chunks` would only add a remainder to handle.
#![allow(clippy::chunks_exact_to_as_chunks)]
mod synth;

use crate::hdr_encode::args::{self, Kind, Nargs, Opt, Outcome, Parsed, Pos, Spec};
use crate::hdr_encode::pil::{self, Rgb};
use crate::hdr_encode::pyfmt::{fmt_f, json_dumps_compact, parse_int, str_repr, Json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// Frame stamp geometry: mirrors crates/layer/examples/pan.rs.
const BLOCK: usize = 8;
const MAGIC: [u8; 4] = [1, 0, 1, 1];
const STAMP_BITS: usize = 32;
const EXCLUDE_TOP: usize = 2 * BLOCK;
const DETAIL_FRACTION: f64 = 0.10;
/// Originals of the same frame index from two runs are bit-identical when pan.rs ran with the same flags; a larger
/// mean difference means the runs are not comparable.
const ORIGINAL_TOLERANCE: f64 = 0.5;
const HELP: &str = include_str!("agreement/help.txt");

/// What an uncaught Python exception becomes: the message on stderr, status 1.
type Fail = String;

/// The pan.rs frame counter in the top-left corner, or None when there is no stamp.
fn decode_stamp(img: &Rgb) -> Option<u64> {
    let n = MAGIC.len() + STAMP_BITS + 1;
    if img.w < n * BLOCK || img.h < BLOCK {
        return None;
    }
    let bits: Vec<u8> = (0..n).map(|i| u8::from(img.luma_at(i * BLOCK + BLOCK / 2, BLOCK / 2) > 127)).collect();
    if bits[..MAGIC.len()] != MAGIC {
        return None;
    }
    let value_bits = &bits[MAGIC.len()..MAGIC.len() + STAMP_BITS];
    let value = value_bits.iter().fold(0u64, |v, &b| (v << 1) | b as u64);
    let parity = value_bits.iter().map(|&b| b as usize).sum::<usize>() % 2;
    if parity as u8 != bits[n - 1] {
        return None;
    }
    Some(value)
}

fn list_pairs(directory: &str) -> Result<Vec<(String, PathBuf, PathBuf)>, Fail> {
    let mut names: Vec<String> = std::fs::read_dir(directory)
        .map_err(|e| format!("{directory}: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let mut pairs = Vec::new();
    for name in names {
        let Some(key) = name.strip_suffix("-original.png").filter(|k| !k.is_empty()) else { continue };
        let composited = Path::new(directory).join(format!("{key}-composited.png"));
        if composited.exists() {
            pairs.push((key.to_string(), Path::new(directory).join(&name), composited));
        }
    }
    Ok(pairs)
}

type Run = BTreeMap<i64, (PathBuf, PathBuf)>;

/// Frame index -> (original, composited) paths, and how many pairs had no readable stamp.
fn index_run(directory: &str, by_seq: bool) -> Result<(Run, usize), Fail> {
    let mut frames = Run::new();
    let mut unreadable = 0;
    for (key, original, composited) in list_pairs(directory)? {
        let index = if by_seq {
            match parse_int(&key) {
                Some(i) => i,
                None => {
                    unreadable += 1;
                    continue;
                }
            }
        } else {
            match decode_stamp(&pil::read_rgb_png(&original)?) {
                Some(i) => i as i64,
                None => {
                    unreadable += 1;
                    continue;
                }
            }
        };
        frames.entry(index).or_insert((original, composited));
    }
    Ok((frames, unreadable))
}

/// A float32 plane (Pillow's "F" images): every operation rounds to float32 as ImageMath does.
type Plane = Vec<f32>;

fn channels_f(img: &Rgb) -> [Plane; 3] {
    let n = img.w * img.h;
    std::array::from_fn(|c| (0..n).map(|i| img.data[i * 3 + c] as f32).collect())
}

/// Mean of an F image: Pillow's reduce over the whole image sums in double and stores a float32.
fn mean_f(a: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &v in a {
        s += v as f64;
    }
    (s * (1.0 / a.len() as f64)) as f32 as f64
}

/// The cropped region (1, EXCLUDE_TOP, w - 1, h - 1) of a plane.
fn crop(p: &[f32], w: usize, h: usize) -> Result<Plane, Fail> {
    if w < 2 || h < EXCLUDE_TOP + 2 {
        return Err("ValueError: the image is too small for the stamp band and border".into());
    }
    let mut out = Vec::with_capacity((w - 2) * (h - 1 - EXCLUDE_TOP));
    for y in EXCLUDE_TOP..h - 1 {
        out.extend_from_slice(&p[y * w + 1..y * w + w - 1]);
    }
    Ok(out)
}

fn zip_map(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32) -> Plane {
    a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
}

/// 1.0 on the top DETAIL_FRACTION of luma gradient magnitude (cropped), else 0.0.
fn detail_mask(original: &Rgb) -> Result<Plane, Fail> {
    let (w, h) = (original.w, original.h);
    if w < 2 || h < EXCLUDE_TOP + 2 {
        return Err("ValueError: the image is too small for the stamp band and border".into());
    }
    let luma: Plane = (0..w * h).map(|i| original.luma_at(i % w, i / w) as f32).collect();
    let mut g2 = Vec::with_capacity((w - 2) * (h - 1 - EXCLUDE_TOP));
    // The offset images wrap around the edges, but those pixels fall outside the crop.
    for y in EXCLUDE_TOP..h - 1 {
        for x in 1..w - 1 {
            let gx = luma[y * w + x + 1] - luma[y * w + x - 1];
            let gy = luma[(y + 1) * w + x] - luma[(y - 1) * w + x];
            g2.push(gx * gx + gy * gy);
        }
    }
    let stride = (g2.len() / 200_000).max(1);
    let mut sample: Vec<f32> = g2.iter().copied().step_by(stride).collect();
    sample.sort_by(f32::total_cmp);
    let idx = ((sample.len() as f64 * (1.0 - DETAIL_FRACTION)) as usize).min(sample.len() - 1);
    let threshold = sample[idx];
    // Strictly above when ties at the threshold would otherwise take in a flat area.
    Ok(if threshold <= 0.0 { g2.iter().map(|&g| f32::from(g > 0.0)).collect() } else { g2.iter().map(|&g| f32::from(g >= threshold)).collect() })
}

struct FrameMetrics {
    skip: Option<&'static str>,
    ncc: Option<f64>,
    detail_ref: f64,
    detail_test: f64,
}

fn skipped(why: &'static str) -> FrameMetrics {
    FrameMetrics { skip: Some(why), ncc: None, detail_ref: 0.0, detail_test: 0.0 }
}

fn frame_metrics(ref_pair: &(PathBuf, PathBuf), test_pair: &(PathBuf, PathBuf)) -> Result<FrameMetrics, Fail> {
    let ref_o = pil::read_rgb_png(&ref_pair.0)?;
    let ref_c = pil::read_rgb_png(&ref_pair.1)?;
    let test_o = pil::read_rgb_png(&test_pair.0)?;
    let test_c = pil::read_rgb_png(&test_pair.1)?;
    let size = (ref_o.w, ref_o.h);
    if ![(ref_c.w, ref_c.h), (test_o.w, test_o.h), (test_c.w, test_c.h)].iter().all(|s| *s == size) {
        return Ok(skipped("size"));
    }
    let (w, h) = size;
    let (cro, crc, cto, ctc) = (channels_f(&ref_o), channels_f(&ref_c), channels_f(&test_o), channels_f(&test_c));
    let mut orig_sum = 0.0;
    for c in 0..3 {
        let d = zip_map(&cro[c], &cto[c], |a, b| (a - b).abs());
        orig_sum += mean_f(&crop(&d, w, h)?);
    }
    if orig_sum / 3.0 > ORIGINAL_TOLERANCE {
        return Ok(skipped("original"));
    }
    let mask = detail_mask(&ref_o)?;
    let mask_mean = mean_f(&mask);
    let (mut cov, mut var_r, mut var_t, mut det_r, mut det_t) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for c in 0..3 {
        let er = crop(&zip_map(&crc[c], &cro[c], |a, b| a - b), w, h)?;
        let et = crop(&zip_map(&ctc[c], &cto[c], |a, b| a - b), w, h)?;
        let (mr, mt) = (mean_f(&er), mean_f(&et));
        // Centred before the products: keeps the float32 box means well conditioned.
        let cr: Plane = er.iter().map(|&v| v - mr as f32).collect();
        let ct: Plane = et.iter().map(|&v| v - mt as f32).collect();
        cov += mean_f(&zip_map(&cr, &ct, |a, b| a * b));
        var_r += mean_f(&zip_map(&cr, &cr, |a, b| a * b));
        var_t += mean_f(&zip_map(&ct, &ct, |a, b| a * b));
        det_r += mean_f(&zip_map(&er, &mask, |a, m| a.abs() * m));
        det_t += mean_f(&zip_map(&et, &mask, |a, m| a.abs() * m));
    }
    let ncc = if var_r > 1e-9 && var_t > 1e-9 { Some(cov / (var_r * var_t).sqrt()) } else { None };
    // Mean |edit| per channel sample over the masked pixels.
    let scale = if mask_mean > 0.0 { 1.0 / (3.0 * mask_mean) } else { 0.0 };
    Ok(FrameMetrics { skip: None, ncc, detail_ref: det_r * scale, detail_test: det_t * scale })
}

struct FrameRow {
    index: i64,
    ncc: Option<f64>,
    detail_ref: f64,
    detail_test: f64,
}

struct RunResult {
    reference: String,
    test: String,
    matched: usize,
    ref_only: usize,
    test_only: usize,
    unstamped: usize,
    original_mismatch: usize,
    flat: usize,
    frames: Vec<FrameRow>,
    ncc_mean: Option<f64>,
    ncc_min: Option<f64>,
    detail_pct: Option<f64>,
}

fn rj(w: usize, s: &str) -> String {
    format!("{s:>w$}")
}

fn compare_runs(ref_dir: &str, test_dir: &str, by_seq: bool, per_frame: bool) -> Result<RunResult, Fail> {
    let (reference, ref_bad) = index_run(ref_dir, by_seq)?;
    let (test, test_bad) = index_run(test_dir, by_seq)?;
    let common: Vec<i64> = reference.keys().copied().filter(|k| test.contains_key(k)).collect();
    let mut result = RunResult {
        reference: ref_dir.into(),
        test: test_dir.into(),
        matched: 0,
        ref_only: reference.keys().filter(|k| !test.contains_key(k)).count(),
        test_only: test.keys().filter(|k| !reference.contains_key(k)).count(),
        unstamped: ref_bad + test_bad,
        original_mismatch: 0,
        flat: 0,
        frames: Vec::new(),
        ncc_mean: None,
        ncc_min: None,
        detail_pct: None,
    };
    let mut nccs = Vec::new();
    let (mut det_r, mut det_t) = (0.0, 0.0);
    for index in common {
        let m = frame_metrics(&reference[&index], &test[&index])?;
        if m.skip.is_some() {
            result.original_mismatch += 1;
            continue;
        }
        result.matched += 1;
        match m.ncc {
            None => result.flat += 1,
            Some(n) => nccs.push(n),
        }
        det_r += m.detail_ref;
        det_t += m.detail_test;
        if per_frame {
            let ncc = match m.ncc {
                None => "   n/a".to_string(),
                Some(n) => rj(6, &fmt_f(n, 3)),
            };
            println!("  frame {index:6}  ncc {ncc}  |edit| on detail ref {} test {}", rj(7, &fmt_f(m.detail_ref, 3)), rj(7, &fmt_f(m.detail_test, 3)));
        }
        result.frames.push(FrameRow { index, ncc: m.ncc, detail_ref: m.detail_ref, detail_test: m.detail_test });
    }
    if !nccs.is_empty() {
        result.ncc_mean = Some(nccs.iter().fold(0.0, |a, b| a + b) / nccs.len() as f64);
        result.ncc_min = Some(nccs.iter().copied().fold(f64::INFINITY, |m, x| if x < m { x } else { m }));
    }
    result.detail_pct = if det_r > 0.0 { Some(100.0 * det_t / det_r) } else { None };
    Ok(result)
}

fn fmt_opt(v: Option<f64>, width: usize, prec: usize) -> String {
    v.map_or_else(|| "n/a".to_string(), |x| rj(width, &fmt_f(x, prec)))
}

fn jopt(v: Option<f64>) -> Json {
    v.map_or(Json::Null, Json::Float)
}

fn average(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().fold(0.0, |a, b| a + b) / v.len() as f64)
}

/// Prints the table and the JSON line; returns the summary's ncc mean.
fn report(results: &[RunResult], per_frame_json: bool) -> Option<f64> {
    println!(
        "{}  {}  {}  {}  {}  {}  {}",
        rj(3, "run"),
        rj(7, "matched"),
        rj(7, "skipped"),
        rj(4, "flat"),
        rj(8, "ncc mean"),
        rj(7, "ncc min"),
        rj(8, "detail %")
    );
    for (i, r) in results.iter().enumerate() {
        let skipped = r.ref_only + r.test_only + r.unstamped + r.original_mismatch;
        println!(
            "{:>3}  {:>7}  {skipped:>7}  {:>4}  {}  {}  {}",
            i + 1,
            r.matched,
            r.flat,
            fmt_opt(r.ncc_mean, 8, 4),
            fmt_opt(r.ncc_min, 7, 4),
            fmt_opt(r.detail_pct, 8, 1)
        );
        let detail: Vec<String> =
            [("ref_only", r.ref_only), ("test_only", r.test_only), ("unstamped", r.unstamped), ("original_mismatch", r.original_mismatch)]
                .iter()
                .filter(|(_, n)| *n > 0)
                .map(|(k, n)| format!("{k}={n}"))
                .collect();
        if !detail.is_empty() {
            println!("     skipped: {}", detail.join(", "));
        }
    }
    let ncc_mean = average(&results.iter().filter_map(|r| r.ncc_mean).collect::<Vec<_>>());
    let detail_pct = average(&results.iter().filter_map(|r| r.detail_pct).collect::<Vec<_>>());
    let runs: Vec<Json> = results
        .iter()
        .map(|r| {
            let mut o = vec![
                ("ref".to_string(), Json::Str(r.reference.clone())),
                ("test".to_string(), Json::Str(r.test.clone())),
                ("matched".to_string(), Json::Int(r.matched as i64)),
                ("ref_only".to_string(), Json::Int(r.ref_only as i64)),
                ("test_only".to_string(), Json::Int(r.test_only as i64)),
                ("unstamped".to_string(), Json::Int(r.unstamped as i64)),
                ("original_mismatch".to_string(), Json::Int(r.original_mismatch as i64)),
                ("flat".to_string(), Json::Int(r.flat as i64)),
            ];
            if per_frame_json {
                let frames = r
                    .frames
                    .iter()
                    .map(|f| {
                        Json::Obj(vec![
                            ("index".into(), Json::Int(f.index)),
                            ("ncc".into(), jopt(f.ncc)),
                            ("detail_ref".into(), Json::Float(f.detail_ref)),
                            ("detail_test".into(), Json::Float(f.detail_test)),
                        ])
                    })
                    .collect();
                o.push(("frames".to_string(), Json::List(frames)));
            }
            o.push(("ncc_mean".to_string(), jopt(r.ncc_mean)));
            o.push(("ncc_min".to_string(), jopt(r.ncc_min)));
            o.push(("detail_pct".to_string(), jopt(r.detail_pct)));
            Json::Obj(o)
        })
        .collect();
    if results.len() > 1 {
        println!("avg  {:>7}  {:>7}  {:>4}  {}  {:>7}  {}", "", "", "", fmt_opt(ncc_mean, 8, 4), "", fmt_opt(detail_pct, 8, 1));
    }
    let summary = Json::Obj(vec![("ncc_mean".into(), jopt(ncc_mean)), ("detail_pct".into(), jopt(detail_pct)), ("runs".into(), Json::List(runs))]);
    println!("{}", json_dumps_compact(&summary));
    ncc_mean
}

fn spec() -> Spec {
    Spec {
        prog: "xtask agreement",
        help: HELP,
        opts: vec![
            Opt { names: &["-h", "--help"], dest: "help", kind: Kind::Help, choices: None },
            Opt { names: &["--runs"], dest: "runs", kind: Kind::Many, choices: None },
            Opt { names: &["--by-seq"], dest: "by_seq", kind: Kind::Flag(true), choices: None },
            Opt { names: &["--per-frame"], dest: "per_frame", kind: Kind::Flag(true), choices: None },
            Opt { names: &["--self-test"], dest: "self_test", kind: Kind::Flag(true), choices: None },
        ],
        pos: vec![Pos { dest: "ref", nargs: Nargs::Optional, choices: None }, Pos { dest: "test", nargs: Nargs::Optional, choices: None }],
    }
}

fn run(p: &Parsed) -> Result<u8, Fail> {
    let (by_seq, per_frame) = (p.flag("by_seq"), p.flag("per_frame"));
    let mut pairs: Vec<(String, String)> = Vec::new();
    for item in p.list("runs") {
        match item.rsplit_once(':') {
            Some((r, t)) if !r.is_empty() && !t.is_empty() => pairs.push((r.to_string(), t.to_string())),
            _ => return Err(format!("\u{0}--runs takes REF:TEST pairs, got {}", str_repr(&item))),
        }
    }
    if let (Some(r), Some(t)) = (p.str("ref"), p.str("test")) {
        pairs.insert(0, (r.to_string(), t.to_string()));
    }
    if pairs.is_empty() {
        return Err("\u{0}give REF TEST, --runs REF:TEST ..., or --self-test".into());
    }
    for (r, t) in &pairs {
        for d in [r, t] {
            if !Path::new(d).is_dir() {
                return Err(format!("\u{0}not a directory: {d}"));
            }
        }
    }
    let mut results = Vec::new();
    for (r, t) in &pairs {
        if per_frame {
            println!("{r} vs {t}:");
        }
        results.push(compare_runs(r, t, by_seq, per_frame)?);
    }
    Ok(u8::from(report(&results, per_frame).is_none()))
}

pub fn main(argv: &[String]) -> ExitCode {
    let spec = spec();
    let parsed = match args::parse(&spec, argv) {
        Outcome::Parsed(p) => p,
        Outcome::Exit(c) => return ExitCode::from(c),
    };
    if !parsed.extras.is_empty() {
        args::error(spec.prog, HELP, &format!("unrecognized arguments: {}", parsed.extras.join(" ")));
        return ExitCode::from(2);
    }
    let result = if parsed.flag("self_test") { synth::self_test() } else { run(&parsed) };
    match result {
        Ok(code) => ExitCode::from(code),
        // A leading NUL marks what Python reports through `parser.error` (usage and status 2).
        Err(msg) => match msg.strip_prefix('\u{0}') {
            Some(m) => {
                args::error(spec.prog, HELP, m);
                ExitCode::from(2)
            }
            None => {
                eprintln!("{msg}");
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_passes() {
        assert_eq!(synth::self_test(), Ok(0));
    }

    #[test]
    fn stamp_round_trips_and_rejects_bad_parity() {
        let mut img = Rgb::filled(400, 40, [0, 0, 0]);
        assert_eq!(decode_stamp(&img), None);
        synth::stamp(&mut img, 0xdead_beef);
        assert_eq!(decode_stamp(&img), Some(0xdead_beef));
        // Flip the parity block.
        let n = MAGIC.len() + STAMP_BITS;
        let v = if img.luma_at(n * BLOCK + BLOCK / 2, BLOCK / 2) > 127 { 0 } else { 255 };
        for y in 0..BLOCK {
            for x in n * BLOCK..(n + 1) * BLOCK {
                img.data[(y * img.w + x) * 3..(y * img.w + x) * 3 + 3].copy_from_slice(&[v; 3]);
            }
        }
        assert_eq!(decode_stamp(&img), None);
    }

    #[test]
    fn mean_f_is_the_float32_mean() {
        assert_eq!(mean_f(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(mean_f(&[0.1f32; 7]), 0.1f32 as f64);
    }

    #[test]
    fn json_summary_uses_null_for_missing_scores() {
        let r = RunResult {
            reference: "a".into(),
            test: "b".into(),
            matched: 0,
            ref_only: 0,
            test_only: 0,
            unstamped: 0,
            original_mismatch: 0,
            flat: 0,
            frames: Vec::new(),
            ncc_mean: None,
            ncc_min: None,
            detail_pct: None,
        };
        assert_eq!(report(&[r], false), None);
    }
}
