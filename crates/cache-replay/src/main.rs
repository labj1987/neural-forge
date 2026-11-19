//! cache-replay: the Rust port of scripts/cache_replay.py (and scripts/cache_common.py), with the
//! same command line, printed tables, JSON and exported files.
//!
//! Replays the pre-upscaler edit cache offline on a K-hold dump, and measures what it keeps; writes
//! the synthetic dumps and the exports the layer's shader tests compare with.

mod common;
mod pyfmt;
mod replay;

use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use common::Params;
use pyfmt::J;
use replay::{Cadence, MODES};

const USAGE: &str = "usage: cache-replay [DUMP] [--lags N ...] [--modes M ...] [--oracle] [--json OUT] [--self-check]
                    [--reverse-z auto|yes|no] [--make-synthetic DIR] [--kind still|pan|depth-edge|regrade|jitter]
                    [--export-carry DIR] [--quiet] [--cadence N ...] [--phase K] [--adaptive] [--still-interval N]
                    [--aging] [--half-life X] [--no-guided] [--guided-strength X] [--guided-eps X] [--export-cadence DIR]
                    [cache parameters: --eps-v --max-luma-edit --chroma-stops --depth-tol --colour-tol --no-noise-aware
                    --colour-floor --high-decay --no-fingerprint --print-tol --no-surface-fill --no-bilateral --sigma-depth
                    --sigma-luma --no-nearest-surface --no-jitter --chain recurrence|walk --chain-sample bilinear|nearest
                    --low-gain --high-gain]";

#[derive(Default)]
struct Args {
    dump: Option<String>,
    lags: Vec<i64>,
    modes: Vec<String>,
    oracle: bool,
    json: Option<String>,
    self_check: bool,
    reverse_z: String,
    make_synthetic: Option<String>,
    kind: String,
    export_carry: Option<String>,
    quiet: bool,
    cadence: Vec<i64>,
    phase: usize,
    adaptive: bool,
    still_interval: usize,
    aging: bool,
    half_life: f64,
    no_guided: bool,
    guided_strength: f64,
    guided_eps: f64,
    export_cadence: Option<String>,
    p: Params,
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("{USAGE}\ncache-replay: error: {msg}");
    ExitCode::from(2)
}

/// Python's float(): also "inf", "nan", "1e-3".
fn py_float(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let d = Params::default();
    let mut a = Args {
        lags: vec![1, 2, 3, 4, 8],
        modes: MODES.iter().map(|s| s.to_string()).collect(),
        reverse_z: "auto".into(),
        kind: "pan".into(),
        still_interval: 1,
        half_life: 8.0,
        guided_strength: 1.0,
        guided_eps: 0.005,
        p: d,
        ..Args::default()
    };
    let mut it = argv.iter().flat_map(|s| match s.split_once('=') {
        Some((k, v)) if s.starts_with("--") => vec![k.to_string(), v.to_string()],
        _ => vec![s.clone()],
    });
    let mut items: Vec<String> = Vec::new();
    for s in it.by_ref() {
        items.push(s);
    }
    let mut i = 0;
    let is_opt = |s: &str| s.starts_with("--") || (s.starts_with('-') && s.len() > 1 && s.parse::<f64>().is_err());
    let one = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        items.get(*i).filter(|s| !is_opt(s)).cloned().ok_or_else(|| format!("argument {name}: expected one argument"))
    };
    let float = |i: &mut usize, name: &str| -> Result<f64, String> {
        let v = one(i, name)?;
        py_float(&v).ok_or_else(|| format!("argument {name}: invalid float value: '{v}'"))
    };
    let int = |i: &mut usize, name: &str| -> Result<i64, String> {
        let v = one(i, name)?;
        v.trim().parse::<i64>().map_err(|_| format!("argument {name}: invalid int value: '{v}'"))
    };
    let choice = |i: &mut usize, name: &str, choices: &[&str]| -> Result<String, String> {
        let v = one(i, name)?;
        if choices.contains(&v.as_str()) {
            Ok(v)
        } else {
            Err(format!("argument {name}: invalid choice: '{v}' (choose from {})", choices.iter().map(|c| format!("'{c}'")).collect::<Vec<_>>().join(", ")))
        }
    };
    let many = |i: &mut usize, name: &str| -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        while let Some(s) = items.get(*i + 1).filter(|s| !is_opt(s)) {
            out.push(s.clone());
            *i += 1;
        }
        if out.is_empty() {
            return Err(format!("argument {name}: expected at least one argument"));
        }
        Ok(out)
    };
    let ints = |v: Vec<String>, name: &str| -> Result<Vec<i64>, String> {
        v.iter().map(|s| s.trim().parse::<i64>().map_err(|_| format!("argument {name}: invalid int value: '{s}'"))).collect()
    };
    while i < items.len() {
        let s = items[i].clone();
        match s.as_str() {
            "--lags" => a.lags = ints(many(&mut i, "--lags")?, "--lags")?,
            "--modes" => {
                let m = many(&mut i, "--modes")?;
                if let Some(bad) = m.iter().find(|x| !MODES.contains(&x.as_str())) {
                    return Err(format!("argument --modes: invalid choice: '{bad}' (choose from 'none', 'naive', 'low', 'cache')"));
                }
                a.modes = m;
            }
            "--oracle" => a.oracle = true,
            "--json" => a.json = Some(one(&mut i, "--json")?),
            "--self-check" => a.self_check = true,
            "--reverse-z" => a.reverse_z = choice(&mut i, "--reverse-z", &["auto", "yes", "no"])?,
            "--make-synthetic" => a.make_synthetic = Some(one(&mut i, "--make-synthetic")?),
            "--kind" => a.kind = choice(&mut i, "--kind", &["still", "pan", "depth-edge", "regrade", "jitter"])?,
            "--export-carry" => a.export_carry = Some(one(&mut i, "--export-carry")?),
            "--quiet" => a.quiet = true,
            "--cadence" => a.cadence = ints(many(&mut i, "--cadence")?, "--cadence")?,
            "--phase" => a.phase = int(&mut i, "--phase")?.max(0) as usize,
            "--adaptive" => a.adaptive = true,
            "--still-interval" => a.still_interval = int(&mut i, "--still-interval")?.max(0) as usize,
            "--aging" => a.aging = true,
            "--half-life" => a.half_life = float(&mut i, "--half-life")?,
            "--no-guided" => a.no_guided = true,
            "--guided-strength" => a.guided_strength = float(&mut i, "--guided-strength")?,
            "--guided-eps" => a.guided_eps = float(&mut i, "--guided-eps")?,
            "--export-cadence" => a.export_cadence = Some(one(&mut i, "--export-cadence")?),
            "--eps-v" => a.p.eps_v = float(&mut i, "--eps-v")?,
            "--max-luma-edit" => a.p.max_luma_edit = float(&mut i, "--max-luma-edit")?,
            "--chroma-stops" => a.p.chroma_stops = float(&mut i, "--chroma-stops")?,
            "--depth-tol" => a.p.depth_tol = float(&mut i, "--depth-tol")?,
            "--colour-tol" => a.p.colour_tol = float(&mut i, "--colour-tol")?,
            "--no-noise-aware" => a.p.noise_aware = false,
            "--colour-floor" => a.p.colour_floor = float(&mut i, "--colour-floor")?,
            "--high-decay" => a.p.high_decay = float(&mut i, "--high-decay")?,
            "--no-fingerprint" => a.p.fingerprint = false,
            "--print-tol" => a.p.print_tol = float(&mut i, "--print-tol")?,
            "--no-surface-fill" => a.p.surface_fill = false,
            "--no-bilateral" => a.p.bilateral = false,
            "--sigma-depth" => a.p.sigma_depth = float(&mut i, "--sigma-depth")?,
            "--sigma-luma" => a.p.sigma_luma = float(&mut i, "--sigma-luma")?,
            "--no-nearest-surface" => a.p.nearest_surface = false,
            "--no-jitter" => a.p.jitter = false,
            "--chain" => a.p.chain = choice(&mut i, "--chain", &["recurrence", "walk"])?,
            "--chain-sample" => a.p.chain_sample = choice(&mut i, "--chain-sample", &["bilinear", "nearest"])?,
            "--low-gain" => a.p.low_gain = float(&mut i, "--low-gain")?,
            "--high-gain" => a.p.high_gain = float(&mut i, "--high-gain")?,
            _ if is_opt(&s) => return Err(format!("unrecognized arguments: {s}")),
            _ if a.dump.is_none() => a.dump = Some(s),
            _ => return Err(format!("unrecognized arguments: {s}")),
        }
        i += 1;
    }
    Ok(a)
}

/// os.path.abspath: absolute and normalised, without resolving links.
fn abspath(p: &str) -> String {
    let joined = std::env::current_dir().map(|c| c.join(p)).unwrap_or_else(|_| PathBuf::from(p));
    let mut out = PathBuf::from("/");
    for c in joined.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(x) => out.push(x),
            _ => {}
        }
    }
    out.display().to_string()
}

fn write_json(path: &str, doc: &J) -> Result<(), String> {
    std::fs::write(path, pyfmt::dumps(doc, Some(1))).map_err(|e| format!("{path}: {e}"))
}

fn run(a: Args) -> Result<ExitCode, String> {
    if let Some(dir) = &a.make_synthetic {
        let info = replay::make_synthetic(Path::new(dir), &a.kind)?;
        println!("wrote a synthetic {} dump to {dir} ({})", a.kind, pyfmt::dumps(&info, None));
        return Ok(ExitCode::SUCCESS);
    }
    let Some(dump_arg) = a.dump.clone() else {
        return Ok(usage_error("a dump directory is needed (or --make-synthetic)"));
    };
    if a.lags.iter().any(|&l| l < 0) {
        return Ok(usage_error("lags are 0 or more"));
    }
    let p = a.p.clone();
    let dump = common::load_dump(Path::new(&dump_arg))?;
    let (rev, why) = replay::resolve_reverse_z(&dump, &a.reverse_z)?;
    let s = |k: &str| dump.manifest.get(k).map_or("None".to_string(), |v| v.as_str().map_or_else(|| v.to_string(), str::to_string));
    println!("{dump_arg}: {} holds, {}x{}, game {}, depth {}", dump.len(), dump.width, dump.height, s("game"), s("depth_format"));
    println!("reverse_z: {} ({why})", if rev { "yes" } else { "no" });

    if let Some(dir) = &a.export_carry {
        let n = replay::export_carry(&dump, &p, rev, Path::new(dir))?;
        println!("wrote {n} carried hold(s) to {dir}");
        return Ok(ExitCode::SUCCESS);
    }
    if a.self_check {
        let ok = replay::self_check(&dump, &p, rev)?;
        println!("self-check {}", if ok { "passed" } else { "FAILED" });
        return Ok(if ok { ExitCode::SUCCESS } else { ExitCode::from(1) });
    }
    if !a.cadence.is_empty() {
        if a.cadence.iter().any(|&n| n < 2) {
            return Ok(usage_error("--cadence intervals are 2 or more"));
        }
        let cadence = |n: i64| Cadence {
            interval: n as usize,
            phase: a.phase,
            adaptive: a.adaptive,
            still_interval: a.still_interval,
            aging: a.aging,
            half_life: a.half_life,
            guided: !a.no_guided,
            guided_strength: a.guided_strength,
            guided_eps: a.guided_eps,
            ..Cadence::new(n as usize)
        };
        if let Some(dir) = &a.export_cadence {
            if a.cadence.len() != 1 {
                return Ok(usage_error("--export-cadence takes one --cadence interval"));
            }
            let k = replay::export_cadence(&dump, &cadence(a.cadence[0]), &p, rev, Path::new(dir))?;
            println!("wrote {k} hold(s) to {dir}");
            return Ok(ExitCode::SUCCESS);
        }
        let t0 = Instant::now();
        let rows = a.cadence.iter().map(|&n| replay::cadence_metrics(&dump, &cadence(n), &p, rev)).collect::<Result<Vec<_>, _>>()?;
        println!("cadence, {:.0} s", t0.elapsed().as_secs_f64());
        replay::print_cadence(&rows);
        if let Some(out) = &a.json {
            let doc =
                J::obj(vec![("dump", J::Str(abspath(&dump_arg))), ("params", p.as_json()), ("runs", J::List(rows.iter().map(|r| r.json.clone()).collect()))]);
            write_json(out, &doc)?;
            println!("wrote {out}");
        }
        return Ok(ExitCode::SUCCESS);
    }

    let t0 = Instant::now();
    let lags: Vec<usize> = a.lags.iter().map(|&l| l as usize).collect();
    let out = replay::replay(&dump, &lags, &p, rev, a.oracle, &a.modes, !a.quiet)?;
    if out.unusable_holds > 0 {
        println!("{} hold(s) had no usable exposure and were left out", out.unusable_holds);
    }
    println!("{}, {:.0} s", if a.oracle { "oracle (no carry)" } else { "carried" }, t0.elapsed().as_secs_f64());
    replay::print_table(&out.results);
    if let Some(path) = &a.json {
        let doc = J::obj(vec![
            ("dump", J::Str(abspath(&dump_arg))),
            ("holds", J::Int(dump.len() as i64)),
            ("width", J::Int(dump.width as i64)),
            ("height", J::Int(dump.height as i64)),
            ("reverse_z", J::Bool(rev)),
            ("reverse_z_reason", J::Str(why)),
            ("oracle", J::Bool(a.oracle)),
            ("params", p.as_json()),
            ("unusable_holds", J::Int(out.unusable_holds as i64)),
            ("results", replay::results_json(&out.results)),
        ]);
        write_json(path, &doc)?;
        println!("wrote {path}");
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|s| s == "-h" || s == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match parse(&argv) {
        Err(e) => usage_error(&e),
        Ok(a) => run(a).unwrap_or_else(|e| {
            eprintln!("cache-replay: {e}");
            ExitCode::from(1)
        }),
    }
}
