//! `cargo xtask bench-report` (it replaced scripts/bench-report.py, flag for flag). The text below is also the usage it prints.
//!
//! bench-iq is called in the process instead of loaded from beside the script, and `--host` sends this executable over
//! ssh instead of the two Python files. Directory listings are sorted (Python's glob gives the OS order, which only
//! matters with several matching files in one run directory).
#![doc = include_str!("bench_report/usage.txt")]

use std::io::Write;
use std::process::{Command, ExitCode, Stdio};
use std::sync::LazyLock;

use regex::Regex;

use crate::bench_iq::pyish::{exists, fixed, glob_dir, join, py_float, py_float_repr, py_sum, read_text, read_text_lossy};
use crate::bench_iq::{self, IqError};
use crate::py::{repr_str, splitlines};

mod pytime;
mod selftest;

const USAGE: &str = include_str!("bench_report/usage.txt");

/// Displayed / real at or above this means frame generation presented generated frames (2x is the lowest multiplier GTA offers).
const FG_ENGAGED: f64 = 1.5;

fn re(p: &str) -> Regex {
    Regex::new(p).expect("static regex")
}

static MS_VALUE: LazyLock<Regex> = LazyLock::new(|| re(r"^([0-9.]+)(ns|µs|us|ms|s)$"));
static SYNC_FIELD: LazyLock<Regex> = LazyLock::new(|| re(r"([\w()+]+)=(\S+)"));
static HOLDS: LazyLock<Regex> = LazyLock::new(|| re(r"\[preupscale\] mode=\S+ .* holds_per_s=([0-9.]+)"));
static COMPOSITED: LazyLock<Regex> = LazyLock::new(|| re(r"\[present\] [0-9.]+ fps \(([0-9.]+)/s composited"));
static PASS4_STAMP: LazyLock<Regex> = LazyLock::new(|| re(r"Pass4-(.+)\.txt"));
static CSV_STAMP: LazyLock<Regex> = LazyLock::new(|| re(r"([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})"));

/// The message of a failure that ends the program (on stderr, exit status 1).
type Fail = String;

fn float(s: &str) -> Result<f64, Fail> {
    py_float(s).ok_or_else(|| format!("not a number: {}", repr_str(s)))
}

/// The value of a `[sync]` field: a duration in milliseconds, or the text as it was.
#[derive(Clone, PartialEq)]
enum Field {
    Ms(f64),
    Text(String),
}

impl Field {
    /// `str(value)`.
    fn text(&self) -> String {
        match self {
            Field::Ms(v) => py_float_repr(*v),
            Field::Text(s) => s.clone(),
        }
    }
}

/// `ms(value)`: a duration with its unit in milliseconds, None when it is not one.
fn ms(value: &str) -> Result<Option<f64>, Fail> {
    let Some(c) = MS_VALUE.captures(value) else { return Ok(None) };
    let scale = match &c[2] {
        "ns" => 1e-6,
        "µs" | "us" => 1e-3,
        "ms" => 1.0,
        _ => 1000.0,
    };
    Ok(Some(float(&c[1])? * scale))
}

/// `statistics.median`.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn sync_medians(log: &str) -> Result<String, Fail> {
    let mut rows: Vec<Vec<(String, Field)>> = Vec::new();
    for line in splitlines(log) {
        if !line.contains("[sync]") {
            continue;
        }
        let mut fields: Vec<(String, Field)> = Vec::new();
        for c in SYNC_FIELD.captures_iter(line) {
            let v = match ms(&c[2])? {
                Some(t) => Field::Ms(t),
                None => Field::Text(c[2].to_string()),
            };
            // a repeated key keeps its first position with the last value, like a dict
            match fields.iter_mut().find(|(k, _)| *k == c[1]) {
                Some(slot) => slot.1 = v,
                None => fields.push((c[1].to_string(), v)),
            }
        }
        rows.push(fields);
    }
    if rows.len() > 1 {
        rows.remove(0);
    }
    let Some(first) = rows.first() else { return Ok("-".to_string()) };
    let mut out = Vec::new();
    for (k, _) in first {
        let vals: Vec<&Field> = rows.iter().filter_map(|r| r.iter().find(|(rk, _)| rk == k).map(|(_, v)| v)).collect();
        let nums: Vec<f64> = vals.iter().filter_map(|v| if let Field::Ms(x) = v { Some(*x) } else { None }).collect();
        if nums.len() == vals.len() {
            out.push(format!("{k}={}", fixed(median(nums), 2)));
        } else {
            let mut set: Vec<String> = vals.iter().map(|v| v.text()).collect();
            set.sort();
            set.dedup();
            out.push(format!("{k}={}", set.join("/")));
        }
    }
    Ok(format!("{} lines: {}", rows.len(), out.join(" ")))
}

/// Median `holds_per_s` of the pre-upscaler path's summary lines (one per 300 holds), or None when the log has none (the
/// post-upscaler path ran, or a build before the rate was logged).
fn held_per_second(log: &str) -> Result<Option<f64>, Fail> {
    let mut rates = Vec::new();
    for c in HOLDS.captures_iter(log) {
        let x = float(&c[1])?;
        if x > 0.0 {
            rates.push(x);
        }
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(rates.get(rates.len() / 2).copied())
}

/// The layer's own rate for the report line. When frames were held before the upscaler the composited figure is not the
/// effect's rate, so it is labelled as what it is.
fn rate_text(composited: Option<f64>, held: Option<f64>) -> String {
    let fmt = |v: Option<f64>| v.map_or("NA".to_string(), |v| fixed(v, 1));
    match held {
        None => format!("composited/s {}", fmt(composited)),
        Some(_) => format!("held before upscaler/s {} (composited/s {}: frames not held, e.g. loading screens)", fmt(held), fmt(composited)),
    }
}

/// ` | flicker X% detail xY` when the run has a frame series, else nothing. `fg`: the run's measured frame-generation
/// ratio (displayed / rendered), rounded for the window.
fn iq_text(d: &str, baseline: Option<&str>, fg: Option<f64>) -> Result<String, Fail> {
    let multiplier = match fg {
        Some(f) if f != 0.0 => (f.round_ties_even() as i64).max(1),
        _ => 1,
    };
    match bench_iq::summary(d, baseline, None, None, multiplier) {
        Ok(Some(text)) if !text.is_empty() => Ok(format!(" | {text}")),
        Ok(_) => Ok(String::new()),
        Err(IqError::Os(e)) => Ok(format!(" | flicker: {e}")),
        Err(IqError::Other(e)) => Err(e),
    }
}

/// What a run gave: real fps of pass 4, displayed fps, GPU utilisation, the measured frame-generation multiplier.
type Row = (f64, Option<f64>, Option<f64>, Option<f64>);

fn line_field(line: &str, i: usize) -> Result<f64, Fail> {
    float(line.split(',').nth(i).ok_or("a line has too few fields")?)
}

fn report(label: &str, base: &str, baseline: Option<&str>) -> Result<Option<Row>, Fail> {
    let d = join(base, label);
    if !exists(&format!("{d}/benchmark.txt")) {
        println!("{label}: no benchmark.txt (early exit?)");
        return Ok(None);
    }
    let text = read_text(&format!("{d}/benchmark.txt"))?;
    let b = splitlines(&text);
    let avgs: Vec<f64> = b.iter().skip(1).take(5).map(|l| line_field(l, 3)).collect::<Result<_, _>>()?;
    let p4s = glob_dir(&d, |n| n.starts_with("Pass4-") && n.ends_with(".txt") && n.len() >= 10);
    let Some(p4) = p4s.first() else {
        println!("{label}: no Pass4 frame-time file");
        return Ok(None);
    };
    let ts = PASS4_STAMP.captures(p4).ok_or("no timestamp in the Pass 4 file name")?[1].to_string();
    let end = pytime::parse_gta_stamp(&ts).ok_or_else(|| format!("bad timestamp {}, expected YY-MM-DD-HH-MM-SS", repr_str(&ts)))?;
    let p4_text = read_text(p4)?;
    let mut ft = Vec::new();
    for line in splitlines(&p4_text).into_iter().skip(1).filter(|l| !l.trim().is_empty()) {
        ft.push(float(line.split_whitespace().nth(1).ok_or("a line has too few fields")?)?);
    }
    let dur = py_sum(&ft) / 1000.0;
    if dur == 0.0 {
        return Err("division by zero".into());
    }
    let real4 = ft.len() as f64 / dur;
    let w0 = end - pytime::seconds(dur - 5.0);
    let w1 = end - 5_000_000;
    let mut disp = None;
    for m in
        glob_dir(&d, |n| n.starts_with("GTA5_Enhanced_") && n.ends_with(".csv") && n.len() >= 19 && n[..n.len() - 4].ends_with(|c: char| c.is_ascii_digit()))
    {
        let stamp = CSV_STAMP.captures(&m).ok_or("no timestamp in the sync file")?[1].to_string();
        let st = pytime::parse_csv_stamp(&stamp).ok_or_else(|| format!("bad timestamp {}, expected YYYY-MM-DD_HH-MM-SS", repr_str(&stamp)))?;
        let content = read_text(&m)?;
        let mut n = 0_u64;
        for r in splitlines(&content).into_iter().skip(3).filter(|r| !r.trim().is_empty()) {
            let last = r.rsplit(',').next().unwrap_or("");
            let us = bench_iq::pyish::py_int(last).ok_or_else(|| format!("not an integer: {}", repr_str(last)))?;
            let t = st + pytime::microseconds(us as f64 / 1000.0);
            if w0 <= t && t <= w1 {
                n += 1;
            }
        }
        if n > 0 {
            let secs = pytime::total_seconds(w1 - w0);
            if secs == 0.0 {
                return Err("division by zero".into());
            }
            disp = Some(n as f64 / secs);
        }
    }
    let (mut gpu, mut power) = (None, None);
    if exists(&format!("{d}/gpu.csv")) {
        let content = read_text(&format!("{d}/gpu.csv"))?;
        let mut u: Vec<(f64, f64)> = Vec::new();
        let mut lines: Vec<&str> = content.split('\n').collect();
        // text-mode iteration has no empty piece after the last newline
        if lines.last() == Some(&"") {
            lines.pop();
        }
        for line in lines {
            let p: Vec<&str> = line.split(',').map(str::trim).collect();
            let stamp: String = p[0].chars().take(19).collect();
            let Some(t) = pytime::parse_gpu_stamp(&stamp) else { continue };
            if w0 <= t && t <= w1 {
                let (a, b) = (p.get(1).ok_or("a line has too few fields")?, p.get(2).ok_or("a line has too few fields")?);
                u.push((float(a)?, float(b)?));
            }
        }
        if !u.is_empty() {
            gpu = Some(py_sum(&u.iter().map(|x| x.0).collect::<Vec<_>>()) / u.len() as f64);
            power = Some(py_sum(&u.iter().map(|x| x.1).collect::<Vec<_>>()) / u.len() as f64);
        }
    }
    let log = read_text_lossy(&format!("{d}/launch.log"))?;
    let mut comp = Vec::new();
    for c in COMPOSITED.captures_iter(&log) {
        let x = float(&c[1])?;
        if x > 0.0 {
            comp.push(x);
        }
    }
    comp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = comp.get(comp.len() / 2).copied();
    let held = held_per_second(&log)?;
    let settings = if exists(&format!("{d}/settings")) { read_text(&format!("{d}/settings"))? } else { String::new() };
    let settings: Vec<&str> = settings.split_whitespace().collect();
    let fmt = |v: Option<f64>, prec: usize, suffix: &str| v.map_or("NA".to_string(), |v| format!("{}{suffix}", fixed(v, prec)));
    let fg = disp.filter(|&x| x != 0.0).map(|x| x / real4);
    let passes: Vec<String> = avgs.iter().map(|a| fixed(*a, 1)).collect();
    let mut line = format!(
        "{label}: passes {} | pass4 ({}s) real {}, displayed {} (frame generation {}), {}, GPU {} {}",
        passes.join(" / "),
        fixed(dur, 0),
        fixed(real4, 1),
        fmt(disp, 1, ""),
        fmt(fg, 1, "x"),
        rate_text(nf, held),
        fmt(gpu, 0, "%"),
        fmt(power, 0, "W")
    );
    let baseline_dir = baseline.filter(|b| !b.is_empty()).map(|b| join(base, b));
    line += &iq_text(&d, baseline_dir.as_deref(), fg)?;
    if !settings.is_empty() {
        line += &format!(" | {}", settings.iter().filter(|s| s.contains('=')).copied().collect::<Vec<_>>().join(" "));
    }
    println!("{line}");
    println!("    [sync] {}", sync_medians(&log)?);
    Ok(Some((real4, disp, gpu, fg)))
}

/// Whether a run's displayed / real ratio shows frame generation presenting frames.
fn engaged(fg: Option<f64>) -> bool {
    fg.is_some_and(|f| f >= FG_ENGAGED)
}

/// `os.path.expanduser` for `~` and `~/...`.
fn expanduser(p: &str) -> String {
    match (p.strip_prefix('~'), std::env::var("HOME")) {
        (Some(rest), Ok(home)) if rest.is_empty() || rest.starts_with('/') => format!("{}{rest}", home.trim_end_matches('/')),
        _ => p.to_string(),
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The command run on the host: the executable arrives on stdin, runs with `args` (already quoted) and is removed when the
/// remote shell ends, however it ends (exit, or the connection dropping and the shell being hung up).
fn remote_command(args: &str) -> String {
    format!("f=$(mktemp) || exit 1; trap 'rm -f \"$f\"' EXIT HUP INT TERM; cat > \"$f\" && chmod +x \"$f\" && \"$f\" bench-report {args}")
}

/// `--host`: this executable goes along to the host (the way the script sent bench-iq and itself to `python3 -`), runs
/// there with the remaining arguments and is removed again.
fn run_on_host(host: &str, rest: &[String]) -> Result<ExitCode, Fail> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let bytes = std::fs::read(&exe).map_err(|e| e.to_string())?;
    let args: Vec<String> = rest.iter().map(|a| shell_quote(a)).collect();
    let remote = remote_command(&args.join(" "));
    let mut child = Command::new("ssh").arg(host).arg(remote).stdin(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&bytes);
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    Ok(ExitCode::from(status.code().unwrap_or(255) as u8))
}

fn run(argv: &[String]) -> Result<ExitCode, Fail> {
    if argv.len() > 1 && argv[0] == "--host" {
        return run_on_host(&argv[1], &argv[2..]);
    }
    let mut argv: Vec<String> = argv.to_vec();
    let mut baseline = None;
    if let Some(i) = argv.iter().position(|a| a == "--baseline") {
        baseline = argv.get(i + 1).cloned();
        let end = (i + 2).min(argv.len());
        argv.drain(i..end);
    }
    let mean = argv.iter().any(|a| a == "--mean");
    let want_fg = argv.iter().any(|a| a == "--fg");
    let labels: Vec<&String> = argv.iter().filter(|a| *a != "--mean" && *a != "--fg").collect();
    if labels.is_empty() {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(2));
    }
    let base = expanduser(&std::env::var("NF_BENCH_DIR").unwrap_or_else(|_| "~/nf-spike/gta".to_string()));
    let mut results: Vec<(&str, Row)> = Vec::new();
    for label in labels {
        if let Some(r) = report(label, &base, baseline.as_deref())? {
            results.push((label, r));
        }
    }
    let missed: Vec<&str> = results.iter().filter(|(_, r)| want_fg && !engaged(r.3)).map(|(l, _)| *l).collect();
    for label in &missed {
        println!(
            "{label}: frame generation did not engage (displayed / real under {}x); GTA turned it off itself, so compare this run only with runs without frame generation",
            py_float_repr(FG_ENGAGED)
        );
    }
    let results: Vec<Row> = results.into_iter().filter(|(l, _)| !missed.contains(l)).map(|(_, r)| r).collect();
    if mean && !results.is_empty() {
        let avg = |xs: Vec<f64>| if xs.is_empty() { None } else { Some(py_sum(&xs) / xs.len() as f64) };
        let real = avg(results.iter().map(|r| r.0).collect()).unwrap_or(0.0);
        let disp = avg(results.iter().filter_map(|r| r.1).collect());
        let gpu = avg(results.iter().filter_map(|r| r.2).collect());
        let mut out = format!("mean of {}: real {}", results.len(), fixed(real, 1));
        if let Some(d) = disp.filter(|&d| d != 0.0) {
            out += &format!(", displayed {}", fixed(d, 1));
        }
        if let Some(g) = gpu.filter(|&g| g != 0.0) {
            out += &format!(", GPU {}%", fixed(g, 0));
        }
        println!("{out}");
    }
    Ok(if missed.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

pub fn main(args: &[String]) -> ExitCode {
    if args == ["--self-test"] {
        return match selftest::run() {
            Ok(()) => {
                println!("bench-report self-test ok");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("bench-report self-test failed: {e}");
                ExitCode::from(1)
            }
        };
    }
    match run(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("bench-report: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `remote_command` in a local shell with `binary` as the "executable" on stdin; returns the exit code and
    /// what is left in the shell's temporary directory.
    fn run_remote(binary: &str) -> (Option<i32>, usize) {
        let tmp = std::env::temp_dir().join(format!("nf-xtask-remote-test-{}-{}", std::process::id(), binary.len()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let mut child = Command::new("sh").arg("-c").arg(remote_command("'x'")).env("TMPDIR", &tmp).stdin(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(binary.as_bytes()).unwrap();
        let status = child.wait().unwrap();
        let left = std::fs::read_dir(&tmp).unwrap().count();
        std::fs::remove_dir_all(&tmp).unwrap();
        (status.code(), left)
    }

    #[test]
    fn bench_report_host_binary_is_removed_when_the_remote_shell_ends() {
        assert_eq!(run_remote("#!/bin/sh\nexit 7\n"), (Some(7), 0));
        // The shell is hung up while the binary runs (a dropped ssh connection): the copy must still go.
        let (_, left) = run_remote("#!/bin/sh\nkill -HUP $PPID\n");
        assert_eq!(left, 0, "the copied binary was left behind");
    }
}
