//! Python's behaviour where the two scripts depend on it: `sum` of floats, `float()`/`int()` parsing, `repr`, `{:.Nf}`,
//! `os.path.join`, text-mode newline translation. Shared by bench-iq and bench-report.

use std::fs;

/// `sum()` of floats: Python 3.12 and later add from the second item on with Neumaier compensation.
pub(crate) fn py_sum(v: &[f64]) -> f64 {
    let Some((&first, rest)) = v.split_first() else { return 0.0 };
    let (mut f, mut c) = (first, 0.0_f64);
    for &x in rest {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// `float(s)`: surrounding whitespace and single underscores between digits are allowed.
pub(crate) fn py_float(s: &str) -> Option<f64> {
    let t = underscores_out(s.trim())?;
    t.parse::<f64>().ok()
}

/// `int(s)` in base 10.
pub(crate) fn py_int(s: &str) -> Option<i64> {
    let t = underscores_out(s.trim())?;
    t.parse::<i64>().ok()
}

fn underscores_out(t: &str) -> Option<String> {
    if !t.contains('_') {
        return Some(t.to_string());
    }
    let chars: Vec<char> = t.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' {
            let digit = |j: Option<usize>| j.and_then(|j| chars.get(j)).is_some_and(|c| c.is_ascii_digit());
            if !(digit(i.checked_sub(1)) && digit(Some(i + 1))) {
                return None;
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// `repr(float)` / `str(float)`: the shortest round-trip digits, exponent form below 1e-4 and from 1e16.
pub(crate) fn py_float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if v < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let body = if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `'{:.Nf}'.format(v)`; Rust rounds exact ties to even like Python, only the spelling of nan differs.
pub(crate) fn fixed(v: f64, precision: usize) -> String {
    if v.is_nan() {
        "nan".into()
    } else {
        format!("{v:.precision$}")
    }
}

/// `os.path.join(a, b)`.
pub(crate) fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        b.to_string()
    } else if a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// `os.path.basename(os.path.normpath(p))`.
pub(crate) fn normpath_basename(p: &str) -> String {
    let absolute = p.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." if parts.last().is_some_and(|l| *l != "..") => {
                parts.pop();
            }
            ".." if absolute => {}
            c => parts.push(c),
        }
    }
    parts.last().map_or(if p.is_empty() || absolute { "" } else { "." }, |l| l).to_string()
}

/// `open(path).read()`: UTF-8 (strict), line ends translated. The error is a plain message naming the path.
pub(crate) fn read_text(path: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    String::from_utf8(bytes).map(|s| crate::py::universal_newlines(&s)).map_err(|e| format!("{path} is not valid UTF-8: {e}"))
}

/// `open(path, errors="replace").read()`.
pub(crate) fn read_text_lossy(path: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(crate::py::universal_newlines(&String::from_utf8_lossy(&bytes)))
}

/// An `OSError`-style text, `[Errno 2] No such file or directory: 'path'`: bench-report prints it on stdout for a frame it
/// cannot read, so it stays as it was.
pub(crate) fn os_error(e: &std::io::Error, path: &str) -> String {
    let text = e.to_string();
    let text = text.split(" (os error").next().unwrap_or(&text);
    match e.raw_os_error() {
        Some(n) => format!("[Errno {n}] {text}: {}", crate::py::repr_str(path)),
        None => format!("{text}: {}", crate::py::repr_str(path)),
    }
}

pub(crate) fn is_dir(p: &str) -> bool {
    fs::metadata(p).is_ok_and(|m| m.is_dir())
}

pub(crate) fn exists(p: &str) -> bool {
    fs::metadata(p).is_ok()
}

/// The sorted names in `dir` that `glob` would give for `accept` (hidden names are skipped like `*` does), as `dir/name`.
pub(crate) fn glob_dir(dir: &str, accept: impl Fn(&str) -> bool) -> Vec<String> {
    let Ok(rd) = fs::read_dir(if dir.is_empty() { "." } else { dir }) else { return Vec::new() };
    let mut names: Vec<String> = rd.filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| !n.starts_with('.') && accept(n)).collect();
    names.sort();
    names.into_iter().map(|n| join(dir, &n)).collect()
}
