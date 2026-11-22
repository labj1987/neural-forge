//! Python's number and string formatting, JSON writer, `float()`/`int()` parsing and the pieces of numpy
//! the two ported scripts rely on (pairwise summation, percentile, float16 conversion), reproduced bit for bit.

/// `repr(float)`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    // Rust's `{:e}` gives the shortest round-trip digits, as Python's repr does.
    let s = format!("{:e}", x.abs());
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let n = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let m = if n == 1 { digits.clone() } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        let e = decpt - 1;
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= n {
        format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    format!("{sign}{body}")
}

/// `format(x, ".{prec}g")`.
pub fn fmt_g(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let p = prec.max(1);
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0");
    }
    let s = format!("{:.*e}", p - 1, x.abs());
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let strip = |t: &str| -> String {
        if t.contains('.') {
            t.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            t.to_string()
        }
    };
    if exp < -4 || exp >= p as i32 {
        format!("{sign}{}e{}{:02}", strip(mant), if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        format!("{sign}{}", strip(&format!("{:.*}", decimals, x.abs())))
    }
}

/// `format(x, ".{prec}f")`.
pub fn fmt_f(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        format!("{x:.prec$}")
    }
}

/// `repr(str)`.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr()` of a value read with `json.load` (what the scripts print with an f-string).
pub fn json_value_repr(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                float_repr(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(json_value_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", str_repr(k), json_value_repr(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// `str()` of a value read with `json.load`.
pub fn json_value_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        _ => json_value_repr(v),
    }
}

/// `float(str)`.
pub fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let b = t.as_bytes();
    // Underscores are allowed between digits only.
    if t.contains('_') {
        for (i, c) in b.iter().enumerate() {
            if *c == b'_' && !(i > 0 && b[i - 1].is_ascii_digit() && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
    }
    let t = t.replace('_', "");
    let lower = t.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    let signs = lower.len() - unsigned.len();
    if signs > 1 {
        return None;
    }
    let neg = lower.starts_with('-');
    match unsigned {
        "inf" | "infinity" => return Some(if neg { f64::NEG_INFINITY } else { f64::INFINITY }),
        "nan" => return Some(f64::NAN),
        _ => {}
    }
    if unsigned.is_empty() || !unsigned.bytes().all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'+' | b'-')) {
        return None;
    }
    lower.parse::<f64>().ok()
}

/// `int(str)` (base 10).
pub fn parse_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let b = t.as_bytes();
    if t.contains('_') {
        for (i, c) in b.iter().enumerate() {
            if *c == b'_' && !(i > 0 && b[i - 1].is_ascii_digit() && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                return None;
            }
        }
    }
    t.replace('_', "").parse::<i64>().ok()
}

// ---- JSON (json.dumps) ----

pub enum Json {
    Null,
    Str(String),
    Float(f64),
    Int(i64),
    List(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        float_repr(x)
    }
}

/// `json.dumps(value)` (indent None: ", " and ": ") or `json.dumps(value, indent=1)`.
pub fn json_dumps(v: &Json, indent: Option<usize>) -> String {
    let mut out = String::new();
    json_write(v, indent, false, 0, &mut out);
    out
}

/// `json.dumps(value, separators=(",", ":"))`
pub fn json_dumps_compact(v: &Json) -> String {
    let mut out = String::new();
    json_write(v, None, true, 0, &mut out);
    out
}

fn json_write(v: &Json, indent: Option<usize>, compact: bool, level: usize, out: &mut String) {
    let (open_sep, item_sep, close_sep) = match indent {
        Some(n) => (format!("\n{}", " ".repeat(n * (level + 1))), format!(",\n{}", " ".repeat(n * (level + 1))), format!("\n{}", " ".repeat(n * level))),
        None => (String::new(), if compact { "," } else { ", " }.to_string(), String::new()),
    };
    let colon = if compact { ":" } else { ": " };
    match v {
        Json::Null => out.push_str("null"),
        Json::Str(s) => json_str(s, out),
        Json::Float(x) => out.push_str(&json_float(*x)),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::List(items) if items.is_empty() => out.push_str("[]"),
        Json::List(items) => {
            out.push('[');
            out.push_str(&open_sep);
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&item_sep);
                }
                json_write(item, indent, compact, level + 1, out);
            }
            out.push_str(&close_sep);
            out.push(']');
        }
        Json::Obj(items) if items.is_empty() => out.push_str("{}"),
        Json::Obj(items) => {
            out.push('{');
            out.push_str(&open_sep);
            for (i, (k, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&item_sep);
                }
                json_str(k, out);
                out.push_str(colon);
                json_write(item, indent, compact, level + 1, out);
            }
            out.push_str(&close_sep);
            out.push('}');
        }
    }
}

// ---- numpy ----

/// numpy's pairwise summation of a contiguous (or gathered) float64 run: blocks of 8 up to 128 elements, halves above.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `a.mean()` over everything (an empty slice gives NaN).
pub fn mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(a) / a.len() as f64
}

/// `x.reshape(-1, 3).mean(axis=0)`: a running sum per column.
pub fn mean_columns3(a: &[f64]) -> [f64; 3] {
    let mut s = [0.0; 3];
    for px in a.chunks_exact(3) {
        for c in 0..3 {
            s[c] += px[c];
        }
    }
    let n = (a.len() / 3) as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// numpy's `min`/`max` propagate NaN.
pub fn np_min(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::INFINITY, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else if x < m {
            x
        } else {
            m
        }
    })
}

pub fn np_max(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else if x > m {
            x
        } else {
            m
        }
    })
}

/// `np.maximum(a, b)` (NaN wins).
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `np.percentile(a, q)` with the default linear method; `q` in 0..=100 is the caller's to check.
pub fn percentile(a: &[f64], q: f64) -> f64 {
    if a.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let mut v = a.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    let virt = (n as f64 - 1.0) * (q / 100.0);
    let (prev, next) = if virt >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if virt < 0.0 {
        (0, 0)
    } else {
        (virt.floor() as usize, virt.floor() as usize + 1)
    };
    let gamma = virt - prev as f64;
    let (a0, b0) = (v[prev], v[next]);
    let diff = b0 - a0;
    let mut r = a0 + diff * gamma;
    if gamma >= 0.5 {
        r = b0 - diff * (1.0 - gamma);
    }
    r
}

/// float16 bits to float64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f64;
    match exp {
        0 => sign * man * 2f64.powi(-24),
        0x1f => {
            if man == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// `x.astype(np.float16)` straight from float64, round to nearest even; returns the bits.
pub fn f64_to_f16(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7c00;
    }
    if a < 2f64.powi(-14) {
        // Subnormals: a multiple of 2^-24; 1024 carries into the smallest normal.
        return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
    }
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut q = (a / 2f64.powi(e - 10)).round_ties_even() as u32;
    let mut e = e;
    if q == 2048 {
        q = 1024;
        e += 1;
    }
    sign | ((((e + 15) as u32) << 10) | (q - 1024)) as u16
}
