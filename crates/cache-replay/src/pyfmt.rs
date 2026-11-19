//! Python's text output: float `repr`, `%e`/`%f`/`%g` formatting and `json.dump(indent=1)`, so the
//! port prints and writes what cache_replay.py does.

use std::fmt::Write as _;

/// Shortest round-trip digits and the decimal exponent of a finite, non-zero `x` ("12345", -3 for 0.012345).
fn shortest(x: f64) -> (String, i32) {
    let s = format!("{:e}", x.abs());
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    (mant.replace('.', ""), exp.parse().unwrap_or(0))
}

fn special(x: f64) -> Option<&'static str> {
    if x.is_nan() {
        Some("nan")
    } else if x.is_infinite() {
        Some(if x > 0.0 { "inf" } else { "-inf" })
    } else {
        None
    }
}

/// `repr(float)`.
pub fn repr(x: f64) -> String {
    if let Some(s) = special(x) {
        return s.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let (digits, exp) = shortest(x);
    let n = digits.len() as i32;
    if (-4..16).contains(&exp) {
        if exp >= n - 1 {
            format!("{sign}{digits}{}.0", "0".repeat((exp - n + 1) as usize))
        } else if exp >= 0 {
            let (a, b) = digits.split_at(exp as usize + 1);
            format!("{sign}{a}.{b}")
        } else {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let (a, b) = digits.split_at(1);
        let mant = if b.is_empty() { a.to_string() } else { format!("{a}.{b}") };
        format!("{sign}{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `format(x, ".Ne")`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    if let Some(s) = special(x) {
        return s.to_string();
    }
    let s = format!("{x:.prec$e}");
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

/// `format(x, "W.Nf")` (W 0: no padding).
pub fn fmt_f(x: f64, width: usize, prec: usize) -> String {
    let s = special(x).map_or_else(|| format!("{x:.prec$}"), str::to_string);
    format!("{s:>width$}")
}

/// `format(x, "+W.Nf")`.
pub fn fmt_f_signed(x: f64, width: usize, prec: usize) -> String {
    let s = match special(x) {
        Some("nan") => "+nan".to_string(),
        Some(s) if !s.starts_with('-') => format!("+{s}"),
        Some(s) => s.to_string(),
        None => format!("{x:+.prec$}"),
    };
    format!("{s:>width$}")
}

/// `format(x, ".Ng")`.
pub fn fmt_g(x: f64, prec: usize) -> String {
    if let Some(s) = special(x) {
        return s.to_string();
    }
    let p = prec.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let e = fmt_e(x, p - 1);
    let exp: i32 = e.split_once('e').map_or(0, |(_, x)| x.parse().unwrap_or(0));
    let strip = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if exp >= -4 && exp < p as i32 {
        strip(format!("{x:.*}", (p as i32 - 1 - exp) as usize))
    } else {
        let (m, x) = e.split_once('e').unwrap_or((&e, ""));
        format!("{}e{x}", strip(m.to_string()))
    }
}

/// A JSON value with Python's dict order.
#[derive(Clone, Debug)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn obj(items: Vec<(&str, J)>) -> J {
        J::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
}

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    } else {
        repr(x)
    }
}

/// `json.dumps(v, indent=indent)`; `None`: the one-line form with ", " and ": ".
pub fn dumps(v: &J, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_json(&mut out, v, indent, 0);
    out
}

fn write_json(out: &mut String, v: &J, indent: Option<usize>, level: usize) {
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(i) => {
            let _ = write!(out, "{i}");
        }
        J::Float(x) => out.push_str(&json_float(*x)),
        J::Str(s) => json_str(out, s),
        J::List(items) if items.is_empty() => out.push_str("[]"),
        J::Obj(items) if items.is_empty() => out.push_str("{}"),
        J::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                sep(out, indent, level + 1, i == 0);
                write_json(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push(']');
        }
        J::Obj(items) => {
            out.push('{');
            for (i, (k, item)) in items.iter().enumerate() {
                sep(out, indent, level + 1, i == 0);
                json_str(out, k);
                out.push_str(": ");
                write_json(out, item, indent, level + 1);
            }
            close(out, indent, level);
            out.push('}');
        }
    }
}

fn sep(out: &mut String, indent: Option<usize>, level: usize, first: bool) {
    match indent {
        Some(n) => {
            if !first {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
        None if !first => out.push_str(", "),
        None => {}
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        assert_eq!(repr(1.0), "1.0");
        assert_eq!(repr(0.001953125), "0.001953125");
        assert_eq!(repr(1e-5), "1e-05");
        assert_eq!(repr(-0.20000000298023224), "-0.20000000298023224");
        assert_eq!(repr(1e16), "1e+16");
        assert_eq!(repr(123456.5), "123456.5");
        assert_eq!(fmt_e(1.5e-7, 2), "1.50e-07");
        assert_eq!(fmt_g(0.0123456789, 6), "0.0123457");
        assert_eq!(fmt_g(1234567.0, 6), "1.23457e+06");
        assert_eq!(fmt_g(0.5, 6), "0.5");
        assert_eq!(fmt_f_signed(0.25, 8, 4), " +0.2500");
    }

    #[test]
    fn json_matches_python_layout() {
        let v = J::obj(vec![("a", J::List(vec![J::Int(1), J::Float(0.5)])), ("b", J::Obj(vec![]))]);
        assert_eq!(dumps(&v, Some(1)), "{\n \"a\": [\n  1,\n  0.5\n ],\n \"b\": {}\n}");
        assert_eq!(dumps(&v, None), "{\"a\": [1, 0.5], \"b\": {}}");
    }
}
