//! Python's text behaviour where the replaced scripts' output depended on it.

use serde_json::Value;

/// `str.splitlines()`: splits on every line boundary Python knows, the boundaries dropped, no empty last line.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|&(_, n)| n == '\n') {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `Path.read_text()`'s universal newlines: `\r\n` and a lone `\r` read as `\n`.
pub fn universal_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `repr()` of a string.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
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

/// `repr()` of what `json.loads` returns for `value` (`None` for a missing one).
pub fn repr(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".into(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::String(s)) => repr_str(s),
        Some(Value::Array(items)) => format!("[{}]", items.iter().map(|v| repr(Some(v))).collect::<Vec<_>>().join(", ")),
        Some(Value::Object(map)) => format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr(Some(v)))).collect::<Vec<_>>().join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd\n"), ["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n\nb"), ["a", "", "b"]);
        assert!(splitlines("").is_empty());
        assert_eq!(splitlines("x\u{2028}y"), ["x", "y"]);
    }

    #[test]
    fn repr_matches_python() {
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'\""), "'a\\'\"'");
        let v: Value = serde_json::from_str(r#"{"A": "1", "b": [true, null, 2]}"#).unwrap();
        assert_eq!(repr(Some(&v)), "{'A': '1', 'b': [True, None, 2]}");
        assert_eq!(repr(None), "None");
    }
}
