//! Valve's text KeyValues format ("VDF"): `libraryfolders.vdf`, `appmanifest_*.acf` and
//! `userdata/<id>/config/localconfig.vdf` are all `"key" "value"` pairs and `"key" { ... }` blocks,
//! with `\" \\ \n \t` escapes in quoted strings, `//` line comments, bare (unquoted) tokens and an
//! optional `[$CONDITION]` after a value or a block's key.
//!
//! Reading builds an ordered tree (duplicate keys kept; lookups are case-insensitive, as Steam's
//! own are: its files spell `apps` and `Apps` both ways). Editing never re-serialises the tree:
//! [`set_string`] splices one quoted value (or inserts one key, or one block holding one key) into
//! the original bytes, so a file Steam also writes keeps every other byte, line ending and tab.
//! [`verify_edit`] then re-parses the result and requires it to equal the original tree plus
//! exactly that one change. Works on bytes, not `str`: nothing here assumes the file is UTF-8
//! outside the strings it reads.
//!
//! The splice-don't-serialise approach and the shape of the edit (replace a value, insert a key,
//! insert a block) follow DLSS5oneclick-forlinux's `src/platform/vdf.rs` (MIT; see ATTRIBUTION.md).

use std::ops::Range;

/// A value: a string or a block of entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    Block(Block),
}

/// One `key value [$CONDITION]` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub key: String,
    pub value: Value,
    pub condition: Option<String>,
}

/// Ordered entries; duplicate keys preserved.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block(pub Vec<Entry>);

impl Block {
    /// The first entry named `key` (ASCII case-insensitive).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.iter().find(|e| e.key.eq_ignore_ascii_case(key)).map(|e| &e.value)
    }

    pub fn path(&self, path: &[&str]) -> Option<&Value> {
        let (first, rest) = path.split_first()?;
        let mut cur = self.get(first)?;
        for key in rest {
            match cur {
                Value::Block(b) => cur = b.get(key)?,
                Value::Str(_) => return None,
            }
        }
        Some(cur)
    }

    pub fn block_at(&self, path: &[&str]) -> Option<&Block> {
        match self.path(path)? {
            Value::Block(b) => Some(b),
            Value::Str(_) => None,
        }
    }

    pub fn string_at(&self, path: &[&str]) -> Option<&str> {
        match self.path(path)? {
            Value::Str(s) => Some(s),
            Value::Block(_) => None,
        }
    }
}

/// Why a file could not be read or edited; carries the byte offset where it applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn err<T>(message: String) -> Result<T, Error> {
    Err(Error(message))
}

// ── tokenizer ──────────────────────────────────────────────────────

#[derive(Debug)]
enum Kind {
    Str(String),
    Open,
    Close,
    /// `[$WIN32]` and the like, brackets stripped.
    Condition(String),
}

#[derive(Debug)]
struct Token {
    kind: Kind,
    /// The token's source bytes, quotes included.
    span: Range<usize>,
}

fn unescape(raw: &[u8]) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'\\' && i + 1 < raw.len() {
            match raw[i + 1] {
                b'n' => out.push(b'\n'),
                b't' => out.push(b'\t'),
                b'\\' => out.push(b'\\'),
                b'"' => out.push(b'"'),
                // An unknown escape keeps both bytes, as Valve's lenient reader does.
                other => out.extend_from_slice(&[b'\\', other]),
            }
            i += 2;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `s` as a quoted VDF string, escaped so [`unescape`] gives `s` back.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn tokenize(b: &[u8]) -> Result<Vec<Token>, Error> {
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'{' || c == b'}' {
            tokens.push(Token { kind: if c == b'{' { Kind::Open } else { Kind::Close }, span: i..i + 1 });
            i += 1;
        } else if c == b'"' {
            let start = i;
            i += 1;
            loop {
                match b.get(i) {
                    None => return err(format!("unterminated string at byte {start}")),
                    Some(b'"') => break,
                    Some(b'\\') if i + 1 < b.len() => i += 2,
                    Some(_) => i += 1,
                }
            }
            i += 1;
            tokens.push(Token { kind: Kind::Str(unescape(&b[start + 1..i - 1])), span: start..i });
        } else if c == b'[' {
            let start = i;
            while i < b.len() && b[i] != b']' && b[i] != b'\n' {
                i += 1;
            }
            if b.get(i) != Some(&b']') {
                return err(format!("unterminated condition at byte {start}"));
            }
            i += 1;
            let inner = String::from_utf8_lossy(&b[start + 1..i - 1]).into_owned();
            tokens.push(Token { kind: Kind::Condition(inner), span: start..i });
        } else {
            let start = i;
            while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], b'{' | b'}' | b'"') {
                i += 1;
            }
            tokens.push(Token { kind: Kind::Str(String::from_utf8_lossy(&b[start..i]).into_owned()), span: start..i });
        }
    }
    Ok(tokens)
}

// ── the tree with byte spans (internal) ───────────────────────────

#[derive(Debug)]
enum SpannedValue {
    Str { span: Range<usize>, value: String },
    Block { inner: SpannedBlock, close: usize },
}

#[derive(Debug)]
struct SpannedEntry {
    key: String,
    key_span: Range<usize>,
    value: SpannedValue,
    condition: Option<String>,
}

#[derive(Debug, Default)]
struct SpannedBlock(Vec<SpannedEntry>);

impl SpannedBlock {
    fn get(&self, key: &str) -> Option<&SpannedEntry> {
        self.0.iter().find(|e| e.key.eq_ignore_ascii_case(key))
    }

    fn plain(&self) -> Block {
        Block(
            self.0
                .iter()
                .map(|e| Entry {
                    key: e.key.clone(),
                    value: match &e.value {
                        SpannedValue::Str { value, .. } => Value::Str(value.clone()),
                        SpannedValue::Block { inner, .. } => Value::Block(inner.plain()),
                    },
                    condition: e.condition.clone(),
                })
                .collect(),
        )
    }
}

fn take_condition(tokens: &[Token], pos: &mut usize) -> Option<String> {
    match tokens.get(*pos) {
        Some(Token { kind: Kind::Condition(c), .. }) => {
            *pos += 1;
            Some(c.clone())
        }
        _ => None,
    }
}

/// Parses entries up to the matching `}` (or the end, at the top level); returns the block and the
/// closing brace's offset.
fn parse_block(tokens: &[Token], pos: &mut usize, top: bool) -> Result<(SpannedBlock, usize), Error> {
    let mut out = SpannedBlock::default();
    loop {
        let Some(token) = tokens.get(*pos) else {
            return if top { Ok((out, usize::MAX)) } else { err("unexpected end of file inside a block".into()) };
        };
        *pos += 1;
        let key = match &token.kind {
            Kind::Close if !top => return Ok((out, token.span.start)),
            Kind::Close => return err(format!("stray '}}' at byte {}", token.span.start)),
            Kind::Open => return err(format!("'{{' without a key at byte {}", token.span.start)),
            Kind::Condition(_) => return err(format!("condition without a key at byte {}", token.span.start)),
            Kind::Str(key) => key.clone(),
        };
        let key_span = token.span.clone();
        let mut condition = take_condition(tokens, pos);
        let value = match tokens.get(*pos) {
            Some(Token { kind: Kind::Str(value), span }) => {
                *pos += 1;
                SpannedValue::Str { span: span.clone(), value: value.clone() }
            }
            Some(Token { kind: Kind::Open, .. }) => {
                *pos += 1;
                let (inner, close) = parse_block(tokens, pos, false)?;
                SpannedValue::Block { inner, close }
            }
            _ => return err(format!("key {key:?} at byte {} has no value", key_span.start)),
        };
        if condition.is_none() {
            condition = take_condition(tokens, pos);
        }
        out.0.push(SpannedEntry { key, key_span, value, condition });
    }
}

fn parse_spanned(text: &[u8]) -> Result<SpannedBlock, Error> {
    let tokens = tokenize(text)?;
    let mut pos = 0;
    Ok(parse_block(&tokens, &mut pos, true)?.0)
}

pub fn parse(text: &[u8]) -> Result<Block, Error> {
    Ok(parse_spanned(text)?.plain())
}

// ── editing ────────────────────────────────────────────────────────

/// What [`set_string`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// The existing value's quoted bytes were replaced.
    Replaced,
    /// The final key was added to its existing parent block.
    InsertedKey,
    /// The parent block (holding only the final key) was added to the grandparent.
    InsertedBlock,
}

/// Sets the string at `path` (matched case-insensitively), touching nothing else in `text`.
/// Replaces an existing value's quoted bytes; inserts a missing final key into its parent block
/// (indented like its siblings, with the file's own line ending); or, when the parent block is
/// missing too, inserts that block with the one key into the grandparent. Anything shallower
/// missing is an error: the caller falls back to showing the string rather than guessing at the
/// file's structure.
pub fn set_string(text: &[u8], path: &[&str], value: &str) -> Result<(Vec<u8>, Change), Error> {
    if path.len() < 2 {
        return err("refusing to edit a top-level key".into());
    }
    let root = parse_spanned(text)?;
    // The blocks the path resolves to, with each one's closing brace.
    let mut chain: Vec<(&SpannedBlock, usize)> = vec![(&root, usize::MAX)];
    for key in &path[..path.len() - 1] {
        match chain.last().unwrap().0.get(key).map(|e| &e.value) {
            Some(SpannedValue::Block { inner, close }) => chain.push((inner, *close)),
            Some(SpannedValue::Str { .. }) => return err(format!("{key:?} is a value, not a block")),
            None => break,
        }
    }
    let quoted = quote(value);
    let last = path[path.len() - 1];
    if chain.len() == path.len() {
        let (parent, close) = *chain.last().unwrap();
        return match parent.get(last).map(|e| &e.value) {
            Some(SpannedValue::Str { span, .. }) => Ok((splice(text, span.clone(), quoted.as_bytes()), Change::Replaced)),
            Some(SpannedValue::Block { .. }) => err(format!("{last:?} is a block, not a value")),
            None => {
                let layout = Layout::of(text, parent, close);
                let line = format!("{}{}{}{}", layout.indent, quote(last), layout.separator, quoted);
                Ok((insert_before_close(text, close, &line, &layout), Change::InsertedKey))
            }
        };
    }
    if chain.len() == path.len() - 1 && chain.len() > 1 {
        let (grandparent, close) = *chain.last().unwrap();
        let layout = Layout::of(text, grandparent, close);
        let parent_key = path[path.len() - 2];
        let (indent, eol) = (&layout.indent, layout.eol);
        let block = format!("{indent}{}{eol}{indent}{{{eol}{indent}\t{}{}{}{eol}{indent}}}", quote(parent_key), quote(last), layout.separator, quoted);
        return Ok((insert_before_close(text, close, &block, &layout), Change::InsertedBlock));
    }
    err(format!("cannot create {}: {:?} not found", path.join(" > "), path[chain.len() - 1]))
}

fn splice(text: &[u8], range: Range<usize>, with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + with.len());
    out.extend_from_slice(&text[..range.start]);
    out.extend_from_slice(with);
    out.extend_from_slice(&text[range.end..]);
    out
}

/// How a new entry in a block is laid out: copied from the block's existing entries when it has
/// any, else one tab deeper than its closing brace, Steam's own layout.
struct Layout {
    indent: String,
    separator: String,
    eol: &'static str,
    /// The closing brace is alone on its line (after indentation), so new lines go before that line.
    brace_on_own_line: bool,
}

impl Layout {
    fn of(text: &[u8], block: &SpannedBlock, close: usize) -> Layout {
        let line_start = |at: usize| text[..at].iter().rposition(|&c| c == b'\n').map_or(0, |p| p + 1);
        let leading = |at: usize| {
            let start = line_start(at);
            let prefix = &text[start..at];
            prefix.iter().all(|&c| c == b' ' || c == b'\t').then(|| String::from_utf8_lossy(prefix).into_owned())
        };
        let close_indent = leading(close);
        let indent = block
            .0
            .first()
            .and_then(|e| leading(e.key_span.start))
            .unwrap_or_else(|| format!("{}\t", close_indent.clone().unwrap_or_default()));
        let separator = block
            .0
            .iter()
            .find_map(|e| match &e.value {
                SpannedValue::Str { span, .. } if e.condition.is_none() => {
                    let gap = &text[e.key_span.end..span.start];
                    (!gap.is_empty() && gap.iter().all(|&c| c == b' ' || c == b'\t')).then(|| String::from_utf8_lossy(gap).into_owned())
                }
                _ => None,
            })
            .unwrap_or_else(|| "\t\t".to_string());
        let start = line_start(close);
        let eol = if start >= 2 && &text[start - 2..start] == b"\r\n" { "\r\n" } else { "\n" };
        Layout { indent, separator, eol, brace_on_own_line: close_indent.is_some() && start > 0 }
    }
}

/// Inserts `entry` (no line ending) into the block closed at `close`: as its own line before the
/// brace's line, or, when the brace shares its line with other tokens, inline before the brace.
fn insert_before_close(text: &[u8], close: usize, entry: &str, layout: &Layout) -> Vec<u8> {
    if layout.brace_on_own_line {
        let start = text[..close].iter().rposition(|&c| c == b'\n').map_or(0, |p| p + 1);
        splice(text, start..start, format!("{entry}{}", layout.eol).as_bytes())
    } else {
        splice(text, close..close, format!("{} ", entry.trim_start()).as_bytes())
    }
}

/// `tree` with the string at `path` set the way [`set_string`] sets it: the first matching entry
/// replaced, or the key (or its parent block) appended to its block. `None` where [`set_string`]
/// would refuse.
pub fn with_string(tree: &Block, path: &[&str], value: &str) -> Option<Block> {
    fn go(block: &mut Block, path: &[&str], value: &str, created: bool) -> Option<()> {
        let (first, rest) = path.split_first()?;
        let found = block.0.iter_mut().find(|e| e.key.eq_ignore_ascii_case(first));
        match (found, rest.is_empty()) {
            (Some(Entry { value: Value::Str(s), .. }), true) => {
                *s = value.to_string();
                Some(())
            }
            (Some(Entry { value: Value::Block(inner), .. }), false) => go(inner, rest, value, created),
            (None, true) => {
                block.0.push(Entry { key: first.to_string(), value: Value::Str(value.to_string()), condition: None });
                Some(())
            }
            (None, false) if rest.len() == 1 && !created => {
                let mut inner = Block::default();
                go(&mut inner, rest, value, true)?;
                block.0.push(Entry { key: first.to_string(), value: Value::Block(inner), condition: None });
                Some(())
            }
            _ => None,
        }
    }
    let mut out = tree.clone();
    go(&mut out, path, value, false)?;
    Some(out)
}

/// The edit's self-check: `new` must parse into exactly `old`'s tree with the one string at `path`
/// set to `value` (a replaced value, or one inserted key or block), nothing else added, removed,
/// reordered or changed.
pub fn verify_edit(old: &[u8], new: &[u8], path: &[&str], value: &str) -> bool {
    let (Ok(old_tree), Ok(new_tree)) = (parse(old), parse(new)) else { return false };
    with_string(&old_tree, path, value).is_some_and(|expected| expected == new_tree) && new_tree.string_at(path) == Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\"UserLocalConfigStore\"\n{\n\t\"Software\"\n\t{\n\t\t\"Valve\"\n\t\t{\n\t\t\t\"Steam\"\n\t\t\t{\n\t\t\t\t\"apps\"\n\t\t\t\t{\n\t\t\t\t\t\"1091500\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"LaunchOptions\"\t\t\"MANGOHUD=1 WINEDLLOVERRIDES=\\\"dxgi=n,b\\\" %command%\"\n\t\t\t\t\t\t\"Playtime\"\t\t\"100\"\n\t\t\t\t\t}\n\t\t\t\t\t\"489830\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"Playtime\"\t\t\"7\"\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n\t\"friends\"\n\t{\n\t\t\"x\"\t\t\"1\"\n\t}\n}\n";

    fn path(appid: &str) -> Vec<&str> {
        vec!["UserLocalConfigStore", "Software", "Valve", "Steam", "apps", appid, "LaunchOptions"]
    }

    #[test]
    fn reads_nested_blocks_and_escaped_quotes() {
        let tree = parse(SAMPLE.as_bytes()).unwrap();
        assert_eq!(tree.string_at(&path("1091500")), Some("MANGOHUD=1 WINEDLLOVERRIDES=\"dxgi=n,b\" %command%"));
        assert_eq!(tree.string_at(&path("489830")), None);
        let upper = ["userlocalconfigstore", "SOFTWARE", "valve", "STEAM", "Apps", "1091500", "launchoptions"];
        assert!(tree.string_at(&upper).is_some(), "lookups ignore case");
    }

    #[test]
    fn reads_comments_bare_tokens_conditions_and_duplicates() {
        let text = "// header\n\"k\" { bare token \"v\" \"1\" [$WIN32] \"v\" \"2\" [!$WIN32] \"c\" [$X] { \"a\" \"b\" } }\n\"k\" \"again\"\n";
        let tree = parse(text.as_bytes()).unwrap();
        assert_eq!(tree.0.len(), 2, "duplicate top-level keys are kept");
        let Some(Value::Block(k)) = tree.get("k") else { panic!() };
        assert_eq!(k.0[0], Entry { key: "bare".into(), value: Value::Str("token".into()), condition: None });
        assert_eq!(k.0[1].condition.as_deref(), Some("$WIN32"));
        assert_eq!(k.0[2].condition.as_deref(), Some("!$WIN32"));
        assert_eq!(k.0[3].condition.as_deref(), Some("$X"));
        assert_eq!(k.string_at(&["c", "a"]), Some("b"));
    }

    #[test]
    fn escapes_round_trip() {
        let s = "a \"quoted\" \\back\\ new\nline\ttab";
        let text = format!("\"k\"\t{}\n", quote(s));
        assert_eq!(parse(text.as_bytes()).unwrap().string_at(&["k"]), Some(s));
        // An unknown escape keeps its backslash.
        assert_eq!(parse(b"\"k\" \"C:\\x\"").unwrap().string_at(&["k"]), Some("C:\\x"));
    }

    #[test]
    fn malformed_files_are_errors() {
        for bad in ["\"k\" \"unterminated", "\"k\" {", "}", "\"k\"", "{ }", "\"k\" [$X"] {
            assert!(parse(bad.as_bytes()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn replacing_touches_only_the_value_bytes() {
        let p = path("1091500");
        let (out, change) = set_string(SAMPLE.as_bytes(), &p, "NEW %command%").unwrap();
        assert_eq!(change, Change::Replaced);
        let old_value = "\"MANGOHUD=1 WINEDLLOVERRIDES=\\\"dxgi=n,b\\\" %command%\"";
        assert_eq!(String::from_utf8(out.clone()).unwrap(), SAMPLE.replace(old_value, "\"NEW %command%\""));
        assert!(verify_edit(SAMPLE.as_bytes(), &out, &p, "NEW %command%"));
    }

    #[test]
    fn written_values_are_escaped() {
        let p = path("1091500");
        let value = "WINEDLLOVERRIDES=\"dxgi=n,b\" A='x y' %command%";
        let (out, _) = set_string(SAMPLE.as_bytes(), &p, value).unwrap();
        assert!(String::from_utf8_lossy(&out).contains("\"WINEDLLOVERRIDES=\\\"dxgi=n,b\\\" A='x y' %command%\""));
        assert_eq!(parse(&out).unwrap().string_at(&p), Some(value));
        assert!(verify_edit(SAMPLE.as_bytes(), &out, &p, value));
    }

    #[test]
    fn a_missing_key_is_inserted_as_one_line_with_its_siblings_indentation() {
        let p = path("489830");
        let (out, change) = set_string(SAMPLE.as_bytes(), &p, "X %command%").unwrap();
        assert_eq!(change, Change::InsertedKey);
        let sibling = "\t\t\t\t\t\t\"Playtime\"\t\t\"7\"\n";
        let expected = SAMPLE.replace(sibling, &format!("{sibling}\t\t\t\t\t\t\"LaunchOptions\"\t\t\"X %command%\"\n"));
        assert_eq!(String::from_utf8(out.clone()).unwrap(), expected);
        assert!(verify_edit(SAMPLE.as_bytes(), &out, &p, "X %command%"));
    }

    #[test]
    fn a_missing_appid_block_is_inserted_at_the_end_of_apps() {
        let p = path("999999");
        let (out, change) = set_string(SAMPLE.as_bytes(), &p, "Y %command%").unwrap();
        assert_eq!(change, Change::InsertedBlock);
        let text = String::from_utf8(out.clone()).unwrap();
        let block = "\t\t\t\t\t\"999999\"\n\t\t\t\t\t{\n\t\t\t\t\t\t\"LaunchOptions\"\t\t\"Y %command%\"\n\t\t\t\t\t}\n\t\t\t\t}\n";
        assert!(text.contains(&format!("\t\t\t\t\t}}\n{block}")), "{text}");
        assert_eq!(text.len(), SAMPLE.len() + block.len() - "\t\t\t\t}\n".len());
        assert!(verify_edit(SAMPLE.as_bytes(), &out, &p, "Y %command%"));
    }

    #[test]
    fn insertions_keep_crlf_line_endings() {
        let crlf = SAMPLE.replace('\n', "\r\n");
        for appid in ["489830", "999999"] {
            let (out, _) = set_string(crlf.as_bytes(), &path(appid), "Z %command%").unwrap();
            let text = String::from_utf8(out.clone()).unwrap();
            assert!(!text.replace("\r\n", "").contains('\n'), "a bare LF was written: {text:?}");
            assert!(verify_edit(crlf.as_bytes(), &out, &path(appid), "Z %command%"));
        }
        let (out, _) = set_string(crlf.as_bytes(), &path("1091500"), "W").unwrap();
        assert_eq!(out.len() as isize - crlf.len() as isize, "\"W\"".len() as isize - "\"MANGOHUD=1 WINEDLLOVERRIDES=\\\"dxgi=n,b\\\" %command%\"".len() as isize);
    }

    #[test]
    fn a_brace_sharing_its_line_gets_the_entry_inline() {
        let text = "\"a\" { \"b\" { \"apps\" { \"5\" { \"x\" \"1\" } } } }";
        let p = ["a", "b", "apps", "5", "LaunchOptions"];
        let (out, _) = set_string(text.as_bytes(), &p, "v").unwrap();
        assert!(verify_edit(text.as_bytes(), &out, &p, "v"), "{}", String::from_utf8_lossy(&out));
        let p = ["a", "b", "apps", "6", "LaunchOptions"];
        let (out, _) = set_string(text.as_bytes(), &p, "v").unwrap();
        assert!(verify_edit(text.as_bytes(), &out, &p, "v"), "{}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn refuses_when_more_than_the_last_two_levels_are_missing() {
        let text = "\"UserLocalConfigStore\"\n{\n\t\"Software\"\n\t{\n\t}\n}\n";
        assert!(set_string(text.as_bytes(), &path("1"), "v").is_err());
        assert!(set_string(SAMPLE.as_bytes(), &["UserLocalConfigStore", "friends"], "v").is_err(), "a block is not a value");
        assert!(set_string(SAMPLE.as_bytes(), &["UserLocalConfigStore"], "v").is_err());
    }

    #[test]
    fn verification_rejects_any_other_change() {
        let p = path("1091500");
        let (out, _) = set_string(SAMPLE.as_bytes(), &p, "NEW").unwrap();
        let mangled = String::from_utf8(out).unwrap().replace("\"Playtime\"\t\t\"100\"", "\"Playtime\"\t\t\"999\"");
        assert!(!verify_edit(SAMPLE.as_bytes(), mangled.as_bytes(), &p, "NEW"));
        assert!(!verify_edit(SAMPLE.as_bytes(), SAMPLE.as_bytes(), &p, "NEW"), "the value itself must have changed");
        let dropped = SAMPLE.replace("\t\t\"x\"\t\t\"1\"\n", "");
        assert!(!verify_edit(SAMPLE.as_bytes(), dropped.as_bytes(), &path("1091500"), "MANGOHUD=1 WINEDLLOVERRIDES=\"dxgi=n,b\" %command%"));
        // Whitespace and comments are not part of the tree: a re-indented file still verifies,
        // which is why set_string never re-serialises and the byte-level tests above exist.
        let reflowed = SAMPLE.replace("\t\t\"x\"", "  \"x\"");
        assert!(verify_edit(SAMPLE.as_bytes(), reflowed.as_bytes(), &["UserLocalConfigStore", "friends", "x"], "1"));
    }

    #[test]
    fn non_utf8_bytes_elsewhere_survive_an_edit() {
        let mut text = b"\"a\"\n{\n\t\"bin\"\t\t\"\xff\xfe\"\n\t\"apps\"\n\t{\n\t\t\"1\"\n\t\t{\n\t\t\t\"LaunchOptions\"\t\t\"\"\n\t\t}\n\t}\n}\n".to_vec();
        let p = ["a", "apps", "1", "LaunchOptions"];
        let (out, _) = set_string(&text, &p, "NEURAL_FORGE_ENABLE=1 %command%").unwrap();
        assert!(verify_edit(&text, &out, &p, "NEURAL_FORGE_ENABLE=1 %command%"));
        let at = text.windows(4).position(|w| w == b"\"\"\n\t").unwrap();
        text.splice(at..at + 2, b"\"NEURAL_FORGE_ENABLE=1 %command%\"".iter().copied());
        assert_eq!(out, text);
    }
}
