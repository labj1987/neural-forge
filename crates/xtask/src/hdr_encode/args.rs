//! A small stand-in for the parts of Python's argparse the two scripts use: long options with unique-prefix
//! matching, `--opt=value`, negative numbers as values, `nargs` of one, `?` and `+`, choices, and argparse's
//! error messages (usage, `prog: error: message`, exit status 2).

use super::pyfmt::{parse_float, parse_int, str_repr};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum Conv {
    Str,
    Float,
    Int,
}

#[derive(Clone, Copy)]
pub enum Kind {
    Help,
    /// A flag that stores this value when present.
    Flag(bool),
    One(Conv),
    /// `nargs="+"`
    Many,
}

pub struct Opt {
    pub names: &'static [&'static str],
    pub dest: &'static str,
    pub kind: Kind,
    pub choices: Option<&'static [&'static str]>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Nargs {
    One,
    /// `nargs="?"`
    Optional,
    Plus,
}

pub struct Pos {
    pub dest: &'static str,
    pub nargs: Nargs,
    pub choices: Option<&'static [&'static str]>,
}

pub struct Spec {
    pub prog: &'static str,
    /// The `-h` output; its first paragraph is the usage shown with an error.
    pub help: &'static str,
    pub opts: Vec<Opt>,
    pub pos: Vec<Pos>,
}

#[derive(Clone, Debug)]
pub enum Val {
    Str(String),
    Float(f64),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
}

#[derive(Default, Debug)]
pub struct Parsed {
    pub vals: HashMap<&'static str, Val>,
    /// Arguments argparse could not place; the caller reports them.
    pub extras: Vec<String>,
}

impl Parsed {
    pub fn str(&self, k: &str) -> Option<&str> {
        match self.vals.get(k) {
            Some(Val::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn float(&self, k: &str) -> Option<f64> {
        match self.vals.get(k) {
            Some(Val::Float(f)) => Some(*f),
            _ => None,
        }
    }
    pub fn int(&self, k: &str) -> Option<i64> {
        match self.vals.get(k) {
            Some(Val::Int(i)) => Some(*i),
            _ => None,
        }
    }
    pub fn flag(&self, k: &str) -> bool {
        matches!(self.vals.get(k), Some(Val::Bool(true)))
    }
    pub fn list(&self, k: &str) -> Vec<String> {
        match self.vals.get(k) {
            Some(Val::List(l)) => l.clone(),
            _ => Vec::new(),
        }
    }
}

pub enum Outcome {
    Parsed(Parsed),
    /// Help was printed (exit 0) or an error was printed (exit 2).
    Exit(u8),
}

pub fn usage_of(help: &str) -> &str {
    help.split("\n\n").next().unwrap_or(help)
}

/// `parser.error(message)`.
pub fn error(prog: &str, help: &str, msg: &str) -> Outcome {
    eprintln!("{}\n{prog}: error: {msg}", usage_of(help));
    Outcome::Exit(2)
}

fn is_negative_number(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 2 || b[0] != b'-' {
        return false;
    }
    // Python 3.14's argparse takes "-" followed by a digit, or by "." and a digit, as a value rather than an option.
    let rest = s.as_bytes();
    rest[1].is_ascii_digit() || (rest[1] == b'.' && rest.get(2).is_some_and(u8::is_ascii_digit))
}

enum Tok {
    Positional,
    Option(usize, Option<String>),
    Unknown,
    Ambiguous(String),
}

fn classify(spec: &Spec, tok: &str) -> Tok {
    if !tok.starts_with('-') || tok.len() == 1 {
        return Tok::Positional;
    }
    let find_exact = |name: &str| spec.opts.iter().position(|o| o.names.contains(&name));
    if let Some(i) = find_exact(tok) {
        return Tok::Option(i, None);
    }
    let (name, explicit) = match tok.split_once('=') {
        Some((n, v)) => (n, Some(v.to_string())),
        None => (tok, None),
    };
    if explicit.is_some() {
        if let Some(i) = find_exact(name) {
            return Tok::Option(i, explicit);
        }
    }
    if name.starts_with("--") {
        let mut hits: Vec<(usize, &str)> = Vec::new();
        for (i, o) in spec.opts.iter().enumerate() {
            for n in o.names {
                if n.starts_with(name) {
                    hits.push((i, n));
                }
            }
        }
        if hits.len() > 1 {
            return Tok::Ambiguous(format!("ambiguous option: {name} could match {}", hits.iter().map(|h| h.1).collect::<Vec<_>>().join(", ")));
        }
        if let Some((i, _)) = hits.first() {
            return Tok::Option(*i, explicit);
        }
    }
    if is_negative_number(tok) || tok.contains(' ') {
        return Tok::Positional;
    }
    Tok::Unknown
}

fn convert(spec: &Spec, o: &Opt, name: &str, conv: Conv, raw: &str) -> Result<Val, Outcome> {
    let bad = |what: &str| error(spec.prog, spec.help, &format!("argument {name}: invalid {what} value: {}", str_repr(raw)));
    if let Some(ch) = o.choices {
        if !ch.contains(&raw) {
            return Err(error(spec.prog, spec.help, &format!("argument {name}: invalid choice: {} (choose from {})", str_repr(raw), ch.join(", "))));
        }
    }
    match conv {
        Conv::Str => Ok(Val::Str(raw.to_string())),
        Conv::Float => parse_float(raw).map(Val::Float).ok_or_else(|| bad("float")),
        Conv::Int => parse_int(raw).map(Val::Int).ok_or_else(|| bad("int")),
    }
}

/// Parses `args` (without the program or task name).
pub fn parse(spec: &Spec, args: &[String]) -> Outcome {
    let mut out = Parsed::default();
    for o in &spec.opts {
        if let Kind::Flag(v) = o.kind {
            out.vals.insert(o.dest, Val::Bool(!v));
        }
    }
    let mut next_pos = 0;
    let mut plus_done = false;
    let mut only_positional = false;
    let mut i = 0;
    // Positionals of the current run, handed out once the run ends (a `+` positional takes the rest of its run).
    let mut run: Vec<&str> = Vec::new();
    let flush = |run: &mut Vec<&str>, next_pos: &mut usize, plus_done: &mut bool, out: &mut Parsed| -> Result<(), Outcome> {
        let mut k = 0;
        while k < run.len() {
            if *next_pos >= spec.pos.len() || *plus_done {
                break;
            }
            let p = &spec.pos[*next_pos];
            let take = if p.nargs == Nargs::Plus { run.len() - k } else { 1 };
            for t in &run[k..k + take] {
                if let Some(ch) = p.choices {
                    if !ch.contains(t) {
                        return Err(error(
                            spec.prog,
                            spec.help,
                            &format!("argument {}: invalid choice: {} (choose from {})", p.dest, str_repr(t), ch.join(", ")),
                        ));
                    }
                }
            }
            let vals: Vec<String> = run[k..k + take].iter().map(|s| s.to_string()).collect();
            if p.nargs == Nargs::Plus {
                out.vals.insert(p.dest, Val::List(vals));
                *plus_done = true;
            } else {
                out.vals.insert(p.dest, Val::Str(vals[0].clone()));
            }
            k += take;
            *next_pos += 1;
        }
        out.extras.extend(run[k..].iter().map(|s| s.to_string()));
        run.clear();
        Ok(())
    };
    while i < args.len() {
        let tok = args[i].as_str();
        i += 1;
        if tok == "--" && !only_positional {
            only_positional = true;
            continue;
        }
        let kind = if only_positional { Tok::Positional } else { classify(spec, tok) };
        match kind {
            Tok::Positional => run.push(tok),
            Tok::Unknown => {
                if let Err(e) = flush(&mut run, &mut next_pos, &mut plus_done, &mut out) {
                    return e;
                }
                out.extras.push(tok.to_string());
            }
            Tok::Ambiguous(msg) => return error(spec.prog, spec.help, &msg),
            Tok::Option(idx, explicit) => {
                if let Err(e) = flush(&mut run, &mut next_pos, &mut plus_done, &mut out) {
                    return e;
                }
                let o = &spec.opts[idx];
                let name = o.names.join("/");
                match o.kind {
                    Kind::Help => {
                        print!("{}", spec.help);
                        return Outcome::Exit(0);
                    }
                    Kind::Flag(v) => {
                        if let Some(x) = explicit {
                            return error(spec.prog, spec.help, &format!("argument {name}: ignored explicit argument {}", str_repr(&x)));
                        }
                        out.vals.insert(o.dest, Val::Bool(v));
                    }
                    Kind::One(conv) => {
                        let raw = match explicit {
                            Some(x) => x,
                            None => {
                                if i < args.len() && matches!(classify(spec, &args[i]), Tok::Positional) {
                                    i += 1;
                                    args[i - 1].clone()
                                } else {
                                    return error(spec.prog, spec.help, &format!("argument {name}: expected one argument"));
                                }
                            }
                        };
                        match convert(spec, o, &name, conv, &raw) {
                            Ok(v) => {
                                out.vals.insert(o.dest, v);
                            }
                            Err(e) => return e,
                        }
                    }
                    Kind::Many => {
                        let mut items = Vec::new();
                        if let Some(x) = explicit {
                            items.push(x);
                        }
                        while i < args.len() && matches!(classify(spec, &args[i]), Tok::Positional) {
                            items.push(args[i].clone());
                            i += 1;
                        }
                        if items.is_empty() {
                            return error(spec.prog, spec.help, &format!("argument {name}: expected at least one argument"));
                        }
                        out.vals.insert(o.dest, Val::List(items));
                    }
                }
            }
        }
    }
    if let Err(e) = flush(&mut run, &mut next_pos, &mut plus_done, &mut out) {
        return e;
    }
    let missing: Vec<&str> = spec.pos[next_pos..].iter().filter(|p| p.nargs != Nargs::Optional).map(|p| p.dest).collect();
    if !missing.is_empty() {
        return error(spec.prog, spec.help, &format!("the following arguments are required: {}", missing.join(", ")));
    }
    Outcome::Parsed(out)
}
