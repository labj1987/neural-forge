//! Checks a layer log for GPU stalls the layer caught and recovered from instead of freezing
//! the game: bounded fence-wait timeouts and the breadcrumb dump logged with each one. Also
//! summarizes engage/disengage transitions (loading-screen boundaries) and the presented
//! frame rate.
//!
//! Usage: cargo xtask check-stalls [LOG]
//!   LOG defaults to $NEURAL_FORGE_LOG, then ~/nf-layer.log (the file the game was launched
//!   with NEURAL_FORGE_LOG pointing at).
//! Exit status: 0 = no stalls found, 1 = stalls found, 2 = no log to check.
//!
//! It replaced scripts/check-stalls.sh: the same lines on stdout, byte for byte (the log is read as bytes, as grep and
//! sed read it).

use std::io::Write;
use std::process::ExitCode;

use regex::bytes::Regex;

const TIMEOUT: &[u8] = b"[layer] fence wait timed out";
const PRESENT: &[u8] = b"[present] ";

fn contains(line: &[u8], needle: &[u8]) -> bool {
    line.windows(needle.len()).any(|w| w == needle)
}

/// The log's lines (without their newlines; no empty last line after a final newline).
fn lines(log: &[u8]) -> Vec<&[u8]> {
    let mut lines: Vec<&[u8]> = log.split(|&b| b == b'\n').collect();
    if lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines
}

/// `grep -n -A 70 -F NEEDLE | head -90`: each matching line as `N:line`, the 70 lines after it as `N-line`, `--`
/// between groups that do not touch, the first 90 lines of all that.
fn timeouts_with_context(lines: &[&[u8]], out: &mut Vec<u8>) {
    let mut printed = 0usize;
    let mut context_until = 0usize;
    let mut last: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let matches = contains(line, TIMEOUT);
        if !matches && i >= context_until {
            continue;
        }
        let mut emit = |bytes: &[u8]| {
            if printed < 90 {
                out.extend_from_slice(bytes);
                out.push(b'\n');
            }
            printed += 1;
        };
        if last.is_some_and(|l| l + 1 != i) {
            emit(b"--");
        }
        let mut text = format!("{}{}", i + 1, if matches { ':' } else { '-' }).into_bytes();
        text.extend_from_slice(line);
        emit(&text);
        last = Some(i);
        if matches {
            context_until = i + 71;
        }
        if printed >= 90 {
            return;
        }
    }
}

fn report(log: &[u8]) -> (Vec<u8>, bool) {
    let lines = lines(log);
    let count = |needle: &[u8]| lines.iter().filter(|l| contains(l, needle)).count();
    let timeouts = count(TIMEOUT);
    let dumps = count(b"[breadcrumbs] ");
    let mut out = Vec::new();
    let _ = writeln!(out, "== fence wait timeouts: {timeouts} ==");
    if timeouts > 0 {
        // Each timeout is followed by its breadcrumb dump: the stage markers that led up to it.
        timeouts_with_context(&lines, &mut out);
    }
    let _ = writeln!(out, "== breadcrumb dumps: {dumps} ==");
    let _ = writeln!(out);
    let _ = writeln!(out, "== engage/disengage transitions (loading-screen boundaries) ==");
    let _ = writeln!(out, "{} engage, {} disengage", count(b"game is rendering steadily; engaging"), count(b"passing frames through untouched"));
    let _ = writeln!(out);
    let _ = writeln!(out, "== presented frame rate ==");
    let present: Vec<&[u8]> = lines.iter().copied().filter(|l| contains(l, PRESENT)).collect();
    if let Some(latest) = present.last() {
        let _ = writeln!(out, "{} sample(s); lowest and latest:", present.len());
        let rate = Regex::new(r"^(?s-u:.*)\[present\] ([0-9.]+) fps(?s-u:.*)$").unwrap();
        let mut rates: Vec<Vec<u8>> = present.iter().map(|l| rate.captures(l).map_or_else(|| l.to_vec(), |c| c[1].to_vec())).collect();
        let number = |line: &[u8]| crate::sh::leading_number(&String::from_utf8_lossy(line));
        rates.sort_by(|a, b| number(a).total_cmp(&number(b)).then_with(|| a.cmp(b)));
        out.extend_from_slice(b"  lowest: ");
        out.extend_from_slice(&rates[0]);
        out.extend_from_slice(b" fps\n");
        let at = latest.windows(PRESENT.len()).rposition(|w| w == PRESENT).map_or(0, |i| i + PRESENT.len());
        out.extend_from_slice(b"  latest: ");
        out.extend_from_slice(&latest[at..]);
        out.push(b'\n');
    } else {
        let _ = writeln!(out, "no [present] lines (the layer logs one every 5 s while presenting)");
    }
    let stalls = timeouts > 0 || dumps > 0;
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", if stalls { "STALLS FOUND" } else { "no stalls found" });
    (out, stalls)
}

pub fn main(args: &[String]) -> ExitCode {
    let home = std::env::var("HOME").unwrap_or_default();
    let log = args.first().filter(|a| !a.is_empty()).cloned().or_else(|| crate::sh::var("NEURAL_FORGE_LOG")).unwrap_or_else(|| format!("{home}/nf-layer.log"));
    if !std::path::Path::new(&log).is_file() {
        eprintln!("no log at {log} -- launch the game with NEURAL_FORGE_LOG={log} first, or pass the log path");
        return ExitCode::from(2);
    }
    let bytes = match std::fs::read(&log) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("{log}: {e}");
            return ExitCode::from(2);
        }
    };
    let (out, stalls) = report(&bytes);
    let _ = std::io::stdout().write_all(&out);
    ExitCode::from(u8::from(stalls))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(log: &str) -> (String, bool) {
        let (out, stalls) = report(log.as_bytes());
        (String::from_utf8(out).unwrap(), stalls)
    }

    #[test]
    fn a_clean_log_has_no_stalls() {
        let (out, stalls) =
            text("[layer] game is rendering steadily; engaging\n[present] 118.2 fps (59.0/s)\n[present] 97.5 fps (48.0/s)\n[present] 120.0 fps (60.0/s)\n");
        assert!(!stalls);
        assert_eq!(
            out,
            "== fence wait timeouts: 0 ==\n== breadcrumb dumps: 0 ==\n\n== engage/disengage transitions (loading-screen boundaries) ==\n1 engage, 0 disengage\n\n\
             == presented frame rate ==\n3 sample(s); lowest and latest:\n  lowest: 97.5 fps\n  latest: 120.0 fps (60.0/s)\n\nno stalls found\n"
        );
    }

    #[test]
    fn a_timeout_is_shown_with_its_context_as_grep_shows_it() {
        let mut log = String::from("a\n[layer] fence wait timed out x\n[breadcrumbs] b1\n");
        log.push_str(&"filler\n".repeat(75));
        log.push_str("[layer] fence wait timed out y\nend\n");
        let (out, stalls) = text(&log);
        assert!(stalls);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "== fence wait timeouts: 2 ==");
        assert_eq!(lines[1], "2:[layer] fence wait timed out x");
        assert_eq!(lines[2], "3-[breadcrumbs] b1");
        assert_eq!(lines[71], "72-filler");
        assert_eq!(lines[72], "--");
        assert_eq!(lines[73], "79:[layer] fence wait timed out y");
        assert_eq!(lines[74], "80-end");
        assert_eq!(lines[75], "== breadcrumb dumps: 1 ==");
        assert!(out.ends_with("no [present] lines (the layer logs one every 5 s while presenting)\n\nSTALLS FOUND\n"));
    }

    #[test]
    fn the_context_stops_after_ninety_lines() {
        let log = format!("[layer] fence wait timed out\n{}", "x\n".repeat(60)).repeat(3);
        let (out, _) = text(&log);
        let shown = out.lines().skip(1).take_while(|l| !l.starts_with("== breadcrumb")).count();
        assert_eq!(shown, 90);
    }
}
