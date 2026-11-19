//! HTTP GET through the `curl` binary (no HTTP crate in the workspace; curl is in CI's image). It keeps what the Python
//! scripts' `urllib` calls did: redirects followed (at most 10), any final status outside 200-299 an error that carries
//! the status, reason and headers, and a timeout that applies to a stalled transfer rather than to the whole download.

use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub enum FetchError {
    /// The server answered with a status outside 200-299 (after redirects).
    Http { code: u16, reason: String, headers: Vec<(String, String)> },
    /// curl could not get an answer: DNS, connection, TLS, timeout, a dropped transfer.
    Io(String),
    /// curl itself could not be started.
    Tool(String),
}

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FetchError::Http { code, reason, .. } => write!(f, "HTTP Error {code}: {reason}"),
            FetchError::Io(m) | FetchError::Tool(m) => f.write_str(m),
        }
    }
}

impl FetchError {
    /// A response header of an HTTP error, by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        match self {
            FetchError::Http { headers, .. } => headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str()),
            _ => None,
        }
    }
}

/// The status line and headers of the last response in a curl `-D` dump (one block per redirect hop).
pub type Head = (u16, String, Vec<(String, String)>);

pub fn parse_headers(dump: &str) -> Option<Head> {
    let block = dump.split("\r\n\r\n").map(str::trim).filter(|b| b.starts_with("HTTP/")).last()?;
    let mut lines = block.lines();
    let mut status = lines.next()?.splitn(3, ' ');
    status.next()?;
    let code = status.next()?.trim().parse().ok()?;
    let reason = status.next().unwrap_or("").trim().to_string();
    let headers = lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
    Some((code, reason, headers))
}

fn temp_path() -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!("nf-xtask-curl-{}-{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)))
}

/// Streams the body of `url` into `sink`. `idle_secs` bounds the connect and any stall, like a socket timeout.
pub fn get_stream(url: &str, headers: &[(&str, &str)], idle_secs: u32, sink: &mut dyn FnMut(&[u8])) -> Result<(), FetchError> {
    let dump = temp_path();
    let result = stream(url, headers, idle_secs, &dump, sink);
    let _ = std::fs::remove_file(&dump);
    result
}

fn stream(url: &str, headers: &[(&str, &str)], idle_secs: u32, dump: &Path, sink: &mut dyn FnMut(&[u8])) -> Result<(), FetchError> {
    // Headers (the Authorization token among them) go in on stdin: argv is visible to every process on the machine.
    let mut config = String::new();
    for (k, v) in headers {
        let line = format!("{k}: {v}").replace('\\', "\\\\").replace('"', "\\\"");
        config.push_str(&format!("header = \"{line}\"\n"));
    }
    let mut child = Command::new("curl")
        .args(["--silent", "--show-error", "--location", "--max-redirs", "10", "--http1.1"])
        .args(["--connect-timeout", &idle_secs.to_string(), "--speed-time", &idle_secs.to_string(), "--speed-limit", "1"])
        .arg("--dump-header")
        .arg(dump)
        .args(["--config", "-", "--url", url])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| FetchError::Tool(format!("could not start curl: {e}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(config.as_bytes());
    }
    let mut stdout = child.stdout.take().ok_or_else(|| FetchError::Tool("curl has no stdout".into()))?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => sink(&buf[..n]),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(FetchError::Io(e.to_string()));
            }
        }
    }
    let out = child.wait_with_output().map_err(|e| FetchError::Tool(format!("could not wait for curl: {e}")))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(FetchError::Io(if why.is_empty() { format!("curl failed: {}", out.status) } else { why }));
    }
    let head = std::fs::read_to_string(dump).unwrap_or_default();
    let (code, reason, headers) = parse_headers(&head).ok_or_else(|| FetchError::Io("curl returned no HTTP status".into()))?;
    if (200..300).contains(&code) {
        Ok(())
    } else {
        Err(FetchError::Http { code, reason, headers })
    }
}

/// The whole body of `url`.
pub fn get(url: &str, headers: &[(&str, &str)], idle_secs: u32) -> Result<Vec<u8>, FetchError> {
    let mut body = Vec::new();
    get_stream(url, headers, idle_secs, &mut |chunk| body.extend_from_slice(chunk))?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_pins_http_headers_of_the_last_hop() {
        let dump = "HTTP/1.1 302 Found\r\nLocation: x\r\n\r\nHTTP/1.1 403 Forbidden\r\nX-RateLimit-Remaining: 0\r\nretry-after: 60\r\n\r\n";
        let (code, reason, headers) = parse_headers(dump).unwrap();
        assert_eq!((code, reason.as_str()), (403, "Forbidden"));
        let e = FetchError::Http { code, reason, headers };
        assert_eq!(e.header("x-ratelimit-remaining"), Some("0"));
        assert_eq!(e.header("Retry-After"), Some("60"));
        assert_eq!(e.header("missing"), None);
        assert_eq!(e.to_string(), "HTTP Error 403: Forbidden");
        assert!(parse_headers("").is_none());
    }
}
