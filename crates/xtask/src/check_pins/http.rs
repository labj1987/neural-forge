//! HTTP GET through the `curl` binary (no HTTP crate in the workspace; curl is in CI's image). It keeps what the Python
//! scripts' `urllib` calls did: redirects followed (at most 10), any final status outside 200-299 an error that carries
//! the status, reason and headers, and a timeout that applies to a stalled transfer rather than to the whole download.

use std::fmt;
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug)]
pub enum FetchError {
    /// The server answered with a status outside 200-299 (after redirects).
    Http { code: u16, reason: String, headers: Vec<(String, String)> },
    /// curl could not get an answer: DNS, connection, TLS or certificate, proxy, timeout, a garbled or dropped reply,
    /// a redirect loop (see `NETWORK_EXITS`).
    Io(String),
    /// Anything else: curl could not be started, was killed, or exited with a code that is not a network problem
    /// (a refused option or protocol, a malformed URL, a local write error). That is a failure of the check, not a
    /// reason to retry later.
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

/// curl's exit codes that mean the network let it down: could not resolve the proxy (5) or host (6), could not
/// connect (7), a weird server reply (8), a partial transfer (18), a timeout (28), a TLS handshake failure (35), too many
/// redirects (47), an empty reply (52), a send or receive failure (55, 56), a peer certificate that did not verify (60),
/// a proxy handshake failure (97).
const NETWORK_EXITS: [i32; 13] = [5, 6, 7, 8, 18, 28, 35, 47, 52, 55, 56, 60, 97];

/// What a failed curl run (its exit code, None when a signal ended it, and its stderr) means.
fn classify_exit(code: Option<i32>, stderr: &str) -> FetchError {
    let why = stderr.trim();
    match code {
        Some(c) if NETWORK_EXITS.contains(&c) => FetchError::Io(if why.is_empty() { format!("curl failed: exit status {c}") } else { why.to_string() }),
        Some(c) => FetchError::Tool(if why.is_empty() { format!("curl failed: exit status {c}") } else { format!("curl failed (exit status {c}): {why}") }),
        None => FetchError::Tool(format!("curl was killed by a signal{}", if why.is_empty() { String::new() } else { format!(": {why}") })),
    }
}

/// A private (mode 0700) directory of this process's own, with an unguessable name `nf-xtask-<label>-<random>`, that
/// create_dir refuses when it is already taken (a header dump for curl, a scratch dir for a check).
pub(crate) fn private_dir(label: &str) -> std::io::Result<PathBuf> {
    let mut random = [0u8; 16];
    let from_system = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut random));
    if from_system.is_err() {
        // No /dev/urandom: the clock and the process id still make the name unique, and create_dir refuses a taken one.
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        random.copy_from_slice(&(nanos ^ (u128::from(std::process::id()) << 96)).to_le_bytes());
    }
    let name: String = random.iter().map(|b| format!("{b:02x}")).collect();
    let dir = std::env::temp_dir().join(format!("nf-xtask-{label}-{name}"));
    std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
    Ok(dir)
}

/// Streams the body of `url` into `sink`. `idle_secs` bounds the connect and any stall, like a socket timeout.
pub fn get_stream(url: &str, headers: &[(&str, &str)], idle_secs: u32, sink: &mut dyn FnMut(&[u8])) -> Result<(), FetchError> {
    let dir = private_dir("curl").map_err(|e| FetchError::Tool(format!("could not create a private temporary directory: {e}")))?;
    let dump = dir.join("headers");
    // create_new: never opens a file that is already there (or follows a link planted in its place).
    let created = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&dump);
    let result = match created {
        Ok(_) => stream(url, headers, idle_secs, &dump, sink),
        Err(e) => Err(FetchError::Tool(format!("could not create the header dump: {e}"))),
    };
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn stream(url: &str, headers: &[(&str, &str)], idle_secs: u32, dump: &Path, sink: &mut dyn FnMut(&[u8])) -> Result<(), FetchError> {
    // Headers (the Authorization token among them) go in on stdin: argv is visible to every process on the machine.
    let mut config = String::new();
    for (k, v) in headers {
        // A line break would end the config line and start another option.
        if k.contains(['\r', '\n']) || v.contains(['\r', '\n']) {
            return Err(FetchError::Tool(format!("refusing a request header with a line break in it: {k}")));
        }
        let line = format!("{k}: {v}").replace('\\', "\\\\").replace('"', "\\\"");
        config.push_str(&format!("header = \"{line}\"\n"));
    }
    let mut child = Command::new("curl")
        // -q first: no ~/.curlrc. https only, also after a redirect; no URL globbing.
        .args(["-q", "--globoff", "--proto", "=https", "--proto-redir", "=https"])
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
        return Err(classify_exit(out.status.code(), &String::from_utf8_lossy(&out.stderr)));
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

    #[test]
    fn check_pins_http_curl_exit_classes() {
        for c in NETWORK_EXITS {
            assert!(matches!(classify_exit(Some(c), "boom"), FetchError::Io(m) if m == "boom"), "exit {c} is the network");
        }
        assert!(matches!(classify_exit(Some(28), ""), FetchError::Io(m) if m == "curl failed: exit status 28"));
        // Not the network: an unsupported protocol (1), a malformed URL (3), a bad option (2), a write error (23),
        // an option curl does not know (49), a killed run. (A redirect loop, 47, is the server's: the network.)
        for c in [1, 2, 3, 23, 49, 84] {
            assert!(matches!(classify_exit(Some(c), "x"), FetchError::Tool(m) if m.contains(&format!("exit status {c}"))), "exit {c} is the tool");
        }
        assert!(matches!(classify_exit(None, ""), FetchError::Tool(_)));
    }

    #[test]
    fn check_pins_http_header_dump_is_private_and_unique() {
        let (a, b) = (private_dir("curl").unwrap(), private_dir("curl").unwrap());
        assert_ne!(a, b);
        let mode = std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&a).unwrap().permissions());
        assert_eq!(mode & 0o777, 0o700);
        std::fs::remove_dir_all(&a).unwrap();
        std::fs::remove_dir_all(&b).unwrap();
    }

    #[test]
    fn check_pins_http_refuses_a_header_with_a_line_break() {
        let e = get("https://localhost.invalid/", &[("Authorization", "Bearer a\nheader = \"x\"")], 1).unwrap_err();
        assert!(matches!(e, FetchError::Tool(_)), "{e}");
    }
}
