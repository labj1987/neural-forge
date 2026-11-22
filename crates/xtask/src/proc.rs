//! The running processes, read from /proc: what the replaced scripts asked `pgrep` and `ps` for.

use regex::Regex;

/// Every process id, in ascending order.
pub fn pids() -> Vec<i32> {
    let mut pids: Vec<i32> =
        std::fs::read_dir("/proc").map(|entries| entries.flatten().filter_map(|e| e.file_name().to_string_lossy().parse().ok()).collect()).unwrap_or_default();
    pids.sort_unstable();
    pids
}

/// The process's name (at most 15 bytes of its executable's name).
pub fn comm(pid: i32) -> Option<String> {
    std::fs::read_to_string(format!("/proc/{pid}/comm")).ok().map(|name| name.trim_end_matches('\n').to_string())
}

/// The process's command line, its arguments joined by spaces; its name when it has none (`pgrep -f` reads the same).
pub fn cmdline(pid: i32) -> Option<String> {
    let bytes = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let args: Vec<String> = bytes.split(|&b| b == 0).map(|a| String::from_utf8_lossy(a).into_owned()).collect();
    let line = args.join(" ").trim_end().to_string();
    if line.is_empty() {
        comm(pid)
    } else {
        Some(line)
    }
}

/// The fields of /proc/PID/stat after the name: [0] is the state, [1] the parent's pid, [11] and [12] the user and
/// system CPU ticks (`stat`'s fields 3, 4, 14 and 15).
pub fn stat(path: &str) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(text[text.rfind(')')? + 1..].split_whitespace().map(String::from).collect())
}

pub fn ppid(pid: i32) -> Option<i32> {
    stat(&format!("/proc/{pid}/stat"))?.get(1)?.parse().ok()
}

/// `pgrep -x NAME`: the processes with exactly this name.
pub fn named(name: &str) -> Vec<i32> {
    pids().into_iter().filter(|&pid| comm(pid).is_some_and(|c| c == name)).collect()
}

/// `pgrep -f PATTERN`: the processes whose command line the pattern matches, this one left out.
pub fn matching(pattern: &Regex) -> Vec<i32> {
    let me = std::process::id() as i32;
    pids().into_iter().filter(|&pid| pid != me && cmdline(pid).is_some_and(|line| pattern.is_match(&line))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_process_is_listed_with_its_parent() {
        let me = std::process::id() as i32;
        assert!(pids().contains(&me));
        assert!(ppid(me).is_some_and(|p| p > 0));
        assert!(named(&comm(me).unwrap()).contains(&me));
        assert!(!matching(&Regex::new(".").unwrap()).contains(&me));
    }
}
