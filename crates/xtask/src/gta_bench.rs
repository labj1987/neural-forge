//! `cargo xtask gta-bench` (it replaced the Python program inside scripts/gta-bench.sh, command for command): the checks that
//! read GTA's crash dumps and MangoHud's log, and the Steam lookups that find the Proton the game runs with.
//!
//! gta-bench dumps <CrashLogs dir> <epoch>: one line per dump written since <epoch>.
//! gta-bench fg <MangoHud csv> <cap fps>: "stopped <s> <shown> <best>" or "kept".
//! gta-bench proton <steam root> <appid> [libraries...]: the Proton folder Steam runs the app with.
//! gta-bench self-test.
//!
//! Python stopped on an unexpected failure with a traceback and exit 1; here the last traceback line (`Class: message`) goes to
//! stderr and the exit code is 1 as well. Directory listings are in name order (Python's glob gives the OS order, which only
//! shows with equal modification times), and `glob` metacharacters in a directory name are plain text here.

use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::process::ExitCode;
use std::sync::LazyLock;

use regex::Regex;

use crate::bench_iq::pyish::{glob_dir, join, os_error, py_float, py_int, py_repr_str, read_text_lossy, splitlines};

/// GTA5_Enhanced.exe (2026-09-15 build): the instruction after its hang watchdog's call.
const WATCHDOG: u64 = 0x1_4012_c6eb;
const IMAGE_BASE: u64 = 0x1_4000_0000;
const IMAGE_END: u64 = 0x1_4600_0000;

/// Valve's Protons: the name CompatToolMapping gives each and its Steam app (crates/supervisor/src/proton_tool.rs).
const VALVE_TOOLS: [(&str, &str); 9] = [
    ("proton_experimental", "1493710"),
    ("proton_hotfix", "2180100"),
    ("proton_11", "4628710"),
    ("proton_10", "3658110"),
    ("proton_9", "2805730"),
    ("proton_8", "2348590"),
    ("proton_7", "1887720"),
    ("proton_63", "1580130"),
    ("proton_513", "1420170"),
];

/// A Python exception that ends the program (its `Class: message` line).
type Fail = String;

fn io_fail(e: &std::io::Error, path: &str) -> Fail {
    let class = match e.raw_os_error() {
        Some(2) => "FileNotFoundError",
        Some(13) => "PermissionError",
        Some(17) => "FileExistsError",
        Some(20) => "NotADirectoryError",
        Some(21) => "IsADirectoryError",
        _ => "OSError",
    };
    format!("{class}: {}", os_error(e, path))
}

/// `open(path, errors="replace").read()` as a failure line.
fn read_lossy(path: &str) -> Result<String, Fail> {
    read_text_lossy(path).map_err(|e| {
        // read_text_lossy gives the OSError text without its class; the class follows from the errno in the text.
        let class = match e.split(|c: char| !c.is_ascii_digit()).find(|s| !s.is_empty()) {
            Some("2") => "FileNotFoundError",
            Some("13") => "PermissionError",
            Some("20") => "NotADirectoryError",
            Some("21") => "IsADirectoryError",
            _ => "OSError",
        };
        format!("{class}: {e}")
    })
}

/// `struct.unpack_from(fmt, data, offset)` for a format of `size` bytes: the bytes, or struct's own error message (it names
/// the whole format's size, so a multi-field format is read in one go here too).
fn unpack(data: &[u8], off: i128, size: usize) -> Result<&[u8], Fail> {
    if off > i64::MAX as i128 {
        return Err("OverflowError: Python int too large to convert to C ssize_t".to_string());
    }
    if off < 0 {
        return Err(format!("struct.error: offset {off} out of range for {}-byte buffer", data.len()));
    }
    if off + size as i128 > data.len() as i128 {
        return Err(format!(
            "struct.error: unpack_from requires a buffer of at least {} bytes for unpacking {size} bytes at offset {off} (actual buffer size is {})",
            off + size as i128,
            data.len()
        ));
    }
    Ok(&data[off as usize..off as usize + size])
}

/// The little-endian integer in `b`.
fn le(b: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    w[..b.len()].copy_from_slice(b);
    u64::from_le_bytes(w)
}

fn u32_at(data: &[u8], off: i128) -> Result<u64, Fail> {
    unpack(data, off, 4).map(le)
}

fn u64_at(data: &[u8], off: i128) -> Result<u64, Fail> {
    unpack(data, off, 8).map(le)
}

/// What a GTA minidump says: the faulting site and whether it is the hang watchdog.
fn dump_cause(data: &[u8]) -> Result<String, Fail> {
    if data.len() < 32 || &data[..4] != b"MDMP" {
        return Ok("empty dump (the game died while writing it)".to_string());
    }
    let head = unpack(data, 0, 16)?;
    let (nstreams, rva_dir) = (le(&head[8..12]), le(&head[12..]) as i128);
    let mut streams: Vec<(u64, i128)> = Vec::new();
    for i in 0..nstreams {
        let at = rva_dir + 12 * i as i128;
        let entry = unpack(data, at, 12)?;
        streams.push((le(&entry[..4]), le(&entry[8..]) as i128));
    }
    // A later entry of the same kind replaces an earlier one.
    let stream = |kind: u64| streams.iter().rev().find(|(k, _)| *k == kind).map(|(_, r)| *r);
    let Some(rva) = stream(6) else {
        return Ok("dump without an exception record".to_string());
    };
    let code = u32_at(data, rva + 8)?;
    let addr = u64_at(data, rva + 24)?;
    let ctx = u32_at(data, rva + 8 + 156)? as i128;
    let rsp = u64_at(data, ctx + 0x98)? as i128;
    let mut ret: Option<u64> = None;
    // The memory lists: the word at rsp is the return address of a call.
    for kind in [5u64, 9] {
        let Some(r) = stream(kind) else { continue };
        let mut ranges: Vec<(i128, i128, i128)> = Vec::new();
        if kind == 5 {
            let n = u32_at(data, r)?;
            for i in 0..n {
                let at = r + 4 + 16 * i as i128;
                let e = unpack(data, at, 16)?;
                ranges.push((le(&e[..8]) as i128, le(&e[8..12]) as i128, le(&e[12..]) as i128));
            }
        } else {
            let head = unpack(data, r, 16)?;
            let (n, mut off) = (le(&head[..8]), le(&head[8..]) as i128);
            for i in 0..n {
                let at = r + 16 + 16 * i as i128;
                let e = unpack(data, at, 16)?;
                let (start, size) = (le(&e[..8]) as i128, le(&e[8..]) as i128);
                ranges.push((start, size, off));
                off += size;
            }
        }
        for (start, size, at) in ranges {
            if start <= rsp && rsp < start + size - 8 {
                ret = Some(u64_at(data, at + rsp - start)?);
            }
        }
    }
    let site = if (IMAGE_BASE..IMAGE_END).contains(&addr) { format!("GTA5_Enhanced.exe+{:#x}", addr - IMAGE_BASE) } else { format!("{addr:#x}") };
    if addr == WATCHDOG {
        return Ok(format!("GTA's hang watchdog (main loop stalled 60 s, {site})"));
    }
    if ret == Some(WATCHDOG) {
        return Ok(format!("GTA's hang watchdog (main loop stalled 60 s, stage handler {site})"));
    }
    Ok(format!("exception {code:#010x} at {site}"))
}

/// A file's modification time the way Python's `os.path.getmtime` reports it.
fn mtime(path: &str) -> Result<f64, Fail> {
    let m = fs::metadata(path).map_err(|e| io_fail(&e, path))?;
    Ok(m.mtime() as f64 + m.mtime_nsec() as f64 * 1e-9)
}

fn dumps(folder: &str, since: f64) -> Result<(), Fail> {
    let mut found = Vec::new();
    for path in glob_dir(folder, |n| n.ends_with(".dmp")) {
        found.push((mtime(&path)?, path));
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, path) in found {
        if mtime(&path)? >= since {
            let data = fs::read(&path).map_err(|e| io_fail(&e, &path))?;
            let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            println!("{name}: {}", dump_cause(&data)?);
        }
    }
    Ok(())
}

/// Frame generation stopped: the displayed rate over 10 s windows (every 5 s) under a third of the best window (ignoring
/// loading-screen bursts above cap) for 20 s, once the best passed 150 fps.
fn fg_rule(times: &[f64], cap: f64) -> String {
    let (mut best, mut low, mut t) = (0.0f64, 0, 10.0f64);
    while let Some(&last) = times.last() {
        if t > last {
            break;
        }
        let n = times.iter().filter(|&&x| t - 10.0 <= x && x < t).count() as f64 / 10.0;
        if best < n && n <= cap {
            best = n;
        }
        if best >= 150.0 && n < best / 3.0 {
            low += 1;
            if low == 4 {
                return format!("stopped {:.0} {n:.0} {best:.0}", t - 25.0);
            }
        } else {
            low = 0;
        }
        t += 5.0;
    }
    "kept".to_string()
}

/// `int(s)` as a float: Python's integers do not overflow, so a long digit string still counts.
fn int_as_f64(s: &str) -> Option<f64> {
    if let Some(v) = py_int(s) {
        return Some(v as f64);
    }
    let t = s.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    let ok = !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit() || b == b'_')
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__");
    if ok {
        t.replace('_', "").parse::<f64>().ok()
    } else {
        None
    }
}

fn fg(csv: &str, cap: &str) -> Result<(), Fail> {
    let text = read_lossy(csv)?;
    let times: Vec<f64> = splitlines(&text)
        .into_iter()
        .skip(3)
        .filter(|r| r.matches(',').count() > 3)
        .filter_map(|r| int_as_f64(r.rsplit(',').next().unwrap_or("")).map(|v| v / 1e9))
        .collect();
    let cap = py_float(cap).ok_or_else(|| format!("ValueError: could not convert string to float: {}", py_repr_str(cap)))?;
    println!("{}", fg_rule(&times, cap));
    Ok(())
}

static VDF_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""((?:[^"\\]|\\.)*)"|(//[^\n]*)|([{}])"#).expect("static regex"));

/// One entry of a VDF block: the text, whether it is a brace (`{` or `}`) and the nesting depth inside the block.
type Entry = (String, String, usize);

/// The text inside the nested "key" { ... } blocks of `path` (a plain VDF reader: keys, strings, braces; a // comment runs to
/// the end of its line, but // inside a string is part of it).
fn vdf_block(text: &str, path: &[&str]) -> Option<Vec<Entry>> {
    let mut tokens: Vec<(String, String)> = Vec::new();
    for c in VDF_TOKEN.captures_iter(text) {
        if c.get(2).is_none() {
            tokens.push((c.get(1).map_or("", |m| m.as_str()).to_string(), c.get(3).map_or("", |m| m.as_str()).to_string()));
        }
    }
    let mut i = 0;
    for key in path {
        let mut depth = 0i32;
        let mut found = false;
        while i < tokens.len() {
            let (s, brace) = &tokens[i];
            i += 1;
            if brace == "{" {
                depth += 1;
            } else if brace == "}" {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            } else if depth == 0 && s.to_lowercase() == key.to_lowercase() && i < tokens.len() && tokens[i].1 == "{" {
                i += 1;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }
    }
    let mut out = Vec::new();
    let mut depth = 0usize;
    while i < tokens.len() {
        let (s, brace) = &tokens[i];
        i += 1;
        if brace == "{" {
            depth += 1;
        } else if brace == "}" {
            if depth == 0 {
                break;
            }
            depth -= 1;
        }
        out.push((s.clone(), brace.clone(), depth));
    }
    Some(out)
}

/// The entries directly in `block` (vdf_block's list), in order: (key, Some(value)) for a string value, (key, None) for a nested
/// block. Keys and values are told apart by position, so a value that reads like a key is still a value.
fn vdf_pairs(block: &Option<Vec<Entry>>) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let Some(block) = block else { return out };
    let mut k = 0;
    while k < block.len() {
        let (s, brace, depth) = &block[k];
        if !brace.is_empty() || *depth != 0 || k + 1 >= block.len() {
            k += 1;
            continue;
        }
        let (nxt, nbrace, _) = &block[k + 1];
        if nbrace == "{" {
            out.push((s.clone(), None));
            k += 2;
            while k < block.len() && !(block[k].1 == "}" && block[k].2 == 0) {
                k += 1;
            }
            k += 1;
        } else {
            out.push((s.clone(), Some(nxt.clone())));
            k += 2;
        }
    }
    out
}

/// The string value of `key` directly in `block`, else None.
fn vdf_value(block: &Option<Vec<Entry>>, key: &str) -> Option<String> {
    vdf_pairs(block).into_iter().find(|(k, v)| k.to_lowercase() == key.to_lowercase() && v.is_some()).and_then(|(_, v)| v)
}

/// `os.path.realpath`: symlinks resolved as far as the path exists, the rest kept as written.
fn realpath(p: &str) -> String {
    let abs = if p.starts_with('/') {
        p.to_string()
    } else {
        format!("{}/{p}", std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default())
    };
    let mut todo: Vec<String> = abs.split('/').rev().map(str::to_string).collect();
    let mut resolved: Vec<String> = Vec::new();
    let mut hops = 0;
    while let Some(name) = todo.pop() {
        if name.is_empty() || name == "." {
            continue;
        }
        if name == ".." {
            resolved.pop();
            continue;
        }
        let candidate = format!("/{}", resolved.iter().chain(std::iter::once(&name)).cloned().collect::<Vec<_>>().join("/"));
        match fs::read_link(&candidate) {
            Ok(target) if hops < 40 => {
                hops += 1;
                let target = target.to_string_lossy().into_owned();
                if target.starts_with('/') {
                    resolved.clear();
                }
                todo.extend(target.split('/').rev().map(str::to_string));
            }
            _ => resolved.push(name),
        }
    }
    format!("/{}", resolved.join("/"))
}

/// `root` and every library in its steamapps/libraryfolders.vdf, in order, without repeats.
fn steam_libraries(root: &str) -> Vec<String> {
    let mut out = vec![root.to_string()];
    let text = read_text_lossy(&join(root, "steamapps/libraryfolders.vdf")).unwrap_or_default();
    for (key, value) in vdf_pairs(&vdf_block(&text, &["libraryfolders"])) {
        if value.is_none() {
            if let Some(path) = vdf_value(&vdf_block(&text, &["libraryfolders", &key]), "path").filter(|p| !p.is_empty()) {
                out.push(path);
            }
        }
    }
    let mut seen = HashSet::new();
    out.into_iter().filter(|lib| seen.insert(realpath(lib))).collect()
}

fn is_file(p: &str) -> bool {
    fs::metadata(p).is_ok_and(|m| m.is_file())
}

/// The folder of the Proton Steam runs `appid` with (config.vdf's CompatToolMapping: the game's entry, else the default for
/// every game, "0"), or an error line. Valve's Protons are looked for in `libraries` and every library of `root`.
fn mapped_proton(root: &str, libraries: &[String], appid: &str) -> Result<String, String> {
    let config = read_text_lossy(&join(root, "config/config.vdf")).map_err(|e| format!("no Steam config: {e}"))?;
    let mapping = ["InstallConfigStore", "Software", "Valve", "Steam", "CompatToolMapping"];
    let lookup = |app: &str| {
        let mut path = mapping.to_vec();
        path.push(app);
        vdf_value(&vdf_block(&config, &path), "name")
    };
    let name = lookup(appid).filter(|n| !n.is_empty()).or_else(|| lookup("0")).unwrap_or_default();
    if name.is_empty() {
        return Err(format!("Steam has no Proton chosen for {appid} (choose one in its Properties > Compatibility)"));
    }
    if let Some((_, tool_app)) = VALVE_TOOLS.iter().find(|(n, _)| *n == name) {
        let extra: Vec<String> = steam_libraries(root).into_iter().filter(|l| !libraries.contains(l)).collect();
        for lib in libraries.iter().chain(extra.iter()) {
            let Ok(acf) = read_text_lossy(&join(lib, &format!("steamapps/appmanifest_{tool_app}.acf"))) else { continue };
            let state = vdf_block(&acf, &["AppState"]).unwrap_or_default();
            for (k, (s, brace, depth)) in state.iter().enumerate() {
                if brace.is_empty() && *depth == 0 && s == "installdir" && k + 1 < state.len() {
                    let d = join(&join(lib, "steamapps/common"), &state[k + 1].0);
                    if is_file(&join(&d, "proton")) {
                        return Ok(d);
                    }
                }
            }
        }
        return Err(format!("Steam runs {appid} with {name}, which is not installed"));
    }
    let tools = join(root, "compatibilitytools.d");
    for d in glob_dir(&tools, |_| true) {
        let base = Path::new(&d).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if base.starts_with('.') || !is_file(&join(&d, "proton")) {
            continue;
        }
        let names: Vec<String> = match read_text_lossy(&join(&d, "compatibilitytool.vdf")) {
            Ok(t) => vdf_block(&t, &["compatibilitytools", "compat_tools"])
                .unwrap_or_default()
                .into_iter()
                .filter(|(_, b, depth)| b.is_empty() && *depth == 0)
                .map(|(s, _, _)| s)
                .collect(),
            Err(_) => vec![base.clone()],
        };
        if names.contains(&name) || (names.is_empty() && base == name) {
            return Ok(d);
        }
    }
    Err(format!("Steam runs {appid} with {name}, which is not in {tools}"))
}

fn proton(args: &[String]) -> Result<ExitCode, Fail> {
    if args.len() < 2 {
        return Err(format!(
            "TypeError: proton() missing {} required positional argument{}",
            2 - args.len(),
            if args.is_empty() { "s: 'root' and 'appid'" } else { ": 'appid'" }
        ));
    }
    let libraries: Vec<String> = if args.len() > 2 { args[2..].to_vec() } else { vec![args[0].clone()] };
    match mapped_proton(&args[0], &libraries, &args[1]) {
        Ok(d) => {
            println!("{d}");
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => {
            eprintln!("{e}");
            Ok(ExitCode::FAILURE)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, Fail> {
    let arg = |i: usize| args.get(i).cloned().ok_or_else(|| "IndexError: list index out of range".to_string());
    match arg(0)?.as_str() {
        "dumps" => {
            let folder = arg(1)?;
            let since = arg(2)?;
            let since = py_float(&since).ok_or_else(|| format!("ValueError: could not convert string to float: {}", py_repr_str(&since)))?;
            dumps(&folder, since)?;
        }
        "fg" => fg(&arg(1)?, &arg(2)?)?,
        "proton" => return proton(&args[1..]),
        "self-test" => self_test()?,
        _ => {}
    }
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    run(args).unwrap_or_else(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })
}

fn check(ok: bool, what: &str) -> Result<(), Fail> {
    if ok {
        Ok(())
    } else {
        Err(format!("AssertionError: {what}"))
    }
}

/// The same checks as the Python's self-test, shared by `gta-bench self-test` and the unit tests.
fn self_test() -> Result<(), Fail> {
    fn put32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn put64(b: &mut [u8], at: usize, v: u64) {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    // header, a stream directory of 2 (exception, memory list), the CONTEXT, one stack range
    let minidump = |addr: u64, top_of_stack: u64| {
        let mut b = vec![0u8; 0x600];
        b[..4].copy_from_slice(b"MDMP");
        for (at, v) in [(4, 0xA793), (8, 2), (12, 32), (32, 6), (36, 168), (40, 0x40), (44, 5), (48, 20), (52, 0x100)] {
            put32(&mut b, at, v);
        }
        put32(&mut b, 0x40 + 8, 0xC000_0005);
        put64(&mut b, 0x40 + 24, addr);
        put32(&mut b, 0x40 + 8 + 152, 1232);
        put32(&mut b, 0x40 + 8 + 156, 0x200);
        put64(&mut b, 0x200 + 0x98, 0x4710_FEA8);
        put32(&mut b, 0x100, 1);
        put64(&mut b, 0x104, 0x4710_FEA8);
        put32(&mut b, 0x10C, 16);
        put32(&mut b, 0x110, 0x500);
        put64(&mut b, 0x500, top_of_stack);
        b
    };
    check(dump_cause(&minidump(WATCHDOG, 0))?.starts_with("GTA's hang watchdog (main loop stalled 60 s, GTA5_Enhanced.exe+0x12c6eb"), "watchdog site")?;
    check(dump_cause(&minidump(0x1_4095_1C50, WATCHDOG))?.contains("stage handler GTA5_Enhanced.exe+0x951c50"), "stage handler")?;
    check(dump_cause(&minidump(0x1_4012_3456, 0x1_4000_0000))? == "exception 0xc0000005 at GTA5_Enhanced.exe+0x123456", "other fault")?;
    check(dump_cause(b"")?.starts_with("empty dump"), "empty dump")?;
    let frames = |start: u32, end: u32, fps: u32| -> Vec<f64> { (0..(end - start) * fps).map(|i| start as f64 + i as f64 / fps as f64).collect() };
    let cat = |parts: &[Vec<f64>]| parts.concat();
    check(fg_rule(&cat(&[frames(0, 70, 60), frames(70, 260, 300)]), 576.0) == "kept", "fg steady")?;
    check(fg_rule(&cat(&[frames(0, 70, 60), frames(70, 75, 2000), frames(75, 260, 300)]), 576.0) == "kept", "fg loading burst")?;
    let lost = fg_rule(&cat(&[frames(0, 70, 60), frames(70, 130, 300), frames(130, 260, 80)]), 576.0);
    check(lost.starts_with("stopped 13") && lost.ends_with(" 80 300"), &lost)?;
    // a dip of a few seconds (a scene change) is not a stop
    check(fg_rule(&cat(&[frames(0, 70, 60), frames(70, 110, 300), frames(110, 118, 80), frames(118, 260, 300)]), 576.0) == "kept", "fg short dip")?;

    let root = std::env::temp_dir().join(format!(
        "gta-bench-self-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
    ));
    let result = self_test_steam(&root);
    let _ = fs::remove_dir_all(&root);
    result?;
    println!("gta-bench self-test ok");
    Ok(())
}

fn self_test_steam(root: &Path) -> Result<(), Fail> {
    let r = root.to_string_lossy().into_owned();
    let put = |rel: &str, text: &str| -> Result<(), Fail> {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap_or(root)).map_err(|e| e.to_string())?;
        fs::write(&p, text).map_err(|e| e.to_string())
    };
    let config = |name: &str| {
        put(
            "config/config.vdf",
            &format!(
                "\"InstallConfigStore\"\n{{\n\t\"Software\"\n\t{{\n\t\t\"Valve\"\n\t\t{{\n\t\t\t\"Steam\"\n\t\t\t{{\n\t\t\t\t\"CompatToolMapping\"\n\t\t\t\t{{\n\t\t\t\t\t\"1091500\"\n\t\t\t\t\t{{\n\t\t\t\t\t\t\"name\"\t\t\"other\"\n\t\t\t\t\t}}\n\t\t\t\t\t\"3240220\"\n\t\t\t\t\t{{\n\t\t\t\t\t\t\"name\"\t\t\"{name}\"\n\t\t\t\t\t\t\"config\"\t\t\"\"\n\t\t\t\t\t}}\n\t\t\t\t}}\n\t\t\t}}\n\t\t}}\n\t}}\n}}\n"
            ),
        )
    };
    for (d, internal) in [("Proton-CachyOS Latest", "proton-cachyos"), ("cachyos-11.0-NeuralForge", "cachyos-11.0-NeuralForge")] {
        put(&format!("compatibilitytools.d/{d}/proton"), "")?;
        put(
            &format!("compatibilitytools.d/{d}/compatibilitytool.vdf"),
            &format!("\"compatibilitytools\"\n{{\n  \"compat_tools\"\n  {{\n    \"{internal}\" // Internal name\n    {{\n      \"install_path\" \".\"\n    }}\n  }}\n}}\n"),
        )?;
    }
    put("steamapps/appmanifest_1493710.acf", "\"AppState\"\n{\n\t\"appid\"\t\t\"1493710\"\n\t\"installdir\"\t\t\"Proton - Experimental\"\n}\n")?;
    put("steamapps/common/Proton - Experimental/proton", "")?;
    let roots = vec![r.clone()];
    let at = |rel: &str| format!("{r}/{rel}");
    config("cachyos-11.0-NeuralForge")?;
    check(mapped_proton(&r, &roots, "3240220") == Ok(at("compatibilitytools.d/cachyos-11.0-NeuralForge")), "NF copy")?;
    config("proton-cachyos")?;
    check(mapped_proton(&r, &roots, "3240220") == Ok(at("compatibilitytools.d/Proton-CachyOS Latest")), "internal name")?;
    config("proton_experimental")?;
    check(mapped_proton(&r, &roots, "3240220") == Ok(at("steamapps/common/Proton - Experimental")), "valve tool")?;
    config("gone")?;
    check(mapped_proton(&r, &roots, "3240220").is_err_and(|e| e.ends_with(&format!("not in {}", at("compatibilitytools.d")))), "missing tool")?;
    check(mapped_proton(&r, &roots, "4000").is_err_and(|e| e.contains("no Proton chosen")), "no choice")?;
    // Keys and values by position: a value that reads "name" is not the key, a nested block is not a value.
    let pairs = vdf_pairs(&vdf_block("\"a\"\n{\n \"x\" \"name\"\n \"name\" \"real\"\n \"b\" { \"name\" \"inner\" }\n}\n", &["a"]));
    check(pairs == [("x".into(), Some("name".into())), ("name".into(), Some("real".into())), ("b".into(), None)], "pairs by position")?;
    check(vdf_value(&vdf_block("\"a\" { \"b\" { \"name\" \"inner\" } }", &["a"]), "name").is_none(), "nested block is not a value")?;
    // A // inside a string is not a comment; a // comment is skipped.
    check(vdf_value(&vdf_block("\"a\" // note\n{\n \"path\" \"//srv/x\" // c\n}\n", &["a"]), "path").as_deref() == Some("//srv/x"), "comments")?;
    // No entry for the game: Steam's default for every game ("0").
    put("config/config.vdf", "\"InstallConfigStore\"\n{\n\"Software\"\n{\n\"Valve\"\n{\n\"Steam\"\n{\n\"CompatToolMapping\"\n{\n\"0\"\n{\n\"name\" \"proton_experimental\"\n}\n}\n}\n}\n}\n}\n")?;
    check(mapped_proton(&r, &roots, "4000") == Ok(at("steamapps/common/Proton - Experimental")), "default mapping")?;
    // Valve's Proton in another library, found through libraryfolders.vdf though not passed in.
    let lib2 = at("lib2");
    put("lib2/steamapps/common/Proton 9.0/proton", "")?;
    put("lib2/steamapps/appmanifest_2805730.acf", "\"AppState\"\n{\n\t\"installdir\"\t\t\"Proton 9.0\"\n}\n")?;
    put(
        "steamapps/libraryfolders.vdf",
        &format!("\"libraryfolders\"\n{{\n\t\"0\"\n\t{{\n\t\t\"path\"\t\t\"{r}\"\n\t}}\n\t\"1\"\n\t{{\n\t\t\"path\"\t\t\"{lib2}\"\n\t}}\n}}\n"),
    )?;
    config("proton_9")?;
    check(mapped_proton(&r, &roots, "3240220") == Ok(format!("{lib2}/steamapps/common/Proton 9.0")), "library from libraryfolders.vdf")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_test_passes() {
        super::self_test().unwrap();
    }

    #[test]
    fn truncated_dump_is_a_struct_error() {
        let mut b = vec![0u8; 40];
        b[..4].copy_from_slice(b"MDMP");
        b[8] = 1;
        b[12] = 32;
        assert!(super::dump_cause(&b).is_err_and(|e| e.starts_with("struct.error: unpack_from requires a buffer of at least 44 bytes")));
    }

    #[test]
    fn long_integers_count_in_the_csv() {
        assert_eq!(super::int_as_f64("123456789012345678901234567890"), Some(1.2345678901234568e29));
        assert_eq!(super::int_as_f64("1_0"), Some(10.0));
        assert_eq!(super::int_as_f64("_1"), None);
    }
}
