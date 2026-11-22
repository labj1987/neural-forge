//! The Wine builtin stamp of the NGX wrapper (cargo xtask build-ngx-wrapper): the 32 bytes at file offset 0x40 read
//! "Wine builtin DLL", what `winebuild --builtin` writes and what Wine's loader checks before it serves a DLL from its
//! builtin directories (docs/PROTON_NGX.md).
//!
//!   cargo xtask wine-builtin stamp FILE    write the stamp (fails on a file that is not PE or has no room for it)
//!   cargo xtask wine-builtin check FILE    exit 0 when FILE carries the stamp, 1 when not
//!   cargo xtask wine-builtin exports       read `objdump -p` of a DLL on stdin, print each NVSDK export and its ordinal
//!   cargo xtask wine-builtin --self-test   stamp and check minimal PE headers

use std::io::Read;
use std::process::ExitCode;

use regex::Regex;

const USAGE: &str = "usage: cargo xtask wine-builtin stamp FILE | check FILE | exports | --self-test";

const SIGNATURE: &[u8; 16] = b"Wine builtin DLL";

/// Stamps `bytes` in place.
pub fn stamp(path: &str, bytes: &mut Vec<u8>) -> Result<(), String> {
    if bytes.len() < 0x40 || &bytes[..2] != b"MZ" {
        return Err(format!("{path}: not a PE file"));
    }
    // e_lfanew: where the PE header starts. The stamp's 32 bytes must end before it.
    let lfanew = u32::from_le_bytes(bytes[0x3c..0x40].try_into().unwrap());
    if lfanew < 0x60 {
        return Err(format!("{path}: no room for the Wine builtin signature (e_lfanew {lfanew:#x})"));
    }
    if bytes.len() < 0x60 {
        bytes.resize(0x60, 0);
    }
    bytes[0x40..0x50].copy_from_slice(SIGNATURE);
    bytes[0x50..0x60].fill(0);
    Ok(())
}

pub fn stamped(bytes: &[u8]) -> bool {
    bytes.len() >= 0x60 && &bytes[0x40..0x50] == SIGNATURE && bytes[0x50..0x60].iter().all(|&b| b == 0)
}

/// `(name, ordinal)` of every NVSDK export in `objdump -p` output.
pub fn exports(objdump: &str) -> Vec<(String, String)> {
    Regex::new(r"\+base\[\s*(\d+)\]\s+[0-9a-f]{4}\s+(NVSDK\S+)").unwrap().captures_iter(objdump).map(|c| (c[2].to_string(), c[1].to_string())).collect()
}

/// A minimal PE header whose `e_lfanew` is `lfanew`.
fn minimal_pe(lfanew: u32) -> Vec<u8> {
    let mut b = vec![0u8; 0x100];
    b[..2].copy_from_slice(b"MZ");
    b[0x3c..0x40].copy_from_slice(&lfanew.to_le_bytes());
    b
}

fn self_test() -> Result<(), String> {
    let mut pe = minimal_pe(0x80);
    stamp("pe.dll", &mut pe)?;
    if !stamped(&pe) || pe.len() != 0x100 {
        return Err("stamp not found".into());
    }
    if stamp("small.dll", &mut minimal_pe(0x40)).is_ok() {
        return Err("a header without room was stamped".into());
    }
    if stamp("text.txt", &mut b"not a dll at all, just text, long enough to have a header".to_vec()).is_ok() {
        return Err("a file that is not PE was stamped".into());
    }
    if stamped(&minimal_pe(0x80)) {
        return Err("an unstamped file passed the check".into());
    }
    let dump = "\t[   0] +base[   1] 1a2b NVSDK_NGX_A\n\t[   1] +base[  12] 00ff NVSDK_NGX_B@8\n\t[   2] +base[   3] 0000 Other\n";
    let found = exports(dump);
    if found != [("NVSDK_NGX_A".to_string(), "1".to_string()), ("NVSDK_NGX_B@8".to_string(), "12".to_string())] {
        return Err(format!("exports read as {found:?}"));
    }
    Ok(())
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    match args {
        [flag] if flag == "--self-test" => {
            self_test().map_err(|e| format!("self-test: {e}"))?;
            println!("wine-builtin self-test ok");
            Ok(ExitCode::SUCCESS)
        }
        [cmd, path] if cmd == "stamp" => {
            let mut bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            stamp(path, &mut bytes)?;
            std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))?;
            Ok(ExitCode::SUCCESS)
        }
        [cmd, path] if cmd == "check" => {
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            Ok(if stamped(&bytes) { ExitCode::SUCCESS } else { ExitCode::FAILURE })
        }
        [cmd] if cmd == "exports" => {
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text).map_err(|e| format!("stdin: {e}"))?;
            for (name, ordinal) in exports(&text) {
                println!("{name} {ordinal}");
            }
            Ok(ExitCode::SUCCESS)
        }
        _ => {
            eprintln!("{USAGE}");
            Ok(ExitCode::from(2))
        }
    }
}

pub fn main(args: &[String]) -> ExitCode {
    run(args).unwrap_or_else(|e| {
        eprintln!("{e}");
        ExitCode::FAILURE
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn self_test() {
        super::self_test().unwrap();
    }
}
