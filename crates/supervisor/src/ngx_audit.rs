//! `neural-forge-cli ngx-audit`: checks a driver's NGX core (`nvidia/wine/_nvngx.dll`) against what the NGX wrapper
//! was built for ([`crate::ngx_core_exports`]): the exported names and their ordinals, the hooked exports, which
//! exports read their caller (found by [`scan`], a static scan of the core's code), and that NVIDIA's DLSS feature DLLs
//! beside it still refuse callers outside `nvngx.dll`. Any difference means the wrapper needs a look before games run
//! on this driver (docs/PROTON_NGX.md, "After a driver update").

pub mod scan;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use object::pe::{ImageNtHeaders64, IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG, IMAGE_SCN_MEM_EXECUTE};
use object::read::pe::{ExportTarget, ImageNtHeaders as _, ImageOptionalHeader as _, Import, PeFile64};
use object::LittleEndian as LE;

use crate::ngx_core_exports::{CHECKED_DRIVER, CORE_EXPORTS, HOOKED, READS_CALLER, UNFOLLOWED_TAIL_JUMPS};

/// The feature DLLs whose caller check the audit looks for, and the message of that check.
pub const FEATURE_DLLS: [&str; 2] = ["nvngx_dlssg.dll", "nvngx_dlssd.dll"];
pub const NOT_FROM_NGX: &str = "Not called from NGX runtime";

/// One export of the core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    /// Its name; `#ordinal` for one exported by ordinal only.
    pub name: String,
    pub ordinal: u32,
    /// What its code does with its return address (nothing, for a forwarded export).
    pub flow: scan::Flow,
    /// The DLL and function a forwarded export stands for (`DLL.Name`), whose code is not in the core.
    pub forward: Option<String>,
}

impl Export {
    pub fn reads_caller(&self) -> bool {
        self.flow.reads_caller()
    }
}

/// Reads a core's named exports and scans each one's code.
pub fn read_core(bytes: &[u8]) -> Result<Vec<Export>, String> {
    let pe = PeFile64::parse(bytes).map_err(|e| format!("not a 64-bit PE file: {e}"))?;
    let base = pe.nt_headers().optional_header().image_base();
    let sections = pe.section_table();
    let mut image = scan::Image::default();
    for s in sections.iter() {
        if s.characteristics.get(LE).contains(IMAGE_SCN_MEM_EXECUTE) {
            let (offset, size) = s.pe_file_range();
            let size = size.min(s.virtual_size.get(LE));
            let code = bytes.get(offset as usize..(offset as usize).saturating_add(size as usize)).ok_or("a code section past the file's end")?;
            image.add_code(base + u64::from(s.virtual_address.get(LE)), code);
        }
    }
    if let Some(imports) = pe.import_table().map_err(|e| format!("its import table: {e}"))? {
        let mut descriptors = imports.descriptors().map_err(|e| format!("its import table: {e}"))?;
        while let Some(d) = descriptors.next().map_err(|e| format!("its import table: {e}"))? {
            let iat = d.first_thunk.get(LE);
            let lookup = match d.original_first_thunk.get(LE) {
                0 => iat,
                rva => rva,
            };
            let mut thunks = imports.thunks(lookup).map_err(|e| format!("its import table: {e}"))?;
            let mut slot = u64::from(iat);
            while let Some(thunk) = thunks.next::<ImageNtHeaders64>().map_err(|e| format!("its import table: {e}"))? {
                if let Ok(Import::Name(_, name)) = imports.import::<ImageNtHeaders64>(thunk) {
                    image.add_import(base + slot, String::from_utf8_lossy(name));
                }
                slot += 8;
            }
        }
    }
    // GuardCFDispatchFunctionPointer (IMAGE_LOAD_CONFIG_DIRECTORY64, offset 0x78): Control Flow Guard's calls go
    // through it, to rax.
    let load_config = pe.data_directories().get(IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG).and_then(|d| d.data(bytes, &sections).ok());
    if let Some(dispatch) = load_config.and_then(|c| c.get(0x78..0x80)).map(|b| u64::from_le_bytes(b.try_into().unwrap())).filter(|&va| va != 0) {
        image.set_cfg_dispatch(dispatch);
    }
    let table = pe.export_table().map_err(|e| format!("its export table: {e}"))?.ok_or("it exports nothing")?;
    let mut scanner = scan::Scanner::new(&image);
    let mut out = Vec::new();
    for e in table.exports().map_err(|e| format!("its export table: {e}"))? {
        let ordinal = u32::from(e.ordinal.0);
        let name = e.name.map_or_else(|| format!("#{ordinal}"), |n| String::from_utf8_lossy(n).into_owned());
        let (flow, forward) = match e.target {
            ExportTarget::Address(rva) => (scanner.function(base + u64::from(rva)), None),
            ExportTarget::ForwardByName(dll, f) => (scan::Flow::default(), Some(format!("{}.{}", String::from_utf8_lossy(dll), String::from_utf8_lossy(f)))),
            ExportTarget::ForwardByOrdinal(dll, o) => (scan::Flow::default(), Some(format!("{}.#{o}", String::from_utf8_lossy(dll)))),
        };
        out.push(Export { name, ordinal, flow, forward });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// One line of the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub ok: bool,
    pub text: String,
}

fn ok(text: impl Into<String>) -> Line {
    Line { ok: true, text: text.into() }
}

fn fail(text: impl Into<String>) -> Line {
    Line { ok: false, text: text.into() }
}

/// The ordinal the wrapper gives `name` (its place in [`CORE_EXPORTS`], from 1).
fn expected_ordinal(name: &str) -> Option<u32> {
    CORE_EXPORTS.binary_search(&name).ok().map(|i| i as u32 + 1)
}

fn names<'a>(list: impl IntoIterator<Item = &'a str>) -> String {
    list.into_iter().collect::<Vec<_>>().join(", ")
}

/// The core's exports against the lists the wrapper was built with.
pub fn compare(core: &[Export]) -> Vec<Line> {
    let mut out = Vec::new();
    let has = |n: &str| core.iter().any(|e| e.name == n);

    let added: Vec<&Export> = core.iter().filter(|e| expected_ordinal(&e.name).is_none()).collect();
    let gone: Vec<&str> = CORE_EXPORTS.iter().copied().filter(|n| !has(n)).collect();
    let moved: Vec<String> = core
        .iter()
        .filter_map(|e| expected_ordinal(&e.name).filter(|&o| o != e.ordinal).map(|o| format!("{} {} (the wrapper's {o})", e.name, e.ordinal)))
        .collect();
    if !added.is_empty() {
        out.push(fail(format!(
            "exports: {} new in the core, which the wrapper does not export (a game that looks one up does not find it): {}",
            added.len(),
            names(added.iter().map(|e| e.name.as_str()))
        )));
    }
    let (hooked_gone, stub_gone): (Vec<&str>, Vec<&str>) = gone.iter().partition(|n| HOOKED.contains(n));
    let hooked_gone_any = !hooked_gone.is_empty();
    if hooked_gone_any {
        out.push(fail(format!(
            "hooked exports: {} gone from the core (the wrapper refuses to load, so DLSS does not run): {}",
            hooked_gone.len(),
            names(hooked_gone)
        )));
    }
    if !stub_gone.is_empty() {
        out.push(fail(format!(
            "exports: {} gone from the core (the wrapper hides them, so a game that looks one up does not find it): {}",
            stub_gone.len(),
            names(stub_gone)
        )));
    }
    if !moved.is_empty() {
        out.push(fail(format!(
            "ordinals: {} differ from the wrapper's (a game importing by ordinal reaches the wrong function): {}",
            moved.len(),
            moved.join(", ")
        )));
    }
    if added.is_empty() && gone.is_empty() && moved.is_empty() {
        out.push(ok(format!("exports: the {} names the wrapper exports, with the same ordinals", CORE_EXPORTS.len())));
    }
    if !hooked_gone_any {
        out.push(ok(format!("hooked exports: all {} there", HOOKED.len())));
    }
    let forwarded: Vec<String> = core.iter().filter_map(|e| e.forward.as_ref().map(|f| format!("{} to {f}", e.name))).collect();
    if !forwarded.is_empty() {
        out.push(fail(format!(
            "exports: {} forwarded to another DLL, whose code the scan does not read (whether they read their caller is unknown): {}",
            forwarded.len(),
            forwarded.join(", ")
        )));
    }

    let incomplete: Vec<&str> = core.iter().filter(|e| e.flow.incomplete).map(|e| e.name.as_str()).collect();
    let readers: Vec<&str> = core.iter().filter(|e| e.reads_caller()).map(|e| e.name.as_str()).collect();
    let newly: Vec<&str> = readers.iter().copied().filter(|n| !READS_CALLER.contains(n)).collect();
    let no_longer: Vec<&str> = READS_CALLER.iter().copied().filter(|n| has(n) && !readers.contains(n)).collect();
    let (newly_hooked, newly_passed): (Vec<&str>, Vec<&str>) = newly.iter().partition(|n| HOOKED.contains(n));
    if !newly_hooked.is_empty() {
        out.push(fail(format!(
            "reads its caller: {} hooked export(s) now read their caller: the wrapper calls them itself, so the core takes the wrapper's folder for the game's: {}",
            newly_hooked.len(),
            names(newly_hooked)
        )));
    }
    if !newly_passed.is_empty() {
        out.push(fail(format!(
            "reads its caller: {} more export(s) read their caller than READS_CALLER names (the call trace would hide the game's folder from them): {}",
            newly_passed.len(),
            names(newly_passed)
        )));
    }
    if !no_longer.is_empty() {
        out.push(fail(format!(
            "reads its caller: {} export(s) READS_CALLER names no longer read their caller: {}",
            no_longer.len(),
            names(no_longer.iter().copied())
        )));
    }
    if newly.is_empty() && no_longer.is_empty() {
        out.push(ok(format!("reads its caller: exactly the {} exports READS_CALLER names", READS_CALLER.len())));
    }
    // Tail jumps the scan cannot follow, whose target gets the export's return address: only the known ones.
    let jumps: Vec<&str> =
        core.iter().filter(|e| e.flow.return_address_sinks().contains(&(scan::INDIRECT_JUMP.to_string(), 0))).map(|e| e.name.as_str()).collect();
    let new_jumps: Vec<&str> = jumps.iter().copied().filter(|n| !UNFOLLOWED_TAIL_JUMPS.contains(n)).collect();
    let jumps_gone: Vec<&str> = UNFOLLOWED_TAIL_JUMPS.iter().copied().filter(|n| has(n) && !jumps.contains(n)).collect();
    if !new_jumps.is_empty() {
        out.push(fail(format!(
            "tail jumps: {} export(s) now jump on through a pointer the scan cannot follow, handing their return address to code that may read it: {}",
            new_jumps.len(),
            names(new_jumps.iter().copied())
        )));
    }
    if !jumps_gone.is_empty() {
        out.push(fail(format!(
            "tail jumps: {} export(s) UNFOLLOWED_TAIL_JUMPS names no longer jump through a pointer: {}",
            jumps_gone.len(),
            names(jumps_gone.iter().copied())
        )));
    }
    if new_jumps.is_empty() && jumps_gone.is_empty() {
        out.push(ok(format!(
            "tail jumps the scan cannot follow: only the {} UNFOLLOWED_TAIL_JUMPS names (to the DLSS feature's own function)",
            UNFOLLOWED_TAIL_JUMPS.len()
        )));
    }
    // Anywhere else the return address goes that the scan cannot see the end of: another imported function (another
    // way of finding the caller), a call deeper than it follows, an indirect call.
    let other: Vec<String> = core
        .iter()
        .flat_map(|e| {
            e.flow
                .return_address_sinks()
                .into_iter()
                .filter(|(name, arg)| !(scan::is_module_from_address(name) && *arg == 1) && name != scan::INDIRECT_JUMP)
                .map(move |(name, arg)| match name.as_str() {
                    scan::DEPTH_LIMIT => format!("{} to a call more than {} calls deep (argument {})", e.name, scan::MAX_CALL_DEPTH, arg + 1),
                    scan::INDIRECT_CALL => format!("{} to a call through a pointer (argument {})", e.name, arg + 1),
                    _ => format!("{} to the imported {name} (argument {})", e.name, arg + 1),
                })
        })
        .collect();
    if !other.is_empty() {
        out.push(fail(format!("return address: {} export(s) hand it where the scan cannot see what is done with it: {}", other.len(), other.join(", "))));
    }
    if !incomplete.is_empty() {
        out.push(fail(format!("scan: could not follow every path of {}", names(incomplete))));
    }
    out
}

/// That each DLSS feature DLL in `dir` still has the message with which it refuses callers outside `nvngx.dll`: a
/// marker only (the check's code is not read), whose loss says the check changed.
pub fn feature_dlls(dir: &Path) -> Vec<Line> {
    FEATURE_DLLS
        .iter()
        .map(|dll| match std::fs::read(dir.join(dll)) {
            Err(e) => fail(format!("{dll}: cannot be read in {} ({e})", dir.display())),
            Ok(b) if b.windows(NOT_FROM_NGX.len()).any(|w| w == NOT_FROM_NGX.as_bytes()) => {
                ok(format!("{dll}: still has its message for callers outside nvngx.dll (\"{NOT_FROM_NGX}\")"))
            }
            Ok(_) => fail(format!(
                "{dll}: \"{NOT_FROM_NGX}\" is gone: how it checks its caller changed, and the core's name (lower-case nvngx.dll) may no longer matter or be checked differently"
            )),
        })
        .collect()
}

/// The audit's result.
#[derive(Clone, Debug)]
pub struct Report {
    pub core: PathBuf,
    pub lines: Vec<Line>,
    /// The core could be read and scanned (a failure otherwise says nothing about the driver).
    pub audited: bool,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.lines.iter().all(|l| l.ok)
    }

    pub fn text(&self) -> String {
        let mut s = format!("NGX core: {} (the wrapper's lists are from driver {CHECKED_DRIVER})\n", self.core.display());
        for l in &self.lines {
            let _ = writeln!(s, "{} {}", if l.ok { "OK  " } else { "FAIL" }, l.text);
        }
        let failed = self.lines.iter().filter(|l| !l.ok).count();
        if !self.audited {
            s.push_str("ngx-audit: could not audit this core\n");
        } else if failed == 0 {
            s.push_str("ngx-audit: the NGX wrapper fits this driver\n");
        } else {
            let _ = writeln!(
                s,
                "ngx-audit: {failed} difference(s): the NGX wrapper needs updating for this driver (crates/ngx-wrapper/src/core_exports.rs, docs/PROTON_NGX.md \"After a driver update\")"
            );
        }
        s
    }
}

/// Audits the core at `core`, and the feature DLLs in `driver_dir` (the core's folder when `None`).
pub fn audit(core: &Path, driver_dir: Option<&Path>) -> Report {
    let (mut lines, audited) = match std::fs::read(core).map_err(|e| e.to_string()).and_then(|b| read_core(&b)) {
        Ok(exports) => (compare(&exports), true),
        Err(e) => (vec![fail(format!("the core cannot be read: {e}"))], false),
    };
    if audited || driver_dir.is_some() {
        let dir = driver_dir.map(Path::to_path_buf).or_else(|| core.parent().map(Path::to_path_buf)).unwrap_or_default();
        lines.extend(feature_dlls(&dir));
    }
    Report { core: core.to_path_buf(), lines, audited }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export(name: &str, ordinal: u32, reads: bool) -> Export {
        let mut flow = scan::Flow::default();
        if reads {
            flow.sinks.insert(("GetModuleHandleExA".into(), 1), scan::Reached { ret: true, params: 0 });
        }
        if UNFOLLOWED_TAIL_JUMPS.contains(&name) {
            flow.sinks.insert((scan::INDIRECT_JUMP.into(), 0), scan::Reached { ret: true, params: 0 });
        }
        Export { name: name.into(), ordinal, flow, forward: None }
    }

    /// The core the lists describe.
    fn expected() -> Vec<Export> {
        CORE_EXPORTS.iter().enumerate().map(|(i, n)| export(n, i as u32 + 1, READS_CALLER.contains(n))).collect()
    }

    fn failures(lines: &[Line]) -> Vec<&str> {
        lines.iter().filter(|l| !l.ok).map(|l| l.text.as_str()).collect()
    }

    #[test]
    fn the_core_the_lists_describe_passes() {
        let lines = compare(&expected());
        assert!(failures(&lines).is_empty(), "{lines:?}");
        assert_eq!(lines.len(), 4, "{lines:?}");
    }

    #[test]
    fn a_new_gone_or_renumbered_export_fails() {
        let mut core = expected();
        core.retain(|e| e.name != "NVSDK_NGX_UpdateFeature" && e.name != "NVSDK_NGX_VULKAN_CreateFeature1");
        core.push(export("NVSDK_NGX_VULKAN_Init_Ext3", 76, false));
        core.iter_mut().find(|e| e.name == "NVSDK_NGX_CUDA_Init").unwrap().ordinal = 99;
        core.iter_mut().find(|e| e.name == "NVSDK_NGX_D3D11_Shutdown").unwrap().forward = Some("nvngx_core.NVSDK_NGX_D3D11_Shutdown".into());
        let lines = compare(&core);
        let f = failures(&lines);
        assert_eq!(f.len(), 5, "{lines:?}");
        assert!(f[0].contains("1 new") && f[0].contains("NVSDK_NGX_VULKAN_Init_Ext3"), "{f:?}");
        assert!(f[1].starts_with("hooked exports: 1 gone") && f[1].contains("VULKAN_CreateFeature1"), "{f:?}");
        assert!(f[2].contains("1 gone") && f[2].contains("NVSDK_NGX_UpdateFeature"), "{f:?}");
        assert!(f[3].contains("NVSDK_NGX_CUDA_Init 99 (the wrapper's 11)"), "{f:?}");
        assert!(f[4].contains("1 forwarded") && f[4].ends_with("NVSDK_NGX_D3D11_Shutdown to nvngx_core.NVSDK_NGX_D3D11_Shutdown"), "{f:?}");
    }

    #[test]
    fn a_change_in_who_reads_the_caller_fails() {
        let mut core = expected();
        for e in &mut core {
            match e.name.as_str() {
                "NVSDK_NGX_D3D12_CreateFeature" | "NVSDK_NGX_CUDA_Shutdown" => *e = export(&e.name, e.ordinal, true),
                "NVSDK_NGX_VULKAN_Init" => *e = export(&e.name, e.ordinal, false),
                "NVSDK_NGX_D3D11_Init" => {
                    e.flow.sinks.insert(("VirtualQuery".into(), 0), scan::Reached { ret: true, params: 0 });
                }
                "NVSDK_NGX_D3D11_Shutdown" => e.flow.incomplete = true,
                "NVSDK_NGX_D3D12_Shutdown" => {
                    e.flow.sinks.insert((scan::INDIRECT_JUMP.into(), 0), scan::Reached { ret: true, params: 0 });
                    e.flow.sinks.insert((scan::DEPTH_LIMIT.into(), 0), scan::Reached { ret: true, params: 0 });
                }
                "NVSDK_NGX_CUDA_GetScratchBufferSize" => e.flow.sinks.clear(),
                _ => {}
            }
        }
        let f: Vec<String> = failures(&compare(&core)).into_iter().map(String::from).collect();
        assert_eq!(f.len(), 7, "{f:?}");
        assert!(f[0].contains("hooked export(s) now read") && f[0].ends_with("NVSDK_NGX_D3D12_CreateFeature"), "{f:?}");
        assert!(f[1].contains("more export(s)") && f[1].ends_with("NVSDK_NGX_CUDA_Shutdown"), "{f:?}");
        assert!(f[2].contains("no longer") && f[2].ends_with("NVSDK_NGX_VULKAN_Init"), "{f:?}");
        assert!(f[3].contains("now jump on") && f[3].ends_with("NVSDK_NGX_D3D12_Shutdown"), "{f:?}");
        assert!(f[4].contains("no longer jump") && f[4].ends_with("NVSDK_NGX_CUDA_GetScratchBufferSize"), "{f:?}");
        assert!(f[5].contains("NVSDK_NGX_D3D11_Init to the imported VirtualQuery (argument 1)"), "{f:?}");
        assert!(f[5].contains("NVSDK_NGX_D3D12_Shutdown to a call more than 4 calls deep (argument 1)"), "{f:?}");
        assert!(f[6].ends_with("NVSDK_NGX_D3D11_Shutdown"), "{f:?}");
    }

    #[test]
    fn the_feature_dlls_must_keep_their_caller_check() {
        let dir = std::env::temp_dir().join(format!("nf-ngx-audit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("nvngx_dlssg.dll"), b"MZ...Error: Not called from NGX runtime - %S\0").unwrap();
        std::fs::write(dir.join("nvngx_dlssd.dll"), b"MZ...Error: Not called from the runtime\0").unwrap();
        let lines = feature_dlls(&dir);
        assert!(lines[0].ok, "{lines:?}");
        assert!(!lines[1].ok && lines[1].text.contains("is gone"), "{lines:?}");
        std::fs::remove_file(dir.join("nvngx_dlssd.dll")).unwrap();
        assert!(feature_dlls(&dir)[1].text.contains("cannot be read"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unreadable_core_fails_with_a_reason() {
        let r = audit(Path::new("/nonexistent/_nvngx.dll"), None);
        assert!(!r.passed());
        assert!(r.text().contains("FAIL the core cannot be read"), "{}", r.text());
        assert!(r.text().ends_with("ngx-audit: could not audit this core\n"), "{}", r.text());
        assert_eq!(r.lines.len(), 1, "no feature DLLs looked for beside a core that is not there");
    }

    pub(super) const FORWARD: u32 = u32::MAX;

    /// A minimal PE32+ file: one code section holding `code` at RVA 0x1000, exporting `exports` (name, offset of
    /// its code), importing `imports` from KERNEL32.dll. The IAT's slots follow the code in the same section.
    /// An export at offset [`FORWARD`] is forwarded to `KERNEL32.Sleep`.
    pub(super) fn pe(code: &[u8], exports: &[(&str, u32)], imports: &[&str]) -> Vec<u8> {
        const SEC: u32 = 0x1000;
        let mut sec = code.to_vec();
        let align = |v: &mut Vec<u8>| v.resize(v.len().next_multiple_of(8), 0);
        align(&mut sec);
        // Import names (hint + name), lookup table, IAT, descriptor, DLL name.
        let mut hint_rvas = Vec::new();
        for name in imports {
            hint_rvas.push(SEC + sec.len() as u32);
            sec.extend_from_slice(&0u16.to_le_bytes());
            sec.extend_from_slice(name.as_bytes());
            sec.push(0);
            align(&mut sec);
        }
        let thunks = |sec: &mut Vec<u8>| {
            let at = SEC + sec.len() as u32;
            for r in &hint_rvas {
                sec.extend_from_slice(&u64::from(*r).to_le_bytes());
            }
            sec.extend_from_slice(&0u64.to_le_bytes());
            at
        };
        let lookup = thunks(&mut sec);
        let iat = thunks(&mut sec);
        let dll = SEC + sec.len() as u32;
        sec.extend_from_slice(b"KERNEL32.dll\0");
        align(&mut sec);
        let import_dir = SEC + sec.len() as u32;
        for v in [lookup, 0, 0, dll, iat] {
            sec.extend_from_slice(&v.to_le_bytes());
        }
        sec.extend_from_slice(&[0; 20]);
        // Export directory, then its tables and names: the directory's size covers them all, as in a linked DLL.
        let mut sorted: Vec<(usize, &(&str, u32))> = exports.iter().enumerate().collect();
        sorted.sort_by_key(|(_, (n, _))| *n);
        let export_dir = SEC + sec.len() as u32;
        let n = exports.len() as u32;
        let functions = export_dir + 40;
        let names_at = functions + 4 * n;
        let ordinals = names_at + 4 * n;
        let mut strings = ordinals + 2 * n;
        let forwarder = strings + exports.iter().map(|(name, _)| name.len() as u32 + 1).sum::<u32>();
        for v in [0, 0, 0, 0, 1, n, n, functions, names_at, ordinals] {
            sec.extend_from_slice(&v.to_le_bytes());
        }
        for (_, off) in exports {
            let rva = if *off == FORWARD { forwarder } else { SEC + off };
            sec.extend_from_slice(&rva.to_le_bytes());
        }
        for (_, (name, _)) in &sorted {
            sec.extend_from_slice(&strings.to_le_bytes());
            strings += name.len() as u32 + 1;
        }
        for (i, _) in &sorted {
            sec.extend_from_slice(&(*i as u16).to_le_bytes());
        }
        for (_, (name, _)) in &sorted {
            sec.extend_from_slice(name.as_bytes());
            sec.push(0);
        }
        sec.extend_from_slice(b"KERNEL32.Sleep\0");
        let export_size = SEC + sec.len() as u32 - export_dir;
        sec.resize(sec.len().next_multiple_of(0x200), 0);

        let mut b = vec![0u8; 0x400];
        b[..2].copy_from_slice(b"MZ");
        b[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        let put16 = |b: &mut Vec<u8>, at: usize, v: u16| b[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let put32 = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        b[0x80..0x84].copy_from_slice(b"PE\0\0");
        put16(&mut b, 0x84, 0x8664); // machine
        put16(&mut b, 0x86, 1); // sections
        put16(&mut b, 0x94, 0xf0); // optional header size
        put16(&mut b, 0x96, 0x2022); // DLL, executable, large address aware
        let opt = 0x98;
        put16(&mut b, opt, 0x20b);
        put32(&mut b, opt + 16, SEC); // entry point
        b[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes());
        put32(&mut b, opt + 32, 0x1000); // section alignment
        put32(&mut b, opt + 36, 0x200); // file alignment
        put16(&mut b, opt + 48, 6); // subsystem version
        put32(&mut b, opt + 56, SEC + (sec.len() as u32).next_multiple_of(0x1000)); // image size
        put32(&mut b, opt + 60, 0x400); // headers size
        put16(&mut b, opt + 68, 3); // subsystem: console
        put32(&mut b, opt + 108, 16); // data directories
        put32(&mut b, opt + 112, export_dir);
        put32(&mut b, opt + 116, export_size);
        put32(&mut b, opt + 120, import_dir);
        put32(&mut b, opt + 124, 40);
        let sh = opt + 0xf0;
        b[sh..sh + 5].copy_from_slice(b".text");
        put32(&mut b, sh + 8, sec.len() as u32); // virtual size
        put32(&mut b, sh + 12, SEC);
        put32(&mut b, sh + 16, sec.len() as u32); // raw size
        put32(&mut b, sh + 20, 0x400); // raw offset
        put32(&mut b, sh + 36, 0x6000_0020); // code, execute, read
        b.extend_from_slice(&sec);
        b
    }

    #[test]
    fn a_synthetic_core_is_read_and_scanned() {
        // The IAT follows the code: with these three imports and 0x40 bytes of code it lands at a known RVA, which
        // the code below calls through. The layout is checked through the parse itself.
        let imports = ["GetModuleFileNameW", "GetModuleHandleExA", "Sleep"];
        let iat_of = |code_len: usize| {
            let pe0 = pe(&vec![0xc3; code_len], &[("f", 0)], &imports);
            let parsed = PeFile64::parse(&pe0[..]).unwrap();
            let table = parsed.import_table().unwrap().unwrap();
            let d = table.descriptors().unwrap().next().unwrap().unwrap();
            u64::from(d.first_thunk.get(LE))
        };
        let iat = iat_of(0x40);
        // reads: mov rdx, [rsp]; jmp [rip + IAT slot 1] (GetModuleHandleExA), as a tail call.
        // calls_self: lea rdx, [rip - 7] (its own address); jmp [rip + IAT slot 1].
        let mut code = Vec::new();
        code.extend_from_slice(&[0x48, 0x8b, 0x14, 0x24]); // mov rdx, [rsp]
        let jmp_at = |code: &Vec<u8>, slot: u64| {
            let next = 0x1000 + code.len() as u64 + 6;
            let target = iat + 8 * slot;
            let mut v = vec![0xff, 0x25];
            v.extend_from_slice(&((target as i64 - next as i64) as i32).to_le_bytes());
            v
        };
        let j = jmp_at(&code, 1);
        code.extend_from_slice(&j);
        let second = code.len() as u32;
        code.extend_from_slice(&[0x48, 0x8d, 0x15, 0xf9, 0xff, 0xff, 0xff]); // lea rdx, [rip - 7]
        let j = jmp_at(&code, 1);
        code.extend_from_slice(&j);
        code.resize(0x40, 0xcc);
        let file = pe(&code, &[("Reads", 0), ("Own", second), ("Fwd", FORWARD)], &imports);
        let core = read_core(&file).unwrap();
        assert_eq!(
            core.iter().map(|e| (e.name.as_str(), e.ordinal, e.reads_caller(), e.forward.as_deref())).collect::<Vec<_>>(),
            [("Fwd", 3, false, Some("KERNEL32.Sleep")), ("Own", 2, false, None), ("Reads", 1, true, None)]
        );
        assert!(read_core(b"MZ").is_err());
    }

    /// The scan against the core the lists were checked on ([`CHECKED_DRIVER`]'s), when this machine has exactly
    /// that file: the exports READS_CALLER names and no others, nothing else handed the return address, every export's
    /// paths followed. (Driver 620.73.04: the 20 Ghidra found, at every call depth from 1 to 8; at depth 0 the five
    /// `*GetFeatureRequirements` ones, which hand it to a helper, are missed.)
    #[test]
    fn the_checked_driver_s_core_reads_exactly_the_listed_callers() {
        let Some(bytes) = crate::proton_tool::DRIVER_LIB_DIRS
            .iter()
            .filter_map(|d| std::fs::read(Path::new(d).join("nvidia/wine/_nvngx.dll")).ok())
            .find(|b| crate::install::digest(b) == crate::ngx_core_exports::CHECKED_CORE_SHA256)
        else {
            eprintln!("skipped: driver {CHECKED_DRIVER}'s NGX core is not installed here");
            return;
        };
        let core = read_core(&bytes).unwrap();
        let lines = compare(&core);
        assert!(failures(&lines).is_empty(), "{lines:#?}");
        let readers: Vec<&str> = core.iter().filter(|e| e.reads_caller()).map(|e| e.name.as_str()).collect();
        assert_eq!(readers, READS_CALLER);
    }
}
