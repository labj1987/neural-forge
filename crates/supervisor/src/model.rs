//! The model directory the native backend loads: NVIDIA's packed E4M3 weights, taken from the
//! user's own `nvngx_dlssnr.dll` and written in the layout OpenDLSS-NR's `nr::Model` reads
//! (`manifest.json` plus eleven stage files under `model/`, its docs/weights.md).
//!
//! The weights are one `RT_RCDATA` resource named `WEIGHTS_HT`. Only the PE headers, the
//! resource directory and that resource's own record framing are read; no code in the DLL is.
//! The resource is a `u64` total length followed by records laid end to end:
//!
//! | field | bytes |
//! | --- | --- |
//! | name length `n` | u64 |
//! | name, `blockB.layerL.parameter` | `n` |
//! | record length `a`, counted from after this field | u64 |
//! | `a` again | u64 |
//! | tensor length | u64 |
//! | kind, always 1 | u32 |
//! | the tensor | tensor length |
//! | trailer, unread | the rest of `a` |
//!
//! The gate is the network, not the DLL's version: a build is extracted when its records hold
//! exactly the tensors in [`model_shape::EXPECTED`] (same names, same byte lengths), which is what
//! the graph implements. A build outside [`VERIFIED_BUILDS`] is marked unverified in the manifest.
//!
//! Nothing produced here is ever committed, uploaded or packaged.

use crate::model_shape;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

/// The builds whose output was compared bit for bit with NVIDIA's runtime. Not a gate: another build
/// with the same network is extracted too, and marked unverified.
pub const VERIFIED_BUILDS: &[[u16; 3]] = &[[310, 8, 0]];
pub const DLL_NAME: &str = "nvngx_dlssnr.dll";
const RESOURCE_NAME: &str = "WEIGHTS_HT";
/// How many names each list of a shape report shows.
const REPORT_NAMES: usize = 10;
const RT_RCDATA: u32 = 10;
const RT_VERSION: u32 = 16;

/// Stage files: the network's resolution stages, encoder to decoder (blocks inclusive).
const STAGES: [(&str, u32, u32); 11] = [
    ("encoder32", 0, 4),
    ("encoder64", 5, 8),
    ("encoder128", 9, 14),
    ("encoder256", 15, 22),
    ("encoder512", 23, 30),
    ("vit", 31, 38),
    ("decoder512", 39, 47),
    ("decoder256", 48, 55),
    ("decoder128", 56, 61),
    ("decoder64", 62, 65),
    ("decoder32", 66, 70),
];

#[derive(Debug)]
pub enum ModelError {
    Io(PathBuf, std::io::Error),
    Pe(String),
    Weights(String),
    /// The records do not hold the tensors the graph implements; the text is [`check_shape`]'s report.
    Shape(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(path, e) => write!(f, "{}: {e}", path.display()),
            ModelError::Pe(m) => write!(f, "not a readable DLL: {m}"),
            ModelError::Weights(m) => write!(f, "weights resource: {m}"),
            ModelError::Shape(report) => f.write_str(report),
        }
    }
}

impl std::error::Error for ModelError {}

fn pe_err(m: impl Into<String>) -> ModelError {
    ModelError::Pe(m.into())
}

fn w_err(m: impl Into<String>) -> ModelError {
    ModelError::Weights(m.into())
}

/// `paths::data_dir()/model`, the directory the layer loads.
pub fn model_dir() -> String {
    format!("{}/model", crate::paths::data_dir())
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// A PE32+ image's resource tree, enough to fetch one resource by type and name or id.
struct Resources<'a> {
    image: &'a [u8],
    sections: Vec<(u32, u32, u32, u32)>, // virtual address, virtual size, raw offset, raw size
    root: usize,                    // file offset of the resource directory
}

#[derive(Clone, Copy)]
enum Key<'k> {
    Id(u32),
    Name(&'k str),
}

impl<'a> Resources<'a> {
    fn parse(image: &'a [u8]) -> Result<Self, ModelError> {
        if image.get(0..2) != Some(b"MZ") {
            return Err(pe_err("no MZ header"));
        }
        let pe = u32_at(image, 0x3c).ok_or_else(|| pe_err("truncated DOS header"))? as usize;
        if image.get(pe..pe + 4) != Some(b"PE\0\0") {
            return Err(pe_err("no PE signature"));
        }
        let coff = pe + 4;
        let sections = u16_at(image, coff + 2).ok_or_else(|| pe_err("truncated COFF header"))? as usize;
        let optional_size = u16_at(image, coff + 16).ok_or_else(|| pe_err("truncated COFF header"))? as usize;
        let optional = coff + 20;
        if u16_at(image, optional) != Some(0x20b) {
            return Err(pe_err("not a 64-bit image"));
        }
        // Data directory 2 is the resource table; directories start 112 bytes into PE32+'s optional header.
        let rva = u32_at(image, optional + 112 + 2 * 8).ok_or_else(|| pe_err("no resource directory"))?;
        let table = optional + optional_size;
        let sections = (0..sections)
            .map(|i| {
                let s = table + i * 40;
                let raw_size = u32_at(image, s + 16)?;
                Some((u32_at(image, s + 12)?, u32_at(image, s + 8)?.max(raw_size), u32_at(image, s + 20)?, raw_size))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| pe_err("truncated section table"))?;
        let mut r = Resources { image, sections, root: 0 };
        if rva == 0 {
            return Err(pe_err("no resources"));
        }
        r.root = r.offset(rva).ok_or_else(|| pe_err("resource directory outside every section"))?;
        Ok(r)
    }

    /// The file offset of `rva`: `None` past the section's data in the file (`SizeOfRawData`) or
    /// when a malformed raw offset would overflow.
    fn offset(&self, rva: u32) -> Option<usize> {
        let (va, _, raw, raw_size) = self.sections.iter().find(|(va, size, ..)| rva >= *va && rva - va < *size)?;
        let delta = rva - va;
        if delta >= *raw_size {
            return None;
        }
        raw.checked_add(delta).map(|at| at as usize)
    }

    fn name_at(&self, at: usize) -> Option<String> {
        let len = u16_at(self.image, at)? as usize;
        let units: Vec<u16> = (0..len).map(|i| u16_at(self.image, at + 2 + 2 * i)).collect::<Option<_>>()?;
        Some(String::from_utf16_lossy(&units))
    }

    /// The entry of the directory at `dir` (relative to the root) matching `key`, or the first one for `None`.
    fn child(&self, dir: usize, key: Option<Key>) -> Option<u32> {
        let at = self.root + dir;
        let count = u16_at(self.image, at + 12)? as usize + u16_at(self.image, at + 14)? as usize;
        (0..count).find_map(|i| {
            let e = at + 16 + i * 8;
            let name = u32_at(self.image, e)?;
            let target = u32_at(self.image, e + 4)?;
            let hit = match key {
                None => true,
                Some(Key::Id(id)) => name & 0x8000_0000 == 0 && name == id,
                Some(Key::Name(n)) => {
                    name & 0x8000_0000 != 0
                        && self.name_at(self.root + (name & 0x7fff_ffff) as usize).as_deref() == Some(n)
                }
            };
            hit.then_some(target)
        })
    }

    /// The bytes of resource `kind`/`name`, first language.
    fn get(&self, kind: u32, name: Key) -> Option<&'a [u8]> {
        let mut entry = self.child(0, Some(Key::Id(kind)))?;
        for key in [Some(name), None] {
            if entry & 0x8000_0000 == 0 {
                return None;
            }
            entry = self.child((entry & 0x7fff_ffff) as usize, key)?;
        }
        if entry & 0x8000_0000 != 0 {
            return None;
        }
        let leaf = self.root + entry as usize;
        let start = self.offset(u32_at(self.image, leaf)?)?;
        let size = u32_at(self.image, leaf + 4)? as usize;
        self.image.get(start..start + size)
    }
}

/// The file version from `VS_FIXEDFILEINFO` (signature 0xFEEF04BD), as four parts.
fn file_version(resources: &Resources) -> Option<[u16; 4]> {
    let info = resources.get(RT_VERSION, Key::Id(1))?;
    let at = (0..info.len().saturating_sub(16)).step_by(4).find(|&i| u32_at(info, i) == Some(0xfeef_04bd))?;
    let ms = u32_at(info, at + 8)?;
    let ls = u32_at(info, at + 12)?;
    Some([(ms >> 16) as u16, ms as u16, (ls >> 16) as u16, ls as u16])
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<'a> {
    pub name: String,
    pub block: u32,
    pub layer: u32,
    pub parameter: String,
    pub bytes: &'a [u8],
}

fn parse_name(name: &str) -> Option<(u32, u32, String)> {
    let mut parts = name.splitn(3, '.');
    let block = parts.next()?.strip_prefix("block")?.parse().ok()?;
    let layer = parts.next()?.strip_prefix("layer")?.parse().ok()?;
    let parameter = parts.next()?;
    let valid = !parameter.is_empty() && parameter.bytes().all(|c| c.is_ascii_lowercase() || c == b'_');
    valid.then(|| (block, layer, parameter.to_string()))
}

/// Walks the `WEIGHTS_HT` records (module docs). Every field is checked; any surprise is refused.
/// Which tensors the records hold is [`check_shape`]'s question.
pub fn parse_weights(blob: &[u8]) -> Result<Vec<Tensor<'_>>, ModelError> {
    let total = u64_at(blob, 0).ok_or_else(|| w_err("shorter than its header"))?;
    if total != blob.len() as u64 {
        return Err(w_err(format!("header says {total} bytes, resource has {}", blob.len())));
    }
    let mut tensors = Vec::new();
    let mut at = 8usize;
    while at < blob.len() {
        let bad = |m: &str| w_err(format!("record {} at byte {at}: {m}", tensors.len()));
        let name_len = u64_at(blob, at).ok_or_else(|| bad("truncated"))? as usize;
        if name_len == 0 || name_len > 256 {
            return Err(bad("implausible name length"));
        }
        let name_bytes = blob.get(at + 8..at + 8 + name_len).ok_or_else(|| bad("truncated name"))?;
        let name = std::str::from_utf8(name_bytes).map_err(|_| bad("name is not text"))?.to_string();
        let (block, layer, parameter) = parse_name(&name).ok_or_else(|| bad(&format!("unexpected name {name:?}")))?;
        let fields = at + 8 + name_len;
        let (Some(record), Some(again), Some(length), Some(kind)) =
            (u64_at(blob, fields), u64_at(blob, fields + 8), u64_at(blob, fields + 16), u32_at(blob, fields + 24))
        else {
            return Err(bad("truncated fields"));
        };
        if record != again || kind != 1 {
            return Err(bad(&format!("unexpected framing (lengths {record}/{again}, kind {kind})")));
        }
        let start = fields + 28;
        let next = (fields + 8).checked_add(record as usize).ok_or_else(|| bad("length overflows"))?;
        let end = start.checked_add(length as usize).ok_or_else(|| bad("length overflows"))?;
        if end > next || next > blob.len() {
            return Err(bad(&format!("tensor {name} ({length} bytes) overruns its record")));
        }
        tensors.push(Tensor { name, block, layer, parameter, bytes: &blob[start..end] });
        at = next;
    }
    Ok(tensors)
}

/// `count`, then up to [`REPORT_NAMES`] of `lines`, one per line.
fn report_list(report: &mut String, title: &str, lines: &[String]) {
    report.push_str(&format!("\n{title}: {}", lines.len()));
    for line in lines.iter().take(REPORT_NAMES) {
        report.push_str(&format!("\n  {line}"));
    }
    if lines.len() > REPORT_NAMES {
        report.push_str(&format!("\n  and {} more", lines.len() - REPORT_NAMES));
    }
}

/// "71 blocks (0-70)": how many blocks the names cover, and the lowest and highest.
fn blocks_of<'n>(names: impl Iterator<Item = &'n str>) -> String {
    let blocks: BTreeSet<u32> = names.filter_map(|n| parse_name(n).map(|(block, _, _)| block)).collect();
    match (blocks.first(), blocks.last()) {
        (Some(first), Some(last)) => format!("{} blocks ({first}-{last})", blocks.len()),
        _ => "0 blocks".to_string(),
    }
}

/// Whether the records hold exactly the tensors in `expected` (`model_shape::EXPECTED` outside tests):
/// the same set of names, each the same length. Otherwise a report of what differs.
pub fn check_shape(tensors: &[Tensor], expected: &[(&str, u64)]) -> Result<(), ModelError> {
    let wanted: HashMap<&str, u64> = expected.iter().copied().collect();
    let mut found: HashMap<&str, u64> = HashMap::new();
    let mut unexpected = Vec::new();
    for t in tensors {
        if found.contains_key(t.name.as_str()) {
            unexpected.push(format!("{} (a second time)", t.name));
        } else {
            found.insert(&t.name, t.bytes.len() as u64);
            if !wanted.contains_key(t.name.as_str()) {
                unexpected.push(t.name.clone());
            }
        }
    }
    let missing: Vec<String> = expected.iter().filter(|(n, _)| !found.contains_key(n)).map(|(n, _)| n.to_string()).collect();
    let differing: Vec<String> = expected
        .iter()
        .filter_map(|(n, want)| found.get(n).filter(|got| *got != want).map(|got| format!("{n}: found {got}, expected {want}")))
        .collect();
    if missing.is_empty() && unexpected.is_empty() && differing.is_empty() {
        return Ok(());
    }
    let mut report = format!(
        "{DLL_NAME} holds {} tensors in {}, {} bytes; the graph expects {} tensors in {}, {} bytes.",
        tensors.len(),
        blocks_of(tensors.iter().map(|t| t.name.as_str())),
        tensors.iter().map(|t| t.bytes.len() as u64).sum::<u64>(),
        expected.len(),
        blocks_of(expected.iter().map(|(n, _)| *n)),
        expected.iter().map(|(_, len)| len).sum::<u64>(),
    );
    report_list(&mut report, "Expected but missing", &missing);
    report_list(&mut report, "Present but not expected", &unexpected);
    report_list(&mut report, "Different length", &differing);
    report.push_str("\nThis DLL carries a different network. Neural Forge needs new graph code for it.");
    Err(ModelError::Shape(report))
}

fn stage_of(block: u32) -> &'static str {
    STAGES.iter().find(|(_, first, last)| (*first..=*last).contains(&block)).map(|s| s.0).expect("blocks are 0..=70")
}

/// Upper case: `nr::Model` compares its own upper-case digest (`sha256.h`) as a string.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02X}")).collect()
}

/// Whether `build` is one of [`VERIFIED_BUILDS`].
pub fn is_verified(build: [u16; 4]) -> bool {
    VERIFIED_BUILDS.iter().any(|v| v[..] == build[..3])
}

/// Lays the tensors into the stage files and the manifest. Returns (relative path, bytes) pairs.
pub fn build_model(tensors: &[Tensor], build: [u16; 4], dll_sha256: &str) -> Vec<(String, Vec<u8>)> {
    let mut sorted: Vec<&Tensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| (a.block, a.layer, &a.parameter).cmp(&(b.block, b.layer, &b.parameter)));
    let mut files = Vec::new();
    let mut stages = Vec::new();
    let mut records = Vec::new();
    for (id, _, _) in STAGES {
        let mut packed = Vec::new();
        for t in sorted.iter().filter(|t| stage_of(t.block) == id) {
            // 16-byte alignment keeps every tensor's start aligned for the host's re-layout copies.
            packed.resize(packed.len().next_multiple_of(16), 0);
            records.push(serde_json::json!({
                "name": t.name, "block": t.block, "layer": t.layer, "parameter": t.parameter,
                "stage": id, "stageOffset": packed.len(), "byteLength": t.bytes.len(),
            }));
            packed.extend_from_slice(t.bytes);
        }
        let file = format!("{id}.e4m3");
        stages.push(serde_json::json!({
            "id": id, "file": file, "packedByteLength": packed.len(), "sha256": sha256_hex(&packed),
        }));
        files.push((format!("model/{file}"), packed));
    }
    let manifest = serde_json::json!({
        "source": {
            "file": DLL_NAME,
            "build": format!("{}.{}.{}.{}", build[0], build[1], build[2], build[3]),
            "sha256": dll_sha256,
            "resource": RESOURCE_NAME,
            "verified": is_verified(build),
        },
        "totals": {
            "blockCount": tensors.iter().map(|t| t.block).collect::<BTreeSet<_>>().len(),
            "tensorCount": tensors.len(),
            "byteLength": tensors.iter().map(|t| t.bytes.len()).sum::<usize>(),
        },
        "stages": stages,
        "tensors": records,
    });
    let text = serde_json::to_string_pretty(&manifest).expect("a JSON value always serializes") + "\n";
    files.push(("manifest.json".to_string(), text.into_bytes()));
    files
}

fn read_manifest(dir: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?).ok()
}

/// The build of the DLL the installed model directory came from (the manifest's `source.build`),
/// or `None` when there is no readable manifest there.
pub fn installed(dir: &Path) -> Option<String> {
    read_manifest(dir)?["source"]["build"].as_str().map(str::to_string)
}

/// Whether the installed model came from a build in [`VERIFIED_BUILDS`] (the manifest's
/// `source.verified`), or `None` when there is no readable manifest. A manifest without the field
/// is unverified.
pub fn installed_verified(dir: &Path) -> Option<bool> {
    Some(read_manifest(dir)?["source"]["verified"].as_bool().unwrap_or(false))
}

#[derive(Debug)]
pub struct Extracted {
    pub dir: PathBuf,
    pub build: String,
    /// The build is in [`VERIFIED_BUILDS`].
    pub verified: bool,
    pub tensors: usize,
    pub bytes: usize,
}

/// `source` is the DLL or a directory holding it. Replaces `out` as a whole ([`replace_tree`]), so a
/// model directory is never half old and half new.
/// Any build is taken whose records hold the tensors the graph implements ([`check_shape`]); nothing
/// is written otherwise.
pub fn extract(source: &Path, out: &Path) -> Result<Extracted, ModelError> {
    extract_shaped(source, out, model_shape::EXPECTED)
}

fn extract_shaped(source: &Path, out: &Path, expected: &[(&str, u64)]) -> Result<Extracted, ModelError> {
    let dll = if source.is_dir() { source.join(DLL_NAME) } else { source.to_path_buf() };
    let image = std::fs::read(&dll).map_err(|e| ModelError::Io(dll.clone(), e))?;
    let resources = Resources::parse(&image)?;
    let version = file_version(&resources).ok_or_else(|| pe_err("no version resource"))?;
    let blob = resources
        .get(RT_RCDATA, Key::Name(RESOURCE_NAME))
        .ok_or_else(|| w_err(format!("no RCDATA resource named {RESOURCE_NAME}")))?;
    let tensors = parse_weights(blob)?;
    check_shape(&tensors, expected)?;
    let files = build_model(&tensors, version, &sha256_hex(&image));
    replace_tree(out, &files)?;
    Ok(Extracted {
        dir: out.to_path_buf(),
        build: format!("{}.{}.{}", version[0], version[1], version[2]),
        verified: is_verified(version),
        tensors: tensors.len(),
        bytes: tensors.iter().map(|t| t.bytes.len()).sum(),
    })
}

/// Makes `out` hold exactly `files` (relative path, content): the whole tree is built in a sibling
/// temporary directory and synced, then swapped in by rename, so a reader sees the old directory or
/// the new one and never a mix. The temporary directory is removed on any error, leaving `out` as
/// it was.
fn replace_tree(out: &Path, files: &[(String, Vec<u8>)]) -> Result<(), ModelError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |e| ModelError::Io(path, e)
    };
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = out
        .file_name()
        .ok_or_else(|| ModelError::Io(out.to_path_buf(), std::io::Error::new(std::io::ErrorKind::InvalidInput, "no directory name")))?
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(parent).map_err(io(parent))?;
    let staging = parent.join(format!(".{name}.partial-{}", std::process::id()));
    let previous = parent.join(format!(".{name}.previous-{}", std::process::id()));
    // A run that died left its directories behind; they are only ever ours.
    for leftover in [&staging, &previous] {
        if leftover.exists() {
            std::fs::remove_dir_all(leftover).map_err(io(leftover))?;
        }
    }

    let build = || -> Result<(), ModelError> {
        let mut dirs = vec![staging.clone()];
        std::fs::create_dir(&staging).map_err(io(&staging))?;
        for (relative, bytes) in files {
            let path = staging.join(relative);
            if let Some(dir) = path.parent().filter(|d| !d.exists()) {
                std::fs::create_dir_all(dir).map_err(io(dir))?;
                dirs.push(dir.to_path_buf());
            }
            let mut file = std::fs::File::create(&path).map_err(io(&path))?;
            std::io::Write::write_all(&mut file, bytes).map_err(io(&path))?;
            file.sync_all().map_err(io(&path))?;
        }
        for dir in &dirs {
            std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(io(dir))?;
        }
        Ok(())
    };
    if let Err(e) = build() {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }

    let had_previous = out.exists();
    if had_previous {
        if let Err(e) = std::fs::rename(out, &previous) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(ModelError::Io(out.to_path_buf(), e));
        }
    }
    if let Err(e) = std::fs::rename(&staging, out) {
        if had_previous {
            let _ = std::fs::rename(&previous, out);
        }
        let _ = std::fs::remove_dir_all(&staging);
        return Err(ModelError::Io(out.to_path_buf(), e));
    }
    let _ = std::fs::File::open(parent).and_then(|d| d.sync_all());
    if had_previous {
        let _ = std::fs::remove_dir_all(&previous);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_section_offset_is_bounded_by_its_raw_data_and_never_overflows() {
        let r = Resources { image: &[], sections: vec![(0x1000, 0x2000, 0x400, 0x100), (0x4000, 0x1000, u32::MAX - 4, 0x1000)], root: 0 };
        assert_eq!(r.offset(0x1010), Some(0x410));
        assert_eq!(r.offset(0x1100), None, "past SizeOfRawData is not in the file");
        assert_eq!(r.offset(0x4010), None, "a raw offset that overflows is refused, not wrapped");
        assert_eq!(r.offset(0x9000), None);
    }

    #[test]
    fn replacing_the_model_tree_is_all_or_nothing() {
        let dir = scratch("replace-tree");
        let out = dir.join("model");
        std::fs::create_dir_all(out.join("model")).unwrap();
        std::fs::write(out.join("manifest.json"), "old").unwrap();
        std::fs::write(out.join("model/stale.e4m3"), "old").unwrap();

        // "a" as a file and as a directory: the build fails partway, after "a" is written.
        let bad = vec![("a".to_string(), b"x".to_vec()), ("a/b".to_string(), b"y".to_vec())];
        assert!(replace_tree(&out, &bad).is_err());
        assert_eq!(std::fs::read_to_string(out.join("manifest.json")).unwrap(), "old", "a failed build leaves the live tree alone");
        let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["model"], "the temporary directory is removed on error");

        let good = vec![("manifest.json".to_string(), b"new".to_vec()), ("model/vit.e4m3".to_string(), b"w".to_vec())];
        replace_tree(&out, &good).unwrap();
        assert_eq!(std::fs::read_to_string(out.join("manifest.json")).unwrap(), "new");
        assert!(!out.join("model/stale.e4m3").exists(), "the new tree replaces the old one as a whole");
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, ["model"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn record(name: &str, tensor: &[u8], trailer: usize) -> Vec<u8> {
        let mut r = (name.len() as u64).to_le_bytes().to_vec();
        r.extend_from_slice(name.as_bytes());
        let length = (8 + 8 + 4 + tensor.len() + trailer) as u64;
        r.extend_from_slice(&length.to_le_bytes());
        r.extend_from_slice(&length.to_le_bytes());
        r.extend_from_slice(&(tensor.len() as u64).to_le_bytes());
        r.extend_from_slice(&1u32.to_le_bytes());
        r.extend_from_slice(tensor);
        r.extend(std::iter::repeat_n(0u8, trailer));
        r
    }

    const TENSOR_COUNT: usize = 153;

    /// Names and lengths with every block present and exactly 153 records, like the real ones but
    /// small: tensor `i` is `20 + i` bytes.
    fn shape() -> Vec<(String, u64)> {
        let mut names: Vec<String> = (0..71).map(|b| format!("block{b}.layer0.layer")).collect();
        for b in 23..=47 {
            for l in 1..=4 {
                if names.len() < 152 {
                    names.push(format!("block{b}.layer{l}.layer"));
                }
            }
        }
        names.push("block70.layer0.blend_scale".to_string());
        assert_eq!(names.len(), TENSOR_COUNT);
        names.into_iter().enumerate().map(|(i, n)| (n, 20 + i as u64)).collect()
    }

    /// `shape` as the table `check_shape` takes.
    fn table(shape: &[(String, u64)]) -> Vec<(&str, u64)> {
        shape.iter().map(|(n, len)| (n.as_str(), *len)).collect()
    }

    /// A `WEIGHTS_HT` blob holding `shape`'s tensors, in its order.
    fn blob_of(shape: &[(String, u64)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (i, (n, len)) in shape.iter().enumerate() {
            body.extend(record(n, &vec![i as u8 + 1; *len as usize], 20));
        }
        let mut blob = ((body.len() + 8) as u64).to_le_bytes().to_vec();
        blob.extend(body);
        blob
    }

    fn weights() -> Vec<u8> {
        blob_of(&shape())
    }

    /// `check_shape`'s report for `found` against the fixture's shape.
    fn shape_report(found: &[(String, u64)]) -> String {
        let blob = blob_of(found);
        let tensors = parse_weights(&blob).unwrap();
        match check_shape(&tensors, &table(&shape())) {
            Err(ModelError::Shape(report)) => report,
            other => panic!("expected a shape error, got {other:?}"),
        }
    }

    /// A minimal PE32+ image: one section holding a resource tree with WEIGHTS_HT and a version.
    fn dll(weights: &[u8], version: [u16; 4]) -> Vec<u8> {
        const VA: u32 = 0x1000;
        let mut fixed = vec![0u8; 8];
        fixed.extend(0xfeef_04bdu32.to_le_bytes());
        fixed.extend(0x0001_0000u32.to_le_bytes());
        fixed.extend((((version[0] as u32) << 16) | version[1] as u32).to_le_bytes());
        fixed.extend((((version[2] as u32) << 16) | version[3] as u32).to_le_bytes());
        fixed.resize(64, 0);
        // Resource section: root (2 ids) | RCDATA dir (1 name) | VERSION dir (1 id) | 2 language dirs |
        // 2 data entries | the name string | data.
        let mut rsrc = vec![0u8; 0x200];
        let dir = |r: &mut Vec<u8>, at: usize, named: u16, ids: u16, entries: &[(u32, u32)]| {
            r[at + 12..at + 14].copy_from_slice(&named.to_le_bytes());
            r[at + 14..at + 16].copy_from_slice(&ids.to_le_bytes());
            for (i, (n, t)) in entries.iter().enumerate() {
                r[at + 16 + i * 8..at + 20 + i * 8].copy_from_slice(&n.to_le_bytes());
                r[at + 20 + i * 8..at + 24 + i * 8].copy_from_slice(&t.to_le_bytes());
            }
        };
        let sub = 0x8000_0000u32;
        dir(&mut rsrc, 0x000, 0, 2, &[(RT_RCDATA, sub | 0x40), (RT_VERSION, sub | 0x60)]);
        dir(&mut rsrc, 0x040, 1, 0, &[(sub | 0x180, sub | 0x80)]);
        dir(&mut rsrc, 0x060, 0, 1, &[(1, sub | 0xa0)]);
        dir(&mut rsrc, 0x080, 0, 1, &[(1033, 0x100)]);
        dir(&mut rsrc, 0x0a0, 0, 1, &[(1033, 0x110)]);
        let utf16: Vec<u8> = RESOURCE_NAME.encode_utf16().flat_map(u16::to_le_bytes).collect();
        rsrc[0x180..0x182].copy_from_slice(&(RESOURCE_NAME.len() as u16).to_le_bytes());
        rsrc[0x182..0x182 + utf16.len()].copy_from_slice(&utf16);
        let version_at = rsrc.len();
        rsrc.extend(&fixed);
        let weights_at = rsrc.len();
        rsrc.extend(weights);
        let entry = |r: &mut Vec<u8>, at: usize, data: usize, size: usize| {
            r[at..at + 4].copy_from_slice(&(VA + data as u32).to_le_bytes());
            r[at + 4..at + 8].copy_from_slice(&(size as u32).to_le_bytes());
        };
        entry(&mut rsrc, 0x100, weights_at, weights.len());
        entry(&mut rsrc, 0x110, version_at, fixed.len());

        let mut image = vec![0u8; 0x400];
        image[0..2].copy_from_slice(b"MZ");
        image[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        image[0x80..0x84].copy_from_slice(b"PE\0\0");
        image[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        image[0x94..0x96].copy_from_slice(&240u16.to_le_bytes());
        image[0x98..0x9a].copy_from_slice(&0x20bu16.to_le_bytes());
        image[0x98 + 112 + 16..0x98 + 112 + 20].copy_from_slice(&VA.to_le_bytes());
        let section = 0x98 + 240;
        image[section..section + 5].copy_from_slice(b".rsrc");
        image[section + 8..section + 12].copy_from_slice(&(rsrc.len() as u32).to_le_bytes());
        image[section + 12..section + 16].copy_from_slice(&VA.to_le_bytes());
        image[section + 16..section + 20].copy_from_slice(&(rsrc.len() as u32).to_le_bytes());
        image[section + 20..section + 24].copy_from_slice(&0x400u32.to_le_bytes());
        image.extend(rsrc);
        image
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("neural-forge-model-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn walks_records_and_skips_trailers() {
        let blob = weights();
        let tensors = parse_weights(&blob).unwrap();
        assert_eq!(tensors.len(), TENSOR_COUNT);
        assert_eq!(tensors[0].name, "block0.layer0.layer");
        assert_eq!(tensors[0].bytes, &[1u8; 20][..]);
        assert_eq!(tensors[5].bytes.len(), 25);
        assert_eq!(tensors.last().unwrap().parameter, "blend_scale");
    }

    #[test]
    fn refuses_a_wrong_total() {
        let mut blob = weights();
        blob.push(0);
        assert!(parse_weights(&blob).unwrap_err().to_string().contains("header says"));
    }

    #[test]
    fn accepts_the_expected_shape_in_any_order() {
        let mut found = shape();
        found.reverse();
        let blob = blob_of(&found);
        assert!(check_shape(&parse_weights(&blob).unwrap(), &table(&shape())).is_ok());
    }

    const CLOSING: &str = "This DLL carries a different network. Neural Forge needs new graph code for it.";

    #[test]
    fn reports_a_changed_length() {
        let mut found = shape();
        found[5].1 += 16;
        let report = shape_report(&found);
        assert!(report.contains("Different length: 1\n  block5.layer0.layer: found 41, expected 25"), "{report}");
        assert!(report.contains("Expected but missing: 0") && report.contains("Present but not expected: 0"), "{report}");
        assert!(report.ends_with(CLOSING), "{report}");
    }

    #[test]
    fn reports_a_missing_tensor() {
        let mut found = shape();
        found.remove(3);
        let report = shape_report(&found);
        assert!(report.starts_with("nvngx_dlssnr.dll holds 152 tensors in 70 blocks (0-70)"), "{report}");
        assert!(report.contains("Expected but missing: 1\n  block3.layer0.layer"), "{report}");
        assert!(report.ends_with(CLOSING), "{report}");
    }

    #[test]
    fn reports_an_extra_tensor() {
        let mut found = shape();
        found.push(("block70.layer1.layer".to_string(), 30));
        let report = shape_report(&found);
        assert!(report.contains("Present but not expected: 1\n  block70.layer1.layer"), "{report}");
        assert!(report.ends_with(CLOSING), "{report}");
    }

    #[test]
    fn reports_an_extra_block() {
        let mut found = shape();
        found.push(("block71.layer0.layer".to_string(), 30));
        let report = shape_report(&found);
        let summary = report.lines().next().unwrap();
        assert!(summary.contains("154 tensors in 72 blocks (0-71)") && summary.contains("153 tensors in 71 blocks (0-70)"), "{report}");
        assert!(report.contains("Present but not expected: 1\n  block71.layer0.layer"), "{report}");
    }

    #[test]
    fn reports_a_repeated_name_and_caps_each_list() {
        let mut found = shape();
        found.push(found[0].clone());
        for (_, len) in found.iter_mut().take(12) {
            *len += 1;
        }
        let report = shape_report(&found);
        assert!(report.contains("block0.layer0.layer (a second time)"), "{report}");
        assert!(report.contains("Different length: 12") && report.contains("  and 2 more"), "{report}");
    }

    #[test]
    fn the_real_table_is_the_310_8_0_network() {
        let rows = model_shape::EXPECTED;
        assert_eq!(rows.len(), TENSOR_COUNT);
        assert_eq!(rows.iter().map(|(n, _)| n).collect::<BTreeSet<_>>().len(), rows.len());
        assert!(rows.iter().all(|(n, _)| parse_name(n).is_some()));
        let blocks: BTreeSet<u32> = rows.iter().map(|(n, _)| parse_name(n).unwrap().0).collect();
        assert_eq!(blocks, (0..=70).collect());
        assert_eq!(rows.iter().map(|(_, len)| len).sum::<u64>(), 147_683_778);
        // The manifest the table came from lists block 70's parameters by name: blend_scale, then layer.
        assert_eq!(rows[rows.len() - 2], ("block70.layer0.blend_scale", 2));
        assert_eq!(rows.iter().filter(|(n, _)| n.ends_with(".blend_scale")).count(), 1);
    }

    #[test]
    fn refuses_a_tensor_longer_than_its_record() {
        let mut blob = weights();
        // block0's tensor length field sits after the length (8), name (19) and two record lengths (16).
        blob[8 + 8 + 19 + 16..8 + 8 + 19 + 24].copy_from_slice(&1_000_000u64.to_le_bytes());
        assert!(parse_weights(&blob).unwrap_err().to_string().contains("overruns"));
    }

    #[test]
    fn manifest_slices_reproduce_every_tensor() {
        let blob = weights();
        let tensors = parse_weights(&blob).unwrap();
        let files = build_model(&tensors, [310, 8, 0, 0], "00");
        let manifest: serde_json::Value = serde_json::from_slice(&files.last().unwrap().1).unwrap();
        assert_eq!(manifest["totals"]["blockCount"], 71);
        assert_eq!(manifest["stages"].as_array().unwrap().len(), 11);
        assert_eq!(manifest["tensors"].as_array().unwrap().len(), TENSOR_COUNT);
        for record in manifest["tensors"].as_array().unwrap() {
            let stage = record["stage"].as_str().unwrap();
            let (_, packed) = files.iter().find(|(p, _)| *p == format!("model/{stage}.e4m3")).unwrap();
            let at = record["stageOffset"].as_u64().unwrap() as usize;
            let len = record["byteLength"].as_u64().unwrap() as usize;
            let tensor = tensors.iter().find(|t| t.name == record["name"]).unwrap();
            assert_eq!(&packed[at..at + len], tensor.bytes);
            assert_eq!(at % 16, 0);
        }
        for stage in manifest["stages"].as_array().unwrap() {
            let (_, packed) = files.iter().find(|(p, _)| *p == format!("model/{}", stage["file"].as_str().unwrap())).unwrap();
            assert_eq!(stage["packedByteLength"], packed.len());
            assert_eq!(stage["sha256"], sha256_hex(packed));
        }
    }

    #[test]
    fn extracts_a_verified_build() {
        let dir = scratch("verified");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 8, 0, 3])).unwrap();
        let out = dir.join("model");
        let done = extract_shaped(&dir, &out, &table(&shape())).unwrap();
        assert_eq!((done.build.as_str(), done.verified, done.tensors), ("310.8.0", true, TENSOR_COUNT));
        assert!(out.join("manifest.json").is_file() && out.join("model/vit.e4m3").is_file());
        assert_eq!(installed(&out).as_deref(), Some("310.8.0.3"));
        assert_eq!(installed_verified(&out), Some(true));
        assert_eq!((installed(&dir.join("nowhere")), installed_verified(&dir.join("nowhere"))), (None, None));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "only the DLL and the model directory");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extracts_another_build_with_the_same_network_unverified() {
        let dir = scratch("unverified");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 9, 1, 0])).unwrap();
        let out = dir.join("model");
        let done = extract_shaped(&dir, &out, &table(&shape())).unwrap();
        assert_eq!((done.build.as_str(), done.verified), ("310.9.1", false));
        assert_eq!(installed(&out).as_deref(), Some("310.9.1.0"));
        assert_eq!(installed_verified(&out), Some(false));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_manifest_without_the_verified_field_is_unverified() {
        let dir = scratch("manifest-no-verified");
        std::fs::write(dir.join("manifest.json"), r#"{"source": {"build": "310.8.0.0"}}"#).unwrap();
        assert_eq!(installed_verified(&dir), Some(false));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn another_network_is_refused_and_nothing_is_written() {
        let dir = scratch("other-network");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 8, 0, 0])).unwrap();
        let out = dir.join("model");
        // The fixture's tiny tensors are not the real network.
        let err = extract(&dir, &out).unwrap_err();
        assert!(matches!(err, ModelError::Shape(_)), "{err}");
        assert!(err.to_string().contains("the graph expects 153 tensors in 71 blocks (0-70), 147683778 bytes"), "{err}");
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
