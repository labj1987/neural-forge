//! The model directory the native backend loads: NVIDIA's packed E4M3 weights, taken from the
//! user's own `nvngx_dlssnr.dll` and written in the layout OpenDLSS-NR's `nr::Model` reads
//! (`manifest.json` plus eleven stage files under `model/`, its docs/weights.md), one such directory
//! per network under the model directory: `model/full/` and, from a build that carries it,
//! `model/small/` ([`network_dir`]). The Network setting picks which one the layer runs.
//!
//! A network's weights are one `RT_RCDATA` resource: `WEIGHTS_HT` for the full network (71 blocks),
//! and since build 310.9.16 also `WEIGHTS_CC_SMALL_FP8` for a small one (31 blocks of the same
//! types, fewer per stage; [`Network`]). Only the PE headers, the resource directory and that
//! resource's own record framing are read; no code in the DLL is. Both resources are a `u64` total
//! length followed by records laid end to end:
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
//! exactly the tensors the graph implements for the chosen network ([`model_shape::EXPECTED`],
//! [`model_shape::EXPECTED_SMALL`]: same names, same byte lengths). Only a network of a build in
//! [`VERIFIED`] is marked verified in the manifest.
//!
//! Nothing produced here is ever committed, uploaded or packaged.

use crate::model_shape;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

/// The networks and builds whose output was compared bit for bit with NVIDIA's runtime (the 2.0.10
/// NGX helper; docs/NATIVE_BACKEND.md 0.6 and "The small network"). Not a gate: another build with
/// the same network is extracted too, and marked unverified.
pub const VERIFIED: &[(Network, [u16; 3])] = &[(Network::Full, [310, 8, 0]), (Network::Full, [310, 9, 16]), (Network::Small, [310, 9, 16])];
pub const DLL_NAME: &str = "nvngx_dlssnr.dll";
/// How many names each list of a shape report shows.
const REPORT_NAMES: usize = 10;
const RT_RCDATA: u32 = 10;
const RT_VERSION: u32 = 16;

/// Stage files: the full network's resolution stages, encoder to decoder (blocks inclusive).
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

/// The small network's stages: the same eleven, two blocks per resolution stage where the full one
/// has four to nine (the ViT keeps its eight). The graph's `nr::Topology` (`third_party/opendlss-nr/src/nr_graph.h`) mirrors these.
const STAGES_SMALL: [(&str, u32, u32); 11] = [
    ("encoder32", 0, 2),
    ("encoder64", 3, 4),
    ("encoder128", 5, 6),
    ("encoder256", 7, 8),
    ("encoder512", 9, 10),
    ("vit", 11, 18),
    ("decoder512", 19, 21),
    ("decoder256", 22, 23),
    ("decoder128", 24, 25),
    ("decoder64", 26, 27),
    ("decoder32", 28, 30),
];

/// Which of the DLL's networks the model directory holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Network {
    /// `WEIGHTS_HT`, 71 blocks: the network every build carries.
    Full,
    /// `WEIGHTS_CC_SMALL_FP8`, 31 blocks: about half the GPU time of the full one, a subtler effect.
    Small,
}

impl Network {
    pub const ALL: [Network; 2] = [Network::Full, Network::Small];

    /// The name the CLI takes and the manifest records.
    pub fn name(self) -> &'static str {
        match self {
            Network::Full => "full",
            Network::Small => "small",
        }
    }

    pub fn parse(name: &str) -> Option<Network> {
        Network::ALL.into_iter().find(|n| n.name() == name)
    }

    /// Its value of the Network setting (`neural_forge_protocol::enums::network`).
    pub fn setting(self) -> u32 {
        match self {
            Network::Full => neural_forge_protocol::enums::network::FULL,
            Network::Small => neural_forge_protocol::enums::network::SMALL,
        }
    }

    /// The network a Network setting value names (out of range: the full one, as the layer reads it).
    pub fn from_setting(value: u32) -> Network {
        if value == neural_forge_protocol::enums::network::SMALL { Network::Small } else { Network::Full }
    }

    /// The DLL resource holding its weights.
    pub fn resource(self) -> &'static str {
        match self {
            Network::Full => "WEIGHTS_HT",
            Network::Small => "WEIGHTS_CC_SMALL_FP8",
        }
    }

    fn stages(self) -> &'static [(&'static str, u32, u32); 11] {
        match self {
            Network::Full => &STAGES,
            Network::Small => &STAGES_SMALL,
        }
    }

    /// The tensors the graph implements for it.
    pub fn expected(self) -> &'static [(&'static str, u64)] {
        match self {
            Network::Full => model_shape::EXPECTED,
            Network::Small => model_shape::EXPECTED_SMALL,
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Network::Full => "full network",
            Network::Small => "small network",
        })
    }
}

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

/// `paths::data_dir()/model`, the model directory: one directory per extracted network in it
/// ([`network_dir`]), the one the Network setting names is what the layer loads.
pub fn model_dir() -> String {
    format!("{}/model", crate::paths::data_dir())
}

/// `root/full` or `root/small`: one network's directory under the model directory `root`, in the
/// layout `nr::Model` reads (`manifest.json` plus the stage files under `model/`).
pub fn network_dir(root: &Path, network: Network) -> PathBuf {
    root.join(network.name())
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

/// Walks a weights resource's records (module docs). Every field is checked; any surprise is refused.
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

/// Whether the records hold exactly the tensors in `expected` ([`Network::expected`] outside tests):
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

fn stage_of(network: Network, block: u32) -> Option<&'static str> {
    network.stages().iter().find(|(_, first, last)| (*first..=*last).contains(&block)).map(|s| s.0)
}

/// Upper case: `nr::Model` compares its own upper-case digest (`sha256.h`) as a string.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02X}")).collect()
}

/// Whether `network` of `build` was compared with NVIDIA's runtime: listed in [`VERIFIED`].
pub fn is_verified(network: Network, build: [u16; 4]) -> bool {
    VERIFIED.iter().any(|(n, v)| *n == network && v[..] == build[..3])
}

/// Lays the tensors into the stage files and the manifest. Returns (relative path, bytes) pairs; a
/// tensor of a block no stage holds is an error ([`check_shape`] refuses such a DLL first).
pub fn build_model(tensors: &[Tensor], network: Network, build: [u16; 4], dll_sha256: &str) -> Result<Vec<(String, Vec<u8>)>, ModelError> {
    if let Some(t) = tensors.iter().find(|t| stage_of(network, t.block).is_none()) {
        return Err(ModelError::Weights(format!("{}: block {} is in no stage of the graph", t.name, t.block)));
    }
    let mut sorted: Vec<&Tensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| (a.block, a.layer, &a.parameter).cmp(&(b.block, b.layer, &b.parameter)));
    let mut files = Vec::new();
    let mut stages = Vec::new();
    let mut records = Vec::new();
    for &(id, _, _) in network.stages() {
        let mut packed = Vec::new();
        for t in sorted.iter().filter(|t| stage_of(network, t.block) == Some(id)) {
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
            "resource": network.resource(),
            "verified": is_verified(network, build),
        },
        "network": network.name(),
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
    Ok(files)
}

fn read_manifest(dir: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).ok()?).ok()
}

/// The build of the DLL a network's directory ([`network_dir`]) came from (the manifest's
/// `source.build`), or `None` when there is no readable manifest there.
pub fn installed(dir: &Path) -> Option<String> {
    read_manifest(dir)?["source"]["build"].as_str().map(str::to_string)
}

/// Checks a network's directory ([`network_dir`]) against its manifest: the manifest names
/// `network`, and each stage's file is there with the manifest's length and SHA-256. `Ok(build)`
/// when all are, else what is wrong.
pub fn check_installed(dir: &Path, network: Network) -> Result<String, String> {
    let manifest = read_manifest(dir).ok_or_else(|| format!("no readable manifest.json in {}", dir.display()))?;
    let build = manifest["source"]["build"].as_str().ok_or("the manifest names no source build")?.to_string();
    let named = manifest_network(&manifest).map_err(|name| format!("the manifest names a network this version does not know: {name}"))?;
    if named != network {
        return Err(format!("the manifest is the {named}'s, in the {network}'s directory"));
    }
    let stages = manifest["stages"].as_array().filter(|s| !s.is_empty()).ok_or("the manifest lists no stages")?;
    for stage in stages {
        let (Some(file), Some(len), Some(sha)) = (stage["file"].as_str(), stage["packedByteLength"].as_u64(), stage["sha256"].as_str()) else {
            return Err("a stage in the manifest lacks its file, length or digest".into());
        };
        if file.contains('/') || file.starts_with('.') {
            return Err(format!("the manifest names an unexpected stage file {file:?}"));
        }
        let path = dir.join("model").join(file);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() as u64 != len {
            return Err(format!("{}: {} bytes, the manifest says {len}", path.display(), bytes.len()));
        }
        if !sha256_hex(&bytes).eq_ignore_ascii_case(sha) {
            return Err(format!("{}: its SHA-256 differs from the manifest's", path.display()));
        }
    }
    Ok(build)
}

/// The manifest's `network`, `Err(name)` for none or a name this version does not know.
fn manifest_network(manifest: &serde_json::Value) -> Result<Network, String> {
    match &manifest["network"] {
        serde_json::Value::String(name) => Network::parse(name).ok_or_else(|| format!("{name:?}")),
        other => Err(other.to_string()),
    }
}

/// Which network a directory's manifest says it holds, or `None` when there is no readable
/// manifest or it names none this version knows ([`check_installed`] says which).
pub fn installed_network(dir: &Path) -> Option<Network> {
    manifest_network(&read_manifest(dir)?).ok()
}

/// Whether the network in `dir` is a network and build in [`VERIFIED`] (the manifest's
/// `source.verified`), or `None` when there is no readable manifest. A manifest without the field
/// is unverified.
pub fn installed_verified(dir: &Path) -> Option<bool> {
    Some(read_manifest(dir)?["source"]["verified"].as_bool().unwrap_or(false))
}

/// One extracted network, as the status lines show it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installed {
    pub network: Network,
    pub build: String,
    pub verified: bool,
}

impl fmt::Display for Installed {
    /// "full network, build 310.9.16.0", with ", not verified against NVIDIA's runtime" for a build
    /// not in [`VERIFIED`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, build {}{}", self.network, self.build, if self.verified { "" } else { ", not verified against NVIDIA's runtime" })
    }
}

/// The networks extracted under the model directory `root`, in [`Network::ALL`]'s order: each whose
/// directory has a readable manifest naming it (whether its files are intact is
/// [`check_installed`]'s question).
pub fn installed_all(root: &Path) -> Vec<Installed> {
    Network::ALL
        .into_iter()
        .filter_map(|network| {
            let dir = network_dir(root, network);
            (installed_network(&dir) == Some(network)).then(|| Installed {
                network,
                build: installed(&dir).unwrap_or_default(),
                verified: installed_verified(&dir) == Some(true),
            })
        })
        .collect()
}

/// [`installed_all`] as one line: "full network, build 310.9.16.0; small network, build
/// 310.9.16.0", or `None` when no network is extracted.
pub fn installed_text(root: &Path) -> Option<String> {
    let all = installed_all(root);
    (!all.is_empty()).then(|| all.iter().map(Installed::to_string).collect::<Vec<_>>().join("; "))
}

#[derive(Debug)]
pub struct Extracted {
    pub dir: PathBuf,
    pub network: Network,
    pub build: String,
    /// [`is_verified`].
    pub verified: bool,
    pub tensors: usize,
    pub bytes: usize,
}

/// What [`extract_all`] wrote.
#[derive(Debug)]
pub struct ExtractedAll {
    /// Every network written, in [`Network::ALL`]'s order.
    pub networks: Vec<Extracted>,
    /// The networks the DLL carries that were not written, with the first line of why: a network
    /// whose records this version cannot read or whose shape the graph does not implement.
    pub refused: Vec<(Network, String)>,
}

/// `source` is the DLL or a directory holding it. Writes `network`'s directory under the model
/// directory `root` ([`network_dir`]) and leaves the other network's as it is. The directory is
/// replaced as a whole ([`replace_tree`]), so it is never half old and half new. Any build is taken
/// whose records hold the tensors the graph implements for that network ([`check_shape`]); nothing
/// is written otherwise.
pub fn extract(source: &Path, root: &Path, network: Network) -> Result<Extracted, ModelError> {
    extract_shaped(source, root, network, network.expected())
}

/// Writes every network the DLL carries under the model directory `root`: both of build 310.9.16
/// (the full network and the small one), only the full one of a build without the small one. The
/// model directory is replaced as a whole, so a network the DLL does not carry is gone afterwards.
/// A network the graph does not implement (a later build that changed it) is left out and
/// [`ExtractedAll::refused`] says why; only when no network can be written is that an error (the
/// full network's).
pub fn extract_all(source: &Path, root: &Path) -> Result<ExtractedAll, ModelError> {
    extract_all_with(source, root, &|network| network.expected())
}

fn extract_shaped(source: &Path, root: &Path, network: Network, expected: &[(&str, u64)]) -> Result<Extracted, ModelError> {
    let dll = Dll::read(source)?;
    let resources = Resources::parse(&dll.image)?;
    let version = file_version(&resources).ok_or_else(|| pe_err("no version resource"))?;
    let tensors = shaped(&resources, version, network, expected)?;
    let dir = network_dir(root, network);
    let files = build_model(&tensors, network, version, &dll.sha256())?;
    replace_tree(&dir, &files)?;
    Ok(extracted(dir, network, version, &tensors))
}

/// [`extract_all`] with `expected` giving each network's table.
fn extract_all_with<'t>(source: &Path, root: &Path, expected: &dyn Fn(Network) -> &'t [(&'t str, u64)]) -> Result<ExtractedAll, ModelError> {
    let dll = Dll::read(source)?;
    let resources = Resources::parse(&dll.image)?;
    let version = file_version(&resources).ok_or_else(|| pe_err("no version resource"))?;
    let sha256 = dll.sha256();
    let mut files = Vec::new();
    let mut networks = Vec::new();
    let mut refused = Vec::new();
    let mut first_error = None;
    for network in Network::ALL {
        // The small network is optional: a build without its resource has nothing to refuse.
        if network == Network::Small && resources.get(RT_RCDATA, Key::Name(network.resource())).is_none() {
            continue;
        }
        match shaped(&resources, version, network, expected(network)) {
            Ok(tensors) => {
                for (relative, bytes) in build_model(&tensors, network, version, &sha256)? {
                    files.push((format!("{}/{relative}", network.name()), bytes));
                }
                networks.push(extracted(network_dir(root, network), network, version, &tensors));
            }
            // Records this version cannot read, or a different network: the other may still be the graph's.
            Err(e @ (ModelError::Weights(_) | ModelError::Shape(_))) => {
                refused.push((network, e.to_string().lines().next().unwrap_or_default().to_string()));
                first_error.get_or_insert(e);
            }
            Err(e) => return Err(e),
        }
    }
    if networks.is_empty() {
        return Err(first_error.unwrap_or_else(|| w_err("the DLL carries no network")));
    }
    replace_tree(root, &files)?;
    Ok(ExtractedAll { networks, refused })
}

/// The DLL's bytes, from the DLL itself or a directory holding it.
struct Dll {
    image: Vec<u8>,
}

impl Dll {
    fn read(source: &Path) -> Result<Self, ModelError> {
        let dll = if source.is_dir() { source.join(DLL_NAME) } else { source.to_path_buf() };
        let image = std::fs::read(&dll).map_err(|e| ModelError::Io(dll.clone(), e))?;
        Ok(Dll { image })
    }

    fn sha256(&self) -> String {
        sha256_hex(&self.image)
    }
}

/// `network`'s records from the DLL, checked against `expected` ([`check_shape`]).
fn shaped<'a>(resources: &Resources<'a>, version: [u16; 4], network: Network, expected: &[(&str, u64)]) -> Result<Vec<Tensor<'a>>, ModelError> {
    let resource = network.resource();
    let blob = resources.get(RT_RCDATA, Key::Name(resource)).ok_or_else(|| {
        w_err(format!("build {}.{}.{}.{} has no {network} (no RCDATA resource named {resource})", version[0], version[1], version[2], version[3]))
    })?;
    let tensors = parse_weights(blob)?;
    check_shape(&tensors, expected)?;
    Ok(tensors)
}

fn extracted(dir: PathBuf, network: Network, version: [u16; 4], tensors: &[Tensor]) -> Extracted {
    Extracted {
        dir,
        network,
        build: format!("{}.{}.{}", version[0], version[1], version[2]),
        verified: is_verified(network, version),
        tensors: tensors.len(),
        bytes: tensors.iter().map(|t| t.bytes.len()).sum(),
    }
}

/// Makes `out` hold exactly `files` (relative path, content): the whole tree is built in a sibling
/// temporary directory and synced, then swapped in by rename, so a reader sees the old directory or
/// the new one and never a mix. The temporary directory is removed on any error, leaving `out` as
/// it was. One call at a time in a process: the temporary directories are named by the process, and
/// the GUI can start a second extraction while one is still writing. What a process that died left
/// behind is cleaned up first ([`recover_leftovers`]).
fn replace_tree(out: &Path, files: &[(String, Vec<u8>)]) -> Result<(), ModelError> {
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
    recover_leftovers(parent, &name, out)?;

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

/// Cleans up after [`replace_tree`] runs of processes that are gone (killed, crashed): their
/// `.<name>.partial-<pid>` staging directories are removed, and when `out` is missing because the
/// run died between moving it aside and moving the new tree in, its `.<name>.previous-<pid>` is
/// put back (the newest, if several), so the layer finds the model again; the rest are removed.
/// A live process's directories are left alone: it may still be writing them.
fn recover_leftovers(parent: &Path, name: &str, out: &Path) -> Result<(), ModelError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |e| ModelError::Io(path, e)
    };
    let (partial, previous) = (format!(".{name}.partial-"), format!(".{name}.previous-"));
    let alive = |pid: &str| pid.parse::<u32>().is_ok_and(|pid| pid == std::process::id() || Path::new("/proc").join(pid.to_string()).exists());
    let Ok(entries) = std::fs::read_dir(parent) else { return Ok(()) };
    let mut stale_partial = Vec::new();
    let mut stale_previous = Vec::new();
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else { continue };
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        if file_name.strip_prefix(&partial).is_some_and(|pid| !alive(pid)) {
            stale_partial.push(entry.path());
        } else if file_name.strip_prefix(&previous).is_some_and(|pid| !alive(pid)) {
            let modified = entry.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            stale_previous.push((modified, entry.path()));
        }
    }
    for dir in stale_partial {
        std::fs::remove_dir_all(&dir).map_err(io(&dir))?;
    }
    stale_previous.sort();
    if !out.exists() {
        if let Some((_, newest)) = stale_previous.pop() {
            std::fs::rename(&newest, out).map_err(io(out))?;
        }
    }
    for (_, dir) in stale_previous {
        std::fs::remove_dir_all(&dir).map_err(io(&dir))?;
    }
    Ok(())
}

/// Imports NVIDIA's `nvngx_dlssnr.dll` from `source_dir` into `dest_dir`: the file is read and
/// checked to be a 64-bit DLL with a version (what [`extract`] reads), then written whole by rename
/// (`paths::write_atomic`), so a failed or interrupted import leaves the DLL imported before as it
/// was. Returns where it went and the DLL's build.
pub fn import_dll(source_dir: &Path, dest_dir: &Path) -> Result<(PathBuf, String), ModelError> {
    let from = source_dir.join(DLL_NAME);
    let image = std::fs::read(&from).map_err(|e| ModelError::Io(from.clone(), e))?;
    let resources = Resources::parse(&image)?;
    let version = file_version(&resources).ok_or_else(|| pe_err("no version resource"))?;
    let dest = dest_dir.join(DLL_NAME);
    crate::paths::write_atomic(&dest, &image, 0o644).map_err(|e| ModelError::Io(dest.clone(), e))?;
    Ok((dest, format!("{}.{}.{}.{}", version[0], version[1], version[2], version[3])))
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
        dll_with("WEIGHTS_HT", weights, version)
    }

    /// The same with the weights under the RCDATA name `resource`.
    fn dll_with(resource: &str, weights: &[u8], version: [u16; 4]) -> Vec<u8> {
        dll_of(&[(resource, weights)], version)
    }

    /// The same with one or two RCDATA resources, `(name, weights)` each, as 310.9.16 carries both.
    fn dll_of(resources: &[(&str, &[u8])], version: [u16; 4]) -> Vec<u8> {
        assert!((1..=2).contains(&resources.len()));
        const VA: u32 = 0x1000;
        let mut fixed = vec![0u8; 8];
        fixed.extend(0xfeef_04bdu32.to_le_bytes());
        fixed.extend(0x0001_0000u32.to_le_bytes());
        fixed.extend((((version[0] as u32) << 16) | version[1] as u32).to_le_bytes());
        fixed.extend((((version[2] as u32) << 16) | version[3] as u32).to_le_bytes());
        fixed.resize(64, 0);
        // Resource section: root (2 ids) | RCDATA dir (1-2 names) | VERSION dir (1 id) | a language dir
        // each | a data entry each | the name strings | data.
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
        // Per resource i: name string at 0x180 + 0x40 i, language dir at 0x80 / 0xc0, data entry at 0x100 / 0x120.
        let names: Vec<(u32, u32)> = (0..resources.len() as u32).map(|i| (sub | (0x180 + 0x40 * i), sub | (0x80 + 0x40 * i))).collect();
        dir(&mut rsrc, 0x040, resources.len() as u16, 0, &names);
        dir(&mut rsrc, 0x060, 0, 1, &[(1, sub | 0xa0)]);
        dir(&mut rsrc, 0x0a0, 0, 1, &[(1033, 0x110)]);
        let entry = |r: &mut Vec<u8>, at: usize, data: usize, size: usize| {
            r[at..at + 4].copy_from_slice(&(VA + data as u32).to_le_bytes());
            r[at + 4..at + 8].copy_from_slice(&(size as u32).to_le_bytes());
        };
        let version_at = rsrc.len();
        rsrc.extend(&fixed);
        entry(&mut rsrc, 0x110, version_at, fixed.len());
        for (i, (resource, weights)) in resources.iter().enumerate() {
            dir(&mut rsrc, 0x80 + 0x40 * i, 0, 1, &[(1033, 0x100 + 0x20 * i as u32)]);
            let utf16: Vec<u8> = resource.encode_utf16().flat_map(u16::to_le_bytes).collect();
            let name_at = 0x180 + 0x40 * i;
            rsrc[name_at..name_at + 2].copy_from_slice(&(resource.len() as u16).to_le_bytes());
            rsrc[name_at + 2..name_at + 2 + utf16.len()].copy_from_slice(&utf16);
            let weights_at = rsrc.len();
            rsrc.extend(*weights);
            entry(&mut rsrc, 0x100 + 0x20 * i, weights_at, weights.len());
        }

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

    fn scratch(tag: &str) -> crate::paths::tests::ScratchDir {
        crate::paths::tests::ScratchDir::new(&format!("model-test-{tag}"))
    }

    /// A process id no process has: above the kernel's largest.
    const DEAD_PID: u32 = 4_194_305;

    #[test]
    fn a_killed_runs_leftovers_are_restored_or_removed() {
        let dir = scratch("leftovers");
        let out = dir.join("model");
        // Killed between moving the old model aside and moving the new one in: no `model`, the old
        // one in `.model.previous-<pid>`, and the staging tree beside it.
        let previous = dir.join(format!(".model.previous-{DEAD_PID}"));
        std::fs::create_dir_all(previous.join("full")).unwrap();
        std::fs::write(previous.join("full/weights.bin"), "old").unwrap();
        std::fs::create_dir_all(dir.join(format!(".model.partial-{DEAD_PID}"))).unwrap();
        // An older leftover of another dead run, and a live process's staging dir (this one's parent's).
        let older = dir.join(format!(".model.previous-{}", DEAD_PID + 1));
        std::fs::create_dir_all(&older).unwrap();
        let t = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::open(&older).unwrap().set_modified(t).unwrap();
        let live = dir.join(format!(".model.partial-{}", std::os::unix::process::parent_id()));
        std::fs::create_dir_all(&live).unwrap();
        recover_leftovers(&dir, "model", &out).unwrap();
        assert_eq!(std::fs::read_to_string(out.join("full/weights.bin")).unwrap(), "old", "the model is back");
        let mut left: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, [live.file_name().unwrap().to_string_lossy().into_owned(), "model".to_string()]);
        // With `model` in place, a dead run's previous copy is only removed.
        std::fs::create_dir_all(&previous).unwrap();
        replace_tree(&out, &[("full/weights.bin".to_string(), b"new".to_vec())]).unwrap();
        assert_eq!(std::fs::read_to_string(out.join("full/weights.bin")).unwrap(), "new");
        assert!(!previous.exists() && live.exists());
    }

    #[test]
    fn importing_the_dll_checks_it_and_never_truncates_the_one_there() {
        let dir = scratch("import");
        let (src, dest) = (dir.join("src"), dir.join("binaries"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dest).unwrap();
        let good = dll(&weights(), [310, 9, 16, 0]);
        std::fs::write(src.join(DLL_NAME), &good).unwrap();
        let (to, build) = import_dll(&src, &dest).unwrap();
        assert_eq!((to.clone(), build.as_str()), (dest.join(DLL_NAME), "310.9.16.0"));
        assert_eq!(std::fs::read(&to).unwrap(), good);
        // Not a DLL: refused, and the one imported before stays whole.
        std::fs::write(src.join(DLL_NAME), "not a dll").unwrap();
        assert!(matches!(import_dll(&src, &dest), Err(ModelError::Pe(_))));
        assert_eq!(std::fs::read(&to).unwrap(), good);
        // Replaced by rename: a reader holding the old file keeps its content.
        let inode = |p: &Path| std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(p).unwrap());
        let before = inode(&to);
        std::fs::write(src.join(DLL_NAME), dll(&weights(), [310, 9, 17, 0])).unwrap();
        assert_eq!(import_dll(&src, &dest).unwrap().1, "310.9.17.0");
        assert_ne!(inode(&to), before);
        assert!(matches!(import_dll(&dir.join("missing"), &dest), Err(ModelError::Io(..))));
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
    fn the_small_table_is_the_310_9_16_small_network() {
        let rows = model_shape::EXPECTED_SMALL;
        assert_eq!(rows.len(), 77);
        assert_eq!(rows.iter().map(|(n, _)| n).collect::<BTreeSet<_>>().len(), rows.len());
        let blocks: BTreeSet<u32> = rows.iter().map(|(n, _)| parse_name(n).unwrap().0).collect();
        assert_eq!(blocks, (0..=30).collect());
        assert_eq!(rows.iter().map(|(_, len)| len).sum::<u64>(), 113_887_490);
        assert_eq!(rows[rows.len() - 2], ("block30.layer0.blend_scale", 2));
        // Every small block has the shape of a full-network block: the graph runs it with the same kernels.
        let shape_of = |table: &[(&str, u64)], block: u32| -> Vec<(u32, String, u64)> {
            let mut v: Vec<_> = table
                .iter()
                .filter_map(|(n, len)| parse_name(n).filter(|p| p.0 == block).map(|(_, l, p)| (l, p, *len)))
                .collect();
            v.sort();
            v
        };
        let full: BTreeSet<_> = (0..=70).map(|b| shape_of(model_shape::EXPECTED, b)).collect();
        assert!((0..=30).all(|b| full.contains(&shape_of(rows, b))));
    }

    #[test]
    fn every_block_of_each_network_is_in_one_of_its_stages() {
        for network in Network::ALL {
            for (name, _) in network.expected() {
                assert!(stage_of(network, parse_name(name).unwrap().0).is_some(), "{network}: {name}");
            }
        }
    }

    /// The small network as the fixtures carry it: one 20-to-50-byte tensor per block, 31 blocks.
    fn small_shape() -> Vec<(String, u64)> {
        (0..31).map(|b| (format!("block{b}.layer0.layer"), 20 + b as u64)).collect()
    }

    #[test]
    fn extracts_the_small_network_of_another_build_unverified() {
        let dir = scratch("small");
        let small = small_shape();
        std::fs::write(dir.join(DLL_NAME), dll_with("WEIGHTS_CC_SMALL_FP8", &blob_of(&small), [310, 8, 0, 0])).unwrap();
        let root = dir.join("model");
        let done = extract_shaped(&dir, &root, Network::Small, &table(&small)).unwrap();
        let out = network_dir(&root, Network::Small);
        assert_eq!((done.network, done.verified, done.tensors, &done.dir), (Network::Small, false, 31, &out));
        assert_eq!(installed_network(&out), Some(Network::Small));
        assert_eq!(installed_verified(&out), Some(false));
        let manifest = read_manifest(&out).unwrap();
        assert_eq!(manifest["source"]["resource"], "WEIGHTS_CC_SMALL_FP8");
        assert_eq!(manifest["totals"]["blockCount"], 31);
        // Blocks 9-10 are the small network's encoder 512 stage.
        let stage = |block: u64| manifest["tensors"].as_array().unwrap().iter().find(|t| t["block"] == block).unwrap()["stage"].clone();
        assert_eq!((stage(10), stage(11), stage(30)), ("encoder512".into(), "vit".into(), "decoder32".into()));
        assert_eq!(check_installed(&out, Network::Small), Ok("310.8.0.0".to_string()));
        assert!(check_installed(&out, Network::Full).unwrap_err().contains("the manifest is the small network's, in the full network's directory"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extracts_the_small_network_of_310_9_16_verified() {
        let dir = scratch("small-verified");
        let small = small_shape();
        std::fs::write(dir.join(DLL_NAME), dll_with("WEIGHTS_CC_SMALL_FP8", &blob_of(&small), [310, 9, 16, 0])).unwrap();
        let root = dir.join("model");
        let done = extract_shaped(&dir, &root, Network::Small, &table(&small)).unwrap();
        assert_eq!((done.network, done.build.as_str(), done.verified), (Network::Small, "310.9.16", true));
        assert_eq!(installed_verified(&network_dir(&root, Network::Small)), Some(true));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn both_networks_are_extracted_when_the_dll_has_both_and_only_the_full_one_otherwise() {
        let dir = scratch("all");
        let (small, full) = (small_shape(), shape());
        let (small_table, full_table) = (table(&small), table(&full));
        let expected = |network| if network == Network::Small { &small_table[..] } else { &full_table[..] };
        let root = dir.join("model");
        let (full_blob, small_blob) = (weights(), blob_of(&small));
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &full_blob), ("WEIGHTS_CC_SMALL_FP8", &small_blob)], [310, 9, 16, 0])).unwrap();
        let done = extract_all_with(&dir, &root, &expected).unwrap();
        assert_eq!(done.networks.iter().map(|e| e.network).collect::<Vec<_>>(), Network::ALL);
        assert!(done.refused.is_empty());
        for network in Network::ALL {
            assert_eq!(check_installed(&network_dir(&root, network), network), Ok("310.9.16.0".to_string()), "{network}");
        }
        assert_eq!(installed_all(&root).iter().map(|i| (i.network, i.verified)).collect::<Vec<_>>(), [(Network::Full, true), (Network::Small, true)]);
        assert_eq!(installed_text(&root).unwrap(), "full network, build 310.9.16.0; small network, build 310.9.16.0");
        // A build with only the full network: the full one, and the small one extracted before is gone.
        std::fs::write(dir.join(DLL_NAME), dll(&full_blob, [310, 8, 0, 0])).unwrap();
        let done = extract_all_with(&dir, &root, &expected).unwrap();
        assert_eq!(done.networks.iter().map(|e| e.network).collect::<Vec<_>>(), [Network::Full]);
        assert!(done.refused.is_empty(), "no small network to refuse: nothing to explain");
        assert_eq!(installed_all(&root).iter().map(|i| i.network).collect::<Vec<_>>(), [Network::Full]);
        assert!(!network_dir(&root, Network::Small).exists());
        let mut names: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["full"], "the model directory holds one directory per network and nothing else");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extracting_one_network_leaves_the_other_alone() {
        let dir = scratch("one");
        let (small, full) = (small_shape(), shape());
        let (small_table, full_table) = (table(&small), table(&full));
        let expected = |network| if network == Network::Small { &small_table[..] } else { &full_table[..] };
        let root = dir.join("model");
        let (full_blob, small_blob) = (weights(), blob_of(&small));
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &full_blob), ("WEIGHTS_CC_SMALL_FP8", &small_blob)], [310, 9, 16, 0])).unwrap();
        extract_all_with(&dir, &root, &expected).unwrap();
        std::fs::write(dir.join(DLL_NAME), dll(&full_blob, [310, 9, 1, 0])).unwrap();
        extract_shaped(&dir, &root, Network::Full, &full_table).unwrap();
        assert_eq!(installed(&network_dir(&root, Network::Full)).as_deref(), Some("310.9.1.0"));
        assert_eq!(installed(&network_dir(&root, Network::Small)).as_deref(), Some("310.9.16.0"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_dll_without_the_small_network_says_so() {
        let dir = scratch("no-small");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 8, 0, 0])).unwrap();
        let out = dir.join("model");
        let err = extract(&dir, &out, Network::Small).unwrap_err();
        assert!(err.to_string().contains("build 310.8.0.0 has no small network (no RCDATA resource named WEIGHTS_CC_SMALL_FP8)"), "{err}");
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_refused_network_leaves_the_other_one_extracted() {
        let dir = scratch("refused");
        let (small, full) = (small_shape(), shape());
        let (small_table, full_table) = (table(&small), table(&full));
        let expected = |network| if network == Network::Small { &small_table[..] } else { &full_table[..] };
        let root = dir.join("model");
        let full_blob = weights();
        // A later build whose small network lost a block: the full one, and why the small one is not there.
        let changed_blob = blob_of(&small[..30]);
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &full_blob), ("WEIGHTS_CC_SMALL_FP8", &changed_blob)], [310, 9, 20, 0])).unwrap();
        let done = extract_all_with(&dir, &root, &expected).unwrap();
        assert_eq!(done.networks.iter().map(|e| e.network).collect::<Vec<_>>(), [Network::Full]);
        let [(network, why)] = &done.refused[..] else { panic!("{:?}", done.refused) };
        assert_eq!(*network, Network::Small);
        assert!(why.starts_with("nvngx_dlssnr.dll holds 30 tensors in 30 blocks (0-29)"), "{why}");
        assert_eq!(installed_all(&root).iter().map(|i| i.network).collect::<Vec<_>>(), [Network::Full]);
        // Asked for by name, the small network is refused, and the full one is not touched.
        assert!(matches!(extract_shaped(&dir, &root, Network::Small, &small_table), Err(ModelError::Shape(_))));
        assert!(check_installed(&network_dir(&root, Network::Full), Network::Full).is_ok());
        // The other way round: a full network the graph does not implement, the small one as it has it.
        let small_blob = blob_of(&small);
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &blob_of(&full[1..])), ("WEIGHTS_CC_SMALL_FP8", &small_blob)], [310, 9, 20, 0])).unwrap();
        let done = extract_all_with(&dir, &root, &expected).unwrap();
        assert_eq!(done.networks.iter().map(|e| e.network).collect::<Vec<_>>(), [Network::Small]);
        assert_eq!(done.refused.iter().map(|r| r.0).collect::<Vec<_>>(), [Network::Full]);
        // Neither: the full network's report, and the model directory as it was.
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &blob_of(&full[1..])), ("WEIGHTS_CC_SMALL_FP8", &changed_blob)], [310, 9, 20, 0])).unwrap();
        let err = extract_all_with(&dir, &root, &expected).unwrap_err();
        assert!(err.to_string().contains("the graph expects 153 tensors"), "{err}");
        assert_eq!(installed_all(&root).iter().map(|i| i.network).collect::<Vec<_>>(), [Network::Small]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extractions_into_one_directory_at_the_same_time_each_land_whole() {
        let dir = scratch("concurrent");
        let (small, full) = (small_shape(), shape());
        let (small_table, full_table) = (table(&small), table(&full));
        let (full_blob, small_blob) = (weights(), blob_of(&small));
        std::fs::write(dir.join(DLL_NAME), dll_of(&[("WEIGHTS_HT", &full_blob), ("WEIGHTS_CC_SMALL_FP8", &small_blob)], [310, 9, 16, 0])).unwrap();
        let root = dir.join("model");
        // The GUI's Extract button can start one while another writes; the CLI can write one network.
        std::thread::scope(|s| {
            let runs: Vec<_> = (0..9)
                .map(|i| {
                    let (dir, root, small_table, full_table) = (&dir, &root, &small_table, &full_table);
                    s.spawn(move || {
                        let expected = |n| if n == Network::Small { &small_table[..] } else { &full_table[..] };
                        match i % 3 {
                            0 => extract_all_with(dir, root, &expected).map(|_| ()),
                            k => {
                                let network = Network::ALL[k - 1];
                                extract_shaped(dir, root, network, expected(network)).map(|_| ())
                            }
                        }
                    })
                })
                .collect();
            for run in runs {
                run.join().unwrap().unwrap();
            }
        });
        for network in installed_all(&root).iter().map(|i| i.network) {
            let dir = network_dir(&root, network);
            assert!(check_installed(&dir, network).is_ok(), "{:?}", check_installed(&dir, network));
        }
        assert!(!installed_all(&root).is_empty());
        for d in [&*dir, root.as_path()] {
            let leftovers: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|e| e.file_name()).filter(|n| n.to_string_lossy().starts_with('.')).collect();
            assert!(leftovers.is_empty(), "{leftovers:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_manifest_naming_an_unknown_network_or_none_is_no_network() {
        let dir = scratch("manifest-unknown-network");
        std::fs::write(dir.join("manifest.json"), r#"{"source": {"build": "310.9.20.0"}, "network": "tiny", "stages": []}"#).unwrap();
        assert_eq!(installed_network(&dir), None);
        assert_eq!(check_installed(&dir, Network::Full), Err("the manifest names a network this version does not know: \"tiny\"".to_string()));
        std::fs::write(dir.join("manifest.json"), r#"{"source": {"build": "310.8.0.0"}, "stages": []}"#).unwrap();
        assert_eq!(installed_network(&dir), None);
        assert!(check_installed(&dir, Network::Full).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn network_names_and_settings_round_trip() {
        for network in Network::ALL {
            assert_eq!(Network::parse(network.name()), Some(network));
            assert_eq!(Network::from_setting(network.setting()), network);
            // The layer finds a network's directory by the setting's own name for it.
            assert_eq!(neural_forge_protocol::enums::network::name(network.setting()), network.name());
        }
        assert_eq!(Network::parse("tiny"), None);
        assert_eq!(Network::from_setting(9), Network::Full);
        assert_eq!(Network::from_setting(0), Network::Full, "the setting's default is the full network");
        let dir = scratch("network-dir");
        assert_eq!(network_dir(&dir, Network::Small), dir.join("small"));
        assert!(installed_all(&dir.join("nowhere")).is_empty());
        assert_eq!(installed_text(&dir.join("nowhere")), None);
        std::fs::remove_dir_all(&dir).unwrap();
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
        let files = build_model(&tensors, Network::Full, [310, 8, 0, 0], "00").unwrap();
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
    fn a_block_outside_the_graph_is_an_error_not_a_panic() {
        let bytes = [0u8; 4];
        let tensors = [Tensor { name: "block71.layer0.layer".into(), block: 71, layer: 0, parameter: "layer".into(), bytes: &bytes }];
        assert!(matches!(build_model(&tensors, Network::Full, [310, 8, 0, 0], "00"), Err(ModelError::Weights(_))));
    }

    #[test]
    fn the_installed_model_is_checked_against_its_manifest() {
        let dir = std::env::temp_dir().join(format!("neural-forge-model-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bytes = [7u8; 32];
        let tensors = [Tensor { name: "block0.layer0.layer".into(), block: 0, layer: 0, parameter: "layer".into(), bytes: &bytes }];
        let files = build_model(&tensors, Network::Full, [310, 8, 0, 0], "00").unwrap();
        for (rel, data) in &files {
            std::fs::create_dir_all(dir.join(rel).parent().unwrap()).unwrap();
            std::fs::write(dir.join(rel), data).unwrap();
        }
        assert_eq!(check_installed(&dir, Network::Full), Ok("310.8.0.0".to_string()));
        let (first, _) = files.iter().find(|(p, d)| p.starts_with("model/") && !d.is_empty()).unwrap();
        std::fs::write(dir.join(first), [0u8; 32]).unwrap();
        assert!(check_installed(&dir, Network::Full).unwrap_err().contains("SHA-256"));
        std::fs::write(dir.join(first), [0u8; 3]).unwrap();
        assert!(check_installed(&dir, Network::Full).unwrap_err().contains("bytes"));
        std::fs::remove_file(dir.join(first)).unwrap();
        assert!(check_installed(&dir, Network::Full).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn extracts_a_verified_build() {
        let dir = scratch("verified");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 8, 0, 3])).unwrap();
        let root = dir.join("model");
        let done = extract_shaped(&dir, &root, Network::Full, &table(&shape())).unwrap();
        assert_eq!((done.build.as_str(), done.verified, done.tensors), ("310.8.0", true, TENSOR_COUNT));
        let out = network_dir(&root, Network::Full);
        assert_eq!(installed_network(&out), Some(Network::Full));
        assert!(out.join("manifest.json").is_file() && out.join("model/vit.e4m3").is_file());
        assert_eq!(installed(&out).as_deref(), Some("310.8.0.3"));
        assert_eq!(installed_verified(&out), Some(true));
        assert_eq!((installed(&dir.join("nowhere")), installed_verified(&dir.join("nowhere"))), (None, None));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2, "only the DLL and the model directory");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1, "only the full network's directory");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extracts_another_build_with_the_same_network_unverified() {
        let dir = scratch("unverified");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 9, 1, 0])).unwrap();
        let root = dir.join("model");
        let done = extract_shaped(&dir, &root, Network::Full, &table(&shape())).unwrap();
        assert_eq!((done.build.as_str(), done.verified), ("310.9.1", false));
        let out = network_dir(&root, Network::Full);
        assert_eq!(installed(&out).as_deref(), Some("310.9.1.0"));
        assert_eq!(installed_verified(&out), Some(false));
        assert_eq!(installed_text(&root).unwrap(), "full network, build 310.9.1.0, not verified against NVIDIA's runtime");
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
        let err = extract(&dir, &out, Network::Full).unwrap_err();
        assert!(matches!(err, ModelError::Shape(_)), "{err}");
        assert!(err.to_string().contains("the graph expects 153 tensors in 71 blocks (0-70), 147683778 bytes"), "{err}");
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
