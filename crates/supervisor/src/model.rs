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
//! Nothing produced here is ever committed, uploaded or packaged.

use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};

/// The only build whose network the graph implements (71 blocks).
pub const SUPPORTED_BUILD: [u16; 3] = [310, 8, 0];
pub const DLL_NAME: &str = "nvngx_dlssnr.dll";
const RESOURCE_NAME: &str = "WEIGHTS_HT";
const BLOCK_COUNT: u32 = 71;
const TENSOR_COUNT: usize = 153;
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
    Build(String),
    Weights(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(path, e) => write!(f, "{}: {e}", path.display()),
            ModelError::Pe(m) => write!(f, "not a readable DLL: {m}"),
            ModelError::Build(found) => write!(
                f,
                "{DLL_NAME} is build {found}; the native model needs build {}.{}.{}",
                SUPPORTED_BUILD[0], SUPPORTED_BUILD[1], SUPPORTED_BUILD[2]
            ),
            ModelError::Weights(m) => write!(f, "weights resource: {m}"),
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
    sections: Vec<(u32, u32, u32)>, // virtual address, virtual size, raw offset
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
                Some((u32_at(image, s + 12)?, u32_at(image, s + 8)?.max(u32_at(image, s + 16)?), u32_at(image, s + 20)?))
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

    fn offset(&self, rva: u32) -> Option<usize> {
        self.sections
            .iter()
            .find(|(va, size, _)| rva >= *va && rva - va < *size)
            .map(|(va, _, raw)| (raw + (rva - va)) as usize)
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
    if tensors.len() != TENSOR_COUNT {
        return Err(w_err(format!("{} tensors, expected {TENSOR_COUNT}", tensors.len())));
    }
    let blocks = tensors.iter().map(|t| t.block).max().map_or(0, |m| m + 1);
    if blocks != BLOCK_COUNT || (0..BLOCK_COUNT).any(|b| !tensors.iter().any(|t| t.block == b)) {
        return Err(w_err(format!("blocks 0..{blocks}, expected exactly 0..{BLOCK_COUNT}")));
    }
    Ok(tensors)
}

fn stage_of(block: u32) -> &'static str {
    STAGES.iter().find(|(_, first, last)| (*first..=*last).contains(&block)).map(|s| s.0).expect("blocks are 0..=70")
}

/// Upper case: `nr::Model` compares its own upper-case digest (`sha256.h`) as a string.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02X}")).collect()
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
        },
        "totals": {
            "blockCount": BLOCK_COUNT,
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

/// The build of the DLL the installed model directory came from (the manifest's `source.build`),
/// or `None` when there is no readable manifest there.
pub fn installed(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest["source"]["build"].as_str().map(str::to_string)
}

#[derive(Debug)]
pub struct Extracted {
    pub dir: PathBuf,
    pub build: String,
    pub tensors: usize,
    pub bytes: usize,
}

/// `source` is the DLL or a directory holding it. Writes into `out`, each file through a temporary
/// name and a rename, so a model directory is never half old and half new file by file.
pub fn extract(source: &Path, out: &Path) -> Result<Extracted, ModelError> {
    let dll = if source.is_dir() { source.join(DLL_NAME) } else { source.to_path_buf() };
    let image = std::fs::read(&dll).map_err(|e| ModelError::Io(dll.clone(), e))?;
    let resources = Resources::parse(&image)?;
    let version = file_version(&resources).ok_or_else(|| pe_err("no version resource"))?;
    if version[..3] != SUPPORTED_BUILD {
        return Err(ModelError::Build(format!("{}.{}.{}.{}", version[0], version[1], version[2], version[3])));
    }
    let blob = resources
        .get(RT_RCDATA, Key::Name(RESOURCE_NAME))
        .ok_or_else(|| w_err(format!("no RCDATA resource named {RESOURCE_NAME}")))?;
    let tensors = parse_weights(blob)?;
    let files = build_model(&tensors, version, &sha256_hex(&image));
    std::fs::create_dir_all(out.join("model")).map_err(|e| ModelError::Io(out.to_path_buf(), e))?;
    for (relative, bytes) in &files {
        let path = out.join(relative);
        let temporary = out.join(format!("{relative}.partial"));
        std::fs::write(&temporary, bytes).map_err(|e| ModelError::Io(temporary.clone(), e))?;
        std::fs::rename(&temporary, &path).map_err(|e| ModelError::Io(path.clone(), e))?;
    }
    Ok(Extracted {
        dir: out.to_path_buf(),
        build: format!("{}.{}.{}", version[0], version[1], version[2]),
        tensors: tensors.len(),
        bytes: tensors.iter().map(|t| t.bytes.len()).sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A blob with every block present and exactly 153 records, like the real one.
    fn weights() -> Vec<u8> {
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
        let mut body = Vec::new();
        for (i, n) in names.iter().enumerate() {
            body.extend(record(n, &vec![i as u8 + 1; 20 + i], 20));
        }
        let mut blob = ((body.len() + 8) as u64).to_le_bytes().to_vec();
        blob.extend(body);
        blob
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
    fn refuses_a_wrong_total_or_a_missing_record() {
        let mut blob = weights();
        blob.push(0);
        assert!(parse_weights(&blob).unwrap_err().to_string().contains("header says"));
        let blob = weights();
        let first = 8 + record("block0.layer0.layer", &[1; 20], 20).len();
        let mut short = blob[..8].to_vec();
        short.extend_from_slice(&blob[first..]);
        let total = short.len() as u64;
        short[..8].copy_from_slice(&total.to_le_bytes());
        assert!(parse_weights(&short).is_err());
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
    fn extracts_from_a_dll_and_refuses_another_build() {
        let dir = scratch("extract");
        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 8, 0, 0])).unwrap();
        let out = dir.join("model");
        let done = extract(&dir, &out).unwrap();
        assert_eq!((done.build.as_str(), done.tensors), ("310.8.0", TENSOR_COUNT));
        assert!(out.join("manifest.json").is_file() && out.join("model/vit.e4m3").is_file());
        assert_eq!(installed(&out).as_deref(), Some("310.8.0.0"));
        assert_eq!(installed(&dir.join("nowhere")), None);
        assert!(!out.join("manifest.json.partial").exists());

        std::fs::write(dir.join(DLL_NAME), dll(&weights(), [310, 9, 1, 0])).unwrap();
        let err = extract(&dir, &out).unwrap_err().to_string();
        assert!(err.contains("build 310.9.1.0") && err.contains("needs build 310.8.0"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
