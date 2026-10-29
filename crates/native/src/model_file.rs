//! The model's files, read and checked before the C++ side sees them: `manifest.json` and the stage files it
//! lists under `model/` (`extract-model`'s output). The C++ `nr::Model` gets the result as a table of tensor
//! records over the stages' bytes (`NfNativeModel` in `cpp/nf_native.h`), which [`ModelFile`] owns and keeps
//! where they are for as long as it lives.
//!
//! Everything a record says is checked here: each stage file is a regular file of the manifest's length and
//! SHA-256, every record names one stage and lies inside it, and no two stages or records share a name.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, CString};
use std::path::Path;

use sha2::{Digest, Sha256};

/// One tensor record as the C side reads it (`NfNativeTensor`).
#[repr(C)]
pub(crate) struct Record {
    name: *const c_char,
    parameter: *const c_char,
    stage: *const c_char,
    block: i32,
    layer: i32,
    stage_offset: u32,
    byte_length: u32,
    bytes: *const u8,
}

/// The model as the C side reads it (`NfNativeModel`).
#[repr(C)]
pub(crate) struct View {
    block_count: u32,
    tensors: *const Record,
    tensor_count: usize,
}

/// A model directory, loaded and checked.
pub struct ModelFile {
    block_count: u32,
    // The records point into these: neither is touched again after `load`.
    _stages: Vec<Vec<u8>>,
    _strings: Vec<CString>,
    records: Vec<Record>,
}

impl std::fmt::Debug for ModelFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelFile").field("block_count", &self.block_count).field("tensors", &self.records.len()).finish_non_exhaustive()
    }
}

// SAFETY: the raw pointers in `records` point only into `_stages` and `_strings`, which the value owns and never
// changes after `load`: moving it to another thread moves what they point to with it (the heap buffers stay put).
unsafe impl Send for ModelFile {}

/// `value[key]` of a JSON object, or which key is missing.
fn field<'a>(value: &'a serde_json::Value, key: &str, what: &str) -> Result<&'a serde_json::Value, String> {
    value.get(key).ok_or_else(|| format!("{what} has no {key:?}"))
}

fn text<'a>(value: &'a serde_json::Value, key: &str, what: &str) -> Result<&'a str, String> {
    field(value, key, what)?.as_str().ok_or_else(|| format!("{what}'s {key:?} is not a string"))
}

fn unsigned(value: &serde_json::Value, key: &str, what: &str) -> Result<u32, String> {
    let v = field(value, key, what)?;
    v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| format!("{what}'s {key:?} is not a 32-bit unsigned integer: {v}"))
}

fn signed(value: &serde_json::Value, key: &str, what: &str) -> Result<i32, String> {
    let v = field(value, key, what)?;
    v.as_i64().and_then(|n| i32::try_from(n).ok()).ok_or_else(|| format!("{what}'s {key:?} is not a 32-bit integer: {v}"))
}

fn array<'a>(value: &'a serde_json::Value, key: &str, what: &str) -> Result<&'a [serde_json::Value], String> {
    field(value, key, what)?.as_array().map(Vec::as_slice).ok_or_else(|| format!("{what}'s {key:?} is not an array"))
}

/// The whole of a regular file: a directory, a FIFO or a device is refused before it is opened.
fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("cannot read {} (not a file)", path.display()));
    }
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02X}")).collect()
}

impl ModelFile {
    /// Reads and checks the model in `dir`: the manifest, then each stage file (its length and SHA-256), then
    /// every tensor record against the stages. The first problem found is the error.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("manifest.json");
        let manifest: serde_json::Value = serde_json::from_slice(&read_file(&path)?).map_err(|e| format!("{}: {e}", path.display()))?;
        let block_count = unsigned(field(&manifest, "totals", "the manifest")?, "blockCount", "the manifest's \"totals\"")?;

        let mut stages = Vec::new();
        let mut stage_index = HashMap::new();
        for stage in array(&manifest, "stages", "the manifest")? {
            let id = text(stage, "id", "a stage")?;
            let what = format!("stage {id}");
            let file = text(stage, "file", &what)?;
            // A plain file name: the manifest cannot point outside the model directory.
            if file.is_empty() || file.contains('/') || file.starts_with('.') {
                return Err(format!("{what} names an unexpected file {file:?}"));
            }
            let length = field(stage, "packedByteLength", &what)?;
            let length = length.as_u64().ok_or_else(|| format!("{what}'s \"packedByteLength\" is not an unsigned integer: {length}"))?;
            let digest = text(stage, "sha256", &what)?;
            let path = dir.join("model").join(file);
            let bytes = read_file(&path)?;
            if bytes.len() as u64 != length {
                return Err(format!("{what}: {} is {} bytes, the manifest says {length}", path.display(), bytes.len()));
            }
            if !sha256_hex(&bytes).eq_ignore_ascii_case(digest) {
                return Err(format!("{what}: the SHA-256 of {} differs from the manifest's", path.display()));
            }
            if stage_index.insert(id.to_string(), stages.len()).is_some() {
                return Err(format!("the manifest lists {what} twice"));
            }
            stages.push((CString::new(id).map_err(|_| format!("a NUL in the name of {what}"))?, bytes));
        }

        let entries = array(&manifest, "tensors", "the manifest")?;
        let mut names = HashSet::new();
        let mut strings = Vec::with_capacity(entries.len() * 2);
        let mut records = Vec::with_capacity(entries.len());
        for entry in entries {
            let name = text(entry, "name", "a tensor")?;
            let what = format!("tensor {name}");
            let block = signed(entry, "block", &what)?;
            let layer = signed(entry, "layer", &what)?;
            let parameter = text(entry, "parameter", &what)?;
            let stage = text(entry, "stage", &what)?;
            let stage_offset = unsigned(entry, "stageOffset", &what)?;
            let byte_length = unsigned(entry, "byteLength", &what)?;
            let &index = stage_index.get(stage).ok_or_else(|| format!("{what} is in stage {stage}, which the manifest does not list"))?;
            let (stage_name, bytes) = &stages[index];
            if u64::from(stage_offset) + u64::from(byte_length) > bytes.len() as u64 {
                return Err(format!("{what} ({byte_length} bytes at {stage_offset}) runs past the end of stage {stage} ({} bytes)", bytes.len()));
            }
            // A second record of the same name would hide the first from every lookup.
            if !names.insert(name) {
                return Err(format!("the manifest lists {what} twice"));
            }
            let name = CString::new(name).map_err(|_| format!("a NUL in the name of {what}"))?;
            let parameter = CString::new(parameter).map_err(|_| format!("a NUL in the parameter of {what}"))?;
            records.push(Record {
                name: name.as_ptr(),
                parameter: parameter.as_ptr(),
                stage: stage_name.as_ptr(),
                block,
                layer,
                stage_offset,
                byte_length,
                // SAFETY: inside the stage's bytes (checked above); `add` of the stage's own length at most.
                bytes: unsafe { bytes.as_ptr().add(stage_offset as usize) },
            });
            strings.extend([name, parameter]);
        }
        let (stage_names, stages): (Vec<_>, Vec<_>) = stages.into_iter().unzip();
        strings.extend(stage_names);
        Ok(Self { block_count, _stages: stages, _strings: strings, records })
    }

    /// The table the C side reads: valid while `self` lives, and only as long as nothing else borrows it mutably.
    pub(crate) fn view(&self) -> View {
        View { block_count: self.block_count, tensors: self.records.as_ptr(), tensor_count: self.records.len() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::path::PathBuf;

    /// A scratch model directory under the system's temporary directory, removed at drop.
    struct Dir(PathBuf);

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch(test: &str) -> Dir {
        let dir = std::env::temp_dir().join(format!("model-file-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("model")).unwrap();
        Dir(dir)
    }

    /// Two stages and three records, as `extract-model` lays them out.
    fn manifest(stage0: &[u8], stage1: &[u8]) -> serde_json::Value {
        serde_json::json!({
            "network": "full",
            "totals": { "blockCount": 2 },
            "stages": [
                { "id": "s0", "file": "s0.bin", "packedByteLength": stage0.len(), "sha256": sha256_hex(stage0) },
                { "id": "s1", "file": "s1.bin", "packedByteLength": stage1.len(), "sha256": sha256_hex(stage1).to_lowercase() },
            ],
            "tensors": [
                { "name": "block0.layer0.layer", "block": 0, "layer": 0, "parameter": "layer", "stage": "s0", "stageOffset": 0, "byteLength": 6 },
                { "name": "block0.layer1.bias", "block": 0, "layer": 1, "parameter": "bias", "stage": "s0", "stageOffset": 6, "byteLength": 2 },
                { "name": "block1.layer0.blend_scale", "block": 1, "layer": 0, "parameter": "blend_scale", "stage": "s1", "stageOffset": 1, "byteLength": 3 },
            ],
        })
    }

    fn write(dir: &Dir, manifest: &serde_json::Value, stage0: &[u8], stage1: &[u8]) {
        std::fs::write(dir.0.join("manifest.json"), serde_json::to_string_pretty(manifest).unwrap()).unwrap();
        std::fs::write(dir.0.join("model/s0.bin"), stage0).unwrap();
        std::fs::write(dir.0.join("model/s1.bin"), stage1).unwrap();
    }

    const STAGE0: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];
    const STAGE1: &[u8] = &[9, 10, 11, 12];

    fn load_with(test: &str, edit: impl FnOnce(&mut serde_json::Value)) -> Result<ModelFile, String> {
        let dir = scratch(test);
        let mut m = manifest(STAGE0, STAGE1);
        edit(&mut m);
        write(&dir, &m, STAGE0, STAGE1);
        ModelFile::load(&dir.0)
    }

    fn cstr<'a>(p: *const c_char) -> &'a str {
        // SAFETY: a pointer from a live ModelFile's own strings.
        unsafe { CStr::from_ptr(p) }.to_str().unwrap()
    }

    /// Every record's fields and bytes as the C side sees them; either case of hex digest is taken.
    #[test]
    fn a_good_directory_loads_into_the_c_table() {
        let model = load_with("good", |_| {}).unwrap();
        let view = model.view();
        assert_eq!((view.block_count, view.tensor_count), (2, 3));
        // SAFETY: the table and its bytes live as long as `model`.
        let records = unsafe { std::slice::from_raw_parts(view.tensors, view.tensor_count) };
        let got: Vec<_> = records
            .iter()
            .map(|r| {
                // SAFETY: inside the stage (checked at load).
                let bytes = unsafe { std::slice::from_raw_parts(r.bytes, r.byte_length as usize) };
                (cstr(r.name), cstr(r.parameter), cstr(r.stage), r.block, r.layer, r.stage_offset, bytes.to_vec())
            })
            .collect();
        assert_eq!(
            got,
            [
                ("block0.layer0.layer", "layer", "s0", 0, 0, 0, vec![1, 2, 3, 4, 5, 6]),
                ("block0.layer1.bias", "bias", "s0", 0, 1, 6, vec![7, 8]),
                ("block1.layer0.blend_scale", "blend_scale", "s1", 1, 0, 1, vec![10, 11, 12]),
            ]
        );
        // The table survives a move of its owner (another thread, a struct field).
        let moved = std::thread::spawn(move || model).join().unwrap();
        let view = moved.view();
        // SAFETY: as above.
        let first = unsafe { &*view.tensors };
        assert_eq!(cstr(first.name), "block0.layer0.layer");
        // SAFETY: as above.
        assert_eq!(unsafe { *first.bytes.add(5) }, 6);
    }

    #[test]
    fn the_sha256_is_the_standard_one() {
        assert_eq!(sha256_hex(b""), "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");
        assert_eq!(sha256_hex(&[b'a'; 64]), "FFE054FE7AE0CB6DC65C3AF9B61D5209F439851DB43D0BA5997337DF154668EB");
    }

    #[test]
    fn a_missing_or_unreadable_manifest_is_refused() {
        let dir = scratch("nomanifest");
        assert!(ModelFile::load(&dir.0).unwrap_err().starts_with("cannot read "));
        std::fs::create_dir(dir.0.join("manifest.json")).unwrap();
        assert!(ModelFile::load(&dir.0).unwrap_err().ends_with("manifest.json (not a file)"));
        std::fs::remove_dir(dir.0.join("manifest.json")).unwrap();
        std::fs::write(dir.0.join("manifest.json"), "{\"totals\": tru}").unwrap();
        assert!(ModelFile::load(&dir.0).unwrap_err().contains("manifest.json: "));
    }

    #[test]
    fn a_stage_of_the_wrong_length_or_digest_is_refused() {
        let dir = scratch("stagecheck");
        let m = manifest(STAGE0, STAGE1);
        write(&dir, &m, STAGE0, &STAGE1[..3]);
        assert!(ModelFile::load(&dir.0).unwrap_err().contains("is 3 bytes, the manifest says 4"));
        write(&dir, &m, STAGE0, &[9, 10, 11, 13]);
        assert!(ModelFile::load(&dir.0).unwrap_err().starts_with("stage s1: the SHA-256 of "));
        write(&dir, &m, STAGE0, STAGE1);
        std::fs::remove_file(dir.0.join("model/s1.bin")).unwrap();
        std::fs::create_dir(dir.0.join("model/s1.bin")).unwrap();
        assert!(ModelFile::load(&dir.0).unwrap_err().ends_with("s1.bin (not a file)"));
    }

    #[test]
    fn a_stage_file_outside_the_model_directory_is_refused() {
        for file in ["../s0.bin", ".s0.bin", "", "a/b"] {
            let why = load_with("outside", |m| m["stages"][0]["file"] = file.into()).unwrap_err();
            assert!(why.contains("names an unexpected file"), "{file}: {why}");
        }
    }

    /// A change to the good manifest.
    type Edit = fn(&mut serde_json::Value);

    #[test]
    fn bad_records_are_refused() {
        let cases: [(&str, Edit, &str); 9] = [
            ("unknownstage", |m| m["tensors"][1]["stage"] = "s9".into(), "is in stage s9, which the manifest does not list"),
            ("pastend", |m| m["tensors"][1]["byteLength"] = 3.into(), "runs past the end of stage s0 (8 bytes)"),
            ("offsetwrap", |m| m["tensors"][2]["stageOffset"] = u32::MAX.into(), "runs past the end of stage s1"),
            ("duptensor", |m| m["tensors"][1]["name"] = "block0.layer0.layer".into(), "lists tensor block0.layer0.layer twice"),
            ("dupstage", |m| m["stages"][1]["id"] = "s0".into(), "lists stage s0 twice"),
            ("negative", |m| m["tensors"][0]["stageOffset"] = (-1).into(), "\"stageOffset\" is not a 32-bit unsigned integer"),
            ("fraction", |m| m["tensors"][0]["byteLength"] = 1.5.into(), "\"byteLength\" is not a 32-bit unsigned integer"),
            ("missing", |m| drop(m["tensors"][0].as_object_mut().unwrap().remove("layer")), "tensor block0.layer0.layer has no \"layer\""),
            ("totals", |m| m["totals"]["blockCount"] = "2".into(), "\"blockCount\" is not a 32-bit unsigned integer"),
        ];
        for (test, edit, expected) in cases {
            let why = load_with(test, edit).err().unwrap_or_else(|| panic!("{test}: loaded"));
            assert!(why.contains(expected), "{test}: {why}");
        }
    }

    /// A record of no bytes at the very end of a stage is inside it.
    #[test]
    fn an_empty_record_at_the_end_of_a_stage_loads() {
        let model = load_with("emptyend", |m| {
            m["tensors"][1]["stageOffset"] = 8.into();
            m["tensors"][1]["byteLength"] = 0.into();
        })
        .unwrap();
        let view = model.view();
        assert_eq!((view.block_count, view.tensor_count), (2, 3));
    }
}
