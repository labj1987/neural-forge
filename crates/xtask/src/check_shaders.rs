//! Ties the committed SPIR-V binaries to the GLSL sources they were built from.
//!
//! `crates/layer/shaders/*.spv` are committed binaries (embedded with `include_bytes!`); nothing builds them, so an
//! edited `.comp` could silently ship a stale `.spv`. Each folder's `shaders.sha256` records the SHA-256 of each
//! source and of the binary compiled from it.
//!
//!   cargo xtask check-shaders            verify: every source hash and binary hash matches the manifest, no .spv is
//!                                        left without its source, and each binary passes spirv-val (required:
//!                                        install spirv-tools).
//!   cargo xtask check-shaders --update   recompile every .comp with the pinned glslang (scripts/fetch-native-tools.sh;
//!                                        never one on PATH), validate with spirv-val, remove .spv files whose source
//!                                        is gone, and rewrite the manifest.
//!
//! Edit a shader, run `--update`, commit the .comp, .spv and manifest together. Verification is a pure hash
//! comparison on purpose: SPIR-V bytes differ between glslang releases, so a CI-side recompile-and-compare would fail
//! for reasons that are not a stale binary.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use sha2::{Digest, Sha256};

fn sha(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Exits 1 with the message on stderr (Python's `sys.exit(message)`).
fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1)
}

fn with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == ext) && !name(p).starts_with('.')).collect())
        .unwrap_or_default();
    found.sort();
    found
}

fn name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The glslang fetch-native-tools.sh pins: the committed binaries are its output, and another release writes
/// different bytes.
fn pinned_glslang() -> PathBuf {
    let tools = std::env::var("NEURAL_FORGE_NATIVE_TOOLS").unwrap_or_default();
    let tools = if tools.trim().is_empty() { crate::root().join("tools/native") } else { PathBuf::from(tools.trim()) };
    tools.join("glslang/bin/glslang")
}

fn spirv_val() -> PathBuf {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join("spirv-val"))
        .find(|p| p.is_file() && std::fs::metadata(p).is_ok_and(|m| std::os::unix::fs::PermissionsExt::mode(&m.permissions()) & 0o111 != 0))
        .unwrap_or_else(|| fail("spirv-val is required (apt install spirv-tools): the binaries are validated, not only compared"))
}

fn validate(spv: &Path) -> bool {
    let out = Command::new(spirv_val()).arg(spv).output().unwrap_or_else(|e| fail(&format!("spirv-val: {e}")));
    if !out.status.success() {
        println!("  {} failed spirv-val:\n{}{}", name(spv), String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    }
    out.status.success()
}

/// A private temporary directory, removed when dropped.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn update(shaders: &Path) -> u8 {
    let glslang = pinned_glslang();
    if !glslang.is_file() {
        eprintln!("{} is missing: run scripts/fetch-native-tools.sh (--update uses only the pinned glslang)", glslang.display());
        return 2;
    }
    spirv_val();
    let mut lines = Vec::new();
    let sources = with_extension(shaders, "comp");
    for comp in &sources {
        let spv = comp.with_extension("spv");
        let tmp = TempDir(std::env::temp_dir().join(format!("check-shaders-{}-{}", std::process::id(), name(comp))));
        let _ = std::fs::remove_dir_all(&tmp.0);
        std::fs::create_dir_all(&tmp.0).unwrap_or_else(|e| fail(&format!("{}: {e}", tmp.0.display())));
        let out = tmp.0.join(name(&spv));
        let result = Command::new(&glslang).arg("-V").arg(comp).arg("-o").arg(&out).output().unwrap_or_else(|e| fail(&format!("{}: {e}", glslang.display())));
        if !result.status.success() {
            eprintln!("{} does not compile:\n{}{}", name(comp), String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
            return 1;
        }
        if !validate(&out) {
            return 1;
        }
        std::fs::copy(&out, &spv).unwrap_or_else(|e| fail(&format!("{}: {e}", spv.display())));
        lines.push(format!("{} {} {} {}", name(comp), sha(comp), name(&spv), sha(&spv)));
        println!("rebuilt {}", name(&spv));
    }
    let wanted: BTreeSet<PathBuf> = sources.iter().map(|c| c.with_extension("spv")).collect();
    for orphan in with_extension(shaders, "spv").into_iter().filter(|s| !wanted.contains(s)) {
        std::fs::remove_file(&orphan).unwrap_or_else(|e| fail(&format!("{}: {e}", orphan.display())));
        println!("removed {}: its source is gone", name(&orphan));
    }
    let manifest = shaders.join("shaders.sha256");
    std::fs::write(&manifest, lines.join("\n") + "\n").unwrap_or_else(|e| fail(&format!("{}: {e}", manifest.display())));
    0
}

fn verify(shaders: &Path) -> u8 {
    let manifest = shaders.join("shaders.sha256");
    if !manifest.is_file() {
        eprintln!("missing {}; run cargo xtask check-shaders --update", manifest.display());
        return 1;
    }
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| fail(&format!("{}: {e}", manifest.display())));
    // In the manifest's order, as Python's dict keeps it: what is left over is reported in that order.
    let mut recorded: Vec<(String, (String, String, String))> = Vec::new();
    for line in crate::py::splitlines(&text) {
        if line.trim().is_empty() {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let [comp, comp_hash, spv, spv_hash] = words[..] else { fail(&format!("{}: bad line {line:?}", manifest.display())) };
        let entry = (comp_hash.to_string(), spv.to_string(), spv_hash.to_string());
        match recorded.iter_mut().find(|(c, _)| c == comp) {
            Some(slot) => slot.1 = entry,
            None => recorded.push((comp.to_string(), entry)),
        }
    }
    let mut failures = Vec::new();
    let sources = with_extension(shaders, "comp");
    for comp in &sources {
        let comp_name = name(comp);
        let Some(i) = recorded.iter().position(|(c, _)| *c == comp_name) else {
            failures.push(format!("{comp_name}: not in shaders.sha256"));
            continue;
        };
        let (_, (comp_hash, spv_name, spv_hash)) = recorded.remove(i);
        let spv = shaders.join(&spv_name);
        if sha(comp) != comp_hash {
            failures.push(format!("{comp_name}: source changed since its .spv was built -- run cargo xtask check-shaders --update"));
        } else if !spv.is_file() || sha(&spv) != spv_hash {
            failures.push(format!("{spv_name}: binary does not match the manifest -- run cargo xtask check-shaders --update"));
        } else if !validate(&spv) {
            failures.push(format!("{spv_name}: fails spirv-val"));
        }
    }
    failures.extend(recorded.iter().map(|(name, _)| format!("{name}: listed in shaders.sha256 but the source is gone")));
    let listed: HashMap<String, ()> =
        recorded.iter().map(|(_, e)| (e.1.clone(), ())).chain(sources.iter().map(|c| (name(&c.with_extension("spv")), ()))).collect();
    for spv in with_extension(shaders, "spv") {
        if !listed.contains_key(&name(&spv)) {
            failures.push(format!("{}: a binary without its source -- run cargo xtask check-shaders --update", name(&spv)));
        }
    }
    for failure in &failures {
        eprintln!("{failure}");
    }
    if failures.is_empty() {
        println!("Shaders: OK");
    }
    u8::from(!failures.is_empty())
}

/// The folders of shaders, each with its own `shaders.sha256`.
const DIRS: &[&str] = &["crates/layer/shaders"];

pub fn main(args: &[String]) -> ExitCode {
    let mut status = 0;
    for dir in DIRS {
        let shaders = crate::root().join(dir);
        status |= if args.iter().any(|a| a == "--update") { update(&shaders) } else { verify(&shaders) };
    }
    ExitCode::from(status)
}
