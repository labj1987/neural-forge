//! Regenerates the layer framework's generated sources (third_party/vk-layer-for-rust: the vk_layer.h binding,
//! layer_trait/generated.rs and global_simple_intercept/generated.rs) from the Vulkan headers they are built
//! against, or with --check verifies that the committed files are what the generator gives (CI runs that).
//! The headers are Vulkan-Headers at commit 38899bf6 (1.3.281, the registry ash 0.38.0 matches), fetched
//! once, SHA-256 checked, into tools/native/ (or NEURAL_FORGE_NATIVE_TOOLS). Needs libclang (bindgen; set
//! LIBCLANG_PATH if it is not found) and the pinned toolchain's rustfmt.
//!
//!   cargo xtask vk-layer-codegen [--check]
//!
//! It replaced scripts/vk-layer-codegen.sh: the same output and files, and the generator's own exit code.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::fetch;
use crate::sh::{self, Fail, Res};

const URL: &str = "https://github.com/KhronosGroup/Vulkan-Headers/archive/38899bf60605f0a2c50f7968e5c4c9826a824683.tar.gz";
const SHA: &str = "da587cbace0dd9ea5f2297d4118b4cf63831c2e65dc9a7369a33524b9c56831d";

/// The generator's link to the headers, removed (with the directory made for it, when that is empty) however the run
/// ends.
struct Link(PathBuf);

impl Drop for Link {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        if let Some(dir) = self.0.parent() {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// The versions in a name (`llvm-18` gives [18]), for `sort -V`'s order of the LLVM directories.
fn version_key(name: &str) -> Vec<u64> {
    name.split(|c: char| !c.is_ascii_digit()).filter_map(|n| n.parse().ok()).collect()
}

/// Where libclang is when the loader's cache does not list it: the newest /usr/lib/llvm-*/lib that holds one.
fn llvm_libclang_dir() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir("/usr/lib")
        .map(|entries| entries.flatten().map(|e| e.path().join("lib")).filter(|p| p.parent().is_some_and(is_llvm) && p.exists()).collect())
        .unwrap_or_default();
    dirs.sort_by_key(|d| std::cmp::Reverse(version_key(&d.to_string_lossy())));
    dirs.into_iter().find(|dir| {
        std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.flatten().any(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with("libclang") && name.contains(".so")
            })
        })
    })
}

fn is_llvm(dir: &Path) -> bool {
    dir.file_name().is_some_and(|n| n.to_string_lossy().starts_with("llvm-"))
}

fn ldconfig_lists_libclang() -> bool {
    Command::new("ldconfig").arg("-p").stderr(std::process::Stdio::null()).output().is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libclang.so "))
}

fn run(args: &[String]) -> Res<ExitCode> {
    sh::trap_signals();
    let root = crate::root();
    let framework = root.join("third_party/vk-layer-for-rust");
    let headers = fetch::tools_dir().join("Vulkan-Headers-vk-layer");
    let target = sh::var("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from);

    if !headers.join("registry/vk.xml").exists() || !fetch::stamped(&headers, SHA) {
        fetch::fetch_one_directory("Vulkan-Headers (vk-layer)", "vk-layer-headers", URL, SHA, "Vulkan-Headers-", &headers)?;
    }

    // The generator reads the headers from third_party/Vulkan-Headers inside the framework (git-ignored).
    let link = Link(framework.join("third_party/Vulkan-Headers"));
    fetch::create_dir_all(&framework.join("third_party"))?;
    let _ = std::fs::remove_file(&link.0);
    std::os::unix::fs::symlink(&headers, &link.0).map_err(|e| Fail::new(1, format!("{}: {e}", link.0.display())))?;

    let mut codegen = Command::new("cargo");
    codegen.args(["run", "--quiet", "--locked", "--release", "-p", "vulkan-layer-scripts", "--bin", "codegen", "--"]).args(args);
    codegen.current_dir(&framework).env("CARGO_TARGET_DIR", target).env("RUST_LOG", "error");
    if sh::var("LIBCLANG_PATH").is_none() && !ldconfig_lists_libclang() {
        if let Some(dir) = llvm_libclang_dir() {
            codegen.env("LIBCLANG_PATH", dir);
        }
    }
    sh::check(&mut codegen)?;
    Ok(ExitCode::SUCCESS)
}

pub fn main(args: &[String]) -> ExitCode {
    sh::finish(run(args))
}

#[cfg(test)]
mod tests {
    #[test]
    fn llvm_directories_sort_by_version_not_as_text() {
        let mut names = ["/usr/lib/llvm-9/lib", "/usr/lib/llvm-18/lib", "/usr/lib/llvm-17/lib"];
        names.sort_by_key(|d| std::cmp::Reverse(super::version_key(d)));
        assert_eq!(names, ["/usr/lib/llvm-18/lib", "/usr/lib/llvm-17/lib", "/usr/lib/llvm-9/lib"]);
    }
}
