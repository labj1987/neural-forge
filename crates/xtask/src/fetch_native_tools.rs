//! Downloads the pinned tools the native backend (crates/native) builds with into tools/native/ (git-ignored):
//!   glslang 16.6.0 (Linux x86_64 release binary), Vulkan-Headers vulkan-sdk-1.4.363.0 and volk vulkan-sdk-1.4.363.0.
//! Vulkan-Headers and volk are pinned to the same Vulkan SDK release (vulkan-sdk-* tags, not Khronos's weekly
//! spec-update tags), so the headers and the loader always come from one SDK.
//! The network's own source is in the repository (third_party/opendlss-nr). Every archive is checked against
//! the SHA-256 pinned below before it is unpacked. Re-running skips a tool whose unpacked copy carries the
//! stamp of the same pinned archive, so it is safe to run on every build, and a filled tools/native/ builds
//! offline. cargo xtask check-pins reads the pins below, and its weekly run (.github/workflows/pin-check.yml)
//! rewrites them when upstream moves on.
//!
//!   cargo xtask fetch-native-tools            (NEURAL_FORGE_NATIVE_TOOLS=<dir> to unpack elsewhere)
//!
//! It replaced scripts/fetch-native-tools.sh, line for line: the same output, files and exit codes (curl's and tar's own
//! when one of them fails, 1 for a mismatch).

use std::process::{Command, ExitCode, Stdio};

use crate::fetch;
use crate::sh::{self, Fail, Res};

/// name | url | sha256 | file that must exist after unpacking (relative to tools/native/<name>)
const PINS: &[&str] = &[
    "glslang|https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-linux-x86_64-release.tar.gz|2c34071f56ecf39d16233294d3739cf43f981fd39080f2fd9dccb782101191ce|bin/glslang",
    "Vulkan-Headers|https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz|4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e|include/vulkan/vulkan_core.h",
    "volk|https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz|1547d8d74395d4048fb3f4a6313da56db626e89b55db238d0d7e8944c8a645f3|volk.c",
];

fn run() -> Res<ExitCode> {
    sh::trap_signals();
    let tools = fetch::tools_dir();
    fetch::create_dir_all(&tools)?;
    for pin in PINS {
        let [name, url, sha, check] = pin.split('|').collect::<Vec<_>>()[..] else { return Err(Fail::new(1, format!("bad pin: {pin}"))) };
        let dest = tools.join(name);
        if dest.join(check).exists() && fetch::stamped(&dest, sha) {
            println!("{name}: up to date ({})", dest.display());
            continue;
        }
        let stage = fetch::stage(&tools, name)?;
        let archive = fetch::download(name, url, sha, stage.path())?;
        let mut top = stage.path().join("x");
        fetch::create_dir_all(&top)?;
        fetch::untar(&archive, &top)?;
        // GitHub tag tarballs have one top-level directory; the glslang release archive does not.
        if !top.join(check).exists() {
            if let [only] = &fetch::entries(&top)[..] {
                top = only.clone();
            }
        }
        if !top.join(check).exists() {
            return Err(Fail::new(1, format!("{name}: {check} not found in the archive")));
        }
        fetch::install(&top, sha, &dest)?;
        drop(stage);
        println!("{name}: verified and unpacked into {}", dest.display());
    }

    // The first line of glslang's version, as `glslang --version | head -n 1` gave it.
    let glslang = tools.join("glslang/bin/glslang");
    let mut version = Command::new(&glslang);
    let out = version.arg("--version").stdin(Stdio::null()).stderr(Stdio::inherit()).output().map_err(|e| sh::not_started(&version, e))?;
    if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
        println!("{line}");
    }
    if !out.status.success() {
        return Err(Fail::silent(sh::code(out.status)));
    }
    let header = tools.join("Vulkan-Headers/include/vulkan/vulkan_core.h");
    let text = std::fs::read(&header).map_err(|e| Fail::new(2, format!("{}: {e}", header.display())))?;
    let text = String::from_utf8_lossy(&text);
    let Some(line) = text.lines().find(|l| l.contains("#define VK_HEADER_VERSION ")) else { return Err(Fail::silent(1)) };
    println!("{line}");
    println!("native tools ready in {}", tools.display());
    Ok(ExitCode::SUCCESS)
}

pub fn main(_args: &[String]) -> ExitCode {
    sh::finish(run())
}
