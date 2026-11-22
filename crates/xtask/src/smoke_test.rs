//! `cargo xtask smoke-test`: builds the layer, points a scratch Vulkan layer manifest at it, and runs
//! crates/layer/examples/smoke.rs through the real system Vulkan loader with the layer
//! enabled. See that file's doc comment for what this actually proves.
//!
//! NEURAL_FORGE_SMOKE_SO names a layer library to test instead of building the debug one (the
//! release workflow points it at the library the AppImage installed). NEURAL_FORGE_SMOKE_INSTALLED_DATA names
//! that install's XDG_DATA_HOME: a last pass then finds the layer only through the implicit manifest the install
//! wrote there, as a game does.
//!
//! It replaced scripts/smoke-test.sh: the same passes, output and exit codes (the example's own when it fails).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use crate::sh::{self, Fail, Res};

fn contains(log: &[u8], needle: &str) -> bool {
    log.windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn fail(message: &str) -> Fail {
    Fail::new(1, format!("FAIL: {message}"))
}

/// One run of the smoke example with the environment as it is now; prints the layer's log and returns it (empty when
/// none was written).
fn pass() -> Res<Vec<u8>> {
    sh::check(Command::new("cargo").args(["run", "--locked", "--example", "smoke", "-p", "neural-forge-layer"]))?;
    let log = std::env::var("NEURAL_FORGE_LOG").unwrap_or_default();
    println!();
    println!("==> layer log ({log}):");
    match std::fs::read(&log) {
        Ok(bytes) => {
            let _ = std::io::stdout().write_all(&bytes);
            Ok(bytes)
        }
        Err(_) => {
            println!("(no log written -- the layer never ran)");
            Ok(Vec::new())
        }
    }
}

/// data/neural_forge_layer.json with its library_path set to `so`, written as JSON, so any character the path holds
/// stays the path.
fn manifest(root: &Path, so: &Path) -> Res<String> {
    const SOURCE: &str = "data/neural_forge_layer.json";
    let text = std::fs::read_to_string(root.join(SOURCE)).map_err(|e| Fail::new(2, format!("{SOURCE}: {e}")))?;
    let mut json: serde_json::Value = serde_json::from_str(&text).map_err(|e| Fail::new(2, format!("{SOURCE}: {e}")))?;
    let path = &mut json["layer"]["library_path"];
    if path != "./libneural_forge_layer.so" {
        return Err(Fail::new(5, format!("{SOURCE}: library_path is {}", path.as_str().unwrap_or("not a string"))));
    }
    *path = serde_json::Value::String(so.to_string_lossy().into_owned());
    serde_json::to_string_pretty(&json).map(|text| text + "\n").map_err(|e| Fail::new(2, format!("{SOURCE}: {e}")))
}

fn run() -> Res<ExitCode> {
    sh::trap_signals();
    let root = crate::root();
    std::env::set_current_dir(&root).map_err(|e| Fail::new(1, format!("{}: {e}", root.display())))?;

    let installed_data = sh::var("NEURAL_FORGE_SMOKE_INSTALLED_DATA");
    let so = match sh::var("NEURAL_FORGE_SMOKE_SO") {
        Some(so) => PathBuf::from(so),
        None => {
            sh::check(Command::new("cargo").args(["build", "--locked", "-p", "neural-forge-layer"]))?;
            sh::var("CARGO_TARGET_DIR").map_or_else(|| root.join("target"), PathBuf::from).join("debug/libneural_forge_layer.so")
        }
    };
    if !so.is_file() {
        return Err(Fail::silent(1));
    }
    // Absolute: the loader resolves a relative library_path against the manifest's directory (the scratch
    // dir below), not this one, and CARGO_TARGET_DIR may be relative.
    let so = std::fs::canonicalize(&so).map_err(|e| Fail::new(1, format!("{}: {e}", so.display())))?;

    let scratch = sh::mktemp()?;
    let scratch_path = scratch.path().to_string_lossy().into_owned();
    let manifest_path = scratch.path().join("neural_forge_layer.json");
    std::fs::write(&manifest_path, manifest(&root, &so)?).map_err(|e| Fail::new(1, format!("{}: {e}", manifest_path.display())))?;

    // VK_LAYER_PATH only makes the loader consider a manifest for *explicit* enabling --
    // it does not add it to the implicit_layer.d search path, so an implicit-type layer
    // found this way still needs to be named explicitly to actually get inserted into the
    // call chain. That's a test-harness-only difference: a real install drops this manifest
    // into an actual implicit_layer.d directory instead, where enable_environment alone is
    // enough (see data/neural_forge_layer.json and build-appimage.sh).
    std::env::set_var("VK_LAYER_PATH", &scratch_path);
    // An installed copy's implicit manifest (~/.local/share/vulkan/implicit_layer.d) has the same
    // layer name and wins over VK_LAYER_PATH, so on a machine with Neural Forge installed the
    // smoke test would otherwise exercise the installed library, not this build. Point the
    // per-user data dir at the scratch dir so only the manifest above is found.
    std::env::set_var("XDG_DATA_HOME", format!("{scratch_path}/xdg-data"));
    std::env::set_var("VK_INSTANCE_LAYERS", "VK_LAYER_neuralforge_neural");
    std::env::set_var("NEURAL_FORGE_ENABLE", "1");
    std::env::set_var("NEURAL_FORGE_UID", format!("smoketest-{}", std::process::id()));
    std::env::set_var("NEURAL_FORGE_LOG", format!("{scratch_path}/layer.log"));

    println!("==> manifest: {}", manifest_path.display());
    println!("==> layer library: {}", so.display());
    println!();

    let log = pass()?;
    if contains(&log, "[probe-ngx]") {
        return Err(fail("the NGX probe logged without NEURAL_FORGE_PROBE_NGX"));
    }
    // The default is the pre-upscaler path in model mode (docs/PRE_UPSCALER_DESIGN.md): the local ICD
    // has no VK_NVX_* extensions and no DLSS, so it must load cleanly, give the device no tracking, and
    // hold nothing.
    if !contains(&log, "[preupscale] mode model (default) in pid") {
        return Err(fail("with NEURAL_FORGE_PREUPSCALE unset the pre-upscaler path is not the default"));
    }
    if !contains(&log, "no VK_NVX_image_view_handle, so no DLSS input to find") {
        return Err(fail("a device without NVX did not stay on the post-upscaler path"));
    }
    if ["frame went to DLSS untouched", "resources for", "HDR encode"].iter().any(|s| contains(&log, s)) {
        return Err(fail("the default held a submit on a device without DLSS"));
    }

    // Pass with NEURAL_FORGE_PREUPSCALE=off: the A/B and rollback switch, 1.1.0's behaviour. Nothing of
    // the pre-upscaler path may run or log.
    std::env::set_var("NEURAL_FORGE_PREUPSCALE", "off");
    std::env::set_var("NEURAL_FORGE_LOG", format!("{scratch_path}/layer-off.log"));
    println!();
    println!("==> again with NEURAL_FORGE_PREUPSCALE=off");
    let log = pass()?;
    if !contains(&log, "[layer] hooked device") {
        return Err(fail("the layer did not run with NEURAL_FORGE_PREUPSCALE=off"));
    }
    if contains(&log, "[preupscale]") {
        return Err(fail("the pre-upscaler path ran with NEURAL_FORGE_PREUPSCALE=off"));
    }
    std::env::remove_var("NEURAL_FORGE_PREUPSCALE");

    // Second pass with the diagnostic NGX probe on (docs/PRE_UPSCALER_PROBE.md): the local ICD has
    // no VK_NVX_* extensions, so this proves the probe's hooks install and stay harmless there.
    std::env::set_var("NEURAL_FORGE_PROBE_NGX", "1");
    std::env::set_var("NEURAL_FORGE_LOG", format!("{scratch_path}/layer-probe.log"));
    println!();
    println!("==> again with NEURAL_FORGE_PROBE_NGX=1");
    let log = pass()?;
    if !contains(&log, "[probe-ngx] on in pid") {
        return Err(fail("NEURAL_FORGE_PROBE_NGX=1 did not turn the probe on"));
    }

    // Third pass with the pre-upscaler path's model mode named explicitly (docs/PRE_UPSCALER_DESIGN.md,
    // "Implementation (layer)"): the local ICD has no VK_NVX_* extensions and no DLSS, so this proves
    // the mode is taken, nothing is held, and the probe's logging stays off.
    std::env::remove_var("NEURAL_FORGE_PROBE_NGX");
    std::env::set_var("NEURAL_FORGE_PREUPSCALE", "model");
    std::env::set_var("NEURAL_FORGE_LOG", format!("{scratch_path}/layer-preupscale.log"));
    println!();
    println!("==> again with NEURAL_FORGE_PREUPSCALE=model");
    let log = pass()?;
    if !contains(&log, "[preupscale] mode model in pid") {
        return Err(fail("NEURAL_FORGE_PREUPSCALE=model did not turn the pre-upscaler path on"));
    }
    if contains(&log, "[probe-ngx]") {
        return Err(fail("the NGX probe logged with only NEURAL_FORGE_PREUPSCALE set"));
    }
    if ["frame went to DLSS untouched", "resources for"].iter().any(|s| contains(&log, s)) {
        return Err(fail("the pre-upscaler path held a submit on a device without DLSS"));
    }
    std::env::remove_var("NEURAL_FORGE_PREUPSCALE");

    // The installed implicit manifest alone: no VK_LAYER_PATH, no VK_INSTANCE_LAYERS, only the manifest's
    // enable_environment (NEURAL_FORGE_ENABLE=1), as a game started with the launch option finds it.
    if let Some(installed_data) = installed_data {
        std::env::remove_var("VK_LAYER_PATH");
        std::env::remove_var("VK_INSTANCE_LAYERS");
        std::env::set_var("XDG_DATA_HOME", &installed_data);
        std::env::set_var("NEURAL_FORGE_LOG", format!("{scratch_path}/layer-implicit.log"));
        println!();
        println!("==> again through the implicit manifest in {installed_data}/vulkan/implicit_layer.d only");
        let log = pass()?;
        if !contains(&log, "[layer] hooked device") {
            return Err(fail("the installed implicit manifest did not load the layer with NEURAL_FORGE_ENABLE=1"));
        }
    }
    Ok(ExitCode::SUCCESS)
}

pub fn main(_args: &[String]) -> ExitCode {
    sh::finish(run())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_manifest_names_the_library_whatever_its_path_holds() {
        let text = super::manifest(&crate::root(), std::path::Path::new("/tmp/a \"b\"/lib.so")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["layer"]["library_path"], "/tmp/a \"b\"/lib.so");
        assert_eq!(json["layer"]["name"], "VK_LAYER_neuralforge_neural");
    }
}
