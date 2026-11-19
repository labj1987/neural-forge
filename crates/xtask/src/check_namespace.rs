//! Check activation, desktop, packaging and private API namespace boundaries.
//!
//! Every check runs, and each failure is reported with what was expected; the task exits 1 after listing them all.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use regex::Regex;
use serde_json::{json, Value};

use crate::py::{repr, repr_str};

/// Every file under `dir`, recursively (symlinked directories not followed, as `Path.rglob`).
fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// `glob('*.[ch]*')`: a name with a dot followed by c or h, not hidden.
fn c_sources(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|e| e.flatten().map(|e| e.path()).collect::<Vec<PathBuf>>())
        .unwrap_or_default()
        .into_iter()
        .filter(|p: &PathBuf| {
            let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            !n.starts_with('.') && n.as_bytes().windows(2).any(|w| w[0] == b'.' && (w[1] == b'c' || w[1] == b'h'))
        })
        .collect();
    found.sort();
    found
}

pub fn failures(root: &Path) -> Vec<String> {
    let mut failures = Vec::new();
    let mut check = |ok: bool, message: String| {
        if !ok {
            failures.push(message);
        }
    };
    let text = |rel: &str| crate::py::universal_newlines(&std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}")));

    let manifest: Value = serde_json::from_str(&text("data/neural_forge_layer.json")).expect("data/neural_forge_layer.json is JSON");
    let manifest = &manifest["layer"];
    let field = |k: &str| manifest.get(k);
    check(
        field("enable_environment") == Some(&json!({"NEURAL_FORGE_ENABLE": "1"})),
        format!("data/neural_forge_layer.json: enable_environment is {}, not NEURAL_FORGE_ENABLE=1", repr(field("enable_environment"))),
    );
    check(
        field("disable_environment") == Some(&json!({"NEURAL_FORGE_DISABLE": "1"})),
        format!("data/neural_forge_layer.json: disable_environment is {}, not NEURAL_FORGE_DISABLE=1", repr(field("disable_environment"))),
    );
    check(
        field("name") == Some(&json!("VK_LAYER_neuralforge_neural")),
        format!("data/neural_forge_layer.json: the layer is named {}, not VK_LAYER_neuralforge_neural", repr(field("name"))),
    );
    // Python's str(): a string as itself, anything else as its repr.
    let name = match field("name") {
        Some(Value::String(s)) => s.clone(),
        other => repr(other),
    };
    check(text("crates/layer/src/lib.rs").contains(&name), format!("crates/layer/src/lib.rs does not name the manifest's layer {}", repr(field("name"))));
    check(
        field("library_path") == Some(&json!("./libneural_forge_layer.so")),
        format!("data/neural_forge_layer.json: library_path is {}, not ./libneural_forge_layer.so", repr(field("library_path"))),
    );
    let identity = "io.github.labj1987.NeuralForge";
    let appdata = text(&format!("data/{identity}.appdata.xml"));
    let meta = roxmltree::Document::parse(&appdata).expect("the appdata is XML");
    for (tag, want) in [("id", identity.to_string()), ("launchable", format!("{identity}.desktop"))] {
        let found = meta.root_element().children().find(|n| n.is_element() && n.tag_name().name() == tag);
        // ElementTree's `.text`: the text before the first child, None when there is none.
        let got = found.and_then(|n| n.first_child()).filter(|c| c.is_text()).and_then(|c| c.text());
        let shown = got.map_or("None".to_string(), repr_str);
        check(got == Some(want.as_str()), format!("data/{identity}.appdata.xml: <{tag}> is {shown}, not {}", repr_str(&want)));
    }
    check(text(&format!("data/{identity}.desktop")).contains("\nExec=neural-forge\n"), format!("data/{identity}.desktop has no Exec=neural-forge line"));

    // The pre-0.1.77 spelling (`NEURALFORGE_*`, `neuralforge` paths) is gone everywhere: Rust sources, scripts, data
    // files and the packaging script. Only the two frozen identifiers (and one history file name) keep the old
    // spelling. The history file keeps its name (it records the project before the rename).
    const FROZEN: [&str; 3] = ["VK_LAYER_neuralforge_neural", "io.github.labj1987.NeuralForge", "development-before-neuralforge.md"];
    let mut scanned = Vec::new();
    files(&root.join("crates"), &mut scanned);
    scanned.retain(|p| p.extension().is_some_and(|e| e == "rs"));
    for folder in ["scripts", "data"] {
        let mut found = Vec::new();
        files(&root.join(folder), &mut found);
        scanned.extend(found.into_iter().filter(|p| !p.extension().is_some_and(|e| e == "spv")));
    }
    scanned.push(root.join("build-appimage.sh"));
    // This file names the old spellings to look for them.
    let this = root.join("crates/xtask/src/check_namespace.rs");
    let old_spelling = Regex::new("neuralforge|NEURALFORGE").unwrap();
    let old_env = Regex::new(r#"(?:var|var_os|set_var)\("(?:DLSSNR_|VKLayer_DLSS5)\w*"#).unwrap();
    for path in scanned.iter().filter(|p| **p != this) {
        let Ok(bytes) = std::fs::read(path) else { continue };
        // Binary data.
        let Ok(code) = String::from_utf8(bytes) else { continue };
        let mut code = crate::py::universal_newlines(&code);
        for frozen in FROZEN {
            code = code.replace(frozen, "");
        }
        let rel = path.strip_prefix(root).unwrap_or(path).display().to_string();
        let old = old_spelling.find(&code).map(|m| m.as_str().to_string());
        check(old.is_none(), format!("{rel}: the old spelling {} (use neural-forge / NEURAL_FORGE_)", repr_str(old.as_deref().unwrap_or(""))));
        if path.extension().is_some_and(|e| e == "rs") {
            let env = old_env.find(&code).map(|m| m.as_str().to_string());
            check(env.is_none(), format!("{rel}: reads an old or upstream environment variable ({})", env.unwrap_or_default()));
            check(!code.contains(concat!("NEURAL", "FORGE.Color")), format!("{rel}: the old {}.Color name", concat!("NEURAL", "FORGE")));
        }
    }
    // The network's C++ runs inside the layer: it reads NEURAL_FORGE_ variables only, never upstream's DLSS5VK_ ones
    // (set for DLSS5VKLayer, they would change this layer's network).
    let getenv = Regex::new(r#"getenv\(\s*"(\w+)"|envFlag\(\s*"(\w+)""#).unwrap();
    for path in c_sources(&root.join("third_party/opendlss-nr/src")).into_iter().chain(c_sources(&root.join("crates/native/cpp"))) {
        let code = crate::py::universal_newlines(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())));
        for c in getenv.captures_iter(&code) {
            let name = format!("{}{}", c.get(1).map_or("", |m| m.as_str()), c.get(2).map_or("", |m| m.as_str()));
            let rel = path.strip_prefix(root).unwrap_or(&path).display().to_string();
            check(name.starts_with("NEURAL_FORGE_"), format!("{rel}: reads {name} (only NEURAL_FORGE_ variables)"));
        }
    }
    check(text("README.md").starts_with("# Neural Forge\n"), "README.md does not start with the '# Neural Forge' heading".into());
    check(!text("ATTRIBUTION.md").contains("`dlssnr` is a from-scratch"), "ATTRIBUTION.md still describes the project as dlssnr".into());
    check(
        text("build-appimage.sh").contains("|labj1987|neural-forge|latest|"),
        "build-appimage.sh's update information does not name labj1987/neural-forge".into(),
    );
    check(text("README.md").contains("https://github.com/labj1987/neural-forge"), "README.md does not link https://github.com/labj1987/neural-forge".into());
    failures
}

pub fn main(_args: &[String]) -> ExitCode {
    let failures = failures(&crate::root());
    if !failures.is_empty() {
        for message in &failures {
            eprintln!("error: {message}");
        }
        eprintln!("Namespace contract: {} failures", failures.len());
        return ExitCode::FAILURE;
    }
    println!("Namespace contract: OK");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_repository_passes() {
        assert_eq!(super::failures(&crate::root()), Vec::<String>::new());
    }
}
