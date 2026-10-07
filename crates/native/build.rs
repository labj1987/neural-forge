//! Builds the native backend: compiles OpenDLSS-NR's GLSL kernels (glslang) and generates its PTX kernels
//! (the upstream Python generators), embeds both into a generated C++ source, and compiles that with the
//! vendored host code, volk (as C++, in namespace volk) and `cpp/*.cpp` into a static library. libstdc++ is linked statically so the
//! layer does not depend on the C++ runtime of whatever container a game runs in.
//!
//! Tools come from `NEURAL_FORGE_NATIVE_TOOLS` or `<workspace>/tools/native` (`scripts/fetch-native-tools.sh`);
//! Python from `NEURAL_FORGE_PYTHON` or `python3` on PATH.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// One PTX generator run: the script in `scripts/ptx`, its arguments, and the output's file stem, which takes
/// the place of `@` in the arguments. Same invocations, in the same order, as upstream's build_shaders.ps1.
fn ptx_invocations() -> Vec<(&'static str, String, Vec<String>)> {
    let mut v = Vec::new();
    let mut add = |script: &'static str, stem: String, args: &[&str]| {
        v.push((script, stem, args.iter().map(|a| a.to_string()).collect::<Vec<_>>()));
    };
    for k in ["64", "128", "256"] {
        add("mlp_e4m3.py", format!("mlp_e4m3_K{k}"), &[k, "@", "3", "1", "1"]);
    }
    for c in ["64", "128", "256", "512"] {
        add("qkv_e4m3.py", format!("qkv_e4m3_K{c}"), &[c, "@", "80"]);
    }
    for k in ["32", "64", "128", "256", "512", "1024"] {
        for f in ["5", "13", "4", "8", "6"] {
            add("gemm2_e4m3.py", format!("gemm2_e4m3_K{k}_f{f}"), &[k, f, "@"]);
        }
    }
    for [c, r, x] in [["64", "4", "0"], ["128", "4", "0"], ["256", "3", "80"], ["256", "4", "64"]] {
        add("ffn_e4m3.py", format!("ffn_e4m3_C{c}_R{r}"), &[c, r, "@", x]);
        add("ffn_e4m3.py", format!("ffn_e4m3_C{c}_R{r}_proj"), &[c, r, "@", x, "1"]);
    }
    for [k, f, p, s] in [["4096", "1", "32", "4"], ["1024", "0", "16", "2"], ["1024", "1", "8", "4"], ["1024", "0", "8", "4"]] {
        add("gemmt_e4m3.py", format!("gemmt_e4m3_K{k}_f{f}_p{p}_s{s}"), &[k, f, p, s, "@"]);
    }
    for [s, f] in [["4", "4"], ["2", "8"], ["4", "8"]] {
        add("reduce_e4m3.py", format!("reduce_e4m3_s{s}_f{f}"), &[s, f, "@"]);
    }
    for [k, f, s] in [["1024", "6", "1"], ["4096", "5", "4"], ["1024", "8", "2"], ["1024", "5", "4"]] {
        add("gemmv_e4m3.py", format!("gemmv_e4m3_K{k}_f{f}_s{s}"), &[k, f, s, "@"]);
        add("gemmv_e4m3.py", format!("gemmv_e4m3_K{k}_f{f}_s{s}_m96"), &[k, f, s, "@", "4", "6", "0", "0", "0", "96"]);
    }
    for padded in ["64", "128", "192", "256"] {
        add("global_attention_e4m3.py", format!("global_attention_e4m3_p{padded}"), &[padded, "@"]);
    }
    add("global_attention_stream_e4m3.py", "global_normalize_e4m3".into(), &["normalize", "@"]);
    add("global_attention_stream_e4m3.py", "global_attention_stream_e4m3".into(), &["attention", "@", "4"]);
    for [k, f] in [["512", "4"], ["512", "5"], ["512", "13"], ["256", "5"], ["4096", "5"], ["1024", "8"], ["1024", "5"]] {
        add("gemmv_e4m3.py", format!("gemmv_e4m3_K{k}_f{f}_s1_m96"), &[k, f, "1", "@", "4", "6", "0", "0", "0", "96"]);
    }
    for f in ["2", "66", "74", "130", "48"] {
        add("block32_e4m3.py", format!("block32_e4m3_f{f}"), &[f, "@"]);
    }
    v
}

fn files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    files.sort();
    files
}

fn rerun_if_changed(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

/// Runs every command on all cores; panics with the output of the first failure.
fn run_all(jobs: Vec<(String, Command)>) {
    let jobs: Vec<Mutex<(String, Command)>> = jobs.into_iter().map(Mutex::new).collect();
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(i) else { break };
                let mut job = job.lock().unwrap();
                let (label, command) = &mut *job;
                match command.output() {
                    Ok(out) if out.status.success() => {}
                    Ok(out) => failures.lock().unwrap().push(format!(
                        "{label} failed ({}):\n{}{}",
                        out.status,
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    )),
                    Err(e) => failures.lock().unwrap().push(format!("{label}: cannot run {command:?}: {e}")),
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    if let Some(first) = failures.first() {
        panic!("{} kernel build step(s) failed; the first:\n{first}", failures.len());
    }
}

/// An absolute path as a string an assembler `.incbin "..."` inside a C++ string literal can take.
fn asm_path(path: &Path) -> String {
    let s = path.to_str().expect("OUT_DIR is not UTF-8");
    assert!(!s.contains(['"', '\\', '\n']), "unsupported character in {s}");
    s.to_string()
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().and_then(Path::parent).expect("crates/native is two levels below the workspace").to_path_buf();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vendor = root.join("third_party/opendlss-nr");
    let shaders = vendor.join("shaders");
    let ptx_scripts = vendor.join("scripts/ptx");
    let cpp = manifest.join("cpp");

    println!("cargo:rerun-if-env-changed=NEURAL_FORGE_NATIVE_TOOLS");
    println!("cargo:rerun-if-env-changed=NEURAL_FORGE_PYTHON");
    let tools = env::var_os("NEURAL_FORGE_NATIVE_TOOLS").map(PathBuf::from).unwrap_or_else(|| root.join("tools/native"));
    let glslang = tools.join("glslang/bin/glslang");
    let headers = tools.join("Vulkan-Headers/include");
    let volk = tools.join("volk");
    for need in [&glslang, &headers.join("vulkan/vulkan_core.h"), &volk.join("volk.c")] {
        if !need.exists() {
            panic!(
                "\n\nneural-forge-native: {} is missing.\nRun `bash scripts/fetch-native-tools.sh` once in the workspace root \
                 (or point NEURAL_FORGE_NATIVE_TOOLS at a directory it filled).\n\n",
                need.display()
            );
        }
    }
    for stamp in ["glslang", "Vulkan-Headers", "volk"] {
        rerun_if_changed(&tools.join(stamp).join(".fetched-sha256"));
    }
    let python = env::var_os("NEURAL_FORGE_PYTHON").unwrap_or_else(|| "python3".into());

    // Kernels: SPIR-V with the same glslang flags as upstream, PTX with the same generator invocations.
    let spv_dir = out.join("spv");
    let ptx_dir = out.join("ptx");
    fs::create_dir_all(&spv_dir).unwrap();
    fs::create_dir_all(&ptx_dir).unwrap();
    rerun_if_changed(&shaders);
    rerun_if_changed(&ptx_scripts);
    rerun_if_changed(&manifest.join("run_ptx.py"));
    let mut assets: Vec<(&str, String, PathBuf)> = Vec::new();
    let mut jobs = Vec::new();
    for file in files_with_extension(&shaders, "glsl") {
        rerun_if_changed(&file);
    }
    for comp in files_with_extension(&shaders, "comp") {
        rerun_if_changed(&comp);
        let name = comp.file_stem().unwrap().to_str().unwrap().to_string();
        let spv = spv_dir.join(format!("{name}.spv"));
        let mut cmd = Command::new(&glslang);
        cmd.args(["-V", "--target-env", "vulkan1.3"]).arg(format!("-I{}", shaders.display())).arg(&comp).arg("-o").arg(&spv);
        jobs.push((format!("glslang {name}.comp"), cmd));
        assets.push(("spv", name, spv));
    }
    for script in files_with_extension(&ptx_scripts, "py") {
        rerun_if_changed(&script);
    }
    for (script, stem, args) in ptx_invocations() {
        let ptx = ptx_dir.join(format!("{stem}.ptx"));
        let mut cmd = Command::new(&python);
        cmd.arg("-I").arg(manifest.join("run_ptx.py")).arg(ptx_scripts.join(script));
        for a in &args {
            if a == "@" { cmd.arg(&ptx); } else { cmd.arg(a); }
        }
        jobs.push((format!("{script} {}", args.join(" ")), cmd));
        assets.push(("ptx", stem, ptx));
    }
    run_all(jobs);

    // Embedding: .incbin each file into .rodata, plus a table nf_native::findAsset searches.
    let mut src = String::from(
        "// Generated by crates/native/build.rs: the embedded OpenDLSS-NR kernels. Do not edit.\n\
         #include <cstring>\n\n#include \"nf_assets.h\"\n#include \"vk_context.h\"\n\n__asm__(\n\
         \"  .pushsection .rodata.nf_native_assets,\\\"a\\\",@progbits\\n\"\n",
    );
    for (i, (_, _, path)) in assets.iter().enumerate() {
        assert!(fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false), "{} was not generated", path.display());
        writeln!(
            src,
            "\"  .balign 16\\n  .globl nf_native_asset_{i}\\n  .hidden nf_native_asset_{i}\\nnf_native_asset_{i}:\\n\"\n\
             \"  .incbin \\\"{}\\\"\\n  .globl nf_native_asset_{i}_end\\n  .hidden nf_native_asset_{i}_end\\nnf_native_asset_{i}_end:\\n\"",
            asm_path(path)
        )
        .unwrap();
    }
    src.push_str("\"  .popsection\\n\");\n\nextern \"C\" {\n");
    for i in 0..assets.len() {
        writeln!(
            src,
            "__attribute__((visibility(\"hidden\"))) extern const uint8_t nf_native_asset_{i}[], nf_native_asset_{i}_end[];"
        )
        .unwrap();
    }
    src.push_str(
        "}\n\nnamespace nf_native {\nnamespace {\nstruct Asset { const char* kind; const char* name; const uint8_t* begin; const uint8_t* end; };\n\
         const Asset kAssets[] = {\n",
    );
    for (i, (kind, name, _)) in assets.iter().enumerate() {
        writeln!(src, "  {{\"{kind}\", \"{name}\", nf_native_asset_{i}, nf_native_asset_{i}_end}},").unwrap();
    }
    src.push_str(
        "};\n}  // namespace\n\n\
         bool findAsset(const char* kind, const char* name, const uint8_t** data, size_t* size) {\n\
         \x20 for (const Asset& a : kAssets)\n\
         \x20   if (!strcmp(a.kind, kind) && !strcmp(a.name, name)) { *data = a.begin; *size = (size_t)(a.end - a.begin); return true; }\n\
         \x20 return false;\n}\n\n\
         uint32_t assetCount(const char* kind) {\n\
         \x20 uint32_t n = 0;\n  for (const Asset& a : kAssets) n += !strcmp(a.kind, kind);\n  return n;\n}\n\n\
         void installAssetLoader() { nr::setAssetLoader(&findAsset); }\n\n}  // namespace nf_native\n",
    );
    let embed = out.join("nf_assets.cpp");
    if fs::read_to_string(&embed).ok().as_deref() != Some(src.as_str()) {
        fs::write(&embed, &src).unwrap();
    }

    // C++: the vendored host code, the embedded kernels and the glue in cpp/.
    let vendor_src = vendor.join("src");
    rerun_if_changed(&vendor_src);
    rerun_if_changed(&cpp);
    let mut sources = files_with_extension(&vendor_src, "cpp");
    sources.extend(files_with_extension(&cpp, "cpp"));
    for header in files_with_extension(&vendor_src, "h").into_iter().chain(files_with_extension(&cpp, "h")) {
        rerun_if_changed(&header);
    }
    for s in &sources {
        rerun_if_changed(s);
    }
    sources.push(embed);
    // volk is compiled as C++ in namespace volk (VOLK_NAMESPACE): as C its function-pointer globals are named
    // vkGetInstanceProcAddr and so on, the same as the layer's own exported entry points, and the link fails
    // with duplicate symbols.
    rerun_if_changed(&volk.join("volk.c"));
    rerun_if_changed(&volk.join("volk.h"));
    sources.push(volk.join("volk.c"));
    let mut cxx = cc::Build::new();
    cxx.cpp(true)
        .std("c++20")
        .opt_level(2)
        .pic(true)
        .flag("-fvisibility=hidden")
        .flag("-fvisibility-inlines-hidden")
        .define("VK_NO_PROTOTYPES", None)
        .define("VK_ENABLE_BETA_EXTENSIONS", None)
        .define("VOLK_NAMESPACE", None)
        .include(&headers)
        .include(&volk)
        .include(&vendor_src)
        .include(&cpp)
        .warnings(false)
        .cpp_link_stdlib(None)
        .files(&sources);
    cxx.compile("neural_forge_native_cpp");

    // libstdc++ statically, from the compiler that built the code above. Not bundled into the rlib: the final
    // link (the layer cdylib, or a test binary) takes it from the search path, and a cdylib's version script
    // keeps all of it local.
    let compiler = cxx.get_compiler();
    let found = Command::new(compiler.path()).arg("-print-file-name=libstdc++.a").output().expect("running the C++ compiler");
    let archive = PathBuf::from(String::from_utf8(found.stdout).unwrap().trim());
    if !archive.is_absolute() || !archive.exists() {
        panic!("neural-forge-native: libstdc++.a not found by {} (install the static libstdc++, e.g. libstdc++-dev)", compiler.path().display());
    }
    println!("cargo:rustc-link-search=native={}", archive.parent().unwrap().display());
    println!("cargo:rustc-link-lib=static:-bundle=stdc++");
}
