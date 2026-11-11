//! Windows builds export exactly NVIDIA's `_nvngx.dll` names with its ordinals (`nvngx.def`), nothing else
//! (not `DllMain`).
fn main() {
    println!("cargo:rerun-if-changed=nvngx.def");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let def = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("nvngx.def");
        println!("cargo:rustc-cdylib-link-arg={}", def.display());
    }
}
