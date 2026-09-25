//! Compiles the C setjmp shim `crate::guard` depends on (see `csrc/guard_shim.c`).
fn main() {
    println!("cargo:rerun-if-changed=csrc/guard_shim.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    cc::Build::new().file("csrc/guard_shim.c").warnings(true).compile("nf_guard_shim");
}
