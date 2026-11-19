//! Every kernel the native backend embeds generates: unique names, each a PTX module with its entry point.

#[test]
fn every_kernel_generates() {
    let kernels = opendlss_nr_ptx::kernels();
    assert_eq!(kernels.len(), 76, "crates/native's every_kernel_is_embedded counts them too");
    let mut stems: Vec<&str> = kernels.iter().map(|k| k.stem.as_str()).collect();
    stems.sort_unstable();
    stems.dedup();
    assert_eq!(stems.len(), kernels.len(), "duplicate kernel names");
    for kernel in &kernels {
        let ptx = kernel.generate();
        assert!(ptx.starts_with(".version 8.7\n.target sm_89\n.address_size 64\n"), "{}: no PTX header", kernel.stem);
        assert!(ptx.contains("\n.visible .entry ") && ptx.ends_with("\n}\n"), "{}: no entry point", kernel.stem);
    }
}

/// SHA-256 over every kernel's stem and PTX bytes, in `kernels()` order. The generated code must not change by
/// accident: when a kernel changes on purpose, first diff the output against the old one
/// (`cargo run -p opendlss-nr-ptx --example write_ptx DIR` before and after, then `diff -r`), then update this hash.
const OUTPUT_SHA256: &str = "9cf96ddc545c5cc81414c2ce91c5c830db3e069688b5f4bac5c055ec086e29de";

#[test]
fn generated_output_is_pinned() {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for kernel in opendlss_nr_ptx::kernels() {
        hasher.update(kernel.stem.as_bytes());
        hasher.update(kernel.generate().as_bytes());
    }
    let hash: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hash, OUTPUT_SHA256, "the generated PTX changed: diff it with write_ptx first, then update OUTPUT_SHA256 on purpose");
}
