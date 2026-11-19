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
