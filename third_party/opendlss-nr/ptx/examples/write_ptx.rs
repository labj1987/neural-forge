//! Writes every kernel's PTX to a directory (`<stem>.ptx`), to diff the generated code before and after a change:
//! `cargo run -p opendlss-nr-ptx --example write_ptx DIR [GENERATOR...]` (only those generators' kernels when named).

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("usage: write_ptx DIR [GENERATOR...]"));
    let only: Vec<String> = args.collect();
    std::fs::create_dir_all(&dir).expect("creating the output directory");
    for kernel in opendlss_nr_ptx::kernels() {
        if only.is_empty() || only.contains(&kernel.generator.to_string()) {
            std::fs::write(dir.join(format!("{}.ptx", kernel.stem)), kernel.generate()).expect("writing a kernel");
        }
    }
}
