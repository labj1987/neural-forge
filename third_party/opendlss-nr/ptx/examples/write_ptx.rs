//! Writes every kernel's PTX to a directory (`<stem>.ptx`), to diff the generated code before and after a change:
//! `cargo run -p opendlss-nr-ptx --example write_ptx DIR [GENERATOR...]` (only those generators' kernels when named;
//! a generator that matches no kernel is an error).

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("usage: write_ptx DIR [GENERATOR...]");
        std::process::exit(2);
    };
    let dir = std::path::PathBuf::from(dir);
    let only: Vec<String> = args.collect();
    let kernels = opendlss_nr_ptx::kernels();
    for name in &only {
        if !kernels.iter().any(|k| k.generator == name) {
            eprintln!("write_ptx: no kernel is made by a generator named {name:?}");
            std::process::exit(1);
        }
    }
    std::fs::create_dir_all(&dir).expect("creating the output directory");
    for kernel in kernels {
        if only.is_empty() || only.iter().any(|name| kernel.generator == name) {
            std::fs::write(dir.join(format!("{}.ptx", kernel.stem)), kernel.generate()).expect("writing a kernel");
        }
    }
}
