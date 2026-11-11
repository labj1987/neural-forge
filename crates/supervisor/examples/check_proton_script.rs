//! Runs Neural Forge's patch of Proton's `proton` script (`proton_tool::patch_proton_script`) on each file named
//! on the command line and prints one line per file: `ok <file>` or `fail <file>: <why>`. Exit 0 when every
//! file patched, 10 when one did not (the `NVIDIA_WINE_DLL_DIR` block it anchors on is missing or matched not
//! exactly once), 2 when a file could not be read. `scripts/check_proton_scripts.py` runs it weekly on the newest
//! upstream scripts (.github/workflows/proton-script-check.yml).

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: check_proton_script FILE...");
        std::process::exit(2);
    }
    let mut status = 0;
    for file in files {
        let text = match std::fs::read_to_string(&file) {
            Ok(t) => t,
            Err(e) => {
                println!("unreadable {file}: {e}");
                status = 2;
                continue;
            }
        };
        match neural_forge_supervisor::proton_tool::patch_proton_script(&text) {
            Ok(_) => println!("ok {file}"),
            Err(e) => {
                println!("fail {file}: {e}");
                if status == 0 {
                    status = 10;
                }
            }
        }
    }
    std::process::exit(status);
}
