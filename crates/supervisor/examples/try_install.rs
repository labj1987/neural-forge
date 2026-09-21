//! Manually exercises `neural_forge_supervisor::install::install` against a real,
//! already-built AppDir (`build-appimage.sh`'s own output) instead of the unit
//! tests' small hand-built fixture -- run with `XDG_DATA_HOME` pointed at a scratch
//! directory, never the real one, unless you actually mean to install:
//!
//! ```bash
//! XDG_DATA_HOME=/tmp/neuralforge-install-try cargo run -p neural-forge-supervisor \
//!     --example try_install -- build-appimage/AppDir
//! ```

fn main() {
    let appdir = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: try_install <appdir>  (e.g. build-appimage/AppDir)");
        std::process::exit(2);
    });
    match neural_forge_supervisor::install::install(std::path::Path::new(&appdir)) {
        Ok(report) => println!("OK root={} gui={} cli={}", report.root.display(), report.gui_path.display(), report.cli_path.display()),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }
}
