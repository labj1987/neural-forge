//! Compiles the GUI's own symbolic icons (`icons/`, from the Adwaita icon theme) into a GResource bundle that
//! `main.rs` registers before GTK starts, so the tabs and buttons look the same whatever icon theme the desktop uses.
//! Needs `glib-compile-resources` (GLib's development tools, which GTK's development package pulls in).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let icons = dir.join("icons");
    let sources = icons.join("scalable/actions");
    let xml = icons.join("icons.gresource.xml");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("icons.gresource");
    println!("cargo:rerun-if-changed={}", xml.display());
    println!("cargo:rerun-if-changed={}", sources.display());
    let status = Command::new("glib-compile-resources")
        .arg("--sourcedir")
        .arg(&sources)
        .arg("--target")
        .arg(&out)
        .arg(&xml)
        .status()
        .unwrap_or_else(|e| panic!("glib-compile-resources could not be run ({e}): install GLib's development tools (libglib2.0-dev-bin)"));
    assert!(status.success(), "glib-compile-resources failed on {}: {status}", xml.display());
}
