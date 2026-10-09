mod shm;
mod ui;

use gtk4::prelude::*;

const APP_ID: &str = "io.github.labj1987.NeuralForge";

fn main() {
    // Set program name before GTK init. On Wayland the app_id GNOME sees is the
    // GApplication ID, not prgname; on X11 it's prgname. Setting both prgname and
    // StartupWMClass (in the .desktop file) to the application ID makes the running
    // window match the desktop file on either backend.
    glib::set_prgname(Some(APP_ID));
    glib::set_application_name("Neural Forge");

    let app = libadwaita::Application::builder()
        .application_id(APP_ID)
        .flags(gio::ApplicationFlags::FLAGS_NONE)
        .build();

    // Keep the persistent layer copy (what Steam games load) in step with this
    // AppImage on every launch, so an updated AppImage never runs against a stale layer.
    let install_error = auto_install();

    app.connect_activate(move |app| {
        if let Some(window) = app.windows().first() {
            window.present();
            return;
        }
        ui::build_ui(app, install_error.clone());
    });

    std::process::exit(app.run().get() as i32);
}

/// Installs this AppImage's layer into persistent user storage when they
/// differ from what is installed (a no-op otherwise). `APPDIR` is set by the AppImage
/// runtime only, so a `cargo run` build never installs. Returns the error to show, if any.
fn auto_install() -> Option<String> {
    let appdir = std::env::var("APPDIR").ok()?;
    match neural_forge_supervisor::install::install(std::path::Path::new(&appdir)) {
        Ok(report) => {
            if report.changed {
                eprintln!("neural-forge: installed the layer to {}", report.root.display());
            }
            None
        }
        Err(e) => {
            eprintln!("neural-forge: installing the layer for Steam games failed: {e}");
            Some(format!("Installing the layer for Steam games failed: {e}"))
        }
    }
}
