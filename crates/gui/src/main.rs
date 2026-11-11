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

    let app = libadwaita::Application::builder().application_id(APP_ID).flags(gio::ApplicationFlags::FLAGS_NONE).build();

    // Keep the persistent layer copy (what Steam games load) in step with this
    // AppImage on every launch, so an updated AppImage never runs against a stale layer.
    let install_error = auto_install();
    // Rings the NGX wrapper left for games that are gone (the layer never opened them): never a running game's.
    neural_forge_supervisor::proton_tool::sweep_dead_rings(std::path::Path::new(neural_forge_supervisor::proton_tool::RING_DIR));

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
/// differ from what is installed (a no-op otherwise). Only when this program runs from the AppDir
/// that `APPDIR` names (`install::running_appdir`): the AppImage runtime sets it, so a `cargo run`
/// build never installs, and an `APPDIR` inherited from another AppImage is not ours. Returns the error
/// to show, if any.
fn auto_install() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let appdir = neural_forge_supervisor::install::running_appdir(std::env::var_os("APPDIR").as_deref(), &exe)?;
    match neural_forge_supervisor::install::install(&appdir) {
        Ok(report) => {
            if report.changed {
                eprintln!("neural-forge: installed the layer to {}", report.root.display());
            }
            // Every Neural Forge Proton gets the wrapper just installed (a no-op when they have it).
            use neural_forge_supervisor::{proton_tool, steam};
            for root in steam::roots().into_iter().filter(|r| !r.flatpak) {
                for (name, done) in proton_tool::refresh_wrappers(&root.path, &proton_tool::installed_wrapper()) {
                    match done {
                        Ok(true) => eprintln!("neural-forge: {name} has the new NGX wrapper"),
                        Ok(false) => {}
                        Err(e) => eprintln!("neural-forge: {name}: the new NGX wrapper could not be put in: {e}"),
                    }
                }
            }
            None
        }
        Err(e) => {
            eprintln!("neural-forge: installing the layer for Steam games failed: {e}");
            Some(format!("Installing the layer for Steam games failed: {e}"))
        }
    }
}
