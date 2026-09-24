mod binaries;
mod shm;
mod ui;

use gtk4::prelude::*;

fn main() {
    // Set program name before GTK init. On Wayland the app_id GNOME sees is the
    // GApplication ID, not prgname; on X11 it's prgname. Setting both prgname and
    // StartupWMClass (in the .desktop file) to the application ID makes the running
    // window match the desktop file on either backend.
    glib::set_prgname(Some("io.github.labj1987.NeuralForge"));
    glib::set_application_name("Neural Forge");

    let app = libadwaita::Application::builder()
        .application_id("io.github.labj1987.NeuralForge")
        .flags(gio::ApplicationFlags::FLAGS_NONE)
        .build();

    app.connect_activate(|app| {
        if let Some(window) = app.windows().first() {
            window.present();
            return;
        }
        ui::build_ui(app);
    });

    // The helper is launched as a detached Proton/Wine process tree. Shut it
    // down when the GUI exits so AppImage launchers such as Gear Lever do not
    // keep reporting the application as still running. This one is deliberately
    // synchronous: the window is already gone, and the process must not exit before
    // the helper has been stopped.
    app.connect_shutdown(|_| {
        if neural_forge_supervisor::is_running().is_some() {
            let _ = neural_forge_supervisor::stop(std::time::Duration::from_secs(5));
        }
    });

    std::process::exit(app.run().get() as i32);
}
