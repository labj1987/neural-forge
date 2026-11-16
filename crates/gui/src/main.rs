mod activation;
mod instance;
mod keys;
mod pages;
mod shm;
mod text;
mod ui;

/// The Wayland app_id and X11 window class. It matches `StartupWMClass` in the .desktop file, so the running window
/// is tied to the desktop file (name, icon) on either backend.
pub(crate) const APP_ID: &str = "io.github.labj1987.NeuralForge";

/// The UI toolkit reports renderer and display-server problems through the `log` crate; they go to stderr so a
/// terminal or the journal shows them.
struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("neural-forge: {} [{}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

fn main() {
    // Only the first copy goes on: a second launch has asked the running one to show its window.
    let Some(activations) = instance::claim() else {
        return;
    };
    log::set_logger(&StderrLog).ok();
    // NEURAL_FORGE_GUI_LOG=info (or debug) also shows which renderer was picked.
    let level = neural_forge_protocol::env::var("NEURAL_FORGE_GUI_LOG").and_then(|level| level.parse().ok()).unwrap_or(log::LevelFilter::Warn);
    log::set_max_level(level);

    // Keep the persistent layer copy (what Steam games load) in step with this
    // AppImage on every launch, so an updated AppImage never runs against a stale layer.
    let install_error = auto_install();
    // Rings the NGX wrapper left for games that are gone (the layer never opened them): never a running game's.
    neural_forge_supervisor::proton_tool::sweep_dead_rings(std::path::Path::new(neural_forge_supervisor::proton_tool::RING_DIR));

    gpui_kit::application().with_assets(ui::Assets).run(move |cx| {
        ui::init(cx);
        ui::build_ui(cx, activations, install_error);
    });
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
