mod activation;
mod instance;
mod keymap;
mod keytap;
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
        metadata.level() <= log::max_level() && (metadata.level() <= log::Level::Warn || !chatty(metadata.target()))
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("neural-forge: {} [{}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

/// The toolkit's dependencies that log every message they pass at info level (D-Bus handshakes byte by byte, each
/// traced span): from them only warnings and errors are shown, whatever `NEURAL_FORGE_GUI_LOG` asks for.
fn chatty(target: &str) -> bool {
    ["zbus", "tracing", "naga", "polling", "async_io", "calloop"].iter().any(|noisy| target == *noisy || target.starts_with(&format!("{noisy}::")))
}

fn main() {
    // The window is drawn with Vulkan, and Neural Forge is a Vulkan layer: its switch must never reach this process (a
    // terminal or launcher that exported it for a game), or the window would load the layer into itself. First thing,
    // while the process has one thread.
    // SAFETY: no other thread exists yet to read or write the environment.
    unsafe { std::env::remove_var("NEURAL_FORGE_ENABLE") };
    log::set_logger(&StderrLog).ok();
    // NEURAL_FORGE_GUI_LOG=info (or debug) also shows which renderer was picked.
    let level = neural_forge_protocol::env::var("NEURAL_FORGE_GUI_LOG").and_then(|level| level.parse().ok()).unwrap_or(log::LevelFilter::Warn);
    log::set_max_level(level);

    // Keep the persistent layer copy (what Steam games load) in step with this AppImage on every launch, so an updated
    // AppImage never runs against a stale layer. Before the single-instance check: a second launch from a newer AppImage
    // still installs its layer while an older window is open (the installer replaces files by rename, so the running
    // copy keeps the ones it has mapped).
    let install_error = auto_install();
    // Rings the NGX wrapper left for games that are gone (the layer never opened them): never a running game's.
    neural_forge_supervisor::proton_tool::sweep_dead_rings(std::path::Path::new(neural_forge_supervisor::proton_tool::RING_DIR));

    // Only the first copy goes on: a second launch has asked the running one to show its window.
    let activations = match instance::claim() {
        instance::Claim::Primary(activations) => activations,
        // Shown, still starting and about to show itself, or on its way out.
        instance::Claim::Shown | instance::Claim::Starting | instance::Claim::Gone => return,
        // Its window is stuck (it listens, but its UI does not answer): starting a second one beside it would have two
        // copies writing the settings, so say what to do instead.
        instance::Claim::NotResponding(pid) => {
            ui::say_without_window(&instance::not_responding_message(pid));
            std::process::exit(1);
        }
    };

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

#[cfg(test)]
mod tests {
    #[test]
    fn the_d_bus_crates_chatter_is_not_shown_at_info_level() {
        assert!(super::chatty("zbus::connection::handshake::common") && super::chatty("tracing::span") && super::chatty("zbus"));
        assert!(!super::chatty("gpui_wgpu::wgpu_context"), "which renderer was picked stays");
        assert!(!super::chatty("neural_forge::keytap") && !super::chatty("zbus_names"));
    }
}
