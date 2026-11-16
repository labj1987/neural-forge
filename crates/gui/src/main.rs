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
        shown(metadata.target(), metadata.level(), log::max_level())
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("neural-forge: {} [{}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

/// The most each source may show, whatever `NEURAL_FORGE_GUI_LOG` asks for (a module path and everything under it;
/// the longest match counts). The toolkit's dependencies that log every message they pass at info level (D-Bus
/// handshakes byte by byte, each traced span) show only warnings and errors. The D-Bus crate also warns about what a
/// desktop simply lacks ("Failed to populate properties cache" for a service without properties), and GPUI's
/// accessibility tree warns when two elements ask for the focus in one frame (harmless, seen with a screen reader on):
/// only their errors are shown.
const CAPS: &[(&str, log::LevelFilter)] = &[
    ("zbus", log::LevelFilter::Error),
    ("tracing", log::LevelFilter::Warn),
    ("naga", log::LevelFilter::Warn),
    ("polling", log::LevelFilter::Warn),
    ("async_io", log::LevelFilter::Warn),
    ("calloop", log::LevelFilter::Warn),
    ("gpui::window::a11y", log::LevelFilter::Error),
];

/// Whether a line at `level` from `target` (a module path) is shown when `max` is asked for.
fn shown(target: &str, level: log::Level, max: log::LevelFilter) -> bool {
    let under = |prefix: &str| target == prefix || target.strip_prefix(prefix).is_some_and(|rest| rest.starts_with("::"));
    let cap = CAPS.iter().filter(|(prefix, _)| under(prefix)).max_by_key(|(prefix, _)| prefix.len()).map_or(max, |&(_, cap)| max.min(cap));
    level <= cap
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
        // Shown, or on its way out.
        instance::Claim::Shown | instance::Claim::Gone => return,
        // It said it is starting, and its window had not come up in the time waited: the user hears that rather than
        // nothing (it is not stuck, so nothing is to be closed).
        instance::Claim::Starting(pid) => {
            ui::tell_without_window(&instance::still_starting_message(pid));
            return;
        }
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
    use log::{Level, LevelFilter};

    #[test]
    fn the_d_bus_crates_chatter_is_not_shown_at_info_level() {
        let shown = super::shown;
        assert!(!shown("zbus::connection::handshake::common", Level::Info, LevelFilter::Info));
        assert!(!shown("tracing::span", Level::Info, LevelFilter::Trace) && !shown("zbus", Level::Debug, LevelFilter::Debug));
        assert!(shown("tracing::span", Level::Warn, LevelFilter::Info), "their warnings stay");
        assert!(shown("gpui_wgpu::wgpu_context", Level::Info, LevelFilter::Info), "which renderer was picked stays");
        assert!(shown("neural_forge::keytap", Level::Info, LevelFilter::Info) && shown("zbus_names", Level::Info, LevelFilter::Info), "whole names only");
        assert!(!shown("neural_forge::keytap", Level::Info, LevelFilter::Warn), "the default stays warnings only");
    }

    #[test]
    fn harmless_warnings_of_the_d_bus_crate_and_the_accessibility_tree_are_not_shown() {
        let shown = super::shown;
        // "Failed to populate properties cache via GetAll" and "a11y: set_focus called more than once in a single frame".
        assert!(!shown("zbus::proxy", Level::Warn, LevelFilter::Warn));
        assert!(!shown("zbus::connection", Level::Warn, LevelFilter::Info));
        assert!(!shown("gpui::window::a11y", Level::Warn, LevelFilter::Trace));
        assert!(shown("zbus::proxy", Level::Error, LevelFilter::Warn) && shown("gpui::window::a11y", Level::Error, LevelFilter::Warn), "errors stay");
        assert!(
            shown("gpui::window", Level::Warn, LevelFilter::Warn) && shown("gpui", Level::Warn, LevelFilter::Warn),
            "the rest of the toolkit's warnings stay"
        );
        assert!(shown("gpui::window::a11y_extra", Level::Warn, LevelFilter::Warn), "whole module names only");
        // Through the logger itself.
        let metadata = |target| log::Metadata::builder().level(Level::Warn).target(target).build();
        log::set_max_level(LevelFilter::Warn);
        assert!(!log::Log::enabled(&super::StderrLog, &metadata("zbus::proxy")));
        assert!(!log::Log::enabled(&super::StderrLog, &metadata("gpui::window::a11y")));
        assert!(log::Log::enabled(&super::StderrLog, &metadata("gpui_wgpu::wgpu_context")));
    }
}
