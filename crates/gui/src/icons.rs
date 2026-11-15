//! The symbolic icons the GUI uses, shipped inside it (`icons/`, compiled by build.rs into a GResource) rather than
//! taken from the desktop's icon theme: KDE's Breeze, for one, has no `view-paged-symbolic` or
//! `network-transmit-receive-symbolic` and only a full-colour `applications-graphics`, so tabs showed the
//! missing-image placeholder or a grey blob there. They are Adwaita's (50.0) under an `nf-` prefix: GTK looks a name
//! up in the desktop's theme before the app's own icons, so only a name no theme has makes ours the one shown.
//! `GtkApplication` adds `<resource base path>/icons` (from the application ID) to the icon theme by itself.

pub const MODEL: &str = "nf-applications-graphics-symbolic";
pub const COMPOSITION: &str = "nf-view-paged-symbolic";
pub const DEBUG: &str = "nf-edit-find-symbolic";
pub const STATUS: &str = "nf-network-transmit-receive-symbolic";
pub const SETUP: &str = "nf-preferences-system-symbolic";
pub const ABOUT: &str = "nf-help-about-symbolic";
pub const UNDO: &str = "nf-edit-undo-symbolic";
pub const CLEAR: &str = "nf-edit-clear-symbolic";
pub const REMOVE: &str = "nf-list-remove-symbolic";
pub const EDIT: &str = "nf-document-edit-symbolic";
pub const REFRESH: &str = "nf-view-refresh-symbolic";
pub const ERROR: &str = "nf-dialog-error-symbolic";
pub const WARNING: &str = "nf-dialog-warning-symbolic";
pub const INFORMATION: &str = "nf-dialog-information-symbolic";
pub const OK: &str = "nf-object-select-symbolic";

/// Every icon above.
#[cfg(test)]
const ALL: [&str; 15] = [MODEL, COMPOSITION, DEBUG, STATUS, SETUP, ABOUT, UNDO, CLEAR, REMOVE, EDIT, REFRESH, ERROR, WARNING, INFORMATION, OK];

/// Where the icons are in the resource bundle: the application's resource base path (its ID) and `icons/`, in the
/// theme layout GTK's icon lookup reads (icons.gresource.xml's prefix).
#[cfg(test)]
const RESOURCE_DIR: &str = "/io/github/labj1987/NeuralForge/icons/scalable/actions";

/// Registers the bundle; before GTK starts, so the application finds it at startup.
pub fn register() {
    gio::resources_register_include!("icons.gresource").expect("the icon bundle compiled into this binary is valid");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_is_in_the_bundle_under_the_applications_resource_path() {
        register();
        for name in ALL {
            let path = format!("{RESOURCE_DIR}/{name}.svg");
            let data = gio::resources_lookup_data(&path, gio::ResourceLookupFlags::NONE).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(std::str::from_utf8(&data).is_ok_and(|svg| svg.contains("<svg")), "{path} is an SVG");
        }
        // The path GtkApplication adds for this ID: /io/github/labj1987/NeuralForge, then icons/.
        assert!(RESOURCE_DIR.starts_with(&format!("/{}/icons/", crate::APP_ID.replace('.', "/"))));
    }

    #[test]
    fn the_ui_names_only_the_bundled_icons() {
        // Any `"...-symbolic"` literal in the GUI's sources is a theme lookup that could miss: they all go through here.
        for (file, text) in [("ui.rs", include_str!("ui.rs")), ("main.rs", include_str!("main.rs"))] {
            for (at, _) in text.match_indices("-symbolic\"") {
                let start = text[..at].rfind('"').expect("a string literal") + 1;
                let name = &text[start..at + "-symbolic".len()];
                assert!(ALL.contains(&name), "{file} names the icon {name}, which is not bundled (icons.rs)");
            }
        }
    }
}
