//! Programs the GUI starts get the system's environment, not the AppImage's bundled GTK's.
//!
//! The AppImage carries a second copy of the GUI that uses a bundled GTK 4 and libadwaita, for desktops
//! without them (`build-appimage.sh`, `bundle_gtk_runtime`). When AppRun starts that copy it points the
//! bundled GTK at its own modules, image loaders and schemas through a few environment variables
//! (`GTK_EXE_PREFIX`, `GIO_MODULE_DIR`, ...), names them in `APPIMAGE_BUNDLE_ENV` and keeps each one's
//! earlier value in `APPIMAGE_ORIG_<NAME>` (absent when it had none). A program started with those set
//! would load the bundle's modules into its own GTK or GLib (a file manager, Steam's helpers, a game).
//!
//! Two guards, so nothing the GUI starts ever sees them:
//! - [`release_bundle_env`]: once GTK has read them (at the application's startup), the GUI puts the
//!   host's values back in its own environment. That covers what GTK and GIO start on the GUI's behalf
//!   (`gtk::FileLauncher` hands the process environment to the file manager it launches).
//! - [`command`]: every [`std::process::Command`] the GUI and the supervisor code it runs build is made
//!   with the host's values (a test fails on a bare `Command::new` outside test code), whether or not
//!   the process environment was released yet.
//!
//! The bundled libraries themselves are found through the bundled binary's rpath, never
//! `LD_LIBRARY_PATH`, so nothing else leaks. With the host's GTK (the usual case) or outside the
//! AppImage nothing is set and nothing changes.

use std::ffi::{OsStr, OsString};

const LIST: &str = "APPIMAGE_BUNDLE_ENV";
const ORIG: &str = "APPIMAGE_ORIG_";

/// What a child's environment needs: each variable with the value to set, or `None` to remove it.
/// Empty outside the bundled GUI.
pub fn restored() -> Vec<(OsString, Option<OsString>)> {
    restored_from(|name| std::env::var_os(name))
}

fn restored_from(get: impl Fn(&str) -> Option<OsString>) -> Vec<(OsString, Option<OsString>)> {
    let Some(list) = get(LIST) else { return Vec::new() };
    let mut out = Vec::new();
    for name in list.to_string_lossy().split_whitespace() {
        // AppRun only ever lists plain variable names; anything else is not ours to touch.
        if !name.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_') {
            continue;
        }
        let orig = format!("{ORIG}{name}");
        out.push((name.into(), get(&orig)));
        out.push((orig.into(), None));
    }
    out.push((LIST.into(), None));
    out
}

/// `std::process::Command::new(program)` with the system's environment for the bundle's variables.
pub fn command(program: impl AsRef<OsStr>) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    for (name, value) in restored() {
        match value {
            Some(value) => cmd.env(name, value),
            None => cmd.env_remove(name),
        };
    }
    cmd
}

/// Puts the host's values of the bundle's variables back into this process's environment, once the
/// bundled GTK has read them; a no-op with the host's GTK. Call it on the main thread from the
/// application's `startup` handler (GTK is initialised by then: its module directory, read from
/// `GTK_EXE_PREFIX`, is cached, and its input method modules are loaded).
///
/// The rest of what reads these variables does so lazily, once: GLib's GSettings schema source
/// (`GSETTINGS_SCHEMA_DIR`), GIO's module loading (`GIO_MODULE_DIR`) and gdk-pixbuf's loaders
/// (`GDK_PIXBUF_MODULE_FILE`). Each is made to happen here first, so none reads the host's value later.
///
/// Changing the environment of a process with threads (GDBus's worker exists by then) is safe here:
/// every variable is already set, so glibc overwrites or removes its slot in place, never moving the
/// environment array a concurrent `getenv` may be reading, and none of these threads reads them.
pub fn release_bundle_env(load_lazy_readers: impl FnOnce()) {
    let restore = restored();
    if restore.is_empty() {
        return;
    }
    load_lazy_readers();
    for (name, value) in restore {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = vars.iter().map(|(k, v)| (k.to_string(), OsString::from(v))).collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn outside_the_bundled_gui_nothing_changes() {
        assert!(restored_from(env(&[("GIO_MODULE_DIR", "/x")])).is_empty());
    }

    #[test]
    fn earlier_values_come_back_and_unset_ones_go() {
        let got = restored_from(env(&[
            (LIST, "GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR"),
            ("GIO_MODULE_DIR", "/tmp/.mount_x/usr/lib/gio/modules"),
            ("GSETTINGS_SCHEMA_DIR", "/tmp/.mount_x/usr/share/glib-2.0/schemas:/home/u/schemas"),
            ("APPIMAGE_ORIG_GSETTINGS_SCHEMA_DIR", "/home/u/schemas"),
        ]));
        let want: Vec<(OsString, Option<OsString>)> = vec![
            ("GIO_MODULE_DIR".into(), None),
            ("APPIMAGE_ORIG_GIO_MODULE_DIR".into(), None),
            ("GSETTINGS_SCHEMA_DIR".into(), Some("/home/u/schemas".into())),
            ("APPIMAGE_ORIG_GSETTINGS_SCHEMA_DIR".into(), None),
            (LIST.into(), None),
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn odd_names_are_ignored() {
        let got = restored_from(env(&[(LIST, "PATH=x lower GTK_EXE_PREFIX")]));
        assert_eq!(got[0].0, OsString::from("GTK_EXE_PREFIX"));
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn every_program_is_started_through_command() {
        // Outside test code, nothing the GUI runs may build a Command directly: it would inherit the
        // bundle's variables. The GUI's own crate, and the supervisor and protocol code it runs in-process
        // (build scripts and examples are not part of the GUI).
        fn scan(dir: &std::path::Path, bad: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    scan(&path, bad);
                } else if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("host_env.rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    let code = text.split("#[cfg(test)]").next().unwrap_or_default();
                    for (n, line) in code.lines().enumerate() {
                        let line = line.trim_start();
                        if !line.starts_with("//") && (line.contains("Command::new(") || line.contains("Command::new ")) {
                            bad.push(format!("{}:{}", path.display(), n + 1));
                        }
                    }
                }
            }
        }
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut bad = Vec::new();
        for krate in ["gui", "supervisor", "protocol"] {
            let src = crates.join(krate).join("src");
            assert!(src.is_dir(), "{} is missing", src.display());
            scan(&src, &mut bad);
        }
        assert!(bad.is_empty(), "use neural_forge_supervisor::host_env::command instead of Command::new at {bad:?}");
    }

    #[test]
    fn a_child_sees_the_system_environment() {
        // Run a real child through `command`'s logic, with the parent's environment as AppRun leaves it.
        let parent = [
            (LIST, "GIO_MODULE_DIR GTK_EXE_PREFIX"),
            ("GIO_MODULE_DIR", "/bundle/gio"),
            ("GTK_EXE_PREFIX", "/bundle/usr"),
            ("APPIMAGE_ORIG_GTK_EXE_PREFIX", "/opt/gtk"),
        ];
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "env"]);
        // What the child would inherit from the GUI, then what `command` does to it.
        for (name, value) in parent {
            cmd.env(name, value);
        }
        for (name, value) in restored_from(env(&parent)) {
            match value {
                Some(value) => cmd.env(name, value),
                None => cmd.env_remove(name),
            };
        }
        let out = String::from_utf8(cmd.output().unwrap().stdout).unwrap();
        assert!(out.lines().any(|l| l == "GTK_EXE_PREFIX=/opt/gtk"), "{out}");
        assert!(!out.contains("GIO_MODULE_DIR=") && !out.contains("APPIMAGE_"), "{out}");
    }

    #[test]
    fn released_env_reaches_children_started_any_way() {
        // The process environment itself, as the bundled GUI's startup leaves it: a child started without
        // `command` (as GIO starts a file manager) sees the host's values too. In a child process of the
        // test binary, so the test's own environment is not changed under the other tests.
        if std::env::var_os("NEURAL_FORGE_HOST_ENV_CHILD").is_some() {
            let mut loaded = false;
            release_bundle_env(|| loaded = true);
            assert!(loaded, "the lazy readers were not loaded first");
            let out = std::process::Command::new("sh").args(["-c", "env"]).output().unwrap();
            print!("{}", String::from_utf8(out.stdout).unwrap());
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args(["--exact", "host_env::tests::released_env_reaches_children_started_any_way", "--nocapture", "--test-threads=1"])
            .env("NEURAL_FORGE_HOST_ENV_CHILD", "1")
            .env(LIST, "GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR")
            .env("GIO_MODULE_DIR", "/bundle/gio")
            .env("GSETTINGS_SCHEMA_DIR", "/bundle/schemas:/home/u/schemas")
            .env("APPIMAGE_ORIG_GSETTINGS_SCHEMA_DIR", "/home/u/schemas")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{text}{}", String::from_utf8_lossy(&out.stderr));
        assert!(text.lines().any(|l| l == "GSETTINGS_SCHEMA_DIR=/home/u/schemas"), "{text}");
        assert!(text.lines().any(|l| l == "NEURAL_FORGE_HOST_ENV_CHILD=1"), "the child did not run: {text}");
        assert!(!text.contains("GIO_MODULE_DIR=") && !text.contains("APPIMAGE_"), "{text}");
    }

    #[test]
    fn without_the_bundle_nothing_is_loaded_or_changed() {
        if std::env::var_os(LIST).is_some() {
            return; // the test binary itself runs under the bundled GUI's environment: not this case
        }
        let mut loaded = false;
        release_bundle_env(|| loaded = true);
        assert!(!loaded);
    }
}
