//! Programs the GUI starts get the system's environment, not the AppImage's.
//!
//! The AppImage carries a second copy of the GUI that uses a bundled GTK 4 and libadwaita, for desktops
//! without them (`build-appimage.sh`, `bundle_gtk_runtime`). When AppRun starts that copy it points the
//! bundled GTK at its own modules, image loaders and schemas through a few environment variables
//! ([`GLIB_VARS`], [`GTK_VARS`]), and it puts the AppImage's `usr/bin` first in `PATH` for either copy. It names
//! every variable it changed in `NEURAL_FORGE_BUNDLE_ENV` and keeps each one's earlier value in
//! `NEURAL_FORGE_ORIG_<NAME>` (absent when it had none). A program started with the bundle's values would load
//! its modules into its own GTK or GLib (a file manager, Steam's helpers, a game).
//!
//! Two guards:
//! - [`release`]: once GLib and GTK have read their variables, the GUI puts the host's values back in its
//!   own environment: [`GLIB_VARS`] first thing in `main()` (one thread), [`GTK_VARS`] at the application's
//!   startup. That covers what GTK and GIO start on the GUI's behalf (`gtk::FileLauncher` hands the process
//!   environment to the program it launches). `PATH` stays as AppRun set it in the process: the bundled GIO
//!   finds the AppImage's `gio-launch-desktop` through it, which gives the launched program the host's `PATH`.
//!   Under the host copy, the host's GIO uses its own `gio-launch-desktop`, so its children keep the AppImage's
//!   `PATH`.
//! - [`command`]: every [`std::process::Command`] the GUI and the supervisor code it runs build gets the
//!   host's value of everything still listed, `PATH` included (a test fails on a bare `Command::new` outside
//!   test code).
//!
//! The bundled libraries themselves are found through the bundled binary's rpath, never
//! `LD_LIBRARY_PATH`. Outside the AppImage nothing is listed and nothing changes.

use std::ffi::{OsStr, OsString};

const LIST: &str = "NEURAL_FORGE_BUNDLE_ENV";
const ORIG: &str = "NEURAL_FORGE_ORIG_";

/// The bundle's variables GLib and gdk-pixbuf read, each once: the GSettings schema source, GIO's module
/// loading and gdk-pixbuf's loader list. Released at the top of `main()`.
pub const GLIB_VARS: &[&str] = &["GDK_PIXBUF_MODULE_FILE", "GIO_MODULE_DIR", "GIO_EXTRA_MODULES", "GSETTINGS_SCHEMA_DIR"];
/// The bundle's variables GTK reads at its init (its module directory, which it caches). Released at startup.
pub const GTK_VARS: &[&str] = &["GTK_EXE_PREFIX", "GTK_PATH"];

/// What a child's environment needs: each listed variable with the value to set, or `None` to remove it.
/// Empty outside the AppImage.
pub fn restored() -> Vec<(OsString, Option<OsString>)> {
    restored_from(|name| std::env::var_os(name))
}

/// The listed names AppRun wrote: plain variable names only, anything else is not ours to touch.
fn listed(get: &impl Fn(&str) -> Option<OsString>) -> Option<Vec<String>> {
    let list = get(LIST)?;
    Some(
        list.to_string_lossy()
            .split_whitespace()
            .filter(|n| n.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'))
            .map(String::from)
            .collect(),
    )
}

fn restored_from(get: impl Fn(&str) -> Option<OsString>) -> Vec<(OsString, Option<OsString>)> {
    let Some(names) = listed(&get) else { return Vec::new() };
    let mut out = Vec::new();
    for name in names {
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

/// The changes [`release`] makes for `names`: each listed one's host value (or its removal), its saved value
/// removed, and the list without them (removed when empty). Empty when none of `names` is listed.
fn release_plan(get: impl Fn(&str) -> Option<OsString>, names: &[&str]) -> Vec<(OsString, Option<OsString>)> {
    let Some(listed) = listed(&get) else { return Vec::new() };
    let (go, stay): (Vec<String>, Vec<String>) = listed.into_iter().partition(|n| names.contains(&n.as_str()));
    if go.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for name in go {
        let orig = format!("{ORIG}{name}");
        out.push((name.into(), get(&orig)));
        out.push((orig.into(), None));
    }
    out.push((LIST.into(), (!stay.is_empty()).then(|| stay.join(" ").into())));
    out
}

/// Puts the host's values of `names` back into this process's environment, after `load` has made whatever
/// reads them read them (a no-op, `load` not called, when AppRun listed none of them).
///
/// Changing the environment races with a `getenv` on another thread: while glibc (2.39, 2.40) removes a
/// variable it moves the later entries down one by one, so a concurrent reader can miss a variable or briefly
/// see the array end early, and adding a variable can move the whole array. [`GLIB_VARS`] are released at the
/// top of `main()`, while the process has one thread. [`GTK_VARS`] can only go once GTK is initialised, at the
/// application's startup, when GDBus's worker thread exists: AppRun sets both of them (never leaves one unset), so
/// that release only overwrites a value in place or removes one, and the worker does not read the environment
/// there, so the window of a removal is accepted for these two.
pub fn release(names: &[&str], load: impl FnOnce()) {
    let plan = release_plan(|name| std::env::var_os(name), names);
    if plan.is_empty() {
        return;
    }
    load();
    for (name, value) in plan {
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
    fn outside_the_appimage_nothing_changes() {
        assert!(restored_from(env(&[("GIO_MODULE_DIR", "/x")])).is_empty());
        assert!(release_plan(env(&[("GIO_MODULE_DIR", "/x")]), GLIB_VARS).is_empty());
    }

    #[test]
    fn earlier_values_come_back_and_unset_ones_go() {
        let got = restored_from(env(&[
            (LIST, "PATH GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR"),
            ("PATH", "/tmp/.mount_x/usr/bin:/usr/bin"),
            ("NEURAL_FORGE_ORIG_PATH", "/usr/bin"),
            ("GIO_MODULE_DIR", "/tmp/.mount_x/usr/lib/gio/modules"),
            ("GSETTINGS_SCHEMA_DIR", "/tmp/.mount_x/usr/share/glib-2.0/schemas:/home/u/schemas"),
            ("NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR", "/home/u/schemas"),
        ]));
        let want: Vec<(OsString, Option<OsString>)> = vec![
            ("PATH".into(), Some("/usr/bin".into())),
            ("NEURAL_FORGE_ORIG_PATH".into(), None),
            ("GIO_MODULE_DIR".into(), None),
            ("NEURAL_FORGE_ORIG_GIO_MODULE_DIR".into(), None),
            ("GSETTINGS_SCHEMA_DIR".into(), Some("/home/u/schemas".into())),
            ("NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR".into(), None),
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
    fn a_release_takes_only_its_names_off_the_list() {
        let vars = [
            (LIST, "PATH GTK_EXE_PREFIX GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR"),
            ("GIO_MODULE_DIR", "/b/gio"),
            ("GSETTINGS_SCHEMA_DIR", "/b/schemas:/home/u/schemas"),
            ("NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR", "/home/u/schemas"),
        ];
        let want: Vec<(OsString, Option<OsString>)> = vec![
            ("GIO_MODULE_DIR".into(), None),
            ("NEURAL_FORGE_ORIG_GIO_MODULE_DIR".into(), None),
            ("GSETTINGS_SCHEMA_DIR".into(), Some("/home/u/schemas".into())),
            ("NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR".into(), None),
            (LIST.into(), Some("PATH GTK_EXE_PREFIX".into())),
        ];
        assert_eq!(release_plan(env(&vars), GLIB_VARS), want);
        // GTK's go at startup; PATH is never released (command() still restores it).
        let after = [(LIST, "PATH GTK_EXE_PREFIX"), ("GTK_EXE_PREFIX", "/b/usr")];
        let want: Vec<(OsString, Option<OsString>)> =
            vec![("GTK_EXE_PREFIX".into(), None), ("NEURAL_FORGE_ORIG_GTK_EXE_PREFIX".into(), None), (LIST.into(), Some("PATH".into()))];
        assert_eq!(release_plan(env(&after), GTK_VARS), want);
        assert!(release_plan(env(&[(LIST, "PATH")]), GTK_VARS).is_empty());
        assert_eq!(release_plan(env(&[(LIST, "GTK_PATH")]), GTK_VARS).last().unwrap(), &(OsString::from(LIST), None));
    }

    /// `text` with every `#[cfg(test)]` module's body left out (each test module ends where its braces close).
    fn without_test_modules(text: &str) -> String {
        let mut out = String::new();
        let mut lines = text.lines().peekable();
        while let Some(line) = lines.next() {
            let starts_test_module =
                line.trim() == "#[cfg(test)]" && lines.peek().is_some_and(|next| next.trim_start().starts_with("mod ") && next.trim_end().ends_with('{'));
            if !starts_test_module {
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let mut depth = 0i32;
            for line in lines.by_ref() {
                depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
                out.push('\n'); // keep the line numbers
                if depth <= 0 {
                    break;
                }
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn only_test_modules_are_skipped() {
        let src = "#[cfg(test)]\nconst A: u8 = 1;\nfn a() { Command::new(\"x\"); }\n#[cfg(test)]\nmod tests {\n    fn t() { if x { Command::new(\"y\"); } }\n}\nfn b() {}\n";
        let kept = without_test_modules(src);
        assert!(kept.contains("Command::new(\"x\")") && !kept.contains("Command::new(\"y\")") && kept.contains("fn b()"), "{kept}");
        assert_eq!(kept.lines().count(), src.lines().count());
    }

    #[test]
    fn every_program_is_started_through_command() {
        // Outside test modules, nothing the GUI runs may build a Command directly: it would inherit the bundle's
        // variables. The GUI's own crate, and the supervisor and protocol code it runs in-process (build scripts
        // and examples are not part of the GUI).
        fn scan(dir: &std::path::Path, bad: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    scan(&path, bad);
                } else if path.extension().is_some_and(|e| e == "rs") && !path.ends_with("host_env.rs") {
                    let code = without_test_modules(&std::fs::read_to_string(&path).unwrap());
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
            ("NEURAL_FORGE_ORIG_GTK_EXE_PREFIX", "/opt/gtk"),
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
        assert!(!out.contains("GIO_MODULE_DIR=") && !out.contains("NEURAL_FORGE_"), "{out}");
    }

    #[test]
    fn released_env_reaches_children_started_any_way() {
        // The process environment itself, as the bundled GUI leaves it after both releases: a child started
        // without `command` (as GIO starts a file manager) sees the host's values, except PATH, which only
        // `command` restores. In a child process of the test binary, so the test's own environment is not
        // changed under the other tests.
        if std::env::var_os("NEURAL_FORGE_HOST_ENV_CHILD").is_some() {
            let mut loaded = 0;
            release(GLIB_VARS, || loaded += 1);
            release(GTK_VARS, || loaded += 1);
            assert_eq!(loaded, 2, "the lazy readers were not loaded first");
            let plain = std::process::Command::new("sh").args(["-c", "env"]).output().unwrap();
            let through = command("sh").args(["-c", "env"]).output().unwrap();
            print!("PLAIN\n{}THROUGH\n{}", String::from_utf8(plain.stdout).unwrap(), String::from_utf8(through.stdout).unwrap());
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args(["--exact", "host_env::tests::released_env_reaches_children_started_any_way", "--nocapture", "--test-threads=1"])
            .env("NEURAL_FORGE_HOST_ENV_CHILD", "1")
            .env(LIST, "PATH GTK_EXE_PREFIX GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR")
            .env("PATH", "/bundle/usr/bin:/usr/bin:/bin")
            .env("NEURAL_FORGE_ORIG_PATH", "/usr/bin:/bin")
            .env("GTK_EXE_PREFIX", "/bundle/usr")
            .env("GIO_MODULE_DIR", "/bundle/gio")
            .env("GSETTINGS_SCHEMA_DIR", "/bundle/schemas:/home/u/schemas")
            .env("NEURAL_FORGE_ORIG_GSETTINGS_SCHEMA_DIR", "/home/u/schemas")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{text}{}", String::from_utf8_lossy(&out.stderr));
        let (plain, through) = text.split_once("THROUGH\n").expect("the child did not run");
        let plain = plain.split_once("PLAIN\n").unwrap().1;
        for env in [plain, through] {
            assert!(env.lines().any(|l| l == "GSETTINGS_SCHEMA_DIR=/home/u/schemas"), "{env}");
            assert!(!env.contains("GIO_MODULE_DIR=") && !env.contains("GTK_EXE_PREFIX="), "{env}");
        }
        // The process keeps the AppImage's PATH (the bundled GIO finds its gio-launch-desktop through it, which
        // restores the host's for the launched program); `command`'s children get the host's directly.
        assert!(plain.lines().any(|l| l == "PATH=/bundle/usr/bin:/usr/bin:/bin"), "{plain}");
        assert!(
            through.lines().any(|l| l == "PATH=/usr/bin:/bin") && !through.contains("NEURAL_FORGE_ORIG_") && !through.contains("NEURAL_FORGE_BUNDLE_ENV"),
            "{through}"
        );
    }

    #[test]
    fn without_the_bundle_nothing_is_loaded_or_changed() {
        if std::env::var_os(LIST).is_some() {
            return; // the test binary itself runs under the bundled GUI's environment: not this case
        }
        let mut loaded = false;
        release(GLIB_VARS, || loaded = true);
        assert!(!loaded);
    }
}
