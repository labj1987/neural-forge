//! Named settings profiles: `[name]` sections of the same `set_<field>=<value>` lines
//! `config.ini` itself stores (`neural_forge_protocol::persist::snapshot`/`apply`
//! already define that shape), saved to their own `profiles.ini` next to it rather
//! than as extra sections inside `config.ini` -- keeps `Config::load`'s flat parser
//! untouched and the upstream-compatible `config.ini` format exactly as it was.

use std::collections::BTreeMap;

use crate::paths;

pub type ProfileSettings = BTreeMap<String, String>;

pub fn profiles_file() -> String {
    format!("{}/profiles.ini", paths::config_dir())
}

/// Every saved profile, for listing and loading: none when there is no `profiles.ini` or it cannot
/// be read, and a byte that is not UTF-8 reads as U+FFFD. [`save_profile`] and [`delete_profile`]
/// read it strictly instead: a file there that cannot be read stops them.
pub fn load_all() -> BTreeMap<String, ProfileSettings> {
    match std::fs::read(profiles_file()) {
        Ok(bytes) => parse(&String::from_utf8_lossy(&bytes)),
        Err(_) => BTreeMap::new(),
    }
}

/// The profiles to rewrite `profiles.ini` from: none only when there is no file. A file that cannot
/// be read, or is not UTF-8 (a name or value that would be rewritten with U+FFFD in it), is an error,
/// so a save or delete never replaces it with fewer profiles than it holds.
fn load_for_update() -> std::io::Result<BTreeMap<String, ProfileSettings>> {
    let file = profiles_file();
    let bytes = match std::fs::read(&file) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(std::io::Error::new(e.kind(), format!("{file} cannot be read ({e}); not replacing it, which would lose the profiles it holds"))),
    };
    match String::from_utf8(bytes) {
        Ok(text) => Ok(parse(&text)),
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{file} is not UTF-8 (byte {}); not replacing it, which would change the profiles it holds: fix that line by hand",
                e.utf8_error().valid_up_to()
            ),
        )),
    }
}

fn parse(text: &str) -> BTreeMap<String, ProfileSettings> {
    let mut profiles: BTreeMap<String, ProfileSettings> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
            profiles.entry(name.to_string()).or_default();
            current = Some(name.to_string());
            continue;
        }
        let Some(name) = &current else { continue };
        if let Some((k, v)) = line.split_once('=') {
            profiles.entry(name.clone()).or_default().insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    profiles
}

pub fn save_all(profiles: &BTreeMap<String, ProfileSettings>) -> std::io::Result<()> {
    paths::ensure_dirs()?;
    let mut text = String::new();
    for (name, settings) in profiles {
        text.push_str(&format!("[{name}]\n"));
        for (k, v) in settings {
            text.push_str(&format!("{k}={v}\n"));
        }
        text.push('\n');
    }
    paths::write_atomic(std::path::Path::new(&profiles_file()), text.as_bytes(), 0o644)
}

/// Why `name` cannot name a profile, if it cannot: it must be non-empty, without surrounding
/// spaces, at most 64 characters, and without a line break, `[`, `]`, `=` or `#` (each would
/// read back as another section, a setting or a comment in `profiles.ini`).
pub fn invalid_name(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        Some("a profile needs a name")
    } else if name.trim() != name {
        Some("a profile name cannot start or end with a space")
    } else if name.chars().count() > 64 {
        Some("a profile name is at most 64 characters")
    } else if name.chars().any(|c| c.is_control() || matches!(c, '[' | ']' | '=' | '#')) {
        Some("a profile name cannot hold a line break, [, ], = or #")
    } else {
        None
    }
}

/// Saves `settings` as profile `name` ([`invalid_name`] refuses a name `profiles.ini` cannot hold).
pub fn save_profile(name: &str, settings: ProfileSettings) -> std::io::Result<()> {
    if let Some(why) = invalid_name(name) {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, why));
    }
    let mut all = load_for_update()?;
    all.insert(name.to_string(), settings);
    save_all(&all)
}

/// Returns `true` if `name` existed and was removed.
pub fn delete_profile(name: &str) -> std::io::Result<bool> {
    let mut all = load_for_update()?;
    let existed = all.remove(name).is_some();
    if existed {
        save_all(&all)?;
    }
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_that_would_break_the_file_are_refused() {
        for bad in ["", " a", "a ", "a\nb", "[x]", "a=b", "#c", "a\rb", &"x".repeat(65)] {
            assert!(invalid_name(bad).is_some(), "{bad:?}");
            assert_eq!(save_profile(bad, ProfileSettings::new()).unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
        }
        for good in ["Night", "GTA V – sharp", "a b c"] {
            assert_eq!(invalid_name(good), None, "{good:?}");
        }
    }

    // Guards every test here for the same reason `paths::tests` guards
    // `XDG_DATA_HOME`: `XDG_CONFIG_HOME` is a real process-wide env var, and Rust's
    // default test harness runs these in parallel threads within one process.

    struct ScratchConfigHome {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Vec<(&'static str, Option<String>)>,
        dir: std::path::PathBuf,
    }

    impl ScratchConfigHome {
        fn new(tag: &str) -> Self {
            // The one lock every env-mutating test in this crate takes: `install`'s tests set
            // XDG_CONFIG_HOME too, and a separate lock here did not exclude them.
            let guard = crate::paths::tests::XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-profiles-test-{tag}-{}", std::process::id()));
            // Config, data and state all point into the scratch dir: nothing here may create the
            // real XDG dirs.
            let mut prev = Vec::new();
            for (var, sub) in [("XDG_CONFIG_HOME", ""), ("XDG_DATA_HOME", "data-home"), ("XDG_STATE_HOME", "state-home")] {
                prev.push((var, std::env::var(var).ok()));
                std::env::set_var(var, if sub.is_empty() { dir.clone() } else { dir.join(sub) });
            }
            Self { _guard: guard, prev, dir }
        }
    }

    impl Drop for ScratchConfigHome {
        fn drop(&mut self) {
            for (var, value) in self.prev.drain(..) {
                match value {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    #[test]
    fn save_then_load_round_trips_a_profile() {
        let _scratch = ScratchConfigHome::new("round-trip");
        let mut settings = ProfileSettings::new();
        settings.insert("set_intensity".to_string(), "1.75".to_string());
        settings.insert("set_style".to_string(), "2".to_string());
        save_profile("Racing", settings.clone()).expect("save_profile should succeed");

        let loaded = load_all();
        assert_eq!(loaded.get("Racing"), Some(&settings));
    }

    #[test]
    fn save_profile_does_not_disturb_other_profiles() {
        let _scratch = ScratchConfigHome::new("multi");
        let mut a = ProfileSettings::new();
        a.insert("set_style".to_string(), "1".to_string());
        save_profile("A", a.clone()).unwrap();
        let mut b = ProfileSettings::new();
        b.insert("set_style".to_string(), "3".to_string());
        save_profile("B", b.clone()).unwrap();

        let loaded = load_all();
        assert_eq!(loaded.get("A"), Some(&a));
        assert_eq!(loaded.get("B"), Some(&b));
    }

    #[test]
    fn delete_profile_removes_only_the_named_one() {
        let _scratch = ScratchConfigHome::new("delete");
        save_profile("Keep", ProfileSettings::new()).unwrap();
        save_profile("Drop", ProfileSettings::new()).unwrap();

        assert!(delete_profile("Drop").expect("delete should succeed"));
        let loaded = load_all();
        assert!(loaded.contains_key("Keep"));
        assert!(!loaded.contains_key("Drop"));
    }

    #[test]
    fn delete_profile_returns_false_for_an_unknown_name() {
        let _scratch = ScratchConfigHome::new("delete-missing");
        assert!(!delete_profile("NeverSaved").expect("delete of a missing profile should not error"));
    }

    #[test]
    fn load_all_with_no_file_is_empty() {
        let _scratch = ScratchConfigHome::new("empty");
        assert!(load_all().is_empty());
    }

    #[test]
    fn a_file_that_cannot_be_read_or_is_not_utf8_is_never_rewritten() {
        let scratch = ScratchConfigHome::new("unreadable");
        let file = std::path::PathBuf::from(profiles_file());
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        let held = b"[Caf\xe9]\nset_style=1\n[Day]\nset_style=2\n";
        std::fs::write(&file, held).unwrap();
        assert_eq!(save_profile("Night", ProfileSettings::new()).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
        assert!(delete_profile("Day").is_err());
        assert_eq!(std::fs::read(&file).unwrap(), held, "the file is left as it was");
        let listed: Vec<String> = load_all().into_keys().collect();
        assert_eq!(listed, ["Caf\u{fffd}", "Day"], "listing still shows every profile");
        // A file that exists but cannot be read at all: a directory where it goes.
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        assert!(save_profile("Night", ProfileSettings::new()).is_err());
        assert!(file.is_dir());
        drop(scratch);
    }
}
