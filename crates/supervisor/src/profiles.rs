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

pub fn load_all() -> BTreeMap<String, ProfileSettings> {
    let Ok(text) = std::fs::read_to_string(profiles_file()) else {
        return BTreeMap::new();
    };
    parse(&text)
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
    std::fs::write(profiles_file(), text)
}

pub fn save_profile(name: &str, settings: ProfileSettings) -> std::io::Result<()> {
    let mut all = load_all();
    all.insert(name.to_string(), settings);
    save_all(&all)
}

/// Returns `true` if `name` existed and was removed.
pub fn delete_profile(name: &str) -> std::io::Result<bool> {
    let mut all = load_all();
    let existed = all.remove(name).is_some();
    if existed {
        save_all(&all)?;
    }
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Guards every test here for the same reason `paths::tests` guards
    // `XDG_DATA_HOME`: `XDG_CONFIG_HOME` is a real process-wide env var, and Rust's
    // default test harness runs these in parallel threads within one process.
    static XDG_CONFIG_HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct ScratchConfigHome {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Option<String>,
        dir: std::path::PathBuf,
    }

    impl ScratchConfigHome {
        fn new(tag: &str) -> Self {
            let guard = XDG_CONFIG_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-profiles-test-{tag}-{}", std::process::id()));
            let prev = std::env::var("XDG_CONFIG_HOME").ok();
            std::env::set_var("XDG_CONFIG_HOME", &dir);
            Self { _guard: guard, prev, dir }
        }
    }

    impl Drop for ScratchConfigHome {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
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
}
