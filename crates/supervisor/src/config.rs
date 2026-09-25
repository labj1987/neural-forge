//! The on-disk config file (`~/.config/neural-forge/config.ini`): the same flat
//! `key=value` format upstream's shell script used (a contract other tooling/the
//! user's own scripts might already read, and there's no reason to invent a new
//! format for the same handful of fields), read/written with plain string parsing --
//! this file is small and hand-written, not something that benefits from a real INI
//! crate.

use std::collections::BTreeMap;

use crate::paths;

#[derive(Default, Debug, Clone)]
pub struct Config {
    pub runner_type: String,
    pub runner_path: String,
    pub binaries: String,
    pub shm: String,
    pub log: String,
    pub dxvk_vendor: String,
    pub dxvk_device: String,
    /// Every `set_<name>=<value>` line -- the model/composition tuning
    /// `neural_forge_protocol::persist` round-trips through here so it survives a reboot
    /// (unlike the SHM mapping itself, which lives under `/tmp`). Kept as raw
    /// strings rather than parsed here: this crate doesn't need to know what any of
    /// these settings mean, only that they persist.
    pub settings: BTreeMap<String, String>,
}

impl Config {
    pub fn load() -> Self {
        let Ok(text) = std::fs::read_to_string(paths::config_file()) else {
            return Self::default();
        };
        let mut map = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                map.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        let mut cfg = Self {
            runner_type: map.remove("runner_type").unwrap_or_default(),
            runner_path: map.remove("runner_path").unwrap_or_default(),
            binaries: map.remove("binaries").unwrap_or_default(),
            shm: map.remove("shm").unwrap_or_default(),
            log: map.remove("log").unwrap_or_default(),
            dxvk_vendor: map.remove("dxvk_vendor").unwrap_or_default(),
            dxvk_device: map.remove("dxvk_device").unwrap_or_default(),
            settings: BTreeMap::new(),
        };
        // A config written by an older build (or a stray manual edit) pinning the
        // mapping to $XDG_RUNTIME_DIR is exactly the path a Steam game cannot see
        // (see neural_forge_protocol's own doc comment on this) -- treat that one value as
        // unset rather than let it silently reintroduce the bug it exists to avoid.
        if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
            if cfg.shm == format!("{runtime_dir}/neural-forge/shm.bin") {
                cfg.shm.clear();
            }
        }
        if cfg.shm.is_empty() || !neural_forge_protocol::isolated_path(&cfg.shm) {
            cfg.shm = neural_forge_protocol::shm_default_path();
        }
        if cfg.log.is_empty() || !neural_forge_protocol::isolated_path(&cfg.log) {
            cfg.log = paths::log_file();
        }
        // Whatever's left in `map` after pulling out the known fields above is every
        // `set_*` tuning line (plus, harmlessly, anything else an older/newer build
        // or a stray manual edit left behind) -- kept rather than dropped so `save()`
        // round-trips it.
        cfg.settings = map;
        cfg
    }

    /// Replaces every `set_*` tuning line with `tuning` (a fresh
    /// `neural_forge_protocol::persist::snapshot`), keeping any other key -- such as the
    /// GUI's launch-option switches -- that a wholesale `settings = snapshot` would drop.
    pub fn replace_tuning(&mut self, tuning: BTreeMap<String, String>) {
        self.settings.retain(|k, _| !k.starts_with("set_"));
        self.settings.extend(tuning);
    }

    pub fn save(&self) -> std::io::Result<()> {
        paths::ensure_dirs()?;
        let mut text = format!(
            "runner_type={}\nrunner_path={}\nbinaries={}\nshm={}\nlog={}\ndxvk_vendor={}\ndxvk_device={}\n",
            self.runner_type, self.runner_path, self.binaries, self.shm, self.log, self.dxvk_vendor, self.dxvk_device,
        );
        for (k, v) in &self.settings {
            text.push_str(&format!("{k}={v}\n"));
        }
        std::fs::write(paths::config_file(), text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_tuning_swaps_set_keys_and_keeps_the_rest() {
        let mut cfg = Config::default();
        cfg.settings.insert("set_enabled".into(), "0".into());
        cfg.settings.insert("set_stale".into(), "1".into());
        cfg.settings.insert("launch_smooth_motion".into(), "1".into());
        cfg.replace_tuning(BTreeMap::from([("set_enabled".to_string(), "1".to_string())]));
        assert_eq!(cfg.settings.get("set_enabled").map(String::as_str), Some("1"));
        assert!(!cfg.settings.contains_key("set_stale"));
        assert_eq!(cfg.settings.get("launch_smooth_motion").map(String::as_str), Some("1"));
    }
}
