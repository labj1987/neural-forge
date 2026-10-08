//! The on-disk config file (`~/.config/neural-forge/config.ini`): flat `key=value` lines, read with
//! `neural_forge_protocol::persist::read_config` (the layer reads the same file for the saved settings).

use std::collections::BTreeMap;

use crate::paths;

#[derive(Default, Debug, Clone)]
pub struct Config {
    /// Where NVIDIA's `nvngx_dlssnr.dll` was imported to, the model's source for `extract-model`.
    pub binaries: String,
    /// The channel path `config.ini` pins, or empty for none. Read it through
    /// [`crate::channel_path`], which falls back to `NEURAL_FORGE_SHM` and then the default.
    pub shm: String,
    /// Every other line: the `set_<name>=<value>` settings `neural_forge_protocol::persist`
    /// round-trips (so they survive a reboot, unlike the mapping under `/tmp`), and anything else, kept
    /// so `save()` round-trips it.
    pub settings: BTreeMap<String, String>,
}

impl Config {
    pub fn load() -> Self {
        let mut map = neural_forge_protocol::persist::read_config(&paths::config_file());
        let mut cfg = Self { binaries: map.remove("binaries").unwrap_or_default(), shm: map.remove("shm").unwrap_or_default(), settings: map };
        if !neural_forge_protocol::isolated_path(&cfg.shm) {
            cfg.shm.clear();
        }
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
        let mut text = format!("binaries={}\nshm={}\n", self.binaries, self.shm);
        for (k, v) in &self.settings {
            text.push_str(&format!("{k}={v}\n"));
        }
        paths::write_atomic(std::path::Path::new(&paths::config_file()), text.as_bytes(), 0o644)
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
        cfg.settings.insert("launch_extra".into(), "1".into());
        cfg.replace_tuning(BTreeMap::from([("set_enabled".to_string(), "1".to_string())]));
        assert_eq!(cfg.settings.get("set_enabled").map(String::as_str), Some("1"));
        assert!(!cfg.settings.contains_key("set_stale"));
        assert_eq!(cfg.settings.get("launch_extra").map(String::as_str), Some("1"));
    }
}
