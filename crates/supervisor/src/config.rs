//! The on-disk config file (`~/.config/neural-forge/config.ini`): flat `key=value` lines, the shape
//! `neural_forge_protocol::persist::read_config` reads (the layer reads the same file for the saved
//! settings). Read here with the same rules, but telling a missing file from one that cannot be read:
//! every front end saves this file after changing one key, so a read error taken for "empty" would
//! write back only that key and lose the rest.

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
    /// Why the file could not be read, when it exists but could not (permissions, an I/O error):
    /// [`Config::save`] then refuses, so the file is never replaced by one holding only this run's keys.
    read_error: Option<String>,
    /// The lines that are not UTF-8, by key: the value read (with U+FFFD for each bad byte) and the
    /// line's own bytes, written back unchanged by [`Config::save`] while the value is still the one read.
    raw_lines: BTreeMap<String, (String, Vec<u8>)>,
}

impl Config {
    pub fn load() -> Self {
        Self::load_from(&paths::config_file())
    }

    /// [`Config::load`] from `file` instead of the user's `config.ini`. A missing file is an empty
    /// config; one that cannot be read is too, with [`Config::read_error`] saying why.
    pub fn load_from(file: &str) -> Self {
        let bytes = match std::fs::read(file) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Self { read_error: Some(format!("{file} cannot be read ({e})")), ..Self::default() },
        };
        let mut map = BTreeMap::new();
        let mut raw_lines = BTreeMap::new();
        for line in bytes.split(|&b| b == b'\n') {
            let (text, valid) = match std::str::from_utf8(line) {
                Ok(text) => (std::borrow::Cow::Borrowed(text), true),
                Err(_) => (String::from_utf8_lossy(line), false),
            };
            // `persist::read_config`'s rules: trimmed, blank lines and `#` comments skipped, the first `=` splits.
            let text = text.trim();
            if text.is_empty() || text.starts_with('#') {
                continue;
            }
            let Some((k, v)) = text.split_once('=') else { continue };
            let (k, v) = (k.trim().to_string(), v.trim().to_string());
            if valid {
                raw_lines.remove(&k);
            } else {
                let raw = line.strip_suffix(b"\r").unwrap_or(line).to_vec();
                raw_lines.insert(k.clone(), (v.clone(), raw));
            }
            map.insert(k, v);
        }
        let mut cfg = Self {
            binaries: map.remove("binaries").unwrap_or_default(),
            shm: map.remove("shm").unwrap_or_default(),
            settings: map,
            read_error: None,
            raw_lines,
        };
        if !neural_forge_protocol::isolated_path(&cfg.shm) {
            cfg.shm.clear();
        }
        cfg
    }

    /// Why `config.ini` could not be read, when it exists and could not be: [`Config::save`] refuses then.
    pub fn read_error(&self) -> Option<&str> {
        self.read_error.as_deref()
    }

    /// Replaces every `set_*` tuning line with `tuning` (a fresh
    /// `neural_forge_protocol::persist::snapshot`), keeping any other key -- such as the
    /// GUI's launch-option switches -- that a wholesale `settings = snapshot` would drop.
    pub fn replace_tuning(&mut self, tuning: BTreeMap<String, String>) {
        self.settings.retain(|k, _| !k.starts_with("set_"));
        self.settings.extend(tuning);
    }

    /// The file's bytes: one `key=value` line each; a line that was not UTF-8 is written back as it
    /// was while its value is unchanged.
    fn text(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut line = |k: &str, v: &str| match self.raw_lines.get(k) {
            Some((read, raw)) if read == v => {
                out.extend_from_slice(raw);
                out.push(b'\n');
            }
            _ => out.extend_from_slice(format!("{k}={v}\n").as_bytes()),
        };
        line("binaries", &self.binaries);
        line("shm", &self.shm);
        for (k, v) in &self.settings {
            line(k, v);
        }
        out
    }

    /// Writes `config.ini`, replacing it whole. Refused when it was there but could not be read
    /// ([`Config::read_error`]): the keys this run did not see would be lost.
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to(std::path::Path::new(&paths::config_file()))
    }

    fn save_to(&self, file: &std::path::Path) -> std::io::Result<()> {
        if let Some(why) = &self.read_error {
            return Err(std::io::Error::other(format!("{why}; not replacing it, which would lose the settings it holds")));
        }
        paths::ensure_dirs()?;
        paths::write_atomic(file, &self.text(), 0o644)
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

    fn scratch(tag: &str) -> crate::paths::tests::ScratchDir {
        crate::paths::tests::ScratchDir::new(&format!("config-{tag}"))
    }

    #[test]
    fn a_config_that_cannot_be_read_is_never_overwritten() {
        let dir = scratch("unreadable");
        // A directory where the file goes: it exists, and reading it fails with something other than NotFound.
        let file = dir.join("config.ini");
        std::fs::create_dir(&file).unwrap();
        let cfg = Config::load_from(&file.to_string_lossy());
        assert!(cfg.read_error().is_some_and(|e| e.contains("cannot be read")), "{:?}", cfg.read_error());
        let err = cfg.save_to(&dir.join("out.ini")).unwrap_err();
        assert!(err.to_string().contains("not replacing it"), "{err}");
        assert!(!dir.join("out.ini").exists());
        // A missing file is simply empty, and saving it is fine.
        let missing = Config::load_from(&dir.join("missing.ini").to_string_lossy());
        assert!(missing.read_error().is_none() && missing.settings.is_empty());
    }

    #[test]
    fn a_line_that_is_not_utf8_keeps_the_rest_and_is_written_back_unchanged() {
        let dir = scratch("latin1");
        let file = dir.join("config.ini");
        std::fs::write(&file, b"binaries=/home/x/caf\xe9\nshm=\ntarget_exe_1=a.exe\r\nedit_lut=/l.cube\n# note\nset_intensity=2\n").unwrap();
        let mut cfg = Config::load_from(&file.to_string_lossy());
        assert!(cfg.read_error().is_none());
        assert_eq!(cfg.binaries, "/home/x/caf\u{fffd}");
        assert_eq!(cfg.settings.get("target_exe_1").map(String::as_str), Some("a.exe"));
        assert_eq!(cfg.settings.get("edit_lut").map(String::as_str), Some("/l.cube"));
        cfg.settings.insert("set_intensity".into(), "1".into());
        let out = dir.join("out.ini");
        cfg.save_to(&out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"binaries=/home/x/caf\xe9\nshm=\nedit_lut=/l.cube\nset_intensity=1\ntarget_exe_1=a.exe\n");
        // A changed value is written as it now is.
        cfg.binaries = "/b".into();
        cfg.save_to(&out).unwrap();
        assert!(std::fs::read(&out).unwrap().starts_with(b"binaries=/b\n"));
    }

    #[test]
    fn loading_reads_what_the_layer_reads() {
        let dir = scratch("same-as-layer");
        let file = dir.join("config.ini");
        std::fs::write(&file, " binaries = /b \n\n#c=1\nnoequals\nset_style=2\nset_style=3\nk=a=b\nshm=/tmp/neural-forge-1/x.bin\n").unwrap();
        let cfg = Config::load_from(&file.to_string_lossy());
        let mut all = cfg.settings.clone();
        all.insert("binaries".into(), cfg.binaries.clone());
        all.insert("shm".into(), cfg.shm.clone());
        assert_eq!(all, neural_forge_protocol::persist::read_config(&file.to_string_lossy()));
    }
}
