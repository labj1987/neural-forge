//! Per-game settings for the GUI and the CLI: the user's `games.ini` (its format, the merge and how a game
//! is named are `neural_forge_protocol::games`'s), and saving a change made on the channel to whichever
//! settings the header holds: `config.ini` for all games, or that game's own.
//!
//! A game's section holds only what differs from the settings it would get without it (the defaults,
//! `config.ini` and the overrides shipped for it): saving a value equal to that removes the line, so a
//! later change for all games still reaches the game. Lines and sections this build does not know are kept.

use std::collections::BTreeMap;

use neural_forge_protocol::games;
pub use neural_forge_protocol::games::{GameKey, Section};

use crate::{paths, Config};

/// `games.ini`, the file the layer reads (next to `config.ini`).
pub fn games_file() -> String {
    games::games_file()
}

/// Every section of `games.ini`, for listing and showing: none when it is missing or cannot be read.
pub fn load_all() -> BTreeMap<String, Section> {
    games::read_user()
}

/// The sections to rewrite `games.ini` from: none only when there is no file. A file that cannot be read,
/// or is not UTF-8, is an error, so a change never replaces it with less than it holds.
fn load_for_update() -> std::io::Result<BTreeMap<String, Section>> {
    let file = games_file();
    let bytes = match std::fs::read(&file) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(e) => return Err(std::io::Error::new(e.kind(), format!("{file} cannot be read ({e}); not replacing it, which would lose the settings it holds"))),
    };
    String::from_utf8(bytes).map(|text| games::parse(&text)).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{file} is not UTF-8 (byte {}); not replacing it, which would change the settings it holds: fix that line by hand",
                e.utf8_error().valid_up_to()
            ),
        )
    })
}

fn save_all(sections: &BTreeMap<String, Section>) -> std::io::Result<()> {
    paths::ensure_dirs()?;
    paths::write_atomic(std::path::Path::new(&games_file()), games::render(sections).as_bytes(), 0o644)
}

/// The name of `game`'s section in `sections`: the one already there for it, else its key's text.
fn section_name(sections: &BTreeMap<String, Section>, game: &GameKey) -> String {
    sections.keys().find(|name| GameKey::parse(name).as_ref() == Some(game)).cloned().unwrap_or_else(|| game.text())
}

/// `game`'s own settings in `sections` (its section's `set_` lines).
pub fn overrides(sections: &BTreeMap<String, Section>, game: &GameKey) -> Section {
    games::user_section(sections, game)
        .map(|s| s.iter().filter(|(k, _)| k.starts_with("set_")).map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

/// Every game with settings of its own, the user's and the shipped, each once.
pub fn games_with_settings(sections: &BTreeMap<String, Section>) -> Vec<GameKey> {
    let mut keys: Vec<GameKey> = sections.keys().filter_map(|name| GameKey::parse(name)).filter(|game| games::has_entry(sections, game)).collect();
    keys.extend(games::SHIPPED.iter().filter_map(|(name, _)| GameKey::parse(name)));
    keys.sort();
    keys.dedup();
    keys
}

/// What `game` runs with where it has no setting of its own: the defaults, `config.ini`'s settings and the
/// overrides shipped for it.
fn without_own(game: &GameKey, cfg: &Config) -> BTreeMap<String, String> {
    let mut base = neural_forge_protocol::persist::defaults();
    base.extend(cfg.settings.iter().map(|(k, v)| (k.clone(), v.clone())));
    games::merged(&base, Some(game), &BTreeMap::new())
}

/// Saves `values` (`set_<name>` -> value) as `game`'s own: a value equal to what it gets without it
/// removes its line instead. Every other line and section of `games.ini` is kept.
pub fn store(game: &GameKey, values: &BTreeMap<String, String>) -> std::io::Result<()> {
    let cfg = Config::load();
    if let Some(why) = cfg.read_error() {
        return Err(std::io::Error::other(format!("{why}: the settings for all games are needed to tell what differs from them")));
    }
    let base = without_own(game, &cfg);
    let mut all = load_for_update()?;
    let name = section_name(&all, game);
    let section = all.entry(name.clone()).or_default();
    for (k, v) in values {
        if base.get(k).is_some_and(|b| games::same_value(b, v)) {
            section.remove(k);
        } else {
            section.insert(k.clone(), v.clone());
        }
    }
    if section.is_empty() {
        all.remove(&name);
    }
    save_all(&all)
}

/// Removes `game`'s own settings: every one (`key` `None`, "Restore defaults"), or the one `key`
/// (`set_<name>`). Lines that are not settings are kept. Returns how many were removed.
pub fn clear(game: &GameKey, key: Option<&str>) -> std::io::Result<usize> {
    let mut all = load_for_update()?;
    let name = section_name(&all, game);
    let Some(section) = all.get_mut(&name) else { return Ok(0) };
    let before = section.len();
    section.retain(|k, _| !(k.starts_with("set_") && key.is_none_or(|key| key == k)));
    let removed = before - section.len();
    if section.is_empty() {
        all.remove(&name);
    }
    if removed > 0 {
        save_all(&all)?;
    }
    Ok(removed)
}

/// The game whose settings `header` holds (`settings_game`), `None` for all games'. An executable is
/// named from `games.ini`, the shipped table and the game on the channel (`game_name`).
pub fn held_game(header: &neural_forge_protocol::ShmHeader) -> Option<GameKey> {
    let also: Vec<GameKey> = GameKey::exe(&header.game_name()).into_iter().collect();
    games::key_for_scope(header.settings_game(), &load_all(), &also)
}

/// Saves the live header's settings where they belong: `config.ini` while it holds the settings for all
/// games, else that game's own ([`store`]). `names`: only these settings (the GUI's changed rows), `None`
/// every one (a profile, Reset, `shmctl`).
pub fn save_live(header: &neural_forge_protocol::ShmHeader, names: Option<&[&str]>) -> std::io::Result<()> {
    let snapshot = neural_forge_protocol::persist::snapshot(header);
    let chosen: BTreeMap<String, String> = match names {
        None => snapshot,
        Some(names) => snapshot.into_iter().filter(|(k, _)| k.strip_prefix("set_").is_some_and(|n| names.contains(&n))).collect(),
    };
    match header.settings_game() {
        0 => {
            let mut cfg = Config::load();
            if names.is_some() {
                cfg.settings.extend(chosen);
            } else {
                cfg.replace_tuning(chosen);
            }
            cfg.save()
        }
        scope => match held_game(header) {
            Some(game) => store(&game, &chosen),
            None => Err(std::io::Error::other(format!("the channel holds the settings of a game {} does not name (scope {scope:#x})", games_file()))),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `XDG_CONFIG_HOME` (and data and state) in a scratch dir, under the crate's one lock for env changes.
    struct Home {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Vec<(&'static str, Option<String>)>,
        dir: std::path::PathBuf,
    }

    impl Home {
        fn new(tag: &str) -> Self {
            let guard = crate::paths::tests::XDG_DATA_HOME_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let dir = std::env::temp_dir().join(format!("neural-forge-game-settings-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let mut prev = Vec::new();
            for (var, sub) in [("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state")] {
                prev.push((var, std::env::var(var).ok()));
                std::env::set_var(var, dir.join(sub));
            }
            std::fs::create_dir_all(dir.join("config/neural-forge")).unwrap();
            Self { _guard: guard, prev, dir }
        }

        fn write(&self, name: &str, text: &str) {
            std::fs::write(self.dir.join("config/neural-forge").join(name), text).unwrap();
        }

        fn read(&self, name: &str) -> String {
            std::fs::read_to_string(self.dir.join("config/neural-forge").join(name)).unwrap_or_default()
        }
    }

    impl Drop for Home {
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

    fn values(lines: &[(&str, &str)]) -> BTreeMap<String, String> {
        lines.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn a_game_keeps_only_what_differs_and_unknown_lines_stay() {
        let home = Home::new("store");
        home.write("config.ini", "set_intensity=1.5\n");
        home.write("games.ini", "# mine\nfuture=1\n[570]\nset_style=2\nnote=kept\n[exe:Other.exe]\nset_unknown_setting=3\n");
        let dota = GameKey::Steam(570);
        // 1.5 is all games' intensity, 1 the default local tone: neither is stored; 0.5 is.
        store(&dota, &values(&[("set_intensity", "1.50"), ("set_local_tone", "1"), ("set_skin_structure", "0.5")])).unwrap();
        let all = load_all();
        assert_eq!(overrides(&all, &dota), values(&[("set_skin_structure", "0.5"), ("set_style", "2")]));
        assert_eq!(all["570"]["note"], "kept");
        assert_eq!(all[""]["future"], "1", "a line before the first section is kept");
        assert_eq!(all["exe:Other.exe"]["set_unknown_setting"], "3", "a setting this build does not know is kept");
        // Saving the value it would get anyway takes the line out.
        store(&dota, &values(&[("set_style", "0")])).unwrap();
        assert_eq!(overrides(&load_all(), &dota), values(&[("set_skin_structure", "0.5")]));
        assert_eq!(games_with_settings(&load_all()), [GameKey::Steam(570), GameKey::Exe("other.exe".into())]);
        // Restore defaults: its settings go, its other lines stay.
        assert_eq!(clear(&dota, None).unwrap(), 1);
        assert!(overrides(&load_all(), &dota).is_empty());
        assert_eq!(load_all()["570"]["note"], "kept");
        assert_eq!(clear(&dota, None).unwrap(), 0);
        assert_eq!(clear(&GameKey::Exe("other.exe".into()), Some("set_unknown_setting")).unwrap(), 1, "found by its key whatever the case");
        assert!(!home.read("games.ini").contains("Other.exe"), "an emptied section goes:\n{}", home.read("games.ini"));
    }

    /// A sharpening change made while a game is picked is saved as that game's own, like any other setting, and
    /// all games' stay as they were.
    #[test]
    fn a_sharpening_change_for_a_game_is_saved_as_its_own() {
        let home = Home::new("save-sharpen");
        home.write("config.ini", "set_clarity_radius=24\n");
        let h = neural_forge_protocol::ShmHeader::default();
        h.init_defaults();
        h.write_game_settings(570, &[("sharpen_cas", 0.5f32.to_bits()), ("clarity_strength", 0.4f32.to_bits()), ("clarity_radius", 24.0f32.to_bits())], None);
        save_live(&h, Some(&["sharpen_cas", "clarity_strength", "clarity_radius"])).unwrap();
        assert_eq!(
            overrides(&load_all(), &GameKey::Steam(570)),
            values(&[("set_clarity_strength", "0.4"), ("set_sharpen_cas", "0.5")]),
            "all games' radius is not stored"
        );
        let cfg = Config::load();
        assert!(!cfg.settings.contains_key("set_sharpen_cas") && !cfg.settings.contains_key("set_clarity_strength"));
        assert_eq!(clear(&GameKey::Steam(570), Some("set_sharpen_cas")).unwrap(), 1, "one override dropped (the row's undo)");
        assert_eq!(overrides(&load_all(), &GameKey::Steam(570)), values(&[("set_clarity_strength", "0.4")]));
    }

    #[test]
    fn a_change_on_the_channel_is_saved_where_its_settings_belong() {
        let home = Home::new("save-live");
        home.write("config.ini", "binaries=/b\nset_intensity=1.5\n");
        let h = neural_forge_protocol::ShmHeader::default();
        h.init_defaults();
        // All games': config.ini, only the named setting.
        h.write_settings(&[("intensity", 2.0f32.to_bits()), ("style", 1)]);
        save_live(&h, Some(&["intensity"])).unwrap();
        let cfg = Config::load();
        assert_eq!(cfg.settings.get("set_intensity").map(String::as_str), Some("2"));
        assert!(!cfg.settings.contains_key("set_style"), "only what was named");
        assert!(home.read("games.ini").is_empty());
        // A game's: its own section, config.ini untouched.
        h.write_game_settings(570, &[("intensity", 0.5f32.to_bits())], None);
        save_live(&h, None).unwrap();
        assert_eq!(overrides(&load_all(), &GameKey::Steam(570)), values(&[("set_intensity", "0.5"), ("set_style", "1")]));
        assert_eq!(Config::load().settings.get("set_intensity").map(String::as_str), Some("2"));
        assert_eq!(Config::load().binaries, "/b");
        // An executable's, named by the game on the channel.
        let exe = GameKey::Exe("game.exe".into());
        h.set_game_name("Game.exe");
        h.write_game_settings(exe.scope(), &[], None);
        save_live(&h, Some(&["intensity"])).unwrap();
        assert_eq!(overrides(&load_all(), &exe), values(&[("set_intensity", "0.5")]));
        assert_eq!(held_game(&h), Some(exe));
        // One nothing names cannot be saved.
        h.write_game_settings(GameKey::Exe("unknown.exe".into()).scope(), &[], None);
        assert!(save_live(&h, None).is_err());
    }

    #[test]
    fn a_file_that_cannot_be_read_or_is_not_utf8_is_never_rewritten() {
        let home = Home::new("unreadable");
        let held = b"[570]\nset_style=1\n[exe:caf\xe9.exe]\nset_style=2\n";
        std::fs::write(home.dir.join("config/neural-forge/games.ini"), held).unwrap();
        assert_eq!(store(&GameKey::Steam(570), &values(&[("set_style", "2")])).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
        assert!(clear(&GameKey::Steam(570), None).is_err());
        assert_eq!(std::fs::read(home.dir.join("config/neural-forge/games.ini")).unwrap(), held);
    }
}
