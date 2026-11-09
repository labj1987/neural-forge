//! Per-game settings: `games.ini` next to `config.ini`, one `[<game>]` section per game holding only the
//! `set_<name>=<value>` lines that differ from the settings for all games (`config.ini`'s), and the
//! overrides Neural Forge ships for games it knows need them ([`SHIPPED`]). A game is its Steam app id
//! (`[1174180]`), else its executable's name (`[exe:game.exe]`, lowercase): [`GameKey`].
//!
//! The merge, lowest first: the defaults, `config.ini`, the shipped overrides, the user's (`games.ini`).
//! Keys a game does not override follow `config.ini`, so a later change for all games still reaches it.
//! The layer applies the running game's merge when its session on the channel starts
//! ([`crate::persist::apply_for_game`]); the header's `settings_game` says whose settings it holds, so the
//! GUI and the CLI save a change to that game's section instead of `config.ini`.

use std::collections::BTreeMap;

/// One section's lines (`set_<name>` -> value, and any other key, kept as read).
pub type Section = BTreeMap<String, String>;

/// Overrides shipped for games that need them, by [`GameKey::text`]: `(game, &[(setting key, value)])`.
/// User overrides win over these. Empty: no game in the tested-games list needs a setting of its own
/// (docs/USER_GUIDE.md, "Tested games"; each runs with the defaults).
pub const SHIPPED: &[(&str, &[(&str, &str)])] = &[];

/// Marks an executable's scope ([`GameKey::scope`]): Steam app ids stay below it.
const EXE_SCOPE: u32 = 0x8000_0000;

/// A game as the per-game settings name it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GameKey {
    /// A Steam app id (1 to 2^31 - 1).
    Steam(u32),
    /// A game outside Steam, by its executable's file name, lowercase (Windows names ignore case).
    Exe(String),
}

impl GameKey {
    /// `1174180`, or `exe:<name>` (any case; stored lowercase). `None` for anything else, an app id of
    /// 0 or one at or above 2^31 (a non-Steam shortcut's id), and a name `games.ini` cannot hold.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        if let Some(name) = text.strip_prefix("exe:") {
            return Self::exe(name);
        }
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        text.parse::<u32>().ok().filter(|&id| id != 0 && id < EXE_SCOPE).map(Self::Steam)
    }

    /// The key for executable `name`: its file name only (a path's directories dropped), lowercase.
    /// `None` for an empty name or one holding a character a section name cannot (`[`, `]`, `=`, `#`,
    /// a control character).
    pub fn exe(name: &str) -> Option<Self> {
        let name = name.trim().rsplit(['/', '\\']).next().unwrap_or_default().to_lowercase();
        let bad = |c: char| c.is_control() || matches!(c, '[' | ']' | '=' | '#');
        (!name.is_empty() && !name.chars().any(bad)).then_some(Self::Exe(name))
    }

    /// The running game's key: its Steam app id when Steam started it (`steam_app_id`, the `SteamAppId`
    /// Steam puts in a game's environment), else its executable (`exe`, the target filter's name of the
    /// process: the Windows `.exe` for a Proton game).
    pub fn identify(steam_app_id: Option<&str>, exe: &str) -> Option<Self> {
        steam_app_id
            .and_then(|id| match Self::parse(id) {
                Some(key @ Self::Steam(_)) => Some(key),
                _ => None,
            })
            .or_else(|| Self::exe(exe))
    }

    /// The text `games.ini` names it by: `1174180` or `exe:game.exe`.
    pub fn text(&self) -> String {
        match self {
            Self::Steam(id) => id.to_string(),
            Self::Exe(name) => format!("exe:{name}"),
        }
    }

    /// The header's word for it (`settings_game`, `running_game`): the app id, or for an executable
    /// 2^31 plus 31 bits of its name's FNV-1a hash. Never 0, which means all games.
    pub fn scope(&self) -> u32 {
        match self {
            Self::Steam(id) => *id,
            Self::Exe(name) => EXE_SCOPE | (fnv1a(name.as_bytes()) & !EXE_SCOPE),
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

/// The running game's key, read in the game's process: `SteamAppId`, else Proton's
/// `STEAM_COMPAT_APP_ID`, else the executable `exe`.
pub fn this_game(exe: &str) -> Option<GameKey> {
    let id =
        ["SteamAppId", "STEAM_COMPAT_APP_ID"].into_iter().filter_map(|v| std::env::var(v).ok()).find(|v| matches!(GameKey::parse(v), Some(GameKey::Steam(_))));
    GameKey::identify(id.as_deref(), exe)
}

/// `games.ini`: next to [`crate::persist::config_file`].
pub fn games_file() -> String {
    let config = crate::persist::config_file();
    match config.rsplit_once('/') {
        Some((dir, _)) => format!("{dir}/games.ini"),
        None => "games.ini".to_string(),
    }
}

/// Every section of `text`, by its name as written (the lines before the first section under `""`).
/// The same line rules as `config.ini`: trimmed, blank lines and `#` comments skipped, the first `=`
/// splits, a later line for the same key wins.
pub fn parse(text: &str) -> BTreeMap<String, Section> {
    let mut sections: BTreeMap<String, Section> = BTreeMap::new();
    let mut current = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
            current = name.trim().to_string();
            sections.entry(current.clone()).or_default();
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            sections.entry(current.clone()).or_default().insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    sections
}

/// The file's text for `sections`: the unnamed one's lines first, then each section; every key kept.
pub fn render(sections: &BTreeMap<String, Section>) -> String {
    let mut text = String::new();
    if let Some(lines) = sections.get("") {
        for (k, v) in lines {
            text.push_str(&format!("{k}={v}\n"));
        }
        if !lines.is_empty() {
            text.push('\n');
        }
    }
    for (name, lines) in sections.iter().filter(|(name, _)| !name.is_empty()) {
        text.push_str(&format!("[{name}]\n"));
        for (k, v) in lines {
            text.push_str(&format!("{k}={v}\n"));
        }
        text.push('\n');
    }
    text
}

/// `games.ini` as [`parse`] reads it; empty when it is missing or cannot be read (a byte that is not
/// UTF-8 reads as U+FFFD).
pub fn read_user() -> BTreeMap<String, Section> {
    std::fs::read(games_file()).map(|bytes| parse(&String::from_utf8_lossy(&bytes))).unwrap_or_default()
}

/// The user's section for `game` in `sections`: the one whose name parses to the same key (so
/// `[exe:Game.exe]` written by hand is `exe:game.exe`'s).
pub fn user_section<'a>(sections: &'a BTreeMap<String, Section>, game: &GameKey) -> Option<&'a Section> {
    sections.iter().find(|(name, _)| GameKey::parse(name).as_ref() == Some(game)).map(|(_, s)| s)
}

/// The overrides shipped for `game`, if any.
pub fn shipped(game: &GameKey) -> Option<&'static [(&'static str, &'static str)]> {
    SHIPPED.iter().find(|(name, _)| GameKey::parse(name).as_ref() == Some(game)).map(|&(_, lines)| lines)
}

/// Whether `game` has settings of its own: overrides shipped for it, or a `set_` line in its section.
pub fn has_entry(sections: &BTreeMap<String, Section>, game: &GameKey) -> bool {
    shipped(game).is_some_and(|lines| !lines.is_empty()) || user_section(sections, game).is_some_and(|s| s.keys().any(|k| k.starts_with("set_")))
}

/// The settings `game` runs with: `base` (the defaults and `config.ini`, `set_` lines), then its shipped
/// overrides, then the user's. With `game` `None`, `base` alone. Only `set_` lines are merged.
pub fn merged(base: &BTreeMap<String, String>, game: Option<&GameKey>, sections: &BTreeMap<String, Section>) -> BTreeMap<String, String> {
    let mut all: BTreeMap<String, String> = base.iter().filter(|(k, _)| k.starts_with("set_")).map(|(k, v)| (k.clone(), v.clone())).collect();
    let Some(game) = game else { return all };
    for &(k, v) in shipped(game).unwrap_or_default() {
        all.insert(k.to_string(), v.to_string());
    }
    if let Some(section) = user_section(sections, game) {
        all.extend(section.iter().filter(|(k, _)| k.starts_with("set_")).map(|(k, v)| (k.clone(), v.clone())));
    }
    all
}

/// The key whose [`GameKey::scope`] is `scope`: a Steam app id directly, an executable from the names
/// known for it (the user's sections, the shipped ones, and `also`, such as the running game's).
pub fn key_for_scope(scope: u32, sections: &BTreeMap<String, Section>, also: &[GameKey]) -> Option<GameKey> {
    if scope == 0 {
        return None;
    }
    if scope < EXE_SCOPE {
        return Some(GameKey::Steam(scope));
    }
    sections
        .keys()
        .filter_map(|name| GameKey::parse(name))
        .chain(SHIPPED.iter().filter_map(|(name, _)| GameKey::parse(name)))
        .chain(also.iter().cloned())
        .find(|key| key.scope() == scope)
}

/// Whether two saved values are the same setting value: the same text, or the same number (`1.50` and
/// `1.5`).
pub fn same_value(a: &str, b: &str) -> bool {
    let (a, b) = (a.trim(), b.trim());
    a == b || matches!((a.parse::<f64>(), b.parse::<f64>()), (Ok(x), Ok(y)) if x == y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(lines: &[(&str, &str)]) -> BTreeMap<String, String> {
        lines.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn keys_parse_and_print_back() {
        assert_eq!(GameKey::parse("1174180"), Some(GameKey::Steam(1174180)));
        assert_eq!(GameKey::parse(" 3240220 "), Some(GameKey::Steam(3240220)));
        assert_eq!(GameKey::parse("exe:GTA5_Enhanced.exe"), Some(GameKey::Exe("gta5_enhanced.exe".into())));
        assert_eq!(GameKey::parse("exe:C:\\Games\\X\\Game.exe"), Some(GameKey::Exe("game.exe".into())), "the file name only");
        for bad in ["", "0", "-1", "12a", "2147483648", "4294967296", "exe:", "exe:a]b", "exe:a=b", "steam:5", "[1]"] {
            assert_eq!(GameKey::parse(bad), None, "{bad:?}");
        }
        for key in [GameKey::Steam(570), GameKey::Exe("game.exe".into())] {
            assert_eq!(GameKey::parse(&key.text()), Some(key.clone()));
        }
    }

    #[test]
    fn the_app_id_wins_over_the_executable_and_a_shortcut_id_is_not_one() {
        assert_eq!(GameKey::identify(Some("3240220"), "GTA5_Enhanced.exe"), Some(GameKey::Steam(3240220)));
        assert_eq!(GameKey::identify(Some("0"), "Game.exe"), Some(GameKey::Exe("game.exe".into())), "a non-Steam game started from Steam");
        assert_eq!(GameKey::identify(Some("3000000000"), "Game.exe"), Some(GameKey::Exe("game.exe".into())));
        assert_eq!(GameKey::identify(None, "Game.exe"), Some(GameKey::Exe("game.exe".into())));
        assert_eq!(GameKey::identify(None, ""), None);
    }

    #[test]
    fn scopes_never_collide_with_all_games_or_an_app_id() {
        assert_eq!(GameKey::Steam(570).scope(), 570);
        let exe = GameKey::Exe("game.exe".into()).scope();
        assert!(exe >= EXE_SCOPE, "{exe:#x}");
        assert_eq!(exe, GameKey::exe("GAME.EXE").unwrap().scope(), "case does not matter");
        assert_ne!(exe, GameKey::Exe("other.exe".into()).scope());
        let sections = parse("[exe:Game.exe]\nset_style=1\n");
        assert_eq!(key_for_scope(exe, &sections, &[]), Some(GameKey::Exe("game.exe".into())));
        assert_eq!(key_for_scope(GameKey::Exe("x.exe".into()).scope(), &sections, &[]), None, "an executable nothing names");
        assert_eq!(key_for_scope(GameKey::Exe("x.exe".into()).scope(), &sections, &[GameKey::Exe("x.exe".into())]), Some(GameKey::Exe("x.exe".into())));
        assert_eq!(key_for_scope(570, &sections, &[]), Some(GameKey::Steam(570)));
        assert_eq!(key_for_scope(0, &sections, &[]), None);
    }

    #[test]
    fn a_game_takes_its_overrides_and_the_rest_from_all_games() {
        let base = map(&[("set_intensity", "1.5"), ("set_style", "0"), ("binaries", "/b")]);
        let sections = parse("[570]\nset_style=2\nnote=kept\n[exe:other.exe]\nset_intensity=0.5\n");
        let dota = GameKey::Steam(570);
        let all = merged(&base, Some(&dota), &sections);
        assert_eq!(all, map(&[("set_intensity", "1.5"), ("set_style", "2")]), "only set_ lines, the override on top");
        assert_eq!(merged(&base, None, &sections), map(&[("set_intensity", "1.5"), ("set_style", "0")]));
        assert_eq!(merged(&base, Some(&GameKey::Steam(10)), &sections), merged(&base, None, &sections), "no entry: all games' settings");
        assert!(has_entry(&sections, &dota));
        assert!(!has_entry(&sections, &GameKey::Steam(10)));
        assert!(!has_entry(&parse("[10]\nnote=1\n"), &GameKey::Steam(10)), "a section without a setting is no entry");
    }

    #[test]
    fn the_shipped_table_names_real_games_and_settings() {
        let h = crate::ShmHeader::default();
        h.init_defaults();
        let snapshot = crate::persist::snapshot(&h);
        for (game, lines) in SHIPPED {
            assert!(GameKey::parse(game).is_some(), "{game}");
            for (k, _) in *lines {
                assert!(snapshot.contains_key(*k), "{game}: {k} is not a setting");
            }
        }
    }

    #[test]
    fn the_file_round_trips_with_unknown_keys_and_sections() {
        let text = "top=1\n# comment\n[570]\nset_style=2\nfuture_key=x\n\n[not a game]\nset_style=1\n[exe:Game.exe]\nset_intensity = 0.5\n";
        let sections = parse(text);
        assert_eq!(sections[""]["top"], "1");
        assert_eq!(sections["570"]["future_key"], "x");
        assert_eq!(sections["exe:Game.exe"]["set_intensity"], "0.5");
        assert!(sections.contains_key("not a game"));
        assert_eq!(parse(&render(&sections)), sections, "every section and key comes back");
    }

    #[test]
    fn values_compare_as_numbers() {
        assert!(same_value("1.5", "1.50"));
        assert!(same_value("2", "2.0"));
        assert!(same_value("0.02,0.85,0.25,0.12,keep", "0.02,0.85,0.25,0.12,keep"));
        assert!(!same_value("1.5", "1.25"));
        assert!(!same_value("", "0"));
    }
}
