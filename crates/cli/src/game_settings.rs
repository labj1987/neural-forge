//! `neural-forge-cli game-settings`: a game's own settings (`games.ini`, `neural_forge_supervisor::game_settings`),
//! which it runs with over the settings for all games. `set` takes a value the way `shmctl set` does.

use std::process::ExitCode;

use neural_forge_supervisor::game_settings::{self, GameKey};

fn usage_text() -> &'static str {
    "usage: neural-forge-cli game-settings <list|show|set|clear>\n\n\
     \x20 list                    every game with settings of its own (yours and the ones\n\
     \x20                         Neural Forge has built in) and what they change\n\
     \x20 show GAME               GAME's own settings, and what it runs with for them\n\
     \x20 set GAME NAME VALUE     give GAME its own value of setting NAME (the names and\n\
     \x20                         values `shmctl set` takes); a value equal to the settings\n\
     \x20                         for all games removes it instead\n\
     \x20 clear GAME [NAME]       remove GAME's own settings (all of them: restore its\n\
     \x20                         defaults; or NAME): it follows the settings for all games\n\n\
     GAME: a Steam app id, exe:<file.exe> for a game outside Steam, or an installed Steam\n\
     game's name (exact, or a unique part; quote a name with spaces). A game without\n\
     settings of its own uses the settings for all games (config.ini); one with them uses\n\
     those over them, and later changes for all games still reach it for every other setting.\n\
     They live in $XDG_CONFIG_HOME/neural-forge/games.ini. The game's layer applies them when\n\
     the game starts; while the channel holds GAME's settings (it runs, or the app shows\n\
     them), `set` and `clear` apply at once."
}

pub fn usage() {
    eprintln!("{}", usage_text());
}

/// Checks the arguments' shape before anything runs.
pub fn check_args(rest: &[&str]) -> Result<(), String> {
    match rest {
        [] | ["list"] | ["show", _] | ["set", _, _, _] | ["clear", _] | ["clear", _, _] => Ok(()),
        _ => Err(format!("unexpected arguments: {}", rest.join(" "))),
    }
}

/// The game `query` names: a key as `games.ini` writes it (`570`, `exe:game.exe`), else an installed Steam game.
fn find_game(query: &str) -> Result<(GameKey, String), String> {
    use neural_forge_supervisor::steam::{self, FindError};
    let games = steam::games(&steam::roots());
    if let Some(key) = GameKey::parse(query) {
        let name = match &key {
            GameKey::Steam(id) => games.iter().find(|g| g.appid == id.to_string()).map_or_else(|| format!("Steam app {id}"), |g| g.name.clone()),
            GameKey::Exe(name) => name.clone(),
        };
        return Ok((key, name));
    }
    if query.trim().starts_with("exe:") {
        return Err(format!("{query:?} is not an executable's name"));
    }
    match steam::find(&games, query) {
        Ok(game) => {
            GameKey::parse(&game.appid).map(|key| (key, game.name.clone())).ok_or_else(|| format!("{}'s app id {:?} cannot be used", game.name, game.appid))
        }
        Err(FindError::NotFound) => Err(format!("no installed Steam game matches {query:?}: give its app id, or exe:<file.exe> for a game outside Steam")),
        Err(FindError::Ambiguous(candidates)) => {
            let list: Vec<String> = candidates.iter().map(|(name, appid)| format!("  {appid:>8}  {name}")).collect();
            Err(format!("{query:?} matches more than one game; use the app id or more of the name:\n{}", list.join("\n")))
        }
    }
}

/// Applies `game`'s settings to the channel at once while it holds them (`settings_game`). Only opens a
/// channel that is there: with none, the game's layer applies them when it starts.
fn apply_live(game: &GameKey) {
    let cfg = neural_forge_supervisor::Config::load();
    let path = neural_forge_supervisor::channel_path(&cfg);
    if !std::path::Path::new(&path).exists() {
        return;
    }
    match neural_forge_supervisor::open_channel(&cfg) {
        Ok(mapping) if mapping.header().settings_game() == game.scope() => {
            neural_forge_protocol::persist::apply_scope(mapping.header(), Some(game));
            println!("applied to the live channel");
        }
        Ok(_) => {}
        Err(e) => eprintln!("game-settings: the live channel was not changed: {e}: {path}"),
    }
}

fn cmd_list() -> ExitCode {
    let sections = game_settings::load_all();
    let games = game_settings::games_with_settings(&sections);
    if games.is_empty() {
        println!("no game has settings of its own ({})", game_settings::games_file());
        return ExitCode::SUCCESS;
    }
    for game in games {
        let mut lines: Vec<String> = game_settings::overrides(&sections, &game).iter().map(|(k, v)| format!("{}={v}", k.trim_start_matches("set_"))).collect();
        for (k, v) in neural_forge_protocol::games::shipped(&game).unwrap_or_default() {
            lines.push(format!("{}={v} (built in)", k.trim_start_matches("set_")));
        }
        println!("{}  {}", game.text(), lines.join(" "));
    }
    ExitCode::SUCCESS
}

fn cmd_show(query: &str) -> ExitCode {
    let (game, name) = match find_game(query) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("game-settings show: {e}");
            return ExitCode::FAILURE;
        }
    };
    let own = game_settings::overrides(&game_settings::load_all(), &game);
    let shipped = neural_forge_protocol::games::shipped(&game).unwrap_or_default();
    println!("{name} ({})", game.text());
    if own.is_empty() && shipped.is_empty() {
        println!("  no settings of its own: it uses the settings for all games");
        return ExitCode::SUCCESS;
    }
    for (k, v) in &own {
        println!("  {}={v}", k.trim_start_matches("set_"));
    }
    for (k, v) in shipped.iter().filter(|(k, _)| !own.contains_key(*k)) {
        println!("  {}={v} (built in)", k.trim_start_matches("set_"));
    }
    ExitCode::SUCCESS
}

fn cmd_set(query: &str, name: &str, value: &str) -> ExitCode {
    let (game, title) = match find_game(query) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("game-settings set: {e}");
            return ExitCode::FAILURE;
        }
    };
    // The value as `shmctl set` reads it, written to a scratch header and read back as `config.ini` stores
    // it (clamped, a placement's name as its number).
    let scratch = neural_forge_protocol::ShmHeader::default();
    scratch.init_defaults();
    let parsed = match crate::shmctl::parse_value(&scratch, name, value) {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("game-settings set: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !crate::shmctl::write_value(&scratch, name, &parsed) {
        eprintln!("game-settings set: {name} was not stored");
        return ExitCode::FAILURE;
    }
    let key = format!("set_{name}");
    let Some(text) = neural_forge_protocol::persist::snapshot(&scratch).remove(&key) else {
        eprintln!("game-settings set: unknown setting {name:?}");
        return ExitCode::FAILURE;
    };
    if let Err(e) = game_settings::store(&game, &std::collections::BTreeMap::from([(key.clone(), text.clone())])) {
        eprintln!("game-settings set: saving {} failed: {e}", game_settings::games_file());
        return ExitCode::FAILURE;
    }
    if game_settings::overrides(&game_settings::load_all(), &game).contains_key(&key) {
        println!("{title} ({}): {name}={text}", game.text());
    } else {
        println!("{title} ({}): {name}={text} is what it gets without a setting of its own: it follows the settings for all games", game.text());
    }
    apply_live(&game);
    ExitCode::SUCCESS
}

fn cmd_clear(query: &str, name: Option<&str>) -> ExitCode {
    let (game, title) = match find_game(query) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("game-settings clear: {e}");
            return ExitCode::FAILURE;
        }
    };
    let key = name.map(|n| format!("set_{n}"));
    match game_settings::clear(&game, key.as_deref()) {
        Ok(0) => {
            println!("{title} ({}) has no {}: nothing to clear", game.text(), name.map_or("settings of its own".to_string(), |n| format!("{n} of its own")))
        }
        Ok(n) => {
            println!("{title} ({}): removed {n} setting{} of its own", game.text(), if n == 1 { "" } else { "s" });
            apply_live(&game);
        }
        Err(e) => {
            eprintln!("game-settings clear: saving {} failed: {e}", game_settings::games_file());
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

pub fn run(args: &[String]) -> ExitCode {
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        [] => {
            usage();
            ExitCode::SUCCESS
        }
        ["list"] => cmd_list(),
        ["show", game] => cmd_show(game),
        ["set", game, name, value] => cmd_set(game, name, value),
        ["clear", game] => cmd_clear(game, None),
        ["clear", game, name] => cmd_clear(game, Some(name)),
        _ => {
            usage();
            ExitCode::FAILURE
        }
    }
}
