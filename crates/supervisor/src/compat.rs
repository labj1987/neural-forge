//! Per-game compatibility facts: `data/games.toml`, built into the binaries. Each tested game's engine, graphics
//! API, the DLSS features it was run with, where the model runs in it (with Ray Reconstruction off and on), how its
//! HDR reaches the screen, its known issues and the version it was last seen working in. The Status tab, the Setup
//! tab's games list, `neural-forge-cli games`, `doctor` and Save report show them; the pre-release sweep
//! (`scripts/game-sweep.sh`) reads what to expect from them.
//!
//! The file is a small subset of TOML ([`tables`]): `[[name]]` array-of-tables headers, `key = value` lines with
//! basic or literal strings, integers, booleans and arrays of those (over several lines too), and `#` comments.
//! Nothing else is needed and the file stays valid TOML.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The built-in facts, as committed.
pub const SOURCE: &str = include_str!("../../../data/games.toml");

/// A value of the TOML subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One `[[header]]` table's keys.
pub type Table = BTreeMap<String, Value>;

/// Every `[[header]]` table in `text`, in order. `Err` names the line of the first thing outside the subset: a
/// key before the first header, another header, a key given twice, or a value that does not parse.
pub fn tables(text: &str, header: &str) -> Result<Vec<Table>, String> {
    let mut out: Vec<Table> = Vec::new();
    let want = format!("[[{header}]]");
    let mut lines = text.lines().enumerate().peekable();
    while let Some((n, raw)) = lines.next() {
        let line = strip_comment(raw).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if line != want {
                return Err(format!("line {}: only {want} tables are expected, not {line}", n + 1));
            }
            out.push(Table::new());
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { return Err(format!("line {}: expected key = value", n + 1)) };
        let key = key.trim();
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') {
            return Err(format!("line {}: {key:?} is not a bare key", n + 1));
        }
        let mut value = value.trim().to_string();
        // An array may go on over the next lines until its brackets close.
        while value.starts_with('[') && !brackets_closed(&value) {
            let Some((_, more)) = lines.next() else { return Err(format!("line {}: the array is not closed", n + 1)) };
            value.push(' ');
            value.push_str(strip_comment(more).trim());
        }
        let parsed = parse_value(&value).map_err(|why| format!("line {}: {key}: {why}", n + 1))?;
        let Some(table) = out.last_mut() else { return Err(format!("line {}: {key} comes before the first {want}", n + 1)) };
        if table.insert(key.to_string(), parsed).is_some() {
            return Err(format!("line {}: {key} is given twice", n + 1));
        }
    }
    Ok(out)
}

/// The line up to a `#` outside a string.
fn strip_comment(line: &str) -> &str {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        match quote {
            Some('"') if escaped => escaped = false,
            Some('"') if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '#' => return &line[..i],
            None => {}
        }
    }
    line
}

/// Whether every `[` outside strings in `text` is closed.
fn brackets_closed(text: &str) -> bool {
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for c in text.chars() {
        match quote {
            Some('"') if escaped => escaped = false,
            Some('"') if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '[' => depth += 1,
            None if c == ']' => depth -= 1,
            None => {}
        }
    }
    depth <= 0
}

fn parse_value(text: &str) -> Result<Value, String> {
    let (value, rest) = parse_one(text.trim())?;
    if !rest.trim().is_empty() {
        return Err(format!("unexpected {:?} after the value", rest.trim()));
    }
    Ok(value)
}

/// One value at the start of `text`, and what follows it.
fn parse_one(text: &str) -> Result<(Value, &str), String> {
    if let Some(rest) = text.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.char_indices();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => return Ok((Value::Str(out), &rest[i + 1..])),
                '\\' => match chars.next().map(|(_, e)| e) {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    other => return Err(format!("unsupported escape \\{}", other.map(String::from).unwrap_or_default())),
                },
                c => out.push(c),
            }
        }
        return Err("the string is not closed".into());
    }
    if let Some(rest) = text.strip_prefix('\'') {
        let end = rest.find('\'').ok_or("the string is not closed")?;
        return Ok((Value::Str(rest[..end].to_string()), &rest[end + 1..]));
    }
    if let Some(mut rest) = text.strip_prefix('[') {
        let mut items = Vec::new();
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix(']') {
                return Ok((Value::Array(items), after));
            }
            let (item, after) = parse_one(rest)?;
            items.push(item);
            rest = after.trim_start();
            if let Some(after) = rest.strip_prefix(',') {
                rest = after;
            } else if !rest.starts_with(']') {
                return Err("expected , or ] in the array".into());
            }
        }
    }
    let end = text.find([',', ']', ' ']).unwrap_or(text.len());
    let (word, rest) = text.split_at(end);
    match word {
        "true" => Ok((Value::Bool(true), rest)),
        "false" => Ok((Value::Bool(false), rest)),
        _ => word.replace('_', "").parse::<i64>().map(|n| (Value::Int(n), rest)).map_err(|_| format!("{word:?} is not a string, integer, boolean or array")),
    }
}

/// Where the model runs in a game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Before the upscaler, in front of DLSS's command buffer (the split hold).
    Before,
    /// Before the upscaler, inside DLSS's own command buffer.
    BeforeInside,
    /// On DLSS's output, before the game's HUD (Ray Reconstruction).
    AfterDlss,
    /// On the presented frame.
    FinishedFrame,
}

impl Placement {
    pub const KEYS: [&'static str; 4] = ["before", "before-inside", "after-dlss", "finished-frame"];

    pub fn parse(key: &str) -> Option<Self> {
        Some(match key {
            "before" => Self::Before,
            "before-inside" => Self::BeforeInside,
            "after-dlss" => Self::AfterDlss,
            "finished-frame" => Self::FinishedFrame,
            _ => return None,
        })
    }

    pub fn key(self) -> &'static str {
        Self::KEYS[self as usize]
    }

    /// As the Status tab's Model placement line says it.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Before => "before the upscaler",
            Self::BeforeInside => "before the upscaler, inside DLSS's buffer",
            Self::AfterDlss => "after DLSS",
            Self::FinishedFrame => "on the finished frame",
        }
    }

    /// The `preupscale_state` (`neural_forge_protocol::enums::preupscale_state`) a game running this way settles in.
    pub fn preupscale_state(self) -> u32 {
        use neural_forge_protocol::enums::preupscale_state::*;
        match self {
            Self::Before | Self::BeforeInside => HOLDING,
            Self::AfterDlss => HOLDING_AFTER,
            Self::FinishedFrame => OFF,
        }
    }
}

/// What a game can present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hdr {
    /// HDR10 (PQ), with the game's HDR setting on.
    Hdr10,
    /// SDR only.
    None,
    /// Not tested.
    Unknown,
}

impl Hdr {
    pub fn parse(key: &str) -> Option<Self> {
        Some(match key {
            "hdr10" => Self::Hdr10,
            "none" => Self::None,
            "unknown" => Self::Unknown,
            _ => return None,
        })
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Hdr10 => "hdr10",
            Self::None => "none",
            Self::Unknown => "unknown",
        }
    }
}

/// One game's facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Game {
    pub name: String,
    pub steam_appid: Option<u32>,
    pub exe: Vec<String>,
    pub engine: String,
    pub api: String,
    pub dlss: String,
    pub placement: Placement,
    /// With DLSS Ray Reconstruction on; `None` for a game without it.
    pub placement_rr: Option<Placement>,
    pub hdr: Hdr,
    pub hdr_route: Option<String>,
    pub notes: Vec<String>,
    /// The version it was last seen working in, and the date; `None` when not recorded.
    pub verified: Option<String>,
    pub verified_on: Option<String>,
    pub verified_by: String,
}

const KEYS: [&str; 14] =
    ["name", "steam_appid", "exe", "engine", "api", "dlss", "placement", "placement_rr", "hdr", "hdr_route", "notes", "verified", "verified_on", "verified_by"];

impl Game {
    fn from_table(t: &Table) -> Result<Self, String> {
        if let Some(unknown) = t.keys().find(|k| !KEYS.contains(&k.as_str())) {
            return Err(format!("unknown key {unknown}"));
        }
        let text = |key: &str| -> Result<Option<String>, String> {
            match t.get(key) {
                None => Ok(None),
                Some(Value::Str(s)) if !s.trim().is_empty() => Ok(Some(s.trim().to_string())),
                Some(_) => Err(format!("{key} must be a non-empty string")),
            }
        };
        let required = |key: &str| text(key)?.ok_or_else(|| format!("{key} is missing"));
        let list = |key: &str| -> Result<Vec<String>, String> {
            match t.get(key) {
                None => Ok(Vec::new()),
                Some(Value::Array(items)) => {
                    items.iter().map(|v| v.as_str().map(str::to_string).filter(|s| !s.trim().is_empty()).ok_or(format!("{key} holds a non-string"))).collect()
                }
                Some(_) => Err(format!("{key} must be an array of strings")),
            }
        };
        let name = required("name")?;
        let in_game = |why: String| format!("{name}: {why}");
        let steam_appid = match t.get("steam_appid") {
            None => None,
            Some(Value::Int(id)) if (1..i64::from(i32::MAX)).contains(id) => Some(*id as u32),
            Some(_) => return Err(in_game("steam_appid must be a Steam app id".into())),
        };
        let placement_of = |key: &str| -> Result<Option<Placement>, String> {
            text(key)?.map(|p| Placement::parse(&p).ok_or_else(|| format!("{key} {p:?} is not one of {}", Placement::KEYS.join(", ")))).transpose()
        };
        let placement = placement_of("placement").map_err(in_game)?.ok_or_else(|| in_game("placement is missing".into()))?;
        let hdr = required("hdr").map_err(in_game)?;
        let hdr = Hdr::parse(&hdr).ok_or_else(|| in_game(format!("hdr {hdr:?} is not hdr10, none or unknown")))?;
        let game = Game {
            steam_appid,
            exe: list("exe").map_err(in_game)?,
            engine: required("engine").map_err(in_game)?,
            api: required("api").map_err(in_game)?,
            dlss: required("dlss").map_err(in_game)?,
            placement,
            placement_rr: placement_of("placement_rr").map_err(in_game)?,
            hdr,
            hdr_route: text("hdr_route").map_err(in_game)?,
            notes: list("notes").map_err(in_game)?,
            verified: text("verified").map_err(in_game)?,
            verified_on: text("verified_on").map_err(in_game)?,
            verified_by: required("verified_by").map_err(in_game)?,
            name: name.clone(),
        };
        if game.hdr == Hdr::Hdr10 && game.hdr_route.is_none() {
            return Err(in_game("an HDR10 game needs its hdr_route".into()));
        }
        Ok(game)
    }

    /// Where the model runs, with and without Ray Reconstruction: "before the upscaler, inside DLSS's buffer; after
    /// DLSS with Ray Reconstruction on".
    pub fn placement_text(&self) -> String {
        match self.placement_rr {
            Some(rr) if rr != self.placement => format!("{}; {} with Ray Reconstruction on", self.placement.describe(), rr.describe()),
            _ => self.placement.describe().to_string(),
        }
    }

    /// "Tested in 3.14.0 on 2026-11-10 (test machine)", or "Tested (test machine; version not recorded)".
    pub fn verified_text(&self) -> String {
        match (&self.verified, &self.verified_on) {
            (Some(v), Some(on)) => format!("Tested in {v} on {on} ({})", self.verified_by),
            (Some(v), None) => format!("Tested in {v} ({})", self.verified_by),
            _ => format!("Tested ({}; version not recorded)", self.verified_by),
        }
    }

    /// One line for the Status tab: "Tested in 3.14.0 on 2026-11-10 (test machine): before the upscaler, ...".
    pub fn status_line(&self) -> String {
        format!("{}: {}", self.verified_text(), self.placement_text())
    }

    /// Every fact, one per line, for `neural-forge-cli games GAME` and Save report.
    pub fn details(&self) -> String {
        let mut s = format!("{}\n", self.name);
        let mut line = |k: &str, v: &str| s.push_str(&format!("  {k}: {v}\n"));
        if let Some(id) = self.steam_appid {
            line("Steam app id", &id.to_string());
        }
        if !self.exe.is_empty() {
            line("executables", &self.exe.join(", "));
        }
        line("engine, API", &format!("{}, {}", self.engine, self.api));
        line("DLSS", &self.dlss);
        line("model placement (Auto)", self.placement.describe());
        if let Some(rr) = self.placement_rr {
            line("with Ray Reconstruction on", rr.describe());
        }
        line(
            "HDR",
            &match (self.hdr, &self.hdr_route) {
                (Hdr::Hdr10, Some(route)) => format!("HDR10. {route}"),
                (Hdr::Hdr10, None) => "HDR10".to_string(),
                (Hdr::None, _) => "none (SDR only)".to_string(),
                (Hdr::Unknown, _) => "not tested".to_string(),
            },
        );
        for note in &self.notes {
            line("note", note);
        }
        line("verified", &self.verified_text());
        s
    }
}

/// The games in `text` (the format of `data/games.toml`). `Err` says what is wrong and where.
pub fn parse(text: &str) -> Result<Vec<Game>, String> {
    let games = tables(text, "game")?
        .iter()
        .enumerate()
        .map(|(i, t)| Game::from_table(t).map_err(|why| format!("game {}: {why}", i + 1)))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, g) in games.iter().enumerate() {
        if games[..i].iter().any(|o| o.name == g.name || (g.steam_appid.is_some() && o.steam_appid == g.steam_appid)) {
            return Err(format!("{} is listed twice", g.name));
        }
    }
    Ok(games)
}

/// The built-in games ([`SOURCE`]); a test makes sure it parses, so a broken file never ships.
pub fn games() -> &'static [Game] {
    static GAMES: OnceLock<Vec<Game>> = OnceLock::new();
    GAMES.get_or_init(|| parse(SOURCE).unwrap_or_default())
}

/// The game with Steam app id `appid`, else the one targeting executable `exe` (any case, a path's directories
/// ignored).
pub fn find_in<'a>(games: &'a [Game], appid: Option<u32>, exe: &str) -> Option<&'a Game> {
    let by_id = appid.and_then(|id| games.iter().find(|g| g.steam_appid == Some(id)));
    let exe = exe.trim().rsplit(['/', '\\']).next().unwrap_or_default();
    by_id.or_else(|| (!exe.is_empty()).then(|| games.iter().find(|g| g.exe.iter().any(|e| e.eq_ignore_ascii_case(exe)))).flatten())
}

/// [`find_in`] over the built-in games.
pub fn find(appid: Option<u32>, exe: &str) -> Option<&'static Game> {
    find_in(games(), appid, exe)
}

/// The running game's entry, from the channel's header: its Steam app id (`running_game`, below 2^31) or its
/// executable (`game_name`).
pub fn for_header(header: &neural_forge_protocol::ShmHeader) -> Option<&'static Game> {
    let scope = header.running_game.load(std::sync::atomic::Ordering::Relaxed);
    let appid = (scope != 0 && scope < 0x8000_0000).then_some(scope);
    find(appid, &header.game_name())
}

/// The game a Steam app id names, as text for Save report: its details, or that it is not in the list.
pub fn report_text(appid: Option<&str>, exe: &str) -> String {
    let id = appid.and_then(|a| a.trim().parse::<u32>().ok());
    match find(id, exe) {
        Some(game) => game.details(),
        None => format!(
            "{} is not in Neural Forge's list of tested games ({} games, built in). Where the model runs and what to set up \
             in it are not known yet.\n",
            match (appid, exe.trim()) {
                (Some(a), "") => format!("Steam app {a}"),
                (_, "") => "The game".to_string(),
                (_, e) => e.to_string(),
            },
            games().len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_list_parses_and_names_every_tested_game() {
        let games = parse(SOURCE).unwrap_or_else(|e| panic!("data/games.toml: {e}"));
        assert!(games.len() >= 15, "{}", games.len());
        for name in
            ["GTA V Enhanced", "Cyberpunk 2077", "Red Dead Redemption 2", "Crimson Desert", "Hogwarts Legacy", "Metro Last Light Redux", "Stellar Blade"]
        {
            assert!(games.iter().any(|g| g.name == name), "{name} missing");
        }
        assert_eq!(super::games().len(), games.len());
        let cp = find(Some(1091500), "").unwrap();
        assert_eq!((cp.placement, cp.placement_rr, cp.hdr), (Placement::BeforeInside, Some(Placement::AfterDlss), Hdr::Hdr10));
        assert_eq!(find(None, "Z:\\games\\RDR2.exe").map(|g| g.name.as_str()), Some("Red Dead Redemption 2"), "by executable, any path");
        assert_eq!(find(None, "gta5_enhanced.exe").map(|g| g.steam_appid), Some(Some(3240220)), "any case");
        assert!(find(Some(1), "unknown.exe").is_none());
        // Every game's text renders.
        for g in &games {
            assert!(g.details().contains(&g.verified_text()));
            assert!(!g.status_line().is_empty());
        }
    }

    #[test]
    fn the_subset_reads_strings_arrays_and_comments() {
        let text = "# top\n[[game]]\nname = \"A # not a comment\" # a comment\nn = 1_000\nb = true\nlist = [\n  \"x\", # first\n  'y\\z',\n]\nempty = []\n";
        let t = tables(text, "game").unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0]["name"], Value::Str("A # not a comment".into()));
        assert_eq!(t[0]["n"], Value::Int(1000));
        assert_eq!(t[0]["b"], Value::Bool(true));
        assert_eq!(t[0]["list"], Value::Array(vec![Value::Str("x".into()), Value::Str("y\\z".into())]));
        assert_eq!(t[0]["empty"], Value::Array(vec![]));
        assert_eq!(parse_value("\"a\\\"b\\\\\"").unwrap(), Value::Str("a\"b\\".into()));
    }

    #[test]
    fn mistakes_are_refused_with_their_line() {
        assert!(tables("name = \"x\"\n", "game").unwrap_err().contains("before the first"));
        assert!(tables("[[game]]\nname = \"x\"\nname = \"y\"\n", "game").unwrap_err().contains("line 3: name is given twice"));
        assert!(tables("[game]\n", "game").unwrap_err().contains("line 1"));
        assert!(tables("[[game]]\nx = [\"a\"\n", "game").unwrap_err().contains("not closed"));
        assert!(tables("[[game]]\nx = nope\n", "game").unwrap_err().contains("not a string"));
        let base =
            "[[game]]\nname = \"G\"\nengine = \"E\"\napi = \"DX12\"\ndlss = \"SR\"\nplacement = \"before\"\nhdr = \"none\"\nverified_by = \"test machine\"\n";
        assert_eq!(parse(base).unwrap()[0].placement, Placement::Before);
        assert!(parse(&base.replace("\"before\"", "\"sideways\"")).unwrap_err().contains("placement"));
        assert!(parse(&format!("{base}colour = \"red\"\n")).unwrap_err().contains("unknown key colour"));
        assert!(parse(&base.replace("\"none\"", "\"hdr10\"")).unwrap_err().contains("hdr_route"));
        assert!(parse(&format!("{base}{base}")).unwrap_err().contains("listed twice"));
    }

    #[test]
    fn placements_map_to_the_state_the_layer_settles_in() {
        use neural_forge_protocol::enums::preupscale_state::*;
        for key in Placement::KEYS {
            assert_eq!(Placement::parse(key).unwrap().key(), key);
        }
        assert_eq!(Placement::BeforeInside.preupscale_state(), HOLDING);
        assert_eq!(Placement::AfterDlss.preupscale_state(), HOLDING_AFTER);
        assert_eq!(Placement::FinishedFrame.preupscale_state(), OFF);
    }

    #[test]
    fn the_report_names_an_unknown_game() {
        assert!(report_text(Some("1091500"), "").starts_with("Cyberpunk 2077\n"));
        assert!(report_text(Some("42"), "").starts_with("Steam app 42 is not in"));
        assert!(report_text(None, "foo.exe").starts_with("foo.exe is not in"));
    }
}
