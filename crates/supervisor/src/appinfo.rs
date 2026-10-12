//! Steam's app info cache, `appcache/appinfo.vdf`: the metadata Steam keeps for every app it knows,
//! among it each app's launch entries (`config/launch/<n>`: the executable, its arguments, and the
//! operating system and beta branch it is for). Read only, to learn which executable Steam starts
//! for a game ([`crate::steam::target_exes`]).
//!
//! The file is binary: a 16-byte header (magic `0x07564429`, the universe, the offset of a string
//! table), then one record per app (`appid`, `size`, then `size` bytes: state, last update, access
//! token, two SHA-1s, change number, and the app's key/value tree), ended by appid 0. The tree is
//! Valve's binary KeyValues: a type byte (0 block, 1 string, 2 int32, 3 float, 4 pointer, 6 colour,
//! 7 uint64, 10 int64, 8 or 11 end of block), the key as an index into the string table, the value.
//! Only this version (29, Steam since 2024) is read; any other is an error, and the caller falls
//! back to looking at the game's folder.

use std::path::Path;

const MAGIC_V29: u32 = 0x0756_4429;
/// Bytes of a record before its key/value tree: state, last update, access token, two SHA-1s and
/// the change number.
const RECORD_PREFIX: usize = 4 + 4 + 8 + 20 + 4 + 20;

/// One launch entry of an app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    /// As Steam has it: relative to the install folder, with `\` or `/`.
    pub executable: String,
    /// `config/oslist`: comma-separated (`windows`, `linux`, `macos`); empty means every system.
    pub oslist: String,
    /// `config/BetaKey`, split on commas: the beta branches the entry is for; empty means every branch.
    pub beta_keys: Vec<String>,
}

impl Launch {
    /// Whether the entry is for `os` (`windows`, `linux`).
    pub fn for_os(&self, os: &str) -> bool {
        self.oslist.trim().is_empty() || self.oslist.split(',').any(|o| o.trim().eq_ignore_ascii_case(os))
    }

    /// Whether the entry applies to the installed branch `branch` (empty: the default branch).
    pub fn for_branch(&self, branch: &str) -> bool {
        self.beta_keys.is_empty() || (!branch.is_empty() && self.beta_keys.iter().any(|k| k.eq_ignore_ascii_case(branch)))
    }
}

#[derive(Debug)]
enum Value {
    Block(Vec<(String, Value)>),
    Str(String),
    Other,
}

impl Value {
    fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Block(entries) => entries.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v),
            _ => None,
        }
    }

    fn path(&self, path: &[&str]) -> Option<&Value> {
        path.iter().try_fold(self, |v, key| v.get(key))
    }

    fn str_at(&self, path: &[&str]) -> Option<&str> {
        match self.path(path)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.data.len()).ok_or("appinfo.vdf ends early")?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn cstr(&mut self) -> Result<String, String> {
        let rest = &self.data[self.at..];
        let len = rest.iter().position(|&b| b == 0).ok_or("appinfo.vdf: unterminated string")?;
        let s = String::from_utf8_lossy(&rest[..len]).into_owned();
        self.at += len + 1;
        Ok(s)
    }

    /// A block's entries up to its end byte. `depth` bounds the nesting of a damaged file.
    fn block(&mut self, strings: &[String], depth: usize) -> Result<Value, String> {
        if depth > 64 {
            return Err("appinfo.vdf: nesting too deep".into());
        }
        let mut entries = Vec::new();
        loop {
            let kind = self.u8()?;
            if kind == 8 || kind == 11 {
                return Ok(Value::Block(entries));
            }
            let key = strings.get(self.u32()? as usize).ok_or("appinfo.vdf: key outside the string table")?.clone();
            let value = match kind {
                0 => self.block(strings, depth + 1)?,
                1 => Value::Str(self.cstr()?),
                2 | 3 | 4 | 6 => {
                    self.take(4)?;
                    Value::Other
                }
                7 | 10 => {
                    self.take(8)?;
                    Value::Other
                }
                other => return Err(format!("appinfo.vdf: unknown value type {other}")),
            };
            entries.push((key, value));
        }
    }
}

/// The launch entries of `appid` in an `appinfo.vdf` image; `Ok(None)` when the app is not in it.
pub fn launches_in(data: &[u8], appid: u32) -> Result<Option<Vec<Launch>>, String> {
    let mut r = Reader { data, at: 0 };
    let magic = r.u32()?;
    if magic != MAGIC_V29 {
        return Err(format!("appinfo.vdf version {magic:#x} is not the one this reads ({MAGIC_V29:#x})"));
    }
    r.u32()?; // universe
    let table = u64::from_le_bytes(r.take(8)?.try_into().unwrap());
    let table = usize::try_from(table).ok().filter(|&t| t >= 16 && t < data.len()).ok_or("appinfo.vdf: bad string table offset")?;
    loop {
        let id = r.u32()?;
        if id == 0 {
            return Ok(None);
        }
        let size = r.u32()? as usize;
        let start = r.at;
        if id != appid {
            r.take(size)?;
            continue;
        }
        if size < RECORD_PREFIX || start + size > table {
            return Err("appinfo.vdf: bad record size".into());
        }
        let strings = string_table(data, table)?;
        let mut tree = Reader { data: &data[..start + size], at: start + RECORD_PREFIX };
        let root = tree.block(&strings, 0)?;
        let Some(Value::Block(entries)) = root.path(&["appinfo", "config", "launch"]) else { return Ok(Some(Vec::new())) };
        let launches = entries
            .iter()
            .filter_map(|(_, entry)| {
                let executable = entry.str_at(&["executable"])?.trim().to_string();
                (!executable.is_empty()).then(|| Launch {
                    executable,
                    oslist: entry.str_at(&["config", "oslist"]).unwrap_or_default().to_string(),
                    beta_keys: entry
                        .str_at(&["config", "BetaKey"])
                        .unwrap_or_default()
                        .split(',')
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .collect(),
                })
            })
            .collect();
        return Ok(Some(launches));
    }
}

fn string_table(data: &[u8], at: usize) -> Result<Vec<String>, String> {
    let mut r = Reader { data, at };
    let count = r.u32()? as usize;
    // Every string is at least its terminator: a count larger than the rest of the file is damage.
    if count > data.len() - r.at {
        return Err("appinfo.vdf: bad string table".into());
    }
    (0..count).map(|_| r.cstr()).collect()
}

/// The launch entries of `appid` from `root`'s `appcache/appinfo.vdf`.
pub fn launches(root: &Path, appid: &str) -> Result<Option<Vec<Launch>>, String> {
    let appid: u32 = appid.parse().map_err(|_| format!("{appid:?} is not an appid"))?;
    let path = root.join("appcache/appinfo.vdf");
    let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    launches_in(&data, appid)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Builds a version-29 `appinfo.vdf` image: one record per `(appid, launch entries)`, each entry
    /// `(executable, oslist, BetaKey)`.
    /// One launch entry for [`image`]: (executable, oslist, BetaKey).
    pub(crate) type Entry<'a> = (&'a str, &'a str, &'a str);

    pub(crate) fn image(apps: &[(u32, &[Entry])]) -> Vec<u8> {
        struct Kv {
            out: Vec<u8>,
            strings: Vec<String>,
        }
        impl Kv {
            fn key(&mut self, kind: u8, name: &str) {
                let i = self.strings.iter().position(|k| k == name).unwrap_or_else(|| {
                    self.strings.push(name.to_string());
                    self.strings.len() - 1
                });
                self.out.push(kind);
                self.out.extend((i as u32).to_le_bytes());
            }
            fn string(&mut self, name: &str, value: &str) {
                self.key(1, name);
                self.out.extend(value.as_bytes());
                self.out.push(0);
            }
        }
        let mut kv = Kv { out: Vec::new(), strings: Vec::new() };
        let mut body = Vec::new();
        for (appid, entries) in apps {
            kv.out.clear();
            kv.key(0, "appinfo");
            kv.key(2, "appid");
            kv.out.extend(appid.to_le_bytes());
            kv.key(0, "config");
            kv.string("installdir", "Game");
            kv.key(0, "launch");
            for (i, (exe, os, beta)) in entries.iter().enumerate() {
                kv.key(0, &i.to_string());
                kv.string("executable", exe);
                kv.key(7, "ownsdlc");
                kv.out.extend(0u64.to_le_bytes());
                kv.key(0, "config");
                for (name, value) in [("oslist", os), ("BetaKey", beta)] {
                    if !value.is_empty() {
                        kv.string(name, value);
                    }
                }
                kv.out.extend([8, 8]); // config, entry
            }
            kv.out.extend([8, 8, 8, 8]); // launch, config, appinfo, root
            body.extend(appid.to_le_bytes());
            body.extend(((RECORD_PREFIX + kv.out.len()) as u32).to_le_bytes());
            body.extend(vec![0u8; RECORD_PREFIX]);
            body.extend(&kv.out);
        }
        body.extend(0u32.to_le_bytes());
        let mut out = Vec::new();
        out.extend(MAGIC_V29.to_le_bytes());
        out.extend(1u32.to_le_bytes());
        out.extend(((16 + body.len()) as u64).to_le_bytes());
        out.extend(body);
        out.extend((kv.strings.len() as u32).to_le_bytes());
        for s in &kv.strings {
            out.extend(s.as_bytes());
            out.push(0);
        }
        out
    }

    #[test]
    fn launch_entries_are_read_for_the_one_app() {
        let data = image(&[
            (10, &[("other.exe", "windows", "")]),
            (990080, &[("Phoenix-Win64-Test.exe", "windows", "development, playtest"), ("HogwartsLegacy.exe", "windows", "")]),
            (20, &[]),
        ]);
        let launches = launches_in(&data, 990080).unwrap().unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].executable, "Phoenix-Win64-Test.exe");
        assert_eq!(launches[0].beta_keys, ["development", "playtest"]);
        assert!(!launches[0].for_branch("") && launches[0].for_branch("playtest"));
        assert!(launches[1].for_branch("") && launches[1].for_os("windows") && !launches[1].for_os("linux"));
        assert_eq!(launches_in(&data, 20).unwrap(), Some(Vec::new()));
        assert_eq!(launches_in(&data, 30).unwrap(), None, "not in the cache");
    }

    #[test]
    fn a_damaged_or_other_version_file_is_an_error_not_a_panic() {
        let data = image(&[(5, &[("a.exe", "", "")])]);
        for cut in 0..data.len() {
            let _ = launches_in(&data[..cut], 5);
        }
        let mut v28 = data.clone();
        v28[0] = 0x28;
        assert!(launches_in(&v28, 5).unwrap_err().contains("version"));
        let mut bad = data.clone();
        bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(launches_in(&bad, 5).is_err());
    }
}
