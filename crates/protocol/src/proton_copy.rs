//! What a Neural Forge Proton copy records about the Proton it was made from, and whether that Proton has changed
//! since (docs/PROTON_NGX.md). Shared by the supervisor (`proton_tool`: doctor, the Status and Setup tabs, Make
//! again) and the layer (the notice in a game's log at launch), so there is one check and one reader of the marker.

use std::path::{Path, PathBuf};

/// In a copy: what it was made from (`source=`, `source_version=`, `wrapper_sha256=`, `wine=`).
pub const MARKER: &str = "neural-forge-proton.txt";

/// The fields of a copy's marker.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Marker {
    /// The source's directory when the copy was made.
    pub source: PathBuf,
    /// The source's version name then ([`source_version`]).
    pub source_version: String,
    /// The SHA-256 of the wrapper in the copy.
    pub wrapper_sha256: String,
    /// How many files of the copy were hard links to the source when it was made (`linked=`), and those files
    /// (`link_file=<path>\t<size>\t<mtime>` lines: relative path, byte size, modification time as `secs.nanos`);
    /// absent in a copy made before they were recorded, and 0 files where the source is on another filesystem (a
    /// full copy).
    pub linked: Option<u64>,
    pub link_files: Vec<LinkedFile>,
}

/// One hard-linked file of a copy as it was when the copy was made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkedFile {
    pub path: String,
    pub size: u64,
    pub mtime: String,
}

impl LinkedFile {
    /// The marker line (without the newline).
    pub fn line(&self) -> String {
        format!("link_file={}\t{}\t{}", self.path, self.size, self.mtime)
    }

    fn parse(rest: &str) -> Option<Self> {
        let mut f = rest.rsplitn(3, '\t');
        let mtime = f.next()?.to_string();
        let size = f.next()?.parse().ok()?;
        Some(Self { path: f.next()?.to_string(), size, mtime })
    }
}

impl Marker {
    pub fn parse(text: &str) -> Self {
        let field = |k: &str| text.lines().find_map(|l| l.strip_prefix(k).map(str::to_string)).unwrap_or_default();
        Self {
            source: field("source=").into(),
            source_version: field("source_version="),
            wrapper_sha256: field("wrapper_sha256="),
            linked: field("linked=").parse().ok(),
            link_files: text.lines().filter_map(|l| l.strip_prefix("link_file=").and_then(LinkedFile::parse)).collect(),
        }
    }

    /// The marker of the copy in `dir`; `None` when `dir` is no copy.
    pub fn read(dir: &Path) -> Option<Self> {
        std::fs::read_to_string(dir.join(MARKER)).ok().map(|t| Self::parse(&t))
    }
}

/// The version name of the Proton in `dir`: its `version` file's second word, else its directory's name. `None`
/// when `dir` holds no `proton` script or is itself one of Neural Forge's copies.
pub fn source_version(dir: &Path) -> Option<String> {
    if !dir.join("proton").is_file() || dir.join(MARKER).exists() {
        return None;
    }
    let from_file = std::fs::read_to_string(dir.join("version")).ok().and_then(|t| t.split_whitespace().nth(1).map(str::to_string));
    from_file.or_else(|| dir.file_name().map(|n| n.to_string_lossy().into_owned()))
}

/// The version `source` has now when it is not the one a copy was made from (`was`): a new GE-Proton in a "Latest"
/// folder, a Proton Experimental update. `None` when it is the same, and when `source` is gone or replaced by
/// something that is no Proton (the copy holds its own files and keeps working).
pub fn newer_source(source: &Path, was: &str) -> Option<String> {
    source_version(source).filter(|now| now != was)
}

/// The line the layer writes to a game's log when the copy in `copy_dir` is older than its source; `None` when it
/// is not a copy or is current.
pub fn stale_notice(copy_dir: &Path) -> Option<String> {
    let marker = Marker::read(copy_dir)?;
    let now = newer_source(&marker.source, &marker.source_version)?;
    Some(format!(
        "[proton] this game runs with {}, made from {} but {} has {} now: close Steam and use Make again in Neural Forge's Setup tab \
         (or `neural-forge-cli proton-tool update`) to renew the copy",
        copy_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        marker.source_version,
        marker.source.display(),
        now
    ))
}

/// The Proton folders a game process runs with: Steam's `STEAM_COMPAT_TOOL_PATHS` (a `:`-separated list of the compat
/// tool folders), and the folders Wine's own libraries were loaded from, read from `maps` (`/proc/self/maps`: a
/// Proton's `files/lib/wine/x86_64-unix/ntdll.so` is mapped in every Wine process, and says which copy it is
/// whatever environment reached the process).
pub fn running_proton_dirs(tool_paths: Option<&str>, maps: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = tool_paths.into_iter().flat_map(|p| p.split(':')).filter(|p| !p.is_empty()).map(PathBuf::from).collect();
    for line in maps.lines() {
        // `address perms offset dev inode   path`: the path is what follows the fifth field.
        let mut rest = line;
        let mut fields = 0;
        while fields < 5 {
            rest = rest.trim_start();
            let Some(end) = rest.find(char::is_whitespace) else { break };
            rest = &rest[end..];
            fields += 1;
        }
        let path = rest.trim_start();
        if fields == 5 && path.starts_with('/') {
            if let Some(at) = path.find("/files/lib") {
                let dir = PathBuf::from(&path[..at]);
                if !out.contains(&dir) {
                    out.push(dir);
                }
            }
        }
    }
    out
}

/// [`stale_notice`] for each copy among [`running_proton_dirs`].
pub fn launch_notices(tool_paths: Option<&str>, maps: &str) -> Vec<String> {
    running_proton_dirs(tool_paths, maps).iter().filter_map(|d| stale_notice(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("neural-forge-protocol-proton-copy-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn proton(dir: &Path, version: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("proton"), "#!/bin/sh\n").unwrap();
        std::fs::write(dir.join("version"), format!("1700000000 {version}\n")).unwrap();
    }

    fn copy(dir: &Path, source: &Path, was: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(MARKER), format!("# Made\nsource={}\nsource_version={was}\nwrapper_sha256=ab\nwine=11.0\n", source.display())).unwrap();
    }

    #[test]
    fn a_source_with_another_version_than_the_copys_is_newer() {
        let d = scratch("newer");
        let src = d.join("Proton - Experimental");
        proton(&src, "experimental-11.0-20261111b-x86_64");
        assert_eq!(newer_source(&src, "experimental-11.0-20261111a-x86_64").as_deref(), Some("experimental-11.0-20261111b-x86_64"));
        assert_eq!(newer_source(&src, "experimental-11.0-20261111b-x86_64"), None);
        // Gone, or no Proton any more: nothing to compare with.
        assert_eq!(newer_source(&d.join("missing"), "x"), None);
        std::fs::remove_file(src.join("proton")).unwrap();
        assert_eq!(newer_source(&src, "x"), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_source_without_a_version_file_is_named_by_its_folder_and_a_copy_is_no_source() {
        let d = scratch("folder");
        let src = d.join("Custom");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("proton"), "").unwrap();
        assert_eq!(source_version(&src).as_deref(), Some("Custom"));
        copy(&src, &d, "x");
        assert_eq!(source_version(&src), None, "a Neural Forge copy is never a source");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Lines as a running Wine process has them (read from Proton Experimental's `/proc/PID/maps` on 2026-11-11): the
    /// path follows padding and has spaces in it.
    #[test]
    fn a_wine_process_s_maps_name_its_proton_folder_spaces_included() {
        let maps = "00010000-00026000 r--p 00000000 00:32 8711809                            /mnt/Storage/Steam/steamapps/common/Proton - Experimental/files/lib/wine/x86_64-windows/apisetschema.dll\n\
                    7ffff7a00000-7ffff7a28000 r--p 00000000 00:32 1                          /home/alex/.local/share/Steam/compatibilitytools.d/GE-Proton11-12-NeuralForge/files/lib64/wine/x86_64-unix/ntdll.so\n";
        assert_eq!(
            running_proton_dirs(None, maps),
            vec![
                PathBuf::from("/mnt/Storage/Steam/steamapps/common/Proton - Experimental"),
                PathBuf::from("/home/alex/.local/share/Steam/compatibilitytools.d/GE-Proton11-12-NeuralForge")
            ]
        );
    }

    #[test]
    fn the_launch_notice_names_the_copy_and_the_fix_only_for_a_stale_copy() {
        let d = scratch("notice");
        let src = d.join("GE-Proton-Latest");
        proton(&src, "GE-Proton11-13");
        let (stale, current, other) = (d.join("GE-Proton11-12-NeuralForge"), d.join("GE-Proton11-13-NeuralForge"), d.join("Plain"));
        copy(&stale, &src, "GE-Proton11-12");
        copy(&current, &src, "GE-Proton11-13");
        proton(&other, "Plain");
        let line = stale_notice(&stale).expect("stale");
        assert!(line.contains("GE-Proton11-12-NeuralForge") && line.contains("GE-Proton11-13") && line.contains("Make again"), "{line}");
        assert_eq!(stale_notice(&current), None);
        assert_eq!(stale_notice(&other), None, "not a copy");
        let paths = format!("{}::{}:{}", other.display(), stale.display(), current.display());
        assert_eq!(launch_notices(Some(&paths), ""), vec![line.clone()]);
        assert!(launch_notices(Some(""), "").is_empty() && launch_notices(None, "").is_empty());
        // Without Steam's variable: the copy Wine's libraries were loaded from, named once however many are mapped.
        let maps = format!(
            "7f0000000000-7f0000100000 r--p 00000000 08:01 1234                      {0}/files/lib/wine/x86_64-unix/ntdll.so\n\
             7f0000100000-7f0000200000 r-xp 00100000 08:01 1234                      {0}/files/lib/wine/x86_64-unix/ntdll.so\n\
             7f0000300000-7f0000400000 r--p 00000000 08:01 99                        /usr/lib/libc.so.6\n\
             7f0000500000-7f0000600000 rw-p 00000000 00:00 0                         [heap]\n\
             7f0000700000-7f0000800000 r--p 00000000 08:01 5                         {1}/files/lib/wine/x86_64-unix/ntdll.so\n",
            stale.display(),
            current.display()
        );
        assert_eq!(running_proton_dirs(None, &maps), vec![stale.clone(), current.clone()]);
        assert_eq!(launch_notices(None, &maps), vec![line.clone()]);
        assert_eq!(launch_notices(Some(&stale.display().to_string()), &maps), vec![line], "named by both: once");
        let _ = std::fs::remove_dir_all(&d);
    }
}
