//! Where captures go, and how a capture's PNGs are written (`crate::series` writes the one-shot pair
//! and frame series; the pre-upscaler dump writes its preview through [`write_png`] too).

use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// `$XDG_DATA_HOME/neural-forge/captures`, else `$HOME/.local/share/neural-forge/captures`. A
/// relative `XDG_DATA_HOME` is invalid (the XDG base directory spec says to ignore it), as is a
/// relative `HOME`: neither may decide where files land relative to the game's working directory.
///
/// With neither, the files go under the layer's own runtime directory (`/tmp/neural-forge-<uid>`),
/// and only once that is known to be private (ours, not a symlink, no group or other access,
/// created so if missing): a fixed name in the world-writable `/tmp` could otherwise be another
/// user's directory, who would read or replace the captures. When it is not, the path is one
/// nothing can create ([`REFUSED`]), so every capture fails with a write error instead.
pub(crate) fn captures_dir() -> PathBuf {
    let xdg = neural_forge_protocol::env::var("XDG_DATA_HOME");
    let home = neural_forge_protocol::env::var("HOME");
    let runtime = || {
        let dir = neural_forge_protocol::shm_runtime_dir();
        let private = neural_forge_protocol::private_dir::ensure_private_parent_dir(&format!("{dir}/neural-forge"));
        if !private {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::log!("[capture] no home directory, and {dir} is not a private directory of this user: captures are refused");
            }
        }
        private.then_some(dir)
    };
    captures_dir_from(xdg.as_deref(), home.as_deref(), runtime)
}

/// Under procfs, which creates no directory for anyone (root included): every write under it fails.
const REFUSED: &str = "/proc/neural-forge-refused";

fn captures_dir_from(xdg_data_home: Option<&str>, home: Option<&str>, private_runtime_dir: impl FnOnce() -> Option<String>) -> PathBuf {
    let absolute = |s: &&str| Path::new(s).is_absolute();
    let base = match (xdg_data_home.filter(absolute), home.filter(absolute)) {
        (Some(xdg), _) => PathBuf::from(xdg),
        (None, Some(home)) => Path::new(home).join(".local/share"),
        (None, None) => PathBuf::from(private_runtime_dir().unwrap_or_else(|| REFUSED.to_owned())),
    };
    base.join("neural-forge").join("captures")
}

/// Writes `rgba` (`width`x`height`, 4 bytes a pixel) as a PNG at `path`. The file appears complete or
/// not at all: it is written to a temporary name beside `path`, finished (the PNG's last chunk) and
/// flushed, then renamed into place, so an error part-way (a full disk) is reported here and never
/// leaves a truncated PNG under the final name.
///
/// `fast` picks the PNG encoder's fastest compression (a series writes a pair per presented frame and
/// must keep up); the one-shot keeps the default.
pub(crate) fn write_png(path: &Path, rgba: &[u8], width: u32, height: u32, bgr_order: bool, fast: bool) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| std::io::Error::other("no file name"))?.to_string_lossy();
    let temp = path.with_file_name(format!(".{name}.{}.part", std::process::id()));
    let result = encode_png(&temp, rgba, width, height, bgr_order, fast).and_then(|()| std::fs::rename(&temp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn encode_png(path: &Path, rgba: &[u8], width: u32, height: u32, bgr_order: bool, fast: bool) -> std::io::Result<()> {
    if rgba.len() as u64 != u64::from(width) * u64::from(height) * 4 {
        return Err(std::io::Error::other("pixel data does not match the size"));
    }
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut buffered = BufWriter::new(file);
    let mut encoder = png::Encoder::new(&mut buffered, width, height);
    // Forced fully opaque, on purpose: this is `RGBA8` straight off a real present
    // path (an opaque swapchain's own image), which is under no obligation to write a meaningful
    // alpha channel -- a real opaque-composite-mode present never reads it either. Writing the real
    // (frequently 0) alpha through unmodified would make a real, correctly-composited
    // frame render as fully transparent in any normal PNG viewer, which looks exactly
    // like -- and was once genuinely mistaken here for -- a solid-white broken answer.
    //
    // Also swaps R/B here when the real captured bytes are `B8G8R8A8` order (see
    // `swapchain::is_bgr_order`'s doc comment) -- PNG has no concept of a BGR pixel
    // order, so this dump must always normalize to true RGBA before writing, or the
    // saved image visibly mismatches the real scene (confirmed real, 2026-09-11: a
    // uniform blue tint across an entire real capture, matching a real/blue channel
    // swap exactly, not a rendering bug in the game or the model itself).
    let mut opaque = rgba.to_vec();
    for px in opaque.as_chunks_mut::<4>().0 {
        if bgr_order {
            px.swap(0, 2);
        }
        px[3] = 255;
    }
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    if fast {
        encoder.set_compression(png::Compression::Fast);
    }
    let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
    writer.write_image_data(&opaque).map_err(std::io::Error::other)?;
    // `finish` writes the closing chunk; then the buffer is flushed into the file (`into_inner` reports a
    // failed flush, which dropping the writer would swallow) and the file synced, before the caller
    // renames it into place.
    writer.finish().map_err(std::io::Error::other)?;
    let file = buffered.into_inner().map_err(std::io::IntoInnerError::into_error)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join(format!("target/test-scratch/dump-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn only_absolute_base_directories_are_used() {
        let unused = || -> Option<String> { panic!("the runtime directory is only asked for without a home") };
        assert_eq!(captures_dir_from(Some("/x/data"), Some("/home/u"), unused), PathBuf::from("/x/data/neural-forge/captures"));
        assert_eq!(
            captures_dir_from(Some("rel/data"), Some("/home/u"), unused),
            PathBuf::from("/home/u/.local/share/neural-forge/captures"),
            "a relative XDG_DATA_HOME is ignored"
        );
        assert_eq!(captures_dir_from(None, Some("/home/u"), unused), PathBuf::from("/home/u/.local/share/neural-forge/captures"));
    }

    /// Without a home directory the captures go under the private runtime directory, never a fixed
    /// name in `/tmp` another user could have made; when that is not private, nowhere at all.
    #[test]
    fn without_a_home_only_the_private_runtime_directory_is_used() {
        let private = || Some("/tmp/neural-forge-1000".to_owned());
        assert_eq!(captures_dir_from(Some(""), Some("relative"), private), PathBuf::from("/tmp/neural-forge-1000/neural-forge/captures"));
        let refused = captures_dir_from(None, None, || None);
        assert!(refused.starts_with(REFUSED), "{}", refused.display());
        assert!(std::fs::create_dir_all(&refused).is_err(), "nothing can create the refused directory");
        // The check `captures_dir` makes: a private directory is used, one others can enter is not.
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("runtime");
        let checked = || {
            let dir = dir.display().to_string();
            neural_forge_protocol::private_dir::ensure_private_parent_dir(&format!("{dir}/neural-forge")).then_some(dir)
        };
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(captures_dir_from(None, None, checked), dir.join("neural-forge/captures"));
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(captures_dir_from(None, None, checked).starts_with(REFUSED), "a directory others can write is refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_png_is_written_whole_and_decodes_with_swapped_channels_and_opaque_alpha() {
        let dir = scratch("png");
        let path = dir.join("frame.png");
        let pixels = [10u8, 20, 30, 0, 40, 50, 60, 7];
        write_png(&path, &pixels, 2, 1, true, false).unwrap();
        let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut buf).unwrap();
        assert_eq!(buf, [30, 20, 10, 255, 60, 50, 40, 255]);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no temporary file is left beside it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_write_leaves_no_file_under_either_name() {
        let dir = scratch("fail");
        // A size the pixels do not match is refused before anything is created.
        assert!(write_png(&dir.join("bad.png"), &[0; 7], 2, 1, false, true).is_err());
        // A missing directory fails at the temporary file.
        assert!(write_png(&dir.join("missing/x.png"), &[0; 8], 2, 1, false, true).is_err());
        // A failure after the file was created (the encoder refuses a 0x0 image): nothing is left
        // under the final name, and no temporary file either.
        assert!(write_png(&dir.join("empty.png"), &[], 0, 0, false, true).is_err());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
