//! Real implementation of `neural_forge_protocol::ShmHeader::capture_request`: "writes one
//! set of matched before/after frames per session when the layer next presents" --
//! defined in the protocol from early on but never acted on anywhere until now.
//! Doubles as the tool this project first used to visually confirm
//! `composition::apply`'s output looks right at all, rather than just "doesn't crash
//! and the round trip reports success" (see the crate's `CLAUDE.md` entry).

use std::io::BufWriter;
use std::path::PathBuf;

fn captures_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| {
        format!("{}/.local/share", std::env::var("HOME").unwrap_or_else(|_| "/".to_string()))
    });
    PathBuf::from(base).join("neuralforge").join("captures")
}

/// Writes `original`/`composited` (both `RGBA8`, `width`x`height`, same length) as a
/// matched pair of PNGs under `captures_dir()`, named by the current time so repeated
/// requests never collide. Fails silently (logs and returns) on any I/O error -- a
/// failed debug dump must never be a reason to skip presenting the real frame, the
/// same fail-open discipline `capture::run`'s own caller already applies to it.
pub fn write_pair(original: &[u8], composited: &[u8], width: u32, height: u32, bgr_order: bool) {
    let dir = captures_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        crate::log!("[dump] failed to create {}: {}", dir.display(), e);
        return;
    }
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let ok_original = write_png(&dir.join(format!("{stamp}-original.png")), original, width, height, bgr_order);
    let ok_composited = write_png(&dir.join(format!("{stamp}-composited.png")), composited, width, height, bgr_order);
    crate::log!(
        "[dump] capture_request: wrote {stamp}-{{original,composited}}.png under {} (original={} composited={})",
        dir.display(),
        ok_original,
        ok_composited
    );
}

fn write_png(path: &std::path::Path, rgba: &[u8], width: u32, height: u32, bgr_order: bool) -> bool {
    let Ok(file) = std::fs::File::create(path) else { return false };
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    // Forced fully opaque, on purpose: this is `RGBA8` straight off a real present
    // path (an opaque swapchain's own image, or the helper's `Output` resource), and
    // neither source is under any obligation to write a meaningful alpha channel --
    // a real opaque-composite-mode present never reads it either. Writing the real
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
    for px in opaque.chunks_exact_mut(4) {
        if bgr_order {
            px.swap(0, 2);
        }
        px[3] = 255;
    }
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let Ok(mut writer) = encoder.write_header() else { return false };
    writer.write_image_data(&opaque).is_ok()
}
