//! A deterministic panning picture that presents through a real X11 (xcb) swapchain, so the
//! Neural Forge layer engages on it the way it does on a game -- the moving counterpart to
//! `vkcube`, whose background never moves and so can never show an edit landing on the wrong
//! frame.
//!
//! The picture is a large procedural texture (multi-scale value noise, text-like glyph rows,
//! hard-edged shapes, fine stripes, smooth gradients and flat fills) that translates by
//! `--px-per-frame` in x and half that in y every frame, wrapping, bilinear-sampled so
//! fractional speeds are real sub-pixel motion. Everything is a function of the frame counter
//! alone: frame N is the same picture in every run, whatever the timing. The counter is also
//! stamped into the top-left corner (37 black/white 8x8 blocks: the magic bits 1011, the 32-bit
//! frame number most significant bit first, an even-parity bit), which `scripts/agreement.py`
//! decodes from captured originals.
//!
//! ```text
//! cargo run --release -p neural-forge-layer --example pan -- \
//!     [--width 1280] [--height 720] [--px-per-frame 2.0] [--grain 0.0] [--contrast 1.0]
//!     [--frames 0] [--present-mode fifo|immediate|mailbox] [--save DIR]
//!     [--capture-at FRAME [--capture-frames 120]]
//! ```
//!
//! `--capture-at FRAME` asks the Neural Forge layer in this process for a frame series
//! (`capture_request = --capture-frames`, see `crates/layer/src/series.rs`) right before frame
//! FRAME is presented, through the same shared memory the layer opens (`NEURAL_FORGE_SHM`, else
//! the default path). The layer takes the request on that very present when it is engaged by
//! then, so every run captures the same frames; pan prints the frame the series really started
//! on. Compare the runs with `scripts/agreement.py`.
//!
//! `--frames 0` runs until the window is closed (or Esc). The achieved frame rate goes to
//! stderr once a second. The picture is generated on the CPU into a host-visible staging
//! buffer (all cores) and copied into the swapchain image, which is created with
//! `TRANSFER_DST` (plus `COLOR_ATTACHMENT` when the surface allows, as a game's would).
//! Needs an X server (XWayland is fine); libxcb is loaded at run time, so nothing extra is
//! linked.

use ash::extensions::khr;
use ash::vk;
use std::ffi::{c_char, c_int, c_void, CStr};

// ---------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Args {
    width: u32,
    height: u32,
    px_per_frame: f32,
    grain: f32,
    contrast: f32,
    frames: u64,
    present_mode: vk::PresentModeKHR,
    /// Also write every presented frame as `DIR/<frame:06>.png` (slow; for checking the
    /// picture and building reference originals without the layer).
    save: Option<std::path::PathBuf>,
    /// Request a layer frame series of `capture_frames` frames right before presenting this frame.
    capture_at: Option<u64>,
    capture_frames: u32,
}

fn usage() -> ! {
    eprintln!(
        "usage: pan [--width W] [--height H] [--px-per-frame F] [--grain 0..1] [--contrast C] \
         [--frames N (0 = forever)] [--present-mode fifo|immediate|mailbox] [--save DIR] \
         [--capture-at FRAME [--capture-frames N (> 1, default 120)]]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args { width: 1280, height: 720, px_per_frame: 2.0, grain: 0.0, contrast: 1.0, frames: 0, present_mode: vk::PresentModeKHR::FIFO, save: None, capture_at: None, capture_frames: 120 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            usage();
        }
        let Some(value) = it.next() else { usage() };
        let bad = || -> ! {
            eprintln!("pan: bad value {value:?} for {flag}");
            usage()
        };
        match flag.as_str() {
            "--width" => a.width = value.parse().unwrap_or_else(|_| bad()),
            "--height" => a.height = value.parse().unwrap_or_else(|_| bad()),
            "--px-per-frame" => a.px_per_frame = value.parse().unwrap_or_else(|_| bad()),
            "--grain" => a.grain = value.parse::<f32>().unwrap_or_else(|_| bad()).clamp(0.0, 1.0),
            "--contrast" => a.contrast = value.parse().unwrap_or_else(|_| bad()),
            "--frames" => a.frames = value.parse().unwrap_or_else(|_| bad()),
            "--save" => a.save = Some(value.into()),
            "--capture-at" => a.capture_at = Some(value.parse().unwrap_or_else(|_| bad())),
            "--capture-frames" => a.capture_frames = value.parse::<u32>().ok().filter(|&n| n > 1).unwrap_or_else(|| bad()),
            "--present-mode" => {
                a.present_mode = match value.as_str() {
                    "fifo" => vk::PresentModeKHR::FIFO,
                    "immediate" => vk::PresentModeKHR::IMMEDIATE,
                    "mailbox" => vk::PresentModeKHR::MAILBOX,
                    _ => bad(),
                }
            }
            _ => {
                eprintln!("pan: unknown flag {flag}");
                usage()
            }
        }
    }
    if a.width < STAMP_WIDTH || a.height < 2 * BLOCK {
        eprintln!("pan: the window must be at least {STAMP_WIDTH}x{} for the frame stamp", 2 * BLOCK);
        std::process::exit(2);
    }
    a
}

// ---------------------------------------------------------------------------------------
// The picture
// ---------------------------------------------------------------------------------------

/// Texture size: powers of two, so wrapping is a mask, and a multiple of every noise lattice
/// and cell size used below, so the texture tiles without a seam.
const TW: usize = 4096;
const TH: usize = 2048;
const CELL: usize = 256;

/// Frame stamp geometry (see the module doc comment); `scripts/agreement.py` mirrors it.
const BLOCK: u32 = 8;
const MAGIC: [bool; 4] = [true, false, true, true];
const STAMP_BLOCKS: u32 = 4 + 32 + 1;
const STAMP_WIDTH: u32 = STAMP_BLOCKS * BLOCK;

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    hash(a ^ hash(b ^ hash(c.wrapping_add(0x9e37_79b9))))
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Seamless value noise in 0..1 at lattice spacing `scale` (a power of two dividing TW, TH).
fn value_noise(x: usize, y: usize, scale: usize, seed: u32) -> f32 {
    let (gw, gh) = ((TW / scale) as u32, (TH / scale) as u32);
    let (gx, gy) = ((x / scale) as u32, (y / scale) as u32);
    let fx = (x % scale) as f32 / scale as f32;
    let fy = (y % scale) as f32 / scale as f32;
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let at = |i: u32, j: u32| unit(hash3(i % gw, j % gh, seed));
    let top = at(gx, gy) * (1.0 - sx) + at(gx + 1, gy) * sx;
    let bottom = at(gx, gy + 1) * (1.0 - sx) + at(gx + 1, gy + 1) * sx;
    top * (1.0 - sy) + bottom * sy
}

fn palette(h: u32) -> [f32; 3] {
    [(h & 0xff) as f32, ((h >> 8) & 0xff) as f32, ((h >> 16) & 0xff) as f32]
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// One texel, RGB 0..255 before contrast.
fn texel(x: usize, y: usize) -> [f32; 3] {
    let (cx, cy) = ((x / CELL) as u32, (y / CELL) as u32);
    let (lx, ly) = (x % CELL, y % CELL);
    let h = hash3(cx, cy, 1);
    // Large-scale shading over everything, so cells are not perfectly flat tiles.
    let shade = 0.85 + 0.3 * value_noise(x, y, 512, 2);
    // Eight of ten cell kinds are detailed (noise, glyphs, shapes, stripes); two are smooth.
    let rgb = match h % 10 {
        // Detail: fBm, coloured by two slow fields; kind 9 is a finer "gravel" variant.
        0..=2 | 9 => {
            let (octaves, falloff): (&[usize], f32) = if h % 10 == 9 { (&[16, 8, 4, 2, 1], 0.95) } else { (&[64, 32, 16, 8, 4, 2], 0.8) };
            let mut v = 0.0;
            let mut amp = 0.5;
            let mut total = 0.0;
            for (i, &scale) in octaves.iter().enumerate() {
                v += amp * value_noise(x, y, scale, 10 + i as u32);
                total += amp;
                amp *= falloff;
            }
            let v = (v / total - 0.5) * 2.2 + 0.5;
            let tint = lerp3(palette(hash(h)), palette(hash(h ^ 0xabcd)), value_noise(x, y, 128, 3));
            [tint[0] * v, tint[1] * v, tint[2] * v]
        }
        // Text-like glyph rows: dark 5x7 random glyphs on a light ground.
        3 | 4 => {
            let s = 2 + (hash(h) % 2) as usize;
            let (cw, ch) = (6 * s, 10 * s);
            let ground = 200.0 + (hash(h ^ 7) % 40) as f32;
            let ink = (hash(h ^ 9) % 50) as f32;
            let margin = 8;
            if lx < margin || ly < margin || lx >= CELL - margin || ly >= CELL - margin {
                [ground; 3]
            } else {
                let (px, py) = (lx - margin, ly - margin);
                let (col, row) = ((px / cw) as u32, (py / ch) as u32);
                let (gx, gy) = ((px % cw) / s, (py % ch) / s);
                let glyph = hash3(cx * 64 + col, cy * 64 + row, 4);
                let on = !glyph.is_multiple_of(6) && gx < 5 && gy < 7 && (hash(glyph) >> (gy * 5 + gx) as u32) & 1 == 1;
                if on { [ink; 3] } else { [ground; 3] }
            }
        }
        // Smooth gradient: the flat, low-detail areas.
        5 => {
            let t = if hash(h) & 1 == 0 { lx as f32 / CELL as f32 } else { ly as f32 / CELL as f32 };
            lerp3(palette(hash(h ^ 1)), palette(hash(h ^ 2)), t)
        }
        // Hard-edged shapes and thin lines on a mid ground.
        6 => {
            let mut c = palette(hash(h ^ 3));
            for k in 0..6u32 {
                let sh = hash3(cx, cy, 100 + k);
                let (ox, oy) = ((sh % CELL as u32) as f32, ((sh >> 8) % CELL as u32) as f32);
                let r = 12.0 + ((sh >> 16) % 60) as f32;
                let (dx, dy) = (lx as f32 - ox, ly as f32 - oy);
                let inside = if k % 2 == 0 { dx * dx + dy * dy < r * r } else { dx.abs() < r && dy.abs() < r * 0.6 };
                if inside {
                    c = palette(hash(sh));
                }
            }
            if (lx + 2 * ly) % 37 < 2 {
                c = [20.0, 20.0, 20.0];
            }
            c
        }
        // Fine stripes / checks: the highest-frequency detail.
        7 => {
            let period = 2 + (hash(h) % 4) as usize;
            let on = if hash(h ^ 5) & 1 == 0 { (lx / period + ly / period).is_multiple_of(2) } else { (lx / period).is_multiple_of(2) };
            if on { palette(hash(h ^ 6)) } else { [235.0, 235.0, 235.0] }
        }
        // Flat fill.
        _ => palette(hash(h ^ 8)),
    };
    [rgb[0] * shade, rgb[1] * shade, rgb[2] * shade]
}

/// The whole texture as packed pixels in the swapchain's byte order, alpha 255.
fn build_texture(contrast: f32, bgr: bool) -> Vec<u32> {
    let mut tex = vec![0u32; TW * TH];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per = TH.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk_index, chunk) in tex.chunks_mut(rows_per * TW).enumerate() {
            s.spawn(move || {
                for (i, px) in chunk.iter_mut().enumerate() {
                    let (x, y) = (i % TW, chunk_index * rows_per + i / TW);
                    let c = texel(x, y).map(|v| (128.0 + (v - 128.0) * contrast).round().clamp(0.0, 255.0) as u32);
                    let (b0, b2) = if bgr { (c[2], c[0]) } else { (c[0], c[2]) };
                    *px = b0 | (c[1] << 8) | (b2 << 16) | (255 << 24);
                }
            });
        }
    });
    tex
}

/// `a` to `b` by `w`/256 on all four bytes at once (two bytes per 16-bit lane, no overflow).
#[inline]
fn lerp_packed(a: u32, b: u32, w: u32) -> u32 {
    let iw = 256 - w;
    let even = (((a & 0x00ff_00ff) * iw + (b & 0x00ff_00ff) * w) >> 8) & 0x00ff_00ff;
    let odd = ((((a >> 8) & 0x00ff_00ff) * iw + ((b >> 8) & 0x00ff_00ff) * w) >> 8) & 0x00ff_00ff;
    even | (odd << 8)
}

/// Fills `out` (`width`x`height` packed pixels) with frame `frame`.
fn render_frame(out: &mut [u32], tex: &[u32], width: usize, height: usize, frame: u64, args: &Args) {
    // The picture moves right by `px` and down by `px / 2` per frame: pixel (x, y) shows texel
    // (x - ox, y - oy). Pure translation, so the bilinear weights are the same for every pixel.
    let ox = (frame as f64 * f64::from(args.px_per_frame)).rem_euclid(TW as f64);
    let oy = (frame as f64 * f64::from(args.px_per_frame) * 0.5).rem_euclid(TH as f64);
    let (sx, sy) = ((TW as f64 - ox).rem_euclid(TW as f64), (TH as f64 - oy).rem_euclid(TH as f64));
    let (bx, by) = (sx.floor() as usize, sy.floor() as usize);
    let wx = ((sx - sx.floor()) * 256.0).round() as u32;
    let wy = ((sy - sy.floor()) * 256.0).round() as u32;
    let grain = (args.grain * 48.0) as i32;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per = height.div_ceil(threads);
    std::thread::scope(|s| {
        for (chunk_index, chunk) in out.chunks_mut(rows_per * width).enumerate() {
            s.spawn(move || {
                for (r, row) in chunk.chunks_mut(width).enumerate() {
                    let y = chunk_index * rows_per + r;
                    let ty0 = (by + y) & (TH - 1);
                    let ty1 = (ty0 + 1) & (TH - 1);
                    let (row0, row1) = (&tex[ty0 * TW..][..TW], &tex[ty1 * TW..][..TW]);
                    for (x, px) in row.iter_mut().enumerate() {
                        let tx0 = (bx + x) & (TW - 1);
                        let tx1 = (tx0 + 1) & (TW - 1);
                        let top = lerp_packed(row0[tx0], row0[tx1], wx);
                        let bottom = lerp_packed(row1[tx0], row1[tx1], wx);
                        let mut p = lerp_packed(top, bottom, wy) | 0xff00_0000;
                        if grain > 0 {
                            // Deterministic per (x, y, frame), luma-only.
                            let h = hash3(x as u32, y as u32, frame as u32 ^ (frame >> 32) as u32);
                            let g = (h % (2 * grain as u32 + 1)) as i32 - grain;
                            let ch = |shift: u32| (((p >> shift) & 0xff) as i32 + g).clamp(0, 255) as u32;
                            p = ch(0) | (ch(8) << 8) | (ch(16) << 16) | 0xff00_0000;
                        }
                        *px = p;
                    }
                }
            });
        }
    });
    stamp(out, width, frame as u32);
}

/// Writes the frame stamp (see the module doc comment) into the top-left corner.
fn stamp(out: &mut [u32], width: usize, frame: u32) {
    let mut bits = Vec::with_capacity(STAMP_BLOCKS as usize);
    bits.extend_from_slice(&MAGIC);
    bits.extend((0..32).rev().map(|i| (frame >> i) & 1 == 1));
    bits.push(frame.count_ones() % 2 == 1);
    for (i, &bit) in bits.iter().enumerate() {
        let colour = if bit { 0xffff_ffff } else { 0xff00_0000 };
        for y in 0..BLOCK as usize {
            let start = y * width + i * BLOCK as usize;
            out[start..start + BLOCK as usize].fill(colour);
        }
    }
}

fn save_png(path: &std::path::Path, pixels: &[u32], width: usize, height: usize, bgr: bool) {
    let mut rgba = Vec::with_capacity(pixels.len() * 4);
    for &p in pixels {
        let b = p.to_le_bytes();
        rgba.extend_from_slice(&if bgr { [b[2], b[1], b[0], 255] } else { [b[0], b[1], b[2], 255] });
    }
    let result = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new("."))).map_err(|e| e.to_string()).and_then(|()| {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        encoder.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())
    });
    if let Err(e) = result {
        eprintln!("pan: could not save {}: {e}", path.display());
    }
}

// ---------------------------------------------------------------------------------------
// xcb, loaded at run time
// ---------------------------------------------------------------------------------------

#[repr(C)]
struct XcbScreen {
    root: u32,
    default_colormap: u32,
    white_pixel: u32,
    black_pixel: u32,
    current_input_masks: u32,
    width_in_pixels: u16,
    height_in_pixels: u16,
    width_in_mm: u16,
    height_in_mm: u16,
    min_installed_maps: u16,
    max_installed_maps: u16,
    root_visual: u32,
    backing_stores: u8,
    save_unders: u8,
    root_depth: u8,
    allowed_depths_len: u8,
}

#[repr(C)]
struct XcbScreenIterator {
    data: *mut XcbScreen,
    rem: c_int,
    index: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XcbCookie {
    sequence: u32,
}

#[repr(C)]
struct XcbInternAtomReply {
    response_type: u8,
    pad0: u8,
    sequence: u16,
    length: u32,
    atom: u32,
}

#[repr(C)]
struct XcbGenericEvent {
    response_type: u8,
    detail: u8,
    sequence: u16,
    // ClientMessage: window, type, data[0..5]; KeyPress: time, root, event, ...
    words: [u32; 7],
    full_sequence: u32,
}

struct Xcb {
    conn: *mut c_void,
    window: u32,
    delete_atom: u32,
    poll_for_event: unsafe extern "C" fn(*mut c_void) -> *mut XcbGenericEvent,
    disconnect: unsafe extern "C" fn(*mut c_void),
}

impl Xcb {
    fn open(width: u32, height: u32, title: &str) -> Result<Xcb, String> {
        // SAFETY: dlopen/dlsym with valid NUL-terminated names; each symbol is transmuted to
        // its documented libxcb signature.
        unsafe {
            let lib = libc::dlopen(c"libxcb.so.1".as_ptr(), libc::RTLD_NOW);
            if lib.is_null() {
                return Err("libxcb.so.1 not found".into());
            }
            let sym = |name: &CStr| -> Result<*mut c_void, String> {
                let p = libc::dlsym(lib, name.as_ptr());
                if p.is_null() { Err(format!("libxcb has no {name:?}")) } else { Ok(p) }
            };
            let connect: unsafe extern "C" fn(*const c_char, *mut c_int) -> *mut c_void = std::mem::transmute(sym(c"xcb_connect")?);
            let has_error: unsafe extern "C" fn(*mut c_void) -> c_int = std::mem::transmute(sym(c"xcb_connection_has_error")?);
            let get_setup: unsafe extern "C" fn(*mut c_void) -> *const c_void = std::mem::transmute(sym(c"xcb_get_setup")?);
            let roots: unsafe extern "C" fn(*const c_void) -> XcbScreenIterator = std::mem::transmute(sym(c"xcb_setup_roots_iterator")?);
            let screen_next: unsafe extern "C" fn(*mut XcbScreenIterator) = std::mem::transmute(sym(c"xcb_screen_next")?);
            let generate_id: unsafe extern "C" fn(*mut c_void) -> u32 = std::mem::transmute(sym(c"xcb_generate_id")?);
            let create_window: unsafe extern "C" fn(*mut c_void, u8, u32, u32, i16, i16, u16, u16, u16, u16, u32, u32, *const u32) -> XcbCookie =
                std::mem::transmute(sym(c"xcb_create_window")?);
            let intern_atom: unsafe extern "C" fn(*mut c_void, u8, u16, *const c_char) -> XcbCookie = std::mem::transmute(sym(c"xcb_intern_atom")?);
            let intern_atom_reply: unsafe extern "C" fn(*mut c_void, XcbCookie, *mut *mut c_void) -> *mut XcbInternAtomReply =
                std::mem::transmute(sym(c"xcb_intern_atom_reply")?);
            let change_property: unsafe extern "C" fn(*mut c_void, u8, u32, u32, u32, u8, u32, *const c_void) -> XcbCookie =
                std::mem::transmute(sym(c"xcb_change_property")?);
            let map_window: unsafe extern "C" fn(*mut c_void, u32) -> XcbCookie = std::mem::transmute(sym(c"xcb_map_window")?);
            let flush: unsafe extern "C" fn(*mut c_void) -> c_int = std::mem::transmute(sym(c"xcb_flush")?);
            let poll_for_event = std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void) -> *mut XcbGenericEvent>(sym(c"xcb_poll_for_event")?);
            let disconnect = std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut c_void)>(sym(c"xcb_disconnect")?);

            let mut screen_index: c_int = 0;
            let conn = connect(std::ptr::null(), &mut screen_index);
            if conn.is_null() || has_error(conn) != 0 {
                return Err("cannot connect to the X server (is DISPLAY set?)".into());
            }
            let mut it = roots(get_setup(conn));
            for _ in 0..screen_index {
                screen_next(&mut it);
            }
            let screen = &*it.data;
            let window = generate_id(conn);
            const CW_BACK_PIXEL: u32 = 0x2;
            const CW_EVENT_MASK: u32 = 0x800;
            const EVENT_MASK_KEY_PRESS: u32 = 0x1;
            const EVENT_MASK_STRUCTURE_NOTIFY: u32 = 0x20000;
            let values = [screen.black_pixel, EVENT_MASK_KEY_PRESS | EVENT_MASK_STRUCTURE_NOTIFY];
            create_window(conn, 0, window, screen.root, 0, 0, width as u16, height as u16, 0, 1, screen.root_visual, CW_BACK_PIXEL | CW_EVENT_MASK, values.as_ptr());
            let atom = |name: &str, only_if_exists: u8| {
                let cookie = intern_atom(conn, only_if_exists, name.len() as u16, name.as_ptr().cast());
                let reply = intern_atom_reply(conn, cookie, std::ptr::null_mut());
                if reply.is_null() {
                    return 0;
                }
                let atom = (*reply).atom;
                libc::free(reply.cast());
                atom
            };
            let protocols = atom("WM_PROTOCOLS", 1);
            let delete_atom = atom("WM_DELETE_WINDOW", 0);
            const PROP_MODE_REPLACE: u8 = 0;
            const ATOM_ATOM: u32 = 4;
            const ATOM_STRING: u32 = 31;
            const ATOM_WM_NAME: u32 = 39;
            if protocols != 0 && delete_atom != 0 {
                change_property(conn, PROP_MODE_REPLACE, window, protocols, ATOM_ATOM, 32, 1, (&delete_atom as *const u32).cast());
            }
            change_property(conn, PROP_MODE_REPLACE, window, ATOM_WM_NAME, ATOM_STRING, 8, title.len() as u32, title.as_ptr().cast());
            map_window(conn, window);
            flush(conn);
            Ok(Xcb { conn, window, delete_atom, poll_for_event, disconnect })
        }
    }

    /// Drains pending events; `true` when the window was closed or Esc pressed.
    fn quit_requested(&self) -> bool {
        let mut quit = false;
        loop {
            // SAFETY: `conn` is live; each returned event is malloc'd by libxcb and freed here.
            let event = unsafe { (self.poll_for_event)(self.conn) };
            if event.is_null() {
                return quit;
            }
            let e = unsafe { &*event };
            match e.response_type & 0x7f {
                // KeyPress: keycode 9 is Escape on every X server's evdev keymap.
                2 if e.detail == 9 => quit = true,
                // ClientMessage: words[2] is data32[0].
                33 if e.words[2] == self.delete_atom => quit = true,
                _ => {}
            }
            unsafe { libc::free(event.cast()) };
        }
    }
}

impl Drop for Xcb {
    fn drop(&mut self) {
        // SAFETY: the surface and everything presenting to it are gone before this drops.
        unsafe { (self.disconnect)(self.conn) };
    }
}

// ---------------------------------------------------------------------------------------
// Vulkan
// ---------------------------------------------------------------------------------------

const FRAMES_IN_FLIGHT: usize = 2;

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u32,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    /// One per image: signalled by the copy into it, waited on by its present.
    copied: Vec<vk::Semaphore>,
}

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_fn: khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    swapchain_fn: khr::Swapchain,
    queue: vk::Queue,
    pool: vk::CommandPool,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

fn check<T>(r: Result<T, vk::Result>, what: &str) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("pan: {what} failed: {e:?}");
        std::process::exit(1);
    })
}

impl Gpu {
    fn new(xcb: &Xcb, requested_mode: vk::PresentModeKHR) -> Gpu {
        // SAFETY: loads the system Vulkan loader, the same as every other example here.
        let entry = unsafe { ash::Entry::load() }.unwrap_or_else(|e| {
            eprintln!("pan: no Vulkan loader: {e}");
            std::process::exit(1)
        });
        let app_name = c"neural-forge-pan";
        let app_info = vk::ApplicationInfo::builder().application_name(app_name).api_version(vk::API_VERSION_1_1);
        let extensions = [khr::Surface::name().as_ptr(), khr::XcbSurface::name().as_ptr()];
        let create_info = vk::InstanceCreateInfo::builder().application_info(&app_info).enabled_extension_names(&extensions);
        let instance = check(unsafe { entry.create_instance(&create_info, None) }, "vkCreateInstance");
        let surface_fn = khr::Surface::new(&entry, &instance);
        let xcb_fn = khr::XcbSurface::new(&entry, &instance);
        let surface_info = vk::XcbSurfaceCreateInfoKHR::builder().connection(xcb.conn).window(xcb.window);
        let surface = check(unsafe { xcb_fn.create_xcb_surface(&surface_info, None) }, "vkCreateXcbSurfaceKHR");

        // Prefer a discrete GPU, then integrated, then anything (lavapipe), that can present here.
        let devices = check(unsafe { instance.enumerate_physical_devices() }, "vkEnumeratePhysicalDevices");
        let rank = |t: vk::PhysicalDeviceType| match t {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        };
        let mut candidates: Vec<(u32, vk::PhysicalDevice, u32)> = Vec::new();
        for &pd in &devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            for (i, f) in families.iter().enumerate() {
                let present = unsafe { surface_fn.get_physical_device_surface_support(pd, i as u32, surface) }.unwrap_or(false);
                if present && f.queue_flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
                    candidates.push((rank(props.device_type), pd, i as u32));
                    break;
                }
            }
        }
        candidates.sort_by_key(|c| c.0);
        let Some(&(_, physical_device, family)) = candidates.first() else {
            eprintln!("pan: no Vulkan device can present to this window");
            std::process::exit(1)
        };
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        eprintln!("pan: device {:?}", unsafe { CStr::from_ptr(props.device_name.as_ptr()) });

        let queue_info = [vk::DeviceQueueCreateInfo::builder().queue_family_index(family).queue_priorities(&[1.0]).build()];
        let device_extensions = [khr::Swapchain::name().as_ptr()];
        let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_info).enabled_extension_names(&device_extensions);
        let device = check(unsafe { instance.create_device(physical_device, &device_info, None) }, "vkCreateDevice");
        let queue = unsafe { device.get_device_queue(family, 0) };
        // Loaded through the instance: ash 0.37's `Swapchain::new` asks vkGetDeviceProcAddr for
        // an instance-level function too, which the validation layer reports. The loader's
        // trampolines still dispatch through every layer's device hooks (the present included).
        let swapchain_fn = khr::Swapchain::new_from_instance(&entry, &instance, device.handle());
        let pool_info = vk::CommandPoolCreateInfo::builder().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = check(unsafe { device.create_command_pool(&pool_info, None) }, "vkCreateCommandPool");

        let formats = check(unsafe { surface_fn.get_physical_device_surface_formats(physical_device, surface) }, "surface formats");
        let format = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB]
            .into_iter()
            .find_map(|want| formats.iter().find(|f| f.format == want).copied())
            .unwrap_or_else(|| {
                eprintln!("pan: the surface offers no 8-bit RGBA/BGRA format ({formats:?})");
                std::process::exit(1)
            });
        let modes = check(unsafe { surface_fn.get_physical_device_surface_present_modes(physical_device, surface) }, "present modes");
        let present_mode = if modes.contains(&requested_mode) {
            requested_mode
        } else {
            eprintln!("pan: present mode {requested_mode:?} not supported here ({modes:?}); using FIFO");
            vk::PresentModeKHR::FIFO
        };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        Gpu { _entry: entry, instance, surface_fn, surface, physical_device, device, swapchain_fn, queue, pool, format, present_mode, mem_props }
    }

    fn bgr(&self) -> bool {
        matches!(self.format.format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB)
    }

    fn create_swapchain(&self, width: u32, height: u32, old: vk::SwapchainKHR) -> Swapchain {
        let caps = check(unsafe { self.surface_fn.get_physical_device_surface_capabilities(self.physical_device, self.surface) }, "surface caps");
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            eprintln!("pan: the surface does not allow TRANSFER_DST swapchain images");
            std::process::exit(1);
        }
        let mut usage = vk::ImageUsageFlags::TRANSFER_DST;
        if caps.supported_usage_flags.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        let mut count = (caps.min_image_count + 1).max(3);
        if caps.max_image_count != 0 {
            count = count.min(caps.max_image_count);
        }
        let alpha = [vk::CompositeAlphaFlagsKHR::OPAQUE, vk::CompositeAlphaFlagsKHR::INHERIT, vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED, vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED]
            .into_iter()
            .find(|a| caps.supported_composite_alpha.contains(*a))
            .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);
        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(old);
        let handle = check(unsafe { self.swapchain_fn.create_swapchain(&info, None) }, "vkCreateSwapchainKHR");
        let images = check(unsafe { self.swapchain_fn.get_swapchain_images(handle) }, "vkGetSwapchainImagesKHR");
        let copied = images.iter().map(|_| check(unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }, "vkCreateSemaphore")).collect();
        Swapchain { handle, images, extent, copied }
    }

    unsafe fn destroy_swapchain(&self, sc: Swapchain) {
        unsafe {
            for s in sc.copied {
                self.device.destroy_semaphore(s, None);
            }
            self.swapchain_fn.destroy_swapchain(sc.handle, None);
        }
    }

    fn create_staging(&self, bytes: u64) -> Staging {
        let d = &self.device;
        let info = vk::BufferCreateInfo::builder().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_SRC).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = check(unsafe { d.create_buffer(&info, None) }, "vkCreateBuffer");
        let reqs = unsafe { d.get_buffer_memory_requirements(buffer) };
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = (0..self.mem_props.memory_type_count)
            .find(|&i| reqs.memory_type_bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(wanted))
            .expect("no host-visible coherent memory");
        let memory = check(unsafe { d.allocate_memory(&vk::MemoryAllocateInfo::builder().allocation_size(reqs.size).memory_type_index(type_index), None) }, "vkAllocateMemory");
        check(unsafe { d.bind_buffer_memory(buffer, memory, 0) }, "vkBindBufferMemory");
        let ptr = check(unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }, "vkMapMemory").cast();
        let alloc = vk::CommandBufferAllocateInfo::builder().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        let cmd = check(unsafe { d.allocate_command_buffers(&alloc) }, "vkAllocateCommandBuffers")[0];
        let fence = check(unsafe { d.create_fence(&vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED), None) }, "vkCreateFence");
        let acquired = check(unsafe { d.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }, "vkCreateSemaphore");
        Staging { buffer, memory, ptr, cmd, fence, acquired }
    }

    unsafe fn destroy_staging(&self, s: Staging) {
        unsafe {
            self.device.destroy_semaphore(s.acquired, None);
            self.device.destroy_fence(s.fence, None);
            self.device.free_command_buffers(self.pool, &[s.cmd]);
            self.device.destroy_buffer(s.buffer, None);
            self.device.free_memory(s.memory, None);
        }
    }

    /// Records "staging buffer -> `image`, then ready to present".
    fn record_copy(&self, s: &Staging, image: vk::Image, extent: vk::Extent2D) {
        let d = &self.device;
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let to_dst = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .build();
        let to_present = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)
            .build();
        let region = vk::BufferImageCopy::builder()
            .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
            .image_extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .build();
        // SAFETY: `s.cmd` is not in use (its fence was waited on); all handles are live.
        unsafe {
            check(d.reset_command_buffer(s.cmd, vk::CommandBufferResetFlags::empty()), "vkResetCommandBuffer");
            check(d.begin_command_buffer(s.cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)), "vkBeginCommandBuffer");
            // The acquire semaphore is waited on at TRANSFER; this barrier's first scope chains to it.
            d.cmd_pipeline_barrier(s.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
            d.cmd_copy_buffer_to_image(s.cmd, s.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            d.cmd_pipeline_barrier(s.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[], &[], &[to_present]);
            check(d.end_command_buffer(s.cmd), "vkEndCommandBuffer");
        }
    }
}

fn main() {
    let args = parse_args();
    let xcb = Xcb::open(args.width, args.height, "Neural Forge pan").unwrap_or_else(|e| {
        eprintln!("pan: {e}");
        std::process::exit(1)
    });
    let gpu = Gpu::new(&xcb, args.present_mode);
    eprintln!(
        "pan: {}x{} {:?} {:?}, {} px/frame (x) {} px/frame (y), grain {}, contrast {}",
        args.width,
        args.height,
        gpu.format.format,
        gpu.present_mode,
        args.px_per_frame,
        args.px_per_frame / 2.0,
        args.grain,
        args.contrast
    );
    let t = std::time::Instant::now();
    let tex = build_texture(args.contrast, gpu.bgr());
    eprintln!("pan: texture {TW}x{TH} built in {:.0?}", t.elapsed());

    let mut swapchain = gpu.create_swapchain(args.width, args.height, vk::SwapchainKHR::null());
    let mut staging_bytes = u64::from(swapchain.extent.width) * u64::from(swapchain.extent.height) * 4;
    let mut staging: Vec<Staging> = (0..FRAMES_IN_FLIGHT).map(|_| gpu.create_staging(staging_bytes)).collect();
    let d = &gpu.device;

    let mut frame: u64 = 0;
    let mut slot = 0usize;
    let mut window_start = std::time::Instant::now();
    let mut window_frames = 0u32;
    let mut recreate = false;
    // The layer's shared memory, opened only for `--capture-at`; `Some` while the request
    // is outstanding.
    let mut capture: Option<neural_forge_protocol::mapping::Mapping> = None;
    while args.frames == 0 || frame < args.frames {
        if xcb.quit_requested() {
            break;
        }
        if recreate {
            check(unsafe { d.device_wait_idle() }, "vkDeviceWaitIdle");
            let old = swapchain;
            swapchain = gpu.create_swapchain(args.width, args.height, old.handle);
            unsafe { gpu.destroy_swapchain(old) };
            let bytes = u64::from(swapchain.extent.width) * u64::from(swapchain.extent.height) * 4;
            if bytes != staging_bytes {
                for s in staging.drain(..) {
                    unsafe { gpu.destroy_staging(s) };
                }
                staging_bytes = bytes;
                staging = (0..FRAMES_IN_FLIGHT).map(|_| gpu.create_staging(staging_bytes)).collect();
            }
            recreate = false;
        }
        let s = &staging[slot];
        check(unsafe { d.wait_for_fences(&[s.fence], true, u64::MAX) }, "vkWaitForFences");
        let image_index = match unsafe { gpu.swapchain_fn.acquire_next_image(swapchain.handle, u64::MAX, s.acquired, vk::Fence::null()) } {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                recreate = true;
                continue;
            }
            Err(e) => {
                eprintln!("pan: vkAcquireNextImageKHR failed: {e:?}");
                std::process::exit(1);
            }
        };
        let (w, h) = (swapchain.extent.width as usize, swapchain.extent.height as usize);
        // SAFETY: the staging memory is mapped, `w*h*4` bytes, and not read by the GPU (fence).
        let pixels = unsafe { std::slice::from_raw_parts_mut(s.ptr, w * h) };
        render_frame(pixels, &tex, w, h, frame, &args);
        if let Some(dir) = &args.save {
            save_png(&dir.join(format!("{frame:06}.png")), pixels, w, h, gpu.bgr());
        }
        let image = swapchain.images[image_index as usize];
        gpu.record_copy(s, image, swapchain.extent);
        let copied = swapchain.copied[image_index as usize];
        let stage = [vk::PipelineStageFlags::TRANSFER];
        let submit = vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(&s.acquired))
            .wait_dst_stage_mask(&stage)
            .command_buffers(std::slice::from_ref(&s.cmd))
            .signal_semaphores(std::slice::from_ref(&copied))
            .build();
        check(unsafe { d.reset_fences(&[s.fence]) }, "vkResetFences");
        check(unsafe { d.queue_submit(gpu.queue, &[submit], s.fence) }, "vkQueueSubmit");
        if args.capture_at == Some(frame) {
            match neural_forge_protocol::mapping::open() {
                Ok(mapping) => {
                    mapping.header().capture_request.store(args.capture_frames, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("pan: requested a {}-frame capture at frame {frame}", args.capture_frames);
                    capture = Some(mapping);
                }
                Err(e) => eprintln!("pan: cannot open the layer's shared memory for --capture-at: {e}"),
            }
        }
        let present = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&copied))
            .swapchains(std::slice::from_ref(&swapchain.handle))
            .image_indices(std::slice::from_ref(&image_index));
        let presented = unsafe { gpu.swapchain_fn.queue_present(gpu.queue, &present) };
        if capture.as_ref().is_some_and(|m| m.header().capture_request.load(std::sync::atomic::Ordering::Relaxed) == 0) {
            eprintln!("pan: the layer took the capture request on frame {frame}: the series is frames {frame}..{}", frame + u64::from(args.capture_frames) - 1);
            capture = None;
        }
        match presented {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => recreate = true,
            Err(e) => {
                eprintln!("pan: vkQueuePresentKHR failed: {e:?}");
                std::process::exit(1);
            }
        }
        frame += 1;
        slot = (slot + 1) % FRAMES_IN_FLIGHT;
        window_frames += 1;
        let elapsed = window_start.elapsed();
        if elapsed >= std::time::Duration::from_secs(1) {
            eprintln!("pan: frame {frame}: {:.1} fps", f64::from(window_frames) / elapsed.as_secs_f64());
            window_start = std::time::Instant::now();
            window_frames = 0;
        }
    }
    eprintln!("pan: presented {frame} frames");

    // SAFETY: everything is idle after the wait; destroyed in reverse creation order.
    unsafe {
        check(d.device_wait_idle(), "vkDeviceWaitIdle");
        for s in staging {
            gpu.destroy_staging(s);
        }
        gpu.destroy_swapchain(swapchain);
        d.destroy_command_pool(gpu.pool, None);
        d.destroy_device(None);
        gpu.surface_fn.destroy_surface(gpu.surface, None);
        gpu.instance.destroy_instance(None);
    }
    drop(xcb);
}
