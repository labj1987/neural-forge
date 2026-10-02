//! Drives real request/response round trips against an already-running helper,
//! playing the layer's own role by hand: writes a proxy frame, bumps `seq_req`, waits for
//! `seq_resp`, and reports what came back. No game, no layer, no capture -- just this
//! process and the helper on the other end of the mapping.
//!
//! Built to validate `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`'s helper-side import without a
//! live GTA session: `vkcube` can never exercise it (the render tap never engages for
//! it, so the layer never advances `seq_req` either -- see `docs/HARDWARE_VALIDATION.md`),
//! and this is the only other way to make the helper actually build `FrameResources`
//! against a real proxy/answer region and log whether the import succeeded.
//!
//! Two modes:
//!
//! ```text
//! trigger_helper_roundtrip [W H [SLOT]]
//! trigger_helper_roundtrip --rgba16f FILE --width W --height H [--slot N] [--out FILE] [--repeat N]
//! ```
//!
//! The first sends one synthetic RGBA8 frame (default 64x64, slot 0), as this tool always did.
//!
//! The second is the pre-upscaler HDR check (`docs/PRE_UPSCALER_DESIGN.md`, experiment E1):
//! FILE is a raw scene-linear frame, W*H*8 bytes of little-endian half floats (R, G, B, A per
//! pixel), e.g. a GTA DLSS colour input dumped by the layer. It is sent as an RGBA16F proxy,
//! which makes the helper build the feature with `DLSSNR.Hdr=1`. An odd W or H is padded by
//! repeating the last column/row (the helper needs even sizes) and the answer cropped back.
//! The frame is sent `--repeat` times (default 4: the first request after a size or format
//! change may be echoed while the feature builds, and the model's history settles over a few
//! frames); each round says whether the answer was a real evaluation or an echo of the input.
//! For the last answer it prints per-channel min/max/mean of input and answer (finite values),
//! NaN and Inf counts, and the mean |answer - input| per channel, and writes the answer, in the
//! input's format and size, to `--out`.
//!
//! Respects `$NEURAL_FORGE_SHM`/`$NEURAL_FORGE_UID`, same as every other tool in this
//! workspace. Maps the *full* `shm_total_bytes()` region (unlike
//! `neural_forge_protocol::mapping::open`, which only maps the header -- the GUI/CLI's own
//! use case never needs the pixel regions) -- same reasoning `read_mapping.rs` already
//! uses for going around the library's own (header-only) `mapping` module.
//!
//! SLOT picks the protocol v3 wire slot to drive (`docs/PROTOCOL_V3_DESIGN.md`), default 0.
//! Run it twice concurrently with different slots to confirm the helper answers both
//! independently against real hardware, the same thing
//! `neural_forge_layer::shm::tests::the_two_slots_are_fully_independent` already proves
//! against a fake helper.

use std::os::fd::AsRawFd;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use neural_forge_protocol::enums::proxy_format;
use neural_forge_protocol::{answer_offset_slot, proxy_offset_slot, shm_default_path, shm_total_bytes, ShmHeader};

/// What to send, from the command line.
struct Args {
    width: u32,
    height: u32,
    slot: usize,
    /// `--rgba16f FILE`: the raw half-float frame; `None` for the synthetic RGBA8 frame.
    rgba16f: Option<String>,
    out: Option<String>,
    repeat: u32,
}

fn usage() -> ! {
    eprintln!(
        "usage: trigger_helper_roundtrip [W H [SLOT]]\n       trigger_helper_roundtrip --rgba16f FILE --width W --height H [--slot N] [--out FILE] [--repeat N]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if !raw.iter().any(|a| a.starts_with("--")) {
        // The original positional form.
        let num = |i: usize, default: u32| raw.get(i).map(|s| s.parse().unwrap_or_else(|_| usage())).unwrap_or(default);
        return Args { width: num(0, 64), height: num(1, 64), slot: num(2, 0) as usize, rgba16f: None, out: None, repeat: 1 };
    }
    let mut args = Args { width: 0, height: 0, slot: 0, rgba16f: None, out: None, repeat: 4 };
    let mut it = raw.into_iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--rgba16f" => args.rgba16f = Some(value()),
            "--out" => args.out = Some(value()),
            "--width" => args.width = value().parse().unwrap_or_else(|_| usage()),
            "--height" => args.height = value().parse().unwrap_or_else(|_| usage()),
            "--slot" => args.slot = value().parse().unwrap_or_else(|_| usage()),
            "--repeat" => args.repeat = value().parse::<u32>().unwrap_or_else(|_| usage()).max(1),
            _ => usage(),
        }
    }
    if args.width == 0 || args.height == 0 || args.slot > 1 {
        usage();
    }
    args
}

/// IEEE 754 binary16 bits to `f32`, exact (subnormals, infinities and NaN included). The
/// helper's own decoder is `neural_forge_helper::hdr::f16_to_f32`; this example cannot depend
/// on the helper crate, so it carries its own.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => mant * 2f32.powi(-24),
        0x1f if mant == 0.0 => f32::INFINITY,
        0x1f => f32::NAN,
        e => (1.0 + mant / 1024.0) * 2f32.powi(i32::from(e) - 15),
    }
}

fn halves(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b)))
}

/// Per-channel statistics of one RGBA16F frame.
#[derive(Default, Clone, Copy)]
struct Channel {
    min: f32,
    max: f32,
    sum: f64,
    finite: u64,
    nan: u64,
    inf: u64,
}

fn channel_stats(frame: &[u8]) -> [Channel; 4] {
    let mut c = [Channel { min: f32::INFINITY, max: f32::NEG_INFINITY, ..Default::default() }; 4];
    for (i, v) in halves(frame).enumerate() {
        let ch = &mut c[i % 4];
        if v.is_nan() {
            ch.nan += 1;
        } else if v.is_infinite() {
            ch.inf += 1;
        } else {
            ch.min = ch.min.min(v);
            ch.max = ch.max.max(v);
            ch.sum += f64::from(v);
            ch.finite += 1;
        }
    }
    c
}

fn print_stats(label: &str, stats: &[Channel; 4]) {
    for (name, ch) in ["R", "G", "B", "A"].iter().zip(stats) {
        let mean = if ch.finite > 0 { ch.sum / ch.finite as f64 } else { f64::NAN };
        println!(
            "  {label} {name}: min={:.5} max={:.5} mean={mean:.5} nan={} inf={}",
            ch.min, ch.max, ch.nan, ch.inf
        );
    }
}

/// Mean |answer - input| per channel over the samples where both are finite.
fn mean_abs_diff(input: &[u8], answer: &[u8]) -> [f64; 4] {
    let mut sum = [0f64; 4];
    let mut n = [0u64; 4];
    for (i, (a, b)) in halves(input).zip(halves(answer)).enumerate() {
        if a.is_finite() && b.is_finite() {
            sum[i % 4] += f64::from((b - a).abs());
            n[i % 4] += 1;
        }
    }
    std::array::from_fn(|k| if n[k] > 0 { sum[k] / n[k] as f64 } else { f64::NAN })
}

/// `frame` (`w`x`h`, `bpp` bytes per pixel) padded to even dimensions by repeating its last
/// column and row.
fn pad_even(frame: &[u8], w: u32, h: u32, bpp: usize) -> (Vec<u8>, u32, u32) {
    let (pw, ph) = (w + (w & 1), h + (h & 1));
    let (w, h, pw_us) = (w as usize, h as usize, pw as usize);
    let mut out = Vec::with_capacity(pw_us * ph as usize * bpp);
    for y in 0..ph as usize {
        let row = &frame[y.min(h - 1) * w * bpp..][..w * bpp];
        out.extend_from_slice(row);
        if pw_us > w {
            out.extend_from_slice(&row[(w - 1) * bpp..]);
        }
    }
    (out, pw, ph)
}

/// The top-left `w`x`h` of a `pw`-wide frame.
fn crop(frame: &[u8], pw: u32, w: u32, h: u32, bpp: usize) -> Vec<u8> {
    (0..h as usize).flat_map(|y| frame[y * pw as usize * bpp..][..w as usize * bpp].iter().copied()).collect()
}

fn main() {
    let args = parse_args();
    let path = neural_forge_protocol::env::var("NEURAL_FORGE_SHM").filter(|s| !s.is_empty()).unwrap_or_else(shm_default_path);
    let slot = args.slot;

    // The frame to send, at the (even) size the helper sees.
    let (format, input, width, height) = match &args.rgba16f {
        None => {
            // A real, checkable, non-zero pattern -- not just zero-filled, so a real
            // (as opposed to a silently-echoed-back) evaluation is at least plausible from
            // the byte pattern alone, same spirit as the layer-side `DirectCapture` test's own
            // known fill color.
            let mut frame = vec![0u8; args.width as usize * args.height as usize * 4];
            for (i, chunk) in frame.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *chunk = [(i % 256) as u8, 100, 150, 255];
            }
            (proxy_format::RGBA8, frame, args.width, args.height)
        }
        Some(file) => {
            let raw = std::fs::read(file).unwrap_or_else(|e| panic!("failed to read {file}: {e}"));
            let want = args.width as usize * args.height as usize * 8;
            if raw.len() != want {
                eprintln!(
                    "trigger_helper_roundtrip: {file} is {} bytes; {}x{} RGBA16F needs exactly {want}",
                    raw.len(),
                    args.width,
                    args.height
                );
                std::process::exit(2);
            }
            let (padded, pw, ph) = pad_even(&raw, args.width, args.height, 8);
            if (pw, ph) != (args.width, args.height) {
                println!("trigger_helper_roundtrip: padded {}x{} to {pw}x{ph} (last column/row repeated)", args.width, args.height);
            }
            (proxy_format::RGBA16F, padded, pw, ph)
        }
    };
    let frame_bytes = input.len();
    assert!(frame_bytes <= neural_forge_protocol::MAX_FRAME, "requested frame too large for MAX_FRAME");
    assert!(neural_forge_protocol::frame_dims_valid(width, height, format), "{width}x{height} is outside what the helper accepts");

    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path)
        .unwrap_or_else(|e| panic!("failed to open {path}: {e} -- is a helper actually running against this NEURAL_FORGE_UID?"));
    let total = shm_total_bytes();
    // SAFETY: `file` is open read/write; mapping the full region a real helper/layer
    // would map is exactly this tool's point.
    let map = unsafe { libc::mmap(std::ptr::null_mut(), total, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0) };
    assert_ne!(map, libc::MAP_FAILED, "mmap failed");
    let base = map.cast::<u8>();
    // SAFETY: `map` is a valid mapping of at least `size_of::<ShmHeader>()` bytes
    // (`total` always is, by `shm_total_bytes()`'s own definition).
    let hdr = unsafe { &*map.cast::<ShmHeader>() };
    assert!(hdr.is_valid(), "not a valid NeuralForge mapping at {path}");

    // SAFETY: `base.add(proxy_offset_slot(slot))` is in bounds for `frame_bytes <=
    // MAX_FRAME` by the mapping's own region layout.
    unsafe { std::slice::from_raw_parts_mut(base.add(proxy_offset_slot(slot)), frame_bytes) }.copy_from_slice(&input);

    hdr.width_slot(slot).store(width, Ordering::Relaxed);
    hdr.height_slot(slot).store(height, Ordering::Relaxed);
    hdr.proxy_format_slot(slot).store(format, Ordering::Relaxed);
    hdr.apply_model.store(1, Ordering::Relaxed);
    hdr.enabled.store(1, Ordering::Relaxed);

    // SAFETY: same reasoning as the proxy write above, mirrored for the answer region.
    let answer = unsafe { std::slice::from_raw_parts(base.add(answer_offset_slot(slot)), frame_bytes) };
    let mut evaluated = false;
    for round in 1..=args.repeat {
        let seq = hdr.seq_req_slot(slot).load(Ordering::Relaxed).wrapping_add(1).max(1);
        println!("trigger_helper_roundtrip: slot {slot}: {width}x{height} format={format}, requesting seq={seq} ({round}/{})", args.repeat);
        let sent = Instant::now();
        hdr.seq_req_slot(slot).store(seq, Ordering::Release);

        // Generous: the first request after a size or format change builds the NGX feature.
        let deadline = sent + Duration::from_secs(30);
        while hdr.seq_resp_slot(slot).load(Ordering::Acquire) != seq {
            if Instant::now() > deadline {
                eprintln!("trigger_helper_roundtrip: timed out waiting for seq_resp -- is the helper actually running and model_up?");
                std::process::exit(1);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let ok = hdr.seq_ok.load(Ordering::Relaxed) == seq;
        // The helper echoes the proxy whenever the model did not run (feature still building,
        // wrong format, a failed evaluate).
        evaluated = answer != input.as_slice();
        println!(
            "trigger_helper_roundtrip: response in {:.1} ms (seq_ok match: {ok}): {} eval={:.2} ms features={}",
            sent.elapsed().as_secs_f64() * 1000.0,
            if evaluated { "evaluated" } else { "ECHO (answer identical to input; the model did not run)" },
            f32::from_bits(hdr.helper_eval_ms_bits.load(Ordering::Relaxed)),
            hdr.helper_features.load(Ordering::Relaxed)
        );
    }
    println!("trigger_helper_roundtrip: model_up={} helper_state={}", hdr.model_up.load(Ordering::Relaxed), hdr.helper_state.load(Ordering::Relaxed));

    if format == proxy_format::RGBA8 {
        let all_zero = answer.iter().all(|&b| b == 0);
        println!(
            "trigger_helper_roundtrip: answer first 16 bytes: {:?}{}",
            &answer[..16.min(answer.len())],
            if all_zero { " (all zero -- suspicious for a real evaluation)" } else { "" }
        );
        return;
    }

    // RGBA16F: crop back to the file's size, then report.
    let input = crop(&input, width, args.width, args.height, 8);
    let answer = crop(answer, width, args.width, args.height, 8);
    println!("trigger_helper_roundtrip: {}x{} RGBA16F, last answer {}", args.width, args.height, if evaluated { "evaluated" } else { "ECHOED" });
    print_stats("input ", &channel_stats(&input));
    print_stats("answer", &channel_stats(&answer));
    let diff = mean_abs_diff(&input, &answer);
    println!("  mean |answer - input|: R={:.5} G={:.5} B={:.5} A={:.5}", diff[0], diff[1], diff[2], diff[3]);
    if let Some(out) = &args.out {
        std::fs::write(out, &answer).unwrap_or_else(|e| panic!("failed to write {out}: {e}"));
        println!("trigger_helper_roundtrip: wrote {} bytes ({}x{} RGBA16F) to {out}", answer.len(), args.width, args.height);
    }
    if !evaluated {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn stats_count_nan_and_inf_apart() {
        // Two pixels: (1, NaN, +Inf, 1) and (3, 2, -Inf, 1).
        let px: Vec<u8> = [0x3c00u16, 0x7e00, 0x7c00, 0x3c00, 0x4200, 0x4000, 0xfc00, 0x3c00].iter().flat_map(|h| h.to_le_bytes()).collect();
        let s = channel_stats(&px);
        assert_eq!((s[0].min, s[0].max, s[0].finite), (1.0, 3.0, 2));
        assert_eq!((s[1].nan, s[1].finite), (1, 1));
        assert_eq!((s[2].inf, s[2].finite), (2, 0));
        let other: Vec<u8> = [0x4000u16, 0x7e00, 0x7c00, 0x3c00, 0x4200, 0x3c00, 0xfc00, 0x3c00].iter().flat_map(|h| h.to_le_bytes()).collect();
        let d = mean_abs_diff(&px, &other);
        assert_eq!(d[0], 0.5);
        assert_eq!(d[1], 1.0, "only the finite pair counts");
        assert!(d[2].is_nan());
        assert_eq!(d[3], 0.0);
    }

    #[test]
    fn odd_sizes_pad_and_crop_back() {
        // 3x1, one byte per pixel for readability.
        let (p, w, h) = pad_even(&[1, 2, 3], 3, 1, 1);
        assert_eq!((w, h), (4, 2));
        assert_eq!(p, vec![1, 2, 3, 3, 1, 2, 3, 3]);
        assert_eq!(crop(&p, w, 3, 1, 1), vec![1, 2, 3]);
        let (p, w, h) = pad_even(&[1, 2, 3, 4], 2, 2, 1);
        assert_eq!((w, h, p), (2, 2, vec![1, 2, 3, 4]));
    }
}
