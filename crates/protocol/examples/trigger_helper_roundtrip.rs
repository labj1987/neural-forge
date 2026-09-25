//! Drives one real request/response round trip against an already-running helper,
//! playing the layer's own role by hand: writes a synthetic proxy frame, bumps
//! `seq_req`, waits for `seq_resp`, and reports what came back. No game, no layer,
//! no capture -- just this process and the helper on the other end of the mapping.
//!
//! Built to validate `docs/EXTERNAL_MEMORY_HOST_DESIGN.md`'s helper-side import without a
//! live GTA session: `vkcube` can never exercise it (the render tap never engages for
//! it, so the layer never advances `seq_req` either -- see `docs/HARDWARE_VALIDATION.md`),
//! and this is the only other way to make the helper actually build `FrameResources`
//! against a real proxy/answer region and log whether the import succeeded.
//!
//! Respects `$NEURAL_FORGE_SHM`/`$NEURAL_FORGE_UID`, same as every other tool in this
//! workspace. Maps the *full* `shm_total_bytes()` region (unlike
//! `neural_forge_protocol::mapping::open`, which only maps the header -- the GUI/CLI's own
//! use case never needs the pixel regions) -- same reasoning `read_mapping.rs` already
//! uses for going around the library's own (header-only) `mapping` module.
//!
//! Takes an optional third argument, 0 or 1, for which protocol v3 wire slot to drive
//! (`docs/PROTOCOL_V3_DESIGN.md`) -- defaults to 0, matching this tool's pre-v3 behavior.
//! Run it twice concurrently with different slots to confirm the helper answers both
//! independently against real hardware, the same thing
//! `neural_forge_layer::shm::tests::the_two_slots_are_fully_independent` already proves
//! against a fake helper.

use std::os::fd::AsRawFd;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use neural_forge_protocol::{answer_offset_slot, proxy_offset_slot, shm_default_path, shm_total_bytes, ShmHeader};

fn main() {
    let path = neural_forge_protocol::env::var("NEURAL_FORGE_SHM").filter(|s| !s.is_empty()).unwrap_or_else(shm_default_path);
    let width: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let height: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let slot: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);

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

    let proxy_format = neural_forge_protocol::enums::proxy_format::RGBA8;
    let frame_bytes = (width as usize) * (height as usize) * 4;
    assert!(frame_bytes <= neural_forge_protocol::MAX_FRAME, "requested frame too large for MAX_FRAME");

    // A real, checkable, non-zero pattern -- not just zero-filled, so a real
    // (as opposed to a silently-echoed-back) evaluation is at least plausible from
    // the byte pattern alone, same spirit as the layer-side `DirectCapture` test's own
    // known fill color.
    // SAFETY: `base.add(proxy_offset_slot(slot))` is in bounds for `frame_bytes <=
    // MAX_FRAME` by the mapping's own region layout.
    unsafe {
        let proxy = std::slice::from_raw_parts_mut(base.add(proxy_offset_slot(slot)), frame_bytes);
        for (i, chunk) in proxy.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *chunk = [(i % 256) as u8, 100, 150, 255];
        }
    }

    hdr.width_slot(slot).store(width, Ordering::Relaxed);
    hdr.height_slot(slot).store(height, Ordering::Relaxed);
    hdr.proxy_format_slot(slot).store(proxy_format, Ordering::Relaxed);
    hdr.apply_model.store(1, Ordering::Relaxed);
    hdr.enabled.store(1, Ordering::Relaxed);

    let seq = hdr.seq_req_slot(slot).load(Ordering::Relaxed).wrapping_add(1).max(1);
    println!("trigger_helper_roundtrip: slot {slot}: {width}x{height}, requesting seq={seq}");
    hdr.seq_req_slot(slot).store(seq, Ordering::Release);

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if hdr.seq_resp_slot(slot).load(Ordering::Acquire) == seq {
            break;
        }
        if Instant::now() > deadline {
            eprintln!("trigger_helper_roundtrip: timed out waiting for seq_resp -- is the helper actually running and model_up?");
            std::process::exit(1);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let ok = hdr.seq_ok.load(Ordering::Relaxed) == seq;
    println!("trigger_helper_roundtrip: got a response (seq_ok match: {ok})");

    // SAFETY: same reasoning as the proxy write above, mirrored for the answer region.
    let answer = unsafe { std::slice::from_raw_parts(base.add(answer_offset_slot(slot)), frame_bytes) };
    let all_zero = answer.iter().all(|&b| b == 0);
    println!(
        "trigger_helper_roundtrip: answer first 16 bytes: {:?}{}",
        &answer[..16.min(answer.len())],
        if all_zero { " (all zero -- suspicious for a real evaluation)" } else { "" }
    );
    println!("trigger_helper_roundtrip: model_up={} helper_state={}", hdr.model_up.load(Ordering::Relaxed), hdr.helper_state.load(Ordering::Relaxed));
}
