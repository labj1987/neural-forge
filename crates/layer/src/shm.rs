//! The shared-memory round trip with the helper.
//!
//! This is the seam: everything here only ever touches `neural_forge_protocol::ShmHeader`'s
//! atomics, never anything Windows/NGX-specific. What answers on the other end of the
//! mapping — a Wine-wrapped helper today, a native one later — is none of this
//! module's business.
//!
//! Milestone 2 scope: the request/response sequence-number handshake and the fail-open
//! timing budget, ported for shape from upstream's `ShmOpen`/`ShmNeuralEnabled`/
//! `ShmProcessFrame`.
//!
//! Milestone 4 adds [`ShmClient::write_proxy`]/[`ShmClient::read_answer`]: the mapping
//! now covers the full `neural_forge_protocol::shm_total_bytes()` region (header plus both
//! `MAX_FRAME`-sized pixel buffers), not just the header, so the proxy/answer bytes
//! live in the same `mmap` this type already owns rather than a second one.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use neural_forge_protocol::{enums::helper_state, shm_default_path, MAX_FRAME, SHM_MAGIC};

/// The subset of `ShmHeader`'s composition fields `composition::apply::apply_rgba8`
/// needs, decoded once per frame from the raw atomics. See that field's own doc
/// comment in `neural_forge_protocol::header::ShmHeader` for what each one means.
pub struct CompositionSettings {
    pub colour_strength: f32,
    pub transfer_strength: f32,
    pub max_ratio: f32,
    /// See `ShmHeader::ghost_guard_bits`.
    pub ghost_guard: f32,
    pub compare: crate::composition::gpu::Compare,
    pub colour_trust: f32,
    pub ratio_smooth: f32,
    /// See `ShmHeader::transfer`: 0 classic, 1 matched residual, 2 native + edit.
    pub transfer: u32,
    /// Frame hold: keep working on the same captured frame.
    pub hold_frame: bool,
    /// Run the model on every Nth present (see `ShmHeader::model_interval`).
    pub model_interval: u32,
    pub debug_view: u32,
    /// View 5 only (see `ShmHeader::debug_scale_bits`).
    pub debug_scale: f32,
    pub apply_model: bool,
    pub neural_enabled: bool,
    /// What fraction of the frame's resolution the model works at -- see
    /// [`neural_forge_protocol::ShmHeader::working_scale_bits`]'s own doc comment.
    /// `1.0` (the default) means "model resolution == frame resolution", the only
    /// value this pipeline supported before 2026-09-17 -- callers that skip scaling
    /// whenever this is exactly `1.0` get the identical, unmodified code path.
    ///
    /// Read here but genuinely unused by any caller as of 2026-09-17 (hence the
    /// `dead_code` allow): a real attempt to wire it into `capture::run`'s per-present
    /// hot path used `composition::downscale::resample_rgba8` (a plain CPU resize) and
    /// measured it at 315-546 ms at GTA's own resolution -- far worse than the 87 ms
    /// PCIe-BAR bug this same session fixed, and unusable on the present thread. The
    /// resample function itself is real, tested, and kept (`downscale.rs`'s own tests);
    /// what's missing is a GPU-blit-based version (`vkCmdBlitImage`, a hardware unit,
    /// sub-millisecond) wired into `CapturePipeline`'s capture-side buffer and
    /// `composition::gpu`'s answer-upload step -- real Vulkan surgery in this project's
    /// most crash-prone area, deliberately not attempted unsupervised overnight. See
    /// `docs/GHOSTING_PLAN.md`'s step 1 for the full account and the corrected plan.
    #[allow(dead_code)]
    pub working_scale: f32,
    #[allow(dead_code)]
    pub scaling_downscaler: u32,
    /// What the model should treat as white, as the encode divides by it (see
    /// [`crate::composition::encode`]). The three header fields are one number here:
    /// the manual value times the scale, times the trim when the reading came from a
    /// meter rather than the slider. Clamped away from zero so the divide is safe.
    pub white_point: f32,
    /// Which curve the encode uses -- [`neural_forge_protocol::enums::reversible_mode`].
    pub reversible_mode: u32,
}

/// One process's connection to the mapping. Not `Clone` — there is exactly one of these
/// per device, guarded by a `Mutex` in [`crate::device::NeuralForgeDeviceInfo`].
/// How many bytes of each pixel region this process maps: the protocol's full `MAX_FRAME` on
/// 64-bit; on 32-bit a 4K 8-bit frame, so the whole set fits a 32-bit address space (a larger
/// frame is simply passed through, see [`region_capacity`]).
const REGION_CAP: usize = if cfg!(target_pointer_width = "64") { MAX_FRAME } else { 3840 * 2160 * 4 };

/// The largest frame this process can send or receive.
pub fn region_capacity() -> usize {
    REGION_CAP
}

#[derive(Clone, Copy, Debug)]
enum Region {
    Proxy0 = 0,
    Answer0 = 1,
    Proxy1 = 2,
    Answer1 = 3,
}

impl Region {
    const ALL: [Region; 4] = [Region::Proxy0, Region::Answer0, Region::Proxy1, Region::Answer1];
    fn proxy(slot: usize) -> Self {
        if slot == 0 { Region::Proxy0 } else { Region::Proxy1 }
    }
    fn answer(slot: usize) -> Self {
        if slot == 0 { Region::Answer0 } else { Region::Answer1 }
    }
    fn offset(self) -> usize {
        match self {
            Region::Proxy0 => neural_forge_protocol::proxy_offset_slot(0),
            Region::Answer0 => neural_forge_protocol::answer_offset_slot(0),
            Region::Proxy1 => neural_forge_protocol::proxy_offset_slot(1),
            Region::Answer1 => neural_forge_protocol::answer_offset_slot(1),
        }
    }
}

pub struct ShmClient {
    fd: Option<OwnedFd>,
    header: *mut neural_forge_protocol::ShmHeader,
    /// Where each pixel region is mapped in this process (proxy 0, answer 0, proxy 1,
    /// answer 1), `REGION_CAP` bytes each. One contiguous mapping on 64-bit; separate small
    /// mappings on 32-bit, where the protocol's full ~1.1 GB would not fit the address space.
    regions: [*mut u8; 4],
    path: String,
    timeouts: u32,
    ever_answered: bool,
    retry_after: Option<Instant>,
    last_control_seq: u32,
    last_heartbeat: u32,
    /// The helper heartbeat as last sampled by [`Self::helper_alive`], and when it last moved.
    alive_heartbeat: u32,
    alive_since: Option<Instant>,
    dead: bool,
    frames: u64,
    /// Per-slot: the request number and send time of a round trip issued via
    /// [`Self::begin_async_request`] that [`Self::poll_async_request`] hasn't yet
    /// resolved (answered or timed out). `None` means that slot has no request in
    /// flight -- callers use this to decide whether it's time to capture and send a
    /// new frame on that slot. Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) gives the wire
    /// two fully independent request/response slots instead of one, so this is an
    /// array of two, not a single value -- each slot still only ever has one
    /// outstanding request at a time.
    pending: [Option<(u32, Instant)>; 2],
}

// SAFETY: `header` points at a `MAP_SHARED` mapping that stays valid for the process's
// lifetime once opened (never unmapped or reallocated by this type), and every access
// through it goes through `ShmHeader`'s own atomics/seqlock-guarded accessors -- the
// same invariant that makes `ShmHeader` itself `Sync` (see `neural_forge_protocol::header`).
// `ShmClient` is always accessed from behind a `Mutex`, so only `Send` is needed, never
// concurrent access from two threads at once.
unsafe impl Send for ShmClient {}

impl Default for ShmClient {
    fn default() -> Self {
        Self {
            fd: None,
            header: std::ptr::null_mut(),
            regions: [std::ptr::null_mut(); 4],
            path: String::new(),
            timeouts: 0,
            ever_answered: false,
            retry_after: None,
            last_control_seq: 0,
            last_heartbeat: 0,
            alive_heartbeat: 0,
            alive_since: None,
            dead: false,
            frames: 0,
            pending: [None, None],
        }
    }
}

impl ShmClient {
    /// Cross-module test access to the raw header pointer -- `capture::tests` needs
    /// to poke `helper_state`/`seq_resp` directly to stand in for a fake helper, the
    /// same way this module's own tests do, but `header` is private to this module
    /// and those tests live in a sibling one. Test-only; never called from real code.
    #[cfg(test)]
    pub(crate) fn test_header_ptr(&self) -> usize {
        self.header as usize
    }

    /// Cross-module test access to [`Self::open_at`], same reasoning as
    /// [`Self::test_header_ptr`]: `capture::tests` needs a scratch-path open, exactly
    /// like this module's own tests already do, but the real method is private.
    #[cfg(test)]
    pub(crate) fn test_open_at(&mut self, path: &str) -> bool {
        self.open_at(path)
    }

    fn header(&self) -> Option<&neural_forge_protocol::ShmHeader> {
        // SAFETY: non-null only after a successful `open()`, which mmaps
        // `neural_forge_protocol::shm_total_bytes()` at this address and never unmaps it for
        // the lifetime of the process.
        (!self.header.is_null()).then(|| unsafe { &*self.header })
    }

    /// Whether it's worth paying for a real capture this frame at all. `false` once
    /// the helper has reported the model permanently unavailable (see
    /// `neural_forge_helper::ngx::ensure_feature`'s own one-shot-then-disable design) --
    /// capturing and writing back a frame nobody will ever evaluate is pure overhead
    /// (a full image<->buffer round trip plus a `memcpy` of the whole frame, every
    /// single present call) for zero chance of a different outcome. Reads a single
    /// already-mapped atomic; never blocks and never opens the mapping itself, so it's
    /// always safe to check before deciding whether to call
    /// [`crate::capture::run`] at all.
    pub fn model_known_unavailable(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.helper_state.load(Ordering::Relaxed) == helper_state::MODEL_FAILED
    }

    /// Consumes a pending "dump one matched before/after frame pair" request (see
    /// `neural_forge_protocol::header::ShmHeader::capture_request`'s own doc comment) --
    /// `true` at most once per request, since this resets it to 0 in the same atomic
    /// operation, so the very next present doesn't dump again for a request that was
    /// already served.
    pub fn take_capture_request(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.capture_request.swap(0, Ordering::Relaxed) != 0
    }

    /// Non-consuming version of [`Self::take_capture_request`] -- lets a caller decide
    /// *how* to produce this frame's composited bytes (a fast, GPU-only path with no
    /// CPU-visible result, vs. a path that leaves the result somewhere
    /// [`Self::take_capture_request`]'s caller can dump) before committing to either,
    /// without losing/duplicating the actual one-shot request in the process.
    pub fn capture_request_pending(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.capture_request.load(Ordering::Relaxed) != 0
    }

    /// The settings `composition::apply::apply_rgba8` needs, read fresh every frame
    /// (each is a single atomic load) so a live GUI change takes effect on the very
    /// next present rather than needing a restart. `None` before the mapping is open.
    /// Applies the configured in-game toggle on a physical key press.  It changes
    /// the same shared atomic the GUI uses, so the helper and layer agree immediately.
    pub fn poll_toggle_hotkey(&mut self, poller: &mut crate::hotkey::Poller) {
        let Some(hdr) = self.header() else { return };
        // Existing installations may have persisted the historical default `0`.
        // Treat it as F11 as well, so the new in-game control works immediately.
        let configured = hdr.toggle_key.load(Ordering::Relaxed);
        let key = if configured == 0 { 87 } else { configured };
        if poller.pressed(key) {
            let enabled = hdr.enabled.load(Ordering::Relaxed);
            hdr.enabled.store((enabled == 0) as u32, Ordering::Relaxed);
        }
    }

    /// The raster the helper says its last slot-0 answer was for (`None` before any).
    pub fn answered_dims(&self) -> Option<(u32, u32)> {
        let hdr = self.header()?;
        let dims = (hdr.answered_w.load(Ordering::Relaxed), hdr.answered_h.load(Ordering::Relaxed));
        (dims != (0, 0)).then_some(dims)
    }

    /// Folds a new white-meter reading into the published, smoothed value.
    pub fn publish_measured_white(&self, reading: f32) {
        let Some(hdr) = self.header() else { return };
        let prev = f32::from_bits(hdr.layer_measured_white_bits.load(Ordering::Relaxed));
        let next = if prev.is_finite() && prev > 1e-4 { prev + (reading - prev) * 0.3 } else { reading };
        hdr.layer_measured_white_bits.store(next.to_bits(), Ordering::Relaxed);
    }

    pub fn composition_settings(&self) -> Option<CompositionSettings> {
        let hdr = self.header()?;
        Some(CompositionSettings {
            colour_strength: f32::from_bits(hdr.colour_strength_bits.load(Ordering::Relaxed)),
            transfer_strength: f32::from_bits(hdr.transfer_strength_bits.load(Ordering::Relaxed)),
            max_ratio: f32::from_bits(hdr.max_ratio_bits.load(Ordering::Relaxed)),
            ghost_guard: f32::from_bits(hdr.ghost_guard_bits.load(Ordering::Relaxed)),
            transfer: hdr.transfer.load(Ordering::Relaxed),
            hold_frame: hdr.hold_frame.load(Ordering::Relaxed) != 0,
            model_interval: hdr.model_interval.load(Ordering::Relaxed).clamp(1, 4),
            colour_trust: f32::from_bits(hdr.colour_trust_bits.load(Ordering::Relaxed)),
            ratio_smooth: f32::from_bits(hdr.ratio_smooth_bits.load(Ordering::Relaxed)),
            compare: crate::composition::gpu::Compare {
                mode: hdr.compare_mode.load(Ordering::Relaxed),
                split: f32::from_bits(hdr.compare_split_bits.load(Ordering::Relaxed)),
                zoom: f32::from_bits(hdr.compare_zoom_bits.load(Ordering::Relaxed)),
                swap: hdr.compare_swap.load(Ordering::Relaxed),
            },
            debug_view: hdr.debug_view.load(Ordering::Relaxed),
            debug_scale: f32::from_bits(hdr.debug_scale_bits.load(Ordering::Relaxed)),
            apply_model: hdr.apply_model.load(Ordering::Relaxed) != 0,
            neural_enabled: hdr.neural_enabled(),
            working_scale: f32::from_bits(hdr.working_scale_bits.load(Ordering::Relaxed)),
            scaling_downscaler: hdr.scaling_downscaler.load(Ordering::Relaxed),
            white_point: {
                let manual = f32::from_bits(hdr.white_point_bits.load(Ordering::Relaxed));
                let scale = f32::from_bits(hdr.white_point_scale_bits.load(Ordering::Relaxed));
                let trim = f32::from_bits(hdr.white_point_trim_bits.load(Ordering::Relaxed));
                // The trim belongs to a measured reading, not to the slider -- keeping
                // them apart is upstream's own fix for sharing one stored value.
                // Measured: the meter's reading times the trim; manual: the slider. Either way times
                // the scale. (The measured source used to take the slider's value as well.)
                let measured_white = f32::from_bits(hdr.layer_measured_white_bits.load(Ordering::Relaxed));
                let measured = hdr.white_point_source.load(Ordering::Relaxed) != neural_forge_protocol::enums::white_point_source::MANUAL
                    && measured_white.is_finite()
                    && measured_white > 1e-4;
                let combined = if measured { measured_white * trim } else { manual } * scale;
                if combined.is_finite() && combined > 1e-4 { combined } else { 1.0 }
            },
            reversible_mode: hdr.reversible_mode.load(Ordering::Relaxed),
        })
    }

    /// Records what the proxy bytes about to be written actually are, for the given
    /// slot -- the helper (and, on the way back, this same layer reading the answer)
    /// needs `width`/`height`/`proxy_format` to know how many of the region's bytes
    /// are real for this frame, not the full `MAX_FRAME`-sized reservation. Call
    /// before [`Self::write_proxy`]/[`Self::begin_async_request`]/[`Self::try_round_trip`]
    /// (the last of which only ever uses slot 0) so the helper never observes the
    /// `seq_req` bump before it can see what raster it describes.
    ///
    /// `layer_attached`/`layer_heartbeat`/`layer_width`/`layer_height`/`layer_format`/
    /// `layer_frames` are status telemetry, not part of the handshake -- deliberately
    /// not per-slot; they just reflect whichever slot most recently captured.
    pub fn set_frame_info(&mut self, slot: usize, width: u32, height: u32, proxy_format: u32) {
        self.frames += 1;
        let frames = self.frames;
        let Some(hdr) = self.header() else { return };
        hdr.width_slot(slot).store(width, Ordering::Relaxed);
        hdr.height_slot(slot).store(height, Ordering::Relaxed);
        hdr.proxy_format_slot(slot).store(proxy_format, Ordering::Relaxed);

        // The layer's own "I am alive and capturing" telemetry -- mirrors what
        // `neural_forge_helper::main`'s loop already does for `hdr.helper_*`/`heartbeat`.
        // Nothing else in this crate ever wrote these fields before this (confirmed by
        // grep, 2026-09-10): `layer_attached` was declared, reset to 0 by
        // `ShmHeader::init_defaults`, and read by the GUI (`ui.rs`'s "not attached"
        // label) -- but never once set to 1 anywhere, so that label was always wrong,
        // regardless of whether the layer was actually attached. Confirmed on
        // `lordnikon` the same day: `/proc/<pid>/maps` and a live, advancing helper
        // frame counter both proved the real Vulkan layer was loaded and working the
        // whole time the GUI displayed "not attached". Setting this every frame (not
        // just once at `open()`) also survives a helper restart resetting the shared
        // header out from under an already-open, never-reconnecting layer -- exactly
        // what happened here: `open_at`'s own idempotent early return means a layer
        // that was already attached before the reset never calls it again to re-set a
        // one-shot flag.
        hdr.layer_attached.store(1, Ordering::Relaxed);
        hdr.layer_heartbeat.fetch_add(1, Ordering::Relaxed);
        hdr.layer_width.store(width, Ordering::Relaxed);
        hdr.layer_height.store(height, Ordering::Relaxed);
        hdr.layer_format.store(proxy_format, Ordering::Relaxed);
        neural_forge_protocol::store64(&hdr.layer_frames_lo, &hdr.layer_frames_hi, frames);
        // Restated now and then rather than once: a helper restart re-initialises the header and
        // clears it. Rarely, because it is a seqlock-guarded string write.
        if frames % 120 == 1 {
            let name = crate::ownership::process_name();
            if hdr.game_name() != name {
                hdr.set_game_name(name);
            }
        }
    }

    /// Publishes the layer's host-observed cost for a frame. This is deliberately a
    /// single atomic snapshot rather than per-frame logging: consumers can sample it
    /// through the GUI or CLI without adding I/O to the game's present path.
    pub fn publish_frame_timing(&self, total: Duration, composition_up: bool) {
        let Some(hdr) = self.header() else { return };
        hdr.layer_ms_bits.store(
            (total.as_secs_f64().mul_add(1_000.0, 0.0) as f32).to_bits(),
            Ordering::Relaxed,
        );
        hdr.layer_composition_up.store(u32::from(composition_up), Ordering::Relaxed);
    }

    /// Writes `bytes` (truncated to `MAX_FRAME`, same discipline as the free-text
    /// fields in `ShmHeader`) into the given slot's proxy region -- the frame the
    /// layer is about to hand the model. Call before bumping that slot's `seq_req`
    /// (via [`Self::begin_async_request`]/[`Self::try_round_trip`], the latter always
    /// slot 0): the helper only starts reading once it observes that bump, so there
    /// is no concurrent-write hazard to guard against the way the header's atomics do.
    ///
    /// # Safety
    /// Must only be called after a successful [`Self::open`]/[`Self::try_round_trip`].
    pub fn write_proxy(&self, slot: usize, bytes: &[u8]) {
        let Some((dst, cap)) = self.region(Region::proxy(slot)) else { return };
        let n = bytes.len().min(cap);
        // SAFETY: `base` is the start of this process's own mapping of the full
        // `shm_total_bytes()` region (see `open_at`); `proxy_offset_slot(slot)..+n` is
        // in bounds for any `n <= MAX_FRAME` and `slot < 2` by that region's own
        // definition.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                dst,
                n,
            );
        }
    }

    /// Reads up to `out.len()` (capped at `MAX_FRAME`) bytes back from the given
    /// slot's answer region into `out`, returning the number of bytes copied.
    /// Meaningful only after [`Self::poll_async_request`]/[`Self::try_round_trip`] has
    /// returned a real answer for the request this goes with -- reading it any
    /// earlier just observes whatever the helper last wrote (stale or all-zero),
    /// which is why this never blocks or checks sequence numbers itself; the caller
    /// already knows from the round trip's own return value whether there is a real
    /// answer to read.
    pub fn read_answer(&self, slot: usize, out: &mut [u8]) -> usize {
        let Some((src, cap)) = self.region(Region::answer(slot)) else { return 0 };
        let n = out.len().min(cap);
        // SAFETY: same reasoning as `write_proxy`, mirrored for the answer region.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src,
                out.as_mut_ptr(),
                n,
            );
        }
        n
    }

    /// A mapped pixel region and how many bytes of it this process can reach.
    fn region(&self, which: Region) -> Option<(*mut u8, usize)> {
        let p = self.regions[which as usize];
        (!p.is_null()).then_some((p, REGION_CAP))
    }

    /// The given slot's proxy region address and capacity within this process's
    /// mapping -- `None` before [`Self::open`]/[`Self::open_at`] has actually mapped
    /// anything. For [`crate::capture::DirectCapture`]'s `VK_EXT_external_memory_host`
    /// import: the *only* legitimate reason anything outside this module needs this
    /// address at all, since every other caller goes through [`Self::write_proxy`]
    /// instead.
    pub fn proxy_region(&self, slot: usize) -> Option<(*mut u8, usize)> {
        // SAFETY: `pixel_base` plus `proxy_offset_slot(slot)` stays within the
        // `shm_total_bytes()` mapping `open_at` established, same reasoning as
        // `write_proxy`'s own pointer arithmetic.
        self.region(Region::proxy(slot))
    }

    /// Opens (or creates) the mapping if not already attached. Idempotent.
    pub fn open(&mut self) -> bool {
        if self.header().is_some() { return true; }
        let path = neural_forge_protocol::env::var("NEURAL_FORGE_SHM")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(shm_default_path);
        if !neural_forge_protocol::isolated_path(&path) || !ensure_private_parent_dir(&path) || !crate::ownership::claim(&path) { return false; }
        self.open_at(&path)
    }

    /// The actual implementation, taking the path explicitly so tests can point it at a
    /// scratch directory instead of `$NEURAL_FORGE_SHM`/the real `/tmp/neural-forge-$UID/` -- mutating
    /// process-wide environment variables from parallel `#[test]`s would race.
    fn open_at(&mut self, path: &str) -> bool {
        if self.header().is_some() {
            return true;
        }
        if !ensure_private_parent_dir(path) {
            crate::log!("[shm] refusing {path}: parent directory is not private");
            return false;
        }
        let Ok(c_path) = CString::new(path) else {
            return false;
        };
        // SAFETY: `c_path` is a valid, NUL-terminated C string for the duration of the
        // call. `O_NOFOLLOW` refuses to open through a symlink -- this file lives under
        // a world-writable /tmp, so that refusal matters.
        let raw_fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW,
                0o600,
            )
        };
        if raw_fd < 0 {
            crate::log!("[shm] open {path} failed");
            return false;
        }
        // SAFETY: `raw_fd` was just returned by a successful `open()` above and is not
        // owned anywhere else yet.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        let total = neural_forge_protocol::shm_total_bytes();
        // SAFETY: `stat` is a plain out-parameter; zero-initializing it is always valid
        // and `fstat` either fully populates it or returns an error we check.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let needs_truncate = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0
            || (st.st_size as usize) < total;
        if needs_truncate
            && unsafe { libc::ftruncate(fd.as_raw_fd(), total as libc::off_t) } != 0
        {
            crate::log!("[shm] ftruncate {path} failed");
            return false;
        }

        // SAFETY: `fd` is a valid, open file descriptor sized to at least `total` bytes
        // by the ftruncate above (or already that size); mapping the whole `total`
        // bytes (header plus both pixel regions) is always in-bounds. The mapping is
        // kept for the rest of the process's life, so the returned pointer stays valid
        // for as long as anything derived from it (`header()`'s `&ShmHeader`, or the
        // proxy/answer slices below) is used. A `MAP_SHARED` file mapping is a sparse,
        // page-cache-backed region -- reserving the full `MAX_FRAME*2` up front costs
        // no real memory beyond whatever pages an SDR session actually touches.
        let map_at = |offset: usize, len: usize| unsafe {
            let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), offset as libc::off_t);
            (p != libc::MAP_FAILED).then_some(p.cast::<u8>())
        };
        // 64-bit: the whole file in one mapping. 32-bit: the header, then each region on its own
        // at its fixed offset, capped (the offsets are page-aligned by construction).
        let (map, regions) = if cfg!(target_pointer_width = "64") {
            let Some(base) = map_at(0, total) else {
                crate::log!("[shm] mmap {path} failed");
                return false;
            };
            // SAFETY: every offset is inside the `total`-byte mapping just made.
            let regions = Region::ALL.map(|r| unsafe { base.add(r.offset()) });
            (base.cast::<libc::c_void>(), regions)
        } else {
            let Some(base) = map_at(0, neural_forge_protocol::HEADER_BYTES) else {
                crate::log!("[shm] mmap {path} (header) failed");
                return false;
            };
            let mut regions = [std::ptr::null_mut(); 4];
            for r in Region::ALL {
                let Some(p) = map_at(r.offset(), REGION_CAP) else {
                    crate::log!("[shm] mmap {path} region {r:?} failed");
                    return false;
                };
                regions[r as usize] = p;
            }
            (base.cast::<libc::c_void>(), regions)
        };
        self.regions = regions;

        let header = map as *mut neural_forge_protocol::ShmHeader;
        // SAFETY: just mapped above, `HEADER_BYTES` is large enough for `ShmHeader`
        // (enforced at compile time in `neural_forge_protocol`).
        let hdr = unsafe { &*header };
        if hdr.magic.load(Ordering::Relaxed) != SHM_MAGIC || !hdr.is_valid() {
            // A magic mismatch is some other mapping entirely (or garbage); a version
            // mismatch is a stale build of ours. Both get the same answer: reinitialize
            // rather than half-read a layout we don't agree on.
            hdr.init_defaults();
        }
        self.last_heartbeat = hdr.heartbeat.load(Ordering::Relaxed);
        self.last_control_seq = hdr.control_seq.load(Ordering::Relaxed);
        self.fd = Some(fd);
        self.header = header;
        self.path = path.to_string();
        crate::log!(
            "[shm] attached {} seq_req={} seq_resp={}",
            self.path,
            hdr.seq_req.load(Ordering::Relaxed),
            hdr.seq_resp.load(Ordering::Relaxed)
        );
        true
    }

    /// One request/response cycle: bump `seq_req`, wait up to a budget for `seq_resp` to
    /// catch up. Returns whether the helper answered in time.
    ///
    /// This is a fail-open state machine, same as upstream: a helper that never answers
    /// four times in a row is marked dead and not retried for 5 seconds, so a missing
    /// helper costs one short wait per frame rather than the full budget forever. A
    /// live-but-busy helper (building its first feature) gets a much longer budget on
    /// its very first frame, since that is expected to be slow.
    pub fn try_round_trip(&mut self) -> bool {
        if !self.open() {
            self.dead = true;
            return false;
        }
        self.round_trip_after_open()
    }

    /// Same as [`Self::try_round_trip`], but against an explicit path rather than
    /// `$NEURAL_FORGE_SHM`/the real runtime dir -- so tests can point it at a scratch
    /// directory without racing on process-wide environment variables.
    #[cfg(test)]
    fn try_round_trip_at(&mut self, path: &str) -> bool {
        if !self.open_at(path) {
            self.dead = true;
            return false;
        }
        self.round_trip_after_open()
    }

    fn round_trip_after_open(&mut self) -> bool {
        // Re-check liveness before every attempt: a dead connection can come back
        // either because its retry timer elapsed, or because the helper's control_seq
        // or heartbeat moved, meaning something changed on the other end worth trying
        // again for.
        if self.dead && !self.should_retry() {
            return false;
        }
        self.dead = false;

        let hdr = self.header().expect("just opened above");
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.dead = true;
            return false;
        }

        let req = hdr.seq_req.load(Ordering::Relaxed) + 1;
        std::sync::atomic::fence(Ordering::Release);
        hdr.seq_req.store(req, Ordering::Relaxed);

        let helper_present = hdr.helper_state.load(Ordering::Relaxed) != helper_state::STOPPED;
        // This blocks inside `vkQueuePresentKHR` (the debug-view/capture-request path),
        // so it is capped at one second even while the helper is still warming up --
        // the game's presents must never stall for ten. A slow first answer costs a
        // timeout that the retry logic below absorbs; the async path (`begin_async_request`)
        // keeps the longer warm-up allowance because it never blocks.
        let budget = if !helper_present {
            Duration::from_millis(20)
        } else {
            Duration::from_secs(1)
        };

        let start = Instant::now();
        loop {
            if hdr.seq_resp.load(Ordering::Relaxed) >= req {
                std::sync::atomic::fence(Ordering::Acquire);
                self.timeouts = 0;
                self.ever_answered = true;
                return true;
            }
            if hdr.quit.load(Ordering::Relaxed) != 0 {
                self.dead = true;
                return false;
            }
            if start.elapsed() >= budget {
                break;
            }
            std::thread::sleep(Duration::from_micros(200));
        }

        self.timeouts += 1;
        if self.timeouts >= 4 {
            self.dead = true;
            self.retry_after = Some(Instant::now() + Duration::from_secs(5));
            crate::log!(
                "[shm] no answer in {:?} x4 (helper {}); passing frames through, retrying in 5s",
                budget,
                if helper_present { "is present but silent" } else { "not running" }
            );
        }
        false
    }

    /// Whether a round trip started on this slot by [`Self::begin_async_request`] is
    /// still in flight (sent, not yet resolved by [`Self::poll_async_request`]).
    /// Callers use this to decide whether it's worth capturing and sending a new
    /// frame on this slot this present call -- each slot still only ever supports one
    /// outstanding request at a time (its own `seq_req`/`seq_resp` pair, not a
    /// queue); protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) is what makes there be two
    /// slots to ask this about instead of one.
    pub fn has_pending_request(&self, slot: usize) -> bool {
        self.pending[slot].is_some()
    }

    /// Starts a round trip on the given slot without waiting for it: bumps that
    /// slot's `seq_req` and records when, exactly like the first half of
    /// [`Self::round_trip_after_open`] (which only ever uses slot 0), but returns
    /// immediately instead of blocking. Pair with [`Self::poll_async_request`] on the
    /// same slot, called once per frame thereafter, to find out when (or whether) it
    /// resolves.
    ///
    /// Returns `false` (and starts nothing) on a dead connection whose retry timer
    /// hasn't elapsed, on `quit`, or if the mapping can't be opened -- the same
    /// conditions [`Self::try_round_trip`] fails open on. Only ever call this when
    /// [`Self::has_pending_request`] for this same slot is `false`; calling it with a
    /// request already in flight on that slot would silently abandon that one (its
    /// `seq_req` gets overwritten before `poll_async_request` ever sees a matching
    /// `seq_resp`) -- the two slots are otherwise completely independent, so a
    /// pending request on the *other* slot never blocks this call.
    pub fn begin_async_request(&mut self, slot: usize) -> bool {
        if !self.open() {
            self.dead = true;
            return false;
        }
        if self.dead && !self.should_retry() {
            return false;
        }
        self.dead = false;

        let hdr = self.header().expect("just opened above");
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.dead = true;
            return false;
        }

        let req = hdr.seq_req_slot(slot).load(Ordering::Relaxed) + 1;
        std::sync::atomic::fence(Ordering::Release);
        hdr.seq_req_slot(slot).store(req, Ordering::Relaxed);
        self.pending[slot] = Some((req, Instant::now()));
        true
    }

    /// Non-blocking: checks whether the request [`Self::begin_async_request`] started
    /// on this slot has answered yet. `Some(true)` once, the instant that slot's
    /// `seq_resp` catches up (clears its pending state, so
    /// [`Self::has_pending_request`] for this slot is `false` again afterward -- the
    /// caller is free to start a new one on it). `Some(false)` while still genuinely
    /// waiting, within budget. `None` once the budget is exceeded -- also clears the
    /// pending state (same timeout/dead-connection bookkeeping
    /// [`Self::round_trip_after_open`] already does), so the caller knows to give up
    /// on this slot's cycle and start fresh rather than keep polling a request that
    /// will never resolve. Returns `Some(false)` (never blocks, never panics) if
    /// called with nothing pending on this slot.
    pub fn poll_async_request(&mut self, slot: usize) -> Option<bool> {
        let Some((req, sent_at)) = self.pending[slot] else { return Some(false) };
        let Some(hdr) = self.header() else {
            self.pending[slot] = None;
            return None;
        };
        if hdr.seq_resp_slot(slot).load(Ordering::Relaxed) >= req {
            std::sync::atomic::fence(Ordering::Acquire);
            self.pending[slot] = None;
            self.timeouts = 0;
            self.ever_answered = true;
            return Some(true);
        }
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.pending[slot] = None;
            self.dead = true;
            return None;
        }
        let helper_present = hdr.helper_state.load(Ordering::Relaxed) != helper_state::STOPPED;
        let warming_up = !self.ever_answered;
        let budget = if !helper_present {
            Duration::from_millis(20)
        } else if warming_up {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(1)
        };
        if sent_at.elapsed() < budget {
            return Some(false);
        }
        self.pending[slot] = None;
        self.timeouts += 1;
        if self.timeouts >= 4 {
            self.dead = true;
            self.retry_after = Some(Instant::now() + Duration::from_secs(5));
            crate::log!(
                "[shm] no answer in {:?} x4 (helper {}); passing frames through, retrying in 5s",
                budget,
                if helper_present { "is present but silent" } else { "not running" }
            );
        }
        None
    }

    /// Whether a helper is actually running right now: its heartbeat (bumped every loop,
    /// thousands of times a second) has moved within the last 500 ms. `helper_state` is not
    /// enough -- a helper that is killed never writes STOPPED, so the header keeps saying
    /// RUNNING -- and waiting on a dead helper is a stall on every frame that asks.
    pub fn helper_alive(&mut self) -> bool {
        let Some(hdr) = self.header() else { return false };
        let hb = hdr.heartbeat.load(Ordering::Relaxed);
        let now = Instant::now();
        if hb != self.alive_heartbeat || self.alive_since.is_none() {
            let first = self.alive_since.is_none();
            self.alive_heartbeat = hb;
            self.alive_since = Some(now);
            if first {
                // Nothing to compare against yet: one sample cannot show movement.
                return false;
            }
            return true;
        }
        self.alive_since.is_some_and(|t| now.duration_since(t) < Duration::from_millis(500))
    }

    fn should_retry(&mut self) -> bool {
        let Some(hdr) = self.header() else { return false };
        let control_seq = hdr.control_seq.load(Ordering::Relaxed);
        let heartbeat = hdr.heartbeat.load(Ordering::Relaxed);
        let changed = control_seq != self.last_control_seq || heartbeat != self.last_heartbeat;
        self.last_control_seq = control_seq;
        self.last_heartbeat = heartbeat;
        if !changed {
            return false;
        }
        // A heartbeat alone is not a reason to try again immediately -- the helper
        // ticks it while it sits idle, so a helper that is up but not answering would
        // otherwise re-enable the moment it had just given up, costing another full
        // wait every time. The retry timer is what actually paces retries; a change
        // just means it's worth checking whether that timer has elapsed yet.
        self.retry_after.is_none_or(|t| Instant::now() >= t)
    }
}

/// Refuses to create or use the mapping's directory unless it is private: a directory,
/// owned by this uid, with no group/other permission bits. It lives under the
/// world-writable `/tmp`, so this is the difference between "our socket" and "whatever
/// another local user left in our way."
fn ensure_private_parent_dir(path: &str) -> bool {
    let Some(dir) = path.rfind('/').map(|i| &path[..i]) else {
        return true;
    };
    if dir.is_empty() {
        return true;
    }

    // mkdir -p, ignoring EEXIST at each level -- the same tolerant, idempotent
    // create-if-missing upstream's shell version does.
    let mut built = String::new();
    for part in dir.split('/') {
        if part.is_empty() {
            continue;
        }
        built.push('/');
        built.push_str(part);
        if let Ok(c) = CString::new(built.as_str()) {
            // SAFETY: `c` is a valid NUL-terminated C string for the call's duration.
            // The return value is intentionally ignored: EEXIST (already there) and any
            // other failure are both handled uniformly by the `lstat` check below.
            unsafe {
                libc::mkdir(c.as_ptr(), 0o700);
            }
        }
    }

    let Ok(c_dir) = CString::new(dir) else { return false };
    // SAFETY: `c_dir` is valid for the call's duration; `st` is a plain out-parameter.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::lstat(c_dir.as_ptr(), &mut st) } != 0 {
        return false;
    }
    let is_dir = (st.st_mode & libc::S_IFMT) == libc::S_IFDIR;
    // SAFETY: getuid() takes no arguments and cannot fail.
    let owned_by_us = st.st_uid == unsafe { libc::getuid() };
    let no_group_other_perms = (st.st_mode & (libc::S_IRWXG | libc::S_IRWXO)) == 0;
    is_dir && owned_by_us && no_group_other_perms
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_forge_protocol::ShmHeader;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicU64;

    /// A fresh, private scratch path per test -- never `$TMPDIR/neural-forge-*` or anything
    /// `open()`'s real env-var path would touch, so these can run in parallel with each
    /// other (and with a real layer, if one happened to be running) without colliding.
    fn scratch_path() -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        format!("{}/neural-forge-shm-test-{pid}-{n}/shm.bin", std::env::temp_dir().display())
    }

    fn header_of<'a>(client: &'a ShmClient) -> &'a ShmHeader {
        client.header().expect("open_at should have attached")
    }

    #[test]
    fn open_creates_a_valid_mapping() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        assert!(std::path::Path::new(&path).exists());
        assert!(header_of(&client).is_valid());
        // Idempotent: a second open on the same client is a no-op, not a re-create.
        assert!(client.open_at(&path));
    }

    #[test]
    fn round_trip_with_no_helper_times_out_then_marks_dead() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        // helper_state defaults to STOPPED, so each attempt's budget is the short
        // "nobody's listening" one -- four of them should complete quickly.
        for _ in 0..3 {
            assert!(!client.try_round_trip_at(&path));
            assert!(!client.dead, "should not give up before the fourth timeout");
        }
        assert!(!client.try_round_trip_at(&path));
        assert!(client.dead, "four consecutive timeouts should mark the connection dead");
        assert!(client.retry_after.is_some());

        // Dead, and the retry timer hasn't elapsed yet, and nothing on the other end
        // has changed control_seq/heartbeat -- so this call must not even attempt a
        // round trip.
        let resp_before = header_of(&client).seq_resp.load(Ordering::Relaxed);
        assert!(!client.try_round_trip_at(&path));
        assert_eq!(header_of(&client).seq_resp.load(Ordering::Relaxed), resp_before);
    }

    #[test]
    fn round_trip_succeeds_when_something_answers() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        let hdr_ptr = client.header as usize;

        // A minimal stand-in for the helper: echo every seq_req into seq_resp as soon
        // as it changes. Exactly the contract `try_round_trip` waits on -- nothing
        // upstream-specific about it.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = std::sync::Arc::clone(&stop);
        let echo = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends,
            // and never unmapped by `ShmClient` regardless).
            let hdr = unsafe { &*(hdr_ptr as *mut ShmHeader) };
            while !stop_clone.load(Ordering::Relaxed) {
                let req = hdr.seq_req.load(Ordering::Relaxed);
                if hdr.seq_resp.load(Ordering::Relaxed) != req {
                    hdr.seq_resp.store(req, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        });

        assert!(client.try_round_trip_at(&path));
        assert!(client.ever_answered);
        assert!(!client.dead);

        stop.store(true, Ordering::Relaxed);
        echo.join().unwrap();
    }

    #[test]
    fn async_request_resolves_without_blocking_when_something_answers() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        let hdr_ptr = client.header as usize;
        // A live helper, so `poll_async_request`'s budget is the long "steady state"
        // one rather than the short "nobody's listening" one -- doesn't matter here
        // since the echo thread answers almost immediately either way, but matches
        // what a real run looks like.
        header_of(&client).helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = std::sync::Arc::clone(&stop);
        let echo = std::thread::spawn(move || {
            // SAFETY: the mapping outlives this thread (joined before the test ends).
            let hdr = unsafe { &*(hdr_ptr as *mut ShmHeader) };
            while !stop_clone.load(Ordering::Relaxed) {
                let req = hdr.seq_req.load(Ordering::Relaxed);
                if req != 0 && hdr.seq_resp.load(Ordering::Relaxed) != req {
                    hdr.seq_resp.store(req, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        });

        assert!(client.begin_async_request(0), "should start a request against an already-open mapping");
        assert!(client.has_pending_request(0));
        // Real non-blocking behavior: a call arriving before the echo thread has had a
        // chance to run must not hang waiting -- it either sees `Some(false)` (not
        // answered yet) or, if the thread was fast enough, `Some(true)` -- either way
        // this call itself returns immediately.
        let mut resolved = client.poll_async_request(0) == Some(true);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !resolved {
            assert!(Instant::now() < deadline, "poll_async_request never resolved true");
            resolved = client.poll_async_request(0) == Some(true);
        }
        assert!(!client.has_pending_request(0), "a resolved request must clear pending state");
        assert!(client.ever_answered);
        assert!(!client.dead);

        stop.store(true, Ordering::Relaxed);
        echo.join().unwrap();
    }

    #[test]
    fn the_two_slots_are_fully_independent() {
        // The actual point of protocol v3 (docs/PROTOCOL_V3_DESIGN.md): slot 1 can have a
        // request outstanding while slot 0's is still pending, and each resolves on
        // its own seq_req/seq_resp pair without disturbing the other -- proven here
        // with a helper that only ever answers slot 0, confirming slot 1 staying
        // genuinely pending is not somehow a side effect of slot 0's own state.
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        header_of(&client).helper_state.store(neural_forge_protocol::enums::helper_state::RUNNING, Ordering::Relaxed);

        assert!(client.begin_async_request(0));
        assert!(client.begin_async_request(1));
        assert!(client.has_pending_request(0));
        assert!(client.has_pending_request(1));

        // Nothing has answered either slot yet.
        assert_eq!(client.poll_async_request(0), Some(false));
        assert_eq!(client.poll_async_request(1), Some(false));

        // Answer slot 0 only.
        let req0 = header_of(&client).seq_req.load(Ordering::Relaxed);
        header_of(&client).seq_resp.store(req0, Ordering::Relaxed);

        assert_eq!(client.poll_async_request(0), Some(true), "slot 0 should resolve once its own seq_resp catches up");
        assert!(!client.has_pending_request(0));

        // Slot 1 must still be genuinely pending -- answering slot 0 must not have
        // touched slot 1's own seq_req_b/seq_resp_b handshake at all.
        assert!(client.has_pending_request(1), "slot 1 must still be pending after only slot 0 was answered");
        assert_eq!(client.poll_async_request(1), Some(false));
        assert_eq!(header_of(&client).seq_resp_b.load(Ordering::Relaxed), 0, "nothing answered slot 1's own seq_resp_b");

        // Now answer slot 1 too.
        let req1 = header_of(&client).seq_req_b.load(Ordering::Relaxed);
        header_of(&client).seq_resp_b.store(req1, Ordering::Relaxed);
        assert_eq!(client.poll_async_request(1), Some(true));
        assert!(!client.has_pending_request(1));
    }

    #[test]
    fn async_request_with_no_helper_times_out_without_blocking_and_clears_pending() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        // helper_state defaults to STOPPED -- short "nobody's listening" budget (20ms),
        // so this test still runs fast despite exercising a real timeout.
        assert!(client.begin_async_request(0));
        assert!(client.has_pending_request(0));

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut result = client.poll_async_request(0);
        while result == Some(false) {
            assert!(Instant::now() < deadline, "poll_async_request never gave up");
            result = client.poll_async_request(0);
        }
        assert_eq!(result, None, "an unanswered request past budget must resolve to None, not Some(true)");
        assert!(!client.has_pending_request(0), "a timed-out request must clear pending state too");
    }

    #[test]
    fn private_parent_dir_is_created_when_missing() {
        let path = scratch_path();
        assert!(ensure_private_parent_dir(&path));
        let dir = &path[..path.rfind('/').unwrap()];
        let meta = std::fs::metadata(dir).unwrap();
        assert!(meta.is_dir());
        assert_eq!(meta.permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn world_writable_existing_dir_is_rejected() {
        let dir = format!("{}-{}", scratch_path().trim_end_matches("/shm.bin"), "world-writable");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(!ensure_private_parent_dir(&format!("{dir}/shm.bin")));
    }
}
