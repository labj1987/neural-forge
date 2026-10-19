//! The shared-memory round trip with the model server.
//!
//! This is the seam: everything here only ever touches `neural_forge_protocol::ShmHeader`'s
//! atomics. What answers on the other end of the mapping (the in-process model server,
//! `preupscale::native_post`) is none of this module's business.
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
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use neural_forge_protocol::{enums::server_state, shm_default_path, Slot, MAX_FRAME, SHM_MAGIC};

/// The subset of `ShmHeader`'s composition fields the composition (`capture::run`,
/// `composition::gpu`) needs, decoded once per frame from the raw atomics. See that field's own doc
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
    /// What fraction of the frame's resolution the model works at after the upscaler -- see
    /// [`neural_forge_protocol::ShmHeader::working_scale_bits`]. `capture::run` blits the
    /// swapchain capture down to it and `composition::gpu` blits the answer back up.
    pub working_scale: f32,
    /// What the model should treat as white, as the encode divides by it (see
    /// [`crate::composition::encode`]). The three header fields are one number here:
    /// the manual value times the scale, times the trim when the reading came from a
    /// meter rather than the slider. Clamped away from zero so the divide is safe.
    pub white_point: f32,
    /// Which curve the encode uses -- [`neural_forge_protocol::enums::reversible_mode`].
    pub reversible_mode: u32,
}

/// How many bytes of each pixel region this process maps: the protocol's full `MAX_FRAME`.
const REGION_CAP: usize = MAX_FRAME;

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
    fn proxy(slot: Slot) -> Self {
        match slot {
            Slot::Primary => Region::Proxy0,
            Slot::Secondary => Region::Proxy1,
        }
    }
    fn answer(slot: Slot) -> Self {
        match slot {
            Slot::Primary => Region::Answer0,
            Slot::Secondary => Region::Answer1,
        }
    }
    fn offset(self) -> usize {
        match self {
            Region::Proxy0 => neural_forge_protocol::proxy_offset_slot(Slot::Primary),
            Region::Answer0 => neural_forge_protocol::answer_offset_slot(Slot::Primary),
            Region::Proxy1 => neural_forge_protocol::proxy_offset_slot(Slot::Secondary),
            Region::Answer1 => neural_forge_protocol::answer_offset_slot(Slot::Secondary),
        }
    }
}

/// The mapping as the native backend's after-the-upscaler server reads it from its own thread: the header
/// and the two slots' regions, which stay mapped for the life of the process once opened.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct ShmView {
    header: *const neural_forge_protocol::ShmHeader,
    regions: [*mut u8; 4],
    leased: bool,
}

// SAFETY: the pointers stay valid for the process's life (never unmapped once open); the header is atomics,
// and each slot's regions are owned by whoever saw that slot's `seq_req`, as with the model server.
#[cfg(target_arch = "x86_64")]
unsafe impl Send for ShmView {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for ShmView {}

#[cfg(target_arch = "x86_64")]
impl ShmView {
    pub fn header(&self) -> &neural_forge_protocol::ShmHeader {
        // SAFETY: see the type's Send impl.
        unsafe { &*self.header }
    }

    /// The slot's proxy and answer regions.
    pub fn regions(&self, slot: Slot) -> (*mut u8, *mut u8) {
        (self.regions[Region::proxy(slot) as usize], self.regions[Region::answer(slot) as usize])
    }

    /// Each region's size.
    pub fn capacity(&self) -> usize {
        REGION_CAP
    }

    /// Whether this process may use the channel now: it holds the lease (`ownership::held`), or the
    /// view is not of the real channel.
    pub fn usable(&self) -> bool {
        !self.leased || crate::ownership::held()
    }
}

/// The header of the channel this process opened (the lease lets one game hold it), for the one
/// write that has no `ShmClient` at hand: `note_vk`'s device-lost latch ([`note_device_lost`]).
static CHANNEL: std::sync::atomic::AtomicPtr<neural_forge_protocol::ShmHeader> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

fn unix_now() -> u32 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_secs() as u32).max(1)
}

/// What a game's layer states on the channel when it opens it, once: `device_lost_at` describes this
/// session (cleared, or set when the device was already lost before the channel opened), and a device
/// that cannot run the network says so in `layer_reason`, which the Status tab shows (nothing else
/// writes that line on such a device: the native backend's status comes only from holds).
fn note_channel_open(hdr: &neural_forge_protocol::ShmHeader, lost: bool, cannot_run: Option<&str>) {
    CHANNEL.store(std::ptr::from_ref(hdr).cast_mut(), Ordering::Release);
    state_session(hdr, lost, cannot_run);
}

/// The lease generation (`ownership::generation`) the channel's session was last started for: a
/// second device of the same game opening the channel is not a new session (it must not drop the
/// capture the first one is running), regaining the lease after another process held it is.
static SESSION_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn start_session(hdr: &neural_forge_protocol::ShmHeader) {
    let generation = crate::ownership::generation();
    if SESSION_GENERATION.swap(generation, Ordering::AcqRel) != generation && !own_session_continues(hdr) {
        neural_forge_protocol::capture::reset_session(hdr);
    }
}

/// The lease came back to this process while its own capture still runs on the channel (nobody held it
/// in between to start another): the same session, not reset (that would report this process's capture
/// as ended by the game, then as written).
fn own_session_continues(hdr: &neural_forge_protocol::ShmHeader) -> bool {
    use neural_forge_protocol::capture;
    capture::result(hdr).is_some_and(|o| o.state == capture::state::RUNNING && crate::series::own_capture(o.id))
}

/// The channel's header for another thread of this process (a capture's writers publishing its
/// outcome). Valid for the life of the process: a mapping is never unmapped once made
/// ([`ShmClient::open_at`]). Readers still only publish what this process's capture produced.
#[derive(Clone)]
pub(crate) struct Channel {
    header: *const neural_forge_protocol::ShmHeader,
    leased: bool,
}

// SAFETY: the header is atomics and stays mapped for the life of the process.
unsafe impl Send for Channel {}
unsafe impl Sync for Channel {}

impl Channel {
    /// `None` once this process no longer holds the channel's lease (`ownership::held`).
    pub(crate) fn header(&self) -> Option<&neural_forge_protocol::ShmHeader> {
        // SAFETY: see the type's doc comment.
        (!self.leased || crate::ownership::held()).then(|| unsafe { &*self.header })
    }
}

/// One mapping of a channel file per path for the whole process, shared by every device's
/// [`ShmClient`]: each used to map the whole file (about 1 GiB of address space) again and keep it to
/// the end of the process.
struct Mapped {
    path: String,
    base: usize,
    fd: OwnedFd,
    /// Bytes of each slot's proxy and answer regions known to be backed by the file
    /// (`posix_fallocate`), so writing them can never fault (`ShmClient::reserve`).
    reserved: [u64; Slot::COUNT],
}

static MAPPINGS: std::sync::Mutex<Vec<Mapped>> = std::sync::Mutex::new(Vec::new());

fn state_session(hdr: &neural_forge_protocol::ShmHeader, lost: bool, cannot_run: Option<&str>) {
    hdr.device_lost_at.store(if lost { unix_now() } else { 0 }, Ordering::Relaxed);
    if let Some(line) = cannot_run {
        hdr.set_layer_reason(line);
    }
}

/// Stamps `device_lost_at` on the channel, if this process has it open. Called once, by the latch.
pub(crate) fn note_device_lost() {
    let hdr = CHANNEL.load(Ordering::Acquire);
    if !hdr.is_null() && crate::ownership::held() {
        // SAFETY: set from a mapping that stays mapped for the life of the process (`ShmClient::open`).
        stamp_device_lost(unsafe { &*hdr });
    }
}

fn stamp_device_lost(hdr: &neural_forge_protocol::ShmHeader) {
    let _ = hdr.device_lost_at.compare_exchange(0, unix_now(), Ordering::Relaxed, Ordering::Relaxed);
}

/// One process's connection to the mapping. Not `Clone` — there is exactly one of these
/// per device, guarded by a `Mutex` in [`crate::device::NeuralForgeDeviceInfo`].
pub struct ShmClient {
    header: *mut neural_forge_protocol::ShmHeader,
    /// Opened through [`Self::open`], the real channel: usable only while this process holds its lease.
    leased: bool,
    /// Per slot: bytes of the proxy and answer regions this client saw backed ([`Self::reserve`]), and
    /// whether the frame `set_frame_info` described last fits in them.
    reserved: [u64; Slot::COUNT],
    frame_backed: [bool; Slot::COUNT],
    /// Where each pixel region is mapped in this process (proxy 0, answer 0, proxy 1,
    /// answer 1), `REGION_CAP` bytes each, inside the one mapping of the whole file.
    regions: [*mut u8; 4],
    path: String,
    timeouts: u32,
    ever_answered: bool,
    retry_after: Option<Instant>,
    last_control_seq: u32,
    last_heartbeat: u32,
    /// The model server heartbeat as last sampled by [`Self::server_alive`], and when it last moved.
    alive_heartbeat: u32,
    alive_since: Option<Instant>,
    dead: bool,
    /// Set when mapping the file failed (address space refused): not retried on every
    /// present for the rest of the process, since nothing about the next attempt differs.
    map_refused: bool,
    frames: u64,
    /// Per-slot: the request number and send time of a round trip issued via
    /// [`Self::begin_async_request`] that [`Self::poll_async_request`] hasn't yet
    /// resolved (answered or timed out). `None` means that slot has no request in
    /// flight -- callers use this to decide whether it's time to capture and send a
    /// new frame on that slot. Protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) gives the wire
    /// two fully independent request/response slots instead of one, so this is an
    /// array of two, not a single value -- each slot still only ever has one
    /// outstanding request at a time.
    pending: [Option<(u32, Instant)>; Slot::COUNT],
    /// Per-slot: the request number of the last round trip started with
    /// [`Self::begin_async_request`], kept after it resolves so [`Self::answer_evaluated`] can
    /// compare it with the model server's `seq_eval`.
    last_req: [u32; Slot::COUNT],
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
            header: std::ptr::null_mut(),
            leased: false,
            reserved: [0; Slot::COUNT],
            frame_backed: [true; Slot::COUNT],
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
            map_refused: false,
            frames: 0,
            pending: [None, None],
            last_req: [0, 0],
        }
    }
}

impl ShmClient {
    /// Cross-module test access to the raw header pointer -- `capture::tests` needs
    /// to poke `server_state`/`seq_resp` directly to stand in for a fake model server, the
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

    /// Test-only: a client over a header in ordinary memory, with no pixel regions and no file.
    /// For tests of header-only logic (switches, requests, liveness), which need no real mapping.
    #[cfg(test)]
    pub(crate) fn test_over_header(header: &neural_forge_protocol::ShmHeader) -> Self {
        Self { header: std::ptr::from_ref(header).cast_mut(), reserved: [REGION_CAP as u64; Slot::COUNT], ..Self::default() }
    }

    fn header(&self) -> Option<&neural_forge_protocol::ShmHeader> {
        // SAFETY: non-null only after a successful `open()`, which mmaps
        // `neural_forge_protocol::shm_total_bytes()` at this address and never unmaps it for
        // the lifetime of the process. The real channel is this process's only while it holds the
        // lease: without it, nothing here reads or writes it.
        (!self.header.is_null() && (!self.leased || crate::ownership::held())).then(|| unsafe { &*self.header })
    }

    /// The channel for another thread of this process, once open.
    pub(crate) fn channel(&self) -> Option<Channel> {
        self.header().map(|hdr| Channel { header: hdr, leased: self.leased })
    }

    /// Whether it's worth paying for a real capture this frame at all. `false` once
    /// the model server has reported the model permanently unavailable --
    /// capturing and writing back a frame nobody will ever evaluate is pure overhead
    /// (a full image<->buffer round trip plus a `memcpy` of the whole frame, every
    /// single present call) for zero chance of a different outcome. Reads a single
    /// already-mapped atomic; never blocks and never opens the mapping itself, so it's
    /// always safe to check before deciding whether to call
    /// [`crate::capture::run`] at all.
    pub fn model_known_unavailable(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.server_state.load(Ordering::Relaxed) == server_state::MODEL_FAILED
    }

    /// Takes the waiting capture request when `accept` says the caller serves it
    /// (`neural_forge_protocol::capture::take`).
    pub fn take_capture(&self, accept: impl Fn(&neural_forge_protocol::capture::Request) -> bool) -> Option<neural_forge_protocol::capture::Request> {
        neural_forge_protocol::capture::take(self.header()?, accept)
    }

    /// Takes a waiting one-shot request (one frame): the pre-upscaler dump's, with
    /// `NEURAL_FORGE_PREUPSCALE=dump`.
    pub fn take_capture_request(&self) -> Option<neural_forge_protocol::capture::Request> {
        self.take_capture(|r| r.frames == 1)
    }

    /// Whether any capture is asked for: a one-shot or a series.
    pub fn capture_requested(&self) -> bool {
        self.header().is_some_and(|hdr| neural_forge_protocol::capture::pending(hdr).is_some())
    }

    /// Whether a one-shot is waiting (not taken).
    pub fn capture_request_pending(&self) -> bool {
        self.header().and_then(neural_forge_protocol::capture::pending).is_some_and(|r| r.frames == 1)
    }

    /// Applies the configured in-game toggle on a physical key press.  It changes
    /// the same shared atomic the GUI uses, so the model server and layer agree immediately.
    /// `toggle_key` 0 is no key: nothing is watched.
    pub fn poll_toggle_hotkey(&mut self, poller: &mut crate::hotkey::Poller) {
        let Some(hdr) = self.header() else { return };
        let key = hdr.toggle_key.load(Ordering::Relaxed);
        if key == 0 {
            return;
        }
        if poller.pressed(key) {
            let enabled = hdr.enabled.load(Ordering::Relaxed);
            hdr.enabled.store((enabled == 0) as u32, Ordering::Relaxed);
        }
    }

    /// The raster the model server says its last slot-0 answer was for (`None` before any).
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

    /// The settings the composition needs, read fresh every frame (each is a single atomic load)
    /// so a live GUI change takes effect on the very next present rather than needing a restart.
    /// `None` before the mapping is open.
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
    /// slot -- the model server (and, on the way back, this same layer reading the answer)
    /// needs `width`/`height`/`proxy_format` to know how many of the region's bytes
    /// are real for this frame, not the full `MAX_FRAME`-sized reservation. Call
    /// before [`Self::write_proxy`]/[`Self::begin_async_request`]
    /// so the model server never observes the
    /// `seq_req` bump before it can see what raster it describes.
    ///
    /// `layer_attached`/`layer_heartbeat`/`layer_width`/`layer_height`/`layer_format`/
    /// `layer_frames` are status telemetry, not part of the handshake -- deliberately
    /// not per-slot; they just reflect whichever slot most recently captured.
    /// The mapping for another thread of this process, once open.
    #[cfg(target_arch = "x86_64")]
    pub fn view(&self) -> Option<ShmView> {
        (!self.header.is_null() && self.regions.iter().all(|r| !r.is_null())).then_some(ShmView { header: self.header, regions: self.regions, leased: self.leased })
    }

    /// The Passes setting and each pass's settings (the Model tab's with its overrides).
    #[cfg(target_arch = "x86_64")]
    pub fn pass_tunings(&self) -> Option<neural_forge_protocol::PassTunings> {
        self.header().map(|hdr| hdr.pass_tunings())
    }

    /// The layer's status line for the GUI (the native backend's state, or why it is not running).
    #[cfg(target_arch = "x86_64")]
    pub fn set_layer_reason(&self, reason: &str) {
        if let Some(hdr) = self.header() {
            hdr.set_layer_reason(reason);
        }
    }

    /// The layer's "attached and presenting" telemetry, once per present while the game renders
    /// steadily, whichever path the model takes. It was only written by the after-the-upscaler
    /// capture (`set_frame_info`), so with the model before the upscaler -- which captures nothing at
    /// present -- the Status tab read "none attached" while the model ran. Does nothing until the
    /// channel is open (the capture and the hold open it).
    pub fn beat(&mut self) {
        self.frames += 1;
        let frames = self.frames;
        let Some(hdr) = self.header() else { return };
        hdr.layer_attached.store(1, Ordering::Relaxed);
        hdr.layer_heartbeat.fetch_add(1, Ordering::Relaxed);
        neural_forge_protocol::store64(&hdr.layer_frames_lo, &hdr.layer_frames_hi, frames);
        // Restated now and then rather than once: a model server restart re-initialises the header and
        // clears it. Rarely, because it is a seqlock-guarded string write.
        if frames % 120 == 1 {
            let name = crate::ownership::process_name();
            if hdr.game_name() != name {
                hdr.set_game_name(name);
            }
        }
    }

    pub fn set_frame_info(&mut self, slot: Slot, width: u32, height: u32, proxy_format: u32) {
        let bytes = u64::from(width) * u64::from(height) * neural_forge_protocol::enums::proxy_format::bytes_per_pixel(proxy_format) as u64;
        self.frame_backed[slot.index()] = self.reserve(slot, bytes);
        let Some(hdr) = self.header() else { return };
        hdr.width_slot(slot).store(width, Ordering::Relaxed);
        hdr.height_slot(slot).store(height, Ordering::Relaxed);
        hdr.proxy_format_slot(slot).store(proxy_format, Ordering::Relaxed);

        // The captured frame's size and format; liveness (`layer_attached`, the heartbeat, the
        // frame count and the game's name) is `beat`'s, every present.
        hdr.layer_width.store(width, Ordering::Relaxed);
        hdr.layer_height.store(height, Ordering::Relaxed);
        hdr.layer_format.store(proxy_format, Ordering::Relaxed);
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

    /// Publishes the GPU time of the last capture whose timestamps were read back (see
    /// `crate::gpu_timer`). The latest reading, not an average: one store per completed capture.
    pub fn publish_capture_gpu_ms(&self, ms: f32) {
        let Some(hdr) = self.header() else { return };
        hdr.layer_capture_gpu_ms_bits.store(ms.to_bits(), Ordering::Relaxed);
    }

    /// Publishes the GPU time of the last async compose whose timestamps were read back.
    pub fn publish_compose_gpu_ms(&self, ms: f32) {
        let Some(hdr) = self.header() else { return };
        hdr.layer_compose_gpu_ms_bits.store(ms.to_bits(), Ordering::Relaxed);
    }

    /// The pre-upscaler path's state and identified input extent (`ShmHeader::preupscale_*`).
    pub fn publish_preupscale_state(&self, state: u32, native: bool, width: u32, height: u32) {
        let Some(hdr) = self.header() else { return };
        hdr.preupscale_state.store(state, Ordering::Relaxed);
        hdr.native_running.store(u32::from(state == 2 && native), Ordering::Relaxed);
        hdr.preupscale_width.store(width, Ordering::Relaxed);
        hdr.preupscale_height.store(height, Ordering::Relaxed);
    }

    /// The last hold's CPU milliseconds and the running count of answers over budget.
    pub fn publish_preupscale_hold(&self, hold_ms: f32, misses: u32) {
        let Some(hdr) = self.header() else { return };
        hdr.preupscale_hold_ms_bits.store(hold_ms.to_bits(), Ordering::Relaxed);
        hdr.preupscale_misses.store(misses, Ordering::Relaxed);
    }

    /// `debug_view` (0 before the mapping is open).
    pub fn debug_view(&self) -> u32 {
        self.header().map_or(0, |h| h.debug_view.load(Ordering::Relaxed))
    }

    /// The edit cache's status (`ShmHeader::cache_*`): the hold counts, the last cached hold's
    /// rejected share (0..1, when one was read) and the cache passes' GPU milliseconds (when known).
    pub fn publish_cache(&self, refresh_holds: u32, cached_holds: u32, rejected: Option<f32>, gpu_ms: Option<f32>) {
        let Some(hdr) = self.header() else { return };
        hdr.cache_refresh_holds.store(refresh_holds, Ordering::Relaxed);
        hdr.cache_cached_holds.store(cached_holds, Ordering::Relaxed);
        if let Some(r) = rejected {
            hdr.cache_rejected.store((r.clamp(0.0, 1.0) * 10_000.0).round() as u32, Ordering::Relaxed);
        }
        if let Some(ms) = gpu_ms {
            hdr.cache_gpu_ms_bits.store(ms.to_bits(), Ordering::Relaxed);
        }
    }

    /// History resets for a scene cut before the upscaler since the layer opened the channel
    /// (`ShmHeader::scene_cuts`).
    pub fn publish_scene_cuts(&self, cuts: u32) {
        let Some(hdr) = self.header() else { return };
        hdr.scene_cuts.store(cuts, Ordering::Relaxed);
    }

    /// A pending K-hold cache dump request (`ShmHeader::preupscale_dump_request`), taken: the field is
    /// zeroed so the CLI and the status see it was. One atomic load when there is none.
    pub fn take_preupscale_dump_request(&self) -> Option<u32> {
        let hdr = self.header()?;
        if hdr.preupscale_dump_request.load(Ordering::Relaxed) == 0 {
            return None;
        }
        Some(hdr.preupscale_dump_request.swap(0, Ordering::Relaxed)).filter(|&k| k > 0)
    }

    /// The channel's path once it is open (empty before).
    pub fn channel_path(&self) -> &str {
        &self.path
    }

    /// Whether the model is switched on: the live `enabled` toggle (F11 / the GUI / `shmctl set
    /// enabled`) and `apply_model`. What the native backend checks: it has no model server to report the
    /// model unavailable. `false` before the mapping is open.
    pub fn model_enabled(&self) -> bool {
        let Some(hdr) = self.header() else { return false };
        hdr.neural_enabled() && hdr.apply_model.load(Ordering::Relaxed) != 0
    }

    /// The published `(capture, compose)` GPU milliseconds, for the `[sync]` log line. Zeros
    /// before the first reading (or before the mapping is open).
    pub fn published_gpu_ms(&self) -> (f32, f32) {
        let Some(hdr) = self.header() else { return (0.0, 0.0) };
        (
            f32::from_bits(hdr.layer_capture_gpu_ms_bits.load(Ordering::Relaxed)),
            f32::from_bits(hdr.layer_compose_gpu_ms_bits.load(Ordering::Relaxed)),
        )
    }

    /// Writes `bytes` (truncated to `MAX_FRAME`, same discipline as the free-text
    /// fields in `ShmHeader`) into the given slot's proxy region -- the frame the
    /// layer is about to hand the model. Call before bumping that slot's `seq_req`
    /// (via [`Self::begin_async_request`]): the model server only starts reading once it observes that bump, so there
    /// is no concurrent-write hazard to guard against the way the header's atomics do.
    ///
    /// # Safety
    /// Must only be called after a successful [`Self::open`].
    pub fn write_proxy(&self, slot: Slot, bytes: &[u8]) {
        let Some((dst, cap)) = self.region(Region::proxy(slot)) else { return };
        // Never past what the file is known to back: a write there faults the game (SIGBUS) when
        // the file system is full.
        if bytes.len() as u64 > self.reserved[slot.index()] {
            return;
        }
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
    /// Meaningful only after [`Self::poll_async_request`] has
    /// returned a real answer for the request this goes with -- reading it any
    /// earlier just observes whatever the model server last wrote (stale or all-zero),
    /// which is why this never blocks or checks sequence numbers itself; the caller
    /// already knows from the round trip's own return value whether there is a real
    /// answer to read.
    pub fn read_answer(&self, slot: Slot, out: &mut [u8]) -> usize {
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
        (!p.is_null() && (!self.leased || crate::ownership::held())).then_some((p, REGION_CAP))
    }

    /// Makes sure the file backs the first `bytes` of `slot`'s proxy and answer regions
    /// (`posix_fallocate`), so neither this process nor the model server can fault writing a frame
    /// of that size: the channel lives on `/tmp`, often a tmpfs, and a write to a page of a sparse
    /// file that a full tmpfs cannot back is a SIGBUS, which kills the game. Grows only, and costs a
    /// comparison once a size is backed. `false` (the frame is not sent) when the space is not
    /// there. A client over a header with no file behind it (tests) has nothing to back.
    fn reserve(&mut self, slot: Slot, bytes: u64) -> bool {
        let bytes = bytes.min(REGION_CAP as u64);
        if bytes <= self.reserved[slot.index()] {
            return true;
        }
        let mut mappings = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mapped) = mappings.iter_mut().find(|m| m.base == self.header as usize) else {
            self.reserved[slot.index()] = bytes;
            return true;
        };
        if mapped.reserved[slot.index()] < bytes {
            for region in [Region::proxy(slot), Region::answer(slot)] {
                // SAFETY: a plain call on the open channel file; the range is inside it.
                let err = unsafe { libc::posix_fallocate(mapped.fd.as_raw_fd(), region.offset() as libc::off_t, bytes as libc::off_t) };
                if err != 0 {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Ordering::Relaxed) {
                        crate::log!("[shm] cannot back {bytes} bytes of {} (error {err}, is /tmp full?): frames of this size are not sent", mapped.path);
                        crate::logging::flush();
                    }
                    return false;
                }
            }
            mapped.reserved[slot.index()] = bytes;
        }
        self.reserved[slot.index()] = mapped.reserved[slot.index()];
        true
    }

    /// The given slot's proxy region address and capacity within this process's
    /// mapping -- `None` before [`Self::open`]/[`Self::open_at`] has actually mapped
    /// anything. For [`crate::capture::DirectCapture`]'s `VK_EXT_external_memory_host`
    /// import: the *only* legitimate reason anything outside this module needs this
    /// address at all, since every other caller goes through [`Self::write_proxy`]
    /// instead.
    pub fn proxy_region(&self, slot: Slot) -> Option<(*mut u8, usize)> {
        // SAFETY: `pixel_base` plus `proxy_offset_slot(slot)` stays within the
        // `shm_total_bytes()` mapping `open_at` established, same reasoning as
        // `write_proxy`'s own pointer arithmetic.
        self.region(Region::proxy(slot))
    }

    /// The given slot's answer region address and capacity, mirroring
    /// [`Self::proxy_region`]. For `composition::gpu::GpuCompose`'s zero-copy compose, which
    /// imports it as device memory so the GPU reads the model server's answer where it landed instead
    /// of [`Self::read_answer`] copying it out first.
    pub fn answer_region(&self, slot: Slot) -> Option<(*mut u8, usize)> {
        self.region(Region::answer(slot))
    }

    /// What the model server says its last evaluation cost it (upload + evaluate + readback, in
    /// milliseconds), as published in the header before it answered. 0 before any answer.
    pub fn server_stage_ms(&self) -> f32 {
        let Some(hdr) = self.header() else { return 0.0 };
        let ms = |bits: &std::sync::atomic::AtomicU32| f32::from_bits(bits.load(Ordering::Relaxed));
        let total = ms(&hdr.server_upload_ms_bits) + ms(&hdr.server_eval_ms_bits) + ms(&hdr.server_readback_ms_bits);
        if total.is_finite() && total > 0.0 { total } else { 0.0 }
    }

    /// Opens (or creates) the mapping if not already attached, and takes (or keeps) this process's
    /// lease on it. Idempotent, and cheap once attached and leased.
    pub fn open(&mut self) -> bool {
        if !self.header.is_null() {
            // A client over a scratch mapping (tests) holds no lease.
            if !self.leased {
                return true;
            }
            if !crate::ownership::claim(&self.path) {
                return false;
            }
            // Leased again after another process held the channel: a new session on it.
            if let Some(hdr) = self.header() {
                if SESSION_GENERATION.load(Ordering::Acquire) != crate::ownership::generation() {
                    note_channel_open(hdr, crate::any_device_lost(), crate::native_unavailable().as_deref());
                    start_session(hdr);
                }
            }
            return true;
        }
        let path = neural_forge_protocol::env::var("NEURAL_FORGE_SHM")
            .filter(|s| !s.is_empty())
            .unwrap_or_else(shm_default_path);
        if !neural_forge_protocol::isolated_path(&path) || !ensure_private_parent_dir(&path) || !crate::ownership::claim(&path) { return false; }
        if !self.open_at(&path) {
            return false;
        }
        self.leased = true;
        // A fresh mapping holds the defaults: the saved settings (config.ini's `set_*`) go on it here,
        // since no other process (the GUI may not be running) does it before the game's first frame.
        if let Some(hdr) = self.header() {
            if neural_forge_protocol::persist::apply_saved(hdr) {
                crate::log!("[shm] applied the saved settings from {}", neural_forge_protocol::persist::config_file());
            }
            note_channel_open(hdr, crate::any_device_lost(), crate::native_unavailable().as_deref());
            start_session(hdr);
        }
        true
    }

    /// The actual implementation, taking the path explicitly so tests can point it at a
    /// scratch directory instead of `$NEURAL_FORGE_SHM`/the real `/tmp/neural-forge-$UID/` -- mutating
    /// process-wide environment variables from parallel `#[test]`s would race.
    fn open_at(&mut self, path: &str) -> bool {
        if !self.header.is_null() {
            return true;
        }
        if self.map_refused {
            return false;
        }
        if !ensure_private_parent_dir(path) {
            crate::log!("[shm] refusing {path}: parent directory is not private");
            return false;
        }
        let mut mappings = MAPPINGS.lock().unwrap_or_else(|e| e.into_inner());
        let (base, raw_fd) = match mappings.iter().find(|m| m.path == path) {
            // Another device of this process mapped it already: the same mapping.
            Some(mapped) => (mapped.base as *mut u8, mapped.fd.as_raw_fd()),
            None => match map_channel(path) {
                Ok((base, fd)) => {
                    let raw_fd = fd.as_raw_fd();
                    mappings.push(Mapped { path: path.to_owned(), base: base as usize, fd, reserved: [0; Slot::COUNT] });
                    (base, raw_fd)
                }
                Err(refused) => {
                    self.map_refused = refused;
                    return false;
                }
            },
        };
        drop(mappings);
        // SAFETY: every offset is inside the `shm_total_bytes()` mapping at `base`.
        self.regions = Region::ALL.map(|r| unsafe { base.add(r.offset()) });
        let header = base.cast::<neural_forge_protocol::ShmHeader>();
        // SAFETY: mapped for the life of the process; `HEADER_BYTES` is large enough for `ShmHeader`
        // (enforced at compile time in `neural_forge_protocol`).
        let hdr = unsafe { &*header };
        // SAFETY: the descriptor stays open in `MAPPINGS` for the life of the process.
        let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(raw_fd) };
        neural_forge_protocol::mapping::with_init_lock(fd, || {
            if hdr.magic.load(Ordering::Acquire) != SHM_MAGIC || !hdr.is_valid() {
                // No magic is a new file (a foreign one was refused); a version mismatch is
                // a stale build of ours. Both get the same answer: (re)initialize rather than
                // half-read a layout we don't agree on. Under the file's lock, so a front end
                // opening it at the same moment neither initialises it too nor reads it half done.
                hdr.init_defaults();
            }
        });
        self.last_heartbeat = hdr.server_heartbeat.load(Ordering::Relaxed);
        self.last_control_seq = hdr.control_seq.load(Ordering::Relaxed);
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

    /// Whether a round trip started on this slot by [`Self::begin_async_request`] is
    /// still in flight (sent, not yet resolved by [`Self::poll_async_request`]).
    /// Callers use this to decide whether it's worth capturing and sending a new
    /// frame on this slot this present call -- each slot still only ever supports one
    /// outstanding request at a time (its own `seq_req`/`seq_resp` pair, not a
    /// queue); protocol v3 (`docs/PROTOCOL_V3_DESIGN.md`) is what makes there be two
    /// slots to ask this about instead of one.
    pub fn has_pending_request(&self, slot: Slot) -> bool {
        self.pending[slot.index()].is_some()
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
    /// conditions the round trip fails open on. Only ever call this when
    /// [`Self::has_pending_request`] for this same slot is `false`; calling it with a
    /// request already in flight on that slot would silently abandon that one (its
    /// `seq_req` gets overwritten before `poll_async_request` ever sees a matching
    /// `seq_resp`) -- the two slots are otherwise completely independent, so a
    /// pending request on the *other* slot never blocks this call.
    pub fn begin_async_request(&mut self, slot: Slot) -> bool {
        if !self.open() {
            self.dead = true;
            return false;
        }
        if self.dead && !self.should_retry() {
            return false;
        }
        self.dead = false;

        if !self.frame_backed[slot.index()] {
            return false;
        }
        // Normally just opened above; gone if the lease was let go in between (the mapping released):
        // the round trip fails open, never a panic in the game.
        let Some(hdr) = self.header() else {
            self.dead = true;
            return false;
        };
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.dead = true;
            return false;
        }

        let req = neural_forge_protocol::next_request(hdr.seq_req_slot(slot).load(Ordering::Relaxed));
        std::sync::atomic::fence(Ordering::Release);
        hdr.seq_req_slot(slot).store(req, Ordering::Relaxed);
        self.pending[slot.index()] = Some((req, Instant::now()));
        self.last_req[slot.index()] = req;
        true
    }

    /// Non-blocking: checks whether the request [`Self::begin_async_request`] started
    /// on this slot has answered yet. `Some(true)` once, the instant that slot's
    /// `seq_resp` catches up (clears its pending state, so
    /// [`Self::has_pending_request`] for this slot is `false` again afterward -- the
    /// caller is free to start a new one on it). `Some(false)` while still genuinely
    /// waiting, within budget. `None` once the budget is exceeded -- also clears the
    /// pending state (same timeout/dead-connection bookkeeping
    /// bookkeeping), so the caller knows to give up
    /// on this slot's cycle and start fresh rather than keep polling a request that
    /// will never resolve. Returns `Some(false)` (never blocks, never panics) if
    /// called with nothing pending on this slot.
    pub fn poll_async_request(&mut self, slot: Slot) -> Option<bool> {
        let Some((req, sent_at)) = self.pending[slot.index()] else { return Some(false) };
        let Some(hdr) = self.header() else {
            self.pending[slot.index()] = None;
            return None;
        };
        // Equality, not "at least": the counter wraps, and after the wrap an old answer is a
        // larger number than the new request (`neural_forge_protocol::next_request`).
        if hdr.seq_resp_slot(slot).load(Ordering::Relaxed) == req {
            std::sync::atomic::fence(Ordering::Acquire);
            self.pending[slot.index()] = None;
            self.timeouts = 0;
            self.ever_answered = true;
            return Some(true);
        }
        if hdr.quit.load(Ordering::Relaxed) != 0 {
            self.pending[slot.index()] = None;
            self.dead = true;
            return None;
        }
        let server_present = hdr.server_state.load(Ordering::Relaxed) != server_state::STOPPED;
        let warming_up = !self.ever_answered;
        let budget = if !server_present {
            Duration::from_millis(20)
        } else if warming_up {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(1)
        };
        if sent_at.elapsed() < budget {
            return Some(false);
        }
        self.pending[slot.index()] = None;
        self.timeouts += 1;
        if self.timeouts >= 4 {
            self.dead = true;
            self.retry_after = Some(Instant::now() + Duration::from_secs(5));
            crate::log!(
                "[shm] no answer in {:?} x4 (model server {}); passing frames through, retrying in 5s",
                budget,
                if server_present { "is present but silent" } else { "not running" }
            );
        }
        None
    }

    /// Whether a model server is actually running right now: its heartbeat (bumped every loop,
    /// thousands of times a second) has moved within the last 500 ms. `server_state` is not
    /// enough -- a model server that is killed never writes STOPPED, so the header keeps saying
    /// RUNNING -- and waiting on a dead one is a stall on every frame that asks.
    pub fn server_alive(&mut self) -> bool {
        let Some(hdr) = self.header() else { return false };
        let hb = hdr.server_heartbeat.load(Ordering::Relaxed);
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
        let heartbeat = hdr.server_heartbeat.load(Ordering::Relaxed);
        let changed = control_seq != self.last_control_seq || heartbeat != self.last_heartbeat;
        self.last_control_seq = control_seq;
        self.last_heartbeat = heartbeat;
        if !changed {
            return false;
        }
        // A heartbeat alone is not a reason to try again immediately -- the model server
        // ticks it while it sits idle, so a model server that is up but not answering would
        // otherwise re-enable the moment it had just given up, costing another full
        // wait every time. The retry timer is what actually paces retries; a change
        // just means it's worth checking whether that timer has elapsed yet.
        self.retry_after.is_none_or(|t| Instant::now() >= t)
    }
}

/// Opens (creating if needed) and maps the whole channel file at `path` for the rest of the process's
/// life. `Err(true)` when it must not be retried (a foreign file, or the address space refused);
/// `Err(false)` for a failure the next present may not have.
fn map_channel(path: &str) -> Result<(*mut u8, OwnedFd), bool> {
    let c_path = CString::new(path).map_err(|_| true)?;
    // SAFETY: `c_path` is a valid, NUL-terminated C string for the duration of the
    // call. `O_NOFOLLOW` refuses to open through a symlink -- this file lives under
    // a world-writable /tmp, so that refusal matters.
    let raw_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_NOFOLLOW | libc::O_CLOEXEC, 0o600) };
    if raw_fd < 0 {
        crate::log!("[shm] open {path} failed");
        return Err(false);
    }
    // SAFETY: `raw_fd` was just returned by a successful `open()` and is owned nowhere else.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    neural_forge_protocol::mapping::with_init_lock(fd.as_fd(), || size_channel(path, &fd))?;
    // SAFETY: `fd` is open and sized to at least `total` bytes; mapping the whole file (header plus
    // both slots' regions) is in bounds. It is never unmapped, so everything derived from it stays
    // valid for the life of the process. A `MAP_SHARED` file mapping is sparse and page-cache backed:
    // it costs only the pages a session touches.
    let total = neural_forge_protocol::shm_total_bytes();
    let p = unsafe { libc::mmap(std::ptr::null_mut(), total, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
    if p == libc::MAP_FAILED {
        crate::log!("[shm] mmap {path} failed; not retrying");
        return Err(true);
    }
    Ok((p.cast::<u8>(), fd))
}

/// Classifies and sizes the channel file under its init lock (`mapping::with_init_lock`).
fn size_channel(path: &str, fd: &OwnedFd) -> Result<(), bool> {
    // Sizing the file and writing a header over its start is only for a file that is new or
    // already a mapping; a path naming anything else is left exactly as it is.
    match neural_forge_protocol::mapping::channel_file(fd.as_fd()) {
        Some(neural_forge_protocol::mapping::ChannelFile::Foreign) => {
            crate::log!("[shm] refusing {path}: it is an existing file that is not a Neural Forge mapping (left untouched)");
            return Err(true);
        }
        Some(_) => {}
        None => {
            crate::log!("[shm] cannot examine {path}");
            return Err(false);
        }
    }

    let total = neural_forge_protocol::shm_total_bytes();
    // SAFETY: `stat` is a plain out-parameter; zero-initializing it is always valid
    // and `fstat` either fully populates it or returns an error we check.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let needs_truncate = unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 || (st.st_size as usize) < total;
    // The file stays sparse (its frame regions are backed as frames use them, `ShmClient::reserve`),
    // but the header is written on every present: it is backed now, or the channel is not used.
    // SAFETY: plain calls on the file just opened.
    if needs_truncate && unsafe { libc::ftruncate(fd.as_raw_fd(), total as libc::off_t) } != 0 {
        crate::log!("[shm] ftruncate {path} failed");
        return Err(false);
    }
    let err = unsafe { libc::posix_fallocate(fd.as_raw_fd(), 0, neural_forge_protocol::HEADER_BYTES as libc::off_t) };
    if err != 0 {
        crate::log!("[shm] cannot back the header of {path} (error {err}, is /tmp full?)");
        return Err(false);
    }
    Ok(())
}

/// Refuses to create or use the mapping's directory unless it is private: a directory,
/// owned by this uid, with no group/other permission bits. It lives under the
/// world-writable `/tmp`, so this is the difference between "our socket" and "whatever
/// another local user left in our way."
fn ensure_private_parent_dir(path: &str) -> bool {
    neural_forge_protocol::private_dir::ensure_private_parent_dir(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lease back in the same process while its own capture runs: the session goes on (the capture
    /// is not reported as ended); another process's running capture, or none, is a new session.
    #[test]
    fn a_reacquired_lease_keeps_this_processs_running_capture() {
        use neural_forge_protocol::capture;
        let hdr = neural_forge_protocol::ShmHeader::default();
        hdr.init_defaults();
        assert!(!own_session_continues(&hdr), "nothing running");
        capture::publish(&hdr, &capture::Outcome { id: 4242, state: capture::state::RUNNING, total: 3, ..Default::default() });
        assert!(!own_session_continues(&hdr), "another process's capture");
        crate::series::set_own_capture(4242);
        assert!(own_session_continues(&hdr), "this process's own");
        capture::publish(&hdr, &capture::Outcome { id: 4242, state: capture::state::WRITTEN, total: 3, ..Default::default() });
        assert!(!own_session_continues(&hdr), "finished: a new session");
    }
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
        format!("{}/shm-{pid}-{n}/shm.bin", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().join("target/test-scratch").display())
    }

    fn header_of(client: &ShmClient) -> &ShmHeader {
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

    /// A custom channel path naming an unrelated file: not truncated, not written, and not
    /// retried on every present.
    #[test]
    fn an_unrelated_existing_file_is_not_truncated_or_initialised() {
        let path = scratch_path();
        let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        assert!(ensure_private_parent_dir(&path));
        let content = b"something of the user's, not a mapping".to_vec();
        std::fs::write(&path, &content).unwrap();
        let mut client = ShmClient::default();
        assert!(!client.open_at(&path));
        assert!(client.header().is_none() && client.map_refused);
        assert!(!client.open_at(&path));
        assert_eq!(std::fs::read(&path).unwrap(), content, "contents and length unchanged");
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn requests_with_no_server_time_out_then_mark_dead() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        // server_state defaults to STOPPED, so each request's budget is the short
        // "nobody's listening" one -- four of them complete quickly.
        let time_out = |client: &mut ShmClient| {
            assert!(client.begin_async_request(Slot::Primary));
            let deadline = Instant::now() + Duration::from_secs(5);
            while client.poll_async_request(Slot::Primary) == Some(false) {
                assert!(Instant::now() < deadline, "a request to nobody must time out");
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        for _ in 0..3 {
            time_out(&mut client);
            assert!(!client.dead, "should not give up before the fourth timeout");
        }
        time_out(&mut client);
        assert!(client.dead, "four consecutive timeouts should mark the connection dead");
        assert!(client.retry_after.is_some());
        // Dead, the retry timer not elapsed, and nothing on the other end changed control_seq or
        // the heartbeat: no request goes out.
        let req_before = header_of(&client).seq_req.load(Ordering::Relaxed);
        assert!(!client.begin_async_request(Slot::Primary));
        assert_eq!(header_of(&client).seq_req.load(Ordering::Relaxed), req_before);
    }

    /// Every device of a process shares one mapping of the channel file, and a frame whose regions
    /// the file cannot back is never sent.
    #[test]
    fn one_mapping_per_path_and_frames_are_backed_before_they_are_sent() {
        let path = scratch_path();
        let mut a = ShmClient::default();
        let mut b = ShmClient::default();
        assert!(a.open_at(&path) && b.open_at(&path));
        assert_eq!(a.header as usize, b.header as usize, "the second device reuses the first's mapping");
        assert_eq!(MAPPINGS.lock().unwrap().iter().filter(|m| m.path == path).count(), 1);
        a.set_frame_info(Slot::Primary, 64, 32, neural_forge_protocol::enums::proxy_format::RGBA8);
        assert!(a.frame_backed[0] && a.reserved[0] >= 64 * 32 * 4);
        // The file's allocated blocks now cover the proxy region's start (it was a hole before).
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let c = CString::new(path.clone()).unwrap();
        assert_eq!(unsafe { libc::stat(c.as_ptr(), &mut st) }, 0);
        assert!(st.st_blocks as u64 * 512 >= (neural_forge_protocol::HEADER_BYTES + 2 * 64 * 32 * 4) as u64, "{} blocks", st.st_blocks);
        // What another client backed is known to the next one without another call into the file.
        b.set_frame_info(Slot::Primary, 64, 32, neural_forge_protocol::enums::proxy_format::RGBA8);
        assert!(b.frame_backed[0]);
        // A write longer than what is backed is not made.
        let before = header_of(&a).seq_req.load(Ordering::Relaxed);
        a.write_proxy(Slot::Primary, &vec![7; (a.reserved[0] + 4) as usize]);
        // SAFETY: the proxy region is mapped and backed for its first bytes.
        assert_eq!(unsafe { *a.regions[0] }, 0, "nothing written past the backing");
        assert_eq!(header_of(&a).seq_req.load(Ordering::Relaxed), before);
    }

    #[test]
    fn async_request_resolves_without_blocking_when_something_answers() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        let hdr_ptr = client.header as usize;
        // A live model server, so `poll_async_request`'s budget is the long "steady state"
        // one rather than the short "nobody's listening" one -- doesn't matter here
        // since the echo thread answers almost immediately either way, but matches
        // what a real run looks like.
        header_of(&client).server_state.store(neural_forge_protocol::enums::server_state::RUNNING, Ordering::Relaxed);

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

        assert!(client.begin_async_request(Slot::Primary), "should start a request against an already-open mapping");
        assert!(client.has_pending_request(Slot::Primary));
        // Real non-blocking behavior: a call arriving before the echo thread has had a
        // chance to run must not hang waiting -- it either sees `Some(false)` (not
        // answered yet) or, if the thread was fast enough, `Some(true)` -- either way
        // this call itself returns immediately.
        let mut resolved = client.poll_async_request(Slot::Primary) == Some(true);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !resolved {
            assert!(Instant::now() < deadline, "poll_async_request never resolved true");
            resolved = client.poll_async_request(Slot::Primary) == Some(true);
        }
        assert!(!client.has_pending_request(Slot::Primary), "a resolved request must clear pending state");
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
        // with a model server that only ever answers slot 0, confirming slot 1 staying
        // genuinely pending is not somehow a side effect of slot 0's own state.
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        header_of(&client).server_state.store(neural_forge_protocol::enums::server_state::RUNNING, Ordering::Relaxed);

        assert!(client.begin_async_request(Slot::Primary));
        assert!(client.begin_async_request(Slot::Secondary));
        assert!(client.has_pending_request(Slot::Primary));
        assert!(client.has_pending_request(Slot::Secondary));

        // Nothing has answered either slot yet.
        assert_eq!(client.poll_async_request(Slot::Primary), Some(false));
        assert_eq!(client.poll_async_request(Slot::Secondary), Some(false));

        // Answer slot 0 only.
        let req0 = header_of(&client).seq_req.load(Ordering::Relaxed);
        header_of(&client).seq_resp.store(req0, Ordering::Relaxed);

        assert_eq!(client.poll_async_request(Slot::Primary), Some(true), "slot 0 should resolve once its own seq_resp catches up");
        assert!(!client.has_pending_request(Slot::Primary));

        // Slot 1 must still be genuinely pending -- answering slot 0 must not have
        // touched slot 1's own seq_req_b/seq_resp_b handshake at all.
        assert!(client.has_pending_request(Slot::Secondary), "slot 1 must still be pending after only slot 0 was answered");
        assert_eq!(client.poll_async_request(Slot::Secondary), Some(false));
        assert_eq!(header_of(&client).seq_resp_b.load(Ordering::Relaxed), 0, "nothing answered slot 1's own seq_resp_b");

        // Now answer slot 1 too.
        let req1 = header_of(&client).seq_req_b.load(Ordering::Relaxed);
        header_of(&client).seq_resp_b.store(req1, Ordering::Relaxed);
        assert_eq!(client.poll_async_request(Slot::Secondary), Some(true));
        assert!(!client.has_pending_request(Slot::Secondary));
    }

    #[test]
    fn async_request_with_no_server_times_out_without_blocking_and_clears_pending() {
        let path = scratch_path();
        let mut client = ShmClient::default();
        assert!(client.open_at(&path));
        // server_state defaults to STOPPED -- short "nobody's listening" budget (20ms),
        // so this test still runs fast despite exercising a real timeout.
        assert!(client.begin_async_request(Slot::Primary));
        assert!(client.has_pending_request(Slot::Primary));

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut result = client.poll_async_request(Slot::Primary);
        while result == Some(false) {
            assert!(Instant::now() < deadline, "poll_async_request never gave up");
            result = client.poll_async_request(Slot::Primary);
        }
        assert_eq!(result, None, "an unanswered request past budget must resolve to None, not Some(true)");
        assert!(!client.has_pending_request(Slot::Primary), "a timed-out request must clear pending state too");
    }

    /// The request numbers issued across the `u32` wrap, and the answers they are paired with.
    const ACROSS_THE_WRAP: [u32; 4] = [u32::MAX - 1, u32::MAX, 1, 2];

    #[test]
    fn asynchronous_requests_stay_paired_with_their_answers_across_the_wrap() {
        for slot in Slot::ALL {
            let header = Box::new(ShmHeader::default());
            header.init_defaults();
            header.server_state.store(server_state::RUNNING, Ordering::Relaxed);
            header.seq_req_slot(slot).store(u32::MAX - 2, Ordering::Relaxed);
            header.seq_resp_slot(slot).store(u32::MAX - 2, Ordering::Relaxed);
            let mut client = ShmClient::test_over_header(&header);
            for expected in ACROSS_THE_WRAP {
                assert!(client.begin_async_request(slot));
                // Not answered yet: the previous answer (larger than the request, once the
                // counter has wrapped) does not resolve it.
                assert_eq!(client.poll_async_request(slot), Some(false), "slot {slot}, request {expected:#x}");
                header.seq_resp_slot(slot).store(expected, Ordering::Relaxed);
                assert_eq!(client.poll_async_request(slot), Some(true), "slot {slot}, request {expected:#x}");
                assert!(!client.has_pending_request(slot));
            }
        }
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

    #[test]
    fn opening_the_channel_states_this_sessions_device_loss_and_a_device_that_cannot_run_the_network() {
        // `state_session`/`stamp_device_lost` are what `note_channel_open`/`note_device_lost` do to the
        // registered header; the registration itself is process-wide, so the test stays off it.
        let h = &neural_forge_protocol::ShmHeader::default();
        h.init_defaults();
        // A previous game's loss is cleared by the next game's open.
        h.device_lost_at.store(7, Ordering::Relaxed);
        state_session(h, false, None);
        assert_eq!(h.device_lost_at.load(Ordering::Relaxed), 0);
        assert_eq!(h.layer_reason(), "");
        // The latch fires after the open: stamped once, not moved by a second call.
        stamp_device_lost(h);
        let at = h.device_lost_at.load(Ordering::Relaxed);
        assert!(at > 1_700_000_000, "{at}");
        h.device_lost_at.store(5, Ordering::Relaxed);
        stamp_device_lost(h);
        assert_eq!(h.device_lost_at.load(Ordering::Relaxed), 5);
        // Lost before the open, on a device that cannot run the network.
        state_session(h, true, Some("device cannot run the network: VK_EXT_shader_float8"));
        assert!(h.device_lost_at.load(Ordering::Relaxed) > 1_700_000_000);
        assert_eq!(h.layer_reason(), "device cannot run the network: VK_EXT_shader_float8");
    }
}
