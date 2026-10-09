use std::sync::atomic::{AtomicU32, Ordering};

use crate::{Slot, HEADER_BYTES, NAME_BYTES, REASON_BYTES, SHM_MAGIC, SHM_VERSION};

/// The model's settings (the Model tab's), read out of the header as plain values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// -1 follows local structure; it is not a strength of zero.
    pub skin_structure: f32,
    pub style: u32,
    pub auto_mask: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            local_tone: 1.0,
            local_structure: 1.0,
            skin_structure: -1.0,
            style: 0,
            auto_mask: 1,
        }
    }
}

/// The shared-memory header. See the crate-level docs for the mapping this sits at the
/// front of.
///
/// `#[repr(C)]` and built entirely from `AtomicU32` fields (the free-text fields are
/// arrays of them, holding the UTF-8 bytes little-endian four to a word):
/// every field here has to land at the offset a same-order, same-width C struct would
/// put it at, so every process that maps it (the layer in each game, the GUI, the CLI) agrees
/// on the layout.
///
/// `Default` gives the all-zero value that a fresh, `ftruncate`d mapping already holds
/// before anyone touches it — useful for tests and for constructing one off the heap.
/// It is deliberately not "the real defaults": [`ShmHeader::init_defaults`] is the
/// explicit call (mirroring the original `ShmInitDefaults`) that sets the values this
/// protocol actually wants a fresh session to start at (`enabled = 1`,
/// `intensity = 1.0`, and so on).
#[repr(C)]
pub struct ShmHeader {
    pub magic: AtomicU32,
    pub version: AtomicU32,

    /// The frame handshake of the after-the-upscaler path. The layer bumps `seq_req` after
    /// writing a proxy; the model server (`native_post`, in the game's process) answers by storing
    /// the same number into `seq_resp` once the model's answer is in the output region.
    pub seq_req: AtomicU32,
    pub seq_resp: AtomicU32,
    pub width: AtomicU32,
    pub height: AtomicU32,
    /// Always 1 (RGBA byte order).
    pub format: AtomicU32,
    pub quit: AtomicU32,
    /// Bumped by the model server while it runs (the post path waits for it only then).
    pub server_heartbeat: AtomicU32,

    /// Bumped by whoever writes a setting. The layer watches this rather than re-reading every
    /// field every frame.
    pub control_seq: AtomicU32,
    /// Bumped when a saved setting changes (`persist::apply`); 0 means the saved settings were never
    /// applied to this mapping (`persist::apply_saved`).
    pub tuning_seq: AtomicU32,

    // --- the model ----------------------------------------------------------------
    pub enabled: AtomicU32,
    pub style: AtomicU32,
    pub auto_mask: AtomicU32,
    pub intensity_bits: AtomicU32,
    pub local_tone_bits: AtomicU32,
    pub local_structure_bits: AtomicU32,
    pub skin_structure_bits: AtomicU32,

    // --- the composition ------------------------------------------------------------
    /// How much of the model's edit reaches the frame, and how much of it is allowed to
    /// be colour rather than luminance. Separating the two is what keeps saturated
    /// highlights from shifting hue.
    pub transfer_strength_bits: AtomicU32,
    pub colour_strength_bits: AtomicU32,
    /// The most the pass may multiply or divide a pixel by. The transfer is a ratio,
    /// and a ratio against a near-black proxy pixel is unbounded without one.
    pub max_ratio_bits: AtomicU32,
    /// How a model that worked below the frame's size is brought back. 0 classic, 1
    /// matched residual, 2 native + edit.
    pub transfer: AtomicU32,
    /// 0 normal, 1 original/proxy, 2 the model's raw answer, 3 amplified diff, 4 colour trust
    /// engagement, 5 pre-colour-trust composite -- see `compose.comp`'s own `debug_view` push
    /// constant doc comment for the exact meaning of each (0-3 match
    /// `composition::apply::compose_pixel`'s CPU reference; 4/5 are GPU-only, from upstream).
    pub debug_view: AtomicU32,
    /// View 5 only: multiplies the pre-colour-trust colour before display, so a subtle
    /// difference from the normal (view 0) result is easier to see. 1.0 = no amplification.
    pub debug_scale_bits: AtomicU32,
    pub white_point_bits: AtomicU32,
    pub white_point_scale_bits: AtomicU32,
    pub white_point_source: AtomicU32,
    pub white_point_trim_bits: AtomicU32,
    /// What fraction of the frame's resolution the model works at, after the upscaler only
    /// (the swapchain capture is blitted down before the request and the answer back up).
    /// Capped at 1.0. Before the upscaler the model always works at DLSS's input size.
    pub working_scale_bits: AtomicU32,
    /// 0 off, 1 side by side, 2 a wipe.
    pub compare_mode: AtomicU32,
    pub compare_split_bits: AtomicU32,
    pub compare_zoom_bits: AtomicU32,
    pub compare_swap: AtomicU32,
    pub colour_mode: AtomicU32,
    /// Writes one set of matched before/after frames per session when the layer next presents.
    /// A value N above 1 instead captures the next N presented frames as a numbered series
    /// (the layer's `series` module); 1 is the one-shot dump, unchanged.
    pub capture_request: AtomicU32,
    /// A Linux key code the layer watches to toggle the pass, or 0 for none.
    pub toggle_key: AtomicU32,
    /// Which proxy the model is shown, and whether its answer is composed or
    /// substituted. See [`crate::enums::reversible_mode`].
    pub reversible_mode: AtomicU32,
    /// Whether the model's edit is applied at all. Off keeps the whole pass running —
    /// the capture, the round trip, the encode — and simply presents the clean frame,
    /// which is what makes an honest A/B possible.
    pub apply_model: AtomicU32,
    /// Freeze the frame the pass works on, so changing a setting re-runs the
    /// composition over the same picture instead of over whatever the game has drawn
    /// since.
    pub hold_frame: AtomicU32,

    // --- status, written by the model server (`native_post`) ----------------------------
    pub server_state: AtomicU32,
    pub model_up: AtomicU32,
    pub server_frames_lo: AtomicU32,
    pub server_frames_hi: AtomicU32,
    pub server_eval_ms_bits: AtomicU32,
    pub server_upload_ms_bits: AtomicU32,
    pub server_readback_ms_bits: AtomicU32,

    // --- status, written by the layer -------------------------------------------------
    pub layer_attached: AtomicU32,
    pub layer_frames_lo: AtomicU32,
    pub layer_frames_hi: AtomicU32,
    pub layer_width: AtomicU32,
    pub layer_height: AtomicU32,
    pub layer_format: AtomicU32,
    pub layer_composition_up: AtomicU32,
    pub layer_ms_bits: AtomicU32,
    pub layer_measured_white_bits: AtomicU32,
    pub layer_heartbeat: AtomicU32,

    // --- free text, each guarded by its own sequence number --------------------------
    // Bumped after the bytes are written, so a reader that sees an unchanged number is
    // looking at a whole string. See `store_seq_guarded`/`load_seq_guarded` below.
    pub layer_reason_seq: AtomicU32,
    layer_reason: [AtomicU32; REASON_BYTES / 4],
    pub game_name_seq: AtomicU32,
    game_name: [AtomicU32; NAME_BYTES / 4],

    /// How far the model server has answered *successfully*. `seq_resp` says a frame came
    /// back; this says it was worth using, so the layer can present the game's own
    /// frame when it was not.
    pub seq_ok: AtomicU32,

    /// 0: the composition blends the model's edit onto the frame under the strength and
    /// guard limits. 1: no composition at all — the model's raw answer IS the presented
    /// frame. Default 0: the composition is on; bypass is an explicit debug choice.
    pub composition_bypass: AtomicU32,
    /// The raster the model server actually answered, echoed before `seq_resp`. Without this
    /// echo a swapchain waiting on its own request could be satisfied by another
    /// swapchain's answer and copy the wrong number of bytes.
    pub answered_w: AtomicU32,
    pub answered_h: AtomicU32,

    // --- the HDR input path ----------------------------------------------------------
    pub hdr_mode: AtomicU32,
    pub hdr_detected: AtomicU32,
    pub hdr_active: AtomicU32,
    pub proxy_format: AtomicU32,
    /// What the proxy bytes in the shared region actually are for the request being
    /// made: the layer writes this immediately before `seq_req`, so the model server reads
    /// the width from the same statement that announced the pixels.
    pub hdr_encode: AtomicU32,

    // --- v3: the second, independent request/response slot ---------------------------
    // Appended after everything else on purpose: a new field inserted higher up would move
    // every field below it. See
    // `docs/PROTOCOL_V3_DESIGN.md` for why only these five are duplicated (not, say,
    // `format`/`hdr_encode`/`answered_w`/`answered_h`, which are dead fields on slot 0
    // too -- nothing in this workspace reads or writes them today).
    pub seq_req_b: AtomicU32,
    pub seq_resp_b: AtomicU32,
    pub width_b: AtomicU32,
    pub height_b: AtomicU32,
    pub proxy_format_b: AtomicU32,

    // --- v4: appended after everything else, for the same reason as the v3 slot above -----
    /// 0..1: how much the composition's ghost guard is applied. 0 is off (the relighting
    /// ratio is taken per pixel, so a stale answer can paste its detail onto content that
    /// has moved); 1 is the full guard. See `compose.comp`'s mode 2. Default 1.
    pub ghost_guard_bits: AtomicU32,

    // --- v5 ------------------------------------------------------------------------------
    /// How far the model may move a pixel's colour, relative to its luminance (upstream
    /// colour trust). 0 turns the model's colour off. Default 2.
    pub colour_trust_bits: AtomicU32,
    /// 0..1: how much of the relighting ratio comes from the neighbourhood rather than the pixel
    /// (upstream ratio smooth). Default 1.
    pub ratio_smooth_bits: AtomicU32,

    // --- v6 ------------------------------------------------------------------------------
    /// Run the model on every Nth presented frame (1 = every frame). With frame generation on,
    /// 2 skips the model on roughly the generated half and carries the last answer onto them
    /// (ghost guard applied), instead of making every generated frame wait for its own answer.
    pub model_interval: AtomicU32,

    // --- v8 ------------------------------------------------------------------------------
    /// GPU milliseconds of the layer's capture copy on the last capture whose timestamps were
    /// read back (f32 bits). Measured with a timestamp pair in the capture slot's own command
    /// buffer and read when that slot's fence is next found signalled, so it lags a capture
    /// or two and never stalls. 0 until the first reading.
    pub layer_capture_gpu_ms_bits: AtomicU32,
    /// GPU milliseconds of the layer's compose dispatch, measured the same way per async
    /// compose slot (f32 bits). 0 until the first reading.
    pub layer_compose_gpu_ms_bits: AtomicU32,

    // --- v9 ------------------------------------------------------------------------------
    /// The layer's pre-upscaler path (`NEURAL_FORGE_PREUPSCALE`, docs/PRE_UPSCALER_DESIGN.md):
    /// 0 off (`NEURAL_FORGE_PREUPSCALE=off`, or no device with NVX: nothing is held, the model runs
    /// after the upscaler), 1 waiting for the game's DLSS input (the mode is on, as it is by default,
    /// but no launch-bearing submit was held in the last 500 ms: no DLSS, DLAA, input not
    /// identified, or the toggle is off), 2 holding the DLSS submit.
    pub preupscale_state: AtomicU32,
    /// The identified DLSS colour input's extent (the render resolution), 0 before one is found.
    pub preupscale_width: AtomicU32,
    pub preupscale_height: AtomicU32,
    /// CPU milliseconds the last hold blocked the game's submit (f32 bits).
    pub preupscale_hold_ms_bits: AtomicU32,
    /// Held frames that went to DLSS untouched (the network not ready, a failed submit, ...).
    /// Counts up for the life of the layer's session.
    pub preupscale_misses: AtomicU32,

    // --- v10 -----------------------------------------------------------------------------
    /// Microseconds the model server spent on its last slot-0 request, from seeing `seq_req` move to
    /// just before it stored `seq_resp`. Written before `seq_resp`. 0 before the first answer.
    pub server_busy_us: AtomicU32,

    // --- v11 -----------------------------------------------------------------------------
    /// The last slot-0 request the model server actually ran the model on (`seq_req`'s value), written
    /// before `seq_resp`. A layer that sees `seq_resp` reach its request and this field equal to it
    /// got a model answer; anything else is an echo of its own frame (no feature built, an
    /// evaluate that failed, a refused frame). 0 before the first evaluated request.
    pub seq_eval: AtomicU32,

    // --- v13 -----------------------------------------------------------------------------
    /// 1 while the layer runs the model before the upscaler, 0 otherwise (the GUI shows the
    /// after-the-upscaler settings only then).
    pub native_running: AtomicU32,
}

// Every field is an atomic, so `ShmHeader` is `Sync` without an `unsafe impl`: another
// process writing the mapping at any time is exactly what atomics are for.

impl Default for ShmHeader {
    fn default() -> Self {
        // SAFETY: every field is an `AtomicU32` (valid for any `u32` bit pattern,
        // including all-zero) or an array of them — so
        // the all-zero bit pattern `zeroed()` produces is a valid value of every field,
        // and therefore of the whole struct. This is exactly the value a fresh,
        // `ftruncate`d (zero-filled) mapping already holds before anyone touches it.
        unsafe { std::mem::zeroed() }
    }
}

const _: () = assert!(std::mem::size_of::<ShmHeader>() <= HEADER_BYTES, "ShmHeader outgrew its region");

// The layout, pinned. Every process that maps this file agrees on where each field is
// only because they were compiled from the same header. A field inserted anywhere but
// the end silently moves everything after it, and a build that has not caught up then
// reads its neighbor's value — which is not a crash, it is a status display quietly
// reporting a nonsensical number for a flag that is 0 or 1. If any of these fire, the
// layout changed: bump `SHM_VERSION` in the same commit, then update these numbers.
const _: () = assert!(std::mem::size_of::<ShmHeader>() == 664, "the header layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, enabled) == 44, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, transfer_strength_bits) == 72, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, hdr_mode) == 568, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, seq_req_b) == 588, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, ghost_guard_bits) == 608, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, ratio_smooth_bits) == 616, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, model_interval) == 620, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, layer_capture_gpu_ms_bits) == 624, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, layer_compose_gpu_ms_bits) == 628, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, preupscale_state) == 632, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, preupscale_misses) == 648, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, seq_eval) == 656, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, native_running) == 660, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, layer_reason) == 228, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, game_name) == 424, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, server_state) == 156, "layout changed -- bump SHM_VERSION");
const _: () = assert!(std::mem::offset_of!(ShmHeader, server_busy_us) == 652, "layout changed -- bump SHM_VERSION");
// The free-text fields are whole words; their byte offsets are the ones they had as byte
// arrays (every field before them is a word, so none gained padding).
const _: () = assert!(REASON_BYTES.is_multiple_of(4) && NAME_BYTES.is_multiple_of(4));

/// The allowed range of every persisted setting ([`ShmHeader::persisted_settings`]), in the
/// setting's own units (an enum or flag takes whole numbers). The one place a setting's range is
/// decided: [`ShmHeader::apply_persisted_setting`] clamps to it, `shmctl set` refuses values
/// outside it and the GUI's rows take their bounds from it.
pub const SETTING_BOUNDS: [(&str, f32, f32); 33] = [
    ("white_point", 0.01, 10000.0),
    ("white_point_scale", 0.01, 100.0),
    ("white_point_trim", 0.01, 100.0),
    ("white_point_source", 0.0, 1.0),
    // evdev key codes (KEY_MAX); 0 is unbound.
    ("toggle_key", 0.0, 767.0),
    ("enabled", 0.0, 1.0),
    ("style", 0.0, 2.0),
    ("intensity", 0.0, 4.0),
    ("local_tone", 0.0, 4.0),
    ("local_structure", 0.0, 4.0),
    // -1 follows local structure.
    ("skin_structure", -1.0, 4.0),
    ("auto_mask", 0.0, 1.0),
    ("composition_bypass", 0.0, 1.0),
    ("transfer_strength", 0.0, 4.0),
    ("colour_strength", 0.0, 1.0),
    ("max_ratio", 1.0, 30.0),
    ("working_scale", 0.25, 1.0),
    ("reversible_mode", 0.0, (crate::enums::reversible_mode::COUNT - 1) as f32),
    ("hdr_mode", 0.0, 2.0),
    ("transfer", 0.0, 2.0),
    ("compare_mode", 0.0, 2.0),
    ("compare_split", 0.0, 1.0),
    ("compare_zoom", 1.0, 2.0),
    ("compare_swap", 0.0, 1.0),
    ("colour_mode", 0.0, 2.0),
    ("hold_frame", 0.0, 1.0),
    ("ghost_guard", 0.0, 1.0),
    ("colour_trust", 0.0, 4.0),
    ("ratio_smooth", 0.0, 1.0),
    ("model_interval", 1.0, 4.0),
    ("apply_model", 0.0, 1.0),
    ("debug_view", 0.0, 5.0),
    ("debug_scale", 0.1, 10.0),
];

/// `(min, max)` for a persisted setting, from [`SETTING_BOUNDS`].
pub fn setting_bounds(name: &str) -> Option<(f32, f32)> {
    SETTING_BOUNDS.iter().find(|(n, ..)| *n == name).map(|&(_, min, max)| (min, max))
}

impl ShmHeader {
    /// Resets every field to the defaults a freshly created mapping should hold. Takes
    /// `&self` rather than `&mut self`: this is called on a live mapping another
    /// process may be reading, same as everything else here, so it goes through the
    /// atomics rather than a bulk memory write.
    pub fn init_defaults(&self) {
        self.magic.store(SHM_MAGIC, Ordering::Relaxed);
        self.version.store(SHM_VERSION, Ordering::Relaxed);
        self.seq_req.store(0, Ordering::Relaxed);
        self.seq_resp.store(0, Ordering::Relaxed);
        self.width.store(0, Ordering::Relaxed);
        self.height.store(0, Ordering::Relaxed);
        self.format.store(1, Ordering::Relaxed);
        self.quit.store(0, Ordering::Relaxed);
        self.server_heartbeat.store(0, Ordering::Relaxed);
        self.control_seq.store(0, Ordering::Relaxed);
        self.tuning_seq.store(0, Ordering::Relaxed);

        self.enabled.store(1, Ordering::Relaxed);
        self.style.store(0, Ordering::Relaxed);
        self.auto_mask.store(1, Ordering::Relaxed);
        self.intensity_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_tone_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.local_structure_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.skin_structure_bits.store((-1.0f32).to_bits(), Ordering::Relaxed);

        self.transfer_strength_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.colour_strength_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.max_ratio_bits.store(2.0f32.to_bits(), Ordering::Relaxed);
        self.transfer.store(1, Ordering::Relaxed);
        self.debug_view.store(0, Ordering::Relaxed);
        self.debug_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.white_point_source.store(crate::enums::white_point_source::MANUAL, Ordering::Relaxed);
        self.white_point_trim_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.working_scale_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.compare_mode.store(0, Ordering::Relaxed);
        self.compare_split_bits.store(0.5f32.to_bits(), Ordering::Relaxed);
        self.compare_zoom_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.compare_swap.store(0, Ordering::Relaxed);
        self.colour_mode.store(crate::enums::colour_mode::AUTO, Ordering::Relaxed);
        self.capture_request.store(0, Ordering::Relaxed);
        // F11 is Linux evdev code 87 and is available in-game on XWayland.
        self.toggle_key.store(87, Ordering::Relaxed);
        self.reversible_mode.store(crate::enums::reversible_mode::KNEE, Ordering::Relaxed);
        self.apply_model.store(1, Ordering::Relaxed);
        self.hold_frame.store(0, Ordering::Relaxed);
        self.ghost_guard_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.colour_trust_bits.store(2.0f32.to_bits(), Ordering::Relaxed);
        self.ratio_smooth_bits.store(1.0f32.to_bits(), Ordering::Relaxed);
        self.model_interval.store(1, Ordering::Relaxed);

        self.server_state.store(crate::enums::server_state::STOPPED, Ordering::Relaxed);
        self.model_up.store(0, Ordering::Relaxed);
        self.server_frames_lo.store(0, Ordering::Relaxed);
        self.server_frames_hi.store(0, Ordering::Relaxed);
        self.server_eval_ms_bits.store(0, Ordering::Relaxed);
        self.server_upload_ms_bits.store(0, Ordering::Relaxed);
        self.server_readback_ms_bits.store(0, Ordering::Relaxed);

        self.layer_attached.store(0, Ordering::Relaxed);
        self.layer_frames_lo.store(0, Ordering::Relaxed);
        self.layer_frames_hi.store(0, Ordering::Relaxed);
        self.layer_width.store(0, Ordering::Relaxed);
        self.layer_height.store(0, Ordering::Relaxed);
        self.layer_format.store(0, Ordering::Relaxed);
        self.layer_composition_up.store(0, Ordering::Relaxed);
        self.layer_ms_bits.store(0, Ordering::Relaxed);
        self.layer_capture_gpu_ms_bits.store(0, Ordering::Relaxed);
        self.layer_compose_gpu_ms_bits.store(0, Ordering::Relaxed);
        self.preupscale_state.store(0, Ordering::Relaxed);
        self.preupscale_width.store(0, Ordering::Relaxed);
        self.preupscale_height.store(0, Ordering::Relaxed);
        self.preupscale_hold_ms_bits.store(0, Ordering::Relaxed);
        self.preupscale_misses.store(0, Ordering::Relaxed);
        self.server_busy_us.store(0, Ordering::Relaxed);
        self.seq_eval.store(0, Ordering::Relaxed);
        self.native_running.store(0, Ordering::Relaxed);
        self.layer_measured_white_bits.store(0, Ordering::Relaxed);
        self.layer_heartbeat.store(0, Ordering::Relaxed);

        self.set_layer_reason("");
        self.set_game_name("");

        self.seq_ok.store(0, Ordering::Relaxed);
        // Neural rendering should affect the presented image by default. A
        // bypassed composition is an explicit debug choice, not the normal mode.
        self.composition_bypass.store(0, Ordering::Relaxed);
        self.answered_w.store(0, Ordering::Relaxed);
        self.answered_h.store(0, Ordering::Relaxed);

        self.hdr_mode.store(crate::enums::hdr_mode::AUTO, Ordering::Relaxed);
        self.hdr_detected.store(crate::enums::hdr_kind::NONE, Ordering::Relaxed);
        self.hdr_active.store(0, Ordering::Relaxed);
        self.proxy_format.store(crate::enums::proxy_format::RGBA8, Ordering::Relaxed);
        self.hdr_encode.store(0, Ordering::Relaxed);

        self.seq_req_b.store(0, Ordering::Relaxed);
        self.seq_resp_b.store(0, Ordering::Relaxed);
        self.width_b.store(0, Ordering::Relaxed);
        self.height_b.store(0, Ordering::Relaxed);
        self.proxy_format_b.store(crate::enums::proxy_format::RGBA8, Ordering::Relaxed);
    }

    /// Resets every user-tunable setting -- [`Self::persisted_settings`]'s own list -- to its
    /// default value, on a live mapping the layer may be actively using. Deliberately narrower than
    /// [`Self::init_defaults`]: upstream shipped a real bug here (PR #16), where its own
    /// "reset settings" wiped the live session out from under a running process --
    /// seq words, status and counters, HDR detection, the free-text reason/name fields -- not just the tuning knobs a user
    /// actually meant to reset. This never touches the ownership lease either; that lives
    /// in a separate file (`shm.bin.owner`), entirely outside this struct.
    ///
    /// The default values come from running `init_defaults` on a scratch header, so there
    /// is no second list of defaults to drift from the first, and only the user-tunable
    /// fields are then stored onto `self`. Nothing else in the live header is written at
    /// all: an earlier version snapshotted the live fields, reinitialized everything and
    /// restored the snapshot, which reverted any live write landing in between
    /// and made this a second writer on the single-writer reason strings. Bumps
    /// `tuning_seq` and `control_seq` so the layer picks the change up and a
    /// later `apply_saved_settings` does not read the header as never configured.
    pub fn reset_persisted_settings(&self) {
        let defaults = ShmHeader::default();
        defaults.init_defaults();
        for (name, _, bits) in defaults.persisted_settings() {
            self.apply_persisted_setting(name, bits);
        }
        self.tuning_seq.fetch_add(1, Ordering::Relaxed);
        self.control_seq.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether this mapping is one of ours and laid out the way this build expects.
    pub fn is_valid(&self) -> bool {
        self.magic.load(Ordering::Relaxed) == SHM_MAGIC && self.version.load(Ordering::Relaxed) == SHM_VERSION
    }

    /// v3's two request/response slots (`docs/PROTOCOL_V3_DESIGN.md`) share every field
    /// name and type; these are the one place that picks slot 0's or slot 1's field, so no caller
    /// hand-writes its own choice per field. A [`Slot`] is one of exactly two, so there is no third case.
    pub fn seq_req_slot(&self, slot: Slot) -> &AtomicU32 {
        match slot {
            Slot::Primary => &self.seq_req,
            Slot::Secondary => &self.seq_req_b,
        }
    }
    pub fn seq_resp_slot(&self, slot: Slot) -> &AtomicU32 {
        match slot {
            Slot::Primary => &self.seq_resp,
            Slot::Secondary => &self.seq_resp_b,
        }
    }
    pub fn width_slot(&self, slot: Slot) -> &AtomicU32 {
        match slot {
            Slot::Primary => &self.width,
            Slot::Secondary => &self.width_b,
        }
    }
    pub fn height_slot(&self, slot: Slot) -> &AtomicU32 {
        match slot {
            Slot::Primary => &self.height,
            Slot::Secondary => &self.height_b,
        }
    }
    pub fn proxy_format_slot(&self, slot: Slot) -> &AtomicU32 {
        match slot {
            Slot::Primary => &self.proxy_format,
            Slot::Secondary => &self.proxy_format_b,
        }
    }

    /// Every setting a user can change from the GUI, as `("name", current bits)`
    /// pairs -- what [`crate::persist::snapshot`]/[`crate::persist::apply`] round-trip
    /// through `config.ini` so tuning survives a reboot (the SHM mapping itself lives
    /// under `/tmp` and does not). Add here, not just to the GUI, whenever a new
    /// tunable needs to survive a restart -- this is the one list that decides it.
    pub fn persisted_settings(&self) -> [(&'static str, bool, u32); 33] {
        [
            ("white_point", true, self.white_point_bits.load(Ordering::Relaxed)),
            ("white_point_scale", true, self.white_point_scale_bits.load(Ordering::Relaxed)),
            ("white_point_trim", true, self.white_point_trim_bits.load(Ordering::Relaxed)),
            ("white_point_source", false, self.white_point_source.load(Ordering::Relaxed)),
            ("toggle_key", false, self.toggle_key.load(Ordering::Relaxed)),
            ("enabled", false, self.enabled.load(Ordering::Relaxed)),
            ("style", false, self.style.load(Ordering::Relaxed)),
            ("intensity", true, self.intensity_bits.load(Ordering::Relaxed)),
            ("local_tone", true, self.local_tone_bits.load(Ordering::Relaxed)),
            ("local_structure", true, self.local_structure_bits.load(Ordering::Relaxed)),
            ("skin_structure", true, self.skin_structure_bits.load(Ordering::Relaxed)),
            ("auto_mask", false, self.auto_mask.load(Ordering::Relaxed)),
            ("composition_bypass", false, self.composition_bypass.load(Ordering::Relaxed)),
            ("transfer_strength", true, self.transfer_strength_bits.load(Ordering::Relaxed)),
            ("colour_strength", true, self.colour_strength_bits.load(Ordering::Relaxed)),
            ("max_ratio", true, self.max_ratio_bits.load(Ordering::Relaxed)),
            ("working_scale", true, self.working_scale_bits.load(Ordering::Relaxed)),
            ("reversible_mode", false, self.reversible_mode.load(Ordering::Relaxed)),
            ("hdr_mode", false, self.hdr_mode.load(Ordering::Relaxed)),
            // Added 2026-09-10 alongside the GUI rows that expose them -- see this
            // function's own doc comment on why both have to change together.
            ("transfer", false, self.transfer.load(Ordering::Relaxed)),
            ("compare_mode", false, self.compare_mode.load(Ordering::Relaxed)),
            ("compare_split", true, self.compare_split_bits.load(Ordering::Relaxed)),
            ("compare_zoom", true, self.compare_zoom_bits.load(Ordering::Relaxed)),
            ("compare_swap", false, self.compare_swap.load(Ordering::Relaxed)),
            ("colour_mode", false, self.colour_mode.load(Ordering::Relaxed)),
            ("hold_frame", false, self.hold_frame.load(Ordering::Relaxed)),
            ("ghost_guard", true, self.ghost_guard_bits.load(Ordering::Relaxed)),
            ("colour_trust", true, self.colour_trust_bits.load(Ordering::Relaxed)),
            ("ratio_smooth", true, self.ratio_smooth_bits.load(Ordering::Relaxed)),
            ("model_interval", false, self.model_interval.load(Ordering::Relaxed)),
            ("apply_model", false, self.apply_model.load(Ordering::Relaxed)),
            ("debug_view", false, self.debug_view.load(Ordering::Relaxed)),
            ("debug_scale", true, self.debug_scale_bits.load(Ordering::Relaxed)),
        ]
    }

    /// Stores one persisted setting back by name (as looked up in a `config.ini`
    /// `set_<name>=<value>` line) -- `bits` is already the right representation
    /// (`f32::to_bits()` for the float-valued ones, per [`Self::persisted_settings`]'s
    /// second field). A non-finite float is rejected and anything else is clamped to the
    /// setting's range in [`SETTING_BOUNDS`]. Returns whether a value was stored.
    pub fn apply_persisted_setting(&self, name: &str, bits: u32) -> bool {
        let Some((min, max)) = setting_bounds(name) else { return false };
        let Some(is_float) = self.persisted_settings().iter().find(|(n, ..)| *n == name).map(|&(_, f, _)| f) else {
            return false;
        };
        let bits = if is_float {
            let v = f32::from_bits(bits);
            if !v.is_finite() {
                return false;
            }
            v.clamp(min, max).to_bits()
        } else {
            bits.clamp(min as u32, max as u32)
        };
        let field = match name {
            "white_point" => &self.white_point_bits,
            "white_point_scale" => &self.white_point_scale_bits,
            "white_point_trim" => &self.white_point_trim_bits,
            "white_point_source" => &self.white_point_source,
            "toggle_key" => &self.toggle_key,
            "enabled" => &self.enabled,
            "style" => &self.style,
            "intensity" => &self.intensity_bits,
            "local_tone" => &self.local_tone_bits,
            "local_structure" => &self.local_structure_bits,
            "skin_structure" => &self.skin_structure_bits,
            "auto_mask" => &self.auto_mask,
            "composition_bypass" => &self.composition_bypass,
            "transfer_strength" => &self.transfer_strength_bits,
            "colour_strength" => &self.colour_strength_bits,
            "max_ratio" => &self.max_ratio_bits,
            "working_scale" => &self.working_scale_bits,
            "reversible_mode" => &self.reversible_mode,
            "hdr_mode" => &self.hdr_mode,
            "transfer" => &self.transfer,
            "compare_mode" => &self.compare_mode,
            "compare_split" => &self.compare_split_bits,
            "compare_zoom" => &self.compare_zoom_bits,
            "compare_swap" => &self.compare_swap,
            "colour_mode" => &self.colour_mode,
            "hold_frame" => &self.hold_frame,
            "ghost_guard" => &self.ghost_guard_bits,
            "colour_trust" => &self.colour_trust_bits,
            "ratio_smooth" => &self.ratio_smooth_bits,
            "model_interval" => &self.model_interval,
            "apply_model" => &self.apply_model,
            "debug_view" => &self.debug_view,
            "debug_scale" => &self.debug_scale_bits,
            _ => return false,
        };
        field.store(bits, Ordering::Relaxed);
        true
    }

    pub fn neural_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) != 0
    }

    /// The Model tab's settings, as the network uses them.
    pub fn global_tuning(&self) -> Tuning {
        Tuning {
            intensity: f32::from_bits(self.intensity_bits.load(Ordering::Relaxed)),
            local_tone: f32::from_bits(self.local_tone_bits.load(Ordering::Relaxed)),
            local_structure: f32::from_bits(self.local_structure_bits.load(Ordering::Relaxed)),
            skin_structure: f32::from_bits(self.skin_structure_bits.load(Ordering::Relaxed)),
            style: self.style.load(Ordering::Relaxed),
            auto_mask: self.auto_mask.load(Ordering::Relaxed),
        }
    }

    pub fn set_layer_reason(&self, s: &str) {
        store_seq_guarded(&self.layer_reason_seq, &self.layer_reason, s);
    }
    pub fn layer_reason(&self) -> String {
        load_seq_guarded(&self.layer_reason_seq, &self.layer_reason)
    }
    pub fn set_game_name(&self, s: &str) {
        store_seq_guarded(&self.game_name_seq, &self.game_name, s);
    }
    pub fn game_name(&self) -> String {
        load_seq_guarded(&self.game_name_seq, &self.game_name)
    }
}

/// Reads a lo/hi `AtomicU32` pair as one `u64` frame counter.
///
/// The two halves are separate atomics, so a plain pair of loads can straddle a
/// concurrent [`store64`]. `store64` writes the high half first, so retrying until the
/// high half is unchanged across the low-half read removes every torn read except the
/// instant a counter carries past 2^32 (about 2.3 years of frames at 60 fps), which
/// this layout cannot distinguish without a third word.
pub fn load64(lo: &AtomicU32, hi: &AtomicU32) -> u64 {
    loop {
        let high = hi.load(Ordering::Acquire);
        let low = lo.load(Ordering::Acquire);
        if hi.load(Ordering::Acquire) == high {
            return (u64::from(high) << 32) | u64::from(low);
        }
        std::hint::spin_loop();
    }
}

pub fn store64(lo: &AtomicU32, hi: &AtomicU32, v: u64) {
    hi.store((v >> 32) as u32, Ordering::Release);
    lo.store(v as u32, Ordering::Release);
}

/// Writes a fixed-size text field guarded by a sequence lock. The sequence is made odd
/// *before* the field is touched and even again after, so a reader can tell "a write is
/// in progress" (odd) from "a write finished between my two looks" (changed). Bumping
/// only after the write, as this used to, let a reader see `before == after` while the
/// writer was mid-copy and return a torn string.
///
/// The payload is atomic words (relaxed loads and stores inside the fences), not plain
/// bytes: a reader racing the writer is then only ever a stale read the sequence check
/// throws away, never a data race. The bytes sit little-endian four to a word, so the
/// memory image is the same byte string it always was.
///
/// There is exactly one writer per field. If a previous writer died mid-write and left
/// the sequence odd, it is simply carried through to the next even value.
fn store_seq_guarded<const W: usize>(seq: &AtomicU32, words: &[AtomicU32; W], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(W * 4 - 1);
    let odd = seq.load(Ordering::Relaxed) | 1;
    seq.store(odd, Ordering::Relaxed);
    // Keeps the odd store ordered before the data writes below, as far as a reader
    // that pairs it with an acquire fence is concerned.
    std::sync::atomic::fence(Ordering::Release);
    for (i, word) in words.iter().enumerate() {
        let mut chunk = [0u8; 4];
        for (j, b) in chunk.iter_mut().enumerate() {
            let k = i * 4 + j;
            if k < n {
                *b = bytes[k];
            }
        }
        word.store(u32::from_le_bytes(chunk), Ordering::Relaxed);
    }
    seq.store(odd.wrapping_add(1), Ordering::Release);
}

fn load_seq_guarded<const W: usize>(seq: &AtomicU32, words: &[AtomicU32; W]) -> String {
    for _ in 0..16 {
        let before = seq.load(Ordering::Acquire);
        if before & 1 != 0 {
            // A write is in progress.
            std::hint::spin_loop();
            continue;
        }
        let mut snapshot = Vec::with_capacity(W * 4);
        for word in words {
            snapshot.extend_from_slice(&word.load(Ordering::Relaxed).to_le_bytes());
        }
        std::sync::atomic::fence(Ordering::Acquire);
        if seq.load(Ordering::Relaxed) == before {
            let end = snapshot.iter().position(|&b| b == 0).unwrap_or(snapshot.len());
            return String::from_utf8_lossy(&snapshot[..end]).into_owned();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seqlock_is_odd_only_while_writing_and_readers_never_see_a_torn_string() {
        let h = std::sync::Arc::new(ShmHeader::default());
        h.set_layer_reason("first");
        assert_eq!(h.layer_reason_seq.load(Ordering::Relaxed) & 1, 0, "idle sequence must be even");
        assert_eq!(h.layer_reason(), "first");
        // A reader arriving while a (simulated, stalled) write holds the sequence odd
        // must refuse to return the half-written field rather than trust it.
        h.layer_reason_seq.store(1, Ordering::Release);
        assert_eq!(h.layer_reason(), "");
        h.layer_reason_seq.store(2, Ordering::Release);
        assert_eq!(h.layer_reason(), "first");

        let writer = {
            let h = h.clone();
            std::thread::spawn(move || {
                for i in 0..20_000 {
                    h.set_layer_reason(if i % 2 == 0 { "aaaaaaaaaaaaaaaaaaaa" } else { "bbbbbbbbbbbbbbbbbbbb" });
                }
            })
        };
        for _ in 0..20_000 {
            let s = h.layer_reason();
            assert!(s.is_empty() || s.chars().all(|c| c == 'a') || s.chars().all(|c| c == 'b') || s == "first", "torn read: {s:?}");
        }
        writer.join().unwrap();
        assert_eq!(h.layer_reason_seq.load(Ordering::Relaxed) & 1, 0);
    }

    #[test]
    fn load64_round_trips_across_the_halves() {
        let (lo, hi) = (AtomicU32::new(0), AtomicU32::new(0));
        for v in [0u64, 1, 0xFFFF_FFFF, 0x1_0000_0000, 0xDEAD_BEEF_CAFE_F00D] {
            store64(&lo, &hi, v);
            assert_eq!(load64(&lo, &hi), v);
        }
    }

    #[test]
    fn default_is_all_zero() {
        let h = ShmHeader::default();
        assert_eq!(h.magic.load(Ordering::Relaxed), 0);
        assert_eq!(h.enabled.load(Ordering::Relaxed), 0);
        assert_eq!(h.layer_reason(), "");
    }

    #[test]
    fn slot_accessors_pick_the_matching_field() {
        let h = ShmHeader::default();
        h.seq_req_slot(Slot::Primary).store(11, Ordering::Relaxed);
        h.seq_req_slot(Slot::Secondary).store(22, Ordering::Relaxed);
        assert_eq!(h.seq_req.load(Ordering::Relaxed), 11);
        assert_eq!(h.seq_req_b.load(Ordering::Relaxed), 22);
        assert_eq!(h.seq_req_slot(Slot::Primary).load(Ordering::Relaxed), 11);
        assert_eq!(h.seq_req_slot(Slot::Secondary).load(Ordering::Relaxed), 22);

        h.width_slot(Slot::Primary).store(2560, Ordering::Relaxed);
        h.height_slot(Slot::Primary).store(1440, Ordering::Relaxed);
        h.width_slot(Slot::Secondary).store(1920, Ordering::Relaxed);
        h.height_slot(Slot::Secondary).store(1080, Ordering::Relaxed);
        assert_eq!(h.width.load(Ordering::Relaxed), 2560);
        assert_eq!(h.height.load(Ordering::Relaxed), 1440);
        assert_eq!(h.width_b.load(Ordering::Relaxed), 1920);
        assert_eq!(h.height_b.load(Ordering::Relaxed), 1080);

        h.proxy_format_slot(Slot::Secondary).store(crate::enums::proxy_format::BGRA8, Ordering::Relaxed);
        assert_eq!(h.proxy_format_b.load(Ordering::Relaxed), crate::enums::proxy_format::BGRA8);
        assert_eq!(h.proxy_format.load(Ordering::Relaxed), 0, "slot 0 must be untouched by a slot-1 write");
    }

    #[test]
    fn init_defaults_sets_real_values() {
        let h = ShmHeader::default();
        h.init_defaults();
        assert!(h.is_valid());
        assert!(h.neural_enabled());
        assert_eq!(f32::from_bits(h.intensity_bits.load(Ordering::Relaxed)), 1.0);
        // The composition is on by default; bypass is an explicit debug choice.
        assert_eq!(h.composition_bypass.load(Ordering::Relaxed), 0);
        // Slot 1 (v3) starts idle, same shape as slot 0.
        assert_eq!(h.seq_req_b.load(Ordering::Relaxed), 0);
        assert_eq!(h.seq_resp_b.load(Ordering::Relaxed), 0);
        assert_eq!(h.proxy_format_b.load(Ordering::Relaxed), crate::enums::proxy_format::RGBA8);
    }

    #[test]
    fn reset_persisted_settings_changes_settings_but_preserves_the_live_session() {
        let h = ShmHeader::default();
        h.init_defaults();

        // A user-tunable setting, changed away from its default.
        h.intensity_bits.store(0.4f32.to_bits(), Ordering::Relaxed);
        h.style.store(2, Ordering::Relaxed);

        // Live session/transport state a running layer would have built up --
        // exactly what upstream's PR #16 bug wiped out from under a running process.
        h.seq_req.store(41, Ordering::Relaxed);
        h.seq_resp.store(40, Ordering::Relaxed);
        h.width.store(2560, Ordering::Relaxed);
        h.height.store(1440, Ordering::Relaxed);
        // Slot 1 (v3): a settings reset must not wipe a live second in-flight request
        // out from under a running process either -- the same PR #16 bug class this
        // test already guards slot 0 against.
        h.seq_req_b.store(9, Ordering::Relaxed);
        h.seq_resp_b.store(8, Ordering::Relaxed);
        h.width_b.store(1920, Ordering::Relaxed);
        h.height_b.store(1080, Ordering::Relaxed);
        h.proxy_format_b.store(crate::enums::proxy_format::BGRA8, Ordering::Relaxed);
        h.server_state.store(crate::enums::server_state::RUNNING, Ordering::Relaxed);
        h.model_up.store(1, Ordering::Relaxed);
        store64(&h.server_frames_lo, &h.server_frames_hi, 12_345);
        h.layer_attached.store(1, Ordering::Relaxed);
        store64(&h.layer_frames_lo, &h.layer_frames_hi, 6_789);
        h.hdr_active.store(1, Ordering::Relaxed);
        h.set_layer_reason("native network running");
        h.set_game_name("GTA5_Enhanced.exe");

        h.reset_persisted_settings();

        // The settings actually changed.
        assert_eq!(f32::from_bits(h.intensity_bits.load(Ordering::Relaxed)), 1.0, "settings must reset");
        assert_eq!(h.style.load(Ordering::Relaxed), 0, "settings must reset");

        // Nothing about the live session moved.
        assert_eq!(h.seq_req.load(Ordering::Relaxed), 41);
        assert_eq!(h.seq_resp.load(Ordering::Relaxed), 40);
        assert_eq!(h.width.load(Ordering::Relaxed), 2560);
        assert_eq!(h.height.load(Ordering::Relaxed), 1440);
        assert_eq!(h.seq_req_b.load(Ordering::Relaxed), 9);
        assert_eq!(h.seq_resp_b.load(Ordering::Relaxed), 8);
        assert_eq!(h.width_b.load(Ordering::Relaxed), 1920);
        assert_eq!(h.height_b.load(Ordering::Relaxed), 1080);
        assert_eq!(h.proxy_format_b.load(Ordering::Relaxed), crate::enums::proxy_format::BGRA8);
        assert_eq!(h.server_state.load(Ordering::Relaxed), crate::enums::server_state::RUNNING);
        assert_eq!(h.model_up.load(Ordering::Relaxed), 1);
        assert_eq!(load64(&h.server_frames_lo, &h.server_frames_hi), 12_345);
        assert_eq!(h.layer_attached.load(Ordering::Relaxed), 1);
        assert_eq!(load64(&h.layer_frames_lo, &h.layer_frames_hi), 6_789);
        assert_eq!(h.hdr_active.load(Ordering::Relaxed), 1);
        assert_eq!(h.layer_reason(), "native network running");
        assert_eq!(h.game_name(), "GTA5_Enhanced.exe");
    }

    /// A reset racing a live layer must never revert what it writes: the old
    /// snapshot/init_defaults/restore sequence briefly zeroed every live field.
    #[test]
    fn reset_never_touches_live_fields_even_transiently() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let h = Arc::new(ShmHeader::default());
        h.init_defaults();
        h.seq_req.store(41, Ordering::Relaxed);
        h.width.store(2560, Ordering::Relaxed);
        h.server_state.store(crate::enums::server_state::RUNNING, Ordering::Relaxed);
        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let (h, done) = (h.clone(), done.clone());
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !done.load(Ordering::Relaxed) || reads == 0 {
                    for (name, v) in [("seq_req", &h.seq_req), ("width", &h.width), ("server_state", &h.server_state)] {
                        assert_ne!(v.load(Ordering::Relaxed), 0, "{name} observed as zero during a reset");
                    }
                    reads += 1;
                }
            })
        };
        for _ in 0..10_000 {
            h.reset_persisted_settings();
        }
        done.store(true, Ordering::Relaxed);
        reader.join().expect("a live field was reset under a running reader");
    }

    #[test]
    fn text_fields_keep_their_byte_image() {
        let h = ShmHeader::default();
        h.set_game_name("GTA5.exe");
        // SAFETY: `h` is a plain, fully initialized value; reading its bytes is sound.
        let bytes = unsafe { std::slice::from_raw_parts((&h as *const ShmHeader).cast::<u8>(), std::mem::size_of::<ShmHeader>()) };
        let at = std::mem::offset_of!(ShmHeader, game_name);
        assert_eq!(&bytes[at..at + 9], b"GTA5.exe\0");
        assert_eq!(h.game_name(), "GTA5.exe");
    }

    #[test]
    fn seq_guarded_string_round_trips() {
        let h = ShmHeader::default();
        h.init_defaults();
        h.set_layer_reason("native network running");
        assert_eq!(h.layer_reason(), "native network running");
        // A field longer than its buffer is truncated, not rejected.
        let long = "x".repeat(REASON_BYTES + 50);
        h.set_layer_reason(&long);
        assert_eq!(h.layer_reason().len(), REASON_BYTES - 1);
    }

}
